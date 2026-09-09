#!/usr/bin/env python3
"""Run the VisDrone FPN detector on a Jetson Orin.

Self-contained on purpose: numpy + Pillow + one runtime, nothing from the
training tree. Three backends behind one interface:

    trt   TensorRT engine built on the device by trtexec — the fast one
    ort   onnxruntime on the same ONNX — no engine, runs anywhere (the training
          box dry-runs the whole pipeline this way; an Orin without a working
          TensorRT python binding can too, on CPU)
    iree  the portable route, ~0.5 fps on an Orin, kept for completeness

    python3 orin_detect.py --backend trt --plan build/detector_aff30e28.plan \
        --image testdata/frame.png --out out.png --bench 50
    python3 orin_detect.py --backend ort --onnx build/detector_aff30e28.onnx \
        --image testdata/frame.png --bench 10
    python3 orin_detect.py --gate-decode          # no GPU, no weights

`--bench` reports the three stages separately — host preprocess, forward
(H2D + engine + D2H, synchronized), decode+NMS on the CPU — because on an Orin
Nano the network is the SMALLEST of the three (6.3 ms against 11.9 + 10.0) and
a single number hides where the frame time goes.

## The preprocessing contract, which is the easiest thing to get silently wrong

Training normalizes in the C loader, not in the graph (`ffi/f32_helpers.c`):
resize to 448x448 (SQUASH, not letterbox — measured 23% better on this data),
uint8 -> /255 -> subtract ImageNet mean -> divide by ImageNet std -> CHW.
Deviating produces a detector that still runs and quietly gets worse.

Which of that the host still does depends on how the graph was exported
(`export_onnx.py --fold-preprocess`), and the graph's INPUT TENSOR says which:
`image` [1,3,448,448] f32 wants the normalized tensor; `image_01` the same
shape in [0,1] with the normalize inside the graph; `image_u8` [1,448,448,3]
uint8 wants the resized bytes as they come off the camera. The detectors read
the name and dtype off the engine/model, so a mismatch is impossible unless
`--input-mode` overrides it.

## IREE calling convention (backend iree only)

The graph takes 190 tensors: the image first, then 189 weight tensors, in the
order they appear in the MLIR signature. Their total is exactly
21,548,743 params + 17,024 BN stats, which is `params.bin ++ bn_stats.bin`
concatenated and split by that signature. The signature is PARSED FROM THE MLIR
rather than hardcoded, so a retrained or re-shaped model cannot silently
mismatch — pass --mlir to re-derive it.
"""
import argparse
import re
import time
from pathlib import Path

import numpy as np

IMG_PX = 448
NTOT = 185220
FPN_GRIDS = (56, 28, 14)
PER_ANCHOR = 15          # tx,ty,tw,th,obj + 10 class logits
N_CLASSES = 10
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
ISTD = (1.0 / np.array([0.229, 0.224, 0.225], dtype=np.float32)).reshape(3, 1, 1)

CLASS_NAMES = ["pedestrian", "people", "bicycle", "car", "van",
               "truck", "tricycle", "awning-tricycle", "bus", "motor"]

# Per-scale k-means priors, from demos/MainYolov1VisdroneFpn.lean. These MUST
# match the trained model: the head regresses a residual off these priors, so a
# different set silently produces wrong boxes rather than an error.
ANCHORS = {
    56: [(0.006935, 0.014941), (0.015750, 0.028005), (0.033728, 0.035028)],
    28: [(0.023961, 0.070528), (0.055662, 0.068706), (0.093187, 0.094324)],
    14: [(0.060280, 0.168604), (0.107559, 0.204684), (0.181239, 0.149031)],
}


# ---------------------------------------------------------------- signature

def parse_signature(mlir_path):
    """[(name, [dims...]), ...] for @forward_eval, in argument order."""
    s = Path(mlir_path).read_text()
    i = s.index("func.func @forward_eval(")
    j = s.index(") -> ", i)
    sig = s[i + len("func.func @forward_eval("):j]
    args = re.findall(r"%([A-Za-z0-9_]+):\s*tensor<([0-9x]*)f32>", sig)
    return [(nm, [int(d) for d in sh.split("x") if d]) for nm, sh in args]


def load_weights(sig, params_path, bn_path):
    """Split params.bin ++ bn_stats.bin into the 189 non-image tensors."""
    flat = np.concatenate([
        np.fromfile(params_path, dtype=np.float32),
        np.fromfile(bn_path, dtype=np.float32),
    ])
    want = sum(int(np.prod(sh)) for _nm, sh in sig[1:])
    if flat.size != want:
        raise SystemExit(
            f"weight size mismatch: files hold {flat.size} floats, graph wants "
            f"{want}. Wrong checkpoint for this MLIR, or FPN_TAG picked a "
            f"different arm when the graph was emitted.")
    out, off = [], 0
    for _nm, sh in sig[1:]:
        n = int(np.prod(sh))
        out.append(np.ascontiguousarray(flat[off:off + n].reshape(sh)))
        off += n
    return out


# ------------------------------------------------------------ preprocessing

INPUT_MODES = ("none", "f32", "u8")


def input_mode_of(name, is_uint8):
    """Preprocessing mode implied by the graph's input tensor. Mirrors
    `export_onnx.INPUT_NAMES`: uint8 -> u8, `*_01` -> f32, else none."""
    if is_uint8:
        return "u8"
    return "f32" if name.endswith("_01") else "none"


def preprocess(img_rgb_hwc, mode="none"):
    """HWC uint8 (any size) -> the graph's input tensor for `mode`.

      none  [1,3,448,448] float32, normalized here (the C loader's contract)
      f32   [1,3,448,448] float32 in [0,1]; the graph normalizes
      u8    [1,448,448,3] uint8, the resized bytes untouched; the graph does the
            permute, the /255 and the normalize, and the H2D copy is 4x smaller

    Measured on an Orin Nano, mode none costs 11.9 ms of which the resize is
    0.5 ms — the rest is numpy elementwise work on a CPU that is far worse at it
    than the GPU sitting idle behind it. That is what u8 removes.
    """
    from PIL import Image
    pil = Image.fromarray(img_rgb_hwc).convert("RGB")
    if pil.size != (IMG_PX, IMG_PX):
        pil = pil.resize((IMG_PX, IMG_PX), Image.BILINEAR)   # SQUASH, not letterbox
    if mode == "u8":
        return np.ascontiguousarray(np.asarray(pil, dtype=np.uint8))[None]
    chw = np.asarray(pil, dtype=np.float32).transpose(2, 0, 1) / 255.0
    if mode == "none":
        chw = (chw - MEAN) * ISTD
    return np.ascontiguousarray(chw[None], dtype=np.float32)


# ------------------------------------------------------------------ decode

def _iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def decode_reference(flat, conf_thresh=0.05, nms_iou=0.5, topk=300):
    """The original decode, kept ONLY as the oracle for `decode`.

    Straight-line and obviously correct, and 45 ms on an Orin — 10x the 4.3 ms
    network. `decode` below is the fast one; `_gate_decode` asserts they agree.
    """
    dets, off = [], 0
    for g in FPN_GRIDS:
        anchors = ANCHORS[g]
        A = len(anchors)
        n = A * PER_ANCHOR * g * g
        pred = flat[off:off + n].reshape(A, PER_ANCHOR, g, g).astype(np.float64)
        off += n
        anch = np.asarray(anchors, dtype=np.float64)
        obj = 1.0 / (1.0 + np.exp(-np.clip(pred[:, 4], -60, 60)))
        keep = obj >= conf_thresh
        if not keep.any():
            continue
        cls = pred[:, 5:5 + N_CLASSES]
        cid = cls.argmax(axis=1)
        e = np.exp(cls - cls.max(axis=1, keepdims=True))
        clsp = e.max(axis=1) / e.sum(axis=1)
        conf = obj * clsp
        jj = np.arange(g).reshape(1, 1, g)
        ii = np.arange(g).reshape(1, g, 1)
        sx = 1.0 / (1.0 + np.exp(-np.clip(pred[:, 0], -60, 60)))
        sy = 1.0 / (1.0 + np.exp(-np.clip(pred[:, 1], -60, 60)))
        cx, cy = (jj + sx) / g, (ii + sy) / g
        w = anch[:, 0].reshape(A, 1, 1) * np.exp(np.minimum(pred[:, 2], 8.0))
        h = anch[:, 1].reshape(A, 1, 1) * np.exp(np.minimum(pred[:, 3], 8.0))
        boxes = np.stack([(cx - w / 2)[keep], (cy - h / 2)[keep],
                          (cx + w / 2)[keep], (cy + h / 2)[keep]], axis=1)
        dets += list(zip(cid[keep].tolist(), conf[keep].tolist(), boxes.tolist()))
    if len(dets) > topk:
        dets = sorted(dets, key=lambda d: -d[1])[:topk]
    kept = []
    for c in set(d[0] for d in dets):
        cd = sorted((d for d in dets if d[0] == c), key=lambda d: -d[1])
        while cd:
            top = cd.pop(0)
            kept.append(top)
            cd = [d for d in cd if _iou(top[2], d[2]) < nms_iou]
    return kept


def _batched_nms(boxes, scores, cids, nms_iou):
    """Greedy NMS over every class at once, in numpy.

    Two changes against the reference, neither of which alters the result:

    * **Class offset.** Shifting each class's boxes by a large per-class constant
      makes boxes of different classes non-overlapping by construction, so one
      global pass reproduces the per-class passes exactly. (torchvision's
      `batched_nms` does the same thing.) The reference looped over classes and
      re-sorted inside each.
    * **Precomputed IoU matrix.** At topk=300 that is a 300x300 float array —
      360 KB, built in one vectorized shot. The greedy loop then costs one
      boolean OR per surviving box instead of a Python `_iou` call per PAIR,
      which is where the 34.8 ms went (300 boxes is up to ~45,000 calls).
    """
    if len(scores) == 0:
        return []
    order = np.argsort(-scores, kind="stable")
    b, sc, cd = boxes[order], scores[order], cids[order]
    # Offset must exceed the coordinate range; boxes are normalized, but the
    # exp() on w/h is only capped at 8, so they can run well outside [0,1].
    span = float(np.abs(b).max()) + 1.0
    ob = b + (cd.astype(np.float64) * (2.0 * span)).reshape(-1, 1)

    x0, y0, x1, y1 = ob[:, 0], ob[:, 1], ob[:, 2], ob[:, 3]
    area = np.maximum(x1 - x0, 0.0) * np.maximum(y1 - y0, 0.0)
    ix0 = np.maximum(x0[:, None], x0[None, :])
    iy0 = np.maximum(y0[:, None], y0[None, :])
    ix1 = np.minimum(x1[:, None], x1[None, :])
    iy1 = np.minimum(y1[:, None], y1[None, :])
    inter = np.maximum(ix1 - ix0, 0.0) * np.maximum(iy1 - iy0, 0.0)
    union = area[:, None] + area[None, :] - inter
    iou = np.where(union > 0.0, inter / np.where(union > 0.0, union, 1.0), 0.0)

    # Threshold ONCE into a boolean matrix rather than per row inside the loop:
    # the loop body then costs a single `|=` instead of a compare plus an OR, and
    # this frame runs the loop 238 times.
    sup = iou >= nms_iou
    n = len(sc)
    dead = np.zeros(n, dtype=bool)
    out = []
    for i in range(n):
        if dead[i]:
            continue
        out.append(int(order[i]))
        dead |= sup[i]                  # suppresses i itself; the `dead[i]`
        dead[i] = False                 # check above has already passed
    return out


def decode(flat, conf_thresh=0.05, nms_iou=0.5, topk=300):
    """[NTOT] -> [(cid, score, (x0,y0,x1,y1))], normalized coords.

    Mirrors scripts/yolo_map_visdrone.py's decode_anchor_raw + decode_fpn: per
    scale, sigmoid objectness times max class softmax, centre confined to its own
    cell, size = anchor * exp(t) with t capped at 8 to match the training-time cap.

    Fast path, measured against `decode_reference` by `_gate_decode`. On an Orin
    the reference costs 49 ms against a 4.3 ms network — 11x — so the decode, not
    the model, is what caps end-to-end frame rate. Two costs, both removed here:

    1. **Candidate extraction, 10.3 ms.** The reference promoted all 185,220
       logits to float64 and ran sigmoid + a 10-way softmax over every one of
       them, then threw away 99.8%. Objectness is monotonic in its logit, so
       thresholding on the RAW logit first is equivalent, and everything
       expensive then runs on the few hundred survivors. The threshold is taken
       a hair loose and the exact `obj >= conf_thresh` test is re-applied
       afterwards, so the surviving set is identical rather than merely close.
    2. **NMS, 34.8 ms.** See `_batched_nms`.
    """
    # sigmoid(x) >= t  <=>  x >= log(t/(1-t)); the -1e-3 keeps the boundary
    # inclusive under float error, and the exact test below decides it.
    lg_thr = (np.log(conf_thresh / (1.0 - conf_thresh)) - 1e-3
              if 0.0 < conf_thresh < 1.0 else -np.inf)
    cids, confs, boxes, off = [], [], [], 0
    for g in FPN_GRIDS:
        anchors = ANCHORS[g]
        A = len(anchors)
        n = A * PER_ANCHOR * g * g
        pred = flat[off:off + n].reshape(A, PER_ANCHOR, g, g)
        off += n
        cand = pred[:, 4] >= lg_thr                     # [A,g,g], on float32
        if not cand.any():
            continue
        ai, ii, jj = np.nonzero(cand)
        sel = pred[:, :, ii, jj][ai, :, np.arange(len(ai))].astype(np.float64)
        obj = 1.0 / (1.0 + np.exp(-np.clip(sel[:, 4], -60, 60)))
        exact = obj >= conf_thresh                      # the reference's own test
        if not exact.any():
            continue
        sel, obj = sel[exact], obj[exact]
        ai, ii, jj = ai[exact], ii[exact], jj[exact]
        cls = sel[:, 5:5 + N_CLASSES]
        e = np.exp(cls - cls.max(axis=1, keepdims=True))
        anch = np.asarray(anchors, dtype=np.float64)
        sx = 1.0 / (1.0 + np.exp(-np.clip(sel[:, 0], -60, 60)))
        sy = 1.0 / (1.0 + np.exp(-np.clip(sel[:, 1], -60, 60)))
        cx, cy = (jj + sx) / g, (ii + sy) / g
        w = anch[ai, 0] * np.exp(np.minimum(sel[:, 2], 8.0))
        h = anch[ai, 1] * np.exp(np.minimum(sel[:, 3], 8.0))
        cids.append(cls.argmax(axis=1))
        confs.append(obj * (e.max(axis=1) / e.sum(axis=1)))
        boxes.append(np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1))
    if not cids:
        return []
    cids = np.concatenate(cids)
    confs = np.concatenate(confs)
    boxes = np.concatenate(boxes)
    if len(confs) > topk:
        top = np.argsort(-confs, kind="stable")[:topk]
        cids, confs, boxes = cids[top], confs[top], boxes[top]
    idx = _batched_nms(boxes, confs, cids, nms_iou)
    return [(int(cids[i]), float(confs[i]), boxes[i].tolist()) for i in idx]


def _gate_decode(flat, **kw):
    """Assert the fast decode returns exactly the reference's detection set.

    Compares as SETS: the reference emits grouped by class, the fast path in
    global score order, and nothing downstream depends on that order. Run it on
    the device once — `orin_detect.py --gate-decode` — before trusting a number
    that came out of the fast path.
    """
    def key(ds):
        return sorted((c, round(s, 9), tuple(round(v, 9) for v in b)) for c, s, b in ds)
    a, b = key(decode_reference(flat, **kw)), key(decode(flat, **kw))
    if a != b:
        only_a = [d for d in a if d not in b][:3]
        only_b = [d for d in b if d not in a][:3]
        raise SystemExit(f"⛔ fast decode disagrees with the reference\n"
                         f"   reference {len(a)} dets, fast {len(b)}\n"
                         f"   only in reference: {only_a}\n"
                         f"   only in fast:      {only_b}")
    return len(a)


# ------------------------------------------------------------------ runtime

def _expected_input_shape(mode):
    return (1, IMG_PX, IMG_PX, 3) if mode == "u8" else (1, 3, IMG_PX, IMG_PX)


def _resolve_mode(name, is_uint8, override):
    mode = input_mode_of(name, is_uint8)
    if override not in (None, "auto") and override != mode:
        if is_uint8 != (override == "u8"):
            raise SystemExit(f"--input-mode {override} contradicts the input "
                             f"dtype ({'uint8' if is_uint8 else 'float32'})")
        print(f"⚠ --input-mode {override} overrides the mode the input name "
              f"'{name}' implies ({mode}); only right for an ONNX exported "
              f"before the name carried it")
        mode = override
    return mode


class TrtDetector:
    """TensorRT backend — the one that actually goes fast on an Orin.

    IREE compiles this graph for sm_87 and runs it at ~0.5 fps, because its CUDA
    backend generates its own convolution kernels; TensorRT dispatches to cuDNN
    and tensor cores and does fp16 natively. Weights are baked into the engine at
    build time, so there is nothing to feed but the image.

    Everything about the I/O comes from the ENGINE, not from assumptions: the
    input's name, shape and dtype pick the preprocessing mode (a u8-folded
    engine takes [1,448,448,3] uint8, the plain one [1,3,448,448] float32), the
    output's size sizes the buffer. Host buffers are page-locked, because an
    "async" copy from pageable memory silently degrades to a staged synchronous
    one — that fix was made on the device twice and lost twice; it is in git now.

    ⚠ Written on the build box, where there is no TensorRT to run it. The
    `--backend ort` path exercises the same preprocess / forward / decode /
    bench code on the same ONNX, so what is untested here is exactly the
    TensorRT + pycuda calls. `execute_async_v3` is the TensorRT 10 API (JetPack
    6.x); a TensorRT 8 runtime falls back to `execute_async_v2` with a bindings
    list.

    Build the engine on the device:
        trtexec --onnx=detector_aff30e28.onnx --saveEngine=detector_aff30e28.plan --fp16
    """

    def __init__(self, plan, input_mode="auto"):
        import tensorrt as trt
        import pycuda.autoinit  # noqa: F401  (creates the CUDA context)
        import pycuda.driver as cuda
        self.cuda = cuda
        logger = trt.Logger(trt.Logger.WARNING)
        with open(plan, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise SystemExit(f"TensorRT refused to deserialize {plan} — built by "
                             f"a different TensorRT version, or not an engine")
        self.ctx = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        e = self.engine
        if hasattr(e, "num_io_tensors"):                     # TensorRT >= 8.5
            names = [e.get_tensor_name(i) for i in range(e.num_io_tensors)]
            is_in = {n: e.get_tensor_mode(n) == trt.TensorIOMode.INPUT for n in names}
            shape = {n: tuple(e.get_tensor_shape(n)) for n in names}
            dtype = {n: trt.nptype(e.get_tensor_dtype(n)) for n in names}
        else:                                                # TensorRT 8.0-8.4
            names = [e.get_binding_name(i) for i in range(e.num_bindings)]
            is_in = {n: e.binding_is_input(i) for i, n in enumerate(names)}
            shape = {n: tuple(e.get_binding_shape(i)) for i, n in enumerate(names)}
            dtype = {n: trt.nptype(e.get_binding_dtype(i)) for i, n in enumerate(names)}
        ins = [n for n in names if is_in[n]]
        outs = [n for n in names if not is_in[n]]
        if len(ins) != 1 or len(outs) != 1:
            raise SystemExit(f"expected 1 input + 1 output, engine has {ins} / {outs}")
        self.in_name, self.out_name = ins[0], outs[0]
        in_shape, in_dtype = shape[self.in_name], dtype[self.in_name]
        out_shape, out_dtype = shape[self.out_name], dtype[self.out_name]
        self.mode = _resolve_mode(self.in_name, in_dtype == np.uint8, input_mode)
        if in_shape != _expected_input_shape(self.mode):
            raise SystemExit(f"engine input {self.in_name} is {in_shape} "
                             f"{in_dtype.__name__}, want "
                             f"{_expected_input_shape(self.mode)} for mode {self.mode}")
        n_out = int(np.prod(out_shape))
        if n_out < NTOT:
            raise SystemExit(f"engine output {self.out_name} has {n_out} floats, "
                             f"want at least {NTOT}")

        # Pinned host buffers on both sides of the copy.
        self.h_in = cuda.pagelocked_empty(in_shape, in_dtype)
        self.h_out = cuda.pagelocked_empty(n_out, out_dtype)
        self.d_in = cuda.mem_alloc(self.h_in.nbytes)
        self.d_out = cuda.mem_alloc(self.h_out.nbytes)
        self._v3 = hasattr(self.ctx, "set_tensor_address") and \
            hasattr(self.ctx, "execute_async_v3")
        if self._v3:
            self.ctx.set_tensor_address(self.in_name, int(self.d_in))
            self.ctx.set_tensor_address(self.out_name, int(self.d_out))
        else:
            # TensorRT 8: positional bindings in engine binding order.
            order = [self.engine.get_binding_index(n) for n in (self.in_name, self.out_name)]
            self._bindings = [0, 0]
            self._bindings[order[0]] = int(self.d_in)
            self._bindings[order[1]] = int(self.d_out)
        print(f"  engine input {self.in_name} {in_shape} {in_dtype.__name__} -> "
              f"preprocessing mode '{self.mode}'; output {out_shape}; "
              f"{'execute_async_v3' if self._v3 else 'execute_async_v2'}")

    def forward(self, x):
        """Graph input (from `preprocess`) -> [NTOT] float32 logits. Blocks."""
        np.copyto(self.h_in, x.reshape(self.h_in.shape))
        self.cuda.memcpy_htod_async(self.d_in, self.h_in, self.stream)
        if self._v3:
            self.ctx.execute_async_v3(self.stream.handle)
        else:
            self.ctx.execute_async_v2(bindings=self._bindings,
                                      stream_handle=self.stream.handle)
        self.cuda.memcpy_dtoh_async(self.h_out, self.d_out, self.stream)
        self.stream.synchronize()
        return self.h_out[:NTOT].astype(np.float32, copy=True)

    def __call__(self, img_rgb_hwc):
        return self.forward(preprocess(img_rgb_hwc, self.mode))


class OrtDetector:
    """onnxruntime backend: the identical pipeline with no engine build.

    This is how the training box dry-runs the whole runner against the very
    ONNX it ships (`--backend ort --onnx build/detector_aff30e28.onnx`), and how
    an Orin without a working TensorRT python binding still gets detections —
    slowly, on the CPU provider, unless onnxruntime-gpu is installed.
    """

    def __init__(self, onnx_path, input_mode="auto", providers=None):
        import onnxruntime as ort
        avail = ort.get_available_providers()
        providers = providers or [p for p in ("CUDAExecutionProvider",
                                              "CPUExecutionProvider") if p in avail]
        self.sess = ort.InferenceSession(onnx_path, providers=providers)
        ins, outs = self.sess.get_inputs(), self.sess.get_outputs()
        if len(ins) != 1 or len(outs) != 1:
            raise SystemExit(f"expected 1 input + 1 output, model has "
                             f"{[i.name for i in ins]} / {[o.name for o in outs]}")
        inp = ins[0]
        self.in_name = inp.name
        is_u8 = "uint8" in inp.type
        self.mode = _resolve_mode(inp.name, is_u8, input_mode)
        if tuple(inp.shape) != _expected_input_shape(self.mode):
            raise SystemExit(f"model input {inp.name} is {inp.shape} {inp.type}, "
                             f"want {_expected_input_shape(self.mode)} for mode {self.mode}")
        print(f"  model input {inp.name} {inp.shape} {inp.type} -> preprocessing "
              f"mode '{self.mode}'; providers {self.sess.get_providers()}")

    def forward(self, x):
        return self.sess.run(None, {self.in_name: x})[0].reshape(-1)[:NTOT].astype(np.float32)

    def __call__(self, img_rgb_hwc):
        return self.forward(preprocess(img_rgb_hwc, self.mode))


class Detector:
    """IREE backend. Kept because it is portable and needs no engine build, but
    it is ~0.5 fps on an Orin — use TrtDetector there."""

    mode = "none"

    def __init__(self, vmfb, mlir, params, bn, device="cuda"):
        import iree.runtime as ireert
        self.rt = ireert
        sig = parse_signature(mlir)
        self.weights = load_weights(sig, params, bn)
        self.cfg = ireert.Config(device)
        with open(vmfb, "rb") as f:
            self.ctx = ireert.SystemContext(config=self.cfg)
            self.vm = ireert.VmModule.copy_buffer(self.cfg.vm_instance, f.read())
        self.ctx.add_vm_module(self.vm)
        self.fn = self.ctx.modules.module["forward_eval"]

    def forward(self, x):
        out = self.fn(x.reshape(1, -1), *self.weights)     # the graph takes it flat
        return np.asarray(out).reshape(-1)[:NTOT]

    def __call__(self, img_rgb_hwc):
        return self.forward(preprocess(img_rgb_hwc, self.mode))


# ------------------------------------------------------------------- render

COLORS = [(255, 82, 82), (255, 158, 40), (255, 235, 59), (76, 217, 100),
          (0, 200, 190), (64, 156, 255), (140, 122, 255), (214, 106, 255),
          (255, 92, 170), (170, 170, 170)]


def draw(img_rgb_hwc, dets, out_path):
    from PIL import Image, ImageDraw
    pil = Image.fromarray(img_rgb_hwc).convert("RGB")
    W, H = pil.size
    d = ImageDraw.Draw(pil)
    for cid, score, (x0, y0, x1, y1) in dets:
        d.rectangle([x0 * W, y0 * H, x1 * W, y1 * H],
                    outline=COLORS[cid % len(COLORS)], width=2)
    pil.save(out_path)


def summarize(dets):
    """Count, top score, per-class table — the acceptance numbers a run reports."""
    counts = {}
    for cid, _s, _b in dets:
        counts[CLASS_NAMES[cid]] = counts.get(CLASS_NAMES[cid], 0) + 1
    top = max((s for _c, s, _b in dets), default=0.0)
    print(f"  {len(dets)} detections, top score {top:.4f}")
    for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {k:>16}: {v}")
    return len(dets), top, counts


def bench(det, img, n, conf_thresh):
    """Steady-state timing of the three stages, each on its own clock.

    preprocess = PIL resize + whatever host arithmetic the input mode leaves;
    forward = H2D + engine + D2H, synchronized; decode = decode() + NMS on the
    CPU. Three warm-up frames first: the first calls pay kernel load and
    allocator growth, which is not what a camera sees. Reported as means over n.
    """
    for _ in range(3):
        decode(det(img), conf_thresh=conf_thresh)
    tp = tf = td = 0.0
    for _ in range(n):
        t0 = time.perf_counter()
        x = preprocess(img, det.mode)
        t1 = time.perf_counter()
        flat = det.forward(x)
        t2 = time.perf_counter()
        decode(flat, conf_thresh=conf_thresh)
        t3 = time.perf_counter()
        tp += t1 - t0
        tf += t2 - t1
        td += t3 - t2
    tp, tf, td = 1e3 * tp / n, 1e3 * tf / n, 1e3 * td / n
    tot = tp + tf + td
    # The same forward, back to back with no CPU stage between calls. On a
    # Jetson the governor drops the GPU clock while the CPU stages run —
    # measured on the Orin Nano 2026-09-09 with this engine: 918 MHz under
    # trtexec's continuous driving (4.3 ms), 612 MHz back-to-back (6.8 ms),
    # 306 MHz interleaved with ~24 ms of CPU work (12.7 ms). So the forward
    # column above is the pessimistic in-pipeline number and this one is what
    # a saturated pipeline (or `jetson_clocks`) sees. Both are reported; neither
    # is "the" forward time on its own.
    t0 = time.perf_counter()
    for _ in range(n):
        det.forward(x)
    tf_b2b = 1e3 * (time.perf_counter() - t0) / n
    print(f"bench: {n} frames, input mode '{det.mode}'")
    print(f"  preprocess {tp:6.2f} ms | forward {tf:6.2f} ms | decode+nms "
          f"{td:6.2f} ms | total {tot:6.2f} ms")
    print(f"  forward-only {1e3 / tf:6.1f} fps | end-to-end {1e3 / tot:6.1f} fps")
    print(f"  forward back-to-back {tf_b2b:6.2f} ms = {1e3 / tf_b2b:6.1f} fps "
          f"(GPU clock not throttled by the CPU stages)")
    return tp, tf, td, tf_b2b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vmfb", default="build/detector.vmfb")
    ap.add_argument("--mlir", default="build/detector_fwd_eval_b1.mlir")
    ap.add_argument("--params", default="build/params.bin")
    ap.add_argument("--bn", default="build/bn_stats.bin")
    ap.add_argument("--device", default="cuda", help="cuda on Orin, local-task for CPU")
    ap.add_argument("--backend", default="trt", choices=["trt", "ort", "iree"],
                    help="trt = TensorRT engine (fast on Orin); ort = onnxruntime "
                         "on the ONNX itself (no engine; the dry run); iree = "
                         "portable but ~0.5 fps there")
    ap.add_argument("--plan", default="build/detector_aff30e28.plan",
                    help="TensorRT engine, built on device by trtexec")
    ap.add_argument("--onnx", default="build/detector_aff30e28.onnx",
                    help="the ONNX, for --backend ort")
    ap.add_argument("--input-mode", default="auto", choices=("auto",) + INPUT_MODES,
                    help="host preprocessing mode; auto reads it off the "
                         "engine/model input (name + dtype). Override only for "
                         "an ONNX exported before the input name carried it.")
    ap.add_argument("--image", default=None)
    ap.add_argument("--out", default="out.png")
    ap.add_argument("--camera", type=int, default=None, help="camera index (skeleton)")
    ap.add_argument("--conf-thresh", type=float, default=0.05)
    ap.add_argument("--bench", type=int, default=0,
                    help="time N frames, each stage separately, after warm-up")
    ap.add_argument("--gate-decode", action="store_true",
                    help="assert the fast decode matches decode_reference on "
                         "testdata/frame_logits.bin and exit. Needs no engine, no "
                         "GPU and no weights — run it once on the device before "
                         "trusting a frame rate that came out of the fast path.")
    args = ap.parse_args()

    if args.gate_decode:
        ref = Path(__file__).resolve().parent / "testdata" / "frame_logits.bin"
        flat = np.fromfile(ref, dtype=np.float32)
        n = _gate_decode(flat, conf_thresh=args.conf_thresh)
        print(f"✅ fast decode == decode_reference on {ref.name}: {n} detections")
        print(f"the Lean stack's own decode of testdata/frame.png "
              f"(conf >= {args.conf_thresh}):")
        summarize(decode(flat, conf_thresh=args.conf_thresh))
        return

    if args.backend == "trt":
        det = TrtDetector(args.plan, args.input_mode)
        print(f"loaded {args.plan} (TensorRT)")
    elif args.backend == "ort":
        det = OrtDetector(args.onnx, args.input_mode)
        print(f"loaded {args.onnx} (onnxruntime)")
    else:
        det = Detector(args.vmfb, args.mlir, args.params, args.bn, args.device)
        print(f"loaded {args.vmfb} (IREE, {args.device}) — expect ~0.5 fps on Orin")

    if args.camera is not None:
        # CAMERA — deliberately a stub. On Orin the IMX path is GStreamer via
        # nvarguscamerasrc, not a plain V4L2 index, and the exact pipeline is
        # sensor- and JetPack-specific. Fill this in on the device:
        #   cv2.VideoCapture("nvarguscamerasrc ! video/x-raw(memory:NVMM), "
        #                    "width=1456, height=1088, framerate=60/1 ! "
        #                    "nvvidconv ! video/x-raw, format=BGRx ! "
        #                    "videoconvert ! video/x-raw, format=RGB ! "
        #                    "appsink", cv2.CAP_GSTREAMER)
        # then loop: ret, frame = cap.read(); dets = decode(det(frame)); draw/print.
        print("camera mode is a stub — see the comment in main() for the "
              "GStreamer pipeline to fill in on device")
        return

    if not args.image:
        raise SystemExit("pass --image, or --camera once the pipeline is filled in")

    from PIL import Image
    img = np.asarray(Image.open(args.image).convert("RGB"), dtype=np.uint8)

    t0 = time.perf_counter()
    flat = det(img)
    t1 = time.perf_counter()
    dets = decode(flat, conf_thresh=args.conf_thresh)
    t2 = time.perf_counter()
    print(f"first frame (cold): forward {1e3*(t1-t0):.1f} ms | decode+nms "
          f"{1e3*(t2-t1):.1f} ms")
    summarize(dets)
    draw(img, dets, args.out)
    print(f"wrote {args.out}")

    if args.bench:
        bench(det, img, args.bench, args.conf_thresh)


if __name__ == "__main__":
    main()

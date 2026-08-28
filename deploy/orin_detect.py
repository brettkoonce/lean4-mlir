#!/usr/bin/env python3
"""Run the VisDrone FPN detector on a Jetson Orin (or any IREE target).

Self-contained on purpose: numpy + Pillow + the IREE runtime, nothing from the
training tree. Copy the `deploy/` directory and the two weight files to the
device and this runs.

    python3 orin_detect.py --vmfb build/detector.vmfb \
        --params build/params.bin --bn build/bn_stats.bin \
        --image frame.jpg --out out.png

Camera mode (skeleton — see CAMERA below):
    python3 orin_detect.py ... --camera 0 --fps-window 60

## Calling convention

The graph takes 190 tensors: the image first, then 189 weight tensors, in the
order they appear in the MLIR signature. Their total is exactly
21,548,743 params + 17,024 BN stats, which is `params.bin ++ bn_stats.bin`
concatenated and split by that signature. The signature is PARSED FROM THE MLIR
rather than hardcoded, so a retrained or re-shaped model cannot silently
mismatch — pass --mlir to re-derive it.

## The preprocessing contract, which is the easiest thing to get silently wrong

Training normalizes in the C loader, not in the graph (`ffi/f32_helpers.c`):
resize to 448x448 (SQUASH, not letterbox — measured 23% better on this data),
uint8 -> /255 -> subtract ImageNet mean -> divide by ImageNet std -> CHW ->
flatten. Deviating produces a detector that still runs and quietly gets worse.
"""
import argparse
import json
import re
import sys
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

def preprocess(img_rgb_hwc):
    """HWC uint8 (any size) -> [1, 3*448*448] float32, matching the C loader."""
    from PIL import Image
    pil = Image.fromarray(img_rgb_hwc).convert("RGB").resize(
        (IMG_PX, IMG_PX), Image.BILINEAR)          # SQUASH, not letterbox
    chw = np.asarray(pil, dtype=np.float32).transpose(2, 0, 1) / 255.0
    chw = (chw - MEAN) * ISTD
    return chw.reshape(1, -1).astype(np.float32)


# ------------------------------------------------------------------ decode

def _iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def decode(flat, conf_thresh=0.05, nms_iou=0.5, topk=300):
    """[NTOT] -> [(cid, score, (x0,y0,x1,y1))], normalized coords.

    Mirrors scripts/yolo_map_visdrone.py's decode_anchor_raw + decode_fpn: per
    scale, sigmoid objectness times max class softmax, centre confined to its own
    cell, size = anchor * exp(t) with t capped at 8 to match the training-time cap.
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


# ------------------------------------------------------------------ runtime

class Detector:
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

    def __call__(self, img_rgb_hwc):
        x = preprocess(img_rgb_hwc)
        out = self.fn(x, *self.weights)
        return np.asarray(out).reshape(-1)[:NTOT]


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vmfb", default="build/detector.vmfb")
    ap.add_argument("--mlir", default="build/detector_fwd_eval_b1.mlir")
    ap.add_argument("--params", default="build/params.bin")
    ap.add_argument("--bn", default="build/bn_stats.bin")
    ap.add_argument("--device", default="cuda", help="cuda on Orin, local-task for CPU")
    ap.add_argument("--image", default=None)
    ap.add_argument("--out", default="out.png")
    ap.add_argument("--camera", type=int, default=None, help="camera index (skeleton)")
    ap.add_argument("--conf-thresh", type=float, default=0.05)
    ap.add_argument("--bench", type=int, default=0, help="time N forward passes")
    args = ap.parse_args()

    det = Detector(args.vmfb, args.mlir, args.params, args.bn, args.device)
    print(f"loaded {args.vmfb} on {args.device}")

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
    print(f"forward {1e3*(t1-t0):.1f} ms | decode+nms {1e3*(t2-t1):.1f} ms | "
          f"{len(dets)} detections")
    counts = {}
    for cid, _s, _b in dets:
        counts[CLASS_NAMES[cid]] = counts.get(CLASS_NAMES[cid], 0) + 1
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {k:>16}: {v}")
    draw(img, dets, args.out)
    print(f"wrote {args.out}")

    if args.bench:
        # Warm up first: the first call pays kernel load, which is not the
        # steady-state number a camera would see.
        for _ in range(3):
            det(img)
        t0 = time.perf_counter()
        for _ in range(args.bench):
            det(img)
        dt = (time.perf_counter() - t0) / args.bench
        print(f"forward-only: {1e3*dt:.1f} ms/frame = {1.0/dt:.1f} fps "
              f"(decode+nms adds ~{1e3*(t2-t1):.0f} ms on CPU)")


if __name__ == "__main__":
    main()

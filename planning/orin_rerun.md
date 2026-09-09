# Orin: re-run the TensorRT benchmark on the trained detector

**Opened 2026-09-09. Host-side half DONE the same day on branch `orin/aff30e28-remeasure`;
the device-side half is the Orin Claude's.** See §4 at the end for what landed and what the
device run has to send back. The 35.7 fps figure was measured on the `ctrl12` weights (mAP 0.1526) on
purpose — throughput does not depend on weights. The user wants it re-run on the model that was
actually trained. Companion to `planning/archive/visdrone_detector.md` §13a-ter (deployment) and
`deploy/ORIN_SMOKE_TEST.md`.

## §0 How the number was made (2026-08-28, second device session)

| step | what | where |
|---|---|---|
| weights | `…__ctrl12_params.bin` (86,194,972 B) + `_bn_stats.bin` (68,096 B) | `demos/MainYolov1VisdroneFpn.lean:298-305` naming |
| export | PyTorch replica `demos/visdrone/bespoke/model.py`, `FpnDetector(pool="lean", pad="lean")`, `torch.onnx.export`, opset **18** (torch declined 17), 4 asymmetric-pad convs | `deploy/export_onnx.py:278-296` |
| gate | `--verify-frame`: onnxruntime on `deploy/testdata/frame.png` vs `frame_logits.bin`, rel < 1e-2, per-scale obj r ≥ 0.999 | `export_onnx.py:108-177` |
| engine | `trtexec --onnx=… --saveEngine=… --fp16` — the WHOLE flag set; batch 1 static | `ORIN_SMOKE_TEST.md:119` |
| run | `orin_detect.py --backend trt --plan … --bench 50`; numpy decode + NMS | `deploy/orin_detect.py` |
| result | 28.0 ms = 11.9 preprocess + 6.3 forward + 10.0 decode; Orin Nano 8 GB / 25 W / JetPack 6.2 / TRT 10.3 | `visdrone_detector.md:990-995` |

The 229 fps in `deploy/README.md:27-36` is the pre-`pad="lean"` wrong graph and was never
re-measured. The four `deploy/build/*.onnx` on the box are all `ctrl12` (two byte-identical).

## §1 Blockers, in the order they bite

1. ⛔ **`--verify-frame`'s golden is `ctrl12`'s.** `deploy/testdata/frame_logits.bin` is the Lean
   eval output for val record 374 under those weights. Any other checkpoint fails the gate with
   rel ≈ 1.0 and obj r ≪ 0.999 — exactly the signature `export_onnx.py:161-176` attributes to a
   padding bug, and nothing warns. Regenerate the golden from a Lean `infer` dump, or gate with
   the arm-agnostic `demos/visdrone/bespoke/diff_lean.py`. **Never widen `--tol` to get past it.**
2. ⛔ **The Orin host is documented nowhere in the repo** — only `orin:` placeholders
   (`deploy/README.md:100-104`, `build_orin.sh:71-72`) and `~/ckpt/` paths. It lives in the
   external edge-deploy notes. Write host/user/checkout path into `ORIN_SMOKE_TEST.md`.
3. ⛔ **Device-side patches were lost twice** (`ORIN_SMOKE_TEST.md:179-183, 195-202`): the pinned
   host buffer (`cuda.pagelocked_empty`), whatever made `execute_async_v3` work on that runtime,
   and the three-stage timing split. The repo's `--bench` (`orin_detect.py:455-465`) fuses
   preprocess into "forward" and **cannot reproduce the 11.9/6.3/10.0 breakdown**.
4. `.venv-timm` has torch 2.13 + torchvision but **no `onnx` / `onnxruntime`** — the external-data
   fold degrades silently (`export_onnx.py:306`) and `--verify-frame` hard-exits (:128-131).
   Install there; never into the pinned `.venv`.

Also: the Sep-2 detector binary is from the `yolo-v5-assignment` branch (different box decode) —
rebuild on `main`. The device decode is **argmax** (`orin_detect.py:261`); the 0.1961/0.2363
headlines are the multilabel decode — quote fps, not mAP, next to the device number.
`ORIN_SMOKE_TEST.md:87` names a branch that no longer exists and :143-162's acceptance counts are
`ctrl12`'s.

## §2 Steps (exists / needs writing)

Use `TAG=aff30e28` (0.2363) — same architecture as `cfoc2`, same throughput, 20% better model.

1. `lake build yolov1-visdrone-fpn` on `main`. *exists*
2. Lean reference dump: `LD_LIBRARY_PATH=ffi FPN_BACKBONE=r34 FPN_TOWER=0 FPN_TAG=aff30e28
   .lake/build/bin/yolov1-visdrone-fpn infer data/visdrone_fpn runs/<out>` → `logits.bin`. *exists*
3. `.venv-timm/bin/pip install onnx onnxruntime`. *needs doing*
4. Export: `.venv-timm/bin/python deploy/export_onnx.py --ckpt …aff30e28_params.bin --bn
   …aff30e28_bn_stats.bin --out deploy/build/detector_aff30e28.onnx` — expect opset 18, 4
   asymmetric-pad convs. *exists*
5. Gate: `python -m bespoke.diff_lean --ckpt … --bn-stats … --data data/visdrone_fpn/val.bin
   --lean-logits runs/<out>/logits.bin --eval-mode --n 64` from `demos/visdrone` (~1.4e-3 rel,
   obj r = 1.0000 ×3). ⚠ `export_onnx.py --verify` has an ABSOLUTE tolerance despite its help
   text (:359-382). *exists*
6. New golden: record 374 out of `logits.bin` (`np.fromfile(...,f32).reshape(-1,185220)[374]`)
   → `frame_logits.bin`; `verify_frame` hardcodes the path (:134), so add `--ref-logits` or
   overwrite. Then `--verify-frame` passes again. *needs writing (3 lines + a flag)*
7. New acceptance numbers: `orin_detect.py --gate-decode` on the training box → detection count,
   top score, per-class table → replace `ORIN_SMOKE_TEST.md:143-151`. *exists / doc edit*
8. `md5sum` + size of the ONNX into the runbook. *exists*
9. `scp` to the device (host from §1.2); `trtexec --onnx=… --saveEngine=build/detector_aff30e28.plan
   --fp16` — new engine name, never reuse a `.plan`. *exists*
10. `orin_detect.py --gate-decode`, then `--backend trt --plan … --image testdata/frame.png
    --bench 50`. *exists*
11. Re-add the three-stage timing (`orin_detect.py:331-339, 455-465`) and the pinned buffer in
    `TrtDetector`. *needs writing, ~20 lines* — commit the device diff back THIS time.
12. Optional, the real speed lever: `--fold-preprocess u8` (gated at 4.959e-04, never run on
    device) + a `TrtDetector` that accepts `[1,448,448,3]` uint8 (`_wrap_preprocess` exists at
    `export_onnx.py:180-220`; the runner side does not: `orin_detect.py:326,331-332` hardcode the
    f32 buffer). Projected 28 → ~17 ms, ~59 fps. *needs writing*
13. `runs/2026-09-XX-orin-remeasure/README.md` with the trtexec log, the split, fps, md5. *needs writing*

## §3 Traps

INT8 is off the table without new work: background-cell class logits are unconstrained (±1287
observed) because the class softmax is masked to positives — fine in fp16, fatal to calibration
(`deploy/README.md:73-76`). The padfix (`pad="lean"`) is load-bearing and the correlation-falls
diagnostic at `export_onnx.py:161-176` must not be "fixed" by loosening. Checkpoints are outside
git (`.lake/build/`, 86 MB each); `deploy/build/` is gitignored yet holds four ONNX files locally.

## §4 Status 2026-09-09 — the branch the Orin pulls

Done on the training box (all of §2's items 1–8, plus 11–12's code):

- §1.1 golden: `export_onnx.py --regen-golden <logits.bin>` cuts row 374 out of an `infer`
  dump (and checks the row IS `frame.png` against `val.bin`); `--ref-logits` overrides the
  path. `testdata/frame_logits.bin` is now aff30e28's.
- §1.3 device patches: `TrtDetector` reads input name/shape/dtype and output size off the
  ENGINE, page-locks both host buffers, falls back to `execute_async_v2` on TensorRT 8, and
  splits `forward()` from `preprocess()`; `--bench` times preprocess / forward / decode+nms
  on separate clocks and prints forward-only and end-to-end fps.
- §1.4 `.venv-timm` has onnx 1.22 / onnxruntime 1.23 / onnxscript 0.7 (torch 2.13's
  exporter imports onnxscript unconditionally). Export pins the TorchScript exporter
  (`dynamo=False`) and folds the four explicit Pad ops torch 2.13 leaves behind into the
  convs' `pads`, so the census (4 asymmetric-pad convs, 0 Pad) means what it says.
- §2.5 `--verify` is RELATIVE now and shares `check_against_lean` with `--verify-frame`.
- §2.12 runner side of the u8 fold: the input tensor NAME carries the preprocessing contract
  (`image` / `image_01` / `image_u8`, `INPUT_NAMES` in `export_onnx.py`), the runner derives
  the mode from name + dtype, `--input-mode` overrides for pre-naming files. Both
  `detector_aff30e28.onnx` and `detector_aff30e28_u8.onnx` are exported and gated.
- New `--backend ort`: the whole runner on the ONNX through onnxruntime, so the training box
  dry-runs preprocess → forward → decode → bench without an engine. Both files decode the
  reference frame to 232 detections at top 0.6176.
- Numbers, md5s, and the device-side blanks: `runs/2026-09-09-orin-remeasure/README.md`.
  Runbook rewritten for aff30e28: `deploy/ORIN_SMOKE_TEST.md`.

Device run 2026-09-09 (Orin's commit `d8d238b2`, on the branch once pulled from the device —
that box has no push credentials):

- ✅ md5 gate, `--gate-decode` (232 / 0.6175 exactly), plain engine built in 66 s with no fp16
  fallbacks, reference frame 232 / 0.6180 (fp16 moved one box pedestrian→people).
- ✅ `TrtDetector` ran as written — NO code change needed on device.
- Governor-managed bench: 11.14 / 15.48 / 13.07 / 39.69 ms → **25.2 fps end-to-end**;
  trtexec 4.12 ms GPU compute on the same engine, identical to ctrl12's 4.13. ⚠ The 15.48 ms
  forward is a CLOCK artifact, not a model one: the interleaved bench lets devfreq drop the GPU
  to 306 MHz between forwards (918 MHz continuous 4.3 ms / 612 back-to-back 6.8 / 306
  interleaved 12.7). ctrl12's 6.3 ms was the back-to-back shape. `jetson_clocks` not applied
  (password). `--bench` now prints a back-to-back forward line as well.
- ⛔ u8 engine REFUSED: `legalUINT8` — TensorRT 10.3 takes UINT8 only as network I/O, and the
  export did permute-then-cast (uint8 Transpose at node 0). Fixed 2026-09-09: cast first;
  re-exported (md5 `08f6df63…`, same size) and re-gated (same numbers); needs re-sending to the
  device and a second run.
- Facts: L4T R36.4.7, TRT 10.3.0, pycuda 2026.1 in `~/orinvenv` (not system python),
  `trtexec` at `/usr/src/tensorrt/bin/`, 25 W, 46–47 °C. Host name still outside the repo.

Still open: the u8 engine run (and, with `jetson_clocks`, a pinned-clock bench of both), then
land the branch on main by fast-forward.

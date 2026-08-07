import demos.Yolov1VisdroneFpnCommon

/-! # `yolov1-visdrone-fpn` — the FPN detector on IREE

The shared body lives in `demos/Yolov1VisdroneFpnCommon.lean`; this root exists
only to link it against `ffi/libiree_ffi.so`. See `MainYolov1VisdroneFpnXla.lean`
for the PJRT/XLA twin.

```
lake build yolov1-visdrone-fpn
IREE_BACKEND=rocm HIP_VISIBLE_DEVICES=0 \
  .lake/build/bin/yolov1-visdrone-fpn data/visdrone_fpn
```

⚠ `IREE_BACKEND=rocm` is **required** on this box. It defaults to `cuda`, and the
`gfx1100` reduction workaround in `ireeCompileArgs` (`LeanMlir/Types.lean`) is
gated on `rocm` — without it the multi-scale loss's N-D→scalar reductions abort
with `'func.func' op failed to distribute`.
-/

def main (args : List String) : IO Unit := runYolov1VisdroneFpn args

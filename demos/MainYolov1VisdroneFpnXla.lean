import demos.Yolov1VisdroneFpnCommon

/-! # `yolov1-visdrone-fpn-xla` — the FPN detector on XLA/PJRT

The same program as `yolov1-visdrone-fpn` (shared body in
`demos/Yolov1VisdroneFpnCommon.lean`), linked against `ffi/libpjrt_ffi.so`.

**This is the first demo on the XLA path.** The five verified nets were already
backend-agnostic via `VerifiedTrain.mkSession`; the demos go through
`LeanMlir/Train.lean`, which was IREE-hardcoded. That file now derives every
artifact path through `NetSpec.graphArtifact`, so the `.mlir` goes straight to
PJRT here and no `.vmfb` is ever produced.

**This does not move the verification tier.** Same NetSpec, same emitted
StableHLO, same §1a ties — only the trusted lowerer differs.

```
gcc -fPIC -O2 -shared ffi/pjrt_ffi.c -ldl -o ffi/libpjrt_ffi.so
lake build yolov1-visdrone-fpn-xla
HIP_VISIBLE_DEVICES=0 .lake/build/bin/yolov1-visdrone-fpn-xla data/visdrone_fpn
```

⚠ `IREE_BACKEND` is inert here — PJRT does not read it. Anything setting it and
expecting an effect silently gets none.
-/

def main (args : List String) : IO Unit := runYolov1VisdroneFpn args

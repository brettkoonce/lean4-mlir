import apps.cifar.Cifar8MomCommon

/-! # `cifar8-momentum-xla` — the CIFAR-8 (no BN) + Nesterov-momentum SGD trainer on **XLA/PJRT**

The XLA peer of `cifar8-momentum`: same body (`apps/cifar/Cifar8MomCommon.lean`), same
certified artifact, same schedule and seed — only the linked lowerer differs, which is what
makes a cross-backend disagreement attributable to the lowerer rather than to the run.

Completes `lake run cifar-xla` to the same six-way optimizer ablation `lake run cifar` covers
(BN/no-BN × SGD/momentum/AdamW).

Run: `.lake/build/bin/cifar8-momentum-xla data`  — no `IREE_BACKEND`; the backend is
whichever `.so` this target linked, and `ffi/libpjrt_ffi.so` must exist (`lake run cifar-xla`
builds it for you).
-/

def main (argv : List String) : IO Unit := runCifar8Mom argv

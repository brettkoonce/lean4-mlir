import apps.imagenette.ConvNeXtImagenetCommon

/-! # `convnext-imagenet-verified-xla` — ConvNeXt-T on full ImageNet-1k, verified renderer → XLA

The ConvNeXt peer of `resnet34-imagenet-verified-xla` and `vit-imagenet-verified-xla` (§2p).
Unlike those two, this one needed a renderer change first: `nClasses` was a hardcoded literal and
`-α/K` was a caller-supplied string independent of it — the two-writers-for-one-fact shape that
produced R34's K=10 gradient bug. Both are fixed; `cBS` is still private, so this renders at
batch 32 (global 128 on four replicas).

XLA-only by construction: collectives live on the PJRT path.

⚠ Does NOT move the verification tier, and is NOT the ConvNeXt paper recipe — see
`ConvNeXtImagenetCommon`'s claim-ceiling note before quoting anything from it.
-/

def main (argv : List String) : IO Unit := runConvNeXtImagenet argv

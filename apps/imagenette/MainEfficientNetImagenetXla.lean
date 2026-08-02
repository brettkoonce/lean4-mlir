import apps.imagenette.EfficientNetImagenetCommon

/-! # `efficientnet-imagenet-verified-xla` — EfficientNet-B0 on full ImageNet-1k, verified → XLA

The fourth and last of the ImageNet scale-tier trainers (§2p). `B` and `nClasses` were already
renderer parameters, so this needed only a `slug` — plus the derived −α/K that turned up the third
copy of §2k's hardcoded-K bug, this time in EfficientNet's report-only loss.

⚠ Does NOT move the verification tier, and the OPTIMIZER does not match the reference (RMSProp +
exponential decay there, AdamW + cosine here). See `EfficientNetImagenetCommon`.
-/

def main (argv : List String) : IO Unit := runEfficientNetImagenet argv

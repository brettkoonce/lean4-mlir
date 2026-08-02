import apps.imagenette.MobileNetV2ImagenetCommon

/-! # `mobilenetv2-imagenet-verified-xla` — MobileNetV2 on full ImageNet-1k, verified → XLA

The last of the five scale-tier trainers (§2p). Needed only a `slug` plus derived label-smoothing
constants — and mnv2 was the worst of the five on that axis, carrying the K=10 value in the
COTANGENT (the gradient path, §2k's original bug) as well as in the report-only loss.

⚠ Does NOT move the verification tier; optimizer does not match the reference (RMSProp there,
AdamW here). See `MobileNetV2ImagenetCommon`.
-/

def main (argv : List String) : IO Unit := runMobileNetV2Imagenet argv

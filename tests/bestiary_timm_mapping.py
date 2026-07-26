"""Hand-curated mapping: (bestiary binary, variant name) → timm model name.

Rules of thumb for adding a row:
- Only for image classifier architectures that timm actually ships (timm is
  image-models-only). LLMs, detection, diffusion, speech belong in a
  future HuggingFace-based equivalent script.
- Our bestiary variants labeled `(bestiary approximation)` or similar
  hedges are *expected* to differ from timm by a few %. Rows are still
  useful to pin the magnitude of that divergence.
- `None` as the timm name marks "we know there's no timm counterpart"
  (documented non-coverage, different from "we forgot to add it").

Expand over time; nothing breaks if a row is missing — the oracle simply
covers a smaller subset.
"""

MAPPING: dict[tuple[str, str], str | None] = {
    # AlexNet — not in timm (torchvision ships it; timm skips the pre-ResNet era)
    ("bestiary-alexnet", "AlexNet (Krizhevsky 2012)"): None,

    # Inception family — GoogLeNet/v1 isn't in timm; v3 and v4 are. Don't
    # cross-check v1 against v3 — they're structurally different.
    ("bestiary-inception", "GoogLeNet (Inception v1)"): None,
    ("bestiary-inception", "Inception-v3 (bestiary approximation)"): "inception_v3",
    ("bestiary-inception", "Inception-v4 (bestiary approximation)"): "inception_v4",

    # Xception
    ("bestiary-xception", "Xception"): "xception",

    # ConvNeXt left the bestiary in 65f770f — promoted to the primary
    # sequence (Ch 9), where it has its own verified track. No bestiary-convnext
    # binary exists, so there is nothing here to cross-check.

    # VGG — matches torchvision/timm exactly on sight; these rows pin that.
    ("bestiary-vgg", "VGG-11"): "vgg11",
    ("bestiary-vgg", "VGG-13"): "vgg13",
    ("bestiary-vgg", "VGG-16"): "vgg16",
    ("bestiary-vgg", "VGG-19"): "vgg19",
    ("bestiary-vgg", "tiny-VGG"): None,          # our own fixture

    # ResNet — likewise an exact match on sight for all four.
    ("bestiary-resnet", "ResNet-18"): "resnet18",
    ("bestiary-resnet", "ResNet-50"): "resnet50",
    ("bestiary-resnet", "ResNet-101"): "resnet101",
    ("bestiary-resnet", "ResNet-152"): "resnet152",
    ("bestiary-resnet", "tiny-ResNet (basic blocks)"): None,

    # DenseNet — THE REASON THESE ROWS EXIST. Our counts run ~10-13% under
    # the reference (121: 6,972,234 vs ~7,978,856; 169: 12,511,946 vs
    # ~14,149,480; 201: 18,125,258 vs ~20,013,928) while VGG and ResNet from
    # the same batch match exactly. Each shortfall is just under that net's
    # classifier size and scales with the final feature dimension, so the
    # suspicion is the head or the final BN in `Layer.denseBlock`'s
    # shape-only accounting — NOT the dense blocks (the primitive's own note
    # says it counts the per-layer BN + 1×1 + BN + 3×3 sub-stack). Run this
    # oracle in a timm env to confirm or kill that.
    ("bestiary-densenet", "DenseNet-121"): "densenet121",
    ("bestiary-densenet", "DenseNet-169"): "densenet169",
    ("bestiary-densenet", "DenseNet-201"): "densenet201",
    ("bestiary-densenet", "tiny-DenseNet"): None,

    # Swin Transformer v1
    ("bestiary-swin", "Swin-T"): "swin_tiny_patch4_window7_224",
    ("bestiary-swin", "Swin-S"): "swin_small_patch4_window7_224",
    ("bestiary-swin", "Swin-B"): "swin_base_patch4_window7_224",

    # MobileViT
    ("bestiary-mobilevit", "MobileViT-XXS"): "mobilevit_xxs",
    ("bestiary-mobilevit", "MobileViT-XS"): "mobilevit_xs",
    ("bestiary-mobilevit", "MobileViT-S"): "mobilevit_s",

    # Non-matches (explicit so we don't forget — still in timm's scope but
    # either we have a simplified spec or the timm name doesn't line up)
    ("bestiary-squeezenet", "SqueezeNet 1.0"): None,   # in torchvision, not timm
    ("bestiary-squeezenet", "SqueezeNet 1.1"): None,
    ("bestiary-shufflenet", "ShuffleNet 1.0× (g=3)"): None,
    ("bestiary-shufflenetv2", "ShuffleNet v2 1.0×"): None,
    ("bestiary-lenet", "LeNet-5"): None,               # too small / not a timm target

    # Highway — the 2015 gating paper has no standard implementation in timm
    # (or torchvision); it survives as ResNet's ancestor, not as a model you
    # instantiate. Both halves listed so neither reads as an oversight.
    ("bestiary-highway", "Highway-50 — main path H(x)"): None,
    ("bestiary-highway", "Highway-50 — transform gate T(x)"): None,
    ("bestiary-highway", "Highway-100 — main path H(x)"): None,
    ("bestiary-highway", "Highway-100 — transform gate T(x)"): None,
    ("bestiary-highway", "tiny-Highway — main path H(x)"): None,
    ("bestiary-highway", "tiny-Highway — transform gate T(x)"): None,

    # WRN — timm ships wide_resnet50_2 / wide_resnet101_2, which are
    # ImageNet bottleneck-widened ResNets. Ours are the CIFAR WRN-d-k of
    # Zagoruyko & Komodakis (28-10, 40-2, 22-8). Different family; cross-
    # checking them against timm's would compare unlike things.
    ("bestiary-wrn", "WRN-28-10"): None,
    ("bestiary-wrn", "WRN-40-2"): None,
    ("bestiary-wrn", "WRN-22-8"): None,
    ("bestiary-wrn", "tiny-WRN (10-2)"): None,
}

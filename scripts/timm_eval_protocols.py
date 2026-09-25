#!/usr/bin/env python3
"""timm_eval_protocols.py — timm's validation protocol for every ImageNet trainer we emit.

timm scores a checkpoint at its pretrained config's `test_input_size` / `test_crop_pct` (falling
back to `input_size` / `crop_pct`), resizing the shorter side to size/crop_pct with the config's
interpolation and centre-cropping. That is often NOT the training resolution: ConvNeXt-T
(`fb_in1k`) and RSB-A1/A2 test at 288, MNv4-Conv-M r224 at 256, all at crop 1.0; DeiT crops 0.9.

Each generated trainer (`jax/generated/generated_<stem>.py`) is mapped to the timm pretrained tag
whose recipe it reproduces, and the protocol is READ from the pinned timm (1.0.28, `.venv-timm`),
never transcribed. `jax/scripts/eval_full50k.py` reads the JSON (`PROTOCOL=timm`).

    .venv-timm/bin/python scripts/timm_eval_protocols.py           # write jax/timm_eval_protocols.json
    .venv-timm/bin/python scripts/timm_eval_protocols.py --check   # drift -> exit 1

⚠ Run it from `.venv-timm` only; timm must never be installed into the main `.venv`.
"""
import json, sys, os

OUT = os.path.join(os.path.dirname(__file__), "..", "jax", "timm_eval_protocols.json")

# (generated-file stem prefix, timm pretrained tag, why this tag). Longest prefix wins, so a
# recipe-specific stem overrides its net's default.
MAP = [
    ("resnet34_imagenet",              "resnet34.tv_in1k",        "the 2018 torchvision recipe"),
    ("resnet50_imagenet_2018",         "resnet50.tv_in1k",        "the 2018 torchvision recipe"),
    ("resnet50_imagenet",              "resnet50.a2_in1k",        "`default` is RSB-A2"),
    ("resnet50_imagenet_a2",           "resnet50.a2_in1k",        "RSB-A2"),
    ("resnet50_imagenet_a1",           "resnet50.a1_in1k",        "RSB-A1"),
    ("resnet50_imagenet_short",        "resnet50.a3_in1k",        "RSB-A3 (train 160)"),
    ("resnet50_imagenet_rsbfaithful",  "resnet50.a3_in1k",        "RSB-A3 (train 160)"),
    ("resnet50_imagenet_true2048",     "resnet50.a3_in1k",        "RSB-A3 (train 160)"),
    ("resnet50_imagenet_adamprobe",    "resnet50.a3_in1k",        "RSB-A3 (train 160)"),
    ("mobilenet_v2_imagenet",          "mobilenetv2_100.ra_in1k", "MobileNetV2 1.0"),
    ("mobilenet_v4_imagenet",          "mobilenetv4_conv_medium.e500_r224_in1k", "Conv-M trained at 224"),
    ("efficientnet_b0_imagenet",       "tf_efficientnet_b0.in1k", "the paper's TF recipe, ported"),
    ("convnext_tiny_imagenet",         "convnext_tiny.fb_in1k",   "the paper's weights"),
    ("convnext_s_imagenet",            "convnext_small.fb_in1k",  "the paper's weights"),
    ("convnext_b_imagenet",            "convnext_base.fb_in1k",   "the paper's weights"),
    ("vit_tiny_imagenet",              "deit_tiny_patch16_224.fb_in1k", "DeiT-Ti"),
    ("vit_s_imagenet",                 "deit_small_patch16_224.fb_in1k", "DeiT-S"),
    ("vit_b_imagenet",                 "deit_base_patch16_224.fb_in1k",  "DeiT-B"),
]


def build():
    import timm
    from timm.models import get_pretrained_cfg
    out = {"timm_version": timm.__version__, "nets": {}}
    for prefix, tag, why in MAP:
        c = get_pretrained_cfg(tag)
        if c is None:
            sys.exit(f"timm {timm.__version__} has no pretrained cfg {tag}")
        d = c.to_dict()
        test = d.get("test_input_size") or d["input_size"]
        out["nets"][prefix] = {
            "timm": tag, "why": why,
            "train_size": d["input_size"][1], "train_crop_pct": d["crop_pct"],
            "test_size": test[1], "test_crop_pct": d.get("test_crop_pct") or d["crop_pct"],
            "interpolation": d["interpolation"],
            "mean": [round(x, 6) for x in d["mean"]], "std": [round(x, 6) for x in d["std"]],
        }
    return out


def lookup(table, gen_path):
    """The protocol for a generated trainer path: longest matching stem prefix, else None."""
    stem = os.path.basename(gen_path)
    stem = stem[len("generated_"):] if stem.startswith("generated_") else stem
    stem = stem[:-3] if stem.endswith(".py") else stem
    hits = [p for p in table["nets"] if stem == p or stem.startswith(p + "_")]
    return table["nets"][max(hits, key=len)] if hits else None


if __name__ == "__main__":
    new = json.dumps(build(), indent=1, sort_keys=True) + "\n"
    if "--check" in sys.argv:
        old = open(OUT).read() if os.path.exists(OUT) else ""
        if old != new:
            sys.exit(f"⛔ {OUT} is not what timm {json.loads(new)['timm_version']} says — rerun without --check")
        print(f"✅ {OUT} matches timm")
    else:
        open(OUT, "w").write(new)
        print(f"wrote {OUT} ({len(MAP)} nets)")

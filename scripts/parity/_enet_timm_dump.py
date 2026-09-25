#!/usr/bin/env python3
"""timm half of `scripts/parity/enet_timm_parity.py` — runs in `.venv-timm` (CPU torch, timm 1.0.28).

Builds EfficientNet-B0 at a fixed seed, randomises every BN's affine and running statistics, runs
one TRAIN-mode forward (batch statistics) and one EVAL-mode forward (running statistics), and
writes the parameters in the JAX reference's `params[k][j]` order next to both logits.

`--pad`:
  tf      `tf_efficientnet_b0` — TF's net: SAME padding at every conv (the paper's)
  hybrid  `efficientnet_b0` (symmetric) with the stem swapped for a SAME conv — what the JAX and
          verified paths compute today (planning/imagenet_parity.md §2.1: "SAME stem + symmetric
          strided depthwise")
  sym     `efficientnet_b0` as is — torchvision/timm symmetric everywhere (a control)
Classifier dropout and drop-path are 0, so train mode is deterministic.

The JAX order per MBConv: expand (conv, BN) when t > 1, depthwise (conv, BN), SE reduce (W, b),
SE expand (W, b), project (conv, BN); stem (conv, BN) first, head (conv, BN) and the classifier
(W [out, in], b) last. timm registers its modules in that order; the pairing is asserted.
"""
import argparse
import numpy as np
import timm
import torch
import torch.nn as nn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--classes", type=int, default=10)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--eps", type=float, default=1e-5)
    ap.add_argument("--pad", default="hybrid", choices=["tf", "hybrid", "sym"])
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    name = "tf_efficientnet_b0" if a.pad == "tf" else "efficientnet_b0"
    m = timm.create_model(name, num_classes=a.classes, drop_rate=0.0, drop_path_rate=0.0)
    if a.pad == "hybrid":
        from timm.layers import create_conv2d
        old = m.conv_stem
        new = create_conv2d(3, old.out_channels, 3, stride=2, padding="same", bias=False)
        with torch.no_grad():
            new.weight.copy_(old.weight)
        m.conv_stem = new
    g = torch.Generator().manual_seed(a.seed + 1)
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod, nn.BatchNorm2d):
                c = mod.num_features
                mod.eps = a.eps
                mod.weight.copy_(0.5 + torch.rand(c, generator=g))
                mod.bias.copy_(0.2 * torch.randn(c, generator=g))
                mod.running_mean.copy_(0.1 * torch.randn(c, generator=g))
                mod.running_var.copy_(0.5 + torch.rand(c, generator=g))

    params, stats, pending = [], [], None
    for mname, mod in m.named_modules():
        if isinstance(mod, nn.Conv2d):
            if mod.bias is not None:             # the SE 1x1s: (W, b), no BN
                assert pending is None and ".se." in f".{mname}.", mname
                params.append([mod.weight, mod.bias])
                continue
            assert pending is None, f"two convs without a BN between them: {pending[0]} then {mname}"
            pending = (mname, mod)
        elif isinstance(mod, nn.BatchNorm2d):
            assert pending is not None, f"BN {mname} with no conv before it"
            cname, conv = pending
            assert conv.out_channels == mod.num_features, (cname, mname)
            params.append([conv.weight, mod.weight, mod.bias])
            stats.append([mod.running_mean, mod.running_var])
            pending = None
        elif isinstance(mod, nn.Linear):
            assert pending is None
            params.append([mod.weight, mod.bias])
    assert pending is None

    x = torch.randn(a.batch, 3, a.res, a.res, generator=g)
    with torch.no_grad():
        m.train()
        for mod in m.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.momentum = 0.0
        y_train = m(x).numpy()
        m.eval()
        y_eval = m(x).numpy()

    out = {"x": x.numpy(), "y_train": y_train, "y_eval": y_eval,
           "n_params": np.array(len(params)), "n_stats": np.array(len(stats))}
    for k, group in enumerate(params):
        for j, t in enumerate(group):
            out[f"p{k}_{j}"] = t.detach().numpy()
    for k, (mu, var) in enumerate(stats):
        out[f"s{k}_0"] = mu.numpy()
        out[f"s{k}_1"] = var.numpy()
    np.savez(a.out, **out)
    print(f"timm {name} (pad {a.pad}, BN ε {a.eps:g}): {len(params)} param groups, {len(stats)} BN, "
          f"{sum(p.numel() for p in m.parameters())} params")


if __name__ == "__main__":
    main()

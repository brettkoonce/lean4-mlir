#!/usr/bin/env python3
"""timm half of `scripts/mnv4_timm_parity.py` — runs in `.venv-timm` (CPU torch, timm 1.0.28).

Builds `mobilenetv4_conv_medium` at a fixed seed with every BatchNorm's affine and running
statistics randomised (so no BN parameter is exercised only at its init value), runs one forward
in TRAIN mode (batch statistics) and one in EVAL mode (running statistics), and writes the
parameters in the JAX reference's `params[k][j]` order next to both logits.

The JAX order is the forward order of (conv, BN) pairs, then the classifier `(W [out, in], b)`.
timm registers its modules in forward order (`conv_stem, bn1, blocks, global_pool, conv_head,
norm_head, …, classifier`), so pairing each Conv2d with the next BatchNorm2d in `named_modules()`
reproduces it; the pairing is asserted, not assumed.

Usage (called by the parity script; not meant to be run by hand):
    .venv-timm/bin/python scripts/_mnv4_timm_dump.py OUT.npz --classes 10 --batch 4 --seed 0
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
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    m = timm.create_model("mobilenetv4_conv_medium", num_classes=a.classes)
    g = torch.Generator().manual_seed(a.seed + 1)
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod, nn.BatchNorm2d):
                c = mod.num_features
                mod.weight.copy_(0.5 + torch.rand(c, generator=g))
                mod.bias.copy_(0.2 * torch.randn(c, generator=g))
                mod.running_mean.copy_(0.1 * torch.randn(c, generator=g))
                mod.running_var.copy_(0.5 + torch.rand(c, generator=g))

    params, stats, pending = [], [], None
    for name, mod in m.named_modules():
        if isinstance(mod, nn.Conv2d):
            assert pending is None, f"two convs without a BN between them: {pending[0]} then {name}"
            assert mod.bias is None, f"{name}: timm's MNv4 convs carry no bias"
            pending = (name, mod)
        elif isinstance(mod, nn.BatchNorm2d):
            assert pending is not None, f"BN {name} with no conv before it"
            cname, conv = pending
            assert conv.out_channels == mod.num_features, (cname, name)
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
        # BN momentum 0 keeps the running statistics untouched by the train-mode pass, so the
        # eval pass below reads exactly what was dumped.
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
    print(f"timm mobilenetv4_conv_medium: {len(params)} param groups, {len(stats)} BN, "
          f"{sum(p.numel() for p in m.parameters())} params")


if __name__ == "__main__":
    main()

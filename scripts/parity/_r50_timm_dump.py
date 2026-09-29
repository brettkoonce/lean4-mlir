#!/usr/bin/env python3
"""timm half of `scripts/parity/r50_timm_parity.py` — runs in `.venv-timm` (CPU torch, timm 1.0.28).

Builds `resnet50` (the RSB-A1/A2/A3 checkpoints' architecture: `resnet50.a{1,2,3}_in1k`) at a fixed
seed with every BatchNorm's affine and running statistics randomised, and writes, in the JAX
reference's `params[k][j]` order:

  * the parameters: (conv W, BN γ, BN β) per conv in forward order, each block's downsample last,
    then the classifier (W [out, in], b);
  * the running statistics (μ, σ²) per BN in the same order;
  * logits in TRAIN mode (batch statistics) at 224 and 160 (RSB-A3's train size), and in EVAL mode
    (running statistics) at 224 and 288 (timm's A1/A2 test size);
  * the per-block drop-path probabilities timm assigns at `drop_path_rate` (RSB's 0.05).

The (conv, BN) pairing is asserted, not assumed.

Usage (called by the parity script):
    .venv-timm/bin/python scripts/parity/_r50_timm_dump.py OUT.npz --batch 4 --seed 0
"""
import argparse
import numpy as np
import timm
import torch
import torch.nn as nn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--classes", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--drop-path", type=float, default=0.05)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    m = timm.create_model("resnet50", num_classes=a.classes, drop_path_rate=a.drop_path)
    g = torch.Generator().manual_seed(a.seed + 1)
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod, nn.BatchNorm2d):
                c = mod.num_features
                mod.weight.copy_(0.5 + torch.rand(c, generator=g))
                mod.bias.copy_(0.2 * torch.randn(c, generator=g))
                mod.running_mean.copy_(0.1 * torch.randn(c, generator=g))
                mod.running_var.copy_(0.5 + torch.rand(c, generator=g))
                assert mod.eps == 1e-5, f"BN eps {mod.eps}"

    params, stats, pending = [], [], None
    for name, mod in m.named_modules():
        if isinstance(mod, nn.Conv2d):
            assert pending is None, f"two convs without a BN between them: {pending[0]} then {name}"
            assert mod.bias is None, f"{name}: timm's ResNet convs carry no bias"
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

    blocks = [b for layer in (m.layer1, m.layer2, m.layer3, m.layer4) for b in layer]
    drop = [float(getattr(b.drop_path, "drop_prob", 0.0)) for b in blocks]

    out = {"n_params": np.array(len(params)), "n_stats": np.array(len(stats)),
           "drop_probs": np.array(drop)}
    with torch.no_grad():
        # BN momentum 0 keeps the running statistics untouched by the train-mode passes, so the
        # eval passes read exactly what was dumped. Drop-path is identity in eval and is switched
        # off for the train passes (a random mask is not comparable across frameworks).
        for mod in m.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.momentum = 0.0
        for b in blocks:
            if hasattr(b.drop_path, "drop_prob"):
                b.drop_path.drop_prob = 0.0
        for mode, res in (("train", 224), ("train", 160), ("eval", 224), ("eval", 288)):
            x = torch.randn(a.batch, 3, res, res, generator=g)
            m.train(mode == "train")
            out[f"x_{mode}{res}"] = x.numpy()
            out[f"y_{mode}{res}"] = m(x).numpy()
    for k, group in enumerate(params):
        for j, t in enumerate(group):
            out[f"p{k}_{j}"] = t.detach().numpy()
    for k, (mu, var) in enumerate(stats):
        out[f"s{k}_0"] = mu.numpy()
        out[f"s{k}_1"] = var.numpy()
    np.savez(a.out, **out)
    print(f"timm resnet50: {len(params)} param groups, {len(stats)} BN, "
          f"{sum(p.numel() for p in m.parameters())} params, drop-path {drop[0]:.4f}…{drop[-1]:.4f}")


if __name__ == "__main__":
    main()

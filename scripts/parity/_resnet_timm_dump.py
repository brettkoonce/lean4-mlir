#!/usr/bin/env python3
"""timm half of `scripts/parity/resnet_timm_parity.py` — runs in `.venv-timm` (CPU torch, timm 1.0.28).

Builds `resnet50` (the RSB-A1/A2/A3 and 2018 nets: `resnet50.a{1,2,3}_in1k`, `resnet50.tv_in1k`) or
`resnet34` (`resnet34.tv_in1k`, the 2018 recipe) at a fixed seed with every BatchNorm's affine and running statistics randomised, and writes, in the JAX
reference's `params[k][j]` order:

  * the parameters: (conv W, BN γ, BN β) per conv in forward order, each block's downsample last,
    then the classifier (W [out, in], b);
  * the running statistics (μ, σ²) per BN in the same order;
  * logits in TRAIN mode (batch statistics) and EVAL mode (running statistics) at each
    `--train-res` / `--eval-res` (R50: train 224 and RSB-A3's 160, eval 224 and A1/A2's 288);
  * the per-block drop-path probabilities timm assigns at `--drop-path` (RSB's 0.05; R34 has none).

The (conv, BN) pairing is asserted, not assumed.

Usage (called by the parity script):
    .venv-timm/bin/python scripts/parity/_resnet_timm_dump.py OUT.npz --model resnet50 --batch 4 --seed 0
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
    ap.add_argument("--model", default="resnet50", choices=["resnet50", "resnet34"])
    ap.add_argument("--drop-path", type=float, default=0.05)
    ap.add_argument("--train-res", default="224,160")
    ap.add_argument("--eval-res", default="224,288")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    m = timm.create_model(a.model, num_classes=a.classes, drop_path_rate=a.drop_path)
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
        runs = [("train", int(r)) for r in a.train_res.split(",")] + \
               [("eval", int(r)) for r in a.eval_res.split(",")]
        for mode, res in runs:
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
    print(f"timm {a.model}: {len(params)} param groups, {len(stats)} BN, "
          f"{sum(p.numel() for p in m.parameters())} params, drop-path {drop[0]:.4f}…{drop[-1]:.4f}")


if __name__ == "__main__":
    main()

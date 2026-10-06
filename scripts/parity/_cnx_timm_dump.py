#!/usr/bin/env python3
"""timm half of `scripts/parity/cnx_timm_parity.py` — runs in `.venv-timm` (CPU torch, timm 1.0.28).

Builds `convnext_{tiny,small,base}` at a fixed seed, randomises every LayerNorm's affine and every
block's LayerScale γ (timm initialises γ at 1e-6, which would make every block nearly the identity
and the check nearly blind), and writes, in the JAX reference's `params[k][j]` order:

    stem conv (W, b), stem LN (γ, β),
    per stage: [downsample LN (γ, β), downsample conv (W, b)] (stages 1–3), then per block
      dw conv (W, b), LN (γ, β), fc1 (W as [out, in, 1, 1], b), fc2 (same), LayerScale (γ,)
    head LN (γ, β), fc (W [out, in], b)

with logits at 224 and 288 (eval mode: ConvNeXt has no BatchNorm, so there is one forward) and the
per-block drop-path probabilities timm assigns at the paper's `drop_path_rate` for the size.

`--gelu erf` is timm's own GELU and the reference's; `--gelu tanh` builds timm with the tanh
approximation (the parity gate's control).

Usage (called by the parity script):
    .venv-timm/bin/python scripts/parity/_cnx_timm_dump.py OUT.npz --model convnext_small --batch 2
"""
import argparse
from functools import partial
import numpy as np
import timm
import torch
import torch.nn as nn

DROP_PATH = {"convnext_tiny": 0.1, "convnext_small": 0.4, "convnext_base": 0.5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--model", default="convnext_tiny", choices=sorted(DROP_PATH))
    ap.add_argument("--classes", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gelu", default="tanh", choices=["tanh", "erf"])
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    kw = {"act_layer": partial(nn.GELU, approximate="tanh")} if a.gelu == "tanh" else {}
    m = timm.create_model(a.model, num_classes=a.classes, drop_path_rate=DROP_PATH[a.model], **kw)
    g = torch.Generator().manual_seed(a.seed + 1)
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod, nn.LayerNorm):
                c = mod.normalized_shape[0]
                assert mod.eps == 1e-6, f"LN eps {mod.eps}"
                mod.weight.copy_(0.5 + torch.rand(c, generator=g))
                mod.bias.copy_(0.2 * torch.randn(c, generator=g))
        blocks = [b for s in m.stages for b in s.blocks]
        for b in blocks:
            b.gamma.copy_(0.5 + torch.rand(b.gamma.shape, generator=g))

    t = lambda p: p.detach().numpy()
    fc = lambda lin: [t(lin.weight)[:, :, None, None], t(lin.bias)]
    params = [[t(m.stem[0].weight), t(m.stem[0].bias)], [t(m.stem[1].weight), t(m.stem[1].bias)]]
    for i, s in enumerate(m.stages):
        if i > 0:
            ln, conv = s.downsample[0], s.downsample[1]
            params += [[t(ln.weight), t(ln.bias)], [t(conv.weight), t(conv.bias)]]
        for b in s.blocks:
            params += [[t(b.conv_dw.weight), t(b.conv_dw.bias)], [t(b.norm.weight), t(b.norm.bias)],
                       fc(b.mlp.fc1), fc(b.mlp.fc2), [t(b.gamma)]]
    params += [[t(m.head.norm.weight), t(m.head.norm.bias)], [t(m.head.fc.weight), t(m.head.fc.bias)]]
    assert sum(sum(p.size for p in grp) for grp in params) == sum(p.numel() for p in m.parameters())

    drop = [float(getattr(b.drop_path, "drop_prob", 0.0)) for b in blocks]
    out = {"n_params": np.array(len(params)), "drop_probs": np.array(drop)}
    m.eval()
    with torch.no_grad():
        for res in (224, 288):
            x = torch.randn(a.batch, 3, res, res, generator=g)
            out[f"x{res}"] = x.numpy()
            out[f"y{res}"] = m(x).numpy()
    for k, group in enumerate(params):
        out[f"n{k}"] = np.array(len(group))
        for j, arr in enumerate(group):
            out[f"p{k}_{j}"] = arr
    np.savez(a.out, **out)
    print(f"timm {a.model} (GELU {a.gelu}): {len(params)} param groups, "
          f"{sum(p.numel() for p in m.parameters())} params, drop-path {drop[0]:.4f}…{drop[-1]:.4f}")


if __name__ == "__main__":
    main()

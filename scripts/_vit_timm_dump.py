#!/usr/bin/env python3
"""timm half of `scripts/vit_timm_parity.py` — runs in `.venv-timm` (CPU torch, timm 1.0.28).

Builds `deit_tiny_patch16_224` at a fixed seed with drop/drop-path 0, randomises every LayerNorm's
affine and the cls / position embeddings (so none sits at an init value that hides a slot mix-up),
runs one forward, and writes the parameters in the JAX reference's `params[k][j]` order:

    patch (W [D,3,16,16], b) · cls (1,1,D) · pos (1,197,D)
    per block: ln1 (γ,β) · q (W,b) · k (W,b) · v (W,b) · proj (W,b) · ln2 (γ,β) · fc1 (W,b) · fc2 (W,b)
    final ln (γ,β) · head (W [K,D], b)

timm fuses q/k/v into one `qkv` Linear; its rows are [q; k; v], split here.

`--gelu tanh|erf` and `--ln-eps` pick the activation and LayerNorm ε: DeiT's own net is erf and
1e-6; the JAX reference (and the verified render) computes tanh and 1e-5 today.
`--swap-kv` swaps k and v in the dump — a CONTROL the parity script must see red.
"""
import argparse
from functools import partial
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
    ap.add_argument("--gelu", default="tanh", choices=["tanh", "erf"])
    ap.add_argument("--ln-eps", type=float, default=1e-5)
    ap.add_argument("--swap-kv", action="store_true")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    act = partial(nn.GELU, approximate="tanh") if a.gelu == "tanh" else nn.GELU
    m = timm.create_model("deit_tiny_patch16_224", num_classes=a.classes, drop_rate=0.0,
                          drop_path_rate=0.0, act_layer=act,
                          norm_layer=partial(nn.LayerNorm, eps=a.ln_eps))
    g = torch.Generator().manual_seed(a.seed + 1)
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod, nn.LayerNorm):
                c = mod.normalized_shape[0]
                mod.weight.copy_(0.5 + torch.rand(c, generator=g))
                mod.bias.copy_(0.2 * torch.randn(c, generator=g))
        m.cls_token.copy_(0.5 * torch.randn(m.cls_token.shape, generator=g))
        m.pos_embed.copy_(0.5 * torch.randn(m.pos_embed.shape, generator=g))
    m.eval()

    t = lambda p: p.detach().numpy()
    params = [[t(m.patch_embed.proj.weight), t(m.patch_embed.proj.bias)],
              [t(m.cls_token)], [t(m.pos_embed)]]
    D = m.embed_dim
    for blk in m.blocks:
        W, b = t(blk.attn.qkv.weight), t(blk.attn.qkv.bias)
        q, k, v = (W[i * D:(i + 1) * D] for i in range(3))
        qb, kb, vb = (b[i * D:(i + 1) * D] for i in range(3))
        if a.swap_kv:
            k, v, kb, vb = v, k, vb, kb
        params += [[t(blk.norm1.weight), t(blk.norm1.bias)], [q, qb], [k, kb], [v, vb],
                   [t(blk.attn.proj.weight), t(blk.attn.proj.bias)],
                   [t(blk.norm2.weight), t(blk.norm2.bias)],
                   [t(blk.mlp.fc1.weight), t(blk.mlp.fc1.bias)],
                   [t(blk.mlp.fc2.weight), t(blk.mlp.fc2.bias)]]
    params += [[t(m.norm.weight), t(m.norm.bias)], [t(m.head.weight), t(m.head.bias)]]

    x = torch.randn(a.batch, 3, 224, 224, generator=g)
    with torch.no_grad():
        y = m(x).numpy()
    out = {"x": x.numpy(), "y": y, "n_params": np.array(len(params))}
    for k_, group in enumerate(params):
        out[f"n{k_}"] = np.array(len(group))
        for j, arr in enumerate(group):
            out[f"p{k_}_{j}"] = arr
    np.savez(a.out, **out)
    print(f"timm deit_tiny_patch16_224 (GELU {a.gelu}, LN ε {a.ln_eps:g}{', k/v SWAPPED' if a.swap_kv else ''}): "
          f"{len(params)} param groups, {sum(p.numel() for p in m.parameters())} params")


if __name__ == "__main__":
    main()

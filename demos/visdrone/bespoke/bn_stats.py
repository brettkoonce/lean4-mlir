"""Load a Lean `*_bn_stats.bin` into the twin's BatchNorm running buffers.

Layout is `NetSpec.evalShapes` (LeanMlir/SpecHelpers.lean:195): after the params,
per BN layer, `[oc]` running mean then `[oc]` running var -- interleaved per
layer, NOT all-means-then-all-vars.

BN layer order is `collectBnLayers` (MlirCodegen.lean:2883): per residual block
conv1 then conv2, and for block 0 of a projecting stage the projection LAST.
torchvision's BasicBlock declares conv1, bn1, relu, conv2, bn2, downsample, so
`named_modules()` yields bn1, bn2, downsample.1 -- the same order.

Without these, twin eval mode runs on torchvision's untrained (0, 1) buffers,
which is not a comparison against Lean, it is a comparison against noise.
"""
import numpy as np
import torch
import torch.nn as nn


def lean_ordered_bns(model):
    """BatchNorm2d modules in collectBnLayers order."""
    out = []
    for mod_name, mod in (("stem", model.stem), ("layer2", model.layer2),
                          ("layer3", model.layer3), ("layer4", model.layer4)):
        out.extend((f"{mod_name}.{n}", m) for n, m in mod.named_modules()
                   if isinstance(m, nn.BatchNorm2d))
    return out


def load_bn_stats(model, path, verbose=True):
    flat = np.fromfile(path, dtype=np.float32)
    bns = lean_ordered_bns(model)
    want = sum(2 * m.num_features for _, m in bns)
    if flat.size != want:
        raise ValueError(f"{path}: {flat.size} floats, {len(bns)} BN layers want {want}")

    off = 0
    with torch.no_grad():
        for _, m in bns:
            c = m.num_features
            m.running_mean.copy_(torch.from_numpy(flat[off:off + c].copy()))
            off += c
            m.running_var.copy_(torch.from_numpy(flat[off:off + c].copy()))
            off += c
    assert off == flat.size
    if verbose:
        v = torch.cat([m.running_var.flatten() for _, m in bns])
        print(f"loaded {path}: {len(bns)} BN layers, {flat.size:,} floats "
              f"(var min {float(v.min()):.4g} max {float(v.max()):.4g})")
    return model

"""Load a Lean flat `*_params.bin` checkpoint into the PyTorch twin.

This turns the twin into a REFERENCE ORACLE for the Lean stack: with identical
weights on an identical batch, any disagreement localizes to a specific emitter.

  forward output disagrees  -> the emitted FORWARD is wrong
  forward agrees, grads no  -> the emitted BACKWARD is wrong
  both agree, params after a step disagree -> the Adam/update path is wrong

That distinction is exactly what FD probes cannot make: they check
d(emitted loss)/dtheta against the SAME emitted forward, so a wrong forward
passes every FD probe, iree-compile, and the data round-trip simultaneously.

Layout comes from `NetSpec.paramShapes` (LeanMlir/SpecHelpers.lean:23), replicated
below for the r34FpnDet spec. Lean emits, per conv: [oc,ic,k,k] then BN gamma [oc]
then BN beta [oc]; per residual block: conv1,bn1,conv2,bn2 and -- for block 0 of a
projecting stage -- the 1x1 projection LAST. torchvision's BasicBlock declares
conv1,bn1,conv2,bn2,downsample in that same order, so the two sequences zip
directly once shapes are checked.

BN running mean/var live in a separate `*_bn_stats.bin` and are NOT needed to
reproduce a TRAINING loss, which uses batch statistics.
"""
import numpy as np
import torch

BACKBONE_FLOATS = 21_284_672


def r34_fpn_param_shapes(oc=256, c3=128, c4=256, c5=512, A=3, tower=0):
    """Replicates paramShapes for the r34FpnDet layer list."""
    sh = []

    def conv_bn(ic, o, k):
        sh.extend([(o, ic, k, k), (o,), (o,)])

    def residual(ic, o, n, first_stride):
        needs_proj = not (ic == o and first_stride == 1)
        for bi in range(n):
            bic = ic if bi == 0 else o
            conv_bn(bic, o, 3)
            conv_bn(o, o, 3)
            if bi == 0 and needs_proj:
                conv_bn(ic, o, 1)

    conv_bn(3, 64, 7)                 # .convBn 3 64 7 2   (maxPool has no params)
    residual(64, 64, 3, 1)            # stride 4
    residual(64, 128, 4, 2)           # C3
    residual(128, 256, 6, 2)          # C4
    residual(256, 512, 3, 2)          # C5
    backbone_n = sum(int(np.prod(s)) for s in sh)
    assert backbone_n == BACKBONE_FLOATS, f"backbone {backbone_n} != {BACKBONE_FLOATS}"

    # .fpnDetect: Wn3,Wn4,Wn5, Wh3,Wh4,Wh5, then head biases LAST
    head = [(oc, c3), (oc, c4), (oc, c5)]
    assert tower == 0, "tower>0 param order not transcribed"
    ap = A * 15
    head += [(ap, oc)] * 3
    head += [(ap,)] * 3
    sh.extend(head)
    return sh


def torch_backbone_params(model):
    """Backbone params in Lean's emission order."""
    out = []
    for mod in (model.stem, model.layer2, model.layer3, model.layer4):
        out.extend(p for _, p in mod.named_parameters())
    return out


def load_lean_params(model, path, strict=True, verbose=True):
    flat = np.fromfile(path, dtype=np.float32)
    shapes = r34_fpn_param_shapes()
    total = sum(int(np.prod(s)) for s in shapes)
    if flat.size != total:
        raise ValueError(f"{path}: {flat.size:,} floats, spec wants {total:,}")

    tparams = torch_backbone_params(model)
    head = [model.lat3.weight, model.lat4.weight, model.lat5.weight,
            model.heads[0].weight, model.heads[1].weight, model.heads[2].weight,
            model.heads[0].bias, model.heads[1].bias, model.heads[2].bias]
    # Lean neck order is Wn3,Wn4,Wn5 = C3,C4,C5 laterals -> lat3,lat4,lat5
    targets = tparams + head
    if len(targets) != len(shapes):
        raise ValueError(f"{len(targets)} torch tensors vs {len(shapes)} lean shapes")

    off, nmis = 0, 0
    with torch.no_grad():
        for i, (p, s) in enumerate(zip(targets, shapes)):
            n = int(np.prod(s))
            chunk = flat[off:off + n]
            off += n
            want = tuple(p.shape)
            # Lean stores 1x1 neck/head weights as [oc, ic]; torch wants [oc,ic,1,1]
            arr = chunk.reshape(s) if len(s) > 1 else chunk
            if arr.shape != want:
                if int(np.prod(arr.shape)) == int(np.prod(want)):
                    arr = arr.reshape(want)
                else:
                    nmis += 1
                    if strict:
                        raise ValueError(
                            f"param {i}: lean {s} vs torch {want}")
                    continue
            p.copy_(torch.from_numpy(np.ascontiguousarray(arr)))
    assert off == total
    if verbose:
        print(f"loaded {path}: {len(targets)} tensors, {total:,} floats, "
              f"{nmis} shape mismatches")
    return model

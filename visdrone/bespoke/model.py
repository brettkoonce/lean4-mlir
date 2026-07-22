"""PyTorch twin of the Lean FPN detector (`Layer.fpnDetect`).

Matches `demos/MainYolov1VisdroneFpn.lean` / `emitFpnNeckForward` +
`emitFpnDetectForward` exactly at the default settings:

  backbone  ResNet-34, C3/C4/C5 = layer2/layer3/layer4 = 128/256/512 ch at 56/28/14
  neck      P5 = conv1x1(C5)          [no bias, no norm, no activation]
            P4 = conv1x1(C4) + up2(P5)
            P3 = conv1x1(C3) + up2(P4)
  head      conv1x1(Pn, oc -> A*15) + bias   [no norm, no activation at tower=0]
  output    per level [B, A*15, g, g] -> C-order flatten -> concat [P3|P4|P5]

`norm` and `tower` exist so the twin can A/B the fixes the Lean side cannot
cheaply try. `norm=None, tower=0` is the faithful twin; anything else is an
experiment, not the twin.

The Lean neck has NO bias on the laterals (6 weights Wn3/4/5 + Wh3/4/5, then 3
head biases = 264,071 params at oc=256) and the head bias is initialized to the
RetinaNet prior on the objectness channels only.
"""
import math

import torch
import torch.nn as nn
import torchvision

P = 15          # slots per anchor: 4 box + 1 obj + 10 class
NC = 10
IMG = 448

# (grid, anchors) in the codegen's concat order. MUST stay [P3|P4|P5].
FPN_SCALES = [
    (56, [(0.006935, 0.014941), (0.015750, 0.028005), (0.033728, 0.035028)]),
    (28, [(0.023961, 0.070528), (0.055662, 0.068706), (0.093187, 0.094324)]),
    (14, [(0.060280, 0.168604), (0.107559, 0.204684), (0.181239, 0.149031)]),
]
NTOT = sum(len(a) * P * g * g for g, a in FPN_SCALES)   # 185220

BACKBONES = {
    # name: (torchvision ctor, (C3, C4, C5) channels)
    "r34": (torchvision.models.resnet34, (128, 256, 512)),
    "r50": (torchvision.models.resnet50, (512, 1024, 2048)),
}


def _norm(kind, ch):
    if kind is None:
        return nn.Identity()
    if kind == "bn":
        return nn.BatchNorm2d(ch)
    if kind == "gn":
        return nn.GroupNorm(32, ch)
    raise ValueError(f"unknown norm {kind!r}")


class Tower(nn.Module):
    """RetinaNet-style head tower: `depth` x (3x3 conv [+ norm] + ReLU).

    The Lean T2a tower is plain conv+bias+ReLU with NO norm; pass norm=None to
    reproduce it.
    """

    def __init__(self, oc, depth, norm=None):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers += [nn.Conv2d(oc, oc, 3, padding=1, bias=True),
                       _norm(norm, oc), nn.ReLU(inplace=True)]
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return self.body(x)


def _pre_pad(pad):
    """forward_pre_hook that zero-pads the input, so the conv itself can run
    with padding=0. A hook rather than a module swap: wrapping the conv would
    shift its position in `named_parameters()` and silently break the flat
    checkpoint ordering that `lean_ckpt.load_lean_params` depends on."""
    def hook(_module, inputs):
        return (nn.functional.pad(inputs[0], pad),) + tuple(inputs[1:])
    return hook


def apply_lean_padding(model):
    """Switch the stride-2 convs to Lean's ASYMMETRIC SAME padding.

    `MlirCodegen.samePad` is TensorFlow-style SAME: total pad t is split
    (t/2, t-t/2), i.e. the extra pixel goes on the HIGH side. torchvision
    instead uses a symmetric `padding=` on every conv. Both yield identical
    output sizes, so nothing about shapes, parameter counts, or `iree-compile`
    can catch the difference -- but the sampling grid is offset by one pixel,
    and the offset compounds through all four downsampling stages.

        stem conv1  448 -> 224, k=7 s=2:  lean (2,3)  vs torchvision (3,3)
        layer{2,3,4}[0].conv1, k=3 s=2:   lean (0,1)  vs torchvision (1,1)

    Every stride-1 3x3 conv works out to (1,1) both ways, and the 1x1
    stride-2 downsample convs to (0,0) both ways, so those need no change.
    """
    conv1 = model.stem[0]
    assert conv1.kernel_size == (7, 7) and conv1.stride == (2, 2), \
        f"stem conv is {conv1.kernel_size}/{conv1.stride}, not the 7x7 s2 expected"
    conv1.padding = (0, 0)
    conv1.register_forward_pre_hook(_pre_pad((2, 3, 2, 3)))   # (W_lo,W_hi,H_lo,H_hi)

    for name, layer in (("layer2", model.layer2), ("layer3", model.layer3),
                        ("layer4", model.layer4)):
        c = layer[0].conv1
        assert c.kernel_size == (3, 3) and c.stride == (2, 2), \
            f"{name}[0].conv1 is {c.kernel_size}/{c.stride}, not the 3x3 s2 expected"
        c.padding = (0, 0)
        c.register_forward_pre_hook(_pre_pad((0, 1, 0, 1)))
    return model


class FpnDetector(nn.Module):
    def __init__(self, backbone="r34", oc=256, tower=0, norm=None,
                 pretrained=True, prior_pi=0.01, scales=FPN_SCALES, pool="torchvision",
                 pad="torchvision"):
        super().__init__()
        ctor, (c3, c4, c5) = BACKBONES[backbone]
        net = ctor(weights="DEFAULT" if pretrained else None)
        # The Lean spec is `.maxPool 2 2` = kernel 2, stride 2, no padding;
        # torchvision's ResNet stem is kernel 3, stride 2, padding 1. Both emit
        # 112x112 from 224x224 but they are NOT the same function, so a forward
        # diff against a Lean checkpoint MUST use pool="lean".
        if pool == "lean":
            net.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool != "torchvision":
            raise ValueError(f"unknown pool {pool!r}")
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1)
        self.layer2, self.layer3, self.layer4 = net.layer2, net.layer3, net.layer4

        self.scales = scales
        self.A = [len(a) for _, a in scales]
        assert len(set(self.A)) == 1, "codegen assumes one A across levels"
        ap = self.A[0] * P

        # neck laterals: 1x1, NO bias (Lean has 6 weights, no neck bias)
        self.lat3 = nn.Conv2d(c3, oc, 1, bias=False)
        self.lat4 = nn.Conv2d(c4, oc, 1, bias=False)
        self.lat5 = nn.Conv2d(c5, oc, 1, bias=False)
        self.neck_norm = nn.ModuleList([_norm(norm, oc) for _ in range(3)])

        self.towers = nn.ModuleList([Tower(oc, tower, norm) for _ in range(3)])
        self.heads = nn.ModuleList([nn.Conv2d(oc, ap, 1, bias=True) for _ in range(3)])

        if pad == "lean":
            apply_lean_padding(self)
        elif pad != "torchvision":
            raise ValueError(f"unknown pad {pad!r}")

        # RetinaNet prior on objectness channels only (SpecHelpers.applyDetPriorBias)
        if prior_pi and prior_pi > 0:
            b0 = -math.log((1.0 - prior_pi) / prior_pi)
            for h in self.heads:
                nn.init.zeros_(h.bias)
                with torch.no_grad():
                    h.bias[4::P] = b0

    def forward(self, x):
        x = self.stem(x)
        c3 = self.layer2(x)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        p5 = self.lat5(c5)
        p4 = self.lat4(c4) + nn.functional.interpolate(
            p5, scale_factor=2, mode="bilinear", align_corners=False)
        p3 = self.lat3(c3) + nn.functional.interpolate(
            p4, scale_factor=2, mode="bilinear", align_corners=False)

        outs = []
        for i, p in enumerate((p3, p4, p5)):
            p = self.neck_norm[i](p)
            z = self.heads[i](self.towers[i](p))       # [B, A*P, g, g]
            outs.append(z.flatten(1))                  # C-order, matches codegen
        return torch.cat(outs, dim=1)                  # [B, NTOT]

    def param_groups(self, weight_decay):
        """Lean keys weight decay on RANK (`wsh.length > 1`), so all biases and
        all norm params are exempt."""
        decay, no_decay = [], []
        for _, prm in self.named_parameters():
            if not prm.requires_grad:
                continue
            (decay if prm.dim() > 1 else no_decay).append(prm)
        return [{"params": decay, "weight_decay": weight_decay},
                {"params": no_decay, "weight_decay": 0.0}]

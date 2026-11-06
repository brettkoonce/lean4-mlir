#!/usr/bin/env python3
"""The 80-epoch imagenette-iree ETA from the measured ms/step probe (runs/2026-11-05-iree-cuda-smoke/
probe/probe.tsv): 80 epochs x 296 steps/epoch at that rate, plus a 14% per-epoch-eval share (the
Arc record's measured training→full-run ratio). Four cards, longest net to its own card, the three
shortest trailing the long ones; ViT alone is the critical path."""
import csv, pathlib
P = pathlib.Path('runs/2026-11-05-iree-cuda-smoke/probe/probe.tsv')
ms = {}
for r in csv.DictReader((l for l in P.read_text().splitlines() if not l.startswith('#')), delimiter='\t'):
    ms[r['net']] = int(r['median_ms'])
STEPS = 296 * 80
def h(net): return ms[net] * STEPS / 3.6e6
NAMES = {'vit-verified-adam':'ViT-Tiny','resnet50-verified-adam':'ResNet-50','resnet34-verified-adam':'ResNet-34',
         'convnext-verified-adam':'ConvNeXt-T','mobilenetv4-verified-adam':'MobileNetV4','efficientnet-verified-adam':'EfficientNet-B0',
         'mobilenetv2-verified-adam':'MobileNetV2'}
print(f"{'net':16} {'ms/step':>8} {'train h':>8} {'+eval h':>8}")
for n in sorted(ms, key=lambda n: -ms[n]):
    print(f"{NAMES[n]:16} {ms[n]:>8} {h(n):>8.1f} {h(n)*1.14:>8.1f}")
# card schedules (train+eval), cifar tail ~5h on cards 2/3
cards = {
 'card0 (gpu0)': [('ViT-Tiny', h('vit-verified-adam'))],
 'card1 (gpu1)': [('ResNet-50', h('resnet50-verified-adam')), ('MobileNetV2', h('mobilenetv2-verified-adam'))],
 'card2 (gpu2)': [('cifar-iree tail', 5.0), ('ResNet-34', h('resnet34-verified-adam')), ('EfficientNet-B0', h('efficientnet-verified-adam'))],
 'card3 (gpu3)': [('cifar-iree tail', 5.0), ('ConvNeXt-T', h('convnext-verified-adam')), ('MobileNetV4', h('mobilenetv4-verified-adam'))],
}
print("\ncard schedule (hours, +eval where training):")
crit = 0
for c, jobs in cards.items():
    tot = sum((t*1.14 if t>5.1 else t) for _, t in jobs)
    crit = max(crit, tot)
    print(f"  {c}: " + " → ".join(f"{n} {t*1.14 if t>5.1 else t:.0f}h" for n,t in jobs) + f"  = {tot:.0f}h")
print(f"\ncritical path ≈ {crit:.0f} h ≈ {crit/24:.1f} days (ViT on card0 is {h('vit-verified-adam')*1.14:.0f}h alone)")

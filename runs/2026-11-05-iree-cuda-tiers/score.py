#!/usr/bin/env python3
"""Score the three `-iree` tiers on CUDA against the XLA/CUDA references, per the bar in
planning/iree_platform.md §1: MNIST finals against the 30-seed medians (runs/2026-09-01-mnist-n30),
CIFAR's arms against the five-seed range per arm (runs/2026-09-01-cifar8w-6arm-constlr), Imagenette's
80-epoch top-1 against the five-seed range per net (runs/2026-08-31-imagenette-n3; ConvNeXt-T's seeds
predate the LayerScale-1e-6 recipe and MobileNetV4's the timm rebuild, so those two are compared to
the single post-change CUDA runs the Arc record names). Writes <tier dir>/results.tsv for each tier
that has logs. Run from the repo root: python3 runs/2026-11-05-iree-cuda-tiers/score.py"""
import re, glob, statistics as st, pathlib

def finals(path, key):
    """Final accuracy per arm: the last `key = n/N = x%` before each ════ banner, and at the end."""
    v = []; cur = None
    for line in open(path, errors='replace'):
        if '════' in line or line.startswith('━━━'):
            if cur is not None: v.append(cur); cur = None
            continue
        m = re.search(key + r' = (\d+)/(\d+) = ([0-9.]+)%', line)
        if m: cur = (float(m.group(3)), int(m.group(1)), int(m.group(2)))
    if cur is not None: v.append(cur)
    return v

def wall(tier, n):
    p = pathlib.Path(f'runs/2026-11-05-{tier}-iree-cuda/{n}.time')
    m = re.search(r'wall ([0-9.]+) s', p.read_text()) if p.exists() else None
    return float(m.group(1)) if m else None

def fmt_w(s): return '' if s is None else (f'{s/60:.1f} min' if s < 3600 else f'{s/3600:.2f} h')

def verdict(x, lo, hi, n_images):
    if lo <= x <= hi: return 'inside'
    edge = lo if x < lo else hi
    imgs = abs(x - edge) / 100 * n_images
    return f'{"under" if x < lo else "over"} by {abs(x-edge):.2f} pt = {imgs:.1f} images'

out = []
# ── mnist ──
D = pathlib.Path('runs/2026-11-05-mnist-iree-cuda')
if D.exists():
    rows = ['# `lake run mnist-iree`\'s three binaries, IREE 3.12.0 on one RTX 4060 Ti (cuda, sm_86 PTX), ares, 2026-11-05: 12 epochs each, the final test accuracy against XLA/CUDA\'s 30-seed median and range (runs/2026-09-01-mnist-n30).',
            'net\ttop1\txla_median_30\txla_range_30\tverdict\twall']
    for n, ref in [('mnist-linear-verified', 'linear'), ('mnist-mlp-verified', 'mlp'), ('mnist-cnn-verified', 'cnn')]:
        L = D / f'{n}.log'
        if not L.exists(): continue
        f = finals(L, 'test_acc')
        xs = [finals(p, 'test_acc')[-1][0] for p in sorted(glob.glob(f'runs/2026-09-01-mnist-n30/{ref}_s*.log')) if finals(p, 'test_acc')]
        if f:
            x = f[-1][0]
            rows.append(f'{n}\t{x:.2f} %\t{st.median(xs):.2f} %\t{min(xs):.2f}–{max(xs):.2f}\t{verdict(x, min(xs), max(xs), 10000)}\t{fmt_w(wall("mnist", n))}')
        else: rows.append(f'{n}\tno final\t\t\t\t{fmt_w(wall("mnist", n))}')
    (D / 'results.tsv').write_text('\n'.join(rows) + '\n'); out.append(D / 'results.tsv')
# ── cifar ──
D = pathlib.Path('runs/2026-11-05-cifar-iree-cuda')
if D.exists():
    rows = ['# `lake run cifar-iree`\'s three binaries, IREE 3.12.0 on one RTX 4060 Ti each (cuda, sm_86 PTX), ares, 2026-11-05: 40 epochs per arm, the final test accuracy against XLA/CUDA\'s five seeds per arm (runs/2026-09-01-cifar8w-6arm-constlr: nobn_s*, bn_s*, bf16_s* = the bbn net\'s bf16 arms; its f32 arms have no five-seed set and are compared to bn_s*).',
            'net\tarm\ttop1\txla_range_5\tverdict\twall']
    ARMS = {'cifar8w-ablation': ('nobn', ['SGD', 'Nesterov', 'AdamW']),
            'cifar8w-bn-ablation': ('bn', ['SGD', 'Nesterov', 'AdamW']),
            'cifar8wb-bbn-ablation': ('bn', ['f32 SGD', 'f32 Nesterov', 'f32 AdamW', 'bf16 SGD', 'bf16 Nesterov', 'bf16 AdamW'])}
    for n, (ref, arms) in ARMS.items():
        L = D / f'{n}.log'
        if not L.exists(): continue
        f = finals(L, 'test_acc')
        for i, a in enumerate(arms):
            r = ref if i < 3 else 'bf16'
            xs = [finals(p, 'test_acc') for p in sorted(glob.glob(f'runs/2026-09-01-cifar8w-6arm-constlr/{r}_s*.log'))]
            xs = [v[i % 3][0] for v in xs if len(v) > i % 3]
            if i < len(f):
                x = f[i][0]
                rows.append(f'{n}\t{a}\t{x:.2f} %\t{min(xs):.2f}–{max(xs):.2f}\t{verdict(x, min(xs), max(xs), 10000)}\t{fmt_w(wall("cifar", n)) if i == 0 else ""}')
            else: rows.append(f'{n}\t{a}\tno final\t{min(xs):.2f}–{max(xs):.2f}\t\t')
    (D / 'results.tsv').write_text('\n'.join(rows) + '\n'); out.append(D / 'results.tsv')
# ── imagenette ──
D = pathlib.Path('runs/2026-11-05-imagenette-iree-cuda')
if D.exists():
    rows = ['# `lake run imagenette-iree`\'s seven binaries, IREE 3.12.0 on one RTX 4060 Ti each (cuda, sm_86 PTX), ares, 2026-11-05/06: 80 epochs, the final top-1 against XLA/CUDA\'s five seeds per net (runs/2026-08-31-imagenette-n3), except ConvNeXt-T (LayerScale 1e-6 recipe: the one CUDA run, runs/2026-09-13-convnext-imagenette-ls1e-6-resident, 82.27 %) and MobileNetV4 (timm conv_medium rebuild of 2026-09-25: no post-rebuild 80-epoch CUDA run on record; the pre-rebuild seeds are shown).',
            'net\ttop1\ttop5\txla_ref\tverdict\twall\tepochs_done']
    SPECIAL = {'convnext-verified-adam': (82.27, 82.27, 'one run'), }
    for n, ref in [('resnet34-verified-adam', 'resnet34'), ('resnet50-verified-adam', 'resnet50'), ('mobilenetv2-verified-adam', 'mobilenetv2'),
                   ('mobilenetv4-verified-adam', 'mobilenetv4'), ('efficientnet-verified-adam', 'efficientnet'), ('convnext-verified-adam', 'convnext'), ('vit-verified-adam', 'vit')]:
        L = D / f'{n}.log'
        if not L.exists(): continue
        txt = L.read_text(errors='replace')
        ep = re.findall(r'epoch (\d+): val_acc = (\d+)/(\d+) = ([0-9.]+)%\s+top5 = (\d+)/(\d+) = ([0-9.]+)%', txt)
        xs = [finals(p, 'val_acc')[-1][0] for p in sorted(glob.glob(f'runs/2026-08-31-imagenette-n3/{ref}_s*.log')) if finals(p, 'val_acc')]
        lo, hi, tag = SPECIAL.get(n, (min(xs), max(xs), '5 seeds'))
        if ep:
            e, c, N, x, c5, _, x5 = ep[-1]
            v = verdict(float(x), lo, hi, int(N)) + (' (pre-rebuild seeds)' if n == 'mobilenetv4-verified-adam' else '')
            rows.append(f'{n}\t{float(x):.2f} %\t{float(x5):.2f} %\t{lo:.2f}–{hi:.2f} ({tag})\t{v}\t{fmt_w(wall("imagenette", n))}\t{e}/80')
        else: rows.append(f'{n}\tno eval yet\t\t{lo:.2f}–{hi:.2f} ({tag})\t\t\t0/80')
    (D / 'results.tsv').write_text('\n'.join(rows) + '\n'); out.append(D / 'results.tsv')
for p in out: print(f'── {p}'); print(p.read_text())

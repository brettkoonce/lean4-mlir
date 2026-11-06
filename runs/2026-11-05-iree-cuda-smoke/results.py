#!/usr/bin/env python3
"""results.tsv for the 2026-11-05 IREE-on-CUDA smoke: the 10-28 CPU table's columns (status, wall,
max RSS, step-0 loss, epoch-1 mean loss, eval after the ten steps) from each <net>.log and
<net>.time, plus the median ms/step from probe/probe.tsv and the 80-epoch ETA it implies.
Run from the repo root: python3 runs/2026-11-05-iree-cuda-smoke/results.py"""
import re, pathlib, csv
D = pathlib.Path('runs/2026-11-05-iree-cuda-smoke')
NETS = ['mnist-linear-verified', 'mnist-mlp-verified', 'mnist-cnn-verified',
        'cifar8w-ablation', 'cifar8w-bn-ablation', 'cifar8wb-bbn-ablation',
        'resnet34-verified-adam', 'resnet50-verified-adam', 'mobilenetv2-verified-adam',
        'mobilenetv4-verified-adam', 'efficientnet-verified-adam', 'convnext-verified-adam',
        'vit-verified-adam']
probe = {}
pt = D / 'probe' / 'probe.tsv'
if pt.exists():
    for row in csv.DictReader((l for l in pt.read_text().splitlines() if not l.startswith('#')), delimiter='\t'):
        probe[row['net']] = row
rows = []
for n in NETS:
    log = D / f'{n}.log'; tm = D / f'{n}.time'
    if not log.exists(): rows.append((n, 'MISSING', '', '', '', '', '', '', '')); continue
    txt = log.read_text(errors='replace')
    done = 'done (' in txt or 'test_acc' in txt
    m = re.search(r'(\w+) wall ([0-9.]+) s, maxrss (\d+) KB', tm.read_text()) if tm.exists() else None
    wall = f'{float(m.group(2)):.1f}' if m else ''; rss = f'{int(m.group(3))/1e6:.1f}' if m else ''
    s0 = re.findall(r'step 0/\d+: loss=([0-9.]+)', txt)
    ep = re.findall(r'Epoch 1/\d+: loss=([0-9.]+)', txt)
    ev = re.findall(r'epoch 1: (?:loss = [0-9.]+, )?(?:val|test)_acc = (\d+/\d+ = [0-9.]+)%', txt)
    arms = len(re.findall(r'^════', txt, re.M))
    note = f'{arms} arms, one row each: ' + ' / '.join(x.split(' = ')[1] + '%' for x in ev) if arms > 1 else ''
    ms = probe.get(n, {}).get('median_ms', ''); eta = probe.get(n, {}).get('train_80ep_min', '')
    rows.append((n, 'PASS' if done else 'FAIL', wall, rss, s0[0] if s0 else '', ep[0] if ep else '',
                 (ev[0] + '%') if ev else '', ms, eta, note))
hdr = ('# The thirteen `-iree` twin binaries (lake run mnist-iree / cifar-iree / imagenette-iree) under '
       'LEAN_MLIR_LOWERER=iree IREE_BACKEND=cuda (sm_86 PTX), IREE 3.12.0, ares (one RTX 4060 Ti each), 2026-11-05.\n'
       '# Each run: iree-compile of the train step(s) and the forwards, LEAN_MLIR_MAX_EPOCHS=1 (and LEAN_MLIR_G2_STEPS=10 '
       'where the trainer takes it: the CIFAR and Imagenette ones), then the full eval. wall is end to end; the step-0 loss '
       'is ln 10 plus the head\'s init; the eval after one short epoch means nothing beyond "the eval forward runs". '
       'ms/step is the median over steps 21-60 of a separate 60-step probe (probe/), and the 80-ep column is 295 steps x 80 at that rate, training only.\n'
       'net\tstatus\twall_s\tmaxrss_gb\tloss_step0\tloss_epoch1\teval_after_epoch1\tms_per_step\ttrain_80ep_min\tnote\n')
(D / 'results.tsv').write_text(hdr + '\n'.join('\t'.join(map(str, r)) for r in rows) + '\n')
print((D / 'results.tsv').read_text())

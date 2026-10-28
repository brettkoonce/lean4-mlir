#!/usr/bin/env python3
"""Three-seed NEU-DET table in the shape of runs/2026-09-17-neudet-fpn-run1/README.md's Table 1:
test mAP@0.5 at each arm's val-peak epoch, per seed, then mean ± sd; the val sweep rows; the
per-class test AP at the val peak. Reads the sweep dirs' table_{val,test}.txt (seed 0 = the
09-17 sweeps, seeds 1 and 2 = this run's)."""
import glob
import os
import statistics as st
import sys

ARMS = [('fpn', 'run1', 'R34+FPN, VisDrone recipe'),
        ('fpn', 'noaug', 'R34+FPN, HSV+hflip off'),
        ('fpn', 'noboot', 'R34+FPN, no ImageNet bootstrap'),
        ('grid', 'run1', 'R34 single grid 14x14')]
CLASSES = ['crazing', 'inclusion', 'patches', 'pitted', 'rolled-in', 'scratches']


def sweep_dir(head, tag, seed):
    pat = (f'runs/2026-09-17-neudet-{head}-{tag}-sweep' if seed == 0
           else f'runs/2026-10-2*-neudet-{head}-{tag}_s{seed}-sweep')
    hits = sorted(glob.glob(pat))
    if not hits:
        sys.exit(f'no sweep dir for {pat}')
    return hits[-1]


def table(path):
    rows = {}
    with open(path) as f:
        for line in f:
            if not (line.startswith('e') and line.split()[0][1:].isdigit()):
                continue
            left, _, right = line.partition('|')
            parts = left.split()
            ep = int(parts[0][1:])
            rows[ep] = {'map': float(parts[1]), 'caap': float(parts[2]), 'recall': float(parts[3]),
                        'argmax': float(parts[4]), 'cls': [float(x) for x in right.split()]}
    return rows


def main():
    seeds = (0, 1, 2)
    print('## test mAP@0.5 at the val-peak epoch, by seed\n')
    print('| arm | seed | val-peak epoch | test mAP@0.5 | recall | class-agnostic AP | val mAP@0.5 | test at e30 |')
    print('|---|---|---|---|---|---|---|---|')
    peaks = {}
    cls_at_peak = {}
    for head, tag, name in ARMS:
        vals = []
        for s in seeds:
            d = sweep_dir(head, tag, s)
            val, test = table(f'{d}/table_val.txt'), table(f'{d}/table_test.txt')
            ep = max(val, key=lambda e: val[e]['map'])
            t = test.get(ep)
            if t is None:
                sys.exit(f'{d}: no test row at the val peak e{ep}')
            t30 = test.get(30, {}).get('map', float('nan'))
            vals.append((t['map'], t['recall'], t['caap'], val[ep]['map'], t30))
            cls_at_peak[(head, tag, s)] = t['cls']
            print(f"| {name} | {s} | {ep} | {t['map']:.3f} | {t['recall']:.3f} | {t['caap']:.3f} | "
                  f"{val[ep]['map']:.3f} | {t30:.3f} |")
        m = [st.mean(v[i] for v in vals) for i in range(5)]
        sd = [st.stdev(v[i] for v in vals) for i in range(5)]
        peaks[(head, tag)] = [v[0] for v in vals]
        print(f"| | **mean ± sd** | | **{m[0]:.3f} ± {sd[0]:.3f}** | {m[1]:.3f} ± {sd[1]:.3f} | "
              f"{m[2]:.3f} ± {sd[2]:.3f} | {m[3]:.3f} ± {sd[3]:.3f} | {m[4]:.3f} ± {sd[4]:.3f} |")

    print('\n## paired differences on test at the val peak, by seed (mean)\n')
    base = peaks[('fpn', 'run1')]
    for key, label in ((('fpn', 'noaug'), 'HSV+hflip off − recipe'),
                       (('fpn', 'noboot'), 'no bootstrap − recipe'),
                       (('grid', 'run1'), 'single grid − FPN recipe')):
        d = [a - b for a, b in zip(peaks[key], base)]
        print(f"- {label}: {' / '.join(f'{x:+.3f}' for x in d)} (mean {st.mean(d):+.3f})")

    print('\n## val sweep, mAP@0.5, by seed (`table_val.txt`)\n')
    eps = [10, 14, 18, 22, 26, 28, 30]
    print('| arm | seed | ' + ' | '.join(f'e{e}' for e in eps) + ' |')
    print('|---|---|' + '---|' * len(eps))
    for head, tag, name in ARMS:
        for s in seeds:
            val = table(f'{sweep_dir(head, tag, s)}/table_val.txt')
            best = max(val, key=lambda e: val[e]['map'])
            cells = [(f"**{val[e]['map']:.3f}**" if e == best else f"{val[e]['map']:.3f}") for e in eps]
            print(f"| {name} | {s} | " + ' | '.join(cells) + ' |')

    print('\n## per-class test AP at the val peak, mean ± sd over seeds\n')
    print('| arm | ' + ' | '.join(CLASSES) + ' |')
    print('|---|' + '---|' * len(CLASSES))
    for head, tag, name in ARMS:
        cols = []
        for i in range(len(CLASSES)):
            v = [cls_at_peak[(head, tag, s)][i] for s in seeds]
            cols.append(f'{st.mean(v):.2f} ± {st.stdev(v):.2f}')
        print(f'| {name} | ' + ' | '.join(cols) + ' |')


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/../..')
    main()

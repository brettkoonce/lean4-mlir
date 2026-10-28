#!/usr/bin/env python3
"""The four R34+FPN rows of tab:visdrone at three seeds, from this dir's score_<tag>.txt
(multilabel, the canonical decode) and score_<tag>_argmax.txt (the historical one); seed 0 is
the 08-28/08-29 checkpoint rescored here (rescore_s0.sh). Prints the per-seed table with both
decodes, the 50-epoch rows' checkpoint ladder, and the per-class table."""
import glob
import os
import re
import statistics as st

ROWS = [('cfoc2', '12 ep, +photometric aug, clsw none, focal 2'),
        ('aug50', '50 ep, +photometric aug'),
        ('ctrl12', '12 ep, no aug'),
        ('long50', '50 ep, no aug')]
CLASSES = ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tri', 'bus', 'motor']
SEEDS = (0, 1, 2)


def parse(path):
    if not os.path.exists(path):
        return None
    r = {'cls': {}}
    for line in open(path):
        m = re.search(r'class-agnostic localization AP@0.50 = ([\d.]+).*recall=([\d.]+)', line)
        if m:
            r['caap'], r['recall'] = float(m.group(1)), float(m.group(2))
        m = re.match(r'\s*([\w-]+): AP@0.50 = ([\d.]+)', line)
        if m:
            r['cls'][m.group(1)] = float(m.group(2))
        m = re.search(r'^\s*mAP@0.50 = ([\d.]+)', line)
        if m:
            r['map'] = float(m.group(1))
    return r if 'map' in r else None


def load(tag):
    return parse(f'score_{tag}.txt'), parse(f'score_{tag}_argmax.txt')


def msd(v):
    v = [x for x in v if x is not None]
    if not v:
        return '—'
    if len(v) == 1:
        return f'{v[0]:.4f}'
    return f'{st.mean(v):.4f} ± {st.stdev(v):.4f}'


def main():
    print('## the rows, final checkpoint, by seed (VisDrone-DET val, 548 images, 38,759 uncapped GT boxes)\n')
    print('| row | seed | multilabel mAP@0.5 | recall | class-agnostic AP | argmax mAP@0.5 | recall | class-agnostic AP |')
    print('|---|---|---|---|---|---|---|---|')
    for row, name in ROWS:
        acc = {k: [] for k in ('ml', 'mlr', 'mlc', 'am', 'amr', 'amc')}
        for s in SEEDS:
            ml, am = load(f'{row}_s{s}')
            cells = []
            for r, keys in ((ml, ('ml', 'mlr', 'mlc')), (am, ('am', 'amr', 'amc'))):
                if r is None:
                    cells += ['—'] * 3
                    for k in keys:
                        acc[k].append(None)
                else:
                    cells += [f"{r['map']:.4f}", f"{r['recall']:.3f}", f"{r['caap']:.3f}"]
                    for k, v in zip(keys, (r['map'], r['recall'], r['caap'])):
                        acc[k].append(v)
            print(f'| `{row}` ({name}) | {s} | ' + ' | '.join(cells) + ' |')
        print(f"| | **mean ± sd** | **{msd(acc['ml'])}** | {msd(acc['mlr'])} | {msd(acc['mlc'])} | "
              f"**{msd(acc['am'])}** | {msd(acc['amr'])} | {msd(acc['amc'])} |")

    print('\n## the 50-epoch rows along their schedule, multilabel mAP@0.5 (argmax in brackets)\n')
    print('| row | seed | e10 | e20 | e30 | e40 | e50 |')
    print('|---|---|---|---|---|---|---|')
    for row in ('aug50', 'long50'):
        for s in SEEDS:
            cells = []
            for e in ('e10', 'e20', 'e30', 'e40', ''):
                ml, am = load(f'{row}_s{s}{e}')
                cells.append('—' if ml is None else f"{ml['map']:.4f}" + ('' if am is None else f" ({am['map']:.4f})"))
            print(f'| `{row}` | {s} | ' + ' | '.join(cells) + ' |')

    print('\n## the 12-epoch focal row along its schedule, multilabel mAP@0.5 (class-agnostic AP / recall)\n')
    print('| row | seed | e6 | e8 | e10 | e12 |')
    print('|---|---|---|---|---|---|')
    for s in SEEDS:
        cells = []
        for e in ('e6', 'e8', 'e10', ''):
            ml, _ = load(f'cfoc2_s{s}{e}')
            cells.append('—' if ml is None else f"{ml['map']:.4f} ({ml['caap']:.3f} / {ml['recall']:.3f})")
        print(f'| `cfoc2` | {s} | ' + ' | '.join(cells) + ' |')

    print('\n## per-class AP@0.5 at the final checkpoint, multilabel, mean ± sd over seeds\n')
    print('| row | ' + ' | '.join(CLASSES) + ' |')
    print('|---|' + '---|' * len(CLASSES))
    for row, _ in ROWS:
        cols = []
        for c in CLASSES:
            v = [load(f'{row}_s{s}')[0] for s in SEEDS]
            v = [x['cls'].get(c) for x in v if x is not None]
            cols.append('—' if not v else (f'{v[0]:.3f}' if len(v) == 1 else f'{st.mean(v):.3f} ± {st.stdev(v):.3f}'))
        print(f'| `{row}` | ' + ' | '.join(cols) + ' |')


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()

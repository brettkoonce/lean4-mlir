#!/usr/bin/env python3
"""Score the R34-bootstrap vs He-init A/B, with the guards baked in.

The transfer claim is about **sample-efficiency**, not peak Dice — the usual
payoff of a pretrained backbone is reaching a given number in fewer epochs
rather than reaching a better number eventually. So this prints the whole
epoch-by-epoch curve for both arms and an epochs-to-target row, not a pair of
final scores.

Two guards run before anything is reported, both from
planning/archive/r34_brats_retrain.md §5, and both earned by real bugs:

  * **Identical consecutive eval rows.** The VisDrone `long30` eval ran without
    its arm tag and silently scored the same checkpoint six times, printing six
    identical rows that read as "converged". Two consecutive byte-identical
    eval rows are treated as a failure, not a plateau.

  * **Arm/tag agreement.** Each log must announce the arm this script is
    filing it under. A log pasted into the wrong column produces a plausible,
    wrong table.

Usage:
    python3 scripts/probes/brats_r34_ab.py runs/brats_r34_gpu0.log runs/brats_scratch_gpu1.log

    # any two logs of the same arm at different data — the 2D and 2.5D `r34`
    # arms, say — filed under names of your own; the arm guard then checks
    # that BOTH logs declare the arm `--arm` names
    python3 scripts/probes/brats_r34_ab.py 2d.log 25d.log --names "2D r34,2.5D r34 ctx=1" --arm r34
"""
import argparse
import re
import sys

EPOCH_RE = re.compile(r'Epoch (\d+)/(\d+): loss=([\d.]+)')
MIOU_RE = re.compile(r'val mIoU: ([\d.]+)\s+\(per-class: ([^)]*)\)')
DICE_RE = re.compile(r'val Dice (WT|TC|ET): ([\d.]+)')
ARM_RE = re.compile(r'^\s*arm: (\S+)')


def parse(path):
    """Pull (epoch, loss, mIoU, WT, TC, ET) rows out of one training log."""
    rows, cur, arm = [], {}, None
    with open(path) as f:
        for line in f:
            m = ARM_RE.match(line)
            if m and arm is None:
                arm = m.group(1)
            m = EPOCH_RE.search(line)
            if m:
                cur = {'epoch': int(m.group(1)), 'loss': float(m.group(3))}
                continue
            m = MIOU_RE.search(line)
            if m and cur:
                cur['miou'] = float(m.group(1))
                cur['perclass'] = m.group(2).strip()
                continue
            m = DICE_RE.search(line)
            if m and cur:
                cur[m.group(1)] = float(m.group(2))
                if m.group(1) == 'ET':
                    rows.append(cur)
                    cur = {}
    return arm, rows


def guard_distinct(name, rows):
    """The long30 guard: consecutive identical eval rows mean the eval is not
    seeing the checkpoint it thinks it is."""
    bad = []
    for a, b in zip(rows, rows[1:]):
        ka = (a.get('miou'), a.get('WT'), a.get('TC'), a.get('ET'))
        kb = (b.get('miou'), b.get('WT'), b.get('TC'), b.get('ET'))
        if ka == kb and None not in ka:
            bad.append((a['epoch'], b['epoch']))
    if bad:
        print(f"  GUARD FAILED [{name}]: byte-identical eval rows at epochs {bad}")
        print("    (this is the long30 signature — the eval is scoring one checkpoint repeatedly)")
        return False
    return True


def guard_arm(name, declared, path):
    if name is None:
        name = (declared or '').split('_')[0]
    if declared is None:
        print(f"  GUARD FAILED [{name}]: {path} has no 'arm:' line — cannot confirm which arm this is")
        return False
    stem = declared.split('_')[0]
    if stem != name:
        print(f"  GUARD FAILED [{name}]: {path} declares arm '{declared}', filed as '{name}'")
        return False
    return True


def table(name, rows):
    print(f"\n=== {name} ===")
    print(f"  {'ep':>3}  {'loss':>8}  {'mIoU':>7}  {'WT':>7}  {'TC':>7}  {'ET':>7}")
    for r in rows:
        print(f"  {r['epoch']:>3}  {r['loss']:>8.4f}  {r.get('miou', float('nan')):>7.4f}"
              f"  {r.get('WT', float('nan')):>7.4f}  {r.get('TC', float('nan')):>7.4f}"
              f"  {r.get('ET', float('nan')):>7.4f}")


def epochs_to(rows, key, target):
    for r in rows:
        if r.get(key, 0.0) >= target:
            return r['epoch']
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('r34_log')
    ap.add_argument('scratch_log')
    ap.add_argument('--names', default=None,
                    help='"A,B": file the two logs under these names instead of r34/scratch')
    ap.add_argument('--arm', default=None,
                    help='with --names: the arm stem BOTH logs must declare (e.g. r34)')
    args = ap.parse_args()

    arm_a, rows_a = parse(args.r34_log)
    arm_b, rows_b = parse(args.scratch_log)
    if args.names:
        name_a, name_b = (n.strip() for n in args.names.split(','))
        stem_a = stem_b = args.arm or arm_a.split('_')[0] if arm_a else None
        title_a, title_b = name_a, name_b
    else:
        name_a, name_b, stem_a, stem_b = 'r34', 'scratch', 'r34', 'scratch'
        title_a, title_b = 'r34 (ImageNet bootstrap)', 'scratch (He-init control)'
    col_a, col_b = name_a[:12], name_b[:12]

    print("guards:")
    ok = all([
        guard_arm(stem_a, arm_a, args.r34_log),
        guard_arm(stem_b, arm_b, args.scratch_log),
        guard_distinct(name_a, rows_a),
        guard_distinct(name_b, rows_b),
    ])
    if not rows_a or not rows_b:
        print("  GUARD FAILED: one arm has no completed eval rows yet")
        ok = False
    if not ok:
        sys.exit(1)
    print(f"  OK — {len(rows_a)} {name_a} rows / {len(rows_b)} {name_b} rows, no repeats")

    table(title_a, rows_a)
    table(title_b, rows_b)

    # The headline: same net, same data, same schedule — only the init differs.
    print("\n=== epochs to reach a target (lower is the win) ===")
    print(f"  {'metric':>6} {'target':>7}  {col_a:>12}  {col_b:>12}")
    for key in ('miou', 'WT', 'TC', 'ET'):
        best = max([r.get(key, 0.0) for r in rows_a + rows_b] or [0.0])
        for frac in (0.5, 0.8, 0.9):
            tgt = best * frac
            ea, eb = epochs_to(rows_a, key, tgt), epochs_to(rows_b, key, tgt)
            print(f"  {key:>6} {tgt:>7.4f}  {str(ea):>12}  {str(eb):>12}")

    peak = lambda rows, k: max((r.get(k, 0.0) for r in rows), default=0.0)
    print("\n=== peak ===")
    print(f"  {'metric':>6}  {col_a:>12}  {col_b:>12}  {'delta':>7}")
    for key in ('miou', 'WT', 'TC', 'ET'):
        pa, pb = peak(rows_a, key), peak(rows_b, key)
        print(f"  {key:>6}  {pa:>12.4f}  {pb:>12.4f}  {pa - pb:>+7.4f}")


if __name__ == '__main__':
    main()

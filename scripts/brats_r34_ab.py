#!/usr/bin/env python3
"""Score the R34-bootstrap vs He-init A/B, with the guards baked in.

The transfer claim is about **sample-efficiency**, not peak Dice — the usual
payoff of a pretrained backbone is reaching a given number in fewer epochs
rather than reaching a better number eventually. So this prints the whole
epoch-by-epoch curve for both arms and an epochs-to-target row, not a pair of
final scores.

Two guards run before anything is reported, both from
planning/r34_brats_retrain.md §5, and both earned by real bugs:

  * **Identical consecutive eval rows.** The VisDrone `long30` eval ran without
    its arm tag and silently scored the same checkpoint six times, printing six
    identical rows that read as "converged". Two consecutive byte-identical
    eval rows are treated as a failure, not a plateau.

  * **Arm/tag agreement.** Each log must announce the arm this script is
    filing it under. A log pasted into the wrong column produces a plausible,
    wrong table.

Usage:
    python3 scripts/brats_r34_ab.py runs/brats_r34_gpu0.log runs/brats_scratch_gpu1.log
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
    args = ap.parse_args()

    arm_a, rows_a = parse(args.r34_log)
    arm_b, rows_b = parse(args.scratch_log)

    print("guards:")
    ok = all([
        guard_arm('r34', arm_a, args.r34_log),
        guard_arm('scratch', arm_b, args.scratch_log),
        guard_distinct('r34', rows_a),
        guard_distinct('scratch', rows_b),
    ])
    if not rows_a or not rows_b:
        print("  GUARD FAILED: one arm has no completed eval rows yet")
        ok = False
    if not ok:
        sys.exit(1)
    print(f"  OK — arms distinct, {len(rows_a)} r34 rows / {len(rows_b)} scratch rows, no repeats")

    table('r34 (ImageNet bootstrap)', rows_a)
    table('scratch (He-init control)', rows_b)

    # The headline: same net, same data, same schedule — only the init differs.
    print("\n=== transfer: epochs to reach a target (lower is the win) ===")
    print(f"  {'metric':>6} {'target':>7}  {'r34':>6}  {'scratch':>8}")
    for key in ('miou', 'WT', 'TC', 'ET'):
        best = max([r.get(key, 0.0) for r in rows_a + rows_b] or [0.0])
        for frac in (0.5, 0.8, 0.9):
            tgt = best * frac
            ea, eb = epochs_to(rows_a, key, tgt), epochs_to(rows_b, key, tgt)
            print(f"  {key:>6} {tgt:>7.4f}  {str(ea):>6}  {str(eb):>8}")

    peak = lambda rows, k: max((r.get(k, 0.0) for r in rows), default=0.0)
    print("\n=== peak ===")
    print(f"  {'metric':>6}  {'r34':>7}  {'scratch':>7}  {'delta':>7}")
    for key in ('miou', 'WT', 'TC', 'ET'):
        pa, pb = peak(rows_a, key), peak(rows_b, key)
        print(f"  {key:>6}  {pa:>7.4f}  {pb:>7.4f}  {pa - pb:>+7.4f}")


if __name__ == '__main__':
    main()

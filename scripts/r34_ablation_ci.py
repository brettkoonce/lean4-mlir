#!/usr/bin/env python3
"""Error bars over seeds for the ResNet-34 recipe ablation (planning/r34_ablation_seeds.md §3.2).

    scripts/r34_ablation_ci.py runs/<date>-r34-ablation-bf16-seeds            # table to stdout
    scripts/r34_ablation_ci.py <dir> --write                                   # + <dir>/RESULTS.md

Reads `<arm>_s<seed>.log` (the multi-seed layout `run_r34_ablation.sh` writes) and takes each
log's LAST `val_acc = N/M = xx.xx%` line. Per arm: n, mean, sd, and a t-based 95% interval
(t_{0.975, n-1}; n = 3 gives ±4.30·sd/√3, n = 5 gives ±2.78·sd/√5). The delta from `full` gets
two intervals: PAIRED by seed (arms sharing a seed share init and data order, so the pairing
usually tightens it) and UNPAIRED (Welch), because the plan said to measure that rather than
assume it. The pgfplots block is the figure's `\\addplot ... error bars` coordinates, paired.

⚠ A log without a final val_acc line is a run that did not finish; it is listed, not silently
dropped, and the arm's n excludes it.
"""
import argparse, math, re, sys
from collections import defaultdict
from pathlib import Path

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
        9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 15: 2.131, 20: 2.086, 30: 2.042}
ARMS = ["full", "nowd", "nowarm", "nols", "noadam", "nocos", "noaug", "bare"]
LINE = re.compile(r"val_acc = (\d+)/(\d+) = ([0-9.]+)%")


def t975(df):
    if df <= 0:
        return float("nan")
    if df in T975:
        return T975[df]
    return next((v for k, v in sorted(T975.items()) if k >= df), 1.960)


def mean_sd(xs):
    n = len(xs)
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else float("nan")
    return m, sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--write", action="store_true", help="also write <dir>/RESULTS.md")
    a = ap.parse_args()
    d = Path(a.dir)
    prec = (d / ".precision").read_text().strip() if (d / ".precision").exists() else "?"
    acc = defaultdict(dict)          # arm -> seed -> top-1
    unfinished = []
    for f in sorted(d.glob("*_s*.log")):
        m = re.fullmatch(r"(.+)_s(\d+)\.log", f.name)
        if not m:
            continue
        arm, seed = m.group(1), int(m.group(2))
        hits = LINE.findall(f.read_text(errors="replace"))
        if not hits:
            unfinished.append(f.name)
            continue
        acc[arm][seed] = float(hits[-1][2])
    if "full" not in acc:
        sys.exit("no finished `full` arm — nothing to take deltas against")
    arms = [x for x in ARMS if x in acc] + sorted(set(acc) - set(ARMS))
    out = []
    out.append(f"# ResNet-34 / Imagenette recipe ablation — {prec}, mean ± 95% CI over seeds\n")
    out.append(f"Directory `{d}`. Interval = t_(0.975, n−1)·sd/√n. Δ intervals: paired by seed "
               f"(shared init + data order) and unpaired (Welch).\n")
    out.append("| arm | n | seeds | top-1 mean | sd | ±95% | Δ vs full | ±95% paired | ±95% unpaired |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    full = acc["full"]
    fm, fsd = mean_sd(list(full.values()))
    coords = []
    for arm in arms:
        xs = acc[arm]
        n = len(xs)
        m, sd = mean_sd(list(xs.values()))
        hw = t975(n - 1) * sd / math.sqrt(n) if n > 1 else float("nan")
        delta = m - fm
        shared = sorted(set(xs) & set(full))
        if arm == "full":
            dp = du = 0.0
        else:
            if len(shared) > 1:
                diffs = [xs[s] - full[s] for s in shared]
                dm, dsd = mean_sd(diffs)
                dp = t975(len(diffs) - 1) * dsd / math.sqrt(len(diffs))
            else:
                dp = float("nan")
            nf = len(full)
            if n > 1 and nf > 1:
                se = math.sqrt(sd ** 2 / n + fsd ** 2 / nf)
                df = se ** 4 / ((sd ** 2 / n) ** 2 / (n - 1) + (fsd ** 2 / nf) ** 2 / (nf - 1))
                du = t975(max(1, round(df))) * se
            else:
                du = float("nan")
        seeds = ",".join(str(s) for s in sorted(xs))
        out.append(f"| {arm} | {n} | {seeds} | {m:.2f} | {sd:.2f} | ±{hw:.2f} | {delta:+.2f} | "
                   f"±{dp:.2f} | ±{du:.2f} |")
        coords.append((arm, delta, dp if arm != "full" else 0.0))
    if unfinished:
        out.append("\n⚠ unfinished (no final val_acc line): " + ", ".join(unfinished))
    out.append("\n## pgfplots (Δ vs full, paired 95% half-width as the x error bar)\n")
    out.append("```")
    out.append("\\addplot[only marks, mark=*, error bars/.cd, x dir=both, x explicit] coordinates {")
    out.append("  " + " ".join(f"({dl:.2f},{arm}) +- ({hw:.2f},0)" for arm, dl, hw in coords))
    out.append("};")
    out.append("```")
    text = "\n".join(out) + "\n"
    print(text)
    if a.write:
        (d / "RESULTS.md").write_text(text)
        print(f"wrote {d / 'RESULTS.md'}", file=sys.stderr)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Emit a pgfplots `\\addplot coordinates {...}` block from an ablation log.

The blueprint's training curves are inline native pgfplots (see the R34
validation curve in the ResNet-34 chapter) rather than raster PNGs: vector,
font-matched, and diffable in git. This regenerates the coordinate block for a
chosen metric so the inline \\addplot in content.tex can be refreshed from the
committed logs/ with one command (parse → paste).

Usage:
  python3 figures/log_to_pgfplots.py logs/ablation_cifar-bn-sgd.log loss
  python3 figures/log_to_pgfplots.py logs/ablation_r34-full.log val

Metrics:
  loss  -> per-epoch training loss   (from "Epoch N/M: loss=X")
  val   -> validation accuracy %     (from "val accuracy ... = X%", tagged to
                                       the most recent epoch seen)
"""
import re
import sys
from pathlib import Path

LOSS_RE = re.compile(r"^Epoch\s+(\d+)/\d+: loss=([\d.]+)")
VAL_RE = re.compile(r"val accuracy.*?=\s*([\d.]+)%")


def parse(path: Path, metric: str) -> list[tuple[int, float]]:
    pts: list[tuple[int, float]] = []
    cur_epoch = 0
    for line in Path(path).read_text().splitlines():
        m = LOSS_RE.match(line)
        if m:
            cur_epoch = int(m.group(1))
            if metric == "loss":
                pts.append((cur_epoch, float(m.group(2))))
            continue
        if metric == "val":
            v = VAL_RE.search(line)
            if v:
                pts.append((cur_epoch, float(v.group(1))))
    return pts


def main() -> None:
    if len(sys.argv) != 3 or sys.argv[2] not in ("loss", "val"):
        print(__doc__)
        sys.exit(1)
    pts = parse(Path(sys.argv[1]), sys.argv[2])
    print(" ".join(f"({e},{v:g})" for e, v in pts))


if __name__ == "__main__":
    main()

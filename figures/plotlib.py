"""Shared plotting helpers for ablation logs.

Kept minimal on purpose — we expect to migrate most of this to Lean
eventually. For now: parse the logs, produce a 2-panel figure with
training loss overlay + validation accuracy markers.
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt

LOSS_RE = re.compile(r"^Epoch\s+(\d+)/\d+: loss=([\d.]+)")
VAL_RE  = re.compile(r"val accuracy.*=\s*([\d.]+)%")


def parse_log(path: Path) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """Return (per-epoch losses, per-measurement val accs).

    Val accs come with the most-recent `Epoch N/...` we've seen,
    so the x-axis aligns with whatever cadence the trainer used.
    """
    losses: list[tuple[int, float]] = []
    vals:   list[tuple[int, float]] = []
    cur_epoch = 0
    for line in Path(path).read_text().splitlines():
        m = LOSS_RE.match(line)
        if m:
            cur_epoch = int(m.group(1))
            losses.append((cur_epoch, float(m.group(2))))
            continue
        m = VAL_RE.search(line)
        if m:
            vals.append((cur_epoch, float(m.group(1))))
    return losses, vals


def plot_bn_vs_nobn(
    log_nobn: Path,
    log_bn:   Path,
    out:      Path,
    suptitle: str,
) -> None:
    """Two-panel figure: training loss + validation accuracy."""
    nobn_loss, nobn_val = parse_log(log_nobn)
    bn_loss,   bn_val   = parse_log(log_bn)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Panel 1: training loss
    e, l = zip(*nobn_loss)
    ax1.plot(e, l, label="no BN", color="#d62728", lw=2)
    e, l = zip(*bn_loss)
    ax1.plot(e, l, label="BN",    color="#1f77b4", lw=2)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("training loss")
    ax1.set_title("Training loss")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper right")

    # Panel 2: validation accuracy
    if nobn_val:
        e, v = zip(*nobn_val)
        ax2.plot(e, v, "o-", label="no BN", color="#d62728", lw=2, ms=8)
        for ei, vi in nobn_val:
            ax2.annotate(f"{vi:.1f}%", (ei, vi), xytext=(5, -14),
                         textcoords="offset points", fontsize=9, color="#d62728")
    if bn_val:
        e, v = zip(*bn_val)
        ax2.plot(e, v, "o-", label="BN",    color="#1f77b4", lw=2, ms=8)
        for ei, vi in bn_val:
            ax2.annotate(f"{vi:.1f}%", (ei, vi), xytext=(5, 6),
                         textcoords="offset points", fontsize=9, color="#1f77b4")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("val accuracy (%)")
    ax2.set_title("Validation accuracy")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="lower right")

    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")

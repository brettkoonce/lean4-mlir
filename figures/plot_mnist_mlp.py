#!/usr/bin/env python3
"""MNIST MLP (s4tfBaseline) — single training-loss curve.

Picks the one actual ablation log used in the blueprint's MLP chapter
and produces a two-panel figure (training loss per epoch + validation
accuracy at reported intervals). Output embedded in the chapter.
"""
from pathlib import Path

import matplotlib.pyplot as plt

from plotlib import parse_log

REPO = Path(__file__).resolve().parent.parent
LOG  = REPO / "logs" / "ablation_mlp-sgd.log"
OUT  = REPO / "blueprint" / "src" / "figures" / "curves" / "mnist_mlp_training.png"


def main() -> None:
    losses, vals = parse_log(LOG)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Training loss, log scale (drops from 2.4 to 1e-3 — need log)
    e, l = zip(*losses)
    ax1.semilogy(e, l, "o-", color="#1f77b4", lw=2, ms=6)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("training loss (log scale)")
    ax1.set_title("Training loss")
    ax1.grid(alpha=0.3, which="both")

    # Validation accuracy
    if vals:
        e, v = zip(*vals)
        ax2.plot(e, v, "o-", color="#2ca02c", lw=2, ms=8)
        for ei, vi in vals:
            ax2.annotate(f"{vi:.2f}%", (ei, vi), xytext=(6, -14),
                         textcoords="offset points", fontsize=10, color="#2ca02c")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("val accuracy (%)")
    ax2.set_title("Validation accuracy")
    ax2.grid(alpha=0.3)

    fig.suptitle(
        "MNIST · MLP 784→512→512→10 · SGD 0.1 · 12 epochs  "
        "(s4tfBaseline, logs/ablation_mlp-sgd.log)",
        fontsize=10,
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

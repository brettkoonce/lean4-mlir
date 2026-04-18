#!/usr/bin/env python3
"""MLP vs CNN on MNIST — same s4tfBaseline config, different architectures.

Shows the +0.4pt lift convolutions provide on MNIST when optimizer,
learning rate, and epoch count are all held fixed. Embedded in the
blueprint's CNN chapter.
"""
from pathlib import Path

import matplotlib.pyplot as plt

from plotlib import parse_log

REPO = Path(__file__).resolve().parent.parent
LOG_MLP = REPO / "logs" / "ablation_mlp-sgd.log"
LOG_CNN = REPO / "logs" / "ablation_cnn-nobn-sgd.log"
OUT     = REPO / "blueprint" / "src" / "figures" / "curves" / "mnist_mlp_vs_cnn.png"


def main() -> None:
    mlp_loss, mlp_val = parse_log(LOG_MLP)
    cnn_loss, cnn_val = parse_log(LOG_CNN)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Training loss (log y-scale, both drop several orders of magnitude)
    e, l = zip(*mlp_loss)
    ax1.semilogy(e, l, "o-", label="MLP (670K p)",    color="#1f77b4", lw=2, ms=6)
    e, l = zip(*cnn_loss)
    ax1.semilogy(e, l, "s-", label="CNN (3.5M p)",    color="#d62728", lw=2, ms=6)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("training loss (log scale)")
    ax1.set_title("Training loss")
    ax1.grid(alpha=0.3, which="both")
    ax1.legend(loc="upper right")

    # Validation accuracy
    if mlp_val:
        e, v = zip(*mlp_val)
        ax2.plot(e, v, "o-", label="MLP (670K p)", color="#1f77b4", lw=2, ms=8)
        for ei, vi in mlp_val:
            ax2.annotate(f"{vi:.2f}%", (ei, vi), xytext=(6, -16),
                         textcoords="offset points", fontsize=10, color="#1f77b4")
    if cnn_val:
        e, v = zip(*cnn_val)
        ax2.plot(e, v, "s-", label="CNN (3.5M p)", color="#d62728", lw=2, ms=8)
        for ei, vi in cnn_val:
            ax2.annotate(f"{vi:.2f}%", (ei, vi), xytext=(6, 8),
                         textcoords="offset points", fontsize=10, color="#d62728")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("val accuracy (%)")
    ax2.set_title("Validation accuracy")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="lower right")

    fig.suptitle(
        "MNIST · MLP vs CNN · same s4tfBaseline (SGD 0.1, 12 ep)  "
        "— convolutions buy ~0.4 pt at the cost of 5× more params",
        fontsize=10,
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

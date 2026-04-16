#!/usr/bin/env python3
"""MNIST-Lite BN vs no-BN learning curves."""
from pathlib import Path

from plotlib import plot_bn_vs_nobn

REPO = Path(__file__).resolve().parent.parent

plot_bn_vs_nobn(
    log_nobn = REPO / "logs" / "ablation_cnn-lite-nobn-sgd.log",
    log_bn   = REPO / "logs" / "ablation_cnn-lite-bn-sgd.log",
    out      = REPO / "figures" / "mnist_lite_bn_vs_nobn.png",
    suptitle = "MNIST · lite arch (2×[3×3]@128 + GAP + Dense(128→10)) · SGD 0.1",
)

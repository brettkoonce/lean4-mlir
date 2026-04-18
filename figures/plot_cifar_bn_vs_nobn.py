#!/usr/bin/env python3
"""CIFAR-10 CNN, BN vs no-BN at SGD 0.1 — THE BN-lift demo.

The no-BN run is frozen at loss ~log(10) = 2.302 for all 30 epochs
(random guess on 10 classes). The BN run drops cleanly. Same
architecture otherwise, same config otherwise. Embedded in the
blueprint's BatchNorm chapter.
"""
from pathlib import Path

from plotlib import plot_bn_vs_nobn

REPO = Path(__file__).resolve().parent.parent

plot_bn_vs_nobn(
    log_nobn = REPO / "logs" / "ablation_cifar-nobn-sgd.log",
    log_bn   = REPO / "logs" / "ablation_cifar-bn-sgd.log",
    out      = REPO / "blueprint" / "src" / "figures" / "curves" / "cifar_bn_vs_nobn.png",
    suptitle = "CIFAR-10 · full 4-conv arch · SGD 0.1 · 30 epochs  "
               "(BN vs no-BN; logs/ablation_cifar-{bn,nobn}-sgd.log)",
)

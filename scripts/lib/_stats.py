"""The small statistics the demo scorers share: Wilson intervals and the energy distance."""
import math

import numpy as np


def wilson(k, n, z=1.959964):
    """95% Wilson score interval for k successes in n, in percent."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z / d) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, c - h) * 100, min(1.0, c + h) * 100)


def acc_str(k, n):
    """`k/n` as a percentage with its Wilson interval, e.g. ` 98.62% [98.31, 98.88]`."""
    lo, hi = wilson(k, n)
    return f"{100 * k / n:6.2f}% [{lo:.2f}, {hi:.2f}]"


def energy_distance(x, y, cap=2048, seed=0):
    """2E|X-Y| - E|X-X'| - E|Y-Y'| between two point sets `[n, d]`. O(n^2), so each side is
    subsampled to `cap` points under a fixed seed."""
    rng = np.random.default_rng(seed)
    if len(x) > cap: x = x[rng.choice(len(x), cap, replace=False)]
    if len(y) > cap: y = y[rng.choice(len(y), cap, replace=False)]
    d = lambda a, b: np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))
    return float(2 * d(x, y).mean() - d(x, x).mean() - d(y, y).mean())

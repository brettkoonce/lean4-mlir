#!/usr/bin/env python3
"""The noise-level sidecar: one σ̂ per record, estimated from the LOW-dose image alone.

  .venv-lidc/bin/python scripts/datasets/ct_noise_level.py <pairs.bin> [<pairs.bin> ...] [--workers 8] [--force] \
      [--estimator mad3|g8]

Writes `<pairs.bin>.sigma.f32` — `count` float32 little-endian values in record order, σ̂ in HU —
for the noise-conditioned generator of planning/lidc_ct_denoising_demo.md §11: the net is told
the slice's noise level through one constant channel, and since the number must exist on the
held-out set, it is computed from the low-dose image only, the same way at train and test time.
The estimator is the median absolute deviation of a 3×3 high-pass over body pixels,

    body = (low > −500 HU) eroded 5 px;  hp = low − median₃ₓ₃(low);  σ̂ = 1.4826 · median |hp[body]|

— it needs no flat region (the preprocessor's flat-soft-tissue estimator wants 7×7 windows inside
−200..300 HU, which a 200 HU noise fails), carries no correction factor (a 3×3 median high-pass
of white noise has MAD ≈ 0.55σ, so σ̂ is a monotone proxy of the noise level, not the level
itself; what the conditioning needs is that the proxy be the same function of the image
everywhere), and is cheap. A record with fewer than 1,000 body pixels gets the file's median σ̂.

The gate is stability, not value: on an LIDC file the rows carry the simulation's own
`sigma_add_hu` and the scan's `sigma_full_hu`, and σ̂ / √(sigma_add² + sigma_full²) is printed by
dose quartile (`dose` per row on a range build; overall otherwise) — within ±10 % across the
quartiles means the proxy tracks the level across the range. A second check, available on every
file, is σ̂ against the body std of (low − full), the inserted noise itself.

Measured on the 4-patient mixed build (2026-11-03): `mad3` is flat across dose quartiles (0.54–0.58
of the total) but reads the TEXTURE too — 0.82 of the level on ramp-filtered noise, 0.39 on Hann —
so on a sharp-kernel scanner it says "noisier" than the level alone would. `--estimator g8`, the
MAD of low − Gaussian₈(low) about its median, is flat across filters (1.16–1.33) but rises with
dose (1.07 at the heaviest quartile to 1.45 at the lightest: the anatomy left under an 8-px blur
shows once the noise is small). Both are the same function of the image at train and test; which
one the conditioned arm takes is planning/lidc_ct_denoising_demo.md §11.2's call. The sidecar's
layout is the same either way.
"""
import argparse
import json
import multiprocessing
import os
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lidc_records as R  # noqa: E402

MIN_BODY = 1000
W = {}


def init(path):
    W["count"], W["n"], W["mm"] = R.open_records(path)


def sigma_of(k):
    full, low, _ = R.decode(W["mm"][k], W["n"])
    body = ndimage.binary_erosion(low > -500.0, iterations=5)
    nb = int(body.sum())
    if nb < MIN_BODY:
        return k, float("nan"), nb, float("nan")
    if W["est"] == "g8":
        hp = (low - ndimage.gaussian_filter(low, 8.0))[body]
        s_hat = 1.4826 * float(np.median(np.abs(hp - np.median(hp))))
    else:
        hp = low - ndimage.median_filter(low, size=3)
        s_hat = 1.4826 * float(np.median(np.abs(hp[body])))
    ins = float((low - full)[body].std())
    return k, s_hat, nb, ins


def spread(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return (np.median(v), np.percentile(v, 5), np.percentile(v, 95)) if len(v) else (np.nan,) * 3


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pairs", nargs="+")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--estimator", choices=["mad3", "g8"], default="mad3")
    args = ap.parse_args()
    W["est"] = args.estimator
    ctx = multiprocessing.get_context("fork")
    for path in args.pairs:
        out = path + ".sigma.f32"
        count, n, _ = R.open_records(path)
        if os.path.exists(out) and os.path.getsize(out) == 4 * count and not args.force:
            print(f"{out}: present ({count} records), skipped")
            continue
        init(path)
        sig = np.full(count, np.nan, np.float32)
        ins = np.full(count, np.nan)
        nb = np.zeros(count, int)
        with ctx.Pool(args.workers) as pool:
            for k, s, b, i in pool.imap_unordered(sigma_of, range(count), chunksize=16):
                sig[k], nb[k], ins[k] = s, b, i
        few = int(np.sum(~np.isfinite(sig)))
        med = float(np.nanmedian(sig))
        sig[~np.isfinite(sig)] = med
        sig.astype("<f4").tofile(out + ".tmp")
        os.replace(out + ".tmp", out)
        m, p5, p95 = spread(sig)
        print(f"{out}: {count} records; σ̂ median {m:.1f} HU, p5 {p5:.1f}, p95 {p95:.1f}"
              + (f"; {few} with < {MIN_BODY} body pixels took the median" if few else ""))
        ok = np.isfinite(ins) & (ins > 0)
        r_ins = sig[ok] / ins[ok]
        print(f"  σ̂ / body std(low − full): median {np.median(r_ins):.3f} [p5 {np.percentile(r_ins, 5):.3f}, p95 {np.percentile(r_ins, 95):.3f}]")
        jpath = path[: -len(".bin")] + ".json"
        if not os.path.exists(jpath):
            continue
        rows = json.load(open(jpath))
        if len(rows) != count or "sigma_add_hu" not in rows[0]:
            continue
        add = np.array([r.get("sigma_add_hu", np.nan) for r in rows], float)
        own = np.array([r.get("sigma_full_hu", np.nan) for r in rows], float)
        dose = np.array([r.get("dose", np.nan) for r in rows], float)
        cal = np.isfinite(add) & np.isfinite(own)
        ratio = sig / np.sqrt(add ** 2 + own ** 2)
        groups = [("all calibrated", cal)]
        if np.isfinite(dose).any() and len(np.unique(dose[cal])) > 4:
            edges = np.quantile(dose[cal], [0.25, 0.5, 0.75])
            q = np.searchsorted(edges, dose)
            groups += [(f"dose quartile {i} ({dose[cal & (q == i)].min():.3f}-{dose[cal & (q == i)].max():.3f})", cal & (q == i)) for i in range(4)]
        print("  σ̂ / √(sigma_add² + sigma_full²), the simulation's total noise by its own estimator:")
        for name, sel in groups:
            if sel.sum() == 0:
                continue
            rr = ratio[sel]
            print(f"    {name:<38} n {sel.sum():5d}   median {np.median(rr):.3f}  [p5 {np.percentile(rr, 5):.3f}, p95 {np.percentile(rr, 95):.3f}]"
                  f"   σ̂ {np.median(sig[sel]):6.1f}  total {np.median(np.sqrt(add[sel] ** 2 + own[sel] ** 2)):6.1f} HU")


if __name__ == "__main__":
    main()

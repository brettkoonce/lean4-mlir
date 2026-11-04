#!/usr/bin/env python3
"""The CT denoising demo's figure — planning/lidc_ct_denoising_demo.md §6 phase 5 (a draft).

  (a) one full-dose validation slice in the lung window, the < 3 mm nodule boxed: the object
  (b) three nodules cropped, in every arm: full dose (the readers' outline drawn), the simulated
      quarter dose, TV, BM3D, the MSE UNet, Noise2Noise — a < 3 mm one, a ground-glass one (the
      readers' texture score <= 2, faint against lung), and the split's largest
  (c) HU along a line through the < 3 mm and the ground-glass nodule, the same arms: how much
      of each one's contrast each arm keeps

  .venv-lidc/bin/python scripts/demos/lidc_figure.py <nodules.json> [out.png] [--data=data/lidc]

Every arm's output is read from its dump (`data/lidc/out/<arm>/…eps.f32`, the noise in HU/100
on the val split's realisation 0; `scripts/datasets/lidc_records.py` finds and reads them). The nodules are chosen, not hand-picked. The < 3 mm one is the
typical one of `lidc_nodules.py`'s < 3 mm row (its `--json` output, <nodules.json>): the nodule
whose contrast kept is closest, summed over the arms drawn, to that row's medians, shown on its
highest-contrast slice; the largest is the connected region of >= 3 readers' agreement with the
largest equivalent diameter sitting in lung (median full-dose HU of a 4-px ring < -700); the
ground-glass one is the
reader-marked texture-1/2 nodule of 5-12 mm (`nodules.json`, matched to a val slice by patient,
z and centroid) with the most readers' agreement on its slice.
"""
import json
import os
import sys

import numpy as np
from scipy import ndimage

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets"))
import lidc_records as R  # noqa: E402

S, HW = R.S, R.HW
LUNG = (-1350, 150)
ARM_DIRS = ["tv", "bm3d", "mse", "n2n"]  # the arms drawn, as `<data>/out/<arm>/` dir names; labels from lidc_records
COLORS = {"full dose": "black", "quarter dose": "0.6", "TV": "tab:orange", "BM3D": "tab:purple",
          "MSE UNet": "tab:blue", "Noise2Noise": "tab:green"}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9.5})


def main():
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    table, out = pos[0], (pos[1] if len(pos) > 1 else "lidc_denoise.png")
    data = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--data=")), "data/lidc")
    vbin = f"{data}/val.bin"
    count, nreal, mm = R.open_records(vbin)
    meta = json.load(open(f"{data}/val.json"))
    dumps = R.find_dumps(data)
    ARMS = [(R.label(a), dumps[a]) for a in ARM_DIRS if a in dumps]
    eps = {name: R.open_dump(p, count) for name, p in ARMS}

    def decode(k):
        return R.decode(mm[k], nreal)

    def arms(k):
        full, low, mask = decode(k)
        imgs = {"full dose": full, "quarter dose": low}
        for name, _ in ARMS:
            imgs[name] = low - 100.0 * eps[name][k].astype(np.float64)
        return imgs, mask

    # the largest: >= 3 readers, in lung
    best_large = None
    for k, r in enumerate(meta):
        if r["nodule_px"] == 0:
            continue
        full, _, mask = decode(k)
        lab, n = ndimage.label(mask >= 3)
        for j in range(1, n + 1):
            reg = lab == j
            area = reg.sum()
            d = 2 * np.sqrt(area / np.pi) * r["spacing"]
            ring = ndimage.binary_dilation(reg, iterations=4) & ~ndimage.binary_dilation(reg, iterations=1)
            if np.median(full[ring]) >= -700:
                continue
            cy, cx = (int(round(v)) for v in ndimage.center_of_mass(reg))
            if best_large is None or d > best_large[0]:
                best_large = (d, k, cy, cx)
    tiny = [r for r in json.load(open(table)) if r["texture"] == "unrated"]
    drawn = [n for n, _ in ARMS]
    med = {n: np.median([r["kept"][n] for r in tiny]) for n in drawn}
    t = min(tiny, key=lambda r: sum(abs(r["kept"][n] - med[n]) for n in drawn))
    best_small = (3.0, t["record"], t["cy"], t["cx"])
    print(f"< 3 mm nodule: record {t['record']} ({t['patient']}), kept "
          + ", ".join(f"{n} {100 * t['kept'][n]:.0f}%" for n in drawn)
          + " (row medians " + ", ".join(f"{100 * med[n]:.0f}%" for n in drawn) + ")")
    print(f"large nodule: {best_large[0]:.1f} mm, record {best_large[1]} ({meta[best_large[1]]['patient']})")

    by_patient = {}
    for k, r in enumerate(meta):
        if r["nodule_px"]:
            by_patient.setdefault(r["patient"], []).append(k)
    best_ggo = None
    for x in json.load(open(f"{data}/nodules.json")):
        if x["small"] or x.get("texture") not in (1, 2) or not 5 <= x["diam_mm"] <= 12:
            continue
        cxn, cyn, czn = x["centroid"]
        for k in by_patient.get(x["patient"], []):
            if abs(meta[k]["z_mm"] - czn) > 1.5:
                continue
            _, _, mask = decode(k)
            yy, xx = int(round(cyn)), int(round(cxn))
            agree = mask[yy - 2:yy + 3, xx - 2:xx + 3].mean()
            if best_ggo is None or agree > best_ggo[4]:
                best_ggo = (x["diam_mm"], k, yy, xx, agree)
    print(f"ground-glass nodule: {best_ggo[0]:.1f} mm, record {best_ggo[1]} ({meta[best_ggo[1]]['patient']}), "
          f"{best_ggo[4]:.1f} readers at its centre")
    best_ggo = best_ggo[:4]

    names = ["full dose", "quarter dose"] + [n for n, _ in ARMS]
    fig = plt.figure(figsize=(13.0, 7.9))
    gs = fig.add_gridspec(5, 8, height_ratios=[1, 1, 1, 0.22, 1.25], hspace=0.08, wspace=0.05,
                          width_ratios=[2.2, 0.32] + [1] * 6)

    def win(img):
        return np.clip((img - LUNG[0]) / (LUNG[1] - LUNG[0]), 0, 1)

    d, k, cy, cx = best_small
    imgs, mask = arms(k)
    ax = fig.add_subplot(gs[0:3, 0])
    ax.imshow(win(imgs["full dose"]), cmap="gray", vmin=0, vmax=1)
    h = 32
    ax.add_patch(Rectangle((cx - h, cy - h), 2 * h, 2 * h, fill=False, ec="tab:red", lw=1.2))
    ax.set_title("(a) a full-dose chest CT slice\n(lung window), the nodule boxed", loc="left")
    ax.axis("off")

    rows = ((0, best_small, 32, "nodule"), (1, best_ggo, 32, "ground-glass"), (2, best_large, 64, "solid"))
    for row, (dd, kk, yy, xx), half, label in rows:
        im, mk = arms(kk)
        yy, xx = min(max(yy, half), S - half), min(max(xx, half), S - half)
        for c, name in enumerate(names):
            a = fig.add_subplot(gs[row, 2 + c])
            a.imshow(win(im[name][yy - half:yy + half, xx - half:xx + half]), cmap="gray", vmin=0, vmax=1,
                     interpolation="nearest")
            if name == "full dose":
                a.contour(mk[yy - half:yy + half, xx - half:xx + half] >= 2, levels=[0.5], colors="tab:red",
                          linewidths=0.7)
            if row == 0:
                a.set_title(("(b) " if c == 0 else "") + name)
            if c == 0:
                a.set_ylabel(("< 3 mm" if row == 0 else f"{dd:.0f} mm") + f"\n{label}", fontsize=8.5)
            a.set_xticks([])
            a.set_yticks([])
            if row < 2:
                a.axhline(half, color="yellow", lw=0.6, ls=":", alpha=0.8)

    sub = gs[4, :].subgridspec(1, 2, wspace=0.16)
    for j, ((dd, kk, cy, cx), label) in enumerate(((best_small, "< 3 mm"), (best_ggo, "ground-glass"))):
        im, mk = arms(kk)
        ax = fig.add_subplot(sub[0, j])
        w = 20
        xs = (np.arange(cx - w, cx + w + 1) - cx) * meta[kk]["spacing"]
        for name in names:
            ax.plot(xs, im[name][cy, cx - w:cx + w + 1], color=COLORS[name],
                    lw=1.9 if name == "full dose" else 1.2, alpha=0.7 if name == "quarter dose" else 1.0,
                    ls="--" if name == "Noise2Noise" else "-", label=name)
        inside = mk[cy, cx - w:cx + w + 1] >= 2
        if inside.any():
            ax.axvspan(xs[inside][0], xs[inside][-1], color="tab:red", alpha=0.08, lw=0)
        ax.set_xlabel("mm along the dotted line through the "
                      + ("< 3 mm" if j == 0 else f"{dd:.0f} mm ground-glass") + " nodule")
        if j == 0:
            ax.set_ylabel("HU")
            ax.legend(ncol=2, fontsize=7.5, frameon=False, loc="center left", bbox_to_anchor=(0.0, 0.52))
        ax.set_title(("(c) " if j == 0 else "") + f"{label}: shaded = inside the readers' outline", loc="left")
        ax.grid(alpha=0.25)
    fig.savefig(out, dpi=190, facecolor="white", bbox_inches="tight")
    print(f"→ {out}")


if __name__ == "__main__":
    main()

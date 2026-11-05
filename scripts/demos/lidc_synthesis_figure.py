#!/usr/bin/env python3
"""The CT synthesis demo's figure (planning/lidc_ct_denoising_demo.md §10.6; the book's
fig:ct_synthesis): the object first, then the findings, then the texture.
  (a) one Mayo chest slice, lung window, cropped to the body: the vendor's 10 % dose, one draw of each
      generator in --mayo-arms (arm dirs under data/mayo/out/, default the generator as trained and the
      one trained on Mayo's kernel), the routine-dose scan — the slice is chosen by rule: of the scored
      Mayo records whose low-dose RMSE is within 10 % of the median (--mayo-csv, lidc_per_slice.py's
      per-record table), the one with the most lung in frame
  (b) three nodules of the 10 % LIDC validation set, cropped, in: the scanned slice (the readers'
      outline drawn), the simulated tenth dose, BM3D, the MSE UNet, two samples of the generator —
      chosen as lidc_figure.py chooses them: the < 3 mm one whose contrast kept is closest to its
      row's medians over the arms drawn, the ground-glass one (readers' texture 1–2, 5–12 mm) with
      the most agreement, the largest ≥ 3-reader region sitting in lung
  (c) the radial noise power spectrum of each arm's output and of the scanned slices' own grain,
      LIDC 10 % left and Mayo right (lidc_texture.py's --nps-csv), log scale; --nps10b50 overlays the
      low-dose curve of the build simulated under Mayo's kernel on the LIDC panel
  .venv-lidc/bin/python scripts/demos/lidc_synthesis_figure.py out.png --nodules=<nodules_lidc10.json> \
      --mayo-csv=<perslice.csv> --nps10=<nps_lidc10.csv> --nps-mayo=<nps_mayo.csv> [--nps10b50=<nps_lidc10b50.csv>] \
      [--nps-mayo2=<nps_mayo.csv of the x₀-target run, its generator's curve added to the Mayo panel>] \
      [--mayo-arms=diff,diff_b50] [--data10=data/lidc10] [--mayo=data/mayo] \
      [--k1-dump=<a second-sample dump for records k1 lacks>]
The second sample is `diff.k1` where that dump covers the record (the spread set), else --k1-dump,
else the cell says so. Prints the records it chose.
"""
import json
import os
import sys

import matplotlib
import numpy as np
from scipy import ndimage

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets"))
import lidc_records as R  # noqa: E402

S = R.S
LUNG = (-1350, 150)
GEN_TITLES = {"diff": "the generator\nas trained", "diff_b50": "the same, trained\non Mayo's kernel",
              "diff25": "the generator trained\nat a quarter dose",
              "diffmixsig": "the generator trained\nacross doses, told the level",
              "diffmixsig_b50": "the same, across doses\non Mayo's kernel",
              "diffx0_b50": "the kernel one, trained\nto predict the clean slice",
              "diffmixsigx0_b50": "the same, trained to\npredict the clean slice",
              "mse": "the MSE UNet\n", "mse_b50": "the MSE UNet trained\non Mayo's kernel",
              "msemix": "the MSE UNet\ntrained across doses"}  # two lines each; the RMSE line goes under
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9.5})


def win(img):
    return np.clip((img - LUNG[0]) / (LUNG[1] - LUNG[0]), 0, 1)


def read_nps(path):
    """{arm label: curve}, f: the texture scorer's csv, whose arm labels may hold commas."""
    lines = open(path).read().splitlines()
    toks = lines[0].split(",")
    names = []
    for t in toks:
        if t.startswith(" ") and names:
            names[-1] += "," + t
        else:
            names.append(t)
    rows = np.array([[float(v) for v in ln.split(",")] for ln in lines[1:] if ln.strip()])
    assert rows.shape[1] == len(names), (rows.shape, names)
    return {n: rows[:, i] for i, n in enumerate(names)}


def main():
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    opt = {a[2:].split("=", 1)[0]: a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--") and "=" in a}
    out = pos[0] if pos else "lidc_ct_synthesis.png"
    data10, mayo = opt.get("data10", "data/lidc10"), opt.get("mayo", "data/mayo")
    mayo_arms = opt.get("mayo-arms", opt.get("mayo-arm", "diff,diff_b50")).split(",")

    # ── (a) the Mayo slice ──
    mcount, mn, mmm = R.open_records(f"{mayo}/val.bin")
    mdumps = R.find_dumps(mayo)
    gen_keys = [a + (".k0" if a + ".k0" in mdumps else "") for a in mayo_arms]
    gens = [R.open_dump(mdumps[k], mcount) for k in gen_keys]
    csv = [ln.split(",") for ln in open(opt["mayo-csv"]).read().splitlines()]
    hdr, rows = csv[0], csv[1:]
    fi, ki = hdr.index("floor"), hdr.index("record")
    gis = [hdr.index(k) if k in hdr else None for k in gen_keys]
    floors = np.array([float(r[fi]) for r in rows])
    med = np.median(floors)
    cands = [r for r in rows if abs(float(r[fi]) - med) <= 0.10 * med]
    best = None
    for r in cands:
        k = int(r[ki])
        full, _low, _m = R.decode(mmm[k], mn)
        c = full[S // 5: 4 * S // 5, S // 5: 4 * S // 5]
        lung = float(((c < -700) & (c > -1000)).mean())
        if best is None or lung > best[0]:
            best = (lung, k, float(r[fi]), [float(r[gi]) if gi is not None else float("nan") for gi in gis])
    _, km, floor_m, gen_m = best
    full_m, low_m, _ = R.decode(mmm[km], mn)
    gen_imgs = [low_m - 100.0 * g[km].astype(np.float64) for g in gens]
    body = full_m >= -950
    ys, xs = np.where(ndimage.binary_opening(body, iterations=3))
    y0, y1, x0, x1 = max(ys.min() - 8, 0), min(ys.max() + 8, S), max(xs.min() - 8, 0), min(xs.max() + 8, S)
    print(f"(a) Mayo record {km}: low-dose RMSE {floor_m:.0f} HU (median of the scored set {med:.0f}), "
          + ", ".join(f"{k} {v:.0f} HU" for k, v in zip(gen_keys, gen_m)) + f", lung fraction {best[0]:.2f}")

    # ── (b) the nodules ──
    count, nreal, mm = R.open_records(f"{data10}/val.bin")
    meta = json.load(open(f"{data10}/val.json"))
    dumps = R.find_dumps(data10)
    arm_keys = [("BM3D", "bm3d"), ("MSE UNet", "mse"), ("sample 1", "diff.k0")]
    eps = {name: R.open_dump(dumps[key], count) for name, key in arm_keys}
    k1_recs = set(R.records_of(dumps["diff.k1"]) or []) if "diff.k1" in dumps else set()
    k1 = R.open_dump(dumps["diff.k1"], count) if "diff.k1" in dumps else None
    extra, extra_recs = None, set()
    if opt.get("k1-dump") and os.path.exists(opt["k1-dump"]):
        extra = R.open_dump(opt["k1-dump"], count)
        extra_recs = set(R.records_of(opt["k1-dump"]) or range(count))

    def arms(k):
        full, low, mask = R.decode(mm[k], nreal)
        imgs = {"scanned": full, "tenth dose": low}
        for name, _ in arm_keys:
            imgs[name] = low - 100.0 * eps[name][k].astype(np.float64)
        if k in k1_recs:
            imgs["sample 2"] = low - 100.0 * k1[k].astype(np.float64)
        elif k in extra_recs:
            imgs["sample 2"] = low - 100.0 * extra[k].astype(np.float64)
        else:
            imgs["sample 2"] = None
        return imgs, mask

    table = json.load(open(opt["nodules"]))
    drawn = ["BM3D", "MSE UNet", "diffusion k0"]
    tiny = [r for r in table if r["texture"] == "unrated"]
    medk = {n: np.median([r["kept"][n] for r in tiny]) for n in drawn}
    t = min(tiny, key=lambda r: sum(abs(r["kept"][n] - medk[n]) for n in drawn))
    best_small = (3.0, t["record"], t["cy"], t["cx"])
    best_large = None
    for k, r in enumerate(meta):
        if r["nodule_px"] == 0:
            continue
        full, _, mask = R.decode(mm[k], nreal)
        lab, n = ndimage.label(mask >= 3)
        for j in range(1, n + 1):
            reg = lab == j
            d = 2 * np.sqrt(reg.sum() / np.pi) * r["spacing"]
            ring = ndimage.binary_dilation(reg, iterations=4) & ~ndimage.binary_dilation(reg, iterations=1)
            if np.median(full[ring]) >= -700:
                continue
            cy, cx = (int(round(v)) for v in ndimage.center_of_mass(reg))
            if best_large is None or d > best_large[0]:
                best_large = (d, k, cy, cx)
    by_patient = {}
    for k, r in enumerate(meta):
        if r["nodule_px"]:
            by_patient.setdefault(r["patient"], []).append(k)
    best_ggo = None
    for x in json.load(open(f"{data10}/nodules.json")):
        if x["small"] or x.get("texture") not in (1, 2) or not 5 <= x["diam_mm"] <= 12:
            continue
        cxn, cyn, czn = x["centroid"]
        for k in by_patient.get(x["patient"], []):
            if abs(meta[k]["z_mm"] - czn) > 1.5:
                continue
            _, _, mask = R.decode(mm[k], nreal)
            yy, xx = int(round(cyn)), int(round(cxn))
            agree = mask[yy - 2:yy + 3, xx - 2:xx + 3].mean()
            if best_ggo is None or agree > best_ggo[4]:
                best_ggo = (x["diam_mm"], k, yy, xx, agree)
    best_ggo = best_ggo[:4]
    print(f"(b) < 3 mm: record {best_small[1]} ({meta[best_small[1]]['patient']}); ground-glass {best_ggo[0]:.1f} mm: "
          f"record {best_ggo[1]} ({meta[best_ggo[1]]['patient']}); largest {best_large[0]:.1f} mm: record {best_large[1]} "
          f"({meta[best_large[1]]['patient']}); second sample present for "
          + ", ".join(str(k) for k in (best_small[1], best_ggo[1], best_large[1]) if k in k1_recs or k in extra_recs) or "none")

    # ── the figure ──
    fig = plt.figure(figsize=(13.0, 12.6))
    n_a = 2 + len(mayo_arms)  # the top row's height follows its column count (the crops keep their aspect)
    outer = fig.add_gridspec(3, 1, height_ratios=[1.2 * 4 / n_a, 1.75, 1.0], hspace=0.16)
    ga = outer[0].subgridspec(1, n_a, wspace=0.03)
    panels = [(low_m, f"(a) Mayo, the vendor's\n10 % dose\n{floor_m:,.0f} HU from the scan")]
    panels += [(img, f"{GEN_TITLES.get(a, a)}\n{v:,.0f} HU from the scan") for img, a, v in zip(gen_imgs, mayo_arms, gen_m)]
    panels += [(full_m, "the routine-dose\nscan\n")]
    for i, (img, title) in enumerate(panels):
        ax = fig.add_subplot(ga[0, i])
        ax.imshow(win(img[y0:y1, x0:x1]), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(title, loc="left")
        ax.axis("off")
    gb = outer[1].subgridspec(3, 6, wspace=0.05, hspace=0.08)
    names = ["scanned", "tenth dose", "BM3D", "MSE UNet", "sample 1", "sample 2"]
    rows = ((0, best_small, 32, "nodule"), (1, best_ggo, 32, "ground-glass"), (2, best_large, 64, "solid"))
    for row, (dd, kk, yy, xx), half, label in rows:
        im, mk = arms(kk)
        yy, xx = min(max(yy, half), S - half), min(max(xx, half), S - half)
        for c, name in enumerate(names):
            a = fig.add_subplot(gb[row, c])
            if im[name] is None:
                a.text(0.5, 0.5, "second sample\npending", ha="center", va="center", fontsize=8, color="0.4")
                a.set_facecolor("0.93")
            else:
                a.imshow(win(im[name][yy - half:yy + half, xx - half:xx + half]), cmap="gray", vmin=0, vmax=1,
                         interpolation="nearest")
            if name == "scanned":
                a.contour(mk[yy - half:yy + half, xx - half:xx + half] >= 2, levels=[0.5], colors="tab:red",
                          linewidths=0.7)
            if row == 0:
                a.set_title(("(b) " if c == 0 else "") + name)
            if c == 0:
                a.set_ylabel(("< 3 mm" if row == 0 else f"{dd:.0f} mm") + f"\n{label}", fontsize=8.5)
            a.set_xticks([])
            a.set_yticks([])
    gc = outer[2].subgridspec(1, 2, wspace=0.18)
    styles = {"full dose": ("black", 2.0, "-"), "low dose": ("0.6", 1.2, "-"), "MSE UNet": ("tab:blue", 1.4, "-"),
              "diffusion k0": ("tab:red", 1.4, "-"), "BM3D": ("tab:purple", 1.0, ":"), "Gaussian": ("tab:green", 1.0, ":"),
              "MSE UNet, range": ("tab:blue", 1.4, "--"), "diffusion, range + σ̂ k0": ("tab:red", 1.4, "--"),
              "MSE UNet, b50f": ("tab:blue", 1.4, "--"), "diffusion, b50f k0": ("tab:red", 1.4, "--"),
              "low dose, b50f": ("0.6", 1.2, "--"),
              "diffusion, b50f, x₀-target k0": ("tab:red", 1.6, "-")}
    labels = {"full dose": "the scanned slices' own grain", "low dose": "low dose", "MSE UNet": "MSE UNet",
              "diffusion k0": "diffusion, one sample", "BM3D": "BM3D", "Gaussian": "Gaussian",
              "MSE UNet, range": "MSE UNet, trained across doses", "diffusion, range + σ̂ k0": "diffusion, across doses + level",
              "MSE UNet, b50f": "MSE UNet, trained on Mayo's kernel", "diffusion, b50f k0": "diffusion, trained on Mayo's kernel",
              "low dose, b50f": "low dose, simulated under Mayo's kernel",
              "diffusion, b50f, x₀-target k0": "the same, trained to predict the clean slice"}
    for j, (path, title, want) in enumerate(((opt["nps10"], "(c) LIDC validation, 10 % dose",
                                              ["full dose", "low dose", "BM3D", "MSE UNet", "diffusion k0"]),
                                             (opt["nps-mayo"], "Mayo, held out",
                                              ["full dose", "low dose", "Gaussian", "MSE UNet", "MSE UNet, b50f",
                                               "diffusion, b50f k0", "diffusion, b50f, x₀-target k0"]))):
        nps = read_nps(path)
        if j == 1 and opt.get("nps-mayo2"):  # the x₀-target run's Mayo curves, the same frequency grid
            nps = {**read_nps(opt["nps-mayo2"]), **nps}
        f = nps["f_per_mm"]
        ax = fig.add_subplot(gc[0, j])
        for name in want:
            if name not in nps:
                continue
            col, lw, ls = styles[name]
            ax.plot(f, nps[name], color=col, lw=lw, ls=ls, label=labels[name])
        if j == 0 and opt.get("nps10b50"):
            n2 = read_nps(opt["nps10b50"])
            col, lw, ls = styles["low dose, b50f"]
            ax.plot(n2["f_per_mm"], n2["low dose"], color=col, lw=lw, ls=ls, label=labels["low dose, b50f"])
        ax.set_yscale("log")
        ax.set_xlim(0, float(f.max()))
        ax.set_xlabel("spatial frequency (1/mm)")
        if j == 0:
            ax.set_ylabel("noise power (HU² mm²)")
        ax.set_title(title, loc="left")
        ax.legend(fontsize=7.5, frameon=False)
        ax.grid(alpha=0.25, which="both")
    fig.savefig(out, dpi=170, facecolor="white", bbox_inches="tight")
    print(f"→ {out}")


if __name__ == "__main__":
    main()

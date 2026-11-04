#!/usr/bin/env python3
"""The CT denoising demo's nodule table — planning/lidc_ct_denoising_demo.md §1: does a denoiser
keep the nodules? Contrast kept, by nodule size and by texture, for every arm with a dump.

  .venv-lidc/bin/python scripts/demos/lidc_nodules.py [--data=data/lidc] [--pairs=<file.bin>] [--json=out.json] \
      [--records=<file>] [--arms=mse,n2n,diff.k0]

Arms are every dump under `<data>/out/<arm>/` (`lidc_records.find_dumps`; a generator's sample
seeds `diff.k0 …` are columns of their own, "diffusion k0 …", and when several exist a last
line gives the median over nodules of the kept contrast averaged across the seeds). Records:
all of `--pairs` (default `<data>/val.bin`), cut to `--records=<file>` and to every dump's
`.records` sidecar, so every column is measured on the same nodules.

A nodule on a slice is a connected region where >= 2 of the 4 readers drew (the val masks). Its
background is the ring 2-6 px outside it, kept to lung (full-dose HU < -500) and away from every
reader mark; regions with < 20 ring px (on the pleura, the mediastinum) are left out and counted.
Contrast C = mean HU inside − mean HU of the ring, measured in each arm's image; **kept** = the arm's
C over the full-dose C, pooled over the nodule's slices (ΣC_arm / ΣC_full), so 100% is the full dose
and smoothing pulls it down. CNR = C over the ring's std in that arm (it rewards smoothing too: a
flat ring has no std — the reason kept is the headline).

Each region is matched to the readers' nodules (`nodules.json`: same patient, centroid within
the region's radius + 6 px, z within 2 mm), and reader entries closer than 10 px / 3 mm are one
nodule. Size = the matched readers' mean largest-slice diameter, binned 3–6 / 6–10 / >10 mm;
texture = their mean texture score, solid (>= 4) or subsolid (ground-glass and part-solid, < 4).
Nodules every reader called < 3 mm are a row of their own: the readers mark only their centre
and rate nothing, so the region is the preprocessor's 3 mm stand-in disk, the same pixels in
every arm. Cells are medians over nodules with the count; the kept table's brackets are the
median's 95% bootstrap interval (2,000 resamples of the group's nodules, seed 0).
"""
import json
import os
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets"))
import lidc_records as R  # noqa: E402

S, HW = R.S, R.HW
ARM_ORDER = ["gauss", "tv", "bm3d", "bm3dpsd", "mse", "n2n", "diff", "diffmix", "diffmixsig", "msemix",
             "diff_clamp", "diff25", "mse25", "n2n25"]  # table order; unknown arms follow
SIZES = [("3–6 mm", 3, 6), ("6–10 mm", 6, 10), (">10 mm", 10, 1e9)]
TEXTURES = ["subsolid", "solid"]


def texture_class(t):
    return "solid" if t >= 4 else "subsolid"


def arg(name, default=None):
    return next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith(f"--{name}=")), default)


def main():
    data = arg("data", "data/lidc")
    out_json = arg("json")
    vbin = arg("pairs", f"{data}/val.bin")
    count, nreal, mm = R.open_records(vbin)
    meta = json.load(open(vbin[: -len(".bin")] + ".json"))
    dumps = R.find_dumps(data, os.path.basename(vbin)[: -len(".bin")])
    if arg("arms"):  # the arms to table; the record set is the intersection of THEIR sidecars only
        want = arg("arms").split(",")
        dumps = {a: p for a, p in dumps.items() if a in want}
    order = {a: i for i, a in enumerate(ARM_ORDER)}
    keys = sorted(dumps, key=lambda a: (order.get(R.arm_base(a), len(order)), a))
    arms = [("quarter dose", None)] + [(R.label(a), dumps[a]) for a in keys]
    eps = {n: R.open_dump(p, count) for n, p in arms if p}
    recs = set(R.common_records(count, dumps.values(), R.read_record_list(arg("records")) if arg("records") else None))

    # reader entries → nodules, per patient
    patients = {r["patient"] for r in meta}
    entries = [x for x in json.load(open(f"{data}/nodules.json")) if x["patient"] in patients]
    nod = []  # each: patient, list of entries
    for x in entries:
        cx, cy, cz = x["centroid"]
        for g in nod:
            if g["patient"] != x["patient"]:
                continue
            gx, gy, gz = np.mean([e["centroid"] for e in g["e"]], axis=0)
            if np.hypot(cx - gx, cy - gy) < 10 and abs(cz - gz) < 3:
                g["e"].append(x)
                break
        else:
            nod.append({"patient": x["patient"], "e": [x]})
    for i, g in enumerate(nod):
        g["id"] = i
        g["small"] = all(e["small"] for e in g["e"])
        big = [e for e in g["e"] if not e["small"]]
        g["diam"] = float(np.mean([e["diam_mm"] for e in big])) if big else 0.0
        g["texture"] = float(np.mean([e["texture"] for e in big])) if big else 0.0
        g["readers"] = len({e["reader"] for e in g["e"]})
        g["xyz"] = np.mean([e["centroid"] for e in g["e"]], axis=0)
        g["z"] = (min(e["z_mm"][0] for e in g["e"]), max(e["z_mm"][1] for e in g["e"]))

    sums = {}  # nodule id → arm → [ΣC, Σ CNR, slices]
    skipped = {"thin ring": 0, "unmatched": 0}
    for k, r in enumerate(meta):
        if not r["nodule_px"] or k not in recs:
            continue
        full, low, mask = R.decode(mm[k], nreal)
        img = {"full dose": full, "quarter dose": low}
        for n, _ in arms:
            if n in eps:
                img[n] = low - 100.0 * eps[n][k].astype(np.float64)
        lab, nl = ndimage.label(mask >= 2)
        anymark = ndimage.binary_dilation(mask > 0, iterations=1)
        for j in range(1, nl + 1):
            reg = lab == j
            cy, cx = ndimage.center_of_mass(reg)
            rad = np.sqrt(reg.sum() / np.pi)
            match = [g for g in nod if g["patient"] == r["patient"]
                     and np.hypot(g["xyz"][0] - cx, g["xyz"][1] - cy) < rad + 6
                     and g["z"][0] - 2 <= r["z_mm"] <= g["z"][1] + 2]
            if not match:
                skipped["unmatched"] += 1
                continue
            g = min(match, key=lambda g: np.hypot(g["xyz"][0] - cx, g["xyz"][1] - cy))
            ring = (ndimage.binary_dilation(reg, iterations=6) & ~ndimage.binary_dilation(reg, iterations=2)
                    & ~anymark & (full < -500))
            if ring.sum() < 20:
                skipped["thin ring"] += 1
                continue
            acc = sums.setdefault(g["id"], {})
            cfull = full[reg].mean() - full[ring].mean()
            if cfull > acc.get("_best", (-1e9,))[0]:  # the slice that shows it best, for the figure
                acc["_best"] = (cfull, k, int(round(cy)), int(round(cx)))
            for n, im in img.items():
                c = im[reg].mean() - im[ring].mean()
                a = acc.setdefault(n, [0.0, 0.0, 0])
                a[0] += c
                a[1] += c / max(im[ring].std(), 1e-6)
                a[2] += 1

    names = [n for n, _ in arms]
    rows = []
    for gid, acc in sums.items():
        g = nod[gid]
        cf = acc["full dose"][0]
        if cf <= 0:
            continue
        rows.append({"patient": g["patient"], "diam": g["diam"],
                     "texture": "unrated" if g["small"] else texture_class(g["texture"]),
                     "readers": g["readers"], "slices": acc["full dose"][2], "C_full": cf / acc["full dose"][2],
                     "kept": {n: acc[n][0] / cf for n in names},
                     "cnr": {n: acc[n][1] / acc[n][2] for n in ["full dose"] + names},
                     "record": acc["_best"][1], "cy": acc["_best"][2], "cx": acc["_best"][3]})
    print(f"{len(rows)} nodules ({sum(r['slices'] for r in rows)} slice regions) on "
          + (f"{count} val slices" if len(recs) == count else f"{len(recs)} of {count} val slices")
          + "; left out: " + ", ".join(f"{v} {k}" for k, v in skipped.items()))

    def cell(sel, n, key):
        v = [r[key][n] for r in sel]
        return np.median(v) if v else float("nan")

    groups = [("< 3 mm, unrated", [r for r in rows if r["texture"] == "unrated"])]
    groups += [(f"{t}, {s}", [r for r in rows if r["texture"] == t and lo <= r["diam"] < hi])
               for t in TEXTURES for s, lo, hi in SIZES]
    groups = [(g, sel) for g, sel in groups if sel] + [("all", rows)]
    w = max(len(g) for g, _ in groups)
    rng = np.random.default_rng(0)

    def kept(sel, n):
        v = np.array([r["kept"][n] for r in sel]) * 100
        b = np.median(v[rng.integers(0, len(v), (2000, len(v)))], axis=1)
        return f"{np.median(v):5.1f} [{np.percentile(b, 2.5):5.1f}, {np.percentile(b, 97.5):5.1f}]"

    print("\ncontrast kept (median over nodules, % of the full dose's, [95% bootstrap interval])")
    print(f"{'':{w}}  {'n':>4} {'C full':>7}  " + " ".join(f"{n:>20}" for n in names))
    for gname, sel in groups:
        print(f"{gname:{w}}  {len(sel):4d} {np.median([r['C_full'] for r in sel]):7.0f}  "
              + " ".join(f"{kept(sel, n):>20}" for n in names))
    bases = {R.arm_base(a) for a in keys if ".k" in a}
    for b in sorted(bases):
        cols = [R.label(a) for a in keys if R.arm_base(a) == b and ".k" in a]
        if len(cols) < 2:
            continue
        v = np.array([np.mean([r["kept"][c] for c in cols]) for r in rows]) * 100
        bs = np.median(v[rng.integers(0, len(v), (2000, len(v)))], axis=1)
        print(f"{R.label(b) + ', kept averaged over ' + str(len(cols)) + ' samples, all nodules':{w}}  "
              f"{np.median(v):5.1f} [{np.percentile(bs, 2.5):5.1f}, {np.percentile(bs, 97.5):5.1f}]")
    print("\nCNR (median over nodules; full dose first)")
    print(f"{'':{w}}  {'n':>4} {'full':>7}  " + " ".join(f"{n:>12}" for n in names))
    for gname, sel in groups:
        print(f"{gname:{w}}  {len(sel):4d} {cell(sel, 'full dose', 'cnr'):7.1f}  "
              + " ".join(f"{cell(sel, n, 'cnr'):12.1f}" for n in names))
    if out_json:
        json.dump(rows, open(out_json, "w"), indent=1)
        print(f"\n→ {out_json}")


if __name__ == "__main__":
    main()

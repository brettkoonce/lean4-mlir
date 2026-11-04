#!/usr/bin/env python3
"""The per-slice view of a denoising table: for each arm, the share of slices WORSE than the
low-dose floor and the median slice RMSE, per noise bin — the view that separates typical
behaviour from a runaway tail, which a pooled RMSE squares (planning/lidc_ct_denoising_demo.md
§11.6; the generator's Mayo tail).

  .venv-lidc/bin/python scripts/probes/lidc_per_slice.py data/mayo/val.bin --arms=diff_b50.k0,mse_b50,diff.k0 \
      [--records=runs/…/records_mayo.txt] [--by=sigma_add_hu:120,180,250,350] [--csv=out.csv]

Arms are `lidc_records.find_dumps` keys under `<data>/out/` (the pairs file's directory); the
record set is the intersection of the requested list and every arm's sidecar. RMSE is over the
body (scanned HU ≥ −950), as the trainer and lidc_denoise_score.py take it. Bins are on the row
field of `<pairs>.json` as the scorer's --by: half-open at the edges; a `sigma_add_hu` the row
lacks (Mayo's uncalibrated rows) is the measured body std of low − full.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets"))
import lidc_records as R  # noqa: E402


def main():
    args = sys.argv[1:]
    pos = [a for a in args if not a.startswith("--")]
    opt = {a[2:].split("=", 1)[0]: a.split("=", 1)[1] for a in args if a.startswith("--") and "=" in a}
    path = pos[0]
    data = os.path.dirname(path)
    split = os.path.basename(path)[: -len(".bin")]
    count, n, mm = R.open_records(path)
    meta = json.load(open(path[: -len(".bin")] + ".json"))
    dumps = R.find_dumps(data, split)
    arms = opt["arms"].split(",") if opt.get("arms") else sorted(dumps)
    missing = [a for a in arms if a not in dumps]
    if missing:
        sys.exit(f"no dump for {missing}; have {sorted(dumps)}")
    requested = R.read_record_list(opt["records"]) if opt.get("records") else None
    recs = R.common_records(count, [dumps[a] for a in arms], requested)
    eps = {a: R.open_dump(dumps[a], count) for a in arms}
    field, edges = (opt["by"].split(":", 1) if opt.get("by") else ("sigma_add_hu", ""))
    edges = [float(e) for e in edges.split(",")] if edges else []
    rows = []
    for k in recs:
        full, low, _mask = R.decode(mm[k], n)
        body = full >= -950
        nb = max(int(body.sum()), 1)
        d = low - full
        floor = float(np.sqrt((d[body] ** 2).sum() / nb))
        v = meta[k].get(field)
        v = float(v) if v is not None else float("nan")
        if not np.isfinite(v) and field == "sigma_add_hu":
            v = float(d[body].std())
        row = {"record": k, field: v, "floor": floor}
        for a in arms:
            den = low - 100.0 * np.asarray(eps[a][k], dtype=np.float64)
            row[a] = float(np.sqrt(((den - full)[body] ** 2).sum() / nb))
        rows.append(row)
    vals = np.array([r[field] for r in rows])
    bins = [("all", np.ones(len(rows), bool))]
    if edges:
        lo = [-np.inf] + edges
        hi = edges + [np.inf]
        for a, b in zip(lo, hi):
            name = f"<{b:g}" if a == -np.inf else (f"≥{a:g}" if b == np.inf else f"{a:g}–{b:g}")
            bins.append((name, (vals >= a) & (vals < b)))
    w = max(len(R.label(a)) for a in arms) + 2
    print(f"{path}: {len(rows)} slices ({field} bins); per arm: % of slices worse than the floor / median slice RMSE (HU)")
    print(" " * 26 + "".join(f"{R.label(a):>{w}}" for a in arms))
    for name, sel in bins:
        if not sel.any():
            continue
        fl = np.array([r["floor"] for r in rows])[sel]
        line = f"  {name:>10} n {sel.sum():4d} floor {np.median(fl):6.0f}"
        for a in arms:
            x = np.array([r[a] for r in rows])[sel]
            line += f"{100 * (x > fl).mean():5.0f} % /{np.median(x):5.0f}".rjust(w)
        print(line)
    if opt.get("csv"):
        with open(opt["csv"], "w") as f:
            f.write(",".join(["record", field, "floor"] + arms) + "\n")
            for r in rows:
                f.write(",".join(str(r[c]) for c in ["record", field, "floor"] + arms) + "\n")
        print(f"→ {opt['csv']}")


if __name__ == "__main__":
    main()

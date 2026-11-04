#!/usr/bin/env python3
"""Rows of the book's `tab:ct_synthesis` (the LIDC CT synthesis demo) from the score directories.

One `--row` per table row; every number is read from the files the row names, so the table has a
provenance and no transcription:

  --row "LABEL | texture=texture_lidc10.json#ARM | nodules=nodules_lidc10.json#LABEL
                | fidelity=fidelity_mayo_ARM.txt | perslice=perslice_mayo.csv#COL"

  texture   → RMSE (HU), grain (NPS integral ratio), freq. (mean-frequency ratio), invented blobs / slice
  nodules   → contrast kept, median % over all nodules / over those under 3 mm
  fidelity  → Mayo pooled body RMSE (the "→" number of the first `RMSE body` line; `#floor` the input's)
  perslice  → Mayo median slice RMSE and the share of slices the arm leaves worse than the floor
               (`#floor` for the input row)

A field left out prints as `---`. Labels and column names must be spelled as the files spell them;
a miss lists what the file has. A diverged arm (pooled RMSE of 10^5 HU or more: a sampler that ran
away on some slices) prints the order of magnitude in its RMSE cells and `---` for the texture cells
that divergence dominates; nodule contrast, a median over nodules, and the per-slice cells stay.

  scripts/demos/lidc_synthesis_table.py --row "..." --row "..."
"""
import argparse
import csv
import json
import re
import sys

import numpy as np
from decimal import ROUND_HALF_UP, Decimal


def r(x, nd):
    """Round half up on the value as printed to four places, the way the score reports round."""
    return str(Decimal(f"{x:.4f}").quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


def rmse_cell(x):
    """A pooled RMSE: one decimal, or its order of magnitude once a diverged arm has made the decimal moot."""
    return f"$10^{{{int(round(np.log10(x)))}}}$" if x >= 1e5 else r(x, 1)


def parse_row(spec):
    parts = [p.strip() for p in spec.split("|")]
    label, fields = parts[0], {}
    for p in parts[1:]:
        if not p:
            continue
        key, val = p.split("=", 1)
        path, _, sel = val.partition("#")
        fields[key.strip()] = (path.strip(), sel.strip())
    return label, fields


def texture_cols(path, arm):
    arms = json.load(open(path))["arms"]
    if arm not in arms:
        sys.exit(f"⛔ {path}: no arm {arm!r}; has {list(arms)}")
    a = arms[arm]
    inv = a["invented_per_slice"]
    if a["rmse_body"] >= 1e5:
        return [rmse_cell(a["rmse_body"]), "---", "---", "---"]
    return [r(a["rmse_body"], 1), r(a["nps_integral_ratio"][0], 2), r(a["mean_frequency_ratio"][0], 2),
            r(inv, 1) if inv < 10 else r(inv, 0)]


def nodule_cols(path, label):
    rows = json.load(open(path))
    have = list(rows[0]["kept"])
    if label not in have:
        sys.exit(f"⛔ {path}: no arm {label!r}; has {have}")
    kept = np.array([r["kept"][label] for r in rows]) * 100.0
    small = np.array([r["diam"] < 3.0 for r in rows])
    return [f"{r(np.median(kept), 1)} / {r(np.median(kept[small]), 1)}"], int(small.sum()), len(rows)


def fidelity_col(path, sel=""):
    for line in open(path):
        m = re.match(r"\s*RMSE body\s+([\d.]+)\s+→\s+([\d.]+)", line)
        if m:
            return [rmse_cell(float(m.group(2 if sel != "floor" else 1)))]
    sys.exit(f"⛔ {path}: no `RMSE body … → …` line")


def perslice_cols(path, col):
    rows = list(csv.DictReader(open(path)))
    if col not in rows[0]:
        sys.exit(f"⛔ {path}: no column {col!r}; has {list(rows[0])}")
    arm = np.array([float(row[col]) for row in rows])
    floor = np.array([float(row["floor"]) for row in rows])
    worse = "---" if col == "floor" else r(100.0 * np.mean(arm > floor), 0) + "\\,\\%"
    return [r(np.median(arm), 0), worse], len(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--row", action="append", required=True, help="see the module docstring")
    args = ap.parse_args()
    notes = []
    for spec in args.row:
        label, f = parse_row(spec)
        cols = []
        cols += texture_cols(*f["texture"]) if "texture" in f else ["---"] * 4
        if "nodules" in f:
            c, n_small, n_all = nodule_cols(*f["nodules"])
            notes.append(f"{label}: {n_all} nodules, {n_small} under 3 mm")
        else:
            c = ["---"]
        # table order: RMSE, grain, freq., nodules kept, invented
        cols = cols[:3] + c + cols[3:]
        cols += fidelity_col(*f["fidelity"]) if "fidelity" in f else ["---"]
        if "perslice" in f:
            c, n = perslice_cols(*f["perslice"])
            notes.append(f"{label}: {n} Mayo slices")
        else:
            c = ["---", "---"]
        cols += c
        print(f"{label:<44} & " + " & ".join(cols) + r" \\")
    for n in dict.fromkeys(notes):
        print("%", n, file=sys.stderr)


if __name__ == "__main__":
    main()

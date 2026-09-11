#!/usr/bin/env python3
"""One table over every trained arm of the gravitational-wave demo: for each net, the
SNR at P_d = 1/2 on both val sets from the Gaussian-trained and the real-trained
model, from the `table_val.json` each run directory holds. Rows are given as
  <label>=<gauss run dir>:<real run dir>[:params]
and the matched-filter search closes the table.

  .venv-gw/bin/python scripts/gw_compare.py "B0 3ep=runs/a:runs/b:7.1M" ...
"""
import json
import sys


def r50(path, tested):
    try:
        T = json.load(open(f"{path}/table_val.json"))
    except FileNotFoundError:
        return None
    v = T["results"][tested].get("CNN", {}).get("rho50")
    return v


def fmt(v):
    return f"{v:6.2f}" if isinstance(v, float) else "   n/a"


rows = []
for a in sys.argv[1:]:
    label, spec = a.split("=", 1)
    parts = spec.split(":")
    g, r = parts[0], parts[1]
    params = parts[2] if len(parts) > 2 else ""
    rows.append((label, params, r50(g, "gauss"), r50(g, "real"), r50(r, "gauss"), r50(r, "real")))
ref = json.load(open(f"{sys.argv[1].split('=', 1)[1].split(':')[0]}/table_val.json"))
mf = (ref["results"]["gauss"]["matched filter network"]["rho50"],
      ref["results"]["real"]["matched filter network"]["rho50"])
pfa = ref["pfa"]
print(f"SNR at P_d = 1/2, P_fa = {pfa:g} per window (val: 4051 injected / 8177 noise-only)")
print(f"{'':30s}{'params':>7s}   {'trained on gauss':^15s}   {'trained on real':^15s}")
print(f"{'':30s}{'':7s}   {'-> gauss':>7s}{'-> real':>8s}   {'-> gauss':>7s}{'-> real':>8s}")
for label, params, gg, gr, rg, rr in rows:
    print(f"{label:30s}{params:>7s}   {fmt(gg)} {fmt(gr)}   {fmt(rg)} {fmt(rr)}")
print(f"{'PyCBC coherent search':30s}{'':7s}   {fmt(mf[0])} {fmt(mf[1])}   {fmt(mf[0])} {fmt(mf[1])}")

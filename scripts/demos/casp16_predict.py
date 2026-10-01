#!/usr/bin/env python3
"""Assemble the distogram demo's tiled predictions (planning/casp16_distogram_demo.md §3 step 8).

`lake exe distogram-casp predict` writes, per evaluation unit, the summed window logits
<dir>/<EU>.acc.bin (f32 [L, L, 66]) and the window counts <EU>.cnt.bin (f32 [L, L]). Here:
average, symmetrize (a pair's distance does not depend on which residue is i), softmax over
the 66 classes; P(contact) = Σ of the classes whose lower edge is under 8 Å (bins 0..19);
expected distance from the bin centres; top-L/5 long-range precision against the truth in
data/casp16/targets/<EU>.npz, beside ESM-2's own contact head from targets/summary.csv.
Writes <dir>/<EU>.pred.npz (probs f16 [L, L, 66], pcontact f32, edist f32) and <dir>/table.csv."""
import argparse, csv, os, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "datasets"))
from casp16_labels import EDGES, NBINS, FAR, UNOBS
from casp16_targets import top_l5_precision

ROOT = Path(os.environ.get("CASP16_DIR", Path(__file__).resolve().parents[2] / "data" / "casp16"))
NC = NBINS + 2
CENTRES = np.concatenate([(EDGES[:-1] + EDGES[1:]) / 2, [EDGES[-1] + 1.0, 0.0]])  # far bin: 23 Å
CONTACT_MAX = int(np.searchsorted(EDGES, 8.0, side="right") - 2)                    # 19


def assemble(acc, cnt, L):
    cnt = np.maximum(cnt, 1.0)[:, :, None]
    logits = acc.reshape(L, L, NC) / cnt
    logits = 0.5 * (logits + logits.transpose(1, 0, 2))
    logits -= logits.max(-1, keepdims=True)
    p = np.exp(logits)
    p /= p.sum(-1, keepdims=True)
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    a = ap.parse_args()
    d = Path(a.dir)
    base = {r["eu"]: r for r in csv.DictReader(open(ROOT / "targets" / "summary.csv"))}
    rows = []
    print(f"{'EU':12s} {'L':>4s} {'ours P@L/5':>10s} {'ESM-2 head':>10s}  difficulty")
    for accf in sorted(d.glob("*.acc.bin")):
        eu = accf.name[:-8]
        t = np.load(ROOT / "targets" / f"{eu}.npz")
        L = len(t["obs"])
        acc = np.fromfile(accf, np.float32)
        cnt = np.fromfile(d / f"{eu}.cnt.bin", np.float32).reshape(L, L)
        assert acc.size == L * L * NC, (eu, acc.size, L)
        p = assemble(acc, cnt, L)
        pcontact = p[:, :, : CONTACT_MAX + 1].sum(-1).astype(np.float32)
        edist = (p[:, :, :NC - 1] * CENTRES[:NC - 1]).sum(-1).astype(np.float32)
        prec, _ = top_l5_precision(pcontact, t["cls"], t["obs"])
        np.savez_compressed(d / f"{eu}.pred.npz", probs=p.astype(np.float16), pcontact=pcontact, edist=edist,
                            windows=cnt.max())
        b = base.get(eu, {})
        rows.append(dict(eu=eu, L=L, ours=prec, esm=float(b.get("esm_p", "nan")), difficulty=b.get("difficulty", "")))
        print(f"{eu:12s} {L:4d} {prec:10.3f} {rows[-1]['esm']:10.3f}  {rows[-1]['difficulty']}")
    with open(d / "table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    for diff in ("easy", "medium", "hard"):
        sel = [r for r in rows if r["difficulty"] == diff]
        if sel:
            print(f"{diff:6s}: ours {np.mean([r['ours'] for r in sel]):.3f}  ESM-2 head {np.mean([r['esm'] for r in sel]):.3f}  ({len(sel)} EUs)")
    print(f"all   : ours {np.mean([r['ours'] for r in rows]):.3f}  ESM-2 head {np.nanmean([r['esm'] for r in rows]):.3f}  ({len(rows)} EUs) -> {d / 'table.csv'}")

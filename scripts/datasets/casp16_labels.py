#!/usr/bin/env python3
"""Step 4 of planning/casp16_distogram_demo.md §3: distogram labels for every chain in
data/casp16/pdb/<entity>.npz -> data/casp16/labels/<entity>.npz.

Per chain of canonical length L: the Cβ of every residue (Cα for glycine, or when the Cβ is
missing), an L×L matrix of Cβ–Cβ distances, and the class of every pair:
    0..63   64 equal bins over [2, 22) Å   (bin k covers 2 + k·0.3125 .. 2 + (k+1)·0.3125)
    64      22 Å and beyond
    65      unobserved: either residue has no coordinates
    d < 2 Å lands in bin 0 (clashes in the deposited model; rare).
Class 65 has weight 0 in perPixelWeightedCE, so it masks, and the same class pads crops that run
off a short chain. Stored uint8 [L, L] plus the observed mask and the Cβ coordinates (NaN where
unobserved) so the fold step can be checked against the truth. Also prints the sanity numbers
of the batch: fraction observed, fraction of pairs under 8 Å, adjacent-pair Cβ distance."""
import argparse, csv, glob, json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(os.environ.get("CASP16_DIR", Path(__file__).resolve().parents[2] / "data" / "casp16"))
NBINS, LO, HI = 64, 2.0, 22.0
EDGES = np.linspace(LO, HI, NBINS + 1)
FAR, UNOBS = 64, 65


def cb_coords(d, L):
    """[L, 3] Cβ (Cα for Gly / missing Cβ), NaN where the residue has neither."""
    xyz = np.full((L, 3), np.nan, np.float32)
    seq, atom, pos = d["label_seq"], d["atom"], d["xyz"]
    ca = {int(s): p for s, a, p in zip(seq, atom, pos) if a == 1}
    cb = {int(s): p for s, a, p in zip(seq, atom, pos) if a == 4}
    for s in set(ca) | set(cb):
        if 1 <= s <= L:
            xyz[s - 1] = cb.get(s, ca.get(s))
    return xyz


def labels(xyz):
    obs = ~np.isnan(xyz[:, 0])
    diff = xyz[:, None, :] - xyz[None, :, :]
    dist = np.sqrt((diff ** 2).sum(-1))
    cls = np.digitize(dist, EDGES) - 1            # NaN -> 64 then masked below; d < 2 -> -1
    cls = np.clip(cls, 0, FAR).astype(np.uint8)
    cls[~obs, :] = UNOBS
    cls[:, ~obs] = UNOBS
    return cls, obs, dist


def one(path, L, out):
    d = np.load(path)
    xyz = cb_coords(d, L)
    cls, obs, dist = labels(xyz)
    np.savez_compressed(out, cls=cls, obs=obs, cb=xyz)
    adj = dist[np.arange(L - 1), np.arange(1, L)]
    adj = adj[~np.isnan(adj)]
    pairs = cls[np.ix_(obs, obs)]
    return dict(L=L, n_obs=int(obs.sum()), n_pairs=pairs.size,
                n_contact=int(((pairs < FAR) & (EDGES[np.minimum(pairs, 63)] < 8.0)).sum()),
                adj_mean=float(adj.mean()) if adj.size else float("nan"),
                adj_far=int((adj > 7.0).sum()))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list", default=str(ROOT / "train" / "reps.csv"))
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    out = ROOT / "labels"; out.mkdir(exist_ok=True)
    rows = list(csv.DictReader(open(a.list)))
    if a.limit:
        rows = rows[: a.limit]
    t0 = time.time(); stats = []; missing = skipped = 0
    for i, r in enumerate(rows, 1):
        src = ROOT / "pdb" / f"{r['id']}.npz"; dst = out / f"{r['id']}.npz"
        if not src.exists():
            missing += 1; continue
        if dst.exists():
            skipped += 1; continue
        stats.append(one(src, int(r["length"]), dst))
        if i % 2000 == 0:
            print(f"  {i}/{len(rows)}  {i / (time.time() - t0):.0f}/s", flush=True)
    if stats:
        S = {k: np.array([s[k] for s in stats], float) for k in stats[0]}
        print(f"{len(stats)} chains labelled ({skipped} already done, {missing} without coordinates yet): "
              f"observed {S['n_obs'].sum() / S['L'].sum():.3f} of residues; "
              f"{S['n_contact'].sum() / S['n_pairs'].sum():.4f} of observed pairs under 8 Å; "
              f"adjacent Cβ–Cβ mean {np.nanmean(S['adj_mean']):.2f} Å, {int(S['adj_far'].sum())} adjacent pairs over 7 Å "
              f"(chain breaks); {len(rows)} rows in {time.time() - t0:.0f} s")
    size = sum(f.stat().st_size for f in out.glob("*.npz"))
    print(f"-> {out}: {len(list(out.glob('*.npz')))} files, {size / 1e6:.0f} MB")

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


# ── orientation labels (trRosetta's ω, θ, φ; planning/casp16_distogram_demo.md §10) ──────────
OBINS, PBINS, ORIENT_CUT = 24, 12, 20.0      # 15° bins; pairs at Cβ–Cβ ≥ 20 Å are "no contact"
O_NONE, O_UNOBS = OBINS, OBINS + 1           # ω, θ: 24 bins + none + unobserved = 26 classes
P_NONE, P_UNOBS = PBINS, PBINS + 1           # φ: 12 bins + none + unobserved = 14 classes
ORIENT_CLASSES = (OBINS + 2, OBINS + 2, PBINS + 2)


def backbone(d, L):
    """N, Cα, Cβ as [L, 3] (NaN where the residue lacks the atom); Cβ is the deposited one where
    present and otherwise (glycine, missing side chain) the virtual Cβ from N, Cα, C — trRosetta's
    convention and formula. The distance labels (`cb_coords`) keep Cα for glycine."""
    seq, atom, pos = d["label_seq"], d["atom"], d["xyz"]
    out = {a: np.full((L, 3), np.nan, np.float32) for a in (0, 1, 2, 4)}
    for s, a, p in zip(seq, atom, pos):
        if 1 <= s <= L and a in out:
            out[a][s - 1] = p
    N, CA, C, CB = out[0], out[1], out[2], out[4]
    return N, CA, np.where(np.isnan(CB[:, :1]), virtual_cb(N, CA, C), CB)


def virtual_cb(N, CA, C):
    """The ideal Cβ from the backbone (trRosetta's constants), [L, 3]."""
    b, c = CA - N, C - CA
    a = np.cross(b, c)
    return -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA


def dihedral(p0, p1, p2, p3):
    """Signed dihedral in (−π, π] over the last axis, broadcasting over the rest."""
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1n = b1 / np.maximum(np.linalg.norm(b1, axis=-1, keepdims=True), 1e-8)
    v = b0 - (b0 * b1n).sum(-1, keepdims=True) * b1n
    w = b2 - (b2 * b1n).sum(-1, keepdims=True) * b1n
    return np.arctan2((np.cross(b1n, v) * w).sum(-1), (v * w).sum(-1))


def orient_labels(N, CA, cb):
    """[L, L] uint8 planes ω, θ, φ: ω(i, j) = dihedral Cα_i–Cβ_i–Cβ_j–Cα_j (symmetric), θ(i, j) =
    dihedral N_i–Cα_i–Cβ_i–Cβ_j, φ(i, j) = angle Cα_i–Cβ_i–Cβ_j (both asymmetric), in 15° bins;
    `none` beyond 20 Å Cβ–Cβ, `unobserved` where either residue lacks N, Cα or C, and on i = j."""
    L = len(CA)
    obs = ~np.isnan(N[:, 0]) & ~np.isnan(CA[:, 0]) & ~np.isnan(cb[:, 0])
    Ni, CAi, CBi = N[:, None, :], CA[:, None, :], cb[:, None, :]
    CAj, CBj = CA[None, :, :], cb[None, :, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        dist = np.linalg.norm(CBi - CBj, axis=-1)
        omega = dihedral(np.broadcast_to(CAi, (L, L, 3)), np.broadcast_to(CBi, (L, L, 3)),
                         np.broadcast_to(CBj, (L, L, 3)), np.broadcast_to(CAj, (L, L, 3)))
        theta = dihedral(np.broadcast_to(Ni, (L, L, 3)), np.broadcast_to(CAi, (L, L, 3)),
                         np.broadcast_to(CBi, (L, L, 3)), np.broadcast_to(CBj, (L, L, 3)))
        u, w = CAi - CBi, CBj - CBi
        cosphi = (u * w).sum(-1) / np.maximum(np.linalg.norm(u, axis=-1) * np.linalg.norm(w, axis=-1), 1e-8)
        phi = np.arccos(np.clip(cosphi, -1.0, 1.0))
    def binned(x, nb, lo, hi, none, unobs):
        k = np.floor((np.nan_to_num(x, nan=lo) - lo) / (hi - lo) * nb).astype(np.int64)
        k = np.clip(k, 0, nb - 1).astype(np.uint8)
        k[dist >= ORIENT_CUT] = none
        k[~obs, :] = unobs; k[:, ~obs] = unobs
        k[np.isnan(dist)] = unobs
        np.fill_diagonal(k, unobs)
        return k
    return (binned(omega, OBINS, -np.pi, np.pi, O_NONE, O_UNOBS),
            binned(theta, OBINS, -np.pi, np.pi, O_NONE, O_UNOBS),
            binned(phi, PBINS, 0.0, np.pi, P_NONE, P_UNOBS))


def add_orient(src, dst):
    """Add the ω, θ, φ planes to an existing labels file (idempotent)."""
    lab = dict(np.load(dst))
    if "omega" in lab:
        return None
    d = np.load(src)
    L = len(lab["obs"])
    om, th, ph = orient_labels(*backbone(d, L))
    np.savez_compressed(dst, **lab, omega=om, theta=th, phi=ph)
    return om, th, ph


def _orient_job(job):
    """Pool worker: add the planes to one labels file, return their class histograms (None if present)."""
    planes = add_orient(*job)
    if planes is None:
        return None
    return [np.bincount(pl.ravel(), minlength=n) for pl, n in zip(planes, ORIENT_CLASSES)]


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
    p.add_argument("--orient", action="store_true", help="add the ω, θ, φ planes to every existing labels file")
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    out = ROOT / "labels"; out.mkdir(exist_ok=True)
    rows = list(csv.DictReader(open(a.list)))
    if a.limit:
        rows = rows[: a.limit]
    if a.orient:
        from multiprocessing import Pool
        t0 = time.time(); hist = [np.zeros(n, np.int64) for n in ORIENT_CLASSES]; done = 0
        jobs = [(ROOT / "pdb" / f"{r['id']}.npz", out / f"{r['id']}.npz") for r in rows]
        jobs = [j for j in jobs if j[0].exists() and j[1].exists()]
        with Pool(a.workers) as pool:
            for i, hs in enumerate(pool.imap_unordered(_orient_job, jobs, chunksize=16), 1):
                if hs is not None:
                    done += 1
                    for h, hh in zip(hist, hs):
                        h += hh
                if i % 2000 == 0:
                    print(f"  {i}/{len(jobs)}  {i / (time.time() - t0):.0f}/s", flush=True)
        for name, h in zip(("omega", "theta", "phi"), hist):
            n = h.sum(); bins = h[:-2]
            print(f"{name}: {n and bins.sum() / n:.4f} of pairs binned, {h[-2] / n:.4f} none (≥ {ORIENT_CUT:.0f} Å), "
                  f"{h[-1] / n:.4f} unobserved; bin histogram (% of binned) "
                  + " ".join(f"{100 * b / max(bins.sum(), 1):.1f}" for b in bins))
        print(f"{done} chains given orientation planes in {time.time() - t0:.0f} s")
        sys.exit(0)
    # L is the canonical sequence's length (what label_seq_id indexes and what ESM-2 embeds), NOT
    # reps.csv's `length` (rcsb_sample_sequence_length): the two differ for a few entities
    # (8Q79_1: 234 vs 236), and a label matrix of the wrong size shifted every later chain's
    # labels in the packed pool on 2026-10-01.
    seq_len = {}
    with open(ROOT / "train" / "entities.jsonl") as f:
        for line in f:
            e = json.loads(line)
            if e["seq"]:
                seq_len[e["id"]] = len(e["seq"])
    t0 = time.time(); stats = []; missing = skipped = 0
    for i, r in enumerate(rows, 1):
        src = ROOT / "pdb" / f"{r['id']}.npz"; dst = out / f"{r['id']}.npz"
        if not src.exists() or r["id"] not in seq_len:
            missing += 1; continue
        if dst.exists():
            skipped += 1; continue
        stats.append(one(src, seq_len[r["id"]], dst))
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

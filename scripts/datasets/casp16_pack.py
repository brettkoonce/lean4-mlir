#!/usr/bin/env python3
"""Step 7a of planning/casp16_distogram_demo.md §3: pack the per-chain files into the three flat
binaries the Lean trainer reads whole (the GW / BraTS pattern: one ByteArray per set, batches
gathered by index in C):

  <set>_feat.bin    f16, per residue [489]: ESM-2 35M representation (480) | i / 512 |
                    sin/cos(i / 10000^(k/4)) for k = 0..3 (8); chains back to back
  <set>_lab.bin     u8, per chain the L×L class matrix (0..63 bins, 64 far, 65 unobserved),
                    row-major, chains back to back
  <set>_idx.bin     i64 per chain: feat offset (residues), label offset (bytes), L; then the
                    chain's reps.csv row number, so a sample can be traced back

for <set> in train_full (headline), train (purged ablation), val; the two train sets share
chains, so each chain's bytes are written once into a common pool and the set files hold the
index only (train_full_idx.bin, train_idx.bin, val_idx.bin over pool_feat.bin / pool_lab.bin).
The 84 CASP16 EUs go to targets_feat.bin / targets_lab.bin / targets_idx.bin the same way, in
eu_list.csv order, so the trainer's inference pass needs no Python. Prints the footprint."""
import argparse, csv, json, os, struct, time
from pathlib import Path
import numpy as np

ROOT = Path(os.environ.get("CASP16_DIR", Path(__file__).resolve().parents[2] / "data" / "casp16"))
NPOS = 9


def pos_feats(L):
    i = np.arange(L, dtype=np.float32)
    cols = [i / 512.0]
    for k in range(4):
        w = 1.0 / (10000.0 ** (k / 4.0))
        cols += [np.sin(i * w), np.cos(i * w)]
    return np.stack(cols, 1).astype(np.float16)


def pack(rows, feat_of, lab_of, out, name, feat_f=None, lab_f=None, idx_rows=None):
    """Append each row's features + labels to the pool files; return the index rows."""
    own = feat_f is None
    if own:
        feat_f = open(out / f"{name}_feat.bin", "wb"); lab_f = open(out / f"{name}_lab.bin", "wb")
    idx, foff, loff = [], feat_f.tell() // (2 * (480 + NPOS)), lab_f.tell()
    for k, r in enumerate(rows):
        emb = feat_of(r)
        if emb is None:
            continue
        L = emb.shape[0]
        feat = np.concatenate([emb.astype(np.float16), pos_feats(L)], 1)
        assert feat.shape == (L, 480 + NPOS)
        lab = lab_of(r, L)
        feat_f.write(feat.tobytes()); lab_f.write(lab.tobytes())
        idx.append((foff, loff, L, k))
        foff += L; loff += L * L
    if own:
        feat_f.close(); lab_f.close()
    return idx


def write_idx(path, idx):
    with open(path, "wb") as f:
        for t in idx:
            f.write(struct.pack("<4q", *t))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(ROOT / "packed"))
    a = p.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True)
    t0 = time.time()
    # training pool: every representative that has both an embedding and labels
    reps = list(csv.DictReader(open(ROOT / "train" / "reps.csv")))
    have = lambda r: (ROOT / "emb" / f"{r['id']}.npy").exists() and (ROOT / "labels" / f"{r['id']}.npz").exists()
    pool_rows = [r for r in reps if have(r)]
    pos = {r["id"]: k for k, r in enumerate(pool_rows)}
    with open(out / "pool_feat.bin", "wb") as ff, open(out / "pool_lab.bin", "wb") as lf:
        idx = pack(pool_rows, lambda r: np.load(ROOT / "emb" / f"{r['id']}.npy"),
                   lambda r, L: np.load(ROOT / "labels" / f"{r['id']}.npz")["cls"], out, "pool", ff, lf)
    by_id = {pool_rows[t[3]]["id"]: t for t in idx}
    for name in ("train_full", "train", "val"):
        rows = list(csv.DictReader(open(ROOT / "train" / f"{name}.csv")))
        sel = [by_id[r["id"]] for r in rows if r["id"] in by_id]
        write_idx(out / f"{name}_idx.bin", sel)
        print(f"{name}: {len(sel)} chains ({len(rows) - len(sel)} without files yet), {sum(t[2] for t in sel):,} residues")
    # the EUs, in eu_list.csv order; missing ones (no structure) are skipped and listed
    eus = list(csv.DictReader(open(ROOT / "eu_list.csv")))
    tr = [e for e in eus if (ROOT / "targets" / f"{e['eu']}.npz").exists()]
    tidx = pack(tr, lambda e: np.load(ROOT / "targets" / f"{e['eu']}.npz")["emb"],
                lambda e, L: np.load(ROOT / "targets" / f"{e['eu']}.npz")["cls"], out, "targets")
    write_idx(out / "targets_idx.bin", tidx)
    with open(out / "targets_order.txt", "w") as f:
        f.write("\n".join(e["eu"] for e in tr) + "\n")
    print(f"targets: {len(tr)} EUs (skipped {[e['eu'] for e in eus if e not in tr]})")
    tot = sum(f.stat().st_size for f in out.glob("*.bin"))
    print(f"-> {out}: {tot / 1e9:.2f} GB in {time.time() - t0:.0f} s; feature width {480 + NPOS}, "
          f"label classes 66, index row = 4 × i64 (feat off, lab off, L, csv row)")

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
import argparse, csv, json, os, struct, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(os.environ.get("CASP16_DIR", Path(__file__).resolve().parents[2] / "data" / "casp16"))
NPOS = 9
AA = "ACDEFGHIKLMNPQRSTVWY"
# feature sets: name -> (per-residue width before the 9 position features, embedding dir, targets dir)
FEATURES = {"esm": (480, "emb", "targets"), "esm150": (640, "emb150", "targets_esm150"),
            "esm650": (1280, "emb650", "targets_esm650"), "onehot": (21, None, "targets")}


def onehot(seq):
    x = np.zeros((len(seq), 21), np.float16)
    for i, c in enumerate(seq):
        x[i, AA.index(c) if c in AA else 20] = 1.0
    return x


def pos_feats(L):
    i = np.arange(L, dtype=np.float32)
    cols = [i / 512.0]
    for k in range(4):
        w = 1.0 / (10000.0 ** (k / 4.0))
        cols += [np.sin(i * w), np.cos(i * w)]
    return np.stack(cols, 1).astype(np.float16)


def pack(rows, feat_of, lab_of, out, name, feat_f=None, lab_f=None, width=480, fs="", labels=True):
    """Append each row's features (+ labels) to the pool files; return the index rows. With
    `labels=False` only the feature file of feature set `fs` is written (same rows, same
    residue offsets, so the existing index and label pool serve it)."""
    own = feat_f is None
    sfx = f"_{fs}" if fs else ""
    if own:
        feat_f = open(out / f"{name}{sfx}_feat.bin", "wb")
        lab_f = open(out / f"{name}_lab.bin", "wb") if labels else None
    idx, foff, loff = [], feat_f.tell() // (2 * (width + NPOS)), (lab_f.tell() if lab_f else 0)
    for k, r in enumerate(rows):
        emb = feat_of(r)
        if emb is None:
            continue
        L = emb.shape[0]
        feat = np.concatenate([emb.astype(np.float16), pos_feats(L)], 1)
        assert feat.shape == (L, width + NPOS), (feat.shape, width)
        feat_f.write(feat.tobytes())
        if lab_f is not None:
            lab = lab_of(r, L)
            assert lab.shape == (L, L) and lab.dtype == np.uint8, f"{r}: labels {lab.shape}, features {L} rows"
            lab_f.write(lab.tobytes())
        idx.append((foff, loff, L, k))
        foff += L; loff += L * L
    if own:
        feat_f.close()
        if lab_f: lab_f.close()
    return idx


def write_idx(path, idx):
    with open(path, "wb") as f:
        for t in idx:
            f.write(struct.pack("<4q", *t))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(ROOT / "packed"))
    p.add_argument("--val-name", default="valsub", help="name of the val subset (valsub_long for a second one)")
    p.add_argument("--val-len", type=int, nargs=2, default=[80, 200], metavar=("LO", "HI"), help="length window of the val subset")
    p.add_argument("--val-targets", type=int, default=24,
                   help="also pack the N shortest val chains in the targets format (valsub_*) for fold tuning")
    p.add_argument("--orient-only", action="store_true",
                   help="write only pool_orient.bin: the ω, θ, φ planes (u8 [L, L, 3] per chain) over the pool's rows")
    p.add_argument("--sets", default="pool,targets,valsub",
                   help="which sets to pack (the pool's feature file can instead come from casp16_embed.py --pool-out)")
    p.add_argument("--features", default="esm", choices=sorted(FEATURES),
                   help="esm (ESM-2 35M, the default pools) | onehot | esm150: the alternatives write only "
                        "<set>_<features>_feat.bin, over the same rows as the default pools")
    a = p.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True)
    t0 = time.time()
    width, emb_dir, tdir = FEATURES[a.features]
    fs = "" if a.features == "esm" else a.features
    alt = bool(fs)
    seqs = {}
    if a.features == "onehot":
        with open(ROOT / "train" / "entities.jsonl") as f:
            for line in f:
                e = json.loads(line)
                if e["seq"]:
                    seqs[e["id"]] = e["seq"]
    feat_of = (lambda r: onehot(seqs[r["id"]])) if a.features == "onehot" else \
              (lambda r: np.load(ROOT / emb_dir / f"{r['id']}.npy"))
    # training pool: every representative that has both an embedding and labels (the DEFAULT
    # set's membership, so every feature set shares one index and one label pool)
    reps = list(csv.DictReader(open(ROOT / "train" / "reps.csv")))
    # the pool's membership and order, frozen in packed/pool_order.txt the first time (so the per-chain
    # 35M embeddings it was defined by can be dropped once packed); otherwise every representative
    # with an embedding and labels
    order_f = out / "pool_order.txt"
    if order_f.exists():
        ids = set(order_f.read_text().split())
        have = lambda r: r["id"] in ids
    else:
        have = lambda r: (ROOT / "emb" / f"{r['id']}.npy").exists() and (ROOT / "labels" / f"{r['id']}.npz").exists()
    pool_rows = [r for r in reps if have(r)]
    if not order_f.exists():
        order_f.write_text("\n".join(r["id"] for r in pool_rows) + "\n")
    pos = {r["id"]: k for k, r in enumerate(pool_rows)}
    lab_of = lambda r, L: np.load(ROOT / "labels" / f"{r['id']}.npz")["cls"]
    sets = set(a.sets.split(","))
    if a.orient_only:
        # the same rows in the same order as pool_lab.bin, three bytes per label byte, so a pair's
        # planes sit at three times its label offset (casp16_labels.py --orient wrote them)
        n = 0
        with open(out / "pool_orient.bin", "wb") as f:
            for k, r in enumerate(pool_rows):
                lab = np.load(ROOT / "labels" / f"{r['id']}.npz")
                planes = np.stack([lab["omega"], lab["theta"], lab["phi"]], -1)
                assert planes.shape == lab["cls"].shape + (3,) and planes.dtype == np.uint8, (r["id"], planes.shape)
                f.write(planes.tobytes()); n += planes.size
                if (k + 1) % 5000 == 0:
                    print(f"  {k + 1}/{len(pool_rows)}", flush=True)
        lab_bytes = (out / "pool_lab.bin").stat().st_size
        assert n == 3 * lab_bytes, (n, lab_bytes)
        print(f"-> {out / 'pool_orient.bin'}: {n / 1e9:.2f} GB over {len(pool_rows)} chains in {time.time() - t0:.0f} s")
        sys.exit(0)
    if "pool" in sets:
        idx = pack(pool_rows, feat_of, lab_of, out, "pool", width=width, fs=fs, labels=not alt)
        by_id = {pool_rows[t[3]]["id"]: t for t in idx}
        for name in ("train_full", "train", "val"):
            rows = list(csv.DictReader(open(ROOT / "train" / f"{name}.csv")))
            sel = [by_id[r["id"]] for r in rows if r["id"] in by_id]
            if not alt:
                write_idx(out / f"{name}_idx.bin", sel)
            print(f"{name}: {len(sel)} chains ({len(rows) - len(sel)} without files yet), {sum(t[2] for t in sel):,} residues")
    # the EUs, in eu_list.csv order; missing ones (no structure) are skipped and listed
    eus = list(csv.DictReader(open(ROOT / "eu_list.csv")))
    tr = [e for e in eus if (ROOT / "targets" / f"{e['eu']}.npz").exists()]
    t_feat = (lambda e: onehot(str(np.load(ROOT / "targets" / f"{e['eu']}.npz")["seq"]))) if a.features == "onehot" else \
             (lambda e: np.load(ROOT / tdir / f"{e['eu']}.npz")["emb"])
    if "targets" in sets:
        tidx = pack(tr, t_feat, lambda e, L: np.load(ROOT / "targets" / f"{e['eu']}.npz")["cls"], out, "targets",
                    width=width, fs=fs, labels=not alt)
        if not alt:
            write_idx(out / "targets_idx.bin", tidx)
            with open(out / "targets_order.txt", "w") as f:
                f.write("\n".join(e["eu"] for e in tr) + "\n")
        print(f"targets: {len(tr)} EUs (skipped {[e['eu'] for e in eus if e not in tr]})")
    # a val subset in the targets format: fold hyperparameters get tuned here, not on the EUs
    if a.val_targets and ("valsub" in sets or a.val_name in sets):
        lo, hi = a.val_len
        vrows = sorted(csv.DictReader(open(ROOT / "train" / "val.csv")), key=lambda r: int(r["length"]))
        vrows = [r for r in vrows if lo <= int(r["length"]) <= hi and have(r)]
        vrows = vrows[:: max(1, len(vrows) // a.val_targets)][: a.val_targets]   # a spread of lengths
        order_f = out / f"{a.val_name}_order.txt"
        if alt and order_f.exists():                       # an alternative feature set follows the set's frozen order
            ids = order_f.read_text().split(); by = {r["id"]: r for r in vrows}
            vrows = [by[i] for i in ids]
        vidx = pack(vrows, feat_of, lab_of, out, a.val_name, width=width, fs=fs, labels=not alt)
        if not alt:
            write_idx(out / f"{a.val_name}_idx.bin", vidx)
            with open(order_f, "w") as f:
                f.write("\n".join(r["id"] for r in vrows) + "\n")
        print(f"{a.val_name}: {len(vrows)} val chains of {lo}–{hi} residues in the targets format")
    tot = sum(f.stat().st_size for f in out.glob("*.bin"))
    print(f"-> {out}: {tot / 1e9:.2f} GB in {time.time() - t0:.0f} s; feature set {a.features}, width {width + NPOS}, "
          f"label classes 66, index row = 4 × i64 (feat off, lab off, L, csv row)")

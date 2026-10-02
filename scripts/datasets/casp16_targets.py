#!/usr/bin/env python3
"""Step 6 of planning/casp16_distogram_demo.md §3: the CASP16 evaluation units as the model will
see and be scored on them -> data/casp16/targets/<EU>.npz with
    seq, resnum   the EU's residues (eu_list.csv segments of the Phase-1 target sequence)
    emb           ESM-2 35M representation of the FULL target sequence, sliced to the EU (f16)
    cls, obs, cb  the same labels as the training set, from raw/dom/<EU>.pdb (target numbering)
    esm_contacts  ESM-2's own contact head on the full sequence, sliced (the "ESM-2 alone" row)
Prints per EU the residue-name check against the sequence, the fraction observed, and the
contact head's top-L/5 long-range precision (|i-j| >= 24, Cβ < 8 Å), which is the first
baseline number of the table."""
import csv, os, re, sys
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from casp16_embed import load_model, embed_batch
from casp16_labels import labels, orient_labels, virtual_cb, EDGES, FAR, UNOBS

ROOT = Path(os.environ.get("CASP16_DIR", Path(__file__).resolve().parents[2] / "data" / "casp16"))
AA3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
       "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
       "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V", "MSE": "M"}


def read_targets():
    seqs, name = {}, None
    for line in open(ROOT / "raw" / "casp16.T1.seq.txt"):
        if line.startswith(">"):
            name = line[1:].split()[0]; seqs[name] = ""
        elif name:
            seqs[name] += line.strip()
    return seqs


def pdb_backbone(path, resnum):
    """N, Cα, Cβ as [L, 3] in `resnum` order (NaN where absent); Cβ deposited or virtual, as
    `casp16_labels.backbone` builds it for the training chains."""
    atoms = {}
    for line in open(path):
        if not line.startswith("ATOM") or line[16] not in " A":
            continue
        n = int(line[22:26]); atom = line[12:16].strip()
        if atom in ("N", "CA", "C", "CB"):
            atoms.setdefault((n, atom), np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])], np.float32))
    arr = {a: np.full((len(resnum), 3), np.nan, np.float32) for a in ("N", "CA", "C", "CB")}
    for k, n in enumerate(resnum):
        for a in arr:
            if (n, a) in atoms:
                arr[a][k] = atoms[(n, a)]
    cb = np.where(np.isnan(arr["CB"][:, :1]), virtual_cb(arr["N"], arr["CA"], arr["C"]), arr["CB"])
    return arr["N"], arr["CA"], cb


def pdb_cb(path):
    """auth resnum -> (resname, Cβ or Cα-for-Gly coordinates); first altloc."""
    ca, cb, names = {}, {}, {}
    for line in open(path):
        if not line.startswith("ATOM") or line[16] not in " A":
            continue
        n = int(line[22:26]); atom = line[12:16].strip()
        xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])], np.float32)
        names[n] = line[17:20]
        if atom == "CA": ca.setdefault(n, xyz)
        if atom == "CB": cb.setdefault(n, xyz)
    return {n: (names[n], cb.get(n, ca.get(n))) for n in names if n in ca or n in cb}


def top_l5_precision(score, cls, obs, sep=24, frac=0.2):
    """Contact precision the CASP RR way: rank pairs with |i-j| >= sep by score, take L/5,
    count those whose true Cβ distance is under 8 Å. Pairs with an unobserved residue are skipped."""
    L = len(obs)
    i, j = np.triu_indices(L, sep)
    keep = obs[i] & obs[j]
    i, j = i[keep], j[keep]
    order = np.argsort(-score[i, j])[: max(1, round(L * frac))]
    true = (cls[i[order], j[order]] < FAR) & (EDGES[np.minimum(cls[i[order], j[order]], 63)] < 8.0)
    return float(true.mean()), len(order)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="esm2_t12_35M_UR50D")
    ap.add_argument("--out", default="targets", help="output dir under data/casp16 (targets_esm150 for the 150M model)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--orient", action="store_true",
                    help="add the ω, θ, φ planes (casp16_labels.orient_labels) to every existing <out>/<EU>.npz and stop")
    a = ap.parse_args()
    if a.orient:
        out = ROOT / a.out; n = 0; binned = []
        for f in sorted(out.glob("*.npz")):
            t = dict(np.load(f))
            N, CA, cb = pdb_backbone(ROOT / "raw" / "dom" / f"{f.stem}.pdb", t["resnum"])
            om, th, ph = orient_labels(N, CA, cb)
            assert om.shape == t["cls"].shape, (f.stem, om.shape, t["cls"].shape)
            # the distance labels' Cβ (Cα for glycine) and this Cβ agree wherever a Cβ was deposited
            both = ~np.isnan(cb[:, 0]) & ~np.isnan(t["cb"][:, 0]) & np.array([c != "G" for c in str(t["seq"])])
            assert np.nanmax(np.abs(cb[both] - t["cb"][both])) < 1e-3 if both.any() else True, f.stem
            np.savez_compressed(f, **{k: v for k, v in t.items() if k not in ("omega", "theta", "phi")}, omega=om, theta=th, phi=ph)
            n += 1; binned.append((om < 24).mean())
        print(f"{n} EUs given orientation planes in {out}; binned pairs {np.mean(binned):.3f} of all")
        sys.exit(0)
    torch.set_num_threads(16)
    seqs = read_targets()
    eus = list(csv.DictReader(open(ROOT / "eu_list.csv")))
    out = ROOT / a.out; out.mkdir(exist_ok=True)
    model, alphabet = load_model(a.model, a.device)
    cache = {}
    print(f"{'EU':12s} {'L':>4s} {'obs':>5s} {'mismatch':>8s} {'ESM-2 head P@L/5':>17s}  difficulty")
    rows = []
    for e in eus:
        t = e["target"]
        if t not in seqs and re.sub(r"v\d+$", "v1", t) in seqs:
            t = re.sub(r"v\d+$", "v1", t)  # T1228v2 is T1228v1's sequence in another conformation
        if t not in seqs:
            print(f"{e['eu']:12s} no sequence for {t}"); continue
        if not (ROOT / "raw" / "dom" / f"{e['eu']}.pdb").exists():
            print(f"{e['eu']:12s} no domain structure (never scored)"); continue
        if t not in cache:
            cache[t] = embed_batch(model, alphabet, [(t, seqs[t])], contacts=True)[0]
        emb_full, con_full = cache[t]
        segs = [tuple(map(int, s.split("-"))) for s in e["segments"].split(",")]
        resnum = np.array([n for a, b in segs for n in range(a, b + 1)])
        idx = resnum - 1
        seq = "".join(seqs[t][k] for k in idx)
        pdb = pdb_cb(ROOT / "raw" / "dom" / f"{e['eu']}.pdb")
        cb = np.full((len(idx), 3), np.nan, np.float32); mism = 0
        for k, n in enumerate(resnum):
            if n in pdb:
                name, xyz = pdb[n]
                if xyz is not None:
                    cb[k] = xyz
                if AA3.get(name, "X") != seq[k]:
                    mism += 1
        cls, obs, _ = labels(cb)
        con = con_full[np.ix_(idx, idx)].astype(np.float32)
        prec, n_top = top_l5_precision(con, cls, obs)
        np.savez_compressed(out / f"{e['eu']}.npz", seq=seq, resnum=resnum, emb=emb_full[idx], cls=cls, obs=obs,
                            cb=cb, esm_contacts=con.astype(np.float16))
        rows.append(dict(eu=e["eu"], L=len(idx), obs=float(obs.mean()), mismatch=mism, esm_p=prec, difficulty=e["difficulty"]))
        print(f"{e['eu']:12s} {len(idx):4d} {obs.mean():5.2f} {mism:8d} {prec:17.3f}  {e['difficulty']}")
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    P = np.array([r["esm_p"] for r in rows])
    for d in ("easy", "medium", "hard"):
        sel = [r["esm_p"] for r in rows if r["difficulty"] == d]
        print(f"ESM-2 contact head, top-L/5 long-range precision, {d}: mean {np.mean(sel):.3f} over {len(sel)} EUs")
    print(f"all {len(rows)} EUs: mean {P.mean():.3f}, median {np.median(P):.3f}; residues observed "
          f"{np.mean([r['obs'] for r in rows]):.3f}; name mismatches {sum(r['mismatch'] for r in rows)}")

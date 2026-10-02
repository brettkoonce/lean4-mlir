#!/usr/bin/env python3
"""Fold-hyperparameter bench on VAL chains, so nothing is tuned on the CASP16 EUs.

Input: `lake exe distogram-casp predict … pool=valsub` output (<dir>/<id>.acc.bin / .cnt.bin for
the 24 val chains casp16_pack.py --val-targets packs). For each setting of (reference state,
w_chain, w_clash, σ) every chain is folded (casp16_fold.fold) and scored against its own true
pseudo-Cβ trace (labels/<id>.npz `cb`): Cβ-lDDT (the numpy lddt) and TM-score (US-align
-TMscore 1), mean over chains. Prints one line per setting and writes <dir>/tune.csv."""
import argparse, csv, itertools, json, os, sys, time
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "datasets"))
from casp16_fold import fold, write_pdb, set_device, SIGMA, ROOT
from casp16_predict import assemble, NC, ORIENT
from casp16_score import lddt, read_atoms, usalign_tm

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("--restarts", type=int, default=2)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--name", default="valsub", help="the val subset (packed/<name>_order.txt): valsub, valsub_long")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda (the bench ran on CPU before this flag existed: same numbers, 5–20× slower)")
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--orient", action="store_true", help="fold with the prediction's ω / φ heads too (casp16_fold --orient)")
    ap.add_argument("--w-orient", type=float, default=1.0)
    ap.add_argument("--grid", default="ref:1,0;chain:1,0.3,3;clash:3,1,10;sigma:1,2",
                    help="semicolon-separated name:values; the first value of each is the default when the others vary "
                         "(with --orient, `orient:1,0.3,3` sweeps the restraint weight)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    print(f"device {set_device(a.device)}", flush=True)
    d = Path(a.dir)
    ids = open(ROOT / "packed" / f"{a.name}_order.txt").read().split()
    seqs = {}
    with open(ROOT / "train" / "entities.jsonl") as f:
        for line in f:
            e = json.loads(line)
            if e["id"] in ids:
                seqs[e["id"]] = e["seq"]
    ref = np.load(ROOT / "packed" / "ref_by_sep.npy")
    oref = np.load(ROOT / "packed" / "ref_orient_by_sep.npz") if a.orient else None
    chains = []
    for cid in ids:
        lab = np.load(ROOT / "labels" / f"{cid}.npz")
        L = len(lab["obs"])
        acc = np.fromfile(d / f"{cid}.acc.bin", np.float32); cnt = np.fromfile(d / f"{cid}.cnt.bin", np.float32).reshape(L, L)
        total = acc.size // (L * L); acc = acc.reshape(L, L, total)
        probs = assemble(acc[:, :, :NC], cnt, L)
        orient = None
        if a.orient:
            assert total > NC, f"{cid}: no orientation heads in the prediction ({total} channels)"
            orient, off = {}, NC
            for name, n, sym in ORIENT:
                orient[name] = assemble(acc[:, :, off:off + n], cnt, L, n, sym); off += n
        refp = d / f"{cid}.ref.pdb"
        cb = lab["cb"]; keep = ~np.isnan(cb[:, 0])
        write_pdb(refp, np.nan_to_num(cb)[keep], "".join(c for c, k in zip(seqs[cid], keep) if k), np.arange(1, L + 1)[keep])
        chains.append((cid, probs, refp, seqs[cid], L, orient))
    grid = {k: [float(x) for x in v.split(",")] for k, v in (g.split(":") for g in a.grid.split(";"))}
    base = {k: v[0] for k, v in grid.items()}
    settings = [dict(base)] + [dict(base, **{k: x}) for k, vs in grid.items() for x in vs[1:]]
    rows = []
    print(f"{'ref':>4s} {'chain':>6s} {'clash':>6s} {'sigma':>6s} {'orient':>6s} {'steps':>5s} {'lr':>5s} {'Cβ-lDDT':>8s} {'TM':>6s} {'hand ok':>8s}  s")
    for st in settings:
        t0 = time.time(); ld, tm, hands = [], [], []
        for cid, probs, refp, seq, L, orient in chains:
            x, xm, _ = fold(probs, a.restarts, int(st.get("steps", a.steps)), ref=ref if st["ref"] else None,
                            w_chain=st["chain"], w_clash=st["clash"], sigma=SIGMA * st["sigma"], lr=st.get("lr", a.lr),
                            orient=orient, orient_ref=oref if st["ref"] else None, w_orient=st.get("orient", a.w_orient))
            mp = d / f"{cid}.tune.pdb"; write_pdb(mp, x, seq, np.arange(1, L + 1))
            mm = d / f"{cid}.tune_m.pdb"; write_pdb(mm, xm, seq, np.arange(1, L + 1))
            ld.append(lddt(read_atoms(mp, True), read_atoms(refp, True)))
            t1, t2 = usalign_tm(mp, refp), usalign_tm(mm, refp)
            tm.append(t1); hands.append(t1 >= t2)
        row = dict(st, lddt=float(np.mean(ld)), tm=float(np.mean(tm)), hand_ok=float(np.mean(hands)), secs=time.time() - t0)
        rows.append(row)
        print(f"{int(row['ref']):>4d} {row['chain']:>6.2f} {row['clash']:>6.2f} {row['sigma']:>6.2f} {row.get('orient', a.w_orient if a.orient else 0.0):>6.2f} "
              f"{int(row.get('steps', a.steps)):>5d} {row.get('lr', a.lr):>5.2f} {row['lddt']:>8.3f} {row['tm']:>6.3f} {row['hand_ok']:>8.2f}  {row['secs']:.0f}", flush=True)
    with open(d / "tune.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"-> {d / 'tune.csv'}")

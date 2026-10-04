#!/usr/bin/env python3
"""ESMFold on the CASP16 targets: the reference row "the same language model under its published
folding head" of planning/casp16_distogram_demo.md §10. ESMFold v1 is ESM-2 3B — the LM the book
config reads — under a 48-block folding trunk and a structure module, trained end to end. It is
given the Phase-1 target sequence, as our features are (casp16_targets.py), and its model is
scored per evaluation unit by the scorer our folds go through (casp16_score.score at pseudo-Cβ),
so its Cβ-lDDT / TM columns sit beside casp16_fold_score.py's.

    <out>/<target>.pdb      the full-chain model, residues numbered from 1
    <out>/fold_scores.csv   eu, L, difficulty, cb_lddt, tm, plddt (mean Cα pLDDT over the EU)

Runs under .venv-casp with `transformers` (facebook/esmfold_v1 is downloaded on first use). On a
card the LM runs in fp16 and the trunk's axial attention in chunks; a target that does not fit is
retried at smaller chunks and then skipped with a note — `--device cpu --only <targets>` folds
those. A target whose model is on disk is not folded again; `--score-only` rescans the directory."""
import argparse, csv, re, sys, time
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "datasets"))
from casp16_score import ROOT, eu_ranges, score
from casp16_targets import read_targets


def seq_target(t, seqs):
    """The target whose sequence `t` is folded from: T1228v2 is T1228v1's sequence in another conformation."""
    v1 = re.sub(r"v\d+$", "v1", t)
    return t if t in seqs else v1 if v1 in seqs else None


def fold(model, tok, seq, device, chunks):
    """(pdb text, per-residue Cα pLDDT on 0–100, chunk size used), or None when no chunk size fits the card."""
    inp = tok([seq], return_tensors="pt", add_special_tokens=False).to(device)
    for c in chunks:
        try:
            model.trunk.set_chunk_size(c)
            with torch.no_grad():
                out = model(**inp)
            pl = out["plddt"][0, :, 1].float().cpu().numpy()
            return model.output_to_pdb(out)[0], pl * (100.0 if pl.max() <= 1.0 else 1.0), c
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
    return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="esmfold", help="directory under data/casp16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only", nargs="*", help="fold these targets only")
    ap.add_argument("--chunks", nargs="*", type=int, default=[64, 16, 4], help="axial-attention chunk sizes to try, in order")
    ap.add_argument("--score-only", action="store_true")
    a = ap.parse_args()
    out = ROOT / a.out; out.mkdir(exist_ok=True)
    seqs, eus = read_targets(), eu_ranges()
    meta = {r["eu"]: r for r in csv.DictReader(open(ROOT / "targets" / "summary.csv"))}
    todo = sorted({seq_target(eus[eu]["target"], seqs) for eu in meta} - {None}, key=lambda t: len(seqs[t]))
    todo = [t for t in todo if (not a.only or t in a.only) and not (out / f"{t}.pdb").exists()]
    if todo and not a.score_only:
        from transformers import AutoTokenizer, EsmForProteinFolding
        tok = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True).eval().to(a.device)
        if a.device != "cpu":
            model.esm = model.esm.half()
        for t in todo:
            t0 = time.time()
            r = fold(model, tok, seqs[t], a.device, a.chunks)
            if r is None:
                print(f"{t:10s} L {len(seqs[t]):4d}  does not fit the card at chunk {a.chunks[-1]} — fold it with --device cpu --only {t}", flush=True)
                continue
            pdb, pl, c = r
            (out / f"{t}.pdb").write_text(pdb); np.save(out / f"{t}.plddt.npy", pl.astype(np.float32))
            print(f"{t:10s} L {len(seqs[t]):4d}  pLDDT {pl.mean():5.1f}  chunk {c:2d}  {time.time() - t0:6.1f} s", flush=True)
    rows = []
    for eu in sorted(meta):
        t = seq_target(eus[eu]["target"], seqs)
        if t is None or not (out / f"{t}.pdb").exists():
            continue
        r = score(out / f"{t}.pdb", eu, eus=eus, tag=f"{a.out}-{eu}", pseudo_cb=True)
        idx = np.array([n for s, e in eus[eu]["segs"] for n in range(s, e + 1)]) - 1
        rows.append(dict(eu=eu, L=int(meta[eu]["L"]), difficulty=meta[eu]["difficulty"], cb_lddt=round(r["ca_lddt"], 4),
                         tm=round(r["tm"], 4), plddt=round(float(np.load(out / f"{t}.plddt.npy")[idx].mean()), 1)))
        print(f"{eu:12s} L {rows[-1]['L']:4d} {rows[-1]['difficulty']:6s} Cβ-lDDT {rows[-1]['cb_lddt']:.3f} TM {rows[-1]['tm']:.3f} pLDDT {rows[-1]['plddt']:5.1f}", flush=True)
    if rows:
        with open(out / "fold_scores.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        for diff in ("easy", "medium", "hard", None):
            sel = [r for r in rows if diff is None or r["difficulty"] == diff]
            if sel:
                print(f"{diff or 'all':6s}: Cβ-lDDT mean {np.mean([r['cb_lddt'] for r in sel]):.3f} median {np.median([r['cb_lddt'] for r in sel]):.3f}"
                      f"  TM mean {np.mean([r['tm'] for r in sel]):.3f} median {np.median([r['tm'] for r in sel]):.3f}  ({len(sel)} EUs)")
        print(f"-> {out / 'fold_scores.csv'}")

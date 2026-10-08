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
retried at smaller chunks and then, with `--window W`, folded per evaluation unit from a W-residue
window of its sequence centred on the unit (clamped to the chain; units sharing a window are folded
once; `<out>/<target>.w<start>-<end>.pdb`, residues numbered as in the target) — T1218 and T1269,
over 1,000 residues, need W = 800 on a 16 GB card — or skipped with a note (`--device cpu --only
<targets>` folds those whole). A target whose model is on disk is not folded again; `--score-only`
rescans the directory. The score rows say which model each unit came from (`window`)."""
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


def windows(units, L, W):
    """One W-residue window per unit of a target too long for the card, the unit's segments
    centred in it and the window clamped to the chain, as {eu: (start, end)}, 1-based inclusive."""
    out = {}
    for eu, segs in units:
        lo, hi = min(s for s, _ in segs), max(e for _, e in segs)
        start = lo if hi - lo + 1 > W else max(1, min((lo + hi) // 2 - W // 2, L - W + 1))
        out[eu] = (start, min(start + W - 1, L))
    return out


def renumber(pdb, offset):
    """Shift every ATOM record's residue number by `offset`, so a window's model carries the target's numbering."""
    out = []
    for ln in pdb.splitlines():
        if ln.startswith(("ATOM", "HETATM")):
            ln = ln[:22] + f"{int(ln[22:26]) + offset:4d}" + ln[26:]
        out.append(ln)
    return "\n".join(out) + "\n"


def window_model(out, t, segs):
    """The window model of target `t` that covers every segment of a unit — the one centred nearest
    the unit when several do — or None: (path, start)."""
    lo, hi = min(s for s, _ in segs), max(e for _, e in segs)
    best = None
    for f in out.glob(f"{t}.w*.pdb"):
        a, b = map(int, re.search(r"\.w(\d+)-(\d+)\.pdb$", f.name).groups())
        if all(a <= s and e <= b for s, e in segs):
            off = abs((a + b) - (lo + hi))
            if best is None or off < best[0]:
                best = (off, f, a)
    return None if best is None else (best[1], best[2])


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
    ap.add_argument("--window", type=int, default=0, help="fold a target that does not fit per unit from a window of this many residues (0 = skip it)")
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
            if r is None and a.window:
                units = [(eu, eus[eu]["segs"]) for eu in sorted(meta) if seq_target(eus[eu]["target"], seqs) == t]
                for (s0, e0) in sorted(set(windows(units, len(seqs[t]), a.window).values())):
                    f = out / f"{t}.w{s0}-{e0}.pdb"
                    if f.exists():
                        continue
                    t1 = time.time()
                    rw = fold(model, tok, seqs[t][s0 - 1:e0], a.device, a.chunks)
                    if rw is None:
                        print(f"{t:10s} window {s0}-{e0} does not fit the card either", flush=True); continue
                    pdb, pl, c = rw
                    f.write_text(renumber(pdb, s0 - 1)); np.save(out / f"{t}.w{s0}-{e0}.plddt.npy", pl.astype(np.float32))
                    print(f"{t:10s} L {len(seqs[t]):4d}  window {s0}-{e0}  pLDDT {pl.mean():5.1f}  chunk {c:2d}  {time.time() - t1:6.1f} s", flush=True)
                continue
            if r is None:
                print(f"{t:10s} L {len(seqs[t]):4d}  does not fit the card at chunk {a.chunks[-1]} — fold it with --window 800, or --device cpu --only {t}", flush=True)
                continue
            pdb, pl, c = r
            (out / f"{t}.pdb").write_text(pdb); np.save(out / f"{t}.plddt.npy", pl.astype(np.float32))
            print(f"{t:10s} L {len(seqs[t]):4d}  pLDDT {pl.mean():5.1f}  chunk {c:2d}  {time.time() - t0:6.1f} s", flush=True)
    rows = []
    for eu in sorted(meta):
        t = seq_target(eus[eu]["target"], seqs)
        if t is None:
            continue
        model_f, start = (out / f"{t}.pdb", 1) if (out / f"{t}.pdb").exists() else (window_model(out, t, eus[eu]["segs"]) or (None, 1))
        if model_f is None:
            continue
        r = score(model_f, eu, eus=eus, tag=f"{a.out}-{eu}", pseudo_cb=True)
        idx = np.array([n for s, e in eus[eu]["segs"] for n in range(s, e + 1)]) - start
        rows.append(dict(eu=eu, L=int(meta[eu]["L"]), difficulty=meta[eu]["difficulty"], cb_lddt=round(r["ca_lddt"], 4),
                         tm=round(r["tm"], 4), plddt=round(float(np.load(model_f.with_suffix(".plddt.npy"))[idx].mean()), 1),
                         window="full" if start == 1 and model_f.name == f"{t}.pdb" else model_f.name.split(".")[-2][1:]))
        print(f"{eu:12s} L {rows[-1]['L']:4d} {rows[-1]['difficulty']:6s} Cβ-lDDT {rows[-1]['cb_lddt']:.3f} TM {rows[-1]['tm']:.3f} pLDDT {rows[-1]['plddt']:5.1f}  {rows[-1]['window']}", flush=True)
    if rows:
        with open(out / "fold_scores.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        for diff in ("easy", "medium", "hard", None):
            sel = [r for r in rows if diff is None or r["difficulty"] == diff]
            if sel:
                print(f"{diff or 'all':6s}: Cβ-lDDT mean {np.mean([r['cb_lddt'] for r in sel]):.3f} median {np.median([r['cb_lddt'] for r in sel]):.3f}"
                      f"  TM mean {np.mean([r['tm'] for r in sel]):.3f} median {np.median([r['tm'] for r in sel]):.3f}  ({len(sel)} EUs)")
        print(f"-> {out / 'fold_scores.csv'}")

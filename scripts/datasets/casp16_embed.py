#!/usr/bin/env python3
"""Step 5 of planning/casp16_distogram_demo.md §3: frozen ESM-2 35M (esm2_t12_35M_UR50D, Lin et
al. 2022; UniRef50 2021_04, sequence-only, so nothing after the CASP16 cutoff) per-residue
representations for every chain in the training list -> data/casp16/emb/<entity>.npy, f16
[L, 480], the final layer with BOS/EOS stripped. CPU is enough for 35M parameters (set threads
with --threads). The representations, not the model's contact head: that head is the
"ESM-2 alone" baseline row of the table and is computed in casp16_targets.py."""
import argparse, csv, json, os, time
from pathlib import Path
import numpy as np, torch

ROOT = Path(os.environ.get("CASP16_DIR", Path(__file__).resolve().parents[2] / "data" / "casp16"))
LAYER = 12


def load_model():
    import esm
    model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    model.eval()
    return model, alphabet


@torch.no_grad()
def embed_batch(model, alphabet, named_seqs, contacts=False):
    """[(name, seq)] -> list of f16 [L, 480]; with contacts=True also the [L, L] contact map."""
    _, _, toks = alphabet.get_batch_converter()(named_seqs)
    out = model(toks, repr_layers=[LAYER], return_contacts=contacts)
    rep = out["representations"][LAYER]
    res = []
    for i, (_, s) in enumerate(named_seqs):
        r = rep[i, 1:len(s) + 1].to(torch.float16).numpy()
        res.append((r, out["contacts"][i, :len(s), :len(s)].to(torch.float16).numpy()) if contacts else r)
    return res


def batches(rows, max_tokens):
    rows = sorted(rows, key=lambda r: len(r[1]))
    cur, cur_tok = [], 0
    for r in rows:
        L = len(r[1]) + 2
        if cur and (len(cur) + 1) * max(L, cur_tok) > max_tokens:
            yield cur; cur, cur_tok = [], 0
        cur.append(r); cur_tok = max(cur_tok, L)
    if cur:
        yield cur


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list", default=str(ROOT / "train" / "reps.csv"))
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--max-tokens", type=int, default=16384, help="padded tokens per batch")
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    out = ROOT / "emb"; out.mkdir(exist_ok=True)
    seqs = {}
    with open(ROOT / "train" / "entities.jsonl") as f:
        for line in f:
            e = json.loads(line)
            if e["seq"]:
                seqs[e["id"]] = e["seq"]
    rows = [(r["id"], seqs[r["id"]]) for r in csv.DictReader(open(a.list)) if r["id"] in seqs]
    if a.limit:
        rows = rows[: a.limit]
    todo = [r for r in rows if not (out / f"{r[0]}.npy").exists()]
    print(f"{len(rows)} chains, {len(rows) - len(todo)} done, {len(todo)} to embed", flush=True)
    model, alphabet = load_model()
    t0, n_tok, n = time.time(), 0, 0
    for b in batches(todo, a.max_tokens):
        reps = embed_batch(model, alphabet, b)
        for (name, s), r in zip(b, reps):
            np.save(out / f"{name}.npy", r)
            n_tok += len(s); n += 1
        if n % 1000 < len(b):
            dt = time.time() - t0
            print(f"  {n}/{len(todo)}  {n_tok / dt:.0f} tok/s  eta {(len(todo) - n) * (n_tok / max(n, 1)) / max(n_tok / dt, 1) / 60:.0f} min", flush=True)
    print(f"{n} chains, {n_tok} residues in {time.time() - t0:.0f} s -> {out} "
          f"({sum(f.stat().st_size for f in out.glob('*.npy')) / 1e9:.2f} GB)")

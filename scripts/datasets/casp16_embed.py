#!/usr/bin/env python3
"""Step 5 of planning/casp16_distogram_demo.md §3: frozen ESM-2 35M (esm2_t12_35M_UR50D, Lin et
al. 2022; UniRef50 2021_04, sequence-only, so nothing after the CASP16 cutoff) per-residue
representations for every chain in the training list -> data/casp16/emb/<entity>.npy, f16
[L, 480], the final layer with BOS/EOS stripped. CPU is enough for 35M parameters (set threads
with --threads). The representations, not the model's contact head: that head is the
"ESM-2 alone" baseline row of the table and is computed in casp16_targets.py.
`--model esm2_t36_3B_UR50D --half --device cuda --shard k/n`: the 3B model in fp16 (5.6 GB of
weights, so it fits a 16 GB card) with the pool's chains split k-of-n across cards; each shard
keeps its own `.done.k` list and writes its rows at their offsets of the one pool file.
`--pair-out packed/pool_pair_<fs>.bin`: the model's own contact head over the pool, one u8
plane per chain at its pair offset (Σ L² in pool order, the label pool's layout), byte =
clip(128 + 16·logit, 0, 255) — the `pair=1` input of the distogram net (`Layer.pairTile`'s
`pairIn`; the gather maps it back to logit/4). The contact head keeps every layer's attention
maps, so `--max-pairs` caps the padded B·L² of a batch."""
import argparse, csv, json, os, time
from pathlib import Path
import numpy as np, torch

ROOT = Path(os.environ.get("CASP16_DIR", Path(__file__).resolve().parents[2] / "data" / "casp16"))
LAYER = 12


MODELS = {"esm2_t12_35M_UR50D": 12, "esm2_t30_150M_UR50D": 30, "esm2_t33_650M_UR50D": 33,
          "esm2_t36_3B_UR50D": 36}
WIDTH = {12: 480, 30: 640, 33: 1280, 36: 2560}            # embedding dim by final layer


def load_model(name="esm2_t12_35M_UR50D", device="cpu", half=False):
    import esm
    model, alphabet = getattr(esm.pretrained, name)()
    model.eval()
    if half:
        model.half()
    model.to(device)
    global LAYER
    LAYER = MODELS[name]
    return model, alphabet


@torch.no_grad()
def embed_batch(model, alphabet, named_seqs, contacts=False):
    """[(name, seq)] -> list of f16 [L, D]; with contacts=True also the [L, L] contact map."""
    _, _, toks = alphabet.get_batch_converter()(named_seqs)
    dev = next(model.parameters()).device
    out = model(toks.to(dev), repr_layers=[LAYER], return_contacts=contacts)
    rep = out["representations"][LAYER]
    res = []
    for i, (_, s) in enumerate(named_seqs):
        r = rep[i, 1:len(s) + 1].to(torch.float16).cpu().numpy()
        res.append((r, out["contacts"][i, :len(s), :len(s)].to(torch.float16).cpu().numpy()) if contacts else r)
    return res


def contact_logit_u8(p):
    """[L, L] contact probabilities -> u8 plane, byte = clip(128 + 16·logit)."""
    p = np.clip(p.astype(np.float32), 1e-6, 1 - 1e-6)
    return np.clip(np.rint(128 + 16 * (np.log(p) - np.log1p(-p))), 0, 255).astype(np.uint8)


def batches(rows, max_tokens, max_pairs=0):
    rows = sorted(rows, key=lambda r: len(r[1]))
    cur, cur_tok = [], 0
    for r in rows:
        L = len(r[1]) + 2
        if cur and ((len(cur) + 1) * max(L, cur_tok) > max_tokens
                    or (max_pairs and (len(cur) + 1) * max(L, cur_tok) ** 2 > max_pairs)):
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
    p.add_argument("--model", default="esm2_t12_35M_UR50D", choices=sorted(MODELS))
    p.add_argument("--out", default="emb", help="output dir under data/casp16 (emb150 for the 150M model)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--half", action="store_true", help="fp16 weights (the 3B model on a 16 GB card)")
    p.add_argument("--shard", default="0/1", metavar="K/N", help="embed every n-th chain starting at k (one process per card)")
    p.add_argument("--pair-out", default="",
                   help="write the pool's contact-head logit planes (u8 [L, L] per chain at its pair offset) to this "
                        "file: with --pool-out beside the features, alone just the planes")
    p.add_argument("--max-pairs", type=int, default=0, help="cap on padded B·L² per batch (the attention maps the contact head keeps; 0 = none)")
    p.add_argument("--pool-out", default="",
                   help="write the training pool's packed feature file (casp16_pack.py's pool rows and order, "
                        "position features appended) directly, with no per-chain .npy — for a model whose "
                        "per-chain files would not fit the disk; per-chain .npy are still written for the "
                        "valsub chains so the packer can build that set")
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    out = ROOT / a.out; out.mkdir(exist_ok=True)
    seqs = {}
    with open(ROOT / "train" / "entities.jsonl") as f:
        for line in f:
            e = json.loads(line)
            if e["seq"]:
                seqs[e["id"]] = e["seq"]
    pool_f = pair_f = None
    if a.pool_out or a.pair_out:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from casp16_pack import pos_feats, NPOS
        reps = list(csv.DictReader(open(ROOT / "train" / "reps.csv")))
        order_f = ROOT / "packed" / "pool_order.txt"      # the pool's frozen membership (casp16_pack.py)
        if order_f.exists():
            ids = set(order_f.read_text().split()); have = lambda r: r["id"] in ids
        else:
            have = lambda r: (ROOT / "emb" / f"{r['id']}.npy").exists() and (ROOT / "labels" / f"{r['id']}.npz").exists()
        pool_rows = [r for r in reps if have(r)]
        rows = [(r["id"], seqs[r["id"]]) for r in pool_rows]
        if a.limit:
            rows = rows[: a.limit]
        width = WIDTH[MODELS[a.model]] + NPOS
        offs, off = {}, 0
        for name, sq in [(r["id"], seqs[r["id"]]) for r in pool_rows]:
            offs[name] = off; off += len(sq)
        total = off
        pool_path = Path(a.pool_out or a.pair_out)       # the .done lists follow the file being written
        k, n = map(int, a.shard.split("/"))
        done_path = Path(str(pool_path) + (".done" if n == 1 else f".done.{k}"))
        done = set(w for f in pool_path.parent.glob(pool_path.name + ".done*") for w in f.read_text().split())
        def grown(path, size):
            # created without O_TRUNC and only ever grown, so n shards can open it at once
            fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
            if os.fstat(fd).st_size < size:
                os.ftruncate(fd, size)
            return os.fdopen(fd, "r+b")
        if a.pool_out:
            pool_f = grown(a.pool_out, total * width * 2)
        if a.pair_out:
            poffs, poff = {}, 0
            for name, sq in [(r["id"], seqs[r["id"]]) for r in pool_rows]:
                poffs[name] = poff; poff += len(sq) ** 2
            pair_f = grown(a.pair_out, poff)
        done_f = open(done_path, "a")
        valsub = set((ROOT / "packed" / "valsub_order.txt").read_text().split()) if (ROOT / "packed" / "valsub_order.txt").exists() else set()
        todo = [r for r in rows if r[0] not in done][k::n]
        print(f"pool: {len(pool_rows)} chains, {total:,} residues, width {width} -> {a.pool_out or '(no features)'}"
              f"{' + contact planes -> ' + a.pair_out if a.pair_out else ''}; "
              f"{len(done)} done, {len(todo)} to embed in shard {k}/{n}", flush=True)
    else:
        rows = [(r["id"], seqs[r["id"]]) for r in csv.DictReader(open(a.list)) if r["id"] in seqs]
        if a.limit:
            rows = rows[: a.limit]
        todo = [r for r in rows if not (out / f"{r[0]}.npy").exists()]
        print(f"{len(rows)} chains, {len(rows) - len(todo)} done, {len(todo)} to embed", flush=True)
    model, alphabet = load_model(a.model, a.device, a.half)
    t0, n_tok, n = time.time(), 0, 0
    for b in batches(todo, a.max_tokens, a.max_pairs):
        reps = embed_batch(model, alphabet, b, contacts=pair_f is not None)
        for (name, s), r in zip(b, reps):
            if pair_f is not None:
                r, con = r
                plane = contact_logit_u8(con)
                assert plane.shape == (len(s), len(s)), (name, plane.shape)
                pair_f.seek(poffs[name]); pair_f.write(plane.tobytes())
            if pool_f is not None:
                feat = np.concatenate([r.astype(np.float16), pos_feats(len(s))], 1)
                assert feat.shape == (len(s), width), (name, feat.shape)
                pool_f.seek(offs[name] * width * 2); pool_f.write(feat.tobytes())
                if name in valsub:
                    np.save(out / f"{name}.npy", r)
            if pool_f is not None or pair_f is not None:
                done_f.write(name + "\n"); done_f.flush()
            else:
                np.save(out / f"{name}.npy", r)
            n_tok += len(s); n += 1
        if n % 1000 < len(b):
            dt = time.time() - t0
            print(f"  {n}/{len(todo)}  {n_tok / dt:.0f} tok/s  eta {(len(todo) - n) * (n_tok / max(n, 1)) / max(n_tok / dt, 1) / 60:.0f} min", flush=True)
    if pool_f is not None or pair_f is not None:
        for f in (pool_f, pair_f, done_f):
            if f is not None:
                f.close()
        outs = [f for f in (a.pool_out, a.pair_out) if f]
        print(f"{n} chains, {n_tok} residues in {time.time() - t0:.0f} s -> "
              + ", ".join(f"{o} ({Path(o).stat().st_size / 1e9:.2f} GB)" for o in outs))
    else:
        print(f"{n} chains, {n_tok} residues in {time.time() - t0:.0f} s -> {out} "
              f"({sum(f.stat().st_size for f in out.glob('*.npy')) / 1e9:.2f} GB)")

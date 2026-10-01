#!/usr/bin/env python3
"""Step 2 of planning/casp16_distogram_demo.md §3: coordinates for every representative chain in
data/casp16/train/reps.csv, kept as data/casp16/pdb/<entity>.npz — backbone N, CA, C, O and Cβ
of the entity's first chain, indexed by label_seq_id, which numbers the entity's canonical
sequence and so lines coordinates up with the ESM-2 embedding of step 5 with no alignment.

Source: the entry's mmCIF from the RCSB CDN (files.rcsb.org, ~0.2 s), parsed with gemmi and cut
to the one chain; entries whose file exceeds --big MB (ribosomes, capsids) come instead from the
ModelServer as a single-chain query (slow, ~8 s, so only for those; the CDN sends no
Content-Length, so the entry file is read before the size is known). Resumable; failures go to
pdb/failed.txt. ~27.7k chains, ~20 KB each on disk."""
import argparse, concurrent.futures, csv, gzip, json, os, time, urllib.parse, urllib.request
from pathlib import Path
import gemmi, numpy as np

ROOT = Path(os.environ.get("CASP16_DIR", Path(__file__).resolve().parents[2] / "data" / "casp16"))
OUT = ROOT / "pdb"
CDN = "https://files.rcsb.org/download/{entry}.cif.gz"
MODELSERVER = "https://models.rcsb.org/v1/{entry}/atoms?label_asym_id={asym}&encoding=cif"
ATOMS = {"N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4}
UA = {"User-Agent": "lean4-jax-mlir casp16 demo"}


def get(url, method="GET", timeout=180):
    req = urllib.request.Request(url, headers=UA, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read() if method == "GET" else int(r.headers.get("Content-Length") or 0)


def extract(cif_text, asym):
    """The subchain's backbone + Cβ as flat arrays (first altloc of each atom)."""
    doc = gemmi.cif.read_string(cif_text)
    st = gemmi.make_structure_from_block(doc.sole_block())
    span = st[0].get_subchain(asym)
    if len(span) == 0:
        raise ValueError(f"no subchain {asym}")
    seq, atom, xyz, res_seq, res_name = [], [], [], [], []
    for res in span:
        if res.label_seq is None:
            continue
        res_seq.append(res.label_seq); res_name.append(res.name)
        seen = set()
        for at in res:
            code = ATOMS.get(at.name)
            if code is None or code in seen:
                continue
            seen.add(code)
            seq.append(res.label_seq); atom.append(code); xyz.append([at.pos.x, at.pos.y, at.pos.z])
    return dict(label_seq=np.array(seq, np.int16), atom=np.array(atom, np.uint8),
                xyz=np.array(xyz, np.float32), res_seq=np.array(res_seq, np.int16),
                res_name=np.array(res_name))


def fetch(row, big):
    dst = OUT / f"{row['id']}.npz"
    if dst.exists():
        return "skip", 0, None
    err = None
    for i in range(3):
        try:
            data = get(CDN.format(entry=row["entry"]))
            size = len(data)
            if size > big * 1e6:  # the CDN sends no Content-Length, so the cap is checked after the fact
                text = get(MODELSERVER.format(entry=row["entry"], asym=urllib.parse.quote(row["asym"]))).decode()
                how = "ms"
            else:
                text = gzip.decompress(data).decode()
                how = "cdn"
            d = extract(text, row["asym"])
            tmp = dst.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, **d)
            os.replace(tmp, dst)
            return how, size, None
        except Exception as e:
            err = e
            time.sleep(3 * (i + 1))
    return "fail", 0, f"{row['id']}\t{row['entry']}\t{row['asym']}\t{err}"


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list", default=str(ROOT / "train" / "reps.csv"))
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--big", type=float, default=20, help="MB; larger entry files go via the ModelServer")
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    asym = {}
    with open(ROOT / "train" / "entities.jsonl") as f:
        for line in f:
            e = json.loads(line)
            if e.get("chains"):
                asym[e["id"]] = e["chains"][0]
    rows = [dict(r, asym=asym[r["id"]]) for r in csv.DictReader(open(a.list)) if r["id"] in asym]
    if a.limit:
        rows = rows[: a.limit]
    t0, n, dl, fails = time.time(), {"cdn": 0, "ms": 0, "skip": 0, "fail": 0}, 0, []
    with concurrent.futures.ProcessPoolExecutor(a.workers) as ex:  # gemmi parsing is CPU-bound
        for i, (st, size, msg) in enumerate(ex.map(fetch, rows, [a.big] * len(rows), chunksize=4), 1):
            n[st] += 1; dl += size
            if msg:
                fails.append(msg)
            if i % 500 == 0 or i == len(rows):
                done = i - n["skip"]
                rate = done / max(time.time() - t0, 1e-9)
                print(f"  {i}/{len(rows)}  cdn {n['cdn']} ms {n['ms']} skip {n['skip']} fail {n['fail']}  "
                      f"{rate:.1f}/s  {dl / 1e9:.2f} GB down  eta {(len(rows) - i) / max(rate, 1e-9) / 60:.0f} min", flush=True)
    if fails:
        with open(OUT / "failed.txt", "a") as f:
            f.write("\n".join(fails) + "\n")
    files = list(OUT.glob("*.npz"))
    print(f"{n}  on disk: {len(files)} files, {sum(f.stat().st_size for f in files) / 1e6:.0f} MB")

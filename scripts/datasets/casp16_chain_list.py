#!/usr/bin/env python3
"""Training-set chain list for the distogram demo (planning/casp16_distogram_demo.md §3, steps 1
and 3): every PDB protein chain the CASP16 field could have trained on, one per 40 % sequence
cluster, purged of anything homologous to a CASP16 target.

  search   RCSB search API: released before 2024-05-01, X-ray or EM at <= 3.0 A, protein
           entities of 40-512 residues -> train/entities.txt (polymer-entity ids)
  fetch    RCSB data API (GraphQL), 200 entities per request, resumable: sequence, length,
           chain ids, resolution, method, release date -> train/entities.jsonl
  cluster  RCSB's precomputed 40 % identity clusters (clusters-by-entity-40.txt): one
           representative per cluster (best resolution, then earliest release), sequences
           with > 5 % unknown residues dropped -> train/reps.csv + train/reps.fasta
  purge    MMseqs2 easy-search of the 62 Phase-1 target sequences against reps.fasta; any
           representative hit at >= 30 % identity over >= 50 % of the target is dropped with its
           whole cluster; 5 % of the surviving clusters become val -> train/train.csv, val.csv.
           train/train_full.csv is the date-cut-only list (the purged chains put back, same val):
           the headline run trains on it, because those chains are templates the CASP16 field
           was allowed to use; train.csv is the "no templates" ablation

Runs under .venv-casp. Writes under data/casp16/train/ (gitignored). The date cut is the one the
competitors were held to; the identity cut is the CASP-style redundancy filter (AlphaFold 1
used 30 % too). Entities are PDB "polymer entities" (one sequence; a homodimer is one entity
with two chains), and the first author chain id is the one steps 2 and 4 read."""
import argparse, csv, json, os, random, subprocess, sys, time, urllib.request
from pathlib import Path

ROOT = Path(os.environ.get("CASP16_DIR", Path(__file__).resolve().parents[2] / "data" / "casp16"))
OUT = ROOT / "train"
CUTOFF = "2024-05-01"
SEARCH = "https://search.rcsb.org/rcsbsearch/v2/query"
GRAPHQL = "https://data.rcsb.org/graphql"
CLUSTERS = "https://cdn.rcsb.org/resources/sequence/clusters/clusters-by-entity-40.txt"
MMSEQS = ROOT / "tools" / "mmseqs" / "bin" / "mmseqs"


def post(url, payload, tries=5):
    data = json.dumps(payload).encode()
    for i in range(tries):
        try:
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json",
                                                                   "User-Agent": "lean4-jax-mlir casp16 demo"})
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.load(r)
        except Exception as e:  # 429 / 5xx / timeouts: back off and retry
            if i == tries - 1:
                raise
            time.sleep(5 * (i + 1))


def cmd_search(_):
    node = lambda attr, op, val: {"type": "terminal", "service": "text",
                                  "parameters": {"attribute": attr, "operator": op, "value": val}}
    q = {"query": {"type": "group", "logical_operator": "and", "nodes": [
            node("rcsb_accession_info.initial_release_date", "less", f"{CUTOFF}T00:00:00Z"),
            node("exptl.method", "in", ["X-RAY DIFFRACTION", "ELECTRON MICROSCOPY"]),
            node("rcsb_entry_info.resolution_combined", "less_or_equal", 3.0),
            node("entity_poly.rcsb_entity_polymer_type", "exact_match", "Protein"),
            node("entity_poly.rcsb_sample_sequence_length", "range",
                 {"from": 40, "to": 512, "include_lower": True, "include_upper": True})]},
         "return_type": "polymer_entity",
         "request_options": {"return_all_hits": True, "results_content_type": ["experimental"]}}
    r = post(SEARCH, q)
    ids = [h["identifier"] for h in r["result_set"]]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "entities.txt").write_text("\n".join(ids) + "\n")
    print(f"{len(ids)} polymer entities (total_count {r.get('total_count')}) -> {OUT / 'entities.txt'}")


QUERY = """query($ids: [String!]!) { polymer_entities(entity_ids: $ids) {
  rcsb_id
  entity_poly { pdbx_seq_one_letter_code_can rcsb_sample_sequence_length }
  rcsb_polymer_entity_container_identifiers { entry_id entity_id auth_asym_ids asym_ids }
  rcsb_polymer_entity { pdbx_description }
  entry { rcsb_accession_info { initial_release_date }
          rcsb_entry_info { resolution_combined }
          exptl { method } } } }"""


def cmd_fetch(a):
    ids = (OUT / "entities.txt").read_text().split()
    done = set()
    path = OUT / "entities.jsonl"
    if path.exists():
        with open(path) as f:
            for line in f:
                done.add(json.loads(line)["id"])
    todo = [i for i in ids if i not in done]
    print(f"{len(ids)} entities, {len(done)} fetched, {len(todo)} to go", flush=True)
    t0 = time.time()
    with open(path, "a") as f:
        for k in range(0, len(todo), a.batch):
            chunk = todo[k:k + a.batch]
            r = post(GRAPHQL, {"query": QUERY, "variables": {"ids": chunk}})
            for e in r["data"]["polymer_entities"] or []:
                if e is None:
                    continue
                ent = e["entry"] or {}
                res = (ent.get("rcsb_entry_info") or {}).get("resolution_combined") or [None]
                f.write(json.dumps(dict(
                    id=e["rcsb_id"],
                    seq=(e["entity_poly"] or {}).get("pdbx_seq_one_letter_code_can"),
                    length=(e["entity_poly"] or {}).get("rcsb_sample_sequence_length"),
                    entry=e["rcsb_polymer_entity_container_identifiers"]["entry_id"],
                    auth_chains=e["rcsb_polymer_entity_container_identifiers"].get("auth_asym_ids"),
                    chains=e["rcsb_polymer_entity_container_identifiers"].get("asym_ids"),
                    desc=(e["rcsb_polymer_entity"] or {}).get("pdbx_description"),
                    released=((ent.get("rcsb_accession_info") or {}).get("initial_release_date") or "")[:10],
                    resolution=res[0] if res else None,
                    method=",".join(m["method"] for m in (ent.get("exptl") or [])))) + "\n")
            f.flush()
            n = k + len(chunk)
            if (k // a.batch) % 50 == 0 or n == len(todo):
                rate = n / max(time.time() - t0, 1e-9)
                print(f"  {n}/{len(todo)}  {rate:.0f}/s  eta {(len(todo) - n) / max(rate, 1e-9) / 60:.1f} min", flush=True)
            time.sleep(a.sleep)
    print(f"-> {path}")


def load_entities():
    ents = {}
    with open(OUT / "entities.jsonl") as f:
        for line in f:
            e = json.loads(line)
            if e["seq"] and e["length"] and e["auth_chains"]:
                ents[e["id"]] = e
    return ents


def cmd_cluster(a):
    cl = OUT / "clusters-by-entity-40.txt"
    if not cl.exists():
        urllib.request.urlretrieve(CLUSTERS, cl)
    ents = load_entities()
    unknown = lambda s: sum(c not in "ACDEFGHIKLMNPQRSTVWY" for c in s) / len(s)
    reps, n_in_clusters, n_clusters = [], 0, 0
    seen = set()
    with open(cl) as f:
        for ci, line in enumerate(f):
            members = [m for m in line.split() if m in ents]
            if not members:
                continue
            n_clusters += 1
            n_in_clusters += len(members)
            seen.update(members)
            ok = [m for m in members if unknown(ents[m]["seq"]) <= 0.05]
            if not ok:
                continue
            best = min(ok, key=lambda m: (ents[m]["resolution"] or 9.9, ents[m]["released"], m))
            e = ents[best]
            reps.append(dict(id=best, entry=e["entry"], chain=e["auth_chains"][0], length=e["length"],
                             resolution=e["resolution"], method=e["method"], released=e["released"],
                             cluster=ci, cluster_size=len(members), desc=(e["desc"] or "")[:80]))
    orphans = len(set(ents) - seen)
    with open(OUT / "reps.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(reps[0].keys()))
        w.writeheader(); w.writerows(reps)
    with open(OUT / "reps.fasta", "w") as f:
        for r in reps:
            f.write(f">{r['id']}\n{ents[r['id']]['seq']}\n")
    L = sorted(r["length"] for r in reps)
    print(f"{len(ents)} entities with sequence; {n_clusters} clusters cover {n_in_clusters} of them "
          f"({orphans} entities absent from the cluster file); {len(reps)} representatives "
          f"(length median {L[len(L) // 2]}, mean {sum(L) / len(L):.0f}; residues {sum(L):,})")
    print(f"-> {OUT / 'reps.csv'}, {OUT / 'reps.fasta'}")


def cmd_purge(a):
    tgt = OUT / "targets.fasta"
    with open(ROOT / "raw" / "casp16.T1.seq.txt") as f, open(tgt, "w") as g:
        for line in f:
            g.write(line.split()[0] + "\n" if line.startswith(">") else line)
    tmp = OUT / "mmseqs_tmp"; tmp.mkdir(exist_ok=True)
    hits = OUT / "target_hits.m8"
    subprocess.run([str(MMSEQS), "easy-search", str(tgt), str(OUT / "reps.fasta"), str(hits), str(tmp),
                    "--min-seq-id", str(a.min_id), "-c", str(a.min_cov), "--cov-mode", "1",
                    "-s", "7.5", "--max-seqs", "100000", "--format-output",
                    "query,target,pident,alnlen,qlen,tlen,qcov,tcov,evalue", "-v", "1"], check=True)
    drop_ids = set(); n_hits = 0
    with open(hits) as f:
        for line in f:
            q, t, pid, alen, qlen, tlen, qcov, tcov, ev = line.split()
            n_hits += 1
            drop_ids.add(t)
    reps = list(csv.DictReader(open(OUT / "reps.csv")))
    drop_clusters = {r["cluster"] for r in reps if r["id"] in drop_ids}
    keep = [r for r in reps if r["cluster"] not in drop_clusters]
    rng = random.Random(0)
    clusters = sorted({r["cluster"] for r in keep}); rng.shuffle(clusters)
    val_clusters = set(clusters[: int(0.05 * len(clusters))])
    train = [r for r in keep if r["cluster"] not in val_clusters]
    val = [r for r in keep if r["cluster"] in val_clusters]
    train_full = [r for r in reps if r["cluster"] not in val_clusters]
    for name, rows in (("train", train), ("val", val), ("train_full", train_full)):
        with open(OUT / f"{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    targets_hit = len({l.split()[0] for l in open(hits)})
    print(f"{n_hits} target-vs-rep hits at >= {a.min_id:.0%} identity over >= {a.min_cov:.0%} of the train chain "
          f"({targets_hit} of 62 targets have a homolog in the pre-cutoff PDB); "
          f"{len(drop_ids)} representatives dropped -> {len(train)} train / {len(val)} val chains; "
          f"train_full (date cut only) {len(train_full)}")
    print(f"-> {OUT / 'train.csv'}, {OUT / 'val.csv'}, {OUT / 'train_full.csv'}, hits in {hits}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    sp.add_parser("search").set_defaults(fn=cmd_search)
    f = sp.add_parser("fetch"); f.add_argument("--batch", type=int, default=200)
    f.add_argument("--sleep", type=float, default=0.2); f.set_defaults(fn=cmd_fetch)
    sp.add_parser("cluster").set_defaults(fn=cmd_cluster)
    pu = sp.add_parser("purge"); pu.add_argument("--min-id", type=float, default=0.3)
    pu.add_argument("--min-cov", type=float, default=0.5); pu.set_defaults(fn=cmd_purge)
    a = p.parse_args(); a.fn(a)

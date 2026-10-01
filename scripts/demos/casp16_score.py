#!/usr/bin/env python3
"""CASP16 scoring for the distogram demo: the official table, reproduced with open tools.

  eus       raw/domains_summary.html -> eu_list.csv (evaluation units: target, segments, length,
            difficulty, PDB id)
  groups    raw/groups.html          -> group_names.json (group number -> name, type)
  score     one model file against one EU: lDDT (all-atom, OST), CA-lDDT (lddt() below, which
            equals OST --bb-lddt), TM-score (US-align -TMscore 1, CASP's residue-index
            superposition, normalized by the target), GDT_TS (OST --rigid-scores)
  validate  three groups x three EUs, model 1, beside the official scores.csv columns
  field     every group's model N for one EU -> CSV, the same scorer our model goes through

Runs under .venv-casp (numpy). OST is the docker image
registry.scicore.unibas.ch/schwede/openstructure (2.12.0 reproduced CASP16's 2.9 numbers);
US-align is data/casp16/tools/USalign (pylelab/USalign, built static). CASP model files carry a
PFRMAT header and blank chain IDs, which OST refuses, so every file is rewritten ATOM-only with
chain A and trimmed to the EU's residue ranges before scoring (prep()).

Checked 2026-10-01 on T1235-D1 / T1267s1-D1 / T1226-D1 x groups 304, 051, 145: lDDT equal to the
official column in all nine cases, TM-score equal in all nine, GDT_TS within one point."""
import argparse, csv, html, json, os, re, subprocess, sys
from pathlib import Path
import numpy as np

ROOT = Path(os.environ.get("CASP16_DIR", Path(__file__).resolve().parents[2] / "data" / "casp16"))
OST_IMAGE = "registry.scicore.unibas.ch/schwede/openstructure:latest"
USALIGN = ROOT / "tools" / "USalign"


# ── tables ───────────────────────────────────────────────────────────────────────────────────────
def _cells(row):
    return [html.unescape(re.sub(r"<[^>]+>", " ", c)).strip()
            for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, flags=re.S)]


def eu_ranges():
    """EU -> {target, segs [(a, b)], len, difficulty, pdb} from the official domain table."""
    s = (ROOT / "raw" / "domains_summary.html").read_text()
    out = {}
    for r in re.findall(r"<tr[^>]*>(.*?)</tr>", s, flags=re.S):
        c = _cells(r)
        if len(c) >= 6 and re.match(r"T\d{4}", c[1]) and ":" in c[3]:
            eu, rng = c[3].split(":")
            segs = [tuple(map(int, x.split("-"))) for x in rng.replace(" ", "").split(",")]
            out[eu.strip()] = dict(target=c[1], segs=segs, len=int(c[4]), difficulty=c[5],
                                   pdb="" if len(c) < 7 or c[6] == "-" else c[6])
    return out


def group_names():
    s = (ROOT / "raw" / "groups.html").read_text()
    out = {}
    for r in re.findall(r"<tr[^>]*>(.*?)</tr>", s, flags=re.S):
        c = _cells(r)
        if len(c) >= 3 and re.fullmatch(r"\d{3}", c[1]):
            out[c[1]] = dict(name=c[0], kind=c[2])
    return out


def official():
    """Official rows keyed by model name, e.g. 'T1235TS304_1-D1'."""
    rows = {}
    with open(ROOT / "raw" / "CASP16_prot_domains.scores.csv") as f:
        hdr = f.readline().split()
        for line in f:
            p = line.split()
            if len(p) >= len(hdr) - 2:
                rows[p[1]] = dict(zip(hdr[1:], p[1:]))
    return rows


# ── structures ───────────────────────────────────────────────────────────────────────────────────
def prep(src, dst, segs=None, pseudo_cb=False):
    """CASP PDB -> OST-clean PDB: ATOM lines only, chain A, kept to the EU's residue ranges.
    `pseudo_cb`: one atom per residue, the Cβ (Cα for glycine or when no Cβ), written as CA —
    the atom set the distogram is defined on, for scoring a folded pseudo-Cβ trace against the
    field's models on equal terms (a file that already has only CA atoms passes through)."""
    lines = []
    with open(src) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            resnum = int(line[22:26])
            if segs and not any(a <= resnum <= b for a, b in segs):
                continue
            lines.append(line[:21] + "A" + line[22:])
    if pseudo_cb:
        by_res, order = {}, []
        for line in lines:
            key = (int(line[22:26]), line[26])
            if key not in by_res:
                by_res[key] = {}; order.append(key)
            by_res[key].setdefault(line[12:16].strip(), line)
        lines = []
        for key in order:
            atoms = by_res[key]
            pick = atoms.get("CB") if atoms.get("CB") and atoms[next(iter(atoms))][17:20] != "GLY" else atoms.get("CA")
            if pick:
                lines.append(pick[:12] + " CA " + pick[16:])
    with open(dst, "w") as g:
        g.writelines(lines)
        g.write("END\n")


def read_atoms(path, ca_only=False):
    """(resnum, icode) -> {atom name: xyz}; heavy atoms, first altloc."""
    res = {}
    for line in open(path):
        if not line.startswith("ATOM"):
            continue
        name = line[12:16].strip()
        if ca_only and name != "CA":
            continue
        el = (line[76:78].strip() or name[0]).upper()
        if el == "H" or line[16] not in " A":
            continue
        key = (int(line[22:26]), line[26])
        res.setdefault(key, {}).setdefault(
            name, np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]))
    return res


def lddt(model, ref, r0=15.0, thr=(0.5, 1.0, 2.0, 4.0)):
    """lDDT (Mariani 2013): over reference atom pairs in different residues within r0, the mean
    over the four thresholds of the fraction whose distance the model keeps within threshold;
    an atom the model lacks keeps nothing. On CA atoms this equals OST's --bb-lddt exactly; on
    all atoms it sits 0.01-0.02 under OST's --lddt, which also swaps symmetric side-chain atoms."""
    keys = [(r, a) for r in sorted(ref) for a in ref[r]]
    X = np.array([ref[r][a] for r, a in keys])
    rid = np.array([hash(r) for r, _ in keys])
    Y = np.array([model.get(r, {}).get(a, [np.nan] * 3) for r, a in keys], dtype=float)
    dr = np.linalg.norm(X[:, None] - X[None], axis=-1)
    dm = np.linalg.norm(Y[:, None] - Y[None], axis=-1)
    iu = np.triu_indices(len(keys), 1)
    sel = (dr[iu] < r0) & (rid[iu[0]] != rid[iu[1]])
    diff = np.abs(dm[iu][sel] - dr[iu][sel])
    with np.errstate(invalid="ignore"):
        cons = sum(int(np.sum(diff < t)) for t in thr)
    return cons / (len(thr) * int(sel.sum()))


def ost(model, ref, out):
    """compare-structures: all-atom lDDT, CA lDDT (bb_lddt), rigid GDT; None on failure."""
    cmd = ["docker", "run", "--rm", "-v", f"{ROOT}:/data", OST_IMAGE, "compare-structures",
           "-m", f"/data/{Path(model).relative_to(ROOT)}", "-mf", "pdb",
           "-r", f"/data/{Path(ref).relative_to(ROOT)}", "-rf", "pdb",
           "--lddt", "--bb-lddt", "--rigid-scores", "-o", f"/data/{Path(out).relative_to(ROOT)}"]
    subprocess.run(cmd, capture_output=True, text=True)
    d = json.load(open(out))
    if d.get("status") == "FAILURE":
        print(f"OST failed on {model}: {d.get('exception')}", file=sys.stderr)
        return {}
    return dict(lddt=d.get("lddt"), ost_ca_lddt=d.get("bb_lddt"),
                gdtts=None if d.get("oligo_gdtts") is None else 100 * d["oligo_gdtts"],
                gdtha=None if d.get("oligo_gdtha") is None else 100 * d["oligo_gdtha"],
                rms_ca=d.get("rmsd"))


def usalign_tm(model, ref):
    """CASP's TMscore column: superpose by residue index, normalize by the target length."""
    o = subprocess.run([str(USALIGN), str(model), str(ref), "-outfmt", "2", "-TMscore", "1"],
                       capture_output=True, text=True).stdout
    f = o.strip().splitlines()[-1].split("\t")
    return float(f[3])


def score(model_file, eu, eus=None, tag=None, keep=False, pseudo_cb=False):
    """One model against one EU. Returns the metrics dict. `pseudo_cb`: both sides reduced to
    one pseudo-Cβ atom per residue (prep); then `ca_lddt` is the Cβ-lDDT and `tm` the TM-score
    over those atoms, and OST (which needs a backbone) is skipped."""
    eus = eus or eu_ranges()
    e = eus[eu]
    work = ROOT / "work" / "prep"
    work.mkdir(parents=True, exist_ok=True)
    tag = tag or Path(model_file).name
    suffix = ".cb" if pseudo_cb else ""
    ref = work / f"{eu}.ref{suffix}.pdb"
    if not ref.exists():
        prep(ROOT / "raw" / "dom" / f"{eu}.pdb", ref, pseudo_cb=pseudo_cb)
    mdl = work / f"{tag}-{eu}{suffix}.pdb"
    prep(model_file, mdl, e["segs"], pseudo_cb=pseudo_cb)
    r = dict(model=tag, eu=eu)
    r["ca_lddt"] = lddt(read_atoms(mdl, True), read_atoms(ref, True))
    r["tm"] = usalign_tm(mdl, ref)
    if not pseudo_cb:
        r.update(ost(mdl, ref, work / f"{tag}-{eu}.ost.json"))
    if not keep:
        mdl.unlink(missing_ok=True)
        (work / f"{tag}-{eu}.ost.json").unlink(missing_ok=True)
    return r


# ── commands ─────────────────────────────────────────────────────────────────────────────────────
def cmd_eus(_):
    eus = eu_ranges()
    with open(ROOT / "eu_list.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["eu", "target", "segments", "length", "difficulty", "pdb"])
        for eu, e in eus.items():
            w.writerow([eu, e["target"], ",".join(f"{a}-{b}" for a, b in e["segs"]),
                        e["len"], e["difficulty"], e["pdb"]])
    by = {}
    for e in eus.values():
        by[e["difficulty"]] = by.get(e["difficulty"], 0) + 1
    print(f"{len(eus)} EUs -> {ROOT / 'eu_list.csv'}  {by}")


def cmd_groups(_):
    g = group_names()
    json.dump(g, open(ROOT / "group_names.json", "w"), indent=1, sort_keys=True)
    print(f"{len(g)} groups -> {ROOT / 'group_names.json'}")


def cmd_validate(a):
    eus, off = eu_ranges(), official()
    print(f"{'model':22s} {'lDDT ours/off':>14s} {'CA-lDDT np/ost':>15s} {'TM ours/off':>12s} "
          f"{'GDT_TS ours/off':>16s}")
    for eu in a.eus:
        t = eus[eu]["target"]
        for g in a.groups:
            name = f"{t}TS{g}_{a.model}"
            src = ROOT / "raw" / "predictions" / t / name
            if not src.exists():
                print(f"{name}: no model")
                continue
            r = score(src, eu, eus, tag=name, pseudo_cb=a.pseudo_cb)
            o = off.get(f"{name}-{eu.split('-')[1]}", {})
            f3 = lambda x: "  -  " if x is None else f"{x:.3f}"
            print(f"{name + '-' + eu.split('-')[1]:22s} {f3(r.get('lddt')) + '/' + o.get('LDDT', '-'):>14s} "
                  f"{f3(r['ca_lddt']) + '/' + f3(r.get('ost_ca_lddt')):>15s} "
                  f"{f3(r['tm']) + '/' + o.get('TMscore', '-'):>12s} "
                  f"{('  -  ' if r.get('gdtts') is None else f'{r[chr(103)+chr(100)+chr(116)+chr(116)+chr(115)]:.1f}') + '/' + o.get('GDT_TS', '-'):>16s}")


def cmd_field(a):
    eus = eu_ranges()
    t = eus[a.eu]["target"]
    out = ROOT / "work" / f"field_{a.eu}_m{a.model}{'_cb' if a.pseudo_cb else ''}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    files = sorted((ROOT / "raw" / "predictions" / t).glob(f"{t}TS???_{a.model}"))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "eu", "group", "ca_lddt", "tm", "lddt",
                                          "ost_ca_lddt", "gdtts", "gdtha", "rms_ca"])
        w.writeheader()
        for i, src in enumerate(files):
            r = score(src, a.eu, eus, tag=src.name, pseudo_cb=a.pseudo_cb)
            r["group"] = src.name.split("TS")[1].split("_")[0]
            w.writerow({k: r.get(k) for k in w.fieldnames})
            if a.verbose:
                print(f"[{i + 1}/{len(files)}] {src.name} CA-lDDT {r['ca_lddt']:.3f} TM {r['tm']:.3f}")
    print(f"{len(files)} models -> {out}")


def cmd_score(a):
    r = score(a.model, a.eu, keep=a.keep, pseudo_cb=a.pseudo_cb)
    print(json.dumps(r, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    sp.add_parser("eus").set_defaults(fn=cmd_eus)
    sp.add_parser("groups").set_defaults(fn=cmd_groups)
    v = sp.add_parser("validate")
    v.add_argument("--eus", nargs="+", default=["T1235-D1", "T1267s1-D1", "T1226-D1"])
    v.add_argument("--groups", nargs="+", default=["304", "051", "145"])
    v.add_argument("--model", type=int, default=1)
    v.add_argument("--pseudo-cb", action="store_true")
    v.set_defaults(fn=cmd_validate)
    s = sp.add_parser("score")
    s.add_argument("model"); s.add_argument("eu"); s.add_argument("--keep", action="store_true")
    s.add_argument("--pseudo-cb", action="store_true")
    s.set_defaults(fn=cmd_score)
    fl = sp.add_parser("field")
    fl.add_argument("eu"); fl.add_argument("--model", type=int, default=1)
    fl.add_argument("-v", "--verbose", action="store_true")
    fl.add_argument("--pseudo-cb", action="store_true")
    fl.set_defaults(fn=cmd_field)
    a = p.parse_args()
    a.fn(a)

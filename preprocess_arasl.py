#!/usr/bin/env python3
"""Build the ArASL sets under two splits — planning/arasl_people_watching_demo.md §2.

ArASL (Latif et al. 2019, Mendeley y7pckrw6z2 v1, CC BY 4.0) is 54,049 grey 64×64
crops of hands spelling the 32 letters of the Arabic alphabet, 1,293–2,114 per class.
The file numbers are capture order and the images are video BURSTS: consecutive
files differ by a few grey levels, and a random split puts near-identical frames of
the same hand on both sides of the line. This writes the same images under both
protocols, 80/10/10 per class from one seed, so the two columns of the demo's table
are one code path with one constant changed:

  random    a stratified permutation of images — the literature's protocol
  blocked   within each class, capture order, cut at the 80% and 90% marks; each cut
            is moved to the nearest chain boundary so no burst straddles two parts

Files, per protocol P in {random, blocked} and part S in {train, val, test}:

  data/arasl/P_S.bin           f32 [N, 1, 64, 64] in [0, 1]   (what the chapter's CIFAR
                                                              loader feeds: raw / 255)
  data/arasl/labels_P_S.bin    int32, class id 0..31 in alphabetical folder order
  data/arasl/meta_P.npz        per image, in the concatenated train|val|test order:
                               label, file number, chain id, part (0/1/2), global index
                               into images_u8.npy; and for every TEST image the nearest
                               train image (any class) at 16×16, its mean |Δ| there and at
                               64×64 — the leak audit, computed once here, read by the scorer
  data/arasl/images_u8.npy     uint8 [54049, 64, 64], all classes in capture order —
                               the figure script and the scorer's neighbour panel read it
  data/arasl/classes.txt       the 32 names, one per line, in label order
  data/arasl/manifest.json     census, chain statistic, split sizes, audit summary

Every non-64×64 file (638 at 256², 10 at 768×1024) and the 10 RGB files are resized
and converted with ONE resampler; --stats lists them so a mirror that differs is caught.

  .venv/bin/python preprocess_arasl.py data/arasl data/arasl [--stats] [--seed=0]
                   [--chain-thr=6] [--size=64]

`--size 32` writes `<protocol>_<part>_32.bin` (+ labels, meta, manifest with the same
suffix): the SAME splits, downsampled — the chapter net's native input, for the
`size=32` arm.
"""
import argparse
import glob
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

N_EXPECTED = 54049
N_CLASSES = 32
FRACS = (0.8, 0.9)             # cumulative train | val | test marks
PARTS = ("train", "val", "test")
THUMB = 16                     # leak-audit resolution
LEAK_THR = 6.0                 # grey levels; the chain threshold, reused


def file_num(path):
    m = re.search(r"\((\d+)\)", os.path.basename(path))
    return int(m.group(1)) if m else -1


def load_one(path, size):
    im = Image.open(path)
    flags = []
    if im.mode != "L":
        flags.append(im.mode)
        im = im.convert("L")
    if im.size != (size, size):
        flags.append(f"{im.size[0]}x{im.size[1]}")
        im = im.resize((size, size), Image.LANCZOS)
    return np.asarray(im, dtype=np.uint8), flags


def chains_of(imgs, thr):
    """Chain ids along capture order: a new chain starts where the mean |Δ| to the
    previous frame exceeds `thr`. Returns (chain id per frame, the per-pair |Δ| array)."""
    x = imgs.reshape(len(imgs), -1).astype(np.int16)
    d = np.abs(x[1:] - x[:-1]).mean(axis=1)
    cut = np.concatenate([[True], d > thr])
    return np.cumsum(cut) - 1, d


def snap(pos, boundaries):
    """The chain boundary nearest to `pos` (boundaries are sorted positions where a
    chain starts, plus n)."""
    i = np.searchsorted(boundaries, pos)
    lo = boundaries[max(i - 1, 0)]
    hi = boundaries[min(i, len(boundaries) - 1)]
    return lo if pos - lo <= hi - pos else hi


def leak_audit(thumbs, train_idx, test_idx, chunk=128, tchunk=4096):
    """For every test image, the nearest train image (any class) by mean |Δ| at
    THUMB×THUMB, and that distance. Exact, chunked int16 broadcast."""
    tr = thumbs[train_idx].astype(np.int16)
    nn = np.zeros(len(test_idx), dtype=np.int64)
    nd = np.zeros(len(test_idx), dtype=np.float32)
    for a in range(0, len(test_idx), chunk):
        q = thumbs[test_idx[a:a + chunk]].astype(np.int16)
        best = np.full(len(q), np.inf, dtype=np.float32)
        arg = np.zeros(len(q), dtype=np.int64)
        for b in range(0, len(tr), tchunk):
            d = np.abs(q[:, None, :] - tr[None, b:b + tchunk, :]).sum(axis=2).astype(np.float32)
            j = d.argmin(axis=1)
            v = d[np.arange(len(q)), j]
            better = v < best
            best[better] = v[better]
            arg[better] = j[better] + b
        nn[a:a + chunk] = train_idx[arg]
        nd[a:a + chunk] = best / (THUMB * THUMB)
    return nn, nd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="data/arasl (holds ArASL_Database_54K_Final/)")
    ap.add_argument("out", help="data/arasl")
    ap.add_argument("--stats", action="store_true", help="print the census, chain statistic and audit")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chain-thr", type=float, default=LEAK_THR)
    ap.add_argument("--size", type=int, default=64)
    args = ap.parse_args()
    root = os.path.join(args.src, "ArASL_Database_54K_Final")
    os.makedirs(args.out, exist_ok=True)
    sfx = "" if args.size == 64 else f"_{args.size}"
    t0 = time.time()

    # ── census + load, class by class in capture order ──
    classes = sorted(os.listdir(root))
    if len(classes) != N_CLASSES:
        sys.exit(f"{len(classes)} class folders, expected {N_CLASSES}: {classes}")
    paths, labels, nums = [], [], []
    for ci, c in enumerate(classes):
        fs = sorted(glob.glob(os.path.join(root, c, "*")), key=file_num)
        ns = [file_num(f) for f in fs]
        if ns != list(range(1, len(fs) + 1)):
            sys.exit(f"{c}: file numbers are not 1..{len(fs)} — capture order is not recoverable")
        paths += fs
        labels += [ci] * len(fs)
        nums += ns
    labels = np.asarray(labels, dtype=np.int32)
    nums = np.asarray(nums, dtype=np.int32)
    n = len(paths)
    if n != N_EXPECTED:
        sys.exit(f"{n} images, expected {N_EXPECTED}")
    # always loaded at the native 64×64: the chains, the splits and the audit are defined
    # there, and a --size set is the SAME split downsampled, not a split of its own
    with ThreadPoolExecutor(16) as ex:
        loaded = list(ex.map(lambda p: load_one(p, 64), paths))
    imgs = np.stack([a for a, _ in loaded])
    odd = [(os.path.relpath(paths[i], root), fl) for i, (_, fl) in enumerate(loaded) if fl]
    per_class = np.bincount(labels, minlength=N_CLASSES)
    print(f"{n} images, {N_CLASSES} classes, per class {per_class.min()} ({classes[per_class.argmin()]}) "
          f"to {per_class.max()} ({classes[per_class.argmax()]}), {len(odd)} not 64x64 L "
          f"({time.time() - t0:.0f} s)")
    if args.stats:
        by = {}
        for p, fl in odd:
            by.setdefault((p.split("/")[0], " ".join(fl)), 0)
            by[(p.split("/")[0], " ".join(fl))] += 1
        for (c, fl), k in sorted(by.items()):
            print(f"  {c:6s} {fl:12s} ×{k}")

    # ── the chain statistic, per class, on the 64×64 images ──
    chain = np.zeros(n, dtype=np.int32)
    chain_rows, next_id = [], 0
    rng = np.random.RandomState(args.seed)
    for ci, c in enumerate(classes):
        sel = np.flatnonzero(labels == ci)
        ids, d = chains_of(imgs[sel], args.chain_thr)
        chain[sel] = ids + next_id
        next_id += ids[-1] + 1
        lens = np.bincount(ids)
        x = imgs[sel].reshape(len(sel), -1).astype(np.int16)
        a, b = rng.randint(0, len(sel), 2000), rng.randint(0, len(sel), 2000)
        rand_d = np.abs(x[a] - x[b]).mean(axis=1)[a != b]
        chain_rows.append(dict(cls=c, n=int(len(sel)), median_consec=float(np.median(d)),
                               frac_under=float((d < args.chain_thr).mean()), chains=int(len(lens)),
                               mean_len=float(lens.mean()), max_len=int(lens.max()),
                               median_random=float(np.median(rand_d))))
    all_d = np.concatenate([chains_of(imgs[labels == ci], args.chain_thr)[1] for ci in range(N_CLASSES)])
    n_chains = int(chain.max()) + 1
    frac_all = float((all_d < args.chain_thr).mean())
    print(f"bursts: median consecutive |Δ| {np.median(all_d):.1f} grey levels "
          f"(random pair within a class: {np.median([r['median_random'] for r in chain_rows]):.1f}); "
          f"{100 * frac_all:.1f}% of consecutive pairs under {args.chain_thr:g}; "
          f"{n_chains} chains, {n / n_chains:.1f} frames each, longest {max(r['max_len'] for r in chain_rows)}")
    worst = min(chain_rows, key=lambda r: r["frac_under"])
    if args.stats:
        for r in chain_rows:
            print(f"  {r['cls']:6s} n={r['n']:5d}  consec |Δ| {r['median_consec']:5.1f}  random {r['median_random']:5.1f}  "
                  f"under-thr {100 * r['frac_under']:5.1f}%  chains {r['chains']:4d}  mean {r['mean_len']:4.1f}  max {r['max_len']:3d}")
    gate_chain = worst["frac_under"] >= 0.6
    print(f"  weakest class {worst['cls']}: {100 * worst['frac_under']:.1f}% under threshold "
          f"— Gate 0 chain premise {'HOLDS' if gate_chain else 'FAILS'} (≥ 60% in every class)")

    # ── the two splits, 80/10/10 per class ──
    part = {}
    for proto in ("random", "blocked"):
        p = np.zeros(n, dtype=np.int8)
        prng = np.random.RandomState(args.seed)
        for ci in range(N_CLASSES):
            sel = np.flatnonzero(labels == ci)
            m = len(sel)
            cuts = [int(round(f * m)) for f in FRACS]
            if proto == "random":
                order = prng.permutation(m)
                p[sel[order[cuts[0]:cuts[1]]]] = 1
                p[sel[order[cuts[1]:]]] = 2
            else:
                ids = chain[sel]
                bounds = np.concatenate([[0], np.flatnonzero(ids[1:] != ids[:-1]) + 1, [m]])
                c0, c1 = snap(cuts[0], bounds), snap(cuts[1], bounds)
                if not (0 < c0 < c1 < m):
                    sys.exit(f"{classes[ci]}: blocked cuts {c0},{c1} of {m} degenerate")
                p[sel[c0:c1]] = 1
                p[sel[c1:]] = 2
        part[proto] = p
        # no chain straddles two parts under blocked
        if proto == "blocked":
            straddle = sum(1 for cid in range(n_chains) if len(set(p[chain == cid])) > 1)
            if straddle:
                sys.exit(f"blocked: {straddle} chains straddle a cut — the boundary snap is wrong")
        sizes = np.bincount(p, minlength=3)
        print(f"{proto:8s} split: train {sizes[0]}  val {sizes[1]}  test {sizes[2]}")

    # ── the leak audit: test → nearest train image at 16×16, under each protocol ──
    thumbs = np.stack([np.asarray(Image.fromarray(a).resize((THUMB, THUMB), Image.BOX)) for a in imgs])
    thumbs = thumbs.reshape(n, -1)
    audit = {}
    for proto in ("random", "blocked"):
        p = part[proto]
        tr, te = np.flatnonzero(p == 0), np.flatnonzero(p == 2)
        t1 = time.time()
        nn, nd = leak_audit(thumbs, tr, te)
        same = labels[nn] == labels[te]
        frac = float((nd < LEAK_THR).mean())
        # the same pair at full resolution (the chain threshold's footing), and how far
        # apart in capture order a same-class near-duplicate sits — a burst neighbour
        # (gap 1–3) or the same hand returning in a later sitting (gap in the hundreds)
        x = imgs.reshape(n, -1).astype(np.int16)
        nd64 = np.abs(x[te] - x[nn]).mean(axis=1).astype(np.float32)
        frac64 = float((nd64 < LEAK_THR).mean())
        gap = np.abs(nums[te] - nums[nn])[same & (nd < LEAK_THR)]
        audit[proto] = dict(n_test=int(len(te)), frac_under=frac, median_nn=float(np.median(nd)),
                            frac_under_64=frac64, median_nn_64=float(np.median(nd64)),
                            frac_same_class_nn=float(same.mean()),
                            median_gap_of_same_class_leaks=float(np.median(gap)) if len(gap) else None,
                            nn=nn, nd=nd, nd64=nd64)
        print(f"leak audit {proto:8s}: {100 * frac:5.1f}% of test images have a train image within "
              f"{LEAK_THR:g} grey levels at {THUMB}×{THUMB} (median nearest {np.median(nd):.1f}); "
              f"{100 * frac64:5.1f}% at 64×64 (median {np.median(nd64):.1f}); nearest is same class "
              f"{100 * same.mean():.1f}%; same-class leaks sit a median {np.median(gap) if len(gap) else 0:.0f} "
              f"files apart  ({time.time() - t1:.0f} s)")
    gate_leak = audit["random"]["frac_under"] > 3 * max(audit["blocked"]["frac_under"], 0.01)
    print(f"  Gate 0 audit: random ≫ blocked {'HOLDS' if gate_leak else 'FAILS'}")

    # ── write ──
    out_imgs = imgs if args.size == 64 else np.stack(
        [np.asarray(Image.fromarray(a).resize((args.size, args.size), Image.LANCZOS)) for a in imgs])
    f32 = out_imgs.astype(np.float32) / 255.0
    for proto in ("random", "blocked"):
        p = part[proto]
        order = np.concatenate([np.flatnonzero(p == k) for k in range(3)])
        for k, s in enumerate(PARTS):
            sel = np.flatnonzero(p == k)
            f32[sel].reshape(len(sel), 1, args.size, args.size).tofile(os.path.join(args.out, f"{proto}_{s}{sfx}.bin"))
            labels[sel].tofile(os.path.join(args.out, f"labels_{proto}_{s}{sfx}.bin"))
        te = np.flatnonzero(p == 2)
        np.savez(os.path.join(args.out, f"meta_{proto}{sfx}.npz"), label=labels[order], file_num=nums[order],
                 chain=chain[order], part=p[order], index=order, classes=np.array(classes),
                 test_index=te, test_nn_train_index=audit[proto]["nn"], test_nn_mad=audit[proto]["nd"],
                 test_nn_mad64=audit[proto]["nd64"])
    if not sfx:
        np.save(os.path.join(args.out, "images_u8.npy"), imgs)
        with open(os.path.join(args.out, "classes.txt"), "w") as f:
            f.write("\n".join(classes) + "\n")
    manifest = dict(n=n, classes=classes, per_class=per_class.tolist(), size=args.size, seed=args.seed,
                    non_conforming=[dict(file=p, what=fl) for p, fl in odd],
                    chain_thr=args.chain_thr, n_chains=n_chains, frac_consecutive_under=frac_all,
                    median_consecutive=float(np.median(all_d)), chains_per_class=chain_rows,
                    splits={pr: np.bincount(part[pr], minlength=3).tolist() for pr in part},
                    leak_audit={pr: {k: v for k, v in a.items() if k not in ("nn", "nd", "nd64")} for pr, a in audit.items()},
                    gate0=dict(chain_premise=bool(gate_chain), random_gg_blocked=bool(gate_leak)))
    with open(os.path.join(args.out, f"manifest{sfx}.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"wrote {args.out}/{{random,blocked}}_{{train,val,test}}{sfx}.bin + labels + meta "
          f"({time.time() - t0:.0f} s total)")


if __name__ == "__main__":
    main()

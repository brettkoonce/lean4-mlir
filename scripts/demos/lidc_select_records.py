#!/usr/bin/env python3
"""Which records of a CT pair file the generator samples and every arm is scored on —
planning/lidc_ct_denoising_demo.md §10.5, chosen by rule, written once and read by
`lidc-denoise-sample --records=` and by every scorer's `--records=`.

  .venv-lidc/bin/python scripts/demos/lidc_select_records.py --data=data/lidc \
      --out=runs/2026-11-02-lidc-diffusion/records_val.txt \
      --spread-out=runs/2026-11-02-lidc-diffusion/records_spread.txt

The scoring set (`--out`): every record with a reader-marked pixel (`nodule_px > 0` in
`<data>/val.json`; 1,205 on the 25% LIDC val split) plus `--others` (400) records without one,
evenly spaced through the file (np.linspace over the nodule-free indices, rounded, duplicates
dropped), so the fidelity and texture numbers see every patient and the nodule table sees every
nodule. The spread set (`--spread-out`, 4 samples a record): the `--spread`/2 nodule records with
the most reader-marked pixels plus `--spread`/2 of the scoring set's nodule-free records, evenly
spaced — a subset of the scoring set, so sample seed 0 serves both.

Mayo has no readers: `--per-patient=20` takes 20 evenly spaced records from each volume of
`<data>/val.idx` (uint32 volume count, then one record count per volume, in file order) for the
scoring set and `--per-patient-spread` (default 4) for the spread set; a record whose two arms are
identical (the vendor inserted no noise: 2 of Mayo's 4,348, C158's first slices) is not a pair and
is replaced by the next slice of its volume. Deterministic: no RNG.
Output is one record index per line with `#` comments; `lidc_records.read_record_list` parses it.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets"))
import lidc_records as R  # noqa: E402


def arg(name, default=None):
    return next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith(f"--{name}=")), default)


def spaced(pool, n):
    """`n` entries of `pool` at evenly spaced positions (all of them when n >= len)."""
    if n >= len(pool):
        return list(pool)
    if n <= 0:
        return []
    return sorted({pool[i] for i in np.linspace(0, len(pool) - 1, n).round().astype(int)})


def write(path, recs, lines):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        for line in lines:
            f.write(f"# {line}\n")
        f.write(f"# {len(recs)} records\n")
        for k in recs:
            f.write(f"{k}\n")
    print(f"{path}: {len(recs)} records")


def main():
    data = arg("data", "data/lidc")
    split = arg("split", "val")
    pairs = arg("pairs", f"{data}/{split}.bin")
    out = arg("out", f"runs/2026-11-02-lidc-diffusion/records_{os.path.basename(data)}_{split}.txt")
    spread_out = arg("spread-out")
    others = int(arg("others", 400))
    spread = int(arg("spread", 200))
    per_patient = arg("per-patient")
    count, n, mm = R.open_records(pairs)
    if per_patient is not None:
        per = int(per_patient)
        per_spread = int(arg("per-patient-spread", 4))
        idx = np.fromfile(pairs[: -len(".bin")] + ".idx", dtype="<u4")
        vols = [int(v) for v in idx[1: 1 + int(idx[0])]]
        assert sum(vols) == count, f"{pairs}: .idx covers {sum(vols)} records of {count}"
        starts = [sum(vols[:i]) for i in range(len(vols))]

        def is_pair(k):
            full, low, _ = R.decode(mm[k], n)
            return bool((low != full).any())

        def pick(s, v, m):
            chosen, skipped = [], 0
            for k in spaced(list(range(s, s + v)), m):
                while k < s + v and (not is_pair(k) or k in chosen):
                    k += 1; skipped += 1
                if k < s + v:
                    chosen.append(k)
            return chosen, skipped

        picks = [pick(s, v, per) for s, v in zip(starts, vols)]
        recs = sorted(k for c, _ in picks for k in c)
        nskip = sum(sk for _, sk in picks)
        write(out, recs, [f"{pairs}: {per} evenly spaced records of each of {len(vols)} volumes (no readers)"
                          + (f"; {nskip} identical-arm records replaced by the next slice" if nskip else "")])
        if spread_out:
            srecs = sorted(k for s, v in zip(starts, vols) for k in pick(s, v, per_spread)[0])
            write(spread_out, srecs, [f"{pairs}: {per_spread} evenly spaced records of each of {len(vols)} volumes, "
                                      "for the spread across samples"])
        return
    meta = json.load(open(pairs[: -len(".bin")] + ".json"))
    assert len(meta) == count, f"{len(meta)} rows for {count} records"
    nod = [k for k, r in enumerate(meta) if r["nodule_px"] > 0]
    free = [k for k, r in enumerate(meta) if r["nodule_px"] == 0]
    recs = sorted(set(nod) | set(spaced(free, others)))
    write(out, recs, [f"{pairs}: every record with a reader-marked pixel ({len(nod)}) + {min(others, len(free))} "
                      f"evenly spaced records without one, of {count}"])
    if spread_out:
        # a subset of the scoring set, so sample seed 0 of the spread study is also the scoring sample
        top = sorted(sorted(nod, key=lambda k: (-meta[k]["nodule_px"], k))[: spread // 2])
        chosen_free = [k for k in recs if meta[k]["nodule_px"] == 0]
        srecs = sorted(set(top) | set(spaced(chosen_free, spread - len(top))))
        write(spread_out, srecs, [f"{pairs}: the {len(top)} nodule records with the most reader-marked pixels + "
                                  f"{len(srecs) - len(top)} evenly spaced records without one (a subset of the "
                                  "scoring set), for the spread across samples"])


if __name__ == "__main__":
    main()

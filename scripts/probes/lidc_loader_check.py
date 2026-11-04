"""Gate: the Lean CT pair loader returns exactly the pairs preprocess_lidc.py / preprocess_mayo.py wrote.

  .venv-lidc/bin/python scripts/probes/lidc_loader_check.py data/lidc/val.bin [record ...]

Runs `.lake/build/bin/test-lidc-loader` on the file, decodes the same records with numpy straight
from the record layout (uint32 count; per record full, N low realisations as uint16 HU + 1024,
then the uint8 reader-count mask; N from the file size, as the Lean side reads it), and requires
every printed statistic to match exactly: they are integer sums of values on the 1/4096 grid, so
there is no tolerance to choose. Covers every supervised realisation, the Noise2Noise pairing
when N = 2, the masks, and the loader's refusal of a wrong realisation count. Exit 0 = PASS.
"""
import os
import re
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets"))
from lidc_records import HW, open_records  # noqa: E402


def expected(path, recs):
    count, n, mm = open_records(path)
    if not recs:
        recs = [0, count // 2, count - 1]
    out = {}
    pairs = [(0, r, 1 + r, 0) for r in range(n)] + ([(1, 0, 1, 2)] if n >= 2 else [])
    for k in recs:
        r = mm[k]
        imgs = r[: 2 * (1 + n) * HW].view("<u2").reshape(1 + n, HW).astype(np.int64)
        mask = r[2 * (1 + n) * HW:].astype(np.int64)
        for p, rr, a, b in pairs:
            x, y = imgs[a], imgs[b]
            out[f"rec {k} pair {p} r {rr}"] = f"in {x.sum()} tgt {y.sum()} ssd {((x - y) ** 2).sum()}"
        out[f"rec {k} mask"] = f"{mask.sum()} nz {(mask > 0).sum()}"
    return count, n, recs, out


def main():
    path, recs = sys.argv[1], [int(a) for a in sys.argv[2:]]
    count, n, recs, want = expected(path, recs)
    run = subprocess.run([".lake/build/bin/test-lidc-loader", path, *map(str, recs)],
                         capture_output=True, text=True)
    print(run.stdout, end="")
    got = {}
    for line in run.stdout.splitlines():
        m = re.match(r"(rec \d+ (?:pair \d+ r \d+|mask)) (.*)", line)
        if m:
            got[m.group(1)] = m.group(2)
    bad = [k for k in want if got.get(k) != want[k]]
    for k in bad:
        print(f"MISMATCH {k}: lean {got.get(k)!r} numpy {want[k]!r}")
    heads = {(int(c), int(nn)) for c, nn in re.findall(r"^count (\d+) N (\d+)$", run.stdout, re.M)}
    if heads != {(count, n)}:
        print(f"MISMATCH count/N: lean {sorted(heads)} numpy {(count, n)}")
        bad.append("count")
    if run.returncode != 0 or f"refused N={n + 1}" not in run.stdout:
        print(f"FAIL loader exit {run.returncode}\n{run.stderr}")
        bad.append("exit")
    print(f"=== {len(want)} statistics over {len(recs)} records (N = {n}): {'PASS' if not bad else 'FAIL'} ===")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

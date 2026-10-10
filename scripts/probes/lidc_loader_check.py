"""Gate: the Lean LIDC loader returns exactly the pairs preprocess_lidc.py wrote.

  .venv-lidc/bin/python scripts/probes/lidc_loader_check.py data/lidc/val.bin [record ...]

Runs `.lake/build/bin/test-lidc-loader` on the file, decodes the same records with numpy straight
from the record layout (uint32 count; per record full, N = 2 low realisations as uint16 HU + 1024,
then the uint8 reader-count mask), and requires every printed statistic to match exactly: they
are integer sums of values on the 1/4096 grid, so there is no tolerance to choose. Covers both
supervised realisations, the Noise2Noise pairing, the masks, and the loader's refusal of a wrong
realisation count. Exit 0 = PASS.
"""
import re
import subprocess
import sys

import numpy as np

S, N = 512, 2
HW = S * S
REC = (3 + 2 * N) * HW


def expected(path, recs):
    count = int(np.fromfile(path, dtype="<u4", count=1)[0])
    mm = np.memmap(path, dtype=np.uint8, mode="r", offset=4, shape=(count, REC))
    if not recs:
        recs = [0, count // 2, count - 1]
    out = {}
    for k in recs:
        r = mm[k]
        imgs = r[: 2 * (1 + N) * HW].view("<u2").reshape(1 + N, HW).astype(np.int64)
        mask = r[2 * (1 + N) * HW:].astype(np.int64)
        for p, rr, a, b in [(0, 0, 1, 0), (0, 1, 2, 0), (1, 0, 1, 2)]:
            x, y = imgs[a], imgs[b]
            out[f"rec {k} pair {p} r {rr}"] = f"in {x.sum()} tgt {y.sum()} ssd {((x - y) ** 2).sum()}"
        out[f"rec {k} mask"] = f"{mask.sum()} nz {(mask > 0).sum()}"
    return count, recs, out


def main():
    path, recs = sys.argv[1], [int(a) for a in sys.argv[2:]]
    count, recs, want = expected(path, recs)
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
    counts = {int(c) for c in re.findall(r"^count (\d+)$", run.stdout, re.M)}
    if counts != {count}:
        print(f"MISMATCH count: lean {sorted(counts)} numpy {count}")
        bad.append("count")
    if run.returncode != 0 or "refused N=3" not in run.stdout:
        print(f"FAIL loader exit {run.returncode}\n{run.stderr}")
        bad.append("exit")
    print(f"=== {len(want)} statistics over {len(recs)} records: {'PASS' if not bad else 'FAIL'} ===")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

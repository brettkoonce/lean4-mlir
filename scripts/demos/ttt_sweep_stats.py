"""The numbers behind the AlphaZero tic-tac-toe table (planning/alphazero_ttt_demo.md §5),
from one `<prefix>_sweep.csv` that `lake exe alphazero-ttt` writes: the net-alone misses
against the solved game, split by the exact value of the position, by the stones on the
board, and by whether self-play ever stood at the position (the `seen` column) — the
instrument for the gap between an unbeaten match record and a sweep under 100%.

  python scripts/demos/ttt_sweep_stats.py runs/2026-09-29-alphazero-ttt/n3_sweep.csv [n=3]
"""
import collections, csv, sys

path = sys.argv[1]
n = int(sys.argv[2].split("=")[-1]) if len(sys.argv) > 2 else 3
nc = n * n
rows = list(csv.DictReader(open(path)))

def stones(idx):
    s = 0
    for _ in range(nc):
        s += (idx % 3) != 0
        idx //= 3
    return s

total = len(rows)
miss = [r for r in rows if r["optimal"] == "0"]
seen = [r for r in rows if r.get("seen", "0") == "1"]
print(f"{total} decision positions in the sweep, {len(miss)} net-alone misses ({100 * len(miss) / total:.2f}%)")
if "seen" in rows[0]:
    miss_seen = sum(1 for r in miss if r["seen"] == "1")
    print(f"self-play stood at {len(seen)} of them ({100 * len(seen) / total:.1f}%); "
          f"misses among those: {miss_seen} of {len(seen)}; among the rest: {len(miss) - miss_seen} of {total - len(seen)}")
byv = collections.Counter(int(r["exact"]) for r in miss)
allv = collections.Counter(int(r["exact"]) for r in rows)
print("misses by the exact value for the mover:",
      ", ".join(f"{name} {byv[v]}/{allv[v]}" for v, name in ((-1, "lost"), (0, "drawn"), (1, "won"))))
bys = collections.Counter(stones(int(r["index"])) for r in miss)
alls = collections.Counter(stones(int(r["index"])) for r in rows)
print("misses by stones on the board:", ", ".join(f"{s}: {bys[s]}/{alls[s]}" for s in sorted(alls)))
sg = collections.Counter(); tot = collections.Counter(); mse = 0.0
for r in rows:
    z = int(r["exact"]); t = float(r["value"]); ts = 1 if t > 0.5 else (-1 if t < -0.5 else 0)
    tot[z] += 1; sg[z] += ts == z; mse += (t - z) ** 2
print("value head, sign agreement by exact value:",
      ", ".join(f"{name} {sg[z]}/{tot[z]} ({100 * sg[z] / tot[z]:.1f}%)" for z, name in ((-1, "lost"), (0, "drawn"), (1, "won"))),
      f"; mse {mse / total:.4f}")

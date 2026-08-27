# 2026-08-27 — the verified wall clock for RSB-A2 and RSB-A1, measured

`sec:r50_a2_a1_cost`'s tier table carried `[todo]` in the *verified wall clock* column, and the
section closed with an extrapolation: *"scaling by the one tier where both paths have a number
($19.2$ verified minutes per epoch against $7.5$) puts a verified A2 near seven days and A1 near
fifteen."* This replaces that with a measurement.

## Method

**A3 is the anchor, and it is the only tier that can be.** It is the one recipe with BOTH a
completed verified run (19.2 min/epoch, 32.1 h, four 4060 Ti, fp32) and a renderable graph, so it
fixes the one quantity a device probe cannot see — what the trainer adds around the executable.

    scripts/bf16_device_step.py --replicas 4 --reps 12 <artifact>

⚠ `bf16_device_step.py` and not a trainer's own `ms/step`: a trainer figure is a SYSTEM result, and
this repo has recorded three separate ways it misleads (the shim feed, the f32 all-reduce, and the
parameter round trip, which is off by default). This script compiles the artifact, feeds zeros and
times the executable — so what it reports is the graph's. Invokes per epoch is
`1,281,167 / (per-device batch × 4)`.

## ⚠⚠ THE OVERHEAD MODEL, and the first version of this file got it wrong

The anchor gives `230.2 − 191.4 = 38.8 ms` of trainer cost around the executable, and that can be
carried to other rows as a **multiplier** (1.203×) or as an **additive constant**. They agree at
fp32/160 by construction and diverge everywhere else — by 17 % on the bf16 rows, where a smaller
compute makes a fixed overhead matter more. The first version of this file used the multiplier
without saying so.

⭐ **A cross-net check settles it, and it is the repo's own ViT finding one net over**
(`bf16_renderer.md`: ViT's device figures are four-replica, so *"the only thing a trainer adds is
the data feed — ~65 ms at 128 img/dev for BOTH Tiny and S, i.e. an additive constant"*):

| | overhead | per image | at $224^2$-equivalent |
|---|---|---|---|
| ViT, 512 img/invoke, 224² | 65.0 ms | 0.127 ms | 0.127 ms |
| R50/A3, 256 img/invoke, 160² | 38.8 ms | **0.152 ms** | 0.297 ms |

▶ **Scaling by IMAGES puts two different nets 20 % apart; scaling by BYTES puts them 2.3× apart.**
So the cost is the host blob patch and the per-invoke overhead, not the pixel feed — which is what
the depth-$n$ prefetch exists to hide, and evidently does. Every row below is
`device + 38.8 × (images per invoke / 256)`.

## Measured, four 4060 Ti, 4 replicas

| render | device ms | + overhead | trainer ms | min/epoch | A2 / A3 | A1 (600 ep) |
|---|---|---|---|---|---|---|
| **A3 anchor** 160 fp32 — *a real run* | 191.4 | 38.8 | 230.2 | **19.2** | **32.0 h** | — |
| A3 160 **bf16** | 126.6 | 38.8 | 165.4 | **13.8** | **23.0 h** | — |
| A2/A1 `8×64` fp32 + EMA + sd | 341.1 | 38.8 | 379.9 | 31.7 | 158.4 h | 316.8 h |
| A2/A1 `8×64` bf16 | 208.1 | 38.8 | 246.9 | 20.6 | 103.0 h | 205.9 h |
| A2/A1 `4×128` fp32 + EMA + sd | 641.0 | 77.6 | 718.6 | 30.0 | 149.8 h | 299.7 h |
| **A2/A1 `4×128` bf16 + EMA + sd** | **376.9** | 77.6 | 454.5 | **19.0** | **94.8 h (3.9 d)** | **189.5 h (7.9 d)** |

⚠ The `4×128` fp32 row needs `LEAN_MLIR_MEM_FRACTION=0.97`; at the plugin's 0.75 default it does not
fit. ⭐ The book quotes the last row, the most faithful arrangement and also the fastest.

## What the numbers say

⭐ **A3 in bf16 would finish in 23 hours rather than 32.1** — a 1.40× wall clock against the graph's
1.51×, the gap being the additive overhead the speedup cannot touch. ⚠ It does not restate the
result: 77.91% is an fp32 number and stays one. **The 160 tier had no bf16 twin at all until now**,
which is §4c's rule landing on the one recipe that has actually run.

⭐ **The ghost-BN-aligned factorisation is also the FASTER one.** `4×128` runs 30.0 min/epoch
against `8×64`'s 31.7 in fp32, and 19.0 against 20.6 in bf16 — a bigger per-device batch amortises
the per-invoke overhead over twice the images. The arrangement that closes half the batch-norm gap
does not cost, it pays.

⭐ **EMA and stochastic depth are free in time as well as memory**: 341.1 against 338.4 ms/step,
**+0.8 %**, beside a memory answer of 0.19 GiB and 4 KB.

⭐ **bf16 on the graph**, four replicas with the all-reduce inside: **1.51×** at A3's 160²,
**1.63×** at `8×64`/224² (a clean pair differing in precision alone), **1.70×** at `4×128`.

⚠ **The old extrapolation was close but flattering.** *"Near seven days and near fifteen"* for the
fp32 8×64 arrangement; measured, 6.6 and 13.2. The scaling route was right in shape and ~7 % out.

## ⚠ What these are not

⛔ **Not a run.** No epoch of A2 or A1 has executed on either path. These are a measured device step
times a measured dilution.

⚠ **The bf16 rows are not directly comparable to A3's 32.1 h**, which was an fp32 run. The tier
table quotes the fp32 basis for that reason; the bf16 figures are in the section's closing sentence.

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
times the executable — so what it reports is the graph's.

    A3 anchor   5004 invokes/epoch (1,281,167 / 256)
                191.6 ms/step device   vs   230.2 ms/step implied by the completed run
                ⇒ trainer dilution = 1.20x

⚠ The dilution is transferred from $160^2$ to $224^2$, which is the estimate's one real assumption.
It should if anything be conservative: the overhead is the data feed and the host blob patch, the
patch is parameter-count-identical, and the feed hides behind a compute that grows faster than it
does.

## Measured, four 4060 Ti, 4 replicas

| render | device ms/step | invokes/epoch | trainer ms/step | min/epoch | A2 (300 ep) | A1 (600 ep) |
|---|---|---|---|---|---|---|
| **A3 anchor** (160, fp32, 8×64) | 191.6 | 5004 | 230.2 | **19.2** | — | — |
| 8×64 fp32 | 338.4 | 5004 | 406.7 | 33.9 | 169.6 h | 339.2 h |
| **8×64 fp32 + EMA + sd** | **341.1** | 5004 | 409.9 | **34.2** | **170.9 h (7.1 d)** | **341.8 h (14.2 d)** |
| 8×64 bf16 | 208.1 | 5004 | 250.1 | 20.8 | 104.2 h (4.3 d) | 208.5 h (8.7 d) |
| 4×128 fp32 + EMA + sd | 641.0 | 2502 | 770.3 | 32.1 | 160.6 h (6.7 d) | 321.2 h (13.4 d) |
| **4×128 bf16 + EMA + sd** | **376.9** | 2502 | 453.0 | **18.9** | **94.4 h (3.9 d)** | **188.9 h (7.9 d)** |

⚠ The `4×128` fp32 row needs `LEAN_MLIR_MEM_FRACTION=0.97`; at the plugin's 0.75 default it does not
fit. See `runs/2026-08-27-r50-a2-a1-ema-fifth-region/README.md`.

## What the numbers say

⭐ **The old extrapolation was right.** "Near seven days and near fifteen" against a measured
**7.1** and **14.2** days on the same fp32 basis A3 ran at. Two routes to the same figure, one
scaled from epoch times and one from device steps.

⭐ **Cross-check, and it is independent.** The verified path costs **2.40×** the JAX reference per
image (3279 ms for 2048 images against `a2accum`'s 1368), where the book's epoch-time route gives
**2.6×**. Within 8% by two different measurements.

⭐ **EMA and stochastic depth are free in time as well as memory**: 341.1 against 338.4 ms/step,
**+0.8%**. The memory answer was 0.19 GiB and 4 KB; the time answer matches it.

⭐⭐ **The ghost-BN-aligned factorisation is also the FASTER one.** `4×128` runs 32.1 min/epoch
against `8×64`'s 34.2 in fp32, and 18.9 against 20.8 in bf16 — a bigger per-device batch amortises
better even though it halves the invoke count. So the arrangement that closes half the batch-norm
gap costs nothing; it pays.

⭐ **bf16 on the graph, at four replicas with the all-reduce inside**: **1.63×** at `8×64`
(338.4 → 208.1, a clean pair differing in precision alone) and **1.70×** at `4×128`
(641.0 → 376.9, both carrying EMA and sd).

## ⚠ What these are not

⛔ **Not a run.** No epoch of A2 or A1 has executed on either path. These are a measured device step
times a measured dilution.

⚠ **The bf16 rows are not directly comparable to A3's 32.1 h**, which was an fp32 run. The tier
table quotes the fp32 basis for that reason; the bf16 figures are in the section's closing sentence.

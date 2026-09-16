# EfficientNet-B0 bf16 `SHIM_WORKERS` sweep — 4× RTX 3060, 2026-09-12

Budgets `enet-default-4gpu` (`emarmsdp64dropdobf16`: RMSProp + EMA + stochastic depth + classifier
dropout, 4 × bs 64). Run by `sweep.sh` through `scripts/bf16_probe_3060.sh`, row `enetema`.

**Preconditions, all checked on the day:** exe rebuilt 22:37 UTC, after `13d90e68` (mimalloc RSS)
and `46dfa714`; `scripts/regen_jax_generated.sh box` → 78 artifacts in sync (so the runtime shims
carry `4a0a2781`'s determinism-OFF default); GPUs idle before the first arm. Window
`WARM=200 STEPS=1000` — 800 measured steps, twice the 2026-09-11 table's. Per-step series
(`step ms wait issue invoke`) in `steps_*.tsv`.

| arm | workers | median | mean | p90 | min | steps > 500 ms |
|---|---|---|---|---|---|---|
| fed | **4** | **134** | **134** | 151 | 102 | 0 / 800 |
| fed | 6 | 136 | 135 | 153 | 99 | 0 / 800 |
| fed | 8 | 137 | 136 | 153 | 103 | 0 / 800 |
| fed | 10 | 137 | 137 | 154 | 102 | 0 / 800 |
| fed | 12 | 139 | 140 | 161 | 101 | 1 / 800 |
| synth | — | 98 | 97 | 99 | 96 | 0 / 800 |

ms/step.

## Reading

* **Flat from 4 to 10, up at 12.** 4 is the cheapest point on the plateau and is what the conf now
  sets. The conf's old 8 and its header's 14 were both set against producers capped at ~1.7 cores
  by the stale-shim determinism bug.
* **No stall tail on this box.** Mean = median on every arm and one step over 500 ms in 4,000 fed
  steps. The 2026-09-11 table's bf16 row (149 median / **279 mean** / 170 p90) was measured on a
  4× 4060 Ti with 32 threads (`runs/2026-09-11-imagenet-probe-postfix/README.md`) and does not
  reproduce here.
* **The feed still costs 37 ms of 134 (28%)**, and that cost does not move with the worker count —
  the producers' aggregate CPU is the ceiling, not their number. The 97 ms synth arm is the
  compute floor.

## ETA

5,004 × 134 ms = 11.2 min/epoch of steps. MNv2's run on this box realised ~5% over its probe for
eval and checkpointing, so **~11.7 min/epoch × 350 ≈ 68 h (2.8 d)**. At the compute floor it would be
~47 h. The forecast is to be checked against consecutive checkpoint mtimes once the run is going.

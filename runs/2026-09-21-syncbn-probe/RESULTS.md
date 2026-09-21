# Sync-BN probe — bf16 ms/step for the §3.5 ResNet legs, and the A3 4×128 render

2026-09-21, ares (4× RTX 4060 Ti), `scripts/bf16_probe_3060.sh` with `run.sh` / `run2.sh` here:
600 steps, clock from step 200, `PJRT_FFI_RESIDENT=1`, each row at its job conf's workers.
Median is the number (`imagenet_mean_is_a_memory_leak`); the mean is shown for the tail.

| row | variant | bs/replica | arm | median | min | p90 | mean | per-replica BN, same box (09-10 / 09-11 median) |
|---|---|---|---|---|---|---|---|---|
| R34 2018 | `momdp64bf16` | 64 | fed | 139 | 136 | — | 139 | 142 / 154 |
| R50 2018 | `momdp64bf16` | 64 | fed | 228 | 224 | 233 | 228 | 222 / 222 |
| R50 A3 | `lambaccdp8x64wxclipbcebf16` | 64 | fed | 153 | 141 | 164 | 153 | 144 / 139 |
| R50 A3 | `lambaccdp4x128wxclipbcebf16` | 128 | fed | 282 | 245 | 315 | 546 | — (new render) |
| R50 A3 | `lambaccdp4x128wxclipbcebf16` | 128 | synth | 232 | 229 | 233 | 231 | — |
| R50 A3 | `lambaccdp4x128wxclipbcebf16` | 128 | fed (repeat) | 285 | 251 | 305 | 404 | — |

**Sync-BN's cost** (three small collectives per BN layer, §2e): none measurable on R34, +3 % on
R50 2018, +6–10 % on A3 8×64 — inside §2e's +2–10 % costing.

**A3 at 4×128.** It fits: XLA's compiled peak is 4.39 GiB per card at 4 replicas (8×64: 2.61 GiB;
`scripts/bf16_peak_memory.py --replicas 4`), 49 % of a 12 GB RTX 3060's 9 GiB default arena. Its
compute floor is 232 ms per micro-step of 512 images; fed it runs 282–285, so the 8 producers set
its pace (~1,800 img/s, the same ceiling the 8×64 row runs at). Per epoch of training: 2,500 ×
~284 ms = 11.8 min, against the 8×64 render's 5,000 × 153 ms = 12.8 min.

The fed 4×128 rows' mean tail (404, 546) is the feed, not the graph: the synth arm has none
(mean 231 = median). Its 400-step window is also the only one here longer than the box-wide
~40 s / ~85 s stall period (`runs/2026-09-14-invoke-stall/`); the shorter rows had less chance of
catching one. A long run on ares pays that stall whichever render it trains.

**Before the first attempt:** ares' `ffi/libpjrt_ffi.so` predated `be70ae32` (no
`pjrt_ffi_session_create_dp`, so the sharded eval refused to start). Rebuilt with
`gcc -fPIC -O2 -shared ffi/pjrt_ffi.c -ldl -o ffi/libpjrt_ffi.so`. The other box needs the same
check before any §3.5 leg.

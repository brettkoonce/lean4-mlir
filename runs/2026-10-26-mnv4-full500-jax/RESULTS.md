# MNv4-Conv-M, 500 epochs at 256² — the JAX reference half (paper tier: the paper's own tf-models yaml on timm's net)

Ran 2026-10-30 20:59 → 2026-11-11 ~11:15 UTC on ares (4× 4060 Ti, 4 × 128 × 8 accumulation = 4096), 223,575 s of
training (62.1 h) across three legs: suspended at e114 (10-31) and at e230 (11-02) for the user's own work, each time
resumed from the per-epoch full-state checkpoint (`/home/skoonce/mnv4_full_r256_2612/`, `queue.log`). At the 11-08
resume the checkout was on main 139ed446; the trainer re-synced from it differs only in default-off knobs
(`LEAN_MLIR_SEED_OFFSET`, `SHIM_MIX_CHUNK`) and the 0f224436 host-array resume load.

| | top-1 | top-5 | 95% CI |
|---|---|---|---|
| paper (arXiv 2404.10518, Table 6) | 79.9 | — | — |
| JAX reference, e500 (f32 eval, 256 / 0.875) | **79.82** | 94.84 | 79.46–80.17 |
| verified, e500 (`runs/2026-10-26-mnv4-full500-verified/`) | **80.05** | 94.95 | 79.69–80.39 |

Ladder (JAX / verified): e190 66.72 / 66.16, e405 77.06 / 77.16, e460 79.36 / 79.66, e500 79.82 / 80.05.

McNemar (`rescore.sh` → `mcnemar.txt`; JAX e500 rescored with a bitmap, 39,909 exact): both right 37,929, only JAX
1,980, only verified 2,094 — exact two-sided **p = 0.077**, not significant. The 100-epoch pair was p = 0.014 with
JAX ahead; at 500 epochs the two paths tie, both within 0.1 pt of the paper's 79.9.

At timm's 320 / crop 1.0 the JAX e500 scores 80.89 / 95.62 against the verified 81.05 and timm's own weights' 80.86
(McNemar p = 0.21; `../2026-10-26-mnv4-full500-verified/RESULTS.md`).

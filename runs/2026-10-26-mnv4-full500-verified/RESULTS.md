# MobileNetV4-Conv-M, the paper tier at 256², 500 epochs, verified half — 80.05 / 94.95 (ares 4× 4060 Ti, 2026-10-26 → 10-30)

`mnv4-full-4gpu` (render `accdp8x128wxdropdowd01bf16`, 4 × 128 accumulated ×8 to 4096, bf16 train / f32 eval,
Arm B), launched by `../2026-10-26-mnv4-full500-jax/queue.sh` at 07:34 UTC 10-26. Stopped at its epoch-190
checkpoint 10-27 17:30 to lend the box to the Part 2 grind (`../2026-10-27-part2-grind/`), resumed 10-28 12:12 by
`../2026-10-26-mnv4-full500-jax/resume.sh` (after a stale-`libpjrt_ffi.so` precheck refusal; the rebuilt shim's
CUDA path is the same code), ✅ complete 10-30 20:59: ~91 h of training, one attempt a leg, no restarts. Per-epoch
top-1 bitmaps in `bitmaps/` (gitignored).

| epoch | 30 | 100 | 190 | 300 | 405 | 460 | 495 | 500 |
|---|---|---|---|---|---|---|---|---|
| verified top-1 | 60.73 | 63.32 | 66.16 | 71.27 | 77.16 | 79.66 | 80.07 | **80.05 / 94.95** |
| JAX (same recipe) | | | 66.72 | | 77.06 | 79.36 | 79.80 | **79.82 / 94.84** |

e500: 40,023 / 50,000, 95 % CI 79.69–80.39; the paper (arXiv 2404.10518, Table 6) 79.9. The JAX half and the
McNemar pairing (exact p = 0.077, 2,094 only-verified against 1,980 only-JAX — a tie) are in
`../2026-10-26-mnv4-full500-jax/RESULTS.md`.

**At timm's test protocol, 320 / crop 1.0** (`rescore_timm.sh`, 11-11: the verified checkpoint through
`scripts/parity/score_timm.sh mnv4-in256`, eval graph `mnv4in256_fwd_eval_s320`; the JAX state through
`eval_full50k.py PROTOCOL=timm`, f32; `mcnemar_timm320.txt`):

| | top-1 | top-5 | 95% CI |
|---|---|---|---|
| timm `mobilenetv4_conv_medium.e500_r256_in1k` (timm 1.0.28) | 80.86 | — | — |
| JAX reference, e500 | 80.89 | 95.62 | 80.55–81.24 |
| verified, e500 | **81.05** | 95.64 | 80.70–81.39 |

Both checkpoints gain about 1 pt at 320 / 1.0 (+1.07 JAX, +1.00 verified), and both land on timm's own weights
(+0.03, +0.19). McNemar: both right 38,604, both wrong 7,632, only JAX 1,843, only verified 1,921 — exact
p = 0.21, a tie again.

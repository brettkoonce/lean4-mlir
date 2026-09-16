# efficientnet-imagenet-verified — EfficientNet-B0 / ImageNet-1k, VERIFIED (Lean render → XLA/PJRT), 350 epochs

**Launched** 2026-09-12 23:06:23 UTC · **Finished** 2026-09-16 00:29:28 UTC
**Status** ✅ **COMPLETE — 76.878 / 93.154 in 73 h 23 m 05 s**, zero crashes, zero thermal rests.
9 trainer launches, all deliberate: 1 epoch-1 resume test, 1 one-off feed-refresh restart, 6 planned
`REST_EPOCHS` rests.

## RESULT — 0.27 behind its own reference, and the offset is real

| | top-1 | top-5 | wall clock | steps/epoch |
|---|---|---|---|---|
| EfficientNet-B0, paper | 77.1 | 93.3 | — | — |
| phase-2 JAX reference (§7, rescored over the full 50,000) | 77.15 | 93.30 | 46.6 h | 5004 |
| **phase-4 verified (Lean render → XLA/PJRT)** | **76.878** | **93.154** | **73.38 h** | 5004 |

**−0.27 top-1 / −0.15 top-5 against the reference.** Best epochs: verified **76.906 @ ep 346**;
reference **77.220 @ ep 349**.

⚠ **The single-epoch comparison says "inside the interval"; the averaged one says "a real offset",
and the averaged one is right.** The final's 95% Wilson interval is [76.51, 77.25] and the reference
sits inside it — but one eval moves ±0.4, while a 50-epoch mean does not: over epochs 301–350 the
gap is **−0.32 / −0.19**, and over the last 10 epochs 76.860 / 93.175 against 77.182 / 93.306. This
pair is NOT a dead heat the way MobileNetV2's was (+0.01).

## Pair tracking — Δ = verified − reference, per-epoch evals averaged in 50-epoch windows

| epochs | verified top-1 / top-5 | reference | **Δ** |
|---|---|---|---|
| 1–50 | 64.27 / 85.11 | 64.10 / 84.71 | **+0.17** / +0.40 |
| 51–100 | 71.73 / 90.33 | 72.26 / 90.61 | −0.53 / −0.29 |
| 101–150 | 73.67 / 91.53 | 74.24 / 91.77 | −0.57 / −0.23 |
| 151–200 | 74.99 / 92.23 | 75.55 / 92.51 | −0.56 / −0.28 |
| 201–250 | 75.81 / 92.64 | 76.21 / 92.90 | −0.40 / −0.26 |
| 251–300 | 76.34 / 92.92 | 76.67 / 93.14 | −0.33 / −0.22 |
| 301–350 | 76.71 / 93.09 | 77.03 / 93.28 | **−0.32** / −0.19 |

The verified path led for the first 50 epochs, fell behind by ~0.55 through the middle, and closed
to ~0.32 — where it stopped closing. `enet_verified_curve.csv` and `reference_curve.tsv` hold both
curves per epoch.

## ⛔ What this pair does NOT isolate

Two things differ between the sides, not one:

1. **The lowerer** — the thing the comparison is for.
2. **The BatchNorm statistic group.** `allReduceMeanF` is the only `SHlo` constructor taking a
   replica family and every BatchNorm constructor is `SHlo n → SHlo n`, so the verified render
   normalizes over **64 per replica** while the JAX reference under `@jit` + `NamedSharding` reduces
   globally over **256**. §5.7 measured that fourfold difference as 0.02 points on ResNet-34 — but
   EfficientNet has far more BatchNorm per unit of compute (49 layers, 42,016 running-stat floats),
   and this net's EMA shadows those buffers as well as the weights, so the ResNet-34 figure does not
   transfer. ▶ `planning/global_bn_verified.md` §1 bounds it JAX-side for free (train the reference
   with per-shard BN); **worth doing before §7 claims a cause.**

A third, smaller: the drop-path and dropout masks are host-drawn here and JAX-PRNG there. That is a
variance source between runs, not a bias, and it does not survive averaging over 50 evals.

## Throughput — and the shim-loader defect that cost ~4.6 h

| window | s/epoch |
|---|---|
| epochs 3–64 (launch state) | ~690–700 |
| epochs 65–122 (degraded) | up to 1,430 |
| epochs 126–350, clean (211 of 227) | **704** |
| epochs after the fix, all 227 | 716 |
| whole run | 754 |

⛔ **One of the four tf.data shim loaders degrades, and the round-robin feed runs at its pace.** From
epoch 65 (~13 h of loader uptime) steps went 134 → 240–250 ms. In 20 of 20 samples three loaders sat
blocked in `anon_pipe_write` while the fourth never did; the trainer waited on that one. Ruled out:
CPU (no loader thread over 18%, box 58% idle), disk (zero real reads, 81% of shards in page cache),
swap, and CPU throttling (zero throttle events, 30–34 °C). The growth is in glibc malloc arena-class
mappings, and a restart clears it: 130 ms/step immediately after.

Mitigation, tested live here: `REST_EPOCHS="125 160 200 240 280 320"` with `REST_SECS=15` — the
supervisor kills right after a checkpoint and resumes with fresh loaders, ~1.75 min each. All six
fired cleanly; only the epoch right after each ran long (~835 s).

Per-loader RSS at the end of each generation (`loader_rss.tsv`), last-spawned loader last:

| generation (epochs) | siblings | the outlier |
|---|---|---|
| 123→125 | 5.26 / 5.34 / 5.54 | **9.28 GiB** (8.0 h) |
| 125→160 | 5.28 / 5.31 / 5.61 | **7.04 GiB** (7.7 h) |
| 160→200 | 5.82 / 6.13 / 6.42 | **8.44 GiB** (7.7 h) |
| 200→240 | 5.35 / 5.39 / 5.48 | 5.67 GiB — **no outlier** |
| 320→350 | 5.94 / 5.98 / 6.06 | **9.51 GiB** (5.9 h) |

⚠ So it is the last-spawned loader in four generations of five, and the timing varies (5.9 h to
never). Probabilistic, not positional. Root cause unknown — no `py-spy` on the box.

▶ **At the clean 704 s the run would have taken 68.5 h**, against the reference's 46.6 h (455–465 s
train + ~17 s val per epoch). Neither number is a renderer measurement: most of the gap is this
net's CPU augmentation feed (AutoAugment + RandAugment per image), not the lowerer — the 2026-09-12
sweep put the compute floor at 97 ms/step against 134 fed (`runs/2026-09-12-enet-bf16-sweep/`).

## Resume — exercised 8 times, first time deliberately

This is the first run on the checkpoint/resume fix (`367bb28b`): the BatchNorm running stats and
their EMA shadow are written to `<ckpt>.bn` beside the blob, and the companion, blob and epoch
marker are each written through a temp file and renamed. Before it, a resume restarted `ema_bn` at
ZERO under the mature 0.9999 decay — ~61% of the eval's BN statistics still zero one epoch later.

The epoch-1 test was a deliberate SIGTERM right after the checkpoint: attempt 2 read the companion
back **hash-identical** (`7468205912590194089`, 336,128 bytes = 2 × 42,016 × 4). Seven more resumes
followed from the planned rests, each logging its companion hash. Accuracy shows no step at any
restart boundary — 73.68 / 73.61 / 73.64 across the epoch-123 restart, and epochs 124 and 125 were
scored right after resumes.

## Config (read from the run's own startup banner, not the conf)

`emarmsdp64dropdobf16` — RMSProp (ρ 0.9, μ 0.9, ε 1e-3, mean-square init 1.0, TF convention, no
bias correction), peak LR 0.016 with 5-epoch warmup then ×0.97 every 2.4 epochs; EMA decay
min(0.9999, (1+t)/(10+t)), shadow starting at the weights, **eval and checkpoint score the shadow**;
stochastic depth over 9 sites (keeps 0.973 → 0.813); classifier dropout keep 0.8 over
`tensor<256x1280xf32>`; BatchNorm 49 layers / 42,016 stat floats, decay 0.99; 4 replicas × bs 64 =
global 256, 5,004 steps/epoch; `PJRT_FFI_RESIDENT=1` (852 tensors, 80.7 MB resident); `SHIM_WORKERS=4`.

## Files

`RESULTS.md` · `enet_verified_curve.csv` (350 epochs: epoch, train_loss, lr, top1, top5) ·
`reference_curve.tsv` (the phase-2 curve, same epochs) · `full.log` (every attempt) · `attempt.log`
(the last) · `master.log` (supervisor) · `epoch_clock.tsv` (per-epoch wall clock + trainer RSS) ·
`loader_rss.tsv` (per-loader memory per epoch) · `log_ts.log` (timestamped step lines) ·
`resume_test_epoch1.log` + `resume_test.sh` (the deliberate resume test) · `restart_refresh.log` +
`restart_refresh.sh` (the one-off feed refresh) · `summarize.sh` (rebuilds the curve and these facts)
· `epoch_clock.sh`, `loader_rss.sh`, `log_ts.sh` (the monitors).

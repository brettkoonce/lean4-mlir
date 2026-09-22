# ConvNeXt-T / ImageNet-1k, VERIFIED path, 300 epochs — the re-run with `cnxInit` and the sharded eval

**✅ COMPLETE 2026-09-22 14:03:53 UTC.** Final (raw weights, e300): **40,651 / 50,000 = 81.302%
top-1, 95.608% top-5** [95% CI 80.96–81.64]. Best epoch 81.390% @ e291.

**Against the JAX reference's RAW arm, 81.51 / 95.50: −0.21 / +0.11.** Paired McNemar over the
same 50,000 images gives **p = 0.10, not significant**: the two checkpoints are not separable on
this val set (§2).

---

## 1. What ran

| | |
|---|---|
| conf | `scripts/jobs/cnx-default-4gpu.conf` via `run_cnx_verified.sh` (`systemd-run --user --unit=cnx-verified`) |
| variant | `adamdpwxclipdropbf16`, **4 × 64 = global 256**, 5,004 steps/epoch, LR 2.5e-4, 20-epoch warmup, cosine over 300, `ema := false` |
| init | **`cnxInit`** (`83d132e8`): σ = 0.02 on every conv and the head. Banner confirmed; step-0 loss 7.08 |
| eval | **sharded over 4 GPUs** (`be70ae32`): `4 × 64 = 256` per invoke, 196 invokes, last one 80 real + 176 pad |
| box | 4× RTX 3060 12 GB ("mars"), CUDA 13.3, shim `ffi/libpjrt_ffi.so` rebuilt 2026-09-18 |
| why a re-run | the 09-17 run (`runs/2026-09-17-cnx-verified-300ep/`) was killed at e67: its two arms did not share a weight init (He vs the reference's σ = 0.02) |

**Wall clock: launched 2026-09-18 14:26:13, finished 2026-09-22 14:03:53 UTC = 95.63 h.**
That includes ~3.25 h of downtime after a hardware crash at e263 (§4).

## 2. ⭐ The pairing — McNemar, paired over the same 50,000 images

Bitmaps are one byte per val image (1 = top-1 correct), in tfds `validation` file order:
* **verified**: `score-checkpoint convnext-in` on the e300 checkpoint, `LEAN_MLIR_REGION=live`,
  4 replicas, the 09-18 binary, which drains val exactly as the run's evals did.
* **reference**: `jax/scripts/eval_convnext_arms_full50k.py` with the new `DUMP_CORRECT`, on
  `/home/skoonce/convnext_t300_3060/convnext_tiny_imagenet_e300.state.npz`. That is the full
  state; the per-epoch `.bin`s are EMA only.

| pair | both ✓ | both ✗ | only A ✓ | only B ✓ | net | exact p |
|---|---|---|---|---|---|---|
| **A = reference RAW, B = verified** | 38,850 | 7,445 | 1,903 | 1,802 | −101 | **0.100** |
| A = reference EMA, B = verified (arm-mismatched) | 38,861 | 7,440 | 1,908 | 1,791 | −117 | 0.056 |
| CONTROL: A = reference raw, B = reference EMA | 40,721 | 9,199 | 32 | 48 | +16 | 0.093 |

* **The reference arms reproduce.** Raw 40,753 (81.51 / 95.50) is exactly the documented raw arm.
  EMA gives 40,769 = 81.54 / 95.51; the book's 81.53 came from `eval_convnext_full50k.py`, a
  different eval path, a few images apart.
* **The verified rescore reproduces within one image**: 40,652 against the run's in-training
  40,651, top-5 equal. That is the per-process XLA compile noise measured on 2026-09-18 (ViT, 9
  images once). The e263 rescore matched exactly.
* **Image alignment was CHECKED, not assumed.** The two models agree on **92.6%** of images; if
  the orders were unrelated that would be **69.7%**, and a shuffled pairing measures 69.6%.
* **p is cross-checked**: `scipy.stats.binomtest(1802, 3705)` = 0.1004.
* ⚠ **Scope.** This is about these two CHECKPOINTS. The arms still differ in host-drawn vs
  `jax.random` drop-path/mixup masks and in their seeds, so this is a statement of "not
  separable", not of "identical". About 7.4% of images are decided differently; that is the
  run-to-run spread of two ConvNeXt-Ts, and a tie needs no smaller discordance than that.

Reproduce (`runs/**/*.bin` is gitignored, so the bitmaps are regenerated rather than
committed; `mcnemar/mcnemar.txt` is the committed output):

```
LEAN_MLIR_VARIANT=adamdpwxclipdropbf16 LEAN_MLIR_REGION=live LEAN_MLIR_REPLICAS=4 PJRT_REPLICAS=4 \
  PJRT_FFI_RESIDENT=1 SHIM_PYTHON=… PJRT_PLUGIN=… LEAN_MLIR_DUMP_CORRECT=$M/verified_e300 \
  .lake/build/bin/score-checkpoint convnext-in data
GEN=jax/.lake/build/generated_convnext_tiny_imagenet_full.py \
  CKPT=/home/skoonce/convnext_t300_3060/convnext_tiny_imagenet_e300.state.npz \
  DUMP_CORRECT=$M/reference_e300 python3 jax/scripts/eval_convnext_arms_full50k.py
python3 scripts/mcnemar.py $M/reference_e300_raw.bin $M/verified_e300.bin
```

⚠ `scripts/mcnemar.py` crashed on this pair with an `OverflowError`: its exact p used
`2.0 ** (b + c)`, a float, which overflows past ~1,024 discordant images. It had only ever seen
Imagenette-sized counts. Fixed to integer arithmetic in the same commit.

## 3. ⚠ Reading the per-epoch curve — the reference's is EMA, ours is raw

The reference's per-epoch eval is `eval_batch(ema_params, …)`
(`jax/generated/generated_convnext_tiny_imagenet_full.py:1827`, decay 0.9999 ≈ a 2-epoch time
constant). Ours is raw weights. So the curve comparison is arm-mismatched:
* the EMA lags while accuracy climbs fast, so ours LED from e3 to e17 (+2.56 at e10);
* at the peak LR, averaging is worth several points, so ours trailed by up to ~−4 (e24–e60);
* the gap closes as the cosine anneals.

| window | verified (raw) | reference (EMA) | Δ top-1 / top-5 |
|---|---|---|---|
| e1–50 | 53.01 / 72.47 | 54.75 / 72.95 | −1.75 / −0.48 |
| e51–100 | 74.35 / 92.34 | 77.62 / 93.84 | −3.27 / −1.49 |
| e101–150 | 76.75 / 93.63 | 79.44 / 94.74 | −2.69 / −1.11 |
| e151–200 | 78.61 / 94.50 | 80.60 / 95.31 | −1.99 / −0.82 |
| e201–250 | 80.20 / 95.19 | 81.25 / 95.52 | −1.06 / −0.33 |
| e251–300 | 81.20 / 95.53 | 81.48 / 95.51 | −0.28 / +0.02 |

"15 of 300 epochs above the reference" (all early, while the EMA lags) is the same mismatch.
⛔ `summarize.sh` says the curve is "unaffected"; that is wrong. Only the FINAL pairing (§2) is
raw-to-raw. A raw reference curve cannot be rebuilt: its raw weights survive only for e298–300.

## 4. The crash at e263 — hardware, and the resume held

* **2026-09-21 22:25 UTC**: DIMM `CPU_SrcID#0_MC#1_Chan#1_DIMM#0` (32 GB) began logging corrected
  ECC errors under this run's load: 8,835 in 14 min. **Uncorrectable** errors followed at
  22:35:57 and 22:38:29, and the box reset at ~22:39 (rebooted 22:42, kernel 7.0.0-30 → -31).
  tmux and every `systemd-run --user` unit died with it.
* **The e263 checkpoint (22:30:19) predates the uncorrectable errors**, and rescoring it matched
  its in-run eval exactly: 40,552 / 50,000, top-5 47,754. Resumed 2026-09-22 01:54 UTC with the
  DIMM still installed and a new watcher, `edac_watch.sh` → `edac.tsv`.
  **Zero** memory errors followed.
* **The firmware mapped out exactly 16 GiB at that boot** (the `Memory:` boot line dropped by
  16,777,216 K). The page cache then held ~114 GiB against the 137 GiB train split, so the loaders
  read 100–170 MB/s from disk every epoch. **Pace went 1,096 → 1,178 s/epoch.** The fix is
  `planning/streaming_val.md` (now on main as `8182b6e1`): it frees the trainer's 30 GB val buffer.

## 5. Health

| | |
|---|---|
| pace | 1,096 s/epoch mean over e3–e263 (1,081–1,123), 1,178 after the crash |
| eval (sharded) | **28–31 s/epoch**, against ~101 s single-GPU on the killed run |
| launches | 3: the original, the deliberate epoch-1 resume test, and the post-crash resume |
| resume test (e1) | 5/5: resumed at 1, checkpoint byte-identical, no `.bn`, stepping, e2 > e1 (`resume_test.log`) |
| shim respawns | 29, one every 10 epochs (e10–e290) on schedule; loader RSS flat at 4.6–6.1 GiB |
| thermal | 0 events, GPUs 52–65 °C |
| fault watcher | never fired; `WATCH_EVENT` is its end-of-run record |

## 6. Files

`full.log`, `attempt.log`, `master.log` (supervisor), `log_ts.log` (timestamped key lines),
`epoch_clock.tsv`, `loader_rss.tsv`, `edac.tsv`, `resume_test.log`, `WATCH_EVENT`,
`cnx_verified_curve.csv` + `reference_curve.tsv` (from `summarize.sh` / `extract_reference.sh`),
`mcnemar/` (score logs + `mcnemar.txt`), and the watcher scripts.

# MobileNetV4-Conv-M, bf16, EMA — verified path, 100 epochs

**✅ 76.678 % top-1 / 93.144 % top-5** (38,339 / 46,572 of 50,000, EMA shadow), epoch 100/100,
95 % CI 76.31–77.05. Landed 2026-09-27 11:53:47 UTC. **18.46 h** wall, including **one restart**
for a loader fault at e84 (below). EDAC 0 / 0 throughout.

The verified half of the MNv4 100-epoch pair (`planning/imagenet_parity.md` §7 R1). Job
`scripts/jobs/mnv4-default-4gpu.conf`, variant `emaaccdp8x128wxdowd005bf16`, checkpoint tag `e100`,
unit `mnv4-verified`, launched 2026-09-26 17:26:09 UTC on the 4× 3060 box. Its partner is
`runs/2026-09-26-mnv4-jax-bf16-100ep/` (76.572 / 92.980), read in place by `summarize.sh`.

## ⭐ The pair

| | verified (this run) | JAX reference | Δ |
|---|---|---|---|
| top-1, e100 | **76.678** | 76.572 | **+0.11** |
| top-5, e100 | **93.144** | 92.980 | **+0.16** |
| mean Δ over e90 / 95 / 100 | | | +0.15 / +0.18 |

⇒ **A tie.** +0.11 top-1 is under a third of one CI half-width (±0.37). Both paths score the EMA
shadow, train the same timm-parity net (9,715,512 parameters) at the same shape (grad-accum 8 ×
(4 × 128) = 4096) and the same `default` recipe.

The per-epoch gap (`summarize.sh`, every 5th epoch): verified is behind at e5 (−2.19), ahead from
e10 (+4.40) and **narrows through the anneal**, not monotonically (it ticks up at e25, e60, e70):
+1.13 at e20, +0.48 at e50, +0.32 at e80, +0.11 at e100. The two curves converge, as every other pair in the book does.

⚠ Unpaired as it stands: this run wrote 20 per-image bitmaps (every scored epoch), but the JAX
trainer writes none. Its e100 EMA weights are kept, so a per-image re-score would pair it.

## Verified from the run's own output

`emaaccdp8x128wxdowd005bf16` · `cosine+warmup 5ep, baseLR 0.004000` · `DATA-PARALLEL: 4 replicas x
bs 128 = global batch 512, 2496 steps/epoch` (312 optimizer steps of k = 8) · ⭐ `new-batch weight
0.001256 = 1 − 0.990000^(1/8), compensated for grad-accum` — k spelled once · `running-stats BN:
77 layers, 67904 stat floats → eval via @mnv4in_fwd_eval` · `EMA: … decay min(0.999900,
(1+t)/(10+t)) … EVAL AND CHECKPOINT SCORE THE SHADOW` · train step RESIDENT (1,165 tensors,
185.3 MB). The resume printed `resumed BN running stats + ema_bn from … .bn`.

## ⚠ The loader fault, and the one restart

The conf ran its shim producers bare (no `LEAN_MLIR_SHIM_RESPAWN_EPOCHS`). Epochs held at 643 s
mean through e76, then climbed: 741, 799, 694, 763, 990, 818, **1,095 s** (e77–83). `fault_watch`
fired at 08:41:27 UTC on three consecutive epochs over 1.25 × its 629 s baseline, with 131.6 GiB
host memory free, i.e. not a host-memory stall. Producers at detection (15.2 h old):

| pid | RSS | arena-class |
|---|---|---|
| 1896515 | 6.64 GiB | 4.54 GiB |
| 1896611 | 7.29 GiB | 5.12 GiB |
| 1896800 | 7.69 GiB | 5.63 GiB |
| **1896983** | **10.10 GiB** | **8.19 GiB** |

One producer running away while the others creep: B0's loader fault
(`runs/2026-09-12-enet-verified-350ep/`, cured there by `REST_EPOCHS`). Handled the same way: stopped right after
the **e84** checkpoint (08:53:37) and relaunched the same conf (08:54:01), which resumed at e84
with weights, optimizer, EMA, BN running stats and `ema_bn` restored. Only the data stream
restarts (a fresh shuffle order). e86–100 then averaged 664 s. The whole sequence is in
`RESTARTS`, `restart_after_ckpt.sh`, `WATCH_EVENT.loader-fault-e83` (the detection, verbatim) and
`WATCH_EVENT.run-stopped-e84` (the watcher's end-of-run note from the stop).

⇒ Every verified conf but R34's now sets `LEAN_MLIR_SHIM_RESPAWN_EPOCHS=10` (committed
separately), so the next long run replaces one producer every 10 epochs instead of needing this.

## The launch that did not happen

The run was queued behind the JAX run by `chain_after_jax.sh`. Its first launch (17:25) was
**refused by the conf's precheck**, correctly, and started nothing (`failed-precheck-1725/`): a
`systemd-run --user` unit does not inherit the shell's `PATH`, so `pc_exe`'s `lake build` failed;
and `ffi/libpjrt_ffi.so` (gitignored, 2026-09-18) no longer matched `ffi/pjrt_ffi.c` (changed
2026-09-25). The `.so` was rebuilt with the precheck's own `gcc` line and the run launched a minute
later with `--setenv=PATH`.

## Pace

643 s/epoch through e76 and 664 s after the restart, eval (every 5) included; 18.46 h wall with
the ~3 h fault window and the restart. The 2026-09-25 probe measured 252 ms/micro-step, i.e.
~17.7 h; the conf's `~42 h` is a 4× 4060 Ti number, where this net is feed-bound.

## Artifacts

* checkpoint `.lake/build/mnv4in_emaaccdp8x128wxdowd005bf16_ckpt_xla_e100.bin{,.bn,.epoch}` (marker 100)
* per-image top-1 bitmaps for the 20 scored epochs, `bitmaps/mnv4_emaaccdp8x128wxdowd005bf16_e{5,10,…,100}.bin` —
  LOCAL ONLY (`runs/**/*.bin` is gitignored)
* `attempt.log` / `full.log` / `master.log`, `log_ts.log`, `epoch_clock.tsv`, `loader_rss.tsv`,
  `edac.tsv`, the watcher scripts, `chain_after_jax.sh`, `restart_after_ckpt.sh`, `RESTARTS`,
  `WATCH_EVENT` (the end-of-run note), the `WATCH_EVENT.*` files and `failed-precheck-1725/`
* `mnv4_verified_curve.csv` — written by `summarize.sh`

## Book rows this bears on (not edited here)

`content.tex` :8041 — "the 75.48% above is this network's target … a `TBD` accuracy, because the
network has been timed and has not been trained to convergence". It now has; and 75.48 (:7972,
:7978, :8020) is the pre-timm transcription's JAX number, superseded by this pair's 76.57.

## Reproduce

    bash runs/2026-09-26-mnv4-jax-bf16-100ep/summarize.sh
    bash runs/2026-09-26-mnv4-verified-bf16-100ep/summarize.sh

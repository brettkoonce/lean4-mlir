# MNv4-Conv-M 50-epoch pair: JAX reference × verified PJRT, same recipe, bf16

2026-09-25. First runs of the timm-parity net (90e4af7e). The owner's call: bf16, a 50-epoch
half-length recipe on both paths, PJRT first. Both run as overnight chunks (`supervise.sh`
START_AT/STOP_AT). Code first, then launch.

## The recipe (the JAX reference's tier-2 recipe at 50 epochs)

| knob | JAX `default` today | this pair (both paths) | PJRT today (`adamdp64bf16`) |
|---|---|---|---|
| epochs / cosine | 100, warmup 5 | **50**, warmup 5 | 100 (conf 500) |
| optimizer | AdamW 0.004 @ 4096 | same | AdamW 1e-3 @ 256 |
| batch | 512 × accum 8 (128/GPU) | **4 × 128, accum 8** | 4 × 64, no accum |
| BN statistic group | 512 (the jit mesh) | 512 (sync-BN over 4 × 128) | 256 |
| weight decay | 0.05, not on norm/bias | same | 1e-4 on everything |
| label smoothing | 0.1 | same | 0.1 ✓ |
| classifier dropout | 0.1 | same | none |
| augmentation | RRC + flip + RandAugment N2 m9 | same (the shim is byte-identical) | ✓ |
| EMA | 0.9999, warmup-corrected, eval on EMA | same | none |
| eval | every 5 epochs + last, running BN | same | every epoch |

## PJRT: one new variant, `emaaccdp8x128wxdowd005bf16`

It reuses the ResNet-family optimizer stage (`optAllParams`: `.adamwAccum 8`, `ema`, the `wx`
mask) and a classifier-dropout splice at the head relu, the site timm and the JAX reference use.
The driver needs no change: it keys `emaOn` / `accOn` / `accK` / `cdOn` off the name.
- `mobilenetv4AdamTrainStepFaithfulB` takes the recipe axes, trailing and defaulted, so every
  committed render is byte-identical. The single-device peer `emaacc8x128wxdowd005bf16` is
  `mnv4-dp-check`'s reference.
- Eval stays on `mnv4in_fwd_eval.mlir` at 64; the driver reads the eval batch off it.
- `mnv4ImagenetVerified.dropoutKeep := some (0.9, 1280)`.
- Epoch = 2,496 micro-batches (`LEAN_MLIR_G2_STEPS`) = 312 optimizer steps, the reference's
  1,281,167 // 4096. 2,502 is not a multiple of k and the driver refuses a straddling cycle.
- ⚠ `dropoutP` no longer lifts to the operand's 4-D shape (`StableHLOPretty`): MNv4's head relu
  carries `[1280,1,1]`, and a 4-D multiply against the `[B, 1280]` mask input does not parse.
  EfficientNet's renders are byte-identical.

## Gates (runs/2026-09-25-mnv4-half-gates)
| gate | result |
|---|---|
| `mnv4-dp-check` 4 × 128 vs 1 × 128, duplicated batch | forward bit-exact (bnstat 67,904/67,904); m 1.1e-2, G 2.1e-2 ≤ 5e-2; scalars and mask exact |
| same, sum-not-mean control (464 divisors 4 → 1) | red: bnstat norm-rel 62, m 2e7 |
| `mnv4-dp-check` on `adamdp64bf16` (regression) | unchanged, m 1.7e-2 |
| `dropout-tie --net` (gate W) / its fault control | green / red |
| `variant-predicates` | 5 regions, k = 8, dropout on |
| `parse_verified_mlir.py`, `regen_verified_mlir.sh check` | green |
| trainer smoke, 2 + 1 epochs of 16 micro-batches, resume | lr 0.0008 → 0.0016 → 0.0024 (warmup over 10 updates), loss 7.14 → 7.01, EMA eval, resume at epoch 2 with BN + EMA-BN |

## JAX: a `half` recipe and a supervised conf
- `jax/MainMobilenetV4Imagenet.lean` recipe `half` = `default` at 50 epochs, emitted as
  `generated_mobilenet_v4_imagenet_half.py` (the only diff from `default` is `EPOCHS`).
- `scripts/jobs/mnv4-half-jax-4gpu.conf`, the first JAX-path conf. `epoch_now` = newest
  `_e<N>.state.npz`, and `CMD` wraps `LEAN_MLIR_RESUME`.

## Throughput (runs/2026-09-25-mnv4-timm-probe)
4 × 128 bf16 fed at 4 workers: median 260 / mean 362 ms per micro-batch, so 11–15 min per epoch.
8 workers is worse. 50 epochs is two midnight–8 AM chunks per path.

## Status
- [x] supervise.sh START_AT / STOP_AT (+ `chunktest` self-test)
- [x] JAX half recipe + conf
- [x] PJRT variant render + gates + smoke
- [ ] PJRT 50 ep (2 nights), then JAX 50 ep (1–2 nights)

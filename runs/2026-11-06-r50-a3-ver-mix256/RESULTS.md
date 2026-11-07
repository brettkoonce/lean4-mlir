# ResNet-50 RSB-A3, VERIFIED path, Mixup/CutMix drawn per 256 rows — mars (4× RTX 3060), 2026-11-06 → 07

The verified half of the per-chunk draw A/B (`../2026-11-06-r50-a3mix8-jax/RESULTS.md`). The verified
shim drew one Mixup/CutMix per 512-row record (4 replicas × 128); here `SHIM_MIX_CHUNK=256`, two draws a
record. Everything else is the A3 verified recipe at seed 1.

* `MIX_CHUNK=256 RUNDIR=runs/2026-11-06-r50-a3-ver-mix256 scripts/supervise.sh r50-a3-wxclip4x128-bf16-4gpu`
  (checkpoint tag `2611m256`); every shim producer logged "drawn per 256 rows"
* one attempt, 17:24:38 UTC 11-06 → ✅ 14:21:47 UTC 11-07 (20.95 h)
* `after.sh` (its own systemd unit): bitmap check, pairing, then the ViT-S resume (`after.log`)

## Top-1 at epoch 100, all 50,000 val images

| run | draw | top-1 |
|---|---|---|
| **verified, this run (seed 1)** | per 256 | **77.99** (38,996) / 93.69 |
| verified 10-09 / s2 / s3 | per 512 | 77.91 / 77.82 / 77.77 |
| JAX mix8 (seed 1) | per 256 | 78.14 |
| JAX s1 / s2 / s3 | per 2048 | 77.60 / 77.66 / 77.64 |
| timm RSB-A3 (paper) | per GPU batch | 78.1 |

The e100 bitmap reproduces the log's count exactly (38,996).

## Paired tests (`scripts/demos/mcnemar.py`, this run's e100 bitmap against each)

| against | discordant | McNemar exact p |
|---|---|---|
| JAX mix8 (per 256) | 3,960 | 0.23 |
| JAX s2 (per 2048) | 4,067 | **0.011** |
| JAX s3 (per 2048) | 4,079 | **0.006** |
| verified 10-09 (per 512) | 4,057 | 0.55 |
| verified s2 (per 512) | 3,997 | 0.18 |
| verified s3 (per 512) | 3,962 | 0.083 |

`10-09` is `../2026-10-09-r50-a3-verified-bf16-100ep/` (77.91, mars), not seed 1; seed 1's bitmap
(78.08, ares) is not on this box.

## Reading

* At 256 rows both paths TIE (p = 0.23), and both beat the per-2048 JAX seeds.
* Per 512 → per 256 moves the verified path by +0.08 to +0.22, inside its own 0.31 seed range and
  separating from none of its per-512 runs. The cost was in the 2048-row draw; 512 was already
  close to free.
* So the A3 verified-vs-JAX gap of the Tier-2 seeds was the reference's draw, not the verified path.

# Sync-BN for ResNet-50 and MobileNetV4 — gate logs (2026-09-21)

`planning/global_bn_verified.md` §3.4. Every log here is one run of a gate against the sync-BN
renders of this change. Four GPUs (0–3) unless noted, XLA backend, default arena unless noted.

## The split-batch gate — `imagenet-syncbn-check <net> [f32]`

R50 runs 4×32 against 1×128 (DP step rendered at run time); MNv4 runs 4×64 against 1×256.
`SYNCBN_VERBOSE=1` on the first run of each, so those logs carry the per-layer table.
f32 runs used `LEAN_MLIR_MEM_FRACTION=0.97`.

| log | stats TEST | CONTROL | stats SENSITIVITY | verdict |
|---|---|---|---|---|
| `r50_bf16.log`, `resnet50_bf16_rep2.log` | 8.6e-3, 6.7e-3 | 0.38 | 1.4e-2 | pass (at 3e-2) |
| `r50_f32.log`, `resnet50_f32_rep2.log` | 2.1e-3, 1.7e-3 | 0.38 | 1.4e-3 | pass (at 5e-3) |
| `r50bce_bf16.log`, `resnet50bce_bf16_rep2.log`, `final_resnet50bce_bf16.log` | 1.08e-2, 1.09e-2, 1.22e-2 | 0.37 | 1.5e-2 | pass (at 3e-2) |
| `r50bce_f32.log` | 2.2e-3 | 0.37 | 1.7e-3 | pass |
| `mnv4_bf16.log`, `mnv4_bf16_rep2.log` | 6.9e-3, 1.16e-2 | 0.076 | 1.6e-2 | pass (at 3e-2) |
| `mnv4_f32.log`, `mnv4_f32_rep2.log` | 1.2e-3, 1.3e-3 | 0.077 | 1.3e-3 | pass |

In every run the FORMULATION and DUPLICATED statistics and the first BN layer's split error
are 0. These runs predate the final bounds. They ran under 1e-2 bf16 / 3e-3 f32 placeholders, so
`r50bce_bf16.log`, `resnet50bce_bf16_rep2.log` and `mnv4_bf16_rep2.log` exit 1 on the split bound
alone, at 1.08–1.16e-2. The `final_*` logs are the three bf16 configs re-run by the binary built
with the final bounds. The first BCE bf16 attempt, under the 0.97 arena, died on a d2h
`CUDA_ERROR_OUT_OF_MEMORY`, and `r50bce_bf16.log` is its re-run. The `final_*` runs all pass:
`resnet50` 6.7e-3, `resnet50bce` 1.22e-2, `mnv4` 6.8e-3.

## The re-pointed and retired gates

| log | gate | result |
|---|---|---|
| `mnv4_dpcheck_{f32,bf16}_before.log` | `mnv4-dp-check` at its old 1e-4 bound | fail: gradient 1.65e-3 / 2.21e-2, forward bit-exact |
| `mnv4_dpcheck_{f32,bf16}.log` | `mnv4-dp-check` at 1e-2 / 5e-2 | pass: 1.45e-3 / 2.06e-2, `bnstat` 67,904/67,904 bit-exact |
| `mnv4_dpcheck_control_sum.log` | the same, sum-not-mean render | fails as required (forward norm-rel 80) |
| `r50_gradcheck_twopass.log` | `r50-gradcheck`, committed `adam64` | pass: tier 1 7.5e-5, tier 2 0.168 |
| `r50_gradcheck_sync.log` | `r50-gradcheck`, `R50_GC_PATH=.lake/build/r50sync` (one-replica sync graph) | pass: tier 1 5.9e-5, tier 2 0.166 |
| `r50_accum_shard_tie_vs_committed_sync_dp.log` | `r50-accum-shard-tie` against the committed (sync) DP render | fails by design: m′ rel 1.07 |
| `r50_accum_shard_tie.log` | the same, peer at `noSync` | pass: ≤ 1e-6, control 1.0e-2 |
| `r50_accum_shard_tie_160lambbce.log` | `R50_ACC_RES=160 R50_ACC_VARIANT=lambacc4x64bce R50_ACC_PEER=lambdp64bce` | pass |
| `drop_shard_check_b0.log` | `drop-shard-check` (2 GPUs, deterministic shim) | pass: 12,061,074 + 42,016 statistics bit-identical under the swap, `%loss` moves |
| `drop_shard_check_b0_control_replicate.log` | the same, `DROP_FAULT=replicate` | fails as required (① and ①b) |
| `drop_shard_check_b0_control.log` | the same, `PJRT_DP_NO_MASK_SHARD=1` | refuses on arity, as it always has |
| `r34_dp_shard.log` | `r34-dp-shard` | pass (TEST 3.6e-2, CONTROL fires) |
| `shard_check_convnext.log` | `shard-check convnext` (2 GPUs), a kept LayerNorm row | pass (TEST 5.9e-8, CONTROL 1.82) |

# EfficientNet-B0 TF recipe smoke — 2026-09-25, ares (4x 4060 Ti)

`enet-default-4gpu`'s variant `emarmsdp64dropdowxeps0001bf16` (`wx`, BN ε 1e-3 in the render; the
×0.97/2.4-epoch staircase on the global step under a 5-epoch warmup, i/16 drop-connect ramp),
40 steps/epoch capped (`LEAN_MLIR_G2_STEPS=40`), checkpoint tag `smoke` (deleted after).
`smoke.sh` reruns it.

| leg | epochs | lr printed | eval graph | top-1 / top-5 | exit |
|---|---|---|---|---|---|
| 1 | 1 | 0.003200 | `@efficientnetin_fwd_eval_eps0001` | 0.072% / 0.474% | 0 |
| 2 (resume at epoch 1) | 2 | 0.006400 | same | 0.094% / 0.496% | 0 |

At 40 steps/epoch the 5-epoch warmup is 200 steps, so both legs sit inside it: 0.016·40/200 and
·80/200. The staircase itself is past the warmup, so it is checked off-GPU instead: the driver's
formula against the JAX trainer's emitted one, at nb = 5004, over every warmup step plus 900 steps
spread across the 350 epochs — 0 mismatches at 1e-9 relative. Banner: drop keeps
`0.975 … 0.825` (0.2·i/16 at the nine skip sites), classifier dropout keep 0.8. The resume restored
θ/m/v/EMA and the BN running stats + `ema_bn` from the `.bn` companion (hash matches leg 1's).
Accuracy is chance, as it should be at 40 steps.

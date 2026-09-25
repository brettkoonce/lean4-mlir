# MNv2 TF-slim recipe smoke — 2026-09-25, ares (4x 4060 Ti)

`mnv2-default-4gpu`'s variant `rmsdp64wxdols0eps0001bf16` (BN ε 1e-3 in the render, decay 0.997,
staircase ×0.98/epoch from step 0, no warmup), 40 steps/epoch capped (`LEAN_MLIR_G2_STEPS=40`),
checkpoint tag `smoke` (deleted after). `smoke.sh` reruns it.

| leg | epochs | lr printed | eval graph | top-1 / top-5 | exit |
|---|---|---|---|---|---|
| 1 | 1 | 0.045000 | `@mobilenetv2in_fwd_eval_eps0001` | 0.098% / 0.506% | 0 |
| 2 (resume at epoch 1) | 2 | 0.044100 | same | 0.100% / 0.520% | 0 |

The printed lr is the epoch's last step. A continuous schedule would print 0.04411 at both
(0.045·0.98^(39/40) and ^(79/40)); 0.045000 then 0.044100 is the staircase with no warmup.
Accuracy is chance, as it should be at 40 steps. Banner: `decay 0.997000 … new-batch weight
0.003000`; the resume restored θ/m/v and the BN running stats + `ema_bn` from the `.bn` companion.

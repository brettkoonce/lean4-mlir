# ConvNeXt-T / ImageNet-1k — 300 epochs, 4× RTX 3060, bf16

Final, scored over **all 50,000** validation images:

| arm | top-1 | top-5 |
|---|---|---|
| **EMA (reported)** | **81.53%** | **95.50%** |
| raw weights | 81.51% | 95.50% |

Weight averaging is worth **+0.03 points** — the cosine anneals the LR to zero over the
final epochs, so the raw weights have stopped moving and the average has nothing to remove.
(EfficientNet-B0's EMA was worth +0.82; do not carry that framing into this chapter.)

## Run

- Artifact `jax/.lake/build/generated_convnext_tiny_imagenet_full.py`, params **28,589,128**
  (`GAP → LN(768) → 768→1000`, `cnxInit` on). `regen_jax_generated.sh box` green at launch.
- 300 ep · batch 256 (4×64) · lr 2.5e-4 · 20-ep warmup · AdamW + cosine · RandAugment/mixup/
  cutmix/random-erasing · drop-path 0.1 · EMA 0.9999.
- **76.49 h** (275,347.6 s) — one process, **zero restarts, zero thermal rests**. Cards held
  55–67 °C against an 80 °C trip.
- Steady state **180.0 ms/step**, **917.5 s/epoch** (898 s train + 17.7 s val + ~1.2 s ckpt).

## Notes

- The in-training per-epoch eval already covers all 50,000 (`drop_remainder=training`), so the
  rescore confirms the EMA figure rather than correcting it.
- Two eval paths over bit-identical weights differ by 5 images / 50,000 (0.01%) from sharding
  reduction order. The canonical number is the `.bin` path via `eval_convnext_full50k.py`.
- Raw arm needs `eval_convnext_arms_full50k.py`: the `.bin` **is** `ema_params`, so the raw
  weights exist only inside `<base>.state.npz`.

Supervisor: `jax/scripts/supervise_convnext_t_300ep_3060.sh`.

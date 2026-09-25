# MNv4-Conv-M ms/step after the timm-parity change (90e4af7e), 4× 4060 Ti, 4×64 = global 256

Measured with `scripts/bf16_probe_3060.sh`, clock over steps 200–600, `PJRT_FFI_RESIDENT=1`.
The old net is 17fb0c26's renders swapped into `verified_mlir/` for the synth rows and restored
afterwards (`run2.sh`). It uses the same binary, since the parameter shapes are identical.

| net | arm | workers | f32 med / mean | bf16 med / mean |
|---|---|---|---|---|
| timm (new) | synth (compute floor) | — | 167 / 167 | 97 / 97 |
| pre-timm (old) | synth (compute floor) | — | 146 / 145 | 88 / 87 |
| timm (new) | fed | 4 (the job's) | **198 / 198** | **136 / 136** |
| timm (new) | fed | 8 | 233 / 234 | 168 / 280 |

- The compute floor is +14% at f32 and +10% at bf16. The multiply-add count moves only 1%, so
  the change comes from memory traffic: rows 1, 3 and 11 now expand at 2h.
- Feed wait on top of the floor is 31 ms at f32 and 39 ms at bf16. 8 workers is worse than 4,
  as the conf says.
- 5,004 steps/epoch gives:
  - f32: 16.5 min/epoch, 100 ep ≈ 27.5 h;
  - bf16: 11.3 min/epoch, 100 ep ≈ 18.9 h.
- Both exclude the roughly 45 s eval per epoch.
- `probe.tsv` is the first pass. Its fed rows ran at the script's default 8 workers, because the
  default row table lacks the conf's `SHIM_WORKERS=4`.

## 4 × 128 bf16 (the 50-epoch pair's geometry; `run3_b128.sh`, `run4_b128_workers.sh`)

These use scratch renders at B = 128, swapped into `verified_mlir/` and restored. The optimizer is
the committed AdamW with no accumulation; the recipe variant adds one buffer and one op per
parameter.

| arm | workers | med / mean ms per micro-batch (512 images) |
|---|---|---|
| synth | — | 162 / 161 |
| fed | 4 | **260 / 362** |
| fed | 6 | 282 / 523 |
| fed | 8 | 320 / 678 |

- It fits in the default memory pool.
- Per image, the compute floor is better than 4 × 64: 162 ms per 512 images against 97 per 256.
- The fed step is feed-bound: 4 workers is best, and more workers make it worse.
- At 2,496 micro-batches per epoch that is 11–15 min per epoch.

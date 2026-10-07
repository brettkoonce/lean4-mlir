# 2026-10-07 — the D17 side-quest ETA re-probe (ares, 4× 4060 Ti)

`planning/imagenet_parity.md` §9.2 D17: every rate in `planning/side_quest_runs.md` §1 was taken on
2026-10-05, before X2 (ViT-S/B ε / min-lr / f32 stem+head), X5 (A2/A1 bicubic RandAugment geometry),
X6 (MNv4 `full` RandAugment p 0.7), torchvision's crop (aa974a95), the exact GELU (`geluExact`, the
`…erf…` renders) and the uint8 wire. Same method as `runs/2026-10-04-dimm-fan-thermal/` and the 3060
box's `runs/2026-10-06-sidequest-probe-3060/`, so the tables compare row for row; `queue.sh` is the
script (two phases: `jax`, then `verified` after the D5 rebuild), `winrate.py` the JAX window tool.
Tree: 9d7def4d + this session's staged edits (D4 `dwFanK2`, D5 kind 7, the resnet confs).

| run | ms/step (10-05) | bound | hottest DIMM |
|---|---|---|---|
| A2 JAX | 1,440 (1,467) per optimizer step | compute, flat | 50.3 °C |
| A1 JAX | 1,441 (1,469) | compute, flat | 51.3 |
| ViT-S JAX | 341 (344) | tf.data, flat 338–351 | 52.0 |
| ViT-B JAX | 739 (708) | compute, flat | 46.0 |
| ConvNeXt-S JAX | 293 (304) | compute, flat | 48.3 |
| ConvNeXt-B JAX | 400 (401) | compute, flat | 46.6 |
| MNv4 `full` JAX | 2,573 (2,543 → 2,303) per optimizer step, flat over 3 windows | compute | 53.0 |
| ViT-S verified | mean 339 / med 332 / min 314 (379 / 381 / 302) | shim: 18 ms starved (was 79) | 53.9 |
| ViT-B verified | 876 / 876 / 863 (887 / 884 / 839) | compute | 47.8 |
| A2 verified | 476 / 474 / 432 per micro-batch (446 / 446 / 415) | shim: 42 ms starved (X5's bicubic geometry) | 52.3 |
| A1 verified | 463 / 460 / 425 (444 / 444 / 412) | shim: 35 ms starved | 52.0 |
| ConvNeXt-S verified | 318 / 318 / 315 (373 / 372 / 355) | compute | 50.1 |
| ConvNeXt-B verified | **652 / 646 / 636** (518 / 518 / 507) | compute — see below | 46.6 |
| MNv4 verified | 274 / 275 / 240 (290 / 290 / 233) | shim CPU: 35 ms starved (was 57) | 52.1 |

JAX = `winrate.py` steady median (skip 2 windows); verified = `PROBE-SPREAD` mean / median / min over
optimizer steps 201–600, under `LEAN_MLIR_CKPT_TAG=probe` (no lineage touched; the 10-05 MNv4 smoke's
600 optimizer steps crossed its 312-step epoch and left `.lake/build/mnv4in_…_ckpt_xla.bin` at epoch
1, now in `.lake/build/stale/`). No DIMM above 54 °C, no window stalled.

## The ConvNeXt-B exact-GELU render is 33% slower; ConvNeXt-S's is not

Same box, minutes apart, the conf's own env, 600 steps each (`v_cnx{b,s}-tanh-control.log`):

| render | ConvNeXt-B | ConvNeXt-S |
|---|---|---|
| `emadpwxclipdropbf16` (tanh GELU, the 10-05 variant) | 490 ms/step | 320 |
| `emadpwxclipdroperfbf16` (exact GELU, the conf's variant since §3.8) | 652 | 318 |

So today's box is ~5% faster than 10-05 on the tanh renders (490 vs 518, 320 vs 373), the exact GELU
costs ConvNeXt-S nothing, and it costs ConvNeXt-B +162 ms/step = **+68 h** on its 300 epochs (272 h
against 204 at today's tanh rate). ViT-B's exact render shows no such cost (876 vs 887 on 10-05), and
the JAX side's `approximate=False` costs ConvNeXt-B nothing (400 vs 401), so it is not the erfc
arithmetic as such. The GELU runs in f32 between bf16 1×1 convs on `64×512×56×56` (105 M-element,
420 MB) tensors in B's stage 1; B's tanh render already peaked at 9.53 of the 11.68 GiB arena, so the
leading guess is memory pressure (XLA rematerialising or de-fusing around erfc's extra f32
intermediates), which S at 384 wide and 6.80 GiB does not reach. `v_cnxb-erf-mem097.log` is the test
of that guess: the same erf render with `LEAN_MLIR_MEM_FRACTION=0.97` (a 15.1 GiB arena).

**Result (`v_cnxb-erf-mem097.log`, `v_cnxb-erf-mem090-eval.log`, `v_cnxb-erf-mem090.log`):** the erf
render runs **490** at 0.97 and **482** at 0.90 — the tanh rate — so it was the arena. 0.97 is what
OOMed ConvNeXt's bf16 arms in August (`cnxb-default-4gpu.conf`'s header: `CUDA_ERROR_OUT_OF_MEMORY
in d2h(res)`, a 97% pool starving what lives outside it), so the conf takes **0.90** (14.0 GiB, 1.6
GiB left outside), and the 0.90 smoke under `LEAN_MLIR_G2_STEPS=300` ran five epoch boundaries
(checkpoint + d2h each) and the epoch-5 streamed eval over all 50,000 without an OOM.
`cnxb-default-emabf16-4gpu` sets and prechecks it; ConvNeXt-B verified is ~201 h, not 272.

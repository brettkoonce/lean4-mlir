# 2026-10-04/05 — DIMM-fan re-test and the side-quest ETA re-probe

The DIMMs re-slotted per the board manual (sensors now i2c 1-0018/19/1c/1d), two fans on them.
Results and the schedule they feed: `planning/side_quest_runs.md` §1, §2, §5.2.

| script | what |
|---|---|
| `queue.sh` | ViT-S JAX 15 min (the §5.1 thermal smoke), then R50 A2 JAX |
| `queue2.sh` | A1 / ViT-B / ConvNeXt-S / ConvNeXt-B JAX, 12 min each; relink the verified trainers (its verified loop was stopped and moved to queue4) |
| `queue4.sh` | the six verified §3a smokes (`ONCE=1 LEAN_MLIR_MAX_STEPS=600 LEAN_MLIR_PROBE_WARM=200`) |
| `queue5.sh` | MNv4 `full`, JAX 20 min and verified |
| `queue6.sh` | re-render R50 with the drop-mask passthrough, gate, relink, re-probe A2 and A1 verified |
| `winrate.py` | 100-step window ms/step from the wall-stamped trainer lines |

Every run has `temps_<run>.tsv` (four DIMMs, Tctl, GPU temps and utilisation every 2 s) beside its log.

| run | ms/step | hottest DIMM |
|---|---|---|
| ViT-S JAX | 344 flat | 56.8 °C |
| A2 / A1 JAX | 1,467 / 1,469 per optimizer step | 50.6 / 50.8 |
| ViT-B JAX | 708 | 52.3 |
| ConvNeXt-S / -B JAX | 304 / 401 | 54.1 / 52.4 |
| MNv4 JAX | 2,543 → 2,386 → 2,303 per optimizer step | 57.9 |
| ViT-S verified | mean 609, median 595, floor 290 | 55.4 |
| ViT-B verified | 887 / 884 / 839 | 51.0 |
| A2 / A1 verified | 446 / 444 per micro-batch | 46.2 / 46.2 |
| ConvNeXt-S / -B verified | 373 / 518 | 53.9 / 52.3 |
| MNv4 verified | 329 / 323 / 198 per micro-batch | 50.8 |

Failed logs kept beside the reruns:
- `*.precheck-fail.log`: ViT-S verified refused. Two ViT-Ti files in `jax/.lake/build` were stale
  (`scripts/regen_jax_generated.sh sync`).
- `*.g4-fail.log`: A2/A1 verified before the fix. The PJRT shim refused 755 outputs against 771
  destinations, because the R50 drop renders did not return their masks.

`r50_render_diff.txt` / `r50_arity.txt` record the re-render: 12 files, two lines each, 12/12 pass.
`full_build.log` is `lake build` + `lake build Certs` + `regen_verified_mlir.sh check` on the fix.

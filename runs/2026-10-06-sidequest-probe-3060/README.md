# 2026-10-06 — the side-quest ETA probe on the 3060 box (mars)

The same 14 runs as ares' 2026-10-05 probe (`runs/2026-10-04-dimm-fan-thermal/`, the source of
`planning/side_quest_runs.md` §1), measured the same way on mars: 4× RTX 3060 12 GB, CUDA 13.3,
`/home/skoonce/.venv-cuda`, tree at `fc320752`.
* **JAX:** the trainer run directly, a 12-minute window (MNv4 20), with ms/step from 100-step windows
  (`runs/2026-10-04-dimm-fan-thermal/winrate.py`).
* **Verified:** the §3a smoke (`ONCE=1 LEAN_MLIR_MAX_STEPS=600 LEAN_MLIR_PROBE_WARM=200
  scripts/supervise.sh <job>`), with the median over steps 201–600.

⚠ Measured on `fc320752`, BEFORE `aa974a95` (torchvision's RandomResizedCrop on every net here) and the
exact GELU on ViT / ConvNeXt (`0a474ae8`, `2855e413`, `a7a9320c`): the feed-bound rows (ViT-S, MNv4,
the R50 verified smokes) and the ViT / ConvNeXt compute can move. Re-probe before trusting a row.

Hours are the full schedule (steps/epoch × epochs × ms/step), with no per-epoch eval.

Before the probe:
* `scripts/regen_jax_generated.sh sync`: 72 copies in `jax/.lake/build` predated the 10-05 commits.
* The six verified exes were rebuilt, and `ffi/libpjrt_ffi.so` was rebuilt from the current `pjrt_ffi.c`.
* All 14 confs passed `DRY_RUN=1` with their PRECHECK.

| job | JAX ms/step | **JAX h, mars** | JAX h, ares | verified ms/step | **verified h, mars** | verified h, ares |
|---|---|---|---|---|---|---|
| R50 A2 (300 ep) | 1,487 / opt step | 77 | 76 | 458 / micro (34 starved) | 95 | 93 |
| R50 A1 (600 ep) | 1,473 / opt step | 153 | 153 | 457 / micro (35 starved) | 190 | 185 |
| ViT-S (300 ep) | 321 | **67** | 72 | 399 (53 starved) | 83 | **79** |
| ViT-B (300 ep) | 905 | 189 | **148** | 977 (8) | 204 | **185** |
| ConvNeXt-S (300 ep) | 305 | 127 | 127 | 349 (4) | **146** | 156 |
| ConvNeXt-B (300 ep) | 468 ¹ | 195 | **167** | 529 (7) ² | 221 | **216** |
| MNv4 `full` (500 ep) | 2,099 / opt step, flat | **~91** | ~100–105 | 231 / micro (51 starved) | **80** | 100 |

"Starved" is the smoke's median − min, i.e. the time spent waiting on the shim.

**Memory on 12 GB:**
* ¹ ConvNeXt-B JAX OOMs at JAX's default 0.75 preallocation. It asks for one 5.98 GiB block
  (`cnxb.log`). It runs at `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` (`cnxb_f95.log`, `queue2.sh`).
* ² ConvNeXt-B verified OOMs at the default arena (a 5.87 GiB d2h, `v_cnxb-default-emabf16-4gpu.log`).
  (That log's BFC chunk-map dump, the `InUse/Free at …` lines, is replaced by a count to fit the repo-shape cap.)
  It runs at `LEAN_MLIR_MEM_FRACTION=0.85`, which was the first fraction tried (`v_cnxb_f85.log`,
  `queue3.sh`). The conf's PRECHECK forbids the knob, so `queue3.sh` ran the trainer with the conf's
  own ENV_EXTRA.
* ViT-B verified fits as committed: the conf's 0.97 reserves 11.85 GB, and it ran 600 steps clean.

**Where each box wins:**
* **mars:**
  * MNv4 on both paths (verified 231 vs 290 ms)
  * ConvNeXt-S verified (349 vs 373)
  * ViT-S JAX (321 vs 344). Here the GPUs are the limit; ares was tf.data-bound.
* **ares:** the heavy compute-bound jobs, i.e. ViT-B (1.28× JAX / 1.10× verified) and ConvNeXt-B
  (1.17× / 1.02×).
* **Within ~3 %:** R50 A2/A1 on both paths, and ConvNeXt-S JAX.

Files:
* `queue.sh` / `queue2.sh` / `queue3.sh` and their `.out` files
* `<run>.log` (JAX, wall-stamped) and `v_<job>.log` (verified)
* `gpu_<run>.tsv`: per-GPU temperature, utilisation, power, SM clock and memory, plus host load, every 2 s.
  There is no `sensors` output on this box.
* `dry_<job>.log` and `lake_build.log`

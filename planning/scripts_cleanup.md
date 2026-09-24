# scripts_cleanup.md — `scripts/` audit and cleanup

Started 2026-09-24 from a five-way read-only audit of `scripts/` (194 files: 149 Python, 45 shell,
~33k lines), the proof tree having had three passes and the scripts none. Rubric: dead, broken,
vacuous gate, dangerous, environment-coupled, duplicated, unguarded generator, stale docs. Every
finding below was read at file:line; "unverified" marks what the audit could not confirm.
Reference counts come from one `git grep` per basename (runs/ excluded).

## 0. Rules for this thread

* Delete vs archive: a script that produced a committed result or is cited as provenance from Lean
  or a demo goes to `historical/` (with its citations repointed); a one-off probe of a finished,
  void or retired-hardware investigation is deleted (git keeps it).
* A gate fix lands with its control: show the gate failing on a broken input before trusting it.
* Generators: confirm byte-reproducible before editing, as in `proof_cleanup.md` §0.
* Nothing here runs GPU jobs; fixes to launchers are checked with `bash -n` + a `DRY_RUN`/`plan` pass.

## 1. Fix first — dangerous or silently wrong

| # | where | problem | fix |
|---|---|---|---|
| 1 | `jobs/supervise.sh:107` | `RUNDIR` defaults to `/tmp/supervise_$JOB`; `lake run <job>` passes none, so every lake-launched master log lands in /tmp (R34's was lost that way, `cnx-default-4gpu.conf:7`) | default `runs/$(date +%F)-$JOB` |
| 2 | `residency_gate.sh:102` | `rm -f "$CKPT" "$CKPT.epoch"` on the trainer's shared `.lake/build/*_ckpt_xla.bin` (voided a gate once) | per-arm `LEAN_MLIR_CKPT_TAG=gate-$tag`, no rm |
| 3 | `jax_probe_4gpu.sh:32` | `kill -9` every GPU compute process on the box | delete (dead probe) |
| 4 | `convention_audit.py:486-529` + `convention_baseline.txt` | baseline still lists `mnv2:bn-split`, `r34:bn-split` (resolved) — a regression would read as "known"; resolved rows don't fail | drop the two rows; `--baseline` fails when a baselined row is gone |
| 5 | `validate_linear_faithful.sh:22,25` | re-elaborates `StableHLOPretty`, whose `linear_*` writers moved to `ChapterArtifacts` (fda72f61): the diff can never fail | delete (proofs.yml:133 covers the drift) or repoint |
| 6 | `gen_mlir_manifest.py --check` | fails now (ChapterArtifacts move + run counts); runs column counts local untracked `runs/**/*.log` (machine-dependent); comment filter at `:186` tests grep's `path:N:` prefix, never fires | regenerate; count `git ls-files runs/` or drop the column; fix the filter; then add to CI |
| 7 | `lipschitz_cert_scorecard_full.py` | no `__main__` guard — importing it (pair_sdp_full, scorecard_ibp) retrains and rewrites 3+ Lean files; `trained_cnn_seal.py:30` `exec`s the witness generator | `main()` guards; import helpers, not modules with side effects |
| 8 | `lipschitz_cert_scorecard_full.py:139,160`, `…_ibp.py:102` | trained weights / PGD counts cached in /tmp with no seed or code hash — a stale cache feeds committed Lean | key on a hash of the script, or `--no-cache` default |

### §1 status (2026-09-24)

1. ✅ `supervise.sh`: default RUNDIR is the newest `runs/*-<job>/` whose master log records a launch,
   else `runs/<today>-<job>`; a `DRY_RUN` logs to /tmp (a plan leaves no run dir). `selftest.conf`
   keeps /tmp and clears its fake epoch in PRECHECK (a second run used to report COMPLETE at once).
2. ✅ `residency_gate.sh` and `eval_residency_gate.sh`: every arm runs under its own
   `LEAN_MLIR_CKPT_TAG` (unique per invocation), cleared before and after; nothing shared is deleted
   (`GATE_CKPTS` is gone). Also: default exe `resnet34-verified-adam`, dead `mnist-cnn-verified-xla`
   → `mnist-cnn-verified` with a `lint-targets:` marker (control: the lint catches the old name),
   a missing binary FAILS instead of SKIP.
3. ✅ `jax_probe_4gpu.sh` deleted.
4. ✅ `convention_audit`: baseline emptied; a resolved-but-baselined row now fails the ratchet
   (control: the old baseline exits 2; the empty one passes with the selftest).
5. ✅ `validate_linear_faithful.sh` repointed, not deleted — its (b) half (iree-compile validity) is
   not in CI: drift from `ChapterArtifacts`, defaults `cuda`/`sm_86`; passes end to end here.
6. ✅ `gen_mlir_manifest.py`: writers via `git grep` with the prefix stripped before the comment test;
   run counts from TRACKED logs only; regenerated; `--check` added to proofs.yml (+ path filter).
   ⚠ A committed run log that compiles an artifact now needs a manifest regen before the next Lean push.
7. ✅ `lipschitz_cert_scorecard_full.py`: the three file writes are `write_files()` under `__main__`;
   `need` moved above them (the importers read it). Import writes nothing (mtimes checked); `__main__`
   is byte-identical. `trained_cnn_witness.py` skips its write when `trained_cnn_seal.py` execs it
   (`WITNESS_NO_WRITE`); both regenerate byte-identically.
8. ✅ /tmp caches keyed on `cache_key(source of the producing function, hyperparameters[, weights])`.

## 2. Vacuous or weak gates

| where | passes when | fix |
|---|---|---|
| `verify_excerpt.py:46-59` | 0 lines matched → "all verified" | fail on `checked == 0` |
| `batch_divisor_gate.py:88-107, 60-61` | empty glob → exit 0; K ∉ {10, 1000} silently skipped | fail on empty; report skipped |
| `eval_residency_gate.sh:46,62` | dead exe name → SKIP, exit 0 | fix the name, fail on SKIP (as `residency_gate_all.sh` does) |
| `check_pinned_env.py:63-68,105` | JAX falls back to CPU → conv check skipped, success printed | require `platform == "gpu"` unless `--allow-cpu` |
| `bf16_gate2.py:28` | verdict from whatever backend compiled | assert GPU platform |
| `mixup_gate.py:236` | empty cutmix box satisfies every check | require `M.sum() > 0` somewhere |
| `check_fpn_affine.py:171-180` | property 3 (pixels) printed, never enforced | tolerance + `FAILED`; drop unused `nga, nra` (`:165`) |
| `convention_audit.py:252-256, 369` | BN world decided by substring `dimensions = [0, 2, 3]` anywhere (a conv-bias reduce matches) — latent | count BN reduce sites |
| `jobs/selftest.conf:9-14` | stale `epoch` file → second run reports COMPLETE, tests nothing | PRECHECK clears it |

Latent bugs: `check_fpn_affine.py:73` buffer `12348 * 20` for a 40-byte C struct (overflows past
~6k boxes); `check_audit_coverage.py:34-39` parses past `roots := #[...]` into the next lib's
docstring (no false coverage today).

## 3. Broken on this box

| where | problem | fix |
|---|---|---|
| `residency_gate.sh:55` | default exe `resnet34-verified-adam-xla` renamed 2026-08-10 | `resnet34-verified-adam` |
| `grad_tie.py:51-53`, `mnv4_forward_tie.py:51-55`, `enet_forward_tie.py:34-36`, `convnext_forward_tie.py:59,61`, `mnv2_forward_tie.py:52,54`, `xla_pad_op_check.py:28,30`, `render_parity.py:23,46`, `seg_grad_scorecard.py:36-37`, 5 `*_probe_check.py` (C1) | IREE binaries at missing `.venv/bin/iree-*` or sibling checkouts; `IREE_CHIP=gfx1100` (AMD); some not env-overridable | one `scripts/_iree.py` resolver (env override, working default, CUDA/CPU target) |
| `grad_fd_bisect.py:24,29-30` (backs lake `grad-fd-probe`) | scratch dir from another checkout's session; ROCm/HIP env | tempfile + CUDA env |
| `coco_anchors.py:33` | imports `preprocess_coco` (now `historical/`) | move to `historical/` beside its caller |
| `arasl_score.py:110` | nested same-quote f-string: SyntaxError on system 3.10; `demos/README.md:220` says `python3` | double quotes inside, or README → `.venv/bin/python` |
| `margin_probe.py:32`, `mnist_e4m3_demo.py:30`, `mnist_e4m3_train_demo.py:32` | default DATA in a sibling checkout (live: cited from Lean) | default `data` |
| `bf16_probe_3060.sh:38-39`, `run_r34_ablation.sh:80` | default venv `/home/skoonce/.venv-cuda` (absent) | box-detect as the confs do |

### §2 + §3 status (2026-09-24)

§2, each with its control:
* `verify_excerpt.py`: a range with no log lines exits 1 (control: `content.tex:1-5`).
* `batch_divisor_gate.py`: anchored on ROOT; no artifacts or all skipped → 1; skipped ones are
  listed (none today, 193/193 checked; `--control` still fires; an empty root exits 1).
* `eval_residency_gate.sh`: done in §1.
* `check_pinned_env.py`: a non-GPU default device fails unless `--allow-cpu` (control:
  `JAX_PLATFORMS=cpu` exits 1; on the GPU it passes with cuDNN loaded and the bf16 conv run).
* `bf16_gate2.py`: refuses a non-`gpu` backend; the `--against` option, which only printed
  "speedup not measured", is gone.
* `mixup_gate.py`: gate 3 requires a non-empty cutmix box in at least one batch; docstring
  `--break` → rc 0 (controls pass by being rejected). The stream reader was still on wire v2
  and died at the preamble; now v4 (per-record row count). Gates 1b/2/3 and the controls pass.
  ⚠ Finding, not fixed: gate 1 FAILS — the pinned mixing-off digests (`:163-164`) no longer
  match (v1 now 8a204cd5…, v2 ad5f9e0b…). It failed before this change too. Several shim commits
  could have moved them (wire v3/v4, 8182b6e1; per-net shims fd2d9fbe; 4a0a2781) — find which
  one before re-pinning; a re-pin without a cause approves the change blindly.
* `check_fpn_affine.py`: property 3 enforced (mean ≤ 0.1, correlation ≥ 0.99; measured 0.0529 /
  0.9954); `--break` flips the inverse translation's sign and fails (0.87 / 0.14). Box buffer is
  a ctypes `fpn_aff_box` array of NTOT/15 entries; dead `nga, nra` out.
* `convention_audit.py`: `bn_world()` counts `[0,2,3]` vs `[2,3]` reduces, `none` without an
  `rsqrt`. The substring rule was not only latent: it called all 13 `cifar8*_bn_*` train steps
  (per-example, 24 vs 64) and the BN-free CIFAR/MNIST steps "batch". Audited nets unchanged.
* `jobs/selftest.conf`: done in §1.
* `check_audit_coverage.py`: roots read from the `roots := #[...]` array only (the old scan took
  2–4 extra names per lib from comments/docstrings); one `lib_roots` instead of two parsers
  (control: dropping `StableHLOParse` from the roots exits 1).

§3:
* `scripts/_iree.py`: `$IREE_COMPILE`/`$IREE_RUN_MODULE`, else the interpreter's bin, repo
  `.venv/bin`, PATH, sibling `lean4-jax/.venv/bin` (compiler and runtime from one install);
  backends `llvm-cpu` / `cuda` (`$IREE_CHIP`, default sm_86); local-task → local-sync fallback;
  signals named. On it: the 7 `*_probe_check.py` (output byte-identical to the pre-change run),
  `grad_tie`, the mnv2/mnv4/enet/convnext forward ties, `xla_pad_op_check` (numbers identical),
  `render_parity` (now `--backend`, default cuda; self-parity 38/38 bit-identical on
  `cifar8_bn_train_step`, eps 1e-5→1e-3 control 21/38 over 1e-3, rc 1), `seg_grad_scorecard`
  (reproduces the `SpecHelpers.lean` row). ROCm/`gfx1100` is gone from all of them.
* After the move, with no IREE env set: every tie reproduces its pre-change numbers
  (mnv4 / convnext / grad_tie mnv4 pass; enet — never runnable here before — passes);
  `convention_audit` clean on all five nets and `--selftest` passes.
* ⚠ Finding, not fixed: `mnv2_forward_tie` fails its own 1e-4 bound at 6.2e-4 (before and after
  the refactor). `--diag`: only the as-is/batch-BN row is small (every other ≈0.3), so the
  structure ties; the bound is likely stale against the ~1e-3 MNv2 fp32 split (sync-BN §3.3).
  Decide: widen the bound with a measured justification, or chase the 6e-4.
* `grad_fd_bisect.py`: tempfile scratch, `CUDA_VISIBLE_DEVICES` default 0, data `data` (the
  idx files live there). Not run.
* `coco_anchors.py` → `historical/` (download_coco.sh + neu_anchors.py repointed). Its
  `from scripts.visdrone_anchors` import — and `visdrone_fpn_coverage.py`'s — lose to anaconda's
  site-packages `scripts` package; both now import the sibling module directly.
* `arasl_score.py`: the nested f-string hoisted to a variable; compiles on 3.10.
* `margin_probe.py`, `mnist_e4m3_{demo,train_demo}.py`: default DATA is ROOT/data (standard
  MNIST, md5-checked).
* `bf16_probe_3060.sh`, `run_r34_ablation.sh`: the confs' box detect (dry run finds the plugin).
* Still stale: `tests/TestConvNeXt{T,}TrainPC.lean` docstrings describe the old ROCm/PATH setup
  for `render_parity.py` (§7).

## 4. Dead — delete or move to `historical/`

~58 scripts. Delete unless marked (H) = move to `historical/` (cited as provenance or produced a
committed result; repoint the citation).

* **ROCm / AMD-era (retired hardware):** `miopen_conv_probe.py`, `miopen_im2col_repro.py`,
  `miopen_mem_probe.py` (byte-identical copies already in `upstream-issues/`), `jax_multigpu_probe.py`,
  `jax_multigpu_smoke.py`, `jax_r34_imagenet_bench.py` (H, next to its upstream-issues README),
  `log_gpu_temps.py` (reword `log_gpu_temps_cuda.py:4`), `kernel_faithfulness_probe.py` (H),
  `transcendental_probe.py` (H), `jax_imagenet_bench.py` (H), `eval_curve_2gpu.sh`,
  `eval_when_done.sh`, `run_cnn_grid_sweep.sh`, `run_mlp_grid_sweep.sh`, `run_smooth.sh`,
  `run_smooth_2gpu.sh`, `run_smooth_dump_2gpu.sh`, `run_convnext_diag.sh`,
  `run_convnext_smooth_2gpu.sh`, `run_vit_muon_ab.sh`, `run_yolo_mixed.sh`,
  `run_vit_deit_300ep_paced.sh`, `run_smooth_scorecard.sh` (H: cited by `SmoothingCPScorecard`),
  `jax_probe_4gpu.sh`.
* **BraTS (demo VOID):** `brats224_crop_guard.py`, `brats_overfit_subset.py`,
  `brats_oversample_probe.py`, `brats_r34_ab.py`, `brats_r34_fd_probe.py`, `seg_region_dice_check.py`,
  `run_brats_ablation.sh`, `run_brats_r34_ab.sh` (H: `demos/README.md` cites it).
  `brats_class_weights.py` follows the demo code's fate (cited by `MainUnetBratsTrain.lean`).
* **VisDrone/FPN investigation (finished, archived plans only):** `fpn_affine_knob_cost.py`,
  `fpn_duplicate_oracle.py`, `fpn_iou_target_probe.py`, `fpn_neighbor_align_check.py`,
  `fpn_neighbor_separation.py`, `fpn_objectness_readout_probe.py` (repoint the two
  `yolo_map_visdrone.py` comments), `fpn_prior_bias_check.py`, `fpn_resolution_probe.py`,
  `fpn_ring_boxes.py`, `run_fpn_pb_eval_watch.sh`, `run_fpn_t2a_eval_watch.sh` (replaced by
  `neudet_eval_sweep.sh`).
* **Pets demo (archived):** `yolo_map.py` (H), `yolo_render.py` (H).
* **Pre-result mocks:** `mock_nqs_figure.py`, `mock_boltzmann_figure.py`, `mock_gw_figure.py`
  (soften `gw_figure.py:3` + two planning mentions).
* **Superseded / finished:** `bf16_probe_4gpu.sh` (its header: "SUPERSEDED — DO NOT USE"; repoint
  `probe_to_eta.py` + 2 confs to `bf16_probe_3060.sh`), `queue_r50_a3_pair.sh` (A3 pair finished
  2026-08; drop the conf comments), `bf16_boundary_probe.py` (H), `geo_aug_pil_diff.py` (H),
  `lipschitz_cert_power_iter.py` (H), `lipschitz_cert_rationalize.py` (H: fix
  `TrainedMlpWitness.lean:26-27`, weights are hand-maintained in `LipschitzCertInstance`),
  `emit_smoothing_tikz.py` (H), `emit_lipschitz_tikz.py` (H).
* **Decide:** `convnext_forward_tie.py` (working gate, zero refs — wire in next to its
  mnv2/mnv4/enet siblings, or delete), `gw_compare.py` (zero refs — cite from the GW plan or delete),
  `check_pinned_env.py` (zero refs — fix §2 and wire into jax.yml, or delete),
  `rope_test.sh` / `run_tinystories_8k.sh` (IREE-era LM runs — fold into
  `lm_demos_modernization.md`).

## 5. Unguarded generators

~30 committed Lean files have a generator that no CI job re-runs: the Lipschitz / IBP / CROWN /
smoothing scorecards, `Training/Trained{MlpWitness,CnnWitness,CnnSeal,LinearDescent}`,
`ffi/pjrt_compile_options.h`, `blueprint/src/figures/depgraph/*`. Cheap first guards (inputs are
committed files only, no `data/`): `lipschitz_cert_float.py`, `smooth_scorecard_gen.py`,
`smooth_dec_scorecard_gen.py` — add `--check`, confirm byte-reproducible (unverified), wire into CI.
The data-dependent ones (MNIST retraining) stay manual; record that in their headers.
Also: `smooth_scorecard_gen.py:33`, `smooth_dec_scorecard_gen.py:47-48`,
`smoothing_net_witness_gen.py:30-32`, both `emit_*_tikz.py` use cwd-relative paths — anchor on `ROOT`.

## 6. Duplication → shared modules

| module | replaces |
|---|---|
| `scripts/_iree.py` | the IREE compile/run harness in 7 A-group checkers + `make_runner` ×5 (C1) + `run_iree` ×2 — and fixes §3's defaults in one place |
| `scripts/_leanlit.py` | Lean-literal printers (`frac`/`row`/`zlist`) ×7 |
| `scripts/_mnist_io.py` | MNIST idx readers ×12 (9 generators, 3 probes), the seed-0 49→8→10 training loop ×3, e4m3 quantizers ×2 |
| `scripts/_stats.py` | `wilson` ×3, `acc_str` ×2, `energy_distance` ×3 (`mnist_ddpm_score.py` claims to reuse it) |
| `scripts/lean_graph.py` | lakefile-roots parsing + import BFS in `check_audit_coverage` (×2), `audit_census/run.sh` heredoc |
| `scripts/jobs/_box.sh` | the box-detect block in 18 of 19 confs + 2 gates' plugin search |
| `scripts/lib/gpu.sh` | det-shim build-or-reuse ×4 (3 different dirs), idle-GPU check ×3, work-queue packing ×3 |
| (import) | FPN neck oracle ×2 (`fpn_neck_probe_check` ← `fpn_neck_check`), anchor loaders ×6 (← `yolo_map_visdrone`/`visdrone_anchors`), CROWN/IBP helpers ×2, `SHIM_HASH` parse ×2, `read_part` ×2 |

## 7. Docs

`bf16_probe_3060.sh:2,8-10` (box is 4× 4060 Ti; pins 0,1,2,3); `streamed_val_gate.sh:7` (goldens need a pre-8182b6e1 build); `yolo_map_visdrone.py:13` ("a copy
of yolo_map.py with two changes" — now ~460 diff lines); `toy2d_metrics.py:57` (moved path);
`fpn_loss_probe_check.py:17`, `render_parity.py:10-15`, `gen_pjrt_compile_options.py:39`,
`jax_imagenet_bench.py:9-13` (ROCm/IREE-era); `jax.yml:243` ("5 nets" → 8 + 4 variants);
`shim_wiring_gate.py:16,351`; `mixup_gate.py:32` (`--break` expects rc=0); `convention_audit.py:146-149,525`;
`bf16_gate2.py:5` (no speedup measured); `verify_excerpt.py:4`, `measure_prose.py:2` (plan archived);
`blueprint_depgraph_tikz.py:330` (`--help` creates a dir; argparse).

## 8. Decided

* `probe_to_eta.py`: the user chose the ImageNet default — `--stat med` is now the default and the
  docstring says why; starving rows keep the mean in their annotation ("⚠ today's means …"), and
  `--stat mean` stays for a wall-clock estimate of a known-starving configuration. The stale val-drain
  sentence is out. (Done 2026-09-24.)

## 9. Order

1. §1 (dangerous / silently wrong) — small diffs, each with its control.
2. §2 gates + §3 broken defaults (`_iree.py` first, it fixes ~15 scripts).
3. §4 dead scripts — one commit per cluster.
4. §5 guards for the three cheap generators.
5. §6 shared modules, largest duplication first; §7 docs as each file is touched.

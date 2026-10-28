# CIFAR chapter: re-render the eight-conv nets with batch-statistics BatchNorm

Status (2026-10-19): **built and checked on a CPU box; the GPU runs are the only thing left
before the book text.** Everything in "What changes" below exists, is rendered, compiles for
IREE's CPU targets, runs end to end on synthetic data through the real driver, ties its forwards,
and has its backward gated against an independent JAX autodiff. The plan-only text this note held
at the start of the day is folded into "What changes" with what was actually done. Committed
2026-10-19; see the end for the file list.

## Why

The CIFAR-8 "BatchNorm" renders normalize each example on its own: the statistic
is per image and per channel over the 32×32 cells. The batch axis is never
reduced (`cifar8w_bn_*_train_step.mlir` and `cifar8w_bn_fwd.mlir` reduce over
`dimensions = [2, 3]`), there are no running statistics (`runningBN := false` on
`cifar8wBnVerified`, so `bnChannels = #[]`), and train = eval. The math the
chapter proves (`bnForward` on a `Vec n`) holds for any group, so no theorem is
wrong, but every number in §4's experiments — the +5.8 / +4.2 / +1.5 BN deltas,
the 2.5–13× spread reduction, the bf16 table, the head-width sweep — was
measured on per-image normalization, and the chapter's framing (Ioffe & Szegedy,
batch statistics, running stats threaded through `bnChannels`) describes the
layer Chapter 5 uses. The honest fix landed in the prose; the complete fix is to
train the layer the prose describes.

## What changes — done

1. **Render.** `LeanMlir/Proofs/Codegen/CnnBbnRender.lean`: the eight-conv net on the batched op
   family with every BN site the shared `SyncBnSites` ones (`bnFwdSite` / `bnBackSite` /
   `bnGammaSite` → `bnBatchF` / `bnBatchBack` / `bnGammaGradB`, reduce `[0,2,3]`), `bnBetaGradB`,
   and the running-statistics protocol ResNet-34 uses: 16 passthrough stat inputs after `%bc2`,
   16 batch-statistic outputs (`bnBatchMeanB` / `bnBatchVarB`) after `%bc2`. One forward chain
   (`cifar8BbnFwdChainB`) feeds the train step and `@cifar8wb_bbn_fwd`, so the forward is a byte
   prefix of every f32 train step; `@cifar8wb_bbn_fwd_eval` is the per-example chain with
   `bnEvalSite` (frozen running statistics, 16 stat inputs). Writers in `CnnArtifacts.lean`;
   eight artifacts in `verified_mlir/`: `cifar8wb_bbn_{sgd,mom,adam,bf16sgd,bf16mom,bf16adam}_train_step`,
   `cifar8wb_bbn_fwd`, `cifar8wb_bbn_fwd_eval`. 135 inputs / 133 outputs per step. Every existing
   artifact is byte-identical after the regen.
2. **Spec.** `cifar8wbBbnVerified` (slug `cifar8wb_bbn`, `NetsCore.lean`): `cifar8wBnVerified`
   with `runningBN := true`; `bnChannels = #[16,16,16,16,32,32,32,32]`, guarded. The per-image
   specs and artifacts are untouched.
3. **Driver.** `apps/cifar/MainCifar8WideBbnAblation.lean` → `cifar8wb-bbn-ablation`, one arm per
   invocation (`data <sgd|mom|adam|bf16sgd|bf16mom|bf16adam|all>`), same rates and schedule as
   `cifar8w-bn-ablation`; the generic `trainAdamSched` threads the EMA (`bnMomentum` 0.99), writes
   the `.bn` companion and scores through `_fwd_eval` — no driver logic was needed beyond the
   spec. `scripts/sweeps/seed_sweep.sh` has a `SUITE=bbn`.
4. **Theorem.** `LeanMlir/Proofs/Nets/Small/Cifar8BbnStepTieB.lean`:
   `cifar8Bbn_train_step_tiedB` — all 38 gradient nodes at batch `N` (a binder) and `bf16` (a
   flag), each denoting the certified batched `Σ_n` gradient at the render's own forward values and
   the cotangent its backward chain delivers (`GradNodeB.ConvWTiedBAt` / `ConvBTiedB` /
   `BnPairTiedB` / `DenseWTiedB` / `DenseBTiedB`; BN link `bnInB`), rooted in `Certs`. Three
   standard axioms. **Not yet stated:** that the chain IS the certified whole-net backward
   (ResNet-34's `*CotIn_eq_vjp` half) — needs the batched-BN cifar8 forward's `HasVJPAt` and the
   per-link `_eq_vjp` composition; each link's `den` lemma exists. The budget-table row "BN at
   batch statistics" (ch. 4, `blueprint/src/content.tex:381`) can cite the node tie now and the
   full tie later.

## Checks that ran here (no GPU)

| check | result |
|---|---|
| `scripts/regen_verified_mlir.sh check` | manifest, entry names, both prefix audits (the new forward is paired with the sgd and adam steps), divisor and twin gates all pass |
| `lake env lean tests/TestCifar8AdamTrain.lean` at `IREE_BACKEND=llvm-cpu` | 22 compile OK, the eight new ones included |
| `fwd-tie cifar8wb_bbn` and `--eval` | both self-tie on non-degenerate logits |
| `cifar8wb-bbn-ablation data sgd` on IREE llvm-cpu, synthetic data, 3 steps | compiles all three artifacts, returns and EMAs the 384 stat floats, evaluates through `_fwd_eval`, writes the `.bn` companion |
| `tests/cifar8_bbn_jax_ref.py` (JAX f64 autodiff of the same forward, from a `C8BBN_GC_DUMP`) | loss agrees to 7e-7; gradient agrees in every tensor on a sound lowerer (vmvx: cosine 1.0000 ×38) |
| `cifar8-bbn-gradcheck` at `IREE_BACKEND=vmvx` | see the table below |

### ⚠ IREE 3.11 llvm-cpu miscompiles the pooled train steps — this is not a render bug

The first gradcheck run, on llvm-cpu, reported layers 1–6 wrong (cosine 0.4–0.8 against JAX; 7, 8
and the head exact) for the NEW artifact **and** for the committed per-image
`cifar8wb_bn_adam_train_step.mlir`. Pinned down as the lowerer: the forward matches JAX at every
probed point on either backend; the same text on IREE's reference `vmvx` backend is exact in every
layer for both artifacts; tapping extra outputs out of the llvm-cpu compile changes which layers
come out right; `-O0` / no-fusion / no-vectorisation builds all agree with each other and are all
wrong; and the compile warns `reads uninitialized values from an operand produced by a tensor.empty
op` at every `select_and_scatter` (the max-pool backward), whose lowering's running-max
accumulator is uninitialised. Consequences: the GPU trainers (XLA) are unaffected; `IREE_BACKEND`
now accepts `vmvx` (no chip flag) and the CPU gate runs there; the CI gradchecks/oracle on llvm-cpu
do not gate any pooled net's backward until IREE is fixed (the MNIST cnn oracle passed, presumably
by allocation luck — worth moving to vmvx). IREE's CUDA target also fails to compile these steps
(`'func.func' op failed to distribute` at the `[0,2,3]` reductions) unless
`IREE_EXTRA_FLAGS=--iree-codegen-llvmgpu-use-reduction-vector-distribution=false`, the flag
`ireeCompileArgs` already applies for rocm.

**2026-10-28, IREE 3.13.0 llvm-cpu (atlas, `deploy/iree_cpu_setup.sh`):** the gate PASSES on
llvm-cpu for both artifacts — `cifar8wb_bbn` worst rel 0.0051, `cifar8wb_bn` 0.0067, tier-1
|cos∠| ≤ 1e-6, 17–18 s each on local-task. So the miscompile is 3.11's; the vmvx route stays for
CI's pin. (3.11.0's local-task device also segfaults on every 224² artifact, which is why the
CPU stack moved — historical/IREE_BUILD.md.)

### Gradcheck on vmvx (2026-10-19)

`cifar8-bbn-gradcheck` on `cifar8wb_bbn_adam_train_step.mlir`, bs 128, h = 6e-5, 25 invokes,
31 min on vmvx. **Passes.** Tier 1: all 8 BN-followed convs at |cos∠| ≤ 1e-6 against head
controls ≥ 5.2e-3. Tier 2:

| group | ⟨g,δ⟩ | (L₊−L₋)/2 | rel |
|---|---|---|---|
| L1 conv 3→16 @32² | 4.10e-4 | 4.10e-4 | 1.2e-4 |
| L2 conv 16→16 @32² | 4.62e-4 | 4.61e-4 | 7.8e-4 |
| L3 conv 16→16 @16² | 3.29e-4 | 3.29e-4 | 2.3e-4 |
| L4 conv 16→16 @16² | 2.87e-4 | 2.87e-4 | 1.3e-3 |
| L5 conv 16→32 @8² | 3.66e-4 | 3.66e-4 | 3.0e-5 |
| L6 conv 32→32 @8² | 3.62e-4 | 3.62e-4 | 6.7e-4 |
| L7 conv 32→32 @4² | 3.39e-4 | 3.40e-4 | 1.0e-3 |
| L8 conv 32→32 @4² | 2.66e-4 | 2.66e-4 | 4.8e-4 |
| head | 2.555e-3 | 2.556e-3 | 5.4e-4 |
| whole net | 6.162e-3 | 6.131e-3 | 5.1e-3 |

Worst rel 5.1e-3 (whole), fp32 noise floor 2.9e-4, doubled-gradient control misses by 0.999.

## Runs to do (one 4060 Ti; `timing.txt` has the per-image BN binary at 757 s for 120 epochs)

Build: `lake build cifar8wb-bbn-ablation cifar8-bbn-gradcheck fwd-tie`. First, on the card:
`.lake/build/bin/cifar8-bbn-gradcheck` (XLA, seconds) — the gate at the lowerer the runs use.

| artifact | arms | command | cost |
|---|---|---|---|
| `runs/2026-09-01-cifar8w-6arm-constlr/` BN half → `runs/<date>-bbn-seeds/` | 3 optimizers × 5 seeds × 40 ep, f32 | `SUITE=bbn scripts/sweeps/seed_sweep.sh 0 "1 2 3 4 5"` runs all six arms per seed | ~25 min × 5 ≈ 2 h |
| `runs/2026-09-01-bnprec-seeds/` (tab:cifar_bf16) | the bf16 twins | the same sweep (six lines per log) | included |
| head-width sweep (`results.tsv`, cosine, 25 ep) → `runs/<date>-cifar8wb-bbn-grid-xla/` | 10 widths, single runs | `GPUS=0 scripts/sweeps/run_cifar8wb_bbn_grid_sweep.sh` (built 2026-10-22: `cifar8wb-bbn-grid`, `cifar8wbBbnG d`, renders its own width-slugged artifacts) | ≈ 25 min on one card |
| `fig:cifar_bn_curves` | from the SGD arm | free |
| `runs/2026-09-01-cifar8w-bn-xla-cuda/` (the §4 transcript) | 1 run, 3 arms | `.lake/build/bin/cifar8wb-bbn-ablation data adam` etc. | ≈ 13 min |

The no-BN half of every table stands. Expect the BN numbers to move: with batch statistics the
eval forward uses running means, which is the usual source of the val-accuracy dips the MNv2
chapter sees at full learning rate, so the "2.5–13× tighter spread" claim in particular needs
re-measuring rather than re-asserting. The sweep's OK line reports six final-epoch accuracies per
seed in arm order sgd / mom / adam / bf16sgd / bf16mom / bf16adam.

## Book text to touch afterwards

- §4 "What BN actually is": drop the per-image paragraph (or shrink it to the
  aside), restore "batch".
- `tab:cifar_optimizers`, `tab:cifar_bf16`, the head-width figure, the BN
  deltas in the lever text, the Recap's "number format is not measurable
  against seed noise" (holds only on the normalized net).
- The transcript in "Run it first" (new log).
- `README.md` CIFAR row (76.3) and `PLATFORMS.md` cifar column; the lakefile
  `cifar` tier wall-clock (currently claimed ~5 min; measured 19).
- Downstream demos that reuse the chapter-4 CNN: the GW demo's `cifarbn` arms
  (`runs/2026-09-11-gw-cifarbn-*`), ArASL, PlantVillage, EuroSAT. They are
  framed as "the chapter-4 CNN", so either they keep the per-image layer (and
  say so) or they rerun. Decide before touching the Bestiary text.
- The budget table row 4 ("BN at batch statistics") now has a cifar8 node tie to point at.

## Open questions

- Keep both layers in the chapter? Per-image normalization at 1,024 cells per
  group is a legitimate variant (InstanceNorm with affine) and the train = eval
  property is pedagogically useful for the step theorems. One option: the
  chapter trains batch BN, and a short "per-image variant" subsection keeps
  today's numbers as a comparison row. The per-image artifacts and specs are untouched, so both
  options stay open.
- Batch 128 at 32×32 gives 131,072 cells per channel statistic; the eval-time
  running-stat momentum (0.99 per step, ~100-step memory; `VerifiedConfig.bnMomentum`, Chapter 5's
  setting) is the knob that decides how noisy the val curve gets.
- The second half of the tie (chain = certified whole-net backward), and a JAX verified-path
  oracle case for `cifar8wb_bbn` (the per-net mirror in `jax/` does not exist yet; the ad-hoc
  `tests/cifar8_bbn_jax_ref.py` covers the same ground from a dump).

## Files

New: `LeanMlir/Proofs/Codegen/CnnBbnRender.lean`, `LeanMlir/Proofs/Nets/Small/Cifar8BbnStepTieB.lean`,
`apps/cifar/MainCifar8WideBbnAblation.lean`, `tests/TestCifar8BbnGradCheck.lean`,
`tests/cifar8_bbn_jax_ref.py`, the eight `verified_mlir/cifar8wb_bbn_*.mlir`.
Changed: `LeanMlir/Verified/NetsCore.lean` (spec), `LeanMlir/Proofs/Codegen/CnnArtifacts.lean`
(writers), `LeanMlir/Types.lean` (`vmvx` in `ireeCompileArgs`), `lakefile.lean` (two exes, the
tie in `Certs`), `scripts/regen_verified_mlir.sh` (prefix pairs), `scripts/sweeps/seed_sweep.sh`
(`bbn` suite), `tests/TestFwdTie.lean` (slug), `tests/TestCifar8AdamTrain.lean` (smokes),
`.github/workflows/proofs.yml` (the eight artifacts in the drift guard), `verified_mlir/MANIFEST.md`.

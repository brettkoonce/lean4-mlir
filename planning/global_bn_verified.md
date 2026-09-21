# global_bn_verified.md — all-reducing the BatchNorm statistics in the verified renders

**Opened 2026-09-12**, out of §6's phase-4 half. Every verified ImageNet pair that has a
BatchNorm net carries the same unmeasured confound, and this is the plan to remove it.

⭐⭐ **Decided 2026-09-20: remove it, not bound it.** The verified path is to *be* the JAX side —
BatchNorm over the global batch, the same function the reference computes — and the pair re-runs
in §3.5 are accepted if they follow. The doc opened leaving "bound it cheaply and decide it is
not worth removing" on the table; that option is closed. §1 now sizes the re-runs rather than
gating the work.

---

## ▶ NEXT SESSION — start here (updated 2026-09-21, night: §3.4 DONE as `260df19d`, §3.6 bf16 node kit; §3.5 is next)

▶▶ **Start at §3.5 — the pair re-runs.** Every BN net's DP render is sync-BN now, so every
committed BN pair number was trained on a function the tree no longer renders. §3.5 has the
queue (job config, variant, cost). Each is a multi-day training run: propose it with its
wall-clock and wait for the user's go, one at a time. Nothing below this line is open work
except §3.5 and the optional twin extensions (item 5).

**State.** §3.1–§3.4 are DONE — all five BatchNorm nets (ResNet-34, MobileNetV2, EfficientNet-B0,
ResNet-50, MobileNetV4) have a sync-BN DP render, a passing split-batch gate, and both proof twins
(forward: `*FwdGraphSync_full_shard`; whole step: `r34_net_syncTiedB`, `mnv2_net_syncTiedB`,
`efficientnet_net_syncTiedG`, `r50_net_syncTiedB`, `mnv4_net_syncTiedB` — every all-reduced
gradient IS the single-device node at batch `R·N`). §3.1–§3.4 are commits on `main`,
`2a39842f` … `99fc6a8d`, then `260df19d` (§3.4) and `55c130f7` (bounds table), ⚠ **NOT PUSHED**. Push is the user's call.
Read §2b's ⛔⛔, §3.2's gate table and §3.3's before touching numerics.

**Then (2026-09-21, later), `23db65a2`:** the ImageNet-shape probe, §3.3b —
`imagenet-syncbn-check` runs the artifacts the pairs train from (4×64, bf16) against a 1×256
single-device step, and passes on all three nets in both precisions. The shared runner gained a
replica count, shifted shards, a REPEAT column and per-precision bounds.

**Where things live.**

| what | file |
|---|---|
| the eight sync-BN ops + `R = 1` anchors | `Codegen/StableHLO.lean` |
| P1–P4 on the graph, `syncStats` | `Foundation/DataParallelSync.lean` |
| the three emit sites every renderer shares | `Codegen/SyncBnSites.lean` |
| gate runner (one `Cfg` per net) | `LeanMlir/SyncBnCheck.lean`; entries `tests/Test{Mnv2,Enet}SyncBnCheck.lean` (R34's own: `tests/TestR34SyncBnCheck.lean`) |
| the ImageNet-shape gate (4×64, bf16 / f32) | `tests/TestImagenetSyncBnCheck.lean` (`imagenet-syncbn-check <net> [f32]`) |
| twins | `Nets/ResNet/ResNet{34,50}Sync{B,StepTieB}.lean`, `Nets/MobileNet/MobileNetV{2,4}Sync{B,StepTieB}.lean`, `Nets/EfficientNet/EfficientNetSync{B,StepTieG}.lean` |
| the MBConv pieces the MNv2 / B0 / MNv4 step twins share | `Nets/EfficientNet/MBConvSyncTieB.lean` — depthwise / GAP / dense links, the depthwise, strided-depthwise and XLA-stem collectives |
| R50's one-replica sync graphs (gate inputs, never committed) | `.lake/build/r50sync/` — written by `ResNet50RenderB.lean` at build time; read by `tests/r50_dp_render_tie.py` and `r50-gradcheck` (`R50_GC_PATH`) |

**Run the gates** (2 GPUs, XLA; ~30–45 s each; each prints its column table):

    lake build resnet34-syncbn-check mobilenetv2-syncbn-check efficientnet-syncbn-check \
      mobilenetv2-dp-check efficientnet-dp-check
    unset HIP_VISIBLE_DEVICES
    export CUDA_VISIBLE_DEVICES=0,1 PJRT_REPLICAS=2 \
      PJRT_PLUGIN=$PWD/.venv/lib/python3.12/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so
    .lake/build/bin/mobilenetv2-syncbn-check    # SYNCBN_VERBOSE=1: per-BN-layer split table

The ImageNet-shape gate (4 GPUs; ~1.5–2 min each, most of it the first two XLA compiles):

    lake build imagenet-syncbn-check
    export CUDA_VISIBLE_DEVICES=0,1,2,3 PJRT_REPLICAS=4
    .lake/build/bin/imagenet-syncbn-check resnet34     # mobilenetv2 | efficientnet | resnet50 | resnet50bce | mnv4
    LEAN_MLIR_MEM_FRACTION=0.97 .lake/build/bin/imagenet-syncbn-check resnet34 f32   # ⚠ f32 only — see gotchas

**What is next, in order.**

1. ✅ **The bf16 / 4-replica probe** — §3.3b. Passes on R34 `momdp64bf16`, MNv2 `rmsdp64bf16`,
   B0 `rmsdp64bf16` and their f32 peers. Not yet covered: B0's `emarmsdp64dropdobf16` (EMA +
   host-fed drop masks, which the runner cannot feed) and MNv2's `adamdp64bf16`.
2. ✅ **The duplicated lemmas are one module** — `Nets/EfficientNet/MBConvSyncTieB.lean`
   (`Proofs.MBConvSyncTieB`), imported and opened by both twins; `inv_mul_R` is public in
   `ResNet34SyncStepTieB`. It also takes B0's symmetric strided-depthwise kit, because MNv4's
   render emits `depthwiseStridedWeightGradB`; MNv2's XLA-`SAME` strided depthwise stays in its
   twin. Net −362 lines across the twins (−373 / +11), +253 in the module.
3. ✅ **§3.4 ResNet-50 / MobileNetV4** — renders, gates, twins, and FIVE per-replica-identity
   gates re-pointed or retired (three planned, two found red on the way). §3.4 has the account.
4. ▶ **§3.5 re-runs**, then the
   book rows (`content.tex:5671`, `:7059`, `:8266` — `BN statistic group … 64 (per replica)`, the
   R50 `BN group` rows `:5935`, `:6089`, and the two `[TODO: global BN.]`). ⚠ Until a re-run, the
   committed pair numbers were produced by the OLD per-replica renders; the rows stay as they are.
5. ✅ **bf16, R34 and R50 — the node kit, §3.6.** ⛔ The sentence this item used to carry
   ("bf16 rounding is per element, so it commutes with sharding") is FALSE for the conv weight
   gradients: their `den` rounds the whole batch sum once, so on `R` replicas each rounds its own
   partial sum before the all-reduce. Every other bf16 node the two nets emit shards exactly.
   §3.6 states both, per node. Still outside every twin: the drop-path / dropout DP artifacts
   (B0's `*drop*` / `*do*`, R50's A2/A1 `…drop…`), and the depthwise bf16 kinds MNv2 / B0 / MNv4
   emit (same shape as §3.6's; their headers still carry the old one-line caveat).

**Gate policy decided this session (2026-09-21, approved).**
* MobileNetV2's split-batch bound is 3e-3, not R34's 1e-3 — XLA's GPU reductions are not
  bit-deterministic (0.97 / 1.05 / 1.14 / 1.49e-3 on the same seeds) and the per-layer table shows
  depth-compounded rounding (first six layers split-exact). Every net also carries a first-BN-layer
  criterion (≤ 1e-5). A new net sets its own bound beside its `Cfg`, with its per-layer evidence.
* `mobilenetv2-dp-check` / `efficientnet-dp-check` gradient bound 1e-4 → 1e-2: their DP backward
  is the sync-BN graph and the single-device artifact's is the two-pass graph (~1.1e-3 on `m`);
  forward still bit-exact. `mnv4-dp-check` took the same in §3.4 (1e-2 f32 / 5e-2 bf16); R50 has
  no dp-check (nor had R34).
* The DUPLICATED criterion on the gradient is its slot's **norm-rel** (the `*-dp-check` metric),
  not the regions' max-abs: one squared entry of RMSProp's `v'` moved the max-abs 4× between runs
  (1.5e-2 → 5.7e-2) while the norm-rel held at 1.6–1.7e-2. f32 reads 6.8e-4 – 1.0e-3 on every gate,
  so the 5e-3 default stands; a sum-not-mean divisor reads `R − 1`.
* ✅ **Per-net split bounds, APPROVED 2026-09-21.** The sharp criteria (first-layer split,
  DUPLICATED and FORMULATION statistics ≤ 1e-5, CONTROL ≥ 2e-3) are shared by every net and
  precision. The split-batch TEST bound is per net, because it bounds rounding that compounds
  with depth and conditioning — measured f32 spans 1.3e-4 (R34) to 2.2e-3 (R50), so one shared
  bound would leave the shallow nets' gates blind below 38× their own error. The one table of
  every value and its reason is `tests/TestImagenetSyncBnCheck.lean`'s module docstring
  ("The bounds, and why the split one is per net"): bf16 3e-3 / 1e-2 / 6e-3 / 3e-2 / 3e-2 and
  f32 1e-3 / 3e-3 / 1e-3 / 5e-3 / 3e-3 (R34 / MNv2 / B0 / R50 / MNv4); DUPLICATED gradient
  5e-2 bf16; B0's bf16 FORMULATION 5e-3; `mnv4-dp-check` gradient 1e-2 f32 / 5e-2 bf16. A new
  net sets its own, with evidence, in that table.

**Gotchas that cost time.**
* Editing `StableHLO.lean` is a 6-minute rebuild plus the corpus; `PerChannelBN.lean` rebuilds
  ~290 modules — batch changes to either.
* `simp` cannot rewrite inside `conv2d_weight_grad_has_vjp b x` (dependent position) — `rw`.
* Plain `simp` turns `Fin.natAdd` into `Fin.addNat` before `Fin.append_right` can fire — `simp only`.
* ⭐ Twins compile in 2–4 s: block lemmas at VARIABLE shapes, whole-net proofs as `have`-chains,
  each block's input passed explicitly (unification cannot recover it). Keep that shape.
* The index seam (conv chain `N·(c·h·w)` vs BN ops `N·(c·(h·w))`): typed graphs use `castIdx`
  (`h ▸ e`, `den_castIdx` by `subst`), ℝ-level chains use `.operand` leaves at `reassocB`;
  `batchShard_castIdx` commutes either with sharding.
* The per-layer statistics MEAN column shows huge relative errors on every expand-BN input: a
  bias-free 1×1 conv of a β = 0 BN output has batch mean exactly 0 at init — noise over noise.
* A shared helper module cannot live under `tests/` (not a lean_lib — "object file … does not
  exist"); put it under `LeanMlir/`.
* B0's loss divisor is a STRING argument (`bStr`); a run-time render at batch `B` passes `"{B}.0"`.
* The REORDER probe (swap batch halves) is exactly commutative and measures nothing; SENSITIVITY
  (perturb `x` by 1e-4) is the yardstick for gradient columns at a random-init operating point.
* A `formalization.yaml` row naming `config-tier.json` must also go in
  `scripts/gen_comparator_tier.py`'s `DECLS` (+ `MODULES`), then regenerate — CI's `--check`
  fails otherwise (it was red at `95fc72d0`; fixed in `d19009e5`). 30 theorems in the tier now.
* The docstring gate rejects bare backticked `dir/File.lean` paths in `LeanMlir/` docstrings —
  write a markdown link to the GitHub path.
* f32 R34 at batch 256 OOMs the default 11.68 GiB arena (`d2h: Out of memory … 8.54GiB`);
  `LEAN_MLIR_MEM_FRACTION=0.97` fits it. bf16 at 256 peaks 5.8–6.7 GiB on all three nets
  (`scripts/bf16_peak_memory.py`), so bf16 stays at the default.
* ⚠ `XLA_FLAGS=--xla_gpu_autotune_level=0` changed neither a number nor a compile time (51.6 s vs
  51.7 s) — the PJRT plugin appears not to read `XLA_FLAGS`. Don't cite a run under it as
  "autotuning off".
* ⛔ `LEAN_MLIR_MEM_FRACTION=0.97` is for f32 only. On `imagenet-syncbn-check resnet50bce` in
  bf16 it crowded out the d2h staging (`CUDA_ERROR_OUT_OF_MEMORY` on a host transfer, not a
  compile); the default arena runs it clean.
* ⛔ Run gate binaries as `lake build X && .lake/build/bin/X`. A `;` ran a stale
  `r50-accum-shard-tie` after its build failed (an imported renderer brought a public `cmp` that
  clashed with the file's `private def cmp`).
* A DP render change can falsify gates that never mention BatchNorm. §3.4 found two the plan had
  not listed: `regen_verified_mlir.sh check`'s forward ⊂ train-step pairs (red since `cad811ac`)
  and `drop-shard-check`'s "BN statistics must MOVE" witness. Grep `tests/` and `scripts/` for
  every consumer of a changed `*dp*` artifact and run each one.
* Full gate list for a commit here: `lake build Certs`, `lake build`, `lake build Apps`,
  `lake env lean tests/AuditAxioms.lean` (verdicts = directives, no `sorryAx`),
  `lake exe docstring-checkrefs`, `scripts/regen_verified_mlir.sh check`,
  `scripts/check_audit_coverage.py`, `scripts/check_target_names.sh`,
  `scripts/gen_comparator_tier.py --check`, `tests/comparator/run.sh` (~6 min), and the GPU gates
  — for R50 also `python3 tests/r50_dp_render_tie.py` (after building `ResNet50RenderB`).

---

## 0. The finding, and why it is fleet-wide rather than per-net

⭐⭐ **`allReduceMeanF` is the ONLY constructor in `SHlo` that takes a replica family.**

    | allReduceMeanF {n : Nat} (R : Nat) (hR : 0 < R) (t : String) (ds : List Nat)
        (g : Fin R → SHlo n) : SHlo n
    den (.allReduceMeanF R _ _ _ g) = fun i => (1 / (R : ℝ)) * ∑ r : Fin R, den (g r) i

Every BatchNorm constructor — `bnBatchF`, `bnBatchBack`, `bnBatchLABack`, `bnF`, `bnBack`,
`bnPerChannelF`, `bnPerChannelBack`, `bnPerChannelEvalF` — has the shape `SHlo n → SHlo n`.
A single expression. No replica family, so **structurally nothing to reduce over**.

⇒ **No verified render can compute global BN statistics, for any net, by construction.** This is
not an omission that some net's emitter forgot; it is a property of the AST. Do not go looking
for the one net that got it right.

Confirmed independently in the artifact: in `mobilenetv2in_rmsdp64bf16_train_step.mlir` every
`"stablehlo.all_reduce"` is named `%arsum<param-tag>` (`W`, `g`, `bt`, …), the first appears at
line 6964 of 12037, and the forward pass (lines 54–6963) contains no collective at all.

## 0b. And every JAX reference is on the other side of it

Audited 2026-09-12 across `jax/generated/`: **`pmap` 0, `shard_map` 0, `NamedSharding` everywhere,
`train_step` under `@jit`**, data placed with `NamedSharding(mesh, P('batch'))`. Under `@jit` the
program is written against the GLOBAL array, so `jnp.mean(x, axis=(0,2,3))` has global semantics
and GSPMD inserts the collective. ⚠ **The absence of an explicit `lax.pmean` is not evidence of
per-device statistics — it is what global semantics look like under GSPMD.** Had these been
`pmap`, that same line would be per-device and there would be no confound at all.

## 0c. Who is affected

| net | render | JAX ref | confound | disclosed in the book |
|---|---|---|---|---|
| ResNet-34 | BN, per-replica 64 | global 256 | yes | ✅ `BN statistic group` row, §5 |
| ResNet-50 | BN, per-replica 64 (**sync since §3.4**) | global 256 (2018) / 512 (A3) | yes, until §3.5 | ✅ `BN group` column; called "the largest known difference between the two paths, and *untested*" |
| MobileNetV2 | BN, per-replica 64 | global 256 | yes | ✅ §6, 2026-09-12 |
| MobileNetV4 | BN, **sync since §3.4** | global | **no** | n/a — no verified pair run yet; its first one trains on the sync render |
| EfficientNet-B0 | BN, per-replica 64 | global 256 | yes | ✅ `BN statistic group` row, §7 (`content.tex:8266`) |
| ConvNeXt-T | LayerNorm, no `bnBatchF` | no batch stats | **no** | n/a |
| ViT-Ti | LayerNorm, no `bnBatchF` | no batch stats | **no** | n/a |

The render column is what produced the committed pair numbers. Since §3.2 / §3.3 / §3.4 the R34,
MNv2, B0 and R50 DP renders normalise globally; their rows stand until §3.5 re-runs them.

⚠ LayerNorm normalizes per example over channels, so it is replica-invariant by construction.
ConvNeXt and ViT are the only two nets whose pair isolates the lowerer, and that is *why*.

---

## 1. Bounding the effect on the JAX side — the tool already existed

**Re-scoped 2026-09-20.** This was "⭐ DO THIS FIRST", framed as the cheap thing that decides
whether §2 is worth doing. ⛔ **It no longer decides that.** The call is that the verified path
should *be* the JAX side — global statistics, not a bounded difference from them — and re-runs
are accepted if they follow. So §2 is happening on its own case (`dpMeanGrad_ne_globalBatchGrad`:
the DP step provably is not the batch-`R·N` step today), and this section now sizes the **re-runs**
in §3.5 rather than gating the work.

⭐⭐ **`jax/scripts/bn_sharding_demo.py` is this measurement, and it has existed since
`8d488368`.** Imagenette R50, one device, fixed init and fixed data order, `conv_bn`
monkeypatched so all 53 BN sites pick up the knob: `GHOST_BN_GROUPS=k` reshapes `[N,C,H,W]` to
`[k, N/k, C,H,W]` and reduces over the inner axes, which is bit-identical to what `k` physical
shards compute. Configs are **paired by construction** — within a seed the only difference
between two runs is the BN grouping — so the per-seed difference is the estimator and init/
data-order noise cancels.

⚠ It had **never been run**: no `BN_SHARDING_RESULT` line anywhere in `runs/` or the docs.
`planning/archive/xla_pjrt_ladder.md:833` says it "already quantifies the synced-vs-sharded top-1
gap", which reads as a result but is only the tool existing. Anything citing that sentence is
citing nothing.

Three things were wrong with it, fixed 2026-09-20 before the first run:

* ⛔⛔ **The eval BN group moved with the training one.** The probe has no `training` flag —
  `evaluate` → `eval_batch` → `forward` → the same patched `conv_bn`, at eval batch 512. So
  `groups=k` normalised eval over `512/k` as well, and each config differed from its neighbour in
  two places at once. Eval is now pinned to global BN whatever the training grouping, which is
  also the faithful match: both ImageNet paths score from running statistics that estimate the
  GLOBAL mean/var. Checked numerically — across configs the eval path now differs by exactly 0,
  `groups=1` is still bit-identical to the stock global BN, and training genuinely differs.
* ⛔⛔ **A silent CPU fallback.** With four configs starting at once one lost its CUDA init
  ("a CUDA-enabled jaxlib is not installed. Falling back to cpu"), then sat at 1316 % CPU and
  56 GB of host RAM — starving the three GPU runs beside it from 100 % GPU down to 0 %, and
  heading for a number that was never going to arrive. The demo now aborts unless the backend is
  `gpu` (`ALLOW_CPU=1` to override). A run that scores on CPU is not a slower run, it is a fake one.
* A `SEED` knob, shifting init and data order together so configs stay paired within a seed.

▶ **The sweep is the whole curve, not one pair.** Batch 192 split `k` ways, one config per card,
80 epochs, fully annealed: `k = 1` (group 192, what the reference does), 2 (96), 4 (**48** — the
fourfold split the verified path actually has), 8 (24). `k = 4` is the point that matters; the
others say whether the effect is monotone and how fast it bites.

⚠⚠ **A short schedule does not give "the sign and the order" — it gives an upper bound.** The
book's own epoch tables settle this: ResNet-34's pair runs $+1.69$ at epoch 30 and $+2.04$ at 45
and ends at $-0.10$; MobileNetV2's runs $+4.67$ at epoch 10 and ends at $+0.01$. The BN-group
effect is large mid-schedule and washes out by the end of a cosine. A fully-annealed short run is
a legitimate experiment, but it anneals a schedule with less time to wash out, so read it as a
**conservative upper bound** on the converged effect: hundredths at 80 epochs settles §3.5 without
running anything at ImageNet scale; a point is inconclusive and the ImageNet probe is next.

⛔⛔ **RAN IT. Imagenette cannot resolve this question — closed 2026-09-21.** Two rounds, eight
80-epoch runs. The no-aug probe gave a clean monotone curve (192 → 38.70, 96 → 40.79, 48 → 43.30,
24 → 47.32); the augmented probe, two seeds, gave 41.83 global vs 47.95 at group 48, **Δ +6.13**.
Consistent, and unusable, for three reasons together:

1. **Wrong regime.** Both arms sit at 41–48 % with eval loss ≈ 5.0 against `ln 10 = 2.30` — R50's
   23.5 M parameters against 9,469 images, heavily overfitting. There the noise in a small BN
   group is the main regulariser available, so the gap measures regularisation headroom.
2. **Opposite sign to the real pairs.** Here the verified side's grouping looks +6.13 BETTER; on
   the committed pairs the reference won slightly at convergence (R34 $-0.10$, MNv2 $+0.01$).
3. **Augmentation barely moved it** (+4.60 → +6.13). Closing most of the regularisation gap
   should have shrunk a real BN-group effect.

⚠ The probe's crop is far weaker than the reference's and does not stop the overfitting:
`np.pad(224→252)` then crop back, translation only, **one offset shared by all 192 images**,
versus the reference's per-image `sample_distorted_bounding_box` with scale and aspect jitter.

⭐ **The best evidence on the converged effect is already in the book** — R34's and MNv2's own
epoch tables, real nets at real scale under real recipes, both converging to ≤ 0.10. No
Imagenette proxy improves on that. If a direct number is ever wanted it is a ~30-epoch ImageNet
R34 reference at BN64 vs BN256 (~4.5 h per arm on four cards), and nothing else.

Write-ups: `runs/2026-09-20-bn-group-sweep/RESULTS.md`, `runs/2026-09-20-bn-group-aug/RESULTS.md`.

---

## 2. The render work — global statistics by composition, not by a new BN node

**Rewritten 2026-09-20.** The 2026-09-12 version of this section scoped a replica-family BN
constructor and concluded "the spec moves — every tie restated". Neither holds once the pieces
already in the AST are used: the collective composes, and the tiers are stated at `bnBatchLA N`
with `N` a binder, so the global-batch spec is the existing spec at `N := R·N`. What moves is the
render and its faithfulness proofs, not the mathematics they are tied to.

### 2a. What the AST already has

* `allReduceMeanF {n} R hR t ds (g : Fin R → SHlo n) : SHlo n`, `den = (1/R) Σ_r den (g r)`
  (`den_allReduceMeanF`, `DataParallelNode.lean`). **Generic in `n`**: it reduces a `[oc]`
  statistic vector as readily as a `[oc,ic,k,k]` gradient. Emit and parser case exist; the emitted
  text is the pre-existing `%arsum{t}` / `%armean{t}` block, named from the tag.
* `bnBatchMeanB`, `bnBatchVarB : SHlo (N*(oc*(h*w))) → SHlo oc` — the per-channel batch
  statistics as their own nodes, `den = bnMean (N*(h*w)) (…bnchwFwd…)` / `bnVar …`
  (`StableHLO.lean:1957`). They exist so the train step can hand the statistics back to the
  driver, and `den` is by construction the statistic `bnBatchTensor4` normalises by.
* `bnBatchF` = `bnBatchLA` = `bnBatchTensor4 N oc h w`, which is
  `bnPerChannelFlat oc (N*(h*w))` conjugated by the `[N,C,H,W] → [C, N·H·W]` reindex: **each
  channel normalised over all its `N·h·w` cells**. `bnBatchBack` = `bnBatchTensor4_grad_input`,
  the three-term formula with its two reductions over the same `N·h·w`.

So the only thing missing is "normalise with statistics handed in", forward and backward.

### 2b. The sync-BN ops

**Rewritten 2026-09-20 (second pass), after pricing the edit against the skeleton language.**
The first pass gave each op its mathematical arity — `bnSyncF` three operands, the dy-statistic
four, `bnSyncBack` five. ⛔ **`SHlo`'s skeleton language does not have those arities.** It has
`.batched` (one operand) and `.batched2` (two), and nothing above two: `Raw` 4197/4202, `Tok`
4798/4799. Anything wider is a new `Raw` **and** `Tok` constructor **and** `toToks` **and**
`parseStack` **and** an emit arm — `batched2` itself is 15 sites across three files. Three new
arities is most of a session before a line of mathematics is written.

⭐⭐ **The way out is the one §2e already suggests for a different reason: carry the statistics
PACKED.** One `[2·oc]` vector instead of two `[oc]`s, one `[4·oc]` instead of four, and every
sync op lands back on the arities that already exist.

It works because of how this AST already separates its two kinds of input. A value the *graph*
computes is an `SHlo` operand; everything else arrives as a **named host buffer** — which is
exactly why `bnBatchBack` is single-operand today, taking `(xName : String) (x : Vec …)` beside
its one `dy`. μ and m2 are all-reduced *in* the graph, so they cannot be host buffers and must be
operands. Packing is what keeps that from costing an arity.

| op | type | operands | skeleton |
|---|---|---|---|
| ~~`bnBatchStatsB`~~ | removed 2026-09-21 — see ⛔⛔ below | | |
| `bnSyncF gName bName epsStr ε γ β` | `SHlo n → SHlo (oc+oc) → SHlo n` | 2 | `.batched2` |
| `bnSyncDyStatsB xName ε x` | `SHlo n → SHlo (oc+oc) → SHlo (oc+oc+oc+oc)` | 2 | `.batched2` |
| `bnSyncBack gName xName epsStr ε γ x` | `SHlo n → SHlo (oc+oc+oc+oc) → SHlo n` | 2 | `.batched2` |
| `bnSyncGammaGradB xName epsStr ε x` | `SHlo n → SHlo (oc+oc) → SHlo oc` | 2 | `.batched2` |
| `bnBatchVarAtB` | `SHlo n → SHlo oc → SHlo oc` | 2 | `.batched2` |
| `bnPackB` | `SHlo oc → SHlo oc → SHlo (oc+oc)` | 2 | `.batched2` |
| `bnStatsMeanB` / `bnStatsVarB` | `SHlo (oc+oc) → SHlo oc` | 1 | `.batched` |

⛔⛔ **`bnBatchStatsB` (`[μ ‖ E[x²]]` in one round) is GONE — measured wrong in f32, 2026-09-21.**
`resnet34-syncbn-check` (§3.2) put the one-round exchange 2e-4 off `adam64` in the handed-back
statistics after 36 layers and 15 % off in `m' = 0.1·g`; and its SENSITIVITY probe — the two-pass
graph on the same batch perturbed by 1e-4·N(0,1) per pixel — moved the two-pass graph's OWN `m'`
by 0.22. So the seven ops were right and the arithmetic was not: `σ² = E[x²] − μ²` costs
`ε·E[x²]/σ²` per layer (up to ~30× rounding at R34's activation scales), which compounds to 2e-4
over the depth, and at random init the gradient amplifies forward drift ~1000×. The exchange is
now **Chan's parallel variance, in two rounds**: `bnBatchMeanB` all-reduced → μ; then
`bnBatchVarAtB x μ = σ²_r + (μ_r − μ)²` (two-pass on the replica) all-reduced → σ² EXACTLY
(`bnVar_shard_chan`, `bnVar_row_shard_chan`); `bnPackB` packs `[μ ‖ σ²]` for the consumers, which
read σ² directly. Three collectives per BN layer per step (§2e), each a `[oc]` or `[4·oc]` vector.
The ℝ-level API stays at `(μ, m2)`: a consumer's `den` hands `bnSyncTensor4` `m2 := σ² + μ²`, and
`global_var_add_sq` folds that back to the second moment in the graph lemmas.

⛔⛔ **A FIFTH op, found 2026-09-21 while stating P4.** `bnGammaGradB`'s emit recomputes μ/σ²
from its own operand — `reduce … [0,2,3]` over `B·h·w`, i.e. the SHARD — and builds `x̂` from
them; its `den` is `bnPerChannel_grad_gamma`, whose `x̂` is `bnXhat` of the shard's row. Under
sync-BN the forward normalised with the GLOBAL `x̂`, and `∂L/∂γ_c = Σ dy·x̂` has to use the same
one, so a sync render that kept `bnGammaGradB` would emit the wrong γ gradient with every other
node right. `bnSyncGammaGradB` reads μ/m2 off the same packed operand the forward read
(`bnSync`'s prologue verbatim, then `bnGammaGrad`'s tail); `den` is
`bnSyncPerChannel_grad_gamma`, anchored at `R = 1` by
`bnSyncPerChannel_grad_gamma_at_own_stats` and `den_bnSyncGammaGradB_allReduce_R1`. β's
gradient is `Σ dy`, reads no statistic, and `bnBetaGradB` stays. §3.2's emit count per net is
therefore the BN-forward sites PLUS the γ-gradient sites.

⭐⭐ **Index the packed results `oc+oc`, NOT `2*oc`** — then the packing function is Mathlib's
`Fin.append` (`Mathlib/Data/Fin/Tuple/Basic.lean:309`, itself `Fin.addCases`) and the two
projections are `Fin.append_left` / `Fin.append_right`, already proved. At `2*oc` none of that
applies without a `two_mul` rewrite, and the alternative is hand-rolling a `packPair` with a
`Fin` subtraction and an `omega` side-goal at every use. `Vec n` is `Fin n → ℝ`, so
`den (bnPackB a b) = Fin.append (den a) (den b)` is the whole definition and the consumers read
its halves off the two `append` lemmas (`den_syncStats_left` / `_right`). ⚠ The 2026-09-18 Mathlib-reuse
audit found every lemma this tree hand-rolled was already in Mathlib; this is that lesson applied
before the fact rather than after.

* **The statistics are `syncStats`** (`Foundation/DataParallelSync.lean`): `bnBatchMeanB`
  all-reduced → μ; `bnBatchVarAtB x μ` (the replica's two-pass `σ²_r + (μ_r − μ)²`) all-reduced
  → σ²; `bnPackB` → `[μ ‖ σ²]`. Never the per-replica variance averaged on its own — the variance
  of a union is not the mean of the variances, and the `(μ_r − μ)²` term is what Chan adds.
* `bnSyncF`'s `x` is the graph operand (it is the activation flowing through); `γ β` are host
  literals. `den` is `bnSyncTensor4` at `μ` and `m2 := σ² + μ²` read out of the packed operand,
  i.e. `γ·(x − μ)·(σ² + ε)^{-1/2} + β` per channel.
* ⭐ **`bnSyncDyStatsB` passes `[μ ‖ σ²]` THROUGH into its result**, so its output is
  `[μ ‖ σ² ‖ mdy ‖ mdyx]` and ONE all-reduce carries all four into the backward. Re-averaging an
  already-global quantity over the replicas is the identity (`dpMean_const_mul`), so the
  pass-through is free mathematically and buys the collective. `x` is a host literal, as in
  `bnBatchBack`.
* `bnSyncBack` then needs only `dy` and that one `[4·oc]` vector.

A BN layer on replica `r`, with `x : Fin R → SHlo n` the replica family of its input, renders as

    st  := syncStats R _ "{g}mu" "{g}var" [] [] x                               -- [μ ‖ σ²]
    y_r := bnSyncF … (x r) st

and its backward, with `dy` the family of output cotangents, as

    dst  := allReduceMeanF R _ "{g}dst" [] (fun r => bnSyncDyStatsB … (dy r) st)  -- [μ‖σ²‖mdy‖mdyx]
    dx_r := bnSyncBack … (dy r) dst

⭐ **Three collectives per BN layer per step** (μ, then Chan's σ², then the backward's packed
four) — the first-pass costing was two, before the one-round `E[x²]` exchange was measured to
drift; the payload is still one small vector each (`[4·oc]` ≤ 8,192 floats, 32 KB).

This stays inside the SPMD convention `allReduceMeanF` already uses (one skeleton, `skel` reads
replica 0; each replica's own activation is the `r`-th member). At `R = 1` every `allReduceMeanF`
emits nothing and threads its operand, so the single-device render is
`bnSyncF x (bnPackB (bnBatchMeanB x) (bnBatchVarAtB x (bnBatchMeanB x)))`, which denotes
`bnBatchF x`: the `R = 1` artifacts need not move at all, and 2e says whether to let them.

The running statistics the step hands back (`bnBatchMeanB`/`bnBatchVarB` passthrough slots,
`VerifiedTrain.lean:108`) become `bnStatsMeanB`/`bnStatsVarB` of `st` — `μ` and `σ²` read straight
off the packed vector (`den_bnStatsMeanB_allReduce` / `_VarB_`): global, exactly what the reference's `_bn`
EMAs. Today the driver reads replica 0's shard-of-64 statistics, so this also aligns the eval
running stats, not only the training forward.

⚠ Adding a `batched`/`batched2`-riding `SHlo` op is **5 sites, all in `StableHLO.lean`** —
inductive, the `den` arm, the `den_*` simp theorem, the `skel` arm, the emit arm. `parseStack`
dispatches on the tag generically (`StableHLOParse.lean:183`), so the round-trip needs nothing;
the emit arms are a `match tag, info` case each (8624). ⛔ Still gate with `lake build Certs` — a
bare `lake build` misses it. Editing `StableHLO.lean` rebuilds the corpus (~6 min for the file,
then every module below it), so: every op change in one edit session.

### 2c. The proofs, and why the spec does not move

Let `e : Fin R × Fin N ≃ Fin (R*N)` be the shard (`DataParallel.lean` states its theorems at an
arbitrary one; the contiguous cut the shim makes is `finProdFinEquiv`), lifted to the
`[N,C,H,W]` layout.

* **P1, forward.** `den (bnSyncF (x r) μ_g m2_g) = shard_r (bnBatchLA (R*N) … (concat x))`.
  Algebra over `bnPerChannelFlat`: with equal shard sizes the global mean is the mean of the shard
  means, likewise the second moment, and `bnVar = meanSq − mean²`. One lemma, generic in
  `R N oc h w`.
* **P2, backward.** `den (bnSyncBack … μ_g m2_g mdy_g mdyx_g (dy r)) = shard_r
  (bnBatchTensor4_grad_input (R*N) … (concat x) (concat dy))`. The three-term formula's two
  reductions are means over the global batch, which are means of the per-shard means. Same shape
  as P1. `bnBatchTensor4_grad_input_correct` then says the global backward is the VJP of the
  global forward, so the sync backward on shard `r` is the shard-`r` block of the true
  global-batch input-VJP — including the cross-shard terms, which is what makes the all-reduced
  parameter gradient below exact.
✅ **P1 and P2 are PROVED** (2026-09-21) — see §3.1. Both landed in the form stated here: the
right-hand side is the existing definition at `N := R·N`, so the spec did not move.
✅ **P3's kit and P4 landed the same day** — §3.1b. ⚠ P4 carries a `1/R` the paragraph below
does not: see "the divisor" in §3.1b.

* **P3, whole net.** Every other op in every BN net's chain is `batchMap N` of a per-example op
  (conv, relu, pool, GAP, dense, the CE cotangent), and `batchMap` commutes with `shard`/`concat`
  by definition. So by induction on the chain, the sync-DP forward graph on replica `r` denotes
  `shard_r` of the single-device batch-`R·N` forward, and the per-replica gradient nodes denote
  the shard-`r` blocks of the single-device batch-`R·N` backward. ⭐ **The spec is unchanged:**
  it is `*ForwardB_full (R*N) w`, the existing T1/T2/T3 statements at `N := R·N`.
* **P4, the step.** The positive twin of `dpMeanGrad_ne_globalBatchGrad`:
  `dpSyncGrad_eq_globalBatchGrad` — the all-reduced mean of the per-replica parameter-gradient
  nodes equals the batch-`R·N` parameter gradient (each replica's weight gradient is its shard's
  contribution to the global one, P2/P3; the mean over `R` of `(1/N)`-scaled shard sums is the
  `(1/(R·N))`-scaled global sum). With `dpIterate_lockstep`, `n` sync-DP steps are `n`
  single-device steps at batch `R·N`. That retires the sentence every DP render header carries
  today: "this does NOT equal a single-device step at the global batch — BN normalises per
  replica" (`ResNet34RenderB.lean:1586`).

What the per-net files gain is a **DP twin** of the T2 forward-graph faithfulness and of the T3
whole-net backward tie, each proven from the existing per-op lemmas plus P1/P2 through the chain;
the whole-net ℝ statements they tie to are the ones already there. Sizes of the files being
twinned (T2 + T3): R34 318 + 462, R50 455 + 291, MNv2 365 + 488, MNv4 924 + 658, B0 527 + 306.
Expect a twin at a third to a half of that, mostly the chain walk with one lemma swapped.

### 2d. The numeric gate that becomes possible

Today's `*-dp-check` gates give both replicas the **same** batch, so `all_reduce(add)/R` is the
identity and the check is bit-exact but blind to statistics ("R34 is about SPLITTING a batch —
2×32 really is not 1×64", `lakefile.lean:1583`). With sync-BN, 2×32 **is** 1×64 up to float
summation order. New gate per net, `<net>-syncbn-check`: one batch of `2N` on one device versus
the same batch split across two replicas with the sync render; compare the returned loss, every
parameter gradient (or `m`), and the handed-back statistics at a tight relative tolerance
(reduction order differs, so not bit-exact; `1e-5` is the right order in f32, tighter still if the
graph is run in f32 with bf16 off). This is the check the render header asks for ("a collective
needs its own numeric check"), and it is exactly the identity that per-replica BN cannot satisfy.

### 2e. Cost

**Three per BN layer per step** (revised 2026-09-21 — see §2b's ⛔⛔): μ (`[oc]`), Chan's σ²
(`[oc]`), and the backward's `[μ ‖ σ² ‖ mdy ‖ mdyx]` (`[4·oc]`, ≤ 8,192 floats / 32 KB):

| net | BN layers | collectives / step | verified ms/step today | at 30–60 µs each |
|---|---|---|---|---|
| ResNet-34 | 36 | 108 | ~175 (14.6 min / 5,004 steps) | +3–6 ms, 2–4 % |
| MobileNetV2 | 52 | 156 | 97 | +5–9 ms, 5–10 % |
| EfficientNet-B0 | 49 | 147 | 134 | +4–9 ms, 3–7 % |
| ResNet-50 | 53 | 159 | — | +5–10 ms |
| MobileNetV4 | 77 | 231 | — | +7–14 ms |

⚠ These are latency-bound, not bandwidth-bound — `[4·oc]` at 32 KB is nothing on the wire, so
doubling the payload to halve the count is the right trade at every net here.

XLA's all-reduce combiner cannot merge them: each layer's forward depends on the previous layer's
normalised output, so the collectives are sequential by construction. The reference pays the same
collectives — GSPMD inserts one per BN for `jnp.mean(axis=(0,2,3))` on a sharded array — so the
pairs stay like-for-like, and the reference's 79 / ~92 ms/step already include them.

The `R = 1` artifacts: leaving them on `bnBatchF` keeps 0 bytes moving in `verified_mlir/` for
the single-device renders and keeps every existing tie untouched; the DP renders are where the
function changes. Decide per net at render time; the default is to touch only the `*dp*` files.

---

## 3. Work packages

### 3.1 The kit (once)

The sync-BN ops (2b), P1/P2 (2c) in `Foundation/PerChannelBN.lean` beside `bnBatchTensor4`, P4 in
`Foundation/DataParallel.lean` beside its negative twin, the `allReduceMeanF`-at-`[2·oc]`/`[4·oc]`
emit checked against the parser. Gate: `lake build Certs`, parser round-trip, AuditAxioms.

▶ **Landed 2026-09-21 — the Foundation `R = 1` story, complete both directions, green.**
`Certs` 4,018 jobs exit 0; `AuditAxioms` 0 `sorryAx`, 1,381 verdicts for 1,381 directives, every
new theorem on the standard 3 axioms.

| in `Architectures/BatchNorm.lean` | |
|---|---|
| `bnMeanSq` | the second moment |
| `bnVar_eq_bnMeanSq_sub_sq` | `σ² = E[x²] − μ²` |
| `bnMean_shard` / `bnMeanSq_shard` | statistic of the whole = mean of the shards' statistics |
| `bnSync_grad_input` | the three-term backward, every reduction handed in |
| `bnSync_grad_input_at_own_stats` | …at its own statistics it IS `bn_grad_input` |

| in `Foundation/PerChannelBN.lean` | |
|---|---|
| `bnEvalForward_at_own_stats` | frozen-stats BN at own stats IS `bnForward` |
| `bnSyncTensor4` + `_at_own_stats` | sync forward at `[N,C,H,W]`, and the `R=1` anchor |
| `bnSyncPerChannel_grad_input`, `bnSyncTensor4_grad_input` + `_at_own_stats` | ditto backward |

⭐⭐ **The four `*_at_own_stats` anchors are the drop-in licence.** At `R = 1` every
`allReduceMeanF` threads its operand, so the sync render hands in exactly the statistics the
batch would have computed — and these say the result is the function the committed tiers are
already tied to. So the `R = 1` artifacts need not move, and `R > 1` is a statement about
SHARDS, not about BatchNorm.

⭐ `bnMeanSq_shard` is a **one-line application** of `bnMean_shard`, no tactic proof. The
variance has no such sibling and cannot: that is the asymmetry forcing the design, now visible
in the code rather than asserted in prose.

⚠ In `bnSync_grad_input_at_own_stats` the variance rewrite must fire BEFORE `bnMean` unfolds, or
the two sides get different arguments under the square root. P2 will bite the same way.

▶ **The ops landed 2026-09-21**, each 5 sites, parser untouched. After §3.2's Chan rework the set
is eight: `bnBatchVarAtB`, `bnPackB`, `bnSyncF`, `bnSyncDyStatsB`, `bnSyncBack`,
`bnSyncGammaGradB`, `bnStatsMeanB`, `bnStatsVarB` (the first cut's `bnBatchStatsB` is gone).

⭐⭐ **And the DROP-IN is proved on actual graph nodes, both directions:**

    StableHLO.den_bnSyncF_allReduce_R1     -- bnSyncF ∘ (R=1 syncStats) = bnBatchTensor4
    StableHLO.den_bnSyncBack_allReduce_R1  -- bnSyncBack ∘ allReduceMeanF 1 ∘ bnSyncDyStatsB
                                           --   ∘ (R=1 syncStats) = bnBatchTensor4_grad_input

These are the graphs a sync render actually emits. At `R = 1` both collectives collapse to their
single operand and what is left is exactly what today's `bnBatchF`/`bnBatchBack` renders denote.
⇒ **the `R = 1` artifacts need not move, and `R > 1` is now purely a question about how shard
statistics compose** (`bnMean_shard`/`bnMeanSq_shard`) — BatchNorm itself is already accounted for.

✅ **`StableHLO.roundtrip` covers the new ops for free.** It is `∀ {k} (a : SHlo k), parse (toToks
(skel a)) = some (skel a)` — universally quantified, and it still compiles, because every sync op
rides `.batched`/`.batched2`.

⚠ For 3.2: `bnSyncF` is indexed `N*(oc*(h*w))` (matching `bnBatchTensor4`/`bnBatchMeanB`), where
`bnBatchF` is at `N*(oc*h*w)` — `bnBatchLA` is `bnBatchTensor4` conjugated by
`Fin.cast (Nat.mul_assoc oc h w)`. Not a blocker: the renderers build a fresh `.operand` per node
and `ResNet34RenderB` already carries BOTH zero-vectors (`zin` at `c*hh*ww`, `zbn` at
`c*(hh*ww)`) for exactly this.

▶ ⭐⭐⭐ **P1 AND P2 LANDED 2026-09-21.** `Certs` 4,018 exit 0; audit 1,392/1,392, no `sorryAx`.

    bnSyncTensor4_shard_eq_global             -- P1
    bnSyncTensor4_grad_input_shard_eq_global  -- P2

Handed the global statistics — each `(1/R)·Σ_r'` of a per-replica quantity, i.e. what
`syncStats` (then `allReduceMeanF` of `bnSyncDyStatsB`) denotes — replica `r`'s sync
forward/backward equals `batchShard r` of `bnBatchTensor4` / `bnBatchTensor4_grad_input` run on
the whole `R·N` batch. ⭐ **The spec does not move: both right-hand sides are the EXISTING
definitions at `N := R·N`**, exactly as §2c predicted.

⭐⭐ **The decomposition that made it tractable.** Each splits in two, and only one half carries
mathematics:

* **(a) pointwise ⇒ sharding commutes** (`bnSyncTensor4_batchShard`,
  `bnSyncTensor4_grad_input_batchShard`). Once the statistics are FIXED, sync-BN reads only
  `x t` and `dy t` (`bnSyncTensor4_apply`, `bnSyncTensor4_grad_input_apply`) — cells mix only
  when statistics are computed, and sync-BN has hoisted that into the collective. Sharding
  preserves a cell's channel (`bnchwChan_batchShard`), so this half is pure index bookkeeping
  and holds for ANY statistics, right or wrong. Three-line proofs.
* **(b) the statistics really are the all-reduced ones** — `bnMean_row_shard`,
  `bnMeanSq_row_shard`, `bnMean_pair_row_shard`, all `bnMean_shard` transported along
  `bnchwFwd_row_batchShard`: **the global channel row restricted to replica `r`'s block IS that
  replica's own channel row**. That layout step is the one the `[N,C,H,W] → [C, N·H·W]` relabel
  was hiding, and it is the only place the two worlds have to be reconciled.

⭐ **The cross-shard terms are in.** `mdyx` averages `x̂·dx̂` with `x̂` built from the GLOBAL `μ`,
`m2` — not the shard's own — which is why `bnSyncDyStatsB` consumes the already-reduced vector
instead of recomputing. That is what makes each replica's output the true shard-`r` block, and
hence the all-reduced PARAMETER gradient exact rather than approximate. P4 now has what it needs.

⚠ `bnMean_shard` was generalised to an arbitrary target index `Fin M` with `M = R*m` DERIVED from
the equiv by cardinality, rather than assumed — without that it does not apply at `(R*N)*(h*w)`
vs `R*(N*(h*w))`, and every use downstream needs it.

▶ **Increment 1 of 4 — the ops** (history; ⛔ the one-round exchange it describes was replaced
by Chan's two rounds in §3.2, and `bnBatchStatsB` removed).
* `bnMeanSq` + `bnVar_eq_bnMeanSq_sub_sq : bnVar n x = bnMeanSq n x - bnMean n x * bnMean n x`
  (`Architectures/BatchNorm.lean`, beside `bnVar`). Still how a consumer's `den`, stated at
  `(μ, m2)`, reads σ² (`global_var_add_sq`); no longer what the replicas exchange.
* `bnBatchStatsB {N oc h w} : SHlo (N*(oc*(h*w))) → SHlo (oc+oc)`, all 5 sites — `[μ ‖ E[x²]]`
  by one reduction pair and a `stablehlo.concatenate`. Removed 2026-09-21.
* Gates: `lake build Certs` 4,018 jobs exit 0, no warnings. `tests/AuditAxioms.lean` clean — the
  new theorem on the standard 3 axioms, 1,375 verdicts for 1,375 directives, no `sorryAx`.
* ✅ **Both predictions held**: `StableHLO.lean` rebuilt in **374 s**, and the **parser needed zero
  changes** — `.batched` is generic, so the remaining three ops are the same 5-site shape.
* The emitted MLIR was first checked numerically by 3.2's R34 swap and `resnet34-syncbn-check` —
  which is what found the one-round exchange's drift.

(History.) Priced 2026-09-20: **20 sites, all in `StableHLO.lean`** — 5 per op (inductive, `den` arm, `den_*`
simp theorem, `skel` arm, emit arm), because 2b's packing keeps every op on the existing
`.batched`/`.batched2` skeletons and `parseStack` dispatches on the tag generically. The
mathematics, not the plumbing, is what makes this a session: P2 is the three-term backward with
its cross-shard terms. ⭐ Land `bnBatchStatsB` first — it is a pure sibling of the existing
`bnBatchMeanB`/`bnBatchVarB`, needs no new lemma, and proves out the 5-site path before the three
ops that carry real proof obligations.

### 3.1b ✅ P3 kit + P4 — landed 2026-09-21

**Where it lives.** `Foundation/DataParallelSync.lean` (new, piece 3 of the DP story, a leaf
importing `DataParallelNode`; in the `Certs` roots and the axiom audit), plus the ℝ-level
positive twin beside its negative one in `Foundation/DataParallel.lean`, plus the fifth op
(§2b) and the γ/β row-split lemmas in `PerChannelBN.lean`.

* **P3 — the kit, not the walk.** `batchShard_batchMap`, `batchShard_batchMapAux`,
  `batchShard_map` / `batchShard_zipWith`, `batchSlice_batchShard`: sharding commutes with
  every per-example lift, definitionally. The chain induction itself is per net and stays in
  §3.2–3.4 — this file supplies its non-BN cases (those five) and its BN cases (next bullet).
* ⭐⭐ **P1 / P2 / P2γ on the GRAPH at any `R`** — `den_bnSyncF_allReduce`,
  `den_bnSyncBack_allReduce`, `den_allReduceMeanF_bnSyncGammaGradB`. The exact subgraphs a sync
  render emits (`bnSyncF` fed by `syncStats`; `bnSyncBack` fed by the outer collective over the
  replicas' `bnSyncDyStatsB`, each fed by `syncStats`), under the
  induction hypothesis `∀ r, den (x r) = batchShard r X` (and its `xv`/`dy` twins), denote
  `batchShard r` of `bnBatchTensor4` / `bnBatchTensor4_grad_input` at `N := R·N`, and the γ
  collective denotes `1/R` of `bnPerChannel_grad_gamma` at `N := R·N`. The `*_allReduce_R1`
  anchors are these at `R := 1`. ⭐ The pass-through in `bnSyncDyStatsB` costs one lemma,
  `dpMean_const_mul` — re-averaging a replica-independent value is the identity — and nothing
  else.
* ⭐⭐ **P4.** ℝ level (`DataParallel.lean`, the positive twin of
  `dpMeanGrad_ne_globalBatchGrad`): `dpSyncGrad_eq_globalBatchGrad` — when replica `r`'s
  gradient is `(1/N) Σ_n c (e (r,n))`, ITS examples' terms of a global gradient
  `(1/(R·N)) Σ_m c m`, the collective's mean is that global gradient. `meanLoss_shard`'s
  arithmetic for vectors; the `c m` may depend on the whole batch (they do, through the
  statistics), which is exactly what `dpMeanGrad_eq_globalBatchGrad_of_perExample` could not
  allow. Node level (`DataParallelSync.lean`): `den_allReduceMeanF_convWeightGradB_shard`,
  `den_allReduceMeanF_bnBetaGradB_shard` and the γ statement — the collective over the replicas'
  gradient nodes, each on its shard at the shard-`r` block of the global cotangent, is `1/R` of
  the batch-`R·N` gradient node at the same cotangents. Every other `*GradB` composes by the
  same three lines (`simp only [den]`, the shard hypothesis, `sum_finProdFinEquiv`).

⚠⚠ **The divisor, and where the `1/R` goes.** A DP render divides its loss cotangent by the
PER-REPLICA batch (`divConstB N`, `ResNet34RenderB.lean:1448`); the batch-`R·N` step it is
compared to divides by `R·N`. So at a common per-example cotangent the replica mean is `1/R` of
the global sum, and the two divisors differ by exactly that `R`. §2c's P4 sentence ("the mean
over `R` of the `(1/N)`-scaled shard sums is the `(1/(R·N))`-scaled global sum") is the ℝ-level
theorem; at the node the `1/N` sits upstream in the cotangent graph, and the per-net twin
reconciles them with ONE linearity step — `HasVJP.backward_smul` (in `DataParallelSync.lean`;
the whole-net backward is a `HasVJP.backward`) — rather than by carrying a factor through every
op of the chain. ⭐ T3's ties already bind the divisor as a real `B` separate from `N`
(`r34_net_tiedB (N : Nat) … (α B : ℝ)`), so the DP twin can be stated at `B := N` against the
global instance at `B := R·N` without touching the tie.

Gates (after §3.2's Chan rework, 2026-09-21): `lake build Certs` 4,019 jobs exit 0, no
warnings; `AuditAxioms` 0 `sorryAx`, 1,417 verdicts for 1,417 directives, every new theorem on
the standard three axioms; `StableHLO.roundtrip` still universally quantified, so the parser
cases of all eight ops are free; `regen_verified_mlir.sh check` 192/192; `resnet34-syncbn-check`
passes.

▶ **Then §3.2** — whose R34 swap plus `resnet34-syncbn-check` was the first numeric check of the
emitted MLIR.

### 3.2 ResNet-34 — first, the template

▶ **Render + gate LANDED 2026-09-21; the T2/T3 DP twins are what remains.**

* **Render.** Three helpers in `ResNet34RenderB.lean` — `bnFwdSite` / `bnBackSite` /
  `bnGammaSite` — replace all 36 BN sites (8 forward emit sites, 8 backward, 8 γ-gradient, the
  stem's three, and `bnStat`); `replicas`/`sync` are threaded through the 32 block calls and the
  chain. At `replicas ≤ 1` every helper emits the old node, so all 20 single-device R34 artifacts
  are byte-identical (checked); the five `*dp*` artifacts re-rendered (`adamdp`, `adamdp128`,
  `resnet34in_momdp64`, `momdp64bf16`, `momdp128`): 218 collectives in `adamdp` = 110 parameters
  + 36 × 3. The header's "does NOT equal" paragraph is rewritten; `formalization.yaml` carries
  `dpSyncGrad_eq_globalBatchGrad` and `den_bnSyncBack_allReduce` beside the negative twin.
  ⚠ A `forceSync` knob renders the sync graph at ONE replica (every collective empty) for the
  gate; never a committed artifact.
* ⭐⭐ **`resnet34-syncbn-check` (2 GPUs, XLA), PASSING** — and it earned its keep first. The
  one-round `E[x²]` exchange failed it (§2b ⛔⛔); the gate's columns are what diagnosed why, and
  they are the reading of the kit now:

  | column | what it compares | statistics | `m' = 0.1·g` |
  |---|---|---|---|
  | FORMULATION | one-replica sync graph vs two-pass `adam64`, same batch | **0.000000** | 7e-4 |
  | DUPLICATED | `DP_sync([A\|A])` vs one-replica sync graph on `A` | **0.000000** | 5e-4 |
  | TEST (split) | `DP_sync([A\|B])` at 2×32 vs `adam64` on `[A\|B]` | 1.7e-4 | 0.15 |
  | CONTROL | `DP_sync([A\|B])` vs `mean(single_32(A), single_32(B))` (the old identity) | 3.4e-3 | 0.55 |
  | SENSITIVITY | two-pass graph vs itself, `x` perturbed by 1e-4·N(0,1) | 3.5e-4 | **0.22** |
  | REORDER | each graph vs itself on `[B\|A]` | 0 | 1.8e-5 |

  So: the sync ops' arithmetic IS the two-pass arithmetic (FORMULATION bit-exact); the collective
  composes them exactly (DUPLICATED bit-exact); the split identity holds to 1.7e-4 on the
  statistics — the 32-row and 64-row programs round their reductions differently (~1e-6/layer,
  compounded by 36 layers), 20× under the per-replica CONTROL; and `m'` cannot be bounded at
  this operating point at all, by any two f32 implementations: a 1e-4 forward perturbation moves
  the two-pass graph's OWN gradient by 0.22 (random init, random data, ~1000× amplification).
  The verdict is therefore on FORMULATION ≤ 1e-5, DUPLICATED ≤ 1e-5 (statistics) / 5e-3 (`m'`),
  TEST statistics ≤ 1e-3, CONTROL ≥ 2e-3; the gradient columns are printed against SENSITIVITY.
  ⚠ REORDER (swapping the two halves) is exactly commutative and measures nothing about rounding;
  the sensitivity probe is the yardstick.
* ✅ **The DP twins of T2 and T3 — LANDED 2026-09-21.** Both on the standard three axioms, each
  file ~3 s to elaborate.

  | file | capstone | says |
  |---|---|---|
  | `Nets/ResNet/ResNet34SyncB.lean` | `StableHLO.resnet34FwdGraphSync_full_shard` | the sync-BN forward graph, as a family over `R` replicas, denotes on replica `r` `batchShard r` of `resnet34ForwardB_full (R * N) w X` |
  | `Nets/ResNet/ResNet34SyncStepTieB.lean` | `ResNet34SyncTieB.r34_net_syncTiedB` | every one of the 110 all-reduced parameter gradients — replicas at loss divisor `B`, their own sync-BN backward chain — IS the single-device gradient node at batch `R·N`, loss divisor `R·B`, at T3's chain there |

  The right-hand sides are the existing definitions at `N := R·N`, so `r34_net_tiedB` at that
  instance ties them to the certified gradient: the spec did not move. The DP render header now
  names both theorems (five `*dp*` artifacts, three comment lines each; single-device artifacts
  untouched).

  **The index seam went BOTH ways, one per tier.** T2 is a typed graph, so it took option (b):
  `castIdx h e := h ▸ e` with `den_castIdx` by `subst` — the move `ConvNeXtRenderB.reassocB`
  already made, and emission-neutral. The sync site is `bnSyncSiteLA` and its shard lemma
  `den_bnSyncSiteLA` is `den_bnSyncF_allReduce` read through `batchShard_castIdx` (sharding
  commutes with any per-example relabelling, `subst`-proved). T3 is ℝ-level like its twin, so it
  took option (a): the replica BN link `bnSyncInB` is the `den` of the emitted nodes over
  `.operand` leaves at `reassocB`, the replica peer of `bnInB`.

  **The `1/R`, resolved without a factor in the statement.** A replica divides its loss by `B`,
  the single-device step by `R·B`, so replica `r`'s loss cotangent is `R ×` its shard of the
  global one (`replicaLossCot_eq`). The proof carries that as its invariant — each replica family
  equals `R ×` the shards of the global cotangent — down the chain: sharding (§2 of the file,
  `bnSyncInB_shard` the one real case) plus homogeneity (§1: `bn_grad_input_smul`,
  `maxPool3s2BackFlat_smul`, `HasVJP.backward_smul` for the conv and head links — no `0 < ε`
  needed, the three-term formula is linear in `dy` as written). At each parameter the collective's
  `1/R` (`den_allReduceMeanF_*_shard`, three new: strided conv, dense weight, dense bias) cancels
  the `R`. ⚠ The conv-bias nodes are not tied: the DP artifacts run `convBias := false` and do
  not emit them.

  ⚠ Not claimed: the replicas' saved activations enter as shards of the single-device forward's
  (`batchShard r (r34Pre_k (R*N) w X)`) — the forward twin is what says the replica graph
  computes exactly those, the same split T2/T3 make on one device. A better-conditioned operating
  point for the gate's gradient columns (trained weights) is optional and separate.

### 3.3 MobileNetV2, then EfficientNet-B0

The two BN-dense nets with a pair already in the book (§6, §7).

✅ **LANDED 2026-09-21 — render, gates and proofs, both nets.**

* **The three sites are shared now.** `bnFwdSite` / `bnBackSite` / `bnGammaSite` moved out of
  `ResNet34RenderB.lean` into `Codegen/SyncBnSites.lean` (taking `hh` and `ww` separately); R34
  re-renders byte-identical. MobileNetV2: 13 forward, 10 backward, 10 γ sites through them, the
  packed statistics threaded on `MBFwdB` / `MNV2FwdRecB`. B0: its mode-switched `bnSiteB` calls
  `bnFwdSite` at `.train` (`.eval` untouched), `bnG` calls `bnGammaSite` on the un-fused path, and
  a `bnSt` list rides beside `bns` so the handed-back statistics read the right packed vector.
  Tags are the γ parameter's name without `%` (`b{k}eg`, `sg`, `hg`, …). Only the `*dp*`
  artifacts moved — 6 MobileNetV2, 10 B0 (all_reduces: MNv2 158 → 314 = 158 + 52 × 3; B0
  213 → 360 = 213 + 49 × 3); every single-device and eval artifact is byte-identical. The DP
  headers state the step identity and name the theorems; the bf16 DP artifacts (R34 `momdp64bf16`
  included) add that the theorems are stated at the f32 nodes.
* **Gates.** `mobilenetv2-syncbn-check` / `efficientnet-syncbn-check`, one shared runner
  (`LeanMlir/SyncBnCheck.lean`, R34's columns; the `2b` single-device step and both one-replica
  sync graphs rendered at run time — ⚠ B0's loss divisor is a string, passed as `"{B}.0"`):

  | net | stats FORMULATION | stats DUPLICATED | stats TEST (split) | stats CONTROL | m' TEST / SENSITIVITY |
  |---|---|---|---|---|---|
  | ResNet-34 | 0 | 0 | 1.7e-4 | 3.4e-3 | 0.15 / 0.22 |
  | MobileNetV2 | 0 | 0 | 1.0–1.5e-3 | 1.7e-2 | 0.24 / 0.27 |
  | EfficientNet-B0 | 0 | 0 | 3.4–3.7e-4 | 8.8e-3 | 6.3e-3 / 1.1e-2 |

  ⚠ **MobileNetV2's split error sits at R34's 1e-3 bound**, and XLA's GPU reductions are not
  bit-deterministic (the same seeds gave 0.97e-3, 1.05e-3, 1.14e-3, 1.49e-3 on four runs). The
  per-layer table (`SYNCBN_VERBOSE=1`) says it is rounding compounding with depth: the first six
  BN layers are split-EXACT, and the variance error grows smoothly to 1.4e-3 at the 52nd layer.
  (The per-layer MEAN column shows relative errors up to ~37 on every expand-BN input — noise over
  noise: a bias-free 1×1 conv of a β = 0 BatchNorm output has batch mean exactly 0 at init.) So
  MobileNetV2's bound is 3e-3, written beside its config with that evidence, and every net now also
  has a FIRST-LAYER criterion (split error ≤ 1e-5 on the first BN layer's statistics, measured 0
  on both) — the place a wrong exchange would show at full size before any depth compounds.
* ⚠ **`mobilenetv2-dp-check` / `efficientnet-dp-check` needed their gradient bound moved 1e-4 →
  1e-2.** They compare the DP step on a DUPLICATED batch with the committed single-device step.
  The forward is still bit-exact (all 104 / 98 statistics), but the DP render's backward is now the
  sync-BN graph and the single-device one the two-pass graph — one function, two f32
  arithmetics — and `m` differs by ~1.1e-3 norm-rel even there (measured; against the one-replica
  SYNC graph it is still ~1e-3, so re-pointing the reference would not have saved 1e-4). A wrong
  collective (sum, not mean) still moves `m` by ~1. Both pass at 8.4–8.6e-4.

▶ **Proofs.** MobileNetV2 LANDED: `Nets/MobileNet/MobileNetV2SyncB.lean`
(`StableHLO.mobilenetv2FwdGraphSync_full_shard`) and `MobileNetV2SyncStepTieB.lean`
(`MobileNetV2SyncTieB.mnv2_net_syncTiedB` — all 158 emitted gradients; the 52 conv / depthwise /
project biases are not emitted at `convBias := false`). New pieces beyond R34's: relu6 shard +
mask homogeneity, depthwise and XLA-strided-depthwise input-VJPs, the three XLA-strided / depthwise
weight collectives, and `hasVJP3_backward_smul`. Both elaborate in ~3–4 s.

EfficientNet-B0 LANDED: `Nets/EfficientNet/EfficientNetSyncB.lean`
(`StableHLO.efficientnetFwdGraphSync_full_shard`) and `EfficientNetSyncStepTieG.lean`
(`EnetSyncTieG.efficientnet_net_syncTiedG` — all 213 emitted gradients; the 49 conv-bias
conjuncts of `efficientnet_net_tiedG` are not emitted at `convBias := false`). It carries that
tie's 50 `0 < ε` hypotheses, because B0's single-device chain threads each block's cotangent as
a certified VJP's `.backward`: the twin adds `xCotIn`/`rCotIn`/`sCotIn`/`nCotIn`/`hdCotIn_eq_vjp`
(the explicit chain IS the block VJP, by `HasVJP.backward_unique`) and `bnSyncInB_shard_bnBackB`.
⚠ **Scope, stated in each DP header:** the step twin is stated without drop-path and dropout and
at the f32 nodes, so it covers `efficientnet_adamdp`, `adamdp128`, `efficientnetin_adamdp64`,
`rmsdp64`, `emarmsdp64` exactly (EMA and the optimizers run after the all-reduced gradient); the
`*drop*` / `*do*` variants add per-example masks and the `*bf16` ones bf16 conv twins, both
outside the statement — the same holds for MobileNetV2's and R34's bf16 DP artifacts.
✅ The generic lemmas the two twins each carried are one module now,
`Nets/EfficientNet/MBConvSyncTieB.lean` (NEXT SESSION item 2).

### 3.3b The ImageNet-shape probe — 4×64, bf16

✅ **LANDED 2026-09-21, `23db65a2`.** `imagenet-syncbn-check <net> [f32]`
(`tests/TestImagenetSyncBnCheck.lean`) runs the shared runner on the DP artifact each book pair
trained from — R34 `resnet34in_momdp64bf16`, MNv2 `mobilenetv2in_rmsdp64bf16` — and B0's
`efficientnetin_rmsdp64bf16` (the pair's `emarmsdp64dropdobf16` adds EMA and host-fed masks, the
scope the twin leaves out too), against a 1×256 single-device step rendered at run time.

What the runner gained, all defaulted so the Imagenette gates' inputs and verdicts are unchanged
(re-run twice: B0 2.6–3.4e-4, MNv2 9.8–9.9e-4 on the statistics — last session 3.4–3.7e-4 and
0.97–1.49e-3):

* `replicas` — `R` shards, seeds `555 + 444·k`, labels offset `5·k` (so `R = 2` is the old xA/xB).
* `gradSlot` / `gradLabel` / `vZero` — heavy-ball keeps its velocity in `v` and passes `m`
  through, so R34's gradient column is `v' = g + wd·θ` with `v = 0` in.
* ⭐ `shardShift` — shard `k`'s pixels shifted by `0.5·k`. With iid shards the probe was weak at
  224²: R34's CONTROL was 2.7e-3 against a TEST of 1.0e-3, and Chan's `(μ_r − μ)²` term was
  ~1e-6, so a render that dropped it would have passed. Shifted, CONTROL is 0.72 / 0.074 / 0.40
  and the first BN layer is split-exact.
* REPEAT — the `Rb` graph run twice on the same input. 0 everywhere: one program is
  deterministic run to run.
* `formTol`, `dupTol` per config.

| net | prec | stats TEST | FORMULATION stats | DUPLICATED stats | CONTROL | DUP gradient norm-rel | FORMULATION gradient |
|---|---|---|---|---|---|---|---|
| R34 | f32 | 1.3e-4 | 0 | 0 | 0.72 | 7.0–7.2e-4 | 2.2e-4 |
| R34 | bf16 | 0.70–1.0e-3 | 0 | 0 | 0.72 | 1.27e-2 | 1.2–1.3e-2 |
| MNv2 | f32 | 3.6–4.8e-4 | 0 | 0 | 0.074 | 1.0e-3 | 6.4e-4 |
| MNv2 | bf16 | 4.3–5.2e-3 | 0 | 0 | 0.074 | 1.62–1.70e-2 | 1.4–1.9e-2 |
| B0 | f32 | 1.6–2.0e-4 | 0 | 0 | 0.40 | 6.8–7.0e-4 | 1.4e-3 |
| B0 | bf16 | 2.5–3.1e-3 | 1.4–2.0e-3 | 0 | 0.40 | 1.19–1.20e-2 | 7–9e-2 |

**Reading it.** The exchange is right at four replicas in both precisions: DUPLICATED statistics
are 0 (the collective composes exactly), the first BN layer is split-exact with O(1) Chan terms,
and TEST sits 14× (MNv2 bf16, the tightest) to 5,500× (R34 f32) under CONTROL. f32 passes at the
Imagenette bounds unchanged. **What bf16 moves is every comparison between two compiled
programs**, and FORMULATION measures that floor with no collective in either graph — one
function, one shape: gradient 1.2e-2 – 9e-2 in bf16 against 2.2e-4 – 1.4e-3 in f32. REPEAT = 0
puts the floor between programs, not between runs. Consistent with an f32 last-bit difference
becoming a bf16-ulp difference at the next f32→bf16 convert, then amplified by a random-init
backward whose SENSITIVITY (1e-4 input noise) moves the gradient by 0.4–0.8. Not isolated
further; the claim the gate needs is only that it is not the exchange, and the two sharp columns
settle that. bf16 also starts compounding earlier: MNv2's and B0's second BN layer already
splits at 1–4e-5, where f32 is exact through the first several.

▶ For §3.5 this says a bf16 re-run trains on the global-batch function to within bf16's own
rounding — the same rounding every bf16 single-device run has.

### 3.4 ResNet-50, MobileNetV4 — ✅ LANDED 2026-09-21 (`260df19d`)

§3.3's pattern, net by net: each renderer calls `SyncBnSites`, the gates are
`imagenet-syncbn-check` configs, the twins follow `ResNet34Sync*` (R50) and `MobileNetV2Sync*` +
`MBConvSyncTieB` (MNv4). Neither net has an Imagenette-scale DP render, so the ImageNet gate is
each one's only split check. MNv4 has no pair run yet, so its chapter never needs a caveat row.
Logs: `runs/2026-09-21-syncbn-r50-mnv4/`.

**Files** (`260df19d`): `Codegen/ResNet50RenderB.lean`, `Codegen/MobileNetV4RenderB.lean`,
17 `verified_mlir/` DP artifacts (15 `resnet50in*dp*`, 2 `mnv4in_adamdp64*`), the four twins,
`LeanMlir/SyncBnCheck.lean`, `tests/TestImagenetSyncBnCheck.lean`, `tests/TestShardCheck.lean`,
`tests/TestMnv4DpCheck.lean`, `tests/TestR50AccumShardTie.lean`, `tests/TestR50GradCheck.lean`,
`tests/TestDropShardCheck.lean`, `tests/r50_dp_render_tie.py`, `scripts/regen_verified_mlir.sh`,
`scripts/check_render_coverage.py`, `scripts/bf16_probe_4gpu.sh`, `scripts/gen_comparator_tier.py`
+ the three regenerated `tests/comparator/*Tier*`, `tests/AuditAxioms.lean`, `formalization.yaml`,
`lakefile.lean`, the two T2 docstrings (`ResNet50FullB`, `MobileNetV4FullB`),
`apps/imagenette/MainMobilenetV4Imagenet.lean`, and this doc.

**Step 0 — the gates that asserted the old identity. FIVE, not the three planned.** Each retired
or re-pointed, none loosened:

| gate | what broke | now |
|---|---|---|
| `shard-check` BN rows | `DP([A\|B]) = mean(single(A), single(B))` is what sync-BN's CONTROL requires to fail (`efficientnet` 0.21, `mobilenetv2` 1.60 vs 1e-4) | `efficientnet`, `mobilenetv2`, `…in`, `mnv4in` RETIRED; a retired slug exits 2 naming its replacement (`*-syncbn-check` / `imagenet-syncbn-check <net>`). ConvNeXt, ViT stay. |
| `tests/r50_dp_render_tie.py` | the DP render is no longer the two-pass render + gradient all-reduces (11,289 differing lines) | 1-replica side is the `forceSync` graph `ResNet50RenderB` writes to `.lake/build/r50sync/`; all 5 pairs IDENTICAL, 320 values routed (161 gradients + 159 BN statistics) |
| `r50-gradcheck`'s transfer | it certified the two-pass render, which the DP render no longer contains | **option (a)**: `R50_GC_PATH=.lake/build/r50sync` gradchecks the sync graph directly — tier 1 5.9e-5, tier 2 0.166 vs a doubled-gradient control 6.0× off (two-pass: 7.5e-5 / 0.168 / 5.9×) |
| `r50-accum-shard-tie` | against the committed (sync) DP render: m′ rel 1.07 | peer rendered at run time with `noSync := true` (per-replica BN, never committed): `acc4x64` θ′/m′/v′ ≤ 1e-6 vs control 1.0e-2; the 160 `lambacc4x64bce` / `lambdp64bce` arm too |
| ⛔ `drop-shard-check` (not in the plan) | its anti-vacuity half required the BN statistics to MOVE under the mask swap — they were replica-0-local; now all-reduced, so 0/42,016 moved | the statistics joined ① (must be bit-identical: 42,016/42,016 are); `%loss` alone is the movement witness, as on the LayerNorm nets. `DROP_FAULT=replicate` still fires ① and ①b. |
| ⛔ `regen_verified_mlir.sh check` (not in the plan) | forward ⊂ train-step pairs took the DP variant; the sync DP forward is not the `_fwd` text — MNv2's and B0's pairs RED since `cad811ac`, R50's and MNv4's with this section | the five BN forwards pair with single-device peers; 20 paired, 0 unaccounted. The DP forward's tie is the forward twins + FORMULATION. |

`r34-dp-shard` re-run: passes (TEST 3.6e-2, CONTROL fires).

**Renders.** Both byte-identical at `replicas ≤ 1` — every single-device artifact and forward of
either net unchanged (checked: exactly the 17 `*dp*` artifacts moved). R50: 12 forward, 12
backward and 12 γ sites through `SyncBnSites`, statistics off the packed vectors; each of the 15
DP artifacts gains 53 × 3 = 159 collectives (161 → 320). Two trailing knobs, neither ever
committed: `forceSync` (the one-replica sync graph) and `noSync` (a DP graph with per-replica BN,
for `r50-accum-shard-tie`). MNv4: `mnv4Bn`'s `.train` arm is `bnFwdSite` (`.eval` untouched), 16
backward and 16 γ sites; both DP artifacts gain 77 × 3 = 231 (233 → 464). DP headers state the
step identity and name the theorems; the `acc` ones add "per micro-step", the `drop` / `bf16`
ones the twins' scope. ⚠ Accumulation: the gradient collective sits before the accumulator, so
each micro-step is one sync-BN step over `R × micro` rows — BN groups 256 at A3's
`lambaccdp8x64…` (reference 512) and 512 at A2/A1's `…accdp4x128…` (equal). The book's `BN group`
rows (`:5935`, `:6089`) take those at §3.5.

**Gates** (`imagenet-syncbn-check`; R50 at 4×32 against 1×128 with its DP step rendered at run
time — the runner's new `dpPath := ""` / `renderDp` — because the 1×256 reference peaks at 94–95 %
of the raised arena; `resnet50bce` is LAMB + BCE at 160²). Two runs each:

| config | stats TEST | FORM / DUP stats | first layer | CONTROL | stats SENSITIVITY | DUP grad norm-rel |
|---|---|---|---|---|---|---|
| R50 bf16 | 6.7e-3, 8.6e-3 | 0 / 0 | 0 | 0.38 | 1.4e-2 | 2.05–2.07e-2 |
| R50 f32 | 1.7e-3, 2.1e-3 | 0 / 0 | 0 | 0.38 | 1.4–1.5e-3 | 1.0e-3 |
| R50 bce bf16 | 1.08e-2, 1.09e-2, 1.22e-2 | 0 / 0 | 0 | 0.37 | 1.5e-2 | 1.69–1.72e-2 |
| R50 bce f32 | 2.2e-3 | 0 / 0 | 0 | 0.37 | 1.7e-3 | 1.0e-3 |
| MNv4 bf16 | 6.9e-3, 1.16e-2 | 0 / 0 | 0 | 0.076 | 1.6e-2 | 1.83–1.85e-2 |
| MNv4 f32 | 1.2e-3, 1.3e-3 | 0 / 0 | 0 | 0.077 | 1.3e-3 | 1.2e-3 |

⚠ **On these two deeper nets (53 and 77 BN layers) TEST sits at the forward's own conditioning**:
the statistics SENSITIVITY column (the two-pass graph against itself on `x` + 1e-4 noise) is the
same size, and in f32 R50's TEST slightly exceeds it. So the §3.3 placeholder bounds (1e-2 bf16 /
3e-3 f32) were too tight — the BCE run read 1.08e-2 — and the bounds are ≈ 2× SENSITIVITY: 3e-2
bf16 for both, 5e-3 / 3e-3 f32 (R50 / MNv4) — approved 2026-09-21 (NEXT SESSION's gate policy). The
per-layer tables (`SYNCBN_VERBOSE=1`) show the variance error growing smoothly with depth from
split-exact first layers — rounding, not a wrong exchange; the sharp columns carry the exchange.
`mnv4-dp-check`: forward BIT-EXACT (67,904/67,904) in both arms; gradient 1.45–1.65e-3 f32,
2.06–2.21e-2 bf16 → bounds 1e-2 / 5e-2; its sum-not-mean control now breaks the forward too
(norm-rel 80), because the BN collectives also sum.

**Twins** (two parallel subagents, ~13 and ~25 min; every file elaborates in 2–4 s with no
heartbeat override — the R34 and MNv2 twins and the R50 and MNv4 single-device T3s all carry
`maxHeartbeats 1600000`, these four don't):

| file | lines | capstone |
|---|---|---|
| `Nets/ResNet/ResNet50SyncB.lean` | 367 | `StableHLO.resnet50FwdGraphSync_full_shard` — `q` stays a binder (both resolutions) |
| `Nets/ResNet/ResNet50SyncStepTieB.lean` | 1044 | `ResNet50SyncTieB.r50_net_syncTiedB` — all 161 gradients; corollaries `_smoothedCE`, `_bce` (+ `replicaBceLossCot_eq`, BCE's peer of `replicaLossCot_eq`) |
| `Nets/MobileNet/MobileNetV4SyncB.lean` | 693 | `StableHLO.mnv4FwdGraphSync_full_shard` |
| `Nets/MobileNet/MobileNetV4SyncStepTieB.lean` | 1455 | `MobileNetV4SyncTieB.mnv4_net_syncTiedB` — all 233 gradients; corollary `_smoothedCE` |

Both T3s bind the loss cotangent, so each capstone takes the scaled-shard hypothesis
`∀ r, gs r = batchShard r (fun i => R * G i)` and concludes a named predicate
(`r50NetSyncTiedB` / `mnv4NetSyncTiedB`) that the loss corollaries restate with the hypothesis
discharged. Both take `hN : 0 < N` (R50 also `hq : 0 < q`), as R34's twins do. Scope as before:
f32 nodes, no drop-path, no conv-bias nodes (not emitted). Conv-M has no post-strided UIB row, so
that backward form has no chain in T3 and none in the twin.

**Bookkeeping.** `Certs` roots (+4), `AuditAxioms` (+55 directives; 1,560/1,560, no `sorryAx`),
`formalization.yaml` (+4 rows) and `gen_comparator_tier.py` (34 theorems, `--check` green).
Gates at the end of the session: `lake build Certs` 4,031 jobs, `lake build`, `lake build Apps`,
`docstring-checkrefs`, `check_audit_coverage`, `check_target_names`, `check_render_coverage`,
`regen_verified_mlir.sh check`, `r50_dp_render_tie.py`, and every GPU gate above.

### 3.5 Re-running the pairs

▶ **The queue.** Each job trains the committed DP artifact, which is sync-BN since §3.2–3.4; the
verified side only. Wall-clocks are the per-replica renders' measured ones. Sync-BN adds three
small collectives per BN layer: +2–4 % (R34), +5–10 % (MNv2), +3–7 % (B0) by §2e's costing,
unmeasured. The user's ceiling for a sanctioned run is ~40 h, and anything longer is their call
explicitly (run it through `scripts/supervise.sh`).

| order | net | job | variant | verified side | what it retires |
|---|---|---|---|---|---|
| 1 | ResNet-34 | `scripts/jobs/r34-default-bf16-4gpu.conf` | `momdp64bf16` | ~22 h | `content.tex:5671` `BN statistic group` row |
| 2 | MobileNetV2 | `scripts/jobs/mnv2-default-4gpu.conf` | `rmsdp64bf16` | ~51 h | `:7059` row + §6's `[TODO: global BN.]` |
| 3 | EfficientNet-B0 | `scripts/jobs/enet-default-4gpu.conf` | `emarmsdp64dropdobf16` | ~73 h | `:8266` row |
| — | ResNet-50 | `r50-a3-wxclip-bf16-4gpu.conf` / `r50-2018-bf16-4gpu.conf` | `lambaccdp8x64wxclipbcebf16` / `momdp64bf16` | not costed here | `:5935`, `:6089` `BN group` rows (A3's group is now 256 against the reference's 512) |
| — | MobileNetV4 | `scripts/jobs/mnv4-default-4gpu.conf` | `adamdp64` | not costed here | none — its FIRST pair; no caveat row needed |

⚠⚠ **R50 A3 is not BN-matched by re-running its conf.** Sync-BN normalises over `R × micro`
rows per micro-step, so `lambaccdp8x64…` gives 256 against the reference's 512 (its `4 × 512`,
128 per GPU under GSPMD). A `lambaccdp4x128wxclipbcebf16` render at 160² matches both the BN
group and the reference's `k = 4`; A2/A1 already run 128 per replica at 224². Unrendered and
unprobed (memory and ms/step). Without it the `:5935` A3 column keeps a `BN group` caveat and
`:6089`'s "only half of it is reachable here" stays true. R34 and R50 2018 are matched at 256.

⚠ B0's job trains `emarmsdp64dropdobf16`. Its twin and `imagenet-syncbn-check` cover
`rmsdp64bf16`, the same net without EMA and the host-fed masks. The render is sync-BN either way.

These are now **scheduling** decisions, not whether-to: once a net's DP render normalises globally, its
committed pair number was produced by a function the tree no longer renders, and the chapter
either re-runs it or says so. §1's Imagenette probe was meant to predict whether a re-run will
MOVE the number and could not (closed 2026-09-21); what predicts it is the book's own epoch
tables, whose converged deltas are ≤ 0.10 — so expect re-runs that confirm rather than
overturn, and order the queue by cost.

⚠ A re-run is also the only thing that retires the `[TODO: global BN.]` sitting in §6's
MobileNetV2 section (`content.tex`, after the "Separating those would take a verified render that
all-reduces the batch statistics" paragraph) — that paragraph is a description of exactly this
plan, and it goes when the render lands.

### 3.6 bf16 — what sharding does to the bf16 nodes (R34, R50) — ✅ 2026-09-21

`Foundation/DataParallelSyncBf16.lean`, a leaf on `DataParallelSync` + `Float/Binary32Instance`.
The bf16 DP artifacts R34 and R50 train from swap six nodes for bf16 kinds; the kit states each
under sharding. Node to node throughout: there is no single-device bf16 chain to tie a whole-net
twin to, in this tier or the f32 one.

| kind | under sharding | lemma |
|---|---|---|
| `convBf16`, `convStridedBf16` (forward) | exact | `den_convBf16_shard`, `den_convStridedBf16_shard` (via `den_batchOp_shard_node`) |
| `convBackBatchedBf16`, `convStridedBackBatchedBf16` | exact | `den_convBackBatchedBf16_shard`, `den_convStridedBackBatchedBf16_shard` |
| `convWeightGradBBf16`, `convStridedWeightGradBBf16`, all-reduced | **rounded per replica** | `den_allReduceMeanF_convWeightGradBBf16_sub_global` + strided peer |

⭐⭐ **The weight gradient.** Its `den` is `rnd` of the f32 node at rounded operands
(`den_convWeightGradBBf16_eq_rnd`, `rfl`). With `S_r` replica `r`'s partial sum
(`convWGradShardSum`), the collective is `(1/R)·Σ_r rnd S_r` (`…_shard`) and the batch-`R·N`
node is `rnd (Σ_r S_r)` (`…_global_split`); `…_sub_global` states the difference exactly, and at
`rnd := id` it vanishes (`…_shard_id`), which is the f32 statement.

**The divisor step through `rnd`.** The `*_smul` lemmas carry a scale through the four bf16
backward kinds under `∀ x, rnd (s * x) = s * rnd x`, and `rndP_two_pow_mul` proves that for the
repo's rounding model at every power of two (`rndP_mul_four` at `R = 4`) — the unbounded-exponent
model, so bf16 overflow and subnormals are outside it.

Bookkeeping: `Certs` root, 22 `AuditAxioms` directives (1,582/1,582, no `sorryAx`, all on the
standard three), one `formalization.yaml` row + `gen_comparator_tier.py` (35 theorems). The five
R34/R50 bf16 DP headers name the kit; comment lines only, no op changed.

---

## 4. What "done" looks like

* ✅ The eight ops in the AST with `den`, emit, parser round-trip; P1, P2, P3's kit, P4 proved
  (2026-09-21). ✅ R34's DP render is sync-BN and `resnet34-syncbn-check` passes (2026-09-21).
  ✅ R34's DP twins: forward and whole step tied to the existing spec at `N := R·N` (2026-09-21).
  ✅ MobileNetV2 and EfficientNet-B0: sync-BN DP renders, `*-syncbn-check` gates, both twins (2026-09-21).
  ✅ ResNet-50 and MobileNetV4: the same, plus five per-replica-identity gates re-pointed or
  retired (2026-09-21, `260df19d`).
* ✅ Every BN net's DP render normalises over the global batch; its DP twin ties it to the existing
  spec at `N := R·N`; its `syncbn-check` passes on a split batch.
* The side-by-side tables' `BN statistic group` row reads `256 (global)` in both columns, the
  render headers and yaml 4k say the DP step is the batch-`R·N` step, and
  `dpMeanGrad_ne_globalBatchGrad` keeps its place as the statement of what the older runs did.
* Only then may a chapter say the remaining difference between the two columns is the lowerer.

## 5. Not this doc

* ε and momentum: not wrong. Both nets' specs run `runningBN := true`, the emitter EMAs the
  running buffers (`_bn`, momentum from `TrainConfig.bnMomentum`), B0 shadows them under EMA
  (`ema_bn`), and the verified driver EMAs the handed-back statistics at `bnMomentum 0.99`.
  timm 1.0.28's `tf_efficientnet_b0` carries `eps = 1e-3` with PyTorch's momentum 0.1 because a
  `tf_*` model ports what changes the function, not the optimisation; `mobilenetv2_100` and
  `efficientnet_b0` are at `1e-5` / 0.1. Nothing to plan.
* Ghost-BN on the reference (per-shard `jnp.mean`) is §1's measurement tool, not a fix: it would
  make the two columns agree by moving the reference off the recipe it was implemented to.
* ConvNeXt and ViT: LayerNorm, no batch statistics, unaffected.

## 6. Related

* `sec:r34_pjrt` and the R34 table (`content.tex:5671`); §6 / §7 tables (`:7059`, `:8266`).
* `Foundation/DataParallel.lean` — `dpMeanGrad_ne_globalBatchGrad`, the theorem P4 twins.
* `Foundation/DataParallelNode.lean` — `den_allReduceMeanF_eq_dpMean`, the collective's `den`.
* `runs/2026-09-12-enet-verified-350ep/RESULTS.md` §2 — the B0 pair's own note on the confound.

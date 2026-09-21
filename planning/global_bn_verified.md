# global_bn_verified.md — all-reducing the BatchNorm statistics in the verified renders

**Opened 2026-09-12**, out of §6's phase-4 half. Every verified ImageNet pair that has a
BatchNorm net carries the same unmeasured confound, and this is the plan to remove it.

⭐⭐ **Decided 2026-09-20: remove it, not bound it.** The verified path is to *be* the JAX side —
BatchNorm over the global batch, the same function the reference computes — and the pair re-runs
in §3.5 are accepted if they follow. The doc opened leaving "bound it cheaply and decide it is
not worth removing" on the table; that option is closed. §1 now sizes the re-runs rather than
gating the work.

---

## ▶ NEXT SESSION — start here (written 2026-09-21, end of day)

**State.** Everything through §3.2's render and gate is on `main`: the sync-BN kit (eight `SHlo`
ops, two-round Chan exchange), P1–P4 (`Foundation/DataParallelSync.lean`, `DataParallel.lean`),
ResNet-34's DP render swapped to sync-BN under `replicas > 1` (five `*dp*` artifacts re-rendered,
all single-device artifacts byte-identical), and `resnet34-syncbn-check` passing on two GPUs.
Read §2b's ⛔⛔ and §3.2's table before touching numerics — the one-round `E[x²]` exchange was
measured wrong and replaced; the gate's columns are how the kit is now read.

**Run the gate** (2 GPUs, XLA; ~30 s):

    lake build resnet34-syncbn-check
    unset HIP_VISIBLE_DEVICES
    CUDA_VISIBLE_DEVICES=0,1 PJRT_REPLICAS=2 \
      PJRT_PLUGIN=$PWD/.venv/lib/python3.12/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so \
      .lake/build/bin/resnet34-syncbn-check          # SYNCBN_VERBOSE=1 for per-parameter rows

**What is next, in order.**

1. **§3.2's remaining half — the R34 DP twins of T2 and T3.** The per-net chain walk that says
   the sync-DP render on replica `r` denotes `batchShard r` of `resnet34ForwardB_full (R*N) w X`
   and its gradient nodes the shard-`r` blocks of the batch-`R·N` backward. Every piece is in
   `DataParallelSync.lean`: `batchShard_batchMap` / `_batchMapAux` / `_map` / `_zipWith` for the
   non-BN ops, `den_bnSyncF_allReduce` / `den_bnSyncBack_allReduce` /
   `den_allReduceMeanF_bnSyncGammaGradB` for the BN cases (all take the induction hypothesis
   `hx : ∀ r, den (x r) = batchShard R N _ X r` and its `xv`/`dy` twins), `syncStats` for the
   statistics subgraph, and `HasVJP.backward_smul` for the divisor step (§3.1b's "the 1/R").
   ⚠ **Decide the index seam first.** The typed T2 graph (`ResNet34FullB.resnet34FwdGraphB_full`)
   feeds `bnBatchF` at the conv index `N·(c·h·w)`; `bnSyncF` sits at `N·(c·(h·w))` (matching
   `bnBatchTensor4`), and the AST has no index-cast node. Two options: (a) state the twin per
   block with `.operand` leaves, exactly T3's shape (`r34IdTiedB` — the tie composes at the ℝ
   level through `reassocB`), which needs no new AST machinery; (b) a `castIdx (h : n = m) :
   SHlo n → SHlo m := h ▸ ·` helper with `den_castIdx` by `subst`, so one typed sync graph can be
   written. (a) is the safer first cut. The render's `bnFwdSite` is the shape to mirror; its
   `tag` strings are `{p}g1` (→ `%arsum{p}g1mu`, `…var`) and `{p}g1dst` for the backward.
   Then `formalization.yaml` gets the twin beside `dpSyncGrad_eq_globalBatchGrad`, and the
   render header's claim ("this step IS the single-device step at the global batch") is tied
   rather than asserted.
2. **§3.3 MobileNetV2 and EfficientNet-B0, §3.4 ResNet-50 / MobileNetV4** — the same three
   helpers per renderer (`bnFwdSite`/`bnBackSite`/`bnGammaSite` are `private` in
   `ResNet34RenderB.lean`; lift them to `StableHLO.lean` or a shared codegen module first rather
   than copying), plus a `<net>-syncbn-check` each (the R34 harness parameterised by net, as
   `shard-check` is). ⚠ MNv2/B0 have `bnBatchLABack` sites too — check which BN backward each
   renderer uses before assuming the R34 pattern.
3. **§3.5 re-runs**, then the book rows (`content.tex:5671`, `:7059`, `:8266` — `BN statistic
   group … 64 (per replica)`, and the two `[TODO: global BN.]`). ⚠ Until a re-run, the committed
   pair numbers were produced by the OLD per-replica renders; the rows stay as they are.

**Gotchas that cost time this session.**
* Editing `StableHLO.lean` is a 6-minute rebuild plus the corpus; batch every op change.
* `simp` cannot rewrite inside `conv2d_weight_grad_has_vjp b x` (dependent position) — `rw`.
* `intro m2 h` inside a proof shadows the spatial `h`; name hypotheses `hm2`.
* Plain `simp` turns `Fin.natAdd` into `Fin.addNat` before `Fin.append_right` can fire — `simp only`.
* The REORDER probe (swap the two batch halves) is exactly commutative and measures nothing;
  the SENSITIVITY probe (perturb `x` by 1e-4) is the yardstick for any gradient comparison at a
  random-init operating point — there, no two f32 implementations agree on `m'` better than ~1e-2.
* XLA rounds a 32-row and a 64-row reduction differently (~1e-6/layer); 36 layers compound that
  to ~2e-4 in the statistics. That is the floor of the split identity, and it is not the render.
* A new `lean_exe` needs nothing but the `lakefile.lean` entry (`check_target_names.sh` lints
  references, `check_audit_coverage.py` the audit's imports); a new Foundation module needs the
  `Certs` root and the `AuditAxioms` import.

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
| ResNet-50 | BN, per-replica 64 | global 512 | yes | ✅ `BN group` column; called "the largest known difference between the two paths, and *untested*" |
| MobileNetV2 | BN, per-replica 64 | global 256 | yes | ✅ §6, 2026-09-12 |
| MobileNetV4 | BN, per-replica | global | yes | ▶ not run yet — will need the row |
| EfficientNet-B0 | BN, per-replica | global | yes | ▶ not run yet — BN-dense, needs it most |
| ConvNeXt-T | LayerNorm, no `bnBatchF` | no batch stats | **no** | n/a |
| ViT-Ti | LayerNorm, no `bnBatchF` | no batch stats | **no** | n/a |

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

### 2b. The four new ops

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
| `bnBatchStatsB` | `SHlo (N*(oc*(h*w))) → SHlo (oc+oc)` | 1 | `.batched` |
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
`den (bnBatchStatsB e) = Fin.append (fun c => bnMean …) (fun c => bnMeanSq …)` is the whole
definition and P1 reads its halves off the two `append` lemmas. ⚠ The 2026-09-18 Mathlib-reuse
audit found every lemma this tree hand-rolled was already in Mathlib; this is that lesson applied
before the fact rather than after.

* `bnBatchStatsB` replaces the first pass's `bnBatchMeanSqB` and subsumes `bnBatchMeanB`'s job at
  the sync sites: `den` is `μ` in the low `oc` slots and `E[x²]` in the high ones, both the
  `bnMean`/`bnMeanSq` of `bnchwFwd`, so it is the same statistic `bnBatchTensor4` normalises by.
  Still the second moment, never the variance — `var_g = E[x²]_g − μ_g²`, and the variance of a
  union is not the mean of the variances.
* `bnSyncF`'s `x` is the graph operand (it is the activation flowing through); `γ β` are host
  literals. `den` is `γ·(x − μ)·(m2 − μ² + ε)^{-1/2} + β` per channel, reading `μ`, `m2` out of
  the packed operand.
* ⭐ **`bnSyncDyStatsB` passes `μ`, `m2` THROUGH into its result**, so its output is
  `[μ ‖ m2 ‖ mdy ‖ mdyx]` and ONE all-reduce carries all four into the backward. Re-averaging an
  already-global quantity over the replicas is the identity, so the pass-through is free
  mathematically and buys the collective. `x` is a host literal, as in `bnBatchBack`.
* `bnSyncBack` then needs only `dy` and that one `[4·oc]` vector.

A BN layer on replica `r`, with `x : Fin R → SHlo n` the replica family of its input, renders as

    st  := allReduceMeanF R _ "{g}st"  [] (fun r => bnBatchStatsB (x r))        -- [μ ‖ m2]
    y_r := bnSyncF … (x r) st

and its backward, with `dy` the family of output cotangents, as

    dst  := allReduceMeanF R _ "{g}dst" [] (fun r => bnSyncDyStatsB … (dy r) st)  -- [μ‖m2‖mdy‖mdyx]
    dx_r := bnSyncBack … (dy r) dst

⭐ **Three collectives per BN layer per step** (μ, then Chan's σ², then the backward's packed
four) — the first-pass costing was two, before the one-round `E[x²]` exchange was measured to
drift; the payload is still one small vector each (`[4·oc]` ≤ 8,192 floats, 32 KB).

This stays inside the SPMD convention `allReduceMeanF` already uses (one skeleton, `skel` reads
replica 0; each replica's own activation is the `r`-th member). At `R = 1` every `allReduceMeanF`
emits nothing and threads its operand, so the single-device render is `bnSyncF x (bnBatchStatsB x)`,
which denotes `bnBatchF x`: the `R = 1` artifacts need not move at all, and 2e says whether to
let them.

The running statistics the step hands back (`bnBatchMeanB`/`bnBatchVarB` passthrough slots,
`VerifiedTrain.lean:108`) become `μ` and `m2 − μ²`: global, exactly what the reference's `_bn`
EMAs. Today the driver reads replica 0's shard-of-64 statistics, so this also aligns the eval
running stats, not only the training forward.

⚠ Adding a `batched`/`batched2`-riding `SHlo` op is **5 sites, all in `StableHLO.lean`** —
inductive, the `den` arm, the `den_*` simp theorem, the `skel` arm, the emit arm. `parseStack`
dispatches on the tag generically (`StableHLOParse.lean:183`), so the round-trip needs nothing;
the emit arms are a `match tag, info` case each (8624). ⛔ Still gate with `lake build Certs` — a
bare `lake build` misses it. Editing `StableHLO.lean` rebuilds the corpus (~6 min for the file,
then every module below it), so: four ops, one edit session, all four at once.

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

The four ops (2b), P1/P2 (2c) in `Foundation/PerChannelBN.lean` beside `bnBatchTensor4`, P4 in
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

▶ **All four ops landed 2026-09-21**, each 5 sites, parser untouched:
`bnBatchStatsB : SHlo (N*(oc*(h*w))) → SHlo (oc+oc)`,
`bnSyncF : … → SHlo (oc+oc) → …`,
`bnSyncDyStatsB : … → SHlo (oc+oc) → SHlo (oc+oc+(oc+oc))`,
`bnSyncBack : … → SHlo (oc+oc+(oc+oc)) → …`.

⭐⭐ **And the DROP-IN is now proved on actual graph nodes, both directions:**

    StableHLO.den_bnSyncF_allReduce_R1     -- bnSyncF ∘ allReduceMeanF 1 ∘ bnBatchStatsB
                                           --   = bnBatchTensor4
    StableHLO.den_bnSyncBack_allReduce_R1  -- bnSyncBack ∘ allReduceMeanF 1 ∘ bnSyncDyStatsB
                                           --   ∘ allReduceMeanF 1 ∘ bnBatchStatsB
                                           --   = bnBatchTensor4_grad_input

These are the graphs a sync render actually emits. At `R = 1` both collectives collapse to their
single operand and what is left is exactly what today's `bnBatchF`/`bnBatchBack` renders denote.
⇒ **the `R = 1` artifacts need not move, and `R > 1` is now purely a question about how shard
statistics compose** (`bnMean_shard`/`bnMeanSq_shard`) — BatchNorm itself is already accounted for.

✅ **`StableHLO.roundtrip` covers the new ops for free.** It is `∀ {k} (a : SHlo k), parse (toToks
(skel a)) = some (skel a)` — universally quantified, and it still compiles, because all four ride
`.batched`/`.batched2`.

⚠ For 3.2: `bnSyncF` is indexed `N*(oc*(h*w))` (matching `bnBatchTensor4`/`bnBatchMeanB`), where
`bnBatchF` is at `N*(oc*h*w)` — `bnBatchLA` is `bnBatchTensor4` conjugated by
`Fin.cast (Nat.mul_assoc oc h w)`. Not a blocker: the renderers build a fresh `.operand` per node
and `ResNet34RenderB` already carries BOTH zero-vectors (`zin` at `c*hh*ww`, `zbn` at
`c*(hh*ww)`) for exactly this.

▶ ⭐⭐⭐ **P1 AND P2 LANDED 2026-09-21.** `Certs` 4,018 exit 0; audit 1,392/1,392, no `sorryAx`.

    bnSyncTensor4_shard_eq_global             -- P1
    bnSyncTensor4_grad_input_shard_eq_global  -- P2

Handed the all-reduced statistics — each literally `(1/R)·Σ_r'` of a per-replica quantity, i.e.
what `allReduceMeanF` of `bnBatchStatsB` (then of `bnSyncDyStatsB`) denotes — replica `r`'s sync
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

▶ **Increment 1 of 4 — the ops.**
* `bnMeanSq` + `bnVar_eq_bnMeanSq_sub_sq : bnVar n x = bnMeanSq n x - bnMean n x * bnMean n x`
  (`Architectures/BatchNorm.lean`, beside `bnVar`). The identity the whole design rests on: it is
  why replicas exchange the second moment and each recovers `σ²` locally.
* `bnBatchStatsB {N oc h w} : SHlo (N*(oc*(h*w))) → SHlo (oc+oc)`, all 5 sites. `den` is
  `Fin.append` of the two statistics; the emit is `bnBatchMean`'s text verbatim for μ, the same
  reduction over `x·x` for the second moment, and a `stablehlo.concatenate` so one collective
  carries both.
* Gates: `lake build Certs` 4,018 jobs exit 0, no warnings. `tests/AuditAxioms.lean` clean — the
  new theorem on the standard 3 axioms, 1,375 verdicts for 1,375 directives, no `sorryAx`.
* ✅ **Both predictions held**: `StableHLO.lean` rebuilt in **374 s**, and the **parser needed zero
  changes** — `.batched` is generic, so the remaining three ops are the same 5-site shape.
* ⚠ NOT yet checked: the emitted MLIR is not numerically verified, because no renderer uses the
  op. That arrives with 3.2's R34 swap and `resnet34-syncbn-check`.

Priced 2026-09-20: **20 sites, all in `StableHLO.lean`** — 5 per op (inductive, `den` arm, `den_*`
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
  render emits (`bnSyncF` fed by `allReduceMeanF` over the replicas' `bnBatchStatsB`; `bnSyncBack`
  fed by the outer collective over their `bnSyncDyStatsB`, each fed by the inner one), under the
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

▶ **Then §3.2.** ⚠ The emitted MLIR of all five ops is still unverified numerically — nothing
renders them. That gate is §3.2's R34 swap plus `resnet34-syncbn-check`, the first thing that can
catch a wrong `stablehlo.slice` bound or a bad `concatenate`.

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
* ▶ **Still to do here:** the DP twins of `ResNet34FullB` (T2) and `ResNet34BackCertifiedTieB`
  (T3) — the per-net chain walk with `DataParallelSync.lean`'s BN cases (`den_bnSyncF_allReduce`,
  `den_bnSyncBack_allReduce`, `den_allReduceMeanF_bnSyncGammaGradB`) and `batchShard_batchMap`
  for every other op. ⚠ The typed T2 graph feeds `bnBatchF` at the conv index `N·(c·h·w)`;
  `bnSyncF` sits at `N·(c·(h·w))` and the AST has no index-cast node, so the twin is stated
  per block with operand leaves (T3's shape) or with a `castIdx` helper on the AST value —
  decide at the start of that session. A better-conditioned operating point for the gate's
  gradient columns (trained weights) is optional and separate.

### 3.3 MobileNetV2, then EfficientNet-B0

The two BN-dense nets with a pair already in the book (§6, §7). 14 and 5 emit sites
(`MobileNetV2RenderB.lean`, `EfficientNetRender.lean`); DP twins; gates as 3.2. B0's
`efficientnet-dp-check` already checks 98 handed-back statistics bit-exact under a duplicated
batch — its sync twin checks them under a split batch.

### 3.4 ResNet-50, MobileNetV4

R50: 14 sites, DP twins, gate. MNv4: 1 emit site (its renderer threads BN through one
function), the largest twins (924 + 658), and no pair run yet — land this **before** its first
pair so the chapter never needs the caveat row.

### 3.5 Re-running the pairs

Per net, after 3.2–3.4: R34 ~22 h, MNv2 ~51 h, B0 ~73 h on the verified side. These are now
**scheduling** decisions, not whether-to: once a net's DP render normalises globally, its
committed pair number was produced by a function the tree no longer renders, and the chapter
either re-runs it or says so. §1 is the cheap predictor of whether a re-run will MOVE the number
— which orders the queue and sets expectations — and the book's own epoch tables predict the
converged deltas are small, so expect re-runs that confirm rather than overturn.

⚠ A re-run is also the only thing that retires the `[TODO: global BN.]` sitting in §6's
MobileNetV2 section (`content.tex`, after the "Separating those would take a verified render that
all-reduces the batch statistics" paragraph) — that paragraph is a description of exactly this
plan, and it goes when the render lands.

---

## 4. What "done" looks like

* ✅ The eight ops in the AST with `den`, emit, parser round-trip; P1, P2, P3's kit, P4 proved
  (2026-09-21). ✅ R34's DP render is sync-BN and `resnet34-syncbn-check` passes (2026-09-21).
* Every BN net's DP render normalises over the global batch; its DP twin ties it to the existing
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

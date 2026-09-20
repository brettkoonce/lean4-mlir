# global_bn_verified.md — all-reducing the BatchNorm statistics in the verified renders

**Opened 2026-09-12**, out of §6's phase-4 half. Every verified ImageNet pair that has a
BatchNorm net carries the same unmeasured confound, and this is the plan to remove it — or to
bound it cheaply and decide it is not worth removing.

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

## 1. ⭐ DO THIS FIRST — bound the effect without touching a single proof

The question the confound raises is "how much of the verified-vs-reference delta is BN grouping?"
That can be answered entirely on the JAX side, with **zero** proof work:

▶ Take the reference trainer, keep everything identical, and make BN normalize **per shard**
  (group 64) instead of globally — `shard_map` over the batch axis, or an explicit reshape to
  `[R, N/R, …]` with the mean taken over the inner axis only. Train. Compare against the same
  reference at global BN.

**reference@BN64 − reference@BN256 IS the confound, measured**, on the same lowerer, same seed
discipline, same everything. Whatever is left over in the verified-vs-reference delta is not BN.

⚠ This is what §5's `sec:r34_pjrt` effectively did for ResNet-34 to get its $0.02$ figure. It is
cheap, it is a JAX-only diff, and it decides whether the **pair re-runs** (§3.5) are worth
repeating — if the effect is hundredths on the BN-dense nets too, the existing numbers stand.
The render work (§2) does not wait on it: its case is that the DP step provably is not the
batch-`R·N` step today (`dpMeanGrad_ne_globalBatchGrad`), whatever the accuracy delta.

⚠ Do it at short schedule first (Imagenette, or ~30 ImageNet epochs) to get the sign and the order
before spending a 50-hour run on it.

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

Global statistics over `R` equal shards are means of per-shard means: `μ_g = (1/R) Σ_r μ_r`,
`E[x²]_g = (1/R) Σ_r E[x²]_r`, `var_g = E[x²]_g − μ_g²`. (Not `(1/R) Σ_r var_r` — the variance of
the union is not the mean of the variances; reduce the second moment, not the variance.)

| op | type | `den` | role |
|---|---|---|---|
| `bnBatchMeanSqB` | `SHlo (N*(oc*(h*w))) → SHlo oc` | per-channel mean of `x²` over `N·h·w` | the second moment; sibling of `bnBatchMeanB` |
| `bnSyncF gName bName epsStr ε γ β` | `SHlo n → SHlo oc → SHlo oc → SHlo n` | `γ·(x − μ)·(m2 − μ² + ε)^{-1/2} + β` per channel, with `μ`, `m2` the two `SHlo oc` arguments | the forward, statistics handed in |
| `bnSyncDyXhatMeanB` | `SHlo n → SHlo oc → SHlo oc → SHlo n → SHlo oc` | per-channel mean over `N·h·w` of `dy · xhat`, `xhat` from the handed-in `μ`, `m2` | the backward's second reduction (its first is `bnBatchMeanB dy`) |
| `bnSyncBack gName xName epsStr ε γ x` | `SHlo oc → SHlo oc → SHlo oc → SHlo oc → SHlo n → SHlo n` | `γ·istd·(dy − mdy − xhat·mdyx)`, with `μ`, `m2`, `mdy`, `mdyx` handed in | the backward, both reductions handed in |

A BN layer on replica `r`, with `x : Fin R → SHlo n` the replica family of its input, then renders
as

    μ   := allReduceMeanF R _ "{g}mu"  [] (fun r => bnBatchMeanB   (x r))
    m2  := allReduceMeanF R _ "{g}m2"  [] (fun r => bnBatchMeanSqB (x r))
    y_r := bnSyncF … (x r) μ m2

and its backward, with `dy` the family of output cotangents, as

    mdy  := allReduceMeanF R _ "{g}mdy"  [] (fun r => bnBatchMeanB (dy r))
    mdyx := allReduceMeanF R _ "{g}mdyx" [] (fun r => bnSyncDyXhatMeanB (x r) μ m2 (dy r))
    dx_r := bnSyncBack … μ m2 mdy mdyx (dy r)

This stays inside the SPMD convention `allReduceMeanF` already uses (one skeleton, `skel` reads
replica 0; each replica's own activation is the `r`-th member). At `R = 1` every `allReduceMeanF`
emits nothing and threads its operand, so the single-device render is `bnSyncF x (mean x) (meanSq x)`,
which denotes `bnBatchF x`: the `R = 1` artifacts need not move at all, and 2e says whether to
let them.

The running statistics the step hands back (`bnBatchMeanB`/`bnBatchVarB` passthrough slots,
`VerifiedTrain.lean:108`) become `μ` and `m2 − μ²`: global, exactly what the reference's `_bn`
EMAs. Today the driver reads replica 0's shard-of-64 statistics, so this also aligns the eval
running stats, not only the training forward.

⚠ Adding an `SHlo` op is ~10 sites across two files — inductive, `den`, `skel`/`toToks`, the
parser round-trip (⛔ under `Certs`; a bare `lake build` misses it), `pretty`/`emitTok`, shape
lemmas — and editing `StableHLO.lean` rebuilds the whole corpus (~6 min for the file, then every
module below it). Four ops, one edit session, all four at once.

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

Collectives per step: two per BN layer forward (`μ`, `E[x²]`), two per layer backward
(`mdy`, `mdyx`), each a `[oc]` vector (≤ 2,048 floats, 8 KB):

| net | BN layers | collectives / step | verified ms/step today | at 30–60 µs each |
|---|---|---|---|---|
| ResNet-34 | 36 | 144 | ~175 (14.6 min / 5,004 steps) | +4–9 ms, 2–5 % |
| MobileNetV2 | 52 | 208 | 97 | +6–12 ms, 6–13 % |
| EfficientNet-B0 | 49 | 196 | 134 | +6–12 ms, 4–9 % |
| ResNet-50 | 53 | 212 | — | +6–13 ms |
| MobileNetV4 | 77 | 308 | — | +9–18 ms |

XLA's all-reduce combiner cannot merge them: each layer's forward depends on the previous layer's
normalised output, so the collectives are sequential by construction. The reference pays the same
collectives — GSPMD inserts one per BN for `jnp.mean(axis=(0,2,3))` on a sharded array — so the
pairs stay like-for-like, and the reference's 79 / ~92 ms/step already include them. If the
overhead matters, one concat op would halve the count (reduce `[μ ‖ E[x²]]` as one `[2·oc]`
vector); not in the first cut.

The `R = 1` artifacts: leaving them on `bnBatchF` keeps 0 bytes moving in `verified_mlir/` for
the single-device renders and keeps every existing tie untouched; the DP renders are where the
function changes. Decide per net at render time; the default is to touch only the `*dp*` files.

---

## 3. Work packages

### 3.1 The kit (once)

The four ops (2b), P1/P2 (2c) in `Foundation/PerChannelBN.lean` beside `bnBatchTensor4`, P4 in
`Foundation/DataParallel.lean` beside its negative twin, the `allReduceMeanF`-at-`[oc]` emit
checked against the parser. Gate: `lake build Certs`, parser round-trip, AuditAxioms. One session.

### 3.2 ResNet-34 — first, the template

8 `bnBatchF` emit sites in `ResNet34RenderB.lean`; swap to the 2b pattern under `replicas > 1`.
Re-render the R34 `*dp*` artifacts; `git diff verified_mlir/` shows the added `%arsum{g}mu` …
blocks and nothing else. DP twins of `ResNet34FullB` (T2) and `ResNet34BackCertifiedTieB` (T3).
`resnet34-syncbn-check`. The render header's "does NOT equal" paragraph and `formalization.yaml`
4k (the DP disclosure) rewritten. One to two sessions.

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

Separate decisions, per net, after 3.2–3.4: R34 ~22 h, MNv2 ~51 h, B0 ~73 h on the verified
side. §1's JAX-side measurement is the cheap predictor of whether a re-run will move the number;
it does not gate the render work, because the render work is what lets a chapter say the two
columns differ only in the lowerer.

---

## 4. What "done" looks like

* The four ops in the AST with `den`, emit, parser round-trip; P1, P2, P4 proved.
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

# bf16_dtype_ir.md — letting activations STAY bf16 between ops

Scoped 2026-08-25 on ares (6× RTX 4060 Ti, CUDA 12.9), out of `planning/bf16_renderer.md` §16.3
and §17. **Read this file before writing any code; read `bf16_renderer.md` §16–§17 for the
measurements that motivate it, and §19 for what changed when ViT was built anyway.**

---

## 0. ⭐⭐ THE ONE-PARAGRAPH VERSION

Seven nets render in bf16 on the verified path. Every one of them **converts back to f32 after
every operation**, because the bf16 ops bundle their casts internally: `f32 in → bf16 operands →
compute → f32 out`. That was the right first design and it carried all seven. It is also now
measured to be what caps them: it costs ConvNeXt **2.70× → 1.68×** on its conv work (§16.3), and it
is the gap between ViT's realized **1.23×** (§19) and the **1.71×** the same matmuls reach with
activations staying bf16 (§17.3). This document scopes letting a value stay bf16 from one op to the
next. ▶ **The recommended route is a defaulted `dt : Dtype`
FIELD on the ops plus a dtype-carrying emit stack — not an index on `SHlo`** — because the emitted
text is not theorem-tied, which makes that route cost ~zero proof work. §4 has the numbers.

---

## 1. Why — the two measurements that force it

Both from `bf16_renderer.md`, both taken 2026-08-24 on this box, both gate-2 checked (the bf16
really did reach the hardware in every arm):

| net | work timed | converts free to fuse | f32 boundary forced | kept |
|---|---|---|---|---|
| ConvNeXt-T | its own 173 convolutions, fwd+bwd | 111.7 → 41.3 = **2.70×** | 111.7 → 66.5 = **1.68×** | 64 % |
| ViT-Tiny | its own 387 matmuls, fwd+bwd, B=32 | 26.9 → 15.7 = **1.71×** | 26.9 → 26.2 = **1.03×** | **6 %** |

⚠⚠ **ViT'S ROW IS AN ISOLATED-STACK MEASUREMENT AND THE ARTIFACT BEAT IT — 1.23×, NOT 1.03×**
(`bf16_renderer.md` §19, built 2026-08-25). The six ops now exist and the net is wired. In isolation
every boundary convert is paid; in the real net many fuse into the LayerNorm, GELU and softmax that
sit between the matmuls. ▶ **So this project's ViT prize is 1.23× → ~1.7×, not 1.03× → ~1.7×** —
smaller than scoped, and starting from a wired net instead of a blank one, which is the bigger
change. ⭐ Read any isolated op-set timing here as a LOWER BOUND on the artifact.

⭐ ViT's "converts free" column brackets the independently-measured JAX-side 1.57× for ViT-Tiny
(§9.1), which is what says this model of the two worlds is the right one.

**Why the two nets differ so much** — arithmetic intensity, reasoning from these numbers rather
than a separate measurement: a convolution reuses each loaded input across many output positions,
so a cast amortises over a lot of arithmetic. ViT's matmuls are skinny (contracting dim 192, or 768
at the MLP) and bandwidth-bound on the activations, so an f32→bf16→f32 round trip at every op costs
about what the tensor cores save.

**The prize, if this lands:**

* ConvNeXt-T: **1.30× → ~1.40×** (§16.3's ceiling, at the boundary-constrained conv speedup).
* ViT-Tiny: **1.23× → ~1.7×**. ⭐ §17.1's six ops are WRITTEN (§19.3) and gate-2 correct, so this
  project no longer has to build them first — it has to give them bf16-in/bf16-out forms.
* EfficientNet-B0's 1.09× is **not** known to be helped — §14.3 is still open and its cause has not
  been attributed. Do not fold it into this project's justification.
  ⭐⭐ **AND §19.1 GIVES IT A NEW PRIME SUSPECT THAT IS NOT THIS PROJECT**: ViT's stem weight
  gradient is **0.19×** its f32 peer purely through cuDNN kernel selection, with every gate green.
  B0's depthwise set is the most varied in the repo, that check has never been run on it, and it is
  cheap. ▶ Run it before assuming B0 needs anything here.

---

## 2. ⚠ CORRECTING §17.4 BEFORE ANYONE BUILDS ON IT

`bf16_renderer.md` §17.4 says the original scoping's **Option 1** ("a bf16 `dotIn` variant … no
dtype in the type system; the cast lives inside the op") *"runs out"* at ViT, and that getting the
win *"needs the IR to carry a dtype — Option 2"*. **That is too strong, and the overstatement
matters because it makes the cheap route look unavailable.**

What is true: Option 1 **as practised** forces the f32 boundary, because every bf16 op built so far
takes f32 in and gives f32 out. What is false: that this is forced by the design. A
`bf16-in / bf16-out` variant of the same op bundles its casts exactly as today and does **not**
force a boundary.

▶ So the real question is not "index or no index". It is: **what stops a bf16-producing op being
wired to an f32-consuming one?** That is a type-discipline question, and it has a cheap answer and
an expensive one. §4.

---

## 3. The surface, counted (2026-08-25, `LeanMlir/Proofs/Codegen/StableHLO.lean`, 9908 lines)

| thing | count |
|---|---|
| `SHlo` constructors | **196** |
| `BatchableOp` constructors | 40 |
| `Raw` constructors | 94 |
| `Tok` constructors | 94 |
| batched-tag emit cases (`\| "tag", …`) | 118 |
| theorems in the file | 230 |
| files importing `StableHLO` | 65 |
| `ty [...]` call sites (the f32 type printer) | **1780** |
| `tyBf16 [...]` call sites | 85 |

### 3.1 ⭐⭐ THE COST FACT THAT DECIDES THE ROUTE — the emitted TEXT is not theorem-tied

This is the single most important thing to know before costing anything:

* `parse_toToks (r : Raw) : parse (toToks r) = some r` (`StableHLOParse.lean:300`) ties the
  **skeleton** to its token stream. One line per `Raw` constructor, all `simp only [...]; rfl`.
* **Nothing ties the emitted MLIR text.** `emitTok` is documented in the file as *"the audited
  lexical boundary (validated by `iree-compile` + GPU run)"*. The verified-lexer project that would
  have tied it is a **deliberate STOP** (`StableHLOLex.lean`, decided 2026-06-27, "low ROI"), and
  the practical guard is instead the CI byte-diff of every committed
  `verified_mlir/<net>_train_step.mlir` against its renderer.

▶ **Consequence: changing how the emitter chooses type strings costs NO proof work.** Changing
`den` does. Changing the `SHlo` *type* costs proof work across all 230 theorems and potentially 65
files. That asymmetry is the whole basis of §4's recommendation.

### 3.2 The work that is route-INDEPENDENT

Whatever route is taken, this half is the same and it is not optional:

**Every op that STORES bf16 must carry that rounding in its `den`.** The hardware rounds the
output; a `den` that does not say so claims more precision than the hardware delivers, which is the
unsound direction. This is exactly why `convBf16`'s `den` has an outer `rnd` and `dotInBf16`'s does
not (§9.2) — and it stays true here.

⭐ But **pass-through ops need nothing**. A transpose, reshape, slice, zero-pad or concatenate is an
exact permutation/selection of already-stored values: no new rounding, so `den` is unchanged and
only the emitted type string moves. On ViT that is `transpose`, `clsSlice`, `clsPad`, `headSlice`,
`headPad`; on ConvNeXt it is `transpose` and the reshapes.

⚠ **Reductions must accumulate in f32.** LayerNorm's mean/variance, softmax's sum, GAP, and every
bias/γ/β gradient are reductions over many bf16 values. Compute them in f32 internally and store
bf16 on the way out; a bf16 accumulate is what §9.3's vacuity argument is about
(`(1+u)^(n+1) − 1` at n = 3072 is **1.6e5**).

---

## 4. ⭐⭐ THE THREE ROUTES, COSTED

### Route A — a dtype INDEX on `SHlo` ("Option 2" as originally named)

`inductive SHlo : Dtype → Nat → Type`, every op reindexed.

* **Buys:** a static guarantee. A dtype mismatch cannot be written, let alone rendered.
* **Costs:** 196 + 40 constructors reindexed; 230 theorems restated at the f32 index; up to 65
  importing files touched; every existing render's elaboration disturbed.
* ⛔ **Risk is the real objection, not the typing.** `bf16_renderer.md` §11.1 records a *duplicate
  constant name* silently breaking `lake build LeanMlir` for three commits. A 236-constructor
  reindex of a proof library has a blast radius orders of magnitude larger, and it buys **nothing
  measurable** — the speedup comes from §3.2's work, which Route A does not do.

### Route B — a defaulted `dt : Dtype` FIELD + a dtype-carrying emit stack  ⭐ RECOMMENDED

Two changes, neither touching a type index:

1. **A field, not an index.** `Dtype` is a two-constructor inductive (`.f32 | .bf16`) with
   `DecidableEq`. Ops on the activation path gain **`(dt : Dtype := .f32)` as a TRAILING DEFAULTED
   field** — the `wx`/`clip`/`sd`/`bf16` idiom this repo uses everywhere, and the reason gate 1 is
   free: every existing call site elaborates unchanged and every committed artifact re-renders
   byte-identically.
   * `denOp`/`den` case on it: `.f32 ⇒ (what it does today)`, `.bf16 ⇒ rnd ∘ (the same thing)`.
     Because `dt` defaults to a literal, the existing `rfl`/`simp` lemmas still reduce.
   * `batchOpDescr`/`skel` must give a **DISTINCT TAG** per dtype (`"gelu"` vs `"geluBf16"`). Two
     graphs that emit different text must not share a `Raw` — §12.2's rule, and it is the same
     reason `convBf16` has its own tag today.
2. **The emit stack carries dtypes.** `emitTok (B : Nat) : Tok → List String → StateM Nat (String ×
   List String)` becomes `… → List (String × Dtype) → StateM Nat (String × List (String × Dtype))`.
   Each case reads its operand's dtype instead of assuming f32, and declares its result's.
   * `ty` gains a dtype argument (or a `tyOf : Dtype → List Nat → String` is introduced and `ty` is
     kept as `tyOf .f32`). **The 1780 `ty [...]` sites are mechanical**, and most are in ops that
     will never be bf16 — leave them.
   * ⭐ **And the stack can ASSERT.** Because the dtype is on the stack, `emitTok` can refuse a
     mismatch at *render* time rather than letting the lowerer catch it. That is a stronger gate
     than Route A's, at the point where the mistake is actually made.

* **Buys:** the measured speedup, with **zero proof churn** (§3.1) and gate 1 free.
* **Costs:** the `emitTok` signature + the cases actually made dtype-aware; §3.2's `den` work.
* ⚠ **Does not buy:** a compile-time guarantee. But the failure is **LOUD and was verified**: a
  module where a `bf16` value feeds an `f32`-declared op is **refused at parse** ("Unable to parse
  module assembly"), not silently accepted. Checked 2026-08-25.

### Route C — a peephole over the `Raw` tree

Emit as today, then delete `convert(bf16→f32); …; convert(f32→bf16)` pairs.

* ⛔ **Rejected.** The pair is not adjacent — a bias add, a LayerNorm and a residual add sit between
  them — so the pass would have to reason about which intervening ops are dtype-transparent, which
  is Route B's dtype information rediscovered in a harder place. And it puts an unaudited rewrite
  between the AST and the text, which is exactly the layer this repo keeps thin on purpose.

---

## 5. ⭐ THE FIRST INCREMENT — smallest thing that MEASURES

▶ **ConvNeXt's block interior, expand → GELU → project.** Not ViT, and not a whole net.

Reasons, in order:

1. **It is where the biggest tensor lives.** The expand output is `[B, 4c, h, w]` — the widest
   activation in the block. Today it is materialised in f32 twice (once out of the expand, once
   into the project). Keeping it bf16 halves that traffic.
2. **Three op kinds, all of which already have a bf16 twin to copy**: `convBf16` (needs a bf16-OUT
   form), `gelu` (needs bf16-in/bf16-out), `convBf16` again (needs a bf16-IN form).
3. **ConvNeXt is already measured end to end** (§16.2/§16.3), so there is a denominator: 157.6 ms
   f32 / 120.9 ms bf16 on the bare device, conv work 71 % of the step, ceiling ~1.40×.
4. ⚠ **ViT is the bigger prize and was the wrong place to start** — *when this was written*. It
   needed §17.1's six ops as well, so a failure there could not be attributed to this project or to
   those ops. ⭐ **That objection is now spent: the six ops are built, gate-2 correct and measured at
   1.23× (§19), so ViT has a clean baseline to flip against.** ConvNeXt is still the recommended
   first increment — it is where the biggest activation lives and its attribution is complete
   (§16.3) — but ViT is now a legitimate second, not a blocked one.

### 5.1 The gate that is new, and it inverts the usual one

Every bf16 increment so far has ADDED converts — `3 × nconv`, the histogram check in §16.1. **This
one must REMOVE them.** The gate is:

* the op histogram differs from the current bf16 artifact by **fewer `stablehlo.convert`**, and by
  nothing else;
* `scripts/bf16_gate2.py` still reports **173/173** convolutions with bf16 operands;
* `scripts/bf16_device_step.py` shows a wall-clock move on the bare device (§16.2 — never quote a
  trainer ms/step for a renderer claim, and never without stating `PJRT_FFI_RESIDENT`);
* gate 1: `git status verified_mlir/` shows only the new file.

⚠ **A convert count that does not fall means the round trip is still there** and the increment did
nothing, whatever the type strings say. That is this project's version of §9.2's trap, and it
deserves the same suspicion.

---

## 6. ⚠ THE TRAPS THIS PROJECT SPECIFICALLY CARRIES

1. **The f32 round trip XLA DELETES.** `convert(f32→bf16) → convert(bf16→f32)` as an *adjacent
   pair* is removed by the algebraic simplifier (`bf16_renderer.md`, "the trap is WORSE than
   written"). Entry and exit casts to a bf16 region are fine — they are separated by real work —
   but a region of *one* op is exactly that deleted pair, and it will read as "no speedup" rather
   than as an error.
2. **Reductions in bf16 are vacuous, not merely imprecise.** §3.2. LayerNorm's variance, softmax's
   sum, GAP, and every bias/γ/β gradient accumulate in f32 or the accuracy bound is worthless.
3. **The `den` must round on STORE.** An op that emits a bf16 result and whose `den` does not round
   claims precision the hardware does not deliver — the unsound direction, and nothing structural
   catches it. §9.2's note on `convBf16` is the precedent.
4. **Distinct tags per dtype.** Two graphs whose emitted text differs must not share a `Raw`, or
   `skel` makes them indistinguishable. §12.2.
5. **The entry-name bug, if any variant marker is added.** It has shipped **five times** across
   this repo by three distinct routes (`bf16_renderer.md` §15.2). If this work introduces a slug,
   `#guard` the variant function's returned STRING, not just its signature.
6. ⚠⚠ **A bf16 op can be SLOWER than its f32 peer, and no gate here will see it.** ViT's stem
   weight gradient is **0.19×** — `bf16_renderer.md` §19.1 — because cuDNN has a direct f32 kernel
   for its 209×209 transpose-trick window and no bf16 one. §5.1's gates check the convert count and
   the wall clock of the WHOLE step; a single pathological op can be buried by them if the rest
   improves. ▶ This project widens the bf16 surface to LayerNorm/GELU-adjacent shapes that have
   never been timed in bf16 — **profile each new op standalone against its f32 peer at the net's own
   shape**, per §19.4's added recipe step, before wiring it.
7. ⚠ **Master weights stay f32.** Casting a *weight* per op is cheap (weights are small next to
   activations) and the optimizer state must not degrade — the rule every net here already follows.

---

## 7. What this project is NOT

* **Not a fix for EfficientNet-B0.** §14.3 is open and unattributed; B0's 1.09× is confirmed on the
  bare device (§16.4) and nothing here predicts it improves. ⭐ §19.1 makes per-op kernel selection
  the cheaper hypothesis to test first. Attribute B0 first, with §16.3's
  method, before claiming this helps it.
* **Not a training run.** Nothing on this branch has been trained to convergence in bf16, and this
  work does not change that.
* **Not the verified lexer.** The emitted text stays untied (§3.1); the guard stays the CI
  byte-diff. If that ever changes, this project's cost model changes with it.
* **Not Route A.** If, after Route B has landed and been measured, the op count has grown enough
  that dtype mistakes are a real maintenance burden, Route A becomes a *refactor with a known
  payoff* rather than a speculative one. That is the same Option-1-then-Option-2 sequencing this
  repo used at the op level, one level up.

---

## 8. ▶ WHERE TO START, IN ORDER

1. Read `bf16_renderer.md` §16.2 (residency — a trainer ms/step is not the graph's), §16.3 (the
   attribution method and the boundary-convert measurement), §17 (why ViT is not built).
2. Reproduce the ConvNeXt baseline: `scripts/bf16_device_step.py` on
   `verified_mlir/convnextin_adamwxclipdrop{,bf16}_train_step.mlir` — expect **157.6 → 120.9 ms,
   1.30×**. If it does not reproduce, stop and find out why before building anything.
3. Prototype the emit shape **standalone, by hand**, before any Lean: a StableHLO module doing
   `conv(bf16) → gelu(bf16) → conv(bf16)` with no f32 in the middle, at ConvNeXt's own stage-1
   shapes (`c = 96`, `4c = 384`, `56²`, B = 32). Check it compiles, check the converts are gone,
   and time it against the f32-boundary version. **This is §15.3 step 2 and it has paid three
   times.** It is also the cheapest possible refutation of this whole document.
4. Only then: `Dtype`, the `dt` field on those three ops, the `emitTok` stack, §5.1's gates.

⚠ Step 3 can refute step 4. If the hand-built bf16-through block does not beat the
bf16-with-boundary block at ConvNeXt's real shapes, this project is not worth doing and the honest
outcome is to write that down here.

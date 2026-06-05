# typed_ir.md — closing the language gap: a denoted StableHLO IR

Design spike + plan for the one piece the proof suite still cannot
talk about: **the StableHLO the codegen actually emits.** Today the
proofs (`LeanMlir/Proofs/`) verify pure-math `Vec → Vec` backward
functions, and `MlirCodegen.lean` emits StableHLO *text* that is
connected to those proofs only by comments + the numerical oracle
(`check_jacobians.py`, `tests/vjp_oracle/`). This doc scopes turning
that comment into a theorem.

Status: **Phases 0–3 landed** in `LeanMlir/Proofs/IR.lean` — every per-op
backward bridge plus the composition (IR chain rule), audited under the
three-axiom closure. See **Results** at the end. The design rationale below
(written as the original spike) is kept as-is; it called the shape right.

## Framing: tests vs. the language gap

Discharged instances (`Mini`, `Spatial` in `MnistCNN.lean`) are, honestly,
**non-vacuity unit tests** — they confirm the conditional capstones are
instantiable at non-degenerate points. Valuable, but diminishing: a
tenth discharged net proves little the first two didn't.

The *language gap* is qualitatively different. Right now "verified
backprop that compiles to GPU" is really "the **math** is verified; the
compiler from math to GPU is **trusted** + numerically spot-checked." A
denoted IR converts the per-op correspondence from a comment into a
machine-checked equality. That's the prize.

## The goal, precisely

Give the emitted backward a **denotational semantics** `⟦·⟧` landing in
the *same* `Vec`/`Mat`/`Tensor3` types the proofs use, and prove, per op
and composed:

```
⟦ emitBackward L ⟧  =  (L_has_vjp …).backward          -- smooth ops, globally
⟦ emitBackward L ⟧  =  (L_has_vjp_at …).backward       -- kinked ops, at smooth points
```

For a whole network this should fall out of per-op bridges + an IR-level
composition lemma (`⟦g ∘ᵢᵣ f⟧ = ⟦g⟧ ∘ ⟦f⟧`), mirroring how `vjp_comp`
gives whole-net VJPs from per-layer ones.

## The op surface (grounded)

`grep stablehlo.* MlirCodegen.lean` — the **backward-relevant** subset is
small:

| Category | Ops | Used by (backward of) |
|---|---|---|
| contraction | `dot_general` | dense, attention, conv-weight grad |
| spatial | `convolution`, `reverse`, `pad` | conv input/weight grad (transposed conv) |
| reduction | `reduce`, `reduce_window` | softmax/BN/LN sums, pool |
| select | `compare`, `select` | relu, maxpool, relu6 kinks |
| shape | `broadcast_in_dim`, `reshape`, `transpose`, `slice`, `concatenate` | everywhere |
| elementwise | `add`, `multiply`, `subtract`, `divide`, `maximum`, `minimum`, `negate` | everywhere |
| unary | `exp`, `log`, `sqrt`, `rsqrt`, `tanh`, `logistic`, `power` | softmax, BN, GELU, swish, sigmoid |

~25 ops total; ~10 carry the structural weight. `reshape`/`transpose`/
`broadcast` are free (data-preserving reindexings — the repo already has
`flatten`/`unflatten` bijections for exactly this).

## IR design spike

### D1 — Tensor representation: flat + shape metadata

```lean
-- Row-major flat buffer + a shape, matching StableHLO's memory model.
def Tensor (s : List Nat) : Type := Fin s.prod → ℝ
```

`reshape` is literally `id` on data (when `prod`s match); `flatten`/
`unflatten` to the proofs' `Mat`/`Tensor3` are the existing bijections
(`Mat.flatten : Mat m n → Vec (m*n) = Tensor [m, n]`). So the semantics
lands in `Vec`/`Mat`/`Tensor3` "for free" via lemmas we already have.
**Spike shortcut:** for the first slice, skip the general `Tensor` and
use `Vec`/`Mat` directly — generalize once it pays off.

### D2 — Backward graphs are expression trees, not SSA

The real emitter threads SSA names (`%cva0`, sharing via reuse). But a
**backward subgraph is a small straight-line expression**, and *sharing
is a printing optimization that doesn't change the denotation.* So model
the IR as an **intrinsically-typed expression tree** producing a tensor,
denote that, and treat SSA/let-sharing as an orthogonal,
semantics-preserving printer concern. This sidesteps the entire
SSA/dominance/let-binding swamp for the correctness proof.

```lean
-- Illustrative (not yet compiling). The StableHLO subset as typed exprs.
inductive Expr : List Nat → Type where
  | var      (s : List Nat) : Expr s                         -- a graph input (e.g. dy)
  | const    {s} (t : Tensor s) : Expr s                     -- stablehlo.constant
  | dotGen   {a b c} (l : Expr [a,b]) (r : Expr [b,c]) : Expr [a,c]   -- stablehlo.dot_general
  | conv     {…} (x : Expr …) (w : Expr …) (cfg : ConvCfg) : Expr …   -- stablehlo.convolution
  | reverse  {s} (dims : List Nat) (x : Expr s) : Expr s      -- kernel flip for transposed conv
  | redWindow{…} (cmp : Cmp) (x : Expr …) (win : WinCfg) : Expr …     -- reduce_window
  | compareEQ{s} (a b : Expr s) : Expr s
  | select   {s} (m a b : Expr s) : Expr s
  | bcast    {s t} (x : Expr s) : Expr t
  | binop    {s} (op : BinOp) (a b : Expr s) : Expr s         -- add/mul/sub/div/max/min
  | unop     {s} (op : UnOp) (a : Expr s) : Expr s            -- exp/log/sqrt/rsqrt/tanh/logistic
```

### D3 — The denotation `⟦·⟧`

```lean
-- Env maps the graph's free `var`s (the cotangent, the saved forward
-- activations) to concrete tensors; ⟦e⟧ ρ is the value e computes.
def denote : Expr s → Env → Tensor s
  | .dotGen l r, ρ => Mat.mul (denote l ρ) (denote r ρ)   -- reuse Mat.mul
  | .binop .add a b, ρ => fun i => denote a ρ i + denote b ρ i
  | .unop .exp a, ρ => fun i => Real.exp (denote a ρ i)
  | .conv x w cfg, ρ => conv2d … (denote x ρ) (denote w ρ)  -- DEFINE = conv2d (see R2)
  | …
```

Key move (see **R2/R4**): define the IR's `conv`/`reduce_window`
denotations to **be** the proofs' `conv2d`/`maxPool2`, so the spatial
bridges are definitional rather than re-derived.

### D4 — Emit into the IR; print the IR to text

Eventual end state: retarget the emitter `NetSpec → Expr` (one source of
truth), with a separate trivial `Expr → String` printer whose output
matches today's StableHLO. **Spike shortcut:** build a *parallel*
`emitBackward : Layer → Expr` next to the existing string emitter, prove
it correct, and defer the retarget. (Caveat in **R3**.)

### D5 — Bridge theorem shape

Per op, e.g. dense input-grad (the trivial base case — one `dot_general`):

```lean
theorem dense_backward_bridge (W : Mat m n) (b : Vec n) (dy : Vec n) :
    denote (emitDenseBackward W) (env dy) = (dense_has_vjp W b).backward 0 dy := …
```

For kinked ops, the bridge is **conditional on smoothness** and reuses
the theorems that already exist:
`relu_codegen_matches_canonical` (`MLP.lean:446`),
`maxPool2_codegen_matches_canonical` (`CNN.lean:2291`). Those become the
op-level obligations for relu/maxpool, lifted to the `⟦·⟧` level.

## Phased plan

| Phase | Scope | Why first / cost |
|---|---|---|
| **0a** | **Dense slice** — `Expr` scaffolding, `denote`, `dense_backward_bridge`, axiom-audit | Establishes the plumbing. Bridge is ~`rfl` (dense backward is one `dot_general`) — deliberately trivial, tests the *framework*, not a proof. Hours. |
| **0b** | **ReLU slice** — `compare`+`select`, reuse `relu_codegen_matches_canonical` | First *conditional* (smooth-point) bridge; tests the kink pattern with an existing theorem. Hours–day. |
| **1** | Elementwise + unary + `dot_general` general semantics → the **smooth layers** (softmax, BN, LN, GELU, sigmoid, swish, SE) | The bulk of "more of the same." Each is a small expr tree over Phase-0/1 primitives. Days each. |
| **2** | `convolution` (+`reverse`/`pad` for transposed conv) and `reduce_window`/maxpool | The hard ones: re-opens conv index arithmetic; maxpool reuses the smooth-point bridge. Weeks. |
| **3** | IR composition lemma + **one whole-network bridge** (start with `mnistCnnNoBn`, then ViT) | The payoff theorem. Compositional once Phase 2 lands. |
| **4** (optional, big) | **Retarget** the real emitter to `NetSpec → Expr`; show `Expr`-printer string == legacy string | Closes R3 so the *trained binary* is covered, not just a mirror. |

## Decisions & risks

- **R1 — SSA/sharing.** Sidestepped by D2 (expression trees; sharing is
  printer-only). Revisit only if a backward graph genuinely needs a DAG.
- **R2 — Conv index arithmetic.** The interpreter's `conv` must match
  `conv2d`. *Mitigation:* **define** `⟦conv⟧ := conv2d` (and
  `⟦reduce_window max⟧ := maxPool2`) so the op-level bridge is
  definitional; the work moves to proving the *transposed-conv backward
  graph* (a `convolution` with flipped/`reverse`d kernel and swapped
  channels) denotes `conv2d_input_grad_formula`. Still real, but it's the
  same algebra as `conv2d_has_vjp3`, not new.
- **R3 — Mirror drift (the teeth question).** Until Phase 4, the proof
  covers the *IR emitter*, which could diverge from the legacy string
  emitter. A parallel `emitBackward` proven correct says nothing about
  the binary that trains unless `Expr`-printer-string == legacy-string.
  Honest interim claim: "the IR backward is correct, and is what the
  retargeted emitter will produce." Phase 4 removes the asterisk.
- **R4 — `⟦·⟧` fidelity to *real* StableHLO (the intellectual crux).**
  We don't have an upstream formal StableHLO semantics; `⟦·⟧` is a
  **hand-written model**. Proving against it doesn't prove against IREE's
  StableHLO unless the model is faithful. So the bridge *moves* the trust
  from "the emitter matches the math" to "`⟦·⟧` matches the StableHLO
  spec for the configs we emit" — a **smaller, centralized, auditable**
  surface (one `denote` def, ~10 ops, fixed configs: stride 1, SAME pad,
  one dim-number layout), and one the existing numerical oracle already
  cross-checks against real IREE. That's a genuine improvement, but it is
  not zero trust — state it plainly.
- **D-scope — restrict configs.** Only model the exact op configurations
  the codegen emits (no general dilation/grouping/batching). Document the
  restriction; a graph using an unmodeled config simply has no `⟦·⟧`.

## What this does NOT close (the honest ceiling)

Below the `⟦·⟧` boundary stays trusted, same as the float discussion:

1. `Expr`-printer → text faithful (small; round-trip-testable).
2. **IREE/XLA** lowering StableHLO → GPU (not verifying IREE).
3. **float32 ≠ ℝ.** `⟦·⟧` is over ℝ; the hardware rounds. Even a perfect
   compiler has rounding error. The honest end-state claim is: *"the
   emitted StableHLO **denotes** the proven real-valued VJP; the float
   realization is the standard numerical-analysis gap, bounded
   empirically by the oracles."* Far stronger than today's "trust the
   comment," and the strongest statement available without a Flocq-grade
   float library in Lean (which doesn't exist) + a float error-analysis
   layer on top.

## Success criteria

- Phase 0a/0b: `Expr`/`denote`/bridges compile; `#print axioms` on the
  bridge theorems = `[propext, Classical.choice, Quot.sound]`; wired into
  `tests/AuditAxioms.lean`.
- Phase 3: `⟦emitBackward mnistCnnNoBn⟧ = (…_has_vjp_at).backward` at a
  smooth point, axiom-clean.
- The README "codegen trust boundary" section shrinks from "numerical
  oracle only" to "denotational bridge for ops X…Z; oracle for the rest."

## Results (landed in `LeanMlir/Proofs/IR.lean`)

All per-op backward bridges + the composition are done and audited in
`tests/AuditAxioms.lean` (three-axiom closure, no `native_decide`, no
`sorry`). The IR `Back` carries `cotangent`, `dotGeneral`, `selectPos`,
`scale`, `sumBroadcast`, `sub`, `scaleConst`, `add`, plus `subst`.

| op | bridge | how |
|---|---|---|
| dense | `dense_back_bridge` | definitional (`dot_general` = `Mat.mulVec`) |
| conv | `conv_back_bridge_{1to2,2to2}` | reversed-kernel identity, by expansion at the `Spatial` shapes |
| relu | `relu_back_bridge` | smooth-point, via `relu_codegen_matches_canonical` |
| maxpool | `maxpool_back_bridge` | smooth-point, via `maxPool2_codegen_matches_canonical` |
| gelu / swish / sigmoid | `*_back_bridge` | definitional (diagonal `scale`) |
| BN / LayerNorm | `bn_normalize_back_bridge`, `bn_back_bridge`, `layernorm_back_bridge` | 3-term rank-1 via `sumBroadcast`+`sub`+`scaleConst` |
| softmax | `softmax_back_bridge` | rank-1, same reduce/broadcast shape |
| SE | `se_back_bridge` | fan-in `add` + `denote_subst` (gate plugged in) |
| **composition** | `denote_subst` + `twoDense_back_bridge` | the IR chain rule + an end-to-end composite |

**Findings vs. the original risk table:**

- **`⟦conv⟧ := conv2d` (D3) held up.** The conv backward bridge is *cheap at
  concrete shapes* — `fin_cases` over the spatial positions + `simp`
  (~7s/shape), no partial bijection. It discharges the reversed-kernel
  identity `dx = conv(dy, reverse(Wᵀ))` that `CNN.lean:288` only asserts in
  prose. So "conv is weeks" → "concrete conv is hours (done)."
- **The kink pattern reused the existing bridge theorems** verbatim (relu,
  maxpool), conditional on the same smoothness hyps as the codegen boundary.
- **BN's 3-term rank-1 formula bridged cleanly** once `sumBroadcast`/`sub`/
  `scaleConst` were added; the `bnForward = bnAffine ∘ bnNormalize` cast
  collapsed by `rfl`, and LayerNorm came free (definitionally BN).
- **`denote_subst` (the chain rule) is the keystone** — SE (fan-in) and the
  composite demonstrator both go through it.

## Remaining gap to the full `mnistCnnNoBn` whole-network bridge

The per-op + composition machinery is complete; three well-scoped
extensions stand between here and `⟦emit mnistCnnNoBn⟧ = mnistCnnNoBn_has_vjp_at.backward`:

1. **Tensor3 IR.** conv/maxpool bridges are Tensor3-level (`convBackDenote`,
   `maxPoolBackDenote`), separate from the Vec `Back`; composing the whole
   CNN into one graph needs `Back`/`subst`/`denote_subst` lifted to Tensor3
   (or everything flattened to Vec).
2. **`HasVJPAt` smooth-point variants.** Per-op bridges target the global
   `HasVJP`; `mnistCnnNoBn_has_vjp_at` composes via `vjp_comp_at`, so the
   chained bridge needs `_at` versions (the kink bridges are already
   smooth-point-conditional, so this is mostly mechanical).
3. **General-shape conv identity.** The remaining ~100–150-line
   partial-bijection sum-reindex (all dims, odd kernels) — the one genuine
   hard lemma; concrete shapes are done.

None is new research; each is bounded plumbing on top of what's landed.

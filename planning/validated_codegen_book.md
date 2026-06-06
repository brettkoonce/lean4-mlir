# validated_codegen_book.md — closing printer faithfulness (R4) for the full tour

**Scope.** This doc is about *one* thing: closing **R4 — printer faithfulness**
— the last trusted surface in the verified-codegen story, for the full
classification tour **MNIST linear (ch 2) → ViT (ch 10)**. It is the deepest
and only *research-grade* item in `planning/validation_sweep.md` (everything
else there is done). Predecessors: `validation_sweep.md` (the completed book
sweep), `verified_codegen.md` (the pipeline design).

---

## The gap, precisely

Today's verified path is a two-link chain:

```
proven math (ℝ, Mathlib fderiv)
        ▲
        │  bridge theorems          ← PROVEN  (⟦g⟧ = proven backward/forward),
        │  (IR.lean)                          3-axiom, in AuditAxioms
        │
   IR AST  g : Back / Fwd / Back3   ← g.denote : … → Vec/Tensor3   (semantics in ℝ)
        │
        │  render : IR → String     ← TRUSTED  (this is R4)
        ▼
   StableHLO text  →  IREE  →  vmfb  →  GPU
```

The bridge proves the **IR denotes the proven math**. What is *not* proven is
that `render g` (the StableHLO **text** in `IRPrint.lean`) actually means the
same function as `g.denote`. "The emitted text is `print(IR)` by construction"
is a statement about how the printer was *written*, not a theorem. That is R4.

Two things ride on R4 that are *not* part of it and stay trusted regardless
(they belong to separate surfaces): IREE's lowering of StableHLO→GPU, and
`float32 ≈ ℝ`. R4 is strictly "the text faithfully encodes `g.denote`."

**Why it's finite.** The printer emits a *fixed, small* subset of StableHLO —
**~24 ops across the entire ch 2–10 tour** (grep `stablehlo.` in
`IRPrint.lean`), no control flow, no dynamic shapes, every shape a literal.
That turns "a semantics for StableHLO" (huge) into "a semantics for *these 24
ops at static shapes*" (a finite, inspectable build).

---

## What "printer faithfulness" means formally

Introduce a Lean **denotational semantics for the emitted StableHLO subset**
and prove the printer preserves it. Concretely, the theorem to land is:

> For every emitted graph `g`, `⟦ render g ⟧ₛₜ = g.denote`,

where `⟦·⟧ₛₜ` is a denotation of StableHLO *programs* into the same `Vec` /
`Tensor` world the proofs use. Composed with the existing bridge
(`g.denote = proven fderiv`), this gives the end-to-end statement the book
wants: **the StableHLO text computes the Mathlib-proven derivative.**

It splits into a syntactic and a semantic half:

1. **Syntactic (round-trip).** The text is a faithful encoding of a typed AST:
   `parse (pretty a) = a` for the emitted AST `a`. SSA names and op ordering are
   the only freedom; they are denotation-irrelevant (a printer concern, "D2" in
   `verified_codegen.md`), so round-trip is *modulo* α-renaming of SSA values.
2. **Semantic (denotation preservation).** The AST means what the IR means:
   `⟦ emit g ⟧ₐ = g.denote`, where `emit : IR → StableHLOAst` and `⟦·⟧ₐ` is the
   AST denotation. This is the load-bearing half.

`IRPrint.lean` already has the seed of the AST side: `Hlo` / `HloF`
(`inductive Hlo`, line 38) with `Hlo.render : Hlo → StateM Nat (String × String)`
— a typed node walked by a stateful pretty-printer. Today most per-op renderers
(`renderDense`, `renderSoftmax`, `renderConvWGrad`, …) build strings *directly*,
bypassing `Hlo`. **Step 0 of this work is consolidation:** route every renderer
through one typed AST, so the printer becomes `pretty ∘ emit` and R4 reduces to
properties of `emit` (semantic) and `pretty` (syntactic).

---

## The components to build

1. **`StableHLO` subset AST** (`Proofs/Hlo/Syntax.lean`). A typed, shape-indexed
   inductive covering the 24 ops (table below), at static shapes — extend the
   existing `Hlo`. Regions (the `reduce`/`select_and_scatter` reducer bodies) are
   a small closed sub-language (binary `add`/`maximum`), not arbitrary MLIR.
2. **AST denotation** `⟦·⟧ₐ` (`Proofs/Hlo/Denote.lean`) into `Vec`/`Tensor` over
   `ℝ`. This is the formal *meaning* of each op — the artifact that makes
   "faithful" a theorem instead of a comment.
3. **`emit : IR → StableHLO AST`** + the **faithfulness theorem**
   `⟦emit g⟧ₐ = g.denote`, proved per node and closed under composition
   (mirrors how `denote_subst` already chains the IR bridges).
4. **`pretty : AST → String`** as the only remaining textual step, plus either
   (a) a verified `parse` with `parse (pretty a) = a` (closes the syntactic
   half formally), or (b) leave `pretty` as a small audited lexical encoder and
   cross-check against the StableHLO reference interpreter (validated, not
   proven). (a) is the clean end; (b) is the pragmatic 90%.
5. **Per-op spec-conformance ledger** — see "irreducible residue."

End state: the trusted surface shrinks from "the whole printer (a ~2k-line
string function)" to "our `⟦·⟧ₐ` for 24 ops matches the StableHLO spec," each
line of which is cross-validated against the reference interpreter.

---

## The op surface for the full tour (the finite obligation)

Every `stablehlo.*` the printer emits for MNIST-linear → ViT, by denotation
difficulty (the real measure of work):

| tier | ops | denotation | conformance risk |
|---|---|---|---|
| **Elementwise** (easy) | `add` `subtract` `multiply` `divide` `maximum` `minimum` `compare` `select` `and` `exponential` `tanh` `logistic` `rsqrt` `constant` | pointwise `ℝ` map; `select`/`compare` are pointwise on a bool tensor | low — matches the obvious pointwise spec (float rounding is the *float32* surface, not R4) |
| **Shape / layout** (fiddly indices) | `reshape` `broadcast_in_dim` `transpose` `reverse` | pure reindex: a permutation/replication of indices | medium — must match StableHLO's `broadcast_dimensions` / `permutation` / dim-number conventions exactly |
| **Reductions** (regions) | `reduce` `reduce_window` `select_and_scatter` | fold over (windowed) index sets with a region reducer | medium-high — the reducer region + window/stride/padding semantics |
| **Contractions** (the hard core) | `dot_general` `convolution` | sum-of-products over contracting/batch dims; windowed correlation | **high** — `dot_general` dimension-numbers and `convolution` window/dim-number spec are where most StableHLO complexity lives; getting `⟦·⟧ₐ` to match the spec is the crux |

That is the **entire** R4 obligation for the tour: ~24 ops, four tiers. There
is no control flow, no dynamic shape, no custom call. The contraction tier
(`dot_general`, `convolution`) is ~80% of the difficulty; everything else is
mechanical once the index conventions are pinned.

Op → chapter (where the obligation first appears):
`dot_general`, elementwise, shape → **ch 2–3** (linear, MLP);
`convolution`, `reverse`, `reduce_window`/`select_and_scatter` → **ch 4** (CNN);
`reduce`, `rsqrt`, `broadcast_in_dim` → **ch 5** (BatchNorm);
`exponential`/`divide`/`maximum` (softmax), `tanh` (gelu), `logistic` (sigmoid)
→ **ch 3/8/9/10**; batched `dot_general` → **ch 10** (attention).

---

## Staged plan (chapter-aligned; each stage closes R4 for its chapters)

Each stage = extend the AST + denotation with that tier's ops, prove
`⟦emit g⟧ₐ = g.denote` for the chapter's graphs, and cross-check `⟦·⟧ₐ` against
the StableHLO reference interpreter on the emitted text. Each is independently
shippable and is a real result on its own.

- **Stage A — the dense/relu fragment (ch 2–3).** Elementwise + `dot_general` +
  shape ops. Close R4 for `mlp_back` / `mlp_train_step` end to end: the MLP
  StableHLO **provably** computes the proven VJP + param grads + loss cotangent.
  Smallest closed loop; the template for everything after. Start here.
- **Stage B — convolution + pooling (ch 4).** Add `convolution`, `reverse`,
  `reduce_window`/`select_and_scatter`. The hard tier; budget most of the
  project here (the `convolution` dimension-numbers denotation).
- **Stage C — reductions + norm (ch 5).** Add `reduce`, `rsqrt`,
  `broadcast_in_dim` reductions; close BN/LN.
- **Stage D — smooth activations + softmax (ch 3/8/9).** `exponential`, `tanh`,
  `logistic`, `divide`, `maximum`, `select` — mostly elementwise; fast.
- **Stage E — attention / ViT (ch 10).** Batched `dot_general` + the SDPA graph;
  reuses A–D. Closes the tour.

After each stage, the chapter's `MLIR:` section can upgrade its ledger line from
*"printer faithfully renders ⟦·⟧ (trusted, validated numerically)"* to
*"printer faithfulness proven for these ops (`⟦render g⟧ₛₜ = g.denote`),
conformance to the StableHLO spec cross-checked against the reference
interpreter."*

---

## The irreducible residue (what stays trusted even after R4)

Closing R4 does not make the system axiom-free end-to-end; it *relocates and
shrinks* the trust:

1. **Per-op spec conformance.** `⟦·⟧ₐ` is *our* reading of StableHLO; it must
   match the official spec / what IREE implements. This cannot be proven inside
   Lean (it is a spec-conformance claim). Make it an **explicit, finite per-op
   ledger** (~24 entries) and validate each by running the **StableHLO reference
   interpreter** on emitted snippets and diffing against `⟦·⟧ₐ` evaluated in
   Lean (a sister oracle to `check_ir_codegen.py`, which today only checks IREE
   numerics). 24 audited lines ≫ better than one trusted 2k-line printer.
2. **IREE lowering** (StableHLO → GPU) — a separate, large surface; out of
   scope here, covered empirically by the existing GPU oracle.
3. **`float32 ≈ ℝ`** — the proofs and `⟦·⟧ₐ` are over `ℝ`; kernels run `float32`.
   Its own surface (a per-op error-bound layer); the oracle's ~1e-7 agreement is
   the empirical stand-in. R4 and float are orthogonal — R4 is exact-real
   faithfulness of the text.

So after the full tour: **`⟦render g⟧ₛₜ = g.denote = proven fderiv`** is a Lean
theorem; the residual trust is a 24-line per-op conformance ledger (validated
vs the reference interpreter) + IREE + float — each named, none a 2k-line
black box.

---

## Effort & honesty

This is **not** the mechanical ch-sweep; it is a (small, finite) verified-
compiler-backend build — CompCert/Vellvm in spirit, scoped to a 24-op static
fragment. Honest sizing: the elementwise + shape tiers are days; `reduce`/
windowed are weeks; `dot_general` + `convolution` denotation-vs-spec is the
multi-week crux; the verified `parse` round-trip (optional half (4a)) is a
separate chunk. Months, not days — but bounded, no open research question, and
deliverable stage-by-stage (Stage A alone is a complete, publishable "verified
StableHLO for an MLP train step").

It is the natural spine of **ch 13 ("On Verification")**: the book can state R4
as the honest frontier, show Stage A closed end to end, and lay out B–E as the
mapped remainder — turning the one hand-wave in the verified-codegen story into
a finite, in-progress theorem.

---

## Start here

1. Read this + `validation_sweep.md` ("Extending…" section) + `verified_codegen.md`.
2. **Step 0 (consolidate):** route the ch 2–3 renderers through the existing
   `Hlo` AST so the printer is `pretty ∘ emit`; nothing else changes.
3. **Stage A:** add `Proofs/Hlo/{Syntax,Denote}.lean` for the elementwise +
   `dot_general` + shape ops; prove `⟦emit g⟧ₐ = g.denote` for `mlp_*`; add the
   reference-interpreter cross-check oracle. Put the resulting theorem in
   `AuditAxioms.lean` (3-axiom gate) and upgrade the ch-3 `MLIR:` ledger line.
4. Then B → C → D → E, one chapter tier at a time.

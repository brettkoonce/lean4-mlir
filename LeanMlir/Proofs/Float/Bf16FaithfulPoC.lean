import LeanMlir.Proofs.Codegen.StableHLO

/-! # PoC: the bf16-mixed MNIST-linear render-tie (planning §5, the symmetric gap)

The companion to `E4M3FaithfulPoC.lean`. Where the fp8 render-tie is **proven but
un-lowered** (IREE's CUDA backend can't emit f8 — see
`upstream-issues/2026-06-iree-cuda-fp8-nvptx-lowering/`), bf16 is the inverse: its
mixed-precision *accuracy* bound exists (`dense_close_mixed`,
`u_leaf = 2⁻⁸`) and it **does lower on CUDA** (a `bf16`-in / `f32`-accumulate
`dot_general` compiles for `sm_86`/`sm_89`), but its proof was **untied** — nothing
connected the emitted bf16 graph to that bound. This file closes the structural
half: the bf16-mixed linear graph **denotes** the rounded-operand linear, for any
rounding `rnd` (bf16 being one instance). No accuracy claim — purely "the bf16 cast
+ matmul + fp32 accumulate computes `dense` of the rounded operands".

**Why this is the simpler twin of §3b.** The deployed bf16-mixed kernel is "cast
operands to bf16, multiply-accumulate in fp32". The fp32 accumulate makes the
reduction the *exact* `∑` (in ℝ), so — exactly as the fp8 §3b tie treats the
int-accumulate — the only deviation from exact is in the *operands*. There is no
block scale to factor through the sum (fp8's `dequant_factors`), so the tie falls
straight out of the `den`-faithful `operand`/`dotIn`/`addBcast` ops.

**No `SHlo` surgery (at depth 1).** The bf16 cast on the *input* activations is
baked into the operand value `rnd ∘ x` — the byte preparation the kernel does
before the GEMM — so depth-1 needs no new op, mirroring `E4M3FaithfulPoC`'s
host-side `actCode`. Depth > 1 (rounding *intermediate* activations) needs an
in-graph `convertF` round node (`den (convertF rnd e) = rnd ∘ den e`); that op is
the same ingredient fp8's depth-> 1 in-graph quant (planning §5) would use.

**Accuracy companion (separate, already exists).** Instantiate this tie's abstract
`rnd` at bf16 round-to-nearest and feed `|rnd x − x| ≤ 2⁻⁸|x|` into
`dense_close_mixed` (`u_leaf = 2⁻⁸`, `u_acc = 2⁻²⁴` for the fp32
accumulate). Render-tie (here) ∘ accuracy (there) = the tied-and-lowered bf16
forward — the thing fp8 can prove but not run, and bf16 can now do both.

All theorems kernel-close under `[propext, Classical.choice, Quot.sound]`
(`tests/AuditAxioms.lean`); names kept short for the audit's per-line grep.
-/

open Proofs Proofs.StableHLO

namespace Proofs.Bf16PoC

variable {m n : Nat}

/-- The bf16-rounded **input activations** `rnd(xᵢ)` — the bf16 bytes the kernel
    feeds the GEMM (the leaf cast, baked into the operand like fp8's `actCode`). -/
noncomputable def actBf16 (rnd : ℝ → ℝ) (x : Vec m) : Vec m := fun i => rnd (x i)

/-- The bf16-rounded **weights** `rnd(Wᵢⱼ)` (cast once; the other GEMM operand). -/
noncomputable def wBf16 (rnd : ℝ → ℝ) (W : Mat m n) : Mat m n := fun i j => rnd (W i j)

/-- **The emitted bf16-mixed linear graph.** Built only from `den`-faithful `SHlo`
    ops: `operand` (the bf16 activation bytes) → `dotIn` (bf16 weights; the `den`
    `∑` is the fp32 accumulate) → `addBcast` (the fp32 bias). The "bf16 leaf, fp32
    accumulate" linear. No block scale (cf. `e4m3LinearGraph`'s `layerScaleF`). -/
noncomputable def bf16LinearGraph (rnd : ℝ → ℝ) (W : Mat m n) (b : Vec n) (x : Vec m) : SHlo n :=
  .addBcast "%b0" b (.dotIn "%Wb" (wBf16 rnd W) (.operand "%xb" (actBf16 rnd x)))

/-- **The intended algorithm:** the exact-ℝ linear map on the rounded operands,
    `mnistLinear (rnd∘W) b (rnd∘x)`. fp32 accumulate ⇒ the `∑` is exact; the only
    deviation from `mnistLinear W b x` lives in `rnd`. -/
noncomputable def bf16Linear (rnd : ℝ → ℝ) (W : Mat m n) (b : Vec n) (x : Vec m) : Vec n :=
  mnistLinear (wBf16 rnd W) b (actBf16 rnd x)

/-- **bf16 render-tie (structural faithfulness, planning §5).** The emitted
    bf16-mixed graph denotes exactly the rounded-operand linear, for **any** `rnd`
    (bf16 round-to-nearest being one instance). No accuracy claim — purely "the
    bf16 cast + fp32-accumulate matmul implements `dense` of the rounded operands".
    Strictly simpler than `e4m3_render_faithful`: no scale to factor, so it is the
    `den` of the faithful ops unfolded. -/
theorem bf16_render_faithful (rnd : ℝ → ℝ) (W : Mat m n) (b : Vec n) (x : Vec m) :
    den (bf16LinearGraph rnd W b x) = bf16Linear rnd W b x := rfl

/-! ## The EMITTABLE graph — same tie, one `dotInBf16` node

`bf16LinearGraph` above pre-rounds its operands (`wBf16`/`actBf16`), which is right for the
proof and unemittable in practice: expressing "round, then multiply" as separate nodes
means a convert PAIR, and XLA deletes those (measured — see `convertF`'s comment). The
`dotInBf16` node bundles the casts into the matmul so the value stays bf16 *across* the
op, which is the only shape that survives the optimizer and reaches tensor cores.

The point of this section is that bundling costs no proof: the tie is the SAME `rfl`. -/

/-- The bf16 linear graph as actually emitted: raw operands, one `dotInBf16`. -/
noncomputable def bf16LinearGraphEmit (rnd : ℝ → ℝ) (W : Mat m n) (b : Vec n) (x : Vec m) :
    SHlo n :=
  .addBcast "%b0" b (.dotInBf16 rnd "%W" W (.operand "%x" x))

/-- **The emittable graph denotes the same thing as the pre-rounded one.** So
    `bf16_render_faithful` transfers to the node that can actually be lowered, and the
    render-tie covers the graph we ship rather than an idealisation of it. -/
theorem bf16_render_faithful_emit (rnd : ℝ → ℝ) (W : Mat m n) (b : Vec n) (x : Vec m) :
    den (bf16LinearGraphEmit rnd W b x) = bf16Linear rnd W b x := rfl

/-- Stated directly: the two graphs are denotationally interchangeable. -/
theorem bf16_emit_eq_prerounded (rnd : ℝ → ℝ) (W : Mat m n) (b : Vec n) (x : Vec m) :
    den (bf16LinearGraphEmit rnd W b x) = den (bf16LinearGraph rnd W b x) := rfl

/-! ## Depth > 1 — closed with the `convertF` round node

The header above says depth-1 needs no new op because the leaf cast is baked into the
operand value, but that rounding *intermediate* activations "needs an in-graph
`convertF` round node (`den (convertF rnd e) = rnd ∘ den e`)". That op now exists
(`Proofs/Codegen/StableHLO.lean`), so this section closes the gap the header left open.

The point is that the tie **composes**: a rounded activation feeding the next layer is
still exactly `dense` of rounded operands, with no cross-layer error term to track,
because the fp32 accumulate keeps every `∑` exact in ℝ. So the whole-net statement is
the one-layer statement applied twice — which is what makes bf16 the easy twin of fp8.
-/

/-- A **depth-2** bf16-mixed graph: rounded operands at BOTH layers, and — the new
    part — the intermediate activation rounded *in the graph* by `convertF`, exactly
    where a bf16 kernel would hand its f32 accumulator back as bf16 bytes for the next
    GEMM. -/
noncomputable def bf16Depth2Graph (rnd : ℝ → ℝ) {p : Nat}
    (W₀ : Mat m n) (b₀ : Vec n) (W₁ : Mat n p) (b₁ : Vec p) (x : Vec m) : SHlo p :=
  .addBcast "%b1" b₁ (.dotIn "%W1b" (wBf16 rnd W₁)
    (.convertF rnd
      (.addBcast "%b0" b₀ (.dotIn "%W0b" (wBf16 rnd W₀) (.operand "%xb" (actBf16 rnd x))))))

/-- The intended depth-2 algorithm: **`bf16Linear` applied to itself.** No explicit
    `rnd ∘` between the layers — `bf16Linear` already rounds its input operand via
    `actBf16`, and that is precisely the rounding the in-graph `convertF` performs. The
    layer-1 leaf cast and the layer-0 output round are THE SAME CAST, seen from the two
    sides, which is why the composition needs no glue. (Writing `rnd ∘ …` here instead
    would round twice and the tie below would fail — it did, first try.) -/
noncomputable def bf16Depth2 (rnd : ℝ → ℝ) {p : Nat}
    (W₀ : Mat m n) (b₀ : Vec n) (W₁ : Mat n p) (b₁ : Vec p) (x : Vec m) : Vec p :=
  bf16Linear rnd W₁ b₁ (bf16Linear rnd W₀ b₀ x)

/-- **bf16 render-tie at depth 2.** The emitted graph — including the in-graph round —
    denotes the rounded-operand two-layer linear, for any `rnd`. Still `rfl`: no
    accuracy reasoning, no error propagation, purely denotational. This is the
    statement the header flagged as needing `convertF`, and it discharges the same way
    the depth-1 one does. -/
theorem bf16_render_faithful_depth2 (rnd : ℝ → ℝ) {p : Nat}
    (W₀ : Mat m n) (b₀ : Vec n) (W₁ : Mat n p) (b₁ : Vec p) (x : Vec m) :
    den (bf16Depth2Graph rnd W₀ b₀ W₁ b₁ x) = bf16Depth2 rnd W₀ b₀ W₁ b₁ x := rfl

/-! Depth `k` follows by iterating `bf16_render_faithful_depth2` — there is no
depth-dependent constant to accumulate, which is the structural content of "bf16 has no
block scale to factor through the sum". The round node in isolation is
`Proofs.StableHLO.convertF_faithful`; it is not restated here. -/

end Proofs.Bf16PoC

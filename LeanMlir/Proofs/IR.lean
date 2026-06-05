import LeanMlir.Proofs.MLP
import LeanMlir.Proofs.CNN
import LeanMlir.Proofs.LayerNorm
import LeanMlir.Proofs.EfficientNet

/-! # A denoted StableHLO-subset IR — Phase 0a/0b spike

Spike for `planning/typed_ir.md`: give the *emitted backward graph* a
denotational semantics `⟦·⟧` landing in the proofs' own `Vec` type, then
prove the emitted graph denotes the proven `HasVJP.backward`. This turns
the per-op proof↔codegen correspondence from a comment into a theorem.

This file is the **scaffolding probe**, not the full ladder:

* **Phase 0a (dense)** — `dense_back_bridge`. Dense's input-gradient is a
  single `dot_general`, so the bridge is definitional; its job is to pin
  the `Back` / `denote` plumbing.
* **Phase 0b (relu, smooth point)** — `relu_back_bridge`. The ReLU
  backward graph is `compare(x > 0)` + `select`; at a point off the kink
  it denotes the canonical `pdiv`-derived ReLU backward. This is the real
  content: it is *conditional on smoothness* and reuses the existing
  `relu_codegen_matches_canonical`, exactly matching the codegen trust
  boundary.

Design notes (see `planning/typed_ir.md`): the backward is modelled as an
expression tree rooted at the cotangent — SSA/sharing is a
semantics-preserving printer concern (D2), so the correctness proof never
touches it. The spike uses `Vec`/`Mat` directly (D1 shortcut) rather than
the general flat-tensor type.

Everything closes under `[propext, Classical.choice, Quot.sound]` (audited
in `tests/AuditAxioms.lean`); no `native_decide`.
-/

namespace Proofs
namespace IR

-- `MaxPool2IsArgmax` is a `∀`-quantified order Prop over `ℝ`; its `if`
-- needs the classical decidability instance `CNN.lean` uses (it `open`s
-- `Classical`). Low priority, so it doesn't disturb the `Nat`/`ℝ`-order
-- decidability the dense/relu/conv bridges already rely on.
open Classical

/-- A backward subgraph, rooted at the cotangent `dy : Vec inp`, producing
    a `Vec out`. Saved forward data (weights `A`, the ReLU pre-activation
    `x`) is baked into the constructors. Each constructor models the
    StableHLO op a backward pass uses:

    * `cotangent`  — the graph input `dy`,
    * `dotGeneral` — `stablehlo.dot_general` (here: matrix · vector),
    * `selectPos`  — `stablehlo.compare GT 0` + `stablehlo.select`. -/
inductive Back (inp : Nat) : Nat → Type where
  | cotangent : Back inp inp
  | dotGeneral {m n : Nat} (A : Mat m n) : Back inp n → Back inp m
  | selectPos {n : Nat} (x : Vec n) : Back inp n → Back inp n
  | scale {n : Nat} (s : Vec n) : Back inp n → Back inp n
  | sumBroadcast {n : Nat} : Back inp n → Back inp n
  | sub {n : Nat} : Back inp n → Back inp n → Back inp n
  | scaleConst {n : Nat} (c : ℝ) : Back inp n → Back inp n

/-- **Denotational semantics** of a backward graph, into the proofs' own
    `Vec` type — so a bridge theorem can equate it with a proven
    `HasVJP.backward`. -/
noncomputable def Back.denote {inp out : Nat} (e : Back inp out) (dy : Vec inp) : Vec out :=
  match e with
  | .cotangent       => dy
  | .dotGeneral A e' => Mat.mulVec A (e'.denote dy)
  | .selectPos x e'  => fun i => if x i > 0 then e'.denote dy i else 0
  | .scale s e'      => fun i => e'.denote dy i * s i
  | .sumBroadcast e' => fun _ => ∑ j, e'.denote dy j
  | .sub e1 e2       => fun i => e1.denote dy i - e2.denote dy i
  | .scaleConst c e' => fun i => c * e'.denote dy i

/-- **Composition lemma** (the Phase-3 mechanism in miniature): a
    `dotGeneral` node denotes post-composition with `Mat.mulVec`.
    Whole-network bridges will chain lemmas of this shape, mirroring how
    `vjp_comp` builds whole-net VJPs from per-layer ones. -/
theorem denote_dotGeneral {inp m n : Nat} (A : Mat m n) (e : Back inp n) (dy : Vec inp) :
    (Back.dotGeneral A e).denote dy = Mat.mulVec A (e.denote dy) := rfl

-- ════════════════════════════════════════════════════════════════
-- § Phase 0a — dense
-- ════════════════════════════════════════════════════════════════

/-- The dense input-gradient backward graph: one `dot_general` of the
    weight matrix with the cotangent. -/
def emitDenseBack {m n : Nat} (W : Mat m n) : Back n m := .dotGeneral W .cotangent

/-- **Dense bridge.** The emitted graph denotes the proven dense backward
    `Mat.mulVec W dy`. Base case — dense's backward is a single
    `dot_general` — so this is definitional; it pins the plumbing. -/
theorem dense_back_bridge {m n : Nat} (W : Mat m n) (b : Vec n) (x : Vec m) (dy : Vec n) :
    (emitDenseBack W).denote dy = (dense_has_vjp W b).backward x dy := rfl

-- ════════════════════════════════════════════════════════════════
-- § Phase 0b — ReLU at a smooth point
-- ════════════════════════════════════════════════════════════════

/-- The ReLU backward graph: `compare(x > 0)` then `select` on the
    cotangent. `x` is the saved forward pre-activation. -/
def emitReluBack {n : Nat} (x : Vec n) : Back n n := .selectPos x .cotangent

/-- **ReLU bridge (smooth point).** At a point with no coordinate on the
    kink (`∀ k, x k ≠ 0`), the emitted compare/select graph denotes the
    canonical `pdiv`-derived ReLU backward. The real content: *conditional
    on smoothness*, reusing `relu_codegen_matches_canonical`. The
    Lean-vs-codegen gap at the kink is exactly the codegen trust
    boundary — and exactly where this equality is allowed to fail. -/
theorem relu_back_bridge {n : Nat} (x : Vec n) (h_smooth : ∀ k, x k ≠ 0)
    (dy : Vec n) (i : Fin n) :
    (emitReluBack x).denote dy i = (relu_has_vjp n).backward x dy i := by
  show (if x i > 0 then dy i else 0) = (relu_has_vjp n).backward x dy i
  exact (relu_codegen_matches_canonical n x h_smooth dy i).symm

-- ════════════════════════════════════════════════════════════════
-- § Phase 2 — convolution (the real spatial op)
--
-- The conv input-gradient is the StableHLO `convolution(dy, reverse(Wᵀ))`
-- — transpose channels, flip the kernel spatially, convolve. Under
-- `⟦conv⟧ := conv2d` (D3) that graph denotes a forward `conv2d` of the
-- reversed-swapped kernel, so the backward bridge reduces to the
-- "reversed-kernel identity" `dx = conv(dy, reverse(Wᵀ))` that `CNN.lean`
-- only *asserts* in prose (CNN.lean:288–290, "Equivalent under the partial
-- bijection …") and never proves — the repo deliberately uses the
-- (co, ho, wo) form of `conv2d_input_grad_formula` to avoid this bijection.
-- Here it is discharged by expansion at the concrete shapes the Spatial
-- instance uses (the partial-bijection-free route the repo wanted). The
-- general-shape proof is the remaining Phase-2 item.
-- ════════════════════════════════════════════════════════════════

/-- Spatial reversal of a kernel index: `k − 1 − i`. -/
def kRev {k : Nat} (i : Fin k) : Fin k := ⟨k - 1 - i.val, by omega⟩

/-- Transpose-and-flip a kernel — swap in/out channels and reverse both
    spatial axes. This is the kernel the codegen feeds to the backward
    `stablehlo.convolution` (`transpose dims [1,0,2,3]` + `reverse [2,3]`). -/
noncomputable def reverseSwap {ic oc kH kW : Nat} (W : Kernel4 oc ic kH kW) :
    Kernel4 ic oc kH kW := fun ci co kh kw => W co ci (kRev kh) (kRev kw)

/-- **Denotation of the emitted conv input-gradient graph.** The codegen
    emits `convolution(dy, reverse(transpose(W)))`; under `⟦conv⟧ := conv2d`
    (D3) that denotes a forward `conv2d` of the reversed-swapped kernel. -/
noncomputable def convBackDenote {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) : Tensor3 oc h w → Tensor3 ic h w :=
  conv2d (reverseSwap W) (fun _ => 0)

/-- **Conv backward bridge, 1→2 channels (the Spatial instance's first
    conv: `Kernel4 2 1 3 3` at 4×4).** The emitted transposed-convolution
    graph denotes the proven conv input-VJP `(conv2d_has_vjp3 W b).backward`.
    This discharges the reversed-kernel identity `CNN.lean` only asserts,
    by expansion at the concrete shape. -/
theorem conv_back_bridge_1to2 (W : Kernel4 2 1 3 3) (b : Vec 2)
    (x : Tensor3 1 (2*2) (2*2)) (dy : Tensor3 2 (2*2) (2*2)) :
    convBackDenote W dy = (conv2d_has_vjp3 W b).backward x dy := by
  show conv2d (reverseSwap W) (fun _ => 0) dy = conv2d_input_grad_formula W dy
  funext ci hi wi
  fin_cases ci <;> fin_cases hi <;> fin_cases wi <;>
    simp [conv2d, conv2d_input_grad_formula, reverseSwap, kRev, Fin.sum_univ_succ]

/-- **Conv backward bridge, 2→2 channels (the Spatial instance's second
    conv: `Kernel4 2 2 3 3` at 4×4).** Same identity at the 2→2 shape. -/
theorem conv_back_bridge_2to2 (W : Kernel4 2 2 3 3) (b : Vec 2)
    (x : Tensor3 2 (2*2) (2*2)) (dy : Tensor3 2 (2*2) (2*2)) :
    convBackDenote W dy = (conv2d_has_vjp3 W b).backward x dy := by
  show conv2d (reverseSwap W) (fun _ => 0) dy = conv2d_input_grad_formula W dy
  funext ci hi wi
  fin_cases ci <;> fin_cases hi <;> fin_cases wi <;>
    simp [conv2d, conv2d_input_grad_formula, reverseSwap, kRev, Fin.sum_univ_succ]

-- ════════════════════════════════════════════════════════════════
-- § Phase 2 — max-pool (the other kinked op)
--
-- The pooling analogue of the ReLU bridge: the codegen emits
-- tile-compare-select (broadcast `dy` and the pooled output, `compare EQ`
-- to find the argmax cells, `select` `dy` through the mask). At a smooth
-- point (every 2×2 window has a unique strict argmax) that graph routes
-- `dy` to the argmax cell — the canonical pdiv-derived maxpool backward.
-- Conditional on `MaxPool2Smooth`, reusing `maxPool2_codegen_matches_canonical`.
-- ════════════════════════════════════════════════════════════════

/-- **Denotation of the emitted maxpool input-gradient graph** (StableHLO
    tile-compare-select): at a smooth point, route `dy` to each window's
    argmax input cell, zero elsewhere. -/
noncomputable def maxPoolBackDenote {c h w : Nat} (x : Tensor3 c (2*h) (2*w)) :
    Tensor3 c h w → Tensor3 c (2*h) (2*w) :=
  fun dy ci hi_in wi_in =>
    if MaxPool2IsArgmax x ci hi_in wi_in then dy ci (winRow hi_in) (winCol wi_in) else 0

/-- **MaxPool backward bridge (smooth point).** The emitted
    tile-compare-select graph denotes the canonical pdiv-derived maxpool
    backward, *conditional on no argmax ties* (`MaxPool2Smooth`). The
    spatial-pooling analogue of `relu_back_bridge`; reuses
    `maxPool2_codegen_matches_canonical`. The Lean-vs-codegen gap at
    argmax-tie boundaries is exactly the codegen trust boundary. -/
theorem maxpool_back_bridge {c h w : Nat} (x : Tensor3 c (2*h) (2*w))
    (h_smooth : MaxPool2Smooth x) (dy : Tensor3 c h w)
    (ci : Fin c) (hi_in : Fin (2*h)) (wi_in : Fin (2*w)) :
    maxPoolBackDenote x dy ci hi_in wi_in
      = (maxPool2_has_vjp3 :
          HasVJP3 (maxPool2 : Tensor3 c (2*h) (2*w) → Tensor3 c h w)).backward
          x dy ci hi_in wi_in := by
  show (if MaxPool2IsArgmax x ci hi_in wi_in
        then dy ci (winRow hi_in) (winCol wi_in) else 0) = _
  exact (maxPool2_codegen_matches_canonical x h_smooth dy ci hi_in wi_in).symm

-- ════════════════════════════════════════════════════════════════
-- § Phase 1 — smooth elementwise activations (diagonal Jacobian)
--
-- GELU, Swish/SiLU and sigmoid all have a diagonal Jacobian, so their
-- proven `HasVJP.backward` is the closed form `dy ⊙ act'(x)` — a single
-- `stablehlo.multiply` of the cotangent against the saved
-- activation-derivative vector. The IR `scale` node models exactly that
-- multiply, so each bridge is definitional — the smooth-activation
-- analogue of `dense_back_bridge`.
--
-- (BN/LayerNorm's 3-term rank-1 backward and softmax's rank-1 backward
-- are closed forms too, but their emitted graphs are multi-op
-- reduce+elementwise — they need a `reduce`/`broadcast` IR extension, and
-- SE is compositional; those are the remaining smooth layers.)
-- ════════════════════════════════════════════════════════════════

/-- The emitted elementwise-activation backward graph: `stablehlo.multiply`
    of the cotangent with the saved activation-derivative `s = act'(x)`. -/
def emitActBack {n : Nat} (s : Vec n) : Back n n := .scale s .cotangent

/-- **GELU backward bridge.** The emitted `dy ⊙ gelu'(x)` graph denotes the
    proven GELU backward. Definitional — GELU's diagonal Jacobian makes its
    `HasVJP.backward` exactly this elementwise scaling. -/
theorem gelu_back_bridge (n : Nat) (x dy : Vec n) :
    (emitActBack (fun i => geluScalarDeriv (x i))).denote dy
      = (gelu_has_vjp n).backward x dy := rfl

/-- **Swish / SiLU backward bridge.** Same diagonal pattern. -/
theorem swish_back_bridge (n : Nat) (x dy : Vec n) :
    (emitActBack (fun i => swishScalarDeriv (x i))).denote dy
      = (swish_has_vjp n).backward x dy := rfl

/-- **Sigmoid backward bridge.** Same diagonal pattern. -/
theorem sigmoid_back_bridge (n : Nat) (x dy : Vec n) :
    (emitActBack (fun i => sigmoidScalarDeriv (x i))).denote dy
      = (sigmoid_has_vjp n).backward x dy := rfl

-- ════════════════════════════════════════════════════════════════
-- § Phase 1 — BatchNorm / LayerNorm (the rank-1 "wringer")
--
-- BN's normalize backward is the consolidated 3-term rank-1 formula
--   dxᵢ = invN·s·( N·dx̂ᵢ − Σⱼ dx̂ⱼ − x̂ᵢ·Σⱼ x̂ⱼ·dx̂ⱼ )
-- which the codegen emits as two `stablehlo.reduce` sums + broadcast +
-- elementwise subtract/scale. The IR now carries `sumBroadcast`
-- (reduce+broadcast), `sub`, and `scaleConst`; the bridge shows that graph
-- denotes the proven `bnNormalize_has_vjp.backward`. The affine half
-- (`γ·dy`) is one `scaleConst`. `bn_has_vjp = vjp_comp normalize affine`,
-- so the full BN backward is the normalize graph fed `γ ⊙ dy`. LayerNorm
-- is definitionally BN, so it inherits all of this.
-- ════════════════════════════════════════════════════════════════

/-- The emitted BN-normalize input-gradient graph, as a function of an
    input subgraph (the cotangent for normalize alone; `γ ⊙ dy` for full
    BN). Two `sumBroadcast` reductions + `sub` + `scaleConst` — exactly the
    consolidated three-term formula. -/
noncomputable def bnNormalizeBackOf {n : Nat} (xh : Vec n) (s invN : ℝ)
    (input : Back n n) : Back n n :=
  .scaleConst (invN * s)
    (.sub
      (.sub (.scaleConst (n : ℝ) input) (.sumBroadcast input))
      (.scale xh (.sumBroadcast (.scale xh input))))

/-- **BN affine backward bridge** — the `γ·dy` half is one `scaleConst`. -/
theorem bn_affine_back_bridge {n : Nat} (γ β : ℝ) (v dy : Vec n) :
    (Back.scaleConst γ Back.cotangent).denote dy
      = (bnAffine_has_vjp n γ β).backward v dy := rfl

/-- **BN normalize backward bridge — the 3-term rank-1 wringer.** The
    emitted reduce+broadcast+elementwise graph denotes the proven
    consolidated BN-normalize backward `bnNormalize_has_vjp.backward`
    (the cross-coordinate `Σ dx̂` and `Σ x̂·dx̂` reductions and the
    rank-1 `x̂ᵢ·Σx̂·dx̂` correction, matched termwise). -/
theorem bn_normalize_back_bridge {n : Nat} (ε : ℝ) (hε : 0 < ε) (x dxhat : Vec n) :
    (bnNormalizeBackOf (bnXhat n ε x) (bnIstd n x ε) (1 / (n : ℝ))
        Back.cotangent).denote dxhat
      = (bnNormalize_has_vjp n ε hε).backward x dxhat := by
  funext i
  simp only [bnNormalizeBackOf, Back.denote, bnNormalize_has_vjp]
  rw [show (∑ j, dxhat j * bnXhat n ε x j) = ∑ j, bnXhat n ε x j * dxhat j from
        Finset.sum_congr rfl (fun j _ => mul_comm _ _)]
  ring

/-- **Full BatchNorm backward bridge.** `bn_has_vjp = vjp_comp normalize
    affine`, so the emitted graph is the 3-term normalize graph fed
    `γ ⊙ dy` (the affine backward). Denotes `(bn_has_vjp …).backward`.
    The `bnForward = bnAffine ∘ bnNormalize` cast collapses by `rfl`. -/
theorem bn_back_bridge {n : Nat} (ε γ β : ℝ) (hε : 0 < ε) (x dy : Vec n) :
    (bnNormalizeBackOf (bnXhat n ε x) (bnIstd n x ε) (1 / (n : ℝ))
        (Back.scaleConst γ Back.cotangent)).denote dy
      = (bn_has_vjp n ε γ β hε).backward x dy := by
  have h : (bn_has_vjp n ε γ β hε).backward x dy
         = (bnNormalize_has_vjp n ε hε).backward x
             ((bnAffine_has_vjp n γ β).backward (bnNormalize n ε x) dy) := by
    simp only [bn_has_vjp, vjp_comp, eq_mpr_eq_cast]; rfl
  rw [h]
  funext i
  simp only [bnNormalizeBackOf, Back.denote, bnNormalize_has_vjp, bnAffine_has_vjp]
  rw [show (∑ j, γ * dy j * bnXhat n ε x j) = ∑ j, bnXhat n ε x j * (γ * dy j) from
        Finset.sum_congr rfl (fun j _ => mul_comm _ _)]
  ring

/-- **LayerNorm backward bridge — free.** `layerNorm_has_vjp` is
    definitionally `bn_has_vjp` (LayerNorm is BN on a different axis), so
    the same emitted graph denotes its backward. -/
theorem layernorm_back_bridge {n : Nat} (ε γ β : ℝ) (hε : 0 < ε) (x dy : Vec n) :
    (bnNormalizeBackOf (bnXhat n ε x) (bnIstd n x ε) (1 / (n : ℝ))
        (Back.scaleConst γ Back.cotangent)).denote dy
      = (layerNorm_has_vjp n ε γ β hε).backward x dy :=
  bn_back_bridge ε γ β hε x dy

end IR
end Proofs

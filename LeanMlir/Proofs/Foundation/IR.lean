import LeanMlir.Proofs.Foundation.MLP
import LeanMlir.Proofs.CNN
import LeanMlir.Proofs.LayerNorm
import LeanMlir.Proofs.EfficientNet
import LeanMlir.Proofs.Attention
import LeanMlir.Proofs.SE

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
  | add {n : Nat} : Back inp n → Back inp n → Back inp n

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
  | .add e1 e2       => fun i => e1.denote dy i + e2.denote dy i

/-- **Composition lemma** (the Phase-3 mechanism in miniature): a
    `dotGeneral` node denotes post-composition with `Mat.mulVec`.
    Whole-network bridges will chain lemmas of this shape, mirroring how
    `vjp_comp` builds whole-net VJPs from per-layer ones. -/
theorem denote_dotGeneral {inp m n : Nat} (A : Mat m n) (e : Back inp n) (dy : Vec inp) :
    (Back.dotGeneral A e).denote dy = Mat.mulVec A (e.denote dy) := rfl

-- ════════════════════════════════════════════════════════════════
-- § Phase 3 — composition (the IR-level chain rule)
--
-- A backward graph is rooted at the cotangent leaf; composing two layers'
-- backwards means plugging one graph into the other's cotangent. `subst`
-- does that, and `denote_subst` proves it denotes the composition of the
-- denotations — the IR analogue of `vjp_comp`/`vjp_comp_at`. This is the
-- mechanism that assembles per-op bridges into a whole-network bridge.
-- ════════════════════════════════════════════════════════════════

/-- Plug the backward graph `g` into the cotangent leaf of `e`. For a
    composite `g_layer ∘ f_layer`, `e` is `f_layer`'s backward and `g` is
    `g_layer`'s, giving the composite's backward. -/
def Back.subst {inp inp' out : Nat} (e : Back inp out) (g : Back inp' inp) : Back inp' out :=
  match e with
  | .cotangent       => g
  | .dotGeneral A e' => .dotGeneral A (e'.subst g)
  | .selectPos x e'  => .selectPos x (e'.subst g)
  | .scale s e'      => .scale s (e'.subst g)
  | .sumBroadcast e' => .sumBroadcast (e'.subst g)
  | .sub e1 e2       => .sub (e1.subst g) (e2.subst g)
  | .scaleConst c e' => .scaleConst c (e'.subst g)
  | .add e1 e2       => .add (e1.subst g) (e2.subst g)

/-- **IR-level chain rule.** `subst` denotes the composition of
    denotations: `⟦e[g/cotangent]⟧ dz = ⟦e⟧ (⟦g⟧ dz)`. The analogue of
    `vjp_comp` — chains per-op bridges into a whole-network bridge. -/
theorem denote_subst {inp inp' out : Nat} (e : Back inp out) (g : Back inp' inp)
    (dz : Vec inp') : (e.subst g).denote dz = e.denote (g.denote dz) := by
  induction e with
  | cotangent => rfl
  | dotGeneral A e' ih => simp only [Back.subst, Back.denote, ih]
  | selectPos x e' ih => simp only [Back.subst, Back.denote, ih]
  | scale s e' ih => simp only [Back.subst, Back.denote, ih]
  | sumBroadcast e' ih => simp only [Back.subst, Back.denote, ih]
  | sub e1 e2 ih1 ih2 => simp only [Back.subst, Back.denote, ih1, ih2]
  | scaleConst c e' ih => simp only [Back.subst, Back.denote, ih]
  | add e1 e2 ih1 ih2 => simp only [Back.subst, Back.denote, ih1, ih2]

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

/-- **The general conv-adjoint identity (odd kernels), all dims.** The emitted
    reversed-kernel forward conv `conv2d (reverseSwap W) 0` equals the certified
    conv input-gradient `conv2d_input_grad_formula W`, for ARBITRARY
    `ic oc h w kH kW` with odd `kH`, `kW` (`2·⌊(kH-1)/2⌋+1 = kH`). This is the
    reversed-kernel ⇒ correlation-adjoint reindex that `conv_back_bridge_{1to2,2to2}`
    previously asserted only at two toy 4×4 shapes by exhaustive `fin_cases`.

    Proof: per output coordinate, both sides sum over the input channel `co`; the
    inner `(kh,kw)` sum (LHS, over the kernel window) and the `(ho,wo)` sum (RHS,
    over output positions) range over the SAME set of valid alignments via the
    partial bijection `(kh,kw) ↦ (kh+hi-pH, kw+wi-pW)` on the pad supports. Under
    oddness `2·pH = kH-1`, the reversed-kernel index `kH-1-kh` matches the formula's
    `hi+pH-ho`, and the data indices coincide — so the matched summands are equal.
    `Finset.sum_bij'` over the pad-filtered supports; all index arithmetic by `omega`.

    The single load-bearing leaf for the §B certified-VJP tie: every conv-heavy
    net's backward (`convFlatBack`) routes its conv input-grad through this. -/
theorem convBackDenote_eq_input_grad_formula {ic oc h w kH kW : Nat}
    (hkH : 2 * ((kH - 1) / 2) + 1 = kH) (hkW : 2 * ((kW - 1) / 2) + 1 = kW)
    (W : Kernel4 oc ic kH kW) (dy : Tensor3 oc h w) :
    conv2d (reverseSwap W) (fun _ => 0) dy = conv2d_input_grad_formula W dy := by
  funext ci hi wi
  simp only [conv2d, reverseSwap, kRev, zero_add, conv2d_input_grad_formula]
  apply Finset.sum_congr rfl
  intro co _
  rw [← Finset.sum_product', ← Finset.sum_product', Finset.univ_product_univ,
      Finset.univ_product_univ]
  rw [← Finset.sum_subset (Finset.filter_subset
        (fun p : Fin kH × Fin kW => (kH-1)/2 ≤ p.1.val + hi.val ∧ p.1.val + hi.val - (kH-1)/2 < h ∧
             (kW-1)/2 ≤ p.2.val + wi.val ∧ p.2.val + wi.val - (kW-1)/2 < w) Finset.univ) ?lv,
      ← Finset.sum_subset (Finset.filter_subset
        (fun q : Fin h × Fin w => q.1.val ≤ hi.val + (kH-1)/2 ∧ hi.val + (kH-1)/2 - q.1.val < kH ∧
             q.2.val ≤ wi.val + (kW-1)/2 ∧ wi.val + (kW-1)/2 - q.2.val < kW) Finset.univ) ?rv]
  case lv =>
    intro p _ hp
    rw [Finset.mem_filter] at hp
    rw [dif_neg (fun hpr => hp ⟨Finset.mem_univ p, hpr⟩), mul_zero]
  case rv =>
    intro q _ hq
    rw [Finset.mem_filter] at hq
    rw [dif_neg (fun hpr => hq ⟨Finset.mem_univ q, hpr⟩)]
  -- the partial bijection on the pad supports
  refine Finset.sum_bij'
    (fun p hp => ((⟨p.1.val + hi.val - (kH-1)/2, by
        have := (Finset.mem_filter.mp hp).2; omega⟩ : Fin h),
       (⟨p.2.val + wi.val - (kW-1)/2, by
        have := (Finset.mem_filter.mp hp).2; omega⟩ : Fin w)))
    (fun q _ => ((⟨kH - 1 - (hi.val + (kH-1)/2 - q.1.val), by omega⟩ : Fin kH),
       (⟨kW - 1 - (wi.val + (kW-1)/2 - q.2.val), by omega⟩ : Fin kW)))
    ?hi ?hj ?linv ?rinv ?heq
  case hi =>
    intro p hp
    have hb := (Finset.mem_filter.mp hp).2
    have := p.1.isLt; have := p.2.isLt
    rw [Finset.mem_filter]
    refine ⟨Finset.mem_univ _, ?_, ?_, ?_, ?_⟩ <;> simp only <;> omega
  case hj =>
    intro q hq
    have hb := (Finset.mem_filter.mp hq).2
    have := q.1.isLt; have := q.2.isLt
    rw [Finset.mem_filter]
    refine ⟨Finset.mem_univ _, ?_, ?_, ?_, ?_⟩ <;> simp only <;> omega
  case linv =>
    intro p hp
    have hb := (Finset.mem_filter.mp hp).2
    have := p.1.isLt; have := p.2.isLt
    apply Prod.ext <;> apply Fin.ext <;> simp only <;> omega
  case rinv =>
    intro q hq
    have hb := (Finset.mem_filter.mp hq).2
    have := q.1.isLt; have := q.2.isLt
    apply Prod.ext <;> apply Fin.ext <;> simp only <;> omega
  case heq =>
    intro p hp
    have hb := (Finset.mem_filter.mp hp).2
    have h1 := p.1.isLt; have h2 := p.2.isLt
    rw [dif_pos hb, dif_pos (by refine ⟨?_, ?_, ?_, ?_⟩ <;> simp only <;> omega)]
    dsimp only
    have ea : kH - 1 - p.1.val = hi.val + (kH - 1) / 2 - (p.1.val + hi.val - (kH - 1) / 2) := by omega
    have eb : kW - 1 - p.2.val = wi.val + (kW - 1) / 2 - (p.2.val + wi.val - (kW - 1) / 2) := by omega
    simp only [ea, eb]

/-- **Conv backward bridge, 1→2 channels (the Spatial instance's first
    conv: `Kernel4 2 1 3 3` at 4×4).** The emitted transposed-convolution
    graph denotes the proven conv input-VJP `(conv2d_has_vjp3 W b).backward`.
    Now a one-line instance of the general `convBackDenote_eq_input_grad_formula`
    (3×3 is odd) — no longer the brute-force `fin_cases` expansion. -/
theorem conv_back_bridge_1to2 (W : Kernel4 2 1 3 3) (b : Vec 2)
    (x : Tensor3 1 (2*2) (2*2)) (dy : Tensor3 2 (2*2) (2*2)) :
    convBackDenote W dy = (conv2d_has_vjp3 W b).backward x dy := by
  show conv2d (reverseSwap W) (fun _ => 0) dy = conv2d_input_grad_formula W dy
  exact convBackDenote_eq_input_grad_formula (by decide) (by decide) W dy

/-- **Conv backward bridge, 2→2 channels (the Spatial instance's second
    conv: `Kernel4 2 2 3 3` at 4×4).** Same identity at the 2→2 shape — also a
    one-line instance of the general lemma. -/
theorem conv_back_bridge_2to2 (W : Kernel4 2 2 3 3) (b : Vec 2)
    (x : Tensor3 2 (2*2) (2*2)) (dy : Tensor3 2 (2*2) (2*2)) :
    convBackDenote W dy = (conv2d_has_vjp3 W b).backward x dy := by
  show conv2d (reverseSwap W) (fun _ => 0) dy = conv2d_input_grad_formula W dy
  exact convBackDenote_eq_input_grad_formula (by decide) (by decide) W dy

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

-- ════════════════════════════════════════════════════════════════
-- § Phase 1 — softmax (rank-1, like BN)
--
-- softmax's backward is the rank-1 `dzᵢ = pᵢ·(dyᵢ − ⟨p, dy⟩)` (one
-- reduction `⟨p, dy⟩` + a broadcast-subtract + a scale by `p`), the same
-- optimization shape as BN. With the reduce/broadcast IR in place it is
-- `scale p (sub cotangent (sumBroadcast (scale p cotangent)))`.
-- ════════════════════════════════════════════════════════════════

/-- The emitted softmax input-gradient graph: scale by `p`, subtract the
    broadcast inner product `⟨p, dy⟩`, scale by `p`. -/
noncomputable def emitSoftmaxBack {c : Nat} (p : Vec c) : Back c c :=
  .scale p (.sub .cotangent (.sumBroadcast (.scale p .cotangent)))

/-- **Softmax backward bridge.** The emitted reduce+broadcast+scale graph
    denotes the proven rank-1 softmax backward `pᵢ·(dyᵢ − ⟨p, dy⟩)`. -/
theorem softmax_back_bridge (c : Nat) (z dy : Vec c) :
    (emitSoftmaxBack (softmax c z)).denote dy = (softmax_has_vjp c).backward z dy := by
  funext i
  simp only [emitSoftmaxBack, Back.denote, softmax_has_vjp]
  rw [show (∑ j, dy j * softmax c z j) = ∑ j, softmax c z j * dy j from
        Finset.sum_congr rfl (fun j _ => mul_comm _ _)]
  ring

-- ════════════════════════════════════════════════════════════════
-- § Phase 3 — whole-network bridge (demonstrator)
--
-- Assemble per-op bridges into a composite via `denote_subst`. Two dense
-- layers `dense W₂ ∘ dense W₁`: the IR `subst` of their per-layer backward
-- graphs denotes the proven composite backward `(vjp_comp …).backward`.
-- This is the assembly pattern a full whole-network bridge uses;
-- `denote_subst` chains it to arbitrary depth. (Reaching the full
-- `mnistCnnNoBn` additionally needs a Tensor3 IR for conv/maxpool and the
-- `HasVJPAt` smooth-point variants; SE needs an `add` fan-in node.)
-- ════════════════════════════════════════════════════════════════

/-- **End-to-end composition bridge.** The IR `subst` of two dense layers'
    backward graphs denotes the proven composite VJP `(vjp_comp …).backward`
    — `denote_subst` (IR chain rule) ∘ the per-op `dense` bridge. -/
theorem twoDense_back_bridge {d₀ d₁ d₂ : Nat}
    (W₁ : Mat d₀ d₁) (b₁ : Vec d₁) (W₂ : Mat d₁ d₂) (b₂ : Vec d₂)
    (x : Vec d₀) (dz : Vec d₂) :
    ((emitDenseBack W₁).subst (emitDenseBack W₂)).denote dz
      = (vjp_comp (dense W₁ b₁) (dense W₂ b₂)
          (dense_differentiable W₁ b₁) (dense_differentiable W₂ b₂)
          (dense_has_vjp W₁ b₁) (dense_has_vjp W₂ b₂)).backward x dz := by
  rw [denote_subst]
  simp only [vjp_comp, emitDenseBack, Back.denote, dense_has_vjp]

-- ════════════════════════════════════════════════════════════════
-- § Squeeze-and-Excitation (the fan-in case)
--
-- `seBlock gate = elemwiseProduct id gate`, so its backward is a fan-in:
--   dxᵢ = gate(x)ᵢ·dyᵢ + gate_backward(x ⊙ dy)ᵢ
-- — `stablehlo.add` of (a) `scale` by the gate output and (b) the gate's
-- own backward graph fed `x ⊙ dy`. This is where the IR chain rule
-- (`denote_subst`, to plug the gate's backward graph `bg` in) and the new
-- `add` fan-in node both earn their keep. Generic in the gate: given the
-- gate's per-op bridge `hbg`, SE's bridge follows. -/
-- ════════════════════════════════════════════════════════════════

/-- **Squeeze-and-Excitation backward bridge.** Given the gate's backward
    graph `bg` bridged to its proven backward at `x` (`hbg`), the emitted SE
    backward graph — `add (scale (gate x) dy) (bg[x ⊙ dy])` — denotes the
    proven `seBlock_has_vjp.backward`. The fan-in + `denote_subst` (to plug
    the gate graph in) assemble the per-op bridges through a non-composition
    combinator. -/
theorem se_back_bridge {n : Nat} (gate : Vec n → Vec n)
    (hg_diff : Differentiable ℝ gate) (hg : HasVJP gate)
    (bg : Back n n) (x dy : Vec n) (hbg : ∀ z, bg.denote z = hg.backward x z) :
    (Back.add (Back.scale (gate x) Back.cotangent)
        (bg.subst (Back.scale x Back.cotangent))).denote dy
      = (seBlock_has_vjp gate hg_diff hg).backward x dy := by
  funext i
  simp only [Back.denote, seBlock_has_vjp, elemwiseProduct_has_vjp, identity_has_vjp]
  rw [denote_subst]
  simp only [Back.denote]
  rw [hbg, show (fun j => dy j * x j) = (fun j => x j * dy j) from
        funext (fun j => mul_comm _ _)]
  ring

-- ════════════════════════════════════════════════════════════════
-- § Tensor3 IR — lifting conv/maxpool into a composable backward graph
--
-- conv2d and maxPool2 are Tensor3 → Tensor3, so they don't fit the Vec
-- `Back`. `Back3` is the Tensor3 analogue: a backward graph rooted at the
-- (Tensor3) cotangent, with `conv` (transposed-conv backward) and
-- `maxpool` (route-to-argmax) nodes whose denotations are the already-proven
-- `convBackDenote`/`maxPoolBackDenote`. `denote_subst3` is the Tensor3
-- chain rule, so conv/maxpool now compose — the Tensor3 half of what
-- `Back`/`denote_subst` give the Vec layers.
-- ════════════════════════════════════════════════════════════════

/-- Tensor3-level backward graph: indexed by the top cotangent shape
    `(c₁ h₁ w₁)` and the current shape `(c₂ h₂ w₂)` (walking toward the
    input gradient). -/
inductive Back3 (c₁ h₁ w₁ : Nat) : Nat → Nat → Nat → Type where
  | cot : Back3 c₁ h₁ w₁ c₁ h₁ w₁
  | conv {ic oc h w kH kW : Nat} (W : Kernel4 oc ic kH kW) :
      Back3 c₁ h₁ w₁ oc h w → Back3 c₁ h₁ w₁ ic h w
  | maxpool {c h w : Nat} (x : Tensor3 c (2*h) (2*w)) :
      Back3 c₁ h₁ w₁ c h w → Back3 c₁ h₁ w₁ c (2*h) (2*w)

/-- Denote a `Back3` graph as a `Tensor3 → Tensor3` function, via the
    per-op Tensor3 backward denotations. -/
noncomputable def Back3.denote {c₁ h₁ w₁ c₂ h₂ w₂ : Nat}
    (e : Back3 c₁ h₁ w₁ c₂ h₂ w₂) (dy : Tensor3 c₁ h₁ w₁) : Tensor3 c₂ h₂ w₂ :=
  match e with
  | .cot          => dy
  | .conv W e'    => convBackDenote W (e'.denote dy)
  | .maxpool x e' => maxPoolBackDenote x (e'.denote dy)

/-- Plug `g` into the cotangent leaf of `e` (Tensor3 composition). -/
def Back3.subst {c₁ h₁ w₁ c₀ h₀ w₀ c₂ h₂ w₂ : Nat}
    (e : Back3 c₁ h₁ w₁ c₂ h₂ w₂) (g : Back3 c₀ h₀ w₀ c₁ h₁ w₁) :
    Back3 c₀ h₀ w₀ c₂ h₂ w₂ :=
  match e with
  | .cot          => g
  | .conv W e'    => .conv W (e'.subst g)
  | .maxpool x e' => .maxpool x (e'.subst g)

/-- **Tensor3 chain rule** — the `Back3` analogue of `denote_subst`. -/
theorem denote_subst3 {c₁ h₁ w₁ c₀ h₀ w₀ c₂ h₂ w₂ : Nat}
    (e : Back3 c₁ h₁ w₁ c₂ h₂ w₂) (g : Back3 c₀ h₀ w₀ c₁ h₁ w₁)
    (dz : Tensor3 c₀ h₀ w₀) : (e.subst g).denote dz = e.denote (g.denote dz) := by
  induction e with
  | cot => rfl
  | conv W e' ih => simp only [Back3.subst, Back3.denote, ih]
  | maxpool x e' ih => simp only [Back3.subst, Back3.denote, ih]

/-- The `Back3` maxpool node denotes the proven pointwise maxpool backward
    `maxPool2_has_vjp_at3` — `maxPoolBackDenote` *is* that backward. -/
theorem maxpool3_node_bridge {c h w : Nat} (x : Tensor3 c (2*h) (2*w))
    (h_smooth : MaxPool2Smooth x) (dy : Tensor3 c h w) :
    (Back3.maxpool x Back3.cot).denote dy = (maxPool2_has_vjp_at3 x h_smooth).backward dy := by
  funext ci hi wi
  simp only [Back3.denote, maxPoolBackDenote, maxPool2_has_vjp_at3]

/-- The `Back3` conv node denotes the proven conv backward, at the Spatial
    instance's `1→2` conv shape (via `conv_back_bridge_1to2`). -/
theorem conv3_node_bridge_1to2 (W : Kernel4 2 1 3 3) (b : Vec 2)
    (x : Tensor3 1 (2*2) (2*2)) (dy : Tensor3 2 (2*2) (2*2)) :
    (Back3.conv W Back3.cot).denote dy = (conv2d_has_vjp3 W b).backward x dy := by
  simp only [Back3.denote]
  exact conv_back_bridge_1to2 W b x dy

/-- **Tensor3 composition demonstrator.** The `Back3` `subst` of two conv
    layers' backward graphs denotes the composition of their Tensor3
    backwards, via the Tensor3 chain rule `denote_subst3` — the conv/maxpool
    analogue of `twoDense_back_bridge`. -/
theorem conv_compose3 {ic mc oc h w kH₁ kW₁ kH₂ kW₂ : Nat}
    (W₁ : Kernel4 mc ic kH₁ kW₁) (W₂ : Kernel4 oc mc kH₂ kW₂) (dz : Tensor3 oc h w) :
    ((Back3.conv (h := h) (w := w) W₁ Back3.cot).subst
        (Back3.conv (h := h) (w := w) W₂ Back3.cot)).denote dz
      = convBackDenote W₁ (convBackDenote W₂ dz) := by
  rw [denote_subst3]; simp only [Back3.denote]

-- ════════════════════════════════════════════════════════════════
-- § Flatten bridge — `Back3` (Tensor3) into flattened Vec space
--
-- `mnistCnnNoBn` runs in flattened Vec space (`flatConv`, `maxPoolFlat`),
-- so the connective step is to view a `Back3` graph through the
-- `Tensor3.flatten` bijection and show it denotes the proven *flattened*
-- layer backward (`hasVJP3_to_hasVJP` / `maxPoolFlat_has_vjp_at`). With
-- this, the Tensor3 conv/maxpool and the Vec dense/relu speak the same
-- (Vec) language and can be chained.
-- ════════════════════════════════════════════════════════════════

/-- View a `Back3` graph in flattened Vec space: `flatten ∘ denote ∘ unflatten`. -/
noncomputable def Back3.flatDenote {c₁ h₁ w₁ c₂ h₂ w₂ : Nat}
    (e : Back3 c₁ h₁ w₁ c₂ h₂ w₂) (dy : Vec (c₁ * h₁ * w₁)) : Vec (c₂ * h₂ * w₂) :=
  Tensor3.flatten (e.denote (Tensor3.unflatten dy))

/-- **Flatten bridge, max-pool.** The flattened `Back3` maxpool graph
    denotes the proven flattened maxpool layer backward
    `maxPoolFlat_has_vjp_at` (the form `mnistCnnNoBn` composes). -/
theorem maxpool_flatten_bridge {c h w : Nat} (x : Tensor3 c (2*h) (2*w))
    (h_smooth : MaxPool2Smooth x) (dy : Vec (c * h * w)) :
    (Back3.maxpool x Back3.cot).flatDenote dy
      = (maxPoolFlat_has_vjp_at x h_smooth).backward dy := by
  funext idx
  simp only [Back3.flatDenote, Back3.denote, maxPoolFlat_has_vjp_at,
             hasVJPAt3_to_hasVJPAt, maxPoolBackDenote, maxPool2_has_vjp_at3,
             Tensor3.flatten]

/-- **Flatten bridge, conv (Spatial `1→2` shape).** The flattened `Back3`
    conv graph denotes the proven flattened conv layer backward
    `hasVJP3_to_hasVJP (conv2d_has_vjp3 W b)` — chains `conv_back_bridge_1to2`
    (the reversed-kernel identity) with the `Tensor3.flatten` decode. -/
theorem conv_flatten_bridge_1to2 (W : Kernel4 2 1 3 3) (b : Vec 2)
    (v : Vec (1 * (2*2) * (2*2))) (dy : Vec (2 * (2*2) * (2*2))) :
    (Back3.conv W Back3.cot).flatDenote dy
      = (hasVJP3_to_hasVJP (conv2d_has_vjp3 W b)).backward v dy := by
  funext idx
  simp only [Back3.flatDenote, Back3.denote, hasVJP3_to_hasVJP, Tensor3.flatten]
  rw [conv_back_bridge_1to2 W b (Tensor3.unflatten v) (Tensor3.unflatten dy)]

-- ════════════════════════════════════════════════════════════════
-- § `HasVJPAt` (smooth-point) variants — the form whole-network VJPs use
--
-- `mnistCnnNoBn_has_vjp_at` composes layers via `vjp_comp_at` over their
-- `HasVJPAt` instances. The leaves bridge trivially: `relu_has_vjp_at`'s
-- backward IS the `compare`/`select` formula (rfl), and
-- `(dense_has_vjp).toHasVJPAt` just wraps the global. The payoff is the
-- block: the IR `subst` of the dense + relu backward graphs denotes the
-- `vjp_comp_at` block backward — the actual `mnistCnnNoBn` building block.
-- (conv/maxpool `_at` forms are the flatten bridges above.)
-- ════════════════════════════════════════════════════════════════

/-- **ReLU `_at` bridge.** The `compare`/`select` graph denotes the pointwise
    `relu_has_vjp_at` backward directly — definitional (no canonical sum). -/
theorem relu_at_bridge (n : Nat) (x : Vec n) (h_smooth : ∀ k, x k ≠ 0) (dy : Vec n) :
    (emitReluBack x).denote dy = (relu_has_vjp_at n x h_smooth).backward dy := rfl

/-- **Dense `_at` bridge.** `(dense_has_vjp).toHasVJPAt` wraps the global
    instance, so the dense graph still denotes it (rfl). -/
theorem dense_at_bridge {m n : Nat} (W : Mat m n) (b : Vec n) (v : Vec m) (dy : Vec n) :
    (emitDenseBack W).denote dy = ((dense_has_vjp W b).toHasVJPAt v).backward dy := rfl

/-- **Dense→ReLU block `_at` bridge.** The IR `subst` of the dense and relu
    backward graphs denotes the proven `vjp_comp_at` block backward — a real
    `mnistCnnNoBn` building block, assembled from the per-op `_at` bridges via
    `denote_subst`. -/
theorem denseRelu_at_bridge {m n : Nat} (W : Mat m n) (b : Vec n) (v : Vec m)
    (h_smooth : ∀ k, dense W b v k ≠ 0) (dy : Vec n) :
    ((emitDenseBack W).subst (emitReluBack (dense W b v))).denote dy
      = (vjp_comp_at (dense W b) (relu n) v
          ((dense_differentiable W b) v)
          (relu_differentiableAt_of_smooth n _ h_smooth)
          ((dense_has_vjp W b).toHasVJPAt v)
          (relu_has_vjp_at n _ h_smooth)).backward dy := by
  rw [denote_subst]
  simp only [vjp_comp_at, emitDenseBack, emitReluBack, Back.denote,
             HasVJP.toHasVJPAt, dense_has_vjp, relu_has_vjp_at]

-- ════════════════════════════════════════════════════════════════
-- § Final assembly — a whole-network bridge
--
-- `mlpForward = dense W₂ ∘ relu ∘ dense W₁ ∘ relu ∘ dense W₀` is a genuine
-- whole network (all in Vec), and `mlp_has_vjp_at` is its proven VJP at a
-- smooth point (built by chaining `vjp_comp_at`). The emitted whole
-- backward is a single Vec `subst` chain of the per-op graphs; the bridge
-- shows it denotes `mlp_has_vjp_at.backward` — every per-op `_at` bridge
-- assembled through `denote_subst` into one machine-checked statement that
-- the emitted StableHLO backward graph computes the proven whole-network VJP.
-- ════════════════════════════════════════════════════════════════

/-- The emitted backward graph for the whole MLP: the `subst` chain
    `dense₀ ∘ relu(p₀) ∘ dense₁ ∘ relu(p₁) ∘ dense₂` (backward order), where
    `p₀ = dense W₀ b₀ x`, `p₁ = dense W₁ b₁ (relu (dense W₀ b₀ x))` are the
    ReLU pre-activations. -/
noncomputable def emitMlpBack {d₀ d₁ d₂ d₃ : Nat}
    (W₀ : Mat d₀ d₁) (W₁ : Mat d₁ d₂) (W₂ : Mat d₂ d₃)
    (p₀ : Vec d₁) (p₁ : Vec d₂) : Back d₃ d₀ :=
  (emitDenseBack W₀).subst
    ((emitReluBack p₀).subst
      ((emitDenseBack W₁).subst
        ((emitReluBack p₁).subst (emitDenseBack W₂))))

/-- **Whole-network bridge.** The emitted MLP backward graph denotes the
    proven `mlp_has_vjp_at.backward` — the full assembly: per-op `_at`
    bridges chained through `denote_subst`, matching the nested `vjp_comp_at`.
    A machine-checked statement that the emitted backward graph computes the
    proven whole-network VJP at a smooth point. -/
theorem mlp_whole_bridge {d₀ d₁ d₂ d₃ : Nat}
    (W₀ : Mat d₀ d₁) (b₀ : Vec d₁) (W₁ : Mat d₁ d₂) (b₁ : Vec d₂)
    (W₂ : Mat d₂ d₃) (b₂ : Vec d₃) (x : Vec d₀)
    (h_smooth_0 : ∀ k, dense W₀ b₀ x k ≠ 0)
    (h_smooth_1 : ∀ k, dense W₁ b₁ (relu d₁ (dense W₀ b₀ x)) k ≠ 0) (dy : Vec d₃) :
    (emitMlpBack W₀ W₁ W₂ (dense W₀ b₀ x)
        (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x)))).denote dy
      = (mlp_has_vjp_at W₀ b₀ W₁ b₁ W₂ b₂ x h_smooth_0 h_smooth_1).backward dy := by
  simp only [emitMlpBack, denote_subst, mlp_has_vjp_at, vjp_comp_at, Back.denote,
             emitDenseBack, emitReluBack, HasVJP.toHasVJPAt, dense_has_vjp, relu_has_vjp_at,
             id_eq, Function.comp_apply]
  rfl

-- ════════════════════════════════════════════════════════════════
-- § Parameter gradients — weight/bias backward (the train-step pieces)
--
-- The input-gradient bridges above carry the cotangent *through* the net
-- (`dx`). A train step also needs the *parameter* gradients at each dense
-- layer: given the cotangent `dyℓ` arriving at that layer's output (computed
-- by a backward subgraph `e`) and the layer's saved forward input `x`,
--   dW = outer(x, dyℓ)   — a `dot_general` contracting the batch axis,
--   db = dyℓ             — a `reduce`-add over the batch axis.
-- These are the proven `dense_weight_grad`/`dense_bias_grad`, which
-- `dense_weight_grad_correct`/`dense_bias_grad_correct` certify *are* the
-- cotangent-contracted Jacobians of the dense layer wrt `W` and `b`.
-- Emitting them off the (already-bridged) backward chain promotes the
-- input-gradient bridge to a full **train-step** bridge: every gradient the
-- optimizer consumes is the rendering of a proof-backed quantity.
-- ════════════════════════════════════════════════════════════════

/-- Weight-gradient emitter: the outer product of the dense layer's saved
    forward input `x` with the cotangent at the layer's output (the
    denotation of the backward subgraph `e`). Mirrors the `dot_general`
    that contracts the batch axis (`dW = xᵀ · dy`). -/
noncomputable def emitWeightGrad {inp m n : Nat} (x : Vec m) (e : Back inp n)
    (dy : Vec inp) : Mat m n :=
  Mat.outer x (e.denote dy)

/-- Bias-gradient emitter: the cotangent at the layer's output. Mirrors the
    `reduce`-add over the batch axis (`db = Σ_batch dy`). -/
noncomputable def emitBiasGrad {inp n : Nat} (e : Back inp n) (dy : Vec inp) : Vec n :=
  e.denote dy

/-- **Weight-gradient bridge.** The emitted outer-product graph, fed the
    cotangent the backward subgraph `e` delivers, computes coordinate-wise
    the cotangent-contracted Jacobian of the dense layer wrt `W` — the
    proven `dense_weight_grad`. Certified by `dense_weight_grad_correct` at
    the *actual* chain cotangent `e.denote dy`, so it composes with any of
    the input-gradient bridges above. -/
theorem weight_grad_bridge {inp m n : Nat} (W : Mat m n) (b : Vec n) (x : Vec m)
    (e : Back inp n) (dy : Vec inp) (i : Fin m) (j : Fin n) :
    emitWeightGrad x e dy i j
      = ∑ k : Fin n,
          pdiv (fun v : Vec (m * n) => dense (Mat.unflatten v) b x)
               (Mat.flatten W) (finProdFinEquiv (i, j)) k * (e.denote dy) k :=
  dense_weight_grad_correct W b x (e.denote dy) i j

/-- **Bias-gradient bridge.** The emitted graph (the cotangent itself,
    reduce-summed over the batch) computes the cotangent-contracted Jacobian
    of the dense layer wrt `b` — the proven `dense_bias_grad`. Certified by
    `dense_bias_grad_correct`. -/
theorem bias_grad_bridge {inp m n : Nat} (W : Mat m n) (b : Vec n) (x : Vec m)
    (e : Back inp n) (dy : Vec inp) (i : Fin n) :
    emitBiasGrad e dy i
      = ∑ j : Fin n, pdiv (fun b' : Vec n => dense W b' x) b i j * (e.denote dy) j :=
  dense_bias_grad_correct W b x (e.denote dy) i

/-- The backward subgraph delivering the cotangent at the MLP's **layer-1**
    dense output: `relu'(p₁) ⊙ (W₂ · dy)` — ReLU-back composed with the
    layer-2 dense input-gradient. (Layer 2's output cotangent is the top
    `cotangent`; layer 0's prepends another `relu'(p₀) ⊙ (W₁ · ·)`.) -/
def mlpCotOut1 {d₂ d₃ : Nat} (W₂ : Mat d₂ d₃) (p₁ : Vec d₂) : Back d₃ d₂ :=
  (emitReluBack p₁).subst (emitDenseBack W₂)

/-- **MLP hidden-layer parameter-gradient bridge (representative).** At the
    interesting layer — layer 1, whose output cotangent is a genuine
    backward subgraph `mlpCotOut1`, not just the top cotangent — the emitted
    weight and bias gradients equal the certified Jacobians of that dense
    layer wrt `W₁`/`b₁`, contracted with the cotangent the backward chain
    actually delivers there. Instantiates the generic bridges at the MLP's
    layer-1 subgraph; the other two layers are the same bridges at
    `Back.cotangent` (layer 2) and `mlpCotOut0` (layer 0). -/
theorem mlp_layer1_weight_grad_bridge {d₁ d₂ d₃ : Nat}
    (W₁ : Mat d₁ d₂) (b₁ : Vec d₂) (W₂ : Mat d₂ d₃) (x₁ : Vec d₁) (p₁ : Vec d₂)
    (dy : Vec d₃) (i : Fin d₁) (j : Fin d₂) :
    emitWeightGrad x₁ (mlpCotOut1 W₂ p₁) dy i j
      = ∑ k : Fin d₂,
          pdiv (fun v : Vec (d₁ * d₂) => dense (Mat.unflatten v) b₁ x₁)
               (Mat.flatten W₁) (finProdFinEquiv (i, j)) k
            * ((mlpCotOut1 W₂ p₁).denote dy) k :=
  weight_grad_bridge W₁ b₁ x₁ (mlpCotOut1 W₂ p₁) dy i j

-- ════════════════════════════════════════════════════════════════
-- § Forward IR — the other half of the train step (Phase 2)
--
-- `Back` denotes the backward (cotangent → gradients); `Fwd` is its mirror
-- for the *forward* pass (input → output). With it, the forward StableHLO
-- the train step recomputes is no longer just trusted: it is the rendering
-- of `emitMlpFwd`, whose denotation is *proven* equal to the proven forward
-- map `mlpForward` (`mlp_fwd_bridge`). The whole MLP module — forward AND
-- backward AND parameter gradients — is then the rendering of proof-backed
-- IR (only the SGD arithmetic and the printer/IREE/float stay trusted).
-- `Fwd.subst` / `denote_subst_fwd` give the forward chain rule, mirroring
-- `Back.subst` / `denote_subst`.
-- ════════════════════════════════════════════════════════════════

/-- A forward graph: input `x : Vec inp`, producing a `Vec out`. Each
    constructor is a forward op (the affine `dense`, the `relu`
    nonlinearity); the StableHLO mirror renders `dense` as
    `dot_general + broadcast_in_dim + add` and `relu` as `maximum 0`. -/
inductive Fwd (inp : Nat) : Nat → Type where
  | input : Fwd inp inp
  | dense {m n : Nat} (W : Mat m n) (b : Vec n) : Fwd inp m → Fwd inp n
  | relu {n : Nat} : Fwd inp n → Fwd inp n

/-- **Denotational semantics** of a forward graph, into the proofs' `Vec`
    type — so a bridge can equate it with the proven forward map. -/
noncomputable def Fwd.denote {inp out : Nat} (e : Fwd inp out) (x : Vec inp) : Vec out :=
  -- (`_root_.Proofs.dense`/`relu` — the bare names resolve to the `Fwd`
  -- constructors inside this namespace.)
  match e with
  | .input        => x
  | .dense W b e' => _root_.Proofs.dense W b (e'.denote x)
  | .relu e'      => _root_.Proofs.relu _ (e'.denote x)

/-- Plug `g` into the input leaf of `e` (forward composition). The forward
    analogue of `Back.subst`. -/
def Fwd.subst {inp mid out : Nat} (e : Fwd mid out) (g : Fwd inp mid) : Fwd inp out :=
  match e with
  | .input        => g
  | .dense W b e' => .dense W b (e'.subst g)
  | .relu e'      => .relu (e'.subst g)

/-- **Forward IR chain rule** — `subst` denotes composition. Mirror of
    `denote_subst`; lets forward graphs compose to arbitrary depth. -/
theorem denote_subst_fwd {inp mid out : Nat} (e : Fwd mid out) (g : Fwd inp mid)
    (x : Vec inp) : (e.subst g).denote x = e.denote (g.denote x) := by
  induction e with
  | input => rfl
  | dense W b e' ih => simp only [Fwd.subst, Fwd.denote, ih]
  | relu e' ih => simp only [Fwd.subst, Fwd.denote, ih]

/-- The emitted forward graph for the whole MLP:
    `dense W₂ ∘ relu ∘ dense W₁ ∘ relu ∘ dense W₀` as a `Fwd` tree. -/
def emitMlpFwd {d₀ d₁ d₂ d₃ : Nat}
    (W₀ : Mat d₀ d₁) (b₀ : Vec d₁) (W₁ : Mat d₁ d₂) (b₁ : Vec d₂)
    (W₂ : Mat d₂ d₃) (b₂ : Vec d₃) : Fwd d₀ d₃ :=
  .dense W₂ b₂ (.relu (.dense W₁ b₁ (.relu (.dense W₀ b₀ .input))))

/-- **Forward bridge.** The emitted forward graph denotes the proven forward
    map `mlpForward`. Promotes the train step's forward from trusted to
    proof-backed (up to the printer): the emitted forward StableHLO is the
    rendering of an IR proven to compute `mlpForward`. -/
theorem mlp_fwd_bridge {d₀ d₁ d₂ d₃ : Nat}
    (W₀ : Mat d₀ d₁) (b₀ : Vec d₁) (W₁ : Mat d₁ d₂) (b₁ : Vec d₂)
    (W₂ : Mat d₂ d₃) (b₂ : Vec d₃) (x : Vec d₀) :
    (emitMlpFwd W₀ b₀ W₁ b₁ W₂ b₂).denote x = mlpForward W₀ b₀ W₁ b₁ W₂ b₂ x :=
  rfl

/-- **Splice contract, layer-0 pre-activation.** The forward IR's sub-graph
    up to the first ReLU input denotes exactly the pre-activation
    `dense W₀ b₀ x` that the backward graph reads as `p₀` (its first
    `compare`/`select` mask). Forward output ↦ backward input, proven. -/
theorem mlp_fwd_preact0 {d₀ d₁ : Nat} (W₀ : Mat d₀ d₁) (b₀ : Vec d₁) (x : Vec d₀) :
    (Fwd.dense W₀ b₀ Fwd.input).denote x = dense W₀ b₀ x := rfl

/-- **Splice contract, layer-1 pre-activation.** Likewise the sub-graph up to
    the second ReLU input denotes the `p₁` the backward reads. -/
theorem mlp_fwd_preact1 {d₀ d₁ d₂ : Nat} (W₀ : Mat d₀ d₁) (b₀ : Vec d₁)
    (W₁ : Mat d₁ d₂) (b₁ : Vec d₂) (x : Vec d₀) :
    (Fwd.dense W₁ b₁ (Fwd.relu (Fwd.dense W₀ b₀ Fwd.input))).denote x
      = dense W₁ b₁ (relu d₁ (dense W₀ b₀ x)) := rfl

-- ════════════════════════════════════════════════════════════════
-- § Loss cotangent — closing the last supplied input (rest of Phase 4)
--
-- The train step fed the backward a *supplied* cotangent dy = ∂L/∂logits.
-- For softmax-cross-entropy the repo PROVES that gradient is
-- `softmax(logits) − onehot(label)` (`softmaxCE_grad`). Emitting that as a
-- loss-head graph (`exp`+`reduce`+`broadcast`+`divide`, then `subtract` the
-- target) makes dy itself proof-backed: the train step then takes the target
-- distribution as input and the only trusted numerics left is SGD.
-- ════════════════════════════════════════════════════════════════

/-- The emitted loss-cotangent (softmax-CE head): `softmax(logits) −
    onehot(label)`. Rendered as `exp` + `reduce`(add) + `broadcast` + `divide`
    (softmax) then `subtract` the target. Feeds the backward's cotangent leaf. -/
noncomputable def emitLossCot (c : Nat) (logits : Vec c) (label : Fin c) : Vec c :=
  fun j => softmax c logits j - oneHot c label j

/-- **Loss-cotangent bridge.** The emitted softmax−onehot graph denotes the
    proven cross-entropy gradient `∂(crossEntropy)/∂logits` (`softmaxCE_grad`).
    So the cotangent fed to the backward is itself proof-backed, not supplied:
    the whole train step `forward → loss → backward → grads` is proof-backed
    end to end, and only the SGD arithmetic (and printer/IREE/float) stays
    trusted. -/
theorem lossCot_bridge (c : Nat) (logits : Vec c) (label : Fin c) (j : Fin c) :
    emitLossCot c logits label j
      = pdiv (fun (z : Vec c) (_ : Fin 1) => crossEntropy c z label) logits j 0 :=
  (softmaxCE_grad c logits label j).symm

end IR
end Proofs

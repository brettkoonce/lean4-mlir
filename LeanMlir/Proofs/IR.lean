import LeanMlir.Proofs.MLP
import LeanMlir.Proofs.CNN

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

/-- **Denotational semantics** of a backward graph, into the proofs' own
    `Vec` type — so a bridge theorem can equate it with a proven
    `HasVJP.backward`. -/
noncomputable def Back.denote {inp out : Nat} (e : Back inp out) (dy : Vec inp) : Vec out :=
  match e with
  | .cotangent       => dy
  | .dotGeneral A e' => Mat.mulVec A (e'.denote dy)
  | .selectPos x e'  => fun i => if x i > 0 then e'.denote dy i else 0

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

end IR
end Proofs

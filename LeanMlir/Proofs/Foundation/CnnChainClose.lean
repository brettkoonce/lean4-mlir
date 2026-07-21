import LeanMlir.Proofs.Foundation.CnnTrainStep
import LeanMlir.Proofs.Foundation.MlpTrainStep

/-! # Upgrading the CNN conv close from a generic cotangent to the actual backward chain

`cnn_render_conv{W,b}_certified` (CnnTrainStep.lean) certify each conv parameter output for
*any* cotangent `c` at that conv layer's output. This file pins `c` to the cotangent the CNN
backward chain *actually* delivers — the conv analogue of the MLP's `mlpCotOut0/1`.

The chain, from the loss cotangent `dy` at the logits, all in flattened `Vec` space:
- **dense head** (`W₅→relu→W₄→relu→W₃`) is a flat `IR.Back` chain (`emitDenseBack`/`emitReluBack`
  + `subst`, the `mlpCotOut` mechanism) → the cotangent `cpool` at the flattened pool output;
- **maxpool-back** is an `IR.Back3` node viewed through `flatDenote` (crossing the flatten
  boundary), then **relu-back** (the `selMask4` mask `relu'(hc2)⊙·`) → the cotangent at conv2's
  output (`W₂`'s layer);
- **conv2-back** is another `Back3` node via `flatDenote`, then relu-back → conv1's output (`W₁`).

The relu masks sit *between* the maxpool and conv `Back3` nodes, so the cotangent is a flat-level
composition of the rendered backward denotations (not a single `Back3` graph) — but the
maxpool/conv steps are exactly the `Back3` subgraphs `flatDenote` denotes. Instantiating the
generic conv bridges at these cotangents gives: each conv `θ` output denotes `θ − lr·(certified
∂conv/∂θ · the-actual-chain-cotangent)`. (This pins the cotangent — the further "= ∂loss/∂θ" fold
is the separate `pdiv G = Back.denote` step, as in the MLP `mlp_*_total_loss_grad`.)
See `planning/render_close_handoff.md` §1 "Optional polish".
-/

namespace Proofs

open Proofs.IR

/-- **Dense-head backward subgraph** — `dy` at the logits to the cotangent at the flattened
    pool output: `W₃·(relu'(h3)⊙(W₄·(relu'(h4)⊙(W₅·dy))))`. The flat `Back` chain of the CNN's
    classifier head (3 dense + 2 relu); the 3-layer analogue of `mlpCotOut0`. -/
def cnnDenseHeadCot {c h w d1 nClasses : Nat}
    (W₃ : Mat (c * h * w) d1) (W₄ : Mat d1 d1) (W₅ : Mat d1 nClasses) (h3 h4 : Vec d1) :
    Back nClasses (c * h * w) :=
  (emitDenseBack W₃).subst ((emitReluBack h3).subst
    ((emitDenseBack W₄).subst ((emitReluBack h4).subst (emitDenseBack W₅))))

/-- The cotangent the backward chain delivers at **conv2's output** (`W₂`'s layer):
    `relu'(hc2) ⊙ maxpool-back(dense-head-cot dy)`. The maxpool step is the `Back3` maxpool node
    through `flatDenote` (crossing the flatten/pool boundary); the relu-back is the rendered
    `selMask4` mask. -/
noncomputable def cnnChainCotW2 {c h w d1 nClasses : Nat}
    (W₃ : Mat (c * h * w) d1) (W₄ : Mat d1 d1) (W₅ : Mat d1 nClasses) (h3 h4 : Vec d1)
    (ac2 : Tensor3 c (2 * h) (2 * w)) (hc2 : Vec (c * (2 * h) * (2 * w))) (dy : Vec nClasses) :
    Vec (c * (2 * h) * (2 * w)) :=
  fun i => if hc2 i > 0
    then (Back3.maxpool (c₁ := c) (h₁ := h) (w₁ := w) ac2 Back3.cot).flatDenote
           ((cnnDenseHeadCot W₃ W₄ W₅ h3 h4).denote dy) i
    else 0

/-- The cotangent the backward chain delivers at **conv1's output** (`W₁`'s layer):
    `relu'(hc1) ⊙ conv2-back(W₂, conv2-cotangent)`. The conv2-back step is the `Back3` conv node
    through `flatDenote`; the relu-back is the rendered `selMask4` mask. Builds on
    `cnnChainCotW2` exactly as `mlpCotOut0` prepends one more `relu-back ∘ dense-back` to
    `mlpCotOut1`. -/
noncomputable def cnnChainCotW1 {c h w kH kW : Nat}
    (W₂ : Kernel4 c c kH kW) (hc1 : Vec (c * (2 * h) * (2 * w)))
    (cotW2 : Vec (c * (2 * h) * (2 * w))) : Vec (c * (2 * h) * (2 * w)) :=
  fun i => if hc1 i > 0
    then (Back3.conv (c₁ := c) (h₁ := 2 * h) (w₁ := 2 * w) W₂ Back3.cot).flatDenote cotW2 i
    else 0

/-- The conv1-output cotangent equals the explicit rendered backward form
    `relu'(hc1) ⊙ flatten(convBackDenote W₂ (unflatten cotW2))` — i.e. the relu mask applied to
    the reversed-kernel conv backward the renderer emits (`selMask4 ∘ convBack`). -/
theorem cnnChainCotW1_eq {c h w kH kW : Nat}
    (W₂ : Kernel4 c c kH kW) (hc1 cotW2 : Vec (c * (2 * h) * (2 * w))) :
    cnnChainCotW1 W₂ hc1 cotW2
      = fun i => if hc1 i > 0
          then Tensor3.flatten (convBackDenote W₂ (Tensor3.unflatten cotW2)) i else 0 := by
  rfl

/-- The conv2-output cotangent equals the explicit rendered backward form
    `relu'(hc2) ⊙ flatten(maxPoolBackDenote ac2 (unflatten (dense-head-cot dy)))` — the relu mask
    applied to the `select_and_scatter` maxpool backward the renderer emits. -/
theorem cnnChainCotW2_eq {c h w d1 nClasses : Nat}
    (W₃ : Mat (c * h * w) d1) (W₄ : Mat d1 d1) (W₅ : Mat d1 nClasses) (h3 h4 : Vec d1)
    (ac2 : Tensor3 c (2 * h) (2 * w)) (hc2 : Vec (c * (2 * h) * (2 * w))) (dy : Vec nClasses) :
    cnnChainCotW2 W₃ W₄ W₅ h3 h4 ac2 hc2 dy
      = fun i => if hc2 i > 0
          then Tensor3.flatten
                 (maxPoolBackDenote ac2 (Tensor3.unflatten ((cnnDenseHeadCot W₃ W₄ W₅ h3 h4).denote dy))) i
          else 0 := by
  rfl

/-- **Dense-head cotangent, explicit.** `cnnDenseHeadCot.denote dy` is the explicit dense
    backprop `W₃·(relu'(h3)⊙(W₄·(relu'(h4)⊙(W₅·dy))))` — the `mlpCotOut`-style chain spelled out,
    via the `IR.Back` chain rule `denote_subst`. -/
theorem cnnDenseHeadCot_denote {c h w d1 nClasses : Nat}
    (W₃ : Mat (c * h * w) d1) (W₄ : Mat d1 d1) (W₅ : Mat d1 nClasses) (h3 h4 : Vec d1)
    (dy : Vec nClasses) :
    (cnnDenseHeadCot W₃ W₄ W₅ h3 h4).denote dy
      = Mat.mulVec W₃ (fun i => if h3 i > 0
          then Mat.mulVec W₄ (fun k => if h4 k > 0 then Mat.mulVec W₅ dy k else 0) i else 0) := by
  simp only [cnnDenseHeadCot, denote_subst, emitDenseBack, emitReluBack, Back.denote]

-- ════════════════════════════════════════════════════════════════
-- § The chain-pinned conv closes — the generic bridges at the actual cotangents
-- ════════════════════════════════════════════════════════════════

/-- **Conv-2 weight output, chain-certified.** `W₂ⁿ = W₂ − lr·(transpose-trick kernel grad)`
    denotes `W₂ − lr·(certified ∂conv2/∂W₂ · the cotangent the chain delivers at conv2)` — the
    generic `cnn_render_convW_certified` instantiated at `cnnChainCotW2`. -/
theorem cnn_render_convW2_chain_certified {c h w d1 nClasses kH kW : Nat}
    (b₂ : Vec c) (ac1 : Tensor3 c (2 * h) (2 * w))
    (W₃ : Mat (c * h * w) d1) (W₄ : Mat d1 d1) (W₅ : Mat d1 nClasses) (h3 h4 : Vec d1)
    (ac2 : Tensor3 c (2 * h) (2 * w)) (hc2 : Vec (c * (2 * h) * (2 * w))) (dy : Vec nClasses)
    (v : Vec (c * c * kH * kW)) (lr : ℝ) (idx : Fin (c * c * kH * kW)) :
    v idx - lr * (conv2d_weight_grad_has_vjp b₂ ac1).backward v
        (cnnChainCotW2 W₃ W₄ W₅ h3 h4 ac2 hc2 dy) idx
      = v idx - lr * ∑ j : Fin (c * (2 * h) * (2 * w)),
          pdiv (fun v' : Vec (c * c * kH * kW) =>
                  Tensor3.flatten (conv2d (Kernel4.unflatten v') b₂ ac1))
               v idx j * cnnChainCotW2 W₃ W₄ W₅ h3 h4 ac2 hc2 dy j :=
  cnn_render_convW_certified b₂ ac1 v (cnnChainCotW2 W₃ W₄ W₅ h3 h4 ac2 hc2 dy) lr idx

/-- **Conv-2 bias output, chain-certified.** -/
theorem cnn_render_convb2_chain_certified {c h w d1 nClasses kH kW : Nat}
    (W₂ : Kernel4 c c kH kW) (b₂ : Vec c) (ac1 : Tensor3 c (2 * h) (2 * w))
    (W₃ : Mat (c * h * w) d1) (W₄ : Mat d1 d1) (W₅ : Mat d1 nClasses) (h3 h4 : Vec d1)
    (ac2 : Tensor3 c (2 * h) (2 * w)) (hc2 : Vec (c * (2 * h) * (2 * w))) (dy : Vec nClasses)
    (lr : ℝ) (o : Fin c) :
    b₂ o - lr * (conv2d_bias_grad_has_vjp W₂ ac1).backward b₂
        (cnnChainCotW2 W₃ W₄ W₅ h3 h4 ac2 hc2 dy) o
      = b₂ o - lr * ∑ j : Fin (c * (2 * h) * (2 * w)),
          pdiv (fun b' : Vec c => Tensor3.flatten (conv2d W₂ b' ac1)) b₂ o j
            * cnnChainCotW2 W₃ W₄ W₅ h3 h4 ac2 hc2 dy j :=
  cnn_render_convb_certified W₂ ac1 b₂ (cnnChainCotW2 W₃ W₄ W₅ h3 h4 ac2 hc2 dy) lr o

/-- **Conv-1 weight output, chain-certified.** `W₁ⁿ` denotes `W₁ − lr·(certified ∂conv1/∂W₁ ·
    the deepest chain cotangent)` — the generic bridge at `cnnChainCotW1` (which crosses one
    more conv-back than `cnnChainCotW2`, the `Back3` chain step). -/
theorem cnn_render_convW1_chain_certified {ic c h w kH kW : Nat}
    (b₁ : Vec c) (x : Tensor3 ic (2 * h) (2 * w))
    (hc1 : Vec (c * (2 * h) * (2 * w))) (cotW2 : Vec (c * (2 * h) * (2 * w)))
    (W₂ : Kernel4 c c kH kW) (v : Vec (c * ic * kH * kW)) (lr : ℝ)
    (idx : Fin (c * ic * kH * kW)) :
    v idx - lr * (conv2d_weight_grad_has_vjp b₁ x).backward v (cnnChainCotW1 W₂ hc1 cotW2) idx
      = v idx - lr * ∑ j : Fin (c * (2 * h) * (2 * w)),
          pdiv (fun v' : Vec (c * ic * kH * kW) =>
                  Tensor3.flatten (conv2d (Kernel4.unflatten v') b₁ x))
               v idx j * cnnChainCotW1 W₂ hc1 cotW2 j :=
  cnn_render_convW_certified b₁ x v (cnnChainCotW1 W₂ hc1 cotW2) lr idx

/-- **Conv-1 bias output, chain-certified.** -/
theorem cnn_render_convb1_chain_certified {ic c h w kH kW : Nat}
    (W₁ : Kernel4 c ic kH kW) (b₁ : Vec c) (x : Tensor3 ic (2 * h) (2 * w))
    (hc1 : Vec (c * (2 * h) * (2 * w))) (cotW2 : Vec (c * (2 * h) * (2 * w)))
    (W₂ : Kernel4 c c kH kW) (lr : ℝ) (o : Fin c) :
    b₁ o - lr * (conv2d_bias_grad_has_vjp W₁ x).backward b₁ (cnnChainCotW1 W₂ hc1 cotW2) o
      = b₁ o - lr * ∑ j : Fin (c * (2 * h) * (2 * w)),
          pdiv (fun b' : Vec c => Tensor3.flatten (conv2d W₁ b' x)) b₁ o j
            * cnnChainCotW1 W₂ hc1 cotW2 j :=
  cnn_render_convb_certified W₁ x b₁ (cnnChainCotW1 W₂ hc1 cotW2) lr o

end Proofs

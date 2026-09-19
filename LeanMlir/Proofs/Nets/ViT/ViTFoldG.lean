import LeanMlir.Proofs.Nets.ViT.ViTFold

/-! # ViT-Tiny un-fused embedding-gradient nodes at the per-example index

Two per-example lemmas that `ViTFoldGB` lifts over the batch: the positional-table gradient
(`posEmbedGrad_den`) and the CLS-token gradient (`clsGrad_den`), each `den`-faithful at the RAW
gradient node every optimizer tail consumes. The fusion is `rfl` (`StableHLO.lean`'s
`*Sgd_eq_grad` family), so each proof is its fused peer's (`ViTPoC.posEmbedSgd_den`,
`ViTTiePoC.vit_cls_den`) with the `θ − lr·` wrapper dropped. Every Adam artifact of this net renders
from the batched chain; its fold is `ViTFoldGB`.
-/

open Proofs Proofs.StableHLO Proofs.IR

namespace Proofs.ViTPoCG

open scoped BigOperators

-- ════════════════════════════════════════════════════════════════
-- § The patch embedding — conv weight/bias, the CLS token, the positional table
-- ════════════════════════════════════════════════════════════════

/-- **Positional-embed GRADIENT denotes the certified gradient** — the cotangent itself, since the
    positional table is added to every token and its Jacobian is the identity. -/
theorem posEmbedGrad_den {ic H W P N D : Nat} (cotN : String)
    (Wc : Kernel4 D ic P P) (bc cls : Vec D) (pos : Mat (N + 1) D) (img : Vec (ic * H * W))
    (dy : Vec ((N + 1) * D)) (i : Fin ((N + 1) * D)) :
    den (SHlo.posEmbedGrad (.operand cotN dy)) i
      = ∑ j : Fin ((N + 1) * D),
          pdiv (fun p : Vec ((N + 1) * D) =>
                  patchEmbed_flat ic H W P N D Wc bc cls (Mat.unflatten p) img)
            (Mat.flatten pos) i j * dy j := by
  simp [den, pdiv_patchEmbed_pos]

/-- **CLS-token GRADIENT denotes the certified gradient.** The render slices row 0 of the embed
    cotangent (`clsSliceF`) and then reduces it as a `[1, D]` batch, so the op is
    `denseBiasGradB` at `N = 1` and its `den` IS `cls_token_grad`. ⚠ Stated at the committed
    ViT-Tiny dims rather than generically, for the reason `ViTTiePoC.vit_cls_den` is: the operand's
    type is `Vec (1 * D)`, which reduces to `Vec D` only at a literal `D`.

    ⭐ The fused peer's proof ends in `vit_render_cls_certified`, whose statement carries the
    `θ − lr·` wrapper and has no un-wrapped twin; instantiating it at `lr = 1` un-fuses it, which
    is the same content as a `*Sgd_eq_grad` `rfl` read backwards. -/
theorem clsGrad_den (cotN : String)
    (Wc : Kernel4 192 3 16 16) (bc cls : Vec 192) (pos : Mat 197 192)
    (img : Vec (3 * 224 * 224)) (dyEmbed : Vec (197 * 192)) (i : Fin 192) :
    den (SHlo.denseBiasGradB (N := 1) (c := 192)
            (.operand cotN (clsSliceFlat 196 192 dyEmbed))) i
      = ∑ j : Fin (197 * 192),
          pdiv (fun cl : Vec 192 =>
                  patchEmbed_flat 3 224 224 16 196 192 Wc bc cl pos img) cls i j * dyEmbed j := by
  have hstep : den (SHlo.denseBiasGradB (N := 1) (c := 192)
            (.operand cotN (clsSliceFlat 196 192 dyEmbed))) i = cls_token_grad dyEmbed i := by
    simp only [den, batchSlice, cls_token_grad]; rw [Fin.sum_univ_one]; rfl
  rw [hstep]
  have h := vit_render_cls_certified Wc bc cls pos img dyEmbed 1 i
  linarith

end Proofs.ViTPoCG

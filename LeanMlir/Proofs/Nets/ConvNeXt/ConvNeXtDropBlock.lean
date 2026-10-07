import LeanMlir.Proofs.Nets.ConvNeXt.ConvNeXtStepTie
import LeanMlir.Proofs.Nets.ConvNeXt.ConvNeXtWholeBackCertifiedTie
import LeanMlir.Proofs.Foundation.Batched.Indexed

/-! # One ConvNeXt block with its drop site, per example — forward, VJP, input cotangent

The `*drop*` ConvNeXt renders put one stochastic-depth site per block on the residual branch,
between LayerScale and the skip add (`ConvNeXtRenderB`'s block forward), and the backward puts the
same op on the block-output cotangent before the WHOLE branch reads it — LayerScale γ's node
included — while the skip fan-in reads the raw one (`ConvNeXtRenderB.bwdBlockB`;
`dropPath_vjp_is_self`). Per example the site is a scalar or absent (`dropScalarOpt`,
`Foundation.Batched.Indexed`).

* `CnxTieBlk.fwdOD` — the block forward with the site, `siteScale s (body x) + x`; at `none` it IS
  `fwdO` (`fwdOD_none`, `rfl`).
* `CnxTieBlk.cotInD` — the render's input cotangent: `cnxBlockCotInChAt`'s chain fed `s ⊙ dy`, the
  skip raw; `cotIn` at `none` (`cotInD_none`, `rfl`).
* `CnxTieBlk.fwdODHasVJP` / `cotInD_eq_vjp` — the block is the residual over `dropScalarOpt s ∘ body`,
  and the chain IS that witness's backward; the branch backward is read off `cnxBlockCotInChAt_eq_vjp`
  with its skip taken off.
-/

open Proofs Proofs.StableHLO

namespace Proofs.CnxTieGB

open scoped BigOperators
open Proofs.CnxTie (CnxTieBlk cnxBlockFwdChO cnxBlockCotInChAt)

variable {c cExp : Nat}

/-- **A ConvNeXt block's input cotangent is its certified VJP's backward.** The chain's
    per-op pieces (`depthwiseFlatHasVJP`, the 1×1 `conv2dHasVJP3`s, the GELU mask, layer scale,
    `chanLNTensor3Back`) are rewritten into `cnxBlockChBack_eq_vjp`'s form, which ties the block. -/
theorem cnxBlockCotInChAt_eq_vjp {gf : GeluForm} {c cExp h w : Nat} (ε : ℝ) (hε : 0 < ε)
    (Wdw : DepthwiseKernel c 7 7) (bdw : Vec c) (ng nbt : Vec c)
    (Wex : Kernel4 cExp c 1 1) (bex : Vec cExp) (Wpr : Kernel4 c cExp 1 1) (bpr : Vec c)
    (lg : Vec c) (xin dyOut : Vec (c*h*w)) :
    cnxBlockCotInChAt gf ε Wdw bdw ng nbt Wex bex Wpr bpr lg xin dyOut
      = (cnxBlockChWHasVJP gf (h := h) (w := w)
          ⟨Wdw, bdw, ε, ng, nbt, Wex, bex, Wpr, bpr, lg⟩ hε).backward xin dyOut := by
  rw [← cnxBlockChBack_eq_vjp (by norm_num) (by norm_num)
    (⟨Wdw, bdw, ε, ng, nbt, Wex, bex, Wpr, bpr, lg⟩ : CnxBlockParamsCh c cExp h w 7 7) hε xin]
  simp only [cnxBlockBodyBack]
  rw [depthwiseFlatBack_eq_vjp_backward (by norm_num) (by norm_num) Wdw bdw xin,
    convFlatBack_eq_vjp_backward (by norm_num) (by norm_num) Wex bex
      (chanLNTensor3 c h w ε ng nbt (depthwiseFlat (h := h) (w := w) Wdw bdw xin)),
    convFlatBack_eq_vjp_backward (by norm_num) (by norm_num) Wpr bpr
      (gf.map (cExp * h * w) (flatConv (h := h) (w := w) Wex bex
        (chanLNTensor3 c h w ε ng nbt (depthwiseFlat (h := h) (w := w) Wdw bdw xin))))]
  rfl

/-- The block's weights as `ConvNeXtFullT`'s record, at the shared `ε`. -/
abbrev _root_.Proofs.CnxTie.CnxTieBlk.toCh (p : CnxTieBlk c cExp) (h w : Nat) (ε : ℝ) :
    CnxBlockParamsCh c cExp h w 7 7 :=
  ⟨p.aW, p.aB, ε, p.nG, p.nB, p.eW, p.eB, p.pW, p.pB, p.sL⟩

/-- The block's residual branch, flat: depthwise → channel LN → expand → GELU → project →
    LayerScale (`cnxBodyWith` at the record). -/
noncomputable abbrev _root_.Proofs.CnxTie.CnxTieBlk.bodyF (gf : GeluForm) (p : CnxTieBlk c cExp)
    {h w : Nat} (ε : ℝ) : Vec (c * h * w) → Vec (c * h * w) :=
  cnxBodyWith gf (chanLNTensor3 c h w ε p.nG p.nB) p.aW p.aB p.eW p.eB p.pW p.pB
    (cnxGlsCh (p.toCh h w ε))

/-- **The block forward at its drop site**: the branch through `siteScale s`, then the skip. -/
noncomputable abbrev _root_.Proofs.CnxTie.CnxTieBlk.fwdOD (gf : GeluForm) (p : CnxTieBlk c cExp)
    {h w : Nat} (ε : ℝ) (s : Option ℝ) : Vec (c * h * w) → Vec (c * h * w) :=
  fun xin i => siteScale s (p.bodyF gf ε xin i) + xin i

/-- With no site rendered the block is the drop-free one. -/
theorem fwdOD_none {gf : GeluForm} {h w : Nat} (ε : ℝ) (p : CnxTieBlk c cExp) :
    p.fwdOD gf (h := h) (w := w) ε none = p.fwdO gf ε := rfl

/-- **The block's input cotangent at its drop site** — `cnxBlockCotInChAt`'s `let` chain with the
    branch fed `s ⊙ dyOut` and the skip fan-in the raw `dyOut`. -/
noncomputable def cnxBlockCotInChAtD (gf : GeluForm) {c cExp h w : Nat} (ε : ℝ)
    (Wdw : DepthwiseKernel c 7 7) (bdw : Vec c) (ng nbt : Vec c)
    (Wex : Kernel4 cExp c 1 1) (bex : Vec cExp) (Wpr : Kernel4 c cExp 1 1) (bpr : Vec c)
    (lg : Vec c) (s : Option ℝ) (xin dyOut : Vec (c*h*w)) : Vec (c*h*w) :=
  let γlsB : Vec (c*h*w) := fun k => lg (chanIdx c h w k)
  let d := depthwiseFlat (h := h) (w := w) Wdw bdw xin
  let nl := chanLNTensor3 c h w ε ng nbt d
  let e := flatConv (h := h) (w := w) Wex bex nl
  let g := gf.map (cExp*h*w) e
  let cotD := chanLNTensor3Back c h w ε ng d
    (cnxCotN gf γlsB Wex bex Wpr bpr nl g e (dropScalarOpt s dyOut))
  fun i => (depthwiseFlatHasVJP (h := h) (w := w) Wdw bdw).backward xin cotD i + dyOut i

/-- The block's input cotangent at its drop site (`cnxBlockCotInChAt` at `none`). -/
noncomputable abbrev _root_.Proofs.CnxTie.CnxTieBlk.cotInD (gf : GeluForm) (p : CnxTieBlk c cExp)
    {h w : Nat} (ε : ℝ) (s : Option ℝ) : Vec (c*h*w) → Vec (c*h*w) → Vec (c*h*w) :=
  cnxBlockCotInChAtD gf ε p.aW p.aB p.nG p.nB p.eW p.eB p.pW p.pB p.sL s

/-- With no site rendered the chain is the drop-free one. -/
theorem cotInD_none {gf : GeluForm} {h w : Nat} (ε : ℝ) (p : CnxTieBlk c cExp) :
    p.cotInD gf (h := h) (w := w) ε none = p.cotIn gf ε := rfl

/-- The chain at the site is the drop-free chain at the dropped cotangent, with the skip's
    `s ⊙ dy` swapped back for `dy`. -/
theorem cotInD_eq_cotIn {gf : GeluForm} {h w : Nat} (ε : ℝ) (p : CnxTieBlk c cExp) (s : Option ℝ)
    (xin dy : Vec (c*h*w)) :
    p.cotInD gf (h := h) (w := w) ε s xin dy
      = fun i => p.cotIn gf ε xin (dropScalarOpt s dy) i - dropScalarOpt s dy i + dy i := by
  funext i
  simp only [CnxTieBlk.cotInD, CnxTieBlk.cotIn, cnxBlockCotInChAtD, cnxBlockCotInChAt]
  ring

theorem bodyF_differentiable {gf : GeluForm} {h w : Nat} (ε : ℝ) (hε : 0 < ε) (p : CnxTieBlk c cExp) :
    Differentiable ℝ (p.bodyF gf (h := h) (w := w) ε) :=
  cnxBodyWith_differentiable (chanLNTensor3_differentiable c h w ε p.nG p.nB hε) _ _ _ _ _ _ _

/-- The branch's VJP — the one `cnxBlockChWHasVJP` puts under its residual. -/
noncomputable def bodyFHasVJP (gf : GeluForm) {h w : Nat} (ε : ℝ) (hε : 0 < ε) (p : CnxTieBlk c cExp) :
    HasVJP (p.bodyF gf (h := h) (w := w) ε) :=
  cnxBodyWithHasVJP gf (chanLNTensor3_differentiable c h w ε p.nG p.nB hε)
    (chanLNTensor3HasVJP c h w ε p.nG p.nB hε) p.aW p.aB p.eW p.eB p.pW p.pB (cnxGlsCh (p.toCh h w ε))

/-- **The branch's backward is the render's chain without its skip** (`cnxBlockCotInChAt_eq_vjp`
    minus the residual's identity). -/
theorem bodyF_back {gf : GeluForm} {h w : Nat} (ε : ℝ) (hε : 0 < ε) (p : CnxTieBlk c cExp)
    (x v : Vec (c*h*w)) :
    (bodyFHasVJP gf ε hε p).backward x v = fun i => p.cotIn gf (h := h) (w := w) ε x v i - v i := by
  funext i
  have hb := congrFun (cnxBlockCotInChAt_eq_vjp (gf := gf) (h := h) (w := w) ε hε p.aW p.aB p.nG p.nB
    p.eW p.eB p.pW p.pB p.sL x v) i
  simp only [cnxBlockChWHasVJP, residualHasVJP, biPathHasVJP, identityHasVJP] at hb
  unfold bodyFHasVJP
  simp only [CnxTieBlk.cotIn]
  linarith

theorem fwdOD_differentiable {gf : GeluForm} {h w : Nat} (ε : ℝ) (hε : 0 < ε) (p : CnxTieBlk c cExp)
    (s : Option ℝ) : Differentiable ℝ (p.fwdOD gf (h := h) (w := w) ε s) :=
  ((dropScalarOpt_differentiable s).comp (bodyF_differentiable ε hε p)).add differentiable_id

/-- **The block's VJP at its drop site**: the residual over the dropped branch; its backward is
    `body.back x (s ⊙ dy) + dy` by definition. -/
noncomputable def _root_.Proofs.CnxTie.CnxTieBlk.fwdODHasVJP (gf : GeluForm) (p : CnxTieBlk c cExp)
    {h w : Nat} (ε : ℝ) (hε : 0 < ε) (s : Option ℝ) : HasVJP (p.fwdOD gf (h := h) (w := w) ε s) :=
  biPathHasVJP (dropScalarOpt s ∘ p.bodyF gf ε) (fun x => x)
    ((dropScalarOpt_differentiable s).comp (bodyF_differentiable ε hε p)) differentiable_id
    (vjpComp (p.bodyF gf ε) (dropScalarOpt s) (bodyF_differentiable ε hε p)
      (dropScalarOpt_differentiable s) (bodyFHasVJP gf ε hε p) (dropScalarOptHasVJP s))
    (identityHasVJP _)

/-- **The chain's block-input cotangent at the site is the block VJP's backward.** -/
theorem cotInD_eq_vjp {gf : GeluForm} {h w : Nat} (ε : ℝ) (hε : 0 < ε) (p : CnxTieBlk c cExp)
    (s : Option ℝ) (xin dy : Vec (c*h*w)) :
    p.cotInD gf (h := h) (w := w) ε s xin dy = (p.fwdODHasVJP gf ε hε s).backward xin dy := by
  rw [cotInD_eq_cotIn]
  show _ = fun i => (bodyFHasVJP gf ε hε p).backward xin (dropScalarOpt s dy) i + dy i
  rw [bodyF_back]

end Proofs.CnxTieGB

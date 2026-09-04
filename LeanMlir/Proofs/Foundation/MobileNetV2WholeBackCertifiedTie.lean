import LeanMlir.Proofs.Architectures.MobileNetV2BackCertifiedTie
import LeanMlir.Proofs.Foundation.Resnet34BackCertifiedTie
import LeanMlir.Proofs.Float.MobileNetV2BackFloatBridge
import LeanMlir.Proofs.Codegen.MobileNetV2RenderPC

/-! # ⭐⭐ `mnv2InputGrad` IS the certified whole-net MobileNetV2 gradient

`MobileNetV2BackCertifiedTie.lean` (`Architectures/`) closed §B for the inverted-residual
BODIES: each hand-assembled reverse-mode transcription IS the certified input-gradient VJP of the
body it reverses, in the deployed non-batched per-channel-BN vocabulary. This file closes the
same question for the WHOLE NET, so the reading of `mnv2_grad_float_le`
(`MobileNetV2BackFloatBudget.lean`) is no longer *"every piece of this chain is the certified
gradient"* but **"the chain IS the certified whole-net gradient"** — the MobileNetV2 peer of
`r34InputGrad_eq_resnet34_vjp` (`Resnet34BackCertifiedTie.lean`).

Five pieces, and only the first two are new mathematics:

1. `convStridedBnRelu6PC_has_vjp_at` — the STEM stage's certified VJP, `relu6 ∘ bnPC ∘
   flatConvStride2`. The repo had the strided-conv-with-**relu** peer (r34's `cbrStridedPC`) and
   the **non-strided** relu6 peer (`convBnRelu6PC_has_vjp_at`); this is the missing corner.
2. `convStridedBnRelu6PCBack_eq_vjp_backward` / `convBnRelu6PCBack_eq_vjp_backward` — the stem and
   head leaf ties, both closing on one conv-leaf rewrite (`flatConvStride2Back_eq_vjp_backward` /
   `convFlatBack_eq_vjp_backward`) and then `rfl`: relu6's certified backward IS
   `reluMaskBack (0 < · ∧ · < 6)`, and the pinned BN-back is definitionally the certified one.
3. `residualBack_eq_vjp_backward` — the additive skip, `rfl`. `residual_has_vjp_at`'s backward is
   `hf.backward dy i + dy i`, which is `Proofs.residual hf.backward`. It is what a caller uses to
   discharge the `b2`/`b4` slots concretely.
4. `mobilenetv2PC_has_vjp_at` — the apex. ⭐ Structurally SIMPLER than `resnet34_has_vjp_at`:
   MobileNetV2's skips live INSIDE the block maps and its stride changes inside the strided
   bodies, so there is no `ChainData` list and no separate downsample slot — nine
   `vjp_comp_diff_at`s and nothing else, dimension-generic and parametric in every component.
5. `mnv2InputGrad_eq_mobilenetv2_vjp` — the whole-net tie. The six blocks stay OPAQUE (they enter
   as `PProd (HasVJPAt …) (DifferentiableAt …)` witnesses and the backward's block slots are
   pinned to `.fst.backward`), so the composition is checked between variables and costs nothing;
   only the four concrete endpoints — stem, head, GAP, dense — are rewritten, and the proof is
   `unfold`, three `rw`s, `rfl`. The whole file elaborates in ~2 s.

⭐⭐ **And `mobilenetv2Forward_full_pc_eq_chain` is the piece `Resnet34BackCertifiedTie.lean` does
NOT have.** It states, by `rfl`, that the ten-stage chain the apex is instantiated at IS the
committed forward — `b1/b3/b5/b6` the strided bodies, `b2/b4` those bodies under
`Proofs.residual`, and the two endpoints spelled as the render spells them. ⛔ **That is the
theorem that would have caught ResNet-34's wrong pool.** §3.10's drift — `r34InputGrad` reversing
the 2×2 pool while the committed forward pools 3×3/s2 — survived a month because *"the same net as
the tie"* was prose in a docstring. Here it is a `rfl` the kernel checks.

⭐ **No drift was found on this net**, and that is a result rather than a non-event: the tie went
through against `mnv2InputGrad` exactly as committed, so `mnv2_grad_float_le`'s
4.750·10¹⁵³ / 1.076·10¹⁵² stand unchanged, where closing r34's moved its number 4×.

⚠ It stays a SMOOTH-POINT statement, as every `HasVJPAt` in this cone is: the stem's and head's
post-BN clamp windows (`≠ 0 ∧ ≠ 6`) and the six blocks' own VJP witnesses are hypotheses.
-/

namespace Proofs


/-- **conv(stride-2) → per-channel-BN → relu6 VJP at a smooth point** — MobileNetV2's STEM. -/
noncomputable def convStridedBnRelu6PC_has_vjp_at {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (γ β : Vec oc) (hε : 0 < ε)
    (v : Vec (ic * (2 * h) * (2 * w)))
    (h_smooth : ∀ k, (bnPerChannelTensor3 oc h w ε γ β (flatConvStride2 W b v) k ≠ 0 ∧
                       bnPerChannelTensor3 oc h w ε γ β (flatConvStride2 W b v) k ≠ 6)) :
    HasVJPAt (relu6 (oc * h * w) ∘ bnPerChannelTensor3 oc h w ε γ β
      ∘ flatConvStride2 (h := h) (w := w) W b) v := by
  have hconv_diff : Differentiable ℝ
      (flatConvStride2 W b : Vec (ic * (2*h) * (2*w)) → Vec (oc * h * w)) :=
    flatConvStride2_differentiable W b
  have hbn_diff : Differentiable ℝ (bnPerChannelTensor3 oc h w ε γ β) :=
    bnPerChannelTensor3_differentiable oc h w ε hε γ β
  have step1 : HasVJPAt (bnPerChannelTensor3 oc h w ε γ β ∘ flatConvStride2 W b) v :=
    vjp_comp_at (flatConvStride2 W b) (bnPerChannelTensor3 oc h w ε γ β) v
      (hconv_diff v) (hbn_diff _)
      ((flatConvStride2_has_vjp W b).toHasVJPAt v)
      ((bnPerChannelTensor3_has_vjp oc h w ε hε γ β).toHasVJPAt _)
  have step1_diff : DifferentiableAt ℝ
      (bnPerChannelTensor3 oc h w ε γ β ∘ flatConvStride2 W b) v :=
    DifferentiableAt.comp v (hbn_diff (flatConvStride2 W b v)) (hconv_diff v)
  exact vjp_comp_at (bnPerChannelTensor3 oc h w ε γ β ∘ flatConvStride2 W b)
    (relu6 (oc * h * w)) v step1_diff
    (relu6_differentiableAt_of_smooth (oc * h * w) _ h_smooth)
    step1 (relu6_has_vjp_at (oc * h * w) _ h_smooth)

theorem convStridedBnRelu6PC_differentiableAt {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (γ β : Vec oc) (hε : 0 < ε)
    (v : Vec (ic * (2 * h) * (2 * w)))
    (h_smooth : ∀ k, (bnPerChannelTensor3 oc h w ε γ β (flatConvStride2 W b v) k ≠ 0 ∧
                       bnPerChannelTensor3 oc h w ε γ β (flatConvStride2 W b v) k ≠ 6)) :
    DifferentiableAt ℝ (relu6 (oc * h * w) ∘ bnPerChannelTensor3 oc h w ε γ β
      ∘ flatConvStride2 (h := h) (w := w) W b) v := by
  have hinner : DifferentiableAt ℝ
      (bnPerChannelTensor3 oc h w ε γ β ∘ flatConvStride2 W b) v :=
    ((bnPerChannelTensor3_differentiable oc h w ε hε γ β).comp
      (flatConvStride2_differentiable W b)) v
  exact (relu6_differentiableAt_of_smooth (oc * h * w) _ h_smooth).comp v hinner

/-- **The STEM tie.** -/
theorem convStridedBnRelu6PCBack_eq_vjp_backward {ic oc h w kH kW : Nat}
    (hkH : 2 * ((kH - 1) / 2) + 1 = kH) (hkW : 2 * ((kW - 1) / 2) + 1 = kW)
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (γ β : Vec oc) (hε : 0 < ε)
    (v : Vec (ic * (2 * h) * (2 * w)))
    (h_smooth : ∀ k, (bnPerChannelTensor3 oc h w ε γ β (flatConvStride2 W b v) k ≠ 0 ∧
                       bnPerChannelTensor3 oc h w ε γ β (flatConvStride2 W b v) k ≠ 6)) :
    flatConvStride2Back (h := h) (w := w) W
      ∘ (bnPerChannelTensor3_has_vjp oc h w ε hε γ β).backward (flatConvStride2 W b v)
      ∘ reluMaskBack (fun i => 0 < bnPerChannelTensor3 oc h w ε γ β (flatConvStride2 W b v) i ∧
          bnPerChannelTensor3 oc h w ε γ β (flatConvStride2 W b v) i < 6)
      = (convStridedBnRelu6PC_has_vjp_at W b ε γ β hε v h_smooth).backward := by
  funext dy
  rw [flatConvStride2Back_eq_vjp_backward hkH hkW W b v]
  rfl

/-- **The HEAD tie.** -/
theorem convBnRelu6PCBack_eq_vjp_backward {ic oc h w kH kW : Nat}
    (hkH : 2 * ((kH - 1) / 2) + 1 = kH) (hkW : 2 * ((kW - 1) / 2) + 1 = kW)
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (γ β : Vec oc) (hε : 0 < ε)
    (v : Vec (ic * h * w))
    (h_smooth : ∀ k, (bnPerChannelTensor3 oc h w ε γ β (flatConv W b v) k ≠ 0 ∧
                       bnPerChannelTensor3 oc h w ε γ β (flatConv W b v) k ≠ 6)) :
    convFlatBack (h := h) (w := w) W
      ∘ (bnPerChannelTensor3_has_vjp oc h w ε hε γ β).backward (flatConv W b v)
      ∘ reluMaskBack (fun i => 0 < bnPerChannelTensor3 oc h w ε γ β (flatConv W b v) i ∧
          bnPerChannelTensor3 oc h w ε γ β (flatConv W b v) i < 6)
      = (convBnRelu6PC_has_vjp_at W b ε γ β hε v h_smooth).backward := by
  funext dy
  rw [convFlatBack_eq_vjp_backward (W := W) (b := b) (x := v) hkH hkW]
  rfl

/-- **The additive-skip tie.** -/
theorem residualBack_eq_vjp_backward {n : Nat} (f : Vec n → Vec n) (x : Vec n)
    (hf_diff : DifferentiableAt ℝ f x) (hf : HasVJPAt f x) :
    Proofs.residual hf.backward = (residual_has_vjp_at f x hf_diff hf).backward := rfl



/-- **Whole-network MobileNetV2 VJP.** The conditional VJP of an inverted-residual network at
    an input `x` — a STRAIGHT ten-stage chain:

      `dense ∘ GAP ∘ head ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ stem`

    ⭐ Simpler than `resnet34_has_vjp_at` in one structural way: MobileNetV2's skips live INSIDE
    the block maps (`residual (invresBodyPC …)` at `b2`/`b4`) and its stride changes live inside
    the strided bodies, so there is no `ChainData` list and no separate downsample slot — nine
    `vjp_comp_diff_at`s and nothing else. Dimension-generic and parametric in every component. -/
noncomputable def mobilenetv2PC_has_vjp_at
    {s0 s1 s2 s3 s4 s5 s6 s7 s8 : Nat}
    (stem : Vec s0 → Vec s1) (b1 : Vec s1 → Vec s2) (b2 : Vec s2 → Vec s2)
    (b3 : Vec s2 → Vec s3) (b4 : Vec s3 → Vec s3) (b5 : Vec s3 → Vec s4)
    (b6 : Vec s4 → Vec s5) (head : Vec s5 → Vec s6)
    (gap : Vec s6 → Vec s7) (dns : Vec s7 → Vec s8)
    (x : Vec s0)
    (hstem : PProd (HasVJPAt stem x) (DifferentiableAt ℝ stem x))
    (hb1 : PProd (HasVJPAt b1 (stem x)) (DifferentiableAt ℝ b1 (stem x)))
    (hb2 : PProd (HasVJPAt b2 (b1 (stem x))) (DifferentiableAt ℝ b2 (b1 (stem x))))
    (hb3 : PProd (HasVJPAt b3 (b2 (b1 (stem x))))
                 (DifferentiableAt ℝ b3 (b2 (b1 (stem x)))))
    (hb4 : PProd (HasVJPAt b4 (b3 (b2 (b1 (stem x)))))
                 (DifferentiableAt ℝ b4 (b3 (b2 (b1 (stem x))))))
    (hb5 : PProd (HasVJPAt b5 (b4 (b3 (b2 (b1 (stem x))))))
                 (DifferentiableAt ℝ b5 (b4 (b3 (b2 (b1 (stem x)))))))
    (hb6 : PProd (HasVJPAt b6 (b5 (b4 (b3 (b2 (b1 (stem x)))))))
                 (DifferentiableAt ℝ b6 (b5 (b4 (b3 (b2 (b1 (stem x))))))))
    (hhead : PProd (HasVJPAt head (b6 (b5 (b4 (b3 (b2 (b1 (stem x))))))))
                   (DifferentiableAt ℝ head (b6 (b5 (b4 (b3 (b2 (b1 (stem x)))))))))
    (hgap : PProd (HasVJPAt gap (head (b6 (b5 (b4 (b3 (b2 (b1 (stem x)))))))))
                  (DifferentiableAt ℝ gap (head (b6 (b5 (b4 (b3 (b2 (b1 (stem x))))))))))
    (hdns : PProd (HasVJPAt dns (gap (head (b6 (b5 (b4 (b3 (b2 (b1 (stem x))))))))))
                  (DifferentiableAt ℝ dns (gap (head (b6 (b5 (b4 (b3 (b2 (b1 (stem x)))))))))))
    : HasVJPAt (dns ∘ gap ∘ head ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ stem) x :=
  let p1 := vjp_comp_diff_at stem b1 x hstem hb1
  let p2 := vjp_comp_diff_at (b1 ∘ stem) b2 x p1 hb2
  let p3 := vjp_comp_diff_at (b2 ∘ b1 ∘ stem) b3 x p2 hb3
  let p4 := vjp_comp_diff_at (b3 ∘ b2 ∘ b1 ∘ stem) b4 x p3 hb4
  let p5 := vjp_comp_diff_at (b4 ∘ b3 ∘ b2 ∘ b1 ∘ stem) b5 x p4 hb5
  let p6 := vjp_comp_diff_at (b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ stem) b6 x p5 hb6
  let p7 := vjp_comp_diff_at (b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ stem) head x p6 hhead
  let p8 := vjp_comp_diff_at (head ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ stem) gap x p7 hgap
  let p9 := vjp_comp_diff_at (gap ∘ head ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ stem) dns x p8 hdns
  p9.fst



set_option maxRecDepth 400000 in
set_option maxHeartbeats 2000000 in
theorem mnv2InputGrad_eq_mobilenetv2_vjp
    (Ws : Kernel4 16 3 3 3) (bs : Vec 16) (εs : ℝ) (γs βs : Vec 16) (hεs : 0 < εs)
    (Wh : Kernel4 128 64 1 1) (bh : Vec 128) (εh : ℝ) (γh βh : Vec 128) (hεh : 0 < εh)
    (Wfc : Mat 128 10) (bfc : Vec 10)
    (b1 : Vec (16 * 112 * 112) → Vec (24 * 56 * 56))
    (b2 : Vec (24 * 56 * 56) → Vec (24 * 56 * 56))
    (b3 : Vec (24 * 56 * 56) → Vec (32 * 28 * 28))
    (b4 : Vec (32 * 28 * 28) → Vec (32 * 28 * 28))
    (b5 : Vec (32 * 28 * 28) → Vec (64 * 14 * 14))
    (b6 : Vec (64 * 14 * 14) → Vec (64 * 7 * 7))
    (x : Vec (3 * 224 * 224))
    (hstem_smooth : ∀ k,
      bnPerChannelTensor3 16 112 112 εs γs βs
        (flatConvStride2 (h := 112) (w := 112) Ws bs x) k ≠ 0 ∧
      bnPerChannelTensor3 16 112 112 εs γs βs
        (flatConvStride2 (h := 112) (w := 112) Ws bs x) k ≠ 6)
    (hb1 : PProd (HasVJPAt b1 ((relu6 (16 * 112 * 112) ∘ bnPerChannelTensor3 16 112 112 εs γs βs
              ∘ flatConvStride2 (h := 112) (w := 112) Ws bs) x))
             (DifferentiableAt ℝ b1 ((relu6 (16 * 112 * 112)
              ∘ bnPerChannelTensor3 16 112 112 εs γs βs
              ∘ flatConvStride2 (h := 112) (w := 112) Ws bs) x)))
    (hb2 : PProd (HasVJPAt b2 (b1 ((relu6 (16 * 112 * 112)
              ∘ bnPerChannelTensor3 16 112 112 εs γs βs
              ∘ flatConvStride2 (h := 112) (w := 112) Ws bs) x)))
             (DifferentiableAt ℝ b2 (b1 ((relu6 (16 * 112 * 112)
              ∘ bnPerChannelTensor3 16 112 112 εs γs βs
              ∘ flatConvStride2 (h := 112) (w := 112) Ws bs) x))))
    (hb3 : PProd (HasVJPAt b3 (b2 (b1 ((relu6 (16 * 112 * 112)
              ∘ bnPerChannelTensor3 16 112 112 εs γs βs
              ∘ flatConvStride2 (h := 112) (w := 112) Ws bs) x))))
             (DifferentiableAt ℝ b3 (b2 (b1 ((relu6 (16 * 112 * 112)
              ∘ bnPerChannelTensor3 16 112 112 εs γs βs
              ∘ flatConvStride2 (h := 112) (w := 112) Ws bs) x)))))
    (hb4 : PProd (HasVJPAt b4 (b3 (b2 (b1 ((relu6 (16 * 112 * 112)
              ∘ bnPerChannelTensor3 16 112 112 εs γs βs
              ∘ flatConvStride2 (h := 112) (w := 112) Ws bs) x)))))
             (DifferentiableAt ℝ b4 (b3 (b2 (b1 ((relu6 (16 * 112 * 112)
              ∘ bnPerChannelTensor3 16 112 112 εs γs βs
              ∘ flatConvStride2 (h := 112) (w := 112) Ws bs) x))))))
    (hb5 : PProd (HasVJPAt b5 (b4 (b3 (b2 (b1 ((relu6 (16 * 112 * 112)
              ∘ bnPerChannelTensor3 16 112 112 εs γs βs
              ∘ flatConvStride2 (h := 112) (w := 112) Ws bs) x))))))
             (DifferentiableAt ℝ b5 (b4 (b3 (b2 (b1 ((relu6 (16 * 112 * 112)
              ∘ bnPerChannelTensor3 16 112 112 εs γs βs
              ∘ flatConvStride2 (h := 112) (w := 112) Ws bs) x)))))))
    (hb6 : PProd (HasVJPAt b6 (b5 (b4 (b3 (b2 (b1 ((relu6 (16 * 112 * 112)
              ∘ bnPerChannelTensor3 16 112 112 εs γs βs
              ∘ flatConvStride2 (h := 112) (w := 112) Ws bs) x)))))))
             (DifferentiableAt ℝ b6 (b5 (b4 (b3 (b2 (b1 ((relu6 (16 * 112 * 112)
              ∘ bnPerChannelTensor3 16 112 112 εs γs βs
              ∘ flatConvStride2 (h := 112) (w := 112) Ws bs) x))))))))
    (hhead_smooth : ∀ k,
      bnPerChannelTensor3 128 7 7 εh γh βh (flatConv (h := 7) (w := 7) Wh bh
        (b6 (b5 (b4 (b3 (b2 (b1 ((relu6 (16 * 112 * 112)
          ∘ bnPerChannelTensor3 16 112 112 εs γs βs
          ∘ flatConvStride2 (h := 112) (w := 112) Ws bs) x)))))))) k ≠ 0 ∧
      bnPerChannelTensor3 128 7 7 εh γh βh (flatConv (h := 7) (w := 7) Wh bh
        (b6 (b5 (b4 (b3 (b2 (b1 ((relu6 (16 * 112 * 112)
          ∘ bnPerChannelTensor3 16 112 112 εs γs βs
          ∘ flatConvStride2 (h := 112) (w := 112) Ws bs) x)))))))) k ≠ 6) :
    mnv2InputGrad Ws Wh Wfc
      ((bnPerChannelTensor3_has_vjp 16 112 112 εs hεs γs βs).backward
        (flatConvStride2 (h := 112) (w := 112) Ws bs x))
      ((bnPerChannelTensor3_has_vjp 128 7 7 εh hεh γh βh).backward
        (flatConv (h := 7) (w := 7) Wh bh
          (b6 (b5 (b4 (b3 (b2 (b1 ((relu6 (16 * 112 * 112)
            ∘ bnPerChannelTensor3 16 112 112 εs γs βs
            ∘ flatConvStride2 (h := 112) (w := 112) Ws bs) x)))))))))
      hb1.fst.backward hb2.fst.backward hb3.fst.backward
      hb4.fst.backward hb5.fst.backward hb6.fst.backward
      (fun i => 0 < bnPerChannelTensor3 16 112 112 εs γs βs
                  (flatConvStride2 (h := 112) (w := 112) Ws bs x) i ∧
                bnPerChannelTensor3 16 112 112 εs γs βs
                  (flatConvStride2 (h := 112) (w := 112) Ws bs x) i < 6)
      (fun i => 0 < bnPerChannelTensor3 128 7 7 εh γh βh (flatConv (h := 7) (w := 7) Wh bh
                  (b6 (b5 (b4 (b3 (b2 (b1 ((relu6 (16 * 112 * 112)
                    ∘ bnPerChannelTensor3 16 112 112 εs γs βs
                    ∘ flatConvStride2 (h := 112) (w := 112) Ws bs) x)))))))) i ∧
                bnPerChannelTensor3 128 7 7 εh γh βh (flatConv (h := 7) (w := 7) Wh bh
                  (b6 (b5 (b4 (b3 (b2 (b1 ((relu6 (16 * 112 * 112)
                    ∘ bnPerChannelTensor3 16 112 112 εs γs βs
                    ∘ flatConvStride2 (h := 112) (w := 112) Ws bs) x)))))))) i < 6)
      = (mobilenetv2PC_has_vjp_at
          (relu6 (16 * 112 * 112) ∘ bnPerChannelTensor3 16 112 112 εs γs βs
            ∘ flatConvStride2 (h := 112) (w := 112) Ws bs)
          b1 b2 b3 b4 b5 b6
          (relu6 (128 * 7 * 7) ∘ bnPerChannelTensor3 128 7 7 εh γh βh
            ∘ flatConv (h := 7) (w := 7) Wh bh)
          (globalAvgPoolFlat 128 7 7) (dense Wfc bfc) x
          ⟨convStridedBnRelu6PC_has_vjp_at (ic := 3) (oc := 16) (h := 112) (w := 112)
              Ws bs εs γs βs hεs x hstem_smooth,
            convStridedBnRelu6PC_differentiableAt (ic := 3) (oc := 16) (h := 112) (w := 112)
              Ws bs εs γs βs hεs x hstem_smooth⟩
          hb1 hb2 hb3 hb4 hb5 hb6
          ⟨convBnRelu6PC_has_vjp_at (ic := 64) (oc := 128) (h := 7) (w := 7)
              Wh bh εh γh βh hεh _ hhead_smooth,
            convBnRelu6PC_differentiableAt (ic := 64) (oc := 128) (h := 7) (w := 7)
              Wh bh εh γh βh hεh _ hhead_smooth⟩
          ⟨(globalAvgPoolFlat_has_vjp 128 7 7).toHasVJPAt _,
            (globalAvgPoolFlat_differentiable 128 7 7) _⟩
          ⟨(dense_has_vjp Wfc bfc).toHasVJPAt _, (dense_differentiable Wfc bfc) _⟩).backward := by
  unfold mnv2InputGrad
  rw [convStridedBnRelu6PCBack_eq_vjp_backward (ic := 3) (oc := 16) (h := 112) (w := 112)
        (by decide) (by decide) Ws bs εs γs βs hεs x hstem_smooth,
      convBnRelu6PCBack_eq_vjp_backward (ic := 64) (oc := 128) (h := 7) (w := 7)
        (by decide) (by decide) Wh bh εh γh βh hεh _ hhead_smooth,
      dense_transpose_eq_vjp_backward Wfc bfc
        (globalAvgPoolFlat 128 7 7 ((relu6 (128 * 7 * 7)
          ∘ bnPerChannelTensor3 128 7 7 εh γh βh ∘ flatConv (h := 7) (w := 7) Wh bh)
          (b6 (b5 (b4 (b3 (b2 (b1 ((relu6 (16 * 112 * 112)
            ∘ bnPerChannelTensor3 16 112 112 εs γs βs
            ∘ flatConvStride2 (h := 112) (w := 112) Ws bs) x)))))))))]
  rfl


/-- ⭐⭐ **THE SHAPE CHECK — the ten-stage chain the tie is about IS the committed forward.**
    `mobilenetv2Forward_full_pc` by `rfl`, with the six block slots read off: `b1/b3/b5/b6` the
    strided inverted-residual bodies, `b2/b4` those bodies under `Proofs.residual`, the stem
    `relu6 ∘ bnPC ∘ flatConvStride2` and the head `relu6 ∘ bnPC ∘ flatConv`.

    ⛔ **This is the theorem that would have caught ResNet-34's wrong pool.** §3.10's drift lived
    a month because *"the same net as the tie"* was prose in a docstring; here it is a `rfl` the
    kernel checks, and `mnv2InputGrad_eq_mobilenetv2_vjp` instantiates `mobilenetv2PC_has_vjp_at`
    at exactly these ten slots. -/
theorem mobilenetv2Forward_full_pc_eq_chain
    (Ws : Kernel4 16 3 3 3) (bs : Vec 16) (εs : ℝ) (γs βs : Vec 16)
    (We1 : Kernel4 64 16 1 1) (be1 : Vec 64) (εe1 : ℝ) (γe1 βe1 : Vec 64)
    (Wd1 : DepthwiseKernel 64 3 3) (bd1 : Vec 64) (εd1 : ℝ) (γd1 βd1 : Vec 64)
    (Wp1 : Kernel4 24 64 1 1) (bp1 : Vec 24) (εp1 : ℝ) (γp1 βp1 : Vec 24)
    (We2 : Kernel4 96 24 1 1) (be2 : Vec 96) (εe2 : ℝ) (γe2 βe2 : Vec 96)
    (Wd2 : DepthwiseKernel 96 3 3) (bd2 : Vec 96) (εd2 : ℝ) (γd2 βd2 : Vec 96)
    (Wp2 : Kernel4 24 96 1 1) (bp2 : Vec 24) (εp2 : ℝ) (γp2 βp2 : Vec 24)
    (We3 : Kernel4 96 24 1 1) (be3 : Vec 96) (εe3 : ℝ) (γe3 βe3 : Vec 96)
    (Wd3 : DepthwiseKernel 96 3 3) (bd3 : Vec 96) (εd3 : ℝ) (γd3 βd3 : Vec 96)
    (Wp3 : Kernel4 32 96 1 1) (bp3 : Vec 32) (εp3 : ℝ) (γp3 βp3 : Vec 32)
    (We4 : Kernel4 128 32 1 1) (be4 : Vec 128) (εe4 : ℝ) (γe4 βe4 : Vec 128)
    (Wd4 : DepthwiseKernel 128 3 3) (bd4 : Vec 128) (εd4 : ℝ) (γd4 βd4 : Vec 128)
    (Wp4 : Kernel4 32 128 1 1) (bp4 : Vec 32) (εp4 : ℝ) (γp4 βp4 : Vec 32)
    (We5 : Kernel4 128 32 1 1) (be5 : Vec 128) (εe5 : ℝ) (γe5 βe5 : Vec 128)
    (Wd5 : DepthwiseKernel 128 3 3) (bd5 : Vec 128) (εd5 : ℝ) (γd5 βd5 : Vec 128)
    (Wp5 : Kernel4 64 128 1 1) (bp5 : Vec 64) (εp5 : ℝ) (γp5 βp5 : Vec 64)
    (We6 : Kernel4 256 64 1 1) (be6 : Vec 256) (εe6 : ℝ) (γe6 βe6 : Vec 256)
    (Wd6 : DepthwiseKernel 256 3 3) (bd6 : Vec 256) (εd6 : ℝ) (γd6 βd6 : Vec 256)
    (Wp6 : Kernel4 64 256 1 1) (bp6 : Vec 64) (εp6 : ℝ) (γp6 βp6 : Vec 64)
    (Wh : Kernel4 128 64 1 1) (bh : Vec 128) (εh : ℝ) (γh βh : Vec 128)
    (Wfc : Mat 128 10) (bfc : Vec 10) :
    mobilenetv2Forward_full_pc
      Ws bs εs γs βs We1 be1 εe1 γe1 βe1 Wd1 bd1 εd1 γd1 βd1 Wp1 bp1 εp1 γp1 βp1 We2 be2 εe2
      γe2 βe2 Wd2 bd2 εd2 γd2 βd2 Wp2 bp2 εp2 γp2 βp2 We3 be3 εe3 γe3 βe3 Wd3 bd3 εd3 γd3
      βd3 Wp3 bp3 εp3 γp3 βp3 We4 be4 εe4 γe4 βe4 Wd4 bd4 εd4 γd4 βd4 Wp4 bp4 εp4 γp4 βp4
      We5 be5 εe5 γe5 βe5 Wd5 bd5 εd5 γd5 βd5 Wp5 bp5 εp5 γp5 βp5 We6 be6 εe6 γe6 βe6 Wd6
      bd6 εd6 γd6 βd6 Wp6 bp6 εp6 γp6 βp6 Wh bh εh γh βh Wfc bfc
      = dense Wfc bfc
        ∘ globalAvgPoolFlat 128 7 7
        ∘ (relu6 (128 * 7 * 7) ∘ bnPerChannelTensor3 128 7 7 εh γh βh
            ∘ flatConv (h := 7) (w := 7) Wh bh)
        ∘ invresBodyStridedPC (h := 7) (w := 7)
            We6 be6 εe6 γe6 βe6 Wd6 bd6 εd6 γd6 βd6 Wp6 bp6 εp6 γp6 βp6
        ∘ invresBodyStridedPC (h := 14) (w := 14)
            We5 be5 εe5 γe5 βe5 Wd5 bd5 εd5 γd5 βd5 Wp5 bp5 εp5 γp5 βp5
        ∘ Proofs.residual (invresBodyPC (h := 28) (w := 28)
            We4 be4 εe4 γe4 βe4 Wd4 bd4 εd4 γd4 βd4 Wp4 bp4 εp4 γp4 βp4)
        ∘ invresBodyStridedPC (h := 28) (w := 28)
            We3 be3 εe3 γe3 βe3 Wd3 bd3 εd3 γd3 βd3 Wp3 bp3 εp3 γp3 βp3
        ∘ Proofs.residual (invresBodyPC (h := 56) (w := 56)
            We2 be2 εe2 γe2 βe2 Wd2 bd2 εd2 γd2 βd2 Wp2 bp2 εp2 γp2 βp2)
        ∘ invresBodyStridedPC (h := 56) (w := 56)
            We1 be1 εe1 γe1 βe1 Wd1 bd1 εd1 γd1 βd1 Wp1 bp1 εp1 γp1 βp1
        ∘ (relu6 (16 * 112 * 112) ∘ bnPerChannelTensor3 16 112 112 εs γs βs
            ∘ flatConvStride2 (h := 112) (w := 112) Ws bs) := rfl

end Proofs

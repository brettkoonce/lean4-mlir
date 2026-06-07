import LeanMlir.Proofs.CNN
import LeanMlir.Proofs.StridedConv

/-! # Toward real ResNet-34 — the deep-block chain (Chapter 6 Milestone B4)

A real ResNet-34 stacks **16 basic blocks** in four stages (3+4+6+3). Within a
stage every block is a self-map `Vec n → Vec n` (same channel count) but with its
**own** weights — so it is a *composition of a list* of distinct same-type maps,
not an `iterate` of one map.

This file proves the generic enabler: if every map in a list is differentiable
and has a VJP, their composition (`chainComp`) does too — by induction chaining
`vjp_comp`. That turns "16 blocks deep" into a `List.length`, no per-block
boilerplate. The full ResNet-34 forward (strided proj blocks via `flatConvStride2`
+ chained identity blocks + per-channel BN) is assembled on top of this.

Closes under `[propext, Classical.choice, Quot.sound]`.
-/

open Finset BigOperators

namespace Proofs

-- ════════════════════════════════════════════════════════════════
-- § Composition of a list of same-type self-maps
-- ════════════════════════════════════════════════════════════════

/-- Compose a list of self-maps left-to-right as data flows: `chainComp [f₁,…,fₖ]
    = f₁ ∘ … ∘ fₖ` (the last list element runs first, i.e. is the deepest). A
    ResNet stage is `chainComp` of its blocks. -/
noncomputable def chainComp {n : Nat} (fs : List (Vec n → Vec n)) : Vec n → Vec n :=
  fs.foldr (· ∘ ·) id

@[simp] theorem chainComp_nil {n : Nat} : chainComp ([] : List (Vec n → Vec n)) = id := rfl

@[simp] theorem chainComp_cons {n : Nat} (f : Vec n → Vec n) (fs : List (Vec n → Vec n)) :
    chainComp (f :: fs) = f ∘ chainComp fs := rfl

/-- A chain of differentiable maps is differentiable. -/
theorem chainComp_differentiable {n : Nat} (fs : List (Vec n → Vec n))
    (hdiff : ∀ f ∈ fs, Differentiable ℝ f) : Differentiable ℝ (chainComp fs) := by
  induction fs with
  | nil => exact differentiable_id
  | cons f fs ih =>
    rw [chainComp_cons]
    exact (hdiff f (List.mem_cons.2 (Or.inl rfl))).comp
      (ih (fun g hg => hdiff g (List.mem_cons.2 (Or.inr hg))))

/-- **Deep-chain VJP.** A composition of a list of differentiable maps that each
    have a VJP has a VJP — the backward runs each block's backward in reverse
    order. By induction chaining `vjp_comp`; the structural heart of a deep
    ResNet stage (k distinct-weight basic blocks). -/
noncomputable def vjp_chain {n : Nat} (fs : List (Vec n → Vec n))
    (hdiff : ∀ f ∈ fs, Differentiable ℝ f) (hvjp : ∀ f ∈ fs, HasVJP f) :
    HasVJP (chainComp fs) :=
  match fs with
  | [] => show HasVJP (id : Vec n → Vec n) from identity_has_vjp n
  | f :: rest =>
    show HasVJP (f ∘ chainComp rest) from
    vjp_comp (chainComp rest) f
      (chainComp_differentiable rest (fun g hg => hdiff g (List.mem_cons.2 (Or.inr hg))))
      (hdiff f (List.mem_cons.2 (Or.inl rfl)))
      (vjp_chain rest (fun g hg => hdiff g (List.mem_cons.2 (Or.inr hg)))
                      (fun g hg => hvjp g (List.mem_cons.2 (Or.inr hg))))
      (hvjp f (List.mem_cons.2 (Or.inl rfl)))

/-- **Deep-chain VJP correctness** (ℝ-headline): the chained backward equals the
    `pdiv`-contracted Jacobian of the whole composition. -/
theorem vjp_chain_correct {n : Nat} (fs : List (Vec n → Vec n))
    (hdiff : ∀ f ∈ fs, Differentiable ℝ f) (hvjp : ∀ f ∈ fs, HasVJP f)
    (x dy : Vec n) (i : Fin n) :
    (vjp_chain fs hdiff hvjp).backward x dy i
      = ∑ j : Fin n, pdiv (chainComp fs) x i j * dy j :=
  (vjp_chain fs hdiff hvjp).correct x dy i

-- ════════════════════════════════════════════════════════════════
-- § Strided downsampling block — conv(stride 2) → BN → relu
-- ════════════════════════════════════════════════════════════════

/-- **conv(stride-2) → bn block VJP (no ReLU), everywhere.** The strided peer of
    `convBn_has_vjp`: `flatConvStride2` then `bnForward`, both differentiable
    everywhere, so a global `HasVJP`. The downsampling body of a stage-start
    block and its strided 1×1 projection skip. -/
noncomputable def convBnStrided_has_vjp {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε γ β : ℝ) (hε : 0 < ε) :
    HasVJP (bnForward (oc * h * w) ε γ β ∘ flatConvStride2 W b
      : Vec (ic * (2 * h) * (2 * w)) → Vec (oc * h * w)) :=
  vjp_comp (flatConvStride2 W b) (bnForward (oc * h * w) ε γ β)
    (flatConvStride2_differentiable W b)
    (bnForward_differentiable (oc * h * w) ε γ β hε)
    (flatConvStride2_has_vjp W b)
    (bn_has_vjp (oc * h * w) ε γ β hε)

/-- **conv(stride-2) → bn is differentiable everywhere.** -/
theorem convBnStrided_differentiable {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε γ β : ℝ) (hε : 0 < ε) :
    Differentiable ℝ (bnForward (oc * h * w) ε γ β ∘ flatConvStride2 W b
      : Vec (ic * (2 * h) * (2 * w)) → Vec (oc * h * w)) :=
  (bnForward_differentiable (oc * h * w) ε γ β hε).comp (flatConvStride2_differentiable W b)

/-- **conv(stride-2) → bn → relu block VJP at a smooth point.** The strided peer
    of `convBnRelu_has_vjp_at` (the workhorse opening each downsampling stage):
    two `vjp_comp_at`, with `flatConvStride2_has_vjp` for the conv and the ReLU
    smoothness hypothesis `h_smooth` (no post-BN activation hits the kink). -/
noncomputable def convBnReluStrided_has_vjp_at {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε γ β : ℝ) (hε : 0 < ε)
    (v : Vec (ic * (2 * h) * (2 * w)))
    (h_smooth : ∀ k, bnForward (oc * h * w) ε γ β (flatConvStride2 W b v) k ≠ 0) :
    HasVJPAt (relu (oc * h * w) ∘ bnForward (oc * h * w) ε γ β ∘ flatConvStride2 W b) v := by
  have hconv_diff : Differentiable ℝ
      (flatConvStride2 W b : Vec (ic * (2 * h) * (2 * w)) → Vec (oc * h * w)) :=
    flatConvStride2_differentiable W b
  have hbn_diff : Differentiable ℝ (bnForward (oc * h * w) ε γ β) :=
    bnForward_differentiable (oc * h * w) ε γ β hε
  have step1 : HasVJPAt (bnForward (oc * h * w) ε γ β ∘ flatConvStride2 W b) v :=
    vjp_comp_at (flatConvStride2 W b) (bnForward (oc * h * w) ε γ β) v
      (hconv_diff v) (hbn_diff _)
      ((flatConvStride2_has_vjp W b).toHasVJPAt v)
      ((bn_has_vjp (oc * h * w) ε γ β hε).toHasVJPAt _)
  have step1_diff : DifferentiableAt ℝ
      (bnForward (oc * h * w) ε γ β ∘ flatConvStride2 W b) v :=
    DifferentiableAt.comp v (hbn_diff _) (hconv_diff v)
  exact vjp_comp_at (bnForward (oc * h * w) ε γ β ∘ flatConvStride2 W b)
    (relu (oc * h * w)) v step1_diff
    (relu_differentiableAt_of_smooth (oc * h * w) _ h_smooth) step1
    (relu_has_vjp_at (oc * h * w) _ h_smooth)

/-- **Strided block VJP correctness** (ℝ-headline): the strided downsampling
    block's backward equals the `pdiv`-Jacobian of `relu ∘ bn ∘ conv_stride2`. -/
theorem convBnReluStrided_has_vjp_at_correct {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε γ β : ℝ) (hε : 0 < ε)
    (v : Vec (ic * (2 * h) * (2 * w)))
    (h_smooth : ∀ k, bnForward (oc * h * w) ε γ β (flatConvStride2 W b v) k ≠ 0)
    (dy : Vec (oc * h * w)) (i : Fin (ic * (2 * h) * (2 * w))) :
    (convBnReluStrided_has_vjp_at W b ε γ β hε v h_smooth).backward dy i
      = ∑ j : Fin (oc * h * w),
          pdiv (relu (oc * h * w) ∘ bnForward (oc * h * w) ε γ β ∘ flatConvStride2 W b) v i j * dy j :=
  (convBnReluStrided_has_vjp_at W b ε γ β hε v h_smooth).correct dy i

-- ════════════════════════════════════════════════════════════════
-- § Strided residual-projection block (the stage-start downsampling block)
-- ════════════════════════════════════════════════════════════════

/-- **Strided basic-block body VJP** `F = convBn₂(stride 1) ∘ convBnRelu₁(stride 2)`
    (channels `ic → oc`, spatial `2h×2w → h×w`). The strided peer of
    `resblock_body_has_vjp_at`: inner downsampling conv→bn→relu (needs `h_smooth₁`),
    outer stride-1 conv→bn (everywhere); two `vjp_comp_at`. -/
noncomputable def resblock_bodyStrided_has_vjp_at {ic oc h w kH₁ kW₁ kH₂ kW₂ : Nat}
    (W₁ : Kernel4 oc ic kH₁ kW₁) (b₁ : Vec oc)
    (W₂ : Kernel4 oc oc kH₂ kW₂) (b₂ : Vec oc)
    (ε₁ γ₁ β₁ ε₂ γ₂ β₂ : ℝ) (hε₁ : 0 < ε₁) (hε₂ : 0 < ε₂)
    (v : Vec (ic * (2 * h) * (2 * w)))
    (h_smooth₁ : ∀ k, bnForward (oc * h * w) ε₁ γ₁ β₁ (flatConvStride2 W₁ b₁ v) k ≠ 0) :
    HasVJPAt
      ((bnForward (oc * h * w) ε₂ γ₂ β₂ ∘ flatConv W₂ b₂) ∘
        (relu (oc * h * w) ∘ bnForward (oc * h * w) ε₁ γ₁ β₁ ∘ flatConvStride2 W₁ b₁)) v := by
  have hconv1_diff : Differentiable ℝ
      (flatConvStride2 W₁ b₁ : Vec (ic * (2 * h) * (2 * w)) → Vec (oc * h * w)) :=
    flatConvStride2_differentiable W₁ b₁
  have step1 : HasVJPAt
      (relu (oc * h * w) ∘ bnForward (oc * h * w) ε₁ γ₁ β₁ ∘ flatConvStride2 W₁ b₁) v :=
    convBnReluStrided_has_vjp_at W₁ b₁ ε₁ γ₁ β₁ hε₁ v h_smooth₁
  have step1_diff : DifferentiableAt ℝ
      (relu (oc * h * w) ∘ bnForward (oc * h * w) ε₁ γ₁ β₁ ∘ flatConvStride2 W₁ b₁) v := by
    apply DifferentiableAt.comp
    · exact relu_differentiableAt_of_smooth (oc * h * w) _ h_smooth₁
    · exact ((bnForward_differentiable (oc * h * w) ε₁ γ₁ β₁ hε₁).comp hconv1_diff) v
  exact vjp_comp_at
    (relu (oc * h * w) ∘ bnForward (oc * h * w) ε₁ γ₁ β₁ ∘ flatConvStride2 W₁ b₁)
    (bnForward (oc * h * w) ε₂ γ₂ β₂ ∘ flatConv W₂ b₂) v
    step1_diff
    ((convBn_differentiable W₂ b₂ ε₂ γ₂ β₂ hε₂) _)
    step1
    ((convBn_has_vjp W₂ b₂ ε₂ γ₂ β₂ hε₂).toHasVJPAt _)

/-- **Strided basic-block body is `DifferentiableAt`** at a smooth point. -/
theorem resblock_bodyStrided_differentiableAt {ic oc h w kH₁ kW₁ kH₂ kW₂ : Nat}
    (W₁ : Kernel4 oc ic kH₁ kW₁) (b₁ : Vec oc)
    (W₂ : Kernel4 oc oc kH₂ kW₂) (b₂ : Vec oc)
    (ε₁ γ₁ β₁ ε₂ γ₂ β₂ : ℝ) (hε₁ : 0 < ε₁) (hε₂ : 0 < ε₂)
    (v : Vec (ic * (2 * h) * (2 * w)))
    (h_smooth₁ : ∀ k, bnForward (oc * h * w) ε₁ γ₁ β₁ (flatConvStride2 W₁ b₁ v) k ≠ 0) :
    DifferentiableAt ℝ
      ((bnForward (oc * h * w) ε₂ γ₂ β₂ ∘ flatConv W₂ b₂) ∘
        (relu (oc * h * w) ∘ bnForward (oc * h * w) ε₁ γ₁ β₁ ∘ flatConvStride2 W₁ b₁)) v := by
  have hconv1_diff : Differentiable ℝ
      (flatConvStride2 W₁ b₁ : Vec (ic * (2 * h) * (2 * w)) → Vec (oc * h * w)) :=
    flatConvStride2_differentiable W₁ b₁
  apply DifferentiableAt.comp
  · exact (convBn_differentiable W₂ b₂ ε₂ γ₂ β₂ hε₂) _
  · apply DifferentiableAt.comp
    · exact relu_differentiableAt_of_smooth (oc * h * w) _ h_smooth₁
    · exact ((bnForward_differentiable (oc * h * w) ε₁ γ₁ β₁ hε₁).comp hconv1_diff) v

/-- **Full strided residual-projection block VJP** — the block that opens each
    ResNet-34 downsampling stage: `relu( proj(x) + F(x) )` where both the body's
    first conv `W₁` and the 1×1 projection skip `Wp` are stride-2 (so `ic→oc`,
    `2h×2w → h×w`), and the body's second conv `W₂` is stride-1. Built via
    `residualProj_has_vjp_at` (fan-in of the strided proj `convBnStrided` and the
    strided body) then a final `vjp_comp_at` with the post-add ReLU. The strided
    peer of `resblockProj_has_vjp_at`. -/
noncomputable def rblkPStrided_has_vjp_at
    {ic oc h w kH₁ kW₁ kH₂ kW₂ kHp kWp : Nat}
    (W₁ : Kernel4 oc ic kH₁ kW₁) (b₁ : Vec oc)
    (W₂ : Kernel4 oc oc kH₂ kW₂) (b₂ : Vec oc)
    (Wp : Kernel4 oc ic kHp kWp) (bp : Vec oc)
    (ε₁ γ₁ β₁ ε₂ γ₂ β₂ εp γp βp : ℝ)
    (hε₁ : 0 < ε₁) (hε₂ : 0 < ε₂) (hεp : 0 < εp)
    (v : Vec (ic * (2 * h) * (2 * w)))
    (h_smooth₁ : ∀ k, bnForward (oc * h * w) ε₁ γ₁ β₁ (flatConvStride2 W₁ b₁ v) k ≠ 0)
    (h_smooth_out : ∀ k,
      ((bnForward (oc * h * w) εp γp βp ∘ flatConvStride2 Wp bp) v k)
      + ((bnForward (oc * h * w) ε₂ γ₂ β₂ ∘ flatConv W₂ b₂) ∘
          (relu (oc * h * w) ∘ bnForward (oc * h * w) ε₁ γ₁ β₁ ∘ flatConvStride2 W₁ b₁)) v k
        ≠ 0) :
    HasVJPAt
      (relu (oc * h * w) ∘
        residualProj
          (bnForward (oc * h * w) εp γp βp ∘ flatConvStride2 Wp bp)
          ((bnForward (oc * h * w) ε₂ γ₂ β₂ ∘ flatConv W₂ b₂) ∘
            (relu (oc * h * w) ∘ bnForward (oc * h * w) ε₁ γ₁ β₁ ∘ flatConvStride2 W₁ b₁))) v := by
  let proj : Vec (ic * (2 * h) * (2 * w)) → Vec (oc * h * w) :=
    bnForward (oc * h * w) εp γp βp ∘ flatConvStride2 Wp bp
  let F : Vec (ic * (2 * h) * (2 * w)) → Vec (oc * h * w) :=
    (bnForward (oc * h * w) ε₂ γ₂ β₂ ∘ flatConv W₂ b₂) ∘
      (relu (oc * h * w) ∘ bnForward (oc * h * w) ε₁ γ₁ β₁ ∘ flatConvStride2 W₁ b₁)
  show HasVJPAt (relu (oc * h * w) ∘ residualProj proj F) v
  have hproj_diff : DifferentiableAt ℝ proj v :=
    (convBnStrided_differentiable Wp bp εp γp βp hεp) v
  have hproj_vjp : HasVJPAt proj v :=
    (convBnStrided_has_vjp Wp bp εp γp βp hεp).toHasVJPAt v
  have hF_diff : DifferentiableAt ℝ F v :=
    resblock_bodyStrided_differentiableAt W₁ b₁ W₂ b₂ ε₁ γ₁ β₁ ε₂ γ₂ β₂ hε₁ hε₂ v h_smooth₁
  have hF : HasVJPAt F v :=
    resblock_bodyStrided_has_vjp_at W₁ b₁ W₂ b₂ ε₁ γ₁ β₁ ε₂ γ₂ β₂ hε₁ hε₂ v h_smooth₁
  have hres : HasVJPAt (residualProj proj F) v :=
    residualProj_has_vjp_at proj F v hproj_diff hF_diff hproj_vjp hF
  have hres_diff : DifferentiableAt ℝ (residualProj proj F) v :=
    DifferentiableAt.add hproj_diff hF_diff
  have h_smooth_res : ∀ k, residualProj proj F v k ≠ 0 := h_smooth_out
  exact vjp_comp_at (residualProj proj F) (relu (oc * h * w)) v
    hres_diff
    (relu_differentiableAt_of_smooth (oc * h * w) _ h_smooth_res)
    hres
    (relu_has_vjp_at (oc * h * w) _ h_smooth_res)

/-- **Strided residual-projection block VJP correctness** (ℝ-headline): the
    downsampling block's backward equals the `pdiv`-Jacobian of
    `relu ∘ residualProj (strided proj) (strided body)`. -/
theorem rblkPStrided_has_vjp_at_correct
    {ic oc h w kH₁ kW₁ kH₂ kW₂ kHp kWp : Nat}
    (W₁ : Kernel4 oc ic kH₁ kW₁) (b₁ : Vec oc)
    (W₂ : Kernel4 oc oc kH₂ kW₂) (b₂ : Vec oc)
    (Wp : Kernel4 oc ic kHp kWp) (bp : Vec oc)
    (ε₁ γ₁ β₁ ε₂ γ₂ β₂ εp γp βp : ℝ)
    (hε₁ : 0 < ε₁) (hε₂ : 0 < ε₂) (hεp : 0 < εp)
    (v : Vec (ic * (2 * h) * (2 * w)))
    (h_smooth₁ : ∀ k, bnForward (oc * h * w) ε₁ γ₁ β₁ (flatConvStride2 W₁ b₁ v) k ≠ 0)
    (h_smooth_out : ∀ k,
      ((bnForward (oc * h * w) εp γp βp ∘ flatConvStride2 Wp bp) v k)
      + ((bnForward (oc * h * w) ε₂ γ₂ β₂ ∘ flatConv W₂ b₂) ∘
          (relu (oc * h * w) ∘ bnForward (oc * h * w) ε₁ γ₁ β₁ ∘ flatConvStride2 W₁ b₁)) v k
        ≠ 0)
    (dy : Vec (oc * h * w)) (i : Fin (ic * (2 * h) * (2 * w))) :
    (rblkPStrided_has_vjp_at W₁ b₁ W₂ b₂ Wp bp ε₁ γ₁ β₁ ε₂ γ₂ β₂ εp γp βp
        hε₁ hε₂ hεp v h_smooth₁ h_smooth_out).backward dy i
      = ∑ j : Fin (oc * h * w),
          pdiv (relu (oc * h * w) ∘
            residualProj
              (bnForward (oc * h * w) εp γp βp ∘ flatConvStride2 Wp bp)
              ((bnForward (oc * h * w) ε₂ γ₂ β₂ ∘ flatConv W₂ b₂) ∘
                (relu (oc * h * w) ∘ bnForward (oc * h * w) ε₁ γ₁ β₁ ∘ flatConvStride2 W₁ b₁)))
            v i j * dy j :=
  (rblkPStrided_has_vjp_at W₁ b₁ W₂ b₂ Wp bp ε₁ γ₁ β₁ ε₂ γ₂ β₂ εp γp βp
      hε₁ hε₂ hεp v h_smooth₁ h_smooth_out).correct dy i

end Proofs

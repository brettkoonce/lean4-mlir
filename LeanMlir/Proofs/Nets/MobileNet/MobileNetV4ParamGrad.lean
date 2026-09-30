import LeanMlir.Proofs.Nets.MobileNet.MobileNetV4StepTieB
import LeanMlir.Proofs.Foundation.ParamGradNodes

/-! # MobileNetV4-Conv-M — every parameter gradient node IS the loss's derivative in that parameter

`mnv4_net_tiedB` says each of the 233 parameter gradient nodes denotes its layer's parameter
Jacobian contracted with the cotangent the emitted backward chain threads to it, from a loss
cotangent `g`; the `*CotIn_eq_vjp` lemmas say each block's input cotangent is its `CertLayer`'s
certified backward. `mnv4_net_lossGrad` composes them: for any loss `L` of the logits whose gradient
at the net's output is `g`, every node is `∂L/∂θ` of the WHOLE net with that one parameter varied.
`mnv4_net_lossGrad_smoothedCE` discharges `hL` for the label-smoothed loss the artifacts ship.

**How.** `ResNet50ParamGrad`'s shape, with two things MNv4 adds:

* **The depthwise slots.** A UIB body's pre- and post-depthwise are `if k = 0 then id' else …`
  layers read off the row, and the depthwise weights live in a `DWSlot` that is a `DWBnParams` only
  at `k > 0`. So each family bundle (`mnv4ExtraDWLossTiedB`, `mnv4ConvNeXtLossTiedB`,
  `mnv4FfnLossTiedB`, `mnv4StridedLossTiedB`) takes its row's `k ≠ 0` / `k = 0` facts, the slots'
  forwards reduce by `mnv4PreDWSlot_fwd_of_ne_zero` and friends, and a depthwise parameter is varied
  through `UibParams.withPre` / `withPost` (the slot record with one field replaced).
* **The layer-group trunk.** The forward nests the five resolution groups, not the 21 blocks, so
  each block's factor lemma peels its group with `CertLayer.comp_fwd_apply`, and `mnv4Pre{k}` meets
  `mnv4Blk{k}` at the group boundaries (`mnv4Pre2_eq_blk` …). At the net's literal widths these
  checks stay cheap because every peel is a named rewrite, never an unfolding.

**Hypotheses.** `Mnv4SmoothAt` (every relu off its kink at the real activations — the stem's
clause and each group's `.ok`); the BN `ε > 0` facts live in the weights. For the smoothed loss,
every example's target summing to one and `0 < nCls`.
-/

open Proofs Proofs.StableHLO

namespace Proofs.StableHLO.UibParams

variable {s : UibSpec}

/-- The record with its pre-depthwise slot's parameters replaced. -/
def withPre (p : UibParams s) (q : DWBnParams s.ic s.preDWk) : UibParams s :=
  { p with pre := DWSlot.ofParams q }

/-- The record with its post-depthwise slot's parameters replaced. -/
def withPost (p : UibParams s) (q : DWBnParams (s.ic * s.expand) s.postDWk) : UibParams s :=
  { p with post := DWSlot.ofParams q }

theorem withPre_pre_params (p : UibParams s) (q : DWBnParams s.ic s.preDWk)
    (hk : s.preDWk ≠ 0) : (p.withPre q).pre.params = q :=
  DWSlot.params_ofParams hk q

theorem withPost_post_params (p : UibParams s) (q : DWBnParams (s.ic * s.expand) s.postDWk)
    (hk : s.postDWk ≠ 0) : (p.withPost q).post.params = q :=
  DWSlot.params_ofParams hk q

end Proofs.StableHLO.UibParams

namespace Proofs.Mnv4TieB

open Proofs.BackLinks (bnInB reluMaskB cInB dInB cStridedInB dStridedInB gapInB reassocB rowB unrowB)
open Proofs.GradNodeB (hasGradAt_bnBatchLA hasGradAt_relu hasGradAt_conv hasGradAt_convStrided
  hasGradAt_depthwise hasGradAt_depthwiseStrided)
open scoped BigOperators

-- ════════════════════════════════════════════════════════════════
-- § The slots' forwards, by the row's kernel size
-- ════════════════════════════════════════════════════════════════

theorem mnv4PreDWSlot_fwd_of_ne_zero (N : Nat) {c h w kH kW : Nat} {k : Nat} (hk : k ≠ 0)
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε : ℝ) (hε : 0 < ε) (γ β : Vec c) :
    (mnv4PreDWSlot (h := h) (w := w) N k W b ε hε γ β).fwd = dwbB N W b ε γ β := by
  rw [mnv4PreDWSlot_of_ne_zero N hk]; rfl

theorem mnv4PreDWSlot_fwd_of_eq_zero (N : Nat) {c h w kH kW : Nat} {k : Nat} (hk : k = 0)
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε : ℝ) (hε : 0 < ε) (γ β : Vec c) :
    (mnv4PreDWSlot (h := h) (w := w) N k W b ε hε γ β).fwd = fun y => y := by
  rw [mnv4PreDWSlot_of_eq_zero N hk]; rfl

theorem mnv4PostDWSlot_fwd_of_ne_zero (N : Nat) {c h w kH kW : Nat} {k : Nat} (hk : k ≠ 0)
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε : ℝ) (hε : 0 < ε) (γ β : Vec c) :
    (mnv4PostDWSlot (h := h) (w := w) N k W b ε hε γ β).fwd = dwbReluB N W b ε γ β := by
  rw [mnv4PostDWSlot_of_ne_zero N hk]; rfl

theorem mnv4PostDWSlot_fwd_of_eq_zero (N : Nat) {c h w kH kW : Nat} {k : Nat} (hk : k = 0)
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε : ℝ) (hε : 0 < ε) (γ β : Vec c) :
    (mnv4PostDWSlot (h := h) (w := w) N k W b ε hε γ β).fwd = fun y => y := by
  rw [mnv4PostDWSlot_of_eq_zero N hk]; rfl

theorem mnv4PreDWSlot_fwd_apply_of_ne_zero (N : Nat) {c h w kH kW : Nat} {k : Nat} (hk : k ≠ 0)
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε : ℝ) (hε : 0 < ε) (γ β : Vec c)
    (x : Vec (N * (c * h * w))) :
    (mnv4PreDWSlot (h := h) (w := w) N k W b ε hε γ β).fwd x = dwbB N W b ε γ β x := by
  rw [mnv4PreDWSlot_fwd_of_ne_zero N hk]

theorem mnv4PreDWSlot_fwd_apply_of_eq_zero (N : Nat) {c h w kH kW : Nat} {k : Nat} (hk : k = 0)
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε : ℝ) (hε : 0 < ε) (γ β : Vec c)
    (x : Vec (N * (c * h * w))) :
    (mnv4PreDWSlot (h := h) (w := w) N k W b ε hε γ β).fwd x = x := by
  rw [mnv4PreDWSlot_fwd_of_eq_zero N hk]

theorem mnv4PostDWSlot_fwd_apply_of_ne_zero (N : Nat) {c h w kH kW : Nat} {k : Nat} (hk : k ≠ 0)
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε : ℝ) (hε : 0 < ε) (γ β : Vec c)
    (x : Vec (N * (c * h * w))) :
    (mnv4PostDWSlot (h := h) (w := w) N k W b ε hε γ β).fwd x = dwbReluB N W b ε γ β x := by
  rw [mnv4PostDWSlot_fwd_of_ne_zero N hk]

theorem mnv4PostDWSlot_fwd_apply_of_eq_zero (N : Nat) {c h w kH kW : Nat} {k : Nat} (hk : k = 0)
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε : ℝ) (hε : 0 < ε) (γ β : Vec c)
    (x : Vec (N * (c * h * w))) :
    (mnv4PostDWSlot (h := h) (w := w) N k W b ε hε γ β).fwd x = x := by
  rw [mnv4PostDWSlot_fwd_of_eq_zero N hk]

/-- Back through the head's `[N, c] ↔ [N, c, 1, 1]` relabelling: the cotangent is read back
    along the same cast. -/
theorem hasGradAt_cast {n m : Nat} (e : n = m) (x : Vec n) {G : Vec m → Vec 1} {dy : Vec m}
    (hG : HasGradAt G (fun j => x (Fin.cast e.symm j)) dy) :
    HasGradAt (fun u => G (fun j => u (Fin.cast e.symm j))) x (fun i => dy (Fin.cast e i)) := by
  refine (HasGradAt.comp (f := reindexCLM (Fin.cast e.symm)) (x := x) hG
    (reindexCLM _).differentiableAt ((reindexHasVJP (Fin.cast e.symm)).toHasVJPAt x)).of_eq ?_
  funext i
  show ∑ k : Fin m, (if i = Fin.cast e.symm k then dy k else 0) = dy (Fin.cast e i)
  have hk : ∀ k : Fin m, i = Fin.cast e.symm k ↔ Fin.cast e i = k := fun k => by
    constructor
    · rintro rfl; exact Fin.ext rfl
    · rintro rfl; exact Fin.ext rfl
  simp only [hk, Finset.sum_ite_eq, Finset.mem_univ, ite_true]

-- ════════════════════════════════════════════════════════════════
-- § The stem — symmetric strided conv, batch BN, relu (no bias node)
-- ════════════════════════════════════════════════════════════════

section Stem
variable {N h w ic oc kH kW : Nat}

/-- **Stem, every parameter node a loss derivative** — the three nodes `mnv4StemTiedB` ties, `Φ`
    the loss as a function of the stem's `(W, γ, β)`. -/
def mnv4StemLossTiedB (xN cotN vN epsStr : String) (Ws : Kernel4 oc ic kH kW) (bs : Vec oc)
    (εs : ℝ) (γs βs : Vec oc) (x : Vec (N * (ic * (2 * h) * (2 * w))))
    (Φ : Kernel4 oc ic kH kW → Vec oc → Vec oc → Vec 1) (dy : Vec (N * (oc * h * w))) : Prop :=
  let sc := batchMap N (flatConvStride2 Ws bs) x
  (HasGradAt (fun θ => Φ (Kernel4.unflatten θ) γs βs) (Kernel4.flatten Ws)
        (den (SHlo.convStridedWeightGradB xN bs x Ws
          (.operand cotN (mnv4StemCotC N h w Ws bs εs γs βs x dy)))))
  ∧ (HasGradAt (fun θ => Φ Ws θ βs) γs
        (den (SHlo.bnGammaGradB vN epsStr εs (reassocB N oc h w sc)
          (.operand cotN (reassocB N oc h w (mnv4StemCotN N h w Ws bs εs γs βs x dy))))))
  ∧ (HasGradAt (fun θ => Φ Ws γs θ) βs
        (den (SHlo.bnBetaGradB (N := N) (oc := oc) (h := h) (w := w)
          (.operand cotN (reassocB N oc h w (mnv4StemCotN N h w Ws bs εs γs βs x dy))))))

theorem mnv4_stem_lossTiedB (xN cotN vN epsStr : String) (Ws : Kernel4 oc ic kH kW) (bs : Vec oc)
    (εs : ℝ) (hεs : 0 < εs) (γs βs : Vec oc) (x : Vec (N * (ic * (2 * h) * (2 * w))))
    (hs : Mnv4StemSmoothAtB N h w Ws bs εs γs βs x)
    {Gn : Vec (N * (oc * h * w)) → Vec 1} {dy : Vec (N * (oc * h * w))}
    (hGn : HasGradAt Gn (mnv4StemB N h w Ws bs εs γs βs x) dy)
    {Φ : Kernel4 oc ic kH kW → Vec oc → Vec oc → Vec 1}
    (hΦ : ∀ W γ β, Φ W γ β = Gn (mnv4StemB N h w W bs εs γ β x)) :
    mnv4StemLossTiedB xN cotN vN epsStr Ws bs εs γs βs x Φ dy := by
  rw [show Φ = fun W γ β => Gn (mnv4StemB N h w W bs εs γ β x) from
    funext fun W => funext fun γ => funext fun β => hΦ W γ β]
  have hN := hasGradAt_relu _ hs hGn
  have hC := hasGradAt_bnBatchLA εs hεs γs βs _ hN
  exact ⟨GradNodeB.convStridedW_hasGradAt xN cotN bs x Ws hC,
    GradNodeB.bnGamma_hasGradAt vN epsStr cotN εs γs βs _ hN,
    GradNodeB.bnBeta_hasGradAt cotN εs γs βs _ hN⟩

end Stem

-- ════════════════════════════════════════════════════════════════
-- § The fused stage — symmetric strided conv-BN-relu, then the 1×1 project
-- ════════════════════════════════════════════════════════════════

section Fused
variable {N h w ic mid oc kH kW : Nat}

/-- **Fused stage, every parameter node a loss derivative** — the six nodes `mnv4FusedTiedB`
    ties, `Φ` the loss as a function of `(Wc, γc, βc, Wp, γp, βp)`. -/
def mnv4FusedLossTiedB (Wc : Kernel4 mid ic kH kW) (bc : Vec mid) (εc : ℝ) (γc βc : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (εp : ℝ) (γp βp : Vec oc) (xN cotN vN epsStr : String)
    (xin : Vec (N * (ic * (2 * h) * (2 * w))))
    (Φ : Kernel4 mid ic kH kW → Vec mid → Vec mid → Kernel4 oc mid 1 1 → Vec oc → Vec oc → Vec 1)
    (dyF : Vec (N * (oc * h * w))) : Prop :=
  let sw := cbReluStridedB N (h := h) (w := w) Wc bc εc γc βc xin
  let fc := batchMap N (flatConvStride2 Wc bc) xin
  let pc := batchMap N (flatConv Wp bp) sw
  let cotN' := mnv4FusedCotN N h w Wc bc εc γc βc Wp bp εp γp βp xin dyF
  let cotC := mnv4FusedCotC N h w Wc bc εc γc βc Wp bp εp γp βp xin dyF
  let cotPc := mnv4FusedCotPc N h w Wc bc εc γc βc Wp bp εp γp βp xin dyF
  (HasGradAt (fun θ => Φ (Kernel4.unflatten θ) γc βc Wp γp βp) (Kernel4.flatten Wc)
        (den (SHlo.convStridedWeightGradB xN bc xin Wc (.operand cotN cotC))))
  ∧ (HasGradAt (fun θ => Φ Wc θ βc Wp γp βp) γc
        (den (SHlo.bnGammaGradB vN epsStr εc (reassocB N mid h w fc)
          (.operand cotN (reassocB N mid h w cotN')))))
  ∧ (HasGradAt (fun θ => Φ Wc γc θ Wp γp βp) βc
        (den (SHlo.bnBetaGradB (N := N) (oc := mid) (h := h) (w := w)
          (.operand cotN (reassocB N mid h w cotN')))))
  ∧ (HasGradAt (fun θ => Φ Wc γc βc (Kernel4.unflatten θ) γp βp) (Kernel4.flatten Wp)
        (den (SHlo.convWeightGradB xN bp sw Wp (.operand cotN cotPc))))
  ∧ (HasGradAt (fun θ => Φ Wc γc βc Wp θ βp) γp
        (den (SHlo.bnGammaGradB vN epsStr εp (reassocB N oc h w pc)
          (.operand cotN (reassocB N oc h w dyF)))))
  ∧ (HasGradAt (fun θ => Φ Wc γc βc Wp γp θ) βp
        (den (SHlo.bnBetaGradB (N := N) (oc := oc) (h := h) (w := w)
          (.operand cotN (reassocB N oc h w dyF)))))

theorem mnv4_fused_lossTiedB (Wc : Kernel4 mid ic kH kW) (bc : Vec mid) (εc : ℝ) (hεc : 0 < εc)
    (γc βc : Vec mid) (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (εp : ℝ) (hεp : 0 < εp)
    (γp βp : Vec oc) (xN cotN vN epsStr : String) (xin : Vec (N * (ic * (2 * h) * (2 * w))))
    (hs : ∀ k, bnBatchLA N mid h w εc γc βc (batchMap N (flatConvStride2 Wc bc) xin) k ≠ 0)
    {Gn : Vec (N * (oc * h * w)) → Vec 1} {dy : Vec (N * (oc * h * w))}
    (hGn : HasGradAt Gn (projB N (h := h) (w := w) Wp bp εp γp βp
      (cbReluStridedB N (h := h) (w := w) Wc bc εc γc βc xin)) dy)
    {Φ : Kernel4 mid ic kH kW → Vec mid → Vec mid → Kernel4 oc mid 1 1 → Vec oc → Vec oc → Vec 1}
    (hΦ : ∀ W1 γ1 β1 W2 γ2 β2, Φ W1 γ1 β1 W2 γ2 β2 = Gn (projB N (h := h) (w := w) W2 bp εp γ2 β2
      (cbReluStridedB N (h := h) (w := w) W1 bc εc γ1 β1 xin))) :
    mnv4FusedLossTiedB Wc bc εc γc βc Wp bp εp γp βp xN cotN vN epsStr xin Φ dy := by
  rw [show Φ = fun W1 γ1 β1 W2 γ2 β2 => Gn (projB N (h := h) (w := w) W2 bp εp γ2 β2
      (cbReluStridedB N (h := h) (w := w) W1 bc εc γ1 β1 xin)) from
    funext fun W1 => funext fun γ1 => funext fun β1 => funext fun W2 => funext fun γ2 =>
      funext fun β2 => hΦ W1 γ1 β1 W2 γ2 β2]
  have hPc := hasGradAt_bnBatchLA εp hεp γp βp _ hGn
  have hN := hasGradAt_relu _ hs (hasGradAt_conv Wp bp _ hPc)
  have hC := hasGradAt_bnBatchLA εc hεc γc βc _ hN
  exact ⟨GradNodeB.convStridedW_hasGradAt xN cotN bc xin Wc hC,
    GradNodeB.bnGamma_hasGradAt vN epsStr cotN εc γc βc _ hN,
    GradNodeB.bnBeta_hasGradAt cotN εc γc βc _ hN,
    GradNodeB.convW_hasGradAt xN cotN bp _ Wp hPc,
    GradNodeB.bnGamma_hasGradAt vN epsStr cotN εp γp βp _ hGn,
    GradNodeB.bnBeta_hasGradAt cotN εp γp βp _ hGn⟩

end Fused

-- ════════════════════════════════════════════════════════════════
-- § The head — conv-BN-relu, GAP, `conv_head`-BN-relu on the pooled features, dense
-- ════════════════════════════════════════════════════════════════

section Head
variable {N h w c mid oc nCls : Nat}

/-- The head as one plain function of its eight trained parameters (the ε's and conv biases held
    at the given values): `mnv4Head`'s forward. -/
noncomputable def mnv4HeadFwd (N h w : Nat) (W1 : Kernel4 mid c 1 1) (b1 : Vec mid) (ε1 : ℝ)
    (γ1 β1 : Vec mid) (W2 : Kernel4 oc mid 1 1) (b2 : Vec oc) (ε2 : ℝ) (γ2 β2 : Vec oc)
    (Wd : Mat oc nCls) (bd : Vec nCls) (xin : Vec (N * (c * h * w))) : Vec (N * nCls) :=
  batchMap N (dense Wd bd) (mnv4HeadFeat N h w W1 b1 ε1 γ1 β1 W2 b2 ε2 γ2 β2 xin)

theorem mnv4Head_fwd (W1 : Kernel4 mid c 1 1) (b1 : Vec mid) (ε1 : ℝ) (hε1 : 0 < ε1)
    (γ1 β1 : Vec mid) (W2 : Kernel4 oc mid 1 1) (b2 : Vec oc) (ε2 : ℝ) (hε2 : 0 < ε2)
    (γ2 β2 : Vec oc) (Wd : Mat oc nCls) (bd : Vec nCls) (xin : Vec (N * (c * h * w))) :
    (mnv4Head N (cbReluLayer (h := h) (w := w) N W1 b1 ε1 hε1 γ1 β1)
      (gapLayer N (c := mid) (h := h) (w := w))
      (cbReluLayer (h := 1) (w := 1) N W2 b2 ε2 hε2 γ2 β2) (denseLayer N Wd bd)).fwd xin
      = mnv4HeadFwd N h w W1 b1 ε1 γ1 β1 W2 b2 ε2 γ2 β2 Wd bd xin := rfl

/-- **Head, every parameter node a loss derivative** — the eight nodes `mnv4HeadTiedB` ties, `Φ`
    the loss as a function of `(W1, γ1, β1, W2, γ2, β2, Wd, bd)`. -/
def mnv4HeadLossTiedB (W1 : Kernel4 mid c 1 1) (b1 : Vec mid) (ε1 : ℝ) (γ1 β1 : Vec mid)
    (W2 : Kernel4 oc mid 1 1) (b2 : Vec oc) (ε2 : ℝ) (γ2 β2 : Vec oc)
    (Wd : Mat oc nCls) (bd : Vec nCls) (xN cotN vN epsStr : String) (xin : Vec (N * (c * h * w)))
    (Φ : Kernel4 mid c 1 1 → Vec mid → Vec mid → Kernel4 oc mid 1 1 → Vec oc → Vec oc →
      Mat oc nCls → Vec nCls → Vec 1)
    (g : Vec (N * nCls)) : Prop :=
  let pool := mnv4HeadPool N h w W1 b1 ε1 γ1 β1 xin
  let feat := mnv4HeadFeat N h w W1 b1 ε1 γ1 β1 W2 b2 ε2 γ2 β2 xin
  let c1 := batchMap N (flatConv W1 b1) xin
  let c2 := batchMap N (flatConv W2 b2) pool
  let cotH1n := mnv4HeadCotH1n N h w W1 b1 ε1 γ1 β1 W2 b2 ε2 γ2 β2 Wd bd xin g
  let cotH1c := mnv4HeadCotH1c N h w W1 b1 ε1 γ1 β1 W2 b2 ε2 γ2 β2 Wd bd xin g
  let cotHn := mnv4HeadCotHn N h w W1 b1 ε1 γ1 β1 W2 b2 ε2 γ2 β2 Wd bd xin g
  let cotHc := mnv4HeadCotHc N h w W1 b1 ε1 γ1 β1 W2 b2 ε2 γ2 β2 Wd bd xin g
  (HasGradAt (fun θ => Φ (Kernel4.unflatten θ) γ1 β1 W2 γ2 β2 Wd bd) (Kernel4.flatten W1)
        (den (SHlo.convWeightGradB xN b1 xin W1 (.operand cotN cotH1c))))
  ∧ (HasGradAt (fun θ => Φ W1 θ β1 W2 γ2 β2 Wd bd) γ1
        (den (SHlo.bnGammaGradB vN epsStr ε1 (reassocB N mid h w c1)
          (.operand cotN (reassocB N mid h w cotH1n)))))
  ∧ (HasGradAt (fun θ => Φ W1 γ1 θ W2 γ2 β2 Wd bd) β1
        (den (SHlo.bnBetaGradB (N := N) (oc := mid) (h := h) (w := w)
          (.operand cotN (reassocB N mid h w cotH1n)))))
  ∧ (HasGradAt (fun θ => Φ W1 γ1 β1 (Kernel4.unflatten θ) γ2 β2 Wd bd) (Kernel4.flatten W2)
        (den (SHlo.convWeightGradB xN b2 pool W2 (.operand cotN cotHc))))
  ∧ (HasGradAt (fun θ => Φ W1 γ1 β1 W2 θ β2 Wd bd) γ2
        (den (SHlo.bnGammaGradB vN epsStr ε2 (reassocB N oc 1 1 c2)
          (.operand cotN (reassocB N oc 1 1 cotHn)))))
  ∧ (HasGradAt (fun θ => Φ W1 γ1 β1 W2 γ2 θ Wd bd) β2
        (den (SHlo.bnBetaGradB (N := N) (oc := oc) (h := 1) (w := 1)
          (.operand cotN (reassocB N oc 1 1 cotHn)))))
  ∧ (HasGradAt (fun θ => Φ W1 γ1 β1 W2 γ2 β2 (Mat.unflatten θ) bd) (Mat.flatten Wd)
        (den (SHlo.denseWeightGradB (c := nCls) xN feat (.operand cotN g))))
  ∧ (HasGradAt (fun θ => Φ W1 γ1 β1 W2 γ2 β2 Wd θ) bd
        (den (SHlo.denseBiasGradB (N := N) (.operand cotN g))))

theorem mnv4_head_lossTiedB (W1 : Kernel4 mid c 1 1) (b1 : Vec mid) (ε1 : ℝ) (hε1 : 0 < ε1)
    (γ1 β1 : Vec mid) (W2 : Kernel4 oc mid 1 1) (b2 : Vec oc) (ε2 : ℝ) (hε2 : 0 < ε2)
    (γ2 β2 : Vec oc) (Wd : Mat oc nCls) (bd : Vec nCls) (xN cotN vN epsStr : String)
    (xin : Vec (N * (c * h * w)))
    (hs1 : ∀ k, bnBatchLA N mid h w ε1 γ1 β1 (batchMap N (flatConv W1 b1) xin) k ≠ 0)
    (hs2 : ∀ k, bnBatchLA N oc 1 1 ε2 γ2 β2
      (batchMap N (flatConv W2 b2) (mnv4HeadPool N h w W1 b1 ε1 γ1 β1 xin)) k ≠ 0)
    {L : Vec (N * nCls) → Vec 1} {g : Vec (N * nCls)}
    (hL : HasGradAt L (mnv4HeadFwd N h w W1 b1 ε1 γ1 β1 W2 b2 ε2 γ2 β2 Wd bd xin) g)
    {Φ : Kernel4 mid c 1 1 → Vec mid → Vec mid → Kernel4 oc mid 1 1 → Vec oc → Vec oc →
      Mat oc nCls → Vec nCls → Vec 1}
    (hΦ : ∀ W1' γ1' β1' W2' γ2' β2' Wd' bd', Φ W1' γ1' β1' W2' γ2' β2' Wd' bd'
      = L (mnv4HeadFwd N h w W1' b1 ε1 γ1' β1' W2' b2 ε2 γ2' β2' Wd' bd' xin)) :
    mnv4HeadLossTiedB W1 b1 ε1 γ1 β1 W2 b2 ε2 γ2 β2 Wd bd xN cotN vN epsStr xin Φ g := by
  rw [show Φ = fun W1' γ1' β1' W2' γ2' β2' Wd' bd' =>
      L (mnv4HeadFwd N h w W1' b1 ε1 γ1' β1' W2' b2 ε2 γ2' β2' Wd' bd' xin) from
    funext fun _ => funext fun _ => funext fun _ => funext fun _ => funext fun _ =>
      funext fun _ => funext fun _ => funext fun _ => hΦ _ _ _ _ _ _ _ _]
  have hA : HasGradAt (fun a => L (batchMap N (dense Wd bd) a))
      (mnv4HeadFeat N h w W1 b1 ε1 γ1 β1 W2 b2 ε2 γ2 β2 xin) (rowDenseBackFlat N oc nCls Wd g) :=
    HasGradAt.comp (f := batchMap N (dense Wd bd))
      (x := mnv4HeadFeat N h w W1 b1 ε1 γ1 β1 W2 b2 ε2 γ2 β2 xin) hL
      ((batchMap_differentiable _ (dense_differentiable Wd bd)) _)
      ((batchMapHasVJP _ (denseHasVJP Wd bd) (dense_differentiable Wd bd)).toHasVJPAt _)
  have hR2 := hasGradAt_cast (mnv4_pool11 N oc).symm _ hA
  have hHn := hasGradAt_relu _ hs2 hR2
  have hHc := hasGradAt_bnBatchLA ε2 hε2 γ2 β2 _ hHn
  have hP := hasGradAt_conv W2 b2 _ hHc
  have hAvg := hasGradAt_cast (mnv4_pool11 N mid) _ hP
  have hR1 :=
    HasGradAt.comp (f := batchMap N (globalAvgPoolFlat mid h w))
      (x := cbReluB N (h := h) (w := w) W1 b1 ε1 γ1 β1 xin) hAvg
      ((batchMap_differentiable _ (globalAvgPoolFlat_differentiable mid h w)) _)
      ((batchMapHasVJP _ (globalAvgPoolFlatHasVJP mid h w)
        (globalAvgPoolFlat_differentiable mid h w)).toHasVJPAt _)
  have hH1n := hasGradAt_relu _ hs1 hR1
  have hH1c := hasGradAt_bnBatchLA ε1 hε1 γ1 β1 _ hH1n
  exact ⟨GradNodeB.convW_hasGradAt xN cotN b1 xin W1 hH1c,
    GradNodeB.bnGamma_hasGradAt vN epsStr cotN ε1 γ1 β1 _ hH1n,
    GradNodeB.bnBeta_hasGradAt cotN ε1 γ1 β1 _ hH1n,
    GradNodeB.convW_hasGradAt (h := 1) (w := 1) xN cotN b2 _ W2 hHc,
    GradNodeB.bnGamma_hasGradAt vN epsStr cotN ε2 γ2 β2 _ hHn,
    GradNodeB.bnBeta_hasGradAt cotN ε2 γ2 β2 _ hHn,
    GradNodeB.denseW_hasGradAt xN cotN _ Wd bd hL,
    GradNodeB.denseB_hasGradAt cotN Wd (fun _ => 0) _ bd hL⟩

end Head

-- ════════════════════════════════════════════════════════════════
-- § The ExtraDW body — both depthwise slots present (13 of the 21 rows)
--   `Gb` is the loss read at the BODY output (the skip is added by `mnv4_resid_*`).
-- ════════════════════════════════════════════════════════════════

section ExtraDW
variable {N : Nat} {s : UibSpec}

/-- **ExtraDW body, every parameter node a loss derivative** — the twelve nodes
    `mnv4ExtraDWTiedB` ties, `Φ` the loss at the body output as a function of the row's weight
    record. A depthwise slot's parameter is varied through `withPre` / `withPost`. -/
def mnv4ExtraDWLossTiedB (N : Nat) (s : UibSpec) (xN cotN vN epsStr : String) (p : UibParams s)
    (xin : Vec (N * (s.ic * s.h * s.h))) (Φ : UibParams s → Vec 1)
    (dyOut : Vec (N * (s.oc * s.h * s.h))) : Prop :=
  let qr := (mnv4PreDWSlot (h := s.h) (w := s.h) N s.preDWk p.Wq p.bq p.eq_ p.hq p.gq p.bq2).fwd xin
  let er := (cbReluLayer (h := s.h) (w := s.h) N p.We p.be p.ee p.he p.ge p.be2).fwd qr
  let dr := (mnv4PostDWSlot (h := s.h) (w := s.h) N s.postDWk p.Wd p.bd p.ed p.hd p.gd p.bd2).fwd er
  let qc := batchMap N (depthwiseFlat p.Wq p.bq) xin
  let ec := batchMap N (flatConv p.We p.be) qr
  let dc := batchMap N (depthwiseFlat p.Wd p.bd) er
  let pc := batchMap N (flatConv p.Wz p.bz) dr
  (HasGradAt (fun θ => Φ (p.withPre { p.pre.params with W := Tensor3.unflatten θ })) (Tensor3.flatten p.Wq)
        (den (SHlo.depthwiseWeightGradB xN p.bq xin p.Wq
          (.operand cotN (mnv4CotQc N s p xin dyOut)))))
  ∧ (HasGradAt (fun θ => Φ (p.withPre { p.pre.params with γ := θ })) p.gq
        (den (SHlo.bnGammaGradB vN epsStr p.eq_ (reassocB N s.ic s.h s.h qc)
          (.operand cotN (reassocB N s.ic s.h s.h (mnv4CotQn N s p xin dyOut))))))
  ∧ (HasGradAt (fun θ => Φ (p.withPre { p.pre.params with β := θ })) p.bq2
        (den (SHlo.bnBetaGradB (N := N) (oc := s.ic) (h := s.h) (w := s.h)
          (.operand cotN (reassocB N s.ic s.h s.h (mnv4CotQn N s p xin dyOut))))))
  ∧ (HasGradAt (fun θ => Φ { p with We := Kernel4.unflatten θ }) (Kernel4.flatten p.We)
        (den (SHlo.convWeightGradB xN p.be qr p.We (.operand cotN (mnv4CotEc N s p xin dyOut)))))
  ∧ (HasGradAt (fun θ => Φ { p with ge := θ }) p.ge
        (den (SHlo.bnGammaGradB vN epsStr p.ee (reassocB N (s.ic * s.expand) s.h s.h ec)
          (.operand cotN (reassocB N (s.ic * s.expand) s.h s.h (mnv4CotEn N s p xin dyOut))))))
  ∧ (HasGradAt (fun θ => Φ { p with be2 := θ }) p.be2
        (den (SHlo.bnBetaGradB (N := N) (oc := s.ic * s.expand) (h := s.h) (w := s.h)
          (.operand cotN (reassocB N (s.ic * s.expand) s.h s.h (mnv4CotEn N s p xin dyOut))))))
  ∧ (HasGradAt (fun θ => Φ (p.withPost { p.post.params with W := Tensor3.unflatten θ })) (Tensor3.flatten p.Wd)
        (den (SHlo.depthwiseWeightGradB xN p.bd er p.Wd
          (.operand cotN (mnv4CotDc N s p xin dyOut)))))
  ∧ (HasGradAt (fun θ => Φ (p.withPost { p.post.params with γ := θ })) p.gd
        (den (SHlo.bnGammaGradB vN epsStr p.ed (reassocB N (s.ic * s.expand) s.h s.h dc)
          (.operand cotN (reassocB N (s.ic * s.expand) s.h s.h (mnv4CotDn N s p xin dyOut))))))
  ∧ (HasGradAt (fun θ => Φ (p.withPost { p.post.params with β := θ })) p.bd2
        (den (SHlo.bnBetaGradB (N := N) (oc := s.ic * s.expand) (h := s.h) (w := s.h)
          (.operand cotN (reassocB N (s.ic * s.expand) s.h s.h (mnv4CotDn N s p xin dyOut))))))
  ∧ (HasGradAt (fun θ => Φ { p with Wz := Kernel4.unflatten θ }) (Kernel4.flatten p.Wz)
        (den (SHlo.convWeightGradB xN p.bz dr p.Wz (.operand cotN (mnv4CotPc N s p xin dyOut)))))
  ∧ (HasGradAt (fun θ => Φ { p with gz := θ }) p.gz
        (den (SHlo.bnGammaGradB vN epsStr p.ez (reassocB N s.oc s.h s.h pc)
          (.operand cotN (reassocB N s.oc s.h s.h dyOut)))))
  ∧ (HasGradAt (fun θ => Φ { p with bz2 := θ }) p.bz2
        (den (SHlo.bnBetaGradB (N := N) (oc := s.oc) (h := s.h) (w := s.h)
          (.operand cotN (reassocB N s.oc s.h s.h dyOut)))))

theorem mnv4_extradw_lossTiedB (xN cotN vN epsStr : String) (p : UibParams s)
    (hq0 : s.preDWk ≠ 0) (hd0 : s.postDWk ≠ 0) (v : Vec (N * (s.ic * s.h * s.h)))
    (hok : (mnv4BodyOfRow N s p).ok v) {Gb : Vec (N * (s.oc * s.h * s.h)) → Vec 1}
    {dy : Vec (N * (s.oc * s.h * s.h))} (hGb : HasGradAt Gb ((mnv4BodyOfRow N s p).fwd v) dy)
    {Φ : UibParams s → Vec 1} (hΦ : ∀ p', Φ p' = Gb ((mnv4BodyOfRow N s p').fwd v)) :
    mnv4ExtraDWLossTiedB N s xN cotN vN epsStr p v Φ dy := by
  have hfwd : ∀ p' : UibParams s, (mnv4BodyOfRow N s p').fwd v
      = projB N (h := s.h) (w := s.h) p'.Wz p'.bz p'.ez p'.gz p'.bz2
          (dwbReluB N (h := s.h) (w := s.h) p'.Wd p'.bd p'.ed p'.gd p'.bd2
            (cbReluB N (h := s.h) (w := s.h) p'.We p'.be p'.ee p'.ge p'.be2
              (dwbB N (h := s.h) (w := s.h) p'.Wq p'.bq p'.eq_ p'.gq p'.bq2 v))) := fun p' => by
    rw [mnv4BodyOfRow, mnv4UibBody, CertLayer.comp_fwd_apply, CertLayer.comp_fwd_apply,
      CertLayer.comp_fwd_apply, mnv4PreDWSlot_fwd_apply_of_ne_zero N hq0,
      mnv4PostDWSlot_fwd_apply_of_ne_zero N hd0]
    rfl
  have hsE : ∀ k, bnBatchLA N (s.ic * s.expand) s.h s.h p.ee p.ge p.be2
      (batchMap N (flatConv p.We p.be) (dwbB N (h := s.h) (w := s.h) p.Wq p.bq p.eq_ p.gq p.bq2 v))
        k ≠ 0 := by
    have h := hok.2.1
    rw [mnv4PreDWSlot_fwd_apply_of_ne_zero N hq0] at h
    exact h
  have hsD : ∀ k, bnBatchLA N (s.ic * s.expand) s.h s.h p.ed p.gd p.bd2
      (batchMap N (depthwiseFlat p.Wd p.bd) (cbReluB N (h := s.h) (w := s.h) p.We p.be p.ee p.ge
        p.be2 (dwbB N (h := s.h) (w := s.h) p.Wq p.bq p.eq_ p.gq p.bq2 v))) k ≠ 0 := by
    have h := hok.2.2.1
    rw [mnv4PostDWSlot_of_ne_zero N hd0, mnv4PreDWSlot_fwd_apply_of_ne_zero N hq0] at h
    exact h
  rw [show Φ = fun p' => Gb ((mnv4BodyOfRow N s p').fwd v) from funext hΦ]
  rw [hfwd p] at hGb
  have hPc := hasGradAt_bnBatchLA p.ez p.hz p.gz p.bz2 _ hGb
  have hDn := hasGradAt_relu _ hsD (hasGradAt_conv p.Wz p.bz _ hPc)
  have hDc := hasGradAt_bnBatchLA p.ed p.hd p.gd p.bd2 _ hDn
  have hEn := hasGradAt_relu _ hsE (hasGradAt_depthwise p.Wd p.bd _ hDc)
  have hEc := hasGradAt_bnBatchLA p.ee p.he p.ge p.be2 _ hEn
  have hQn := hasGradAt_conv p.We p.be _ hEc
  have hQc := hasGradAt_bnBatchLA p.eq_ p.hq p.gq p.bq2 _ hQn
  simp only [mnv4ExtraDWLossTiedB, mnv4CotQc, mnv4CotQn, mnv4CotEc, mnv4CotEn, mnv4CotDc,
    mnv4CotDn, mnv4CotPc, mnv4PreDWSlot_fwd_apply_of_ne_zero N hq0,
    mnv4PostDWSlot_fwd_apply_of_ne_zero N hd0, hd0, ↓reduceIte, hfwd, UibParams.Wq, UibParams.bq,
    UibParams.eq_, UibParams.gq, UibParams.bq2, UibParams.Wd, UibParams.bd, UibParams.ed,
    UibParams.gd, UibParams.bd2, UibParams.withPre_pre_params _ _ hq0,
    UibParams.withPost_post_params _ _ hd0]
  exact ⟨GradNodeB.depthwiseW_hasGradAt xN cotN _ v _ hQc,
    GradNodeB.bnGamma_hasGradAt vN epsStr cotN _ _ _ _ hQn,
    GradNodeB.bnBeta_hasGradAt cotN _ _ _ _ hQn,
    GradNodeB.convW_hasGradAt xN cotN _ _ _ hEc,
    GradNodeB.bnGamma_hasGradAt vN epsStr cotN _ _ _ _ hEn,
    GradNodeB.bnBeta_hasGradAt cotN _ _ _ _ hEn,
    GradNodeB.depthwiseW_hasGradAt xN cotN _ _ _ hDc,
    GradNodeB.bnGamma_hasGradAt vN epsStr cotN _ _ _ _ hDn,
    GradNodeB.bnBeta_hasGradAt cotN _ _ _ _ hDn,
    GradNodeB.convW_hasGradAt xN cotN _ _ _ hPc,
    GradNodeB.bnGamma_hasGradAt vN epsStr cotN _ _ _ _ hGb,
    GradNodeB.bnBeta_hasGradAt cotN _ _ _ _ hGb⟩

end ExtraDW

-- ════════════════════════════════════════════════════════════════
-- § The ConvNeXt-like body (pre-DW only) and the FFN body (no depthwise)
-- ════════════════════════════════════════════════════════════════

section ConvNeXtFfn
variable {N : Nat} {s : UibSpec}

/-- **ConvNeXt-like body, every parameter node a loss derivative** — the nine nodes
    `mnv4ConvNeXtTiedB` ties. -/
def mnv4ConvNeXtLossTiedB (N : Nat) (s : UibSpec) (xN cotN vN epsStr : String) (p : UibParams s)
    (xin : Vec (N * (s.ic * s.h * s.h))) (Φ : UibParams s → Vec 1)
    (dyOut : Vec (N * (s.oc * s.h * s.h))) : Prop :=
  let qr := (mnv4PreDWSlot (h := s.h) (w := s.h) N s.preDWk p.Wq p.bq p.eq_ p.hq p.gq p.bq2).fwd xin
  let er := (cbReluLayer (h := s.h) (w := s.h) N p.We p.be p.ee p.he p.ge p.be2).fwd qr
  let dr := (mnv4PostDWSlot (h := s.h) (w := s.h) N s.postDWk p.Wd p.bd p.ed p.hd p.gd p.bd2).fwd er
  let qc := batchMap N (depthwiseFlat p.Wq p.bq) xin
  let ec := batchMap N (flatConv p.We p.be) qr
  let pc := batchMap N (flatConv p.Wz p.bz) dr
  (HasGradAt (fun θ => Φ (p.withPre { p.pre.params with W := Tensor3.unflatten θ })) (Tensor3.flatten p.Wq)
        (den (SHlo.depthwiseWeightGradB xN p.bq xin p.Wq
          (.operand cotN (mnv4CotQc N s p xin dyOut)))))
  ∧ (HasGradAt (fun θ => Φ (p.withPre { p.pre.params with γ := θ })) p.gq
        (den (SHlo.bnGammaGradB vN epsStr p.eq_ (reassocB N s.ic s.h s.h qc)
          (.operand cotN (reassocB N s.ic s.h s.h (mnv4CotQn N s p xin dyOut))))))
  ∧ (HasGradAt (fun θ => Φ (p.withPre { p.pre.params with β := θ })) p.bq2
        (den (SHlo.bnBetaGradB (N := N) (oc := s.ic) (h := s.h) (w := s.h)
          (.operand cotN (reassocB N s.ic s.h s.h (mnv4CotQn N s p xin dyOut))))))
  ∧ (HasGradAt (fun θ => Φ { p with We := Kernel4.unflatten θ }) (Kernel4.flatten p.We)
        (den (SHlo.convWeightGradB xN p.be qr p.We (.operand cotN (mnv4CotEc N s p xin dyOut)))))
  ∧ (HasGradAt (fun θ => Φ { p with ge := θ }) p.ge
        (den (SHlo.bnGammaGradB vN epsStr p.ee (reassocB N (s.ic * s.expand) s.h s.h ec)
          (.operand cotN (reassocB N (s.ic * s.expand) s.h s.h (mnv4CotEn N s p xin dyOut))))))
  ∧ (HasGradAt (fun θ => Φ { p with be2 := θ }) p.be2
        (den (SHlo.bnBetaGradB (N := N) (oc := s.ic * s.expand) (h := s.h) (w := s.h)
          (.operand cotN (reassocB N (s.ic * s.expand) s.h s.h (mnv4CotEn N s p xin dyOut))))))
  ∧ (HasGradAt (fun θ => Φ { p with Wz := Kernel4.unflatten θ }) (Kernel4.flatten p.Wz)
        (den (SHlo.convWeightGradB xN p.bz dr p.Wz (.operand cotN (mnv4CotPc N s p xin dyOut)))))
  ∧ (HasGradAt (fun θ => Φ { p with gz := θ }) p.gz
        (den (SHlo.bnGammaGradB vN epsStr p.ez (reassocB N s.oc s.h s.h pc)
          (.operand cotN (reassocB N s.oc s.h s.h dyOut)))))
  ∧ (HasGradAt (fun θ => Φ { p with bz2 := θ }) p.bz2
        (den (SHlo.bnBetaGradB (N := N) (oc := s.oc) (h := s.h) (w := s.h)
          (.operand cotN (reassocB N s.oc s.h s.h dyOut)))))

theorem mnv4_convnext_lossTiedB (xN cotN vN epsStr : String) (p : UibParams s)
    (hq0 : s.preDWk ≠ 0) (hd0 : s.postDWk = 0) (v : Vec (N * (s.ic * s.h * s.h)))
    (hok : (mnv4BodyOfRow N s p).ok v) {Gb : Vec (N * (s.oc * s.h * s.h)) → Vec 1}
    {dy : Vec (N * (s.oc * s.h * s.h))} (hGb : HasGradAt Gb ((mnv4BodyOfRow N s p).fwd v) dy)
    {Φ : UibParams s → Vec 1} (hΦ : ∀ p', Φ p' = Gb ((mnv4BodyOfRow N s p').fwd v)) :
    mnv4ConvNeXtLossTiedB N s xN cotN vN epsStr p v Φ dy := by
  have hfwd : ∀ p' : UibParams s, (mnv4BodyOfRow N s p').fwd v
      = projB N (h := s.h) (w := s.h) p'.Wz p'.bz p'.ez p'.gz p'.bz2
          (cbReluB N (h := s.h) (w := s.h) p'.We p'.be p'.ee p'.ge p'.be2
            (dwbB N (h := s.h) (w := s.h) p'.Wq p'.bq p'.eq_ p'.gq p'.bq2 v)) := fun p' => by
    rw [mnv4BodyOfRow, mnv4UibBody, CertLayer.comp_fwd_apply, CertLayer.comp_fwd_apply,
      CertLayer.comp_fwd_apply, mnv4PreDWSlot_fwd_apply_of_ne_zero N hq0,
      mnv4PostDWSlot_fwd_apply_of_eq_zero N hd0]
    rfl
  have hsE : ∀ k, bnBatchLA N (s.ic * s.expand) s.h s.h p.ee p.ge p.be2
      (batchMap N (flatConv p.We p.be) (dwbB N (h := s.h) (w := s.h) p.Wq p.bq p.eq_ p.gq p.bq2 v))
        k ≠ 0 := by
    have h := hok.2.1
    rw [mnv4PreDWSlot_fwd_apply_of_ne_zero N hq0] at h
    exact h
  rw [show Φ = fun p' => Gb ((mnv4BodyOfRow N s p').fwd v) from funext hΦ]
  rw [hfwd p] at hGb
  have hPc := hasGradAt_bnBatchLA p.ez p.hz p.gz p.bz2 _ hGb
  have hEn := hasGradAt_relu _ hsE (hasGradAt_conv p.Wz p.bz _ hPc)
  have hEc := hasGradAt_bnBatchLA p.ee p.he p.ge p.be2 _ hEn
  have hQn := hasGradAt_conv p.We p.be _ hEc
  have hQc := hasGradAt_bnBatchLA p.eq_ p.hq p.gq p.bq2 _ hQn
  simp only [mnv4ConvNeXtLossTiedB, mnv4CotQc, mnv4CotQn, mnv4CotEc, mnv4CotEn, mnv4CotPc,
    mnv4PreDWSlot_fwd_apply_of_ne_zero N hq0, hd0,
    ↓reduceIte, hfwd, UibParams.Wq, UibParams.bq, UibParams.eq_, UibParams.gq, UibParams.bq2,
    UibParams.withPre_pre_params _ _ hq0]
  exact ⟨GradNodeB.depthwiseW_hasGradAt xN cotN _ v _ hQc,
    GradNodeB.bnGamma_hasGradAt vN epsStr cotN _ _ _ _ hQn,
    GradNodeB.bnBeta_hasGradAt cotN _ _ _ _ hQn,
    GradNodeB.convW_hasGradAt xN cotN _ _ _ hEc,
    GradNodeB.bnGamma_hasGradAt vN epsStr cotN _ _ _ _ hEn,
    GradNodeB.bnBeta_hasGradAt cotN _ _ _ _ hEn,
    GradNodeB.convW_hasGradAt xN cotN _ _ _ hPc,
    GradNodeB.bnGamma_hasGradAt vN epsStr cotN _ _ _ _ hGb,
    GradNodeB.bnBeta_hasGradAt cotN _ _ _ _ hGb⟩

/-- **FFN body, every parameter node a loss derivative** — the six nodes `mnv4FfnTiedB` ties. -/
def mnv4FfnLossTiedB (N : Nat) (s : UibSpec) (xN cotN vN epsStr : String) (p : UibParams s)
    (xin : Vec (N * (s.ic * s.h * s.h))) (Φ : UibParams s → Vec 1)
    (dyOut : Vec (N * (s.oc * s.h * s.h))) : Prop :=
  let qr := (mnv4PreDWSlot (h := s.h) (w := s.h) N s.preDWk p.Wq p.bq p.eq_ p.hq p.gq p.bq2).fwd xin
  let er := (cbReluLayer (h := s.h) (w := s.h) N p.We p.be p.ee p.he p.ge p.be2).fwd qr
  let dr := (mnv4PostDWSlot (h := s.h) (w := s.h) N s.postDWk p.Wd p.bd p.ed p.hd p.gd p.bd2).fwd er
  let ec := batchMap N (flatConv p.We p.be) qr
  let pc := batchMap N (flatConv p.Wz p.bz) dr
  (HasGradAt (fun θ => Φ { p with We := Kernel4.unflatten θ }) (Kernel4.flatten p.We)
        (den (SHlo.convWeightGradB xN p.be qr p.We (.operand cotN (mnv4CotEc N s p xin dyOut)))))
  ∧ (HasGradAt (fun θ => Φ { p with ge := θ }) p.ge
        (den (SHlo.bnGammaGradB vN epsStr p.ee (reassocB N (s.ic * s.expand) s.h s.h ec)
          (.operand cotN (reassocB N (s.ic * s.expand) s.h s.h (mnv4CotEn N s p xin dyOut))))))
  ∧ (HasGradAt (fun θ => Φ { p with be2 := θ }) p.be2
        (den (SHlo.bnBetaGradB (N := N) (oc := s.ic * s.expand) (h := s.h) (w := s.h)
          (.operand cotN (reassocB N (s.ic * s.expand) s.h s.h (mnv4CotEn N s p xin dyOut))))))
  ∧ (HasGradAt (fun θ => Φ { p with Wz := Kernel4.unflatten θ }) (Kernel4.flatten p.Wz)
        (den (SHlo.convWeightGradB xN p.bz dr p.Wz (.operand cotN (mnv4CotPc N s p xin dyOut)))))
  ∧ (HasGradAt (fun θ => Φ { p with gz := θ }) p.gz
        (den (SHlo.bnGammaGradB vN epsStr p.ez (reassocB N s.oc s.h s.h pc)
          (.operand cotN (reassocB N s.oc s.h s.h dyOut)))))
  ∧ (HasGradAt (fun θ => Φ { p with bz2 := θ }) p.bz2
        (den (SHlo.bnBetaGradB (N := N) (oc := s.oc) (h := s.h) (w := s.h)
          (.operand cotN (reassocB N s.oc s.h s.h dyOut)))))

theorem mnv4_ffn_lossTiedB (xN cotN vN epsStr : String) (p : UibParams s)
    (hq0 : s.preDWk = 0) (hd0 : s.postDWk = 0) (v : Vec (N * (s.ic * s.h * s.h)))
    (hok : (mnv4BodyOfRow N s p).ok v) {Gb : Vec (N * (s.oc * s.h * s.h)) → Vec 1}
    {dy : Vec (N * (s.oc * s.h * s.h))} (hGb : HasGradAt Gb ((mnv4BodyOfRow N s p).fwd v) dy)
    {Φ : UibParams s → Vec 1} (hΦ : ∀ p', Φ p' = Gb ((mnv4BodyOfRow N s p').fwd v)) :
    mnv4FfnLossTiedB N s xN cotN vN epsStr p v Φ dy := by
  have hfwd : ∀ p' : UibParams s, (mnv4BodyOfRow N s p').fwd v
      = projB N (h := s.h) (w := s.h) p'.Wz p'.bz p'.ez p'.gz p'.bz2
          (cbReluB N (h := s.h) (w := s.h) p'.We p'.be p'.ee p'.ge p'.be2 v) := fun p' => by
    rw [mnv4BodyOfRow, mnv4UibBody, CertLayer.comp_fwd_apply, CertLayer.comp_fwd_apply,
      CertLayer.comp_fwd_apply, mnv4PreDWSlot_fwd_apply_of_eq_zero N hq0,
      mnv4PostDWSlot_fwd_apply_of_eq_zero N hd0]
    rfl
  have hsE : ∀ k, bnBatchLA N (s.ic * s.expand) s.h s.h p.ee p.ge p.be2
      (batchMap N (flatConv p.We p.be) v) k ≠ 0 := by
    have h := hok.2.1
    rw [mnv4PreDWSlot_fwd_apply_of_eq_zero N hq0] at h
    exact h
  rw [show Φ = fun p' => Gb ((mnv4BodyOfRow N s p').fwd v) from funext hΦ]
  rw [hfwd p] at hGb
  have hPc := hasGradAt_bnBatchLA p.ez p.hz p.gz p.bz2 _ hGb
  have hEn := hasGradAt_relu _ hsE (hasGradAt_conv p.Wz p.bz _ hPc)
  have hEc := hasGradAt_bnBatchLA p.ee p.he p.ge p.be2 _ hEn
  simp only [mnv4FfnLossTiedB, mnv4CotEc, mnv4CotEn, mnv4CotPc,
    mnv4PreDWSlot_fwd_apply_of_eq_zero N hq0, hd0, ↓reduceIte, hfwd]
  exact ⟨GradNodeB.convW_hasGradAt xN cotN _ _ _ hEc,
    GradNodeB.bnGamma_hasGradAt vN epsStr cotN _ _ _ _ hEn,
    GradNodeB.bnBeta_hasGradAt cotN _ _ _ _ hEn,
    GradNodeB.convW_hasGradAt xN cotN _ _ _ hPc,
    GradNodeB.bnGamma_hasGradAt vN epsStr cotN _ _ _ _ hGb,
    GradNodeB.bnBeta_hasGradAt cotN _ _ _ _ hGb⟩

end ConvNeXtFfn

-- ════════════════════════════════════════════════════════════════
-- § The strided body (rows 1, 3, 11) — pre-DW and expand at `2h`, the post-DW carries the stride
-- ════════════════════════════════════════════════════════════════

section Strided
variable {N : Nat} {s : UibSpec}

/-- **Strided body, every parameter node a loss derivative** — the twelve nodes
    `mnv4StridedTiedB` ties; the post-DW is the symmetric strided depthwise. -/
def mnv4StridedLossTiedB (N : Nat) (s : UibSpec) (xN cotN vN epsStr : String) (p : UibParams s)
    (xin : Vec (N * (s.ic * (2 * s.h) * (2 * s.h)))) (Φ : UibParams s → Vec 1)
    (dyOut : Vec (N * (s.oc * s.h * s.h))) : Prop :=
  let qr := (mnv4PreDWSlot (h := 2 * s.h) (w := 2 * s.h) N s.preDWk p.Wq p.bq p.eq_ p.hq p.gq p.bq2).fwd xin
  let er := (cbReluLayer (h := 2 * s.h) (w := 2 * s.h) N p.We p.be p.ee p.he p.ge p.be2).fwd qr
  let dr := (mnv4DWReluStridedLayer (h := s.h) (w := s.h) N p.Wd p.bd p.ed p.hd p.gd p.bd2).fwd er
  let qc := batchMap N (depthwiseFlat p.Wq p.bq) xin
  let ec := batchMap N (flatConv p.We p.be) qr
  let dc := batchMap N (depthwiseStride2Flat p.Wd p.bd) er
  let pc := batchMap N (flatConv p.Wz p.bz) dr
  (HasGradAt (fun θ => Φ (p.withPre { p.pre.params with W := Tensor3.unflatten θ })) (Tensor3.flatten p.Wq)
        (den (SHlo.depthwiseWeightGradB xN p.bq xin p.Wq
          (.operand cotN (mnv4SCotQc N s p xin dyOut)))))
  ∧ (HasGradAt (fun θ => Φ (p.withPre { p.pre.params with γ := θ })) p.gq
        (den (SHlo.bnGammaGradB vN epsStr p.eq_ (reassocB N s.ic (2 * s.h) (2 * s.h) qc)
          (.operand cotN (reassocB N s.ic (2 * s.h) (2 * s.h) (mnv4SCotQn N s p xin dyOut))))))
  ∧ (HasGradAt (fun θ => Φ (p.withPre { p.pre.params with β := θ })) p.bq2
        (den (SHlo.bnBetaGradB (N := N) (oc := s.ic) (h := 2 * s.h) (w := 2 * s.h)
          (.operand cotN (reassocB N s.ic (2 * s.h) (2 * s.h) (mnv4SCotQn N s p xin dyOut))))))
  ∧ (HasGradAt (fun θ => Φ { p with We := Kernel4.unflatten θ }) (Kernel4.flatten p.We)
        (den (SHlo.convWeightGradB xN p.be qr p.We (.operand cotN (mnv4SCotEc N s p xin dyOut)))))
  ∧ (HasGradAt (fun θ => Φ { p with ge := θ }) p.ge
        (den (SHlo.bnGammaGradB vN epsStr p.ee
          (reassocB N (s.ic * s.expand) (2 * s.h) (2 * s.h) ec)
          (.operand cotN (reassocB N (s.ic * s.expand) (2 * s.h) (2 * s.h)
          (mnv4SCotEn N s p xin dyOut))))))
  ∧ (HasGradAt (fun θ => Φ { p with be2 := θ }) p.be2
        (den (SHlo.bnBetaGradB (N := N) (oc := s.ic * s.expand) (h := 2 * s.h) (w := 2 * s.h)
          (.operand cotN (reassocB N (s.ic * s.expand) (2 * s.h) (2 * s.h)
          (mnv4SCotEn N s p xin dyOut))))))
  ∧ (HasGradAt (fun θ => Φ (p.withPost { p.post.params with W := Tensor3.unflatten θ })) (Tensor3.flatten p.Wd)
        (den (SHlo.depthwiseStridedWeightGradB xN p.bd er p.Wd
          (.operand cotN (mnv4SCotDc N s p xin dyOut)))))
  ∧ (HasGradAt (fun θ => Φ (p.withPost { p.post.params with γ := θ })) p.gd
        (den (SHlo.bnGammaGradB vN epsStr p.ed (reassocB N (s.ic * s.expand) s.h s.h dc)
          (.operand cotN (reassocB N (s.ic * s.expand) s.h s.h (mnv4SCotDn N s p xin dyOut))))))
  ∧ (HasGradAt (fun θ => Φ (p.withPost { p.post.params with β := θ })) p.bd2
        (den (SHlo.bnBetaGradB (N := N) (oc := s.ic * s.expand) (h := s.h) (w := s.h)
          (.operand cotN (reassocB N (s.ic * s.expand) s.h s.h (mnv4SCotDn N s p xin dyOut))))))
  ∧ (HasGradAt (fun θ => Φ { p with Wz := Kernel4.unflatten θ }) (Kernel4.flatten p.Wz)
        (den (SHlo.convWeightGradB xN p.bz dr p.Wz (.operand cotN (mnv4SCotPc N s p xin dyOut)))))
  ∧ (HasGradAt (fun θ => Φ { p with gz := θ }) p.gz
        (den (SHlo.bnGammaGradB vN epsStr p.ez (reassocB N s.oc s.h s.h pc)
          (.operand cotN (reassocB N s.oc s.h s.h dyOut)))))
  ∧ (HasGradAt (fun θ => Φ { p with bz2 := θ }) p.bz2
        (den (SHlo.bnBetaGradB (N := N) (oc := s.oc) (h := s.h) (w := s.h)
          (.operand cotN (reassocB N s.oc s.h s.h dyOut)))))

theorem mnv4_strided_lossTiedB (xN cotN vN epsStr : String) (p : UibParams s)
    (hq0 : s.preDWk ≠ 0) (hd0 : s.postDWk ≠ 0) (v : Vec (N * (s.ic * (2 * s.h) * (2 * s.h))))
    (hok : (mnv4StridedBodyOfRow N s p).ok v) {Gn : Vec (N * (s.oc * s.h * s.h)) → Vec 1}
    {dy : Vec (N * (s.oc * s.h * s.h))} (hGn : HasGradAt Gn ((mnv4StridedBodyOfRow N s p).fwd v) dy)
    {Φ : UibParams s → Vec 1} (hΦ : ∀ p', Φ p' = Gn ((mnv4StridedBodyOfRow N s p').fwd v)) :
    mnv4StridedLossTiedB N s xN cotN vN epsStr p v Φ dy := by
  have hfwd : ∀ p' : UibParams s, (mnv4StridedBodyOfRow N s p').fwd v
      = projB N (h := s.h) (w := s.h) p'.Wz p'.bz p'.ez p'.gz p'.bz2
          (dwbReluBstrided N (h := s.h) (w := s.h) p'.Wd p'.bd p'.ed p'.gd p'.bd2
            (cbReluB N (h := 2 * s.h) (w := 2 * s.h) p'.We p'.be p'.ee p'.ge p'.be2
              (dwbB N (h := 2 * s.h) (w := 2 * s.h) p'.Wq p'.bq p'.eq_ p'.gq p'.bq2 v))) :=
    fun p' => by
    rw [mnv4StridedBodyOfRow, mnv4UibStridedBody, CertLayer.comp_fwd_apply,
      CertLayer.comp_fwd_apply, CertLayer.comp_fwd_apply, mnv4PreDWSlot_fwd_apply_of_ne_zero N hq0]
    rfl
  have hsE : ∀ k, bnBatchLA N (s.ic * s.expand) (2 * s.h) (2 * s.h) p.ee p.ge p.be2
      (batchMap N (flatConv p.We p.be)
        (dwbB N (h := 2 * s.h) (w := 2 * s.h) p.Wq p.bq p.eq_ p.gq p.bq2 v)) k ≠ 0 := by
    have h := hok.2.1
    rw [mnv4PreDWSlot_fwd_apply_of_ne_zero N hq0] at h
    exact h
  have hsD : ∀ k, bnBatchLA N (s.ic * s.expand) s.h s.h p.ed p.gd p.bd2
      (batchMap N (depthwiseStride2Flat p.Wd p.bd)
        (cbReluB N (h := 2 * s.h) (w := 2 * s.h) p.We p.be p.ee p.ge p.be2
          (dwbB N (h := 2 * s.h) (w := 2 * s.h) p.Wq p.bq p.eq_ p.gq p.bq2 v))) k ≠ 0 := by
    have h := hok.2.2.1
    rw [mnv4PreDWSlot_fwd_apply_of_ne_zero N hq0] at h
    exact h
  rw [show Φ = fun p' => Gn ((mnv4StridedBodyOfRow N s p').fwd v) from funext hΦ]
  rw [hfwd p] at hGn
  have hPc := hasGradAt_bnBatchLA p.ez p.hz p.gz p.bz2 _ hGn
  have hDn := hasGradAt_relu _ hsD (hasGradAt_conv p.Wz p.bz _ hPc)
  have hDc := hasGradAt_bnBatchLA p.ed p.hd p.gd p.bd2 _ hDn
  have hEn := hasGradAt_relu _ hsE (hasGradAt_depthwiseStrided p.Wd p.bd _ hDc)
  have hEc := hasGradAt_bnBatchLA p.ee p.he p.ge p.be2 _ hEn
  have hQn := hasGradAt_conv p.We p.be _ hEc
  have hQc := hasGradAt_bnBatchLA p.eq_ p.hq p.gq p.bq2 _ hQn
  simp only [mnv4StridedLossTiedB, mnv4SCotQc, mnv4SCotQn, mnv4SCotEc, mnv4SCotEn, mnv4SCotDc,
    mnv4SCotDn, mnv4SCotPc, mnv4PreDWSlot_fwd_apply_of_ne_zero N hq0, hfwd, UibParams.Wq,
    UibParams.bq, UibParams.eq_, UibParams.gq, UibParams.bq2, UibParams.Wd, UibParams.bd,
    UibParams.ed, UibParams.gd, UibParams.bd2, UibParams.withPre_pre_params _ _ hq0,
    UibParams.withPost_post_params _ _ hd0]
  exact ⟨GradNodeB.depthwiseW_hasGradAt xN cotN _ v _ hQc,
    GradNodeB.bnGamma_hasGradAt vN epsStr cotN _ _ _ _ hQn,
    GradNodeB.bnBeta_hasGradAt cotN _ _ _ _ hQn,
    GradNodeB.convW_hasGradAt xN cotN _ _ _ hEc,
    GradNodeB.bnGamma_hasGradAt vN epsStr cotN _ _ _ _ hEn,
    GradNodeB.bnBeta_hasGradAt cotN _ _ _ _ hEn,
    GradNodeB.depthwiseStridedW_hasGradAt xN cotN _ _ _ hDc,
    GradNodeB.bnGamma_hasGradAt vN epsStr cotN _ _ _ _ hDn,
    GradNodeB.bnBeta_hasGradAt cotN _ _ _ _ hDn,
    GradNodeB.convW_hasGradAt xN cotN _ _ _ hPc,
    GradNodeB.bnGamma_hasGradAt vN epsStr cotN _ _ _ _ hGn,
    GradNodeB.bnBeta_hasGradAt cotN _ _ _ _ hGn⟩

end Strided

-- ════════════════════════════════════════════════════════════════
-- § The whole net: the loss after each block, and the net with one block's weights varied
-- ════════════════════════════════════════════════════════════════

/-- Pull a gradient back through any certified layer, at a point it certifies. -/
theorem certLayer_hasGradAt_comp {m n : Nat} (L : CertLayer m n) (x : Vec m) (hx : L.ok x)
    {G : Vec n → Vec 1} {dy : Vec n} (hG : HasGradAt G (L.fwd x) dy) :
    HasGradAt (fun y => G (L.fwd y)) x ((L.vjp x hx).backward dy) :=
  HasGradAt.comp (f := L.fwd) (x := x) hG (L.diff x hx) (L.vjp x hx)

/-- A skip block's step: the emitted fan-in `body dx + dyOut` is the residual layer's backward. -/
theorem mnv4Skip_hasGradAt_comp {N n : Nat} (L : CertLayer (N * n) (N * n)) (v : Vec (N * n))
    (hok : L.ok v) {G : Vec (N * n) → Vec 1} {dy bodyDx : Vec (N * n)}
    (hdx : bodyDx = (L.vjp v hok).backward dy) (hG : HasGradAt G ((CertLayer.residual L).fwd v) dy) :
    HasGradAt (fun y => G ((CertLayer.residual L).fwd y)) v (mnv4SkipCotIn bodyDx dy) :=
  (certLayer_hasGradAt_comp (CertLayer.residual L) v hok hG).of_eq
    (by rw [hdx, mnv4SkipCotIn_eq_vjp])

/-- At a skip block the loss read at the BODY output is `u ↦ G (u + v)`: the skip is a constant
    once a body parameter varies, so its gradient there is still the block-output cotangent. -/
theorem mnv4_residual_body_hasGradAt {N n : Nat} (L : CertLayer (N * n) (N * n)) (v : Vec (N * n))
    {G : Vec (N * n) → Vec 1} {dy : Vec (N * n)} (hG : HasGradAt G ((CertLayer.residual L).fwd v) dy) :
    HasGradAt (fun u => G (fun i => u i + v i)) (L.fwd v) dy :=
  HasGradAt.comp (f := fun u i => u i + v i) (x := L.fwd v) hG (differentiableAt_id.add_const v)
    (addConstHasVJPAt (fun u => u) v _ differentiableAt_id (identityHasVJPAt _ _))

/-- The group prefixes meet the block prefixes at every group boundary. -/
theorem mnv4Pre2_eq_blk (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) (x : Vec (N * (3 * 224 * 224))) :
    mnv4Pre2 N w x = mnv4Blk2 N w x := by
  rw [mnv4Pre2, mnv4Res28Layer, CertLayer.comp_fwd_apply]; rfl

theorem mnv4Pre3_eq_blk (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) (x : Vec (N * (3 * 224 * 224))) :
    mnv4Pre3 N w x = mnv4Blk6 N w x := by
  rw [mnv4Pre3, mnv4Res14aLayer, CertLayer.comp_fwd_apply, CertLayer.comp_fwd_apply,
    CertLayer.comp_fwd_apply, mnv4Pre2_eq_blk]; rfl

theorem mnv4Pre4_eq_blk (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) (x : Vec (N * (3 * 224 * 224))) :
    mnv4Pre4 N w x = mnv4Blk10 N w x := by
  rw [mnv4Pre4, mnv4Res14bLayer, CertLayer.comp_fwd_apply, CertLayer.comp_fwd_apply,
    CertLayer.comp_fwd_apply, mnv4Pre3_eq_blk]; rfl

theorem mnv4Pre5_eq_blk (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) (x : Vec (N * (3 * 224 * 224))) :
    mnv4Pre5 N w x = mnv4Blk15 N w x := by
  rw [mnv4Pre5, mnv4Res7aLayer, CertLayer.comp_fwd_apply, CertLayer.comp_fwd_apply,
    CertLayer.comp_fwd_apply, CertLayer.comp_fwd_apply, mnv4Pre4_eq_blk]; rfl

theorem mnv4Pre6_eq_blk (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) (x : Vec (N * (3 * 224 * 224))) :
    mnv4Pre6 N w x = mnv4Blk21 N w x := by
  rw [mnv4Pre6, mnv4Res7bLayer, CertLayer.comp_fwd_apply, CertLayer.comp_fwd_apply,
    CertLayer.comp_fwd_apply, CertLayer.comp_fwd_apply, CertLayer.comp_fwd_apply,
    mnv4Pre5_eq_blk]; rfl

/-- The fused stage's output, as the plain stage functions. -/
theorem mnv4Blk0_eq (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) (x : Vec (N * (3 * 224 * 224))) :
    mnv4Blk0 N w x = projB N (h := 56) (w := 56) w.f0pW w.f0pb w.f0pE w.f0pg w.f0pbt
      (cbReluStridedB N (h := 56) (w := 56) w.f0cW w.f0cb w.f0cE w.f0cg w.f0cbt (mnv4Pre0 N w x)) := by
  rw [mnv4Blk0, mnv4Pre1, mnv4FusedStack, mnv4FusedStage, CertLayer.comp_fwd_apply]; rfl

theorem mnv4Pre0_eq (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) (x : Vec (N * (3 * 224 * 224))) :
    mnv4Pre0 N w x = mnv4StemB N 112 112 w.sW w.sb w.sE w.sg w.sbt x := rfl

/-- The net after block 21 — the head. -/
noncomputable def mnv4Suf21 (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) :
    Vec (N * (256 * 7 * 7)) → Vec (N * nCls) :=
  (mnv4HeadStack N w).fwd

/-- The net after block 20: block 21, then the rest. -/
noncomputable def mnv4Suf20 (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) :
    Vec (N * (256 * 7 * 7)) → Vec (N * nCls) :=
  fun y => mnv4Suf21 N w ((CertLayer.residual (mnv4BodyOfRow N mnv4Row21 w.b21)).fwd y)

/-- The net after block 19: block 20, then the rest. -/
noncomputable def mnv4Suf19 (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) :
    Vec (N * (256 * 7 * 7)) → Vec (N * nCls) :=
  fun y => mnv4Suf20 N w ((CertLayer.residual (mnv4BodyOfRow N mnv4Row20 w.b20)).fwd y)

/-- The net after block 18: block 19, then the rest. -/
noncomputable def mnv4Suf18 (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) :
    Vec (N * (256 * 7 * 7)) → Vec (N * nCls) :=
  fun y => mnv4Suf19 N w ((CertLayer.residual (mnv4BodyOfRow N mnv4Row19 w.b19)).fwd y)

/-- The net after block 17: block 18, then the rest. -/
noncomputable def mnv4Suf17 (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) :
    Vec (N * (256 * 7 * 7)) → Vec (N * nCls) :=
  fun y => mnv4Suf18 N w ((CertLayer.residual (mnv4BodyOfRow N mnv4Row18 w.b18)).fwd y)

/-- The net after block 16: block 17, then the rest. -/
noncomputable def mnv4Suf16 (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) :
    Vec (N * (256 * 7 * 7)) → Vec (N * nCls) :=
  fun y => mnv4Suf17 N w ((CertLayer.residual (mnv4BodyOfRow N mnv4Row17 w.b17)).fwd y)

/-- The net after block 15: block 16, then the rest. -/
noncomputable def mnv4Suf15 (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) :
    Vec (N * (256 * 7 * 7)) → Vec (N * nCls) :=
  fun y => mnv4Suf16 N w ((CertLayer.residual (mnv4BodyOfRow N mnv4Row16 w.b16)).fwd y)

/-- The net after block 14: block 15, then the rest. -/
noncomputable def mnv4Suf14 (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) :
    Vec (N * (256 * 7 * 7)) → Vec (N * nCls) :=
  fun y => mnv4Suf15 N w ((CertLayer.residual (mnv4BodyOfRow N mnv4Row15 w.b15)).fwd y)

/-- The net after block 13: block 14, then the rest. -/
noncomputable def mnv4Suf13 (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) :
    Vec (N * (256 * 7 * 7)) → Vec (N * nCls) :=
  fun y => mnv4Suf14 N w ((CertLayer.residual (mnv4BodyOfRow N mnv4Row14 w.b14)).fwd y)

/-- The net after block 12: block 13, then the rest. -/
noncomputable def mnv4Suf12 (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) :
    Vec (N * (256 * 7 * 7)) → Vec (N * nCls) :=
  fun y => mnv4Suf13 N w ((CertLayer.residual (mnv4BodyOfRow N mnv4Row13 w.b13)).fwd y)

/-- The net after block 11: block 12, then the rest. -/
noncomputable def mnv4Suf11 (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) :
    Vec (N * (256 * 7 * 7)) → Vec (N * nCls) :=
  fun y => mnv4Suf12 N w ((CertLayer.residual (mnv4BodyOfRow N mnv4Row12 w.b12)).fwd y)

/-- The net after block 10: block 11, then the rest. -/
noncomputable def mnv4Suf10 (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) :
    Vec (N * (160 * 14 * 14)) → Vec (N * nCls) :=
  fun y => mnv4Suf11 N w ((mnv4StridedBodyOfRow N mnv4Row11 w.b11).fwd y)

/-- The net after block 9: block 10, then the rest. -/
noncomputable def mnv4Suf9 (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) :
    Vec (N * (160 * 14 * 14)) → Vec (N * nCls) :=
  fun y => mnv4Suf10 N w ((CertLayer.residual (mnv4BodyOfRow N mnv4Row10 w.b10)).fwd y)

/-- The net after block 8: block 9, then the rest. -/
noncomputable def mnv4Suf8 (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) :
    Vec (N * (160 * 14 * 14)) → Vec (N * nCls) :=
  fun y => mnv4Suf9 N w ((CertLayer.residual (mnv4BodyOfRow N mnv4Row9 w.b9)).fwd y)

/-- The net after block 7: block 8, then the rest. -/
noncomputable def mnv4Suf7 (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) :
    Vec (N * (160 * 14 * 14)) → Vec (N * nCls) :=
  fun y => mnv4Suf8 N w ((CertLayer.residual (mnv4BodyOfRow N mnv4Row8 w.b8)).fwd y)

/-- The net after block 6: block 7, then the rest. -/
noncomputable def mnv4Suf6 (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) :
    Vec (N * (160 * 14 * 14)) → Vec (N * nCls) :=
  fun y => mnv4Suf7 N w ((CertLayer.residual (mnv4BodyOfRow N mnv4Row7 w.b7)).fwd y)

/-- The net after block 5: block 6, then the rest. -/
noncomputable def mnv4Suf5 (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) :
    Vec (N * (160 * 14 * 14)) → Vec (N * nCls) :=
  fun y => mnv4Suf6 N w ((CertLayer.residual (mnv4BodyOfRow N mnv4Row6 w.b6)).fwd y)

/-- The net after block 4: block 5, then the rest. -/
noncomputable def mnv4Suf4 (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) :
    Vec (N * (160 * 14 * 14)) → Vec (N * nCls) :=
  fun y => mnv4Suf5 N w ((CertLayer.residual (mnv4BodyOfRow N mnv4Row5 w.b5)).fwd y)

/-- The net after block 3: block 4, then the rest. -/
noncomputable def mnv4Suf3 (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) :
    Vec (N * (160 * 14 * 14)) → Vec (N * nCls) :=
  fun y => mnv4Suf4 N w ((CertLayer.residual (mnv4BodyOfRow N mnv4Row4 w.b4)).fwd y)

/-- The net after block 2: block 3, then the rest. -/
noncomputable def mnv4Suf2 (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) :
    Vec (N * (80 * 28 * 28)) → Vec (N * nCls) :=
  fun y => mnv4Suf3 N w ((mnv4StridedBodyOfRow N mnv4Row3 w.b3).fwd y)

/-- The net after block 1: block 2, then the rest. -/
noncomputable def mnv4Suf1 (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) :
    Vec (N * (80 * 28 * 28)) → Vec (N * nCls) :=
  fun y => mnv4Suf2 N w ((CertLayer.residual (mnv4BodyOfRow N mnv4Row2 w.b2)).fwd y)

/-- The net after the fused stage: block 1, then the rest. -/
noncomputable def mnv4Suf0 (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) :
    Vec (N * (48 * 56 * 56)) → Vec (N * nCls) :=
  fun y => mnv4Suf1 N w ((mnv4StridedBodyOfRow N mnv4Row1 w.b1).fwd y)

/-- The net after the stem: the fused stage, then the rest. -/
noncomputable def mnv4SufStem (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) :
    Vec (N * (32 * 112 * 112)) → Vec (N * nCls) :=
  fun y => mnv4Suf0 N w ((mnv4FusedStack N w).fwd y)

/-- **The net with the stem's trained parameters varied** is the suffix after the stem at the
    varied stem. -/
theorem mnv4_factor_stem (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) (x : Vec (N * (3 * 224 * 224)))
    (W : Kernel4 32 3 3 3) (γ β : Vec 32) :
    mobilenetv4ForwardBFull N { w with sW := W, sg := γ, sbt := β } x
      = mnv4SufStem N w (mnv4StemB N 112 112 W w.sb w.sE γ β x) := by
  simp only [mobilenetv4ForwardBFull, mnv4Pre6, mnv4Pre5, mnv4Pre4, mnv4Pre3, mnv4Pre2, mnv4Pre1,
    mnv4Pre0, mnv4Res28Layer, mnv4Res14aLayer, mnv4Res14bLayer, mnv4Res7aLayer, mnv4Res7bLayer, CertLayer.comp_fwd_apply, mnv4HeadStack, mnv4FusedStack,
    mnv4SufStem, mnv4Suf0, mnv4Suf1, mnv4Suf2, mnv4Suf3, mnv4Suf4, mnv4Suf5, mnv4Suf6, mnv4Suf7, mnv4Suf8, mnv4Suf9, mnv4Suf10, mnv4Suf11, mnv4Suf12, mnv4Suf13, mnv4Suf14, mnv4Suf15, mnv4Suf16, mnv4Suf17, mnv4Suf18, mnv4Suf19, mnv4Suf20, mnv4Suf21]

/-- **The net with the fused stage's trained parameters varied.** -/
theorem mnv4_factor_fused (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) (x : Vec (N * (3 * 224 * 224)))
    (Wc : Kernel4 128 32 3 3) (γc βc : Vec 128) (Wp : Kernel4 48 128 1 1) (γp βp : Vec 48) :
    mobilenetv4ForwardBFull N
        { w with f0cW := Wc, f0cg := γc, f0cbt := βc, f0pW := Wp, f0pg := γp, f0pbt := βp } x
      = mnv4Suf0 N w (projB N (h := 56) (w := 56) Wp w.f0pb w.f0pE γp βp
          (cbReluStridedB N (h := 56) (w := 56) Wc w.f0cb w.f0cE γc βc (mnv4Pre0 N w x))) := by
  have h : mobilenetv4ForwardBFull N
        { w with f0cW := Wc, f0cg := γc, f0cbt := βc, f0pW := Wp, f0pg := γp, f0pbt := βp } x
      = mnv4Suf0 N w ((mnv4FusedStack N
        { w with f0cW := Wc, f0cg := γc, f0cbt := βc, f0pW := Wp, f0pg := γp, f0pbt := βp }).fwd
          (mnv4Pre0 N w x)) := by
    simp only [mobilenetv4ForwardBFull, mnv4Pre6, mnv4Pre5, mnv4Pre4, mnv4Pre3, mnv4Pre2, mnv4Pre1, mnv4Res28Layer, mnv4Res14aLayer, mnv4Res14bLayer, mnv4Res7aLayer, mnv4Res7bLayer, CertLayer.comp_fwd_apply]
    rfl
  rw [h, mnv4FusedStack, mnv4FusedStage, CertLayer.comp_fwd_apply]
  rfl

/-- **The net with block 1's weights varied** is the suffix after block 1 at the varied block. -/
theorem mnv4_factor_b1 (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) (x : Vec (N * (3 * 224 * 224)))
    (p : UibParams mnv4Row1) :
    mobilenetv4ForwardBFull N { w with b1 := p } x
      = mnv4Suf1 N w ((mnv4StridedBodyOfRow N mnv4Row1 p).fwd (mnv4Blk0 N w x)) := by
  simp only [mobilenetv4ForwardBFull, mnv4Pre6, mnv4Pre5, mnv4Pre4, mnv4Pre3, mnv4Pre2, mnv4Pre1, mnv4Res28Layer, mnv4Res14aLayer, mnv4Res14bLayer, mnv4Res7aLayer, mnv4Res7bLayer, CertLayer.comp_fwd_apply]
  rfl

/-- **The net with block 2's weights varied** is the suffix after block 2 at the varied block. -/
theorem mnv4_factor_b2 (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) (x : Vec (N * (3 * 224 * 224)))
    (p : UibParams mnv4Row2) :
    mobilenetv4ForwardBFull N { w with b2 := p } x
      = mnv4Suf2 N w ((CertLayer.residual (mnv4BodyOfRow N mnv4Row2 p)).fwd (mnv4Blk1 N w x)) := by
  simp only [mobilenetv4ForwardBFull, mnv4Pre6, mnv4Pre5, mnv4Pre4, mnv4Pre3, mnv4Pre2, mnv4Pre1, mnv4Res28Layer, mnv4Res14aLayer, mnv4Res14bLayer, mnv4Res7aLayer, mnv4Res7bLayer, CertLayer.comp_fwd_apply]
  rfl

/-- **The net with block 3's weights varied** is the suffix after block 3 at the varied block. -/
theorem mnv4_factor_b3 (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) (x : Vec (N * (3 * 224 * 224)))
    (p : UibParams mnv4Row3) :
    mobilenetv4ForwardBFull N { w with b3 := p } x
      = mnv4Suf3 N w ((mnv4StridedBodyOfRow N mnv4Row3 p).fwd (mnv4Blk2 N w x)) := by
  simp only [mobilenetv4ForwardBFull, mnv4Pre6, mnv4Pre5, mnv4Pre4, mnv4Pre3, mnv4Pre2, mnv4Pre1, mnv4Res28Layer, mnv4Res14aLayer, mnv4Res14bLayer, mnv4Res7aLayer, mnv4Res7bLayer, CertLayer.comp_fwd_apply]
  rfl

/-- **The net with block 4's weights varied** is the suffix after block 4 at the varied block. -/
theorem mnv4_factor_b4 (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) (x : Vec (N * (3 * 224 * 224)))
    (p : UibParams mnv4Row4) :
    mobilenetv4ForwardBFull N { w with b4 := p } x
      = mnv4Suf4 N w ((CertLayer.residual (mnv4BodyOfRow N mnv4Row4 p)).fwd (mnv4Blk3 N w x)) := by
  simp only [mobilenetv4ForwardBFull, mnv4Pre6, mnv4Pre5, mnv4Pre4, mnv4Pre3, mnv4Pre2, mnv4Pre1, mnv4Res28Layer, mnv4Res14aLayer, mnv4Res14bLayer, mnv4Res7aLayer, mnv4Res7bLayer, CertLayer.comp_fwd_apply]
  rfl

/-- **The net with block 5's weights varied** is the suffix after block 5 at the varied block. -/
theorem mnv4_factor_b5 (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) (x : Vec (N * (3 * 224 * 224)))
    (p : UibParams mnv4Row5) :
    mobilenetv4ForwardBFull N { w with b5 := p } x
      = mnv4Suf5 N w ((CertLayer.residual (mnv4BodyOfRow N mnv4Row5 p)).fwd (mnv4Blk4 N w x)) := by
  simp only [mobilenetv4ForwardBFull, mnv4Pre6, mnv4Pre5, mnv4Pre4, mnv4Pre3, mnv4Pre2, mnv4Pre1, mnv4Res28Layer, mnv4Res14aLayer, mnv4Res14bLayer, mnv4Res7aLayer, mnv4Res7bLayer, CertLayer.comp_fwd_apply]
  rfl

/-- **The net with block 6's weights varied** is the suffix after block 6 at the varied block. -/
theorem mnv4_factor_b6 (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) (x : Vec (N * (3 * 224 * 224)))
    (p : UibParams mnv4Row6) :
    mobilenetv4ForwardBFull N { w with b6 := p } x
      = mnv4Suf6 N w ((CertLayer.residual (mnv4BodyOfRow N mnv4Row6 p)).fwd (mnv4Blk5 N w x)) := by
  simp only [mobilenetv4ForwardBFull, mnv4Pre6, mnv4Pre5, mnv4Pre4, mnv4Pre3, mnv4Pre2, mnv4Pre1, mnv4Res28Layer, mnv4Res14aLayer, mnv4Res14bLayer, mnv4Res7aLayer, mnv4Res7bLayer, CertLayer.comp_fwd_apply]
  rfl

/-- **The net with block 7's weights varied** is the suffix after block 7 at the varied block. -/
theorem mnv4_factor_b7 (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) (x : Vec (N * (3 * 224 * 224)))
    (p : UibParams mnv4Row7) :
    mobilenetv4ForwardBFull N { w with b7 := p } x
      = mnv4Suf7 N w ((CertLayer.residual (mnv4BodyOfRow N mnv4Row7 p)).fwd (mnv4Blk6 N w x)) := by
  simp only [mobilenetv4ForwardBFull, mnv4Pre6, mnv4Pre5, mnv4Pre4, mnv4Pre3, mnv4Pre2, mnv4Pre1, mnv4Res28Layer, mnv4Res14aLayer, mnv4Res14bLayer, mnv4Res7aLayer, mnv4Res7bLayer, CertLayer.comp_fwd_apply]
  rfl

/-- **The net with block 8's weights varied** is the suffix after block 8 at the varied block. -/
theorem mnv4_factor_b8 (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) (x : Vec (N * (3 * 224 * 224)))
    (p : UibParams mnv4Row8) :
    mobilenetv4ForwardBFull N { w with b8 := p } x
      = mnv4Suf8 N w ((CertLayer.residual (mnv4BodyOfRow N mnv4Row8 p)).fwd (mnv4Blk7 N w x)) := by
  simp only [mobilenetv4ForwardBFull, mnv4Pre6, mnv4Pre5, mnv4Pre4, mnv4Pre3, mnv4Pre2, mnv4Pre1, mnv4Res28Layer, mnv4Res14aLayer, mnv4Res14bLayer, mnv4Res7aLayer, mnv4Res7bLayer, CertLayer.comp_fwd_apply]
  rfl

/-- **The net with block 9's weights varied** is the suffix after block 9 at the varied block. -/
theorem mnv4_factor_b9 (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) (x : Vec (N * (3 * 224 * 224)))
    (p : UibParams mnv4Row9) :
    mobilenetv4ForwardBFull N { w with b9 := p } x
      = mnv4Suf9 N w ((CertLayer.residual (mnv4BodyOfRow N mnv4Row9 p)).fwd (mnv4Blk8 N w x)) := by
  simp only [mobilenetv4ForwardBFull, mnv4Pre6, mnv4Pre5, mnv4Pre4, mnv4Pre3, mnv4Pre2, mnv4Pre1, mnv4Res28Layer, mnv4Res14aLayer, mnv4Res14bLayer, mnv4Res7aLayer, mnv4Res7bLayer, CertLayer.comp_fwd_apply]
  rfl

/-- **The net with block 10's weights varied** is the suffix after block 10 at the varied block. -/
theorem mnv4_factor_b10 (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) (x : Vec (N * (3 * 224 * 224)))
    (p : UibParams mnv4Row10) :
    mobilenetv4ForwardBFull N { w with b10 := p } x
      = mnv4Suf10 N w ((CertLayer.residual (mnv4BodyOfRow N mnv4Row10 p)).fwd (mnv4Blk9 N w x)) := by
  simp only [mobilenetv4ForwardBFull, mnv4Pre6, mnv4Pre5, mnv4Pre4, mnv4Pre3, mnv4Pre2, mnv4Pre1, mnv4Res28Layer, mnv4Res14aLayer, mnv4Res14bLayer, mnv4Res7aLayer, mnv4Res7bLayer, CertLayer.comp_fwd_apply]
  rfl

/-- **The net with block 11's weights varied** is the suffix after block 11 at the varied block. -/
theorem mnv4_factor_b11 (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) (x : Vec (N * (3 * 224 * 224)))
    (p : UibParams mnv4Row11) :
    mobilenetv4ForwardBFull N { w with b11 := p } x
      = mnv4Suf11 N w ((mnv4StridedBodyOfRow N mnv4Row11 p).fwd (mnv4Blk10 N w x)) := by
  simp only [mobilenetv4ForwardBFull, mnv4Pre6, mnv4Pre5, mnv4Pre4, mnv4Pre3, mnv4Pre2, mnv4Pre1, mnv4Res28Layer, mnv4Res14aLayer, mnv4Res14bLayer, mnv4Res7aLayer, mnv4Res7bLayer, CertLayer.comp_fwd_apply]
  rfl

/-- **The net with block 12's weights varied** is the suffix after block 12 at the varied block. -/
theorem mnv4_factor_b12 (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) (x : Vec (N * (3 * 224 * 224)))
    (p : UibParams mnv4Row12) :
    mobilenetv4ForwardBFull N { w with b12 := p } x
      = mnv4Suf12 N w ((CertLayer.residual (mnv4BodyOfRow N mnv4Row12 p)).fwd (mnv4Blk11 N w x)) := by
  simp only [mobilenetv4ForwardBFull, mnv4Pre6, mnv4Pre5, mnv4Pre4, mnv4Pre3, mnv4Pre2, mnv4Pre1, mnv4Res28Layer, mnv4Res14aLayer, mnv4Res14bLayer, mnv4Res7aLayer, mnv4Res7bLayer, CertLayer.comp_fwd_apply]
  rfl

/-- **The net with block 13's weights varied** is the suffix after block 13 at the varied block. -/
theorem mnv4_factor_b13 (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) (x : Vec (N * (3 * 224 * 224)))
    (p : UibParams mnv4Row13) :
    mobilenetv4ForwardBFull N { w with b13 := p } x
      = mnv4Suf13 N w ((CertLayer.residual (mnv4BodyOfRow N mnv4Row13 p)).fwd (mnv4Blk12 N w x)) := by
  simp only [mobilenetv4ForwardBFull, mnv4Pre6, mnv4Pre5, mnv4Pre4, mnv4Pre3, mnv4Pre2, mnv4Pre1, mnv4Res28Layer, mnv4Res14aLayer, mnv4Res14bLayer, mnv4Res7aLayer, mnv4Res7bLayer, CertLayer.comp_fwd_apply]
  rfl

/-- **The net with block 14's weights varied** is the suffix after block 14 at the varied block. -/
theorem mnv4_factor_b14 (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) (x : Vec (N * (3 * 224 * 224)))
    (p : UibParams mnv4Row14) :
    mobilenetv4ForwardBFull N { w with b14 := p } x
      = mnv4Suf14 N w ((CertLayer.residual (mnv4BodyOfRow N mnv4Row14 p)).fwd (mnv4Blk13 N w x)) := by
  simp only [mobilenetv4ForwardBFull, mnv4Pre6, mnv4Pre5, mnv4Pre4, mnv4Pre3, mnv4Pre2, mnv4Pre1, mnv4Res28Layer, mnv4Res14aLayer, mnv4Res14bLayer, mnv4Res7aLayer, mnv4Res7bLayer, CertLayer.comp_fwd_apply]
  rfl

/-- **The net with block 15's weights varied** is the suffix after block 15 at the varied block. -/
theorem mnv4_factor_b15 (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) (x : Vec (N * (3 * 224 * 224)))
    (p : UibParams mnv4Row15) :
    mobilenetv4ForwardBFull N { w with b15 := p } x
      = mnv4Suf15 N w ((CertLayer.residual (mnv4BodyOfRow N mnv4Row15 p)).fwd (mnv4Blk14 N w x)) := by
  simp only [mobilenetv4ForwardBFull, mnv4Pre6, mnv4Pre5, mnv4Pre4, mnv4Pre3, mnv4Pre2, mnv4Pre1, mnv4Res28Layer, mnv4Res14aLayer, mnv4Res14bLayer, mnv4Res7aLayer, mnv4Res7bLayer, CertLayer.comp_fwd_apply]
  rfl

/-- **The net with block 16's weights varied** is the suffix after block 16 at the varied block. -/
theorem mnv4_factor_b16 (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) (x : Vec (N * (3 * 224 * 224)))
    (p : UibParams mnv4Row16) :
    mobilenetv4ForwardBFull N { w with b16 := p } x
      = mnv4Suf16 N w ((CertLayer.residual (mnv4BodyOfRow N mnv4Row16 p)).fwd (mnv4Blk15 N w x)) := by
  simp only [mobilenetv4ForwardBFull, mnv4Pre6, mnv4Pre5, mnv4Pre4, mnv4Pre3, mnv4Pre2, mnv4Pre1, mnv4Res28Layer, mnv4Res14aLayer, mnv4Res14bLayer, mnv4Res7aLayer, mnv4Res7bLayer, CertLayer.comp_fwd_apply]
  rfl

/-- **The net with block 17's weights varied** is the suffix after block 17 at the varied block. -/
theorem mnv4_factor_b17 (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) (x : Vec (N * (3 * 224 * 224)))
    (p : UibParams mnv4Row17) :
    mobilenetv4ForwardBFull N { w with b17 := p } x
      = mnv4Suf17 N w ((CertLayer.residual (mnv4BodyOfRow N mnv4Row17 p)).fwd (mnv4Blk16 N w x)) := by
  simp only [mobilenetv4ForwardBFull, mnv4Pre6, mnv4Pre5, mnv4Pre4, mnv4Pre3, mnv4Pre2, mnv4Pre1, mnv4Res28Layer, mnv4Res14aLayer, mnv4Res14bLayer, mnv4Res7aLayer, mnv4Res7bLayer, CertLayer.comp_fwd_apply]
  rfl

/-- **The net with block 18's weights varied** is the suffix after block 18 at the varied block. -/
theorem mnv4_factor_b18 (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) (x : Vec (N * (3 * 224 * 224)))
    (p : UibParams mnv4Row18) :
    mobilenetv4ForwardBFull N { w with b18 := p } x
      = mnv4Suf18 N w ((CertLayer.residual (mnv4BodyOfRow N mnv4Row18 p)).fwd (mnv4Blk17 N w x)) := by
  simp only [mobilenetv4ForwardBFull, mnv4Pre6, mnv4Pre5, mnv4Pre4, mnv4Pre3, mnv4Pre2, mnv4Pre1, mnv4Res28Layer, mnv4Res14aLayer, mnv4Res14bLayer, mnv4Res7aLayer, mnv4Res7bLayer, CertLayer.comp_fwd_apply]
  rfl

/-- **The net with block 19's weights varied** is the suffix after block 19 at the varied block. -/
theorem mnv4_factor_b19 (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) (x : Vec (N * (3 * 224 * 224)))
    (p : UibParams mnv4Row19) :
    mobilenetv4ForwardBFull N { w with b19 := p } x
      = mnv4Suf19 N w ((CertLayer.residual (mnv4BodyOfRow N mnv4Row19 p)).fwd (mnv4Blk18 N w x)) := by
  simp only [mobilenetv4ForwardBFull, mnv4Pre6, mnv4Pre5, mnv4Pre4, mnv4Pre3, mnv4Pre2, mnv4Pre1, mnv4Res28Layer, mnv4Res14aLayer, mnv4Res14bLayer, mnv4Res7aLayer, mnv4Res7bLayer, CertLayer.comp_fwd_apply]
  rfl

/-- **The net with block 20's weights varied** is the suffix after block 20 at the varied block. -/
theorem mnv4_factor_b20 (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) (x : Vec (N * (3 * 224 * 224)))
    (p : UibParams mnv4Row20) :
    mobilenetv4ForwardBFull N { w with b20 := p } x
      = mnv4Suf20 N w ((CertLayer.residual (mnv4BodyOfRow N mnv4Row20 p)).fwd (mnv4Blk19 N w x)) := by
  simp only [mobilenetv4ForwardBFull, mnv4Pre6, mnv4Pre5, mnv4Pre4, mnv4Pre3, mnv4Pre2, mnv4Pre1, mnv4Res28Layer, mnv4Res14aLayer, mnv4Res14bLayer, mnv4Res7aLayer, mnv4Res7bLayer, CertLayer.comp_fwd_apply]
  rfl

/-- **The net with block 21's weights varied** is the suffix after block 21 at the varied block. -/
theorem mnv4_factor_b21 (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) (x : Vec (N * (3 * 224 * 224)))
    (p : UibParams mnv4Row21) :
    mobilenetv4ForwardBFull N { w with b21 := p } x
      = mnv4Suf21 N w ((CertLayer.residual (mnv4BodyOfRow N mnv4Row21 p)).fwd (mnv4Blk20 N w x)) := by
  simp only [mobilenetv4ForwardBFull, mnv4Pre6, mnv4Pre5, mnv4Pre4, mnv4Pre3, mnv4Pre2, mnv4Pre1, mnv4Res28Layer, mnv4Res14aLayer, mnv4Res14bLayer, mnv4Res7aLayer, mnv4Res7bLayer, CertLayer.comp_fwd_apply]
  rfl

/-- **The net with the head's trained parameters varied** is the head at them. -/
theorem mnv4_factor_head (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) (x : Vec (N * (3 * 224 * 224)))
    (W1 : Kernel4 960 256 1 1) (γ1 β1 : Vec 960) (W2 : Kernel4 1280 960 1 1) (γ2 β2 : Vec 1280)
    (Wd : Mat 1280 nCls) (bd : Vec nCls) :
    mobilenetv4ForwardBFull N
        { w with h1W := W1, h1g := γ1, h1bt := β1, hW := W2, hg := γ2, hbt := β2, Wd := Wd, bd := bd } x
      = mnv4HeadFwd N 7 7 W1 w.h1b w.h1E γ1 β1 W2 w.hb w.hE γ2 β2 Wd bd (mnv4Blk21 N w x) := by
  rw [mobilenetv4ForwardBFull, mnv4Pre6_eq_blk]
  rfl

/-- **Every MobileNetV4-Conv-M parameter gradient node is the derivative of `L` in that
    parameter**, for a loss `L` of the logits and `g` the cotangent the chain starts from: the 233
    nodes `mnv4_net_tiedB` ties, each at the cotangent the emitted chain threads to it, stated
    against `L` of `mobilenetv4ForwardBFull` with that one parameter varied. -/
def Mnv4NetLossTiedB (N : Nat) {nCls : Nat} (xN cotN vN epsStr : String) (w : Mnv4BWeights nCls)
    (x : Vec (N * (3 * 224 * 224))) (L : Vec (N * nCls) → Vec 1) (g : Vec (N * nCls)) : Prop :=
    let dy21 := mnv4HeadCotIn N 7 7 w.h1W w.h1b w.h1E w.h1g w.h1bt w.hW w.hb w.hE w.hg w.hbt
                 w.Wd w.bd (mnv4Blk21 N w x) g
    let dy20 := mnv4SkipCotIn (mnv4BodyCotIn N mnv4Row21 w.b21 (mnv4Blk20 N w x) dy21) dy21
    let dy19 := mnv4SkipCotIn (mnv4BodyCotIn N mnv4Row20 w.b20 (mnv4Blk19 N w x) dy20) dy20
    let dy18 := mnv4SkipCotIn (mnv4BodyCotIn N mnv4Row19 w.b19 (mnv4Blk18 N w x) dy19) dy19
    let dy17 := mnv4SkipCotIn (mnv4BodyCotIn N mnv4Row18 w.b18 (mnv4Blk17 N w x) dy18) dy18
    let dy16 := mnv4SkipCotIn (mnv4BodyCotIn N mnv4Row17 w.b17 (mnv4Blk16 N w x) dy17) dy17
    let dy15 := mnv4SkipCotIn (mnv4BodyCotIn N mnv4Row16 w.b16 (mnv4Blk15 N w x) dy16) dy16
    let dy14 := mnv4SkipCotIn (mnv4BodyCotIn N mnv4Row15 w.b15 (mnv4Blk14 N w x) dy15) dy15
    let dy13 := mnv4SkipCotIn (mnv4BodyCotIn N mnv4Row14 w.b14 (mnv4Blk13 N w x) dy14) dy14
    let dy12 := mnv4SkipCotIn (mnv4BodyCotIn N mnv4Row13 w.b13 (mnv4Blk12 N w x) dy13) dy13
    let dy11 := mnv4SkipCotIn (mnv4BodyCotIn N mnv4Row12 w.b12 (mnv4Blk11 N w x) dy12) dy12
    let dy10 := mnv4SBodyCotIn N mnv4Row11 w.b11 (mnv4Blk10 N w x) dy11
    let dy9 := mnv4SkipCotIn (mnv4BodyCotIn N mnv4Row10 w.b10 (mnv4Blk9 N w x) dy10) dy10
    let dy8 := mnv4SkipCotIn (mnv4BodyCotIn N mnv4Row9 w.b9 (mnv4Blk8 N w x) dy9) dy9
    let dy7 := mnv4SkipCotIn (mnv4BodyCotIn N mnv4Row8 w.b8 (mnv4Blk7 N w x) dy8) dy8
    let dy6 := mnv4SkipCotIn (mnv4BodyCotIn N mnv4Row7 w.b7 (mnv4Blk6 N w x) dy7) dy7
    let dy5 := mnv4SkipCotIn (mnv4BodyCotIn N mnv4Row6 w.b6 (mnv4Blk5 N w x) dy6) dy6
    let dy4 := mnv4SkipCotIn (mnv4BodyCotIn N mnv4Row5 w.b5 (mnv4Blk4 N w x) dy5) dy5
    let dy3 := mnv4SkipCotIn (mnv4BodyCotIn N mnv4Row4 w.b4 (mnv4Blk3 N w x) dy4) dy4
    let dy2 := mnv4SBodyCotIn N mnv4Row3 w.b3 (mnv4Blk2 N w x) dy3
    let dy1 := mnv4SkipCotIn (mnv4BodyCotIn N mnv4Row2 w.b2 (mnv4Blk1 N w x) dy2) dy2
    let dy0 := mnv4SBodyCotIn N mnv4Row1 w.b1 (mnv4Blk0 N w x) dy1
    let dyStem := mnv4FusedCotIn N 56 56 w.f0cW w.f0cb w.f0cE w.f0cg w.f0cbt
                   w.f0pW w.f0pb w.f0pE w.f0pg w.f0pbt (mnv4Pre0 N w x) dy0
    mnv4StemLossTiedB (N := N) (h := 112) (w := 112) xN cotN vN epsStr w.sW w.sb w.sE w.sg w.sbt x
      (fun W γ β => L (mobilenetv4ForwardBFull N { w with sW := W, sg := γ, sbt := β } x)) dyStem
  ∧ mnv4FusedLossTiedB (N := N) (h := 56) (w := 56) w.f0cW w.f0cb w.f0cE w.f0cg w.f0cbt
      w.f0pW w.f0pb w.f0pE w.f0pg w.f0pbt xN cotN vN epsStr (mnv4Pre0 N w x)
      (fun Wc γc βc Wp γp βp => L (mobilenetv4ForwardBFull N
        { w with f0cW := Wc, f0cg := γc, f0cbt := βc, f0pW := Wp, f0pg := γp, f0pbt := βp } x)) dy0
  ∧ mnv4StridedLossTiedB N mnv4Row1 xN cotN vN epsStr w.b1 (mnv4Blk0 N w x)
      (fun p => L (mobilenetv4ForwardBFull N { w with b1 := p } x)) dy1
  ∧ mnv4ExtraDWLossTiedB N mnv4Row2 xN cotN vN epsStr w.b2 (mnv4Blk1 N w x)
      (fun p => L (mobilenetv4ForwardBFull N { w with b2 := p } x)) dy2
  ∧ mnv4StridedLossTiedB N mnv4Row3 xN cotN vN epsStr w.b3 (mnv4Blk2 N w x)
      (fun p => L (mobilenetv4ForwardBFull N { w with b3 := p } x)) dy3
  ∧ mnv4ExtraDWLossTiedB N mnv4Row4 xN cotN vN epsStr w.b4 (mnv4Blk3 N w x)
      (fun p => L (mobilenetv4ForwardBFull N { w with b4 := p } x)) dy4
  ∧ mnv4ExtraDWLossTiedB N mnv4Row5 xN cotN vN epsStr w.b5 (mnv4Blk4 N w x)
      (fun p => L (mobilenetv4ForwardBFull N { w with b5 := p } x)) dy5
  ∧ mnv4ExtraDWLossTiedB N mnv4Row6 xN cotN vN epsStr w.b6 (mnv4Blk5 N w x)
      (fun p => L (mobilenetv4ForwardBFull N { w with b6 := p } x)) dy6
  ∧ mnv4ExtraDWLossTiedB N mnv4Row7 xN cotN vN epsStr w.b7 (mnv4Blk6 N w x)
      (fun p => L (mobilenetv4ForwardBFull N { w with b7 := p } x)) dy7
  ∧ mnv4ConvNeXtLossTiedB N mnv4Row8 xN cotN vN epsStr w.b8 (mnv4Blk7 N w x)
      (fun p => L (mobilenetv4ForwardBFull N { w with b8 := p } x)) dy8
  ∧ mnv4FfnLossTiedB N mnv4Row9 xN cotN vN epsStr w.b9 (mnv4Blk8 N w x)
      (fun p => L (mobilenetv4ForwardBFull N { w with b9 := p } x)) dy9
  ∧ mnv4ConvNeXtLossTiedB N mnv4Row10 xN cotN vN epsStr w.b10 (mnv4Blk9 N w x)
      (fun p => L (mobilenetv4ForwardBFull N { w with b10 := p } x)) dy10
  ∧ mnv4StridedLossTiedB N mnv4Row11 xN cotN vN epsStr w.b11 (mnv4Blk10 N w x)
      (fun p => L (mobilenetv4ForwardBFull N { w with b11 := p } x)) dy11
  ∧ mnv4ExtraDWLossTiedB N mnv4Row12 xN cotN vN epsStr w.b12 (mnv4Blk11 N w x)
      (fun p => L (mobilenetv4ForwardBFull N { w with b12 := p } x)) dy12
  ∧ mnv4ExtraDWLossTiedB N mnv4Row13 xN cotN vN epsStr w.b13 (mnv4Blk12 N w x)
      (fun p => L (mobilenetv4ForwardBFull N { w with b13 := p } x)) dy13
  ∧ mnv4ExtraDWLossTiedB N mnv4Row14 xN cotN vN epsStr w.b14 (mnv4Blk13 N w x)
      (fun p => L (mobilenetv4ForwardBFull N { w with b14 := p } x)) dy14
  ∧ mnv4FfnLossTiedB N mnv4Row15 xN cotN vN epsStr w.b15 (mnv4Blk14 N w x)
      (fun p => L (mobilenetv4ForwardBFull N { w with b15 := p } x)) dy15
  ∧ mnv4ConvNeXtLossTiedB N mnv4Row16 xN cotN vN epsStr w.b16 (mnv4Blk15 N w x)
      (fun p => L (mobilenetv4ForwardBFull N { w with b16 := p } x)) dy16
  ∧ mnv4ExtraDWLossTiedB N mnv4Row17 xN cotN vN epsStr w.b17 (mnv4Blk16 N w x)
      (fun p => L (mobilenetv4ForwardBFull N { w with b17 := p } x)) dy17
  ∧ mnv4ExtraDWLossTiedB N mnv4Row18 xN cotN vN epsStr w.b18 (mnv4Blk17 N w x)
      (fun p => L (mobilenetv4ForwardBFull N { w with b18 := p } x)) dy18
  ∧ mnv4FfnLossTiedB N mnv4Row19 xN cotN vN epsStr w.b19 (mnv4Blk18 N w x)
      (fun p => L (mobilenetv4ForwardBFull N { w with b19 := p } x)) dy19
  ∧ mnv4FfnLossTiedB N mnv4Row20 xN cotN vN epsStr w.b20 (mnv4Blk19 N w x)
      (fun p => L (mobilenetv4ForwardBFull N { w with b20 := p } x)) dy20
  ∧ mnv4ConvNeXtLossTiedB N mnv4Row21 xN cotN vN epsStr w.b21 (mnv4Blk20 N w x)
      (fun p => L (mobilenetv4ForwardBFull N { w with b21 := p } x)) dy21
  ∧ mnv4HeadLossTiedB (N := N) (h := 7) (w := 7) w.h1W w.h1b w.h1E w.h1g w.h1bt w.hW w.hb w.hE w.hg
      w.hbt w.Wd w.bd xN cotN vN epsStr (mnv4Blk21 N w x)
      (fun W1 γ1 β1 W2 γ2 β2 Wd bd => L (mobilenetv4ForwardBFull N
        { w with h1W := W1, h1g := γ1, h1bt := β1, hW := W2, hg := γ2, hbt := β2, Wd := Wd, bd := bd } x))
      g

/-- **Every MobileNetV4-Conv-M parameter gradient node is the derivative of the loss in that
    parameter.** For any loss `L` of the logits with gradient `g` at the net's output, each of the
    233 nodes `mnv4_net_tiedB` ties — at the same cotangent — is `∂L/∂θ` of the WHOLE net,
    `mobilenetv4ForwardBFull` with that one parameter varied (a stem, fused-stage or head field,
    or a block's weight record `w.bk := p` with one slot changed — a depthwise slot through
    `withPre` / `withPost`).

    The only hypothesis is `Mnv4SmoothAt` (the stem's relu clause and each group's `.ok`); the BN
    `ε > 0` facts are fields of the weights. The loss enters only through `hL`;
    `mnv4_net_lossGrad_smoothedCE` discharges it for the loss the artifacts ship. -/
theorem mnv4_net_lossGrad (N : Nat) {nCls : Nat} (xN cotN vN epsStr : String)
    (w : Mnv4BWeights nCls) (x : Vec (N * (3 * 224 * 224))) (hx : Mnv4SmoothAt N w x)
    {L : Vec (N * nCls) → Vec 1} {g : Vec (N * nCls)}
    (hL : HasGradAt L (mobilenetv4ForwardBFull N w x) g) :
    Mnv4NetLossTiedB N xN cotN vN epsStr w x L g := by
  unfold Mnv4NetLossTiedB
  intro dy21 dy20 dy19 dy18 dy17 dy16 dy15 dy14 dy13 dy12 dy11 dy10 dy9 dy8 dy7 dy6 dy5 dy4 dy3
    dy2 dy1 dy0 dyStem
  have hh := hx.head
  rw [mnv4Pre6_eq_blk] at hh
  have hL' : HasGradAt L ((mnv4HeadStack N w).fwd (mnv4Blk21 N w x)) g :=
    hL.congr_point (by rw [mobilenetv4ForwardBFull, mnv4Pre6_eq_blk])
  have h21 : HasGradAt (fun y => L (mnv4Suf21 N w y)) (mnv4Blk21 N w x) dy21 :=
    (certLayer_hasGradAt_comp (mnv4HeadStack N w) _ hh hL').of_eq
      (mnv4HeadCotIn_eq_vjp N 7 7 w.h1W w.h1b w.h1E w.hh1E w.h1g w.h1bt w.hW w.hb w.hE w.hhE w.hg
        w.hbt w.Wd w.bd _ g hh).symm
  have ok1 : (mnv4StridedBodyOfRow N mnv4Row1 w.b1).ok (mnv4Blk0 N w x) := hx.g28.1
  have ok2 : (mnv4BodyOfRow N mnv4Row2 w.b2).ok (mnv4Blk1 N w x) := hx.g28.2
  have ok3 : (mnv4StridedBodyOfRow N mnv4Row3 w.b3).ok (mnv4Blk2 N w x) := hx.g14a.1
  have ok4 : (mnv4BodyOfRow N mnv4Row4 w.b4).ok (mnv4Blk3 N w x) := hx.g14a.2.1
  have ok5 : (mnv4BodyOfRow N mnv4Row5 w.b5).ok (mnv4Blk4 N w x) := hx.g14a.2.2.1
  have ok6 : (mnv4BodyOfRow N mnv4Row6 w.b6).ok (mnv4Blk5 N w x) := hx.g14a.2.2.2
  have ok7 : (mnv4BodyOfRow N mnv4Row7 w.b7).ok (mnv4Blk6 N w x) := hx.g14b.1
  have ok8 : (mnv4BodyOfRow N mnv4Row8 w.b8).ok (mnv4Blk7 N w x) := hx.g14b.2.1
  have ok9 : (mnv4BodyOfRow N mnv4Row9 w.b9).ok (mnv4Blk8 N w x) := hx.g14b.2.2.1
  have ok10 : (mnv4BodyOfRow N mnv4Row10 w.b10).ok (mnv4Blk9 N w x) := hx.g14b.2.2.2
  have ok11 : (mnv4StridedBodyOfRow N mnv4Row11 w.b11).ok (mnv4Blk10 N w x) := hx.g7a.1
  have ok12 : (mnv4BodyOfRow N mnv4Row12 w.b12).ok (mnv4Blk11 N w x) := hx.g7a.2.1
  have ok13 : (mnv4BodyOfRow N mnv4Row13 w.b13).ok (mnv4Blk12 N w x) := hx.g7a.2.2.1
  have ok14 : (mnv4BodyOfRow N mnv4Row14 w.b14).ok (mnv4Blk13 N w x) := hx.g7a.2.2.2.1
  have ok15 : (mnv4BodyOfRow N mnv4Row15 w.b15).ok (mnv4Blk14 N w x) := hx.g7a.2.2.2.2
  have ok16 : (mnv4BodyOfRow N mnv4Row16 w.b16).ok (mnv4Blk15 N w x) := hx.g7b.1
  have ok17 : (mnv4BodyOfRow N mnv4Row17 w.b17).ok (mnv4Blk16 N w x) := hx.g7b.2.1
  have ok18 : (mnv4BodyOfRow N mnv4Row18 w.b18).ok (mnv4Blk17 N w x) := hx.g7b.2.2.1
  have ok19 : (mnv4BodyOfRow N mnv4Row19 w.b19).ok (mnv4Blk18 N w x) := hx.g7b.2.2.2.1
  have ok20 : (mnv4BodyOfRow N mnv4Row20 w.b20).ok (mnv4Blk19 N w x) := hx.g7b.2.2.2.2.1
  have ok21 : (mnv4BodyOfRow N mnv4Row21 w.b21).ok (mnv4Blk20 N w x) := hx.g7b.2.2.2.2.2
  have h20 : HasGradAt (fun y => L (mnv4Suf20 N w y)) (mnv4Blk20 N w x) dy20 :=
    mnv4Skip_hasGradAt_comp (mnv4BodyOfRow N mnv4Row21 w.b21) _ ok21
      (mnv4BodyCotIn_eq_vjp N mnv4Row21 w.b21 _ _ ok21) h21
  have h19 : HasGradAt (fun y => L (mnv4Suf19 N w y)) (mnv4Blk19 N w x) dy19 :=
    mnv4Skip_hasGradAt_comp (mnv4BodyOfRow N mnv4Row20 w.b20) _ ok20
      (mnv4BodyCotIn_eq_vjp N mnv4Row20 w.b20 _ _ ok20) h20
  have h18 : HasGradAt (fun y => L (mnv4Suf18 N w y)) (mnv4Blk18 N w x) dy18 :=
    mnv4Skip_hasGradAt_comp (mnv4BodyOfRow N mnv4Row19 w.b19) _ ok19
      (mnv4BodyCotIn_eq_vjp N mnv4Row19 w.b19 _ _ ok19) h19
  have h17 : HasGradAt (fun y => L (mnv4Suf17 N w y)) (mnv4Blk17 N w x) dy17 :=
    mnv4Skip_hasGradAt_comp (mnv4BodyOfRow N mnv4Row18 w.b18) _ ok18
      (mnv4BodyCotIn_eq_vjp N mnv4Row18 w.b18 _ _ ok18) h18
  have h16 : HasGradAt (fun y => L (mnv4Suf16 N w y)) (mnv4Blk16 N w x) dy16 :=
    mnv4Skip_hasGradAt_comp (mnv4BodyOfRow N mnv4Row17 w.b17) _ ok17
      (mnv4BodyCotIn_eq_vjp N mnv4Row17 w.b17 _ _ ok17) h17
  have h15 : HasGradAt (fun y => L (mnv4Suf15 N w y)) (mnv4Blk15 N w x) dy15 :=
    mnv4Skip_hasGradAt_comp (mnv4BodyOfRow N mnv4Row16 w.b16) _ ok16
      (mnv4BodyCotIn_eq_vjp N mnv4Row16 w.b16 _ _ ok16) h16
  have h14 : HasGradAt (fun y => L (mnv4Suf14 N w y)) (mnv4Blk14 N w x) dy14 :=
    mnv4Skip_hasGradAt_comp (mnv4BodyOfRow N mnv4Row15 w.b15) _ ok15
      (mnv4BodyCotIn_eq_vjp N mnv4Row15 w.b15 _ _ ok15) h15
  have h13 : HasGradAt (fun y => L (mnv4Suf13 N w y)) (mnv4Blk13 N w x) dy13 :=
    mnv4Skip_hasGradAt_comp (mnv4BodyOfRow N mnv4Row14 w.b14) _ ok14
      (mnv4BodyCotIn_eq_vjp N mnv4Row14 w.b14 _ _ ok14) h14
  have h12 : HasGradAt (fun y => L (mnv4Suf12 N w y)) (mnv4Blk12 N w x) dy12 :=
    mnv4Skip_hasGradAt_comp (mnv4BodyOfRow N mnv4Row13 w.b13) _ ok13
      (mnv4BodyCotIn_eq_vjp N mnv4Row13 w.b13 _ _ ok13) h13
  have h11 : HasGradAt (fun y => L (mnv4Suf11 N w y)) (mnv4Blk11 N w x) dy11 :=
    mnv4Skip_hasGradAt_comp (mnv4BodyOfRow N mnv4Row12 w.b12) _ ok12
      (mnv4BodyCotIn_eq_vjp N mnv4Row12 w.b12 _ _ ok12) h12
  have h10 : HasGradAt (fun y => L (mnv4Suf10 N w y)) (mnv4Blk10 N w x) dy10 :=
    (certLayer_hasGradAt_comp (mnv4StridedBodyOfRow N mnv4Row11 w.b11) _ ok11 h11).of_eq
      (mnv4SBodyCotIn_eq_vjp N mnv4Row11 w.b11 _ _ ok11).symm
  have h9 : HasGradAt (fun y => L (mnv4Suf9 N w y)) (mnv4Blk9 N w x) dy9 :=
    mnv4Skip_hasGradAt_comp (mnv4BodyOfRow N mnv4Row10 w.b10) _ ok10
      (mnv4BodyCotIn_eq_vjp N mnv4Row10 w.b10 _ _ ok10) h10
  have h8 : HasGradAt (fun y => L (mnv4Suf8 N w y)) (mnv4Blk8 N w x) dy8 :=
    mnv4Skip_hasGradAt_comp (mnv4BodyOfRow N mnv4Row9 w.b9) _ ok9
      (mnv4BodyCotIn_eq_vjp N mnv4Row9 w.b9 _ _ ok9) h9
  have h7 : HasGradAt (fun y => L (mnv4Suf7 N w y)) (mnv4Blk7 N w x) dy7 :=
    mnv4Skip_hasGradAt_comp (mnv4BodyOfRow N mnv4Row8 w.b8) _ ok8
      (mnv4BodyCotIn_eq_vjp N mnv4Row8 w.b8 _ _ ok8) h8
  have h6 : HasGradAt (fun y => L (mnv4Suf6 N w y)) (mnv4Blk6 N w x) dy6 :=
    mnv4Skip_hasGradAt_comp (mnv4BodyOfRow N mnv4Row7 w.b7) _ ok7
      (mnv4BodyCotIn_eq_vjp N mnv4Row7 w.b7 _ _ ok7) h7
  have h5 : HasGradAt (fun y => L (mnv4Suf5 N w y)) (mnv4Blk5 N w x) dy5 :=
    mnv4Skip_hasGradAt_comp (mnv4BodyOfRow N mnv4Row6 w.b6) _ ok6
      (mnv4BodyCotIn_eq_vjp N mnv4Row6 w.b6 _ _ ok6) h6
  have h4 : HasGradAt (fun y => L (mnv4Suf4 N w y)) (mnv4Blk4 N w x) dy4 :=
    mnv4Skip_hasGradAt_comp (mnv4BodyOfRow N mnv4Row5 w.b5) _ ok5
      (mnv4BodyCotIn_eq_vjp N mnv4Row5 w.b5 _ _ ok5) h5
  have h3 : HasGradAt (fun y => L (mnv4Suf3 N w y)) (mnv4Blk3 N w x) dy3 :=
    mnv4Skip_hasGradAt_comp (mnv4BodyOfRow N mnv4Row4 w.b4) _ ok4
      (mnv4BodyCotIn_eq_vjp N mnv4Row4 w.b4 _ _ ok4) h4
  have h2 : HasGradAt (fun y => L (mnv4Suf2 N w y)) (mnv4Blk2 N w x) dy2 :=
    (certLayer_hasGradAt_comp (mnv4StridedBodyOfRow N mnv4Row3 w.b3) _ ok3 h3).of_eq
      (mnv4SBodyCotIn_eq_vjp N mnv4Row3 w.b3 _ _ ok3).symm
  have h1 : HasGradAt (fun y => L (mnv4Suf1 N w y)) (mnv4Blk1 N w x) dy1 :=
    mnv4Skip_hasGradAt_comp (mnv4BodyOfRow N mnv4Row2 w.b2) _ ok2
      (mnv4BodyCotIn_eq_vjp N mnv4Row2 w.b2 _ _ ok2) h2
  have h0 : HasGradAt (fun y => L (mnv4Suf0 N w y)) (mnv4Blk0 N w x) dy0 :=
    (certLayer_hasGradAt_comp (mnv4StridedBodyOfRow N mnv4Row1 w.b1) _ ok1 h1).of_eq
      (mnv4SBodyCotIn_eq_vjp N mnv4Row1 w.b1 _ _ ok1).symm
  have hs2 : ∀ k, bnBatchLA N 1280 1 1 w.hE w.hg w.hbt (batchMap N (flatConv w.hW w.hb)
      (mnv4HeadPool N 7 7 w.h1W w.h1b w.h1E w.h1g w.h1bt (mnv4Blk21 N w x))) k ≠ 0 := by
    have h := hh.2.2.2.1
    rw [castLayer_fwd_apply] at h
    exact h
  have hStem : HasGradAt (fun y => L (mnv4SufStem N w y)) (mnv4Pre0 N w x) dyStem :=
    (certLayer_hasGradAt_comp (mnv4FusedStack N w) _ hx.fused h0).of_eq
      (mnv4FusedCotIn_eq_vjp N 56 56 w.f0cW w.f0cb w.f0cE w.hf0cE w.f0cg w.f0cbt w.f0pW w.f0pb
        w.f0pE w.hf0pE w.f0pg w.f0pbt _ dy0 hx.fused).symm
  refine ⟨?cStem, ?cFused, ?c1, ?c2, ?c3, ?c4, ?c5, ?c6, ?c7, ?c8, ?c9, ?c10, ?c11, ?c12, ?c13, ?c14, ?c15, ?c16, ?c17, ?c18, ?c19, ?c20, ?c21, ?cHead⟩
  case cStem =>
    exact mnv4_stem_lossTiedB (N := N) (h := 112) (w := 112) (ic := 3) (oc := 32) xN cotN vN epsStr w.sW w.sb w.sE w.hsE w.sg w.sbt x hx.stem
      (hStem.congr_point (mnv4Pre0_eq N w x)) (fun W γ β => congrArg L (mnv4_factor_stem N w x W γ β))
  case cFused =>
    exact mnv4_fused_lossTiedB (N := N) (h := 56) (w := 56) (ic := 32) w.f0cW w.f0cb w.f0cE w.hf0cE w.f0cg w.f0cbt w.f0pW w.f0pb w.f0pE
      w.hf0pE w.f0pg w.f0pbt xN cotN vN epsStr (mnv4Pre0 N w x) hx.fused.1
      (h0.congr_point (mnv4Blk0_eq N w x))
      (fun Wc γc βc Wp γp βp => congrArg L (mnv4_factor_fused N w x Wc γc βc Wp γp βp))
  case c1 =>
    exact mnv4_strided_lossTiedB xN cotN vN epsStr w.b1 (by decide) (by decide) (mnv4Blk0 N w x) ok1
      (h1.congr_point (by rw [mnv4Blk1])) (fun p => congrArg L (mnv4_factor_b1 N w x p))
  case c2 =>
    exact mnv4_extradw_lossTiedB xN cotN vN epsStr w.b2 (by decide) (by decide) (mnv4Blk1 N w x) ok2
      (mnv4_residual_body_hasGradAt (mnv4BodyOfRow N mnv4Row2 w.b2) _ (h2.congr_point (by rw [mnv4Blk2]))) (fun p => congrArg L (mnv4_factor_b2 N w x p))
  case c3 =>
    exact mnv4_strided_lossTiedB xN cotN vN epsStr w.b3 (by decide) (by decide) (mnv4Blk2 N w x) ok3
      (h3.congr_point (by rw [mnv4Blk3])) (fun p => congrArg L (mnv4_factor_b3 N w x p))
  case c4 =>
    exact mnv4_extradw_lossTiedB xN cotN vN epsStr w.b4 (by decide) (by decide) (mnv4Blk3 N w x) ok4
      (mnv4_residual_body_hasGradAt (mnv4BodyOfRow N mnv4Row4 w.b4) _ (h4.congr_point (by rw [mnv4Blk4]))) (fun p => congrArg L (mnv4_factor_b4 N w x p))
  case c5 =>
    exact mnv4_extradw_lossTiedB xN cotN vN epsStr w.b5 (by decide) (by decide) (mnv4Blk4 N w x) ok5
      (mnv4_residual_body_hasGradAt (mnv4BodyOfRow N mnv4Row5 w.b5) _ (h5.congr_point (by rw [mnv4Blk5]))) (fun p => congrArg L (mnv4_factor_b5 N w x p))
  case c6 =>
    exact mnv4_extradw_lossTiedB xN cotN vN epsStr w.b6 (by decide) (by decide) (mnv4Blk5 N w x) ok6
      (mnv4_residual_body_hasGradAt (mnv4BodyOfRow N mnv4Row6 w.b6) _ (h6.congr_point (by rw [mnv4Blk6]))) (fun p => congrArg L (mnv4_factor_b6 N w x p))
  case c7 =>
    exact mnv4_extradw_lossTiedB xN cotN vN epsStr w.b7 (by decide) (by decide) (mnv4Blk6 N w x) ok7
      (mnv4_residual_body_hasGradAt (mnv4BodyOfRow N mnv4Row7 w.b7) _ (h7.congr_point (by rw [mnv4Blk7]))) (fun p => congrArg L (mnv4_factor_b7 N w x p))
  case c8 =>
    exact mnv4_convnext_lossTiedB xN cotN vN epsStr w.b8 (by decide) rfl (mnv4Blk7 N w x) ok8
      (mnv4_residual_body_hasGradAt (mnv4BodyOfRow N mnv4Row8 w.b8) _ (h8.congr_point (by rw [mnv4Blk8]))) (fun p => congrArg L (mnv4_factor_b8 N w x p))
  case c9 =>
    exact mnv4_ffn_lossTiedB xN cotN vN epsStr w.b9 rfl rfl (mnv4Blk8 N w x) ok9
      (mnv4_residual_body_hasGradAt (mnv4BodyOfRow N mnv4Row9 w.b9) _ (h9.congr_point (by rw [mnv4Blk9]))) (fun p => congrArg L (mnv4_factor_b9 N w x p))
  case c10 =>
    exact mnv4_convnext_lossTiedB xN cotN vN epsStr w.b10 (by decide) rfl (mnv4Blk9 N w x) ok10
      (mnv4_residual_body_hasGradAt (mnv4BodyOfRow N mnv4Row10 w.b10) _ (h10.congr_point (by rw [mnv4Blk10]))) (fun p => congrArg L (mnv4_factor_b10 N w x p))
  case c11 =>
    exact mnv4_strided_lossTiedB xN cotN vN epsStr w.b11 (by decide) (by decide) (mnv4Blk10 N w x) ok11
      (h11.congr_point (by rw [mnv4Blk11])) (fun p => congrArg L (mnv4_factor_b11 N w x p))
  case c12 =>
    exact mnv4_extradw_lossTiedB xN cotN vN epsStr w.b12 (by decide) (by decide) (mnv4Blk11 N w x) ok12
      (mnv4_residual_body_hasGradAt (mnv4BodyOfRow N mnv4Row12 w.b12) _ (h12.congr_point (by rw [mnv4Blk12]))) (fun p => congrArg L (mnv4_factor_b12 N w x p))
  case c13 =>
    exact mnv4_extradw_lossTiedB xN cotN vN epsStr w.b13 (by decide) (by decide) (mnv4Blk12 N w x) ok13
      (mnv4_residual_body_hasGradAt (mnv4BodyOfRow N mnv4Row13 w.b13) _ (h13.congr_point (by rw [mnv4Blk13]))) (fun p => congrArg L (mnv4_factor_b13 N w x p))
  case c14 =>
    exact mnv4_extradw_lossTiedB xN cotN vN epsStr w.b14 (by decide) (by decide) (mnv4Blk13 N w x) ok14
      (mnv4_residual_body_hasGradAt (mnv4BodyOfRow N mnv4Row14 w.b14) _ (h14.congr_point (by rw [mnv4Blk14]))) (fun p => congrArg L (mnv4_factor_b14 N w x p))
  case c15 =>
    exact mnv4_ffn_lossTiedB xN cotN vN epsStr w.b15 rfl rfl (mnv4Blk14 N w x) ok15
      (mnv4_residual_body_hasGradAt (mnv4BodyOfRow N mnv4Row15 w.b15) _ (h15.congr_point (by rw [mnv4Blk15]))) (fun p => congrArg L (mnv4_factor_b15 N w x p))
  case c16 =>
    exact mnv4_convnext_lossTiedB xN cotN vN epsStr w.b16 (by decide) rfl (mnv4Blk15 N w x) ok16
      (mnv4_residual_body_hasGradAt (mnv4BodyOfRow N mnv4Row16 w.b16) _ (h16.congr_point (by rw [mnv4Blk16]))) (fun p => congrArg L (mnv4_factor_b16 N w x p))
  case c17 =>
    exact mnv4_extradw_lossTiedB xN cotN vN epsStr w.b17 (by decide) (by decide) (mnv4Blk16 N w x) ok17
      (mnv4_residual_body_hasGradAt (mnv4BodyOfRow N mnv4Row17 w.b17) _ (h17.congr_point (by rw [mnv4Blk17]))) (fun p => congrArg L (mnv4_factor_b17 N w x p))
  case c18 =>
    exact mnv4_extradw_lossTiedB xN cotN vN epsStr w.b18 (by decide) (by decide) (mnv4Blk17 N w x) ok18
      (mnv4_residual_body_hasGradAt (mnv4BodyOfRow N mnv4Row18 w.b18) _ (h18.congr_point (by rw [mnv4Blk18]))) (fun p => congrArg L (mnv4_factor_b18 N w x p))
  case c19 =>
    exact mnv4_ffn_lossTiedB xN cotN vN epsStr w.b19 rfl rfl (mnv4Blk18 N w x) ok19
      (mnv4_residual_body_hasGradAt (mnv4BodyOfRow N mnv4Row19 w.b19) _ (h19.congr_point (by rw [mnv4Blk19]))) (fun p => congrArg L (mnv4_factor_b19 N w x p))
  case c20 =>
    exact mnv4_ffn_lossTiedB xN cotN vN epsStr w.b20 rfl rfl (mnv4Blk19 N w x) ok20
      (mnv4_residual_body_hasGradAt (mnv4BodyOfRow N mnv4Row20 w.b20) _ (h20.congr_point (by rw [mnv4Blk20]))) (fun p => congrArg L (mnv4_factor_b20 N w x p))
  case c21 =>
    exact mnv4_convnext_lossTiedB xN cotN vN epsStr w.b21 (by decide) rfl (mnv4Blk20 N w x) ok21
      (mnv4_residual_body_hasGradAt (mnv4BodyOfRow N mnv4Row21 w.b21) _ (h21.congr_point (by rw [mnv4Blk21]))) (fun p => congrArg L (mnv4_factor_b21 N w x p))
  case cHead =>
    exact mnv4_head_lossTiedB (N := N) (h := 7) (w := 7) (c := 256) w.h1W w.h1b w.h1E w.hh1E w.h1g w.h1bt w.hW w.hb w.hE w.hhE w.hg w.hbt
      w.Wd w.bd xN cotN vN epsStr (mnv4Blk21 N w x) hh.1 hs2
      (hL'.congr_point (mnv4Head_fwd _ _ _ _ _ _ _ _ _ _ _ _ _ _ _))
      (fun W1 γ1 β1 W2 γ2 β2 Wd bd => congrArg L (mnv4_factor_head N w x W1 γ1 β1 W2 γ2 β2 Wd bd))

/-- **The loss the artifacts ship**: every node is the derivative of the batched label-smoothed
    cross-entropy `smoothedBatchLoss`, `g` the six-op cotangent the render emits. -/
theorem mnv4_net_lossGrad_smoothedCE (N : Nat) {nCls : Nat} (hK : 0 < nCls)
    (xN cotN vN epsStr aStr negAK bStr logN ohN : String) (α B : ℝ) (w : Mnv4BWeights nCls)
    (x : Vec (N * (3 * 224 * 224))) (hx : Mnv4SmoothAt N w x) (t : Vec (N * (1 * nCls)))
    (ht : ∀ n, ∑ k : Fin nCls, targetRow N nCls t n k = 1) :
    Mnv4NetLossTiedB N xN cotN vN epsStr w x (smoothedBatchLoss N nCls α B t)
      (unrowB N nCls (den (smoothedLossCotGraph N nCls α B aStr negAK bStr logN ohN
        (rowB N nCls (mobilenetv4ForwardBFull N w x)) t))) :=
  mnv4_net_lossGrad N xN cotN vN epsStr w x hx
    ⟨(smoothedBatchLoss_differentiable N nCls α B t) _,
      fun J => smoothedBatchLoss_grad N nCls hK α B aStr negAK bStr logN ohN t _ ht J⟩


/-- **The emitted MobileNetV4 step's gradient nodes ARE the loss's gradient, at one chain.** For
    each of the 233 parameter slots, at ONE cotangent chain (the tie's own, from `g`): the node
    denotes its layer's Jacobian against the chain cotangent (`mnv4_net_tiedB`), and any loss `L` of
    the logits with gradient `g` at the network's output, read on `mobilenetv4ForwardBFull` with
    that one slot varied, is differentiable there with the node as its gradient
    (`mnv4_net_lossGrad`). The two theorems each state the chain; this one states it once, so an
    edit to either chain breaks its proof. -/
theorem mnv4_net_tied_lossGrad (N : Nat) {nCls : Nat} (xN cotN vN epsStr : String)
    (w : Mnv4BWeights nCls) (x : Vec (N * (3 * 224 * 224))) (g : Vec (N * nCls))
    (hx : Mnv4SmoothAt N w x) {L : Vec (N * nCls) → Vec 1}
    (hL : HasGradAt L (mobilenetv4ForwardBFull N w x) g) :
  let dy21 := mnv4HeadCotIn N 7 7 w.h1W w.h1b w.h1E w.h1g w.h1bt w.hW w.hb w.hE w.hg w.hbt
               w.Wd w.bd (mnv4Blk21 N w x) g
  let dy20 := mnv4SkipCotIn (mnv4BodyCotIn N mnv4Row21 w.b21 (mnv4Blk20 N w x) dy21) dy21
  let dy19 := mnv4SkipCotIn (mnv4BodyCotIn N mnv4Row20 w.b20 (mnv4Blk19 N w x) dy20) dy20
  let dy18 := mnv4SkipCotIn (mnv4BodyCotIn N mnv4Row19 w.b19 (mnv4Blk18 N w x) dy19) dy19
  let dy17 := mnv4SkipCotIn (mnv4BodyCotIn N mnv4Row18 w.b18 (mnv4Blk17 N w x) dy18) dy18
  let dy16 := mnv4SkipCotIn (mnv4BodyCotIn N mnv4Row17 w.b17 (mnv4Blk16 N w x) dy17) dy17
  let dy15 := mnv4SkipCotIn (mnv4BodyCotIn N mnv4Row16 w.b16 (mnv4Blk15 N w x) dy16) dy16
  let dy14 := mnv4SkipCotIn (mnv4BodyCotIn N mnv4Row15 w.b15 (mnv4Blk14 N w x) dy15) dy15
  let dy13 := mnv4SkipCotIn (mnv4BodyCotIn N mnv4Row14 w.b14 (mnv4Blk13 N w x) dy14) dy14
  let dy12 := mnv4SkipCotIn (mnv4BodyCotIn N mnv4Row13 w.b13 (mnv4Blk12 N w x) dy13) dy13
  let dy11 := mnv4SkipCotIn (mnv4BodyCotIn N mnv4Row12 w.b12 (mnv4Blk11 N w x) dy12) dy12
  let dy10 := mnv4SBodyCotIn N mnv4Row11 w.b11 (mnv4Blk10 N w x) dy11
  let dy9 := mnv4SkipCotIn (mnv4BodyCotIn N mnv4Row10 w.b10 (mnv4Blk9 N w x) dy10) dy10
  let dy8 := mnv4SkipCotIn (mnv4BodyCotIn N mnv4Row9 w.b9 (mnv4Blk8 N w x) dy9) dy9
  let dy7 := mnv4SkipCotIn (mnv4BodyCotIn N mnv4Row8 w.b8 (mnv4Blk7 N w x) dy8) dy8
  let dy6 := mnv4SkipCotIn (mnv4BodyCotIn N mnv4Row7 w.b7 (mnv4Blk6 N w x) dy7) dy7
  let dy5 := mnv4SkipCotIn (mnv4BodyCotIn N mnv4Row6 w.b6 (mnv4Blk5 N w x) dy6) dy6
  let dy4 := mnv4SkipCotIn (mnv4BodyCotIn N mnv4Row5 w.b5 (mnv4Blk4 N w x) dy5) dy5
  let dy3 := mnv4SkipCotIn (mnv4BodyCotIn N mnv4Row4 w.b4 (mnv4Blk3 N w x) dy4) dy4
  let dy2 := mnv4SBodyCotIn N mnv4Row3 w.b3 (mnv4Blk2 N w x) dy3
  let dy1 := mnv4SkipCotIn (mnv4BodyCotIn N mnv4Row2 w.b2 (mnv4Blk1 N w x) dy2) dy2
  let dy0 := mnv4SBodyCotIn N mnv4Row1 w.b1 (mnv4Blk0 N w x) dy1
  let dyStem := mnv4FusedCotIn N 56 56 w.f0cW w.f0cb w.f0cE w.f0cg w.f0cbt
                 w.f0pW w.f0pb w.f0pE w.f0pg w.f0pbt (mnv4Pre0 N w x) dy0
  (mnv4StemTiedB N 112 112 xN cotN vN epsStr w.sW w.sb w.sE w.sg w.sbt x dyStem
      ∧ mnv4StemLossTiedB (N := N) (h := 112) (w := 112) xN cotN vN epsStr w.sW w.sb w.sE w.sg w.sbt x
        (fun W γ β => L (mobilenetv4ForwardBFull N { w with sW := W, sg := γ, sbt := β } x)) dyStem)
  ∧ (mnv4FusedTiedB N 56 56 w.f0cW w.f0cb w.f0cE w.f0cg w.f0cbt w.f0pW w.f0pb w.f0pE w.f0pg
        w.f0pbt xN cotN vN epsStr (mnv4Pre0 N w x) dy0
      ∧ mnv4FusedLossTiedB (N := N) (h := 56) (w := 56) w.f0cW w.f0cb w.f0cE w.f0cg w.f0cbt
        w.f0pW w.f0pb w.f0pE w.f0pg w.f0pbt xN cotN vN epsStr (mnv4Pre0 N w x)
        (fun Wc γc βc Wp γp βp => L (mobilenetv4ForwardBFull N
        { w with f0cW := Wc, f0cg := γc, f0cbt := βc, f0pW := Wp, f0pg := γp, f0pbt := βp } x)) dy0)
  ∧ (mnv4StridedTiedB N mnv4Row1 xN cotN vN epsStr w.b1 (mnv4Blk0 N w x) dy1
      ∧ mnv4StridedLossTiedB N mnv4Row1 xN cotN vN epsStr w.b1 (mnv4Blk0 N w x)
        (fun p => L (mobilenetv4ForwardBFull N { w with b1 := p } x)) dy1)
  ∧ (mnv4ExtraDWTiedB N mnv4Row2 xN cotN vN epsStr w.b2 (mnv4Blk1 N w x) dy2
      ∧ mnv4ExtraDWLossTiedB N mnv4Row2 xN cotN vN epsStr w.b2 (mnv4Blk1 N w x)
        (fun p => L (mobilenetv4ForwardBFull N { w with b2 := p } x)) dy2)
  ∧ (mnv4StridedTiedB N mnv4Row3 xN cotN vN epsStr w.b3 (mnv4Blk2 N w x) dy3
      ∧ mnv4StridedLossTiedB N mnv4Row3 xN cotN vN epsStr w.b3 (mnv4Blk2 N w x)
        (fun p => L (mobilenetv4ForwardBFull N { w with b3 := p } x)) dy3)
  ∧ (mnv4ExtraDWTiedB N mnv4Row4 xN cotN vN epsStr w.b4 (mnv4Blk3 N w x) dy4
      ∧ mnv4ExtraDWLossTiedB N mnv4Row4 xN cotN vN epsStr w.b4 (mnv4Blk3 N w x)
        (fun p => L (mobilenetv4ForwardBFull N { w with b4 := p } x)) dy4)
  ∧ (mnv4ExtraDWTiedB N mnv4Row5 xN cotN vN epsStr w.b5 (mnv4Blk4 N w x) dy5
      ∧ mnv4ExtraDWLossTiedB N mnv4Row5 xN cotN vN epsStr w.b5 (mnv4Blk4 N w x)
        (fun p => L (mobilenetv4ForwardBFull N { w with b5 := p } x)) dy5)
  ∧ (mnv4ExtraDWTiedB N mnv4Row6 xN cotN vN epsStr w.b6 (mnv4Blk5 N w x) dy6
      ∧ mnv4ExtraDWLossTiedB N mnv4Row6 xN cotN vN epsStr w.b6 (mnv4Blk5 N w x)
        (fun p => L (mobilenetv4ForwardBFull N { w with b6 := p } x)) dy6)
  ∧ (mnv4ExtraDWTiedB N mnv4Row7 xN cotN vN epsStr w.b7 (mnv4Blk6 N w x) dy7
      ∧ mnv4ExtraDWLossTiedB N mnv4Row7 xN cotN vN epsStr w.b7 (mnv4Blk6 N w x)
        (fun p => L (mobilenetv4ForwardBFull N { w with b7 := p } x)) dy7)
  ∧ (mnv4ConvNeXtTiedB N mnv4Row8 xN cotN vN epsStr w.b8 (mnv4Blk7 N w x) dy8
      ∧ mnv4ConvNeXtLossTiedB N mnv4Row8 xN cotN vN epsStr w.b8 (mnv4Blk7 N w x)
        (fun p => L (mobilenetv4ForwardBFull N { w with b8 := p } x)) dy8)
  ∧ (mnv4FfnTiedB N mnv4Row9 xN cotN vN epsStr w.b9 (mnv4Blk8 N w x) dy9
      ∧ mnv4FfnLossTiedB N mnv4Row9 xN cotN vN epsStr w.b9 (mnv4Blk8 N w x)
        (fun p => L (mobilenetv4ForwardBFull N { w with b9 := p } x)) dy9)
  ∧ (mnv4ConvNeXtTiedB N mnv4Row10 xN cotN vN epsStr w.b10 (mnv4Blk9 N w x) dy10
      ∧ mnv4ConvNeXtLossTiedB N mnv4Row10 xN cotN vN epsStr w.b10 (mnv4Blk9 N w x)
        (fun p => L (mobilenetv4ForwardBFull N { w with b10 := p } x)) dy10)
  ∧ (mnv4StridedTiedB N mnv4Row11 xN cotN vN epsStr w.b11 (mnv4Blk10 N w x) dy11
      ∧ mnv4StridedLossTiedB N mnv4Row11 xN cotN vN epsStr w.b11 (mnv4Blk10 N w x)
        (fun p => L (mobilenetv4ForwardBFull N { w with b11 := p } x)) dy11)
  ∧ (mnv4ExtraDWTiedB N mnv4Row12 xN cotN vN epsStr w.b12 (mnv4Blk11 N w x) dy12
      ∧ mnv4ExtraDWLossTiedB N mnv4Row12 xN cotN vN epsStr w.b12 (mnv4Blk11 N w x)
        (fun p => L (mobilenetv4ForwardBFull N { w with b12 := p } x)) dy12)
  ∧ (mnv4ExtraDWTiedB N mnv4Row13 xN cotN vN epsStr w.b13 (mnv4Blk12 N w x) dy13
      ∧ mnv4ExtraDWLossTiedB N mnv4Row13 xN cotN vN epsStr w.b13 (mnv4Blk12 N w x)
        (fun p => L (mobilenetv4ForwardBFull N { w with b13 := p } x)) dy13)
  ∧ (mnv4ExtraDWTiedB N mnv4Row14 xN cotN vN epsStr w.b14 (mnv4Blk13 N w x) dy14
      ∧ mnv4ExtraDWLossTiedB N mnv4Row14 xN cotN vN epsStr w.b14 (mnv4Blk13 N w x)
        (fun p => L (mobilenetv4ForwardBFull N { w with b14 := p } x)) dy14)
  ∧ (mnv4FfnTiedB N mnv4Row15 xN cotN vN epsStr w.b15 (mnv4Blk14 N w x) dy15
      ∧ mnv4FfnLossTiedB N mnv4Row15 xN cotN vN epsStr w.b15 (mnv4Blk14 N w x)
        (fun p => L (mobilenetv4ForwardBFull N { w with b15 := p } x)) dy15)
  ∧ (mnv4ConvNeXtTiedB N mnv4Row16 xN cotN vN epsStr w.b16 (mnv4Blk15 N w x) dy16
      ∧ mnv4ConvNeXtLossTiedB N mnv4Row16 xN cotN vN epsStr w.b16 (mnv4Blk15 N w x)
        (fun p => L (mobilenetv4ForwardBFull N { w with b16 := p } x)) dy16)
  ∧ (mnv4ExtraDWTiedB N mnv4Row17 xN cotN vN epsStr w.b17 (mnv4Blk16 N w x) dy17
      ∧ mnv4ExtraDWLossTiedB N mnv4Row17 xN cotN vN epsStr w.b17 (mnv4Blk16 N w x)
        (fun p => L (mobilenetv4ForwardBFull N { w with b17 := p } x)) dy17)
  ∧ (mnv4ExtraDWTiedB N mnv4Row18 xN cotN vN epsStr w.b18 (mnv4Blk17 N w x) dy18
      ∧ mnv4ExtraDWLossTiedB N mnv4Row18 xN cotN vN epsStr w.b18 (mnv4Blk17 N w x)
        (fun p => L (mobilenetv4ForwardBFull N { w with b18 := p } x)) dy18)
  ∧ (mnv4FfnTiedB N mnv4Row19 xN cotN vN epsStr w.b19 (mnv4Blk18 N w x) dy19
      ∧ mnv4FfnLossTiedB N mnv4Row19 xN cotN vN epsStr w.b19 (mnv4Blk18 N w x)
        (fun p => L (mobilenetv4ForwardBFull N { w with b19 := p } x)) dy19)
  ∧ (mnv4FfnTiedB N mnv4Row20 xN cotN vN epsStr w.b20 (mnv4Blk19 N w x) dy20
      ∧ mnv4FfnLossTiedB N mnv4Row20 xN cotN vN epsStr w.b20 (mnv4Blk19 N w x)
        (fun p => L (mobilenetv4ForwardBFull N { w with b20 := p } x)) dy20)
  ∧ (mnv4ConvNeXtTiedB N mnv4Row21 xN cotN vN epsStr w.b21 (mnv4Blk20 N w x) dy21
      ∧ mnv4ConvNeXtLossTiedB N mnv4Row21 xN cotN vN epsStr w.b21 (mnv4Blk20 N w x)
        (fun p => L (mobilenetv4ForwardBFull N { w with b21 := p } x)) dy21)
  ∧ (mnv4HeadTiedB N 7 7 w.h1W w.h1b w.h1E w.h1g w.h1bt w.hW w.hb w.hE w.hg w.hbt
        w.Wd w.bd xN cotN vN epsStr (mnv4Blk21 N w x) g
      ∧ mnv4HeadLossTiedB (N := N) (h := 7) (w := 7) w.h1W w.h1b w.h1E w.h1g w.h1bt w.hW w.hb w.hE w.hg
        w.hbt w.Wd w.bd xN cotN vN epsStr (mnv4Blk21 N w x)
        (fun W1 γ1 β1 W2 γ2 β2 Wd bd => L (mobilenetv4ForwardBFull N
        { w with h1W := W1, h1g := γ1, h1bt := β1, hW := W2, hg := γ2, hbt := β2, Wd := Wd, bd := bd } x))
        g) := by
  intro dy21 dy20 dy19 dy18 dy17 dy16 dy15 dy14 dy13 dy12 dy11 dy10 dy9 dy8 dy7 dy6 dy5 dy4 dy3 dy2 dy1 dy0 dyStem
  obtain ⟨t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19,
    t20, t21, t22, t23⟩ :=
    mnv4_net_tiedB N xN cotN vN epsStr w x g
  have hl :=
    mnv4_net_lossGrad N xN cotN vN epsStr w x hx hL
  obtain ⟨l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15, l16, l17, l18, l19,
    l20, l21, l22, l23⟩ := hl
  exact ⟨⟨t0, l0⟩, ⟨t1, l1⟩, ⟨t2, l2⟩, ⟨t3, l3⟩, ⟨t4, l4⟩, ⟨t5, l5⟩, ⟨t6, l6⟩, ⟨t7, l7⟩, ⟨t8, l8⟩,
    ⟨t9, l9⟩, ⟨t10, l10⟩, ⟨t11, l11⟩, ⟨t12, l12⟩, ⟨t13, l13⟩, ⟨t14, l14⟩, ⟨t15, l15⟩, ⟨t16, l16⟩,
    ⟨t17, l17⟩, ⟨t18, l18⟩, ⟨t19, l19⟩, ⟨t20, l20⟩, ⟨t21, l21⟩, ⟨t22, l22⟩, ⟨t23, l23⟩⟩

end Proofs.Mnv4TieB

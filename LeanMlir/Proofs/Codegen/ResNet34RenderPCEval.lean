import LeanMlir.Proofs.Codegen.ResNet34RenderPC

/-! # r34 — the INFERENCE forward graph, and its faithfulness

The eval twin of `ResNet34RenderPC.lean`'s part 2. That file proves
`den (resnet34FwdGraphFullPC …) = resnet34Forward_full_pc …` — the whole `[3,4,6,3]` net's
typed `SHlo` graph denotes the certified ℝ forward — for the **training** BN chain, the one
`resnet34_train_step.mlir` differentiates. `@resnet34_fwd_eval` renders the other chain (every
BN site reading frozen running statistics through `.bnPerChannelEvalF`) and had no such
theorem, so the eval forward was rendered from the verified AST but not tied to an ℝ def.

This file closes that. Same three rungs as the training pair, one for one:
`idBlockGraphPCEval` / `downBlockGraphPCEval` + their `_faithful` lemmas, then
`resnet34FwdGraphFullPCEval` + `resnet34FwdGraphFullPCEval_faithful` against
`resnet34Forward_full_pc_eval`. Every BN node's `den` is `bnPerChannelEvalTensor3`
(`bnPerChannelEvalF_faithful`, `rfl`), so nothing new is assumed — the eval BN op was already
proven, it just had no whole-net chain to sit in.

⭐ Why it matters here and not only for tidiness: `Resnet34FloatBudget.lean` states the repo's
first ImageNet-scale float number about exactly this net, and until now had to disclose the tie
as skeleton-level (`formalization.yaml` §4d). `r34EvalForward_eq_full_pc_eval` there is now a
`rfl` onto `resnet34Forward_full_pc_eval`, and this file's faithfulness carries it the rest of
the way to the graph.

The SSA names mirror `ResNet34Render.lean`'s `bnSite` convention (`%{p}g1`/`%{p}bt1` with the
frozen statistics at `%{p}n1mu`/`%{p}n1var`, and `stn` for the stem), so the typed graph and the
emitted text name the same inputs. Names are pretty-printing metadata and do not enter `den`.
No new tokens. 3-axiom clean.
-/

namespace Proofs

-- ════════════════════════════════════════════════════════════════
-- § The inference building-block ℝ-forwards (eval mirrors of cbrStridedPC / rblkPC / rblkPStridedPC)
-- ════════════════════════════════════════════════════════════════

/-- 7×7 strided stem conv → inference BN → relu. -/
@[reducible] noncomputable def cbrStridedPCEval {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (γ β μ v : Vec oc) :
    Vec (ic * (2 * h) * (2 * w)) → Vec (oc * h * w) :=
  relu (oc * h * w) ∘ bnPerChannelEvalTensor3 oc h w ε γ β μ v ∘ flatConvStride2 W b

/-- Identity basic block at inference: `rblkPC` with frozen running statistics. -/
@[reducible] noncomputable def rblkPCEval {c h w kH₁ kW₁ kH₂ kW₂ : Nat}
    (W₁ : Kernel4 c c kH₁ kW₁) (b₁ : Vec c) (ε₁ : ℝ) (γ₁ β₁ μ₁ v₁ : Vec c)
    (W₂ : Kernel4 c c kH₂ kW₂) (b₂ : Vec c) (ε₂ : ℝ) (γ₂ β₂ μ₂ v₂ : Vec c) :
    Vec (c * h * w) → Vec (c * h * w) :=
  relu (c * h * w) ∘ residual
    ((bnPerChannelEvalTensor3 c h w ε₂ γ₂ β₂ μ₂ v₂ ∘ flatConv W₂ b₂) ∘
      (relu (c * h * w) ∘ bnPerChannelEvalTensor3 c h w ε₁ γ₁ β₁ μ₁ v₁ ∘ flatConv W₁ b₁))

/-- Downsample basic block at inference: `rblkPStridedPC` with frozen running statistics. -/
@[reducible] noncomputable def rblkPStridedPCEval {ic oc h w kH₁ kW₁ kH₂ kW₂ kHp kWp : Nat}
    (W₁ : Kernel4 oc ic kH₁ kW₁) (b₁ : Vec oc) (ε₁ : ℝ) (γ₁ β₁ μ₁ v₁ : Vec oc)
    (W₂ : Kernel4 oc oc kH₂ kW₂) (b₂ : Vec oc) (ε₂ : ℝ) (γ₂ β₂ μ₂ v₂ : Vec oc)
    (Wp : Kernel4 oc ic kHp kWp) (bp : Vec oc) (εp : ℝ) (γp βp μp vp : Vec oc) :
    Vec (ic * (2 * h) * (2 * w)) → Vec (oc * h * w) :=
  relu (oc * h * w) ∘ residualProj
    (bnPerChannelEvalTensor3 oc h w εp γp βp μp vp ∘ flatConvStride2 Wp bp)
    ((bnPerChannelEvalTensor3 oc h w ε₂ γ₂ β₂ μ₂ v₂ ∘ flatConv W₂ b₂) ∘
      (relu (oc * h * w) ∘ bnPerChannelEvalTensor3 oc h w ε₁ γ₁ β₁ μ₁ v₁ ∘ flatConvStride2 W₁ b₁))

/-- Identity-block inference ℝ-forward at shared ε (the partial application the net takes). -/
@[reducible] noncomputable def idFwdEval {c h w : Nat} (ε : ℝ)
    (W₁ : Kernel4 c c 3 3) (b₁ : Vec c) (γ₁ β₁ μ₁ v₁ : Vec c)
    (W₂ : Kernel4 c c 3 3) (b₂ : Vec c) (γ₂ β₂ μ₂ v₂ : Vec c) : Vec (c*h*w) → Vec (c*h*w) :=
  rblkPCEval (h := h) (w := w) W₁ b₁ ε γ₁ β₁ μ₁ v₁ W₂ b₂ ε γ₂ β₂ μ₂ v₂

/-- Downsample-block inference ℝ-forward at shared ε. -/
@[reducible] noncomputable def downFwdEval {ic oc h w : Nat} (ε : ℝ)
    (W₁ : Kernel4 oc ic 3 3) (b₁ : Vec oc) (γ₁ β₁ μ₁ v₁ : Vec oc)
    (W₂ : Kernel4 oc oc 3 3) (b₂ : Vec oc) (γ₂ β₂ μ₂ v₂ : Vec oc)
    (Wp : Kernel4 oc ic 1 1) (bp : Vec oc) (γp βp μp vp : Vec oc) :
    Vec (ic*(2*h)*(2*w)) → Vec (oc*h*w) :=
  rblkPStridedPCEval (h := h) (w := w) W₁ b₁ ε γ₁ β₁ μ₁ v₁ W₂ b₂ ε γ₂ β₂ μ₂ v₂
    Wp bp ε γp βp μp vp

/-- **The full inference ResNet-34 forward** (render dims 3×224² → 7×7×512) — the ℝ map
    `@resnet34_fwd_eval` computes, and the map `Resnet34FloatBudget.lean` states its number
    about. The eval twin of `resnet34Forward_full_pc`, argument for argument, plus each BN
    site's frozen running mean and variance. -/
noncomputable def resnet34Forward_full_pc_eval (ε : ℝ)
    (Ws : Kernel4 64 3 7 7) (bs : Vec 64) (γs βs μs vs : Vec 64)
    (a0W1 : Kernel4 64 64 3 3) (a0b1 : Vec 64) (a0g1 a0t1 a0m1 a0v1 : Vec 64) (a0W2 : Kernel4 64 64 3 3) (a0b2 : Vec 64) (a0g2 a0t2 a0m2 a0v2 : Vec 64)
    (a1W1 : Kernel4 64 64 3 3) (a1b1 : Vec 64) (a1g1 a1t1 a1m1 a1v1 : Vec 64) (a1W2 : Kernel4 64 64 3 3) (a1b2 : Vec 64) (a1g2 a1t2 a1m2 a1v2 : Vec 64)
    (a2W1 : Kernel4 64 64 3 3) (a2b1 : Vec 64) (a2g1 a2t1 a2m1 a2v1 : Vec 64) (a2W2 : Kernel4 64 64 3 3) (a2b2 : Vec 64) (a2g2 a2t2 a2m2 a2v2 : Vec 64)
    (d2W1 : Kernel4 128 64 3 3) (d2b1 : Vec 128) (d2g1 d2t1 d2m1 d2v1 : Vec 128) (d2W2 : Kernel4 128 128 3 3) (d2b2 : Vec 128) (d2g2 d2t2 d2m2 d2v2 : Vec 128) (d2Wp : Kernel4 128 64 1 1) (d2bp : Vec 128) (d2gp d2tp d2mp d2vp : Vec 128)
    (b0W1 : Kernel4 128 128 3 3) (b0b1 : Vec 128) (b0g1 b0t1 b0m1 b0v1 : Vec 128) (b0W2 : Kernel4 128 128 3 3) (b0b2 : Vec 128) (b0g2 b0t2 b0m2 b0v2 : Vec 128)
    (b1W1 : Kernel4 128 128 3 3) (b1b1 : Vec 128) (b1g1 b1t1 b1m1 b1v1 : Vec 128) (b1W2 : Kernel4 128 128 3 3) (b1b2 : Vec 128) (b1g2 b1t2 b1m2 b1v2 : Vec 128)
    (b2W1 : Kernel4 128 128 3 3) (b2b1 : Vec 128) (b2g1 b2t1 b2m1 b2v1 : Vec 128) (b2W2 : Kernel4 128 128 3 3) (b2b2 : Vec 128) (b2g2 b2t2 b2m2 b2v2 : Vec 128)
    (d3W1 : Kernel4 256 128 3 3) (d3b1 : Vec 256) (d3g1 d3t1 d3m1 d3v1 : Vec 256) (d3W2 : Kernel4 256 256 3 3) (d3b2 : Vec 256) (d3g2 d3t2 d3m2 d3v2 : Vec 256) (d3Wp : Kernel4 256 128 1 1) (d3bp : Vec 256) (d3gp d3tp d3mp d3vp : Vec 256)
    (c0W1 : Kernel4 256 256 3 3) (c0b1 : Vec 256) (c0g1 c0t1 c0m1 c0v1 : Vec 256) (c0W2 : Kernel4 256 256 3 3) (c0b2 : Vec 256) (c0g2 c0t2 c0m2 c0v2 : Vec 256)
    (c1W1 : Kernel4 256 256 3 3) (c1b1 : Vec 256) (c1g1 c1t1 c1m1 c1v1 : Vec 256) (c1W2 : Kernel4 256 256 3 3) (c1b2 : Vec 256) (c1g2 c1t2 c1m2 c1v2 : Vec 256)
    (c2W1 : Kernel4 256 256 3 3) (c2b1 : Vec 256) (c2g1 c2t1 c2m1 c2v1 : Vec 256) (c2W2 : Kernel4 256 256 3 3) (c2b2 : Vec 256) (c2g2 c2t2 c2m2 c2v2 : Vec 256)
    (c3W1 : Kernel4 256 256 3 3) (c3b1 : Vec 256) (c3g1 c3t1 c3m1 c3v1 : Vec 256) (c3W2 : Kernel4 256 256 3 3) (c3b2 : Vec 256) (c3g2 c3t2 c3m2 c3v2 : Vec 256)
    (c4W1 : Kernel4 256 256 3 3) (c4b1 : Vec 256) (c4g1 c4t1 c4m1 c4v1 : Vec 256) (c4W2 : Kernel4 256 256 3 3) (c4b2 : Vec 256) (c4g2 c4t2 c4m2 c4v2 : Vec 256)
    (d4W1 : Kernel4 512 256 3 3) (d4b1 : Vec 512) (d4g1 d4t1 d4m1 d4v1 : Vec 512) (d4W2 : Kernel4 512 512 3 3) (d4b2 : Vec 512) (d4g2 d4t2 d4m2 d4v2 : Vec 512) (d4Wp : Kernel4 512 256 1 1) (d4bp : Vec 512) (d4gp d4tp d4mp d4vp : Vec 512)
    (e0W1 : Kernel4 512 512 3 3) (e0b1 : Vec 512) (e0g1 e0t1 e0m1 e0v1 : Vec 512) (e0W2 : Kernel4 512 512 3 3) (e0b2 : Vec 512) (e0g2 e0t2 e0m2 e0v2 : Vec 512)
    (e1W1 : Kernel4 512 512 3 3) (e1b1 : Vec 512) (e1g1 e1t1 e1m1 e1v1 : Vec 512) (e1W2 : Kernel4 512 512 3 3) (e1b2 : Vec 512) (e1g2 e1t2 e1m2 e1v2 : Vec 512)
    (Wd : Mat 512 10) (bd : Vec 10) :
    Vec (3 * 224 * 224) → Vec 10 :=
  dense Wd bd ∘ globalAvgPoolFlat 512 7 7 ∘
  idFwdEval (h := 7) (w := 7) ε e1W1 e1b1 e1g1 e1t1 e1m1 e1v1 e1W2 e1b2 e1g2 e1t2 e1m2 e1v2 ∘
  idFwdEval (h := 7) (w := 7) ε e0W1 e0b1 e0g1 e0t1 e0m1 e0v1 e0W2 e0b2 e0g2 e0t2 e0m2 e0v2 ∘
  downFwdEval (h := 7) (w := 7) ε d4W1 d4b1 d4g1 d4t1 d4m1 d4v1 d4W2 d4b2 d4g2 d4t2 d4m2 d4v2 d4Wp d4bp d4gp d4tp d4mp d4vp ∘
  idFwdEval (h := 14) (w := 14) ε c4W1 c4b1 c4g1 c4t1 c4m1 c4v1 c4W2 c4b2 c4g2 c4t2 c4m2 c4v2 ∘
  idFwdEval (h := 14) (w := 14) ε c3W1 c3b1 c3g1 c3t1 c3m1 c3v1 c3W2 c3b2 c3g2 c3t2 c3m2 c3v2 ∘
  idFwdEval (h := 14) (w := 14) ε c2W1 c2b1 c2g1 c2t1 c2m1 c2v1 c2W2 c2b2 c2g2 c2t2 c2m2 c2v2 ∘
  idFwdEval (h := 14) (w := 14) ε c1W1 c1b1 c1g1 c1t1 c1m1 c1v1 c1W2 c1b2 c1g2 c1t2 c1m2 c1v2 ∘
  idFwdEval (h := 14) (w := 14) ε c0W1 c0b1 c0g1 c0t1 c0m1 c0v1 c0W2 c0b2 c0g2 c0t2 c0m2 c0v2 ∘
  downFwdEval (h := 14) (w := 14) ε d3W1 d3b1 d3g1 d3t1 d3m1 d3v1 d3W2 d3b2 d3g2 d3t2 d3m2 d3v2 d3Wp d3bp d3gp d3tp d3mp d3vp ∘
  idFwdEval (h := 28) (w := 28) ε b2W1 b2b1 b2g1 b2t1 b2m1 b2v1 b2W2 b2b2 b2g2 b2t2 b2m2 b2v2 ∘
  idFwdEval (h := 28) (w := 28) ε b1W1 b1b1 b1g1 b1t1 b1m1 b1v1 b1W2 b1b2 b1g2 b1t2 b1m2 b1v2 ∘
  idFwdEval (h := 28) (w := 28) ε b0W1 b0b1 b0g1 b0t1 b0m1 b0v1 b0W2 b0b2 b0g2 b0t2 b0m2 b0v2 ∘
  downFwdEval (h := 28) (w := 28) ε d2W1 d2b1 d2g1 d2t1 d2m1 d2v1 d2W2 d2b2 d2g2 d2t2 d2m2 d2v2 d2Wp d2bp d2gp d2tp d2mp d2vp ∘
  idFwdEval (h := 56) (w := 56) ε a2W1 a2b1 a2g1 a2t1 a2m1 a2v1 a2W2 a2b2 a2g2 a2t2 a2m2 a2v2 ∘
  idFwdEval (h := 56) (w := 56) ε a1W1 a1b1 a1g1 a1t1 a1m1 a1v1 a1W2 a1b2 a1g2 a1t2 a1m2 a1v2 ∘
  idFwdEval (h := 56) (w := 56) ε a0W1 a0b1 a0g1 a0t1 a0m1 a0v1 a0W2 a0b2 a0g2 a0t2 a0m2 a0v2 ∘
  maxPool3s2Flat 64 56 56 ∘
  cbrStridedPCEval (h := 112) (w := 112) Ws bs ε γs βs μs vs

namespace StableHLO

-- ════════════════════════════════════════════════════════════════
-- § Per-block inference `SHlo` graphs + faithfulness
-- ════════════════════════════════════════════════════════════════

/-- Identity-block inference graph: `relu(addV(bnEval∘conv∘relu∘bnEval∘conv, skip))`; the skip
    reuses the block-input subtree `e`, as in the training twin. -/
def idBlockGraphPCEval (p epsStr : String) {c h w : Nat}
    (W₁ : Kernel4 c c 3 3) (b₁ : Vec c) (ε₁ : ℝ) (γ₁ β₁ μ₁ v₁ : Vec c)
    (W₂ : Kernel4 c c 3 3) (b₂ : Vec c) (ε₂ : ℝ) (γ₂ β₂ μ₂ v₂ : Vec c)
    (e : SHlo (c * h * w)) : SHlo (c * h * w) :=
  .reluF (.addV
    (.bnPerChannelEvalF (oc := c) (h := h) (w := w) s!"%{p}g2" s!"%{p}bt2"
      s!"%{p}n2mu" s!"%{p}n2var" epsStr ε₂ γ₂ β₂ μ₂ v₂
      (.flatConvF (h := h) (w := w) s!"%{p}W2" s!"%{p}b2" W₂ b₂
        (.reluF (.bnPerChannelEvalF (oc := c) (h := h) (w := w) s!"%{p}g1" s!"%{p}bt1"
          s!"%{p}n1mu" s!"%{p}n1var" epsStr ε₁ γ₁ β₁ μ₁ v₁
          (.flatConvF (h := h) (w := w) s!"%{p}W1" s!"%{p}b1" W₁ b₁ e)))))
    e)

/-- Downsample-block inference graph: `relu(addV(projection, body))`, both reading `e`. -/
def downBlockGraphPCEval (p epsStr : String) {ic oc h w kHp kWp : Nat}
    (W₁ : Kernel4 oc ic 3 3) (b₁ : Vec oc) (ε₁ : ℝ) (γ₁ β₁ μ₁ v₁ : Vec oc)
    (W₂ : Kernel4 oc oc 3 3) (b₂ : Vec oc) (ε₂ : ℝ) (γ₂ β₂ μ₂ v₂ : Vec oc)
    (Wp : Kernel4 oc ic kHp kWp) (bp : Vec oc) (εp : ℝ) (γp βp μp vp : Vec oc)
    (e : SHlo (ic * (2 * h) * (2 * w))) : SHlo (oc * h * w) :=
  .reluF (.addV
    (.bnPerChannelEvalF (oc := oc) (h := h) (w := w) s!"%{p}gp" s!"%{p}btp"
      s!"%{p}npmu" s!"%{p}npvar" epsStr εp γp βp μp vp
      (.flatConvStridedF (h := h) (w := w) s!"%{p}Wp" s!"%{p}bp" Wp bp e))
    (.bnPerChannelEvalF (oc := oc) (h := h) (w := w) s!"%{p}g2" s!"%{p}bt2"
      s!"%{p}n2mu" s!"%{p}n2var" epsStr ε₂ γ₂ β₂ μ₂ v₂
      (.flatConvF (h := h) (w := w) s!"%{p}W2" s!"%{p}b2" W₂ b₂
        (.reluF (.bnPerChannelEvalF (oc := oc) (h := h) (w := w) s!"%{p}g1" s!"%{p}bt1"
          s!"%{p}n1mu" s!"%{p}n1var" epsStr ε₁ γ₁ β₁ μ₁ v₁
          (.flatConvStridedF (h := h) (w := w) s!"%{p}W1" s!"%{p}b1" W₁ b₁ e))))))

/-- **Identity block inference faithfulness.** `den (idBlockGraphPCEval … e) = rblkPCEval … (den e)`. -/
theorem idBlockGraphPCEval_faithful (p epsStr : String) {c h w : Nat}
    (W₁ : Kernel4 c c 3 3) (b₁ : Vec c) (ε₁ : ℝ) (γ₁ β₁ μ₁ v₁ : Vec c)
    (W₂ : Kernel4 c c 3 3) (b₂ : Vec c) (ε₂ : ℝ) (γ₂ β₂ μ₂ v₂ : Vec c)
    (e : SHlo (c * h * w)) :
    den (idBlockGraphPCEval p epsStr W₁ b₁ ε₁ γ₁ β₁ μ₁ v₁ W₂ b₂ ε₂ γ₂ β₂ μ₂ v₂ e)
      = rblkPCEval W₁ b₁ ε₁ γ₁ β₁ μ₁ v₁ W₂ b₂ ε₂ γ₂ β₂ μ₂ v₂ (den e) := by
  simp only [idBlockGraphPCEval, reluF_faithful, bnPerChannelEvalF_faithful, flatConvF_faithful,
             den_addV]
  unfold rblkPCEval residual biPath
  simp only [Function.comp_apply]

/-- **Downsample block inference faithfulness.** -/
theorem downBlockGraphPCEval_faithful (p epsStr : String) {ic oc h w kHp kWp : Nat}
    (W₁ : Kernel4 oc ic 3 3) (b₁ : Vec oc) (ε₁ : ℝ) (γ₁ β₁ μ₁ v₁ : Vec oc)
    (W₂ : Kernel4 oc oc 3 3) (b₂ : Vec oc) (ε₂ : ℝ) (γ₂ β₂ μ₂ v₂ : Vec oc)
    (Wp : Kernel4 oc ic kHp kWp) (bp : Vec oc) (εp : ℝ) (γp βp μp vp : Vec oc)
    (e : SHlo (ic * (2 * h) * (2 * w))) :
    den (downBlockGraphPCEval p epsStr W₁ b₁ ε₁ γ₁ β₁ μ₁ v₁ W₂ b₂ ε₂ γ₂ β₂ μ₂ v₂
          Wp bp εp γp βp μp vp e)
      = rblkPStridedPCEval W₁ b₁ ε₁ γ₁ β₁ μ₁ v₁ W₂ b₂ ε₂ γ₂ β₂ μ₂ v₂ Wp bp εp γp βp μp vp
          (den e) := by
  simp only [downBlockGraphPCEval, reluF_faithful, bnPerChannelEvalF_faithful, flatConvF_faithful,
             flatConvStridedF_faithful, den_addV]
  unfold rblkPStridedPCEval residualProj biPath
  simp only [Function.comp_apply]

-- ════════════════════════════════════════════════════════════════
-- § The full inference ResNet-34 typed `SHlo` forward graph + faithfulness
-- ════════════════════════════════════════════════════════════════

/-- Whole **inference ResNet-34 forward** graph at the render dims (3×224² → 7×7×512): 7×7
    strided stem → eval BN → relu → `maxPoolF` → `[3,4,6,3]` basic blocks → GAP → dense, with
    every BN site reading frozen running statistics. The eval twin of
    `resnet34FwdGraphFullPC`. -/
def resnet34FwdGraphFullPCEval (epsStr : String) (ε : ℝ)
    (Ws : Kernel4 64 3 7 7) (bs : Vec 64) (γs βs μs vs : Vec 64)
    (a0W1 : Kernel4 64 64 3 3) (a0b1 : Vec 64) (a0g1 a0t1 a0m1 a0v1 : Vec 64) (a0W2 : Kernel4 64 64 3 3) (a0b2 : Vec 64) (a0g2 a0t2 a0m2 a0v2 : Vec 64)
    (a1W1 : Kernel4 64 64 3 3) (a1b1 : Vec 64) (a1g1 a1t1 a1m1 a1v1 : Vec 64) (a1W2 : Kernel4 64 64 3 3) (a1b2 : Vec 64) (a1g2 a1t2 a1m2 a1v2 : Vec 64)
    (a2W1 : Kernel4 64 64 3 3) (a2b1 : Vec 64) (a2g1 a2t1 a2m1 a2v1 : Vec 64) (a2W2 : Kernel4 64 64 3 3) (a2b2 : Vec 64) (a2g2 a2t2 a2m2 a2v2 : Vec 64)
    (d2W1 : Kernel4 128 64 3 3) (d2b1 : Vec 128) (d2g1 d2t1 d2m1 d2v1 : Vec 128) (d2W2 : Kernel4 128 128 3 3) (d2b2 : Vec 128) (d2g2 d2t2 d2m2 d2v2 : Vec 128) (d2Wp : Kernel4 128 64 1 1) (d2bp : Vec 128) (d2gp d2tp d2mp d2vp : Vec 128)
    (b0W1 : Kernel4 128 128 3 3) (b0b1 : Vec 128) (b0g1 b0t1 b0m1 b0v1 : Vec 128) (b0W2 : Kernel4 128 128 3 3) (b0b2 : Vec 128) (b0g2 b0t2 b0m2 b0v2 : Vec 128)
    (b1W1 : Kernel4 128 128 3 3) (b1b1 : Vec 128) (b1g1 b1t1 b1m1 b1v1 : Vec 128) (b1W2 : Kernel4 128 128 3 3) (b1b2 : Vec 128) (b1g2 b1t2 b1m2 b1v2 : Vec 128)
    (b2W1 : Kernel4 128 128 3 3) (b2b1 : Vec 128) (b2g1 b2t1 b2m1 b2v1 : Vec 128) (b2W2 : Kernel4 128 128 3 3) (b2b2 : Vec 128) (b2g2 b2t2 b2m2 b2v2 : Vec 128)
    (d3W1 : Kernel4 256 128 3 3) (d3b1 : Vec 256) (d3g1 d3t1 d3m1 d3v1 : Vec 256) (d3W2 : Kernel4 256 256 3 3) (d3b2 : Vec 256) (d3g2 d3t2 d3m2 d3v2 : Vec 256) (d3Wp : Kernel4 256 128 1 1) (d3bp : Vec 256) (d3gp d3tp d3mp d3vp : Vec 256)
    (c0W1 : Kernel4 256 256 3 3) (c0b1 : Vec 256) (c0g1 c0t1 c0m1 c0v1 : Vec 256) (c0W2 : Kernel4 256 256 3 3) (c0b2 : Vec 256) (c0g2 c0t2 c0m2 c0v2 : Vec 256)
    (c1W1 : Kernel4 256 256 3 3) (c1b1 : Vec 256) (c1g1 c1t1 c1m1 c1v1 : Vec 256) (c1W2 : Kernel4 256 256 3 3) (c1b2 : Vec 256) (c1g2 c1t2 c1m2 c1v2 : Vec 256)
    (c2W1 : Kernel4 256 256 3 3) (c2b1 : Vec 256) (c2g1 c2t1 c2m1 c2v1 : Vec 256) (c2W2 : Kernel4 256 256 3 3) (c2b2 : Vec 256) (c2g2 c2t2 c2m2 c2v2 : Vec 256)
    (c3W1 : Kernel4 256 256 3 3) (c3b1 : Vec 256) (c3g1 c3t1 c3m1 c3v1 : Vec 256) (c3W2 : Kernel4 256 256 3 3) (c3b2 : Vec 256) (c3g2 c3t2 c3m2 c3v2 : Vec 256)
    (c4W1 : Kernel4 256 256 3 3) (c4b1 : Vec 256) (c4g1 c4t1 c4m1 c4v1 : Vec 256) (c4W2 : Kernel4 256 256 3 3) (c4b2 : Vec 256) (c4g2 c4t2 c4m2 c4v2 : Vec 256)
    (d4W1 : Kernel4 512 256 3 3) (d4b1 : Vec 512) (d4g1 d4t1 d4m1 d4v1 : Vec 512) (d4W2 : Kernel4 512 512 3 3) (d4b2 : Vec 512) (d4g2 d4t2 d4m2 d4v2 : Vec 512) (d4Wp : Kernel4 512 256 1 1) (d4bp : Vec 512) (d4gp d4tp d4mp d4vp : Vec 512)
    (e0W1 : Kernel4 512 512 3 3) (e0b1 : Vec 512) (e0g1 e0t1 e0m1 e0v1 : Vec 512) (e0W2 : Kernel4 512 512 3 3) (e0b2 : Vec 512) (e0g2 e0t2 e0m2 e0v2 : Vec 512)
    (e1W1 : Kernel4 512 512 3 3) (e1b1 : Vec 512) (e1g1 e1t1 e1m1 e1v1 : Vec 512) (e1W2 : Kernel4 512 512 3 3) (e1b2 : Vec 512) (e1g2 e1t2 e1m2 e1v2 : Vec 512)
    (Wd : Mat 512 10) (bd : Vec 10)
    (x : Vec (3 * 224 * 224)) : SHlo 10 :=
  let pooled : SHlo (64 * 56 * 56) :=
    .maxPool3s2F (c := 64) (h := 56) (w := 56)
      (.reluF (.bnPerChannelEvalF (oc := 64) (h := 112) (w := 112) "%sg" "%sbt" "%stnmu" "%stnvar"
        epsStr ε γs βs μs vs
        (.flatConvStridedF (h := 112) (w := 112) "%sW" "%sb" Ws bs (.operand "%x" x))))
  let s1b0 := idBlockGraphPCEval "s1b0" epsStr a0W1 a0b1 ε a0g1 a0t1 a0m1 a0v1 a0W2 a0b2 ε a0g2 a0t2 a0m2 a0v2 pooled
  let s1b1 := idBlockGraphPCEval "s1b1" epsStr a1W1 a1b1 ε a1g1 a1t1 a1m1 a1v1 a1W2 a1b2 ε a1g2 a1t2 a1m2 a1v2 s1b0
  let s1b2 := idBlockGraphPCEval "s1b2" epsStr a2W1 a2b1 ε a2g1 a2t1 a2m1 a2v1 a2W2 a2b2 ε a2g2 a2t2 a2m2 a2v2 s1b1
  let d2 := downBlockGraphPCEval "d2" epsStr d2W1 d2b1 ε d2g1 d2t1 d2m1 d2v1 d2W2 d2b2 ε d2g2 d2t2 d2m2 d2v2 d2Wp d2bp ε d2gp d2tp d2mp d2vp s1b2
  let s2b0 := idBlockGraphPCEval "s2b0" epsStr b0W1 b0b1 ε b0g1 b0t1 b0m1 b0v1 b0W2 b0b2 ε b0g2 b0t2 b0m2 b0v2 d2
  let s2b1 := idBlockGraphPCEval "s2b1" epsStr b1W1 b1b1 ε b1g1 b1t1 b1m1 b1v1 b1W2 b1b2 ε b1g2 b1t2 b1m2 b1v2 s2b0
  let s2b2 := idBlockGraphPCEval "s2b2" epsStr b2W1 b2b1 ε b2g1 b2t1 b2m1 b2v1 b2W2 b2b2 ε b2g2 b2t2 b2m2 b2v2 s2b1
  let d3 := downBlockGraphPCEval "d3" epsStr d3W1 d3b1 ε d3g1 d3t1 d3m1 d3v1 d3W2 d3b2 ε d3g2 d3t2 d3m2 d3v2 d3Wp d3bp ε d3gp d3tp d3mp d3vp s2b2
  let s3b0 := idBlockGraphPCEval "s3b0" epsStr c0W1 c0b1 ε c0g1 c0t1 c0m1 c0v1 c0W2 c0b2 ε c0g2 c0t2 c0m2 c0v2 d3
  let s3b1 := idBlockGraphPCEval "s3b1" epsStr c1W1 c1b1 ε c1g1 c1t1 c1m1 c1v1 c1W2 c1b2 ε c1g2 c1t2 c1m2 c1v2 s3b0
  let s3b2 := idBlockGraphPCEval "s3b2" epsStr c2W1 c2b1 ε c2g1 c2t1 c2m1 c2v1 c2W2 c2b2 ε c2g2 c2t2 c2m2 c2v2 s3b1
  let s3b3 := idBlockGraphPCEval "s3b3" epsStr c3W1 c3b1 ε c3g1 c3t1 c3m1 c3v1 c3W2 c3b2 ε c3g2 c3t2 c3m2 c3v2 s3b2
  let s3b4 := idBlockGraphPCEval "s3b4" epsStr c4W1 c4b1 ε c4g1 c4t1 c4m1 c4v1 c4W2 c4b2 ε c4g2 c4t2 c4m2 c4v2 s3b3
  let d4 := downBlockGraphPCEval "d4" epsStr d4W1 d4b1 ε d4g1 d4t1 d4m1 d4v1 d4W2 d4b2 ε d4g2 d4t2 d4m2 d4v2 d4Wp d4bp ε d4gp d4tp d4mp d4vp s3b4
  let s4b0 := idBlockGraphPCEval "s4b0" epsStr e0W1 e0b1 ε e0g1 e0t1 e0m1 e0v1 e0W2 e0b2 ε e0g2 e0t2 e0m2 e0v2 d4
  let s4b1 := idBlockGraphPCEval "s4b1" epsStr e1W1 e1b1 ε e1g1 e1t1 e1m1 e1v1 e1W2 e1b2 ε e1g2 e1t2 e1m2 e1v2 s4b0
  denseF "%Wd" "%bd" Wd bd (.gapF (c := 512) (h := 7) (w := 7) s4b1)

set_option maxRecDepth 10000 in
/-- ⭐ **Full inference ResNet-34 forward faithfulness.** `den (resnet34FwdGraphFullPCEval …) =
    resnet34Forward_full_pc_eval …`, chaining the per-block faithful lemmas + stem/maxpool/GAP/
    dense. The eval half of "text = render of a proven graph", and the tie
    `Resnet34FloatBudget.lean`'s number was missing. -/
theorem resnet34FwdGraphFullPCEval_faithful (epsStr : String) (ε : ℝ)
    (Ws : Kernel4 64 3 7 7) (bs : Vec 64) (γs βs μs vs : Vec 64)
    (a0W1 : Kernel4 64 64 3 3) (a0b1 : Vec 64) (a0g1 a0t1 a0m1 a0v1 : Vec 64) (a0W2 : Kernel4 64 64 3 3) (a0b2 : Vec 64) (a0g2 a0t2 a0m2 a0v2 : Vec 64)
    (a1W1 : Kernel4 64 64 3 3) (a1b1 : Vec 64) (a1g1 a1t1 a1m1 a1v1 : Vec 64) (a1W2 : Kernel4 64 64 3 3) (a1b2 : Vec 64) (a1g2 a1t2 a1m2 a1v2 : Vec 64)
    (a2W1 : Kernel4 64 64 3 3) (a2b1 : Vec 64) (a2g1 a2t1 a2m1 a2v1 : Vec 64) (a2W2 : Kernel4 64 64 3 3) (a2b2 : Vec 64) (a2g2 a2t2 a2m2 a2v2 : Vec 64)
    (d2W1 : Kernel4 128 64 3 3) (d2b1 : Vec 128) (d2g1 d2t1 d2m1 d2v1 : Vec 128) (d2W2 : Kernel4 128 128 3 3) (d2b2 : Vec 128) (d2g2 d2t2 d2m2 d2v2 : Vec 128) (d2Wp : Kernel4 128 64 1 1) (d2bp : Vec 128) (d2gp d2tp d2mp d2vp : Vec 128)
    (b0W1 : Kernel4 128 128 3 3) (b0b1 : Vec 128) (b0g1 b0t1 b0m1 b0v1 : Vec 128) (b0W2 : Kernel4 128 128 3 3) (b0b2 : Vec 128) (b0g2 b0t2 b0m2 b0v2 : Vec 128)
    (b1W1 : Kernel4 128 128 3 3) (b1b1 : Vec 128) (b1g1 b1t1 b1m1 b1v1 : Vec 128) (b1W2 : Kernel4 128 128 3 3) (b1b2 : Vec 128) (b1g2 b1t2 b1m2 b1v2 : Vec 128)
    (b2W1 : Kernel4 128 128 3 3) (b2b1 : Vec 128) (b2g1 b2t1 b2m1 b2v1 : Vec 128) (b2W2 : Kernel4 128 128 3 3) (b2b2 : Vec 128) (b2g2 b2t2 b2m2 b2v2 : Vec 128)
    (d3W1 : Kernel4 256 128 3 3) (d3b1 : Vec 256) (d3g1 d3t1 d3m1 d3v1 : Vec 256) (d3W2 : Kernel4 256 256 3 3) (d3b2 : Vec 256) (d3g2 d3t2 d3m2 d3v2 : Vec 256) (d3Wp : Kernel4 256 128 1 1) (d3bp : Vec 256) (d3gp d3tp d3mp d3vp : Vec 256)
    (c0W1 : Kernel4 256 256 3 3) (c0b1 : Vec 256) (c0g1 c0t1 c0m1 c0v1 : Vec 256) (c0W2 : Kernel4 256 256 3 3) (c0b2 : Vec 256) (c0g2 c0t2 c0m2 c0v2 : Vec 256)
    (c1W1 : Kernel4 256 256 3 3) (c1b1 : Vec 256) (c1g1 c1t1 c1m1 c1v1 : Vec 256) (c1W2 : Kernel4 256 256 3 3) (c1b2 : Vec 256) (c1g2 c1t2 c1m2 c1v2 : Vec 256)
    (c2W1 : Kernel4 256 256 3 3) (c2b1 : Vec 256) (c2g1 c2t1 c2m1 c2v1 : Vec 256) (c2W2 : Kernel4 256 256 3 3) (c2b2 : Vec 256) (c2g2 c2t2 c2m2 c2v2 : Vec 256)
    (c3W1 : Kernel4 256 256 3 3) (c3b1 : Vec 256) (c3g1 c3t1 c3m1 c3v1 : Vec 256) (c3W2 : Kernel4 256 256 3 3) (c3b2 : Vec 256) (c3g2 c3t2 c3m2 c3v2 : Vec 256)
    (c4W1 : Kernel4 256 256 3 3) (c4b1 : Vec 256) (c4g1 c4t1 c4m1 c4v1 : Vec 256) (c4W2 : Kernel4 256 256 3 3) (c4b2 : Vec 256) (c4g2 c4t2 c4m2 c4v2 : Vec 256)
    (d4W1 : Kernel4 512 256 3 3) (d4b1 : Vec 512) (d4g1 d4t1 d4m1 d4v1 : Vec 512) (d4W2 : Kernel4 512 512 3 3) (d4b2 : Vec 512) (d4g2 d4t2 d4m2 d4v2 : Vec 512) (d4Wp : Kernel4 512 256 1 1) (d4bp : Vec 512) (d4gp d4tp d4mp d4vp : Vec 512)
    (e0W1 : Kernel4 512 512 3 3) (e0b1 : Vec 512) (e0g1 e0t1 e0m1 e0v1 : Vec 512) (e0W2 : Kernel4 512 512 3 3) (e0b2 : Vec 512) (e0g2 e0t2 e0m2 e0v2 : Vec 512)
    (e1W1 : Kernel4 512 512 3 3) (e1b1 : Vec 512) (e1g1 e1t1 e1m1 e1v1 : Vec 512) (e1W2 : Kernel4 512 512 3 3) (e1b2 : Vec 512) (e1g2 e1t2 e1m2 e1v2 : Vec 512)
    (Wd : Mat 512 10) (bd : Vec 10)
    (x : Vec (3 * 224 * 224)) :
    den (resnet34FwdGraphFullPCEval epsStr ε Ws bs γs βs μs vs a0W1 a0b1 a0g1 a0t1 a0m1 a0v1 a0W2 a0b2 a0g2 a0t2 a0m2 a0v2 a1W1 a1b1 a1g1 a1t1 a1m1 a1v1 a1W2 a1b2 a1g2 a1t2 a1m2 a1v2 a2W1 a2b1 a2g1 a2t1 a2m1 a2v1 a2W2 a2b2 a2g2 a2t2 a2m2 a2v2 d2W1 d2b1 d2g1 d2t1 d2m1 d2v1 d2W2 d2b2 d2g2 d2t2 d2m2 d2v2 d2Wp d2bp d2gp d2tp d2mp d2vp b0W1 b0b1 b0g1 b0t1 b0m1 b0v1 b0W2 b0b2 b0g2 b0t2 b0m2 b0v2 b1W1 b1b1 b1g1 b1t1 b1m1 b1v1 b1W2 b1b2 b1g2 b1t2 b1m2 b1v2 b2W1 b2b1 b2g1 b2t1 b2m1 b2v1 b2W2 b2b2 b2g2 b2t2 b2m2 b2v2 d3W1 d3b1 d3g1 d3t1 d3m1 d3v1 d3W2 d3b2 d3g2 d3t2 d3m2 d3v2 d3Wp d3bp d3gp d3tp d3mp d3vp c0W1 c0b1 c0g1 c0t1 c0m1 c0v1 c0W2 c0b2 c0g2 c0t2 c0m2 c0v2 c1W1 c1b1 c1g1 c1t1 c1m1 c1v1 c1W2 c1b2 c1g2 c1t2 c1m2 c1v2 c2W1 c2b1 c2g1 c2t1 c2m1 c2v1 c2W2 c2b2 c2g2 c2t2 c2m2 c2v2 c3W1 c3b1 c3g1 c3t1 c3m1 c3v1 c3W2 c3b2 c3g2 c3t2 c3m2 c3v2 c4W1 c4b1 c4g1 c4t1 c4m1 c4v1 c4W2 c4b2 c4g2 c4t2 c4m2 c4v2 d4W1 d4b1 d4g1 d4t1 d4m1 d4v1 d4W2 d4b2 d4g2 d4t2 d4m2 d4v2 d4Wp d4bp d4gp d4tp d4mp d4vp e0W1 e0b1 e0g1 e0t1 e0m1 e0v1 e0W2 e0b2 e0g2 e0t2 e0m2 e0v2 e1W1 e1b1 e1g1 e1t1 e1m1 e1v1 e1W2 e1b2 e1g2 e1t2 e1m2 e1v2 Wd bd x)
      = resnet34Forward_full_pc_eval ε Ws bs γs βs μs vs a0W1 a0b1 a0g1 a0t1 a0m1 a0v1 a0W2 a0b2 a0g2 a0t2 a0m2 a0v2 a1W1 a1b1 a1g1 a1t1 a1m1 a1v1 a1W2 a1b2 a1g2 a1t2 a1m2 a1v2 a2W1 a2b1 a2g1 a2t1 a2m1 a2v1 a2W2 a2b2 a2g2 a2t2 a2m2 a2v2 d2W1 d2b1 d2g1 d2t1 d2m1 d2v1 d2W2 d2b2 d2g2 d2t2 d2m2 d2v2 d2Wp d2bp d2gp d2tp d2mp d2vp b0W1 b0b1 b0g1 b0t1 b0m1 b0v1 b0W2 b0b2 b0g2 b0t2 b0m2 b0v2 b1W1 b1b1 b1g1 b1t1 b1m1 b1v1 b1W2 b1b2 b1g2 b1t2 b1m2 b1v2 b2W1 b2b1 b2g1 b2t1 b2m1 b2v1 b2W2 b2b2 b2g2 b2t2 b2m2 b2v2 d3W1 d3b1 d3g1 d3t1 d3m1 d3v1 d3W2 d3b2 d3g2 d3t2 d3m2 d3v2 d3Wp d3bp d3gp d3tp d3mp d3vp c0W1 c0b1 c0g1 c0t1 c0m1 c0v1 c0W2 c0b2 c0g2 c0t2 c0m2 c0v2 c1W1 c1b1 c1g1 c1t1 c1m1 c1v1 c1W2 c1b2 c1g2 c1t2 c1m2 c1v2 c2W1 c2b1 c2g1 c2t1 c2m1 c2v1 c2W2 c2b2 c2g2 c2t2 c2m2 c2v2 c3W1 c3b1 c3g1 c3t1 c3m1 c3v1 c3W2 c3b2 c3g2 c3t2 c3m2 c3v2 c4W1 c4b1 c4g1 c4t1 c4m1 c4v1 c4W2 c4b2 c4g2 c4t2 c4m2 c4v2 d4W1 d4b1 d4g1 d4t1 d4m1 d4v1 d4W2 d4b2 d4g2 d4t2 d4m2 d4v2 d4Wp d4bp d4gp d4tp d4mp d4vp e0W1 e0b1 e0g1 e0t1 e0m1 e0v1 e0W2 e0b2 e0g2 e0t2 e0m2 e0v2 e1W1 e1b1 e1g1 e1t1 e1m1 e1v1 e1W2 e1b2 e1g2 e1t2 e1m2 e1v2 Wd bd x := by
  simp only [resnet34FwdGraphFullPCEval, resnet34Forward_full_pc_eval, idFwdEval, downFwdEval,
             cbrStridedPCEval, idBlockGraphPCEval_faithful, downBlockGraphPCEval_faithful,
             reluF_faithful, bnPerChannelEvalF_faithful, flatConvStridedF_faithful,
             maxPool3s2F_faithful, gapF_faithful, denseF_faithful, den_operand,
             Function.comp_apply]

end StableHLO
end Proofs

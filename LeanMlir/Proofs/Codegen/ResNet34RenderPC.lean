import LeanMlir.Proofs.Codegen.StableHLO

/-! # r34 Item A — the PER-CHANNEL-BN ResNet-34 forward graph (matches the render)

The ResNet-34 peer of `MobileNetV2RenderPC.lean`. `StableHLO.lean`'s `resnetFwdGraph` is a
*representative* (stem + 1 identity + 1 projection block + GAP + dense) using **scalar** `bnF`; the
operational render (`tests/TestResnet34Train.lean`) emits **per-channel** BN, the full 16-block
`[3,4,6,3]` net, a 7×7 strided stem and a maxpool. This file is the per-channel twin matching the
render:

* per-channel building blocks `cbrStridedPC` (7×7 strided stem conv→bn→relu), `rblkPC` (identity
  basic block `relu(F(x)+x)`), `rblkPStridedPC` (downsample basic block `relu(F_s(x)+proj_s(x))`,
  with a 3×3 strided projection skip) — per-channel `bnPerChannelTensor3` mirrors of `cbr`/`rblk`/
  `rblkP` (CNN.lean).
* per-block typed `SHlo` graphs `idBlockGraphPC`/`downBlockGraphPC` + their `_faithful` lemmas
  (`den (block graph) = block forward (den input)`), via `bnPerChannelF_faithful`/`reluF_faithful`/
  `flatConv(Strided)F_faithful`/`den_addV`. The residual skip reuses the block-input subtree in both
  `addV` operands (tree-safe), as in `resnetFwdGraph`.

Part 2 (below) chains these into the full `resnet34FwdGraphFullPC` + `resnet34Forward_full_pc` +
faithfulness at the render dims (3×224² → 7×7×512). Prerequisite for the structured render (Item B).
No new tokens. 3-axiom clean.
-/

namespace Proofs

open scoped BigOperators

-- ════════════════════════════════════════════════════════════════
-- § Per-channel building-block ℝ-forwards (per-channel mirrors of cbr / rblk / rblkP)
-- ════════════════════════════════════════════════════════════════

/-- 7×7 strided stem conv → bn → relu, per-channel BN. -/
@[reducible] noncomputable def cbrStridedPC {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (γ β : Vec oc) :
    Vec (ic * (2 * h) * (2 * w)) → Vec (oc * h * w) :=
  relu (oc * h * w) ∘ bnPerChannelTensor3 oc h w ε γ β ∘ flatConvStride2 W b

/-- Identity basic block `relu(F(x) + x)`, `F = (bn∘conv) ∘ (relu∘bn∘conv)`, per-channel BN. -/
@[reducible] noncomputable def rblkPC {c h w kH₁ kW₁ kH₂ kW₂ : Nat}
    (W₁ : Kernel4 c c kH₁ kW₁) (b₁ : Vec c) (ε₁ : ℝ) (γ₁ β₁ : Vec c)
    (W₂ : Kernel4 c c kH₂ kW₂) (b₂ : Vec c) (ε₂ : ℝ) (γ₂ β₂ : Vec c) :
    Vec (c * h * w) → Vec (c * h * w) :=
  relu (c * h * w) ∘ residual
    ((bnPerChannelTensor3 c h w ε₂ γ₂ β₂ ∘ flatConv W₂ b₂) ∘
      (relu (c * h * w) ∘ bnPerChannelTensor3 c h w ε₁ γ₁ β₁ ∘ flatConv W₁ b₁))

/-- Downsample basic block `relu(F_s(x) + proj_s(x))`: body `bn∘conv ∘ relu∘bn∘conv_strided`
    (`ic→oc`, halves spatial), projection `bn∘conv_strided` (3×3 stride-2 skip). Per-channel BN. -/
@[reducible] noncomputable def rblkPStridedPC {ic oc h w kH₁ kW₁ kH₂ kW₂ kHp kWp : Nat}
    (W₁ : Kernel4 oc ic kH₁ kW₁) (b₁ : Vec oc) (ε₁ : ℝ) (γ₁ β₁ : Vec oc)
    (W₂ : Kernel4 oc oc kH₂ kW₂) (b₂ : Vec oc) (ε₂ : ℝ) (γ₂ β₂ : Vec oc)
    (Wp : Kernel4 oc ic kHp kWp) (bp : Vec oc) (εp : ℝ) (γp βp : Vec oc) :
    Vec (ic * (2 * h) * (2 * w)) → Vec (oc * h * w) :=
  relu (oc * h * w) ∘ residualProj
    (bnPerChannelTensor3 oc h w εp γp βp ∘ flatConvStride2 Wp bp)
    ((bnPerChannelTensor3 oc h w ε₂ γ₂ β₂ ∘ flatConv W₂ b₂) ∘
      (relu (oc * h * w) ∘ bnPerChannelTensor3 oc h w ε₁ γ₁ β₁ ∘ flatConvStride2 W₁ b₁))

-- ════════════════════════════════════════════════════════════════
-- § The full per-channel-BN ResNet-34 ℝ-forward (ch6 render dims, shared ε)
--   stem(7×7-s2)→bn→relu → maxpool → [3,4,6,3] basic blocks → GAP → dense
-- ════════════════════════════════════════════════════════════════

/-- Identity-block ℝ-forward at shared ε (the partial application `rblkPC` takes). -/
@[reducible] noncomputable def idFwd {c h w : Nat} (ε : ℝ)
    (W₁ : Kernel4 c c 3 3) (b₁ : Vec c) (γ₁ β₁ : Vec c)
    (W₂ : Kernel4 c c 3 3) (b₂ : Vec c) (γ₂ β₂ : Vec c) : Vec (c*h*w) → Vec (c*h*w) :=
  rblkPC (h := h) (w := w) W₁ b₁ ε γ₁ β₁ W₂ b₂ ε γ₂ β₂

/-- Downsample-block ℝ-forward at shared ε. -/
@[reducible] noncomputable def downFwd {ic oc h w : Nat} (ε : ℝ)
    (W₁ : Kernel4 oc ic 3 3) (b₁ : Vec oc) (γ₁ β₁ : Vec oc)
    (W₂ : Kernel4 oc oc 3 3) (b₂ : Vec oc) (γ₂ β₂ : Vec oc)
    (Wp : Kernel4 oc ic 3 3) (bp : Vec oc) (γp βp : Vec oc) :
    Vec (ic*(2*h)*(2*w)) → Vec (oc*h*w) :=
  rblkPStridedPC (h := h) (w := w) W₁ b₁ ε γ₁ β₁ W₂ b₂ ε γ₂ β₂ Wp bp ε γp βp

/-- The full per-channel ResNet-34 forward (render dims 3×224² → 7×7×512). -/
noncomputable def resnet34Forward_full_pc (ε : ℝ)
    (Ws : Kernel4 64 3 7 7) (bs : Vec 64) (γs βs : Vec 64)
    (a0W1 : Kernel4 64 64 3 3) (a0b1 : Vec 64) (a0g1 a0t1 : Vec 64) (a0W2 : Kernel4 64 64 3 3) (a0b2 : Vec 64) (a0g2 a0t2 : Vec 64)
    (a1W1 : Kernel4 64 64 3 3) (a1b1 : Vec 64) (a1g1 a1t1 : Vec 64) (a1W2 : Kernel4 64 64 3 3) (a1b2 : Vec 64) (a1g2 a1t2 : Vec 64)
    (a2W1 : Kernel4 64 64 3 3) (a2b1 : Vec 64) (a2g1 a2t1 : Vec 64) (a2W2 : Kernel4 64 64 3 3) (a2b2 : Vec 64) (a2g2 a2t2 : Vec 64)
    (d2W1 : Kernel4 128 64 3 3) (d2b1 : Vec 128) (d2g1 d2t1 : Vec 128) (d2W2 : Kernel4 128 128 3 3) (d2b2 : Vec 128) (d2g2 d2t2 : Vec 128) (d2Wp : Kernel4 128 64 3 3) (d2bp : Vec 128) (d2gp d2tp : Vec 128)
    (b0W1 : Kernel4 128 128 3 3) (b0b1 : Vec 128) (b0g1 b0t1 : Vec 128) (b0W2 : Kernel4 128 128 3 3) (b0b2 : Vec 128) (b0g2 b0t2 : Vec 128)
    (b1W1 : Kernel4 128 128 3 3) (b1b1 : Vec 128) (b1g1 b1t1 : Vec 128) (b1W2 : Kernel4 128 128 3 3) (b1b2 : Vec 128) (b1g2 b1t2 : Vec 128)
    (b2W1 : Kernel4 128 128 3 3) (b2b1 : Vec 128) (b2g1 b2t1 : Vec 128) (b2W2 : Kernel4 128 128 3 3) (b2b2 : Vec 128) (b2g2 b2t2 : Vec 128)
    (d3W1 : Kernel4 256 128 3 3) (d3b1 : Vec 256) (d3g1 d3t1 : Vec 256) (d3W2 : Kernel4 256 256 3 3) (d3b2 : Vec 256) (d3g2 d3t2 : Vec 256) (d3Wp : Kernel4 256 128 3 3) (d3bp : Vec 256) (d3gp d3tp : Vec 256)
    (c0W1 : Kernel4 256 256 3 3) (c0b1 : Vec 256) (c0g1 c0t1 : Vec 256) (c0W2 : Kernel4 256 256 3 3) (c0b2 : Vec 256) (c0g2 c0t2 : Vec 256)
    (c1W1 : Kernel4 256 256 3 3) (c1b1 : Vec 256) (c1g1 c1t1 : Vec 256) (c1W2 : Kernel4 256 256 3 3) (c1b2 : Vec 256) (c1g2 c1t2 : Vec 256)
    (c2W1 : Kernel4 256 256 3 3) (c2b1 : Vec 256) (c2g1 c2t1 : Vec 256) (c2W2 : Kernel4 256 256 3 3) (c2b2 : Vec 256) (c2g2 c2t2 : Vec 256)
    (c3W1 : Kernel4 256 256 3 3) (c3b1 : Vec 256) (c3g1 c3t1 : Vec 256) (c3W2 : Kernel4 256 256 3 3) (c3b2 : Vec 256) (c3g2 c3t2 : Vec 256)
    (c4W1 : Kernel4 256 256 3 3) (c4b1 : Vec 256) (c4g1 c4t1 : Vec 256) (c4W2 : Kernel4 256 256 3 3) (c4b2 : Vec 256) (c4g2 c4t2 : Vec 256)
    (d4W1 : Kernel4 512 256 3 3) (d4b1 : Vec 512) (d4g1 d4t1 : Vec 512) (d4W2 : Kernel4 512 512 3 3) (d4b2 : Vec 512) (d4g2 d4t2 : Vec 512) (d4Wp : Kernel4 512 256 3 3) (d4bp : Vec 512) (d4gp d4tp : Vec 512)
    (e0W1 : Kernel4 512 512 3 3) (e0b1 : Vec 512) (e0g1 e0t1 : Vec 512) (e0W2 : Kernel4 512 512 3 3) (e0b2 : Vec 512) (e0g2 e0t2 : Vec 512)
    (e1W1 : Kernel4 512 512 3 3) (e1b1 : Vec 512) (e1g1 e1t1 : Vec 512) (e1W2 : Kernel4 512 512 3 3) (e1b2 : Vec 512) (e1g2 e1t2 : Vec 512)
    (Wd : Mat 512 10) (bd : Vec 10) :
    Vec (3 * 224 * 224) → Vec 10 :=
  dense Wd bd ∘ globalAvgPoolFlat 512 7 7 ∘
  idFwd (h := 7) (w := 7) ε e1W1 e1b1 e1g1 e1t1 e1W2 e1b2 e1g2 e1t2 ∘
  idFwd (h := 7) (w := 7) ε e0W1 e0b1 e0g1 e0t1 e0W2 e0b2 e0g2 e0t2 ∘
  downFwd (h := 7) (w := 7) ε d4W1 d4b1 d4g1 d4t1 d4W2 d4b2 d4g2 d4t2 d4Wp d4bp d4gp d4tp ∘
  idFwd (h := 14) (w := 14) ε c4W1 c4b1 c4g1 c4t1 c4W2 c4b2 c4g2 c4t2 ∘
  idFwd (h := 14) (w := 14) ε c3W1 c3b1 c3g1 c3t1 c3W2 c3b2 c3g2 c3t2 ∘
  idFwd (h := 14) (w := 14) ε c2W1 c2b1 c2g1 c2t1 c2W2 c2b2 c2g2 c2t2 ∘
  idFwd (h := 14) (w := 14) ε c1W1 c1b1 c1g1 c1t1 c1W2 c1b2 c1g2 c1t2 ∘
  idFwd (h := 14) (w := 14) ε c0W1 c0b1 c0g1 c0t1 c0W2 c0b2 c0g2 c0t2 ∘
  downFwd (h := 14) (w := 14) ε d3W1 d3b1 d3g1 d3t1 d3W2 d3b2 d3g2 d3t2 d3Wp d3bp d3gp d3tp ∘
  idFwd (h := 28) (w := 28) ε b2W1 b2b1 b2g1 b2t1 b2W2 b2b2 b2g2 b2t2 ∘
  idFwd (h := 28) (w := 28) ε b1W1 b1b1 b1g1 b1t1 b1W2 b1b2 b1g2 b1t2 ∘
  idFwd (h := 28) (w := 28) ε b0W1 b0b1 b0g1 b0t1 b0W2 b0b2 b0g2 b0t2 ∘
  downFwd (h := 28) (w := 28) ε d2W1 d2b1 d2g1 d2t1 d2W2 d2b2 d2g2 d2t2 d2Wp d2bp d2gp d2tp ∘
  idFwd (h := 56) (w := 56) ε a2W1 a2b1 a2g1 a2t1 a2W2 a2b2 a2g2 a2t2 ∘
  idFwd (h := 56) (w := 56) ε a1W1 a1b1 a1g1 a1t1 a1W2 a1b2 a1g2 a1t2 ∘
  idFwd (h := 56) (w := 56) ε a0W1 a0b1 a0g1 a0t1 a0W2 a0b2 a0g2 a0t2 ∘
  maxPoolFlat 64 56 56 ∘
  cbrStridedPC (h := 112) (w := 112) Ws bs ε γs βs

namespace StableHLO

-- ════════════════════════════════════════════════════════════════
-- § Per-block typed `SHlo` graphs + faithfulness
-- ════════════════════════════════════════════════════════════════

/-- Identity-block forward graph: `relu(addV(bn∘conv∘relu∘bn∘conv, skip))`; the skip reuses the
    block-input subtree `e`. -/
def idBlockGraphPC (p epsStr : String) {c h w : Nat}
    (W₁ : Kernel4 c c 3 3) (b₁ : Vec c) (ε₁ : ℝ) (γ₁ β₁ : Vec c)
    (W₂ : Kernel4 c c 3 3) (b₂ : Vec c) (ε₂ : ℝ) (γ₂ β₂ : Vec c)
    (e : SHlo (c * h * w)) : SHlo (c * h * w) :=
  .reluF (.addV
    (.bnPerChannelF (oc := c) (h := h) (w := w) s!"%{p}g2" s!"%{p}bt2" epsStr ε₂ γ₂ β₂
      (.flatConvF (h := h) (w := w) s!"%{p}W2" s!"%{p}b2" W₂ b₂
        (.reluF (.bnPerChannelF (oc := c) (h := h) (w := w) s!"%{p}g1" s!"%{p}bt1" epsStr ε₁ γ₁ β₁
          (.flatConvF (h := h) (w := w) s!"%{p}W1" s!"%{p}b1" W₁ b₁ e)))))
    e)

/-- Downsample-block forward graph: `relu(addV(body, projection))`, body strided conv1 + conv2,
    projection a 3×3 strided conv; both read the block-input subtree `e`. -/
def downBlockGraphPC (p epsStr : String) {ic oc h w : Nat}
    (W₁ : Kernel4 oc ic 3 3) (b₁ : Vec oc) (ε₁ : ℝ) (γ₁ β₁ : Vec oc)
    (W₂ : Kernel4 oc oc 3 3) (b₂ : Vec oc) (ε₂ : ℝ) (γ₂ β₂ : Vec oc)
    (Wp : Kernel4 oc ic 3 3) (bp : Vec oc) (εp : ℝ) (γp βp : Vec oc)
    (e : SHlo (ic * (2 * h) * (2 * w))) : SHlo (oc * h * w) :=
  -- `addV proj body` (proj first, matching `residualProj proj body`; `+` is commutative)
  .reluF (.addV
    (.bnPerChannelF (oc := oc) (h := h) (w := w) s!"%{p}gp" s!"%{p}btp" epsStr εp γp βp
      (.flatConvStridedF (h := h) (w := w) s!"%{p}Wp" s!"%{p}bp" Wp bp e))
    (.bnPerChannelF (oc := oc) (h := h) (w := w) s!"%{p}g2" s!"%{p}bt2" epsStr ε₂ γ₂ β₂
      (.flatConvF (h := h) (w := w) s!"%{p}W2" s!"%{p}b2" W₂ b₂
        (.reluF (.bnPerChannelF (oc := oc) (h := h) (w := w) s!"%{p}g1" s!"%{p}bt1" epsStr ε₁ γ₁ β₁
          (.flatConvStridedF (h := h) (w := w) s!"%{p}W1" s!"%{p}b1" W₁ b₁ e))))))

/-- **Identity block faithfulness.** `den (idBlockGraphPC … e) = rblkPC … (den e)`. -/
theorem idBlockGraphPC_faithful (p epsStr : String) {c h w : Nat}
    (W₁ : Kernel4 c c 3 3) (b₁ : Vec c) (ε₁ : ℝ) (γ₁ β₁ : Vec c)
    (W₂ : Kernel4 c c 3 3) (b₂ : Vec c) (ε₂ : ℝ) (γ₂ β₂ : Vec c)
    (e : SHlo (c * h * w)) :
    den (idBlockGraphPC p epsStr W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ e)
      = rblkPC W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ (den e) := by
  simp only [idBlockGraphPC, reluF_faithful, bnPerChannelF_faithful, flatConvF_faithful, den_addV]
  unfold rblkPC residual biPath
  simp only [Function.comp_apply]

/-- **Downsample block faithfulness.** `den (downBlockGraphPC … e) = rblkPStridedPC … (den e)`. -/
theorem downBlockGraphPC_faithful (p epsStr : String) {ic oc h w : Nat}
    (W₁ : Kernel4 oc ic 3 3) (b₁ : Vec oc) (ε₁ : ℝ) (γ₁ β₁ : Vec oc)
    (W₂ : Kernel4 oc oc 3 3) (b₂ : Vec oc) (ε₂ : ℝ) (γ₂ β₂ : Vec oc)
    (Wp : Kernel4 oc ic 3 3) (bp : Vec oc) (εp : ℝ) (γp βp : Vec oc)
    (e : SHlo (ic * (2 * h) * (2 * w))) :
    den (downBlockGraphPC p epsStr W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ Wp bp εp γp βp e)
      = rblkPStridedPC W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ Wp bp εp γp βp (den e) := by
  simp only [downBlockGraphPC, reluF_faithful, bnPerChannelF_faithful, flatConvF_faithful,
             flatConvStridedF_faithful, den_addV]
  unfold rblkPStridedPC residualProj biPath
  simp only [Function.comp_apply]

-- ════════════════════════════════════════════════════════════════
-- § The full per-channel-BN ResNet-34 typed `SHlo` forward graph + faithfulness
-- ════════════════════════════════════════════════════════════════

/-- Whole **per-channel-BN ResNet-34 forward** graph at the render dims (3×224² → 7×7×512): 7×7
    strided stem (`flatConvStridedF`) → bn → relu → `maxPoolF` → `[3,4,6,3]` basic blocks (each via
    `idBlockGraphPC`/`downBlockGraphPC`) → GAP → dense. **Per-channel** BN throughout, matching the
    render. The per-channel twin of `resnetFwdGraph`, at full depth. -/
def resnet34FwdGraphFullPC (epsStr : String) (ε : ℝ)
    (Ws : Kernel4 64 3 7 7) (bs : Vec 64) (γs βs : Vec 64)
    (a0W1 : Kernel4 64 64 3 3) (a0b1 : Vec 64) (a0g1 a0t1 : Vec 64) (a0W2 : Kernel4 64 64 3 3) (a0b2 : Vec 64) (a0g2 a0t2 : Vec 64)
    (a1W1 : Kernel4 64 64 3 3) (a1b1 : Vec 64) (a1g1 a1t1 : Vec 64) (a1W2 : Kernel4 64 64 3 3) (a1b2 : Vec 64) (a1g2 a1t2 : Vec 64)
    (a2W1 : Kernel4 64 64 3 3) (a2b1 : Vec 64) (a2g1 a2t1 : Vec 64) (a2W2 : Kernel4 64 64 3 3) (a2b2 : Vec 64) (a2g2 a2t2 : Vec 64)
    (d2W1 : Kernel4 128 64 3 3) (d2b1 : Vec 128) (d2g1 d2t1 : Vec 128) (d2W2 : Kernel4 128 128 3 3) (d2b2 : Vec 128) (d2g2 d2t2 : Vec 128) (d2Wp : Kernel4 128 64 3 3) (d2bp : Vec 128) (d2gp d2tp : Vec 128)
    (b0W1 : Kernel4 128 128 3 3) (b0b1 : Vec 128) (b0g1 b0t1 : Vec 128) (b0W2 : Kernel4 128 128 3 3) (b0b2 : Vec 128) (b0g2 b0t2 : Vec 128)
    (b1W1 : Kernel4 128 128 3 3) (b1b1 : Vec 128) (b1g1 b1t1 : Vec 128) (b1W2 : Kernel4 128 128 3 3) (b1b2 : Vec 128) (b1g2 b1t2 : Vec 128)
    (b2W1 : Kernel4 128 128 3 3) (b2b1 : Vec 128) (b2g1 b2t1 : Vec 128) (b2W2 : Kernel4 128 128 3 3) (b2b2 : Vec 128) (b2g2 b2t2 : Vec 128)
    (d3W1 : Kernel4 256 128 3 3) (d3b1 : Vec 256) (d3g1 d3t1 : Vec 256) (d3W2 : Kernel4 256 256 3 3) (d3b2 : Vec 256) (d3g2 d3t2 : Vec 256) (d3Wp : Kernel4 256 128 3 3) (d3bp : Vec 256) (d3gp d3tp : Vec 256)
    (c0W1 : Kernel4 256 256 3 3) (c0b1 : Vec 256) (c0g1 c0t1 : Vec 256) (c0W2 : Kernel4 256 256 3 3) (c0b2 : Vec 256) (c0g2 c0t2 : Vec 256)
    (c1W1 : Kernel4 256 256 3 3) (c1b1 : Vec 256) (c1g1 c1t1 : Vec 256) (c1W2 : Kernel4 256 256 3 3) (c1b2 : Vec 256) (c1g2 c1t2 : Vec 256)
    (c2W1 : Kernel4 256 256 3 3) (c2b1 : Vec 256) (c2g1 c2t1 : Vec 256) (c2W2 : Kernel4 256 256 3 3) (c2b2 : Vec 256) (c2g2 c2t2 : Vec 256)
    (c3W1 : Kernel4 256 256 3 3) (c3b1 : Vec 256) (c3g1 c3t1 : Vec 256) (c3W2 : Kernel4 256 256 3 3) (c3b2 : Vec 256) (c3g2 c3t2 : Vec 256)
    (c4W1 : Kernel4 256 256 3 3) (c4b1 : Vec 256) (c4g1 c4t1 : Vec 256) (c4W2 : Kernel4 256 256 3 3) (c4b2 : Vec 256) (c4g2 c4t2 : Vec 256)
    (d4W1 : Kernel4 512 256 3 3) (d4b1 : Vec 512) (d4g1 d4t1 : Vec 512) (d4W2 : Kernel4 512 512 3 3) (d4b2 : Vec 512) (d4g2 d4t2 : Vec 512) (d4Wp : Kernel4 512 256 3 3) (d4bp : Vec 512) (d4gp d4tp : Vec 512)
    (e0W1 : Kernel4 512 512 3 3) (e0b1 : Vec 512) (e0g1 e0t1 : Vec 512) (e0W2 : Kernel4 512 512 3 3) (e0b2 : Vec 512) (e0g2 e0t2 : Vec 512)
    (e1W1 : Kernel4 512 512 3 3) (e1b1 : Vec 512) (e1g1 e1t1 : Vec 512) (e1W2 : Kernel4 512 512 3 3) (e1b2 : Vec 512) (e1g2 e1t2 : Vec 512)
    (Wd : Mat 512 10) (bd : Vec 10)
    (x : Vec (3 * 224 * 224)) : SHlo 10 :=
  let pooled : SHlo (64 * 56 * 56) :=
    .maxPoolF (c := 64) (h := 56) (w := 56)
      (.reluF (.bnPerChannelF (oc := 64) (h := 112) (w := 112) "%sg" "%sbt" epsStr ε γs βs
        (.flatConvStridedF (h := 112) (w := 112) "%sW" "%sb" Ws bs (.operand "%x" x))))
  let s1b0 := idBlockGraphPC "s1b0" epsStr a0W1 a0b1 ε a0g1 a0t1 a0W2 a0b2 ε a0g2 a0t2 pooled
  let s1b1 := idBlockGraphPC "s1b1" epsStr a1W1 a1b1 ε a1g1 a1t1 a1W2 a1b2 ε a1g2 a1t2 s1b0
  let s1b2 := idBlockGraphPC "s1b2" epsStr a2W1 a2b1 ε a2g1 a2t1 a2W2 a2b2 ε a2g2 a2t2 s1b1
  let d2 := downBlockGraphPC "d2" epsStr d2W1 d2b1 ε d2g1 d2t1 d2W2 d2b2 ε d2g2 d2t2 d2Wp d2bp ε d2gp d2tp s1b2
  let s2b0 := idBlockGraphPC "s2b0" epsStr b0W1 b0b1 ε b0g1 b0t1 b0W2 b0b2 ε b0g2 b0t2 d2
  let s2b1 := idBlockGraphPC "s2b1" epsStr b1W1 b1b1 ε b1g1 b1t1 b1W2 b1b2 ε b1g2 b1t2 s2b0
  let s2b2 := idBlockGraphPC "s2b2" epsStr b2W1 b2b1 ε b2g1 b2t1 b2W2 b2b2 ε b2g2 b2t2 s2b1
  let d3 := downBlockGraphPC "d3" epsStr d3W1 d3b1 ε d3g1 d3t1 d3W2 d3b2 ε d3g2 d3t2 d3Wp d3bp ε d3gp d3tp s2b2
  let s3b0 := idBlockGraphPC "s3b0" epsStr c0W1 c0b1 ε c0g1 c0t1 c0W2 c0b2 ε c0g2 c0t2 d3
  let s3b1 := idBlockGraphPC "s3b1" epsStr c1W1 c1b1 ε c1g1 c1t1 c1W2 c1b2 ε c1g2 c1t2 s3b0
  let s3b2 := idBlockGraphPC "s3b2" epsStr c2W1 c2b1 ε c2g1 c2t1 c2W2 c2b2 ε c2g2 c2t2 s3b1
  let s3b3 := idBlockGraphPC "s3b3" epsStr c3W1 c3b1 ε c3g1 c3t1 c3W2 c3b2 ε c3g2 c3t2 s3b2
  let s3b4 := idBlockGraphPC "s3b4" epsStr c4W1 c4b1 ε c4g1 c4t1 c4W2 c4b2 ε c4g2 c4t2 s3b3
  let d4 := downBlockGraphPC "d4" epsStr d4W1 d4b1 ε d4g1 d4t1 d4W2 d4b2 ε d4g2 d4t2 d4Wp d4bp ε d4gp d4tp s3b4
  let s4b0 := idBlockGraphPC "s4b0" epsStr e0W1 e0b1 ε e0g1 e0t1 e0W2 e0b2 ε e0g2 e0t2 d4
  let s4b1 := idBlockGraphPC "s4b1" epsStr e1W1 e1b1 ε e1g1 e1t1 e1W2 e1b2 ε e1g2 e1t2 s4b0
  denseF "%Wd" "%bd" Wd bd (.gapF (c := 512) (h := 7) (w := 7) s4b1)

/-- **Full per-channel-BN ResNet-34 forward faithfulness.** `den (resnet34FwdGraphFullPC …) =
    resnet34Forward_full_pc …`, chaining the per-block faithful lemmas + stem/maxpool/GAP/dense. The
    "text = render of a proven graph" forward half at the render's per-channel BN. -/
theorem resnet34FwdGraphFullPC_faithful (epsStr : String) (ε : ℝ)
    (Ws : Kernel4 64 3 7 7) (bs : Vec 64) (γs βs : Vec 64)
    (a0W1 : Kernel4 64 64 3 3) (a0b1 : Vec 64) (a0g1 a0t1 : Vec 64) (a0W2 : Kernel4 64 64 3 3) (a0b2 : Vec 64) (a0g2 a0t2 : Vec 64)
    (a1W1 : Kernel4 64 64 3 3) (a1b1 : Vec 64) (a1g1 a1t1 : Vec 64) (a1W2 : Kernel4 64 64 3 3) (a1b2 : Vec 64) (a1g2 a1t2 : Vec 64)
    (a2W1 : Kernel4 64 64 3 3) (a2b1 : Vec 64) (a2g1 a2t1 : Vec 64) (a2W2 : Kernel4 64 64 3 3) (a2b2 : Vec 64) (a2g2 a2t2 : Vec 64)
    (d2W1 : Kernel4 128 64 3 3) (d2b1 : Vec 128) (d2g1 d2t1 : Vec 128) (d2W2 : Kernel4 128 128 3 3) (d2b2 : Vec 128) (d2g2 d2t2 : Vec 128) (d2Wp : Kernel4 128 64 3 3) (d2bp : Vec 128) (d2gp d2tp : Vec 128)
    (b0W1 : Kernel4 128 128 3 3) (b0b1 : Vec 128) (b0g1 b0t1 : Vec 128) (b0W2 : Kernel4 128 128 3 3) (b0b2 : Vec 128) (b0g2 b0t2 : Vec 128)
    (b1W1 : Kernel4 128 128 3 3) (b1b1 : Vec 128) (b1g1 b1t1 : Vec 128) (b1W2 : Kernel4 128 128 3 3) (b1b2 : Vec 128) (b1g2 b1t2 : Vec 128)
    (b2W1 : Kernel4 128 128 3 3) (b2b1 : Vec 128) (b2g1 b2t1 : Vec 128) (b2W2 : Kernel4 128 128 3 3) (b2b2 : Vec 128) (b2g2 b2t2 : Vec 128)
    (d3W1 : Kernel4 256 128 3 3) (d3b1 : Vec 256) (d3g1 d3t1 : Vec 256) (d3W2 : Kernel4 256 256 3 3) (d3b2 : Vec 256) (d3g2 d3t2 : Vec 256) (d3Wp : Kernel4 256 128 3 3) (d3bp : Vec 256) (d3gp d3tp : Vec 256)
    (c0W1 : Kernel4 256 256 3 3) (c0b1 : Vec 256) (c0g1 c0t1 : Vec 256) (c0W2 : Kernel4 256 256 3 3) (c0b2 : Vec 256) (c0g2 c0t2 : Vec 256)
    (c1W1 : Kernel4 256 256 3 3) (c1b1 : Vec 256) (c1g1 c1t1 : Vec 256) (c1W2 : Kernel4 256 256 3 3) (c1b2 : Vec 256) (c1g2 c1t2 : Vec 256)
    (c2W1 : Kernel4 256 256 3 3) (c2b1 : Vec 256) (c2g1 c2t1 : Vec 256) (c2W2 : Kernel4 256 256 3 3) (c2b2 : Vec 256) (c2g2 c2t2 : Vec 256)
    (c3W1 : Kernel4 256 256 3 3) (c3b1 : Vec 256) (c3g1 c3t1 : Vec 256) (c3W2 : Kernel4 256 256 3 3) (c3b2 : Vec 256) (c3g2 c3t2 : Vec 256)
    (c4W1 : Kernel4 256 256 3 3) (c4b1 : Vec 256) (c4g1 c4t1 : Vec 256) (c4W2 : Kernel4 256 256 3 3) (c4b2 : Vec 256) (c4g2 c4t2 : Vec 256)
    (d4W1 : Kernel4 512 256 3 3) (d4b1 : Vec 512) (d4g1 d4t1 : Vec 512) (d4W2 : Kernel4 512 512 3 3) (d4b2 : Vec 512) (d4g2 d4t2 : Vec 512) (d4Wp : Kernel4 512 256 3 3) (d4bp : Vec 512) (d4gp d4tp : Vec 512)
    (e0W1 : Kernel4 512 512 3 3) (e0b1 : Vec 512) (e0g1 e0t1 : Vec 512) (e0W2 : Kernel4 512 512 3 3) (e0b2 : Vec 512) (e0g2 e0t2 : Vec 512)
    (e1W1 : Kernel4 512 512 3 3) (e1b1 : Vec 512) (e1g1 e1t1 : Vec 512) (e1W2 : Kernel4 512 512 3 3) (e1b2 : Vec 512) (e1g2 e1t2 : Vec 512)
    (Wd : Mat 512 10) (bd : Vec 10) (x : Vec (3 * 224 * 224)) :
    den (resnet34FwdGraphFullPC epsStr ε Ws bs γs βs a0W1 a0b1 a0g1 a0t1 a0W2 a0b2 a0g2 a0t2 a1W1 a1b1 a1g1 a1t1 a1W2 a1b2 a1g2 a1t2 a2W1 a2b1 a2g1 a2t1 a2W2 a2b2 a2g2 a2t2 d2W1 d2b1 d2g1 d2t1 d2W2 d2b2 d2g2 d2t2 d2Wp d2bp d2gp d2tp b0W1 b0b1 b0g1 b0t1 b0W2 b0b2 b0g2 b0t2 b1W1 b1b1 b1g1 b1t1 b1W2 b1b2 b1g2 b1t2 b2W1 b2b1 b2g1 b2t1 b2W2 b2b2 b2g2 b2t2 d3W1 d3b1 d3g1 d3t1 d3W2 d3b2 d3g2 d3t2 d3Wp d3bp d3gp d3tp c0W1 c0b1 c0g1 c0t1 c0W2 c0b2 c0g2 c0t2 c1W1 c1b1 c1g1 c1t1 c1W2 c1b2 c1g2 c1t2 c2W1 c2b1 c2g1 c2t1 c2W2 c2b2 c2g2 c2t2 c3W1 c3b1 c3g1 c3t1 c3W2 c3b2 c3g2 c3t2 c4W1 c4b1 c4g1 c4t1 c4W2 c4b2 c4g2 c4t2 d4W1 d4b1 d4g1 d4t1 d4W2 d4b2 d4g2 d4t2 d4Wp d4bp d4gp d4tp e0W1 e0b1 e0g1 e0t1 e0W2 e0b2 e0g2 e0t2 e1W1 e1b1 e1g1 e1t1 e1W2 e1b2 e1g2 e1t2 Wd bd x)
      = resnet34Forward_full_pc ε Ws bs γs βs a0W1 a0b1 a0g1 a0t1 a0W2 a0b2 a0g2 a0t2 a1W1 a1b1 a1g1 a1t1 a1W2 a1b2 a1g2 a1t2 a2W1 a2b1 a2g1 a2t1 a2W2 a2b2 a2g2 a2t2 d2W1 d2b1 d2g1 d2t1 d2W2 d2b2 d2g2 d2t2 d2Wp d2bp d2gp d2tp b0W1 b0b1 b0g1 b0t1 b0W2 b0b2 b0g2 b0t2 b1W1 b1b1 b1g1 b1t1 b1W2 b1b2 b1g2 b1t2 b2W1 b2b1 b2g1 b2t1 b2W2 b2b2 b2g2 b2t2 d3W1 d3b1 d3g1 d3t1 d3W2 d3b2 d3g2 d3t2 d3Wp d3bp d3gp d3tp c0W1 c0b1 c0g1 c0t1 c0W2 c0b2 c0g2 c0t2 c1W1 c1b1 c1g1 c1t1 c1W2 c1b2 c1g2 c1t2 c2W1 c2b1 c2g1 c2t1 c2W2 c2b2 c2g2 c2t2 c3W1 c3b1 c3g1 c3t1 c3W2 c3b2 c3g2 c3t2 c4W1 c4b1 c4g1 c4t1 c4W2 c4b2 c4g2 c4t2 d4W1 d4b1 d4g1 d4t1 d4W2 d4b2 d4g2 d4t2 d4Wp d4bp d4gp d4tp e0W1 e0b1 e0g1 e0t1 e0W2 e0b2 e0g2 e0t2 e1W1 e1b1 e1g1 e1t1 e1W2 e1b2 e1g2 e1t2 Wd bd x := by
  simp only [resnet34FwdGraphFullPC, resnet34Forward_full_pc, idFwd, downFwd, cbrStridedPC,
             idBlockGraphPC_faithful, downBlockGraphPC_faithful,
             reluF_faithful, bnPerChannelF_faithful, flatConvStridedF_faithful,
             maxPoolF_faithful, gapF_faithful, denseF_faithful, den_operand, Function.comp_apply]

end StableHLO
end Proofs

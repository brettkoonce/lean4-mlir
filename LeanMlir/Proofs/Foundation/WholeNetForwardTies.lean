import LeanMlir.Proofs.Resnet34WholeFloatBridge
import LeanMlir.Proofs.MobileNetV2WholeFloatBridge
import LeanMlir.Proofs.ConvNeXtWholeFloatBridge

/-! # The skeleton ↔ real-net forward ties (item #5 — cosmetic polish)

Each whole-net forward float bridge (`r34_floatBridges` / `mnv2Forward_floatBridges` /
`convnext_floatBridges`) is stated on a fresh structural skeleton (`r34Forward` / `mnv2Forward` /
`convnextForward`) whose repeated blocks are supplied as abstract `FloatBridges` arguments. These
lemmas tie each skeleton back to **the** committed real ℝ-forward def by plugging the concrete blocks
into the skeleton's abstract slots — so the bridges provably apply to the actual nets, not just
look-alike skeletons.

All three are pure definitional unfolds (`rfl`, modulo the nested-application `convNextForwardT` which
goes through its `_eq_chain` to dodge the kernel blow-up). EfficientNet already ties directly (its
bridge is stated on the `∘`-form that IS `efficientnetForwardB`). ViT's tie is NOT a definitional
unfold — it decomposes `vit_body = (per-token finalLN) ∘ transformerTower` into the flattened
`perRowFlat finalLN ∘ towerBack blocks`, reconciling the tower's `Nat.rec` fold with the
`towerBack`/`List.replicate` fold via `Function.iterate`; it lives next to `vitForwardFlat` as
`vit_full_eq_vitForwardFlat` (`ViTWholeFloatBridge.lean`), completing the 5-net forward tie sweep. The
backward skeletons' "real def" is the certified VJP (the §B work, done for r34), a genuinely different
tie than these forward net-def unfolds.
-/

namespace Proofs

open scoped Real

/-- **The ConvNeXt-T forward tie.** The committed `convNextForwardT` (nested-application form) equals
    the `convnextForward` skeleton with the stem/head LayerNorms, the 4 stages (`convNextStageK`), and
    the 3 downsamples (`cnxDownW`) plugged into its supplied slots. Through `convNextForwardT_eq_chain`
    (the kernel-safe `∘`-form), then `rfl`. -/
theorem convNextForwardT_eq_skeleton (w : CnxTWeights) :
    convNextForwardT w = convnextForward w.sW w.sb w.Wd w.bd
      (layerNormForward (96 * 56 * 56) w.sε w.sγ w.sβ)
      (layerNormForward 768 w.hε w.hγ w.hβ)
      (convNextStageK 3 w.s1) (cnxDownW 28 28 w.d1)
      (convNextStageK 3 w.s2) (cnxDownW 14 14 w.d2)
      (convNextStageK 9 w.s3) (cnxDownW 7 7 w.d3)
      (convNextStageK 3 w.s4) := by
  funext x
  rw [convNextForwardT_eq_chain w x]
  rfl

/-- **The MobileNetV2 forward tie.** The committed `mobilenetv2Forward_full_pc` (the ch7 6-block
    per-channel render) equals the `mnv2Forward` skeleton with the stem/head per-channel BNs and the 6
    inverted-residual blocks (`invresBodyStridedPC` / `residual (invresBodyPC)`) plugged into its
    supplied slots. Pure `∘`-chain `rfl`. -/
theorem mobilenetv2Forward_full_pc_eq_skeleton
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
    mobilenetv2Forward_full_pc Ws bs εs γs βs We1 be1 εe1 γe1 βe1 Wd1 bd1 εd1 γd1 βd1
        Wp1 bp1 εp1 γp1 βp1 We2 be2 εe2 γe2 βe2 Wd2 bd2 εd2 γd2 βd2 Wp2 bp2 εp2 γp2 βp2
        We3 be3 εe3 γe3 βe3 Wd3 bd3 εd3 γd3 βd3 Wp3 bp3 εp3 γp3 βp3 We4 be4 εe4 γe4 βe4
        Wd4 bd4 εd4 γd4 βd4 Wp4 bp4 εp4 γp4 βp4 We5 be5 εe5 γe5 βe5 Wd5 bd5 εd5 γd5 βd5
        Wp5 bp5 εp5 γp5 βp5 We6 be6 εe6 γe6 βe6 Wd6 bd6 εd6 γd6 βd6 Wp6 bp6 εp6 γp6 βp6
        Wh bh εh γh βh Wfc bfc
      = mnv2Forward Ws bs Wh bh Wfc bfc
          (bnPerChannelTensor3 16 112 112 εs γs βs)
          (bnPerChannelTensor3 128 7 7 εh γh βh)
          (invresBodyStridedPC (h := 56) (w := 56)
            We1 be1 εe1 γe1 βe1 Wd1 bd1 εd1 γd1 βd1 Wp1 bp1 εp1 γp1 βp1)
          (residual (invresBodyPC (h := 56) (w := 56)
            We2 be2 εe2 γe2 βe2 Wd2 bd2 εd2 γd2 βd2 Wp2 bp2 εp2 γp2 βp2))
          (invresBodyStridedPC (h := 28) (w := 28)
            We3 be3 εe3 γe3 βe3 Wd3 bd3 εd3 γd3 βd3 Wp3 bp3 εp3 γp3 βp3)
          (residual (invresBodyPC (h := 28) (w := 28)
            We4 be4 εe4 γe4 βe4 Wd4 bd4 εd4 γd4 βd4 Wp4 bp4 εp4 γp4 βp4))
          (invresBodyStridedPC (h := 14) (w := 14)
            We5 be5 εe5 γe5 βe5 Wd5 bd5 εd5 γd5 βd5 Wp5 bp5 εp5 γp5 βp5)
          (invresBodyStridedPC (h := 7) (w := 7)
            We6 be6 εe6 γe6 βe6 Wd6 bd6 εd6 γd6 βd6 Wp6 bp6 εp6 γp6 βp6) := by
  rfl

/-- **The ResNet-34 forward tie.** The committed `resnet34Forward_full_pc` (the ch6 [3,4,6,3]
    per-channel render) equals the `r34Forward` skeleton with the stem per-channel BN and the 16 basic
    blocks (`idFwd` = `rblkPC` / `downFwd` = `rblkPStridedPC`) plugged into its supplied slots. Pure
    `∘`-chain `rfl` (the stem `cbrStridedPC = relu ∘ bnPC ∘ flatConvStride2` matches with the supplied
    `bnPerChannelTensor3`). -/
theorem resnet34Forward_full_pc_eq_skeleton (ε : ℝ)
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
    resnet34Forward_full_pc ε Ws bs γs βs
        a0W1 a0b1 a0g1 a0t1 a0W2 a0b2 a0g2 a0t2 a1W1 a1b1 a1g1 a1t1 a1W2 a1b2 a1g2 a1t2
        a2W1 a2b1 a2g1 a2t1 a2W2 a2b2 a2g2 a2t2 d2W1 d2b1 d2g1 d2t1 d2W2 d2b2 d2g2 d2t2 d2Wp d2bp d2gp d2tp
        b0W1 b0b1 b0g1 b0t1 b0W2 b0b2 b0g2 b0t2 b1W1 b1b1 b1g1 b1t1 b1W2 b1b2 b1g2 b1t2
        b2W1 b2b1 b2g1 b2t1 b2W2 b2b2 b2g2 b2t2 d3W1 d3b1 d3g1 d3t1 d3W2 d3b2 d3g2 d3t2 d3Wp d3bp d3gp d3tp
        c0W1 c0b1 c0g1 c0t1 c0W2 c0b2 c0g2 c0t2 c1W1 c1b1 c1g1 c1t1 c1W2 c1b2 c1g2 c1t2
        c2W1 c2b1 c2g1 c2t1 c2W2 c2b2 c2g2 c2t2 c3W1 c3b1 c3g1 c3t1 c3W2 c3b2 c3g2 c3t2
        c4W1 c4b1 c4g1 c4t1 c4W2 c4b2 c4g2 c4t2 d4W1 d4b1 d4g1 d4t1 d4W2 d4b2 d4g2 d4t2 d4Wp d4bp d4gp d4tp
        e0W1 e0b1 e0g1 e0t1 e0W2 e0b2 e0g2 e0t2 e1W1 e1b1 e1g1 e1t1 e1W2 e1b2 e1g2 e1t2 Wd bd
      = r34Forward Ws bs Wd bd
          (bnPerChannelTensor3 64 112 112 ε γs βs)
          (idFwd (h := 56) (w := 56) ε a0W1 a0b1 a0g1 a0t1 a0W2 a0b2 a0g2 a0t2)
          (idFwd (h := 56) (w := 56) ε a1W1 a1b1 a1g1 a1t1 a1W2 a1b2 a1g2 a1t2)
          (idFwd (h := 56) (w := 56) ε a2W1 a2b1 a2g1 a2t1 a2W2 a2b2 a2g2 a2t2)
          (downFwd (h := 28) (w := 28) ε d2W1 d2b1 d2g1 d2t1 d2W2 d2b2 d2g2 d2t2 d2Wp d2bp d2gp d2tp)
          (idFwd (h := 28) (w := 28) ε b0W1 b0b1 b0g1 b0t1 b0W2 b0b2 b0g2 b0t2)
          (idFwd (h := 28) (w := 28) ε b1W1 b1b1 b1g1 b1t1 b1W2 b1b2 b1g2 b1t2)
          (idFwd (h := 28) (w := 28) ε b2W1 b2b1 b2g1 b2t1 b2W2 b2b2 b2g2 b2t2)
          (downFwd (h := 14) (w := 14) ε d3W1 d3b1 d3g1 d3t1 d3W2 d3b2 d3g2 d3t2 d3Wp d3bp d3gp d3tp)
          (idFwd (h := 14) (w := 14) ε c0W1 c0b1 c0g1 c0t1 c0W2 c0b2 c0g2 c0t2)
          (idFwd (h := 14) (w := 14) ε c1W1 c1b1 c1g1 c1t1 c1W2 c1b2 c1g2 c1t2)
          (idFwd (h := 14) (w := 14) ε c2W1 c2b1 c2g1 c2t1 c2W2 c2b2 c2g2 c2t2)
          (idFwd (h := 14) (w := 14) ε c3W1 c3b1 c3g1 c3t1 c3W2 c3b2 c3g2 c3t2)
          (idFwd (h := 14) (w := 14) ε c4W1 c4b1 c4g1 c4t1 c4W2 c4b2 c4g2 c4t2)
          (downFwd (h := 7) (w := 7) ε d4W1 d4b1 d4g1 d4t1 d4W2 d4b2 d4g2 d4t2 d4Wp d4bp d4gp d4tp)
          (idFwd (h := 7) (w := 7) ε e0W1 e0b1 e0g1 e0t1 e0W2 e0b2 e0g2 e0t2)
          (idFwd (h := 7) (w := 7) ε e1W1 e1b1 e1g1 e1t1 e1W2 e1b2 e1g2 e1t2) := by
  rfl

end Proofs

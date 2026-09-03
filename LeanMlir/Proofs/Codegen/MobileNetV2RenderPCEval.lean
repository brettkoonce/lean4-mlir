import LeanMlir.Proofs.Codegen.MobileNetV2RenderPC

/-! # MobileNetV2 — the INFERENCE forward graph, and its faithfulness

The eval twin of `MobileNetV2RenderPC.lean`, and the mnv2 peer of `ResNet34RenderPCEval.lean`.
That file proves `den (mobilenetv2FwdGraphFullPC …) = mobilenetv2Forward_full_pc …` for the
**training** BN chain — the one the trainer differentiates. The deployed forward reads frozen
running statistics at every BN site (`.bnPerChannelEvalF`) and had no such theorem, so the
inference forward was rendered from the verified AST but not tied to an ℝ def.

⭐ Why it matters here and not only for tidiness: a whole-net float NUMBER exists only at
inference BatchNorm. Training-mode BN reduces its statistics out of its own input, so its error
modulus is quadratic in the window and the fold squares at every one of this net's 20 BN sites;
inference BN is affine in `x`, its modulus is linear, and the fold stays statable
(`planning/float_budget_numbers.md` §0.1). `MobileNetV2FloatBudget.lean` states its number about
exactly the forward defined here, and this file's faithfulness is what carries that number from
a hand-assembled skeleton to the graph the render emits.

Same rungs as the training file, one for one: the four per-channel stage abbreviations at frozen
statistics (`ivExpandPCEval` / `ivDepthwisePCEval` / `ivDepthwiseStridedPCEval` /
`ivProjectPCEval`), the two bodies, then `mobilenetv2Forward_full_pc_eval` +
`mobilenetv2FwdGraphFullPCEval` + `mobilenetv2FwdGraphFullPCEval_faithful`. Every BN node's `den`
is `bnPerChannelEvalTensor3` (`bnPerChannelEvalF_faithful`, `rfl`), so nothing new is assumed —
the eval BN op was already proven, it just had no whole-net mnv2 chain to sit in.

⚠ One ε for the whole net, as in `resnet34Forward_full_pc_eval` and as the render emits (a single
`eps` constant), where the training def carries a separate `ε` per site. Each BN site instead
grows by its frozen mean and variance, so the signature goes 102 arguments → 123.

The SSA names extend the training graph's (`%gs`/`%bts` → `%mus`/`%vars`, `%ge1`/`%bte1` →
`%mue1`/`%vare1`, …) so the typed graph and the emitted text name the same inputs. Names are
pretty-printing metadata and do not enter `den`. No new tokens. 3-axiom clean.
-/

namespace Proofs

-- ════════════════════════════════════════════════════════════════
-- § The inference stage abbreviations (eval mirrors of ivExpandPC / … / ivProjectPC)
-- ════════════════════════════════════════════════════════════════

/-- Expand stage at inference: `relu6 ∘ bnEval ∘ conv(1×1)`. -/
@[reducible] noncomputable def ivExpandPCEval {ic mid h w kHe kWe : Nat}
    (We : Kernel4 mid ic kHe kWe) (be : Vec mid) (εe : ℝ) (γe βe μe ve : Vec mid) :
    Vec (ic * h * w) → Vec (mid * h * w) :=
  relu6 (mid * h * w) ∘ bnPerChannelEvalTensor3 mid h w εe γe βe μe ve ∘ flatConv We be

/-- Depthwise stage (stride-1) at inference. -/
@[reducible] noncomputable def ivDepthwisePCEval {mid h w kHd kWd : Nat}
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (γd βd μd vd : Vec mid) :
    Vec (mid * h * w) → Vec (mid * h * w) :=
  relu6 (mid * h * w) ∘ bnPerChannelEvalTensor3 mid h w εd γd βd μd vd ∘ depthwiseFlat Wd bd

/-- Depthwise stage (stride-2 downsample) at inference. -/
@[reducible] noncomputable def ivDepthwiseStridedPCEval {mid h w kHd kWd : Nat}
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (γd βd μd vd : Vec mid) :
    Vec (mid * (2 * h) * (2 * w)) → Vec (mid * h * w) :=
  relu6 (mid * h * w) ∘ bnPerChannelEvalTensor3 mid h w εd γd βd μd vd ∘
    depthwiseStride2Flat Wd bd

/-- Project (linear bottleneck) stage at inference — no relu6. -/
@[reducible] noncomputable def ivProjectPCEval {mid oc h w kHp kWp : Nat}
    (Wp : Kernel4 oc mid kHp kWp) (bp : Vec oc) (εp : ℝ) (γp βp μp vp : Vec oc) :
    Vec (mid * h * w) → Vec (oc * h * w) :=
  bnPerChannelEvalTensor3 oc h w εp γp βp μp vp ∘ flatConv Wp bp

/-- Inverted-residual body (stride-1) at inference, at one shared ε. -/
@[reducible] noncomputable def invresBodyPCEval {ic mid oc h w : Nat} (ε : ℝ)
    (We : Kernel4 mid ic 1 1) (be : Vec mid) (γe βe μe ve : Vec mid)
    (Wd : DepthwiseKernel mid 3 3) (bd : Vec mid) (γd βd μd vd : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (γp βp μp vp : Vec oc) :
    Vec (ic * h * w) → Vec (oc * h * w) :=
  ivProjectPCEval (h := h) (w := w) Wp bp ε γp βp μp vp ∘
    (ivDepthwisePCEval (h := h) (w := w) Wd bd ε γd βd μd vd ∘
      ivExpandPCEval (h := h) (w := w) We be ε γe βe μe ve)

/-- Inverted-residual body (stride-2 downsample) at inference: expand at `2h×2w`, then the
    strided depthwise, then project. -/
@[reducible] noncomputable def invresBodyStridedPCEval {ic mid oc h w : Nat} (ε : ℝ)
    (We : Kernel4 mid ic 1 1) (be : Vec mid) (γe βe μe ve : Vec mid)
    (Wd : DepthwiseKernel mid 3 3) (bd : Vec mid) (γd βd μd vd : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (γp βp μp vp : Vec oc) :
    Vec (ic * (2 * h) * (2 * w)) → Vec (oc * h * w) :=
  ivProjectPCEval (h := h) (w := w) Wp bp ε γp βp μp vp ∘
    (ivDepthwiseStridedPCEval (h := h) (w := w) Wd bd ε γd βd μd vd ∘
      ivExpandPCEval (h := 2 * h) (w := 2 * w) We be ε γe βe μe ve)

-- ════════════════════════════════════════════════════════════════
-- § The full inference MobileNetV2 ℝ-forward (ch7 render dims)
-- ════════════════════════════════════════════════════════════════

/-- **The full inference MobileNetV2 forward** (render dims 3×224² → 7×7×64) — the ℝ map the
    deployed eval render computes, and the map `MobileNetV2FloatBudget.lean` states its number
    about. The eval twin of `mobilenetv2Forward_full_pc`, argument for argument, plus each BN
    site's frozen running mean and variance, at one shared ε. -/
noncomputable def mobilenetv2Forward_full_pc_eval (ε : ℝ)
    (Ws : Kernel4 16 3 3 3) (bs : Vec 16) (γs βs μs vs : Vec 16)
    (We1 : Kernel4 64 16 1 1) (be1 : Vec 64) (γe1 βe1 μe1 ve1 : Vec 64)
    (Wd1 : DepthwiseKernel 64 3 3) (bd1 : Vec 64) (γd1 βd1 μd1 vd1 : Vec 64)
    (Wp1 : Kernel4 24 64 1 1) (bp1 : Vec 24) (γp1 βp1 μp1 vp1 : Vec 24)
    (We2 : Kernel4 96 24 1 1) (be2 : Vec 96) (γe2 βe2 μe2 ve2 : Vec 96)
    (Wd2 : DepthwiseKernel 96 3 3) (bd2 : Vec 96) (γd2 βd2 μd2 vd2 : Vec 96)
    (Wp2 : Kernel4 24 96 1 1) (bp2 : Vec 24) (γp2 βp2 μp2 vp2 : Vec 24)
    (We3 : Kernel4 96 24 1 1) (be3 : Vec 96) (γe3 βe3 μe3 ve3 : Vec 96)
    (Wd3 : DepthwiseKernel 96 3 3) (bd3 : Vec 96) (γd3 βd3 μd3 vd3 : Vec 96)
    (Wp3 : Kernel4 32 96 1 1) (bp3 : Vec 32) (γp3 βp3 μp3 vp3 : Vec 32)
    (We4 : Kernel4 128 32 1 1) (be4 : Vec 128) (γe4 βe4 μe4 ve4 : Vec 128)
    (Wd4 : DepthwiseKernel 128 3 3) (bd4 : Vec 128) (γd4 βd4 μd4 vd4 : Vec 128)
    (Wp4 : Kernel4 32 128 1 1) (bp4 : Vec 32) (γp4 βp4 μp4 vp4 : Vec 32)
    (We5 : Kernel4 128 32 1 1) (be5 : Vec 128) (γe5 βe5 μe5 ve5 : Vec 128)
    (Wd5 : DepthwiseKernel 128 3 3) (bd5 : Vec 128) (γd5 βd5 μd5 vd5 : Vec 128)
    (Wp5 : Kernel4 64 128 1 1) (bp5 : Vec 64) (γp5 βp5 μp5 vp5 : Vec 64)
    (We6 : Kernel4 256 64 1 1) (be6 : Vec 256) (γe6 βe6 μe6 ve6 : Vec 256)
    (Wd6 : DepthwiseKernel 256 3 3) (bd6 : Vec 256) (γd6 βd6 μd6 vd6 : Vec 256)
    (Wp6 : Kernel4 64 256 1 1) (bp6 : Vec 64) (γp6 βp6 μp6 vp6 : Vec 64)
    (Wh : Kernel4 128 64 1 1) (bh : Vec 128) (γh βh μh vh : Vec 128)
    (Wfc : Mat 128 10) (bfc : Vec 10) :
    Vec (3 * 224 * 224) → Vec 10 :=
  dense Wfc bfc ∘
  globalAvgPoolFlat 128 7 7 ∘
  (relu6 (128 * 7 * 7) ∘ bnPerChannelEvalTensor3 128 7 7 ε γh βh μh vh ∘
    flatConv (h := 7) (w := 7) Wh bh) ∘
  invresBodyStridedPCEval (h := 7) (w := 7) ε We6 be6 γe6 βe6 μe6 ve6 Wd6 bd6 γd6 βd6 μd6 vd6 Wp6 bp6 γp6 βp6 μp6 vp6 ∘
  invresBodyStridedPCEval (h := 14) (w := 14) ε We5 be5 γe5 βe5 μe5 ve5 Wd5 bd5 γd5 βd5 μd5 vd5 Wp5 bp5 γp5 βp5 μp5 vp5 ∘
  residual (invresBodyPCEval (h := 28) (w := 28) ε We4 be4 γe4 βe4 μe4 ve4 Wd4 bd4 γd4 βd4 μd4 vd4 Wp4 bp4 γp4 βp4 μp4 vp4) ∘
  invresBodyStridedPCEval (h := 28) (w := 28) ε We3 be3 γe3 βe3 μe3 ve3 Wd3 bd3 γd3 βd3 μd3 vd3 Wp3 bp3 γp3 βp3 μp3 vp3 ∘
  residual (invresBodyPCEval (h := 56) (w := 56) ε We2 be2 γe2 βe2 μe2 ve2 Wd2 bd2 γd2 βd2 μd2 vd2 Wp2 bp2 γp2 βp2 μp2 vp2) ∘
  invresBodyStridedPCEval (h := 56) (w := 56) ε We1 be1 γe1 βe1 μe1 ve1 Wd1 bd1 γd1 βd1 μd1 vd1 Wp1 bp1 γp1 βp1 μp1 vp1 ∘
  (relu6 (16 * 112 * 112) ∘ bnPerChannelEvalTensor3 16 112 112 ε γs βs μs vs ∘
    flatConvStride2 (h := 112) (w := 112) Ws bs)

namespace StableHLO

-- ════════════════════════════════════════════════════════════════
-- § The full inference typed `SHlo` forward graph + faithfulness
-- ════════════════════════════════════════════════════════════════

/-- Whole **inference MobileNetV2 forward** graph at the full ch7 render dims (3×224² →
    7×7×64): strided stem → 6 inverted-residual blocks (`b1/b3/b5/b6` stride-2 downsample,
    `b2/b4` stride-1 SAME with an `addV` skip) → 1×1 conv-bn-relu6 head → global-avg-pool →
    dense, with every BN site reading frozen running statistics. The eval twin of
    `mobilenetv2FwdGraphFullPC`. -/
def mobilenetv2FwdGraphFullPCEval (epsStr : String) (ε : ℝ)
    (Ws : Kernel4 16 3 3 3) (bs : Vec 16) (γs βs μs vs : Vec 16)
    (We1 : Kernel4 64 16 1 1) (be1 : Vec 64) (γe1 βe1 μe1 ve1 : Vec 64)
    (Wd1 : DepthwiseKernel 64 3 3) (bd1 : Vec 64) (γd1 βd1 μd1 vd1 : Vec 64)
    (Wp1 : Kernel4 24 64 1 1) (bp1 : Vec 24) (γp1 βp1 μp1 vp1 : Vec 24)
    (We2 : Kernel4 96 24 1 1) (be2 : Vec 96) (γe2 βe2 μe2 ve2 : Vec 96)
    (Wd2 : DepthwiseKernel 96 3 3) (bd2 : Vec 96) (γd2 βd2 μd2 vd2 : Vec 96)
    (Wp2 : Kernel4 24 96 1 1) (bp2 : Vec 24) (γp2 βp2 μp2 vp2 : Vec 24)
    (We3 : Kernel4 96 24 1 1) (be3 : Vec 96) (γe3 βe3 μe3 ve3 : Vec 96)
    (Wd3 : DepthwiseKernel 96 3 3) (bd3 : Vec 96) (γd3 βd3 μd3 vd3 : Vec 96)
    (Wp3 : Kernel4 32 96 1 1) (bp3 : Vec 32) (γp3 βp3 μp3 vp3 : Vec 32)
    (We4 : Kernel4 128 32 1 1) (be4 : Vec 128) (γe4 βe4 μe4 ve4 : Vec 128)
    (Wd4 : DepthwiseKernel 128 3 3) (bd4 : Vec 128) (γd4 βd4 μd4 vd4 : Vec 128)
    (Wp4 : Kernel4 32 128 1 1) (bp4 : Vec 32) (γp4 βp4 μp4 vp4 : Vec 32)
    (We5 : Kernel4 128 32 1 1) (be5 : Vec 128) (γe5 βe5 μe5 ve5 : Vec 128)
    (Wd5 : DepthwiseKernel 128 3 3) (bd5 : Vec 128) (γd5 βd5 μd5 vd5 : Vec 128)
    (Wp5 : Kernel4 64 128 1 1) (bp5 : Vec 64) (γp5 βp5 μp5 vp5 : Vec 64)
    (We6 : Kernel4 256 64 1 1) (be6 : Vec 256) (γe6 βe6 μe6 ve6 : Vec 256)
    (Wd6 : DepthwiseKernel 256 3 3) (bd6 : Vec 256) (γd6 βd6 μd6 vd6 : Vec 256)
    (Wp6 : Kernel4 64 256 1 1) (bp6 : Vec 64) (γp6 βp6 μp6 vp6 : Vec 64)
    (Wh : Kernel4 128 64 1 1) (bh : Vec 128) (γh βh μh vh : Vec 128)
    (Wfc : Mat 128 10) (bfc : Vec 10)
    (x : Vec (3 * 224 * 224)) : SHlo 10 :=
  let stemOut : SHlo (16 * 112 * 112) :=
    .relu6F (.bnPerChannelEvalF (oc := 16) (h := 112) (w := 112) "%gs" "%bts"
      "%mus" "%vars" epsStr ε γs βs μs vs
      (.flatConvStridedF (h := 112) (w := 112) "%Ws" "%bs" Ws bs (.operand "%x" x)))
  let b1Out : SHlo (24 * 56 * 56) :=
    .bnPerChannelEvalF (oc := 24) (h := 56) (w := 56) "%gp1" "%btp1"
      "%mup1" "%varp1" epsStr ε γp1 βp1 μp1 vp1
      (.flatConvF (h := 56) (w := 56) "%Wp1" "%bp1" Wp1 bp1
        (.relu6F (.bnPerChannelEvalF (oc := 64) (h := 56) (w := 56) "%gd1" "%btd1"
          "%mud1" "%vard1" epsStr ε γd1 βd1 μd1 vd1
          (.depthwiseStridedF (h := 56) (w := 56) "%Wd1" "%bd1" Wd1 bd1
            (.relu6F (.bnPerChannelEvalF (oc := 64) (h := 112) (w := 112) "%ge1" "%bte1"
              "%mue1" "%vare1" epsStr ε γe1 βe1 μe1 ve1
              (.flatConvF (h := 112) (w := 112) "%We1" "%be1" We1 be1 stemOut)))))))
  let b2Out : SHlo (24 * 56 * 56) :=
    .addV (.bnPerChannelEvalF (oc := 24) (h := 56) (w := 56) "%gp2" "%btp2"
      "%mup2" "%varp2" epsStr ε γp2 βp2 μp2 vp2
      (.flatConvF (h := 56) (w := 56) "%Wp2" "%bp2" Wp2 bp2
        (.relu6F (.bnPerChannelEvalF (oc := 96) (h := 56) (w := 56) "%gd2" "%btd2"
          "%mud2" "%vard2" epsStr ε γd2 βd2 μd2 vd2
          (.depthwiseF (h := 56) (w := 56) "%Wd2" "%bd2" Wd2 bd2
            (.relu6F (.bnPerChannelEvalF (oc := 96) (h := 56) (w := 56) "%ge2" "%bte2"
              "%mue2" "%vare2" epsStr ε γe2 βe2 μe2 ve2
              (.flatConvF (h := 56) (w := 56) "%We2" "%be2" We2 be2 b1Out)))))))) b1Out
  let b3Out : SHlo (32 * 28 * 28) :=
    .bnPerChannelEvalF (oc := 32) (h := 28) (w := 28) "%gp3" "%btp3"
      "%mup3" "%varp3" epsStr ε γp3 βp3 μp3 vp3
      (.flatConvF (h := 28) (w := 28) "%Wp3" "%bp3" Wp3 bp3
        (.relu6F (.bnPerChannelEvalF (oc := 96) (h := 28) (w := 28) "%gd3" "%btd3"
          "%mud3" "%vard3" epsStr ε γd3 βd3 μd3 vd3
          (.depthwiseStridedF (h := 28) (w := 28) "%Wd3" "%bd3" Wd3 bd3
            (.relu6F (.bnPerChannelEvalF (oc := 96) (h := 56) (w := 56) "%ge3" "%bte3"
              "%mue3" "%vare3" epsStr ε γe3 βe3 μe3 ve3
              (.flatConvF (h := 56) (w := 56) "%We3" "%be3" We3 be3 b2Out)))))))
  let b4Out : SHlo (32 * 28 * 28) :=
    .addV (.bnPerChannelEvalF (oc := 32) (h := 28) (w := 28) "%gp4" "%btp4"
      "%mup4" "%varp4" epsStr ε γp4 βp4 μp4 vp4
      (.flatConvF (h := 28) (w := 28) "%Wp4" "%bp4" Wp4 bp4
        (.relu6F (.bnPerChannelEvalF (oc := 128) (h := 28) (w := 28) "%gd4" "%btd4"
          "%mud4" "%vard4" epsStr ε γd4 βd4 μd4 vd4
          (.depthwiseF (h := 28) (w := 28) "%Wd4" "%bd4" Wd4 bd4
            (.relu6F (.bnPerChannelEvalF (oc := 128) (h := 28) (w := 28) "%ge4" "%bte4"
              "%mue4" "%vare4" epsStr ε γe4 βe4 μe4 ve4
              (.flatConvF (h := 28) (w := 28) "%We4" "%be4" We4 be4 b3Out)))))))) b3Out
  let b5Out : SHlo (64 * 14 * 14) :=
    .bnPerChannelEvalF (oc := 64) (h := 14) (w := 14) "%gp5" "%btp5"
      "%mup5" "%varp5" epsStr ε γp5 βp5 μp5 vp5
      (.flatConvF (h := 14) (w := 14) "%Wp5" "%bp5" Wp5 bp5
        (.relu6F (.bnPerChannelEvalF (oc := 128) (h := 14) (w := 14) "%gd5" "%btd5"
          "%mud5" "%vard5" epsStr ε γd5 βd5 μd5 vd5
          (.depthwiseStridedF (h := 14) (w := 14) "%Wd5" "%bd5" Wd5 bd5
            (.relu6F (.bnPerChannelEvalF (oc := 128) (h := 28) (w := 28) "%ge5" "%bte5"
              "%mue5" "%vare5" epsStr ε γe5 βe5 μe5 ve5
              (.flatConvF (h := 28) (w := 28) "%We5" "%be5" We5 be5 b4Out)))))))
  let b6Out : SHlo (64 * 7 * 7) :=
    .bnPerChannelEvalF (oc := 64) (h := 7) (w := 7) "%gp6" "%btp6"
      "%mup6" "%varp6" epsStr ε γp6 βp6 μp6 vp6
      (.flatConvF (h := 7) (w := 7) "%Wp6" "%bp6" Wp6 bp6
        (.relu6F (.bnPerChannelEvalF (oc := 256) (h := 7) (w := 7) "%gd6" "%btd6"
          "%mud6" "%vard6" epsStr ε γd6 βd6 μd6 vd6
          (.depthwiseStridedF (h := 7) (w := 7) "%Wd6" "%bd6" Wd6 bd6
            (.relu6F (.bnPerChannelEvalF (oc := 256) (h := 14) (w := 14) "%ge6" "%bte6"
              "%mue6" "%vare6" epsStr ε γe6 βe6 μe6 ve6
              (.flatConvF (h := 14) (w := 14) "%We6" "%be6" We6 be6 b5Out)))))))
  let headOut : SHlo (128 * 7 * 7) :=
    .relu6F (.bnPerChannelEvalF (oc := 128) (h := 7) (w := 7) "%gh" "%bth"
      "%muh" "%varh" epsStr ε γh βh μh vh
      (.flatConvF (h := 7) (w := 7) "%Wh" "%bh" Wh bh b6Out))
  denseF "%Wfc" "%bfc" Wfc bfc (.gapF (c := 128) (h := 7) (w := 7) headOut)

set_option maxRecDepth 10000 in
/-- ⭐ **Full inference MobileNetV2 forward faithfulness.**
    `den (mobilenetv2FwdGraphFullPCEval …) = mobilenetv2Forward_full_pc_eval …` — the eval
    half of "text = render of a proven graph" for this net, and the tie
    `MobileNetV2FloatBudget.lean`'s number needs. Same `simp`-then-`unfold` recipe as
    `mobilenetv2FwdGraphFullPC_faithful`, with `bnPerChannelEvalF_faithful` in place of
    `bnPerChannelF_faithful`. -/
theorem mobilenetv2FwdGraphFullPCEval_faithful (epsStr : String) (ε : ℝ)
    (Ws : Kernel4 16 3 3 3) (bs : Vec 16) (γs βs μs vs : Vec 16)
    (We1 : Kernel4 64 16 1 1) (be1 : Vec 64) (γe1 βe1 μe1 ve1 : Vec 64)
    (Wd1 : DepthwiseKernel 64 3 3) (bd1 : Vec 64) (γd1 βd1 μd1 vd1 : Vec 64)
    (Wp1 : Kernel4 24 64 1 1) (bp1 : Vec 24) (γp1 βp1 μp1 vp1 : Vec 24)
    (We2 : Kernel4 96 24 1 1) (be2 : Vec 96) (γe2 βe2 μe2 ve2 : Vec 96)
    (Wd2 : DepthwiseKernel 96 3 3) (bd2 : Vec 96) (γd2 βd2 μd2 vd2 : Vec 96)
    (Wp2 : Kernel4 24 96 1 1) (bp2 : Vec 24) (γp2 βp2 μp2 vp2 : Vec 24)
    (We3 : Kernel4 96 24 1 1) (be3 : Vec 96) (γe3 βe3 μe3 ve3 : Vec 96)
    (Wd3 : DepthwiseKernel 96 3 3) (bd3 : Vec 96) (γd3 βd3 μd3 vd3 : Vec 96)
    (Wp3 : Kernel4 32 96 1 1) (bp3 : Vec 32) (γp3 βp3 μp3 vp3 : Vec 32)
    (We4 : Kernel4 128 32 1 1) (be4 : Vec 128) (γe4 βe4 μe4 ve4 : Vec 128)
    (Wd4 : DepthwiseKernel 128 3 3) (bd4 : Vec 128) (γd4 βd4 μd4 vd4 : Vec 128)
    (Wp4 : Kernel4 32 128 1 1) (bp4 : Vec 32) (γp4 βp4 μp4 vp4 : Vec 32)
    (We5 : Kernel4 128 32 1 1) (be5 : Vec 128) (γe5 βe5 μe5 ve5 : Vec 128)
    (Wd5 : DepthwiseKernel 128 3 3) (bd5 : Vec 128) (γd5 βd5 μd5 vd5 : Vec 128)
    (Wp5 : Kernel4 64 128 1 1) (bp5 : Vec 64) (γp5 βp5 μp5 vp5 : Vec 64)
    (We6 : Kernel4 256 64 1 1) (be6 : Vec 256) (γe6 βe6 μe6 ve6 : Vec 256)
    (Wd6 : DepthwiseKernel 256 3 3) (bd6 : Vec 256) (γd6 βd6 μd6 vd6 : Vec 256)
    (Wp6 : Kernel4 64 256 1 1) (bp6 : Vec 64) (γp6 βp6 μp6 vp6 : Vec 64)
    (Wh : Kernel4 128 64 1 1) (bh : Vec 128) (γh βh μh vh : Vec 128)
    (Wfc : Mat 128 10) (bfc : Vec 10)
    (x : Vec (3 * 224 * 224)) :
    den (mobilenetv2FwdGraphFullPCEval epsStr ε Ws bs γs βs μs vs We1 be1 γe1 βe1 μe1 ve1 Wd1 bd1 γd1 βd1 μd1 vd1 Wp1 bp1 γp1 βp1 μp1 vp1 We2 be2 γe2 βe2 μe2 ve2 Wd2 bd2 γd2 βd2 μd2 vd2 Wp2 bp2 γp2 βp2 μp2 vp2 We3 be3 γe3 βe3 μe3 ve3 Wd3 bd3 γd3 βd3 μd3 vd3 Wp3 bp3 γp3 βp3 μp3 vp3 We4 be4 γe4 βe4 μe4 ve4 Wd4 bd4 γd4 βd4 μd4 vd4 Wp4 bp4 γp4 βp4 μp4 vp4 We5 be5 γe5 βe5 μe5 ve5 Wd5 bd5 γd5 βd5 μd5 vd5 Wp5 bp5 γp5 βp5 μp5 vp5 We6 be6 γe6 βe6 μe6 ve6 Wd6 bd6 γd6 βd6 μd6 vd6 Wp6 bp6 γp6 βp6 μp6 vp6 Wh bh γh βh μh vh Wfc bfc x)
      = mobilenetv2Forward_full_pc_eval ε Ws bs γs βs μs vs We1 be1 γe1 βe1 μe1 ve1 Wd1 bd1 γd1 βd1 μd1 vd1 Wp1 bp1 γp1 βp1 μp1 vp1 We2 be2 γe2 βe2 μe2 ve2 Wd2 bd2 γd2 βd2 μd2 vd2 Wp2 bp2 γp2 βp2 μp2 vp2 We3 be3 γe3 βe3 μe3 ve3 Wd3 bd3 γd3 βd3 μd3 vd3 Wp3 bp3 γp3 βp3 μp3 vp3 We4 be4 γe4 βe4 μe4 ve4 Wd4 bd4 γd4 βd4 μd4 vd4 Wp4 bp4 γp4 βp4 μp4 vp4 We5 be5 γe5 βe5 μe5 ve5 Wd5 bd5 γd5 βd5 μd5 vd5 Wp5 bp5 γp5 βp5 μp5 vp5 We6 be6 γe6 βe6 μe6 ve6 Wd6 bd6 γd6 βd6 μd6 vd6 Wp6 bp6 γp6 βp6 μp6 vp6 Wh bh γh βh μh vh Wfc bfc x := by
  simp only [mobilenetv2FwdGraphFullPCEval, denseF_faithful, gapF_faithful,
             relu6F_faithful, bnPerChannelEvalF_faithful, flatConvF_faithful,
             flatConvStridedF_faithful, depthwiseF_faithful, depthwiseStridedF_faithful,
             den_addV, den_operand]
  unfold mobilenetv2Forward_full_pc_eval invresBodyStridedPCEval invresBodyPCEval
         ivExpandPCEval ivDepthwiseStridedPCEval ivDepthwisePCEval ivProjectPCEval
         residual biPath
  simp only [Function.comp_apply]

end StableHLO
end Proofs

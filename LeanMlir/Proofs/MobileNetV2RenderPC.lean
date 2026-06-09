import LeanMlir.Proofs.StableHLO

/-! # Item A — the PER-CHANNEL-BN MobileNetV2 forward graph (matches the render)

`planning/mobilenetv2_close.md` Item A. `StableHLO.lean` already has the full strided MobileNetV2
forward graph `mobilenetv2FwdGraphFull` + `mobilenetv2FwdGraphFull_faithful` — but those use **scalar**
`bnF` (one γ/β over the whole `c·h·w`), tied to the scalar ℝ-forward `mobilenetv2Forward_full`. The
**operational render** (`tests/TestMobilenetV2Train.lean`, the `bnPC` block) emits **per-channel** BN
(reduce over spatial `[2,3]`, `γ/β : Vec c`). So neither existing graph is a faithful "render of a
proven graph": they compute a different function than the render.

This file closes that gap — the per-channel-BN twin of `mobilenetv2FwdGraphFull`:

* **`mobilenetv2Forward_full_pc`** — the ℝ-forward with `bnPerChannelTensor3` (per-channel, `γ/β : Vec c`)
  at every BN site, same topology/stride schedule/relu6 placement as `mobilenetv2Forward_full`. Built
  from per-channel stage abbreviations `ivExpandPC`/`ivDepthwisePC`/`ivDepthwiseStridedPC`/`ivProjectPC`
  (per-channel mirrors of `ivExpand`/…/`ivProject`).
* **`mobilenetv2FwdGraphFullPC`** — the typed `SHlo` forward graph using `bnPerChannelF` tokens.
* **`mobilenetv2FwdGraphFullPC_faithful`** — `den (graph) = mobilenetv2Forward_full_pc`, via
  `bnPerChannelF_faithful` (`den (bnPerChannelF …) = bnPerChannelTensor3 …`, an `rfl` lemma). Same
  `simp`-then-`unfold` recipe as `mobilenetv2FwdGraphFull_faithful`.

This is the prerequisite for the structured render (Item B): now MobileNetV2 has a *proven* per-channel
forward graph whose `pretty` matches the render's forward text. Everything closes 3-axiom-clean
(`tests/AuditAxioms.lean`). Stride-2 `flatConvStridedF`/`depthwiseStridedF` (4 downsampling blocks) and
the residual `addV` skip (`b2/b4`) are all assembled here, not just exercised at the op level.
-/

namespace Proofs

open scoped BigOperators

-- ════════════════════════════════════════════════════════════════
-- § Per-channel inverted-residual stage abbreviations
--   (per-channel-BN mirrors of ivExpand / ivDepthwise / ivDepthwiseStrided / ivProject)
-- ════════════════════════════════════════════════════════════════

/-- Expand stage, per-channel BN: `relu6 ∘ bnPC ∘ conv(1×1)`. -/
@[reducible] noncomputable def ivExpandPC {ic mid h w kHe kWe : Nat}
    (We : Kernel4 mid ic kHe kWe) (be : Vec mid) (εe : ℝ) (γe βe : Vec mid) :
    Vec (ic * h * w) → Vec (mid * h * w) :=
  relu6 (mid * h * w) ∘ bnPerChannelTensor3 mid h w εe γe βe ∘ flatConv We be

/-- Depthwise stage (stride-1), per-channel BN: `relu6 ∘ bnPC ∘ depthwise`. -/
@[reducible] noncomputable def ivDepthwisePC {mid h w kHd kWd : Nat}
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (γd βd : Vec mid) :
    Vec (mid * h * w) → Vec (mid * h * w) :=
  relu6 (mid * h * w) ∘ bnPerChannelTensor3 mid h w εd γd βd ∘ depthwiseFlat Wd bd

/-- Depthwise stage (stride-2 downsample), per-channel BN: `relu6 ∘ bnPC ∘ depthwiseStrided`. -/
@[reducible] noncomputable def ivDepthwiseStridedPC {mid h w kHd kWd : Nat}
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (γd βd : Vec mid) :
    Vec (mid * (2 * h) * (2 * w)) → Vec (mid * h * w) :=
  relu6 (mid * h * w) ∘ bnPerChannelTensor3 mid h w εd γd βd ∘ depthwiseStride2Flat Wd bd

/-- Project (linear bottleneck) stage, per-channel BN: `bnPC ∘ conv(1×1)` (no relu6). -/
@[reducible] noncomputable def ivProjectPC {mid oc h w kHp kWp : Nat}
    (Wp : Kernel4 oc mid kHp kWp) (bp : Vec oc) (εp : ℝ) (γp βp : Vec oc) :
    Vec (mid * h * w) → Vec (oc * h * w) :=
  bnPerChannelTensor3 oc h w εp γp βp ∘ flatConv Wp bp

/-- Inverted-residual body (stride-1), per-channel BN: `project ∘ depthwise ∘ expand`. -/
@[reducible] noncomputable def invresBodyPC {ic mid oc h w kHe kWe kHd kWd kHp kWp : Nat}
    (We : Kernel4 mid ic kHe kWe) (be : Vec mid) (εe : ℝ) (γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (γd βd : Vec mid)
    (Wp : Kernel4 oc mid kHp kWp) (bp : Vec oc) (εp : ℝ) (γp βp : Vec oc) :
    Vec (ic * h * w) → Vec (oc * h * w) :=
  ivProjectPC (h := h) (w := w) Wp bp εp γp βp ∘
    (ivDepthwisePC (h := h) (w := w) Wd bd εd γd βd ∘
      ivExpandPC (h := h) (w := w) We be εe γe βe)

/-- Inverted-residual body (stride-2 downsample), per-channel BN: expand SAME (at `2h×2w`) →
    depthwise-strided → project. -/
@[reducible] noncomputable def invresBodyStridedPC {ic mid oc h w kHe kWe kHd kWd kHp kWp : Nat}
    (We : Kernel4 mid ic kHe kWe) (be : Vec mid) (εe : ℝ) (γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (γd βd : Vec mid)
    (Wp : Kernel4 oc mid kHp kWp) (bp : Vec oc) (εp : ℝ) (γp βp : Vec oc) :
    Vec (ic * (2 * h) * (2 * w)) → Vec (oc * h * w) :=
  ivProjectPC (h := h) (w := w) Wp bp εp γp βp ∘
    (ivDepthwiseStridedPC (h := h) (w := w) Wd bd εd γd βd ∘
      ivExpandPC (h := 2 * h) (w := 2 * w) We be εe γe βe)

-- ════════════════════════════════════════════════════════════════
-- § The full per-channel-BN MobileNetV2 ℝ-forward (ch7 render dims)
--   Same topology/stride/relu6 schedule as `mobilenetv2Forward_full`, PER-CHANNEL BN.
-- ════════════════════════════════════════════════════════════════

/-- The full MobileNetV2 forward with **per-channel** BN (ch7 render): strided stem (224→112) →
    6 inverted-residual blocks (`b1/b3/b5/b6` stride-2 downsample, `b2/b4` stride-1 skip) → 1×1
    conv-bn-relu6 head → global-avg-pool → dense. Per-channel-BN twin of `mobilenetv2Forward_full`;
    matches the operational render's BN flavor. -/
noncomputable def mobilenetv2Forward_full_pc
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
    Vec (3 * 224 * 224) → Vec 10 :=
  dense Wfc bfc ∘
  globalAvgPoolFlat 128 7 7 ∘
  (relu6 (128 * 7 * 7) ∘ bnPerChannelTensor3 128 7 7 εh γh βh ∘ flatConv (h := 7) (w := 7) Wh bh) ∘
  invresBodyStridedPC (h := 7) (w := 7)
    We6 be6 εe6 γe6 βe6 Wd6 bd6 εd6 γd6 βd6 Wp6 bp6 εp6 γp6 βp6 ∘
  invresBodyStridedPC (h := 14) (w := 14)
    We5 be5 εe5 γe5 βe5 Wd5 bd5 εd5 γd5 βd5 Wp5 bp5 εp5 γp5 βp5 ∘
  residual (invresBodyPC (h := 28) (w := 28)
    We4 be4 εe4 γe4 βe4 Wd4 bd4 εd4 γd4 βd4 Wp4 bp4 εp4 γp4 βp4) ∘
  invresBodyStridedPC (h := 28) (w := 28)
    We3 be3 εe3 γe3 βe3 Wd3 bd3 εd3 γd3 βd3 Wp3 bp3 εp3 γp3 βp3 ∘
  residual (invresBodyPC (h := 56) (w := 56)
    We2 be2 εe2 γe2 βe2 Wd2 bd2 εd2 γd2 βd2 Wp2 bp2 εp2 γp2 βp2) ∘
  invresBodyStridedPC (h := 56) (w := 56)
    We1 be1 εe1 γe1 βe1 Wd1 bd1 εd1 γd1 βd1 Wp1 bp1 εp1 γp1 βp1 ∘
  (relu6 (16 * 112 * 112) ∘ bnPerChannelTensor3 16 112 112 εs γs βs ∘
    flatConvStride2 (h := 112) (w := 112) Ws bs)

namespace StableHLO

-- ════════════════════════════════════════════════════════════════
-- § The typed `SHlo` per-channel-BN forward graph + faithfulness
-- ════════════════════════════════════════════════════════════════

/-- Whole **per-channel-BN MobileNetV2 forward** graph at the full ch7 render dims (3×224² →
    7×7×64): strided stem (`flatConvStridedF`, 224→112) → 6 inverted-residual blocks (`b1/b3/b5/b6`
    stride-2 downsample via `depthwiseStridedF`, `b2/b4` stride-1 SAME with an `addV` skip) → 1×1
    conv-bn-relu6 head → global-avg-pool → dense. **Per-channel** BN (`bnPerChannelF`, `γ/β : Vec c`)
    at every BN site — matches the operational render. The per-channel twin of `mobilenetv2FwdGraphFull`. -/
def mobilenetv2FwdGraphFullPC
    (epsStr : String)
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
    (Wfc : Mat 128 10) (bfc : Vec 10)
    (x : Vec (3 * 224 * 224)) : SHlo 10 :=
  let stemOut : SHlo (16 * 112 * 112) :=
    .relu6F (.bnPerChannelF (oc := 16) (h := 112) (w := 112) "%gs" "%bts" epsStr εs γs βs
      (.flatConvStridedF (h := 112) (w := 112) "%Ws" "%bs" Ws bs (.operand "%x" x)))
  let b1Out : SHlo (24 * 56 * 56) :=
    .bnPerChannelF (oc := 24) (h := 56) (w := 56) "%gp1" "%btp1" epsStr εp1 γp1 βp1
      (.flatConvF (h := 56) (w := 56) "%Wp1" "%bp1" Wp1 bp1
        (.relu6F (.bnPerChannelF (oc := 64) (h := 56) (w := 56) "%gd1" "%btd1" epsStr εd1 γd1 βd1
          (.depthwiseStridedF (h := 56) (w := 56) "%Wd1" "%bd1" Wd1 bd1
            (.relu6F (.bnPerChannelF (oc := 64) (h := 112) (w := 112) "%ge1" "%bte1" epsStr εe1 γe1 βe1
              (.flatConvF (h := 112) (w := 112) "%We1" "%be1" We1 be1 stemOut)))))))
  let b2Out : SHlo (24 * 56 * 56) :=
    .addV (.bnPerChannelF (oc := 24) (h := 56) (w := 56) "%gp2" "%btp2" epsStr εp2 γp2 βp2
      (.flatConvF (h := 56) (w := 56) "%Wp2" "%bp2" Wp2 bp2
        (.relu6F (.bnPerChannelF (oc := 96) (h := 56) (w := 56) "%gd2" "%btd2" epsStr εd2 γd2 βd2
          (.depthwiseF (h := 56) (w := 56) "%Wd2" "%bd2" Wd2 bd2
            (.relu6F (.bnPerChannelF (oc := 96) (h := 56) (w := 56) "%ge2" "%bte2" epsStr εe2 γe2 βe2
              (.flatConvF (h := 56) (w := 56) "%We2" "%be2" We2 be2 b1Out)))))))) b1Out
  let b3Out : SHlo (32 * 28 * 28) :=
    .bnPerChannelF (oc := 32) (h := 28) (w := 28) "%gp3" "%btp3" epsStr εp3 γp3 βp3
      (.flatConvF (h := 28) (w := 28) "%Wp3" "%bp3" Wp3 bp3
        (.relu6F (.bnPerChannelF (oc := 96) (h := 28) (w := 28) "%gd3" "%btd3" epsStr εd3 γd3 βd3
          (.depthwiseStridedF (h := 28) (w := 28) "%Wd3" "%bd3" Wd3 bd3
            (.relu6F (.bnPerChannelF (oc := 96) (h := 56) (w := 56) "%ge3" "%bte3" epsStr εe3 γe3 βe3
              (.flatConvF (h := 56) (w := 56) "%We3" "%be3" We3 be3 b2Out)))))))
  let b4Out : SHlo (32 * 28 * 28) :=
    .addV (.bnPerChannelF (oc := 32) (h := 28) (w := 28) "%gp4" "%btp4" epsStr εp4 γp4 βp4
      (.flatConvF (h := 28) (w := 28) "%Wp4" "%bp4" Wp4 bp4
        (.relu6F (.bnPerChannelF (oc := 128) (h := 28) (w := 28) "%gd4" "%btd4" epsStr εd4 γd4 βd4
          (.depthwiseF (h := 28) (w := 28) "%Wd4" "%bd4" Wd4 bd4
            (.relu6F (.bnPerChannelF (oc := 128) (h := 28) (w := 28) "%ge4" "%bte4" epsStr εe4 γe4 βe4
              (.flatConvF (h := 28) (w := 28) "%We4" "%be4" We4 be4 b3Out)))))))) b3Out
  let b5Out : SHlo (64 * 14 * 14) :=
    .bnPerChannelF (oc := 64) (h := 14) (w := 14) "%gp5" "%btp5" epsStr εp5 γp5 βp5
      (.flatConvF (h := 14) (w := 14) "%Wp5" "%bp5" Wp5 bp5
        (.relu6F (.bnPerChannelF (oc := 128) (h := 14) (w := 14) "%gd5" "%btd5" epsStr εd5 γd5 βd5
          (.depthwiseStridedF (h := 14) (w := 14) "%Wd5" "%bd5" Wd5 bd5
            (.relu6F (.bnPerChannelF (oc := 128) (h := 28) (w := 28) "%ge5" "%bte5" epsStr εe5 γe5 βe5
              (.flatConvF (h := 28) (w := 28) "%We5" "%be5" We5 be5 b4Out)))))))
  let b6Out : SHlo (64 * 7 * 7) :=
    .bnPerChannelF (oc := 64) (h := 7) (w := 7) "%gp6" "%btp6" epsStr εp6 γp6 βp6
      (.flatConvF (h := 7) (w := 7) "%Wp6" "%bp6" Wp6 bp6
        (.relu6F (.bnPerChannelF (oc := 256) (h := 7) (w := 7) "%gd6" "%btd6" epsStr εd6 γd6 βd6
          (.depthwiseStridedF (h := 7) (w := 7) "%Wd6" "%bd6" Wd6 bd6
            (.relu6F (.bnPerChannelF (oc := 256) (h := 14) (w := 14) "%ge6" "%bte6" epsStr εe6 γe6 βe6
              (.flatConvF (h := 14) (w := 14) "%We6" "%be6" We6 be6 b5Out)))))))
  let headOut : SHlo (128 * 7 * 7) :=
    .relu6F (.bnPerChannelF (oc := 128) (h := 7) (w := 7) "%gh" "%bth" epsStr εh γh βh
      (.flatConvF (h := 7) (w := 7) "%Wh" "%bh" Wh bh b6Out))
  denseF "%Wfc" "%bfc" Wfc bfc (.gapF (c := 128) (h := 7) (w := 7) headOut)

/-- **Full per-channel-BN MobileNetV2 forward faithfulness.** The per-channel strided render graph
    denotes the proven `mobilenetv2Forward_full_pc`. `simp`-based (like `mobilenetv2FwdGraphFull_faithful`,
    so it avoids the concrete-dim `isDefEq` wall), with `bnPerChannelF_faithful` replacing `bnF_faithful`.
    This is the "text = render of a proven graph" forward half at the render's BN flavor. -/
theorem mobilenetv2FwdGraphFullPC_faithful
    (epsStr : String)
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
    (Wfc : Mat 128 10) (bfc : Vec 10)
    (x : Vec (3 * 224 * 224)) :
    den (mobilenetv2FwdGraphFullPC epsStr Ws bs εs γs βs We1 be1 εe1 γe1 βe1 Wd1 bd1 εd1 γd1 βd1 Wp1 bp1 εp1 γp1 βp1 We2 be2 εe2 γe2 βe2 Wd2 bd2 εd2 γd2 βd2 Wp2 bp2 εp2 γp2 βp2 We3 be3 εe3 γe3 βe3 Wd3 bd3 εd3 γd3 βd3 Wp3 bp3 εp3 γp3 βp3 We4 be4 εe4 γe4 βe4 Wd4 bd4 εd4 γd4 βd4 Wp4 bp4 εp4 γp4 βp4 We5 be5 εe5 γe5 βe5 Wd5 bd5 εd5 γd5 βd5 Wp5 bp5 εp5 γp5 βp5 We6 be6 εe6 γe6 βe6 Wd6 bd6 εd6 γd6 βd6 Wp6 bp6 εp6 γp6 βp6 Wh bh εh γh βh Wfc bfc x)
      = mobilenetv2Forward_full_pc Ws bs εs γs βs We1 be1 εe1 γe1 βe1 Wd1 bd1 εd1 γd1 βd1 Wp1 bp1 εp1 γp1 βp1 We2 be2 εe2 γe2 βe2 Wd2 bd2 εd2 γd2 βd2 Wp2 bp2 εp2 γp2 βp2 We3 be3 εe3 γe3 βe3 Wd3 bd3 εd3 γd3 βd3 Wp3 bp3 εp3 γp3 βp3 We4 be4 εe4 γe4 βe4 Wd4 bd4 εd4 γd4 βd4 Wp4 bp4 εp4 γp4 βp4 We5 be5 εe5 γe5 βe5 Wd5 bd5 εd5 γd5 βd5 Wp5 bp5 εp5 γp5 βp5 We6 be6 εe6 γe6 βe6 Wd6 bd6 εd6 γd6 βd6 Wp6 bp6 εp6 γp6 βp6 Wh bh εh γh βh Wfc bfc x := by
  simp only [mobilenetv2FwdGraphFullPC, denseF_faithful, gapF_faithful, relu6F_faithful,
             bnPerChannelF_faithful, flatConvF_faithful, flatConvStridedF_faithful,
             depthwiseF_faithful, depthwiseStridedF_faithful, den_addV, den_operand]
  unfold mobilenetv2Forward_full_pc invresBodyStridedPC invresBodyPC ivExpandPC
         ivDepthwiseStridedPC ivDepthwisePC ivProjectPC residual biPath
  simp only [Function.comp_apply]

end StableHLO
end Proofs

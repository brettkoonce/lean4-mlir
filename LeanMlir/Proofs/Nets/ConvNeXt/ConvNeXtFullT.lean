import LeanMlir.Proofs.Nets.ConvNeXt.ConvNeXtChainClose
import LeanMlir.Proofs.Nets.ConvNeXt.ConvNeXtChannelLN

/-! # The full ConvNeXt-T — `[3,3,9,3]`, forward + whole-net VJP + graph + faithfulness

The real ConvNeXt-T spec, beside the two-block scalar-LN net of `ConvNeXt.lean`:

  4×4/s4 patchify stem (3→96, 224→56) → stem-LN → stage1 (3 blocks @96/56²) →
  downsample (LN + 2×2/s2 conv 96→192) → stage2 (3 @192/28²) → ds (192→384) →
  stage3 (9 @384/14²) → ds (384→768) → stage4 (3 @768/7²) → GAP → head LN → dense.

1. **Depth-k within a stage** — `CnxBlockParamsCh` bundles the block 10-tuple;
   `convNextStageChK (k) (ps : Fin k → CnxBlockParamsCh …)` folds blocks head-first with
   VJP by induction — the ViT depth-k recipe, simpler here (same-shape blocks within a stage).
2. **Downsample boundaries** — `cnxDownChW` = `flatConvStride2(2×2) ∘ channel-LN`; both VJPs existed.
3. **4×4/s4 patchify stem** — `flatConvStride4` (= decimate ∘ decimateOdd ∘ stride-1
   SAME conv, `StridedConv.lean`: the left-aligned window `x[4i..4i+3]` of the paper's
   pad-0 `Conv2d(4, s=4)`) + the `.flatConvStride4F` graph node.

GELU/LN/conv are smooth, so the whole-net VJP `convNextForwardTChHasVJP` is global, under the 23
LN positivities (1 stem + 18 block + 3 downsample + head) and no other hypothesis — as are
`efficientnetForwardBFullHasVJP` and `vitForwardKVHasVJP`. The `ConvNeXtFold`/`ConvNeXtChainClose`
param bridges are dim-generic and cover all 18 blocks verbatim; the downsample conv W/b reuse the
proven stride-2 bridges.
-/

-- Maintainer note: the scalar-LN twin of this chain (one mean and variance over the whole
-- `c·h·w` map, scalar γ/β) was deleted. `CnxBlockParams`, `cnxBlockW`, `convNextStageK`,
-- `CnxDownParams`, `cnxDownW`, `CnxTWeights`, `convNextForwardT`/`TC` and their graph section
-- were retired, not moved.

namespace Proofs

open scoped BigOperators

-- ════════════════════════════════════════════════════════════════
-- § The [3,3,9,3] chain at ConvNeXt's REAL channel LayerNorm (§2m)
-- ════════════════════════════════════════════════════════════════

/-! ConvNeXt specifies `channel_layer_norm` — `h·w` statistics per example, each over the `c`
channels at one spatial position, per-channel `[c]` affine. See `ChannelLN.lean` for the
primitive and for why Route A needs no new op and no new VJP.

The two-block scalar-LN net in `ConvNeXt.lean` (`convNextForward`, `convNextBlock`,
`convNextBlockBody`) is a different net: it backs the comparator and a book chapter. The LN
sites here are 1 stem + 18 block + 3 downsample (all channel-LN) + the head LN after GAP. -/

-- ── the block body, abstracted over its normalisation ──

/-- The ConvNeXt block body with the LN left as a parameter. Written once and instantiated at
    `chanLNTensor3`, so the channel-LN world costs one definition rather than a second copy of
    `convNextBlockBody`'s six-piece `vjpComp` chain. -/
noncomputable def cnxBodyWith {c cExp h w kH kW : Nat}
    (LN : Vec (c * h * w) → Vec (c * h * w))
    (Wdw : DepthwiseKernel c kH kW) (bdw : Vec c)
    (Wex : Kernel4 cExp c 1 1) (bex : Vec cExp)
    (Wpr : Kernel4 c cExp 1 1) (bpr : Vec c)
    (γls : Vec (c * h * w)) :
    Vec (c * h * w) → Vec (c * h * w) :=
  layerScale γls ∘
  (flatConv (h := h) (w := w) Wpr bpr) ∘
  (gelu (cExp * h * w)) ∘
  (flatConv (h := h) (w := w) Wex bex) ∘
  LN ∘
  (depthwiseFlat (h := h) (w := w) Wdw bdw)

theorem cnxBodyWith_differentiable {c cExp h w kH kW : Nat}
    {LN : Vec (c * h * w) → Vec (c * h * w)} (hLN : Differentiable ℝ LN)
    (Wdw : DepthwiseKernel c kH kW) (bdw : Vec c)
    (Wex : Kernel4 cExp c 1 1) (bex : Vec cExp)
    (Wpr : Kernel4 c cExp 1 1) (bpr : Vec c)
    (γls : Vec (c * h * w)) :
    Differentiable ℝ (cnxBodyWith LN Wdw bdw Wex bex Wpr bpr γls) := by
  unfold cnxBodyWith layerScale gelu; fun_prop

/-- The body VJP, given the LN's. Only the LN carries a hypothesis — gelu is smooth and
    conv/layerScale are linear, so this is global exactly as `convNextBlockBodyHasVJP` is. -/
noncomputable def cnxBodyWithHasVJP {c cExp h w kH kW : Nat}
    {LN : Vec (c * h * w) → Vec (c * h * w)}
    (hLN : Differentiable ℝ LN) (vLN : HasVJP LN)
    (Wdw : DepthwiseKernel c kH kW) (bdw : Vec c)
    (Wex : Kernel4 cExp c 1 1) (bex : Vec cExp)
    (Wpr : Kernel4 c cExp 1 1) (bpr : Vec c)
    (γls : Vec (c * h * w)) :
    HasVJP (cnxBodyWith LN Wdw bdw Wex bex Wpr bpr γls) := by
  unfold cnxBodyWith
  have hdw := depthwiseFlat_differentiable (h := h) (w := w) Wdw bdw
  have hex := flatConv_differentiable (h := h) (w := w) Wex bex
  have hge := gelu_differentiable (cExp * h * w)
  have hpr := flatConv_differentiable (h := h) (w := w) Wpr bpr
  have hls := layerScale_differentiable γls
  have e1 := vjpComp _ _ hdw hLN
    (depthwiseFlatHasVJP (h := h) (w := w) Wdw bdw) vLN
  have f1 := hLN.comp hdw
  have e2 := vjpComp _ _ f1 hex e1 (HasVJP3.toHasVJP (conv2dHasVJP3 Wex bex))
  have f2 := hex.comp f1
  have e3 := vjpComp _ _ f2 hge e2 (geluHasVJP (cExp * h * w))
  have f3 := hge.comp f2
  have e4 := vjpComp _ _ f3 hpr e3 (HasVJP3.toHasVJP (conv2dHasVJP3 Wpr bpr))
  have f4 := hpr.comp f3
  exact vjpComp _ _ f4 hls e4 (layerScaleHasVJP γls)

-- ── per-block params + the stage fold, at the channel LN ──

/-- One channel-LN ConvNeXt block's 10 parameters. `γn`/`βn` are `Vec c` — 2c floats per LN
    site. -/
structure CnxBlockParamsCh (c cExp h w kH kW : Nat) where
  Wdw : DepthwiseKernel c kH kW
  bdw : Vec c
  εn : ℝ
  γn : Vec c
  βn : Vec c
  Wex : Kernel4 cExp c 1 1
  bex : Vec cExp
  Wpr : Kernel4 c cExp 1 1
  bpr : Vec c
  γls : Vec c

/-- The per-channel layer-scale expanded to the flat map. -/
noncomputable def cnxGlsCh {c cExp h w kH kW : Nat} (p : CnxBlockParamsCh c cExp h w kH kW) :
    Vec (c * h * w) :=
  fun k => p.γls (StableHLO.chanIdx c h w k)

/-- The packaged ConvNeXt block: `residual` of the shared body at `chanLNTensor3`. -/
noncomputable def cnxBlockChW {c cExp h w kH kW : Nat}
    (p : CnxBlockParamsCh c cExp h w kH kW) :
    Vec (c * h * w) → Vec (c * h * w) :=
  residual (cnxBodyWith (chanLNTensor3 c h w p.εn p.γn p.βn)
    p.Wdw p.bdw p.Wex p.bex p.Wpr p.bpr (cnxGlsCh p))

theorem cnxBlockChW_differentiable {c cExp h w kH kW : Nat}
    (p : CnxBlockParamsCh c cExp h w kH kW) (hε : 0 < p.εn) :
    Differentiable ℝ (cnxBlockChW p) := by
  unfold cnxBlockChW residual
  exact (cnxBodyWith_differentiable (chanLNTensor3_differentiable c h w p.εn p.γn p.βn hε)
    p.Wdw p.bdw p.Wex p.bex p.Wpr p.bpr (cnxGlsCh p)).add differentiable_id

noncomputable def cnxBlockChWHasVJP {c cExp h w kH kW : Nat}
    (p : CnxBlockParamsCh c cExp h w kH kW) (hε : 0 < p.εn) :
    HasVJP (cnxBlockChW p) :=
  residualHasVJP _
    (cnxBodyWith_differentiable (chanLNTensor3_differentiable c h w p.εn p.γn p.βn hε)
      p.Wdw p.bdw p.Wex p.bex p.Wpr p.bpr (cnxGlsCh p))
    (cnxBodyWithHasVJP (chanLNTensor3_differentiable c h w p.εn p.γn p.βn hε)
      (chanLNTensor3HasVJP c h w p.εn p.γn p.βn hε)
      p.Wdw p.bdw p.Wex p.bex p.Wpr p.bpr (cnxGlsCh p))

/-- **Depth-`k` channel-LN stage fold** (head recursion — block `0` runs first). -/
noncomputable def convNextStageChK {c cExp h w kH kW : Nat} :
    (k : Nat) → (Fin k → CnxBlockParamsCh c cExp h w kH kW) →
    Vec (c * h * w) → Vec (c * h * w)
  | 0, _ => fun v => v
  | k + 1, ps => convNextStageChK k (fun i => ps i.succ) ∘ cnxBlockChW (ps 0)

theorem convNextStageChK_differentiable {c cExp h w kH kW : Nat} :
    ∀ (k : Nat) (ps : Fin k → CnxBlockParamsCh c cExp h w kH kW),
      (∀ i, 0 < (ps i).εn) → Differentiable ℝ (convNextStageChK k ps)
  | 0, _, _ => differentiable_id
  | k + 1, ps, hε =>
      (convNextStageChK_differentiable k (fun i => ps i.succ) (fun i => hε i.succ)).comp
        (cnxBlockChW_differentiable (ps 0) (hε 0))

noncomputable def convNextStageChKHasVJP {c cExp h w kH kW : Nat} :
    (k : Nat) → (ps : Fin k → CnxBlockParamsCh c cExp h w kH kW) →
    (∀ i, 0 < (ps i).εn) → HasVJP (convNextStageChK k ps)
  | 0, _, _ => identityHasVJP _
  | k + 1, ps, hε =>
      vjpComp (cnxBlockChW (ps 0)) (convNextStageChK k (fun i => ps i.succ))
        (cnxBlockChW_differentiable (ps 0) (hε 0))
        (convNextStageChK_differentiable k (fun i => ps i.succ) (fun i => hε i.succ))
        (cnxBlockChWHasVJP (ps 0) (hε 0))
        (convNextStageChKHasVJP k (fun i => ps i.succ) (fun i => hε i.succ))

-- ── the stage-boundary downsample, at the channel LN ──

/-- The stage-boundary downsample's parameters: the LN affine is `Vec cin`, over the
    PRE-downsample channel width. -/
structure CnxDownParamsCh (cin cout : Nat) where
  ε : ℝ
  γ : Vec cin
  β : Vec cin
  W : Kernel4 cout cin 2 2
  b : Vec cout

/-- Stage-boundary downsample: `2×2/s2 conv ∘ channel-LN` (`cin@2h×2w → cout@h×w`). -/
noncomputable def cnxDownChW (h w : Nat) {cin cout : Nat} (p : CnxDownParamsCh cin cout) :
    Vec (cin * (2 * h) * (2 * w)) → Vec (cout * h * w) :=
  flatConvStride2 (h := h) (w := w) p.W p.b ∘
    chanLNTensor3 cin (2 * h) (2 * w) p.ε p.γ p.β

theorem cnxDownChW_differentiable (h w : Nat) {cin cout : Nat} (p : CnxDownParamsCh cin cout)
    (hε : 0 < p.ε) : Differentiable ℝ (cnxDownChW h w p) := by
  unfold cnxDownChW
  exact (flatConvStride2_differentiable p.W p.b).comp
    (chanLNTensor3_differentiable cin (2 * h) (2 * w) p.ε p.γ p.β hε)

noncomputable def cnxDownChWHasVJP (h w : Nat) {cin cout : Nat} (p : CnxDownParamsCh cin cout)
    (hε : 0 < p.ε) : HasVJP (cnxDownChW h w p) := by
  unfold cnxDownChW
  exact vjpComp _ _
    (chanLNTensor3_differentiable cin (2 * h) (2 * w) p.ε p.γ p.β hε)
    (flatConvStride2_differentiable p.W p.b)
    (chanLNTensor3HasVJP cin (2 * h) (2 * w) p.ε p.γ p.β hε)
    (flatConvStride2HasVJP p.W p.b)

-- ── the whole channel-LN ConvNeXt-T ──

/-- All ConvNeXt-T parameters. Every LN affine is a `Vec`, and BOTH the stem LN (`sγ`/`sβ :
    Vec 96`) and the **head LN** (`hγ`/`hβ : Vec 768`) are present — the paper's `forward` is
    `patchify → channel_layer_norm → stages → GAP → LN → dense`. The official implementation and
    timm both have the stem and head LNs:
    `facebookresearch/ConvNeXt` does `self.norm(x.mean([-2,-1]))` with
    `nn.LayerNorm(dims[-1], eps=1e-6)`, and timm's `convnext_tiny` head is
    `NormMlpClassifierHead(global_pool → LayerNorm2d(768) → flatten → fc)`. -/
-- History (maintainer note): the head LN was once deleted to match
-- jax/MainConvNeXtImagenet.lean, which was itself missing it. The parameter count caught it:
-- 28,587,592 at K=1000 against timm `convnext_tiny`'s 28,589,128, short by 1,536 = 2×768, the
-- head LN's γ and β. A parameter count that matches a reference is a decomposition test; the
-- reference can be wrong.
structure CnxTWeightsCh (nC : Nat) where
  sW : Kernel4 96 3 4 4
  sb : Vec 96
  sε : ℝ
  sγ : Vec 96
  sβ : Vec 96
  s1 : Fin 3 → CnxBlockParamsCh 96 384 56 56 7 7
  d1 : CnxDownParamsCh 96 192
  s2 : Fin 3 → CnxBlockParamsCh 192 768 28 28 7 7
  d2 : CnxDownParamsCh 192 384
  s3 : Fin 9 → CnxBlockParamsCh 384 1536 14 14 7 7
  d3 : CnxDownParamsCh 384 768
  s4 : Fin 3 → CnxBlockParamsCh 768 3072 7 7 7 7
  /-- Head LayerNorm, between GAP and the classifier: ε, then the `Vec 768` affine.
      Plain `layerNormVec`, not `chanLNTensor3`: after GAP the tensor is `[768]`, one row, so
      the channel LN and the vector LN are the same function and this is the cheaper spelling —
      it is also `rowLNVecFlat 1 768`, i.e. ViT's per-token LN at one row, whose
      `rowLNVecFlat_differentiable` and `HasVJP` are already proven and whose graph mirror is
      `rowLN_affine_eq`. -/
  hε : ℝ
  hγ : Vec 768
  hβ : Vec 768
  Wd : Mat 768 nC
  bd : Vec nC

/-- **The channel-LN ConvNeXt-T forward** (3×224² → `nC`). Nested-application form, so the graph
    faithfulness `convNextFwdGraphTCh_faithful` closes stage by stage. `nC` is a binder: the
    Imagenette artifacts run it at 10 and the `convnextin_*` ImageNet artifacts at 1000. It has no
    drop-path; the `*drop*` artifacts compute another function. -/
noncomputable def convNextForwardTCh {nC : Nat} (w : CnxTWeightsCh nC) (x : Vec (3 * 224 * 224)) :
    Vec nC :=
  dense w.Wd w.bd
   (rowLNVecFlat 1 768 w.hε w.hγ w.hβ
    (globalAvgPoolFlat 768 7 7
      (convNextStageChK 3 w.s4
        (cnxDownChW 7 7 w.d3
          (convNextStageChK 9 w.s3
            (cnxDownChW 14 14 w.d2
              (convNextStageChK 3 w.s2
                (cnxDownChW 28 28 w.d1
                  (convNextStageChK 3 w.s1
                    (chanLNTensor3 96 56 56 w.sε w.sγ w.sβ
                      (flatConvStride4 (h := 56) (w := 56) w.sW w.sb x)))))))))))

/-- **The channel-LN ConvNeXt-T has a (correct) VJP — at every input.** 23 LayerNorm
    positivities: stem + 18 blocks (via the per-stage `∀ i`) + 3 downsamples + **the head LN**,
    which this statement composes as `rowLNVecFlat 1 768 w.hε w.hγ w.hβ` and takes `hhε` for.
    Chain-stated to keep the blocks opaque. -/
noncomputable def convNextForwardTChHasVJP {nC : Nat} (w : CnxTWeightsCh nC)
    (hsε : 0 < w.sε)
    (h1 : ∀ i, 0 < (w.s1 i).εn) (hd1 : 0 < w.d1.ε)
    (h2 : ∀ i, 0 < (w.s2 i).εn) (hd2 : 0 < w.d2.ε)
    (h3 : ∀ i, 0 < (w.s3 i).εn) (hd3 : 0 < w.d3.ε)
    (h4 : ∀ i, 0 < (w.s4 i).εn) (hhε : 0 < w.hε) :
    HasVJP
      (dense w.Wd w.bd ∘
        rowLNVecFlat 1 768 w.hε w.hγ w.hβ ∘
        globalAvgPoolFlat 768 7 7 ∘
        convNextStageChK 3 w.s4 ∘
        cnxDownChW 7 7 w.d3 ∘
        convNextStageChK 9 w.s3 ∘
        cnxDownChW 14 14 w.d2 ∘
        convNextStageChK 3 w.s2 ∘
        cnxDownChW 28 28 w.d1 ∘
        convNextStageChK 3 w.s1 ∘
        chanLNTensor3 96 56 56 w.sε w.sγ w.sβ ∘
        flatConvStride4 (h := 56) (w := 56) w.sW w.sb) := by
  have st_diff := flatConvStride4_differentiable (h := 56) (w := 56) w.sW w.sb
  have st_vjp := flatConvStride4HasVJP (h := 56) (w := 56) w.sW w.sb
  have lns_diff := chanLNTensor3_differentiable 96 56 56 w.sε w.sγ w.sβ hsε
  have lns_vjp := chanLNTensor3HasVJP 96 56 56 w.sε w.sγ w.sβ hsε
  have e1 := vjpComp _ _ st_diff lns_diff st_vjp lns_vjp
  have f1 := lns_diff.comp st_diff
  have s1d := convNextStageChK_differentiable 3 w.s1 h1
  have e2 := vjpComp _ _ f1 s1d e1 (convNextStageChKHasVJP 3 w.s1 h1)
  have f2 := s1d.comp f1
  have d1d := cnxDownChW_differentiable 28 28 w.d1 hd1
  have e3 := vjpComp _ _ f2 d1d e2 (cnxDownChWHasVJP 28 28 w.d1 hd1)
  have f3 := d1d.comp f2
  have s2d := convNextStageChK_differentiable 3 w.s2 h2
  have e4 := vjpComp _ _ f3 s2d e3 (convNextStageChKHasVJP 3 w.s2 h2)
  have f4 := s2d.comp f3
  have d2d := cnxDownChW_differentiable 14 14 w.d2 hd2
  have e5 := vjpComp _ _ f4 d2d e4 (cnxDownChWHasVJP 14 14 w.d2 hd2)
  have f5 := d2d.comp f4
  have s3d := convNextStageChK_differentiable 9 w.s3 h3
  have e6 := vjpComp _ _ f5 s3d e5 (convNextStageChKHasVJP 9 w.s3 h3)
  have f6 := s3d.comp f5
  have d3d := cnxDownChW_differentiable 7 7 w.d3 hd3
  have e7 := vjpComp _ _ f6 d3d e6 (cnxDownChWHasVJP 7 7 w.d3 hd3)
  have f7 := d3d.comp f6
  have s4d := convNextStageChK_differentiable 3 w.s4 h4
  have e8 := vjpComp _ _ f7 s4d e7 (convNextStageChKHasVJP 3 w.s4 h4)
  have f8 := s4d.comp f7
  have gap_diff := globalAvgPoolFlat_differentiable 768 7 7
  have e9 := vjpComp _ _ f8 gap_diff e8 (globalAvgPoolFlatHasVJP 768 7 7)
  have f9 := gap_diff.comp f8
  -- ▶ the HEAD LN, restored 2026-08-30 (the paper's `norm(x.mean([-2,-1]))`). One more
  -- `vjpComp` link and one more positivity — `layerNormVec` is ViT's final-LN primitive and
  -- carries its own `_diff`/`HasVJP`, so nothing new had to be proven here.
  have hln_diff := rowLNVecFlat_differentiable 1 768 w.hε w.hγ w.hβ hhε
  have e10 := vjpComp _ _ f9 hln_diff e9 (rowLNVecFlatHasVJP 1 768 w.hε w.hγ w.hβ hhε)
  have f10 := hln_diff.comp f9
  exact vjpComp _ _ f10 (dense_differentiable w.Wd w.bd) e10 (denseHasVJP w.Wd w.bd)

/-- **The chain is differentiable everywhere** (the 23 LayerNorm positivities only) — the
    `Differentiable` peer of `convNextForwardTChHasVJP`, on the same twelve-factor chain, which
    `batchMapHasVJP` asks for beside the `HasVJP` when the net is lifted over a batch
    (`ConvNeXtWholeBackCertifiedTieB.lean`). -/
theorem convNextForwardTCh_differentiable {nC : Nat} (w : CnxTWeightsCh nC)
    (hsε : 0 < w.sε)
    (h1 : ∀ i, 0 < (w.s1 i).εn) (hd1 : 0 < w.d1.ε)
    (h2 : ∀ i, 0 < (w.s2 i).εn) (hd2 : 0 < w.d2.ε)
    (h3 : ∀ i, 0 < (w.s3 i).εn) (hd3 : 0 < w.d3.ε)
    (h4 : ∀ i, 0 < (w.s4 i).εn) (hhε : 0 < w.hε) :
    Differentiable ℝ
      (dense w.Wd w.bd ∘
        rowLNVecFlat 1 768 w.hε w.hγ w.hβ ∘
        globalAvgPoolFlat 768 7 7 ∘
        convNextStageChK 3 w.s4 ∘
        cnxDownChW 7 7 w.d3 ∘
        convNextStageChK 9 w.s3 ∘
        cnxDownChW 14 14 w.d2 ∘
        convNextStageChK 3 w.s2 ∘
        cnxDownChW 28 28 w.d1 ∘
        convNextStageChK 3 w.s1 ∘
        chanLNTensor3 96 56 56 w.sε w.sγ w.sβ ∘
        flatConvStride4 (h := 56) (w := 56) w.sW w.sb) :=
  (dense_differentiable w.Wd w.bd).comp
    ((rowLNVecFlat_differentiable 1 768 w.hε w.hγ w.hβ hhε).comp
      ((globalAvgPoolFlat_differentiable 768 7 7).comp
        ((convNextStageChK_differentiable 3 w.s4 h4).comp
          ((cnxDownChW_differentiable 7 7 w.d3 hd3).comp
            ((convNextStageChK_differentiable 9 w.s3 h3).comp
              ((cnxDownChW_differentiable 14 14 w.d2 hd2).comp
                ((convNextStageChK_differentiable 3 w.s2 h2).comp
                  ((cnxDownChW_differentiable 28 28 w.d1 hd1).comp
                    ((convNextStageChK_differentiable 3 w.s1 h1).comp
                      ((chanLNTensor3_differentiable 96 56 56 w.sε w.sγ w.sβ hsε).comp
                        (flatConvStride4_differentiable (h := 56) (w := 56) w.sW w.sb)))))))))))

/-- The nested↔chain bridge: `convNextForwardTCh w x` equals the twelve-factor `∘` chain
    `convNextForwardTChHasVJP` is stated on. -/
theorem convNextForwardTCh_eq_chain {nC : Nat} (w : CnxTWeightsCh nC) (x : Vec (3 * 224 * 224)) :
    convNextForwardTCh w x =
      (dense w.Wd w.bd ∘
        rowLNVecFlat 1 768 w.hε w.hγ w.hβ ∘
        globalAvgPoolFlat 768 7 7 ∘
        convNextStageChK 3 w.s4 ∘
        cnxDownChW 7 7 w.d3 ∘
        convNextStageChK 9 w.s3 ∘
        cnxDownChW 14 14 w.d2 ∘
        convNextStageChK 3 w.s2 ∘
        cnxDownChW 28 28 w.d1 ∘
        convNextStageChK 3 w.s1 ∘
        chanLNTensor3 96 56 56 w.sε w.sγ w.sβ ∘
        flatConvStride4 (h := 56) (w := 56) w.sW w.sb) x := by
  -- `rw`, not `simp`/`rfl`: those die in the kernel on the recursive stage folds.
  rw [convNextForwardTCh]
  rw [Function.comp_apply, Function.comp_apply, Function.comp_apply, Function.comp_apply,
      Function.comp_apply, Function.comp_apply, Function.comp_apply, Function.comp_apply,
      Function.comp_apply, Function.comp_apply, Function.comp_apply]

/-- Correctness on `convNextForwardTCh` itself (via the bridge). -/
theorem convNextForwardTChHasVJP_correct {nC : Nat} (w : CnxTWeightsCh nC)
    (hsε : 0 < w.sε)
    (h1 : ∀ i, 0 < (w.s1 i).εn) (hd1 : 0 < w.d1.ε)
    (h2 : ∀ i, 0 < (w.s2 i).εn) (hd2 : 0 < w.d2.ε)
    (h3 : ∀ i, 0 < (w.s3 i).εn) (hd3 : 0 < w.d3.ε)
    (h4 : ∀ i, 0 < (w.s4 i).εn) (hhε : 0 < w.hε)
    (x : Vec (3 * 224 * 224)) (dy : Vec nC) (i : Fin (3 * 224 * 224)) :
    (convNextForwardTChHasVJP w hsε h1 hd1 h2 hd2 h3 hd3 h4 hhε).backward x dy i =
      ∑ j : Fin nC, pdiv (convNextForwardTCh w) x i j * dy j := by
  have h := (convNextForwardTChHasVJP w hsε h1 hd1 h2 hd2 h3 hd3 h4 hhε).correct x dy i
  rwa [show convNextForwardTCh w =
        (dense w.Wd w.bd ∘
          rowLNVecFlat 1 768 w.hε w.hγ w.hβ ∘
        globalAvgPoolFlat 768 7 7 ∘
          convNextStageChK 3 w.s4 ∘
          cnxDownChW 7 7 w.d3 ∘
          convNextStageChK 9 w.s3 ∘
          cnxDownChW 14 14 w.d2 ∘
          convNextStageChK 3 w.s2 ∘
          cnxDownChW 28 28 w.d1 ∘
          convNextStageChK 3 w.s1 ∘
          chanLNTensor3 96 56 56 w.sε w.sγ w.sβ ∘
          flatConvStride4 (h := 56) (w := 56) w.sW w.sb)
      from funext (convNextForwardTCh_eq_chain w)]

end Proofs

namespace Proofs.StableHLO

-- ════════════════════════════════════════════════════════════════
-- § §2m — the channel-LN graph + faithfulness (rung E's new apex)
-- ════════════════════════════════════════════════════════════════

/-- **One channel-LN forward site**, mirroring `ConvNeXtRender.lnFwdSite` at `chLN := true`
    op-for-op: transpose to `[h·w, c]`, normalise each spatial row over its channels at the
    scalar identities `%one`/`%zero`, apply the real `[c]` affine, transpose back. The two `▸`
    transports are the `Nat`-associativity casts the render spells the same way; `den_reassocS`
    (`ConvNeXtChannelLN.lean`) is what makes them the math's Mat-split bridge. -/
def chanLNGraph (gN btN epsStr : String) {c h w : Nat} (ε : ℝ) (γ β : Vec c)
    (e : SHlo (c * h * w)) : SHlo (c * h * w) :=
  (Nat.mul_assoc c h w).symm ▸
    (.transposeF (m := h * w) (n := c)
      (.rowBiasF (m := h * w) (n := c) btN β
        (.rowScaleF (m := h * w) (n := c) gN γ
          (.lnRowF (m := h * w) (n := c) "%one" "%zero" epsStr ε 1 0
            (.transposeF (m := c) (n := h * w) ((Nat.mul_assoc c h w) ▸ e))))))

theorem chanLNGraph_faithful (gN btN epsStr : String) {c h w : Nat} (ε : ℝ) (γ β : Vec c)
    (e : SHlo (c * h * w)) :
    den (chanLNGraph gN btN epsStr ε γ β e) = chanLNTensor3 c h w ε γ β (den e) := by
  unfold chanLNGraph chanLNTensor3
  rw [den_unassocS, transposeF_faithful, rowBiasF_faithful, rowScaleF_faithful,
      lnRowF_faithful, transposeF_faithful, den_reassocS, rowLN_affine_eq]
  rfl

/-- **The head LN forward site** — the paper's `norm(x.mean([-2,-1]))`.

    It is `chanLNGraph` with the transposes deleted, and that is not a shortcut: after GAP the
    tensor is a single `[768]` row, so "normalise each spatial row over its channels" and
    "normalise the feature vector" are the same function — `m = 1`. The render emits exactly these
    three ops, which is what makes `ConvNeXtRender.headLnFwdSite` a mirror rather than a peer.
    Note: indexed `SHlo (1 * c)`; `c` is the literal 768 at every call site, so `1 * c` reduces
    and no transport is needed. Do not generalise `c` to a variable without adding one — that is the
    trap `convNextBackAll`'s `Vec (1 * nClasses)` annotations already record. -/
def headLNGraph (gN btN epsStr : String) {c : Nat} (ε : ℝ) (γ β : Vec c)
    (e : SHlo (1 * c)) : SHlo (1 * c) :=
  .rowBiasF (m := 1) (n := c) btN β
    (.rowScaleF (m := 1) (n := c) gN γ
      (.lnRowF (m := 1) (n := c) "%one" "%zero" epsStr ε 1 0 e))

theorem headLNGraph_faithful (gN btN epsStr : String) {c : Nat} (ε : ℝ) (γ β : Vec c)
    (e : SHlo (1 * c)) :
    den (headLNGraph gN btN epsStr ε γ β e) = rowLNVecFlat 1 c ε γ β (den e) := by
  unfold headLNGraph
  rw [rowBiasF_faithful, rowScaleF_faithful, lnRowF_faithful, rowLN_affine_eq]

/-- The ConvNeXt block graph — the `[3,3,9,3]` block segment, with `chanLNGraph` at its LN
    site. -/
def cnxBlockChGraphW (pfx epsStr : String) {c cExp h w kH kW : Nat}
    (p : CnxBlockParamsCh c cExp h w kH kW) (e : SHlo (c * h * w)) : SHlo (c * h * w) :=
  .addV
    (.layerScaleChF s!"%{pfx}gls" p.γls
      (.flatConvF (h := h) (w := w) s!"%{pfx}Wpr" s!"%{pfx}bpr" p.Wpr p.bpr
        (.geluF
          (.flatConvF (h := h) (w := w) s!"%{pfx}Wex" s!"%{pfx}bex" p.Wex p.bex
            (chanLNGraph s!"%{pfx}gn" s!"%{pfx}btn" epsStr p.εn p.γn p.βn
              (.depthwiseF (h := h) (w := w) s!"%{pfx}Wdw" s!"%{pfx}bdw" p.Wdw p.bdw e))))))
    e

theorem cnxBlockChGraphW_faithful (pfx epsStr : String) {c cExp h w kH kW : Nat}
    (p : CnxBlockParamsCh c cExp h w kH kW) (e : SHlo (c * h * w)) :
    den (cnxBlockChGraphW pfx epsStr p e) = cnxBlockChW p (den e) := by
  unfold cnxBlockChGraphW cnxBlockChW cnxGlsCh cnxBodyWith residual biPath
  simp only [layerScaleChF_faithful, flatConvF_faithful, geluF_faithful,
             chanLNGraph_faithful, depthwiseF_faithful, den_addV, Function.comp_apply]

/-- **Depth-`k` channel-LN stage graph fold** — block `base+1` first, prefixes `b{base+1}_`. -/
def cnxStageChGraphK (epsStr : String) {c cExp h w kH kW : Nat} :
    (base k : Nat) → (Fin k → CnxBlockParamsCh c cExp h w kH kW) →
    SHlo (c * h * w) → SHlo (c * h * w)
  | _, 0, _, e => e
  | base, k + 1, ps, e =>
      cnxStageChGraphK epsStr (base + 1) k (fun i => ps i.succ)
        (cnxBlockChGraphW s!"b{base + 1}_" epsStr (ps 0) e)

lemma cnxStageChGraphK_den (epsStr : String) {c cExp h w kH kW : Nat} :
    ∀ (base k : Nat) (ps : Fin k → CnxBlockParamsCh c cExp h w kH kW)
      (e : SHlo (c * h * w)),
      den (cnxStageChGraphK epsStr base k ps e) = convNextStageChK k ps (den e)
  | _, 0, _, _ => rfl
  | base, k + 1, ps, e => by
      have ih := cnxStageChGraphK_den epsStr (base + 1) k (fun i => ps i.succ)
        (cnxBlockChGraphW s!"b{base + 1}_" epsStr (ps 0) e)
      rw [show cnxStageChGraphK epsStr base (k + 1) ps e =
            cnxStageChGraphK epsStr (base + 1) k (fun i => ps i.succ)
              (cnxBlockChGraphW s!"b{base + 1}_" epsStr (ps 0) e) from rfl,
          ih, cnxBlockChGraphW_faithful]
      rfl

/-- Channel-LN downsample graph: channel-LN → 2×2/s2 widening conv. -/
def cnxDownChGraphW (pfx epsStr : String) (h w : Nat) {cin cout : Nat}
    (p : CnxDownParamsCh cin cout) (e : SHlo (cin * (2 * h) * (2 * w))) :
    SHlo (cout * h * w) :=
  .flatConvStridedF (h := h) (w := w) s!"%{pfx}W" s!"%{pfx}b" p.W p.b
    (chanLNGraph s!"%{pfx}gn" s!"%{pfx}btn" epsStr p.ε p.γ p.β e)

theorem cnxDownChGraphW_faithful (pfx epsStr : String) (h w : Nat) {cin cout : Nat}
    (p : CnxDownParamsCh cin cout) (e : SHlo (cin * (2 * h) * (2 * w))) :
    den (cnxDownChGraphW pfx epsStr h w p e) = cnxDownChW h w p (den e) := by
  unfold cnxDownChGraphW cnxDownChW
  simp only [flatConvStridedF_faithful, chanLNGraph_faithful, Function.comp_apply]

/-- The **channel-LN ConvNeXt-T forward graph** (3×224² → `nC`): patchify stem → **stem
    channel-LN** → the `[3,3,9,3]` stages with 3 channel-LN + 2×2/s2 downsample boundaries →
    GAP → **head LN** → dense. 23 LN sites: 1 stem + 18 block + 3 downsample + head. -/
def convNextFwdGraphTCh (epsStr : String) {nC : Nat} (w : CnxTWeightsCh nC)
    (x : Vec (3 * 224 * 224)) : SHlo nC :=
  denseF "%Wd" "%bd" w.Wd w.bd
   (headLNGraph "%hng" "%hnbt" epsStr w.hε w.hγ w.hβ
    (.gapF (c := 768) (h := 7) (w := 7)
      (cnxStageChGraphK epsStr 15 3 w.s4
        (cnxDownChGraphW "d3" epsStr 7 7 w.d3
          (cnxStageChGraphK epsStr 6 9 w.s3
            (cnxDownChGraphW "d2" epsStr 14 14 w.d2
              (cnxStageChGraphK epsStr 3 3 w.s2
                (cnxDownChGraphW "d1" epsStr 28 28 w.d1
                  (cnxStageChGraphK epsStr 0 3 w.s1
                    (chanLNGraph "%gst" "%btst" epsStr w.sε w.sγ w.sβ
                      (.flatConvStride4F (h := 56) (w := 56) "%Wst" "%bst" w.sW w.sb
                        (.operand "%x" x))))))))))))

/-- **Channel-LN forward faithfulness** — the `[3,3,9,3]` channel-LN graph denotes
    `convNextForwardTCh`. One `rw` per stage: `cnxStageChGraphK_den` / `cnxDownChGraphW_faithful`
    at the stages and downsamples, `chanLNGraph_faithful` at the stem LN and `headLNGraph_faithful`
    at the head. -/
theorem convNextFwdGraphTCh_faithful (epsStr : String) {nC : Nat} (w : CnxTWeightsCh nC)
    (x : Vec (3 * 224 * 224)) :
    den (convNextFwdGraphTCh epsStr w x) = convNextForwardTCh w x := by
  rw [convNextFwdGraphTCh, denseF_faithful, headLNGraph_faithful, gapF_faithful,
      cnxStageChGraphK_den, cnxDownChGraphW_faithful,
      cnxStageChGraphK_den, cnxDownChGraphW_faithful,
      cnxStageChGraphK_den, cnxDownChGraphW_faithful,
      cnxStageChGraphK_den, chanLNGraph_faithful, flatConvStride4F_faithful, den_operand]
  rfl

end Proofs.StableHLO

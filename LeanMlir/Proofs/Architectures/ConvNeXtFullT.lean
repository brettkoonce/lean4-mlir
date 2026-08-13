import LeanMlir.Proofs.Architectures.ConvNeXtChainClose
import LeanMlir.Proofs.Architectures.ConvNeXtChannelLN
import LeanMlir.Proofs.Codegen.StableHLO

/-! # The FULL ConvNeXt-T — `[3,3,9,3]`, forward + whole-net VJP + graph + faithfulness

Scales the ch9 representative (1×1 stem + 2 blocks at one scale) to the real ConvNeXt-T
spec, closing the "full-architecture" gap in `planning/convnext_close.md`:

  4×4/s4 patchify stem (3→96, 224→56) → stem-LN → stage1 (3 blocks @96/56²) →
  downsample (LN + 2×2/s2 conv 96→192) → stage2 (3 @192/28²) → ds (192→384) →
  stage3 (9 @384/14²) → ds (384→768) → stage4 (3 @768/7²) → GAP → dense.

Per the handoff recipe (`planning/convnext_close.md` §"Scaling handoff"):
1. **Depth-k within a stage** — `CnxBlockParamsCh` bundles the block 10-tuple;
   `convNextStageChK (k) (ps : Fin k → CnxBlockParamsCh …)` folds blocks head-first with
   VJP by induction — the ViT depth-k recipe, simpler here (same-shape blocks within a stage).
2. **Downsample boundaries** — `cnxDownChW` = `flatConvStride2(2×2) ∘ channel-LN`; both VJPs existed.
3. **4×4/s4 patchify stem** — `flatConvStride4` (= decimate ∘ decimateOdd ∘ stride-1
   SAME conv, `StridedConv.lean`: the left-aligned window `x[4i..4i+3]` of the paper's
   pad-0 `Conv2d(4, s=4)`) + the `flatConvStride4F` token.

GELU/LN/conv are smooth, so the whole-net VJP is GLOBAL (unconditional except the 22 LN
positivities) — ConvNeXt-T joins `efficientnetForwardB_full_has_vjp` and `vitForwardKV`.
The `ConvNeXtClose`/`ConvNeXtChainClose` param bridges are dim-generic and cover all 18 blocks
verbatim; the downsample conv W/b reuse the proven stride-2 bridges.

⚠ **§2n (2026-07-31): the scalar-LN twin of this chain is GONE.** Until then every definition
here had a `layerNormForward` peer — one mean and one variance over the whole `c·h·w` map with
scalar γ/β — which is what the repo shipped before §2m flipped ConvNeXt to its real channel
LayerNorm. `CnxBlockParams`, `cnxBlockW`, `convNextStageK`, `CnxDownParams`, `cnxDownW`,
`CnxTWeights`, `convNextForwardT`/`TC` and their graph section were deleted once the float
bridges (their last live consumers) had `…Ch` peers. If you are chasing a dangling reference to
one of those names, it was retired, not moved. `planning/xla_pjrt_handoff.md` §2n has the
checklist and what the drop did and did not touch.
-/

namespace Proofs

open scoped BigOperators

-- ════════════════════════════════════════════════════════════════
-- § The [3,3,9,3] chain at ConvNeXt's REAL channel LayerNorm (§2m)
-- ════════════════════════════════════════════════════════════════

/-! ConvNeXt specifies `channel_layer_norm` — `h·w` statistics per example, each over the `c`
channels at one spatial position, per-channel `[c]` affine. See `ConvNeXtChannelLN.lean` for the
primitive and for why Route A needs no new op and no new VJP.

§2m built this as a PARALLEL chain beside the scalar-LN one it superseded, so that flipping the
net could not change what the then-committed `convNextForwardTC` denoted (`MobileNetV2RenderB`'s
reason, §2f). **§2n then DROPPED that scalar chain** — once its last live consumers (the float
bridges) had channel-LN peers, a retired chain that still elaborates is one more thing to drift
(§2a's lesson). What is here is what ships. ⛔ The `Architectures/ConvNeXt.lean` ch9
representative (`convNextForward`, `convNextBlock`, `convNextBlockBody`) is a different thing and
SURVIVED the drop: it backs the Diderot comparator and a book chapter.

Two deviations §2m found by the parameter count NOT matching, both invisible to a VJP argument:
the reference's 22 LN sites are **1 stem + 18 block + 3 downsample**, where the pre-§2m net had
**18 + 3 + 1 head**. So this forward has a stem LN and no head LN. They nearly cancel
(+2·768 − 2·96 = +1,344 of 28.6M), which is exactly why a matching parameter count is a
decomposition test and not an architecture check. -/

-- ── the block body, abstracted over its normalisation ──

/-- The ConvNeXt block body with the LN left as a parameter. Written once and instantiated at
    `chanLNTensor3`, so the channel-LN world costs one definition rather than a second copy of
    `convNextBlockBody`'s six-piece `vjp_comp` chain. -/
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

theorem cnxBodyWith_diff {c cExp h w kH kW : Nat}
    {LN : Vec (c * h * w) → Vec (c * h * w)} (hLN : Differentiable ℝ LN)
    (Wdw : DepthwiseKernel c kH kW) (bdw : Vec c)
    (Wex : Kernel4 cExp c 1 1) (bex : Vec cExp)
    (Wpr : Kernel4 c cExp 1 1) (bpr : Vec c)
    (γls : Vec (c * h * w)) :
    Differentiable ℝ (cnxBodyWith LN Wdw bdw Wex bex Wpr bpr γls) := by
  unfold cnxBodyWith
  exact (layerScale_differentiable γls).comp
    ((flatConv_differentiable (h := h) (w := w) Wpr bpr).comp
      ((gelu_diff (cExp * h * w)).comp
        ((flatConv_differentiable (h := h) (w := w) Wex bex).comp
          (hLN.comp (depthwiseFlat_differentiable (h := h) (w := w) Wdw bdw)))))

/-- The body VJP, given the LN's. Only the LN carries a hypothesis — gelu is smooth and
    conv/layerScale are linear, so this is global exactly as `convNextBlockBody_has_vjp` is. -/
noncomputable def cnxBodyWith_has_vjp {c cExp h w kH kW : Nat}
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
  have hge := gelu_diff (cExp * h * w)
  have hpr := flatConv_differentiable (h := h) (w := w) Wpr bpr
  have hls := layerScale_differentiable γls
  have e1 := vjp_comp _ _ hdw hLN
    (depthwiseFlat_has_vjp (h := h) (w := w) Wdw bdw) vLN
  have f1 := hLN.comp hdw
  have e2 := vjp_comp _ _ f1 hex e1 (hasVJP3_to_hasVJP (conv2d_has_vjp3 Wex bex))
  have f2 := hex.comp f1
  have e3 := vjp_comp _ _ f2 hge e2 (gelu_has_vjp (cExp * h * w))
  have f3 := hge.comp f2
  have e4 := vjp_comp _ _ f3 hpr e3 (hasVJP3_to_hasVJP (conv2d_has_vjp3 Wpr bpr))
  have f4 := hpr.comp f3
  exact vjp_comp _ _ f4 hls e4 (layerScale_has_vjp γls)

-- ── per-block params + the stage fold, at the channel LN ──

/-- One channel-LN ConvNeXt block's 10 parameters. `γn`/`βn` are `Vec c` — 2c floats per LN site,
    where the retired scalar-LN spelling had 2 (§2n deleted it). -/
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

/-- The per-channel layer-scale expanded to the flat map (`cnxGls`'s peer). -/
noncomputable def cnxGlsCh {c cExp h w kH kW : Nat} (p : CnxBlockParamsCh c cExp h w kH kW) :
    Vec (c * h * w) :=
  fun k => p.γls (StableHLO.chanIdx c h w k)

/-- The packaged ConvNeXt block: `residual` of the shared body at `chanLNTensor3`. -/
noncomputable def cnxBlockChW {c cExp h w kH kW : Nat}
    (p : CnxBlockParamsCh c cExp h w kH kW) :
    Vec (c * h * w) → Vec (c * h * w) :=
  residual (cnxBodyWith (chanLNTensor3 c h w p.εn p.γn p.βn)
    p.Wdw p.bdw p.Wex p.bex p.Wpr p.bpr (cnxGlsCh p))

theorem cnxBlockChW_diff {c cExp h w kH kW : Nat}
    (p : CnxBlockParamsCh c cExp h w kH kW) (hε : 0 < p.εn) :
    Differentiable ℝ (cnxBlockChW p) := by
  unfold cnxBlockChW residual
  intro v
  exact DifferentiableAt.add
    ((cnxBodyWith_diff (chanLNTensor3_diff c h w p.εn p.γn p.βn hε)
      p.Wdw p.bdw p.Wex p.bex p.Wpr p.bpr (cnxGlsCh p)) v)
    differentiable_id.differentiableAt

noncomputable def cnxBlockChW_has_vjp {c cExp h w kH kW : Nat}
    (p : CnxBlockParamsCh c cExp h w kH kW) (hε : 0 < p.εn) :
    HasVJP (cnxBlockChW p) :=
  residual_has_vjp _
    (cnxBodyWith_diff (chanLNTensor3_diff c h w p.εn p.γn p.βn hε)
      p.Wdw p.bdw p.Wex p.bex p.Wpr p.bpr (cnxGlsCh p))
    (cnxBodyWith_has_vjp (chanLNTensor3_diff c h w p.εn p.γn p.βn hε)
      (chanLNTensor3_has_vjp c h w p.εn p.γn p.βn hε)
      p.Wdw p.bdw p.Wex p.bex p.Wpr p.bpr (cnxGlsCh p))

/-- **Depth-`k` channel-LN stage fold** (head recursion — block `0` runs first). -/
noncomputable def convNextStageChK {c cExp h w kH kW : Nat} :
    (k : Nat) → (Fin k → CnxBlockParamsCh c cExp h w kH kW) →
    Vec (c * h * w) → Vec (c * h * w)
  | 0, _ => fun v => v
  | k + 1, ps => convNextStageChK k (fun i => ps i.succ) ∘ cnxBlockChW (ps 0)

theorem convNextStageChK_diff {c cExp h w kH kW : Nat} :
    ∀ (k : Nat) (ps : Fin k → CnxBlockParamsCh c cExp h w kH kW),
      (∀ i, 0 < (ps i).εn) → Differentiable ℝ (convNextStageChK k ps)
  | 0, _, _ => differentiable_id
  | k + 1, ps, hε =>
      (convNextStageChK_diff k (fun i => ps i.succ) (fun i => hε i.succ)).comp
        (cnxBlockChW_diff (ps 0) (hε 0))

noncomputable def convNextStageChK_has_vjp {c cExp h w kH kW : Nat} :
    (k : Nat) → (ps : Fin k → CnxBlockParamsCh c cExp h w kH kW) →
    (∀ i, 0 < (ps i).εn) → HasVJP (convNextStageChK k ps)
  | 0, _, _ => identity_has_vjp _
  | k + 1, ps, hε =>
      vjp_comp (cnxBlockChW (ps 0)) (convNextStageChK k (fun i => ps i.succ))
        (cnxBlockChW_diff (ps 0) (hε 0))
        (convNextStageChK_diff k (fun i => ps i.succ) (fun i => hε i.succ))
        (cnxBlockChW_has_vjp (ps 0) (hε 0))
        (convNextStageChK_has_vjp k (fun i => ps i.succ) (fun i => hε i.succ))

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

theorem cnxDownChW_diff (h w : Nat) {cin cout : Nat} (p : CnxDownParamsCh cin cout)
    (hε : 0 < p.ε) : Differentiable ℝ (cnxDownChW h w p) := by
  unfold cnxDownChW
  exact (flatConvStride2_differentiable p.W p.b).comp
    (chanLNTensor3_diff cin (2 * h) (2 * w) p.ε p.γ p.β hε)

noncomputable def cnxDownChW_has_vjp (h w : Nat) {cin cout : Nat} (p : CnxDownParamsCh cin cout)
    (hε : 0 < p.ε) : HasVJP (cnxDownChW h w p) := by
  unfold cnxDownChW
  exact vjp_comp _ _
    (chanLNTensor3_diff cin (2 * h) (2 * w) p.ε p.γ p.β hε)
    (flatConvStride2_differentiable p.W p.b)
    (chanLNTensor3_has_vjp cin (2 * h) (2 * w) p.ε p.γ p.β hε)
    (flatConvStride2_has_vjp p.W p.b)

-- ── the whole channel-LN ConvNeXt-T ──

/-- All ConvNeXt-T parameters. Every LN affine is a `Vec`, the stem LN is PRESENT
    (`sγ`/`sβ : Vec 96`), and there is no head LN — the reference's `forward` is
    `patchify → channel_layer_norm → stages → GAP → dense`. (The pre-§2m net had scalar affines,
    no stem LN and a head LN; §2n deleted it.) -/
structure CnxTWeightsCh where
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
  Wd : Mat 768 10
  bd : Vec 10

/-- **The channel-LN ConvNeXt-T forward** (3×224² → 10). Nested-application form, as the scalar
    peers, so the graph faithfulness closes by a structural `rfl`. -/
noncomputable def convNextForwardTCh (w : CnxTWeightsCh) (x : Vec (3 * 224 * 224)) : Vec 10 :=
  dense w.Wd w.bd
    (globalAvgPoolFlat 768 7 7
      (convNextStageChK 3 w.s4
        (cnxDownChW 7 7 w.d3
          (convNextStageChK 9 w.s3
            (cnxDownChW 14 14 w.d2
              (convNextStageChK 3 w.s2
                (cnxDownChW 28 28 w.d1
                  (convNextStageChK 3 w.s1
                    (chanLNTensor3 96 56 56 w.sε w.sγ w.sβ
                      (flatConvStride4 (h := 56) (w := 56) w.sW w.sb x))))))))))

/-- **The channel-LN ConvNeXt-T has a (correct) VJP — at every input.** Same shape and the same
    hypothesis count as `convNextForwardTCh_has_vjp`: 22 LN positivities (stem + 18 blocks via the
    per-stage `∀ i` + 3 downsamples), no head LN. Chain-stated to keep the blocks opaque. -/
noncomputable def convNextForwardTCh_has_vjp (w : CnxTWeightsCh)
    (hsε : 0 < w.sε)
    (h1 : ∀ i, 0 < (w.s1 i).εn) (hd1 : 0 < w.d1.ε)
    (h2 : ∀ i, 0 < (w.s2 i).εn) (hd2 : 0 < w.d2.ε)
    (h3 : ∀ i, 0 < (w.s3 i).εn) (hd3 : 0 < w.d3.ε)
    (h4 : ∀ i, 0 < (w.s4 i).εn) :
    HasVJP
      (dense w.Wd w.bd ∘
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
  have st_vjp := flatConvStride4_has_vjp (h := 56) (w := 56) w.sW w.sb
  have lns_diff := chanLNTensor3_diff 96 56 56 w.sε w.sγ w.sβ hsε
  have lns_vjp := chanLNTensor3_has_vjp 96 56 56 w.sε w.sγ w.sβ hsε
  have e1 := vjp_comp _ _ st_diff lns_diff st_vjp lns_vjp
  have f1 := lns_diff.comp st_diff
  have s1d := convNextStageChK_diff 3 w.s1 h1
  have e2 := vjp_comp _ _ f1 s1d e1 (convNextStageChK_has_vjp 3 w.s1 h1)
  have f2 := s1d.comp f1
  have d1d := cnxDownChW_diff 28 28 w.d1 hd1
  have e3 := vjp_comp _ _ f2 d1d e2 (cnxDownChW_has_vjp 28 28 w.d1 hd1)
  have f3 := d1d.comp f2
  have s2d := convNextStageChK_diff 3 w.s2 h2
  have e4 := vjp_comp _ _ f3 s2d e3 (convNextStageChK_has_vjp 3 w.s2 h2)
  have f4 := s2d.comp f3
  have d2d := cnxDownChW_diff 14 14 w.d2 hd2
  have e5 := vjp_comp _ _ f4 d2d e4 (cnxDownChW_has_vjp 14 14 w.d2 hd2)
  have f5 := d2d.comp f4
  have s3d := convNextStageChK_diff 9 w.s3 h3
  have e6 := vjp_comp _ _ f5 s3d e5 (convNextStageChK_has_vjp 9 w.s3 h3)
  have f6 := s3d.comp f5
  have d3d := cnxDownChW_diff 7 7 w.d3 hd3
  have e7 := vjp_comp _ _ f6 d3d e6 (cnxDownChW_has_vjp 7 7 w.d3 hd3)
  have f7 := d3d.comp f6
  have s4d := convNextStageChK_diff 3 w.s4 h4
  have e8 := vjp_comp _ _ f7 s4d e7 (convNextStageChK_has_vjp 3 w.s4 h4)
  have f8 := s4d.comp f7
  have gap_diff := globalAvgPoolFlat_differentiable 768 7 7
  have e9 := vjp_comp _ _ f8 gap_diff e8 (globalAvgPoolFlat_has_vjp 768 7 7)
  have f9 := gap_diff.comp f8
  exact vjp_comp _ _ f9 (dense_differentiable w.Wd w.bd) e9 (dense_has_vjp w.Wd w.bd)

/-- The nested↔chain bridge (see `convNextForwardTCh_eq_chain` for why the proof shape matters —
    a `simp`/`rfl` proof of this statement dies in the kernel on the recursive stage folds). -/
theorem convNextForwardTCh_eq_chain (w : CnxTWeightsCh) (x : Vec (3 * 224 * 224)) :
    convNextForwardTCh w x =
      (dense w.Wd w.bd ∘
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
  rw [convNextForwardTCh]
  rw [Function.comp_apply, Function.comp_apply, Function.comp_apply, Function.comp_apply,
      Function.comp_apply, Function.comp_apply, Function.comp_apply, Function.comp_apply,
      Function.comp_apply, Function.comp_apply]

/-- Correctness on `convNextForwardTCh` itself (via the bridge). -/
theorem convNextForwardTCh_has_vjp_correct (w : CnxTWeightsCh)
    (hsε : 0 < w.sε)
    (h1 : ∀ i, 0 < (w.s1 i).εn) (hd1 : 0 < w.d1.ε)
    (h2 : ∀ i, 0 < (w.s2 i).εn) (hd2 : 0 < w.d2.ε)
    (h3 : ∀ i, 0 < (w.s3 i).εn) (hd3 : 0 < w.d3.ε)
    (h4 : ∀ i, 0 < (w.s4 i).εn)
    (x : Vec (3 * 224 * 224)) (dy : Vec 10) (i : Fin (3 * 224 * 224)) :
    (convNextForwardTCh_has_vjp w hsε h1 hd1 h2 hd2 h3 hd3 h4).backward x dy i =
      ∑ j : Fin 10, pdiv (convNextForwardTCh w) x i j * dy j := by
  have h := (convNextForwardTCh_has_vjp w hsε h1 hd1 h2 hd2 h3 hd3 h4).correct x dy i
  rwa [show convNextForwardTCh w =
        (dense w.Wd w.bd ∘
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

/-- The ConvNeXt block graph — the `[3,3,9,3]` block segment, with `chanLNGraph` at its LN site
    (the retired scalar spelling put a `.bnF` there). -/
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

/-- The **channel-LN ConvNeXt-T forward graph** (3×224² → 10): patchify stem → **stem
    channel-LN** → the `[3,3,9,3]` stages with 3 channel-LN + 2×2/s2 downsample boundaries →
    GAP → dense. Note what moved against the retired scalar graph: the stem LN is back and the head
    LN is gone — the reference's 22 sites are 1 stem + 18 block + 3 downsample. -/
def convNextFwdGraphTCh (epsStr : String) (w : CnxTWeightsCh)
    (x : Vec (3 * 224 * 224)) : SHlo 10 :=
  denseF "%Wd" "%bd" w.Wd w.bd
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
                        (.operand "%x" x)))))))))))

/-- **Channel-LN forward faithfulness** — the `[3,3,9,3]` channel-LN graph denotes
    `convNextForwardTCh`. Same `rw` chain as the scalar apex, with `chanLNGraph_faithful` where
    the `bnF`s were. The full-architecture apex for the net §2m makes ConvNeXt actually be. -/
theorem convNextFwdGraphTCh_faithful (epsStr : String) (w : CnxTWeightsCh)
    (x : Vec (3 * 224 * 224)) :
    den (convNextFwdGraphTCh epsStr w x) = convNextForwardTCh w x := by
  rw [convNextFwdGraphTCh, denseF_faithful, gapF_faithful,
      cnxStageChGraphK_den, cnxDownChGraphW_faithful,
      cnxStageChGraphK_den, cnxDownChGraphW_faithful,
      cnxStageChGraphK_den, cnxDownChGraphW_faithful,
      cnxStageChGraphK_den, chanLNGraph_faithful, flatConvStride4F_faithful, den_operand]
  rfl

end Proofs.StableHLO

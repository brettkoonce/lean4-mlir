import LeanMlir.Proofs.Architectures.ConvNeXtChainClose
import LeanMlir.Proofs.Codegen.StableHLO

/-! # The FULL ConvNeXt-T — `[3,3,9,3]`, forward + whole-net VJP + graph + faithfulness

Scales the ch9 representative (1×1 stem + 2 blocks at one scale) to the real ConvNeXt-T
spec, closing the "full-architecture" gap in `planning/convnext_close.md`:

  4×4/s4 patchify stem (3→96, 224→56) → stem-LN → stage1 (3 blocks @96/56²) →
  downsample (LN + 2×2/s2 conv 96→192) → stage2 (3 @192/28²) → ds (192→384) →
  stage3 (9 @384/14²) → ds (384→768) → stage4 (3 @768/7²) → GAP → head-LN → dense.

Per the handoff recipe (`planning/convnext_close.md` §"Scaling handoff"):
1. **Depth-k within a stage** — `CnxBlockParams` bundles the block 10-tuple;
   `convNextStageK (k) (ps : Fin k → CnxBlockParams …)` folds blocks head-first with
   VJP by induction (`vjp_comp` + the existing `convNextBlock_has_vjp` as the step) —
   the ViT depth-k recipe, simpler here (same-shape blocks within a stage).
2. **Downsample boundaries** — `cnxDownW` = `flatConvStride2(2×2) ∘ LN`; both VJPs existed.
3. **4×4/s4 patchify stem** — NEW `flatConvStride4` (= decimate ∘ decimateOdd ∘ stride-1
   SAME conv, `StridedConv.lean`: the left-aligned window `x[4i..4i+3]` of the paper's
   pad-0 `Conv2d(4, s=4)`) + the `flatConvStride4F` token.

GELU/LN/conv are smooth, so the whole-net VJP is GLOBAL (unconditional except the 10 LN
positivities) — ConvNeXt-T joins `efficientnetForwardB_full_has_vjp` and `vitForwardKV`.
Same scalar-LN representation caveat as the representative (`convNextBlockBody`'s doc);
the faithful channel-LN remains the optional follow-up. The `ConvNeXtClose`/
`ConvNeXtChainClose` param bridges are dim-generic and cover all 18 blocks verbatim;
the downsample conv W/b reuse the proven stride-2 bridges.
-/

namespace Proofs

open scoped BigOperators

-- ════════════════════════════════════════════════════════════════
-- § Per-block params + the stage depth-k fold
-- ════════════════════════════════════════════════════════════════

/-- One ConvNeXt block's 10 parameters (depthwise `kH×kW`, LN (scalar), expand `c→cExp`,
    project `cExp→c`, **per-channel** layer-scale `γls : Vec c` — the paper's form;
    it enters `convNextBlock` channel-expanded via `StableHLO.chanIdx`). -/
structure CnxBlockParams (c cExp h w kH kW : Nat) where
  Wdw : DepthwiseKernel c kH kW
  bdw : Vec c
  εn : ℝ
  γn : ℝ
  βn : ℝ
  Wex : Kernel4 cExp c 1 1
  bex : Vec cExp
  Wpr : Kernel4 c cExp 1 1
  bpr : Vec c
  γls : Vec c

/-- The per-channel layer-scale expanded to the flat `c·h·w` map (a constant
    reindex of the parameter — the layer-scale `convNextBlock` actually applies). -/
noncomputable def cnxGls {c cExp h w kH kW : Nat} (p : CnxBlockParams c cExp h w kH kW) :
    Vec (c * h * w) :=
  fun k => p.γls (StableHLO.chanIdx c h w k)

/-- `convNextBlock` at a bundled param block. -/
noncomputable def cnxBlockW {c cExp h w kH kW : Nat} (p : CnxBlockParams c cExp h w kH kW) :
    Vec (c * h * w) → Vec (c * h * w) :=
  convNextBlock p.Wdw p.bdw p.εn p.γn p.βn p.Wex p.bex p.Wpr p.bpr (cnxGls p)

/-- **Depth-`k` stage fold** (head recursion — block `0` runs first). -/
noncomputable def convNextStageK {c cExp h w kH kW : Nat} :
    (k : Nat) → (Fin k → CnxBlockParams c cExp h w kH kW) →
    Vec (c * h * w) → Vec (c * h * w)
  | 0, _ => fun v => v
  | k + 1, ps => convNextStageK k (fun i => ps i.succ) ∘ cnxBlockW (ps 0)

theorem convNextStageK_diff {c cExp h w kH kW : Nat} :
    ∀ (k : Nat) (ps : Fin k → CnxBlockParams c cExp h w kH kW),
      (∀ i, 0 < (ps i).εn) → Differentiable ℝ (convNextStageK k ps)
  | 0, _, _ => differentiable_id
  | k + 1, ps, hε =>
      (convNextStageK_diff k (fun i => ps i.succ) (fun i => hε i.succ)).comp
        (convNextBlock_differentiable (ps 0).Wdw (ps 0).bdw (ps 0).εn (hε 0)
          (ps 0).γn (ps 0).βn (ps 0).Wex (ps 0).bex (ps 0).Wpr (ps 0).bpr (cnxGls (ps 0)))

/-- **Depth-`k` stage VJP** — induction with `convNextBlock_has_vjp` as the chain step. -/
noncomputable def convNextStageK_has_vjp {c cExp h w kH kW : Nat} :
    (k : Nat) → (ps : Fin k → CnxBlockParams c cExp h w kH kW) →
    (∀ i, 0 < (ps i).εn) → HasVJP (convNextStageK k ps)
  | 0, _, _ => identity_has_vjp _
  | k + 1, ps, hε =>
      vjp_comp (cnxBlockW (ps 0)) (convNextStageK k (fun i => ps i.succ))
        (convNextBlock_differentiable (ps 0).Wdw (ps 0).bdw (ps 0).εn (hε 0)
          (ps 0).γn (ps 0).βn (ps 0).Wex (ps 0).bex (ps 0).Wpr (ps 0).bpr (cnxGls (ps 0)))
        (convNextStageK_diff k (fun i => ps i.succ) (fun i => hε i.succ))
        (convNextBlock_has_vjp (ps 0).Wdw (ps 0).bdw (ps 0).εn (hε 0)
          (ps 0).γn (ps 0).βn (ps 0).Wex (ps 0).bex (ps 0).Wpr (ps 0).bpr (cnxGls (ps 0)))
        (convNextStageK_has_vjp k (fun i => ps i.succ) (fun i => hε i.succ))

-- ════════════════════════════════════════════════════════════════
-- § Stage-boundary downsample (LN + 2×2/s2 conv)
-- ════════════════════════════════════════════════════════════════

/-- ConvNeXt downsample params: scalar LN + the 2×2/s2 widening conv. -/
structure CnxDownParams (cin cout : Nat) where
  ε : ℝ
  γ : ℝ
  β : ℝ
  W : Kernel4 cout cin 2 2
  b : Vec cout

/-- Stage-boundary downsample: `2×2/s2 conv ∘ LN` (`cin@2h×2w → cout@h×w`). -/
noncomputable def cnxDownW (h w : Nat) {cin cout : Nat} (p : CnxDownParams cin cout) :
    Vec (cin * (2 * h) * (2 * w)) → Vec (cout * h * w) :=
  flatConvStride2 (h := h) (w := w) p.W p.b ∘
    layerNormForward (cin * (2 * h) * (2 * w)) p.ε p.γ p.β

theorem cnxDownW_diff (h w : Nat) {cin cout : Nat} (p : CnxDownParams cin cout)
    (hε : 0 < p.ε) : Differentiable ℝ (cnxDownW h w p) := by
  unfold cnxDownW
  exact (flatConvStride2_differentiable p.W p.b).comp
    (bnForward_differentiable (cin * (2 * h) * (2 * w)) p.ε p.γ p.β hε)

noncomputable def cnxDownW_has_vjp (h w : Nat) {cin cout : Nat} (p : CnxDownParams cin cout)
    (hε : 0 < p.ε) : HasVJP (cnxDownW h w p) := by
  unfold cnxDownW
  exact vjp_comp _ _
    (bnForward_differentiable (cin * (2 * h) * (2 * w)) p.ε p.γ p.β hε)
    (flatConvStride2_differentiable p.W p.b)
    (layerNorm_has_vjp (cin * (2 * h) * (2 * w)) p.ε p.γ p.β hε)
    (flatConvStride2_has_vjp p.W p.b)

-- ════════════════════════════════════════════════════════════════
-- § The full ConvNeXt-T weights + forward + whole-net VJP
-- ════════════════════════════════════════════════════════════════

/-- All ConvNeXt-T parameters: 4×4/s4 stem (3→96) + stem-LN, the `[3,3,9,3]` stages
    (`Fin k → CnxBlockParams`, 7×7 depthwise, 4× expand), 3 downsamples
    (96→192→384→768), head-LN + dense (768→10). -/
structure CnxTWeights where
  sW : Kernel4 96 3 4 4
  sb : Vec 96
  sε : ℝ
  sγ : ℝ
  sβ : ℝ
  s1 : Fin 3 → CnxBlockParams 96 384 56 56 7 7
  d1 : CnxDownParams 96 192
  s2 : Fin 3 → CnxBlockParams 192 768 28 28 7 7
  d2 : CnxDownParams 192 384
  s3 : Fin 9 → CnxBlockParams 384 1536 14 14 7 7
  d3 : CnxDownParams 384 768
  s4 : Fin 3 → CnxBlockParams 768 3072 7 7 7 7
  hε : ℝ
  hγ : ℝ
  hβ : ℝ
  Wd : Mat 768 10
  bd : Vec 10

/-- **The full ConvNeXt-T forward** (3×224² → 10). Nested-application form (not `∘`),
    blocks opaque, so the forward-graph faithfulness closes by a structural `rfl` —
    the `efficientnetForwardB_full` recipe. -/
noncomputable def convNextForwardT (w : CnxTWeights) (x : Vec (3 * 224 * 224)) : Vec 10 :=
  dense w.Wd w.bd
    (layerNormForward 768 w.hε w.hγ w.hβ
      (globalAvgPoolFlat 768 7 7
        (convNextStageK 3 w.s4
          (cnxDownW 7 7 w.d3
            (convNextStageK 9 w.s3
              (cnxDownW 14 14 w.d2
                (convNextStageK 3 w.s2
                  (cnxDownW 28 28 w.d1
                    (convNextStageK 3 w.s1
                      (layerNormForward (96 * 56 * 56) w.sε w.sγ w.sβ
                        (flatConvStride4 (h := 56) (w := 56) w.sW w.sb x)))))))))))

/-- **The full ConvNeXt-T has a (correct) VJP — at every input.** GELU/LN/conv are
    smooth, so the only hypotheses are the 10 LN positivities (stem + 18 blocks via
    the per-stage `∀ i` + 3 downsamples + head). Stated on the `∘`-composition of the
    stages (= `convNextForwardT` by construction; keeps the blocks opaque so the
    `vjp_comp` chain closes structurally — the `efficientnetForwardB_full_has_vjp`
    recipe). The full-depth analogue of `convnext_has_vjp`. -/
noncomputable def convNextForwardT_has_vjp (w : CnxTWeights)
    (hsε : 0 < w.sε)
    (h1 : ∀ i, 0 < (w.s1 i).εn) (hd1 : 0 < w.d1.ε)
    (h2 : ∀ i, 0 < (w.s2 i).εn) (hd2 : 0 < w.d2.ε)
    (h3 : ∀ i, 0 < (w.s3 i).εn) (hd3 : 0 < w.d3.ε)
    (h4 : ∀ i, 0 < (w.s4 i).εn)
    (hhε : 0 < w.hε) :
    HasVJP
      (dense w.Wd w.bd ∘
        layerNormForward 768 w.hε w.hγ w.hβ ∘
        globalAvgPoolFlat 768 7 7 ∘
        convNextStageK 3 w.s4 ∘
        cnxDownW 7 7 w.d3 ∘
        convNextStageK 9 w.s3 ∘
        cnxDownW 14 14 w.d2 ∘
        convNextStageK 3 w.s2 ∘
        cnxDownW 28 28 w.d1 ∘
        convNextStageK 3 w.s1 ∘
        layerNormForward (96 * 56 * 56) w.sε w.sγ w.sβ ∘
        flatConvStride4 (h := 56) (w := 56) w.sW w.sb) := by
  have st_diff := flatConvStride4_differentiable (h := 56) (w := 56) w.sW w.sb
  have st_vjp := flatConvStride4_has_vjp (h := 56) (w := 56) w.sW w.sb
  have lns_diff := bnForward_differentiable (96 * 56 * 56) w.sε w.sγ w.sβ hsε
  have lns_vjp := layerNorm_has_vjp (96 * 56 * 56) w.sε w.sγ w.sβ hsε
  have e1 := vjp_comp _ _ st_diff lns_diff st_vjp lns_vjp
  have f1 := lns_diff.comp st_diff
  have s1d := convNextStageK_diff 3 w.s1 h1
  have e2 := vjp_comp _ _ f1 s1d e1 (convNextStageK_has_vjp 3 w.s1 h1)
  have f2 := s1d.comp f1
  have d1d := cnxDownW_diff 28 28 w.d1 hd1
  have e3 := vjp_comp _ _ f2 d1d e2 (cnxDownW_has_vjp 28 28 w.d1 hd1)
  have f3 := d1d.comp f2
  have s2d := convNextStageK_diff 3 w.s2 h2
  have e4 := vjp_comp _ _ f3 s2d e3 (convNextStageK_has_vjp 3 w.s2 h2)
  have f4 := s2d.comp f3
  have d2d := cnxDownW_diff 14 14 w.d2 hd2
  have e5 := vjp_comp _ _ f4 d2d e4 (cnxDownW_has_vjp 14 14 w.d2 hd2)
  have f5 := d2d.comp f4
  have s3d := convNextStageK_diff 9 w.s3 h3
  have e6 := vjp_comp _ _ f5 s3d e5 (convNextStageK_has_vjp 9 w.s3 h3)
  have f6 := s3d.comp f5
  have d3d := cnxDownW_diff 7 7 w.d3 hd3
  have e7 := vjp_comp _ _ f6 d3d e6 (cnxDownW_has_vjp 7 7 w.d3 hd3)
  have f7 := d3d.comp f6
  have s4d := convNextStageK_diff 3 w.s4 h4
  have e8 := vjp_comp _ _ f7 s4d e7 (convNextStageK_has_vjp 3 w.s4 h4)
  have f8 := s4d.comp f7
  have gap_diff := globalAvgPoolFlat_differentiable 768 7 7
  have e9 := vjp_comp _ _ f8 gap_diff e8 (globalAvgPoolFlat_has_vjp 768 7 7)
  have f9 := gap_diff.comp f8
  have lnh_diff := bnForward_differentiable 768 w.hε w.hγ w.hβ hhε
  have e10 := vjp_comp _ _ f9 lnh_diff e9 (layerNorm_has_vjp 768 w.hε w.hγ w.hβ hhε)
  have f10 := lnh_diff.comp f9
  exact vjp_comp _ _ f10 (dense_differentiable w.Wd w.bd) e10 (dense_has_vjp w.Wd w.bd)

/-- **`convNextForwardT` = the `∘`-chain of `convNextForwardT_has_vjp`'s statement** —
    the kernel-checked bridge between the nested-application and composition forms.
    PROOF-SHAPE MATTERS: the equation-lemma `rw` + 11 `Function.comp_apply` rewrites
    close SYNTACTICALLY, so the kernel never reduces anything. (A `simp`-closed or
    `rfl` proof of the same statement dies in the kernel: its closing defeq
    iota-unrolls the recursive `convNextStageK` folds at literal depth — the kernel
    ignores reducibility and has no defeq cache.) -/
theorem convNextForwardT_eq_chain (w : CnxTWeights) (x : Vec (3 * 224 * 224)) :
    convNextForwardT w x =
      (dense w.Wd w.bd ∘
        layerNormForward 768 w.hε w.hγ w.hβ ∘
        globalAvgPoolFlat 768 7 7 ∘
        convNextStageK 3 w.s4 ∘
        cnxDownW 7 7 w.d3 ∘
        convNextStageK 9 w.s3 ∘
        cnxDownW 14 14 w.d2 ∘
        convNextStageK 3 w.s2 ∘
        cnxDownW 28 28 w.d1 ∘
        convNextStageK 3 w.s1 ∘
        layerNormForward (96 * 56 * 56) w.sε w.sγ w.sβ ∘
        flatConvStride4 (h := 56) (w := 56) w.sW w.sb) x := by
  rw [convNextForwardT]
  rw [Function.comp_apply, Function.comp_apply, Function.comp_apply, Function.comp_apply,
      Function.comp_apply, Function.comp_apply, Function.comp_apply, Function.comp_apply,
      Function.comp_apply, Function.comp_apply, Function.comp_apply]

/-- **Public correctness theorem for `convNextForwardT_has_vjp`** — the full
    ConvNeXt-T's backward equals the `pdiv`-contracted Jacobian of
    `convNextForwardT` itself at every input, tying the chain-stated VJP back to
    the nested forward via `convNextForwardT_eq_chain`. -/
theorem convNextForwardT_has_vjp_correct (w : CnxTWeights)
    (hsε : 0 < w.sε)
    (h1 : ∀ i, 0 < (w.s1 i).εn) (hd1 : 0 < w.d1.ε)
    (h2 : ∀ i, 0 < (w.s2 i).εn) (hd2 : 0 < w.d2.ε)
    (h3 : ∀ i, 0 < (w.s3 i).εn) (hd3 : 0 < w.d3.ε)
    (h4 : ∀ i, 0 < (w.s4 i).εn)
    (hhε : 0 < w.hε)
    (x : Vec (3 * 224 * 224)) (dy : Vec 10) (i : Fin (3 * 224 * 224)) :
    (convNextForwardT_has_vjp w hsε h1 hd1 h2 hd2 h3 hd3 h4 hhε).backward x dy i =
      ∑ j : Fin 10, pdiv (convNextForwardT w) x i j * dy j := by
  have h := (convNextForwardT_has_vjp w hsε h1 hd1 h2 hd2 h3 hd3 h4 hhε).correct x dy i
  rwa [show convNextForwardT w =
        (dense w.Wd w.bd ∘
          layerNormForward 768 w.hε w.hγ w.hβ ∘
          globalAvgPoolFlat 768 7 7 ∘
          convNextStageK 3 w.s4 ∘
          cnxDownW 7 7 w.d3 ∘
          convNextStageK 9 w.s3 ∘
          cnxDownW 14 14 w.d2 ∘
          convNextStageK 3 w.s2 ∘
          cnxDownW 28 28 w.d1 ∘
          convNextStageK 3 w.s1 ∘
          layerNormForward (96 * 56 * 56) w.sε w.sγ w.sβ ∘
          flatConvStride4 (h := 56) (w := 56) w.sW w.sb)
      from funext (convNextForwardT_eq_chain w)]

-- ════════════════════════════════════════════════════════════════
-- § Committed-render config (no stem-LN) — for the render capstone
-- ════════════════════════════════════════════════════════════════

/-- **Committed-render config**: the committed `convnext_train_step.mlir` omits the
    paper's stem-LN (the patchify output feeds block `s0b0` directly — 180 params,
    not 182). This variant matches that signature exactly for the render capstone
    (`tests/TestConvNeXtTTrainPC.lean`); `convNextForwardT` keeps the paper form
    (`w.sε/sγ/sβ` are simply unused here). -/
noncomputable def convNextForwardTC (w : CnxTWeights) (x : Vec (3 * 224 * 224)) : Vec 10 :=
  dense w.Wd w.bd
    (layerNormForward 768 w.hε w.hγ w.hβ
      (globalAvgPoolFlat 768 7 7
        (convNextStageK 3 w.s4
          (cnxDownW 7 7 w.d3
            (convNextStageK 9 w.s3
              (cnxDownW 14 14 w.d2
                (convNextStageK 3 w.s2
                  (cnxDownW 28 28 w.d1
                    (convNextStageK 3 w.s1
                      (flatConvStride4 (h := 56) (w := 56) w.sW w.sb x))))))))))

/-- The committed-config whole-net VJP (chain-stated, as `convNextForwardT_has_vjp`;
    one fewer LN positivity — no stem-LN). -/
noncomputable def convNextForwardTC_has_vjp (w : CnxTWeights)
    (h1 : ∀ i, 0 < (w.s1 i).εn) (hd1 : 0 < w.d1.ε)
    (h2 : ∀ i, 0 < (w.s2 i).εn) (hd2 : 0 < w.d2.ε)
    (h3 : ∀ i, 0 < (w.s3 i).εn) (hd3 : 0 < w.d3.ε)
    (h4 : ∀ i, 0 < (w.s4 i).εn)
    (hhε : 0 < w.hε) :
    HasVJP
      (dense w.Wd w.bd ∘
        layerNormForward 768 w.hε w.hγ w.hβ ∘
        globalAvgPoolFlat 768 7 7 ∘
        convNextStageK 3 w.s4 ∘
        cnxDownW 7 7 w.d3 ∘
        convNextStageK 9 w.s3 ∘
        cnxDownW 14 14 w.d2 ∘
        convNextStageK 3 w.s2 ∘
        cnxDownW 28 28 w.d1 ∘
        convNextStageK 3 w.s1 ∘
        flatConvStride4 (h := 56) (w := 56) w.sW w.sb) := by
  have st_diff := flatConvStride4_differentiable (h := 56) (w := 56) w.sW w.sb
  have st_vjp := flatConvStride4_has_vjp (h := 56) (w := 56) w.sW w.sb
  have s1d := convNextStageK_diff 3 w.s1 h1
  have e2 := vjp_comp _ _ st_diff s1d st_vjp (convNextStageK_has_vjp 3 w.s1 h1)
  have f2 := s1d.comp st_diff
  have d1d := cnxDownW_diff 28 28 w.d1 hd1
  have e3 := vjp_comp _ _ f2 d1d e2 (cnxDownW_has_vjp 28 28 w.d1 hd1)
  have f3 := d1d.comp f2
  have s2d := convNextStageK_diff 3 w.s2 h2
  have e4 := vjp_comp _ _ f3 s2d e3 (convNextStageK_has_vjp 3 w.s2 h2)
  have f4 := s2d.comp f3
  have d2d := cnxDownW_diff 14 14 w.d2 hd2
  have e5 := vjp_comp _ _ f4 d2d e4 (cnxDownW_has_vjp 14 14 w.d2 hd2)
  have f5 := d2d.comp f4
  have s3d := convNextStageK_diff 9 w.s3 h3
  have e6 := vjp_comp _ _ f5 s3d e5 (convNextStageK_has_vjp 9 w.s3 h3)
  have f6 := s3d.comp f5
  have d3d := cnxDownW_diff 7 7 w.d3 hd3
  have e7 := vjp_comp _ _ f6 d3d e6 (cnxDownW_has_vjp 7 7 w.d3 hd3)
  have f7 := d3d.comp f6
  have s4d := convNextStageK_diff 3 w.s4 h4
  have e8 := vjp_comp _ _ f7 s4d e7 (convNextStageK_has_vjp 3 w.s4 h4)
  have f8 := s4d.comp f7
  have gap_diff := globalAvgPoolFlat_differentiable 768 7 7
  have e9 := vjp_comp _ _ f8 gap_diff e8 (globalAvgPoolFlat_has_vjp 768 7 7)
  have f9 := gap_diff.comp f8
  have lnh_diff := bnForward_differentiable 768 w.hε w.hγ w.hβ hhε
  have e10 := vjp_comp _ _ f9 lnh_diff e9 (layerNorm_has_vjp 768 w.hε w.hγ w.hβ hhε)
  have f10 := lnh_diff.comp f9
  exact vjp_comp _ _ f10 (dense_differentiable w.Wd w.bd) e10 (dense_has_vjp w.Wd w.bd)

/-- The committed-config nested↔chain bridge (see `convNextForwardT_eq_chain` for
    why the proof shape matters). -/
theorem convNextForwardTC_eq_chain (w : CnxTWeights) (x : Vec (3 * 224 * 224)) :
    convNextForwardTC w x =
      (dense w.Wd w.bd ∘
        layerNormForward 768 w.hε w.hγ w.hβ ∘
        globalAvgPoolFlat 768 7 7 ∘
        convNextStageK 3 w.s4 ∘
        cnxDownW 7 7 w.d3 ∘
        convNextStageK 9 w.s3 ∘
        cnxDownW 14 14 w.d2 ∘
        convNextStageK 3 w.s2 ∘
        cnxDownW 28 28 w.d1 ∘
        convNextStageK 3 w.s1 ∘
        flatConvStride4 (h := 56) (w := 56) w.sW w.sb) x := by
  rw [convNextForwardTC]
  rw [Function.comp_apply, Function.comp_apply, Function.comp_apply, Function.comp_apply,
      Function.comp_apply, Function.comp_apply, Function.comp_apply, Function.comp_apply,
      Function.comp_apply, Function.comp_apply]

/-- Committed-config correctness on `convNextForwardTC` itself (via the bridge). -/
theorem convNextForwardTC_has_vjp_correct (w : CnxTWeights)
    (h1 : ∀ i, 0 < (w.s1 i).εn) (hd1 : 0 < w.d1.ε)
    (h2 : ∀ i, 0 < (w.s2 i).εn) (hd2 : 0 < w.d2.ε)
    (h3 : ∀ i, 0 < (w.s3 i).εn) (hd3 : 0 < w.d3.ε)
    (h4 : ∀ i, 0 < (w.s4 i).εn)
    (hhε : 0 < w.hε)
    (x : Vec (3 * 224 * 224)) (dy : Vec 10) (i : Fin (3 * 224 * 224)) :
    (convNextForwardTC_has_vjp w h1 hd1 h2 hd2 h3 hd3 h4 hhε).backward x dy i =
      ∑ j : Fin 10, pdiv (convNextForwardTC w) x i j * dy j := by
  have h := (convNextForwardTC_has_vjp w h1 hd1 h2 hd2 h3 hd3 h4 hhε).correct x dy i
  rwa [show convNextForwardTC w =
        (dense w.Wd w.bd ∘
          layerNormForward 768 w.hε w.hγ w.hβ ∘
          globalAvgPoolFlat 768 7 7 ∘
          convNextStageK 3 w.s4 ∘
          cnxDownW 7 7 w.d3 ∘
          convNextStageK 9 w.s3 ∘
          cnxDownW 14 14 w.d2 ∘
          convNextStageK 3 w.s2 ∘
          cnxDownW 28 28 w.d1 ∘
          convNextStageK 3 w.s1 ∘
          flatConvStride4 (h := 56) (w := 56) w.sW w.sb)
      from funext (convNextForwardTC_eq_chain w)]

end Proofs

namespace Proofs.StableHLO

-- ════════════════════════════════════════════════════════════════
-- § Per-block / downsample graphs + the stage graph fold
-- ════════════════════════════════════════════════════════════════

/-- One ConvNeXt block graph at a bundled param block (the `convNextFwdGraph` block
    segment): depthwise → LN → 1×1 expand → GELU → 1×1 project → layerScale, then the
    `addV` identity skip (input subtree shared between both arms). -/
def cnxBlockGraphW (pfx epsStr : String) {c cExp h w kH kW : Nat}
    (p : CnxBlockParams c cExp h w kH kW) (e : SHlo (c * h * w)) : SHlo (c * h * w) :=
  .addV
    (.layerScaleChF s!"%{pfx}gls" p.γls
      (.flatConvF (h := h) (w := w) s!"%{pfx}Wpr" s!"%{pfx}bpr" p.Wpr p.bpr
        (.geluF
          (.flatConvF (h := h) (w := w) s!"%{pfx}Wex" s!"%{pfx}bex" p.Wex p.bex
            (.bnF s!"%{pfx}gn" s!"%{pfx}btn" epsStr p.εn p.γn p.βn
              (.depthwiseF (h := h) (w := w) s!"%{pfx}Wdw" s!"%{pfx}bdw" p.Wdw p.bdw e))))))
    e

theorem cnxBlockGraphW_faithful (pfx epsStr : String) {c cExp h w kH kW : Nat}
    (p : CnxBlockParams c cExp h w kH kW) (e : SHlo (c * h * w)) :
    den (cnxBlockGraphW pfx epsStr p e) = cnxBlockW p (den e) := by
  unfold cnxBlockGraphW cnxBlockW cnxGls convNextBlock convNextBlockBody residual biPath
         layerNormForward
  simp only [layerScaleChF_faithful, flatConvF_faithful, geluF_faithful, bnF_faithful,
             depthwiseF_faithful, den_addV, Function.comp_apply]

/-- **Depth-`k` stage graph fold** — block `base+1` first, SSA prefixes `b{base+1}_`…. -/
def cnxStageGraphK (epsStr : String) {c cExp h w kH kW : Nat} :
    (base k : Nat) → (Fin k → CnxBlockParams c cExp h w kH kW) →
    SHlo (c * h * w) → SHlo (c * h * w)
  | _, 0, _, e => e
  | base, k + 1, ps, e =>
      cnxStageGraphK epsStr (base + 1) k (fun i => ps i.succ)
        (cnxBlockGraphW s!"b{base + 1}_" epsStr (ps 0) e)

/-- The stage graph fold denotes the stage block fold — induction on `k`, chaining
    `cnxBlockGraphW_faithful` per block. -/
lemma cnxStageGraphK_den (epsStr : String) {c cExp h w kH kW : Nat} :
    ∀ (base k : Nat) (ps : Fin k → CnxBlockParams c cExp h w kH kW)
      (e : SHlo (c * h * w)),
      den (cnxStageGraphK epsStr base k ps e) = convNextStageK k ps (den e)
  | _, 0, _, _ => rfl
  | base, k + 1, ps, e => by
      have ih := cnxStageGraphK_den epsStr (base + 1) k (fun i => ps i.succ)
        (cnxBlockGraphW s!"b{base + 1}_" epsStr (ps 0) e)
      rw [show cnxStageGraphK epsStr base (k + 1) ps e =
            cnxStageGraphK epsStr (base + 1) k (fun i => ps i.succ)
              (cnxBlockGraphW s!"b{base + 1}_" epsStr (ps 0) e) from rfl,
          ih, cnxBlockGraphW_faithful]
      rfl

/-- Downsample graph: scalar LN → 2×2/s2 widening conv. -/
def cnxDownGraphW (pfx epsStr : String) (h w : Nat) {cin cout : Nat}
    (p : CnxDownParams cin cout) (e : SHlo (cin * (2 * h) * (2 * w))) :
    SHlo (cout * h * w) :=
  .flatConvStridedF (h := h) (w := w) s!"%{pfx}W" s!"%{pfx}b" p.W p.b
    (.bnF s!"%{pfx}g" s!"%{pfx}bt" epsStr p.ε p.γ p.β e)

theorem cnxDownGraphW_faithful (pfx epsStr : String) (h w : Nat) {cin cout : Nat}
    (p : CnxDownParams cin cout) (e : SHlo (cin * (2 * h) * (2 * w))) :
    den (cnxDownGraphW pfx epsStr h w p e) = cnxDownW h w p (den e) := by
  unfold cnxDownGraphW cnxDownW layerNormForward
  simp only [flatConvStridedF_faithful, bnF_faithful, Function.comp_apply]

-- ════════════════════════════════════════════════════════════════
-- § The full ConvNeXt-T forward graph + faithfulness
-- ════════════════════════════════════════════════════════════════

/-- The full **ConvNeXt-T forward graph** (3×224² → 10): `flatConvStride4F` patchify
    stem → stem-LN → the `[3,3,9,3]` stages (`b1_`…`b18_`) with the 3 LN+2×2/s2
    downsample boundaries → GAP → head-LN → dense. -/
def convNextFwdGraphT (epsStr : String) (w : CnxTWeights)
    (x : Vec (3 * 224 * 224)) : SHlo 10 :=
  denseF "%Wd" "%bd" w.Wd w.bd
    (.bnF "%ghd" "%bthd" epsStr w.hε w.hγ w.hβ
      (.gapF (c := 768) (h := 7) (w := 7)
        (cnxStageGraphK epsStr 15 3 w.s4
          (cnxDownGraphW "d3" epsStr 7 7 w.d3
            (cnxStageGraphK epsStr 6 9 w.s3
              (cnxDownGraphW "d2" epsStr 14 14 w.d2
                (cnxStageGraphK epsStr 3 3 w.s2
                  (cnxDownGraphW "d1" epsStr 28 28 w.d1
                    (cnxStageGraphK epsStr 0 3 w.s1
                      (.bnF "%gst" "%btst" epsStr w.sε w.sγ w.sβ
                        (.flatConvStride4F (h := 56) (w := 56) "%Wst" "%bst" w.sW w.sb
                          (.operand "%x" x))))))))))))

/-- **Full ConvNeXt-T forward faithfulness** — the `[3,3,9,3]` graph denotes
    `convNextForwardT`, chained from the stage den induction + the downsample/stem
    faithfulness (one `rw` per segment, outermost→innermost), then a structural `rfl`
    (the forward is nested-application form, blocks opaque — the
    `efficientnetFwdGraphB_full_faithful` recipe). The full-architecture apex for
    ConvNeXt. -/
theorem convNextFwdGraphT_faithful (epsStr : String) (w : CnxTWeights)
    (x : Vec (3 * 224 * 224)) :
    den (convNextFwdGraphT epsStr w x) = convNextForwardT w x := by
  rw [convNextFwdGraphT, denseF_faithful, bnF_faithful, gapF_faithful,
      cnxStageGraphK_den, cnxDownGraphW_faithful,
      cnxStageGraphK_den, cnxDownGraphW_faithful,
      cnxStageGraphK_den, cnxDownGraphW_faithful,
      cnxStageGraphK_den, bnF_faithful, flatConvStride4F_faithful, den_operand]
  rfl

-- ════════════════════════════════════════════════════════════════
-- § Committed-render config graph (no stem-LN) + faithfulness
-- ════════════════════════════════════════════════════════════════

/-- The committed-config forward graph: `convNextFwdGraphT` minus the stem-LN
    (matching the committed 180-param `convnext_train_step.mlir` exactly). -/
def convNextFwdGraphTC (epsStr : String) (w : CnxTWeights)
    (x : Vec (3 * 224 * 224)) : SHlo 10 :=
  denseF "%Wd" "%bd" w.Wd w.bd
    (.bnF "%ghd" "%bthd" epsStr w.hε w.hγ w.hβ
      (.gapF (c := 768) (h := 7) (w := 7)
        (cnxStageGraphK epsStr 15 3 w.s4
          (cnxDownGraphW "d3" epsStr 7 7 w.d3
            (cnxStageGraphK epsStr 6 9 w.s3
              (cnxDownGraphW "d2" epsStr 14 14 w.d2
                (cnxStageGraphK epsStr 3 3 w.s2
                  (cnxDownGraphW "d1" epsStr 28 28 w.d1
                    (cnxStageGraphK epsStr 0 3 w.s1
                      (.flatConvStride4F (h := 56) (w := 56) "%Wst" "%bst" w.sW w.sb
                        (.operand "%x" x)))))))))))

/-- **Committed-config forward faithfulness** — same `rw` chain as the apex, one
    `bnF` fewer. The graph `tests/TestConvNeXtTTrainPC.lean` renders. -/
theorem convNextFwdGraphTC_faithful (epsStr : String) (w : CnxTWeights)
    (x : Vec (3 * 224 * 224)) :
    den (convNextFwdGraphTC epsStr w x) = convNextForwardTC w x := by
  rw [convNextFwdGraphTC, denseF_faithful, bnF_faithful, gapF_faithful,
      cnxStageGraphK_den, cnxDownGraphW_faithful,
      cnxStageGraphK_den, cnxDownGraphW_faithful,
      cnxStageGraphK_den, cnxDownGraphW_faithful,
      cnxStageGraphK_den, flatConvStride4F_faithful, den_operand]
  rfl

end Proofs.StableHLO

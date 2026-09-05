import LeanMlir.Proofs.Codegen.MobileNetV2RenderPCEval

/-! # The PAPER-SPEC MobileNetV2 at INFERENCE — all 17 bottlenecks, forward + graph + faithfulness

The eval twin of `MobileNetV2FullPaper.lean`. That file states the seventeen-block `[t,c,n,s]` net
at TRAINING BatchNorm, the world its VJP and its typed graph live in; this file states the same
ladder at INFERENCE BatchNorm — frozen running statistics at all **52** sites, one shared `ε`, as
`mobilenetv2Forward_full_pc_eval` and the shipped `mobilenetv2_fwd_eval` both do — and proves its
typed `SHlo` graph denotes it. It exists so that `MobileNetV2PaperFloatBudget.lean`'s number can
be restated with the rendered net on the real side (`mnv2Paper_float_logits_le_committed`), which
is the rung `MobileNetV2FloatBudget.lean` has at six blocks and the seventeen-block file lacked
(`planning/proofs_tier_to_paper_nets.md` 3.2(e)).

Pure enumeration and chaining of `MobileNetV2RenderPCEval.lean`'s four inference stage
abbreviations (`ivExpandPCEval` / `ivDepthwisePCEval` / `ivDepthwiseStridedPCEval` /
`ivProjectPCEval`) and its two bodies, generic in the class count. No new mathematics and no new
tokens: every BatchNorm node's `den` is `bnPerChannelEvalTensor3`, proved once.

⭐ **The SSA names are the committed ones, and that is the point of this file's graph.** The
six-block eval graph (`mobilenetv2FwdGraphFullPCEval`) names its statistic slots `%mue1`/`%vare1`,
which matches no artifact and could not, since its net has none; the seventeen-block TRAINING
graph names its parameters `%b17gp`/`%b17btp`, where the render emits `%gp17`/`%btp17`. This
file's graph carries `bnSiteP`'s names verbatim: `%stnmu`/`%stnvar` for the stem, `%b{k}enmu`,
`%b{k}dnmu`, `%b{k}pnmu` and their `nvar` peers per block, `%hnmu`/`%hnvar` for the head, around
`irSig`/`irSigNoExp`'s `%We{k}`/`%ge{k}`/`%bte{k}`/`%Wd{k}`/`%gd{k}`/`%btd{k}`/`%Wp{k}`/`%gp{k}`/
`%btp{k}`. Names are pretty-printing metadata and do not enter `den`; matching them is what lets a
reader diff the typed graph against the committed text line for line.

**What it is tied to.** `mobilenetv2_fwd_eval.mlir` is THIS net: 263 inputs — `%x`, 158 parameter
tensors (`paperSig` at `convBias := false`, which is why the graph's bias slots `%bs`/`%bd{k}`/…
have no argument: the render folds each conv bias into the BatchNorm that follows it) and 104
statistic slots (52 sites × μ, var) — with `mobilenetv2in_fwd_eval.mlir` its 1000-class twin. The
classifier here is generic in `nCls`, so one theorem covers both.

Paper `[t,c,n,s]` spec (stem 3×3-s2 3→32 at the XLA-`SAME` phase; head 1×1 320→1280 → GAP → dense):
  (1, 16,1,1) (6, 24,2,2) (6, 32,3,2) (6, 64,4,2) (6, 96,3,1) (6,160,3,2) (6,320,1,1)
-/

namespace Proofs

-- ════════════════════════════════════════════════════════════════
-- § Per-block weight bundles at inference (γ, β and the two frozen statistics per BN site)
-- ════════════════════════════════════════════════════════════════

/-- Weights and running statistics of one MobileNetV2 bottleneck at inference. ⚠ No per-site `ε`:
    the eval forward takes ONE shared `ε`, as `mobilenetv2Forward_full_pc_eval` does and as the
    render emits (a single `eps` constant), where the training bundle `IVW` carries one per site. -/
structure IVWEval (ic mid oc : Nat) where
  eW : Kernel4 mid ic 1 1
  eb : Vec mid
  eγ : Vec mid
  eβ : Vec mid
  eμ : Vec mid
  ev : Vec mid
  dW : DepthwiseKernel mid 3 3
  db : Vec mid
  dγ : Vec mid
  dβ : Vec mid
  dμ : Vec mid
  dv : Vec mid
  pW : Kernel4 oc mid 1 1
  pb : Vec oc
  pγ : Vec oc
  pβ : Vec oc
  pμ : Vec oc
  pv : Vec oc

/-- Weights and running statistics of the t=1 first bottleneck (no expand conv) at inference. -/
structure IVWNoExpEval (ic oc : Nat) where
  dW : DepthwiseKernel ic 3 3
  db : Vec ic
  dγ : Vec ic
  dβ : Vec ic
  dμ : Vec ic
  dv : Vec ic
  pW : Kernel4 oc ic 1 1
  pb : Vec oc
  pγ : Vec oc
  pβ : Vec oc
  pμ : Vec oc
  pv : Vec oc

/-- All paper-spec MobileNetV2 parameters and running statistics at inference: stem (3×3-s2 3→32)
    + the 17 bottlenecks of the `[t,c,n,s]` table + head (1×1 320→1280) + dense (1280→`nCls`).
    52 BatchNorm sites, each with its frozen `μ` and `v`. -/
structure MNV2PaperWeightsEval (nCls : Nat) where
  sW : Kernel4 32 3 3 3
  sb : Vec 32
  sγ : Vec 32
  sβ : Vec 32
  sμ : Vec 32
  sv : Vec 32
  b1 : IVWNoExpEval 32 16
  b2 : IVWEval 16 96 24
  b3 : IVWEval 24 144 24
  b4 : IVWEval 24 144 32
  b5 : IVWEval 32 192 32
  b6 : IVWEval 32 192 32
  b7 : IVWEval 32 192 64
  b8 : IVWEval 64 384 64
  b9 : IVWEval 64 384 64
  b10 : IVWEval 64 384 64
  b11 : IVWEval 64 384 96
  b12 : IVWEval 96 576 96
  b13 : IVWEval 96 576 96
  b14 : IVWEval 96 576 160
  b15 : IVWEval 160 960 160
  b16 : IVWEval 160 960 160
  b17 : IVWEval 160 960 320
  hW : Kernel4 1280 320 1 1
  hb : Vec 1280
  hγ : Vec 1280
  hβ : Vec 1280
  hμ : Vec 1280
  hv : Vec 1280
  fcW : Mat 1280 nCls
  fcb : Vec nCls

variable {nCls : Nat}

-- ════════════════════════════════════════════════════════════════
-- § Weight-bundle wrappers (forward ℝ-fns) — `(h w)` and `ε` explicit, dims from the bundle
-- ════════════════════════════════════════════════════════════════

/-- t=1 bottleneck at inference (no expand, no skip): `project ∘ depthwise`. -/
noncomputable def ivNoExpEvalW (h w : Nat) (ε : ℝ) {ic oc : Nat} (p : IVWNoExpEval ic oc) :
    Vec (ic * h * w) → Vec (oc * h * w) :=
  ivProjectPCEval (h := h) (w := w) p.pW p.pb ε p.pγ p.pβ p.pμ p.pv ∘
    ivDepthwisePCEval (h := h) (w := w) p.dW p.db ε p.dγ p.dβ p.dμ p.dv

/-- Stride-1 bottleneck WITHOUT skip (`ic ≠ oc`) at inference. -/
noncomputable def ivExpOnlyEvalW (h w : Nat) (ε : ℝ) {ic mid oc : Nat} (p : IVWEval ic mid oc) :
    Vec (ic * h * w) → Vec (oc * h * w) :=
  invresBodyPCEval (h := h) (w := w) ε p.eW p.eb p.eγ p.eβ p.eμ p.ev p.dW p.db p.dγ p.dβ p.dμ p.dv
    p.pW p.pb p.pγ p.pβ p.pμ p.pv

/-- Stride-1 bottleneck WITH the identity skip at inference. -/
noncomputable def ivResidEvalW (h w : Nat) (ε : ℝ) {c mid : Nat} (p : IVWEval c mid c) :
    Vec (c * h * w) → Vec (c * h * w) :=
  residual (invresBodyPCEval (h := h) (w := w) ε p.eW p.eb p.eγ p.eβ p.eμ p.ev
    p.dW p.db p.dγ p.dβ p.dμ p.dv p.pW p.pb p.pγ p.pβ p.pμ p.pv)

/-- Stride-2 downsampling bottleneck at inference (XLA-`SAME` depthwise). -/
noncomputable def ivStridedEvalW (h w : Nat) (ε : ℝ) {ic mid oc : Nat} (p : IVWEval ic mid oc) :
    Vec (ic * (2 * h) * (2 * w)) → Vec (oc * h * w) :=
  invresBodyStridedPCEval (h := h) (w := w) ε p.eW p.eb p.eγ p.eβ p.eμ p.ev
    p.dW p.db p.dγ p.dβ p.dμ p.dv p.pW p.pb p.pγ p.pβ p.pμ p.pv

-- ════════════════════════════════════════════════════════════════
-- § The full paper-spec inference ℝ-forward — all 17 bottlenecks, nested-application form
-- ════════════════════════════════════════════════════════════════

/-- **The seventeen-block MobileNetV2 inference forward** — `mobilenetv2ForwardPaper`'s ladder with
    frozen running statistics at all 52 BatchNorm sites, at one shared `ε`. Nested-application
    form (NOT `∘`), as the training twin, so the faithfulness proof closes by `simp` and `rfl`. -/
noncomputable def mobilenetv2ForwardPaperEval (ε : ℝ) (w : MNV2PaperWeightsEval nCls)
    (x : Vec (3 * 224 * 224)) : Vec nCls :=
  dense w.fcW w.fcb
    (globalAvgPoolFlat 1280 7 7
      (relu6 (1280 * 7 * 7) (bnPerChannelEvalTensor3 1280 7 7 ε w.hγ w.hβ w.hμ w.hv
        (flatConv (h := 7) (w := 7) w.hW w.hb
        (ivExpOnlyEvalW 7 7 ε w.b17
          (ivResidEvalW 7 7 ε w.b16
            (ivResidEvalW 7 7 ε w.b15
              (ivStridedEvalW 7 7 ε w.b14
                (ivResidEvalW 14 14 ε w.b13
                  (ivResidEvalW 14 14 ε w.b12
                    (ivExpOnlyEvalW 14 14 ε w.b11
                      (ivResidEvalW 14 14 ε w.b10
                        (ivResidEvalW 14 14 ε w.b9
                          (ivResidEvalW 14 14 ε w.b8
                            (ivStridedEvalW 14 14 ε w.b7
                              (ivResidEvalW 28 28 ε w.b6
                                (ivResidEvalW 28 28 ε w.b5
                                  (ivStridedEvalW 28 28 ε w.b4
                                    (ivResidEvalW 56 56 ε w.b3
                                      (ivStridedEvalW 56 56 ε w.b2
                                        (ivNoExpEvalW 112 112 ε w.b1
                                          (relu6 (32 * 112 * 112)
                                            (bnPerChannelEvalTensor3 32 112 112 ε w.sγ w.sβ w.sμ w.sv
                                              (flatConvStride2Xla (h := 112) (w := 112) w.sW w.sb x))))))))))))))))))))))))

namespace StableHLO

-- ════════════════════════════════════════════════════════════════
-- § Per-block-kind typed `SHlo` inference graphs + faithfulness
--   ⭐ `k` is the block INDEX, so the emitted names are `irSig`'s: `%We{k}`/`%ge{k}`/… and
--   `bnSiteP`'s statistics `%b{k}enmu`/`%b{k}envar`/… — the committed artifact's, not the
--   six-block eval graph's `%mue1`/`%vare1`.
-- ════════════════════════════════════════════════════════════════

/-- t=1 bottleneck inference graph: `bnEval ∘ conv1×1 ∘ relu6 ∘ bnEval ∘ depthwise`. -/
def ivNoExpGraphEvalW (k epsStr : String) (h w : Nat) (ε : ℝ) {ic oc : Nat}
    (p : IVWNoExpEval ic oc) (e : SHlo (ic * h * w)) : SHlo (oc * h * w) :=
  .bnPerChannelEvalF (oc := oc) (h := h) (w := w) s!"%gp{k}" s!"%btp{k}"
      s!"%b{k}pnmu" s!"%b{k}pnvar" epsStr ε p.pγ p.pβ p.pμ p.pv
    (.flatConvF (h := h) (w := w) s!"%Wp{k}" s!"%bp{k}" p.pW p.pb
      (.relu6F (.bnPerChannelEvalF (oc := ic) (h := h) (w := w) s!"%gd{k}" s!"%btd{k}"
          s!"%b{k}dnmu" s!"%b{k}dnvar" epsStr ε p.dγ p.dβ p.dμ p.dv
        (.depthwiseF (h := h) (w := w) s!"%Wd{k}" s!"%bd{k}" p.dW p.db e))))

theorem ivNoExpGraphEvalW_faithful (k epsStr : String) (h w : Nat) (ε : ℝ) {ic oc : Nat}
    (p : IVWNoExpEval ic oc) (e : SHlo (ic * h * w)) :
    den (ivNoExpGraphEvalW k epsStr h w ε p e) = ivNoExpEvalW h w ε p (den e) := by
  unfold ivNoExpGraphEvalW ivNoExpEvalW
  simp only [bnPerChannelEvalF_faithful, flatConvF_faithful, relu6F_faithful, depthwiseF_faithful]
  simp only [ivProjectPCEval, ivDepthwisePCEval, Function.comp_apply]

/-- Stride-1 no-skip bottleneck inference graph: expand → depthwise → project. -/
def ivExpOnlyGraphEvalW (k epsStr : String) (h w : Nat) (ε : ℝ) {ic mid oc : Nat}
    (p : IVWEval ic mid oc) (e : SHlo (ic * h * w)) : SHlo (oc * h * w) :=
  .bnPerChannelEvalF (oc := oc) (h := h) (w := w) s!"%gp{k}" s!"%btp{k}"
      s!"%b{k}pnmu" s!"%b{k}pnvar" epsStr ε p.pγ p.pβ p.pμ p.pv
    (.flatConvF (h := h) (w := w) s!"%Wp{k}" s!"%bp{k}" p.pW p.pb
      (.relu6F (.bnPerChannelEvalF (oc := mid) (h := h) (w := w) s!"%gd{k}" s!"%btd{k}"
          s!"%b{k}dnmu" s!"%b{k}dnvar" epsStr ε p.dγ p.dβ p.dμ p.dv
        (.depthwiseF (h := h) (w := w) s!"%Wd{k}" s!"%bd{k}" p.dW p.db
          (.relu6F (.bnPerChannelEvalF (oc := mid) (h := h) (w := w) s!"%ge{k}" s!"%bte{k}"
              s!"%b{k}enmu" s!"%b{k}envar" epsStr ε p.eγ p.eβ p.eμ p.ev
            (.flatConvF (h := h) (w := w) s!"%We{k}" s!"%be{k}" p.eW p.eb e)))))))

theorem ivExpOnlyGraphEvalW_faithful (k epsStr : String) (h w : Nat) (ε : ℝ) {ic mid oc : Nat}
    (p : IVWEval ic mid oc) (e : SHlo (ic * h * w)) :
    den (ivExpOnlyGraphEvalW k epsStr h w ε p e) = ivExpOnlyEvalW h w ε p (den e) := by
  unfold ivExpOnlyGraphEvalW ivExpOnlyEvalW
  simp only [bnPerChannelEvalF_faithful, flatConvF_faithful, relu6F_faithful, depthwiseF_faithful]
  simp only [invresBodyPCEval, ivExpandPCEval, ivDepthwisePCEval, ivProjectPCEval,
             Function.comp_apply]

/-- Stride-1 skip bottleneck inference graph: the body + the `addV` identity skip. -/
def ivResidGraphEvalW (k epsStr : String) (h w : Nat) (ε : ℝ) {c mid : Nat}
    (p : IVWEval c mid c) (e : SHlo (c * h * w)) : SHlo (c * h * w) :=
  .addV (ivExpOnlyGraphEvalW k epsStr h w ε p e) e

theorem ivResidGraphEvalW_faithful (k epsStr : String) (h w : Nat) (ε : ℝ) {c mid : Nat}
    (p : IVWEval c mid c) (e : SHlo (c * h * w)) :
    den (ivResidGraphEvalW k epsStr h w ε p e) = ivResidEvalW h w ε p (den e) := by
  unfold ivResidGraphEvalW ivResidEvalW
  simp only [den_addV, ivExpOnlyGraphEvalW_faithful]
  unfold ivExpOnlyEvalW residual biPath
  rfl

/-- Stride-2 downsampling bottleneck inference graph: expand at `2h×2w` → XLA-`SAME` strided
    depthwise → project. -/
def ivStridedGraphEvalW (k epsStr : String) (h w : Nat) (ε : ℝ) {ic mid oc : Nat}
    (p : IVWEval ic mid oc) (e : SHlo (ic * (2 * h) * (2 * w))) : SHlo (oc * h * w) :=
  .bnPerChannelEvalF (oc := oc) (h := h) (w := w) s!"%gp{k}" s!"%btp{k}"
      s!"%b{k}pnmu" s!"%b{k}pnvar" epsStr ε p.pγ p.pβ p.pμ p.pv
    (.flatConvF (h := h) (w := w) s!"%Wp{k}" s!"%bp{k}" p.pW p.pb
      (.relu6F (.bnPerChannelEvalF (oc := mid) (h := h) (w := w) s!"%gd{k}" s!"%btd{k}"
          s!"%b{k}dnmu" s!"%b{k}dnvar" epsStr ε p.dγ p.dβ p.dμ p.dv
        (.depthwiseStridedXlaF (h := h) (w := w) s!"%Wd{k}" s!"%bd{k}" p.dW p.db
          (.relu6F (.bnPerChannelEvalF (oc := mid) (h := 2 * h) (w := 2 * w) s!"%ge{k}"
              s!"%bte{k}" s!"%b{k}enmu" s!"%b{k}envar" epsStr ε p.eγ p.eβ p.eμ p.ev
            (.flatConvF (h := 2 * h) (w := 2 * w) s!"%We{k}" s!"%be{k}" p.eW p.eb e)))))))

theorem ivStridedGraphEvalW_faithful (k epsStr : String) (h w : Nat) (ε : ℝ) {ic mid oc : Nat}
    (p : IVWEval ic mid oc) (e : SHlo (ic * (2 * h) * (2 * w))) :
    den (ivStridedGraphEvalW k epsStr h w ε p e) = ivStridedEvalW h w ε p (den e) := by
  unfold ivStridedGraphEvalW ivStridedEvalW
  simp only [bnPerChannelEvalF_faithful, flatConvF_faithful, relu6F_faithful,
             depthwiseStridedXlaF_faithful]
  simp only [invresBodyStridedPCEval, ivExpandPCEval, ivDepthwiseStridedPCEval, ivProjectPCEval,
             Function.comp_apply]

-- ════════════════════════════════════════════════════════════════
-- § The full paper-spec inference forward graph + faithfulness (all 17 bottlenecks)
-- ════════════════════════════════════════════════════════════════

/-- The full **paper-spec MobileNetV2 inference forward graph** (3×224² → `nCls`): XLA-`SAME`
    strided stem → the 17 bottlenecks of the `[t,c,n,s]` table → 1×1 conv-bn-relu6 head → GAP →
    dense, every one of the 52 BatchNorm sites reading frozen running statistics through
    `bnPerChannelEvalF`. The typed form of the shipped `mobilenetv2_fwd_eval`. -/
def mobilenetv2FwdGraphPaperEval (epsStr : String) (ε : ℝ) (w : MNV2PaperWeightsEval nCls)
    (x : Vec (3 * 224 * 224)) : SHlo nCls :=
  denseF "%Wfc" "%bfc" w.fcW w.fcb
    (.gapF (c := 1280) (h := 7) (w := 7)
      (.relu6F (.bnPerChannelEvalF (oc := 1280) (h := 7) (w := 7) "%gh" "%bth"
          "%hnmu" "%hnvar" epsStr ε w.hγ w.hβ w.hμ w.hv
        (.flatConvF (h := 7) (w := 7) "%Wh" "%bh" w.hW w.hb
        (ivExpOnlyGraphEvalW "17" epsStr 7 7 ε w.b17
          (ivResidGraphEvalW "16" epsStr 7 7 ε w.b16
            (ivResidGraphEvalW "15" epsStr 7 7 ε w.b15
              (ivStridedGraphEvalW "14" epsStr 7 7 ε w.b14
                (ivResidGraphEvalW "13" epsStr 14 14 ε w.b13
                  (ivResidGraphEvalW "12" epsStr 14 14 ε w.b12
                    (ivExpOnlyGraphEvalW "11" epsStr 14 14 ε w.b11
                      (ivResidGraphEvalW "10" epsStr 14 14 ε w.b10
                        (ivResidGraphEvalW "9" epsStr 14 14 ε w.b9
                          (ivResidGraphEvalW "8" epsStr 14 14 ε w.b8
                            (ivStridedGraphEvalW "7" epsStr 14 14 ε w.b7
                              (ivResidGraphEvalW "6" epsStr 28 28 ε w.b6
                                (ivResidGraphEvalW "5" epsStr 28 28 ε w.b5
                                  (ivStridedGraphEvalW "4" epsStr 28 28 ε w.b4
                                    (ivResidGraphEvalW "3" epsStr 56 56 ε w.b3
                                      (ivStridedGraphEvalW "2" epsStr 56 56 ε w.b2
                                        (ivNoExpGraphEvalW "1" epsStr 112 112 ε w.b1
                                          (.relu6F (.bnPerChannelEvalF (oc := 32) (h := 112) (w := 112) "%gs" "%bts"
                                            "%stnmu" "%stnvar" epsStr ε w.sγ w.sβ w.sμ w.sv
                                            (.flatConvStridedXlaF (h := 112) (w := 112) "%Ws" "%bs" w.sW w.sb
                                              (.operand "%x" x)))))))))))))))))))))))))

set_option maxRecDepth 20000 in
/-- ⭐ **Seventeen-block inference MobileNetV2 forward faithfulness.** The typed graph denotes
    `mobilenetv2ForwardPaperEval` — chained from the per-block-kind `*GraphEvalW_faithful` lemmas
    and then a structural `rfl`, the training twin's recipe with `bnPerChannelEvalF_faithful` in
    place of `bnPerChannelF_faithful`. -/
theorem mobilenetv2FwdGraphPaperEval_faithful (epsStr : String) (ε : ℝ)
    (w : MNV2PaperWeightsEval nCls) (x : Vec (3 * 224 * 224)) :
    den (mobilenetv2FwdGraphPaperEval epsStr ε w x) = mobilenetv2ForwardPaperEval ε w x := by
  simp only [mobilenetv2FwdGraphPaperEval, denseF_faithful, gapF_faithful, relu6F_faithful,
             bnPerChannelEvalF_faithful, flatConvF_faithful, flatConvStridedXlaF_faithful,
             ivExpOnlyGraphEvalW_faithful, ivResidGraphEvalW_faithful,
             ivStridedGraphEvalW_faithful, ivNoExpGraphEvalW_faithful, den_operand]
  rfl

end StableHLO
end Proofs

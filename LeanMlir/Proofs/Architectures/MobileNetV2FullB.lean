import LeanMlir.Proofs.Architectures.MobileNetV2BackB0
import LeanMlir.Proofs.Architectures.MobileNetV2FullPaper

/-! # MobileNetV2 at TRUE BATCH-NORM — the whole net's forward and graph (T1-forward, T2)

The MobileNetV2 peer of `ResNet34FullB.lean`, and the first half of
`planning/archive/proofs_tier_to_paper_nets.md` section 4.2's MobileNetV2 column.

`MobileNetV2FullPaper.lean` states this net's whole-net ℝ forward and typed graph at **per-example**
BatchNorm (`bnPerChannelTensor3`, reduce `[2,3]`). That was the world of `mobilenetv2_fwd.mlir` and
the Imagenette SGD trainer `mobilenetv2_train_step.mlir`, and every tier built on it is true and was
correctly paired with those bytes. It is NOT the world of `mobilenetv2_adam_train_step.mlir`,
`mobilenetv2_rms_train_step.mlir` or any ImageNet artifact — including `mobilenetv2in_rmsdp64`,
whose accuracy the book quotes — all of which reduce `[0,2,3]`: one mu/var per channel across the
batch, the one op that couples examples.

⛔ **MobileNetV2's two renderers did not overlap**, so this was not a flag away.
`MobileNetV2Render` was SGD-inline and per-example only; `MobileNetV2RenderB` is AdamW/RMSProp-only,
at the batched index and at batch BatchNorm. This file re-states the ladder at `bnBatchLA` (= the
proven `bnBatchTensor4` at the network's left-assoc index), which is that renderer's world. ⭐ Since
4c leg 2 (2026-09-06) it is the ONLY renderer: the per-example one and its train step are retired
and `mobilenetv2_fwd.mlir` comes from the batched chain too, so this file's world is now the whole
net's.

## What is new here, and what is not

⭐⭐ **Nothing about the blocks is new.** `MobileNetV2BackB0.lean` already carries the batched
relu6 stages (`cbrB`, `dwbrB`, `dwbrBstrided`, and `projB` from `EfficientNetRenderPC.lean`), their
`_at` VJPs and their backward-graph faithfulness, all at `bnBatchLA`. What was missing is the level
above: a net-level ℝ forward, a net-level forward graph, and the faithfulness tying them. This file
is that enumeration.

⭐ **The weight bundles are reused, not re-declared.** `IVW` / `IVWNoExp`
(`MobileNetV2FullPaper.lean`) hold kernels, epsilons, gammas and betas — nothing that knows which
BatchNorm world reduces them — so the batched net binds the same records the per-example one does.
Only the top-level bundle is new, because it is generic in the class count where
`MNV2PaperWeights` is pinned at 10.

⚠ **Padding is XLA-`SAME` at all five stride-2 sites** — the stem 3x3/s2 conv and the four stride-2
depthwises (`b2`, `b4`, `b7`, `b14`). These are `flatConvStride2Xla` / `depthwiseStride2FlatXla`,
NOT r34's symmetric `flatConvStride2` peers; the two families have identical types and identical
emitted shapes, so only the certificate distinguishes them, and MobileNetV2 is the TF-origin net.
`scripts/convention_audit.py` sees this at the artifact tier and nothing sees it here, so it is
stated.

⚠ **There is no max-pool.** MobileNetV2's stem is conv-BN-relu6 and downsamples once; r34's stem is
conv-BN-relu, then a 3x3/s2 pool. That is why this net needs no `batchMap_has_vjp_at`.

## Conventions this net runs at

| | |
|---|---|
| depth | 17 bottlenecks, the paper `[t,c,n,s]` table, stem 32 to head 1280 |
| BatchNorm | **batch** (`bnBatchLA`, reduce `[0,2,3]`, width `N*h*w`), 52 sites |
| activation | relu6 (TWO kinks, at 0 and at 6); 35 sites, none after a `project` |
| stride-2 padding | XLA-`SAME` at all five sites |
| stem | 3x3/s2 conv-bn-relu6, 3 to 32, 224 to 112 (NO pool) |
| head | 1x1 conv-bn-relu6 320 to 1280, then GAP and dense, generic in the class count |
| artifacts | `mobilenetv2_fwd` and every train step — this net now has ONE chain (4c leg 2) |

⭐ The head is generic in `nCls`, so one statement covers the 10-class Imagenette artifacts and the
1000-class `mobilenetv2in` ones.

⚠ `N` stays a variable throughout. T1 and T2 carry no numerals, so the batch size does not need
pinning here; it is pinned only where a `Maps` envelope turns a width into a rational (T4/T5), and
the artifacts' `N` is the PER-REPLICA batch (64 on the data-parallel runs) because the collectives
average gradients and no BatchNorm statistic is all-reduced.

⚠ **The bias operand names are the render's DEFAULT `convBias := false` ones** — `%zb{c}`, the
shared zero constant each conv, depthwise and project bias is bound to once its real bias has been
folded into the BatchNorm that follows it. That is what makes the shipped parameter census 158 and
not 210. Every graph below is `∀`-quantified over the bias VALUE, so it covers the
`convBias := true` render too; only the name would differ there.
-/

namespace Proofs

open scoped BigOperators

-- ════════════════════════════════════════════════════════════════
-- § The whole net's weights — the 17-block `[t,c,n,s]` table, generic in the class count
-- ════════════════════════════════════════════════════════════════

/-- Every paper-spec MobileNetV2 parameter: stem (3x3/s2, 3 to 32) + the 17 bottlenecks of the
    `[t,c,n,s]` table + the 1x1 head (320 to 1280) + the dense classifier. The batch-BN peer of
    `MNV2PaperWeights`, generic in `nCls` — the per-example record is pinned at 10, and the lesson
    `MobileNetV2FullPaperEval.lean` and B0's eval twin both paid for is that the head's envelope
    depends on the fan-in and never on the output count. -/
structure MNV2BWeights (nCls : Nat) where
  sW : Kernel4 32 3 3 3
  sb : Vec 32
  sε : ℝ
  sγ : Vec 32
  sβ : Vec 32
  b1 : IVWNoExp 32 16
  b2 : IVW 16 96 24
  b3 : IVW 24 144 24
  b4 : IVW 24 144 32
  b5 : IVW 32 192 32
  b6 : IVW 32 192 32
  b7 : IVW 32 192 64
  b8 : IVW 64 384 64
  b9 : IVW 64 384 64
  b10 : IVW 64 384 64
  b11 : IVW 64 384 96
  b12 : IVW 96 576 96
  b13 : IVW 96 576 96
  b14 : IVW 96 576 160
  b15 : IVW 160 960 160
  b16 : IVW 160 960 160
  b17 : IVW 160 960 320
  hW : Kernel4 1280 320 1 1
  hb : Vec 1280
  hε : ℝ
  hγ : Vec 1280
  hβ : Vec 1280
  fcW : Mat 1280 nCls
  fcb : Vec nCls

-- ════════════════════════════════════════════════════════════════
-- § Block real-forwards, batched
-- ════════════════════════════════════════════════════════════════

/-- Batched stem: 3x3/s2 XLA-`SAME` conv -> batch BN -> relu6. MobileNetV2 has no stem pool. -/
@[reducible] noncomputable def mnv2StemB (N h w : Nat) {ic oc kH kW : Nat}
    (Ws : Kernel4 oc ic kH kW) (bs : Vec oc) (εs : ℝ) (γs βs : Vec oc) :
    Vec (N * (ic * (2 * h) * (2 * w))) → Vec (N * (oc * h * w)) :=
  relu6 (N * (oc * h * w)) ∘ StableHLO.bnBatchLA N oc h w εs γs βs ∘
    StableHLO.batchMap N (flatConvStride2Xla Ws bs)

/-- Batched `t = 1` first bottleneck (b1): depthwise-bn-relu6 then the linear-bottleneck project.
    No expand conv, no skip. The one block shape with no `mnv2*BodyB` lemma to delegate to. -/
@[reducible] noncomputable def mnv2NoExpB (N h w : Nat) {ic oc : Nat} (p : IVWNoExp ic oc) :
    Vec (N * (ic * h * w)) → Vec (N * (oc * h * w)) :=
  projB N (h := h) (w := w) p.pW p.pb p.pε p.pγ p.pβ ∘
    StableHLO.dwbrB N (h := h) (w := w) p.dW p.db p.dε p.dγ p.dβ

/-- Batched stride-1 inverted-residual BODY, no skip (`ic ≠ oc`; `b11`, `b17`):
    `projB ∘ dwbrB ∘ cbrB`. -/
@[reducible] noncomputable def mnv2ExpOnlyB (N h w : Nat) {ic mid oc : Nat} (p : IVW ic mid oc) :
    Vec (N * (ic * h * w)) → Vec (N * (oc * h * w)) :=
  projB N (h := h) (w := w) p.pW p.pb p.pε p.pγ p.pβ ∘
    StableHLO.dwbrB N (h := h) (w := w) p.dW p.db p.dε p.dγ p.dβ ∘
      StableHLO.cbrB N (h := h) (w := w) p.eW p.eb p.eε p.eγ p.eβ

/-- Batched stride-1 inverted residual WITH the identity skip (`s = 1 ∧ ic = oc`). ⭐ The residual
    add IS the block output — unlike ResNet, there is no relu after it, which is why a MobileNetV2
    block carries two kink clauses and not three. -/
@[reducible] noncomputable def mnv2ResidB (N h w : Nat) {c mid : Nat} (p : IVW c mid c) :
    Vec (N * (c * h * w)) → Vec (N * (c * h * w)) :=
  residual (mnv2ExpOnlyB N h w p)

/-- Batched stride-2 downsampling bottleneck (no skip): expand at `2h x 2w`, the XLA-`SAME`
    strided depthwise halving spatial, then project at `h x w`. -/
@[reducible] noncomputable def mnv2StridedB (N h w : Nat) {ic mid oc : Nat} (p : IVW ic mid oc) :
    Vec (N * (ic * (2 * h) * (2 * w))) → Vec (N * (oc * h * w)) :=
  projB N (h := h) (w := w) p.pW p.pb p.pε p.pγ p.pβ ∘
    StableHLO.dwbrBstrided N (h := h) (w := w) p.dW p.db p.dε p.dγ p.dβ ∘
      StableHLO.cbrB N (h := 2 * h) (w := 2 * w) p.eW p.eb p.eε p.eγ p.eβ

/-- Batched head: 1x1 conv-bn-relu6 (`ic` to `oc`), global average pool, dense classifier. -/
@[reducible] noncomputable def mnv2HeadB (N h w : Nat) {ic oc nCls : Nat}
    (Wh : Kernel4 oc ic 1 1) (bh : Vec oc) (εh : ℝ) (γh βh : Vec oc)
    (Wd : Mat oc nCls) (bd : Vec nCls) : Vec (N * (ic * h * w)) → Vec (N * nCls) :=
  StableHLO.batchMap N (dense Wd bd) ∘ StableHLO.batchMap N (globalAvgPoolFlat oc h w) ∘
    StableHLO.cbrB N (h := h) (w := w) Wh bh εh γh βh

-- ════════════════════════════════════════════════════════════════
-- § The whole net, nested-application form
--   stem(224 -> 112) -> b1@112 -> b2(112->56) -> b3@56 -> b4(56->28) -> b5,b6@28
--   -> b7(28->14) -> b8..b13@14 -> b14(14->7) -> b15,b16,b17@7 -> head -> GAP -> dense
-- ════════════════════════════════════════════════════════════════

/-- **The full batch-BN MobileNetV2 forward**, `N*(3*224*224) -> N*nCls`. The batched peer of
    `mobilenetv2ForwardPaper`; nested-application form, as `resnet34ForwardB_full` and
    `efficientnetForwardB_full` both are, so a T6 tie can peel it one block at a time. -/
noncomputable def mobilenetv2ForwardB_full (N : Nat) {nCls : Nat} (w : MNV2BWeights nCls)
    (x : Vec (N * (3 * (2 * 112) * (2 * 112)))) : Vec (N * nCls) :=
  mnv2HeadB N 7 7 w.hW w.hb w.hε w.hγ w.hβ w.fcW w.fcb
    (mnv2ExpOnlyB N 7 7 w.b17
      (mnv2ResidB N 7 7 w.b16
        (mnv2ResidB N 7 7 w.b15
          (mnv2StridedB N 7 7 w.b14
            (mnv2ResidB N 14 14 w.b13
              (mnv2ResidB N 14 14 w.b12
                (mnv2ExpOnlyB N 14 14 w.b11
                  (mnv2ResidB N 14 14 w.b10
                    (mnv2ResidB N 14 14 w.b9
                      (mnv2ResidB N 14 14 w.b8
                        (mnv2StridedB N 14 14 w.b7
                          (mnv2ResidB N 28 28 w.b6
                            (mnv2ResidB N 28 28 w.b5
                              (mnv2StridedB N 28 28 w.b4
                                (mnv2ResidB N 56 56 w.b3
                                  (mnv2StridedB N 56 56 w.b2
                                    (mnv2NoExpB N 112 112 w.b1
                                      (mnv2StemB N 112 112 w.sW w.sb w.sε w.sγ w.sβ x))))))))))))))))))

namespace StableHLO

-- ════════════════════════════════════════════════════════════════
-- § Per-block typed graphs + faithfulness
--   Tokens are the ones `MobileNetV2RenderB.lean` emits: `.batchOp .convStridedXla` / `.conv` /
--   `.depthwise` / `.depthwiseStridedXla` / `.relu6` / `.gap` / `.dense`, `.bnBatchF` for the
--   batch-coupled norm, and `.addVB` for the identity skip.
-- ════════════════════════════════════════════════════════════════

/-- Stem graph: 3x3/s2 XLA-`SAME` conv -> batch BN -> relu6. -/
def mnv2StemGraphB (epsStr : String) (N h w : Nat) {ic oc kH kW : Nat}
    (Ws : Kernel4 oc ic kH kW) (bs : Vec oc) (εs : ℝ) (γs βs : Vec oc)
    (e : SHlo (N * (ic * (2 * h) * (2 * w)))) : SHlo (N * (oc * h * w)) :=
  .batchOp (N := N) (.relu6 (n := oc * h * w))
    (.bnBatchF "%sg" "%sbt" epsStr εs γs βs
      (.batchOp (N := N) (.convStridedXla (h := h) (w := w) "%sW" s!"%zb{oc}" Ws bs) e))

theorem mnv2StemGraphB_faithful (epsStr : String) (N h w : Nat) {ic oc kH kW : Nat}
    (Ws : Kernel4 oc ic kH kW) (bs : Vec oc) (εs : ℝ) (γs βs : Vec oc)
    (e : SHlo (N * (ic * (2 * h) * (2 * w)))) :
    den (mnv2StemGraphB epsStr N h w Ws bs εs γs βs e)
      = mnv2StemB N h w Ws bs εs γs βs (den e) := by
  unfold mnv2StemGraphB mnv2StemB
  simp only [den_batchOp_relu6_eq_relu6F, relu6F_faithful, den_batchOp_convStridedXla,
    den_bnBatchF, Function.comp_apply]

/-- `t = 1` bottleneck graph (b1): depthwise -> BN -> relu6 -> project 1x1 -> BN. -/
def mnv2NoExpGraphB (pfx epsStr : String) (N h w : Nat) {ic oc : Nat} (p : IVWNoExp ic oc)
    (e : SHlo (N * (ic * h * w))) : SHlo (N * (oc * h * w)) :=
  .bnBatchF s!"%b{pfx}pg" s!"%b{pfx}pbt" epsStr p.pε p.pγ p.pβ
    (.batchOp (N := N) (.conv (h := h) (w := w) s!"%b{pfx}pW" s!"%zb{oc}" p.pW p.pb)
      (.batchOp (N := N) (.relu6 (n := ic * h * w))
        (.bnBatchF s!"%b{pfx}dg" s!"%b{pfx}dbt" epsStr p.dε p.dγ p.dβ
          (.batchOp (N := N) (.depthwise (h := h) (w := w) s!"%b{pfx}dW" s!"%zb{ic}" p.dW p.db)
            e))))

theorem mnv2NoExpGraphB_faithful (pfx epsStr : String) (N h w : Nat) {ic oc : Nat}
    (p : IVWNoExp ic oc) (e : SHlo (N * (ic * h * w))) :
    den (mnv2NoExpGraphB pfx epsStr N h w p e) = mnv2NoExpB N h w p (den e) := by
  unfold mnv2NoExpGraphB mnv2NoExpB projB dwbrB
  simp only [den_batchOp_relu6_eq_relu6F, relu6F_faithful, den_batchOp_conv,
    den_batchOp_depthwise, den_bnBatchF, Function.comp_apply]

/-- Stride-1 no-skip bottleneck graph (b11, b17): expand -> depthwise -> project, batch BN after
    each, relu6 after the first two. -/
def mnv2ExpOnlyGraphB (pfx epsStr : String) (N h w : Nat) {ic mid oc : Nat} (p : IVW ic mid oc)
    (e : SHlo (N * (ic * h * w))) : SHlo (N * (oc * h * w)) :=
  .bnBatchF s!"%b{pfx}pg" s!"%b{pfx}pbt" epsStr p.pε p.pγ p.pβ
    (.batchOp (N := N) (.conv (h := h) (w := w) s!"%b{pfx}pW" s!"%zb{oc}" p.pW p.pb)
      (.batchOp (N := N) (.relu6 (n := mid * h * w))
        (.bnBatchF s!"%b{pfx}dg" s!"%b{pfx}dbt" epsStr p.dε p.dγ p.dβ
          (.batchOp (N := N) (.depthwise (h := h) (w := w) s!"%b{pfx}dW" s!"%zb{mid}" p.dW p.db)
            (.batchOp (N := N) (.relu6 (n := mid * h * w))
              (.bnBatchF s!"%b{pfx}eg" s!"%b{pfx}ebt" epsStr p.eε p.eγ p.eβ
                (.batchOp (N := N)
                  (.conv (h := h) (w := w) s!"%b{pfx}eW" s!"%zb{mid}" p.eW p.eb) e)))))))

theorem mnv2ExpOnlyGraphB_faithful (pfx epsStr : String) (N h w : Nat) {ic mid oc : Nat}
    (p : IVW ic mid oc) (e : SHlo (N * (ic * h * w))) :
    den (mnv2ExpOnlyGraphB pfx epsStr N h w p e) = mnv2ExpOnlyB N h w p (den e) := by
  unfold mnv2ExpOnlyGraphB mnv2ExpOnlyB projB dwbrB cbrB
  simp only [den_batchOp_relu6_eq_relu6F, relu6F_faithful, den_batchOp_conv,
    den_batchOp_depthwise, den_bnBatchF, Function.comp_apply]

/-- Stride-1 skip bottleneck graph: the body plus the `addVB` identity skip, the block-input
    subtree `e` shared between both arms as the render emits it. -/
def mnv2ResidGraphB (pfx epsStr : String) (N h w : Nat) {c mid : Nat} (p : IVW c mid c)
    (e : SHlo (N * (c * h * w))) : SHlo (N * (c * h * w)) :=
  .addVB (mnv2ExpOnlyGraphB pfx epsStr N h w p e) e

theorem mnv2ResidGraphB_faithful (pfx epsStr : String) (N h w : Nat) {c mid : Nat}
    (p : IVW c mid c) (e : SHlo (N * (c * h * w))) :
    den (mnv2ResidGraphB pfx epsStr N h w p e) = mnv2ResidB N h w p (den e) := by
  unfold mnv2ResidGraphB mnv2ResidB
  simp only [den_addVB, mnv2ExpOnlyGraphB_faithful]
  unfold residual biPath
  rfl

/-- Stride-2 downsampling bottleneck graph: expand at `2h x 2w`, XLA-`SAME` strided depthwise,
    project at `h x w`. -/
def mnv2StridedGraphB (pfx epsStr : String) (N h w : Nat) {ic mid oc : Nat} (p : IVW ic mid oc)
    (e : SHlo (N * (ic * (2 * h) * (2 * w)))) : SHlo (N * (oc * h * w)) :=
  .bnBatchF s!"%b{pfx}pg" s!"%b{pfx}pbt" epsStr p.pε p.pγ p.pβ
    (.batchOp (N := N) (.conv (h := h) (w := w) s!"%b{pfx}pW" s!"%zb{oc}" p.pW p.pb)
      (.batchOp (N := N) (.relu6 (n := mid * h * w))
        (.bnBatchF s!"%b{pfx}dg" s!"%b{pfx}dbt" epsStr p.dε p.dγ p.dβ
          (.batchOp (N := N)
            (.depthwiseStridedXla (h := h) (w := w) s!"%b{pfx}dW" s!"%zb{mid}" p.dW p.db)
            (.batchOp (N := N) (.relu6 (n := mid * (2 * h) * (2 * w)))
              (.bnBatchF s!"%b{pfx}eg" s!"%b{pfx}ebt" epsStr p.eε p.eγ p.eβ
                (.batchOp (N := N)
                  (.conv (h := 2 * h) (w := 2 * w) s!"%b{pfx}eW" s!"%zb{mid}" p.eW p.eb)
                  e)))))))

theorem mnv2StridedGraphB_faithful (pfx epsStr : String) (N h w : Nat) {ic mid oc : Nat}
    (p : IVW ic mid oc) (e : SHlo (N * (ic * (2 * h) * (2 * w)))) :
    den (mnv2StridedGraphB pfx epsStr N h w p e) = mnv2StridedB N h w p (den e) := by
  unfold mnv2StridedGraphB mnv2StridedB projB dwbrBstrided cbrB
  simp only [den_batchOp_relu6_eq_relu6F, relu6F_faithful, den_batchOp_conv,
    den_batchOp_depthwiseStridedXla, den_bnBatchF, Function.comp_apply]

/-- Head graph: 1x1 conv -> batch BN -> relu6 -> GAP -> dense. -/
def mnv2HeadGraphB (epsStr : String) (N h w : Nat) {ic oc nCls : Nat}
    (Wh : Kernel4 oc ic 1 1) (bh : Vec oc) (εh : ℝ) (γh βh : Vec oc)
    (Wd : Mat oc nCls) (bd : Vec nCls) (e : SHlo (N * (ic * h * w))) : SHlo (N * nCls) :=
  .batchOp (N := N) (.dense "%Wd" "%bd" Wd bd)
    (.batchOp (N := N) (.gap (c := oc) (h := h) (w := w))
      (.batchOp (N := N) (.relu6 (n := oc * h * w))
        (.bnBatchF "%hg" "%hbt" epsStr εh γh βh
          (.batchOp (N := N) (.conv (h := h) (w := w) "%hW" s!"%zb{oc}" Wh bh) e))))

theorem mnv2HeadGraphB_faithful (epsStr : String) (N h w : Nat) {ic oc nCls : Nat}
    (Wh : Kernel4 oc ic 1 1) (bh : Vec oc) (εh : ℝ) (γh βh : Vec oc)
    (Wd : Mat oc nCls) (bd : Vec nCls) (e : SHlo (N * (ic * h * w))) :
    den (mnv2HeadGraphB epsStr N h w Wh bh εh γh βh Wd bd e)
      = mnv2HeadB N h w Wh bh εh γh βh Wd bd (den e) := by
  unfold mnv2HeadGraphB mnv2HeadB cbrB
  simp only [den_batchOp_dense, den_batchOp_gap, den_batchOp_relu6_eq_relu6F, relu6F_faithful,
    den_batchOp_conv, den_bnBatchF, Function.comp_apply]

-- ════════════════════════════════════════════════════════════════
-- § The whole graph + faithfulness
-- ════════════════════════════════════════════════════════════════

/-- **The full batch-BN MobileNetV2 forward graph.** Block prefixes are the render's (`b1` … `b17`,
    each parameter `%b{k}{e,d,p}{W,g,bt}`), so the typed graph diffs against
    `mobilenetv2_adam_train_step`'s forward half name for name. -/
def mobilenetv2FwdGraphB_full (N : Nat) (epsStr : String) {nCls : Nat} (w : MNV2BWeights nCls)
    (e : SHlo (N * (3 * (2 * 112) * (2 * 112)))) : SHlo (N * nCls) :=
  mnv2HeadGraphB epsStr N 7 7 w.hW w.hb w.hε w.hγ w.hβ w.fcW w.fcb
    (mnv2ExpOnlyGraphB "17" epsStr N 7 7 w.b17
      (mnv2ResidGraphB "16" epsStr N 7 7 w.b16
        (mnv2ResidGraphB "15" epsStr N 7 7 w.b15
          (mnv2StridedGraphB "14" epsStr N 7 7 w.b14
            (mnv2ResidGraphB "13" epsStr N 14 14 w.b13
              (mnv2ResidGraphB "12" epsStr N 14 14 w.b12
                (mnv2ExpOnlyGraphB "11" epsStr N 14 14 w.b11
                  (mnv2ResidGraphB "10" epsStr N 14 14 w.b10
                    (mnv2ResidGraphB "9" epsStr N 14 14 w.b9
                      (mnv2ResidGraphB "8" epsStr N 14 14 w.b8
                        (mnv2StridedGraphB "7" epsStr N 14 14 w.b7
                          (mnv2ResidGraphB "6" epsStr N 28 28 w.b6
                            (mnv2ResidGraphB "5" epsStr N 28 28 w.b5
                              (mnv2StridedGraphB "4" epsStr N 28 28 w.b4
                                (mnv2ResidGraphB "3" epsStr N 56 56 w.b3
                                  (mnv2StridedGraphB "2" epsStr N 56 56 w.b2
                                    (mnv2NoExpGraphB "1" epsStr N 112 112 w.b1
                                      (mnv2StemGraphB epsStr N 112 112 w.sW w.sb w.sε w.sγ w.sβ
                                        e))))))))))))))))))

/-- ⭐ **T2 for MobileNetV2 at batch BN**: the typed graph denotes the whole-net forward. One `rw`
    per block over the six per-kind faithfulness lemmas. -/
theorem mobilenetv2FwdGraphB_full_faithful (N : Nat) (epsStr : String) {nCls : Nat}
    (w : MNV2BWeights nCls) (e : SHlo (N * (3 * (2 * 112) * (2 * 112)))) :
    den (mobilenetv2FwdGraphB_full N epsStr w e) = mobilenetv2ForwardB_full N w (den e) := by
  unfold mobilenetv2FwdGraphB_full mobilenetv2ForwardB_full
  rw [mnv2HeadGraphB_faithful, mnv2ExpOnlyGraphB_faithful, mnv2ResidGraphB_faithful,
      mnv2ResidGraphB_faithful, mnv2StridedGraphB_faithful, mnv2ResidGraphB_faithful,
      mnv2ResidGraphB_faithful, mnv2ExpOnlyGraphB_faithful, mnv2ResidGraphB_faithful,
      mnv2ResidGraphB_faithful, mnv2ResidGraphB_faithful, mnv2StridedGraphB_faithful,
      mnv2ResidGraphB_faithful, mnv2ResidGraphB_faithful, mnv2StridedGraphB_faithful,
      mnv2ResidGraphB_faithful, mnv2StridedGraphB_faithful, mnv2NoExpGraphB_faithful,
      mnv2StemGraphB_faithful]

end StableHLO

end Proofs

import LeanMlir.Proofs.Nets.ResNet.ResNet34BackB0
import LeanMlir.Proofs.Architectures.MaxPool3s2

/-! # ResNet-34 at TRUE BATCH-NORM — the whole net's forward and graph (T1-forward, T2)

`ResNet34RenderPC.lean` states ResNet-34's whole-net ℝ forward and typed graph at **per-example**
BatchNorm (`bnPerChannelTensor3`, reduce `[2,3]`). That is the world of `resnet34_fwd.mlir` and the
Imagenette SGD trainer, and every tier built on it is true and correctly paired with those bytes.
It is NOT the world of `resnet34_sgd_train_step.mlir`, `resnet34in_mom256_train_step.mlir` or any
of the Adam/momentum steps, which reduce `[0,2,3]` — one mu/var per channel across the batch, the
one op that couples examples. Those are the artifacts the quoted ImageNet accuracies come from.

This file re-states the ladder at `bnBatchLA` (= the proven `bnBatchTensor4` at the network's
left-assoc index). `formalization.yaml` 4e records the decision and
`planning/archive/proofs_tier_to_paper_nets.md` section 4 the work packages.

## What is new here, and what is not

⭐⭐ **Nothing about the blocks is new.** `ResNet34BackB0.lean` already carries the batched stages
(`cbReluB`, `cbReluStridedB`, `projStridedB`, and `projB` from `EfficientNetRenderPC.lean`), their
`_at` VJPs and their backward-graph faithfulness, all at `bnBatchLA`; `BackNetFolds.lean` folds
them to the paper depth as `r34Trunk_3463`. What was missing is the level above: a net-level ℝ
forward, a net-level forward graph, and the faithfulness tying them. This file is that enumeration.

⚠ **Padding is symmetric at every stride-2 site**, as ResNet-34's render emits and as the
PyTorch-origin convention requires (`.convStrided`, NOT `.convStridedXla` — B0's stem is the
XLA-`SAME` one and the two tokens have identical types). `scripts/convention_audit.py` checks this
at the artifact tier and nothing checks it here, so it is stated: stem 7x7/s2, the three
downsample `conv1`s and the three 1x1 projections are all `flatConvStride2`.

⚠ **The stem pool is 3x3/s2 (`maxPool3s2Flat`), not 2x2.** Same type, different function; the
render carried `.maxPool` until 2026-08-04 and nothing failed.

## Conventions this net runs at

| | |
|---|---|
| depth | [3,4,6,3] basic blocks, 64 to 512 |
| BatchNorm | **batch** (`bnBatchLA`, reduce `[0,2,3]`, width `N*h*w`) |
| activation | relu (one kink), TWO per block — the body's mid-relu and the post-residual one |
| stride-2 padding | symmetric at all seven sites |
| stem | 7x7/s2 conv-bn-relu, then 3x3/s2 max-pool |
| head | GAP then dense, generic in the class count |
| artifacts | `resnet34_sgd_train_step`, `resnet34_adam*`, `resnet34in_mom*` |

⭐ The head is generic in `nCls`, so one statement covers the 10-class Imagenette artifacts and the
1000-class `resnet34in` ones — the lesson `MobileNetV2FullPaperEval.lean` and B0's eval twin both
paid for.

⚠ `N` stays a variable throughout. T1 and T2 carry no numerals, so the batch size does not need
pinning here; it is pinned only where a `Maps` envelope turns a width into a rational (T4/T5), and
the artifacts' `N` is the PER-REPLICA batch (64 on the data-parallel runs) because the collectives
average gradients and no BatchNorm statistic is all-reduced.
-/

namespace Proofs

open scoped BigOperators

-- ════════════════════════════════════════════════════════════════
-- § Per-block weight bundles
-- ════════════════════════════════════════════════════════════════

/-- Weights of one identity basic block: two 3x3 convs at width `c`, BatchNorm after each. -/
structure R34IdW (c : Nat) where
  W₁ : Kernel4 c c 3 3
  b₁ : Vec c
  ε₁ : ℝ
  γ₁ : Vec c
  β₁ : Vec c
  W₂ : Kernel4 c c 3 3
  b₂ : Vec c
  ε₂ : ℝ
  γ₂ : Vec c
  β₂ : Vec c

/-- Weights of one downsample basic block: strided 3x3 (`ic -> oc`, halves spatial), 3x3, and the
    **1x1** stride-2 option-B projection on the skip. BatchNorm after each of the three. -/
structure R34DownW (ic oc : Nat) where
  W₁ : Kernel4 oc ic 3 3
  b₁ : Vec oc
  ε₁ : ℝ
  γ₁ : Vec oc
  β₁ : Vec oc
  W₂ : Kernel4 oc oc 3 3
  b₂ : Vec oc
  ε₂ : ℝ
  γ₂ : Vec oc
  β₂ : Vec oc
  Wp : Kernel4 oc ic 1 1
  bp : Vec oc
  εp : ℝ
  γp : Vec oc
  βp : Vec oc

/-- Every ResNet-34 parameter: stem (7x7/s2, 3->64) + the [3,4,6,3] basic blocks + dense head.
    Generic in the class count. -/
structure R34BWeights (nCls : Nat) where
  sW : Kernel4 64 3 7 7
  sb : Vec 64
  sε : ℝ
  sγ : Vec 64
  sβ : Vec 64
  a0 : R34IdW 64
  a1 : R34IdW 64
  a2 : R34IdW 64
  d2 : R34DownW 64 128
  b0 : R34IdW 128
  b1 : R34IdW 128
  b2 : R34IdW 128
  d3 : R34DownW 128 256
  c0 : R34IdW 256
  c1 : R34IdW 256
  c2 : R34IdW 256
  c3 : R34IdW 256
  c4 : R34IdW 256
  d4 : R34DownW 256 512
  e0 : R34IdW 512
  e1 : R34IdW 512
  Wd : Mat 512 nCls
  bd : Vec nCls

-- ════════════════════════════════════════════════════════════════
-- § Block real-forwards, batched
-- ════════════════════════════════════════════════════════════════

/-- Batched identity basic block `relu(F(x) + x)`, `F = projB . cbReluB`. The outer relu after the
    residual add is ResNet's structural difference from MobileNetV2/EfficientNet, whose residual
    add IS the block output — and it is why every r34 block carries TWO smoothness clauses. -/
@[reducible] noncomputable def r34IdB (N h w : Nat) {c : Nat} (p : R34IdW c) :
    Vec (N * (c * h * w)) → Vec (N * (c * h * w)) :=
  relu (N * (c * h * w)) ∘ residual
    (projB N (h := h) (w := w) p.W₂ p.b₂ p.ε₂ p.γ₂ p.β₂ ∘
      StableHLO.cbReluB N (h := h) (w := w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁)

/-- Batched downsample basic block `relu(F_s(x) + proj_s(x))`: body strided-conv1 then conv2,
    skip a 1x1 stride-2 projection. Both stride-2 sites are SYMMETRIC padding. -/
@[reducible] noncomputable def r34DownB (N h w : Nat) {ic oc : Nat} (p : R34DownW ic oc) :
    Vec (N * (ic * (2 * h) * (2 * w))) → Vec (N * (oc * h * w)) :=
  relu (N * (oc * h * w)) ∘ residualProj
    (StableHLO.projStridedB N (h := h) (w := w) p.Wp p.bp p.εp p.γp p.βp)
    (projB N (h := h) (w := w) p.W₂ p.b₂ p.ε₂ p.γ₂ p.β₂ ∘
      StableHLO.cbReluStridedB N (h := h) (w := w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁)

/-- Batched stem: 7x7/s2 conv -> bn -> relu, then He et al.'s 3x3/s2 max-pool. -/
@[reducible] noncomputable def r34StemB (N h w : Nat) {ic oc : Nat}
    (Ws : Kernel4 oc ic 7 7) (bs : Vec oc) (εs : ℝ) (γs βs : Vec oc) :
    Vec (N * (ic * (2 * (2 * h)) * (2 * (2 * w)))) → Vec (N * (oc * h * w)) :=
  StableHLO.batchMap N (maxPool3s2Flat oc h w) ∘
    StableHLO.cbReluStridedB N (h := 2 * h) (w := 2 * w) Ws bs εs γs βs

/-- Batched head: global average pool, then the dense classifier. -/
@[reducible] noncomputable def r34HeadB (N h w : Nat) {c nCls : Nat}
    (Wd : Mat c nCls) (bd : Vec nCls) : Vec (N * (c * h * w)) → Vec (N * nCls) :=
  StableHLO.batchMap N (dense Wd bd) ∘ StableHLO.batchMap N (globalAvgPoolFlat c h w)

/-- ResNet-34's GAP-and-dense tail, APPLIED. -/
theorem r34HeadB_apply (N h w : Nat) {c nCls : Nat} (Wd : Mat c nCls) (bd : Vec nCls)
    (v : Vec (N * (c * h * w))) :
    r34HeadB N h w Wd bd v
      = StableHLO.batchMap N (Proofs.dense Wd bd)
          (StableHLO.batchMap N (globalAvgPoolFlat c h w) v) := rfl

-- ════════════════════════════════════════════════════════════════
-- § The whole net, nested-application form
--   stem(224 -> 112 -> 56) -> a0,a1,a2@56 -> d2(56->28) -> b0,b1,b2@28
--   -> d3(28->14) -> c0..c4@14 -> d4(14->7) -> e0,e1@7 -> GAP -> dense
-- ════════════════════════════════════════════════════════════════

/-- **The full batch-BN ResNet-34 forward**, `N*(3*224*224) -> N*nCls`. The batched peer of
    `resnet34Forward_full_pc`; nested-application form, as `efficientnetForwardB_full` and
    `mobilenetv2ForwardPaper` both are, so the T6 tie can peel it one block at a time. -/
noncomputable def resnet34ForwardB_full (N : Nat) {nCls : Nat} (w : R34BWeights nCls)
    (x : Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56))))) : Vec (N * nCls) :=
  r34HeadB N 7 7 w.Wd w.bd
    (r34IdB N 7 7 w.e1
      (r34IdB N 7 7 w.e0
        (r34DownB N 7 7 w.d4
          (r34IdB N 14 14 w.c4
            (r34IdB N 14 14 w.c3
              (r34IdB N 14 14 w.c2
                (r34IdB N 14 14 w.c1
                  (r34IdB N 14 14 w.c0
                    (r34DownB N 14 14 w.d3
                      (r34IdB N 28 28 w.b2
                        (r34IdB N 28 28 w.b1
                          (r34IdB N 28 28 w.b0
                            (r34DownB N 28 28 w.d2
                              (r34IdB N 56 56 w.a2
                                (r34IdB N 56 56 w.a1
                                  (r34IdB N 56 56 w.a0
                                    (r34StemB N 56 56 w.sW w.sb w.sε w.sγ w.sβ x)))))))))))))))))

namespace StableHLO

-- ════════════════════════════════════════════════════════════════
-- § Per-block typed graphs + faithfulness
--   Tokens are the ones `ResNet34RenderB.lean` emits: `.batchOp .convStrided` / `.conv` /
--   `.relu` / `.maxPool3s2` / `.gap` / `.dense`, and `.bnBatchF` for the batch-coupled norm.
-- ════════════════════════════════════════════════════════════════

/-- Identity basic-block graph: `relu(addV(bn(conv(relu(bn(conv e)))), e))`. The skip reuses the
    block-input subtree `e` in both `addV` operands, as the render does. -/
def r34IdGraphB (p epsStr : String) (N h w : Nat) {c : Nat} (pw : R34IdW c)
    (e : SHlo (N * (c * h * w))) : SHlo (N * (c * h * w)) :=
  .batchOp (N := N) (.relu (n := c * h * w))
    (.addVB
      (.bnBatchF s!"%{p}g2" s!"%{p}bt2" epsStr pw.ε₂ pw.γ₂ pw.β₂
        (.batchOp (N := N) (.conv (h := h) (w := w) s!"%{p}W2" (biasName false "" c) pw.W₂ pw.b₂)
          (.batchOp (N := N) (.relu (n := c * h * w))
            (.bnBatchF s!"%{p}g1" s!"%{p}bt1" epsStr pw.ε₁ pw.γ₁ pw.β₁
              (.batchOp (N := N) (.conv (h := h) (w := w) s!"%{p}W1" (biasName false "" c) pw.W₁ pw.b₁) e)))))
      e)

theorem r34IdGraphB_faithful (p epsStr : String) (N h w : Nat) {c : Nat} (pw : R34IdW c)
    (e : SHlo (N * (c * h * w))) :
    den (r34IdGraphB p epsStr N h w pw e) = r34IdB N h w pw (den e) := by
  unfold r34IdGraphB r34IdB projB cbReluB residual biPath
  simp only [den_batchOp_relu_eq_reluF, reluF_faithful, den_batchOp_conv, den_bnBatchF, den_addVB,
    Function.comp_apply]

/-- Downsample basic-block graph: `relu(addVB(projection, body))` — projection first, matching
    `residualProj proj body`. Both branches read the block-input subtree `e`; both stride-2 convs
    are `.convStrided` (symmetric padding). -/
def r34DownGraphB (p epsStr : String) (N h w : Nat) {ic oc : Nat} (pw : R34DownW ic oc)
    (e : SHlo (N * (ic * (2 * h) * (2 * w)))) : SHlo (N * (oc * h * w)) :=
  .batchOp (N := N) (.relu (n := oc * h * w))
    (.addVB
      (.bnBatchF s!"%{p}gp" s!"%{p}btp" epsStr pw.εp pw.γp pw.βp
        (.batchOp (N := N) (.convStrided (h := h) (w := w) s!"%{p}Wp" (biasName false "" oc) pw.Wp pw.bp) e))
      (.bnBatchF s!"%{p}g2" s!"%{p}bt2" epsStr pw.ε₂ pw.γ₂ pw.β₂
        (.batchOp (N := N) (.conv (h := h) (w := w) s!"%{p}W2" (biasName false "" oc) pw.W₂ pw.b₂)
          (.batchOp (N := N) (.relu (n := oc * h * w))
            (.bnBatchF s!"%{p}g1" s!"%{p}bt1" epsStr pw.ε₁ pw.γ₁ pw.β₁
              (.batchOp (N := N)
                (.convStrided (h := h) (w := w) s!"%{p}W1" (biasName false "" oc) pw.W₁ pw.b₁) e))))))

theorem r34DownGraphB_faithful (p epsStr : String) (N h w : Nat) {ic oc : Nat}
    (pw : R34DownW ic oc) (e : SHlo (N * (ic * (2 * h) * (2 * w)))) :
    den (r34DownGraphB p epsStr N h w pw e) = r34DownB N h w pw (den e) := by
  unfold r34DownGraphB r34DownB projB projStridedB cbReluStridedB residualProj biPath
  simp only [den_batchOp_relu_eq_reluF, reluF_faithful, den_batchOp_conv, den_batchOp_convStrided,
    den_bnBatchF, den_addVB, Function.comp_apply]

/-- Stem graph: 7x7/s2 conv -> bn -> relu -> 3x3/s2 max-pool. -/
def r34StemGraphB (epsStr : String) (N h w : Nat) {ic oc : Nat}
    (Ws : Kernel4 oc ic 7 7) (bs : Vec oc) (εs : ℝ) (γs βs : Vec oc)
    (e : SHlo (N * (ic * (2 * (2 * h)) * (2 * (2 * w))))) : SHlo (N * (oc * h * w)) :=
  .batchOp (N := N) (.maxPool3s2 (c := oc) (h := h) (w := w))
    (.batchOp (N := N) (.relu (n := oc * (2 * h) * (2 * w)))
      (.bnBatchF "%sg" "%sbt" epsStr εs γs βs
        (.batchOp (N := N) (.convStrided (h := 2 * h) (w := 2 * w) "%sW" (biasName false "" oc) Ws bs) e)))

theorem r34StemGraphB_faithful (epsStr : String) (N h w : Nat) {ic oc : Nat}
    (Ws : Kernel4 oc ic 7 7) (bs : Vec oc) (εs : ℝ) (γs βs : Vec oc)
    (e : SHlo (N * (ic * (2 * (2 * h)) * (2 * (2 * w))))) :
    den (r34StemGraphB epsStr N h w Ws bs εs γs βs e)
      = r34StemB N h w Ws bs εs γs βs (den e) := by
  unfold r34StemGraphB r34StemB cbReluStridedB
  simp only [den_batchOp_maxPool3s2, den_batchOp_relu_eq_reluF, reluF_faithful,
    den_batchOp_convStrided, den_bnBatchF, Function.comp_apply]

/-- Head graph: GAP then dense. -/
def r34HeadGraphB (N h w : Nat) {c nCls : Nat} (Wd : Mat c nCls) (bd : Vec nCls)
    (e : SHlo (N * (c * h * w))) : SHlo (N * nCls) :=
  .batchOp (N := N) (.dense "%Wd" "%bd" Wd bd)
    (.batchOp (N := N) (.gap (c := c) (h := h) (w := w)) e)

theorem r34HeadGraphB_faithful (N h w : Nat) {c nCls : Nat} (Wd : Mat c nCls) (bd : Vec nCls)
    (e : SHlo (N * (c * h * w))) :
    den (r34HeadGraphB N h w Wd bd e) = r34HeadB N h w Wd bd (den e) := by
  unfold r34HeadGraphB r34HeadB
  simp only [den_batchOp_dense, den_batchOp_gap, Function.comp_apply]

-- ════════════════════════════════════════════════════════════════
-- § The whole graph + faithfulness
-- ════════════════════════════════════════════════════════════════

/-- **The full batch-BN ResNet-34 forward graph.** Block prefixes are the render's
    (`s1b0`/`d2`/`s2b0`/... ), so the typed graph diffs against `resnet34_fwd`'s batched peers
    name for name. -/
def resnet34FwdGraphB_full (N : Nat) (epsStr : String) {nCls : Nat} (w : R34BWeights nCls)
    (e : SHlo (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56))))) : SHlo (N * nCls) :=
  r34HeadGraphB N 7 7 w.Wd w.bd
    (r34IdGraphB "s4b1" epsStr N 7 7 w.e1
      (r34IdGraphB "s4b0" epsStr N 7 7 w.e0
        (r34DownGraphB "d4" epsStr N 7 7 w.d4
          (r34IdGraphB "s3b4" epsStr N 14 14 w.c4
            (r34IdGraphB "s3b3" epsStr N 14 14 w.c3
              (r34IdGraphB "s3b2" epsStr N 14 14 w.c2
                (r34IdGraphB "s3b1" epsStr N 14 14 w.c1
                  (r34IdGraphB "s3b0" epsStr N 14 14 w.c0
                    (r34DownGraphB "d3" epsStr N 14 14 w.d3
                      (r34IdGraphB "s2b2" epsStr N 28 28 w.b2
                        (r34IdGraphB "s2b1" epsStr N 28 28 w.b1
                          (r34IdGraphB "s2b0" epsStr N 28 28 w.b0
                            (r34DownGraphB "d2" epsStr N 28 28 w.d2
                              (r34IdGraphB "s1b2" epsStr N 56 56 w.a2
                                (r34IdGraphB "s1b1" epsStr N 56 56 w.a1
                                  (r34IdGraphB "s1b0" epsStr N 56 56 w.a0
                                    (r34StemGraphB epsStr N 56 56 w.sW w.sb w.sε w.sγ w.sβ
                                      e)))))))))))))))))

/-- ⭐ **T2 for ResNet-34 at batch BN**: the typed graph denotes the whole-net forward. One `rw`
    per block over the eighteen per-kind faithfulness lemmas. -/
theorem resnet34FwdGraphB_full_faithful (N : Nat) (epsStr : String) {nCls : Nat}
    (w : R34BWeights nCls) (e : SHlo (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56))))) :
    den (resnet34FwdGraphB_full N epsStr w e) = resnet34ForwardB_full N w (den e) := by
  unfold resnet34FwdGraphB_full resnet34ForwardB_full
  rw [r34HeadGraphB_faithful, r34IdGraphB_faithful, r34IdGraphB_faithful, r34DownGraphB_faithful,
      r34IdGraphB_faithful, r34IdGraphB_faithful, r34IdGraphB_faithful, r34IdGraphB_faithful,
      r34IdGraphB_faithful, r34DownGraphB_faithful, r34IdGraphB_faithful, r34IdGraphB_faithful,
      r34IdGraphB_faithful, r34DownGraphB_faithful, r34IdGraphB_faithful, r34IdGraphB_faithful,
      r34IdGraphB_faithful, r34StemGraphB_faithful]

end StableHLO

end Proofs

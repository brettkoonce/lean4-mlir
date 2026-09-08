import LeanMlir.Proofs.Architectures.ResNet34FullB
import LeanMlir.Proofs.Foundation.ResNet50BackB0

/-! # ResNet-50 at TRUE BATCH-NORM — the whole net's forward and graph (T1-forward, T2)

ResNet-50 is the largest hole in the Proofs tier: `planning/archive/proofs_tier_to_paper_nets.md` §2's
audit row reads "none; `r50Trunk_3463` is a backward fold" with every tier ✗. §3.5(a) is this file
— a net-level ℝ forward at the [3,4,6,3] bottleneck ladder, in the world the artifacts run —
and §3.5(b) is the typed graph over it, in the second half of this file.

⭐ **This is the one net where T1 matches the trained world from the start.** ResNet-34's and
MobileNetV2's Proofs tiers were written at per-example BatchNorm and had to be ported (§4);
ResNet-50 has only ever had a batched renderer (`ResNet50RenderB.lean`), so `bnBatchLA` is the
world of `resnet50_fwd`, `resnet50in160_lambaccdp8x64bce` and everything between. There is no
BatchNorm-world split to port later, and none of 4b's or 4c's axes apply to this net.

## What is new here, and what is not

⭐⭐ **Nothing about the blocks is new.** `ResNet50BackB0.lean` already carries all three batched
bottleneck forms at `bnBatchLA` with their `_at` VJPs and backward-graph faithfulness, and
`BackNetFolds.lean` folds them to the paper depth as `r50Trunk_3463`. What was missing is the level
above. This file is that enumeration, exactly as `ResNet34FullB.lean` was for r34.

⭐ **The stem and the head are ResNet-34's, imported rather than re-declared.** `r34StemB` is
generic in `{ic oc}` and `r34HeadB` in `{c nCls}`, and R50's stem (7×7/s2 conv-bn-relu, then He et
al.'s 3×3/s2 max-pool, 3 → 64) and head (GAP then dense) are the same functions at different
widths. So their VJP lemmas are reused verbatim one tier up, the way `ResNet50BackB0.lean` reuses
all four of `ResNet34BackB0.lean`'s stages. A second `r50StemB` would be two writers for one fact.

⭐ **ONE weight record serves both projection forms.** The stride-1 projection block (stage 1
block 0) and the strided one (stages 2/3/4 block 0) have identical parameter shapes — 1×1, 3×3,
1×1 and a 1×1 skip — and differ only in which convolutions are strided. `R50ProjW` is that record;
`r50ProjB` and `r50DownB` are the two forwards over it. ResNet-34 needed two records for its two
block kinds.

## Conventions this net runs at

| | |
|---|---|
| depth | [3,4,6,3] BOTTLENECKS, 64/256 → 128/512 → 256/1024 → 512/2048 |
| BatchNorm | **batch** (`bnBatchLA`, reduce `[0,2,3]`, width `N*h*w`) |
| activation | relu, **THREE kinks per block** — two interior and the post-residual one |
| stride-2 padding | symmetric at all five sites (stem + three downsample 3×3 + three 1×1 skips) |
| stride placement | **v1.5**: the stride is on the 3×3, not the leading 1×1 |
| stem | 7×7/s2 conv-bn-relu, then 3×3/s2 max-pool |
| head | GAP then dense, generic in the class count |
| artifacts | `resnet50_fwd`, `resnet50in_fwd`, `resnet50in160_fwd`, every `resnet50in*_train_step` |

⚠⚠ **THE STRIDE IS ON THE 3×3.** `r50DownB` puts `cbReluStridedB` on the SECOND convolution, so
the leading 1×1 runs at the INPUT resolution and carries `mid` channels there until `W₂` decimates.
That is ResNet **v1.5** / torchvision, which is what `jax/MainResnet50Imagenet.lean` trains. The v1
placement compiles, trains and descends, and is a different net worth ~0.5 pt of top-1
(`VerifiedSpec.lean:46`). Nothing in the types sees the difference.

⚠ **Stage 1 block 0 is a STRIDE-1 projection, and it is the block with no ResNet-34 analogue.**
Channels go 64 → 256 at unchanged resolution, so it needs a projection but not a strided one.
⛔ `r50DownB` cannot be substituted — the halving is in its signature, so that is a shape error and
would be caught. Reaching for the identity form `r50IdB` is the dangerous one: it is well-typed
only when `ic = oc`, which is exactly why this block exists.

⭐⭐ **The spatial size is a BINDER, `q`, and that is not a stylistic choice.** ResNet-50 ships at
TWO resolutions — `resnet50in_fwd` at 224 and `resnet50in160_fwd` at 160, the second being where
`resnet50in160_lambaccdp8x64bce`'s 76.66% comes from. The ladder is `q`, `2q`, `4q`, `8q` with the
input at `32q`, so `q = 7` is the 224 net and `q = 5` the 160 one and ONE statement covers both.
ResNet-34 could pin 56 because it has a single shipped resolution. ⚠ Every resolution is written as
an explicit nest of `2 * (…)` rather than a product like `8 * q`: `2 * (4 * q)` and `8 * q` are
equal Nats and NOT definitionally equal terms at a variable `q`, and the block signatures demand
the operand at exactly the spelling they name. The render's own `q1 … q5` comment records the same
trap on the emitter side.

⚠ `N` stays a variable throughout, as at r34: T1 carries no numerals, and the artifacts' `N` is the
PER-REPLICA batch (`DataParallel.lean`, §4d) because the collectives average gradients and no
BatchNorm statistic is all-reduced.

⭐ **The census is 161 updated parameters**, which is `ResNet50RenderB`'s own docstring ("161 θ /
161 m / 161 v"): stem 3 (`sW`, `sγ`, `sβ`) + 12 identity bottlenecks × 9 + 4 projection
bottlenecks × 12 + head 2. ⚠ The records ALSO carry a bias slot per convolution, as ResNet-34's and
MobileNetV2's do: both R50 renders run `convBias := false`, each conv bias is folded into the
BatchNorm after it and bound to a `zeroBiasPrelude` zero, so those fields are the `convBias := true`
census and are `∀`-quantified over — `bias = 0` is one instance. The 106 running-statistic slots the
render's signature also carries belong to inference and do not appear here, since training-mode
BatchNorm computes its statistics from the batch.

✅ **The typed forward graph (T2) is the second half of this file** — `resnet50FwdGraphB_full` and
its `_faithful`, over four per-block-kind graphs at `r50FwdChainB`'s own tokens. ✅ Checked against
the committed bytes: `verified_mlir/resnet50_fwd.mlir`'s signature is **162 arguments = `%x` + 161
parameters**, with 12 projection slots, and every name this file writes (`%sW`, `%sg`, `%sbt`,
`%zb64` … `%zb2048`, `%s1b0W1` … `%s4b2bt3`, `%s1b0Wp`/`%gp`/`%btp`, `%Wd`, `%bd`) appears there.
-/

namespace Proofs

open scoped BigOperators

-- ════════════════════════════════════════════════════════════════
-- § Per-block weight bundles
-- ════════════════════════════════════════════════════════════════

/-- Weights of one identity bottleneck: 1×1 (`oc → mid`), 3×3 (`mid → mid`), 1×1 (`mid → oc`),
    BatchNorm after each. The third convolution has NO activation — the outer relu comes after
    the residual add. -/
structure R50IdW (mid oc : Nat) where
  W₁ : Kernel4 mid oc 1 1
  b₁ : Vec mid
  ε₁ : ℝ
  γ₁ : Vec mid
  β₁ : Vec mid
  W₂ : Kernel4 mid mid 3 3
  b₂ : Vec mid
  ε₂ : ℝ
  γ₂ : Vec mid
  β₂ : Vec mid
  W₃ : Kernel4 oc mid 1 1
  b₃ : Vec oc
  ε₃ : ℝ
  γ₃ : Vec oc
  β₃ : Vec oc

/-- Weights of a PROJECTION bottleneck — `ic → mid → mid → oc` plus the 1×1 option-B skip.

    ⭐ ONE record for BOTH projection forms. Stage 1 block 0 (stride 1, `64 → 256`) and stages
    2/3/4 block 0 (strided) have identical parameter shapes and differ only in which convolutions
    are strided, which is a property of the forward and not of the weights. `r50ProjB` and
    `r50DownB` are those two forwards. -/
structure R50ProjW (ic mid oc : Nat) where
  W₁ : Kernel4 mid ic 1 1
  b₁ : Vec mid
  ε₁ : ℝ
  γ₁ : Vec mid
  β₁ : Vec mid
  W₂ : Kernel4 mid mid 3 3
  b₂ : Vec mid
  ε₂ : ℝ
  γ₂ : Vec mid
  β₂ : Vec mid
  W₃ : Kernel4 oc mid 1 1
  b₃ : Vec oc
  ε₃ : ℝ
  γ₃ : Vec oc
  β₃ : Vec oc
  Wp : Kernel4 oc ic 1 1
  bp : Vec oc
  εp : ℝ
  γp : Vec oc
  βp : Vec oc

/-- Every ResNet-50 parameter: stem (7×7/s2, 3 → 64) + the [3,4,6,3] bottleneck ladder + the dense
    head. Generic in the class count, so one statement covers the 10-class Imagenette artifacts and
    the 1000-class `resnet50in` ones.

    ⭐ The field names are `ResNet50RenderB`'s own SSA prefixes (`s1b0` … `s4b2`), so a reader can
    match a parameter to its emitted name without a table. -/
structure R50BWeights (nCls : Nat) where
  sW : Kernel4 64 3 7 7
  sb : Vec 64
  sε : ℝ
  sγ : Vec 64
  sβ : Vec 64
  s1b0 : R50ProjW 64 64 256
  s1b1 : R50IdW 64 256
  s1b2 : R50IdW 64 256
  s2b0 : R50ProjW 256 128 512
  s2b1 : R50IdW 128 512
  s2b2 : R50IdW 128 512
  s2b3 : R50IdW 128 512
  s3b0 : R50ProjW 512 256 1024
  s3b1 : R50IdW 256 1024
  s3b2 : R50IdW 256 1024
  s3b3 : R50IdW 256 1024
  s3b4 : R50IdW 256 1024
  s3b5 : R50IdW 256 1024
  s4b0 : R50ProjW 1024 512 2048
  s4b1 : R50IdW 512 2048
  s4b2 : R50IdW 512 2048
  Wd : Mat 2048 nCls
  bd : Vec nCls

-- ════════════════════════════════════════════════════════════════
-- § Block real-forwards, batched
-- ════════════════════════════════════════════════════════════════

/-- Batched identity bottleneck `relu(F(x) + x)`, `F = projB ∘ cbReluB ∘ cbReluB`: 1×1-reduce-relu,
    3×3-relu, 1×1-expand with no activation. THREE relu kinks — the two interior ones and the
    post-residual outer one — where ResNet-34's basic block has two. -/
@[reducible] noncomputable def r50IdB (N h w : Nat) {mid oc : Nat} (p : R50IdW mid oc) :
    Vec (N * (oc * h * w)) → Vec (N * (oc * h * w)) :=
  relu (N * (oc * h * w)) ∘ residual
    (projB N (h := h) (w := w) p.W₃ p.b₃ p.ε₃ p.γ₃ p.β₃ ∘
      StableHLO.cbReluB N (h := h) (w := w) p.W₂ p.b₂ p.ε₂ p.γ₂ p.β₂ ∘
      StableHLO.cbReluB N (h := h) (w := w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁)

/-- ⭐ Batched **stride-1 projection** bottleneck `relu(F(x) + proj(x))` — stage 1 block 0 and
    nowhere else in the net. The channels change (`64 → 256`) so a projection is needed; the
    resolution does not, so that projection is a plain 1×1 conv-BN. The form with no ResNet-34
    analogue: R34's stage 1 runs at `ic = oc = 64`, where block 0 is an identity block. -/
@[reducible] noncomputable def r50ProjB (N h w : Nat) {ic mid oc : Nat} (p : R50ProjW ic mid oc) :
    Vec (N * (ic * h * w)) → Vec (N * (oc * h * w)) :=
  relu (N * (oc * h * w)) ∘ residualProj
    (projB N (h := h) (w := w) p.Wp p.bp p.εp p.γp p.βp)
    (projB N (h := h) (w := w) p.W₃ p.b₃ p.ε₃ p.γ₃ p.β₃ ∘
      StableHLO.cbReluB N (h := h) (w := w) p.W₂ p.b₂ p.ε₂ p.γ₂ p.β₂ ∘
      StableHLO.cbReluB N (h := h) (w := w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁)

/-- Batched **strided projection** bottleneck — stages 2/3/4 block 0, halving the grid.

    ⚠⚠ v1.5: the stride is on the 3×3 (`cbReluStridedB` is the SECOND stage), so the leading 1×1
    and its BN and relu run at the INPUT resolution `2h × 2w`. Both stride-2 sites — the 3×3 and
    the 1×1 skip — are SYMMETRIC padding, as the PyTorch-origin convention requires. -/
@[reducible] noncomputable def r50DownB (N h w : Nat) {ic mid oc : Nat} (p : R50ProjW ic mid oc) :
    Vec (N * (ic * (2 * h) * (2 * w))) → Vec (N * (oc * h * w)) :=
  relu (N * (oc * h * w)) ∘ residualProj
    (StableHLO.projStridedB N (h := h) (w := w) p.Wp p.bp p.εp p.γp p.βp)
    (projB N (h := h) (w := w) p.W₃ p.b₃ p.ε₃ p.γ₃ p.β₃ ∘
      StableHLO.cbReluStridedB N (h := h) (w := w) p.W₂ p.b₂ p.ε₂ p.γ₂ p.β₂ ∘
      StableHLO.cbReluB N (h := 2 * h) (w := 2 * w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁)

-- ════════════════════════════════════════════════════════════════
-- § The whole net, nested-application form
--   stem(32q -> 16q -> 8q) -> s1b0,s1b1,s1b2 @8q -> s2b0(8q->4q), s2b1..3 @4q
--   -> s3b0(4q->2q), s3b1..5 @2q -> s4b0(2q->q), s4b1,s4b2 @q -> GAP -> dense
--   q = 7 is the 224-px net, q = 5 the 160-px one.
-- ════════════════════════════════════════════════════════════════

/-- **The full batch-BN ResNet-50 forward**, `N*(3*32q*32q) -> N*nCls`, at the [3,4,6,3] bottleneck
    ladder. Nested-application form, as `resnet34ForwardB_full` and `efficientnetForwardB_full`
    both are, so a later tie can peel it one block at a time.

    ⭐ `q` is a binder: `q = 7` is `resnet50in_fwd` and `q = 5` is `resnet50in160_fwd`, the net the
    quoted 76.66% trains. ⭐ The stem and head are ResNet-34's functions at R50's widths. -/
noncomputable def resnet50ForwardB_full (N q : Nat) {nCls : Nat} (w : R50BWeights nCls)
    (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q)))))))) :
    Vec (N * nCls) :=
  r34HeadB N q q w.Wd w.bd
    (r50IdB N q q w.s4b2
      (r50IdB N q q w.s4b1
        (r50DownB N q q w.s4b0
          (r50IdB N (2 * q) (2 * q) w.s3b5
            (r50IdB N (2 * q) (2 * q) w.s3b4
              (r50IdB N (2 * q) (2 * q) w.s3b3
                (r50IdB N (2 * q) (2 * q) w.s3b2
                  (r50IdB N (2 * q) (2 * q) w.s3b1
                    (r50DownB N (2 * q) (2 * q) w.s3b0
                      (r50IdB N (2 * (2 * q)) (2 * (2 * q)) w.s2b3
                        (r50IdB N (2 * (2 * q)) (2 * (2 * q)) w.s2b2
                          (r50IdB N (2 * (2 * q)) (2 * (2 * q)) w.s2b1
                            (r50DownB N (2 * (2 * q)) (2 * (2 * q)) w.s2b0
                              (r50IdB N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) w.s1b2
                                (r50IdB N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) w.s1b1
                                  (r50ProjB N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) w.s1b0
                                    (r34StemB N (2 * (2 * (2 * q))) (2 * (2 * (2 * q)))
                                      w.sW w.sb w.sε w.sγ w.sβ x)))))))))))))))))

-- ════════════════════════════════════════════════════════════════
-- § The ladder's arithmetic, and the two shipped resolutions, machine-checked
-- ════════════════════════════════════════════════════════════════

-- The stage resolutions at `q = 7` (the 224-px net) and `q = 5` (the 160-px one). These are
-- `#guard`s rather than prose because the `2 * (...)` nests are the one thing a reader cannot
-- check by eye, and a wrong nest depth is well-typed at a variable `q`.
#guard 2 * (2 * (2 * (2 * (2 * 7)))) == 224
#guard 2 * (2 * (2 * (2 * (2 * 5)))) == 160
#guard 2 * (2 * (2 * 7)) == 56
#guard 2 * (2 * (2 * 5)) == 40
#guard 2 * (2 * 7) == 28
#guard 2 * (2 * 5) == 20

-- ⭐ `q = 7` IS the 224-px net: the input binds at the literal `Vec (N*(3*224*224))`. The whole
-- point of the `q` binder is that both shipped resolutions are instances, so it is CHECKED here
-- rather than asserted in the header.
example (N : Nat) {nCls : Nat} (w : R50BWeights nCls) (x : Vec (N * (3 * 224 * 224))) :
    resnet50ForwardB_full N 7 w x = resnet50ForwardB_full N 7 w x := rfl

-- ⭐ And `q = 5` IS the 160-px net -- `resnet50in160_*`, where the quoted 76.66% comes from.
example (N : Nat) {nCls : Nat} (w : R50BWeights nCls) (x : Vec (N * (3 * 160 * 160))) :
    resnet50ForwardB_full N 5 w x = resnet50ForwardB_full N 5 w x := rfl


namespace StableHLO

-- ════════════════════════════════════════════════════════════════
-- § Per-block typed graphs + faithfulness (T2)
--   Tokens are the ones `ResNet50RenderB.lean` emits: `.batchOp .conv` / `.convStrided` /
--   `.relu` / `.maxPool3s2` / `.gap` / `.dense`, `.bnBatchF` for the batch-coupled norm, and
--   `.addVB` for the residual add.
--
--   ⚠ **`.addVB`, not `.addV`.** `ResNet50RenderB` emits the batched add; `den` is identical
--   (both are `fun j => den a j + den b j`, both by `rfl`) but `skel` is not, so the emitted shape
--   annotation differs. `ResNet34FullB.lean` uses `.addV` where its own render emits `.addVB` —
--   recorded in §4.2b and left alone there; this file does not repeat it.
--
--   ⚠ **The bias operands are `biasName false "" c`, the render's own function.** `ResNet50RenderB`
--   has no `convBias` flag at all — its `zb` bakes `false` — so `%zb{c}`, the shared zero constant
--   each conv bias is folded into its BatchNorm and bound to, is the ONLY name this net emits.
--   Calling the shared function rather than writing the literal is what keeps the two from
--   drifting. ⛔ `ResNet34FullB.lean` writes `"%sb"` / `"%{p}b1"`, which are the `convBias := true`
--   names its render does NOT emit by default — the graph-operand form of the census trap, and a
--   cosmetic gap on that file worth fixing when it is next touched.
-- ════════════════════════════════════════════════════════════════

/-- Identity bottleneck graph: `relu(addVB(bn3(conv3(relu(bn2(conv2(relu(bn1(conv1 e))))))), e))`.
    The skip reuses the block-input subtree `e` in both `addVB` operands, as the render does, and
    the operand ORDER is the render's (body first) — which is also `residual`'s. -/
def r50IdGraphB (p epsStr : String) (N h w : Nat) {mid oc : Nat} (pw : R50IdW mid oc)
    (e : SHlo (N * (oc * h * w))) : SHlo (N * (oc * h * w)) :=
  .batchOp (N := N) (.relu (n := oc * h * w))
    (.addVB
      (.bnBatchF s!"%{p}g3" s!"%{p}bt3" epsStr pw.ε₃ pw.γ₃ pw.β₃
        (.batchOp (N := N)
          (.conv (h := h) (w := w) s!"%{p}W3" (biasName false "" oc) pw.W₃ pw.b₃)
          (.batchOp (N := N) (.relu (n := mid * h * w))
            (.bnBatchF s!"%{p}g2" s!"%{p}bt2" epsStr pw.ε₂ pw.γ₂ pw.β₂
              (.batchOp (N := N)
                (.conv (h := h) (w := w) s!"%{p}W2" (biasName false "" mid) pw.W₂ pw.b₂)
                (.batchOp (N := N) (.relu (n := mid * h * w))
                  (.bnBatchF s!"%{p}g1" s!"%{p}bt1" epsStr pw.ε₁ pw.γ₁ pw.β₁
                    (.batchOp (N := N)
                      (.conv (h := h) (w := w) s!"%{p}W1" (biasName false "" mid) pw.W₁ pw.b₁)
                      e))))))))
      e)

theorem r50IdGraphB_faithful (p epsStr : String) (N h w : Nat) {mid oc : Nat} (pw : R50IdW mid oc)
    (e : SHlo (N * (oc * h * w))) :
    den (r50IdGraphB p epsStr N h w pw e) = r50IdB N h w pw (den e) := by
  unfold r50IdGraphB r50IdB projB cbReluB residual biPath
  simp only [den_batchOp_relu_eq_reluF, reluF_faithful, den_batchOp_conv, den_bnBatchF,
    den_addVB, Function.comp_apply]

/-- ⭐ Stride-1 projection bottleneck graph — stage 1 block 0. The skip is a plain `1×1` conv → BN
    (`.conv`, NOT `.convStrided`), which is the whole point of this form. Both `addVB` operands are
    nontrivial subtrees and both read the block-input subtree `e`.

    ⚠ The render emits `addVB(body, projection)` where `residualProj proj body` adds
    `proj + body` — so this graph is in the RENDER's order and the faithfulness proof carries one
    `add_comm`. The alternative, writing the graph in `residualProj`'s order, would make `den`
    close by `rfl` and the emitted operand order wrong. -/
def r50ProjGraphB (p epsStr : String) (N h w : Nat) {ic mid oc : Nat} (pw : R50ProjW ic mid oc)
    (e : SHlo (N * (ic * h * w))) : SHlo (N * (oc * h * w)) :=
  .batchOp (N := N) (.relu (n := oc * h * w))
    (.addVB
      (.bnBatchF s!"%{p}g3" s!"%{p}bt3" epsStr pw.ε₃ pw.γ₃ pw.β₃
        (.batchOp (N := N)
          (.conv (h := h) (w := w) s!"%{p}W3" (biasName false "" oc) pw.W₃ pw.b₃)
          (.batchOp (N := N) (.relu (n := mid * h * w))
            (.bnBatchF s!"%{p}g2" s!"%{p}bt2" epsStr pw.ε₂ pw.γ₂ pw.β₂
              (.batchOp (N := N)
                (.conv (h := h) (w := w) s!"%{p}W2" (biasName false "" mid) pw.W₂ pw.b₂)
                (.batchOp (N := N) (.relu (n := mid * h * w))
                  (.bnBatchF s!"%{p}g1" s!"%{p}bt1" epsStr pw.ε₁ pw.γ₁ pw.β₁
                    (.batchOp (N := N)
                      (.conv (h := h) (w := w) s!"%{p}W1" (biasName false "" mid) pw.W₁ pw.b₁)
                      e))))))))
      (.bnBatchF s!"%{p}gp" s!"%{p}btp" epsStr pw.εp pw.γp pw.βp
        (.batchOp (N := N)
          (.conv (h := h) (w := w) s!"%{p}Wp" (biasName false "" oc) pw.Wp pw.bp) e)))

theorem r50ProjGraphB_faithful (p epsStr : String) (N h w : Nat) {ic mid oc : Nat}
    (pw : R50ProjW ic mid oc) (e : SHlo (N * (ic * h * w))) :
    den (r50ProjGraphB p epsStr N h w pw e) = r50ProjB N h w pw (den e) := by
  unfold r50ProjGraphB r50ProjB projB cbReluB residualProj biPath
  simp only [den_batchOp_relu_eq_reluF, reluF_faithful, den_batchOp_conv, den_bnBatchF,
    den_addVB, Function.comp_apply]
  congr 1
  funext i
  ring

/-- Strided projection bottleneck graph — stages 2/3/4 block 0. ⚠⚠ v1.5: `.convStrided` appears at
    the **3×3** and at the 1×1 skip, and `conv1`/`bn1`/`relu1` run at the input resolution
    `2h × 2w`. Both stride-2 sites are SYMMETRIC padding (`.convStrided`, not `.convStridedXla`). -/
def r50DownGraphB (p epsStr : String) (N h w : Nat) {ic mid oc : Nat} (pw : R50ProjW ic mid oc)
    (e : SHlo (N * (ic * (2 * h) * (2 * w)))) : SHlo (N * (oc * h * w)) :=
  .batchOp (N := N) (.relu (n := oc * h * w))
    (.addVB
      (.bnBatchF s!"%{p}g3" s!"%{p}bt3" epsStr pw.ε₃ pw.γ₃ pw.β₃
        (.batchOp (N := N)
          (.conv (h := h) (w := w) s!"%{p}W3" (biasName false "" oc) pw.W₃ pw.b₃)
          (.batchOp (N := N) (.relu (n := mid * h * w))
            (.bnBatchF s!"%{p}g2" s!"%{p}bt2" epsStr pw.ε₂ pw.γ₂ pw.β₂
              (.batchOp (N := N)
                (.convStrided (h := h) (w := w) s!"%{p}W2" (biasName false "" mid) pw.W₂ pw.b₂)
                (.batchOp (N := N) (.relu (n := mid * (2 * h) * (2 * w)))
                  (.bnBatchF s!"%{p}g1" s!"%{p}bt1" epsStr pw.ε₁ pw.γ₁ pw.β₁
                    (.batchOp (N := N)
                      (.conv (h := 2 * h) (w := 2 * w) s!"%{p}W1" (biasName false "" mid)
                        pw.W₁ pw.b₁)
                      e))))))))
      (.bnBatchF s!"%{p}gp" s!"%{p}btp" epsStr pw.εp pw.γp pw.βp
        (.batchOp (N := N)
          (.convStrided (h := h) (w := w) s!"%{p}Wp" (biasName false "" oc) pw.Wp pw.bp) e)))

theorem r50DownGraphB_faithful (p epsStr : String) (N h w : Nat) {ic mid oc : Nat}
    (pw : R50ProjW ic mid oc) (e : SHlo (N * (ic * (2 * h) * (2 * w)))) :
    den (r50DownGraphB p epsStr N h w pw e) = r50DownB N h w pw (den e) := by
  unfold r50DownGraphB r50DownB projB projStridedB cbReluB cbReluStridedB residualProj biPath
  simp only [den_batchOp_relu_eq_reluF, reluF_faithful, den_batchOp_conv,
    den_batchOp_convStrided, den_bnBatchF, den_addVB, Function.comp_apply]
  congr 1
  funext i
  ring

/-- Stem graph: 7×7/s2 conv → batch BN → relu → He et al.'s 3×3/s2 max-pool.

    ⚠ NOT `r34StemGraphB`, and the difference is one string: that graph names the bias operand
    `"%sb"`, the `convBias := true` name, and `ResNet50RenderB` emits `%zb64`. The ops, their order
    and their `den` are identical. -/
def r50StemGraphB (epsStr : String) (N h w : Nat) {ic oc : Nat}
    (Ws : Kernel4 oc ic 7 7) (bs : Vec oc) (εs : ℝ) (γs βs : Vec oc)
    (e : SHlo (N * (ic * (2 * (2 * h)) * (2 * (2 * w))))) : SHlo (N * (oc * h * w)) :=
  .batchOp (N := N) (.maxPool3s2 (c := oc) (h := h) (w := w))
    (.batchOp (N := N) (.relu (n := oc * (2 * h) * (2 * w)))
      (.bnBatchF "%sg" "%sbt" epsStr εs γs βs
        (.batchOp (N := N)
          (.convStrided (h := 2 * h) (w := 2 * w) "%sW" (biasName false "" oc) Ws bs) e)))

theorem r50StemGraphB_faithful (epsStr : String) (N h w : Nat) {ic oc : Nat}
    (Ws : Kernel4 oc ic 7 7) (bs : Vec oc) (εs : ℝ) (γs βs : Vec oc)
    (e : SHlo (N * (ic * (2 * (2 * h)) * (2 * (2 * w))))) :
    den (r50StemGraphB epsStr N h w Ws bs εs γs βs e)
      = r34StemB N h w Ws bs εs γs βs (den e) := by
  unfold r50StemGraphB r34StemB cbReluStridedB
  simp only [den_batchOp_maxPool3s2, den_batchOp_relu_eq_reluF, reluF_faithful,
    den_batchOp_convStrided, den_bnBatchF, Function.comp_apply]

-- ════════════════════════════════════════════════════════════════
-- § The whole graph + faithfulness
-- ════════════════════════════════════════════════════════════════

/-- **The full batch-BN ResNet-50 forward graph.** Block prefixes are `ResNet50RenderB`'s own
    (`s1b0` … `s4b2`) and the head's are `%Wd`/`%bd`, so the typed graph diffs against
    `resnet50_fwd` and its ImageNet twins name for name. ⭐ The head graph is ResNet-34's,
    unchanged: `r34HeadGraphB` is generic in `{c nCls}` and emits the same two tokens. -/
def resnet50FwdGraphB_full (N q : Nat) (epsStr : String) {nCls : Nat} (w : R50BWeights nCls)
    (e : SHlo (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q)))))))) : SHlo (N * nCls) :=
  r34HeadGraphB N q q w.Wd w.bd
    (r50IdGraphB "s4b2" epsStr N q q w.s4b2
      (r50IdGraphB "s4b1" epsStr N q q w.s4b1
        (r50DownGraphB "s4b0" epsStr N q q w.s4b0
          (r50IdGraphB "s3b5" epsStr N (2 * q) (2 * q) w.s3b5
            (r50IdGraphB "s3b4" epsStr N (2 * q) (2 * q) w.s3b4
              (r50IdGraphB "s3b3" epsStr N (2 * q) (2 * q) w.s3b3
                (r50IdGraphB "s3b2" epsStr N (2 * q) (2 * q) w.s3b2
                  (r50IdGraphB "s3b1" epsStr N (2 * q) (2 * q) w.s3b1
                    (r50DownGraphB "s3b0" epsStr N (2 * q) (2 * q) w.s3b0
                      (r50IdGraphB "s2b3" epsStr N (2 * (2 * q)) (2 * (2 * q)) w.s2b3
                        (r50IdGraphB "s2b2" epsStr N (2 * (2 * q)) (2 * (2 * q)) w.s2b2
                          (r50IdGraphB "s2b1" epsStr N (2 * (2 * q)) (2 * (2 * q)) w.s2b1
                            (r50DownGraphB "s2b0" epsStr N (2 * (2 * q)) (2 * (2 * q)) w.s2b0
                              (r50IdGraphB "s1b2" epsStr N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) w.s1b2
                                (r50IdGraphB "s1b1" epsStr N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) w.s1b1
                                  (r50ProjGraphB "s1b0" epsStr N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) w.s1b0
                                    (r50StemGraphB epsStr N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) w.sW w.sb w.sε w.sγ w.sβ
                                      e)))))))))))))))))

/-- ⭐ **T2 for ResNet-50 at batch BN**: the typed graph denotes the whole-net forward. One `rw`
    per block over the four per-kind faithfulness lemmas — the first graph-level tier this net has
    ever had. -/
theorem resnet50FwdGraphB_full_faithful (N q : Nat) (epsStr : String) {nCls : Nat}
    (w : R50BWeights nCls) (e : SHlo (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q)))))))) :
    den (resnet50FwdGraphB_full N q epsStr w e) = resnet50ForwardB_full N q w (den e) := by
  unfold resnet50FwdGraphB_full resnet50ForwardB_full
  rw [r34HeadGraphB_faithful, r50IdGraphB_faithful, r50IdGraphB_faithful, r50DownGraphB_faithful, r50IdGraphB_faithful, r50IdGraphB_faithful, r50IdGraphB_faithful, r50IdGraphB_faithful, r50IdGraphB_faithful, r50DownGraphB_faithful, r50IdGraphB_faithful, r50IdGraphB_faithful, r50IdGraphB_faithful, r50DownGraphB_faithful, r50IdGraphB_faithful, r50IdGraphB_faithful, r50ProjGraphB_faithful,
      r50StemGraphB_faithful]

end StableHLO

end Proofs

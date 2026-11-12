module

public import LeanMlir.Proofs.Nets.ResNet.ResNet50FullB
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp

@[expose] public section

/-! # ResNet-34 and ResNet-50 at inference — eval forward + graph + faithfulness

The inference twins of `ResNet34FullB.lean` and `ResNet50FullB.lean`. Those files state the two
nets at TRAINING BatchNorm (`bnBatchLA`, statistics reduced out of the batch); this one states
the same block ladders at INFERENCE BatchNorm — frozen running statistics at every site, one
shared `ε` — and proves the typed `SHlo` graphs denote them:

* `resnet34FwdGraphEval_faithful` — the graph of `resnet34_fwd_eval.mlir` (36 BatchNorm sites);
* `resnet50FwdGraphEval_faithful` — the graph of `resnet50in_fwd_eval.mlir`,
  `resnet50in160_fwd_eval.mlir` and the `_s288` twin (53 sites), at the resolution binder `q`
  (`q = 7` is 224, `q = 5` is 160, `q = 9` is 288), as `ResNet50FullB`.

These are the forwards the ImageNet scores are computed with. The train steps return each
site's batch statistics, the driver keeps their running averages, and the eval forward reads
those as inputs.

**Per example, and the artifact is batched.** Both are stated for one example. Frozen-statistics
BatchNorm does not couple examples: it reads `μ, σ²` as inputs and reduces nothing. The renders
(`r34FwdChain`, `r50FwdChain`) write these graphs' own per-example constructors printed at the
artifact's batch `B`, so the artifact computes this forward at every example of its batch.
`FwdGraphTextTies` checks each block kind, the stem and the head against the render.

**The SSA names are the eval render's**: the parameters are the training graph's (`%sW`,
`%{p}W1`, `%{p}g1`, `%{p}bt1`, …, `%Wd`, `%bd`), and each BatchNorm's statistics are
`RenderKit.bnEvalSite`'s `%{p}n1mu` / `%{p}n1var` (`%stnmu` / `%stnvar` at the stem,
`%{p}npmu` / `%{p}npvar` at a projection). Names are pretty-printing metadata and do not enter
`den`.

**The residual add is in the render's order**, `addV(body, projection)`, and the forwards here
are written in that order, so no commutation is needed. `ResNet50FullB`'s training forward
writes `residualProj proj body`, the other order.

**The artifacts are the zero-bias instance.** The weight records carry a conv-bias field per
conv site, which `den` reads, while the graph spells each one as the `%zb{c}` zero operand
(`biasName false`). So the committed evals are the instance with every conv bias `0`. At
inference BatchNorm a conv bias does not cancel as it does under batch BN, so this is a
restriction, not a reparametrisation.

Nothing differentiates the eval forward, so there is no `CertLayer`, no VJP and no smoothness
hypothesis here.

## References

- He, Zhang, Ren, Sun 2016, *Deep Residual Learning for Image Recognition*. <https://arxiv.org/abs/1512.03385>
- Ioffe, Szegedy 2015, *Batch Normalization* (inference with population statistics). <https://arxiv.org/abs/1502.03167>
-/

namespace Proofs

-- ════════════════════════════════════════════════════════════════
-- § Weights and running statistics at inference
--   Per BatchNorm site: `γ, β` and the frozen `μ, v` (mean and variance). No per-site `ε`: the
--   eval forward takes one shared `ε`, as the render emits a single `eps` constant.
-- ════════════════════════════════════════════════════════════════

/-- One ResNet-34 identity basic block at inference: two 3×3 convs, each with its BatchNorm. -/
structure R34IdWEval (c : Nat) where
  W₁ : Kernel4 c c 3 3
  b₁ : Vec c
  γ₁ : Vec c
  β₁ : Vec c
  μ₁ : Vec c
  v₁ : Vec c
  W₂ : Kernel4 c c 3 3
  b₂ : Vec c
  γ₂ : Vec c
  β₂ : Vec c
  μ₂ : Vec c
  v₂ : Vec c

/-- One ResNet-34 downsample basic block at inference: strided 3×3, 3×3, and the 1×1 strided
    option-B skip, each with its BatchNorm. -/
structure R34DownWEval (ic oc : Nat) where
  W₁ : Kernel4 oc ic 3 3
  b₁ : Vec oc
  γ₁ : Vec oc
  β₁ : Vec oc
  μ₁ : Vec oc
  v₁ : Vec oc
  W₂ : Kernel4 oc oc 3 3
  b₂ : Vec oc
  γ₂ : Vec oc
  β₂ : Vec oc
  μ₂ : Vec oc
  v₂ : Vec oc
  Wp : Kernel4 oc ic 1 1
  bp : Vec oc
  γp : Vec oc
  βp : Vec oc
  μp : Vec oc
  vp : Vec oc

/-- Every ResNet-34 parameter and running statistic at inference. Field names are
    `R34BWeights`'s; the render's SSA prefixes for them are `s1b0 s1b1 s1b2 d2 s2b0 … s4b1`. -/
structure R34WeightsEval (nCls : Nat) where
  sW : Kernel4 64 3 7 7
  sb : Vec 64
  sγ : Vec 64
  sβ : Vec 64
  sμ : Vec 64
  sv : Vec 64
  a0 : R34IdWEval 64
  a1 : R34IdWEval 64
  a2 : R34IdWEval 64
  d2 : R34DownWEval 64 128
  b0 : R34IdWEval 128
  b1 : R34IdWEval 128
  b2 : R34IdWEval 128
  d3 : R34DownWEval 128 256
  c0 : R34IdWEval 256
  c1 : R34IdWEval 256
  c2 : R34IdWEval 256
  c3 : R34IdWEval 256
  c4 : R34IdWEval 256
  d4 : R34DownWEval 256 512
  e0 : R34IdWEval 512
  e1 : R34IdWEval 512
  Wd : Mat 512 nCls
  bd : Vec nCls

/-- One ResNet-50 identity bottleneck at inference: 1×1, 3×3, 1×1, each with its BatchNorm. -/
structure R50IdWEval (mid oc : Nat) where
  W₁ : Kernel4 mid oc 1 1
  b₁ : Vec mid
  γ₁ : Vec mid
  β₁ : Vec mid
  μ₁ : Vec mid
  v₁ : Vec mid
  W₂ : Kernel4 mid mid 3 3
  b₂ : Vec mid
  γ₂ : Vec mid
  β₂ : Vec mid
  μ₂ : Vec mid
  v₂ : Vec mid
  W₃ : Kernel4 oc mid 1 1
  b₃ : Vec oc
  γ₃ : Vec oc
  β₃ : Vec oc
  μ₃ : Vec oc
  v₃ : Vec oc

/-- One ResNet-50 projection bottleneck at inference, for both projection forms (stride-1 and
    strided), as `R50ProjW` is one record for both. -/
structure R50ProjWEval (ic mid oc : Nat) where
  W₁ : Kernel4 mid ic 1 1
  b₁ : Vec mid
  γ₁ : Vec mid
  β₁ : Vec mid
  μ₁ : Vec mid
  v₁ : Vec mid
  W₂ : Kernel4 mid mid 3 3
  b₂ : Vec mid
  γ₂ : Vec mid
  β₂ : Vec mid
  μ₂ : Vec mid
  v₂ : Vec mid
  W₃ : Kernel4 oc mid 1 1
  b₃ : Vec oc
  γ₃ : Vec oc
  β₃ : Vec oc
  μ₃ : Vec oc
  v₃ : Vec oc
  Wp : Kernel4 oc ic 1 1
  bp : Vec oc
  γp : Vec oc
  βp : Vec oc
  μp : Vec oc
  vp : Vec oc

/-- Every ResNet-50 parameter and running statistic at inference. Field names are
    `R50BWeights`'s, which are the render's SSA prefixes. 53 BatchNorm sites. -/
structure R50WeightsEval (nCls : Nat) where
  sW : Kernel4 64 3 7 7
  sb : Vec 64
  sγ : Vec 64
  sβ : Vec 64
  sμ : Vec 64
  sv : Vec 64
  s1b0 : R50ProjWEval 64 64 256
  s1b1 : R50IdWEval 64 256
  s1b2 : R50IdWEval 64 256
  s2b0 : R50ProjWEval 256 128 512
  s2b1 : R50IdWEval 128 512
  s2b2 : R50IdWEval 128 512
  s2b3 : R50IdWEval 128 512
  s3b0 : R50ProjWEval 512 256 1024
  s3b1 : R50IdWEval 256 1024
  s3b2 : R50IdWEval 256 1024
  s3b3 : R50IdWEval 256 1024
  s3b4 : R50IdWEval 256 1024
  s3b5 : R50IdWEval 256 1024
  s4b0 : R50ProjWEval 1024 512 2048
  s4b1 : R50IdWEval 512 2048
  s4b2 : R50IdWEval 512 2048
  Wd : Mat 2048 nCls
  bd : Vec nCls

-- ════════════════════════════════════════════════════════════════
-- § Block forwards at inference, per example
-- ════════════════════════════════════════════════════════════════
/-- ResNet-34 identity basic block at inference: `relu(bn₂(conv₂(relu(bn₁(conv₁ x)))) + x)`. -/
noncomputable def r34IdEval (h w : Nat) (ε : ℝ) {c : Nat} (p : R34IdWEval c)
    (x : Vec (c * h * w)) : Vec (c * h * w) :=
  relu (c * h * w) (fun i =>
    bnPerChannelEvalTensor3 c h w ε p.γ₂ p.β₂ p.μ₂ p.v₂ (flatConv (h := h) (w := w) p.W₂ p.b₂
      (relu (c * h * w) (bnPerChannelEvalTensor3 c h w ε p.γ₁ p.β₁ p.μ₁ p.v₁
        (flatConv (h := h) (w := w) p.W₁ p.b₁ x)))) i + x i)

/-- ResNet-34 downsample basic block at inference: strided 3×3 → BN → relu → 3×3 → BN, plus the
    strided 1×1 → BN skip, then relu. Both stride-2 sites are symmetric padding. -/
noncomputable def r34DownEval (h w : Nat) (ε : ℝ) {ic oc : Nat} (p : R34DownWEval ic oc)
    (x : Vec (ic * (2 * h) * (2 * w))) : Vec (oc * h * w) :=
  relu (oc * h * w) (fun i =>
    bnPerChannelEvalTensor3 oc h w ε p.γ₂ p.β₂ p.μ₂ p.v₂ (flatConv (h := h) (w := w) p.W₂ p.b₂
      (relu (oc * h * w) (bnPerChannelEvalTensor3 oc h w ε p.γ₁ p.β₁ p.μ₁ p.v₁
        (flatConvStride2 (h := h) (w := w) p.W₁ p.b₁ x)))) i
    + bnPerChannelEvalTensor3 oc h w ε p.γp p.βp p.μp p.vp
        (flatConvStride2 (h := h) (w := w) p.Wp p.bp x) i)

/-- The body of a ResNet-50 bottleneck at inference up to bn₃, stride 1:
    `bn₃(conv₃(relu(bn₂(conv₂(relu(bn₁(conv₁ x)))))))`. -/
noncomputable def r50BodyEval (h w : Nat) (ε : ℝ) {ic mid oc : Nat}
    (W₁ : Kernel4 mid ic 1 1) (b₁ γ₁ β₁ μ₁ v₁ : Vec mid)
    (W₂ : Kernel4 mid mid 3 3) (b₂ γ₂ β₂ μ₂ v₂ : Vec mid)
    (W₃ : Kernel4 oc mid 1 1) (b₃ γ₃ β₃ μ₃ v₃ : Vec oc)
    (x : Vec (ic * h * w)) : Vec (oc * h * w) :=
  bnPerChannelEvalTensor3 oc h w ε γ₃ β₃ μ₃ v₃ (flatConv (h := h) (w := w) W₃ b₃
    (relu (mid * h * w) (bnPerChannelEvalTensor3 mid h w ε γ₂ β₂ μ₂ v₂
      (flatConv (h := h) (w := w) W₂ b₂
        (relu (mid * h * w) (bnPerChannelEvalTensor3 mid h w ε γ₁ β₁ μ₁ v₁
          (flatConv (h := h) (w := w) W₁ b₁ x)))))))

/-- The strided bottleneck body at inference. v1.5: conv₁, bn₁ and relu₁ run at the input grid
    `2h × 2w`, and the stride is on the 3×3. -/
noncomputable def r50DownBodyEval (h w : Nat) (ε : ℝ) {ic mid oc : Nat}
    (W₁ : Kernel4 mid ic 1 1) (b₁ γ₁ β₁ μ₁ v₁ : Vec mid)
    (W₂ : Kernel4 mid mid 3 3) (b₂ γ₂ β₂ μ₂ v₂ : Vec mid)
    (W₃ : Kernel4 oc mid 1 1) (b₃ γ₃ β₃ μ₃ v₃ : Vec oc)
    (x : Vec (ic * (2 * h) * (2 * w))) : Vec (oc * h * w) :=
  bnPerChannelEvalTensor3 oc h w ε γ₃ β₃ μ₃ v₃ (flatConv (h := h) (w := w) W₃ b₃
    (relu (mid * h * w) (bnPerChannelEvalTensor3 mid h w ε γ₂ β₂ μ₂ v₂
      (flatConvStride2 (h := h) (w := w) W₂ b₂
        (relu (mid * (2 * h) * (2 * w)) (bnPerChannelEvalTensor3 mid (2 * h) (2 * w) ε γ₁ β₁ μ₁ v₁
          (flatConv (h := 2 * h) (w := 2 * w) W₁ b₁ x)))))))

/-- ResNet-50 identity bottleneck at inference: `relu(body x + x)`. -/
noncomputable def r50IdEval (h w : Nat) (ε : ℝ) {mid oc : Nat} (p : R50IdWEval mid oc)
    (x : Vec (oc * h * w)) : Vec (oc * h * w) :=
  relu (oc * h * w) (fun i =>
    r50BodyEval h w ε p.W₁ p.b₁ p.γ₁ p.β₁ p.μ₁ p.v₁ p.W₂ p.b₂ p.γ₂ p.β₂ p.μ₂ p.v₂
      p.W₃ p.b₃ p.γ₃ p.β₃ p.μ₃ p.v₃ x i + x i)

/-- ResNet-50 stride-1 projection bottleneck at inference (stage 1 block 0):
    `relu(body x + bn_p(conv_p x))`. -/
noncomputable def r50ProjEval (h w : Nat) (ε : ℝ) {ic mid oc : Nat} (p : R50ProjWEval ic mid oc)
    (x : Vec (ic * h * w)) : Vec (oc * h * w) :=
  relu (oc * h * w) (fun i =>
    r50BodyEval h w ε p.W₁ p.b₁ p.γ₁ p.β₁ p.μ₁ p.v₁ p.W₂ p.b₂ p.γ₂ p.β₂ p.μ₂ p.v₂
      p.W₃ p.b₃ p.γ₃ p.β₃ p.μ₃ p.v₃ x i
    + bnPerChannelEvalTensor3 oc h w ε p.γp p.βp p.μp p.vp
        (flatConv (h := h) (w := w) p.Wp p.bp x) i)

/-- ResNet-50 strided projection bottleneck at inference (stages 2/3/4 block 0). -/
noncomputable def r50DownEval (h w : Nat) (ε : ℝ) {ic mid oc : Nat} (p : R50ProjWEval ic mid oc)
    (x : Vec (ic * (2 * h) * (2 * w))) : Vec (oc * h * w) :=
  relu (oc * h * w) (fun i =>
    r50DownBodyEval h w ε p.W₁ p.b₁ p.γ₁ p.β₁ p.μ₁ p.v₁ p.W₂ p.b₂ p.γ₂ p.β₂ p.μ₂ p.v₂
      p.W₃ p.b₃ p.γ₃ p.β₃ p.μ₃ p.v₃ x i
    + bnPerChannelEvalTensor3 oc h w ε p.γp p.βp p.μp p.vp
        (flatConvStride2 (h := h) (w := w) p.Wp p.bp x) i)

/-- The ResNet stem at inference: 7×7/s2 conv → BN → relu → 3×3/s2 max-pool. Shared by both nets
    (the stem is the same function at the same widths). -/
noncomputable def resnetStemEval (h w : Nat) (ε : ℝ) (W : Kernel4 64 3 7 7) (b γ β μ v : Vec 64)
    (x : Vec (3 * (2 * (2 * h)) * (2 * (2 * w)))) : Vec (64 * h * w) :=
  maxPool3s2Flat 64 h w (relu (64 * (2 * h) * (2 * w))
    (bnPerChannelEvalTensor3 64 (2 * h) (2 * w) ε γ β μ v
      (flatConvStride2 (h := 2 * h) (w := 2 * w) W b x)))

-- ════════════════════════════════════════════════════════════════
-- § The whole nets at inference
-- ════════════════════════════════════════════════════════════════

/-- **The ResNet-34 inference forward**, one example, `3×224×224 → nCls`: the `[3,4,6,3]`
    basic-block ladder with frozen running statistics at all 36 BatchNorm sites, at one shared
    `ε`. Nested-application form, as the training twin. -/
noncomputable def resnet34ForwardEval {nCls : Nat} (ε : ℝ) (w : R34WeightsEval nCls)
    (x : Vec (3 * 224 * 224)) : Vec nCls :=
  dense w.Wd w.bd
    (globalAvgPoolFlat 512 7 7
    (r34IdEval 7 7 ε w.e1
      (r34IdEval 7 7 ε w.e0
        (r34DownEval 7 7 ε w.d4
          (r34IdEval 14 14 ε w.c4
            (r34IdEval 14 14 ε w.c3
              (r34IdEval 14 14 ε w.c2
                (r34IdEval 14 14 ε w.c1
                  (r34IdEval 14 14 ε w.c0
                    (r34DownEval 14 14 ε w.d3
                      (r34IdEval 28 28 ε w.b2
                        (r34IdEval 28 28 ε w.b1
                          (r34IdEval 28 28 ε w.b0
                            (r34DownEval 28 28 ε w.d2
                              (r34IdEval 56 56 ε w.a2
                                (r34IdEval 56 56 ε w.a1
                                  (r34IdEval 56 56 ε w.a0
                                    (resnetStemEval 56 56 ε w.sW w.sb w.sγ w.sβ w.sμ w.sv x))))))))))))))))))

/-- **The ResNet-50 inference forward**, one example, at the resolution binder `q` (input side
    `32q`): the `[3,4,6,3]` bottleneck ladder with frozen running statistics at all 53 BatchNorm
    sites, at one shared `ε`. `q = 7` is `resnet50in_fwd_eval`, `q = 5` the 160-px twin. -/
noncomputable def resnet50ForwardEval (q : Nat) {nCls : Nat} (ε : ℝ) (w : R50WeightsEval nCls)
    (x : Vec (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q))))))) : Vec nCls :=
  dense w.Wd w.bd
    (globalAvgPoolFlat 2048 q q
    (r50IdEval q q ε w.s4b2
      (r50IdEval q q ε w.s4b1
        (r50DownEval q q ε w.s4b0
          (r50IdEval (2 * q) (2 * q) ε w.s3b5
            (r50IdEval (2 * q) (2 * q) ε w.s3b4
              (r50IdEval (2 * q) (2 * q) ε w.s3b3
                (r50IdEval (2 * q) (2 * q) ε w.s3b2
                  (r50IdEval (2 * q) (2 * q) ε w.s3b1
                    (r50DownEval (2 * q) (2 * q) ε w.s3b0
                      (r50IdEval (2 * (2 * q)) (2 * (2 * q)) ε w.s2b3
                        (r50IdEval (2 * (2 * q)) (2 * (2 * q)) ε w.s2b2
                          (r50IdEval (2 * (2 * q)) (2 * (2 * q)) ε w.s2b1
                            (r50DownEval (2 * (2 * q)) (2 * (2 * q)) ε w.s2b0
                              (r50IdEval (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) ε w.s1b2
                                (r50IdEval (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) ε w.s1b1
                                  (r50ProjEval (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) ε w.s1b0
                                    (resnetStemEval (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) ε w.sW w.sb w.sγ w.sβ w.sμ w.sv x))))))))))))))))))

namespace StableHLO

-- ════════════════════════════════════════════════════════════════
-- § Per-block typed inference graphs + faithfulness
--   `p` is the render's SSA prefix. The statistics are `bnEvalSite`'s `%{p}n1mu` / `%{p}n1var`.
-- ════════════════════════════════════════════════════════════════

/-- ResNet-34 identity basic block inference graph: `relu(addV(bn₂(conv₂(relu(bn₁(conv₁ e)))), e))`. -/
def r34IdGraphEval (p epsStr : String) (h w : Nat) (ε : ℝ) {c : Nat} (pw : R34IdWEval c)
    (e : SHlo (c * h * w)) : SHlo (c * h * w) :=
  .reluF (.addV
    (.bnPerChannelEvalF (oc := c) (h := h) (w := w) s!"%{p}g2" s!"%{p}bt2"
      s!"%{p}n2mu" s!"%{p}n2var" epsStr ε pw.γ₂ pw.β₂ pw.μ₂ pw.v₂
      (.flatConvF (h := h) (w := w) s!"%{p}W2" (biasName false "" c) pw.W₂ pw.b₂ (.reluF
      (.bnPerChannelEvalF (oc := c) (h := h) (w := w) s!"%{p}g1" s!"%{p}bt1"
      s!"%{p}n1mu" s!"%{p}n1var" epsStr ε pw.γ₁ pw.β₁ pw.μ₁ pw.v₁
      (.flatConvF (h := h) (w := w) s!"%{p}W1" (biasName false "" c) pw.W₁ pw.b₁ (e))))))
    (e))

theorem r34IdGraphEval_faithful (p epsStr : String) (h w : Nat) (ε : ℝ) {c : Nat}
    (pw : R34IdWEval c) (e : SHlo (c * h * w)) :
    den (r34IdGraphEval p epsStr h w ε pw e) = r34IdEval h w ε pw (den e) := by
  simp only [r34IdGraphEval, r34IdEval, reluF_faithful, den_addV, bnPerChannelEvalF_faithful, flatConvF_faithful]

/-- ResNet-34 downsample basic block inference graph, `addV(body, projection)` in the render's order. -/
def r34DownGraphEval (p epsStr : String) (h w : Nat) (ε : ℝ) {ic oc : Nat} (pw : R34DownWEval ic oc)
    (e : SHlo (ic * (2 * h) * (2 * w))) : SHlo (oc * h * w) :=
  .reluF (.addV
    (.bnPerChannelEvalF (oc := oc) (h := h) (w := w) s!"%{p}g2" s!"%{p}bt2"
      s!"%{p}n2mu" s!"%{p}n2var" epsStr ε pw.γ₂ pw.β₂ pw.μ₂ pw.v₂
      (.flatConvF (h := h) (w := w) s!"%{p}W2" (biasName false "" oc) pw.W₂ pw.b₂ (.reluF
      (.bnPerChannelEvalF (oc := oc) (h := h) (w := w) s!"%{p}g1" s!"%{p}bt1"
      s!"%{p}n1mu" s!"%{p}n1var" epsStr ε pw.γ₁ pw.β₁ pw.μ₁ pw.v₁
      (.flatConvStridedF (h := h) (w := w) s!"%{p}W1" (biasName false "" oc) pw.W₁ pw.b₁ (e))))))
    (.bnPerChannelEvalF (oc := oc) (h := h) (w := w) s!"%{p}gp" s!"%{p}btp"
      s!"%{p}npmu" s!"%{p}npvar" epsStr ε pw.γp pw.βp pw.μp pw.vp
      (.flatConvStridedF (h := h) (w := w) s!"%{p}Wp" (biasName false "" oc) pw.Wp pw.bp (e))))

theorem r34DownGraphEval_faithful (p epsStr : String) (h w : Nat) (ε : ℝ) {ic oc : Nat}
    (pw : R34DownWEval ic oc) (e : SHlo (ic * (2 * h) * (2 * w))) :
    den (r34DownGraphEval p epsStr h w ε pw e) = r34DownEval h w ε pw (den e) := by
  simp only [r34DownGraphEval, r34DownEval, reluF_faithful, den_addV, bnPerChannelEvalF_faithful, flatConvF_faithful,
    flatConvStridedF_faithful]

/-- ResNet-50 identity bottleneck inference graph. -/
def r50IdGraphEval (p epsStr : String) (h w : Nat) (ε : ℝ) {mid oc : Nat} (pw : R50IdWEval mid oc)
    (e : SHlo (oc * h * w)) : SHlo (oc * h * w) :=
  .reluF (.addV
    (.bnPerChannelEvalF (oc := oc) (h := h) (w := w) s!"%{p}g3" s!"%{p}bt3"
      s!"%{p}n3mu" s!"%{p}n3var" epsStr ε pw.γ₃ pw.β₃ pw.μ₃ pw.v₃
      (.flatConvF (h := h) (w := w) s!"%{p}W3" (biasName false "" oc) pw.W₃ pw.b₃ (.reluF
      (.bnPerChannelEvalF (oc := mid) (h := h) (w := w) s!"%{p}g2" s!"%{p}bt2"
      s!"%{p}n2mu" s!"%{p}n2var" epsStr ε pw.γ₂ pw.β₂ pw.μ₂ pw.v₂
      (.flatConvF (h := h) (w := w) s!"%{p}W2" (biasName false "" mid) pw.W₂ pw.b₂ (.reluF
      (.bnPerChannelEvalF (oc := mid) (h := h) (w := w) s!"%{p}g1" s!"%{p}bt1"
      s!"%{p}n1mu" s!"%{p}n1var" epsStr ε pw.γ₁ pw.β₁ pw.μ₁ pw.v₁
      (.flatConvF (h := h) (w := w) s!"%{p}W1" (biasName false "" mid) pw.W₁ pw.b₁ (e)))))))))
    (e))

theorem r50IdGraphEval_faithful (p epsStr : String) (h w : Nat) (ε : ℝ) {mid oc : Nat}
    (pw : R50IdWEval mid oc) (e : SHlo (oc * h * w)) :
    den (r50IdGraphEval p epsStr h w ε pw e) = r50IdEval h w ε pw (den e) := by
  simp only [r50IdGraphEval, r50IdEval, r50BodyEval, reluF_faithful, den_addV, bnPerChannelEvalF_faithful, flatConvF_faithful]

/-- ResNet-50 stride-1 projection bottleneck inference graph (stage 1 block 0), `addV(body, projection)`. -/
def r50ProjGraphEval (p epsStr : String) (h w : Nat) (ε : ℝ) {ic mid oc : Nat} (pw : R50ProjWEval ic mid oc)
    (e : SHlo (ic * h * w)) : SHlo (oc * h * w) :=
  .reluF (.addV
    (.bnPerChannelEvalF (oc := oc) (h := h) (w := w) s!"%{p}g3" s!"%{p}bt3"
      s!"%{p}n3mu" s!"%{p}n3var" epsStr ε pw.γ₃ pw.β₃ pw.μ₃ pw.v₃
      (.flatConvF (h := h) (w := w) s!"%{p}W3" (biasName false "" oc) pw.W₃ pw.b₃ (.reluF
      (.bnPerChannelEvalF (oc := mid) (h := h) (w := w) s!"%{p}g2" s!"%{p}bt2"
      s!"%{p}n2mu" s!"%{p}n2var" epsStr ε pw.γ₂ pw.β₂ pw.μ₂ pw.v₂
      (.flatConvF (h := h) (w := w) s!"%{p}W2" (biasName false "" mid) pw.W₂ pw.b₂ (.reluF
      (.bnPerChannelEvalF (oc := mid) (h := h) (w := w) s!"%{p}g1" s!"%{p}bt1"
      s!"%{p}n1mu" s!"%{p}n1var" epsStr ε pw.γ₁ pw.β₁ pw.μ₁ pw.v₁
      (.flatConvF (h := h) (w := w) s!"%{p}W1" (biasName false "" mid) pw.W₁ pw.b₁ (e)))))))))
    (.bnPerChannelEvalF (oc := oc) (h := h) (w := w) s!"%{p}gp" s!"%{p}btp"
      s!"%{p}npmu" s!"%{p}npvar" epsStr ε pw.γp pw.βp pw.μp pw.vp
      (.flatConvF (h := h) (w := w) s!"%{p}Wp" (biasName false "" oc) pw.Wp pw.bp (e))))

theorem r50ProjGraphEval_faithful (p epsStr : String) (h w : Nat) (ε : ℝ) {ic mid oc : Nat}
    (pw : R50ProjWEval ic mid oc) (e : SHlo (ic * h * w)) :
    den (r50ProjGraphEval p epsStr h w ε pw e) = r50ProjEval h w ε pw (den e) := by
  simp only [r50ProjGraphEval, r50ProjEval, r50BodyEval, reluF_faithful, den_addV, bnPerChannelEvalF_faithful, flatConvF_faithful]

/-- ResNet-50 strided projection bottleneck inference graph (stages 2/3/4 block 0). v1.5: the stride is on the 3×3 and the skip. -/
def r50DownGraphEval (p epsStr : String) (h w : Nat) (ε : ℝ) {ic mid oc : Nat} (pw : R50ProjWEval ic mid oc)
    (e : SHlo (ic * (2 * h) * (2 * w))) : SHlo (oc * h * w) :=
  .reluF (.addV
    (.bnPerChannelEvalF (oc := oc) (h := h) (w := w) s!"%{p}g3" s!"%{p}bt3"
      s!"%{p}n3mu" s!"%{p}n3var" epsStr ε pw.γ₃ pw.β₃ pw.μ₃ pw.v₃
      (.flatConvF (h := h) (w := w) s!"%{p}W3" (biasName false "" oc) pw.W₃ pw.b₃ (.reluF
      (.bnPerChannelEvalF (oc := mid) (h := h) (w := w) s!"%{p}g2" s!"%{p}bt2"
      s!"%{p}n2mu" s!"%{p}n2var" epsStr ε pw.γ₂ pw.β₂ pw.μ₂ pw.v₂
      (.flatConvStridedF (h := h) (w := w) s!"%{p}W2" (biasName false "" mid) pw.W₂ pw.b₂ (.reluF
      (.bnPerChannelEvalF (oc := mid) (h := 2 * h) (w := 2 * w) s!"%{p}g1" s!"%{p}bt1"
      s!"%{p}n1mu" s!"%{p}n1var" epsStr ε pw.γ₁ pw.β₁ pw.μ₁ pw.v₁
      (.flatConvF (h := 2 * h) (w := 2 * w) s!"%{p}W1" (biasName false "" mid) pw.W₁ pw.b₁ (e)))))))))
    (.bnPerChannelEvalF (oc := oc) (h := h) (w := w) s!"%{p}gp" s!"%{p}btp"
      s!"%{p}npmu" s!"%{p}npvar" epsStr ε pw.γp pw.βp pw.μp pw.vp
      (.flatConvStridedF (h := h) (w := w) s!"%{p}Wp" (biasName false "" oc) pw.Wp pw.bp (e))))

theorem r50DownGraphEval_faithful (p epsStr : String) (h w : Nat) (ε : ℝ) {ic mid oc : Nat}
    (pw : R50ProjWEval ic mid oc) (e : SHlo (ic * (2 * h) * (2 * w))) :
    den (r50DownGraphEval p epsStr h w ε pw e) = r50DownEval h w ε pw (den e) := by
  simp only [r50DownGraphEval, r50DownEval, r50DownBodyEval, reluF_faithful, den_addV, bnPerChannelEvalF_faithful, flatConvF_faithful,
    flatConvStridedF_faithful]

/-- The ResNet stem inference graph: 7×7/s2 conv → BN (`stn`) → relu → 3×3/s2 max-pool. -/
def resnetStemGraphEval (epsStr : String) (h w : Nat) (ε : ℝ) (W : Kernel4 64 3 7 7)
    (b γ β μ v : Vec 64) (e : SHlo (3 * (2 * (2 * h)) * (2 * (2 * w)))) : SHlo (64 * h * w) :=
  .maxPool3s2F (c := 64) (h := h) (w := w)
    (.reluF (.bnPerChannelEvalF (oc := 64) (h := 2 * h) (w := 2 * w) "%sg" "%sbt" "%stnmu" "%stnvar"
        epsStr ε γ β μ v
      (.flatConvStridedF (h := 2 * h) (w := 2 * w) "%sW" (biasName false "" 64) W b e)))

theorem resnetStemGraphEval_faithful (epsStr : String) (h w : Nat) (ε : ℝ) (W : Kernel4 64 3 7 7)
    (b γ β μ v : Vec 64) (e : SHlo (3 * (2 * (2 * h)) * (2 * (2 * w)))) :
    den (resnetStemGraphEval epsStr h w ε W b γ β μ v e) = resnetStemEval h w ε W b γ β μ v (den e) := by
  simp only [resnetStemGraphEval, resnetStemEval, maxPool3s2F_faithful, reluF_faithful,
    bnPerChannelEvalF_faithful, flatConvStridedF_faithful]

/-- The ResNet head inference graph: GAP → dense. No BatchNorm, so it is the training head's
    per-example form. -/
def resnetHeadGraphEval {c nCls : Nat} (h w : Nat) (Wd : Mat c nCls) (bd : Vec nCls)
    (e : SHlo (c * h * w)) : SHlo nCls :=
  denseF "%Wd" "%bd" Wd bd (.gapF (c := c) (h := h) (w := w) e)

-- ════════════════════════════════════════════════════════════════
-- § The whole graphs + faithfulness
-- ════════════════════════════════════════════════════════════════

/-- **The ResNet-34 inference forward graph** (`3×224×224 → nCls`), the typed form of
    `resnet34_fwd_eval`: every one of the 36 BatchNorm sites a `bnPerChannelEvalF` reading frozen
    running statistics, block prefixes the render's. -/
def resnet34FwdGraphEval {nCls : Nat} (epsStr : String) (ε : ℝ) (w : R34WeightsEval nCls)
    (x : Vec (3 * 224 * 224)) : SHlo nCls :=
  resnetHeadGraphEval 7 7 w.Wd w.bd
    (r34IdGraphEval "s4b1" epsStr 7 7 ε w.e1
      (r34IdGraphEval "s4b0" epsStr 7 7 ε w.e0
        (r34DownGraphEval "d4" epsStr 7 7 ε w.d4
          (r34IdGraphEval "s3b4" epsStr 14 14 ε w.c4
            (r34IdGraphEval "s3b3" epsStr 14 14 ε w.c3
              (r34IdGraphEval "s3b2" epsStr 14 14 ε w.c2
                (r34IdGraphEval "s3b1" epsStr 14 14 ε w.c1
                  (r34IdGraphEval "s3b0" epsStr 14 14 ε w.c0
                    (r34DownGraphEval "d3" epsStr 14 14 ε w.d3
                      (r34IdGraphEval "s2b2" epsStr 28 28 ε w.b2
                        (r34IdGraphEval "s2b1" epsStr 28 28 ε w.b1
                          (r34IdGraphEval "s2b0" epsStr 28 28 ε w.b0
                            (r34DownGraphEval "d2" epsStr 28 28 ε w.d2
                              (r34IdGraphEval "s1b2" epsStr 56 56 ε w.a2
                                (r34IdGraphEval "s1b1" epsStr 56 56 ε w.a1
                                  (r34IdGraphEval "s1b0" epsStr 56 56 ε w.a0
                                    (resnetStemGraphEval epsStr 56 56 ε w.sW w.sb w.sγ w.sβ w.sμ w.sv (.operand "%x" x))))))))))))))))))

/-- **ResNet-34 inference forward faithfulness.** The typed graph denotes `resnet34ForwardEval`,
    for every shared `ε`, weight record and input example. -/
theorem resnet34FwdGraphEval_faithful {nCls : Nat} (epsStr : String) (ε : ℝ)
    (w : R34WeightsEval nCls) (x : Vec (3 * 224 * 224)) :
    den (resnet34FwdGraphEval epsStr ε w x) = resnet34ForwardEval ε w x := by
  simp only [resnet34FwdGraphEval, resnet34ForwardEval, resnetHeadGraphEval, denseF_faithful,
    gapF_faithful, r34IdGraphEval_faithful, r34DownGraphEval_faithful,
    resnetStemGraphEval_faithful, den_operand]

/-- **The ResNet-50 inference forward graph** at the resolution binder `q`, the typed form of
    `resnet50in_fwd_eval` (`q = 7`) and `resnet50in160_fwd_eval` (`q = 5`): every one of the 53
    BatchNorm sites a `bnPerChannelEvalF` reading frozen running statistics. -/
def resnet50FwdGraphEval (q : Nat) {nCls : Nat} (epsStr : String) (ε : ℝ)
    (w : R50WeightsEval nCls) (x : Vec (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q))))))) : SHlo nCls :=
  resnetHeadGraphEval q q w.Wd w.bd
    (r50IdGraphEval "s4b2" epsStr q q ε w.s4b2
      (r50IdGraphEval "s4b1" epsStr q q ε w.s4b1
        (r50DownGraphEval "s4b0" epsStr q q ε w.s4b0
          (r50IdGraphEval "s3b5" epsStr (2 * q) (2 * q) ε w.s3b5
            (r50IdGraphEval "s3b4" epsStr (2 * q) (2 * q) ε w.s3b4
              (r50IdGraphEval "s3b3" epsStr (2 * q) (2 * q) ε w.s3b3
                (r50IdGraphEval "s3b2" epsStr (2 * q) (2 * q) ε w.s3b2
                  (r50IdGraphEval "s3b1" epsStr (2 * q) (2 * q) ε w.s3b1
                    (r50DownGraphEval "s3b0" epsStr (2 * q) (2 * q) ε w.s3b0
                      (r50IdGraphEval "s2b3" epsStr (2 * (2 * q)) (2 * (2 * q)) ε w.s2b3
                        (r50IdGraphEval "s2b2" epsStr (2 * (2 * q)) (2 * (2 * q)) ε w.s2b2
                          (r50IdGraphEval "s2b1" epsStr (2 * (2 * q)) (2 * (2 * q)) ε w.s2b1
                            (r50DownGraphEval "s2b0" epsStr (2 * (2 * q)) (2 * (2 * q)) ε w.s2b0
                              (r50IdGraphEval "s1b2" epsStr (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) ε w.s1b2
                                (r50IdGraphEval "s1b1" epsStr (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) ε w.s1b1
                                  (r50ProjGraphEval "s1b0" epsStr (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) ε w.s1b0
                                    (resnetStemGraphEval epsStr (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) ε w.sW w.sb w.sγ w.sβ w.sμ w.sv (.operand "%x" x))))))))))))))))))

/-- **ResNet-50 inference forward faithfulness**, at every resolution `q`: the typed graph
    denotes `resnet50ForwardEval`, for every shared `ε`, weight record and input example. -/
theorem resnet50FwdGraphEval_faithful (q : Nat) {nCls : Nat} (epsStr : String) (ε : ℝ)
    (w : R50WeightsEval nCls) (x : Vec (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q))))))) :
    den (resnet50FwdGraphEval q epsStr ε w x) = resnet50ForwardEval q ε w x := by
  simp only [resnet50FwdGraphEval, resnet50ForwardEval, resnetHeadGraphEval, denseF_faithful,
    gapF_faithful, r50IdGraphEval_faithful, r50ProjGraphEval_faithful, r50DownGraphEval_faithful,
    resnetStemGraphEval_faithful, den_operand]

-- `q = 7` IS the 224-px net and `q = 5` the 160-px one: the input binds at the literal shape.
example {nCls : Nat} (epsStr : String) (ε : ℝ) (w : R50WeightsEval nCls)
    (x : Vec (3 * 224 * 224)) :
    den (resnet50FwdGraphEval 7 epsStr ε w x) = resnet50ForwardEval 7 ε w x :=
  resnet50FwdGraphEval_faithful 7 epsStr ε w x

example {nCls : Nat} (epsStr : String) (ε : ℝ) (w : R50WeightsEval nCls)
    (x : Vec (3 * 160 * 160)) :
    den (resnet50FwdGraphEval 5 epsStr ε w x) = resnet50ForwardEval 5 ε w x :=
  resnet50FwdGraphEval_faithful 5 epsStr ε w x

end StableHLO

end Proofs

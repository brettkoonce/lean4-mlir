import LeanMlir.Proofs.Nets.MobileNet.MobileNetV4FullB

/-! # MobileNetV4-Conv-M at inference — eval forward + graph + faithfulness, at any resolution

The eval twin of `MobileNetV4FullB.lean`. That file states Conv-M at TRAINING BatchNorm
(`bnBatchLA`) at 224×224; this one states the same 21-row table at INFERENCE BatchNorm — frozen
running statistics at all **77** sites, one shared `ε` — and proves its typed `SHlo` graph denotes
it (`mnv4FwdGraphBFullEval_faithful`). That is the graph of `mnv4_fwd_eval.mlir`,
`mnv4in_fwd_eval.mlir` and `mnv4in_fwd_eval_s256.mlir`.

**One statement, every input size.** The net is stated at a binder `f`, the final feature side:
the input is `32f`, the stem out `16f`, the fused stage `8f`, and the three stride-2 rows take it to
`4f`, `2f`, `f`. The committed evals are `f = 7` (224) and `f = 8` (256, timm's test size for
`mobilenetv4_conv_medium.e500_r224_in1k`) — `mnv4FwdChainB`'s own `f`. The ladder is written as
nested doublings (`2 * (2 * f)`, not `4 * f`) so a strided row's input side is its output side's
`2 * h` by the types alone.

**Why no `CertLayer` and no resolution groups.** Nothing differentiates the eval forward, so the
blocks are plain functions; and at a variable `f` `den` stays stuck, so the single rewrite chain
that times out at the training file's literal resolutions is small here.

**Families by dispatch, as the render does it.** A row's two depthwise positions are `if`s on
`s.preDWk` / `s.postDWk` in the graph (`mnv4PreDWGraphBEval`, `mnv4PostDWGraphBEval`), exactly the
`if`s `uibFwdSkipB` / `uibFwdStridedB` branch on, so one body graph covers ExtraDW, ConvNeXt-like,
FFN (and IB, which Conv-M does not use) and one theorem proves it. The table picks the family.

**What it is tied to.** `%x` + 233 parameters + 154 statistic slots = 388 inputs. The SSA names
are the eval render's: each BN's statistics are `%{site}mu` / `%{site}var` with the site the
render's `mnv4Bn` `statP` (`stn`, `f0cn`, `u{p}qn`, …, `hn`). `FwdGraphTextTies` checks every
block, the stem, the fused stage and the head against the render at `.eval`, at both `f = 7` and
`f = 8`.
-/

namespace Proofs.StableHLO

-- ════════════════════════════════════════════════════════════════
-- § The batched inference stages
-- ════════════════════════════════════════════════════════════════

/-- Batched `k×k` conv (any stride-1 extent) → inference BN → relu. -/
@[reducible] noncomputable def mnv4CbReluBEval (N : Nat) {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (γ β μ v : Vec oc) :
    Vec (N * (ic * h * w)) → Vec (N * (oc * h * w)) :=
  batchMap N (relu (oc * h * w)) ∘ batchMap N (bnPerChannelEvalTensor3 oc h w ε γ β μ v) ∘
    batchMap N (flatConv W b)

/-- Batched stride-2 conv, SYMMETRIC padding (the stem and the fused stage) → inference BN →
    relu. -/
@[reducible] noncomputable def mnv4CbReluSBEval (N : Nat) {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (γ β μ v : Vec oc) :
    Vec (N * (ic * (2 * h) * (2 * w))) → Vec (N * (oc * h * w)) :=
  batchMap N (relu (oc * h * w)) ∘ batchMap N (bnPerChannelEvalTensor3 oc h w ε γ β μ v) ∘
    batchMap N (flatConvStride2 W b)

/-- Batched 1×1 project → inference BN, no activation (the linear bottleneck). -/
@[reducible] noncomputable def mnv4ProjBEval (N : Nat) {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (γ β μ v : Vec oc) :
    Vec (N * (ic * h * w)) → Vec (N * (oc * h * w)) :=
  batchMap N (bnPerChannelEvalTensor3 oc h w ε γ β μ v) ∘ batchMap N (flatConv W b)

/-- Batched depthwise → inference BN, no activation (timm's `dw_start`, the pre-DW). -/
@[reducible] noncomputable def mnv4DWBEval (N : Nat) {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε : ℝ) (γ β μ v : Vec c) :
    Vec (N * (c * h * w)) → Vec (N * (c * h * w)) :=
  batchMap N (bnPerChannelEvalTensor3 c h w ε γ β μ v) ∘ batchMap N (depthwiseFlat W b)

/-- Batched depthwise → inference BN → relu (the stride-1 post-DW). -/
@[reducible] noncomputable def mnv4DWReluBEval (N : Nat) {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε : ℝ) (γ β μ v : Vec c) :
    Vec (N * (c * h * w)) → Vec (N * (c * h * w)) :=
  batchMap N (relu (c * h * w)) ∘ batchMap N (bnPerChannelEvalTensor3 c h w ε γ β μ v) ∘
    batchMap N (depthwiseFlat W b)

/-- Batched stride-2 depthwise, symmetric → inference BN → relu (the strided post-DW). -/
@[reducible] noncomputable def mnv4DWReluSBEval (N : Nat) {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε : ℝ) (γ β μ v : Vec c) :
    Vec (N * (c * (2 * h) * (2 * w))) → Vec (N * (c * h * w)) :=
  batchMap N (relu (c * h * w)) ∘ batchMap N (bnPerChannelEvalTensor3 c h w ε γ β μ v) ∘
    batchMap N (depthwiseStride2Flat W b)

-- ════════════════════════════════════════════════════════════════
-- § The weights at inference, typed by their table rows
-- ════════════════════════════════════════════════════════════════

/-- One depthwise site at inference: kernel, bias, and the BN's `γ, β` plus its frozen `μ, σ²`. -/
structure Mnv4DWEval (c k : Nat) where
  W : DepthwiseKernel c k k
  b : Vec c
  γ : Vec c
  β : Vec c
  μ : Vec c
  v : Vec c

/-- A depthwise slot at inference: nothing at `k = 0`, a `Mnv4DWEval` otherwise — `DWSlot`'s
    eval twin. -/
def Mnv4DWEvalSlot (c : Nat) : Nat → Type
  | 0 => PUnit
  | k + 1 => Mnv4DWEval c (k + 1)

/-- The slot's parameters at any `k`: the stored ones at `k > 0`, a zero placeholder at `k = 0`
    that the absent slot never reads. -/
def Mnv4DWEvalSlot.params {c : Nat} : {k : Nat} → Mnv4DWEvalSlot c k → Mnv4DWEval c k
  | 0, _ => ⟨fun _ _ _ => 0, fun _ => 0, fun _ => 0, fun _ => 0, fun _ => 0, fun _ => 0⟩
  | _ + 1, p => p

/-- **One UIB block's inference parameters, typed by its table row** — `UibParams`'s eval twin:
    each BN carries its frozen `μ, σ²` and no `ε` (the net shares one). The row's `h` appears in
    no field, which is what lets one record serve every input size. -/
structure UibEvalParams (s : UibSpec) where
  pre : Mnv4DWEvalSlot s.ic s.preDWk
  We : Kernel4 (s.ic * s.expand) s.ic 1 1
  be : Vec (s.ic * s.expand)
  ge : Vec (s.ic * s.expand)
  bte : Vec (s.ic * s.expand)
  mue : Vec (s.ic * s.expand)
  ve : Vec (s.ic * s.expand)
  post : Mnv4DWEvalSlot (s.ic * s.expand) s.postDWk
  Wz : Kernel4 s.oc (s.ic * s.expand) 1 1
  bz : Vec s.oc
  gz : Vec s.oc
  btz : Vec s.oc
  muz : Vec s.oc
  vz : Vec s.oc

/-- **Every MobileNetV4-Conv-M inference parameter**, generic in the class count: the 233
    parameters of `Mnv4BWeights` without its per-site `ε`s, plus the 77 BN sites' `μ, σ²`. Field
    names follow the render's SSA prefixes. -/
structure Mnv4BWeightsEval (nCls : Nat) where
  sW : Kernel4 32 3 3 3
  sb : Vec 32
  sg : Vec 32
  sbt : Vec 32
  smu : Vec 32
  sv : Vec 32
  f0cW : Kernel4 128 32 3 3
  f0cb : Vec 128
  f0cg : Vec 128
  f0cbt : Vec 128
  f0cmu : Vec 128
  f0cv : Vec 128
  f0pW : Kernel4 48 128 1 1
  f0pb : Vec 48
  f0pg : Vec 48
  f0pbt : Vec 48
  f0pmu : Vec 48
  f0pv : Vec 48
  b1 : UibEvalParams mnv4Row1
  b2 : UibEvalParams mnv4Row2
  b3 : UibEvalParams mnv4Row3
  b4 : UibEvalParams mnv4Row4
  b5 : UibEvalParams mnv4Row5
  b6 : UibEvalParams mnv4Row6
  b7 : UibEvalParams mnv4Row7
  b8 : UibEvalParams mnv4Row8
  b9 : UibEvalParams mnv4Row9
  b10 : UibEvalParams mnv4Row10
  b11 : UibEvalParams mnv4Row11
  b12 : UibEvalParams mnv4Row12
  b13 : UibEvalParams mnv4Row13
  b14 : UibEvalParams mnv4Row14
  b15 : UibEvalParams mnv4Row15
  b16 : UibEvalParams mnv4Row16
  b17 : UibEvalParams mnv4Row17
  b18 : UibEvalParams mnv4Row18
  b19 : UibEvalParams mnv4Row19
  b20 : UibEvalParams mnv4Row20
  b21 : UibEvalParams mnv4Row21
  h1W : Kernel4 960 256 1 1
  h1b : Vec 960
  h1g : Vec 960
  h1bt : Vec 960
  h1mu : Vec 960
  h1v : Vec 960
  hW : Kernel4 1280 960 1 1
  hb : Vec 1280
  hg : Vec 1280
  hbt : Vec 1280
  hmu : Vec 1280
  hv : Vec 1280
  Wd : Mat 1280 nCls
  bd : Vec nCls

-- ════════════════════════════════════════════════════════════════
-- § Block, stem and head forwards at inference
-- ════════════════════════════════════════════════════════════════

/-- The pre-DW slot at inference: identity at `k = 0`, depthwise → BN otherwise. -/
noncomputable def mnv4PreDWSlotEval (N h : Nat) (ε : ℝ) {c : Nat} (k : Nat)
    (p : Mnv4DWEvalSlot c k) : Vec (N * (c * h * h)) → Vec (N * (c * h * h)) :=
  if k = 0 then id
  else mnv4DWBEval N (h := h) (w := h) p.params.W p.params.b ε p.params.γ p.params.β
    p.params.μ p.params.v

/-- The stride-1 post-DW slot at inference: identity at `k = 0`, depthwise → BN → relu
    otherwise. -/
noncomputable def mnv4PostDWSlotEval (N h : Nat) (ε : ℝ) {c : Nat} (k : Nat)
    (p : Mnv4DWEvalSlot c k) : Vec (N * (c * h * h)) → Vec (N * (c * h * h)) :=
  if k = 0 then id
  else mnv4DWReluBEval N (h := h) (w := h) p.params.W p.params.b ε p.params.γ p.params.β
    p.params.μ p.params.v

/-- **A stride-1 UIB body at inference, read off its row**: pre-DW? → expand-BN-relu → post-DW? →
    project-BN, at side `h`. The skip is added by the caller. -/
noncomputable def mnv4BodyEval (N h : Nat) (ε : ℝ) (s : UibSpec) (p : UibEvalParams s) :
    Vec (N * (s.ic * h * h)) → Vec (N * (s.oc * h * h)) :=
  mnv4ProjBEval N (h := h) (w := h) p.Wz p.bz ε p.gz p.btz p.muz p.vz ∘
    mnv4PostDWSlotEval N h ε s.postDWk p.post ∘
    mnv4CbReluBEval N (h := h) (w := h) p.We p.be ε p.ge p.bte p.mue p.ve ∘
    mnv4PreDWSlotEval N h ε s.preDWk p.pre

/-- **A stride-2 UIB block at inference**: the pre-DW slot and the expand at the input side `2h`,
    the post-DW carrying the stride to `h` (timm's `dw_mid`), the project at `h`. No skip. -/
noncomputable def mnv4StridedEval (N h : Nat) (ε : ℝ) (s : UibSpec) (p : UibEvalParams s) :
    Vec (N * (s.ic * (2 * h) * (2 * h))) → Vec (N * (s.oc * h * h)) :=
  mnv4ProjBEval N (h := h) (w := h) p.Wz p.bz ε p.gz p.btz p.muz p.vz ∘
    mnv4DWReluSBEval N (h := h) (w := h) p.post.params.W p.post.params.b ε p.post.params.γ
      p.post.params.β p.post.params.μ p.post.params.v ∘
    mnv4CbReluBEval N (h := 2 * h) (w := 2 * h) p.We p.be ε p.ge p.bte p.mue p.ve ∘
    mnv4PreDWSlotEval N (2 * h) ε s.preDWk p.pre

/-- The fused stage at inference: stride-2 `k×k` conv-BN-relu, then the 1×1 project-BN. -/
noncomputable def mnv4FusedBEval (N h : Nat) (ε : ℝ) {ic mid oc kH kW : Nat}
    (Wc : Kernel4 mid ic kH kW) (bc : Vec mid) (γc βc μc vc : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (γp βp μp vp : Vec oc) :
    Vec (N * (ic * (2 * h) * (2 * h))) → Vec (N * (oc * h * h)) :=
  mnv4ProjBEval N (h := h) (w := h) Wp bp ε γp βp μp vp ∘
    mnv4CbReluSBEval N (h := h) (w := h) Wc bc ε γc βc μc vc

/-- The head at inference, timm's order: 1×1 conv-BN-relu at `h`, GAP, `conv_head` 1×1
    conv-BN-relu on the pooled `[N, mid, 1, 1]`, dense — with `mnv4Head`'s two relabellings. -/
noncomputable def mnv4HeadBEval (N h : Nat) (ε : ℝ) {c mid oc nCls : Nat}
    (W1 : Kernel4 mid c 1 1) (b1 : Vec mid) (γ1 β1 μ1 v1 : Vec mid)
    (W2 : Kernel4 oc mid 1 1) (b2 : Vec oc) (γ2 β2 μ2 v2 : Vec oc)
    (Wd : Mat oc nCls) (bd : Vec nCls) :
    Vec (N * (c * h * h)) → Vec (N * nCls) :=
  batchMap N (dense Wd bd) ∘ reindexCLM (Fin.cast (mnv4_pool11 N oc)) ∘
    mnv4CbReluBEval N (h := 1) (w := 1) W2 b2 ε γ2 β2 μ2 v2 ∘
    reindexCLM (Fin.cast (mnv4_pool11 N mid).symm) ∘
    batchMap N (globalAvgPoolFlat mid h h) ∘
    mnv4CbReluBEval N (h := h) (w := h) W1 b1 ε γ1 β1 μ1 v1

/-- **The inference MobileNetV4-Conv-M** at final feature side `f`,
    `N·(3·32f·32f) → N·nCls`: stem, fused stage, the 21 table rows (the eighteen stride-1 ones
    under `residual`), head. Every BN reads frozen statistics, so the whole net is per-example:
    each stage is `batchMap N` of a per-example map or a pointwise map. -/
noncomputable def mobilenetv4ForwardBFullEval (N f : Nat) (ε : ℝ) {nCls : Nat}
    (w : Mnv4BWeightsEval nCls)
    (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * f))))) * (2 * (2 * (2 * (2 * (2 * f)))))))) :
    Vec (N * nCls) :=
  mnv4HeadBEval N f ε w.h1W w.h1b w.h1g w.h1bt w.h1mu w.h1v
    w.hW w.hb w.hg w.hbt w.hmu w.hv w.Wd w.bd
  (residual (mnv4BodyEval N f ε mnv4Row21 w.b21)
  (residual (mnv4BodyEval N f ε mnv4Row20 w.b20)
  (residual (mnv4BodyEval N f ε mnv4Row19 w.b19)
  (residual (mnv4BodyEval N f ε mnv4Row18 w.b18)
  (residual (mnv4BodyEval N f ε mnv4Row17 w.b17)
  (residual (mnv4BodyEval N f ε mnv4Row16 w.b16)
  (residual (mnv4BodyEval N f ε mnv4Row15 w.b15)
  (residual (mnv4BodyEval N f ε mnv4Row14 w.b14)
  (residual (mnv4BodyEval N f ε mnv4Row13 w.b13)
  (residual (mnv4BodyEval N f ε mnv4Row12 w.b12)
  (mnv4StridedEval N f ε mnv4Row11 w.b11
  (residual (mnv4BodyEval N (2 * f) ε mnv4Row10 w.b10)
  (residual (mnv4BodyEval N (2 * f) ε mnv4Row9 w.b9)
  (residual (mnv4BodyEval N (2 * f) ε mnv4Row8 w.b8)
  (residual (mnv4BodyEval N (2 * f) ε mnv4Row7 w.b7)
  (residual (mnv4BodyEval N (2 * f) ε mnv4Row6 w.b6)
  (residual (mnv4BodyEval N (2 * f) ε mnv4Row5 w.b5)
  (residual (mnv4BodyEval N (2 * f) ε mnv4Row4 w.b4)
  (mnv4StridedEval N (2 * f) ε mnv4Row3 w.b3
  (residual (mnv4BodyEval N (2 * (2 * f)) ε mnv4Row2 w.b2)
  (mnv4StridedEval N (2 * (2 * f)) ε mnv4Row1 w.b1
  (mnv4FusedBEval N (2 * (2 * (2 * f))) ε w.f0cW w.f0cb w.f0cg w.f0cbt w.f0cmu w.f0cv
    w.f0pW w.f0pb w.f0pg w.f0pbt w.f0pmu w.f0pv
  (mnv4CbReluSBEval N (h := 2 * (2 * (2 * (2 * f)))) (w := 2 * (2 * (2 * (2 * f))))
    w.sW w.sb ε w.sg w.sbt w.smu w.sv x)))))))))))))))))))))))

-- ════════════════════════════════════════════════════════════════
-- § The typed inference graphs, at `mnv4FwdChainB`'s `.eval` tokens and names
-- ════════════════════════════════════════════════════════════════

/-- One inference BN node, named as `mnv4Bn` names it at `.eval`: `γ, β` by their parameter names,
    `μ, σ²` as `%{site}mu` / `%{site}var`. -/
def mnv4BnGraphBEval (epsStr gName btName site : String) (N c h : Nat) (ε : ℝ)
    (γ β μ v : Vec c) (e : SHlo (N * (c * h * h))) : SHlo (N * (c * h * h)) :=
  .batchOp (N := N) (.bnEval (h := h) (w := h) gName btName s!"%{site}mu" s!"%{site}var"
    epsStr ε γ β μ v) e

/-- The pre-DW slot's graph: no tokens at `k = 0`, depthwise → BN otherwise — `uibFwdSkipB`'s
    and `uibFwdStridedB`'s `if preDWk > 0`. -/
def mnv4PreDWGraphBEval (epsStr p : String) (N h : Nat) (ε : ℝ) {c : Nat} (k : Nat)
    (q : Mnv4DWEvalSlot c k) (e : SHlo (N * (c * h * h))) : SHlo (N * (c * h * h)) :=
  if k = 0 then e
  else mnv4BnGraphBEval epsStr s!"%u{p}qg" s!"%u{p}qbt" s!"u{p}qn" N c h ε
    q.params.γ q.params.β q.params.μ q.params.v
    (.batchOp (N := N) (.depthwise (h := h) (w := h) s!"%u{p}qW" s!"%zb{c}" q.params.W q.params.b) e)

/-- The stride-1 post-DW slot's graph: no tokens at `k = 0`, depthwise → BN → relu otherwise. -/
def mnv4PostDWGraphBEval (epsStr p : String) (N h : Nat) (ε : ℝ) {c : Nat} (k : Nat)
    (q : Mnv4DWEvalSlot c k) (e : SHlo (N * (c * h * h))) : SHlo (N * (c * h * h)) :=
  if k = 0 then e
  else .batchOp (N := N) (.relu (n := c * h * h))
    (mnv4BnGraphBEval epsStr s!"%u{p}dg" s!"%u{p}dbt" s!"u{p}dn" N c h ε
      q.params.γ q.params.β q.params.μ q.params.v
      (.batchOp (N := N) (.depthwise (h := h) (w := h) s!"%u{p}dW" s!"%zb{c}" q.params.W q.params.b) e))

/-- **A stride-1 UIB body's inference graph** at side `h`, every name read off the row. -/
def mnv4BodyGraphBEval (epsStr : String) (N h : Nat) (ε : ℝ) (s : UibSpec) (p : UibEvalParams s)
    (e : SHlo (N * (s.ic * h * h))) : SHlo (N * (s.oc * h * h)) :=
  mnv4BnGraphBEval epsStr s!"%u{s.p}pg" s!"%u{s.p}pbt" s!"u{s.p}pn" N s.oc h ε p.gz p.btz p.muz p.vz
    (.batchOp (N := N) (.conv (ic := s.ic * s.expand) (oc := s.oc) (h := h) (w := h)
        s!"%u{s.p}pW" s!"%zb{s.oc}" p.Wz p.bz)
      (mnv4PostDWGraphBEval epsStr s.p N h ε s.postDWk p.post
        (.batchOp (N := N) (.relu (n := s.ic * s.expand * h * h))
          (mnv4BnGraphBEval epsStr s!"%u{s.p}eg" s!"%u{s.p}ebt" s!"u{s.p}en" N (s.ic * s.expand) h ε
              p.ge p.bte p.mue p.ve
            (.batchOp (N := N) (.conv (ic := s.ic) (oc := s.ic * s.expand) (h := h) (w := h)
                s!"%u{s.p}eW" s!"%zb{s.ic * s.expand}" p.We p.be)
              (mnv4PreDWGraphBEval epsStr s.p N h ε s.preDWk p.pre e))))))

/-- **A stride-2 UIB block's inference graph**: pre-DW slot and expand at `2h`, the strided
    post-DW (`.depthwiseStrided`, symmetric) to `h`, project at `h`. -/
def mnv4StridedGraphBEval (epsStr : String) (N h : Nat) (ε : ℝ) (s : UibSpec)
    (p : UibEvalParams s) (e : SHlo (N * (s.ic * (2 * h) * (2 * h)))) :
    SHlo (N * (s.oc * h * h)) :=
  mnv4BnGraphBEval epsStr s!"%u{s.p}pg" s!"%u{s.p}pbt" s!"u{s.p}pn" N s.oc h ε p.gz p.btz p.muz p.vz
    (.batchOp (N := N) (.conv (ic := s.ic * s.expand) (oc := s.oc) (h := h) (w := h)
        s!"%u{s.p}pW" s!"%zb{s.oc}" p.Wz p.bz)
      (.batchOp (N := N) (.relu (n := s.ic * s.expand * h * h))
        (mnv4BnGraphBEval epsStr s!"%u{s.p}dg" s!"%u{s.p}dbt" s!"u{s.p}dn" N (s.ic * s.expand) h ε
            p.post.params.γ p.post.params.β p.post.params.μ p.post.params.v
          (.batchOp (N := N) (.depthwiseStrided (c := s.ic * s.expand) (h := h) (w := h)
              s!"%u{s.p}dW" s!"%zb{s.ic * s.expand}" p.post.params.W p.post.params.b)
            (.batchOp (N := N) (.relu (n := s.ic * s.expand * (2 * h) * (2 * h)))
              (mnv4BnGraphBEval epsStr s!"%u{s.p}eg" s!"%u{s.p}ebt" s!"u{s.p}en" N (s.ic * s.expand)
                  (2 * h) ε p.ge p.bte p.mue p.ve
                (.batchOp (N := N) (.conv (ic := s.ic) (oc := s.ic * s.expand) (h := 2 * h)
                    (w := 2 * h) s!"%u{s.p}eW" s!"%zb{s.ic * s.expand}" p.We p.be)
                  (mnv4PreDWGraphBEval epsStr s.p N (2 * h) ε s.preDWk p.pre e))))))))

/-- Stem inference graph: 3×3/s2 symmetric conv → BN (`stn`) → relu. Generic in the widths, for
    the reason `mnv4StemGraphB` records. -/
def mnv4StemGraphBEval (epsStr : String) (N h : Nat) (ε : ℝ) {ic oc kH kW : Nat}
    (Ws : Kernel4 oc ic kH kW) (bs : Vec oc) (γs βs μs vs : Vec oc)
    (e : SHlo (N * (ic * (2 * h) * (2 * h)))) : SHlo (N * (oc * h * h)) :=
  .batchOp (N := N) (.relu (n := oc * h * h))
    (mnv4BnGraphBEval epsStr "%sg" "%sbt" "stn" N oc h ε γs βs μs vs
      (.batchOp (N := N) (.convStrided (h := h) (w := h) "%sW" s!"%zb{oc}" Ws bs) e))

/-- Fused-stage inference graph: 3×3/s2 conv → BN (`f0cn`) → relu → 1×1 project → BN (`f0pn`). -/
def mnv4FusedGraphBEval (epsStr : String) (N h : Nat) (ε : ℝ) {ic mid oc kH kW : Nat}
    (Wc : Kernel4 mid ic kH kW) (bc : Vec mid) (γc βc μc vc : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (γp βp μp vp : Vec oc)
    (e : SHlo (N * (ic * (2 * h) * (2 * h)))) : SHlo (N * (oc * h * h)) :=
  mnv4BnGraphBEval epsStr "%f0pg" "%f0pbt" "f0pn" N oc h ε γp βp μp vp
    (.batchOp (N := N) (.conv (h := h) (w := h) "%f0pW" s!"%zb{oc}" Wp bp)
      (.batchOp (N := N) (.relu (n := mid * h * h))
        (mnv4BnGraphBEval epsStr "%f0cg" "%f0cbt" "f0cn" N mid h ε γc βc μc vc
          (.batchOp (N := N) (.convStrided (h := h) (w := h) "%f0cW" s!"%zb{mid}" Wc bc) e))))

/-- Head inference graph, `mnv4HeadGraphB`'s tokens with the two BNs at frozen statistics (`h1n`,
    `hn`). -/
def mnv4HeadGraphBEval (epsStr : String) (N h : Nat) (ε : ℝ) {c mid oc nCls : Nat}
    (W1 : Kernel4 mid c 1 1) (b1 : Vec mid) (γ1 β1 μ1 v1 : Vec mid)
    (W2 : Kernel4 oc mid 1 1) (b2 : Vec oc) (γ2 β2 μ2 v2 : Vec oc)
    (Wd : Mat oc nCls) (bd : Vec nCls)
    (e : SHlo (N * (c * h * h))) : SHlo (N * nCls) :=
  .batchOp (N := N) (.dense "%Wd" "%bd" Wd bd)
    (castIdx (mnv4_pool11 N oc).symm
      (.batchOp (N := N) (.relu (n := oc * 1 * 1))
        (mnv4BnGraphBEval epsStr "%hg" "%hbt" "hn" N oc 1 ε γ2 β2 μ2 v2
          (.batchOp (N := N) (.conv (h := 1) (w := 1) "%hW" s!"%zb{oc}" W2 b2)
            (castIdx (mnv4_pool11 N mid)
              (.batchOp (N := N) (.gap (c := mid) (h := h) (w := h))
                (.batchOp (N := N) (.relu (n := mid * h * h))
                  (mnv4BnGraphBEval epsStr "%h1g" "%h1bt" "h1n" N mid h ε γ1 β1 μ1 v1
                    (.batchOp (N := N) (.conv (h := h) (w := h) "%h1W" s!"%zb{mid}" W1 b1)
                      e)))))))))

-- ════════════════════════════════════════════════════════════════
-- § Faithfulness, block by block
-- ════════════════════════════════════════════════════════════════

private theorem mnv4BnGraphBEval_faithful (epsStr gName btName site : String) (N c h : Nat)
    (ε : ℝ) (γ β μ v : Vec c) (e : SHlo (N * (c * h * h))) :
    den (mnv4BnGraphBEval epsStr gName btName site N c h ε γ β μ v e)
      = batchMap N (bnPerChannelEvalTensor3 c h h ε γ β μ v) (den e) := by
  simp only [mnv4BnGraphBEval, den_batchOp, denOp]

private theorem mnv4PreDWGraphBEval_faithful (epsStr p : String) (N h : Nat) (ε : ℝ) {c : Nat}
    (k : Nat) (q : Mnv4DWEvalSlot c k) (e : SHlo (N * (c * h * h))) :
    den (mnv4PreDWGraphBEval epsStr p N h ε k q e) = mnv4PreDWSlotEval N h ε k q (den e) := by
  unfold mnv4PreDWGraphBEval mnv4PreDWSlotEval
  split_ifs
  · rfl
  · simp only [mnv4BnGraphBEval_faithful, mnv4DWBEval, den_batchOp, denOp, Function.comp_apply]

private theorem mnv4PostDWGraphBEval_faithful (epsStr p : String) (N h : Nat) (ε : ℝ) {c : Nat}
    (k : Nat) (q : Mnv4DWEvalSlot c k) (e : SHlo (N * (c * h * h))) :
    den (mnv4PostDWGraphBEval epsStr p N h ε k q e) = mnv4PostDWSlotEval N h ε k q (den e) := by
  unfold mnv4PostDWGraphBEval mnv4PostDWSlotEval
  split_ifs
  · rfl
  · simp only [mnv4BnGraphBEval_faithful,
      mnv4DWReluBEval, den_batchOp, denOp, Function.comp_apply]

/-- **The stride-1 UIB body graph denotes its inference forward — every family, one theorem.** -/
theorem mnv4BodyGraphBEval_faithful (epsStr : String) (N h : Nat) (ε : ℝ) (s : UibSpec)
    (p : UibEvalParams s) (e : SHlo (N * (s.ic * h * h))) :
    den (mnv4BodyGraphBEval epsStr N h ε s p e) = mnv4BodyEval N h ε s p (den e) := by
  simp only [mnv4BodyGraphBEval, mnv4BodyEval, mnv4BnGraphBEval_faithful,
    mnv4PostDWGraphBEval_faithful, mnv4PreDWGraphBEval_faithful, mnv4ProjBEval, mnv4CbReluBEval, den_batchOp, denOp, Function.comp_apply]

/-- The stride-2 UIB block graph denotes its inference forward. -/
theorem mnv4StridedGraphBEval_faithful (epsStr : String) (N h : Nat) (ε : ℝ) (s : UibSpec)
    (p : UibEvalParams s) (e : SHlo (N * (s.ic * (2 * h) * (2 * h)))) :
    den (mnv4StridedGraphBEval epsStr N h ε s p e) = mnv4StridedEval N h ε s p (den e) := by
  simp only [mnv4StridedGraphBEval, mnv4StridedEval, mnv4BnGraphBEval_faithful,
    mnv4PreDWGraphBEval_faithful, mnv4ProjBEval,
    mnv4DWReluSBEval, mnv4CbReluBEval, den_batchOp, denOp, Function.comp_apply]

/-- A skip row: the body's inference forward under `residual`. Generic in the body, applied at
    each of the eighteen rows (where `s.ic` and `s.oc` are the same literal). -/
private theorem mnv4SkipGraphBEval_faithful {N n : Nat} (body : SHlo (N * n) → SHlo (N * n))
    (g : Vec (N * n) → Vec (N * n))
    (hb : ∀ e' : SHlo (N * n), den (body e') = g (den e')) (e : SHlo (N * n)) :
    den (mnv4SkipGraphB body e) = residual g (den e) := by
  simp only [mnv4SkipGraphB, den_addVB, hb, residual]

private theorem mnv4StemGraphBEval_faithful (epsStr : String) (N h : Nat) (ε : ℝ)
    {ic oc kH kW : Nat} (Ws : Kernel4 oc ic kH kW) (bs : Vec oc) (γs βs μs vs : Vec oc)
    (e : SHlo (N * (ic * (2 * h) * (2 * h)))) :
    den (mnv4StemGraphBEval epsStr N h ε Ws bs γs βs μs vs e)
      = mnv4CbReluSBEval N (h := h) (w := h) Ws bs ε γs βs μs vs (den e) := by
  simp only [mnv4StemGraphBEval, mnv4CbReluSBEval, mnv4BnGraphBEval_faithful, den_batchOp, denOp, Function.comp_apply]

private theorem mnv4FusedGraphBEval_faithful (epsStr : String) (N h : Nat) (ε : ℝ)
    {ic mid oc kH kW : Nat} (Wc : Kernel4 mid ic kH kW) (bc : Vec mid) (γc βc μc vc : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (γp βp μp vp : Vec oc)
    (e : SHlo (N * (ic * (2 * h) * (2 * h)))) :
    den (mnv4FusedGraphBEval epsStr N h ε Wc bc γc βc μc vc Wp bp γp βp μp vp e)
      = mnv4FusedBEval N h ε Wc bc γc βc μc vc Wp bp γp βp μp vp (den e) := by
  simp only [mnv4FusedGraphBEval, mnv4FusedBEval, mnv4ProjBEval, mnv4CbReluSBEval,
    mnv4BnGraphBEval_faithful, den_batchOp, denOp,
    Function.comp_apply]

private theorem mnv4HeadGraphBEval_faithful (epsStr : String) (N h : Nat) (ε : ℝ)
    {c mid oc nCls : Nat} (W1 : Kernel4 mid c 1 1) (b1 : Vec mid) (γ1 β1 μ1 v1 : Vec mid)
    (W2 : Kernel4 oc mid 1 1) (b2 : Vec oc) (γ2 β2 μ2 v2 : Vec oc)
    (Wd : Mat oc nCls) (bd : Vec nCls) (e : SHlo (N * (c * h * h))) :
    den (mnv4HeadGraphBEval epsStr N h ε W1 b1 γ1 β1 μ1 v1 W2 b2 γ2 β2 μ2 v2 Wd bd e)
      = mnv4HeadBEval N h ε W1 b1 γ1 β1 μ1 v1 W2 b2 γ2 β2 μ2 v2 Wd bd (den e) := by
  simp only [mnv4HeadGraphBEval, mnv4HeadBEval, mnv4CbReluBEval, mnv4BnGraphBEval_faithful, den_castIdx, reindexCLM_apply, den_batchOp, denOp,
    Function.comp_apply]

-- ════════════════════════════════════════════════════════════════
-- § The whole inference graph + faithfulness
-- ════════════════════════════════════════════════════════════════

/-- **The MobileNetV4-Conv-M inference forward graph** at final feature side `f`, at
    `mnv4FwdChainB`'s `.eval` tokens and names: `mnv4_fwd_eval.mlir` and `mnv4in_fwd_eval.mlir`
    at `f = 7`, `mnv4in_fwd_eval_s256.mlir` at `f = 8`. -/
def mnv4FwdGraphBFullEval (N f : Nat) (epsStr : String) (ε : ℝ) {nCls : Nat}
    (w : Mnv4BWeightsEval nCls)
    (e : SHlo (N * (3 * (2 * (2 * (2 * (2 * (2 * f))))) * (2 * (2 * (2 * (2 * (2 * f)))))))) :
    SHlo (N * nCls) :=
  mnv4HeadGraphBEval epsStr N f ε w.h1W w.h1b w.h1g w.h1bt w.h1mu w.h1v
    w.hW w.hb w.hg w.hbt w.hmu w.hv w.Wd w.bd
  (mnv4SkipGraphB (mnv4BodyGraphBEval epsStr N f ε mnv4Row21 w.b21)
  (mnv4SkipGraphB (mnv4BodyGraphBEval epsStr N f ε mnv4Row20 w.b20)
  (mnv4SkipGraphB (mnv4BodyGraphBEval epsStr N f ε mnv4Row19 w.b19)
  (mnv4SkipGraphB (mnv4BodyGraphBEval epsStr N f ε mnv4Row18 w.b18)
  (mnv4SkipGraphB (mnv4BodyGraphBEval epsStr N f ε mnv4Row17 w.b17)
  (mnv4SkipGraphB (mnv4BodyGraphBEval epsStr N f ε mnv4Row16 w.b16)
  (mnv4SkipGraphB (mnv4BodyGraphBEval epsStr N f ε mnv4Row15 w.b15)
  (mnv4SkipGraphB (mnv4BodyGraphBEval epsStr N f ε mnv4Row14 w.b14)
  (mnv4SkipGraphB (mnv4BodyGraphBEval epsStr N f ε mnv4Row13 w.b13)
  (mnv4SkipGraphB (mnv4BodyGraphBEval epsStr N f ε mnv4Row12 w.b12)
  (mnv4StridedGraphBEval epsStr N f ε mnv4Row11 w.b11
  (mnv4SkipGraphB (mnv4BodyGraphBEval epsStr N (2 * f) ε mnv4Row10 w.b10)
  (mnv4SkipGraphB (mnv4BodyGraphBEval epsStr N (2 * f) ε mnv4Row9 w.b9)
  (mnv4SkipGraphB (mnv4BodyGraphBEval epsStr N (2 * f) ε mnv4Row8 w.b8)
  (mnv4SkipGraphB (mnv4BodyGraphBEval epsStr N (2 * f) ε mnv4Row7 w.b7)
  (mnv4SkipGraphB (mnv4BodyGraphBEval epsStr N (2 * f) ε mnv4Row6 w.b6)
  (mnv4SkipGraphB (mnv4BodyGraphBEval epsStr N (2 * f) ε mnv4Row5 w.b5)
  (mnv4SkipGraphB (mnv4BodyGraphBEval epsStr N (2 * f) ε mnv4Row4 w.b4)
  (mnv4StridedGraphBEval epsStr N (2 * f) ε mnv4Row3 w.b3
  (mnv4SkipGraphB (mnv4BodyGraphBEval epsStr N (2 * (2 * f)) ε mnv4Row2 w.b2)
  (mnv4StridedGraphBEval epsStr N (2 * (2 * f)) ε mnv4Row1 w.b1
  (mnv4FusedGraphBEval epsStr N (2 * (2 * (2 * f))) ε w.f0cW w.f0cb w.f0cg w.f0cbt w.f0cmu w.f0cv
    w.f0pW w.f0pb w.f0pg w.f0pbt w.f0pmu w.f0pv
  (mnv4StemGraphBEval epsStr N (2 * (2 * (2 * (2 * f)))) ε
    w.sW w.sb w.sg w.sbt w.smu w.sv e)))))))))))))))))))))))

/-- **The MobileNetV4-Conv-M inference graph denotes the inference forward**, at every final
    feature side `f`. One rewrite per stage, outside-in; the eighteen skip rows each go through
    `mnv4SkipGraphBEval_faithful` with the body theorem, so the term stays linear in the depth. -/
theorem mnv4FwdGraphBFullEval_faithful (N f : Nat) (epsStr : String) (ε : ℝ) {nCls : Nat}
    (w : Mnv4BWeightsEval nCls)
    (e : SHlo (N * (3 * (2 * (2 * (2 * (2 * (2 * f))))) * (2 * (2 * (2 * (2 * (2 * f)))))))) :
    den (mnv4FwdGraphBFullEval N f epsStr ε w e) = mobilenetv4ForwardBFullEval N f ε w (den e) := by
  unfold mnv4FwdGraphBFullEval mobilenetv4ForwardBFullEval
  rw [mnv4HeadGraphBEval_faithful,
    mnv4SkipGraphBEval_faithful _ _ (mnv4BodyGraphBEval_faithful epsStr N f ε mnv4Row21 w.b21),
    mnv4SkipGraphBEval_faithful _ _ (mnv4BodyGraphBEval_faithful epsStr N f ε mnv4Row20 w.b20),
    mnv4SkipGraphBEval_faithful _ _ (mnv4BodyGraphBEval_faithful epsStr N f ε mnv4Row19 w.b19),
    mnv4SkipGraphBEval_faithful _ _ (mnv4BodyGraphBEval_faithful epsStr N f ε mnv4Row18 w.b18),
    mnv4SkipGraphBEval_faithful _ _ (mnv4BodyGraphBEval_faithful epsStr N f ε mnv4Row17 w.b17),
    mnv4SkipGraphBEval_faithful _ _ (mnv4BodyGraphBEval_faithful epsStr N f ε mnv4Row16 w.b16),
    mnv4SkipGraphBEval_faithful _ _ (mnv4BodyGraphBEval_faithful epsStr N f ε mnv4Row15 w.b15),
    mnv4SkipGraphBEval_faithful _ _ (mnv4BodyGraphBEval_faithful epsStr N f ε mnv4Row14 w.b14),
    mnv4SkipGraphBEval_faithful _ _ (mnv4BodyGraphBEval_faithful epsStr N f ε mnv4Row13 w.b13),
    mnv4SkipGraphBEval_faithful _ _ (mnv4BodyGraphBEval_faithful epsStr N f ε mnv4Row12 w.b12),
    mnv4StridedGraphBEval_faithful,
    mnv4SkipGraphBEval_faithful _ _ (mnv4BodyGraphBEval_faithful epsStr N (2 * f) ε mnv4Row10 w.b10),
    mnv4SkipGraphBEval_faithful _ _ (mnv4BodyGraphBEval_faithful epsStr N (2 * f) ε mnv4Row9 w.b9),
    mnv4SkipGraphBEval_faithful _ _ (mnv4BodyGraphBEval_faithful epsStr N (2 * f) ε mnv4Row8 w.b8),
    mnv4SkipGraphBEval_faithful _ _ (mnv4BodyGraphBEval_faithful epsStr N (2 * f) ε mnv4Row7 w.b7),
    mnv4SkipGraphBEval_faithful _ _ (mnv4BodyGraphBEval_faithful epsStr N (2 * f) ε mnv4Row6 w.b6),
    mnv4SkipGraphBEval_faithful _ _ (mnv4BodyGraphBEval_faithful epsStr N (2 * f) ε mnv4Row5 w.b5),
    mnv4SkipGraphBEval_faithful _ _ (mnv4BodyGraphBEval_faithful epsStr N (2 * f) ε mnv4Row4 w.b4),
    mnv4StridedGraphBEval_faithful,
    mnv4SkipGraphBEval_faithful _ _
      (mnv4BodyGraphBEval_faithful epsStr N (2 * (2 * f)) ε mnv4Row2 w.b2),
    mnv4StridedGraphBEval_faithful, mnv4FusedGraphBEval_faithful, mnv4StemGraphBEval_faithful]

end Proofs.StableHLO

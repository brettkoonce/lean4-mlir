import LeanMlir.Proofs.Foundation.CertifiedChain
import LeanMlir.Proofs.Nets.ViT.ViTBackB0

/-! # ViT on `CertLayer`, folded image → logits

ViT's whole-net backward graph `vitNetBackGraph` (patchEmbed → tower → final vec-LN →
classifier, at every depth) is derived here from the generic `CertifiedChain` fold:
`vitTrunkV_graph` proves the `CertLayer` chain produces the hand-written `vitBodyBackGraphKMHV`
term for term, and `vitTrunkV_fwd` proves its forward is the shipped `vitBodyKVFlat`. So the
bespoke tower induction in `ViTBackB0.lean` is an instance of the generic one.

## Pluggable blocks

`ViTBackB0.lean` threads an incoming-cotangent subgraph `ecot : SHlo n` through the vec-LN
production chain (`transformerMlpBackGraph` → `mlpSublayerV*` → `attnSublayerV*` →
`transformerBlockV*` → the tower), each faithfulness statement carrying `den ecot = Mat.flatten dz`;
every statement at a fixed `Vec` cotangent is the one at `ecot := .operand "%d…" (Mat.flatten dz)`.
The two sublayers of a block compose as subgraphs, and so do successive blocks in the tower.
Note: `attnSublayerVInnerBackGraphMH` still passes `den e` into `mhsaBackGraphMH`, because that
MHSA graph takes a `Vec` cotangent. It is inside the attention arm, not between blocks, so it does
not block composition.

## The fold runs image → logits

ViT's stem is an affine patchify conv and its head is a CLS slice plus a dense — both linear, so
both backward graphs are activation-independent. So `vitNetLayer = stem ∘ trunk ∘ finalLN ∘ head`
is one `CertLayer`, assembled by `comp` alone, and `vitNetBackGraph_faithful` (the whole-net
capstone) follows from it — including that the fold's VJP is the shipped `vitForwardKVHasVJP`,
not merely another VJP of the same map. That last step is `HasVJPAt.backward_unique_of_eq` along
the forward equation `vitNetLayer_fwd`.

## `ok = True`

GELU and LayerNorm are smooth everywhere, so a ViT block has a global `HasVJPMat`, lifted
pointwise by `.toHasVJPAt`, and every layer's precondition is `True`. EfficientNet (swish) and
ConvNeXt (GELU) are the same; ResNet-34/50 (ReLU), MobileNetV2 (ReLU6) and MobileNetV4 (ReLU)
carry `_at` hypotheses because those activations have no derivative at their kinks.

**Scope.** This is a certified composition of backward graphs; it is not a proof that
`verified_mlir/vit_train_step.mlir` is this graph.
-/

open Proofs Proofs.StableHLO

namespace Proofs.StableHLO

-- ════════════════════════════════════════════════════════════════
-- § The ViT block as a `CertLayer`
-- ════════════════════════════════════════════════════════════════

/-- **A vec-LN multi-head transformer block as a `CertLayer`**, at the flat index.

    The bridge from ViT's per-token `Mat` world to `CertLayer`'s `Vec → Vec` one is
    `HasVJPMat.toHasVJP`, whose statement `HasVJP (fun v => Mat.flatten (f (Mat.unflatten v)))`
    is *definitionally* `blockVFlat` — so the lift costs nothing. The backward graph is the
    committed `transformerBlockVBackGraphMHP` at the unflattened saved activation.

    `ok = True`: GELU and LayerNorm are smooth, so the graph denotes the VJP at **every**
    input, with no side condition to discharge. -/
noncomputable def vitBlockVLayer {Np1 hm1 d mlpDim : Nat} (ε : ℝ) (hε : 0 < ε)
    (p : BlockParamsV ((hm1+1) * d) mlpDim) :
    CertLayer (Np1 * ((hm1+1) * d)) (Np1 * ((hm1+1) * d)) where
  fwd := blockVFlat Np1 (hm1+1) d mlpDim ε p
  ok := fun _ => True
  diff := fun x _ =>
    (transformerBlockV_flat_differentiable Np1 (hm1+1) d mlpDim ε p.γ1 p.β1 hε
      p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo p.γ2 p.β2 p.Wfc1 p.bfc1 p.Wfc2 p.bfc2) x
  vjp := fun x _ => (HasVJPMat.toHasVJP (transformerBlockVHasVJPMatP ε hε p)).toHasVJPAt x
  graph := fun v e => transformerBlockVBackGraphMHP ε p (Mat.unflatten v) e
  faithful := by
    intro v _ e
    -- The block capstone at the unflattened cotangent; its den-hypothesis is the
    -- `flatten ∘ unflatten` round-trip, which is a LEMMA (not `rfl`).
    rw [transformerBlockVBackGraphMHP_faithful ε hε p (Mat.unflatten v)
          (Mat.unflatten (den e)) e (Mat.flatten_unflatten (den e)).symm]
    rfl

-- ════════════════════════════════════════════════════════════════
-- § The depth-`k` trunk — `comp`, and nothing else
-- ════════════════════════════════════════════════════════════════

/-- **The depth-`k` ViT trunk as one `CertLayer`.** Written in the shipped `Fin k → BlockParamsV`
    shape and mirroring `vitBodyKVFlat`'s recursion (block `0` runs first), so the two can be
    compared term for term below. The only content is `CertLayer.comp`, which is already proven —
    depth costs nothing. -/
noncomputable def vitTrunkV {Np1 hm1 d mlpDim : Nat} (ε : ℝ) (hε : 0 < ε) :
    (k : Nat) → (Fin k → BlockParamsV ((hm1+1) * d) mlpDim) →
    CertLayer (Np1 * ((hm1+1) * d)) (Np1 * ((hm1+1) * d))
  | 0, _ => CertLayer.id' _
  | k + 1, ps =>
      (vitBlockVLayer (Np1 := Np1) ε hε (ps 0)).comp (vitTrunkV ε hε k (fun i => ps i.succ))

/-- **The trunk's forward is the shipped depth-`k` body.** Without this the fold would be a
    chain of blocks that merely resembles ViT's; with it, `vitTrunkV` is `vitBodyKVFlat`. -/
theorem vitTrunkV_fwd {Np1 hm1 d mlpDim : Nat} (ε : ℝ) (hε : 0 < ε) :
    ∀ (k : Nat) (ps : Fin k → BlockParamsV ((hm1+1) * d) mlpDim)
      (v : Vec (Np1 * ((hm1+1) * d))),
      (vitTrunkV (Np1 := Np1) ε hε k ps).fwd v
        = vitBodyKVFlat Np1 (hm1+1) d mlpDim ε k ps v
  | 0, _, _ => rfl
  | k + 1, ps, v => by
      show (vitTrunkV (Np1 := Np1) ε hε k (fun i => ps i.succ)).fwd
            (blockVFlat Np1 (hm1+1) d mlpDim ε (ps 0) v) = _
      exact vitTrunkV_fwd ε hε k (fun i => ps i.succ)
        (blockVFlat Np1 (hm1+1) d mlpDim ε (ps 0) v)

/-- **The generic fold reproduces the hand-written tower, term for term.**

    `vitBodyBackGraphKMHV` is `ViTBackB0`'s bespoke depth-`k` reverse fold, proven faithful there
    by an induction on `k` that re-does the chain-rule argument at every depth. This theorem says
    the `CertLayer` chain's graph — built by `comp`, whose faithfulness was proven ONCE and for
    all nets — is that same term. So the bespoke induction is not a second, parallel artifact to
    keep in sync; it is an instance.

    The proof is the two round-trips `unflatten (flatten A) = A` (the saved activation the block
    graph differentiates at) and `blockVFlat (flatten A) = flatten (blockV A)` (the activation the
    TAIL differentiates at) — which is exactly the fact `CertLayer.comp` encodes and the endo slip
    would break. -/
theorem vitTrunkV_graph {Np1 hm1 d mlpDim : Nat} (ε : ℝ) (hε : 0 < ε) :
    ∀ (k : Nat) (ps : Fin k → BlockParamsV ((hm1+1) * d) mlpDim)
      (A : Mat Np1 ((hm1+1) * d)) (e : SHlo (Np1 * ((hm1+1) * d))),
      (vitTrunkV (Np1 := Np1) ε hε k ps).graph (Mat.flatten A) e
        = vitBodyBackGraphKMHV ε k ps A e
  | 0, _, _, _ => rfl
  | k + 1, ps, A, e => by
      have hb : blockVFlat Np1 (hm1+1) d mlpDim ε (ps 0) (Mat.flatten A)
          = Mat.flatten (blockV Np1 (hm1+1) d mlpDim ε (ps 0) A) := by
        unfold blockVFlat; rw [Mat.unflatten_flatten]
      show transformerBlockVBackGraphMHP ε (ps 0) (Mat.unflatten (Mat.flatten A))
            ((vitTrunkV (Np1 := Np1) ε hε k (fun i => ps i.succ)).graph
              (blockVFlat Np1 (hm1+1) d mlpDim ε (ps 0) (Mat.flatten A)) e) = _
      rw [Mat.unflatten_flatten, hb,
        vitTrunkV_graph ε hε k (fun i => ps i.succ)
          (blockV Np1 (hm1+1) d mlpDim ε (ps 0) A) e]
      rfl

/-- **A ViT trunk's `ok` is `True` at every depth.** `CertLayer.comp` conjoins preconditions, so
    for a relu net this is a stack of side conditions growing with depth; for ViT the conjunction
    collapses and the whole depth-`k` trunk is certified at every input. -/
theorem vitTrunkV_ok {Np1 hm1 d mlpDim : Nat} (ε : ℝ) (hε : 0 < ε) :
    ∀ (k : Nat) (ps : Fin k → BlockParamsV ((hm1+1) * d) mlpDim)
      (v : Vec (Np1 * ((hm1+1) * d))), (vitTrunkV (Np1 := Np1) ε hε k ps).ok v
  | 0, _, _ => trivial
  | k + 1, ps, _v => ⟨trivial, vitTrunkV_ok ε hε k (fun i => ps i.succ) _⟩

-- ════════════════════════════════════════════════════════════════
-- § ⭐⭐ THE STEM AND THE HEAD — so the fold runs IMAGE → LOGITS
-- ════════════════════════════════════════════════════════════════

/-! ViT's stem is an affine patchify conv and its head is GAP-free (a CLS slice + dense), so both
are linear and their backward graphs are activation-independent. -/

/-- **The patch-embedding stem as a `CertLayer`.** Its `graph` ignores the saved activation
    entirely — patchEmbed is affine, so the input-VJP is the same linear map everywhere. -/
noncomputable def vitPatchEmbedLayer (ic H W patchSize N D : Nat)
    (Wc : Kernel4 D ic patchSize patchSize) (bc cls : Vec D) (pos : Mat (N + 1) D) :
    CertLayer (ic * H * W) ((N + 1) * D) where
  fwd := patchEmbedFlat ic H W patchSize N D Wc bc cls pos
  ok := fun _ => True
  diff := fun x _ => (patchEmbedFlat_differentiable ic H W patchSize N D Wc bc cls pos) x
  vjp := fun x _ => (patchEmbedFlatHasVJP ic H W patchSize N D Wc bc cls pos).toHasVJPAt x
  graph := fun _ e => patchEmbedBackGraph ic H W patchSize N D Wc e
  faithful := fun x _ e =>
    patchEmbedBackGraph_faithful ic H W patchSize N D Wc bc cls pos x e

/-- **The final (pre-head) vector-LN as a `CertLayer`.** Smooth, so `ok = True`. -/
noncomputable def vitFinalLNLayer (N D : Nat) (ε : ℝ) (γF βF : Vec D) (hε : 0 < ε) :
    CertLayer ((N + 1) * D) ((N + 1) * D) where
  fwd := fun v => Mat.flatten (fun n => layerNormVec D ε γF βF ((Mat.unflatten v) n))
  ok := fun _ => True
  diff := fun x _ => (layerNormVec_per_token_flat_differentiable (N + 1) D ε γF βF hε) x
  vjp := fun x _ =>
    (HasVJPMat.toHasVJP (layerNormVecPerTokenHasVJPMat (N + 1) D ε γF βF hε)).toHasVJPAt x
  graph := fun v e => finalLNBackGraph N D ε γF v e
  faithful := by
    intro v _ e
    have h := finalLNBackGraph_faithful N D ε γF βF hε (Mat.unflatten v) e
    rw [Mat.flatten_unflatten] at h
    exact h

/-- **The classifier head as a `CertLayer`** — CLS-slice then dense, both linear, so the graph
    ignores its activation for the same reason the stem's does. -/
noncomputable def vitClassifierLayer (N D nClasses : Nat)
    (Wcls : Mat D nClasses) (bcls : Vec nClasses) :
    CertLayer ((N + 1) * D) nClasses where
  fwd := classifierFlat N D nClasses Wcls bcls
  ok := fun _ => True
  diff := fun x _ => (classifierFlat_differentiable N D nClasses Wcls bcls) x
  vjp := fun x _ => (classifierFlatHasVJP N D nClasses Wcls bcls).toHasVJPAt x
  graph := fun _ e => classifierBackGraph N D nClasses Wcls e
  faithful := fun v _ e => classifierBackGraph_faithful N D nClasses Wcls bcls v e

/-- **The whole net as one `CertLayer`** — stem, depth-`k` trunk, final LN, head, composed
    by `comp` alone. Image in, logits out, and the backward graph and its faithfulness come with
    it. -/
noncomputable def vitNetLayer (ic H W patchSize N mlpDim hm1 d nClasses k : Nat)
    (ε : ℝ) (hε : 0 < ε)
    (Wc : Kernel4 ((hm1+1) * d) ic patchSize patchSize) (bc cls : Vec ((hm1+1) * d))
    (pos : Mat (N + 1) ((hm1+1) * d))
    (ps : Fin k → BlockParamsV ((hm1+1) * d) mlpDim)
    (γF βF : Vec ((hm1+1) * d))
    (Wcls : Mat ((hm1+1) * d) nClasses) (bcls : Vec nClasses) :
    CertLayer (ic * H * W) nClasses :=
  (vitPatchEmbedLayer ic H W patchSize N ((hm1+1) * d) Wc bc cls pos).comp
    ((vitTrunkV (Np1 := N + 1) ε hε k ps).comp
      ((vitFinalLNLayer N ((hm1+1) * d) ε γF βF hε).comp
        (vitClassifierLayer N ((hm1+1) * d) nClasses Wcls bcls)))

/-- The whole-net layer's forward **is** the shipped `vitForwardKV`. -/
theorem vitNetLayer_fwd (ic H W patchSize N mlpDim hm1 d nClasses k : Nat)
    (ε : ℝ) (hε : 0 < ε)
    (Wc : Kernel4 ((hm1+1) * d) ic patchSize patchSize) (bc cls : Vec ((hm1+1) * d))
    (pos : Mat (N + 1) ((hm1+1) * d))
    (ps : Fin k → BlockParamsV ((hm1+1) * d) mlpDim)
    (γF βF : Vec ((hm1+1) * d))
    (Wcls : Mat ((hm1+1) * d) nClasses) (bcls : Vec nClasses)
    (x : Vec (ic * H * W)) :
    (vitNetLayer ic H W patchSize N mlpDim hm1 d nClasses k ε hε
        Wc bc cls pos ps γF βF Wcls bcls).fwd x
      = vitForwardKV ic H W patchSize N mlpDim (hm1+1) d nClasses k
          Wc bc cls pos ε ps γF βF Wcls bcls x := by
  show (vitClassifierLayer N ((hm1+1) * d) nClasses Wcls bcls).fwd
        ((vitFinalLNLayer N ((hm1+1) * d) ε γF βF hε).fwd
          ((vitTrunkV (Np1 := N + 1) ε hε k ps).fwd
            (patchEmbedFlat ic H W patchSize N ((hm1+1) * d) Wc bc cls pos x))) = _
  rw [vitTrunkV_fwd ε hε k ps (patchEmbedFlat ic H W patchSize N ((hm1+1) * d) Wc bc cls pos x)]
  rfl

/-- **The whole-net chain's backward graph is `vitNetBackGraph`.** The stem/trunk/LN/head
    version of `vitTrunkV_graph`: `ViTBackB0`'s hand-composed whole-net graph is what
    `CertLayer.comp` produces, so `vitNetBackGraph_faithful` is a consequence of the shared
    combinator rather than a parallel result. The saved activations `comp` threads
    automatically are exactly the ones that theorem pins by hand. -/
theorem vitNetLayer_graph (ic H W patchSize N mlpDim hm1 d nClasses k : Nat)
    (ε : ℝ) (hε : 0 < ε)
    (Wc : Kernel4 ((hm1+1) * d) ic patchSize patchSize) (bc cls : Vec ((hm1+1) * d))
    (pos : Mat (N + 1) ((hm1+1) * d))
    (ps : Fin k → BlockParamsV ((hm1+1) * d) mlpDim)
    (γF βF : Vec ((hm1+1) * d))
    (Wcls : Mat ((hm1+1) * d) nClasses) (bcls : Vec nClasses)
    (x : Vec (ic * H * W)) (e : SHlo nClasses) :
    (vitNetLayer ic H W patchSize N mlpDim hm1 d nClasses k ε hε
        Wc bc cls pos ps γF βF Wcls bcls).graph x e
      = vitNetBackGraph ic H W patchSize N mlpDim hm1 d nClasses k ε Wc ps γF Wcls
          (Mat.unflatten (patchEmbedFlat ic H W patchSize N ((hm1+1) * d) Wc bc cls pos x))
          (Mat.unflatten
            (vitBodyKVFlat (N + 1) (hm1+1) d mlpDim ε k ps
              (patchEmbedFlat ic H W patchSize N ((hm1+1) * d) Wc bc cls pos x)))
          e := by
  set PE := patchEmbedFlat ic H W patchSize N ((hm1+1) * d) Wc bc cls pos x with hPE
  show patchEmbedBackGraph ic H W patchSize N ((hm1+1) * d) Wc
        ((vitTrunkV (Np1 := N + 1) ε hε k ps).graph PE
          (finalLNBackGraph N ((hm1+1) * d) ε γF
            ((vitTrunkV (Np1 := N + 1) ε hε k ps).fwd PE)
            (classifierBackGraph N ((hm1+1) * d) nClasses Wcls e))) = _
  rw [vitTrunkV_fwd ε hε k ps PE]
  -- The trunk's graph is stated at a FLATTENED saved activation; `PE` is a raw `Vec`, so
  -- present it as `flatten (unflatten ·)` — the same round-trip `vitTrunkV_graph` consumes.
  conv_lhs => rw [show PE = Mat.flatten (Mat.unflatten PE) from (Mat.flatten_unflatten PE).symm]
  rw [vitTrunkV_graph ε hε k ps (Mat.unflatten PE)]
  show _ = patchEmbedBackGraph ic H W patchSize N ((hm1+1) * d) Wc
        (vitBodyBackGraphKMHV ε k ps (Mat.unflatten PE)
          (finalLNBackGraph N ((hm1+1) * d) ε γF
            (Mat.flatten (Mat.unflatten
              (vitBodyKVFlat (N + 1) (hm1+1) d mlpDim ε k ps PE)))
            (classifierBackGraph N ((hm1+1) * d) nClasses Wcls e)))
  -- ⚠ Both sides carry a `flatten ∘ unflatten` on the body output — the LHS's arrived from the
  -- `conv_lhs` round-trip above, the RHS's from `vitNetBackGraph`'s own `Mat.flatten bodyOut`.
  -- A bare `rw` cancels only the first and leaves the goal looking mismatched; cancel both.
  simp only [Mat.flatten_unflatten]

/-- The whole net is certified at **every** input — `ok = True` end to end, because every stage
    is smooth (affine stem, GELU/LN blocks, LN, affine head). No side condition anywhere. -/
theorem vitNetLayer_ok (ic H W patchSize N mlpDim hm1 d nClasses k : Nat)
    (ε : ℝ) (hε : 0 < ε)
    (Wc : Kernel4 ((hm1+1) * d) ic patchSize patchSize) (bc cls : Vec ((hm1+1) * d))
    (pos : Mat (N + 1) ((hm1+1) * d))
    (ps : Fin k → BlockParamsV ((hm1+1) * d) mlpDim)
    (γF βF : Vec ((hm1+1) * d))
    (Wcls : Mat ((hm1+1) * d) nClasses) (bcls : Vec nClasses)
    (x : Vec (ic * H * W)) :
    (vitNetLayer ic H W patchSize N mlpDim hm1 d nClasses k ε hε
      Wc bc cls pos ps γF βF Wcls bcls).ok x :=
  ⟨trivial, vitTrunkV_ok ε hε k ps _, trivial, trivial⟩

/-- **Whole-net backward-graph faithfulness.** The reverse-composed backward graph
    `vitNetBackGraph` denotes the proven whole-net VJP `vitForwardKVHasVJP.backward` at every
    input image and output cotangent, at every depth `k` (multi-head, vector-LN). It falls out of
    `CertLayer.faithful` at `vitNetLayer` plus `vitNetLayer_graph` — the composition argument is
    `comp`'s, proven once in `CertifiedChain`, and the only ViT-specific input is the forward
    equality. -/
theorem vitNetBackGraph_faithful
    (ic H W patchSize N mlpDim hm1 d nClasses k : Nat) (ε : ℝ) (hε : 0 < ε)
    (Wc : Kernel4 ((hm1+1) * d) ic patchSize patchSize) (bc cls : Vec ((hm1+1) * d))
    (pos : Mat (N + 1) ((hm1+1) * d))
    (ps : Fin k → BlockParamsV ((hm1+1) * d) mlpDim)
    (γF βF : Vec ((hm1+1) * d))
    (Wcls : Mat ((hm1+1) * d) nClasses) (bcls : Vec nClasses)
    (x : Vec (ic * H * W)) (e : SHlo nClasses) :
    den (vitNetBackGraph ic H W patchSize N mlpDim hm1 d nClasses k ε Wc ps γF Wcls
          (Mat.unflatten (patchEmbedFlat ic H W patchSize N ((hm1+1) * d) Wc bc cls pos x))
          (Mat.unflatten
            (vitBodyKVFlat (N + 1) (hm1+1) d mlpDim ε k ps
              (patchEmbedFlat ic H W patchSize N ((hm1+1) * d) Wc bc cls pos x)))
          e)
      = (vitForwardKVHasVJP ic H W patchSize N mlpDim (hm1+1) d nClasses k
          Wc bc cls pos ε hε ps γF βF Wcls bcls).backward x (den e) := by
  rw [← vitNetLayer_graph ic H W patchSize N mlpDim hm1 d nClasses k ε hε
        Wc bc cls pos ps γF βF Wcls bcls x e,
    (vitNetLayer ic H W patchSize N mlpDim hm1 d nClasses k ε hε
        Wc bc cls pos ps γF βF Wcls bcls).faithful x
      (vitNetLayer_ok ic H W patchSize N mlpDim hm1 d nClasses k ε hε
        Wc bc cls pos ps γF βF Wcls bcls x) e]
  -- The fold's VJP is a `vjpCompAt` chain and the shipped one a `vjpComp` chain: different
  -- terms for maps that are equal only propositionally (`vitNetLayer_fwd`).
  exact HasVJPAt.backward_unique_of_eq
    (funext (vitNetLayer_fwd ic H W patchSize N mlpDim hm1 d nClasses k ε hε
      Wc bc cls pos ps γF βF Wcls bcls)) _
    ((vitForwardKVHasVJP ic H W patchSize N mlpDim (hm1+1) d nClasses k
      Wc bc cls pos ε hε ps γF βF Wcls bcls).toHasVJPAt x) (den e)

-- ════════════════════════════════════════════════════════════════
-- § ViT-Tiny's depth, as a check rather than prose
-- ════════════════════════════════════════════════════════════════

/-! The peer of `R34BWeights` / `R50BWeights` / `mnv4Blocks`: the shipped config's shape
pinned in the types instead of stated in a docstring. ViT-Tiny is 12 identical-shaped blocks at
`heads = 3`, `d_head = 64` (`D = 192`), `mlpDim = 768`, `N + 1 = 197` tokens — so unlike the conv
nets there is no ladder to pin, only the depth and the widths, and every block has the same type
(which is why `chain` over a `List.replicate`-shaped index is the right form here and a
per-row-typed parameter record buys nothing). -/

end Proofs.StableHLO

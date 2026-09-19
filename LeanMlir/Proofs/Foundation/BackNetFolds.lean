import LeanMlir.Proofs.Foundation.CertifiedChain
import LeanMlir.Proofs.Nets.ResNet.ResNet50BackB0
import LeanMlir.Proofs.Nets.ResNet.ResNet34BackB0
import LeanMlir.Proofs.Nets.EfficientNet.EfficientNetBackB0
import LeanMlir.Proofs.Nets.ConvNeXt.ConvNeXtBackB0
import LeanMlir.Proofs.Nets.ConvNeXt.ConvNeXtFullT
import LeanMlir.Proofs.Nets.EfficientNet.EfficientNetChainClose

/-! # `CertLayer` instances for the conv nets' blocks — enet, convnext, mnv2

`CertifiedChain.lean` made the backward-graph machinery net-agnostic: a block capstone packaged as a
`CertLayer` composes with `CertLayer.comp` / `CertLayer.chain` into stages and trunks with **no new
proof per net and no new proof per depth**. This file packages EfficientNet's MBConv, ConvNeXt's
channel-LN block and MobileNetV2's residual block. (R34's and R50's block layers live beside their
capstones in `ResNet34BackB0` / `ResNet50BackB0`, MobileNetV2's body in `MobileNetV2BackB0`, each
composed from its stage layers.)

⭐ **The per-net work was making the blocks pluggable, not proving anything.** Each capstone took
its cotangent as `dy : Vec n` and wrapped it internally as `.operand "%dy" dy`, so a block could
only ever be the LAST thing in a graph. All four now take `ecot : SHlo n` — strictly more general,
since the old statement is this one at `ecot := .operand "%dy" dy`. `residualBackGraph` was
generalized the same way, which is what let mnv2/enet/convnext follow.

## ⭐ Two tiers of layer, and the split is the ACTIVATION

| net | block activation | `ok` | why |
|---|---|---|---|
| **enet** | swish (smooth) | `True` | global `HasVJP`, lifted by `.toHasVJPAt` |
| **convnext** | gelu (smooth) | `True` | global `HasVJP` |
| **r34** | relu (one kink) | 2 clauses | `_at`: mid-relu + outer post-residual relu |
| **mnv2** | relu6 (two kinks) | 2 clauses | `_at`: `≠ 0 ∧ ≠ 6` at expand and depthwise |
| *(r50)* | relu | 3 clauses | 3 convs ⇒ 2 interior relus + the outer one |

⚠ A globally-smooth net is **not** a weaker certificate — it is a stronger one: `ok = True` means
the backward graph denotes the VJP *everywhere*, with no side condition to discharge. The `_at`
nets carry conditions because relu genuinely has no derivative at 0, and `CertLayer.comp` conjoins
those at the right activations rather than quietly dropping them.

## What this does NOT do

Stages and trunks are `comp`/`chain` applications, so they are available for every net here — but
none is written out, because a trunk needs the net's block table and resolution ladder spelled out
and that is per-net bookkeeping, not proof. ⚠ ViT is **not**
here: its blocks are per-token `Mat`-shaped with a different backward vocabulary
(`transformerBlockVBackGraphMH`). It is a separate sitting (`ViTBackNet.lean`).

✅ **That sitting happened — [`Nets/ViT/ViTBackNet.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/LeanMlir/Proofs/Nets/ViT/ViTBackNet.lean) (2026-08-10).** Two corrections it
forced, both worth reading before trusting this file's framing:
* ViT was never the *least*-folded net; it was the only one with a concrete whole-net backward
  graph (stem and head included, at every depth). See `CertifiedChain.lean`'s correction block.
* The ~11 min / ~14 GB figure is a **2-core CI runner** number. On a workstation `ViTBackB0`
  rebuilds in seconds, so "budget the CI cost" was not the constraint it looked like, and it is
  not a reason to defer work on this module.
-/

namespace Proofs.StableHLO

-- ════════════════════════════════════════════════════════════════
-- § EfficientNet — swish is smooth, so `ok = True`
-- ════════════════════════════════════════════════════════════════

/-- The batched EfficientNet MBConv residual block as a `CertLayer`. ⭐ Globally certified
    (`ok = True`): swish and sigmoid are smooth, so the block has a global `HasVJP` and the
    backward graph denotes it at **every** input. An endomorphism, so `chain` iterates it. -/
noncomputable def enetMBConvLayer (N : Nat) {c mid h w kHd kWd r : Nat}
    (We : Kernel4 mid c 1 1) (be : Vec mid) (εe : ℝ) (hεe : 0 < εe) (γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (hεd : 0 < εd) (γd βd : Vec mid)
    (Wz₁ : Mat mid r) (bz₁ : Vec r) (Wz₂ : Mat r mid) (bz₂ : Vec mid)
    (Wp : Kernel4 c mid 1 1) (bp : Vec c) (εp : ℝ) (hεp : 0 < εp) (γp βp : Vec c) :
    CertLayer (N * (c * h * w)) (N * (c * h * w)) where
  fwd := mbResidFwdB N (h := h) (w := w) We be εe γe βe Wd bd εd γd βd
    Wz₁ bz₁ Wz₂ bz₂ Wp bp εp γp βp
  ok := fun _ => True
  diff := fun x _ => (mbResidFwdB_differentiable N (h := h) (w := w) We be εe hεe γe βe
    Wd bd εd hεd γd βd Wz₁ bz₁ Wz₂ bz₂ Wp bp εp hεp γp βp) x
  vjp := fun x _ => (mbResidFwdB_has_vjp N We be εe hεe γe βe Wd bd εd hεd γd βd
    Wz₁ bz₁ Wz₂ bz₂ Wp bp εp hεp γp βp).toHasVJPAt x
  graph := fun x e => mbResidBlockBackBatchedGraph We be εe γe βe Wd bd εd γd βd
    Wz₁ bz₁ Wz₂ bz₂ Wp bp εp γp βp x e
  faithful := fun x _ e => mbResidBlockBackBatchedGraph_faithful We be εe hεe γe βe
    Wd bd εd hεd γd βd Wz₁ bz₁ Wz₂ bz₂ Wp bp εp hεp γp βp x e

-- ════════════════════════════════════════════════════════════════
-- § ConvNeXt — gelu is smooth, so `ok = True`
-- ════════════════════════════════════════════════════════════════

/-- ⭐ The **channel-LN** ConvNeXt block as a `CertLayer` — the form the *shipped* net's stages are
    actually built from (`cnxResidBlockChBackGraph_faithful` is described in `ConvNeXtBackB0` as
    "the capstone the shipped net was missing"). This is the one to chain for a real ConvNeXt
    stage. -/
noncomputable def cnxBlockChLayer {c cExp h w kH kW : Nat}
    (p : CnxBlockParamsCh c cExp h w kH kW) (hε : 0 < p.εn) :
    CertLayer (c * h * w) (c * h * w) where
  fwd := cnxBlockChW p
  ok := fun _ => True
  diff := fun x _ => (cnxBlockChW_diff p hε) x
  vjp := fun x _ => (cnxBlockChW_has_vjp p hε).toHasVJPAt x
  graph := fun x e => cnxResidBlockChBackGraph p x e
  faithful := fun x _ e => cnxResidBlockChBackGraph_faithful p hε x e

-- ════════════════════════════════════════════════════════════════
-- § MobileNetV2 — relu6, so `_at` with TWO-SIDED clauses
-- ════════════════════════════════════════════════════════════════

/-- The batched MobileNetV2 inverted-residual block as a `CertLayer`.

    ⚠ Its `ok` clauses are **two-sided** (`≠ 0 ∧ ≠ 6`) because relu6 has a kink at each end — the
    difference `MobileNetV2BackB0` records against R34's one-sided relu. ⚠ And unlike R34/R50 there
    is **no outer relu**: mnv2's block output IS the residual add, so `ok` has two clauses covering
    the expand and depthwise stages and none for an output activation — `mnv2BodyLayer`'s, since an
    identity skip adds no condition. -/
noncomputable def mnv2ResidBlockLayer (N : Nat) {c mid h w kHd kWd : Nat}
    (We : Kernel4 mid c 1 1) (be : Vec mid) (εe : ℝ) (hεe : 0 < εe) (γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (hεd : 0 < εd) (γd βd : Vec mid)
    (Wp : Kernel4 c mid 1 1) (bp : Vec c) (εp : ℝ) (hεp : 0 < εp) (γp βp : Vec c) :
    CertLayer (N * (c * h * w)) (N * (c * h * w)) :=
  CertLayer.residual (mnv2BodyLayer N (h := h) (w := w) We be εe hεe γe βe Wd bd εd hεd γd βd
    Wp bp εp hεp γp βp)

-- ════════════════════════════════════════════════════════════════
-- § The payoff — every net folds with the SAME two combinators
-- ════════════════════════════════════════════════════════════════

/-- ⭐⭐ **A chain of MBConv blocks is certified, at any depth.** Immediate from
    `CertLayer.chain_faithful`; stated per net only to make the payoff visible. The same one-liner
    works for every layer in this file, which is what "the machinery is net-agnostic" means. -/
theorem enetChain_faithful {N c _mid h w _kHd _kWd _r : Nat}
    (Ls : List (CertLayer (N * (c * h * w)) (N * (c * h * w))))
    (x : Vec (N * (c * h * w))) (hx : (CertLayer.chain Ls).ok x)
    (e : SHlo (N * (c * h * w))) :
    den ((CertLayer.chain Ls).graph x e) = ((CertLayer.chain Ls).vjp x hx).backward (den e) :=
  CertLayer.chain_faithful Ls x hx e

end Proofs.StableHLO

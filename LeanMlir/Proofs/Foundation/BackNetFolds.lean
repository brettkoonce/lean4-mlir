import LeanMlir.Proofs.Foundation.CertifiedChain
import LeanMlir.Proofs.Nets.ResNet.ResNet50BackB0
import LeanMlir.Proofs.Nets.ResNet.ResNet34BackB0
import LeanMlir.Proofs.Nets.EfficientNet.EfficientNetBackB0
import LeanMlir.Proofs.Nets.ConvNeXt.ConvNeXtBackB0
import LeanMlir.Proofs.Nets.ConvNeXt.ConvNeXtFullT
import LeanMlir.Proofs.Nets.EfficientNet.EfficientNetChainClose

/-! # ConvNeXt's channel-LN block as a `CertLayer`

`CertifiedChain.lean` made the backward-graph machinery net-agnostic: a block capstone packaged as a
`CertLayer` composes with `CertLayer.comp` / `CertLayer.chain` into stages and trunks with **no new
proof per net and no new proof per depth**. The conv nets' block layers live beside their capstones
(`ResNet34BackB0`, `ResNet50BackB0`, `MobileNetV2BackB0`, `MobileNetV4BackB0`,
`EfficientNetBackNet`), each composed from its stage layers; this file holds ConvNeXt's channel-LN
block, `cnxBlockChLayer`.

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

end Proofs.StableHLO

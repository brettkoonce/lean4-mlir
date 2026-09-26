import LeanMlir.Proofs.Nets.MobileNet.MobileNetV2Close
import LeanMlir.Proofs.Nets.ConvNeXt.ConvNeXt

/-! # ConvNeXt — the block cotangent chain

The param bridges (`ConvNeXtFold` and the shared conv and dense bridges) certify
each ConvNeXt param output for *any* cotangent `dy` at that layer's output. This file defines the
cotangent the **actual backward chain delivers** at each layer, and the step ties
(`ConvNeXtStepTie`, `ConvNeXtStepTieGB`) feed those cotangents to the bridges at the real forward.
The definitions are per example; the channel LayerNorm is per-example separable, so the batched
tie lifts them with `batchMapAux`.

The chain through a ConvNeXt block composes the *rendered* backward denotations — layer-scale back
(`layerScale γls` applied to the cotangent: the input-VJP `γ ⊙ dy` is the forward map itself, the
`layerScaleF`-on-the-cotangent trick the render uses), the 1×1 conv input-VJP
(`conv2dHasVJP3` via the flatten bridge, = `convBack`'s denotation), the GELU mask
(`dy ⊙ geluScalarDeriv`, = `geluHasVJP`'s backward; `geluScalarDeriv_eq` certifies the closed
form `geluBack` emits) — back through `layerScale → project → gelu → expand` to the LN output,
where the channel-LN (`chanLNTensor3`, `Vec c` γ/β) grads read it:

  block:  o = addV( layerScale γls (conv₁ₓ₁ₚᵣ( gelu( conv₁ₓ₁ₑₓ( LN( dw₇ₓ₇(x) ))))), x )

The residual `addV` is the outermost op and passes the block cotangent `dyOut` straight through to
the layer-scale output (ConvNeXt has no post-add activation — the r34 `relu(add(…))` mask never
appears), and the identity skip adds `dyOut` back at the block input. Unlike MNV2/r34 there is no
stride split: ConvNeXt blocks keep resolution (stride-1 7×7 depthwise), so one set of cotangent
definitions covers every block, including the cotangents of the two ConvNeXt-signature families
(layer-scale `γ`, channel-LN (`chanLNTensor3`, `Vec c` γ/β)), which MNV2/r34 had no analogue of. The
step ties carry the chain on through the LN input-VJP and the depthwise.
-/

namespace Proofs

open scoped BigOperators

-- ════════════════════════════════════════════════════════════════
-- § The cotangent the block backward chain delivers at each layer output
--   (saved activations named as in the Item B render: xin → d → nl → e → g → p → ls → out)
-- ════════════════════════════════════════════════════════════════

/-- Cotangent at the **project conv output** (= the layer-scale input): `layerScale γls dyOut`
    — the forward map applied to the block cotangent. The residual `addV` passes `dyOut` through
    to the layer-scale output unchanged (no post-add activation), and `layerScale`'s input-VJP
    `γ ⊙ dy` is `layerScale γ` itself (diagonal/symmetric — `layerScaleHasVJP`), which is why
    the render emits a second `layerScaleF` on the cotangent rather than a backward token. -/
noncomputable def cnxCotP {n : Nat} (γls : Vec n) (dyOut : Vec n) : Vec n :=
  layerScale γls dyOut

/-- Cotangent at the **expand conv output** (`cExp` ch, pre-GELU): continue through the project
    1×1 conv-back and the GELU mask (`geluScalarDeriv` at the saved pre-GELU activation `e`). -/
noncomputable def cnxCotE {c cExp h w : Nat} (γls : Vec (c * h * w))
    (Wpr : Kernel4 c cExp 1 1) (bpr : Vec c)
    (g e : Vec (cExp * h * w)) (dyOut : Vec (c * h * w)) : Vec (cExp * h * w) :=
  let cotG := (HasVJP3.toHasVJP (conv2dHasVJP3 (h := h) (w := w) Wpr bpr)).backward g
    (cnxCotP γls dyOut)
  fun i => cotG i * geluScalarDeriv (e i)

/-- Cotangent at the **LN output** (`c` ch): continue through the expand 1×1 conv-back. This is
    the cotangent the channel-LN (`chanLNTensor3`, `Vec c` γ/β) grads contract with. -/
noncomputable def cnxCotN {c cExp h w : Nat} (γls : Vec (c * h * w))
    (Wex : Kernel4 cExp c 1 1) (bex : Vec cExp) (Wpr : Kernel4 c cExp 1 1) (bpr : Vec c)
    (nl : Vec (c * h * w)) (g e : Vec (cExp * h * w)) (dyOut : Vec (c * h * w)) :
    Vec (c * h * w) :=
  (HasVJP3.toHasVJP (conv2dHasVJP3 (h := h) (w := w) Wex bex)).backward nl
    (cnxCotE γls Wpr bpr g e dyOut)

end Proofs

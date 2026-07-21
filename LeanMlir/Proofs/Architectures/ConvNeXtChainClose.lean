import LeanMlir.Proofs.Architectures.ConvNeXtClose

/-! # ConvNeXt Item D — pinning the block cotangent chain

`ConvNeXtClose.lean` (Item C) certifies each ConvNeXt param output for *any* cotangent `dy` at that
layer's output. This file pins `dy` to the cotangent the **actual backward chain delivers** — the
ConvNeXt analogue of `MobileNetV2ChainClose` / `ResNet34ChainClose` (`planning/convnext_close.md`
Item D). Pure-Lean, batch-1 — LayerNorm is per-example separable, so none of EfficientNet's
batched-VJP machinery (`batchMap_has_vjp`) is needed.

The chain through a ConvNeXt block composes the *rendered* backward denotations — layer-scale back
(`layerScale γls` applied to the cotangent: the input-VJP `γ ⊙ dy` is the forward map itself, the
`layerScaleF`-on-the-cotangent trick the Item B render uses), the 1×1 conv input-VJP
(`conv2d_has_vjp3` via the flatten bridge, = `convBack`'s denotation), the GELU mask
(`dy ⊙ geluScalarDeriv`, = `gelu_has_vjp`'s backward; `geluScalarDeriv_eq` certifies the closed
form `geluBack` emits), the scalar-LN input-VJP (`bn_grad_input`, = `bnBack`'s denotation), and the
depthwise input-VJP (`depthwiseFlat_has_vjp`, = `depthwiseBack`'s denotation) — back through
`layerScale → project → gelu → expand → LN → depthwise`:

  block:  o = addV( layerScale γls (conv₁ₓ₁ₚᵣ( gelu( conv₁ₓ₁ₑₓ( LN( dw₇ₓ₇(x) ))))), x )

The residual `addV` is the outermost op and passes the block cotangent `dyOut` straight through to
the layer-scale output (ConvNeXt has no post-add activation — the r34 `relu(add(…))` mask never
appears), and the identity skip adds `dyOut` back at the block input. Unlike MNV2/r34 there is no
stride split: ConvNeXt blocks keep resolution (stride-1 7×7 depthwise), so one set of cotangent
definitions covers every block. Each param output then denotes `θ − lr·(certified ∂/∂θ · the
actual-chain cotangent)` — including the two ConvNeXt-signature families (layer-scale `γ`,
scalar-LN `γ/β`), which MNV2/r34 had no analogue of. Pins the cotangent; the `= ∂loss/∂θ` fold
stays separate, as for the CNN. 3-axiom clean.
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
    `γ ⊙ dy` is `layerScale γ` itself (diagonal/symmetric — `layerScale_has_vjp`), which is why
    the Item B render emits a second `layerScaleF` on the cotangent rather than a backward token. -/
noncomputable def cnxCotP {n : Nat} (γls : Vec n) (dyOut : Vec n) : Vec n :=
  layerScale γls dyOut

/-- Cotangent at the **expand conv output** (`cExp` ch, pre-GELU): continue through the project
    1×1 conv-back and the GELU mask (`geluScalarDeriv` at the saved pre-GELU activation `e`). -/
noncomputable def cnxCotE {c cExp h w : Nat} (γls : Vec (c * h * w))
    (Wpr : Kernel4 c cExp 1 1) (bpr : Vec c)
    (g e : Vec (cExp * h * w)) (dyOut : Vec (c * h * w)) : Vec (cExp * h * w) :=
  let cotG := (hasVJP3_to_hasVJP (conv2d_has_vjp3 (h := h) (w := w) Wpr bpr)).backward g
    (cnxCotP γls dyOut)
  fun i => cotG i * geluScalarDeriv (e i)

/-- Cotangent at the **LN output** (`c` ch): continue through the expand 1×1 conv-back. This is
    the cotangent the scalar-LN `γ/β` grads contract with — the substantive LN pin. -/
noncomputable def cnxCotN {c cExp h w : Nat} (γls : Vec (c * h * w))
    (Wex : Kernel4 cExp c 1 1) (bex : Vec cExp) (Wpr : Kernel4 c cExp 1 1) (bpr : Vec c)
    (nl : Vec (c * h * w)) (g e : Vec (cExp * h * w)) (dyOut : Vec (c * h * w)) :
    Vec (c * h * w) :=
  (hasVJP3_to_hasVJP (conv2d_has_vjp3 (h := h) (w := w) Wex bex)).backward nl
    (cnxCotE γls Wpr bpr g e dyOut)

/-- Cotangent at the **depthwise conv output** (`c` ch): continue through the scalar-LN
    input-VJP (`bn_grad_input` — the three-term `bnBack` denotation, recomputing x̂/istd from
    the saved LN input `d`). -/
noncomputable def cnxCotD {c cExp h w : Nat} (ε γn : ℝ) (γls : Vec (c * h * w))
    (Wex : Kernel4 cExp c 1 1) (bex : Vec cExp) (Wpr : Kernel4 c cExp 1 1) (bpr : Vec c)
    (d nl : Vec (c * h * w)) (g e : Vec (cExp * h * w)) (dyOut : Vec (c * h * w)) :
    Vec (c * h * w) :=
  bn_grad_input (c * h * w) ε γn d (cnxCotN γls Wex bex Wpr bpr nl g e dyOut)

/-- Cotangent at the **block input**: the depthwise input-VJP of `cnxCotD`, plus the identity
    skip's `dyOut`. This is what the block hands upstream — the next block's `dyOut`, and at
    block 1 the stem's `dyStem`. -/
noncomputable def cnxCotXin {c cExp h w kH kW : Nat} (ε γn : ℝ) (γls : Vec (c * h * w))
    (Wdw : DepthwiseKernel c kH kW) (bdw : Vec c)
    (Wex : Kernel4 cExp c 1 1) (bex : Vec cExp) (Wpr : Kernel4 c cExp 1 1) (bpr : Vec c)
    (xin d nl : Vec (c * h * w)) (g e : Vec (cExp * h * w)) (dyOut : Vec (c * h * w)) :
    Vec (c * h * w) :=
  fun i => (depthwiseFlat_has_vjp (h := h) (w := w) Wdw bdw).backward xin
      (cnxCotD ε γn γls Wex bex Wpr bpr d nl g e dyOut) i + dyOut i

/-- Cotangent at the **stem conv output**: `bn_grad_input` at the saved patchify output (the stem
    is conv → scalar-LN, nothing else — no activation, no pool), applied to `dyStem`, the
    cotangent block 1 delivers at the stem-LN output (= block 1's `cnxCotXin`). -/
noncomputable def cnxStemCot {n : Nat} (ε γst : ℝ) (patch dyStem : Vec n) : Vec n :=
  bn_grad_input n ε γst patch dyStem

-- ════════════════════════════════════════════════════════════════
-- § The chain-pinned closes — the Item C bridges at the actual cotangents
-- ════════════════════════════════════════════════════════════════

/-- **Layer-scale γ, chain-certified.** The chain cotangent at the layer-scale output IS the block
    cotangent `dyOut` — the residual `addV` is the outermost op and ConvNeXt has no post-add
    activation, so the passthrough is exact. `γlsⁿ` denotes `γls − lr·(certified ∂(layerScale)/∂γ ·
    dyOut)` with the saved project output `p` as the layer input. -/
theorem cnx_render_lsgamma_chain_certified {n : Nat} (p : Vec n) (γls : Vec n) (dyOut : Vec n)
    (lr : ℝ) (i : Fin n) :
    γls i - lr * layerScale_grad_gamma p dyOut i
      = γls i - lr * ∑ j : Fin n, pdiv (fun γ' : Vec n => layerScale γ' p) γls i j * dyOut j :=
  cnx_render_lsgamma_certified p γls dyOut lr i

/-- **Project 1×1 conv weight, chain-certified.** `Wprⁿ` denotes `Wpr − lr·(certified ∂conv/∂Wpr ·
    layerScale γls dyOut)` — the conv weight bridge at the layer-scale-back cotangent, with the
    saved GELU output `g` as the conv input. -/
theorem cnx_render_projW_chain_certified {c cExp h w : Nat}
    (bpr : Vec c) (g : Vec (cExp * h * w)) (γls : Vec (c * h * w)) (dyOut : Vec (c * h * w))
    (v : Vec (c * cExp * 1 * 1)) (lr : ℝ) (idx : Fin (c * cExp * 1 * 1)) :
    v idx - lr * (conv2d_weight_grad_has_vjp bpr (Tensor3.unflatten g)).backward v
        (cnxCotP γls dyOut) idx
      = v idx - lr * ∑ j : Fin (c * h * w),
          pdiv (fun v' : Vec (c * cExp * 1 * 1) =>
                  Tensor3.flatten (conv2d (Kernel4.unflatten v') bpr (Tensor3.unflatten g))) v idx j
            * cnxCotP γls dyOut j :=
  cnn_render_convW_certified bpr (Tensor3.unflatten g) v (cnxCotP γls dyOut) lr idx

/-- **Project 1×1 conv bias, chain-certified.** -/
theorem cnx_render_projb_chain_certified {c cExp h w : Nat}
    (Wpr : Kernel4 c cExp 1 1) (g : Vec (cExp * h * w)) (bpr : Vec c)
    (γls : Vec (c * h * w)) (dyOut : Vec (c * h * w)) (lr : ℝ) (o : Fin c) :
    bpr o - lr * (conv2d_bias_grad_has_vjp Wpr (Tensor3.unflatten g)).backward bpr
        (cnxCotP γls dyOut) o
      = bpr o - lr * ∑ j : Fin (c * h * w),
          pdiv (fun b' : Vec c =>
                  Tensor3.flatten (conv2d Wpr b' (Tensor3.unflatten g))) bpr o j
            * cnxCotP γls dyOut j :=
  cnn_render_convb_certified Wpr (Tensor3.unflatten g) bpr (cnxCotP γls dyOut) lr o

/-- **Expand 1×1 conv weight, chain-certified.** `Wexⁿ` denotes `Wex − lr·(certified ∂conv/∂Wex ·
    the chain cotangent at the expand output)` — through layer-scale back, project conv-back, and
    the GELU mask — with the saved LN output `nl` as the conv input. -/
theorem cnx_render_expW_chain_certified {c cExp h w : Nat}
    (bex : Vec cExp) (nl : Vec (c * h * w)) (γls : Vec (c * h * w))
    (Wpr : Kernel4 c cExp 1 1) (bpr : Vec c) (g e : Vec (cExp * h * w)) (dyOut : Vec (c * h * w))
    (v : Vec (cExp * c * 1 * 1)) (lr : ℝ) (idx : Fin (cExp * c * 1 * 1)) :
    v idx - lr * (conv2d_weight_grad_has_vjp bex (Tensor3.unflatten nl)).backward v
        (cnxCotE γls Wpr bpr g e dyOut) idx
      = v idx - lr * ∑ j : Fin (cExp * h * w),
          pdiv (fun v' : Vec (cExp * c * 1 * 1) =>
                  Tensor3.flatten (conv2d (Kernel4.unflatten v') bex (Tensor3.unflatten nl))) v idx j
            * cnxCotE γls Wpr bpr g e dyOut j :=
  cnn_render_convW_certified bex (Tensor3.unflatten nl) v (cnxCotE γls Wpr bpr g e dyOut) lr idx

/-- **Expand 1×1 conv bias, chain-certified.** -/
theorem cnx_render_expb_chain_certified {c cExp h w : Nat}
    (Wex : Kernel4 cExp c 1 1) (nl : Vec (c * h * w)) (bex : Vec cExp) (γls : Vec (c * h * w))
    (Wpr : Kernel4 c cExp 1 1) (bpr : Vec c) (g e : Vec (cExp * h * w)) (dyOut : Vec (c * h * w))
    (lr : ℝ) (o : Fin cExp) :
    bex o - lr * (conv2d_bias_grad_has_vjp Wex (Tensor3.unflatten nl)).backward bex
        (cnxCotE γls Wpr bpr g e dyOut) o
      = bex o - lr * ∑ j : Fin (cExp * h * w),
          pdiv (fun b' : Vec cExp =>
                  Tensor3.flatten (conv2d Wex b' (Tensor3.unflatten nl))) bex o j
            * cnxCotE γls Wpr bpr g e dyOut j :=
  cnn_render_convb_certified Wex (Tensor3.unflatten nl) bex (cnxCotE γls Wpr bpr g e dyOut) lr o

/-- **Block scalar-LN γ, chain-certified.** The substantive LN pin: the chain cotangent at the LN
    output is `cnxCotN` (through layer-scale back → project conv-back → GELU mask → expand
    conv-back), and `γⁿ` denotes `γ − lr·(certified ∂LN/∂γ · cnxCotN)` with the saved depthwise
    output `d` as the LN input — the Item C `Vec 1` embedding at the actual chain cotangent. -/
theorem cnx_render_lngamma_chain_certified {c cExp h w : Nat}
    (ε βn : ℝ) (γ : Vec 1) (γls : Vec (c * h * w))
    (Wex : Kernel4 cExp c 1 1) (bex : Vec cExp) (Wpr : Kernel4 c cExp 1 1) (bpr : Vec c)
    (d nl : Vec (c * h * w)) (g e : Vec (cExp * h * w)) (dyOut : Vec (c * h * w)) (lr : ℝ) :
    γ 0 - lr * bn_grad_gamma (c * h * w) ε d (cnxCotN γls Wex bex Wpr bpr nl g e dyOut)
      = γ 0 - lr * ∑ j : Fin (c * h * w),
          pdiv (fun γ' : Vec 1 => layerNormForward (c * h * w) ε (γ' 0) βn d) γ 0 j
            * cnxCotN γls Wex bex Wpr bpr nl g e dyOut j :=
  cnx_render_lngamma_certified (c * h * w) ε βn γ d
    (cnxCotN γls Wex bex Wpr bpr nl g e dyOut) lr

/-- **Block scalar-LN β, chain-certified.** -/
theorem cnx_render_lnbeta_chain_certified {c cExp h w : Nat}
    (ε γn : ℝ) (β : Vec 1) (γls : Vec (c * h * w))
    (Wex : Kernel4 cExp c 1 1) (bex : Vec cExp) (Wpr : Kernel4 c cExp 1 1) (bpr : Vec c)
    (d nl : Vec (c * h * w)) (g e : Vec (cExp * h * w)) (dyOut : Vec (c * h * w)) (lr : ℝ) :
    β 0 - lr * bn_grad_beta (c * h * w) (cnxCotN γls Wex bex Wpr bpr nl g e dyOut)
      = β 0 - lr * ∑ j : Fin (c * h * w),
          pdiv (fun β' : Vec 1 => layerNormForward (c * h * w) ε γn (β' 0) d) β 0 j
            * cnxCotN γls Wex bex Wpr bpr nl g e dyOut j :=
  cnx_render_lnbeta_certified (c * h * w) ε γn β d
    (cnxCotN γls Wex bex Wpr bpr nl g e dyOut) lr

/-- **Depthwise 7×7 weight, chain-certified.** `Wdwⁿ` denotes `Wdw − lr·(certified
    ∂(depthwiseConv2d)/∂Wdw · the deepest in-block cotangent)` — through the whole chain down to
    the scalar-LN input-VJP — with the saved block input `xin` as the conv input. -/
theorem cnx_render_dw7W_chain_certified {c cExp h w : Nat}
    (bdw : Vec c) (xin : Vec (c * h * w)) (ε γn : ℝ) (γls : Vec (c * h * w))
    (Wex : Kernel4 cExp c 1 1) (bex : Vec cExp) (Wpr : Kernel4 c cExp 1 1) (bpr : Vec c)
    (Wdw : DepthwiseKernel c 7 7) (d nl : Vec (c * h * w)) (g e : Vec (cExp * h * w))
    (dyOut : Vec (c * h * w)) (lr : ℝ) (ci : Fin c) (hi : Fin 7) (wi : Fin 7) :
    Wdw ci hi wi - lr * (depthwise_weight_grad_has_vjp3 bdw (Tensor3.unflatten xin)).backward Wdw
        (Tensor3.unflatten (cnxCotD ε γn γls Wex bex Wpr bpr d nl g e dyOut)) ci hi wi
      = Wdw ci hi wi - lr * ∑ co : Fin c, ∑ ho : Fin h, ∑ wo : Fin w,
          pdiv3 (fun W' : DepthwiseKernel c 7 7 => depthwiseConv2d W' bdw (Tensor3.unflatten xin))
              Wdw ci hi wi co ho wo
            * (Tensor3.unflatten (cnxCotD ε γn γls Wex bex Wpr bpr d nl g e dyOut) :
                Tensor3 c h w) co ho wo :=
  cnx_render_dw7W_certified bdw (Tensor3.unflatten xin) Wdw
    (Tensor3.unflatten (cnxCotD ε γn γls Wex bex Wpr bpr d nl g e dyOut)) lr ci hi wi

/-- **Depthwise 7×7 bias, chain-certified.** -/
theorem cnx_render_dw7b_chain_certified {c cExp h w : Nat}
    (Wdw : DepthwiseKernel c 7 7) (xin : Vec (c * h * w)) (bdw : Vec c) (ε γn : ℝ)
    (γls : Vec (c * h * w))
    (Wex : Kernel4 cExp c 1 1) (bex : Vec cExp) (Wpr : Kernel4 c cExp 1 1) (bpr : Vec c)
    (d nl : Vec (c * h * w)) (g e : Vec (cExp * h * w)) (dyOut : Vec (c * h * w))
    (lr : ℝ) (cc : Fin c) :
    bdw cc - lr * (depthwise_bias_grad_has_vjp Wdw (Tensor3.unflatten xin)).backward bdw
        (cnxCotD ε γn γls Wex bex Wpr bpr d nl g e dyOut) cc
      = bdw cc - lr * ∑ j : Fin (c * h * w),
          pdiv (fun b' : Vec c =>
                  Tensor3.flatten (depthwiseConv2d Wdw b' (Tensor3.unflatten xin))) bdw cc j
            * cnxCotD ε γn γls Wex bex Wpr bpr d nl g e dyOut j :=
  cnx_render_dw7b_certified Wdw (Tensor3.unflatten xin) bdw
    (cnxCotD ε γn γls Wex bex Wpr bpr d nl g e dyOut) lr cc

/-- **Stem 1×1 patchify conv weight, chain-certified.** `Wstⁿ` denotes `Wst − lr·(certified
    ∂conv/∂Wst · bn_grad_input(patch, dyStem))` — the stem-LN input-VJP at the cotangent block 1
    delivers (`dyStem` = block 1's `cnxCotXin`; generic here, as for the MNV2/r34 stems). -/
theorem cnx_stem_render_convW_chain_certified {ic c h w : Nat}
    (bst : Vec c) (x : Vec (ic * h * w)) (ε γst : ℝ) (patch dyStem : Vec (c * h * w))
    (v : Vec (c * ic * 1 * 1)) (lr : ℝ) (idx : Fin (c * ic * 1 * 1)) :
    v idx - lr * (conv2d_weight_grad_has_vjp bst (Tensor3.unflatten x)).backward v
        (cnxStemCot ε γst patch dyStem) idx
      = v idx - lr * ∑ j : Fin (c * h * w),
          pdiv (fun v' : Vec (c * ic * 1 * 1) =>
                  Tensor3.flatten (conv2d (Kernel4.unflatten v') bst (Tensor3.unflatten x))) v idx j
            * cnxStemCot ε γst patch dyStem j :=
  cnn_render_convW_certified bst (Tensor3.unflatten x) v (cnxStemCot ε γst patch dyStem) lr idx

/-- **Stem 1×1 patchify conv bias, chain-certified.** -/
theorem cnx_stem_render_convb_chain_certified {ic c h w : Nat}
    (Wst : Kernel4 c ic 1 1) (x : Vec (ic * h * w)) (bst : Vec c) (ε γst : ℝ)
    (patch dyStem : Vec (c * h * w)) (lr : ℝ) (o : Fin c) :
    bst o - lr * (conv2d_bias_grad_has_vjp Wst (Tensor3.unflatten x)).backward bst
        (cnxStemCot ε γst patch dyStem) o
      = bst o - lr * ∑ j : Fin (c * h * w),
          pdiv (fun b' : Vec c =>
                  Tensor3.flatten (conv2d Wst b' (Tensor3.unflatten x))) bst o j
            * cnxStemCot ε γst patch dyStem j :=
  cnn_render_convb_certified Wst (Tensor3.unflatten x) bst (cnxStemCot ε γst patch dyStem) lr o

-- The blocks compose by instantiation: block 1's `dyOut` is block 2's `cnxCotXin`, and the stem's
-- `dyStem` is block 1's `cnxCotXin` — every theorem above is generic in its block-output
-- cotangent, exactly as in `MobileNetV2ChainClose`/`ResNet34ChainClose`. The stem-LN γ/β at
-- `dyStem` and the head-LN/dense at the loss cotangent add no chain content beyond Item C (their
-- cotangents are the generic parameters themselves); the loss-side `= ∂loss/∂θ` fold is the
-- separate `ConvLossFold` concern.

end Proofs

import LeanMlir.Proofs.Codegen.StableHLO

/-! # The LAMB triple, assembled — the peer of `adamW_triple_faithful`

`Lamb.lean` gives the ℝ reference (`lambDir`, `lambTrust`, `lambScale`) and `StableHLO.lean`
gives each emitted op its own `den` (`lambDirF_faithful`, `lambScaleF_faithful`,
`gradSumSqAccF_faithful`). What was missing is the level above: the `(θ', m', v')` a train step
returns per parameter, which is what `adamW_triple_faithful` states for AdamW and what the
ResNet-50 LAMB train steps (`resnet50in160_lambaccdp8x64bce`, …) return.

`Lamb.lean` proves properties of the trust ratio; the per-op denotations are `lambDirF_faithful`
and `lambScaleF_faithful` (both `rfl`, at `adamWParamF_faithful`'s bar). This file assembles them.

## The four ops, as `ResNet34RenderB`'s `.lamb` arm emits them

```
%r  = lambDirF   (θ, m, v, %b1 %ob1 %b2 %ob2 %bc1 %bc2 %eps %wd)  g   -- r = m̂/(√v̂+ε) + wd·θ
%n2 = gradSumSqAccF  %lzero  θ                                        -- ‖θ‖², THIS tensor's own
%s  = lambScaleF %n2 %r                                               -- trust(‖θ‖²,‖r‖) · r
%θ' = sgdParamF  θ %lr %s                                             -- θ − lr · that
%m' = adamMNextF …  g          %v' = adamVNextF …  g                  -- LAMB's m and v ARE Adam's
```

## What is proved

* `lambStep` — the ℝ triple, `(sgdParam lr θ (lambScale wn2 (lambDir …)), adamMNext, adamVNext)`.
* `lamb_triple_faithful` — the emitted four-op composition denotes it, `rfl`, at an ARBITRARY
  scalar child. The trust ratio's `‖θ‖²` is a graph operand, so the general statement is the one
  the AST supports and the two shipped instantiations are corollaries.
* `lamb_triple_faithful_committed` — at the render's own seed (`gradSumSqAccF` from `%lzero`
  over `θ` alone, one leaf deep), the scalar IS `gradSumSq θ`. **That single-leaf fold is the
  entire difference from the global-norm clip**, whose content is that ONE scalar is shared
  (`clipFactor_shared` against `lambScale_not_shared`); the two emit nearly the same lines.
* `lamb_triple_faithful_excluded` / `lambScale_zero_weight` — the `no_weight_decay` group
  feeds `%lzero` straight in as the scalar, and the step is then EXACTLY `θ − lr·r`: trust
  is 1, not 0 and not `0/0`. timm reads `if weight_decay != 0 or group['always_adapt']:` before
  computing the ratio, so an excluded parameter takes a plain Adam step; this says the render's
  "skip the op and pass the zero" implements that, rather than only that the guard does not crash.

## What is NOT claimed

**Faithfulness and well-definedness only**, `Lamb.lean`'s ceiling verbatim: that the rendered
LAMB denotes these functions, never that LAMB converges or beats AdamW.

**The accumulated form is the same tail at a different gradient node.** `.lambAccum k` emits
these four ops character-for-character against `Gt` (the `momVNextF`-at-`μ := akeep` accumulator)
rather than against `g`, so `lamb_triple_faithful` covers it at `e := ` that node — the theorem is
`∀ e`. Likewise the clip, which sits between the two.

**One replica.** Under `*dp*` the gradient node feeds `allReduceMeanF`, the collective as an
AST node, so `den e` is the per-replica gradient here and `DataParallelNode.lean`'s
`adamW_at_allReduceMeanF` is the shape that composes a tail with the replica mean.

`lambStep` and `lambScale_zero_weight` are defined in `Lamb.lean`.
-/

namespace Proofs

namespace StableHLO

/-- **The rendered LAMB triple is `Proofs.lambStep`** — `rfl`, i.e. the four emitted ops compose
    to exactly the ℝ definition, at `adamW_triple_faithful`'s bar.

    `s` is the scalar child carrying `‖θ‖²`; it is a binder because the AST makes it one, and the
    two shapes the render actually emits are the corollaries below. `e` is the gradient node,
    also a binder, so the theorem covers the plain, accumulated, clipped and data-parallel
    spellings without restatement. -/
theorem lamb_triple_faithful {n : Nat}
    (θN lrN mN vN b1N ob1N b2N ob2N bc1N bc2N epsN wdN : String) (ds : List Nat)
    (β₁ β₂ ε lr wd bc₁ bc₂ : ℝ) (θ m v : Vec n) (s : SHlo 1) (e : SHlo n) :
    (den (.sgdParamF θN lrN ds lr θ
            (.lambScaleF ds s
              (.lambDirF θN mN vN b1N ob1N b2N ob2N bc1N bc2N epsN wdN ds
                 β₁ β₂ ε wd bc₁ bc₂ θ m v e))),
     den (.adamMNextF mN b1N ob1N ds β₁ m e),
     den (.adamVNextF vN b2N ob2N ds β₂ v e))
      = lambStep β₁ β₂ ε lr wd bc₁ bc₂ (scalarOf (den s)) θ m v (den e) := rfl

/-- **The shipped scalar is THIS parameter's own squared norm.** The render seeds
    `gradSumSqAccF` at `%lzero` and folds over `θ` alone — one leaf deep, never across
    parameters. That is the entire structural difference from `clipGrad_faithful`, which folds
    the same op across every leaf and shares the result; the emitted lines are nearly identical
    and the quantifier is the whole content (`clipFactor_shared` / `lambScale_not_shared`). -/
theorem lamb_triple_faithful_committed {n : Nat}
    (θN lrN mN vN b1N ob1N b2N ob2N bc1N bc2N epsN wdN lzN : String) (ds : List Nat)
    (β₁ β₂ ε lr wd bc₁ bc₂ : ℝ) (θ m v : Vec n) (e : SHlo n) :
    (den (.sgdParamF θN lrN ds lr θ
            (.lambScaleF ds
              (.gradSumSqAccF ds (.operand lzN (fun _ => 0)) (.operand θN θ))
              (.lambDirF θN mN vN b1N ob1N b2N ob2N bc1N bc2N epsN wdN ds
                 β₁ β₂ ε wd bc₁ bc₂ θ m v e))),
     den (.adamMNextF mN b1N ob1N ds β₁ m e),
     den (.adamVNextF vN b2N ob2N ds β₂ v e))
      = lambStep β₁ β₂ ε lr wd bc₁ bc₂ (gradSumSq θ) θ m v (den e) := by
  rw [lamb_triple_faithful]
  congr 1
  show scalarOf (fun _ => (0 : ℝ) + gradSumSq θ) = gradSumSq θ
  simp only [scalarOf, zero_add]

/-- **The `no_weight_decay` group: the emitted step IS a plain Adam step at trust 1.**
    timm reads `if weight_decay != 0 or group['always_adapt']:` before computing the ratio, so an
    excluded parameter is NOT layer-adapted. The render implements that by skipping the norm op
    and passing `%lzero` — the same zero the fold would have been seeded from — and this says the
    result is `θ − lr·r` exactly, with `lambDir` untouched.

    The pre-existing zero-norm guard does not already give this at the artifact. It fires at
    `‖θ‖ = 0` exactly, i.e. step one, where every BatchNorm β and dense bias starts; from step two
    the parameter is small-but-nonzero and `‖θ‖/‖r‖` collapses to ~0.01–0.1 against timm's 1.0.
    So `lambTrust_zero_weight` alone does not give the excluded group's plain-Adam step; this
    theorem does. -/
theorem lamb_triple_faithful_excluded {n : Nat}
    (θN lrN mN vN b1N ob1N b2N ob2N bc1N bc2N epsN wdN lzN : String) (ds : List Nat)
    (β₁ β₂ ε lr wd bc₁ bc₂ : ℝ) (θ m v : Vec n) (e : SHlo n) :
    (den (.sgdParamF θN lrN ds lr θ
            (.lambScaleF ds (.operand lzN (fun _ => 0))
              (.lambDirF θN mN vN b1N ob1N b2N ob2N bc1N bc2N epsN wdN ds
                 β₁ β₂ ε wd bc₁ bc₂ θ m v e))),
     den (.adamMNextF mN b1N ob1N ds β₁ m e),
     den (.adamVNextF vN b2N ob2N ds β₂ v e))
      = (sgdParam lr θ (lambDir β₁ β₂ ε wd bc₁ bc₂ θ m v (den e)),
         adamMNext β₁ m (den e),
         adamVNext β₂ v (den e)) := by
  rw [lamb_triple_faithful]
  show (sgdParam lr θ (lambScale (scalarOf (fun _ => (0 : ℝ))) _), _, _) = _
  rw [show scalarOf (fun _ => (0 : ℝ)) = 0 from rfl, lambScale_zero_weight]

end StableHLO

end Proofs

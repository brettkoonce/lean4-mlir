import LeanMlir.Proofs.Nets.ResNet.ResNet50FullBVJP
import LeanMlir.Proofs.Nets.ResNet.ResNet34BackCertifiedTieB

/-! # `r50InputGradB` IS the certified whole-net ResNet-50 gradient

The whole-net input-gradient tie for `resnet50ForwardBFull`, the [3,4,6,3] bottleneck ladder at
batch BatchNorm, at a variable batch `N` and resolution `q`. The parameter-gradient tie is
`ResNet50StepTieB`'s.

## Almost all of it is ResNet-34's, reused rather than rewritten

`resnet50ForwardBFull` is `r34HeadB ∘ [3,4,6,3] bottlenecks ∘ r34StemB` — the stem and head are
literally ResNet-34's functions at R50's widths — so `ResNet34BackCertifiedTieB` supplies:

* `cbReluStridedBBack_eq_vjp_backward` and `r34HeadBBack_eq_vjp_backward`, the two endpoint ties;
* `maxPool3s2FlatBackB` and its `rfl` tie, plus the `StableHLO.batchMapAux` lift the batched
  3×3/s2 pool needed;
* `opaqueA0 … A16` and **`r34BFullHasVJPAt` itself** — the generic eighteen-stage apex.
  [3,4,6,3] is sixteen blocks for both nets, so the chain is the same construction and a second
  copy would be two writers for one fact. It is ResNet-34's only by where it was written; every
  dimension in it is a variable. The prefixes themselves are [`Foundation/OpaquePrefix.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/LeanMlir/Proofs/Foundation/OpaquePrefix.lean)'s.

What this file adds is the sixteen bottleneck slots, the tie, its `pdiv` reading, and the shape
check `resnet50ForwardBFull_eq_slots`.

## The two things ResNet-34's file did not face

**`q` is a BINDER.** One statement covers `resnet50in_fwd` (`q = 7`, 224 px) and
`resnet50in160_fwd` (`q = 5`, 160 px). So every dimension is
an explicit `2 * (…)` nest rather than `8 * q`: those are equal Nats and NOT definitionally equal
terms at a variable `q`. And **`0 < q` is a real hypothesis** where ResNet-34 needed none — the
stem pool's VJP needs its output grid nonempty, and at literal 56 that closed by `norm_num`.

## Scope

A smooth-point statement: the tie assumes `0 < q`, `0 < εs`, the stem relu clause (`h_stem`) and
the stem pool's per-example no-tie (`h_pool`), and takes each of the sixteen bottlenecks as an
opaque `HasVJPDiffAt` witness at its running activation (a bottleneck's three relu clauses — the
two interior ones and the post-residual outer one — are the caller's, inside that witness).
`resnet50in160_lambaccdp8x64bce` all-reduces every gradient (`allReduceMeanF`), and this is at the
per-replica gradient before it. It is about the INPUT gradient; the 161 parameter gradients are
`ResNet50StepTieB.lean`'s tie.
-/

-- Build note: the blocks stay opaque and there is no `backward_unique` step, for the reason
-- `ResNet34BackCertifiedTieB`'s build notes record: instantiating a tie of this shape at the
-- concrete blocks is a KERNEL deterministic timeout when the witnesses are `HasVJPAt` carrying a
-- saved activation. B0 takes that step only because swish has no kink. The shape check
-- `resnet50ForwardBFull_eq_slots` is what replaces it.

namespace Proofs

open scoped BigOperators

/-- **`r50InputGradB` IS the certified whole-net ResNet-50 gradient.** The committed backward
    chain, with its stem BatchNorm and relu-mask slots filled by the certified per-op backwards,
    its saved pool activation the stem's own, and its sixteen bottlenecks left OPAQUE, equals the
    backward of `r34BFullHasVJPAt` at those eighteen stages. `unfold`, two `rw`s, `rfl` — the
    pool needs no rewrite, being definitionally `batchMapHasVJPAt`'s backward. -/
theorem r50InputGradB_eq_r34B_full_vjp (N q : Nat) {nCls : Nat}
    (hq0 : 0 < q)
    (Ws : Kernel4 64 3 7 7) (bs : Vec 64) (εs : ℝ) (hεs : 0 < εs) (γs βs : Vec 64)
    (Wd : Mat 2048 nCls) (bd : Vec nCls)
    (b1 : Vec (N * (64 * (2 * (2 * (2 * q))) * (2 * (2 * (2 * q))))) → Vec (N * (256 * (2 * (2 * (2 * q))) * (2 * (2 * (2 * q))))))
    (b2 : Vec (N * (256 * (2 * (2 * (2 * q))) * (2 * (2 * (2 * q))))) → Vec (N * (256 * (2 * (2 * (2 * q))) * (2 * (2 * (2 * q))))))
    (b3 : Vec (N * (256 * (2 * (2 * (2 * q))) * (2 * (2 * (2 * q))))) → Vec (N * (256 * (2 * (2 * (2 * q))) * (2 * (2 * (2 * q))))))
    (b4 : Vec (N * (256 * (2 * (2 * (2 * q))) * (2 * (2 * (2 * q))))) → Vec (N * (512 * (2 * (2 * q)) * (2 * (2 * q)))))
    (b5 : Vec (N * (512 * (2 * (2 * q)) * (2 * (2 * q)))) → Vec (N * (512 * (2 * (2 * q)) * (2 * (2 * q)))))
    (b6 : Vec (N * (512 * (2 * (2 * q)) * (2 * (2 * q)))) → Vec (N * (512 * (2 * (2 * q)) * (2 * (2 * q)))))
    (b7 : Vec (N * (512 * (2 * (2 * q)) * (2 * (2 * q)))) → Vec (N * (512 * (2 * (2 * q)) * (2 * (2 * q)))))
    (b8 : Vec (N * (512 * (2 * (2 * q)) * (2 * (2 * q)))) → Vec (N * (1024 * (2 * q) * (2 * q))))
    (b9 : Vec (N * (1024 * (2 * q) * (2 * q))) → Vec (N * (1024 * (2 * q) * (2 * q))))
    (b10 : Vec (N * (1024 * (2 * q) * (2 * q))) → Vec (N * (1024 * (2 * q) * (2 * q))))
    (b11 : Vec (N * (1024 * (2 * q) * (2 * q))) → Vec (N * (1024 * (2 * q) * (2 * q))))
    (b12 : Vec (N * (1024 * (2 * q) * (2 * q))) → Vec (N * (1024 * (2 * q) * (2 * q))))
    (b13 : Vec (N * (1024 * (2 * q) * (2 * q))) → Vec (N * (1024 * (2 * q) * (2 * q))))
    (b14 : Vec (N * (1024 * (2 * q) * (2 * q))) → Vec (N * (2048 * q * q)))
    (b15 : Vec (N * (2048 * q * q)) → Vec (N * (2048 * q * q)))
    (b16 : Vec (N * (2048 * q * q)) → Vec (N * (2048 * q * q)))
    (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q))))))))
    (h_stem : R34StemSmoothAt N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) Ws bs εs γs βs x)
    (h_pool : R34PoolSmoothAt N (2 * (2 * (2 * q))) (2 * (2 * (2 * q)))
      (StableHLO.cbReluStridedB N (h := (2 * (2 * (2 * (2 * q))))) (w := (2 * (2 * (2 * (2 * q))))) Ws bs εs γs βs x))
    (hb1 : HasVJPDiffAt b1 (opaqueA0 (r34StemB N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) Ws bs εs γs βs) x))
    (hb2 : HasVJPDiffAt b2 (opaqueA1 (r34StemB N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) Ws bs εs γs βs) b1 x))
    (hb3 : HasVJPDiffAt b3 (opaqueA2 (r34StemB N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) Ws bs εs γs βs) b1 b2 x))
    (hb4 : HasVJPDiffAt b4 (opaqueA3 (r34StemB N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) Ws bs εs γs βs) b1 b2 b3 x))
    (hb5 : HasVJPDiffAt b5 (opaqueA4 (r34StemB N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) Ws bs εs γs βs) b1 b2 b3 b4 x))
    (hb6 : HasVJPDiffAt b6 (opaqueA5 (r34StemB N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) Ws bs εs γs βs) b1 b2 b3 b4 b5 x))
    (hb7 : HasVJPDiffAt b7 (opaqueA6 (r34StemB N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 x))
    (hb8 : HasVJPDiffAt b8 (opaqueA7 (r34StemB N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 x))
    (hb9 : HasVJPDiffAt b9 (opaqueA8 (r34StemB N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 x))
    (hb10 : HasVJPDiffAt b10 (opaqueA9 (r34StemB N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 x))
    (hb11 : HasVJPDiffAt b11 (opaqueA10 (r34StemB N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 x))
    (hb12 : HasVJPDiffAt b12 (opaqueA11 (r34StemB N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 x))
    (hb13 : HasVJPDiffAt b13 (opaqueA12 (r34StemB N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 x))
    (hb14 : HasVJPDiffAt b14 (opaqueA13 (r34StemB N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 x))
    (hb15 : HasVJPDiffAt b15 (opaqueA14 (r34StemB N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 x))
    (hb16 : HasVJPDiffAt b16 (opaqueA15 (r34StemB N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 x)) :
    r50InputGradB N q Ws Wd
      ((bnBatchLAHasVJP N 64 (2 * (2 * (2 * (2 * q)))) (2 * (2 * (2 * (2 * q)))) εs hεs γs βs).backward
        (StableHLO.batchMap N (flatConvStride2 Ws bs) x))
      (StableHLO.cbReluStridedB N (h := (2 * (2 * (2 * (2 * q))))) (w := (2 * (2 * (2 * (2 * q))))) Ws bs εs γs βs x)
      hb1.fst.backward
      hb2.fst.backward
      hb3.fst.backward
      hb4.fst.backward
      hb5.fst.backward
      hb6.fst.backward
      hb7.fst.backward
      hb8.fst.backward
      hb9.fst.backward
      hb10.fst.backward
      hb11.fst.backward
      hb12.fst.backward
      hb13.fst.backward
      hb14.fst.backward
      hb15.fst.backward
      hb16.fst.backward
      (fun i => StableHLO.bnBatchLA N 64 (2 * (2 * (2 * (2 * q)))) (2 * (2 * (2 * (2 * q)))) εs γs βs
        (StableHLO.batchMap N (flatConvStride2 Ws bs) x) i > 0)
      = (r34BFullHasVJPAt (r34StemB N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16
          (r34HeadB N q q Wd bd) x
          ⟨r34StemBHasVJPAt N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) Ws bs εs hεs γs βs
              (by norm_num) (by omega) (by omega) x h_stem h_pool,
            r34StemB_differentiableAt N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) Ws bs εs hεs γs βs
              (by norm_num) (by omega) (by omega) x h_stem h_pool⟩
          hb1 hb2 hb3 hb4 hb5 hb6 hb7 hb8 hb9 hb10 hb11 hb12 hb13 hb14 hb15 hb16
          ⟨(r34HeadBHasVJP N q q Wd bd).toHasVJPAt _,
            (r34HeadB_differentiable N q q Wd bd) _⟩).backward := by
  unfold r50InputGradB
  rw [cbReluStridedBBack_eq_vjp_backward (by decide) (by decide) Ws bs εs hεs γs βs x h_stem,
      r34HeadBBack_eq_vjp_backward Wd bd (opaqueA16 (r34StemB N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 x)]
  funext dy
  rw [r34BFullHasVJPAt_backward, r34StemBHasVJPAt_backward]
  repeat rw [Function.comp_apply]
  rfl

/-- **The chain IS the `pdiv`-contracted Jacobian of the eighteen-stage net** — at every batch
    size, resolution `q > 0` (`hq0`), loss cotangent and input pixel, at any input `x` where the
    stem relu is off its kink (`h_stem`) and no stem-pool window ties (`h_pool`), for any block maps
    `b1 … b16` carrying `HasVJPDiffAt` witnesses at their running activations (`hb1 … hb16`), with
    `0 < εs`. The tie above read through the apex's own `.correct`;
    `resnet50ForwardBFull_eq_slots` below identifies those eighteen stages with the committed
    forward. -/
theorem r50InputGradB_correct (N q : Nat) {nCls : Nat}
    (hq0 : 0 < q)
    (Ws : Kernel4 64 3 7 7) (bs : Vec 64) (εs : ℝ) (hεs : 0 < εs) (γs βs : Vec 64)
    (Wd : Mat 2048 nCls) (bd : Vec nCls)
    (b1 : Vec (N * (64 * (2 * (2 * (2 * q))) * (2 * (2 * (2 * q))))) → Vec (N * (256 * (2 * (2 * (2 * q))) * (2 * (2 * (2 * q))))))
    (b2 : Vec (N * (256 * (2 * (2 * (2 * q))) * (2 * (2 * (2 * q))))) → Vec (N * (256 * (2 * (2 * (2 * q))) * (2 * (2 * (2 * q))))))
    (b3 : Vec (N * (256 * (2 * (2 * (2 * q))) * (2 * (2 * (2 * q))))) → Vec (N * (256 * (2 * (2 * (2 * q))) * (2 * (2 * (2 * q))))))
    (b4 : Vec (N * (256 * (2 * (2 * (2 * q))) * (2 * (2 * (2 * q))))) → Vec (N * (512 * (2 * (2 * q)) * (2 * (2 * q)))))
    (b5 : Vec (N * (512 * (2 * (2 * q)) * (2 * (2 * q)))) → Vec (N * (512 * (2 * (2 * q)) * (2 * (2 * q)))))
    (b6 : Vec (N * (512 * (2 * (2 * q)) * (2 * (2 * q)))) → Vec (N * (512 * (2 * (2 * q)) * (2 * (2 * q)))))
    (b7 : Vec (N * (512 * (2 * (2 * q)) * (2 * (2 * q)))) → Vec (N * (512 * (2 * (2 * q)) * (2 * (2 * q)))))
    (b8 : Vec (N * (512 * (2 * (2 * q)) * (2 * (2 * q)))) → Vec (N * (1024 * (2 * q) * (2 * q))))
    (b9 : Vec (N * (1024 * (2 * q) * (2 * q))) → Vec (N * (1024 * (2 * q) * (2 * q))))
    (b10 : Vec (N * (1024 * (2 * q) * (2 * q))) → Vec (N * (1024 * (2 * q) * (2 * q))))
    (b11 : Vec (N * (1024 * (2 * q) * (2 * q))) → Vec (N * (1024 * (2 * q) * (2 * q))))
    (b12 : Vec (N * (1024 * (2 * q) * (2 * q))) → Vec (N * (1024 * (2 * q) * (2 * q))))
    (b13 : Vec (N * (1024 * (2 * q) * (2 * q))) → Vec (N * (1024 * (2 * q) * (2 * q))))
    (b14 : Vec (N * (1024 * (2 * q) * (2 * q))) → Vec (N * (2048 * q * q)))
    (b15 : Vec (N * (2048 * q * q)) → Vec (N * (2048 * q * q)))
    (b16 : Vec (N * (2048 * q * q)) → Vec (N * (2048 * q * q)))
    (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q))))))))
    (h_stem : R34StemSmoothAt N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) Ws bs εs γs βs x)
    (h_pool : R34PoolSmoothAt N (2 * (2 * (2 * q))) (2 * (2 * (2 * q)))
      (StableHLO.cbReluStridedB N (h := (2 * (2 * (2 * (2 * q))))) (w := (2 * (2 * (2 * (2 * q))))) Ws bs εs γs βs x))
    (hb1 : HasVJPDiffAt b1 (opaqueA0 (r34StemB N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) Ws bs εs γs βs) x))
    (hb2 : HasVJPDiffAt b2 (opaqueA1 (r34StemB N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) Ws bs εs γs βs) b1 x))
    (hb3 : HasVJPDiffAt b3 (opaqueA2 (r34StemB N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) Ws bs εs γs βs) b1 b2 x))
    (hb4 : HasVJPDiffAt b4 (opaqueA3 (r34StemB N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) Ws bs εs γs βs) b1 b2 b3 x))
    (hb5 : HasVJPDiffAt b5 (opaqueA4 (r34StemB N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) Ws bs εs γs βs) b1 b2 b3 b4 x))
    (hb6 : HasVJPDiffAt b6 (opaqueA5 (r34StemB N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) Ws bs εs γs βs) b1 b2 b3 b4 b5 x))
    (hb7 : HasVJPDiffAt b7 (opaqueA6 (r34StemB N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 x))
    (hb8 : HasVJPDiffAt b8 (opaqueA7 (r34StemB N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 x))
    (hb9 : HasVJPDiffAt b9 (opaqueA8 (r34StemB N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 x))
    (hb10 : HasVJPDiffAt b10 (opaqueA9 (r34StemB N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 x))
    (hb11 : HasVJPDiffAt b11 (opaqueA10 (r34StemB N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 x))
    (hb12 : HasVJPDiffAt b12 (opaqueA11 (r34StemB N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 x))
    (hb13 : HasVJPDiffAt b13 (opaqueA12 (r34StemB N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 x))
    (hb14 : HasVJPDiffAt b14 (opaqueA13 (r34StemB N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 x))
    (hb15 : HasVJPDiffAt b15 (opaqueA14 (r34StemB N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 x))
    (hb16 : HasVJPDiffAt b16 (opaqueA15 (r34StemB N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 x))
    (dy : Vec (N * nCls)) (i : Fin (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q)))))))) :
    r50InputGradB N q Ws Wd
      ((bnBatchLAHasVJP N 64 (2 * (2 * (2 * (2 * q)))) (2 * (2 * (2 * (2 * q)))) εs hεs γs βs).backward
        (StableHLO.batchMap N (flatConvStride2 Ws bs) x))
      (StableHLO.cbReluStridedB N (h := (2 * (2 * (2 * (2 * q))))) (w := (2 * (2 * (2 * (2 * q))))) Ws bs εs γs βs x)
      hb1.fst.backward
      hb2.fst.backward
      hb3.fst.backward
      hb4.fst.backward
      hb5.fst.backward
      hb6.fst.backward
      hb7.fst.backward
      hb8.fst.backward
      hb9.fst.backward
      hb10.fst.backward
      hb11.fst.backward
      hb12.fst.backward
      hb13.fst.backward
      hb14.fst.backward
      hb15.fst.backward
      hb16.fst.backward
      (fun i => StableHLO.bnBatchLA N 64 (2 * (2 * (2 * (2 * q)))) (2 * (2 * (2 * (2 * q)))) εs γs βs
        (StableHLO.batchMap N (flatConvStride2 Ws bs) x) i > 0)
      dy i
      = ∑ j : Fin (N * nCls),
          pdiv (r34HeadB N q q Wd bd
          ∘ b16 ∘ b15 ∘ b14 ∘ b13 ∘ b12 ∘ b11 ∘ b10 ∘ b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1
          ∘ r34StemB N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) Ws bs εs γs βs) x i j * dy j := by
  exact HasVJPAt.correct_of_backward_eq _ (r50InputGradB_eq_r34B_full_vjp N q hq0 Ws bs εs hεs γs βs Wd bd
    b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 x h_stem h_pool hb1 hb2 hb3 hb4 hb5 hb6 hb7 hb8 hb9 hb10 hb11 hb12 hb13 hb14 hb15 hb16) dy i

/-- **THE SHAPE CHECK — the eighteen slots the tie is about ARE the committed forward.**
    `resnet50ForwardBFull`, regrouped into exactly the eighteen arguments `r34BFullHasVJPAt`
    takes: ResNet-34's stem, the [3,4,6,3] bottleneck ladder as one stride-1 projection block
    (`s1b0`, the form with no ResNet-34 analogue), three strided projections and twelve identity
    bottlenecks, and ResNet-34's head.

    The tie keeps its blocks opaque, so its subject is a chain of VARIABLES and nothing in it says
    which net they are; this theorem says it. `q` is a binder here too, so it checks both shipped
    resolutions. -/
theorem resnet50ForwardBFull_eq_slots (N q : Nat) {nCls : Nat} (w : R50BWeights nCls)
    (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q)))))))) :
    resnet50ForwardBFull N q w x
      = (r34HeadB N q q w.Wd w.bd
          ∘ r50IdB N q q w.s4b2
          ∘ r50IdB N q q w.s4b1
          ∘ r50DownB N q q w.s4b0
          ∘ r50IdB N (2 * q) (2 * q) w.s3b5
          ∘ r50IdB N (2 * q) (2 * q) w.s3b4
          ∘ r50IdB N (2 * q) (2 * q) w.s3b3
          ∘ r50IdB N (2 * q) (2 * q) w.s3b2
          ∘ r50IdB N (2 * q) (2 * q) w.s3b1
          ∘ r50DownB N (2 * q) (2 * q) w.s3b0
          ∘ r50IdB N (2 * (2 * q)) (2 * (2 * q)) w.s2b3
          ∘ r50IdB N (2 * (2 * q)) (2 * (2 * q)) w.s2b2
          ∘ r50IdB N (2 * (2 * q)) (2 * (2 * q)) w.s2b1
          ∘ r50DownB N (2 * (2 * q)) (2 * (2 * q)) w.s2b0
          ∘ r50IdB N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) w.s1b2
          ∘ r50IdB N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) w.s1b1
          ∘ r50ProjB N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) w.s1b0
          ∘ r34StemB N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) w.sW w.sb w.sε w.sγ w.sβ) x := by
  -- `resnet50ForwardBFull_eq_chain` for the depth-16 half, then unfold the named prefixes
  rw [resnet50ForwardBFull_eq_chain N q w x]
  simp only [r50Pre16, r50Pre15, r50Pre14, r50Pre13, r50Pre12, r50Pre11, r50Pre10, r50Pre9, r50Pre8, r50Pre7, r50Pre6, r50Pre5, r50Pre4, r50Pre3, r50Pre2, r50Pre1, r50Pre0]

end Proofs

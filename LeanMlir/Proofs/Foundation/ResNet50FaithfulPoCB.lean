import LeanMlir.Proofs.Foundation.ResNet34FaithfulPoCB

/-! # T3 §1 fold for ResNet-50 — every parameter GRADIENT node, by block profile

`ResNet50FullB.lean` gives ResNet-50 its ℝ forward and typed graph (T1, T2). This is the first
half of T3: every parameter gradient node the batched train step emits denotes the certified
gradient, for an arbitrary cotangent.

⭐⭐ **Zero new op-kind lemmas, and that is the finding.** 4b's biggest lesson was that op kinds
are shared across nets far more than the per-net file names suggest — five of EfficientNet-B0's
eight lemmas turned out to be `ResNet34FaithfulPoCB.lean`'s. For ResNet-50 it is **six of six**:
that file's `convWGradB_den`, `convStridedWGradB_den`, `bnGammaGradB_den`, `bnBetaGradB_den`,
`denseWGradB_den` and `denseBGradB_den` are statements about OP KINDS at full generality in
`{N ic oc h w kH kW}`, and the bottleneck's third convolution is one more instance of the first.
So this file is an ENUMERATION of the artifact's op table by block profile, not new mathematics.

⛔ **ResNet-50 emits no conv BIAS gradient at all.** `ResNet34FaithfulPoCB.lean` and
`MobileNetV2FaithfulPoCPaperG.lean` both carry bias conjuncts to cover the `convBias := true`
render; `ResNet50RenderB` has no such flag — its `zb` bakes `false` — so `convBiasGradB` and
`convStridedBiasGradB` are never emitted here and there is nothing to state. That is why this
net's table is six op kinds where ResNet-34's is eight.

## The op table, per profile

| profile | where | conv-weight nodes | BN pairs | count |
|---|---|---|---|---|
| stem | 7×7/s2, 3 → 64 | `convStridedWeightGradB` | 1 | 3 |
| identity bottleneck | 12 blocks | 1×1, 3×3, 1×1 — all `convWeightGradB` | 3 | 9 |
| stride-1 projection | stage 1 block 0 | the same three, plus a 1×1 `convWeightGradB` skip | 4 | 12 |
| strided projection | stages 2/3/4 block 0 | 1×1 at `2h`, **`convStridedWeightGradB`** 3×3, 1×1, **strided** 1×1 skip | 4 | 12 |
| head | GAP → dense | `denseWeightGradB` + `denseBiasGradB` | — | 2 |

3 + 12×9 + 4×12 + 2 = **161**, which is `ResNet50RenderB`'s own "161 θ / 161 m / 161 v" and the
162-argument signature of `verified_mlir/resnet50_fwd.mlir` minus `%x`.

⚠ **The BatchNorm γ/β lemmas are width-generic and are stated ONCE** (`r50BnGradsCertified`), not
per profile: `bnGammaGradB_den` and `bnBetaGradB_den` take `{N oc h w}` as binders, so every one of
the net's 53 BatchNorm sites is an instance. Only the convolution nodes vary in kind, because only
they carry a kernel shape and a stride.

⚠⚠ **The strided nodes are SYMMETRIC padding** — `convStridedWeightGradB`, whose `den` is
`flatConvStride2_*`, not B0's XLA-`SAME` `convStridedXla*` peers. Identical types, identical emitted
shapes, different certificates. ResNet-50 is a PyTorch-origin net.

⚠⚠ **v1.5: the strided node in a downsample block is `W2`, the 3×3** — `W1` is an ordinary
`convWeightGradB` at the INPUT grid `2h × 2w`. Putting the stride on `W1` is ResNet v1, a different
net, and nothing in the types would notice.

## Honest residual

* The cotangents are free variables — every conjunct is `∀ cot`, so each holds at the actual
  backward-chain cotangent without naming it. Pinning them to the emitted backward subgraph is the
  §1a tie.
* ⛔ **One replica.** Under `*dp*` each of these nodes is followed by `all_reduce(add)/R` as
  emitted text outside the AST, so the statement is at the per-replica gradient
  (`Foundation/DataParallel.lean`, §4d).
-/

open Proofs Proofs.StableHLO Proofs.IR

namespace Proofs.ResNet50PoCB

open scoped BigOperators

-- ════════════════════════════════════════════════════════════════
-- § BatchNorm — one statement for all 53 sites
-- ════════════════════════════════════════════════════════════════

/-- **Every BatchNorm γ and β gradient node denotes the certified per-channel gradient**, at the
    merged batch+spatial width `m = N·(h·w)`. Width-generic, so the net's 53 sites — three per
    bottleneck, one per projection skip, one in the stem — are all instances of these two. -/
theorem r50BnGradsCertified {N oc h w : Nat} :
    (∀ (vN epsStr cotN : String) (ε : ℝ) (γ β : Vec oc)
       (v : Vec (N * (oc * (h * w)))) (cot : Vec (N * (oc * (h * w)))) (c : Fin oc),
        den (SHlo.bnGammaGradB vN epsStr ε v (.operand cotN cot)) c
          = ∑ j : Fin (oc * (N * (h * w))),
              pdiv (fun γ' : Vec oc =>
                      bnPerChannelFlat oc (N * (h * w)) ε γ' β (bnchwFwd N oc h w v))
                   γ c j * bnchwFwd N oc h w cot j) ∧
    (∀ (cotN : String) (ε : ℝ) (γ β : Vec oc)
       (v : Vec (oc * (N * (h * w)))) (cot : Vec (N * (oc * (h * w)))) (c : Fin oc),
        den (SHlo.bnBetaGradB (N := N) (oc := oc) (h := h) (w := w) (.operand cotN cot)) c
          = ∑ j : Fin (oc * (N * (h * w))),
              pdiv (fun β' : Vec oc => bnPerChannelFlat oc (N * (h * w)) ε γ β' v)
                   β c j * bnchwFwd N oc h w cot j) :=
  ⟨fun vN epsStr cotN ε γ β v cot c => ResNet34PoCB.bnGammaGradB_den vN epsStr cotN ε γ β v cot c,
   fun cotN ε γ β v cot c => ResNet34PoCB.bnBetaGradB_den cotN ε γ β v cot c⟩

-- ════════════════════════════════════════════════════════════════
-- § The stem — 7x7/s2, SYMMETRIC padding
-- ════════════════════════════════════════════════════════════════

/-- **The stem's weight gradient node denotes the certified `Σ_n` weight gradient.** The one place
    in ResNet-50 where a 7×7 kernel appears, and the same op kind the downsample skips use —
    `convStridedWGradB_den` is generic in the kernel size. -/
theorem r50StemGradsCertified {N ic oc h w kH kW : Nat} :
    ∀ (xN cotN : String) (b : Vec oc) (x : Vec (N * (ic * (2 * h) * (2 * w))))
      (W : Kernel4 oc ic kH kW) (cot : Vec (N * (oc * h * w))) (idx : Fin (oc * ic * kH * kW)),
      den (SHlo.convStridedWeightGradB xN b x W (.operand cotN cot)) idx
        = ∑ n : Fin N, ∑ j : Fin (oc * h * w),
            pdiv (fun v' : Vec (oc * ic * kH * kW) =>
                    flatConvStride2 (Kernel4.unflatten v') b
                      (batchSlice N (ic * (2 * h) * (2 * w)) x n))
                 (Kernel4.flatten W) idx j * batchSlice N (oc * h * w) cot n j :=
  fun xN cotN b x W cot idx => ResNet34PoCB.convStridedWGradB_den xN cotN b x W cot idx

-- ════════════════════════════════════════════════════════════════
-- § Identity bottleneck — 12 blocks, three stride-1 convs
-- ════════════════════════════════════════════════════════════════

/-- **The identity bottleneck's three weight gradient nodes all denote the certified gradient**:
    the 1×1 reduce `oc → mid`, the 3×3 `mid → mid` and the 1×1 expand `mid → oc`, all at `h × w`.
    With `r50BnGradsCertified`'s pair at the three BatchNorm sites that is the block's nine
    parameters. -/
theorem r50IdGradsCertified {N mid oc h w : Nat} :
    (∀ (xN cotN : String) (b : Vec mid) (x : Vec (N * (oc * h * w))) (W : Kernel4 mid oc 1 1)
       (cot : Vec (N * (mid * h * w))) (idx : Fin (mid * oc * 1 * 1)),
        den (SHlo.convWeightGradB xN b x W (.operand cotN cot)) idx
          = ∑ n : Fin N, ∑ j : Fin (mid * h * w),
              pdiv (fun v' : Vec (mid * oc * 1 * 1) =>
                      Tensor3.flatten (conv2d (Kernel4.unflatten v') b
                        (Tensor3.unflatten (batchSlice N (oc * h * w) x n))))
                   (Kernel4.flatten W) idx j * batchSlice N (mid * h * w) cot n j) ∧
    (∀ (xN cotN : String) (b : Vec mid) (x : Vec (N * (mid * h * w))) (W : Kernel4 mid mid 3 3)
       (cot : Vec (N * (mid * h * w))) (idx : Fin (mid * mid * 3 * 3)),
        den (SHlo.convWeightGradB xN b x W (.operand cotN cot)) idx
          = ∑ n : Fin N, ∑ j : Fin (mid * h * w),
              pdiv (fun v' : Vec (mid * mid * 3 * 3) =>
                      Tensor3.flatten (conv2d (Kernel4.unflatten v') b
                        (Tensor3.unflatten (batchSlice N (mid * h * w) x n))))
                   (Kernel4.flatten W) idx j * batchSlice N (mid * h * w) cot n j) ∧
    (∀ (xN cotN : String) (b : Vec oc) (x : Vec (N * (mid * h * w))) (W : Kernel4 oc mid 1 1)
       (cot : Vec (N * (oc * h * w))) (idx : Fin (oc * mid * 1 * 1)),
        den (SHlo.convWeightGradB xN b x W (.operand cotN cot)) idx
          = ∑ n : Fin N, ∑ j : Fin (oc * h * w),
              pdiv (fun v' : Vec (oc * mid * 1 * 1) =>
                      Tensor3.flatten (conv2d (Kernel4.unflatten v') b
                        (Tensor3.unflatten (batchSlice N (mid * h * w) x n))))
                   (Kernel4.flatten W) idx j * batchSlice N (oc * h * w) cot n j) :=
  ⟨fun xN cotN b x W cot idx => ResNet34PoCB.convWGradB_den xN cotN b x W cot idx,
   fun xN cotN b x W cot idx => ResNet34PoCB.convWGradB_den xN cotN b x W cot idx,
   fun xN cotN b x W cot idx => ResNet34PoCB.convWGradB_den xN cotN b x W cot idx⟩

-- ════════════════════════════════════════════════════════════════
-- § Stride-1 projection bottleneck — stage 1 block 0 ONLY
-- ════════════════════════════════════════════════════════════════

/-- ⭐ **The stride-1 projection bottleneck's four weight gradient nodes.** The body's three are
    `r50IdGradsCertified`'s at `ic → mid → mid → oc`; the fourth is the 1×1 skip, and it is an
    ORDINARY `convWeightGradB` — the resolution does not change here, which is the whole point of
    this block form. ⛔ Reaching for the strided op is the mistake: it would be a shape error on
    the forward and is silently a different gradient here. -/
theorem r50ProjGradsCertified {N ic mid oc h w : Nat} :
    (∀ (xN cotN : String) (b : Vec mid) (x : Vec (N * (ic * h * w))) (W : Kernel4 mid ic 1 1)
       (cot : Vec (N * (mid * h * w))) (idx : Fin (mid * ic * 1 * 1)),
        den (SHlo.convWeightGradB xN b x W (.operand cotN cot)) idx
          = ∑ n : Fin N, ∑ j : Fin (mid * h * w),
              pdiv (fun v' : Vec (mid * ic * 1 * 1) =>
                      Tensor3.flatten (conv2d (Kernel4.unflatten v') b
                        (Tensor3.unflatten (batchSlice N (ic * h * w) x n))))
                   (Kernel4.flatten W) idx j * batchSlice N (mid * h * w) cot n j) ∧
    (∀ (xN cotN : String) (b : Vec mid) (x : Vec (N * (mid * h * w))) (W : Kernel4 mid mid 3 3)
       (cot : Vec (N * (mid * h * w))) (idx : Fin (mid * mid * 3 * 3)),
        den (SHlo.convWeightGradB xN b x W (.operand cotN cot)) idx
          = ∑ n : Fin N, ∑ j : Fin (mid * h * w),
              pdiv (fun v' : Vec (mid * mid * 3 * 3) =>
                      Tensor3.flatten (conv2d (Kernel4.unflatten v') b
                        (Tensor3.unflatten (batchSlice N (mid * h * w) x n))))
                   (Kernel4.flatten W) idx j * batchSlice N (mid * h * w) cot n j) ∧
    (∀ (xN cotN : String) (b : Vec oc) (x : Vec (N * (mid * h * w))) (W : Kernel4 oc mid 1 1)
       (cot : Vec (N * (oc * h * w))) (idx : Fin (oc * mid * 1 * 1)),
        den (SHlo.convWeightGradB xN b x W (.operand cotN cot)) idx
          = ∑ n : Fin N, ∑ j : Fin (oc * h * w),
              pdiv (fun v' : Vec (oc * mid * 1 * 1) =>
                      Tensor3.flatten (conv2d (Kernel4.unflatten v') b
                        (Tensor3.unflatten (batchSlice N (mid * h * w) x n))))
                   (Kernel4.flatten W) idx j * batchSlice N (oc * h * w) cot n j) ∧
    (∀ (xN cotN : String) (b : Vec oc) (x : Vec (N * (ic * h * w))) (W : Kernel4 oc ic 1 1)
       (cot : Vec (N * (oc * h * w))) (idx : Fin (oc * ic * 1 * 1)),
        den (SHlo.convWeightGradB xN b x W (.operand cotN cot)) idx
          = ∑ n : Fin N, ∑ j : Fin (oc * h * w),
              pdiv (fun v' : Vec (oc * ic * 1 * 1) =>
                      Tensor3.flatten (conv2d (Kernel4.unflatten v') b
                        (Tensor3.unflatten (batchSlice N (ic * h * w) x n))))
                   (Kernel4.flatten W) idx j * batchSlice N (oc * h * w) cot n j) :=
  ⟨fun xN cotN b x W cot idx => ResNet34PoCB.convWGradB_den xN cotN b x W cot idx,
   fun xN cotN b x W cot idx => ResNet34PoCB.convWGradB_den xN cotN b x W cot idx,
   fun xN cotN b x W cot idx => ResNet34PoCB.convWGradB_den xN cotN b x W cot idx,
   fun xN cotN b x W cot idx => ResNet34PoCB.convWGradB_den xN cotN b x W cot idx⟩

-- ════════════════════════════════════════════════════════════════
-- § Strided projection bottleneck — stages 2/3/4 block 0
-- ════════════════════════════════════════════════════════════════

/-- **The strided projection bottleneck's four weight gradient nodes**, and TWO of them are the
    strided op.

    ⚠⚠ v1.5: `W1` is an ORDINARY `convWeightGradB` at the input grid `2h × 2w`, `W2` is the
    strided 3×3, `W3` an ordinary 1×1 at `h × w`, and `Wp` the strided 1×1 skip. Moving the stride
    to `W1` is ResNet v1 — a different net that compiles, trains and descends, and nothing in the
    types would notice. ⚠ Both strided nodes are SYMMETRIC padding. -/
theorem r50DownGradsCertified {N ic mid oc h w : Nat} :
    (∀ (xN cotN : String) (b : Vec mid) (x : Vec (N * (ic * (2 * h) * (2 * w))))
       (W : Kernel4 mid ic 1 1) (cot : Vec (N * (mid * (2 * h) * (2 * w))))
       (idx : Fin (mid * ic * 1 * 1)),
        den (SHlo.convWeightGradB xN b x W (.operand cotN cot)) idx
          = ∑ n : Fin N, ∑ j : Fin (mid * (2 * h) * (2 * w)),
              pdiv (fun v' : Vec (mid * ic * 1 * 1) =>
                      Tensor3.flatten (conv2d (Kernel4.unflatten v') b
                        (Tensor3.unflatten (batchSlice N (ic * (2 * h) * (2 * w)) x n))))
                   (Kernel4.flatten W) idx j
                * batchSlice N (mid * (2 * h) * (2 * w)) cot n j) ∧
    (∀ (xN cotN : String) (b : Vec mid) (x : Vec (N * (mid * (2 * h) * (2 * w))))
       (W : Kernel4 mid mid 3 3) (cot : Vec (N * (mid * h * w))) (idx : Fin (mid * mid * 3 * 3)),
        den (SHlo.convStridedWeightGradB xN b x W (.operand cotN cot)) idx
          = ∑ n : Fin N, ∑ j : Fin (mid * h * w),
              pdiv (fun v' : Vec (mid * mid * 3 * 3) =>
                      flatConvStride2 (Kernel4.unflatten v') b
                        (batchSlice N (mid * (2 * h) * (2 * w)) x n))
                   (Kernel4.flatten W) idx j * batchSlice N (mid * h * w) cot n j) ∧
    (∀ (xN cotN : String) (b : Vec oc) (x : Vec (N * (mid * h * w))) (W : Kernel4 oc mid 1 1)
       (cot : Vec (N * (oc * h * w))) (idx : Fin (oc * mid * 1 * 1)),
        den (SHlo.convWeightGradB xN b x W (.operand cotN cot)) idx
          = ∑ n : Fin N, ∑ j : Fin (oc * h * w),
              pdiv (fun v' : Vec (oc * mid * 1 * 1) =>
                      Tensor3.flatten (conv2d (Kernel4.unflatten v') b
                        (Tensor3.unflatten (batchSlice N (mid * h * w) x n))))
                   (Kernel4.flatten W) idx j * batchSlice N (oc * h * w) cot n j) ∧
    (∀ (xN cotN : String) (b : Vec oc) (x : Vec (N * (ic * (2 * h) * (2 * w))))
       (W : Kernel4 oc ic 1 1) (cot : Vec (N * (oc * h * w))) (idx : Fin (oc * ic * 1 * 1)),
        den (SHlo.convStridedWeightGradB xN b x W (.operand cotN cot)) idx
          = ∑ n : Fin N, ∑ j : Fin (oc * h * w),
              pdiv (fun v' : Vec (oc * ic * 1 * 1) =>
                      flatConvStride2 (Kernel4.unflatten v') b
                        (batchSlice N (ic * (2 * h) * (2 * w)) x n))
                   (Kernel4.flatten W) idx j * batchSlice N (oc * h * w) cot n j) :=
  ⟨fun xN cotN b x W cot idx => ResNet34PoCB.convWGradB_den xN cotN b x W cot idx,
   fun xN cotN b x W cot idx => ResNet34PoCB.convStridedWGradB_den xN cotN b x W cot idx,
   fun xN cotN b x W cot idx => ResNet34PoCB.convWGradB_den xN cotN b x W cot idx,
   fun xN cotN b x W cot idx => ResNet34PoCB.convStridedWGradB_den xN cotN b x W cot idx⟩

-- ════════════════════════════════════════════════════════════════
-- § The head — dense weight and bias
-- ════════════════════════════════════════════════════════════════

/-- **The head's two gradient nodes denote the certified `Σ_n` outer product and cotangent sum.**
    Generic in the class count, so one statement covers the 10-class Imagenette artifacts and the
    1000-class `resnet50in` ones. -/
theorem r50HeadGradsCertified {N a c : Nat} :
    (∀ (xN cotN : String) (x : Vec (N * a)) (W : Mat a c) (b : Vec c) (cot : Vec (N * c))
       (i : Fin a) (j : Fin c),
        den (SHlo.denseWeightGradB (c := c) xN x (.operand cotN cot)) (finProdFinEquiv (i, j))
          = ∑ n : Fin N, ∑ k : Fin c,
              pdiv (fun v : Vec (a * c) => dense (Mat.unflatten v) b (batchSlice N a x n))
                   (Mat.flatten W) (finProdFinEquiv (i, j)) k * batchSlice N c cot n k) ∧
    (∀ (cotN : String) (W : Mat c c) (x : Vec c) (b : Vec c) (cot : Vec (N * c)) (j : Fin c),
        den (SHlo.denseBiasGradB (N := N) (.operand cotN cot)) j
          = ∑ n : Fin N, ∑ k : Fin c,
              pdiv (fun b' : Vec c => dense W b' x) b j k * batchSlice N c cot n k) :=
  ⟨fun xN cotN x W b cot i j => ResNet34PoCB.denseWGradB_den xN cotN x W b cot i j,
   fun cotN W x b cot j => ResNet34PoCB.denseBGradB_den cotN W x b cot j⟩

end Proofs.ResNet50PoCB

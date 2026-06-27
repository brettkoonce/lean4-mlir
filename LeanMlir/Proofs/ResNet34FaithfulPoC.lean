import LeanMlir.Proofs.ResNet34Close
import LeanMlir.Proofs.Cifar8FaithfulPoC
import LeanMlir.Proofs.CifarBnFaithfulPoC

/-! # PoC: the ResNet-34 (Chapter 5) train step, proof-tied to the certified SGD step

The Chapter-5 capstone — the full `[3,4,6,3]` ResNet-34 (146 params: a 7×7/s2 stem, 16
residual blocks, GAP + final dense). `MainResnet34Verified` trains on
`verified_mlir/resnet34_train_step.mlir`; this file makes its parameter updates
`den`-faithful — each emitted SGD op denotes the certified loss-descent step.

**Two new core ops, ZERO new theorems for 142 of the 146 params.** Like cifar8-bn, the
overwhelming majority of ResNet-34's parameter outputs fold by *reusing the existing generic
`den = certified` lemmas*:

* **3×3 stride-1 block convs** (identity blocks `W1`/`W2`, downsample body `W2`):
  `CifarPoC.convW_den`/`convB_den` (the `conv2d` weight/bias VJP is dim- and cotangent-generic).
* **per-channel BN γ/β** (every block): `CifarBnPoC.bnGamma_den`/`bnBeta_den` (the `oc·h·w ↔ oc·m`
  reassoc bridge).
* **final dense `Wd`/`bd`**: `Cifar8PoC.denseW_den`/`denseB_den`.

The genuinely-new shapes are the **strided convolutions** — the 7×7/s2 stem and the 3×3/s2
downsample bodies + projection skips — which no prior fold exercised through an SGD op. They get
the two new core ops `convStridedWeightSgd`/`convStridedBiasSgd` (StableHLO.lean) and the two
`den = certified` lemmas below. Both are *one-line delegations* to the generic strided bridge
`mnv2_render_stem_conv{W,b}_certified` (the same bridge ResNet34Close pins to r34's kernel sizes
via `r34_render_{down,stem}Conv{W,b}_certified`), exactly mirroring `CifarPoC.convW_den`'s
delegation to `cnn_render_convW_certified`.

* **`convStridedWeightSgd`** emits the strided weight-grad text (zero-upsample the cotangent —
  the decimate-backward — then the SAME transpose-trick stride-1 weight-grad conv on the 2h×2w
  grid). Its `den` reduces (`rfl`) to `flatten W − lr·(flatConvStride2_weight_grad · c)`, the LHS
  of `mnv2_render_stem_convW_certified` — generic in `kH/kW`, so the *single* lemma below certifies
  the 7×7 stem AND every 3×3 strided conv (downsample `W1` + projection `Wp`).
* **`convStridedBiasSgd`** — the bias grad is stride-INDEPENDENT (`Σ_{batch,spatial} dy`), so it
  emits the same `reduce` op text as `convBiasSgd` (its `skel` aliases that op); only its `den`
  differs (the strided VJP), closing via `mnv2_render_stem_convb_certified`.

## Honest residual (same boundary as every prior fold)
* The conv cotangents here are free variables `c` (each lemma is ∀ c) — they hold at the actual
  backward-chain cotangent the renderer feeds, without naming it. Pinning each `c` to the exact
  emitted residual-backward subgraph (the cotangent-sum at each skip merge) is the remaining polish.
* Per-op `pretty` lexing + BN `0 < ε` smoothness + ℝ → Float32.
-/

open Proofs Proofs.StableHLO

namespace Proofs.ResNet34PoC

/-! ## Strided convolutions — the two new `den = certified` lemmas (the only new content) -/

/-- **Any emitted STRIDED conv weight op = certified.** Generic in the conv dims, the kernel size
    (covers the 7×7 stem AND every 3×3 downsample/projection) and the cotangent `c`: the
    `convStridedWeightSgd` op denotes `flatten W − lr·(certified ∂(flatConvStride2)/∂W · c)`, the
    emitted op's `den` reduced (`rfl`) to the LHS of the generic strided weight bridge. The strided
    peer of `CifarPoC.convW_den`. -/
theorem convStridedW_den {ic oc h w kH kW : Nat}
    (xN wN lrStr cotN : String) (b : Vec oc) (x : Vec (ic*(2*h)*(2*w)))
    (W : Kernel4 oc ic kH kW) (c : Vec (oc*h*w)) (lr : ℝ) (idx : Fin (oc*ic*kH*kW)) :
    den (SHlo.convStridedWeightSgd xN wN lrStr b x W lr (.operand cotN c)) idx
      = Kernel4.flatten W idx - lr * ∑ j : Fin (oc*h*w),
          pdiv (fun v' : Vec (oc*ic*kH*kW) => flatConvStride2 (Kernel4.unflatten v') b x)
               (Kernel4.flatten W) idx j * c j :=
  mnv2_render_stem_convW_certified b x (Kernel4.flatten W) c lr idx

/-- **Any emitted STRIDED conv bias op = certified.** The bias peer of `convStridedW_den`; the
    `convStridedBiasSgd` op (which emits the same `reduce` text as `convBiasSgd`) denotes
    `b − lr·(certified ∂(flatConvStride2)/∂b · c)`. -/
theorem convStridedB_den {ic oc h w kH kW : Nat}
    (bN lrStr cotN : String) (W : Kernel4 oc ic kH kW) (x : Vec (ic*(2*h)*(2*w)))
    (b : Vec oc) (c : Vec (oc*h*w)) (lr : ℝ) (o : Fin oc) :
    den (SHlo.convStridedBiasSgd bN lrStr W x b lr (.operand cotN c)) o
      = b o - lr * ∑ j : Fin (oc*h*w),
          pdiv (fun b' : Vec oc => flatConvStride2 W b' x) b o j * c j :=
  mnv2_render_stem_convb_certified W x b c lr o

end Proofs.ResNet34PoC

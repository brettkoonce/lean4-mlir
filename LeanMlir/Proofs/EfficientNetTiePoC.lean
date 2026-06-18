import LeanMlir.Proofs.EfficientNetFaithfulPoC

/-! # PoC: the full-16 EfficientNet-B0 train step §1a TIE (whole-net thread) — IN PROGRESS

The EfficientNet peer of `MobileNetV2TiePoCPaper` (the §1a tie). The §1 fold
(`EfficientNetFaithfulPoC`) gives every batched param-SGD op `den = certified ∀ cotangent`; this
file pins each cotangent to the **actual loss-driven backward chain** of the rendered net — threading
the real forward activations through every param op and composing the backward cotangent from the
loss down through all 16 MBConv blocks (with the residual fan-in at every stride-1 skip AND the SE
gate fan-in), so each output's `den = certified` becomes a single composed theorem with the forward
= the proven `efficientnetForwardB`.

**What is NEW vs mnv2's tie** (the harder content, hence a dedicated effort):
* **swish** masks instead of relu6 — the cotangent crosses `swishBack` (smooth, no two-kink
  `selectMid`) at every conv-bn-swish / depthwise-bn-swish stage.
* the **SE multiplicative gate fan-in** — the cotangent at an SE input is `gate ⊙ dyOut` (the
  fused `seBackBatched` value) and the SE dense param cotangents come from `seReduceB` (the gate
  cotangent `Σ_{h,w}(x⊙dy)`) threaded back through `sigmoidBack → denseRowBack(W₂) → swishBack`.
* **true batch-norm** backward (`bnBatchBack`) — batch-coupled, vs mnv2's per-example BN.
* `EfficientNetChainClose`'s whole-net backward is **HasVJP-composition** style (`vjp_comp` of the
  per-block `_has_vjp`), NOT explicit cotangent-vector defs like mnv2's `invresCot*` — so the tie
  must BUILD explicit chain-cot constructors (the bulk of the remaining work).

## Landed so far
* `efficientnetLossCot_den` — the emitted loss-cotangent graph (`softmaxRowF − %onehot`, the batched
  per-row softmax-CE gradient) denotes `rowSoftmax(logits) − onehot`. The top of the cotangent chain.

## Remaining (the substantial thread — a dedicated session, à la `MobileNetV2TiePoCPaper`'s 851 lines)
1. Per-block-type tie defs+theorems (`enet{NoExp,Stride1,Strided,Head,Stem}Tied`) — each block's 12/16
   param ops instantiated at the block's real activations + its chain cotangents (the §1-fold generics
   `EnetPoC.*`, generic in the cotangent, supply the per-op step). The SE four (`zW1/zb1/zW2/zb2`) tie at
   the `seReduceB → sigmoidBack → denseRowBack` gate-cot chain.
2. The chain-cot constructors (`@[irreducible]`): `swishBack`/`bnBatchBack`/`convBackBatched`/
   `depthwiseBackBatched`/`seBackBatched`+`seReduceB` composed per stage; the residual `+ dyOut` and SE
   gate fan-ins.
3. The dense-head total-loss fold (`Wd → ∂CE/∂Wd`) — needs the batched-`Σ_n` analogue of
   `mlp_output_total_loss_grad` (per-row softmax-CE through the batched dense).
4. The whole-net thread `efficientnet_net_tied_certified` over the 16 blocks + stem + head, threading
   the full forward `efficientnetForwardB_full`, with `@[irreducible]` `FwdO`/`CotInAt`/`TiedAt`
   wrappers to dodge the heartbeat blowup (the r34/mnv2 lesson, more acute at 16 blocks + SE). -/

open Proofs Proofs.StableHLO

namespace Proofs.EnetTiePoC

open scoped BigOperators

/-- **The emitted loss-cotangent graph denotes the (batched, per-row) softmax-CE gradient at the
    logits.** The renderer's `sub (softmaxRowF logits) %onehot`; `rowSoftmaxFlat` is the per-row
    softmax over the `n=10` classes. The top of the §1a cotangent chain (generic in the batch unit
    `m`; the renderer instantiates `m=1`). -/
theorem efficientnetLossCot_den {m : Nat} (nlogN ohN : String) (logits oh : Vec (m * 10)) :
    den (SHlo.sub (SHlo.softmaxRowF (m := m) (n := 10) (.operand nlogN logits)) (.operand ohN oh))
      = fun idx => rowSoftmaxFlat m 10 logits idx - oh idx := by
  funext idx
  simp only [den_sub, softmaxRowF_faithful, den_operand]

end Proofs.EnetTiePoC

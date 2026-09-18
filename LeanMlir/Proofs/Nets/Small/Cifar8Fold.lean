import LeanMlir.Proofs.Nets.Small.CifarFold

/-! # PoC: the deeper 8-conv CIFAR (cifar8, no-BN) train step, proof-tied

The 4-stage (8-conv) peer of `CifarFold`: `(conv→relu)×2 → pool, four times,
→ (dense→relu)×2 → dense` — 22 params (8 conv kernels/biases, 3 dense layers).
`MainCifar8Verified` trains on `verified_mlir/cifar8_train_step.mlir`.

**Zero new core ops, and zero new proof.** Every conv layer is covered by the *generic*
`CifarPoC.convW_den`/`convB_den` (dim- and cotangent-generic — they certify W₁…W₈ by
instantiation), and the three dense layers by the generic `denseW_den`/`denseB_den`
(`MlpTrainStep.lean`: free in the activation, weight, bias and cotangent, via the M2
`weight_grad_bridge` / `bias_grad_bridge` at `Back.cotangent`).

Residual: as the non-BN cifar fold (conv/dense cotangents are free vars; cotangent-
subgraph⇄SHlo pin; per-op `pretty` lexing; ℝ→Float32).
-/


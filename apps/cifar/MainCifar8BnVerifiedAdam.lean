import apps.cifar.Cifar8BnAdamCommon

/-! # `cifar8-bn-verified-adam` — the 8-conv CIFAR-10 CNN **+ per-channel BN** with **AdamW**

The Adam peer of `cifar8-bn-verified` (SGD), and the BN half of the BN/noBN × SGD/Adam
ablation. Same proof-rendered forward + backward + param gradients as `cifar8BnVerified`
(whole-net VJP `Proofs.cifarCnnBn8_has_vjp_at`, 8× per-channel BN), with the SGD update
swapped for AdamW via `ViTRender.emitAdamV` and driven by `VerifiedNet.trainAdamSched`:
`[θ|m|v]` (38 params: 22 conv/dense + 16 BN γ/β) packed + runtime `lr`/`bc₁`/`bc₂`. Trains
on `verified_mlir/cifar8_bn_adam_train_step.mlir`, rendered as `pretty(provenGraph)` by
`LeanMlir/Proofs/Codegen/CnnRender.lean` (§2i — it was `tests/`-written until 2026-07-30).

Per-channel BN is per-example ⇒ train=eval (no running stats), so `bnChannels` stays empty
and the γ/β are Adam-updated like any other param; eval is plain `@cifar8_bn_fwd`. AdamW
lr 1e-3, β₁ .9, β₂ .999, wd 1e-4 (baked), 3-epoch warmup + cosine decay.

The body is shared with the XLA build (`cifar8-bn-verified-adam-xla`); see
`apps/cifar/Cifar8BnAdamCommon.lean` and `planning/xla_pjrt_ladder.md`.

Run (GPU): `IREE_BACKEND=rocm .lake/build/bin/cifar8-bn-verified-adam data`
-/

def main (argv : List String) : IO Unit := runCifar8BnAdam argv

import LeanMlir.Proofs.Codegen.CnnRender

/-! # The chapter 3–4 artifact writers

Each `#eval` below writes one committed `verified_mlir/` file (the MNIST CNN, the CIFAR CNN and
the cifar8 family) from `CnnRender.lean`'s faithful renderers, when this module is elaborated.
Nothing imports this file: the proofs import `CnnRender`, so building them never rewrites an
artifact. `scripts/regen_verified_mlir.sh proofs` and the proofs.yml drift guard both elaborate
it. -/

-- Regenerate `verified_mlir/cnn_train_step.mlir` (what MainMnistCnnVerified trains on)
-- from the faithful renderer; the den-certified proofs live in CnnFold.lean.
-- Dims `128 1 32 14 14 512 10 3 3`: B=128, ic=1, c=32, h=w=14 (post-pool,
-- image 28×28), d1=512, nClasses=10, 3×3 kernels; lr = 0.1/128 (mean-loss equiv).
#eval IO.FS.writeFile "verified_mlir/cnn_train_step.mlir"
  (Proofs.StableHLO.cnnTrainStepFaithfulV 128 1 32 14 14 512 10 3 3 "0.00078125"
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0)
    (fun _ => 0))

-- Regenerate `verified_mlir/cifar_train_step.mlir` (what MainCifarVerified trains on)
-- from the faithful renderer; the den-certified proofs live in CifarFold.lean.
-- Dims `128 3 32 64 8 8 512 10 3 3`: B=128, ic=3, c1=32, c2=64, h=w=8
-- (final pooled, image 32×32), d1=512, nClasses=10, 3×3 kernels; lr = 0.1/128.
#eval IO.FS.writeFile "verified_mlir/cifar_train_step.mlir"
  (Proofs.StableHLO.cifarTrainStepFaithfulV 128 3 32 64 8 8 512 10 3 3 "0.00078125"
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0)
    (fun _ => 0))

-- Regenerate `verified_mlir/cifar8_train_step.mlir` (what MainCifar8Verified trains on)
-- from the faithful renderer; the den-certified proofs live in Cifar8StepTie.lean.
-- Dims `128 3 16 16 32 32 2 2 64 10 3 3`: h=w=2 (final pooled, image 32×32).
-- Regenerate `verified_mlir/cifar8_adam_train_step.mlir` — the AdamW peer, same forward/backward
-- with the fused SGD tail replaced by un-fused gradients + the proven AdamW ops
-- (planning/archive/xla_pjrt_handoff.md §2a-ter). Hyperparameters match the retired tests render:
-- β₁ 0.9, β₂ 0.999, ε 1e-8, wd 1e-4; 1/B = 1/128 = 0.0078125 (exact in binary32).
#eval IO.FS.writeFile "verified_mlir/cifar8_adam_train_step.mlir"
  (Proofs.StableHLO.cifar8AdamTrainStepFaithfulV 128 3 16 16 32 32 2 2 64 10 3 3
    "0.0078125" "0.9" "0.1" "0.999" "0.001" "1.0e-8" "0.0001"
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0)
    (fun _ => 0))

-- ── §2i: the SGD and NESTEROV peers, staged to their OWN paths ──────────────────────────────
-- ✅ SWAPPED 2026-07-29. These were staged beside the incumbents, tied, and swapped — the
-- hand-written emitters in `tests/TestCifar8AdamTrain.lean` are retired, so each `#eval` below is
-- now its artifact's ONLY writer. `cifar8-opt-tie {sgd,mom}` came back BIT-EXACT on all 52,858
-- recovered gradient coordinates with the m/v passthrough slots bit-exact, and both negative
-- controls fire (÷B 0.0078125→0.008 ⇒ 0.024; μ 0.9→0.91 ⇒ 1.6e-4 with 0/52858 exact).
--
-- The interfaces are IDENTICAL to the AdamW render (71 in / 69 out) because the packed `[θ|m|v]`
-- signature is shared — only the tail moves. `%mu` is baked at 0.9 and `%lr` stays runtime, both
-- matching the retired emitters.
--
-- ⚠ These will NOT be byte-identical to the incumbents: `momVNextF`/`momParamF` are separate SHlo
-- nodes so `v'` is computed twice (SHlo is single-result — `adamWParamF` recomputes `m'`/`v'` the
-- same way), where the retired `emitMomentum` emitted one fused block. XLA's CSE folds it, and
-- §2b-bis measured exactly that pattern costing nothing on R34. Hence the tie is NUMERIC.
#eval IO.FS.writeFile "verified_mlir/cifar8_sgd_train_step.mlir"
  (Proofs.StableHLO.cifar8SgdTrainStepFaithful)
#eval IO.FS.writeFile "verified_mlir/cifar8_mom_train_step.mlir"
  (Proofs.StableHLO.cifar8MomTrainStepFaithful)

-- ── the data-parallel exact gate (handoff §2b-quater) ────────────────────────────────────────
-- cifar8 has NO BatchNorm, so the batch decomposition is an identity and 2 replicas × B=128 with
-- an all_reduce'd gradient must equal 1 device × B=256 to fp rounding. That is the ONLY check
-- that pins the collective's SEMANTICS rather than its syntax; R34 cannot be checked this way
-- (BN normalises per replica). Driven by `cifar8-dp-check`.
--
-- 1/256 = 0.00390625 and 1/128 = 0.0078125 are both exact in binary32, so the loss scaling
-- contributes no rounding of its own to the comparison.
#eval IO.FS.writeFile "verified_mlir/cifar8_adam256_train_step.mlir"
  (Proofs.StableHLO.cifar8AdamTrainStepFaithfulV 256 3 16 16 32 32 2 2 64 10 3 3
    "0.00390625" "0.9" "0.1" "0.999" "0.001" "1.0e-8" "0.0001"
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0)
    (fun _ => 0))

#eval IO.FS.writeFile "verified_mlir/cifar8_adamdp_train_step.mlir"
  (Proofs.StableHLO.cifar8AdamTrainStepFaithfulV 128 3 16 16 32 32 2 2 64 10 3 3
    "0.0078125" "0.9" "0.1" "0.999" "0.001" "1.0e-8" "0.0001"
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0)
    (fun _ => 0) (replicas := 2))

#eval IO.FS.writeFile "verified_mlir/cifar8_train_step.mlir"
  (Proofs.StableHLO.cifar8TrainStepFaithfulV 128 3 16 16 32 32 2 2 64 10 3 3 "0.00078125"
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0)
    (fun _ => 0))

-- ═══ bf16 CIFAR — the third precision of the §5.2 optimizer sweep ═══════════════════════════
-- SAME renderers, SAME arguments, `bf16 := true`. Each artifact therefore differs from its f32
-- peer ONLY in the eight forward convs' emit (`flatConvFBf16` vs `flatConvF`) — which is exactly
-- the claim: the OPTIMIZER ORDERING (SGD < AdamW < Nesterov) is invariant under precision. The
-- f32 peers stay byte-identical, which is the gate that the threading is a no-op at the default.
--
-- Slug is `cifar8_bf16`, so `VerifiedTrain.mkSession` resolves
--   `{slug}_train_step.mlir` / `{slug}_{variant}_train_step.mlir` / `{slug}_fwd.mlir`
-- exactly as it does for the f32 and fp8 arms. Func symbols are renamed to match, because the
-- driver calls `m.{slug}_{variant}_train_step` by name.
--
-- ⚠ FORWARD ONLY: cifar8's backward is on the PER-EXAMPLE `convBack`/`dotOut` and the 27 bf16
-- ops were built for ImageNet's BATCHED family, so `convBackBf16`/`dotOutBf16` do not exist.
-- planning/archive/cifar_lowprec_stability.md §4.1 has why the fix is unification, not two new ops.
-- ⚠⚠ NO SPEEDUP, by design — §5.3 measured bf16 at 0.87× across cifar8's conv stack. These
-- artifacts demonstrate that the MATH scales across precision, never the throughput.
#eval IO.FS.writeFile "verified_mlir/cifar8_bf16_train_step.mlir"
  ((Proofs.StableHLO.cifar8TrainStepFaithfulV 128 3 16 16 32 32 2 2 64 10 3 3 "0.00078125"
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0)
    (fun _ => 0) (bf16 := true)).replace "@cifar8_train_step" "@cifar8_bf16_train_step")

#eval IO.FS.writeFile "verified_mlir/cifar8_bf16_mom_train_step.mlir"
  ((Proofs.StableHLO.cifar8AdamTrainStepFaithfulV 128 3 16 16 32 32 2 2 64 10 3 3
    "0.0078125" "0.9" "0.1" "0.999" "0.001" "1.0e-8" "0.0001"
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0)
    (fun _ => 0) 1 .nesterov (bf16 := true)).replace "@cifar8_adam_train_step" "@cifar8_bf16_mom_train_step")

-- ═══ the BATCHED render (`…FaithfulB`) — the unification, and FULL bf16 ═══════════════════
-- Emitted from `cifar8AdamTrainStepFaithfulB`, which is on ImageNet's batched op family. Unlike
-- the `…V` artifacts above (bf16 forward convs only), these carry bf16 through the BACKWARD as
-- well — `convBackBatchedBf16` + `convWeightGradBBf16` — because those twins exist for the
-- batched family and not for the per-example one. Zero new verified ops; see §4.1.
-- ⭐ THE FIRST ARTIFACT IN THIS REPO CONTAINING AN f8 TYPE. Forward convs only for now
-- (`convBackBatchedF8` / `convWeightGradBF8` do not exist yet — planning/archive/fp8_in_graph.md §6
-- step 1), and UNSCALED, so this is a lowering probe rather than a trainable arm: E4M3 maxes
-- at 448 and XLA synthesises scale = 1.0 when given no scale operand (§4).
#eval IO.FS.writeFile "verified_mlir/cifar8b_fp8_adam_train_step.mlir"
  ((Proofs.StableHLO.cifar8AdamTrainStepFaithfulB 128 3 16 16 32 32 2 2 64 10 3 3
    "0.0078125" "0.9" "0.1" "0.999" "0.001" "1.0e-8" "0.0001"
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0)
    (fun _ => 0) 1 .adamw (bf16 := false) (fp8 := true)).replace
      "@cifar8b_adam_train_step" "@cifar8b_fp8_adam_train_step")

#eval IO.FS.writeFile "verified_mlir/cifar8b_adam_train_step.mlir"
  (Proofs.StableHLO.cifar8AdamTrainStepFaithfulB 128 3 16 16 32 32 2 2 64 10 3 3
    "0.0078125" "0.9" "0.1" "0.999" "0.001" "1.0e-8" "0.0001"
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0)
    (fun _ => 0) 1 .adamw)

-- ⚠ The entry symbol MUST match the file stem: `regen_verified_mlir.sh check` fails any
-- artifact whose declared `@name` differs from its path, because the driver resolves the
-- entry as `m.{slug}_{variant}_train_step` and so could never load it.
#eval IO.FS.writeFile "verified_mlir/cifar8b_bf16_adam_train_step.mlir"
  ((Proofs.StableHLO.cifar8AdamTrainStepFaithfulB 128 3 16 16 32 32 2 2 64 10 3 3
    "0.0078125" "0.9" "0.1" "0.999" "0.001" "1.0e-8" "0.0001"
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0)
    (fun _ => 0) 1 .adamw (bf16 := true)).replace
      "@cifar8b_adam_train_step" "@cifar8b_bf16_adam_train_step")

-- Eval forward for the batched slug. The forward graph is IDENTICAL either way (the batch was
-- always in the MLIR; only the Lean-side type changed), so this is the f32 `cifar8_fwd` renamed
-- rather than a re-render — it cannot drift from what the `…V` arms evaluate against, which is
-- exactly what makes the B-vs-V training comparison a controlled one.
#eval do
  let fwd ← IO.FS.readFile "verified_mlir/cifar8_fwd.mlir"
  IO.FS.writeFile "verified_mlir/cifar8b_fwd.mlir" (fwd.replace "@cifar8_fwd" "@cifar8b_fwd")

#eval IO.FS.writeFile "verified_mlir/cifar8_bf16_adam_train_step.mlir"
  ((Proofs.StableHLO.cifar8AdamTrainStepFaithfulV 128 3 16 16 32 32 2 2 64 10 3 3
    "0.0078125" "0.9" "0.1" "0.999" "0.001" "1.0e-8" "0.0001"
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0)
    (fun _ => 0) 1 .adamw (bf16 := true)).replace "@cifar8_adam_train_step" "@cifar8_bf16_adam_train_step")

-- The eval forward stays f32 — you train in bf16 and evaluate in f32, so this is the f32
-- `cifar8_fwd` renamed to the bf16 slug rather than a re-render. Making that a copy (not a
-- second renderer call) is deliberate: it cannot drift from the artifact the f32 arm evaluates.
#eval do
  let fwd ← IO.FS.readFile "verified_mlir/cifar8_fwd.mlir"
  IO.FS.writeFile "verified_mlir/cifar8_bf16_fwd.mlir"
    (fwd.replace "@cifar8_fwd" "@cifar8_bf16_fwd")

-- Regenerate `verified_mlir/cifar8_bn_train_step.mlir` (what MainCifar8BnVerified trains on)
-- from the faithful renderer; den-certified by the existing generics (CifarPoC.conv{W,B}_den,
-- CifarBnPoC.bn{Gamma,Beta}_den, Cifar8PoC.dense{W,B}_den).
#eval IO.FS.writeFile "verified_mlir/cifar8_bn_train_step.mlir"
  (Proofs.StableHLO.cifar8BnTrainStepFaithfulV 128 3 16 16 32 32 2 2 64 10 3 3 "1.0e-05" "0.00078125"
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0)
    (fun _ => 0))

-- ── §2i: the three PACKED BN variants ───────────────────────────────────────────────────────
-- ✅ SWAPPED 2026-07-30. Staged to `_cert` paths, tied against the hand-written incumbents, then
-- swapped onto the canonical names; the three `IO.FS.writeFile` calls in
-- `tests/TestCifar8AdamTrain.lean` are retired (the renders stay as the ties' references), so each
-- `#eval` below is now its artifact's ONLY writer. `cifar8_bn_adam` is the one artifact of §2i's 13
-- that backs a REAL trainer (`cifar8-bn-verified-adam{,-xla}`).
--
-- `cifar8-opt-tie bn_{adam,mom,sgd}`, all three against a BIT-EXACT A-vs-A floor and a
-- semantics-preserving reorder control (the reference render vs itself on the reversed batch):
--   bn_adam  gradient norm-rel 1.0e-6, spread 8/38 — the control's own 8, the SAME param indices
--   bn_mom   gradient norm-rel 1.0e-6, spread 8/38 = the control's 8; `m` passthrough bit-exact
--   bn_sgd   gradient norm-rel 3.8e-5 vs the control's 1.9e-5 (2.0×), spread 11/38 ⊂ the control's 12
-- The 8 are the CONV BIASES, whose gradient `Σ_{b,h,w} dy` is a cancelling reduce over 128·H·W
-- terms — §2f-bis's finding on a second net, and the reason the spread gate is control-relative and
-- not absolute (an absolute 1e-4 per-param bound FAILS the real tie here).
--
-- `lr` is a RUNTIME arg for all three (the cosine+warmup schedule drives it), so unlike the fused
-- render the batch mean cannot fold into it: `invB` = 1/128 = 0.0078125, exact in binary32. The
-- AdamW constants match the retired emitter: β₁ 0.9, β₂ 0.999, ε 1e-8, wd 1e-4; Nesterov μ 0.9.
-- BN ε stays 1e-05, as in the fused render.
--
-- ⚠ These will NOT be byte-identical to the incumbents even where the arithmetic agrees: the
-- retired emitter spelled the batch mean `divide by 128.0` where `scaleF` multiplies by 0.0078125,
-- and `momVNextF`/`momParamF` are separate SHlo nodes so `v'` is computed twice (SHlo is
-- single-result). Hence the tie is NUMERIC — `cifar8-opt-tie bn_{adam,mom,sgd}`.
private def c8bnPacked (opt : Proofs.StableHLO.CifarOpt) : String :=
  Proofs.StableHLO.cifar8BnTrainStepFaithfulV 128 3 16 16 32 32 2 2 64 10 3 3 "1.0e-05" "0.00078125"
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0)
    (fun _ => 0) (some opt) "0.0078125" "0.9" "0.1" "0.999" "0.001" "1.0e-8" "0.0001"

#eval IO.FS.writeFile "verified_mlir/cifar8_bn_adam_train_step.mlir" (c8bnPacked .adamw)
#eval IO.FS.writeFile "verified_mlir/cifar8_bn_mom_train_step.mlir" (c8bnPacked .nesterov)
#eval IO.FS.writeFile "verified_mlir/cifar8_bn_sgd_train_step.mlir" (c8bnPacked .sgd)

-- ── §2i: the cifar8-WIDE family — `cifar8` at `d1 := 512`, NOT a second net ─────────────────
-- Measured 2026-07-30: `cifar8{,Bn}wVerified` agree with `cifar8{,Bn}Verified` layer-for-layer up to
-- the head width, and the committed `cifar8w_bn_adam_train_step.mlir` is **byte-identical modulo the
-- entry name** to the width-sweep's `cifar8_bn_512_adam_train_step.mlir`. So all six wide train steps
-- are these same two renderers at 512, and the interfaces match the committed artifacts exactly —
-- 71/69 and 119/117, arg + return types AND names positionally identical, 0 MALFORMED.
--
-- These back the Chapter-5 "bridge" table (`runs/ablation_cifar8w/README.md`): the wide-vs-narrow
-- comparison behind *"head width barely matters — 7.1× the params, accuracy within a point; the
-- depth, not the head, is the lever."* All six cells are load-bearing, and the `cifar8_bn_{d}` width
-- sweep is adam-only so it covers just one of them.
--
-- ✅ SWAPPED 2026-07-30, all six tied (no-BN three BIT-EXACT; BN three at 1e-6/3.4e-5 against a
-- reorder control they match or beat) and each `#eval` is now its artifact's ONLY writer. The entry rename is what the retired writer did too: both
-- renderers emit the narrow slug, and the wide drivers ask for `m.cifar8w[_bn]_<opt>_train_step`.
private def c8wPacked (opt : Proofs.StableHLO.CifarOpt) (entry : String) : String :=
  (Proofs.StableHLO.cifar8AdamTrainStepFaithfulV 128 3 16 16 32 32 2 2 512 10 3 3
    "0.0078125" "0.9" "0.1" "0.999" "0.001" "1.0e-8" "0.0001"
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0)
    (fun _ => 0) 1 opt).replace "@cifar8_adam_train_step" s!"@{entry}"

/-- Wide-head (d1=512) peer of `c8wPacked` on the **BATCHED** op family, with a bf16 switch.
    Same net, same hyperparameters, same packed signature as `c8wPacked`; the only differences are
    the op family (`…FaithfulB`) and the `bf16` flag. This is what the §4.3 "Lever 3: precision"
    sweep trains on, so f32 and bf16 come from ONE renderer and differ only in the emit — which is
    what makes that lever a controlled comparison rather than two nets.

    ⚠ Unlike the `…V` bf16 artifacts, bf16 here reaches the BACKWARD too (23/23 convolutions,
    vs 8/23), because the batched family is the one the 27 bf16 ops were built for. -/
private def c8wbPacked (opt : Proofs.StableHLO.CifarOpt) (bf16 : Bool) (entry : String) : String :=
  (Proofs.StableHLO.cifar8AdamTrainStepFaithfulB 128 3 16 16 32 32 2 2 512 10 3 3
    "0.0078125" "0.9" "0.1" "0.999" "0.001" "1.0e-8" "0.0001"
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0)
    (fun _ => 0) 1 opt (bf16 := bf16)).replace "@cifar8b_adam_train_step" s!"@{entry}"

#eval IO.FS.writeFile "verified_mlir/cifar8wb_adam_train_step.mlir"      (c8wbPacked .adamw    false "cifar8wb_adam_train_step")
#eval IO.FS.writeFile "verified_mlir/cifar8wb_mom_train_step.mlir"       (c8wbPacked .nesterov false "cifar8wb_mom_train_step")
#eval IO.FS.writeFile "verified_mlir/cifar8wb_sgd_train_step.mlir"       (c8wbPacked .sgd      false "cifar8wb_sgd_train_step")
#eval IO.FS.writeFile "verified_mlir/cifar8wb_bf16adam_train_step.mlir"  (c8wbPacked .adamw    true  "cifar8wb_bf16adam_train_step")
#eval IO.FS.writeFile "verified_mlir/cifar8wb_bf16mom_train_step.mlir"   (c8wbPacked .nesterov true  "cifar8wb_bf16mom_train_step")
#eval IO.FS.writeFile "verified_mlir/cifar8wb_bf16sgd_train_step.mlir"   (c8wbPacked .sgd      true  "cifar8wb_bf16sgd_train_step")

-- Eval forward for the batched wide slug: the f32 `cifar8w_fwd` renamed. The forward graph does
-- not change with the op family or with bf16 (you train low-precision and evaluate in f32), and
-- copying rather than re-rendering means it cannot drift from what the f32 arms evaluate against.
#eval do
  let fwd ← IO.FS.readFile "verified_mlir/cifar8w_fwd.mlir"
  IO.FS.writeFile "verified_mlir/cifar8wb_fwd.mlir" (fwd.replace "@cifar8w_fwd" "@cifar8wb_fwd")

private def c8wBnPacked (opt : Proofs.StableHLO.CifarOpt) (from_ entry : String) : String :=
  (Proofs.StableHLO.cifar8BnTrainStepFaithfulV 128 3 16 16 32 32 2 2 512 10 3 3
    "1.0e-05" "0.00078125"
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0)
    (fun _ => 0) (some opt) "0.0078125" "0.9" "0.1" "0.999" "0.001" "1.0e-8" "0.0001"
   ).replace s!"@{from_}" s!"@{entry}"

#eval IO.FS.writeFile "verified_mlir/cifar8w_adam_train_step.mlir"
  (c8wPacked .adamw "cifar8w_adam_train_step")
#eval IO.FS.writeFile "verified_mlir/cifar8w_mom_train_step.mlir"
  (c8wPacked .nesterov "cifar8w_mom_train_step")
#eval IO.FS.writeFile "verified_mlir/cifar8w_sgd_train_step.mlir"
  (c8wPacked .sgd "cifar8w_sgd_train_step")
#eval IO.FS.writeFile "verified_mlir/cifar8w_bn_adam_train_step.mlir"
  (c8wBnPacked .adamw "cifar8_bn_adam_train_step" "cifar8w_bn_adam_train_step")
#eval IO.FS.writeFile "verified_mlir/cifar8w_bn_mom_train_step.mlir"
  (c8wBnPacked .nesterov "cifar8_bn_mom_train_step" "cifar8w_bn_mom_train_step")
#eval IO.FS.writeFile "verified_mlir/cifar8w_bn_sgd_train_step.mlir"
  (c8wBnPacked .sgd "cifar8_bn_sgd_train_step" "cifar8w_bn_sgd_train_step")

-- ── the cifar8-WIDE BN family on the BATCHED op family, with the bf16 switch ────────────────
-- The peer of `c8wbPacked` for the NORMALIZED net, and what Chapter 4's precision lever
-- (planning/archive/bf16_batchnorm.md) trains on. Six artifacts from ONE renderer: three optimizers ×
-- {f32, bf16}, so precision is the only thing that moves inside a pair and the comparison is
-- controlled by construction rather than by two nets agreeing to be similar.
--
-- ⭐ bf16 reaches all 23 convolutions here — forward, dgrad AND wgrad — because the batched family
-- is the one the 27 bf16 ops were built for. The `…V` BN render could reach none of them.
-- ⚠ BatchNorm stays f32 and PER-EXAMPLE in both arms, which is the point rather than an omission:
-- it keeps the net identical to `cifar8w_bn_*`, it lets `cifar8w_bn_fwd.mlir` be reused verbatim
-- (per-example BN needs no running statistics), and it is what every bf16 net in this repo does.
-- See the `cifar8BnTrainStepFaithfulB` docstring.
-- ⚠ The entry symbol MUST match the file stem: `regen_verified_mlir.sh check` fails any artifact
-- whose declared `@name` differs from its path, because the driver resolves the entry as
-- `m.{slug}_{variant}_train_step` and so could never load it.
private def c8wbBnPacked (opt : Proofs.StableHLO.CifarOpt) (bf16 : Bool) (entry : String) : String :=
  (Proofs.StableHLO.cifar8BnTrainStepFaithfulB 128 3 16 16 32 32 2 2 512 10 3 3
    "1.0e-05" "0.0078125" "0.9" "0.1" "0.999" "0.001" "1.0e-8" "0.0001"
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
    (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0)
    (fun _ => 0) 1 opt (bf16 := bf16)).replace "@cifar8b_bn_adam_train_step" s!"@{entry}"

#eval IO.FS.writeFile "verified_mlir/cifar8wb_bn_adam_train_step.mlir"
  (c8wbBnPacked .adamw    false "cifar8wb_bn_adam_train_step")
#eval IO.FS.writeFile "verified_mlir/cifar8wb_bn_mom_train_step.mlir"
  (c8wbBnPacked .nesterov false "cifar8wb_bn_mom_train_step")
#eval IO.FS.writeFile "verified_mlir/cifar8wb_bn_sgd_train_step.mlir"
  (c8wbBnPacked .sgd      false "cifar8wb_bn_sgd_train_step")
#eval IO.FS.writeFile "verified_mlir/cifar8wb_bn_bf16adam_train_step.mlir"
  (c8wbBnPacked .adamw    true  "cifar8wb_bn_bf16adam_train_step")
#eval IO.FS.writeFile "verified_mlir/cifar8wb_bn_bf16mom_train_step.mlir"
  (c8wbBnPacked .nesterov true  "cifar8wb_bn_bf16mom_train_step")
#eval IO.FS.writeFile "verified_mlir/cifar8wb_bn_bf16sgd_train_step.mlir"
  (c8wbBnPacked .sgd      true  "cifar8wb_bn_bf16sgd_train_step")

-- Eval forward for the batched wide BN slug: the `cifar8w_bn_fwd` renamed. Per-example BN means
-- train and eval normalize identically, so there is nothing to freeze and nothing to re-render —
-- and copying rather than re-rendering means it cannot drift from what the f32 arm evaluates
-- against, which is exactly what makes the f32-vs-bf16 comparison a controlled one.
#eval do
  let fwd ← IO.FS.readFile "verified_mlir/cifar8w_bn_fwd.mlir"
  IO.FS.writeFile "verified_mlir/cifar8wb_bn_fwd.mlir" (fwd.replace "@cifar8w_bn_fwd" "@cifar8wb_bn_fwd")

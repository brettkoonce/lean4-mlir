# Planning — fp8 (E4M3): where it is, and the real-hardware lowering handoff

Stash of notes + a handoff for the **fp8-lowering** project (best done on a CUDA / Ada
box). The fp8 *numerics + proofs* are done and emulated; the open work is executing on
real fp8 tensor cores. This doc captures the state, the design, and the step order.

## 0. TL;DR

- **fp8 today = emulation.** The E4M3 trainers round weights+input to the E4M3 grid
  *in Lean on the host* (`LeanMlir/E4M3Quant.lean`) and feed the **same fp32** verified
  StableHLO (fp32 matmul, fp32 accumulate, fp32 master). The GPU never sees an 8-bit
  number. The accuracy numbers are the *true fp8 numerics*; only the speed is missing.
- **The proofs are grid-agnostic and already cover the deployed kernel.** §3b
  (`LeanMlir/Proofs/E4M3FaithfulPoC.lean`, `e4m3_render_faithful`) proves the emitted
  block-scaled-E4M3 graph denotes "dequant each operand, then matmul (fp32 accumulate)",
  for **any** quantizer `q`. §3c (`FloatBridge.lean`, `linear_e4m3_argmax_preserved`) is
  the accuracy bound.
- **fp8 lowering = re-type §3b's graph to f8 + IREE f8 GEMM + f8 FFI + fp8 silicon.**
  Mostly *engineering*; the proof carries over under one documented HW-semantics
  assumption (the f8 dot computes the exact ∑ with fp32 accumulate).
- **Hardware:** gfx1100 (RDNA3, this box) has **no** fp8 units → emulation is the only
  option here. **RTX 4060 Ti (Ada) HAS fp8 tensor cores** → that is the box for real fp8
  lowering. (fp4 is Blackwell-only; not on the 4060 Ti.)

## 1. The precision / lowering map (mental model)

Two independent axes: is there a proof? does it run on real ops of that precision?

| precision | proof? | real-hw lowering? | status |
|---|---|---|---|
| **fp32** | ✅ VJP suite + render-tie + Float32 descent bridge | ✅ `*-verified` exes run fp32 MLIR on GPU | proof + lowering, **tied** (render-tie) — gold |
| **bf16** | ⚠️ abstract bound only (FloatBridge §1c, `u=2⁻⁸`), **untied** | ✅ real bf16 ops (`NetConfig.bf16`/`bf16Conv`), in the **unverified perf codegen** | lowered, loosely-proven; "same accuracy" is empirical |
| **fp8** | ✅ §3b render-tie + §3c accuracy (abstract `q`) | ❌ emulated (fp32 math on E4M3-grid operands) | **proven, not lowered** ← this doc |

One-liner: *the proof frontier (fp8) is ahead of the lowering frontier (bf16); they only
meet + get tied at fp32.* fp8's gap is engineering; bf16's gap is proof.

## 2. What exists now (the emulated fp8 trainers)

All on the SAME verified StableHLO as their fp32 peers; fp8 = host-side operand byte-prep,
fp32 master, fp32 accumulate. Scope: **fp8 weights + fp8 input, fp32 intermediates**
(the relu/pool/flatten activations + backward cotangents live inside the fused kernel;
host prep can't reach them — that needs in-graph quant, see §5).

- `LeanMlir/E4M3Quant.lean` — pure-Lean E4M3 (1-4-3) round-to-nearest grid (matches
  `scripts/mnist_e4m3_demo.py`): `roundE4M3`, `quantPerTensor`, `quantPerColumn` (dense),
  `quantPerLeadingBlock` (conv per-output-channel), `quantPackedParams` (per-slot over a
  packed buffer), `addDelta` (fp32-master gradient-delta recovery). f32 bytes via
  `Float32.toBits`.
- `LeanMlir/VerifiedTrain.lean`:
  - `trainLinearE4M3` — depth-1 linear (fully fp8).
  - `trainE4M3` — packed SGD (MLP / CNN / CIFAR): quantize θ each step, `addDelta` to master.
  - `trainAdamSchedE4M3` — Adam / Nesterov-momentum: quantize the θ third of `[θ|m|v]`,
    keep m/v + master fp32, recover the optimizer-step delta; distinct `_e4m3` checkpoint;
    honors `LEAN_MLIR_MAX_EPOCHS`.
- exes: `mnist-linear-e4m3-verified`, `mnist-mlp-e4m3-verified`, `mnist-cnn-e4m3-verified`,
  `cifar-e4m3-verified`, `cifar8-e4m3-verified{,-momentum,-adam}`.
- Run (ROCm box): `IREE_BACKEND=rocm .lake/build/bin/<exe> data`. Needs `iree-compile` on
  PATH (e.g. `proof_verify_demo/lean4-mlir/.venv/bin`).

### Measured fp8 vs fp32 (verified cifar8 CNN, gfx1100, test acc)

| optimizer | fp8 @20 | fp32 @20 | fp32 final @40 | fp8 penalty @20 |
|---|---|---|---|---|
| plain SGD (const lr) | 63.5% | 65.7% | 66.7% | −2.2 pt |
| AdamW (cosine) | 71.4% | 72.1% | 74.0% | −0.7 pt |
| Nesterov-mom (cosine) | 75.4% | 75.1% | 76.8% | ≈0 (+0.3) |

Findings: (1) optimizer ranking (Nesterov > Adam > SGD) is **identical** in fp8 and fp32;
(2) fp8 penalty is small and **optimizer-dependent** — plain SGD eats the per-step rounding
(~2pt), Nesterov's fp32-master velocity averages it away (~0). **These are the accuracy
numbers real fp8 silicon must reproduce** (the lowering is a speed/validation move, not an
accuracy move).

## 3. fp8 lowering — the design (the actual handoff)

§3b was built in the shape of a real fp8 kernel: operands are the **grid codes** (what fp8
hardware feeds), `dotIn`'s `den` is the **exact ∑** (= fp32 accumulate), `layerScaleF` is
the per-output-column dequant `sx·sWⱼ`. So lowering = **re-type that graph to f8 and point
IREE at the fp8 path**, not a re-proof.

| piece | emulation (now) | lowering |
|---|---|---|
| operands | fp32 on the E4M3 grid | `tensor<...xf8E4M3FN>` (1 byte each) |
| matmul | fp32 `dot` (software) | `stablehlo.dot_general` f8 in, **f32 accumulate** → fp8 tensor cores |
| dequant scale | `layerScaleF` (fp32) | unchanged (explicit op; or HW microscaling on Blackwell) |
| host | `roundE4M3` → fp32 bytes | `roundE4M3` → **pack as f8 bytes** |
| FFI | f32 `ByteArray` operands | f8 operand buffers (new signatures) |

**Why the proof carries over (≈ free):** the render-tie's `den` is about the math (the ∑
and the scale factoring), not the storage type — `q(x/sx)` is the same value as f8 bits or
fp32-on-grid. So `e4m3_render_faithful` still holds for the f8-typed graph **provided the
f8 `dot_general` computes the exact ∑ of products with fp32 accumulate** — which is what
fp8 tensor cores do. That is **one documented HW-semantics assumption** (same trust tier as
"IREE/StableHLO lowering is faithful"), not a new theorem. Keep fp32 master + fp32 accumulate.

## 4. fp8 lowering — step order (by risk)

1. **IREE f8 GEMM spike (do FIRST).** Standalone: does IREE's CUDA backend lower
   `dot_general` f8E4M3FN→f32 to the Ada fp8 MMA, and is it faster than fp32? One-op test
   on the 4060 Ti before touching the trainer. If IREE's f8 path is immature, that gates
   everything — find out cheaply.
2. **Renderer emits f8 types.** The SHlo pretty-printer gains `f8E4M3FN` operand/dot types;
   `den` is unchanged so the proofs don't move. Start with the **depth-1 linear**
   (`linear_train_step`), mirroring §3b's scope.
3. **f8 FFI + operand packing.** `linearTrainStepV` (and later `mlpTrainStepV`) currently
   take f32 `ByteArray`s; add f8 operand buffers; pack `roundE4M3` output as bytes.
4. **Validate against the emulated numbers** (§2 table) — real Ada fp8 should reproduce them
   to within rounding-convention noise. This is the payoff: a hardware-faithfulness check,
   like the JAX vjp_oracle.
5. **Scale up** per-op (dense → conv → the packed nets), and keep the per-column/per-channel
   block scale (Ada has no HW microscaling; do the scale as the explicit `layerScaleF`).

Env on the CUDA box: install IREE's CUDA toolchain in a venv, put its `iree-compile` on
PATH, `IREE_BACKEND=cuda` (the `lake run` demo auto-detects cuda via `nvidia-smi`).

## 5. Related open gaps (not blocking lowering)

- **In-graph quant for depth>1.** The MLP/CNN/CIFAR fp8 trainers only quantize weights +
  the *input*; intermediate activations are fp32 (inside the fused kernel). Truly
  all-operand fp8 needs an **in-graph E4M3 round op** (a faithful `convertF`, see the bf16
  note) so the deeper-matmul activations land on the grid. Orthogonal to hardware lowering.
- **bf16 render-tie** (the symmetric gap). bf16 has lowering but its proof (§1c) is untied.
  To tie it: add a faithful in-graph `convertF` op (`den(convertF rnd x) = rnd x`, kept
  abstract like §3b's `q`), prove `bf16Round` satisfies `|·−·| ≤ 2⁻⁸|·|`, show
  `den(bf16graph) = denseMixed`, compose with `dense_close_mixed`. The hard part is routing
  the bf16 cast (today in the unverified `NetConfig`/Train.lean perf codegen) through the
  **verified renderer**. This `convertF` op is the same ingredient §5's in-graph fp8 quant
  would use.
- **fp4 (E2M1).** Render-tie covers it for free (abstract `q`); blocked numerically
  (`u≈25%`, Higham fan-in wall at 4 → worst-case bounds vacuous) and on hardware
  (Blackwell-only). An inference/depth-1 demo is the honest ceiling; not a training target.

## 6. File index

- Quantizer: `LeanMlir/E4M3Quant.lean`
- Drivers: `LeanMlir/VerifiedTrain.lean` (`trainLinearE4M3` / `trainE4M3` / `trainAdamSchedE4M3`)
- Proof — render-tie (§3b): `LeanMlir/Proofs/E4M3FaithfulPoC.lean` (`e4m3_render_faithful`)
- Proof — accuracy (§3c) + bf16 §1c: `LeanMlir/Proofs/FloatBridge.lean`
- Numpy oracle (the grid + the optimizer dynamics): `scripts/mnist_e4m3_demo.py`,
  `scripts/mnist_e4m3_train_demo.py`
- bf16 codegen flags: `LeanMlir/Types.lean` (`NetConfig.bf16` / `bf16Conv`)
- Quant-planning (precision-scaling table, §1c/§3a/§3b/§3c): `planning/floatbridge_quantization.md`

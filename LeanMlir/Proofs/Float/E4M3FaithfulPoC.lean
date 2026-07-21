import LeanMlir.Proofs.Codegen.StableHLO

/-! # PoC: the E4M3 (fp8) quantized MNIST-linear render-tie (planning §3b)

`planning/floatbridge_quantization.md` §3b: the **structural faithfulness** of the
low-precision scheme. Where §3c bounds the *accuracy* of E4M3-mixed inference, this
file proves the *correctness of the implementation*: the emitted block-scaled-E4M3
matmul graph **denotes** the intended algorithm, with **no accuracy claim**.

The deployed fp8 kernel is *block-scaled with fp32 accumulate*: weights are quantized
offline to an integer grid with a per-output-column scale `sWⱼ`, activations at runtime
with a per-tensor scale `sx`; the integer codes are multiply-accumulated in fp32; a
single per-output dequant `sx·sWⱼ` scales the result; the fp32 bias is added. The one
fact that makes this *equal* to "dequantize each tensor, then do the exact matmul" is
that the per-output dequant scale **factors out of the accumulate** —
`(sx·sWⱼ)·∑ᵢ q(xᵢ/sx)·q(Wᵢⱼ/sWⱼ) = ∑ᵢ (sx·q(xᵢ/sx))·(sWⱼ·q(Wᵢⱼ/sWⱼ))` — which is
exactly what "fp32 accumulate" buys you (the scales are constant across the reduction).

**No `SHlo` surgery.** The emitted graph is built entirely from existing `den`-faithful
ops: `operand` (the int activation code = the stored bytes), `dotIn` (the int weight
code; its `den` is the exact `∑`, i.e. the fp32 accumulate), `layerScaleF` (the
per-output dequant block-scale, `layerScaleF_faithful`), `addBcast` (the fp32 bias).
The quantizer `q : ℝ → ℝ` (E4M3 round-to-nearest on the 1-4-3 grid; see
`scripts/mnist_e4m3_demo.py`) is left **abstract** — the scheme is faithful for *any*
grid, E4M3 being one instance. Quantization-to-code is the offline/runtime byte
preparation that produces the operands (exactly as real fp8 inference does), so the
render-tie is the genuine "the bytes implement block-scaled-E4M3 matmul with fp32
accumulate" claim.

All theorems kernel-close under `[propext, Classical.choice, Quot.sound]`
(`tests/AuditAxioms.lean`). (Namespace/name kept short for the audit's per-line
`#print axioms` grep — cf. `LinearFaithfulPoC.lean`.)
-/

open Proofs Proofs.StableHLO

namespace Proofs.QuantPoC

variable {m n : Nat}

/-- The stored integer-grid **activation code** (per-tensor scale `sx`):
    `q(xᵢ / sx)`. These are the int8-style bytes the runtime feeds the kernel. -/
noncomputable def actCode (q : ℝ → ℝ) (sx : ℝ) (x : Vec m) : Vec m :=
  fun i => q (x i / sx)

/-- The stored integer-grid **weight code** (per-output-column scale `sW j`):
    `q(Wᵢⱼ / sWⱼ)`. Quantized offline; the per-column scale is the "block scale". -/
noncomputable def weightCode (q : ℝ → ℝ) (sW : Vec n) (W : Mat m n) : Mat m n :=
  fun i j => q (W i j / sW j)

/-- **The emitted block-scaled-E4M3 linear graph.** Built only from `den`-faithful
    `SHlo` ops: `operand` (int activation code) → `dotIn` (int weight code; the `den`
    `∑` is the fp32 accumulate) → `layerScaleF` (per-output dequant `sx·sWⱼ`) →
    `addBcast` (fp32 bias). The "int matmul, fp32 accumulate, single dequant" kernel. -/
noncomputable def e4m3LinearGraph (q : ℝ → ℝ) (sx : ℝ) (sW : Vec n)
    (W : Mat m n) (b : Vec n) (x : Vec m) : SHlo n :=
  .addBcast "%b0" b
    (.layerScaleF "%deq" (fun j => sx * sW j)
      (.dotIn "%Wq" (weightCode q sW W) (.operand "%xq" (actCode q sx x))))

/-- **The intended algorithm, dequant-first form.** Dequantize each tensor —
    activation `sx·q(xᵢ/sx)`, weight `sWⱼ·q(Wᵢⱼ/sWⱼ)` (i.e. `dequant ∘ quant`,
    the round-trip) — then the exact-ℝ linear map `mnistLinear`. This is the
    reference semantics the kernel must match. -/
noncomputable def quantLinear (q : ℝ → ℝ) (sx : ℝ) (sW : Vec n)
    (W : Mat m n) (b : Vec n) (x : Vec m) : Vec n :=
  mnistLinear (fun i j => sW j * q (W i j / sW j)) b (fun i => sx * q (x i / sx))

/-- **The block-scale factors out of the fp32 accumulate** — the arithmetic heart of
    §3b. "Int matmul then one per-output dequant" equals "dequantize each operand then
    matmul"; the per-output scale `sx·sWⱼ` is constant across the reduction, so it pulls
    through the `∑`. (This is *why* fp32 accumulate is the faithful choice.) -/
theorem dequant_factors (q : ℝ → ℝ) (sx : ℝ) (sW : Vec n)
    (W : Mat m n) (x : Vec m) (j : Fin n) :
    (sx * sW j) * (∑ i, q (x i / sx) * q (W i j / sW j))
      = ∑ i, (sx * q (x i / sx)) * (sW j * q (W i j / sW j)) := by
  rw [Finset.mul_sum]
  exact Finset.sum_congr rfl (fun i _ => by ring)

/-- **E4M3 render-tie (structural faithfulness, planning §3b).** The emitted
    block-scaled int-matmul graph denotes exactly the intended dequant-first
    algorithm, for **any** quantizer `q` and scales `sx`, `sW`. The proof is the
    `dequant_factors` identity composed with the `den` of each (verified-faithful)
    op. No accuracy claim — purely "the bytes implement block-scaled-E4M3 matmul
    with fp32 accumulate". -/
theorem e4m3_render_faithful (q : ℝ → ℝ) (sx : ℝ) (sW : Vec n)
    (W : Mat m n) (b : Vec n) (x : Vec m) :
    den (e4m3LinearGraph q sx sW W b x) = quantLinear q sx sW W b x := by
  funext j
  simp only [e4m3LinearGraph, quantLinear, mnistLinear, dense, den, layerScale,
    actCode, weightCode]
  rw [dequant_factors q sx sW W x j]

end Proofs.QuantPoC

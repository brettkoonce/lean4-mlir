# IREE CUDA: f8E4M3FN / f8E5M2 fail to lower on the NVPTX backend

**Status:** OPEN — to file at `iree-org/iree`. Reproduced locally 2026-06-23.
**Workaround:** none on CUDA; use `llvm-cpu` for f8 (works), or stay fp32/bf16 on GPU.

---

## Issue text (paste into iree-org/iree)

### Summary

Any `f8E4M3FN` (or `f8E5M2`) operation fails to compile for the **CUDA/NVPTX**
backend. Even a single scalar `stablehlo.convert f8E4M3FN -> f32` aborts during
MLIR→LLVM translation on an unresolved `builtin.unrealized_conversion_cast`
between `i8` and `f8E4M3FN` vectors. The same input lowers fine on `llvm-cpu`,
and the fp32 equivalent lowers fine on CUDA — so this is specific to f8 in the
NVPTX codegen path.

There are really **two gaps**:

1. **The f8 type doesn't survive NVPTX translation** (the blocking error below).
2. **No fp8 tensor-core path.** When an f8 `dot_general` does reach codegen, the
   strategy is `arith.extf f8E4M3FN -> f32` followed by a **fp32** matmul — i.e.
   even a hypothetical successful compile would not use the Ada/Hopper fp8 MMA
   units, so there would be no fp8 speedup. A real fp8 GEMM lowering (f8 in, f32
   accumulate, fp8 MMA) appears to be unimplemented for CUDA.

### Minimal reproducer

`convert_f8_to_f32.mlir`:
```mlir
func.func @c(%a: tensor<1024xf8E4M3FN>) -> tensor<1024xf32> {
  %0 = stablehlo.convert %a : (tensor<1024xf8E4M3FN>) -> tensor<1024xf32>
  return %0 : tensor<1024xf32>
}
```

```bash
iree-compile convert_f8_to_f32.mlir \
  --iree-hal-target-backends=cuda --iree-cuda-target=sm_86 -o /dev/null
```

(`sm_86` reproduces on any recent build; `sm_89` fails identically on rc20260623+,
the first version that accepts the Ada target — see Versions below.)

### Actual result

```
<unknown>:0: error: LLVM Translation failed for operation: builtin.unrealized_conversion_cast
<unknown>:0: note: see current operation:
  %14 = "builtin.unrealized_conversion_cast"(%13) : (vector<1xi8>) -> vector<1xf8E4M3FN>
error: failed to translate the MLIR LLVM dialect to the native llvm::Module
error: failed to serialize executable for target backend cuda
```

(For the GEMM the same cast appears at the tile width, e.g. `vector<32xi8> -> vector<32xf8E4M3FN>`.)

### Expected result

The f8 convert/GEMM compiles for CUDA, as it does for `llvm-cpu`. Ideally the
f8 `dot_general` lowers to the fp8 MMA on `sm_89+`.

### What I tried (scope)

| variation | result |
|---|---|
| `f8E4M3FN` convert, CUDA | ❌ fails (above) |
| `f8E5M2` convert, CUDA | ❌ identical failure |
| `f8E4M3FN` GEMM (f32 accumulate), CUDA | ❌ identical failure |
| `+ --iree-llvmgpu-enable-small-float-emulation` | ❌ identical failure (flag is recognized but does not resolve the cast) |
| same `f8E4M3FN` convert/GEMM on **`llvm-cpu`** | ✅ compiles |
| **fp32** GEMM on CUDA `sm_89` | ✅ compiles |

So: CUDA backend + fp32 = OK, CUDA backend + f8 = broken, CPU backend + f8 = OK.

### Versions

Both of these fail **identically**:

- `iree-base-compiler 3.12.0rc20260428` @ `af030e43d8343263a6c869eae32f958f229ff7af`
- `iree-base-compiler 3.12.0rc20260623` @ `ac077d8815292c93149ac387a33ca4a844d3641e` (latest at filing)

Note: `rc20260623` newly accepts `--iree-cuda-target=sm_89` (fp32 GEMM compiles
at sm_89; `sm_90` is still "missing GPU target"), but the f8 codegen gap did
**not** follow the new target support.

### Environment

- GPU: NVIDIA GeForce RTX 4060 Ti, compute capability **8.9** (sm_89, Ada — has fp8 tensor cores)
- CUDA 12.9, driver 575.57.08, Linux x86_64
- Frontend: StableHLO via the pip `iree-base-compiler` wheels above

### Files

- `convert_f8_to_f32.mlir` — the minimal scalar repro
- `gemm_f8_f32acc.mlir` — the f8-in/f32-accumulate GEMM (the real workload)
- `reproduce.sh` — runs the failing CUDA cases + the passing CPU/fp32 controls

---

## Why we care (project context)

We render verified StableHLO (Lean → IREE → GPU). The fp8 numerics + proofs are
done and emulated (host-side E4M3 rounding into the same fp32 graph); the open
work was lowering to real Ada fp8 tensor cores. This spike was the gate — and it
says IREE's CUDA f8 path is the blocker. Pivoting to bf16 (which *does* lower on
CUDA) in the meantime. See `planning/fp8_lowering.md`.

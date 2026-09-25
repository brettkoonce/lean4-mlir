// Minimal repro: a single f8E4M3FN -> f32 elementwise convert.
// Fails to lower on the CUDA/NVPTX backend (passes on llvm-cpu).
func.func @c(%a: tensor<1024xf8E4M3FN>) -> tensor<1024xf32> {
  %0 = stablehlo.convert %a : (tensor<1024xf8E4M3FN>) -> tensor<1024xf32>
  return %0 : tensor<1024xf32>
}

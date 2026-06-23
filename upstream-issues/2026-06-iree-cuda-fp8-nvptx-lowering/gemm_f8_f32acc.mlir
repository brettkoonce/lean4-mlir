// f8E4M3FN operands, f32 accumulate — the "fp8 tensor core" GEMM we want.
// Same unrealized_conversion_cast failure as the scalar convert.
func.func @gemm(%a: tensor<4096x4096xf8E4M3FN>, %b: tensor<4096x4096xf8E4M3FN>) -> tensor<4096x4096xf32> {
  %0 = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT]
       : (tensor<4096x4096xf8E4M3FN>, tensor<4096x4096xf8E4M3FN>) -> tensor<4096x4096xf32>
  return %0 : tensor<4096x4096xf32>
}

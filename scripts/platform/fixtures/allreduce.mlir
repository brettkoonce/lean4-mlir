module @m {
  func.func @main(%x: tensor<4xf32>) -> tensor<4xf32> {
    %0 = "stablehlo.all_reduce"(%x) ({
    ^bb0(%a: tensor<f32>, %b: tensor<f32>):
      %s = stablehlo.add %a, %b : tensor<f32>
      stablehlo.return %s : tensor<f32>
    }) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}

module @m {
  func.func @linear_train_step(%x: tensor<128x784xf32>, %W0: tensor<784x10xf32>, %b0: tensor<10xf32>, %onehot: tensor<128x10xf32>) -> (tensor<784x10xf32>, tensor<10xf32>) {
    // ── linear train step: every line is pretty(verified AST node) ──
    %v0 = stablehlo.dot_general %x, %W0, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x784xf32>, tensor<784x10xf32>) -> tensor<128x10xf32>
    %v1 = stablehlo.broadcast_in_dim %b0, dims = [1] : (tensor<10xf32>) -> tensor<128x10xf32>
    %v2 = stablehlo.add %v0, %v1 : tensor<128x10xf32>
    %v3 = stablehlo.exponential %v2 : tensor<128x10xf32>
    %v4 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5 = stablehlo.reduce(%v3 init: %v4) applies stablehlo.add across dimensions = [1] : (tensor<128x10xf32>, tensor<f32>) -> tensor<128xf32>
    %v6 = stablehlo.broadcast_in_dim %v5, dims = [0] : (tensor<128xf32>) -> tensor<128x10xf32>
    %v7 = stablehlo.divide %v3, %v6 : tensor<128x10xf32>
    %v8 = stablehlo.subtract %v7, %onehot : tensor<128x10xf32>
    %v9 = stablehlo.dot_general %x, %v8, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x784xf32>, tensor<128x10xf32>) -> tensor<784x10xf32>
    %v10 = stablehlo.constant dense<0.00078125> : tensor<784x10xf32>
    %v11 = stablehlo.multiply %v9, %v10 : tensor<784x10xf32>
    %v12 = stablehlo.subtract %W0, %v11 : tensor<784x10xf32>
    %v13 = stablehlo.dot_general %x, %W0, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x784xf32>, tensor<784x10xf32>) -> tensor<128x10xf32>
    %v14 = stablehlo.broadcast_in_dim %b0, dims = [1] : (tensor<10xf32>) -> tensor<128x10xf32>
    %v15 = stablehlo.add %v13, %v14 : tensor<128x10xf32>
    %v16 = stablehlo.exponential %v15 : tensor<128x10xf32>
    %v17 = stablehlo.constant dense<0.0> : tensor<f32>
    %v18 = stablehlo.reduce(%v16 init: %v17) applies stablehlo.add across dimensions = [1] : (tensor<128x10xf32>, tensor<f32>) -> tensor<128xf32>
    %v19 = stablehlo.broadcast_in_dim %v18, dims = [0] : (tensor<128xf32>) -> tensor<128x10xf32>
    %v20 = stablehlo.divide %v16, %v19 : tensor<128x10xf32>
    %v21 = stablehlo.subtract %v20, %onehot : tensor<128x10xf32>
    %v22 = stablehlo.constant dense<0.0> : tensor<f32>
    %v23 = stablehlo.reduce(%v21 init: %v22) applies stablehlo.add across dimensions = [0] : (tensor<128x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v24 = stablehlo.constant dense<0.00078125> : tensor<10xf32>
    %v25 = stablehlo.multiply %v23, %v24 : tensor<10xf32>
    %v26 = stablehlo.subtract %b0, %v25 : tensor<10xf32>
    return %v12, %v26 : tensor<784x10xf32>, tensor<10xf32>
  }
}

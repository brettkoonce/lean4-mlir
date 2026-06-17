module @m {
  func.func @mlp_train_step(%x: tensor<128x784xf32>, %W0: tensor<784x512xf32>, %b0: tensor<512xf32>, %W1: tensor<512x512xf32>, %b1: tensor<512xf32>, %W2: tensor<512x10xf32>, %b2: tensor<10xf32>, %onehot: tensor<128x10xf32>) -> (tensor<784x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>) {
    // ── mlp train step: every line is pretty(verified AST node) ──
    %v0 = stablehlo.dot_general %x, %W0, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x784xf32>, tensor<784x512xf32>) -> tensor<128x512xf32>
    %v1 = stablehlo.broadcast_in_dim %b0, dims = [1] : (tensor<512xf32>) -> tensor<128x512xf32>
    %v2 = stablehlo.add %v0, %v1 : tensor<128x512xf32>
    %v3 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v4 = stablehlo.maximum %v2, %v3 : tensor<128x512xf32>
    %v5 = stablehlo.dot_general %v4, %W1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<512x512xf32>) -> tensor<128x512xf32>
    %v6 = stablehlo.broadcast_in_dim %b1, dims = [1] : (tensor<512xf32>) -> tensor<128x512xf32>
    %v7 = stablehlo.add %v5, %v6 : tensor<128x512xf32>
    %v8 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v9 = stablehlo.maximum %v7, %v8 : tensor<128x512xf32>
    %v10 = stablehlo.dot_general %v9, %W2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<512x10xf32>) -> tensor<128x10xf32>
    %v11 = stablehlo.broadcast_in_dim %b2, dims = [1] : (tensor<10xf32>) -> tensor<128x10xf32>
    %v12 = stablehlo.add %v10, %v11 : tensor<128x10xf32>
    %v13 = stablehlo.exponential %v12 : tensor<128x10xf32>
    %v14 = stablehlo.constant dense<0.0> : tensor<f32>
    %v15 = stablehlo.reduce(%v13 init: %v14) applies stablehlo.add across dimensions = [1] : (tensor<128x10xf32>, tensor<f32>) -> tensor<128xf32>
    %v16 = stablehlo.broadcast_in_dim %v15, dims = [0] : (tensor<128xf32>) -> tensor<128x10xf32>
    %v17 = stablehlo.divide %v13, %v16 : tensor<128x10xf32>
    %v18 = stablehlo.subtract %v17, %onehot : tensor<128x10xf32>
    %v19 = stablehlo.dot_general %v18, %W2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x10xf32>, tensor<512x10xf32>) -> tensor<128x512xf32>
    %v20 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v21 = stablehlo.compare GT, %v7, %v20 : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<128x512xi1>
    %v22 = stablehlo.select %v21, %v19, %v20 : tensor<128x512xi1>, tensor<128x512xf32>
    %v23 = stablehlo.dot_general %v22, %W1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<512x512xf32>) -> tensor<128x512xf32>
    %v24 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v25 = stablehlo.compare GT, %v2, %v24 : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<128x512xi1>
    %v26 = stablehlo.select %v25, %v23, %v24 : tensor<128x512xi1>, tensor<128x512xf32>
    %v27 = stablehlo.dot_general %v9, %v18, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<128x10xf32>) -> tensor<512x10xf32>
    %v28 = stablehlo.constant dense<0.00078125> : tensor<512x10xf32>
    %v29 = stablehlo.multiply %v27, %v28 : tensor<512x10xf32>
    %v30 = stablehlo.subtract %W2, %v29 : tensor<512x10xf32>
    %v31 = stablehlo.constant dense<0.0> : tensor<f32>
    %v32 = stablehlo.reduce(%v18 init: %v31) applies stablehlo.add across dimensions = [0] : (tensor<128x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v33 = stablehlo.constant dense<0.00078125> : tensor<10xf32>
    %v34 = stablehlo.multiply %v32, %v33 : tensor<10xf32>
    %v35 = stablehlo.subtract %b2, %v34 : tensor<10xf32>
    %v36 = stablehlo.dot_general %v4, %v22, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<512x512xf32>
    %v37 = stablehlo.constant dense<0.00078125> : tensor<512x512xf32>
    %v38 = stablehlo.multiply %v36, %v37 : tensor<512x512xf32>
    %v39 = stablehlo.subtract %W1, %v38 : tensor<512x512xf32>
    %v40 = stablehlo.constant dense<0.0> : tensor<f32>
    %v41 = stablehlo.reduce(%v22 init: %v40) applies stablehlo.add across dimensions = [0] : (tensor<128x512xf32>, tensor<f32>) -> tensor<512xf32>
    %v42 = stablehlo.constant dense<0.00078125> : tensor<512xf32>
    %v43 = stablehlo.multiply %v41, %v42 : tensor<512xf32>
    %v44 = stablehlo.subtract %b1, %v43 : tensor<512xf32>
    %v45 = stablehlo.dot_general %x, %v26, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x784xf32>, tensor<128x512xf32>) -> tensor<784x512xf32>
    %v46 = stablehlo.constant dense<0.00078125> : tensor<784x512xf32>
    %v47 = stablehlo.multiply %v45, %v46 : tensor<784x512xf32>
    %v48 = stablehlo.subtract %W0, %v47 : tensor<784x512xf32>
    %v49 = stablehlo.constant dense<0.0> : tensor<f32>
    %v50 = stablehlo.reduce(%v26 init: %v49) applies stablehlo.add across dimensions = [0] : (tensor<128x512xf32>, tensor<f32>) -> tensor<512xf32>
    %v51 = stablehlo.constant dense<0.00078125> : tensor<512xf32>
    %v52 = stablehlo.multiply %v50, %v51 : tensor<512xf32>
    %v53 = stablehlo.subtract %b0, %v52 : tensor<512xf32>
    return %v48, %v53, %v39, %v44, %v30, %v35 : tensor<784x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>
  }
}

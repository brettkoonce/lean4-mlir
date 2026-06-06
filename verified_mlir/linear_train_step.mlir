module @m {
  func.func @linear_train_step(%x: tensor<128x784xf32>, %W0: tensor<784x10xf32>, %b0: tensor<10xf32>, %onehot: tensor<128x10xf32>) -> (tensor<784x10xf32>, tensor<10xf32>) {
    // ── forward + softmax-CE cotangent — rendered from the verified AST (lossCotGraph) ──
    %v0 = stablehlo.dot_general %x, %W0, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x784xf32>, tensor<784x10xf32>) -> tensor<128x10xf32>
    %v1 = stablehlo.broadcast_in_dim %b0, dims = [1] : (tensor<10xf32>) -> tensor<128x10xf32>
    %v2 = stablehlo.add %v0, %v1 : tensor<128x10xf32>
    %v3 = stablehlo.exponential %v2 : tensor<128x10xf32>
    %v4 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5 = stablehlo.reduce(%v3 init: %v4) applies stablehlo.add across dimensions = [1] : (tensor<128x10xf32>, tensor<f32>) -> tensor<128xf32>
    %v6 = stablehlo.broadcast_in_dim %v5, dims = [0] : (tensor<128xf32>) -> tensor<128x10xf32>
    %v7 = stablehlo.divide %v3, %v6 : tensor<128x10xf32>
    %v8 = stablehlo.subtract %v7, %onehot : tensor<128x10xf32>
    // dy = %v8 = ⟦lossCotGraph⟧ = ∂CE/∂logits (lossCotGraph_isCEgrad)
    // ── param grads: dW0 = x⊗dy, db0 = Σ_batch dy (wGrad/bGrad_is*Jacobian) ──
    %sc = stablehlo.constant dense<0.0> : tensor<f32>
    %dW0 = stablehlo.dot_general %x, %v8, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x784xf32>, tensor<128x10xf32>) -> tensor<784x10xf32>
    %db0 = stablehlo.reduce(%v8 init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<128x10xf32>, tensor<f32>) -> tensor<10xf32>
    // ── SGD update θ' = θ − lr·∇ (sgdW/sgdB_descends_certified_grad) ──
    %lW0 = stablehlo.constant dense<0.00078125> : tensor<784x10xf32>
    %sW0 = stablehlo.multiply %dW0, %lW0 : tensor<784x10xf32>
    %W0n = stablehlo.subtract %W0, %sW0 : tensor<784x10xf32>
    %lb0 = stablehlo.constant dense<0.00078125> : tensor<10xf32>
    %sb0 = stablehlo.multiply %db0, %lb0 : tensor<10xf32>
    %b0n = stablehlo.subtract %b0, %sb0 : tensor<10xf32>
    return %W0n, %b0n : tensor<784x10xf32>, tensor<10xf32>
  }
}

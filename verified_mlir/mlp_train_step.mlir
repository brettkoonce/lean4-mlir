module @m {
  func.func @mlp_train_step(%x: tensor<128x784xf32>, %W0: tensor<784x512xf32>, %b0: tensor<512xf32>, %W1: tensor<512x512xf32>, %b1: tensor<512xf32>, %W2: tensor<512x10xf32>, %b2: tensor<10xf32>, %onehot: tensor<128x10xf32>) -> (tensor<784x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>) {
    // ── forward (mlpFwdGraph): %h0,%h1 pre-acts, %a0,%a1 activations, %logits ──
    %h0d = stablehlo.dot_general %x, %W0, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x784xf32>, tensor<784x512xf32>) -> tensor<128x512xf32>
    %h0b = stablehlo.broadcast_in_dim %b0, dims = [1] : (tensor<512xf32>) -> tensor<128x512xf32>
    %h0 = stablehlo.add %h0d, %h0b : tensor<128x512xf32>
    %a0z = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %a0 = stablehlo.maximum %h0, %a0z : tensor<128x512xf32>
    %h1d = stablehlo.dot_general %a0, %W1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<512x512xf32>) -> tensor<128x512xf32>
    %h1b = stablehlo.broadcast_in_dim %b1, dims = [1] : (tensor<512xf32>) -> tensor<128x512xf32>
    %h1 = stablehlo.add %h1d, %h1b : tensor<128x512xf32>
    %a1z = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %a1 = stablehlo.maximum %h1, %a1z : tensor<128x512xf32>
    %logitsd = stablehlo.dot_general %a1, %W2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<512x10xf32>) -> tensor<128x10xf32>
    %logitsb = stablehlo.broadcast_in_dim %b2, dims = [1] : (tensor<10xf32>) -> tensor<128x10xf32>
    %logits = stablehlo.add %logitsd, %logitsb : tensor<128x10xf32>
    // ── loss cotangent dy = softmax(logits) − onehot (lossCotGraph_isCEgrad) ──
    %le = stablehlo.exponential %logits : tensor<128x10xf32>
    %lz = stablehlo.constant dense<0.0> : tensor<f32>
    %lsum = stablehlo.reduce(%le init: %lz) applies stablehlo.add across dimensions = [1] : (tensor<128x10xf32>, tensor<f32>) -> tensor<128xf32>
    %lsb = stablehlo.broadcast_in_dim %lsum, dims = [0] : (tensor<128xf32>) -> tensor<128x10xf32>
    %lsm = stablehlo.divide %le, %lsb : tensor<128x10xf32>
    %dy = stablehlo.subtract %lsm, %onehot : tensor<128x10xf32>
    // ── backward (mlpBackGraph): dotOut + select masks reading %h1,%h0 ──
    %dx2 = stablehlo.dot_general %dy, %W2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x10xf32>, tensor<512x10xf32>) -> tensor<128x512xf32>
    %bz1 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %bm1 = stablehlo.compare GT, %h1, %bz1 : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<128x512xi1>
    %dy1 = stablehlo.select %bm1, %dx2, %bz1 : tensor<128x512xi1>, tensor<128x512xf32>
    %dx1 = stablehlo.dot_general %dy1, %W1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<512x512xf32>) -> tensor<128x512xf32>
    %bz0 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %bm0 = stablehlo.compare GT, %h0, %bz0 : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<128x512xi1>
    %dy0 = stablehlo.select %bm0, %dx1, %bz0 : tensor<128x512xi1>, tensor<128x512xf32>
    // ── param grads (wGrad/bGrad) ──
    %sc = stablehlo.constant dense<0.0> : tensor<f32>
    %dW2 = stablehlo.dot_general %a1, %dy, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<128x10xf32>) -> tensor<512x10xf32>
    %db2 = stablehlo.reduce(%dy init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<128x10xf32>, tensor<f32>) -> tensor<10xf32>
    %dW1 = stablehlo.dot_general %a0, %dy1, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<512x512xf32>
    %db1 = stablehlo.reduce(%dy1 init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<128x512xf32>, tensor<f32>) -> tensor<512xf32>
    %dW0 = stablehlo.dot_general %x, %dy0, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x784xf32>, tensor<128x512xf32>) -> tensor<784x512xf32>
    %db0 = stablehlo.reduce(%dy0 init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<128x512xf32>, tensor<f32>) -> tensor<512xf32>
    // ── SGD θ' = θ − lr·∇ ──
    %W0l = stablehlo.constant dense<0.00078125> : tensor<784x512xf32>
    %W0s = stablehlo.multiply %dW0, %W0l : tensor<784x512xf32>
    %W0n = stablehlo.subtract %W0, %W0s : tensor<784x512xf32>
    %b0l = stablehlo.constant dense<0.00078125> : tensor<512xf32>
    %b0s = stablehlo.multiply %db0, %b0l : tensor<512xf32>
    %b0n = stablehlo.subtract %b0, %b0s : tensor<512xf32>
    %W1l = stablehlo.constant dense<0.00078125> : tensor<512x512xf32>
    %W1s = stablehlo.multiply %dW1, %W1l : tensor<512x512xf32>
    %W1n = stablehlo.subtract %W1, %W1s : tensor<512x512xf32>
    %b1l = stablehlo.constant dense<0.00078125> : tensor<512xf32>
    %b1s = stablehlo.multiply %db1, %b1l : tensor<512xf32>
    %b1n = stablehlo.subtract %b1, %b1s : tensor<512xf32>
    %W2l = stablehlo.constant dense<0.00078125> : tensor<512x10xf32>
    %W2s = stablehlo.multiply %dW2, %W2l : tensor<512x10xf32>
    %W2n = stablehlo.subtract %W2, %W2s : tensor<512x10xf32>
    %b2l = stablehlo.constant dense<0.00078125> : tensor<10xf32>
    %b2s = stablehlo.multiply %db2, %b2l : tensor<10xf32>
    %b2n = stablehlo.subtract %b2, %b2s : tensor<10xf32>
    return %W0n, %b0n, %W1n, %b1n, %W2n, %b2n : tensor<784x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>
  }
}

module @m {
  func.func @cnn_train_step(%x: tensor<128x784xf32>, %W1: tensor<32x1x3x3xf32>, %b1: tensor<32xf32>, %W2: tensor<32x32x3x3xf32>, %b2: tensor<32xf32>, %W3: tensor<6272x512xf32>, %b3: tensor<512xf32>, %W4: tensor<512x512xf32>, %b4: tensor<512xf32>, %W5: tensor<512x10xf32>, %b5: tensor<10xf32>, %onehot: tensor<128x10xf32>) -> (tensor<32x1x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<6272x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>) {
    %sc = stablehlo.constant dense<0.0> : tensor<f32>
    // ── forward: conv→relu→conv→relu→maxpool→flatten→dense→relu→dense→relu→dense ──
    %xr = stablehlo.reshape %x : (tensor<128x784xf32>) -> tensor<128x1x28x28xf32>
    %hc1c = stablehlo.convolution(%xr, %W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x1x28x28xf32>, tensor<32x1x3x3xf32>) -> tensor<128x32x28x28xf32>
    %hc1b = stablehlo.broadcast_in_dim %b1, dims = [1] : (tensor<32xf32>) -> tensor<128x32x28x28xf32>
    %hc1 = stablehlo.add %hc1c, %hc1b : tensor<128x32x28x28xf32>
    %ac1z = stablehlo.constant dense<0.0> : tensor<128x32x28x28xf32>
    %ac1 = stablehlo.maximum %hc1, %ac1z : tensor<128x32x28x28xf32>
    %hc2c = stablehlo.convolution(%ac1, %W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x28x28xf32>
    %hc2b = stablehlo.broadcast_in_dim %b2, dims = [1] : (tensor<32xf32>) -> tensor<128x32x28x28xf32>
    %hc2 = stablehlo.add %hc2c, %hc2b : tensor<128x32x28x28xf32>
    %ac2z = stablehlo.constant dense<0.0> : tensor<128x32x28x28xf32>
    %ac2 = stablehlo.maximum %hc2, %ac2z : tensor<128x32x28x28xf32>
    %poolninf = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %pool = "stablehlo.reduce_window"(%ac2, %poolninf) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x28x28xf32>, tensor<f32>) -> tensor<128x32x14x14xf32>
    %flat = stablehlo.reshape %pool : (tensor<128x32x14x14xf32>) -> tensor<128x6272xf32>
    %h3d = stablehlo.dot_general %flat, %W3, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x6272xf32>, tensor<6272x512xf32>) -> tensor<128x512xf32>
    %h3b = stablehlo.broadcast_in_dim %b3, dims = [1] : (tensor<512xf32>) -> tensor<128x512xf32>
    %h3 = stablehlo.add %h3d, %h3b : tensor<128x512xf32>
    %a3z = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %a3 = stablehlo.maximum %h3, %a3z : tensor<128x512xf32>
    %h4d = stablehlo.dot_general %a3, %W4, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<512x512xf32>) -> tensor<128x512xf32>
    %h4b = stablehlo.broadcast_in_dim %b4, dims = [1] : (tensor<512xf32>) -> tensor<128x512xf32>
    %h4 = stablehlo.add %h4d, %h4b : tensor<128x512xf32>
    %a4z = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %a4 = stablehlo.maximum %h4, %a4z : tensor<128x512xf32>
    %logitsd = stablehlo.dot_general %a4, %W5, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<512x10xf32>) -> tensor<128x10xf32>
    %logitsb = stablehlo.broadcast_in_dim %b5, dims = [1] : (tensor<10xf32>) -> tensor<128x10xf32>
    %logits = stablehlo.add %logitsd, %logitsb : tensor<128x10xf32>
    // ── loss cotangent dy = softmax(logits) − onehot (lossCotGraph_isCEgrad) ──
    %le = stablehlo.exponential %logits : tensor<128x10xf32>
    %lsum = stablehlo.reduce(%le init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<128x10xf32>, tensor<f32>) -> tensor<128xf32>
    %lsb = stablehlo.broadcast_in_dim %lsum, dims = [0] : (tensor<128xf32>) -> tensor<128x10xf32>
    %lsm = stablehlo.divide %le, %lsb : tensor<128x10xf32>
    %dy = stablehlo.subtract %lsm, %onehot : tensor<128x10xf32>
    // ── backward: dense (dotOut) + relu masks → reshape → select_and_scatter → convBack ──
    %dx5 = stablehlo.dot_general %dy, %W5, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x10xf32>, tensor<512x10xf32>) -> tensor<128x512xf32>
    %dy4z = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %dy4m = stablehlo.compare GT, %h4, %dy4z : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<128x512xi1>
    %dy4 = stablehlo.select %dy4m, %dx5, %dy4z : tensor<128x512xi1>, tensor<128x512xf32>
    %dx4 = stablehlo.dot_general %dy4, %W4, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<512x512xf32>) -> tensor<128x512xf32>
    %dy3z = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %dy3m = stablehlo.compare GT, %h3, %dy3z : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<128x512xi1>
    %dy3 = stablehlo.select %dy3m, %dx4, %dy3z : tensor<128x512xi1>, tensor<128x512xf32>
    %dx3 = stablehlo.dot_general %dy3, %W3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<6272x512xf32>) -> tensor<128x6272xf32>
    %dpool = stablehlo.reshape %dx3 : (tensor<128x6272xf32>) -> tensor<128x32x14x14xf32>
    %dac2 = "stablehlo.select_and_scatter"(%ac2, %dpool, %sc) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%su: tensor<f32>, %sv: tensor<f32>):
        %ss = stablehlo.add %su, %sv : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x28x28xf32>, tensor<128x32x14x14xf32>, tensor<f32>) -> tensor<128x32x28x28xf32>
    %dhc2z = stablehlo.constant dense<0.0> : tensor<128x32x28x28xf32>
    %dhc2m = stablehlo.compare GT, %hc2, %dhc2z : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x32x28x28xi1>
    %dhc2 = stablehlo.select %dhc2m, %dac2, %dhc2z : tensor<128x32x28x28xi1>, tensor<128x32x28x28xf32>
    %dac1t = stablehlo.transpose %W2, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %dac1r = stablehlo.reverse %dac1t, dims = [2, 3] : tensor<32x32x3x3xf32>
    %dac1 = stablehlo.convolution(%dhc2, %dac1r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x28x28xf32>
    %dhc1z = stablehlo.constant dense<0.0> : tensor<128x32x28x28xf32>
    %dhc1m = stablehlo.compare GT, %hc1, %dhc1z : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x32x28x28xi1>
    %dhc1 = stablehlo.select %dhc1m, %dac1, %dhc1z : tensor<128x32x28x28xi1>, tensor<128x32x28x28xf32>
    // ── param grads: dense W/b (dot_general/reduce); conv dW (transpose trick), db (reduce) ──
    %dW5 = stablehlo.dot_general %a4, %dy, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<128x10xf32>) -> tensor<512x10xf32>
    %db5 = stablehlo.reduce(%dy init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<128x10xf32>, tensor<f32>) -> tensor<10xf32>
    %dW4 = stablehlo.dot_general %a3, %dy4, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<512x512xf32>
    %db4 = stablehlo.reduce(%dy4 init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<128x512xf32>, tensor<f32>) -> tensor<512xf32>
    %dW3 = stablehlo.dot_general %flat, %dy3, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x6272xf32>, tensor<128x512xf32>) -> tensor<6272x512xf32>
    %db3 = stablehlo.reduce(%dy3 init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<128x512xf32>, tensor<f32>) -> tensor<512xf32>
    %dW2xt = stablehlo.transpose %ac1, dims = [1, 0, 2, 3] : (tensor<128x32x28x28xf32>) -> tensor<32x128x28x28xf32>
    %dW2dt = stablehlo.transpose %dhc2, dims = [1, 0, 2, 3] : (tensor<128x32x28x28xf32>) -> tensor<32x128x28x28xf32>
    %dW2raw = stablehlo.convolution(%dW2xt, %dW2dt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x32x3x3xf32>
    %dW2 = stablehlo.transpose %dW2raw, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %db2 = stablehlo.reduce(%dhc2 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %dW1xt = stablehlo.transpose %xr, dims = [1, 0, 2, 3] : (tensor<128x1x28x28xf32>) -> tensor<1x128x28x28xf32>
    %dW1dt = stablehlo.transpose %dhc1, dims = [1, 0, 2, 3] : (tensor<128x32x28x28xf32>) -> tensor<32x128x28x28xf32>
    %dW1raw = stablehlo.convolution(%dW1xt, %dW1dt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<1x32x3x3xf32>
    %dW1 = stablehlo.transpose %dW1raw, dims = [1, 0, 2, 3] : (tensor<1x32x3x3xf32>) -> tensor<32x1x3x3xf32>
    %db1 = stablehlo.reduce(%dhc1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    // ── SGD θ' = θ − lr·∇ (all 10 params) ──
    %W1l = stablehlo.constant dense<0.00078125> : tensor<32x1x3x3xf32>
    %W1s = stablehlo.multiply %dW1, %W1l : tensor<32x1x3x3xf32>
    %W1n = stablehlo.subtract %W1, %W1s : tensor<32x1x3x3xf32>
    %b1l = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %b1s = stablehlo.multiply %db1, %b1l : tensor<32xf32>
    %b1n = stablehlo.subtract %b1, %b1s : tensor<32xf32>
    %W2l = stablehlo.constant dense<0.00078125> : tensor<32x32x3x3xf32>
    %W2s = stablehlo.multiply %dW2, %W2l : tensor<32x32x3x3xf32>
    %W2n = stablehlo.subtract %W2, %W2s : tensor<32x32x3x3xf32>
    %b2l = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %b2s = stablehlo.multiply %db2, %b2l : tensor<32xf32>
    %b2n = stablehlo.subtract %b2, %b2s : tensor<32xf32>
    %W3l = stablehlo.constant dense<0.00078125> : tensor<6272x512xf32>
    %W3s = stablehlo.multiply %dW3, %W3l : tensor<6272x512xf32>
    %W3n = stablehlo.subtract %W3, %W3s : tensor<6272x512xf32>
    %b3l = stablehlo.constant dense<0.00078125> : tensor<512xf32>
    %b3s = stablehlo.multiply %db3, %b3l : tensor<512xf32>
    %b3n = stablehlo.subtract %b3, %b3s : tensor<512xf32>
    %W4l = stablehlo.constant dense<0.00078125> : tensor<512x512xf32>
    %W4s = stablehlo.multiply %dW4, %W4l : tensor<512x512xf32>
    %W4n = stablehlo.subtract %W4, %W4s : tensor<512x512xf32>
    %b4l = stablehlo.constant dense<0.00078125> : tensor<512xf32>
    %b4s = stablehlo.multiply %db4, %b4l : tensor<512xf32>
    %b4n = stablehlo.subtract %b4, %b4s : tensor<512xf32>
    %W5l = stablehlo.constant dense<0.00078125> : tensor<512x10xf32>
    %W5s = stablehlo.multiply %dW5, %W5l : tensor<512x10xf32>
    %W5n = stablehlo.subtract %W5, %W5s : tensor<512x10xf32>
    %b5l = stablehlo.constant dense<0.00078125> : tensor<10xf32>
    %b5s = stablehlo.multiply %db5, %b5l : tensor<10xf32>
    %b5n = stablehlo.subtract %b5, %b5s : tensor<10xf32>
    return %W1n, %b1n, %W2n, %b2n, %W3n, %b3n, %W4n, %b4n, %W5n, %b5n : tensor<32x1x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<6272x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>
  }
}

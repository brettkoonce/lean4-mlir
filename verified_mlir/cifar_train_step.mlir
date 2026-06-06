module @m {
  func.func @cifar_train_step(%x: tensor<128x3072xf32>, %W1: tensor<32x3x3x3xf32>, %b1: tensor<32xf32>, %W2: tensor<32x32x3x3xf32>, %b2: tensor<32xf32>, %W3: tensor<64x32x3x3xf32>, %b3: tensor<64xf32>, %W4: tensor<64x64x3x3xf32>, %b4: tensor<64xf32>, %W5: tensor<4096x512xf32>, %b5: tensor<512xf32>, %W6: tensor<512x512xf32>, %b6: tensor<512xf32>, %W7: tensor<512x10xf32>, %b7: tensor<10xf32>, %onehot: tensor<128x10xf32>) -> (tensor<32x3x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<64x32x3x3xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<4096x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>) {
    %sc = stablehlo.constant dense<0.0> : tensor<f32>
    // ── forward: (conv→relu)×2→pool →(conv→relu)×2→pool →flatten→(dense→relu)×2→dense ──
    %xr = stablehlo.reshape %x : (tensor<128x3072xf32>) -> tensor<128x3x32x32xf32>
    %hc1c = stablehlo.convolution(%xr, %W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x3x32x32xf32>, tensor<32x3x3x3xf32>) -> tensor<128x32x32x32xf32>
    %hc1b = stablehlo.broadcast_in_dim %b1, dims = [1] : (tensor<32xf32>) -> tensor<128x32x32x32xf32>
    %hc1 = stablehlo.add %hc1c, %hc1b : tensor<128x32x32x32xf32>
    %ac1z = stablehlo.constant dense<0.0> : tensor<128x32x32x32xf32>
    %ac1 = stablehlo.maximum %hc1, %ac1z : tensor<128x32x32x32xf32>
    %hc2c = stablehlo.convolution(%ac1, %W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x32x32xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x32x32xf32>
    %hc2b = stablehlo.broadcast_in_dim %b2, dims = [1] : (tensor<32xf32>) -> tensor<128x32x32x32xf32>
    %hc2 = stablehlo.add %hc2c, %hc2b : tensor<128x32x32x32xf32>
    %ac2z = stablehlo.constant dense<0.0> : tensor<128x32x32x32xf32>
    %ac2 = stablehlo.maximum %hc2, %ac2z : tensor<128x32x32x32xf32>
    %pool1ninf = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %pool1 = "stablehlo.reduce_window"(%ac2, %pool1ninf) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x32x32xf32>, tensor<f32>) -> tensor<128x32x16x16xf32>
    %hc3c = stablehlo.convolution(%pool1, %W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x16x16xf32>, tensor<64x32x3x3xf32>) -> tensor<128x64x16x16xf32>
    %hc3b = stablehlo.broadcast_in_dim %b3, dims = [1] : (tensor<64xf32>) -> tensor<128x64x16x16xf32>
    %hc3 = stablehlo.add %hc3c, %hc3b : tensor<128x64x16x16xf32>
    %ac3z = stablehlo.constant dense<0.0> : tensor<128x64x16x16xf32>
    %ac3 = stablehlo.maximum %hc3, %ac3z : tensor<128x64x16x16xf32>
    %hc4c = stablehlo.convolution(%ac3, %W4)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x16x16xf32>, tensor<64x64x3x3xf32>) -> tensor<128x64x16x16xf32>
    %hc4b = stablehlo.broadcast_in_dim %b4, dims = [1] : (tensor<64xf32>) -> tensor<128x64x16x16xf32>
    %hc4 = stablehlo.add %hc4c, %hc4b : tensor<128x64x16x16xf32>
    %ac4z = stablehlo.constant dense<0.0> : tensor<128x64x16x16xf32>
    %ac4 = stablehlo.maximum %hc4, %ac4z : tensor<128x64x16x16xf32>
    %pool2ninf = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %pool2 = "stablehlo.reduce_window"(%ac4, %pool2ninf) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x64x16x16xf32>, tensor<f32>) -> tensor<128x64x8x8xf32>
    %flat = stablehlo.reshape %pool2 : (tensor<128x64x8x8xf32>) -> tensor<128x4096xf32>
    %h5d = stablehlo.dot_general %flat, %W5, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x4096xf32>, tensor<4096x512xf32>) -> tensor<128x512xf32>
    %h5b = stablehlo.broadcast_in_dim %b5, dims = [1] : (tensor<512xf32>) -> tensor<128x512xf32>
    %h5 = stablehlo.add %h5d, %h5b : tensor<128x512xf32>
    %a5z = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %a5 = stablehlo.maximum %h5, %a5z : tensor<128x512xf32>
    %h6d = stablehlo.dot_general %a5, %W6, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<512x512xf32>) -> tensor<128x512xf32>
    %h6b = stablehlo.broadcast_in_dim %b6, dims = [1] : (tensor<512xf32>) -> tensor<128x512xf32>
    %h6 = stablehlo.add %h6d, %h6b : tensor<128x512xf32>
    %a6z = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %a6 = stablehlo.maximum %h6, %a6z : tensor<128x512xf32>
    %logitsd = stablehlo.dot_general %a6, %W7, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<512x10xf32>) -> tensor<128x10xf32>
    %logitsb = stablehlo.broadcast_in_dim %b7, dims = [1] : (tensor<10xf32>) -> tensor<128x10xf32>
    %logits = stablehlo.add %logitsd, %logitsb : tensor<128x10xf32>
    // ── loss cotangent dy = softmax(logits) − onehot (lossCotGraph_isCEgrad) ──
    %le = stablehlo.exponential %logits : tensor<128x10xf32>
    %lsum = stablehlo.reduce(%le init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<128x10xf32>, tensor<f32>) -> tensor<128xf32>
    %lsb = stablehlo.broadcast_in_dim %lsum, dims = [0] : (tensor<128xf32>) -> tensor<128x10xf32>
    %lsm = stablehlo.divide %le, %lsb : tensor<128x10xf32>
    %dy = stablehlo.subtract %lsm, %onehot : tensor<128x10xf32>
    // ── backward: dense (dotOut)+relu masks → scatter → convBack, twice through ──
    %dx7 = stablehlo.dot_general %dy, %W7, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x10xf32>, tensor<512x10xf32>) -> tensor<128x512xf32>
    %dy6z = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %dy6m = stablehlo.compare GT, %h6, %dy6z : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<128x512xi1>
    %dy6 = stablehlo.select %dy6m, %dx7, %dy6z : tensor<128x512xi1>, tensor<128x512xf32>
    %dx6 = stablehlo.dot_general %dy6, %W6, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<512x512xf32>) -> tensor<128x512xf32>
    %dy5z = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %dy5m = stablehlo.compare GT, %h5, %dy5z : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<128x512xi1>
    %dy5 = stablehlo.select %dy5m, %dx6, %dy5z : tensor<128x512xi1>, tensor<128x512xf32>
    %dx5 = stablehlo.dot_general %dy5, %W5, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<4096x512xf32>) -> tensor<128x4096xf32>
    %dpool2 = stablehlo.reshape %dx5 : (tensor<128x4096xf32>) -> tensor<128x64x8x8xf32>
    %dac4 = "stablehlo.select_and_scatter"(%ac4, %dpool2, %sc) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%su: tensor<f32>, %sv: tensor<f32>):
        %ss = stablehlo.add %su, %sv : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x64x16x16xf32>, tensor<128x64x8x8xf32>, tensor<f32>) -> tensor<128x64x16x16xf32>
    %dhc4z = stablehlo.constant dense<0.0> : tensor<128x64x16x16xf32>
    %dhc4m = stablehlo.compare GT, %hc4, %dhc4z : (tensor<128x64x16x16xf32>, tensor<128x64x16x16xf32>) -> tensor<128x64x16x16xi1>
    %dhc4 = stablehlo.select %dhc4m, %dac4, %dhc4z : tensor<128x64x16x16xi1>, tensor<128x64x16x16xf32>
    %dac3t = stablehlo.transpose %W4, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %dac3r = stablehlo.reverse %dac3t, dims = [2, 3] : tensor<64x64x3x3xf32>
    %dac3 = stablehlo.convolution(%dhc4, %dac3r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x16x16xf32>, tensor<64x64x3x3xf32>) -> tensor<128x64x16x16xf32>
    %dhc3z = stablehlo.constant dense<0.0> : tensor<128x64x16x16xf32>
    %dhc3m = stablehlo.compare GT, %hc3, %dhc3z : (tensor<128x64x16x16xf32>, tensor<128x64x16x16xf32>) -> tensor<128x64x16x16xi1>
    %dhc3 = stablehlo.select %dhc3m, %dac3, %dhc3z : tensor<128x64x16x16xi1>, tensor<128x64x16x16xf32>
    %dpool1t = stablehlo.transpose %W3, dims = [1, 0, 2, 3] : (tensor<64x32x3x3xf32>) -> tensor<32x64x3x3xf32>
    %dpool1r = stablehlo.reverse %dpool1t, dims = [2, 3] : tensor<32x64x3x3xf32>
    %dpool1 = stablehlo.convolution(%dhc3, %dpool1r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x16x16xf32>, tensor<32x64x3x3xf32>) -> tensor<128x32x16x16xf32>
    %dac2 = "stablehlo.select_and_scatter"(%ac2, %dpool1, %sc) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%su: tensor<f32>, %sv: tensor<f32>):
        %ss = stablehlo.add %su, %sv : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x32x32xf32>, tensor<128x32x16x16xf32>, tensor<f32>) -> tensor<128x32x32x32xf32>
    %dhc2z = stablehlo.constant dense<0.0> : tensor<128x32x32x32xf32>
    %dhc2m = stablehlo.compare GT, %hc2, %dhc2z : (tensor<128x32x32x32xf32>, tensor<128x32x32x32xf32>) -> tensor<128x32x32x32xi1>
    %dhc2 = stablehlo.select %dhc2m, %dac2, %dhc2z : tensor<128x32x32x32xi1>, tensor<128x32x32x32xf32>
    %dac1t = stablehlo.transpose %W2, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %dac1r = stablehlo.reverse %dac1t, dims = [2, 3] : tensor<32x32x3x3xf32>
    %dac1 = stablehlo.convolution(%dhc2, %dac1r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x32x32xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x32x32xf32>
    %dhc1z = stablehlo.constant dense<0.0> : tensor<128x32x32x32xf32>
    %dhc1m = stablehlo.compare GT, %hc1, %dhc1z : (tensor<128x32x32x32xf32>, tensor<128x32x32x32xf32>) -> tensor<128x32x32x32xi1>
    %dhc1 = stablehlo.select %dhc1m, %dac1, %dhc1z : tensor<128x32x32x32xi1>, tensor<128x32x32x32xf32>
    // ── param grads: dense W/b (dot_general/reduce); conv dW (transpose trick), db (reduce) ──
    %dW7 = stablehlo.dot_general %a6, %dy, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<128x10xf32>) -> tensor<512x10xf32>
    %db7 = stablehlo.reduce(%dy init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<128x10xf32>, tensor<f32>) -> tensor<10xf32>
    %dW6 = stablehlo.dot_general %a5, %dy6, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<512x512xf32>
    %db6 = stablehlo.reduce(%dy6 init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<128x512xf32>, tensor<f32>) -> tensor<512xf32>
    %dW5 = stablehlo.dot_general %flat, %dy5, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x4096xf32>, tensor<128x512xf32>) -> tensor<4096x512xf32>
    %db5 = stablehlo.reduce(%dy5 init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<128x512xf32>, tensor<f32>) -> tensor<512xf32>
    %dW4xt = stablehlo.transpose %ac3, dims = [1, 0, 2, 3] : (tensor<128x64x16x16xf32>) -> tensor<64x128x16x16xf32>
    %dW4dt = stablehlo.transpose %dhc4, dims = [1, 0, 2, 3] : (tensor<128x64x16x16xf32>) -> tensor<64x128x16x16xf32>
    %dW4raw = stablehlo.convolution(%dW4xt, %dW4dt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x16x16xf32>, tensor<64x128x16x16xf32>) -> tensor<64x64x3x3xf32>
    %dW4 = stablehlo.transpose %dW4raw, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %db4 = stablehlo.reduce(%dhc4 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x16x16xf32>, tensor<f32>) -> tensor<64xf32>
    %dW3xt = stablehlo.transpose %pool1, dims = [1, 0, 2, 3] : (tensor<128x32x16x16xf32>) -> tensor<32x128x16x16xf32>
    %dW3dt = stablehlo.transpose %dhc3, dims = [1, 0, 2, 3] : (tensor<128x64x16x16xf32>) -> tensor<64x128x16x16xf32>
    %dW3raw = stablehlo.convolution(%dW3xt, %dW3dt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x16x16xf32>, tensor<64x128x16x16xf32>) -> tensor<32x64x3x3xf32>
    %dW3 = stablehlo.transpose %dW3raw, dims = [1, 0, 2, 3] : (tensor<32x64x3x3xf32>) -> tensor<64x32x3x3xf32>
    %db3 = stablehlo.reduce(%dhc3 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x16x16xf32>, tensor<f32>) -> tensor<64xf32>
    %dW2xt = stablehlo.transpose %ac1, dims = [1, 0, 2, 3] : (tensor<128x32x32x32xf32>) -> tensor<32x128x32x32xf32>
    %dW2dt = stablehlo.transpose %dhc2, dims = [1, 0, 2, 3] : (tensor<128x32x32x32xf32>) -> tensor<32x128x32x32xf32>
    %dW2raw = stablehlo.convolution(%dW2xt, %dW2dt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x32x32xf32>, tensor<32x128x32x32xf32>) -> tensor<32x32x3x3xf32>
    %dW2 = stablehlo.transpose %dW2raw, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %db2 = stablehlo.reduce(%dhc2 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x32x32xf32>, tensor<f32>) -> tensor<32xf32>
    %dW1xt = stablehlo.transpose %xr, dims = [1, 0, 2, 3] : (tensor<128x3x32x32xf32>) -> tensor<3x128x32x32xf32>
    %dW1dt = stablehlo.transpose %dhc1, dims = [1, 0, 2, 3] : (tensor<128x32x32x32xf32>) -> tensor<32x128x32x32xf32>
    %dW1raw = stablehlo.convolution(%dW1xt, %dW1dt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x128x32x32xf32>, tensor<32x128x32x32xf32>) -> tensor<3x32x3x3xf32>
    %dW1 = stablehlo.transpose %dW1raw, dims = [1, 0, 2, 3] : (tensor<3x32x3x3xf32>) -> tensor<32x3x3x3xf32>
    %db1 = stablehlo.reduce(%dhc1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x32x32xf32>, tensor<f32>) -> tensor<32xf32>
    // ── SGD θ' = θ − lr·∇ (all 14 params) ──
    %W1l = stablehlo.constant dense<0.00078125> : tensor<32x3x3x3xf32>
    %W1s = stablehlo.multiply %dW1, %W1l : tensor<32x3x3x3xf32>
    %W1n = stablehlo.subtract %W1, %W1s : tensor<32x3x3x3xf32>
    %b1l = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %b1s = stablehlo.multiply %db1, %b1l : tensor<32xf32>
    %b1n = stablehlo.subtract %b1, %b1s : tensor<32xf32>
    %W2l = stablehlo.constant dense<0.00078125> : tensor<32x32x3x3xf32>
    %W2s = stablehlo.multiply %dW2, %W2l : tensor<32x32x3x3xf32>
    %W2n = stablehlo.subtract %W2, %W2s : tensor<32x32x3x3xf32>
    %b2l = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %b2s = stablehlo.multiply %db2, %b2l : tensor<32xf32>
    %b2n = stablehlo.subtract %b2, %b2s : tensor<32xf32>
    %W3l = stablehlo.constant dense<0.00078125> : tensor<64x32x3x3xf32>
    %W3s = stablehlo.multiply %dW3, %W3l : tensor<64x32x3x3xf32>
    %W3n = stablehlo.subtract %W3, %W3s : tensor<64x32x3x3xf32>
    %b3l = stablehlo.constant dense<0.00078125> : tensor<64xf32>
    %b3s = stablehlo.multiply %db3, %b3l : tensor<64xf32>
    %b3n = stablehlo.subtract %b3, %b3s : tensor<64xf32>
    %W4l = stablehlo.constant dense<0.00078125> : tensor<64x64x3x3xf32>
    %W4s = stablehlo.multiply %dW4, %W4l : tensor<64x64x3x3xf32>
    %W4n = stablehlo.subtract %W4, %W4s : tensor<64x64x3x3xf32>
    %b4l = stablehlo.constant dense<0.00078125> : tensor<64xf32>
    %b4s = stablehlo.multiply %db4, %b4l : tensor<64xf32>
    %b4n = stablehlo.subtract %b4, %b4s : tensor<64xf32>
    %W5l = stablehlo.constant dense<0.00078125> : tensor<4096x512xf32>
    %W5s = stablehlo.multiply %dW5, %W5l : tensor<4096x512xf32>
    %W5n = stablehlo.subtract %W5, %W5s : tensor<4096x512xf32>
    %b5l = stablehlo.constant dense<0.00078125> : tensor<512xf32>
    %b5s = stablehlo.multiply %db5, %b5l : tensor<512xf32>
    %b5n = stablehlo.subtract %b5, %b5s : tensor<512xf32>
    %W6l = stablehlo.constant dense<0.00078125> : tensor<512x512xf32>
    %W6s = stablehlo.multiply %dW6, %W6l : tensor<512x512xf32>
    %W6n = stablehlo.subtract %W6, %W6s : tensor<512x512xf32>
    %b6l = stablehlo.constant dense<0.00078125> : tensor<512xf32>
    %b6s = stablehlo.multiply %db6, %b6l : tensor<512xf32>
    %b6n = stablehlo.subtract %b6, %b6s : tensor<512xf32>
    %W7l = stablehlo.constant dense<0.00078125> : tensor<512x10xf32>
    %W7s = stablehlo.multiply %dW7, %W7l : tensor<512x10xf32>
    %W7n = stablehlo.subtract %W7, %W7s : tensor<512x10xf32>
    %b7l = stablehlo.constant dense<0.00078125> : tensor<10xf32>
    %b7s = stablehlo.multiply %db7, %b7l : tensor<10xf32>
    %b7n = stablehlo.subtract %b7, %b7s : tensor<10xf32>
    return %W1n, %b1n, %W2n, %b2n, %W3n, %b3n, %W4n, %b4n, %W5n, %b5n, %W6n, %b6n, %W7n, %b7n : tensor<32x3x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<64x32x3x3xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<4096x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>
  }
}

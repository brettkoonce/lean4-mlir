module @m {
  func.func @cifar8_adam_train_step(%x: tensor<128x3072xf32>, %W1: tensor<16x3x3x3xf32>, %cb1: tensor<16xf32>, %W2: tensor<16x16x3x3xf32>, %cb2: tensor<16xf32>, %W3: tensor<16x16x3x3xf32>, %cb3: tensor<16xf32>, %W4: tensor<16x16x3x3xf32>, %cb4: tensor<16xf32>, %W5: tensor<32x16x3x3xf32>, %cb5: tensor<32xf32>, %W6: tensor<32x32x3x3xf32>, %cb6: tensor<32xf32>, %W7: tensor<32x32x3x3xf32>, %cb7: tensor<32xf32>, %W8: tensor<32x32x3x3xf32>, %cb8: tensor<32xf32>, %W9: tensor<128x64xf32>, %b9: tensor<64xf32>, %Wa: tensor<64x64xf32>, %ba: tensor<64xf32>, %Wb: tensor<64x10xf32>, %bb: tensor<10xf32>, %W1m: tensor<16x3x3x3xf32>, %cb1m: tensor<16xf32>, %W2m: tensor<16x16x3x3xf32>, %cb2m: tensor<16xf32>, %W3m: tensor<16x16x3x3xf32>, %cb3m: tensor<16xf32>, %W4m: tensor<16x16x3x3xf32>, %cb4m: tensor<16xf32>, %W5m: tensor<32x16x3x3xf32>, %cb5m: tensor<32xf32>, %W6m: tensor<32x32x3x3xf32>, %cb6m: tensor<32xf32>, %W7m: tensor<32x32x3x3xf32>, %cb7m: tensor<32xf32>, %W8m: tensor<32x32x3x3xf32>, %cb8m: tensor<32xf32>, %W9m: tensor<128x64xf32>, %b9m: tensor<64xf32>, %Wam: tensor<64x64xf32>, %bam: tensor<64xf32>, %Wbm: tensor<64x10xf32>, %bbm: tensor<10xf32>, %W1v: tensor<16x3x3x3xf32>, %cb1v: tensor<16xf32>, %W2v: tensor<16x16x3x3xf32>, %cb2v: tensor<16xf32>, %W3v: tensor<16x16x3x3xf32>, %cb3v: tensor<16xf32>, %W4v: tensor<16x16x3x3xf32>, %cb4v: tensor<16xf32>, %W5v: tensor<32x16x3x3xf32>, %cb5v: tensor<32xf32>, %W6v: tensor<32x32x3x3xf32>, %cb6v: tensor<32xf32>, %W7v: tensor<32x32x3x3xf32>, %cb7v: tensor<32xf32>, %W8v: tensor<32x32x3x3xf32>, %cb8v: tensor<32xf32>, %W9v: tensor<128x64xf32>, %b9v: tensor<64xf32>, %Wav: tensor<64x64xf32>, %bav: tensor<64xf32>, %Wbv: tensor<64x10xf32>, %bbv: tensor<10xf32>, %lr: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>, %onehot: tensor<128x10xf32>) -> (tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>) {
    %sc = stablehlo.constant dense<0.0> : tensor<f32>
    // ── forward: (conv→relu)×2→pool ×4 →flatten→(dense→relu)×2→dense ──
    %xr = stablehlo.reshape %x : (tensor<128x3072xf32>) -> tensor<128x3x32x32xf32>
    %hc1c = stablehlo.convolution(%xr, %W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x3x32x32xf32>, tensor<16x3x3x3xf32>) -> tensor<128x16x32x32xf32>
    %hc1b = stablehlo.broadcast_in_dim %cb1, dims = [1] : (tensor<16xf32>) -> tensor<128x16x32x32xf32>
    %hc1 = stablehlo.add %hc1c, %hc1b : tensor<128x16x32x32xf32>
    %ac1z = stablehlo.constant dense<0.0> : tensor<128x16x32x32xf32>
    %ac1 = stablehlo.maximum %hc1, %ac1z : tensor<128x16x32x32xf32>
    %hc2c = stablehlo.convolution(%ac1, %W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x32x32xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x32x32xf32>
    %hc2b = stablehlo.broadcast_in_dim %cb2, dims = [1] : (tensor<16xf32>) -> tensor<128x16x32x32xf32>
    %hc2 = stablehlo.add %hc2c, %hc2b : tensor<128x16x32x32xf32>
    %ac2z = stablehlo.constant dense<0.0> : tensor<128x16x32x32xf32>
    %ac2 = stablehlo.maximum %hc2, %ac2z : tensor<128x16x32x32xf32>
    %pool1ninf = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %pool1 = "stablehlo.reduce_window"(%ac2, %pool1ninf) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16x16x16xf32>
    %hc3c = stablehlo.convolution(%pool1, %W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x16x16xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x16x16xf32>
    %hc3b = stablehlo.broadcast_in_dim %cb3, dims = [1] : (tensor<16xf32>) -> tensor<128x16x16x16xf32>
    %hc3 = stablehlo.add %hc3c, %hc3b : tensor<128x16x16x16xf32>
    %ac3z = stablehlo.constant dense<0.0> : tensor<128x16x16x16xf32>
    %ac3 = stablehlo.maximum %hc3, %ac3z : tensor<128x16x16x16xf32>
    %hc4c = stablehlo.convolution(%ac3, %W4)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x16x16xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x16x16xf32>
    %hc4b = stablehlo.broadcast_in_dim %cb4, dims = [1] : (tensor<16xf32>) -> tensor<128x16x16x16xf32>
    %hc4 = stablehlo.add %hc4c, %hc4b : tensor<128x16x16x16xf32>
    %ac4z = stablehlo.constant dense<0.0> : tensor<128x16x16x16xf32>
    %ac4 = stablehlo.maximum %hc4, %ac4z : tensor<128x16x16x16xf32>
    %pool2ninf = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %pool2 = "stablehlo.reduce_window"(%ac4, %pool2ninf) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16x8x8xf32>
    %hc5c = stablehlo.convolution(%pool2, %W5)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x8x8xf32>, tensor<32x16x3x3xf32>) -> tensor<128x32x8x8xf32>
    %hc5b = stablehlo.broadcast_in_dim %cb5, dims = [1] : (tensor<32xf32>) -> tensor<128x32x8x8xf32>
    %hc5 = stablehlo.add %hc5c, %hc5b : tensor<128x32x8x8xf32>
    %ac5z = stablehlo.constant dense<0.0> : tensor<128x32x8x8xf32>
    %ac5 = stablehlo.maximum %hc5, %ac5z : tensor<128x32x8x8xf32>
    %hc6c = stablehlo.convolution(%ac5, %W6)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x8x8xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x8x8xf32>
    %hc6b = stablehlo.broadcast_in_dim %cb6, dims = [1] : (tensor<32xf32>) -> tensor<128x32x8x8xf32>
    %hc6 = stablehlo.add %hc6c, %hc6b : tensor<128x32x8x8xf32>
    %ac6z = stablehlo.constant dense<0.0> : tensor<128x32x8x8xf32>
    %ac6 = stablehlo.maximum %hc6, %ac6z : tensor<128x32x8x8xf32>
    %pool3ninf = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %pool3 = "stablehlo.reduce_window"(%ac6, %pool3ninf) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32x4x4xf32>
    %hc7c = stablehlo.convolution(%pool3, %W7)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x4x4xf32>
    %hc7b = stablehlo.broadcast_in_dim %cb7, dims = [1] : (tensor<32xf32>) -> tensor<128x32x4x4xf32>
    %hc7 = stablehlo.add %hc7c, %hc7b : tensor<128x32x4x4xf32>
    %ac7z = stablehlo.constant dense<0.0> : tensor<128x32x4x4xf32>
    %ac7 = stablehlo.maximum %hc7, %ac7z : tensor<128x32x4x4xf32>
    %hc8c = stablehlo.convolution(%ac7, %W8)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x4x4xf32>
    %hc8b = stablehlo.broadcast_in_dim %cb8, dims = [1] : (tensor<32xf32>) -> tensor<128x32x4x4xf32>
    %hc8 = stablehlo.add %hc8c, %hc8b : tensor<128x32x4x4xf32>
    %ac8z = stablehlo.constant dense<0.0> : tensor<128x32x4x4xf32>
    %ac8 = stablehlo.maximum %hc8, %ac8z : tensor<128x32x4x4xf32>
    %pool4ninf = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %pool4 = "stablehlo.reduce_window"(%ac8, %pool4ninf) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32x2x2xf32>
    %flat = stablehlo.reshape %pool4 : (tensor<128x32x2x2xf32>) -> tensor<128x128xf32>
    %h9d = stablehlo.dot_general %flat, %W9, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x128xf32>, tensor<128x64xf32>) -> tensor<128x64xf32>
    %h9b = stablehlo.broadcast_in_dim %b9, dims = [1] : (tensor<64xf32>) -> tensor<128x64xf32>
    %h9 = stablehlo.add %h9d, %h9b : tensor<128x64xf32>
    %a9z = stablehlo.constant dense<0.0> : tensor<128x64xf32>
    %a9 = stablehlo.maximum %h9, %a9z : tensor<128x64xf32>
    %had = stablehlo.dot_general %a9, %Wa, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<64x64xf32>) -> tensor<128x64xf32>
    %hab = stablehlo.broadcast_in_dim %ba, dims = [1] : (tensor<64xf32>) -> tensor<128x64xf32>
    %ha = stablehlo.add %had, %hab : tensor<128x64xf32>
    %aaz = stablehlo.constant dense<0.0> : tensor<128x64xf32>
    %aa = stablehlo.maximum %ha, %aaz : tensor<128x64xf32>
    %logitsd = stablehlo.dot_general %aa, %Wb, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<64x10xf32>) -> tensor<128x10xf32>
    %logitsb = stablehlo.broadcast_in_dim %bb, dims = [1] : (tensor<10xf32>) -> tensor<128x10xf32>
    %logits = stablehlo.add %logitsd, %logitsb : tensor<128x10xf32>
    // ── mean loss cotangent dy = (softmax(logits) − onehot) / B + scalar %loss ──
    %le = stablehlo.exponential %logits : tensor<128x10xf32>
    %lsum = stablehlo.reduce(%le init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<128x10xf32>, tensor<f32>) -> tensor<128xf32>
    %lsb = stablehlo.broadcast_in_dim %lsum, dims = [0] : (tensor<128xf32>) -> tensor<128x10xf32>
    %lsm = stablehlo.divide %le, %lsb : tensor<128x10xf32>
    %dyr = stablehlo.subtract %lsm, %onehot : tensor<128x10xf32>
    %bnc = stablehlo.constant dense<128.0> : tensor<128x10xf32>
    %dy = stablehlo.divide %dyr, %bnc : tensor<128x10xf32>
    %llog = stablehlo.log %lsm : tensor<128x10xf32>
    %ohll = stablehlo.multiply %onehot, %llog : tensor<128x10xf32>
    %csum = stablehlo.reduce(%ohll init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<128x10xf32>, tensor<f32>) -> tensor<f32>
    %cneg = stablehlo.negate %csum : tensor<f32>
    %lbf = stablehlo.constant dense<128.0> : tensor<f32>
    %loss = stablehlo.divide %cneg, %lbf : tensor<f32>
    // ── backward: dense (dotOut)+relu masks → scatter → convBack, four stages ──
    %dxb = stablehlo.dot_general %dy, %Wb, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x10xf32>, tensor<64x10xf32>) -> tensor<128x64xf32>
    %dyaz = stablehlo.constant dense<0.0> : tensor<128x64xf32>
    %dyam = stablehlo.compare GT, %ha, %dyaz : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<128x64xi1>
    %dya = stablehlo.select %dyam, %dxb, %dyaz : tensor<128x64xi1>, tensor<128x64xf32>
    %dxa = stablehlo.dot_general %dya, %Wa, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<64x64xf32>) -> tensor<128x64xf32>
    %dy9z = stablehlo.constant dense<0.0> : tensor<128x64xf32>
    %dy9m = stablehlo.compare GT, %h9, %dy9z : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<128x64xi1>
    %dy9 = stablehlo.select %dy9m, %dxa, %dy9z : tensor<128x64xi1>, tensor<128x64xf32>
    %dx9 = stablehlo.dot_general %dy9, %W9, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<128x128xf32>
    %dpool4 = stablehlo.reshape %dx9 : (tensor<128x128xf32>) -> tensor<128x32x2x2xf32>
    %dac8 = "stablehlo.select_and_scatter"(%ac8, %dpool4, %sc) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%su: tensor<f32>, %sv: tensor<f32>):
        %ss = stablehlo.add %su, %sv : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x4x4xf32>, tensor<128x32x2x2xf32>, tensor<f32>) -> tensor<128x32x4x4xf32>
    %dhc8z = stablehlo.constant dense<0.0> : tensor<128x32x4x4xf32>
    %dhc8m = stablehlo.compare GT, %hc8, %dhc8z : (tensor<128x32x4x4xf32>, tensor<128x32x4x4xf32>) -> tensor<128x32x4x4xi1>
    %dhc8 = stablehlo.select %dhc8m, %dac8, %dhc8z : tensor<128x32x4x4xi1>, tensor<128x32x4x4xf32>
    %dac7t = stablehlo.transpose %W8, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %dac7r = stablehlo.reverse %dac7t, dims = [2, 3] : tensor<32x32x3x3xf32>
    %dac7 = stablehlo.convolution(%dhc8, %dac7r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x4x4xf32>
    %dhc7z = stablehlo.constant dense<0.0> : tensor<128x32x4x4xf32>
    %dhc7m = stablehlo.compare GT, %hc7, %dhc7z : (tensor<128x32x4x4xf32>, tensor<128x32x4x4xf32>) -> tensor<128x32x4x4xi1>
    %dhc7 = stablehlo.select %dhc7m, %dac7, %dhc7z : tensor<128x32x4x4xi1>, tensor<128x32x4x4xf32>
    %dpool3t = stablehlo.transpose %W7, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %dpool3r = stablehlo.reverse %dpool3t, dims = [2, 3] : tensor<32x32x3x3xf32>
    %dpool3 = stablehlo.convolution(%dhc7, %dpool3r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x4x4xf32>
    %dac6 = "stablehlo.select_and_scatter"(%ac6, %dpool3, %sc) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%su: tensor<f32>, %sv: tensor<f32>):
        %ss = stablehlo.add %su, %sv : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x8x8xf32>, tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32x8x8xf32>
    %dhc6z = stablehlo.constant dense<0.0> : tensor<128x32x8x8xf32>
    %dhc6m = stablehlo.compare GT, %hc6, %dhc6z : (tensor<128x32x8x8xf32>, tensor<128x32x8x8xf32>) -> tensor<128x32x8x8xi1>
    %dhc6 = stablehlo.select %dhc6m, %dac6, %dhc6z : tensor<128x32x8x8xi1>, tensor<128x32x8x8xf32>
    %dac5t = stablehlo.transpose %W6, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %dac5r = stablehlo.reverse %dac5t, dims = [2, 3] : tensor<32x32x3x3xf32>
    %dac5 = stablehlo.convolution(%dhc6, %dac5r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x8x8xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x8x8xf32>
    %dhc5z = stablehlo.constant dense<0.0> : tensor<128x32x8x8xf32>
    %dhc5m = stablehlo.compare GT, %hc5, %dhc5z : (tensor<128x32x8x8xf32>, tensor<128x32x8x8xf32>) -> tensor<128x32x8x8xi1>
    %dhc5 = stablehlo.select %dhc5m, %dac5, %dhc5z : tensor<128x32x8x8xi1>, tensor<128x32x8x8xf32>
    %dpool2t = stablehlo.transpose %W5, dims = [1, 0, 2, 3] : (tensor<32x16x3x3xf32>) -> tensor<16x32x3x3xf32>
    %dpool2r = stablehlo.reverse %dpool2t, dims = [2, 3] : tensor<16x32x3x3xf32>
    %dpool2 = stablehlo.convolution(%dhc5, %dpool2r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x8x8xf32>, tensor<16x32x3x3xf32>) -> tensor<128x16x8x8xf32>
    %dac4 = "stablehlo.select_and_scatter"(%ac4, %dpool2, %sc) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%su: tensor<f32>, %sv: tensor<f32>):
        %ss = stablehlo.add %su, %sv : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x16x16x16xf32>, tensor<128x16x8x8xf32>, tensor<f32>) -> tensor<128x16x16x16xf32>
    %dhc4z = stablehlo.constant dense<0.0> : tensor<128x16x16x16xf32>
    %dhc4m = stablehlo.compare GT, %hc4, %dhc4z : (tensor<128x16x16x16xf32>, tensor<128x16x16x16xf32>) -> tensor<128x16x16x16xi1>
    %dhc4 = stablehlo.select %dhc4m, %dac4, %dhc4z : tensor<128x16x16x16xi1>, tensor<128x16x16x16xf32>
    %dac3t = stablehlo.transpose %W4, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %dac3r = stablehlo.reverse %dac3t, dims = [2, 3] : tensor<16x16x3x3xf32>
    %dac3 = stablehlo.convolution(%dhc4, %dac3r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x16x16xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x16x16xf32>
    %dhc3z = stablehlo.constant dense<0.0> : tensor<128x16x16x16xf32>
    %dhc3m = stablehlo.compare GT, %hc3, %dhc3z : (tensor<128x16x16x16xf32>, tensor<128x16x16x16xf32>) -> tensor<128x16x16x16xi1>
    %dhc3 = stablehlo.select %dhc3m, %dac3, %dhc3z : tensor<128x16x16x16xi1>, tensor<128x16x16x16xf32>
    %dpool1t = stablehlo.transpose %W3, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %dpool1r = stablehlo.reverse %dpool1t, dims = [2, 3] : tensor<16x16x3x3xf32>
    %dpool1 = stablehlo.convolution(%dhc3, %dpool1r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x16x16xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x16x16xf32>
    %dac2 = "stablehlo.select_and_scatter"(%ac2, %dpool1, %sc) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%su: tensor<f32>, %sv: tensor<f32>):
        %ss = stablehlo.add %su, %sv : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x16x32x32xf32>, tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16x32x32xf32>
    %dhc2z = stablehlo.constant dense<0.0> : tensor<128x16x32x32xf32>
    %dhc2m = stablehlo.compare GT, %hc2, %dhc2z : (tensor<128x16x32x32xf32>, tensor<128x16x32x32xf32>) -> tensor<128x16x32x32xi1>
    %dhc2 = stablehlo.select %dhc2m, %dac2, %dhc2z : tensor<128x16x32x32xi1>, tensor<128x16x32x32xf32>
    %dac1t = stablehlo.transpose %W2, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %dac1r = stablehlo.reverse %dac1t, dims = [2, 3] : tensor<16x16x3x3xf32>
    %dac1 = stablehlo.convolution(%dhc2, %dac1r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x32x32xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x32x32xf32>
    %dhc1z = stablehlo.constant dense<0.0> : tensor<128x16x32x32xf32>
    %dhc1m = stablehlo.compare GT, %hc1, %dhc1z : (tensor<128x16x32x32xf32>, tensor<128x16x32x32xf32>) -> tensor<128x16x32x32xi1>
    %dhc1 = stablehlo.select %dhc1m, %dac1, %dhc1z : tensor<128x16x32x32xi1>, tensor<128x16x32x32xf32>
    // ── param grads: dense W/b; conv dW (transpose trick), db (reduce) ──
    %dWb = stablehlo.dot_general %aa, %dy, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<128x10xf32>) -> tensor<64x10xf32>
    %dbb = stablehlo.reduce(%dy init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<128x10xf32>, tensor<f32>) -> tensor<10xf32>
    %dWa = stablehlo.dot_general %a9, %dya, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<64x64xf32>
    %dba = stablehlo.reduce(%dya init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<128x64xf32>, tensor<f32>) -> tensor<64xf32>
    %dW9 = stablehlo.dot_general %flat, %dy9, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x128xf32>, tensor<128x64xf32>) -> tensor<128x64xf32>
    %db9 = stablehlo.reduce(%dy9 init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<128x64xf32>, tensor<f32>) -> tensor<64xf32>
    %dW8xt = stablehlo.transpose %ac7, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %dW8dt = stablehlo.transpose %dhc8, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %dW8raw = stablehlo.convolution(%dW8xt, %dW8dt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xf32>, tensor<32x128x4x4xf32>) -> tensor<32x32x3x3xf32>
    %dW8 = stablehlo.transpose %dW8raw, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %db8 = stablehlo.reduce(%dhc8 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %dW7xt = stablehlo.transpose %pool3, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %dW7dt = stablehlo.transpose %dhc7, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %dW7raw = stablehlo.convolution(%dW7xt, %dW7dt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xf32>, tensor<32x128x4x4xf32>) -> tensor<32x32x3x3xf32>
    %dW7 = stablehlo.transpose %dW7raw, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %db7 = stablehlo.reduce(%dhc7 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %dW6xt = stablehlo.transpose %ac5, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %dW6dt = stablehlo.transpose %dhc6, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %dW6raw = stablehlo.convolution(%dW6xt, %dW6dt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x8x8xf32>, tensor<32x128x8x8xf32>) -> tensor<32x32x3x3xf32>
    %dW6 = stablehlo.transpose %dW6raw, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %db6 = stablehlo.reduce(%dhc6 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %dW5xt = stablehlo.transpose %pool2, dims = [1, 0, 2, 3] : (tensor<128x16x8x8xf32>) -> tensor<16x128x8x8xf32>
    %dW5dt = stablehlo.transpose %dhc5, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %dW5raw = stablehlo.convolution(%dW5xt, %dW5dt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x8x8xf32>, tensor<32x128x8x8xf32>) -> tensor<16x32x3x3xf32>
    %dW5 = stablehlo.transpose %dW5raw, dims = [1, 0, 2, 3] : (tensor<16x32x3x3xf32>) -> tensor<32x16x3x3xf32>
    %db5 = stablehlo.reduce(%dhc5 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %dW4xt = stablehlo.transpose %ac3, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %dW4dt = stablehlo.transpose %dhc4, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %dW4raw = stablehlo.convolution(%dW4xt, %dW4dt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xf32>, tensor<16x128x16x16xf32>) -> tensor<16x16x3x3xf32>
    %dW4 = stablehlo.transpose %dW4raw, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %db4 = stablehlo.reduce(%dhc4 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %dW3xt = stablehlo.transpose %pool1, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %dW3dt = stablehlo.transpose %dhc3, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %dW3raw = stablehlo.convolution(%dW3xt, %dW3dt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xf32>, tensor<16x128x16x16xf32>) -> tensor<16x16x3x3xf32>
    %dW3 = stablehlo.transpose %dW3raw, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %db3 = stablehlo.reduce(%dhc3 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %dW2xt = stablehlo.transpose %ac1, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %dW2dt = stablehlo.transpose %dhc2, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %dW2raw = stablehlo.convolution(%dW2xt, %dW2dt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x32x32xf32>, tensor<16x128x32x32xf32>) -> tensor<16x16x3x3xf32>
    %dW2 = stablehlo.transpose %dW2raw, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %db2 = stablehlo.reduce(%dhc2 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %dW1xt = stablehlo.transpose %xr, dims = [1, 0, 2, 3] : (tensor<128x3x32x32xf32>) -> tensor<3x128x32x32xf32>
    %dW1dt = stablehlo.transpose %dhc1, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %dW1raw = stablehlo.convolution(%dW1xt, %dW1dt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x128x32x32xf32>, tensor<16x128x32x32xf32>) -> tensor<3x16x3x3xf32>
    %dW1 = stablehlo.transpose %dW1raw, dims = [1, 0, 2, 3] : (tensor<3x16x3x3xf32>) -> tensor<16x3x3x3xf32>
    %db1 = stablehlo.reduce(%dhc1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %b1 = stablehlo.constant dense<0.9> : tensor<f32>
    %ob1 = stablehlo.constant dense<0.1> : tensor<f32>
    %b2 = stablehlo.constant dense<0.999> : tensor<f32>
    %ob2 = stablehlo.constant dense<0.001> : tensor<f32>
    %eps = stablehlo.constant dense<1.0e-8> : tensor<f32>
    %wd = stablehlo.constant dense<0.0001> : tensor<f32>
    %adb1W1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %adob1W1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %admsW1 = stablehlo.multiply %adb1W1, %W1m : tensor<16x3x3x3xf32>
    %admgW1 = stablehlo.multiply %adob1W1, %dW1 : tensor<16x3x3x3xf32>
    %admnW1 = stablehlo.add %admsW1, %admgW1 : tensor<16x3x3x3xf32>
    %adb2W1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %adob2W1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %advsW1 = stablehlo.multiply %adb2W1, %W1v : tensor<16x3x3x3xf32>
    %adg2W1 = stablehlo.multiply %dW1, %dW1 : tensor<16x3x3x3xf32>
    %advgW1 = stablehlo.multiply %adob2W1, %adg2W1 : tensor<16x3x3x3xf32>
    %advnW1 = stablehlo.add %advsW1, %advgW1 : tensor<16x3x3x3xf32>
    %adbc1W1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %adbc2W1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %admhW1 = stablehlo.divide %admnW1, %adbc1W1 : tensor<16x3x3x3xf32>
    %advhW1 = stablehlo.divide %advnW1, %adbc2W1 : tensor<16x3x3x3xf32>
    %adlrW1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %adepsW1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %adsqW1 = stablehlo.sqrt %advhW1 : tensor<16x3x3x3xf32>
    %addenW1 = stablehlo.add %adsqW1, %adepsW1 : tensor<16x3x3x3xf32>
    %adratW1 = stablehlo.divide %admhW1, %addenW1 : tensor<16x3x3x3xf32>
    %adstW1 = stablehlo.multiply %adlrW1, %adratW1 : tensor<16x3x3x3xf32>
    %adsubW1 = stablehlo.subtract %W1, %adstW1 : tensor<16x3x3x3xf32>
    %adwdW1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %adwdlrW1 = stablehlo.multiply %adwdW1, %adlrW1 : tensor<16x3x3x3xf32>
    %adwdpW1 = stablehlo.multiply %adwdlrW1, %W1 : tensor<16x3x3x3xf32>
    %adnewW1 = stablehlo.subtract %adsubW1, %adwdpW1 : tensor<16x3x3x3xf32>
    %adb1cb1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adob1cb1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %admscb1 = stablehlo.multiply %adb1cb1, %cb1m : tensor<16xf32>
    %admgcb1 = stablehlo.multiply %adob1cb1, %db1 : tensor<16xf32>
    %admncb1 = stablehlo.add %admscb1, %admgcb1 : tensor<16xf32>
    %adb2cb1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adob2cb1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %advscb1 = stablehlo.multiply %adb2cb1, %cb1v : tensor<16xf32>
    %adg2cb1 = stablehlo.multiply %db1, %db1 : tensor<16xf32>
    %advgcb1 = stablehlo.multiply %adob2cb1, %adg2cb1 : tensor<16xf32>
    %advncb1 = stablehlo.add %advscb1, %advgcb1 : tensor<16xf32>
    %adbc1cb1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adbc2cb1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %admhcb1 = stablehlo.divide %admncb1, %adbc1cb1 : tensor<16xf32>
    %advhcb1 = stablehlo.divide %advncb1, %adbc2cb1 : tensor<16xf32>
    %adlrcb1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adepscb1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adsqcb1 = stablehlo.sqrt %advhcb1 : tensor<16xf32>
    %addencb1 = stablehlo.add %adsqcb1, %adepscb1 : tensor<16xf32>
    %adratcb1 = stablehlo.divide %admhcb1, %addencb1 : tensor<16xf32>
    %adstcb1 = stablehlo.multiply %adlrcb1, %adratcb1 : tensor<16xf32>
    %adsubcb1 = stablehlo.subtract %cb1, %adstcb1 : tensor<16xf32>
    %adwdcb1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adwdlrcb1 = stablehlo.multiply %adwdcb1, %adlrcb1 : tensor<16xf32>
    %adwdpcb1 = stablehlo.multiply %adwdlrcb1, %cb1 : tensor<16xf32>
    %adnewcb1 = stablehlo.subtract %adsubcb1, %adwdpcb1 : tensor<16xf32>
    %adb1W2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %adob1W2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %admsW2 = stablehlo.multiply %adb1W2, %W2m : tensor<16x16x3x3xf32>
    %admgW2 = stablehlo.multiply %adob1W2, %dW2 : tensor<16x16x3x3xf32>
    %admnW2 = stablehlo.add %admsW2, %admgW2 : tensor<16x16x3x3xf32>
    %adb2W2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %adob2W2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %advsW2 = stablehlo.multiply %adb2W2, %W2v : tensor<16x16x3x3xf32>
    %adg2W2 = stablehlo.multiply %dW2, %dW2 : tensor<16x16x3x3xf32>
    %advgW2 = stablehlo.multiply %adob2W2, %adg2W2 : tensor<16x16x3x3xf32>
    %advnW2 = stablehlo.add %advsW2, %advgW2 : tensor<16x16x3x3xf32>
    %adbc1W2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %adbc2W2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %admhW2 = stablehlo.divide %admnW2, %adbc1W2 : tensor<16x16x3x3xf32>
    %advhW2 = stablehlo.divide %advnW2, %adbc2W2 : tensor<16x16x3x3xf32>
    %adlrW2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %adepsW2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %adsqW2 = stablehlo.sqrt %advhW2 : tensor<16x16x3x3xf32>
    %addenW2 = stablehlo.add %adsqW2, %adepsW2 : tensor<16x16x3x3xf32>
    %adratW2 = stablehlo.divide %admhW2, %addenW2 : tensor<16x16x3x3xf32>
    %adstW2 = stablehlo.multiply %adlrW2, %adratW2 : tensor<16x16x3x3xf32>
    %adsubW2 = stablehlo.subtract %W2, %adstW2 : tensor<16x16x3x3xf32>
    %adwdW2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %adwdlrW2 = stablehlo.multiply %adwdW2, %adlrW2 : tensor<16x16x3x3xf32>
    %adwdpW2 = stablehlo.multiply %adwdlrW2, %W2 : tensor<16x16x3x3xf32>
    %adnewW2 = stablehlo.subtract %adsubW2, %adwdpW2 : tensor<16x16x3x3xf32>
    %adb1cb2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adob1cb2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %admscb2 = stablehlo.multiply %adb1cb2, %cb2m : tensor<16xf32>
    %admgcb2 = stablehlo.multiply %adob1cb2, %db2 : tensor<16xf32>
    %admncb2 = stablehlo.add %admscb2, %admgcb2 : tensor<16xf32>
    %adb2cb2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adob2cb2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %advscb2 = stablehlo.multiply %adb2cb2, %cb2v : tensor<16xf32>
    %adg2cb2 = stablehlo.multiply %db2, %db2 : tensor<16xf32>
    %advgcb2 = stablehlo.multiply %adob2cb2, %adg2cb2 : tensor<16xf32>
    %advncb2 = stablehlo.add %advscb2, %advgcb2 : tensor<16xf32>
    %adbc1cb2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adbc2cb2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %admhcb2 = stablehlo.divide %admncb2, %adbc1cb2 : tensor<16xf32>
    %advhcb2 = stablehlo.divide %advncb2, %adbc2cb2 : tensor<16xf32>
    %adlrcb2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adepscb2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adsqcb2 = stablehlo.sqrt %advhcb2 : tensor<16xf32>
    %addencb2 = stablehlo.add %adsqcb2, %adepscb2 : tensor<16xf32>
    %adratcb2 = stablehlo.divide %admhcb2, %addencb2 : tensor<16xf32>
    %adstcb2 = stablehlo.multiply %adlrcb2, %adratcb2 : tensor<16xf32>
    %adsubcb2 = stablehlo.subtract %cb2, %adstcb2 : tensor<16xf32>
    %adwdcb2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adwdlrcb2 = stablehlo.multiply %adwdcb2, %adlrcb2 : tensor<16xf32>
    %adwdpcb2 = stablehlo.multiply %adwdlrcb2, %cb2 : tensor<16xf32>
    %adnewcb2 = stablehlo.subtract %adsubcb2, %adwdpcb2 : tensor<16xf32>
    %adb1W3 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %adob1W3 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %admsW3 = stablehlo.multiply %adb1W3, %W3m : tensor<16x16x3x3xf32>
    %admgW3 = stablehlo.multiply %adob1W3, %dW3 : tensor<16x16x3x3xf32>
    %admnW3 = stablehlo.add %admsW3, %admgW3 : tensor<16x16x3x3xf32>
    %adb2W3 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %adob2W3 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %advsW3 = stablehlo.multiply %adb2W3, %W3v : tensor<16x16x3x3xf32>
    %adg2W3 = stablehlo.multiply %dW3, %dW3 : tensor<16x16x3x3xf32>
    %advgW3 = stablehlo.multiply %adob2W3, %adg2W3 : tensor<16x16x3x3xf32>
    %advnW3 = stablehlo.add %advsW3, %advgW3 : tensor<16x16x3x3xf32>
    %adbc1W3 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %adbc2W3 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %admhW3 = stablehlo.divide %admnW3, %adbc1W3 : tensor<16x16x3x3xf32>
    %advhW3 = stablehlo.divide %advnW3, %adbc2W3 : tensor<16x16x3x3xf32>
    %adlrW3 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %adepsW3 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %adsqW3 = stablehlo.sqrt %advhW3 : tensor<16x16x3x3xf32>
    %addenW3 = stablehlo.add %adsqW3, %adepsW3 : tensor<16x16x3x3xf32>
    %adratW3 = stablehlo.divide %admhW3, %addenW3 : tensor<16x16x3x3xf32>
    %adstW3 = stablehlo.multiply %adlrW3, %adratW3 : tensor<16x16x3x3xf32>
    %adsubW3 = stablehlo.subtract %W3, %adstW3 : tensor<16x16x3x3xf32>
    %adwdW3 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %adwdlrW3 = stablehlo.multiply %adwdW3, %adlrW3 : tensor<16x16x3x3xf32>
    %adwdpW3 = stablehlo.multiply %adwdlrW3, %W3 : tensor<16x16x3x3xf32>
    %adnewW3 = stablehlo.subtract %adsubW3, %adwdpW3 : tensor<16x16x3x3xf32>
    %adb1cb3 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adob1cb3 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %admscb3 = stablehlo.multiply %adb1cb3, %cb3m : tensor<16xf32>
    %admgcb3 = stablehlo.multiply %adob1cb3, %db3 : tensor<16xf32>
    %admncb3 = stablehlo.add %admscb3, %admgcb3 : tensor<16xf32>
    %adb2cb3 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adob2cb3 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %advscb3 = stablehlo.multiply %adb2cb3, %cb3v : tensor<16xf32>
    %adg2cb3 = stablehlo.multiply %db3, %db3 : tensor<16xf32>
    %advgcb3 = stablehlo.multiply %adob2cb3, %adg2cb3 : tensor<16xf32>
    %advncb3 = stablehlo.add %advscb3, %advgcb3 : tensor<16xf32>
    %adbc1cb3 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adbc2cb3 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %admhcb3 = stablehlo.divide %admncb3, %adbc1cb3 : tensor<16xf32>
    %advhcb3 = stablehlo.divide %advncb3, %adbc2cb3 : tensor<16xf32>
    %adlrcb3 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adepscb3 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adsqcb3 = stablehlo.sqrt %advhcb3 : tensor<16xf32>
    %addencb3 = stablehlo.add %adsqcb3, %adepscb3 : tensor<16xf32>
    %adratcb3 = stablehlo.divide %admhcb3, %addencb3 : tensor<16xf32>
    %adstcb3 = stablehlo.multiply %adlrcb3, %adratcb3 : tensor<16xf32>
    %adsubcb3 = stablehlo.subtract %cb3, %adstcb3 : tensor<16xf32>
    %adwdcb3 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adwdlrcb3 = stablehlo.multiply %adwdcb3, %adlrcb3 : tensor<16xf32>
    %adwdpcb3 = stablehlo.multiply %adwdlrcb3, %cb3 : tensor<16xf32>
    %adnewcb3 = stablehlo.subtract %adsubcb3, %adwdpcb3 : tensor<16xf32>
    %adb1W4 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %adob1W4 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %admsW4 = stablehlo.multiply %adb1W4, %W4m : tensor<16x16x3x3xf32>
    %admgW4 = stablehlo.multiply %adob1W4, %dW4 : tensor<16x16x3x3xf32>
    %admnW4 = stablehlo.add %admsW4, %admgW4 : tensor<16x16x3x3xf32>
    %adb2W4 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %adob2W4 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %advsW4 = stablehlo.multiply %adb2W4, %W4v : tensor<16x16x3x3xf32>
    %adg2W4 = stablehlo.multiply %dW4, %dW4 : tensor<16x16x3x3xf32>
    %advgW4 = stablehlo.multiply %adob2W4, %adg2W4 : tensor<16x16x3x3xf32>
    %advnW4 = stablehlo.add %advsW4, %advgW4 : tensor<16x16x3x3xf32>
    %adbc1W4 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %adbc2W4 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %admhW4 = stablehlo.divide %admnW4, %adbc1W4 : tensor<16x16x3x3xf32>
    %advhW4 = stablehlo.divide %advnW4, %adbc2W4 : tensor<16x16x3x3xf32>
    %adlrW4 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %adepsW4 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %adsqW4 = stablehlo.sqrt %advhW4 : tensor<16x16x3x3xf32>
    %addenW4 = stablehlo.add %adsqW4, %adepsW4 : tensor<16x16x3x3xf32>
    %adratW4 = stablehlo.divide %admhW4, %addenW4 : tensor<16x16x3x3xf32>
    %adstW4 = stablehlo.multiply %adlrW4, %adratW4 : tensor<16x16x3x3xf32>
    %adsubW4 = stablehlo.subtract %W4, %adstW4 : tensor<16x16x3x3xf32>
    %adwdW4 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %adwdlrW4 = stablehlo.multiply %adwdW4, %adlrW4 : tensor<16x16x3x3xf32>
    %adwdpW4 = stablehlo.multiply %adwdlrW4, %W4 : tensor<16x16x3x3xf32>
    %adnewW4 = stablehlo.subtract %adsubW4, %adwdpW4 : tensor<16x16x3x3xf32>
    %adb1cb4 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adob1cb4 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %admscb4 = stablehlo.multiply %adb1cb4, %cb4m : tensor<16xf32>
    %admgcb4 = stablehlo.multiply %adob1cb4, %db4 : tensor<16xf32>
    %admncb4 = stablehlo.add %admscb4, %admgcb4 : tensor<16xf32>
    %adb2cb4 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adob2cb4 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %advscb4 = stablehlo.multiply %adb2cb4, %cb4v : tensor<16xf32>
    %adg2cb4 = stablehlo.multiply %db4, %db4 : tensor<16xf32>
    %advgcb4 = stablehlo.multiply %adob2cb4, %adg2cb4 : tensor<16xf32>
    %advncb4 = stablehlo.add %advscb4, %advgcb4 : tensor<16xf32>
    %adbc1cb4 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adbc2cb4 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %admhcb4 = stablehlo.divide %admncb4, %adbc1cb4 : tensor<16xf32>
    %advhcb4 = stablehlo.divide %advncb4, %adbc2cb4 : tensor<16xf32>
    %adlrcb4 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adepscb4 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adsqcb4 = stablehlo.sqrt %advhcb4 : tensor<16xf32>
    %addencb4 = stablehlo.add %adsqcb4, %adepscb4 : tensor<16xf32>
    %adratcb4 = stablehlo.divide %admhcb4, %addencb4 : tensor<16xf32>
    %adstcb4 = stablehlo.multiply %adlrcb4, %adratcb4 : tensor<16xf32>
    %adsubcb4 = stablehlo.subtract %cb4, %adstcb4 : tensor<16xf32>
    %adwdcb4 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adwdlrcb4 = stablehlo.multiply %adwdcb4, %adlrcb4 : tensor<16xf32>
    %adwdpcb4 = stablehlo.multiply %adwdlrcb4, %cb4 : tensor<16xf32>
    %adnewcb4 = stablehlo.subtract %adsubcb4, %adwdpcb4 : tensor<16xf32>
    %adb1W5 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %adob1W5 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %admsW5 = stablehlo.multiply %adb1W5, %W5m : tensor<32x16x3x3xf32>
    %admgW5 = stablehlo.multiply %adob1W5, %dW5 : tensor<32x16x3x3xf32>
    %admnW5 = stablehlo.add %admsW5, %admgW5 : tensor<32x16x3x3xf32>
    %adb2W5 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %adob2W5 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %advsW5 = stablehlo.multiply %adb2W5, %W5v : tensor<32x16x3x3xf32>
    %adg2W5 = stablehlo.multiply %dW5, %dW5 : tensor<32x16x3x3xf32>
    %advgW5 = stablehlo.multiply %adob2W5, %adg2W5 : tensor<32x16x3x3xf32>
    %advnW5 = stablehlo.add %advsW5, %advgW5 : tensor<32x16x3x3xf32>
    %adbc1W5 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %adbc2W5 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %admhW5 = stablehlo.divide %admnW5, %adbc1W5 : tensor<32x16x3x3xf32>
    %advhW5 = stablehlo.divide %advnW5, %adbc2W5 : tensor<32x16x3x3xf32>
    %adlrW5 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %adepsW5 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %adsqW5 = stablehlo.sqrt %advhW5 : tensor<32x16x3x3xf32>
    %addenW5 = stablehlo.add %adsqW5, %adepsW5 : tensor<32x16x3x3xf32>
    %adratW5 = stablehlo.divide %admhW5, %addenW5 : tensor<32x16x3x3xf32>
    %adstW5 = stablehlo.multiply %adlrW5, %adratW5 : tensor<32x16x3x3xf32>
    %adsubW5 = stablehlo.subtract %W5, %adstW5 : tensor<32x16x3x3xf32>
    %adwdW5 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %adwdlrW5 = stablehlo.multiply %adwdW5, %adlrW5 : tensor<32x16x3x3xf32>
    %adwdpW5 = stablehlo.multiply %adwdlrW5, %W5 : tensor<32x16x3x3xf32>
    %adnewW5 = stablehlo.subtract %adsubW5, %adwdpW5 : tensor<32x16x3x3xf32>
    %adb1cb5 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adob1cb5 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %admscb5 = stablehlo.multiply %adb1cb5, %cb5m : tensor<32xf32>
    %admgcb5 = stablehlo.multiply %adob1cb5, %db5 : tensor<32xf32>
    %admncb5 = stablehlo.add %admscb5, %admgcb5 : tensor<32xf32>
    %adb2cb5 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adob2cb5 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %advscb5 = stablehlo.multiply %adb2cb5, %cb5v : tensor<32xf32>
    %adg2cb5 = stablehlo.multiply %db5, %db5 : tensor<32xf32>
    %advgcb5 = stablehlo.multiply %adob2cb5, %adg2cb5 : tensor<32xf32>
    %advncb5 = stablehlo.add %advscb5, %advgcb5 : tensor<32xf32>
    %adbc1cb5 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adbc2cb5 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %admhcb5 = stablehlo.divide %admncb5, %adbc1cb5 : tensor<32xf32>
    %advhcb5 = stablehlo.divide %advncb5, %adbc2cb5 : tensor<32xf32>
    %adlrcb5 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adepscb5 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adsqcb5 = stablehlo.sqrt %advhcb5 : tensor<32xf32>
    %addencb5 = stablehlo.add %adsqcb5, %adepscb5 : tensor<32xf32>
    %adratcb5 = stablehlo.divide %admhcb5, %addencb5 : tensor<32xf32>
    %adstcb5 = stablehlo.multiply %adlrcb5, %adratcb5 : tensor<32xf32>
    %adsubcb5 = stablehlo.subtract %cb5, %adstcb5 : tensor<32xf32>
    %adwdcb5 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adwdlrcb5 = stablehlo.multiply %adwdcb5, %adlrcb5 : tensor<32xf32>
    %adwdpcb5 = stablehlo.multiply %adwdlrcb5, %cb5 : tensor<32xf32>
    %adnewcb5 = stablehlo.subtract %adsubcb5, %adwdpcb5 : tensor<32xf32>
    %adb1W6 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %adob1W6 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %admsW6 = stablehlo.multiply %adb1W6, %W6m : tensor<32x32x3x3xf32>
    %admgW6 = stablehlo.multiply %adob1W6, %dW6 : tensor<32x32x3x3xf32>
    %admnW6 = stablehlo.add %admsW6, %admgW6 : tensor<32x32x3x3xf32>
    %adb2W6 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %adob2W6 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %advsW6 = stablehlo.multiply %adb2W6, %W6v : tensor<32x32x3x3xf32>
    %adg2W6 = stablehlo.multiply %dW6, %dW6 : tensor<32x32x3x3xf32>
    %advgW6 = stablehlo.multiply %adob2W6, %adg2W6 : tensor<32x32x3x3xf32>
    %advnW6 = stablehlo.add %advsW6, %advgW6 : tensor<32x32x3x3xf32>
    %adbc1W6 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %adbc2W6 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %admhW6 = stablehlo.divide %admnW6, %adbc1W6 : tensor<32x32x3x3xf32>
    %advhW6 = stablehlo.divide %advnW6, %adbc2W6 : tensor<32x32x3x3xf32>
    %adlrW6 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %adepsW6 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %adsqW6 = stablehlo.sqrt %advhW6 : tensor<32x32x3x3xf32>
    %addenW6 = stablehlo.add %adsqW6, %adepsW6 : tensor<32x32x3x3xf32>
    %adratW6 = stablehlo.divide %admhW6, %addenW6 : tensor<32x32x3x3xf32>
    %adstW6 = stablehlo.multiply %adlrW6, %adratW6 : tensor<32x32x3x3xf32>
    %adsubW6 = stablehlo.subtract %W6, %adstW6 : tensor<32x32x3x3xf32>
    %adwdW6 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %adwdlrW6 = stablehlo.multiply %adwdW6, %adlrW6 : tensor<32x32x3x3xf32>
    %adwdpW6 = stablehlo.multiply %adwdlrW6, %W6 : tensor<32x32x3x3xf32>
    %adnewW6 = stablehlo.subtract %adsubW6, %adwdpW6 : tensor<32x32x3x3xf32>
    %adb1cb6 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adob1cb6 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %admscb6 = stablehlo.multiply %adb1cb6, %cb6m : tensor<32xf32>
    %admgcb6 = stablehlo.multiply %adob1cb6, %db6 : tensor<32xf32>
    %admncb6 = stablehlo.add %admscb6, %admgcb6 : tensor<32xf32>
    %adb2cb6 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adob2cb6 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %advscb6 = stablehlo.multiply %adb2cb6, %cb6v : tensor<32xf32>
    %adg2cb6 = stablehlo.multiply %db6, %db6 : tensor<32xf32>
    %advgcb6 = stablehlo.multiply %adob2cb6, %adg2cb6 : tensor<32xf32>
    %advncb6 = stablehlo.add %advscb6, %advgcb6 : tensor<32xf32>
    %adbc1cb6 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adbc2cb6 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %admhcb6 = stablehlo.divide %admncb6, %adbc1cb6 : tensor<32xf32>
    %advhcb6 = stablehlo.divide %advncb6, %adbc2cb6 : tensor<32xf32>
    %adlrcb6 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adepscb6 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adsqcb6 = stablehlo.sqrt %advhcb6 : tensor<32xf32>
    %addencb6 = stablehlo.add %adsqcb6, %adepscb6 : tensor<32xf32>
    %adratcb6 = stablehlo.divide %admhcb6, %addencb6 : tensor<32xf32>
    %adstcb6 = stablehlo.multiply %adlrcb6, %adratcb6 : tensor<32xf32>
    %adsubcb6 = stablehlo.subtract %cb6, %adstcb6 : tensor<32xf32>
    %adwdcb6 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adwdlrcb6 = stablehlo.multiply %adwdcb6, %adlrcb6 : tensor<32xf32>
    %adwdpcb6 = stablehlo.multiply %adwdlrcb6, %cb6 : tensor<32xf32>
    %adnewcb6 = stablehlo.subtract %adsubcb6, %adwdpcb6 : tensor<32xf32>
    %adb1W7 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %adob1W7 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %admsW7 = stablehlo.multiply %adb1W7, %W7m : tensor<32x32x3x3xf32>
    %admgW7 = stablehlo.multiply %adob1W7, %dW7 : tensor<32x32x3x3xf32>
    %admnW7 = stablehlo.add %admsW7, %admgW7 : tensor<32x32x3x3xf32>
    %adb2W7 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %adob2W7 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %advsW7 = stablehlo.multiply %adb2W7, %W7v : tensor<32x32x3x3xf32>
    %adg2W7 = stablehlo.multiply %dW7, %dW7 : tensor<32x32x3x3xf32>
    %advgW7 = stablehlo.multiply %adob2W7, %adg2W7 : tensor<32x32x3x3xf32>
    %advnW7 = stablehlo.add %advsW7, %advgW7 : tensor<32x32x3x3xf32>
    %adbc1W7 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %adbc2W7 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %admhW7 = stablehlo.divide %admnW7, %adbc1W7 : tensor<32x32x3x3xf32>
    %advhW7 = stablehlo.divide %advnW7, %adbc2W7 : tensor<32x32x3x3xf32>
    %adlrW7 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %adepsW7 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %adsqW7 = stablehlo.sqrt %advhW7 : tensor<32x32x3x3xf32>
    %addenW7 = stablehlo.add %adsqW7, %adepsW7 : tensor<32x32x3x3xf32>
    %adratW7 = stablehlo.divide %admhW7, %addenW7 : tensor<32x32x3x3xf32>
    %adstW7 = stablehlo.multiply %adlrW7, %adratW7 : tensor<32x32x3x3xf32>
    %adsubW7 = stablehlo.subtract %W7, %adstW7 : tensor<32x32x3x3xf32>
    %adwdW7 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %adwdlrW7 = stablehlo.multiply %adwdW7, %adlrW7 : tensor<32x32x3x3xf32>
    %adwdpW7 = stablehlo.multiply %adwdlrW7, %W7 : tensor<32x32x3x3xf32>
    %adnewW7 = stablehlo.subtract %adsubW7, %adwdpW7 : tensor<32x32x3x3xf32>
    %adb1cb7 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adob1cb7 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %admscb7 = stablehlo.multiply %adb1cb7, %cb7m : tensor<32xf32>
    %admgcb7 = stablehlo.multiply %adob1cb7, %db7 : tensor<32xf32>
    %admncb7 = stablehlo.add %admscb7, %admgcb7 : tensor<32xf32>
    %adb2cb7 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adob2cb7 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %advscb7 = stablehlo.multiply %adb2cb7, %cb7v : tensor<32xf32>
    %adg2cb7 = stablehlo.multiply %db7, %db7 : tensor<32xf32>
    %advgcb7 = stablehlo.multiply %adob2cb7, %adg2cb7 : tensor<32xf32>
    %advncb7 = stablehlo.add %advscb7, %advgcb7 : tensor<32xf32>
    %adbc1cb7 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adbc2cb7 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %admhcb7 = stablehlo.divide %admncb7, %adbc1cb7 : tensor<32xf32>
    %advhcb7 = stablehlo.divide %advncb7, %adbc2cb7 : tensor<32xf32>
    %adlrcb7 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adepscb7 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adsqcb7 = stablehlo.sqrt %advhcb7 : tensor<32xf32>
    %addencb7 = stablehlo.add %adsqcb7, %adepscb7 : tensor<32xf32>
    %adratcb7 = stablehlo.divide %admhcb7, %addencb7 : tensor<32xf32>
    %adstcb7 = stablehlo.multiply %adlrcb7, %adratcb7 : tensor<32xf32>
    %adsubcb7 = stablehlo.subtract %cb7, %adstcb7 : tensor<32xf32>
    %adwdcb7 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adwdlrcb7 = stablehlo.multiply %adwdcb7, %adlrcb7 : tensor<32xf32>
    %adwdpcb7 = stablehlo.multiply %adwdlrcb7, %cb7 : tensor<32xf32>
    %adnewcb7 = stablehlo.subtract %adsubcb7, %adwdpcb7 : tensor<32xf32>
    %adb1W8 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %adob1W8 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %admsW8 = stablehlo.multiply %adb1W8, %W8m : tensor<32x32x3x3xf32>
    %admgW8 = stablehlo.multiply %adob1W8, %dW8 : tensor<32x32x3x3xf32>
    %admnW8 = stablehlo.add %admsW8, %admgW8 : tensor<32x32x3x3xf32>
    %adb2W8 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %adob2W8 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %advsW8 = stablehlo.multiply %adb2W8, %W8v : tensor<32x32x3x3xf32>
    %adg2W8 = stablehlo.multiply %dW8, %dW8 : tensor<32x32x3x3xf32>
    %advgW8 = stablehlo.multiply %adob2W8, %adg2W8 : tensor<32x32x3x3xf32>
    %advnW8 = stablehlo.add %advsW8, %advgW8 : tensor<32x32x3x3xf32>
    %adbc1W8 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %adbc2W8 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %admhW8 = stablehlo.divide %admnW8, %adbc1W8 : tensor<32x32x3x3xf32>
    %advhW8 = stablehlo.divide %advnW8, %adbc2W8 : tensor<32x32x3x3xf32>
    %adlrW8 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %adepsW8 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %adsqW8 = stablehlo.sqrt %advhW8 : tensor<32x32x3x3xf32>
    %addenW8 = stablehlo.add %adsqW8, %adepsW8 : tensor<32x32x3x3xf32>
    %adratW8 = stablehlo.divide %admhW8, %addenW8 : tensor<32x32x3x3xf32>
    %adstW8 = stablehlo.multiply %adlrW8, %adratW8 : tensor<32x32x3x3xf32>
    %adsubW8 = stablehlo.subtract %W8, %adstW8 : tensor<32x32x3x3xf32>
    %adwdW8 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %adwdlrW8 = stablehlo.multiply %adwdW8, %adlrW8 : tensor<32x32x3x3xf32>
    %adwdpW8 = stablehlo.multiply %adwdlrW8, %W8 : tensor<32x32x3x3xf32>
    %adnewW8 = stablehlo.subtract %adsubW8, %adwdpW8 : tensor<32x32x3x3xf32>
    %adb1cb8 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adob1cb8 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %admscb8 = stablehlo.multiply %adb1cb8, %cb8m : tensor<32xf32>
    %admgcb8 = stablehlo.multiply %adob1cb8, %db8 : tensor<32xf32>
    %admncb8 = stablehlo.add %admscb8, %admgcb8 : tensor<32xf32>
    %adb2cb8 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adob2cb8 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %advscb8 = stablehlo.multiply %adb2cb8, %cb8v : tensor<32xf32>
    %adg2cb8 = stablehlo.multiply %db8, %db8 : tensor<32xf32>
    %advgcb8 = stablehlo.multiply %adob2cb8, %adg2cb8 : tensor<32xf32>
    %advncb8 = stablehlo.add %advscb8, %advgcb8 : tensor<32xf32>
    %adbc1cb8 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adbc2cb8 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %admhcb8 = stablehlo.divide %admncb8, %adbc1cb8 : tensor<32xf32>
    %advhcb8 = stablehlo.divide %advncb8, %adbc2cb8 : tensor<32xf32>
    %adlrcb8 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adepscb8 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adsqcb8 = stablehlo.sqrt %advhcb8 : tensor<32xf32>
    %addencb8 = stablehlo.add %adsqcb8, %adepscb8 : tensor<32xf32>
    %adratcb8 = stablehlo.divide %admhcb8, %addencb8 : tensor<32xf32>
    %adstcb8 = stablehlo.multiply %adlrcb8, %adratcb8 : tensor<32xf32>
    %adsubcb8 = stablehlo.subtract %cb8, %adstcb8 : tensor<32xf32>
    %adwdcb8 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adwdlrcb8 = stablehlo.multiply %adwdcb8, %adlrcb8 : tensor<32xf32>
    %adwdpcb8 = stablehlo.multiply %adwdlrcb8, %cb8 : tensor<32xf32>
    %adnewcb8 = stablehlo.subtract %adsubcb8, %adwdpcb8 : tensor<32xf32>
    %adb1W9 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %adob1W9 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %admsW9 = stablehlo.multiply %adb1W9, %W9m : tensor<128x64xf32>
    %admgW9 = stablehlo.multiply %adob1W9, %dW9 : tensor<128x64xf32>
    %admnW9 = stablehlo.add %admsW9, %admgW9 : tensor<128x64xf32>
    %adb2W9 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %adob2W9 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %advsW9 = stablehlo.multiply %adb2W9, %W9v : tensor<128x64xf32>
    %adg2W9 = stablehlo.multiply %dW9, %dW9 : tensor<128x64xf32>
    %advgW9 = stablehlo.multiply %adob2W9, %adg2W9 : tensor<128x64xf32>
    %advnW9 = stablehlo.add %advsW9, %advgW9 : tensor<128x64xf32>
    %adbc1W9 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %adbc2W9 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %admhW9 = stablehlo.divide %admnW9, %adbc1W9 : tensor<128x64xf32>
    %advhW9 = stablehlo.divide %advnW9, %adbc2W9 : tensor<128x64xf32>
    %adlrW9 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %adepsW9 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %adsqW9 = stablehlo.sqrt %advhW9 : tensor<128x64xf32>
    %addenW9 = stablehlo.add %adsqW9, %adepsW9 : tensor<128x64xf32>
    %adratW9 = stablehlo.divide %admhW9, %addenW9 : tensor<128x64xf32>
    %adstW9 = stablehlo.multiply %adlrW9, %adratW9 : tensor<128x64xf32>
    %adsubW9 = stablehlo.subtract %W9, %adstW9 : tensor<128x64xf32>
    %adwdW9 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %adwdlrW9 = stablehlo.multiply %adwdW9, %adlrW9 : tensor<128x64xf32>
    %adwdpW9 = stablehlo.multiply %adwdlrW9, %W9 : tensor<128x64xf32>
    %adnewW9 = stablehlo.subtract %adsubW9, %adwdpW9 : tensor<128x64xf32>
    %adb1b9 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adob1b9 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %admsb9 = stablehlo.multiply %adb1b9, %b9m : tensor<64xf32>
    %admgb9 = stablehlo.multiply %adob1b9, %db9 : tensor<64xf32>
    %admnb9 = stablehlo.add %admsb9, %admgb9 : tensor<64xf32>
    %adb2b9 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adob2b9 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %advsb9 = stablehlo.multiply %adb2b9, %b9v : tensor<64xf32>
    %adg2b9 = stablehlo.multiply %db9, %db9 : tensor<64xf32>
    %advgb9 = stablehlo.multiply %adob2b9, %adg2b9 : tensor<64xf32>
    %advnb9 = stablehlo.add %advsb9, %advgb9 : tensor<64xf32>
    %adbc1b9 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adbc2b9 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %admhb9 = stablehlo.divide %admnb9, %adbc1b9 : tensor<64xf32>
    %advhb9 = stablehlo.divide %advnb9, %adbc2b9 : tensor<64xf32>
    %adlrb9 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adepsb9 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adsqb9 = stablehlo.sqrt %advhb9 : tensor<64xf32>
    %addenb9 = stablehlo.add %adsqb9, %adepsb9 : tensor<64xf32>
    %adratb9 = stablehlo.divide %admhb9, %addenb9 : tensor<64xf32>
    %adstb9 = stablehlo.multiply %adlrb9, %adratb9 : tensor<64xf32>
    %adsubb9 = stablehlo.subtract %b9, %adstb9 : tensor<64xf32>
    %adwdb9 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adwdlrb9 = stablehlo.multiply %adwdb9, %adlrb9 : tensor<64xf32>
    %adwdpb9 = stablehlo.multiply %adwdlrb9, %b9 : tensor<64xf32>
    %adnewb9 = stablehlo.subtract %adsubb9, %adwdpb9 : tensor<64xf32>
    %adb1Wa = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %adob1Wa = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %admsWa = stablehlo.multiply %adb1Wa, %Wam : tensor<64x64xf32>
    %admgWa = stablehlo.multiply %adob1Wa, %dWa : tensor<64x64xf32>
    %admnWa = stablehlo.add %admsWa, %admgWa : tensor<64x64xf32>
    %adb2Wa = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %adob2Wa = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %advsWa = stablehlo.multiply %adb2Wa, %Wav : tensor<64x64xf32>
    %adg2Wa = stablehlo.multiply %dWa, %dWa : tensor<64x64xf32>
    %advgWa = stablehlo.multiply %adob2Wa, %adg2Wa : tensor<64x64xf32>
    %advnWa = stablehlo.add %advsWa, %advgWa : tensor<64x64xf32>
    %adbc1Wa = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %adbc2Wa = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %admhWa = stablehlo.divide %admnWa, %adbc1Wa : tensor<64x64xf32>
    %advhWa = stablehlo.divide %advnWa, %adbc2Wa : tensor<64x64xf32>
    %adlrWa = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %adepsWa = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %adsqWa = stablehlo.sqrt %advhWa : tensor<64x64xf32>
    %addenWa = stablehlo.add %adsqWa, %adepsWa : tensor<64x64xf32>
    %adratWa = stablehlo.divide %admhWa, %addenWa : tensor<64x64xf32>
    %adstWa = stablehlo.multiply %adlrWa, %adratWa : tensor<64x64xf32>
    %adsubWa = stablehlo.subtract %Wa, %adstWa : tensor<64x64xf32>
    %adwdWa = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %adwdlrWa = stablehlo.multiply %adwdWa, %adlrWa : tensor<64x64xf32>
    %adwdpWa = stablehlo.multiply %adwdlrWa, %Wa : tensor<64x64xf32>
    %adnewWa = stablehlo.subtract %adsubWa, %adwdpWa : tensor<64x64xf32>
    %adb1ba = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adob1ba = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %admsba = stablehlo.multiply %adb1ba, %bam : tensor<64xf32>
    %admgba = stablehlo.multiply %adob1ba, %dba : tensor<64xf32>
    %admnba = stablehlo.add %admsba, %admgba : tensor<64xf32>
    %adb2ba = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adob2ba = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %advsba = stablehlo.multiply %adb2ba, %bav : tensor<64xf32>
    %adg2ba = stablehlo.multiply %dba, %dba : tensor<64xf32>
    %advgba = stablehlo.multiply %adob2ba, %adg2ba : tensor<64xf32>
    %advnba = stablehlo.add %advsba, %advgba : tensor<64xf32>
    %adbc1ba = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adbc2ba = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %admhba = stablehlo.divide %admnba, %adbc1ba : tensor<64xf32>
    %advhba = stablehlo.divide %advnba, %adbc2ba : tensor<64xf32>
    %adlrba = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adepsba = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adsqba = stablehlo.sqrt %advhba : tensor<64xf32>
    %addenba = stablehlo.add %adsqba, %adepsba : tensor<64xf32>
    %adratba = stablehlo.divide %admhba, %addenba : tensor<64xf32>
    %adstba = stablehlo.multiply %adlrba, %adratba : tensor<64xf32>
    %adsubba = stablehlo.subtract %ba, %adstba : tensor<64xf32>
    %adwdba = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adwdlrba = stablehlo.multiply %adwdba, %adlrba : tensor<64xf32>
    %adwdpba = stablehlo.multiply %adwdlrba, %ba : tensor<64xf32>
    %adnewba = stablehlo.subtract %adsubba, %adwdpba : tensor<64xf32>
    %adb1Wb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %adob1Wb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %admsWb = stablehlo.multiply %adb1Wb, %Wbm : tensor<64x10xf32>
    %admgWb = stablehlo.multiply %adob1Wb, %dWb : tensor<64x10xf32>
    %admnWb = stablehlo.add %admsWb, %admgWb : tensor<64x10xf32>
    %adb2Wb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %adob2Wb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %advsWb = stablehlo.multiply %adb2Wb, %Wbv : tensor<64x10xf32>
    %adg2Wb = stablehlo.multiply %dWb, %dWb : tensor<64x10xf32>
    %advgWb = stablehlo.multiply %adob2Wb, %adg2Wb : tensor<64x10xf32>
    %advnWb = stablehlo.add %advsWb, %advgWb : tensor<64x10xf32>
    %adbc1Wb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %adbc2Wb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %admhWb = stablehlo.divide %admnWb, %adbc1Wb : tensor<64x10xf32>
    %advhWb = stablehlo.divide %advnWb, %adbc2Wb : tensor<64x10xf32>
    %adlrWb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %adepsWb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %adsqWb = stablehlo.sqrt %advhWb : tensor<64x10xf32>
    %addenWb = stablehlo.add %adsqWb, %adepsWb : tensor<64x10xf32>
    %adratWb = stablehlo.divide %admhWb, %addenWb : tensor<64x10xf32>
    %adstWb = stablehlo.multiply %adlrWb, %adratWb : tensor<64x10xf32>
    %adsubWb = stablehlo.subtract %Wb, %adstWb : tensor<64x10xf32>
    %adwdWb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %adwdlrWb = stablehlo.multiply %adwdWb, %adlrWb : tensor<64x10xf32>
    %adwdpWb = stablehlo.multiply %adwdlrWb, %Wb : tensor<64x10xf32>
    %adnewWb = stablehlo.subtract %adsubWb, %adwdpWb : tensor<64x10xf32>
    %adb1bb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %adob1bb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %admsbb = stablehlo.multiply %adb1bb, %bbm : tensor<10xf32>
    %admgbb = stablehlo.multiply %adob1bb, %dbb : tensor<10xf32>
    %admnbb = stablehlo.add %admsbb, %admgbb : tensor<10xf32>
    %adb2bb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %adob2bb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %advsbb = stablehlo.multiply %adb2bb, %bbv : tensor<10xf32>
    %adg2bb = stablehlo.multiply %dbb, %dbb : tensor<10xf32>
    %advgbb = stablehlo.multiply %adob2bb, %adg2bb : tensor<10xf32>
    %advnbb = stablehlo.add %advsbb, %advgbb : tensor<10xf32>
    %adbc1bb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %adbc2bb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %admhbb = stablehlo.divide %admnbb, %adbc1bb : tensor<10xf32>
    %advhbb = stablehlo.divide %advnbb, %adbc2bb : tensor<10xf32>
    %adlrbb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %adepsbb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %adsqbb = stablehlo.sqrt %advhbb : tensor<10xf32>
    %addenbb = stablehlo.add %adsqbb, %adepsbb : tensor<10xf32>
    %adratbb = stablehlo.divide %admhbb, %addenbb : tensor<10xf32>
    %adstbb = stablehlo.multiply %adlrbb, %adratbb : tensor<10xf32>
    %adsubbb = stablehlo.subtract %bb, %adstbb : tensor<10xf32>
    %adwdbb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %adwdlrbb = stablehlo.multiply %adwdbb, %adlrbb : tensor<10xf32>
    %adwdpbb = stablehlo.multiply %adwdlrbb, %bb : tensor<10xf32>
    %adnewbb = stablehlo.subtract %adsubbb, %adwdpbb : tensor<10xf32>
    return %adnewW1, %adnewcb1, %adnewW2, %adnewcb2, %adnewW3, %adnewcb3, %adnewW4, %adnewcb4, %adnewW5, %adnewcb5, %adnewW6, %adnewcb6, %adnewW7, %adnewcb7, %adnewW8, %adnewcb8, %adnewW9, %adnewb9, %adnewWa, %adnewba, %adnewWb, %adnewbb, %admnW1, %admncb1, %admnW2, %admncb2, %admnW3, %admncb3, %admnW4, %admncb4, %admnW5, %admncb5, %admnW6, %admncb6, %admnW7, %admncb7, %admnW8, %admncb8, %admnW9, %admnb9, %admnWa, %admnba, %admnWb, %admnbb, %advnW1, %advncb1, %advnW2, %advncb2, %advnW3, %advncb3, %advnW4, %advncb4, %advnW5, %advncb5, %advnW6, %advncb6, %advnW7, %advncb7, %advnW8, %advncb8, %advnW9, %advnb9, %advnWa, %advnba, %advnWb, %advnbb, %loss, %bc1, %bc2 : tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>
  }
}

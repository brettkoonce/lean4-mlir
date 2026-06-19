module @m {
  func.func @cifar8_mom_train_step(%x: tensor<128x3072xf32>, %W1: tensor<16x3x3x3xf32>, %cb1: tensor<16xf32>, %W2: tensor<16x16x3x3xf32>, %cb2: tensor<16xf32>, %W3: tensor<16x16x3x3xf32>, %cb3: tensor<16xf32>, %W4: tensor<16x16x3x3xf32>, %cb4: tensor<16xf32>, %W5: tensor<32x16x3x3xf32>, %cb5: tensor<32xf32>, %W6: tensor<32x32x3x3xf32>, %cb6: tensor<32xf32>, %W7: tensor<32x32x3x3xf32>, %cb7: tensor<32xf32>, %W8: tensor<32x32x3x3xf32>, %cb8: tensor<32xf32>, %W9: tensor<128x64xf32>, %b9: tensor<64xf32>, %Wa: tensor<64x64xf32>, %ba: tensor<64xf32>, %Wb: tensor<64x10xf32>, %bb: tensor<10xf32>, %W1m: tensor<16x3x3x3xf32>, %cb1m: tensor<16xf32>, %W2m: tensor<16x16x3x3xf32>, %cb2m: tensor<16xf32>, %W3m: tensor<16x16x3x3xf32>, %cb3m: tensor<16xf32>, %W4m: tensor<16x16x3x3xf32>, %cb4m: tensor<16xf32>, %W5m: tensor<32x16x3x3xf32>, %cb5m: tensor<32xf32>, %W6m: tensor<32x32x3x3xf32>, %cb6m: tensor<32xf32>, %W7m: tensor<32x32x3x3xf32>, %cb7m: tensor<32xf32>, %W8m: tensor<32x32x3x3xf32>, %cb8m: tensor<32xf32>, %W9m: tensor<128x64xf32>, %b9m: tensor<64xf32>, %Wam: tensor<64x64xf32>, %bam: tensor<64xf32>, %Wbm: tensor<64x10xf32>, %bbm: tensor<10xf32>, %W1v: tensor<16x3x3x3xf32>, %cb1v: tensor<16xf32>, %W2v: tensor<16x16x3x3xf32>, %cb2v: tensor<16xf32>, %W3v: tensor<16x16x3x3xf32>, %cb3v: tensor<16xf32>, %W4v: tensor<16x16x3x3xf32>, %cb4v: tensor<16xf32>, %W5v: tensor<32x16x3x3xf32>, %cb5v: tensor<32xf32>, %W6v: tensor<32x32x3x3xf32>, %cb6v: tensor<32xf32>, %W7v: tensor<32x32x3x3xf32>, %cb7v: tensor<32xf32>, %W8v: tensor<32x32x3x3xf32>, %cb8v: tensor<32xf32>, %W9v: tensor<128x64xf32>, %b9v: tensor<64xf32>, %Wav: tensor<64x64xf32>, %bav: tensor<64xf32>, %Wbv: tensor<64x10xf32>, %bbv: tensor<10xf32>, %lr: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>, %onehot: tensor<128x10xf32>) -> (tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>) {
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
    %mu = stablehlo.constant dense<0.9> : tensor<f32>
    %mommuW1 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %momvgW1 = stablehlo.multiply %mommuW1, %W1v : tensor<16x3x3x3xf32>
    %momvelW1 = stablehlo.add %momvgW1, %dW1 : tensor<16x3x3x3xf32>
    %momnvW1 = stablehlo.multiply %mommuW1, %momvelW1 : tensor<16x3x3x3xf32>
    %momlkW1 = stablehlo.add %momnvW1, %dW1 : tensor<16x3x3x3xf32>
    %momlrW1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %momstW1 = stablehlo.multiply %momlrW1, %momlkW1 : tensor<16x3x3x3xf32>
    %momnewW1 = stablehlo.subtract %W1, %momstW1 : tensor<16x3x3x3xf32>
    %mommucb1 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %momvgcb1 = stablehlo.multiply %mommucb1, %cb1v : tensor<16xf32>
    %momvelcb1 = stablehlo.add %momvgcb1, %db1 : tensor<16xf32>
    %momnvcb1 = stablehlo.multiply %mommucb1, %momvelcb1 : tensor<16xf32>
    %momlkcb1 = stablehlo.add %momnvcb1, %db1 : tensor<16xf32>
    %momlrcb1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %momstcb1 = stablehlo.multiply %momlrcb1, %momlkcb1 : tensor<16xf32>
    %momnewcb1 = stablehlo.subtract %cb1, %momstcb1 : tensor<16xf32>
    %mommuW2 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %momvgW2 = stablehlo.multiply %mommuW2, %W2v : tensor<16x16x3x3xf32>
    %momvelW2 = stablehlo.add %momvgW2, %dW2 : tensor<16x16x3x3xf32>
    %momnvW2 = stablehlo.multiply %mommuW2, %momvelW2 : tensor<16x16x3x3xf32>
    %momlkW2 = stablehlo.add %momnvW2, %dW2 : tensor<16x16x3x3xf32>
    %momlrW2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %momstW2 = stablehlo.multiply %momlrW2, %momlkW2 : tensor<16x16x3x3xf32>
    %momnewW2 = stablehlo.subtract %W2, %momstW2 : tensor<16x16x3x3xf32>
    %mommucb2 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %momvgcb2 = stablehlo.multiply %mommucb2, %cb2v : tensor<16xf32>
    %momvelcb2 = stablehlo.add %momvgcb2, %db2 : tensor<16xf32>
    %momnvcb2 = stablehlo.multiply %mommucb2, %momvelcb2 : tensor<16xf32>
    %momlkcb2 = stablehlo.add %momnvcb2, %db2 : tensor<16xf32>
    %momlrcb2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %momstcb2 = stablehlo.multiply %momlrcb2, %momlkcb2 : tensor<16xf32>
    %momnewcb2 = stablehlo.subtract %cb2, %momstcb2 : tensor<16xf32>
    %mommuW3 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %momvgW3 = stablehlo.multiply %mommuW3, %W3v : tensor<16x16x3x3xf32>
    %momvelW3 = stablehlo.add %momvgW3, %dW3 : tensor<16x16x3x3xf32>
    %momnvW3 = stablehlo.multiply %mommuW3, %momvelW3 : tensor<16x16x3x3xf32>
    %momlkW3 = stablehlo.add %momnvW3, %dW3 : tensor<16x16x3x3xf32>
    %momlrW3 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %momstW3 = stablehlo.multiply %momlrW3, %momlkW3 : tensor<16x16x3x3xf32>
    %momnewW3 = stablehlo.subtract %W3, %momstW3 : tensor<16x16x3x3xf32>
    %mommucb3 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %momvgcb3 = stablehlo.multiply %mommucb3, %cb3v : tensor<16xf32>
    %momvelcb3 = stablehlo.add %momvgcb3, %db3 : tensor<16xf32>
    %momnvcb3 = stablehlo.multiply %mommucb3, %momvelcb3 : tensor<16xf32>
    %momlkcb3 = stablehlo.add %momnvcb3, %db3 : tensor<16xf32>
    %momlrcb3 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %momstcb3 = stablehlo.multiply %momlrcb3, %momlkcb3 : tensor<16xf32>
    %momnewcb3 = stablehlo.subtract %cb3, %momstcb3 : tensor<16xf32>
    %mommuW4 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %momvgW4 = stablehlo.multiply %mommuW4, %W4v : tensor<16x16x3x3xf32>
    %momvelW4 = stablehlo.add %momvgW4, %dW4 : tensor<16x16x3x3xf32>
    %momnvW4 = stablehlo.multiply %mommuW4, %momvelW4 : tensor<16x16x3x3xf32>
    %momlkW4 = stablehlo.add %momnvW4, %dW4 : tensor<16x16x3x3xf32>
    %momlrW4 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %momstW4 = stablehlo.multiply %momlrW4, %momlkW4 : tensor<16x16x3x3xf32>
    %momnewW4 = stablehlo.subtract %W4, %momstW4 : tensor<16x16x3x3xf32>
    %mommucb4 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %momvgcb4 = stablehlo.multiply %mommucb4, %cb4v : tensor<16xf32>
    %momvelcb4 = stablehlo.add %momvgcb4, %db4 : tensor<16xf32>
    %momnvcb4 = stablehlo.multiply %mommucb4, %momvelcb4 : tensor<16xf32>
    %momlkcb4 = stablehlo.add %momnvcb4, %db4 : tensor<16xf32>
    %momlrcb4 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %momstcb4 = stablehlo.multiply %momlrcb4, %momlkcb4 : tensor<16xf32>
    %momnewcb4 = stablehlo.subtract %cb4, %momstcb4 : tensor<16xf32>
    %mommuW5 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %momvgW5 = stablehlo.multiply %mommuW5, %W5v : tensor<32x16x3x3xf32>
    %momvelW5 = stablehlo.add %momvgW5, %dW5 : tensor<32x16x3x3xf32>
    %momnvW5 = stablehlo.multiply %mommuW5, %momvelW5 : tensor<32x16x3x3xf32>
    %momlkW5 = stablehlo.add %momnvW5, %dW5 : tensor<32x16x3x3xf32>
    %momlrW5 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %momstW5 = stablehlo.multiply %momlrW5, %momlkW5 : tensor<32x16x3x3xf32>
    %momnewW5 = stablehlo.subtract %W5, %momstW5 : tensor<32x16x3x3xf32>
    %mommucb5 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %momvgcb5 = stablehlo.multiply %mommucb5, %cb5v : tensor<32xf32>
    %momvelcb5 = stablehlo.add %momvgcb5, %db5 : tensor<32xf32>
    %momnvcb5 = stablehlo.multiply %mommucb5, %momvelcb5 : tensor<32xf32>
    %momlkcb5 = stablehlo.add %momnvcb5, %db5 : tensor<32xf32>
    %momlrcb5 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %momstcb5 = stablehlo.multiply %momlrcb5, %momlkcb5 : tensor<32xf32>
    %momnewcb5 = stablehlo.subtract %cb5, %momstcb5 : tensor<32xf32>
    %mommuW6 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %momvgW6 = stablehlo.multiply %mommuW6, %W6v : tensor<32x32x3x3xf32>
    %momvelW6 = stablehlo.add %momvgW6, %dW6 : tensor<32x32x3x3xf32>
    %momnvW6 = stablehlo.multiply %mommuW6, %momvelW6 : tensor<32x32x3x3xf32>
    %momlkW6 = stablehlo.add %momnvW6, %dW6 : tensor<32x32x3x3xf32>
    %momlrW6 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %momstW6 = stablehlo.multiply %momlrW6, %momlkW6 : tensor<32x32x3x3xf32>
    %momnewW6 = stablehlo.subtract %W6, %momstW6 : tensor<32x32x3x3xf32>
    %mommucb6 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %momvgcb6 = stablehlo.multiply %mommucb6, %cb6v : tensor<32xf32>
    %momvelcb6 = stablehlo.add %momvgcb6, %db6 : tensor<32xf32>
    %momnvcb6 = stablehlo.multiply %mommucb6, %momvelcb6 : tensor<32xf32>
    %momlkcb6 = stablehlo.add %momnvcb6, %db6 : tensor<32xf32>
    %momlrcb6 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %momstcb6 = stablehlo.multiply %momlrcb6, %momlkcb6 : tensor<32xf32>
    %momnewcb6 = stablehlo.subtract %cb6, %momstcb6 : tensor<32xf32>
    %mommuW7 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %momvgW7 = stablehlo.multiply %mommuW7, %W7v : tensor<32x32x3x3xf32>
    %momvelW7 = stablehlo.add %momvgW7, %dW7 : tensor<32x32x3x3xf32>
    %momnvW7 = stablehlo.multiply %mommuW7, %momvelW7 : tensor<32x32x3x3xf32>
    %momlkW7 = stablehlo.add %momnvW7, %dW7 : tensor<32x32x3x3xf32>
    %momlrW7 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %momstW7 = stablehlo.multiply %momlrW7, %momlkW7 : tensor<32x32x3x3xf32>
    %momnewW7 = stablehlo.subtract %W7, %momstW7 : tensor<32x32x3x3xf32>
    %mommucb7 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %momvgcb7 = stablehlo.multiply %mommucb7, %cb7v : tensor<32xf32>
    %momvelcb7 = stablehlo.add %momvgcb7, %db7 : tensor<32xf32>
    %momnvcb7 = stablehlo.multiply %mommucb7, %momvelcb7 : tensor<32xf32>
    %momlkcb7 = stablehlo.add %momnvcb7, %db7 : tensor<32xf32>
    %momlrcb7 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %momstcb7 = stablehlo.multiply %momlrcb7, %momlkcb7 : tensor<32xf32>
    %momnewcb7 = stablehlo.subtract %cb7, %momstcb7 : tensor<32xf32>
    %mommuW8 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %momvgW8 = stablehlo.multiply %mommuW8, %W8v : tensor<32x32x3x3xf32>
    %momvelW8 = stablehlo.add %momvgW8, %dW8 : tensor<32x32x3x3xf32>
    %momnvW8 = stablehlo.multiply %mommuW8, %momvelW8 : tensor<32x32x3x3xf32>
    %momlkW8 = stablehlo.add %momnvW8, %dW8 : tensor<32x32x3x3xf32>
    %momlrW8 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %momstW8 = stablehlo.multiply %momlrW8, %momlkW8 : tensor<32x32x3x3xf32>
    %momnewW8 = stablehlo.subtract %W8, %momstW8 : tensor<32x32x3x3xf32>
    %mommucb8 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %momvgcb8 = stablehlo.multiply %mommucb8, %cb8v : tensor<32xf32>
    %momvelcb8 = stablehlo.add %momvgcb8, %db8 : tensor<32xf32>
    %momnvcb8 = stablehlo.multiply %mommucb8, %momvelcb8 : tensor<32xf32>
    %momlkcb8 = stablehlo.add %momnvcb8, %db8 : tensor<32xf32>
    %momlrcb8 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %momstcb8 = stablehlo.multiply %momlrcb8, %momlkcb8 : tensor<32xf32>
    %momnewcb8 = stablehlo.subtract %cb8, %momstcb8 : tensor<32xf32>
    %mommuW9 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %momvgW9 = stablehlo.multiply %mommuW9, %W9v : tensor<128x64xf32>
    %momvelW9 = stablehlo.add %momvgW9, %dW9 : tensor<128x64xf32>
    %momnvW9 = stablehlo.multiply %mommuW9, %momvelW9 : tensor<128x64xf32>
    %momlkW9 = stablehlo.add %momnvW9, %dW9 : tensor<128x64xf32>
    %momlrW9 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %momstW9 = stablehlo.multiply %momlrW9, %momlkW9 : tensor<128x64xf32>
    %momnewW9 = stablehlo.subtract %W9, %momstW9 : tensor<128x64xf32>
    %mommub9 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %momvgb9 = stablehlo.multiply %mommub9, %b9v : tensor<64xf32>
    %momvelb9 = stablehlo.add %momvgb9, %db9 : tensor<64xf32>
    %momnvb9 = stablehlo.multiply %mommub9, %momvelb9 : tensor<64xf32>
    %momlkb9 = stablehlo.add %momnvb9, %db9 : tensor<64xf32>
    %momlrb9 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %momstb9 = stablehlo.multiply %momlrb9, %momlkb9 : tensor<64xf32>
    %momnewb9 = stablehlo.subtract %b9, %momstb9 : tensor<64xf32>
    %mommuWa = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %momvgWa = stablehlo.multiply %mommuWa, %Wav : tensor<64x64xf32>
    %momvelWa = stablehlo.add %momvgWa, %dWa : tensor<64x64xf32>
    %momnvWa = stablehlo.multiply %mommuWa, %momvelWa : tensor<64x64xf32>
    %momlkWa = stablehlo.add %momnvWa, %dWa : tensor<64x64xf32>
    %momlrWa = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %momstWa = stablehlo.multiply %momlrWa, %momlkWa : tensor<64x64xf32>
    %momnewWa = stablehlo.subtract %Wa, %momstWa : tensor<64x64xf32>
    %mommuba = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %momvgba = stablehlo.multiply %mommuba, %bav : tensor<64xf32>
    %momvelba = stablehlo.add %momvgba, %dba : tensor<64xf32>
    %momnvba = stablehlo.multiply %mommuba, %momvelba : tensor<64xf32>
    %momlkba = stablehlo.add %momnvba, %dba : tensor<64xf32>
    %momlrba = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %momstba = stablehlo.multiply %momlrba, %momlkba : tensor<64xf32>
    %momnewba = stablehlo.subtract %ba, %momstba : tensor<64xf32>
    %mommuWb = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %momvgWb = stablehlo.multiply %mommuWb, %Wbv : tensor<64x10xf32>
    %momvelWb = stablehlo.add %momvgWb, %dWb : tensor<64x10xf32>
    %momnvWb = stablehlo.multiply %mommuWb, %momvelWb : tensor<64x10xf32>
    %momlkWb = stablehlo.add %momnvWb, %dWb : tensor<64x10xf32>
    %momlrWb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %momstWb = stablehlo.multiply %momlrWb, %momlkWb : tensor<64x10xf32>
    %momnewWb = stablehlo.subtract %Wb, %momstWb : tensor<64x10xf32>
    %mommubb = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %momvgbb = stablehlo.multiply %mommubb, %bbv : tensor<10xf32>
    %momvelbb = stablehlo.add %momvgbb, %dbb : tensor<10xf32>
    %momnvbb = stablehlo.multiply %mommubb, %momvelbb : tensor<10xf32>
    %momlkbb = stablehlo.add %momnvbb, %dbb : tensor<10xf32>
    %momlrbb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %momstbb = stablehlo.multiply %momlrbb, %momlkbb : tensor<10xf32>
    %momnewbb = stablehlo.subtract %bb, %momstbb : tensor<10xf32>
    return %momnewW1, %momnewcb1, %momnewW2, %momnewcb2, %momnewW3, %momnewcb3, %momnewW4, %momnewcb4, %momnewW5, %momnewcb5, %momnewW6, %momnewcb6, %momnewW7, %momnewcb7, %momnewW8, %momnewcb8, %momnewW9, %momnewb9, %momnewWa, %momnewba, %momnewWb, %momnewbb, %W1m, %cb1m, %W2m, %cb2m, %W3m, %cb3m, %W4m, %cb4m, %W5m, %cb5m, %W6m, %cb6m, %W7m, %cb7m, %W8m, %cb8m, %W9m, %b9m, %Wam, %bam, %Wbm, %bbm, %momvelW1, %momvelcb1, %momvelW2, %momvelcb2, %momvelW3, %momvelcb3, %momvelW4, %momvelcb4, %momvelW5, %momvelcb5, %momvelW6, %momvelcb6, %momvelW7, %momvelcb7, %momvelW8, %momvelcb8, %momvelW9, %momvelb9, %momvelWa, %momvelba, %momvelWb, %momvelbb, %loss, %bc1, %bc2 : tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>
  }
}

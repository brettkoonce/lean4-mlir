module @m {
  func.func @cifar8_bn_adam_train_step(%x: tensor<128x3072xf32>, %W1: tensor<16x3x3x3xf32>, %cb1: tensor<16xf32>, %g1: tensor<16xf32>, %bt1: tensor<16xf32>, %W2: tensor<16x16x3x3xf32>, %cb2: tensor<16xf32>, %g2: tensor<16xf32>, %bt2: tensor<16xf32>, %W3: tensor<16x16x3x3xf32>, %cb3: tensor<16xf32>, %g3: tensor<16xf32>, %bt3: tensor<16xf32>, %W4: tensor<16x16x3x3xf32>, %cb4: tensor<16xf32>, %g4: tensor<16xf32>, %bt4: tensor<16xf32>, %W5: tensor<32x16x3x3xf32>, %cb5: tensor<32xf32>, %g5: tensor<32xf32>, %bt5: tensor<32xf32>, %W6: tensor<32x32x3x3xf32>, %cb6: tensor<32xf32>, %g6: tensor<32xf32>, %bt6: tensor<32xf32>, %W7: tensor<32x32x3x3xf32>, %cb7: tensor<32xf32>, %g7: tensor<32xf32>, %bt7: tensor<32xf32>, %W8: tensor<32x32x3x3xf32>, %cb8: tensor<32xf32>, %g8: tensor<32xf32>, %bt8: tensor<32xf32>, %W9: tensor<128x64xf32>, %b9: tensor<64xf32>, %Wa: tensor<64x64xf32>, %ba: tensor<64xf32>, %Wb: tensor<64x10xf32>, %bb: tensor<10xf32>, %W1m: tensor<16x3x3x3xf32>, %cb1m: tensor<16xf32>, %g1m: tensor<16xf32>, %bt1m: tensor<16xf32>, %W2m: tensor<16x16x3x3xf32>, %cb2m: tensor<16xf32>, %g2m: tensor<16xf32>, %bt2m: tensor<16xf32>, %W3m: tensor<16x16x3x3xf32>, %cb3m: tensor<16xf32>, %g3m: tensor<16xf32>, %bt3m: tensor<16xf32>, %W4m: tensor<16x16x3x3xf32>, %cb4m: tensor<16xf32>, %g4m: tensor<16xf32>, %bt4m: tensor<16xf32>, %W5m: tensor<32x16x3x3xf32>, %cb5m: tensor<32xf32>, %g5m: tensor<32xf32>, %bt5m: tensor<32xf32>, %W6m: tensor<32x32x3x3xf32>, %cb6m: tensor<32xf32>, %g6m: tensor<32xf32>, %bt6m: tensor<32xf32>, %W7m: tensor<32x32x3x3xf32>, %cb7m: tensor<32xf32>, %g7m: tensor<32xf32>, %bt7m: tensor<32xf32>, %W8m: tensor<32x32x3x3xf32>, %cb8m: tensor<32xf32>, %g8m: tensor<32xf32>, %bt8m: tensor<32xf32>, %W9m: tensor<128x64xf32>, %b9m: tensor<64xf32>, %Wam: tensor<64x64xf32>, %bam: tensor<64xf32>, %Wbm: tensor<64x10xf32>, %bbm: tensor<10xf32>, %W1v: tensor<16x3x3x3xf32>, %cb1v: tensor<16xf32>, %g1v: tensor<16xf32>, %bt1v: tensor<16xf32>, %W2v: tensor<16x16x3x3xf32>, %cb2v: tensor<16xf32>, %g2v: tensor<16xf32>, %bt2v: tensor<16xf32>, %W3v: tensor<16x16x3x3xf32>, %cb3v: tensor<16xf32>, %g3v: tensor<16xf32>, %bt3v: tensor<16xf32>, %W4v: tensor<16x16x3x3xf32>, %cb4v: tensor<16xf32>, %g4v: tensor<16xf32>, %bt4v: tensor<16xf32>, %W5v: tensor<32x16x3x3xf32>, %cb5v: tensor<32xf32>, %g5v: tensor<32xf32>, %bt5v: tensor<32xf32>, %W6v: tensor<32x32x3x3xf32>, %cb6v: tensor<32xf32>, %g6v: tensor<32xf32>, %bt6v: tensor<32xf32>, %W7v: tensor<32x32x3x3xf32>, %cb7v: tensor<32xf32>, %g7v: tensor<32xf32>, %bt7v: tensor<32xf32>, %W8v: tensor<32x32x3x3xf32>, %cb8v: tensor<32xf32>, %g8v: tensor<32xf32>, %bt8v: tensor<32xf32>, %W9v: tensor<128x64xf32>, %b9v: tensor<64xf32>, %Wav: tensor<64x64xf32>, %bav: tensor<64xf32>, %Wbv: tensor<64x10xf32>, %bbv: tensor<10xf32>, %lr: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>, %onehot: tensor<128x10xf32>) -> (tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>) {
    %sc = stablehlo.constant dense<0.0> : tensor<f32>
    // ── forward: (conv→BN→relu)×2→pool ×4 →flatten→(dense→relu)×2→dense ──
    %xr = stablehlo.reshape %x : (tensor<128x3072xf32>) -> tensor<128x3x32x32xf32>
    %hc1c = stablehlo.convolution(%xr, %W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x3x32x32xf32>, tensor<16x3x3x3xf32>) -> tensor<128x16x32x32xf32>
    %hc1b = stablehlo.broadcast_in_dim %cb1, dims = [1] : (tensor<16xf32>) -> tensor<128x16x32x32xf32>
    %hc1 = stablehlo.add %hc1c, %hc1b : tensor<128x16x32x32xf32>
    %hc1f = stablehlo.reshape %hc1 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %bn1_xr = stablehlo.reshape %hc1f : (tensor<128x16384xf32>) -> tensor<128x16x1024xf32>
    %bn1_nf = stablehlo.constant dense<1024.0> : tensor<128x16x1024xf32>
    %bn1_ep = stablehlo.constant dense<1.0e-05> : tensor<128x16x1024xf32>
    %bn1_smr = stablehlo.reduce(%bn1_xr init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<128x16x1024xf32>, tensor<f32>) -> tensor<128x16xf32>
    %bn1_sm = stablehlo.broadcast_in_dim %bn1_smr, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x1024xf32>
    %bn1_mu = stablehlo.divide %bn1_sm, %bn1_nf : tensor<128x16x1024xf32>
    %bn1_xc = stablehlo.subtract %bn1_xr, %bn1_mu : tensor<128x16x1024xf32>
    %bn1_sq = stablehlo.multiply %bn1_xc, %bn1_xc : tensor<128x16x1024xf32>
    %bn1_vsr = stablehlo.reduce(%bn1_sq init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<128x16x1024xf32>, tensor<f32>) -> tensor<128x16xf32>
    %bn1_vs = stablehlo.broadcast_in_dim %bn1_vsr, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x1024xf32>
    %bn1_var = stablehlo.divide %bn1_vs, %bn1_nf : tensor<128x16x1024xf32>
    %bn1_ve = stablehlo.add %bn1_var, %bn1_ep : tensor<128x16x1024xf32>
    %bn1_istd = stablehlo.rsqrt %bn1_ve : tensor<128x16x1024xf32>
    %bn1_xhat = stablehlo.multiply %bn1_xc, %bn1_istd : tensor<128x16x1024xf32>
    %bn1_gb = stablehlo.broadcast_in_dim %g1, dims = [1] : (tensor<16xf32>) -> tensor<128x16x1024xf32>
    %bn1_bb = stablehlo.broadcast_in_dim %bt1, dims = [1] : (tensor<16xf32>) -> tensor<128x16x1024xf32>
    %bn1_gx = stablehlo.multiply %bn1_xhat, %bn1_gb : tensor<128x16x1024xf32>
    %bn1_y3 = stablehlo.add %bn1_gx, %bn1_bb : tensor<128x16x1024xf32>
    %bn1 = stablehlo.reshape %bn1_y3 : (tensor<128x16x1024xf32>) -> tensor<128x16384xf32>
    %ac1fz = stablehlo.constant dense<0.0> : tensor<128x16384xf32>
    %ac1f = stablehlo.maximum %bn1, %ac1fz : tensor<128x16384xf32>
    %ac1 = stablehlo.reshape %ac1f : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %hc2c = stablehlo.convolution(%ac1, %W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x32x32xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x32x32xf32>
    %hc2b = stablehlo.broadcast_in_dim %cb2, dims = [1] : (tensor<16xf32>) -> tensor<128x16x32x32xf32>
    %hc2 = stablehlo.add %hc2c, %hc2b : tensor<128x16x32x32xf32>
    %hc2f = stablehlo.reshape %hc2 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %bn2_xr = stablehlo.reshape %hc2f : (tensor<128x16384xf32>) -> tensor<128x16x1024xf32>
    %bn2_nf = stablehlo.constant dense<1024.0> : tensor<128x16x1024xf32>
    %bn2_ep = stablehlo.constant dense<1.0e-05> : tensor<128x16x1024xf32>
    %bn2_smr = stablehlo.reduce(%bn2_xr init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<128x16x1024xf32>, tensor<f32>) -> tensor<128x16xf32>
    %bn2_sm = stablehlo.broadcast_in_dim %bn2_smr, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x1024xf32>
    %bn2_mu = stablehlo.divide %bn2_sm, %bn2_nf : tensor<128x16x1024xf32>
    %bn2_xc = stablehlo.subtract %bn2_xr, %bn2_mu : tensor<128x16x1024xf32>
    %bn2_sq = stablehlo.multiply %bn2_xc, %bn2_xc : tensor<128x16x1024xf32>
    %bn2_vsr = stablehlo.reduce(%bn2_sq init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<128x16x1024xf32>, tensor<f32>) -> tensor<128x16xf32>
    %bn2_vs = stablehlo.broadcast_in_dim %bn2_vsr, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x1024xf32>
    %bn2_var = stablehlo.divide %bn2_vs, %bn2_nf : tensor<128x16x1024xf32>
    %bn2_ve = stablehlo.add %bn2_var, %bn2_ep : tensor<128x16x1024xf32>
    %bn2_istd = stablehlo.rsqrt %bn2_ve : tensor<128x16x1024xf32>
    %bn2_xhat = stablehlo.multiply %bn2_xc, %bn2_istd : tensor<128x16x1024xf32>
    %bn2_gb = stablehlo.broadcast_in_dim %g2, dims = [1] : (tensor<16xf32>) -> tensor<128x16x1024xf32>
    %bn2_bb = stablehlo.broadcast_in_dim %bt2, dims = [1] : (tensor<16xf32>) -> tensor<128x16x1024xf32>
    %bn2_gx = stablehlo.multiply %bn2_xhat, %bn2_gb : tensor<128x16x1024xf32>
    %bn2_y3 = stablehlo.add %bn2_gx, %bn2_bb : tensor<128x16x1024xf32>
    %bn2 = stablehlo.reshape %bn2_y3 : (tensor<128x16x1024xf32>) -> tensor<128x16384xf32>
    %ac2fz = stablehlo.constant dense<0.0> : tensor<128x16384xf32>
    %ac2f = stablehlo.maximum %bn2, %ac2fz : tensor<128x16384xf32>
    %ac2 = stablehlo.reshape %ac2f : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
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
    %hc3f = stablehlo.reshape %hc3 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %bn3_xr = stablehlo.reshape %hc3f : (tensor<128x4096xf32>) -> tensor<128x16x256xf32>
    %bn3_nf = stablehlo.constant dense<256.0> : tensor<128x16x256xf32>
    %bn3_ep = stablehlo.constant dense<1.0e-05> : tensor<128x16x256xf32>
    %bn3_smr = stablehlo.reduce(%bn3_xr init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<128x16x256xf32>, tensor<f32>) -> tensor<128x16xf32>
    %bn3_sm = stablehlo.broadcast_in_dim %bn3_smr, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x256xf32>
    %bn3_mu = stablehlo.divide %bn3_sm, %bn3_nf : tensor<128x16x256xf32>
    %bn3_xc = stablehlo.subtract %bn3_xr, %bn3_mu : tensor<128x16x256xf32>
    %bn3_sq = stablehlo.multiply %bn3_xc, %bn3_xc : tensor<128x16x256xf32>
    %bn3_vsr = stablehlo.reduce(%bn3_sq init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<128x16x256xf32>, tensor<f32>) -> tensor<128x16xf32>
    %bn3_vs = stablehlo.broadcast_in_dim %bn3_vsr, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x256xf32>
    %bn3_var = stablehlo.divide %bn3_vs, %bn3_nf : tensor<128x16x256xf32>
    %bn3_ve = stablehlo.add %bn3_var, %bn3_ep : tensor<128x16x256xf32>
    %bn3_istd = stablehlo.rsqrt %bn3_ve : tensor<128x16x256xf32>
    %bn3_xhat = stablehlo.multiply %bn3_xc, %bn3_istd : tensor<128x16x256xf32>
    %bn3_gb = stablehlo.broadcast_in_dim %g3, dims = [1] : (tensor<16xf32>) -> tensor<128x16x256xf32>
    %bn3_bb = stablehlo.broadcast_in_dim %bt3, dims = [1] : (tensor<16xf32>) -> tensor<128x16x256xf32>
    %bn3_gx = stablehlo.multiply %bn3_xhat, %bn3_gb : tensor<128x16x256xf32>
    %bn3_y3 = stablehlo.add %bn3_gx, %bn3_bb : tensor<128x16x256xf32>
    %bn3 = stablehlo.reshape %bn3_y3 : (tensor<128x16x256xf32>) -> tensor<128x4096xf32>
    %ac3fz = stablehlo.constant dense<0.0> : tensor<128x4096xf32>
    %ac3f = stablehlo.maximum %bn3, %ac3fz : tensor<128x4096xf32>
    %ac3 = stablehlo.reshape %ac3f : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %hc4c = stablehlo.convolution(%ac3, %W4)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x16x16xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x16x16xf32>
    %hc4b = stablehlo.broadcast_in_dim %cb4, dims = [1] : (tensor<16xf32>) -> tensor<128x16x16x16xf32>
    %hc4 = stablehlo.add %hc4c, %hc4b : tensor<128x16x16x16xf32>
    %hc4f = stablehlo.reshape %hc4 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %bn4_xr = stablehlo.reshape %hc4f : (tensor<128x4096xf32>) -> tensor<128x16x256xf32>
    %bn4_nf = stablehlo.constant dense<256.0> : tensor<128x16x256xf32>
    %bn4_ep = stablehlo.constant dense<1.0e-05> : tensor<128x16x256xf32>
    %bn4_smr = stablehlo.reduce(%bn4_xr init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<128x16x256xf32>, tensor<f32>) -> tensor<128x16xf32>
    %bn4_sm = stablehlo.broadcast_in_dim %bn4_smr, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x256xf32>
    %bn4_mu = stablehlo.divide %bn4_sm, %bn4_nf : tensor<128x16x256xf32>
    %bn4_xc = stablehlo.subtract %bn4_xr, %bn4_mu : tensor<128x16x256xf32>
    %bn4_sq = stablehlo.multiply %bn4_xc, %bn4_xc : tensor<128x16x256xf32>
    %bn4_vsr = stablehlo.reduce(%bn4_sq init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<128x16x256xf32>, tensor<f32>) -> tensor<128x16xf32>
    %bn4_vs = stablehlo.broadcast_in_dim %bn4_vsr, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x256xf32>
    %bn4_var = stablehlo.divide %bn4_vs, %bn4_nf : tensor<128x16x256xf32>
    %bn4_ve = stablehlo.add %bn4_var, %bn4_ep : tensor<128x16x256xf32>
    %bn4_istd = stablehlo.rsqrt %bn4_ve : tensor<128x16x256xf32>
    %bn4_xhat = stablehlo.multiply %bn4_xc, %bn4_istd : tensor<128x16x256xf32>
    %bn4_gb = stablehlo.broadcast_in_dim %g4, dims = [1] : (tensor<16xf32>) -> tensor<128x16x256xf32>
    %bn4_bb = stablehlo.broadcast_in_dim %bt4, dims = [1] : (tensor<16xf32>) -> tensor<128x16x256xf32>
    %bn4_gx = stablehlo.multiply %bn4_xhat, %bn4_gb : tensor<128x16x256xf32>
    %bn4_y3 = stablehlo.add %bn4_gx, %bn4_bb : tensor<128x16x256xf32>
    %bn4 = stablehlo.reshape %bn4_y3 : (tensor<128x16x256xf32>) -> tensor<128x4096xf32>
    %ac4fz = stablehlo.constant dense<0.0> : tensor<128x4096xf32>
    %ac4f = stablehlo.maximum %bn4, %ac4fz : tensor<128x4096xf32>
    %ac4 = stablehlo.reshape %ac4f : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
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
    %hc5f = stablehlo.reshape %hc5 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %bn5_xr = stablehlo.reshape %hc5f : (tensor<128x2048xf32>) -> tensor<128x32x64xf32>
    %bn5_nf = stablehlo.constant dense<64.0> : tensor<128x32x64xf32>
    %bn5_ep = stablehlo.constant dense<1.0e-05> : tensor<128x32x64xf32>
    %bn5_smr = stablehlo.reduce(%bn5_xr init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<128x32x64xf32>, tensor<f32>) -> tensor<128x32xf32>
    %bn5_sm = stablehlo.broadcast_in_dim %bn5_smr, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x64xf32>
    %bn5_mu = stablehlo.divide %bn5_sm, %bn5_nf : tensor<128x32x64xf32>
    %bn5_xc = stablehlo.subtract %bn5_xr, %bn5_mu : tensor<128x32x64xf32>
    %bn5_sq = stablehlo.multiply %bn5_xc, %bn5_xc : tensor<128x32x64xf32>
    %bn5_vsr = stablehlo.reduce(%bn5_sq init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<128x32x64xf32>, tensor<f32>) -> tensor<128x32xf32>
    %bn5_vs = stablehlo.broadcast_in_dim %bn5_vsr, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x64xf32>
    %bn5_var = stablehlo.divide %bn5_vs, %bn5_nf : tensor<128x32x64xf32>
    %bn5_ve = stablehlo.add %bn5_var, %bn5_ep : tensor<128x32x64xf32>
    %bn5_istd = stablehlo.rsqrt %bn5_ve : tensor<128x32x64xf32>
    %bn5_xhat = stablehlo.multiply %bn5_xc, %bn5_istd : tensor<128x32x64xf32>
    %bn5_gb = stablehlo.broadcast_in_dim %g5, dims = [1] : (tensor<32xf32>) -> tensor<128x32x64xf32>
    %bn5_bb = stablehlo.broadcast_in_dim %bt5, dims = [1] : (tensor<32xf32>) -> tensor<128x32x64xf32>
    %bn5_gx = stablehlo.multiply %bn5_xhat, %bn5_gb : tensor<128x32x64xf32>
    %bn5_y3 = stablehlo.add %bn5_gx, %bn5_bb : tensor<128x32x64xf32>
    %bn5 = stablehlo.reshape %bn5_y3 : (tensor<128x32x64xf32>) -> tensor<128x2048xf32>
    %ac5fz = stablehlo.constant dense<0.0> : tensor<128x2048xf32>
    %ac5f = stablehlo.maximum %bn5, %ac5fz : tensor<128x2048xf32>
    %ac5 = stablehlo.reshape %ac5f : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %hc6c = stablehlo.convolution(%ac5, %W6)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x8x8xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x8x8xf32>
    %hc6b = stablehlo.broadcast_in_dim %cb6, dims = [1] : (tensor<32xf32>) -> tensor<128x32x8x8xf32>
    %hc6 = stablehlo.add %hc6c, %hc6b : tensor<128x32x8x8xf32>
    %hc6f = stablehlo.reshape %hc6 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %bn6_xr = stablehlo.reshape %hc6f : (tensor<128x2048xf32>) -> tensor<128x32x64xf32>
    %bn6_nf = stablehlo.constant dense<64.0> : tensor<128x32x64xf32>
    %bn6_ep = stablehlo.constant dense<1.0e-05> : tensor<128x32x64xf32>
    %bn6_smr = stablehlo.reduce(%bn6_xr init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<128x32x64xf32>, tensor<f32>) -> tensor<128x32xf32>
    %bn6_sm = stablehlo.broadcast_in_dim %bn6_smr, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x64xf32>
    %bn6_mu = stablehlo.divide %bn6_sm, %bn6_nf : tensor<128x32x64xf32>
    %bn6_xc = stablehlo.subtract %bn6_xr, %bn6_mu : tensor<128x32x64xf32>
    %bn6_sq = stablehlo.multiply %bn6_xc, %bn6_xc : tensor<128x32x64xf32>
    %bn6_vsr = stablehlo.reduce(%bn6_sq init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<128x32x64xf32>, tensor<f32>) -> tensor<128x32xf32>
    %bn6_vs = stablehlo.broadcast_in_dim %bn6_vsr, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x64xf32>
    %bn6_var = stablehlo.divide %bn6_vs, %bn6_nf : tensor<128x32x64xf32>
    %bn6_ve = stablehlo.add %bn6_var, %bn6_ep : tensor<128x32x64xf32>
    %bn6_istd = stablehlo.rsqrt %bn6_ve : tensor<128x32x64xf32>
    %bn6_xhat = stablehlo.multiply %bn6_xc, %bn6_istd : tensor<128x32x64xf32>
    %bn6_gb = stablehlo.broadcast_in_dim %g6, dims = [1] : (tensor<32xf32>) -> tensor<128x32x64xf32>
    %bn6_bb = stablehlo.broadcast_in_dim %bt6, dims = [1] : (tensor<32xf32>) -> tensor<128x32x64xf32>
    %bn6_gx = stablehlo.multiply %bn6_xhat, %bn6_gb : tensor<128x32x64xf32>
    %bn6_y3 = stablehlo.add %bn6_gx, %bn6_bb : tensor<128x32x64xf32>
    %bn6 = stablehlo.reshape %bn6_y3 : (tensor<128x32x64xf32>) -> tensor<128x2048xf32>
    %ac6fz = stablehlo.constant dense<0.0> : tensor<128x2048xf32>
    %ac6f = stablehlo.maximum %bn6, %ac6fz : tensor<128x2048xf32>
    %ac6 = stablehlo.reshape %ac6f : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
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
    %hc7f = stablehlo.reshape %hc7 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %bn7_xr = stablehlo.reshape %hc7f : (tensor<128x512xf32>) -> tensor<128x32x16xf32>
    %bn7_nf = stablehlo.constant dense<16.0> : tensor<128x32x16xf32>
    %bn7_ep = stablehlo.constant dense<1.0e-05> : tensor<128x32x16xf32>
    %bn7_smr = stablehlo.reduce(%bn7_xr init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<128x32x16xf32>, tensor<f32>) -> tensor<128x32xf32>
    %bn7_sm = stablehlo.broadcast_in_dim %bn7_smr, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x16xf32>
    %bn7_mu = stablehlo.divide %bn7_sm, %bn7_nf : tensor<128x32x16xf32>
    %bn7_xc = stablehlo.subtract %bn7_xr, %bn7_mu : tensor<128x32x16xf32>
    %bn7_sq = stablehlo.multiply %bn7_xc, %bn7_xc : tensor<128x32x16xf32>
    %bn7_vsr = stablehlo.reduce(%bn7_sq init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<128x32x16xf32>, tensor<f32>) -> tensor<128x32xf32>
    %bn7_vs = stablehlo.broadcast_in_dim %bn7_vsr, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x16xf32>
    %bn7_var = stablehlo.divide %bn7_vs, %bn7_nf : tensor<128x32x16xf32>
    %bn7_ve = stablehlo.add %bn7_var, %bn7_ep : tensor<128x32x16xf32>
    %bn7_istd = stablehlo.rsqrt %bn7_ve : tensor<128x32x16xf32>
    %bn7_xhat = stablehlo.multiply %bn7_xc, %bn7_istd : tensor<128x32x16xf32>
    %bn7_gb = stablehlo.broadcast_in_dim %g7, dims = [1] : (tensor<32xf32>) -> tensor<128x32x16xf32>
    %bn7_bb = stablehlo.broadcast_in_dim %bt7, dims = [1] : (tensor<32xf32>) -> tensor<128x32x16xf32>
    %bn7_gx = stablehlo.multiply %bn7_xhat, %bn7_gb : tensor<128x32x16xf32>
    %bn7_y3 = stablehlo.add %bn7_gx, %bn7_bb : tensor<128x32x16xf32>
    %bn7 = stablehlo.reshape %bn7_y3 : (tensor<128x32x16xf32>) -> tensor<128x512xf32>
    %ac7fz = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %ac7f = stablehlo.maximum %bn7, %ac7fz : tensor<128x512xf32>
    %ac7 = stablehlo.reshape %ac7f : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %hc8c = stablehlo.convolution(%ac7, %W8)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x4x4xf32>
    %hc8b = stablehlo.broadcast_in_dim %cb8, dims = [1] : (tensor<32xf32>) -> tensor<128x32x4x4xf32>
    %hc8 = stablehlo.add %hc8c, %hc8b : tensor<128x32x4x4xf32>
    %hc8f = stablehlo.reshape %hc8 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %bn8_xr = stablehlo.reshape %hc8f : (tensor<128x512xf32>) -> tensor<128x32x16xf32>
    %bn8_nf = stablehlo.constant dense<16.0> : tensor<128x32x16xf32>
    %bn8_ep = stablehlo.constant dense<1.0e-05> : tensor<128x32x16xf32>
    %bn8_smr = stablehlo.reduce(%bn8_xr init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<128x32x16xf32>, tensor<f32>) -> tensor<128x32xf32>
    %bn8_sm = stablehlo.broadcast_in_dim %bn8_smr, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x16xf32>
    %bn8_mu = stablehlo.divide %bn8_sm, %bn8_nf : tensor<128x32x16xf32>
    %bn8_xc = stablehlo.subtract %bn8_xr, %bn8_mu : tensor<128x32x16xf32>
    %bn8_sq = stablehlo.multiply %bn8_xc, %bn8_xc : tensor<128x32x16xf32>
    %bn8_vsr = stablehlo.reduce(%bn8_sq init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<128x32x16xf32>, tensor<f32>) -> tensor<128x32xf32>
    %bn8_vs = stablehlo.broadcast_in_dim %bn8_vsr, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x16xf32>
    %bn8_var = stablehlo.divide %bn8_vs, %bn8_nf : tensor<128x32x16xf32>
    %bn8_ve = stablehlo.add %bn8_var, %bn8_ep : tensor<128x32x16xf32>
    %bn8_istd = stablehlo.rsqrt %bn8_ve : tensor<128x32x16xf32>
    %bn8_xhat = stablehlo.multiply %bn8_xc, %bn8_istd : tensor<128x32x16xf32>
    %bn8_gb = stablehlo.broadcast_in_dim %g8, dims = [1] : (tensor<32xf32>) -> tensor<128x32x16xf32>
    %bn8_bb = stablehlo.broadcast_in_dim %bt8, dims = [1] : (tensor<32xf32>) -> tensor<128x32x16xf32>
    %bn8_gx = stablehlo.multiply %bn8_xhat, %bn8_gb : tensor<128x32x16xf32>
    %bn8_y3 = stablehlo.add %bn8_gx, %bn8_bb : tensor<128x32x16xf32>
    %bn8 = stablehlo.reshape %bn8_y3 : (tensor<128x32x16xf32>) -> tensor<128x512xf32>
    %ac8fz = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %ac8f = stablehlo.maximum %bn8, %ac8fz : tensor<128x512xf32>
    %ac8 = stablehlo.reshape %ac8f : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
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
    // ── backward: dense+relu → scatter → (relu→BN-back→convBack)×stage, four stages ──
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
    %dac8f = stablehlo.reshape %dac8 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %dbn8z = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %dbn8m = stablehlo.compare GT, %bn8, %dbn8z : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<128x512xi1>
    %dbn8 = stablehlo.select %dbn8m, %dac8f, %dbn8z : tensor<128x512xi1>, tensor<128x512xf32>
    %dhc8f_dyr = stablehlo.reshape %dbn8 : (tensor<128x512xf32>) -> tensor<128x32x16xf32>
    %dhc8f_gb = stablehlo.broadcast_in_dim %g8, dims = [1] : (tensor<32xf32>) -> tensor<128x32x16xf32>
    %dhc8f_dxh = stablehlo.multiply %dhc8f_gb, %dhc8f_dyr : tensor<128x32x16xf32>
    %dhc8f_sdxr = stablehlo.reduce(%dhc8f_dxh init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<128x32x16xf32>, tensor<f32>) -> tensor<128x32xf32>
    %dhc8f_sdx = stablehlo.broadcast_in_dim %dhc8f_sdxr, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x16xf32>
    %dhc8f_xd = stablehlo.multiply %bn8_xhat, %dhc8f_dxh : tensor<128x32x16xf32>
    %dhc8f_sxdr = stablehlo.reduce(%dhc8f_xd init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<128x32x16xf32>, tensor<f32>) -> tensor<128x32xf32>
    %dhc8f_sxd = stablehlo.broadcast_in_dim %dhc8f_sxdr, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x16xf32>
    %dhc8f_t1 = stablehlo.multiply %dhc8f_dxh, %bn8_nf : tensor<128x32x16xf32>
    %dhc8f_i1 = stablehlo.subtract %dhc8f_t1, %dhc8f_sdx : tensor<128x32x16xf32>
    %dhc8f_xs = stablehlo.multiply %bn8_xhat, %dhc8f_sxd : tensor<128x32x16xf32>
    %dhc8f_i2 = stablehlo.subtract %dhc8f_i1, %dhc8f_xs : tensor<128x32x16xf32>
    %dhc8f_s = stablehlo.divide %bn8_istd, %bn8_nf : tensor<128x32x16xf32>
    %dhc8f_dx3 = stablehlo.multiply %dhc8f_s, %dhc8f_i2 : tensor<128x32x16xf32>
    %dhc8f = stablehlo.reshape %dhc8f_dx3 : (tensor<128x32x16xf32>) -> tensor<128x512xf32>
    %dg8_dyr = stablehlo.reshape %dbn8 : (tensor<128x512xf32>) -> tensor<128x32x16xf32>
    %dg8_p = stablehlo.multiply %dg8_dyr, %bn8_xhat : tensor<128x32x16xf32>
    %dg8 = stablehlo.reduce(%dg8_p init: %sc) applies stablehlo.add across dimensions = [0, 2] : (tensor<128x32x16xf32>, tensor<f32>) -> tensor<32xf32>
    %dbt8 = stablehlo.reduce(%dg8_dyr init: %sc) applies stablehlo.add across dimensions = [0, 2] : (tensor<128x32x16xf32>, tensor<f32>) -> tensor<32xf32>
    %dhc8 = stablehlo.reshape %dhc8f : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %dac7t = stablehlo.transpose %W8, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %dac7r = stablehlo.reverse %dac7t, dims = [2, 3] : tensor<32x32x3x3xf32>
    %dac7 = stablehlo.convolution(%dhc8, %dac7r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x4x4xf32>
    %dac7f = stablehlo.reshape %dac7 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %dbn7z = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %dbn7m = stablehlo.compare GT, %bn7, %dbn7z : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<128x512xi1>
    %dbn7 = stablehlo.select %dbn7m, %dac7f, %dbn7z : tensor<128x512xi1>, tensor<128x512xf32>
    %dhc7f_dyr = stablehlo.reshape %dbn7 : (tensor<128x512xf32>) -> tensor<128x32x16xf32>
    %dhc7f_gb = stablehlo.broadcast_in_dim %g7, dims = [1] : (tensor<32xf32>) -> tensor<128x32x16xf32>
    %dhc7f_dxh = stablehlo.multiply %dhc7f_gb, %dhc7f_dyr : tensor<128x32x16xf32>
    %dhc7f_sdxr = stablehlo.reduce(%dhc7f_dxh init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<128x32x16xf32>, tensor<f32>) -> tensor<128x32xf32>
    %dhc7f_sdx = stablehlo.broadcast_in_dim %dhc7f_sdxr, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x16xf32>
    %dhc7f_xd = stablehlo.multiply %bn7_xhat, %dhc7f_dxh : tensor<128x32x16xf32>
    %dhc7f_sxdr = stablehlo.reduce(%dhc7f_xd init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<128x32x16xf32>, tensor<f32>) -> tensor<128x32xf32>
    %dhc7f_sxd = stablehlo.broadcast_in_dim %dhc7f_sxdr, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x16xf32>
    %dhc7f_t1 = stablehlo.multiply %dhc7f_dxh, %bn7_nf : tensor<128x32x16xf32>
    %dhc7f_i1 = stablehlo.subtract %dhc7f_t1, %dhc7f_sdx : tensor<128x32x16xf32>
    %dhc7f_xs = stablehlo.multiply %bn7_xhat, %dhc7f_sxd : tensor<128x32x16xf32>
    %dhc7f_i2 = stablehlo.subtract %dhc7f_i1, %dhc7f_xs : tensor<128x32x16xf32>
    %dhc7f_s = stablehlo.divide %bn7_istd, %bn7_nf : tensor<128x32x16xf32>
    %dhc7f_dx3 = stablehlo.multiply %dhc7f_s, %dhc7f_i2 : tensor<128x32x16xf32>
    %dhc7f = stablehlo.reshape %dhc7f_dx3 : (tensor<128x32x16xf32>) -> tensor<128x512xf32>
    %dg7_dyr = stablehlo.reshape %dbn7 : (tensor<128x512xf32>) -> tensor<128x32x16xf32>
    %dg7_p = stablehlo.multiply %dg7_dyr, %bn7_xhat : tensor<128x32x16xf32>
    %dg7 = stablehlo.reduce(%dg7_p init: %sc) applies stablehlo.add across dimensions = [0, 2] : (tensor<128x32x16xf32>, tensor<f32>) -> tensor<32xf32>
    %dbt7 = stablehlo.reduce(%dg7_dyr init: %sc) applies stablehlo.add across dimensions = [0, 2] : (tensor<128x32x16xf32>, tensor<f32>) -> tensor<32xf32>
    %dhc7 = stablehlo.reshape %dhc7f : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
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
    %dac6f = stablehlo.reshape %dac6 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %dbn6z = stablehlo.constant dense<0.0> : tensor<128x2048xf32>
    %dbn6m = stablehlo.compare GT, %bn6, %dbn6z : (tensor<128x2048xf32>, tensor<128x2048xf32>) -> tensor<128x2048xi1>
    %dbn6 = stablehlo.select %dbn6m, %dac6f, %dbn6z : tensor<128x2048xi1>, tensor<128x2048xf32>
    %dhc6f_dyr = stablehlo.reshape %dbn6 : (tensor<128x2048xf32>) -> tensor<128x32x64xf32>
    %dhc6f_gb = stablehlo.broadcast_in_dim %g6, dims = [1] : (tensor<32xf32>) -> tensor<128x32x64xf32>
    %dhc6f_dxh = stablehlo.multiply %dhc6f_gb, %dhc6f_dyr : tensor<128x32x64xf32>
    %dhc6f_sdxr = stablehlo.reduce(%dhc6f_dxh init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<128x32x64xf32>, tensor<f32>) -> tensor<128x32xf32>
    %dhc6f_sdx = stablehlo.broadcast_in_dim %dhc6f_sdxr, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x64xf32>
    %dhc6f_xd = stablehlo.multiply %bn6_xhat, %dhc6f_dxh : tensor<128x32x64xf32>
    %dhc6f_sxdr = stablehlo.reduce(%dhc6f_xd init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<128x32x64xf32>, tensor<f32>) -> tensor<128x32xf32>
    %dhc6f_sxd = stablehlo.broadcast_in_dim %dhc6f_sxdr, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x64xf32>
    %dhc6f_t1 = stablehlo.multiply %dhc6f_dxh, %bn6_nf : tensor<128x32x64xf32>
    %dhc6f_i1 = stablehlo.subtract %dhc6f_t1, %dhc6f_sdx : tensor<128x32x64xf32>
    %dhc6f_xs = stablehlo.multiply %bn6_xhat, %dhc6f_sxd : tensor<128x32x64xf32>
    %dhc6f_i2 = stablehlo.subtract %dhc6f_i1, %dhc6f_xs : tensor<128x32x64xf32>
    %dhc6f_s = stablehlo.divide %bn6_istd, %bn6_nf : tensor<128x32x64xf32>
    %dhc6f_dx3 = stablehlo.multiply %dhc6f_s, %dhc6f_i2 : tensor<128x32x64xf32>
    %dhc6f = stablehlo.reshape %dhc6f_dx3 : (tensor<128x32x64xf32>) -> tensor<128x2048xf32>
    %dg6_dyr = stablehlo.reshape %dbn6 : (tensor<128x2048xf32>) -> tensor<128x32x64xf32>
    %dg6_p = stablehlo.multiply %dg6_dyr, %bn6_xhat : tensor<128x32x64xf32>
    %dg6 = stablehlo.reduce(%dg6_p init: %sc) applies stablehlo.add across dimensions = [0, 2] : (tensor<128x32x64xf32>, tensor<f32>) -> tensor<32xf32>
    %dbt6 = stablehlo.reduce(%dg6_dyr init: %sc) applies stablehlo.add across dimensions = [0, 2] : (tensor<128x32x64xf32>, tensor<f32>) -> tensor<32xf32>
    %dhc6 = stablehlo.reshape %dhc6f : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %dac5t = stablehlo.transpose %W6, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %dac5r = stablehlo.reverse %dac5t, dims = [2, 3] : tensor<32x32x3x3xf32>
    %dac5 = stablehlo.convolution(%dhc6, %dac5r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x8x8xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x8x8xf32>
    %dac5f = stablehlo.reshape %dac5 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %dbn5z = stablehlo.constant dense<0.0> : tensor<128x2048xf32>
    %dbn5m = stablehlo.compare GT, %bn5, %dbn5z : (tensor<128x2048xf32>, tensor<128x2048xf32>) -> tensor<128x2048xi1>
    %dbn5 = stablehlo.select %dbn5m, %dac5f, %dbn5z : tensor<128x2048xi1>, tensor<128x2048xf32>
    %dhc5f_dyr = stablehlo.reshape %dbn5 : (tensor<128x2048xf32>) -> tensor<128x32x64xf32>
    %dhc5f_gb = stablehlo.broadcast_in_dim %g5, dims = [1] : (tensor<32xf32>) -> tensor<128x32x64xf32>
    %dhc5f_dxh = stablehlo.multiply %dhc5f_gb, %dhc5f_dyr : tensor<128x32x64xf32>
    %dhc5f_sdxr = stablehlo.reduce(%dhc5f_dxh init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<128x32x64xf32>, tensor<f32>) -> tensor<128x32xf32>
    %dhc5f_sdx = stablehlo.broadcast_in_dim %dhc5f_sdxr, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x64xf32>
    %dhc5f_xd = stablehlo.multiply %bn5_xhat, %dhc5f_dxh : tensor<128x32x64xf32>
    %dhc5f_sxdr = stablehlo.reduce(%dhc5f_xd init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<128x32x64xf32>, tensor<f32>) -> tensor<128x32xf32>
    %dhc5f_sxd = stablehlo.broadcast_in_dim %dhc5f_sxdr, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x64xf32>
    %dhc5f_t1 = stablehlo.multiply %dhc5f_dxh, %bn5_nf : tensor<128x32x64xf32>
    %dhc5f_i1 = stablehlo.subtract %dhc5f_t1, %dhc5f_sdx : tensor<128x32x64xf32>
    %dhc5f_xs = stablehlo.multiply %bn5_xhat, %dhc5f_sxd : tensor<128x32x64xf32>
    %dhc5f_i2 = stablehlo.subtract %dhc5f_i1, %dhc5f_xs : tensor<128x32x64xf32>
    %dhc5f_s = stablehlo.divide %bn5_istd, %bn5_nf : tensor<128x32x64xf32>
    %dhc5f_dx3 = stablehlo.multiply %dhc5f_s, %dhc5f_i2 : tensor<128x32x64xf32>
    %dhc5f = stablehlo.reshape %dhc5f_dx3 : (tensor<128x32x64xf32>) -> tensor<128x2048xf32>
    %dg5_dyr = stablehlo.reshape %dbn5 : (tensor<128x2048xf32>) -> tensor<128x32x64xf32>
    %dg5_p = stablehlo.multiply %dg5_dyr, %bn5_xhat : tensor<128x32x64xf32>
    %dg5 = stablehlo.reduce(%dg5_p init: %sc) applies stablehlo.add across dimensions = [0, 2] : (tensor<128x32x64xf32>, tensor<f32>) -> tensor<32xf32>
    %dbt5 = stablehlo.reduce(%dg5_dyr init: %sc) applies stablehlo.add across dimensions = [0, 2] : (tensor<128x32x64xf32>, tensor<f32>) -> tensor<32xf32>
    %dhc5 = stablehlo.reshape %dhc5f : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
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
    %dac4f = stablehlo.reshape %dac4 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %dbn4z = stablehlo.constant dense<0.0> : tensor<128x4096xf32>
    %dbn4m = stablehlo.compare GT, %bn4, %dbn4z : (tensor<128x4096xf32>, tensor<128x4096xf32>) -> tensor<128x4096xi1>
    %dbn4 = stablehlo.select %dbn4m, %dac4f, %dbn4z : tensor<128x4096xi1>, tensor<128x4096xf32>
    %dhc4f_dyr = stablehlo.reshape %dbn4 : (tensor<128x4096xf32>) -> tensor<128x16x256xf32>
    %dhc4f_gb = stablehlo.broadcast_in_dim %g4, dims = [1] : (tensor<16xf32>) -> tensor<128x16x256xf32>
    %dhc4f_dxh = stablehlo.multiply %dhc4f_gb, %dhc4f_dyr : tensor<128x16x256xf32>
    %dhc4f_sdxr = stablehlo.reduce(%dhc4f_dxh init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<128x16x256xf32>, tensor<f32>) -> tensor<128x16xf32>
    %dhc4f_sdx = stablehlo.broadcast_in_dim %dhc4f_sdxr, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x256xf32>
    %dhc4f_xd = stablehlo.multiply %bn4_xhat, %dhc4f_dxh : tensor<128x16x256xf32>
    %dhc4f_sxdr = stablehlo.reduce(%dhc4f_xd init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<128x16x256xf32>, tensor<f32>) -> tensor<128x16xf32>
    %dhc4f_sxd = stablehlo.broadcast_in_dim %dhc4f_sxdr, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x256xf32>
    %dhc4f_t1 = stablehlo.multiply %dhc4f_dxh, %bn4_nf : tensor<128x16x256xf32>
    %dhc4f_i1 = stablehlo.subtract %dhc4f_t1, %dhc4f_sdx : tensor<128x16x256xf32>
    %dhc4f_xs = stablehlo.multiply %bn4_xhat, %dhc4f_sxd : tensor<128x16x256xf32>
    %dhc4f_i2 = stablehlo.subtract %dhc4f_i1, %dhc4f_xs : tensor<128x16x256xf32>
    %dhc4f_s = stablehlo.divide %bn4_istd, %bn4_nf : tensor<128x16x256xf32>
    %dhc4f_dx3 = stablehlo.multiply %dhc4f_s, %dhc4f_i2 : tensor<128x16x256xf32>
    %dhc4f = stablehlo.reshape %dhc4f_dx3 : (tensor<128x16x256xf32>) -> tensor<128x4096xf32>
    %dg4_dyr = stablehlo.reshape %dbn4 : (tensor<128x4096xf32>) -> tensor<128x16x256xf32>
    %dg4_p = stablehlo.multiply %dg4_dyr, %bn4_xhat : tensor<128x16x256xf32>
    %dg4 = stablehlo.reduce(%dg4_p init: %sc) applies stablehlo.add across dimensions = [0, 2] : (tensor<128x16x256xf32>, tensor<f32>) -> tensor<16xf32>
    %dbt4 = stablehlo.reduce(%dg4_dyr init: %sc) applies stablehlo.add across dimensions = [0, 2] : (tensor<128x16x256xf32>, tensor<f32>) -> tensor<16xf32>
    %dhc4 = stablehlo.reshape %dhc4f : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %dac3t = stablehlo.transpose %W4, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %dac3r = stablehlo.reverse %dac3t, dims = [2, 3] : tensor<16x16x3x3xf32>
    %dac3 = stablehlo.convolution(%dhc4, %dac3r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x16x16xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x16x16xf32>
    %dac3f = stablehlo.reshape %dac3 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %dbn3z = stablehlo.constant dense<0.0> : tensor<128x4096xf32>
    %dbn3m = stablehlo.compare GT, %bn3, %dbn3z : (tensor<128x4096xf32>, tensor<128x4096xf32>) -> tensor<128x4096xi1>
    %dbn3 = stablehlo.select %dbn3m, %dac3f, %dbn3z : tensor<128x4096xi1>, tensor<128x4096xf32>
    %dhc3f_dyr = stablehlo.reshape %dbn3 : (tensor<128x4096xf32>) -> tensor<128x16x256xf32>
    %dhc3f_gb = stablehlo.broadcast_in_dim %g3, dims = [1] : (tensor<16xf32>) -> tensor<128x16x256xf32>
    %dhc3f_dxh = stablehlo.multiply %dhc3f_gb, %dhc3f_dyr : tensor<128x16x256xf32>
    %dhc3f_sdxr = stablehlo.reduce(%dhc3f_dxh init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<128x16x256xf32>, tensor<f32>) -> tensor<128x16xf32>
    %dhc3f_sdx = stablehlo.broadcast_in_dim %dhc3f_sdxr, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x256xf32>
    %dhc3f_xd = stablehlo.multiply %bn3_xhat, %dhc3f_dxh : tensor<128x16x256xf32>
    %dhc3f_sxdr = stablehlo.reduce(%dhc3f_xd init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<128x16x256xf32>, tensor<f32>) -> tensor<128x16xf32>
    %dhc3f_sxd = stablehlo.broadcast_in_dim %dhc3f_sxdr, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x256xf32>
    %dhc3f_t1 = stablehlo.multiply %dhc3f_dxh, %bn3_nf : tensor<128x16x256xf32>
    %dhc3f_i1 = stablehlo.subtract %dhc3f_t1, %dhc3f_sdx : tensor<128x16x256xf32>
    %dhc3f_xs = stablehlo.multiply %bn3_xhat, %dhc3f_sxd : tensor<128x16x256xf32>
    %dhc3f_i2 = stablehlo.subtract %dhc3f_i1, %dhc3f_xs : tensor<128x16x256xf32>
    %dhc3f_s = stablehlo.divide %bn3_istd, %bn3_nf : tensor<128x16x256xf32>
    %dhc3f_dx3 = stablehlo.multiply %dhc3f_s, %dhc3f_i2 : tensor<128x16x256xf32>
    %dhc3f = stablehlo.reshape %dhc3f_dx3 : (tensor<128x16x256xf32>) -> tensor<128x4096xf32>
    %dg3_dyr = stablehlo.reshape %dbn3 : (tensor<128x4096xf32>) -> tensor<128x16x256xf32>
    %dg3_p = stablehlo.multiply %dg3_dyr, %bn3_xhat : tensor<128x16x256xf32>
    %dg3 = stablehlo.reduce(%dg3_p init: %sc) applies stablehlo.add across dimensions = [0, 2] : (tensor<128x16x256xf32>, tensor<f32>) -> tensor<16xf32>
    %dbt3 = stablehlo.reduce(%dg3_dyr init: %sc) applies stablehlo.add across dimensions = [0, 2] : (tensor<128x16x256xf32>, tensor<f32>) -> tensor<16xf32>
    %dhc3 = stablehlo.reshape %dhc3f : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
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
    %dac2f = stablehlo.reshape %dac2 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %dbn2z = stablehlo.constant dense<0.0> : tensor<128x16384xf32>
    %dbn2m = stablehlo.compare GT, %bn2, %dbn2z : (tensor<128x16384xf32>, tensor<128x16384xf32>) -> tensor<128x16384xi1>
    %dbn2 = stablehlo.select %dbn2m, %dac2f, %dbn2z : tensor<128x16384xi1>, tensor<128x16384xf32>
    %dhc2f_dyr = stablehlo.reshape %dbn2 : (tensor<128x16384xf32>) -> tensor<128x16x1024xf32>
    %dhc2f_gb = stablehlo.broadcast_in_dim %g2, dims = [1] : (tensor<16xf32>) -> tensor<128x16x1024xf32>
    %dhc2f_dxh = stablehlo.multiply %dhc2f_gb, %dhc2f_dyr : tensor<128x16x1024xf32>
    %dhc2f_sdxr = stablehlo.reduce(%dhc2f_dxh init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<128x16x1024xf32>, tensor<f32>) -> tensor<128x16xf32>
    %dhc2f_sdx = stablehlo.broadcast_in_dim %dhc2f_sdxr, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x1024xf32>
    %dhc2f_xd = stablehlo.multiply %bn2_xhat, %dhc2f_dxh : tensor<128x16x1024xf32>
    %dhc2f_sxdr = stablehlo.reduce(%dhc2f_xd init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<128x16x1024xf32>, tensor<f32>) -> tensor<128x16xf32>
    %dhc2f_sxd = stablehlo.broadcast_in_dim %dhc2f_sxdr, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x1024xf32>
    %dhc2f_t1 = stablehlo.multiply %dhc2f_dxh, %bn2_nf : tensor<128x16x1024xf32>
    %dhc2f_i1 = stablehlo.subtract %dhc2f_t1, %dhc2f_sdx : tensor<128x16x1024xf32>
    %dhc2f_xs = stablehlo.multiply %bn2_xhat, %dhc2f_sxd : tensor<128x16x1024xf32>
    %dhc2f_i2 = stablehlo.subtract %dhc2f_i1, %dhc2f_xs : tensor<128x16x1024xf32>
    %dhc2f_s = stablehlo.divide %bn2_istd, %bn2_nf : tensor<128x16x1024xf32>
    %dhc2f_dx3 = stablehlo.multiply %dhc2f_s, %dhc2f_i2 : tensor<128x16x1024xf32>
    %dhc2f = stablehlo.reshape %dhc2f_dx3 : (tensor<128x16x1024xf32>) -> tensor<128x16384xf32>
    %dg2_dyr = stablehlo.reshape %dbn2 : (tensor<128x16384xf32>) -> tensor<128x16x1024xf32>
    %dg2_p = stablehlo.multiply %dg2_dyr, %bn2_xhat : tensor<128x16x1024xf32>
    %dg2 = stablehlo.reduce(%dg2_p init: %sc) applies stablehlo.add across dimensions = [0, 2] : (tensor<128x16x1024xf32>, tensor<f32>) -> tensor<16xf32>
    %dbt2 = stablehlo.reduce(%dg2_dyr init: %sc) applies stablehlo.add across dimensions = [0, 2] : (tensor<128x16x1024xf32>, tensor<f32>) -> tensor<16xf32>
    %dhc2 = stablehlo.reshape %dhc2f : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %dac1t = stablehlo.transpose %W2, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %dac1r = stablehlo.reverse %dac1t, dims = [2, 3] : tensor<16x16x3x3xf32>
    %dac1 = stablehlo.convolution(%dhc2, %dac1r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x32x32xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x32x32xf32>
    %dac1f = stablehlo.reshape %dac1 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %dbn1z = stablehlo.constant dense<0.0> : tensor<128x16384xf32>
    %dbn1m = stablehlo.compare GT, %bn1, %dbn1z : (tensor<128x16384xf32>, tensor<128x16384xf32>) -> tensor<128x16384xi1>
    %dbn1 = stablehlo.select %dbn1m, %dac1f, %dbn1z : tensor<128x16384xi1>, tensor<128x16384xf32>
    %dhc1f_dyr = stablehlo.reshape %dbn1 : (tensor<128x16384xf32>) -> tensor<128x16x1024xf32>
    %dhc1f_gb = stablehlo.broadcast_in_dim %g1, dims = [1] : (tensor<16xf32>) -> tensor<128x16x1024xf32>
    %dhc1f_dxh = stablehlo.multiply %dhc1f_gb, %dhc1f_dyr : tensor<128x16x1024xf32>
    %dhc1f_sdxr = stablehlo.reduce(%dhc1f_dxh init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<128x16x1024xf32>, tensor<f32>) -> tensor<128x16xf32>
    %dhc1f_sdx = stablehlo.broadcast_in_dim %dhc1f_sdxr, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x1024xf32>
    %dhc1f_xd = stablehlo.multiply %bn1_xhat, %dhc1f_dxh : tensor<128x16x1024xf32>
    %dhc1f_sxdr = stablehlo.reduce(%dhc1f_xd init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<128x16x1024xf32>, tensor<f32>) -> tensor<128x16xf32>
    %dhc1f_sxd = stablehlo.broadcast_in_dim %dhc1f_sxdr, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x1024xf32>
    %dhc1f_t1 = stablehlo.multiply %dhc1f_dxh, %bn1_nf : tensor<128x16x1024xf32>
    %dhc1f_i1 = stablehlo.subtract %dhc1f_t1, %dhc1f_sdx : tensor<128x16x1024xf32>
    %dhc1f_xs = stablehlo.multiply %bn1_xhat, %dhc1f_sxd : tensor<128x16x1024xf32>
    %dhc1f_i2 = stablehlo.subtract %dhc1f_i1, %dhc1f_xs : tensor<128x16x1024xf32>
    %dhc1f_s = stablehlo.divide %bn1_istd, %bn1_nf : tensor<128x16x1024xf32>
    %dhc1f_dx3 = stablehlo.multiply %dhc1f_s, %dhc1f_i2 : tensor<128x16x1024xf32>
    %dhc1f = stablehlo.reshape %dhc1f_dx3 : (tensor<128x16x1024xf32>) -> tensor<128x16384xf32>
    %dg1_dyr = stablehlo.reshape %dbn1 : (tensor<128x16384xf32>) -> tensor<128x16x1024xf32>
    %dg1_p = stablehlo.multiply %dg1_dyr, %bn1_xhat : tensor<128x16x1024xf32>
    %dg1 = stablehlo.reduce(%dg1_p init: %sc) applies stablehlo.add across dimensions = [0, 2] : (tensor<128x16x1024xf32>, tensor<f32>) -> tensor<16xf32>
    %dbt1 = stablehlo.reduce(%dg1_dyr init: %sc) applies stablehlo.add across dimensions = [0, 2] : (tensor<128x16x1024xf32>, tensor<f32>) -> tensor<16xf32>
    %dhc1 = stablehlo.reshape %dhc1f : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
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
    %adb1g1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adob1g1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %admsg1 = stablehlo.multiply %adb1g1, %g1m : tensor<16xf32>
    %admgg1 = stablehlo.multiply %adob1g1, %dg1 : tensor<16xf32>
    %admng1 = stablehlo.add %admsg1, %admgg1 : tensor<16xf32>
    %adb2g1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adob2g1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %advsg1 = stablehlo.multiply %adb2g1, %g1v : tensor<16xf32>
    %adg2g1 = stablehlo.multiply %dg1, %dg1 : tensor<16xf32>
    %advgg1 = stablehlo.multiply %adob2g1, %adg2g1 : tensor<16xf32>
    %advng1 = stablehlo.add %advsg1, %advgg1 : tensor<16xf32>
    %adbc1g1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adbc2g1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %admhg1 = stablehlo.divide %admng1, %adbc1g1 : tensor<16xf32>
    %advhg1 = stablehlo.divide %advng1, %adbc2g1 : tensor<16xf32>
    %adlrg1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adepsg1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adsqg1 = stablehlo.sqrt %advhg1 : tensor<16xf32>
    %addeng1 = stablehlo.add %adsqg1, %adepsg1 : tensor<16xf32>
    %adratg1 = stablehlo.divide %admhg1, %addeng1 : tensor<16xf32>
    %adstg1 = stablehlo.multiply %adlrg1, %adratg1 : tensor<16xf32>
    %adsubg1 = stablehlo.subtract %g1, %adstg1 : tensor<16xf32>
    %adwdg1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adwdlrg1 = stablehlo.multiply %adwdg1, %adlrg1 : tensor<16xf32>
    %adwdpg1 = stablehlo.multiply %adwdlrg1, %g1 : tensor<16xf32>
    %adnewg1 = stablehlo.subtract %adsubg1, %adwdpg1 : tensor<16xf32>
    %adb1bt1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adob1bt1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %admsbt1 = stablehlo.multiply %adb1bt1, %bt1m : tensor<16xf32>
    %admgbt1 = stablehlo.multiply %adob1bt1, %dbt1 : tensor<16xf32>
    %admnbt1 = stablehlo.add %admsbt1, %admgbt1 : tensor<16xf32>
    %adb2bt1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adob2bt1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %advsbt1 = stablehlo.multiply %adb2bt1, %bt1v : tensor<16xf32>
    %adg2bt1 = stablehlo.multiply %dbt1, %dbt1 : tensor<16xf32>
    %advgbt1 = stablehlo.multiply %adob2bt1, %adg2bt1 : tensor<16xf32>
    %advnbt1 = stablehlo.add %advsbt1, %advgbt1 : tensor<16xf32>
    %adbc1bt1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adbc2bt1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %admhbt1 = stablehlo.divide %admnbt1, %adbc1bt1 : tensor<16xf32>
    %advhbt1 = stablehlo.divide %advnbt1, %adbc2bt1 : tensor<16xf32>
    %adlrbt1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adepsbt1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adsqbt1 = stablehlo.sqrt %advhbt1 : tensor<16xf32>
    %addenbt1 = stablehlo.add %adsqbt1, %adepsbt1 : tensor<16xf32>
    %adratbt1 = stablehlo.divide %admhbt1, %addenbt1 : tensor<16xf32>
    %adstbt1 = stablehlo.multiply %adlrbt1, %adratbt1 : tensor<16xf32>
    %adsubbt1 = stablehlo.subtract %bt1, %adstbt1 : tensor<16xf32>
    %adwdbt1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adwdlrbt1 = stablehlo.multiply %adwdbt1, %adlrbt1 : tensor<16xf32>
    %adwdpbt1 = stablehlo.multiply %adwdlrbt1, %bt1 : tensor<16xf32>
    %adnewbt1 = stablehlo.subtract %adsubbt1, %adwdpbt1 : tensor<16xf32>
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
    %adb1g2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adob1g2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %admsg2 = stablehlo.multiply %adb1g2, %g2m : tensor<16xf32>
    %admgg2 = stablehlo.multiply %adob1g2, %dg2 : tensor<16xf32>
    %admng2 = stablehlo.add %admsg2, %admgg2 : tensor<16xf32>
    %adb2g2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adob2g2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %advsg2 = stablehlo.multiply %adb2g2, %g2v : tensor<16xf32>
    %adg2g2 = stablehlo.multiply %dg2, %dg2 : tensor<16xf32>
    %advgg2 = stablehlo.multiply %adob2g2, %adg2g2 : tensor<16xf32>
    %advng2 = stablehlo.add %advsg2, %advgg2 : tensor<16xf32>
    %adbc1g2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adbc2g2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %admhg2 = stablehlo.divide %admng2, %adbc1g2 : tensor<16xf32>
    %advhg2 = stablehlo.divide %advng2, %adbc2g2 : tensor<16xf32>
    %adlrg2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adepsg2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adsqg2 = stablehlo.sqrt %advhg2 : tensor<16xf32>
    %addeng2 = stablehlo.add %adsqg2, %adepsg2 : tensor<16xf32>
    %adratg2 = stablehlo.divide %admhg2, %addeng2 : tensor<16xf32>
    %adstg2 = stablehlo.multiply %adlrg2, %adratg2 : tensor<16xf32>
    %adsubg2 = stablehlo.subtract %g2, %adstg2 : tensor<16xf32>
    %adwdg2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adwdlrg2 = stablehlo.multiply %adwdg2, %adlrg2 : tensor<16xf32>
    %adwdpg2 = stablehlo.multiply %adwdlrg2, %g2 : tensor<16xf32>
    %adnewg2 = stablehlo.subtract %adsubg2, %adwdpg2 : tensor<16xf32>
    %adb1bt2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adob1bt2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %admsbt2 = stablehlo.multiply %adb1bt2, %bt2m : tensor<16xf32>
    %admgbt2 = stablehlo.multiply %adob1bt2, %dbt2 : tensor<16xf32>
    %admnbt2 = stablehlo.add %admsbt2, %admgbt2 : tensor<16xf32>
    %adb2bt2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adob2bt2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %advsbt2 = stablehlo.multiply %adb2bt2, %bt2v : tensor<16xf32>
    %adg2bt2 = stablehlo.multiply %dbt2, %dbt2 : tensor<16xf32>
    %advgbt2 = stablehlo.multiply %adob2bt2, %adg2bt2 : tensor<16xf32>
    %advnbt2 = stablehlo.add %advsbt2, %advgbt2 : tensor<16xf32>
    %adbc1bt2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adbc2bt2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %admhbt2 = stablehlo.divide %admnbt2, %adbc1bt2 : tensor<16xf32>
    %advhbt2 = stablehlo.divide %advnbt2, %adbc2bt2 : tensor<16xf32>
    %adlrbt2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adepsbt2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adsqbt2 = stablehlo.sqrt %advhbt2 : tensor<16xf32>
    %addenbt2 = stablehlo.add %adsqbt2, %adepsbt2 : tensor<16xf32>
    %adratbt2 = stablehlo.divide %admhbt2, %addenbt2 : tensor<16xf32>
    %adstbt2 = stablehlo.multiply %adlrbt2, %adratbt2 : tensor<16xf32>
    %adsubbt2 = stablehlo.subtract %bt2, %adstbt2 : tensor<16xf32>
    %adwdbt2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adwdlrbt2 = stablehlo.multiply %adwdbt2, %adlrbt2 : tensor<16xf32>
    %adwdpbt2 = stablehlo.multiply %adwdlrbt2, %bt2 : tensor<16xf32>
    %adnewbt2 = stablehlo.subtract %adsubbt2, %adwdpbt2 : tensor<16xf32>
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
    %adb1g3 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adob1g3 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %admsg3 = stablehlo.multiply %adb1g3, %g3m : tensor<16xf32>
    %admgg3 = stablehlo.multiply %adob1g3, %dg3 : tensor<16xf32>
    %admng3 = stablehlo.add %admsg3, %admgg3 : tensor<16xf32>
    %adb2g3 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adob2g3 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %advsg3 = stablehlo.multiply %adb2g3, %g3v : tensor<16xf32>
    %adg2g3 = stablehlo.multiply %dg3, %dg3 : tensor<16xf32>
    %advgg3 = stablehlo.multiply %adob2g3, %adg2g3 : tensor<16xf32>
    %advng3 = stablehlo.add %advsg3, %advgg3 : tensor<16xf32>
    %adbc1g3 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adbc2g3 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %admhg3 = stablehlo.divide %admng3, %adbc1g3 : tensor<16xf32>
    %advhg3 = stablehlo.divide %advng3, %adbc2g3 : tensor<16xf32>
    %adlrg3 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adepsg3 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adsqg3 = stablehlo.sqrt %advhg3 : tensor<16xf32>
    %addeng3 = stablehlo.add %adsqg3, %adepsg3 : tensor<16xf32>
    %adratg3 = stablehlo.divide %admhg3, %addeng3 : tensor<16xf32>
    %adstg3 = stablehlo.multiply %adlrg3, %adratg3 : tensor<16xf32>
    %adsubg3 = stablehlo.subtract %g3, %adstg3 : tensor<16xf32>
    %adwdg3 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adwdlrg3 = stablehlo.multiply %adwdg3, %adlrg3 : tensor<16xf32>
    %adwdpg3 = stablehlo.multiply %adwdlrg3, %g3 : tensor<16xf32>
    %adnewg3 = stablehlo.subtract %adsubg3, %adwdpg3 : tensor<16xf32>
    %adb1bt3 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adob1bt3 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %admsbt3 = stablehlo.multiply %adb1bt3, %bt3m : tensor<16xf32>
    %admgbt3 = stablehlo.multiply %adob1bt3, %dbt3 : tensor<16xf32>
    %admnbt3 = stablehlo.add %admsbt3, %admgbt3 : tensor<16xf32>
    %adb2bt3 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adob2bt3 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %advsbt3 = stablehlo.multiply %adb2bt3, %bt3v : tensor<16xf32>
    %adg2bt3 = stablehlo.multiply %dbt3, %dbt3 : tensor<16xf32>
    %advgbt3 = stablehlo.multiply %adob2bt3, %adg2bt3 : tensor<16xf32>
    %advnbt3 = stablehlo.add %advsbt3, %advgbt3 : tensor<16xf32>
    %adbc1bt3 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adbc2bt3 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %admhbt3 = stablehlo.divide %admnbt3, %adbc1bt3 : tensor<16xf32>
    %advhbt3 = stablehlo.divide %advnbt3, %adbc2bt3 : tensor<16xf32>
    %adlrbt3 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adepsbt3 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adsqbt3 = stablehlo.sqrt %advhbt3 : tensor<16xf32>
    %addenbt3 = stablehlo.add %adsqbt3, %adepsbt3 : tensor<16xf32>
    %adratbt3 = stablehlo.divide %admhbt3, %addenbt3 : tensor<16xf32>
    %adstbt3 = stablehlo.multiply %adlrbt3, %adratbt3 : tensor<16xf32>
    %adsubbt3 = stablehlo.subtract %bt3, %adstbt3 : tensor<16xf32>
    %adwdbt3 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adwdlrbt3 = stablehlo.multiply %adwdbt3, %adlrbt3 : tensor<16xf32>
    %adwdpbt3 = stablehlo.multiply %adwdlrbt3, %bt3 : tensor<16xf32>
    %adnewbt3 = stablehlo.subtract %adsubbt3, %adwdpbt3 : tensor<16xf32>
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
    %adb1g4 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adob1g4 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %admsg4 = stablehlo.multiply %adb1g4, %g4m : tensor<16xf32>
    %admgg4 = stablehlo.multiply %adob1g4, %dg4 : tensor<16xf32>
    %admng4 = stablehlo.add %admsg4, %admgg4 : tensor<16xf32>
    %adb2g4 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adob2g4 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %advsg4 = stablehlo.multiply %adb2g4, %g4v : tensor<16xf32>
    %adg2g4 = stablehlo.multiply %dg4, %dg4 : tensor<16xf32>
    %advgg4 = stablehlo.multiply %adob2g4, %adg2g4 : tensor<16xf32>
    %advng4 = stablehlo.add %advsg4, %advgg4 : tensor<16xf32>
    %adbc1g4 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adbc2g4 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %admhg4 = stablehlo.divide %admng4, %adbc1g4 : tensor<16xf32>
    %advhg4 = stablehlo.divide %advng4, %adbc2g4 : tensor<16xf32>
    %adlrg4 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adepsg4 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adsqg4 = stablehlo.sqrt %advhg4 : tensor<16xf32>
    %addeng4 = stablehlo.add %adsqg4, %adepsg4 : tensor<16xf32>
    %adratg4 = stablehlo.divide %admhg4, %addeng4 : tensor<16xf32>
    %adstg4 = stablehlo.multiply %adlrg4, %adratg4 : tensor<16xf32>
    %adsubg4 = stablehlo.subtract %g4, %adstg4 : tensor<16xf32>
    %adwdg4 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adwdlrg4 = stablehlo.multiply %adwdg4, %adlrg4 : tensor<16xf32>
    %adwdpg4 = stablehlo.multiply %adwdlrg4, %g4 : tensor<16xf32>
    %adnewg4 = stablehlo.subtract %adsubg4, %adwdpg4 : tensor<16xf32>
    %adb1bt4 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adob1bt4 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %admsbt4 = stablehlo.multiply %adb1bt4, %bt4m : tensor<16xf32>
    %admgbt4 = stablehlo.multiply %adob1bt4, %dbt4 : tensor<16xf32>
    %admnbt4 = stablehlo.add %admsbt4, %admgbt4 : tensor<16xf32>
    %adb2bt4 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adob2bt4 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %advsbt4 = stablehlo.multiply %adb2bt4, %bt4v : tensor<16xf32>
    %adg2bt4 = stablehlo.multiply %dbt4, %dbt4 : tensor<16xf32>
    %advgbt4 = stablehlo.multiply %adob2bt4, %adg2bt4 : tensor<16xf32>
    %advnbt4 = stablehlo.add %advsbt4, %advgbt4 : tensor<16xf32>
    %adbc1bt4 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adbc2bt4 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %admhbt4 = stablehlo.divide %admnbt4, %adbc1bt4 : tensor<16xf32>
    %advhbt4 = stablehlo.divide %advnbt4, %adbc2bt4 : tensor<16xf32>
    %adlrbt4 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adepsbt4 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adsqbt4 = stablehlo.sqrt %advhbt4 : tensor<16xf32>
    %addenbt4 = stablehlo.add %adsqbt4, %adepsbt4 : tensor<16xf32>
    %adratbt4 = stablehlo.divide %admhbt4, %addenbt4 : tensor<16xf32>
    %adstbt4 = stablehlo.multiply %adlrbt4, %adratbt4 : tensor<16xf32>
    %adsubbt4 = stablehlo.subtract %bt4, %adstbt4 : tensor<16xf32>
    %adwdbt4 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %adwdlrbt4 = stablehlo.multiply %adwdbt4, %adlrbt4 : tensor<16xf32>
    %adwdpbt4 = stablehlo.multiply %adwdlrbt4, %bt4 : tensor<16xf32>
    %adnewbt4 = stablehlo.subtract %adsubbt4, %adwdpbt4 : tensor<16xf32>
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
    %adb1g5 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adob1g5 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %admsg5 = stablehlo.multiply %adb1g5, %g5m : tensor<32xf32>
    %admgg5 = stablehlo.multiply %adob1g5, %dg5 : tensor<32xf32>
    %admng5 = stablehlo.add %admsg5, %admgg5 : tensor<32xf32>
    %adb2g5 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adob2g5 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %advsg5 = stablehlo.multiply %adb2g5, %g5v : tensor<32xf32>
    %adg2g5 = stablehlo.multiply %dg5, %dg5 : tensor<32xf32>
    %advgg5 = stablehlo.multiply %adob2g5, %adg2g5 : tensor<32xf32>
    %advng5 = stablehlo.add %advsg5, %advgg5 : tensor<32xf32>
    %adbc1g5 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adbc2g5 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %admhg5 = stablehlo.divide %admng5, %adbc1g5 : tensor<32xf32>
    %advhg5 = stablehlo.divide %advng5, %adbc2g5 : tensor<32xf32>
    %adlrg5 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adepsg5 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adsqg5 = stablehlo.sqrt %advhg5 : tensor<32xf32>
    %addeng5 = stablehlo.add %adsqg5, %adepsg5 : tensor<32xf32>
    %adratg5 = stablehlo.divide %admhg5, %addeng5 : tensor<32xf32>
    %adstg5 = stablehlo.multiply %adlrg5, %adratg5 : tensor<32xf32>
    %adsubg5 = stablehlo.subtract %g5, %adstg5 : tensor<32xf32>
    %adwdg5 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adwdlrg5 = stablehlo.multiply %adwdg5, %adlrg5 : tensor<32xf32>
    %adwdpg5 = stablehlo.multiply %adwdlrg5, %g5 : tensor<32xf32>
    %adnewg5 = stablehlo.subtract %adsubg5, %adwdpg5 : tensor<32xf32>
    %adb1bt5 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adob1bt5 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %admsbt5 = stablehlo.multiply %adb1bt5, %bt5m : tensor<32xf32>
    %admgbt5 = stablehlo.multiply %adob1bt5, %dbt5 : tensor<32xf32>
    %admnbt5 = stablehlo.add %admsbt5, %admgbt5 : tensor<32xf32>
    %adb2bt5 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adob2bt5 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %advsbt5 = stablehlo.multiply %adb2bt5, %bt5v : tensor<32xf32>
    %adg2bt5 = stablehlo.multiply %dbt5, %dbt5 : tensor<32xf32>
    %advgbt5 = stablehlo.multiply %adob2bt5, %adg2bt5 : tensor<32xf32>
    %advnbt5 = stablehlo.add %advsbt5, %advgbt5 : tensor<32xf32>
    %adbc1bt5 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adbc2bt5 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %admhbt5 = stablehlo.divide %admnbt5, %adbc1bt5 : tensor<32xf32>
    %advhbt5 = stablehlo.divide %advnbt5, %adbc2bt5 : tensor<32xf32>
    %adlrbt5 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adepsbt5 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adsqbt5 = stablehlo.sqrt %advhbt5 : tensor<32xf32>
    %addenbt5 = stablehlo.add %adsqbt5, %adepsbt5 : tensor<32xf32>
    %adratbt5 = stablehlo.divide %admhbt5, %addenbt5 : tensor<32xf32>
    %adstbt5 = stablehlo.multiply %adlrbt5, %adratbt5 : tensor<32xf32>
    %adsubbt5 = stablehlo.subtract %bt5, %adstbt5 : tensor<32xf32>
    %adwdbt5 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adwdlrbt5 = stablehlo.multiply %adwdbt5, %adlrbt5 : tensor<32xf32>
    %adwdpbt5 = stablehlo.multiply %adwdlrbt5, %bt5 : tensor<32xf32>
    %adnewbt5 = stablehlo.subtract %adsubbt5, %adwdpbt5 : tensor<32xf32>
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
    %adb1g6 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adob1g6 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %admsg6 = stablehlo.multiply %adb1g6, %g6m : tensor<32xf32>
    %admgg6 = stablehlo.multiply %adob1g6, %dg6 : tensor<32xf32>
    %admng6 = stablehlo.add %admsg6, %admgg6 : tensor<32xf32>
    %adb2g6 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adob2g6 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %advsg6 = stablehlo.multiply %adb2g6, %g6v : tensor<32xf32>
    %adg2g6 = stablehlo.multiply %dg6, %dg6 : tensor<32xf32>
    %advgg6 = stablehlo.multiply %adob2g6, %adg2g6 : tensor<32xf32>
    %advng6 = stablehlo.add %advsg6, %advgg6 : tensor<32xf32>
    %adbc1g6 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adbc2g6 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %admhg6 = stablehlo.divide %admng6, %adbc1g6 : tensor<32xf32>
    %advhg6 = stablehlo.divide %advng6, %adbc2g6 : tensor<32xf32>
    %adlrg6 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adepsg6 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adsqg6 = stablehlo.sqrt %advhg6 : tensor<32xf32>
    %addeng6 = stablehlo.add %adsqg6, %adepsg6 : tensor<32xf32>
    %adratg6 = stablehlo.divide %admhg6, %addeng6 : tensor<32xf32>
    %adstg6 = stablehlo.multiply %adlrg6, %adratg6 : tensor<32xf32>
    %adsubg6 = stablehlo.subtract %g6, %adstg6 : tensor<32xf32>
    %adwdg6 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adwdlrg6 = stablehlo.multiply %adwdg6, %adlrg6 : tensor<32xf32>
    %adwdpg6 = stablehlo.multiply %adwdlrg6, %g6 : tensor<32xf32>
    %adnewg6 = stablehlo.subtract %adsubg6, %adwdpg6 : tensor<32xf32>
    %adb1bt6 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adob1bt6 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %admsbt6 = stablehlo.multiply %adb1bt6, %bt6m : tensor<32xf32>
    %admgbt6 = stablehlo.multiply %adob1bt6, %dbt6 : tensor<32xf32>
    %admnbt6 = stablehlo.add %admsbt6, %admgbt6 : tensor<32xf32>
    %adb2bt6 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adob2bt6 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %advsbt6 = stablehlo.multiply %adb2bt6, %bt6v : tensor<32xf32>
    %adg2bt6 = stablehlo.multiply %dbt6, %dbt6 : tensor<32xf32>
    %advgbt6 = stablehlo.multiply %adob2bt6, %adg2bt6 : tensor<32xf32>
    %advnbt6 = stablehlo.add %advsbt6, %advgbt6 : tensor<32xf32>
    %adbc1bt6 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adbc2bt6 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %admhbt6 = stablehlo.divide %admnbt6, %adbc1bt6 : tensor<32xf32>
    %advhbt6 = stablehlo.divide %advnbt6, %adbc2bt6 : tensor<32xf32>
    %adlrbt6 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adepsbt6 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adsqbt6 = stablehlo.sqrt %advhbt6 : tensor<32xf32>
    %addenbt6 = stablehlo.add %adsqbt6, %adepsbt6 : tensor<32xf32>
    %adratbt6 = stablehlo.divide %admhbt6, %addenbt6 : tensor<32xf32>
    %adstbt6 = stablehlo.multiply %adlrbt6, %adratbt6 : tensor<32xf32>
    %adsubbt6 = stablehlo.subtract %bt6, %adstbt6 : tensor<32xf32>
    %adwdbt6 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adwdlrbt6 = stablehlo.multiply %adwdbt6, %adlrbt6 : tensor<32xf32>
    %adwdpbt6 = stablehlo.multiply %adwdlrbt6, %bt6 : tensor<32xf32>
    %adnewbt6 = stablehlo.subtract %adsubbt6, %adwdpbt6 : tensor<32xf32>
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
    %adb1g7 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adob1g7 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %admsg7 = stablehlo.multiply %adb1g7, %g7m : tensor<32xf32>
    %admgg7 = stablehlo.multiply %adob1g7, %dg7 : tensor<32xf32>
    %admng7 = stablehlo.add %admsg7, %admgg7 : tensor<32xf32>
    %adb2g7 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adob2g7 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %advsg7 = stablehlo.multiply %adb2g7, %g7v : tensor<32xf32>
    %adg2g7 = stablehlo.multiply %dg7, %dg7 : tensor<32xf32>
    %advgg7 = stablehlo.multiply %adob2g7, %adg2g7 : tensor<32xf32>
    %advng7 = stablehlo.add %advsg7, %advgg7 : tensor<32xf32>
    %adbc1g7 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adbc2g7 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %admhg7 = stablehlo.divide %admng7, %adbc1g7 : tensor<32xf32>
    %advhg7 = stablehlo.divide %advng7, %adbc2g7 : tensor<32xf32>
    %adlrg7 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adepsg7 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adsqg7 = stablehlo.sqrt %advhg7 : tensor<32xf32>
    %addeng7 = stablehlo.add %adsqg7, %adepsg7 : tensor<32xf32>
    %adratg7 = stablehlo.divide %admhg7, %addeng7 : tensor<32xf32>
    %adstg7 = stablehlo.multiply %adlrg7, %adratg7 : tensor<32xf32>
    %adsubg7 = stablehlo.subtract %g7, %adstg7 : tensor<32xf32>
    %adwdg7 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adwdlrg7 = stablehlo.multiply %adwdg7, %adlrg7 : tensor<32xf32>
    %adwdpg7 = stablehlo.multiply %adwdlrg7, %g7 : tensor<32xf32>
    %adnewg7 = stablehlo.subtract %adsubg7, %adwdpg7 : tensor<32xf32>
    %adb1bt7 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adob1bt7 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %admsbt7 = stablehlo.multiply %adb1bt7, %bt7m : tensor<32xf32>
    %admgbt7 = stablehlo.multiply %adob1bt7, %dbt7 : tensor<32xf32>
    %admnbt7 = stablehlo.add %admsbt7, %admgbt7 : tensor<32xf32>
    %adb2bt7 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adob2bt7 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %advsbt7 = stablehlo.multiply %adb2bt7, %bt7v : tensor<32xf32>
    %adg2bt7 = stablehlo.multiply %dbt7, %dbt7 : tensor<32xf32>
    %advgbt7 = stablehlo.multiply %adob2bt7, %adg2bt7 : tensor<32xf32>
    %advnbt7 = stablehlo.add %advsbt7, %advgbt7 : tensor<32xf32>
    %adbc1bt7 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adbc2bt7 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %admhbt7 = stablehlo.divide %admnbt7, %adbc1bt7 : tensor<32xf32>
    %advhbt7 = stablehlo.divide %advnbt7, %adbc2bt7 : tensor<32xf32>
    %adlrbt7 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adepsbt7 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adsqbt7 = stablehlo.sqrt %advhbt7 : tensor<32xf32>
    %addenbt7 = stablehlo.add %adsqbt7, %adepsbt7 : tensor<32xf32>
    %adratbt7 = stablehlo.divide %admhbt7, %addenbt7 : tensor<32xf32>
    %adstbt7 = stablehlo.multiply %adlrbt7, %adratbt7 : tensor<32xf32>
    %adsubbt7 = stablehlo.subtract %bt7, %adstbt7 : tensor<32xf32>
    %adwdbt7 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adwdlrbt7 = stablehlo.multiply %adwdbt7, %adlrbt7 : tensor<32xf32>
    %adwdpbt7 = stablehlo.multiply %adwdlrbt7, %bt7 : tensor<32xf32>
    %adnewbt7 = stablehlo.subtract %adsubbt7, %adwdpbt7 : tensor<32xf32>
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
    %adb1g8 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adob1g8 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %admsg8 = stablehlo.multiply %adb1g8, %g8m : tensor<32xf32>
    %admgg8 = stablehlo.multiply %adob1g8, %dg8 : tensor<32xf32>
    %admng8 = stablehlo.add %admsg8, %admgg8 : tensor<32xf32>
    %adb2g8 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adob2g8 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %advsg8 = stablehlo.multiply %adb2g8, %g8v : tensor<32xf32>
    %adg2g8 = stablehlo.multiply %dg8, %dg8 : tensor<32xf32>
    %advgg8 = stablehlo.multiply %adob2g8, %adg2g8 : tensor<32xf32>
    %advng8 = stablehlo.add %advsg8, %advgg8 : tensor<32xf32>
    %adbc1g8 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adbc2g8 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %admhg8 = stablehlo.divide %admng8, %adbc1g8 : tensor<32xf32>
    %advhg8 = stablehlo.divide %advng8, %adbc2g8 : tensor<32xf32>
    %adlrg8 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adepsg8 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adsqg8 = stablehlo.sqrt %advhg8 : tensor<32xf32>
    %addeng8 = stablehlo.add %adsqg8, %adepsg8 : tensor<32xf32>
    %adratg8 = stablehlo.divide %admhg8, %addeng8 : tensor<32xf32>
    %adstg8 = stablehlo.multiply %adlrg8, %adratg8 : tensor<32xf32>
    %adsubg8 = stablehlo.subtract %g8, %adstg8 : tensor<32xf32>
    %adwdg8 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adwdlrg8 = stablehlo.multiply %adwdg8, %adlrg8 : tensor<32xf32>
    %adwdpg8 = stablehlo.multiply %adwdlrg8, %g8 : tensor<32xf32>
    %adnewg8 = stablehlo.subtract %adsubg8, %adwdpg8 : tensor<32xf32>
    %adb1bt8 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adob1bt8 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %admsbt8 = stablehlo.multiply %adb1bt8, %bt8m : tensor<32xf32>
    %admgbt8 = stablehlo.multiply %adob1bt8, %dbt8 : tensor<32xf32>
    %admnbt8 = stablehlo.add %admsbt8, %admgbt8 : tensor<32xf32>
    %adb2bt8 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adob2bt8 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %advsbt8 = stablehlo.multiply %adb2bt8, %bt8v : tensor<32xf32>
    %adg2bt8 = stablehlo.multiply %dbt8, %dbt8 : tensor<32xf32>
    %advgbt8 = stablehlo.multiply %adob2bt8, %adg2bt8 : tensor<32xf32>
    %advnbt8 = stablehlo.add %advsbt8, %advgbt8 : tensor<32xf32>
    %adbc1bt8 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adbc2bt8 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %admhbt8 = stablehlo.divide %admnbt8, %adbc1bt8 : tensor<32xf32>
    %advhbt8 = stablehlo.divide %advnbt8, %adbc2bt8 : tensor<32xf32>
    %adlrbt8 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adepsbt8 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adsqbt8 = stablehlo.sqrt %advhbt8 : tensor<32xf32>
    %addenbt8 = stablehlo.add %adsqbt8, %adepsbt8 : tensor<32xf32>
    %adratbt8 = stablehlo.divide %admhbt8, %addenbt8 : tensor<32xf32>
    %adstbt8 = stablehlo.multiply %adlrbt8, %adratbt8 : tensor<32xf32>
    %adsubbt8 = stablehlo.subtract %bt8, %adstbt8 : tensor<32xf32>
    %adwdbt8 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %adwdlrbt8 = stablehlo.multiply %adwdbt8, %adlrbt8 : tensor<32xf32>
    %adwdpbt8 = stablehlo.multiply %adwdlrbt8, %bt8 : tensor<32xf32>
    %adnewbt8 = stablehlo.subtract %adsubbt8, %adwdpbt8 : tensor<32xf32>
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
    return %adnewW1, %adnewcb1, %adnewg1, %adnewbt1, %adnewW2, %adnewcb2, %adnewg2, %adnewbt2, %adnewW3, %adnewcb3, %adnewg3, %adnewbt3, %adnewW4, %adnewcb4, %adnewg4, %adnewbt4, %adnewW5, %adnewcb5, %adnewg5, %adnewbt5, %adnewW6, %adnewcb6, %adnewg6, %adnewbt6, %adnewW7, %adnewcb7, %adnewg7, %adnewbt7, %adnewW8, %adnewcb8, %adnewg8, %adnewbt8, %adnewW9, %adnewb9, %adnewWa, %adnewba, %adnewWb, %adnewbb, %admnW1, %admncb1, %admng1, %admnbt1, %admnW2, %admncb2, %admng2, %admnbt2, %admnW3, %admncb3, %admng3, %admnbt3, %admnW4, %admncb4, %admng4, %admnbt4, %admnW5, %admncb5, %admng5, %admnbt5, %admnW6, %admncb6, %admng6, %admnbt6, %admnW7, %admncb7, %admng7, %admnbt7, %admnW8, %admncb8, %admng8, %admnbt8, %admnW9, %admnb9, %admnWa, %admnba, %admnWb, %admnbb, %advnW1, %advncb1, %advng1, %advnbt1, %advnW2, %advncb2, %advng2, %advnbt2, %advnW3, %advncb3, %advng3, %advnbt3, %advnW4, %advncb4, %advng4, %advnbt4, %advnW5, %advncb5, %advng5, %advnbt5, %advnW6, %advncb6, %advng6, %advnbt6, %advnW7, %advncb7, %advng7, %advnbt7, %advnW8, %advncb8, %advng8, %advnbt8, %advnW9, %advnb9, %advnWa, %advnba, %advnWb, %advnbb, %loss, %bc1, %bc2 : tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>
  }
}

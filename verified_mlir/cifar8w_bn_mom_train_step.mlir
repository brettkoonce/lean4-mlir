module @m {
  func.func @cifar8w_bn_mom_train_step(%x: tensor<128x3072xf32>, %W1: tensor<16x3x3x3xf32>, %cb1: tensor<16xf32>, %g1: tensor<16xf32>, %bt1: tensor<16xf32>, %W2: tensor<16x16x3x3xf32>, %cb2: tensor<16xf32>, %g2: tensor<16xf32>, %bt2: tensor<16xf32>, %W3: tensor<16x16x3x3xf32>, %cb3: tensor<16xf32>, %g3: tensor<16xf32>, %bt3: tensor<16xf32>, %W4: tensor<16x16x3x3xf32>, %cb4: tensor<16xf32>, %g4: tensor<16xf32>, %bt4: tensor<16xf32>, %W5: tensor<32x16x3x3xf32>, %cb5: tensor<32xf32>, %g5: tensor<32xf32>, %bt5: tensor<32xf32>, %W6: tensor<32x32x3x3xf32>, %cb6: tensor<32xf32>, %g6: tensor<32xf32>, %bt6: tensor<32xf32>, %W7: tensor<32x32x3x3xf32>, %cb7: tensor<32xf32>, %g7: tensor<32xf32>, %bt7: tensor<32xf32>, %W8: tensor<32x32x3x3xf32>, %cb8: tensor<32xf32>, %g8: tensor<32xf32>, %bt8: tensor<32xf32>, %W9: tensor<128x512xf32>, %b9: tensor<512xf32>, %Wa: tensor<512x512xf32>, %ba: tensor<512xf32>, %Wb: tensor<512x10xf32>, %bb: tensor<10xf32>, %W1m: tensor<16x3x3x3xf32>, %cb1m: tensor<16xf32>, %g1m: tensor<16xf32>, %bt1m: tensor<16xf32>, %W2m: tensor<16x16x3x3xf32>, %cb2m: tensor<16xf32>, %g2m: tensor<16xf32>, %bt2m: tensor<16xf32>, %W3m: tensor<16x16x3x3xf32>, %cb3m: tensor<16xf32>, %g3m: tensor<16xf32>, %bt3m: tensor<16xf32>, %W4m: tensor<16x16x3x3xf32>, %cb4m: tensor<16xf32>, %g4m: tensor<16xf32>, %bt4m: tensor<16xf32>, %W5m: tensor<32x16x3x3xf32>, %cb5m: tensor<32xf32>, %g5m: tensor<32xf32>, %bt5m: tensor<32xf32>, %W6m: tensor<32x32x3x3xf32>, %cb6m: tensor<32xf32>, %g6m: tensor<32xf32>, %bt6m: tensor<32xf32>, %W7m: tensor<32x32x3x3xf32>, %cb7m: tensor<32xf32>, %g7m: tensor<32xf32>, %bt7m: tensor<32xf32>, %W8m: tensor<32x32x3x3xf32>, %cb8m: tensor<32xf32>, %g8m: tensor<32xf32>, %bt8m: tensor<32xf32>, %W9m: tensor<128x512xf32>, %b9m: tensor<512xf32>, %Wam: tensor<512x512xf32>, %bam: tensor<512xf32>, %Wbm: tensor<512x10xf32>, %bbm: tensor<10xf32>, %W1v: tensor<16x3x3x3xf32>, %cb1v: tensor<16xf32>, %g1v: tensor<16xf32>, %bt1v: tensor<16xf32>, %W2v: tensor<16x16x3x3xf32>, %cb2v: tensor<16xf32>, %g2v: tensor<16xf32>, %bt2v: tensor<16xf32>, %W3v: tensor<16x16x3x3xf32>, %cb3v: tensor<16xf32>, %g3v: tensor<16xf32>, %bt3v: tensor<16xf32>, %W4v: tensor<16x16x3x3xf32>, %cb4v: tensor<16xf32>, %g4v: tensor<16xf32>, %bt4v: tensor<16xf32>, %W5v: tensor<32x16x3x3xf32>, %cb5v: tensor<32xf32>, %g5v: tensor<32xf32>, %bt5v: tensor<32xf32>, %W6v: tensor<32x32x3x3xf32>, %cb6v: tensor<32xf32>, %g6v: tensor<32xf32>, %bt6v: tensor<32xf32>, %W7v: tensor<32x32x3x3xf32>, %cb7v: tensor<32xf32>, %g7v: tensor<32xf32>, %bt7v: tensor<32xf32>, %W8v: tensor<32x32x3x3xf32>, %cb8v: tensor<32xf32>, %g8v: tensor<32xf32>, %bt8v: tensor<32xf32>, %W9v: tensor<128x512xf32>, %b9v: tensor<512xf32>, %Wav: tensor<512x512xf32>, %bav: tensor<512xf32>, %Wbv: tensor<512x10xf32>, %bbv: tensor<10xf32>, %lr: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>, %onehot: tensor<128x10xf32>) -> (tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>) {
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
    %h9d = stablehlo.dot_general %flat, %W9, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x128xf32>, tensor<128x512xf32>) -> tensor<128x512xf32>
    %h9b = stablehlo.broadcast_in_dim %b9, dims = [1] : (tensor<512xf32>) -> tensor<128x512xf32>
    %h9 = stablehlo.add %h9d, %h9b : tensor<128x512xf32>
    %a9z = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %a9 = stablehlo.maximum %h9, %a9z : tensor<128x512xf32>
    %had = stablehlo.dot_general %a9, %Wa, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<512x512xf32>) -> tensor<128x512xf32>
    %hab = stablehlo.broadcast_in_dim %ba, dims = [1] : (tensor<512xf32>) -> tensor<128x512xf32>
    %ha = stablehlo.add %had, %hab : tensor<128x512xf32>
    %aaz = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %aa = stablehlo.maximum %ha, %aaz : tensor<128x512xf32>
    %logitsd = stablehlo.dot_general %aa, %Wb, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<512x10xf32>) -> tensor<128x10xf32>
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
    %dxb = stablehlo.dot_general %dy, %Wb, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x10xf32>, tensor<512x10xf32>) -> tensor<128x512xf32>
    %dyaz = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %dyam = stablehlo.compare GT, %ha, %dyaz : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<128x512xi1>
    %dya = stablehlo.select %dyam, %dxb, %dyaz : tensor<128x512xi1>, tensor<128x512xf32>
    %dxa = stablehlo.dot_general %dya, %Wa, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<512x512xf32>) -> tensor<128x512xf32>
    %dy9z = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %dy9m = stablehlo.compare GT, %h9, %dy9z : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<128x512xi1>
    %dy9 = stablehlo.select %dy9m, %dxa, %dy9z : tensor<128x512xi1>, tensor<128x512xf32>
    %dx9 = stablehlo.dot_general %dy9, %W9, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<128x128xf32>
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
    %dWb = stablehlo.dot_general %aa, %dy, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<128x10xf32>) -> tensor<512x10xf32>
    %dbb = stablehlo.reduce(%dy init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<128x10xf32>, tensor<f32>) -> tensor<10xf32>
    %dWa = stablehlo.dot_general %a9, %dya, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<512x512xf32>
    %dba = stablehlo.reduce(%dya init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<128x512xf32>, tensor<f32>) -> tensor<512xf32>
    %dW9 = stablehlo.dot_general %flat, %dy9, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x128xf32>, tensor<128x512xf32>) -> tensor<128x512xf32>
    %db9 = stablehlo.reduce(%dy9 init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<128x512xf32>, tensor<f32>) -> tensor<512xf32>
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
    %mommug1 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %momvgg1 = stablehlo.multiply %mommug1, %g1v : tensor<16xf32>
    %momvelg1 = stablehlo.add %momvgg1, %dg1 : tensor<16xf32>
    %momnvg1 = stablehlo.multiply %mommug1, %momvelg1 : tensor<16xf32>
    %momlkg1 = stablehlo.add %momnvg1, %dg1 : tensor<16xf32>
    %momlrg1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %momstg1 = stablehlo.multiply %momlrg1, %momlkg1 : tensor<16xf32>
    %momnewg1 = stablehlo.subtract %g1, %momstg1 : tensor<16xf32>
    %mommubt1 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %momvgbt1 = stablehlo.multiply %mommubt1, %bt1v : tensor<16xf32>
    %momvelbt1 = stablehlo.add %momvgbt1, %dbt1 : tensor<16xf32>
    %momnvbt1 = stablehlo.multiply %mommubt1, %momvelbt1 : tensor<16xf32>
    %momlkbt1 = stablehlo.add %momnvbt1, %dbt1 : tensor<16xf32>
    %momlrbt1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %momstbt1 = stablehlo.multiply %momlrbt1, %momlkbt1 : tensor<16xf32>
    %momnewbt1 = stablehlo.subtract %bt1, %momstbt1 : tensor<16xf32>
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
    %mommug2 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %momvgg2 = stablehlo.multiply %mommug2, %g2v : tensor<16xf32>
    %momvelg2 = stablehlo.add %momvgg2, %dg2 : tensor<16xf32>
    %momnvg2 = stablehlo.multiply %mommug2, %momvelg2 : tensor<16xf32>
    %momlkg2 = stablehlo.add %momnvg2, %dg2 : tensor<16xf32>
    %momlrg2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %momstg2 = stablehlo.multiply %momlrg2, %momlkg2 : tensor<16xf32>
    %momnewg2 = stablehlo.subtract %g2, %momstg2 : tensor<16xf32>
    %mommubt2 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %momvgbt2 = stablehlo.multiply %mommubt2, %bt2v : tensor<16xf32>
    %momvelbt2 = stablehlo.add %momvgbt2, %dbt2 : tensor<16xf32>
    %momnvbt2 = stablehlo.multiply %mommubt2, %momvelbt2 : tensor<16xf32>
    %momlkbt2 = stablehlo.add %momnvbt2, %dbt2 : tensor<16xf32>
    %momlrbt2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %momstbt2 = stablehlo.multiply %momlrbt2, %momlkbt2 : tensor<16xf32>
    %momnewbt2 = stablehlo.subtract %bt2, %momstbt2 : tensor<16xf32>
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
    %mommug3 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %momvgg3 = stablehlo.multiply %mommug3, %g3v : tensor<16xf32>
    %momvelg3 = stablehlo.add %momvgg3, %dg3 : tensor<16xf32>
    %momnvg3 = stablehlo.multiply %mommug3, %momvelg3 : tensor<16xf32>
    %momlkg3 = stablehlo.add %momnvg3, %dg3 : tensor<16xf32>
    %momlrg3 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %momstg3 = stablehlo.multiply %momlrg3, %momlkg3 : tensor<16xf32>
    %momnewg3 = stablehlo.subtract %g3, %momstg3 : tensor<16xf32>
    %mommubt3 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %momvgbt3 = stablehlo.multiply %mommubt3, %bt3v : tensor<16xf32>
    %momvelbt3 = stablehlo.add %momvgbt3, %dbt3 : tensor<16xf32>
    %momnvbt3 = stablehlo.multiply %mommubt3, %momvelbt3 : tensor<16xf32>
    %momlkbt3 = stablehlo.add %momnvbt3, %dbt3 : tensor<16xf32>
    %momlrbt3 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %momstbt3 = stablehlo.multiply %momlrbt3, %momlkbt3 : tensor<16xf32>
    %momnewbt3 = stablehlo.subtract %bt3, %momstbt3 : tensor<16xf32>
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
    %mommug4 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %momvgg4 = stablehlo.multiply %mommug4, %g4v : tensor<16xf32>
    %momvelg4 = stablehlo.add %momvgg4, %dg4 : tensor<16xf32>
    %momnvg4 = stablehlo.multiply %mommug4, %momvelg4 : tensor<16xf32>
    %momlkg4 = stablehlo.add %momnvg4, %dg4 : tensor<16xf32>
    %momlrg4 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %momstg4 = stablehlo.multiply %momlrg4, %momlkg4 : tensor<16xf32>
    %momnewg4 = stablehlo.subtract %g4, %momstg4 : tensor<16xf32>
    %mommubt4 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %momvgbt4 = stablehlo.multiply %mommubt4, %bt4v : tensor<16xf32>
    %momvelbt4 = stablehlo.add %momvgbt4, %dbt4 : tensor<16xf32>
    %momnvbt4 = stablehlo.multiply %mommubt4, %momvelbt4 : tensor<16xf32>
    %momlkbt4 = stablehlo.add %momnvbt4, %dbt4 : tensor<16xf32>
    %momlrbt4 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %momstbt4 = stablehlo.multiply %momlrbt4, %momlkbt4 : tensor<16xf32>
    %momnewbt4 = stablehlo.subtract %bt4, %momstbt4 : tensor<16xf32>
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
    %mommug5 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %momvgg5 = stablehlo.multiply %mommug5, %g5v : tensor<32xf32>
    %momvelg5 = stablehlo.add %momvgg5, %dg5 : tensor<32xf32>
    %momnvg5 = stablehlo.multiply %mommug5, %momvelg5 : tensor<32xf32>
    %momlkg5 = stablehlo.add %momnvg5, %dg5 : tensor<32xf32>
    %momlrg5 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %momstg5 = stablehlo.multiply %momlrg5, %momlkg5 : tensor<32xf32>
    %momnewg5 = stablehlo.subtract %g5, %momstg5 : tensor<32xf32>
    %mommubt5 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %momvgbt5 = stablehlo.multiply %mommubt5, %bt5v : tensor<32xf32>
    %momvelbt5 = stablehlo.add %momvgbt5, %dbt5 : tensor<32xf32>
    %momnvbt5 = stablehlo.multiply %mommubt5, %momvelbt5 : tensor<32xf32>
    %momlkbt5 = stablehlo.add %momnvbt5, %dbt5 : tensor<32xf32>
    %momlrbt5 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %momstbt5 = stablehlo.multiply %momlrbt5, %momlkbt5 : tensor<32xf32>
    %momnewbt5 = stablehlo.subtract %bt5, %momstbt5 : tensor<32xf32>
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
    %mommug6 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %momvgg6 = stablehlo.multiply %mommug6, %g6v : tensor<32xf32>
    %momvelg6 = stablehlo.add %momvgg6, %dg6 : tensor<32xf32>
    %momnvg6 = stablehlo.multiply %mommug6, %momvelg6 : tensor<32xf32>
    %momlkg6 = stablehlo.add %momnvg6, %dg6 : tensor<32xf32>
    %momlrg6 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %momstg6 = stablehlo.multiply %momlrg6, %momlkg6 : tensor<32xf32>
    %momnewg6 = stablehlo.subtract %g6, %momstg6 : tensor<32xf32>
    %mommubt6 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %momvgbt6 = stablehlo.multiply %mommubt6, %bt6v : tensor<32xf32>
    %momvelbt6 = stablehlo.add %momvgbt6, %dbt6 : tensor<32xf32>
    %momnvbt6 = stablehlo.multiply %mommubt6, %momvelbt6 : tensor<32xf32>
    %momlkbt6 = stablehlo.add %momnvbt6, %dbt6 : tensor<32xf32>
    %momlrbt6 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %momstbt6 = stablehlo.multiply %momlrbt6, %momlkbt6 : tensor<32xf32>
    %momnewbt6 = stablehlo.subtract %bt6, %momstbt6 : tensor<32xf32>
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
    %mommug7 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %momvgg7 = stablehlo.multiply %mommug7, %g7v : tensor<32xf32>
    %momvelg7 = stablehlo.add %momvgg7, %dg7 : tensor<32xf32>
    %momnvg7 = stablehlo.multiply %mommug7, %momvelg7 : tensor<32xf32>
    %momlkg7 = stablehlo.add %momnvg7, %dg7 : tensor<32xf32>
    %momlrg7 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %momstg7 = stablehlo.multiply %momlrg7, %momlkg7 : tensor<32xf32>
    %momnewg7 = stablehlo.subtract %g7, %momstg7 : tensor<32xf32>
    %mommubt7 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %momvgbt7 = stablehlo.multiply %mommubt7, %bt7v : tensor<32xf32>
    %momvelbt7 = stablehlo.add %momvgbt7, %dbt7 : tensor<32xf32>
    %momnvbt7 = stablehlo.multiply %mommubt7, %momvelbt7 : tensor<32xf32>
    %momlkbt7 = stablehlo.add %momnvbt7, %dbt7 : tensor<32xf32>
    %momlrbt7 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %momstbt7 = stablehlo.multiply %momlrbt7, %momlkbt7 : tensor<32xf32>
    %momnewbt7 = stablehlo.subtract %bt7, %momstbt7 : tensor<32xf32>
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
    %mommug8 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %momvgg8 = stablehlo.multiply %mommug8, %g8v : tensor<32xf32>
    %momvelg8 = stablehlo.add %momvgg8, %dg8 : tensor<32xf32>
    %momnvg8 = stablehlo.multiply %mommug8, %momvelg8 : tensor<32xf32>
    %momlkg8 = stablehlo.add %momnvg8, %dg8 : tensor<32xf32>
    %momlrg8 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %momstg8 = stablehlo.multiply %momlrg8, %momlkg8 : tensor<32xf32>
    %momnewg8 = stablehlo.subtract %g8, %momstg8 : tensor<32xf32>
    %mommubt8 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %momvgbt8 = stablehlo.multiply %mommubt8, %bt8v : tensor<32xf32>
    %momvelbt8 = stablehlo.add %momvgbt8, %dbt8 : tensor<32xf32>
    %momnvbt8 = stablehlo.multiply %mommubt8, %momvelbt8 : tensor<32xf32>
    %momlkbt8 = stablehlo.add %momnvbt8, %dbt8 : tensor<32xf32>
    %momlrbt8 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %momstbt8 = stablehlo.multiply %momlrbt8, %momlkbt8 : tensor<32xf32>
    %momnewbt8 = stablehlo.subtract %bt8, %momstbt8 : tensor<32xf32>
    %mommuW9 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %momvgW9 = stablehlo.multiply %mommuW9, %W9v : tensor<128x512xf32>
    %momvelW9 = stablehlo.add %momvgW9, %dW9 : tensor<128x512xf32>
    %momnvW9 = stablehlo.multiply %mommuW9, %momvelW9 : tensor<128x512xf32>
    %momlkW9 = stablehlo.add %momnvW9, %dW9 : tensor<128x512xf32>
    %momlrW9 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %momstW9 = stablehlo.multiply %momlrW9, %momlkW9 : tensor<128x512xf32>
    %momnewW9 = stablehlo.subtract %W9, %momstW9 : tensor<128x512xf32>
    %mommub9 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %momvgb9 = stablehlo.multiply %mommub9, %b9v : tensor<512xf32>
    %momvelb9 = stablehlo.add %momvgb9, %db9 : tensor<512xf32>
    %momnvb9 = stablehlo.multiply %mommub9, %momvelb9 : tensor<512xf32>
    %momlkb9 = stablehlo.add %momnvb9, %db9 : tensor<512xf32>
    %momlrb9 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %momstb9 = stablehlo.multiply %momlrb9, %momlkb9 : tensor<512xf32>
    %momnewb9 = stablehlo.subtract %b9, %momstb9 : tensor<512xf32>
    %mommuWa = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %momvgWa = stablehlo.multiply %mommuWa, %Wav : tensor<512x512xf32>
    %momvelWa = stablehlo.add %momvgWa, %dWa : tensor<512x512xf32>
    %momnvWa = stablehlo.multiply %mommuWa, %momvelWa : tensor<512x512xf32>
    %momlkWa = stablehlo.add %momnvWa, %dWa : tensor<512x512xf32>
    %momlrWa = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %momstWa = stablehlo.multiply %momlrWa, %momlkWa : tensor<512x512xf32>
    %momnewWa = stablehlo.subtract %Wa, %momstWa : tensor<512x512xf32>
    %mommuba = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %momvgba = stablehlo.multiply %mommuba, %bav : tensor<512xf32>
    %momvelba = stablehlo.add %momvgba, %dba : tensor<512xf32>
    %momnvba = stablehlo.multiply %mommuba, %momvelba : tensor<512xf32>
    %momlkba = stablehlo.add %momnvba, %dba : tensor<512xf32>
    %momlrba = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %momstba = stablehlo.multiply %momlrba, %momlkba : tensor<512xf32>
    %momnewba = stablehlo.subtract %ba, %momstba : tensor<512xf32>
    %mommuWb = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %momvgWb = stablehlo.multiply %mommuWb, %Wbv : tensor<512x10xf32>
    %momvelWb = stablehlo.add %momvgWb, %dWb : tensor<512x10xf32>
    %momnvWb = stablehlo.multiply %mommuWb, %momvelWb : tensor<512x10xf32>
    %momlkWb = stablehlo.add %momnvWb, %dWb : tensor<512x10xf32>
    %momlrWb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %momstWb = stablehlo.multiply %momlrWb, %momlkWb : tensor<512x10xf32>
    %momnewWb = stablehlo.subtract %Wb, %momstWb : tensor<512x10xf32>
    %mommubb = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %momvgbb = stablehlo.multiply %mommubb, %bbv : tensor<10xf32>
    %momvelbb = stablehlo.add %momvgbb, %dbb : tensor<10xf32>
    %momnvbb = stablehlo.multiply %mommubb, %momvelbb : tensor<10xf32>
    %momlkbb = stablehlo.add %momnvbb, %dbb : tensor<10xf32>
    %momlrbb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %momstbb = stablehlo.multiply %momlrbb, %momlkbb : tensor<10xf32>
    %momnewbb = stablehlo.subtract %bb, %momstbb : tensor<10xf32>
    return %momnewW1, %momnewcb1, %momnewg1, %momnewbt1, %momnewW2, %momnewcb2, %momnewg2, %momnewbt2, %momnewW3, %momnewcb3, %momnewg3, %momnewbt3, %momnewW4, %momnewcb4, %momnewg4, %momnewbt4, %momnewW5, %momnewcb5, %momnewg5, %momnewbt5, %momnewW6, %momnewcb6, %momnewg6, %momnewbt6, %momnewW7, %momnewcb7, %momnewg7, %momnewbt7, %momnewW8, %momnewcb8, %momnewg8, %momnewbt8, %momnewW9, %momnewb9, %momnewWa, %momnewba, %momnewWb, %momnewbb, %W1m, %cb1m, %g1m, %bt1m, %W2m, %cb2m, %g2m, %bt2m, %W3m, %cb3m, %g3m, %bt3m, %W4m, %cb4m, %g4m, %bt4m, %W5m, %cb5m, %g5m, %bt5m, %W6m, %cb6m, %g6m, %bt6m, %W7m, %cb7m, %g7m, %bt7m, %W8m, %cb8m, %g8m, %bt8m, %W9m, %b9m, %Wam, %bam, %Wbm, %bbm, %momvelW1, %momvelcb1, %momvelg1, %momvelbt1, %momvelW2, %momvelcb2, %momvelg2, %momvelbt2, %momvelW3, %momvelcb3, %momvelg3, %momvelbt3, %momvelW4, %momvelcb4, %momvelg4, %momvelbt4, %momvelW5, %momvelcb5, %momvelg5, %momvelbt5, %momvelW6, %momvelcb6, %momvelg6, %momvelbt6, %momvelW7, %momvelcb7, %momvelg7, %momvelbt7, %momvelW8, %momvelcb8, %momvelg8, %momvelbt8, %momvelW9, %momvelb9, %momvelWa, %momvelba, %momvelWb, %momvelbb, %loss, %bc1, %bc2 : tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>
  }
}

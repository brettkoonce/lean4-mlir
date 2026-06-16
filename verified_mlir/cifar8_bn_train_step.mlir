module @m {
  func.func @cifar8_bn_train_step(%x: tensor<128x3072xf32>, %W1: tensor<16x3x3x3xf32>, %b1: tensor<16xf32>, %g1: tensor<16xf32>, %bt1: tensor<16xf32>, %W2: tensor<16x16x3x3xf32>, %b2: tensor<16xf32>, %g2: tensor<16xf32>, %bt2: tensor<16xf32>, %W3: tensor<16x16x3x3xf32>, %b3: tensor<16xf32>, %g3: tensor<16xf32>, %bt3: tensor<16xf32>, %W4: tensor<16x16x3x3xf32>, %b4: tensor<16xf32>, %g4: tensor<16xf32>, %bt4: tensor<16xf32>, %W5: tensor<32x16x3x3xf32>, %b5: tensor<32xf32>, %g5: tensor<32xf32>, %bt5: tensor<32xf32>, %W6: tensor<32x32x3x3xf32>, %b6: tensor<32xf32>, %g6: tensor<32xf32>, %bt6: tensor<32xf32>, %W7: tensor<32x32x3x3xf32>, %b7: tensor<32xf32>, %g7: tensor<32xf32>, %bt7: tensor<32xf32>, %W8: tensor<32x32x3x3xf32>, %b8: tensor<32xf32>, %g8: tensor<32xf32>, %bt8: tensor<32xf32>, %W9: tensor<128x64xf32>, %b9: tensor<64xf32>, %Wa: tensor<64x64xf32>, %ba: tensor<64xf32>, %Wb: tensor<64x10xf32>, %bb: tensor<10xf32>, %onehot: tensor<128x10xf32>) -> (tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>) {
    %sc = stablehlo.constant dense<0.0> : tensor<f32>
    // ── forward: (conv→BN→relu)×2→pool ×4 →flatten→(dense→relu)×2→dense ──
    %xr = stablehlo.reshape %x : (tensor<128x3072xf32>) -> tensor<128x3x32x32xf32>
    %hc1c = stablehlo.convolution(%xr, %W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x3x32x32xf32>, tensor<16x3x3x3xf32>) -> tensor<128x16x32x32xf32>
    %hc1b = stablehlo.broadcast_in_dim %b1, dims = [1] : (tensor<16xf32>) -> tensor<128x16x32x32xf32>
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
    %hc2b = stablehlo.broadcast_in_dim %b2, dims = [1] : (tensor<16xf32>) -> tensor<128x16x32x32xf32>
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
    %hc3b = stablehlo.broadcast_in_dim %b3, dims = [1] : (tensor<16xf32>) -> tensor<128x16x16x16xf32>
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
    %hc4b = stablehlo.broadcast_in_dim %b4, dims = [1] : (tensor<16xf32>) -> tensor<128x16x16x16xf32>
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
    %hc5b = stablehlo.broadcast_in_dim %b5, dims = [1] : (tensor<32xf32>) -> tensor<128x32x8x8xf32>
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
    %hc6b = stablehlo.broadcast_in_dim %b6, dims = [1] : (tensor<32xf32>) -> tensor<128x32x8x8xf32>
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
    %hc7b = stablehlo.broadcast_in_dim %b7, dims = [1] : (tensor<32xf32>) -> tensor<128x32x4x4xf32>
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
    %hc8b = stablehlo.broadcast_in_dim %b8, dims = [1] : (tensor<32xf32>) -> tensor<128x32x4x4xf32>
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
    // ── loss cotangent dy = softmax(logits) − onehot ──
    %le = stablehlo.exponential %logits : tensor<128x10xf32>
    %lsum = stablehlo.reduce(%le init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<128x10xf32>, tensor<f32>) -> tensor<128xf32>
    %lsb = stablehlo.broadcast_in_dim %lsum, dims = [0] : (tensor<128xf32>) -> tensor<128x10xf32>
    %lsm = stablehlo.divide %le, %lsb : tensor<128x10xf32>
    %dy = stablehlo.subtract %lsm, %onehot : tensor<128x10xf32>
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
    // ── SGD θ' = θ − lr·∇ (all 38 params, incl. per-channel γ/β) ──
    %W1l = stablehlo.constant dense<0.00078125> : tensor<16x3x3x3xf32>
    %W1s = stablehlo.multiply %dW1, %W1l : tensor<16x3x3x3xf32>
    %W1n = stablehlo.subtract %W1, %W1s : tensor<16x3x3x3xf32>
    %b1l = stablehlo.constant dense<0.00078125> : tensor<16xf32>
    %b1s = stablehlo.multiply %db1, %b1l : tensor<16xf32>
    %b1n = stablehlo.subtract %b1, %b1s : tensor<16xf32>
    %g1l = stablehlo.constant dense<0.00078125> : tensor<16xf32>
    %g1s = stablehlo.multiply %dg1, %g1l : tensor<16xf32>
    %g1n = stablehlo.subtract %g1, %g1s : tensor<16xf32>
    %bt1l = stablehlo.constant dense<0.00078125> : tensor<16xf32>
    %bt1s = stablehlo.multiply %dbt1, %bt1l : tensor<16xf32>
    %bt1n = stablehlo.subtract %bt1, %bt1s : tensor<16xf32>
    %W2l = stablehlo.constant dense<0.00078125> : tensor<16x16x3x3xf32>
    %W2s = stablehlo.multiply %dW2, %W2l : tensor<16x16x3x3xf32>
    %W2n = stablehlo.subtract %W2, %W2s : tensor<16x16x3x3xf32>
    %b2l = stablehlo.constant dense<0.00078125> : tensor<16xf32>
    %b2s = stablehlo.multiply %db2, %b2l : tensor<16xf32>
    %b2n = stablehlo.subtract %b2, %b2s : tensor<16xf32>
    %g2l = stablehlo.constant dense<0.00078125> : tensor<16xf32>
    %g2s = stablehlo.multiply %dg2, %g2l : tensor<16xf32>
    %g2n = stablehlo.subtract %g2, %g2s : tensor<16xf32>
    %bt2l = stablehlo.constant dense<0.00078125> : tensor<16xf32>
    %bt2s = stablehlo.multiply %dbt2, %bt2l : tensor<16xf32>
    %bt2n = stablehlo.subtract %bt2, %bt2s : tensor<16xf32>
    %W3l = stablehlo.constant dense<0.00078125> : tensor<16x16x3x3xf32>
    %W3s = stablehlo.multiply %dW3, %W3l : tensor<16x16x3x3xf32>
    %W3n = stablehlo.subtract %W3, %W3s : tensor<16x16x3x3xf32>
    %b3l = stablehlo.constant dense<0.00078125> : tensor<16xf32>
    %b3s = stablehlo.multiply %db3, %b3l : tensor<16xf32>
    %b3n = stablehlo.subtract %b3, %b3s : tensor<16xf32>
    %g3l = stablehlo.constant dense<0.00078125> : tensor<16xf32>
    %g3s = stablehlo.multiply %dg3, %g3l : tensor<16xf32>
    %g3n = stablehlo.subtract %g3, %g3s : tensor<16xf32>
    %bt3l = stablehlo.constant dense<0.00078125> : tensor<16xf32>
    %bt3s = stablehlo.multiply %dbt3, %bt3l : tensor<16xf32>
    %bt3n = stablehlo.subtract %bt3, %bt3s : tensor<16xf32>
    %W4l = stablehlo.constant dense<0.00078125> : tensor<16x16x3x3xf32>
    %W4s = stablehlo.multiply %dW4, %W4l : tensor<16x16x3x3xf32>
    %W4n = stablehlo.subtract %W4, %W4s : tensor<16x16x3x3xf32>
    %b4l = stablehlo.constant dense<0.00078125> : tensor<16xf32>
    %b4s = stablehlo.multiply %db4, %b4l : tensor<16xf32>
    %b4n = stablehlo.subtract %b4, %b4s : tensor<16xf32>
    %g4l = stablehlo.constant dense<0.00078125> : tensor<16xf32>
    %g4s = stablehlo.multiply %dg4, %g4l : tensor<16xf32>
    %g4n = stablehlo.subtract %g4, %g4s : tensor<16xf32>
    %bt4l = stablehlo.constant dense<0.00078125> : tensor<16xf32>
    %bt4s = stablehlo.multiply %dbt4, %bt4l : tensor<16xf32>
    %bt4n = stablehlo.subtract %bt4, %bt4s : tensor<16xf32>
    %W5l = stablehlo.constant dense<0.00078125> : tensor<32x16x3x3xf32>
    %W5s = stablehlo.multiply %dW5, %W5l : tensor<32x16x3x3xf32>
    %W5n = stablehlo.subtract %W5, %W5s : tensor<32x16x3x3xf32>
    %b5l = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %b5s = stablehlo.multiply %db5, %b5l : tensor<32xf32>
    %b5n = stablehlo.subtract %b5, %b5s : tensor<32xf32>
    %g5l = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %g5s = stablehlo.multiply %dg5, %g5l : tensor<32xf32>
    %g5n = stablehlo.subtract %g5, %g5s : tensor<32xf32>
    %bt5l = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %bt5s = stablehlo.multiply %dbt5, %bt5l : tensor<32xf32>
    %bt5n = stablehlo.subtract %bt5, %bt5s : tensor<32xf32>
    %W6l = stablehlo.constant dense<0.00078125> : tensor<32x32x3x3xf32>
    %W6s = stablehlo.multiply %dW6, %W6l : tensor<32x32x3x3xf32>
    %W6n = stablehlo.subtract %W6, %W6s : tensor<32x32x3x3xf32>
    %b6l = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %b6s = stablehlo.multiply %db6, %b6l : tensor<32xf32>
    %b6n = stablehlo.subtract %b6, %b6s : tensor<32xf32>
    %g6l = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %g6s = stablehlo.multiply %dg6, %g6l : tensor<32xf32>
    %g6n = stablehlo.subtract %g6, %g6s : tensor<32xf32>
    %bt6l = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %bt6s = stablehlo.multiply %dbt6, %bt6l : tensor<32xf32>
    %bt6n = stablehlo.subtract %bt6, %bt6s : tensor<32xf32>
    %W7l = stablehlo.constant dense<0.00078125> : tensor<32x32x3x3xf32>
    %W7s = stablehlo.multiply %dW7, %W7l : tensor<32x32x3x3xf32>
    %W7n = stablehlo.subtract %W7, %W7s : tensor<32x32x3x3xf32>
    %b7l = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %b7s = stablehlo.multiply %db7, %b7l : tensor<32xf32>
    %b7n = stablehlo.subtract %b7, %b7s : tensor<32xf32>
    %g7l = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %g7s = stablehlo.multiply %dg7, %g7l : tensor<32xf32>
    %g7n = stablehlo.subtract %g7, %g7s : tensor<32xf32>
    %bt7l = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %bt7s = stablehlo.multiply %dbt7, %bt7l : tensor<32xf32>
    %bt7n = stablehlo.subtract %bt7, %bt7s : tensor<32xf32>
    %W8l = stablehlo.constant dense<0.00078125> : tensor<32x32x3x3xf32>
    %W8s = stablehlo.multiply %dW8, %W8l : tensor<32x32x3x3xf32>
    %W8n = stablehlo.subtract %W8, %W8s : tensor<32x32x3x3xf32>
    %b8l = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %b8s = stablehlo.multiply %db8, %b8l : tensor<32xf32>
    %b8n = stablehlo.subtract %b8, %b8s : tensor<32xf32>
    %g8l = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %g8s = stablehlo.multiply %dg8, %g8l : tensor<32xf32>
    %g8n = stablehlo.subtract %g8, %g8s : tensor<32xf32>
    %bt8l = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %bt8s = stablehlo.multiply %dbt8, %bt8l : tensor<32xf32>
    %bt8n = stablehlo.subtract %bt8, %bt8s : tensor<32xf32>
    %W9l = stablehlo.constant dense<0.00078125> : tensor<128x64xf32>
    %W9s = stablehlo.multiply %dW9, %W9l : tensor<128x64xf32>
    %W9n = stablehlo.subtract %W9, %W9s : tensor<128x64xf32>
    %b9l = stablehlo.constant dense<0.00078125> : tensor<64xf32>
    %b9s = stablehlo.multiply %db9, %b9l : tensor<64xf32>
    %b9n = stablehlo.subtract %b9, %b9s : tensor<64xf32>
    %Wal = stablehlo.constant dense<0.00078125> : tensor<64x64xf32>
    %Was = stablehlo.multiply %dWa, %Wal : tensor<64x64xf32>
    %Wan = stablehlo.subtract %Wa, %Was : tensor<64x64xf32>
    %bal = stablehlo.constant dense<0.00078125> : tensor<64xf32>
    %bas = stablehlo.multiply %dba, %bal : tensor<64xf32>
    %ban = stablehlo.subtract %ba, %bas : tensor<64xf32>
    %Wbl = stablehlo.constant dense<0.00078125> : tensor<64x10xf32>
    %Wbs = stablehlo.multiply %dWb, %Wbl : tensor<64x10xf32>
    %Wbn = stablehlo.subtract %Wb, %Wbs : tensor<64x10xf32>
    %bbl = stablehlo.constant dense<0.00078125> : tensor<10xf32>
    %bbs = stablehlo.multiply %dbb, %bbl : tensor<10xf32>
    %bbn = stablehlo.subtract %bb, %bbs : tensor<10xf32>
    return %W1n, %b1n, %g1n, %bt1n, %W2n, %b2n, %g2n, %bt2n, %W3n, %b3n, %g3n, %bt3n, %W4n, %b4n, %g4n, %bt4n, %W5n, %b5n, %g5n, %bt5n, %W6n, %b6n, %g6n, %bt6n, %W7n, %b7n, %g7n, %bt7n, %W8n, %b8n, %g8n, %bt8n, %W9n, %b9n, %Wan, %ban, %Wbn, %bbn : tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>
  }
}

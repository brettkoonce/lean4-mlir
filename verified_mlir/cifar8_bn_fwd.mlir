module @m {
  func.func @cifar8_bn_fwd(%x: tensor<128x3072xf32>, %W1: tensor<16x3x3x3xf32>, %b1: tensor<16xf32>, %g1: tensor<16xf32>, %bt1: tensor<16xf32>, %W2: tensor<16x16x3x3xf32>, %b2: tensor<16xf32>, %g2: tensor<16xf32>, %bt2: tensor<16xf32>, %W3: tensor<16x16x3x3xf32>, %b3: tensor<16xf32>, %g3: tensor<16xf32>, %bt3: tensor<16xf32>, %W4: tensor<16x16x3x3xf32>, %b4: tensor<16xf32>, %g4: tensor<16xf32>, %bt4: tensor<16xf32>, %W5: tensor<32x16x3x3xf32>, %b5: tensor<32xf32>, %g5: tensor<32xf32>, %bt5: tensor<32xf32>, %W6: tensor<32x32x3x3xf32>, %b6: tensor<32xf32>, %g6: tensor<32xf32>, %bt6: tensor<32xf32>, %W7: tensor<32x32x3x3xf32>, %b7: tensor<32xf32>, %g7: tensor<32xf32>, %bt7: tensor<32xf32>, %W8: tensor<32x32x3x3xf32>, %b8: tensor<32xf32>, %g8: tensor<32xf32>, %bt8: tensor<32xf32>, %W9: tensor<128x64xf32>, %b9: tensor<64xf32>, %Wa: tensor<64x64xf32>, %ba: tensor<64xf32>, %Wb: tensor<64x10xf32>, %bb: tensor<10xf32>) -> tensor<128x10xf32> {
    %sc = stablehlo.constant dense<0.0> : tensor<f32>
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
    return %logits : tensor<128x10xf32>
  }
}

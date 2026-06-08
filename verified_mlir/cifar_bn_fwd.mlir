module @m {
  func.func @cifar_bn_fwd(%x: tensor<128x3072xf32>, %W1: tensor<32x3x3x3xf32>, %b1: tensor<32xf32>, %g1: tensor<32xf32>, %bt1: tensor<32xf32>, %W2: tensor<32x32x3x3xf32>, %b2: tensor<32xf32>, %g2: tensor<32xf32>, %bt2: tensor<32xf32>, %W3: tensor<64x32x3x3xf32>, %b3: tensor<64xf32>, %g3: tensor<64xf32>, %bt3: tensor<64xf32>, %W4: tensor<64x64x3x3xf32>, %b4: tensor<64xf32>, %g4: tensor<64xf32>, %bt4: tensor<64xf32>, %W5: tensor<4096x512xf32>, %b5: tensor<512xf32>, %W6: tensor<512x512xf32>, %b6: tensor<512xf32>, %W7: tensor<512x10xf32>, %b7: tensor<10xf32>) -> tensor<128x10xf32> {
    %sc = stablehlo.constant dense<0.0> : tensor<f32>
    %xr = stablehlo.reshape %x : (tensor<128x3072xf32>) -> tensor<128x3x32x32xf32>
    %hc1c = stablehlo.convolution(%xr, %W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x3x32x32xf32>, tensor<32x3x3x3xf32>) -> tensor<128x32x32x32xf32>
    %hc1b = stablehlo.broadcast_in_dim %b1, dims = [1] : (tensor<32xf32>) -> tensor<128x32x32x32xf32>
    %hc1 = stablehlo.add %hc1c, %hc1b : tensor<128x32x32x32xf32>
    %hc1f = stablehlo.reshape %hc1 : (tensor<128x32x32x32xf32>) -> tensor<128x32768xf32>
    %bn1_xr = stablehlo.reshape %hc1f : (tensor<128x32768xf32>) -> tensor<128x32x1024xf32>
    %bn1_nf = stablehlo.constant dense<1024.0> : tensor<128x32x1024xf32>
    %bn1_ep = stablehlo.constant dense<1.0e-05> : tensor<128x32x1024xf32>
    %bn1_smr = stablehlo.reduce(%bn1_xr init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<128x32x1024xf32>, tensor<f32>) -> tensor<128x32xf32>
    %bn1_sm = stablehlo.broadcast_in_dim %bn1_smr, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x1024xf32>
    %bn1_mu = stablehlo.divide %bn1_sm, %bn1_nf : tensor<128x32x1024xf32>
    %bn1_xc = stablehlo.subtract %bn1_xr, %bn1_mu : tensor<128x32x1024xf32>
    %bn1_sq = stablehlo.multiply %bn1_xc, %bn1_xc : tensor<128x32x1024xf32>
    %bn1_vsr = stablehlo.reduce(%bn1_sq init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<128x32x1024xf32>, tensor<f32>) -> tensor<128x32xf32>
    %bn1_vs = stablehlo.broadcast_in_dim %bn1_vsr, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x1024xf32>
    %bn1_var = stablehlo.divide %bn1_vs, %bn1_nf : tensor<128x32x1024xf32>
    %bn1_ve = stablehlo.add %bn1_var, %bn1_ep : tensor<128x32x1024xf32>
    %bn1_istd = stablehlo.rsqrt %bn1_ve : tensor<128x32x1024xf32>
    %bn1_xhat = stablehlo.multiply %bn1_xc, %bn1_istd : tensor<128x32x1024xf32>
    %bn1_gb = stablehlo.broadcast_in_dim %g1, dims = [1] : (tensor<32xf32>) -> tensor<128x32x1024xf32>
    %bn1_bb = stablehlo.broadcast_in_dim %bt1, dims = [1] : (tensor<32xf32>) -> tensor<128x32x1024xf32>
    %bn1_gx = stablehlo.multiply %bn1_xhat, %bn1_gb : tensor<128x32x1024xf32>
    %bn1_y3 = stablehlo.add %bn1_gx, %bn1_bb : tensor<128x32x1024xf32>
    %bn1 = stablehlo.reshape %bn1_y3 : (tensor<128x32x1024xf32>) -> tensor<128x32768xf32>
    %ac1fz = stablehlo.constant dense<0.0> : tensor<128x32768xf32>
    %ac1f = stablehlo.maximum %bn1, %ac1fz : tensor<128x32768xf32>
    %ac1 = stablehlo.reshape %ac1f : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %hc2c = stablehlo.convolution(%ac1, %W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x32x32xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x32x32xf32>
    %hc2b = stablehlo.broadcast_in_dim %b2, dims = [1] : (tensor<32xf32>) -> tensor<128x32x32x32xf32>
    %hc2 = stablehlo.add %hc2c, %hc2b : tensor<128x32x32x32xf32>
    %hc2f = stablehlo.reshape %hc2 : (tensor<128x32x32x32xf32>) -> tensor<128x32768xf32>
    %bn2_xr = stablehlo.reshape %hc2f : (tensor<128x32768xf32>) -> tensor<128x32x1024xf32>
    %bn2_nf = stablehlo.constant dense<1024.0> : tensor<128x32x1024xf32>
    %bn2_ep = stablehlo.constant dense<1.0e-05> : tensor<128x32x1024xf32>
    %bn2_smr = stablehlo.reduce(%bn2_xr init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<128x32x1024xf32>, tensor<f32>) -> tensor<128x32xf32>
    %bn2_sm = stablehlo.broadcast_in_dim %bn2_smr, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x1024xf32>
    %bn2_mu = stablehlo.divide %bn2_sm, %bn2_nf : tensor<128x32x1024xf32>
    %bn2_xc = stablehlo.subtract %bn2_xr, %bn2_mu : tensor<128x32x1024xf32>
    %bn2_sq = stablehlo.multiply %bn2_xc, %bn2_xc : tensor<128x32x1024xf32>
    %bn2_vsr = stablehlo.reduce(%bn2_sq init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<128x32x1024xf32>, tensor<f32>) -> tensor<128x32xf32>
    %bn2_vs = stablehlo.broadcast_in_dim %bn2_vsr, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x1024xf32>
    %bn2_var = stablehlo.divide %bn2_vs, %bn2_nf : tensor<128x32x1024xf32>
    %bn2_ve = stablehlo.add %bn2_var, %bn2_ep : tensor<128x32x1024xf32>
    %bn2_istd = stablehlo.rsqrt %bn2_ve : tensor<128x32x1024xf32>
    %bn2_xhat = stablehlo.multiply %bn2_xc, %bn2_istd : tensor<128x32x1024xf32>
    %bn2_gb = stablehlo.broadcast_in_dim %g2, dims = [1] : (tensor<32xf32>) -> tensor<128x32x1024xf32>
    %bn2_bb = stablehlo.broadcast_in_dim %bt2, dims = [1] : (tensor<32xf32>) -> tensor<128x32x1024xf32>
    %bn2_gx = stablehlo.multiply %bn2_xhat, %bn2_gb : tensor<128x32x1024xf32>
    %bn2_y3 = stablehlo.add %bn2_gx, %bn2_bb : tensor<128x32x1024xf32>
    %bn2 = stablehlo.reshape %bn2_y3 : (tensor<128x32x1024xf32>) -> tensor<128x32768xf32>
    %ac2fz = stablehlo.constant dense<0.0> : tensor<128x32768xf32>
    %ac2f = stablehlo.maximum %bn2, %ac2fz : tensor<128x32768xf32>
    %ac2 = stablehlo.reshape %ac2f : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
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
    %hc3f = stablehlo.reshape %hc3 : (tensor<128x64x16x16xf32>) -> tensor<128x16384xf32>
    %bn3_xr = stablehlo.reshape %hc3f : (tensor<128x16384xf32>) -> tensor<128x64x256xf32>
    %bn3_nf = stablehlo.constant dense<256.0> : tensor<128x64x256xf32>
    %bn3_ep = stablehlo.constant dense<1.0e-05> : tensor<128x64x256xf32>
    %bn3_smr = stablehlo.reduce(%bn3_xr init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<128x64x256xf32>, tensor<f32>) -> tensor<128x64xf32>
    %bn3_sm = stablehlo.broadcast_in_dim %bn3_smr, dims = [0, 1] : (tensor<128x64xf32>) -> tensor<128x64x256xf32>
    %bn3_mu = stablehlo.divide %bn3_sm, %bn3_nf : tensor<128x64x256xf32>
    %bn3_xc = stablehlo.subtract %bn3_xr, %bn3_mu : tensor<128x64x256xf32>
    %bn3_sq = stablehlo.multiply %bn3_xc, %bn3_xc : tensor<128x64x256xf32>
    %bn3_vsr = stablehlo.reduce(%bn3_sq init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<128x64x256xf32>, tensor<f32>) -> tensor<128x64xf32>
    %bn3_vs = stablehlo.broadcast_in_dim %bn3_vsr, dims = [0, 1] : (tensor<128x64xf32>) -> tensor<128x64x256xf32>
    %bn3_var = stablehlo.divide %bn3_vs, %bn3_nf : tensor<128x64x256xf32>
    %bn3_ve = stablehlo.add %bn3_var, %bn3_ep : tensor<128x64x256xf32>
    %bn3_istd = stablehlo.rsqrt %bn3_ve : tensor<128x64x256xf32>
    %bn3_xhat = stablehlo.multiply %bn3_xc, %bn3_istd : tensor<128x64x256xf32>
    %bn3_gb = stablehlo.broadcast_in_dim %g3, dims = [1] : (tensor<64xf32>) -> tensor<128x64x256xf32>
    %bn3_bb = stablehlo.broadcast_in_dim %bt3, dims = [1] : (tensor<64xf32>) -> tensor<128x64x256xf32>
    %bn3_gx = stablehlo.multiply %bn3_xhat, %bn3_gb : tensor<128x64x256xf32>
    %bn3_y3 = stablehlo.add %bn3_gx, %bn3_bb : tensor<128x64x256xf32>
    %bn3 = stablehlo.reshape %bn3_y3 : (tensor<128x64x256xf32>) -> tensor<128x16384xf32>
    %ac3fz = stablehlo.constant dense<0.0> : tensor<128x16384xf32>
    %ac3f = stablehlo.maximum %bn3, %ac3fz : tensor<128x16384xf32>
    %ac3 = stablehlo.reshape %ac3f : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %hc4c = stablehlo.convolution(%ac3, %W4)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x16x16xf32>, tensor<64x64x3x3xf32>) -> tensor<128x64x16x16xf32>
    %hc4b = stablehlo.broadcast_in_dim %b4, dims = [1] : (tensor<64xf32>) -> tensor<128x64x16x16xf32>
    %hc4 = stablehlo.add %hc4c, %hc4b : tensor<128x64x16x16xf32>
    %hc4f = stablehlo.reshape %hc4 : (tensor<128x64x16x16xf32>) -> tensor<128x16384xf32>
    %bn4_xr = stablehlo.reshape %hc4f : (tensor<128x16384xf32>) -> tensor<128x64x256xf32>
    %bn4_nf = stablehlo.constant dense<256.0> : tensor<128x64x256xf32>
    %bn4_ep = stablehlo.constant dense<1.0e-05> : tensor<128x64x256xf32>
    %bn4_smr = stablehlo.reduce(%bn4_xr init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<128x64x256xf32>, tensor<f32>) -> tensor<128x64xf32>
    %bn4_sm = stablehlo.broadcast_in_dim %bn4_smr, dims = [0, 1] : (tensor<128x64xf32>) -> tensor<128x64x256xf32>
    %bn4_mu = stablehlo.divide %bn4_sm, %bn4_nf : tensor<128x64x256xf32>
    %bn4_xc = stablehlo.subtract %bn4_xr, %bn4_mu : tensor<128x64x256xf32>
    %bn4_sq = stablehlo.multiply %bn4_xc, %bn4_xc : tensor<128x64x256xf32>
    %bn4_vsr = stablehlo.reduce(%bn4_sq init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<128x64x256xf32>, tensor<f32>) -> tensor<128x64xf32>
    %bn4_vs = stablehlo.broadcast_in_dim %bn4_vsr, dims = [0, 1] : (tensor<128x64xf32>) -> tensor<128x64x256xf32>
    %bn4_var = stablehlo.divide %bn4_vs, %bn4_nf : tensor<128x64x256xf32>
    %bn4_ve = stablehlo.add %bn4_var, %bn4_ep : tensor<128x64x256xf32>
    %bn4_istd = stablehlo.rsqrt %bn4_ve : tensor<128x64x256xf32>
    %bn4_xhat = stablehlo.multiply %bn4_xc, %bn4_istd : tensor<128x64x256xf32>
    %bn4_gb = stablehlo.broadcast_in_dim %g4, dims = [1] : (tensor<64xf32>) -> tensor<128x64x256xf32>
    %bn4_bb = stablehlo.broadcast_in_dim %bt4, dims = [1] : (tensor<64xf32>) -> tensor<128x64x256xf32>
    %bn4_gx = stablehlo.multiply %bn4_xhat, %bn4_gb : tensor<128x64x256xf32>
    %bn4_y3 = stablehlo.add %bn4_gx, %bn4_bb : tensor<128x64x256xf32>
    %bn4 = stablehlo.reshape %bn4_y3 : (tensor<128x64x256xf32>) -> tensor<128x16384xf32>
    %ac4fz = stablehlo.constant dense<0.0> : tensor<128x16384xf32>
    %ac4f = stablehlo.maximum %bn4, %ac4fz : tensor<128x16384xf32>
    %ac4 = stablehlo.reshape %ac4f : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
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
    return %logits : tensor<128x10xf32>
  }
}

module @m {
  func.func @cifar_bn_train_step(%x: tensor<128x3072xf32>, %W1: tensor<32x3x3x3xf32>, %b1: tensor<32xf32>, %g1: tensor<32xf32>, %bt1: tensor<32xf32>, %W2: tensor<32x32x3x3xf32>, %b2: tensor<32xf32>, %g2: tensor<32xf32>, %bt2: tensor<32xf32>, %W3: tensor<64x32x3x3xf32>, %b3: tensor<64xf32>, %g3: tensor<64xf32>, %bt3: tensor<64xf32>, %W4: tensor<64x64x3x3xf32>, %b4: tensor<64xf32>, %g4: tensor<64xf32>, %bt4: tensor<64xf32>, %W5: tensor<4096x512xf32>, %b5: tensor<512xf32>, %W6: tensor<512x512xf32>, %b6: tensor<512xf32>, %W7: tensor<512x10xf32>, %b7: tensor<10xf32>, %onehot: tensor<128x10xf32>) -> (tensor<32x3x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<64x32x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<4096x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>) {
    %sc = stablehlo.constant dense<0.0> : tensor<f32>
    // ── forward: (conv→BN→relu)×2→pool →(conv→BN→relu)×2→pool →flatten→(dense→relu)×2→dense ──
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
    // ── loss cotangent dy = softmax(logits) − onehot ──
    %le = stablehlo.exponential %logits : tensor<128x10xf32>
    %lsum = stablehlo.reduce(%le init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<128x10xf32>, tensor<f32>) -> tensor<128xf32>
    %lsb = stablehlo.broadcast_in_dim %lsum, dims = [0] : (tensor<128xf32>) -> tensor<128x10xf32>
    %lsm = stablehlo.divide %le, %lsb : tensor<128x10xf32>
    %dy = stablehlo.subtract %lsm, %onehot : tensor<128x10xf32>
    // ── backward: dense (dotOut)+relu → scatter → (relu→BN-back→convBack)×stage, twice ──
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
    %dac4f = stablehlo.reshape %dac4 : (tensor<128x64x16x16xf32>) -> tensor<128x16384xf32>
    %dbn4z = stablehlo.constant dense<0.0> : tensor<128x16384xf32>
    %dbn4m = stablehlo.compare GT, %bn4, %dbn4z : (tensor<128x16384xf32>, tensor<128x16384xf32>) -> tensor<128x16384xi1>
    %dbn4 = stablehlo.select %dbn4m, %dac4f, %dbn4z : tensor<128x16384xi1>, tensor<128x16384xf32>
    %dhc4f_dyr = stablehlo.reshape %dbn4 : (tensor<128x16384xf32>) -> tensor<128x64x256xf32>
    %dhc4f_gb = stablehlo.broadcast_in_dim %g4, dims = [1] : (tensor<64xf32>) -> tensor<128x64x256xf32>
    %dhc4f_dxh = stablehlo.multiply %dhc4f_gb, %dhc4f_dyr : tensor<128x64x256xf32>
    %dhc4f_sdxr = stablehlo.reduce(%dhc4f_dxh init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<128x64x256xf32>, tensor<f32>) -> tensor<128x64xf32>
    %dhc4f_sdx = stablehlo.broadcast_in_dim %dhc4f_sdxr, dims = [0, 1] : (tensor<128x64xf32>) -> tensor<128x64x256xf32>
    %dhc4f_xd = stablehlo.multiply %bn4_xhat, %dhc4f_dxh : tensor<128x64x256xf32>
    %dhc4f_sxdr = stablehlo.reduce(%dhc4f_xd init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<128x64x256xf32>, tensor<f32>) -> tensor<128x64xf32>
    %dhc4f_sxd = stablehlo.broadcast_in_dim %dhc4f_sxdr, dims = [0, 1] : (tensor<128x64xf32>) -> tensor<128x64x256xf32>
    %dhc4f_t1 = stablehlo.multiply %dhc4f_dxh, %bn4_nf : tensor<128x64x256xf32>
    %dhc4f_i1 = stablehlo.subtract %dhc4f_t1, %dhc4f_sdx : tensor<128x64x256xf32>
    %dhc4f_xs = stablehlo.multiply %bn4_xhat, %dhc4f_sxd : tensor<128x64x256xf32>
    %dhc4f_i2 = stablehlo.subtract %dhc4f_i1, %dhc4f_xs : tensor<128x64x256xf32>
    %dhc4f_s = stablehlo.divide %bn4_istd, %bn4_nf : tensor<128x64x256xf32>
    %dhc4f_dx3 = stablehlo.multiply %dhc4f_s, %dhc4f_i2 : tensor<128x64x256xf32>
    %dhc4f = stablehlo.reshape %dhc4f_dx3 : (tensor<128x64x256xf32>) -> tensor<128x16384xf32>
    %dg4_dyr = stablehlo.reshape %dbn4 : (tensor<128x16384xf32>) -> tensor<128x64x256xf32>
    %dg4_p = stablehlo.multiply %dg4_dyr, %bn4_xhat : tensor<128x64x256xf32>
    %dg4 = stablehlo.reduce(%dg4_p init: %sc) applies stablehlo.add across dimensions = [0, 2] : (tensor<128x64x256xf32>, tensor<f32>) -> tensor<64xf32>
    %dbt4 = stablehlo.reduce(%dg4_dyr init: %sc) applies stablehlo.add across dimensions = [0, 2] : (tensor<128x64x256xf32>, tensor<f32>) -> tensor<64xf32>
    %dhc4 = stablehlo.reshape %dhc4f : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %dac3t = stablehlo.transpose %W4, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %dac3r = stablehlo.reverse %dac3t, dims = [2, 3] : tensor<64x64x3x3xf32>
    %dac3 = stablehlo.convolution(%dhc4, %dac3r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x16x16xf32>, tensor<64x64x3x3xf32>) -> tensor<128x64x16x16xf32>
    %dac3f = stablehlo.reshape %dac3 : (tensor<128x64x16x16xf32>) -> tensor<128x16384xf32>
    %dbn3z = stablehlo.constant dense<0.0> : tensor<128x16384xf32>
    %dbn3m = stablehlo.compare GT, %bn3, %dbn3z : (tensor<128x16384xf32>, tensor<128x16384xf32>) -> tensor<128x16384xi1>
    %dbn3 = stablehlo.select %dbn3m, %dac3f, %dbn3z : tensor<128x16384xi1>, tensor<128x16384xf32>
    %dhc3f_dyr = stablehlo.reshape %dbn3 : (tensor<128x16384xf32>) -> tensor<128x64x256xf32>
    %dhc3f_gb = stablehlo.broadcast_in_dim %g3, dims = [1] : (tensor<64xf32>) -> tensor<128x64x256xf32>
    %dhc3f_dxh = stablehlo.multiply %dhc3f_gb, %dhc3f_dyr : tensor<128x64x256xf32>
    %dhc3f_sdxr = stablehlo.reduce(%dhc3f_dxh init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<128x64x256xf32>, tensor<f32>) -> tensor<128x64xf32>
    %dhc3f_sdx = stablehlo.broadcast_in_dim %dhc3f_sdxr, dims = [0, 1] : (tensor<128x64xf32>) -> tensor<128x64x256xf32>
    %dhc3f_xd = stablehlo.multiply %bn3_xhat, %dhc3f_dxh : tensor<128x64x256xf32>
    %dhc3f_sxdr = stablehlo.reduce(%dhc3f_xd init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<128x64x256xf32>, tensor<f32>) -> tensor<128x64xf32>
    %dhc3f_sxd = stablehlo.broadcast_in_dim %dhc3f_sxdr, dims = [0, 1] : (tensor<128x64xf32>) -> tensor<128x64x256xf32>
    %dhc3f_t1 = stablehlo.multiply %dhc3f_dxh, %bn3_nf : tensor<128x64x256xf32>
    %dhc3f_i1 = stablehlo.subtract %dhc3f_t1, %dhc3f_sdx : tensor<128x64x256xf32>
    %dhc3f_xs = stablehlo.multiply %bn3_xhat, %dhc3f_sxd : tensor<128x64x256xf32>
    %dhc3f_i2 = stablehlo.subtract %dhc3f_i1, %dhc3f_xs : tensor<128x64x256xf32>
    %dhc3f_s = stablehlo.divide %bn3_istd, %bn3_nf : tensor<128x64x256xf32>
    %dhc3f_dx3 = stablehlo.multiply %dhc3f_s, %dhc3f_i2 : tensor<128x64x256xf32>
    %dhc3f = stablehlo.reshape %dhc3f_dx3 : (tensor<128x64x256xf32>) -> tensor<128x16384xf32>
    %dg3_dyr = stablehlo.reshape %dbn3 : (tensor<128x16384xf32>) -> tensor<128x64x256xf32>
    %dg3_p = stablehlo.multiply %dg3_dyr, %bn3_xhat : tensor<128x64x256xf32>
    %dg3 = stablehlo.reduce(%dg3_p init: %sc) applies stablehlo.add across dimensions = [0, 2] : (tensor<128x64x256xf32>, tensor<f32>) -> tensor<64xf32>
    %dbt3 = stablehlo.reduce(%dg3_dyr init: %sc) applies stablehlo.add across dimensions = [0, 2] : (tensor<128x64x256xf32>, tensor<f32>) -> tensor<64xf32>
    %dhc3 = stablehlo.reshape %dhc3f : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
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
    %dac2f = stablehlo.reshape %dac2 : (tensor<128x32x32x32xf32>) -> tensor<128x32768xf32>
    %dbn2z = stablehlo.constant dense<0.0> : tensor<128x32768xf32>
    %dbn2m = stablehlo.compare GT, %bn2, %dbn2z : (tensor<128x32768xf32>, tensor<128x32768xf32>) -> tensor<128x32768xi1>
    %dbn2 = stablehlo.select %dbn2m, %dac2f, %dbn2z : tensor<128x32768xi1>, tensor<128x32768xf32>
    %dhc2f_dyr = stablehlo.reshape %dbn2 : (tensor<128x32768xf32>) -> tensor<128x32x1024xf32>
    %dhc2f_gb = stablehlo.broadcast_in_dim %g2, dims = [1] : (tensor<32xf32>) -> tensor<128x32x1024xf32>
    %dhc2f_dxh = stablehlo.multiply %dhc2f_gb, %dhc2f_dyr : tensor<128x32x1024xf32>
    %dhc2f_sdxr = stablehlo.reduce(%dhc2f_dxh init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<128x32x1024xf32>, tensor<f32>) -> tensor<128x32xf32>
    %dhc2f_sdx = stablehlo.broadcast_in_dim %dhc2f_sdxr, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x1024xf32>
    %dhc2f_xd = stablehlo.multiply %bn2_xhat, %dhc2f_dxh : tensor<128x32x1024xf32>
    %dhc2f_sxdr = stablehlo.reduce(%dhc2f_xd init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<128x32x1024xf32>, tensor<f32>) -> tensor<128x32xf32>
    %dhc2f_sxd = stablehlo.broadcast_in_dim %dhc2f_sxdr, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x1024xf32>
    %dhc2f_t1 = stablehlo.multiply %dhc2f_dxh, %bn2_nf : tensor<128x32x1024xf32>
    %dhc2f_i1 = stablehlo.subtract %dhc2f_t1, %dhc2f_sdx : tensor<128x32x1024xf32>
    %dhc2f_xs = stablehlo.multiply %bn2_xhat, %dhc2f_sxd : tensor<128x32x1024xf32>
    %dhc2f_i2 = stablehlo.subtract %dhc2f_i1, %dhc2f_xs : tensor<128x32x1024xf32>
    %dhc2f_s = stablehlo.divide %bn2_istd, %bn2_nf : tensor<128x32x1024xf32>
    %dhc2f_dx3 = stablehlo.multiply %dhc2f_s, %dhc2f_i2 : tensor<128x32x1024xf32>
    %dhc2f = stablehlo.reshape %dhc2f_dx3 : (tensor<128x32x1024xf32>) -> tensor<128x32768xf32>
    %dg2_dyr = stablehlo.reshape %dbn2 : (tensor<128x32768xf32>) -> tensor<128x32x1024xf32>
    %dg2_p = stablehlo.multiply %dg2_dyr, %bn2_xhat : tensor<128x32x1024xf32>
    %dg2 = stablehlo.reduce(%dg2_p init: %sc) applies stablehlo.add across dimensions = [0, 2] : (tensor<128x32x1024xf32>, tensor<f32>) -> tensor<32xf32>
    %dbt2 = stablehlo.reduce(%dg2_dyr init: %sc) applies stablehlo.add across dimensions = [0, 2] : (tensor<128x32x1024xf32>, tensor<f32>) -> tensor<32xf32>
    %dhc2 = stablehlo.reshape %dhc2f : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %dac1t = stablehlo.transpose %W2, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %dac1r = stablehlo.reverse %dac1t, dims = [2, 3] : tensor<32x32x3x3xf32>
    %dac1 = stablehlo.convolution(%dhc2, %dac1r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x32x32xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x32x32xf32>
    %dac1f = stablehlo.reshape %dac1 : (tensor<128x32x32x32xf32>) -> tensor<128x32768xf32>
    %dbn1z = stablehlo.constant dense<0.0> : tensor<128x32768xf32>
    %dbn1m = stablehlo.compare GT, %bn1, %dbn1z : (tensor<128x32768xf32>, tensor<128x32768xf32>) -> tensor<128x32768xi1>
    %dbn1 = stablehlo.select %dbn1m, %dac1f, %dbn1z : tensor<128x32768xi1>, tensor<128x32768xf32>
    %dhc1f_dyr = stablehlo.reshape %dbn1 : (tensor<128x32768xf32>) -> tensor<128x32x1024xf32>
    %dhc1f_gb = stablehlo.broadcast_in_dim %g1, dims = [1] : (tensor<32xf32>) -> tensor<128x32x1024xf32>
    %dhc1f_dxh = stablehlo.multiply %dhc1f_gb, %dhc1f_dyr : tensor<128x32x1024xf32>
    %dhc1f_sdxr = stablehlo.reduce(%dhc1f_dxh init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<128x32x1024xf32>, tensor<f32>) -> tensor<128x32xf32>
    %dhc1f_sdx = stablehlo.broadcast_in_dim %dhc1f_sdxr, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x1024xf32>
    %dhc1f_xd = stablehlo.multiply %bn1_xhat, %dhc1f_dxh : tensor<128x32x1024xf32>
    %dhc1f_sxdr = stablehlo.reduce(%dhc1f_xd init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<128x32x1024xf32>, tensor<f32>) -> tensor<128x32xf32>
    %dhc1f_sxd = stablehlo.broadcast_in_dim %dhc1f_sxdr, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x1024xf32>
    %dhc1f_t1 = stablehlo.multiply %dhc1f_dxh, %bn1_nf : tensor<128x32x1024xf32>
    %dhc1f_i1 = stablehlo.subtract %dhc1f_t1, %dhc1f_sdx : tensor<128x32x1024xf32>
    %dhc1f_xs = stablehlo.multiply %bn1_xhat, %dhc1f_sxd : tensor<128x32x1024xf32>
    %dhc1f_i2 = stablehlo.subtract %dhc1f_i1, %dhc1f_xs : tensor<128x32x1024xf32>
    %dhc1f_s = stablehlo.divide %bn1_istd, %bn1_nf : tensor<128x32x1024xf32>
    %dhc1f_dx3 = stablehlo.multiply %dhc1f_s, %dhc1f_i2 : tensor<128x32x1024xf32>
    %dhc1f = stablehlo.reshape %dhc1f_dx3 : (tensor<128x32x1024xf32>) -> tensor<128x32768xf32>
    %dg1_dyr = stablehlo.reshape %dbn1 : (tensor<128x32768xf32>) -> tensor<128x32x1024xf32>
    %dg1_p = stablehlo.multiply %dg1_dyr, %bn1_xhat : tensor<128x32x1024xf32>
    %dg1 = stablehlo.reduce(%dg1_p init: %sc) applies stablehlo.add across dimensions = [0, 2] : (tensor<128x32x1024xf32>, tensor<f32>) -> tensor<32xf32>
    %dbt1 = stablehlo.reduce(%dg1_dyr init: %sc) applies stablehlo.add across dimensions = [0, 2] : (tensor<128x32x1024xf32>, tensor<f32>) -> tensor<32xf32>
    %dhc1 = stablehlo.reshape %dhc1f : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    // ── param grads: dense W/b; conv dW (transpose trick), db (reduce) ──
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
    // ── SGD θ' = θ − lr·∇ (all 22 params, incl. scalar γ/β) ──
    %W1l = stablehlo.constant dense<0.00078125> : tensor<32x3x3x3xf32>
    %W1s = stablehlo.multiply %dW1, %W1l : tensor<32x3x3x3xf32>
    %W1n = stablehlo.subtract %W1, %W1s : tensor<32x3x3x3xf32>
    %b1l = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %b1s = stablehlo.multiply %db1, %b1l : tensor<32xf32>
    %b1n = stablehlo.subtract %b1, %b1s : tensor<32xf32>
    %g1l = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %g1s = stablehlo.multiply %dg1, %g1l : tensor<32xf32>
    %g1n = stablehlo.subtract %g1, %g1s : tensor<32xf32>
    %bt1l = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %bt1s = stablehlo.multiply %dbt1, %bt1l : tensor<32xf32>
    %bt1n = stablehlo.subtract %bt1, %bt1s : tensor<32xf32>
    %W2l = stablehlo.constant dense<0.00078125> : tensor<32x32x3x3xf32>
    %W2s = stablehlo.multiply %dW2, %W2l : tensor<32x32x3x3xf32>
    %W2n = stablehlo.subtract %W2, %W2s : tensor<32x32x3x3xf32>
    %b2l = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %b2s = stablehlo.multiply %db2, %b2l : tensor<32xf32>
    %b2n = stablehlo.subtract %b2, %b2s : tensor<32xf32>
    %g2l = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %g2s = stablehlo.multiply %dg2, %g2l : tensor<32xf32>
    %g2n = stablehlo.subtract %g2, %g2s : tensor<32xf32>
    %bt2l = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %bt2s = stablehlo.multiply %dbt2, %bt2l : tensor<32xf32>
    %bt2n = stablehlo.subtract %bt2, %bt2s : tensor<32xf32>
    %W3l = stablehlo.constant dense<0.00078125> : tensor<64x32x3x3xf32>
    %W3s = stablehlo.multiply %dW3, %W3l : tensor<64x32x3x3xf32>
    %W3n = stablehlo.subtract %W3, %W3s : tensor<64x32x3x3xf32>
    %b3l = stablehlo.constant dense<0.00078125> : tensor<64xf32>
    %b3s = stablehlo.multiply %db3, %b3l : tensor<64xf32>
    %b3n = stablehlo.subtract %b3, %b3s : tensor<64xf32>
    %g3l = stablehlo.constant dense<0.00078125> : tensor<64xf32>
    %g3s = stablehlo.multiply %dg3, %g3l : tensor<64xf32>
    %g3n = stablehlo.subtract %g3, %g3s : tensor<64xf32>
    %bt3l = stablehlo.constant dense<0.00078125> : tensor<64xf32>
    %bt3s = stablehlo.multiply %dbt3, %bt3l : tensor<64xf32>
    %bt3n = stablehlo.subtract %bt3, %bt3s : tensor<64xf32>
    %W4l = stablehlo.constant dense<0.00078125> : tensor<64x64x3x3xf32>
    %W4s = stablehlo.multiply %dW4, %W4l : tensor<64x64x3x3xf32>
    %W4n = stablehlo.subtract %W4, %W4s : tensor<64x64x3x3xf32>
    %b4l = stablehlo.constant dense<0.00078125> : tensor<64xf32>
    %b4s = stablehlo.multiply %db4, %b4l : tensor<64xf32>
    %b4n = stablehlo.subtract %b4, %b4s : tensor<64xf32>
    %g4l = stablehlo.constant dense<0.00078125> : tensor<64xf32>
    %g4s = stablehlo.multiply %dg4, %g4l : tensor<64xf32>
    %g4n = stablehlo.subtract %g4, %g4s : tensor<64xf32>
    %bt4l = stablehlo.constant dense<0.00078125> : tensor<64xf32>
    %bt4s = stablehlo.multiply %dbt4, %bt4l : tensor<64xf32>
    %bt4n = stablehlo.subtract %bt4, %bt4s : tensor<64xf32>
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
    return %W1n, %b1n, %g1n, %bt1n, %W2n, %b2n, %g2n, %bt2n, %W3n, %b3n, %g3n, %bt3n, %W4n, %b4n, %g4n, %bt4n, %W5n, %b5n, %W6n, %b6n, %W7n, %b7n : tensor<32x3x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<64x32x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<4096x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>
  }
}

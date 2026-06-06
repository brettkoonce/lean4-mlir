module @m {
  func.func @resnet_train_step(%x: tensor<128x3072xf32>, %Ws: tensor<32x3x3x3xf32>, %bs: tensor<32xf32>, %gs: tensor<f32>, %bts: tensor<f32>, %W1: tensor<32x32x3x3xf32>, %b1: tensor<32xf32>, %g1: tensor<f32>, %bt1: tensor<f32>, %W2: tensor<32x32x3x3xf32>, %b2: tensor<32xf32>, %g2: tensor<f32>, %bt2: tensor<f32>, %W1p: tensor<64x32x3x3xf32>, %b1p: tensor<64xf32>, %g1p: tensor<f32>, %bt1p: tensor<f32>, %W2p: tensor<64x64x3x3xf32>, %b2p: tensor<64xf32>, %g2p: tensor<f32>, %bt2p: tensor<f32>, %Wp: tensor<64x32x3x3xf32>, %bp: tensor<64xf32>, %gp: tensor<f32>, %btp: tensor<f32>, %Wd: tensor<64x10xf32>, %bd: tensor<10xf32>, %onehot: tensor<128x10xf32>) -> (tensor<32x3x3x3xf32>, tensor<32xf32>, tensor<f32>, tensor<f32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<f32>, tensor<f32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<f32>, tensor<f32>, tensor<64x32x3x3xf32>, tensor<64xf32>, tensor<f32>, tensor<f32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<f32>, tensor<f32>, tensor<64x32x3x3xf32>, tensor<64xf32>, tensor<f32>, tensor<f32>, tensor<64x10xf32>, tensor<10xf32>) {
    %sc = stablehlo.constant dense<0.0> : tensor<f32>
    // ══ forward: convBnRelu(stem) → maxpool → rblk(id) → rblkP(proj) → GAP → dense ══
    %xr = stablehlo.reshape %x : (tensor<128x3072xf32>) -> tensor<128x3x32x32xf32>
    // stem: conv ic→c (H×W, SAME) → BN → relu
    %hcsc = stablehlo.convolution(%xr, %Ws)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x3x32x32xf32>, tensor<32x3x3x3xf32>) -> tensor<128x32x32x32xf32>
    %hcsb = stablehlo.broadcast_in_dim %bs, dims = [1] : (tensor<32xf32>) -> tensor<128x32x32x32xf32>
    %hcs = stablehlo.add %hcsc, %hcsb : tensor<128x32x32x32xf32>
    %hcsf = stablehlo.reshape %hcs : (tensor<128x32x32x32xf32>) -> tensor<128x32768xf32>
    %bns_nf = stablehlo.constant dense<32768.0> : tensor<128x32768xf32>
    %bns_ep = stablehlo.constant dense<1.0e-05> : tensor<128x32768xf32>
    %bns_smr = stablehlo.reduce(%hcsf init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<128x32768xf32>, tensor<f32>) -> tensor<128xf32>
    %bns_sm = stablehlo.broadcast_in_dim %bns_smr, dims = [0] : (tensor<128xf32>) -> tensor<128x32768xf32>
    %bns_mu = stablehlo.divide %bns_sm, %bns_nf : tensor<128x32768xf32>
    %bns_xc = stablehlo.subtract %hcsf, %bns_mu : tensor<128x32768xf32>
    %bns_sq = stablehlo.multiply %bns_xc, %bns_xc : tensor<128x32768xf32>
    %bns_vsr = stablehlo.reduce(%bns_sq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<128x32768xf32>, tensor<f32>) -> tensor<128xf32>
    %bns_vs = stablehlo.broadcast_in_dim %bns_vsr, dims = [0] : (tensor<128xf32>) -> tensor<128x32768xf32>
    %bns_var = stablehlo.divide %bns_vs, %bns_nf : tensor<128x32768xf32>
    %bns_ve = stablehlo.add %bns_var, %bns_ep : tensor<128x32768xf32>
    %bns_istd = stablehlo.rsqrt %bns_ve : tensor<128x32768xf32>
    %bns_xhat = stablehlo.multiply %bns_xc, %bns_istd : tensor<128x32768xf32>
    %bns_gb = stablehlo.broadcast_in_dim %gs, dims = [] : (tensor<f32>) -> tensor<128x32768xf32>
    %bns_bb = stablehlo.broadcast_in_dim %bts, dims = [] : (tensor<f32>) -> tensor<128x32768xf32>
    %bns_gx = stablehlo.multiply %bns_xhat, %bns_gb : tensor<128x32768xf32>
    %bns = stablehlo.add %bns_gx, %bns_bb : tensor<128x32768xf32>
    %acsfz = stablehlo.constant dense<0.0> : tensor<128x32768xf32>
    %acsf = stablehlo.maximum %bns, %acsfz : tensor<128x32768xf32>
    %acs = stablehlo.reshape %acsf : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %poolninf = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %pool = "stablehlo.reduce_window"(%acs, %poolninf) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x32x32xf32>, tensor<f32>) -> tensor<128x32x16x16xf32>
    // identity block: relu( bn2(conv2(relu(bn1(conv1(pool))))) + pool )
    %hc1c = stablehlo.convolution(%pool, %W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x16x16xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x16x16xf32>
    %hc1b = stablehlo.broadcast_in_dim %b1, dims = [1] : (tensor<32xf32>) -> tensor<128x32x16x16xf32>
    %hc1 = stablehlo.add %hc1c, %hc1b : tensor<128x32x16x16xf32>
    %hc1f = stablehlo.reshape %hc1 : (tensor<128x32x16x16xf32>) -> tensor<128x8192xf32>
    %bn1_nf = stablehlo.constant dense<8192.0> : tensor<128x8192xf32>
    %bn1_ep = stablehlo.constant dense<1.0e-05> : tensor<128x8192xf32>
    %bn1_smr = stablehlo.reduce(%hc1f init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<128x8192xf32>, tensor<f32>) -> tensor<128xf32>
    %bn1_sm = stablehlo.broadcast_in_dim %bn1_smr, dims = [0] : (tensor<128xf32>) -> tensor<128x8192xf32>
    %bn1_mu = stablehlo.divide %bn1_sm, %bn1_nf : tensor<128x8192xf32>
    %bn1_xc = stablehlo.subtract %hc1f, %bn1_mu : tensor<128x8192xf32>
    %bn1_sq = stablehlo.multiply %bn1_xc, %bn1_xc : tensor<128x8192xf32>
    %bn1_vsr = stablehlo.reduce(%bn1_sq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<128x8192xf32>, tensor<f32>) -> tensor<128xf32>
    %bn1_vs = stablehlo.broadcast_in_dim %bn1_vsr, dims = [0] : (tensor<128xf32>) -> tensor<128x8192xf32>
    %bn1_var = stablehlo.divide %bn1_vs, %bn1_nf : tensor<128x8192xf32>
    %bn1_ve = stablehlo.add %bn1_var, %bn1_ep : tensor<128x8192xf32>
    %bn1_istd = stablehlo.rsqrt %bn1_ve : tensor<128x8192xf32>
    %bn1_xhat = stablehlo.multiply %bn1_xc, %bn1_istd : tensor<128x8192xf32>
    %bn1_gb = stablehlo.broadcast_in_dim %g1, dims = [] : (tensor<f32>) -> tensor<128x8192xf32>
    %bn1_bb = stablehlo.broadcast_in_dim %bt1, dims = [] : (tensor<f32>) -> tensor<128x8192xf32>
    %bn1_gx = stablehlo.multiply %bn1_xhat, %bn1_gb : tensor<128x8192xf32>
    %bn1 = stablehlo.add %bn1_gx, %bn1_bb : tensor<128x8192xf32>
    %ac1fz = stablehlo.constant dense<0.0> : tensor<128x8192xf32>
    %ac1f = stablehlo.maximum %bn1, %ac1fz : tensor<128x8192xf32>
    %ac1 = stablehlo.reshape %ac1f : (tensor<128x8192xf32>) -> tensor<128x32x16x16xf32>
    %hc2c = stablehlo.convolution(%ac1, %W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x16x16xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x16x16xf32>
    %hc2b = stablehlo.broadcast_in_dim %b2, dims = [1] : (tensor<32xf32>) -> tensor<128x32x16x16xf32>
    %hc2 = stablehlo.add %hc2c, %hc2b : tensor<128x32x16x16xf32>
    %hc2f = stablehlo.reshape %hc2 : (tensor<128x32x16x16xf32>) -> tensor<128x8192xf32>
    %bn2_nf = stablehlo.constant dense<8192.0> : tensor<128x8192xf32>
    %bn2_ep = stablehlo.constant dense<1.0e-05> : tensor<128x8192xf32>
    %bn2_smr = stablehlo.reduce(%hc2f init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<128x8192xf32>, tensor<f32>) -> tensor<128xf32>
    %bn2_sm = stablehlo.broadcast_in_dim %bn2_smr, dims = [0] : (tensor<128xf32>) -> tensor<128x8192xf32>
    %bn2_mu = stablehlo.divide %bn2_sm, %bn2_nf : tensor<128x8192xf32>
    %bn2_xc = stablehlo.subtract %hc2f, %bn2_mu : tensor<128x8192xf32>
    %bn2_sq = stablehlo.multiply %bn2_xc, %bn2_xc : tensor<128x8192xf32>
    %bn2_vsr = stablehlo.reduce(%bn2_sq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<128x8192xf32>, tensor<f32>) -> tensor<128xf32>
    %bn2_vs = stablehlo.broadcast_in_dim %bn2_vsr, dims = [0] : (tensor<128xf32>) -> tensor<128x8192xf32>
    %bn2_var = stablehlo.divide %bn2_vs, %bn2_nf : tensor<128x8192xf32>
    %bn2_ve = stablehlo.add %bn2_var, %bn2_ep : tensor<128x8192xf32>
    %bn2_istd = stablehlo.rsqrt %bn2_ve : tensor<128x8192xf32>
    %bn2_xhat = stablehlo.multiply %bn2_xc, %bn2_istd : tensor<128x8192xf32>
    %bn2_gb = stablehlo.broadcast_in_dim %g2, dims = [] : (tensor<f32>) -> tensor<128x8192xf32>
    %bn2_bb = stablehlo.broadcast_in_dim %bt2, dims = [] : (tensor<f32>) -> tensor<128x8192xf32>
    %bn2_gx = stablehlo.multiply %bn2_xhat, %bn2_gb : tensor<128x8192xf32>
    %bn2 = stablehlo.add %bn2_gx, %bn2_bb : tensor<128x8192xf32>
    %poolf = stablehlo.reshape %pool : (tensor<128x32x16x16xf32>) -> tensor<128x8192xf32>
    %radd = stablehlo.add %bn2, %poolf : tensor<128x8192xf32>
    %rblkz = stablehlo.constant dense<0.0> : tensor<128x8192xf32>
    %rblk = stablehlo.maximum %radd, %rblkz : tensor<128x8192xf32>
    %rblkr = stablehlo.reshape %rblk : (tensor<128x8192xf32>) -> tensor<128x32x16x16xf32>
    // projection block: relu( proj(rblk) + bn2p(conv2p(relu(bn1p(conv1p(rblk))))) ),  c→oc
    %hc1pc = stablehlo.convolution(%rblkr, %W1p)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x16x16xf32>, tensor<64x32x3x3xf32>) -> tensor<128x64x16x16xf32>
    %hc1pb = stablehlo.broadcast_in_dim %b1p, dims = [1] : (tensor<64xf32>) -> tensor<128x64x16x16xf32>
    %hc1p = stablehlo.add %hc1pc, %hc1pb : tensor<128x64x16x16xf32>
    %hc1pf = stablehlo.reshape %hc1p : (tensor<128x64x16x16xf32>) -> tensor<128x16384xf32>
    %bn1p_nf = stablehlo.constant dense<16384.0> : tensor<128x16384xf32>
    %bn1p_ep = stablehlo.constant dense<1.0e-05> : tensor<128x16384xf32>
    %bn1p_smr = stablehlo.reduce(%hc1pf init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<128x16384xf32>, tensor<f32>) -> tensor<128xf32>
    %bn1p_sm = stablehlo.broadcast_in_dim %bn1p_smr, dims = [0] : (tensor<128xf32>) -> tensor<128x16384xf32>
    %bn1p_mu = stablehlo.divide %bn1p_sm, %bn1p_nf : tensor<128x16384xf32>
    %bn1p_xc = stablehlo.subtract %hc1pf, %bn1p_mu : tensor<128x16384xf32>
    %bn1p_sq = stablehlo.multiply %bn1p_xc, %bn1p_xc : tensor<128x16384xf32>
    %bn1p_vsr = stablehlo.reduce(%bn1p_sq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<128x16384xf32>, tensor<f32>) -> tensor<128xf32>
    %bn1p_vs = stablehlo.broadcast_in_dim %bn1p_vsr, dims = [0] : (tensor<128xf32>) -> tensor<128x16384xf32>
    %bn1p_var = stablehlo.divide %bn1p_vs, %bn1p_nf : tensor<128x16384xf32>
    %bn1p_ve = stablehlo.add %bn1p_var, %bn1p_ep : tensor<128x16384xf32>
    %bn1p_istd = stablehlo.rsqrt %bn1p_ve : tensor<128x16384xf32>
    %bn1p_xhat = stablehlo.multiply %bn1p_xc, %bn1p_istd : tensor<128x16384xf32>
    %bn1p_gb = stablehlo.broadcast_in_dim %g1p, dims = [] : (tensor<f32>) -> tensor<128x16384xf32>
    %bn1p_bb = stablehlo.broadcast_in_dim %bt1p, dims = [] : (tensor<f32>) -> tensor<128x16384xf32>
    %bn1p_gx = stablehlo.multiply %bn1p_xhat, %bn1p_gb : tensor<128x16384xf32>
    %bn1p = stablehlo.add %bn1p_gx, %bn1p_bb : tensor<128x16384xf32>
    %ac1pfz = stablehlo.constant dense<0.0> : tensor<128x16384xf32>
    %ac1pf = stablehlo.maximum %bn1p, %ac1pfz : tensor<128x16384xf32>
    %ac1p = stablehlo.reshape %ac1pf : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %hc2pc = stablehlo.convolution(%ac1p, %W2p)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x16x16xf32>, tensor<64x64x3x3xf32>) -> tensor<128x64x16x16xf32>
    %hc2pb = stablehlo.broadcast_in_dim %b2p, dims = [1] : (tensor<64xf32>) -> tensor<128x64x16x16xf32>
    %hc2p = stablehlo.add %hc2pc, %hc2pb : tensor<128x64x16x16xf32>
    %hc2pf = stablehlo.reshape %hc2p : (tensor<128x64x16x16xf32>) -> tensor<128x16384xf32>
    %bn2p_nf = stablehlo.constant dense<16384.0> : tensor<128x16384xf32>
    %bn2p_ep = stablehlo.constant dense<1.0e-05> : tensor<128x16384xf32>
    %bn2p_smr = stablehlo.reduce(%hc2pf init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<128x16384xf32>, tensor<f32>) -> tensor<128xf32>
    %bn2p_sm = stablehlo.broadcast_in_dim %bn2p_smr, dims = [0] : (tensor<128xf32>) -> tensor<128x16384xf32>
    %bn2p_mu = stablehlo.divide %bn2p_sm, %bn2p_nf : tensor<128x16384xf32>
    %bn2p_xc = stablehlo.subtract %hc2pf, %bn2p_mu : tensor<128x16384xf32>
    %bn2p_sq = stablehlo.multiply %bn2p_xc, %bn2p_xc : tensor<128x16384xf32>
    %bn2p_vsr = stablehlo.reduce(%bn2p_sq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<128x16384xf32>, tensor<f32>) -> tensor<128xf32>
    %bn2p_vs = stablehlo.broadcast_in_dim %bn2p_vsr, dims = [0] : (tensor<128xf32>) -> tensor<128x16384xf32>
    %bn2p_var = stablehlo.divide %bn2p_vs, %bn2p_nf : tensor<128x16384xf32>
    %bn2p_ve = stablehlo.add %bn2p_var, %bn2p_ep : tensor<128x16384xf32>
    %bn2p_istd = stablehlo.rsqrt %bn2p_ve : tensor<128x16384xf32>
    %bn2p_xhat = stablehlo.multiply %bn2p_xc, %bn2p_istd : tensor<128x16384xf32>
    %bn2p_gb = stablehlo.broadcast_in_dim %g2p, dims = [] : (tensor<f32>) -> tensor<128x16384xf32>
    %bn2p_bb = stablehlo.broadcast_in_dim %bt2p, dims = [] : (tensor<f32>) -> tensor<128x16384xf32>
    %bn2p_gx = stablehlo.multiply %bn2p_xhat, %bn2p_gb : tensor<128x16384xf32>
    %bn2p = stablehlo.add %bn2p_gx, %bn2p_bb : tensor<128x16384xf32>
    %hcppc = stablehlo.convolution(%rblkr, %Wp)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x16x16xf32>, tensor<64x32x3x3xf32>) -> tensor<128x64x16x16xf32>
    %hcppb = stablehlo.broadcast_in_dim %bp, dims = [1] : (tensor<64xf32>) -> tensor<128x64x16x16xf32>
    %hcpp = stablehlo.add %hcppc, %hcppb : tensor<128x64x16x16xf32>
    %hcppf = stablehlo.reshape %hcpp : (tensor<128x64x16x16xf32>) -> tensor<128x16384xf32>
    %bnp_nf = stablehlo.constant dense<16384.0> : tensor<128x16384xf32>
    %bnp_ep = stablehlo.constant dense<1.0e-05> : tensor<128x16384xf32>
    %bnp_smr = stablehlo.reduce(%hcppf init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<128x16384xf32>, tensor<f32>) -> tensor<128xf32>
    %bnp_sm = stablehlo.broadcast_in_dim %bnp_smr, dims = [0] : (tensor<128xf32>) -> tensor<128x16384xf32>
    %bnp_mu = stablehlo.divide %bnp_sm, %bnp_nf : tensor<128x16384xf32>
    %bnp_xc = stablehlo.subtract %hcppf, %bnp_mu : tensor<128x16384xf32>
    %bnp_sq = stablehlo.multiply %bnp_xc, %bnp_xc : tensor<128x16384xf32>
    %bnp_vsr = stablehlo.reduce(%bnp_sq init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<128x16384xf32>, tensor<f32>) -> tensor<128xf32>
    %bnp_vs = stablehlo.broadcast_in_dim %bnp_vsr, dims = [0] : (tensor<128xf32>) -> tensor<128x16384xf32>
    %bnp_var = stablehlo.divide %bnp_vs, %bnp_nf : tensor<128x16384xf32>
    %bnp_ve = stablehlo.add %bnp_var, %bnp_ep : tensor<128x16384xf32>
    %bnp_istd = stablehlo.rsqrt %bnp_ve : tensor<128x16384xf32>
    %bnp_xhat = stablehlo.multiply %bnp_xc, %bnp_istd : tensor<128x16384xf32>
    %bnp_gb = stablehlo.broadcast_in_dim %gp, dims = [] : (tensor<f32>) -> tensor<128x16384xf32>
    %bnp_bb = stablehlo.broadcast_in_dim %btp, dims = [] : (tensor<f32>) -> tensor<128x16384xf32>
    %bnp_gx = stablehlo.multiply %bnp_xhat, %bnp_gb : tensor<128x16384xf32>
    %bnp = stablehlo.add %bnp_gx, %bnp_bb : tensor<128x16384xf32>
    %padd = stablehlo.add %bnp, %bn2p : tensor<128x16384xf32>
    %rblkpz = stablehlo.constant dense<0.0> : tensor<128x16384xf32>
    %rblkp = stablehlo.maximum %padd, %rblkpz : tensor<128x16384xf32>
    %rblkpr = stablehlo.reshape %rblkp : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    // GAP (mean over H2×W2) → dense oc→nClasses
    %gapr = stablehlo.reduce(%rblkpr init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x64x16x16xf32>, tensor<f32>) -> tensor<128x64xf32>
    %gapnf = stablehlo.constant dense<256.0> : tensor<128x64xf32>
    %gap = stablehlo.divide %gapr, %gapnf : tensor<128x64xf32>
    %logitsd = stablehlo.dot_general %gap, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<64x10xf32>) -> tensor<128x10xf32>
    %logitsb = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<10xf32>) -> tensor<128x10xf32>
    %logits = stablehlo.add %logitsd, %logitsb : tensor<128x10xf32>
    // ── loss cotangent dy = softmax(logits) − onehot ──
    %le = stablehlo.exponential %logits : tensor<128x10xf32>
    %lsum = stablehlo.reduce(%le init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<128x10xf32>, tensor<f32>) -> tensor<128xf32>
    %lsb = stablehlo.broadcast_in_dim %lsum, dims = [0] : (tensor<128xf32>) -> tensor<128x10xf32>
    %lsm = stablehlo.divide %le, %lsb : tensor<128x10xf32>
    %dy = stablehlo.subtract %lsm, %onehot : tensor<128x10xf32>
    // ══ backward ══
    // dense back + GAP back (broadcast dy/(H2·W2) over spatial)
    %dgap = stablehlo.dot_general %dy, %Wd, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x10xf32>, tensor<64x10xf32>) -> tensor<128x64xf32>
    %dgapinnf = stablehlo.constant dense<256.0> : tensor<128x64xf32>
    %dgapins = stablehlo.divide %dgap, %dgapinnf : tensor<128x64xf32>
    %dgapinb = stablehlo.broadcast_in_dim %dgapins, dims = [0, 1] : (tensor<128x64xf32>) -> tensor<128x64x16x16xf32>
    %dgapin = stablehlo.reshape %dgapinb : (tensor<128x64x16x16xf32>) -> tensor<128x16384xf32>
    // projection block back: relu mask, then fan-IN dRblk = projBack + FBack
    %drblkpz = stablehlo.constant dense<0.0> : tensor<128x16384xf32>
    %drblkpm = stablehlo.compare GT, %padd, %drblkpz : (tensor<128x16384xf32>, tensor<128x16384xf32>) -> tensor<128x16384xi1>
    %drblkp = stablehlo.select %drblkpm, %dgapin, %drblkpz : tensor<128x16384xi1>, tensor<128x16384xf32>
    %dhcppf_gb = stablehlo.broadcast_in_dim %gp, dims = [] : (tensor<f32>) -> tensor<128x16384xf32>
    %dhcppf_dxh = stablehlo.multiply %dhcppf_gb, %drblkp : tensor<128x16384xf32>
    %dhcppf_sdxr = stablehlo.reduce(%dhcppf_dxh init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<128x16384xf32>, tensor<f32>) -> tensor<128xf32>
    %dhcppf_sdx = stablehlo.broadcast_in_dim %dhcppf_sdxr, dims = [0] : (tensor<128xf32>) -> tensor<128x16384xf32>
    %dhcppf_xd = stablehlo.multiply %bnp_xhat, %dhcppf_dxh : tensor<128x16384xf32>
    %dhcppf_sxdr = stablehlo.reduce(%dhcppf_xd init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<128x16384xf32>, tensor<f32>) -> tensor<128xf32>
    %dhcppf_sxd = stablehlo.broadcast_in_dim %dhcppf_sxdr, dims = [0] : (tensor<128xf32>) -> tensor<128x16384xf32>
    %dhcppf_t1 = stablehlo.multiply %dhcppf_dxh, %bnp_nf : tensor<128x16384xf32>
    %dhcppf_i1 = stablehlo.subtract %dhcppf_t1, %dhcppf_sdx : tensor<128x16384xf32>
    %dhcppf_xs = stablehlo.multiply %bnp_xhat, %dhcppf_sxd : tensor<128x16384xf32>
    %dhcppf_i2 = stablehlo.subtract %dhcppf_i1, %dhcppf_xs : tensor<128x16384xf32>
    %dhcppf_s = stablehlo.divide %bnp_istd, %bnp_nf : tensor<128x16384xf32>
    %dhcppf = stablehlo.multiply %dhcppf_s, %dhcppf_i2 : tensor<128x16384xf32>
    %dgp_p = stablehlo.multiply %drblkp, %bnp_xhat : tensor<128x16384xf32>
    %dgp = stablehlo.reduce(%dgp_p init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<128x16384xf32>, tensor<f32>) -> tensor<f32>
    %dbtp = stablehlo.reduce(%drblkp init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<128x16384xf32>, tensor<f32>) -> tensor<f32>
    %dhcpp = stablehlo.reshape %dhcppf : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %dprojint = stablehlo.transpose %Wp, dims = [1, 0, 2, 3] : (tensor<64x32x3x3xf32>) -> tensor<32x64x3x3xf32>
    %dprojinr = stablehlo.reverse %dprojint, dims = [2, 3] : tensor<32x64x3x3xf32>
    %dprojin = stablehlo.convolution(%dhcpp, %dprojinr)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x16x16xf32>, tensor<32x64x3x3xf32>) -> tensor<128x32x16x16xf32>
    %dhc2pf_gb = stablehlo.broadcast_in_dim %g2p, dims = [] : (tensor<f32>) -> tensor<128x16384xf32>
    %dhc2pf_dxh = stablehlo.multiply %dhc2pf_gb, %drblkp : tensor<128x16384xf32>
    %dhc2pf_sdxr = stablehlo.reduce(%dhc2pf_dxh init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<128x16384xf32>, tensor<f32>) -> tensor<128xf32>
    %dhc2pf_sdx = stablehlo.broadcast_in_dim %dhc2pf_sdxr, dims = [0] : (tensor<128xf32>) -> tensor<128x16384xf32>
    %dhc2pf_xd = stablehlo.multiply %bn2p_xhat, %dhc2pf_dxh : tensor<128x16384xf32>
    %dhc2pf_sxdr = stablehlo.reduce(%dhc2pf_xd init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<128x16384xf32>, tensor<f32>) -> tensor<128xf32>
    %dhc2pf_sxd = stablehlo.broadcast_in_dim %dhc2pf_sxdr, dims = [0] : (tensor<128xf32>) -> tensor<128x16384xf32>
    %dhc2pf_t1 = stablehlo.multiply %dhc2pf_dxh, %bn2p_nf : tensor<128x16384xf32>
    %dhc2pf_i1 = stablehlo.subtract %dhc2pf_t1, %dhc2pf_sdx : tensor<128x16384xf32>
    %dhc2pf_xs = stablehlo.multiply %bn2p_xhat, %dhc2pf_sxd : tensor<128x16384xf32>
    %dhc2pf_i2 = stablehlo.subtract %dhc2pf_i1, %dhc2pf_xs : tensor<128x16384xf32>
    %dhc2pf_s = stablehlo.divide %bn2p_istd, %bn2p_nf : tensor<128x16384xf32>
    %dhc2pf = stablehlo.multiply %dhc2pf_s, %dhc2pf_i2 : tensor<128x16384xf32>
    %dg2p_p = stablehlo.multiply %drblkp, %bn2p_xhat : tensor<128x16384xf32>
    %dg2p = stablehlo.reduce(%dg2p_p init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<128x16384xf32>, tensor<f32>) -> tensor<f32>
    %dbt2p = stablehlo.reduce(%drblkp init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<128x16384xf32>, tensor<f32>) -> tensor<f32>
    %dhc2p = stablehlo.reshape %dhc2pf : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %dac1pt = stablehlo.transpose %W2p, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %dac1pr = stablehlo.reverse %dac1pt, dims = [2, 3] : tensor<64x64x3x3xf32>
    %dac1p = stablehlo.convolution(%dhc2p, %dac1pr)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x16x16xf32>, tensor<64x64x3x3xf32>) -> tensor<128x64x16x16xf32>
    %dac1pf = stablehlo.reshape %dac1p : (tensor<128x64x16x16xf32>) -> tensor<128x16384xf32>
    %dbn1pz = stablehlo.constant dense<0.0> : tensor<128x16384xf32>
    %dbn1pm = stablehlo.compare GT, %bn1p, %dbn1pz : (tensor<128x16384xf32>, tensor<128x16384xf32>) -> tensor<128x16384xi1>
    %dbn1p = stablehlo.select %dbn1pm, %dac1pf, %dbn1pz : tensor<128x16384xi1>, tensor<128x16384xf32>
    %dhc1pf_gb = stablehlo.broadcast_in_dim %g1p, dims = [] : (tensor<f32>) -> tensor<128x16384xf32>
    %dhc1pf_dxh = stablehlo.multiply %dhc1pf_gb, %dbn1p : tensor<128x16384xf32>
    %dhc1pf_sdxr = stablehlo.reduce(%dhc1pf_dxh init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<128x16384xf32>, tensor<f32>) -> tensor<128xf32>
    %dhc1pf_sdx = stablehlo.broadcast_in_dim %dhc1pf_sdxr, dims = [0] : (tensor<128xf32>) -> tensor<128x16384xf32>
    %dhc1pf_xd = stablehlo.multiply %bn1p_xhat, %dhc1pf_dxh : tensor<128x16384xf32>
    %dhc1pf_sxdr = stablehlo.reduce(%dhc1pf_xd init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<128x16384xf32>, tensor<f32>) -> tensor<128xf32>
    %dhc1pf_sxd = stablehlo.broadcast_in_dim %dhc1pf_sxdr, dims = [0] : (tensor<128xf32>) -> tensor<128x16384xf32>
    %dhc1pf_t1 = stablehlo.multiply %dhc1pf_dxh, %bn1p_nf : tensor<128x16384xf32>
    %dhc1pf_i1 = stablehlo.subtract %dhc1pf_t1, %dhc1pf_sdx : tensor<128x16384xf32>
    %dhc1pf_xs = stablehlo.multiply %bn1p_xhat, %dhc1pf_sxd : tensor<128x16384xf32>
    %dhc1pf_i2 = stablehlo.subtract %dhc1pf_i1, %dhc1pf_xs : tensor<128x16384xf32>
    %dhc1pf_s = stablehlo.divide %bn1p_istd, %bn1p_nf : tensor<128x16384xf32>
    %dhc1pf = stablehlo.multiply %dhc1pf_s, %dhc1pf_i2 : tensor<128x16384xf32>
    %dg1p_p = stablehlo.multiply %dbn1p, %bn1p_xhat : tensor<128x16384xf32>
    %dg1p = stablehlo.reduce(%dg1p_p init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<128x16384xf32>, tensor<f32>) -> tensor<f32>
    %dbt1p = stablehlo.reduce(%dbn1p init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<128x16384xf32>, tensor<f32>) -> tensor<f32>
    %dhc1p = stablehlo.reshape %dhc1pf : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %dFint = stablehlo.transpose %W1p, dims = [1, 0, 2, 3] : (tensor<64x32x3x3xf32>) -> tensor<32x64x3x3xf32>
    %dFinr = stablehlo.reverse %dFint, dims = [2, 3] : tensor<32x64x3x3xf32>
    %dFin = stablehlo.convolution(%dhc1p, %dFinr)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x16x16xf32>, tensor<32x64x3x3xf32>) -> tensor<128x32x16x16xf32>
    %drblkr = stablehlo.add %dprojin, %dFin : tensor<128x32x16x16xf32>
    %drblkrf = stablehlo.reshape %drblkr : (tensor<128x32x16x16xf32>) -> tensor<128x8192xf32>
    // identity block back: relu mask, then fan-IN dPool = FBack + skip
    %drblkz = stablehlo.constant dense<0.0> : tensor<128x8192xf32>
    %drblkm = stablehlo.compare GT, %radd, %drblkz : (tensor<128x8192xf32>, tensor<128x8192xf32>) -> tensor<128x8192xi1>
    %drblk = stablehlo.select %drblkm, %drblkrf, %drblkz : tensor<128x8192xi1>, tensor<128x8192xf32>
    %dhc2f_gb = stablehlo.broadcast_in_dim %g2, dims = [] : (tensor<f32>) -> tensor<128x8192xf32>
    %dhc2f_dxh = stablehlo.multiply %dhc2f_gb, %drblk : tensor<128x8192xf32>
    %dhc2f_sdxr = stablehlo.reduce(%dhc2f_dxh init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<128x8192xf32>, tensor<f32>) -> tensor<128xf32>
    %dhc2f_sdx = stablehlo.broadcast_in_dim %dhc2f_sdxr, dims = [0] : (tensor<128xf32>) -> tensor<128x8192xf32>
    %dhc2f_xd = stablehlo.multiply %bn2_xhat, %dhc2f_dxh : tensor<128x8192xf32>
    %dhc2f_sxdr = stablehlo.reduce(%dhc2f_xd init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<128x8192xf32>, tensor<f32>) -> tensor<128xf32>
    %dhc2f_sxd = stablehlo.broadcast_in_dim %dhc2f_sxdr, dims = [0] : (tensor<128xf32>) -> tensor<128x8192xf32>
    %dhc2f_t1 = stablehlo.multiply %dhc2f_dxh, %bn2_nf : tensor<128x8192xf32>
    %dhc2f_i1 = stablehlo.subtract %dhc2f_t1, %dhc2f_sdx : tensor<128x8192xf32>
    %dhc2f_xs = stablehlo.multiply %bn2_xhat, %dhc2f_sxd : tensor<128x8192xf32>
    %dhc2f_i2 = stablehlo.subtract %dhc2f_i1, %dhc2f_xs : tensor<128x8192xf32>
    %dhc2f_s = stablehlo.divide %bn2_istd, %bn2_nf : tensor<128x8192xf32>
    %dhc2f = stablehlo.multiply %dhc2f_s, %dhc2f_i2 : tensor<128x8192xf32>
    %dg2_p = stablehlo.multiply %drblk, %bn2_xhat : tensor<128x8192xf32>
    %dg2 = stablehlo.reduce(%dg2_p init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<128x8192xf32>, tensor<f32>) -> tensor<f32>
    %dbt2 = stablehlo.reduce(%drblk init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<128x8192xf32>, tensor<f32>) -> tensor<f32>
    %dhc2 = stablehlo.reshape %dhc2f : (tensor<128x8192xf32>) -> tensor<128x32x16x16xf32>
    %dac1t = stablehlo.transpose %W2, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %dac1r = stablehlo.reverse %dac1t, dims = [2, 3] : tensor<32x32x3x3xf32>
    %dac1 = stablehlo.convolution(%dhc2, %dac1r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x16x16xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x16x16xf32>
    %dac1f = stablehlo.reshape %dac1 : (tensor<128x32x16x16xf32>) -> tensor<128x8192xf32>
    %dbn1z = stablehlo.constant dense<0.0> : tensor<128x8192xf32>
    %dbn1m = stablehlo.compare GT, %bn1, %dbn1z : (tensor<128x8192xf32>, tensor<128x8192xf32>) -> tensor<128x8192xi1>
    %dbn1 = stablehlo.select %dbn1m, %dac1f, %dbn1z : tensor<128x8192xi1>, tensor<128x8192xf32>
    %dhc1f_gb = stablehlo.broadcast_in_dim %g1, dims = [] : (tensor<f32>) -> tensor<128x8192xf32>
    %dhc1f_dxh = stablehlo.multiply %dhc1f_gb, %dbn1 : tensor<128x8192xf32>
    %dhc1f_sdxr = stablehlo.reduce(%dhc1f_dxh init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<128x8192xf32>, tensor<f32>) -> tensor<128xf32>
    %dhc1f_sdx = stablehlo.broadcast_in_dim %dhc1f_sdxr, dims = [0] : (tensor<128xf32>) -> tensor<128x8192xf32>
    %dhc1f_xd = stablehlo.multiply %bn1_xhat, %dhc1f_dxh : tensor<128x8192xf32>
    %dhc1f_sxdr = stablehlo.reduce(%dhc1f_xd init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<128x8192xf32>, tensor<f32>) -> tensor<128xf32>
    %dhc1f_sxd = stablehlo.broadcast_in_dim %dhc1f_sxdr, dims = [0] : (tensor<128xf32>) -> tensor<128x8192xf32>
    %dhc1f_t1 = stablehlo.multiply %dhc1f_dxh, %bn1_nf : tensor<128x8192xf32>
    %dhc1f_i1 = stablehlo.subtract %dhc1f_t1, %dhc1f_sdx : tensor<128x8192xf32>
    %dhc1f_xs = stablehlo.multiply %bn1_xhat, %dhc1f_sxd : tensor<128x8192xf32>
    %dhc1f_i2 = stablehlo.subtract %dhc1f_i1, %dhc1f_xs : tensor<128x8192xf32>
    %dhc1f_s = stablehlo.divide %bn1_istd, %bn1_nf : tensor<128x8192xf32>
    %dhc1f = stablehlo.multiply %dhc1f_s, %dhc1f_i2 : tensor<128x8192xf32>
    %dg1_p = stablehlo.multiply %dbn1, %bn1_xhat : tensor<128x8192xf32>
    %dg1 = stablehlo.reduce(%dg1_p init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<128x8192xf32>, tensor<f32>) -> tensor<f32>
    %dbt1 = stablehlo.reduce(%dbn1 init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<128x8192xf32>, tensor<f32>) -> tensor<f32>
    %dhc1 = stablehlo.reshape %dhc1f : (tensor<128x8192xf32>) -> tensor<128x32x16x16xf32>
    %dpoolFt = stablehlo.transpose %W1, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %dpoolFr = stablehlo.reverse %dpoolFt, dims = [2, 3] : tensor<32x32x3x3xf32>
    %dpoolF = stablehlo.convolution(%dhc1, %dpoolFr)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x16x16xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x16x16xf32>
    %dskip = stablehlo.reshape %drblk : (tensor<128x8192xf32>) -> tensor<128x32x16x16xf32>
    %dpool = stablehlo.add %dpoolF, %dskip : tensor<128x32x16x16xf32>
    // maxpool back (scatter into H×W) → stem relu mask → stem BN back
    %dacs = "stablehlo.select_and_scatter"(%acs, %dpool, %sc) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%su: tensor<f32>, %sv: tensor<f32>):
        %ss = stablehlo.add %su, %sv : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x32x32xf32>, tensor<128x32x16x16xf32>, tensor<f32>) -> tensor<128x32x32x32xf32>
    %dacsf = stablehlo.reshape %dacs : (tensor<128x32x32x32xf32>) -> tensor<128x32768xf32>
    %dbnsz = stablehlo.constant dense<0.0> : tensor<128x32768xf32>
    %dbnsm = stablehlo.compare GT, %bns, %dbnsz : (tensor<128x32768xf32>, tensor<128x32768xf32>) -> tensor<128x32768xi1>
    %dbns = stablehlo.select %dbnsm, %dacsf, %dbnsz : tensor<128x32768xi1>, tensor<128x32768xf32>
    %dhcsf_gb = stablehlo.broadcast_in_dim %gs, dims = [] : (tensor<f32>) -> tensor<128x32768xf32>
    %dhcsf_dxh = stablehlo.multiply %dhcsf_gb, %dbns : tensor<128x32768xf32>
    %dhcsf_sdxr = stablehlo.reduce(%dhcsf_dxh init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<128x32768xf32>, tensor<f32>) -> tensor<128xf32>
    %dhcsf_sdx = stablehlo.broadcast_in_dim %dhcsf_sdxr, dims = [0] : (tensor<128xf32>) -> tensor<128x32768xf32>
    %dhcsf_xd = stablehlo.multiply %bns_xhat, %dhcsf_dxh : tensor<128x32768xf32>
    %dhcsf_sxdr = stablehlo.reduce(%dhcsf_xd init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<128x32768xf32>, tensor<f32>) -> tensor<128xf32>
    %dhcsf_sxd = stablehlo.broadcast_in_dim %dhcsf_sxdr, dims = [0] : (tensor<128xf32>) -> tensor<128x32768xf32>
    %dhcsf_t1 = stablehlo.multiply %dhcsf_dxh, %bns_nf : tensor<128x32768xf32>
    %dhcsf_i1 = stablehlo.subtract %dhcsf_t1, %dhcsf_sdx : tensor<128x32768xf32>
    %dhcsf_xs = stablehlo.multiply %bns_xhat, %dhcsf_sxd : tensor<128x32768xf32>
    %dhcsf_i2 = stablehlo.subtract %dhcsf_i1, %dhcsf_xs : tensor<128x32768xf32>
    %dhcsf_s = stablehlo.divide %bns_istd, %bns_nf : tensor<128x32768xf32>
    %dhcsf = stablehlo.multiply %dhcsf_s, %dhcsf_i2 : tensor<128x32768xf32>
    %dgs_p = stablehlo.multiply %dbns, %bns_xhat : tensor<128x32768xf32>
    %dgs = stablehlo.reduce(%dgs_p init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<128x32768xf32>, tensor<f32>) -> tensor<f32>
    %dbts = stablehlo.reduce(%dbns init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<128x32768xf32>, tensor<f32>) -> tensor<f32>
    %dhcs = stablehlo.reshape %dhcsf : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    // ── param grads: dense; conv dW (transpose trick) + db (reduce) ──
    %dWd = stablehlo.dot_general %gap, %dy, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<128x10xf32>) -> tensor<64x10xf32>
    %dbd = stablehlo.reduce(%dy init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<128x10xf32>, tensor<f32>) -> tensor<10xf32>
    %dWsxt = stablehlo.transpose %xr, dims = [1, 0, 2, 3] : (tensor<128x3x32x32xf32>) -> tensor<3x128x32x32xf32>
    %dWsdt = stablehlo.transpose %dhcs, dims = [1, 0, 2, 3] : (tensor<128x32x32x32xf32>) -> tensor<32x128x32x32xf32>
    %dWsraw = stablehlo.convolution(%dWsxt, %dWsdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x128x32x32xf32>, tensor<32x128x32x32xf32>) -> tensor<3x32x3x3xf32>
    %dWs = stablehlo.transpose %dWsraw, dims = [1, 0, 2, 3] : (tensor<3x32x3x3xf32>) -> tensor<32x3x3x3xf32>
    %dbs = stablehlo.reduce(%dhcs init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x32x32xf32>, tensor<f32>) -> tensor<32xf32>
    %dW1xt = stablehlo.transpose %pool, dims = [1, 0, 2, 3] : (tensor<128x32x16x16xf32>) -> tensor<32x128x16x16xf32>
    %dW1dt = stablehlo.transpose %dhc1, dims = [1, 0, 2, 3] : (tensor<128x32x16x16xf32>) -> tensor<32x128x16x16xf32>
    %dW1raw = stablehlo.convolution(%dW1xt, %dW1dt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x16x16xf32>, tensor<32x128x16x16xf32>) -> tensor<32x32x3x3xf32>
    %dW1 = stablehlo.transpose %dW1raw, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %db1 = stablehlo.reduce(%dhc1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x16x16xf32>, tensor<f32>) -> tensor<32xf32>
    %dW2xt = stablehlo.transpose %ac1, dims = [1, 0, 2, 3] : (tensor<128x32x16x16xf32>) -> tensor<32x128x16x16xf32>
    %dW2dt = stablehlo.transpose %dhc2, dims = [1, 0, 2, 3] : (tensor<128x32x16x16xf32>) -> tensor<32x128x16x16xf32>
    %dW2raw = stablehlo.convolution(%dW2xt, %dW2dt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x16x16xf32>, tensor<32x128x16x16xf32>) -> tensor<32x32x3x3xf32>
    %dW2 = stablehlo.transpose %dW2raw, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %db2 = stablehlo.reduce(%dhc2 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x16x16xf32>, tensor<f32>) -> tensor<32xf32>
    %dW1pxt = stablehlo.transpose %rblkr, dims = [1, 0, 2, 3] : (tensor<128x32x16x16xf32>) -> tensor<32x128x16x16xf32>
    %dW1pdt = stablehlo.transpose %dhc1p, dims = [1, 0, 2, 3] : (tensor<128x64x16x16xf32>) -> tensor<64x128x16x16xf32>
    %dW1praw = stablehlo.convolution(%dW1pxt, %dW1pdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x16x16xf32>, tensor<64x128x16x16xf32>) -> tensor<32x64x3x3xf32>
    %dW1p = stablehlo.transpose %dW1praw, dims = [1, 0, 2, 3] : (tensor<32x64x3x3xf32>) -> tensor<64x32x3x3xf32>
    %db1p = stablehlo.reduce(%dhc1p init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x16x16xf32>, tensor<f32>) -> tensor<64xf32>
    %dW2pxt = stablehlo.transpose %ac1p, dims = [1, 0, 2, 3] : (tensor<128x64x16x16xf32>) -> tensor<64x128x16x16xf32>
    %dW2pdt = stablehlo.transpose %dhc2p, dims = [1, 0, 2, 3] : (tensor<128x64x16x16xf32>) -> tensor<64x128x16x16xf32>
    %dW2praw = stablehlo.convolution(%dW2pxt, %dW2pdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x16x16xf32>, tensor<64x128x16x16xf32>) -> tensor<64x64x3x3xf32>
    %dW2p = stablehlo.transpose %dW2praw, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %db2p = stablehlo.reduce(%dhc2p init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x16x16xf32>, tensor<f32>) -> tensor<64xf32>
    %dWpxt = stablehlo.transpose %rblkr, dims = [1, 0, 2, 3] : (tensor<128x32x16x16xf32>) -> tensor<32x128x16x16xf32>
    %dWpdt = stablehlo.transpose %dhcpp, dims = [1, 0, 2, 3] : (tensor<128x64x16x16xf32>) -> tensor<64x128x16x16xf32>
    %dWpraw = stablehlo.convolution(%dWpxt, %dWpdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x16x16xf32>, tensor<64x128x16x16xf32>) -> tensor<32x64x3x3xf32>
    %dWp = stablehlo.transpose %dWpraw, dims = [1, 0, 2, 3] : (tensor<32x64x3x3xf32>) -> tensor<64x32x3x3xf32>
    %dbp = stablehlo.reduce(%dhcpp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x16x16xf32>, tensor<f32>) -> tensor<64xf32>
    // ── SGD θ' = θ − lr·∇ (all 26 params, incl. scalar γ/β) ──
    %Wsl = stablehlo.constant dense<0.00078125> : tensor<32x3x3x3xf32>
    %Wss = stablehlo.multiply %dWs, %Wsl : tensor<32x3x3x3xf32>
    %Wsn = stablehlo.subtract %Ws, %Wss : tensor<32x3x3x3xf32>
    %bsl = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %bss = stablehlo.multiply %dbs, %bsl : tensor<32xf32>
    %bsn = stablehlo.subtract %bs, %bss : tensor<32xf32>
    %gsl = stablehlo.constant dense<0.00078125> : tensor<f32>
    %gss = stablehlo.multiply %dgs, %gsl : tensor<f32>
    %gsn = stablehlo.subtract %gs, %gss : tensor<f32>
    %btsl = stablehlo.constant dense<0.00078125> : tensor<f32>
    %btss = stablehlo.multiply %dbts, %btsl : tensor<f32>
    %btsn = stablehlo.subtract %bts, %btss : tensor<f32>
    %W1l = stablehlo.constant dense<0.00078125> : tensor<32x32x3x3xf32>
    %W1s = stablehlo.multiply %dW1, %W1l : tensor<32x32x3x3xf32>
    %W1n = stablehlo.subtract %W1, %W1s : tensor<32x32x3x3xf32>
    %b1l = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %b1s = stablehlo.multiply %db1, %b1l : tensor<32xf32>
    %b1n = stablehlo.subtract %b1, %b1s : tensor<32xf32>
    %g1l = stablehlo.constant dense<0.00078125> : tensor<f32>
    %g1s = stablehlo.multiply %dg1, %g1l : tensor<f32>
    %g1n = stablehlo.subtract %g1, %g1s : tensor<f32>
    %bt1l = stablehlo.constant dense<0.00078125> : tensor<f32>
    %bt1s = stablehlo.multiply %dbt1, %bt1l : tensor<f32>
    %bt1n = stablehlo.subtract %bt1, %bt1s : tensor<f32>
    %W2l = stablehlo.constant dense<0.00078125> : tensor<32x32x3x3xf32>
    %W2s = stablehlo.multiply %dW2, %W2l : tensor<32x32x3x3xf32>
    %W2n = stablehlo.subtract %W2, %W2s : tensor<32x32x3x3xf32>
    %b2l = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %b2s = stablehlo.multiply %db2, %b2l : tensor<32xf32>
    %b2n = stablehlo.subtract %b2, %b2s : tensor<32xf32>
    %g2l = stablehlo.constant dense<0.00078125> : tensor<f32>
    %g2s = stablehlo.multiply %dg2, %g2l : tensor<f32>
    %g2n = stablehlo.subtract %g2, %g2s : tensor<f32>
    %bt2l = stablehlo.constant dense<0.00078125> : tensor<f32>
    %bt2s = stablehlo.multiply %dbt2, %bt2l : tensor<f32>
    %bt2n = stablehlo.subtract %bt2, %bt2s : tensor<f32>
    %W1pl = stablehlo.constant dense<0.00078125> : tensor<64x32x3x3xf32>
    %W1ps = stablehlo.multiply %dW1p, %W1pl : tensor<64x32x3x3xf32>
    %W1pn = stablehlo.subtract %W1p, %W1ps : tensor<64x32x3x3xf32>
    %b1pl = stablehlo.constant dense<0.00078125> : tensor<64xf32>
    %b1ps = stablehlo.multiply %db1p, %b1pl : tensor<64xf32>
    %b1pn = stablehlo.subtract %b1p, %b1ps : tensor<64xf32>
    %g1pl = stablehlo.constant dense<0.00078125> : tensor<f32>
    %g1ps = stablehlo.multiply %dg1p, %g1pl : tensor<f32>
    %g1pn = stablehlo.subtract %g1p, %g1ps : tensor<f32>
    %bt1pl = stablehlo.constant dense<0.00078125> : tensor<f32>
    %bt1ps = stablehlo.multiply %dbt1p, %bt1pl : tensor<f32>
    %bt1pn = stablehlo.subtract %bt1p, %bt1ps : tensor<f32>
    %W2pl = stablehlo.constant dense<0.00078125> : tensor<64x64x3x3xf32>
    %W2ps = stablehlo.multiply %dW2p, %W2pl : tensor<64x64x3x3xf32>
    %W2pn = stablehlo.subtract %W2p, %W2ps : tensor<64x64x3x3xf32>
    %b2pl = stablehlo.constant dense<0.00078125> : tensor<64xf32>
    %b2ps = stablehlo.multiply %db2p, %b2pl : tensor<64xf32>
    %b2pn = stablehlo.subtract %b2p, %b2ps : tensor<64xf32>
    %g2pl = stablehlo.constant dense<0.00078125> : tensor<f32>
    %g2ps = stablehlo.multiply %dg2p, %g2pl : tensor<f32>
    %g2pn = stablehlo.subtract %g2p, %g2ps : tensor<f32>
    %bt2pl = stablehlo.constant dense<0.00078125> : tensor<f32>
    %bt2ps = stablehlo.multiply %dbt2p, %bt2pl : tensor<f32>
    %bt2pn = stablehlo.subtract %bt2p, %bt2ps : tensor<f32>
    %Wpl = stablehlo.constant dense<0.00078125> : tensor<64x32x3x3xf32>
    %Wps = stablehlo.multiply %dWp, %Wpl : tensor<64x32x3x3xf32>
    %Wpn = stablehlo.subtract %Wp, %Wps : tensor<64x32x3x3xf32>
    %bpl = stablehlo.constant dense<0.00078125> : tensor<64xf32>
    %bps = stablehlo.multiply %dbp, %bpl : tensor<64xf32>
    %bpn = stablehlo.subtract %bp, %bps : tensor<64xf32>
    %gpl = stablehlo.constant dense<0.00078125> : tensor<f32>
    %gps = stablehlo.multiply %dgp, %gpl : tensor<f32>
    %gpn = stablehlo.subtract %gp, %gps : tensor<f32>
    %btpl = stablehlo.constant dense<0.00078125> : tensor<f32>
    %btps = stablehlo.multiply %dbtp, %btpl : tensor<f32>
    %btpn = stablehlo.subtract %btp, %btps : tensor<f32>
    %Wdl = stablehlo.constant dense<0.00078125> : tensor<64x10xf32>
    %Wds = stablehlo.multiply %dWd, %Wdl : tensor<64x10xf32>
    %Wdn = stablehlo.subtract %Wd, %Wds : tensor<64x10xf32>
    %bdl = stablehlo.constant dense<0.00078125> : tensor<10xf32>
    %bds = stablehlo.multiply %dbd, %bdl : tensor<10xf32>
    %bdn = stablehlo.subtract %bd, %bds : tensor<10xf32>
    return %Wsn, %bsn, %gsn, %btsn, %W1n, %b1n, %g1n, %bt1n, %W2n, %b2n, %g2n, %bt2n, %W1pn, %b1pn, %g1pn, %bt1pn, %W2pn, %b2pn, %g2pn, %bt2pn, %Wpn, %bpn, %gpn, %btpn, %Wdn, %bdn : tensor<32x3x3x3xf32>, tensor<32xf32>, tensor<f32>, tensor<f32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<f32>, tensor<f32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<f32>, tensor<f32>, tensor<64x32x3x3xf32>, tensor<64xf32>, tensor<f32>, tensor<f32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<f32>, tensor<f32>, tensor<64x32x3x3xf32>, tensor<64xf32>, tensor<f32>, tensor<f32>, tensor<64x10xf32>, tensor<10xf32>
  }
}

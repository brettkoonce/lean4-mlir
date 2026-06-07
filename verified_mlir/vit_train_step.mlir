module @m {
  func.func @vit_train_step(%x: tensor<32x150528xf32>, %wConv: tensor<192x3x16x16xf32>, %bConv: tensor<192xf32>, %cls: tensor<1x192xf32>, %pos: tensor<197x192xf32>, %g1_0: tensor<192xf32>, %b1_0: tensor<192xf32>, %Wq_0: tensor<192x192xf32>, %bq_0: tensor<192xf32>, %Wk_0: tensor<192x192xf32>, %bk_0: tensor<192xf32>, %Wv_0: tensor<192x192xf32>, %bv_0: tensor<192xf32>, %Wo_0: tensor<192x192xf32>, %bo_0: tensor<192xf32>, %g2_0: tensor<192xf32>, %b2_0: tensor<192xf32>, %Wfc1_0: tensor<192x768xf32>, %bfc1_0: tensor<768xf32>, %Wfc2_0: tensor<768x192xf32>, %bfc2_0: tensor<192xf32>, %g1_1: tensor<192xf32>, %b1_1: tensor<192xf32>, %Wq_1: tensor<192x192xf32>, %bq_1: tensor<192xf32>, %Wk_1: tensor<192x192xf32>, %bk_1: tensor<192xf32>, %Wv_1: tensor<192x192xf32>, %bv_1: tensor<192xf32>, %Wo_1: tensor<192x192xf32>, %bo_1: tensor<192xf32>, %g2_1: tensor<192xf32>, %b2_1: tensor<192xf32>, %Wfc1_1: tensor<192x768xf32>, %bfc1_1: tensor<768xf32>, %Wfc2_1: tensor<768x192xf32>, %bfc2_1: tensor<192xf32>, %g1_2: tensor<192xf32>, %b1_2: tensor<192xf32>, %Wq_2: tensor<192x192xf32>, %bq_2: tensor<192xf32>, %Wk_2: tensor<192x192xf32>, %bk_2: tensor<192xf32>, %Wv_2: tensor<192x192xf32>, %bv_2: tensor<192xf32>, %Wo_2: tensor<192x192xf32>, %bo_2: tensor<192xf32>, %g2_2: tensor<192xf32>, %b2_2: tensor<192xf32>, %Wfc1_2: tensor<192x768xf32>, %bfc1_2: tensor<768xf32>, %Wfc2_2: tensor<768x192xf32>, %bfc2_2: tensor<192xf32>, %g1_3: tensor<192xf32>, %b1_3: tensor<192xf32>, %Wq_3: tensor<192x192xf32>, %bq_3: tensor<192xf32>, %Wk_3: tensor<192x192xf32>, %bk_3: tensor<192xf32>, %Wv_3: tensor<192x192xf32>, %bv_3: tensor<192xf32>, %Wo_3: tensor<192x192xf32>, %bo_3: tensor<192xf32>, %g2_3: tensor<192xf32>, %b2_3: tensor<192xf32>, %Wfc1_3: tensor<192x768xf32>, %bfc1_3: tensor<768xf32>, %Wfc2_3: tensor<768x192xf32>, %bfc2_3: tensor<192xf32>, %g1_4: tensor<192xf32>, %b1_4: tensor<192xf32>, %Wq_4: tensor<192x192xf32>, %bq_4: tensor<192xf32>, %Wk_4: tensor<192x192xf32>, %bk_4: tensor<192xf32>, %Wv_4: tensor<192x192xf32>, %bv_4: tensor<192xf32>, %Wo_4: tensor<192x192xf32>, %bo_4: tensor<192xf32>, %g2_4: tensor<192xf32>, %b2_4: tensor<192xf32>, %Wfc1_4: tensor<192x768xf32>, %bfc1_4: tensor<768xf32>, %Wfc2_4: tensor<768x192xf32>, %bfc2_4: tensor<192xf32>, %g1_5: tensor<192xf32>, %b1_5: tensor<192xf32>, %Wq_5: tensor<192x192xf32>, %bq_5: tensor<192xf32>, %Wk_5: tensor<192x192xf32>, %bk_5: tensor<192xf32>, %Wv_5: tensor<192x192xf32>, %bv_5: tensor<192xf32>, %Wo_5: tensor<192x192xf32>, %bo_5: tensor<192xf32>, %g2_5: tensor<192xf32>, %b2_5: tensor<192xf32>, %Wfc1_5: tensor<192x768xf32>, %bfc1_5: tensor<768xf32>, %Wfc2_5: tensor<768x192xf32>, %bfc2_5: tensor<192xf32>, %g1_6: tensor<192xf32>, %b1_6: tensor<192xf32>, %Wq_6: tensor<192x192xf32>, %bq_6: tensor<192xf32>, %Wk_6: tensor<192x192xf32>, %bk_6: tensor<192xf32>, %Wv_6: tensor<192x192xf32>, %bv_6: tensor<192xf32>, %Wo_6: tensor<192x192xf32>, %bo_6: tensor<192xf32>, %g2_6: tensor<192xf32>, %b2_6: tensor<192xf32>, %Wfc1_6: tensor<192x768xf32>, %bfc1_6: tensor<768xf32>, %Wfc2_6: tensor<768x192xf32>, %bfc2_6: tensor<192xf32>, %g1_7: tensor<192xf32>, %b1_7: tensor<192xf32>, %Wq_7: tensor<192x192xf32>, %bq_7: tensor<192xf32>, %Wk_7: tensor<192x192xf32>, %bk_7: tensor<192xf32>, %Wv_7: tensor<192x192xf32>, %bv_7: tensor<192xf32>, %Wo_7: tensor<192x192xf32>, %bo_7: tensor<192xf32>, %g2_7: tensor<192xf32>, %b2_7: tensor<192xf32>, %Wfc1_7: tensor<192x768xf32>, %bfc1_7: tensor<768xf32>, %Wfc2_7: tensor<768x192xf32>, %bfc2_7: tensor<192xf32>, %g1_8: tensor<192xf32>, %b1_8: tensor<192xf32>, %Wq_8: tensor<192x192xf32>, %bq_8: tensor<192xf32>, %Wk_8: tensor<192x192xf32>, %bk_8: tensor<192xf32>, %Wv_8: tensor<192x192xf32>, %bv_8: tensor<192xf32>, %Wo_8: tensor<192x192xf32>, %bo_8: tensor<192xf32>, %g2_8: tensor<192xf32>, %b2_8: tensor<192xf32>, %Wfc1_8: tensor<192x768xf32>, %bfc1_8: tensor<768xf32>, %Wfc2_8: tensor<768x192xf32>, %bfc2_8: tensor<192xf32>, %g1_9: tensor<192xf32>, %b1_9: tensor<192xf32>, %Wq_9: tensor<192x192xf32>, %bq_9: tensor<192xf32>, %Wk_9: tensor<192x192xf32>, %bk_9: tensor<192xf32>, %Wv_9: tensor<192x192xf32>, %bv_9: tensor<192xf32>, %Wo_9: tensor<192x192xf32>, %bo_9: tensor<192xf32>, %g2_9: tensor<192xf32>, %b2_9: tensor<192xf32>, %Wfc1_9: tensor<192x768xf32>, %bfc1_9: tensor<768xf32>, %Wfc2_9: tensor<768x192xf32>, %bfc2_9: tensor<192xf32>, %g1_10: tensor<192xf32>, %b1_10: tensor<192xf32>, %Wq_10: tensor<192x192xf32>, %bq_10: tensor<192xf32>, %Wk_10: tensor<192x192xf32>, %bk_10: tensor<192xf32>, %Wv_10: tensor<192x192xf32>, %bv_10: tensor<192xf32>, %Wo_10: tensor<192x192xf32>, %bo_10: tensor<192xf32>, %g2_10: tensor<192xf32>, %b2_10: tensor<192xf32>, %Wfc1_10: tensor<192x768xf32>, %bfc1_10: tensor<768xf32>, %Wfc2_10: tensor<768x192xf32>, %bfc2_10: tensor<192xf32>, %g1_11: tensor<192xf32>, %b1_11: tensor<192xf32>, %Wq_11: tensor<192x192xf32>, %bq_11: tensor<192xf32>, %Wk_11: tensor<192x192xf32>, %bk_11: tensor<192xf32>, %Wv_11: tensor<192x192xf32>, %bv_11: tensor<192xf32>, %Wo_11: tensor<192x192xf32>, %bo_11: tensor<192xf32>, %g2_11: tensor<192xf32>, %b2_11: tensor<192xf32>, %Wfc1_11: tensor<192x768xf32>, %bfc1_11: tensor<768xf32>, %Wfc2_11: tensor<768x192xf32>, %bfc2_11: tensor<192xf32>, %gF: tensor<192xf32>, %bF: tensor<192xf32>, %Wc: tensor<192x10xf32>, %bc: tensor<10xf32>, %onehot: tensor<32x10xf32>) -> (tensor<192x3x16x16xf32>, tensor<192xf32>, tensor<1x192xf32>, tensor<197x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x10xf32>, tensor<10xf32>) {
    %sc = stablehlo.constant dense<0.0> : tensor<f32>
    %xr = stablehlo.reshape %x : (tensor<32x150528xf32>) -> tensor<32x3x224x224xf32>
    %vitpec = stablehlo.convolution(%xr, %wConv)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [16, 16], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3x224x224xf32>, tensor<192x3x16x16xf32>) -> tensor<32x192x14x14xf32>
    %vitpecbb = stablehlo.broadcast_in_dim %bConv, dims = [1] : (tensor<192xf32>) -> tensor<32x192x14x14xf32>
    %vitpepe = stablehlo.add %vitpec, %vitpecbb : tensor<32x192x14x14xf32>
    %vitpept = stablehlo.transpose %vitpepe, dims = [0, 2, 3, 1] : (tensor<32x192x14x14xf32>) -> tensor<32x14x14x192xf32>
    %vitpetok = stablehlo.reshape %vitpept : (tensor<32x14x14x192xf32>) -> tensor<32x196x192xf32>
    %vitcpclsb = stablehlo.broadcast_in_dim %cls, dims = [1, 2] : (tensor<1x192xf32>) -> tensor<32x1x192xf32>
    %vitcpcat = stablehlo.concatenate %vitcpclsb, %vitpetok, dim = 1 : (tensor<32x1x192xf32>, tensor<32x196x192xf32>) -> tensor<32x197x192xf32>
    %vitcpposb = stablehlo.broadcast_in_dim %pos, dims = [1, 2] : (tensor<197x192xf32>) -> tensor<32x197x192xf32>
    %vitcpz = stablehlo.add %vitcpcat, %vitcpposb : tensor<32x197x192xf32>
    %vitb0_1sum = stablehlo.reduce(%vitcpz init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb0_1nf = stablehlo.constant dense<192.0> : tensor<32x197xf32>
    %vitb0_1mu = stablehlo.divide %vitb0_1sum, %vitb0_1nf : tensor<32x197xf32>
    %vitb0_1mub = stablehlo.broadcast_in_dim %vitb0_1mu, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb0_1xc = stablehlo.subtract %vitcpz, %vitb0_1mub : tensor<32x197x192xf32>
    %vitb0_1sq = stablehlo.multiply %vitb0_1xc, %vitb0_1xc : tensor<32x197x192xf32>
    %vitb0_1vsum = stablehlo.reduce(%vitb0_1sq init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb0_1var = stablehlo.divide %vitb0_1vsum, %vitb0_1nf : tensor<32x197xf32>
    %vitb0_1eps = stablehlo.constant dense<1.0e-5> : tensor<32x197xf32>
    %vitb0_1ve = stablehlo.add %vitb0_1var, %vitb0_1eps : tensor<32x197xf32>
    %vitb0_1istd = stablehlo.rsqrt %vitb0_1ve : tensor<32x197xf32>
    %vitb0_1istdb = stablehlo.broadcast_in_dim %vitb0_1istd, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb0_1xhat = stablehlo.multiply %vitb0_1xc, %vitb0_1istdb : tensor<32x197x192xf32>
    %vitb0_1gb = stablehlo.broadcast_in_dim %g1_0, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb0_1bbc = stablehlo.broadcast_in_dim %b1_0, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb0_1gx = stablehlo.multiply %vitb0_1xhat, %vitb0_1gb : tensor<32x197x192xf32>
    %vitb0_1y = stablehlo.add %vitb0_1gx, %vitb0_1bbc : tensor<32x197x192xf32>
    %vitb0_mQd = stablehlo.dot_general %vitb0_1y, %Wq_0, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb0_mQbb = stablehlo.broadcast_in_dim %bq_0, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb0_mQ = stablehlo.add %vitb0_mQd, %vitb0_mQbb : tensor<32x197x192xf32>
    %vitb0_mKd = stablehlo.dot_general %vitb0_1y, %Wk_0, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb0_mKbb = stablehlo.broadcast_in_dim %bk_0, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb0_mK = stablehlo.add %vitb0_mKd, %vitb0_mKbb : tensor<32x197x192xf32>
    %vitb0_mVd = stablehlo.dot_general %vitb0_1y, %Wv_0, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb0_mVbb = stablehlo.broadcast_in_dim %bv_0, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb0_mV = stablehlo.add %vitb0_mVd, %vitb0_mVbb : tensor<32x197x192xf32>
    %vitb0_mQhr = stablehlo.reshape %vitb0_mQ : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb0_mQh = stablehlo.transpose %vitb0_mQhr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb0_mKhr = stablehlo.reshape %vitb0_mK : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb0_mKh = stablehlo.transpose %vitb0_mKhr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb0_mVhr = stablehlo.reshape %vitb0_mV : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb0_mVh = stablehlo.transpose %vitb0_mVhr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb0_mS = stablehlo.dot_general %vitb0_mQh, %vitb0_mKh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x64xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x197xf32>
    %vitb0_mscl = stablehlo.constant dense<0.125> : tensor<32x3x197x197xf32>
    %vitb0_mSs = stablehlo.multiply %vitb0_mS, %vitb0_mscl : tensor<32x3x197x197xf32>
    %vitb0_mse = stablehlo.exponential %vitb0_mSs : tensor<32x3x197x197xf32>
    %vitb0_msum = stablehlo.reduce(%vitb0_mse init: %sc) applies stablehlo.add across dimensions = [3] : (tensor<32x3x197x197xf32>, tensor<f32>) -> tensor<32x3x197xf32>
    %vitb0_msumb = stablehlo.broadcast_in_dim %vitb0_msum, dims = [0, 1, 2] : (tensor<32x3x197xf32>) -> tensor<32x3x197x197xf32>
    %vitb0_mW = stablehlo.divide %vitb0_mse, %vitb0_msumb : tensor<32x3x197x197xf32>
    %vitb0_mA = stablehlo.dot_general %vitb0_mW, %vitb0_mVh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb0_mAT = stablehlo.transpose %vitb0_mA, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb0_mP = stablehlo.reshape %vitb0_mAT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb0_mod = stablehlo.dot_general %vitb0_mP, %Wo_0, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb0_mobb = stablehlo.broadcast_in_dim %bo_0, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb0_mO = stablehlo.add %vitb0_mod, %vitb0_mobb : tensor<32x197x192xf32>
    %vitb0_r1 = stablehlo.add %vitcpz, %vitb0_mO : tensor<32x197x192xf32>
    %vitb0_2sum = stablehlo.reduce(%vitb0_r1 init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb0_2nf = stablehlo.constant dense<192.0> : tensor<32x197xf32>
    %vitb0_2mu = stablehlo.divide %vitb0_2sum, %vitb0_2nf : tensor<32x197xf32>
    %vitb0_2mub = stablehlo.broadcast_in_dim %vitb0_2mu, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb0_2xc = stablehlo.subtract %vitb0_r1, %vitb0_2mub : tensor<32x197x192xf32>
    %vitb0_2sq = stablehlo.multiply %vitb0_2xc, %vitb0_2xc : tensor<32x197x192xf32>
    %vitb0_2vsum = stablehlo.reduce(%vitb0_2sq init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb0_2var = stablehlo.divide %vitb0_2vsum, %vitb0_2nf : tensor<32x197xf32>
    %vitb0_2eps = stablehlo.constant dense<1.0e-5> : tensor<32x197xf32>
    %vitb0_2ve = stablehlo.add %vitb0_2var, %vitb0_2eps : tensor<32x197xf32>
    %vitb0_2istd = stablehlo.rsqrt %vitb0_2ve : tensor<32x197xf32>
    %vitb0_2istdb = stablehlo.broadcast_in_dim %vitb0_2istd, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb0_2xhat = stablehlo.multiply %vitb0_2xc, %vitb0_2istdb : tensor<32x197x192xf32>
    %vitb0_2gb = stablehlo.broadcast_in_dim %g2_0, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb0_2bbc = stablehlo.broadcast_in_dim %b2_0, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb0_2gx = stablehlo.multiply %vitb0_2xhat, %vitb0_2gb : tensor<32x197x192xf32>
    %vitb0_2y = stablehlo.add %vitb0_2gx, %vitb0_2bbc : tensor<32x197x192xf32>
    %vitb0_ph1d = stablehlo.dot_general %vitb0_2y, %Wfc1_0, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x768xf32>) -> tensor<32x197x768xf32>
    %vitb0_ph1bb = stablehlo.broadcast_in_dim %bfc1_0, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %vitb0_ph1 = stablehlo.add %vitb0_ph1d, %vitb0_ph1bb : tensor<32x197x768xf32>
    %vitb0_pgx2 = stablehlo.multiply %vitb0_ph1, %vitb0_ph1 : tensor<32x197x768xf32>
    %vitb0_pgx3 = stablehlo.multiply %vitb0_pgx2, %vitb0_ph1 : tensor<32x197x768xf32>
    %vitb0_pgck = stablehlo.constant dense<0.044715> : tensor<32x197x768xf32>
    %vitb0_pgkx3 = stablehlo.multiply %vitb0_pgck, %vitb0_pgx3 : tensor<32x197x768xf32>
    %vitb0_pginn = stablehlo.add %vitb0_ph1, %vitb0_pgkx3 : tensor<32x197x768xf32>
    %vitb0_pgcsqrt = stablehlo.constant dense<0.7978845608028654> : tensor<32x197x768xf32>
    %vitb0_pgu = stablehlo.multiply %vitb0_pgcsqrt, %vitb0_pginn : tensor<32x197x768xf32>
    %vitb0_pgt = stablehlo.tanh %vitb0_pgu : tensor<32x197x768xf32>
    %vitb0_pgone = stablehlo.constant dense<1.0> : tensor<32x197x768xf32>
    %vitb0_pgopt = stablehlo.add %vitb0_pgone, %vitb0_pgt : tensor<32x197x768xf32>
    %vitb0_pgchalf = stablehlo.constant dense<0.5> : tensor<32x197x768xf32>
    %vitb0_pghx = stablehlo.multiply %vitb0_pgchalf, %vitb0_ph1 : tensor<32x197x768xf32>
    %vitb0_pga = stablehlo.multiply %vitb0_pghx, %vitb0_pgopt : tensor<32x197x768xf32>
    %vitb0_py2d = stablehlo.dot_general %vitb0_pga, %Wfc2_0, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x192xf32>) -> tensor<32x197x192xf32>
    %vitb0_py2bb = stablehlo.broadcast_in_dim %bfc2_0, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb0_py = stablehlo.add %vitb0_py2d, %vitb0_py2bb : tensor<32x197x192xf32>
    %vitb0_out = stablehlo.add %vitb0_r1, %vitb0_py : tensor<32x197x192xf32>
    %vitb1_1sum = stablehlo.reduce(%vitb0_out init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb1_1nf = stablehlo.constant dense<192.0> : tensor<32x197xf32>
    %vitb1_1mu = stablehlo.divide %vitb1_1sum, %vitb1_1nf : tensor<32x197xf32>
    %vitb1_1mub = stablehlo.broadcast_in_dim %vitb1_1mu, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb1_1xc = stablehlo.subtract %vitb0_out, %vitb1_1mub : tensor<32x197x192xf32>
    %vitb1_1sq = stablehlo.multiply %vitb1_1xc, %vitb1_1xc : tensor<32x197x192xf32>
    %vitb1_1vsum = stablehlo.reduce(%vitb1_1sq init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb1_1var = stablehlo.divide %vitb1_1vsum, %vitb1_1nf : tensor<32x197xf32>
    %vitb1_1eps = stablehlo.constant dense<1.0e-5> : tensor<32x197xf32>
    %vitb1_1ve = stablehlo.add %vitb1_1var, %vitb1_1eps : tensor<32x197xf32>
    %vitb1_1istd = stablehlo.rsqrt %vitb1_1ve : tensor<32x197xf32>
    %vitb1_1istdb = stablehlo.broadcast_in_dim %vitb1_1istd, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb1_1xhat = stablehlo.multiply %vitb1_1xc, %vitb1_1istdb : tensor<32x197x192xf32>
    %vitb1_1gb = stablehlo.broadcast_in_dim %g1_1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb1_1bbc = stablehlo.broadcast_in_dim %b1_1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb1_1gx = stablehlo.multiply %vitb1_1xhat, %vitb1_1gb : tensor<32x197x192xf32>
    %vitb1_1y = stablehlo.add %vitb1_1gx, %vitb1_1bbc : tensor<32x197x192xf32>
    %vitb1_mQd = stablehlo.dot_general %vitb1_1y, %Wq_1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb1_mQbb = stablehlo.broadcast_in_dim %bq_1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb1_mQ = stablehlo.add %vitb1_mQd, %vitb1_mQbb : tensor<32x197x192xf32>
    %vitb1_mKd = stablehlo.dot_general %vitb1_1y, %Wk_1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb1_mKbb = stablehlo.broadcast_in_dim %bk_1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb1_mK = stablehlo.add %vitb1_mKd, %vitb1_mKbb : tensor<32x197x192xf32>
    %vitb1_mVd = stablehlo.dot_general %vitb1_1y, %Wv_1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb1_mVbb = stablehlo.broadcast_in_dim %bv_1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb1_mV = stablehlo.add %vitb1_mVd, %vitb1_mVbb : tensor<32x197x192xf32>
    %vitb1_mQhr = stablehlo.reshape %vitb1_mQ : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb1_mQh = stablehlo.transpose %vitb1_mQhr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb1_mKhr = stablehlo.reshape %vitb1_mK : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb1_mKh = stablehlo.transpose %vitb1_mKhr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb1_mVhr = stablehlo.reshape %vitb1_mV : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb1_mVh = stablehlo.transpose %vitb1_mVhr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb1_mS = stablehlo.dot_general %vitb1_mQh, %vitb1_mKh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x64xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x197xf32>
    %vitb1_mscl = stablehlo.constant dense<0.125> : tensor<32x3x197x197xf32>
    %vitb1_mSs = stablehlo.multiply %vitb1_mS, %vitb1_mscl : tensor<32x3x197x197xf32>
    %vitb1_mse = stablehlo.exponential %vitb1_mSs : tensor<32x3x197x197xf32>
    %vitb1_msum = stablehlo.reduce(%vitb1_mse init: %sc) applies stablehlo.add across dimensions = [3] : (tensor<32x3x197x197xf32>, tensor<f32>) -> tensor<32x3x197xf32>
    %vitb1_msumb = stablehlo.broadcast_in_dim %vitb1_msum, dims = [0, 1, 2] : (tensor<32x3x197xf32>) -> tensor<32x3x197x197xf32>
    %vitb1_mW = stablehlo.divide %vitb1_mse, %vitb1_msumb : tensor<32x3x197x197xf32>
    %vitb1_mA = stablehlo.dot_general %vitb1_mW, %vitb1_mVh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb1_mAT = stablehlo.transpose %vitb1_mA, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb1_mP = stablehlo.reshape %vitb1_mAT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb1_mod = stablehlo.dot_general %vitb1_mP, %Wo_1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb1_mobb = stablehlo.broadcast_in_dim %bo_1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb1_mO = stablehlo.add %vitb1_mod, %vitb1_mobb : tensor<32x197x192xf32>
    %vitb1_r1 = stablehlo.add %vitb0_out, %vitb1_mO : tensor<32x197x192xf32>
    %vitb1_2sum = stablehlo.reduce(%vitb1_r1 init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb1_2nf = stablehlo.constant dense<192.0> : tensor<32x197xf32>
    %vitb1_2mu = stablehlo.divide %vitb1_2sum, %vitb1_2nf : tensor<32x197xf32>
    %vitb1_2mub = stablehlo.broadcast_in_dim %vitb1_2mu, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb1_2xc = stablehlo.subtract %vitb1_r1, %vitb1_2mub : tensor<32x197x192xf32>
    %vitb1_2sq = stablehlo.multiply %vitb1_2xc, %vitb1_2xc : tensor<32x197x192xf32>
    %vitb1_2vsum = stablehlo.reduce(%vitb1_2sq init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb1_2var = stablehlo.divide %vitb1_2vsum, %vitb1_2nf : tensor<32x197xf32>
    %vitb1_2eps = stablehlo.constant dense<1.0e-5> : tensor<32x197xf32>
    %vitb1_2ve = stablehlo.add %vitb1_2var, %vitb1_2eps : tensor<32x197xf32>
    %vitb1_2istd = stablehlo.rsqrt %vitb1_2ve : tensor<32x197xf32>
    %vitb1_2istdb = stablehlo.broadcast_in_dim %vitb1_2istd, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb1_2xhat = stablehlo.multiply %vitb1_2xc, %vitb1_2istdb : tensor<32x197x192xf32>
    %vitb1_2gb = stablehlo.broadcast_in_dim %g2_1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb1_2bbc = stablehlo.broadcast_in_dim %b2_1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb1_2gx = stablehlo.multiply %vitb1_2xhat, %vitb1_2gb : tensor<32x197x192xf32>
    %vitb1_2y = stablehlo.add %vitb1_2gx, %vitb1_2bbc : tensor<32x197x192xf32>
    %vitb1_ph1d = stablehlo.dot_general %vitb1_2y, %Wfc1_1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x768xf32>) -> tensor<32x197x768xf32>
    %vitb1_ph1bb = stablehlo.broadcast_in_dim %bfc1_1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %vitb1_ph1 = stablehlo.add %vitb1_ph1d, %vitb1_ph1bb : tensor<32x197x768xf32>
    %vitb1_pgx2 = stablehlo.multiply %vitb1_ph1, %vitb1_ph1 : tensor<32x197x768xf32>
    %vitb1_pgx3 = stablehlo.multiply %vitb1_pgx2, %vitb1_ph1 : tensor<32x197x768xf32>
    %vitb1_pgck = stablehlo.constant dense<0.044715> : tensor<32x197x768xf32>
    %vitb1_pgkx3 = stablehlo.multiply %vitb1_pgck, %vitb1_pgx3 : tensor<32x197x768xf32>
    %vitb1_pginn = stablehlo.add %vitb1_ph1, %vitb1_pgkx3 : tensor<32x197x768xf32>
    %vitb1_pgcsqrt = stablehlo.constant dense<0.7978845608028654> : tensor<32x197x768xf32>
    %vitb1_pgu = stablehlo.multiply %vitb1_pgcsqrt, %vitb1_pginn : tensor<32x197x768xf32>
    %vitb1_pgt = stablehlo.tanh %vitb1_pgu : tensor<32x197x768xf32>
    %vitb1_pgone = stablehlo.constant dense<1.0> : tensor<32x197x768xf32>
    %vitb1_pgopt = stablehlo.add %vitb1_pgone, %vitb1_pgt : tensor<32x197x768xf32>
    %vitb1_pgchalf = stablehlo.constant dense<0.5> : tensor<32x197x768xf32>
    %vitb1_pghx = stablehlo.multiply %vitb1_pgchalf, %vitb1_ph1 : tensor<32x197x768xf32>
    %vitb1_pga = stablehlo.multiply %vitb1_pghx, %vitb1_pgopt : tensor<32x197x768xf32>
    %vitb1_py2d = stablehlo.dot_general %vitb1_pga, %Wfc2_1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x192xf32>) -> tensor<32x197x192xf32>
    %vitb1_py2bb = stablehlo.broadcast_in_dim %bfc2_1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb1_py = stablehlo.add %vitb1_py2d, %vitb1_py2bb : tensor<32x197x192xf32>
    %vitb1_out = stablehlo.add %vitb1_r1, %vitb1_py : tensor<32x197x192xf32>
    %vitb2_1sum = stablehlo.reduce(%vitb1_out init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb2_1nf = stablehlo.constant dense<192.0> : tensor<32x197xf32>
    %vitb2_1mu = stablehlo.divide %vitb2_1sum, %vitb2_1nf : tensor<32x197xf32>
    %vitb2_1mub = stablehlo.broadcast_in_dim %vitb2_1mu, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb2_1xc = stablehlo.subtract %vitb1_out, %vitb2_1mub : tensor<32x197x192xf32>
    %vitb2_1sq = stablehlo.multiply %vitb2_1xc, %vitb2_1xc : tensor<32x197x192xf32>
    %vitb2_1vsum = stablehlo.reduce(%vitb2_1sq init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb2_1var = stablehlo.divide %vitb2_1vsum, %vitb2_1nf : tensor<32x197xf32>
    %vitb2_1eps = stablehlo.constant dense<1.0e-5> : tensor<32x197xf32>
    %vitb2_1ve = stablehlo.add %vitb2_1var, %vitb2_1eps : tensor<32x197xf32>
    %vitb2_1istd = stablehlo.rsqrt %vitb2_1ve : tensor<32x197xf32>
    %vitb2_1istdb = stablehlo.broadcast_in_dim %vitb2_1istd, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb2_1xhat = stablehlo.multiply %vitb2_1xc, %vitb2_1istdb : tensor<32x197x192xf32>
    %vitb2_1gb = stablehlo.broadcast_in_dim %g1_2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb2_1bbc = stablehlo.broadcast_in_dim %b1_2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb2_1gx = stablehlo.multiply %vitb2_1xhat, %vitb2_1gb : tensor<32x197x192xf32>
    %vitb2_1y = stablehlo.add %vitb2_1gx, %vitb2_1bbc : tensor<32x197x192xf32>
    %vitb2_mQd = stablehlo.dot_general %vitb2_1y, %Wq_2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb2_mQbb = stablehlo.broadcast_in_dim %bq_2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb2_mQ = stablehlo.add %vitb2_mQd, %vitb2_mQbb : tensor<32x197x192xf32>
    %vitb2_mKd = stablehlo.dot_general %vitb2_1y, %Wk_2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb2_mKbb = stablehlo.broadcast_in_dim %bk_2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb2_mK = stablehlo.add %vitb2_mKd, %vitb2_mKbb : tensor<32x197x192xf32>
    %vitb2_mVd = stablehlo.dot_general %vitb2_1y, %Wv_2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb2_mVbb = stablehlo.broadcast_in_dim %bv_2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb2_mV = stablehlo.add %vitb2_mVd, %vitb2_mVbb : tensor<32x197x192xf32>
    %vitb2_mQhr = stablehlo.reshape %vitb2_mQ : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb2_mQh = stablehlo.transpose %vitb2_mQhr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb2_mKhr = stablehlo.reshape %vitb2_mK : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb2_mKh = stablehlo.transpose %vitb2_mKhr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb2_mVhr = stablehlo.reshape %vitb2_mV : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb2_mVh = stablehlo.transpose %vitb2_mVhr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb2_mS = stablehlo.dot_general %vitb2_mQh, %vitb2_mKh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x64xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x197xf32>
    %vitb2_mscl = stablehlo.constant dense<0.125> : tensor<32x3x197x197xf32>
    %vitb2_mSs = stablehlo.multiply %vitb2_mS, %vitb2_mscl : tensor<32x3x197x197xf32>
    %vitb2_mse = stablehlo.exponential %vitb2_mSs : tensor<32x3x197x197xf32>
    %vitb2_msum = stablehlo.reduce(%vitb2_mse init: %sc) applies stablehlo.add across dimensions = [3] : (tensor<32x3x197x197xf32>, tensor<f32>) -> tensor<32x3x197xf32>
    %vitb2_msumb = stablehlo.broadcast_in_dim %vitb2_msum, dims = [0, 1, 2] : (tensor<32x3x197xf32>) -> tensor<32x3x197x197xf32>
    %vitb2_mW = stablehlo.divide %vitb2_mse, %vitb2_msumb : tensor<32x3x197x197xf32>
    %vitb2_mA = stablehlo.dot_general %vitb2_mW, %vitb2_mVh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb2_mAT = stablehlo.transpose %vitb2_mA, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb2_mP = stablehlo.reshape %vitb2_mAT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb2_mod = stablehlo.dot_general %vitb2_mP, %Wo_2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb2_mobb = stablehlo.broadcast_in_dim %bo_2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb2_mO = stablehlo.add %vitb2_mod, %vitb2_mobb : tensor<32x197x192xf32>
    %vitb2_r1 = stablehlo.add %vitb1_out, %vitb2_mO : tensor<32x197x192xf32>
    %vitb2_2sum = stablehlo.reduce(%vitb2_r1 init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb2_2nf = stablehlo.constant dense<192.0> : tensor<32x197xf32>
    %vitb2_2mu = stablehlo.divide %vitb2_2sum, %vitb2_2nf : tensor<32x197xf32>
    %vitb2_2mub = stablehlo.broadcast_in_dim %vitb2_2mu, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb2_2xc = stablehlo.subtract %vitb2_r1, %vitb2_2mub : tensor<32x197x192xf32>
    %vitb2_2sq = stablehlo.multiply %vitb2_2xc, %vitb2_2xc : tensor<32x197x192xf32>
    %vitb2_2vsum = stablehlo.reduce(%vitb2_2sq init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb2_2var = stablehlo.divide %vitb2_2vsum, %vitb2_2nf : tensor<32x197xf32>
    %vitb2_2eps = stablehlo.constant dense<1.0e-5> : tensor<32x197xf32>
    %vitb2_2ve = stablehlo.add %vitb2_2var, %vitb2_2eps : tensor<32x197xf32>
    %vitb2_2istd = stablehlo.rsqrt %vitb2_2ve : tensor<32x197xf32>
    %vitb2_2istdb = stablehlo.broadcast_in_dim %vitb2_2istd, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb2_2xhat = stablehlo.multiply %vitb2_2xc, %vitb2_2istdb : tensor<32x197x192xf32>
    %vitb2_2gb = stablehlo.broadcast_in_dim %g2_2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb2_2bbc = stablehlo.broadcast_in_dim %b2_2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb2_2gx = stablehlo.multiply %vitb2_2xhat, %vitb2_2gb : tensor<32x197x192xf32>
    %vitb2_2y = stablehlo.add %vitb2_2gx, %vitb2_2bbc : tensor<32x197x192xf32>
    %vitb2_ph1d = stablehlo.dot_general %vitb2_2y, %Wfc1_2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x768xf32>) -> tensor<32x197x768xf32>
    %vitb2_ph1bb = stablehlo.broadcast_in_dim %bfc1_2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %vitb2_ph1 = stablehlo.add %vitb2_ph1d, %vitb2_ph1bb : tensor<32x197x768xf32>
    %vitb2_pgx2 = stablehlo.multiply %vitb2_ph1, %vitb2_ph1 : tensor<32x197x768xf32>
    %vitb2_pgx3 = stablehlo.multiply %vitb2_pgx2, %vitb2_ph1 : tensor<32x197x768xf32>
    %vitb2_pgck = stablehlo.constant dense<0.044715> : tensor<32x197x768xf32>
    %vitb2_pgkx3 = stablehlo.multiply %vitb2_pgck, %vitb2_pgx3 : tensor<32x197x768xf32>
    %vitb2_pginn = stablehlo.add %vitb2_ph1, %vitb2_pgkx3 : tensor<32x197x768xf32>
    %vitb2_pgcsqrt = stablehlo.constant dense<0.7978845608028654> : tensor<32x197x768xf32>
    %vitb2_pgu = stablehlo.multiply %vitb2_pgcsqrt, %vitb2_pginn : tensor<32x197x768xf32>
    %vitb2_pgt = stablehlo.tanh %vitb2_pgu : tensor<32x197x768xf32>
    %vitb2_pgone = stablehlo.constant dense<1.0> : tensor<32x197x768xf32>
    %vitb2_pgopt = stablehlo.add %vitb2_pgone, %vitb2_pgt : tensor<32x197x768xf32>
    %vitb2_pgchalf = stablehlo.constant dense<0.5> : tensor<32x197x768xf32>
    %vitb2_pghx = stablehlo.multiply %vitb2_pgchalf, %vitb2_ph1 : tensor<32x197x768xf32>
    %vitb2_pga = stablehlo.multiply %vitb2_pghx, %vitb2_pgopt : tensor<32x197x768xf32>
    %vitb2_py2d = stablehlo.dot_general %vitb2_pga, %Wfc2_2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x192xf32>) -> tensor<32x197x192xf32>
    %vitb2_py2bb = stablehlo.broadcast_in_dim %bfc2_2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb2_py = stablehlo.add %vitb2_py2d, %vitb2_py2bb : tensor<32x197x192xf32>
    %vitb2_out = stablehlo.add %vitb2_r1, %vitb2_py : tensor<32x197x192xf32>
    %vitb3_1sum = stablehlo.reduce(%vitb2_out init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb3_1nf = stablehlo.constant dense<192.0> : tensor<32x197xf32>
    %vitb3_1mu = stablehlo.divide %vitb3_1sum, %vitb3_1nf : tensor<32x197xf32>
    %vitb3_1mub = stablehlo.broadcast_in_dim %vitb3_1mu, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb3_1xc = stablehlo.subtract %vitb2_out, %vitb3_1mub : tensor<32x197x192xf32>
    %vitb3_1sq = stablehlo.multiply %vitb3_1xc, %vitb3_1xc : tensor<32x197x192xf32>
    %vitb3_1vsum = stablehlo.reduce(%vitb3_1sq init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb3_1var = stablehlo.divide %vitb3_1vsum, %vitb3_1nf : tensor<32x197xf32>
    %vitb3_1eps = stablehlo.constant dense<1.0e-5> : tensor<32x197xf32>
    %vitb3_1ve = stablehlo.add %vitb3_1var, %vitb3_1eps : tensor<32x197xf32>
    %vitb3_1istd = stablehlo.rsqrt %vitb3_1ve : tensor<32x197xf32>
    %vitb3_1istdb = stablehlo.broadcast_in_dim %vitb3_1istd, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb3_1xhat = stablehlo.multiply %vitb3_1xc, %vitb3_1istdb : tensor<32x197x192xf32>
    %vitb3_1gb = stablehlo.broadcast_in_dim %g1_3, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb3_1bbc = stablehlo.broadcast_in_dim %b1_3, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb3_1gx = stablehlo.multiply %vitb3_1xhat, %vitb3_1gb : tensor<32x197x192xf32>
    %vitb3_1y = stablehlo.add %vitb3_1gx, %vitb3_1bbc : tensor<32x197x192xf32>
    %vitb3_mQd = stablehlo.dot_general %vitb3_1y, %Wq_3, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb3_mQbb = stablehlo.broadcast_in_dim %bq_3, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb3_mQ = stablehlo.add %vitb3_mQd, %vitb3_mQbb : tensor<32x197x192xf32>
    %vitb3_mKd = stablehlo.dot_general %vitb3_1y, %Wk_3, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb3_mKbb = stablehlo.broadcast_in_dim %bk_3, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb3_mK = stablehlo.add %vitb3_mKd, %vitb3_mKbb : tensor<32x197x192xf32>
    %vitb3_mVd = stablehlo.dot_general %vitb3_1y, %Wv_3, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb3_mVbb = stablehlo.broadcast_in_dim %bv_3, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb3_mV = stablehlo.add %vitb3_mVd, %vitb3_mVbb : tensor<32x197x192xf32>
    %vitb3_mQhr = stablehlo.reshape %vitb3_mQ : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb3_mQh = stablehlo.transpose %vitb3_mQhr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb3_mKhr = stablehlo.reshape %vitb3_mK : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb3_mKh = stablehlo.transpose %vitb3_mKhr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb3_mVhr = stablehlo.reshape %vitb3_mV : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb3_mVh = stablehlo.transpose %vitb3_mVhr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb3_mS = stablehlo.dot_general %vitb3_mQh, %vitb3_mKh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x64xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x197xf32>
    %vitb3_mscl = stablehlo.constant dense<0.125> : tensor<32x3x197x197xf32>
    %vitb3_mSs = stablehlo.multiply %vitb3_mS, %vitb3_mscl : tensor<32x3x197x197xf32>
    %vitb3_mse = stablehlo.exponential %vitb3_mSs : tensor<32x3x197x197xf32>
    %vitb3_msum = stablehlo.reduce(%vitb3_mse init: %sc) applies stablehlo.add across dimensions = [3] : (tensor<32x3x197x197xf32>, tensor<f32>) -> tensor<32x3x197xf32>
    %vitb3_msumb = stablehlo.broadcast_in_dim %vitb3_msum, dims = [0, 1, 2] : (tensor<32x3x197xf32>) -> tensor<32x3x197x197xf32>
    %vitb3_mW = stablehlo.divide %vitb3_mse, %vitb3_msumb : tensor<32x3x197x197xf32>
    %vitb3_mA = stablehlo.dot_general %vitb3_mW, %vitb3_mVh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb3_mAT = stablehlo.transpose %vitb3_mA, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb3_mP = stablehlo.reshape %vitb3_mAT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb3_mod = stablehlo.dot_general %vitb3_mP, %Wo_3, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb3_mobb = stablehlo.broadcast_in_dim %bo_3, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb3_mO = stablehlo.add %vitb3_mod, %vitb3_mobb : tensor<32x197x192xf32>
    %vitb3_r1 = stablehlo.add %vitb2_out, %vitb3_mO : tensor<32x197x192xf32>
    %vitb3_2sum = stablehlo.reduce(%vitb3_r1 init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb3_2nf = stablehlo.constant dense<192.0> : tensor<32x197xf32>
    %vitb3_2mu = stablehlo.divide %vitb3_2sum, %vitb3_2nf : tensor<32x197xf32>
    %vitb3_2mub = stablehlo.broadcast_in_dim %vitb3_2mu, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb3_2xc = stablehlo.subtract %vitb3_r1, %vitb3_2mub : tensor<32x197x192xf32>
    %vitb3_2sq = stablehlo.multiply %vitb3_2xc, %vitb3_2xc : tensor<32x197x192xf32>
    %vitb3_2vsum = stablehlo.reduce(%vitb3_2sq init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb3_2var = stablehlo.divide %vitb3_2vsum, %vitb3_2nf : tensor<32x197xf32>
    %vitb3_2eps = stablehlo.constant dense<1.0e-5> : tensor<32x197xf32>
    %vitb3_2ve = stablehlo.add %vitb3_2var, %vitb3_2eps : tensor<32x197xf32>
    %vitb3_2istd = stablehlo.rsqrt %vitb3_2ve : tensor<32x197xf32>
    %vitb3_2istdb = stablehlo.broadcast_in_dim %vitb3_2istd, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb3_2xhat = stablehlo.multiply %vitb3_2xc, %vitb3_2istdb : tensor<32x197x192xf32>
    %vitb3_2gb = stablehlo.broadcast_in_dim %g2_3, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb3_2bbc = stablehlo.broadcast_in_dim %b2_3, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb3_2gx = stablehlo.multiply %vitb3_2xhat, %vitb3_2gb : tensor<32x197x192xf32>
    %vitb3_2y = stablehlo.add %vitb3_2gx, %vitb3_2bbc : tensor<32x197x192xf32>
    %vitb3_ph1d = stablehlo.dot_general %vitb3_2y, %Wfc1_3, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x768xf32>) -> tensor<32x197x768xf32>
    %vitb3_ph1bb = stablehlo.broadcast_in_dim %bfc1_3, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %vitb3_ph1 = stablehlo.add %vitb3_ph1d, %vitb3_ph1bb : tensor<32x197x768xf32>
    %vitb3_pgx2 = stablehlo.multiply %vitb3_ph1, %vitb3_ph1 : tensor<32x197x768xf32>
    %vitb3_pgx3 = stablehlo.multiply %vitb3_pgx2, %vitb3_ph1 : tensor<32x197x768xf32>
    %vitb3_pgck = stablehlo.constant dense<0.044715> : tensor<32x197x768xf32>
    %vitb3_pgkx3 = stablehlo.multiply %vitb3_pgck, %vitb3_pgx3 : tensor<32x197x768xf32>
    %vitb3_pginn = stablehlo.add %vitb3_ph1, %vitb3_pgkx3 : tensor<32x197x768xf32>
    %vitb3_pgcsqrt = stablehlo.constant dense<0.7978845608028654> : tensor<32x197x768xf32>
    %vitb3_pgu = stablehlo.multiply %vitb3_pgcsqrt, %vitb3_pginn : tensor<32x197x768xf32>
    %vitb3_pgt = stablehlo.tanh %vitb3_pgu : tensor<32x197x768xf32>
    %vitb3_pgone = stablehlo.constant dense<1.0> : tensor<32x197x768xf32>
    %vitb3_pgopt = stablehlo.add %vitb3_pgone, %vitb3_pgt : tensor<32x197x768xf32>
    %vitb3_pgchalf = stablehlo.constant dense<0.5> : tensor<32x197x768xf32>
    %vitb3_pghx = stablehlo.multiply %vitb3_pgchalf, %vitb3_ph1 : tensor<32x197x768xf32>
    %vitb3_pga = stablehlo.multiply %vitb3_pghx, %vitb3_pgopt : tensor<32x197x768xf32>
    %vitb3_py2d = stablehlo.dot_general %vitb3_pga, %Wfc2_3, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x192xf32>) -> tensor<32x197x192xf32>
    %vitb3_py2bb = stablehlo.broadcast_in_dim %bfc2_3, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb3_py = stablehlo.add %vitb3_py2d, %vitb3_py2bb : tensor<32x197x192xf32>
    %vitb3_out = stablehlo.add %vitb3_r1, %vitb3_py : tensor<32x197x192xf32>
    %vitb4_1sum = stablehlo.reduce(%vitb3_out init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb4_1nf = stablehlo.constant dense<192.0> : tensor<32x197xf32>
    %vitb4_1mu = stablehlo.divide %vitb4_1sum, %vitb4_1nf : tensor<32x197xf32>
    %vitb4_1mub = stablehlo.broadcast_in_dim %vitb4_1mu, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb4_1xc = stablehlo.subtract %vitb3_out, %vitb4_1mub : tensor<32x197x192xf32>
    %vitb4_1sq = stablehlo.multiply %vitb4_1xc, %vitb4_1xc : tensor<32x197x192xf32>
    %vitb4_1vsum = stablehlo.reduce(%vitb4_1sq init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb4_1var = stablehlo.divide %vitb4_1vsum, %vitb4_1nf : tensor<32x197xf32>
    %vitb4_1eps = stablehlo.constant dense<1.0e-5> : tensor<32x197xf32>
    %vitb4_1ve = stablehlo.add %vitb4_1var, %vitb4_1eps : tensor<32x197xf32>
    %vitb4_1istd = stablehlo.rsqrt %vitb4_1ve : tensor<32x197xf32>
    %vitb4_1istdb = stablehlo.broadcast_in_dim %vitb4_1istd, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb4_1xhat = stablehlo.multiply %vitb4_1xc, %vitb4_1istdb : tensor<32x197x192xf32>
    %vitb4_1gb = stablehlo.broadcast_in_dim %g1_4, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb4_1bbc = stablehlo.broadcast_in_dim %b1_4, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb4_1gx = stablehlo.multiply %vitb4_1xhat, %vitb4_1gb : tensor<32x197x192xf32>
    %vitb4_1y = stablehlo.add %vitb4_1gx, %vitb4_1bbc : tensor<32x197x192xf32>
    %vitb4_mQd = stablehlo.dot_general %vitb4_1y, %Wq_4, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb4_mQbb = stablehlo.broadcast_in_dim %bq_4, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb4_mQ = stablehlo.add %vitb4_mQd, %vitb4_mQbb : tensor<32x197x192xf32>
    %vitb4_mKd = stablehlo.dot_general %vitb4_1y, %Wk_4, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb4_mKbb = stablehlo.broadcast_in_dim %bk_4, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb4_mK = stablehlo.add %vitb4_mKd, %vitb4_mKbb : tensor<32x197x192xf32>
    %vitb4_mVd = stablehlo.dot_general %vitb4_1y, %Wv_4, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb4_mVbb = stablehlo.broadcast_in_dim %bv_4, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb4_mV = stablehlo.add %vitb4_mVd, %vitb4_mVbb : tensor<32x197x192xf32>
    %vitb4_mQhr = stablehlo.reshape %vitb4_mQ : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb4_mQh = stablehlo.transpose %vitb4_mQhr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb4_mKhr = stablehlo.reshape %vitb4_mK : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb4_mKh = stablehlo.transpose %vitb4_mKhr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb4_mVhr = stablehlo.reshape %vitb4_mV : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb4_mVh = stablehlo.transpose %vitb4_mVhr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb4_mS = stablehlo.dot_general %vitb4_mQh, %vitb4_mKh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x64xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x197xf32>
    %vitb4_mscl = stablehlo.constant dense<0.125> : tensor<32x3x197x197xf32>
    %vitb4_mSs = stablehlo.multiply %vitb4_mS, %vitb4_mscl : tensor<32x3x197x197xf32>
    %vitb4_mse = stablehlo.exponential %vitb4_mSs : tensor<32x3x197x197xf32>
    %vitb4_msum = stablehlo.reduce(%vitb4_mse init: %sc) applies stablehlo.add across dimensions = [3] : (tensor<32x3x197x197xf32>, tensor<f32>) -> tensor<32x3x197xf32>
    %vitb4_msumb = stablehlo.broadcast_in_dim %vitb4_msum, dims = [0, 1, 2] : (tensor<32x3x197xf32>) -> tensor<32x3x197x197xf32>
    %vitb4_mW = stablehlo.divide %vitb4_mse, %vitb4_msumb : tensor<32x3x197x197xf32>
    %vitb4_mA = stablehlo.dot_general %vitb4_mW, %vitb4_mVh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb4_mAT = stablehlo.transpose %vitb4_mA, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb4_mP = stablehlo.reshape %vitb4_mAT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb4_mod = stablehlo.dot_general %vitb4_mP, %Wo_4, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb4_mobb = stablehlo.broadcast_in_dim %bo_4, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb4_mO = stablehlo.add %vitb4_mod, %vitb4_mobb : tensor<32x197x192xf32>
    %vitb4_r1 = stablehlo.add %vitb3_out, %vitb4_mO : tensor<32x197x192xf32>
    %vitb4_2sum = stablehlo.reduce(%vitb4_r1 init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb4_2nf = stablehlo.constant dense<192.0> : tensor<32x197xf32>
    %vitb4_2mu = stablehlo.divide %vitb4_2sum, %vitb4_2nf : tensor<32x197xf32>
    %vitb4_2mub = stablehlo.broadcast_in_dim %vitb4_2mu, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb4_2xc = stablehlo.subtract %vitb4_r1, %vitb4_2mub : tensor<32x197x192xf32>
    %vitb4_2sq = stablehlo.multiply %vitb4_2xc, %vitb4_2xc : tensor<32x197x192xf32>
    %vitb4_2vsum = stablehlo.reduce(%vitb4_2sq init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb4_2var = stablehlo.divide %vitb4_2vsum, %vitb4_2nf : tensor<32x197xf32>
    %vitb4_2eps = stablehlo.constant dense<1.0e-5> : tensor<32x197xf32>
    %vitb4_2ve = stablehlo.add %vitb4_2var, %vitb4_2eps : tensor<32x197xf32>
    %vitb4_2istd = stablehlo.rsqrt %vitb4_2ve : tensor<32x197xf32>
    %vitb4_2istdb = stablehlo.broadcast_in_dim %vitb4_2istd, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb4_2xhat = stablehlo.multiply %vitb4_2xc, %vitb4_2istdb : tensor<32x197x192xf32>
    %vitb4_2gb = stablehlo.broadcast_in_dim %g2_4, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb4_2bbc = stablehlo.broadcast_in_dim %b2_4, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb4_2gx = stablehlo.multiply %vitb4_2xhat, %vitb4_2gb : tensor<32x197x192xf32>
    %vitb4_2y = stablehlo.add %vitb4_2gx, %vitb4_2bbc : tensor<32x197x192xf32>
    %vitb4_ph1d = stablehlo.dot_general %vitb4_2y, %Wfc1_4, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x768xf32>) -> tensor<32x197x768xf32>
    %vitb4_ph1bb = stablehlo.broadcast_in_dim %bfc1_4, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %vitb4_ph1 = stablehlo.add %vitb4_ph1d, %vitb4_ph1bb : tensor<32x197x768xf32>
    %vitb4_pgx2 = stablehlo.multiply %vitb4_ph1, %vitb4_ph1 : tensor<32x197x768xf32>
    %vitb4_pgx3 = stablehlo.multiply %vitb4_pgx2, %vitb4_ph1 : tensor<32x197x768xf32>
    %vitb4_pgck = stablehlo.constant dense<0.044715> : tensor<32x197x768xf32>
    %vitb4_pgkx3 = stablehlo.multiply %vitb4_pgck, %vitb4_pgx3 : tensor<32x197x768xf32>
    %vitb4_pginn = stablehlo.add %vitb4_ph1, %vitb4_pgkx3 : tensor<32x197x768xf32>
    %vitb4_pgcsqrt = stablehlo.constant dense<0.7978845608028654> : tensor<32x197x768xf32>
    %vitb4_pgu = stablehlo.multiply %vitb4_pgcsqrt, %vitb4_pginn : tensor<32x197x768xf32>
    %vitb4_pgt = stablehlo.tanh %vitb4_pgu : tensor<32x197x768xf32>
    %vitb4_pgone = stablehlo.constant dense<1.0> : tensor<32x197x768xf32>
    %vitb4_pgopt = stablehlo.add %vitb4_pgone, %vitb4_pgt : tensor<32x197x768xf32>
    %vitb4_pgchalf = stablehlo.constant dense<0.5> : tensor<32x197x768xf32>
    %vitb4_pghx = stablehlo.multiply %vitb4_pgchalf, %vitb4_ph1 : tensor<32x197x768xf32>
    %vitb4_pga = stablehlo.multiply %vitb4_pghx, %vitb4_pgopt : tensor<32x197x768xf32>
    %vitb4_py2d = stablehlo.dot_general %vitb4_pga, %Wfc2_4, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x192xf32>) -> tensor<32x197x192xf32>
    %vitb4_py2bb = stablehlo.broadcast_in_dim %bfc2_4, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb4_py = stablehlo.add %vitb4_py2d, %vitb4_py2bb : tensor<32x197x192xf32>
    %vitb4_out = stablehlo.add %vitb4_r1, %vitb4_py : tensor<32x197x192xf32>
    %vitb5_1sum = stablehlo.reduce(%vitb4_out init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb5_1nf = stablehlo.constant dense<192.0> : tensor<32x197xf32>
    %vitb5_1mu = stablehlo.divide %vitb5_1sum, %vitb5_1nf : tensor<32x197xf32>
    %vitb5_1mub = stablehlo.broadcast_in_dim %vitb5_1mu, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb5_1xc = stablehlo.subtract %vitb4_out, %vitb5_1mub : tensor<32x197x192xf32>
    %vitb5_1sq = stablehlo.multiply %vitb5_1xc, %vitb5_1xc : tensor<32x197x192xf32>
    %vitb5_1vsum = stablehlo.reduce(%vitb5_1sq init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb5_1var = stablehlo.divide %vitb5_1vsum, %vitb5_1nf : tensor<32x197xf32>
    %vitb5_1eps = stablehlo.constant dense<1.0e-5> : tensor<32x197xf32>
    %vitb5_1ve = stablehlo.add %vitb5_1var, %vitb5_1eps : tensor<32x197xf32>
    %vitb5_1istd = stablehlo.rsqrt %vitb5_1ve : tensor<32x197xf32>
    %vitb5_1istdb = stablehlo.broadcast_in_dim %vitb5_1istd, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb5_1xhat = stablehlo.multiply %vitb5_1xc, %vitb5_1istdb : tensor<32x197x192xf32>
    %vitb5_1gb = stablehlo.broadcast_in_dim %g1_5, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb5_1bbc = stablehlo.broadcast_in_dim %b1_5, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb5_1gx = stablehlo.multiply %vitb5_1xhat, %vitb5_1gb : tensor<32x197x192xf32>
    %vitb5_1y = stablehlo.add %vitb5_1gx, %vitb5_1bbc : tensor<32x197x192xf32>
    %vitb5_mQd = stablehlo.dot_general %vitb5_1y, %Wq_5, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb5_mQbb = stablehlo.broadcast_in_dim %bq_5, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb5_mQ = stablehlo.add %vitb5_mQd, %vitb5_mQbb : tensor<32x197x192xf32>
    %vitb5_mKd = stablehlo.dot_general %vitb5_1y, %Wk_5, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb5_mKbb = stablehlo.broadcast_in_dim %bk_5, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb5_mK = stablehlo.add %vitb5_mKd, %vitb5_mKbb : tensor<32x197x192xf32>
    %vitb5_mVd = stablehlo.dot_general %vitb5_1y, %Wv_5, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb5_mVbb = stablehlo.broadcast_in_dim %bv_5, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb5_mV = stablehlo.add %vitb5_mVd, %vitb5_mVbb : tensor<32x197x192xf32>
    %vitb5_mQhr = stablehlo.reshape %vitb5_mQ : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb5_mQh = stablehlo.transpose %vitb5_mQhr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb5_mKhr = stablehlo.reshape %vitb5_mK : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb5_mKh = stablehlo.transpose %vitb5_mKhr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb5_mVhr = stablehlo.reshape %vitb5_mV : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb5_mVh = stablehlo.transpose %vitb5_mVhr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb5_mS = stablehlo.dot_general %vitb5_mQh, %vitb5_mKh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x64xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x197xf32>
    %vitb5_mscl = stablehlo.constant dense<0.125> : tensor<32x3x197x197xf32>
    %vitb5_mSs = stablehlo.multiply %vitb5_mS, %vitb5_mscl : tensor<32x3x197x197xf32>
    %vitb5_mse = stablehlo.exponential %vitb5_mSs : tensor<32x3x197x197xf32>
    %vitb5_msum = stablehlo.reduce(%vitb5_mse init: %sc) applies stablehlo.add across dimensions = [3] : (tensor<32x3x197x197xf32>, tensor<f32>) -> tensor<32x3x197xf32>
    %vitb5_msumb = stablehlo.broadcast_in_dim %vitb5_msum, dims = [0, 1, 2] : (tensor<32x3x197xf32>) -> tensor<32x3x197x197xf32>
    %vitb5_mW = stablehlo.divide %vitb5_mse, %vitb5_msumb : tensor<32x3x197x197xf32>
    %vitb5_mA = stablehlo.dot_general %vitb5_mW, %vitb5_mVh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb5_mAT = stablehlo.transpose %vitb5_mA, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb5_mP = stablehlo.reshape %vitb5_mAT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb5_mod = stablehlo.dot_general %vitb5_mP, %Wo_5, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb5_mobb = stablehlo.broadcast_in_dim %bo_5, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb5_mO = stablehlo.add %vitb5_mod, %vitb5_mobb : tensor<32x197x192xf32>
    %vitb5_r1 = stablehlo.add %vitb4_out, %vitb5_mO : tensor<32x197x192xf32>
    %vitb5_2sum = stablehlo.reduce(%vitb5_r1 init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb5_2nf = stablehlo.constant dense<192.0> : tensor<32x197xf32>
    %vitb5_2mu = stablehlo.divide %vitb5_2sum, %vitb5_2nf : tensor<32x197xf32>
    %vitb5_2mub = stablehlo.broadcast_in_dim %vitb5_2mu, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb5_2xc = stablehlo.subtract %vitb5_r1, %vitb5_2mub : tensor<32x197x192xf32>
    %vitb5_2sq = stablehlo.multiply %vitb5_2xc, %vitb5_2xc : tensor<32x197x192xf32>
    %vitb5_2vsum = stablehlo.reduce(%vitb5_2sq init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb5_2var = stablehlo.divide %vitb5_2vsum, %vitb5_2nf : tensor<32x197xf32>
    %vitb5_2eps = stablehlo.constant dense<1.0e-5> : tensor<32x197xf32>
    %vitb5_2ve = stablehlo.add %vitb5_2var, %vitb5_2eps : tensor<32x197xf32>
    %vitb5_2istd = stablehlo.rsqrt %vitb5_2ve : tensor<32x197xf32>
    %vitb5_2istdb = stablehlo.broadcast_in_dim %vitb5_2istd, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb5_2xhat = stablehlo.multiply %vitb5_2xc, %vitb5_2istdb : tensor<32x197x192xf32>
    %vitb5_2gb = stablehlo.broadcast_in_dim %g2_5, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb5_2bbc = stablehlo.broadcast_in_dim %b2_5, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb5_2gx = stablehlo.multiply %vitb5_2xhat, %vitb5_2gb : tensor<32x197x192xf32>
    %vitb5_2y = stablehlo.add %vitb5_2gx, %vitb5_2bbc : tensor<32x197x192xf32>
    %vitb5_ph1d = stablehlo.dot_general %vitb5_2y, %Wfc1_5, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x768xf32>) -> tensor<32x197x768xf32>
    %vitb5_ph1bb = stablehlo.broadcast_in_dim %bfc1_5, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %vitb5_ph1 = stablehlo.add %vitb5_ph1d, %vitb5_ph1bb : tensor<32x197x768xf32>
    %vitb5_pgx2 = stablehlo.multiply %vitb5_ph1, %vitb5_ph1 : tensor<32x197x768xf32>
    %vitb5_pgx3 = stablehlo.multiply %vitb5_pgx2, %vitb5_ph1 : tensor<32x197x768xf32>
    %vitb5_pgck = stablehlo.constant dense<0.044715> : tensor<32x197x768xf32>
    %vitb5_pgkx3 = stablehlo.multiply %vitb5_pgck, %vitb5_pgx3 : tensor<32x197x768xf32>
    %vitb5_pginn = stablehlo.add %vitb5_ph1, %vitb5_pgkx3 : tensor<32x197x768xf32>
    %vitb5_pgcsqrt = stablehlo.constant dense<0.7978845608028654> : tensor<32x197x768xf32>
    %vitb5_pgu = stablehlo.multiply %vitb5_pgcsqrt, %vitb5_pginn : tensor<32x197x768xf32>
    %vitb5_pgt = stablehlo.tanh %vitb5_pgu : tensor<32x197x768xf32>
    %vitb5_pgone = stablehlo.constant dense<1.0> : tensor<32x197x768xf32>
    %vitb5_pgopt = stablehlo.add %vitb5_pgone, %vitb5_pgt : tensor<32x197x768xf32>
    %vitb5_pgchalf = stablehlo.constant dense<0.5> : tensor<32x197x768xf32>
    %vitb5_pghx = stablehlo.multiply %vitb5_pgchalf, %vitb5_ph1 : tensor<32x197x768xf32>
    %vitb5_pga = stablehlo.multiply %vitb5_pghx, %vitb5_pgopt : tensor<32x197x768xf32>
    %vitb5_py2d = stablehlo.dot_general %vitb5_pga, %Wfc2_5, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x192xf32>) -> tensor<32x197x192xf32>
    %vitb5_py2bb = stablehlo.broadcast_in_dim %bfc2_5, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb5_py = stablehlo.add %vitb5_py2d, %vitb5_py2bb : tensor<32x197x192xf32>
    %vitb5_out = stablehlo.add %vitb5_r1, %vitb5_py : tensor<32x197x192xf32>
    %vitb6_1sum = stablehlo.reduce(%vitb5_out init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb6_1nf = stablehlo.constant dense<192.0> : tensor<32x197xf32>
    %vitb6_1mu = stablehlo.divide %vitb6_1sum, %vitb6_1nf : tensor<32x197xf32>
    %vitb6_1mub = stablehlo.broadcast_in_dim %vitb6_1mu, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb6_1xc = stablehlo.subtract %vitb5_out, %vitb6_1mub : tensor<32x197x192xf32>
    %vitb6_1sq = stablehlo.multiply %vitb6_1xc, %vitb6_1xc : tensor<32x197x192xf32>
    %vitb6_1vsum = stablehlo.reduce(%vitb6_1sq init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb6_1var = stablehlo.divide %vitb6_1vsum, %vitb6_1nf : tensor<32x197xf32>
    %vitb6_1eps = stablehlo.constant dense<1.0e-5> : tensor<32x197xf32>
    %vitb6_1ve = stablehlo.add %vitb6_1var, %vitb6_1eps : tensor<32x197xf32>
    %vitb6_1istd = stablehlo.rsqrt %vitb6_1ve : tensor<32x197xf32>
    %vitb6_1istdb = stablehlo.broadcast_in_dim %vitb6_1istd, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb6_1xhat = stablehlo.multiply %vitb6_1xc, %vitb6_1istdb : tensor<32x197x192xf32>
    %vitb6_1gb = stablehlo.broadcast_in_dim %g1_6, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb6_1bbc = stablehlo.broadcast_in_dim %b1_6, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb6_1gx = stablehlo.multiply %vitb6_1xhat, %vitb6_1gb : tensor<32x197x192xf32>
    %vitb6_1y = stablehlo.add %vitb6_1gx, %vitb6_1bbc : tensor<32x197x192xf32>
    %vitb6_mQd = stablehlo.dot_general %vitb6_1y, %Wq_6, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb6_mQbb = stablehlo.broadcast_in_dim %bq_6, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb6_mQ = stablehlo.add %vitb6_mQd, %vitb6_mQbb : tensor<32x197x192xf32>
    %vitb6_mKd = stablehlo.dot_general %vitb6_1y, %Wk_6, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb6_mKbb = stablehlo.broadcast_in_dim %bk_6, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb6_mK = stablehlo.add %vitb6_mKd, %vitb6_mKbb : tensor<32x197x192xf32>
    %vitb6_mVd = stablehlo.dot_general %vitb6_1y, %Wv_6, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb6_mVbb = stablehlo.broadcast_in_dim %bv_6, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb6_mV = stablehlo.add %vitb6_mVd, %vitb6_mVbb : tensor<32x197x192xf32>
    %vitb6_mQhr = stablehlo.reshape %vitb6_mQ : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb6_mQh = stablehlo.transpose %vitb6_mQhr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb6_mKhr = stablehlo.reshape %vitb6_mK : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb6_mKh = stablehlo.transpose %vitb6_mKhr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb6_mVhr = stablehlo.reshape %vitb6_mV : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb6_mVh = stablehlo.transpose %vitb6_mVhr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb6_mS = stablehlo.dot_general %vitb6_mQh, %vitb6_mKh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x64xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x197xf32>
    %vitb6_mscl = stablehlo.constant dense<0.125> : tensor<32x3x197x197xf32>
    %vitb6_mSs = stablehlo.multiply %vitb6_mS, %vitb6_mscl : tensor<32x3x197x197xf32>
    %vitb6_mse = stablehlo.exponential %vitb6_mSs : tensor<32x3x197x197xf32>
    %vitb6_msum = stablehlo.reduce(%vitb6_mse init: %sc) applies stablehlo.add across dimensions = [3] : (tensor<32x3x197x197xf32>, tensor<f32>) -> tensor<32x3x197xf32>
    %vitb6_msumb = stablehlo.broadcast_in_dim %vitb6_msum, dims = [0, 1, 2] : (tensor<32x3x197xf32>) -> tensor<32x3x197x197xf32>
    %vitb6_mW = stablehlo.divide %vitb6_mse, %vitb6_msumb : tensor<32x3x197x197xf32>
    %vitb6_mA = stablehlo.dot_general %vitb6_mW, %vitb6_mVh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb6_mAT = stablehlo.transpose %vitb6_mA, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb6_mP = stablehlo.reshape %vitb6_mAT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb6_mod = stablehlo.dot_general %vitb6_mP, %Wo_6, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb6_mobb = stablehlo.broadcast_in_dim %bo_6, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb6_mO = stablehlo.add %vitb6_mod, %vitb6_mobb : tensor<32x197x192xf32>
    %vitb6_r1 = stablehlo.add %vitb5_out, %vitb6_mO : tensor<32x197x192xf32>
    %vitb6_2sum = stablehlo.reduce(%vitb6_r1 init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb6_2nf = stablehlo.constant dense<192.0> : tensor<32x197xf32>
    %vitb6_2mu = stablehlo.divide %vitb6_2sum, %vitb6_2nf : tensor<32x197xf32>
    %vitb6_2mub = stablehlo.broadcast_in_dim %vitb6_2mu, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb6_2xc = stablehlo.subtract %vitb6_r1, %vitb6_2mub : tensor<32x197x192xf32>
    %vitb6_2sq = stablehlo.multiply %vitb6_2xc, %vitb6_2xc : tensor<32x197x192xf32>
    %vitb6_2vsum = stablehlo.reduce(%vitb6_2sq init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb6_2var = stablehlo.divide %vitb6_2vsum, %vitb6_2nf : tensor<32x197xf32>
    %vitb6_2eps = stablehlo.constant dense<1.0e-5> : tensor<32x197xf32>
    %vitb6_2ve = stablehlo.add %vitb6_2var, %vitb6_2eps : tensor<32x197xf32>
    %vitb6_2istd = stablehlo.rsqrt %vitb6_2ve : tensor<32x197xf32>
    %vitb6_2istdb = stablehlo.broadcast_in_dim %vitb6_2istd, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb6_2xhat = stablehlo.multiply %vitb6_2xc, %vitb6_2istdb : tensor<32x197x192xf32>
    %vitb6_2gb = stablehlo.broadcast_in_dim %g2_6, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb6_2bbc = stablehlo.broadcast_in_dim %b2_6, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb6_2gx = stablehlo.multiply %vitb6_2xhat, %vitb6_2gb : tensor<32x197x192xf32>
    %vitb6_2y = stablehlo.add %vitb6_2gx, %vitb6_2bbc : tensor<32x197x192xf32>
    %vitb6_ph1d = stablehlo.dot_general %vitb6_2y, %Wfc1_6, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x768xf32>) -> tensor<32x197x768xf32>
    %vitb6_ph1bb = stablehlo.broadcast_in_dim %bfc1_6, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %vitb6_ph1 = stablehlo.add %vitb6_ph1d, %vitb6_ph1bb : tensor<32x197x768xf32>
    %vitb6_pgx2 = stablehlo.multiply %vitb6_ph1, %vitb6_ph1 : tensor<32x197x768xf32>
    %vitb6_pgx3 = stablehlo.multiply %vitb6_pgx2, %vitb6_ph1 : tensor<32x197x768xf32>
    %vitb6_pgck = stablehlo.constant dense<0.044715> : tensor<32x197x768xf32>
    %vitb6_pgkx3 = stablehlo.multiply %vitb6_pgck, %vitb6_pgx3 : tensor<32x197x768xf32>
    %vitb6_pginn = stablehlo.add %vitb6_ph1, %vitb6_pgkx3 : tensor<32x197x768xf32>
    %vitb6_pgcsqrt = stablehlo.constant dense<0.7978845608028654> : tensor<32x197x768xf32>
    %vitb6_pgu = stablehlo.multiply %vitb6_pgcsqrt, %vitb6_pginn : tensor<32x197x768xf32>
    %vitb6_pgt = stablehlo.tanh %vitb6_pgu : tensor<32x197x768xf32>
    %vitb6_pgone = stablehlo.constant dense<1.0> : tensor<32x197x768xf32>
    %vitb6_pgopt = stablehlo.add %vitb6_pgone, %vitb6_pgt : tensor<32x197x768xf32>
    %vitb6_pgchalf = stablehlo.constant dense<0.5> : tensor<32x197x768xf32>
    %vitb6_pghx = stablehlo.multiply %vitb6_pgchalf, %vitb6_ph1 : tensor<32x197x768xf32>
    %vitb6_pga = stablehlo.multiply %vitb6_pghx, %vitb6_pgopt : tensor<32x197x768xf32>
    %vitb6_py2d = stablehlo.dot_general %vitb6_pga, %Wfc2_6, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x192xf32>) -> tensor<32x197x192xf32>
    %vitb6_py2bb = stablehlo.broadcast_in_dim %bfc2_6, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb6_py = stablehlo.add %vitb6_py2d, %vitb6_py2bb : tensor<32x197x192xf32>
    %vitb6_out = stablehlo.add %vitb6_r1, %vitb6_py : tensor<32x197x192xf32>
    %vitb7_1sum = stablehlo.reduce(%vitb6_out init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb7_1nf = stablehlo.constant dense<192.0> : tensor<32x197xf32>
    %vitb7_1mu = stablehlo.divide %vitb7_1sum, %vitb7_1nf : tensor<32x197xf32>
    %vitb7_1mub = stablehlo.broadcast_in_dim %vitb7_1mu, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb7_1xc = stablehlo.subtract %vitb6_out, %vitb7_1mub : tensor<32x197x192xf32>
    %vitb7_1sq = stablehlo.multiply %vitb7_1xc, %vitb7_1xc : tensor<32x197x192xf32>
    %vitb7_1vsum = stablehlo.reduce(%vitb7_1sq init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb7_1var = stablehlo.divide %vitb7_1vsum, %vitb7_1nf : tensor<32x197xf32>
    %vitb7_1eps = stablehlo.constant dense<1.0e-5> : tensor<32x197xf32>
    %vitb7_1ve = stablehlo.add %vitb7_1var, %vitb7_1eps : tensor<32x197xf32>
    %vitb7_1istd = stablehlo.rsqrt %vitb7_1ve : tensor<32x197xf32>
    %vitb7_1istdb = stablehlo.broadcast_in_dim %vitb7_1istd, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb7_1xhat = stablehlo.multiply %vitb7_1xc, %vitb7_1istdb : tensor<32x197x192xf32>
    %vitb7_1gb = stablehlo.broadcast_in_dim %g1_7, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb7_1bbc = stablehlo.broadcast_in_dim %b1_7, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb7_1gx = stablehlo.multiply %vitb7_1xhat, %vitb7_1gb : tensor<32x197x192xf32>
    %vitb7_1y = stablehlo.add %vitb7_1gx, %vitb7_1bbc : tensor<32x197x192xf32>
    %vitb7_mQd = stablehlo.dot_general %vitb7_1y, %Wq_7, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb7_mQbb = stablehlo.broadcast_in_dim %bq_7, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb7_mQ = stablehlo.add %vitb7_mQd, %vitb7_mQbb : tensor<32x197x192xf32>
    %vitb7_mKd = stablehlo.dot_general %vitb7_1y, %Wk_7, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb7_mKbb = stablehlo.broadcast_in_dim %bk_7, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb7_mK = stablehlo.add %vitb7_mKd, %vitb7_mKbb : tensor<32x197x192xf32>
    %vitb7_mVd = stablehlo.dot_general %vitb7_1y, %Wv_7, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb7_mVbb = stablehlo.broadcast_in_dim %bv_7, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb7_mV = stablehlo.add %vitb7_mVd, %vitb7_mVbb : tensor<32x197x192xf32>
    %vitb7_mQhr = stablehlo.reshape %vitb7_mQ : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb7_mQh = stablehlo.transpose %vitb7_mQhr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb7_mKhr = stablehlo.reshape %vitb7_mK : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb7_mKh = stablehlo.transpose %vitb7_mKhr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb7_mVhr = stablehlo.reshape %vitb7_mV : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb7_mVh = stablehlo.transpose %vitb7_mVhr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb7_mS = stablehlo.dot_general %vitb7_mQh, %vitb7_mKh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x64xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x197xf32>
    %vitb7_mscl = stablehlo.constant dense<0.125> : tensor<32x3x197x197xf32>
    %vitb7_mSs = stablehlo.multiply %vitb7_mS, %vitb7_mscl : tensor<32x3x197x197xf32>
    %vitb7_mse = stablehlo.exponential %vitb7_mSs : tensor<32x3x197x197xf32>
    %vitb7_msum = stablehlo.reduce(%vitb7_mse init: %sc) applies stablehlo.add across dimensions = [3] : (tensor<32x3x197x197xf32>, tensor<f32>) -> tensor<32x3x197xf32>
    %vitb7_msumb = stablehlo.broadcast_in_dim %vitb7_msum, dims = [0, 1, 2] : (tensor<32x3x197xf32>) -> tensor<32x3x197x197xf32>
    %vitb7_mW = stablehlo.divide %vitb7_mse, %vitb7_msumb : tensor<32x3x197x197xf32>
    %vitb7_mA = stablehlo.dot_general %vitb7_mW, %vitb7_mVh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb7_mAT = stablehlo.transpose %vitb7_mA, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb7_mP = stablehlo.reshape %vitb7_mAT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb7_mod = stablehlo.dot_general %vitb7_mP, %Wo_7, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb7_mobb = stablehlo.broadcast_in_dim %bo_7, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb7_mO = stablehlo.add %vitb7_mod, %vitb7_mobb : tensor<32x197x192xf32>
    %vitb7_r1 = stablehlo.add %vitb6_out, %vitb7_mO : tensor<32x197x192xf32>
    %vitb7_2sum = stablehlo.reduce(%vitb7_r1 init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb7_2nf = stablehlo.constant dense<192.0> : tensor<32x197xf32>
    %vitb7_2mu = stablehlo.divide %vitb7_2sum, %vitb7_2nf : tensor<32x197xf32>
    %vitb7_2mub = stablehlo.broadcast_in_dim %vitb7_2mu, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb7_2xc = stablehlo.subtract %vitb7_r1, %vitb7_2mub : tensor<32x197x192xf32>
    %vitb7_2sq = stablehlo.multiply %vitb7_2xc, %vitb7_2xc : tensor<32x197x192xf32>
    %vitb7_2vsum = stablehlo.reduce(%vitb7_2sq init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb7_2var = stablehlo.divide %vitb7_2vsum, %vitb7_2nf : tensor<32x197xf32>
    %vitb7_2eps = stablehlo.constant dense<1.0e-5> : tensor<32x197xf32>
    %vitb7_2ve = stablehlo.add %vitb7_2var, %vitb7_2eps : tensor<32x197xf32>
    %vitb7_2istd = stablehlo.rsqrt %vitb7_2ve : tensor<32x197xf32>
    %vitb7_2istdb = stablehlo.broadcast_in_dim %vitb7_2istd, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb7_2xhat = stablehlo.multiply %vitb7_2xc, %vitb7_2istdb : tensor<32x197x192xf32>
    %vitb7_2gb = stablehlo.broadcast_in_dim %g2_7, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb7_2bbc = stablehlo.broadcast_in_dim %b2_7, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb7_2gx = stablehlo.multiply %vitb7_2xhat, %vitb7_2gb : tensor<32x197x192xf32>
    %vitb7_2y = stablehlo.add %vitb7_2gx, %vitb7_2bbc : tensor<32x197x192xf32>
    %vitb7_ph1d = stablehlo.dot_general %vitb7_2y, %Wfc1_7, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x768xf32>) -> tensor<32x197x768xf32>
    %vitb7_ph1bb = stablehlo.broadcast_in_dim %bfc1_7, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %vitb7_ph1 = stablehlo.add %vitb7_ph1d, %vitb7_ph1bb : tensor<32x197x768xf32>
    %vitb7_pgx2 = stablehlo.multiply %vitb7_ph1, %vitb7_ph1 : tensor<32x197x768xf32>
    %vitb7_pgx3 = stablehlo.multiply %vitb7_pgx2, %vitb7_ph1 : tensor<32x197x768xf32>
    %vitb7_pgck = stablehlo.constant dense<0.044715> : tensor<32x197x768xf32>
    %vitb7_pgkx3 = stablehlo.multiply %vitb7_pgck, %vitb7_pgx3 : tensor<32x197x768xf32>
    %vitb7_pginn = stablehlo.add %vitb7_ph1, %vitb7_pgkx3 : tensor<32x197x768xf32>
    %vitb7_pgcsqrt = stablehlo.constant dense<0.7978845608028654> : tensor<32x197x768xf32>
    %vitb7_pgu = stablehlo.multiply %vitb7_pgcsqrt, %vitb7_pginn : tensor<32x197x768xf32>
    %vitb7_pgt = stablehlo.tanh %vitb7_pgu : tensor<32x197x768xf32>
    %vitb7_pgone = stablehlo.constant dense<1.0> : tensor<32x197x768xf32>
    %vitb7_pgopt = stablehlo.add %vitb7_pgone, %vitb7_pgt : tensor<32x197x768xf32>
    %vitb7_pgchalf = stablehlo.constant dense<0.5> : tensor<32x197x768xf32>
    %vitb7_pghx = stablehlo.multiply %vitb7_pgchalf, %vitb7_ph1 : tensor<32x197x768xf32>
    %vitb7_pga = stablehlo.multiply %vitb7_pghx, %vitb7_pgopt : tensor<32x197x768xf32>
    %vitb7_py2d = stablehlo.dot_general %vitb7_pga, %Wfc2_7, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x192xf32>) -> tensor<32x197x192xf32>
    %vitb7_py2bb = stablehlo.broadcast_in_dim %bfc2_7, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb7_py = stablehlo.add %vitb7_py2d, %vitb7_py2bb : tensor<32x197x192xf32>
    %vitb7_out = stablehlo.add %vitb7_r1, %vitb7_py : tensor<32x197x192xf32>
    %vitb8_1sum = stablehlo.reduce(%vitb7_out init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb8_1nf = stablehlo.constant dense<192.0> : tensor<32x197xf32>
    %vitb8_1mu = stablehlo.divide %vitb8_1sum, %vitb8_1nf : tensor<32x197xf32>
    %vitb8_1mub = stablehlo.broadcast_in_dim %vitb8_1mu, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb8_1xc = stablehlo.subtract %vitb7_out, %vitb8_1mub : tensor<32x197x192xf32>
    %vitb8_1sq = stablehlo.multiply %vitb8_1xc, %vitb8_1xc : tensor<32x197x192xf32>
    %vitb8_1vsum = stablehlo.reduce(%vitb8_1sq init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb8_1var = stablehlo.divide %vitb8_1vsum, %vitb8_1nf : tensor<32x197xf32>
    %vitb8_1eps = stablehlo.constant dense<1.0e-5> : tensor<32x197xf32>
    %vitb8_1ve = stablehlo.add %vitb8_1var, %vitb8_1eps : tensor<32x197xf32>
    %vitb8_1istd = stablehlo.rsqrt %vitb8_1ve : tensor<32x197xf32>
    %vitb8_1istdb = stablehlo.broadcast_in_dim %vitb8_1istd, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb8_1xhat = stablehlo.multiply %vitb8_1xc, %vitb8_1istdb : tensor<32x197x192xf32>
    %vitb8_1gb = stablehlo.broadcast_in_dim %g1_8, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb8_1bbc = stablehlo.broadcast_in_dim %b1_8, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb8_1gx = stablehlo.multiply %vitb8_1xhat, %vitb8_1gb : tensor<32x197x192xf32>
    %vitb8_1y = stablehlo.add %vitb8_1gx, %vitb8_1bbc : tensor<32x197x192xf32>
    %vitb8_mQd = stablehlo.dot_general %vitb8_1y, %Wq_8, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb8_mQbb = stablehlo.broadcast_in_dim %bq_8, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb8_mQ = stablehlo.add %vitb8_mQd, %vitb8_mQbb : tensor<32x197x192xf32>
    %vitb8_mKd = stablehlo.dot_general %vitb8_1y, %Wk_8, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb8_mKbb = stablehlo.broadcast_in_dim %bk_8, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb8_mK = stablehlo.add %vitb8_mKd, %vitb8_mKbb : tensor<32x197x192xf32>
    %vitb8_mVd = stablehlo.dot_general %vitb8_1y, %Wv_8, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb8_mVbb = stablehlo.broadcast_in_dim %bv_8, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb8_mV = stablehlo.add %vitb8_mVd, %vitb8_mVbb : tensor<32x197x192xf32>
    %vitb8_mQhr = stablehlo.reshape %vitb8_mQ : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb8_mQh = stablehlo.transpose %vitb8_mQhr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb8_mKhr = stablehlo.reshape %vitb8_mK : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb8_mKh = stablehlo.transpose %vitb8_mKhr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb8_mVhr = stablehlo.reshape %vitb8_mV : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb8_mVh = stablehlo.transpose %vitb8_mVhr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb8_mS = stablehlo.dot_general %vitb8_mQh, %vitb8_mKh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x64xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x197xf32>
    %vitb8_mscl = stablehlo.constant dense<0.125> : tensor<32x3x197x197xf32>
    %vitb8_mSs = stablehlo.multiply %vitb8_mS, %vitb8_mscl : tensor<32x3x197x197xf32>
    %vitb8_mse = stablehlo.exponential %vitb8_mSs : tensor<32x3x197x197xf32>
    %vitb8_msum = stablehlo.reduce(%vitb8_mse init: %sc) applies stablehlo.add across dimensions = [3] : (tensor<32x3x197x197xf32>, tensor<f32>) -> tensor<32x3x197xf32>
    %vitb8_msumb = stablehlo.broadcast_in_dim %vitb8_msum, dims = [0, 1, 2] : (tensor<32x3x197xf32>) -> tensor<32x3x197x197xf32>
    %vitb8_mW = stablehlo.divide %vitb8_mse, %vitb8_msumb : tensor<32x3x197x197xf32>
    %vitb8_mA = stablehlo.dot_general %vitb8_mW, %vitb8_mVh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb8_mAT = stablehlo.transpose %vitb8_mA, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb8_mP = stablehlo.reshape %vitb8_mAT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb8_mod = stablehlo.dot_general %vitb8_mP, %Wo_8, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb8_mobb = stablehlo.broadcast_in_dim %bo_8, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb8_mO = stablehlo.add %vitb8_mod, %vitb8_mobb : tensor<32x197x192xf32>
    %vitb8_r1 = stablehlo.add %vitb7_out, %vitb8_mO : tensor<32x197x192xf32>
    %vitb8_2sum = stablehlo.reduce(%vitb8_r1 init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb8_2nf = stablehlo.constant dense<192.0> : tensor<32x197xf32>
    %vitb8_2mu = stablehlo.divide %vitb8_2sum, %vitb8_2nf : tensor<32x197xf32>
    %vitb8_2mub = stablehlo.broadcast_in_dim %vitb8_2mu, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb8_2xc = stablehlo.subtract %vitb8_r1, %vitb8_2mub : tensor<32x197x192xf32>
    %vitb8_2sq = stablehlo.multiply %vitb8_2xc, %vitb8_2xc : tensor<32x197x192xf32>
    %vitb8_2vsum = stablehlo.reduce(%vitb8_2sq init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb8_2var = stablehlo.divide %vitb8_2vsum, %vitb8_2nf : tensor<32x197xf32>
    %vitb8_2eps = stablehlo.constant dense<1.0e-5> : tensor<32x197xf32>
    %vitb8_2ve = stablehlo.add %vitb8_2var, %vitb8_2eps : tensor<32x197xf32>
    %vitb8_2istd = stablehlo.rsqrt %vitb8_2ve : tensor<32x197xf32>
    %vitb8_2istdb = stablehlo.broadcast_in_dim %vitb8_2istd, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb8_2xhat = stablehlo.multiply %vitb8_2xc, %vitb8_2istdb : tensor<32x197x192xf32>
    %vitb8_2gb = stablehlo.broadcast_in_dim %g2_8, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb8_2bbc = stablehlo.broadcast_in_dim %b2_8, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb8_2gx = stablehlo.multiply %vitb8_2xhat, %vitb8_2gb : tensor<32x197x192xf32>
    %vitb8_2y = stablehlo.add %vitb8_2gx, %vitb8_2bbc : tensor<32x197x192xf32>
    %vitb8_ph1d = stablehlo.dot_general %vitb8_2y, %Wfc1_8, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x768xf32>) -> tensor<32x197x768xf32>
    %vitb8_ph1bb = stablehlo.broadcast_in_dim %bfc1_8, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %vitb8_ph1 = stablehlo.add %vitb8_ph1d, %vitb8_ph1bb : tensor<32x197x768xf32>
    %vitb8_pgx2 = stablehlo.multiply %vitb8_ph1, %vitb8_ph1 : tensor<32x197x768xf32>
    %vitb8_pgx3 = stablehlo.multiply %vitb8_pgx2, %vitb8_ph1 : tensor<32x197x768xf32>
    %vitb8_pgck = stablehlo.constant dense<0.044715> : tensor<32x197x768xf32>
    %vitb8_pgkx3 = stablehlo.multiply %vitb8_pgck, %vitb8_pgx3 : tensor<32x197x768xf32>
    %vitb8_pginn = stablehlo.add %vitb8_ph1, %vitb8_pgkx3 : tensor<32x197x768xf32>
    %vitb8_pgcsqrt = stablehlo.constant dense<0.7978845608028654> : tensor<32x197x768xf32>
    %vitb8_pgu = stablehlo.multiply %vitb8_pgcsqrt, %vitb8_pginn : tensor<32x197x768xf32>
    %vitb8_pgt = stablehlo.tanh %vitb8_pgu : tensor<32x197x768xf32>
    %vitb8_pgone = stablehlo.constant dense<1.0> : tensor<32x197x768xf32>
    %vitb8_pgopt = stablehlo.add %vitb8_pgone, %vitb8_pgt : tensor<32x197x768xf32>
    %vitb8_pgchalf = stablehlo.constant dense<0.5> : tensor<32x197x768xf32>
    %vitb8_pghx = stablehlo.multiply %vitb8_pgchalf, %vitb8_ph1 : tensor<32x197x768xf32>
    %vitb8_pga = stablehlo.multiply %vitb8_pghx, %vitb8_pgopt : tensor<32x197x768xf32>
    %vitb8_py2d = stablehlo.dot_general %vitb8_pga, %Wfc2_8, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x192xf32>) -> tensor<32x197x192xf32>
    %vitb8_py2bb = stablehlo.broadcast_in_dim %bfc2_8, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb8_py = stablehlo.add %vitb8_py2d, %vitb8_py2bb : tensor<32x197x192xf32>
    %vitb8_out = stablehlo.add %vitb8_r1, %vitb8_py : tensor<32x197x192xf32>
    %vitb9_1sum = stablehlo.reduce(%vitb8_out init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb9_1nf = stablehlo.constant dense<192.0> : tensor<32x197xf32>
    %vitb9_1mu = stablehlo.divide %vitb9_1sum, %vitb9_1nf : tensor<32x197xf32>
    %vitb9_1mub = stablehlo.broadcast_in_dim %vitb9_1mu, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb9_1xc = stablehlo.subtract %vitb8_out, %vitb9_1mub : tensor<32x197x192xf32>
    %vitb9_1sq = stablehlo.multiply %vitb9_1xc, %vitb9_1xc : tensor<32x197x192xf32>
    %vitb9_1vsum = stablehlo.reduce(%vitb9_1sq init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb9_1var = stablehlo.divide %vitb9_1vsum, %vitb9_1nf : tensor<32x197xf32>
    %vitb9_1eps = stablehlo.constant dense<1.0e-5> : tensor<32x197xf32>
    %vitb9_1ve = stablehlo.add %vitb9_1var, %vitb9_1eps : tensor<32x197xf32>
    %vitb9_1istd = stablehlo.rsqrt %vitb9_1ve : tensor<32x197xf32>
    %vitb9_1istdb = stablehlo.broadcast_in_dim %vitb9_1istd, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb9_1xhat = stablehlo.multiply %vitb9_1xc, %vitb9_1istdb : tensor<32x197x192xf32>
    %vitb9_1gb = stablehlo.broadcast_in_dim %g1_9, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb9_1bbc = stablehlo.broadcast_in_dim %b1_9, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb9_1gx = stablehlo.multiply %vitb9_1xhat, %vitb9_1gb : tensor<32x197x192xf32>
    %vitb9_1y = stablehlo.add %vitb9_1gx, %vitb9_1bbc : tensor<32x197x192xf32>
    %vitb9_mQd = stablehlo.dot_general %vitb9_1y, %Wq_9, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb9_mQbb = stablehlo.broadcast_in_dim %bq_9, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb9_mQ = stablehlo.add %vitb9_mQd, %vitb9_mQbb : tensor<32x197x192xf32>
    %vitb9_mKd = stablehlo.dot_general %vitb9_1y, %Wk_9, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb9_mKbb = stablehlo.broadcast_in_dim %bk_9, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb9_mK = stablehlo.add %vitb9_mKd, %vitb9_mKbb : tensor<32x197x192xf32>
    %vitb9_mVd = stablehlo.dot_general %vitb9_1y, %Wv_9, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb9_mVbb = stablehlo.broadcast_in_dim %bv_9, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb9_mV = stablehlo.add %vitb9_mVd, %vitb9_mVbb : tensor<32x197x192xf32>
    %vitb9_mQhr = stablehlo.reshape %vitb9_mQ : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb9_mQh = stablehlo.transpose %vitb9_mQhr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb9_mKhr = stablehlo.reshape %vitb9_mK : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb9_mKh = stablehlo.transpose %vitb9_mKhr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb9_mVhr = stablehlo.reshape %vitb9_mV : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb9_mVh = stablehlo.transpose %vitb9_mVhr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb9_mS = stablehlo.dot_general %vitb9_mQh, %vitb9_mKh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x64xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x197xf32>
    %vitb9_mscl = stablehlo.constant dense<0.125> : tensor<32x3x197x197xf32>
    %vitb9_mSs = stablehlo.multiply %vitb9_mS, %vitb9_mscl : tensor<32x3x197x197xf32>
    %vitb9_mse = stablehlo.exponential %vitb9_mSs : tensor<32x3x197x197xf32>
    %vitb9_msum = stablehlo.reduce(%vitb9_mse init: %sc) applies stablehlo.add across dimensions = [3] : (tensor<32x3x197x197xf32>, tensor<f32>) -> tensor<32x3x197xf32>
    %vitb9_msumb = stablehlo.broadcast_in_dim %vitb9_msum, dims = [0, 1, 2] : (tensor<32x3x197xf32>) -> tensor<32x3x197x197xf32>
    %vitb9_mW = stablehlo.divide %vitb9_mse, %vitb9_msumb : tensor<32x3x197x197xf32>
    %vitb9_mA = stablehlo.dot_general %vitb9_mW, %vitb9_mVh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb9_mAT = stablehlo.transpose %vitb9_mA, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb9_mP = stablehlo.reshape %vitb9_mAT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb9_mod = stablehlo.dot_general %vitb9_mP, %Wo_9, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb9_mobb = stablehlo.broadcast_in_dim %bo_9, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb9_mO = stablehlo.add %vitb9_mod, %vitb9_mobb : tensor<32x197x192xf32>
    %vitb9_r1 = stablehlo.add %vitb8_out, %vitb9_mO : tensor<32x197x192xf32>
    %vitb9_2sum = stablehlo.reduce(%vitb9_r1 init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb9_2nf = stablehlo.constant dense<192.0> : tensor<32x197xf32>
    %vitb9_2mu = stablehlo.divide %vitb9_2sum, %vitb9_2nf : tensor<32x197xf32>
    %vitb9_2mub = stablehlo.broadcast_in_dim %vitb9_2mu, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb9_2xc = stablehlo.subtract %vitb9_r1, %vitb9_2mub : tensor<32x197x192xf32>
    %vitb9_2sq = stablehlo.multiply %vitb9_2xc, %vitb9_2xc : tensor<32x197x192xf32>
    %vitb9_2vsum = stablehlo.reduce(%vitb9_2sq init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb9_2var = stablehlo.divide %vitb9_2vsum, %vitb9_2nf : tensor<32x197xf32>
    %vitb9_2eps = stablehlo.constant dense<1.0e-5> : tensor<32x197xf32>
    %vitb9_2ve = stablehlo.add %vitb9_2var, %vitb9_2eps : tensor<32x197xf32>
    %vitb9_2istd = stablehlo.rsqrt %vitb9_2ve : tensor<32x197xf32>
    %vitb9_2istdb = stablehlo.broadcast_in_dim %vitb9_2istd, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb9_2xhat = stablehlo.multiply %vitb9_2xc, %vitb9_2istdb : tensor<32x197x192xf32>
    %vitb9_2gb = stablehlo.broadcast_in_dim %g2_9, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb9_2bbc = stablehlo.broadcast_in_dim %b2_9, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb9_2gx = stablehlo.multiply %vitb9_2xhat, %vitb9_2gb : tensor<32x197x192xf32>
    %vitb9_2y = stablehlo.add %vitb9_2gx, %vitb9_2bbc : tensor<32x197x192xf32>
    %vitb9_ph1d = stablehlo.dot_general %vitb9_2y, %Wfc1_9, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x768xf32>) -> tensor<32x197x768xf32>
    %vitb9_ph1bb = stablehlo.broadcast_in_dim %bfc1_9, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %vitb9_ph1 = stablehlo.add %vitb9_ph1d, %vitb9_ph1bb : tensor<32x197x768xf32>
    %vitb9_pgx2 = stablehlo.multiply %vitb9_ph1, %vitb9_ph1 : tensor<32x197x768xf32>
    %vitb9_pgx3 = stablehlo.multiply %vitb9_pgx2, %vitb9_ph1 : tensor<32x197x768xf32>
    %vitb9_pgck = stablehlo.constant dense<0.044715> : tensor<32x197x768xf32>
    %vitb9_pgkx3 = stablehlo.multiply %vitb9_pgck, %vitb9_pgx3 : tensor<32x197x768xf32>
    %vitb9_pginn = stablehlo.add %vitb9_ph1, %vitb9_pgkx3 : tensor<32x197x768xf32>
    %vitb9_pgcsqrt = stablehlo.constant dense<0.7978845608028654> : tensor<32x197x768xf32>
    %vitb9_pgu = stablehlo.multiply %vitb9_pgcsqrt, %vitb9_pginn : tensor<32x197x768xf32>
    %vitb9_pgt = stablehlo.tanh %vitb9_pgu : tensor<32x197x768xf32>
    %vitb9_pgone = stablehlo.constant dense<1.0> : tensor<32x197x768xf32>
    %vitb9_pgopt = stablehlo.add %vitb9_pgone, %vitb9_pgt : tensor<32x197x768xf32>
    %vitb9_pgchalf = stablehlo.constant dense<0.5> : tensor<32x197x768xf32>
    %vitb9_pghx = stablehlo.multiply %vitb9_pgchalf, %vitb9_ph1 : tensor<32x197x768xf32>
    %vitb9_pga = stablehlo.multiply %vitb9_pghx, %vitb9_pgopt : tensor<32x197x768xf32>
    %vitb9_py2d = stablehlo.dot_general %vitb9_pga, %Wfc2_9, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x192xf32>) -> tensor<32x197x192xf32>
    %vitb9_py2bb = stablehlo.broadcast_in_dim %bfc2_9, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb9_py = stablehlo.add %vitb9_py2d, %vitb9_py2bb : tensor<32x197x192xf32>
    %vitb9_out = stablehlo.add %vitb9_r1, %vitb9_py : tensor<32x197x192xf32>
    %vitb10_1sum = stablehlo.reduce(%vitb9_out init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb10_1nf = stablehlo.constant dense<192.0> : tensor<32x197xf32>
    %vitb10_1mu = stablehlo.divide %vitb10_1sum, %vitb10_1nf : tensor<32x197xf32>
    %vitb10_1mub = stablehlo.broadcast_in_dim %vitb10_1mu, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb10_1xc = stablehlo.subtract %vitb9_out, %vitb10_1mub : tensor<32x197x192xf32>
    %vitb10_1sq = stablehlo.multiply %vitb10_1xc, %vitb10_1xc : tensor<32x197x192xf32>
    %vitb10_1vsum = stablehlo.reduce(%vitb10_1sq init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb10_1var = stablehlo.divide %vitb10_1vsum, %vitb10_1nf : tensor<32x197xf32>
    %vitb10_1eps = stablehlo.constant dense<1.0e-5> : tensor<32x197xf32>
    %vitb10_1ve = stablehlo.add %vitb10_1var, %vitb10_1eps : tensor<32x197xf32>
    %vitb10_1istd = stablehlo.rsqrt %vitb10_1ve : tensor<32x197xf32>
    %vitb10_1istdb = stablehlo.broadcast_in_dim %vitb10_1istd, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb10_1xhat = stablehlo.multiply %vitb10_1xc, %vitb10_1istdb : tensor<32x197x192xf32>
    %vitb10_1gb = stablehlo.broadcast_in_dim %g1_10, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb10_1bbc = stablehlo.broadcast_in_dim %b1_10, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb10_1gx = stablehlo.multiply %vitb10_1xhat, %vitb10_1gb : tensor<32x197x192xf32>
    %vitb10_1y = stablehlo.add %vitb10_1gx, %vitb10_1bbc : tensor<32x197x192xf32>
    %vitb10_mQd = stablehlo.dot_general %vitb10_1y, %Wq_10, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb10_mQbb = stablehlo.broadcast_in_dim %bq_10, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb10_mQ = stablehlo.add %vitb10_mQd, %vitb10_mQbb : tensor<32x197x192xf32>
    %vitb10_mKd = stablehlo.dot_general %vitb10_1y, %Wk_10, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb10_mKbb = stablehlo.broadcast_in_dim %bk_10, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb10_mK = stablehlo.add %vitb10_mKd, %vitb10_mKbb : tensor<32x197x192xf32>
    %vitb10_mVd = stablehlo.dot_general %vitb10_1y, %Wv_10, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb10_mVbb = stablehlo.broadcast_in_dim %bv_10, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb10_mV = stablehlo.add %vitb10_mVd, %vitb10_mVbb : tensor<32x197x192xf32>
    %vitb10_mQhr = stablehlo.reshape %vitb10_mQ : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb10_mQh = stablehlo.transpose %vitb10_mQhr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb10_mKhr = stablehlo.reshape %vitb10_mK : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb10_mKh = stablehlo.transpose %vitb10_mKhr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb10_mVhr = stablehlo.reshape %vitb10_mV : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb10_mVh = stablehlo.transpose %vitb10_mVhr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb10_mS = stablehlo.dot_general %vitb10_mQh, %vitb10_mKh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x64xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x197xf32>
    %vitb10_mscl = stablehlo.constant dense<0.125> : tensor<32x3x197x197xf32>
    %vitb10_mSs = stablehlo.multiply %vitb10_mS, %vitb10_mscl : tensor<32x3x197x197xf32>
    %vitb10_mse = stablehlo.exponential %vitb10_mSs : tensor<32x3x197x197xf32>
    %vitb10_msum = stablehlo.reduce(%vitb10_mse init: %sc) applies stablehlo.add across dimensions = [3] : (tensor<32x3x197x197xf32>, tensor<f32>) -> tensor<32x3x197xf32>
    %vitb10_msumb = stablehlo.broadcast_in_dim %vitb10_msum, dims = [0, 1, 2] : (tensor<32x3x197xf32>) -> tensor<32x3x197x197xf32>
    %vitb10_mW = stablehlo.divide %vitb10_mse, %vitb10_msumb : tensor<32x3x197x197xf32>
    %vitb10_mA = stablehlo.dot_general %vitb10_mW, %vitb10_mVh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb10_mAT = stablehlo.transpose %vitb10_mA, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb10_mP = stablehlo.reshape %vitb10_mAT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb10_mod = stablehlo.dot_general %vitb10_mP, %Wo_10, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb10_mobb = stablehlo.broadcast_in_dim %bo_10, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb10_mO = stablehlo.add %vitb10_mod, %vitb10_mobb : tensor<32x197x192xf32>
    %vitb10_r1 = stablehlo.add %vitb9_out, %vitb10_mO : tensor<32x197x192xf32>
    %vitb10_2sum = stablehlo.reduce(%vitb10_r1 init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb10_2nf = stablehlo.constant dense<192.0> : tensor<32x197xf32>
    %vitb10_2mu = stablehlo.divide %vitb10_2sum, %vitb10_2nf : tensor<32x197xf32>
    %vitb10_2mub = stablehlo.broadcast_in_dim %vitb10_2mu, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb10_2xc = stablehlo.subtract %vitb10_r1, %vitb10_2mub : tensor<32x197x192xf32>
    %vitb10_2sq = stablehlo.multiply %vitb10_2xc, %vitb10_2xc : tensor<32x197x192xf32>
    %vitb10_2vsum = stablehlo.reduce(%vitb10_2sq init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb10_2var = stablehlo.divide %vitb10_2vsum, %vitb10_2nf : tensor<32x197xf32>
    %vitb10_2eps = stablehlo.constant dense<1.0e-5> : tensor<32x197xf32>
    %vitb10_2ve = stablehlo.add %vitb10_2var, %vitb10_2eps : tensor<32x197xf32>
    %vitb10_2istd = stablehlo.rsqrt %vitb10_2ve : tensor<32x197xf32>
    %vitb10_2istdb = stablehlo.broadcast_in_dim %vitb10_2istd, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb10_2xhat = stablehlo.multiply %vitb10_2xc, %vitb10_2istdb : tensor<32x197x192xf32>
    %vitb10_2gb = stablehlo.broadcast_in_dim %g2_10, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb10_2bbc = stablehlo.broadcast_in_dim %b2_10, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb10_2gx = stablehlo.multiply %vitb10_2xhat, %vitb10_2gb : tensor<32x197x192xf32>
    %vitb10_2y = stablehlo.add %vitb10_2gx, %vitb10_2bbc : tensor<32x197x192xf32>
    %vitb10_ph1d = stablehlo.dot_general %vitb10_2y, %Wfc1_10, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x768xf32>) -> tensor<32x197x768xf32>
    %vitb10_ph1bb = stablehlo.broadcast_in_dim %bfc1_10, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %vitb10_ph1 = stablehlo.add %vitb10_ph1d, %vitb10_ph1bb : tensor<32x197x768xf32>
    %vitb10_pgx2 = stablehlo.multiply %vitb10_ph1, %vitb10_ph1 : tensor<32x197x768xf32>
    %vitb10_pgx3 = stablehlo.multiply %vitb10_pgx2, %vitb10_ph1 : tensor<32x197x768xf32>
    %vitb10_pgck = stablehlo.constant dense<0.044715> : tensor<32x197x768xf32>
    %vitb10_pgkx3 = stablehlo.multiply %vitb10_pgck, %vitb10_pgx3 : tensor<32x197x768xf32>
    %vitb10_pginn = stablehlo.add %vitb10_ph1, %vitb10_pgkx3 : tensor<32x197x768xf32>
    %vitb10_pgcsqrt = stablehlo.constant dense<0.7978845608028654> : tensor<32x197x768xf32>
    %vitb10_pgu = stablehlo.multiply %vitb10_pgcsqrt, %vitb10_pginn : tensor<32x197x768xf32>
    %vitb10_pgt = stablehlo.tanh %vitb10_pgu : tensor<32x197x768xf32>
    %vitb10_pgone = stablehlo.constant dense<1.0> : tensor<32x197x768xf32>
    %vitb10_pgopt = stablehlo.add %vitb10_pgone, %vitb10_pgt : tensor<32x197x768xf32>
    %vitb10_pgchalf = stablehlo.constant dense<0.5> : tensor<32x197x768xf32>
    %vitb10_pghx = stablehlo.multiply %vitb10_pgchalf, %vitb10_ph1 : tensor<32x197x768xf32>
    %vitb10_pga = stablehlo.multiply %vitb10_pghx, %vitb10_pgopt : tensor<32x197x768xf32>
    %vitb10_py2d = stablehlo.dot_general %vitb10_pga, %Wfc2_10, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x192xf32>) -> tensor<32x197x192xf32>
    %vitb10_py2bb = stablehlo.broadcast_in_dim %bfc2_10, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb10_py = stablehlo.add %vitb10_py2d, %vitb10_py2bb : tensor<32x197x192xf32>
    %vitb10_out = stablehlo.add %vitb10_r1, %vitb10_py : tensor<32x197x192xf32>
    %vitb11_1sum = stablehlo.reduce(%vitb10_out init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb11_1nf = stablehlo.constant dense<192.0> : tensor<32x197xf32>
    %vitb11_1mu = stablehlo.divide %vitb11_1sum, %vitb11_1nf : tensor<32x197xf32>
    %vitb11_1mub = stablehlo.broadcast_in_dim %vitb11_1mu, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb11_1xc = stablehlo.subtract %vitb10_out, %vitb11_1mub : tensor<32x197x192xf32>
    %vitb11_1sq = stablehlo.multiply %vitb11_1xc, %vitb11_1xc : tensor<32x197x192xf32>
    %vitb11_1vsum = stablehlo.reduce(%vitb11_1sq init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb11_1var = stablehlo.divide %vitb11_1vsum, %vitb11_1nf : tensor<32x197xf32>
    %vitb11_1eps = stablehlo.constant dense<1.0e-5> : tensor<32x197xf32>
    %vitb11_1ve = stablehlo.add %vitb11_1var, %vitb11_1eps : tensor<32x197xf32>
    %vitb11_1istd = stablehlo.rsqrt %vitb11_1ve : tensor<32x197xf32>
    %vitb11_1istdb = stablehlo.broadcast_in_dim %vitb11_1istd, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb11_1xhat = stablehlo.multiply %vitb11_1xc, %vitb11_1istdb : tensor<32x197x192xf32>
    %vitb11_1gb = stablehlo.broadcast_in_dim %g1_11, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb11_1bbc = stablehlo.broadcast_in_dim %b1_11, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb11_1gx = stablehlo.multiply %vitb11_1xhat, %vitb11_1gb : tensor<32x197x192xf32>
    %vitb11_1y = stablehlo.add %vitb11_1gx, %vitb11_1bbc : tensor<32x197x192xf32>
    %vitb11_mQd = stablehlo.dot_general %vitb11_1y, %Wq_11, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb11_mQbb = stablehlo.broadcast_in_dim %bq_11, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb11_mQ = stablehlo.add %vitb11_mQd, %vitb11_mQbb : tensor<32x197x192xf32>
    %vitb11_mKd = stablehlo.dot_general %vitb11_1y, %Wk_11, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb11_mKbb = stablehlo.broadcast_in_dim %bk_11, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb11_mK = stablehlo.add %vitb11_mKd, %vitb11_mKbb : tensor<32x197x192xf32>
    %vitb11_mVd = stablehlo.dot_general %vitb11_1y, %Wv_11, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb11_mVbb = stablehlo.broadcast_in_dim %bv_11, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb11_mV = stablehlo.add %vitb11_mVd, %vitb11_mVbb : tensor<32x197x192xf32>
    %vitb11_mQhr = stablehlo.reshape %vitb11_mQ : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb11_mQh = stablehlo.transpose %vitb11_mQhr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb11_mKhr = stablehlo.reshape %vitb11_mK : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb11_mKh = stablehlo.transpose %vitb11_mKhr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb11_mVhr = stablehlo.reshape %vitb11_mV : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb11_mVh = stablehlo.transpose %vitb11_mVhr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb11_mS = stablehlo.dot_general %vitb11_mQh, %vitb11_mKh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x64xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x197xf32>
    %vitb11_mscl = stablehlo.constant dense<0.125> : tensor<32x3x197x197xf32>
    %vitb11_mSs = stablehlo.multiply %vitb11_mS, %vitb11_mscl : tensor<32x3x197x197xf32>
    %vitb11_mse = stablehlo.exponential %vitb11_mSs : tensor<32x3x197x197xf32>
    %vitb11_msum = stablehlo.reduce(%vitb11_mse init: %sc) applies stablehlo.add across dimensions = [3] : (tensor<32x3x197x197xf32>, tensor<f32>) -> tensor<32x3x197xf32>
    %vitb11_msumb = stablehlo.broadcast_in_dim %vitb11_msum, dims = [0, 1, 2] : (tensor<32x3x197xf32>) -> tensor<32x3x197x197xf32>
    %vitb11_mW = stablehlo.divide %vitb11_mse, %vitb11_msumb : tensor<32x3x197x197xf32>
    %vitb11_mA = stablehlo.dot_general %vitb11_mW, %vitb11_mVh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb11_mAT = stablehlo.transpose %vitb11_mA, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb11_mP = stablehlo.reshape %vitb11_mAT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb11_mod = stablehlo.dot_general %vitb11_mP, %Wo_11, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb11_mobb = stablehlo.broadcast_in_dim %bo_11, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb11_mO = stablehlo.add %vitb11_mod, %vitb11_mobb : tensor<32x197x192xf32>
    %vitb11_r1 = stablehlo.add %vitb10_out, %vitb11_mO : tensor<32x197x192xf32>
    %vitb11_2sum = stablehlo.reduce(%vitb11_r1 init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb11_2nf = stablehlo.constant dense<192.0> : tensor<32x197xf32>
    %vitb11_2mu = stablehlo.divide %vitb11_2sum, %vitb11_2nf : tensor<32x197xf32>
    %vitb11_2mub = stablehlo.broadcast_in_dim %vitb11_2mu, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb11_2xc = stablehlo.subtract %vitb11_r1, %vitb11_2mub : tensor<32x197x192xf32>
    %vitb11_2sq = stablehlo.multiply %vitb11_2xc, %vitb11_2xc : tensor<32x197x192xf32>
    %vitb11_2vsum = stablehlo.reduce(%vitb11_2sq init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb11_2var = stablehlo.divide %vitb11_2vsum, %vitb11_2nf : tensor<32x197xf32>
    %vitb11_2eps = stablehlo.constant dense<1.0e-5> : tensor<32x197xf32>
    %vitb11_2ve = stablehlo.add %vitb11_2var, %vitb11_2eps : tensor<32x197xf32>
    %vitb11_2istd = stablehlo.rsqrt %vitb11_2ve : tensor<32x197xf32>
    %vitb11_2istdb = stablehlo.broadcast_in_dim %vitb11_2istd, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb11_2xhat = stablehlo.multiply %vitb11_2xc, %vitb11_2istdb : tensor<32x197x192xf32>
    %vitb11_2gb = stablehlo.broadcast_in_dim %g2_11, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb11_2bbc = stablehlo.broadcast_in_dim %b2_11, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb11_2gx = stablehlo.multiply %vitb11_2xhat, %vitb11_2gb : tensor<32x197x192xf32>
    %vitb11_2y = stablehlo.add %vitb11_2gx, %vitb11_2bbc : tensor<32x197x192xf32>
    %vitb11_ph1d = stablehlo.dot_general %vitb11_2y, %Wfc1_11, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x768xf32>) -> tensor<32x197x768xf32>
    %vitb11_ph1bb = stablehlo.broadcast_in_dim %bfc1_11, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %vitb11_ph1 = stablehlo.add %vitb11_ph1d, %vitb11_ph1bb : tensor<32x197x768xf32>
    %vitb11_pgx2 = stablehlo.multiply %vitb11_ph1, %vitb11_ph1 : tensor<32x197x768xf32>
    %vitb11_pgx3 = stablehlo.multiply %vitb11_pgx2, %vitb11_ph1 : tensor<32x197x768xf32>
    %vitb11_pgck = stablehlo.constant dense<0.044715> : tensor<32x197x768xf32>
    %vitb11_pgkx3 = stablehlo.multiply %vitb11_pgck, %vitb11_pgx3 : tensor<32x197x768xf32>
    %vitb11_pginn = stablehlo.add %vitb11_ph1, %vitb11_pgkx3 : tensor<32x197x768xf32>
    %vitb11_pgcsqrt = stablehlo.constant dense<0.7978845608028654> : tensor<32x197x768xf32>
    %vitb11_pgu = stablehlo.multiply %vitb11_pgcsqrt, %vitb11_pginn : tensor<32x197x768xf32>
    %vitb11_pgt = stablehlo.tanh %vitb11_pgu : tensor<32x197x768xf32>
    %vitb11_pgone = stablehlo.constant dense<1.0> : tensor<32x197x768xf32>
    %vitb11_pgopt = stablehlo.add %vitb11_pgone, %vitb11_pgt : tensor<32x197x768xf32>
    %vitb11_pgchalf = stablehlo.constant dense<0.5> : tensor<32x197x768xf32>
    %vitb11_pghx = stablehlo.multiply %vitb11_pgchalf, %vitb11_ph1 : tensor<32x197x768xf32>
    %vitb11_pga = stablehlo.multiply %vitb11_pghx, %vitb11_pgopt : tensor<32x197x768xf32>
    %vitb11_py2d = stablehlo.dot_general %vitb11_pga, %Wfc2_11, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x192xf32>) -> tensor<32x197x192xf32>
    %vitb11_py2bb = stablehlo.broadcast_in_dim %bfc2_11, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb11_py = stablehlo.add %vitb11_py2d, %vitb11_py2bb : tensor<32x197x192xf32>
    %vitb11_out = stablehlo.add %vitb11_r1, %vitb11_py : tensor<32x197x192xf32>
    %vitflnsum = stablehlo.reduce(%vitb11_out init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitflnnf = stablehlo.constant dense<192.0> : tensor<32x197xf32>
    %vitflnmu = stablehlo.divide %vitflnsum, %vitflnnf : tensor<32x197xf32>
    %vitflnmub = stablehlo.broadcast_in_dim %vitflnmu, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitflnxc = stablehlo.subtract %vitb11_out, %vitflnmub : tensor<32x197x192xf32>
    %vitflnsq = stablehlo.multiply %vitflnxc, %vitflnxc : tensor<32x197x192xf32>
    %vitflnvsum = stablehlo.reduce(%vitflnsq init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitflnvar = stablehlo.divide %vitflnvsum, %vitflnnf : tensor<32x197xf32>
    %vitflneps = stablehlo.constant dense<1.0e-5> : tensor<32x197xf32>
    %vitflnve = stablehlo.add %vitflnvar, %vitflneps : tensor<32x197xf32>
    %vitflnistd = stablehlo.rsqrt %vitflnve : tensor<32x197xf32>
    %vitflnistdb = stablehlo.broadcast_in_dim %vitflnistd, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitflnxhat = stablehlo.multiply %vitflnxc, %vitflnistdb : tensor<32x197x192xf32>
    %vitflngb = stablehlo.broadcast_in_dim %gF, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitflnbbc = stablehlo.broadcast_in_dim %bF, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitflngx = stablehlo.multiply %vitflnxhat, %vitflngb : tensor<32x197x192xf32>
    %vitflny = stablehlo.add %vitflngx, %vitflnbbc : tensor<32x197x192xf32>
    %vithdcls = stablehlo.slice %vitflny [0:32, 0:1, 0:192] : (tensor<32x197x192xf32>) -> tensor<32x1x192xf32>
    %vithdclsv = stablehlo.reshape %vithdcls : (tensor<32x1x192xf32>) -> tensor<32x192xf32>
    %vithdhd = stablehlo.dot_general %vithdclsv, %Wc, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x192xf32>, tensor<192x10xf32>) -> tensor<32x10xf32>
    %vithdhbb = stablehlo.broadcast_in_dim %bc, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %vithdlogits = stablehlo.add %vithdhd, %vithdhbb : tensor<32x10xf32>
    %le = stablehlo.exponential %vithdlogits : tensor<32x10xf32>
    %lsum = stablehlo.reduce(%le init: %sc) applies stablehlo.add across dimensions = [1] : (tensor<32x10xf32>, tensor<f32>) -> tensor<32xf32>
    %lsb = stablehlo.broadcast_in_dim %lsum, dims = [0] : (tensor<32xf32>) -> tensor<32x10xf32>
    %lsm = stablehlo.divide %le, %lsb : tensor<32x10xf32>
    %dyr = stablehlo.subtract %lsm, %onehot : tensor<32x10xf32>
    %bnc = stablehlo.constant dense<32.0> : tensor<32x10xf32>
    %dy = stablehlo.divide %dyr, %bnc : tensor<32x10xf32>
    %vithddWc = stablehlo.dot_general %vithdclsv, %dy, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x192xf32>, tensor<32x10xf32>) -> tensor<192x10xf32>
    %vithddbc = stablehlo.reduce(%dy init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<32x10xf32>, tensor<f32>) -> tensor<10xf32>
    %vithddclsv = stablehlo.dot_general %dy, %Wc, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<192x10xf32>) -> tensor<32x192xf32>
    %vithddclsr = stablehlo.reshape %vithddclsv : (tensor<32x192xf32>) -> tensor<32x1x192xf32>
    %vithddz = stablehlo.pad %vithddclsr, %sc, low = [0, 0, 0], high = [0, 196, 0], interior = [0, 0, 0] : (tensor<32x1x192xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %vitflngbk = stablehlo.broadcast_in_dim %gF, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitflndxhat = stablehlo.multiply %vithddz, %vitflngbk : tensor<32x197x192xf32>
    %vitflndgpre = stablehlo.multiply %vithddz, %vitflnxhat : tensor<32x197x192xf32>
    %vitflndg = stablehlo.reduce(%vitflndgpre init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitflndb = stablehlo.reduce(%vithddz init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitflnm1s = stablehlo.reduce(%vitflndxhat init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitflnm1 = stablehlo.divide %vitflnm1s, %vitflnnf : tensor<32x197xf32>
    %vitflndxxh = stablehlo.multiply %vitflndxhat, %vitflnxhat : tensor<32x197x192xf32>
    %vitflnm2s = stablehlo.reduce(%vitflndxxh init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitflnm2 = stablehlo.divide %vitflnm2s, %vitflnnf : tensor<32x197xf32>
    %vitflnm1b = stablehlo.broadcast_in_dim %vitflnm1, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitflnm2b = stablehlo.broadcast_in_dim %vitflnm2, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitflnt1 = stablehlo.subtract %vitflndxhat, %vitflnm1b : tensor<32x197x192xf32>
    %vitflnxm2 = stablehlo.multiply %vitflnxhat, %vitflnm2b : tensor<32x197x192xf32>
    %vitflnt2 = stablehlo.subtract %vitflnt1, %vitflnxm2 : tensor<32x197x192xf32>
    %vitflndx = stablehlo.multiply %vitflnistdb, %vitflnt2 : tensor<32x197x192xf32>
    %vitb11_pda1 = stablehlo.dot_general %vitflndx, %Wfc2_11, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<768x192xf32>) -> tensor<32x197x768xf32>
    %vitb11_pdWfc2 = stablehlo.dot_general %vitb11_pga, %vitflndx, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<32x197x192xf32>) -> tensor<768x192xf32>
    %vitb11_pdbfc2 = stablehlo.reduce(%vitflndx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb11_pgbbx2 = stablehlo.multiply %vitb11_ph1, %vitb11_ph1 : tensor<32x197x768xf32>
    %vitb11_pgbbx3 = stablehlo.multiply %vitb11_pgbbx2, %vitb11_ph1 : tensor<32x197x768xf32>
    %vitb11_pgbbck = stablehlo.constant dense<0.044715> : tensor<32x197x768xf32>
    %vitb11_pgbbkx3 = stablehlo.multiply %vitb11_pgbbck, %vitb11_pgbbx3 : tensor<32x197x768xf32>
    %vitb11_pgbbinn = stablehlo.add %vitb11_ph1, %vitb11_pgbbkx3 : tensor<32x197x768xf32>
    %vitb11_pgbbcsqrt = stablehlo.constant dense<0.7978845608028654> : tensor<32x197x768xf32>
    %vitb11_pgbbu = stablehlo.multiply %vitb11_pgbbcsqrt, %vitb11_pgbbinn : tensor<32x197x768xf32>
    %vitb11_pgbbt = stablehlo.tanh %vitb11_pgbbu : tensor<32x197x768xf32>
    %vitb11_pgbbone = stablehlo.constant dense<1.0> : tensor<32x197x768xf32>
    %vitb11_pgbbopt = stablehlo.add %vitb11_pgbbone, %vitb11_pgbbt : tensor<32x197x768xf32>
    %vitb11_pgbbchalf = stablehlo.constant dense<0.5> : tensor<32x197x768xf32>
    %vitb11_pgbbterm1 = stablehlo.multiply %vitb11_pgbbchalf, %vitb11_pgbbopt : tensor<32x197x768xf32>
    %vitb11_pgbbt2 = stablehlo.multiply %vitb11_pgbbt, %vitb11_pgbbt : tensor<32x197x768xf32>
    %vitb11_pgbbomt2 = stablehlo.subtract %vitb11_pgbbone, %vitb11_pgbbt2 : tensor<32x197x768xf32>
    %vitb11_pgbbhx = stablehlo.multiply %vitb11_pgbbchalf, %vitb11_ph1 : tensor<32x197x768xf32>
    %vitb11_pgbbhxo = stablehlo.multiply %vitb11_pgbbhx, %vitb11_pgbbomt2 : tensor<32x197x768xf32>
    %vitb11_pgbbc3b = stablehlo.constant dense<0.134145> : tensor<32x197x768xf32>
    %vitb11_pgbba3x2 = stablehlo.multiply %vitb11_pgbbc3b, %vitb11_pgbbx2 : tensor<32x197x768xf32>
    %vitb11_pgbbin2 = stablehlo.add %vitb11_pgbbone, %vitb11_pgbba3x2 : tensor<32x197x768xf32>
    %vitb11_pgbbup = stablehlo.multiply %vitb11_pgbbcsqrt, %vitb11_pgbbin2 : tensor<32x197x768xf32>
    %vitb11_pgbbterm2 = stablehlo.multiply %vitb11_pgbbhxo, %vitb11_pgbbup : tensor<32x197x768xf32>
    %vitb11_pgbbgp = stablehlo.add %vitb11_pgbbterm1, %vitb11_pgbbterm2 : tensor<32x197x768xf32>
    %vitb11_pgbdx = stablehlo.multiply %vitb11_pda1, %vitb11_pgbbgp : tensor<32x197x768xf32>
    %vitb11_pdx = stablehlo.dot_general %vitb11_pgbdx, %Wfc1_11, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<192x768xf32>) -> tensor<32x197x192xf32>
    %vitb11_pdWfc1 = stablehlo.dot_general %vitb11_2y, %vitb11_pgbdx, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x768xf32>) -> tensor<192x768xf32>
    %vitb11_pdbfc1 = stablehlo.reduce(%vitb11_pgbdx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<768xf32>
    %vitb11_2gbk = stablehlo.broadcast_in_dim %g2_11, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb11_2dxhat = stablehlo.multiply %vitb11_pdx, %vitb11_2gbk : tensor<32x197x192xf32>
    %vitb11_2dgpre = stablehlo.multiply %vitb11_pdx, %vitb11_2xhat : tensor<32x197x192xf32>
    %vitb11_2dg = stablehlo.reduce(%vitb11_2dgpre init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb11_2db = stablehlo.reduce(%vitb11_pdx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb11_2m1s = stablehlo.reduce(%vitb11_2dxhat init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb11_2m1 = stablehlo.divide %vitb11_2m1s, %vitb11_2nf : tensor<32x197xf32>
    %vitb11_2dxxh = stablehlo.multiply %vitb11_2dxhat, %vitb11_2xhat : tensor<32x197x192xf32>
    %vitb11_2m2s = stablehlo.reduce(%vitb11_2dxxh init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb11_2m2 = stablehlo.divide %vitb11_2m2s, %vitb11_2nf : tensor<32x197xf32>
    %vitb11_2m1b = stablehlo.broadcast_in_dim %vitb11_2m1, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb11_2m2b = stablehlo.broadcast_in_dim %vitb11_2m2, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb11_2t1 = stablehlo.subtract %vitb11_2dxhat, %vitb11_2m1b : tensor<32x197x192xf32>
    %vitb11_2xm2 = stablehlo.multiply %vitb11_2xhat, %vitb11_2m2b : tensor<32x197x192xf32>
    %vitb11_2t2 = stablehlo.subtract %vitb11_2t1, %vitb11_2xm2 : tensor<32x197x192xf32>
    %vitb11_2dx = stablehlo.multiply %vitb11_2istdb, %vitb11_2t2 : tensor<32x197x192xf32>
    %vitb11_dr1 = stablehlo.add %vitflndx, %vitb11_2dx : tensor<32x197x192xf32>
    %vitb11_mdP = stablehlo.dot_general %vitb11_dr1, %Wo_11, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb11_mdWo = stablehlo.dot_general %vitb11_mP, %vitb11_dr1, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb11_mdbo = stablehlo.reduce(%vitb11_dr1 init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb11_mdPr = stablehlo.reshape %vitb11_mdP : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb11_mdA = stablehlo.transpose %vitb11_mdPr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb11_mdW = stablehlo.dot_general %vitb11_mdA, %vitb11_mVh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x64xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x197xf32>
    %vitb11_mdVh = stablehlo.dot_general %vitb11_mW, %vitb11_mdA, batching_dims = [0, 1] x [0, 1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb11_mpdw = stablehlo.multiply %vitb11_mW, %vitb11_mdW : tensor<32x3x197x197xf32>
    %vitb11_msrow = stablehlo.reduce(%vitb11_mpdw init: %sc) applies stablehlo.add across dimensions = [3] : (tensor<32x3x197x197xf32>, tensor<f32>) -> tensor<32x3x197xf32>
    %vitb11_msrowb = stablehlo.broadcast_in_dim %vitb11_msrow, dims = [0, 1, 2] : (tensor<32x3x197xf32>) -> tensor<32x3x197x197xf32>
    %vitb11_mdiff = stablehlo.subtract %vitb11_mdW, %vitb11_msrowb : tensor<32x3x197x197xf32>
    %vitb11_mdSs = stablehlo.multiply %vitb11_mW, %vitb11_mdiff : tensor<32x3x197x197xf32>
    %vitb11_msclb = stablehlo.constant dense<0.125> : tensor<32x3x197x197xf32>
    %vitb11_mdS = stablehlo.multiply %vitb11_mdSs, %vitb11_msclb : tensor<32x3x197x197xf32>
    %vitb11_mdQh = stablehlo.dot_general %vitb11_mdS, %vitb11_mKh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb11_mdKh = stablehlo.dot_general %vitb11_mdS, %vitb11_mQh, batching_dims = [0, 1] x [0, 1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb11_mdQT = stablehlo.transpose %vitb11_mdQh, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb11_mdQ = stablehlo.reshape %vitb11_mdQT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb11_mdKT = stablehlo.transpose %vitb11_mdKh, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb11_mdK = stablehlo.reshape %vitb11_mdKT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb11_mdVT = stablehlo.transpose %vitb11_mdVh, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb11_mdV = stablehlo.reshape %vitb11_mdVT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb11_mdxQ = stablehlo.dot_general %vitb11_mdQ, %Wq_11, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb11_mdWQ = stablehlo.dot_general %vitb11_1y, %vitb11_mdQ, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb11_mdbQ = stablehlo.reduce(%vitb11_mdQ init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb11_mdxK = stablehlo.dot_general %vitb11_mdK, %Wk_11, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb11_mdWK = stablehlo.dot_general %vitb11_1y, %vitb11_mdK, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb11_mdbK = stablehlo.reduce(%vitb11_mdK init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb11_mdxV = stablehlo.dot_general %vitb11_mdV, %Wv_11, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb11_mdWV = stablehlo.dot_general %vitb11_1y, %vitb11_mdV, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb11_mdbV = stablehlo.reduce(%vitb11_mdV init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb11_mdxa = stablehlo.add %vitb11_mdxQ, %vitb11_mdxK : tensor<32x197x192xf32>
    %vitb11_mdx = stablehlo.add %vitb11_mdxa, %vitb11_mdxV : tensor<32x197x192xf32>
    %vitb11_1gbk = stablehlo.broadcast_in_dim %g1_11, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb11_1dxhat = stablehlo.multiply %vitb11_mdx, %vitb11_1gbk : tensor<32x197x192xf32>
    %vitb11_1dgpre = stablehlo.multiply %vitb11_mdx, %vitb11_1xhat : tensor<32x197x192xf32>
    %vitb11_1dg = stablehlo.reduce(%vitb11_1dgpre init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb11_1db = stablehlo.reduce(%vitb11_mdx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb11_1m1s = stablehlo.reduce(%vitb11_1dxhat init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb11_1m1 = stablehlo.divide %vitb11_1m1s, %vitb11_1nf : tensor<32x197xf32>
    %vitb11_1dxxh = stablehlo.multiply %vitb11_1dxhat, %vitb11_1xhat : tensor<32x197x192xf32>
    %vitb11_1m2s = stablehlo.reduce(%vitb11_1dxxh init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb11_1m2 = stablehlo.divide %vitb11_1m2s, %vitb11_1nf : tensor<32x197xf32>
    %vitb11_1m1b = stablehlo.broadcast_in_dim %vitb11_1m1, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb11_1m2b = stablehlo.broadcast_in_dim %vitb11_1m2, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb11_1t1 = stablehlo.subtract %vitb11_1dxhat, %vitb11_1m1b : tensor<32x197x192xf32>
    %vitb11_1xm2 = stablehlo.multiply %vitb11_1xhat, %vitb11_1m2b : tensor<32x197x192xf32>
    %vitb11_1t2 = stablehlo.subtract %vitb11_1t1, %vitb11_1xm2 : tensor<32x197x192xf32>
    %vitb11_1dx = stablehlo.multiply %vitb11_1istdb, %vitb11_1t2 : tensor<32x197x192xf32>
    %vitb11_dx = stablehlo.add %vitb11_dr1, %vitb11_1dx : tensor<32x197x192xf32>
    %vitb10_pda1 = stablehlo.dot_general %vitb11_dx, %Wfc2_10, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<768x192xf32>) -> tensor<32x197x768xf32>
    %vitb10_pdWfc2 = stablehlo.dot_general %vitb10_pga, %vitb11_dx, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<32x197x192xf32>) -> tensor<768x192xf32>
    %vitb10_pdbfc2 = stablehlo.reduce(%vitb11_dx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb10_pgbbx2 = stablehlo.multiply %vitb10_ph1, %vitb10_ph1 : tensor<32x197x768xf32>
    %vitb10_pgbbx3 = stablehlo.multiply %vitb10_pgbbx2, %vitb10_ph1 : tensor<32x197x768xf32>
    %vitb10_pgbbck = stablehlo.constant dense<0.044715> : tensor<32x197x768xf32>
    %vitb10_pgbbkx3 = stablehlo.multiply %vitb10_pgbbck, %vitb10_pgbbx3 : tensor<32x197x768xf32>
    %vitb10_pgbbinn = stablehlo.add %vitb10_ph1, %vitb10_pgbbkx3 : tensor<32x197x768xf32>
    %vitb10_pgbbcsqrt = stablehlo.constant dense<0.7978845608028654> : tensor<32x197x768xf32>
    %vitb10_pgbbu = stablehlo.multiply %vitb10_pgbbcsqrt, %vitb10_pgbbinn : tensor<32x197x768xf32>
    %vitb10_pgbbt = stablehlo.tanh %vitb10_pgbbu : tensor<32x197x768xf32>
    %vitb10_pgbbone = stablehlo.constant dense<1.0> : tensor<32x197x768xf32>
    %vitb10_pgbbopt = stablehlo.add %vitb10_pgbbone, %vitb10_pgbbt : tensor<32x197x768xf32>
    %vitb10_pgbbchalf = stablehlo.constant dense<0.5> : tensor<32x197x768xf32>
    %vitb10_pgbbterm1 = stablehlo.multiply %vitb10_pgbbchalf, %vitb10_pgbbopt : tensor<32x197x768xf32>
    %vitb10_pgbbt2 = stablehlo.multiply %vitb10_pgbbt, %vitb10_pgbbt : tensor<32x197x768xf32>
    %vitb10_pgbbomt2 = stablehlo.subtract %vitb10_pgbbone, %vitb10_pgbbt2 : tensor<32x197x768xf32>
    %vitb10_pgbbhx = stablehlo.multiply %vitb10_pgbbchalf, %vitb10_ph1 : tensor<32x197x768xf32>
    %vitb10_pgbbhxo = stablehlo.multiply %vitb10_pgbbhx, %vitb10_pgbbomt2 : tensor<32x197x768xf32>
    %vitb10_pgbbc3b = stablehlo.constant dense<0.134145> : tensor<32x197x768xf32>
    %vitb10_pgbba3x2 = stablehlo.multiply %vitb10_pgbbc3b, %vitb10_pgbbx2 : tensor<32x197x768xf32>
    %vitb10_pgbbin2 = stablehlo.add %vitb10_pgbbone, %vitb10_pgbba3x2 : tensor<32x197x768xf32>
    %vitb10_pgbbup = stablehlo.multiply %vitb10_pgbbcsqrt, %vitb10_pgbbin2 : tensor<32x197x768xf32>
    %vitb10_pgbbterm2 = stablehlo.multiply %vitb10_pgbbhxo, %vitb10_pgbbup : tensor<32x197x768xf32>
    %vitb10_pgbbgp = stablehlo.add %vitb10_pgbbterm1, %vitb10_pgbbterm2 : tensor<32x197x768xf32>
    %vitb10_pgbdx = stablehlo.multiply %vitb10_pda1, %vitb10_pgbbgp : tensor<32x197x768xf32>
    %vitb10_pdx = stablehlo.dot_general %vitb10_pgbdx, %Wfc1_10, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<192x768xf32>) -> tensor<32x197x192xf32>
    %vitb10_pdWfc1 = stablehlo.dot_general %vitb10_2y, %vitb10_pgbdx, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x768xf32>) -> tensor<192x768xf32>
    %vitb10_pdbfc1 = stablehlo.reduce(%vitb10_pgbdx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<768xf32>
    %vitb10_2gbk = stablehlo.broadcast_in_dim %g2_10, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb10_2dxhat = stablehlo.multiply %vitb10_pdx, %vitb10_2gbk : tensor<32x197x192xf32>
    %vitb10_2dgpre = stablehlo.multiply %vitb10_pdx, %vitb10_2xhat : tensor<32x197x192xf32>
    %vitb10_2dg = stablehlo.reduce(%vitb10_2dgpre init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb10_2db = stablehlo.reduce(%vitb10_pdx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb10_2m1s = stablehlo.reduce(%vitb10_2dxhat init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb10_2m1 = stablehlo.divide %vitb10_2m1s, %vitb10_2nf : tensor<32x197xf32>
    %vitb10_2dxxh = stablehlo.multiply %vitb10_2dxhat, %vitb10_2xhat : tensor<32x197x192xf32>
    %vitb10_2m2s = stablehlo.reduce(%vitb10_2dxxh init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb10_2m2 = stablehlo.divide %vitb10_2m2s, %vitb10_2nf : tensor<32x197xf32>
    %vitb10_2m1b = stablehlo.broadcast_in_dim %vitb10_2m1, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb10_2m2b = stablehlo.broadcast_in_dim %vitb10_2m2, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb10_2t1 = stablehlo.subtract %vitb10_2dxhat, %vitb10_2m1b : tensor<32x197x192xf32>
    %vitb10_2xm2 = stablehlo.multiply %vitb10_2xhat, %vitb10_2m2b : tensor<32x197x192xf32>
    %vitb10_2t2 = stablehlo.subtract %vitb10_2t1, %vitb10_2xm2 : tensor<32x197x192xf32>
    %vitb10_2dx = stablehlo.multiply %vitb10_2istdb, %vitb10_2t2 : tensor<32x197x192xf32>
    %vitb10_dr1 = stablehlo.add %vitb11_dx, %vitb10_2dx : tensor<32x197x192xf32>
    %vitb10_mdP = stablehlo.dot_general %vitb10_dr1, %Wo_10, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb10_mdWo = stablehlo.dot_general %vitb10_mP, %vitb10_dr1, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb10_mdbo = stablehlo.reduce(%vitb10_dr1 init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb10_mdPr = stablehlo.reshape %vitb10_mdP : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb10_mdA = stablehlo.transpose %vitb10_mdPr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb10_mdW = stablehlo.dot_general %vitb10_mdA, %vitb10_mVh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x64xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x197xf32>
    %vitb10_mdVh = stablehlo.dot_general %vitb10_mW, %vitb10_mdA, batching_dims = [0, 1] x [0, 1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb10_mpdw = stablehlo.multiply %vitb10_mW, %vitb10_mdW : tensor<32x3x197x197xf32>
    %vitb10_msrow = stablehlo.reduce(%vitb10_mpdw init: %sc) applies stablehlo.add across dimensions = [3] : (tensor<32x3x197x197xf32>, tensor<f32>) -> tensor<32x3x197xf32>
    %vitb10_msrowb = stablehlo.broadcast_in_dim %vitb10_msrow, dims = [0, 1, 2] : (tensor<32x3x197xf32>) -> tensor<32x3x197x197xf32>
    %vitb10_mdiff = stablehlo.subtract %vitb10_mdW, %vitb10_msrowb : tensor<32x3x197x197xf32>
    %vitb10_mdSs = stablehlo.multiply %vitb10_mW, %vitb10_mdiff : tensor<32x3x197x197xf32>
    %vitb10_msclb = stablehlo.constant dense<0.125> : tensor<32x3x197x197xf32>
    %vitb10_mdS = stablehlo.multiply %vitb10_mdSs, %vitb10_msclb : tensor<32x3x197x197xf32>
    %vitb10_mdQh = stablehlo.dot_general %vitb10_mdS, %vitb10_mKh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb10_mdKh = stablehlo.dot_general %vitb10_mdS, %vitb10_mQh, batching_dims = [0, 1] x [0, 1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb10_mdQT = stablehlo.transpose %vitb10_mdQh, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb10_mdQ = stablehlo.reshape %vitb10_mdQT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb10_mdKT = stablehlo.transpose %vitb10_mdKh, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb10_mdK = stablehlo.reshape %vitb10_mdKT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb10_mdVT = stablehlo.transpose %vitb10_mdVh, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb10_mdV = stablehlo.reshape %vitb10_mdVT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb10_mdxQ = stablehlo.dot_general %vitb10_mdQ, %Wq_10, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb10_mdWQ = stablehlo.dot_general %vitb10_1y, %vitb10_mdQ, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb10_mdbQ = stablehlo.reduce(%vitb10_mdQ init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb10_mdxK = stablehlo.dot_general %vitb10_mdK, %Wk_10, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb10_mdWK = stablehlo.dot_general %vitb10_1y, %vitb10_mdK, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb10_mdbK = stablehlo.reduce(%vitb10_mdK init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb10_mdxV = stablehlo.dot_general %vitb10_mdV, %Wv_10, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb10_mdWV = stablehlo.dot_general %vitb10_1y, %vitb10_mdV, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb10_mdbV = stablehlo.reduce(%vitb10_mdV init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb10_mdxa = stablehlo.add %vitb10_mdxQ, %vitb10_mdxK : tensor<32x197x192xf32>
    %vitb10_mdx = stablehlo.add %vitb10_mdxa, %vitb10_mdxV : tensor<32x197x192xf32>
    %vitb10_1gbk = stablehlo.broadcast_in_dim %g1_10, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb10_1dxhat = stablehlo.multiply %vitb10_mdx, %vitb10_1gbk : tensor<32x197x192xf32>
    %vitb10_1dgpre = stablehlo.multiply %vitb10_mdx, %vitb10_1xhat : tensor<32x197x192xf32>
    %vitb10_1dg = stablehlo.reduce(%vitb10_1dgpre init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb10_1db = stablehlo.reduce(%vitb10_mdx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb10_1m1s = stablehlo.reduce(%vitb10_1dxhat init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb10_1m1 = stablehlo.divide %vitb10_1m1s, %vitb10_1nf : tensor<32x197xf32>
    %vitb10_1dxxh = stablehlo.multiply %vitb10_1dxhat, %vitb10_1xhat : tensor<32x197x192xf32>
    %vitb10_1m2s = stablehlo.reduce(%vitb10_1dxxh init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb10_1m2 = stablehlo.divide %vitb10_1m2s, %vitb10_1nf : tensor<32x197xf32>
    %vitb10_1m1b = stablehlo.broadcast_in_dim %vitb10_1m1, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb10_1m2b = stablehlo.broadcast_in_dim %vitb10_1m2, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb10_1t1 = stablehlo.subtract %vitb10_1dxhat, %vitb10_1m1b : tensor<32x197x192xf32>
    %vitb10_1xm2 = stablehlo.multiply %vitb10_1xhat, %vitb10_1m2b : tensor<32x197x192xf32>
    %vitb10_1t2 = stablehlo.subtract %vitb10_1t1, %vitb10_1xm2 : tensor<32x197x192xf32>
    %vitb10_1dx = stablehlo.multiply %vitb10_1istdb, %vitb10_1t2 : tensor<32x197x192xf32>
    %vitb10_dx = stablehlo.add %vitb10_dr1, %vitb10_1dx : tensor<32x197x192xf32>
    %vitb9_pda1 = stablehlo.dot_general %vitb10_dx, %Wfc2_9, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<768x192xf32>) -> tensor<32x197x768xf32>
    %vitb9_pdWfc2 = stablehlo.dot_general %vitb9_pga, %vitb10_dx, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<32x197x192xf32>) -> tensor<768x192xf32>
    %vitb9_pdbfc2 = stablehlo.reduce(%vitb10_dx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb9_pgbbx2 = stablehlo.multiply %vitb9_ph1, %vitb9_ph1 : tensor<32x197x768xf32>
    %vitb9_pgbbx3 = stablehlo.multiply %vitb9_pgbbx2, %vitb9_ph1 : tensor<32x197x768xf32>
    %vitb9_pgbbck = stablehlo.constant dense<0.044715> : tensor<32x197x768xf32>
    %vitb9_pgbbkx3 = stablehlo.multiply %vitb9_pgbbck, %vitb9_pgbbx3 : tensor<32x197x768xf32>
    %vitb9_pgbbinn = stablehlo.add %vitb9_ph1, %vitb9_pgbbkx3 : tensor<32x197x768xf32>
    %vitb9_pgbbcsqrt = stablehlo.constant dense<0.7978845608028654> : tensor<32x197x768xf32>
    %vitb9_pgbbu = stablehlo.multiply %vitb9_pgbbcsqrt, %vitb9_pgbbinn : tensor<32x197x768xf32>
    %vitb9_pgbbt = stablehlo.tanh %vitb9_pgbbu : tensor<32x197x768xf32>
    %vitb9_pgbbone = stablehlo.constant dense<1.0> : tensor<32x197x768xf32>
    %vitb9_pgbbopt = stablehlo.add %vitb9_pgbbone, %vitb9_pgbbt : tensor<32x197x768xf32>
    %vitb9_pgbbchalf = stablehlo.constant dense<0.5> : tensor<32x197x768xf32>
    %vitb9_pgbbterm1 = stablehlo.multiply %vitb9_pgbbchalf, %vitb9_pgbbopt : tensor<32x197x768xf32>
    %vitb9_pgbbt2 = stablehlo.multiply %vitb9_pgbbt, %vitb9_pgbbt : tensor<32x197x768xf32>
    %vitb9_pgbbomt2 = stablehlo.subtract %vitb9_pgbbone, %vitb9_pgbbt2 : tensor<32x197x768xf32>
    %vitb9_pgbbhx = stablehlo.multiply %vitb9_pgbbchalf, %vitb9_ph1 : tensor<32x197x768xf32>
    %vitb9_pgbbhxo = stablehlo.multiply %vitb9_pgbbhx, %vitb9_pgbbomt2 : tensor<32x197x768xf32>
    %vitb9_pgbbc3b = stablehlo.constant dense<0.134145> : tensor<32x197x768xf32>
    %vitb9_pgbba3x2 = stablehlo.multiply %vitb9_pgbbc3b, %vitb9_pgbbx2 : tensor<32x197x768xf32>
    %vitb9_pgbbin2 = stablehlo.add %vitb9_pgbbone, %vitb9_pgbba3x2 : tensor<32x197x768xf32>
    %vitb9_pgbbup = stablehlo.multiply %vitb9_pgbbcsqrt, %vitb9_pgbbin2 : tensor<32x197x768xf32>
    %vitb9_pgbbterm2 = stablehlo.multiply %vitb9_pgbbhxo, %vitb9_pgbbup : tensor<32x197x768xf32>
    %vitb9_pgbbgp = stablehlo.add %vitb9_pgbbterm1, %vitb9_pgbbterm2 : tensor<32x197x768xf32>
    %vitb9_pgbdx = stablehlo.multiply %vitb9_pda1, %vitb9_pgbbgp : tensor<32x197x768xf32>
    %vitb9_pdx = stablehlo.dot_general %vitb9_pgbdx, %Wfc1_9, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<192x768xf32>) -> tensor<32x197x192xf32>
    %vitb9_pdWfc1 = stablehlo.dot_general %vitb9_2y, %vitb9_pgbdx, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x768xf32>) -> tensor<192x768xf32>
    %vitb9_pdbfc1 = stablehlo.reduce(%vitb9_pgbdx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<768xf32>
    %vitb9_2gbk = stablehlo.broadcast_in_dim %g2_9, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb9_2dxhat = stablehlo.multiply %vitb9_pdx, %vitb9_2gbk : tensor<32x197x192xf32>
    %vitb9_2dgpre = stablehlo.multiply %vitb9_pdx, %vitb9_2xhat : tensor<32x197x192xf32>
    %vitb9_2dg = stablehlo.reduce(%vitb9_2dgpre init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb9_2db = stablehlo.reduce(%vitb9_pdx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb9_2m1s = stablehlo.reduce(%vitb9_2dxhat init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb9_2m1 = stablehlo.divide %vitb9_2m1s, %vitb9_2nf : tensor<32x197xf32>
    %vitb9_2dxxh = stablehlo.multiply %vitb9_2dxhat, %vitb9_2xhat : tensor<32x197x192xf32>
    %vitb9_2m2s = stablehlo.reduce(%vitb9_2dxxh init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb9_2m2 = stablehlo.divide %vitb9_2m2s, %vitb9_2nf : tensor<32x197xf32>
    %vitb9_2m1b = stablehlo.broadcast_in_dim %vitb9_2m1, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb9_2m2b = stablehlo.broadcast_in_dim %vitb9_2m2, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb9_2t1 = stablehlo.subtract %vitb9_2dxhat, %vitb9_2m1b : tensor<32x197x192xf32>
    %vitb9_2xm2 = stablehlo.multiply %vitb9_2xhat, %vitb9_2m2b : tensor<32x197x192xf32>
    %vitb9_2t2 = stablehlo.subtract %vitb9_2t1, %vitb9_2xm2 : tensor<32x197x192xf32>
    %vitb9_2dx = stablehlo.multiply %vitb9_2istdb, %vitb9_2t2 : tensor<32x197x192xf32>
    %vitb9_dr1 = stablehlo.add %vitb10_dx, %vitb9_2dx : tensor<32x197x192xf32>
    %vitb9_mdP = stablehlo.dot_general %vitb9_dr1, %Wo_9, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb9_mdWo = stablehlo.dot_general %vitb9_mP, %vitb9_dr1, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb9_mdbo = stablehlo.reduce(%vitb9_dr1 init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb9_mdPr = stablehlo.reshape %vitb9_mdP : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb9_mdA = stablehlo.transpose %vitb9_mdPr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb9_mdW = stablehlo.dot_general %vitb9_mdA, %vitb9_mVh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x64xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x197xf32>
    %vitb9_mdVh = stablehlo.dot_general %vitb9_mW, %vitb9_mdA, batching_dims = [0, 1] x [0, 1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb9_mpdw = stablehlo.multiply %vitb9_mW, %vitb9_mdW : tensor<32x3x197x197xf32>
    %vitb9_msrow = stablehlo.reduce(%vitb9_mpdw init: %sc) applies stablehlo.add across dimensions = [3] : (tensor<32x3x197x197xf32>, tensor<f32>) -> tensor<32x3x197xf32>
    %vitb9_msrowb = stablehlo.broadcast_in_dim %vitb9_msrow, dims = [0, 1, 2] : (tensor<32x3x197xf32>) -> tensor<32x3x197x197xf32>
    %vitb9_mdiff = stablehlo.subtract %vitb9_mdW, %vitb9_msrowb : tensor<32x3x197x197xf32>
    %vitb9_mdSs = stablehlo.multiply %vitb9_mW, %vitb9_mdiff : tensor<32x3x197x197xf32>
    %vitb9_msclb = stablehlo.constant dense<0.125> : tensor<32x3x197x197xf32>
    %vitb9_mdS = stablehlo.multiply %vitb9_mdSs, %vitb9_msclb : tensor<32x3x197x197xf32>
    %vitb9_mdQh = stablehlo.dot_general %vitb9_mdS, %vitb9_mKh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb9_mdKh = stablehlo.dot_general %vitb9_mdS, %vitb9_mQh, batching_dims = [0, 1] x [0, 1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb9_mdQT = stablehlo.transpose %vitb9_mdQh, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb9_mdQ = stablehlo.reshape %vitb9_mdQT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb9_mdKT = stablehlo.transpose %vitb9_mdKh, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb9_mdK = stablehlo.reshape %vitb9_mdKT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb9_mdVT = stablehlo.transpose %vitb9_mdVh, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb9_mdV = stablehlo.reshape %vitb9_mdVT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb9_mdxQ = stablehlo.dot_general %vitb9_mdQ, %Wq_9, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb9_mdWQ = stablehlo.dot_general %vitb9_1y, %vitb9_mdQ, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb9_mdbQ = stablehlo.reduce(%vitb9_mdQ init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb9_mdxK = stablehlo.dot_general %vitb9_mdK, %Wk_9, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb9_mdWK = stablehlo.dot_general %vitb9_1y, %vitb9_mdK, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb9_mdbK = stablehlo.reduce(%vitb9_mdK init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb9_mdxV = stablehlo.dot_general %vitb9_mdV, %Wv_9, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb9_mdWV = stablehlo.dot_general %vitb9_1y, %vitb9_mdV, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb9_mdbV = stablehlo.reduce(%vitb9_mdV init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb9_mdxa = stablehlo.add %vitb9_mdxQ, %vitb9_mdxK : tensor<32x197x192xf32>
    %vitb9_mdx = stablehlo.add %vitb9_mdxa, %vitb9_mdxV : tensor<32x197x192xf32>
    %vitb9_1gbk = stablehlo.broadcast_in_dim %g1_9, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb9_1dxhat = stablehlo.multiply %vitb9_mdx, %vitb9_1gbk : tensor<32x197x192xf32>
    %vitb9_1dgpre = stablehlo.multiply %vitb9_mdx, %vitb9_1xhat : tensor<32x197x192xf32>
    %vitb9_1dg = stablehlo.reduce(%vitb9_1dgpre init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb9_1db = stablehlo.reduce(%vitb9_mdx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb9_1m1s = stablehlo.reduce(%vitb9_1dxhat init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb9_1m1 = stablehlo.divide %vitb9_1m1s, %vitb9_1nf : tensor<32x197xf32>
    %vitb9_1dxxh = stablehlo.multiply %vitb9_1dxhat, %vitb9_1xhat : tensor<32x197x192xf32>
    %vitb9_1m2s = stablehlo.reduce(%vitb9_1dxxh init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb9_1m2 = stablehlo.divide %vitb9_1m2s, %vitb9_1nf : tensor<32x197xf32>
    %vitb9_1m1b = stablehlo.broadcast_in_dim %vitb9_1m1, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb9_1m2b = stablehlo.broadcast_in_dim %vitb9_1m2, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb9_1t1 = stablehlo.subtract %vitb9_1dxhat, %vitb9_1m1b : tensor<32x197x192xf32>
    %vitb9_1xm2 = stablehlo.multiply %vitb9_1xhat, %vitb9_1m2b : tensor<32x197x192xf32>
    %vitb9_1t2 = stablehlo.subtract %vitb9_1t1, %vitb9_1xm2 : tensor<32x197x192xf32>
    %vitb9_1dx = stablehlo.multiply %vitb9_1istdb, %vitb9_1t2 : tensor<32x197x192xf32>
    %vitb9_dx = stablehlo.add %vitb9_dr1, %vitb9_1dx : tensor<32x197x192xf32>
    %vitb8_pda1 = stablehlo.dot_general %vitb9_dx, %Wfc2_8, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<768x192xf32>) -> tensor<32x197x768xf32>
    %vitb8_pdWfc2 = stablehlo.dot_general %vitb8_pga, %vitb9_dx, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<32x197x192xf32>) -> tensor<768x192xf32>
    %vitb8_pdbfc2 = stablehlo.reduce(%vitb9_dx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb8_pgbbx2 = stablehlo.multiply %vitb8_ph1, %vitb8_ph1 : tensor<32x197x768xf32>
    %vitb8_pgbbx3 = stablehlo.multiply %vitb8_pgbbx2, %vitb8_ph1 : tensor<32x197x768xf32>
    %vitb8_pgbbck = stablehlo.constant dense<0.044715> : tensor<32x197x768xf32>
    %vitb8_pgbbkx3 = stablehlo.multiply %vitb8_pgbbck, %vitb8_pgbbx3 : tensor<32x197x768xf32>
    %vitb8_pgbbinn = stablehlo.add %vitb8_ph1, %vitb8_pgbbkx3 : tensor<32x197x768xf32>
    %vitb8_pgbbcsqrt = stablehlo.constant dense<0.7978845608028654> : tensor<32x197x768xf32>
    %vitb8_pgbbu = stablehlo.multiply %vitb8_pgbbcsqrt, %vitb8_pgbbinn : tensor<32x197x768xf32>
    %vitb8_pgbbt = stablehlo.tanh %vitb8_pgbbu : tensor<32x197x768xf32>
    %vitb8_pgbbone = stablehlo.constant dense<1.0> : tensor<32x197x768xf32>
    %vitb8_pgbbopt = stablehlo.add %vitb8_pgbbone, %vitb8_pgbbt : tensor<32x197x768xf32>
    %vitb8_pgbbchalf = stablehlo.constant dense<0.5> : tensor<32x197x768xf32>
    %vitb8_pgbbterm1 = stablehlo.multiply %vitb8_pgbbchalf, %vitb8_pgbbopt : tensor<32x197x768xf32>
    %vitb8_pgbbt2 = stablehlo.multiply %vitb8_pgbbt, %vitb8_pgbbt : tensor<32x197x768xf32>
    %vitb8_pgbbomt2 = stablehlo.subtract %vitb8_pgbbone, %vitb8_pgbbt2 : tensor<32x197x768xf32>
    %vitb8_pgbbhx = stablehlo.multiply %vitb8_pgbbchalf, %vitb8_ph1 : tensor<32x197x768xf32>
    %vitb8_pgbbhxo = stablehlo.multiply %vitb8_pgbbhx, %vitb8_pgbbomt2 : tensor<32x197x768xf32>
    %vitb8_pgbbc3b = stablehlo.constant dense<0.134145> : tensor<32x197x768xf32>
    %vitb8_pgbba3x2 = stablehlo.multiply %vitb8_pgbbc3b, %vitb8_pgbbx2 : tensor<32x197x768xf32>
    %vitb8_pgbbin2 = stablehlo.add %vitb8_pgbbone, %vitb8_pgbba3x2 : tensor<32x197x768xf32>
    %vitb8_pgbbup = stablehlo.multiply %vitb8_pgbbcsqrt, %vitb8_pgbbin2 : tensor<32x197x768xf32>
    %vitb8_pgbbterm2 = stablehlo.multiply %vitb8_pgbbhxo, %vitb8_pgbbup : tensor<32x197x768xf32>
    %vitb8_pgbbgp = stablehlo.add %vitb8_pgbbterm1, %vitb8_pgbbterm2 : tensor<32x197x768xf32>
    %vitb8_pgbdx = stablehlo.multiply %vitb8_pda1, %vitb8_pgbbgp : tensor<32x197x768xf32>
    %vitb8_pdx = stablehlo.dot_general %vitb8_pgbdx, %Wfc1_8, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<192x768xf32>) -> tensor<32x197x192xf32>
    %vitb8_pdWfc1 = stablehlo.dot_general %vitb8_2y, %vitb8_pgbdx, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x768xf32>) -> tensor<192x768xf32>
    %vitb8_pdbfc1 = stablehlo.reduce(%vitb8_pgbdx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<768xf32>
    %vitb8_2gbk = stablehlo.broadcast_in_dim %g2_8, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb8_2dxhat = stablehlo.multiply %vitb8_pdx, %vitb8_2gbk : tensor<32x197x192xf32>
    %vitb8_2dgpre = stablehlo.multiply %vitb8_pdx, %vitb8_2xhat : tensor<32x197x192xf32>
    %vitb8_2dg = stablehlo.reduce(%vitb8_2dgpre init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb8_2db = stablehlo.reduce(%vitb8_pdx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb8_2m1s = stablehlo.reduce(%vitb8_2dxhat init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb8_2m1 = stablehlo.divide %vitb8_2m1s, %vitb8_2nf : tensor<32x197xf32>
    %vitb8_2dxxh = stablehlo.multiply %vitb8_2dxhat, %vitb8_2xhat : tensor<32x197x192xf32>
    %vitb8_2m2s = stablehlo.reduce(%vitb8_2dxxh init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb8_2m2 = stablehlo.divide %vitb8_2m2s, %vitb8_2nf : tensor<32x197xf32>
    %vitb8_2m1b = stablehlo.broadcast_in_dim %vitb8_2m1, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb8_2m2b = stablehlo.broadcast_in_dim %vitb8_2m2, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb8_2t1 = stablehlo.subtract %vitb8_2dxhat, %vitb8_2m1b : tensor<32x197x192xf32>
    %vitb8_2xm2 = stablehlo.multiply %vitb8_2xhat, %vitb8_2m2b : tensor<32x197x192xf32>
    %vitb8_2t2 = stablehlo.subtract %vitb8_2t1, %vitb8_2xm2 : tensor<32x197x192xf32>
    %vitb8_2dx = stablehlo.multiply %vitb8_2istdb, %vitb8_2t2 : tensor<32x197x192xf32>
    %vitb8_dr1 = stablehlo.add %vitb9_dx, %vitb8_2dx : tensor<32x197x192xf32>
    %vitb8_mdP = stablehlo.dot_general %vitb8_dr1, %Wo_8, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb8_mdWo = stablehlo.dot_general %vitb8_mP, %vitb8_dr1, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb8_mdbo = stablehlo.reduce(%vitb8_dr1 init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb8_mdPr = stablehlo.reshape %vitb8_mdP : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb8_mdA = stablehlo.transpose %vitb8_mdPr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb8_mdW = stablehlo.dot_general %vitb8_mdA, %vitb8_mVh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x64xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x197xf32>
    %vitb8_mdVh = stablehlo.dot_general %vitb8_mW, %vitb8_mdA, batching_dims = [0, 1] x [0, 1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb8_mpdw = stablehlo.multiply %vitb8_mW, %vitb8_mdW : tensor<32x3x197x197xf32>
    %vitb8_msrow = stablehlo.reduce(%vitb8_mpdw init: %sc) applies stablehlo.add across dimensions = [3] : (tensor<32x3x197x197xf32>, tensor<f32>) -> tensor<32x3x197xf32>
    %vitb8_msrowb = stablehlo.broadcast_in_dim %vitb8_msrow, dims = [0, 1, 2] : (tensor<32x3x197xf32>) -> tensor<32x3x197x197xf32>
    %vitb8_mdiff = stablehlo.subtract %vitb8_mdW, %vitb8_msrowb : tensor<32x3x197x197xf32>
    %vitb8_mdSs = stablehlo.multiply %vitb8_mW, %vitb8_mdiff : tensor<32x3x197x197xf32>
    %vitb8_msclb = stablehlo.constant dense<0.125> : tensor<32x3x197x197xf32>
    %vitb8_mdS = stablehlo.multiply %vitb8_mdSs, %vitb8_msclb : tensor<32x3x197x197xf32>
    %vitb8_mdQh = stablehlo.dot_general %vitb8_mdS, %vitb8_mKh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb8_mdKh = stablehlo.dot_general %vitb8_mdS, %vitb8_mQh, batching_dims = [0, 1] x [0, 1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb8_mdQT = stablehlo.transpose %vitb8_mdQh, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb8_mdQ = stablehlo.reshape %vitb8_mdQT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb8_mdKT = stablehlo.transpose %vitb8_mdKh, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb8_mdK = stablehlo.reshape %vitb8_mdKT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb8_mdVT = stablehlo.transpose %vitb8_mdVh, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb8_mdV = stablehlo.reshape %vitb8_mdVT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb8_mdxQ = stablehlo.dot_general %vitb8_mdQ, %Wq_8, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb8_mdWQ = stablehlo.dot_general %vitb8_1y, %vitb8_mdQ, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb8_mdbQ = stablehlo.reduce(%vitb8_mdQ init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb8_mdxK = stablehlo.dot_general %vitb8_mdK, %Wk_8, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb8_mdWK = stablehlo.dot_general %vitb8_1y, %vitb8_mdK, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb8_mdbK = stablehlo.reduce(%vitb8_mdK init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb8_mdxV = stablehlo.dot_general %vitb8_mdV, %Wv_8, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb8_mdWV = stablehlo.dot_general %vitb8_1y, %vitb8_mdV, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb8_mdbV = stablehlo.reduce(%vitb8_mdV init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb8_mdxa = stablehlo.add %vitb8_mdxQ, %vitb8_mdxK : tensor<32x197x192xf32>
    %vitb8_mdx = stablehlo.add %vitb8_mdxa, %vitb8_mdxV : tensor<32x197x192xf32>
    %vitb8_1gbk = stablehlo.broadcast_in_dim %g1_8, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb8_1dxhat = stablehlo.multiply %vitb8_mdx, %vitb8_1gbk : tensor<32x197x192xf32>
    %vitb8_1dgpre = stablehlo.multiply %vitb8_mdx, %vitb8_1xhat : tensor<32x197x192xf32>
    %vitb8_1dg = stablehlo.reduce(%vitb8_1dgpre init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb8_1db = stablehlo.reduce(%vitb8_mdx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb8_1m1s = stablehlo.reduce(%vitb8_1dxhat init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb8_1m1 = stablehlo.divide %vitb8_1m1s, %vitb8_1nf : tensor<32x197xf32>
    %vitb8_1dxxh = stablehlo.multiply %vitb8_1dxhat, %vitb8_1xhat : tensor<32x197x192xf32>
    %vitb8_1m2s = stablehlo.reduce(%vitb8_1dxxh init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb8_1m2 = stablehlo.divide %vitb8_1m2s, %vitb8_1nf : tensor<32x197xf32>
    %vitb8_1m1b = stablehlo.broadcast_in_dim %vitb8_1m1, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb8_1m2b = stablehlo.broadcast_in_dim %vitb8_1m2, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb8_1t1 = stablehlo.subtract %vitb8_1dxhat, %vitb8_1m1b : tensor<32x197x192xf32>
    %vitb8_1xm2 = stablehlo.multiply %vitb8_1xhat, %vitb8_1m2b : tensor<32x197x192xf32>
    %vitb8_1t2 = stablehlo.subtract %vitb8_1t1, %vitb8_1xm2 : tensor<32x197x192xf32>
    %vitb8_1dx = stablehlo.multiply %vitb8_1istdb, %vitb8_1t2 : tensor<32x197x192xf32>
    %vitb8_dx = stablehlo.add %vitb8_dr1, %vitb8_1dx : tensor<32x197x192xf32>
    %vitb7_pda1 = stablehlo.dot_general %vitb8_dx, %Wfc2_7, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<768x192xf32>) -> tensor<32x197x768xf32>
    %vitb7_pdWfc2 = stablehlo.dot_general %vitb7_pga, %vitb8_dx, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<32x197x192xf32>) -> tensor<768x192xf32>
    %vitb7_pdbfc2 = stablehlo.reduce(%vitb8_dx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb7_pgbbx2 = stablehlo.multiply %vitb7_ph1, %vitb7_ph1 : tensor<32x197x768xf32>
    %vitb7_pgbbx3 = stablehlo.multiply %vitb7_pgbbx2, %vitb7_ph1 : tensor<32x197x768xf32>
    %vitb7_pgbbck = stablehlo.constant dense<0.044715> : tensor<32x197x768xf32>
    %vitb7_pgbbkx3 = stablehlo.multiply %vitb7_pgbbck, %vitb7_pgbbx3 : tensor<32x197x768xf32>
    %vitb7_pgbbinn = stablehlo.add %vitb7_ph1, %vitb7_pgbbkx3 : tensor<32x197x768xf32>
    %vitb7_pgbbcsqrt = stablehlo.constant dense<0.7978845608028654> : tensor<32x197x768xf32>
    %vitb7_pgbbu = stablehlo.multiply %vitb7_pgbbcsqrt, %vitb7_pgbbinn : tensor<32x197x768xf32>
    %vitb7_pgbbt = stablehlo.tanh %vitb7_pgbbu : tensor<32x197x768xf32>
    %vitb7_pgbbone = stablehlo.constant dense<1.0> : tensor<32x197x768xf32>
    %vitb7_pgbbopt = stablehlo.add %vitb7_pgbbone, %vitb7_pgbbt : tensor<32x197x768xf32>
    %vitb7_pgbbchalf = stablehlo.constant dense<0.5> : tensor<32x197x768xf32>
    %vitb7_pgbbterm1 = stablehlo.multiply %vitb7_pgbbchalf, %vitb7_pgbbopt : tensor<32x197x768xf32>
    %vitb7_pgbbt2 = stablehlo.multiply %vitb7_pgbbt, %vitb7_pgbbt : tensor<32x197x768xf32>
    %vitb7_pgbbomt2 = stablehlo.subtract %vitb7_pgbbone, %vitb7_pgbbt2 : tensor<32x197x768xf32>
    %vitb7_pgbbhx = stablehlo.multiply %vitb7_pgbbchalf, %vitb7_ph1 : tensor<32x197x768xf32>
    %vitb7_pgbbhxo = stablehlo.multiply %vitb7_pgbbhx, %vitb7_pgbbomt2 : tensor<32x197x768xf32>
    %vitb7_pgbbc3b = stablehlo.constant dense<0.134145> : tensor<32x197x768xf32>
    %vitb7_pgbba3x2 = stablehlo.multiply %vitb7_pgbbc3b, %vitb7_pgbbx2 : tensor<32x197x768xf32>
    %vitb7_pgbbin2 = stablehlo.add %vitb7_pgbbone, %vitb7_pgbba3x2 : tensor<32x197x768xf32>
    %vitb7_pgbbup = stablehlo.multiply %vitb7_pgbbcsqrt, %vitb7_pgbbin2 : tensor<32x197x768xf32>
    %vitb7_pgbbterm2 = stablehlo.multiply %vitb7_pgbbhxo, %vitb7_pgbbup : tensor<32x197x768xf32>
    %vitb7_pgbbgp = stablehlo.add %vitb7_pgbbterm1, %vitb7_pgbbterm2 : tensor<32x197x768xf32>
    %vitb7_pgbdx = stablehlo.multiply %vitb7_pda1, %vitb7_pgbbgp : tensor<32x197x768xf32>
    %vitb7_pdx = stablehlo.dot_general %vitb7_pgbdx, %Wfc1_7, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<192x768xf32>) -> tensor<32x197x192xf32>
    %vitb7_pdWfc1 = stablehlo.dot_general %vitb7_2y, %vitb7_pgbdx, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x768xf32>) -> tensor<192x768xf32>
    %vitb7_pdbfc1 = stablehlo.reduce(%vitb7_pgbdx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<768xf32>
    %vitb7_2gbk = stablehlo.broadcast_in_dim %g2_7, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb7_2dxhat = stablehlo.multiply %vitb7_pdx, %vitb7_2gbk : tensor<32x197x192xf32>
    %vitb7_2dgpre = stablehlo.multiply %vitb7_pdx, %vitb7_2xhat : tensor<32x197x192xf32>
    %vitb7_2dg = stablehlo.reduce(%vitb7_2dgpre init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb7_2db = stablehlo.reduce(%vitb7_pdx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb7_2m1s = stablehlo.reduce(%vitb7_2dxhat init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb7_2m1 = stablehlo.divide %vitb7_2m1s, %vitb7_2nf : tensor<32x197xf32>
    %vitb7_2dxxh = stablehlo.multiply %vitb7_2dxhat, %vitb7_2xhat : tensor<32x197x192xf32>
    %vitb7_2m2s = stablehlo.reduce(%vitb7_2dxxh init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb7_2m2 = stablehlo.divide %vitb7_2m2s, %vitb7_2nf : tensor<32x197xf32>
    %vitb7_2m1b = stablehlo.broadcast_in_dim %vitb7_2m1, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb7_2m2b = stablehlo.broadcast_in_dim %vitb7_2m2, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb7_2t1 = stablehlo.subtract %vitb7_2dxhat, %vitb7_2m1b : tensor<32x197x192xf32>
    %vitb7_2xm2 = stablehlo.multiply %vitb7_2xhat, %vitb7_2m2b : tensor<32x197x192xf32>
    %vitb7_2t2 = stablehlo.subtract %vitb7_2t1, %vitb7_2xm2 : tensor<32x197x192xf32>
    %vitb7_2dx = stablehlo.multiply %vitb7_2istdb, %vitb7_2t2 : tensor<32x197x192xf32>
    %vitb7_dr1 = stablehlo.add %vitb8_dx, %vitb7_2dx : tensor<32x197x192xf32>
    %vitb7_mdP = stablehlo.dot_general %vitb7_dr1, %Wo_7, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb7_mdWo = stablehlo.dot_general %vitb7_mP, %vitb7_dr1, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb7_mdbo = stablehlo.reduce(%vitb7_dr1 init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb7_mdPr = stablehlo.reshape %vitb7_mdP : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb7_mdA = stablehlo.transpose %vitb7_mdPr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb7_mdW = stablehlo.dot_general %vitb7_mdA, %vitb7_mVh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x64xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x197xf32>
    %vitb7_mdVh = stablehlo.dot_general %vitb7_mW, %vitb7_mdA, batching_dims = [0, 1] x [0, 1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb7_mpdw = stablehlo.multiply %vitb7_mW, %vitb7_mdW : tensor<32x3x197x197xf32>
    %vitb7_msrow = stablehlo.reduce(%vitb7_mpdw init: %sc) applies stablehlo.add across dimensions = [3] : (tensor<32x3x197x197xf32>, tensor<f32>) -> tensor<32x3x197xf32>
    %vitb7_msrowb = stablehlo.broadcast_in_dim %vitb7_msrow, dims = [0, 1, 2] : (tensor<32x3x197xf32>) -> tensor<32x3x197x197xf32>
    %vitb7_mdiff = stablehlo.subtract %vitb7_mdW, %vitb7_msrowb : tensor<32x3x197x197xf32>
    %vitb7_mdSs = stablehlo.multiply %vitb7_mW, %vitb7_mdiff : tensor<32x3x197x197xf32>
    %vitb7_msclb = stablehlo.constant dense<0.125> : tensor<32x3x197x197xf32>
    %vitb7_mdS = stablehlo.multiply %vitb7_mdSs, %vitb7_msclb : tensor<32x3x197x197xf32>
    %vitb7_mdQh = stablehlo.dot_general %vitb7_mdS, %vitb7_mKh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb7_mdKh = stablehlo.dot_general %vitb7_mdS, %vitb7_mQh, batching_dims = [0, 1] x [0, 1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb7_mdQT = stablehlo.transpose %vitb7_mdQh, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb7_mdQ = stablehlo.reshape %vitb7_mdQT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb7_mdKT = stablehlo.transpose %vitb7_mdKh, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb7_mdK = stablehlo.reshape %vitb7_mdKT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb7_mdVT = stablehlo.transpose %vitb7_mdVh, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb7_mdV = stablehlo.reshape %vitb7_mdVT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb7_mdxQ = stablehlo.dot_general %vitb7_mdQ, %Wq_7, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb7_mdWQ = stablehlo.dot_general %vitb7_1y, %vitb7_mdQ, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb7_mdbQ = stablehlo.reduce(%vitb7_mdQ init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb7_mdxK = stablehlo.dot_general %vitb7_mdK, %Wk_7, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb7_mdWK = stablehlo.dot_general %vitb7_1y, %vitb7_mdK, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb7_mdbK = stablehlo.reduce(%vitb7_mdK init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb7_mdxV = stablehlo.dot_general %vitb7_mdV, %Wv_7, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb7_mdWV = stablehlo.dot_general %vitb7_1y, %vitb7_mdV, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb7_mdbV = stablehlo.reduce(%vitb7_mdV init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb7_mdxa = stablehlo.add %vitb7_mdxQ, %vitb7_mdxK : tensor<32x197x192xf32>
    %vitb7_mdx = stablehlo.add %vitb7_mdxa, %vitb7_mdxV : tensor<32x197x192xf32>
    %vitb7_1gbk = stablehlo.broadcast_in_dim %g1_7, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb7_1dxhat = stablehlo.multiply %vitb7_mdx, %vitb7_1gbk : tensor<32x197x192xf32>
    %vitb7_1dgpre = stablehlo.multiply %vitb7_mdx, %vitb7_1xhat : tensor<32x197x192xf32>
    %vitb7_1dg = stablehlo.reduce(%vitb7_1dgpre init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb7_1db = stablehlo.reduce(%vitb7_mdx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb7_1m1s = stablehlo.reduce(%vitb7_1dxhat init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb7_1m1 = stablehlo.divide %vitb7_1m1s, %vitb7_1nf : tensor<32x197xf32>
    %vitb7_1dxxh = stablehlo.multiply %vitb7_1dxhat, %vitb7_1xhat : tensor<32x197x192xf32>
    %vitb7_1m2s = stablehlo.reduce(%vitb7_1dxxh init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb7_1m2 = stablehlo.divide %vitb7_1m2s, %vitb7_1nf : tensor<32x197xf32>
    %vitb7_1m1b = stablehlo.broadcast_in_dim %vitb7_1m1, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb7_1m2b = stablehlo.broadcast_in_dim %vitb7_1m2, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb7_1t1 = stablehlo.subtract %vitb7_1dxhat, %vitb7_1m1b : tensor<32x197x192xf32>
    %vitb7_1xm2 = stablehlo.multiply %vitb7_1xhat, %vitb7_1m2b : tensor<32x197x192xf32>
    %vitb7_1t2 = stablehlo.subtract %vitb7_1t1, %vitb7_1xm2 : tensor<32x197x192xf32>
    %vitb7_1dx = stablehlo.multiply %vitb7_1istdb, %vitb7_1t2 : tensor<32x197x192xf32>
    %vitb7_dx = stablehlo.add %vitb7_dr1, %vitb7_1dx : tensor<32x197x192xf32>
    %vitb6_pda1 = stablehlo.dot_general %vitb7_dx, %Wfc2_6, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<768x192xf32>) -> tensor<32x197x768xf32>
    %vitb6_pdWfc2 = stablehlo.dot_general %vitb6_pga, %vitb7_dx, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<32x197x192xf32>) -> tensor<768x192xf32>
    %vitb6_pdbfc2 = stablehlo.reduce(%vitb7_dx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb6_pgbbx2 = stablehlo.multiply %vitb6_ph1, %vitb6_ph1 : tensor<32x197x768xf32>
    %vitb6_pgbbx3 = stablehlo.multiply %vitb6_pgbbx2, %vitb6_ph1 : tensor<32x197x768xf32>
    %vitb6_pgbbck = stablehlo.constant dense<0.044715> : tensor<32x197x768xf32>
    %vitb6_pgbbkx3 = stablehlo.multiply %vitb6_pgbbck, %vitb6_pgbbx3 : tensor<32x197x768xf32>
    %vitb6_pgbbinn = stablehlo.add %vitb6_ph1, %vitb6_pgbbkx3 : tensor<32x197x768xf32>
    %vitb6_pgbbcsqrt = stablehlo.constant dense<0.7978845608028654> : tensor<32x197x768xf32>
    %vitb6_pgbbu = stablehlo.multiply %vitb6_pgbbcsqrt, %vitb6_pgbbinn : tensor<32x197x768xf32>
    %vitb6_pgbbt = stablehlo.tanh %vitb6_pgbbu : tensor<32x197x768xf32>
    %vitb6_pgbbone = stablehlo.constant dense<1.0> : tensor<32x197x768xf32>
    %vitb6_pgbbopt = stablehlo.add %vitb6_pgbbone, %vitb6_pgbbt : tensor<32x197x768xf32>
    %vitb6_pgbbchalf = stablehlo.constant dense<0.5> : tensor<32x197x768xf32>
    %vitb6_pgbbterm1 = stablehlo.multiply %vitb6_pgbbchalf, %vitb6_pgbbopt : tensor<32x197x768xf32>
    %vitb6_pgbbt2 = stablehlo.multiply %vitb6_pgbbt, %vitb6_pgbbt : tensor<32x197x768xf32>
    %vitb6_pgbbomt2 = stablehlo.subtract %vitb6_pgbbone, %vitb6_pgbbt2 : tensor<32x197x768xf32>
    %vitb6_pgbbhx = stablehlo.multiply %vitb6_pgbbchalf, %vitb6_ph1 : tensor<32x197x768xf32>
    %vitb6_pgbbhxo = stablehlo.multiply %vitb6_pgbbhx, %vitb6_pgbbomt2 : tensor<32x197x768xf32>
    %vitb6_pgbbc3b = stablehlo.constant dense<0.134145> : tensor<32x197x768xf32>
    %vitb6_pgbba3x2 = stablehlo.multiply %vitb6_pgbbc3b, %vitb6_pgbbx2 : tensor<32x197x768xf32>
    %vitb6_pgbbin2 = stablehlo.add %vitb6_pgbbone, %vitb6_pgbba3x2 : tensor<32x197x768xf32>
    %vitb6_pgbbup = stablehlo.multiply %vitb6_pgbbcsqrt, %vitb6_pgbbin2 : tensor<32x197x768xf32>
    %vitb6_pgbbterm2 = stablehlo.multiply %vitb6_pgbbhxo, %vitb6_pgbbup : tensor<32x197x768xf32>
    %vitb6_pgbbgp = stablehlo.add %vitb6_pgbbterm1, %vitb6_pgbbterm2 : tensor<32x197x768xf32>
    %vitb6_pgbdx = stablehlo.multiply %vitb6_pda1, %vitb6_pgbbgp : tensor<32x197x768xf32>
    %vitb6_pdx = stablehlo.dot_general %vitb6_pgbdx, %Wfc1_6, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<192x768xf32>) -> tensor<32x197x192xf32>
    %vitb6_pdWfc1 = stablehlo.dot_general %vitb6_2y, %vitb6_pgbdx, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x768xf32>) -> tensor<192x768xf32>
    %vitb6_pdbfc1 = stablehlo.reduce(%vitb6_pgbdx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<768xf32>
    %vitb6_2gbk = stablehlo.broadcast_in_dim %g2_6, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb6_2dxhat = stablehlo.multiply %vitb6_pdx, %vitb6_2gbk : tensor<32x197x192xf32>
    %vitb6_2dgpre = stablehlo.multiply %vitb6_pdx, %vitb6_2xhat : tensor<32x197x192xf32>
    %vitb6_2dg = stablehlo.reduce(%vitb6_2dgpre init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb6_2db = stablehlo.reduce(%vitb6_pdx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb6_2m1s = stablehlo.reduce(%vitb6_2dxhat init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb6_2m1 = stablehlo.divide %vitb6_2m1s, %vitb6_2nf : tensor<32x197xf32>
    %vitb6_2dxxh = stablehlo.multiply %vitb6_2dxhat, %vitb6_2xhat : tensor<32x197x192xf32>
    %vitb6_2m2s = stablehlo.reduce(%vitb6_2dxxh init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb6_2m2 = stablehlo.divide %vitb6_2m2s, %vitb6_2nf : tensor<32x197xf32>
    %vitb6_2m1b = stablehlo.broadcast_in_dim %vitb6_2m1, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb6_2m2b = stablehlo.broadcast_in_dim %vitb6_2m2, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb6_2t1 = stablehlo.subtract %vitb6_2dxhat, %vitb6_2m1b : tensor<32x197x192xf32>
    %vitb6_2xm2 = stablehlo.multiply %vitb6_2xhat, %vitb6_2m2b : tensor<32x197x192xf32>
    %vitb6_2t2 = stablehlo.subtract %vitb6_2t1, %vitb6_2xm2 : tensor<32x197x192xf32>
    %vitb6_2dx = stablehlo.multiply %vitb6_2istdb, %vitb6_2t2 : tensor<32x197x192xf32>
    %vitb6_dr1 = stablehlo.add %vitb7_dx, %vitb6_2dx : tensor<32x197x192xf32>
    %vitb6_mdP = stablehlo.dot_general %vitb6_dr1, %Wo_6, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb6_mdWo = stablehlo.dot_general %vitb6_mP, %vitb6_dr1, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb6_mdbo = stablehlo.reduce(%vitb6_dr1 init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb6_mdPr = stablehlo.reshape %vitb6_mdP : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb6_mdA = stablehlo.transpose %vitb6_mdPr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb6_mdW = stablehlo.dot_general %vitb6_mdA, %vitb6_mVh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x64xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x197xf32>
    %vitb6_mdVh = stablehlo.dot_general %vitb6_mW, %vitb6_mdA, batching_dims = [0, 1] x [0, 1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb6_mpdw = stablehlo.multiply %vitb6_mW, %vitb6_mdW : tensor<32x3x197x197xf32>
    %vitb6_msrow = stablehlo.reduce(%vitb6_mpdw init: %sc) applies stablehlo.add across dimensions = [3] : (tensor<32x3x197x197xf32>, tensor<f32>) -> tensor<32x3x197xf32>
    %vitb6_msrowb = stablehlo.broadcast_in_dim %vitb6_msrow, dims = [0, 1, 2] : (tensor<32x3x197xf32>) -> tensor<32x3x197x197xf32>
    %vitb6_mdiff = stablehlo.subtract %vitb6_mdW, %vitb6_msrowb : tensor<32x3x197x197xf32>
    %vitb6_mdSs = stablehlo.multiply %vitb6_mW, %vitb6_mdiff : tensor<32x3x197x197xf32>
    %vitb6_msclb = stablehlo.constant dense<0.125> : tensor<32x3x197x197xf32>
    %vitb6_mdS = stablehlo.multiply %vitb6_mdSs, %vitb6_msclb : tensor<32x3x197x197xf32>
    %vitb6_mdQh = stablehlo.dot_general %vitb6_mdS, %vitb6_mKh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb6_mdKh = stablehlo.dot_general %vitb6_mdS, %vitb6_mQh, batching_dims = [0, 1] x [0, 1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb6_mdQT = stablehlo.transpose %vitb6_mdQh, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb6_mdQ = stablehlo.reshape %vitb6_mdQT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb6_mdKT = stablehlo.transpose %vitb6_mdKh, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb6_mdK = stablehlo.reshape %vitb6_mdKT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb6_mdVT = stablehlo.transpose %vitb6_mdVh, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb6_mdV = stablehlo.reshape %vitb6_mdVT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb6_mdxQ = stablehlo.dot_general %vitb6_mdQ, %Wq_6, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb6_mdWQ = stablehlo.dot_general %vitb6_1y, %vitb6_mdQ, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb6_mdbQ = stablehlo.reduce(%vitb6_mdQ init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb6_mdxK = stablehlo.dot_general %vitb6_mdK, %Wk_6, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb6_mdWK = stablehlo.dot_general %vitb6_1y, %vitb6_mdK, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb6_mdbK = stablehlo.reduce(%vitb6_mdK init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb6_mdxV = stablehlo.dot_general %vitb6_mdV, %Wv_6, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb6_mdWV = stablehlo.dot_general %vitb6_1y, %vitb6_mdV, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb6_mdbV = stablehlo.reduce(%vitb6_mdV init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb6_mdxa = stablehlo.add %vitb6_mdxQ, %vitb6_mdxK : tensor<32x197x192xf32>
    %vitb6_mdx = stablehlo.add %vitb6_mdxa, %vitb6_mdxV : tensor<32x197x192xf32>
    %vitb6_1gbk = stablehlo.broadcast_in_dim %g1_6, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb6_1dxhat = stablehlo.multiply %vitb6_mdx, %vitb6_1gbk : tensor<32x197x192xf32>
    %vitb6_1dgpre = stablehlo.multiply %vitb6_mdx, %vitb6_1xhat : tensor<32x197x192xf32>
    %vitb6_1dg = stablehlo.reduce(%vitb6_1dgpre init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb6_1db = stablehlo.reduce(%vitb6_mdx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb6_1m1s = stablehlo.reduce(%vitb6_1dxhat init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb6_1m1 = stablehlo.divide %vitb6_1m1s, %vitb6_1nf : tensor<32x197xf32>
    %vitb6_1dxxh = stablehlo.multiply %vitb6_1dxhat, %vitb6_1xhat : tensor<32x197x192xf32>
    %vitb6_1m2s = stablehlo.reduce(%vitb6_1dxxh init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb6_1m2 = stablehlo.divide %vitb6_1m2s, %vitb6_1nf : tensor<32x197xf32>
    %vitb6_1m1b = stablehlo.broadcast_in_dim %vitb6_1m1, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb6_1m2b = stablehlo.broadcast_in_dim %vitb6_1m2, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb6_1t1 = stablehlo.subtract %vitb6_1dxhat, %vitb6_1m1b : tensor<32x197x192xf32>
    %vitb6_1xm2 = stablehlo.multiply %vitb6_1xhat, %vitb6_1m2b : tensor<32x197x192xf32>
    %vitb6_1t2 = stablehlo.subtract %vitb6_1t1, %vitb6_1xm2 : tensor<32x197x192xf32>
    %vitb6_1dx = stablehlo.multiply %vitb6_1istdb, %vitb6_1t2 : tensor<32x197x192xf32>
    %vitb6_dx = stablehlo.add %vitb6_dr1, %vitb6_1dx : tensor<32x197x192xf32>
    %vitb5_pda1 = stablehlo.dot_general %vitb6_dx, %Wfc2_5, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<768x192xf32>) -> tensor<32x197x768xf32>
    %vitb5_pdWfc2 = stablehlo.dot_general %vitb5_pga, %vitb6_dx, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<32x197x192xf32>) -> tensor<768x192xf32>
    %vitb5_pdbfc2 = stablehlo.reduce(%vitb6_dx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb5_pgbbx2 = stablehlo.multiply %vitb5_ph1, %vitb5_ph1 : tensor<32x197x768xf32>
    %vitb5_pgbbx3 = stablehlo.multiply %vitb5_pgbbx2, %vitb5_ph1 : tensor<32x197x768xf32>
    %vitb5_pgbbck = stablehlo.constant dense<0.044715> : tensor<32x197x768xf32>
    %vitb5_pgbbkx3 = stablehlo.multiply %vitb5_pgbbck, %vitb5_pgbbx3 : tensor<32x197x768xf32>
    %vitb5_pgbbinn = stablehlo.add %vitb5_ph1, %vitb5_pgbbkx3 : tensor<32x197x768xf32>
    %vitb5_pgbbcsqrt = stablehlo.constant dense<0.7978845608028654> : tensor<32x197x768xf32>
    %vitb5_pgbbu = stablehlo.multiply %vitb5_pgbbcsqrt, %vitb5_pgbbinn : tensor<32x197x768xf32>
    %vitb5_pgbbt = stablehlo.tanh %vitb5_pgbbu : tensor<32x197x768xf32>
    %vitb5_pgbbone = stablehlo.constant dense<1.0> : tensor<32x197x768xf32>
    %vitb5_pgbbopt = stablehlo.add %vitb5_pgbbone, %vitb5_pgbbt : tensor<32x197x768xf32>
    %vitb5_pgbbchalf = stablehlo.constant dense<0.5> : tensor<32x197x768xf32>
    %vitb5_pgbbterm1 = stablehlo.multiply %vitb5_pgbbchalf, %vitb5_pgbbopt : tensor<32x197x768xf32>
    %vitb5_pgbbt2 = stablehlo.multiply %vitb5_pgbbt, %vitb5_pgbbt : tensor<32x197x768xf32>
    %vitb5_pgbbomt2 = stablehlo.subtract %vitb5_pgbbone, %vitb5_pgbbt2 : tensor<32x197x768xf32>
    %vitb5_pgbbhx = stablehlo.multiply %vitb5_pgbbchalf, %vitb5_ph1 : tensor<32x197x768xf32>
    %vitb5_pgbbhxo = stablehlo.multiply %vitb5_pgbbhx, %vitb5_pgbbomt2 : tensor<32x197x768xf32>
    %vitb5_pgbbc3b = stablehlo.constant dense<0.134145> : tensor<32x197x768xf32>
    %vitb5_pgbba3x2 = stablehlo.multiply %vitb5_pgbbc3b, %vitb5_pgbbx2 : tensor<32x197x768xf32>
    %vitb5_pgbbin2 = stablehlo.add %vitb5_pgbbone, %vitb5_pgbba3x2 : tensor<32x197x768xf32>
    %vitb5_pgbbup = stablehlo.multiply %vitb5_pgbbcsqrt, %vitb5_pgbbin2 : tensor<32x197x768xf32>
    %vitb5_pgbbterm2 = stablehlo.multiply %vitb5_pgbbhxo, %vitb5_pgbbup : tensor<32x197x768xf32>
    %vitb5_pgbbgp = stablehlo.add %vitb5_pgbbterm1, %vitb5_pgbbterm2 : tensor<32x197x768xf32>
    %vitb5_pgbdx = stablehlo.multiply %vitb5_pda1, %vitb5_pgbbgp : tensor<32x197x768xf32>
    %vitb5_pdx = stablehlo.dot_general %vitb5_pgbdx, %Wfc1_5, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<192x768xf32>) -> tensor<32x197x192xf32>
    %vitb5_pdWfc1 = stablehlo.dot_general %vitb5_2y, %vitb5_pgbdx, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x768xf32>) -> tensor<192x768xf32>
    %vitb5_pdbfc1 = stablehlo.reduce(%vitb5_pgbdx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<768xf32>
    %vitb5_2gbk = stablehlo.broadcast_in_dim %g2_5, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb5_2dxhat = stablehlo.multiply %vitb5_pdx, %vitb5_2gbk : tensor<32x197x192xf32>
    %vitb5_2dgpre = stablehlo.multiply %vitb5_pdx, %vitb5_2xhat : tensor<32x197x192xf32>
    %vitb5_2dg = stablehlo.reduce(%vitb5_2dgpre init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb5_2db = stablehlo.reduce(%vitb5_pdx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb5_2m1s = stablehlo.reduce(%vitb5_2dxhat init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb5_2m1 = stablehlo.divide %vitb5_2m1s, %vitb5_2nf : tensor<32x197xf32>
    %vitb5_2dxxh = stablehlo.multiply %vitb5_2dxhat, %vitb5_2xhat : tensor<32x197x192xf32>
    %vitb5_2m2s = stablehlo.reduce(%vitb5_2dxxh init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb5_2m2 = stablehlo.divide %vitb5_2m2s, %vitb5_2nf : tensor<32x197xf32>
    %vitb5_2m1b = stablehlo.broadcast_in_dim %vitb5_2m1, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb5_2m2b = stablehlo.broadcast_in_dim %vitb5_2m2, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb5_2t1 = stablehlo.subtract %vitb5_2dxhat, %vitb5_2m1b : tensor<32x197x192xf32>
    %vitb5_2xm2 = stablehlo.multiply %vitb5_2xhat, %vitb5_2m2b : tensor<32x197x192xf32>
    %vitb5_2t2 = stablehlo.subtract %vitb5_2t1, %vitb5_2xm2 : tensor<32x197x192xf32>
    %vitb5_2dx = stablehlo.multiply %vitb5_2istdb, %vitb5_2t2 : tensor<32x197x192xf32>
    %vitb5_dr1 = stablehlo.add %vitb6_dx, %vitb5_2dx : tensor<32x197x192xf32>
    %vitb5_mdP = stablehlo.dot_general %vitb5_dr1, %Wo_5, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb5_mdWo = stablehlo.dot_general %vitb5_mP, %vitb5_dr1, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb5_mdbo = stablehlo.reduce(%vitb5_dr1 init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb5_mdPr = stablehlo.reshape %vitb5_mdP : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb5_mdA = stablehlo.transpose %vitb5_mdPr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb5_mdW = stablehlo.dot_general %vitb5_mdA, %vitb5_mVh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x64xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x197xf32>
    %vitb5_mdVh = stablehlo.dot_general %vitb5_mW, %vitb5_mdA, batching_dims = [0, 1] x [0, 1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb5_mpdw = stablehlo.multiply %vitb5_mW, %vitb5_mdW : tensor<32x3x197x197xf32>
    %vitb5_msrow = stablehlo.reduce(%vitb5_mpdw init: %sc) applies stablehlo.add across dimensions = [3] : (tensor<32x3x197x197xf32>, tensor<f32>) -> tensor<32x3x197xf32>
    %vitb5_msrowb = stablehlo.broadcast_in_dim %vitb5_msrow, dims = [0, 1, 2] : (tensor<32x3x197xf32>) -> tensor<32x3x197x197xf32>
    %vitb5_mdiff = stablehlo.subtract %vitb5_mdW, %vitb5_msrowb : tensor<32x3x197x197xf32>
    %vitb5_mdSs = stablehlo.multiply %vitb5_mW, %vitb5_mdiff : tensor<32x3x197x197xf32>
    %vitb5_msclb = stablehlo.constant dense<0.125> : tensor<32x3x197x197xf32>
    %vitb5_mdS = stablehlo.multiply %vitb5_mdSs, %vitb5_msclb : tensor<32x3x197x197xf32>
    %vitb5_mdQh = stablehlo.dot_general %vitb5_mdS, %vitb5_mKh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb5_mdKh = stablehlo.dot_general %vitb5_mdS, %vitb5_mQh, batching_dims = [0, 1] x [0, 1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb5_mdQT = stablehlo.transpose %vitb5_mdQh, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb5_mdQ = stablehlo.reshape %vitb5_mdQT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb5_mdKT = stablehlo.transpose %vitb5_mdKh, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb5_mdK = stablehlo.reshape %vitb5_mdKT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb5_mdVT = stablehlo.transpose %vitb5_mdVh, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb5_mdV = stablehlo.reshape %vitb5_mdVT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb5_mdxQ = stablehlo.dot_general %vitb5_mdQ, %Wq_5, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb5_mdWQ = stablehlo.dot_general %vitb5_1y, %vitb5_mdQ, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb5_mdbQ = stablehlo.reduce(%vitb5_mdQ init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb5_mdxK = stablehlo.dot_general %vitb5_mdK, %Wk_5, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb5_mdWK = stablehlo.dot_general %vitb5_1y, %vitb5_mdK, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb5_mdbK = stablehlo.reduce(%vitb5_mdK init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb5_mdxV = stablehlo.dot_general %vitb5_mdV, %Wv_5, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb5_mdWV = stablehlo.dot_general %vitb5_1y, %vitb5_mdV, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb5_mdbV = stablehlo.reduce(%vitb5_mdV init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb5_mdxa = stablehlo.add %vitb5_mdxQ, %vitb5_mdxK : tensor<32x197x192xf32>
    %vitb5_mdx = stablehlo.add %vitb5_mdxa, %vitb5_mdxV : tensor<32x197x192xf32>
    %vitb5_1gbk = stablehlo.broadcast_in_dim %g1_5, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb5_1dxhat = stablehlo.multiply %vitb5_mdx, %vitb5_1gbk : tensor<32x197x192xf32>
    %vitb5_1dgpre = stablehlo.multiply %vitb5_mdx, %vitb5_1xhat : tensor<32x197x192xf32>
    %vitb5_1dg = stablehlo.reduce(%vitb5_1dgpre init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb5_1db = stablehlo.reduce(%vitb5_mdx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb5_1m1s = stablehlo.reduce(%vitb5_1dxhat init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb5_1m1 = stablehlo.divide %vitb5_1m1s, %vitb5_1nf : tensor<32x197xf32>
    %vitb5_1dxxh = stablehlo.multiply %vitb5_1dxhat, %vitb5_1xhat : tensor<32x197x192xf32>
    %vitb5_1m2s = stablehlo.reduce(%vitb5_1dxxh init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb5_1m2 = stablehlo.divide %vitb5_1m2s, %vitb5_1nf : tensor<32x197xf32>
    %vitb5_1m1b = stablehlo.broadcast_in_dim %vitb5_1m1, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb5_1m2b = stablehlo.broadcast_in_dim %vitb5_1m2, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb5_1t1 = stablehlo.subtract %vitb5_1dxhat, %vitb5_1m1b : tensor<32x197x192xf32>
    %vitb5_1xm2 = stablehlo.multiply %vitb5_1xhat, %vitb5_1m2b : tensor<32x197x192xf32>
    %vitb5_1t2 = stablehlo.subtract %vitb5_1t1, %vitb5_1xm2 : tensor<32x197x192xf32>
    %vitb5_1dx = stablehlo.multiply %vitb5_1istdb, %vitb5_1t2 : tensor<32x197x192xf32>
    %vitb5_dx = stablehlo.add %vitb5_dr1, %vitb5_1dx : tensor<32x197x192xf32>
    %vitb4_pda1 = stablehlo.dot_general %vitb5_dx, %Wfc2_4, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<768x192xf32>) -> tensor<32x197x768xf32>
    %vitb4_pdWfc2 = stablehlo.dot_general %vitb4_pga, %vitb5_dx, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<32x197x192xf32>) -> tensor<768x192xf32>
    %vitb4_pdbfc2 = stablehlo.reduce(%vitb5_dx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb4_pgbbx2 = stablehlo.multiply %vitb4_ph1, %vitb4_ph1 : tensor<32x197x768xf32>
    %vitb4_pgbbx3 = stablehlo.multiply %vitb4_pgbbx2, %vitb4_ph1 : tensor<32x197x768xf32>
    %vitb4_pgbbck = stablehlo.constant dense<0.044715> : tensor<32x197x768xf32>
    %vitb4_pgbbkx3 = stablehlo.multiply %vitb4_pgbbck, %vitb4_pgbbx3 : tensor<32x197x768xf32>
    %vitb4_pgbbinn = stablehlo.add %vitb4_ph1, %vitb4_pgbbkx3 : tensor<32x197x768xf32>
    %vitb4_pgbbcsqrt = stablehlo.constant dense<0.7978845608028654> : tensor<32x197x768xf32>
    %vitb4_pgbbu = stablehlo.multiply %vitb4_pgbbcsqrt, %vitb4_pgbbinn : tensor<32x197x768xf32>
    %vitb4_pgbbt = stablehlo.tanh %vitb4_pgbbu : tensor<32x197x768xf32>
    %vitb4_pgbbone = stablehlo.constant dense<1.0> : tensor<32x197x768xf32>
    %vitb4_pgbbopt = stablehlo.add %vitb4_pgbbone, %vitb4_pgbbt : tensor<32x197x768xf32>
    %vitb4_pgbbchalf = stablehlo.constant dense<0.5> : tensor<32x197x768xf32>
    %vitb4_pgbbterm1 = stablehlo.multiply %vitb4_pgbbchalf, %vitb4_pgbbopt : tensor<32x197x768xf32>
    %vitb4_pgbbt2 = stablehlo.multiply %vitb4_pgbbt, %vitb4_pgbbt : tensor<32x197x768xf32>
    %vitb4_pgbbomt2 = stablehlo.subtract %vitb4_pgbbone, %vitb4_pgbbt2 : tensor<32x197x768xf32>
    %vitb4_pgbbhx = stablehlo.multiply %vitb4_pgbbchalf, %vitb4_ph1 : tensor<32x197x768xf32>
    %vitb4_pgbbhxo = stablehlo.multiply %vitb4_pgbbhx, %vitb4_pgbbomt2 : tensor<32x197x768xf32>
    %vitb4_pgbbc3b = stablehlo.constant dense<0.134145> : tensor<32x197x768xf32>
    %vitb4_pgbba3x2 = stablehlo.multiply %vitb4_pgbbc3b, %vitb4_pgbbx2 : tensor<32x197x768xf32>
    %vitb4_pgbbin2 = stablehlo.add %vitb4_pgbbone, %vitb4_pgbba3x2 : tensor<32x197x768xf32>
    %vitb4_pgbbup = stablehlo.multiply %vitb4_pgbbcsqrt, %vitb4_pgbbin2 : tensor<32x197x768xf32>
    %vitb4_pgbbterm2 = stablehlo.multiply %vitb4_pgbbhxo, %vitb4_pgbbup : tensor<32x197x768xf32>
    %vitb4_pgbbgp = stablehlo.add %vitb4_pgbbterm1, %vitb4_pgbbterm2 : tensor<32x197x768xf32>
    %vitb4_pgbdx = stablehlo.multiply %vitb4_pda1, %vitb4_pgbbgp : tensor<32x197x768xf32>
    %vitb4_pdx = stablehlo.dot_general %vitb4_pgbdx, %Wfc1_4, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<192x768xf32>) -> tensor<32x197x192xf32>
    %vitb4_pdWfc1 = stablehlo.dot_general %vitb4_2y, %vitb4_pgbdx, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x768xf32>) -> tensor<192x768xf32>
    %vitb4_pdbfc1 = stablehlo.reduce(%vitb4_pgbdx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<768xf32>
    %vitb4_2gbk = stablehlo.broadcast_in_dim %g2_4, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb4_2dxhat = stablehlo.multiply %vitb4_pdx, %vitb4_2gbk : tensor<32x197x192xf32>
    %vitb4_2dgpre = stablehlo.multiply %vitb4_pdx, %vitb4_2xhat : tensor<32x197x192xf32>
    %vitb4_2dg = stablehlo.reduce(%vitb4_2dgpre init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb4_2db = stablehlo.reduce(%vitb4_pdx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb4_2m1s = stablehlo.reduce(%vitb4_2dxhat init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb4_2m1 = stablehlo.divide %vitb4_2m1s, %vitb4_2nf : tensor<32x197xf32>
    %vitb4_2dxxh = stablehlo.multiply %vitb4_2dxhat, %vitb4_2xhat : tensor<32x197x192xf32>
    %vitb4_2m2s = stablehlo.reduce(%vitb4_2dxxh init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb4_2m2 = stablehlo.divide %vitb4_2m2s, %vitb4_2nf : tensor<32x197xf32>
    %vitb4_2m1b = stablehlo.broadcast_in_dim %vitb4_2m1, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb4_2m2b = stablehlo.broadcast_in_dim %vitb4_2m2, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb4_2t1 = stablehlo.subtract %vitb4_2dxhat, %vitb4_2m1b : tensor<32x197x192xf32>
    %vitb4_2xm2 = stablehlo.multiply %vitb4_2xhat, %vitb4_2m2b : tensor<32x197x192xf32>
    %vitb4_2t2 = stablehlo.subtract %vitb4_2t1, %vitb4_2xm2 : tensor<32x197x192xf32>
    %vitb4_2dx = stablehlo.multiply %vitb4_2istdb, %vitb4_2t2 : tensor<32x197x192xf32>
    %vitb4_dr1 = stablehlo.add %vitb5_dx, %vitb4_2dx : tensor<32x197x192xf32>
    %vitb4_mdP = stablehlo.dot_general %vitb4_dr1, %Wo_4, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb4_mdWo = stablehlo.dot_general %vitb4_mP, %vitb4_dr1, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb4_mdbo = stablehlo.reduce(%vitb4_dr1 init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb4_mdPr = stablehlo.reshape %vitb4_mdP : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb4_mdA = stablehlo.transpose %vitb4_mdPr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb4_mdW = stablehlo.dot_general %vitb4_mdA, %vitb4_mVh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x64xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x197xf32>
    %vitb4_mdVh = stablehlo.dot_general %vitb4_mW, %vitb4_mdA, batching_dims = [0, 1] x [0, 1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb4_mpdw = stablehlo.multiply %vitb4_mW, %vitb4_mdW : tensor<32x3x197x197xf32>
    %vitb4_msrow = stablehlo.reduce(%vitb4_mpdw init: %sc) applies stablehlo.add across dimensions = [3] : (tensor<32x3x197x197xf32>, tensor<f32>) -> tensor<32x3x197xf32>
    %vitb4_msrowb = stablehlo.broadcast_in_dim %vitb4_msrow, dims = [0, 1, 2] : (tensor<32x3x197xf32>) -> tensor<32x3x197x197xf32>
    %vitb4_mdiff = stablehlo.subtract %vitb4_mdW, %vitb4_msrowb : tensor<32x3x197x197xf32>
    %vitb4_mdSs = stablehlo.multiply %vitb4_mW, %vitb4_mdiff : tensor<32x3x197x197xf32>
    %vitb4_msclb = stablehlo.constant dense<0.125> : tensor<32x3x197x197xf32>
    %vitb4_mdS = stablehlo.multiply %vitb4_mdSs, %vitb4_msclb : tensor<32x3x197x197xf32>
    %vitb4_mdQh = stablehlo.dot_general %vitb4_mdS, %vitb4_mKh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb4_mdKh = stablehlo.dot_general %vitb4_mdS, %vitb4_mQh, batching_dims = [0, 1] x [0, 1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb4_mdQT = stablehlo.transpose %vitb4_mdQh, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb4_mdQ = stablehlo.reshape %vitb4_mdQT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb4_mdKT = stablehlo.transpose %vitb4_mdKh, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb4_mdK = stablehlo.reshape %vitb4_mdKT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb4_mdVT = stablehlo.transpose %vitb4_mdVh, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb4_mdV = stablehlo.reshape %vitb4_mdVT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb4_mdxQ = stablehlo.dot_general %vitb4_mdQ, %Wq_4, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb4_mdWQ = stablehlo.dot_general %vitb4_1y, %vitb4_mdQ, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb4_mdbQ = stablehlo.reduce(%vitb4_mdQ init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb4_mdxK = stablehlo.dot_general %vitb4_mdK, %Wk_4, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb4_mdWK = stablehlo.dot_general %vitb4_1y, %vitb4_mdK, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb4_mdbK = stablehlo.reduce(%vitb4_mdK init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb4_mdxV = stablehlo.dot_general %vitb4_mdV, %Wv_4, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb4_mdWV = stablehlo.dot_general %vitb4_1y, %vitb4_mdV, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb4_mdbV = stablehlo.reduce(%vitb4_mdV init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb4_mdxa = stablehlo.add %vitb4_mdxQ, %vitb4_mdxK : tensor<32x197x192xf32>
    %vitb4_mdx = stablehlo.add %vitb4_mdxa, %vitb4_mdxV : tensor<32x197x192xf32>
    %vitb4_1gbk = stablehlo.broadcast_in_dim %g1_4, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb4_1dxhat = stablehlo.multiply %vitb4_mdx, %vitb4_1gbk : tensor<32x197x192xf32>
    %vitb4_1dgpre = stablehlo.multiply %vitb4_mdx, %vitb4_1xhat : tensor<32x197x192xf32>
    %vitb4_1dg = stablehlo.reduce(%vitb4_1dgpre init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb4_1db = stablehlo.reduce(%vitb4_mdx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb4_1m1s = stablehlo.reduce(%vitb4_1dxhat init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb4_1m1 = stablehlo.divide %vitb4_1m1s, %vitb4_1nf : tensor<32x197xf32>
    %vitb4_1dxxh = stablehlo.multiply %vitb4_1dxhat, %vitb4_1xhat : tensor<32x197x192xf32>
    %vitb4_1m2s = stablehlo.reduce(%vitb4_1dxxh init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb4_1m2 = stablehlo.divide %vitb4_1m2s, %vitb4_1nf : tensor<32x197xf32>
    %vitb4_1m1b = stablehlo.broadcast_in_dim %vitb4_1m1, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb4_1m2b = stablehlo.broadcast_in_dim %vitb4_1m2, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb4_1t1 = stablehlo.subtract %vitb4_1dxhat, %vitb4_1m1b : tensor<32x197x192xf32>
    %vitb4_1xm2 = stablehlo.multiply %vitb4_1xhat, %vitb4_1m2b : tensor<32x197x192xf32>
    %vitb4_1t2 = stablehlo.subtract %vitb4_1t1, %vitb4_1xm2 : tensor<32x197x192xf32>
    %vitb4_1dx = stablehlo.multiply %vitb4_1istdb, %vitb4_1t2 : tensor<32x197x192xf32>
    %vitb4_dx = stablehlo.add %vitb4_dr1, %vitb4_1dx : tensor<32x197x192xf32>
    %vitb3_pda1 = stablehlo.dot_general %vitb4_dx, %Wfc2_3, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<768x192xf32>) -> tensor<32x197x768xf32>
    %vitb3_pdWfc2 = stablehlo.dot_general %vitb3_pga, %vitb4_dx, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<32x197x192xf32>) -> tensor<768x192xf32>
    %vitb3_pdbfc2 = stablehlo.reduce(%vitb4_dx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb3_pgbbx2 = stablehlo.multiply %vitb3_ph1, %vitb3_ph1 : tensor<32x197x768xf32>
    %vitb3_pgbbx3 = stablehlo.multiply %vitb3_pgbbx2, %vitb3_ph1 : tensor<32x197x768xf32>
    %vitb3_pgbbck = stablehlo.constant dense<0.044715> : tensor<32x197x768xf32>
    %vitb3_pgbbkx3 = stablehlo.multiply %vitb3_pgbbck, %vitb3_pgbbx3 : tensor<32x197x768xf32>
    %vitb3_pgbbinn = stablehlo.add %vitb3_ph1, %vitb3_pgbbkx3 : tensor<32x197x768xf32>
    %vitb3_pgbbcsqrt = stablehlo.constant dense<0.7978845608028654> : tensor<32x197x768xf32>
    %vitb3_pgbbu = stablehlo.multiply %vitb3_pgbbcsqrt, %vitb3_pgbbinn : tensor<32x197x768xf32>
    %vitb3_pgbbt = stablehlo.tanh %vitb3_pgbbu : tensor<32x197x768xf32>
    %vitb3_pgbbone = stablehlo.constant dense<1.0> : tensor<32x197x768xf32>
    %vitb3_pgbbopt = stablehlo.add %vitb3_pgbbone, %vitb3_pgbbt : tensor<32x197x768xf32>
    %vitb3_pgbbchalf = stablehlo.constant dense<0.5> : tensor<32x197x768xf32>
    %vitb3_pgbbterm1 = stablehlo.multiply %vitb3_pgbbchalf, %vitb3_pgbbopt : tensor<32x197x768xf32>
    %vitb3_pgbbt2 = stablehlo.multiply %vitb3_pgbbt, %vitb3_pgbbt : tensor<32x197x768xf32>
    %vitb3_pgbbomt2 = stablehlo.subtract %vitb3_pgbbone, %vitb3_pgbbt2 : tensor<32x197x768xf32>
    %vitb3_pgbbhx = stablehlo.multiply %vitb3_pgbbchalf, %vitb3_ph1 : tensor<32x197x768xf32>
    %vitb3_pgbbhxo = stablehlo.multiply %vitb3_pgbbhx, %vitb3_pgbbomt2 : tensor<32x197x768xf32>
    %vitb3_pgbbc3b = stablehlo.constant dense<0.134145> : tensor<32x197x768xf32>
    %vitb3_pgbba3x2 = stablehlo.multiply %vitb3_pgbbc3b, %vitb3_pgbbx2 : tensor<32x197x768xf32>
    %vitb3_pgbbin2 = stablehlo.add %vitb3_pgbbone, %vitb3_pgbba3x2 : tensor<32x197x768xf32>
    %vitb3_pgbbup = stablehlo.multiply %vitb3_pgbbcsqrt, %vitb3_pgbbin2 : tensor<32x197x768xf32>
    %vitb3_pgbbterm2 = stablehlo.multiply %vitb3_pgbbhxo, %vitb3_pgbbup : tensor<32x197x768xf32>
    %vitb3_pgbbgp = stablehlo.add %vitb3_pgbbterm1, %vitb3_pgbbterm2 : tensor<32x197x768xf32>
    %vitb3_pgbdx = stablehlo.multiply %vitb3_pda1, %vitb3_pgbbgp : tensor<32x197x768xf32>
    %vitb3_pdx = stablehlo.dot_general %vitb3_pgbdx, %Wfc1_3, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<192x768xf32>) -> tensor<32x197x192xf32>
    %vitb3_pdWfc1 = stablehlo.dot_general %vitb3_2y, %vitb3_pgbdx, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x768xf32>) -> tensor<192x768xf32>
    %vitb3_pdbfc1 = stablehlo.reduce(%vitb3_pgbdx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<768xf32>
    %vitb3_2gbk = stablehlo.broadcast_in_dim %g2_3, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb3_2dxhat = stablehlo.multiply %vitb3_pdx, %vitb3_2gbk : tensor<32x197x192xf32>
    %vitb3_2dgpre = stablehlo.multiply %vitb3_pdx, %vitb3_2xhat : tensor<32x197x192xf32>
    %vitb3_2dg = stablehlo.reduce(%vitb3_2dgpre init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb3_2db = stablehlo.reduce(%vitb3_pdx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb3_2m1s = stablehlo.reduce(%vitb3_2dxhat init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb3_2m1 = stablehlo.divide %vitb3_2m1s, %vitb3_2nf : tensor<32x197xf32>
    %vitb3_2dxxh = stablehlo.multiply %vitb3_2dxhat, %vitb3_2xhat : tensor<32x197x192xf32>
    %vitb3_2m2s = stablehlo.reduce(%vitb3_2dxxh init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb3_2m2 = stablehlo.divide %vitb3_2m2s, %vitb3_2nf : tensor<32x197xf32>
    %vitb3_2m1b = stablehlo.broadcast_in_dim %vitb3_2m1, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb3_2m2b = stablehlo.broadcast_in_dim %vitb3_2m2, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb3_2t1 = stablehlo.subtract %vitb3_2dxhat, %vitb3_2m1b : tensor<32x197x192xf32>
    %vitb3_2xm2 = stablehlo.multiply %vitb3_2xhat, %vitb3_2m2b : tensor<32x197x192xf32>
    %vitb3_2t2 = stablehlo.subtract %vitb3_2t1, %vitb3_2xm2 : tensor<32x197x192xf32>
    %vitb3_2dx = stablehlo.multiply %vitb3_2istdb, %vitb3_2t2 : tensor<32x197x192xf32>
    %vitb3_dr1 = stablehlo.add %vitb4_dx, %vitb3_2dx : tensor<32x197x192xf32>
    %vitb3_mdP = stablehlo.dot_general %vitb3_dr1, %Wo_3, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb3_mdWo = stablehlo.dot_general %vitb3_mP, %vitb3_dr1, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb3_mdbo = stablehlo.reduce(%vitb3_dr1 init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb3_mdPr = stablehlo.reshape %vitb3_mdP : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb3_mdA = stablehlo.transpose %vitb3_mdPr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb3_mdW = stablehlo.dot_general %vitb3_mdA, %vitb3_mVh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x64xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x197xf32>
    %vitb3_mdVh = stablehlo.dot_general %vitb3_mW, %vitb3_mdA, batching_dims = [0, 1] x [0, 1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb3_mpdw = stablehlo.multiply %vitb3_mW, %vitb3_mdW : tensor<32x3x197x197xf32>
    %vitb3_msrow = stablehlo.reduce(%vitb3_mpdw init: %sc) applies stablehlo.add across dimensions = [3] : (tensor<32x3x197x197xf32>, tensor<f32>) -> tensor<32x3x197xf32>
    %vitb3_msrowb = stablehlo.broadcast_in_dim %vitb3_msrow, dims = [0, 1, 2] : (tensor<32x3x197xf32>) -> tensor<32x3x197x197xf32>
    %vitb3_mdiff = stablehlo.subtract %vitb3_mdW, %vitb3_msrowb : tensor<32x3x197x197xf32>
    %vitb3_mdSs = stablehlo.multiply %vitb3_mW, %vitb3_mdiff : tensor<32x3x197x197xf32>
    %vitb3_msclb = stablehlo.constant dense<0.125> : tensor<32x3x197x197xf32>
    %vitb3_mdS = stablehlo.multiply %vitb3_mdSs, %vitb3_msclb : tensor<32x3x197x197xf32>
    %vitb3_mdQh = stablehlo.dot_general %vitb3_mdS, %vitb3_mKh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb3_mdKh = stablehlo.dot_general %vitb3_mdS, %vitb3_mQh, batching_dims = [0, 1] x [0, 1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb3_mdQT = stablehlo.transpose %vitb3_mdQh, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb3_mdQ = stablehlo.reshape %vitb3_mdQT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb3_mdKT = stablehlo.transpose %vitb3_mdKh, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb3_mdK = stablehlo.reshape %vitb3_mdKT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb3_mdVT = stablehlo.transpose %vitb3_mdVh, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb3_mdV = stablehlo.reshape %vitb3_mdVT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb3_mdxQ = stablehlo.dot_general %vitb3_mdQ, %Wq_3, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb3_mdWQ = stablehlo.dot_general %vitb3_1y, %vitb3_mdQ, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb3_mdbQ = stablehlo.reduce(%vitb3_mdQ init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb3_mdxK = stablehlo.dot_general %vitb3_mdK, %Wk_3, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb3_mdWK = stablehlo.dot_general %vitb3_1y, %vitb3_mdK, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb3_mdbK = stablehlo.reduce(%vitb3_mdK init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb3_mdxV = stablehlo.dot_general %vitb3_mdV, %Wv_3, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb3_mdWV = stablehlo.dot_general %vitb3_1y, %vitb3_mdV, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb3_mdbV = stablehlo.reduce(%vitb3_mdV init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb3_mdxa = stablehlo.add %vitb3_mdxQ, %vitb3_mdxK : tensor<32x197x192xf32>
    %vitb3_mdx = stablehlo.add %vitb3_mdxa, %vitb3_mdxV : tensor<32x197x192xf32>
    %vitb3_1gbk = stablehlo.broadcast_in_dim %g1_3, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb3_1dxhat = stablehlo.multiply %vitb3_mdx, %vitb3_1gbk : tensor<32x197x192xf32>
    %vitb3_1dgpre = stablehlo.multiply %vitb3_mdx, %vitb3_1xhat : tensor<32x197x192xf32>
    %vitb3_1dg = stablehlo.reduce(%vitb3_1dgpre init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb3_1db = stablehlo.reduce(%vitb3_mdx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb3_1m1s = stablehlo.reduce(%vitb3_1dxhat init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb3_1m1 = stablehlo.divide %vitb3_1m1s, %vitb3_1nf : tensor<32x197xf32>
    %vitb3_1dxxh = stablehlo.multiply %vitb3_1dxhat, %vitb3_1xhat : tensor<32x197x192xf32>
    %vitb3_1m2s = stablehlo.reduce(%vitb3_1dxxh init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb3_1m2 = stablehlo.divide %vitb3_1m2s, %vitb3_1nf : tensor<32x197xf32>
    %vitb3_1m1b = stablehlo.broadcast_in_dim %vitb3_1m1, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb3_1m2b = stablehlo.broadcast_in_dim %vitb3_1m2, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb3_1t1 = stablehlo.subtract %vitb3_1dxhat, %vitb3_1m1b : tensor<32x197x192xf32>
    %vitb3_1xm2 = stablehlo.multiply %vitb3_1xhat, %vitb3_1m2b : tensor<32x197x192xf32>
    %vitb3_1t2 = stablehlo.subtract %vitb3_1t1, %vitb3_1xm2 : tensor<32x197x192xf32>
    %vitb3_1dx = stablehlo.multiply %vitb3_1istdb, %vitb3_1t2 : tensor<32x197x192xf32>
    %vitb3_dx = stablehlo.add %vitb3_dr1, %vitb3_1dx : tensor<32x197x192xf32>
    %vitb2_pda1 = stablehlo.dot_general %vitb3_dx, %Wfc2_2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<768x192xf32>) -> tensor<32x197x768xf32>
    %vitb2_pdWfc2 = stablehlo.dot_general %vitb2_pga, %vitb3_dx, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<32x197x192xf32>) -> tensor<768x192xf32>
    %vitb2_pdbfc2 = stablehlo.reduce(%vitb3_dx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb2_pgbbx2 = stablehlo.multiply %vitb2_ph1, %vitb2_ph1 : tensor<32x197x768xf32>
    %vitb2_pgbbx3 = stablehlo.multiply %vitb2_pgbbx2, %vitb2_ph1 : tensor<32x197x768xf32>
    %vitb2_pgbbck = stablehlo.constant dense<0.044715> : tensor<32x197x768xf32>
    %vitb2_pgbbkx3 = stablehlo.multiply %vitb2_pgbbck, %vitb2_pgbbx3 : tensor<32x197x768xf32>
    %vitb2_pgbbinn = stablehlo.add %vitb2_ph1, %vitb2_pgbbkx3 : tensor<32x197x768xf32>
    %vitb2_pgbbcsqrt = stablehlo.constant dense<0.7978845608028654> : tensor<32x197x768xf32>
    %vitb2_pgbbu = stablehlo.multiply %vitb2_pgbbcsqrt, %vitb2_pgbbinn : tensor<32x197x768xf32>
    %vitb2_pgbbt = stablehlo.tanh %vitb2_pgbbu : tensor<32x197x768xf32>
    %vitb2_pgbbone = stablehlo.constant dense<1.0> : tensor<32x197x768xf32>
    %vitb2_pgbbopt = stablehlo.add %vitb2_pgbbone, %vitb2_pgbbt : tensor<32x197x768xf32>
    %vitb2_pgbbchalf = stablehlo.constant dense<0.5> : tensor<32x197x768xf32>
    %vitb2_pgbbterm1 = stablehlo.multiply %vitb2_pgbbchalf, %vitb2_pgbbopt : tensor<32x197x768xf32>
    %vitb2_pgbbt2 = stablehlo.multiply %vitb2_pgbbt, %vitb2_pgbbt : tensor<32x197x768xf32>
    %vitb2_pgbbomt2 = stablehlo.subtract %vitb2_pgbbone, %vitb2_pgbbt2 : tensor<32x197x768xf32>
    %vitb2_pgbbhx = stablehlo.multiply %vitb2_pgbbchalf, %vitb2_ph1 : tensor<32x197x768xf32>
    %vitb2_pgbbhxo = stablehlo.multiply %vitb2_pgbbhx, %vitb2_pgbbomt2 : tensor<32x197x768xf32>
    %vitb2_pgbbc3b = stablehlo.constant dense<0.134145> : tensor<32x197x768xf32>
    %vitb2_pgbba3x2 = stablehlo.multiply %vitb2_pgbbc3b, %vitb2_pgbbx2 : tensor<32x197x768xf32>
    %vitb2_pgbbin2 = stablehlo.add %vitb2_pgbbone, %vitb2_pgbba3x2 : tensor<32x197x768xf32>
    %vitb2_pgbbup = stablehlo.multiply %vitb2_pgbbcsqrt, %vitb2_pgbbin2 : tensor<32x197x768xf32>
    %vitb2_pgbbterm2 = stablehlo.multiply %vitb2_pgbbhxo, %vitb2_pgbbup : tensor<32x197x768xf32>
    %vitb2_pgbbgp = stablehlo.add %vitb2_pgbbterm1, %vitb2_pgbbterm2 : tensor<32x197x768xf32>
    %vitb2_pgbdx = stablehlo.multiply %vitb2_pda1, %vitb2_pgbbgp : tensor<32x197x768xf32>
    %vitb2_pdx = stablehlo.dot_general %vitb2_pgbdx, %Wfc1_2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<192x768xf32>) -> tensor<32x197x192xf32>
    %vitb2_pdWfc1 = stablehlo.dot_general %vitb2_2y, %vitb2_pgbdx, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x768xf32>) -> tensor<192x768xf32>
    %vitb2_pdbfc1 = stablehlo.reduce(%vitb2_pgbdx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<768xf32>
    %vitb2_2gbk = stablehlo.broadcast_in_dim %g2_2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb2_2dxhat = stablehlo.multiply %vitb2_pdx, %vitb2_2gbk : tensor<32x197x192xf32>
    %vitb2_2dgpre = stablehlo.multiply %vitb2_pdx, %vitb2_2xhat : tensor<32x197x192xf32>
    %vitb2_2dg = stablehlo.reduce(%vitb2_2dgpre init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb2_2db = stablehlo.reduce(%vitb2_pdx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb2_2m1s = stablehlo.reduce(%vitb2_2dxhat init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb2_2m1 = stablehlo.divide %vitb2_2m1s, %vitb2_2nf : tensor<32x197xf32>
    %vitb2_2dxxh = stablehlo.multiply %vitb2_2dxhat, %vitb2_2xhat : tensor<32x197x192xf32>
    %vitb2_2m2s = stablehlo.reduce(%vitb2_2dxxh init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb2_2m2 = stablehlo.divide %vitb2_2m2s, %vitb2_2nf : tensor<32x197xf32>
    %vitb2_2m1b = stablehlo.broadcast_in_dim %vitb2_2m1, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb2_2m2b = stablehlo.broadcast_in_dim %vitb2_2m2, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb2_2t1 = stablehlo.subtract %vitb2_2dxhat, %vitb2_2m1b : tensor<32x197x192xf32>
    %vitb2_2xm2 = stablehlo.multiply %vitb2_2xhat, %vitb2_2m2b : tensor<32x197x192xf32>
    %vitb2_2t2 = stablehlo.subtract %vitb2_2t1, %vitb2_2xm2 : tensor<32x197x192xf32>
    %vitb2_2dx = stablehlo.multiply %vitb2_2istdb, %vitb2_2t2 : tensor<32x197x192xf32>
    %vitb2_dr1 = stablehlo.add %vitb3_dx, %vitb2_2dx : tensor<32x197x192xf32>
    %vitb2_mdP = stablehlo.dot_general %vitb2_dr1, %Wo_2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb2_mdWo = stablehlo.dot_general %vitb2_mP, %vitb2_dr1, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb2_mdbo = stablehlo.reduce(%vitb2_dr1 init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb2_mdPr = stablehlo.reshape %vitb2_mdP : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb2_mdA = stablehlo.transpose %vitb2_mdPr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb2_mdW = stablehlo.dot_general %vitb2_mdA, %vitb2_mVh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x64xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x197xf32>
    %vitb2_mdVh = stablehlo.dot_general %vitb2_mW, %vitb2_mdA, batching_dims = [0, 1] x [0, 1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb2_mpdw = stablehlo.multiply %vitb2_mW, %vitb2_mdW : tensor<32x3x197x197xf32>
    %vitb2_msrow = stablehlo.reduce(%vitb2_mpdw init: %sc) applies stablehlo.add across dimensions = [3] : (tensor<32x3x197x197xf32>, tensor<f32>) -> tensor<32x3x197xf32>
    %vitb2_msrowb = stablehlo.broadcast_in_dim %vitb2_msrow, dims = [0, 1, 2] : (tensor<32x3x197xf32>) -> tensor<32x3x197x197xf32>
    %vitb2_mdiff = stablehlo.subtract %vitb2_mdW, %vitb2_msrowb : tensor<32x3x197x197xf32>
    %vitb2_mdSs = stablehlo.multiply %vitb2_mW, %vitb2_mdiff : tensor<32x3x197x197xf32>
    %vitb2_msclb = stablehlo.constant dense<0.125> : tensor<32x3x197x197xf32>
    %vitb2_mdS = stablehlo.multiply %vitb2_mdSs, %vitb2_msclb : tensor<32x3x197x197xf32>
    %vitb2_mdQh = stablehlo.dot_general %vitb2_mdS, %vitb2_mKh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb2_mdKh = stablehlo.dot_general %vitb2_mdS, %vitb2_mQh, batching_dims = [0, 1] x [0, 1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb2_mdQT = stablehlo.transpose %vitb2_mdQh, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb2_mdQ = stablehlo.reshape %vitb2_mdQT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb2_mdKT = stablehlo.transpose %vitb2_mdKh, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb2_mdK = stablehlo.reshape %vitb2_mdKT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb2_mdVT = stablehlo.transpose %vitb2_mdVh, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb2_mdV = stablehlo.reshape %vitb2_mdVT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb2_mdxQ = stablehlo.dot_general %vitb2_mdQ, %Wq_2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb2_mdWQ = stablehlo.dot_general %vitb2_1y, %vitb2_mdQ, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb2_mdbQ = stablehlo.reduce(%vitb2_mdQ init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb2_mdxK = stablehlo.dot_general %vitb2_mdK, %Wk_2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb2_mdWK = stablehlo.dot_general %vitb2_1y, %vitb2_mdK, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb2_mdbK = stablehlo.reduce(%vitb2_mdK init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb2_mdxV = stablehlo.dot_general %vitb2_mdV, %Wv_2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb2_mdWV = stablehlo.dot_general %vitb2_1y, %vitb2_mdV, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb2_mdbV = stablehlo.reduce(%vitb2_mdV init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb2_mdxa = stablehlo.add %vitb2_mdxQ, %vitb2_mdxK : tensor<32x197x192xf32>
    %vitb2_mdx = stablehlo.add %vitb2_mdxa, %vitb2_mdxV : tensor<32x197x192xf32>
    %vitb2_1gbk = stablehlo.broadcast_in_dim %g1_2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb2_1dxhat = stablehlo.multiply %vitb2_mdx, %vitb2_1gbk : tensor<32x197x192xf32>
    %vitb2_1dgpre = stablehlo.multiply %vitb2_mdx, %vitb2_1xhat : tensor<32x197x192xf32>
    %vitb2_1dg = stablehlo.reduce(%vitb2_1dgpre init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb2_1db = stablehlo.reduce(%vitb2_mdx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb2_1m1s = stablehlo.reduce(%vitb2_1dxhat init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb2_1m1 = stablehlo.divide %vitb2_1m1s, %vitb2_1nf : tensor<32x197xf32>
    %vitb2_1dxxh = stablehlo.multiply %vitb2_1dxhat, %vitb2_1xhat : tensor<32x197x192xf32>
    %vitb2_1m2s = stablehlo.reduce(%vitb2_1dxxh init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb2_1m2 = stablehlo.divide %vitb2_1m2s, %vitb2_1nf : tensor<32x197xf32>
    %vitb2_1m1b = stablehlo.broadcast_in_dim %vitb2_1m1, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb2_1m2b = stablehlo.broadcast_in_dim %vitb2_1m2, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb2_1t1 = stablehlo.subtract %vitb2_1dxhat, %vitb2_1m1b : tensor<32x197x192xf32>
    %vitb2_1xm2 = stablehlo.multiply %vitb2_1xhat, %vitb2_1m2b : tensor<32x197x192xf32>
    %vitb2_1t2 = stablehlo.subtract %vitb2_1t1, %vitb2_1xm2 : tensor<32x197x192xf32>
    %vitb2_1dx = stablehlo.multiply %vitb2_1istdb, %vitb2_1t2 : tensor<32x197x192xf32>
    %vitb2_dx = stablehlo.add %vitb2_dr1, %vitb2_1dx : tensor<32x197x192xf32>
    %vitb1_pda1 = stablehlo.dot_general %vitb2_dx, %Wfc2_1, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<768x192xf32>) -> tensor<32x197x768xf32>
    %vitb1_pdWfc2 = stablehlo.dot_general %vitb1_pga, %vitb2_dx, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<32x197x192xf32>) -> tensor<768x192xf32>
    %vitb1_pdbfc2 = stablehlo.reduce(%vitb2_dx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb1_pgbbx2 = stablehlo.multiply %vitb1_ph1, %vitb1_ph1 : tensor<32x197x768xf32>
    %vitb1_pgbbx3 = stablehlo.multiply %vitb1_pgbbx2, %vitb1_ph1 : tensor<32x197x768xf32>
    %vitb1_pgbbck = stablehlo.constant dense<0.044715> : tensor<32x197x768xf32>
    %vitb1_pgbbkx3 = stablehlo.multiply %vitb1_pgbbck, %vitb1_pgbbx3 : tensor<32x197x768xf32>
    %vitb1_pgbbinn = stablehlo.add %vitb1_ph1, %vitb1_pgbbkx3 : tensor<32x197x768xf32>
    %vitb1_pgbbcsqrt = stablehlo.constant dense<0.7978845608028654> : tensor<32x197x768xf32>
    %vitb1_pgbbu = stablehlo.multiply %vitb1_pgbbcsqrt, %vitb1_pgbbinn : tensor<32x197x768xf32>
    %vitb1_pgbbt = stablehlo.tanh %vitb1_pgbbu : tensor<32x197x768xf32>
    %vitb1_pgbbone = stablehlo.constant dense<1.0> : tensor<32x197x768xf32>
    %vitb1_pgbbopt = stablehlo.add %vitb1_pgbbone, %vitb1_pgbbt : tensor<32x197x768xf32>
    %vitb1_pgbbchalf = stablehlo.constant dense<0.5> : tensor<32x197x768xf32>
    %vitb1_pgbbterm1 = stablehlo.multiply %vitb1_pgbbchalf, %vitb1_pgbbopt : tensor<32x197x768xf32>
    %vitb1_pgbbt2 = stablehlo.multiply %vitb1_pgbbt, %vitb1_pgbbt : tensor<32x197x768xf32>
    %vitb1_pgbbomt2 = stablehlo.subtract %vitb1_pgbbone, %vitb1_pgbbt2 : tensor<32x197x768xf32>
    %vitb1_pgbbhx = stablehlo.multiply %vitb1_pgbbchalf, %vitb1_ph1 : tensor<32x197x768xf32>
    %vitb1_pgbbhxo = stablehlo.multiply %vitb1_pgbbhx, %vitb1_pgbbomt2 : tensor<32x197x768xf32>
    %vitb1_pgbbc3b = stablehlo.constant dense<0.134145> : tensor<32x197x768xf32>
    %vitb1_pgbba3x2 = stablehlo.multiply %vitb1_pgbbc3b, %vitb1_pgbbx2 : tensor<32x197x768xf32>
    %vitb1_pgbbin2 = stablehlo.add %vitb1_pgbbone, %vitb1_pgbba3x2 : tensor<32x197x768xf32>
    %vitb1_pgbbup = stablehlo.multiply %vitb1_pgbbcsqrt, %vitb1_pgbbin2 : tensor<32x197x768xf32>
    %vitb1_pgbbterm2 = stablehlo.multiply %vitb1_pgbbhxo, %vitb1_pgbbup : tensor<32x197x768xf32>
    %vitb1_pgbbgp = stablehlo.add %vitb1_pgbbterm1, %vitb1_pgbbterm2 : tensor<32x197x768xf32>
    %vitb1_pgbdx = stablehlo.multiply %vitb1_pda1, %vitb1_pgbbgp : tensor<32x197x768xf32>
    %vitb1_pdx = stablehlo.dot_general %vitb1_pgbdx, %Wfc1_1, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<192x768xf32>) -> tensor<32x197x192xf32>
    %vitb1_pdWfc1 = stablehlo.dot_general %vitb1_2y, %vitb1_pgbdx, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x768xf32>) -> tensor<192x768xf32>
    %vitb1_pdbfc1 = stablehlo.reduce(%vitb1_pgbdx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<768xf32>
    %vitb1_2gbk = stablehlo.broadcast_in_dim %g2_1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb1_2dxhat = stablehlo.multiply %vitb1_pdx, %vitb1_2gbk : tensor<32x197x192xf32>
    %vitb1_2dgpre = stablehlo.multiply %vitb1_pdx, %vitb1_2xhat : tensor<32x197x192xf32>
    %vitb1_2dg = stablehlo.reduce(%vitb1_2dgpre init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb1_2db = stablehlo.reduce(%vitb1_pdx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb1_2m1s = stablehlo.reduce(%vitb1_2dxhat init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb1_2m1 = stablehlo.divide %vitb1_2m1s, %vitb1_2nf : tensor<32x197xf32>
    %vitb1_2dxxh = stablehlo.multiply %vitb1_2dxhat, %vitb1_2xhat : tensor<32x197x192xf32>
    %vitb1_2m2s = stablehlo.reduce(%vitb1_2dxxh init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb1_2m2 = stablehlo.divide %vitb1_2m2s, %vitb1_2nf : tensor<32x197xf32>
    %vitb1_2m1b = stablehlo.broadcast_in_dim %vitb1_2m1, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb1_2m2b = stablehlo.broadcast_in_dim %vitb1_2m2, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb1_2t1 = stablehlo.subtract %vitb1_2dxhat, %vitb1_2m1b : tensor<32x197x192xf32>
    %vitb1_2xm2 = stablehlo.multiply %vitb1_2xhat, %vitb1_2m2b : tensor<32x197x192xf32>
    %vitb1_2t2 = stablehlo.subtract %vitb1_2t1, %vitb1_2xm2 : tensor<32x197x192xf32>
    %vitb1_2dx = stablehlo.multiply %vitb1_2istdb, %vitb1_2t2 : tensor<32x197x192xf32>
    %vitb1_dr1 = stablehlo.add %vitb2_dx, %vitb1_2dx : tensor<32x197x192xf32>
    %vitb1_mdP = stablehlo.dot_general %vitb1_dr1, %Wo_1, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb1_mdWo = stablehlo.dot_general %vitb1_mP, %vitb1_dr1, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb1_mdbo = stablehlo.reduce(%vitb1_dr1 init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb1_mdPr = stablehlo.reshape %vitb1_mdP : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb1_mdA = stablehlo.transpose %vitb1_mdPr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb1_mdW = stablehlo.dot_general %vitb1_mdA, %vitb1_mVh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x64xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x197xf32>
    %vitb1_mdVh = stablehlo.dot_general %vitb1_mW, %vitb1_mdA, batching_dims = [0, 1] x [0, 1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb1_mpdw = stablehlo.multiply %vitb1_mW, %vitb1_mdW : tensor<32x3x197x197xf32>
    %vitb1_msrow = stablehlo.reduce(%vitb1_mpdw init: %sc) applies stablehlo.add across dimensions = [3] : (tensor<32x3x197x197xf32>, tensor<f32>) -> tensor<32x3x197xf32>
    %vitb1_msrowb = stablehlo.broadcast_in_dim %vitb1_msrow, dims = [0, 1, 2] : (tensor<32x3x197xf32>) -> tensor<32x3x197x197xf32>
    %vitb1_mdiff = stablehlo.subtract %vitb1_mdW, %vitb1_msrowb : tensor<32x3x197x197xf32>
    %vitb1_mdSs = stablehlo.multiply %vitb1_mW, %vitb1_mdiff : tensor<32x3x197x197xf32>
    %vitb1_msclb = stablehlo.constant dense<0.125> : tensor<32x3x197x197xf32>
    %vitb1_mdS = stablehlo.multiply %vitb1_mdSs, %vitb1_msclb : tensor<32x3x197x197xf32>
    %vitb1_mdQh = stablehlo.dot_general %vitb1_mdS, %vitb1_mKh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb1_mdKh = stablehlo.dot_general %vitb1_mdS, %vitb1_mQh, batching_dims = [0, 1] x [0, 1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb1_mdQT = stablehlo.transpose %vitb1_mdQh, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb1_mdQ = stablehlo.reshape %vitb1_mdQT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb1_mdKT = stablehlo.transpose %vitb1_mdKh, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb1_mdK = stablehlo.reshape %vitb1_mdKT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb1_mdVT = stablehlo.transpose %vitb1_mdVh, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb1_mdV = stablehlo.reshape %vitb1_mdVT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb1_mdxQ = stablehlo.dot_general %vitb1_mdQ, %Wq_1, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb1_mdWQ = stablehlo.dot_general %vitb1_1y, %vitb1_mdQ, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb1_mdbQ = stablehlo.reduce(%vitb1_mdQ init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb1_mdxK = stablehlo.dot_general %vitb1_mdK, %Wk_1, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb1_mdWK = stablehlo.dot_general %vitb1_1y, %vitb1_mdK, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb1_mdbK = stablehlo.reduce(%vitb1_mdK init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb1_mdxV = stablehlo.dot_general %vitb1_mdV, %Wv_1, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb1_mdWV = stablehlo.dot_general %vitb1_1y, %vitb1_mdV, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb1_mdbV = stablehlo.reduce(%vitb1_mdV init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb1_mdxa = stablehlo.add %vitb1_mdxQ, %vitb1_mdxK : tensor<32x197x192xf32>
    %vitb1_mdx = stablehlo.add %vitb1_mdxa, %vitb1_mdxV : tensor<32x197x192xf32>
    %vitb1_1gbk = stablehlo.broadcast_in_dim %g1_1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb1_1dxhat = stablehlo.multiply %vitb1_mdx, %vitb1_1gbk : tensor<32x197x192xf32>
    %vitb1_1dgpre = stablehlo.multiply %vitb1_mdx, %vitb1_1xhat : tensor<32x197x192xf32>
    %vitb1_1dg = stablehlo.reduce(%vitb1_1dgpre init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb1_1db = stablehlo.reduce(%vitb1_mdx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb1_1m1s = stablehlo.reduce(%vitb1_1dxhat init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb1_1m1 = stablehlo.divide %vitb1_1m1s, %vitb1_1nf : tensor<32x197xf32>
    %vitb1_1dxxh = stablehlo.multiply %vitb1_1dxhat, %vitb1_1xhat : tensor<32x197x192xf32>
    %vitb1_1m2s = stablehlo.reduce(%vitb1_1dxxh init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb1_1m2 = stablehlo.divide %vitb1_1m2s, %vitb1_1nf : tensor<32x197xf32>
    %vitb1_1m1b = stablehlo.broadcast_in_dim %vitb1_1m1, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb1_1m2b = stablehlo.broadcast_in_dim %vitb1_1m2, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb1_1t1 = stablehlo.subtract %vitb1_1dxhat, %vitb1_1m1b : tensor<32x197x192xf32>
    %vitb1_1xm2 = stablehlo.multiply %vitb1_1xhat, %vitb1_1m2b : tensor<32x197x192xf32>
    %vitb1_1t2 = stablehlo.subtract %vitb1_1t1, %vitb1_1xm2 : tensor<32x197x192xf32>
    %vitb1_1dx = stablehlo.multiply %vitb1_1istdb, %vitb1_1t2 : tensor<32x197x192xf32>
    %vitb1_dx = stablehlo.add %vitb1_dr1, %vitb1_1dx : tensor<32x197x192xf32>
    %vitb0_pda1 = stablehlo.dot_general %vitb1_dx, %Wfc2_0, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<768x192xf32>) -> tensor<32x197x768xf32>
    %vitb0_pdWfc2 = stablehlo.dot_general %vitb0_pga, %vitb1_dx, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<32x197x192xf32>) -> tensor<768x192xf32>
    %vitb0_pdbfc2 = stablehlo.reduce(%vitb1_dx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb0_pgbbx2 = stablehlo.multiply %vitb0_ph1, %vitb0_ph1 : tensor<32x197x768xf32>
    %vitb0_pgbbx3 = stablehlo.multiply %vitb0_pgbbx2, %vitb0_ph1 : tensor<32x197x768xf32>
    %vitb0_pgbbck = stablehlo.constant dense<0.044715> : tensor<32x197x768xf32>
    %vitb0_pgbbkx3 = stablehlo.multiply %vitb0_pgbbck, %vitb0_pgbbx3 : tensor<32x197x768xf32>
    %vitb0_pgbbinn = stablehlo.add %vitb0_ph1, %vitb0_pgbbkx3 : tensor<32x197x768xf32>
    %vitb0_pgbbcsqrt = stablehlo.constant dense<0.7978845608028654> : tensor<32x197x768xf32>
    %vitb0_pgbbu = stablehlo.multiply %vitb0_pgbbcsqrt, %vitb0_pgbbinn : tensor<32x197x768xf32>
    %vitb0_pgbbt = stablehlo.tanh %vitb0_pgbbu : tensor<32x197x768xf32>
    %vitb0_pgbbone = stablehlo.constant dense<1.0> : tensor<32x197x768xf32>
    %vitb0_pgbbopt = stablehlo.add %vitb0_pgbbone, %vitb0_pgbbt : tensor<32x197x768xf32>
    %vitb0_pgbbchalf = stablehlo.constant dense<0.5> : tensor<32x197x768xf32>
    %vitb0_pgbbterm1 = stablehlo.multiply %vitb0_pgbbchalf, %vitb0_pgbbopt : tensor<32x197x768xf32>
    %vitb0_pgbbt2 = stablehlo.multiply %vitb0_pgbbt, %vitb0_pgbbt : tensor<32x197x768xf32>
    %vitb0_pgbbomt2 = stablehlo.subtract %vitb0_pgbbone, %vitb0_pgbbt2 : tensor<32x197x768xf32>
    %vitb0_pgbbhx = stablehlo.multiply %vitb0_pgbbchalf, %vitb0_ph1 : tensor<32x197x768xf32>
    %vitb0_pgbbhxo = stablehlo.multiply %vitb0_pgbbhx, %vitb0_pgbbomt2 : tensor<32x197x768xf32>
    %vitb0_pgbbc3b = stablehlo.constant dense<0.134145> : tensor<32x197x768xf32>
    %vitb0_pgbba3x2 = stablehlo.multiply %vitb0_pgbbc3b, %vitb0_pgbbx2 : tensor<32x197x768xf32>
    %vitb0_pgbbin2 = stablehlo.add %vitb0_pgbbone, %vitb0_pgbba3x2 : tensor<32x197x768xf32>
    %vitb0_pgbbup = stablehlo.multiply %vitb0_pgbbcsqrt, %vitb0_pgbbin2 : tensor<32x197x768xf32>
    %vitb0_pgbbterm2 = stablehlo.multiply %vitb0_pgbbhxo, %vitb0_pgbbup : tensor<32x197x768xf32>
    %vitb0_pgbbgp = stablehlo.add %vitb0_pgbbterm1, %vitb0_pgbbterm2 : tensor<32x197x768xf32>
    %vitb0_pgbdx = stablehlo.multiply %vitb0_pda1, %vitb0_pgbbgp : tensor<32x197x768xf32>
    %vitb0_pdx = stablehlo.dot_general %vitb0_pgbdx, %Wfc1_0, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<192x768xf32>) -> tensor<32x197x192xf32>
    %vitb0_pdWfc1 = stablehlo.dot_general %vitb0_2y, %vitb0_pgbdx, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x768xf32>) -> tensor<192x768xf32>
    %vitb0_pdbfc1 = stablehlo.reduce(%vitb0_pgbdx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<768xf32>
    %vitb0_2gbk = stablehlo.broadcast_in_dim %g2_0, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb0_2dxhat = stablehlo.multiply %vitb0_pdx, %vitb0_2gbk : tensor<32x197x192xf32>
    %vitb0_2dgpre = stablehlo.multiply %vitb0_pdx, %vitb0_2xhat : tensor<32x197x192xf32>
    %vitb0_2dg = stablehlo.reduce(%vitb0_2dgpre init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb0_2db = stablehlo.reduce(%vitb0_pdx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb0_2m1s = stablehlo.reduce(%vitb0_2dxhat init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb0_2m1 = stablehlo.divide %vitb0_2m1s, %vitb0_2nf : tensor<32x197xf32>
    %vitb0_2dxxh = stablehlo.multiply %vitb0_2dxhat, %vitb0_2xhat : tensor<32x197x192xf32>
    %vitb0_2m2s = stablehlo.reduce(%vitb0_2dxxh init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb0_2m2 = stablehlo.divide %vitb0_2m2s, %vitb0_2nf : tensor<32x197xf32>
    %vitb0_2m1b = stablehlo.broadcast_in_dim %vitb0_2m1, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb0_2m2b = stablehlo.broadcast_in_dim %vitb0_2m2, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb0_2t1 = stablehlo.subtract %vitb0_2dxhat, %vitb0_2m1b : tensor<32x197x192xf32>
    %vitb0_2xm2 = stablehlo.multiply %vitb0_2xhat, %vitb0_2m2b : tensor<32x197x192xf32>
    %vitb0_2t2 = stablehlo.subtract %vitb0_2t1, %vitb0_2xm2 : tensor<32x197x192xf32>
    %vitb0_2dx = stablehlo.multiply %vitb0_2istdb, %vitb0_2t2 : tensor<32x197x192xf32>
    %vitb0_dr1 = stablehlo.add %vitb1_dx, %vitb0_2dx : tensor<32x197x192xf32>
    %vitb0_mdP = stablehlo.dot_general %vitb0_dr1, %Wo_0, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb0_mdWo = stablehlo.dot_general %vitb0_mP, %vitb0_dr1, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb0_mdbo = stablehlo.reduce(%vitb0_dr1 init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb0_mdPr = stablehlo.reshape %vitb0_mdP : (tensor<32x197x192xf32>) -> tensor<32x197x3x64xf32>
    %vitb0_mdA = stablehlo.transpose %vitb0_mdPr, dims = [0, 2, 1, 3] : (tensor<32x197x3x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb0_mdW = stablehlo.dot_general %vitb0_mdA, %vitb0_mVh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x64xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x197xf32>
    %vitb0_mdVh = stablehlo.dot_general %vitb0_mW, %vitb0_mdA, batching_dims = [0, 1] x [0, 1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb0_mpdw = stablehlo.multiply %vitb0_mW, %vitb0_mdW : tensor<32x3x197x197xf32>
    %vitb0_msrow = stablehlo.reduce(%vitb0_mpdw init: %sc) applies stablehlo.add across dimensions = [3] : (tensor<32x3x197x197xf32>, tensor<f32>) -> tensor<32x3x197xf32>
    %vitb0_msrowb = stablehlo.broadcast_in_dim %vitb0_msrow, dims = [0, 1, 2] : (tensor<32x3x197xf32>) -> tensor<32x3x197x197xf32>
    %vitb0_mdiff = stablehlo.subtract %vitb0_mdW, %vitb0_msrowb : tensor<32x3x197x197xf32>
    %vitb0_mdSs = stablehlo.multiply %vitb0_mW, %vitb0_mdiff : tensor<32x3x197x197xf32>
    %vitb0_msclb = stablehlo.constant dense<0.125> : tensor<32x3x197x197xf32>
    %vitb0_mdS = stablehlo.multiply %vitb0_mdSs, %vitb0_msclb : tensor<32x3x197x197xf32>
    %vitb0_mdQh = stablehlo.dot_general %vitb0_mdS, %vitb0_mKh, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb0_mdKh = stablehlo.dot_general %vitb0_mdS, %vitb0_mQh, batching_dims = [0, 1] x [0, 1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x3x197x197xf32>, tensor<32x3x197x64xf32>) -> tensor<32x3x197x64xf32>
    %vitb0_mdQT = stablehlo.transpose %vitb0_mdQh, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb0_mdQ = stablehlo.reshape %vitb0_mdQT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb0_mdKT = stablehlo.transpose %vitb0_mdKh, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb0_mdK = stablehlo.reshape %vitb0_mdKT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb0_mdVT = stablehlo.transpose %vitb0_mdVh, dims = [0, 2, 1, 3] : (tensor<32x3x197x64xf32>) -> tensor<32x197x3x64xf32>
    %vitb0_mdV = stablehlo.reshape %vitb0_mdVT : (tensor<32x197x3x64xf32>) -> tensor<32x197x192xf32>
    %vitb0_mdxQ = stablehlo.dot_general %vitb0_mdQ, %Wq_0, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb0_mdWQ = stablehlo.dot_general %vitb0_1y, %vitb0_mdQ, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb0_mdbQ = stablehlo.reduce(%vitb0_mdQ init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb0_mdxK = stablehlo.dot_general %vitb0_mdK, %Wk_0, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb0_mdWK = stablehlo.dot_general %vitb0_1y, %vitb0_mdK, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb0_mdbK = stablehlo.reduce(%vitb0_mdK init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb0_mdxV = stablehlo.dot_general %vitb0_mdV, %Wv_0, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %vitb0_mdWV = stablehlo.dot_general %vitb0_1y, %vitb0_mdV, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %vitb0_mdbV = stablehlo.reduce(%vitb0_mdV init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb0_mdxa = stablehlo.add %vitb0_mdxQ, %vitb0_mdxK : tensor<32x197x192xf32>
    %vitb0_mdx = stablehlo.add %vitb0_mdxa, %vitb0_mdxV : tensor<32x197x192xf32>
    %vitb0_1gbk = stablehlo.broadcast_in_dim %g1_0, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %vitb0_1dxhat = stablehlo.multiply %vitb0_mdx, %vitb0_1gbk : tensor<32x197x192xf32>
    %vitb0_1dgpre = stablehlo.multiply %vitb0_mdx, %vitb0_1xhat : tensor<32x197x192xf32>
    %vitb0_1dg = stablehlo.reduce(%vitb0_1dgpre init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb0_1db = stablehlo.reduce(%vitb0_mdx init: %sc) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitb0_1m1s = stablehlo.reduce(%vitb0_1dxhat init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb0_1m1 = stablehlo.divide %vitb0_1m1s, %vitb0_1nf : tensor<32x197xf32>
    %vitb0_1dxxh = stablehlo.multiply %vitb0_1dxhat, %vitb0_1xhat : tensor<32x197x192xf32>
    %vitb0_1m2s = stablehlo.reduce(%vitb0_1dxxh init: %sc) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %vitb0_1m2 = stablehlo.divide %vitb0_1m2s, %vitb0_1nf : tensor<32x197xf32>
    %vitb0_1m1b = stablehlo.broadcast_in_dim %vitb0_1m1, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb0_1m2b = stablehlo.broadcast_in_dim %vitb0_1m2, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %vitb0_1t1 = stablehlo.subtract %vitb0_1dxhat, %vitb0_1m1b : tensor<32x197x192xf32>
    %vitb0_1xm2 = stablehlo.multiply %vitb0_1xhat, %vitb0_1m2b : tensor<32x197x192xf32>
    %vitb0_1t2 = stablehlo.subtract %vitb0_1t1, %vitb0_1xm2 : tensor<32x197x192xf32>
    %vitb0_1dx = stablehlo.multiply %vitb0_1istdb, %vitb0_1t2 : tensor<32x197x192xf32>
    %vitb0_dx = stablehlo.add %vitb0_dr1, %vitb0_1dx : tensor<32x197x192xf32>
    %vitcpdpos = stablehlo.reduce(%vitb0_dx init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<197x192xf32>
    %vitcpcslc = stablehlo.slice %vitb0_dx [0:32, 0:1, 0:192] : (tensor<32x197x192xf32>) -> tensor<32x1x192xf32>
    %vitcpcr = stablehlo.reshape %vitcpcslc : (tensor<32x1x192xf32>) -> tensor<32x192xf32>
    %vitcpdcls = stablehlo.reduce(%vitcpcr init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<32x192xf32>, tensor<f32>) -> tensor<192xf32>
    %vitcpdcls2 = stablehlo.reshape %vitcpdcls : (tensor<192xf32>) -> tensor<1x192xf32>
    %vitcpdtok = stablehlo.slice %vitb0_dx [0:32, 1:197, 0:192] : (tensor<32x197x192xf32>) -> tensor<32x196x192xf32>
    %vitpedtr = stablehlo.reshape %vitcpdtok : (tensor<32x196x192xf32>) -> tensor<32x14x14x192xf32>
    %vitpedy = stablehlo.transpose %vitpedtr, dims = [0, 3, 1, 2] : (tensor<32x14x14x192xf32>) -> tensor<32x192x14x14xf32>
    %vitpedb = stablehlo.reduce(%vitpedy init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<192xf32>
    %vitpeu = stablehlo.pad %vitpedy, %sc, low = [0, 0, 0, 0], high = [0, 0, 0, 0], interior = [0, 0, 15, 15] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<32x192x209x209xf32>
    %vitpext = stablehlo.transpose %xr, dims = [1, 0, 2, 3] : (tensor<32x3x224x224xf32>) -> tensor<3x32x224x224xf32>
    %vitpedt = stablehlo.transpose %vitpeu, dims = [1, 0, 2, 3] : (tensor<32x192x209x209xf32>) -> tensor<192x32x209x209xf32>
    %vitperaw = stablehlo.convolution(%vitpext, %vitpedt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x32x224x224xf32>, tensor<192x32x209x209xf32>) -> tensor<3x192x16x16xf32>
    %vitpedw = stablehlo.transpose %vitperaw, dims = [1, 0, 2, 3] : (tensor<3x192x16x16xf32>) -> tensor<192x3x16x16xf32>
    %wConv_lr = stablehlo.constant dense<0.1> : tensor<192x3x16x16xf32>
    %wConv_st = stablehlo.multiply %vitpedw, %wConv_lr : tensor<192x3x16x16xf32>
    %wConvn = stablehlo.subtract %wConv, %wConv_st : tensor<192x3x16x16xf32>
    %bConv_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %bConv_st = stablehlo.multiply %vitpedb, %bConv_lr : tensor<192xf32>
    %bConvn = stablehlo.subtract %bConv, %bConv_st : tensor<192xf32>
    %cls_lr = stablehlo.constant dense<0.1> : tensor<1x192xf32>
    %cls_st = stablehlo.multiply %vitcpdcls2, %cls_lr : tensor<1x192xf32>
    %clsn = stablehlo.subtract %cls, %cls_st : tensor<1x192xf32>
    %pos_lr = stablehlo.constant dense<0.1> : tensor<197x192xf32>
    %pos_st = stablehlo.multiply %vitcpdpos, %pos_lr : tensor<197x192xf32>
    %posn = stablehlo.subtract %pos, %pos_st : tensor<197x192xf32>
    %g1_0_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %g1_0_st = stablehlo.multiply %vitb0_1dg, %g1_0_lr : tensor<192xf32>
    %g1_0n = stablehlo.subtract %g1_0, %g1_0_st : tensor<192xf32>
    %b1_0_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %b1_0_st = stablehlo.multiply %vitb0_1db, %b1_0_lr : tensor<192xf32>
    %b1_0n = stablehlo.subtract %b1_0, %b1_0_st : tensor<192xf32>
    %Wq_0_lr = stablehlo.constant dense<0.1> : tensor<192x192xf32>
    %Wq_0_st = stablehlo.multiply %vitb0_mdWQ, %Wq_0_lr : tensor<192x192xf32>
    %Wq_0n = stablehlo.subtract %Wq_0, %Wq_0_st : tensor<192x192xf32>
    %bq_0_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %bq_0_st = stablehlo.multiply %vitb0_mdbQ, %bq_0_lr : tensor<192xf32>
    %bq_0n = stablehlo.subtract %bq_0, %bq_0_st : tensor<192xf32>
    %Wk_0_lr = stablehlo.constant dense<0.1> : tensor<192x192xf32>
    %Wk_0_st = stablehlo.multiply %vitb0_mdWK, %Wk_0_lr : tensor<192x192xf32>
    %Wk_0n = stablehlo.subtract %Wk_0, %Wk_0_st : tensor<192x192xf32>
    %bk_0_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %bk_0_st = stablehlo.multiply %vitb0_mdbK, %bk_0_lr : tensor<192xf32>
    %bk_0n = stablehlo.subtract %bk_0, %bk_0_st : tensor<192xf32>
    %Wv_0_lr = stablehlo.constant dense<0.1> : tensor<192x192xf32>
    %Wv_0_st = stablehlo.multiply %vitb0_mdWV, %Wv_0_lr : tensor<192x192xf32>
    %Wv_0n = stablehlo.subtract %Wv_0, %Wv_0_st : tensor<192x192xf32>
    %bv_0_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %bv_0_st = stablehlo.multiply %vitb0_mdbV, %bv_0_lr : tensor<192xf32>
    %bv_0n = stablehlo.subtract %bv_0, %bv_0_st : tensor<192xf32>
    %Wo_0_lr = stablehlo.constant dense<0.1> : tensor<192x192xf32>
    %Wo_0_st = stablehlo.multiply %vitb0_mdWo, %Wo_0_lr : tensor<192x192xf32>
    %Wo_0n = stablehlo.subtract %Wo_0, %Wo_0_st : tensor<192x192xf32>
    %bo_0_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %bo_0_st = stablehlo.multiply %vitb0_mdbo, %bo_0_lr : tensor<192xf32>
    %bo_0n = stablehlo.subtract %bo_0, %bo_0_st : tensor<192xf32>
    %g2_0_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %g2_0_st = stablehlo.multiply %vitb0_2dg, %g2_0_lr : tensor<192xf32>
    %g2_0n = stablehlo.subtract %g2_0, %g2_0_st : tensor<192xf32>
    %b2_0_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %b2_0_st = stablehlo.multiply %vitb0_2db, %b2_0_lr : tensor<192xf32>
    %b2_0n = stablehlo.subtract %b2_0, %b2_0_st : tensor<192xf32>
    %Wfc1_0_lr = stablehlo.constant dense<0.1> : tensor<192x768xf32>
    %Wfc1_0_st = stablehlo.multiply %vitb0_pdWfc1, %Wfc1_0_lr : tensor<192x768xf32>
    %Wfc1_0n = stablehlo.subtract %Wfc1_0, %Wfc1_0_st : tensor<192x768xf32>
    %bfc1_0_lr = stablehlo.constant dense<0.1> : tensor<768xf32>
    %bfc1_0_st = stablehlo.multiply %vitb0_pdbfc1, %bfc1_0_lr : tensor<768xf32>
    %bfc1_0n = stablehlo.subtract %bfc1_0, %bfc1_0_st : tensor<768xf32>
    %Wfc2_0_lr = stablehlo.constant dense<0.1> : tensor<768x192xf32>
    %Wfc2_0_st = stablehlo.multiply %vitb0_pdWfc2, %Wfc2_0_lr : tensor<768x192xf32>
    %Wfc2_0n = stablehlo.subtract %Wfc2_0, %Wfc2_0_st : tensor<768x192xf32>
    %bfc2_0_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %bfc2_0_st = stablehlo.multiply %vitb0_pdbfc2, %bfc2_0_lr : tensor<192xf32>
    %bfc2_0n = stablehlo.subtract %bfc2_0, %bfc2_0_st : tensor<192xf32>
    %g1_1_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %g1_1_st = stablehlo.multiply %vitb1_1dg, %g1_1_lr : tensor<192xf32>
    %g1_1n = stablehlo.subtract %g1_1, %g1_1_st : tensor<192xf32>
    %b1_1_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %b1_1_st = stablehlo.multiply %vitb1_1db, %b1_1_lr : tensor<192xf32>
    %b1_1n = stablehlo.subtract %b1_1, %b1_1_st : tensor<192xf32>
    %Wq_1_lr = stablehlo.constant dense<0.1> : tensor<192x192xf32>
    %Wq_1_st = stablehlo.multiply %vitb1_mdWQ, %Wq_1_lr : tensor<192x192xf32>
    %Wq_1n = stablehlo.subtract %Wq_1, %Wq_1_st : tensor<192x192xf32>
    %bq_1_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %bq_1_st = stablehlo.multiply %vitb1_mdbQ, %bq_1_lr : tensor<192xf32>
    %bq_1n = stablehlo.subtract %bq_1, %bq_1_st : tensor<192xf32>
    %Wk_1_lr = stablehlo.constant dense<0.1> : tensor<192x192xf32>
    %Wk_1_st = stablehlo.multiply %vitb1_mdWK, %Wk_1_lr : tensor<192x192xf32>
    %Wk_1n = stablehlo.subtract %Wk_1, %Wk_1_st : tensor<192x192xf32>
    %bk_1_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %bk_1_st = stablehlo.multiply %vitb1_mdbK, %bk_1_lr : tensor<192xf32>
    %bk_1n = stablehlo.subtract %bk_1, %bk_1_st : tensor<192xf32>
    %Wv_1_lr = stablehlo.constant dense<0.1> : tensor<192x192xf32>
    %Wv_1_st = stablehlo.multiply %vitb1_mdWV, %Wv_1_lr : tensor<192x192xf32>
    %Wv_1n = stablehlo.subtract %Wv_1, %Wv_1_st : tensor<192x192xf32>
    %bv_1_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %bv_1_st = stablehlo.multiply %vitb1_mdbV, %bv_1_lr : tensor<192xf32>
    %bv_1n = stablehlo.subtract %bv_1, %bv_1_st : tensor<192xf32>
    %Wo_1_lr = stablehlo.constant dense<0.1> : tensor<192x192xf32>
    %Wo_1_st = stablehlo.multiply %vitb1_mdWo, %Wo_1_lr : tensor<192x192xf32>
    %Wo_1n = stablehlo.subtract %Wo_1, %Wo_1_st : tensor<192x192xf32>
    %bo_1_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %bo_1_st = stablehlo.multiply %vitb1_mdbo, %bo_1_lr : tensor<192xf32>
    %bo_1n = stablehlo.subtract %bo_1, %bo_1_st : tensor<192xf32>
    %g2_1_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %g2_1_st = stablehlo.multiply %vitb1_2dg, %g2_1_lr : tensor<192xf32>
    %g2_1n = stablehlo.subtract %g2_1, %g2_1_st : tensor<192xf32>
    %b2_1_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %b2_1_st = stablehlo.multiply %vitb1_2db, %b2_1_lr : tensor<192xf32>
    %b2_1n = stablehlo.subtract %b2_1, %b2_1_st : tensor<192xf32>
    %Wfc1_1_lr = stablehlo.constant dense<0.1> : tensor<192x768xf32>
    %Wfc1_1_st = stablehlo.multiply %vitb1_pdWfc1, %Wfc1_1_lr : tensor<192x768xf32>
    %Wfc1_1n = stablehlo.subtract %Wfc1_1, %Wfc1_1_st : tensor<192x768xf32>
    %bfc1_1_lr = stablehlo.constant dense<0.1> : tensor<768xf32>
    %bfc1_1_st = stablehlo.multiply %vitb1_pdbfc1, %bfc1_1_lr : tensor<768xf32>
    %bfc1_1n = stablehlo.subtract %bfc1_1, %bfc1_1_st : tensor<768xf32>
    %Wfc2_1_lr = stablehlo.constant dense<0.1> : tensor<768x192xf32>
    %Wfc2_1_st = stablehlo.multiply %vitb1_pdWfc2, %Wfc2_1_lr : tensor<768x192xf32>
    %Wfc2_1n = stablehlo.subtract %Wfc2_1, %Wfc2_1_st : tensor<768x192xf32>
    %bfc2_1_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %bfc2_1_st = stablehlo.multiply %vitb1_pdbfc2, %bfc2_1_lr : tensor<192xf32>
    %bfc2_1n = stablehlo.subtract %bfc2_1, %bfc2_1_st : tensor<192xf32>
    %g1_2_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %g1_2_st = stablehlo.multiply %vitb2_1dg, %g1_2_lr : tensor<192xf32>
    %g1_2n = stablehlo.subtract %g1_2, %g1_2_st : tensor<192xf32>
    %b1_2_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %b1_2_st = stablehlo.multiply %vitb2_1db, %b1_2_lr : tensor<192xf32>
    %b1_2n = stablehlo.subtract %b1_2, %b1_2_st : tensor<192xf32>
    %Wq_2_lr = stablehlo.constant dense<0.1> : tensor<192x192xf32>
    %Wq_2_st = stablehlo.multiply %vitb2_mdWQ, %Wq_2_lr : tensor<192x192xf32>
    %Wq_2n = stablehlo.subtract %Wq_2, %Wq_2_st : tensor<192x192xf32>
    %bq_2_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %bq_2_st = stablehlo.multiply %vitb2_mdbQ, %bq_2_lr : tensor<192xf32>
    %bq_2n = stablehlo.subtract %bq_2, %bq_2_st : tensor<192xf32>
    %Wk_2_lr = stablehlo.constant dense<0.1> : tensor<192x192xf32>
    %Wk_2_st = stablehlo.multiply %vitb2_mdWK, %Wk_2_lr : tensor<192x192xf32>
    %Wk_2n = stablehlo.subtract %Wk_2, %Wk_2_st : tensor<192x192xf32>
    %bk_2_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %bk_2_st = stablehlo.multiply %vitb2_mdbK, %bk_2_lr : tensor<192xf32>
    %bk_2n = stablehlo.subtract %bk_2, %bk_2_st : tensor<192xf32>
    %Wv_2_lr = stablehlo.constant dense<0.1> : tensor<192x192xf32>
    %Wv_2_st = stablehlo.multiply %vitb2_mdWV, %Wv_2_lr : tensor<192x192xf32>
    %Wv_2n = stablehlo.subtract %Wv_2, %Wv_2_st : tensor<192x192xf32>
    %bv_2_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %bv_2_st = stablehlo.multiply %vitb2_mdbV, %bv_2_lr : tensor<192xf32>
    %bv_2n = stablehlo.subtract %bv_2, %bv_2_st : tensor<192xf32>
    %Wo_2_lr = stablehlo.constant dense<0.1> : tensor<192x192xf32>
    %Wo_2_st = stablehlo.multiply %vitb2_mdWo, %Wo_2_lr : tensor<192x192xf32>
    %Wo_2n = stablehlo.subtract %Wo_2, %Wo_2_st : tensor<192x192xf32>
    %bo_2_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %bo_2_st = stablehlo.multiply %vitb2_mdbo, %bo_2_lr : tensor<192xf32>
    %bo_2n = stablehlo.subtract %bo_2, %bo_2_st : tensor<192xf32>
    %g2_2_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %g2_2_st = stablehlo.multiply %vitb2_2dg, %g2_2_lr : tensor<192xf32>
    %g2_2n = stablehlo.subtract %g2_2, %g2_2_st : tensor<192xf32>
    %b2_2_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %b2_2_st = stablehlo.multiply %vitb2_2db, %b2_2_lr : tensor<192xf32>
    %b2_2n = stablehlo.subtract %b2_2, %b2_2_st : tensor<192xf32>
    %Wfc1_2_lr = stablehlo.constant dense<0.1> : tensor<192x768xf32>
    %Wfc1_2_st = stablehlo.multiply %vitb2_pdWfc1, %Wfc1_2_lr : tensor<192x768xf32>
    %Wfc1_2n = stablehlo.subtract %Wfc1_2, %Wfc1_2_st : tensor<192x768xf32>
    %bfc1_2_lr = stablehlo.constant dense<0.1> : tensor<768xf32>
    %bfc1_2_st = stablehlo.multiply %vitb2_pdbfc1, %bfc1_2_lr : tensor<768xf32>
    %bfc1_2n = stablehlo.subtract %bfc1_2, %bfc1_2_st : tensor<768xf32>
    %Wfc2_2_lr = stablehlo.constant dense<0.1> : tensor<768x192xf32>
    %Wfc2_2_st = stablehlo.multiply %vitb2_pdWfc2, %Wfc2_2_lr : tensor<768x192xf32>
    %Wfc2_2n = stablehlo.subtract %Wfc2_2, %Wfc2_2_st : tensor<768x192xf32>
    %bfc2_2_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %bfc2_2_st = stablehlo.multiply %vitb2_pdbfc2, %bfc2_2_lr : tensor<192xf32>
    %bfc2_2n = stablehlo.subtract %bfc2_2, %bfc2_2_st : tensor<192xf32>
    %g1_3_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %g1_3_st = stablehlo.multiply %vitb3_1dg, %g1_3_lr : tensor<192xf32>
    %g1_3n = stablehlo.subtract %g1_3, %g1_3_st : tensor<192xf32>
    %b1_3_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %b1_3_st = stablehlo.multiply %vitb3_1db, %b1_3_lr : tensor<192xf32>
    %b1_3n = stablehlo.subtract %b1_3, %b1_3_st : tensor<192xf32>
    %Wq_3_lr = stablehlo.constant dense<0.1> : tensor<192x192xf32>
    %Wq_3_st = stablehlo.multiply %vitb3_mdWQ, %Wq_3_lr : tensor<192x192xf32>
    %Wq_3n = stablehlo.subtract %Wq_3, %Wq_3_st : tensor<192x192xf32>
    %bq_3_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %bq_3_st = stablehlo.multiply %vitb3_mdbQ, %bq_3_lr : tensor<192xf32>
    %bq_3n = stablehlo.subtract %bq_3, %bq_3_st : tensor<192xf32>
    %Wk_3_lr = stablehlo.constant dense<0.1> : tensor<192x192xf32>
    %Wk_3_st = stablehlo.multiply %vitb3_mdWK, %Wk_3_lr : tensor<192x192xf32>
    %Wk_3n = stablehlo.subtract %Wk_3, %Wk_3_st : tensor<192x192xf32>
    %bk_3_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %bk_3_st = stablehlo.multiply %vitb3_mdbK, %bk_3_lr : tensor<192xf32>
    %bk_3n = stablehlo.subtract %bk_3, %bk_3_st : tensor<192xf32>
    %Wv_3_lr = stablehlo.constant dense<0.1> : tensor<192x192xf32>
    %Wv_3_st = stablehlo.multiply %vitb3_mdWV, %Wv_3_lr : tensor<192x192xf32>
    %Wv_3n = stablehlo.subtract %Wv_3, %Wv_3_st : tensor<192x192xf32>
    %bv_3_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %bv_3_st = stablehlo.multiply %vitb3_mdbV, %bv_3_lr : tensor<192xf32>
    %bv_3n = stablehlo.subtract %bv_3, %bv_3_st : tensor<192xf32>
    %Wo_3_lr = stablehlo.constant dense<0.1> : tensor<192x192xf32>
    %Wo_3_st = stablehlo.multiply %vitb3_mdWo, %Wo_3_lr : tensor<192x192xf32>
    %Wo_3n = stablehlo.subtract %Wo_3, %Wo_3_st : tensor<192x192xf32>
    %bo_3_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %bo_3_st = stablehlo.multiply %vitb3_mdbo, %bo_3_lr : tensor<192xf32>
    %bo_3n = stablehlo.subtract %bo_3, %bo_3_st : tensor<192xf32>
    %g2_3_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %g2_3_st = stablehlo.multiply %vitb3_2dg, %g2_3_lr : tensor<192xf32>
    %g2_3n = stablehlo.subtract %g2_3, %g2_3_st : tensor<192xf32>
    %b2_3_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %b2_3_st = stablehlo.multiply %vitb3_2db, %b2_3_lr : tensor<192xf32>
    %b2_3n = stablehlo.subtract %b2_3, %b2_3_st : tensor<192xf32>
    %Wfc1_3_lr = stablehlo.constant dense<0.1> : tensor<192x768xf32>
    %Wfc1_3_st = stablehlo.multiply %vitb3_pdWfc1, %Wfc1_3_lr : tensor<192x768xf32>
    %Wfc1_3n = stablehlo.subtract %Wfc1_3, %Wfc1_3_st : tensor<192x768xf32>
    %bfc1_3_lr = stablehlo.constant dense<0.1> : tensor<768xf32>
    %bfc1_3_st = stablehlo.multiply %vitb3_pdbfc1, %bfc1_3_lr : tensor<768xf32>
    %bfc1_3n = stablehlo.subtract %bfc1_3, %bfc1_3_st : tensor<768xf32>
    %Wfc2_3_lr = stablehlo.constant dense<0.1> : tensor<768x192xf32>
    %Wfc2_3_st = stablehlo.multiply %vitb3_pdWfc2, %Wfc2_3_lr : tensor<768x192xf32>
    %Wfc2_3n = stablehlo.subtract %Wfc2_3, %Wfc2_3_st : tensor<768x192xf32>
    %bfc2_3_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %bfc2_3_st = stablehlo.multiply %vitb3_pdbfc2, %bfc2_3_lr : tensor<192xf32>
    %bfc2_3n = stablehlo.subtract %bfc2_3, %bfc2_3_st : tensor<192xf32>
    %g1_4_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %g1_4_st = stablehlo.multiply %vitb4_1dg, %g1_4_lr : tensor<192xf32>
    %g1_4n = stablehlo.subtract %g1_4, %g1_4_st : tensor<192xf32>
    %b1_4_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %b1_4_st = stablehlo.multiply %vitb4_1db, %b1_4_lr : tensor<192xf32>
    %b1_4n = stablehlo.subtract %b1_4, %b1_4_st : tensor<192xf32>
    %Wq_4_lr = stablehlo.constant dense<0.1> : tensor<192x192xf32>
    %Wq_4_st = stablehlo.multiply %vitb4_mdWQ, %Wq_4_lr : tensor<192x192xf32>
    %Wq_4n = stablehlo.subtract %Wq_4, %Wq_4_st : tensor<192x192xf32>
    %bq_4_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %bq_4_st = stablehlo.multiply %vitb4_mdbQ, %bq_4_lr : tensor<192xf32>
    %bq_4n = stablehlo.subtract %bq_4, %bq_4_st : tensor<192xf32>
    %Wk_4_lr = stablehlo.constant dense<0.1> : tensor<192x192xf32>
    %Wk_4_st = stablehlo.multiply %vitb4_mdWK, %Wk_4_lr : tensor<192x192xf32>
    %Wk_4n = stablehlo.subtract %Wk_4, %Wk_4_st : tensor<192x192xf32>
    %bk_4_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %bk_4_st = stablehlo.multiply %vitb4_mdbK, %bk_4_lr : tensor<192xf32>
    %bk_4n = stablehlo.subtract %bk_4, %bk_4_st : tensor<192xf32>
    %Wv_4_lr = stablehlo.constant dense<0.1> : tensor<192x192xf32>
    %Wv_4_st = stablehlo.multiply %vitb4_mdWV, %Wv_4_lr : tensor<192x192xf32>
    %Wv_4n = stablehlo.subtract %Wv_4, %Wv_4_st : tensor<192x192xf32>
    %bv_4_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %bv_4_st = stablehlo.multiply %vitb4_mdbV, %bv_4_lr : tensor<192xf32>
    %bv_4n = stablehlo.subtract %bv_4, %bv_4_st : tensor<192xf32>
    %Wo_4_lr = stablehlo.constant dense<0.1> : tensor<192x192xf32>
    %Wo_4_st = stablehlo.multiply %vitb4_mdWo, %Wo_4_lr : tensor<192x192xf32>
    %Wo_4n = stablehlo.subtract %Wo_4, %Wo_4_st : tensor<192x192xf32>
    %bo_4_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %bo_4_st = stablehlo.multiply %vitb4_mdbo, %bo_4_lr : tensor<192xf32>
    %bo_4n = stablehlo.subtract %bo_4, %bo_4_st : tensor<192xf32>
    %g2_4_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %g2_4_st = stablehlo.multiply %vitb4_2dg, %g2_4_lr : tensor<192xf32>
    %g2_4n = stablehlo.subtract %g2_4, %g2_4_st : tensor<192xf32>
    %b2_4_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %b2_4_st = stablehlo.multiply %vitb4_2db, %b2_4_lr : tensor<192xf32>
    %b2_4n = stablehlo.subtract %b2_4, %b2_4_st : tensor<192xf32>
    %Wfc1_4_lr = stablehlo.constant dense<0.1> : tensor<192x768xf32>
    %Wfc1_4_st = stablehlo.multiply %vitb4_pdWfc1, %Wfc1_4_lr : tensor<192x768xf32>
    %Wfc1_4n = stablehlo.subtract %Wfc1_4, %Wfc1_4_st : tensor<192x768xf32>
    %bfc1_4_lr = stablehlo.constant dense<0.1> : tensor<768xf32>
    %bfc1_4_st = stablehlo.multiply %vitb4_pdbfc1, %bfc1_4_lr : tensor<768xf32>
    %bfc1_4n = stablehlo.subtract %bfc1_4, %bfc1_4_st : tensor<768xf32>
    %Wfc2_4_lr = stablehlo.constant dense<0.1> : tensor<768x192xf32>
    %Wfc2_4_st = stablehlo.multiply %vitb4_pdWfc2, %Wfc2_4_lr : tensor<768x192xf32>
    %Wfc2_4n = stablehlo.subtract %Wfc2_4, %Wfc2_4_st : tensor<768x192xf32>
    %bfc2_4_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %bfc2_4_st = stablehlo.multiply %vitb4_pdbfc2, %bfc2_4_lr : tensor<192xf32>
    %bfc2_4n = stablehlo.subtract %bfc2_4, %bfc2_4_st : tensor<192xf32>
    %g1_5_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %g1_5_st = stablehlo.multiply %vitb5_1dg, %g1_5_lr : tensor<192xf32>
    %g1_5n = stablehlo.subtract %g1_5, %g1_5_st : tensor<192xf32>
    %b1_5_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %b1_5_st = stablehlo.multiply %vitb5_1db, %b1_5_lr : tensor<192xf32>
    %b1_5n = stablehlo.subtract %b1_5, %b1_5_st : tensor<192xf32>
    %Wq_5_lr = stablehlo.constant dense<0.1> : tensor<192x192xf32>
    %Wq_5_st = stablehlo.multiply %vitb5_mdWQ, %Wq_5_lr : tensor<192x192xf32>
    %Wq_5n = stablehlo.subtract %Wq_5, %Wq_5_st : tensor<192x192xf32>
    %bq_5_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %bq_5_st = stablehlo.multiply %vitb5_mdbQ, %bq_5_lr : tensor<192xf32>
    %bq_5n = stablehlo.subtract %bq_5, %bq_5_st : tensor<192xf32>
    %Wk_5_lr = stablehlo.constant dense<0.1> : tensor<192x192xf32>
    %Wk_5_st = stablehlo.multiply %vitb5_mdWK, %Wk_5_lr : tensor<192x192xf32>
    %Wk_5n = stablehlo.subtract %Wk_5, %Wk_5_st : tensor<192x192xf32>
    %bk_5_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %bk_5_st = stablehlo.multiply %vitb5_mdbK, %bk_5_lr : tensor<192xf32>
    %bk_5n = stablehlo.subtract %bk_5, %bk_5_st : tensor<192xf32>
    %Wv_5_lr = stablehlo.constant dense<0.1> : tensor<192x192xf32>
    %Wv_5_st = stablehlo.multiply %vitb5_mdWV, %Wv_5_lr : tensor<192x192xf32>
    %Wv_5n = stablehlo.subtract %Wv_5, %Wv_5_st : tensor<192x192xf32>
    %bv_5_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %bv_5_st = stablehlo.multiply %vitb5_mdbV, %bv_5_lr : tensor<192xf32>
    %bv_5n = stablehlo.subtract %bv_5, %bv_5_st : tensor<192xf32>
    %Wo_5_lr = stablehlo.constant dense<0.1> : tensor<192x192xf32>
    %Wo_5_st = stablehlo.multiply %vitb5_mdWo, %Wo_5_lr : tensor<192x192xf32>
    %Wo_5n = stablehlo.subtract %Wo_5, %Wo_5_st : tensor<192x192xf32>
    %bo_5_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %bo_5_st = stablehlo.multiply %vitb5_mdbo, %bo_5_lr : tensor<192xf32>
    %bo_5n = stablehlo.subtract %bo_5, %bo_5_st : tensor<192xf32>
    %g2_5_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %g2_5_st = stablehlo.multiply %vitb5_2dg, %g2_5_lr : tensor<192xf32>
    %g2_5n = stablehlo.subtract %g2_5, %g2_5_st : tensor<192xf32>
    %b2_5_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %b2_5_st = stablehlo.multiply %vitb5_2db, %b2_5_lr : tensor<192xf32>
    %b2_5n = stablehlo.subtract %b2_5, %b2_5_st : tensor<192xf32>
    %Wfc1_5_lr = stablehlo.constant dense<0.1> : tensor<192x768xf32>
    %Wfc1_5_st = stablehlo.multiply %vitb5_pdWfc1, %Wfc1_5_lr : tensor<192x768xf32>
    %Wfc1_5n = stablehlo.subtract %Wfc1_5, %Wfc1_5_st : tensor<192x768xf32>
    %bfc1_5_lr = stablehlo.constant dense<0.1> : tensor<768xf32>
    %bfc1_5_st = stablehlo.multiply %vitb5_pdbfc1, %bfc1_5_lr : tensor<768xf32>
    %bfc1_5n = stablehlo.subtract %bfc1_5, %bfc1_5_st : tensor<768xf32>
    %Wfc2_5_lr = stablehlo.constant dense<0.1> : tensor<768x192xf32>
    %Wfc2_5_st = stablehlo.multiply %vitb5_pdWfc2, %Wfc2_5_lr : tensor<768x192xf32>
    %Wfc2_5n = stablehlo.subtract %Wfc2_5, %Wfc2_5_st : tensor<768x192xf32>
    %bfc2_5_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %bfc2_5_st = stablehlo.multiply %vitb5_pdbfc2, %bfc2_5_lr : tensor<192xf32>
    %bfc2_5n = stablehlo.subtract %bfc2_5, %bfc2_5_st : tensor<192xf32>
    %g1_6_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %g1_6_st = stablehlo.multiply %vitb6_1dg, %g1_6_lr : tensor<192xf32>
    %g1_6n = stablehlo.subtract %g1_6, %g1_6_st : tensor<192xf32>
    %b1_6_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %b1_6_st = stablehlo.multiply %vitb6_1db, %b1_6_lr : tensor<192xf32>
    %b1_6n = stablehlo.subtract %b1_6, %b1_6_st : tensor<192xf32>
    %Wq_6_lr = stablehlo.constant dense<0.1> : tensor<192x192xf32>
    %Wq_6_st = stablehlo.multiply %vitb6_mdWQ, %Wq_6_lr : tensor<192x192xf32>
    %Wq_6n = stablehlo.subtract %Wq_6, %Wq_6_st : tensor<192x192xf32>
    %bq_6_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %bq_6_st = stablehlo.multiply %vitb6_mdbQ, %bq_6_lr : tensor<192xf32>
    %bq_6n = stablehlo.subtract %bq_6, %bq_6_st : tensor<192xf32>
    %Wk_6_lr = stablehlo.constant dense<0.1> : tensor<192x192xf32>
    %Wk_6_st = stablehlo.multiply %vitb6_mdWK, %Wk_6_lr : tensor<192x192xf32>
    %Wk_6n = stablehlo.subtract %Wk_6, %Wk_6_st : tensor<192x192xf32>
    %bk_6_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %bk_6_st = stablehlo.multiply %vitb6_mdbK, %bk_6_lr : tensor<192xf32>
    %bk_6n = stablehlo.subtract %bk_6, %bk_6_st : tensor<192xf32>
    %Wv_6_lr = stablehlo.constant dense<0.1> : tensor<192x192xf32>
    %Wv_6_st = stablehlo.multiply %vitb6_mdWV, %Wv_6_lr : tensor<192x192xf32>
    %Wv_6n = stablehlo.subtract %Wv_6, %Wv_6_st : tensor<192x192xf32>
    %bv_6_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %bv_6_st = stablehlo.multiply %vitb6_mdbV, %bv_6_lr : tensor<192xf32>
    %bv_6n = stablehlo.subtract %bv_6, %bv_6_st : tensor<192xf32>
    %Wo_6_lr = stablehlo.constant dense<0.1> : tensor<192x192xf32>
    %Wo_6_st = stablehlo.multiply %vitb6_mdWo, %Wo_6_lr : tensor<192x192xf32>
    %Wo_6n = stablehlo.subtract %Wo_6, %Wo_6_st : tensor<192x192xf32>
    %bo_6_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %bo_6_st = stablehlo.multiply %vitb6_mdbo, %bo_6_lr : tensor<192xf32>
    %bo_6n = stablehlo.subtract %bo_6, %bo_6_st : tensor<192xf32>
    %g2_6_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %g2_6_st = stablehlo.multiply %vitb6_2dg, %g2_6_lr : tensor<192xf32>
    %g2_6n = stablehlo.subtract %g2_6, %g2_6_st : tensor<192xf32>
    %b2_6_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %b2_6_st = stablehlo.multiply %vitb6_2db, %b2_6_lr : tensor<192xf32>
    %b2_6n = stablehlo.subtract %b2_6, %b2_6_st : tensor<192xf32>
    %Wfc1_6_lr = stablehlo.constant dense<0.1> : tensor<192x768xf32>
    %Wfc1_6_st = stablehlo.multiply %vitb6_pdWfc1, %Wfc1_6_lr : tensor<192x768xf32>
    %Wfc1_6n = stablehlo.subtract %Wfc1_6, %Wfc1_6_st : tensor<192x768xf32>
    %bfc1_6_lr = stablehlo.constant dense<0.1> : tensor<768xf32>
    %bfc1_6_st = stablehlo.multiply %vitb6_pdbfc1, %bfc1_6_lr : tensor<768xf32>
    %bfc1_6n = stablehlo.subtract %bfc1_6, %bfc1_6_st : tensor<768xf32>
    %Wfc2_6_lr = stablehlo.constant dense<0.1> : tensor<768x192xf32>
    %Wfc2_6_st = stablehlo.multiply %vitb6_pdWfc2, %Wfc2_6_lr : tensor<768x192xf32>
    %Wfc2_6n = stablehlo.subtract %Wfc2_6, %Wfc2_6_st : tensor<768x192xf32>
    %bfc2_6_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %bfc2_6_st = stablehlo.multiply %vitb6_pdbfc2, %bfc2_6_lr : tensor<192xf32>
    %bfc2_6n = stablehlo.subtract %bfc2_6, %bfc2_6_st : tensor<192xf32>
    %g1_7_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %g1_7_st = stablehlo.multiply %vitb7_1dg, %g1_7_lr : tensor<192xf32>
    %g1_7n = stablehlo.subtract %g1_7, %g1_7_st : tensor<192xf32>
    %b1_7_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %b1_7_st = stablehlo.multiply %vitb7_1db, %b1_7_lr : tensor<192xf32>
    %b1_7n = stablehlo.subtract %b1_7, %b1_7_st : tensor<192xf32>
    %Wq_7_lr = stablehlo.constant dense<0.1> : tensor<192x192xf32>
    %Wq_7_st = stablehlo.multiply %vitb7_mdWQ, %Wq_7_lr : tensor<192x192xf32>
    %Wq_7n = stablehlo.subtract %Wq_7, %Wq_7_st : tensor<192x192xf32>
    %bq_7_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %bq_7_st = stablehlo.multiply %vitb7_mdbQ, %bq_7_lr : tensor<192xf32>
    %bq_7n = stablehlo.subtract %bq_7, %bq_7_st : tensor<192xf32>
    %Wk_7_lr = stablehlo.constant dense<0.1> : tensor<192x192xf32>
    %Wk_7_st = stablehlo.multiply %vitb7_mdWK, %Wk_7_lr : tensor<192x192xf32>
    %Wk_7n = stablehlo.subtract %Wk_7, %Wk_7_st : tensor<192x192xf32>
    %bk_7_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %bk_7_st = stablehlo.multiply %vitb7_mdbK, %bk_7_lr : tensor<192xf32>
    %bk_7n = stablehlo.subtract %bk_7, %bk_7_st : tensor<192xf32>
    %Wv_7_lr = stablehlo.constant dense<0.1> : tensor<192x192xf32>
    %Wv_7_st = stablehlo.multiply %vitb7_mdWV, %Wv_7_lr : tensor<192x192xf32>
    %Wv_7n = stablehlo.subtract %Wv_7, %Wv_7_st : tensor<192x192xf32>
    %bv_7_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %bv_7_st = stablehlo.multiply %vitb7_mdbV, %bv_7_lr : tensor<192xf32>
    %bv_7n = stablehlo.subtract %bv_7, %bv_7_st : tensor<192xf32>
    %Wo_7_lr = stablehlo.constant dense<0.1> : tensor<192x192xf32>
    %Wo_7_st = stablehlo.multiply %vitb7_mdWo, %Wo_7_lr : tensor<192x192xf32>
    %Wo_7n = stablehlo.subtract %Wo_7, %Wo_7_st : tensor<192x192xf32>
    %bo_7_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %bo_7_st = stablehlo.multiply %vitb7_mdbo, %bo_7_lr : tensor<192xf32>
    %bo_7n = stablehlo.subtract %bo_7, %bo_7_st : tensor<192xf32>
    %g2_7_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %g2_7_st = stablehlo.multiply %vitb7_2dg, %g2_7_lr : tensor<192xf32>
    %g2_7n = stablehlo.subtract %g2_7, %g2_7_st : tensor<192xf32>
    %b2_7_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %b2_7_st = stablehlo.multiply %vitb7_2db, %b2_7_lr : tensor<192xf32>
    %b2_7n = stablehlo.subtract %b2_7, %b2_7_st : tensor<192xf32>
    %Wfc1_7_lr = stablehlo.constant dense<0.1> : tensor<192x768xf32>
    %Wfc1_7_st = stablehlo.multiply %vitb7_pdWfc1, %Wfc1_7_lr : tensor<192x768xf32>
    %Wfc1_7n = stablehlo.subtract %Wfc1_7, %Wfc1_7_st : tensor<192x768xf32>
    %bfc1_7_lr = stablehlo.constant dense<0.1> : tensor<768xf32>
    %bfc1_7_st = stablehlo.multiply %vitb7_pdbfc1, %bfc1_7_lr : tensor<768xf32>
    %bfc1_7n = stablehlo.subtract %bfc1_7, %bfc1_7_st : tensor<768xf32>
    %Wfc2_7_lr = stablehlo.constant dense<0.1> : tensor<768x192xf32>
    %Wfc2_7_st = stablehlo.multiply %vitb7_pdWfc2, %Wfc2_7_lr : tensor<768x192xf32>
    %Wfc2_7n = stablehlo.subtract %Wfc2_7, %Wfc2_7_st : tensor<768x192xf32>
    %bfc2_7_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %bfc2_7_st = stablehlo.multiply %vitb7_pdbfc2, %bfc2_7_lr : tensor<192xf32>
    %bfc2_7n = stablehlo.subtract %bfc2_7, %bfc2_7_st : tensor<192xf32>
    %g1_8_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %g1_8_st = stablehlo.multiply %vitb8_1dg, %g1_8_lr : tensor<192xf32>
    %g1_8n = stablehlo.subtract %g1_8, %g1_8_st : tensor<192xf32>
    %b1_8_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %b1_8_st = stablehlo.multiply %vitb8_1db, %b1_8_lr : tensor<192xf32>
    %b1_8n = stablehlo.subtract %b1_8, %b1_8_st : tensor<192xf32>
    %Wq_8_lr = stablehlo.constant dense<0.1> : tensor<192x192xf32>
    %Wq_8_st = stablehlo.multiply %vitb8_mdWQ, %Wq_8_lr : tensor<192x192xf32>
    %Wq_8n = stablehlo.subtract %Wq_8, %Wq_8_st : tensor<192x192xf32>
    %bq_8_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %bq_8_st = stablehlo.multiply %vitb8_mdbQ, %bq_8_lr : tensor<192xf32>
    %bq_8n = stablehlo.subtract %bq_8, %bq_8_st : tensor<192xf32>
    %Wk_8_lr = stablehlo.constant dense<0.1> : tensor<192x192xf32>
    %Wk_8_st = stablehlo.multiply %vitb8_mdWK, %Wk_8_lr : tensor<192x192xf32>
    %Wk_8n = stablehlo.subtract %Wk_8, %Wk_8_st : tensor<192x192xf32>
    %bk_8_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %bk_8_st = stablehlo.multiply %vitb8_mdbK, %bk_8_lr : tensor<192xf32>
    %bk_8n = stablehlo.subtract %bk_8, %bk_8_st : tensor<192xf32>
    %Wv_8_lr = stablehlo.constant dense<0.1> : tensor<192x192xf32>
    %Wv_8_st = stablehlo.multiply %vitb8_mdWV, %Wv_8_lr : tensor<192x192xf32>
    %Wv_8n = stablehlo.subtract %Wv_8, %Wv_8_st : tensor<192x192xf32>
    %bv_8_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %bv_8_st = stablehlo.multiply %vitb8_mdbV, %bv_8_lr : tensor<192xf32>
    %bv_8n = stablehlo.subtract %bv_8, %bv_8_st : tensor<192xf32>
    %Wo_8_lr = stablehlo.constant dense<0.1> : tensor<192x192xf32>
    %Wo_8_st = stablehlo.multiply %vitb8_mdWo, %Wo_8_lr : tensor<192x192xf32>
    %Wo_8n = stablehlo.subtract %Wo_8, %Wo_8_st : tensor<192x192xf32>
    %bo_8_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %bo_8_st = stablehlo.multiply %vitb8_mdbo, %bo_8_lr : tensor<192xf32>
    %bo_8n = stablehlo.subtract %bo_8, %bo_8_st : tensor<192xf32>
    %g2_8_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %g2_8_st = stablehlo.multiply %vitb8_2dg, %g2_8_lr : tensor<192xf32>
    %g2_8n = stablehlo.subtract %g2_8, %g2_8_st : tensor<192xf32>
    %b2_8_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %b2_8_st = stablehlo.multiply %vitb8_2db, %b2_8_lr : tensor<192xf32>
    %b2_8n = stablehlo.subtract %b2_8, %b2_8_st : tensor<192xf32>
    %Wfc1_8_lr = stablehlo.constant dense<0.1> : tensor<192x768xf32>
    %Wfc1_8_st = stablehlo.multiply %vitb8_pdWfc1, %Wfc1_8_lr : tensor<192x768xf32>
    %Wfc1_8n = stablehlo.subtract %Wfc1_8, %Wfc1_8_st : tensor<192x768xf32>
    %bfc1_8_lr = stablehlo.constant dense<0.1> : tensor<768xf32>
    %bfc1_8_st = stablehlo.multiply %vitb8_pdbfc1, %bfc1_8_lr : tensor<768xf32>
    %bfc1_8n = stablehlo.subtract %bfc1_8, %bfc1_8_st : tensor<768xf32>
    %Wfc2_8_lr = stablehlo.constant dense<0.1> : tensor<768x192xf32>
    %Wfc2_8_st = stablehlo.multiply %vitb8_pdWfc2, %Wfc2_8_lr : tensor<768x192xf32>
    %Wfc2_8n = stablehlo.subtract %Wfc2_8, %Wfc2_8_st : tensor<768x192xf32>
    %bfc2_8_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %bfc2_8_st = stablehlo.multiply %vitb8_pdbfc2, %bfc2_8_lr : tensor<192xf32>
    %bfc2_8n = stablehlo.subtract %bfc2_8, %bfc2_8_st : tensor<192xf32>
    %g1_9_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %g1_9_st = stablehlo.multiply %vitb9_1dg, %g1_9_lr : tensor<192xf32>
    %g1_9n = stablehlo.subtract %g1_9, %g1_9_st : tensor<192xf32>
    %b1_9_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %b1_9_st = stablehlo.multiply %vitb9_1db, %b1_9_lr : tensor<192xf32>
    %b1_9n = stablehlo.subtract %b1_9, %b1_9_st : tensor<192xf32>
    %Wq_9_lr = stablehlo.constant dense<0.1> : tensor<192x192xf32>
    %Wq_9_st = stablehlo.multiply %vitb9_mdWQ, %Wq_9_lr : tensor<192x192xf32>
    %Wq_9n = stablehlo.subtract %Wq_9, %Wq_9_st : tensor<192x192xf32>
    %bq_9_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %bq_9_st = stablehlo.multiply %vitb9_mdbQ, %bq_9_lr : tensor<192xf32>
    %bq_9n = stablehlo.subtract %bq_9, %bq_9_st : tensor<192xf32>
    %Wk_9_lr = stablehlo.constant dense<0.1> : tensor<192x192xf32>
    %Wk_9_st = stablehlo.multiply %vitb9_mdWK, %Wk_9_lr : tensor<192x192xf32>
    %Wk_9n = stablehlo.subtract %Wk_9, %Wk_9_st : tensor<192x192xf32>
    %bk_9_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %bk_9_st = stablehlo.multiply %vitb9_mdbK, %bk_9_lr : tensor<192xf32>
    %bk_9n = stablehlo.subtract %bk_9, %bk_9_st : tensor<192xf32>
    %Wv_9_lr = stablehlo.constant dense<0.1> : tensor<192x192xf32>
    %Wv_9_st = stablehlo.multiply %vitb9_mdWV, %Wv_9_lr : tensor<192x192xf32>
    %Wv_9n = stablehlo.subtract %Wv_9, %Wv_9_st : tensor<192x192xf32>
    %bv_9_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %bv_9_st = stablehlo.multiply %vitb9_mdbV, %bv_9_lr : tensor<192xf32>
    %bv_9n = stablehlo.subtract %bv_9, %bv_9_st : tensor<192xf32>
    %Wo_9_lr = stablehlo.constant dense<0.1> : tensor<192x192xf32>
    %Wo_9_st = stablehlo.multiply %vitb9_mdWo, %Wo_9_lr : tensor<192x192xf32>
    %Wo_9n = stablehlo.subtract %Wo_9, %Wo_9_st : tensor<192x192xf32>
    %bo_9_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %bo_9_st = stablehlo.multiply %vitb9_mdbo, %bo_9_lr : tensor<192xf32>
    %bo_9n = stablehlo.subtract %bo_9, %bo_9_st : tensor<192xf32>
    %g2_9_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %g2_9_st = stablehlo.multiply %vitb9_2dg, %g2_9_lr : tensor<192xf32>
    %g2_9n = stablehlo.subtract %g2_9, %g2_9_st : tensor<192xf32>
    %b2_9_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %b2_9_st = stablehlo.multiply %vitb9_2db, %b2_9_lr : tensor<192xf32>
    %b2_9n = stablehlo.subtract %b2_9, %b2_9_st : tensor<192xf32>
    %Wfc1_9_lr = stablehlo.constant dense<0.1> : tensor<192x768xf32>
    %Wfc1_9_st = stablehlo.multiply %vitb9_pdWfc1, %Wfc1_9_lr : tensor<192x768xf32>
    %Wfc1_9n = stablehlo.subtract %Wfc1_9, %Wfc1_9_st : tensor<192x768xf32>
    %bfc1_9_lr = stablehlo.constant dense<0.1> : tensor<768xf32>
    %bfc1_9_st = stablehlo.multiply %vitb9_pdbfc1, %bfc1_9_lr : tensor<768xf32>
    %bfc1_9n = stablehlo.subtract %bfc1_9, %bfc1_9_st : tensor<768xf32>
    %Wfc2_9_lr = stablehlo.constant dense<0.1> : tensor<768x192xf32>
    %Wfc2_9_st = stablehlo.multiply %vitb9_pdWfc2, %Wfc2_9_lr : tensor<768x192xf32>
    %Wfc2_9n = stablehlo.subtract %Wfc2_9, %Wfc2_9_st : tensor<768x192xf32>
    %bfc2_9_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %bfc2_9_st = stablehlo.multiply %vitb9_pdbfc2, %bfc2_9_lr : tensor<192xf32>
    %bfc2_9n = stablehlo.subtract %bfc2_9, %bfc2_9_st : tensor<192xf32>
    %g1_10_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %g1_10_st = stablehlo.multiply %vitb10_1dg, %g1_10_lr : tensor<192xf32>
    %g1_10n = stablehlo.subtract %g1_10, %g1_10_st : tensor<192xf32>
    %b1_10_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %b1_10_st = stablehlo.multiply %vitb10_1db, %b1_10_lr : tensor<192xf32>
    %b1_10n = stablehlo.subtract %b1_10, %b1_10_st : tensor<192xf32>
    %Wq_10_lr = stablehlo.constant dense<0.1> : tensor<192x192xf32>
    %Wq_10_st = stablehlo.multiply %vitb10_mdWQ, %Wq_10_lr : tensor<192x192xf32>
    %Wq_10n = stablehlo.subtract %Wq_10, %Wq_10_st : tensor<192x192xf32>
    %bq_10_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %bq_10_st = stablehlo.multiply %vitb10_mdbQ, %bq_10_lr : tensor<192xf32>
    %bq_10n = stablehlo.subtract %bq_10, %bq_10_st : tensor<192xf32>
    %Wk_10_lr = stablehlo.constant dense<0.1> : tensor<192x192xf32>
    %Wk_10_st = stablehlo.multiply %vitb10_mdWK, %Wk_10_lr : tensor<192x192xf32>
    %Wk_10n = stablehlo.subtract %Wk_10, %Wk_10_st : tensor<192x192xf32>
    %bk_10_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %bk_10_st = stablehlo.multiply %vitb10_mdbK, %bk_10_lr : tensor<192xf32>
    %bk_10n = stablehlo.subtract %bk_10, %bk_10_st : tensor<192xf32>
    %Wv_10_lr = stablehlo.constant dense<0.1> : tensor<192x192xf32>
    %Wv_10_st = stablehlo.multiply %vitb10_mdWV, %Wv_10_lr : tensor<192x192xf32>
    %Wv_10n = stablehlo.subtract %Wv_10, %Wv_10_st : tensor<192x192xf32>
    %bv_10_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %bv_10_st = stablehlo.multiply %vitb10_mdbV, %bv_10_lr : tensor<192xf32>
    %bv_10n = stablehlo.subtract %bv_10, %bv_10_st : tensor<192xf32>
    %Wo_10_lr = stablehlo.constant dense<0.1> : tensor<192x192xf32>
    %Wo_10_st = stablehlo.multiply %vitb10_mdWo, %Wo_10_lr : tensor<192x192xf32>
    %Wo_10n = stablehlo.subtract %Wo_10, %Wo_10_st : tensor<192x192xf32>
    %bo_10_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %bo_10_st = stablehlo.multiply %vitb10_mdbo, %bo_10_lr : tensor<192xf32>
    %bo_10n = stablehlo.subtract %bo_10, %bo_10_st : tensor<192xf32>
    %g2_10_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %g2_10_st = stablehlo.multiply %vitb10_2dg, %g2_10_lr : tensor<192xf32>
    %g2_10n = stablehlo.subtract %g2_10, %g2_10_st : tensor<192xf32>
    %b2_10_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %b2_10_st = stablehlo.multiply %vitb10_2db, %b2_10_lr : tensor<192xf32>
    %b2_10n = stablehlo.subtract %b2_10, %b2_10_st : tensor<192xf32>
    %Wfc1_10_lr = stablehlo.constant dense<0.1> : tensor<192x768xf32>
    %Wfc1_10_st = stablehlo.multiply %vitb10_pdWfc1, %Wfc1_10_lr : tensor<192x768xf32>
    %Wfc1_10n = stablehlo.subtract %Wfc1_10, %Wfc1_10_st : tensor<192x768xf32>
    %bfc1_10_lr = stablehlo.constant dense<0.1> : tensor<768xf32>
    %bfc1_10_st = stablehlo.multiply %vitb10_pdbfc1, %bfc1_10_lr : tensor<768xf32>
    %bfc1_10n = stablehlo.subtract %bfc1_10, %bfc1_10_st : tensor<768xf32>
    %Wfc2_10_lr = stablehlo.constant dense<0.1> : tensor<768x192xf32>
    %Wfc2_10_st = stablehlo.multiply %vitb10_pdWfc2, %Wfc2_10_lr : tensor<768x192xf32>
    %Wfc2_10n = stablehlo.subtract %Wfc2_10, %Wfc2_10_st : tensor<768x192xf32>
    %bfc2_10_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %bfc2_10_st = stablehlo.multiply %vitb10_pdbfc2, %bfc2_10_lr : tensor<192xf32>
    %bfc2_10n = stablehlo.subtract %bfc2_10, %bfc2_10_st : tensor<192xf32>
    %g1_11_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %g1_11_st = stablehlo.multiply %vitb11_1dg, %g1_11_lr : tensor<192xf32>
    %g1_11n = stablehlo.subtract %g1_11, %g1_11_st : tensor<192xf32>
    %b1_11_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %b1_11_st = stablehlo.multiply %vitb11_1db, %b1_11_lr : tensor<192xf32>
    %b1_11n = stablehlo.subtract %b1_11, %b1_11_st : tensor<192xf32>
    %Wq_11_lr = stablehlo.constant dense<0.1> : tensor<192x192xf32>
    %Wq_11_st = stablehlo.multiply %vitb11_mdWQ, %Wq_11_lr : tensor<192x192xf32>
    %Wq_11n = stablehlo.subtract %Wq_11, %Wq_11_st : tensor<192x192xf32>
    %bq_11_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %bq_11_st = stablehlo.multiply %vitb11_mdbQ, %bq_11_lr : tensor<192xf32>
    %bq_11n = stablehlo.subtract %bq_11, %bq_11_st : tensor<192xf32>
    %Wk_11_lr = stablehlo.constant dense<0.1> : tensor<192x192xf32>
    %Wk_11_st = stablehlo.multiply %vitb11_mdWK, %Wk_11_lr : tensor<192x192xf32>
    %Wk_11n = stablehlo.subtract %Wk_11, %Wk_11_st : tensor<192x192xf32>
    %bk_11_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %bk_11_st = stablehlo.multiply %vitb11_mdbK, %bk_11_lr : tensor<192xf32>
    %bk_11n = stablehlo.subtract %bk_11, %bk_11_st : tensor<192xf32>
    %Wv_11_lr = stablehlo.constant dense<0.1> : tensor<192x192xf32>
    %Wv_11_st = stablehlo.multiply %vitb11_mdWV, %Wv_11_lr : tensor<192x192xf32>
    %Wv_11n = stablehlo.subtract %Wv_11, %Wv_11_st : tensor<192x192xf32>
    %bv_11_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %bv_11_st = stablehlo.multiply %vitb11_mdbV, %bv_11_lr : tensor<192xf32>
    %bv_11n = stablehlo.subtract %bv_11, %bv_11_st : tensor<192xf32>
    %Wo_11_lr = stablehlo.constant dense<0.1> : tensor<192x192xf32>
    %Wo_11_st = stablehlo.multiply %vitb11_mdWo, %Wo_11_lr : tensor<192x192xf32>
    %Wo_11n = stablehlo.subtract %Wo_11, %Wo_11_st : tensor<192x192xf32>
    %bo_11_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %bo_11_st = stablehlo.multiply %vitb11_mdbo, %bo_11_lr : tensor<192xf32>
    %bo_11n = stablehlo.subtract %bo_11, %bo_11_st : tensor<192xf32>
    %g2_11_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %g2_11_st = stablehlo.multiply %vitb11_2dg, %g2_11_lr : tensor<192xf32>
    %g2_11n = stablehlo.subtract %g2_11, %g2_11_st : tensor<192xf32>
    %b2_11_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %b2_11_st = stablehlo.multiply %vitb11_2db, %b2_11_lr : tensor<192xf32>
    %b2_11n = stablehlo.subtract %b2_11, %b2_11_st : tensor<192xf32>
    %Wfc1_11_lr = stablehlo.constant dense<0.1> : tensor<192x768xf32>
    %Wfc1_11_st = stablehlo.multiply %vitb11_pdWfc1, %Wfc1_11_lr : tensor<192x768xf32>
    %Wfc1_11n = stablehlo.subtract %Wfc1_11, %Wfc1_11_st : tensor<192x768xf32>
    %bfc1_11_lr = stablehlo.constant dense<0.1> : tensor<768xf32>
    %bfc1_11_st = stablehlo.multiply %vitb11_pdbfc1, %bfc1_11_lr : tensor<768xf32>
    %bfc1_11n = stablehlo.subtract %bfc1_11, %bfc1_11_st : tensor<768xf32>
    %Wfc2_11_lr = stablehlo.constant dense<0.1> : tensor<768x192xf32>
    %Wfc2_11_st = stablehlo.multiply %vitb11_pdWfc2, %Wfc2_11_lr : tensor<768x192xf32>
    %Wfc2_11n = stablehlo.subtract %Wfc2_11, %Wfc2_11_st : tensor<768x192xf32>
    %bfc2_11_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %bfc2_11_st = stablehlo.multiply %vitb11_pdbfc2, %bfc2_11_lr : tensor<192xf32>
    %bfc2_11n = stablehlo.subtract %bfc2_11, %bfc2_11_st : tensor<192xf32>
    %gF_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %gF_st = stablehlo.multiply %vitflndg, %gF_lr : tensor<192xf32>
    %gFn = stablehlo.subtract %gF, %gF_st : tensor<192xf32>
    %bF_lr = stablehlo.constant dense<0.1> : tensor<192xf32>
    %bF_st = stablehlo.multiply %vitflndb, %bF_lr : tensor<192xf32>
    %bFn = stablehlo.subtract %bF, %bF_st : tensor<192xf32>
    %Wc_lr = stablehlo.constant dense<0.1> : tensor<192x10xf32>
    %Wc_st = stablehlo.multiply %vithddWc, %Wc_lr : tensor<192x10xf32>
    %Wcn = stablehlo.subtract %Wc, %Wc_st : tensor<192x10xf32>
    %bc_lr = stablehlo.constant dense<0.1> : tensor<10xf32>
    %bc_st = stablehlo.multiply %vithddbc, %bc_lr : tensor<10xf32>
    %bcn = stablehlo.subtract %bc, %bc_st : tensor<10xf32>
    return %wConvn, %bConvn, %clsn, %posn, %g1_0n, %b1_0n, %Wq_0n, %bq_0n, %Wk_0n, %bk_0n, %Wv_0n, %bv_0n, %Wo_0n, %bo_0n, %g2_0n, %b2_0n, %Wfc1_0n, %bfc1_0n, %Wfc2_0n, %bfc2_0n, %g1_1n, %b1_1n, %Wq_1n, %bq_1n, %Wk_1n, %bk_1n, %Wv_1n, %bv_1n, %Wo_1n, %bo_1n, %g2_1n, %b2_1n, %Wfc1_1n, %bfc1_1n, %Wfc2_1n, %bfc2_1n, %g1_2n, %b1_2n, %Wq_2n, %bq_2n, %Wk_2n, %bk_2n, %Wv_2n, %bv_2n, %Wo_2n, %bo_2n, %g2_2n, %b2_2n, %Wfc1_2n, %bfc1_2n, %Wfc2_2n, %bfc2_2n, %g1_3n, %b1_3n, %Wq_3n, %bq_3n, %Wk_3n, %bk_3n, %Wv_3n, %bv_3n, %Wo_3n, %bo_3n, %g2_3n, %b2_3n, %Wfc1_3n, %bfc1_3n, %Wfc2_3n, %bfc2_3n, %g1_4n, %b1_4n, %Wq_4n, %bq_4n, %Wk_4n, %bk_4n, %Wv_4n, %bv_4n, %Wo_4n, %bo_4n, %g2_4n, %b2_4n, %Wfc1_4n, %bfc1_4n, %Wfc2_4n, %bfc2_4n, %g1_5n, %b1_5n, %Wq_5n, %bq_5n, %Wk_5n, %bk_5n, %Wv_5n, %bv_5n, %Wo_5n, %bo_5n, %g2_5n, %b2_5n, %Wfc1_5n, %bfc1_5n, %Wfc2_5n, %bfc2_5n, %g1_6n, %b1_6n, %Wq_6n, %bq_6n, %Wk_6n, %bk_6n, %Wv_6n, %bv_6n, %Wo_6n, %bo_6n, %g2_6n, %b2_6n, %Wfc1_6n, %bfc1_6n, %Wfc2_6n, %bfc2_6n, %g1_7n, %b1_7n, %Wq_7n, %bq_7n, %Wk_7n, %bk_7n, %Wv_7n, %bv_7n, %Wo_7n, %bo_7n, %g2_7n, %b2_7n, %Wfc1_7n, %bfc1_7n, %Wfc2_7n, %bfc2_7n, %g1_8n, %b1_8n, %Wq_8n, %bq_8n, %Wk_8n, %bk_8n, %Wv_8n, %bv_8n, %Wo_8n, %bo_8n, %g2_8n, %b2_8n, %Wfc1_8n, %bfc1_8n, %Wfc2_8n, %bfc2_8n, %g1_9n, %b1_9n, %Wq_9n, %bq_9n, %Wk_9n, %bk_9n, %Wv_9n, %bv_9n, %Wo_9n, %bo_9n, %g2_9n, %b2_9n, %Wfc1_9n, %bfc1_9n, %Wfc2_9n, %bfc2_9n, %g1_10n, %b1_10n, %Wq_10n, %bq_10n, %Wk_10n, %bk_10n, %Wv_10n, %bv_10n, %Wo_10n, %bo_10n, %g2_10n, %b2_10n, %Wfc1_10n, %bfc1_10n, %Wfc2_10n, %bfc2_10n, %g1_11n, %b1_11n, %Wq_11n, %bq_11n, %Wk_11n, %bk_11n, %Wv_11n, %bv_11n, %Wo_11n, %bo_11n, %g2_11n, %b2_11n, %Wfc1_11n, %bfc1_11n, %Wfc2_11n, %bfc2_11n, %gFn, %bFn, %Wcn, %bcn : tensor<192x3x16x16xf32>, tensor<192xf32>, tensor<1x192xf32>, tensor<197x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x10xf32>, tensor<10xf32>
  }
}

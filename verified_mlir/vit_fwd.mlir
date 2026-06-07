module @m {
  func.func @vit_fwd(%x: tensor<32x150528xf32>, %wConv: tensor<192x3x16x16xf32>, %bConv: tensor<192xf32>, %cls: tensor<1x192xf32>, %pos: tensor<197x192xf32>, %g1_0: tensor<192xf32>, %b1_0: tensor<192xf32>, %Wq_0: tensor<192x192xf32>, %bq_0: tensor<192xf32>, %Wk_0: tensor<192x192xf32>, %bk_0: tensor<192xf32>, %Wv_0: tensor<192x192xf32>, %bv_0: tensor<192xf32>, %Wo_0: tensor<192x192xf32>, %bo_0: tensor<192xf32>, %g2_0: tensor<192xf32>, %b2_0: tensor<192xf32>, %Wfc1_0: tensor<192x768xf32>, %bfc1_0: tensor<768xf32>, %Wfc2_0: tensor<768x192xf32>, %bfc2_0: tensor<192xf32>, %g1_1: tensor<192xf32>, %b1_1: tensor<192xf32>, %Wq_1: tensor<192x192xf32>, %bq_1: tensor<192xf32>, %Wk_1: tensor<192x192xf32>, %bk_1: tensor<192xf32>, %Wv_1: tensor<192x192xf32>, %bv_1: tensor<192xf32>, %Wo_1: tensor<192x192xf32>, %bo_1: tensor<192xf32>, %g2_1: tensor<192xf32>, %b2_1: tensor<192xf32>, %Wfc1_1: tensor<192x768xf32>, %bfc1_1: tensor<768xf32>, %Wfc2_1: tensor<768x192xf32>, %bfc2_1: tensor<192xf32>, %g1_2: tensor<192xf32>, %b1_2: tensor<192xf32>, %Wq_2: tensor<192x192xf32>, %bq_2: tensor<192xf32>, %Wk_2: tensor<192x192xf32>, %bk_2: tensor<192xf32>, %Wv_2: tensor<192x192xf32>, %bv_2: tensor<192xf32>, %Wo_2: tensor<192x192xf32>, %bo_2: tensor<192xf32>, %g2_2: tensor<192xf32>, %b2_2: tensor<192xf32>, %Wfc1_2: tensor<192x768xf32>, %bfc1_2: tensor<768xf32>, %Wfc2_2: tensor<768x192xf32>, %bfc2_2: tensor<192xf32>, %g1_3: tensor<192xf32>, %b1_3: tensor<192xf32>, %Wq_3: tensor<192x192xf32>, %bq_3: tensor<192xf32>, %Wk_3: tensor<192x192xf32>, %bk_3: tensor<192xf32>, %Wv_3: tensor<192x192xf32>, %bv_3: tensor<192xf32>, %Wo_3: tensor<192x192xf32>, %bo_3: tensor<192xf32>, %g2_3: tensor<192xf32>, %b2_3: tensor<192xf32>, %Wfc1_3: tensor<192x768xf32>, %bfc1_3: tensor<768xf32>, %Wfc2_3: tensor<768x192xf32>, %bfc2_3: tensor<192xf32>, %g1_4: tensor<192xf32>, %b1_4: tensor<192xf32>, %Wq_4: tensor<192x192xf32>, %bq_4: tensor<192xf32>, %Wk_4: tensor<192x192xf32>, %bk_4: tensor<192xf32>, %Wv_4: tensor<192x192xf32>, %bv_4: tensor<192xf32>, %Wo_4: tensor<192x192xf32>, %bo_4: tensor<192xf32>, %g2_4: tensor<192xf32>, %b2_4: tensor<192xf32>, %Wfc1_4: tensor<192x768xf32>, %bfc1_4: tensor<768xf32>, %Wfc2_4: tensor<768x192xf32>, %bfc2_4: tensor<192xf32>, %g1_5: tensor<192xf32>, %b1_5: tensor<192xf32>, %Wq_5: tensor<192x192xf32>, %bq_5: tensor<192xf32>, %Wk_5: tensor<192x192xf32>, %bk_5: tensor<192xf32>, %Wv_5: tensor<192x192xf32>, %bv_5: tensor<192xf32>, %Wo_5: tensor<192x192xf32>, %bo_5: tensor<192xf32>, %g2_5: tensor<192xf32>, %b2_5: tensor<192xf32>, %Wfc1_5: tensor<192x768xf32>, %bfc1_5: tensor<768xf32>, %Wfc2_5: tensor<768x192xf32>, %bfc2_5: tensor<192xf32>, %g1_6: tensor<192xf32>, %b1_6: tensor<192xf32>, %Wq_6: tensor<192x192xf32>, %bq_6: tensor<192xf32>, %Wk_6: tensor<192x192xf32>, %bk_6: tensor<192xf32>, %Wv_6: tensor<192x192xf32>, %bv_6: tensor<192xf32>, %Wo_6: tensor<192x192xf32>, %bo_6: tensor<192xf32>, %g2_6: tensor<192xf32>, %b2_6: tensor<192xf32>, %Wfc1_6: tensor<192x768xf32>, %bfc1_6: tensor<768xf32>, %Wfc2_6: tensor<768x192xf32>, %bfc2_6: tensor<192xf32>, %g1_7: tensor<192xf32>, %b1_7: tensor<192xf32>, %Wq_7: tensor<192x192xf32>, %bq_7: tensor<192xf32>, %Wk_7: tensor<192x192xf32>, %bk_7: tensor<192xf32>, %Wv_7: tensor<192x192xf32>, %bv_7: tensor<192xf32>, %Wo_7: tensor<192x192xf32>, %bo_7: tensor<192xf32>, %g2_7: tensor<192xf32>, %b2_7: tensor<192xf32>, %Wfc1_7: tensor<192x768xf32>, %bfc1_7: tensor<768xf32>, %Wfc2_7: tensor<768x192xf32>, %bfc2_7: tensor<192xf32>, %g1_8: tensor<192xf32>, %b1_8: tensor<192xf32>, %Wq_8: tensor<192x192xf32>, %bq_8: tensor<192xf32>, %Wk_8: tensor<192x192xf32>, %bk_8: tensor<192xf32>, %Wv_8: tensor<192x192xf32>, %bv_8: tensor<192xf32>, %Wo_8: tensor<192x192xf32>, %bo_8: tensor<192xf32>, %g2_8: tensor<192xf32>, %b2_8: tensor<192xf32>, %Wfc1_8: tensor<192x768xf32>, %bfc1_8: tensor<768xf32>, %Wfc2_8: tensor<768x192xf32>, %bfc2_8: tensor<192xf32>, %g1_9: tensor<192xf32>, %b1_9: tensor<192xf32>, %Wq_9: tensor<192x192xf32>, %bq_9: tensor<192xf32>, %Wk_9: tensor<192x192xf32>, %bk_9: tensor<192xf32>, %Wv_9: tensor<192x192xf32>, %bv_9: tensor<192xf32>, %Wo_9: tensor<192x192xf32>, %bo_9: tensor<192xf32>, %g2_9: tensor<192xf32>, %b2_9: tensor<192xf32>, %Wfc1_9: tensor<192x768xf32>, %bfc1_9: tensor<768xf32>, %Wfc2_9: tensor<768x192xf32>, %bfc2_9: tensor<192xf32>, %g1_10: tensor<192xf32>, %b1_10: tensor<192xf32>, %Wq_10: tensor<192x192xf32>, %bq_10: tensor<192xf32>, %Wk_10: tensor<192x192xf32>, %bk_10: tensor<192xf32>, %Wv_10: tensor<192x192xf32>, %bv_10: tensor<192xf32>, %Wo_10: tensor<192x192xf32>, %bo_10: tensor<192xf32>, %g2_10: tensor<192xf32>, %b2_10: tensor<192xf32>, %Wfc1_10: tensor<192x768xf32>, %bfc1_10: tensor<768xf32>, %Wfc2_10: tensor<768x192xf32>, %bfc2_10: tensor<192xf32>, %g1_11: tensor<192xf32>, %b1_11: tensor<192xf32>, %Wq_11: tensor<192x192xf32>, %bq_11: tensor<192xf32>, %Wk_11: tensor<192x192xf32>, %bk_11: tensor<192xf32>, %Wv_11: tensor<192x192xf32>, %bv_11: tensor<192xf32>, %Wo_11: tensor<192x192xf32>, %bo_11: tensor<192xf32>, %g2_11: tensor<192xf32>, %b2_11: tensor<192xf32>, %Wfc1_11: tensor<192x768xf32>, %bfc1_11: tensor<768xf32>, %Wfc2_11: tensor<768x192xf32>, %bfc2_11: tensor<192xf32>, %gF: tensor<192xf32>, %bF: tensor<192xf32>, %Wc: tensor<192x10xf32>, %bc: tensor<10xf32>) -> tensor<32x10xf32> {
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
    return %vithdlogits : tensor<32x10xf32>
  }
}

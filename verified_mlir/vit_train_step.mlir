module @m {
  func.func @vit_train_step(%x: tensor<32x150528xf32>, %wConv: tensor<192x3x16x16xf32>, %bConv: tensor<192xf32>, %cls: tensor<192xf32>, %pos: tensor<197x192xf32>, %b0_g1: tensor<192xf32>, %b0_bt1: tensor<192xf32>, %b0_Wq: tensor<192x192xf32>, %b0_bq: tensor<192xf32>, %b0_Wk: tensor<192x192xf32>, %b0_bk: tensor<192xf32>, %b0_Wv: tensor<192x192xf32>, %b0_bv: tensor<192xf32>, %b0_Wo: tensor<192x192xf32>, %b0_bo: tensor<192xf32>, %b0_g2: tensor<192xf32>, %b0_bt2: tensor<192xf32>, %b0_Wfc1: tensor<192x768xf32>, %b0_bfc1: tensor<768xf32>, %b0_Wfc2: tensor<768x192xf32>, %b0_bfc2: tensor<192xf32>, %b1_g1: tensor<192xf32>, %b1_bt1: tensor<192xf32>, %b1_Wq: tensor<192x192xf32>, %b1_bq: tensor<192xf32>, %b1_Wk: tensor<192x192xf32>, %b1_bk: tensor<192xf32>, %b1_Wv: tensor<192x192xf32>, %b1_bv: tensor<192xf32>, %b1_Wo: tensor<192x192xf32>, %b1_bo: tensor<192xf32>, %b1_g2: tensor<192xf32>, %b1_bt2: tensor<192xf32>, %b1_Wfc1: tensor<192x768xf32>, %b1_bfc1: tensor<768xf32>, %b1_Wfc2: tensor<768x192xf32>, %b1_bfc2: tensor<192xf32>, %b2_g1: tensor<192xf32>, %b2_bt1: tensor<192xf32>, %b2_Wq: tensor<192x192xf32>, %b2_bq: tensor<192xf32>, %b2_Wk: tensor<192x192xf32>, %b2_bk: tensor<192xf32>, %b2_Wv: tensor<192x192xf32>, %b2_bv: tensor<192xf32>, %b2_Wo: tensor<192x192xf32>, %b2_bo: tensor<192xf32>, %b2_g2: tensor<192xf32>, %b2_bt2: tensor<192xf32>, %b2_Wfc1: tensor<192x768xf32>, %b2_bfc1: tensor<768xf32>, %b2_Wfc2: tensor<768x192xf32>, %b2_bfc2: tensor<192xf32>, %b3_g1: tensor<192xf32>, %b3_bt1: tensor<192xf32>, %b3_Wq: tensor<192x192xf32>, %b3_bq: tensor<192xf32>, %b3_Wk: tensor<192x192xf32>, %b3_bk: tensor<192xf32>, %b3_Wv: tensor<192x192xf32>, %b3_bv: tensor<192xf32>, %b3_Wo: tensor<192x192xf32>, %b3_bo: tensor<192xf32>, %b3_g2: tensor<192xf32>, %b3_bt2: tensor<192xf32>, %b3_Wfc1: tensor<192x768xf32>, %b3_bfc1: tensor<768xf32>, %b3_Wfc2: tensor<768x192xf32>, %b3_bfc2: tensor<192xf32>, %b4_g1: tensor<192xf32>, %b4_bt1: tensor<192xf32>, %b4_Wq: tensor<192x192xf32>, %b4_bq: tensor<192xf32>, %b4_Wk: tensor<192x192xf32>, %b4_bk: tensor<192xf32>, %b4_Wv: tensor<192x192xf32>, %b4_bv: tensor<192xf32>, %b4_Wo: tensor<192x192xf32>, %b4_bo: tensor<192xf32>, %b4_g2: tensor<192xf32>, %b4_bt2: tensor<192xf32>, %b4_Wfc1: tensor<192x768xf32>, %b4_bfc1: tensor<768xf32>, %b4_Wfc2: tensor<768x192xf32>, %b4_bfc2: tensor<192xf32>, %b5_g1: tensor<192xf32>, %b5_bt1: tensor<192xf32>, %b5_Wq: tensor<192x192xf32>, %b5_bq: tensor<192xf32>, %b5_Wk: tensor<192x192xf32>, %b5_bk: tensor<192xf32>, %b5_Wv: tensor<192x192xf32>, %b5_bv: tensor<192xf32>, %b5_Wo: tensor<192x192xf32>, %b5_bo: tensor<192xf32>, %b5_g2: tensor<192xf32>, %b5_bt2: tensor<192xf32>, %b5_Wfc1: tensor<192x768xf32>, %b5_bfc1: tensor<768xf32>, %b5_Wfc2: tensor<768x192xf32>, %b5_bfc2: tensor<192xf32>, %b6_g1: tensor<192xf32>, %b6_bt1: tensor<192xf32>, %b6_Wq: tensor<192x192xf32>, %b6_bq: tensor<192xf32>, %b6_Wk: tensor<192x192xf32>, %b6_bk: tensor<192xf32>, %b6_Wv: tensor<192x192xf32>, %b6_bv: tensor<192xf32>, %b6_Wo: tensor<192x192xf32>, %b6_bo: tensor<192xf32>, %b6_g2: tensor<192xf32>, %b6_bt2: tensor<192xf32>, %b6_Wfc1: tensor<192x768xf32>, %b6_bfc1: tensor<768xf32>, %b6_Wfc2: tensor<768x192xf32>, %b6_bfc2: tensor<192xf32>, %b7_g1: tensor<192xf32>, %b7_bt1: tensor<192xf32>, %b7_Wq: tensor<192x192xf32>, %b7_bq: tensor<192xf32>, %b7_Wk: tensor<192x192xf32>, %b7_bk: tensor<192xf32>, %b7_Wv: tensor<192x192xf32>, %b7_bv: tensor<192xf32>, %b7_Wo: tensor<192x192xf32>, %b7_bo: tensor<192xf32>, %b7_g2: tensor<192xf32>, %b7_bt2: tensor<192xf32>, %b7_Wfc1: tensor<192x768xf32>, %b7_bfc1: tensor<768xf32>, %b7_Wfc2: tensor<768x192xf32>, %b7_bfc2: tensor<192xf32>, %b8_g1: tensor<192xf32>, %b8_bt1: tensor<192xf32>, %b8_Wq: tensor<192x192xf32>, %b8_bq: tensor<192xf32>, %b8_Wk: tensor<192x192xf32>, %b8_bk: tensor<192xf32>, %b8_Wv: tensor<192x192xf32>, %b8_bv: tensor<192xf32>, %b8_Wo: tensor<192x192xf32>, %b8_bo: tensor<192xf32>, %b8_g2: tensor<192xf32>, %b8_bt2: tensor<192xf32>, %b8_Wfc1: tensor<192x768xf32>, %b8_bfc1: tensor<768xf32>, %b8_Wfc2: tensor<768x192xf32>, %b8_bfc2: tensor<192xf32>, %b9_g1: tensor<192xf32>, %b9_bt1: tensor<192xf32>, %b9_Wq: tensor<192x192xf32>, %b9_bq: tensor<192xf32>, %b9_Wk: tensor<192x192xf32>, %b9_bk: tensor<192xf32>, %b9_Wv: tensor<192x192xf32>, %b9_bv: tensor<192xf32>, %b9_Wo: tensor<192x192xf32>, %b9_bo: tensor<192xf32>, %b9_g2: tensor<192xf32>, %b9_bt2: tensor<192xf32>, %b9_Wfc1: tensor<192x768xf32>, %b9_bfc1: tensor<768xf32>, %b9_Wfc2: tensor<768x192xf32>, %b9_bfc2: tensor<192xf32>, %b10_g1: tensor<192xf32>, %b10_bt1: tensor<192xf32>, %b10_Wq: tensor<192x192xf32>, %b10_bq: tensor<192xf32>, %b10_Wk: tensor<192x192xf32>, %b10_bk: tensor<192xf32>, %b10_Wv: tensor<192x192xf32>, %b10_bv: tensor<192xf32>, %b10_Wo: tensor<192x192xf32>, %b10_bo: tensor<192xf32>, %b10_g2: tensor<192xf32>, %b10_bt2: tensor<192xf32>, %b10_Wfc1: tensor<192x768xf32>, %b10_bfc1: tensor<768xf32>, %b10_Wfc2: tensor<768x192xf32>, %b10_bfc2: tensor<192xf32>, %b11_g1: tensor<192xf32>, %b11_bt1: tensor<192xf32>, %b11_Wq: tensor<192x192xf32>, %b11_bq: tensor<192xf32>, %b11_Wk: tensor<192x192xf32>, %b11_bk: tensor<192xf32>, %b11_Wv: tensor<192x192xf32>, %b11_bv: tensor<192xf32>, %b11_Wo: tensor<192x192xf32>, %b11_bo: tensor<192xf32>, %b11_g2: tensor<192xf32>, %b11_bt2: tensor<192xf32>, %b11_Wfc1: tensor<192x768xf32>, %b11_bfc1: tensor<768xf32>, %b11_Wfc2: tensor<768x192xf32>, %b11_bfc2: tensor<192xf32>, %gF: tensor<192xf32>, %btF: tensor<192xf32>, %Wc: tensor<192x10xf32>, %bc: tensor<10xf32>, %onehot: tensor<32x10xf32>) -> (tensor<192x3x16x16xf32>, tensor<192xf32>, tensor<192xf32>, tensor<197x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x10xf32>, tensor<10xf32>) {
    %one = stablehlo.constant dense<1.0> : tensor<f32>
    %zero = stablehlo.constant dense<0.0> : tensor<f32>
    %sc = stablehlo.constant dense<0.0> : tensor<f32>
    %ximg = stablehlo.reshape %x : (tensor<32x150528xf32>) -> tensor<32x3x224x224xf32>
    // ── ViT-Tiny depth-12 train step: every line is pretty(verified AST node) ──
    %v0 = stablehlo.reshape %x : (tensor<32x150528xf32>) -> tensor<32x3x224x224xf32>
    %v1 = stablehlo.convolution(%v0, %wConv)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [16, 16], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3x224x224xf32>, tensor<192x3x16x16xf32>) -> tensor<32x192x14x14xf32>
    %v2 = stablehlo.broadcast_in_dim %bConv, dims = [1] : (tensor<192xf32>) -> tensor<32x192x14x14xf32>
    %v3 = stablehlo.add %v1, %v2 : tensor<32x192x14x14xf32>
    %v4 = stablehlo.transpose %v3, dims = [0, 2, 3, 1] : (tensor<32x192x14x14xf32>) -> tensor<32x14x14x192xf32>
    %v5 = stablehlo.reshape %v4 : (tensor<32x14x14x192xf32>) -> tensor<32x196x192xf32>
    %v6 = stablehlo.broadcast_in_dim %cls, dims = [2] : (tensor<192xf32>) -> tensor<32x1x192xf32>
    %v7 = stablehlo.concatenate %v6, %v5, dim = 1 : (tensor<32x1x192xf32>, tensor<32x196x192xf32>) -> tensor<32x197x192xf32>
    %v8 = stablehlo.broadcast_in_dim %pos, dims = [1, 2] : (tensor<197x192xf32>) -> tensor<32x197x192xf32>
    %v9 = stablehlo.add %v7, %v8 : tensor<32x197x192xf32>
    %v10 = stablehlo.reshape %v9 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v11 = stablehlo.reshape %v10 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v12 = stablehlo.constant dense<0.0> : tensor<f32>
    %v13 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v14 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v15 = stablehlo.reduce(%v11 init: %v12) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v16 = stablehlo.broadcast_in_dim %v15, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v17 = stablehlo.divide %v16, %v13 : tensor<32x197x192xf32>
    %v18 = stablehlo.subtract %v11, %v17 : tensor<32x197x192xf32>
    %v19 = stablehlo.multiply %v18, %v18 : tensor<32x197x192xf32>
    %v20 = stablehlo.reduce(%v19 init: %v12) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v21 = stablehlo.broadcast_in_dim %v20, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v22 = stablehlo.divide %v21, %v13 : tensor<32x197x192xf32>
    %v23 = stablehlo.add %v22, %v14 : tensor<32x197x192xf32>
    %v24 = stablehlo.rsqrt %v23 : tensor<32x197x192xf32>
    %v25 = stablehlo.multiply %v18, %v24 : tensor<32x197x192xf32>
    %v26 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v27 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v28 = stablehlo.multiply %v25, %v26 : tensor<32x197x192xf32>
    %v29 = stablehlo.add %v28, %v27 : tensor<32x197x192xf32>
    %v30 = stablehlo.reshape %v29 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v31 = stablehlo.reshape %v30 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v32 = stablehlo.broadcast_in_dim %b0_g1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v33 = stablehlo.multiply %v31, %v32 : tensor<32x197x192xf32>
    %v34 = stablehlo.reshape %v33 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v35 = stablehlo.reshape %v34 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v36 = stablehlo.broadcast_in_dim %b0_bt1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v37 = stablehlo.add %v35, %v36 : tensor<32x197x192xf32>
    %v38 = stablehlo.reshape %v37 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v39 = stablehlo.reshape %v38 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v40 = stablehlo.dot_general %v39, %b0_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v41 = stablehlo.broadcast_in_dim %b0_bq, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v42 = stablehlo.add %v40, %v41 : tensor<32x197x192xf32>
    %v43 = stablehlo.reshape %v42 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v44 = stablehlo.reshape %v38 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v45 = stablehlo.dot_general %v44, %b0_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v46 = stablehlo.broadcast_in_dim %b0_bk, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v47 = stablehlo.add %v45, %v46 : tensor<32x197x192xf32>
    %v48 = stablehlo.reshape %v47 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v49 = stablehlo.reshape %v38 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v50 = stablehlo.dot_general %v49, %b0_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v51 = stablehlo.broadcast_in_dim %b0_bv, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v52 = stablehlo.add %v50, %v51 : tensor<32x197x192xf32>
    %v53 = stablehlo.reshape %v52 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v54 = stablehlo.reshape %v43 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v55 = stablehlo.slice %v54 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v56 = stablehlo.reshape %v55 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v57 = stablehlo.reshape %v48 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v58 = stablehlo.slice %v57 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v59 = stablehlo.reshape %v58 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v60 = stablehlo.reshape %v53 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v61 = stablehlo.slice %v60 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v62 = stablehlo.reshape %v61 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v63 = stablehlo.reshape %v59 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v64 = stablehlo.transpose %v63, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v65 = stablehlo.reshape %v64 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v66 = stablehlo.reshape %v56 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v67 = stablehlo.reshape %v65 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v68 = stablehlo.dot_general %v66, %v67, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v69 = stablehlo.reshape %v68 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v70 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v71 = stablehlo.multiply %v69, %v70 : tensor<32x38809xf32>
    %v72 = stablehlo.reshape %v71 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v73 = stablehlo.constant dense<0.0> : tensor<f32>
    %v74 = stablehlo.exponential %v72 : tensor<32x197x197xf32>
    %v75 = stablehlo.reduce(%v74 init: %v73) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v76 = stablehlo.broadcast_in_dim %v75, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v77 = stablehlo.divide %v74, %v76 : tensor<32x197x197xf32>
    %v78 = stablehlo.reshape %v77 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v79 = stablehlo.reshape %v78 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v80 = stablehlo.reshape %v62 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v81 = stablehlo.dot_general %v79, %v80, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v82 = stablehlo.reshape %v81 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v83 = stablehlo.reshape %v82 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v84 = stablehlo.constant dense<0.0> : tensor<f32>
    %v85 = stablehlo.pad %v83, %v84, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v86 = stablehlo.reshape %v85 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v87 = stablehlo.reshape %v43 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v88 = stablehlo.slice %v87 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v89 = stablehlo.reshape %v88 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v90 = stablehlo.reshape %v48 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v91 = stablehlo.slice %v90 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v92 = stablehlo.reshape %v91 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v93 = stablehlo.reshape %v53 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v94 = stablehlo.slice %v93 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v95 = stablehlo.reshape %v94 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v96 = stablehlo.reshape %v92 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v97 = stablehlo.transpose %v96, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v98 = stablehlo.reshape %v97 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v99 = stablehlo.reshape %v89 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v100 = stablehlo.reshape %v98 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v101 = stablehlo.dot_general %v99, %v100, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v102 = stablehlo.reshape %v101 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v103 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v104 = stablehlo.multiply %v102, %v103 : tensor<32x38809xf32>
    %v105 = stablehlo.reshape %v104 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v106 = stablehlo.constant dense<0.0> : tensor<f32>
    %v107 = stablehlo.exponential %v105 : tensor<32x197x197xf32>
    %v108 = stablehlo.reduce(%v107 init: %v106) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v109 = stablehlo.broadcast_in_dim %v108, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v110 = stablehlo.divide %v107, %v109 : tensor<32x197x197xf32>
    %v111 = stablehlo.reshape %v110 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v112 = stablehlo.reshape %v111 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v113 = stablehlo.reshape %v95 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v114 = stablehlo.dot_general %v112, %v113, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v115 = stablehlo.reshape %v114 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v116 = stablehlo.reshape %v115 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v117 = stablehlo.constant dense<0.0> : tensor<f32>
    %v118 = stablehlo.pad %v116, %v117, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v119 = stablehlo.reshape %v118 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v120 = stablehlo.add %v86, %v119 : tensor<32x37824xf32>
    %v121 = stablehlo.reshape %v43 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v122 = stablehlo.slice %v121 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v123 = stablehlo.reshape %v122 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v124 = stablehlo.reshape %v48 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v125 = stablehlo.slice %v124 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v126 = stablehlo.reshape %v125 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v127 = stablehlo.reshape %v53 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v128 = stablehlo.slice %v127 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v129 = stablehlo.reshape %v128 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v130 = stablehlo.reshape %v126 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v131 = stablehlo.transpose %v130, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v132 = stablehlo.reshape %v131 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v133 = stablehlo.reshape %v123 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v134 = stablehlo.reshape %v132 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v135 = stablehlo.dot_general %v133, %v134, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v136 = stablehlo.reshape %v135 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v137 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v138 = stablehlo.multiply %v136, %v137 : tensor<32x38809xf32>
    %v139 = stablehlo.reshape %v138 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v140 = stablehlo.constant dense<0.0> : tensor<f32>
    %v141 = stablehlo.exponential %v139 : tensor<32x197x197xf32>
    %v142 = stablehlo.reduce(%v141 init: %v140) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v143 = stablehlo.broadcast_in_dim %v142, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v144 = stablehlo.divide %v141, %v143 : tensor<32x197x197xf32>
    %v145 = stablehlo.reshape %v144 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v146 = stablehlo.reshape %v145 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v147 = stablehlo.reshape %v129 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v148 = stablehlo.dot_general %v146, %v147, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v149 = stablehlo.reshape %v148 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v150 = stablehlo.reshape %v149 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v151 = stablehlo.constant dense<0.0> : tensor<f32>
    %v152 = stablehlo.pad %v150, %v151, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v153 = stablehlo.reshape %v152 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v154 = stablehlo.add %v120, %v153 : tensor<32x37824xf32>
    %v155 = stablehlo.reshape %v154 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v156 = stablehlo.dot_general %v155, %b0_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v157 = stablehlo.broadcast_in_dim %b0_bo, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v158 = stablehlo.add %v156, %v157 : tensor<32x197x192xf32>
    %v159 = stablehlo.reshape %v158 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v160 = stablehlo.add %v10, %v159 : tensor<32x37824xf32>
    %v161 = stablehlo.reshape %v160 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v162 = stablehlo.constant dense<0.0> : tensor<f32>
    %v163 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v164 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v165 = stablehlo.reduce(%v161 init: %v162) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v166 = stablehlo.broadcast_in_dim %v165, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v167 = stablehlo.divide %v166, %v163 : tensor<32x197x192xf32>
    %v168 = stablehlo.subtract %v161, %v167 : tensor<32x197x192xf32>
    %v169 = stablehlo.multiply %v168, %v168 : tensor<32x197x192xf32>
    %v170 = stablehlo.reduce(%v169 init: %v162) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v171 = stablehlo.broadcast_in_dim %v170, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v172 = stablehlo.divide %v171, %v163 : tensor<32x197x192xf32>
    %v173 = stablehlo.add %v172, %v164 : tensor<32x197x192xf32>
    %v174 = stablehlo.rsqrt %v173 : tensor<32x197x192xf32>
    %v175 = stablehlo.multiply %v168, %v174 : tensor<32x197x192xf32>
    %v176 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v177 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v178 = stablehlo.multiply %v175, %v176 : tensor<32x197x192xf32>
    %v179 = stablehlo.add %v178, %v177 : tensor<32x197x192xf32>
    %v180 = stablehlo.reshape %v179 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v181 = stablehlo.reshape %v180 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v182 = stablehlo.broadcast_in_dim %b0_g2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v183 = stablehlo.multiply %v181, %v182 : tensor<32x197x192xf32>
    %v184 = stablehlo.reshape %v183 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v185 = stablehlo.reshape %v184 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v186 = stablehlo.broadcast_in_dim %b0_bt2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v187 = stablehlo.add %v185, %v186 : tensor<32x197x192xf32>
    %v188 = stablehlo.reshape %v187 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v189 = stablehlo.reshape %v188 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v190 = stablehlo.dot_general %v189, %b0_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x768xf32>) -> tensor<32x197x768xf32>
    %v191 = stablehlo.broadcast_in_dim %b0_bfc1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v192 = stablehlo.add %v190, %v191 : tensor<32x197x768xf32>
    %v193 = stablehlo.reshape %v192 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v194 = stablehlo.multiply %v193, %v193 : tensor<32x151296xf32>
    %v195 = stablehlo.multiply %v194, %v193 : tensor<32x151296xf32>
    %v196 = stablehlo.constant dense<0.044715> : tensor<32x151296xf32>
    %v197 = stablehlo.multiply %v196, %v195 : tensor<32x151296xf32>
    %v198 = stablehlo.add %v193, %v197 : tensor<32x151296xf32>
    %v199 = stablehlo.constant dense<0.7978845608028654> : tensor<32x151296xf32>
    %v200 = stablehlo.multiply %v199, %v198 : tensor<32x151296xf32>
    %v201 = stablehlo.tanh %v200 : tensor<32x151296xf32>
    %v202 = stablehlo.constant dense<1.0> : tensor<32x151296xf32>
    %v203 = stablehlo.add %v202, %v201 : tensor<32x151296xf32>
    %v204 = stablehlo.constant dense<0.5> : tensor<32x151296xf32>
    %v205 = stablehlo.multiply %v204, %v193 : tensor<32x151296xf32>
    %v206 = stablehlo.multiply %v205, %v203 : tensor<32x151296xf32>
    %v207 = stablehlo.reshape %v206 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v208 = stablehlo.dot_general %v207, %b0_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x192xf32>) -> tensor<32x197x192xf32>
    %v209 = stablehlo.broadcast_in_dim %b0_bfc2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v210 = stablehlo.add %v208, %v209 : tensor<32x197x192xf32>
    %v211 = stablehlo.reshape %v210 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v212 = stablehlo.add %v160, %v211 : tensor<32x37824xf32>
    %v213 = stablehlo.reshape %v212 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v214 = stablehlo.constant dense<0.0> : tensor<f32>
    %v215 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v216 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v217 = stablehlo.reduce(%v213 init: %v214) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v218 = stablehlo.broadcast_in_dim %v217, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v219 = stablehlo.divide %v218, %v215 : tensor<32x197x192xf32>
    %v220 = stablehlo.subtract %v213, %v219 : tensor<32x197x192xf32>
    %v221 = stablehlo.multiply %v220, %v220 : tensor<32x197x192xf32>
    %v222 = stablehlo.reduce(%v221 init: %v214) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v223 = stablehlo.broadcast_in_dim %v222, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v224 = stablehlo.divide %v223, %v215 : tensor<32x197x192xf32>
    %v225 = stablehlo.add %v224, %v216 : tensor<32x197x192xf32>
    %v226 = stablehlo.rsqrt %v225 : tensor<32x197x192xf32>
    %v227 = stablehlo.multiply %v220, %v226 : tensor<32x197x192xf32>
    %v228 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v229 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v230 = stablehlo.multiply %v227, %v228 : tensor<32x197x192xf32>
    %v231 = stablehlo.add %v230, %v229 : tensor<32x197x192xf32>
    %v232 = stablehlo.reshape %v231 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v233 = stablehlo.reshape %v232 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v234 = stablehlo.broadcast_in_dim %b1_g1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v235 = stablehlo.multiply %v233, %v234 : tensor<32x197x192xf32>
    %v236 = stablehlo.reshape %v235 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v237 = stablehlo.reshape %v236 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v238 = stablehlo.broadcast_in_dim %b1_bt1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v239 = stablehlo.add %v237, %v238 : tensor<32x197x192xf32>
    %v240 = stablehlo.reshape %v239 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v241 = stablehlo.reshape %v240 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v242 = stablehlo.dot_general %v241, %b1_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v243 = stablehlo.broadcast_in_dim %b1_bq, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v244 = stablehlo.add %v242, %v243 : tensor<32x197x192xf32>
    %v245 = stablehlo.reshape %v244 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v246 = stablehlo.reshape %v240 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v247 = stablehlo.dot_general %v246, %b1_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v248 = stablehlo.broadcast_in_dim %b1_bk, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v249 = stablehlo.add %v247, %v248 : tensor<32x197x192xf32>
    %v250 = stablehlo.reshape %v249 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v251 = stablehlo.reshape %v240 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v252 = stablehlo.dot_general %v251, %b1_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v253 = stablehlo.broadcast_in_dim %b1_bv, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v254 = stablehlo.add %v252, %v253 : tensor<32x197x192xf32>
    %v255 = stablehlo.reshape %v254 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v256 = stablehlo.reshape %v245 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v257 = stablehlo.slice %v256 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v258 = stablehlo.reshape %v257 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v259 = stablehlo.reshape %v250 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v260 = stablehlo.slice %v259 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v261 = stablehlo.reshape %v260 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v262 = stablehlo.reshape %v255 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v263 = stablehlo.slice %v262 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v264 = stablehlo.reshape %v263 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v265 = stablehlo.reshape %v261 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v266 = stablehlo.transpose %v265, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v267 = stablehlo.reshape %v266 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v268 = stablehlo.reshape %v258 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v269 = stablehlo.reshape %v267 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v270 = stablehlo.dot_general %v268, %v269, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v271 = stablehlo.reshape %v270 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v272 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v273 = stablehlo.multiply %v271, %v272 : tensor<32x38809xf32>
    %v274 = stablehlo.reshape %v273 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v275 = stablehlo.constant dense<0.0> : tensor<f32>
    %v276 = stablehlo.exponential %v274 : tensor<32x197x197xf32>
    %v277 = stablehlo.reduce(%v276 init: %v275) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v278 = stablehlo.broadcast_in_dim %v277, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v279 = stablehlo.divide %v276, %v278 : tensor<32x197x197xf32>
    %v280 = stablehlo.reshape %v279 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v281 = stablehlo.reshape %v280 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v282 = stablehlo.reshape %v264 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v283 = stablehlo.dot_general %v281, %v282, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v284 = stablehlo.reshape %v283 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v285 = stablehlo.reshape %v284 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v286 = stablehlo.constant dense<0.0> : tensor<f32>
    %v287 = stablehlo.pad %v285, %v286, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v288 = stablehlo.reshape %v287 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v289 = stablehlo.reshape %v245 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v290 = stablehlo.slice %v289 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v291 = stablehlo.reshape %v290 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v292 = stablehlo.reshape %v250 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v293 = stablehlo.slice %v292 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v294 = stablehlo.reshape %v293 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v295 = stablehlo.reshape %v255 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v296 = stablehlo.slice %v295 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v297 = stablehlo.reshape %v296 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v298 = stablehlo.reshape %v294 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v299 = stablehlo.transpose %v298, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v300 = stablehlo.reshape %v299 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v301 = stablehlo.reshape %v291 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v302 = stablehlo.reshape %v300 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v303 = stablehlo.dot_general %v301, %v302, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v304 = stablehlo.reshape %v303 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v305 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v306 = stablehlo.multiply %v304, %v305 : tensor<32x38809xf32>
    %v307 = stablehlo.reshape %v306 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v308 = stablehlo.constant dense<0.0> : tensor<f32>
    %v309 = stablehlo.exponential %v307 : tensor<32x197x197xf32>
    %v310 = stablehlo.reduce(%v309 init: %v308) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v311 = stablehlo.broadcast_in_dim %v310, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v312 = stablehlo.divide %v309, %v311 : tensor<32x197x197xf32>
    %v313 = stablehlo.reshape %v312 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v314 = stablehlo.reshape %v313 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v315 = stablehlo.reshape %v297 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v316 = stablehlo.dot_general %v314, %v315, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v317 = stablehlo.reshape %v316 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v318 = stablehlo.reshape %v317 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v319 = stablehlo.constant dense<0.0> : tensor<f32>
    %v320 = stablehlo.pad %v318, %v319, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v321 = stablehlo.reshape %v320 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v322 = stablehlo.add %v288, %v321 : tensor<32x37824xf32>
    %v323 = stablehlo.reshape %v245 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v324 = stablehlo.slice %v323 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v325 = stablehlo.reshape %v324 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v326 = stablehlo.reshape %v250 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v327 = stablehlo.slice %v326 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v328 = stablehlo.reshape %v327 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v329 = stablehlo.reshape %v255 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v330 = stablehlo.slice %v329 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v331 = stablehlo.reshape %v330 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v332 = stablehlo.reshape %v328 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v333 = stablehlo.transpose %v332, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v334 = stablehlo.reshape %v333 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v335 = stablehlo.reshape %v325 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v336 = stablehlo.reshape %v334 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v337 = stablehlo.dot_general %v335, %v336, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v338 = stablehlo.reshape %v337 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v339 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v340 = stablehlo.multiply %v338, %v339 : tensor<32x38809xf32>
    %v341 = stablehlo.reshape %v340 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v342 = stablehlo.constant dense<0.0> : tensor<f32>
    %v343 = stablehlo.exponential %v341 : tensor<32x197x197xf32>
    %v344 = stablehlo.reduce(%v343 init: %v342) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v345 = stablehlo.broadcast_in_dim %v344, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v346 = stablehlo.divide %v343, %v345 : tensor<32x197x197xf32>
    %v347 = stablehlo.reshape %v346 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v348 = stablehlo.reshape %v347 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v349 = stablehlo.reshape %v331 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v350 = stablehlo.dot_general %v348, %v349, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v351 = stablehlo.reshape %v350 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v352 = stablehlo.reshape %v351 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v353 = stablehlo.constant dense<0.0> : tensor<f32>
    %v354 = stablehlo.pad %v352, %v353, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v355 = stablehlo.reshape %v354 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v356 = stablehlo.add %v322, %v355 : tensor<32x37824xf32>
    %v357 = stablehlo.reshape %v356 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v358 = stablehlo.dot_general %v357, %b1_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v359 = stablehlo.broadcast_in_dim %b1_bo, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v360 = stablehlo.add %v358, %v359 : tensor<32x197x192xf32>
    %v361 = stablehlo.reshape %v360 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v362 = stablehlo.add %v212, %v361 : tensor<32x37824xf32>
    %v363 = stablehlo.reshape %v362 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v364 = stablehlo.constant dense<0.0> : tensor<f32>
    %v365 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v366 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v367 = stablehlo.reduce(%v363 init: %v364) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v368 = stablehlo.broadcast_in_dim %v367, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v369 = stablehlo.divide %v368, %v365 : tensor<32x197x192xf32>
    %v370 = stablehlo.subtract %v363, %v369 : tensor<32x197x192xf32>
    %v371 = stablehlo.multiply %v370, %v370 : tensor<32x197x192xf32>
    %v372 = stablehlo.reduce(%v371 init: %v364) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v373 = stablehlo.broadcast_in_dim %v372, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v374 = stablehlo.divide %v373, %v365 : tensor<32x197x192xf32>
    %v375 = stablehlo.add %v374, %v366 : tensor<32x197x192xf32>
    %v376 = stablehlo.rsqrt %v375 : tensor<32x197x192xf32>
    %v377 = stablehlo.multiply %v370, %v376 : tensor<32x197x192xf32>
    %v378 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v379 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v380 = stablehlo.multiply %v377, %v378 : tensor<32x197x192xf32>
    %v381 = stablehlo.add %v380, %v379 : tensor<32x197x192xf32>
    %v382 = stablehlo.reshape %v381 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v383 = stablehlo.reshape %v382 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v384 = stablehlo.broadcast_in_dim %b1_g2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v385 = stablehlo.multiply %v383, %v384 : tensor<32x197x192xf32>
    %v386 = stablehlo.reshape %v385 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v387 = stablehlo.reshape %v386 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v388 = stablehlo.broadcast_in_dim %b1_bt2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v389 = stablehlo.add %v387, %v388 : tensor<32x197x192xf32>
    %v390 = stablehlo.reshape %v389 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v391 = stablehlo.reshape %v390 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v392 = stablehlo.dot_general %v391, %b1_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x768xf32>) -> tensor<32x197x768xf32>
    %v393 = stablehlo.broadcast_in_dim %b1_bfc1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v394 = stablehlo.add %v392, %v393 : tensor<32x197x768xf32>
    %v395 = stablehlo.reshape %v394 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v396 = stablehlo.multiply %v395, %v395 : tensor<32x151296xf32>
    %v397 = stablehlo.multiply %v396, %v395 : tensor<32x151296xf32>
    %v398 = stablehlo.constant dense<0.044715> : tensor<32x151296xf32>
    %v399 = stablehlo.multiply %v398, %v397 : tensor<32x151296xf32>
    %v400 = stablehlo.add %v395, %v399 : tensor<32x151296xf32>
    %v401 = stablehlo.constant dense<0.7978845608028654> : tensor<32x151296xf32>
    %v402 = stablehlo.multiply %v401, %v400 : tensor<32x151296xf32>
    %v403 = stablehlo.tanh %v402 : tensor<32x151296xf32>
    %v404 = stablehlo.constant dense<1.0> : tensor<32x151296xf32>
    %v405 = stablehlo.add %v404, %v403 : tensor<32x151296xf32>
    %v406 = stablehlo.constant dense<0.5> : tensor<32x151296xf32>
    %v407 = stablehlo.multiply %v406, %v395 : tensor<32x151296xf32>
    %v408 = stablehlo.multiply %v407, %v405 : tensor<32x151296xf32>
    %v409 = stablehlo.reshape %v408 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v410 = stablehlo.dot_general %v409, %b1_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x192xf32>) -> tensor<32x197x192xf32>
    %v411 = stablehlo.broadcast_in_dim %b1_bfc2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v412 = stablehlo.add %v410, %v411 : tensor<32x197x192xf32>
    %v413 = stablehlo.reshape %v412 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v414 = stablehlo.add %v362, %v413 : tensor<32x37824xf32>
    %v415 = stablehlo.reshape %v414 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v416 = stablehlo.constant dense<0.0> : tensor<f32>
    %v417 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v418 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v419 = stablehlo.reduce(%v415 init: %v416) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v420 = stablehlo.broadcast_in_dim %v419, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v421 = stablehlo.divide %v420, %v417 : tensor<32x197x192xf32>
    %v422 = stablehlo.subtract %v415, %v421 : tensor<32x197x192xf32>
    %v423 = stablehlo.multiply %v422, %v422 : tensor<32x197x192xf32>
    %v424 = stablehlo.reduce(%v423 init: %v416) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v425 = stablehlo.broadcast_in_dim %v424, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v426 = stablehlo.divide %v425, %v417 : tensor<32x197x192xf32>
    %v427 = stablehlo.add %v426, %v418 : tensor<32x197x192xf32>
    %v428 = stablehlo.rsqrt %v427 : tensor<32x197x192xf32>
    %v429 = stablehlo.multiply %v422, %v428 : tensor<32x197x192xf32>
    %v430 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v431 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v432 = stablehlo.multiply %v429, %v430 : tensor<32x197x192xf32>
    %v433 = stablehlo.add %v432, %v431 : tensor<32x197x192xf32>
    %v434 = stablehlo.reshape %v433 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v435 = stablehlo.reshape %v434 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v436 = stablehlo.broadcast_in_dim %b2_g1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v437 = stablehlo.multiply %v435, %v436 : tensor<32x197x192xf32>
    %v438 = stablehlo.reshape %v437 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v439 = stablehlo.reshape %v438 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v440 = stablehlo.broadcast_in_dim %b2_bt1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v441 = stablehlo.add %v439, %v440 : tensor<32x197x192xf32>
    %v442 = stablehlo.reshape %v441 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v443 = stablehlo.reshape %v442 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v444 = stablehlo.dot_general %v443, %b2_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v445 = stablehlo.broadcast_in_dim %b2_bq, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v446 = stablehlo.add %v444, %v445 : tensor<32x197x192xf32>
    %v447 = stablehlo.reshape %v446 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v448 = stablehlo.reshape %v442 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v449 = stablehlo.dot_general %v448, %b2_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v450 = stablehlo.broadcast_in_dim %b2_bk, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v451 = stablehlo.add %v449, %v450 : tensor<32x197x192xf32>
    %v452 = stablehlo.reshape %v451 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v453 = stablehlo.reshape %v442 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v454 = stablehlo.dot_general %v453, %b2_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v455 = stablehlo.broadcast_in_dim %b2_bv, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v456 = stablehlo.add %v454, %v455 : tensor<32x197x192xf32>
    %v457 = stablehlo.reshape %v456 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v458 = stablehlo.reshape %v447 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v459 = stablehlo.slice %v458 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v460 = stablehlo.reshape %v459 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v461 = stablehlo.reshape %v452 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v462 = stablehlo.slice %v461 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v463 = stablehlo.reshape %v462 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v464 = stablehlo.reshape %v457 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v465 = stablehlo.slice %v464 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v466 = stablehlo.reshape %v465 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v467 = stablehlo.reshape %v463 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v468 = stablehlo.transpose %v467, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v469 = stablehlo.reshape %v468 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v470 = stablehlo.reshape %v460 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v471 = stablehlo.reshape %v469 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v472 = stablehlo.dot_general %v470, %v471, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v473 = stablehlo.reshape %v472 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v474 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v475 = stablehlo.multiply %v473, %v474 : tensor<32x38809xf32>
    %v476 = stablehlo.reshape %v475 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v477 = stablehlo.constant dense<0.0> : tensor<f32>
    %v478 = stablehlo.exponential %v476 : tensor<32x197x197xf32>
    %v479 = stablehlo.reduce(%v478 init: %v477) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v480 = stablehlo.broadcast_in_dim %v479, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v481 = stablehlo.divide %v478, %v480 : tensor<32x197x197xf32>
    %v482 = stablehlo.reshape %v481 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v483 = stablehlo.reshape %v482 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v484 = stablehlo.reshape %v466 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v485 = stablehlo.dot_general %v483, %v484, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v486 = stablehlo.reshape %v485 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v487 = stablehlo.reshape %v486 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v488 = stablehlo.constant dense<0.0> : tensor<f32>
    %v489 = stablehlo.pad %v487, %v488, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v490 = stablehlo.reshape %v489 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v491 = stablehlo.reshape %v447 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v492 = stablehlo.slice %v491 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v493 = stablehlo.reshape %v492 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v494 = stablehlo.reshape %v452 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v495 = stablehlo.slice %v494 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v496 = stablehlo.reshape %v495 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v497 = stablehlo.reshape %v457 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v498 = stablehlo.slice %v497 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v499 = stablehlo.reshape %v498 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v500 = stablehlo.reshape %v496 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v501 = stablehlo.transpose %v500, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v502 = stablehlo.reshape %v501 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v503 = stablehlo.reshape %v493 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v504 = stablehlo.reshape %v502 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v505 = stablehlo.dot_general %v503, %v504, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v506 = stablehlo.reshape %v505 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v507 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v508 = stablehlo.multiply %v506, %v507 : tensor<32x38809xf32>
    %v509 = stablehlo.reshape %v508 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v510 = stablehlo.constant dense<0.0> : tensor<f32>
    %v511 = stablehlo.exponential %v509 : tensor<32x197x197xf32>
    %v512 = stablehlo.reduce(%v511 init: %v510) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v513 = stablehlo.broadcast_in_dim %v512, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v514 = stablehlo.divide %v511, %v513 : tensor<32x197x197xf32>
    %v515 = stablehlo.reshape %v514 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v516 = stablehlo.reshape %v515 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v517 = stablehlo.reshape %v499 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v518 = stablehlo.dot_general %v516, %v517, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v519 = stablehlo.reshape %v518 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v520 = stablehlo.reshape %v519 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v521 = stablehlo.constant dense<0.0> : tensor<f32>
    %v522 = stablehlo.pad %v520, %v521, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v523 = stablehlo.reshape %v522 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v524 = stablehlo.add %v490, %v523 : tensor<32x37824xf32>
    %v525 = stablehlo.reshape %v447 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v526 = stablehlo.slice %v525 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v527 = stablehlo.reshape %v526 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v528 = stablehlo.reshape %v452 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v529 = stablehlo.slice %v528 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v530 = stablehlo.reshape %v529 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v531 = stablehlo.reshape %v457 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v532 = stablehlo.slice %v531 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v533 = stablehlo.reshape %v532 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v534 = stablehlo.reshape %v530 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v535 = stablehlo.transpose %v534, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v536 = stablehlo.reshape %v535 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v537 = stablehlo.reshape %v527 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v538 = stablehlo.reshape %v536 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v539 = stablehlo.dot_general %v537, %v538, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v540 = stablehlo.reshape %v539 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v541 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v542 = stablehlo.multiply %v540, %v541 : tensor<32x38809xf32>
    %v543 = stablehlo.reshape %v542 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v544 = stablehlo.constant dense<0.0> : tensor<f32>
    %v545 = stablehlo.exponential %v543 : tensor<32x197x197xf32>
    %v546 = stablehlo.reduce(%v545 init: %v544) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v547 = stablehlo.broadcast_in_dim %v546, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v548 = stablehlo.divide %v545, %v547 : tensor<32x197x197xf32>
    %v549 = stablehlo.reshape %v548 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v550 = stablehlo.reshape %v549 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v551 = stablehlo.reshape %v533 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v552 = stablehlo.dot_general %v550, %v551, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v553 = stablehlo.reshape %v552 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v554 = stablehlo.reshape %v553 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v555 = stablehlo.constant dense<0.0> : tensor<f32>
    %v556 = stablehlo.pad %v554, %v555, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v557 = stablehlo.reshape %v556 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v558 = stablehlo.add %v524, %v557 : tensor<32x37824xf32>
    %v559 = stablehlo.reshape %v558 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v560 = stablehlo.dot_general %v559, %b2_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v561 = stablehlo.broadcast_in_dim %b2_bo, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v562 = stablehlo.add %v560, %v561 : tensor<32x197x192xf32>
    %v563 = stablehlo.reshape %v562 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v564 = stablehlo.add %v414, %v563 : tensor<32x37824xf32>
    %v565 = stablehlo.reshape %v564 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v566 = stablehlo.constant dense<0.0> : tensor<f32>
    %v567 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v568 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v569 = stablehlo.reduce(%v565 init: %v566) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v570 = stablehlo.broadcast_in_dim %v569, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v571 = stablehlo.divide %v570, %v567 : tensor<32x197x192xf32>
    %v572 = stablehlo.subtract %v565, %v571 : tensor<32x197x192xf32>
    %v573 = stablehlo.multiply %v572, %v572 : tensor<32x197x192xf32>
    %v574 = stablehlo.reduce(%v573 init: %v566) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v575 = stablehlo.broadcast_in_dim %v574, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v576 = stablehlo.divide %v575, %v567 : tensor<32x197x192xf32>
    %v577 = stablehlo.add %v576, %v568 : tensor<32x197x192xf32>
    %v578 = stablehlo.rsqrt %v577 : tensor<32x197x192xf32>
    %v579 = stablehlo.multiply %v572, %v578 : tensor<32x197x192xf32>
    %v580 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v581 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v582 = stablehlo.multiply %v579, %v580 : tensor<32x197x192xf32>
    %v583 = stablehlo.add %v582, %v581 : tensor<32x197x192xf32>
    %v584 = stablehlo.reshape %v583 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v585 = stablehlo.reshape %v584 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v586 = stablehlo.broadcast_in_dim %b2_g2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v587 = stablehlo.multiply %v585, %v586 : tensor<32x197x192xf32>
    %v588 = stablehlo.reshape %v587 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v589 = stablehlo.reshape %v588 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v590 = stablehlo.broadcast_in_dim %b2_bt2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v591 = stablehlo.add %v589, %v590 : tensor<32x197x192xf32>
    %v592 = stablehlo.reshape %v591 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v593 = stablehlo.reshape %v592 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v594 = stablehlo.dot_general %v593, %b2_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x768xf32>) -> tensor<32x197x768xf32>
    %v595 = stablehlo.broadcast_in_dim %b2_bfc1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v596 = stablehlo.add %v594, %v595 : tensor<32x197x768xf32>
    %v597 = stablehlo.reshape %v596 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v598 = stablehlo.multiply %v597, %v597 : tensor<32x151296xf32>
    %v599 = stablehlo.multiply %v598, %v597 : tensor<32x151296xf32>
    %v600 = stablehlo.constant dense<0.044715> : tensor<32x151296xf32>
    %v601 = stablehlo.multiply %v600, %v599 : tensor<32x151296xf32>
    %v602 = stablehlo.add %v597, %v601 : tensor<32x151296xf32>
    %v603 = stablehlo.constant dense<0.7978845608028654> : tensor<32x151296xf32>
    %v604 = stablehlo.multiply %v603, %v602 : tensor<32x151296xf32>
    %v605 = stablehlo.tanh %v604 : tensor<32x151296xf32>
    %v606 = stablehlo.constant dense<1.0> : tensor<32x151296xf32>
    %v607 = stablehlo.add %v606, %v605 : tensor<32x151296xf32>
    %v608 = stablehlo.constant dense<0.5> : tensor<32x151296xf32>
    %v609 = stablehlo.multiply %v608, %v597 : tensor<32x151296xf32>
    %v610 = stablehlo.multiply %v609, %v607 : tensor<32x151296xf32>
    %v611 = stablehlo.reshape %v610 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v612 = stablehlo.dot_general %v611, %b2_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x192xf32>) -> tensor<32x197x192xf32>
    %v613 = stablehlo.broadcast_in_dim %b2_bfc2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v614 = stablehlo.add %v612, %v613 : tensor<32x197x192xf32>
    %v615 = stablehlo.reshape %v614 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v616 = stablehlo.add %v564, %v615 : tensor<32x37824xf32>
    %v617 = stablehlo.reshape %v616 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v618 = stablehlo.constant dense<0.0> : tensor<f32>
    %v619 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v620 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v621 = stablehlo.reduce(%v617 init: %v618) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v622 = stablehlo.broadcast_in_dim %v621, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v623 = stablehlo.divide %v622, %v619 : tensor<32x197x192xf32>
    %v624 = stablehlo.subtract %v617, %v623 : tensor<32x197x192xf32>
    %v625 = stablehlo.multiply %v624, %v624 : tensor<32x197x192xf32>
    %v626 = stablehlo.reduce(%v625 init: %v618) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v627 = stablehlo.broadcast_in_dim %v626, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v628 = stablehlo.divide %v627, %v619 : tensor<32x197x192xf32>
    %v629 = stablehlo.add %v628, %v620 : tensor<32x197x192xf32>
    %v630 = stablehlo.rsqrt %v629 : tensor<32x197x192xf32>
    %v631 = stablehlo.multiply %v624, %v630 : tensor<32x197x192xf32>
    %v632 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v633 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v634 = stablehlo.multiply %v631, %v632 : tensor<32x197x192xf32>
    %v635 = stablehlo.add %v634, %v633 : tensor<32x197x192xf32>
    %v636 = stablehlo.reshape %v635 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v637 = stablehlo.reshape %v636 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v638 = stablehlo.broadcast_in_dim %b3_g1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v639 = stablehlo.multiply %v637, %v638 : tensor<32x197x192xf32>
    %v640 = stablehlo.reshape %v639 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v641 = stablehlo.reshape %v640 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v642 = stablehlo.broadcast_in_dim %b3_bt1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v643 = stablehlo.add %v641, %v642 : tensor<32x197x192xf32>
    %v644 = stablehlo.reshape %v643 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v645 = stablehlo.reshape %v644 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v646 = stablehlo.dot_general %v645, %b3_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v647 = stablehlo.broadcast_in_dim %b3_bq, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v648 = stablehlo.add %v646, %v647 : tensor<32x197x192xf32>
    %v649 = stablehlo.reshape %v648 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v650 = stablehlo.reshape %v644 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v651 = stablehlo.dot_general %v650, %b3_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v652 = stablehlo.broadcast_in_dim %b3_bk, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v653 = stablehlo.add %v651, %v652 : tensor<32x197x192xf32>
    %v654 = stablehlo.reshape %v653 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v655 = stablehlo.reshape %v644 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v656 = stablehlo.dot_general %v655, %b3_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v657 = stablehlo.broadcast_in_dim %b3_bv, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v658 = stablehlo.add %v656, %v657 : tensor<32x197x192xf32>
    %v659 = stablehlo.reshape %v658 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v660 = stablehlo.reshape %v649 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v661 = stablehlo.slice %v660 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v662 = stablehlo.reshape %v661 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v663 = stablehlo.reshape %v654 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v664 = stablehlo.slice %v663 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v665 = stablehlo.reshape %v664 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v666 = stablehlo.reshape %v659 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v667 = stablehlo.slice %v666 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v668 = stablehlo.reshape %v667 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v669 = stablehlo.reshape %v665 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v670 = stablehlo.transpose %v669, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v671 = stablehlo.reshape %v670 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v672 = stablehlo.reshape %v662 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v673 = stablehlo.reshape %v671 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v674 = stablehlo.dot_general %v672, %v673, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v675 = stablehlo.reshape %v674 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v676 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v677 = stablehlo.multiply %v675, %v676 : tensor<32x38809xf32>
    %v678 = stablehlo.reshape %v677 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v679 = stablehlo.constant dense<0.0> : tensor<f32>
    %v680 = stablehlo.exponential %v678 : tensor<32x197x197xf32>
    %v681 = stablehlo.reduce(%v680 init: %v679) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v682 = stablehlo.broadcast_in_dim %v681, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v683 = stablehlo.divide %v680, %v682 : tensor<32x197x197xf32>
    %v684 = stablehlo.reshape %v683 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v685 = stablehlo.reshape %v684 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v686 = stablehlo.reshape %v668 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v687 = stablehlo.dot_general %v685, %v686, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v688 = stablehlo.reshape %v687 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v689 = stablehlo.reshape %v688 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v690 = stablehlo.constant dense<0.0> : tensor<f32>
    %v691 = stablehlo.pad %v689, %v690, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v692 = stablehlo.reshape %v691 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v693 = stablehlo.reshape %v649 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v694 = stablehlo.slice %v693 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v695 = stablehlo.reshape %v694 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v696 = stablehlo.reshape %v654 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v697 = stablehlo.slice %v696 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v698 = stablehlo.reshape %v697 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v699 = stablehlo.reshape %v659 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v700 = stablehlo.slice %v699 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v701 = stablehlo.reshape %v700 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v702 = stablehlo.reshape %v698 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v703 = stablehlo.transpose %v702, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v704 = stablehlo.reshape %v703 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v705 = stablehlo.reshape %v695 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v706 = stablehlo.reshape %v704 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v707 = stablehlo.dot_general %v705, %v706, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v708 = stablehlo.reshape %v707 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v709 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v710 = stablehlo.multiply %v708, %v709 : tensor<32x38809xf32>
    %v711 = stablehlo.reshape %v710 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v712 = stablehlo.constant dense<0.0> : tensor<f32>
    %v713 = stablehlo.exponential %v711 : tensor<32x197x197xf32>
    %v714 = stablehlo.reduce(%v713 init: %v712) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v715 = stablehlo.broadcast_in_dim %v714, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v716 = stablehlo.divide %v713, %v715 : tensor<32x197x197xf32>
    %v717 = stablehlo.reshape %v716 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v718 = stablehlo.reshape %v717 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v719 = stablehlo.reshape %v701 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v720 = stablehlo.dot_general %v718, %v719, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v721 = stablehlo.reshape %v720 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v722 = stablehlo.reshape %v721 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v723 = stablehlo.constant dense<0.0> : tensor<f32>
    %v724 = stablehlo.pad %v722, %v723, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v725 = stablehlo.reshape %v724 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v726 = stablehlo.add %v692, %v725 : tensor<32x37824xf32>
    %v727 = stablehlo.reshape %v649 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v728 = stablehlo.slice %v727 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v729 = stablehlo.reshape %v728 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v730 = stablehlo.reshape %v654 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v731 = stablehlo.slice %v730 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v732 = stablehlo.reshape %v731 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v733 = stablehlo.reshape %v659 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v734 = stablehlo.slice %v733 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v735 = stablehlo.reshape %v734 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v736 = stablehlo.reshape %v732 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v737 = stablehlo.transpose %v736, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v738 = stablehlo.reshape %v737 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v739 = stablehlo.reshape %v729 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v740 = stablehlo.reshape %v738 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v741 = stablehlo.dot_general %v739, %v740, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v742 = stablehlo.reshape %v741 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v743 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v744 = stablehlo.multiply %v742, %v743 : tensor<32x38809xf32>
    %v745 = stablehlo.reshape %v744 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v746 = stablehlo.constant dense<0.0> : tensor<f32>
    %v747 = stablehlo.exponential %v745 : tensor<32x197x197xf32>
    %v748 = stablehlo.reduce(%v747 init: %v746) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v749 = stablehlo.broadcast_in_dim %v748, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v750 = stablehlo.divide %v747, %v749 : tensor<32x197x197xf32>
    %v751 = stablehlo.reshape %v750 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v752 = stablehlo.reshape %v751 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v753 = stablehlo.reshape %v735 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v754 = stablehlo.dot_general %v752, %v753, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v755 = stablehlo.reshape %v754 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v756 = stablehlo.reshape %v755 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v757 = stablehlo.constant dense<0.0> : tensor<f32>
    %v758 = stablehlo.pad %v756, %v757, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v759 = stablehlo.reshape %v758 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v760 = stablehlo.add %v726, %v759 : tensor<32x37824xf32>
    %v761 = stablehlo.reshape %v760 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v762 = stablehlo.dot_general %v761, %b3_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v763 = stablehlo.broadcast_in_dim %b3_bo, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v764 = stablehlo.add %v762, %v763 : tensor<32x197x192xf32>
    %v765 = stablehlo.reshape %v764 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v766 = stablehlo.add %v616, %v765 : tensor<32x37824xf32>
    %v767 = stablehlo.reshape %v766 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v768 = stablehlo.constant dense<0.0> : tensor<f32>
    %v769 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v770 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v771 = stablehlo.reduce(%v767 init: %v768) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v772 = stablehlo.broadcast_in_dim %v771, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v773 = stablehlo.divide %v772, %v769 : tensor<32x197x192xf32>
    %v774 = stablehlo.subtract %v767, %v773 : tensor<32x197x192xf32>
    %v775 = stablehlo.multiply %v774, %v774 : tensor<32x197x192xf32>
    %v776 = stablehlo.reduce(%v775 init: %v768) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v777 = stablehlo.broadcast_in_dim %v776, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v778 = stablehlo.divide %v777, %v769 : tensor<32x197x192xf32>
    %v779 = stablehlo.add %v778, %v770 : tensor<32x197x192xf32>
    %v780 = stablehlo.rsqrt %v779 : tensor<32x197x192xf32>
    %v781 = stablehlo.multiply %v774, %v780 : tensor<32x197x192xf32>
    %v782 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v783 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v784 = stablehlo.multiply %v781, %v782 : tensor<32x197x192xf32>
    %v785 = stablehlo.add %v784, %v783 : tensor<32x197x192xf32>
    %v786 = stablehlo.reshape %v785 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v787 = stablehlo.reshape %v786 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v788 = stablehlo.broadcast_in_dim %b3_g2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v789 = stablehlo.multiply %v787, %v788 : tensor<32x197x192xf32>
    %v790 = stablehlo.reshape %v789 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v791 = stablehlo.reshape %v790 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v792 = stablehlo.broadcast_in_dim %b3_bt2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v793 = stablehlo.add %v791, %v792 : tensor<32x197x192xf32>
    %v794 = stablehlo.reshape %v793 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v795 = stablehlo.reshape %v794 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v796 = stablehlo.dot_general %v795, %b3_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x768xf32>) -> tensor<32x197x768xf32>
    %v797 = stablehlo.broadcast_in_dim %b3_bfc1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v798 = stablehlo.add %v796, %v797 : tensor<32x197x768xf32>
    %v799 = stablehlo.reshape %v798 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v800 = stablehlo.multiply %v799, %v799 : tensor<32x151296xf32>
    %v801 = stablehlo.multiply %v800, %v799 : tensor<32x151296xf32>
    %v802 = stablehlo.constant dense<0.044715> : tensor<32x151296xf32>
    %v803 = stablehlo.multiply %v802, %v801 : tensor<32x151296xf32>
    %v804 = stablehlo.add %v799, %v803 : tensor<32x151296xf32>
    %v805 = stablehlo.constant dense<0.7978845608028654> : tensor<32x151296xf32>
    %v806 = stablehlo.multiply %v805, %v804 : tensor<32x151296xf32>
    %v807 = stablehlo.tanh %v806 : tensor<32x151296xf32>
    %v808 = stablehlo.constant dense<1.0> : tensor<32x151296xf32>
    %v809 = stablehlo.add %v808, %v807 : tensor<32x151296xf32>
    %v810 = stablehlo.constant dense<0.5> : tensor<32x151296xf32>
    %v811 = stablehlo.multiply %v810, %v799 : tensor<32x151296xf32>
    %v812 = stablehlo.multiply %v811, %v809 : tensor<32x151296xf32>
    %v813 = stablehlo.reshape %v812 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v814 = stablehlo.dot_general %v813, %b3_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x192xf32>) -> tensor<32x197x192xf32>
    %v815 = stablehlo.broadcast_in_dim %b3_bfc2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v816 = stablehlo.add %v814, %v815 : tensor<32x197x192xf32>
    %v817 = stablehlo.reshape %v816 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v818 = stablehlo.add %v766, %v817 : tensor<32x37824xf32>
    %v819 = stablehlo.reshape %v818 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v820 = stablehlo.constant dense<0.0> : tensor<f32>
    %v821 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v822 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v823 = stablehlo.reduce(%v819 init: %v820) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v824 = stablehlo.broadcast_in_dim %v823, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v825 = stablehlo.divide %v824, %v821 : tensor<32x197x192xf32>
    %v826 = stablehlo.subtract %v819, %v825 : tensor<32x197x192xf32>
    %v827 = stablehlo.multiply %v826, %v826 : tensor<32x197x192xf32>
    %v828 = stablehlo.reduce(%v827 init: %v820) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v829 = stablehlo.broadcast_in_dim %v828, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v830 = stablehlo.divide %v829, %v821 : tensor<32x197x192xf32>
    %v831 = stablehlo.add %v830, %v822 : tensor<32x197x192xf32>
    %v832 = stablehlo.rsqrt %v831 : tensor<32x197x192xf32>
    %v833 = stablehlo.multiply %v826, %v832 : tensor<32x197x192xf32>
    %v834 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v835 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v836 = stablehlo.multiply %v833, %v834 : tensor<32x197x192xf32>
    %v837 = stablehlo.add %v836, %v835 : tensor<32x197x192xf32>
    %v838 = stablehlo.reshape %v837 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v839 = stablehlo.reshape %v838 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v840 = stablehlo.broadcast_in_dim %b4_g1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v841 = stablehlo.multiply %v839, %v840 : tensor<32x197x192xf32>
    %v842 = stablehlo.reshape %v841 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v843 = stablehlo.reshape %v842 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v844 = stablehlo.broadcast_in_dim %b4_bt1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v845 = stablehlo.add %v843, %v844 : tensor<32x197x192xf32>
    %v846 = stablehlo.reshape %v845 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v847 = stablehlo.reshape %v846 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v848 = stablehlo.dot_general %v847, %b4_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v849 = stablehlo.broadcast_in_dim %b4_bq, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v850 = stablehlo.add %v848, %v849 : tensor<32x197x192xf32>
    %v851 = stablehlo.reshape %v850 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v852 = stablehlo.reshape %v846 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v853 = stablehlo.dot_general %v852, %b4_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v854 = stablehlo.broadcast_in_dim %b4_bk, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v855 = stablehlo.add %v853, %v854 : tensor<32x197x192xf32>
    %v856 = stablehlo.reshape %v855 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v857 = stablehlo.reshape %v846 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v858 = stablehlo.dot_general %v857, %b4_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v859 = stablehlo.broadcast_in_dim %b4_bv, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v860 = stablehlo.add %v858, %v859 : tensor<32x197x192xf32>
    %v861 = stablehlo.reshape %v860 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v862 = stablehlo.reshape %v851 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v863 = stablehlo.slice %v862 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v864 = stablehlo.reshape %v863 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v865 = stablehlo.reshape %v856 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v866 = stablehlo.slice %v865 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v867 = stablehlo.reshape %v866 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v868 = stablehlo.reshape %v861 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v869 = stablehlo.slice %v868 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v870 = stablehlo.reshape %v869 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v871 = stablehlo.reshape %v867 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v872 = stablehlo.transpose %v871, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v873 = stablehlo.reshape %v872 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v874 = stablehlo.reshape %v864 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v875 = stablehlo.reshape %v873 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v876 = stablehlo.dot_general %v874, %v875, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v877 = stablehlo.reshape %v876 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v878 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v879 = stablehlo.multiply %v877, %v878 : tensor<32x38809xf32>
    %v880 = stablehlo.reshape %v879 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v881 = stablehlo.constant dense<0.0> : tensor<f32>
    %v882 = stablehlo.exponential %v880 : tensor<32x197x197xf32>
    %v883 = stablehlo.reduce(%v882 init: %v881) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v884 = stablehlo.broadcast_in_dim %v883, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v885 = stablehlo.divide %v882, %v884 : tensor<32x197x197xf32>
    %v886 = stablehlo.reshape %v885 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v887 = stablehlo.reshape %v886 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v888 = stablehlo.reshape %v870 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v889 = stablehlo.dot_general %v887, %v888, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v890 = stablehlo.reshape %v889 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v891 = stablehlo.reshape %v890 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v892 = stablehlo.constant dense<0.0> : tensor<f32>
    %v893 = stablehlo.pad %v891, %v892, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v894 = stablehlo.reshape %v893 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v895 = stablehlo.reshape %v851 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v896 = stablehlo.slice %v895 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v897 = stablehlo.reshape %v896 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v898 = stablehlo.reshape %v856 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v899 = stablehlo.slice %v898 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v900 = stablehlo.reshape %v899 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v901 = stablehlo.reshape %v861 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v902 = stablehlo.slice %v901 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v903 = stablehlo.reshape %v902 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v904 = stablehlo.reshape %v900 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v905 = stablehlo.transpose %v904, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v906 = stablehlo.reshape %v905 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v907 = stablehlo.reshape %v897 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v908 = stablehlo.reshape %v906 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v909 = stablehlo.dot_general %v907, %v908, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v910 = stablehlo.reshape %v909 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v911 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v912 = stablehlo.multiply %v910, %v911 : tensor<32x38809xf32>
    %v913 = stablehlo.reshape %v912 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v914 = stablehlo.constant dense<0.0> : tensor<f32>
    %v915 = stablehlo.exponential %v913 : tensor<32x197x197xf32>
    %v916 = stablehlo.reduce(%v915 init: %v914) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v917 = stablehlo.broadcast_in_dim %v916, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v918 = stablehlo.divide %v915, %v917 : tensor<32x197x197xf32>
    %v919 = stablehlo.reshape %v918 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v920 = stablehlo.reshape %v919 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v921 = stablehlo.reshape %v903 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v922 = stablehlo.dot_general %v920, %v921, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v923 = stablehlo.reshape %v922 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v924 = stablehlo.reshape %v923 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v925 = stablehlo.constant dense<0.0> : tensor<f32>
    %v926 = stablehlo.pad %v924, %v925, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v927 = stablehlo.reshape %v926 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v928 = stablehlo.add %v894, %v927 : tensor<32x37824xf32>
    %v929 = stablehlo.reshape %v851 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v930 = stablehlo.slice %v929 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v931 = stablehlo.reshape %v930 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v932 = stablehlo.reshape %v856 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v933 = stablehlo.slice %v932 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v934 = stablehlo.reshape %v933 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v935 = stablehlo.reshape %v861 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v936 = stablehlo.slice %v935 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v937 = stablehlo.reshape %v936 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v938 = stablehlo.reshape %v934 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v939 = stablehlo.transpose %v938, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v940 = stablehlo.reshape %v939 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v941 = stablehlo.reshape %v931 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v942 = stablehlo.reshape %v940 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v943 = stablehlo.dot_general %v941, %v942, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v944 = stablehlo.reshape %v943 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v945 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v946 = stablehlo.multiply %v944, %v945 : tensor<32x38809xf32>
    %v947 = stablehlo.reshape %v946 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v948 = stablehlo.constant dense<0.0> : tensor<f32>
    %v949 = stablehlo.exponential %v947 : tensor<32x197x197xf32>
    %v950 = stablehlo.reduce(%v949 init: %v948) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v951 = stablehlo.broadcast_in_dim %v950, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v952 = stablehlo.divide %v949, %v951 : tensor<32x197x197xf32>
    %v953 = stablehlo.reshape %v952 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v954 = stablehlo.reshape %v953 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v955 = stablehlo.reshape %v937 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v956 = stablehlo.dot_general %v954, %v955, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v957 = stablehlo.reshape %v956 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v958 = stablehlo.reshape %v957 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v959 = stablehlo.constant dense<0.0> : tensor<f32>
    %v960 = stablehlo.pad %v958, %v959, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v961 = stablehlo.reshape %v960 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v962 = stablehlo.add %v928, %v961 : tensor<32x37824xf32>
    %v963 = stablehlo.reshape %v962 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v964 = stablehlo.dot_general %v963, %b4_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v965 = stablehlo.broadcast_in_dim %b4_bo, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v966 = stablehlo.add %v964, %v965 : tensor<32x197x192xf32>
    %v967 = stablehlo.reshape %v966 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v968 = stablehlo.add %v818, %v967 : tensor<32x37824xf32>
    %v969 = stablehlo.reshape %v968 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v970 = stablehlo.constant dense<0.0> : tensor<f32>
    %v971 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v972 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v973 = stablehlo.reduce(%v969 init: %v970) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v974 = stablehlo.broadcast_in_dim %v973, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v975 = stablehlo.divide %v974, %v971 : tensor<32x197x192xf32>
    %v976 = stablehlo.subtract %v969, %v975 : tensor<32x197x192xf32>
    %v977 = stablehlo.multiply %v976, %v976 : tensor<32x197x192xf32>
    %v978 = stablehlo.reduce(%v977 init: %v970) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v979 = stablehlo.broadcast_in_dim %v978, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v980 = stablehlo.divide %v979, %v971 : tensor<32x197x192xf32>
    %v981 = stablehlo.add %v980, %v972 : tensor<32x197x192xf32>
    %v982 = stablehlo.rsqrt %v981 : tensor<32x197x192xf32>
    %v983 = stablehlo.multiply %v976, %v982 : tensor<32x197x192xf32>
    %v984 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v985 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v986 = stablehlo.multiply %v983, %v984 : tensor<32x197x192xf32>
    %v987 = stablehlo.add %v986, %v985 : tensor<32x197x192xf32>
    %v988 = stablehlo.reshape %v987 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v989 = stablehlo.reshape %v988 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v990 = stablehlo.broadcast_in_dim %b4_g2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v991 = stablehlo.multiply %v989, %v990 : tensor<32x197x192xf32>
    %v992 = stablehlo.reshape %v991 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v993 = stablehlo.reshape %v992 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v994 = stablehlo.broadcast_in_dim %b4_bt2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v995 = stablehlo.add %v993, %v994 : tensor<32x197x192xf32>
    %v996 = stablehlo.reshape %v995 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v997 = stablehlo.reshape %v996 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v998 = stablehlo.dot_general %v997, %b4_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x768xf32>) -> tensor<32x197x768xf32>
    %v999 = stablehlo.broadcast_in_dim %b4_bfc1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v1000 = stablehlo.add %v998, %v999 : tensor<32x197x768xf32>
    %v1001 = stablehlo.reshape %v1000 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1002 = stablehlo.multiply %v1001, %v1001 : tensor<32x151296xf32>
    %v1003 = stablehlo.multiply %v1002, %v1001 : tensor<32x151296xf32>
    %v1004 = stablehlo.constant dense<0.044715> : tensor<32x151296xf32>
    %v1005 = stablehlo.multiply %v1004, %v1003 : tensor<32x151296xf32>
    %v1006 = stablehlo.add %v1001, %v1005 : tensor<32x151296xf32>
    %v1007 = stablehlo.constant dense<0.7978845608028654> : tensor<32x151296xf32>
    %v1008 = stablehlo.multiply %v1007, %v1006 : tensor<32x151296xf32>
    %v1009 = stablehlo.tanh %v1008 : tensor<32x151296xf32>
    %v1010 = stablehlo.constant dense<1.0> : tensor<32x151296xf32>
    %v1011 = stablehlo.add %v1010, %v1009 : tensor<32x151296xf32>
    %v1012 = stablehlo.constant dense<0.5> : tensor<32x151296xf32>
    %v1013 = stablehlo.multiply %v1012, %v1001 : tensor<32x151296xf32>
    %v1014 = stablehlo.multiply %v1013, %v1011 : tensor<32x151296xf32>
    %v1015 = stablehlo.reshape %v1014 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1016 = stablehlo.dot_general %v1015, %b4_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x192xf32>) -> tensor<32x197x192xf32>
    %v1017 = stablehlo.broadcast_in_dim %b4_bfc2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1018 = stablehlo.add %v1016, %v1017 : tensor<32x197x192xf32>
    %v1019 = stablehlo.reshape %v1018 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1020 = stablehlo.add %v968, %v1019 : tensor<32x37824xf32>
    %v1021 = stablehlo.reshape %v1020 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1022 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1023 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v1024 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v1025 = stablehlo.reduce(%v1021 init: %v1022) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1026 = stablehlo.broadcast_in_dim %v1025, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v1027 = stablehlo.divide %v1026, %v1023 : tensor<32x197x192xf32>
    %v1028 = stablehlo.subtract %v1021, %v1027 : tensor<32x197x192xf32>
    %v1029 = stablehlo.multiply %v1028, %v1028 : tensor<32x197x192xf32>
    %v1030 = stablehlo.reduce(%v1029 init: %v1022) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1031 = stablehlo.broadcast_in_dim %v1030, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v1032 = stablehlo.divide %v1031, %v1023 : tensor<32x197x192xf32>
    %v1033 = stablehlo.add %v1032, %v1024 : tensor<32x197x192xf32>
    %v1034 = stablehlo.rsqrt %v1033 : tensor<32x197x192xf32>
    %v1035 = stablehlo.multiply %v1028, %v1034 : tensor<32x197x192xf32>
    %v1036 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v1037 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v1038 = stablehlo.multiply %v1035, %v1036 : tensor<32x197x192xf32>
    %v1039 = stablehlo.add %v1038, %v1037 : tensor<32x197x192xf32>
    %v1040 = stablehlo.reshape %v1039 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1041 = stablehlo.reshape %v1040 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1042 = stablehlo.broadcast_in_dim %b5_g1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1043 = stablehlo.multiply %v1041, %v1042 : tensor<32x197x192xf32>
    %v1044 = stablehlo.reshape %v1043 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1045 = stablehlo.reshape %v1044 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1046 = stablehlo.broadcast_in_dim %b5_bt1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1047 = stablehlo.add %v1045, %v1046 : tensor<32x197x192xf32>
    %v1048 = stablehlo.reshape %v1047 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1049 = stablehlo.reshape %v1048 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1050 = stablehlo.dot_general %v1049, %b5_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v1051 = stablehlo.broadcast_in_dim %b5_bq, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1052 = stablehlo.add %v1050, %v1051 : tensor<32x197x192xf32>
    %v1053 = stablehlo.reshape %v1052 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1054 = stablehlo.reshape %v1048 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1055 = stablehlo.dot_general %v1054, %b5_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v1056 = stablehlo.broadcast_in_dim %b5_bk, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1057 = stablehlo.add %v1055, %v1056 : tensor<32x197x192xf32>
    %v1058 = stablehlo.reshape %v1057 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1059 = stablehlo.reshape %v1048 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1060 = stablehlo.dot_general %v1059, %b5_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v1061 = stablehlo.broadcast_in_dim %b5_bv, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1062 = stablehlo.add %v1060, %v1061 : tensor<32x197x192xf32>
    %v1063 = stablehlo.reshape %v1062 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1064 = stablehlo.reshape %v1053 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1065 = stablehlo.slice %v1064 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1066 = stablehlo.reshape %v1065 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1067 = stablehlo.reshape %v1058 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1068 = stablehlo.slice %v1067 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1069 = stablehlo.reshape %v1068 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1070 = stablehlo.reshape %v1063 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1071 = stablehlo.slice %v1070 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1072 = stablehlo.reshape %v1071 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1073 = stablehlo.reshape %v1069 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1074 = stablehlo.transpose %v1073, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1075 = stablehlo.reshape %v1074 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1076 = stablehlo.reshape %v1066 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1077 = stablehlo.reshape %v1075 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1078 = stablehlo.dot_general %v1076, %v1077, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1079 = stablehlo.reshape %v1078 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1080 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1081 = stablehlo.multiply %v1079, %v1080 : tensor<32x38809xf32>
    %v1082 = stablehlo.reshape %v1081 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1083 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1084 = stablehlo.exponential %v1082 : tensor<32x197x197xf32>
    %v1085 = stablehlo.reduce(%v1084 init: %v1083) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1086 = stablehlo.broadcast_in_dim %v1085, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1087 = stablehlo.divide %v1084, %v1086 : tensor<32x197x197xf32>
    %v1088 = stablehlo.reshape %v1087 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1089 = stablehlo.reshape %v1088 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1090 = stablehlo.reshape %v1072 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1091 = stablehlo.dot_general %v1089, %v1090, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1092 = stablehlo.reshape %v1091 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1093 = stablehlo.reshape %v1092 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1094 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1095 = stablehlo.pad %v1093, %v1094, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v1096 = stablehlo.reshape %v1095 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1097 = stablehlo.reshape %v1053 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1098 = stablehlo.slice %v1097 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1099 = stablehlo.reshape %v1098 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1100 = stablehlo.reshape %v1058 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1101 = stablehlo.slice %v1100 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1102 = stablehlo.reshape %v1101 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1103 = stablehlo.reshape %v1063 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1104 = stablehlo.slice %v1103 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1105 = stablehlo.reshape %v1104 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1106 = stablehlo.reshape %v1102 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1107 = stablehlo.transpose %v1106, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1108 = stablehlo.reshape %v1107 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1109 = stablehlo.reshape %v1099 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1110 = stablehlo.reshape %v1108 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1111 = stablehlo.dot_general %v1109, %v1110, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1112 = stablehlo.reshape %v1111 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1113 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1114 = stablehlo.multiply %v1112, %v1113 : tensor<32x38809xf32>
    %v1115 = stablehlo.reshape %v1114 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1116 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1117 = stablehlo.exponential %v1115 : tensor<32x197x197xf32>
    %v1118 = stablehlo.reduce(%v1117 init: %v1116) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1119 = stablehlo.broadcast_in_dim %v1118, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1120 = stablehlo.divide %v1117, %v1119 : tensor<32x197x197xf32>
    %v1121 = stablehlo.reshape %v1120 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1122 = stablehlo.reshape %v1121 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1123 = stablehlo.reshape %v1105 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1124 = stablehlo.dot_general %v1122, %v1123, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1125 = stablehlo.reshape %v1124 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1126 = stablehlo.reshape %v1125 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1127 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1128 = stablehlo.pad %v1126, %v1127, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v1129 = stablehlo.reshape %v1128 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1130 = stablehlo.add %v1096, %v1129 : tensor<32x37824xf32>
    %v1131 = stablehlo.reshape %v1053 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1132 = stablehlo.slice %v1131 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1133 = stablehlo.reshape %v1132 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1134 = stablehlo.reshape %v1058 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1135 = stablehlo.slice %v1134 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1136 = stablehlo.reshape %v1135 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1137 = stablehlo.reshape %v1063 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1138 = stablehlo.slice %v1137 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1139 = stablehlo.reshape %v1138 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1140 = stablehlo.reshape %v1136 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1141 = stablehlo.transpose %v1140, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1142 = stablehlo.reshape %v1141 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1143 = stablehlo.reshape %v1133 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1144 = stablehlo.reshape %v1142 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1145 = stablehlo.dot_general %v1143, %v1144, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1146 = stablehlo.reshape %v1145 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1147 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1148 = stablehlo.multiply %v1146, %v1147 : tensor<32x38809xf32>
    %v1149 = stablehlo.reshape %v1148 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1150 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1151 = stablehlo.exponential %v1149 : tensor<32x197x197xf32>
    %v1152 = stablehlo.reduce(%v1151 init: %v1150) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1153 = stablehlo.broadcast_in_dim %v1152, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1154 = stablehlo.divide %v1151, %v1153 : tensor<32x197x197xf32>
    %v1155 = stablehlo.reshape %v1154 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1156 = stablehlo.reshape %v1155 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1157 = stablehlo.reshape %v1139 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1158 = stablehlo.dot_general %v1156, %v1157, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1159 = stablehlo.reshape %v1158 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1160 = stablehlo.reshape %v1159 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1161 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1162 = stablehlo.pad %v1160, %v1161, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v1163 = stablehlo.reshape %v1162 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1164 = stablehlo.add %v1130, %v1163 : tensor<32x37824xf32>
    %v1165 = stablehlo.reshape %v1164 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1166 = stablehlo.dot_general %v1165, %b5_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v1167 = stablehlo.broadcast_in_dim %b5_bo, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1168 = stablehlo.add %v1166, %v1167 : tensor<32x197x192xf32>
    %v1169 = stablehlo.reshape %v1168 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1170 = stablehlo.add %v1020, %v1169 : tensor<32x37824xf32>
    %v1171 = stablehlo.reshape %v1170 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1172 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1173 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v1174 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v1175 = stablehlo.reduce(%v1171 init: %v1172) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1176 = stablehlo.broadcast_in_dim %v1175, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v1177 = stablehlo.divide %v1176, %v1173 : tensor<32x197x192xf32>
    %v1178 = stablehlo.subtract %v1171, %v1177 : tensor<32x197x192xf32>
    %v1179 = stablehlo.multiply %v1178, %v1178 : tensor<32x197x192xf32>
    %v1180 = stablehlo.reduce(%v1179 init: %v1172) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1181 = stablehlo.broadcast_in_dim %v1180, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v1182 = stablehlo.divide %v1181, %v1173 : tensor<32x197x192xf32>
    %v1183 = stablehlo.add %v1182, %v1174 : tensor<32x197x192xf32>
    %v1184 = stablehlo.rsqrt %v1183 : tensor<32x197x192xf32>
    %v1185 = stablehlo.multiply %v1178, %v1184 : tensor<32x197x192xf32>
    %v1186 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v1187 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v1188 = stablehlo.multiply %v1185, %v1186 : tensor<32x197x192xf32>
    %v1189 = stablehlo.add %v1188, %v1187 : tensor<32x197x192xf32>
    %v1190 = stablehlo.reshape %v1189 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1191 = stablehlo.reshape %v1190 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1192 = stablehlo.broadcast_in_dim %b5_g2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1193 = stablehlo.multiply %v1191, %v1192 : tensor<32x197x192xf32>
    %v1194 = stablehlo.reshape %v1193 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1195 = stablehlo.reshape %v1194 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1196 = stablehlo.broadcast_in_dim %b5_bt2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1197 = stablehlo.add %v1195, %v1196 : tensor<32x197x192xf32>
    %v1198 = stablehlo.reshape %v1197 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1199 = stablehlo.reshape %v1198 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1200 = stablehlo.dot_general %v1199, %b5_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x768xf32>) -> tensor<32x197x768xf32>
    %v1201 = stablehlo.broadcast_in_dim %b5_bfc1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v1202 = stablehlo.add %v1200, %v1201 : tensor<32x197x768xf32>
    %v1203 = stablehlo.reshape %v1202 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1204 = stablehlo.multiply %v1203, %v1203 : tensor<32x151296xf32>
    %v1205 = stablehlo.multiply %v1204, %v1203 : tensor<32x151296xf32>
    %v1206 = stablehlo.constant dense<0.044715> : tensor<32x151296xf32>
    %v1207 = stablehlo.multiply %v1206, %v1205 : tensor<32x151296xf32>
    %v1208 = stablehlo.add %v1203, %v1207 : tensor<32x151296xf32>
    %v1209 = stablehlo.constant dense<0.7978845608028654> : tensor<32x151296xf32>
    %v1210 = stablehlo.multiply %v1209, %v1208 : tensor<32x151296xf32>
    %v1211 = stablehlo.tanh %v1210 : tensor<32x151296xf32>
    %v1212 = stablehlo.constant dense<1.0> : tensor<32x151296xf32>
    %v1213 = stablehlo.add %v1212, %v1211 : tensor<32x151296xf32>
    %v1214 = stablehlo.constant dense<0.5> : tensor<32x151296xf32>
    %v1215 = stablehlo.multiply %v1214, %v1203 : tensor<32x151296xf32>
    %v1216 = stablehlo.multiply %v1215, %v1213 : tensor<32x151296xf32>
    %v1217 = stablehlo.reshape %v1216 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1218 = stablehlo.dot_general %v1217, %b5_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x192xf32>) -> tensor<32x197x192xf32>
    %v1219 = stablehlo.broadcast_in_dim %b5_bfc2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1220 = stablehlo.add %v1218, %v1219 : tensor<32x197x192xf32>
    %v1221 = stablehlo.reshape %v1220 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1222 = stablehlo.add %v1170, %v1221 : tensor<32x37824xf32>
    %v1223 = stablehlo.reshape %v1222 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1224 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1225 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v1226 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v1227 = stablehlo.reduce(%v1223 init: %v1224) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1228 = stablehlo.broadcast_in_dim %v1227, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v1229 = stablehlo.divide %v1228, %v1225 : tensor<32x197x192xf32>
    %v1230 = stablehlo.subtract %v1223, %v1229 : tensor<32x197x192xf32>
    %v1231 = stablehlo.multiply %v1230, %v1230 : tensor<32x197x192xf32>
    %v1232 = stablehlo.reduce(%v1231 init: %v1224) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1233 = stablehlo.broadcast_in_dim %v1232, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v1234 = stablehlo.divide %v1233, %v1225 : tensor<32x197x192xf32>
    %v1235 = stablehlo.add %v1234, %v1226 : tensor<32x197x192xf32>
    %v1236 = stablehlo.rsqrt %v1235 : tensor<32x197x192xf32>
    %v1237 = stablehlo.multiply %v1230, %v1236 : tensor<32x197x192xf32>
    %v1238 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v1239 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v1240 = stablehlo.multiply %v1237, %v1238 : tensor<32x197x192xf32>
    %v1241 = stablehlo.add %v1240, %v1239 : tensor<32x197x192xf32>
    %v1242 = stablehlo.reshape %v1241 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1243 = stablehlo.reshape %v1242 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1244 = stablehlo.broadcast_in_dim %b6_g1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1245 = stablehlo.multiply %v1243, %v1244 : tensor<32x197x192xf32>
    %v1246 = stablehlo.reshape %v1245 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1247 = stablehlo.reshape %v1246 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1248 = stablehlo.broadcast_in_dim %b6_bt1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1249 = stablehlo.add %v1247, %v1248 : tensor<32x197x192xf32>
    %v1250 = stablehlo.reshape %v1249 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1251 = stablehlo.reshape %v1250 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1252 = stablehlo.dot_general %v1251, %b6_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v1253 = stablehlo.broadcast_in_dim %b6_bq, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1254 = stablehlo.add %v1252, %v1253 : tensor<32x197x192xf32>
    %v1255 = stablehlo.reshape %v1254 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1256 = stablehlo.reshape %v1250 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1257 = stablehlo.dot_general %v1256, %b6_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v1258 = stablehlo.broadcast_in_dim %b6_bk, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1259 = stablehlo.add %v1257, %v1258 : tensor<32x197x192xf32>
    %v1260 = stablehlo.reshape %v1259 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1261 = stablehlo.reshape %v1250 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1262 = stablehlo.dot_general %v1261, %b6_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v1263 = stablehlo.broadcast_in_dim %b6_bv, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1264 = stablehlo.add %v1262, %v1263 : tensor<32x197x192xf32>
    %v1265 = stablehlo.reshape %v1264 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1266 = stablehlo.reshape %v1255 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1267 = stablehlo.slice %v1266 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1268 = stablehlo.reshape %v1267 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1269 = stablehlo.reshape %v1260 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1270 = stablehlo.slice %v1269 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1271 = stablehlo.reshape %v1270 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1272 = stablehlo.reshape %v1265 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1273 = stablehlo.slice %v1272 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1274 = stablehlo.reshape %v1273 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1275 = stablehlo.reshape %v1271 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1276 = stablehlo.transpose %v1275, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1277 = stablehlo.reshape %v1276 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1278 = stablehlo.reshape %v1268 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1279 = stablehlo.reshape %v1277 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1280 = stablehlo.dot_general %v1278, %v1279, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1281 = stablehlo.reshape %v1280 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1282 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1283 = stablehlo.multiply %v1281, %v1282 : tensor<32x38809xf32>
    %v1284 = stablehlo.reshape %v1283 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1285 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1286 = stablehlo.exponential %v1284 : tensor<32x197x197xf32>
    %v1287 = stablehlo.reduce(%v1286 init: %v1285) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1288 = stablehlo.broadcast_in_dim %v1287, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1289 = stablehlo.divide %v1286, %v1288 : tensor<32x197x197xf32>
    %v1290 = stablehlo.reshape %v1289 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1291 = stablehlo.reshape %v1290 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1292 = stablehlo.reshape %v1274 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1293 = stablehlo.dot_general %v1291, %v1292, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1294 = stablehlo.reshape %v1293 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1295 = stablehlo.reshape %v1294 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1296 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1297 = stablehlo.pad %v1295, %v1296, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v1298 = stablehlo.reshape %v1297 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1299 = stablehlo.reshape %v1255 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1300 = stablehlo.slice %v1299 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1301 = stablehlo.reshape %v1300 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1302 = stablehlo.reshape %v1260 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1303 = stablehlo.slice %v1302 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1304 = stablehlo.reshape %v1303 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1305 = stablehlo.reshape %v1265 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1306 = stablehlo.slice %v1305 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1307 = stablehlo.reshape %v1306 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1308 = stablehlo.reshape %v1304 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1309 = stablehlo.transpose %v1308, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1310 = stablehlo.reshape %v1309 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1311 = stablehlo.reshape %v1301 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1312 = stablehlo.reshape %v1310 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1313 = stablehlo.dot_general %v1311, %v1312, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1314 = stablehlo.reshape %v1313 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1315 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1316 = stablehlo.multiply %v1314, %v1315 : tensor<32x38809xf32>
    %v1317 = stablehlo.reshape %v1316 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1318 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1319 = stablehlo.exponential %v1317 : tensor<32x197x197xf32>
    %v1320 = stablehlo.reduce(%v1319 init: %v1318) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1321 = stablehlo.broadcast_in_dim %v1320, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1322 = stablehlo.divide %v1319, %v1321 : tensor<32x197x197xf32>
    %v1323 = stablehlo.reshape %v1322 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1324 = stablehlo.reshape %v1323 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1325 = stablehlo.reshape %v1307 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1326 = stablehlo.dot_general %v1324, %v1325, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1327 = stablehlo.reshape %v1326 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1328 = stablehlo.reshape %v1327 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1329 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1330 = stablehlo.pad %v1328, %v1329, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v1331 = stablehlo.reshape %v1330 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1332 = stablehlo.add %v1298, %v1331 : tensor<32x37824xf32>
    %v1333 = stablehlo.reshape %v1255 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1334 = stablehlo.slice %v1333 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1335 = stablehlo.reshape %v1334 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1336 = stablehlo.reshape %v1260 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1337 = stablehlo.slice %v1336 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1338 = stablehlo.reshape %v1337 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1339 = stablehlo.reshape %v1265 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1340 = stablehlo.slice %v1339 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1341 = stablehlo.reshape %v1340 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1342 = stablehlo.reshape %v1338 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1343 = stablehlo.transpose %v1342, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1344 = stablehlo.reshape %v1343 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1345 = stablehlo.reshape %v1335 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1346 = stablehlo.reshape %v1344 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1347 = stablehlo.dot_general %v1345, %v1346, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1348 = stablehlo.reshape %v1347 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1349 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1350 = stablehlo.multiply %v1348, %v1349 : tensor<32x38809xf32>
    %v1351 = stablehlo.reshape %v1350 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1352 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1353 = stablehlo.exponential %v1351 : tensor<32x197x197xf32>
    %v1354 = stablehlo.reduce(%v1353 init: %v1352) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1355 = stablehlo.broadcast_in_dim %v1354, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1356 = stablehlo.divide %v1353, %v1355 : tensor<32x197x197xf32>
    %v1357 = stablehlo.reshape %v1356 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1358 = stablehlo.reshape %v1357 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1359 = stablehlo.reshape %v1341 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1360 = stablehlo.dot_general %v1358, %v1359, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1361 = stablehlo.reshape %v1360 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1362 = stablehlo.reshape %v1361 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1363 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1364 = stablehlo.pad %v1362, %v1363, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v1365 = stablehlo.reshape %v1364 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1366 = stablehlo.add %v1332, %v1365 : tensor<32x37824xf32>
    %v1367 = stablehlo.reshape %v1366 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1368 = stablehlo.dot_general %v1367, %b6_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v1369 = stablehlo.broadcast_in_dim %b6_bo, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1370 = stablehlo.add %v1368, %v1369 : tensor<32x197x192xf32>
    %v1371 = stablehlo.reshape %v1370 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1372 = stablehlo.add %v1222, %v1371 : tensor<32x37824xf32>
    %v1373 = stablehlo.reshape %v1372 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1374 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1375 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v1376 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v1377 = stablehlo.reduce(%v1373 init: %v1374) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1378 = stablehlo.broadcast_in_dim %v1377, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v1379 = stablehlo.divide %v1378, %v1375 : tensor<32x197x192xf32>
    %v1380 = stablehlo.subtract %v1373, %v1379 : tensor<32x197x192xf32>
    %v1381 = stablehlo.multiply %v1380, %v1380 : tensor<32x197x192xf32>
    %v1382 = stablehlo.reduce(%v1381 init: %v1374) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1383 = stablehlo.broadcast_in_dim %v1382, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v1384 = stablehlo.divide %v1383, %v1375 : tensor<32x197x192xf32>
    %v1385 = stablehlo.add %v1384, %v1376 : tensor<32x197x192xf32>
    %v1386 = stablehlo.rsqrt %v1385 : tensor<32x197x192xf32>
    %v1387 = stablehlo.multiply %v1380, %v1386 : tensor<32x197x192xf32>
    %v1388 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v1389 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v1390 = stablehlo.multiply %v1387, %v1388 : tensor<32x197x192xf32>
    %v1391 = stablehlo.add %v1390, %v1389 : tensor<32x197x192xf32>
    %v1392 = stablehlo.reshape %v1391 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1393 = stablehlo.reshape %v1392 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1394 = stablehlo.broadcast_in_dim %b6_g2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1395 = stablehlo.multiply %v1393, %v1394 : tensor<32x197x192xf32>
    %v1396 = stablehlo.reshape %v1395 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1397 = stablehlo.reshape %v1396 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1398 = stablehlo.broadcast_in_dim %b6_bt2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1399 = stablehlo.add %v1397, %v1398 : tensor<32x197x192xf32>
    %v1400 = stablehlo.reshape %v1399 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1401 = stablehlo.reshape %v1400 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1402 = stablehlo.dot_general %v1401, %b6_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x768xf32>) -> tensor<32x197x768xf32>
    %v1403 = stablehlo.broadcast_in_dim %b6_bfc1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v1404 = stablehlo.add %v1402, %v1403 : tensor<32x197x768xf32>
    %v1405 = stablehlo.reshape %v1404 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1406 = stablehlo.multiply %v1405, %v1405 : tensor<32x151296xf32>
    %v1407 = stablehlo.multiply %v1406, %v1405 : tensor<32x151296xf32>
    %v1408 = stablehlo.constant dense<0.044715> : tensor<32x151296xf32>
    %v1409 = stablehlo.multiply %v1408, %v1407 : tensor<32x151296xf32>
    %v1410 = stablehlo.add %v1405, %v1409 : tensor<32x151296xf32>
    %v1411 = stablehlo.constant dense<0.7978845608028654> : tensor<32x151296xf32>
    %v1412 = stablehlo.multiply %v1411, %v1410 : tensor<32x151296xf32>
    %v1413 = stablehlo.tanh %v1412 : tensor<32x151296xf32>
    %v1414 = stablehlo.constant dense<1.0> : tensor<32x151296xf32>
    %v1415 = stablehlo.add %v1414, %v1413 : tensor<32x151296xf32>
    %v1416 = stablehlo.constant dense<0.5> : tensor<32x151296xf32>
    %v1417 = stablehlo.multiply %v1416, %v1405 : tensor<32x151296xf32>
    %v1418 = stablehlo.multiply %v1417, %v1415 : tensor<32x151296xf32>
    %v1419 = stablehlo.reshape %v1418 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1420 = stablehlo.dot_general %v1419, %b6_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x192xf32>) -> tensor<32x197x192xf32>
    %v1421 = stablehlo.broadcast_in_dim %b6_bfc2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1422 = stablehlo.add %v1420, %v1421 : tensor<32x197x192xf32>
    %v1423 = stablehlo.reshape %v1422 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1424 = stablehlo.add %v1372, %v1423 : tensor<32x37824xf32>
    %v1425 = stablehlo.reshape %v1424 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1426 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1427 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v1428 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v1429 = stablehlo.reduce(%v1425 init: %v1426) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1430 = stablehlo.broadcast_in_dim %v1429, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v1431 = stablehlo.divide %v1430, %v1427 : tensor<32x197x192xf32>
    %v1432 = stablehlo.subtract %v1425, %v1431 : tensor<32x197x192xf32>
    %v1433 = stablehlo.multiply %v1432, %v1432 : tensor<32x197x192xf32>
    %v1434 = stablehlo.reduce(%v1433 init: %v1426) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1435 = stablehlo.broadcast_in_dim %v1434, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v1436 = stablehlo.divide %v1435, %v1427 : tensor<32x197x192xf32>
    %v1437 = stablehlo.add %v1436, %v1428 : tensor<32x197x192xf32>
    %v1438 = stablehlo.rsqrt %v1437 : tensor<32x197x192xf32>
    %v1439 = stablehlo.multiply %v1432, %v1438 : tensor<32x197x192xf32>
    %v1440 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v1441 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v1442 = stablehlo.multiply %v1439, %v1440 : tensor<32x197x192xf32>
    %v1443 = stablehlo.add %v1442, %v1441 : tensor<32x197x192xf32>
    %v1444 = stablehlo.reshape %v1443 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1445 = stablehlo.reshape %v1444 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1446 = stablehlo.broadcast_in_dim %b7_g1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1447 = stablehlo.multiply %v1445, %v1446 : tensor<32x197x192xf32>
    %v1448 = stablehlo.reshape %v1447 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1449 = stablehlo.reshape %v1448 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1450 = stablehlo.broadcast_in_dim %b7_bt1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1451 = stablehlo.add %v1449, %v1450 : tensor<32x197x192xf32>
    %v1452 = stablehlo.reshape %v1451 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1453 = stablehlo.reshape %v1452 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1454 = stablehlo.dot_general %v1453, %b7_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v1455 = stablehlo.broadcast_in_dim %b7_bq, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1456 = stablehlo.add %v1454, %v1455 : tensor<32x197x192xf32>
    %v1457 = stablehlo.reshape %v1456 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1458 = stablehlo.reshape %v1452 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1459 = stablehlo.dot_general %v1458, %b7_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v1460 = stablehlo.broadcast_in_dim %b7_bk, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1461 = stablehlo.add %v1459, %v1460 : tensor<32x197x192xf32>
    %v1462 = stablehlo.reshape %v1461 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1463 = stablehlo.reshape %v1452 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1464 = stablehlo.dot_general %v1463, %b7_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v1465 = stablehlo.broadcast_in_dim %b7_bv, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1466 = stablehlo.add %v1464, %v1465 : tensor<32x197x192xf32>
    %v1467 = stablehlo.reshape %v1466 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1468 = stablehlo.reshape %v1457 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1469 = stablehlo.slice %v1468 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1470 = stablehlo.reshape %v1469 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1471 = stablehlo.reshape %v1462 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1472 = stablehlo.slice %v1471 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1473 = stablehlo.reshape %v1472 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1474 = stablehlo.reshape %v1467 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1475 = stablehlo.slice %v1474 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1476 = stablehlo.reshape %v1475 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1477 = stablehlo.reshape %v1473 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1478 = stablehlo.transpose %v1477, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1479 = stablehlo.reshape %v1478 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1480 = stablehlo.reshape %v1470 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1481 = stablehlo.reshape %v1479 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1482 = stablehlo.dot_general %v1480, %v1481, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1483 = stablehlo.reshape %v1482 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1484 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1485 = stablehlo.multiply %v1483, %v1484 : tensor<32x38809xf32>
    %v1486 = stablehlo.reshape %v1485 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1487 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1488 = stablehlo.exponential %v1486 : tensor<32x197x197xf32>
    %v1489 = stablehlo.reduce(%v1488 init: %v1487) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1490 = stablehlo.broadcast_in_dim %v1489, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1491 = stablehlo.divide %v1488, %v1490 : tensor<32x197x197xf32>
    %v1492 = stablehlo.reshape %v1491 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1493 = stablehlo.reshape %v1492 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1494 = stablehlo.reshape %v1476 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1495 = stablehlo.dot_general %v1493, %v1494, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1496 = stablehlo.reshape %v1495 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1497 = stablehlo.reshape %v1496 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1498 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1499 = stablehlo.pad %v1497, %v1498, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v1500 = stablehlo.reshape %v1499 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1501 = stablehlo.reshape %v1457 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1502 = stablehlo.slice %v1501 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1503 = stablehlo.reshape %v1502 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1504 = stablehlo.reshape %v1462 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1505 = stablehlo.slice %v1504 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1506 = stablehlo.reshape %v1505 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1507 = stablehlo.reshape %v1467 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1508 = stablehlo.slice %v1507 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1509 = stablehlo.reshape %v1508 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1510 = stablehlo.reshape %v1506 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1511 = stablehlo.transpose %v1510, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1512 = stablehlo.reshape %v1511 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1513 = stablehlo.reshape %v1503 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1514 = stablehlo.reshape %v1512 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1515 = stablehlo.dot_general %v1513, %v1514, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1516 = stablehlo.reshape %v1515 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1517 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1518 = stablehlo.multiply %v1516, %v1517 : tensor<32x38809xf32>
    %v1519 = stablehlo.reshape %v1518 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1520 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1521 = stablehlo.exponential %v1519 : tensor<32x197x197xf32>
    %v1522 = stablehlo.reduce(%v1521 init: %v1520) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1523 = stablehlo.broadcast_in_dim %v1522, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1524 = stablehlo.divide %v1521, %v1523 : tensor<32x197x197xf32>
    %v1525 = stablehlo.reshape %v1524 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1526 = stablehlo.reshape %v1525 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1527 = stablehlo.reshape %v1509 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1528 = stablehlo.dot_general %v1526, %v1527, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1529 = stablehlo.reshape %v1528 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1530 = stablehlo.reshape %v1529 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1531 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1532 = stablehlo.pad %v1530, %v1531, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v1533 = stablehlo.reshape %v1532 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1534 = stablehlo.add %v1500, %v1533 : tensor<32x37824xf32>
    %v1535 = stablehlo.reshape %v1457 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1536 = stablehlo.slice %v1535 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1537 = stablehlo.reshape %v1536 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1538 = stablehlo.reshape %v1462 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1539 = stablehlo.slice %v1538 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1540 = stablehlo.reshape %v1539 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1541 = stablehlo.reshape %v1467 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1542 = stablehlo.slice %v1541 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1543 = stablehlo.reshape %v1542 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1544 = stablehlo.reshape %v1540 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1545 = stablehlo.transpose %v1544, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1546 = stablehlo.reshape %v1545 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1547 = stablehlo.reshape %v1537 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1548 = stablehlo.reshape %v1546 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1549 = stablehlo.dot_general %v1547, %v1548, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1550 = stablehlo.reshape %v1549 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1551 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1552 = stablehlo.multiply %v1550, %v1551 : tensor<32x38809xf32>
    %v1553 = stablehlo.reshape %v1552 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1554 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1555 = stablehlo.exponential %v1553 : tensor<32x197x197xf32>
    %v1556 = stablehlo.reduce(%v1555 init: %v1554) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1557 = stablehlo.broadcast_in_dim %v1556, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1558 = stablehlo.divide %v1555, %v1557 : tensor<32x197x197xf32>
    %v1559 = stablehlo.reshape %v1558 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1560 = stablehlo.reshape %v1559 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1561 = stablehlo.reshape %v1543 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1562 = stablehlo.dot_general %v1560, %v1561, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1563 = stablehlo.reshape %v1562 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1564 = stablehlo.reshape %v1563 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1565 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1566 = stablehlo.pad %v1564, %v1565, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v1567 = stablehlo.reshape %v1566 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1568 = stablehlo.add %v1534, %v1567 : tensor<32x37824xf32>
    %v1569 = stablehlo.reshape %v1568 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1570 = stablehlo.dot_general %v1569, %b7_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v1571 = stablehlo.broadcast_in_dim %b7_bo, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1572 = stablehlo.add %v1570, %v1571 : tensor<32x197x192xf32>
    %v1573 = stablehlo.reshape %v1572 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1574 = stablehlo.add %v1424, %v1573 : tensor<32x37824xf32>
    %v1575 = stablehlo.reshape %v1574 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1576 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1577 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v1578 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v1579 = stablehlo.reduce(%v1575 init: %v1576) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1580 = stablehlo.broadcast_in_dim %v1579, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v1581 = stablehlo.divide %v1580, %v1577 : tensor<32x197x192xf32>
    %v1582 = stablehlo.subtract %v1575, %v1581 : tensor<32x197x192xf32>
    %v1583 = stablehlo.multiply %v1582, %v1582 : tensor<32x197x192xf32>
    %v1584 = stablehlo.reduce(%v1583 init: %v1576) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1585 = stablehlo.broadcast_in_dim %v1584, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v1586 = stablehlo.divide %v1585, %v1577 : tensor<32x197x192xf32>
    %v1587 = stablehlo.add %v1586, %v1578 : tensor<32x197x192xf32>
    %v1588 = stablehlo.rsqrt %v1587 : tensor<32x197x192xf32>
    %v1589 = stablehlo.multiply %v1582, %v1588 : tensor<32x197x192xf32>
    %v1590 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v1591 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v1592 = stablehlo.multiply %v1589, %v1590 : tensor<32x197x192xf32>
    %v1593 = stablehlo.add %v1592, %v1591 : tensor<32x197x192xf32>
    %v1594 = stablehlo.reshape %v1593 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1595 = stablehlo.reshape %v1594 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1596 = stablehlo.broadcast_in_dim %b7_g2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1597 = stablehlo.multiply %v1595, %v1596 : tensor<32x197x192xf32>
    %v1598 = stablehlo.reshape %v1597 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1599 = stablehlo.reshape %v1598 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1600 = stablehlo.broadcast_in_dim %b7_bt2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1601 = stablehlo.add %v1599, %v1600 : tensor<32x197x192xf32>
    %v1602 = stablehlo.reshape %v1601 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1603 = stablehlo.reshape %v1602 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1604 = stablehlo.dot_general %v1603, %b7_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x768xf32>) -> tensor<32x197x768xf32>
    %v1605 = stablehlo.broadcast_in_dim %b7_bfc1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v1606 = stablehlo.add %v1604, %v1605 : tensor<32x197x768xf32>
    %v1607 = stablehlo.reshape %v1606 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1608 = stablehlo.multiply %v1607, %v1607 : tensor<32x151296xf32>
    %v1609 = stablehlo.multiply %v1608, %v1607 : tensor<32x151296xf32>
    %v1610 = stablehlo.constant dense<0.044715> : tensor<32x151296xf32>
    %v1611 = stablehlo.multiply %v1610, %v1609 : tensor<32x151296xf32>
    %v1612 = stablehlo.add %v1607, %v1611 : tensor<32x151296xf32>
    %v1613 = stablehlo.constant dense<0.7978845608028654> : tensor<32x151296xf32>
    %v1614 = stablehlo.multiply %v1613, %v1612 : tensor<32x151296xf32>
    %v1615 = stablehlo.tanh %v1614 : tensor<32x151296xf32>
    %v1616 = stablehlo.constant dense<1.0> : tensor<32x151296xf32>
    %v1617 = stablehlo.add %v1616, %v1615 : tensor<32x151296xf32>
    %v1618 = stablehlo.constant dense<0.5> : tensor<32x151296xf32>
    %v1619 = stablehlo.multiply %v1618, %v1607 : tensor<32x151296xf32>
    %v1620 = stablehlo.multiply %v1619, %v1617 : tensor<32x151296xf32>
    %v1621 = stablehlo.reshape %v1620 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1622 = stablehlo.dot_general %v1621, %b7_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x192xf32>) -> tensor<32x197x192xf32>
    %v1623 = stablehlo.broadcast_in_dim %b7_bfc2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1624 = stablehlo.add %v1622, %v1623 : tensor<32x197x192xf32>
    %v1625 = stablehlo.reshape %v1624 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1626 = stablehlo.add %v1574, %v1625 : tensor<32x37824xf32>
    %v1627 = stablehlo.reshape %v1626 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1628 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1629 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v1630 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v1631 = stablehlo.reduce(%v1627 init: %v1628) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1632 = stablehlo.broadcast_in_dim %v1631, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v1633 = stablehlo.divide %v1632, %v1629 : tensor<32x197x192xf32>
    %v1634 = stablehlo.subtract %v1627, %v1633 : tensor<32x197x192xf32>
    %v1635 = stablehlo.multiply %v1634, %v1634 : tensor<32x197x192xf32>
    %v1636 = stablehlo.reduce(%v1635 init: %v1628) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1637 = stablehlo.broadcast_in_dim %v1636, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v1638 = stablehlo.divide %v1637, %v1629 : tensor<32x197x192xf32>
    %v1639 = stablehlo.add %v1638, %v1630 : tensor<32x197x192xf32>
    %v1640 = stablehlo.rsqrt %v1639 : tensor<32x197x192xf32>
    %v1641 = stablehlo.multiply %v1634, %v1640 : tensor<32x197x192xf32>
    %v1642 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v1643 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v1644 = stablehlo.multiply %v1641, %v1642 : tensor<32x197x192xf32>
    %v1645 = stablehlo.add %v1644, %v1643 : tensor<32x197x192xf32>
    %v1646 = stablehlo.reshape %v1645 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1647 = stablehlo.reshape %v1646 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1648 = stablehlo.broadcast_in_dim %b8_g1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1649 = stablehlo.multiply %v1647, %v1648 : tensor<32x197x192xf32>
    %v1650 = stablehlo.reshape %v1649 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1651 = stablehlo.reshape %v1650 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1652 = stablehlo.broadcast_in_dim %b8_bt1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1653 = stablehlo.add %v1651, %v1652 : tensor<32x197x192xf32>
    %v1654 = stablehlo.reshape %v1653 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1655 = stablehlo.reshape %v1654 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1656 = stablehlo.dot_general %v1655, %b8_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v1657 = stablehlo.broadcast_in_dim %b8_bq, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1658 = stablehlo.add %v1656, %v1657 : tensor<32x197x192xf32>
    %v1659 = stablehlo.reshape %v1658 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1660 = stablehlo.reshape %v1654 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1661 = stablehlo.dot_general %v1660, %b8_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v1662 = stablehlo.broadcast_in_dim %b8_bk, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1663 = stablehlo.add %v1661, %v1662 : tensor<32x197x192xf32>
    %v1664 = stablehlo.reshape %v1663 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1665 = stablehlo.reshape %v1654 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1666 = stablehlo.dot_general %v1665, %b8_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v1667 = stablehlo.broadcast_in_dim %b8_bv, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1668 = stablehlo.add %v1666, %v1667 : tensor<32x197x192xf32>
    %v1669 = stablehlo.reshape %v1668 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1670 = stablehlo.reshape %v1659 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1671 = stablehlo.slice %v1670 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1672 = stablehlo.reshape %v1671 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1673 = stablehlo.reshape %v1664 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1674 = stablehlo.slice %v1673 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1675 = stablehlo.reshape %v1674 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1676 = stablehlo.reshape %v1669 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1677 = stablehlo.slice %v1676 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1678 = stablehlo.reshape %v1677 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1679 = stablehlo.reshape %v1675 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1680 = stablehlo.transpose %v1679, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1681 = stablehlo.reshape %v1680 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1682 = stablehlo.reshape %v1672 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1683 = stablehlo.reshape %v1681 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1684 = stablehlo.dot_general %v1682, %v1683, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1685 = stablehlo.reshape %v1684 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1686 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1687 = stablehlo.multiply %v1685, %v1686 : tensor<32x38809xf32>
    %v1688 = stablehlo.reshape %v1687 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1689 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1690 = stablehlo.exponential %v1688 : tensor<32x197x197xf32>
    %v1691 = stablehlo.reduce(%v1690 init: %v1689) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1692 = stablehlo.broadcast_in_dim %v1691, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1693 = stablehlo.divide %v1690, %v1692 : tensor<32x197x197xf32>
    %v1694 = stablehlo.reshape %v1693 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1695 = stablehlo.reshape %v1694 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1696 = stablehlo.reshape %v1678 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1697 = stablehlo.dot_general %v1695, %v1696, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1698 = stablehlo.reshape %v1697 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1699 = stablehlo.reshape %v1698 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1700 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1701 = stablehlo.pad %v1699, %v1700, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v1702 = stablehlo.reshape %v1701 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1703 = stablehlo.reshape %v1659 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1704 = stablehlo.slice %v1703 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1705 = stablehlo.reshape %v1704 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1706 = stablehlo.reshape %v1664 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1707 = stablehlo.slice %v1706 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1708 = stablehlo.reshape %v1707 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1709 = stablehlo.reshape %v1669 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1710 = stablehlo.slice %v1709 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1711 = stablehlo.reshape %v1710 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1712 = stablehlo.reshape %v1708 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1713 = stablehlo.transpose %v1712, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1714 = stablehlo.reshape %v1713 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1715 = stablehlo.reshape %v1705 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1716 = stablehlo.reshape %v1714 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1717 = stablehlo.dot_general %v1715, %v1716, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1718 = stablehlo.reshape %v1717 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1719 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1720 = stablehlo.multiply %v1718, %v1719 : tensor<32x38809xf32>
    %v1721 = stablehlo.reshape %v1720 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1722 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1723 = stablehlo.exponential %v1721 : tensor<32x197x197xf32>
    %v1724 = stablehlo.reduce(%v1723 init: %v1722) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1725 = stablehlo.broadcast_in_dim %v1724, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1726 = stablehlo.divide %v1723, %v1725 : tensor<32x197x197xf32>
    %v1727 = stablehlo.reshape %v1726 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1728 = stablehlo.reshape %v1727 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1729 = stablehlo.reshape %v1711 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1730 = stablehlo.dot_general %v1728, %v1729, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1731 = stablehlo.reshape %v1730 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1732 = stablehlo.reshape %v1731 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1733 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1734 = stablehlo.pad %v1732, %v1733, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v1735 = stablehlo.reshape %v1734 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1736 = stablehlo.add %v1702, %v1735 : tensor<32x37824xf32>
    %v1737 = stablehlo.reshape %v1659 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1738 = stablehlo.slice %v1737 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1739 = stablehlo.reshape %v1738 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1740 = stablehlo.reshape %v1664 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1741 = stablehlo.slice %v1740 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1742 = stablehlo.reshape %v1741 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1743 = stablehlo.reshape %v1669 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1744 = stablehlo.slice %v1743 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1745 = stablehlo.reshape %v1744 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1746 = stablehlo.reshape %v1742 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1747 = stablehlo.transpose %v1746, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1748 = stablehlo.reshape %v1747 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1749 = stablehlo.reshape %v1739 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1750 = stablehlo.reshape %v1748 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1751 = stablehlo.dot_general %v1749, %v1750, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1752 = stablehlo.reshape %v1751 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1753 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1754 = stablehlo.multiply %v1752, %v1753 : tensor<32x38809xf32>
    %v1755 = stablehlo.reshape %v1754 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1756 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1757 = stablehlo.exponential %v1755 : tensor<32x197x197xf32>
    %v1758 = stablehlo.reduce(%v1757 init: %v1756) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1759 = stablehlo.broadcast_in_dim %v1758, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1760 = stablehlo.divide %v1757, %v1759 : tensor<32x197x197xf32>
    %v1761 = stablehlo.reshape %v1760 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1762 = stablehlo.reshape %v1761 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1763 = stablehlo.reshape %v1745 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1764 = stablehlo.dot_general %v1762, %v1763, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1765 = stablehlo.reshape %v1764 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1766 = stablehlo.reshape %v1765 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1767 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1768 = stablehlo.pad %v1766, %v1767, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v1769 = stablehlo.reshape %v1768 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1770 = stablehlo.add %v1736, %v1769 : tensor<32x37824xf32>
    %v1771 = stablehlo.reshape %v1770 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1772 = stablehlo.dot_general %v1771, %b8_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v1773 = stablehlo.broadcast_in_dim %b8_bo, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1774 = stablehlo.add %v1772, %v1773 : tensor<32x197x192xf32>
    %v1775 = stablehlo.reshape %v1774 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1776 = stablehlo.add %v1626, %v1775 : tensor<32x37824xf32>
    %v1777 = stablehlo.reshape %v1776 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1778 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1779 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v1780 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v1781 = stablehlo.reduce(%v1777 init: %v1778) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1782 = stablehlo.broadcast_in_dim %v1781, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v1783 = stablehlo.divide %v1782, %v1779 : tensor<32x197x192xf32>
    %v1784 = stablehlo.subtract %v1777, %v1783 : tensor<32x197x192xf32>
    %v1785 = stablehlo.multiply %v1784, %v1784 : tensor<32x197x192xf32>
    %v1786 = stablehlo.reduce(%v1785 init: %v1778) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1787 = stablehlo.broadcast_in_dim %v1786, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v1788 = stablehlo.divide %v1787, %v1779 : tensor<32x197x192xf32>
    %v1789 = stablehlo.add %v1788, %v1780 : tensor<32x197x192xf32>
    %v1790 = stablehlo.rsqrt %v1789 : tensor<32x197x192xf32>
    %v1791 = stablehlo.multiply %v1784, %v1790 : tensor<32x197x192xf32>
    %v1792 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v1793 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v1794 = stablehlo.multiply %v1791, %v1792 : tensor<32x197x192xf32>
    %v1795 = stablehlo.add %v1794, %v1793 : tensor<32x197x192xf32>
    %v1796 = stablehlo.reshape %v1795 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1797 = stablehlo.reshape %v1796 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1798 = stablehlo.broadcast_in_dim %b8_g2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1799 = stablehlo.multiply %v1797, %v1798 : tensor<32x197x192xf32>
    %v1800 = stablehlo.reshape %v1799 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1801 = stablehlo.reshape %v1800 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1802 = stablehlo.broadcast_in_dim %b8_bt2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1803 = stablehlo.add %v1801, %v1802 : tensor<32x197x192xf32>
    %v1804 = stablehlo.reshape %v1803 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1805 = stablehlo.reshape %v1804 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1806 = stablehlo.dot_general %v1805, %b8_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x768xf32>) -> tensor<32x197x768xf32>
    %v1807 = stablehlo.broadcast_in_dim %b8_bfc1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v1808 = stablehlo.add %v1806, %v1807 : tensor<32x197x768xf32>
    %v1809 = stablehlo.reshape %v1808 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1810 = stablehlo.multiply %v1809, %v1809 : tensor<32x151296xf32>
    %v1811 = stablehlo.multiply %v1810, %v1809 : tensor<32x151296xf32>
    %v1812 = stablehlo.constant dense<0.044715> : tensor<32x151296xf32>
    %v1813 = stablehlo.multiply %v1812, %v1811 : tensor<32x151296xf32>
    %v1814 = stablehlo.add %v1809, %v1813 : tensor<32x151296xf32>
    %v1815 = stablehlo.constant dense<0.7978845608028654> : tensor<32x151296xf32>
    %v1816 = stablehlo.multiply %v1815, %v1814 : tensor<32x151296xf32>
    %v1817 = stablehlo.tanh %v1816 : tensor<32x151296xf32>
    %v1818 = stablehlo.constant dense<1.0> : tensor<32x151296xf32>
    %v1819 = stablehlo.add %v1818, %v1817 : tensor<32x151296xf32>
    %v1820 = stablehlo.constant dense<0.5> : tensor<32x151296xf32>
    %v1821 = stablehlo.multiply %v1820, %v1809 : tensor<32x151296xf32>
    %v1822 = stablehlo.multiply %v1821, %v1819 : tensor<32x151296xf32>
    %v1823 = stablehlo.reshape %v1822 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1824 = stablehlo.dot_general %v1823, %b8_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x192xf32>) -> tensor<32x197x192xf32>
    %v1825 = stablehlo.broadcast_in_dim %b8_bfc2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1826 = stablehlo.add %v1824, %v1825 : tensor<32x197x192xf32>
    %v1827 = stablehlo.reshape %v1826 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1828 = stablehlo.add %v1776, %v1827 : tensor<32x37824xf32>
    %v1829 = stablehlo.reshape %v1828 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1830 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1831 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v1832 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v1833 = stablehlo.reduce(%v1829 init: %v1830) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1834 = stablehlo.broadcast_in_dim %v1833, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v1835 = stablehlo.divide %v1834, %v1831 : tensor<32x197x192xf32>
    %v1836 = stablehlo.subtract %v1829, %v1835 : tensor<32x197x192xf32>
    %v1837 = stablehlo.multiply %v1836, %v1836 : tensor<32x197x192xf32>
    %v1838 = stablehlo.reduce(%v1837 init: %v1830) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1839 = stablehlo.broadcast_in_dim %v1838, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v1840 = stablehlo.divide %v1839, %v1831 : tensor<32x197x192xf32>
    %v1841 = stablehlo.add %v1840, %v1832 : tensor<32x197x192xf32>
    %v1842 = stablehlo.rsqrt %v1841 : tensor<32x197x192xf32>
    %v1843 = stablehlo.multiply %v1836, %v1842 : tensor<32x197x192xf32>
    %v1844 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v1845 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v1846 = stablehlo.multiply %v1843, %v1844 : tensor<32x197x192xf32>
    %v1847 = stablehlo.add %v1846, %v1845 : tensor<32x197x192xf32>
    %v1848 = stablehlo.reshape %v1847 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1849 = stablehlo.reshape %v1848 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1850 = stablehlo.broadcast_in_dim %b9_g1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1851 = stablehlo.multiply %v1849, %v1850 : tensor<32x197x192xf32>
    %v1852 = stablehlo.reshape %v1851 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1853 = stablehlo.reshape %v1852 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1854 = stablehlo.broadcast_in_dim %b9_bt1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1855 = stablehlo.add %v1853, %v1854 : tensor<32x197x192xf32>
    %v1856 = stablehlo.reshape %v1855 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1857 = stablehlo.reshape %v1856 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1858 = stablehlo.dot_general %v1857, %b9_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v1859 = stablehlo.broadcast_in_dim %b9_bq, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1860 = stablehlo.add %v1858, %v1859 : tensor<32x197x192xf32>
    %v1861 = stablehlo.reshape %v1860 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1862 = stablehlo.reshape %v1856 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1863 = stablehlo.dot_general %v1862, %b9_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v1864 = stablehlo.broadcast_in_dim %b9_bk, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1865 = stablehlo.add %v1863, %v1864 : tensor<32x197x192xf32>
    %v1866 = stablehlo.reshape %v1865 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1867 = stablehlo.reshape %v1856 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1868 = stablehlo.dot_general %v1867, %b9_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v1869 = stablehlo.broadcast_in_dim %b9_bv, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1870 = stablehlo.add %v1868, %v1869 : tensor<32x197x192xf32>
    %v1871 = stablehlo.reshape %v1870 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1872 = stablehlo.reshape %v1861 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1873 = stablehlo.slice %v1872 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1874 = stablehlo.reshape %v1873 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1875 = stablehlo.reshape %v1866 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1876 = stablehlo.slice %v1875 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1877 = stablehlo.reshape %v1876 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1878 = stablehlo.reshape %v1871 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1879 = stablehlo.slice %v1878 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1880 = stablehlo.reshape %v1879 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1881 = stablehlo.reshape %v1877 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1882 = stablehlo.transpose %v1881, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1883 = stablehlo.reshape %v1882 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1884 = stablehlo.reshape %v1874 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1885 = stablehlo.reshape %v1883 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1886 = stablehlo.dot_general %v1884, %v1885, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1887 = stablehlo.reshape %v1886 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1888 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1889 = stablehlo.multiply %v1887, %v1888 : tensor<32x38809xf32>
    %v1890 = stablehlo.reshape %v1889 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1891 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1892 = stablehlo.exponential %v1890 : tensor<32x197x197xf32>
    %v1893 = stablehlo.reduce(%v1892 init: %v1891) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1894 = stablehlo.broadcast_in_dim %v1893, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1895 = stablehlo.divide %v1892, %v1894 : tensor<32x197x197xf32>
    %v1896 = stablehlo.reshape %v1895 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1897 = stablehlo.reshape %v1896 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1898 = stablehlo.reshape %v1880 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1899 = stablehlo.dot_general %v1897, %v1898, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1900 = stablehlo.reshape %v1899 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1901 = stablehlo.reshape %v1900 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1902 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1903 = stablehlo.pad %v1901, %v1902, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v1904 = stablehlo.reshape %v1903 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1905 = stablehlo.reshape %v1861 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1906 = stablehlo.slice %v1905 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1907 = stablehlo.reshape %v1906 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1908 = stablehlo.reshape %v1866 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1909 = stablehlo.slice %v1908 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1910 = stablehlo.reshape %v1909 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1911 = stablehlo.reshape %v1871 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1912 = stablehlo.slice %v1911 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1913 = stablehlo.reshape %v1912 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1914 = stablehlo.reshape %v1910 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1915 = stablehlo.transpose %v1914, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1916 = stablehlo.reshape %v1915 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1917 = stablehlo.reshape %v1907 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1918 = stablehlo.reshape %v1916 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1919 = stablehlo.dot_general %v1917, %v1918, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1920 = stablehlo.reshape %v1919 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1921 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1922 = stablehlo.multiply %v1920, %v1921 : tensor<32x38809xf32>
    %v1923 = stablehlo.reshape %v1922 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1924 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1925 = stablehlo.exponential %v1923 : tensor<32x197x197xf32>
    %v1926 = stablehlo.reduce(%v1925 init: %v1924) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1927 = stablehlo.broadcast_in_dim %v1926, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1928 = stablehlo.divide %v1925, %v1927 : tensor<32x197x197xf32>
    %v1929 = stablehlo.reshape %v1928 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1930 = stablehlo.reshape %v1929 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1931 = stablehlo.reshape %v1913 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1932 = stablehlo.dot_general %v1930, %v1931, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1933 = stablehlo.reshape %v1932 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1934 = stablehlo.reshape %v1933 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1935 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1936 = stablehlo.pad %v1934, %v1935, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v1937 = stablehlo.reshape %v1936 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1938 = stablehlo.add %v1904, %v1937 : tensor<32x37824xf32>
    %v1939 = stablehlo.reshape %v1861 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1940 = stablehlo.slice %v1939 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1941 = stablehlo.reshape %v1940 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1942 = stablehlo.reshape %v1866 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1943 = stablehlo.slice %v1942 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1944 = stablehlo.reshape %v1943 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1945 = stablehlo.reshape %v1871 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1946 = stablehlo.slice %v1945 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1947 = stablehlo.reshape %v1946 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1948 = stablehlo.reshape %v1944 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1949 = stablehlo.transpose %v1948, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1950 = stablehlo.reshape %v1949 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1951 = stablehlo.reshape %v1941 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1952 = stablehlo.reshape %v1950 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1953 = stablehlo.dot_general %v1951, %v1952, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1954 = stablehlo.reshape %v1953 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1955 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1956 = stablehlo.multiply %v1954, %v1955 : tensor<32x38809xf32>
    %v1957 = stablehlo.reshape %v1956 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1958 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1959 = stablehlo.exponential %v1957 : tensor<32x197x197xf32>
    %v1960 = stablehlo.reduce(%v1959 init: %v1958) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1961 = stablehlo.broadcast_in_dim %v1960, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1962 = stablehlo.divide %v1959, %v1961 : tensor<32x197x197xf32>
    %v1963 = stablehlo.reshape %v1962 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1964 = stablehlo.reshape %v1963 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1965 = stablehlo.reshape %v1947 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1966 = stablehlo.dot_general %v1964, %v1965, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1967 = stablehlo.reshape %v1966 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1968 = stablehlo.reshape %v1967 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1969 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1970 = stablehlo.pad %v1968, %v1969, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v1971 = stablehlo.reshape %v1970 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1972 = stablehlo.add %v1938, %v1971 : tensor<32x37824xf32>
    %v1973 = stablehlo.reshape %v1972 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1974 = stablehlo.dot_general %v1973, %b9_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v1975 = stablehlo.broadcast_in_dim %b9_bo, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1976 = stablehlo.add %v1974, %v1975 : tensor<32x197x192xf32>
    %v1977 = stablehlo.reshape %v1976 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1978 = stablehlo.add %v1828, %v1977 : tensor<32x37824xf32>
    %v1979 = stablehlo.reshape %v1978 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1980 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1981 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v1982 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v1983 = stablehlo.reduce(%v1979 init: %v1980) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1984 = stablehlo.broadcast_in_dim %v1983, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v1985 = stablehlo.divide %v1984, %v1981 : tensor<32x197x192xf32>
    %v1986 = stablehlo.subtract %v1979, %v1985 : tensor<32x197x192xf32>
    %v1987 = stablehlo.multiply %v1986, %v1986 : tensor<32x197x192xf32>
    %v1988 = stablehlo.reduce(%v1987 init: %v1980) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1989 = stablehlo.broadcast_in_dim %v1988, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v1990 = stablehlo.divide %v1989, %v1981 : tensor<32x197x192xf32>
    %v1991 = stablehlo.add %v1990, %v1982 : tensor<32x197x192xf32>
    %v1992 = stablehlo.rsqrt %v1991 : tensor<32x197x192xf32>
    %v1993 = stablehlo.multiply %v1986, %v1992 : tensor<32x197x192xf32>
    %v1994 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v1995 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v1996 = stablehlo.multiply %v1993, %v1994 : tensor<32x197x192xf32>
    %v1997 = stablehlo.add %v1996, %v1995 : tensor<32x197x192xf32>
    %v1998 = stablehlo.reshape %v1997 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1999 = stablehlo.reshape %v1998 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2000 = stablehlo.broadcast_in_dim %b9_g2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v2001 = stablehlo.multiply %v1999, %v2000 : tensor<32x197x192xf32>
    %v2002 = stablehlo.reshape %v2001 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2003 = stablehlo.reshape %v2002 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2004 = stablehlo.broadcast_in_dim %b9_bt2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v2005 = stablehlo.add %v2003, %v2004 : tensor<32x197x192xf32>
    %v2006 = stablehlo.reshape %v2005 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2007 = stablehlo.reshape %v2006 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2008 = stablehlo.dot_general %v2007, %b9_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x768xf32>) -> tensor<32x197x768xf32>
    %v2009 = stablehlo.broadcast_in_dim %b9_bfc1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v2010 = stablehlo.add %v2008, %v2009 : tensor<32x197x768xf32>
    %v2011 = stablehlo.reshape %v2010 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2012 = stablehlo.multiply %v2011, %v2011 : tensor<32x151296xf32>
    %v2013 = stablehlo.multiply %v2012, %v2011 : tensor<32x151296xf32>
    %v2014 = stablehlo.constant dense<0.044715> : tensor<32x151296xf32>
    %v2015 = stablehlo.multiply %v2014, %v2013 : tensor<32x151296xf32>
    %v2016 = stablehlo.add %v2011, %v2015 : tensor<32x151296xf32>
    %v2017 = stablehlo.constant dense<0.7978845608028654> : tensor<32x151296xf32>
    %v2018 = stablehlo.multiply %v2017, %v2016 : tensor<32x151296xf32>
    %v2019 = stablehlo.tanh %v2018 : tensor<32x151296xf32>
    %v2020 = stablehlo.constant dense<1.0> : tensor<32x151296xf32>
    %v2021 = stablehlo.add %v2020, %v2019 : tensor<32x151296xf32>
    %v2022 = stablehlo.constant dense<0.5> : tensor<32x151296xf32>
    %v2023 = stablehlo.multiply %v2022, %v2011 : tensor<32x151296xf32>
    %v2024 = stablehlo.multiply %v2023, %v2021 : tensor<32x151296xf32>
    %v2025 = stablehlo.reshape %v2024 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2026 = stablehlo.dot_general %v2025, %b9_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x192xf32>) -> tensor<32x197x192xf32>
    %v2027 = stablehlo.broadcast_in_dim %b9_bfc2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v2028 = stablehlo.add %v2026, %v2027 : tensor<32x197x192xf32>
    %v2029 = stablehlo.reshape %v2028 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2030 = stablehlo.add %v1978, %v2029 : tensor<32x37824xf32>
    %v2031 = stablehlo.reshape %v2030 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2032 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2033 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v2034 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v2035 = stablehlo.reduce(%v2031 init: %v2032) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2036 = stablehlo.broadcast_in_dim %v2035, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v2037 = stablehlo.divide %v2036, %v2033 : tensor<32x197x192xf32>
    %v2038 = stablehlo.subtract %v2031, %v2037 : tensor<32x197x192xf32>
    %v2039 = stablehlo.multiply %v2038, %v2038 : tensor<32x197x192xf32>
    %v2040 = stablehlo.reduce(%v2039 init: %v2032) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2041 = stablehlo.broadcast_in_dim %v2040, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v2042 = stablehlo.divide %v2041, %v2033 : tensor<32x197x192xf32>
    %v2043 = stablehlo.add %v2042, %v2034 : tensor<32x197x192xf32>
    %v2044 = stablehlo.rsqrt %v2043 : tensor<32x197x192xf32>
    %v2045 = stablehlo.multiply %v2038, %v2044 : tensor<32x197x192xf32>
    %v2046 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v2047 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v2048 = stablehlo.multiply %v2045, %v2046 : tensor<32x197x192xf32>
    %v2049 = stablehlo.add %v2048, %v2047 : tensor<32x197x192xf32>
    %v2050 = stablehlo.reshape %v2049 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2051 = stablehlo.reshape %v2050 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2052 = stablehlo.broadcast_in_dim %b10_g1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v2053 = stablehlo.multiply %v2051, %v2052 : tensor<32x197x192xf32>
    %v2054 = stablehlo.reshape %v2053 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2055 = stablehlo.reshape %v2054 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2056 = stablehlo.broadcast_in_dim %b10_bt1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v2057 = stablehlo.add %v2055, %v2056 : tensor<32x197x192xf32>
    %v2058 = stablehlo.reshape %v2057 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2059 = stablehlo.reshape %v2058 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2060 = stablehlo.dot_general %v2059, %b10_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v2061 = stablehlo.broadcast_in_dim %b10_bq, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v2062 = stablehlo.add %v2060, %v2061 : tensor<32x197x192xf32>
    %v2063 = stablehlo.reshape %v2062 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2064 = stablehlo.reshape %v2058 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2065 = stablehlo.dot_general %v2064, %b10_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v2066 = stablehlo.broadcast_in_dim %b10_bk, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v2067 = stablehlo.add %v2065, %v2066 : tensor<32x197x192xf32>
    %v2068 = stablehlo.reshape %v2067 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2069 = stablehlo.reshape %v2058 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2070 = stablehlo.dot_general %v2069, %b10_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v2071 = stablehlo.broadcast_in_dim %b10_bv, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v2072 = stablehlo.add %v2070, %v2071 : tensor<32x197x192xf32>
    %v2073 = stablehlo.reshape %v2072 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2074 = stablehlo.reshape %v2063 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2075 = stablehlo.slice %v2074 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v2076 = stablehlo.reshape %v2075 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2077 = stablehlo.reshape %v2068 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2078 = stablehlo.slice %v2077 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v2079 = stablehlo.reshape %v2078 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2080 = stablehlo.reshape %v2073 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2081 = stablehlo.slice %v2080 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v2082 = stablehlo.reshape %v2081 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2083 = stablehlo.reshape %v2079 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2084 = stablehlo.transpose %v2083, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2085 = stablehlo.reshape %v2084 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2086 = stablehlo.reshape %v2076 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2087 = stablehlo.reshape %v2085 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2088 = stablehlo.dot_general %v2086, %v2087, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2089 = stablehlo.reshape %v2088 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2090 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2091 = stablehlo.multiply %v2089, %v2090 : tensor<32x38809xf32>
    %v2092 = stablehlo.reshape %v2091 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2093 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2094 = stablehlo.exponential %v2092 : tensor<32x197x197xf32>
    %v2095 = stablehlo.reduce(%v2094 init: %v2093) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2096 = stablehlo.broadcast_in_dim %v2095, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2097 = stablehlo.divide %v2094, %v2096 : tensor<32x197x197xf32>
    %v2098 = stablehlo.reshape %v2097 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2099 = stablehlo.reshape %v2098 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2100 = stablehlo.reshape %v2082 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2101 = stablehlo.dot_general %v2099, %v2100, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2102 = stablehlo.reshape %v2101 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2103 = stablehlo.reshape %v2102 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2104 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2105 = stablehlo.pad %v2103, %v2104, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v2106 = stablehlo.reshape %v2105 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2107 = stablehlo.reshape %v2063 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2108 = stablehlo.slice %v2107 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v2109 = stablehlo.reshape %v2108 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2110 = stablehlo.reshape %v2068 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2111 = stablehlo.slice %v2110 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v2112 = stablehlo.reshape %v2111 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2113 = stablehlo.reshape %v2073 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2114 = stablehlo.slice %v2113 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v2115 = stablehlo.reshape %v2114 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2116 = stablehlo.reshape %v2112 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2117 = stablehlo.transpose %v2116, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2118 = stablehlo.reshape %v2117 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2119 = stablehlo.reshape %v2109 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2120 = stablehlo.reshape %v2118 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2121 = stablehlo.dot_general %v2119, %v2120, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2122 = stablehlo.reshape %v2121 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2123 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2124 = stablehlo.multiply %v2122, %v2123 : tensor<32x38809xf32>
    %v2125 = stablehlo.reshape %v2124 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2126 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2127 = stablehlo.exponential %v2125 : tensor<32x197x197xf32>
    %v2128 = stablehlo.reduce(%v2127 init: %v2126) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2129 = stablehlo.broadcast_in_dim %v2128, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2130 = stablehlo.divide %v2127, %v2129 : tensor<32x197x197xf32>
    %v2131 = stablehlo.reshape %v2130 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2132 = stablehlo.reshape %v2131 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2133 = stablehlo.reshape %v2115 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2134 = stablehlo.dot_general %v2132, %v2133, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2135 = stablehlo.reshape %v2134 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2136 = stablehlo.reshape %v2135 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2137 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2138 = stablehlo.pad %v2136, %v2137, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v2139 = stablehlo.reshape %v2138 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2140 = stablehlo.add %v2106, %v2139 : tensor<32x37824xf32>
    %v2141 = stablehlo.reshape %v2063 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2142 = stablehlo.slice %v2141 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v2143 = stablehlo.reshape %v2142 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2144 = stablehlo.reshape %v2068 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2145 = stablehlo.slice %v2144 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v2146 = stablehlo.reshape %v2145 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2147 = stablehlo.reshape %v2073 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2148 = stablehlo.slice %v2147 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v2149 = stablehlo.reshape %v2148 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2150 = stablehlo.reshape %v2146 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2151 = stablehlo.transpose %v2150, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2152 = stablehlo.reshape %v2151 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2153 = stablehlo.reshape %v2143 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2154 = stablehlo.reshape %v2152 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2155 = stablehlo.dot_general %v2153, %v2154, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2156 = stablehlo.reshape %v2155 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2157 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2158 = stablehlo.multiply %v2156, %v2157 : tensor<32x38809xf32>
    %v2159 = stablehlo.reshape %v2158 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2160 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2161 = stablehlo.exponential %v2159 : tensor<32x197x197xf32>
    %v2162 = stablehlo.reduce(%v2161 init: %v2160) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2163 = stablehlo.broadcast_in_dim %v2162, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2164 = stablehlo.divide %v2161, %v2163 : tensor<32x197x197xf32>
    %v2165 = stablehlo.reshape %v2164 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2166 = stablehlo.reshape %v2165 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2167 = stablehlo.reshape %v2149 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2168 = stablehlo.dot_general %v2166, %v2167, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2169 = stablehlo.reshape %v2168 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2170 = stablehlo.reshape %v2169 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2171 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2172 = stablehlo.pad %v2170, %v2171, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v2173 = stablehlo.reshape %v2172 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2174 = stablehlo.add %v2140, %v2173 : tensor<32x37824xf32>
    %v2175 = stablehlo.reshape %v2174 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2176 = stablehlo.dot_general %v2175, %b10_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v2177 = stablehlo.broadcast_in_dim %b10_bo, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v2178 = stablehlo.add %v2176, %v2177 : tensor<32x197x192xf32>
    %v2179 = stablehlo.reshape %v2178 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2180 = stablehlo.add %v2030, %v2179 : tensor<32x37824xf32>
    %v2181 = stablehlo.reshape %v2180 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2182 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2183 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v2184 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v2185 = stablehlo.reduce(%v2181 init: %v2182) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2186 = stablehlo.broadcast_in_dim %v2185, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v2187 = stablehlo.divide %v2186, %v2183 : tensor<32x197x192xf32>
    %v2188 = stablehlo.subtract %v2181, %v2187 : tensor<32x197x192xf32>
    %v2189 = stablehlo.multiply %v2188, %v2188 : tensor<32x197x192xf32>
    %v2190 = stablehlo.reduce(%v2189 init: %v2182) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2191 = stablehlo.broadcast_in_dim %v2190, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v2192 = stablehlo.divide %v2191, %v2183 : tensor<32x197x192xf32>
    %v2193 = stablehlo.add %v2192, %v2184 : tensor<32x197x192xf32>
    %v2194 = stablehlo.rsqrt %v2193 : tensor<32x197x192xf32>
    %v2195 = stablehlo.multiply %v2188, %v2194 : tensor<32x197x192xf32>
    %v2196 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v2197 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v2198 = stablehlo.multiply %v2195, %v2196 : tensor<32x197x192xf32>
    %v2199 = stablehlo.add %v2198, %v2197 : tensor<32x197x192xf32>
    %v2200 = stablehlo.reshape %v2199 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2201 = stablehlo.reshape %v2200 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2202 = stablehlo.broadcast_in_dim %b10_g2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v2203 = stablehlo.multiply %v2201, %v2202 : tensor<32x197x192xf32>
    %v2204 = stablehlo.reshape %v2203 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2205 = stablehlo.reshape %v2204 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2206 = stablehlo.broadcast_in_dim %b10_bt2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v2207 = stablehlo.add %v2205, %v2206 : tensor<32x197x192xf32>
    %v2208 = stablehlo.reshape %v2207 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2209 = stablehlo.reshape %v2208 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2210 = stablehlo.dot_general %v2209, %b10_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x768xf32>) -> tensor<32x197x768xf32>
    %v2211 = stablehlo.broadcast_in_dim %b10_bfc1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v2212 = stablehlo.add %v2210, %v2211 : tensor<32x197x768xf32>
    %v2213 = stablehlo.reshape %v2212 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2214 = stablehlo.multiply %v2213, %v2213 : tensor<32x151296xf32>
    %v2215 = stablehlo.multiply %v2214, %v2213 : tensor<32x151296xf32>
    %v2216 = stablehlo.constant dense<0.044715> : tensor<32x151296xf32>
    %v2217 = stablehlo.multiply %v2216, %v2215 : tensor<32x151296xf32>
    %v2218 = stablehlo.add %v2213, %v2217 : tensor<32x151296xf32>
    %v2219 = stablehlo.constant dense<0.7978845608028654> : tensor<32x151296xf32>
    %v2220 = stablehlo.multiply %v2219, %v2218 : tensor<32x151296xf32>
    %v2221 = stablehlo.tanh %v2220 : tensor<32x151296xf32>
    %v2222 = stablehlo.constant dense<1.0> : tensor<32x151296xf32>
    %v2223 = stablehlo.add %v2222, %v2221 : tensor<32x151296xf32>
    %v2224 = stablehlo.constant dense<0.5> : tensor<32x151296xf32>
    %v2225 = stablehlo.multiply %v2224, %v2213 : tensor<32x151296xf32>
    %v2226 = stablehlo.multiply %v2225, %v2223 : tensor<32x151296xf32>
    %v2227 = stablehlo.reshape %v2226 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2228 = stablehlo.dot_general %v2227, %b10_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x192xf32>) -> tensor<32x197x192xf32>
    %v2229 = stablehlo.broadcast_in_dim %b10_bfc2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v2230 = stablehlo.add %v2228, %v2229 : tensor<32x197x192xf32>
    %v2231 = stablehlo.reshape %v2230 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2232 = stablehlo.add %v2180, %v2231 : tensor<32x37824xf32>
    %v2233 = stablehlo.reshape %v2232 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2234 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2235 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v2236 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v2237 = stablehlo.reduce(%v2233 init: %v2234) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2238 = stablehlo.broadcast_in_dim %v2237, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v2239 = stablehlo.divide %v2238, %v2235 : tensor<32x197x192xf32>
    %v2240 = stablehlo.subtract %v2233, %v2239 : tensor<32x197x192xf32>
    %v2241 = stablehlo.multiply %v2240, %v2240 : tensor<32x197x192xf32>
    %v2242 = stablehlo.reduce(%v2241 init: %v2234) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2243 = stablehlo.broadcast_in_dim %v2242, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v2244 = stablehlo.divide %v2243, %v2235 : tensor<32x197x192xf32>
    %v2245 = stablehlo.add %v2244, %v2236 : tensor<32x197x192xf32>
    %v2246 = stablehlo.rsqrt %v2245 : tensor<32x197x192xf32>
    %v2247 = stablehlo.multiply %v2240, %v2246 : tensor<32x197x192xf32>
    %v2248 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v2249 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v2250 = stablehlo.multiply %v2247, %v2248 : tensor<32x197x192xf32>
    %v2251 = stablehlo.add %v2250, %v2249 : tensor<32x197x192xf32>
    %v2252 = stablehlo.reshape %v2251 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2253 = stablehlo.reshape %v2252 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2254 = stablehlo.broadcast_in_dim %b11_g1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v2255 = stablehlo.multiply %v2253, %v2254 : tensor<32x197x192xf32>
    %v2256 = stablehlo.reshape %v2255 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2257 = stablehlo.reshape %v2256 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2258 = stablehlo.broadcast_in_dim %b11_bt1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v2259 = stablehlo.add %v2257, %v2258 : tensor<32x197x192xf32>
    %v2260 = stablehlo.reshape %v2259 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2261 = stablehlo.reshape %v2260 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2262 = stablehlo.dot_general %v2261, %b11_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v2263 = stablehlo.broadcast_in_dim %b11_bq, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v2264 = stablehlo.add %v2262, %v2263 : tensor<32x197x192xf32>
    %v2265 = stablehlo.reshape %v2264 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2266 = stablehlo.reshape %v2260 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2267 = stablehlo.dot_general %v2266, %b11_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v2268 = stablehlo.broadcast_in_dim %b11_bk, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v2269 = stablehlo.add %v2267, %v2268 : tensor<32x197x192xf32>
    %v2270 = stablehlo.reshape %v2269 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2271 = stablehlo.reshape %v2260 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2272 = stablehlo.dot_general %v2271, %b11_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v2273 = stablehlo.broadcast_in_dim %b11_bv, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v2274 = stablehlo.add %v2272, %v2273 : tensor<32x197x192xf32>
    %v2275 = stablehlo.reshape %v2274 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2276 = stablehlo.reshape %v2265 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2277 = stablehlo.slice %v2276 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v2278 = stablehlo.reshape %v2277 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2279 = stablehlo.reshape %v2270 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2280 = stablehlo.slice %v2279 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v2281 = stablehlo.reshape %v2280 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2282 = stablehlo.reshape %v2275 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2283 = stablehlo.slice %v2282 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v2284 = stablehlo.reshape %v2283 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2285 = stablehlo.reshape %v2281 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2286 = stablehlo.transpose %v2285, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2287 = stablehlo.reshape %v2286 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2288 = stablehlo.reshape %v2278 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2289 = stablehlo.reshape %v2287 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2290 = stablehlo.dot_general %v2288, %v2289, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2291 = stablehlo.reshape %v2290 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2292 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2293 = stablehlo.multiply %v2291, %v2292 : tensor<32x38809xf32>
    %v2294 = stablehlo.reshape %v2293 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2295 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2296 = stablehlo.exponential %v2294 : tensor<32x197x197xf32>
    %v2297 = stablehlo.reduce(%v2296 init: %v2295) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2298 = stablehlo.broadcast_in_dim %v2297, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2299 = stablehlo.divide %v2296, %v2298 : tensor<32x197x197xf32>
    %v2300 = stablehlo.reshape %v2299 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2301 = stablehlo.reshape %v2300 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2302 = stablehlo.reshape %v2284 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2303 = stablehlo.dot_general %v2301, %v2302, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2304 = stablehlo.reshape %v2303 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2305 = stablehlo.reshape %v2304 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2306 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2307 = stablehlo.pad %v2305, %v2306, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v2308 = stablehlo.reshape %v2307 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2309 = stablehlo.reshape %v2265 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2310 = stablehlo.slice %v2309 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v2311 = stablehlo.reshape %v2310 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2312 = stablehlo.reshape %v2270 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2313 = stablehlo.slice %v2312 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v2314 = stablehlo.reshape %v2313 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2315 = stablehlo.reshape %v2275 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2316 = stablehlo.slice %v2315 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v2317 = stablehlo.reshape %v2316 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2318 = stablehlo.reshape %v2314 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2319 = stablehlo.transpose %v2318, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2320 = stablehlo.reshape %v2319 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2321 = stablehlo.reshape %v2311 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2322 = stablehlo.reshape %v2320 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2323 = stablehlo.dot_general %v2321, %v2322, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2324 = stablehlo.reshape %v2323 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2325 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2326 = stablehlo.multiply %v2324, %v2325 : tensor<32x38809xf32>
    %v2327 = stablehlo.reshape %v2326 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2328 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2329 = stablehlo.exponential %v2327 : tensor<32x197x197xf32>
    %v2330 = stablehlo.reduce(%v2329 init: %v2328) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2331 = stablehlo.broadcast_in_dim %v2330, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2332 = stablehlo.divide %v2329, %v2331 : tensor<32x197x197xf32>
    %v2333 = stablehlo.reshape %v2332 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2334 = stablehlo.reshape %v2333 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2335 = stablehlo.reshape %v2317 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2336 = stablehlo.dot_general %v2334, %v2335, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2337 = stablehlo.reshape %v2336 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2338 = stablehlo.reshape %v2337 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2339 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2340 = stablehlo.pad %v2338, %v2339, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v2341 = stablehlo.reshape %v2340 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2342 = stablehlo.add %v2308, %v2341 : tensor<32x37824xf32>
    %v2343 = stablehlo.reshape %v2265 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2344 = stablehlo.slice %v2343 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v2345 = stablehlo.reshape %v2344 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2346 = stablehlo.reshape %v2270 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2347 = stablehlo.slice %v2346 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v2348 = stablehlo.reshape %v2347 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2349 = stablehlo.reshape %v2275 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2350 = stablehlo.slice %v2349 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v2351 = stablehlo.reshape %v2350 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2352 = stablehlo.reshape %v2348 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2353 = stablehlo.transpose %v2352, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2354 = stablehlo.reshape %v2353 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2355 = stablehlo.reshape %v2345 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2356 = stablehlo.reshape %v2354 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2357 = stablehlo.dot_general %v2355, %v2356, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2358 = stablehlo.reshape %v2357 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2359 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2360 = stablehlo.multiply %v2358, %v2359 : tensor<32x38809xf32>
    %v2361 = stablehlo.reshape %v2360 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2362 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2363 = stablehlo.exponential %v2361 : tensor<32x197x197xf32>
    %v2364 = stablehlo.reduce(%v2363 init: %v2362) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2365 = stablehlo.broadcast_in_dim %v2364, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2366 = stablehlo.divide %v2363, %v2365 : tensor<32x197x197xf32>
    %v2367 = stablehlo.reshape %v2366 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2368 = stablehlo.reshape %v2367 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2369 = stablehlo.reshape %v2351 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2370 = stablehlo.dot_general %v2368, %v2369, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2371 = stablehlo.reshape %v2370 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2372 = stablehlo.reshape %v2371 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2373 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2374 = stablehlo.pad %v2372, %v2373, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v2375 = stablehlo.reshape %v2374 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2376 = stablehlo.add %v2342, %v2375 : tensor<32x37824xf32>
    %v2377 = stablehlo.reshape %v2376 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2378 = stablehlo.dot_general %v2377, %b11_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v2379 = stablehlo.broadcast_in_dim %b11_bo, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v2380 = stablehlo.add %v2378, %v2379 : tensor<32x197x192xf32>
    %v2381 = stablehlo.reshape %v2380 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2382 = stablehlo.add %v2232, %v2381 : tensor<32x37824xf32>
    %v2383 = stablehlo.reshape %v2382 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2384 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2385 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v2386 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v2387 = stablehlo.reduce(%v2383 init: %v2384) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2388 = stablehlo.broadcast_in_dim %v2387, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v2389 = stablehlo.divide %v2388, %v2385 : tensor<32x197x192xf32>
    %v2390 = stablehlo.subtract %v2383, %v2389 : tensor<32x197x192xf32>
    %v2391 = stablehlo.multiply %v2390, %v2390 : tensor<32x197x192xf32>
    %v2392 = stablehlo.reduce(%v2391 init: %v2384) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2393 = stablehlo.broadcast_in_dim %v2392, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v2394 = stablehlo.divide %v2393, %v2385 : tensor<32x197x192xf32>
    %v2395 = stablehlo.add %v2394, %v2386 : tensor<32x197x192xf32>
    %v2396 = stablehlo.rsqrt %v2395 : tensor<32x197x192xf32>
    %v2397 = stablehlo.multiply %v2390, %v2396 : tensor<32x197x192xf32>
    %v2398 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v2399 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v2400 = stablehlo.multiply %v2397, %v2398 : tensor<32x197x192xf32>
    %v2401 = stablehlo.add %v2400, %v2399 : tensor<32x197x192xf32>
    %v2402 = stablehlo.reshape %v2401 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2403 = stablehlo.reshape %v2402 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2404 = stablehlo.broadcast_in_dim %b11_g2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v2405 = stablehlo.multiply %v2403, %v2404 : tensor<32x197x192xf32>
    %v2406 = stablehlo.reshape %v2405 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2407 = stablehlo.reshape %v2406 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2408 = stablehlo.broadcast_in_dim %b11_bt2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v2409 = stablehlo.add %v2407, %v2408 : tensor<32x197x192xf32>
    %v2410 = stablehlo.reshape %v2409 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2411 = stablehlo.reshape %v2410 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2412 = stablehlo.dot_general %v2411, %b11_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x768xf32>) -> tensor<32x197x768xf32>
    %v2413 = stablehlo.broadcast_in_dim %b11_bfc1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v2414 = stablehlo.add %v2412, %v2413 : tensor<32x197x768xf32>
    %v2415 = stablehlo.reshape %v2414 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2416 = stablehlo.multiply %v2415, %v2415 : tensor<32x151296xf32>
    %v2417 = stablehlo.multiply %v2416, %v2415 : tensor<32x151296xf32>
    %v2418 = stablehlo.constant dense<0.044715> : tensor<32x151296xf32>
    %v2419 = stablehlo.multiply %v2418, %v2417 : tensor<32x151296xf32>
    %v2420 = stablehlo.add %v2415, %v2419 : tensor<32x151296xf32>
    %v2421 = stablehlo.constant dense<0.7978845608028654> : tensor<32x151296xf32>
    %v2422 = stablehlo.multiply %v2421, %v2420 : tensor<32x151296xf32>
    %v2423 = stablehlo.tanh %v2422 : tensor<32x151296xf32>
    %v2424 = stablehlo.constant dense<1.0> : tensor<32x151296xf32>
    %v2425 = stablehlo.add %v2424, %v2423 : tensor<32x151296xf32>
    %v2426 = stablehlo.constant dense<0.5> : tensor<32x151296xf32>
    %v2427 = stablehlo.multiply %v2426, %v2415 : tensor<32x151296xf32>
    %v2428 = stablehlo.multiply %v2427, %v2425 : tensor<32x151296xf32>
    %v2429 = stablehlo.reshape %v2428 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2430 = stablehlo.dot_general %v2429, %b11_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x192xf32>) -> tensor<32x197x192xf32>
    %v2431 = stablehlo.broadcast_in_dim %b11_bfc2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v2432 = stablehlo.add %v2430, %v2431 : tensor<32x197x192xf32>
    %v2433 = stablehlo.reshape %v2432 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2434 = stablehlo.add %v2382, %v2433 : tensor<32x37824xf32>
    %v2435 = stablehlo.reshape %v2434 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2436 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2437 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v2438 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v2439 = stablehlo.reduce(%v2435 init: %v2436) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2440 = stablehlo.broadcast_in_dim %v2439, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v2441 = stablehlo.divide %v2440, %v2437 : tensor<32x197x192xf32>
    %v2442 = stablehlo.subtract %v2435, %v2441 : tensor<32x197x192xf32>
    %v2443 = stablehlo.multiply %v2442, %v2442 : tensor<32x197x192xf32>
    %v2444 = stablehlo.reduce(%v2443 init: %v2436) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2445 = stablehlo.broadcast_in_dim %v2444, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v2446 = stablehlo.divide %v2445, %v2437 : tensor<32x197x192xf32>
    %v2447 = stablehlo.add %v2446, %v2438 : tensor<32x197x192xf32>
    %v2448 = stablehlo.rsqrt %v2447 : tensor<32x197x192xf32>
    %v2449 = stablehlo.multiply %v2442, %v2448 : tensor<32x197x192xf32>
    %v2450 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v2451 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v2452 = stablehlo.multiply %v2449, %v2450 : tensor<32x197x192xf32>
    %v2453 = stablehlo.add %v2452, %v2451 : tensor<32x197x192xf32>
    %v2454 = stablehlo.reshape %v2453 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2455 = stablehlo.reshape %v2454 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2456 = stablehlo.broadcast_in_dim %gF, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v2457 = stablehlo.multiply %v2455, %v2456 : tensor<32x197x192xf32>
    %v2458 = stablehlo.reshape %v2457 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2459 = stablehlo.reshape %v2458 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2460 = stablehlo.broadcast_in_dim %btF, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v2461 = stablehlo.add %v2459, %v2460 : tensor<32x197x192xf32>
    %v2462 = stablehlo.reshape %v2461 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2463 = stablehlo.reshape %v2462 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2464 = stablehlo.slice %v2463 [0:32, 0:1, 0:192] : (tensor<32x197x192xf32>) -> tensor<32x1x192xf32>
    %v2465 = stablehlo.reshape %v2464 : (tensor<32x1x192xf32>) -> tensor<32x192xf32>
    %v2466 = stablehlo.dot_general %v2465, %Wc, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x192xf32>, tensor<192x10xf32>) -> tensor<32x10xf32>
    %v2467 = stablehlo.broadcast_in_dim %bc, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v2468 = stablehlo.add %v2466, %v2467 : tensor<32x10xf32>
    %v2469 = stablehlo.exponential %v2468 : tensor<32x10xf32>
    %v2470 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2471 = stablehlo.reduce(%v2469 init: %v2470) applies stablehlo.add across dimensions = [1] : (tensor<32x10xf32>, tensor<f32>) -> tensor<32xf32>
    %v2472 = stablehlo.broadcast_in_dim %v2471, dims = [0] : (tensor<32xf32>) -> tensor<32x10xf32>
    %v2473 = stablehlo.divide %v2469, %v2472 : tensor<32x10xf32>
    %v2474 = stablehlo.subtract %v2473, %onehot : tensor<32x10xf32>
    %v2475 = stablehlo.dot_general %v2474, %Wc, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<192x10xf32>) -> tensor<32x192xf32>
    %v2476 = stablehlo.dot_general %v2465, %v2474, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x192xf32>, tensor<32x10xf32>) -> tensor<192x10xf32>
    %v2477 = stablehlo.constant dense<0.003125> : tensor<192x10xf32>
    %v2478 = stablehlo.multiply %v2476, %v2477 : tensor<192x10xf32>
    %v2479 = stablehlo.subtract %Wc, %v2478 : tensor<192x10xf32>
    %v2480 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2481 = stablehlo.reduce(%v2474 init: %v2480) applies stablehlo.add across dimensions = [0] : (tensor<32x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v2482 = stablehlo.constant dense<0.003125> : tensor<10xf32>
    %v2483 = stablehlo.multiply %v2481, %v2482 : tensor<10xf32>
    %v2484 = stablehlo.subtract %bc, %v2483 : tensor<10xf32>
    %v2485 = stablehlo.reshape %v2475 : (tensor<32x192xf32>) -> tensor<32x1x192xf32>
    %v2486 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2487 = stablehlo.pad %v2485, %v2486, low = [0, 0, 0], high = [0, 196, 0], interior = [0, 0, 0] : (tensor<32x1x192xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v2488 = stablehlo.reshape %v2487 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2489 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2490 = stablehlo.reshape %v2488 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2491 = stablehlo.reduce(%v2490 init: %v2489) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v2492 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v2493 = stablehlo.multiply %v2491, %v2492 : tensor<192xf32>
    %v2494 = stablehlo.subtract %btF, %v2493 : tensor<192xf32>
    %v2495 = stablehlo.reshape %v2434 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2496 = stablehlo.reshape %v2488 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2497 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2498 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v2499 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v2500 = stablehlo.reduce(%v2495 init: %v2497) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2501 = stablehlo.broadcast_in_dim %v2500, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v2502 = stablehlo.divide %v2501, %v2498 : tensor<32x197x192xf32>
    %v2503 = stablehlo.subtract %v2495, %v2502 : tensor<32x197x192xf32>
    %v2504 = stablehlo.multiply %v2503, %v2503 : tensor<32x197x192xf32>
    %v2505 = stablehlo.reduce(%v2504 init: %v2497) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2506 = stablehlo.broadcast_in_dim %v2505, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v2507 = stablehlo.divide %v2506, %v2498 : tensor<32x197x192xf32>
    %v2508 = stablehlo.add %v2507, %v2499 : tensor<32x197x192xf32>
    %v2509 = stablehlo.rsqrt %v2508 : tensor<32x197x192xf32>
    %v2510 = stablehlo.multiply %v2503, %v2509 : tensor<32x197x192xf32>
    %v2511 = stablehlo.multiply %v2496, %v2510 : tensor<32x197x192xf32>
    %v2512 = stablehlo.reduce(%v2511 init: %v2497) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v2513 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v2514 = stablehlo.multiply %v2512, %v2513 : tensor<192xf32>
    %v2515 = stablehlo.subtract %gF, %v2514 : tensor<192xf32>
    %v2516 = stablehlo.reshape %v2488 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2517 = stablehlo.broadcast_in_dim %gF, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v2518 = stablehlo.multiply %v2516, %v2517 : tensor<32x197x192xf32>
    %v2519 = stablehlo.reshape %v2518 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2520 = stablehlo.reshape %v2519 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2521 = stablehlo.reshape %v2434 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2522 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2523 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v2524 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v2525 = stablehlo.reduce(%v2521 init: %v2522) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2526 = stablehlo.broadcast_in_dim %v2525, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v2527 = stablehlo.divide %v2526, %v2523 : tensor<32x197x192xf32>
    %v2528 = stablehlo.subtract %v2521, %v2527 : tensor<32x197x192xf32>
    %v2529 = stablehlo.multiply %v2528, %v2528 : tensor<32x197x192xf32>
    %v2530 = stablehlo.reduce(%v2529 init: %v2522) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2531 = stablehlo.broadcast_in_dim %v2530, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v2532 = stablehlo.divide %v2531, %v2523 : tensor<32x197x192xf32>
    %v2533 = stablehlo.add %v2532, %v2524 : tensor<32x197x192xf32>
    %v2534 = stablehlo.rsqrt %v2533 : tensor<32x197x192xf32>
    %v2535 = stablehlo.multiply %v2528, %v2534 : tensor<32x197x192xf32>
    %v2536 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v2537 = stablehlo.multiply %v2536, %v2520 : tensor<32x197x192xf32>
    %v2538 = stablehlo.reduce(%v2537 init: %v2522) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2539 = stablehlo.broadcast_in_dim %v2538, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v2540 = stablehlo.multiply %v2535, %v2537 : tensor<32x197x192xf32>
    %v2541 = stablehlo.reduce(%v2540 init: %v2522) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2542 = stablehlo.broadcast_in_dim %v2541, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v2543 = stablehlo.multiply %v2537, %v2523 : tensor<32x197x192xf32>
    %v2544 = stablehlo.subtract %v2543, %v2539 : tensor<32x197x192xf32>
    %v2545 = stablehlo.multiply %v2535, %v2542 : tensor<32x197x192xf32>
    %v2546 = stablehlo.subtract %v2544, %v2545 : tensor<32x197x192xf32>
    %v2547 = stablehlo.divide %v2534, %v2523 : tensor<32x197x192xf32>
    %v2548 = stablehlo.multiply %v2547, %v2546 : tensor<32x197x192xf32>
    %v2549 = stablehlo.reshape %v2548 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2550 = stablehlo.reshape %v2549 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2551 = stablehlo.dot_general %v2550, %b11_Wfc2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<768x192xf32>) -> tensor<32x197x768xf32>
    %v2552 = stablehlo.reshape %v2551 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2553 = stablehlo.reshape %v2428 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2554 = stablehlo.reshape %v2549 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2555 = stablehlo.dot_general %v2553, %v2554, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<32x197x192xf32>) -> tensor<768x192xf32>
    %v2556 = stablehlo.constant dense<0.003125> : tensor<768x192xf32>
    %v2557 = stablehlo.multiply %v2555, %v2556 : tensor<768x192xf32>
    %v2558 = stablehlo.subtract %b11_Wfc2, %v2557 : tensor<768x192xf32>
    %v2559 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2560 = stablehlo.reshape %v2549 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2561 = stablehlo.reduce(%v2560 init: %v2559) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v2562 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v2563 = stablehlo.multiply %v2561, %v2562 : tensor<192xf32>
    %v2564 = stablehlo.subtract %b11_bfc2, %v2563 : tensor<192xf32>
    %v2565 = stablehlo.multiply %v2415, %v2415 : tensor<32x151296xf32>
    %v2566 = stablehlo.multiply %v2565, %v2415 : tensor<32x151296xf32>
    %v2567 = stablehlo.constant dense<0.044715> : tensor<32x151296xf32>
    %v2568 = stablehlo.multiply %v2567, %v2566 : tensor<32x151296xf32>
    %v2569 = stablehlo.add %v2415, %v2568 : tensor<32x151296xf32>
    %v2570 = stablehlo.constant dense<0.7978845608028654> : tensor<32x151296xf32>
    %v2571 = stablehlo.multiply %v2570, %v2569 : tensor<32x151296xf32>
    %v2572 = stablehlo.tanh %v2571 : tensor<32x151296xf32>
    %v2573 = stablehlo.constant dense<1.0> : tensor<32x151296xf32>
    %v2574 = stablehlo.add %v2573, %v2572 : tensor<32x151296xf32>
    %v2575 = stablehlo.constant dense<0.5> : tensor<32x151296xf32>
    %v2576 = stablehlo.multiply %v2575, %v2574 : tensor<32x151296xf32>
    %v2577 = stablehlo.multiply %v2572, %v2572 : tensor<32x151296xf32>
    %v2578 = stablehlo.subtract %v2573, %v2577 : tensor<32x151296xf32>
    %v2579 = stablehlo.multiply %v2575, %v2415 : tensor<32x151296xf32>
    %v2580 = stablehlo.multiply %v2579, %v2578 : tensor<32x151296xf32>
    %v2581 = stablehlo.constant dense<0.134145> : tensor<32x151296xf32>
    %v2582 = stablehlo.multiply %v2581, %v2565 : tensor<32x151296xf32>
    %v2583 = stablehlo.add %v2573, %v2582 : tensor<32x151296xf32>
    %v2584 = stablehlo.multiply %v2570, %v2583 : tensor<32x151296xf32>
    %v2585 = stablehlo.multiply %v2580, %v2584 : tensor<32x151296xf32>
    %v2586 = stablehlo.add %v2576, %v2585 : tensor<32x151296xf32>
    %v2587 = stablehlo.multiply %v2552, %v2586 : tensor<32x151296xf32>
    %v2588 = stablehlo.reshape %v2587 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2589 = stablehlo.dot_general %v2588, %b11_Wfc1, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<192x768xf32>) -> tensor<32x197x192xf32>
    %v2590 = stablehlo.reshape %v2589 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2591 = stablehlo.reshape %v2410 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2592 = stablehlo.reshape %v2587 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2593 = stablehlo.dot_general %v2591, %v2592, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x768xf32>) -> tensor<192x768xf32>
    %v2594 = stablehlo.constant dense<0.003125> : tensor<192x768xf32>
    %v2595 = stablehlo.multiply %v2593, %v2594 : tensor<192x768xf32>
    %v2596 = stablehlo.subtract %b11_Wfc1, %v2595 : tensor<192x768xf32>
    %v2597 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2598 = stablehlo.reshape %v2587 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2599 = stablehlo.reduce(%v2598 init: %v2597) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<768xf32>
    %v2600 = stablehlo.constant dense<0.003125> : tensor<768xf32>
    %v2601 = stablehlo.multiply %v2599, %v2600 : tensor<768xf32>
    %v2602 = stablehlo.subtract %b11_bfc1, %v2601 : tensor<768xf32>
    %v2603 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2604 = stablehlo.reshape %v2590 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2605 = stablehlo.reduce(%v2604 init: %v2603) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v2606 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v2607 = stablehlo.multiply %v2605, %v2606 : tensor<192xf32>
    %v2608 = stablehlo.subtract %b11_bt2, %v2607 : tensor<192xf32>
    %v2609 = stablehlo.reshape %v2382 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2610 = stablehlo.reshape %v2590 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2611 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2612 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v2613 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v2614 = stablehlo.reduce(%v2609 init: %v2611) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2615 = stablehlo.broadcast_in_dim %v2614, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v2616 = stablehlo.divide %v2615, %v2612 : tensor<32x197x192xf32>
    %v2617 = stablehlo.subtract %v2609, %v2616 : tensor<32x197x192xf32>
    %v2618 = stablehlo.multiply %v2617, %v2617 : tensor<32x197x192xf32>
    %v2619 = stablehlo.reduce(%v2618 init: %v2611) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2620 = stablehlo.broadcast_in_dim %v2619, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v2621 = stablehlo.divide %v2620, %v2612 : tensor<32x197x192xf32>
    %v2622 = stablehlo.add %v2621, %v2613 : tensor<32x197x192xf32>
    %v2623 = stablehlo.rsqrt %v2622 : tensor<32x197x192xf32>
    %v2624 = stablehlo.multiply %v2617, %v2623 : tensor<32x197x192xf32>
    %v2625 = stablehlo.multiply %v2610, %v2624 : tensor<32x197x192xf32>
    %v2626 = stablehlo.reduce(%v2625 init: %v2611) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v2627 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v2628 = stablehlo.multiply %v2626, %v2627 : tensor<192xf32>
    %v2629 = stablehlo.subtract %b11_g2, %v2628 : tensor<192xf32>
    %v2630 = stablehlo.reshape %v2590 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2631 = stablehlo.broadcast_in_dim %b11_g2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v2632 = stablehlo.multiply %v2630, %v2631 : tensor<32x197x192xf32>
    %v2633 = stablehlo.reshape %v2632 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2634 = stablehlo.reshape %v2633 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2635 = stablehlo.reshape %v2382 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2636 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2637 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v2638 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v2639 = stablehlo.reduce(%v2635 init: %v2636) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2640 = stablehlo.broadcast_in_dim %v2639, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v2641 = stablehlo.divide %v2640, %v2637 : tensor<32x197x192xf32>
    %v2642 = stablehlo.subtract %v2635, %v2641 : tensor<32x197x192xf32>
    %v2643 = stablehlo.multiply %v2642, %v2642 : tensor<32x197x192xf32>
    %v2644 = stablehlo.reduce(%v2643 init: %v2636) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2645 = stablehlo.broadcast_in_dim %v2644, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v2646 = stablehlo.divide %v2645, %v2637 : tensor<32x197x192xf32>
    %v2647 = stablehlo.add %v2646, %v2638 : tensor<32x197x192xf32>
    %v2648 = stablehlo.rsqrt %v2647 : tensor<32x197x192xf32>
    %v2649 = stablehlo.multiply %v2642, %v2648 : tensor<32x197x192xf32>
    %v2650 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v2651 = stablehlo.multiply %v2650, %v2634 : tensor<32x197x192xf32>
    %v2652 = stablehlo.reduce(%v2651 init: %v2636) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2653 = stablehlo.broadcast_in_dim %v2652, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v2654 = stablehlo.multiply %v2649, %v2651 : tensor<32x197x192xf32>
    %v2655 = stablehlo.reduce(%v2654 init: %v2636) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2656 = stablehlo.broadcast_in_dim %v2655, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v2657 = stablehlo.multiply %v2651, %v2637 : tensor<32x197x192xf32>
    %v2658 = stablehlo.subtract %v2657, %v2653 : tensor<32x197x192xf32>
    %v2659 = stablehlo.multiply %v2649, %v2656 : tensor<32x197x192xf32>
    %v2660 = stablehlo.subtract %v2658, %v2659 : tensor<32x197x192xf32>
    %v2661 = stablehlo.divide %v2648, %v2637 : tensor<32x197x192xf32>
    %v2662 = stablehlo.multiply %v2661, %v2660 : tensor<32x197x192xf32>
    %v2663 = stablehlo.reshape %v2662 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2664 = stablehlo.add %v2549, %v2663 : tensor<32x37824xf32>
    %v2665 = stablehlo.reshape %v2664 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2666 = stablehlo.dot_general %v2665, %b11_Wo, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v2667 = stablehlo.reshape %v2666 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2668 = stablehlo.reshape %v2376 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2669 = stablehlo.reshape %v2664 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2670 = stablehlo.dot_general %v2668, %v2669, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %v2671 = stablehlo.constant dense<0.003125> : tensor<192x192xf32>
    %v2672 = stablehlo.multiply %v2670, %v2671 : tensor<192x192xf32>
    %v2673 = stablehlo.subtract %b11_Wo, %v2672 : tensor<192x192xf32>
    %v2674 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2675 = stablehlo.reshape %v2664 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2676 = stablehlo.reduce(%v2675 init: %v2674) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v2677 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v2678 = stablehlo.multiply %v2676, %v2677 : tensor<192xf32>
    %v2679 = stablehlo.subtract %b11_bo, %v2678 : tensor<192xf32>
    %v2680 = stablehlo.reshape %v2667 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2681 = stablehlo.slice %v2680 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v2682 = stablehlo.reshape %v2681 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2683 = stablehlo.reshape %v2284 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2684 = stablehlo.transpose %v2683, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2685 = stablehlo.reshape %v2684 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2686 = stablehlo.reshape %v2682 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2687 = stablehlo.reshape %v2685 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2688 = stablehlo.dot_general %v2686, %v2687, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2689 = stablehlo.reshape %v2688 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2690 = stablehlo.reshape %v2300 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2691 = stablehlo.transpose %v2690, dims = [0, 2, 1] : (tensor<32x197x197xf32>) -> tensor<32x197x197xf32>
    %v2692 = stablehlo.reshape %v2691 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2693 = stablehlo.reshape %v2692 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2694 = stablehlo.reshape %v2682 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2695 = stablehlo.dot_general %v2693, %v2694, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2696 = stablehlo.reshape %v2695 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2697 = stablehlo.reshape %v2293 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2698 = stablehlo.reshape %v2689 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2699 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2700 = stablehlo.exponential %v2697 : tensor<32x197x197xf32>
    %v2701 = stablehlo.reduce(%v2700 init: %v2699) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2702 = stablehlo.broadcast_in_dim %v2701, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2703 = stablehlo.divide %v2700, %v2702 : tensor<32x197x197xf32>
    %v2704 = stablehlo.multiply %v2703, %v2698 : tensor<32x197x197xf32>
    %v2705 = stablehlo.reduce(%v2704 init: %v2699) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2706 = stablehlo.broadcast_in_dim %v2705, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2707 = stablehlo.subtract %v2698, %v2706 : tensor<32x197x197xf32>
    %v2708 = stablehlo.multiply %v2703, %v2707 : tensor<32x197x197xf32>
    %v2709 = stablehlo.reshape %v2708 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2710 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2711 = stablehlo.multiply %v2709, %v2710 : tensor<32x38809xf32>
    %v2712 = stablehlo.reshape %v2711 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2713 = stablehlo.reshape %v2281 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2714 = stablehlo.dot_general %v2712, %v2713, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2715 = stablehlo.reshape %v2714 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2716 = stablehlo.reshape %v2278 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2717 = stablehlo.transpose %v2716, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2718 = stablehlo.reshape %v2717 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2719 = stablehlo.reshape %v2718 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2720 = stablehlo.reshape %v2711 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2721 = stablehlo.dot_general %v2719, %v2720, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x64x197xf32>, tensor<32x197x197xf32>) -> tensor<32x64x197xf32>
    %v2722 = stablehlo.reshape %v2721 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2723 = stablehlo.reshape %v2722 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2724 = stablehlo.transpose %v2723, dims = [0, 2, 1] : (tensor<32x64x197xf32>) -> tensor<32x197x64xf32>
    %v2725 = stablehlo.reshape %v2724 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2726 = stablehlo.reshape %v2715 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2727 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2728 = stablehlo.pad %v2726, %v2727, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v2729 = stablehlo.reshape %v2728 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2730 = stablehlo.reshape %v2725 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2731 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2732 = stablehlo.pad %v2730, %v2731, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v2733 = stablehlo.reshape %v2732 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2734 = stablehlo.reshape %v2696 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2735 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2736 = stablehlo.pad %v2734, %v2735, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v2737 = stablehlo.reshape %v2736 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2738 = stablehlo.reshape %v2667 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2739 = stablehlo.slice %v2738 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v2740 = stablehlo.reshape %v2739 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2741 = stablehlo.reshape %v2317 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2742 = stablehlo.transpose %v2741, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2743 = stablehlo.reshape %v2742 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2744 = stablehlo.reshape %v2740 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2745 = stablehlo.reshape %v2743 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2746 = stablehlo.dot_general %v2744, %v2745, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2747 = stablehlo.reshape %v2746 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2748 = stablehlo.reshape %v2333 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2749 = stablehlo.transpose %v2748, dims = [0, 2, 1] : (tensor<32x197x197xf32>) -> tensor<32x197x197xf32>
    %v2750 = stablehlo.reshape %v2749 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2751 = stablehlo.reshape %v2750 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2752 = stablehlo.reshape %v2740 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2753 = stablehlo.dot_general %v2751, %v2752, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2754 = stablehlo.reshape %v2753 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2755 = stablehlo.reshape %v2326 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2756 = stablehlo.reshape %v2747 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2757 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2758 = stablehlo.exponential %v2755 : tensor<32x197x197xf32>
    %v2759 = stablehlo.reduce(%v2758 init: %v2757) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2760 = stablehlo.broadcast_in_dim %v2759, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2761 = stablehlo.divide %v2758, %v2760 : tensor<32x197x197xf32>
    %v2762 = stablehlo.multiply %v2761, %v2756 : tensor<32x197x197xf32>
    %v2763 = stablehlo.reduce(%v2762 init: %v2757) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2764 = stablehlo.broadcast_in_dim %v2763, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2765 = stablehlo.subtract %v2756, %v2764 : tensor<32x197x197xf32>
    %v2766 = stablehlo.multiply %v2761, %v2765 : tensor<32x197x197xf32>
    %v2767 = stablehlo.reshape %v2766 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2768 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2769 = stablehlo.multiply %v2767, %v2768 : tensor<32x38809xf32>
    %v2770 = stablehlo.reshape %v2769 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2771 = stablehlo.reshape %v2314 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2772 = stablehlo.dot_general %v2770, %v2771, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2773 = stablehlo.reshape %v2772 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2774 = stablehlo.reshape %v2311 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2775 = stablehlo.transpose %v2774, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2776 = stablehlo.reshape %v2775 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2777 = stablehlo.reshape %v2776 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2778 = stablehlo.reshape %v2769 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2779 = stablehlo.dot_general %v2777, %v2778, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x64x197xf32>, tensor<32x197x197xf32>) -> tensor<32x64x197xf32>
    %v2780 = stablehlo.reshape %v2779 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2781 = stablehlo.reshape %v2780 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2782 = stablehlo.transpose %v2781, dims = [0, 2, 1] : (tensor<32x64x197xf32>) -> tensor<32x197x64xf32>
    %v2783 = stablehlo.reshape %v2782 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2784 = stablehlo.reshape %v2773 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2785 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2786 = stablehlo.pad %v2784, %v2785, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v2787 = stablehlo.reshape %v2786 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2788 = stablehlo.reshape %v2783 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2789 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2790 = stablehlo.pad %v2788, %v2789, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v2791 = stablehlo.reshape %v2790 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2792 = stablehlo.reshape %v2754 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2793 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2794 = stablehlo.pad %v2792, %v2793, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v2795 = stablehlo.reshape %v2794 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2796 = stablehlo.add %v2729, %v2787 : tensor<32x37824xf32>
    %v2797 = stablehlo.add %v2733, %v2791 : tensor<32x37824xf32>
    %v2798 = stablehlo.add %v2737, %v2795 : tensor<32x37824xf32>
    %v2799 = stablehlo.reshape %v2667 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2800 = stablehlo.slice %v2799 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v2801 = stablehlo.reshape %v2800 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2802 = stablehlo.reshape %v2351 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2803 = stablehlo.transpose %v2802, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2804 = stablehlo.reshape %v2803 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2805 = stablehlo.reshape %v2801 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2806 = stablehlo.reshape %v2804 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2807 = stablehlo.dot_general %v2805, %v2806, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2808 = stablehlo.reshape %v2807 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2809 = stablehlo.reshape %v2367 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2810 = stablehlo.transpose %v2809, dims = [0, 2, 1] : (tensor<32x197x197xf32>) -> tensor<32x197x197xf32>
    %v2811 = stablehlo.reshape %v2810 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2812 = stablehlo.reshape %v2811 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2813 = stablehlo.reshape %v2801 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2814 = stablehlo.dot_general %v2812, %v2813, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2815 = stablehlo.reshape %v2814 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2816 = stablehlo.reshape %v2360 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2817 = stablehlo.reshape %v2808 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2818 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2819 = stablehlo.exponential %v2816 : tensor<32x197x197xf32>
    %v2820 = stablehlo.reduce(%v2819 init: %v2818) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2821 = stablehlo.broadcast_in_dim %v2820, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2822 = stablehlo.divide %v2819, %v2821 : tensor<32x197x197xf32>
    %v2823 = stablehlo.multiply %v2822, %v2817 : tensor<32x197x197xf32>
    %v2824 = stablehlo.reduce(%v2823 init: %v2818) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2825 = stablehlo.broadcast_in_dim %v2824, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2826 = stablehlo.subtract %v2817, %v2825 : tensor<32x197x197xf32>
    %v2827 = stablehlo.multiply %v2822, %v2826 : tensor<32x197x197xf32>
    %v2828 = stablehlo.reshape %v2827 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2829 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2830 = stablehlo.multiply %v2828, %v2829 : tensor<32x38809xf32>
    %v2831 = stablehlo.reshape %v2830 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2832 = stablehlo.reshape %v2348 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2833 = stablehlo.dot_general %v2831, %v2832, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2834 = stablehlo.reshape %v2833 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2835 = stablehlo.reshape %v2345 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2836 = stablehlo.transpose %v2835, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2837 = stablehlo.reshape %v2836 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2838 = stablehlo.reshape %v2837 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2839 = stablehlo.reshape %v2830 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2840 = stablehlo.dot_general %v2838, %v2839, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x64x197xf32>, tensor<32x197x197xf32>) -> tensor<32x64x197xf32>
    %v2841 = stablehlo.reshape %v2840 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2842 = stablehlo.reshape %v2841 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2843 = stablehlo.transpose %v2842, dims = [0, 2, 1] : (tensor<32x64x197xf32>) -> tensor<32x197x64xf32>
    %v2844 = stablehlo.reshape %v2843 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2845 = stablehlo.reshape %v2834 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2846 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2847 = stablehlo.pad %v2845, %v2846, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v2848 = stablehlo.reshape %v2847 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2849 = stablehlo.reshape %v2844 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2850 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2851 = stablehlo.pad %v2849, %v2850, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v2852 = stablehlo.reshape %v2851 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2853 = stablehlo.reshape %v2815 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2854 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2855 = stablehlo.pad %v2853, %v2854, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v2856 = stablehlo.reshape %v2855 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2857 = stablehlo.add %v2796, %v2848 : tensor<32x37824xf32>
    %v2858 = stablehlo.add %v2797, %v2852 : tensor<32x37824xf32>
    %v2859 = stablehlo.add %v2798, %v2856 : tensor<32x37824xf32>
    %v2860 = stablehlo.reshape %v2857 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2861 = stablehlo.dot_general %v2860, %b11_Wq, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v2862 = stablehlo.reshape %v2861 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2863 = stablehlo.reshape %v2260 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2864 = stablehlo.reshape %v2857 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2865 = stablehlo.dot_general %v2863, %v2864, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %v2866 = stablehlo.constant dense<0.003125> : tensor<192x192xf32>
    %v2867 = stablehlo.multiply %v2865, %v2866 : tensor<192x192xf32>
    %v2868 = stablehlo.subtract %b11_Wq, %v2867 : tensor<192x192xf32>
    %v2869 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2870 = stablehlo.reshape %v2857 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2871 = stablehlo.reduce(%v2870 init: %v2869) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v2872 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v2873 = stablehlo.multiply %v2871, %v2872 : tensor<192xf32>
    %v2874 = stablehlo.subtract %b11_bq, %v2873 : tensor<192xf32>
    %v2875 = stablehlo.reshape %v2858 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2876 = stablehlo.dot_general %v2875, %b11_Wk, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v2877 = stablehlo.reshape %v2876 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2878 = stablehlo.reshape %v2260 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2879 = stablehlo.reshape %v2858 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2880 = stablehlo.dot_general %v2878, %v2879, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %v2881 = stablehlo.constant dense<0.003125> : tensor<192x192xf32>
    %v2882 = stablehlo.multiply %v2880, %v2881 : tensor<192x192xf32>
    %v2883 = stablehlo.subtract %b11_Wk, %v2882 : tensor<192x192xf32>
    %v2884 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2885 = stablehlo.reshape %v2858 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2886 = stablehlo.reduce(%v2885 init: %v2884) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v2887 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v2888 = stablehlo.multiply %v2886, %v2887 : tensor<192xf32>
    %v2889 = stablehlo.subtract %b11_bk, %v2888 : tensor<192xf32>
    %v2890 = stablehlo.reshape %v2859 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2891 = stablehlo.dot_general %v2890, %b11_Wv, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v2892 = stablehlo.reshape %v2891 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2893 = stablehlo.reshape %v2260 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2894 = stablehlo.reshape %v2859 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2895 = stablehlo.dot_general %v2893, %v2894, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %v2896 = stablehlo.constant dense<0.003125> : tensor<192x192xf32>
    %v2897 = stablehlo.multiply %v2895, %v2896 : tensor<192x192xf32>
    %v2898 = stablehlo.subtract %b11_Wv, %v2897 : tensor<192x192xf32>
    %v2899 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2900 = stablehlo.reshape %v2859 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2901 = stablehlo.reduce(%v2900 init: %v2899) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v2902 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v2903 = stablehlo.multiply %v2901, %v2902 : tensor<192xf32>
    %v2904 = stablehlo.subtract %b11_bv, %v2903 : tensor<192xf32>
    %v2905 = stablehlo.add %v2862, %v2877 : tensor<32x37824xf32>
    %v2906 = stablehlo.add %v2905, %v2892 : tensor<32x37824xf32>
    %v2907 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2908 = stablehlo.reshape %v2906 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2909 = stablehlo.reduce(%v2908 init: %v2907) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v2910 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v2911 = stablehlo.multiply %v2909, %v2910 : tensor<192xf32>
    %v2912 = stablehlo.subtract %b11_bt1, %v2911 : tensor<192xf32>
    %v2913 = stablehlo.reshape %v2232 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2914 = stablehlo.reshape %v2906 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2915 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2916 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v2917 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v2918 = stablehlo.reduce(%v2913 init: %v2915) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2919 = stablehlo.broadcast_in_dim %v2918, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v2920 = stablehlo.divide %v2919, %v2916 : tensor<32x197x192xf32>
    %v2921 = stablehlo.subtract %v2913, %v2920 : tensor<32x197x192xf32>
    %v2922 = stablehlo.multiply %v2921, %v2921 : tensor<32x197x192xf32>
    %v2923 = stablehlo.reduce(%v2922 init: %v2915) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2924 = stablehlo.broadcast_in_dim %v2923, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v2925 = stablehlo.divide %v2924, %v2916 : tensor<32x197x192xf32>
    %v2926 = stablehlo.add %v2925, %v2917 : tensor<32x197x192xf32>
    %v2927 = stablehlo.rsqrt %v2926 : tensor<32x197x192xf32>
    %v2928 = stablehlo.multiply %v2921, %v2927 : tensor<32x197x192xf32>
    %v2929 = stablehlo.multiply %v2914, %v2928 : tensor<32x197x192xf32>
    %v2930 = stablehlo.reduce(%v2929 init: %v2915) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v2931 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v2932 = stablehlo.multiply %v2930, %v2931 : tensor<192xf32>
    %v2933 = stablehlo.subtract %b11_g1, %v2932 : tensor<192xf32>
    %v2934 = stablehlo.reshape %v2906 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2935 = stablehlo.broadcast_in_dim %b11_g1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v2936 = stablehlo.multiply %v2934, %v2935 : tensor<32x197x192xf32>
    %v2937 = stablehlo.reshape %v2936 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2938 = stablehlo.reshape %v2937 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2939 = stablehlo.reshape %v2232 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2940 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2941 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v2942 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v2943 = stablehlo.reduce(%v2939 init: %v2940) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2944 = stablehlo.broadcast_in_dim %v2943, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v2945 = stablehlo.divide %v2944, %v2941 : tensor<32x197x192xf32>
    %v2946 = stablehlo.subtract %v2939, %v2945 : tensor<32x197x192xf32>
    %v2947 = stablehlo.multiply %v2946, %v2946 : tensor<32x197x192xf32>
    %v2948 = stablehlo.reduce(%v2947 init: %v2940) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2949 = stablehlo.broadcast_in_dim %v2948, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v2950 = stablehlo.divide %v2949, %v2941 : tensor<32x197x192xf32>
    %v2951 = stablehlo.add %v2950, %v2942 : tensor<32x197x192xf32>
    %v2952 = stablehlo.rsqrt %v2951 : tensor<32x197x192xf32>
    %v2953 = stablehlo.multiply %v2946, %v2952 : tensor<32x197x192xf32>
    %v2954 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v2955 = stablehlo.multiply %v2954, %v2938 : tensor<32x197x192xf32>
    %v2956 = stablehlo.reduce(%v2955 init: %v2940) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2957 = stablehlo.broadcast_in_dim %v2956, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v2958 = stablehlo.multiply %v2953, %v2955 : tensor<32x197x192xf32>
    %v2959 = stablehlo.reduce(%v2958 init: %v2940) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2960 = stablehlo.broadcast_in_dim %v2959, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v2961 = stablehlo.multiply %v2955, %v2941 : tensor<32x197x192xf32>
    %v2962 = stablehlo.subtract %v2961, %v2957 : tensor<32x197x192xf32>
    %v2963 = stablehlo.multiply %v2953, %v2960 : tensor<32x197x192xf32>
    %v2964 = stablehlo.subtract %v2962, %v2963 : tensor<32x197x192xf32>
    %v2965 = stablehlo.divide %v2952, %v2941 : tensor<32x197x192xf32>
    %v2966 = stablehlo.multiply %v2965, %v2964 : tensor<32x197x192xf32>
    %v2967 = stablehlo.reshape %v2966 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2968 = stablehlo.add %v2664, %v2967 : tensor<32x37824xf32>
    %v2969 = stablehlo.reshape %v2968 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2970 = stablehlo.dot_general %v2969, %b10_Wfc2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<768x192xf32>) -> tensor<32x197x768xf32>
    %v2971 = stablehlo.reshape %v2970 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2972 = stablehlo.reshape %v2226 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2973 = stablehlo.reshape %v2968 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2974 = stablehlo.dot_general %v2972, %v2973, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<32x197x192xf32>) -> tensor<768x192xf32>
    %v2975 = stablehlo.constant dense<0.003125> : tensor<768x192xf32>
    %v2976 = stablehlo.multiply %v2974, %v2975 : tensor<768x192xf32>
    %v2977 = stablehlo.subtract %b10_Wfc2, %v2976 : tensor<768x192xf32>
    %v2978 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2979 = stablehlo.reshape %v2968 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2980 = stablehlo.reduce(%v2979 init: %v2978) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v2981 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v2982 = stablehlo.multiply %v2980, %v2981 : tensor<192xf32>
    %v2983 = stablehlo.subtract %b10_bfc2, %v2982 : tensor<192xf32>
    %v2984 = stablehlo.multiply %v2213, %v2213 : tensor<32x151296xf32>
    %v2985 = stablehlo.multiply %v2984, %v2213 : tensor<32x151296xf32>
    %v2986 = stablehlo.constant dense<0.044715> : tensor<32x151296xf32>
    %v2987 = stablehlo.multiply %v2986, %v2985 : tensor<32x151296xf32>
    %v2988 = stablehlo.add %v2213, %v2987 : tensor<32x151296xf32>
    %v2989 = stablehlo.constant dense<0.7978845608028654> : tensor<32x151296xf32>
    %v2990 = stablehlo.multiply %v2989, %v2988 : tensor<32x151296xf32>
    %v2991 = stablehlo.tanh %v2990 : tensor<32x151296xf32>
    %v2992 = stablehlo.constant dense<1.0> : tensor<32x151296xf32>
    %v2993 = stablehlo.add %v2992, %v2991 : tensor<32x151296xf32>
    %v2994 = stablehlo.constant dense<0.5> : tensor<32x151296xf32>
    %v2995 = stablehlo.multiply %v2994, %v2993 : tensor<32x151296xf32>
    %v2996 = stablehlo.multiply %v2991, %v2991 : tensor<32x151296xf32>
    %v2997 = stablehlo.subtract %v2992, %v2996 : tensor<32x151296xf32>
    %v2998 = stablehlo.multiply %v2994, %v2213 : tensor<32x151296xf32>
    %v2999 = stablehlo.multiply %v2998, %v2997 : tensor<32x151296xf32>
    %v3000 = stablehlo.constant dense<0.134145> : tensor<32x151296xf32>
    %v3001 = stablehlo.multiply %v3000, %v2984 : tensor<32x151296xf32>
    %v3002 = stablehlo.add %v2992, %v3001 : tensor<32x151296xf32>
    %v3003 = stablehlo.multiply %v2989, %v3002 : tensor<32x151296xf32>
    %v3004 = stablehlo.multiply %v2999, %v3003 : tensor<32x151296xf32>
    %v3005 = stablehlo.add %v2995, %v3004 : tensor<32x151296xf32>
    %v3006 = stablehlo.multiply %v2971, %v3005 : tensor<32x151296xf32>
    %v3007 = stablehlo.reshape %v3006 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3008 = stablehlo.dot_general %v3007, %b10_Wfc1, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<192x768xf32>) -> tensor<32x197x192xf32>
    %v3009 = stablehlo.reshape %v3008 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v3010 = stablehlo.reshape %v2208 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3011 = stablehlo.reshape %v3006 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3012 = stablehlo.dot_general %v3010, %v3011, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x768xf32>) -> tensor<192x768xf32>
    %v3013 = stablehlo.constant dense<0.003125> : tensor<192x768xf32>
    %v3014 = stablehlo.multiply %v3012, %v3013 : tensor<192x768xf32>
    %v3015 = stablehlo.subtract %b10_Wfc1, %v3014 : tensor<192x768xf32>
    %v3016 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3017 = stablehlo.reshape %v3006 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3018 = stablehlo.reduce(%v3017 init: %v3016) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<768xf32>
    %v3019 = stablehlo.constant dense<0.003125> : tensor<768xf32>
    %v3020 = stablehlo.multiply %v3018, %v3019 : tensor<768xf32>
    %v3021 = stablehlo.subtract %b10_bfc1, %v3020 : tensor<768xf32>
    %v3022 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3023 = stablehlo.reshape %v3009 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3024 = stablehlo.reduce(%v3023 init: %v3022) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v3025 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v3026 = stablehlo.multiply %v3024, %v3025 : tensor<192xf32>
    %v3027 = stablehlo.subtract %b10_bt2, %v3026 : tensor<192xf32>
    %v3028 = stablehlo.reshape %v2180 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3029 = stablehlo.reshape %v3009 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3030 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3031 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v3032 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v3033 = stablehlo.reduce(%v3028 init: %v3030) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3034 = stablehlo.broadcast_in_dim %v3033, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v3035 = stablehlo.divide %v3034, %v3031 : tensor<32x197x192xf32>
    %v3036 = stablehlo.subtract %v3028, %v3035 : tensor<32x197x192xf32>
    %v3037 = stablehlo.multiply %v3036, %v3036 : tensor<32x197x192xf32>
    %v3038 = stablehlo.reduce(%v3037 init: %v3030) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3039 = stablehlo.broadcast_in_dim %v3038, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v3040 = stablehlo.divide %v3039, %v3031 : tensor<32x197x192xf32>
    %v3041 = stablehlo.add %v3040, %v3032 : tensor<32x197x192xf32>
    %v3042 = stablehlo.rsqrt %v3041 : tensor<32x197x192xf32>
    %v3043 = stablehlo.multiply %v3036, %v3042 : tensor<32x197x192xf32>
    %v3044 = stablehlo.multiply %v3029, %v3043 : tensor<32x197x192xf32>
    %v3045 = stablehlo.reduce(%v3044 init: %v3030) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v3046 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v3047 = stablehlo.multiply %v3045, %v3046 : tensor<192xf32>
    %v3048 = stablehlo.subtract %b10_g2, %v3047 : tensor<192xf32>
    %v3049 = stablehlo.reshape %v3009 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3050 = stablehlo.broadcast_in_dim %b10_g2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v3051 = stablehlo.multiply %v3049, %v3050 : tensor<32x197x192xf32>
    %v3052 = stablehlo.reshape %v3051 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v3053 = stablehlo.reshape %v3052 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3054 = stablehlo.reshape %v2180 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3055 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3056 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v3057 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v3058 = stablehlo.reduce(%v3054 init: %v3055) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3059 = stablehlo.broadcast_in_dim %v3058, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v3060 = stablehlo.divide %v3059, %v3056 : tensor<32x197x192xf32>
    %v3061 = stablehlo.subtract %v3054, %v3060 : tensor<32x197x192xf32>
    %v3062 = stablehlo.multiply %v3061, %v3061 : tensor<32x197x192xf32>
    %v3063 = stablehlo.reduce(%v3062 init: %v3055) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3064 = stablehlo.broadcast_in_dim %v3063, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v3065 = stablehlo.divide %v3064, %v3056 : tensor<32x197x192xf32>
    %v3066 = stablehlo.add %v3065, %v3057 : tensor<32x197x192xf32>
    %v3067 = stablehlo.rsqrt %v3066 : tensor<32x197x192xf32>
    %v3068 = stablehlo.multiply %v3061, %v3067 : tensor<32x197x192xf32>
    %v3069 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v3070 = stablehlo.multiply %v3069, %v3053 : tensor<32x197x192xf32>
    %v3071 = stablehlo.reduce(%v3070 init: %v3055) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3072 = stablehlo.broadcast_in_dim %v3071, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v3073 = stablehlo.multiply %v3068, %v3070 : tensor<32x197x192xf32>
    %v3074 = stablehlo.reduce(%v3073 init: %v3055) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3075 = stablehlo.broadcast_in_dim %v3074, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v3076 = stablehlo.multiply %v3070, %v3056 : tensor<32x197x192xf32>
    %v3077 = stablehlo.subtract %v3076, %v3072 : tensor<32x197x192xf32>
    %v3078 = stablehlo.multiply %v3068, %v3075 : tensor<32x197x192xf32>
    %v3079 = stablehlo.subtract %v3077, %v3078 : tensor<32x197x192xf32>
    %v3080 = stablehlo.divide %v3067, %v3056 : tensor<32x197x192xf32>
    %v3081 = stablehlo.multiply %v3080, %v3079 : tensor<32x197x192xf32>
    %v3082 = stablehlo.reshape %v3081 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v3083 = stablehlo.add %v2968, %v3082 : tensor<32x37824xf32>
    %v3084 = stablehlo.reshape %v3083 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3085 = stablehlo.dot_general %v3084, %b10_Wo, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v3086 = stablehlo.reshape %v3085 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v3087 = stablehlo.reshape %v2174 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3088 = stablehlo.reshape %v3083 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3089 = stablehlo.dot_general %v3087, %v3088, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %v3090 = stablehlo.constant dense<0.003125> : tensor<192x192xf32>
    %v3091 = stablehlo.multiply %v3089, %v3090 : tensor<192x192xf32>
    %v3092 = stablehlo.subtract %b10_Wo, %v3091 : tensor<192x192xf32>
    %v3093 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3094 = stablehlo.reshape %v3083 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3095 = stablehlo.reduce(%v3094 init: %v3093) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v3096 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v3097 = stablehlo.multiply %v3095, %v3096 : tensor<192xf32>
    %v3098 = stablehlo.subtract %b10_bo, %v3097 : tensor<192xf32>
    %v3099 = stablehlo.reshape %v3086 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3100 = stablehlo.slice %v3099 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v3101 = stablehlo.reshape %v3100 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3102 = stablehlo.reshape %v2082 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3103 = stablehlo.transpose %v3102, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3104 = stablehlo.reshape %v3103 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3105 = stablehlo.reshape %v3101 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3106 = stablehlo.reshape %v3104 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3107 = stablehlo.dot_general %v3105, %v3106, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3108 = stablehlo.reshape %v3107 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3109 = stablehlo.reshape %v2098 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3110 = stablehlo.transpose %v3109, dims = [0, 2, 1] : (tensor<32x197x197xf32>) -> tensor<32x197x197xf32>
    %v3111 = stablehlo.reshape %v3110 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3112 = stablehlo.reshape %v3111 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3113 = stablehlo.reshape %v3101 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3114 = stablehlo.dot_general %v3112, %v3113, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3115 = stablehlo.reshape %v3114 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3116 = stablehlo.reshape %v2091 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3117 = stablehlo.reshape %v3108 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3118 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3119 = stablehlo.exponential %v3116 : tensor<32x197x197xf32>
    %v3120 = stablehlo.reduce(%v3119 init: %v3118) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3121 = stablehlo.broadcast_in_dim %v3120, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3122 = stablehlo.divide %v3119, %v3121 : tensor<32x197x197xf32>
    %v3123 = stablehlo.multiply %v3122, %v3117 : tensor<32x197x197xf32>
    %v3124 = stablehlo.reduce(%v3123 init: %v3118) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3125 = stablehlo.broadcast_in_dim %v3124, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3126 = stablehlo.subtract %v3117, %v3125 : tensor<32x197x197xf32>
    %v3127 = stablehlo.multiply %v3122, %v3126 : tensor<32x197x197xf32>
    %v3128 = stablehlo.reshape %v3127 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3129 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3130 = stablehlo.multiply %v3128, %v3129 : tensor<32x38809xf32>
    %v3131 = stablehlo.reshape %v3130 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3132 = stablehlo.reshape %v2079 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3133 = stablehlo.dot_general %v3131, %v3132, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3134 = stablehlo.reshape %v3133 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3135 = stablehlo.reshape %v2076 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3136 = stablehlo.transpose %v3135, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3137 = stablehlo.reshape %v3136 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3138 = stablehlo.reshape %v3137 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3139 = stablehlo.reshape %v3130 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3140 = stablehlo.dot_general %v3138, %v3139, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x64x197xf32>, tensor<32x197x197xf32>) -> tensor<32x64x197xf32>
    %v3141 = stablehlo.reshape %v3140 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3142 = stablehlo.reshape %v3141 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3143 = stablehlo.transpose %v3142, dims = [0, 2, 1] : (tensor<32x64x197xf32>) -> tensor<32x197x64xf32>
    %v3144 = stablehlo.reshape %v3143 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3145 = stablehlo.reshape %v3134 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3146 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3147 = stablehlo.pad %v3145, %v3146, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v3148 = stablehlo.reshape %v3147 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v3149 = stablehlo.reshape %v3144 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3150 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3151 = stablehlo.pad %v3149, %v3150, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v3152 = stablehlo.reshape %v3151 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v3153 = stablehlo.reshape %v3115 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3154 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3155 = stablehlo.pad %v3153, %v3154, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v3156 = stablehlo.reshape %v3155 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v3157 = stablehlo.reshape %v3086 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3158 = stablehlo.slice %v3157 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v3159 = stablehlo.reshape %v3158 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3160 = stablehlo.reshape %v2115 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3161 = stablehlo.transpose %v3160, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3162 = stablehlo.reshape %v3161 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3163 = stablehlo.reshape %v3159 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3164 = stablehlo.reshape %v3162 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3165 = stablehlo.dot_general %v3163, %v3164, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3166 = stablehlo.reshape %v3165 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3167 = stablehlo.reshape %v2131 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3168 = stablehlo.transpose %v3167, dims = [0, 2, 1] : (tensor<32x197x197xf32>) -> tensor<32x197x197xf32>
    %v3169 = stablehlo.reshape %v3168 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3170 = stablehlo.reshape %v3169 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3171 = stablehlo.reshape %v3159 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3172 = stablehlo.dot_general %v3170, %v3171, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3173 = stablehlo.reshape %v3172 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3174 = stablehlo.reshape %v2124 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3175 = stablehlo.reshape %v3166 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3176 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3177 = stablehlo.exponential %v3174 : tensor<32x197x197xf32>
    %v3178 = stablehlo.reduce(%v3177 init: %v3176) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3179 = stablehlo.broadcast_in_dim %v3178, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3180 = stablehlo.divide %v3177, %v3179 : tensor<32x197x197xf32>
    %v3181 = stablehlo.multiply %v3180, %v3175 : tensor<32x197x197xf32>
    %v3182 = stablehlo.reduce(%v3181 init: %v3176) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3183 = stablehlo.broadcast_in_dim %v3182, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3184 = stablehlo.subtract %v3175, %v3183 : tensor<32x197x197xf32>
    %v3185 = stablehlo.multiply %v3180, %v3184 : tensor<32x197x197xf32>
    %v3186 = stablehlo.reshape %v3185 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3187 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3188 = stablehlo.multiply %v3186, %v3187 : tensor<32x38809xf32>
    %v3189 = stablehlo.reshape %v3188 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3190 = stablehlo.reshape %v2112 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3191 = stablehlo.dot_general %v3189, %v3190, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3192 = stablehlo.reshape %v3191 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3193 = stablehlo.reshape %v2109 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3194 = stablehlo.transpose %v3193, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3195 = stablehlo.reshape %v3194 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3196 = stablehlo.reshape %v3195 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3197 = stablehlo.reshape %v3188 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3198 = stablehlo.dot_general %v3196, %v3197, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x64x197xf32>, tensor<32x197x197xf32>) -> tensor<32x64x197xf32>
    %v3199 = stablehlo.reshape %v3198 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3200 = stablehlo.reshape %v3199 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3201 = stablehlo.transpose %v3200, dims = [0, 2, 1] : (tensor<32x64x197xf32>) -> tensor<32x197x64xf32>
    %v3202 = stablehlo.reshape %v3201 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3203 = stablehlo.reshape %v3192 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3204 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3205 = stablehlo.pad %v3203, %v3204, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v3206 = stablehlo.reshape %v3205 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v3207 = stablehlo.reshape %v3202 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3208 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3209 = stablehlo.pad %v3207, %v3208, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v3210 = stablehlo.reshape %v3209 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v3211 = stablehlo.reshape %v3173 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3212 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3213 = stablehlo.pad %v3211, %v3212, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v3214 = stablehlo.reshape %v3213 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v3215 = stablehlo.add %v3148, %v3206 : tensor<32x37824xf32>
    %v3216 = stablehlo.add %v3152, %v3210 : tensor<32x37824xf32>
    %v3217 = stablehlo.add %v3156, %v3214 : tensor<32x37824xf32>
    %v3218 = stablehlo.reshape %v3086 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3219 = stablehlo.slice %v3218 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v3220 = stablehlo.reshape %v3219 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3221 = stablehlo.reshape %v2149 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3222 = stablehlo.transpose %v3221, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3223 = stablehlo.reshape %v3222 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3224 = stablehlo.reshape %v3220 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3225 = stablehlo.reshape %v3223 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3226 = stablehlo.dot_general %v3224, %v3225, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3227 = stablehlo.reshape %v3226 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3228 = stablehlo.reshape %v2165 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3229 = stablehlo.transpose %v3228, dims = [0, 2, 1] : (tensor<32x197x197xf32>) -> tensor<32x197x197xf32>
    %v3230 = stablehlo.reshape %v3229 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3231 = stablehlo.reshape %v3230 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3232 = stablehlo.reshape %v3220 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3233 = stablehlo.dot_general %v3231, %v3232, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3234 = stablehlo.reshape %v3233 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3235 = stablehlo.reshape %v2158 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3236 = stablehlo.reshape %v3227 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3237 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3238 = stablehlo.exponential %v3235 : tensor<32x197x197xf32>
    %v3239 = stablehlo.reduce(%v3238 init: %v3237) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3240 = stablehlo.broadcast_in_dim %v3239, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3241 = stablehlo.divide %v3238, %v3240 : tensor<32x197x197xf32>
    %v3242 = stablehlo.multiply %v3241, %v3236 : tensor<32x197x197xf32>
    %v3243 = stablehlo.reduce(%v3242 init: %v3237) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3244 = stablehlo.broadcast_in_dim %v3243, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3245 = stablehlo.subtract %v3236, %v3244 : tensor<32x197x197xf32>
    %v3246 = stablehlo.multiply %v3241, %v3245 : tensor<32x197x197xf32>
    %v3247 = stablehlo.reshape %v3246 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3248 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3249 = stablehlo.multiply %v3247, %v3248 : tensor<32x38809xf32>
    %v3250 = stablehlo.reshape %v3249 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3251 = stablehlo.reshape %v2146 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3252 = stablehlo.dot_general %v3250, %v3251, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3253 = stablehlo.reshape %v3252 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3254 = stablehlo.reshape %v2143 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3255 = stablehlo.transpose %v3254, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3256 = stablehlo.reshape %v3255 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3257 = stablehlo.reshape %v3256 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3258 = stablehlo.reshape %v3249 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3259 = stablehlo.dot_general %v3257, %v3258, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x64x197xf32>, tensor<32x197x197xf32>) -> tensor<32x64x197xf32>
    %v3260 = stablehlo.reshape %v3259 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3261 = stablehlo.reshape %v3260 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3262 = stablehlo.transpose %v3261, dims = [0, 2, 1] : (tensor<32x64x197xf32>) -> tensor<32x197x64xf32>
    %v3263 = stablehlo.reshape %v3262 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3264 = stablehlo.reshape %v3253 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3265 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3266 = stablehlo.pad %v3264, %v3265, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v3267 = stablehlo.reshape %v3266 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v3268 = stablehlo.reshape %v3263 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3269 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3270 = stablehlo.pad %v3268, %v3269, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v3271 = stablehlo.reshape %v3270 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v3272 = stablehlo.reshape %v3234 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3273 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3274 = stablehlo.pad %v3272, %v3273, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v3275 = stablehlo.reshape %v3274 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v3276 = stablehlo.add %v3215, %v3267 : tensor<32x37824xf32>
    %v3277 = stablehlo.add %v3216, %v3271 : tensor<32x37824xf32>
    %v3278 = stablehlo.add %v3217, %v3275 : tensor<32x37824xf32>
    %v3279 = stablehlo.reshape %v3276 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3280 = stablehlo.dot_general %v3279, %b10_Wq, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v3281 = stablehlo.reshape %v3280 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v3282 = stablehlo.reshape %v2058 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3283 = stablehlo.reshape %v3276 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3284 = stablehlo.dot_general %v3282, %v3283, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %v3285 = stablehlo.constant dense<0.003125> : tensor<192x192xf32>
    %v3286 = stablehlo.multiply %v3284, %v3285 : tensor<192x192xf32>
    %v3287 = stablehlo.subtract %b10_Wq, %v3286 : tensor<192x192xf32>
    %v3288 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3289 = stablehlo.reshape %v3276 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3290 = stablehlo.reduce(%v3289 init: %v3288) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v3291 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v3292 = stablehlo.multiply %v3290, %v3291 : tensor<192xf32>
    %v3293 = stablehlo.subtract %b10_bq, %v3292 : tensor<192xf32>
    %v3294 = stablehlo.reshape %v3277 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3295 = stablehlo.dot_general %v3294, %b10_Wk, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v3296 = stablehlo.reshape %v3295 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v3297 = stablehlo.reshape %v2058 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3298 = stablehlo.reshape %v3277 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3299 = stablehlo.dot_general %v3297, %v3298, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %v3300 = stablehlo.constant dense<0.003125> : tensor<192x192xf32>
    %v3301 = stablehlo.multiply %v3299, %v3300 : tensor<192x192xf32>
    %v3302 = stablehlo.subtract %b10_Wk, %v3301 : tensor<192x192xf32>
    %v3303 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3304 = stablehlo.reshape %v3277 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3305 = stablehlo.reduce(%v3304 init: %v3303) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v3306 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v3307 = stablehlo.multiply %v3305, %v3306 : tensor<192xf32>
    %v3308 = stablehlo.subtract %b10_bk, %v3307 : tensor<192xf32>
    %v3309 = stablehlo.reshape %v3278 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3310 = stablehlo.dot_general %v3309, %b10_Wv, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v3311 = stablehlo.reshape %v3310 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v3312 = stablehlo.reshape %v2058 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3313 = stablehlo.reshape %v3278 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3314 = stablehlo.dot_general %v3312, %v3313, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %v3315 = stablehlo.constant dense<0.003125> : tensor<192x192xf32>
    %v3316 = stablehlo.multiply %v3314, %v3315 : tensor<192x192xf32>
    %v3317 = stablehlo.subtract %b10_Wv, %v3316 : tensor<192x192xf32>
    %v3318 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3319 = stablehlo.reshape %v3278 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3320 = stablehlo.reduce(%v3319 init: %v3318) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v3321 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v3322 = stablehlo.multiply %v3320, %v3321 : tensor<192xf32>
    %v3323 = stablehlo.subtract %b10_bv, %v3322 : tensor<192xf32>
    %v3324 = stablehlo.add %v3281, %v3296 : tensor<32x37824xf32>
    %v3325 = stablehlo.add %v3324, %v3311 : tensor<32x37824xf32>
    %v3326 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3327 = stablehlo.reshape %v3325 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3328 = stablehlo.reduce(%v3327 init: %v3326) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v3329 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v3330 = stablehlo.multiply %v3328, %v3329 : tensor<192xf32>
    %v3331 = stablehlo.subtract %b10_bt1, %v3330 : tensor<192xf32>
    %v3332 = stablehlo.reshape %v2030 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3333 = stablehlo.reshape %v3325 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3334 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3335 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v3336 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v3337 = stablehlo.reduce(%v3332 init: %v3334) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3338 = stablehlo.broadcast_in_dim %v3337, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v3339 = stablehlo.divide %v3338, %v3335 : tensor<32x197x192xf32>
    %v3340 = stablehlo.subtract %v3332, %v3339 : tensor<32x197x192xf32>
    %v3341 = stablehlo.multiply %v3340, %v3340 : tensor<32x197x192xf32>
    %v3342 = stablehlo.reduce(%v3341 init: %v3334) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3343 = stablehlo.broadcast_in_dim %v3342, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v3344 = stablehlo.divide %v3343, %v3335 : tensor<32x197x192xf32>
    %v3345 = stablehlo.add %v3344, %v3336 : tensor<32x197x192xf32>
    %v3346 = stablehlo.rsqrt %v3345 : tensor<32x197x192xf32>
    %v3347 = stablehlo.multiply %v3340, %v3346 : tensor<32x197x192xf32>
    %v3348 = stablehlo.multiply %v3333, %v3347 : tensor<32x197x192xf32>
    %v3349 = stablehlo.reduce(%v3348 init: %v3334) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v3350 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v3351 = stablehlo.multiply %v3349, %v3350 : tensor<192xf32>
    %v3352 = stablehlo.subtract %b10_g1, %v3351 : tensor<192xf32>
    %v3353 = stablehlo.reshape %v3325 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3354 = stablehlo.broadcast_in_dim %b10_g1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v3355 = stablehlo.multiply %v3353, %v3354 : tensor<32x197x192xf32>
    %v3356 = stablehlo.reshape %v3355 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v3357 = stablehlo.reshape %v3356 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3358 = stablehlo.reshape %v2030 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3359 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3360 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v3361 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v3362 = stablehlo.reduce(%v3358 init: %v3359) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3363 = stablehlo.broadcast_in_dim %v3362, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v3364 = stablehlo.divide %v3363, %v3360 : tensor<32x197x192xf32>
    %v3365 = stablehlo.subtract %v3358, %v3364 : tensor<32x197x192xf32>
    %v3366 = stablehlo.multiply %v3365, %v3365 : tensor<32x197x192xf32>
    %v3367 = stablehlo.reduce(%v3366 init: %v3359) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3368 = stablehlo.broadcast_in_dim %v3367, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v3369 = stablehlo.divide %v3368, %v3360 : tensor<32x197x192xf32>
    %v3370 = stablehlo.add %v3369, %v3361 : tensor<32x197x192xf32>
    %v3371 = stablehlo.rsqrt %v3370 : tensor<32x197x192xf32>
    %v3372 = stablehlo.multiply %v3365, %v3371 : tensor<32x197x192xf32>
    %v3373 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v3374 = stablehlo.multiply %v3373, %v3357 : tensor<32x197x192xf32>
    %v3375 = stablehlo.reduce(%v3374 init: %v3359) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3376 = stablehlo.broadcast_in_dim %v3375, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v3377 = stablehlo.multiply %v3372, %v3374 : tensor<32x197x192xf32>
    %v3378 = stablehlo.reduce(%v3377 init: %v3359) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3379 = stablehlo.broadcast_in_dim %v3378, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v3380 = stablehlo.multiply %v3374, %v3360 : tensor<32x197x192xf32>
    %v3381 = stablehlo.subtract %v3380, %v3376 : tensor<32x197x192xf32>
    %v3382 = stablehlo.multiply %v3372, %v3379 : tensor<32x197x192xf32>
    %v3383 = stablehlo.subtract %v3381, %v3382 : tensor<32x197x192xf32>
    %v3384 = stablehlo.divide %v3371, %v3360 : tensor<32x197x192xf32>
    %v3385 = stablehlo.multiply %v3384, %v3383 : tensor<32x197x192xf32>
    %v3386 = stablehlo.reshape %v3385 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v3387 = stablehlo.add %v3083, %v3386 : tensor<32x37824xf32>
    %v3388 = stablehlo.reshape %v3387 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3389 = stablehlo.dot_general %v3388, %b9_Wfc2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<768x192xf32>) -> tensor<32x197x768xf32>
    %v3390 = stablehlo.reshape %v3389 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3391 = stablehlo.reshape %v2024 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3392 = stablehlo.reshape %v3387 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3393 = stablehlo.dot_general %v3391, %v3392, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<32x197x192xf32>) -> tensor<768x192xf32>
    %v3394 = stablehlo.constant dense<0.003125> : tensor<768x192xf32>
    %v3395 = stablehlo.multiply %v3393, %v3394 : tensor<768x192xf32>
    %v3396 = stablehlo.subtract %b9_Wfc2, %v3395 : tensor<768x192xf32>
    %v3397 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3398 = stablehlo.reshape %v3387 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3399 = stablehlo.reduce(%v3398 init: %v3397) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v3400 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v3401 = stablehlo.multiply %v3399, %v3400 : tensor<192xf32>
    %v3402 = stablehlo.subtract %b9_bfc2, %v3401 : tensor<192xf32>
    %v3403 = stablehlo.multiply %v2011, %v2011 : tensor<32x151296xf32>
    %v3404 = stablehlo.multiply %v3403, %v2011 : tensor<32x151296xf32>
    %v3405 = stablehlo.constant dense<0.044715> : tensor<32x151296xf32>
    %v3406 = stablehlo.multiply %v3405, %v3404 : tensor<32x151296xf32>
    %v3407 = stablehlo.add %v2011, %v3406 : tensor<32x151296xf32>
    %v3408 = stablehlo.constant dense<0.7978845608028654> : tensor<32x151296xf32>
    %v3409 = stablehlo.multiply %v3408, %v3407 : tensor<32x151296xf32>
    %v3410 = stablehlo.tanh %v3409 : tensor<32x151296xf32>
    %v3411 = stablehlo.constant dense<1.0> : tensor<32x151296xf32>
    %v3412 = stablehlo.add %v3411, %v3410 : tensor<32x151296xf32>
    %v3413 = stablehlo.constant dense<0.5> : tensor<32x151296xf32>
    %v3414 = stablehlo.multiply %v3413, %v3412 : tensor<32x151296xf32>
    %v3415 = stablehlo.multiply %v3410, %v3410 : tensor<32x151296xf32>
    %v3416 = stablehlo.subtract %v3411, %v3415 : tensor<32x151296xf32>
    %v3417 = stablehlo.multiply %v3413, %v2011 : tensor<32x151296xf32>
    %v3418 = stablehlo.multiply %v3417, %v3416 : tensor<32x151296xf32>
    %v3419 = stablehlo.constant dense<0.134145> : tensor<32x151296xf32>
    %v3420 = stablehlo.multiply %v3419, %v3403 : tensor<32x151296xf32>
    %v3421 = stablehlo.add %v3411, %v3420 : tensor<32x151296xf32>
    %v3422 = stablehlo.multiply %v3408, %v3421 : tensor<32x151296xf32>
    %v3423 = stablehlo.multiply %v3418, %v3422 : tensor<32x151296xf32>
    %v3424 = stablehlo.add %v3414, %v3423 : tensor<32x151296xf32>
    %v3425 = stablehlo.multiply %v3390, %v3424 : tensor<32x151296xf32>
    %v3426 = stablehlo.reshape %v3425 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3427 = stablehlo.dot_general %v3426, %b9_Wfc1, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<192x768xf32>) -> tensor<32x197x192xf32>
    %v3428 = stablehlo.reshape %v3427 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v3429 = stablehlo.reshape %v2006 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3430 = stablehlo.reshape %v3425 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3431 = stablehlo.dot_general %v3429, %v3430, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x768xf32>) -> tensor<192x768xf32>
    %v3432 = stablehlo.constant dense<0.003125> : tensor<192x768xf32>
    %v3433 = stablehlo.multiply %v3431, %v3432 : tensor<192x768xf32>
    %v3434 = stablehlo.subtract %b9_Wfc1, %v3433 : tensor<192x768xf32>
    %v3435 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3436 = stablehlo.reshape %v3425 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3437 = stablehlo.reduce(%v3436 init: %v3435) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<768xf32>
    %v3438 = stablehlo.constant dense<0.003125> : tensor<768xf32>
    %v3439 = stablehlo.multiply %v3437, %v3438 : tensor<768xf32>
    %v3440 = stablehlo.subtract %b9_bfc1, %v3439 : tensor<768xf32>
    %v3441 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3442 = stablehlo.reshape %v3428 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3443 = stablehlo.reduce(%v3442 init: %v3441) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v3444 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v3445 = stablehlo.multiply %v3443, %v3444 : tensor<192xf32>
    %v3446 = stablehlo.subtract %b9_bt2, %v3445 : tensor<192xf32>
    %v3447 = stablehlo.reshape %v1978 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3448 = stablehlo.reshape %v3428 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3449 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3450 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v3451 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v3452 = stablehlo.reduce(%v3447 init: %v3449) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3453 = stablehlo.broadcast_in_dim %v3452, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v3454 = stablehlo.divide %v3453, %v3450 : tensor<32x197x192xf32>
    %v3455 = stablehlo.subtract %v3447, %v3454 : tensor<32x197x192xf32>
    %v3456 = stablehlo.multiply %v3455, %v3455 : tensor<32x197x192xf32>
    %v3457 = stablehlo.reduce(%v3456 init: %v3449) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3458 = stablehlo.broadcast_in_dim %v3457, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v3459 = stablehlo.divide %v3458, %v3450 : tensor<32x197x192xf32>
    %v3460 = stablehlo.add %v3459, %v3451 : tensor<32x197x192xf32>
    %v3461 = stablehlo.rsqrt %v3460 : tensor<32x197x192xf32>
    %v3462 = stablehlo.multiply %v3455, %v3461 : tensor<32x197x192xf32>
    %v3463 = stablehlo.multiply %v3448, %v3462 : tensor<32x197x192xf32>
    %v3464 = stablehlo.reduce(%v3463 init: %v3449) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v3465 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v3466 = stablehlo.multiply %v3464, %v3465 : tensor<192xf32>
    %v3467 = stablehlo.subtract %b9_g2, %v3466 : tensor<192xf32>
    %v3468 = stablehlo.reshape %v3428 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3469 = stablehlo.broadcast_in_dim %b9_g2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v3470 = stablehlo.multiply %v3468, %v3469 : tensor<32x197x192xf32>
    %v3471 = stablehlo.reshape %v3470 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v3472 = stablehlo.reshape %v3471 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3473 = stablehlo.reshape %v1978 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3474 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3475 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v3476 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v3477 = stablehlo.reduce(%v3473 init: %v3474) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3478 = stablehlo.broadcast_in_dim %v3477, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v3479 = stablehlo.divide %v3478, %v3475 : tensor<32x197x192xf32>
    %v3480 = stablehlo.subtract %v3473, %v3479 : tensor<32x197x192xf32>
    %v3481 = stablehlo.multiply %v3480, %v3480 : tensor<32x197x192xf32>
    %v3482 = stablehlo.reduce(%v3481 init: %v3474) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3483 = stablehlo.broadcast_in_dim %v3482, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v3484 = stablehlo.divide %v3483, %v3475 : tensor<32x197x192xf32>
    %v3485 = stablehlo.add %v3484, %v3476 : tensor<32x197x192xf32>
    %v3486 = stablehlo.rsqrt %v3485 : tensor<32x197x192xf32>
    %v3487 = stablehlo.multiply %v3480, %v3486 : tensor<32x197x192xf32>
    %v3488 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v3489 = stablehlo.multiply %v3488, %v3472 : tensor<32x197x192xf32>
    %v3490 = stablehlo.reduce(%v3489 init: %v3474) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3491 = stablehlo.broadcast_in_dim %v3490, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v3492 = stablehlo.multiply %v3487, %v3489 : tensor<32x197x192xf32>
    %v3493 = stablehlo.reduce(%v3492 init: %v3474) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3494 = stablehlo.broadcast_in_dim %v3493, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v3495 = stablehlo.multiply %v3489, %v3475 : tensor<32x197x192xf32>
    %v3496 = stablehlo.subtract %v3495, %v3491 : tensor<32x197x192xf32>
    %v3497 = stablehlo.multiply %v3487, %v3494 : tensor<32x197x192xf32>
    %v3498 = stablehlo.subtract %v3496, %v3497 : tensor<32x197x192xf32>
    %v3499 = stablehlo.divide %v3486, %v3475 : tensor<32x197x192xf32>
    %v3500 = stablehlo.multiply %v3499, %v3498 : tensor<32x197x192xf32>
    %v3501 = stablehlo.reshape %v3500 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v3502 = stablehlo.add %v3387, %v3501 : tensor<32x37824xf32>
    %v3503 = stablehlo.reshape %v3502 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3504 = stablehlo.dot_general %v3503, %b9_Wo, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v3505 = stablehlo.reshape %v3504 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v3506 = stablehlo.reshape %v1972 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3507 = stablehlo.reshape %v3502 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3508 = stablehlo.dot_general %v3506, %v3507, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %v3509 = stablehlo.constant dense<0.003125> : tensor<192x192xf32>
    %v3510 = stablehlo.multiply %v3508, %v3509 : tensor<192x192xf32>
    %v3511 = stablehlo.subtract %b9_Wo, %v3510 : tensor<192x192xf32>
    %v3512 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3513 = stablehlo.reshape %v3502 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3514 = stablehlo.reduce(%v3513 init: %v3512) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v3515 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v3516 = stablehlo.multiply %v3514, %v3515 : tensor<192xf32>
    %v3517 = stablehlo.subtract %b9_bo, %v3516 : tensor<192xf32>
    %v3518 = stablehlo.reshape %v3505 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3519 = stablehlo.slice %v3518 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v3520 = stablehlo.reshape %v3519 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3521 = stablehlo.reshape %v1880 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3522 = stablehlo.transpose %v3521, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3523 = stablehlo.reshape %v3522 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3524 = stablehlo.reshape %v3520 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3525 = stablehlo.reshape %v3523 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3526 = stablehlo.dot_general %v3524, %v3525, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3527 = stablehlo.reshape %v3526 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3528 = stablehlo.reshape %v1896 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3529 = stablehlo.transpose %v3528, dims = [0, 2, 1] : (tensor<32x197x197xf32>) -> tensor<32x197x197xf32>
    %v3530 = stablehlo.reshape %v3529 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3531 = stablehlo.reshape %v3530 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3532 = stablehlo.reshape %v3520 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3533 = stablehlo.dot_general %v3531, %v3532, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3534 = stablehlo.reshape %v3533 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3535 = stablehlo.reshape %v1889 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3536 = stablehlo.reshape %v3527 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3537 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3538 = stablehlo.exponential %v3535 : tensor<32x197x197xf32>
    %v3539 = stablehlo.reduce(%v3538 init: %v3537) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3540 = stablehlo.broadcast_in_dim %v3539, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3541 = stablehlo.divide %v3538, %v3540 : tensor<32x197x197xf32>
    %v3542 = stablehlo.multiply %v3541, %v3536 : tensor<32x197x197xf32>
    %v3543 = stablehlo.reduce(%v3542 init: %v3537) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3544 = stablehlo.broadcast_in_dim %v3543, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3545 = stablehlo.subtract %v3536, %v3544 : tensor<32x197x197xf32>
    %v3546 = stablehlo.multiply %v3541, %v3545 : tensor<32x197x197xf32>
    %v3547 = stablehlo.reshape %v3546 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3548 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3549 = stablehlo.multiply %v3547, %v3548 : tensor<32x38809xf32>
    %v3550 = stablehlo.reshape %v3549 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3551 = stablehlo.reshape %v1877 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3552 = stablehlo.dot_general %v3550, %v3551, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3553 = stablehlo.reshape %v3552 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3554 = stablehlo.reshape %v1874 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3555 = stablehlo.transpose %v3554, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3556 = stablehlo.reshape %v3555 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3557 = stablehlo.reshape %v3556 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3558 = stablehlo.reshape %v3549 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3559 = stablehlo.dot_general %v3557, %v3558, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x64x197xf32>, tensor<32x197x197xf32>) -> tensor<32x64x197xf32>
    %v3560 = stablehlo.reshape %v3559 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3561 = stablehlo.reshape %v3560 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3562 = stablehlo.transpose %v3561, dims = [0, 2, 1] : (tensor<32x64x197xf32>) -> tensor<32x197x64xf32>
    %v3563 = stablehlo.reshape %v3562 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3564 = stablehlo.reshape %v3553 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3565 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3566 = stablehlo.pad %v3564, %v3565, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v3567 = stablehlo.reshape %v3566 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v3568 = stablehlo.reshape %v3563 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3569 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3570 = stablehlo.pad %v3568, %v3569, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v3571 = stablehlo.reshape %v3570 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v3572 = stablehlo.reshape %v3534 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3573 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3574 = stablehlo.pad %v3572, %v3573, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v3575 = stablehlo.reshape %v3574 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v3576 = stablehlo.reshape %v3505 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3577 = stablehlo.slice %v3576 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v3578 = stablehlo.reshape %v3577 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3579 = stablehlo.reshape %v1913 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3580 = stablehlo.transpose %v3579, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3581 = stablehlo.reshape %v3580 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3582 = stablehlo.reshape %v3578 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3583 = stablehlo.reshape %v3581 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3584 = stablehlo.dot_general %v3582, %v3583, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3585 = stablehlo.reshape %v3584 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3586 = stablehlo.reshape %v1929 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3587 = stablehlo.transpose %v3586, dims = [0, 2, 1] : (tensor<32x197x197xf32>) -> tensor<32x197x197xf32>
    %v3588 = stablehlo.reshape %v3587 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3589 = stablehlo.reshape %v3588 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3590 = stablehlo.reshape %v3578 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3591 = stablehlo.dot_general %v3589, %v3590, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3592 = stablehlo.reshape %v3591 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3593 = stablehlo.reshape %v1922 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3594 = stablehlo.reshape %v3585 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3595 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3596 = stablehlo.exponential %v3593 : tensor<32x197x197xf32>
    %v3597 = stablehlo.reduce(%v3596 init: %v3595) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3598 = stablehlo.broadcast_in_dim %v3597, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3599 = stablehlo.divide %v3596, %v3598 : tensor<32x197x197xf32>
    %v3600 = stablehlo.multiply %v3599, %v3594 : tensor<32x197x197xf32>
    %v3601 = stablehlo.reduce(%v3600 init: %v3595) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3602 = stablehlo.broadcast_in_dim %v3601, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3603 = stablehlo.subtract %v3594, %v3602 : tensor<32x197x197xf32>
    %v3604 = stablehlo.multiply %v3599, %v3603 : tensor<32x197x197xf32>
    %v3605 = stablehlo.reshape %v3604 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3606 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3607 = stablehlo.multiply %v3605, %v3606 : tensor<32x38809xf32>
    %v3608 = stablehlo.reshape %v3607 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3609 = stablehlo.reshape %v1910 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3610 = stablehlo.dot_general %v3608, %v3609, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3611 = stablehlo.reshape %v3610 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3612 = stablehlo.reshape %v1907 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3613 = stablehlo.transpose %v3612, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3614 = stablehlo.reshape %v3613 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3615 = stablehlo.reshape %v3614 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3616 = stablehlo.reshape %v3607 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3617 = stablehlo.dot_general %v3615, %v3616, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x64x197xf32>, tensor<32x197x197xf32>) -> tensor<32x64x197xf32>
    %v3618 = stablehlo.reshape %v3617 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3619 = stablehlo.reshape %v3618 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3620 = stablehlo.transpose %v3619, dims = [0, 2, 1] : (tensor<32x64x197xf32>) -> tensor<32x197x64xf32>
    %v3621 = stablehlo.reshape %v3620 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3622 = stablehlo.reshape %v3611 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3623 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3624 = stablehlo.pad %v3622, %v3623, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v3625 = stablehlo.reshape %v3624 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v3626 = stablehlo.reshape %v3621 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3627 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3628 = stablehlo.pad %v3626, %v3627, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v3629 = stablehlo.reshape %v3628 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v3630 = stablehlo.reshape %v3592 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3631 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3632 = stablehlo.pad %v3630, %v3631, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v3633 = stablehlo.reshape %v3632 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v3634 = stablehlo.add %v3567, %v3625 : tensor<32x37824xf32>
    %v3635 = stablehlo.add %v3571, %v3629 : tensor<32x37824xf32>
    %v3636 = stablehlo.add %v3575, %v3633 : tensor<32x37824xf32>
    %v3637 = stablehlo.reshape %v3505 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3638 = stablehlo.slice %v3637 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v3639 = stablehlo.reshape %v3638 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3640 = stablehlo.reshape %v1947 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3641 = stablehlo.transpose %v3640, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3642 = stablehlo.reshape %v3641 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3643 = stablehlo.reshape %v3639 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3644 = stablehlo.reshape %v3642 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3645 = stablehlo.dot_general %v3643, %v3644, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3646 = stablehlo.reshape %v3645 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3647 = stablehlo.reshape %v1963 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3648 = stablehlo.transpose %v3647, dims = [0, 2, 1] : (tensor<32x197x197xf32>) -> tensor<32x197x197xf32>
    %v3649 = stablehlo.reshape %v3648 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3650 = stablehlo.reshape %v3649 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3651 = stablehlo.reshape %v3639 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3652 = stablehlo.dot_general %v3650, %v3651, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3653 = stablehlo.reshape %v3652 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3654 = stablehlo.reshape %v1956 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3655 = stablehlo.reshape %v3646 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3656 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3657 = stablehlo.exponential %v3654 : tensor<32x197x197xf32>
    %v3658 = stablehlo.reduce(%v3657 init: %v3656) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3659 = stablehlo.broadcast_in_dim %v3658, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3660 = stablehlo.divide %v3657, %v3659 : tensor<32x197x197xf32>
    %v3661 = stablehlo.multiply %v3660, %v3655 : tensor<32x197x197xf32>
    %v3662 = stablehlo.reduce(%v3661 init: %v3656) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3663 = stablehlo.broadcast_in_dim %v3662, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3664 = stablehlo.subtract %v3655, %v3663 : tensor<32x197x197xf32>
    %v3665 = stablehlo.multiply %v3660, %v3664 : tensor<32x197x197xf32>
    %v3666 = stablehlo.reshape %v3665 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3667 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3668 = stablehlo.multiply %v3666, %v3667 : tensor<32x38809xf32>
    %v3669 = stablehlo.reshape %v3668 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3670 = stablehlo.reshape %v1944 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3671 = stablehlo.dot_general %v3669, %v3670, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3672 = stablehlo.reshape %v3671 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3673 = stablehlo.reshape %v1941 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3674 = stablehlo.transpose %v3673, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3675 = stablehlo.reshape %v3674 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3676 = stablehlo.reshape %v3675 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3677 = stablehlo.reshape %v3668 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3678 = stablehlo.dot_general %v3676, %v3677, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x64x197xf32>, tensor<32x197x197xf32>) -> tensor<32x64x197xf32>
    %v3679 = stablehlo.reshape %v3678 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3680 = stablehlo.reshape %v3679 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3681 = stablehlo.transpose %v3680, dims = [0, 2, 1] : (tensor<32x64x197xf32>) -> tensor<32x197x64xf32>
    %v3682 = stablehlo.reshape %v3681 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3683 = stablehlo.reshape %v3672 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3684 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3685 = stablehlo.pad %v3683, %v3684, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v3686 = stablehlo.reshape %v3685 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v3687 = stablehlo.reshape %v3682 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3688 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3689 = stablehlo.pad %v3687, %v3688, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v3690 = stablehlo.reshape %v3689 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v3691 = stablehlo.reshape %v3653 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3692 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3693 = stablehlo.pad %v3691, %v3692, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v3694 = stablehlo.reshape %v3693 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v3695 = stablehlo.add %v3634, %v3686 : tensor<32x37824xf32>
    %v3696 = stablehlo.add %v3635, %v3690 : tensor<32x37824xf32>
    %v3697 = stablehlo.add %v3636, %v3694 : tensor<32x37824xf32>
    %v3698 = stablehlo.reshape %v3695 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3699 = stablehlo.dot_general %v3698, %b9_Wq, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v3700 = stablehlo.reshape %v3699 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v3701 = stablehlo.reshape %v1856 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3702 = stablehlo.reshape %v3695 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3703 = stablehlo.dot_general %v3701, %v3702, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %v3704 = stablehlo.constant dense<0.003125> : tensor<192x192xf32>
    %v3705 = stablehlo.multiply %v3703, %v3704 : tensor<192x192xf32>
    %v3706 = stablehlo.subtract %b9_Wq, %v3705 : tensor<192x192xf32>
    %v3707 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3708 = stablehlo.reshape %v3695 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3709 = stablehlo.reduce(%v3708 init: %v3707) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v3710 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v3711 = stablehlo.multiply %v3709, %v3710 : tensor<192xf32>
    %v3712 = stablehlo.subtract %b9_bq, %v3711 : tensor<192xf32>
    %v3713 = stablehlo.reshape %v3696 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3714 = stablehlo.dot_general %v3713, %b9_Wk, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v3715 = stablehlo.reshape %v3714 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v3716 = stablehlo.reshape %v1856 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3717 = stablehlo.reshape %v3696 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3718 = stablehlo.dot_general %v3716, %v3717, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %v3719 = stablehlo.constant dense<0.003125> : tensor<192x192xf32>
    %v3720 = stablehlo.multiply %v3718, %v3719 : tensor<192x192xf32>
    %v3721 = stablehlo.subtract %b9_Wk, %v3720 : tensor<192x192xf32>
    %v3722 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3723 = stablehlo.reshape %v3696 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3724 = stablehlo.reduce(%v3723 init: %v3722) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v3725 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v3726 = stablehlo.multiply %v3724, %v3725 : tensor<192xf32>
    %v3727 = stablehlo.subtract %b9_bk, %v3726 : tensor<192xf32>
    %v3728 = stablehlo.reshape %v3697 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3729 = stablehlo.dot_general %v3728, %b9_Wv, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v3730 = stablehlo.reshape %v3729 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v3731 = stablehlo.reshape %v1856 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3732 = stablehlo.reshape %v3697 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3733 = stablehlo.dot_general %v3731, %v3732, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %v3734 = stablehlo.constant dense<0.003125> : tensor<192x192xf32>
    %v3735 = stablehlo.multiply %v3733, %v3734 : tensor<192x192xf32>
    %v3736 = stablehlo.subtract %b9_Wv, %v3735 : tensor<192x192xf32>
    %v3737 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3738 = stablehlo.reshape %v3697 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3739 = stablehlo.reduce(%v3738 init: %v3737) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v3740 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v3741 = stablehlo.multiply %v3739, %v3740 : tensor<192xf32>
    %v3742 = stablehlo.subtract %b9_bv, %v3741 : tensor<192xf32>
    %v3743 = stablehlo.add %v3700, %v3715 : tensor<32x37824xf32>
    %v3744 = stablehlo.add %v3743, %v3730 : tensor<32x37824xf32>
    %v3745 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3746 = stablehlo.reshape %v3744 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3747 = stablehlo.reduce(%v3746 init: %v3745) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v3748 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v3749 = stablehlo.multiply %v3747, %v3748 : tensor<192xf32>
    %v3750 = stablehlo.subtract %b9_bt1, %v3749 : tensor<192xf32>
    %v3751 = stablehlo.reshape %v1828 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3752 = stablehlo.reshape %v3744 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3753 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3754 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v3755 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v3756 = stablehlo.reduce(%v3751 init: %v3753) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3757 = stablehlo.broadcast_in_dim %v3756, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v3758 = stablehlo.divide %v3757, %v3754 : tensor<32x197x192xf32>
    %v3759 = stablehlo.subtract %v3751, %v3758 : tensor<32x197x192xf32>
    %v3760 = stablehlo.multiply %v3759, %v3759 : tensor<32x197x192xf32>
    %v3761 = stablehlo.reduce(%v3760 init: %v3753) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3762 = stablehlo.broadcast_in_dim %v3761, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v3763 = stablehlo.divide %v3762, %v3754 : tensor<32x197x192xf32>
    %v3764 = stablehlo.add %v3763, %v3755 : tensor<32x197x192xf32>
    %v3765 = stablehlo.rsqrt %v3764 : tensor<32x197x192xf32>
    %v3766 = stablehlo.multiply %v3759, %v3765 : tensor<32x197x192xf32>
    %v3767 = stablehlo.multiply %v3752, %v3766 : tensor<32x197x192xf32>
    %v3768 = stablehlo.reduce(%v3767 init: %v3753) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v3769 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v3770 = stablehlo.multiply %v3768, %v3769 : tensor<192xf32>
    %v3771 = stablehlo.subtract %b9_g1, %v3770 : tensor<192xf32>
    %v3772 = stablehlo.reshape %v3744 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3773 = stablehlo.broadcast_in_dim %b9_g1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v3774 = stablehlo.multiply %v3772, %v3773 : tensor<32x197x192xf32>
    %v3775 = stablehlo.reshape %v3774 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v3776 = stablehlo.reshape %v3775 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3777 = stablehlo.reshape %v1828 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3778 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3779 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v3780 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v3781 = stablehlo.reduce(%v3777 init: %v3778) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3782 = stablehlo.broadcast_in_dim %v3781, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v3783 = stablehlo.divide %v3782, %v3779 : tensor<32x197x192xf32>
    %v3784 = stablehlo.subtract %v3777, %v3783 : tensor<32x197x192xf32>
    %v3785 = stablehlo.multiply %v3784, %v3784 : tensor<32x197x192xf32>
    %v3786 = stablehlo.reduce(%v3785 init: %v3778) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3787 = stablehlo.broadcast_in_dim %v3786, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v3788 = stablehlo.divide %v3787, %v3779 : tensor<32x197x192xf32>
    %v3789 = stablehlo.add %v3788, %v3780 : tensor<32x197x192xf32>
    %v3790 = stablehlo.rsqrt %v3789 : tensor<32x197x192xf32>
    %v3791 = stablehlo.multiply %v3784, %v3790 : tensor<32x197x192xf32>
    %v3792 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v3793 = stablehlo.multiply %v3792, %v3776 : tensor<32x197x192xf32>
    %v3794 = stablehlo.reduce(%v3793 init: %v3778) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3795 = stablehlo.broadcast_in_dim %v3794, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v3796 = stablehlo.multiply %v3791, %v3793 : tensor<32x197x192xf32>
    %v3797 = stablehlo.reduce(%v3796 init: %v3778) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3798 = stablehlo.broadcast_in_dim %v3797, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v3799 = stablehlo.multiply %v3793, %v3779 : tensor<32x197x192xf32>
    %v3800 = stablehlo.subtract %v3799, %v3795 : tensor<32x197x192xf32>
    %v3801 = stablehlo.multiply %v3791, %v3798 : tensor<32x197x192xf32>
    %v3802 = stablehlo.subtract %v3800, %v3801 : tensor<32x197x192xf32>
    %v3803 = stablehlo.divide %v3790, %v3779 : tensor<32x197x192xf32>
    %v3804 = stablehlo.multiply %v3803, %v3802 : tensor<32x197x192xf32>
    %v3805 = stablehlo.reshape %v3804 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v3806 = stablehlo.add %v3502, %v3805 : tensor<32x37824xf32>
    %v3807 = stablehlo.reshape %v3806 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3808 = stablehlo.dot_general %v3807, %b8_Wfc2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<768x192xf32>) -> tensor<32x197x768xf32>
    %v3809 = stablehlo.reshape %v3808 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3810 = stablehlo.reshape %v1822 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3811 = stablehlo.reshape %v3806 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3812 = stablehlo.dot_general %v3810, %v3811, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<32x197x192xf32>) -> tensor<768x192xf32>
    %v3813 = stablehlo.constant dense<0.003125> : tensor<768x192xf32>
    %v3814 = stablehlo.multiply %v3812, %v3813 : tensor<768x192xf32>
    %v3815 = stablehlo.subtract %b8_Wfc2, %v3814 : tensor<768x192xf32>
    %v3816 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3817 = stablehlo.reshape %v3806 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3818 = stablehlo.reduce(%v3817 init: %v3816) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v3819 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v3820 = stablehlo.multiply %v3818, %v3819 : tensor<192xf32>
    %v3821 = stablehlo.subtract %b8_bfc2, %v3820 : tensor<192xf32>
    %v3822 = stablehlo.multiply %v1809, %v1809 : tensor<32x151296xf32>
    %v3823 = stablehlo.multiply %v3822, %v1809 : tensor<32x151296xf32>
    %v3824 = stablehlo.constant dense<0.044715> : tensor<32x151296xf32>
    %v3825 = stablehlo.multiply %v3824, %v3823 : tensor<32x151296xf32>
    %v3826 = stablehlo.add %v1809, %v3825 : tensor<32x151296xf32>
    %v3827 = stablehlo.constant dense<0.7978845608028654> : tensor<32x151296xf32>
    %v3828 = stablehlo.multiply %v3827, %v3826 : tensor<32x151296xf32>
    %v3829 = stablehlo.tanh %v3828 : tensor<32x151296xf32>
    %v3830 = stablehlo.constant dense<1.0> : tensor<32x151296xf32>
    %v3831 = stablehlo.add %v3830, %v3829 : tensor<32x151296xf32>
    %v3832 = stablehlo.constant dense<0.5> : tensor<32x151296xf32>
    %v3833 = stablehlo.multiply %v3832, %v3831 : tensor<32x151296xf32>
    %v3834 = stablehlo.multiply %v3829, %v3829 : tensor<32x151296xf32>
    %v3835 = stablehlo.subtract %v3830, %v3834 : tensor<32x151296xf32>
    %v3836 = stablehlo.multiply %v3832, %v1809 : tensor<32x151296xf32>
    %v3837 = stablehlo.multiply %v3836, %v3835 : tensor<32x151296xf32>
    %v3838 = stablehlo.constant dense<0.134145> : tensor<32x151296xf32>
    %v3839 = stablehlo.multiply %v3838, %v3822 : tensor<32x151296xf32>
    %v3840 = stablehlo.add %v3830, %v3839 : tensor<32x151296xf32>
    %v3841 = stablehlo.multiply %v3827, %v3840 : tensor<32x151296xf32>
    %v3842 = stablehlo.multiply %v3837, %v3841 : tensor<32x151296xf32>
    %v3843 = stablehlo.add %v3833, %v3842 : tensor<32x151296xf32>
    %v3844 = stablehlo.multiply %v3809, %v3843 : tensor<32x151296xf32>
    %v3845 = stablehlo.reshape %v3844 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3846 = stablehlo.dot_general %v3845, %b8_Wfc1, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<192x768xf32>) -> tensor<32x197x192xf32>
    %v3847 = stablehlo.reshape %v3846 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v3848 = stablehlo.reshape %v1804 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3849 = stablehlo.reshape %v3844 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3850 = stablehlo.dot_general %v3848, %v3849, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x768xf32>) -> tensor<192x768xf32>
    %v3851 = stablehlo.constant dense<0.003125> : tensor<192x768xf32>
    %v3852 = stablehlo.multiply %v3850, %v3851 : tensor<192x768xf32>
    %v3853 = stablehlo.subtract %b8_Wfc1, %v3852 : tensor<192x768xf32>
    %v3854 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3855 = stablehlo.reshape %v3844 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3856 = stablehlo.reduce(%v3855 init: %v3854) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<768xf32>
    %v3857 = stablehlo.constant dense<0.003125> : tensor<768xf32>
    %v3858 = stablehlo.multiply %v3856, %v3857 : tensor<768xf32>
    %v3859 = stablehlo.subtract %b8_bfc1, %v3858 : tensor<768xf32>
    %v3860 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3861 = stablehlo.reshape %v3847 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3862 = stablehlo.reduce(%v3861 init: %v3860) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v3863 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v3864 = stablehlo.multiply %v3862, %v3863 : tensor<192xf32>
    %v3865 = stablehlo.subtract %b8_bt2, %v3864 : tensor<192xf32>
    %v3866 = stablehlo.reshape %v1776 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3867 = stablehlo.reshape %v3847 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3868 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3869 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v3870 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v3871 = stablehlo.reduce(%v3866 init: %v3868) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3872 = stablehlo.broadcast_in_dim %v3871, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v3873 = stablehlo.divide %v3872, %v3869 : tensor<32x197x192xf32>
    %v3874 = stablehlo.subtract %v3866, %v3873 : tensor<32x197x192xf32>
    %v3875 = stablehlo.multiply %v3874, %v3874 : tensor<32x197x192xf32>
    %v3876 = stablehlo.reduce(%v3875 init: %v3868) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3877 = stablehlo.broadcast_in_dim %v3876, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v3878 = stablehlo.divide %v3877, %v3869 : tensor<32x197x192xf32>
    %v3879 = stablehlo.add %v3878, %v3870 : tensor<32x197x192xf32>
    %v3880 = stablehlo.rsqrt %v3879 : tensor<32x197x192xf32>
    %v3881 = stablehlo.multiply %v3874, %v3880 : tensor<32x197x192xf32>
    %v3882 = stablehlo.multiply %v3867, %v3881 : tensor<32x197x192xf32>
    %v3883 = stablehlo.reduce(%v3882 init: %v3868) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v3884 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v3885 = stablehlo.multiply %v3883, %v3884 : tensor<192xf32>
    %v3886 = stablehlo.subtract %b8_g2, %v3885 : tensor<192xf32>
    %v3887 = stablehlo.reshape %v3847 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3888 = stablehlo.broadcast_in_dim %b8_g2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v3889 = stablehlo.multiply %v3887, %v3888 : tensor<32x197x192xf32>
    %v3890 = stablehlo.reshape %v3889 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v3891 = stablehlo.reshape %v3890 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3892 = stablehlo.reshape %v1776 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3893 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3894 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v3895 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v3896 = stablehlo.reduce(%v3892 init: %v3893) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3897 = stablehlo.broadcast_in_dim %v3896, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v3898 = stablehlo.divide %v3897, %v3894 : tensor<32x197x192xf32>
    %v3899 = stablehlo.subtract %v3892, %v3898 : tensor<32x197x192xf32>
    %v3900 = stablehlo.multiply %v3899, %v3899 : tensor<32x197x192xf32>
    %v3901 = stablehlo.reduce(%v3900 init: %v3893) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3902 = stablehlo.broadcast_in_dim %v3901, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v3903 = stablehlo.divide %v3902, %v3894 : tensor<32x197x192xf32>
    %v3904 = stablehlo.add %v3903, %v3895 : tensor<32x197x192xf32>
    %v3905 = stablehlo.rsqrt %v3904 : tensor<32x197x192xf32>
    %v3906 = stablehlo.multiply %v3899, %v3905 : tensor<32x197x192xf32>
    %v3907 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v3908 = stablehlo.multiply %v3907, %v3891 : tensor<32x197x192xf32>
    %v3909 = stablehlo.reduce(%v3908 init: %v3893) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3910 = stablehlo.broadcast_in_dim %v3909, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v3911 = stablehlo.multiply %v3906, %v3908 : tensor<32x197x192xf32>
    %v3912 = stablehlo.reduce(%v3911 init: %v3893) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3913 = stablehlo.broadcast_in_dim %v3912, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v3914 = stablehlo.multiply %v3908, %v3894 : tensor<32x197x192xf32>
    %v3915 = stablehlo.subtract %v3914, %v3910 : tensor<32x197x192xf32>
    %v3916 = stablehlo.multiply %v3906, %v3913 : tensor<32x197x192xf32>
    %v3917 = stablehlo.subtract %v3915, %v3916 : tensor<32x197x192xf32>
    %v3918 = stablehlo.divide %v3905, %v3894 : tensor<32x197x192xf32>
    %v3919 = stablehlo.multiply %v3918, %v3917 : tensor<32x197x192xf32>
    %v3920 = stablehlo.reshape %v3919 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v3921 = stablehlo.add %v3806, %v3920 : tensor<32x37824xf32>
    %v3922 = stablehlo.reshape %v3921 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3923 = stablehlo.dot_general %v3922, %b8_Wo, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v3924 = stablehlo.reshape %v3923 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v3925 = stablehlo.reshape %v1770 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3926 = stablehlo.reshape %v3921 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3927 = stablehlo.dot_general %v3925, %v3926, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %v3928 = stablehlo.constant dense<0.003125> : tensor<192x192xf32>
    %v3929 = stablehlo.multiply %v3927, %v3928 : tensor<192x192xf32>
    %v3930 = stablehlo.subtract %b8_Wo, %v3929 : tensor<192x192xf32>
    %v3931 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3932 = stablehlo.reshape %v3921 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3933 = stablehlo.reduce(%v3932 init: %v3931) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v3934 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v3935 = stablehlo.multiply %v3933, %v3934 : tensor<192xf32>
    %v3936 = stablehlo.subtract %b8_bo, %v3935 : tensor<192xf32>
    %v3937 = stablehlo.reshape %v3924 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3938 = stablehlo.slice %v3937 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v3939 = stablehlo.reshape %v3938 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3940 = stablehlo.reshape %v1678 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3941 = stablehlo.transpose %v3940, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3942 = stablehlo.reshape %v3941 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3943 = stablehlo.reshape %v3939 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3944 = stablehlo.reshape %v3942 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3945 = stablehlo.dot_general %v3943, %v3944, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3946 = stablehlo.reshape %v3945 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3947 = stablehlo.reshape %v1694 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3948 = stablehlo.transpose %v3947, dims = [0, 2, 1] : (tensor<32x197x197xf32>) -> tensor<32x197x197xf32>
    %v3949 = stablehlo.reshape %v3948 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3950 = stablehlo.reshape %v3949 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3951 = stablehlo.reshape %v3939 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3952 = stablehlo.dot_general %v3950, %v3951, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3953 = stablehlo.reshape %v3952 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3954 = stablehlo.reshape %v1687 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3955 = stablehlo.reshape %v3946 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3956 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3957 = stablehlo.exponential %v3954 : tensor<32x197x197xf32>
    %v3958 = stablehlo.reduce(%v3957 init: %v3956) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3959 = stablehlo.broadcast_in_dim %v3958, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3960 = stablehlo.divide %v3957, %v3959 : tensor<32x197x197xf32>
    %v3961 = stablehlo.multiply %v3960, %v3955 : tensor<32x197x197xf32>
    %v3962 = stablehlo.reduce(%v3961 init: %v3956) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3963 = stablehlo.broadcast_in_dim %v3962, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3964 = stablehlo.subtract %v3955, %v3963 : tensor<32x197x197xf32>
    %v3965 = stablehlo.multiply %v3960, %v3964 : tensor<32x197x197xf32>
    %v3966 = stablehlo.reshape %v3965 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3967 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3968 = stablehlo.multiply %v3966, %v3967 : tensor<32x38809xf32>
    %v3969 = stablehlo.reshape %v3968 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3970 = stablehlo.reshape %v1675 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3971 = stablehlo.dot_general %v3969, %v3970, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3972 = stablehlo.reshape %v3971 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3973 = stablehlo.reshape %v1672 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3974 = stablehlo.transpose %v3973, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3975 = stablehlo.reshape %v3974 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3976 = stablehlo.reshape %v3975 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3977 = stablehlo.reshape %v3968 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3978 = stablehlo.dot_general %v3976, %v3977, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x64x197xf32>, tensor<32x197x197xf32>) -> tensor<32x64x197xf32>
    %v3979 = stablehlo.reshape %v3978 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3980 = stablehlo.reshape %v3979 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3981 = stablehlo.transpose %v3980, dims = [0, 2, 1] : (tensor<32x64x197xf32>) -> tensor<32x197x64xf32>
    %v3982 = stablehlo.reshape %v3981 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3983 = stablehlo.reshape %v3972 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3984 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3985 = stablehlo.pad %v3983, %v3984, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v3986 = stablehlo.reshape %v3985 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v3987 = stablehlo.reshape %v3982 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3988 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3989 = stablehlo.pad %v3987, %v3988, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v3990 = stablehlo.reshape %v3989 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v3991 = stablehlo.reshape %v3953 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3992 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3993 = stablehlo.pad %v3991, %v3992, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v3994 = stablehlo.reshape %v3993 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v3995 = stablehlo.reshape %v3924 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v3996 = stablehlo.slice %v3995 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v3997 = stablehlo.reshape %v3996 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3998 = stablehlo.reshape %v1711 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3999 = stablehlo.transpose %v3998, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v4000 = stablehlo.reshape %v3999 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4001 = stablehlo.reshape %v3997 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4002 = stablehlo.reshape %v4000 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4003 = stablehlo.dot_general %v4001, %v4002, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v4004 = stablehlo.reshape %v4003 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4005 = stablehlo.reshape %v1727 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4006 = stablehlo.transpose %v4005, dims = [0, 2, 1] : (tensor<32x197x197xf32>) -> tensor<32x197x197xf32>
    %v4007 = stablehlo.reshape %v4006 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4008 = stablehlo.reshape %v4007 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4009 = stablehlo.reshape %v3997 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4010 = stablehlo.dot_general %v4008, %v4009, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4011 = stablehlo.reshape %v4010 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4012 = stablehlo.reshape %v1720 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4013 = stablehlo.reshape %v4004 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4014 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4015 = stablehlo.exponential %v4012 : tensor<32x197x197xf32>
    %v4016 = stablehlo.reduce(%v4015 init: %v4014) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4017 = stablehlo.broadcast_in_dim %v4016, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4018 = stablehlo.divide %v4015, %v4017 : tensor<32x197x197xf32>
    %v4019 = stablehlo.multiply %v4018, %v4013 : tensor<32x197x197xf32>
    %v4020 = stablehlo.reduce(%v4019 init: %v4014) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4021 = stablehlo.broadcast_in_dim %v4020, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4022 = stablehlo.subtract %v4013, %v4021 : tensor<32x197x197xf32>
    %v4023 = stablehlo.multiply %v4018, %v4022 : tensor<32x197x197xf32>
    %v4024 = stablehlo.reshape %v4023 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4025 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v4026 = stablehlo.multiply %v4024, %v4025 : tensor<32x38809xf32>
    %v4027 = stablehlo.reshape %v4026 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4028 = stablehlo.reshape %v1708 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4029 = stablehlo.dot_general %v4027, %v4028, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4030 = stablehlo.reshape %v4029 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4031 = stablehlo.reshape %v1705 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4032 = stablehlo.transpose %v4031, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v4033 = stablehlo.reshape %v4032 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4034 = stablehlo.reshape %v4033 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4035 = stablehlo.reshape %v4026 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4036 = stablehlo.dot_general %v4034, %v4035, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x64x197xf32>, tensor<32x197x197xf32>) -> tensor<32x64x197xf32>
    %v4037 = stablehlo.reshape %v4036 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4038 = stablehlo.reshape %v4037 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4039 = stablehlo.transpose %v4038, dims = [0, 2, 1] : (tensor<32x64x197xf32>) -> tensor<32x197x64xf32>
    %v4040 = stablehlo.reshape %v4039 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4041 = stablehlo.reshape %v4030 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4042 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4043 = stablehlo.pad %v4041, %v4042, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v4044 = stablehlo.reshape %v4043 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v4045 = stablehlo.reshape %v4040 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4046 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4047 = stablehlo.pad %v4045, %v4046, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v4048 = stablehlo.reshape %v4047 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v4049 = stablehlo.reshape %v4011 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4050 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4051 = stablehlo.pad %v4049, %v4050, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v4052 = stablehlo.reshape %v4051 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v4053 = stablehlo.add %v3986, %v4044 : tensor<32x37824xf32>
    %v4054 = stablehlo.add %v3990, %v4048 : tensor<32x37824xf32>
    %v4055 = stablehlo.add %v3994, %v4052 : tensor<32x37824xf32>
    %v4056 = stablehlo.reshape %v3924 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4057 = stablehlo.slice %v4056 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v4058 = stablehlo.reshape %v4057 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4059 = stablehlo.reshape %v1745 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4060 = stablehlo.transpose %v4059, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v4061 = stablehlo.reshape %v4060 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4062 = stablehlo.reshape %v4058 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4063 = stablehlo.reshape %v4061 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4064 = stablehlo.dot_general %v4062, %v4063, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v4065 = stablehlo.reshape %v4064 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4066 = stablehlo.reshape %v1761 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4067 = stablehlo.transpose %v4066, dims = [0, 2, 1] : (tensor<32x197x197xf32>) -> tensor<32x197x197xf32>
    %v4068 = stablehlo.reshape %v4067 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4069 = stablehlo.reshape %v4068 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4070 = stablehlo.reshape %v4058 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4071 = stablehlo.dot_general %v4069, %v4070, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4072 = stablehlo.reshape %v4071 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4073 = stablehlo.reshape %v1754 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4074 = stablehlo.reshape %v4065 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4075 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4076 = stablehlo.exponential %v4073 : tensor<32x197x197xf32>
    %v4077 = stablehlo.reduce(%v4076 init: %v4075) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4078 = stablehlo.broadcast_in_dim %v4077, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4079 = stablehlo.divide %v4076, %v4078 : tensor<32x197x197xf32>
    %v4080 = stablehlo.multiply %v4079, %v4074 : tensor<32x197x197xf32>
    %v4081 = stablehlo.reduce(%v4080 init: %v4075) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4082 = stablehlo.broadcast_in_dim %v4081, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4083 = stablehlo.subtract %v4074, %v4082 : tensor<32x197x197xf32>
    %v4084 = stablehlo.multiply %v4079, %v4083 : tensor<32x197x197xf32>
    %v4085 = stablehlo.reshape %v4084 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4086 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v4087 = stablehlo.multiply %v4085, %v4086 : tensor<32x38809xf32>
    %v4088 = stablehlo.reshape %v4087 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4089 = stablehlo.reshape %v1742 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4090 = stablehlo.dot_general %v4088, %v4089, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4091 = stablehlo.reshape %v4090 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4092 = stablehlo.reshape %v1739 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4093 = stablehlo.transpose %v4092, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v4094 = stablehlo.reshape %v4093 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4095 = stablehlo.reshape %v4094 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4096 = stablehlo.reshape %v4087 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4097 = stablehlo.dot_general %v4095, %v4096, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x64x197xf32>, tensor<32x197x197xf32>) -> tensor<32x64x197xf32>
    %v4098 = stablehlo.reshape %v4097 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4099 = stablehlo.reshape %v4098 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4100 = stablehlo.transpose %v4099, dims = [0, 2, 1] : (tensor<32x64x197xf32>) -> tensor<32x197x64xf32>
    %v4101 = stablehlo.reshape %v4100 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4102 = stablehlo.reshape %v4091 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4103 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4104 = stablehlo.pad %v4102, %v4103, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v4105 = stablehlo.reshape %v4104 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v4106 = stablehlo.reshape %v4101 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4107 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4108 = stablehlo.pad %v4106, %v4107, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v4109 = stablehlo.reshape %v4108 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v4110 = stablehlo.reshape %v4072 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4111 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4112 = stablehlo.pad %v4110, %v4111, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v4113 = stablehlo.reshape %v4112 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v4114 = stablehlo.add %v4053, %v4105 : tensor<32x37824xf32>
    %v4115 = stablehlo.add %v4054, %v4109 : tensor<32x37824xf32>
    %v4116 = stablehlo.add %v4055, %v4113 : tensor<32x37824xf32>
    %v4117 = stablehlo.reshape %v4114 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4118 = stablehlo.dot_general %v4117, %b8_Wq, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v4119 = stablehlo.reshape %v4118 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v4120 = stablehlo.reshape %v1654 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4121 = stablehlo.reshape %v4114 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4122 = stablehlo.dot_general %v4120, %v4121, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %v4123 = stablehlo.constant dense<0.003125> : tensor<192x192xf32>
    %v4124 = stablehlo.multiply %v4122, %v4123 : tensor<192x192xf32>
    %v4125 = stablehlo.subtract %b8_Wq, %v4124 : tensor<192x192xf32>
    %v4126 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4127 = stablehlo.reshape %v4114 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4128 = stablehlo.reduce(%v4127 init: %v4126) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v4129 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v4130 = stablehlo.multiply %v4128, %v4129 : tensor<192xf32>
    %v4131 = stablehlo.subtract %b8_bq, %v4130 : tensor<192xf32>
    %v4132 = stablehlo.reshape %v4115 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4133 = stablehlo.dot_general %v4132, %b8_Wk, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v4134 = stablehlo.reshape %v4133 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v4135 = stablehlo.reshape %v1654 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4136 = stablehlo.reshape %v4115 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4137 = stablehlo.dot_general %v4135, %v4136, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %v4138 = stablehlo.constant dense<0.003125> : tensor<192x192xf32>
    %v4139 = stablehlo.multiply %v4137, %v4138 : tensor<192x192xf32>
    %v4140 = stablehlo.subtract %b8_Wk, %v4139 : tensor<192x192xf32>
    %v4141 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4142 = stablehlo.reshape %v4115 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4143 = stablehlo.reduce(%v4142 init: %v4141) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v4144 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v4145 = stablehlo.multiply %v4143, %v4144 : tensor<192xf32>
    %v4146 = stablehlo.subtract %b8_bk, %v4145 : tensor<192xf32>
    %v4147 = stablehlo.reshape %v4116 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4148 = stablehlo.dot_general %v4147, %b8_Wv, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v4149 = stablehlo.reshape %v4148 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v4150 = stablehlo.reshape %v1654 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4151 = stablehlo.reshape %v4116 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4152 = stablehlo.dot_general %v4150, %v4151, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %v4153 = stablehlo.constant dense<0.003125> : tensor<192x192xf32>
    %v4154 = stablehlo.multiply %v4152, %v4153 : tensor<192x192xf32>
    %v4155 = stablehlo.subtract %b8_Wv, %v4154 : tensor<192x192xf32>
    %v4156 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4157 = stablehlo.reshape %v4116 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4158 = stablehlo.reduce(%v4157 init: %v4156) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v4159 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v4160 = stablehlo.multiply %v4158, %v4159 : tensor<192xf32>
    %v4161 = stablehlo.subtract %b8_bv, %v4160 : tensor<192xf32>
    %v4162 = stablehlo.add %v4119, %v4134 : tensor<32x37824xf32>
    %v4163 = stablehlo.add %v4162, %v4149 : tensor<32x37824xf32>
    %v4164 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4165 = stablehlo.reshape %v4163 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4166 = stablehlo.reduce(%v4165 init: %v4164) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v4167 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v4168 = stablehlo.multiply %v4166, %v4167 : tensor<192xf32>
    %v4169 = stablehlo.subtract %b8_bt1, %v4168 : tensor<192xf32>
    %v4170 = stablehlo.reshape %v1626 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4171 = stablehlo.reshape %v4163 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4172 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4173 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v4174 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v4175 = stablehlo.reduce(%v4170 init: %v4172) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4176 = stablehlo.broadcast_in_dim %v4175, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v4177 = stablehlo.divide %v4176, %v4173 : tensor<32x197x192xf32>
    %v4178 = stablehlo.subtract %v4170, %v4177 : tensor<32x197x192xf32>
    %v4179 = stablehlo.multiply %v4178, %v4178 : tensor<32x197x192xf32>
    %v4180 = stablehlo.reduce(%v4179 init: %v4172) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4181 = stablehlo.broadcast_in_dim %v4180, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v4182 = stablehlo.divide %v4181, %v4173 : tensor<32x197x192xf32>
    %v4183 = stablehlo.add %v4182, %v4174 : tensor<32x197x192xf32>
    %v4184 = stablehlo.rsqrt %v4183 : tensor<32x197x192xf32>
    %v4185 = stablehlo.multiply %v4178, %v4184 : tensor<32x197x192xf32>
    %v4186 = stablehlo.multiply %v4171, %v4185 : tensor<32x197x192xf32>
    %v4187 = stablehlo.reduce(%v4186 init: %v4172) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v4188 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v4189 = stablehlo.multiply %v4187, %v4188 : tensor<192xf32>
    %v4190 = stablehlo.subtract %b8_g1, %v4189 : tensor<192xf32>
    %v4191 = stablehlo.reshape %v4163 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4192 = stablehlo.broadcast_in_dim %b8_g1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v4193 = stablehlo.multiply %v4191, %v4192 : tensor<32x197x192xf32>
    %v4194 = stablehlo.reshape %v4193 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v4195 = stablehlo.reshape %v4194 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4196 = stablehlo.reshape %v1626 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4197 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4198 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v4199 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v4200 = stablehlo.reduce(%v4196 init: %v4197) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4201 = stablehlo.broadcast_in_dim %v4200, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v4202 = stablehlo.divide %v4201, %v4198 : tensor<32x197x192xf32>
    %v4203 = stablehlo.subtract %v4196, %v4202 : tensor<32x197x192xf32>
    %v4204 = stablehlo.multiply %v4203, %v4203 : tensor<32x197x192xf32>
    %v4205 = stablehlo.reduce(%v4204 init: %v4197) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4206 = stablehlo.broadcast_in_dim %v4205, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v4207 = stablehlo.divide %v4206, %v4198 : tensor<32x197x192xf32>
    %v4208 = stablehlo.add %v4207, %v4199 : tensor<32x197x192xf32>
    %v4209 = stablehlo.rsqrt %v4208 : tensor<32x197x192xf32>
    %v4210 = stablehlo.multiply %v4203, %v4209 : tensor<32x197x192xf32>
    %v4211 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v4212 = stablehlo.multiply %v4211, %v4195 : tensor<32x197x192xf32>
    %v4213 = stablehlo.reduce(%v4212 init: %v4197) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4214 = stablehlo.broadcast_in_dim %v4213, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v4215 = stablehlo.multiply %v4210, %v4212 : tensor<32x197x192xf32>
    %v4216 = stablehlo.reduce(%v4215 init: %v4197) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4217 = stablehlo.broadcast_in_dim %v4216, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v4218 = stablehlo.multiply %v4212, %v4198 : tensor<32x197x192xf32>
    %v4219 = stablehlo.subtract %v4218, %v4214 : tensor<32x197x192xf32>
    %v4220 = stablehlo.multiply %v4210, %v4217 : tensor<32x197x192xf32>
    %v4221 = stablehlo.subtract %v4219, %v4220 : tensor<32x197x192xf32>
    %v4222 = stablehlo.divide %v4209, %v4198 : tensor<32x197x192xf32>
    %v4223 = stablehlo.multiply %v4222, %v4221 : tensor<32x197x192xf32>
    %v4224 = stablehlo.reshape %v4223 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v4225 = stablehlo.add %v3921, %v4224 : tensor<32x37824xf32>
    %v4226 = stablehlo.reshape %v4225 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4227 = stablehlo.dot_general %v4226, %b7_Wfc2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<768x192xf32>) -> tensor<32x197x768xf32>
    %v4228 = stablehlo.reshape %v4227 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4229 = stablehlo.reshape %v1620 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4230 = stablehlo.reshape %v4225 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4231 = stablehlo.dot_general %v4229, %v4230, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<32x197x192xf32>) -> tensor<768x192xf32>
    %v4232 = stablehlo.constant dense<0.003125> : tensor<768x192xf32>
    %v4233 = stablehlo.multiply %v4231, %v4232 : tensor<768x192xf32>
    %v4234 = stablehlo.subtract %b7_Wfc2, %v4233 : tensor<768x192xf32>
    %v4235 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4236 = stablehlo.reshape %v4225 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4237 = stablehlo.reduce(%v4236 init: %v4235) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v4238 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v4239 = stablehlo.multiply %v4237, %v4238 : tensor<192xf32>
    %v4240 = stablehlo.subtract %b7_bfc2, %v4239 : tensor<192xf32>
    %v4241 = stablehlo.multiply %v1607, %v1607 : tensor<32x151296xf32>
    %v4242 = stablehlo.multiply %v4241, %v1607 : tensor<32x151296xf32>
    %v4243 = stablehlo.constant dense<0.044715> : tensor<32x151296xf32>
    %v4244 = stablehlo.multiply %v4243, %v4242 : tensor<32x151296xf32>
    %v4245 = stablehlo.add %v1607, %v4244 : tensor<32x151296xf32>
    %v4246 = stablehlo.constant dense<0.7978845608028654> : tensor<32x151296xf32>
    %v4247 = stablehlo.multiply %v4246, %v4245 : tensor<32x151296xf32>
    %v4248 = stablehlo.tanh %v4247 : tensor<32x151296xf32>
    %v4249 = stablehlo.constant dense<1.0> : tensor<32x151296xf32>
    %v4250 = stablehlo.add %v4249, %v4248 : tensor<32x151296xf32>
    %v4251 = stablehlo.constant dense<0.5> : tensor<32x151296xf32>
    %v4252 = stablehlo.multiply %v4251, %v4250 : tensor<32x151296xf32>
    %v4253 = stablehlo.multiply %v4248, %v4248 : tensor<32x151296xf32>
    %v4254 = stablehlo.subtract %v4249, %v4253 : tensor<32x151296xf32>
    %v4255 = stablehlo.multiply %v4251, %v1607 : tensor<32x151296xf32>
    %v4256 = stablehlo.multiply %v4255, %v4254 : tensor<32x151296xf32>
    %v4257 = stablehlo.constant dense<0.134145> : tensor<32x151296xf32>
    %v4258 = stablehlo.multiply %v4257, %v4241 : tensor<32x151296xf32>
    %v4259 = stablehlo.add %v4249, %v4258 : tensor<32x151296xf32>
    %v4260 = stablehlo.multiply %v4246, %v4259 : tensor<32x151296xf32>
    %v4261 = stablehlo.multiply %v4256, %v4260 : tensor<32x151296xf32>
    %v4262 = stablehlo.add %v4252, %v4261 : tensor<32x151296xf32>
    %v4263 = stablehlo.multiply %v4228, %v4262 : tensor<32x151296xf32>
    %v4264 = stablehlo.reshape %v4263 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4265 = stablehlo.dot_general %v4264, %b7_Wfc1, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<192x768xf32>) -> tensor<32x197x192xf32>
    %v4266 = stablehlo.reshape %v4265 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v4267 = stablehlo.reshape %v1602 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4268 = stablehlo.reshape %v4263 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4269 = stablehlo.dot_general %v4267, %v4268, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x768xf32>) -> tensor<192x768xf32>
    %v4270 = stablehlo.constant dense<0.003125> : tensor<192x768xf32>
    %v4271 = stablehlo.multiply %v4269, %v4270 : tensor<192x768xf32>
    %v4272 = stablehlo.subtract %b7_Wfc1, %v4271 : tensor<192x768xf32>
    %v4273 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4274 = stablehlo.reshape %v4263 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4275 = stablehlo.reduce(%v4274 init: %v4273) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<768xf32>
    %v4276 = stablehlo.constant dense<0.003125> : tensor<768xf32>
    %v4277 = stablehlo.multiply %v4275, %v4276 : tensor<768xf32>
    %v4278 = stablehlo.subtract %b7_bfc1, %v4277 : tensor<768xf32>
    %v4279 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4280 = stablehlo.reshape %v4266 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4281 = stablehlo.reduce(%v4280 init: %v4279) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v4282 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v4283 = stablehlo.multiply %v4281, %v4282 : tensor<192xf32>
    %v4284 = stablehlo.subtract %b7_bt2, %v4283 : tensor<192xf32>
    %v4285 = stablehlo.reshape %v1574 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4286 = stablehlo.reshape %v4266 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4287 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4288 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v4289 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v4290 = stablehlo.reduce(%v4285 init: %v4287) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4291 = stablehlo.broadcast_in_dim %v4290, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v4292 = stablehlo.divide %v4291, %v4288 : tensor<32x197x192xf32>
    %v4293 = stablehlo.subtract %v4285, %v4292 : tensor<32x197x192xf32>
    %v4294 = stablehlo.multiply %v4293, %v4293 : tensor<32x197x192xf32>
    %v4295 = stablehlo.reduce(%v4294 init: %v4287) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4296 = stablehlo.broadcast_in_dim %v4295, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v4297 = stablehlo.divide %v4296, %v4288 : tensor<32x197x192xf32>
    %v4298 = stablehlo.add %v4297, %v4289 : tensor<32x197x192xf32>
    %v4299 = stablehlo.rsqrt %v4298 : tensor<32x197x192xf32>
    %v4300 = stablehlo.multiply %v4293, %v4299 : tensor<32x197x192xf32>
    %v4301 = stablehlo.multiply %v4286, %v4300 : tensor<32x197x192xf32>
    %v4302 = stablehlo.reduce(%v4301 init: %v4287) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v4303 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v4304 = stablehlo.multiply %v4302, %v4303 : tensor<192xf32>
    %v4305 = stablehlo.subtract %b7_g2, %v4304 : tensor<192xf32>
    %v4306 = stablehlo.reshape %v4266 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4307 = stablehlo.broadcast_in_dim %b7_g2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v4308 = stablehlo.multiply %v4306, %v4307 : tensor<32x197x192xf32>
    %v4309 = stablehlo.reshape %v4308 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v4310 = stablehlo.reshape %v4309 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4311 = stablehlo.reshape %v1574 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4312 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4313 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v4314 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v4315 = stablehlo.reduce(%v4311 init: %v4312) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4316 = stablehlo.broadcast_in_dim %v4315, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v4317 = stablehlo.divide %v4316, %v4313 : tensor<32x197x192xf32>
    %v4318 = stablehlo.subtract %v4311, %v4317 : tensor<32x197x192xf32>
    %v4319 = stablehlo.multiply %v4318, %v4318 : tensor<32x197x192xf32>
    %v4320 = stablehlo.reduce(%v4319 init: %v4312) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4321 = stablehlo.broadcast_in_dim %v4320, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v4322 = stablehlo.divide %v4321, %v4313 : tensor<32x197x192xf32>
    %v4323 = stablehlo.add %v4322, %v4314 : tensor<32x197x192xf32>
    %v4324 = stablehlo.rsqrt %v4323 : tensor<32x197x192xf32>
    %v4325 = stablehlo.multiply %v4318, %v4324 : tensor<32x197x192xf32>
    %v4326 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v4327 = stablehlo.multiply %v4326, %v4310 : tensor<32x197x192xf32>
    %v4328 = stablehlo.reduce(%v4327 init: %v4312) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4329 = stablehlo.broadcast_in_dim %v4328, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v4330 = stablehlo.multiply %v4325, %v4327 : tensor<32x197x192xf32>
    %v4331 = stablehlo.reduce(%v4330 init: %v4312) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4332 = stablehlo.broadcast_in_dim %v4331, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v4333 = stablehlo.multiply %v4327, %v4313 : tensor<32x197x192xf32>
    %v4334 = stablehlo.subtract %v4333, %v4329 : tensor<32x197x192xf32>
    %v4335 = stablehlo.multiply %v4325, %v4332 : tensor<32x197x192xf32>
    %v4336 = stablehlo.subtract %v4334, %v4335 : tensor<32x197x192xf32>
    %v4337 = stablehlo.divide %v4324, %v4313 : tensor<32x197x192xf32>
    %v4338 = stablehlo.multiply %v4337, %v4336 : tensor<32x197x192xf32>
    %v4339 = stablehlo.reshape %v4338 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v4340 = stablehlo.add %v4225, %v4339 : tensor<32x37824xf32>
    %v4341 = stablehlo.reshape %v4340 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4342 = stablehlo.dot_general %v4341, %b7_Wo, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v4343 = stablehlo.reshape %v4342 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v4344 = stablehlo.reshape %v1568 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4345 = stablehlo.reshape %v4340 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4346 = stablehlo.dot_general %v4344, %v4345, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %v4347 = stablehlo.constant dense<0.003125> : tensor<192x192xf32>
    %v4348 = stablehlo.multiply %v4346, %v4347 : tensor<192x192xf32>
    %v4349 = stablehlo.subtract %b7_Wo, %v4348 : tensor<192x192xf32>
    %v4350 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4351 = stablehlo.reshape %v4340 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4352 = stablehlo.reduce(%v4351 init: %v4350) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v4353 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v4354 = stablehlo.multiply %v4352, %v4353 : tensor<192xf32>
    %v4355 = stablehlo.subtract %b7_bo, %v4354 : tensor<192xf32>
    %v4356 = stablehlo.reshape %v4343 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4357 = stablehlo.slice %v4356 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v4358 = stablehlo.reshape %v4357 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4359 = stablehlo.reshape %v1476 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4360 = stablehlo.transpose %v4359, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v4361 = stablehlo.reshape %v4360 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4362 = stablehlo.reshape %v4358 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4363 = stablehlo.reshape %v4361 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4364 = stablehlo.dot_general %v4362, %v4363, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v4365 = stablehlo.reshape %v4364 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4366 = stablehlo.reshape %v1492 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4367 = stablehlo.transpose %v4366, dims = [0, 2, 1] : (tensor<32x197x197xf32>) -> tensor<32x197x197xf32>
    %v4368 = stablehlo.reshape %v4367 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4369 = stablehlo.reshape %v4368 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4370 = stablehlo.reshape %v4358 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4371 = stablehlo.dot_general %v4369, %v4370, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4372 = stablehlo.reshape %v4371 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4373 = stablehlo.reshape %v1485 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4374 = stablehlo.reshape %v4365 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4375 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4376 = stablehlo.exponential %v4373 : tensor<32x197x197xf32>
    %v4377 = stablehlo.reduce(%v4376 init: %v4375) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4378 = stablehlo.broadcast_in_dim %v4377, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4379 = stablehlo.divide %v4376, %v4378 : tensor<32x197x197xf32>
    %v4380 = stablehlo.multiply %v4379, %v4374 : tensor<32x197x197xf32>
    %v4381 = stablehlo.reduce(%v4380 init: %v4375) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4382 = stablehlo.broadcast_in_dim %v4381, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4383 = stablehlo.subtract %v4374, %v4382 : tensor<32x197x197xf32>
    %v4384 = stablehlo.multiply %v4379, %v4383 : tensor<32x197x197xf32>
    %v4385 = stablehlo.reshape %v4384 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4386 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v4387 = stablehlo.multiply %v4385, %v4386 : tensor<32x38809xf32>
    %v4388 = stablehlo.reshape %v4387 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4389 = stablehlo.reshape %v1473 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4390 = stablehlo.dot_general %v4388, %v4389, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4391 = stablehlo.reshape %v4390 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4392 = stablehlo.reshape %v1470 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4393 = stablehlo.transpose %v4392, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v4394 = stablehlo.reshape %v4393 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4395 = stablehlo.reshape %v4394 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4396 = stablehlo.reshape %v4387 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4397 = stablehlo.dot_general %v4395, %v4396, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x64x197xf32>, tensor<32x197x197xf32>) -> tensor<32x64x197xf32>
    %v4398 = stablehlo.reshape %v4397 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4399 = stablehlo.reshape %v4398 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4400 = stablehlo.transpose %v4399, dims = [0, 2, 1] : (tensor<32x64x197xf32>) -> tensor<32x197x64xf32>
    %v4401 = stablehlo.reshape %v4400 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4402 = stablehlo.reshape %v4391 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4403 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4404 = stablehlo.pad %v4402, %v4403, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v4405 = stablehlo.reshape %v4404 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v4406 = stablehlo.reshape %v4401 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4407 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4408 = stablehlo.pad %v4406, %v4407, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v4409 = stablehlo.reshape %v4408 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v4410 = stablehlo.reshape %v4372 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4411 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4412 = stablehlo.pad %v4410, %v4411, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v4413 = stablehlo.reshape %v4412 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v4414 = stablehlo.reshape %v4343 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4415 = stablehlo.slice %v4414 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v4416 = stablehlo.reshape %v4415 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4417 = stablehlo.reshape %v1509 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4418 = stablehlo.transpose %v4417, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v4419 = stablehlo.reshape %v4418 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4420 = stablehlo.reshape %v4416 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4421 = stablehlo.reshape %v4419 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4422 = stablehlo.dot_general %v4420, %v4421, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v4423 = stablehlo.reshape %v4422 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4424 = stablehlo.reshape %v1525 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4425 = stablehlo.transpose %v4424, dims = [0, 2, 1] : (tensor<32x197x197xf32>) -> tensor<32x197x197xf32>
    %v4426 = stablehlo.reshape %v4425 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4427 = stablehlo.reshape %v4426 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4428 = stablehlo.reshape %v4416 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4429 = stablehlo.dot_general %v4427, %v4428, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4430 = stablehlo.reshape %v4429 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4431 = stablehlo.reshape %v1518 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4432 = stablehlo.reshape %v4423 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4433 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4434 = stablehlo.exponential %v4431 : tensor<32x197x197xf32>
    %v4435 = stablehlo.reduce(%v4434 init: %v4433) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4436 = stablehlo.broadcast_in_dim %v4435, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4437 = stablehlo.divide %v4434, %v4436 : tensor<32x197x197xf32>
    %v4438 = stablehlo.multiply %v4437, %v4432 : tensor<32x197x197xf32>
    %v4439 = stablehlo.reduce(%v4438 init: %v4433) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4440 = stablehlo.broadcast_in_dim %v4439, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4441 = stablehlo.subtract %v4432, %v4440 : tensor<32x197x197xf32>
    %v4442 = stablehlo.multiply %v4437, %v4441 : tensor<32x197x197xf32>
    %v4443 = stablehlo.reshape %v4442 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4444 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v4445 = stablehlo.multiply %v4443, %v4444 : tensor<32x38809xf32>
    %v4446 = stablehlo.reshape %v4445 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4447 = stablehlo.reshape %v1506 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4448 = stablehlo.dot_general %v4446, %v4447, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4449 = stablehlo.reshape %v4448 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4450 = stablehlo.reshape %v1503 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4451 = stablehlo.transpose %v4450, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v4452 = stablehlo.reshape %v4451 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4453 = stablehlo.reshape %v4452 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4454 = stablehlo.reshape %v4445 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4455 = stablehlo.dot_general %v4453, %v4454, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x64x197xf32>, tensor<32x197x197xf32>) -> tensor<32x64x197xf32>
    %v4456 = stablehlo.reshape %v4455 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4457 = stablehlo.reshape %v4456 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4458 = stablehlo.transpose %v4457, dims = [0, 2, 1] : (tensor<32x64x197xf32>) -> tensor<32x197x64xf32>
    %v4459 = stablehlo.reshape %v4458 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4460 = stablehlo.reshape %v4449 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4461 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4462 = stablehlo.pad %v4460, %v4461, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v4463 = stablehlo.reshape %v4462 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v4464 = stablehlo.reshape %v4459 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4465 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4466 = stablehlo.pad %v4464, %v4465, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v4467 = stablehlo.reshape %v4466 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v4468 = stablehlo.reshape %v4430 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4469 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4470 = stablehlo.pad %v4468, %v4469, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v4471 = stablehlo.reshape %v4470 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v4472 = stablehlo.add %v4405, %v4463 : tensor<32x37824xf32>
    %v4473 = stablehlo.add %v4409, %v4467 : tensor<32x37824xf32>
    %v4474 = stablehlo.add %v4413, %v4471 : tensor<32x37824xf32>
    %v4475 = stablehlo.reshape %v4343 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4476 = stablehlo.slice %v4475 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v4477 = stablehlo.reshape %v4476 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4478 = stablehlo.reshape %v1543 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4479 = stablehlo.transpose %v4478, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v4480 = stablehlo.reshape %v4479 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4481 = stablehlo.reshape %v4477 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4482 = stablehlo.reshape %v4480 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4483 = stablehlo.dot_general %v4481, %v4482, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v4484 = stablehlo.reshape %v4483 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4485 = stablehlo.reshape %v1559 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4486 = stablehlo.transpose %v4485, dims = [0, 2, 1] : (tensor<32x197x197xf32>) -> tensor<32x197x197xf32>
    %v4487 = stablehlo.reshape %v4486 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4488 = stablehlo.reshape %v4487 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4489 = stablehlo.reshape %v4477 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4490 = stablehlo.dot_general %v4488, %v4489, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4491 = stablehlo.reshape %v4490 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4492 = stablehlo.reshape %v1552 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4493 = stablehlo.reshape %v4484 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4494 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4495 = stablehlo.exponential %v4492 : tensor<32x197x197xf32>
    %v4496 = stablehlo.reduce(%v4495 init: %v4494) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4497 = stablehlo.broadcast_in_dim %v4496, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4498 = stablehlo.divide %v4495, %v4497 : tensor<32x197x197xf32>
    %v4499 = stablehlo.multiply %v4498, %v4493 : tensor<32x197x197xf32>
    %v4500 = stablehlo.reduce(%v4499 init: %v4494) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4501 = stablehlo.broadcast_in_dim %v4500, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4502 = stablehlo.subtract %v4493, %v4501 : tensor<32x197x197xf32>
    %v4503 = stablehlo.multiply %v4498, %v4502 : tensor<32x197x197xf32>
    %v4504 = stablehlo.reshape %v4503 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4505 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v4506 = stablehlo.multiply %v4504, %v4505 : tensor<32x38809xf32>
    %v4507 = stablehlo.reshape %v4506 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4508 = stablehlo.reshape %v1540 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4509 = stablehlo.dot_general %v4507, %v4508, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4510 = stablehlo.reshape %v4509 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4511 = stablehlo.reshape %v1537 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4512 = stablehlo.transpose %v4511, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v4513 = stablehlo.reshape %v4512 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4514 = stablehlo.reshape %v4513 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4515 = stablehlo.reshape %v4506 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4516 = stablehlo.dot_general %v4514, %v4515, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x64x197xf32>, tensor<32x197x197xf32>) -> tensor<32x64x197xf32>
    %v4517 = stablehlo.reshape %v4516 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4518 = stablehlo.reshape %v4517 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4519 = stablehlo.transpose %v4518, dims = [0, 2, 1] : (tensor<32x64x197xf32>) -> tensor<32x197x64xf32>
    %v4520 = stablehlo.reshape %v4519 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4521 = stablehlo.reshape %v4510 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4522 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4523 = stablehlo.pad %v4521, %v4522, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v4524 = stablehlo.reshape %v4523 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v4525 = stablehlo.reshape %v4520 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4526 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4527 = stablehlo.pad %v4525, %v4526, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v4528 = stablehlo.reshape %v4527 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v4529 = stablehlo.reshape %v4491 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4530 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4531 = stablehlo.pad %v4529, %v4530, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v4532 = stablehlo.reshape %v4531 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v4533 = stablehlo.add %v4472, %v4524 : tensor<32x37824xf32>
    %v4534 = stablehlo.add %v4473, %v4528 : tensor<32x37824xf32>
    %v4535 = stablehlo.add %v4474, %v4532 : tensor<32x37824xf32>
    %v4536 = stablehlo.reshape %v4533 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4537 = stablehlo.dot_general %v4536, %b7_Wq, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v4538 = stablehlo.reshape %v4537 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v4539 = stablehlo.reshape %v1452 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4540 = stablehlo.reshape %v4533 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4541 = stablehlo.dot_general %v4539, %v4540, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %v4542 = stablehlo.constant dense<0.003125> : tensor<192x192xf32>
    %v4543 = stablehlo.multiply %v4541, %v4542 : tensor<192x192xf32>
    %v4544 = stablehlo.subtract %b7_Wq, %v4543 : tensor<192x192xf32>
    %v4545 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4546 = stablehlo.reshape %v4533 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4547 = stablehlo.reduce(%v4546 init: %v4545) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v4548 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v4549 = stablehlo.multiply %v4547, %v4548 : tensor<192xf32>
    %v4550 = stablehlo.subtract %b7_bq, %v4549 : tensor<192xf32>
    %v4551 = stablehlo.reshape %v4534 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4552 = stablehlo.dot_general %v4551, %b7_Wk, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v4553 = stablehlo.reshape %v4552 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v4554 = stablehlo.reshape %v1452 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4555 = stablehlo.reshape %v4534 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4556 = stablehlo.dot_general %v4554, %v4555, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %v4557 = stablehlo.constant dense<0.003125> : tensor<192x192xf32>
    %v4558 = stablehlo.multiply %v4556, %v4557 : tensor<192x192xf32>
    %v4559 = stablehlo.subtract %b7_Wk, %v4558 : tensor<192x192xf32>
    %v4560 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4561 = stablehlo.reshape %v4534 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4562 = stablehlo.reduce(%v4561 init: %v4560) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v4563 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v4564 = stablehlo.multiply %v4562, %v4563 : tensor<192xf32>
    %v4565 = stablehlo.subtract %b7_bk, %v4564 : tensor<192xf32>
    %v4566 = stablehlo.reshape %v4535 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4567 = stablehlo.dot_general %v4566, %b7_Wv, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v4568 = stablehlo.reshape %v4567 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v4569 = stablehlo.reshape %v1452 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4570 = stablehlo.reshape %v4535 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4571 = stablehlo.dot_general %v4569, %v4570, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %v4572 = stablehlo.constant dense<0.003125> : tensor<192x192xf32>
    %v4573 = stablehlo.multiply %v4571, %v4572 : tensor<192x192xf32>
    %v4574 = stablehlo.subtract %b7_Wv, %v4573 : tensor<192x192xf32>
    %v4575 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4576 = stablehlo.reshape %v4535 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4577 = stablehlo.reduce(%v4576 init: %v4575) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v4578 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v4579 = stablehlo.multiply %v4577, %v4578 : tensor<192xf32>
    %v4580 = stablehlo.subtract %b7_bv, %v4579 : tensor<192xf32>
    %v4581 = stablehlo.add %v4538, %v4553 : tensor<32x37824xf32>
    %v4582 = stablehlo.add %v4581, %v4568 : tensor<32x37824xf32>
    %v4583 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4584 = stablehlo.reshape %v4582 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4585 = stablehlo.reduce(%v4584 init: %v4583) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v4586 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v4587 = stablehlo.multiply %v4585, %v4586 : tensor<192xf32>
    %v4588 = stablehlo.subtract %b7_bt1, %v4587 : tensor<192xf32>
    %v4589 = stablehlo.reshape %v1424 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4590 = stablehlo.reshape %v4582 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4591 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4592 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v4593 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v4594 = stablehlo.reduce(%v4589 init: %v4591) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4595 = stablehlo.broadcast_in_dim %v4594, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v4596 = stablehlo.divide %v4595, %v4592 : tensor<32x197x192xf32>
    %v4597 = stablehlo.subtract %v4589, %v4596 : tensor<32x197x192xf32>
    %v4598 = stablehlo.multiply %v4597, %v4597 : tensor<32x197x192xf32>
    %v4599 = stablehlo.reduce(%v4598 init: %v4591) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4600 = stablehlo.broadcast_in_dim %v4599, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v4601 = stablehlo.divide %v4600, %v4592 : tensor<32x197x192xf32>
    %v4602 = stablehlo.add %v4601, %v4593 : tensor<32x197x192xf32>
    %v4603 = stablehlo.rsqrt %v4602 : tensor<32x197x192xf32>
    %v4604 = stablehlo.multiply %v4597, %v4603 : tensor<32x197x192xf32>
    %v4605 = stablehlo.multiply %v4590, %v4604 : tensor<32x197x192xf32>
    %v4606 = stablehlo.reduce(%v4605 init: %v4591) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v4607 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v4608 = stablehlo.multiply %v4606, %v4607 : tensor<192xf32>
    %v4609 = stablehlo.subtract %b7_g1, %v4608 : tensor<192xf32>
    %v4610 = stablehlo.reshape %v4582 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4611 = stablehlo.broadcast_in_dim %b7_g1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v4612 = stablehlo.multiply %v4610, %v4611 : tensor<32x197x192xf32>
    %v4613 = stablehlo.reshape %v4612 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v4614 = stablehlo.reshape %v4613 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4615 = stablehlo.reshape %v1424 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4616 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4617 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v4618 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v4619 = stablehlo.reduce(%v4615 init: %v4616) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4620 = stablehlo.broadcast_in_dim %v4619, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v4621 = stablehlo.divide %v4620, %v4617 : tensor<32x197x192xf32>
    %v4622 = stablehlo.subtract %v4615, %v4621 : tensor<32x197x192xf32>
    %v4623 = stablehlo.multiply %v4622, %v4622 : tensor<32x197x192xf32>
    %v4624 = stablehlo.reduce(%v4623 init: %v4616) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4625 = stablehlo.broadcast_in_dim %v4624, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v4626 = stablehlo.divide %v4625, %v4617 : tensor<32x197x192xf32>
    %v4627 = stablehlo.add %v4626, %v4618 : tensor<32x197x192xf32>
    %v4628 = stablehlo.rsqrt %v4627 : tensor<32x197x192xf32>
    %v4629 = stablehlo.multiply %v4622, %v4628 : tensor<32x197x192xf32>
    %v4630 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v4631 = stablehlo.multiply %v4630, %v4614 : tensor<32x197x192xf32>
    %v4632 = stablehlo.reduce(%v4631 init: %v4616) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4633 = stablehlo.broadcast_in_dim %v4632, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v4634 = stablehlo.multiply %v4629, %v4631 : tensor<32x197x192xf32>
    %v4635 = stablehlo.reduce(%v4634 init: %v4616) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4636 = stablehlo.broadcast_in_dim %v4635, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v4637 = stablehlo.multiply %v4631, %v4617 : tensor<32x197x192xf32>
    %v4638 = stablehlo.subtract %v4637, %v4633 : tensor<32x197x192xf32>
    %v4639 = stablehlo.multiply %v4629, %v4636 : tensor<32x197x192xf32>
    %v4640 = stablehlo.subtract %v4638, %v4639 : tensor<32x197x192xf32>
    %v4641 = stablehlo.divide %v4628, %v4617 : tensor<32x197x192xf32>
    %v4642 = stablehlo.multiply %v4641, %v4640 : tensor<32x197x192xf32>
    %v4643 = stablehlo.reshape %v4642 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v4644 = stablehlo.add %v4340, %v4643 : tensor<32x37824xf32>
    %v4645 = stablehlo.reshape %v4644 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4646 = stablehlo.dot_general %v4645, %b6_Wfc2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<768x192xf32>) -> tensor<32x197x768xf32>
    %v4647 = stablehlo.reshape %v4646 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4648 = stablehlo.reshape %v1418 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4649 = stablehlo.reshape %v4644 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4650 = stablehlo.dot_general %v4648, %v4649, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<32x197x192xf32>) -> tensor<768x192xf32>
    %v4651 = stablehlo.constant dense<0.003125> : tensor<768x192xf32>
    %v4652 = stablehlo.multiply %v4650, %v4651 : tensor<768x192xf32>
    %v4653 = stablehlo.subtract %b6_Wfc2, %v4652 : tensor<768x192xf32>
    %v4654 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4655 = stablehlo.reshape %v4644 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4656 = stablehlo.reduce(%v4655 init: %v4654) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v4657 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v4658 = stablehlo.multiply %v4656, %v4657 : tensor<192xf32>
    %v4659 = stablehlo.subtract %b6_bfc2, %v4658 : tensor<192xf32>
    %v4660 = stablehlo.multiply %v1405, %v1405 : tensor<32x151296xf32>
    %v4661 = stablehlo.multiply %v4660, %v1405 : tensor<32x151296xf32>
    %v4662 = stablehlo.constant dense<0.044715> : tensor<32x151296xf32>
    %v4663 = stablehlo.multiply %v4662, %v4661 : tensor<32x151296xf32>
    %v4664 = stablehlo.add %v1405, %v4663 : tensor<32x151296xf32>
    %v4665 = stablehlo.constant dense<0.7978845608028654> : tensor<32x151296xf32>
    %v4666 = stablehlo.multiply %v4665, %v4664 : tensor<32x151296xf32>
    %v4667 = stablehlo.tanh %v4666 : tensor<32x151296xf32>
    %v4668 = stablehlo.constant dense<1.0> : tensor<32x151296xf32>
    %v4669 = stablehlo.add %v4668, %v4667 : tensor<32x151296xf32>
    %v4670 = stablehlo.constant dense<0.5> : tensor<32x151296xf32>
    %v4671 = stablehlo.multiply %v4670, %v4669 : tensor<32x151296xf32>
    %v4672 = stablehlo.multiply %v4667, %v4667 : tensor<32x151296xf32>
    %v4673 = stablehlo.subtract %v4668, %v4672 : tensor<32x151296xf32>
    %v4674 = stablehlo.multiply %v4670, %v1405 : tensor<32x151296xf32>
    %v4675 = stablehlo.multiply %v4674, %v4673 : tensor<32x151296xf32>
    %v4676 = stablehlo.constant dense<0.134145> : tensor<32x151296xf32>
    %v4677 = stablehlo.multiply %v4676, %v4660 : tensor<32x151296xf32>
    %v4678 = stablehlo.add %v4668, %v4677 : tensor<32x151296xf32>
    %v4679 = stablehlo.multiply %v4665, %v4678 : tensor<32x151296xf32>
    %v4680 = stablehlo.multiply %v4675, %v4679 : tensor<32x151296xf32>
    %v4681 = stablehlo.add %v4671, %v4680 : tensor<32x151296xf32>
    %v4682 = stablehlo.multiply %v4647, %v4681 : tensor<32x151296xf32>
    %v4683 = stablehlo.reshape %v4682 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4684 = stablehlo.dot_general %v4683, %b6_Wfc1, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<192x768xf32>) -> tensor<32x197x192xf32>
    %v4685 = stablehlo.reshape %v4684 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v4686 = stablehlo.reshape %v1400 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4687 = stablehlo.reshape %v4682 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4688 = stablehlo.dot_general %v4686, %v4687, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x768xf32>) -> tensor<192x768xf32>
    %v4689 = stablehlo.constant dense<0.003125> : tensor<192x768xf32>
    %v4690 = stablehlo.multiply %v4688, %v4689 : tensor<192x768xf32>
    %v4691 = stablehlo.subtract %b6_Wfc1, %v4690 : tensor<192x768xf32>
    %v4692 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4693 = stablehlo.reshape %v4682 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4694 = stablehlo.reduce(%v4693 init: %v4692) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<768xf32>
    %v4695 = stablehlo.constant dense<0.003125> : tensor<768xf32>
    %v4696 = stablehlo.multiply %v4694, %v4695 : tensor<768xf32>
    %v4697 = stablehlo.subtract %b6_bfc1, %v4696 : tensor<768xf32>
    %v4698 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4699 = stablehlo.reshape %v4685 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4700 = stablehlo.reduce(%v4699 init: %v4698) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v4701 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v4702 = stablehlo.multiply %v4700, %v4701 : tensor<192xf32>
    %v4703 = stablehlo.subtract %b6_bt2, %v4702 : tensor<192xf32>
    %v4704 = stablehlo.reshape %v1372 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4705 = stablehlo.reshape %v4685 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4706 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4707 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v4708 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v4709 = stablehlo.reduce(%v4704 init: %v4706) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4710 = stablehlo.broadcast_in_dim %v4709, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v4711 = stablehlo.divide %v4710, %v4707 : tensor<32x197x192xf32>
    %v4712 = stablehlo.subtract %v4704, %v4711 : tensor<32x197x192xf32>
    %v4713 = stablehlo.multiply %v4712, %v4712 : tensor<32x197x192xf32>
    %v4714 = stablehlo.reduce(%v4713 init: %v4706) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4715 = stablehlo.broadcast_in_dim %v4714, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v4716 = stablehlo.divide %v4715, %v4707 : tensor<32x197x192xf32>
    %v4717 = stablehlo.add %v4716, %v4708 : tensor<32x197x192xf32>
    %v4718 = stablehlo.rsqrt %v4717 : tensor<32x197x192xf32>
    %v4719 = stablehlo.multiply %v4712, %v4718 : tensor<32x197x192xf32>
    %v4720 = stablehlo.multiply %v4705, %v4719 : tensor<32x197x192xf32>
    %v4721 = stablehlo.reduce(%v4720 init: %v4706) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v4722 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v4723 = stablehlo.multiply %v4721, %v4722 : tensor<192xf32>
    %v4724 = stablehlo.subtract %b6_g2, %v4723 : tensor<192xf32>
    %v4725 = stablehlo.reshape %v4685 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4726 = stablehlo.broadcast_in_dim %b6_g2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v4727 = stablehlo.multiply %v4725, %v4726 : tensor<32x197x192xf32>
    %v4728 = stablehlo.reshape %v4727 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v4729 = stablehlo.reshape %v4728 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4730 = stablehlo.reshape %v1372 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4731 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4732 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v4733 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v4734 = stablehlo.reduce(%v4730 init: %v4731) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4735 = stablehlo.broadcast_in_dim %v4734, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v4736 = stablehlo.divide %v4735, %v4732 : tensor<32x197x192xf32>
    %v4737 = stablehlo.subtract %v4730, %v4736 : tensor<32x197x192xf32>
    %v4738 = stablehlo.multiply %v4737, %v4737 : tensor<32x197x192xf32>
    %v4739 = stablehlo.reduce(%v4738 init: %v4731) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4740 = stablehlo.broadcast_in_dim %v4739, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v4741 = stablehlo.divide %v4740, %v4732 : tensor<32x197x192xf32>
    %v4742 = stablehlo.add %v4741, %v4733 : tensor<32x197x192xf32>
    %v4743 = stablehlo.rsqrt %v4742 : tensor<32x197x192xf32>
    %v4744 = stablehlo.multiply %v4737, %v4743 : tensor<32x197x192xf32>
    %v4745 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v4746 = stablehlo.multiply %v4745, %v4729 : tensor<32x197x192xf32>
    %v4747 = stablehlo.reduce(%v4746 init: %v4731) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4748 = stablehlo.broadcast_in_dim %v4747, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v4749 = stablehlo.multiply %v4744, %v4746 : tensor<32x197x192xf32>
    %v4750 = stablehlo.reduce(%v4749 init: %v4731) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4751 = stablehlo.broadcast_in_dim %v4750, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v4752 = stablehlo.multiply %v4746, %v4732 : tensor<32x197x192xf32>
    %v4753 = stablehlo.subtract %v4752, %v4748 : tensor<32x197x192xf32>
    %v4754 = stablehlo.multiply %v4744, %v4751 : tensor<32x197x192xf32>
    %v4755 = stablehlo.subtract %v4753, %v4754 : tensor<32x197x192xf32>
    %v4756 = stablehlo.divide %v4743, %v4732 : tensor<32x197x192xf32>
    %v4757 = stablehlo.multiply %v4756, %v4755 : tensor<32x197x192xf32>
    %v4758 = stablehlo.reshape %v4757 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v4759 = stablehlo.add %v4644, %v4758 : tensor<32x37824xf32>
    %v4760 = stablehlo.reshape %v4759 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4761 = stablehlo.dot_general %v4760, %b6_Wo, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v4762 = stablehlo.reshape %v4761 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v4763 = stablehlo.reshape %v1366 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4764 = stablehlo.reshape %v4759 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4765 = stablehlo.dot_general %v4763, %v4764, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %v4766 = stablehlo.constant dense<0.003125> : tensor<192x192xf32>
    %v4767 = stablehlo.multiply %v4765, %v4766 : tensor<192x192xf32>
    %v4768 = stablehlo.subtract %b6_Wo, %v4767 : tensor<192x192xf32>
    %v4769 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4770 = stablehlo.reshape %v4759 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4771 = stablehlo.reduce(%v4770 init: %v4769) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v4772 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v4773 = stablehlo.multiply %v4771, %v4772 : tensor<192xf32>
    %v4774 = stablehlo.subtract %b6_bo, %v4773 : tensor<192xf32>
    %v4775 = stablehlo.reshape %v4762 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4776 = stablehlo.slice %v4775 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v4777 = stablehlo.reshape %v4776 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4778 = stablehlo.reshape %v1274 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4779 = stablehlo.transpose %v4778, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v4780 = stablehlo.reshape %v4779 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4781 = stablehlo.reshape %v4777 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4782 = stablehlo.reshape %v4780 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4783 = stablehlo.dot_general %v4781, %v4782, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v4784 = stablehlo.reshape %v4783 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4785 = stablehlo.reshape %v1290 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4786 = stablehlo.transpose %v4785, dims = [0, 2, 1] : (tensor<32x197x197xf32>) -> tensor<32x197x197xf32>
    %v4787 = stablehlo.reshape %v4786 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4788 = stablehlo.reshape %v4787 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4789 = stablehlo.reshape %v4777 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4790 = stablehlo.dot_general %v4788, %v4789, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4791 = stablehlo.reshape %v4790 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4792 = stablehlo.reshape %v1283 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4793 = stablehlo.reshape %v4784 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4794 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4795 = stablehlo.exponential %v4792 : tensor<32x197x197xf32>
    %v4796 = stablehlo.reduce(%v4795 init: %v4794) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4797 = stablehlo.broadcast_in_dim %v4796, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4798 = stablehlo.divide %v4795, %v4797 : tensor<32x197x197xf32>
    %v4799 = stablehlo.multiply %v4798, %v4793 : tensor<32x197x197xf32>
    %v4800 = stablehlo.reduce(%v4799 init: %v4794) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4801 = stablehlo.broadcast_in_dim %v4800, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4802 = stablehlo.subtract %v4793, %v4801 : tensor<32x197x197xf32>
    %v4803 = stablehlo.multiply %v4798, %v4802 : tensor<32x197x197xf32>
    %v4804 = stablehlo.reshape %v4803 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4805 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v4806 = stablehlo.multiply %v4804, %v4805 : tensor<32x38809xf32>
    %v4807 = stablehlo.reshape %v4806 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4808 = stablehlo.reshape %v1271 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4809 = stablehlo.dot_general %v4807, %v4808, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4810 = stablehlo.reshape %v4809 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4811 = stablehlo.reshape %v1268 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4812 = stablehlo.transpose %v4811, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v4813 = stablehlo.reshape %v4812 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4814 = stablehlo.reshape %v4813 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4815 = stablehlo.reshape %v4806 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4816 = stablehlo.dot_general %v4814, %v4815, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x64x197xf32>, tensor<32x197x197xf32>) -> tensor<32x64x197xf32>
    %v4817 = stablehlo.reshape %v4816 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4818 = stablehlo.reshape %v4817 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4819 = stablehlo.transpose %v4818, dims = [0, 2, 1] : (tensor<32x64x197xf32>) -> tensor<32x197x64xf32>
    %v4820 = stablehlo.reshape %v4819 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4821 = stablehlo.reshape %v4810 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4822 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4823 = stablehlo.pad %v4821, %v4822, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v4824 = stablehlo.reshape %v4823 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v4825 = stablehlo.reshape %v4820 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4826 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4827 = stablehlo.pad %v4825, %v4826, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v4828 = stablehlo.reshape %v4827 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v4829 = stablehlo.reshape %v4791 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4830 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4831 = stablehlo.pad %v4829, %v4830, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v4832 = stablehlo.reshape %v4831 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v4833 = stablehlo.reshape %v4762 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4834 = stablehlo.slice %v4833 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v4835 = stablehlo.reshape %v4834 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4836 = stablehlo.reshape %v1307 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4837 = stablehlo.transpose %v4836, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v4838 = stablehlo.reshape %v4837 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4839 = stablehlo.reshape %v4835 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4840 = stablehlo.reshape %v4838 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4841 = stablehlo.dot_general %v4839, %v4840, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v4842 = stablehlo.reshape %v4841 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4843 = stablehlo.reshape %v1323 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4844 = stablehlo.transpose %v4843, dims = [0, 2, 1] : (tensor<32x197x197xf32>) -> tensor<32x197x197xf32>
    %v4845 = stablehlo.reshape %v4844 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4846 = stablehlo.reshape %v4845 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4847 = stablehlo.reshape %v4835 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4848 = stablehlo.dot_general %v4846, %v4847, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4849 = stablehlo.reshape %v4848 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4850 = stablehlo.reshape %v1316 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4851 = stablehlo.reshape %v4842 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4852 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4853 = stablehlo.exponential %v4850 : tensor<32x197x197xf32>
    %v4854 = stablehlo.reduce(%v4853 init: %v4852) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4855 = stablehlo.broadcast_in_dim %v4854, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4856 = stablehlo.divide %v4853, %v4855 : tensor<32x197x197xf32>
    %v4857 = stablehlo.multiply %v4856, %v4851 : tensor<32x197x197xf32>
    %v4858 = stablehlo.reduce(%v4857 init: %v4852) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4859 = stablehlo.broadcast_in_dim %v4858, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4860 = stablehlo.subtract %v4851, %v4859 : tensor<32x197x197xf32>
    %v4861 = stablehlo.multiply %v4856, %v4860 : tensor<32x197x197xf32>
    %v4862 = stablehlo.reshape %v4861 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4863 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v4864 = stablehlo.multiply %v4862, %v4863 : tensor<32x38809xf32>
    %v4865 = stablehlo.reshape %v4864 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4866 = stablehlo.reshape %v1304 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4867 = stablehlo.dot_general %v4865, %v4866, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4868 = stablehlo.reshape %v4867 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4869 = stablehlo.reshape %v1301 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4870 = stablehlo.transpose %v4869, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v4871 = stablehlo.reshape %v4870 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4872 = stablehlo.reshape %v4871 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4873 = stablehlo.reshape %v4864 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4874 = stablehlo.dot_general %v4872, %v4873, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x64x197xf32>, tensor<32x197x197xf32>) -> tensor<32x64x197xf32>
    %v4875 = stablehlo.reshape %v4874 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4876 = stablehlo.reshape %v4875 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4877 = stablehlo.transpose %v4876, dims = [0, 2, 1] : (tensor<32x64x197xf32>) -> tensor<32x197x64xf32>
    %v4878 = stablehlo.reshape %v4877 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4879 = stablehlo.reshape %v4868 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4880 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4881 = stablehlo.pad %v4879, %v4880, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v4882 = stablehlo.reshape %v4881 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v4883 = stablehlo.reshape %v4878 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4884 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4885 = stablehlo.pad %v4883, %v4884, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v4886 = stablehlo.reshape %v4885 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v4887 = stablehlo.reshape %v4849 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4888 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4889 = stablehlo.pad %v4887, %v4888, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v4890 = stablehlo.reshape %v4889 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v4891 = stablehlo.add %v4824, %v4882 : tensor<32x37824xf32>
    %v4892 = stablehlo.add %v4828, %v4886 : tensor<32x37824xf32>
    %v4893 = stablehlo.add %v4832, %v4890 : tensor<32x37824xf32>
    %v4894 = stablehlo.reshape %v4762 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4895 = stablehlo.slice %v4894 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v4896 = stablehlo.reshape %v4895 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4897 = stablehlo.reshape %v1341 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4898 = stablehlo.transpose %v4897, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v4899 = stablehlo.reshape %v4898 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4900 = stablehlo.reshape %v4896 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4901 = stablehlo.reshape %v4899 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4902 = stablehlo.dot_general %v4900, %v4901, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v4903 = stablehlo.reshape %v4902 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4904 = stablehlo.reshape %v1357 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4905 = stablehlo.transpose %v4904, dims = [0, 2, 1] : (tensor<32x197x197xf32>) -> tensor<32x197x197xf32>
    %v4906 = stablehlo.reshape %v4905 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4907 = stablehlo.reshape %v4906 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4908 = stablehlo.reshape %v4896 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4909 = stablehlo.dot_general %v4907, %v4908, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4910 = stablehlo.reshape %v4909 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4911 = stablehlo.reshape %v1350 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4912 = stablehlo.reshape %v4903 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4913 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4914 = stablehlo.exponential %v4911 : tensor<32x197x197xf32>
    %v4915 = stablehlo.reduce(%v4914 init: %v4913) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4916 = stablehlo.broadcast_in_dim %v4915, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4917 = stablehlo.divide %v4914, %v4916 : tensor<32x197x197xf32>
    %v4918 = stablehlo.multiply %v4917, %v4912 : tensor<32x197x197xf32>
    %v4919 = stablehlo.reduce(%v4918 init: %v4913) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4920 = stablehlo.broadcast_in_dim %v4919, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4921 = stablehlo.subtract %v4912, %v4920 : tensor<32x197x197xf32>
    %v4922 = stablehlo.multiply %v4917, %v4921 : tensor<32x197x197xf32>
    %v4923 = stablehlo.reshape %v4922 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4924 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v4925 = stablehlo.multiply %v4923, %v4924 : tensor<32x38809xf32>
    %v4926 = stablehlo.reshape %v4925 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4927 = stablehlo.reshape %v1338 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4928 = stablehlo.dot_general %v4926, %v4927, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4929 = stablehlo.reshape %v4928 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4930 = stablehlo.reshape %v1335 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4931 = stablehlo.transpose %v4930, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v4932 = stablehlo.reshape %v4931 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4933 = stablehlo.reshape %v4932 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4934 = stablehlo.reshape %v4925 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4935 = stablehlo.dot_general %v4933, %v4934, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x64x197xf32>, tensor<32x197x197xf32>) -> tensor<32x64x197xf32>
    %v4936 = stablehlo.reshape %v4935 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4937 = stablehlo.reshape %v4936 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4938 = stablehlo.transpose %v4937, dims = [0, 2, 1] : (tensor<32x64x197xf32>) -> tensor<32x197x64xf32>
    %v4939 = stablehlo.reshape %v4938 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4940 = stablehlo.reshape %v4929 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4941 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4942 = stablehlo.pad %v4940, %v4941, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v4943 = stablehlo.reshape %v4942 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v4944 = stablehlo.reshape %v4939 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4945 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4946 = stablehlo.pad %v4944, %v4945, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v4947 = stablehlo.reshape %v4946 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v4948 = stablehlo.reshape %v4910 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4949 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4950 = stablehlo.pad %v4948, %v4949, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v4951 = stablehlo.reshape %v4950 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v4952 = stablehlo.add %v4891, %v4943 : tensor<32x37824xf32>
    %v4953 = stablehlo.add %v4892, %v4947 : tensor<32x37824xf32>
    %v4954 = stablehlo.add %v4893, %v4951 : tensor<32x37824xf32>
    %v4955 = stablehlo.reshape %v4952 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4956 = stablehlo.dot_general %v4955, %b6_Wq, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v4957 = stablehlo.reshape %v4956 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v4958 = stablehlo.reshape %v1250 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4959 = stablehlo.reshape %v4952 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4960 = stablehlo.dot_general %v4958, %v4959, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %v4961 = stablehlo.constant dense<0.003125> : tensor<192x192xf32>
    %v4962 = stablehlo.multiply %v4960, %v4961 : tensor<192x192xf32>
    %v4963 = stablehlo.subtract %b6_Wq, %v4962 : tensor<192x192xf32>
    %v4964 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4965 = stablehlo.reshape %v4952 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4966 = stablehlo.reduce(%v4965 init: %v4964) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v4967 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v4968 = stablehlo.multiply %v4966, %v4967 : tensor<192xf32>
    %v4969 = stablehlo.subtract %b6_bq, %v4968 : tensor<192xf32>
    %v4970 = stablehlo.reshape %v4953 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4971 = stablehlo.dot_general %v4970, %b6_Wk, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v4972 = stablehlo.reshape %v4971 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v4973 = stablehlo.reshape %v1250 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4974 = stablehlo.reshape %v4953 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4975 = stablehlo.dot_general %v4973, %v4974, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %v4976 = stablehlo.constant dense<0.003125> : tensor<192x192xf32>
    %v4977 = stablehlo.multiply %v4975, %v4976 : tensor<192x192xf32>
    %v4978 = stablehlo.subtract %b6_Wk, %v4977 : tensor<192x192xf32>
    %v4979 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4980 = stablehlo.reshape %v4953 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4981 = stablehlo.reduce(%v4980 init: %v4979) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v4982 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v4983 = stablehlo.multiply %v4981, %v4982 : tensor<192xf32>
    %v4984 = stablehlo.subtract %b6_bk, %v4983 : tensor<192xf32>
    %v4985 = stablehlo.reshape %v4954 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4986 = stablehlo.dot_general %v4985, %b6_Wv, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v4987 = stablehlo.reshape %v4986 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v4988 = stablehlo.reshape %v1250 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4989 = stablehlo.reshape %v4954 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4990 = stablehlo.dot_general %v4988, %v4989, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %v4991 = stablehlo.constant dense<0.003125> : tensor<192x192xf32>
    %v4992 = stablehlo.multiply %v4990, %v4991 : tensor<192x192xf32>
    %v4993 = stablehlo.subtract %b6_Wv, %v4992 : tensor<192x192xf32>
    %v4994 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4995 = stablehlo.reshape %v4954 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v4996 = stablehlo.reduce(%v4995 init: %v4994) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v4997 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v4998 = stablehlo.multiply %v4996, %v4997 : tensor<192xf32>
    %v4999 = stablehlo.subtract %b6_bv, %v4998 : tensor<192xf32>
    %v5000 = stablehlo.add %v4957, %v4972 : tensor<32x37824xf32>
    %v5001 = stablehlo.add %v5000, %v4987 : tensor<32x37824xf32>
    %v5002 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5003 = stablehlo.reshape %v5001 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5004 = stablehlo.reduce(%v5003 init: %v5002) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v5005 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v5006 = stablehlo.multiply %v5004, %v5005 : tensor<192xf32>
    %v5007 = stablehlo.subtract %b6_bt1, %v5006 : tensor<192xf32>
    %v5008 = stablehlo.reshape %v1222 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5009 = stablehlo.reshape %v5001 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5010 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5011 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v5012 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v5013 = stablehlo.reduce(%v5008 init: %v5010) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5014 = stablehlo.broadcast_in_dim %v5013, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v5015 = stablehlo.divide %v5014, %v5011 : tensor<32x197x192xf32>
    %v5016 = stablehlo.subtract %v5008, %v5015 : tensor<32x197x192xf32>
    %v5017 = stablehlo.multiply %v5016, %v5016 : tensor<32x197x192xf32>
    %v5018 = stablehlo.reduce(%v5017 init: %v5010) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5019 = stablehlo.broadcast_in_dim %v5018, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v5020 = stablehlo.divide %v5019, %v5011 : tensor<32x197x192xf32>
    %v5021 = stablehlo.add %v5020, %v5012 : tensor<32x197x192xf32>
    %v5022 = stablehlo.rsqrt %v5021 : tensor<32x197x192xf32>
    %v5023 = stablehlo.multiply %v5016, %v5022 : tensor<32x197x192xf32>
    %v5024 = stablehlo.multiply %v5009, %v5023 : tensor<32x197x192xf32>
    %v5025 = stablehlo.reduce(%v5024 init: %v5010) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v5026 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v5027 = stablehlo.multiply %v5025, %v5026 : tensor<192xf32>
    %v5028 = stablehlo.subtract %b6_g1, %v5027 : tensor<192xf32>
    %v5029 = stablehlo.reshape %v5001 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5030 = stablehlo.broadcast_in_dim %b6_g1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v5031 = stablehlo.multiply %v5029, %v5030 : tensor<32x197x192xf32>
    %v5032 = stablehlo.reshape %v5031 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v5033 = stablehlo.reshape %v5032 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5034 = stablehlo.reshape %v1222 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5035 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5036 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v5037 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v5038 = stablehlo.reduce(%v5034 init: %v5035) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5039 = stablehlo.broadcast_in_dim %v5038, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v5040 = stablehlo.divide %v5039, %v5036 : tensor<32x197x192xf32>
    %v5041 = stablehlo.subtract %v5034, %v5040 : tensor<32x197x192xf32>
    %v5042 = stablehlo.multiply %v5041, %v5041 : tensor<32x197x192xf32>
    %v5043 = stablehlo.reduce(%v5042 init: %v5035) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5044 = stablehlo.broadcast_in_dim %v5043, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v5045 = stablehlo.divide %v5044, %v5036 : tensor<32x197x192xf32>
    %v5046 = stablehlo.add %v5045, %v5037 : tensor<32x197x192xf32>
    %v5047 = stablehlo.rsqrt %v5046 : tensor<32x197x192xf32>
    %v5048 = stablehlo.multiply %v5041, %v5047 : tensor<32x197x192xf32>
    %v5049 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v5050 = stablehlo.multiply %v5049, %v5033 : tensor<32x197x192xf32>
    %v5051 = stablehlo.reduce(%v5050 init: %v5035) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5052 = stablehlo.broadcast_in_dim %v5051, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v5053 = stablehlo.multiply %v5048, %v5050 : tensor<32x197x192xf32>
    %v5054 = stablehlo.reduce(%v5053 init: %v5035) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5055 = stablehlo.broadcast_in_dim %v5054, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v5056 = stablehlo.multiply %v5050, %v5036 : tensor<32x197x192xf32>
    %v5057 = stablehlo.subtract %v5056, %v5052 : tensor<32x197x192xf32>
    %v5058 = stablehlo.multiply %v5048, %v5055 : tensor<32x197x192xf32>
    %v5059 = stablehlo.subtract %v5057, %v5058 : tensor<32x197x192xf32>
    %v5060 = stablehlo.divide %v5047, %v5036 : tensor<32x197x192xf32>
    %v5061 = stablehlo.multiply %v5060, %v5059 : tensor<32x197x192xf32>
    %v5062 = stablehlo.reshape %v5061 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v5063 = stablehlo.add %v4759, %v5062 : tensor<32x37824xf32>
    %v5064 = stablehlo.reshape %v5063 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5065 = stablehlo.dot_general %v5064, %b5_Wfc2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<768x192xf32>) -> tensor<32x197x768xf32>
    %v5066 = stablehlo.reshape %v5065 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5067 = stablehlo.reshape %v1216 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5068 = stablehlo.reshape %v5063 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5069 = stablehlo.dot_general %v5067, %v5068, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<32x197x192xf32>) -> tensor<768x192xf32>
    %v5070 = stablehlo.constant dense<0.003125> : tensor<768x192xf32>
    %v5071 = stablehlo.multiply %v5069, %v5070 : tensor<768x192xf32>
    %v5072 = stablehlo.subtract %b5_Wfc2, %v5071 : tensor<768x192xf32>
    %v5073 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5074 = stablehlo.reshape %v5063 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5075 = stablehlo.reduce(%v5074 init: %v5073) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v5076 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v5077 = stablehlo.multiply %v5075, %v5076 : tensor<192xf32>
    %v5078 = stablehlo.subtract %b5_bfc2, %v5077 : tensor<192xf32>
    %v5079 = stablehlo.multiply %v1203, %v1203 : tensor<32x151296xf32>
    %v5080 = stablehlo.multiply %v5079, %v1203 : tensor<32x151296xf32>
    %v5081 = stablehlo.constant dense<0.044715> : tensor<32x151296xf32>
    %v5082 = stablehlo.multiply %v5081, %v5080 : tensor<32x151296xf32>
    %v5083 = stablehlo.add %v1203, %v5082 : tensor<32x151296xf32>
    %v5084 = stablehlo.constant dense<0.7978845608028654> : tensor<32x151296xf32>
    %v5085 = stablehlo.multiply %v5084, %v5083 : tensor<32x151296xf32>
    %v5086 = stablehlo.tanh %v5085 : tensor<32x151296xf32>
    %v5087 = stablehlo.constant dense<1.0> : tensor<32x151296xf32>
    %v5088 = stablehlo.add %v5087, %v5086 : tensor<32x151296xf32>
    %v5089 = stablehlo.constant dense<0.5> : tensor<32x151296xf32>
    %v5090 = stablehlo.multiply %v5089, %v5088 : tensor<32x151296xf32>
    %v5091 = stablehlo.multiply %v5086, %v5086 : tensor<32x151296xf32>
    %v5092 = stablehlo.subtract %v5087, %v5091 : tensor<32x151296xf32>
    %v5093 = stablehlo.multiply %v5089, %v1203 : tensor<32x151296xf32>
    %v5094 = stablehlo.multiply %v5093, %v5092 : tensor<32x151296xf32>
    %v5095 = stablehlo.constant dense<0.134145> : tensor<32x151296xf32>
    %v5096 = stablehlo.multiply %v5095, %v5079 : tensor<32x151296xf32>
    %v5097 = stablehlo.add %v5087, %v5096 : tensor<32x151296xf32>
    %v5098 = stablehlo.multiply %v5084, %v5097 : tensor<32x151296xf32>
    %v5099 = stablehlo.multiply %v5094, %v5098 : tensor<32x151296xf32>
    %v5100 = stablehlo.add %v5090, %v5099 : tensor<32x151296xf32>
    %v5101 = stablehlo.multiply %v5066, %v5100 : tensor<32x151296xf32>
    %v5102 = stablehlo.reshape %v5101 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5103 = stablehlo.dot_general %v5102, %b5_Wfc1, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<192x768xf32>) -> tensor<32x197x192xf32>
    %v5104 = stablehlo.reshape %v5103 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v5105 = stablehlo.reshape %v1198 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5106 = stablehlo.reshape %v5101 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5107 = stablehlo.dot_general %v5105, %v5106, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x768xf32>) -> tensor<192x768xf32>
    %v5108 = stablehlo.constant dense<0.003125> : tensor<192x768xf32>
    %v5109 = stablehlo.multiply %v5107, %v5108 : tensor<192x768xf32>
    %v5110 = stablehlo.subtract %b5_Wfc1, %v5109 : tensor<192x768xf32>
    %v5111 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5112 = stablehlo.reshape %v5101 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5113 = stablehlo.reduce(%v5112 init: %v5111) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<768xf32>
    %v5114 = stablehlo.constant dense<0.003125> : tensor<768xf32>
    %v5115 = stablehlo.multiply %v5113, %v5114 : tensor<768xf32>
    %v5116 = stablehlo.subtract %b5_bfc1, %v5115 : tensor<768xf32>
    %v5117 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5118 = stablehlo.reshape %v5104 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5119 = stablehlo.reduce(%v5118 init: %v5117) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v5120 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v5121 = stablehlo.multiply %v5119, %v5120 : tensor<192xf32>
    %v5122 = stablehlo.subtract %b5_bt2, %v5121 : tensor<192xf32>
    %v5123 = stablehlo.reshape %v1170 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5124 = stablehlo.reshape %v5104 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5125 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5126 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v5127 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v5128 = stablehlo.reduce(%v5123 init: %v5125) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5129 = stablehlo.broadcast_in_dim %v5128, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v5130 = stablehlo.divide %v5129, %v5126 : tensor<32x197x192xf32>
    %v5131 = stablehlo.subtract %v5123, %v5130 : tensor<32x197x192xf32>
    %v5132 = stablehlo.multiply %v5131, %v5131 : tensor<32x197x192xf32>
    %v5133 = stablehlo.reduce(%v5132 init: %v5125) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5134 = stablehlo.broadcast_in_dim %v5133, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v5135 = stablehlo.divide %v5134, %v5126 : tensor<32x197x192xf32>
    %v5136 = stablehlo.add %v5135, %v5127 : tensor<32x197x192xf32>
    %v5137 = stablehlo.rsqrt %v5136 : tensor<32x197x192xf32>
    %v5138 = stablehlo.multiply %v5131, %v5137 : tensor<32x197x192xf32>
    %v5139 = stablehlo.multiply %v5124, %v5138 : tensor<32x197x192xf32>
    %v5140 = stablehlo.reduce(%v5139 init: %v5125) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v5141 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v5142 = stablehlo.multiply %v5140, %v5141 : tensor<192xf32>
    %v5143 = stablehlo.subtract %b5_g2, %v5142 : tensor<192xf32>
    %v5144 = stablehlo.reshape %v5104 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5145 = stablehlo.broadcast_in_dim %b5_g2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v5146 = stablehlo.multiply %v5144, %v5145 : tensor<32x197x192xf32>
    %v5147 = stablehlo.reshape %v5146 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v5148 = stablehlo.reshape %v5147 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5149 = stablehlo.reshape %v1170 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5150 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5151 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v5152 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v5153 = stablehlo.reduce(%v5149 init: %v5150) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5154 = stablehlo.broadcast_in_dim %v5153, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v5155 = stablehlo.divide %v5154, %v5151 : tensor<32x197x192xf32>
    %v5156 = stablehlo.subtract %v5149, %v5155 : tensor<32x197x192xf32>
    %v5157 = stablehlo.multiply %v5156, %v5156 : tensor<32x197x192xf32>
    %v5158 = stablehlo.reduce(%v5157 init: %v5150) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5159 = stablehlo.broadcast_in_dim %v5158, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v5160 = stablehlo.divide %v5159, %v5151 : tensor<32x197x192xf32>
    %v5161 = stablehlo.add %v5160, %v5152 : tensor<32x197x192xf32>
    %v5162 = stablehlo.rsqrt %v5161 : tensor<32x197x192xf32>
    %v5163 = stablehlo.multiply %v5156, %v5162 : tensor<32x197x192xf32>
    %v5164 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v5165 = stablehlo.multiply %v5164, %v5148 : tensor<32x197x192xf32>
    %v5166 = stablehlo.reduce(%v5165 init: %v5150) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5167 = stablehlo.broadcast_in_dim %v5166, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v5168 = stablehlo.multiply %v5163, %v5165 : tensor<32x197x192xf32>
    %v5169 = stablehlo.reduce(%v5168 init: %v5150) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5170 = stablehlo.broadcast_in_dim %v5169, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v5171 = stablehlo.multiply %v5165, %v5151 : tensor<32x197x192xf32>
    %v5172 = stablehlo.subtract %v5171, %v5167 : tensor<32x197x192xf32>
    %v5173 = stablehlo.multiply %v5163, %v5170 : tensor<32x197x192xf32>
    %v5174 = stablehlo.subtract %v5172, %v5173 : tensor<32x197x192xf32>
    %v5175 = stablehlo.divide %v5162, %v5151 : tensor<32x197x192xf32>
    %v5176 = stablehlo.multiply %v5175, %v5174 : tensor<32x197x192xf32>
    %v5177 = stablehlo.reshape %v5176 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v5178 = stablehlo.add %v5063, %v5177 : tensor<32x37824xf32>
    %v5179 = stablehlo.reshape %v5178 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5180 = stablehlo.dot_general %v5179, %b5_Wo, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v5181 = stablehlo.reshape %v5180 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v5182 = stablehlo.reshape %v1164 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5183 = stablehlo.reshape %v5178 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5184 = stablehlo.dot_general %v5182, %v5183, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %v5185 = stablehlo.constant dense<0.003125> : tensor<192x192xf32>
    %v5186 = stablehlo.multiply %v5184, %v5185 : tensor<192x192xf32>
    %v5187 = stablehlo.subtract %b5_Wo, %v5186 : tensor<192x192xf32>
    %v5188 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5189 = stablehlo.reshape %v5178 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5190 = stablehlo.reduce(%v5189 init: %v5188) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v5191 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v5192 = stablehlo.multiply %v5190, %v5191 : tensor<192xf32>
    %v5193 = stablehlo.subtract %b5_bo, %v5192 : tensor<192xf32>
    %v5194 = stablehlo.reshape %v5181 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5195 = stablehlo.slice %v5194 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v5196 = stablehlo.reshape %v5195 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5197 = stablehlo.reshape %v1072 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5198 = stablehlo.transpose %v5197, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5199 = stablehlo.reshape %v5198 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5200 = stablehlo.reshape %v5196 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5201 = stablehlo.reshape %v5199 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5202 = stablehlo.dot_general %v5200, %v5201, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v5203 = stablehlo.reshape %v5202 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5204 = stablehlo.reshape %v1088 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5205 = stablehlo.transpose %v5204, dims = [0, 2, 1] : (tensor<32x197x197xf32>) -> tensor<32x197x197xf32>
    %v5206 = stablehlo.reshape %v5205 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5207 = stablehlo.reshape %v5206 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5208 = stablehlo.reshape %v5196 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5209 = stablehlo.dot_general %v5207, %v5208, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5210 = stablehlo.reshape %v5209 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5211 = stablehlo.reshape %v1081 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5212 = stablehlo.reshape %v5203 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5213 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5214 = stablehlo.exponential %v5211 : tensor<32x197x197xf32>
    %v5215 = stablehlo.reduce(%v5214 init: %v5213) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5216 = stablehlo.broadcast_in_dim %v5215, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5217 = stablehlo.divide %v5214, %v5216 : tensor<32x197x197xf32>
    %v5218 = stablehlo.multiply %v5217, %v5212 : tensor<32x197x197xf32>
    %v5219 = stablehlo.reduce(%v5218 init: %v5213) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5220 = stablehlo.broadcast_in_dim %v5219, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5221 = stablehlo.subtract %v5212, %v5220 : tensor<32x197x197xf32>
    %v5222 = stablehlo.multiply %v5217, %v5221 : tensor<32x197x197xf32>
    %v5223 = stablehlo.reshape %v5222 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5224 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v5225 = stablehlo.multiply %v5223, %v5224 : tensor<32x38809xf32>
    %v5226 = stablehlo.reshape %v5225 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5227 = stablehlo.reshape %v1069 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5228 = stablehlo.dot_general %v5226, %v5227, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5229 = stablehlo.reshape %v5228 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5230 = stablehlo.reshape %v1066 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5231 = stablehlo.transpose %v5230, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5232 = stablehlo.reshape %v5231 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5233 = stablehlo.reshape %v5232 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5234 = stablehlo.reshape %v5225 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5235 = stablehlo.dot_general %v5233, %v5234, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x64x197xf32>, tensor<32x197x197xf32>) -> tensor<32x64x197xf32>
    %v5236 = stablehlo.reshape %v5235 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5237 = stablehlo.reshape %v5236 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5238 = stablehlo.transpose %v5237, dims = [0, 2, 1] : (tensor<32x64x197xf32>) -> tensor<32x197x64xf32>
    %v5239 = stablehlo.reshape %v5238 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5240 = stablehlo.reshape %v5229 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5241 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5242 = stablehlo.pad %v5240, %v5241, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v5243 = stablehlo.reshape %v5242 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v5244 = stablehlo.reshape %v5239 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5245 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5246 = stablehlo.pad %v5244, %v5245, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v5247 = stablehlo.reshape %v5246 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v5248 = stablehlo.reshape %v5210 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5249 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5250 = stablehlo.pad %v5248, %v5249, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v5251 = stablehlo.reshape %v5250 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v5252 = stablehlo.reshape %v5181 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5253 = stablehlo.slice %v5252 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v5254 = stablehlo.reshape %v5253 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5255 = stablehlo.reshape %v1105 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5256 = stablehlo.transpose %v5255, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5257 = stablehlo.reshape %v5256 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5258 = stablehlo.reshape %v5254 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5259 = stablehlo.reshape %v5257 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5260 = stablehlo.dot_general %v5258, %v5259, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v5261 = stablehlo.reshape %v5260 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5262 = stablehlo.reshape %v1121 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5263 = stablehlo.transpose %v5262, dims = [0, 2, 1] : (tensor<32x197x197xf32>) -> tensor<32x197x197xf32>
    %v5264 = stablehlo.reshape %v5263 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5265 = stablehlo.reshape %v5264 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5266 = stablehlo.reshape %v5254 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5267 = stablehlo.dot_general %v5265, %v5266, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5268 = stablehlo.reshape %v5267 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5269 = stablehlo.reshape %v1114 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5270 = stablehlo.reshape %v5261 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5271 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5272 = stablehlo.exponential %v5269 : tensor<32x197x197xf32>
    %v5273 = stablehlo.reduce(%v5272 init: %v5271) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5274 = stablehlo.broadcast_in_dim %v5273, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5275 = stablehlo.divide %v5272, %v5274 : tensor<32x197x197xf32>
    %v5276 = stablehlo.multiply %v5275, %v5270 : tensor<32x197x197xf32>
    %v5277 = stablehlo.reduce(%v5276 init: %v5271) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5278 = stablehlo.broadcast_in_dim %v5277, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5279 = stablehlo.subtract %v5270, %v5278 : tensor<32x197x197xf32>
    %v5280 = stablehlo.multiply %v5275, %v5279 : tensor<32x197x197xf32>
    %v5281 = stablehlo.reshape %v5280 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5282 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v5283 = stablehlo.multiply %v5281, %v5282 : tensor<32x38809xf32>
    %v5284 = stablehlo.reshape %v5283 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5285 = stablehlo.reshape %v1102 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5286 = stablehlo.dot_general %v5284, %v5285, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5287 = stablehlo.reshape %v5286 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5288 = stablehlo.reshape %v1099 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5289 = stablehlo.transpose %v5288, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5290 = stablehlo.reshape %v5289 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5291 = stablehlo.reshape %v5290 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5292 = stablehlo.reshape %v5283 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5293 = stablehlo.dot_general %v5291, %v5292, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x64x197xf32>, tensor<32x197x197xf32>) -> tensor<32x64x197xf32>
    %v5294 = stablehlo.reshape %v5293 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5295 = stablehlo.reshape %v5294 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5296 = stablehlo.transpose %v5295, dims = [0, 2, 1] : (tensor<32x64x197xf32>) -> tensor<32x197x64xf32>
    %v5297 = stablehlo.reshape %v5296 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5298 = stablehlo.reshape %v5287 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5299 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5300 = stablehlo.pad %v5298, %v5299, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v5301 = stablehlo.reshape %v5300 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v5302 = stablehlo.reshape %v5297 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5303 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5304 = stablehlo.pad %v5302, %v5303, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v5305 = stablehlo.reshape %v5304 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v5306 = stablehlo.reshape %v5268 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5307 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5308 = stablehlo.pad %v5306, %v5307, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v5309 = stablehlo.reshape %v5308 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v5310 = stablehlo.add %v5243, %v5301 : tensor<32x37824xf32>
    %v5311 = stablehlo.add %v5247, %v5305 : tensor<32x37824xf32>
    %v5312 = stablehlo.add %v5251, %v5309 : tensor<32x37824xf32>
    %v5313 = stablehlo.reshape %v5181 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5314 = stablehlo.slice %v5313 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v5315 = stablehlo.reshape %v5314 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5316 = stablehlo.reshape %v1139 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5317 = stablehlo.transpose %v5316, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5318 = stablehlo.reshape %v5317 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5319 = stablehlo.reshape %v5315 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5320 = stablehlo.reshape %v5318 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5321 = stablehlo.dot_general %v5319, %v5320, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v5322 = stablehlo.reshape %v5321 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5323 = stablehlo.reshape %v1155 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5324 = stablehlo.transpose %v5323, dims = [0, 2, 1] : (tensor<32x197x197xf32>) -> tensor<32x197x197xf32>
    %v5325 = stablehlo.reshape %v5324 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5326 = stablehlo.reshape %v5325 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5327 = stablehlo.reshape %v5315 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5328 = stablehlo.dot_general %v5326, %v5327, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5329 = stablehlo.reshape %v5328 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5330 = stablehlo.reshape %v1148 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5331 = stablehlo.reshape %v5322 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5332 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5333 = stablehlo.exponential %v5330 : tensor<32x197x197xf32>
    %v5334 = stablehlo.reduce(%v5333 init: %v5332) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5335 = stablehlo.broadcast_in_dim %v5334, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5336 = stablehlo.divide %v5333, %v5335 : tensor<32x197x197xf32>
    %v5337 = stablehlo.multiply %v5336, %v5331 : tensor<32x197x197xf32>
    %v5338 = stablehlo.reduce(%v5337 init: %v5332) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5339 = stablehlo.broadcast_in_dim %v5338, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5340 = stablehlo.subtract %v5331, %v5339 : tensor<32x197x197xf32>
    %v5341 = stablehlo.multiply %v5336, %v5340 : tensor<32x197x197xf32>
    %v5342 = stablehlo.reshape %v5341 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5343 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v5344 = stablehlo.multiply %v5342, %v5343 : tensor<32x38809xf32>
    %v5345 = stablehlo.reshape %v5344 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5346 = stablehlo.reshape %v1136 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5347 = stablehlo.dot_general %v5345, %v5346, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5348 = stablehlo.reshape %v5347 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5349 = stablehlo.reshape %v1133 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5350 = stablehlo.transpose %v5349, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5351 = stablehlo.reshape %v5350 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5352 = stablehlo.reshape %v5351 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5353 = stablehlo.reshape %v5344 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5354 = stablehlo.dot_general %v5352, %v5353, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x64x197xf32>, tensor<32x197x197xf32>) -> tensor<32x64x197xf32>
    %v5355 = stablehlo.reshape %v5354 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5356 = stablehlo.reshape %v5355 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5357 = stablehlo.transpose %v5356, dims = [0, 2, 1] : (tensor<32x64x197xf32>) -> tensor<32x197x64xf32>
    %v5358 = stablehlo.reshape %v5357 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5359 = stablehlo.reshape %v5348 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5360 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5361 = stablehlo.pad %v5359, %v5360, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v5362 = stablehlo.reshape %v5361 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v5363 = stablehlo.reshape %v5358 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5364 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5365 = stablehlo.pad %v5363, %v5364, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v5366 = stablehlo.reshape %v5365 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v5367 = stablehlo.reshape %v5329 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5368 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5369 = stablehlo.pad %v5367, %v5368, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v5370 = stablehlo.reshape %v5369 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v5371 = stablehlo.add %v5310, %v5362 : tensor<32x37824xf32>
    %v5372 = stablehlo.add %v5311, %v5366 : tensor<32x37824xf32>
    %v5373 = stablehlo.add %v5312, %v5370 : tensor<32x37824xf32>
    %v5374 = stablehlo.reshape %v5371 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5375 = stablehlo.dot_general %v5374, %b5_Wq, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v5376 = stablehlo.reshape %v5375 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v5377 = stablehlo.reshape %v1048 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5378 = stablehlo.reshape %v5371 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5379 = stablehlo.dot_general %v5377, %v5378, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %v5380 = stablehlo.constant dense<0.003125> : tensor<192x192xf32>
    %v5381 = stablehlo.multiply %v5379, %v5380 : tensor<192x192xf32>
    %v5382 = stablehlo.subtract %b5_Wq, %v5381 : tensor<192x192xf32>
    %v5383 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5384 = stablehlo.reshape %v5371 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5385 = stablehlo.reduce(%v5384 init: %v5383) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v5386 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v5387 = stablehlo.multiply %v5385, %v5386 : tensor<192xf32>
    %v5388 = stablehlo.subtract %b5_bq, %v5387 : tensor<192xf32>
    %v5389 = stablehlo.reshape %v5372 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5390 = stablehlo.dot_general %v5389, %b5_Wk, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v5391 = stablehlo.reshape %v5390 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v5392 = stablehlo.reshape %v1048 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5393 = stablehlo.reshape %v5372 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5394 = stablehlo.dot_general %v5392, %v5393, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %v5395 = stablehlo.constant dense<0.003125> : tensor<192x192xf32>
    %v5396 = stablehlo.multiply %v5394, %v5395 : tensor<192x192xf32>
    %v5397 = stablehlo.subtract %b5_Wk, %v5396 : tensor<192x192xf32>
    %v5398 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5399 = stablehlo.reshape %v5372 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5400 = stablehlo.reduce(%v5399 init: %v5398) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v5401 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v5402 = stablehlo.multiply %v5400, %v5401 : tensor<192xf32>
    %v5403 = stablehlo.subtract %b5_bk, %v5402 : tensor<192xf32>
    %v5404 = stablehlo.reshape %v5373 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5405 = stablehlo.dot_general %v5404, %b5_Wv, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v5406 = stablehlo.reshape %v5405 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v5407 = stablehlo.reshape %v1048 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5408 = stablehlo.reshape %v5373 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5409 = stablehlo.dot_general %v5407, %v5408, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %v5410 = stablehlo.constant dense<0.003125> : tensor<192x192xf32>
    %v5411 = stablehlo.multiply %v5409, %v5410 : tensor<192x192xf32>
    %v5412 = stablehlo.subtract %b5_Wv, %v5411 : tensor<192x192xf32>
    %v5413 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5414 = stablehlo.reshape %v5373 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5415 = stablehlo.reduce(%v5414 init: %v5413) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v5416 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v5417 = stablehlo.multiply %v5415, %v5416 : tensor<192xf32>
    %v5418 = stablehlo.subtract %b5_bv, %v5417 : tensor<192xf32>
    %v5419 = stablehlo.add %v5376, %v5391 : tensor<32x37824xf32>
    %v5420 = stablehlo.add %v5419, %v5406 : tensor<32x37824xf32>
    %v5421 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5422 = stablehlo.reshape %v5420 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5423 = stablehlo.reduce(%v5422 init: %v5421) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v5424 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v5425 = stablehlo.multiply %v5423, %v5424 : tensor<192xf32>
    %v5426 = stablehlo.subtract %b5_bt1, %v5425 : tensor<192xf32>
    %v5427 = stablehlo.reshape %v1020 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5428 = stablehlo.reshape %v5420 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5429 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5430 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v5431 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v5432 = stablehlo.reduce(%v5427 init: %v5429) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5433 = stablehlo.broadcast_in_dim %v5432, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v5434 = stablehlo.divide %v5433, %v5430 : tensor<32x197x192xf32>
    %v5435 = stablehlo.subtract %v5427, %v5434 : tensor<32x197x192xf32>
    %v5436 = stablehlo.multiply %v5435, %v5435 : tensor<32x197x192xf32>
    %v5437 = stablehlo.reduce(%v5436 init: %v5429) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5438 = stablehlo.broadcast_in_dim %v5437, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v5439 = stablehlo.divide %v5438, %v5430 : tensor<32x197x192xf32>
    %v5440 = stablehlo.add %v5439, %v5431 : tensor<32x197x192xf32>
    %v5441 = stablehlo.rsqrt %v5440 : tensor<32x197x192xf32>
    %v5442 = stablehlo.multiply %v5435, %v5441 : tensor<32x197x192xf32>
    %v5443 = stablehlo.multiply %v5428, %v5442 : tensor<32x197x192xf32>
    %v5444 = stablehlo.reduce(%v5443 init: %v5429) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v5445 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v5446 = stablehlo.multiply %v5444, %v5445 : tensor<192xf32>
    %v5447 = stablehlo.subtract %b5_g1, %v5446 : tensor<192xf32>
    %v5448 = stablehlo.reshape %v5420 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5449 = stablehlo.broadcast_in_dim %b5_g1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v5450 = stablehlo.multiply %v5448, %v5449 : tensor<32x197x192xf32>
    %v5451 = stablehlo.reshape %v5450 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v5452 = stablehlo.reshape %v5451 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5453 = stablehlo.reshape %v1020 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5454 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5455 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v5456 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v5457 = stablehlo.reduce(%v5453 init: %v5454) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5458 = stablehlo.broadcast_in_dim %v5457, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v5459 = stablehlo.divide %v5458, %v5455 : tensor<32x197x192xf32>
    %v5460 = stablehlo.subtract %v5453, %v5459 : tensor<32x197x192xf32>
    %v5461 = stablehlo.multiply %v5460, %v5460 : tensor<32x197x192xf32>
    %v5462 = stablehlo.reduce(%v5461 init: %v5454) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5463 = stablehlo.broadcast_in_dim %v5462, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v5464 = stablehlo.divide %v5463, %v5455 : tensor<32x197x192xf32>
    %v5465 = stablehlo.add %v5464, %v5456 : tensor<32x197x192xf32>
    %v5466 = stablehlo.rsqrt %v5465 : tensor<32x197x192xf32>
    %v5467 = stablehlo.multiply %v5460, %v5466 : tensor<32x197x192xf32>
    %v5468 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v5469 = stablehlo.multiply %v5468, %v5452 : tensor<32x197x192xf32>
    %v5470 = stablehlo.reduce(%v5469 init: %v5454) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5471 = stablehlo.broadcast_in_dim %v5470, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v5472 = stablehlo.multiply %v5467, %v5469 : tensor<32x197x192xf32>
    %v5473 = stablehlo.reduce(%v5472 init: %v5454) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5474 = stablehlo.broadcast_in_dim %v5473, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v5475 = stablehlo.multiply %v5469, %v5455 : tensor<32x197x192xf32>
    %v5476 = stablehlo.subtract %v5475, %v5471 : tensor<32x197x192xf32>
    %v5477 = stablehlo.multiply %v5467, %v5474 : tensor<32x197x192xf32>
    %v5478 = stablehlo.subtract %v5476, %v5477 : tensor<32x197x192xf32>
    %v5479 = stablehlo.divide %v5466, %v5455 : tensor<32x197x192xf32>
    %v5480 = stablehlo.multiply %v5479, %v5478 : tensor<32x197x192xf32>
    %v5481 = stablehlo.reshape %v5480 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v5482 = stablehlo.add %v5178, %v5481 : tensor<32x37824xf32>
    %v5483 = stablehlo.reshape %v5482 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5484 = stablehlo.dot_general %v5483, %b4_Wfc2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<768x192xf32>) -> tensor<32x197x768xf32>
    %v5485 = stablehlo.reshape %v5484 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5486 = stablehlo.reshape %v1014 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5487 = stablehlo.reshape %v5482 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5488 = stablehlo.dot_general %v5486, %v5487, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<32x197x192xf32>) -> tensor<768x192xf32>
    %v5489 = stablehlo.constant dense<0.003125> : tensor<768x192xf32>
    %v5490 = stablehlo.multiply %v5488, %v5489 : tensor<768x192xf32>
    %v5491 = stablehlo.subtract %b4_Wfc2, %v5490 : tensor<768x192xf32>
    %v5492 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5493 = stablehlo.reshape %v5482 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5494 = stablehlo.reduce(%v5493 init: %v5492) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v5495 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v5496 = stablehlo.multiply %v5494, %v5495 : tensor<192xf32>
    %v5497 = stablehlo.subtract %b4_bfc2, %v5496 : tensor<192xf32>
    %v5498 = stablehlo.multiply %v1001, %v1001 : tensor<32x151296xf32>
    %v5499 = stablehlo.multiply %v5498, %v1001 : tensor<32x151296xf32>
    %v5500 = stablehlo.constant dense<0.044715> : tensor<32x151296xf32>
    %v5501 = stablehlo.multiply %v5500, %v5499 : tensor<32x151296xf32>
    %v5502 = stablehlo.add %v1001, %v5501 : tensor<32x151296xf32>
    %v5503 = stablehlo.constant dense<0.7978845608028654> : tensor<32x151296xf32>
    %v5504 = stablehlo.multiply %v5503, %v5502 : tensor<32x151296xf32>
    %v5505 = stablehlo.tanh %v5504 : tensor<32x151296xf32>
    %v5506 = stablehlo.constant dense<1.0> : tensor<32x151296xf32>
    %v5507 = stablehlo.add %v5506, %v5505 : tensor<32x151296xf32>
    %v5508 = stablehlo.constant dense<0.5> : tensor<32x151296xf32>
    %v5509 = stablehlo.multiply %v5508, %v5507 : tensor<32x151296xf32>
    %v5510 = stablehlo.multiply %v5505, %v5505 : tensor<32x151296xf32>
    %v5511 = stablehlo.subtract %v5506, %v5510 : tensor<32x151296xf32>
    %v5512 = stablehlo.multiply %v5508, %v1001 : tensor<32x151296xf32>
    %v5513 = stablehlo.multiply %v5512, %v5511 : tensor<32x151296xf32>
    %v5514 = stablehlo.constant dense<0.134145> : tensor<32x151296xf32>
    %v5515 = stablehlo.multiply %v5514, %v5498 : tensor<32x151296xf32>
    %v5516 = stablehlo.add %v5506, %v5515 : tensor<32x151296xf32>
    %v5517 = stablehlo.multiply %v5503, %v5516 : tensor<32x151296xf32>
    %v5518 = stablehlo.multiply %v5513, %v5517 : tensor<32x151296xf32>
    %v5519 = stablehlo.add %v5509, %v5518 : tensor<32x151296xf32>
    %v5520 = stablehlo.multiply %v5485, %v5519 : tensor<32x151296xf32>
    %v5521 = stablehlo.reshape %v5520 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5522 = stablehlo.dot_general %v5521, %b4_Wfc1, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<192x768xf32>) -> tensor<32x197x192xf32>
    %v5523 = stablehlo.reshape %v5522 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v5524 = stablehlo.reshape %v996 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5525 = stablehlo.reshape %v5520 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5526 = stablehlo.dot_general %v5524, %v5525, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x768xf32>) -> tensor<192x768xf32>
    %v5527 = stablehlo.constant dense<0.003125> : tensor<192x768xf32>
    %v5528 = stablehlo.multiply %v5526, %v5527 : tensor<192x768xf32>
    %v5529 = stablehlo.subtract %b4_Wfc1, %v5528 : tensor<192x768xf32>
    %v5530 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5531 = stablehlo.reshape %v5520 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5532 = stablehlo.reduce(%v5531 init: %v5530) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<768xf32>
    %v5533 = stablehlo.constant dense<0.003125> : tensor<768xf32>
    %v5534 = stablehlo.multiply %v5532, %v5533 : tensor<768xf32>
    %v5535 = stablehlo.subtract %b4_bfc1, %v5534 : tensor<768xf32>
    %v5536 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5537 = stablehlo.reshape %v5523 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5538 = stablehlo.reduce(%v5537 init: %v5536) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v5539 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v5540 = stablehlo.multiply %v5538, %v5539 : tensor<192xf32>
    %v5541 = stablehlo.subtract %b4_bt2, %v5540 : tensor<192xf32>
    %v5542 = stablehlo.reshape %v968 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5543 = stablehlo.reshape %v5523 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5544 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5545 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v5546 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v5547 = stablehlo.reduce(%v5542 init: %v5544) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5548 = stablehlo.broadcast_in_dim %v5547, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v5549 = stablehlo.divide %v5548, %v5545 : tensor<32x197x192xf32>
    %v5550 = stablehlo.subtract %v5542, %v5549 : tensor<32x197x192xf32>
    %v5551 = stablehlo.multiply %v5550, %v5550 : tensor<32x197x192xf32>
    %v5552 = stablehlo.reduce(%v5551 init: %v5544) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5553 = stablehlo.broadcast_in_dim %v5552, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v5554 = stablehlo.divide %v5553, %v5545 : tensor<32x197x192xf32>
    %v5555 = stablehlo.add %v5554, %v5546 : tensor<32x197x192xf32>
    %v5556 = stablehlo.rsqrt %v5555 : tensor<32x197x192xf32>
    %v5557 = stablehlo.multiply %v5550, %v5556 : tensor<32x197x192xf32>
    %v5558 = stablehlo.multiply %v5543, %v5557 : tensor<32x197x192xf32>
    %v5559 = stablehlo.reduce(%v5558 init: %v5544) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v5560 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v5561 = stablehlo.multiply %v5559, %v5560 : tensor<192xf32>
    %v5562 = stablehlo.subtract %b4_g2, %v5561 : tensor<192xf32>
    %v5563 = stablehlo.reshape %v5523 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5564 = stablehlo.broadcast_in_dim %b4_g2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v5565 = stablehlo.multiply %v5563, %v5564 : tensor<32x197x192xf32>
    %v5566 = stablehlo.reshape %v5565 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v5567 = stablehlo.reshape %v5566 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5568 = stablehlo.reshape %v968 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5569 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5570 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v5571 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v5572 = stablehlo.reduce(%v5568 init: %v5569) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5573 = stablehlo.broadcast_in_dim %v5572, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v5574 = stablehlo.divide %v5573, %v5570 : tensor<32x197x192xf32>
    %v5575 = stablehlo.subtract %v5568, %v5574 : tensor<32x197x192xf32>
    %v5576 = stablehlo.multiply %v5575, %v5575 : tensor<32x197x192xf32>
    %v5577 = stablehlo.reduce(%v5576 init: %v5569) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5578 = stablehlo.broadcast_in_dim %v5577, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v5579 = stablehlo.divide %v5578, %v5570 : tensor<32x197x192xf32>
    %v5580 = stablehlo.add %v5579, %v5571 : tensor<32x197x192xf32>
    %v5581 = stablehlo.rsqrt %v5580 : tensor<32x197x192xf32>
    %v5582 = stablehlo.multiply %v5575, %v5581 : tensor<32x197x192xf32>
    %v5583 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v5584 = stablehlo.multiply %v5583, %v5567 : tensor<32x197x192xf32>
    %v5585 = stablehlo.reduce(%v5584 init: %v5569) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5586 = stablehlo.broadcast_in_dim %v5585, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v5587 = stablehlo.multiply %v5582, %v5584 : tensor<32x197x192xf32>
    %v5588 = stablehlo.reduce(%v5587 init: %v5569) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5589 = stablehlo.broadcast_in_dim %v5588, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v5590 = stablehlo.multiply %v5584, %v5570 : tensor<32x197x192xf32>
    %v5591 = stablehlo.subtract %v5590, %v5586 : tensor<32x197x192xf32>
    %v5592 = stablehlo.multiply %v5582, %v5589 : tensor<32x197x192xf32>
    %v5593 = stablehlo.subtract %v5591, %v5592 : tensor<32x197x192xf32>
    %v5594 = stablehlo.divide %v5581, %v5570 : tensor<32x197x192xf32>
    %v5595 = stablehlo.multiply %v5594, %v5593 : tensor<32x197x192xf32>
    %v5596 = stablehlo.reshape %v5595 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v5597 = stablehlo.add %v5482, %v5596 : tensor<32x37824xf32>
    %v5598 = stablehlo.reshape %v5597 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5599 = stablehlo.dot_general %v5598, %b4_Wo, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v5600 = stablehlo.reshape %v5599 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v5601 = stablehlo.reshape %v962 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5602 = stablehlo.reshape %v5597 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5603 = stablehlo.dot_general %v5601, %v5602, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %v5604 = stablehlo.constant dense<0.003125> : tensor<192x192xf32>
    %v5605 = stablehlo.multiply %v5603, %v5604 : tensor<192x192xf32>
    %v5606 = stablehlo.subtract %b4_Wo, %v5605 : tensor<192x192xf32>
    %v5607 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5608 = stablehlo.reshape %v5597 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5609 = stablehlo.reduce(%v5608 init: %v5607) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v5610 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v5611 = stablehlo.multiply %v5609, %v5610 : tensor<192xf32>
    %v5612 = stablehlo.subtract %b4_bo, %v5611 : tensor<192xf32>
    %v5613 = stablehlo.reshape %v5600 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5614 = stablehlo.slice %v5613 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v5615 = stablehlo.reshape %v5614 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5616 = stablehlo.reshape %v870 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5617 = stablehlo.transpose %v5616, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5618 = stablehlo.reshape %v5617 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5619 = stablehlo.reshape %v5615 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5620 = stablehlo.reshape %v5618 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5621 = stablehlo.dot_general %v5619, %v5620, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v5622 = stablehlo.reshape %v5621 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5623 = stablehlo.reshape %v886 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5624 = stablehlo.transpose %v5623, dims = [0, 2, 1] : (tensor<32x197x197xf32>) -> tensor<32x197x197xf32>
    %v5625 = stablehlo.reshape %v5624 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5626 = stablehlo.reshape %v5625 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5627 = stablehlo.reshape %v5615 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5628 = stablehlo.dot_general %v5626, %v5627, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5629 = stablehlo.reshape %v5628 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5630 = stablehlo.reshape %v879 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5631 = stablehlo.reshape %v5622 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5632 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5633 = stablehlo.exponential %v5630 : tensor<32x197x197xf32>
    %v5634 = stablehlo.reduce(%v5633 init: %v5632) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5635 = stablehlo.broadcast_in_dim %v5634, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5636 = stablehlo.divide %v5633, %v5635 : tensor<32x197x197xf32>
    %v5637 = stablehlo.multiply %v5636, %v5631 : tensor<32x197x197xf32>
    %v5638 = stablehlo.reduce(%v5637 init: %v5632) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5639 = stablehlo.broadcast_in_dim %v5638, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5640 = stablehlo.subtract %v5631, %v5639 : tensor<32x197x197xf32>
    %v5641 = stablehlo.multiply %v5636, %v5640 : tensor<32x197x197xf32>
    %v5642 = stablehlo.reshape %v5641 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5643 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v5644 = stablehlo.multiply %v5642, %v5643 : tensor<32x38809xf32>
    %v5645 = stablehlo.reshape %v5644 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5646 = stablehlo.reshape %v867 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5647 = stablehlo.dot_general %v5645, %v5646, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5648 = stablehlo.reshape %v5647 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5649 = stablehlo.reshape %v864 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5650 = stablehlo.transpose %v5649, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5651 = stablehlo.reshape %v5650 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5652 = stablehlo.reshape %v5651 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5653 = stablehlo.reshape %v5644 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5654 = stablehlo.dot_general %v5652, %v5653, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x64x197xf32>, tensor<32x197x197xf32>) -> tensor<32x64x197xf32>
    %v5655 = stablehlo.reshape %v5654 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5656 = stablehlo.reshape %v5655 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5657 = stablehlo.transpose %v5656, dims = [0, 2, 1] : (tensor<32x64x197xf32>) -> tensor<32x197x64xf32>
    %v5658 = stablehlo.reshape %v5657 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5659 = stablehlo.reshape %v5648 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5660 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5661 = stablehlo.pad %v5659, %v5660, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v5662 = stablehlo.reshape %v5661 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v5663 = stablehlo.reshape %v5658 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5664 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5665 = stablehlo.pad %v5663, %v5664, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v5666 = stablehlo.reshape %v5665 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v5667 = stablehlo.reshape %v5629 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5668 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5669 = stablehlo.pad %v5667, %v5668, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v5670 = stablehlo.reshape %v5669 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v5671 = stablehlo.reshape %v5600 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5672 = stablehlo.slice %v5671 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v5673 = stablehlo.reshape %v5672 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5674 = stablehlo.reshape %v903 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5675 = stablehlo.transpose %v5674, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5676 = stablehlo.reshape %v5675 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5677 = stablehlo.reshape %v5673 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5678 = stablehlo.reshape %v5676 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5679 = stablehlo.dot_general %v5677, %v5678, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v5680 = stablehlo.reshape %v5679 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5681 = stablehlo.reshape %v919 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5682 = stablehlo.transpose %v5681, dims = [0, 2, 1] : (tensor<32x197x197xf32>) -> tensor<32x197x197xf32>
    %v5683 = stablehlo.reshape %v5682 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5684 = stablehlo.reshape %v5683 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5685 = stablehlo.reshape %v5673 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5686 = stablehlo.dot_general %v5684, %v5685, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5687 = stablehlo.reshape %v5686 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5688 = stablehlo.reshape %v912 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5689 = stablehlo.reshape %v5680 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5690 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5691 = stablehlo.exponential %v5688 : tensor<32x197x197xf32>
    %v5692 = stablehlo.reduce(%v5691 init: %v5690) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5693 = stablehlo.broadcast_in_dim %v5692, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5694 = stablehlo.divide %v5691, %v5693 : tensor<32x197x197xf32>
    %v5695 = stablehlo.multiply %v5694, %v5689 : tensor<32x197x197xf32>
    %v5696 = stablehlo.reduce(%v5695 init: %v5690) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5697 = stablehlo.broadcast_in_dim %v5696, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5698 = stablehlo.subtract %v5689, %v5697 : tensor<32x197x197xf32>
    %v5699 = stablehlo.multiply %v5694, %v5698 : tensor<32x197x197xf32>
    %v5700 = stablehlo.reshape %v5699 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5701 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v5702 = stablehlo.multiply %v5700, %v5701 : tensor<32x38809xf32>
    %v5703 = stablehlo.reshape %v5702 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5704 = stablehlo.reshape %v900 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5705 = stablehlo.dot_general %v5703, %v5704, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5706 = stablehlo.reshape %v5705 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5707 = stablehlo.reshape %v897 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5708 = stablehlo.transpose %v5707, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5709 = stablehlo.reshape %v5708 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5710 = stablehlo.reshape %v5709 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5711 = stablehlo.reshape %v5702 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5712 = stablehlo.dot_general %v5710, %v5711, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x64x197xf32>, tensor<32x197x197xf32>) -> tensor<32x64x197xf32>
    %v5713 = stablehlo.reshape %v5712 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5714 = stablehlo.reshape %v5713 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5715 = stablehlo.transpose %v5714, dims = [0, 2, 1] : (tensor<32x64x197xf32>) -> tensor<32x197x64xf32>
    %v5716 = stablehlo.reshape %v5715 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5717 = stablehlo.reshape %v5706 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5718 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5719 = stablehlo.pad %v5717, %v5718, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v5720 = stablehlo.reshape %v5719 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v5721 = stablehlo.reshape %v5716 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5722 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5723 = stablehlo.pad %v5721, %v5722, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v5724 = stablehlo.reshape %v5723 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v5725 = stablehlo.reshape %v5687 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5726 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5727 = stablehlo.pad %v5725, %v5726, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v5728 = stablehlo.reshape %v5727 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v5729 = stablehlo.add %v5662, %v5720 : tensor<32x37824xf32>
    %v5730 = stablehlo.add %v5666, %v5724 : tensor<32x37824xf32>
    %v5731 = stablehlo.add %v5670, %v5728 : tensor<32x37824xf32>
    %v5732 = stablehlo.reshape %v5600 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5733 = stablehlo.slice %v5732 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v5734 = stablehlo.reshape %v5733 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5735 = stablehlo.reshape %v937 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5736 = stablehlo.transpose %v5735, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5737 = stablehlo.reshape %v5736 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5738 = stablehlo.reshape %v5734 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5739 = stablehlo.reshape %v5737 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5740 = stablehlo.dot_general %v5738, %v5739, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v5741 = stablehlo.reshape %v5740 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5742 = stablehlo.reshape %v953 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5743 = stablehlo.transpose %v5742, dims = [0, 2, 1] : (tensor<32x197x197xf32>) -> tensor<32x197x197xf32>
    %v5744 = stablehlo.reshape %v5743 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5745 = stablehlo.reshape %v5744 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5746 = stablehlo.reshape %v5734 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5747 = stablehlo.dot_general %v5745, %v5746, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5748 = stablehlo.reshape %v5747 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5749 = stablehlo.reshape %v946 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5750 = stablehlo.reshape %v5741 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5751 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5752 = stablehlo.exponential %v5749 : tensor<32x197x197xf32>
    %v5753 = stablehlo.reduce(%v5752 init: %v5751) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5754 = stablehlo.broadcast_in_dim %v5753, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5755 = stablehlo.divide %v5752, %v5754 : tensor<32x197x197xf32>
    %v5756 = stablehlo.multiply %v5755, %v5750 : tensor<32x197x197xf32>
    %v5757 = stablehlo.reduce(%v5756 init: %v5751) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5758 = stablehlo.broadcast_in_dim %v5757, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5759 = stablehlo.subtract %v5750, %v5758 : tensor<32x197x197xf32>
    %v5760 = stablehlo.multiply %v5755, %v5759 : tensor<32x197x197xf32>
    %v5761 = stablehlo.reshape %v5760 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5762 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v5763 = stablehlo.multiply %v5761, %v5762 : tensor<32x38809xf32>
    %v5764 = stablehlo.reshape %v5763 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5765 = stablehlo.reshape %v934 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5766 = stablehlo.dot_general %v5764, %v5765, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5767 = stablehlo.reshape %v5766 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5768 = stablehlo.reshape %v931 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5769 = stablehlo.transpose %v5768, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5770 = stablehlo.reshape %v5769 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5771 = stablehlo.reshape %v5770 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5772 = stablehlo.reshape %v5763 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5773 = stablehlo.dot_general %v5771, %v5772, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x64x197xf32>, tensor<32x197x197xf32>) -> tensor<32x64x197xf32>
    %v5774 = stablehlo.reshape %v5773 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5775 = stablehlo.reshape %v5774 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5776 = stablehlo.transpose %v5775, dims = [0, 2, 1] : (tensor<32x64x197xf32>) -> tensor<32x197x64xf32>
    %v5777 = stablehlo.reshape %v5776 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5778 = stablehlo.reshape %v5767 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5779 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5780 = stablehlo.pad %v5778, %v5779, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v5781 = stablehlo.reshape %v5780 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v5782 = stablehlo.reshape %v5777 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5783 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5784 = stablehlo.pad %v5782, %v5783, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v5785 = stablehlo.reshape %v5784 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v5786 = stablehlo.reshape %v5748 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5787 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5788 = stablehlo.pad %v5786, %v5787, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v5789 = stablehlo.reshape %v5788 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v5790 = stablehlo.add %v5729, %v5781 : tensor<32x37824xf32>
    %v5791 = stablehlo.add %v5730, %v5785 : tensor<32x37824xf32>
    %v5792 = stablehlo.add %v5731, %v5789 : tensor<32x37824xf32>
    %v5793 = stablehlo.reshape %v5790 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5794 = stablehlo.dot_general %v5793, %b4_Wq, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v5795 = stablehlo.reshape %v5794 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v5796 = stablehlo.reshape %v846 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5797 = stablehlo.reshape %v5790 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5798 = stablehlo.dot_general %v5796, %v5797, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %v5799 = stablehlo.constant dense<0.003125> : tensor<192x192xf32>
    %v5800 = stablehlo.multiply %v5798, %v5799 : tensor<192x192xf32>
    %v5801 = stablehlo.subtract %b4_Wq, %v5800 : tensor<192x192xf32>
    %v5802 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5803 = stablehlo.reshape %v5790 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5804 = stablehlo.reduce(%v5803 init: %v5802) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v5805 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v5806 = stablehlo.multiply %v5804, %v5805 : tensor<192xf32>
    %v5807 = stablehlo.subtract %b4_bq, %v5806 : tensor<192xf32>
    %v5808 = stablehlo.reshape %v5791 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5809 = stablehlo.dot_general %v5808, %b4_Wk, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v5810 = stablehlo.reshape %v5809 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v5811 = stablehlo.reshape %v846 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5812 = stablehlo.reshape %v5791 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5813 = stablehlo.dot_general %v5811, %v5812, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %v5814 = stablehlo.constant dense<0.003125> : tensor<192x192xf32>
    %v5815 = stablehlo.multiply %v5813, %v5814 : tensor<192x192xf32>
    %v5816 = stablehlo.subtract %b4_Wk, %v5815 : tensor<192x192xf32>
    %v5817 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5818 = stablehlo.reshape %v5791 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5819 = stablehlo.reduce(%v5818 init: %v5817) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v5820 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v5821 = stablehlo.multiply %v5819, %v5820 : tensor<192xf32>
    %v5822 = stablehlo.subtract %b4_bk, %v5821 : tensor<192xf32>
    %v5823 = stablehlo.reshape %v5792 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5824 = stablehlo.dot_general %v5823, %b4_Wv, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v5825 = stablehlo.reshape %v5824 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v5826 = stablehlo.reshape %v846 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5827 = stablehlo.reshape %v5792 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5828 = stablehlo.dot_general %v5826, %v5827, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %v5829 = stablehlo.constant dense<0.003125> : tensor<192x192xf32>
    %v5830 = stablehlo.multiply %v5828, %v5829 : tensor<192x192xf32>
    %v5831 = stablehlo.subtract %b4_Wv, %v5830 : tensor<192x192xf32>
    %v5832 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5833 = stablehlo.reshape %v5792 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5834 = stablehlo.reduce(%v5833 init: %v5832) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v5835 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v5836 = stablehlo.multiply %v5834, %v5835 : tensor<192xf32>
    %v5837 = stablehlo.subtract %b4_bv, %v5836 : tensor<192xf32>
    %v5838 = stablehlo.add %v5795, %v5810 : tensor<32x37824xf32>
    %v5839 = stablehlo.add %v5838, %v5825 : tensor<32x37824xf32>
    %v5840 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5841 = stablehlo.reshape %v5839 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5842 = stablehlo.reduce(%v5841 init: %v5840) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v5843 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v5844 = stablehlo.multiply %v5842, %v5843 : tensor<192xf32>
    %v5845 = stablehlo.subtract %b4_bt1, %v5844 : tensor<192xf32>
    %v5846 = stablehlo.reshape %v818 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5847 = stablehlo.reshape %v5839 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5848 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5849 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v5850 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v5851 = stablehlo.reduce(%v5846 init: %v5848) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5852 = stablehlo.broadcast_in_dim %v5851, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v5853 = stablehlo.divide %v5852, %v5849 : tensor<32x197x192xf32>
    %v5854 = stablehlo.subtract %v5846, %v5853 : tensor<32x197x192xf32>
    %v5855 = stablehlo.multiply %v5854, %v5854 : tensor<32x197x192xf32>
    %v5856 = stablehlo.reduce(%v5855 init: %v5848) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5857 = stablehlo.broadcast_in_dim %v5856, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v5858 = stablehlo.divide %v5857, %v5849 : tensor<32x197x192xf32>
    %v5859 = stablehlo.add %v5858, %v5850 : tensor<32x197x192xf32>
    %v5860 = stablehlo.rsqrt %v5859 : tensor<32x197x192xf32>
    %v5861 = stablehlo.multiply %v5854, %v5860 : tensor<32x197x192xf32>
    %v5862 = stablehlo.multiply %v5847, %v5861 : tensor<32x197x192xf32>
    %v5863 = stablehlo.reduce(%v5862 init: %v5848) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v5864 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v5865 = stablehlo.multiply %v5863, %v5864 : tensor<192xf32>
    %v5866 = stablehlo.subtract %b4_g1, %v5865 : tensor<192xf32>
    %v5867 = stablehlo.reshape %v5839 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5868 = stablehlo.broadcast_in_dim %b4_g1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v5869 = stablehlo.multiply %v5867, %v5868 : tensor<32x197x192xf32>
    %v5870 = stablehlo.reshape %v5869 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v5871 = stablehlo.reshape %v5870 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5872 = stablehlo.reshape %v818 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5873 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5874 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v5875 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v5876 = stablehlo.reduce(%v5872 init: %v5873) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5877 = stablehlo.broadcast_in_dim %v5876, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v5878 = stablehlo.divide %v5877, %v5874 : tensor<32x197x192xf32>
    %v5879 = stablehlo.subtract %v5872, %v5878 : tensor<32x197x192xf32>
    %v5880 = stablehlo.multiply %v5879, %v5879 : tensor<32x197x192xf32>
    %v5881 = stablehlo.reduce(%v5880 init: %v5873) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5882 = stablehlo.broadcast_in_dim %v5881, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v5883 = stablehlo.divide %v5882, %v5874 : tensor<32x197x192xf32>
    %v5884 = stablehlo.add %v5883, %v5875 : tensor<32x197x192xf32>
    %v5885 = stablehlo.rsqrt %v5884 : tensor<32x197x192xf32>
    %v5886 = stablehlo.multiply %v5879, %v5885 : tensor<32x197x192xf32>
    %v5887 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v5888 = stablehlo.multiply %v5887, %v5871 : tensor<32x197x192xf32>
    %v5889 = stablehlo.reduce(%v5888 init: %v5873) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5890 = stablehlo.broadcast_in_dim %v5889, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v5891 = stablehlo.multiply %v5886, %v5888 : tensor<32x197x192xf32>
    %v5892 = stablehlo.reduce(%v5891 init: %v5873) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5893 = stablehlo.broadcast_in_dim %v5892, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v5894 = stablehlo.multiply %v5888, %v5874 : tensor<32x197x192xf32>
    %v5895 = stablehlo.subtract %v5894, %v5890 : tensor<32x197x192xf32>
    %v5896 = stablehlo.multiply %v5886, %v5893 : tensor<32x197x192xf32>
    %v5897 = stablehlo.subtract %v5895, %v5896 : tensor<32x197x192xf32>
    %v5898 = stablehlo.divide %v5885, %v5874 : tensor<32x197x192xf32>
    %v5899 = stablehlo.multiply %v5898, %v5897 : tensor<32x197x192xf32>
    %v5900 = stablehlo.reshape %v5899 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v5901 = stablehlo.add %v5597, %v5900 : tensor<32x37824xf32>
    %v5902 = stablehlo.reshape %v5901 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5903 = stablehlo.dot_general %v5902, %b3_Wfc2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<768x192xf32>) -> tensor<32x197x768xf32>
    %v5904 = stablehlo.reshape %v5903 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5905 = stablehlo.reshape %v812 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5906 = stablehlo.reshape %v5901 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5907 = stablehlo.dot_general %v5905, %v5906, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<32x197x192xf32>) -> tensor<768x192xf32>
    %v5908 = stablehlo.constant dense<0.003125> : tensor<768x192xf32>
    %v5909 = stablehlo.multiply %v5907, %v5908 : tensor<768x192xf32>
    %v5910 = stablehlo.subtract %b3_Wfc2, %v5909 : tensor<768x192xf32>
    %v5911 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5912 = stablehlo.reshape %v5901 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5913 = stablehlo.reduce(%v5912 init: %v5911) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v5914 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v5915 = stablehlo.multiply %v5913, %v5914 : tensor<192xf32>
    %v5916 = stablehlo.subtract %b3_bfc2, %v5915 : tensor<192xf32>
    %v5917 = stablehlo.multiply %v799, %v799 : tensor<32x151296xf32>
    %v5918 = stablehlo.multiply %v5917, %v799 : tensor<32x151296xf32>
    %v5919 = stablehlo.constant dense<0.044715> : tensor<32x151296xf32>
    %v5920 = stablehlo.multiply %v5919, %v5918 : tensor<32x151296xf32>
    %v5921 = stablehlo.add %v799, %v5920 : tensor<32x151296xf32>
    %v5922 = stablehlo.constant dense<0.7978845608028654> : tensor<32x151296xf32>
    %v5923 = stablehlo.multiply %v5922, %v5921 : tensor<32x151296xf32>
    %v5924 = stablehlo.tanh %v5923 : tensor<32x151296xf32>
    %v5925 = stablehlo.constant dense<1.0> : tensor<32x151296xf32>
    %v5926 = stablehlo.add %v5925, %v5924 : tensor<32x151296xf32>
    %v5927 = stablehlo.constant dense<0.5> : tensor<32x151296xf32>
    %v5928 = stablehlo.multiply %v5927, %v5926 : tensor<32x151296xf32>
    %v5929 = stablehlo.multiply %v5924, %v5924 : tensor<32x151296xf32>
    %v5930 = stablehlo.subtract %v5925, %v5929 : tensor<32x151296xf32>
    %v5931 = stablehlo.multiply %v5927, %v799 : tensor<32x151296xf32>
    %v5932 = stablehlo.multiply %v5931, %v5930 : tensor<32x151296xf32>
    %v5933 = stablehlo.constant dense<0.134145> : tensor<32x151296xf32>
    %v5934 = stablehlo.multiply %v5933, %v5917 : tensor<32x151296xf32>
    %v5935 = stablehlo.add %v5925, %v5934 : tensor<32x151296xf32>
    %v5936 = stablehlo.multiply %v5922, %v5935 : tensor<32x151296xf32>
    %v5937 = stablehlo.multiply %v5932, %v5936 : tensor<32x151296xf32>
    %v5938 = stablehlo.add %v5928, %v5937 : tensor<32x151296xf32>
    %v5939 = stablehlo.multiply %v5904, %v5938 : tensor<32x151296xf32>
    %v5940 = stablehlo.reshape %v5939 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5941 = stablehlo.dot_general %v5940, %b3_Wfc1, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<192x768xf32>) -> tensor<32x197x192xf32>
    %v5942 = stablehlo.reshape %v5941 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v5943 = stablehlo.reshape %v794 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5944 = stablehlo.reshape %v5939 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5945 = stablehlo.dot_general %v5943, %v5944, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x768xf32>) -> tensor<192x768xf32>
    %v5946 = stablehlo.constant dense<0.003125> : tensor<192x768xf32>
    %v5947 = stablehlo.multiply %v5945, %v5946 : tensor<192x768xf32>
    %v5948 = stablehlo.subtract %b3_Wfc1, %v5947 : tensor<192x768xf32>
    %v5949 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5950 = stablehlo.reshape %v5939 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5951 = stablehlo.reduce(%v5950 init: %v5949) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<768xf32>
    %v5952 = stablehlo.constant dense<0.003125> : tensor<768xf32>
    %v5953 = stablehlo.multiply %v5951, %v5952 : tensor<768xf32>
    %v5954 = stablehlo.subtract %b3_bfc1, %v5953 : tensor<768xf32>
    %v5955 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5956 = stablehlo.reshape %v5942 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5957 = stablehlo.reduce(%v5956 init: %v5955) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v5958 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v5959 = stablehlo.multiply %v5957, %v5958 : tensor<192xf32>
    %v5960 = stablehlo.subtract %b3_bt2, %v5959 : tensor<192xf32>
    %v5961 = stablehlo.reshape %v766 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5962 = stablehlo.reshape %v5942 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5963 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5964 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v5965 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v5966 = stablehlo.reduce(%v5961 init: %v5963) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5967 = stablehlo.broadcast_in_dim %v5966, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v5968 = stablehlo.divide %v5967, %v5964 : tensor<32x197x192xf32>
    %v5969 = stablehlo.subtract %v5961, %v5968 : tensor<32x197x192xf32>
    %v5970 = stablehlo.multiply %v5969, %v5969 : tensor<32x197x192xf32>
    %v5971 = stablehlo.reduce(%v5970 init: %v5963) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5972 = stablehlo.broadcast_in_dim %v5971, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v5973 = stablehlo.divide %v5972, %v5964 : tensor<32x197x192xf32>
    %v5974 = stablehlo.add %v5973, %v5965 : tensor<32x197x192xf32>
    %v5975 = stablehlo.rsqrt %v5974 : tensor<32x197x192xf32>
    %v5976 = stablehlo.multiply %v5969, %v5975 : tensor<32x197x192xf32>
    %v5977 = stablehlo.multiply %v5962, %v5976 : tensor<32x197x192xf32>
    %v5978 = stablehlo.reduce(%v5977 init: %v5963) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v5979 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v5980 = stablehlo.multiply %v5978, %v5979 : tensor<192xf32>
    %v5981 = stablehlo.subtract %b3_g2, %v5980 : tensor<192xf32>
    %v5982 = stablehlo.reshape %v5942 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5983 = stablehlo.broadcast_in_dim %b3_g2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v5984 = stablehlo.multiply %v5982, %v5983 : tensor<32x197x192xf32>
    %v5985 = stablehlo.reshape %v5984 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v5986 = stablehlo.reshape %v5985 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5987 = stablehlo.reshape %v766 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v5988 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5989 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v5990 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v5991 = stablehlo.reduce(%v5987 init: %v5988) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5992 = stablehlo.broadcast_in_dim %v5991, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v5993 = stablehlo.divide %v5992, %v5989 : tensor<32x197x192xf32>
    %v5994 = stablehlo.subtract %v5987, %v5993 : tensor<32x197x192xf32>
    %v5995 = stablehlo.multiply %v5994, %v5994 : tensor<32x197x192xf32>
    %v5996 = stablehlo.reduce(%v5995 init: %v5988) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5997 = stablehlo.broadcast_in_dim %v5996, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v5998 = stablehlo.divide %v5997, %v5989 : tensor<32x197x192xf32>
    %v5999 = stablehlo.add %v5998, %v5990 : tensor<32x197x192xf32>
    %v6000 = stablehlo.rsqrt %v5999 : tensor<32x197x192xf32>
    %v6001 = stablehlo.multiply %v5994, %v6000 : tensor<32x197x192xf32>
    %v6002 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v6003 = stablehlo.multiply %v6002, %v5986 : tensor<32x197x192xf32>
    %v6004 = stablehlo.reduce(%v6003 init: %v5988) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6005 = stablehlo.broadcast_in_dim %v6004, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v6006 = stablehlo.multiply %v6001, %v6003 : tensor<32x197x192xf32>
    %v6007 = stablehlo.reduce(%v6006 init: %v5988) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6008 = stablehlo.broadcast_in_dim %v6007, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v6009 = stablehlo.multiply %v6003, %v5989 : tensor<32x197x192xf32>
    %v6010 = stablehlo.subtract %v6009, %v6005 : tensor<32x197x192xf32>
    %v6011 = stablehlo.multiply %v6001, %v6008 : tensor<32x197x192xf32>
    %v6012 = stablehlo.subtract %v6010, %v6011 : tensor<32x197x192xf32>
    %v6013 = stablehlo.divide %v6000, %v5989 : tensor<32x197x192xf32>
    %v6014 = stablehlo.multiply %v6013, %v6012 : tensor<32x197x192xf32>
    %v6015 = stablehlo.reshape %v6014 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v6016 = stablehlo.add %v5901, %v6015 : tensor<32x37824xf32>
    %v6017 = stablehlo.reshape %v6016 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v6018 = stablehlo.dot_general %v6017, %b3_Wo, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v6019 = stablehlo.reshape %v6018 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v6020 = stablehlo.reshape %v760 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v6021 = stablehlo.reshape %v6016 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v6022 = stablehlo.dot_general %v6020, %v6021, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %v6023 = stablehlo.constant dense<0.003125> : tensor<192x192xf32>
    %v6024 = stablehlo.multiply %v6022, %v6023 : tensor<192x192xf32>
    %v6025 = stablehlo.subtract %b3_Wo, %v6024 : tensor<192x192xf32>
    %v6026 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6027 = stablehlo.reshape %v6016 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v6028 = stablehlo.reduce(%v6027 init: %v6026) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v6029 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v6030 = stablehlo.multiply %v6028, %v6029 : tensor<192xf32>
    %v6031 = stablehlo.subtract %b3_bo, %v6030 : tensor<192xf32>
    %v6032 = stablehlo.reshape %v6019 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v6033 = stablehlo.slice %v6032 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v6034 = stablehlo.reshape %v6033 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6035 = stablehlo.reshape %v668 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6036 = stablehlo.transpose %v6035, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v6037 = stablehlo.reshape %v6036 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v6038 = stablehlo.reshape %v6034 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6039 = stablehlo.reshape %v6037 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v6040 = stablehlo.dot_general %v6038, %v6039, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v6041 = stablehlo.reshape %v6040 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v6042 = stablehlo.reshape %v684 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v6043 = stablehlo.transpose %v6042, dims = [0, 2, 1] : (tensor<32x197x197xf32>) -> tensor<32x197x197xf32>
    %v6044 = stablehlo.reshape %v6043 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v6045 = stablehlo.reshape %v6044 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v6046 = stablehlo.reshape %v6034 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6047 = stablehlo.dot_general %v6045, %v6046, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v6048 = stablehlo.reshape %v6047 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6049 = stablehlo.reshape %v677 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v6050 = stablehlo.reshape %v6041 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v6051 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6052 = stablehlo.exponential %v6049 : tensor<32x197x197xf32>
    %v6053 = stablehlo.reduce(%v6052 init: %v6051) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6054 = stablehlo.broadcast_in_dim %v6053, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v6055 = stablehlo.divide %v6052, %v6054 : tensor<32x197x197xf32>
    %v6056 = stablehlo.multiply %v6055, %v6050 : tensor<32x197x197xf32>
    %v6057 = stablehlo.reduce(%v6056 init: %v6051) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6058 = stablehlo.broadcast_in_dim %v6057, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v6059 = stablehlo.subtract %v6050, %v6058 : tensor<32x197x197xf32>
    %v6060 = stablehlo.multiply %v6055, %v6059 : tensor<32x197x197xf32>
    %v6061 = stablehlo.reshape %v6060 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v6062 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v6063 = stablehlo.multiply %v6061, %v6062 : tensor<32x38809xf32>
    %v6064 = stablehlo.reshape %v6063 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v6065 = stablehlo.reshape %v665 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6066 = stablehlo.dot_general %v6064, %v6065, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v6067 = stablehlo.reshape %v6066 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6068 = stablehlo.reshape %v662 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6069 = stablehlo.transpose %v6068, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v6070 = stablehlo.reshape %v6069 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v6071 = stablehlo.reshape %v6070 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v6072 = stablehlo.reshape %v6063 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v6073 = stablehlo.dot_general %v6071, %v6072, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x64x197xf32>, tensor<32x197x197xf32>) -> tensor<32x64x197xf32>
    %v6074 = stablehlo.reshape %v6073 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v6075 = stablehlo.reshape %v6074 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v6076 = stablehlo.transpose %v6075, dims = [0, 2, 1] : (tensor<32x64x197xf32>) -> tensor<32x197x64xf32>
    %v6077 = stablehlo.reshape %v6076 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6078 = stablehlo.reshape %v6067 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6079 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6080 = stablehlo.pad %v6078, %v6079, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v6081 = stablehlo.reshape %v6080 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v6082 = stablehlo.reshape %v6077 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6083 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6084 = stablehlo.pad %v6082, %v6083, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v6085 = stablehlo.reshape %v6084 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v6086 = stablehlo.reshape %v6048 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6087 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6088 = stablehlo.pad %v6086, %v6087, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v6089 = stablehlo.reshape %v6088 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v6090 = stablehlo.reshape %v6019 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v6091 = stablehlo.slice %v6090 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v6092 = stablehlo.reshape %v6091 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6093 = stablehlo.reshape %v701 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6094 = stablehlo.transpose %v6093, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v6095 = stablehlo.reshape %v6094 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v6096 = stablehlo.reshape %v6092 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6097 = stablehlo.reshape %v6095 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v6098 = stablehlo.dot_general %v6096, %v6097, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v6099 = stablehlo.reshape %v6098 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v6100 = stablehlo.reshape %v717 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v6101 = stablehlo.transpose %v6100, dims = [0, 2, 1] : (tensor<32x197x197xf32>) -> tensor<32x197x197xf32>
    %v6102 = stablehlo.reshape %v6101 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v6103 = stablehlo.reshape %v6102 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v6104 = stablehlo.reshape %v6092 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6105 = stablehlo.dot_general %v6103, %v6104, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v6106 = stablehlo.reshape %v6105 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6107 = stablehlo.reshape %v710 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v6108 = stablehlo.reshape %v6099 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v6109 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6110 = stablehlo.exponential %v6107 : tensor<32x197x197xf32>
    %v6111 = stablehlo.reduce(%v6110 init: %v6109) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6112 = stablehlo.broadcast_in_dim %v6111, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v6113 = stablehlo.divide %v6110, %v6112 : tensor<32x197x197xf32>
    %v6114 = stablehlo.multiply %v6113, %v6108 : tensor<32x197x197xf32>
    %v6115 = stablehlo.reduce(%v6114 init: %v6109) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6116 = stablehlo.broadcast_in_dim %v6115, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v6117 = stablehlo.subtract %v6108, %v6116 : tensor<32x197x197xf32>
    %v6118 = stablehlo.multiply %v6113, %v6117 : tensor<32x197x197xf32>
    %v6119 = stablehlo.reshape %v6118 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v6120 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v6121 = stablehlo.multiply %v6119, %v6120 : tensor<32x38809xf32>
    %v6122 = stablehlo.reshape %v6121 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v6123 = stablehlo.reshape %v698 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6124 = stablehlo.dot_general %v6122, %v6123, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v6125 = stablehlo.reshape %v6124 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6126 = stablehlo.reshape %v695 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6127 = stablehlo.transpose %v6126, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v6128 = stablehlo.reshape %v6127 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v6129 = stablehlo.reshape %v6128 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v6130 = stablehlo.reshape %v6121 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v6131 = stablehlo.dot_general %v6129, %v6130, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x64x197xf32>, tensor<32x197x197xf32>) -> tensor<32x64x197xf32>
    %v6132 = stablehlo.reshape %v6131 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v6133 = stablehlo.reshape %v6132 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v6134 = stablehlo.transpose %v6133, dims = [0, 2, 1] : (tensor<32x64x197xf32>) -> tensor<32x197x64xf32>
    %v6135 = stablehlo.reshape %v6134 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6136 = stablehlo.reshape %v6125 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6137 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6138 = stablehlo.pad %v6136, %v6137, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v6139 = stablehlo.reshape %v6138 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v6140 = stablehlo.reshape %v6135 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6141 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6142 = stablehlo.pad %v6140, %v6141, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v6143 = stablehlo.reshape %v6142 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v6144 = stablehlo.reshape %v6106 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6145 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6146 = stablehlo.pad %v6144, %v6145, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v6147 = stablehlo.reshape %v6146 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v6148 = stablehlo.add %v6081, %v6139 : tensor<32x37824xf32>
    %v6149 = stablehlo.add %v6085, %v6143 : tensor<32x37824xf32>
    %v6150 = stablehlo.add %v6089, %v6147 : tensor<32x37824xf32>
    %v6151 = stablehlo.reshape %v6019 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v6152 = stablehlo.slice %v6151 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v6153 = stablehlo.reshape %v6152 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6154 = stablehlo.reshape %v735 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6155 = stablehlo.transpose %v6154, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v6156 = stablehlo.reshape %v6155 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v6157 = stablehlo.reshape %v6153 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6158 = stablehlo.reshape %v6156 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v6159 = stablehlo.dot_general %v6157, %v6158, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v6160 = stablehlo.reshape %v6159 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v6161 = stablehlo.reshape %v751 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v6162 = stablehlo.transpose %v6161, dims = [0, 2, 1] : (tensor<32x197x197xf32>) -> tensor<32x197x197xf32>
    %v6163 = stablehlo.reshape %v6162 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v6164 = stablehlo.reshape %v6163 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v6165 = stablehlo.reshape %v6153 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6166 = stablehlo.dot_general %v6164, %v6165, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v6167 = stablehlo.reshape %v6166 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6168 = stablehlo.reshape %v744 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v6169 = stablehlo.reshape %v6160 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v6170 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6171 = stablehlo.exponential %v6168 : tensor<32x197x197xf32>
    %v6172 = stablehlo.reduce(%v6171 init: %v6170) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6173 = stablehlo.broadcast_in_dim %v6172, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v6174 = stablehlo.divide %v6171, %v6173 : tensor<32x197x197xf32>
    %v6175 = stablehlo.multiply %v6174, %v6169 : tensor<32x197x197xf32>
    %v6176 = stablehlo.reduce(%v6175 init: %v6170) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6177 = stablehlo.broadcast_in_dim %v6176, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v6178 = stablehlo.subtract %v6169, %v6177 : tensor<32x197x197xf32>
    %v6179 = stablehlo.multiply %v6174, %v6178 : tensor<32x197x197xf32>
    %v6180 = stablehlo.reshape %v6179 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v6181 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v6182 = stablehlo.multiply %v6180, %v6181 : tensor<32x38809xf32>
    %v6183 = stablehlo.reshape %v6182 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v6184 = stablehlo.reshape %v732 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6185 = stablehlo.dot_general %v6183, %v6184, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v6186 = stablehlo.reshape %v6185 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6187 = stablehlo.reshape %v729 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6188 = stablehlo.transpose %v6187, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v6189 = stablehlo.reshape %v6188 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v6190 = stablehlo.reshape %v6189 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v6191 = stablehlo.reshape %v6182 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v6192 = stablehlo.dot_general %v6190, %v6191, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x64x197xf32>, tensor<32x197x197xf32>) -> tensor<32x64x197xf32>
    %v6193 = stablehlo.reshape %v6192 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v6194 = stablehlo.reshape %v6193 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v6195 = stablehlo.transpose %v6194, dims = [0, 2, 1] : (tensor<32x64x197xf32>) -> tensor<32x197x64xf32>
    %v6196 = stablehlo.reshape %v6195 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6197 = stablehlo.reshape %v6186 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6198 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6199 = stablehlo.pad %v6197, %v6198, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v6200 = stablehlo.reshape %v6199 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v6201 = stablehlo.reshape %v6196 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6202 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6203 = stablehlo.pad %v6201, %v6202, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v6204 = stablehlo.reshape %v6203 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v6205 = stablehlo.reshape %v6167 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6206 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6207 = stablehlo.pad %v6205, %v6206, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v6208 = stablehlo.reshape %v6207 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v6209 = stablehlo.add %v6148, %v6200 : tensor<32x37824xf32>
    %v6210 = stablehlo.add %v6149, %v6204 : tensor<32x37824xf32>
    %v6211 = stablehlo.add %v6150, %v6208 : tensor<32x37824xf32>
    %v6212 = stablehlo.reshape %v6209 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v6213 = stablehlo.dot_general %v6212, %b3_Wq, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v6214 = stablehlo.reshape %v6213 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v6215 = stablehlo.reshape %v644 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v6216 = stablehlo.reshape %v6209 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v6217 = stablehlo.dot_general %v6215, %v6216, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %v6218 = stablehlo.constant dense<0.003125> : tensor<192x192xf32>
    %v6219 = stablehlo.multiply %v6217, %v6218 : tensor<192x192xf32>
    %v6220 = stablehlo.subtract %b3_Wq, %v6219 : tensor<192x192xf32>
    %v6221 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6222 = stablehlo.reshape %v6209 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v6223 = stablehlo.reduce(%v6222 init: %v6221) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v6224 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v6225 = stablehlo.multiply %v6223, %v6224 : tensor<192xf32>
    %v6226 = stablehlo.subtract %b3_bq, %v6225 : tensor<192xf32>
    %v6227 = stablehlo.reshape %v6210 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v6228 = stablehlo.dot_general %v6227, %b3_Wk, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v6229 = stablehlo.reshape %v6228 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v6230 = stablehlo.reshape %v644 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v6231 = stablehlo.reshape %v6210 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v6232 = stablehlo.dot_general %v6230, %v6231, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %v6233 = stablehlo.constant dense<0.003125> : tensor<192x192xf32>
    %v6234 = stablehlo.multiply %v6232, %v6233 : tensor<192x192xf32>
    %v6235 = stablehlo.subtract %b3_Wk, %v6234 : tensor<192x192xf32>
    %v6236 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6237 = stablehlo.reshape %v6210 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v6238 = stablehlo.reduce(%v6237 init: %v6236) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v6239 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v6240 = stablehlo.multiply %v6238, %v6239 : tensor<192xf32>
    %v6241 = stablehlo.subtract %b3_bk, %v6240 : tensor<192xf32>
    %v6242 = stablehlo.reshape %v6211 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v6243 = stablehlo.dot_general %v6242, %b3_Wv, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v6244 = stablehlo.reshape %v6243 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v6245 = stablehlo.reshape %v644 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v6246 = stablehlo.reshape %v6211 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v6247 = stablehlo.dot_general %v6245, %v6246, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %v6248 = stablehlo.constant dense<0.003125> : tensor<192x192xf32>
    %v6249 = stablehlo.multiply %v6247, %v6248 : tensor<192x192xf32>
    %v6250 = stablehlo.subtract %b3_Wv, %v6249 : tensor<192x192xf32>
    %v6251 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6252 = stablehlo.reshape %v6211 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v6253 = stablehlo.reduce(%v6252 init: %v6251) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v6254 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v6255 = stablehlo.multiply %v6253, %v6254 : tensor<192xf32>
    %v6256 = stablehlo.subtract %b3_bv, %v6255 : tensor<192xf32>
    %v6257 = stablehlo.add %v6214, %v6229 : tensor<32x37824xf32>
    %v6258 = stablehlo.add %v6257, %v6244 : tensor<32x37824xf32>
    %v6259 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6260 = stablehlo.reshape %v6258 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v6261 = stablehlo.reduce(%v6260 init: %v6259) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v6262 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v6263 = stablehlo.multiply %v6261, %v6262 : tensor<192xf32>
    %v6264 = stablehlo.subtract %b3_bt1, %v6263 : tensor<192xf32>
    %v6265 = stablehlo.reshape %v616 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v6266 = stablehlo.reshape %v6258 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v6267 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6268 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v6269 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v6270 = stablehlo.reduce(%v6265 init: %v6267) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6271 = stablehlo.broadcast_in_dim %v6270, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v6272 = stablehlo.divide %v6271, %v6268 : tensor<32x197x192xf32>
    %v6273 = stablehlo.subtract %v6265, %v6272 : tensor<32x197x192xf32>
    %v6274 = stablehlo.multiply %v6273, %v6273 : tensor<32x197x192xf32>
    %v6275 = stablehlo.reduce(%v6274 init: %v6267) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6276 = stablehlo.broadcast_in_dim %v6275, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v6277 = stablehlo.divide %v6276, %v6268 : tensor<32x197x192xf32>
    %v6278 = stablehlo.add %v6277, %v6269 : tensor<32x197x192xf32>
    %v6279 = stablehlo.rsqrt %v6278 : tensor<32x197x192xf32>
    %v6280 = stablehlo.multiply %v6273, %v6279 : tensor<32x197x192xf32>
    %v6281 = stablehlo.multiply %v6266, %v6280 : tensor<32x197x192xf32>
    %v6282 = stablehlo.reduce(%v6281 init: %v6267) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v6283 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v6284 = stablehlo.multiply %v6282, %v6283 : tensor<192xf32>
    %v6285 = stablehlo.subtract %b3_g1, %v6284 : tensor<192xf32>
    %v6286 = stablehlo.reshape %v6258 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v6287 = stablehlo.broadcast_in_dim %b3_g1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v6288 = stablehlo.multiply %v6286, %v6287 : tensor<32x197x192xf32>
    %v6289 = stablehlo.reshape %v6288 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v6290 = stablehlo.reshape %v6289 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v6291 = stablehlo.reshape %v616 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v6292 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6293 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v6294 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v6295 = stablehlo.reduce(%v6291 init: %v6292) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6296 = stablehlo.broadcast_in_dim %v6295, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v6297 = stablehlo.divide %v6296, %v6293 : tensor<32x197x192xf32>
    %v6298 = stablehlo.subtract %v6291, %v6297 : tensor<32x197x192xf32>
    %v6299 = stablehlo.multiply %v6298, %v6298 : tensor<32x197x192xf32>
    %v6300 = stablehlo.reduce(%v6299 init: %v6292) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6301 = stablehlo.broadcast_in_dim %v6300, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v6302 = stablehlo.divide %v6301, %v6293 : tensor<32x197x192xf32>
    %v6303 = stablehlo.add %v6302, %v6294 : tensor<32x197x192xf32>
    %v6304 = stablehlo.rsqrt %v6303 : tensor<32x197x192xf32>
    %v6305 = stablehlo.multiply %v6298, %v6304 : tensor<32x197x192xf32>
    %v6306 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v6307 = stablehlo.multiply %v6306, %v6290 : tensor<32x197x192xf32>
    %v6308 = stablehlo.reduce(%v6307 init: %v6292) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6309 = stablehlo.broadcast_in_dim %v6308, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v6310 = stablehlo.multiply %v6305, %v6307 : tensor<32x197x192xf32>
    %v6311 = stablehlo.reduce(%v6310 init: %v6292) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6312 = stablehlo.broadcast_in_dim %v6311, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v6313 = stablehlo.multiply %v6307, %v6293 : tensor<32x197x192xf32>
    %v6314 = stablehlo.subtract %v6313, %v6309 : tensor<32x197x192xf32>
    %v6315 = stablehlo.multiply %v6305, %v6312 : tensor<32x197x192xf32>
    %v6316 = stablehlo.subtract %v6314, %v6315 : tensor<32x197x192xf32>
    %v6317 = stablehlo.divide %v6304, %v6293 : tensor<32x197x192xf32>
    %v6318 = stablehlo.multiply %v6317, %v6316 : tensor<32x197x192xf32>
    %v6319 = stablehlo.reshape %v6318 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v6320 = stablehlo.add %v6016, %v6319 : tensor<32x37824xf32>
    %v6321 = stablehlo.reshape %v6320 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v6322 = stablehlo.dot_general %v6321, %b2_Wfc2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<768x192xf32>) -> tensor<32x197x768xf32>
    %v6323 = stablehlo.reshape %v6322 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v6324 = stablehlo.reshape %v610 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6325 = stablehlo.reshape %v6320 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v6326 = stablehlo.dot_general %v6324, %v6325, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<32x197x192xf32>) -> tensor<768x192xf32>
    %v6327 = stablehlo.constant dense<0.003125> : tensor<768x192xf32>
    %v6328 = stablehlo.multiply %v6326, %v6327 : tensor<768x192xf32>
    %v6329 = stablehlo.subtract %b2_Wfc2, %v6328 : tensor<768x192xf32>
    %v6330 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6331 = stablehlo.reshape %v6320 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v6332 = stablehlo.reduce(%v6331 init: %v6330) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v6333 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v6334 = stablehlo.multiply %v6332, %v6333 : tensor<192xf32>
    %v6335 = stablehlo.subtract %b2_bfc2, %v6334 : tensor<192xf32>
    %v6336 = stablehlo.multiply %v597, %v597 : tensor<32x151296xf32>
    %v6337 = stablehlo.multiply %v6336, %v597 : tensor<32x151296xf32>
    %v6338 = stablehlo.constant dense<0.044715> : tensor<32x151296xf32>
    %v6339 = stablehlo.multiply %v6338, %v6337 : tensor<32x151296xf32>
    %v6340 = stablehlo.add %v597, %v6339 : tensor<32x151296xf32>
    %v6341 = stablehlo.constant dense<0.7978845608028654> : tensor<32x151296xf32>
    %v6342 = stablehlo.multiply %v6341, %v6340 : tensor<32x151296xf32>
    %v6343 = stablehlo.tanh %v6342 : tensor<32x151296xf32>
    %v6344 = stablehlo.constant dense<1.0> : tensor<32x151296xf32>
    %v6345 = stablehlo.add %v6344, %v6343 : tensor<32x151296xf32>
    %v6346 = stablehlo.constant dense<0.5> : tensor<32x151296xf32>
    %v6347 = stablehlo.multiply %v6346, %v6345 : tensor<32x151296xf32>
    %v6348 = stablehlo.multiply %v6343, %v6343 : tensor<32x151296xf32>
    %v6349 = stablehlo.subtract %v6344, %v6348 : tensor<32x151296xf32>
    %v6350 = stablehlo.multiply %v6346, %v597 : tensor<32x151296xf32>
    %v6351 = stablehlo.multiply %v6350, %v6349 : tensor<32x151296xf32>
    %v6352 = stablehlo.constant dense<0.134145> : tensor<32x151296xf32>
    %v6353 = stablehlo.multiply %v6352, %v6336 : tensor<32x151296xf32>
    %v6354 = stablehlo.add %v6344, %v6353 : tensor<32x151296xf32>
    %v6355 = stablehlo.multiply %v6341, %v6354 : tensor<32x151296xf32>
    %v6356 = stablehlo.multiply %v6351, %v6355 : tensor<32x151296xf32>
    %v6357 = stablehlo.add %v6347, %v6356 : tensor<32x151296xf32>
    %v6358 = stablehlo.multiply %v6323, %v6357 : tensor<32x151296xf32>
    %v6359 = stablehlo.reshape %v6358 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6360 = stablehlo.dot_general %v6359, %b2_Wfc1, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<192x768xf32>) -> tensor<32x197x192xf32>
    %v6361 = stablehlo.reshape %v6360 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v6362 = stablehlo.reshape %v592 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v6363 = stablehlo.reshape %v6358 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6364 = stablehlo.dot_general %v6362, %v6363, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x768xf32>) -> tensor<192x768xf32>
    %v6365 = stablehlo.constant dense<0.003125> : tensor<192x768xf32>
    %v6366 = stablehlo.multiply %v6364, %v6365 : tensor<192x768xf32>
    %v6367 = stablehlo.subtract %b2_Wfc1, %v6366 : tensor<192x768xf32>
    %v6368 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6369 = stablehlo.reshape %v6358 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6370 = stablehlo.reduce(%v6369 init: %v6368) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<768xf32>
    %v6371 = stablehlo.constant dense<0.003125> : tensor<768xf32>
    %v6372 = stablehlo.multiply %v6370, %v6371 : tensor<768xf32>
    %v6373 = stablehlo.subtract %b2_bfc1, %v6372 : tensor<768xf32>
    %v6374 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6375 = stablehlo.reshape %v6361 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v6376 = stablehlo.reduce(%v6375 init: %v6374) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v6377 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v6378 = stablehlo.multiply %v6376, %v6377 : tensor<192xf32>
    %v6379 = stablehlo.subtract %b2_bt2, %v6378 : tensor<192xf32>
    %v6380 = stablehlo.reshape %v564 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v6381 = stablehlo.reshape %v6361 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v6382 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6383 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v6384 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v6385 = stablehlo.reduce(%v6380 init: %v6382) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6386 = stablehlo.broadcast_in_dim %v6385, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v6387 = stablehlo.divide %v6386, %v6383 : tensor<32x197x192xf32>
    %v6388 = stablehlo.subtract %v6380, %v6387 : tensor<32x197x192xf32>
    %v6389 = stablehlo.multiply %v6388, %v6388 : tensor<32x197x192xf32>
    %v6390 = stablehlo.reduce(%v6389 init: %v6382) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6391 = stablehlo.broadcast_in_dim %v6390, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v6392 = stablehlo.divide %v6391, %v6383 : tensor<32x197x192xf32>
    %v6393 = stablehlo.add %v6392, %v6384 : tensor<32x197x192xf32>
    %v6394 = stablehlo.rsqrt %v6393 : tensor<32x197x192xf32>
    %v6395 = stablehlo.multiply %v6388, %v6394 : tensor<32x197x192xf32>
    %v6396 = stablehlo.multiply %v6381, %v6395 : tensor<32x197x192xf32>
    %v6397 = stablehlo.reduce(%v6396 init: %v6382) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v6398 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v6399 = stablehlo.multiply %v6397, %v6398 : tensor<192xf32>
    %v6400 = stablehlo.subtract %b2_g2, %v6399 : tensor<192xf32>
    %v6401 = stablehlo.reshape %v6361 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v6402 = stablehlo.broadcast_in_dim %b2_g2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v6403 = stablehlo.multiply %v6401, %v6402 : tensor<32x197x192xf32>
    %v6404 = stablehlo.reshape %v6403 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v6405 = stablehlo.reshape %v6404 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v6406 = stablehlo.reshape %v564 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v6407 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6408 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v6409 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v6410 = stablehlo.reduce(%v6406 init: %v6407) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6411 = stablehlo.broadcast_in_dim %v6410, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v6412 = stablehlo.divide %v6411, %v6408 : tensor<32x197x192xf32>
    %v6413 = stablehlo.subtract %v6406, %v6412 : tensor<32x197x192xf32>
    %v6414 = stablehlo.multiply %v6413, %v6413 : tensor<32x197x192xf32>
    %v6415 = stablehlo.reduce(%v6414 init: %v6407) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6416 = stablehlo.broadcast_in_dim %v6415, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v6417 = stablehlo.divide %v6416, %v6408 : tensor<32x197x192xf32>
    %v6418 = stablehlo.add %v6417, %v6409 : tensor<32x197x192xf32>
    %v6419 = stablehlo.rsqrt %v6418 : tensor<32x197x192xf32>
    %v6420 = stablehlo.multiply %v6413, %v6419 : tensor<32x197x192xf32>
    %v6421 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v6422 = stablehlo.multiply %v6421, %v6405 : tensor<32x197x192xf32>
    %v6423 = stablehlo.reduce(%v6422 init: %v6407) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6424 = stablehlo.broadcast_in_dim %v6423, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v6425 = stablehlo.multiply %v6420, %v6422 : tensor<32x197x192xf32>
    %v6426 = stablehlo.reduce(%v6425 init: %v6407) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6427 = stablehlo.broadcast_in_dim %v6426, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v6428 = stablehlo.multiply %v6422, %v6408 : tensor<32x197x192xf32>
    %v6429 = stablehlo.subtract %v6428, %v6424 : tensor<32x197x192xf32>
    %v6430 = stablehlo.multiply %v6420, %v6427 : tensor<32x197x192xf32>
    %v6431 = stablehlo.subtract %v6429, %v6430 : tensor<32x197x192xf32>
    %v6432 = stablehlo.divide %v6419, %v6408 : tensor<32x197x192xf32>
    %v6433 = stablehlo.multiply %v6432, %v6431 : tensor<32x197x192xf32>
    %v6434 = stablehlo.reshape %v6433 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v6435 = stablehlo.add %v6320, %v6434 : tensor<32x37824xf32>
    %v6436 = stablehlo.reshape %v6435 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v6437 = stablehlo.dot_general %v6436, %b2_Wo, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v6438 = stablehlo.reshape %v6437 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v6439 = stablehlo.reshape %v558 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v6440 = stablehlo.reshape %v6435 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v6441 = stablehlo.dot_general %v6439, %v6440, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %v6442 = stablehlo.constant dense<0.003125> : tensor<192x192xf32>
    %v6443 = stablehlo.multiply %v6441, %v6442 : tensor<192x192xf32>
    %v6444 = stablehlo.subtract %b2_Wo, %v6443 : tensor<192x192xf32>
    %v6445 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6446 = stablehlo.reshape %v6435 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v6447 = stablehlo.reduce(%v6446 init: %v6445) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v6448 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v6449 = stablehlo.multiply %v6447, %v6448 : tensor<192xf32>
    %v6450 = stablehlo.subtract %b2_bo, %v6449 : tensor<192xf32>
    %v6451 = stablehlo.reshape %v6438 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v6452 = stablehlo.slice %v6451 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v6453 = stablehlo.reshape %v6452 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6454 = stablehlo.reshape %v466 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6455 = stablehlo.transpose %v6454, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v6456 = stablehlo.reshape %v6455 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v6457 = stablehlo.reshape %v6453 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6458 = stablehlo.reshape %v6456 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v6459 = stablehlo.dot_general %v6457, %v6458, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v6460 = stablehlo.reshape %v6459 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v6461 = stablehlo.reshape %v482 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v6462 = stablehlo.transpose %v6461, dims = [0, 2, 1] : (tensor<32x197x197xf32>) -> tensor<32x197x197xf32>
    %v6463 = stablehlo.reshape %v6462 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v6464 = stablehlo.reshape %v6463 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v6465 = stablehlo.reshape %v6453 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6466 = stablehlo.dot_general %v6464, %v6465, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v6467 = stablehlo.reshape %v6466 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6468 = stablehlo.reshape %v475 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v6469 = stablehlo.reshape %v6460 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v6470 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6471 = stablehlo.exponential %v6468 : tensor<32x197x197xf32>
    %v6472 = stablehlo.reduce(%v6471 init: %v6470) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6473 = stablehlo.broadcast_in_dim %v6472, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v6474 = stablehlo.divide %v6471, %v6473 : tensor<32x197x197xf32>
    %v6475 = stablehlo.multiply %v6474, %v6469 : tensor<32x197x197xf32>
    %v6476 = stablehlo.reduce(%v6475 init: %v6470) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6477 = stablehlo.broadcast_in_dim %v6476, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v6478 = stablehlo.subtract %v6469, %v6477 : tensor<32x197x197xf32>
    %v6479 = stablehlo.multiply %v6474, %v6478 : tensor<32x197x197xf32>
    %v6480 = stablehlo.reshape %v6479 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v6481 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v6482 = stablehlo.multiply %v6480, %v6481 : tensor<32x38809xf32>
    %v6483 = stablehlo.reshape %v6482 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v6484 = stablehlo.reshape %v463 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6485 = stablehlo.dot_general %v6483, %v6484, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v6486 = stablehlo.reshape %v6485 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6487 = stablehlo.reshape %v460 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6488 = stablehlo.transpose %v6487, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v6489 = stablehlo.reshape %v6488 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v6490 = stablehlo.reshape %v6489 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v6491 = stablehlo.reshape %v6482 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v6492 = stablehlo.dot_general %v6490, %v6491, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x64x197xf32>, tensor<32x197x197xf32>) -> tensor<32x64x197xf32>
    %v6493 = stablehlo.reshape %v6492 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v6494 = stablehlo.reshape %v6493 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v6495 = stablehlo.transpose %v6494, dims = [0, 2, 1] : (tensor<32x64x197xf32>) -> tensor<32x197x64xf32>
    %v6496 = stablehlo.reshape %v6495 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6497 = stablehlo.reshape %v6486 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6498 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6499 = stablehlo.pad %v6497, %v6498, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v6500 = stablehlo.reshape %v6499 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v6501 = stablehlo.reshape %v6496 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6502 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6503 = stablehlo.pad %v6501, %v6502, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v6504 = stablehlo.reshape %v6503 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v6505 = stablehlo.reshape %v6467 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6506 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6507 = stablehlo.pad %v6505, %v6506, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v6508 = stablehlo.reshape %v6507 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v6509 = stablehlo.reshape %v6438 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v6510 = stablehlo.slice %v6509 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v6511 = stablehlo.reshape %v6510 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6512 = stablehlo.reshape %v499 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6513 = stablehlo.transpose %v6512, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v6514 = stablehlo.reshape %v6513 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v6515 = stablehlo.reshape %v6511 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6516 = stablehlo.reshape %v6514 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v6517 = stablehlo.dot_general %v6515, %v6516, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v6518 = stablehlo.reshape %v6517 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v6519 = stablehlo.reshape %v515 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v6520 = stablehlo.transpose %v6519, dims = [0, 2, 1] : (tensor<32x197x197xf32>) -> tensor<32x197x197xf32>
    %v6521 = stablehlo.reshape %v6520 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v6522 = stablehlo.reshape %v6521 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v6523 = stablehlo.reshape %v6511 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6524 = stablehlo.dot_general %v6522, %v6523, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v6525 = stablehlo.reshape %v6524 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6526 = stablehlo.reshape %v508 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v6527 = stablehlo.reshape %v6518 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v6528 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6529 = stablehlo.exponential %v6526 : tensor<32x197x197xf32>
    %v6530 = stablehlo.reduce(%v6529 init: %v6528) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6531 = stablehlo.broadcast_in_dim %v6530, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v6532 = stablehlo.divide %v6529, %v6531 : tensor<32x197x197xf32>
    %v6533 = stablehlo.multiply %v6532, %v6527 : tensor<32x197x197xf32>
    %v6534 = stablehlo.reduce(%v6533 init: %v6528) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6535 = stablehlo.broadcast_in_dim %v6534, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v6536 = stablehlo.subtract %v6527, %v6535 : tensor<32x197x197xf32>
    %v6537 = stablehlo.multiply %v6532, %v6536 : tensor<32x197x197xf32>
    %v6538 = stablehlo.reshape %v6537 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v6539 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v6540 = stablehlo.multiply %v6538, %v6539 : tensor<32x38809xf32>
    %v6541 = stablehlo.reshape %v6540 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v6542 = stablehlo.reshape %v496 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6543 = stablehlo.dot_general %v6541, %v6542, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v6544 = stablehlo.reshape %v6543 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6545 = stablehlo.reshape %v493 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6546 = stablehlo.transpose %v6545, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v6547 = stablehlo.reshape %v6546 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v6548 = stablehlo.reshape %v6547 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v6549 = stablehlo.reshape %v6540 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v6550 = stablehlo.dot_general %v6548, %v6549, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x64x197xf32>, tensor<32x197x197xf32>) -> tensor<32x64x197xf32>
    %v6551 = stablehlo.reshape %v6550 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v6552 = stablehlo.reshape %v6551 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v6553 = stablehlo.transpose %v6552, dims = [0, 2, 1] : (tensor<32x64x197xf32>) -> tensor<32x197x64xf32>
    %v6554 = stablehlo.reshape %v6553 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6555 = stablehlo.reshape %v6544 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6556 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6557 = stablehlo.pad %v6555, %v6556, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v6558 = stablehlo.reshape %v6557 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v6559 = stablehlo.reshape %v6554 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6560 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6561 = stablehlo.pad %v6559, %v6560, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v6562 = stablehlo.reshape %v6561 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v6563 = stablehlo.reshape %v6525 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6564 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6565 = stablehlo.pad %v6563, %v6564, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v6566 = stablehlo.reshape %v6565 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v6567 = stablehlo.add %v6500, %v6558 : tensor<32x37824xf32>
    %v6568 = stablehlo.add %v6504, %v6562 : tensor<32x37824xf32>
    %v6569 = stablehlo.add %v6508, %v6566 : tensor<32x37824xf32>
    %v6570 = stablehlo.reshape %v6438 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v6571 = stablehlo.slice %v6570 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v6572 = stablehlo.reshape %v6571 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6573 = stablehlo.reshape %v533 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6574 = stablehlo.transpose %v6573, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v6575 = stablehlo.reshape %v6574 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v6576 = stablehlo.reshape %v6572 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6577 = stablehlo.reshape %v6575 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v6578 = stablehlo.dot_general %v6576, %v6577, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v6579 = stablehlo.reshape %v6578 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v6580 = stablehlo.reshape %v549 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v6581 = stablehlo.transpose %v6580, dims = [0, 2, 1] : (tensor<32x197x197xf32>) -> tensor<32x197x197xf32>
    %v6582 = stablehlo.reshape %v6581 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v6583 = stablehlo.reshape %v6582 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v6584 = stablehlo.reshape %v6572 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6585 = stablehlo.dot_general %v6583, %v6584, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v6586 = stablehlo.reshape %v6585 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6587 = stablehlo.reshape %v542 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v6588 = stablehlo.reshape %v6579 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v6589 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6590 = stablehlo.exponential %v6587 : tensor<32x197x197xf32>
    %v6591 = stablehlo.reduce(%v6590 init: %v6589) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6592 = stablehlo.broadcast_in_dim %v6591, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v6593 = stablehlo.divide %v6590, %v6592 : tensor<32x197x197xf32>
    %v6594 = stablehlo.multiply %v6593, %v6588 : tensor<32x197x197xf32>
    %v6595 = stablehlo.reduce(%v6594 init: %v6589) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6596 = stablehlo.broadcast_in_dim %v6595, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v6597 = stablehlo.subtract %v6588, %v6596 : tensor<32x197x197xf32>
    %v6598 = stablehlo.multiply %v6593, %v6597 : tensor<32x197x197xf32>
    %v6599 = stablehlo.reshape %v6598 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v6600 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v6601 = stablehlo.multiply %v6599, %v6600 : tensor<32x38809xf32>
    %v6602 = stablehlo.reshape %v6601 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v6603 = stablehlo.reshape %v530 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6604 = stablehlo.dot_general %v6602, %v6603, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v6605 = stablehlo.reshape %v6604 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6606 = stablehlo.reshape %v527 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6607 = stablehlo.transpose %v6606, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v6608 = stablehlo.reshape %v6607 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v6609 = stablehlo.reshape %v6608 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v6610 = stablehlo.reshape %v6601 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v6611 = stablehlo.dot_general %v6609, %v6610, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x64x197xf32>, tensor<32x197x197xf32>) -> tensor<32x64x197xf32>
    %v6612 = stablehlo.reshape %v6611 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v6613 = stablehlo.reshape %v6612 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v6614 = stablehlo.transpose %v6613, dims = [0, 2, 1] : (tensor<32x64x197xf32>) -> tensor<32x197x64xf32>
    %v6615 = stablehlo.reshape %v6614 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6616 = stablehlo.reshape %v6605 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6617 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6618 = stablehlo.pad %v6616, %v6617, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v6619 = stablehlo.reshape %v6618 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v6620 = stablehlo.reshape %v6615 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6621 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6622 = stablehlo.pad %v6620, %v6621, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v6623 = stablehlo.reshape %v6622 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v6624 = stablehlo.reshape %v6586 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6625 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6626 = stablehlo.pad %v6624, %v6625, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v6627 = stablehlo.reshape %v6626 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v6628 = stablehlo.add %v6567, %v6619 : tensor<32x37824xf32>
    %v6629 = stablehlo.add %v6568, %v6623 : tensor<32x37824xf32>
    %v6630 = stablehlo.add %v6569, %v6627 : tensor<32x37824xf32>
    %v6631 = stablehlo.reshape %v6628 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v6632 = stablehlo.dot_general %v6631, %b2_Wq, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v6633 = stablehlo.reshape %v6632 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v6634 = stablehlo.reshape %v442 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v6635 = stablehlo.reshape %v6628 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v6636 = stablehlo.dot_general %v6634, %v6635, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %v6637 = stablehlo.constant dense<0.003125> : tensor<192x192xf32>
    %v6638 = stablehlo.multiply %v6636, %v6637 : tensor<192x192xf32>
    %v6639 = stablehlo.subtract %b2_Wq, %v6638 : tensor<192x192xf32>
    %v6640 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6641 = stablehlo.reshape %v6628 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v6642 = stablehlo.reduce(%v6641 init: %v6640) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v6643 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v6644 = stablehlo.multiply %v6642, %v6643 : tensor<192xf32>
    %v6645 = stablehlo.subtract %b2_bq, %v6644 : tensor<192xf32>
    %v6646 = stablehlo.reshape %v6629 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v6647 = stablehlo.dot_general %v6646, %b2_Wk, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v6648 = stablehlo.reshape %v6647 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v6649 = stablehlo.reshape %v442 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v6650 = stablehlo.reshape %v6629 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v6651 = stablehlo.dot_general %v6649, %v6650, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %v6652 = stablehlo.constant dense<0.003125> : tensor<192x192xf32>
    %v6653 = stablehlo.multiply %v6651, %v6652 : tensor<192x192xf32>
    %v6654 = stablehlo.subtract %b2_Wk, %v6653 : tensor<192x192xf32>
    %v6655 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6656 = stablehlo.reshape %v6629 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v6657 = stablehlo.reduce(%v6656 init: %v6655) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v6658 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v6659 = stablehlo.multiply %v6657, %v6658 : tensor<192xf32>
    %v6660 = stablehlo.subtract %b2_bk, %v6659 : tensor<192xf32>
    %v6661 = stablehlo.reshape %v6630 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v6662 = stablehlo.dot_general %v6661, %b2_Wv, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v6663 = stablehlo.reshape %v6662 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v6664 = stablehlo.reshape %v442 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v6665 = stablehlo.reshape %v6630 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v6666 = stablehlo.dot_general %v6664, %v6665, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %v6667 = stablehlo.constant dense<0.003125> : tensor<192x192xf32>
    %v6668 = stablehlo.multiply %v6666, %v6667 : tensor<192x192xf32>
    %v6669 = stablehlo.subtract %b2_Wv, %v6668 : tensor<192x192xf32>
    %v6670 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6671 = stablehlo.reshape %v6630 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v6672 = stablehlo.reduce(%v6671 init: %v6670) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v6673 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v6674 = stablehlo.multiply %v6672, %v6673 : tensor<192xf32>
    %v6675 = stablehlo.subtract %b2_bv, %v6674 : tensor<192xf32>
    %v6676 = stablehlo.add %v6633, %v6648 : tensor<32x37824xf32>
    %v6677 = stablehlo.add %v6676, %v6663 : tensor<32x37824xf32>
    %v6678 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6679 = stablehlo.reshape %v6677 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v6680 = stablehlo.reduce(%v6679 init: %v6678) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v6681 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v6682 = stablehlo.multiply %v6680, %v6681 : tensor<192xf32>
    %v6683 = stablehlo.subtract %b2_bt1, %v6682 : tensor<192xf32>
    %v6684 = stablehlo.reshape %v414 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v6685 = stablehlo.reshape %v6677 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v6686 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6687 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v6688 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v6689 = stablehlo.reduce(%v6684 init: %v6686) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6690 = stablehlo.broadcast_in_dim %v6689, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v6691 = stablehlo.divide %v6690, %v6687 : tensor<32x197x192xf32>
    %v6692 = stablehlo.subtract %v6684, %v6691 : tensor<32x197x192xf32>
    %v6693 = stablehlo.multiply %v6692, %v6692 : tensor<32x197x192xf32>
    %v6694 = stablehlo.reduce(%v6693 init: %v6686) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6695 = stablehlo.broadcast_in_dim %v6694, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v6696 = stablehlo.divide %v6695, %v6687 : tensor<32x197x192xf32>
    %v6697 = stablehlo.add %v6696, %v6688 : tensor<32x197x192xf32>
    %v6698 = stablehlo.rsqrt %v6697 : tensor<32x197x192xf32>
    %v6699 = stablehlo.multiply %v6692, %v6698 : tensor<32x197x192xf32>
    %v6700 = stablehlo.multiply %v6685, %v6699 : tensor<32x197x192xf32>
    %v6701 = stablehlo.reduce(%v6700 init: %v6686) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v6702 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v6703 = stablehlo.multiply %v6701, %v6702 : tensor<192xf32>
    %v6704 = stablehlo.subtract %b2_g1, %v6703 : tensor<192xf32>
    %v6705 = stablehlo.reshape %v6677 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v6706 = stablehlo.broadcast_in_dim %b2_g1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v6707 = stablehlo.multiply %v6705, %v6706 : tensor<32x197x192xf32>
    %v6708 = stablehlo.reshape %v6707 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v6709 = stablehlo.reshape %v6708 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v6710 = stablehlo.reshape %v414 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v6711 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6712 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v6713 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v6714 = stablehlo.reduce(%v6710 init: %v6711) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6715 = stablehlo.broadcast_in_dim %v6714, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v6716 = stablehlo.divide %v6715, %v6712 : tensor<32x197x192xf32>
    %v6717 = stablehlo.subtract %v6710, %v6716 : tensor<32x197x192xf32>
    %v6718 = stablehlo.multiply %v6717, %v6717 : tensor<32x197x192xf32>
    %v6719 = stablehlo.reduce(%v6718 init: %v6711) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6720 = stablehlo.broadcast_in_dim %v6719, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v6721 = stablehlo.divide %v6720, %v6712 : tensor<32x197x192xf32>
    %v6722 = stablehlo.add %v6721, %v6713 : tensor<32x197x192xf32>
    %v6723 = stablehlo.rsqrt %v6722 : tensor<32x197x192xf32>
    %v6724 = stablehlo.multiply %v6717, %v6723 : tensor<32x197x192xf32>
    %v6725 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v6726 = stablehlo.multiply %v6725, %v6709 : tensor<32x197x192xf32>
    %v6727 = stablehlo.reduce(%v6726 init: %v6711) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6728 = stablehlo.broadcast_in_dim %v6727, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v6729 = stablehlo.multiply %v6724, %v6726 : tensor<32x197x192xf32>
    %v6730 = stablehlo.reduce(%v6729 init: %v6711) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6731 = stablehlo.broadcast_in_dim %v6730, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v6732 = stablehlo.multiply %v6726, %v6712 : tensor<32x197x192xf32>
    %v6733 = stablehlo.subtract %v6732, %v6728 : tensor<32x197x192xf32>
    %v6734 = stablehlo.multiply %v6724, %v6731 : tensor<32x197x192xf32>
    %v6735 = stablehlo.subtract %v6733, %v6734 : tensor<32x197x192xf32>
    %v6736 = stablehlo.divide %v6723, %v6712 : tensor<32x197x192xf32>
    %v6737 = stablehlo.multiply %v6736, %v6735 : tensor<32x197x192xf32>
    %v6738 = stablehlo.reshape %v6737 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v6739 = stablehlo.add %v6435, %v6738 : tensor<32x37824xf32>
    %v6740 = stablehlo.reshape %v6739 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v6741 = stablehlo.dot_general %v6740, %b1_Wfc2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<768x192xf32>) -> tensor<32x197x768xf32>
    %v6742 = stablehlo.reshape %v6741 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v6743 = stablehlo.reshape %v408 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6744 = stablehlo.reshape %v6739 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v6745 = stablehlo.dot_general %v6743, %v6744, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<32x197x192xf32>) -> tensor<768x192xf32>
    %v6746 = stablehlo.constant dense<0.003125> : tensor<768x192xf32>
    %v6747 = stablehlo.multiply %v6745, %v6746 : tensor<768x192xf32>
    %v6748 = stablehlo.subtract %b1_Wfc2, %v6747 : tensor<768x192xf32>
    %v6749 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6750 = stablehlo.reshape %v6739 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v6751 = stablehlo.reduce(%v6750 init: %v6749) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v6752 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v6753 = stablehlo.multiply %v6751, %v6752 : tensor<192xf32>
    %v6754 = stablehlo.subtract %b1_bfc2, %v6753 : tensor<192xf32>
    %v6755 = stablehlo.multiply %v395, %v395 : tensor<32x151296xf32>
    %v6756 = stablehlo.multiply %v6755, %v395 : tensor<32x151296xf32>
    %v6757 = stablehlo.constant dense<0.044715> : tensor<32x151296xf32>
    %v6758 = stablehlo.multiply %v6757, %v6756 : tensor<32x151296xf32>
    %v6759 = stablehlo.add %v395, %v6758 : tensor<32x151296xf32>
    %v6760 = stablehlo.constant dense<0.7978845608028654> : tensor<32x151296xf32>
    %v6761 = stablehlo.multiply %v6760, %v6759 : tensor<32x151296xf32>
    %v6762 = stablehlo.tanh %v6761 : tensor<32x151296xf32>
    %v6763 = stablehlo.constant dense<1.0> : tensor<32x151296xf32>
    %v6764 = stablehlo.add %v6763, %v6762 : tensor<32x151296xf32>
    %v6765 = stablehlo.constant dense<0.5> : tensor<32x151296xf32>
    %v6766 = stablehlo.multiply %v6765, %v6764 : tensor<32x151296xf32>
    %v6767 = stablehlo.multiply %v6762, %v6762 : tensor<32x151296xf32>
    %v6768 = stablehlo.subtract %v6763, %v6767 : tensor<32x151296xf32>
    %v6769 = stablehlo.multiply %v6765, %v395 : tensor<32x151296xf32>
    %v6770 = stablehlo.multiply %v6769, %v6768 : tensor<32x151296xf32>
    %v6771 = stablehlo.constant dense<0.134145> : tensor<32x151296xf32>
    %v6772 = stablehlo.multiply %v6771, %v6755 : tensor<32x151296xf32>
    %v6773 = stablehlo.add %v6763, %v6772 : tensor<32x151296xf32>
    %v6774 = stablehlo.multiply %v6760, %v6773 : tensor<32x151296xf32>
    %v6775 = stablehlo.multiply %v6770, %v6774 : tensor<32x151296xf32>
    %v6776 = stablehlo.add %v6766, %v6775 : tensor<32x151296xf32>
    %v6777 = stablehlo.multiply %v6742, %v6776 : tensor<32x151296xf32>
    %v6778 = stablehlo.reshape %v6777 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6779 = stablehlo.dot_general %v6778, %b1_Wfc1, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<192x768xf32>) -> tensor<32x197x192xf32>
    %v6780 = stablehlo.reshape %v6779 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v6781 = stablehlo.reshape %v390 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v6782 = stablehlo.reshape %v6777 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6783 = stablehlo.dot_general %v6781, %v6782, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x768xf32>) -> tensor<192x768xf32>
    %v6784 = stablehlo.constant dense<0.003125> : tensor<192x768xf32>
    %v6785 = stablehlo.multiply %v6783, %v6784 : tensor<192x768xf32>
    %v6786 = stablehlo.subtract %b1_Wfc1, %v6785 : tensor<192x768xf32>
    %v6787 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6788 = stablehlo.reshape %v6777 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6789 = stablehlo.reduce(%v6788 init: %v6787) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<768xf32>
    %v6790 = stablehlo.constant dense<0.003125> : tensor<768xf32>
    %v6791 = stablehlo.multiply %v6789, %v6790 : tensor<768xf32>
    %v6792 = stablehlo.subtract %b1_bfc1, %v6791 : tensor<768xf32>
    %v6793 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6794 = stablehlo.reshape %v6780 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v6795 = stablehlo.reduce(%v6794 init: %v6793) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v6796 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v6797 = stablehlo.multiply %v6795, %v6796 : tensor<192xf32>
    %v6798 = stablehlo.subtract %b1_bt2, %v6797 : tensor<192xf32>
    %v6799 = stablehlo.reshape %v362 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v6800 = stablehlo.reshape %v6780 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v6801 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6802 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v6803 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v6804 = stablehlo.reduce(%v6799 init: %v6801) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6805 = stablehlo.broadcast_in_dim %v6804, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v6806 = stablehlo.divide %v6805, %v6802 : tensor<32x197x192xf32>
    %v6807 = stablehlo.subtract %v6799, %v6806 : tensor<32x197x192xf32>
    %v6808 = stablehlo.multiply %v6807, %v6807 : tensor<32x197x192xf32>
    %v6809 = stablehlo.reduce(%v6808 init: %v6801) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6810 = stablehlo.broadcast_in_dim %v6809, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v6811 = stablehlo.divide %v6810, %v6802 : tensor<32x197x192xf32>
    %v6812 = stablehlo.add %v6811, %v6803 : tensor<32x197x192xf32>
    %v6813 = stablehlo.rsqrt %v6812 : tensor<32x197x192xf32>
    %v6814 = stablehlo.multiply %v6807, %v6813 : tensor<32x197x192xf32>
    %v6815 = stablehlo.multiply %v6800, %v6814 : tensor<32x197x192xf32>
    %v6816 = stablehlo.reduce(%v6815 init: %v6801) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v6817 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v6818 = stablehlo.multiply %v6816, %v6817 : tensor<192xf32>
    %v6819 = stablehlo.subtract %b1_g2, %v6818 : tensor<192xf32>
    %v6820 = stablehlo.reshape %v6780 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v6821 = stablehlo.broadcast_in_dim %b1_g2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v6822 = stablehlo.multiply %v6820, %v6821 : tensor<32x197x192xf32>
    %v6823 = stablehlo.reshape %v6822 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v6824 = stablehlo.reshape %v6823 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v6825 = stablehlo.reshape %v362 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v6826 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6827 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v6828 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v6829 = stablehlo.reduce(%v6825 init: %v6826) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6830 = stablehlo.broadcast_in_dim %v6829, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v6831 = stablehlo.divide %v6830, %v6827 : tensor<32x197x192xf32>
    %v6832 = stablehlo.subtract %v6825, %v6831 : tensor<32x197x192xf32>
    %v6833 = stablehlo.multiply %v6832, %v6832 : tensor<32x197x192xf32>
    %v6834 = stablehlo.reduce(%v6833 init: %v6826) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6835 = stablehlo.broadcast_in_dim %v6834, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v6836 = stablehlo.divide %v6835, %v6827 : tensor<32x197x192xf32>
    %v6837 = stablehlo.add %v6836, %v6828 : tensor<32x197x192xf32>
    %v6838 = stablehlo.rsqrt %v6837 : tensor<32x197x192xf32>
    %v6839 = stablehlo.multiply %v6832, %v6838 : tensor<32x197x192xf32>
    %v6840 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v6841 = stablehlo.multiply %v6840, %v6824 : tensor<32x197x192xf32>
    %v6842 = stablehlo.reduce(%v6841 init: %v6826) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6843 = stablehlo.broadcast_in_dim %v6842, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v6844 = stablehlo.multiply %v6839, %v6841 : tensor<32x197x192xf32>
    %v6845 = stablehlo.reduce(%v6844 init: %v6826) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6846 = stablehlo.broadcast_in_dim %v6845, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v6847 = stablehlo.multiply %v6841, %v6827 : tensor<32x197x192xf32>
    %v6848 = stablehlo.subtract %v6847, %v6843 : tensor<32x197x192xf32>
    %v6849 = stablehlo.multiply %v6839, %v6846 : tensor<32x197x192xf32>
    %v6850 = stablehlo.subtract %v6848, %v6849 : tensor<32x197x192xf32>
    %v6851 = stablehlo.divide %v6838, %v6827 : tensor<32x197x192xf32>
    %v6852 = stablehlo.multiply %v6851, %v6850 : tensor<32x197x192xf32>
    %v6853 = stablehlo.reshape %v6852 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v6854 = stablehlo.add %v6739, %v6853 : tensor<32x37824xf32>
    %v6855 = stablehlo.reshape %v6854 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v6856 = stablehlo.dot_general %v6855, %b1_Wo, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v6857 = stablehlo.reshape %v6856 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v6858 = stablehlo.reshape %v356 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v6859 = stablehlo.reshape %v6854 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v6860 = stablehlo.dot_general %v6858, %v6859, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %v6861 = stablehlo.constant dense<0.003125> : tensor<192x192xf32>
    %v6862 = stablehlo.multiply %v6860, %v6861 : tensor<192x192xf32>
    %v6863 = stablehlo.subtract %b1_Wo, %v6862 : tensor<192x192xf32>
    %v6864 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6865 = stablehlo.reshape %v6854 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v6866 = stablehlo.reduce(%v6865 init: %v6864) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v6867 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v6868 = stablehlo.multiply %v6866, %v6867 : tensor<192xf32>
    %v6869 = stablehlo.subtract %b1_bo, %v6868 : tensor<192xf32>
    %v6870 = stablehlo.reshape %v6857 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v6871 = stablehlo.slice %v6870 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v6872 = stablehlo.reshape %v6871 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6873 = stablehlo.reshape %v264 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6874 = stablehlo.transpose %v6873, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v6875 = stablehlo.reshape %v6874 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v6876 = stablehlo.reshape %v6872 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6877 = stablehlo.reshape %v6875 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v6878 = stablehlo.dot_general %v6876, %v6877, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v6879 = stablehlo.reshape %v6878 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v6880 = stablehlo.reshape %v280 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v6881 = stablehlo.transpose %v6880, dims = [0, 2, 1] : (tensor<32x197x197xf32>) -> tensor<32x197x197xf32>
    %v6882 = stablehlo.reshape %v6881 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v6883 = stablehlo.reshape %v6882 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v6884 = stablehlo.reshape %v6872 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6885 = stablehlo.dot_general %v6883, %v6884, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v6886 = stablehlo.reshape %v6885 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6887 = stablehlo.reshape %v273 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v6888 = stablehlo.reshape %v6879 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v6889 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6890 = stablehlo.exponential %v6887 : tensor<32x197x197xf32>
    %v6891 = stablehlo.reduce(%v6890 init: %v6889) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6892 = stablehlo.broadcast_in_dim %v6891, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v6893 = stablehlo.divide %v6890, %v6892 : tensor<32x197x197xf32>
    %v6894 = stablehlo.multiply %v6893, %v6888 : tensor<32x197x197xf32>
    %v6895 = stablehlo.reduce(%v6894 init: %v6889) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6896 = stablehlo.broadcast_in_dim %v6895, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v6897 = stablehlo.subtract %v6888, %v6896 : tensor<32x197x197xf32>
    %v6898 = stablehlo.multiply %v6893, %v6897 : tensor<32x197x197xf32>
    %v6899 = stablehlo.reshape %v6898 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v6900 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v6901 = stablehlo.multiply %v6899, %v6900 : tensor<32x38809xf32>
    %v6902 = stablehlo.reshape %v6901 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v6903 = stablehlo.reshape %v261 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6904 = stablehlo.dot_general %v6902, %v6903, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v6905 = stablehlo.reshape %v6904 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6906 = stablehlo.reshape %v258 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6907 = stablehlo.transpose %v6906, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v6908 = stablehlo.reshape %v6907 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v6909 = stablehlo.reshape %v6908 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v6910 = stablehlo.reshape %v6901 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v6911 = stablehlo.dot_general %v6909, %v6910, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x64x197xf32>, tensor<32x197x197xf32>) -> tensor<32x64x197xf32>
    %v6912 = stablehlo.reshape %v6911 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v6913 = stablehlo.reshape %v6912 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v6914 = stablehlo.transpose %v6913, dims = [0, 2, 1] : (tensor<32x64x197xf32>) -> tensor<32x197x64xf32>
    %v6915 = stablehlo.reshape %v6914 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6916 = stablehlo.reshape %v6905 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6917 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6918 = stablehlo.pad %v6916, %v6917, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v6919 = stablehlo.reshape %v6918 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v6920 = stablehlo.reshape %v6915 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6921 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6922 = stablehlo.pad %v6920, %v6921, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v6923 = stablehlo.reshape %v6922 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v6924 = stablehlo.reshape %v6886 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6925 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6926 = stablehlo.pad %v6924, %v6925, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v6927 = stablehlo.reshape %v6926 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v6928 = stablehlo.reshape %v6857 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v6929 = stablehlo.slice %v6928 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v6930 = stablehlo.reshape %v6929 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6931 = stablehlo.reshape %v297 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6932 = stablehlo.transpose %v6931, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v6933 = stablehlo.reshape %v6932 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v6934 = stablehlo.reshape %v6930 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6935 = stablehlo.reshape %v6933 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v6936 = stablehlo.dot_general %v6934, %v6935, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v6937 = stablehlo.reshape %v6936 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v6938 = stablehlo.reshape %v313 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v6939 = stablehlo.transpose %v6938, dims = [0, 2, 1] : (tensor<32x197x197xf32>) -> tensor<32x197x197xf32>
    %v6940 = stablehlo.reshape %v6939 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v6941 = stablehlo.reshape %v6940 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v6942 = stablehlo.reshape %v6930 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6943 = stablehlo.dot_general %v6941, %v6942, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v6944 = stablehlo.reshape %v6943 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6945 = stablehlo.reshape %v306 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v6946 = stablehlo.reshape %v6937 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v6947 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6948 = stablehlo.exponential %v6945 : tensor<32x197x197xf32>
    %v6949 = stablehlo.reduce(%v6948 init: %v6947) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6950 = stablehlo.broadcast_in_dim %v6949, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v6951 = stablehlo.divide %v6948, %v6950 : tensor<32x197x197xf32>
    %v6952 = stablehlo.multiply %v6951, %v6946 : tensor<32x197x197xf32>
    %v6953 = stablehlo.reduce(%v6952 init: %v6947) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6954 = stablehlo.broadcast_in_dim %v6953, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v6955 = stablehlo.subtract %v6946, %v6954 : tensor<32x197x197xf32>
    %v6956 = stablehlo.multiply %v6951, %v6955 : tensor<32x197x197xf32>
    %v6957 = stablehlo.reshape %v6956 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v6958 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v6959 = stablehlo.multiply %v6957, %v6958 : tensor<32x38809xf32>
    %v6960 = stablehlo.reshape %v6959 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v6961 = stablehlo.reshape %v294 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6962 = stablehlo.dot_general %v6960, %v6961, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v6963 = stablehlo.reshape %v6962 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6964 = stablehlo.reshape %v291 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6965 = stablehlo.transpose %v6964, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v6966 = stablehlo.reshape %v6965 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v6967 = stablehlo.reshape %v6966 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v6968 = stablehlo.reshape %v6959 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v6969 = stablehlo.dot_general %v6967, %v6968, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x64x197xf32>, tensor<32x197x197xf32>) -> tensor<32x64x197xf32>
    %v6970 = stablehlo.reshape %v6969 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v6971 = stablehlo.reshape %v6970 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v6972 = stablehlo.transpose %v6971, dims = [0, 2, 1] : (tensor<32x64x197xf32>) -> tensor<32x197x64xf32>
    %v6973 = stablehlo.reshape %v6972 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6974 = stablehlo.reshape %v6963 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6975 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6976 = stablehlo.pad %v6974, %v6975, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v6977 = stablehlo.reshape %v6976 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v6978 = stablehlo.reshape %v6973 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6979 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6980 = stablehlo.pad %v6978, %v6979, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v6981 = stablehlo.reshape %v6980 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v6982 = stablehlo.reshape %v6944 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6983 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6984 = stablehlo.pad %v6982, %v6983, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v6985 = stablehlo.reshape %v6984 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v6986 = stablehlo.add %v6919, %v6977 : tensor<32x37824xf32>
    %v6987 = stablehlo.add %v6923, %v6981 : tensor<32x37824xf32>
    %v6988 = stablehlo.add %v6927, %v6985 : tensor<32x37824xf32>
    %v6989 = stablehlo.reshape %v6857 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v6990 = stablehlo.slice %v6989 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v6991 = stablehlo.reshape %v6990 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6992 = stablehlo.reshape %v331 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6993 = stablehlo.transpose %v6992, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v6994 = stablehlo.reshape %v6993 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v6995 = stablehlo.reshape %v6991 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6996 = stablehlo.reshape %v6994 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v6997 = stablehlo.dot_general %v6995, %v6996, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v6998 = stablehlo.reshape %v6997 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v6999 = stablehlo.reshape %v347 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v7000 = stablehlo.transpose %v6999, dims = [0, 2, 1] : (tensor<32x197x197xf32>) -> tensor<32x197x197xf32>
    %v7001 = stablehlo.reshape %v7000 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v7002 = stablehlo.reshape %v7001 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v7003 = stablehlo.reshape %v6991 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v7004 = stablehlo.dot_general %v7002, %v7003, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v7005 = stablehlo.reshape %v7004 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v7006 = stablehlo.reshape %v340 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v7007 = stablehlo.reshape %v6998 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v7008 = stablehlo.constant dense<0.0> : tensor<f32>
    %v7009 = stablehlo.exponential %v7006 : tensor<32x197x197xf32>
    %v7010 = stablehlo.reduce(%v7009 init: %v7008) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v7011 = stablehlo.broadcast_in_dim %v7010, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v7012 = stablehlo.divide %v7009, %v7011 : tensor<32x197x197xf32>
    %v7013 = stablehlo.multiply %v7012, %v7007 : tensor<32x197x197xf32>
    %v7014 = stablehlo.reduce(%v7013 init: %v7008) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v7015 = stablehlo.broadcast_in_dim %v7014, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v7016 = stablehlo.subtract %v7007, %v7015 : tensor<32x197x197xf32>
    %v7017 = stablehlo.multiply %v7012, %v7016 : tensor<32x197x197xf32>
    %v7018 = stablehlo.reshape %v7017 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v7019 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v7020 = stablehlo.multiply %v7018, %v7019 : tensor<32x38809xf32>
    %v7021 = stablehlo.reshape %v7020 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v7022 = stablehlo.reshape %v328 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v7023 = stablehlo.dot_general %v7021, %v7022, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v7024 = stablehlo.reshape %v7023 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v7025 = stablehlo.reshape %v325 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v7026 = stablehlo.transpose %v7025, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v7027 = stablehlo.reshape %v7026 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v7028 = stablehlo.reshape %v7027 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v7029 = stablehlo.reshape %v7020 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v7030 = stablehlo.dot_general %v7028, %v7029, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x64x197xf32>, tensor<32x197x197xf32>) -> tensor<32x64x197xf32>
    %v7031 = stablehlo.reshape %v7030 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v7032 = stablehlo.reshape %v7031 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v7033 = stablehlo.transpose %v7032, dims = [0, 2, 1] : (tensor<32x64x197xf32>) -> tensor<32x197x64xf32>
    %v7034 = stablehlo.reshape %v7033 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v7035 = stablehlo.reshape %v7024 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v7036 = stablehlo.constant dense<0.0> : tensor<f32>
    %v7037 = stablehlo.pad %v7035, %v7036, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v7038 = stablehlo.reshape %v7037 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v7039 = stablehlo.reshape %v7034 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v7040 = stablehlo.constant dense<0.0> : tensor<f32>
    %v7041 = stablehlo.pad %v7039, %v7040, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v7042 = stablehlo.reshape %v7041 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v7043 = stablehlo.reshape %v7005 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v7044 = stablehlo.constant dense<0.0> : tensor<f32>
    %v7045 = stablehlo.pad %v7043, %v7044, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v7046 = stablehlo.reshape %v7045 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v7047 = stablehlo.add %v6986, %v7038 : tensor<32x37824xf32>
    %v7048 = stablehlo.add %v6987, %v7042 : tensor<32x37824xf32>
    %v7049 = stablehlo.add %v6988, %v7046 : tensor<32x37824xf32>
    %v7050 = stablehlo.reshape %v7047 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v7051 = stablehlo.dot_general %v7050, %b1_Wq, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v7052 = stablehlo.reshape %v7051 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v7053 = stablehlo.reshape %v240 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v7054 = stablehlo.reshape %v7047 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v7055 = stablehlo.dot_general %v7053, %v7054, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %v7056 = stablehlo.constant dense<0.003125> : tensor<192x192xf32>
    %v7057 = stablehlo.multiply %v7055, %v7056 : tensor<192x192xf32>
    %v7058 = stablehlo.subtract %b1_Wq, %v7057 : tensor<192x192xf32>
    %v7059 = stablehlo.constant dense<0.0> : tensor<f32>
    %v7060 = stablehlo.reshape %v7047 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v7061 = stablehlo.reduce(%v7060 init: %v7059) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v7062 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v7063 = stablehlo.multiply %v7061, %v7062 : tensor<192xf32>
    %v7064 = stablehlo.subtract %b1_bq, %v7063 : tensor<192xf32>
    %v7065 = stablehlo.reshape %v7048 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v7066 = stablehlo.dot_general %v7065, %b1_Wk, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v7067 = stablehlo.reshape %v7066 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v7068 = stablehlo.reshape %v240 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v7069 = stablehlo.reshape %v7048 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v7070 = stablehlo.dot_general %v7068, %v7069, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %v7071 = stablehlo.constant dense<0.003125> : tensor<192x192xf32>
    %v7072 = stablehlo.multiply %v7070, %v7071 : tensor<192x192xf32>
    %v7073 = stablehlo.subtract %b1_Wk, %v7072 : tensor<192x192xf32>
    %v7074 = stablehlo.constant dense<0.0> : tensor<f32>
    %v7075 = stablehlo.reshape %v7048 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v7076 = stablehlo.reduce(%v7075 init: %v7074) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v7077 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v7078 = stablehlo.multiply %v7076, %v7077 : tensor<192xf32>
    %v7079 = stablehlo.subtract %b1_bk, %v7078 : tensor<192xf32>
    %v7080 = stablehlo.reshape %v7049 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v7081 = stablehlo.dot_general %v7080, %b1_Wv, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v7082 = stablehlo.reshape %v7081 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v7083 = stablehlo.reshape %v240 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v7084 = stablehlo.reshape %v7049 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v7085 = stablehlo.dot_general %v7083, %v7084, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %v7086 = stablehlo.constant dense<0.003125> : tensor<192x192xf32>
    %v7087 = stablehlo.multiply %v7085, %v7086 : tensor<192x192xf32>
    %v7088 = stablehlo.subtract %b1_Wv, %v7087 : tensor<192x192xf32>
    %v7089 = stablehlo.constant dense<0.0> : tensor<f32>
    %v7090 = stablehlo.reshape %v7049 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v7091 = stablehlo.reduce(%v7090 init: %v7089) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v7092 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v7093 = stablehlo.multiply %v7091, %v7092 : tensor<192xf32>
    %v7094 = stablehlo.subtract %b1_bv, %v7093 : tensor<192xf32>
    %v7095 = stablehlo.add %v7052, %v7067 : tensor<32x37824xf32>
    %v7096 = stablehlo.add %v7095, %v7082 : tensor<32x37824xf32>
    %v7097 = stablehlo.constant dense<0.0> : tensor<f32>
    %v7098 = stablehlo.reshape %v7096 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v7099 = stablehlo.reduce(%v7098 init: %v7097) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v7100 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v7101 = stablehlo.multiply %v7099, %v7100 : tensor<192xf32>
    %v7102 = stablehlo.subtract %b1_bt1, %v7101 : tensor<192xf32>
    %v7103 = stablehlo.reshape %v212 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v7104 = stablehlo.reshape %v7096 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v7105 = stablehlo.constant dense<0.0> : tensor<f32>
    %v7106 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v7107 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v7108 = stablehlo.reduce(%v7103 init: %v7105) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v7109 = stablehlo.broadcast_in_dim %v7108, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v7110 = stablehlo.divide %v7109, %v7106 : tensor<32x197x192xf32>
    %v7111 = stablehlo.subtract %v7103, %v7110 : tensor<32x197x192xf32>
    %v7112 = stablehlo.multiply %v7111, %v7111 : tensor<32x197x192xf32>
    %v7113 = stablehlo.reduce(%v7112 init: %v7105) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v7114 = stablehlo.broadcast_in_dim %v7113, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v7115 = stablehlo.divide %v7114, %v7106 : tensor<32x197x192xf32>
    %v7116 = stablehlo.add %v7115, %v7107 : tensor<32x197x192xf32>
    %v7117 = stablehlo.rsqrt %v7116 : tensor<32x197x192xf32>
    %v7118 = stablehlo.multiply %v7111, %v7117 : tensor<32x197x192xf32>
    %v7119 = stablehlo.multiply %v7104, %v7118 : tensor<32x197x192xf32>
    %v7120 = stablehlo.reduce(%v7119 init: %v7105) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v7121 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v7122 = stablehlo.multiply %v7120, %v7121 : tensor<192xf32>
    %v7123 = stablehlo.subtract %b1_g1, %v7122 : tensor<192xf32>
    %v7124 = stablehlo.reshape %v7096 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v7125 = stablehlo.broadcast_in_dim %b1_g1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v7126 = stablehlo.multiply %v7124, %v7125 : tensor<32x197x192xf32>
    %v7127 = stablehlo.reshape %v7126 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v7128 = stablehlo.reshape %v7127 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v7129 = stablehlo.reshape %v212 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v7130 = stablehlo.constant dense<0.0> : tensor<f32>
    %v7131 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v7132 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v7133 = stablehlo.reduce(%v7129 init: %v7130) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v7134 = stablehlo.broadcast_in_dim %v7133, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v7135 = stablehlo.divide %v7134, %v7131 : tensor<32x197x192xf32>
    %v7136 = stablehlo.subtract %v7129, %v7135 : tensor<32x197x192xf32>
    %v7137 = stablehlo.multiply %v7136, %v7136 : tensor<32x197x192xf32>
    %v7138 = stablehlo.reduce(%v7137 init: %v7130) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v7139 = stablehlo.broadcast_in_dim %v7138, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v7140 = stablehlo.divide %v7139, %v7131 : tensor<32x197x192xf32>
    %v7141 = stablehlo.add %v7140, %v7132 : tensor<32x197x192xf32>
    %v7142 = stablehlo.rsqrt %v7141 : tensor<32x197x192xf32>
    %v7143 = stablehlo.multiply %v7136, %v7142 : tensor<32x197x192xf32>
    %v7144 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v7145 = stablehlo.multiply %v7144, %v7128 : tensor<32x197x192xf32>
    %v7146 = stablehlo.reduce(%v7145 init: %v7130) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v7147 = stablehlo.broadcast_in_dim %v7146, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v7148 = stablehlo.multiply %v7143, %v7145 : tensor<32x197x192xf32>
    %v7149 = stablehlo.reduce(%v7148 init: %v7130) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v7150 = stablehlo.broadcast_in_dim %v7149, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v7151 = stablehlo.multiply %v7145, %v7131 : tensor<32x197x192xf32>
    %v7152 = stablehlo.subtract %v7151, %v7147 : tensor<32x197x192xf32>
    %v7153 = stablehlo.multiply %v7143, %v7150 : tensor<32x197x192xf32>
    %v7154 = stablehlo.subtract %v7152, %v7153 : tensor<32x197x192xf32>
    %v7155 = stablehlo.divide %v7142, %v7131 : tensor<32x197x192xf32>
    %v7156 = stablehlo.multiply %v7155, %v7154 : tensor<32x197x192xf32>
    %v7157 = stablehlo.reshape %v7156 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v7158 = stablehlo.add %v6854, %v7157 : tensor<32x37824xf32>
    %v7159 = stablehlo.reshape %v7158 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v7160 = stablehlo.dot_general %v7159, %b0_Wfc2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<768x192xf32>) -> tensor<32x197x768xf32>
    %v7161 = stablehlo.reshape %v7160 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v7162 = stablehlo.reshape %v206 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v7163 = stablehlo.reshape %v7158 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v7164 = stablehlo.dot_general %v7162, %v7163, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<32x197x192xf32>) -> tensor<768x192xf32>
    %v7165 = stablehlo.constant dense<0.003125> : tensor<768x192xf32>
    %v7166 = stablehlo.multiply %v7164, %v7165 : tensor<768x192xf32>
    %v7167 = stablehlo.subtract %b0_Wfc2, %v7166 : tensor<768x192xf32>
    %v7168 = stablehlo.constant dense<0.0> : tensor<f32>
    %v7169 = stablehlo.reshape %v7158 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v7170 = stablehlo.reduce(%v7169 init: %v7168) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v7171 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v7172 = stablehlo.multiply %v7170, %v7171 : tensor<192xf32>
    %v7173 = stablehlo.subtract %b0_bfc2, %v7172 : tensor<192xf32>
    %v7174 = stablehlo.multiply %v193, %v193 : tensor<32x151296xf32>
    %v7175 = stablehlo.multiply %v7174, %v193 : tensor<32x151296xf32>
    %v7176 = stablehlo.constant dense<0.044715> : tensor<32x151296xf32>
    %v7177 = stablehlo.multiply %v7176, %v7175 : tensor<32x151296xf32>
    %v7178 = stablehlo.add %v193, %v7177 : tensor<32x151296xf32>
    %v7179 = stablehlo.constant dense<0.7978845608028654> : tensor<32x151296xf32>
    %v7180 = stablehlo.multiply %v7179, %v7178 : tensor<32x151296xf32>
    %v7181 = stablehlo.tanh %v7180 : tensor<32x151296xf32>
    %v7182 = stablehlo.constant dense<1.0> : tensor<32x151296xf32>
    %v7183 = stablehlo.add %v7182, %v7181 : tensor<32x151296xf32>
    %v7184 = stablehlo.constant dense<0.5> : tensor<32x151296xf32>
    %v7185 = stablehlo.multiply %v7184, %v7183 : tensor<32x151296xf32>
    %v7186 = stablehlo.multiply %v7181, %v7181 : tensor<32x151296xf32>
    %v7187 = stablehlo.subtract %v7182, %v7186 : tensor<32x151296xf32>
    %v7188 = stablehlo.multiply %v7184, %v193 : tensor<32x151296xf32>
    %v7189 = stablehlo.multiply %v7188, %v7187 : tensor<32x151296xf32>
    %v7190 = stablehlo.constant dense<0.134145> : tensor<32x151296xf32>
    %v7191 = stablehlo.multiply %v7190, %v7174 : tensor<32x151296xf32>
    %v7192 = stablehlo.add %v7182, %v7191 : tensor<32x151296xf32>
    %v7193 = stablehlo.multiply %v7179, %v7192 : tensor<32x151296xf32>
    %v7194 = stablehlo.multiply %v7189, %v7193 : tensor<32x151296xf32>
    %v7195 = stablehlo.add %v7185, %v7194 : tensor<32x151296xf32>
    %v7196 = stablehlo.multiply %v7161, %v7195 : tensor<32x151296xf32>
    %v7197 = stablehlo.reshape %v7196 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v7198 = stablehlo.dot_general %v7197, %b0_Wfc1, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<192x768xf32>) -> tensor<32x197x192xf32>
    %v7199 = stablehlo.reshape %v7198 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v7200 = stablehlo.reshape %v188 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v7201 = stablehlo.reshape %v7196 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v7202 = stablehlo.dot_general %v7200, %v7201, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x768xf32>) -> tensor<192x768xf32>
    %v7203 = stablehlo.constant dense<0.003125> : tensor<192x768xf32>
    %v7204 = stablehlo.multiply %v7202, %v7203 : tensor<192x768xf32>
    %v7205 = stablehlo.subtract %b0_Wfc1, %v7204 : tensor<192x768xf32>
    %v7206 = stablehlo.constant dense<0.0> : tensor<f32>
    %v7207 = stablehlo.reshape %v7196 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v7208 = stablehlo.reduce(%v7207 init: %v7206) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<768xf32>
    %v7209 = stablehlo.constant dense<0.003125> : tensor<768xf32>
    %v7210 = stablehlo.multiply %v7208, %v7209 : tensor<768xf32>
    %v7211 = stablehlo.subtract %b0_bfc1, %v7210 : tensor<768xf32>
    %v7212 = stablehlo.constant dense<0.0> : tensor<f32>
    %v7213 = stablehlo.reshape %v7199 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v7214 = stablehlo.reduce(%v7213 init: %v7212) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v7215 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v7216 = stablehlo.multiply %v7214, %v7215 : tensor<192xf32>
    %v7217 = stablehlo.subtract %b0_bt2, %v7216 : tensor<192xf32>
    %v7218 = stablehlo.reshape %v160 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v7219 = stablehlo.reshape %v7199 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v7220 = stablehlo.constant dense<0.0> : tensor<f32>
    %v7221 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v7222 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v7223 = stablehlo.reduce(%v7218 init: %v7220) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v7224 = stablehlo.broadcast_in_dim %v7223, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v7225 = stablehlo.divide %v7224, %v7221 : tensor<32x197x192xf32>
    %v7226 = stablehlo.subtract %v7218, %v7225 : tensor<32x197x192xf32>
    %v7227 = stablehlo.multiply %v7226, %v7226 : tensor<32x197x192xf32>
    %v7228 = stablehlo.reduce(%v7227 init: %v7220) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v7229 = stablehlo.broadcast_in_dim %v7228, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v7230 = stablehlo.divide %v7229, %v7221 : tensor<32x197x192xf32>
    %v7231 = stablehlo.add %v7230, %v7222 : tensor<32x197x192xf32>
    %v7232 = stablehlo.rsqrt %v7231 : tensor<32x197x192xf32>
    %v7233 = stablehlo.multiply %v7226, %v7232 : tensor<32x197x192xf32>
    %v7234 = stablehlo.multiply %v7219, %v7233 : tensor<32x197x192xf32>
    %v7235 = stablehlo.reduce(%v7234 init: %v7220) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v7236 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v7237 = stablehlo.multiply %v7235, %v7236 : tensor<192xf32>
    %v7238 = stablehlo.subtract %b0_g2, %v7237 : tensor<192xf32>
    %v7239 = stablehlo.reshape %v7199 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v7240 = stablehlo.broadcast_in_dim %b0_g2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v7241 = stablehlo.multiply %v7239, %v7240 : tensor<32x197x192xf32>
    %v7242 = stablehlo.reshape %v7241 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v7243 = stablehlo.reshape %v7242 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v7244 = stablehlo.reshape %v160 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v7245 = stablehlo.constant dense<0.0> : tensor<f32>
    %v7246 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v7247 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v7248 = stablehlo.reduce(%v7244 init: %v7245) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v7249 = stablehlo.broadcast_in_dim %v7248, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v7250 = stablehlo.divide %v7249, %v7246 : tensor<32x197x192xf32>
    %v7251 = stablehlo.subtract %v7244, %v7250 : tensor<32x197x192xf32>
    %v7252 = stablehlo.multiply %v7251, %v7251 : tensor<32x197x192xf32>
    %v7253 = stablehlo.reduce(%v7252 init: %v7245) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v7254 = stablehlo.broadcast_in_dim %v7253, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v7255 = stablehlo.divide %v7254, %v7246 : tensor<32x197x192xf32>
    %v7256 = stablehlo.add %v7255, %v7247 : tensor<32x197x192xf32>
    %v7257 = stablehlo.rsqrt %v7256 : tensor<32x197x192xf32>
    %v7258 = stablehlo.multiply %v7251, %v7257 : tensor<32x197x192xf32>
    %v7259 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v7260 = stablehlo.multiply %v7259, %v7243 : tensor<32x197x192xf32>
    %v7261 = stablehlo.reduce(%v7260 init: %v7245) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v7262 = stablehlo.broadcast_in_dim %v7261, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v7263 = stablehlo.multiply %v7258, %v7260 : tensor<32x197x192xf32>
    %v7264 = stablehlo.reduce(%v7263 init: %v7245) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v7265 = stablehlo.broadcast_in_dim %v7264, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v7266 = stablehlo.multiply %v7260, %v7246 : tensor<32x197x192xf32>
    %v7267 = stablehlo.subtract %v7266, %v7262 : tensor<32x197x192xf32>
    %v7268 = stablehlo.multiply %v7258, %v7265 : tensor<32x197x192xf32>
    %v7269 = stablehlo.subtract %v7267, %v7268 : tensor<32x197x192xf32>
    %v7270 = stablehlo.divide %v7257, %v7246 : tensor<32x197x192xf32>
    %v7271 = stablehlo.multiply %v7270, %v7269 : tensor<32x197x192xf32>
    %v7272 = stablehlo.reshape %v7271 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v7273 = stablehlo.add %v7158, %v7272 : tensor<32x37824xf32>
    %v7274 = stablehlo.reshape %v7273 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v7275 = stablehlo.dot_general %v7274, %b0_Wo, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v7276 = stablehlo.reshape %v7275 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v7277 = stablehlo.reshape %v154 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v7278 = stablehlo.reshape %v7273 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v7279 = stablehlo.dot_general %v7277, %v7278, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %v7280 = stablehlo.constant dense<0.003125> : tensor<192x192xf32>
    %v7281 = stablehlo.multiply %v7279, %v7280 : tensor<192x192xf32>
    %v7282 = stablehlo.subtract %b0_Wo, %v7281 : tensor<192x192xf32>
    %v7283 = stablehlo.constant dense<0.0> : tensor<f32>
    %v7284 = stablehlo.reshape %v7273 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v7285 = stablehlo.reduce(%v7284 init: %v7283) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v7286 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v7287 = stablehlo.multiply %v7285, %v7286 : tensor<192xf32>
    %v7288 = stablehlo.subtract %b0_bo, %v7287 : tensor<192xf32>
    %v7289 = stablehlo.reshape %v7276 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v7290 = stablehlo.slice %v7289 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v7291 = stablehlo.reshape %v7290 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v7292 = stablehlo.reshape %v62 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v7293 = stablehlo.transpose %v7292, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v7294 = stablehlo.reshape %v7293 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v7295 = stablehlo.reshape %v7291 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v7296 = stablehlo.reshape %v7294 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v7297 = stablehlo.dot_general %v7295, %v7296, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v7298 = stablehlo.reshape %v7297 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v7299 = stablehlo.reshape %v78 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v7300 = stablehlo.transpose %v7299, dims = [0, 2, 1] : (tensor<32x197x197xf32>) -> tensor<32x197x197xf32>
    %v7301 = stablehlo.reshape %v7300 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v7302 = stablehlo.reshape %v7301 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v7303 = stablehlo.reshape %v7291 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v7304 = stablehlo.dot_general %v7302, %v7303, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v7305 = stablehlo.reshape %v7304 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v7306 = stablehlo.reshape %v71 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v7307 = stablehlo.reshape %v7298 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v7308 = stablehlo.constant dense<0.0> : tensor<f32>
    %v7309 = stablehlo.exponential %v7306 : tensor<32x197x197xf32>
    %v7310 = stablehlo.reduce(%v7309 init: %v7308) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v7311 = stablehlo.broadcast_in_dim %v7310, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v7312 = stablehlo.divide %v7309, %v7311 : tensor<32x197x197xf32>
    %v7313 = stablehlo.multiply %v7312, %v7307 : tensor<32x197x197xf32>
    %v7314 = stablehlo.reduce(%v7313 init: %v7308) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v7315 = stablehlo.broadcast_in_dim %v7314, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v7316 = stablehlo.subtract %v7307, %v7315 : tensor<32x197x197xf32>
    %v7317 = stablehlo.multiply %v7312, %v7316 : tensor<32x197x197xf32>
    %v7318 = stablehlo.reshape %v7317 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v7319 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v7320 = stablehlo.multiply %v7318, %v7319 : tensor<32x38809xf32>
    %v7321 = stablehlo.reshape %v7320 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v7322 = stablehlo.reshape %v59 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v7323 = stablehlo.dot_general %v7321, %v7322, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v7324 = stablehlo.reshape %v7323 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v7325 = stablehlo.reshape %v56 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v7326 = stablehlo.transpose %v7325, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v7327 = stablehlo.reshape %v7326 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v7328 = stablehlo.reshape %v7327 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v7329 = stablehlo.reshape %v7320 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v7330 = stablehlo.dot_general %v7328, %v7329, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x64x197xf32>, tensor<32x197x197xf32>) -> tensor<32x64x197xf32>
    %v7331 = stablehlo.reshape %v7330 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v7332 = stablehlo.reshape %v7331 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v7333 = stablehlo.transpose %v7332, dims = [0, 2, 1] : (tensor<32x64x197xf32>) -> tensor<32x197x64xf32>
    %v7334 = stablehlo.reshape %v7333 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v7335 = stablehlo.reshape %v7324 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v7336 = stablehlo.constant dense<0.0> : tensor<f32>
    %v7337 = stablehlo.pad %v7335, %v7336, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v7338 = stablehlo.reshape %v7337 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v7339 = stablehlo.reshape %v7334 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v7340 = stablehlo.constant dense<0.0> : tensor<f32>
    %v7341 = stablehlo.pad %v7339, %v7340, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v7342 = stablehlo.reshape %v7341 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v7343 = stablehlo.reshape %v7305 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v7344 = stablehlo.constant dense<0.0> : tensor<f32>
    %v7345 = stablehlo.pad %v7343, %v7344, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v7346 = stablehlo.reshape %v7345 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v7347 = stablehlo.reshape %v7276 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v7348 = stablehlo.slice %v7347 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v7349 = stablehlo.reshape %v7348 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v7350 = stablehlo.reshape %v95 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v7351 = stablehlo.transpose %v7350, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v7352 = stablehlo.reshape %v7351 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v7353 = stablehlo.reshape %v7349 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v7354 = stablehlo.reshape %v7352 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v7355 = stablehlo.dot_general %v7353, %v7354, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v7356 = stablehlo.reshape %v7355 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v7357 = stablehlo.reshape %v111 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v7358 = stablehlo.transpose %v7357, dims = [0, 2, 1] : (tensor<32x197x197xf32>) -> tensor<32x197x197xf32>
    %v7359 = stablehlo.reshape %v7358 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v7360 = stablehlo.reshape %v7359 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v7361 = stablehlo.reshape %v7349 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v7362 = stablehlo.dot_general %v7360, %v7361, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v7363 = stablehlo.reshape %v7362 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v7364 = stablehlo.reshape %v104 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v7365 = stablehlo.reshape %v7356 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v7366 = stablehlo.constant dense<0.0> : tensor<f32>
    %v7367 = stablehlo.exponential %v7364 : tensor<32x197x197xf32>
    %v7368 = stablehlo.reduce(%v7367 init: %v7366) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v7369 = stablehlo.broadcast_in_dim %v7368, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v7370 = stablehlo.divide %v7367, %v7369 : tensor<32x197x197xf32>
    %v7371 = stablehlo.multiply %v7370, %v7365 : tensor<32x197x197xf32>
    %v7372 = stablehlo.reduce(%v7371 init: %v7366) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v7373 = stablehlo.broadcast_in_dim %v7372, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v7374 = stablehlo.subtract %v7365, %v7373 : tensor<32x197x197xf32>
    %v7375 = stablehlo.multiply %v7370, %v7374 : tensor<32x197x197xf32>
    %v7376 = stablehlo.reshape %v7375 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v7377 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v7378 = stablehlo.multiply %v7376, %v7377 : tensor<32x38809xf32>
    %v7379 = stablehlo.reshape %v7378 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v7380 = stablehlo.reshape %v92 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v7381 = stablehlo.dot_general %v7379, %v7380, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v7382 = stablehlo.reshape %v7381 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v7383 = stablehlo.reshape %v89 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v7384 = stablehlo.transpose %v7383, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v7385 = stablehlo.reshape %v7384 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v7386 = stablehlo.reshape %v7385 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v7387 = stablehlo.reshape %v7378 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v7388 = stablehlo.dot_general %v7386, %v7387, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x64x197xf32>, tensor<32x197x197xf32>) -> tensor<32x64x197xf32>
    %v7389 = stablehlo.reshape %v7388 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v7390 = stablehlo.reshape %v7389 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v7391 = stablehlo.transpose %v7390, dims = [0, 2, 1] : (tensor<32x64x197xf32>) -> tensor<32x197x64xf32>
    %v7392 = stablehlo.reshape %v7391 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v7393 = stablehlo.reshape %v7382 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v7394 = stablehlo.constant dense<0.0> : tensor<f32>
    %v7395 = stablehlo.pad %v7393, %v7394, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v7396 = stablehlo.reshape %v7395 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v7397 = stablehlo.reshape %v7392 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v7398 = stablehlo.constant dense<0.0> : tensor<f32>
    %v7399 = stablehlo.pad %v7397, %v7398, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v7400 = stablehlo.reshape %v7399 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v7401 = stablehlo.reshape %v7363 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v7402 = stablehlo.constant dense<0.0> : tensor<f32>
    %v7403 = stablehlo.pad %v7401, %v7402, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v7404 = stablehlo.reshape %v7403 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v7405 = stablehlo.add %v7338, %v7396 : tensor<32x37824xf32>
    %v7406 = stablehlo.add %v7342, %v7400 : tensor<32x37824xf32>
    %v7407 = stablehlo.add %v7346, %v7404 : tensor<32x37824xf32>
    %v7408 = stablehlo.reshape %v7276 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v7409 = stablehlo.slice %v7408 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v7410 = stablehlo.reshape %v7409 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v7411 = stablehlo.reshape %v129 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v7412 = stablehlo.transpose %v7411, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v7413 = stablehlo.reshape %v7412 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v7414 = stablehlo.reshape %v7410 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v7415 = stablehlo.reshape %v7413 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v7416 = stablehlo.dot_general %v7414, %v7415, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v7417 = stablehlo.reshape %v7416 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v7418 = stablehlo.reshape %v145 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v7419 = stablehlo.transpose %v7418, dims = [0, 2, 1] : (tensor<32x197x197xf32>) -> tensor<32x197x197xf32>
    %v7420 = stablehlo.reshape %v7419 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v7421 = stablehlo.reshape %v7420 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v7422 = stablehlo.reshape %v7410 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v7423 = stablehlo.dot_general %v7421, %v7422, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v7424 = stablehlo.reshape %v7423 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v7425 = stablehlo.reshape %v138 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v7426 = stablehlo.reshape %v7417 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v7427 = stablehlo.constant dense<0.0> : tensor<f32>
    %v7428 = stablehlo.exponential %v7425 : tensor<32x197x197xf32>
    %v7429 = stablehlo.reduce(%v7428 init: %v7427) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v7430 = stablehlo.broadcast_in_dim %v7429, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v7431 = stablehlo.divide %v7428, %v7430 : tensor<32x197x197xf32>
    %v7432 = stablehlo.multiply %v7431, %v7426 : tensor<32x197x197xf32>
    %v7433 = stablehlo.reduce(%v7432 init: %v7427) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v7434 = stablehlo.broadcast_in_dim %v7433, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v7435 = stablehlo.subtract %v7426, %v7434 : tensor<32x197x197xf32>
    %v7436 = stablehlo.multiply %v7431, %v7435 : tensor<32x197x197xf32>
    %v7437 = stablehlo.reshape %v7436 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v7438 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v7439 = stablehlo.multiply %v7437, %v7438 : tensor<32x38809xf32>
    %v7440 = stablehlo.reshape %v7439 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v7441 = stablehlo.reshape %v126 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v7442 = stablehlo.dot_general %v7440, %v7441, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v7443 = stablehlo.reshape %v7442 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v7444 = stablehlo.reshape %v123 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v7445 = stablehlo.transpose %v7444, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v7446 = stablehlo.reshape %v7445 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v7447 = stablehlo.reshape %v7446 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v7448 = stablehlo.reshape %v7439 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v7449 = stablehlo.dot_general %v7447, %v7448, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x64x197xf32>, tensor<32x197x197xf32>) -> tensor<32x64x197xf32>
    %v7450 = stablehlo.reshape %v7449 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v7451 = stablehlo.reshape %v7450 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v7452 = stablehlo.transpose %v7451, dims = [0, 2, 1] : (tensor<32x64x197xf32>) -> tensor<32x197x64xf32>
    %v7453 = stablehlo.reshape %v7452 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v7454 = stablehlo.reshape %v7443 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v7455 = stablehlo.constant dense<0.0> : tensor<f32>
    %v7456 = stablehlo.pad %v7454, %v7455, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v7457 = stablehlo.reshape %v7456 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v7458 = stablehlo.reshape %v7453 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v7459 = stablehlo.constant dense<0.0> : tensor<f32>
    %v7460 = stablehlo.pad %v7458, %v7459, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v7461 = stablehlo.reshape %v7460 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v7462 = stablehlo.reshape %v7424 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v7463 = stablehlo.constant dense<0.0> : tensor<f32>
    %v7464 = stablehlo.pad %v7462, %v7463, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v7465 = stablehlo.reshape %v7464 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v7466 = stablehlo.add %v7405, %v7457 : tensor<32x37824xf32>
    %v7467 = stablehlo.add %v7406, %v7461 : tensor<32x37824xf32>
    %v7468 = stablehlo.add %v7407, %v7465 : tensor<32x37824xf32>
    %v7469 = stablehlo.reshape %v7466 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v7470 = stablehlo.dot_general %v7469, %b0_Wq, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v7471 = stablehlo.reshape %v7470 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v7472 = stablehlo.reshape %v38 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v7473 = stablehlo.reshape %v7466 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v7474 = stablehlo.dot_general %v7472, %v7473, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %v7475 = stablehlo.constant dense<0.003125> : tensor<192x192xf32>
    %v7476 = stablehlo.multiply %v7474, %v7475 : tensor<192x192xf32>
    %v7477 = stablehlo.subtract %b0_Wq, %v7476 : tensor<192x192xf32>
    %v7478 = stablehlo.constant dense<0.0> : tensor<f32>
    %v7479 = stablehlo.reshape %v7466 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v7480 = stablehlo.reduce(%v7479 init: %v7478) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v7481 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v7482 = stablehlo.multiply %v7480, %v7481 : tensor<192xf32>
    %v7483 = stablehlo.subtract %b0_bq, %v7482 : tensor<192xf32>
    %v7484 = stablehlo.reshape %v7467 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v7485 = stablehlo.dot_general %v7484, %b0_Wk, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v7486 = stablehlo.reshape %v7485 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v7487 = stablehlo.reshape %v38 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v7488 = stablehlo.reshape %v7467 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v7489 = stablehlo.dot_general %v7487, %v7488, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %v7490 = stablehlo.constant dense<0.003125> : tensor<192x192xf32>
    %v7491 = stablehlo.multiply %v7489, %v7490 : tensor<192x192xf32>
    %v7492 = stablehlo.subtract %b0_Wk, %v7491 : tensor<192x192xf32>
    %v7493 = stablehlo.constant dense<0.0> : tensor<f32>
    %v7494 = stablehlo.reshape %v7467 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v7495 = stablehlo.reduce(%v7494 init: %v7493) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v7496 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v7497 = stablehlo.multiply %v7495, %v7496 : tensor<192xf32>
    %v7498 = stablehlo.subtract %b0_bk, %v7497 : tensor<192xf32>
    %v7499 = stablehlo.reshape %v7468 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v7500 = stablehlo.dot_general %v7499, %b0_Wv, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v7501 = stablehlo.reshape %v7500 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v7502 = stablehlo.reshape %v38 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v7503 = stablehlo.reshape %v7468 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v7504 = stablehlo.dot_general %v7502, %v7503, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<32x197x192xf32>) -> tensor<192x192xf32>
    %v7505 = stablehlo.constant dense<0.003125> : tensor<192x192xf32>
    %v7506 = stablehlo.multiply %v7504, %v7505 : tensor<192x192xf32>
    %v7507 = stablehlo.subtract %b0_Wv, %v7506 : tensor<192x192xf32>
    %v7508 = stablehlo.constant dense<0.0> : tensor<f32>
    %v7509 = stablehlo.reshape %v7468 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v7510 = stablehlo.reduce(%v7509 init: %v7508) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v7511 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v7512 = stablehlo.multiply %v7510, %v7511 : tensor<192xf32>
    %v7513 = stablehlo.subtract %b0_bv, %v7512 : tensor<192xf32>
    %v7514 = stablehlo.add %v7471, %v7486 : tensor<32x37824xf32>
    %v7515 = stablehlo.add %v7514, %v7501 : tensor<32x37824xf32>
    %v7516 = stablehlo.constant dense<0.0> : tensor<f32>
    %v7517 = stablehlo.reshape %v7515 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v7518 = stablehlo.reduce(%v7517 init: %v7516) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v7519 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v7520 = stablehlo.multiply %v7518, %v7519 : tensor<192xf32>
    %v7521 = stablehlo.subtract %b0_bt1, %v7520 : tensor<192xf32>
    %v7522 = stablehlo.reshape %v10 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v7523 = stablehlo.reshape %v7515 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v7524 = stablehlo.constant dense<0.0> : tensor<f32>
    %v7525 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v7526 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v7527 = stablehlo.reduce(%v7522 init: %v7524) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v7528 = stablehlo.broadcast_in_dim %v7527, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v7529 = stablehlo.divide %v7528, %v7525 : tensor<32x197x192xf32>
    %v7530 = stablehlo.subtract %v7522, %v7529 : tensor<32x197x192xf32>
    %v7531 = stablehlo.multiply %v7530, %v7530 : tensor<32x197x192xf32>
    %v7532 = stablehlo.reduce(%v7531 init: %v7524) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v7533 = stablehlo.broadcast_in_dim %v7532, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v7534 = stablehlo.divide %v7533, %v7525 : tensor<32x197x192xf32>
    %v7535 = stablehlo.add %v7534, %v7526 : tensor<32x197x192xf32>
    %v7536 = stablehlo.rsqrt %v7535 : tensor<32x197x192xf32>
    %v7537 = stablehlo.multiply %v7530, %v7536 : tensor<32x197x192xf32>
    %v7538 = stablehlo.multiply %v7523, %v7537 : tensor<32x197x192xf32>
    %v7539 = stablehlo.reduce(%v7538 init: %v7524) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v7540 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v7541 = stablehlo.multiply %v7539, %v7540 : tensor<192xf32>
    %v7542 = stablehlo.subtract %b0_g1, %v7541 : tensor<192xf32>
    %v7543 = stablehlo.reshape %v7515 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v7544 = stablehlo.broadcast_in_dim %b0_g1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v7545 = stablehlo.multiply %v7543, %v7544 : tensor<32x197x192xf32>
    %v7546 = stablehlo.reshape %v7545 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v7547 = stablehlo.reshape %v7546 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v7548 = stablehlo.reshape %v10 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v7549 = stablehlo.constant dense<0.0> : tensor<f32>
    %v7550 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v7551 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v7552 = stablehlo.reduce(%v7548 init: %v7549) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v7553 = stablehlo.broadcast_in_dim %v7552, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v7554 = stablehlo.divide %v7553, %v7550 : tensor<32x197x192xf32>
    %v7555 = stablehlo.subtract %v7548, %v7554 : tensor<32x197x192xf32>
    %v7556 = stablehlo.multiply %v7555, %v7555 : tensor<32x197x192xf32>
    %v7557 = stablehlo.reduce(%v7556 init: %v7549) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v7558 = stablehlo.broadcast_in_dim %v7557, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v7559 = stablehlo.divide %v7558, %v7550 : tensor<32x197x192xf32>
    %v7560 = stablehlo.add %v7559, %v7551 : tensor<32x197x192xf32>
    %v7561 = stablehlo.rsqrt %v7560 : tensor<32x197x192xf32>
    %v7562 = stablehlo.multiply %v7555, %v7561 : tensor<32x197x192xf32>
    %v7563 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v7564 = stablehlo.multiply %v7563, %v7547 : tensor<32x197x192xf32>
    %v7565 = stablehlo.reduce(%v7564 init: %v7549) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v7566 = stablehlo.broadcast_in_dim %v7565, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v7567 = stablehlo.multiply %v7562, %v7564 : tensor<32x197x192xf32>
    %v7568 = stablehlo.reduce(%v7567 init: %v7549) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v7569 = stablehlo.broadcast_in_dim %v7568, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v7570 = stablehlo.multiply %v7564, %v7550 : tensor<32x197x192xf32>
    %v7571 = stablehlo.subtract %v7570, %v7566 : tensor<32x197x192xf32>
    %v7572 = stablehlo.multiply %v7562, %v7569 : tensor<32x197x192xf32>
    %v7573 = stablehlo.subtract %v7571, %v7572 : tensor<32x197x192xf32>
    %v7574 = stablehlo.divide %v7561, %v7550 : tensor<32x197x192xf32>
    %v7575 = stablehlo.multiply %v7574, %v7573 : tensor<32x197x192xf32>
    %v7576 = stablehlo.reshape %v7575 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v7577 = stablehlo.add %v7273, %v7576 : tensor<32x37824xf32>
    %v7578 = stablehlo.constant dense<0.0> : tensor<f32>
    %v7579 = stablehlo.reshape %v7577 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v7580 = stablehlo.slice %v7579 [0:32, 1:197, 0:192] : (tensor<32x197x192xf32>) -> tensor<32x196x192xf32>
    %v7581 = stablehlo.reshape %v7580 : (tensor<32x196x192xf32>) -> tensor<32x14x14x192xf32>
    %v7582 = stablehlo.transpose %v7581, dims = [0, 3, 1, 2] : (tensor<32x14x14x192xf32>) -> tensor<32x192x14x14xf32>
    %v7583 = stablehlo.pad %v7582, %v7578, low = [0, 0, 0, 0], high = [0, 0, 0, 0], interior = [0, 0, 15, 15] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<32x192x209x209xf32>
    %v7584 = stablehlo.transpose %ximg, dims = [1, 0, 2, 3] : (tensor<32x3x224x224xf32>) -> tensor<3x32x224x224xf32>
    %v7585 = stablehlo.transpose %v7583, dims = [1, 0, 2, 3] : (tensor<32x192x209x209xf32>) -> tensor<192x32x209x209xf32>
    %v7586 = stablehlo.convolution(%v7584, %v7585)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x32x224x224xf32>, tensor<192x32x209x209xf32>) -> tensor<3x192x16x16xf32>
    %v7587 = stablehlo.transpose %v7586, dims = [1, 0, 2, 3] : (tensor<3x192x16x16xf32>) -> tensor<192x3x16x16xf32>
    %v7588 = stablehlo.constant dense<0.003125> : tensor<192x3x16x16xf32>
    %v7589 = stablehlo.multiply %v7587, %v7588 : tensor<192x3x16x16xf32>
    %v7590 = stablehlo.subtract %wConv, %v7589 : tensor<192x3x16x16xf32>
    %v7591 = stablehlo.constant dense<0.0> : tensor<f32>
    %v7592 = stablehlo.reshape %v7577 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v7593 = stablehlo.slice %v7592 [0:32, 1:197, 0:192] : (tensor<32x197x192xf32>) -> tensor<32x196x192xf32>
    %v7594 = stablehlo.reduce(%v7593 init: %v7591) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v7595 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v7596 = stablehlo.multiply %v7594, %v7595 : tensor<192xf32>
    %v7597 = stablehlo.subtract %bConv, %v7596 : tensor<192xf32>
    %v7598 = stablehlo.reshape %v7577 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v7599 = stablehlo.slice %v7598 [0:32, 0:1, 0:192] : (tensor<32x197x192xf32>) -> tensor<32x1x192xf32>
    %v7600 = stablehlo.reshape %v7599 : (tensor<32x1x192xf32>) -> tensor<32x192xf32>
    %v7601 = stablehlo.constant dense<0.0> : tensor<f32>
    %v7602 = stablehlo.reduce(%v7600 init: %v7601) applies stablehlo.add across dimensions = [0] : (tensor<32x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v7603 = stablehlo.constant dense<0.003125> : tensor<192xf32>
    %v7604 = stablehlo.multiply %v7602, %v7603 : tensor<192xf32>
    %v7605 = stablehlo.subtract %cls, %v7604 : tensor<192xf32>
    %v7606 = stablehlo.constant dense<0.0> : tensor<f32>
    %v7607 = stablehlo.reshape %v7577 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v7608 = stablehlo.reduce(%v7607 init: %v7606) applies stablehlo.add across dimensions = [0] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<197x192xf32>
    %v7609 = stablehlo.constant dense<0.003125> : tensor<197x192xf32>
    %v7610 = stablehlo.multiply %v7608, %v7609 : tensor<197x192xf32>
    %v7611 = stablehlo.subtract %pos, %v7610 : tensor<197x192xf32>
    return %v7590, %v7597, %v7605, %v7611, %v7542, %v7521, %v7477, %v7483, %v7492, %v7498, %v7507, %v7513, %v7282, %v7288, %v7238, %v7217, %v7205, %v7211, %v7167, %v7173, %v7123, %v7102, %v7058, %v7064, %v7073, %v7079, %v7088, %v7094, %v6863, %v6869, %v6819, %v6798, %v6786, %v6792, %v6748, %v6754, %v6704, %v6683, %v6639, %v6645, %v6654, %v6660, %v6669, %v6675, %v6444, %v6450, %v6400, %v6379, %v6367, %v6373, %v6329, %v6335, %v6285, %v6264, %v6220, %v6226, %v6235, %v6241, %v6250, %v6256, %v6025, %v6031, %v5981, %v5960, %v5948, %v5954, %v5910, %v5916, %v5866, %v5845, %v5801, %v5807, %v5816, %v5822, %v5831, %v5837, %v5606, %v5612, %v5562, %v5541, %v5529, %v5535, %v5491, %v5497, %v5447, %v5426, %v5382, %v5388, %v5397, %v5403, %v5412, %v5418, %v5187, %v5193, %v5143, %v5122, %v5110, %v5116, %v5072, %v5078, %v5028, %v5007, %v4963, %v4969, %v4978, %v4984, %v4993, %v4999, %v4768, %v4774, %v4724, %v4703, %v4691, %v4697, %v4653, %v4659, %v4609, %v4588, %v4544, %v4550, %v4559, %v4565, %v4574, %v4580, %v4349, %v4355, %v4305, %v4284, %v4272, %v4278, %v4234, %v4240, %v4190, %v4169, %v4125, %v4131, %v4140, %v4146, %v4155, %v4161, %v3930, %v3936, %v3886, %v3865, %v3853, %v3859, %v3815, %v3821, %v3771, %v3750, %v3706, %v3712, %v3721, %v3727, %v3736, %v3742, %v3511, %v3517, %v3467, %v3446, %v3434, %v3440, %v3396, %v3402, %v3352, %v3331, %v3287, %v3293, %v3302, %v3308, %v3317, %v3323, %v3092, %v3098, %v3048, %v3027, %v3015, %v3021, %v2977, %v2983, %v2933, %v2912, %v2868, %v2874, %v2883, %v2889, %v2898, %v2904, %v2673, %v2679, %v2629, %v2608, %v2596, %v2602, %v2558, %v2564, %v2515, %v2494, %v2479, %v2484 : tensor<192x3x16x16xf32>, tensor<192xf32>, tensor<192xf32>, tensor<197x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x768xf32>, tensor<768xf32>, tensor<768x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x10xf32>, tensor<10xf32>
  }
}

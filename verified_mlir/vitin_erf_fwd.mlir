module @m {
  func.func @vitin_erf_fwd(%x: tensor<256x150528xf32>, %wConv: tensor<192x3x16x16xf32>, %bConv: tensor<192xf32>, %cls: tensor<192xf32>, %pos: tensor<197x192xf32>, %b0_g1: tensor<192xf32>, %b0_bt1: tensor<192xf32>, %b0_Wq: tensor<192x192xf32>, %b0_bq: tensor<192xf32>, %b0_Wk: tensor<192x192xf32>, %b0_bk: tensor<192xf32>, %b0_Wv: tensor<192x192xf32>, %b0_bv: tensor<192xf32>, %b0_Wo: tensor<192x192xf32>, %b0_bo: tensor<192xf32>, %b0_g2: tensor<192xf32>, %b0_bt2: tensor<192xf32>, %b0_Wfc1: tensor<192x768xf32>, %b0_bfc1: tensor<768xf32>, %b0_Wfc2: tensor<768x192xf32>, %b0_bfc2: tensor<192xf32>, %b1_g1: tensor<192xf32>, %b1_bt1: tensor<192xf32>, %b1_Wq: tensor<192x192xf32>, %b1_bq: tensor<192xf32>, %b1_Wk: tensor<192x192xf32>, %b1_bk: tensor<192xf32>, %b1_Wv: tensor<192x192xf32>, %b1_bv: tensor<192xf32>, %b1_Wo: tensor<192x192xf32>, %b1_bo: tensor<192xf32>, %b1_g2: tensor<192xf32>, %b1_bt2: tensor<192xf32>, %b1_Wfc1: tensor<192x768xf32>, %b1_bfc1: tensor<768xf32>, %b1_Wfc2: tensor<768x192xf32>, %b1_bfc2: tensor<192xf32>, %b2_g1: tensor<192xf32>, %b2_bt1: tensor<192xf32>, %b2_Wq: tensor<192x192xf32>, %b2_bq: tensor<192xf32>, %b2_Wk: tensor<192x192xf32>, %b2_bk: tensor<192xf32>, %b2_Wv: tensor<192x192xf32>, %b2_bv: tensor<192xf32>, %b2_Wo: tensor<192x192xf32>, %b2_bo: tensor<192xf32>, %b2_g2: tensor<192xf32>, %b2_bt2: tensor<192xf32>, %b2_Wfc1: tensor<192x768xf32>, %b2_bfc1: tensor<768xf32>, %b2_Wfc2: tensor<768x192xf32>, %b2_bfc2: tensor<192xf32>, %b3_g1: tensor<192xf32>, %b3_bt1: tensor<192xf32>, %b3_Wq: tensor<192x192xf32>, %b3_bq: tensor<192xf32>, %b3_Wk: tensor<192x192xf32>, %b3_bk: tensor<192xf32>, %b3_Wv: tensor<192x192xf32>, %b3_bv: tensor<192xf32>, %b3_Wo: tensor<192x192xf32>, %b3_bo: tensor<192xf32>, %b3_g2: tensor<192xf32>, %b3_bt2: tensor<192xf32>, %b3_Wfc1: tensor<192x768xf32>, %b3_bfc1: tensor<768xf32>, %b3_Wfc2: tensor<768x192xf32>, %b3_bfc2: tensor<192xf32>, %b4_g1: tensor<192xf32>, %b4_bt1: tensor<192xf32>, %b4_Wq: tensor<192x192xf32>, %b4_bq: tensor<192xf32>, %b4_Wk: tensor<192x192xf32>, %b4_bk: tensor<192xf32>, %b4_Wv: tensor<192x192xf32>, %b4_bv: tensor<192xf32>, %b4_Wo: tensor<192x192xf32>, %b4_bo: tensor<192xf32>, %b4_g2: tensor<192xf32>, %b4_bt2: tensor<192xf32>, %b4_Wfc1: tensor<192x768xf32>, %b4_bfc1: tensor<768xf32>, %b4_Wfc2: tensor<768x192xf32>, %b4_bfc2: tensor<192xf32>, %b5_g1: tensor<192xf32>, %b5_bt1: tensor<192xf32>, %b5_Wq: tensor<192x192xf32>, %b5_bq: tensor<192xf32>, %b5_Wk: tensor<192x192xf32>, %b5_bk: tensor<192xf32>, %b5_Wv: tensor<192x192xf32>, %b5_bv: tensor<192xf32>, %b5_Wo: tensor<192x192xf32>, %b5_bo: tensor<192xf32>, %b5_g2: tensor<192xf32>, %b5_bt2: tensor<192xf32>, %b5_Wfc1: tensor<192x768xf32>, %b5_bfc1: tensor<768xf32>, %b5_Wfc2: tensor<768x192xf32>, %b5_bfc2: tensor<192xf32>, %b6_g1: tensor<192xf32>, %b6_bt1: tensor<192xf32>, %b6_Wq: tensor<192x192xf32>, %b6_bq: tensor<192xf32>, %b6_Wk: tensor<192x192xf32>, %b6_bk: tensor<192xf32>, %b6_Wv: tensor<192x192xf32>, %b6_bv: tensor<192xf32>, %b6_Wo: tensor<192x192xf32>, %b6_bo: tensor<192xf32>, %b6_g2: tensor<192xf32>, %b6_bt2: tensor<192xf32>, %b6_Wfc1: tensor<192x768xf32>, %b6_bfc1: tensor<768xf32>, %b6_Wfc2: tensor<768x192xf32>, %b6_bfc2: tensor<192xf32>, %b7_g1: tensor<192xf32>, %b7_bt1: tensor<192xf32>, %b7_Wq: tensor<192x192xf32>, %b7_bq: tensor<192xf32>, %b7_Wk: tensor<192x192xf32>, %b7_bk: tensor<192xf32>, %b7_Wv: tensor<192x192xf32>, %b7_bv: tensor<192xf32>, %b7_Wo: tensor<192x192xf32>, %b7_bo: tensor<192xf32>, %b7_g2: tensor<192xf32>, %b7_bt2: tensor<192xf32>, %b7_Wfc1: tensor<192x768xf32>, %b7_bfc1: tensor<768xf32>, %b7_Wfc2: tensor<768x192xf32>, %b7_bfc2: tensor<192xf32>, %b8_g1: tensor<192xf32>, %b8_bt1: tensor<192xf32>, %b8_Wq: tensor<192x192xf32>, %b8_bq: tensor<192xf32>, %b8_Wk: tensor<192x192xf32>, %b8_bk: tensor<192xf32>, %b8_Wv: tensor<192x192xf32>, %b8_bv: tensor<192xf32>, %b8_Wo: tensor<192x192xf32>, %b8_bo: tensor<192xf32>, %b8_g2: tensor<192xf32>, %b8_bt2: tensor<192xf32>, %b8_Wfc1: tensor<192x768xf32>, %b8_bfc1: tensor<768xf32>, %b8_Wfc2: tensor<768x192xf32>, %b8_bfc2: tensor<192xf32>, %b9_g1: tensor<192xf32>, %b9_bt1: tensor<192xf32>, %b9_Wq: tensor<192x192xf32>, %b9_bq: tensor<192xf32>, %b9_Wk: tensor<192x192xf32>, %b9_bk: tensor<192xf32>, %b9_Wv: tensor<192x192xf32>, %b9_bv: tensor<192xf32>, %b9_Wo: tensor<192x192xf32>, %b9_bo: tensor<192xf32>, %b9_g2: tensor<192xf32>, %b9_bt2: tensor<192xf32>, %b9_Wfc1: tensor<192x768xf32>, %b9_bfc1: tensor<768xf32>, %b9_Wfc2: tensor<768x192xf32>, %b9_bfc2: tensor<192xf32>, %b10_g1: tensor<192xf32>, %b10_bt1: tensor<192xf32>, %b10_Wq: tensor<192x192xf32>, %b10_bq: tensor<192xf32>, %b10_Wk: tensor<192x192xf32>, %b10_bk: tensor<192xf32>, %b10_Wv: tensor<192x192xf32>, %b10_bv: tensor<192xf32>, %b10_Wo: tensor<192x192xf32>, %b10_bo: tensor<192xf32>, %b10_g2: tensor<192xf32>, %b10_bt2: tensor<192xf32>, %b10_Wfc1: tensor<192x768xf32>, %b10_bfc1: tensor<768xf32>, %b10_Wfc2: tensor<768x192xf32>, %b10_bfc2: tensor<192xf32>, %b11_g1: tensor<192xf32>, %b11_bt1: tensor<192xf32>, %b11_Wq: tensor<192x192xf32>, %b11_bq: tensor<192xf32>, %b11_Wk: tensor<192x192xf32>, %b11_bk: tensor<192xf32>, %b11_Wv: tensor<192x192xf32>, %b11_bv: tensor<192xf32>, %b11_Wo: tensor<192x192xf32>, %b11_bo: tensor<192xf32>, %b11_g2: tensor<192xf32>, %b11_bt2: tensor<192xf32>, %b11_Wfc1: tensor<192x768xf32>, %b11_bfc1: tensor<768xf32>, %b11_Wfc2: tensor<768x192xf32>, %b11_bfc2: tensor<192xf32>, %gF: tensor<192xf32>, %btF: tensor<192xf32>, %Wc: tensor<192x1000xf32>, %bc: tensor<1000xf32>) -> tensor<256x1000xf32> {
    %one = stablehlo.constant dense<1.0> : tensor<f32>
    %zero = stablehlo.constant dense<0.0> : tensor<f32>
    %sc = stablehlo.constant dense<0.0> : tensor<f32>
    %v0 = stablehlo.reshape %x : (tensor<256x150528xf32>) -> tensor<256x3x224x224xf32>
    %v1 = stablehlo.convolution(%v0, %wConv)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [16, 16], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x3x224x224xf32>, tensor<192x3x16x16xf32>) -> tensor<256x192x14x14xf32>
    %v2 = stablehlo.broadcast_in_dim %bConv, dims = [1] : (tensor<192xf32>) -> tensor<256x192x14x14xf32>
    %v3 = stablehlo.add %v1, %v2 : tensor<256x192x14x14xf32>
    %v4 = stablehlo.transpose %v3, dims = [0, 2, 3, 1] : (tensor<256x192x14x14xf32>) -> tensor<256x14x14x192xf32>
    %v5 = stablehlo.reshape %v4 : (tensor<256x14x14x192xf32>) -> tensor<256x196x192xf32>
    %v6 = stablehlo.broadcast_in_dim %cls, dims = [2] : (tensor<192xf32>) -> tensor<256x1x192xf32>
    %v7 = stablehlo.concatenate %v6, %v5, dim = 1 : (tensor<256x1x192xf32>, tensor<256x196x192xf32>) -> tensor<256x197x192xf32>
    %v8 = stablehlo.broadcast_in_dim %pos, dims = [1, 2] : (tensor<197x192xf32>) -> tensor<256x197x192xf32>
    %v9 = stablehlo.add %v7, %v8 : tensor<256x197x192xf32>
    %v10 = stablehlo.reshape %v9 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v11 = stablehlo.reshape %v10 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v12 = stablehlo.constant dense<0.0> : tensor<f32>
    %v13 = stablehlo.constant dense<192.0> : tensor<256x197x192xf32>
    %v14 = stablehlo.constant dense<1.0e-5> : tensor<256x197x192xf32>
    %v15 = stablehlo.reduce(%v11 init: %v12) applies stablehlo.add across dimensions = [2] : (tensor<256x197x192xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v16 = stablehlo.broadcast_in_dim %v15, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x192xf32>
    %v17 = stablehlo.divide %v16, %v13 : tensor<256x197x192xf32>
    %v18 = stablehlo.subtract %v11, %v17 : tensor<256x197x192xf32>
    %v19 = stablehlo.multiply %v18, %v18 : tensor<256x197x192xf32>
    %v20 = stablehlo.reduce(%v19 init: %v12) applies stablehlo.add across dimensions = [2] : (tensor<256x197x192xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v21 = stablehlo.broadcast_in_dim %v20, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x192xf32>
    %v22 = stablehlo.divide %v21, %v13 : tensor<256x197x192xf32>
    %v23 = stablehlo.add %v22, %v14 : tensor<256x197x192xf32>
    %v24 = stablehlo.rsqrt %v23 : tensor<256x197x192xf32>
    %v25 = stablehlo.multiply %v18, %v24 : tensor<256x197x192xf32>
    %v26 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<256x197x192xf32>
    %v27 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<256x197x192xf32>
    %v28 = stablehlo.multiply %v25, %v26 : tensor<256x197x192xf32>
    %v29 = stablehlo.add %v28, %v27 : tensor<256x197x192xf32>
    %v30 = stablehlo.reshape %v29 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v31 = stablehlo.reshape %v30 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v32 = stablehlo.broadcast_in_dim %b0_g1, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v33 = stablehlo.multiply %v31, %v32 : tensor<256x197x192xf32>
    %v34 = stablehlo.reshape %v33 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v35 = stablehlo.reshape %v34 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v36 = stablehlo.broadcast_in_dim %b0_bt1, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v37 = stablehlo.add %v35, %v36 : tensor<256x197x192xf32>
    %v38 = stablehlo.reshape %v37 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v39 = stablehlo.reshape %v38 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v40 = stablehlo.dot_general %v39, %b0_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x192xf32>, tensor<192x192xf32>) -> tensor<256x197x192xf32>
    %v41 = stablehlo.broadcast_in_dim %b0_bq, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v42 = stablehlo.add %v40, %v41 : tensor<256x197x192xf32>
    %v43 = stablehlo.reshape %v42 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v44 = stablehlo.reshape %v38 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v45 = stablehlo.dot_general %v44, %b0_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x192xf32>, tensor<192x192xf32>) -> tensor<256x197x192xf32>
    %v46 = stablehlo.broadcast_in_dim %b0_bk, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v47 = stablehlo.add %v45, %v46 : tensor<256x197x192xf32>
    %v48 = stablehlo.reshape %v47 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v49 = stablehlo.reshape %v38 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v50 = stablehlo.dot_general %v49, %b0_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x192xf32>, tensor<192x192xf32>) -> tensor<256x197x192xf32>
    %v51 = stablehlo.broadcast_in_dim %b0_bv, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v52 = stablehlo.add %v50, %v51 : tensor<256x197x192xf32>
    %v53 = stablehlo.reshape %v52 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v54 = stablehlo.reshape %v43 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v55 = stablehlo.slice %v54 [0:256, 0:197, 0:64] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v56 = stablehlo.reshape %v55 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v57 = stablehlo.reshape %v48 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v58 = stablehlo.slice %v57 [0:256, 0:197, 0:64] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v59 = stablehlo.reshape %v58 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v60 = stablehlo.reshape %v53 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v61 = stablehlo.slice %v60 [0:256, 0:197, 0:64] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v62 = stablehlo.reshape %v61 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v63 = stablehlo.reshape %v59 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v64 = stablehlo.transpose %v63, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v65 = stablehlo.reshape %v64 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v66 = stablehlo.reshape %v56 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v67 = stablehlo.reshape %v65 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v68 = stablehlo.dot_general %v66, %v67, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v69 = stablehlo.reshape %v68 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v70 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v71 = stablehlo.multiply %v69, %v70 : tensor<256x38809xf32>
    %v72 = stablehlo.reshape %v71 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v73 = stablehlo.constant dense<0.0> : tensor<f32>
    %v74 = stablehlo.exponential %v72 : tensor<256x197x197xf32>
    %v75 = stablehlo.reduce(%v74 init: %v73) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v76 = stablehlo.broadcast_in_dim %v75, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v77 = stablehlo.divide %v74, %v76 : tensor<256x197x197xf32>
    %v78 = stablehlo.reshape %v77 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v79 = stablehlo.reshape %v78 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v80 = stablehlo.reshape %v62 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v81 = stablehlo.dot_general %v79, %v80, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v82 = stablehlo.reshape %v81 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v83 = stablehlo.reshape %v82 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v84 = stablehlo.constant dense<0.0> : tensor<f32>
    %v85 = stablehlo.pad %v83, %v84, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x192xf32>
    %v86 = stablehlo.reshape %v85 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v87 = stablehlo.reshape %v43 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v88 = stablehlo.slice %v87 [0:256, 0:197, 64:128] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v89 = stablehlo.reshape %v88 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v90 = stablehlo.reshape %v48 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v91 = stablehlo.slice %v90 [0:256, 0:197, 64:128] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v92 = stablehlo.reshape %v91 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v93 = stablehlo.reshape %v53 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v94 = stablehlo.slice %v93 [0:256, 0:197, 64:128] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v95 = stablehlo.reshape %v94 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v96 = stablehlo.reshape %v92 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v97 = stablehlo.transpose %v96, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v98 = stablehlo.reshape %v97 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v99 = stablehlo.reshape %v89 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v100 = stablehlo.reshape %v98 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v101 = stablehlo.dot_general %v99, %v100, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v102 = stablehlo.reshape %v101 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v103 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v104 = stablehlo.multiply %v102, %v103 : tensor<256x38809xf32>
    %v105 = stablehlo.reshape %v104 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v106 = stablehlo.constant dense<0.0> : tensor<f32>
    %v107 = stablehlo.exponential %v105 : tensor<256x197x197xf32>
    %v108 = stablehlo.reduce(%v107 init: %v106) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v109 = stablehlo.broadcast_in_dim %v108, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v110 = stablehlo.divide %v107, %v109 : tensor<256x197x197xf32>
    %v111 = stablehlo.reshape %v110 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v112 = stablehlo.reshape %v111 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v113 = stablehlo.reshape %v95 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v114 = stablehlo.dot_general %v112, %v113, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v115 = stablehlo.reshape %v114 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v116 = stablehlo.reshape %v115 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v117 = stablehlo.constant dense<0.0> : tensor<f32>
    %v118 = stablehlo.pad %v116, %v117, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x192xf32>
    %v119 = stablehlo.reshape %v118 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v120 = stablehlo.add %v86, %v119 : tensor<256x37824xf32>
    %v121 = stablehlo.reshape %v43 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v122 = stablehlo.slice %v121 [0:256, 0:197, 128:192] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v123 = stablehlo.reshape %v122 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v124 = stablehlo.reshape %v48 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v125 = stablehlo.slice %v124 [0:256, 0:197, 128:192] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v126 = stablehlo.reshape %v125 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v127 = stablehlo.reshape %v53 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v128 = stablehlo.slice %v127 [0:256, 0:197, 128:192] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v129 = stablehlo.reshape %v128 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v130 = stablehlo.reshape %v126 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v131 = stablehlo.transpose %v130, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v132 = stablehlo.reshape %v131 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v133 = stablehlo.reshape %v123 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v134 = stablehlo.reshape %v132 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v135 = stablehlo.dot_general %v133, %v134, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v136 = stablehlo.reshape %v135 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v137 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v138 = stablehlo.multiply %v136, %v137 : tensor<256x38809xf32>
    %v139 = stablehlo.reshape %v138 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v140 = stablehlo.constant dense<0.0> : tensor<f32>
    %v141 = stablehlo.exponential %v139 : tensor<256x197x197xf32>
    %v142 = stablehlo.reduce(%v141 init: %v140) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v143 = stablehlo.broadcast_in_dim %v142, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v144 = stablehlo.divide %v141, %v143 : tensor<256x197x197xf32>
    %v145 = stablehlo.reshape %v144 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v146 = stablehlo.reshape %v145 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v147 = stablehlo.reshape %v129 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v148 = stablehlo.dot_general %v146, %v147, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v149 = stablehlo.reshape %v148 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v150 = stablehlo.reshape %v149 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v151 = stablehlo.constant dense<0.0> : tensor<f32>
    %v152 = stablehlo.pad %v150, %v151, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x192xf32>
    %v153 = stablehlo.reshape %v152 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v154 = stablehlo.add %v120, %v153 : tensor<256x37824xf32>
    %v155 = stablehlo.reshape %v154 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v156 = stablehlo.dot_general %v155, %b0_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x192xf32>, tensor<192x192xf32>) -> tensor<256x197x192xf32>
    %v157 = stablehlo.broadcast_in_dim %b0_bo, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v158 = stablehlo.add %v156, %v157 : tensor<256x197x192xf32>
    %v159 = stablehlo.reshape %v158 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v160 = stablehlo.add %v10, %v159 : tensor<256x37824xf32>
    %v161 = stablehlo.reshape %v160 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v162 = stablehlo.constant dense<0.0> : tensor<f32>
    %v163 = stablehlo.constant dense<192.0> : tensor<256x197x192xf32>
    %v164 = stablehlo.constant dense<1.0e-5> : tensor<256x197x192xf32>
    %v165 = stablehlo.reduce(%v161 init: %v162) applies stablehlo.add across dimensions = [2] : (tensor<256x197x192xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v166 = stablehlo.broadcast_in_dim %v165, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x192xf32>
    %v167 = stablehlo.divide %v166, %v163 : tensor<256x197x192xf32>
    %v168 = stablehlo.subtract %v161, %v167 : tensor<256x197x192xf32>
    %v169 = stablehlo.multiply %v168, %v168 : tensor<256x197x192xf32>
    %v170 = stablehlo.reduce(%v169 init: %v162) applies stablehlo.add across dimensions = [2] : (tensor<256x197x192xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v171 = stablehlo.broadcast_in_dim %v170, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x192xf32>
    %v172 = stablehlo.divide %v171, %v163 : tensor<256x197x192xf32>
    %v173 = stablehlo.add %v172, %v164 : tensor<256x197x192xf32>
    %v174 = stablehlo.rsqrt %v173 : tensor<256x197x192xf32>
    %v175 = stablehlo.multiply %v168, %v174 : tensor<256x197x192xf32>
    %v176 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<256x197x192xf32>
    %v177 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<256x197x192xf32>
    %v178 = stablehlo.multiply %v175, %v176 : tensor<256x197x192xf32>
    %v179 = stablehlo.add %v178, %v177 : tensor<256x197x192xf32>
    %v180 = stablehlo.reshape %v179 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v181 = stablehlo.reshape %v180 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v182 = stablehlo.broadcast_in_dim %b0_g2, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v183 = stablehlo.multiply %v181, %v182 : tensor<256x197x192xf32>
    %v184 = stablehlo.reshape %v183 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v185 = stablehlo.reshape %v184 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v186 = stablehlo.broadcast_in_dim %b0_bt2, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v187 = stablehlo.add %v185, %v186 : tensor<256x197x192xf32>
    %v188 = stablehlo.reshape %v187 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v189 = stablehlo.reshape %v188 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v190 = stablehlo.dot_general %v189, %b0_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x192xf32>, tensor<192x768xf32>) -> tensor<256x197x768xf32>
    %v191 = stablehlo.broadcast_in_dim %b0_bfc1, dims = [2] : (tensor<768xf32>) -> tensor<256x197x768xf32>
    %v192 = stablehlo.add %v190, %v191 : tensor<256x197x768xf32>
    %v193 = stablehlo.reshape %v192 : (tensor<256x197x768xf32>) -> tensor<256x151296xf32>
    %v194 = stablehlo.constant dense<0.5> : tensor<256x151296xf32>
    %v195 = stablehlo.multiply %v194, %v193 : tensor<256x151296xf32>
    %v196 = stablehlo.negate %v193 : tensor<256x151296xf32>
    %v197 = stablehlo.constant dense<0.7071067811865476> : tensor<256x151296xf32>
    %v198 = stablehlo.multiply %v196, %v197 : tensor<256x151296xf32>
    %v199 = chlo.erfc %v198 : tensor<256x151296xf32> -> tensor<256x151296xf32>
    %v200 = stablehlo.multiply %v195, %v199 : tensor<256x151296xf32>
    %v201 = stablehlo.reshape %v200 : (tensor<256x151296xf32>) -> tensor<256x197x768xf32>
    %v202 = stablehlo.dot_general %v201, %b0_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x768xf32>, tensor<768x192xf32>) -> tensor<256x197x192xf32>
    %v203 = stablehlo.broadcast_in_dim %b0_bfc2, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v204 = stablehlo.add %v202, %v203 : tensor<256x197x192xf32>
    %v205 = stablehlo.reshape %v204 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v206 = stablehlo.add %v160, %v205 : tensor<256x37824xf32>
    %v207 = stablehlo.reshape %v206 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v208 = stablehlo.constant dense<0.0> : tensor<f32>
    %v209 = stablehlo.constant dense<192.0> : tensor<256x197x192xf32>
    %v210 = stablehlo.constant dense<1.0e-5> : tensor<256x197x192xf32>
    %v211 = stablehlo.reduce(%v207 init: %v208) applies stablehlo.add across dimensions = [2] : (tensor<256x197x192xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v212 = stablehlo.broadcast_in_dim %v211, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x192xf32>
    %v213 = stablehlo.divide %v212, %v209 : tensor<256x197x192xf32>
    %v214 = stablehlo.subtract %v207, %v213 : tensor<256x197x192xf32>
    %v215 = stablehlo.multiply %v214, %v214 : tensor<256x197x192xf32>
    %v216 = stablehlo.reduce(%v215 init: %v208) applies stablehlo.add across dimensions = [2] : (tensor<256x197x192xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v217 = stablehlo.broadcast_in_dim %v216, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x192xf32>
    %v218 = stablehlo.divide %v217, %v209 : tensor<256x197x192xf32>
    %v219 = stablehlo.add %v218, %v210 : tensor<256x197x192xf32>
    %v220 = stablehlo.rsqrt %v219 : tensor<256x197x192xf32>
    %v221 = stablehlo.multiply %v214, %v220 : tensor<256x197x192xf32>
    %v222 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<256x197x192xf32>
    %v223 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<256x197x192xf32>
    %v224 = stablehlo.multiply %v221, %v222 : tensor<256x197x192xf32>
    %v225 = stablehlo.add %v224, %v223 : tensor<256x197x192xf32>
    %v226 = stablehlo.reshape %v225 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v227 = stablehlo.reshape %v226 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v228 = stablehlo.broadcast_in_dim %b1_g1, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v229 = stablehlo.multiply %v227, %v228 : tensor<256x197x192xf32>
    %v230 = stablehlo.reshape %v229 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v231 = stablehlo.reshape %v230 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v232 = stablehlo.broadcast_in_dim %b1_bt1, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v233 = stablehlo.add %v231, %v232 : tensor<256x197x192xf32>
    %v234 = stablehlo.reshape %v233 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v235 = stablehlo.reshape %v234 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v236 = stablehlo.dot_general %v235, %b1_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x192xf32>, tensor<192x192xf32>) -> tensor<256x197x192xf32>
    %v237 = stablehlo.broadcast_in_dim %b1_bq, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v238 = stablehlo.add %v236, %v237 : tensor<256x197x192xf32>
    %v239 = stablehlo.reshape %v238 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v240 = stablehlo.reshape %v234 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v241 = stablehlo.dot_general %v240, %b1_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x192xf32>, tensor<192x192xf32>) -> tensor<256x197x192xf32>
    %v242 = stablehlo.broadcast_in_dim %b1_bk, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v243 = stablehlo.add %v241, %v242 : tensor<256x197x192xf32>
    %v244 = stablehlo.reshape %v243 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v245 = stablehlo.reshape %v234 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v246 = stablehlo.dot_general %v245, %b1_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x192xf32>, tensor<192x192xf32>) -> tensor<256x197x192xf32>
    %v247 = stablehlo.broadcast_in_dim %b1_bv, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v248 = stablehlo.add %v246, %v247 : tensor<256x197x192xf32>
    %v249 = stablehlo.reshape %v248 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v250 = stablehlo.reshape %v239 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v251 = stablehlo.slice %v250 [0:256, 0:197, 0:64] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v252 = stablehlo.reshape %v251 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v253 = stablehlo.reshape %v244 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v254 = stablehlo.slice %v253 [0:256, 0:197, 0:64] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v255 = stablehlo.reshape %v254 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v256 = stablehlo.reshape %v249 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v257 = stablehlo.slice %v256 [0:256, 0:197, 0:64] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v258 = stablehlo.reshape %v257 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v259 = stablehlo.reshape %v255 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v260 = stablehlo.transpose %v259, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v261 = stablehlo.reshape %v260 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v262 = stablehlo.reshape %v252 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v263 = stablehlo.reshape %v261 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v264 = stablehlo.dot_general %v262, %v263, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v265 = stablehlo.reshape %v264 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v266 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v267 = stablehlo.multiply %v265, %v266 : tensor<256x38809xf32>
    %v268 = stablehlo.reshape %v267 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v269 = stablehlo.constant dense<0.0> : tensor<f32>
    %v270 = stablehlo.exponential %v268 : tensor<256x197x197xf32>
    %v271 = stablehlo.reduce(%v270 init: %v269) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v272 = stablehlo.broadcast_in_dim %v271, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v273 = stablehlo.divide %v270, %v272 : tensor<256x197x197xf32>
    %v274 = stablehlo.reshape %v273 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v275 = stablehlo.reshape %v274 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v276 = stablehlo.reshape %v258 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v277 = stablehlo.dot_general %v275, %v276, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v278 = stablehlo.reshape %v277 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v279 = stablehlo.reshape %v278 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v280 = stablehlo.constant dense<0.0> : tensor<f32>
    %v281 = stablehlo.pad %v279, %v280, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x192xf32>
    %v282 = stablehlo.reshape %v281 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v283 = stablehlo.reshape %v239 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v284 = stablehlo.slice %v283 [0:256, 0:197, 64:128] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v285 = stablehlo.reshape %v284 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v286 = stablehlo.reshape %v244 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v287 = stablehlo.slice %v286 [0:256, 0:197, 64:128] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v288 = stablehlo.reshape %v287 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v289 = stablehlo.reshape %v249 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v290 = stablehlo.slice %v289 [0:256, 0:197, 64:128] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v291 = stablehlo.reshape %v290 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v292 = stablehlo.reshape %v288 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v293 = stablehlo.transpose %v292, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v294 = stablehlo.reshape %v293 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v295 = stablehlo.reshape %v285 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v296 = stablehlo.reshape %v294 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v297 = stablehlo.dot_general %v295, %v296, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v298 = stablehlo.reshape %v297 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v299 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v300 = stablehlo.multiply %v298, %v299 : tensor<256x38809xf32>
    %v301 = stablehlo.reshape %v300 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v302 = stablehlo.constant dense<0.0> : tensor<f32>
    %v303 = stablehlo.exponential %v301 : tensor<256x197x197xf32>
    %v304 = stablehlo.reduce(%v303 init: %v302) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v305 = stablehlo.broadcast_in_dim %v304, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v306 = stablehlo.divide %v303, %v305 : tensor<256x197x197xf32>
    %v307 = stablehlo.reshape %v306 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v308 = stablehlo.reshape %v307 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v309 = stablehlo.reshape %v291 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v310 = stablehlo.dot_general %v308, %v309, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v311 = stablehlo.reshape %v310 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v312 = stablehlo.reshape %v311 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v313 = stablehlo.constant dense<0.0> : tensor<f32>
    %v314 = stablehlo.pad %v312, %v313, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x192xf32>
    %v315 = stablehlo.reshape %v314 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v316 = stablehlo.add %v282, %v315 : tensor<256x37824xf32>
    %v317 = stablehlo.reshape %v239 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v318 = stablehlo.slice %v317 [0:256, 0:197, 128:192] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v319 = stablehlo.reshape %v318 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v320 = stablehlo.reshape %v244 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v321 = stablehlo.slice %v320 [0:256, 0:197, 128:192] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v322 = stablehlo.reshape %v321 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v323 = stablehlo.reshape %v249 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v324 = stablehlo.slice %v323 [0:256, 0:197, 128:192] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v325 = stablehlo.reshape %v324 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v326 = stablehlo.reshape %v322 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v327 = stablehlo.transpose %v326, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v328 = stablehlo.reshape %v327 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v329 = stablehlo.reshape %v319 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v330 = stablehlo.reshape %v328 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v331 = stablehlo.dot_general %v329, %v330, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v332 = stablehlo.reshape %v331 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v333 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v334 = stablehlo.multiply %v332, %v333 : tensor<256x38809xf32>
    %v335 = stablehlo.reshape %v334 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v336 = stablehlo.constant dense<0.0> : tensor<f32>
    %v337 = stablehlo.exponential %v335 : tensor<256x197x197xf32>
    %v338 = stablehlo.reduce(%v337 init: %v336) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v339 = stablehlo.broadcast_in_dim %v338, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v340 = stablehlo.divide %v337, %v339 : tensor<256x197x197xf32>
    %v341 = stablehlo.reshape %v340 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v342 = stablehlo.reshape %v341 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v343 = stablehlo.reshape %v325 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v344 = stablehlo.dot_general %v342, %v343, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v345 = stablehlo.reshape %v344 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v346 = stablehlo.reshape %v345 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v347 = stablehlo.constant dense<0.0> : tensor<f32>
    %v348 = stablehlo.pad %v346, %v347, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x192xf32>
    %v349 = stablehlo.reshape %v348 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v350 = stablehlo.add %v316, %v349 : tensor<256x37824xf32>
    %v351 = stablehlo.reshape %v350 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v352 = stablehlo.dot_general %v351, %b1_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x192xf32>, tensor<192x192xf32>) -> tensor<256x197x192xf32>
    %v353 = stablehlo.broadcast_in_dim %b1_bo, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v354 = stablehlo.add %v352, %v353 : tensor<256x197x192xf32>
    %v355 = stablehlo.reshape %v354 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v356 = stablehlo.add %v206, %v355 : tensor<256x37824xf32>
    %v357 = stablehlo.reshape %v356 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v358 = stablehlo.constant dense<0.0> : tensor<f32>
    %v359 = stablehlo.constant dense<192.0> : tensor<256x197x192xf32>
    %v360 = stablehlo.constant dense<1.0e-5> : tensor<256x197x192xf32>
    %v361 = stablehlo.reduce(%v357 init: %v358) applies stablehlo.add across dimensions = [2] : (tensor<256x197x192xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v362 = stablehlo.broadcast_in_dim %v361, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x192xf32>
    %v363 = stablehlo.divide %v362, %v359 : tensor<256x197x192xf32>
    %v364 = stablehlo.subtract %v357, %v363 : tensor<256x197x192xf32>
    %v365 = stablehlo.multiply %v364, %v364 : tensor<256x197x192xf32>
    %v366 = stablehlo.reduce(%v365 init: %v358) applies stablehlo.add across dimensions = [2] : (tensor<256x197x192xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v367 = stablehlo.broadcast_in_dim %v366, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x192xf32>
    %v368 = stablehlo.divide %v367, %v359 : tensor<256x197x192xf32>
    %v369 = stablehlo.add %v368, %v360 : tensor<256x197x192xf32>
    %v370 = stablehlo.rsqrt %v369 : tensor<256x197x192xf32>
    %v371 = stablehlo.multiply %v364, %v370 : tensor<256x197x192xf32>
    %v372 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<256x197x192xf32>
    %v373 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<256x197x192xf32>
    %v374 = stablehlo.multiply %v371, %v372 : tensor<256x197x192xf32>
    %v375 = stablehlo.add %v374, %v373 : tensor<256x197x192xf32>
    %v376 = stablehlo.reshape %v375 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v377 = stablehlo.reshape %v376 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v378 = stablehlo.broadcast_in_dim %b1_g2, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v379 = stablehlo.multiply %v377, %v378 : tensor<256x197x192xf32>
    %v380 = stablehlo.reshape %v379 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v381 = stablehlo.reshape %v380 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v382 = stablehlo.broadcast_in_dim %b1_bt2, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v383 = stablehlo.add %v381, %v382 : tensor<256x197x192xf32>
    %v384 = stablehlo.reshape %v383 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v385 = stablehlo.reshape %v384 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v386 = stablehlo.dot_general %v385, %b1_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x192xf32>, tensor<192x768xf32>) -> tensor<256x197x768xf32>
    %v387 = stablehlo.broadcast_in_dim %b1_bfc1, dims = [2] : (tensor<768xf32>) -> tensor<256x197x768xf32>
    %v388 = stablehlo.add %v386, %v387 : tensor<256x197x768xf32>
    %v389 = stablehlo.reshape %v388 : (tensor<256x197x768xf32>) -> tensor<256x151296xf32>
    %v390 = stablehlo.constant dense<0.5> : tensor<256x151296xf32>
    %v391 = stablehlo.multiply %v390, %v389 : tensor<256x151296xf32>
    %v392 = stablehlo.negate %v389 : tensor<256x151296xf32>
    %v393 = stablehlo.constant dense<0.7071067811865476> : tensor<256x151296xf32>
    %v394 = stablehlo.multiply %v392, %v393 : tensor<256x151296xf32>
    %v395 = chlo.erfc %v394 : tensor<256x151296xf32> -> tensor<256x151296xf32>
    %v396 = stablehlo.multiply %v391, %v395 : tensor<256x151296xf32>
    %v397 = stablehlo.reshape %v396 : (tensor<256x151296xf32>) -> tensor<256x197x768xf32>
    %v398 = stablehlo.dot_general %v397, %b1_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x768xf32>, tensor<768x192xf32>) -> tensor<256x197x192xf32>
    %v399 = stablehlo.broadcast_in_dim %b1_bfc2, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v400 = stablehlo.add %v398, %v399 : tensor<256x197x192xf32>
    %v401 = stablehlo.reshape %v400 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v402 = stablehlo.add %v356, %v401 : tensor<256x37824xf32>
    %v403 = stablehlo.reshape %v402 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v404 = stablehlo.constant dense<0.0> : tensor<f32>
    %v405 = stablehlo.constant dense<192.0> : tensor<256x197x192xf32>
    %v406 = stablehlo.constant dense<1.0e-5> : tensor<256x197x192xf32>
    %v407 = stablehlo.reduce(%v403 init: %v404) applies stablehlo.add across dimensions = [2] : (tensor<256x197x192xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v408 = stablehlo.broadcast_in_dim %v407, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x192xf32>
    %v409 = stablehlo.divide %v408, %v405 : tensor<256x197x192xf32>
    %v410 = stablehlo.subtract %v403, %v409 : tensor<256x197x192xf32>
    %v411 = stablehlo.multiply %v410, %v410 : tensor<256x197x192xf32>
    %v412 = stablehlo.reduce(%v411 init: %v404) applies stablehlo.add across dimensions = [2] : (tensor<256x197x192xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v413 = stablehlo.broadcast_in_dim %v412, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x192xf32>
    %v414 = stablehlo.divide %v413, %v405 : tensor<256x197x192xf32>
    %v415 = stablehlo.add %v414, %v406 : tensor<256x197x192xf32>
    %v416 = stablehlo.rsqrt %v415 : tensor<256x197x192xf32>
    %v417 = stablehlo.multiply %v410, %v416 : tensor<256x197x192xf32>
    %v418 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<256x197x192xf32>
    %v419 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<256x197x192xf32>
    %v420 = stablehlo.multiply %v417, %v418 : tensor<256x197x192xf32>
    %v421 = stablehlo.add %v420, %v419 : tensor<256x197x192xf32>
    %v422 = stablehlo.reshape %v421 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v423 = stablehlo.reshape %v422 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v424 = stablehlo.broadcast_in_dim %b2_g1, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v425 = stablehlo.multiply %v423, %v424 : tensor<256x197x192xf32>
    %v426 = stablehlo.reshape %v425 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v427 = stablehlo.reshape %v426 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v428 = stablehlo.broadcast_in_dim %b2_bt1, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v429 = stablehlo.add %v427, %v428 : tensor<256x197x192xf32>
    %v430 = stablehlo.reshape %v429 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v431 = stablehlo.reshape %v430 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v432 = stablehlo.dot_general %v431, %b2_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x192xf32>, tensor<192x192xf32>) -> tensor<256x197x192xf32>
    %v433 = stablehlo.broadcast_in_dim %b2_bq, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v434 = stablehlo.add %v432, %v433 : tensor<256x197x192xf32>
    %v435 = stablehlo.reshape %v434 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v436 = stablehlo.reshape %v430 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v437 = stablehlo.dot_general %v436, %b2_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x192xf32>, tensor<192x192xf32>) -> tensor<256x197x192xf32>
    %v438 = stablehlo.broadcast_in_dim %b2_bk, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v439 = stablehlo.add %v437, %v438 : tensor<256x197x192xf32>
    %v440 = stablehlo.reshape %v439 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v441 = stablehlo.reshape %v430 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v442 = stablehlo.dot_general %v441, %b2_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x192xf32>, tensor<192x192xf32>) -> tensor<256x197x192xf32>
    %v443 = stablehlo.broadcast_in_dim %b2_bv, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v444 = stablehlo.add %v442, %v443 : tensor<256x197x192xf32>
    %v445 = stablehlo.reshape %v444 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v446 = stablehlo.reshape %v435 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v447 = stablehlo.slice %v446 [0:256, 0:197, 0:64] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v448 = stablehlo.reshape %v447 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v449 = stablehlo.reshape %v440 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v450 = stablehlo.slice %v449 [0:256, 0:197, 0:64] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v451 = stablehlo.reshape %v450 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v452 = stablehlo.reshape %v445 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v453 = stablehlo.slice %v452 [0:256, 0:197, 0:64] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v454 = stablehlo.reshape %v453 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v455 = stablehlo.reshape %v451 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v456 = stablehlo.transpose %v455, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v457 = stablehlo.reshape %v456 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v458 = stablehlo.reshape %v448 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v459 = stablehlo.reshape %v457 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v460 = stablehlo.dot_general %v458, %v459, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v461 = stablehlo.reshape %v460 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v462 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v463 = stablehlo.multiply %v461, %v462 : tensor<256x38809xf32>
    %v464 = stablehlo.reshape %v463 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v465 = stablehlo.constant dense<0.0> : tensor<f32>
    %v466 = stablehlo.exponential %v464 : tensor<256x197x197xf32>
    %v467 = stablehlo.reduce(%v466 init: %v465) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v468 = stablehlo.broadcast_in_dim %v467, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v469 = stablehlo.divide %v466, %v468 : tensor<256x197x197xf32>
    %v470 = stablehlo.reshape %v469 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v471 = stablehlo.reshape %v470 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v472 = stablehlo.reshape %v454 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v473 = stablehlo.dot_general %v471, %v472, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v474 = stablehlo.reshape %v473 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v475 = stablehlo.reshape %v474 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v476 = stablehlo.constant dense<0.0> : tensor<f32>
    %v477 = stablehlo.pad %v475, %v476, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x192xf32>
    %v478 = stablehlo.reshape %v477 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v479 = stablehlo.reshape %v435 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v480 = stablehlo.slice %v479 [0:256, 0:197, 64:128] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v481 = stablehlo.reshape %v480 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v482 = stablehlo.reshape %v440 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v483 = stablehlo.slice %v482 [0:256, 0:197, 64:128] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v484 = stablehlo.reshape %v483 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v485 = stablehlo.reshape %v445 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v486 = stablehlo.slice %v485 [0:256, 0:197, 64:128] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v487 = stablehlo.reshape %v486 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v488 = stablehlo.reshape %v484 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v489 = stablehlo.transpose %v488, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v490 = stablehlo.reshape %v489 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v491 = stablehlo.reshape %v481 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v492 = stablehlo.reshape %v490 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v493 = stablehlo.dot_general %v491, %v492, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v494 = stablehlo.reshape %v493 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v495 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v496 = stablehlo.multiply %v494, %v495 : tensor<256x38809xf32>
    %v497 = stablehlo.reshape %v496 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v498 = stablehlo.constant dense<0.0> : tensor<f32>
    %v499 = stablehlo.exponential %v497 : tensor<256x197x197xf32>
    %v500 = stablehlo.reduce(%v499 init: %v498) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v501 = stablehlo.broadcast_in_dim %v500, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v502 = stablehlo.divide %v499, %v501 : tensor<256x197x197xf32>
    %v503 = stablehlo.reshape %v502 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v504 = stablehlo.reshape %v503 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v505 = stablehlo.reshape %v487 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v506 = stablehlo.dot_general %v504, %v505, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v507 = stablehlo.reshape %v506 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v508 = stablehlo.reshape %v507 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v509 = stablehlo.constant dense<0.0> : tensor<f32>
    %v510 = stablehlo.pad %v508, %v509, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x192xf32>
    %v511 = stablehlo.reshape %v510 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v512 = stablehlo.add %v478, %v511 : tensor<256x37824xf32>
    %v513 = stablehlo.reshape %v435 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v514 = stablehlo.slice %v513 [0:256, 0:197, 128:192] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v515 = stablehlo.reshape %v514 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v516 = stablehlo.reshape %v440 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v517 = stablehlo.slice %v516 [0:256, 0:197, 128:192] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v518 = stablehlo.reshape %v517 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v519 = stablehlo.reshape %v445 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v520 = stablehlo.slice %v519 [0:256, 0:197, 128:192] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v521 = stablehlo.reshape %v520 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v522 = stablehlo.reshape %v518 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v523 = stablehlo.transpose %v522, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v524 = stablehlo.reshape %v523 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v525 = stablehlo.reshape %v515 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v526 = stablehlo.reshape %v524 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v527 = stablehlo.dot_general %v525, %v526, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v528 = stablehlo.reshape %v527 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v529 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v530 = stablehlo.multiply %v528, %v529 : tensor<256x38809xf32>
    %v531 = stablehlo.reshape %v530 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v532 = stablehlo.constant dense<0.0> : tensor<f32>
    %v533 = stablehlo.exponential %v531 : tensor<256x197x197xf32>
    %v534 = stablehlo.reduce(%v533 init: %v532) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v535 = stablehlo.broadcast_in_dim %v534, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v536 = stablehlo.divide %v533, %v535 : tensor<256x197x197xf32>
    %v537 = stablehlo.reshape %v536 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v538 = stablehlo.reshape %v537 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v539 = stablehlo.reshape %v521 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v540 = stablehlo.dot_general %v538, %v539, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v541 = stablehlo.reshape %v540 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v542 = stablehlo.reshape %v541 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v543 = stablehlo.constant dense<0.0> : tensor<f32>
    %v544 = stablehlo.pad %v542, %v543, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x192xf32>
    %v545 = stablehlo.reshape %v544 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v546 = stablehlo.add %v512, %v545 : tensor<256x37824xf32>
    %v547 = stablehlo.reshape %v546 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v548 = stablehlo.dot_general %v547, %b2_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x192xf32>, tensor<192x192xf32>) -> tensor<256x197x192xf32>
    %v549 = stablehlo.broadcast_in_dim %b2_bo, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v550 = stablehlo.add %v548, %v549 : tensor<256x197x192xf32>
    %v551 = stablehlo.reshape %v550 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v552 = stablehlo.add %v402, %v551 : tensor<256x37824xf32>
    %v553 = stablehlo.reshape %v552 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v554 = stablehlo.constant dense<0.0> : tensor<f32>
    %v555 = stablehlo.constant dense<192.0> : tensor<256x197x192xf32>
    %v556 = stablehlo.constant dense<1.0e-5> : tensor<256x197x192xf32>
    %v557 = stablehlo.reduce(%v553 init: %v554) applies stablehlo.add across dimensions = [2] : (tensor<256x197x192xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v558 = stablehlo.broadcast_in_dim %v557, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x192xf32>
    %v559 = stablehlo.divide %v558, %v555 : tensor<256x197x192xf32>
    %v560 = stablehlo.subtract %v553, %v559 : tensor<256x197x192xf32>
    %v561 = stablehlo.multiply %v560, %v560 : tensor<256x197x192xf32>
    %v562 = stablehlo.reduce(%v561 init: %v554) applies stablehlo.add across dimensions = [2] : (tensor<256x197x192xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v563 = stablehlo.broadcast_in_dim %v562, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x192xf32>
    %v564 = stablehlo.divide %v563, %v555 : tensor<256x197x192xf32>
    %v565 = stablehlo.add %v564, %v556 : tensor<256x197x192xf32>
    %v566 = stablehlo.rsqrt %v565 : tensor<256x197x192xf32>
    %v567 = stablehlo.multiply %v560, %v566 : tensor<256x197x192xf32>
    %v568 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<256x197x192xf32>
    %v569 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<256x197x192xf32>
    %v570 = stablehlo.multiply %v567, %v568 : tensor<256x197x192xf32>
    %v571 = stablehlo.add %v570, %v569 : tensor<256x197x192xf32>
    %v572 = stablehlo.reshape %v571 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v573 = stablehlo.reshape %v572 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v574 = stablehlo.broadcast_in_dim %b2_g2, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v575 = stablehlo.multiply %v573, %v574 : tensor<256x197x192xf32>
    %v576 = stablehlo.reshape %v575 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v577 = stablehlo.reshape %v576 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v578 = stablehlo.broadcast_in_dim %b2_bt2, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v579 = stablehlo.add %v577, %v578 : tensor<256x197x192xf32>
    %v580 = stablehlo.reshape %v579 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v581 = stablehlo.reshape %v580 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v582 = stablehlo.dot_general %v581, %b2_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x192xf32>, tensor<192x768xf32>) -> tensor<256x197x768xf32>
    %v583 = stablehlo.broadcast_in_dim %b2_bfc1, dims = [2] : (tensor<768xf32>) -> tensor<256x197x768xf32>
    %v584 = stablehlo.add %v582, %v583 : tensor<256x197x768xf32>
    %v585 = stablehlo.reshape %v584 : (tensor<256x197x768xf32>) -> tensor<256x151296xf32>
    %v586 = stablehlo.constant dense<0.5> : tensor<256x151296xf32>
    %v587 = stablehlo.multiply %v586, %v585 : tensor<256x151296xf32>
    %v588 = stablehlo.negate %v585 : tensor<256x151296xf32>
    %v589 = stablehlo.constant dense<0.7071067811865476> : tensor<256x151296xf32>
    %v590 = stablehlo.multiply %v588, %v589 : tensor<256x151296xf32>
    %v591 = chlo.erfc %v590 : tensor<256x151296xf32> -> tensor<256x151296xf32>
    %v592 = stablehlo.multiply %v587, %v591 : tensor<256x151296xf32>
    %v593 = stablehlo.reshape %v592 : (tensor<256x151296xf32>) -> tensor<256x197x768xf32>
    %v594 = stablehlo.dot_general %v593, %b2_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x768xf32>, tensor<768x192xf32>) -> tensor<256x197x192xf32>
    %v595 = stablehlo.broadcast_in_dim %b2_bfc2, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v596 = stablehlo.add %v594, %v595 : tensor<256x197x192xf32>
    %v597 = stablehlo.reshape %v596 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v598 = stablehlo.add %v552, %v597 : tensor<256x37824xf32>
    %v599 = stablehlo.reshape %v598 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v600 = stablehlo.constant dense<0.0> : tensor<f32>
    %v601 = stablehlo.constant dense<192.0> : tensor<256x197x192xf32>
    %v602 = stablehlo.constant dense<1.0e-5> : tensor<256x197x192xf32>
    %v603 = stablehlo.reduce(%v599 init: %v600) applies stablehlo.add across dimensions = [2] : (tensor<256x197x192xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v604 = stablehlo.broadcast_in_dim %v603, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x192xf32>
    %v605 = stablehlo.divide %v604, %v601 : tensor<256x197x192xf32>
    %v606 = stablehlo.subtract %v599, %v605 : tensor<256x197x192xf32>
    %v607 = stablehlo.multiply %v606, %v606 : tensor<256x197x192xf32>
    %v608 = stablehlo.reduce(%v607 init: %v600) applies stablehlo.add across dimensions = [2] : (tensor<256x197x192xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v609 = stablehlo.broadcast_in_dim %v608, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x192xf32>
    %v610 = stablehlo.divide %v609, %v601 : tensor<256x197x192xf32>
    %v611 = stablehlo.add %v610, %v602 : tensor<256x197x192xf32>
    %v612 = stablehlo.rsqrt %v611 : tensor<256x197x192xf32>
    %v613 = stablehlo.multiply %v606, %v612 : tensor<256x197x192xf32>
    %v614 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<256x197x192xf32>
    %v615 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<256x197x192xf32>
    %v616 = stablehlo.multiply %v613, %v614 : tensor<256x197x192xf32>
    %v617 = stablehlo.add %v616, %v615 : tensor<256x197x192xf32>
    %v618 = stablehlo.reshape %v617 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v619 = stablehlo.reshape %v618 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v620 = stablehlo.broadcast_in_dim %b3_g1, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v621 = stablehlo.multiply %v619, %v620 : tensor<256x197x192xf32>
    %v622 = stablehlo.reshape %v621 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v623 = stablehlo.reshape %v622 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v624 = stablehlo.broadcast_in_dim %b3_bt1, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v625 = stablehlo.add %v623, %v624 : tensor<256x197x192xf32>
    %v626 = stablehlo.reshape %v625 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v627 = stablehlo.reshape %v626 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v628 = stablehlo.dot_general %v627, %b3_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x192xf32>, tensor<192x192xf32>) -> tensor<256x197x192xf32>
    %v629 = stablehlo.broadcast_in_dim %b3_bq, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v630 = stablehlo.add %v628, %v629 : tensor<256x197x192xf32>
    %v631 = stablehlo.reshape %v630 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v632 = stablehlo.reshape %v626 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v633 = stablehlo.dot_general %v632, %b3_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x192xf32>, tensor<192x192xf32>) -> tensor<256x197x192xf32>
    %v634 = stablehlo.broadcast_in_dim %b3_bk, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v635 = stablehlo.add %v633, %v634 : tensor<256x197x192xf32>
    %v636 = stablehlo.reshape %v635 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v637 = stablehlo.reshape %v626 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v638 = stablehlo.dot_general %v637, %b3_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x192xf32>, tensor<192x192xf32>) -> tensor<256x197x192xf32>
    %v639 = stablehlo.broadcast_in_dim %b3_bv, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v640 = stablehlo.add %v638, %v639 : tensor<256x197x192xf32>
    %v641 = stablehlo.reshape %v640 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v642 = stablehlo.reshape %v631 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v643 = stablehlo.slice %v642 [0:256, 0:197, 0:64] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v644 = stablehlo.reshape %v643 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v645 = stablehlo.reshape %v636 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v646 = stablehlo.slice %v645 [0:256, 0:197, 0:64] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v647 = stablehlo.reshape %v646 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v648 = stablehlo.reshape %v641 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v649 = stablehlo.slice %v648 [0:256, 0:197, 0:64] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v650 = stablehlo.reshape %v649 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v651 = stablehlo.reshape %v647 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v652 = stablehlo.transpose %v651, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v653 = stablehlo.reshape %v652 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v654 = stablehlo.reshape %v644 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v655 = stablehlo.reshape %v653 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v656 = stablehlo.dot_general %v654, %v655, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v657 = stablehlo.reshape %v656 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v658 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v659 = stablehlo.multiply %v657, %v658 : tensor<256x38809xf32>
    %v660 = stablehlo.reshape %v659 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v661 = stablehlo.constant dense<0.0> : tensor<f32>
    %v662 = stablehlo.exponential %v660 : tensor<256x197x197xf32>
    %v663 = stablehlo.reduce(%v662 init: %v661) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v664 = stablehlo.broadcast_in_dim %v663, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v665 = stablehlo.divide %v662, %v664 : tensor<256x197x197xf32>
    %v666 = stablehlo.reshape %v665 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v667 = stablehlo.reshape %v666 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v668 = stablehlo.reshape %v650 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v669 = stablehlo.dot_general %v667, %v668, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v670 = stablehlo.reshape %v669 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v671 = stablehlo.reshape %v670 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v672 = stablehlo.constant dense<0.0> : tensor<f32>
    %v673 = stablehlo.pad %v671, %v672, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x192xf32>
    %v674 = stablehlo.reshape %v673 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v675 = stablehlo.reshape %v631 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v676 = stablehlo.slice %v675 [0:256, 0:197, 64:128] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v677 = stablehlo.reshape %v676 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v678 = stablehlo.reshape %v636 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v679 = stablehlo.slice %v678 [0:256, 0:197, 64:128] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v680 = stablehlo.reshape %v679 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v681 = stablehlo.reshape %v641 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v682 = stablehlo.slice %v681 [0:256, 0:197, 64:128] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v683 = stablehlo.reshape %v682 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v684 = stablehlo.reshape %v680 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v685 = stablehlo.transpose %v684, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v686 = stablehlo.reshape %v685 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v687 = stablehlo.reshape %v677 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v688 = stablehlo.reshape %v686 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v689 = stablehlo.dot_general %v687, %v688, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v690 = stablehlo.reshape %v689 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v691 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v692 = stablehlo.multiply %v690, %v691 : tensor<256x38809xf32>
    %v693 = stablehlo.reshape %v692 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v694 = stablehlo.constant dense<0.0> : tensor<f32>
    %v695 = stablehlo.exponential %v693 : tensor<256x197x197xf32>
    %v696 = stablehlo.reduce(%v695 init: %v694) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v697 = stablehlo.broadcast_in_dim %v696, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v698 = stablehlo.divide %v695, %v697 : tensor<256x197x197xf32>
    %v699 = stablehlo.reshape %v698 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v700 = stablehlo.reshape %v699 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v701 = stablehlo.reshape %v683 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v702 = stablehlo.dot_general %v700, %v701, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v703 = stablehlo.reshape %v702 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v704 = stablehlo.reshape %v703 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v705 = stablehlo.constant dense<0.0> : tensor<f32>
    %v706 = stablehlo.pad %v704, %v705, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x192xf32>
    %v707 = stablehlo.reshape %v706 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v708 = stablehlo.add %v674, %v707 : tensor<256x37824xf32>
    %v709 = stablehlo.reshape %v631 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v710 = stablehlo.slice %v709 [0:256, 0:197, 128:192] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v711 = stablehlo.reshape %v710 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v712 = stablehlo.reshape %v636 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v713 = stablehlo.slice %v712 [0:256, 0:197, 128:192] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v714 = stablehlo.reshape %v713 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v715 = stablehlo.reshape %v641 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v716 = stablehlo.slice %v715 [0:256, 0:197, 128:192] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v717 = stablehlo.reshape %v716 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v718 = stablehlo.reshape %v714 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v719 = stablehlo.transpose %v718, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v720 = stablehlo.reshape %v719 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v721 = stablehlo.reshape %v711 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v722 = stablehlo.reshape %v720 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v723 = stablehlo.dot_general %v721, %v722, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v724 = stablehlo.reshape %v723 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v725 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v726 = stablehlo.multiply %v724, %v725 : tensor<256x38809xf32>
    %v727 = stablehlo.reshape %v726 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v728 = stablehlo.constant dense<0.0> : tensor<f32>
    %v729 = stablehlo.exponential %v727 : tensor<256x197x197xf32>
    %v730 = stablehlo.reduce(%v729 init: %v728) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v731 = stablehlo.broadcast_in_dim %v730, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v732 = stablehlo.divide %v729, %v731 : tensor<256x197x197xf32>
    %v733 = stablehlo.reshape %v732 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v734 = stablehlo.reshape %v733 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v735 = stablehlo.reshape %v717 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v736 = stablehlo.dot_general %v734, %v735, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v737 = stablehlo.reshape %v736 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v738 = stablehlo.reshape %v737 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v739 = stablehlo.constant dense<0.0> : tensor<f32>
    %v740 = stablehlo.pad %v738, %v739, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x192xf32>
    %v741 = stablehlo.reshape %v740 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v742 = stablehlo.add %v708, %v741 : tensor<256x37824xf32>
    %v743 = stablehlo.reshape %v742 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v744 = stablehlo.dot_general %v743, %b3_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x192xf32>, tensor<192x192xf32>) -> tensor<256x197x192xf32>
    %v745 = stablehlo.broadcast_in_dim %b3_bo, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v746 = stablehlo.add %v744, %v745 : tensor<256x197x192xf32>
    %v747 = stablehlo.reshape %v746 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v748 = stablehlo.add %v598, %v747 : tensor<256x37824xf32>
    %v749 = stablehlo.reshape %v748 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v750 = stablehlo.constant dense<0.0> : tensor<f32>
    %v751 = stablehlo.constant dense<192.0> : tensor<256x197x192xf32>
    %v752 = stablehlo.constant dense<1.0e-5> : tensor<256x197x192xf32>
    %v753 = stablehlo.reduce(%v749 init: %v750) applies stablehlo.add across dimensions = [2] : (tensor<256x197x192xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v754 = stablehlo.broadcast_in_dim %v753, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x192xf32>
    %v755 = stablehlo.divide %v754, %v751 : tensor<256x197x192xf32>
    %v756 = stablehlo.subtract %v749, %v755 : tensor<256x197x192xf32>
    %v757 = stablehlo.multiply %v756, %v756 : tensor<256x197x192xf32>
    %v758 = stablehlo.reduce(%v757 init: %v750) applies stablehlo.add across dimensions = [2] : (tensor<256x197x192xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v759 = stablehlo.broadcast_in_dim %v758, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x192xf32>
    %v760 = stablehlo.divide %v759, %v751 : tensor<256x197x192xf32>
    %v761 = stablehlo.add %v760, %v752 : tensor<256x197x192xf32>
    %v762 = stablehlo.rsqrt %v761 : tensor<256x197x192xf32>
    %v763 = stablehlo.multiply %v756, %v762 : tensor<256x197x192xf32>
    %v764 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<256x197x192xf32>
    %v765 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<256x197x192xf32>
    %v766 = stablehlo.multiply %v763, %v764 : tensor<256x197x192xf32>
    %v767 = stablehlo.add %v766, %v765 : tensor<256x197x192xf32>
    %v768 = stablehlo.reshape %v767 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v769 = stablehlo.reshape %v768 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v770 = stablehlo.broadcast_in_dim %b3_g2, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v771 = stablehlo.multiply %v769, %v770 : tensor<256x197x192xf32>
    %v772 = stablehlo.reshape %v771 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v773 = stablehlo.reshape %v772 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v774 = stablehlo.broadcast_in_dim %b3_bt2, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v775 = stablehlo.add %v773, %v774 : tensor<256x197x192xf32>
    %v776 = stablehlo.reshape %v775 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v777 = stablehlo.reshape %v776 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v778 = stablehlo.dot_general %v777, %b3_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x192xf32>, tensor<192x768xf32>) -> tensor<256x197x768xf32>
    %v779 = stablehlo.broadcast_in_dim %b3_bfc1, dims = [2] : (tensor<768xf32>) -> tensor<256x197x768xf32>
    %v780 = stablehlo.add %v778, %v779 : tensor<256x197x768xf32>
    %v781 = stablehlo.reshape %v780 : (tensor<256x197x768xf32>) -> tensor<256x151296xf32>
    %v782 = stablehlo.constant dense<0.5> : tensor<256x151296xf32>
    %v783 = stablehlo.multiply %v782, %v781 : tensor<256x151296xf32>
    %v784 = stablehlo.negate %v781 : tensor<256x151296xf32>
    %v785 = stablehlo.constant dense<0.7071067811865476> : tensor<256x151296xf32>
    %v786 = stablehlo.multiply %v784, %v785 : tensor<256x151296xf32>
    %v787 = chlo.erfc %v786 : tensor<256x151296xf32> -> tensor<256x151296xf32>
    %v788 = stablehlo.multiply %v783, %v787 : tensor<256x151296xf32>
    %v789 = stablehlo.reshape %v788 : (tensor<256x151296xf32>) -> tensor<256x197x768xf32>
    %v790 = stablehlo.dot_general %v789, %b3_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x768xf32>, tensor<768x192xf32>) -> tensor<256x197x192xf32>
    %v791 = stablehlo.broadcast_in_dim %b3_bfc2, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v792 = stablehlo.add %v790, %v791 : tensor<256x197x192xf32>
    %v793 = stablehlo.reshape %v792 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v794 = stablehlo.add %v748, %v793 : tensor<256x37824xf32>
    %v795 = stablehlo.reshape %v794 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v796 = stablehlo.constant dense<0.0> : tensor<f32>
    %v797 = stablehlo.constant dense<192.0> : tensor<256x197x192xf32>
    %v798 = stablehlo.constant dense<1.0e-5> : tensor<256x197x192xf32>
    %v799 = stablehlo.reduce(%v795 init: %v796) applies stablehlo.add across dimensions = [2] : (tensor<256x197x192xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v800 = stablehlo.broadcast_in_dim %v799, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x192xf32>
    %v801 = stablehlo.divide %v800, %v797 : tensor<256x197x192xf32>
    %v802 = stablehlo.subtract %v795, %v801 : tensor<256x197x192xf32>
    %v803 = stablehlo.multiply %v802, %v802 : tensor<256x197x192xf32>
    %v804 = stablehlo.reduce(%v803 init: %v796) applies stablehlo.add across dimensions = [2] : (tensor<256x197x192xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v805 = stablehlo.broadcast_in_dim %v804, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x192xf32>
    %v806 = stablehlo.divide %v805, %v797 : tensor<256x197x192xf32>
    %v807 = stablehlo.add %v806, %v798 : tensor<256x197x192xf32>
    %v808 = stablehlo.rsqrt %v807 : tensor<256x197x192xf32>
    %v809 = stablehlo.multiply %v802, %v808 : tensor<256x197x192xf32>
    %v810 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<256x197x192xf32>
    %v811 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<256x197x192xf32>
    %v812 = stablehlo.multiply %v809, %v810 : tensor<256x197x192xf32>
    %v813 = stablehlo.add %v812, %v811 : tensor<256x197x192xf32>
    %v814 = stablehlo.reshape %v813 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v815 = stablehlo.reshape %v814 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v816 = stablehlo.broadcast_in_dim %b4_g1, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v817 = stablehlo.multiply %v815, %v816 : tensor<256x197x192xf32>
    %v818 = stablehlo.reshape %v817 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v819 = stablehlo.reshape %v818 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v820 = stablehlo.broadcast_in_dim %b4_bt1, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v821 = stablehlo.add %v819, %v820 : tensor<256x197x192xf32>
    %v822 = stablehlo.reshape %v821 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v823 = stablehlo.reshape %v822 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v824 = stablehlo.dot_general %v823, %b4_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x192xf32>, tensor<192x192xf32>) -> tensor<256x197x192xf32>
    %v825 = stablehlo.broadcast_in_dim %b4_bq, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v826 = stablehlo.add %v824, %v825 : tensor<256x197x192xf32>
    %v827 = stablehlo.reshape %v826 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v828 = stablehlo.reshape %v822 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v829 = stablehlo.dot_general %v828, %b4_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x192xf32>, tensor<192x192xf32>) -> tensor<256x197x192xf32>
    %v830 = stablehlo.broadcast_in_dim %b4_bk, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v831 = stablehlo.add %v829, %v830 : tensor<256x197x192xf32>
    %v832 = stablehlo.reshape %v831 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v833 = stablehlo.reshape %v822 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v834 = stablehlo.dot_general %v833, %b4_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x192xf32>, tensor<192x192xf32>) -> tensor<256x197x192xf32>
    %v835 = stablehlo.broadcast_in_dim %b4_bv, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v836 = stablehlo.add %v834, %v835 : tensor<256x197x192xf32>
    %v837 = stablehlo.reshape %v836 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v838 = stablehlo.reshape %v827 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v839 = stablehlo.slice %v838 [0:256, 0:197, 0:64] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v840 = stablehlo.reshape %v839 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v841 = stablehlo.reshape %v832 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v842 = stablehlo.slice %v841 [0:256, 0:197, 0:64] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v843 = stablehlo.reshape %v842 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v844 = stablehlo.reshape %v837 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v845 = stablehlo.slice %v844 [0:256, 0:197, 0:64] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v846 = stablehlo.reshape %v845 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v847 = stablehlo.reshape %v843 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v848 = stablehlo.transpose %v847, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v849 = stablehlo.reshape %v848 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v850 = stablehlo.reshape %v840 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v851 = stablehlo.reshape %v849 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v852 = stablehlo.dot_general %v850, %v851, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v853 = stablehlo.reshape %v852 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v854 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v855 = stablehlo.multiply %v853, %v854 : tensor<256x38809xf32>
    %v856 = stablehlo.reshape %v855 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v857 = stablehlo.constant dense<0.0> : tensor<f32>
    %v858 = stablehlo.exponential %v856 : tensor<256x197x197xf32>
    %v859 = stablehlo.reduce(%v858 init: %v857) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v860 = stablehlo.broadcast_in_dim %v859, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v861 = stablehlo.divide %v858, %v860 : tensor<256x197x197xf32>
    %v862 = stablehlo.reshape %v861 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v863 = stablehlo.reshape %v862 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v864 = stablehlo.reshape %v846 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v865 = stablehlo.dot_general %v863, %v864, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v866 = stablehlo.reshape %v865 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v867 = stablehlo.reshape %v866 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v868 = stablehlo.constant dense<0.0> : tensor<f32>
    %v869 = stablehlo.pad %v867, %v868, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x192xf32>
    %v870 = stablehlo.reshape %v869 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v871 = stablehlo.reshape %v827 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v872 = stablehlo.slice %v871 [0:256, 0:197, 64:128] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v873 = stablehlo.reshape %v872 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v874 = stablehlo.reshape %v832 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v875 = stablehlo.slice %v874 [0:256, 0:197, 64:128] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v876 = stablehlo.reshape %v875 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v877 = stablehlo.reshape %v837 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v878 = stablehlo.slice %v877 [0:256, 0:197, 64:128] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v879 = stablehlo.reshape %v878 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v880 = stablehlo.reshape %v876 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v881 = stablehlo.transpose %v880, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v882 = stablehlo.reshape %v881 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v883 = stablehlo.reshape %v873 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v884 = stablehlo.reshape %v882 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v885 = stablehlo.dot_general %v883, %v884, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v886 = stablehlo.reshape %v885 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v887 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v888 = stablehlo.multiply %v886, %v887 : tensor<256x38809xf32>
    %v889 = stablehlo.reshape %v888 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v890 = stablehlo.constant dense<0.0> : tensor<f32>
    %v891 = stablehlo.exponential %v889 : tensor<256x197x197xf32>
    %v892 = stablehlo.reduce(%v891 init: %v890) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v893 = stablehlo.broadcast_in_dim %v892, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v894 = stablehlo.divide %v891, %v893 : tensor<256x197x197xf32>
    %v895 = stablehlo.reshape %v894 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v896 = stablehlo.reshape %v895 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v897 = stablehlo.reshape %v879 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v898 = stablehlo.dot_general %v896, %v897, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v899 = stablehlo.reshape %v898 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v900 = stablehlo.reshape %v899 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v901 = stablehlo.constant dense<0.0> : tensor<f32>
    %v902 = stablehlo.pad %v900, %v901, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x192xf32>
    %v903 = stablehlo.reshape %v902 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v904 = stablehlo.add %v870, %v903 : tensor<256x37824xf32>
    %v905 = stablehlo.reshape %v827 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v906 = stablehlo.slice %v905 [0:256, 0:197, 128:192] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v907 = stablehlo.reshape %v906 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v908 = stablehlo.reshape %v832 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v909 = stablehlo.slice %v908 [0:256, 0:197, 128:192] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v910 = stablehlo.reshape %v909 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v911 = stablehlo.reshape %v837 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v912 = stablehlo.slice %v911 [0:256, 0:197, 128:192] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v913 = stablehlo.reshape %v912 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v914 = stablehlo.reshape %v910 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v915 = stablehlo.transpose %v914, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v916 = stablehlo.reshape %v915 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v917 = stablehlo.reshape %v907 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v918 = stablehlo.reshape %v916 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v919 = stablehlo.dot_general %v917, %v918, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v920 = stablehlo.reshape %v919 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v921 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v922 = stablehlo.multiply %v920, %v921 : tensor<256x38809xf32>
    %v923 = stablehlo.reshape %v922 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v924 = stablehlo.constant dense<0.0> : tensor<f32>
    %v925 = stablehlo.exponential %v923 : tensor<256x197x197xf32>
    %v926 = stablehlo.reduce(%v925 init: %v924) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v927 = stablehlo.broadcast_in_dim %v926, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v928 = stablehlo.divide %v925, %v927 : tensor<256x197x197xf32>
    %v929 = stablehlo.reshape %v928 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v930 = stablehlo.reshape %v929 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v931 = stablehlo.reshape %v913 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v932 = stablehlo.dot_general %v930, %v931, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v933 = stablehlo.reshape %v932 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v934 = stablehlo.reshape %v933 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v935 = stablehlo.constant dense<0.0> : tensor<f32>
    %v936 = stablehlo.pad %v934, %v935, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x192xf32>
    %v937 = stablehlo.reshape %v936 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v938 = stablehlo.add %v904, %v937 : tensor<256x37824xf32>
    %v939 = stablehlo.reshape %v938 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v940 = stablehlo.dot_general %v939, %b4_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x192xf32>, tensor<192x192xf32>) -> tensor<256x197x192xf32>
    %v941 = stablehlo.broadcast_in_dim %b4_bo, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v942 = stablehlo.add %v940, %v941 : tensor<256x197x192xf32>
    %v943 = stablehlo.reshape %v942 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v944 = stablehlo.add %v794, %v943 : tensor<256x37824xf32>
    %v945 = stablehlo.reshape %v944 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v946 = stablehlo.constant dense<0.0> : tensor<f32>
    %v947 = stablehlo.constant dense<192.0> : tensor<256x197x192xf32>
    %v948 = stablehlo.constant dense<1.0e-5> : tensor<256x197x192xf32>
    %v949 = stablehlo.reduce(%v945 init: %v946) applies stablehlo.add across dimensions = [2] : (tensor<256x197x192xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v950 = stablehlo.broadcast_in_dim %v949, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x192xf32>
    %v951 = stablehlo.divide %v950, %v947 : tensor<256x197x192xf32>
    %v952 = stablehlo.subtract %v945, %v951 : tensor<256x197x192xf32>
    %v953 = stablehlo.multiply %v952, %v952 : tensor<256x197x192xf32>
    %v954 = stablehlo.reduce(%v953 init: %v946) applies stablehlo.add across dimensions = [2] : (tensor<256x197x192xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v955 = stablehlo.broadcast_in_dim %v954, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x192xf32>
    %v956 = stablehlo.divide %v955, %v947 : tensor<256x197x192xf32>
    %v957 = stablehlo.add %v956, %v948 : tensor<256x197x192xf32>
    %v958 = stablehlo.rsqrt %v957 : tensor<256x197x192xf32>
    %v959 = stablehlo.multiply %v952, %v958 : tensor<256x197x192xf32>
    %v960 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<256x197x192xf32>
    %v961 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<256x197x192xf32>
    %v962 = stablehlo.multiply %v959, %v960 : tensor<256x197x192xf32>
    %v963 = stablehlo.add %v962, %v961 : tensor<256x197x192xf32>
    %v964 = stablehlo.reshape %v963 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v965 = stablehlo.reshape %v964 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v966 = stablehlo.broadcast_in_dim %b4_g2, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v967 = stablehlo.multiply %v965, %v966 : tensor<256x197x192xf32>
    %v968 = stablehlo.reshape %v967 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v969 = stablehlo.reshape %v968 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v970 = stablehlo.broadcast_in_dim %b4_bt2, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v971 = stablehlo.add %v969, %v970 : tensor<256x197x192xf32>
    %v972 = stablehlo.reshape %v971 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v973 = stablehlo.reshape %v972 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v974 = stablehlo.dot_general %v973, %b4_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x192xf32>, tensor<192x768xf32>) -> tensor<256x197x768xf32>
    %v975 = stablehlo.broadcast_in_dim %b4_bfc1, dims = [2] : (tensor<768xf32>) -> tensor<256x197x768xf32>
    %v976 = stablehlo.add %v974, %v975 : tensor<256x197x768xf32>
    %v977 = stablehlo.reshape %v976 : (tensor<256x197x768xf32>) -> tensor<256x151296xf32>
    %v978 = stablehlo.constant dense<0.5> : tensor<256x151296xf32>
    %v979 = stablehlo.multiply %v978, %v977 : tensor<256x151296xf32>
    %v980 = stablehlo.negate %v977 : tensor<256x151296xf32>
    %v981 = stablehlo.constant dense<0.7071067811865476> : tensor<256x151296xf32>
    %v982 = stablehlo.multiply %v980, %v981 : tensor<256x151296xf32>
    %v983 = chlo.erfc %v982 : tensor<256x151296xf32> -> tensor<256x151296xf32>
    %v984 = stablehlo.multiply %v979, %v983 : tensor<256x151296xf32>
    %v985 = stablehlo.reshape %v984 : (tensor<256x151296xf32>) -> tensor<256x197x768xf32>
    %v986 = stablehlo.dot_general %v985, %b4_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x768xf32>, tensor<768x192xf32>) -> tensor<256x197x192xf32>
    %v987 = stablehlo.broadcast_in_dim %b4_bfc2, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v988 = stablehlo.add %v986, %v987 : tensor<256x197x192xf32>
    %v989 = stablehlo.reshape %v988 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v990 = stablehlo.add %v944, %v989 : tensor<256x37824xf32>
    %v991 = stablehlo.reshape %v990 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v992 = stablehlo.constant dense<0.0> : tensor<f32>
    %v993 = stablehlo.constant dense<192.0> : tensor<256x197x192xf32>
    %v994 = stablehlo.constant dense<1.0e-5> : tensor<256x197x192xf32>
    %v995 = stablehlo.reduce(%v991 init: %v992) applies stablehlo.add across dimensions = [2] : (tensor<256x197x192xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v996 = stablehlo.broadcast_in_dim %v995, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x192xf32>
    %v997 = stablehlo.divide %v996, %v993 : tensor<256x197x192xf32>
    %v998 = stablehlo.subtract %v991, %v997 : tensor<256x197x192xf32>
    %v999 = stablehlo.multiply %v998, %v998 : tensor<256x197x192xf32>
    %v1000 = stablehlo.reduce(%v999 init: %v992) applies stablehlo.add across dimensions = [2] : (tensor<256x197x192xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1001 = stablehlo.broadcast_in_dim %v1000, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x192xf32>
    %v1002 = stablehlo.divide %v1001, %v993 : tensor<256x197x192xf32>
    %v1003 = stablehlo.add %v1002, %v994 : tensor<256x197x192xf32>
    %v1004 = stablehlo.rsqrt %v1003 : tensor<256x197x192xf32>
    %v1005 = stablehlo.multiply %v998, %v1004 : tensor<256x197x192xf32>
    %v1006 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<256x197x192xf32>
    %v1007 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<256x197x192xf32>
    %v1008 = stablehlo.multiply %v1005, %v1006 : tensor<256x197x192xf32>
    %v1009 = stablehlo.add %v1008, %v1007 : tensor<256x197x192xf32>
    %v1010 = stablehlo.reshape %v1009 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v1011 = stablehlo.reshape %v1010 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1012 = stablehlo.broadcast_in_dim %b5_g1, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v1013 = stablehlo.multiply %v1011, %v1012 : tensor<256x197x192xf32>
    %v1014 = stablehlo.reshape %v1013 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v1015 = stablehlo.reshape %v1014 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1016 = stablehlo.broadcast_in_dim %b5_bt1, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v1017 = stablehlo.add %v1015, %v1016 : tensor<256x197x192xf32>
    %v1018 = stablehlo.reshape %v1017 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v1019 = stablehlo.reshape %v1018 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1020 = stablehlo.dot_general %v1019, %b5_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x192xf32>, tensor<192x192xf32>) -> tensor<256x197x192xf32>
    %v1021 = stablehlo.broadcast_in_dim %b5_bq, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v1022 = stablehlo.add %v1020, %v1021 : tensor<256x197x192xf32>
    %v1023 = stablehlo.reshape %v1022 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v1024 = stablehlo.reshape %v1018 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1025 = stablehlo.dot_general %v1024, %b5_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x192xf32>, tensor<192x192xf32>) -> tensor<256x197x192xf32>
    %v1026 = stablehlo.broadcast_in_dim %b5_bk, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v1027 = stablehlo.add %v1025, %v1026 : tensor<256x197x192xf32>
    %v1028 = stablehlo.reshape %v1027 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v1029 = stablehlo.reshape %v1018 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1030 = stablehlo.dot_general %v1029, %b5_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x192xf32>, tensor<192x192xf32>) -> tensor<256x197x192xf32>
    %v1031 = stablehlo.broadcast_in_dim %b5_bv, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v1032 = stablehlo.add %v1030, %v1031 : tensor<256x197x192xf32>
    %v1033 = stablehlo.reshape %v1032 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v1034 = stablehlo.reshape %v1023 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1035 = stablehlo.slice %v1034 [0:256, 0:197, 0:64] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v1036 = stablehlo.reshape %v1035 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1037 = stablehlo.reshape %v1028 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1038 = stablehlo.slice %v1037 [0:256, 0:197, 0:64] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v1039 = stablehlo.reshape %v1038 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1040 = stablehlo.reshape %v1033 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1041 = stablehlo.slice %v1040 [0:256, 0:197, 0:64] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v1042 = stablehlo.reshape %v1041 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1043 = stablehlo.reshape %v1039 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1044 = stablehlo.transpose %v1043, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v1045 = stablehlo.reshape %v1044 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v1046 = stablehlo.reshape %v1036 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1047 = stablehlo.reshape %v1045 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v1048 = stablehlo.dot_general %v1046, %v1047, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v1049 = stablehlo.reshape %v1048 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v1050 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v1051 = stablehlo.multiply %v1049, %v1050 : tensor<256x38809xf32>
    %v1052 = stablehlo.reshape %v1051 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v1053 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1054 = stablehlo.exponential %v1052 : tensor<256x197x197xf32>
    %v1055 = stablehlo.reduce(%v1054 init: %v1053) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1056 = stablehlo.broadcast_in_dim %v1055, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v1057 = stablehlo.divide %v1054, %v1056 : tensor<256x197x197xf32>
    %v1058 = stablehlo.reshape %v1057 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v1059 = stablehlo.reshape %v1058 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v1060 = stablehlo.reshape %v1042 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1061 = stablehlo.dot_general %v1059, %v1060, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v1062 = stablehlo.reshape %v1061 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1063 = stablehlo.reshape %v1062 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1064 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1065 = stablehlo.pad %v1063, %v1064, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x192xf32>
    %v1066 = stablehlo.reshape %v1065 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v1067 = stablehlo.reshape %v1023 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1068 = stablehlo.slice %v1067 [0:256, 0:197, 64:128] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v1069 = stablehlo.reshape %v1068 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1070 = stablehlo.reshape %v1028 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1071 = stablehlo.slice %v1070 [0:256, 0:197, 64:128] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v1072 = stablehlo.reshape %v1071 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1073 = stablehlo.reshape %v1033 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1074 = stablehlo.slice %v1073 [0:256, 0:197, 64:128] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v1075 = stablehlo.reshape %v1074 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1076 = stablehlo.reshape %v1072 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1077 = stablehlo.transpose %v1076, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v1078 = stablehlo.reshape %v1077 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v1079 = stablehlo.reshape %v1069 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1080 = stablehlo.reshape %v1078 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v1081 = stablehlo.dot_general %v1079, %v1080, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v1082 = stablehlo.reshape %v1081 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v1083 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v1084 = stablehlo.multiply %v1082, %v1083 : tensor<256x38809xf32>
    %v1085 = stablehlo.reshape %v1084 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v1086 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1087 = stablehlo.exponential %v1085 : tensor<256x197x197xf32>
    %v1088 = stablehlo.reduce(%v1087 init: %v1086) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1089 = stablehlo.broadcast_in_dim %v1088, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v1090 = stablehlo.divide %v1087, %v1089 : tensor<256x197x197xf32>
    %v1091 = stablehlo.reshape %v1090 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v1092 = stablehlo.reshape %v1091 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v1093 = stablehlo.reshape %v1075 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1094 = stablehlo.dot_general %v1092, %v1093, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v1095 = stablehlo.reshape %v1094 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1096 = stablehlo.reshape %v1095 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1097 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1098 = stablehlo.pad %v1096, %v1097, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x192xf32>
    %v1099 = stablehlo.reshape %v1098 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v1100 = stablehlo.add %v1066, %v1099 : tensor<256x37824xf32>
    %v1101 = stablehlo.reshape %v1023 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1102 = stablehlo.slice %v1101 [0:256, 0:197, 128:192] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v1103 = stablehlo.reshape %v1102 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1104 = stablehlo.reshape %v1028 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1105 = stablehlo.slice %v1104 [0:256, 0:197, 128:192] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v1106 = stablehlo.reshape %v1105 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1107 = stablehlo.reshape %v1033 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1108 = stablehlo.slice %v1107 [0:256, 0:197, 128:192] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v1109 = stablehlo.reshape %v1108 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1110 = stablehlo.reshape %v1106 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1111 = stablehlo.transpose %v1110, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v1112 = stablehlo.reshape %v1111 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v1113 = stablehlo.reshape %v1103 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1114 = stablehlo.reshape %v1112 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v1115 = stablehlo.dot_general %v1113, %v1114, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v1116 = stablehlo.reshape %v1115 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v1117 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v1118 = stablehlo.multiply %v1116, %v1117 : tensor<256x38809xf32>
    %v1119 = stablehlo.reshape %v1118 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v1120 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1121 = stablehlo.exponential %v1119 : tensor<256x197x197xf32>
    %v1122 = stablehlo.reduce(%v1121 init: %v1120) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1123 = stablehlo.broadcast_in_dim %v1122, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v1124 = stablehlo.divide %v1121, %v1123 : tensor<256x197x197xf32>
    %v1125 = stablehlo.reshape %v1124 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v1126 = stablehlo.reshape %v1125 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v1127 = stablehlo.reshape %v1109 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1128 = stablehlo.dot_general %v1126, %v1127, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v1129 = stablehlo.reshape %v1128 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1130 = stablehlo.reshape %v1129 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1131 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1132 = stablehlo.pad %v1130, %v1131, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x192xf32>
    %v1133 = stablehlo.reshape %v1132 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v1134 = stablehlo.add %v1100, %v1133 : tensor<256x37824xf32>
    %v1135 = stablehlo.reshape %v1134 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1136 = stablehlo.dot_general %v1135, %b5_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x192xf32>, tensor<192x192xf32>) -> tensor<256x197x192xf32>
    %v1137 = stablehlo.broadcast_in_dim %b5_bo, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v1138 = stablehlo.add %v1136, %v1137 : tensor<256x197x192xf32>
    %v1139 = stablehlo.reshape %v1138 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v1140 = stablehlo.add %v990, %v1139 : tensor<256x37824xf32>
    %v1141 = stablehlo.reshape %v1140 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1142 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1143 = stablehlo.constant dense<192.0> : tensor<256x197x192xf32>
    %v1144 = stablehlo.constant dense<1.0e-5> : tensor<256x197x192xf32>
    %v1145 = stablehlo.reduce(%v1141 init: %v1142) applies stablehlo.add across dimensions = [2] : (tensor<256x197x192xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1146 = stablehlo.broadcast_in_dim %v1145, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x192xf32>
    %v1147 = stablehlo.divide %v1146, %v1143 : tensor<256x197x192xf32>
    %v1148 = stablehlo.subtract %v1141, %v1147 : tensor<256x197x192xf32>
    %v1149 = stablehlo.multiply %v1148, %v1148 : tensor<256x197x192xf32>
    %v1150 = stablehlo.reduce(%v1149 init: %v1142) applies stablehlo.add across dimensions = [2] : (tensor<256x197x192xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1151 = stablehlo.broadcast_in_dim %v1150, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x192xf32>
    %v1152 = stablehlo.divide %v1151, %v1143 : tensor<256x197x192xf32>
    %v1153 = stablehlo.add %v1152, %v1144 : tensor<256x197x192xf32>
    %v1154 = stablehlo.rsqrt %v1153 : tensor<256x197x192xf32>
    %v1155 = stablehlo.multiply %v1148, %v1154 : tensor<256x197x192xf32>
    %v1156 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<256x197x192xf32>
    %v1157 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<256x197x192xf32>
    %v1158 = stablehlo.multiply %v1155, %v1156 : tensor<256x197x192xf32>
    %v1159 = stablehlo.add %v1158, %v1157 : tensor<256x197x192xf32>
    %v1160 = stablehlo.reshape %v1159 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v1161 = stablehlo.reshape %v1160 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1162 = stablehlo.broadcast_in_dim %b5_g2, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v1163 = stablehlo.multiply %v1161, %v1162 : tensor<256x197x192xf32>
    %v1164 = stablehlo.reshape %v1163 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v1165 = stablehlo.reshape %v1164 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1166 = stablehlo.broadcast_in_dim %b5_bt2, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v1167 = stablehlo.add %v1165, %v1166 : tensor<256x197x192xf32>
    %v1168 = stablehlo.reshape %v1167 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v1169 = stablehlo.reshape %v1168 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1170 = stablehlo.dot_general %v1169, %b5_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x192xf32>, tensor<192x768xf32>) -> tensor<256x197x768xf32>
    %v1171 = stablehlo.broadcast_in_dim %b5_bfc1, dims = [2] : (tensor<768xf32>) -> tensor<256x197x768xf32>
    %v1172 = stablehlo.add %v1170, %v1171 : tensor<256x197x768xf32>
    %v1173 = stablehlo.reshape %v1172 : (tensor<256x197x768xf32>) -> tensor<256x151296xf32>
    %v1174 = stablehlo.constant dense<0.5> : tensor<256x151296xf32>
    %v1175 = stablehlo.multiply %v1174, %v1173 : tensor<256x151296xf32>
    %v1176 = stablehlo.negate %v1173 : tensor<256x151296xf32>
    %v1177 = stablehlo.constant dense<0.7071067811865476> : tensor<256x151296xf32>
    %v1178 = stablehlo.multiply %v1176, %v1177 : tensor<256x151296xf32>
    %v1179 = chlo.erfc %v1178 : tensor<256x151296xf32> -> tensor<256x151296xf32>
    %v1180 = stablehlo.multiply %v1175, %v1179 : tensor<256x151296xf32>
    %v1181 = stablehlo.reshape %v1180 : (tensor<256x151296xf32>) -> tensor<256x197x768xf32>
    %v1182 = stablehlo.dot_general %v1181, %b5_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x768xf32>, tensor<768x192xf32>) -> tensor<256x197x192xf32>
    %v1183 = stablehlo.broadcast_in_dim %b5_bfc2, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v1184 = stablehlo.add %v1182, %v1183 : tensor<256x197x192xf32>
    %v1185 = stablehlo.reshape %v1184 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v1186 = stablehlo.add %v1140, %v1185 : tensor<256x37824xf32>
    %v1187 = stablehlo.reshape %v1186 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1188 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1189 = stablehlo.constant dense<192.0> : tensor<256x197x192xf32>
    %v1190 = stablehlo.constant dense<1.0e-5> : tensor<256x197x192xf32>
    %v1191 = stablehlo.reduce(%v1187 init: %v1188) applies stablehlo.add across dimensions = [2] : (tensor<256x197x192xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1192 = stablehlo.broadcast_in_dim %v1191, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x192xf32>
    %v1193 = stablehlo.divide %v1192, %v1189 : tensor<256x197x192xf32>
    %v1194 = stablehlo.subtract %v1187, %v1193 : tensor<256x197x192xf32>
    %v1195 = stablehlo.multiply %v1194, %v1194 : tensor<256x197x192xf32>
    %v1196 = stablehlo.reduce(%v1195 init: %v1188) applies stablehlo.add across dimensions = [2] : (tensor<256x197x192xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1197 = stablehlo.broadcast_in_dim %v1196, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x192xf32>
    %v1198 = stablehlo.divide %v1197, %v1189 : tensor<256x197x192xf32>
    %v1199 = stablehlo.add %v1198, %v1190 : tensor<256x197x192xf32>
    %v1200 = stablehlo.rsqrt %v1199 : tensor<256x197x192xf32>
    %v1201 = stablehlo.multiply %v1194, %v1200 : tensor<256x197x192xf32>
    %v1202 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<256x197x192xf32>
    %v1203 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<256x197x192xf32>
    %v1204 = stablehlo.multiply %v1201, %v1202 : tensor<256x197x192xf32>
    %v1205 = stablehlo.add %v1204, %v1203 : tensor<256x197x192xf32>
    %v1206 = stablehlo.reshape %v1205 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v1207 = stablehlo.reshape %v1206 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1208 = stablehlo.broadcast_in_dim %b6_g1, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v1209 = stablehlo.multiply %v1207, %v1208 : tensor<256x197x192xf32>
    %v1210 = stablehlo.reshape %v1209 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v1211 = stablehlo.reshape %v1210 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1212 = stablehlo.broadcast_in_dim %b6_bt1, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v1213 = stablehlo.add %v1211, %v1212 : tensor<256x197x192xf32>
    %v1214 = stablehlo.reshape %v1213 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v1215 = stablehlo.reshape %v1214 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1216 = stablehlo.dot_general %v1215, %b6_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x192xf32>, tensor<192x192xf32>) -> tensor<256x197x192xf32>
    %v1217 = stablehlo.broadcast_in_dim %b6_bq, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v1218 = stablehlo.add %v1216, %v1217 : tensor<256x197x192xf32>
    %v1219 = stablehlo.reshape %v1218 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v1220 = stablehlo.reshape %v1214 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1221 = stablehlo.dot_general %v1220, %b6_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x192xf32>, tensor<192x192xf32>) -> tensor<256x197x192xf32>
    %v1222 = stablehlo.broadcast_in_dim %b6_bk, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v1223 = stablehlo.add %v1221, %v1222 : tensor<256x197x192xf32>
    %v1224 = stablehlo.reshape %v1223 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v1225 = stablehlo.reshape %v1214 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1226 = stablehlo.dot_general %v1225, %b6_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x192xf32>, tensor<192x192xf32>) -> tensor<256x197x192xf32>
    %v1227 = stablehlo.broadcast_in_dim %b6_bv, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v1228 = stablehlo.add %v1226, %v1227 : tensor<256x197x192xf32>
    %v1229 = stablehlo.reshape %v1228 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v1230 = stablehlo.reshape %v1219 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1231 = stablehlo.slice %v1230 [0:256, 0:197, 0:64] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v1232 = stablehlo.reshape %v1231 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1233 = stablehlo.reshape %v1224 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1234 = stablehlo.slice %v1233 [0:256, 0:197, 0:64] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v1235 = stablehlo.reshape %v1234 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1236 = stablehlo.reshape %v1229 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1237 = stablehlo.slice %v1236 [0:256, 0:197, 0:64] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v1238 = stablehlo.reshape %v1237 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1239 = stablehlo.reshape %v1235 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1240 = stablehlo.transpose %v1239, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v1241 = stablehlo.reshape %v1240 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v1242 = stablehlo.reshape %v1232 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1243 = stablehlo.reshape %v1241 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v1244 = stablehlo.dot_general %v1242, %v1243, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v1245 = stablehlo.reshape %v1244 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v1246 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v1247 = stablehlo.multiply %v1245, %v1246 : tensor<256x38809xf32>
    %v1248 = stablehlo.reshape %v1247 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v1249 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1250 = stablehlo.exponential %v1248 : tensor<256x197x197xf32>
    %v1251 = stablehlo.reduce(%v1250 init: %v1249) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1252 = stablehlo.broadcast_in_dim %v1251, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v1253 = stablehlo.divide %v1250, %v1252 : tensor<256x197x197xf32>
    %v1254 = stablehlo.reshape %v1253 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v1255 = stablehlo.reshape %v1254 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v1256 = stablehlo.reshape %v1238 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1257 = stablehlo.dot_general %v1255, %v1256, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v1258 = stablehlo.reshape %v1257 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1259 = stablehlo.reshape %v1258 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1260 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1261 = stablehlo.pad %v1259, %v1260, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x192xf32>
    %v1262 = stablehlo.reshape %v1261 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v1263 = stablehlo.reshape %v1219 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1264 = stablehlo.slice %v1263 [0:256, 0:197, 64:128] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v1265 = stablehlo.reshape %v1264 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1266 = stablehlo.reshape %v1224 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1267 = stablehlo.slice %v1266 [0:256, 0:197, 64:128] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v1268 = stablehlo.reshape %v1267 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1269 = stablehlo.reshape %v1229 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1270 = stablehlo.slice %v1269 [0:256, 0:197, 64:128] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v1271 = stablehlo.reshape %v1270 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1272 = stablehlo.reshape %v1268 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1273 = stablehlo.transpose %v1272, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v1274 = stablehlo.reshape %v1273 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v1275 = stablehlo.reshape %v1265 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1276 = stablehlo.reshape %v1274 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v1277 = stablehlo.dot_general %v1275, %v1276, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v1278 = stablehlo.reshape %v1277 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v1279 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v1280 = stablehlo.multiply %v1278, %v1279 : tensor<256x38809xf32>
    %v1281 = stablehlo.reshape %v1280 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v1282 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1283 = stablehlo.exponential %v1281 : tensor<256x197x197xf32>
    %v1284 = stablehlo.reduce(%v1283 init: %v1282) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1285 = stablehlo.broadcast_in_dim %v1284, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v1286 = stablehlo.divide %v1283, %v1285 : tensor<256x197x197xf32>
    %v1287 = stablehlo.reshape %v1286 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v1288 = stablehlo.reshape %v1287 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v1289 = stablehlo.reshape %v1271 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1290 = stablehlo.dot_general %v1288, %v1289, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v1291 = stablehlo.reshape %v1290 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1292 = stablehlo.reshape %v1291 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1293 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1294 = stablehlo.pad %v1292, %v1293, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x192xf32>
    %v1295 = stablehlo.reshape %v1294 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v1296 = stablehlo.add %v1262, %v1295 : tensor<256x37824xf32>
    %v1297 = stablehlo.reshape %v1219 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1298 = stablehlo.slice %v1297 [0:256, 0:197, 128:192] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v1299 = stablehlo.reshape %v1298 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1300 = stablehlo.reshape %v1224 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1301 = stablehlo.slice %v1300 [0:256, 0:197, 128:192] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v1302 = stablehlo.reshape %v1301 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1303 = stablehlo.reshape %v1229 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1304 = stablehlo.slice %v1303 [0:256, 0:197, 128:192] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v1305 = stablehlo.reshape %v1304 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1306 = stablehlo.reshape %v1302 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1307 = stablehlo.transpose %v1306, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v1308 = stablehlo.reshape %v1307 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v1309 = stablehlo.reshape %v1299 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1310 = stablehlo.reshape %v1308 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v1311 = stablehlo.dot_general %v1309, %v1310, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v1312 = stablehlo.reshape %v1311 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v1313 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v1314 = stablehlo.multiply %v1312, %v1313 : tensor<256x38809xf32>
    %v1315 = stablehlo.reshape %v1314 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v1316 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1317 = stablehlo.exponential %v1315 : tensor<256x197x197xf32>
    %v1318 = stablehlo.reduce(%v1317 init: %v1316) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1319 = stablehlo.broadcast_in_dim %v1318, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v1320 = stablehlo.divide %v1317, %v1319 : tensor<256x197x197xf32>
    %v1321 = stablehlo.reshape %v1320 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v1322 = stablehlo.reshape %v1321 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v1323 = stablehlo.reshape %v1305 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1324 = stablehlo.dot_general %v1322, %v1323, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v1325 = stablehlo.reshape %v1324 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1326 = stablehlo.reshape %v1325 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1327 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1328 = stablehlo.pad %v1326, %v1327, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x192xf32>
    %v1329 = stablehlo.reshape %v1328 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v1330 = stablehlo.add %v1296, %v1329 : tensor<256x37824xf32>
    %v1331 = stablehlo.reshape %v1330 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1332 = stablehlo.dot_general %v1331, %b6_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x192xf32>, tensor<192x192xf32>) -> tensor<256x197x192xf32>
    %v1333 = stablehlo.broadcast_in_dim %b6_bo, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v1334 = stablehlo.add %v1332, %v1333 : tensor<256x197x192xf32>
    %v1335 = stablehlo.reshape %v1334 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v1336 = stablehlo.add %v1186, %v1335 : tensor<256x37824xf32>
    %v1337 = stablehlo.reshape %v1336 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1338 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1339 = stablehlo.constant dense<192.0> : tensor<256x197x192xf32>
    %v1340 = stablehlo.constant dense<1.0e-5> : tensor<256x197x192xf32>
    %v1341 = stablehlo.reduce(%v1337 init: %v1338) applies stablehlo.add across dimensions = [2] : (tensor<256x197x192xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1342 = stablehlo.broadcast_in_dim %v1341, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x192xf32>
    %v1343 = stablehlo.divide %v1342, %v1339 : tensor<256x197x192xf32>
    %v1344 = stablehlo.subtract %v1337, %v1343 : tensor<256x197x192xf32>
    %v1345 = stablehlo.multiply %v1344, %v1344 : tensor<256x197x192xf32>
    %v1346 = stablehlo.reduce(%v1345 init: %v1338) applies stablehlo.add across dimensions = [2] : (tensor<256x197x192xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1347 = stablehlo.broadcast_in_dim %v1346, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x192xf32>
    %v1348 = stablehlo.divide %v1347, %v1339 : tensor<256x197x192xf32>
    %v1349 = stablehlo.add %v1348, %v1340 : tensor<256x197x192xf32>
    %v1350 = stablehlo.rsqrt %v1349 : tensor<256x197x192xf32>
    %v1351 = stablehlo.multiply %v1344, %v1350 : tensor<256x197x192xf32>
    %v1352 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<256x197x192xf32>
    %v1353 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<256x197x192xf32>
    %v1354 = stablehlo.multiply %v1351, %v1352 : tensor<256x197x192xf32>
    %v1355 = stablehlo.add %v1354, %v1353 : tensor<256x197x192xf32>
    %v1356 = stablehlo.reshape %v1355 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v1357 = stablehlo.reshape %v1356 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1358 = stablehlo.broadcast_in_dim %b6_g2, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v1359 = stablehlo.multiply %v1357, %v1358 : tensor<256x197x192xf32>
    %v1360 = stablehlo.reshape %v1359 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v1361 = stablehlo.reshape %v1360 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1362 = stablehlo.broadcast_in_dim %b6_bt2, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v1363 = stablehlo.add %v1361, %v1362 : tensor<256x197x192xf32>
    %v1364 = stablehlo.reshape %v1363 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v1365 = stablehlo.reshape %v1364 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1366 = stablehlo.dot_general %v1365, %b6_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x192xf32>, tensor<192x768xf32>) -> tensor<256x197x768xf32>
    %v1367 = stablehlo.broadcast_in_dim %b6_bfc1, dims = [2] : (tensor<768xf32>) -> tensor<256x197x768xf32>
    %v1368 = stablehlo.add %v1366, %v1367 : tensor<256x197x768xf32>
    %v1369 = stablehlo.reshape %v1368 : (tensor<256x197x768xf32>) -> tensor<256x151296xf32>
    %v1370 = stablehlo.constant dense<0.5> : tensor<256x151296xf32>
    %v1371 = stablehlo.multiply %v1370, %v1369 : tensor<256x151296xf32>
    %v1372 = stablehlo.negate %v1369 : tensor<256x151296xf32>
    %v1373 = stablehlo.constant dense<0.7071067811865476> : tensor<256x151296xf32>
    %v1374 = stablehlo.multiply %v1372, %v1373 : tensor<256x151296xf32>
    %v1375 = chlo.erfc %v1374 : tensor<256x151296xf32> -> tensor<256x151296xf32>
    %v1376 = stablehlo.multiply %v1371, %v1375 : tensor<256x151296xf32>
    %v1377 = stablehlo.reshape %v1376 : (tensor<256x151296xf32>) -> tensor<256x197x768xf32>
    %v1378 = stablehlo.dot_general %v1377, %b6_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x768xf32>, tensor<768x192xf32>) -> tensor<256x197x192xf32>
    %v1379 = stablehlo.broadcast_in_dim %b6_bfc2, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v1380 = stablehlo.add %v1378, %v1379 : tensor<256x197x192xf32>
    %v1381 = stablehlo.reshape %v1380 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v1382 = stablehlo.add %v1336, %v1381 : tensor<256x37824xf32>
    %v1383 = stablehlo.reshape %v1382 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1384 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1385 = stablehlo.constant dense<192.0> : tensor<256x197x192xf32>
    %v1386 = stablehlo.constant dense<1.0e-5> : tensor<256x197x192xf32>
    %v1387 = stablehlo.reduce(%v1383 init: %v1384) applies stablehlo.add across dimensions = [2] : (tensor<256x197x192xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1388 = stablehlo.broadcast_in_dim %v1387, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x192xf32>
    %v1389 = stablehlo.divide %v1388, %v1385 : tensor<256x197x192xf32>
    %v1390 = stablehlo.subtract %v1383, %v1389 : tensor<256x197x192xf32>
    %v1391 = stablehlo.multiply %v1390, %v1390 : tensor<256x197x192xf32>
    %v1392 = stablehlo.reduce(%v1391 init: %v1384) applies stablehlo.add across dimensions = [2] : (tensor<256x197x192xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1393 = stablehlo.broadcast_in_dim %v1392, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x192xf32>
    %v1394 = stablehlo.divide %v1393, %v1385 : tensor<256x197x192xf32>
    %v1395 = stablehlo.add %v1394, %v1386 : tensor<256x197x192xf32>
    %v1396 = stablehlo.rsqrt %v1395 : tensor<256x197x192xf32>
    %v1397 = stablehlo.multiply %v1390, %v1396 : tensor<256x197x192xf32>
    %v1398 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<256x197x192xf32>
    %v1399 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<256x197x192xf32>
    %v1400 = stablehlo.multiply %v1397, %v1398 : tensor<256x197x192xf32>
    %v1401 = stablehlo.add %v1400, %v1399 : tensor<256x197x192xf32>
    %v1402 = stablehlo.reshape %v1401 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v1403 = stablehlo.reshape %v1402 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1404 = stablehlo.broadcast_in_dim %b7_g1, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v1405 = stablehlo.multiply %v1403, %v1404 : tensor<256x197x192xf32>
    %v1406 = stablehlo.reshape %v1405 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v1407 = stablehlo.reshape %v1406 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1408 = stablehlo.broadcast_in_dim %b7_bt1, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v1409 = stablehlo.add %v1407, %v1408 : tensor<256x197x192xf32>
    %v1410 = stablehlo.reshape %v1409 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v1411 = stablehlo.reshape %v1410 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1412 = stablehlo.dot_general %v1411, %b7_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x192xf32>, tensor<192x192xf32>) -> tensor<256x197x192xf32>
    %v1413 = stablehlo.broadcast_in_dim %b7_bq, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v1414 = stablehlo.add %v1412, %v1413 : tensor<256x197x192xf32>
    %v1415 = stablehlo.reshape %v1414 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v1416 = stablehlo.reshape %v1410 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1417 = stablehlo.dot_general %v1416, %b7_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x192xf32>, tensor<192x192xf32>) -> tensor<256x197x192xf32>
    %v1418 = stablehlo.broadcast_in_dim %b7_bk, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v1419 = stablehlo.add %v1417, %v1418 : tensor<256x197x192xf32>
    %v1420 = stablehlo.reshape %v1419 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v1421 = stablehlo.reshape %v1410 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1422 = stablehlo.dot_general %v1421, %b7_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x192xf32>, tensor<192x192xf32>) -> tensor<256x197x192xf32>
    %v1423 = stablehlo.broadcast_in_dim %b7_bv, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v1424 = stablehlo.add %v1422, %v1423 : tensor<256x197x192xf32>
    %v1425 = stablehlo.reshape %v1424 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v1426 = stablehlo.reshape %v1415 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1427 = stablehlo.slice %v1426 [0:256, 0:197, 0:64] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v1428 = stablehlo.reshape %v1427 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1429 = stablehlo.reshape %v1420 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1430 = stablehlo.slice %v1429 [0:256, 0:197, 0:64] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v1431 = stablehlo.reshape %v1430 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1432 = stablehlo.reshape %v1425 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1433 = stablehlo.slice %v1432 [0:256, 0:197, 0:64] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v1434 = stablehlo.reshape %v1433 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1435 = stablehlo.reshape %v1431 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1436 = stablehlo.transpose %v1435, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v1437 = stablehlo.reshape %v1436 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v1438 = stablehlo.reshape %v1428 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1439 = stablehlo.reshape %v1437 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v1440 = stablehlo.dot_general %v1438, %v1439, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v1441 = stablehlo.reshape %v1440 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v1442 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v1443 = stablehlo.multiply %v1441, %v1442 : tensor<256x38809xf32>
    %v1444 = stablehlo.reshape %v1443 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v1445 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1446 = stablehlo.exponential %v1444 : tensor<256x197x197xf32>
    %v1447 = stablehlo.reduce(%v1446 init: %v1445) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1448 = stablehlo.broadcast_in_dim %v1447, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v1449 = stablehlo.divide %v1446, %v1448 : tensor<256x197x197xf32>
    %v1450 = stablehlo.reshape %v1449 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v1451 = stablehlo.reshape %v1450 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v1452 = stablehlo.reshape %v1434 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1453 = stablehlo.dot_general %v1451, %v1452, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v1454 = stablehlo.reshape %v1453 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1455 = stablehlo.reshape %v1454 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1456 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1457 = stablehlo.pad %v1455, %v1456, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x192xf32>
    %v1458 = stablehlo.reshape %v1457 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v1459 = stablehlo.reshape %v1415 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1460 = stablehlo.slice %v1459 [0:256, 0:197, 64:128] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v1461 = stablehlo.reshape %v1460 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1462 = stablehlo.reshape %v1420 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1463 = stablehlo.slice %v1462 [0:256, 0:197, 64:128] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v1464 = stablehlo.reshape %v1463 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1465 = stablehlo.reshape %v1425 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1466 = stablehlo.slice %v1465 [0:256, 0:197, 64:128] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v1467 = stablehlo.reshape %v1466 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1468 = stablehlo.reshape %v1464 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1469 = stablehlo.transpose %v1468, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v1470 = stablehlo.reshape %v1469 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v1471 = stablehlo.reshape %v1461 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1472 = stablehlo.reshape %v1470 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v1473 = stablehlo.dot_general %v1471, %v1472, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v1474 = stablehlo.reshape %v1473 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v1475 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v1476 = stablehlo.multiply %v1474, %v1475 : tensor<256x38809xf32>
    %v1477 = stablehlo.reshape %v1476 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v1478 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1479 = stablehlo.exponential %v1477 : tensor<256x197x197xf32>
    %v1480 = stablehlo.reduce(%v1479 init: %v1478) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1481 = stablehlo.broadcast_in_dim %v1480, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v1482 = stablehlo.divide %v1479, %v1481 : tensor<256x197x197xf32>
    %v1483 = stablehlo.reshape %v1482 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v1484 = stablehlo.reshape %v1483 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v1485 = stablehlo.reshape %v1467 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1486 = stablehlo.dot_general %v1484, %v1485, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v1487 = stablehlo.reshape %v1486 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1488 = stablehlo.reshape %v1487 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1489 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1490 = stablehlo.pad %v1488, %v1489, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x192xf32>
    %v1491 = stablehlo.reshape %v1490 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v1492 = stablehlo.add %v1458, %v1491 : tensor<256x37824xf32>
    %v1493 = stablehlo.reshape %v1415 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1494 = stablehlo.slice %v1493 [0:256, 0:197, 128:192] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v1495 = stablehlo.reshape %v1494 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1496 = stablehlo.reshape %v1420 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1497 = stablehlo.slice %v1496 [0:256, 0:197, 128:192] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v1498 = stablehlo.reshape %v1497 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1499 = stablehlo.reshape %v1425 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1500 = stablehlo.slice %v1499 [0:256, 0:197, 128:192] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v1501 = stablehlo.reshape %v1500 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1502 = stablehlo.reshape %v1498 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1503 = stablehlo.transpose %v1502, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v1504 = stablehlo.reshape %v1503 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v1505 = stablehlo.reshape %v1495 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1506 = stablehlo.reshape %v1504 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v1507 = stablehlo.dot_general %v1505, %v1506, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v1508 = stablehlo.reshape %v1507 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v1509 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v1510 = stablehlo.multiply %v1508, %v1509 : tensor<256x38809xf32>
    %v1511 = stablehlo.reshape %v1510 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v1512 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1513 = stablehlo.exponential %v1511 : tensor<256x197x197xf32>
    %v1514 = stablehlo.reduce(%v1513 init: %v1512) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1515 = stablehlo.broadcast_in_dim %v1514, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v1516 = stablehlo.divide %v1513, %v1515 : tensor<256x197x197xf32>
    %v1517 = stablehlo.reshape %v1516 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v1518 = stablehlo.reshape %v1517 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v1519 = stablehlo.reshape %v1501 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1520 = stablehlo.dot_general %v1518, %v1519, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v1521 = stablehlo.reshape %v1520 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1522 = stablehlo.reshape %v1521 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1523 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1524 = stablehlo.pad %v1522, %v1523, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x192xf32>
    %v1525 = stablehlo.reshape %v1524 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v1526 = stablehlo.add %v1492, %v1525 : tensor<256x37824xf32>
    %v1527 = stablehlo.reshape %v1526 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1528 = stablehlo.dot_general %v1527, %b7_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x192xf32>, tensor<192x192xf32>) -> tensor<256x197x192xf32>
    %v1529 = stablehlo.broadcast_in_dim %b7_bo, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v1530 = stablehlo.add %v1528, %v1529 : tensor<256x197x192xf32>
    %v1531 = stablehlo.reshape %v1530 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v1532 = stablehlo.add %v1382, %v1531 : tensor<256x37824xf32>
    %v1533 = stablehlo.reshape %v1532 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1534 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1535 = stablehlo.constant dense<192.0> : tensor<256x197x192xf32>
    %v1536 = stablehlo.constant dense<1.0e-5> : tensor<256x197x192xf32>
    %v1537 = stablehlo.reduce(%v1533 init: %v1534) applies stablehlo.add across dimensions = [2] : (tensor<256x197x192xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1538 = stablehlo.broadcast_in_dim %v1537, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x192xf32>
    %v1539 = stablehlo.divide %v1538, %v1535 : tensor<256x197x192xf32>
    %v1540 = stablehlo.subtract %v1533, %v1539 : tensor<256x197x192xf32>
    %v1541 = stablehlo.multiply %v1540, %v1540 : tensor<256x197x192xf32>
    %v1542 = stablehlo.reduce(%v1541 init: %v1534) applies stablehlo.add across dimensions = [2] : (tensor<256x197x192xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1543 = stablehlo.broadcast_in_dim %v1542, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x192xf32>
    %v1544 = stablehlo.divide %v1543, %v1535 : tensor<256x197x192xf32>
    %v1545 = stablehlo.add %v1544, %v1536 : tensor<256x197x192xf32>
    %v1546 = stablehlo.rsqrt %v1545 : tensor<256x197x192xf32>
    %v1547 = stablehlo.multiply %v1540, %v1546 : tensor<256x197x192xf32>
    %v1548 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<256x197x192xf32>
    %v1549 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<256x197x192xf32>
    %v1550 = stablehlo.multiply %v1547, %v1548 : tensor<256x197x192xf32>
    %v1551 = stablehlo.add %v1550, %v1549 : tensor<256x197x192xf32>
    %v1552 = stablehlo.reshape %v1551 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v1553 = stablehlo.reshape %v1552 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1554 = stablehlo.broadcast_in_dim %b7_g2, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v1555 = stablehlo.multiply %v1553, %v1554 : tensor<256x197x192xf32>
    %v1556 = stablehlo.reshape %v1555 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v1557 = stablehlo.reshape %v1556 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1558 = stablehlo.broadcast_in_dim %b7_bt2, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v1559 = stablehlo.add %v1557, %v1558 : tensor<256x197x192xf32>
    %v1560 = stablehlo.reshape %v1559 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v1561 = stablehlo.reshape %v1560 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1562 = stablehlo.dot_general %v1561, %b7_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x192xf32>, tensor<192x768xf32>) -> tensor<256x197x768xf32>
    %v1563 = stablehlo.broadcast_in_dim %b7_bfc1, dims = [2] : (tensor<768xf32>) -> tensor<256x197x768xf32>
    %v1564 = stablehlo.add %v1562, %v1563 : tensor<256x197x768xf32>
    %v1565 = stablehlo.reshape %v1564 : (tensor<256x197x768xf32>) -> tensor<256x151296xf32>
    %v1566 = stablehlo.constant dense<0.5> : tensor<256x151296xf32>
    %v1567 = stablehlo.multiply %v1566, %v1565 : tensor<256x151296xf32>
    %v1568 = stablehlo.negate %v1565 : tensor<256x151296xf32>
    %v1569 = stablehlo.constant dense<0.7071067811865476> : tensor<256x151296xf32>
    %v1570 = stablehlo.multiply %v1568, %v1569 : tensor<256x151296xf32>
    %v1571 = chlo.erfc %v1570 : tensor<256x151296xf32> -> tensor<256x151296xf32>
    %v1572 = stablehlo.multiply %v1567, %v1571 : tensor<256x151296xf32>
    %v1573 = stablehlo.reshape %v1572 : (tensor<256x151296xf32>) -> tensor<256x197x768xf32>
    %v1574 = stablehlo.dot_general %v1573, %b7_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x768xf32>, tensor<768x192xf32>) -> tensor<256x197x192xf32>
    %v1575 = stablehlo.broadcast_in_dim %b7_bfc2, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v1576 = stablehlo.add %v1574, %v1575 : tensor<256x197x192xf32>
    %v1577 = stablehlo.reshape %v1576 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v1578 = stablehlo.add %v1532, %v1577 : tensor<256x37824xf32>
    %v1579 = stablehlo.reshape %v1578 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1580 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1581 = stablehlo.constant dense<192.0> : tensor<256x197x192xf32>
    %v1582 = stablehlo.constant dense<1.0e-5> : tensor<256x197x192xf32>
    %v1583 = stablehlo.reduce(%v1579 init: %v1580) applies stablehlo.add across dimensions = [2] : (tensor<256x197x192xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1584 = stablehlo.broadcast_in_dim %v1583, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x192xf32>
    %v1585 = stablehlo.divide %v1584, %v1581 : tensor<256x197x192xf32>
    %v1586 = stablehlo.subtract %v1579, %v1585 : tensor<256x197x192xf32>
    %v1587 = stablehlo.multiply %v1586, %v1586 : tensor<256x197x192xf32>
    %v1588 = stablehlo.reduce(%v1587 init: %v1580) applies stablehlo.add across dimensions = [2] : (tensor<256x197x192xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1589 = stablehlo.broadcast_in_dim %v1588, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x192xf32>
    %v1590 = stablehlo.divide %v1589, %v1581 : tensor<256x197x192xf32>
    %v1591 = stablehlo.add %v1590, %v1582 : tensor<256x197x192xf32>
    %v1592 = stablehlo.rsqrt %v1591 : tensor<256x197x192xf32>
    %v1593 = stablehlo.multiply %v1586, %v1592 : tensor<256x197x192xf32>
    %v1594 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<256x197x192xf32>
    %v1595 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<256x197x192xf32>
    %v1596 = stablehlo.multiply %v1593, %v1594 : tensor<256x197x192xf32>
    %v1597 = stablehlo.add %v1596, %v1595 : tensor<256x197x192xf32>
    %v1598 = stablehlo.reshape %v1597 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v1599 = stablehlo.reshape %v1598 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1600 = stablehlo.broadcast_in_dim %b8_g1, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v1601 = stablehlo.multiply %v1599, %v1600 : tensor<256x197x192xf32>
    %v1602 = stablehlo.reshape %v1601 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v1603 = stablehlo.reshape %v1602 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1604 = stablehlo.broadcast_in_dim %b8_bt1, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v1605 = stablehlo.add %v1603, %v1604 : tensor<256x197x192xf32>
    %v1606 = stablehlo.reshape %v1605 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v1607 = stablehlo.reshape %v1606 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1608 = stablehlo.dot_general %v1607, %b8_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x192xf32>, tensor<192x192xf32>) -> tensor<256x197x192xf32>
    %v1609 = stablehlo.broadcast_in_dim %b8_bq, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v1610 = stablehlo.add %v1608, %v1609 : tensor<256x197x192xf32>
    %v1611 = stablehlo.reshape %v1610 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v1612 = stablehlo.reshape %v1606 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1613 = stablehlo.dot_general %v1612, %b8_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x192xf32>, tensor<192x192xf32>) -> tensor<256x197x192xf32>
    %v1614 = stablehlo.broadcast_in_dim %b8_bk, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v1615 = stablehlo.add %v1613, %v1614 : tensor<256x197x192xf32>
    %v1616 = stablehlo.reshape %v1615 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v1617 = stablehlo.reshape %v1606 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1618 = stablehlo.dot_general %v1617, %b8_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x192xf32>, tensor<192x192xf32>) -> tensor<256x197x192xf32>
    %v1619 = stablehlo.broadcast_in_dim %b8_bv, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v1620 = stablehlo.add %v1618, %v1619 : tensor<256x197x192xf32>
    %v1621 = stablehlo.reshape %v1620 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v1622 = stablehlo.reshape %v1611 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1623 = stablehlo.slice %v1622 [0:256, 0:197, 0:64] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v1624 = stablehlo.reshape %v1623 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1625 = stablehlo.reshape %v1616 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1626 = stablehlo.slice %v1625 [0:256, 0:197, 0:64] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v1627 = stablehlo.reshape %v1626 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1628 = stablehlo.reshape %v1621 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1629 = stablehlo.slice %v1628 [0:256, 0:197, 0:64] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v1630 = stablehlo.reshape %v1629 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1631 = stablehlo.reshape %v1627 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1632 = stablehlo.transpose %v1631, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v1633 = stablehlo.reshape %v1632 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v1634 = stablehlo.reshape %v1624 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1635 = stablehlo.reshape %v1633 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v1636 = stablehlo.dot_general %v1634, %v1635, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v1637 = stablehlo.reshape %v1636 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v1638 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v1639 = stablehlo.multiply %v1637, %v1638 : tensor<256x38809xf32>
    %v1640 = stablehlo.reshape %v1639 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v1641 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1642 = stablehlo.exponential %v1640 : tensor<256x197x197xf32>
    %v1643 = stablehlo.reduce(%v1642 init: %v1641) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1644 = stablehlo.broadcast_in_dim %v1643, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v1645 = stablehlo.divide %v1642, %v1644 : tensor<256x197x197xf32>
    %v1646 = stablehlo.reshape %v1645 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v1647 = stablehlo.reshape %v1646 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v1648 = stablehlo.reshape %v1630 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1649 = stablehlo.dot_general %v1647, %v1648, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v1650 = stablehlo.reshape %v1649 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1651 = stablehlo.reshape %v1650 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1652 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1653 = stablehlo.pad %v1651, %v1652, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x192xf32>
    %v1654 = stablehlo.reshape %v1653 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v1655 = stablehlo.reshape %v1611 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1656 = stablehlo.slice %v1655 [0:256, 0:197, 64:128] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v1657 = stablehlo.reshape %v1656 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1658 = stablehlo.reshape %v1616 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1659 = stablehlo.slice %v1658 [0:256, 0:197, 64:128] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v1660 = stablehlo.reshape %v1659 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1661 = stablehlo.reshape %v1621 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1662 = stablehlo.slice %v1661 [0:256, 0:197, 64:128] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v1663 = stablehlo.reshape %v1662 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1664 = stablehlo.reshape %v1660 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1665 = stablehlo.transpose %v1664, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v1666 = stablehlo.reshape %v1665 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v1667 = stablehlo.reshape %v1657 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1668 = stablehlo.reshape %v1666 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v1669 = stablehlo.dot_general %v1667, %v1668, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v1670 = stablehlo.reshape %v1669 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v1671 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v1672 = stablehlo.multiply %v1670, %v1671 : tensor<256x38809xf32>
    %v1673 = stablehlo.reshape %v1672 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v1674 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1675 = stablehlo.exponential %v1673 : tensor<256x197x197xf32>
    %v1676 = stablehlo.reduce(%v1675 init: %v1674) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1677 = stablehlo.broadcast_in_dim %v1676, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v1678 = stablehlo.divide %v1675, %v1677 : tensor<256x197x197xf32>
    %v1679 = stablehlo.reshape %v1678 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v1680 = stablehlo.reshape %v1679 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v1681 = stablehlo.reshape %v1663 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1682 = stablehlo.dot_general %v1680, %v1681, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v1683 = stablehlo.reshape %v1682 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1684 = stablehlo.reshape %v1683 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1685 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1686 = stablehlo.pad %v1684, %v1685, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x192xf32>
    %v1687 = stablehlo.reshape %v1686 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v1688 = stablehlo.add %v1654, %v1687 : tensor<256x37824xf32>
    %v1689 = stablehlo.reshape %v1611 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1690 = stablehlo.slice %v1689 [0:256, 0:197, 128:192] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v1691 = stablehlo.reshape %v1690 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1692 = stablehlo.reshape %v1616 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1693 = stablehlo.slice %v1692 [0:256, 0:197, 128:192] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v1694 = stablehlo.reshape %v1693 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1695 = stablehlo.reshape %v1621 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1696 = stablehlo.slice %v1695 [0:256, 0:197, 128:192] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v1697 = stablehlo.reshape %v1696 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1698 = stablehlo.reshape %v1694 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1699 = stablehlo.transpose %v1698, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v1700 = stablehlo.reshape %v1699 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v1701 = stablehlo.reshape %v1691 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1702 = stablehlo.reshape %v1700 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v1703 = stablehlo.dot_general %v1701, %v1702, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v1704 = stablehlo.reshape %v1703 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v1705 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v1706 = stablehlo.multiply %v1704, %v1705 : tensor<256x38809xf32>
    %v1707 = stablehlo.reshape %v1706 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v1708 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1709 = stablehlo.exponential %v1707 : tensor<256x197x197xf32>
    %v1710 = stablehlo.reduce(%v1709 init: %v1708) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1711 = stablehlo.broadcast_in_dim %v1710, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v1712 = stablehlo.divide %v1709, %v1711 : tensor<256x197x197xf32>
    %v1713 = stablehlo.reshape %v1712 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v1714 = stablehlo.reshape %v1713 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v1715 = stablehlo.reshape %v1697 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1716 = stablehlo.dot_general %v1714, %v1715, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v1717 = stablehlo.reshape %v1716 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1718 = stablehlo.reshape %v1717 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1719 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1720 = stablehlo.pad %v1718, %v1719, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x192xf32>
    %v1721 = stablehlo.reshape %v1720 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v1722 = stablehlo.add %v1688, %v1721 : tensor<256x37824xf32>
    %v1723 = stablehlo.reshape %v1722 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1724 = stablehlo.dot_general %v1723, %b8_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x192xf32>, tensor<192x192xf32>) -> tensor<256x197x192xf32>
    %v1725 = stablehlo.broadcast_in_dim %b8_bo, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v1726 = stablehlo.add %v1724, %v1725 : tensor<256x197x192xf32>
    %v1727 = stablehlo.reshape %v1726 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v1728 = stablehlo.add %v1578, %v1727 : tensor<256x37824xf32>
    %v1729 = stablehlo.reshape %v1728 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1730 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1731 = stablehlo.constant dense<192.0> : tensor<256x197x192xf32>
    %v1732 = stablehlo.constant dense<1.0e-5> : tensor<256x197x192xf32>
    %v1733 = stablehlo.reduce(%v1729 init: %v1730) applies stablehlo.add across dimensions = [2] : (tensor<256x197x192xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1734 = stablehlo.broadcast_in_dim %v1733, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x192xf32>
    %v1735 = stablehlo.divide %v1734, %v1731 : tensor<256x197x192xf32>
    %v1736 = stablehlo.subtract %v1729, %v1735 : tensor<256x197x192xf32>
    %v1737 = stablehlo.multiply %v1736, %v1736 : tensor<256x197x192xf32>
    %v1738 = stablehlo.reduce(%v1737 init: %v1730) applies stablehlo.add across dimensions = [2] : (tensor<256x197x192xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1739 = stablehlo.broadcast_in_dim %v1738, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x192xf32>
    %v1740 = stablehlo.divide %v1739, %v1731 : tensor<256x197x192xf32>
    %v1741 = stablehlo.add %v1740, %v1732 : tensor<256x197x192xf32>
    %v1742 = stablehlo.rsqrt %v1741 : tensor<256x197x192xf32>
    %v1743 = stablehlo.multiply %v1736, %v1742 : tensor<256x197x192xf32>
    %v1744 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<256x197x192xf32>
    %v1745 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<256x197x192xf32>
    %v1746 = stablehlo.multiply %v1743, %v1744 : tensor<256x197x192xf32>
    %v1747 = stablehlo.add %v1746, %v1745 : tensor<256x197x192xf32>
    %v1748 = stablehlo.reshape %v1747 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v1749 = stablehlo.reshape %v1748 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1750 = stablehlo.broadcast_in_dim %b8_g2, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v1751 = stablehlo.multiply %v1749, %v1750 : tensor<256x197x192xf32>
    %v1752 = stablehlo.reshape %v1751 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v1753 = stablehlo.reshape %v1752 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1754 = stablehlo.broadcast_in_dim %b8_bt2, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v1755 = stablehlo.add %v1753, %v1754 : tensor<256x197x192xf32>
    %v1756 = stablehlo.reshape %v1755 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v1757 = stablehlo.reshape %v1756 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1758 = stablehlo.dot_general %v1757, %b8_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x192xf32>, tensor<192x768xf32>) -> tensor<256x197x768xf32>
    %v1759 = stablehlo.broadcast_in_dim %b8_bfc1, dims = [2] : (tensor<768xf32>) -> tensor<256x197x768xf32>
    %v1760 = stablehlo.add %v1758, %v1759 : tensor<256x197x768xf32>
    %v1761 = stablehlo.reshape %v1760 : (tensor<256x197x768xf32>) -> tensor<256x151296xf32>
    %v1762 = stablehlo.constant dense<0.5> : tensor<256x151296xf32>
    %v1763 = stablehlo.multiply %v1762, %v1761 : tensor<256x151296xf32>
    %v1764 = stablehlo.negate %v1761 : tensor<256x151296xf32>
    %v1765 = stablehlo.constant dense<0.7071067811865476> : tensor<256x151296xf32>
    %v1766 = stablehlo.multiply %v1764, %v1765 : tensor<256x151296xf32>
    %v1767 = chlo.erfc %v1766 : tensor<256x151296xf32> -> tensor<256x151296xf32>
    %v1768 = stablehlo.multiply %v1763, %v1767 : tensor<256x151296xf32>
    %v1769 = stablehlo.reshape %v1768 : (tensor<256x151296xf32>) -> tensor<256x197x768xf32>
    %v1770 = stablehlo.dot_general %v1769, %b8_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x768xf32>, tensor<768x192xf32>) -> tensor<256x197x192xf32>
    %v1771 = stablehlo.broadcast_in_dim %b8_bfc2, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v1772 = stablehlo.add %v1770, %v1771 : tensor<256x197x192xf32>
    %v1773 = stablehlo.reshape %v1772 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v1774 = stablehlo.add %v1728, %v1773 : tensor<256x37824xf32>
    %v1775 = stablehlo.reshape %v1774 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1776 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1777 = stablehlo.constant dense<192.0> : tensor<256x197x192xf32>
    %v1778 = stablehlo.constant dense<1.0e-5> : tensor<256x197x192xf32>
    %v1779 = stablehlo.reduce(%v1775 init: %v1776) applies stablehlo.add across dimensions = [2] : (tensor<256x197x192xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1780 = stablehlo.broadcast_in_dim %v1779, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x192xf32>
    %v1781 = stablehlo.divide %v1780, %v1777 : tensor<256x197x192xf32>
    %v1782 = stablehlo.subtract %v1775, %v1781 : tensor<256x197x192xf32>
    %v1783 = stablehlo.multiply %v1782, %v1782 : tensor<256x197x192xf32>
    %v1784 = stablehlo.reduce(%v1783 init: %v1776) applies stablehlo.add across dimensions = [2] : (tensor<256x197x192xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1785 = stablehlo.broadcast_in_dim %v1784, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x192xf32>
    %v1786 = stablehlo.divide %v1785, %v1777 : tensor<256x197x192xf32>
    %v1787 = stablehlo.add %v1786, %v1778 : tensor<256x197x192xf32>
    %v1788 = stablehlo.rsqrt %v1787 : tensor<256x197x192xf32>
    %v1789 = stablehlo.multiply %v1782, %v1788 : tensor<256x197x192xf32>
    %v1790 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<256x197x192xf32>
    %v1791 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<256x197x192xf32>
    %v1792 = stablehlo.multiply %v1789, %v1790 : tensor<256x197x192xf32>
    %v1793 = stablehlo.add %v1792, %v1791 : tensor<256x197x192xf32>
    %v1794 = stablehlo.reshape %v1793 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v1795 = stablehlo.reshape %v1794 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1796 = stablehlo.broadcast_in_dim %b9_g1, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v1797 = stablehlo.multiply %v1795, %v1796 : tensor<256x197x192xf32>
    %v1798 = stablehlo.reshape %v1797 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v1799 = stablehlo.reshape %v1798 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1800 = stablehlo.broadcast_in_dim %b9_bt1, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v1801 = stablehlo.add %v1799, %v1800 : tensor<256x197x192xf32>
    %v1802 = stablehlo.reshape %v1801 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v1803 = stablehlo.reshape %v1802 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1804 = stablehlo.dot_general %v1803, %b9_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x192xf32>, tensor<192x192xf32>) -> tensor<256x197x192xf32>
    %v1805 = stablehlo.broadcast_in_dim %b9_bq, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v1806 = stablehlo.add %v1804, %v1805 : tensor<256x197x192xf32>
    %v1807 = stablehlo.reshape %v1806 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v1808 = stablehlo.reshape %v1802 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1809 = stablehlo.dot_general %v1808, %b9_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x192xf32>, tensor<192x192xf32>) -> tensor<256x197x192xf32>
    %v1810 = stablehlo.broadcast_in_dim %b9_bk, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v1811 = stablehlo.add %v1809, %v1810 : tensor<256x197x192xf32>
    %v1812 = stablehlo.reshape %v1811 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v1813 = stablehlo.reshape %v1802 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1814 = stablehlo.dot_general %v1813, %b9_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x192xf32>, tensor<192x192xf32>) -> tensor<256x197x192xf32>
    %v1815 = stablehlo.broadcast_in_dim %b9_bv, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v1816 = stablehlo.add %v1814, %v1815 : tensor<256x197x192xf32>
    %v1817 = stablehlo.reshape %v1816 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v1818 = stablehlo.reshape %v1807 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1819 = stablehlo.slice %v1818 [0:256, 0:197, 0:64] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v1820 = stablehlo.reshape %v1819 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1821 = stablehlo.reshape %v1812 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1822 = stablehlo.slice %v1821 [0:256, 0:197, 0:64] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v1823 = stablehlo.reshape %v1822 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1824 = stablehlo.reshape %v1817 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1825 = stablehlo.slice %v1824 [0:256, 0:197, 0:64] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v1826 = stablehlo.reshape %v1825 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1827 = stablehlo.reshape %v1823 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1828 = stablehlo.transpose %v1827, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v1829 = stablehlo.reshape %v1828 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v1830 = stablehlo.reshape %v1820 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1831 = stablehlo.reshape %v1829 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v1832 = stablehlo.dot_general %v1830, %v1831, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v1833 = stablehlo.reshape %v1832 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v1834 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v1835 = stablehlo.multiply %v1833, %v1834 : tensor<256x38809xf32>
    %v1836 = stablehlo.reshape %v1835 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v1837 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1838 = stablehlo.exponential %v1836 : tensor<256x197x197xf32>
    %v1839 = stablehlo.reduce(%v1838 init: %v1837) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1840 = stablehlo.broadcast_in_dim %v1839, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v1841 = stablehlo.divide %v1838, %v1840 : tensor<256x197x197xf32>
    %v1842 = stablehlo.reshape %v1841 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v1843 = stablehlo.reshape %v1842 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v1844 = stablehlo.reshape %v1826 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1845 = stablehlo.dot_general %v1843, %v1844, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v1846 = stablehlo.reshape %v1845 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1847 = stablehlo.reshape %v1846 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1848 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1849 = stablehlo.pad %v1847, %v1848, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x192xf32>
    %v1850 = stablehlo.reshape %v1849 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v1851 = stablehlo.reshape %v1807 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1852 = stablehlo.slice %v1851 [0:256, 0:197, 64:128] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v1853 = stablehlo.reshape %v1852 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1854 = stablehlo.reshape %v1812 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1855 = stablehlo.slice %v1854 [0:256, 0:197, 64:128] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v1856 = stablehlo.reshape %v1855 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1857 = stablehlo.reshape %v1817 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1858 = stablehlo.slice %v1857 [0:256, 0:197, 64:128] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v1859 = stablehlo.reshape %v1858 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1860 = stablehlo.reshape %v1856 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1861 = stablehlo.transpose %v1860, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v1862 = stablehlo.reshape %v1861 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v1863 = stablehlo.reshape %v1853 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1864 = stablehlo.reshape %v1862 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v1865 = stablehlo.dot_general %v1863, %v1864, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v1866 = stablehlo.reshape %v1865 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v1867 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v1868 = stablehlo.multiply %v1866, %v1867 : tensor<256x38809xf32>
    %v1869 = stablehlo.reshape %v1868 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v1870 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1871 = stablehlo.exponential %v1869 : tensor<256x197x197xf32>
    %v1872 = stablehlo.reduce(%v1871 init: %v1870) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1873 = stablehlo.broadcast_in_dim %v1872, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v1874 = stablehlo.divide %v1871, %v1873 : tensor<256x197x197xf32>
    %v1875 = stablehlo.reshape %v1874 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v1876 = stablehlo.reshape %v1875 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v1877 = stablehlo.reshape %v1859 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1878 = stablehlo.dot_general %v1876, %v1877, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v1879 = stablehlo.reshape %v1878 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1880 = stablehlo.reshape %v1879 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1881 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1882 = stablehlo.pad %v1880, %v1881, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x192xf32>
    %v1883 = stablehlo.reshape %v1882 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v1884 = stablehlo.add %v1850, %v1883 : tensor<256x37824xf32>
    %v1885 = stablehlo.reshape %v1807 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1886 = stablehlo.slice %v1885 [0:256, 0:197, 128:192] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v1887 = stablehlo.reshape %v1886 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1888 = stablehlo.reshape %v1812 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1889 = stablehlo.slice %v1888 [0:256, 0:197, 128:192] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v1890 = stablehlo.reshape %v1889 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1891 = stablehlo.reshape %v1817 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1892 = stablehlo.slice %v1891 [0:256, 0:197, 128:192] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v1893 = stablehlo.reshape %v1892 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1894 = stablehlo.reshape %v1890 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1895 = stablehlo.transpose %v1894, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v1896 = stablehlo.reshape %v1895 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v1897 = stablehlo.reshape %v1887 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1898 = stablehlo.reshape %v1896 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v1899 = stablehlo.dot_general %v1897, %v1898, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v1900 = stablehlo.reshape %v1899 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v1901 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v1902 = stablehlo.multiply %v1900, %v1901 : tensor<256x38809xf32>
    %v1903 = stablehlo.reshape %v1902 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v1904 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1905 = stablehlo.exponential %v1903 : tensor<256x197x197xf32>
    %v1906 = stablehlo.reduce(%v1905 init: %v1904) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1907 = stablehlo.broadcast_in_dim %v1906, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v1908 = stablehlo.divide %v1905, %v1907 : tensor<256x197x197xf32>
    %v1909 = stablehlo.reshape %v1908 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v1910 = stablehlo.reshape %v1909 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v1911 = stablehlo.reshape %v1893 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1912 = stablehlo.dot_general %v1910, %v1911, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v1913 = stablehlo.reshape %v1912 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1914 = stablehlo.reshape %v1913 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1915 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1916 = stablehlo.pad %v1914, %v1915, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x192xf32>
    %v1917 = stablehlo.reshape %v1916 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v1918 = stablehlo.add %v1884, %v1917 : tensor<256x37824xf32>
    %v1919 = stablehlo.reshape %v1918 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1920 = stablehlo.dot_general %v1919, %b9_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x192xf32>, tensor<192x192xf32>) -> tensor<256x197x192xf32>
    %v1921 = stablehlo.broadcast_in_dim %b9_bo, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v1922 = stablehlo.add %v1920, %v1921 : tensor<256x197x192xf32>
    %v1923 = stablehlo.reshape %v1922 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v1924 = stablehlo.add %v1774, %v1923 : tensor<256x37824xf32>
    %v1925 = stablehlo.reshape %v1924 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1926 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1927 = stablehlo.constant dense<192.0> : tensor<256x197x192xf32>
    %v1928 = stablehlo.constant dense<1.0e-5> : tensor<256x197x192xf32>
    %v1929 = stablehlo.reduce(%v1925 init: %v1926) applies stablehlo.add across dimensions = [2] : (tensor<256x197x192xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1930 = stablehlo.broadcast_in_dim %v1929, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x192xf32>
    %v1931 = stablehlo.divide %v1930, %v1927 : tensor<256x197x192xf32>
    %v1932 = stablehlo.subtract %v1925, %v1931 : tensor<256x197x192xf32>
    %v1933 = stablehlo.multiply %v1932, %v1932 : tensor<256x197x192xf32>
    %v1934 = stablehlo.reduce(%v1933 init: %v1926) applies stablehlo.add across dimensions = [2] : (tensor<256x197x192xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1935 = stablehlo.broadcast_in_dim %v1934, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x192xf32>
    %v1936 = stablehlo.divide %v1935, %v1927 : tensor<256x197x192xf32>
    %v1937 = stablehlo.add %v1936, %v1928 : tensor<256x197x192xf32>
    %v1938 = stablehlo.rsqrt %v1937 : tensor<256x197x192xf32>
    %v1939 = stablehlo.multiply %v1932, %v1938 : tensor<256x197x192xf32>
    %v1940 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<256x197x192xf32>
    %v1941 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<256x197x192xf32>
    %v1942 = stablehlo.multiply %v1939, %v1940 : tensor<256x197x192xf32>
    %v1943 = stablehlo.add %v1942, %v1941 : tensor<256x197x192xf32>
    %v1944 = stablehlo.reshape %v1943 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v1945 = stablehlo.reshape %v1944 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1946 = stablehlo.broadcast_in_dim %b9_g2, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v1947 = stablehlo.multiply %v1945, %v1946 : tensor<256x197x192xf32>
    %v1948 = stablehlo.reshape %v1947 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v1949 = stablehlo.reshape %v1948 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1950 = stablehlo.broadcast_in_dim %b9_bt2, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v1951 = stablehlo.add %v1949, %v1950 : tensor<256x197x192xf32>
    %v1952 = stablehlo.reshape %v1951 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v1953 = stablehlo.reshape %v1952 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1954 = stablehlo.dot_general %v1953, %b9_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x192xf32>, tensor<192x768xf32>) -> tensor<256x197x768xf32>
    %v1955 = stablehlo.broadcast_in_dim %b9_bfc1, dims = [2] : (tensor<768xf32>) -> tensor<256x197x768xf32>
    %v1956 = stablehlo.add %v1954, %v1955 : tensor<256x197x768xf32>
    %v1957 = stablehlo.reshape %v1956 : (tensor<256x197x768xf32>) -> tensor<256x151296xf32>
    %v1958 = stablehlo.constant dense<0.5> : tensor<256x151296xf32>
    %v1959 = stablehlo.multiply %v1958, %v1957 : tensor<256x151296xf32>
    %v1960 = stablehlo.negate %v1957 : tensor<256x151296xf32>
    %v1961 = stablehlo.constant dense<0.7071067811865476> : tensor<256x151296xf32>
    %v1962 = stablehlo.multiply %v1960, %v1961 : tensor<256x151296xf32>
    %v1963 = chlo.erfc %v1962 : tensor<256x151296xf32> -> tensor<256x151296xf32>
    %v1964 = stablehlo.multiply %v1959, %v1963 : tensor<256x151296xf32>
    %v1965 = stablehlo.reshape %v1964 : (tensor<256x151296xf32>) -> tensor<256x197x768xf32>
    %v1966 = stablehlo.dot_general %v1965, %b9_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x768xf32>, tensor<768x192xf32>) -> tensor<256x197x192xf32>
    %v1967 = stablehlo.broadcast_in_dim %b9_bfc2, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v1968 = stablehlo.add %v1966, %v1967 : tensor<256x197x192xf32>
    %v1969 = stablehlo.reshape %v1968 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v1970 = stablehlo.add %v1924, %v1969 : tensor<256x37824xf32>
    %v1971 = stablehlo.reshape %v1970 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1972 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1973 = stablehlo.constant dense<192.0> : tensor<256x197x192xf32>
    %v1974 = stablehlo.constant dense<1.0e-5> : tensor<256x197x192xf32>
    %v1975 = stablehlo.reduce(%v1971 init: %v1972) applies stablehlo.add across dimensions = [2] : (tensor<256x197x192xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1976 = stablehlo.broadcast_in_dim %v1975, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x192xf32>
    %v1977 = stablehlo.divide %v1976, %v1973 : tensor<256x197x192xf32>
    %v1978 = stablehlo.subtract %v1971, %v1977 : tensor<256x197x192xf32>
    %v1979 = stablehlo.multiply %v1978, %v1978 : tensor<256x197x192xf32>
    %v1980 = stablehlo.reduce(%v1979 init: %v1972) applies stablehlo.add across dimensions = [2] : (tensor<256x197x192xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1981 = stablehlo.broadcast_in_dim %v1980, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x192xf32>
    %v1982 = stablehlo.divide %v1981, %v1973 : tensor<256x197x192xf32>
    %v1983 = stablehlo.add %v1982, %v1974 : tensor<256x197x192xf32>
    %v1984 = stablehlo.rsqrt %v1983 : tensor<256x197x192xf32>
    %v1985 = stablehlo.multiply %v1978, %v1984 : tensor<256x197x192xf32>
    %v1986 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<256x197x192xf32>
    %v1987 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<256x197x192xf32>
    %v1988 = stablehlo.multiply %v1985, %v1986 : tensor<256x197x192xf32>
    %v1989 = stablehlo.add %v1988, %v1987 : tensor<256x197x192xf32>
    %v1990 = stablehlo.reshape %v1989 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v1991 = stablehlo.reshape %v1990 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1992 = stablehlo.broadcast_in_dim %b10_g1, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v1993 = stablehlo.multiply %v1991, %v1992 : tensor<256x197x192xf32>
    %v1994 = stablehlo.reshape %v1993 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v1995 = stablehlo.reshape %v1994 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v1996 = stablehlo.broadcast_in_dim %b10_bt1, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v1997 = stablehlo.add %v1995, %v1996 : tensor<256x197x192xf32>
    %v1998 = stablehlo.reshape %v1997 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v1999 = stablehlo.reshape %v1998 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v2000 = stablehlo.dot_general %v1999, %b10_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x192xf32>, tensor<192x192xf32>) -> tensor<256x197x192xf32>
    %v2001 = stablehlo.broadcast_in_dim %b10_bq, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v2002 = stablehlo.add %v2000, %v2001 : tensor<256x197x192xf32>
    %v2003 = stablehlo.reshape %v2002 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v2004 = stablehlo.reshape %v1998 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v2005 = stablehlo.dot_general %v2004, %b10_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x192xf32>, tensor<192x192xf32>) -> tensor<256x197x192xf32>
    %v2006 = stablehlo.broadcast_in_dim %b10_bk, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v2007 = stablehlo.add %v2005, %v2006 : tensor<256x197x192xf32>
    %v2008 = stablehlo.reshape %v2007 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v2009 = stablehlo.reshape %v1998 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v2010 = stablehlo.dot_general %v2009, %b10_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x192xf32>, tensor<192x192xf32>) -> tensor<256x197x192xf32>
    %v2011 = stablehlo.broadcast_in_dim %b10_bv, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v2012 = stablehlo.add %v2010, %v2011 : tensor<256x197x192xf32>
    %v2013 = stablehlo.reshape %v2012 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v2014 = stablehlo.reshape %v2003 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v2015 = stablehlo.slice %v2014 [0:256, 0:197, 0:64] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v2016 = stablehlo.reshape %v2015 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2017 = stablehlo.reshape %v2008 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v2018 = stablehlo.slice %v2017 [0:256, 0:197, 0:64] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v2019 = stablehlo.reshape %v2018 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2020 = stablehlo.reshape %v2013 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v2021 = stablehlo.slice %v2020 [0:256, 0:197, 0:64] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v2022 = stablehlo.reshape %v2021 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2023 = stablehlo.reshape %v2019 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2024 = stablehlo.transpose %v2023, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v2025 = stablehlo.reshape %v2024 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v2026 = stablehlo.reshape %v2016 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2027 = stablehlo.reshape %v2025 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v2028 = stablehlo.dot_general %v2026, %v2027, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v2029 = stablehlo.reshape %v2028 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v2030 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v2031 = stablehlo.multiply %v2029, %v2030 : tensor<256x38809xf32>
    %v2032 = stablehlo.reshape %v2031 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v2033 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2034 = stablehlo.exponential %v2032 : tensor<256x197x197xf32>
    %v2035 = stablehlo.reduce(%v2034 init: %v2033) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v2036 = stablehlo.broadcast_in_dim %v2035, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v2037 = stablehlo.divide %v2034, %v2036 : tensor<256x197x197xf32>
    %v2038 = stablehlo.reshape %v2037 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v2039 = stablehlo.reshape %v2038 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v2040 = stablehlo.reshape %v2022 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2041 = stablehlo.dot_general %v2039, %v2040, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v2042 = stablehlo.reshape %v2041 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2043 = stablehlo.reshape %v2042 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2044 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2045 = stablehlo.pad %v2043, %v2044, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x192xf32>
    %v2046 = stablehlo.reshape %v2045 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v2047 = stablehlo.reshape %v2003 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v2048 = stablehlo.slice %v2047 [0:256, 0:197, 64:128] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v2049 = stablehlo.reshape %v2048 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2050 = stablehlo.reshape %v2008 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v2051 = stablehlo.slice %v2050 [0:256, 0:197, 64:128] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v2052 = stablehlo.reshape %v2051 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2053 = stablehlo.reshape %v2013 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v2054 = stablehlo.slice %v2053 [0:256, 0:197, 64:128] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v2055 = stablehlo.reshape %v2054 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2056 = stablehlo.reshape %v2052 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2057 = stablehlo.transpose %v2056, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v2058 = stablehlo.reshape %v2057 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v2059 = stablehlo.reshape %v2049 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2060 = stablehlo.reshape %v2058 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v2061 = stablehlo.dot_general %v2059, %v2060, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v2062 = stablehlo.reshape %v2061 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v2063 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v2064 = stablehlo.multiply %v2062, %v2063 : tensor<256x38809xf32>
    %v2065 = stablehlo.reshape %v2064 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v2066 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2067 = stablehlo.exponential %v2065 : tensor<256x197x197xf32>
    %v2068 = stablehlo.reduce(%v2067 init: %v2066) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v2069 = stablehlo.broadcast_in_dim %v2068, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v2070 = stablehlo.divide %v2067, %v2069 : tensor<256x197x197xf32>
    %v2071 = stablehlo.reshape %v2070 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v2072 = stablehlo.reshape %v2071 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v2073 = stablehlo.reshape %v2055 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2074 = stablehlo.dot_general %v2072, %v2073, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v2075 = stablehlo.reshape %v2074 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2076 = stablehlo.reshape %v2075 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2077 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2078 = stablehlo.pad %v2076, %v2077, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x192xf32>
    %v2079 = stablehlo.reshape %v2078 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v2080 = stablehlo.add %v2046, %v2079 : tensor<256x37824xf32>
    %v2081 = stablehlo.reshape %v2003 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v2082 = stablehlo.slice %v2081 [0:256, 0:197, 128:192] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v2083 = stablehlo.reshape %v2082 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2084 = stablehlo.reshape %v2008 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v2085 = stablehlo.slice %v2084 [0:256, 0:197, 128:192] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v2086 = stablehlo.reshape %v2085 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2087 = stablehlo.reshape %v2013 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v2088 = stablehlo.slice %v2087 [0:256, 0:197, 128:192] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v2089 = stablehlo.reshape %v2088 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2090 = stablehlo.reshape %v2086 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2091 = stablehlo.transpose %v2090, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v2092 = stablehlo.reshape %v2091 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v2093 = stablehlo.reshape %v2083 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2094 = stablehlo.reshape %v2092 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v2095 = stablehlo.dot_general %v2093, %v2094, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v2096 = stablehlo.reshape %v2095 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v2097 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v2098 = stablehlo.multiply %v2096, %v2097 : tensor<256x38809xf32>
    %v2099 = stablehlo.reshape %v2098 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v2100 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2101 = stablehlo.exponential %v2099 : tensor<256x197x197xf32>
    %v2102 = stablehlo.reduce(%v2101 init: %v2100) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v2103 = stablehlo.broadcast_in_dim %v2102, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v2104 = stablehlo.divide %v2101, %v2103 : tensor<256x197x197xf32>
    %v2105 = stablehlo.reshape %v2104 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v2106 = stablehlo.reshape %v2105 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v2107 = stablehlo.reshape %v2089 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2108 = stablehlo.dot_general %v2106, %v2107, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v2109 = stablehlo.reshape %v2108 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2110 = stablehlo.reshape %v2109 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2111 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2112 = stablehlo.pad %v2110, %v2111, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x192xf32>
    %v2113 = stablehlo.reshape %v2112 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v2114 = stablehlo.add %v2080, %v2113 : tensor<256x37824xf32>
    %v2115 = stablehlo.reshape %v2114 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v2116 = stablehlo.dot_general %v2115, %b10_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x192xf32>, tensor<192x192xf32>) -> tensor<256x197x192xf32>
    %v2117 = stablehlo.broadcast_in_dim %b10_bo, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v2118 = stablehlo.add %v2116, %v2117 : tensor<256x197x192xf32>
    %v2119 = stablehlo.reshape %v2118 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v2120 = stablehlo.add %v1970, %v2119 : tensor<256x37824xf32>
    %v2121 = stablehlo.reshape %v2120 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v2122 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2123 = stablehlo.constant dense<192.0> : tensor<256x197x192xf32>
    %v2124 = stablehlo.constant dense<1.0e-5> : tensor<256x197x192xf32>
    %v2125 = stablehlo.reduce(%v2121 init: %v2122) applies stablehlo.add across dimensions = [2] : (tensor<256x197x192xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v2126 = stablehlo.broadcast_in_dim %v2125, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x192xf32>
    %v2127 = stablehlo.divide %v2126, %v2123 : tensor<256x197x192xf32>
    %v2128 = stablehlo.subtract %v2121, %v2127 : tensor<256x197x192xf32>
    %v2129 = stablehlo.multiply %v2128, %v2128 : tensor<256x197x192xf32>
    %v2130 = stablehlo.reduce(%v2129 init: %v2122) applies stablehlo.add across dimensions = [2] : (tensor<256x197x192xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v2131 = stablehlo.broadcast_in_dim %v2130, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x192xf32>
    %v2132 = stablehlo.divide %v2131, %v2123 : tensor<256x197x192xf32>
    %v2133 = stablehlo.add %v2132, %v2124 : tensor<256x197x192xf32>
    %v2134 = stablehlo.rsqrt %v2133 : tensor<256x197x192xf32>
    %v2135 = stablehlo.multiply %v2128, %v2134 : tensor<256x197x192xf32>
    %v2136 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<256x197x192xf32>
    %v2137 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<256x197x192xf32>
    %v2138 = stablehlo.multiply %v2135, %v2136 : tensor<256x197x192xf32>
    %v2139 = stablehlo.add %v2138, %v2137 : tensor<256x197x192xf32>
    %v2140 = stablehlo.reshape %v2139 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v2141 = stablehlo.reshape %v2140 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v2142 = stablehlo.broadcast_in_dim %b10_g2, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v2143 = stablehlo.multiply %v2141, %v2142 : tensor<256x197x192xf32>
    %v2144 = stablehlo.reshape %v2143 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v2145 = stablehlo.reshape %v2144 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v2146 = stablehlo.broadcast_in_dim %b10_bt2, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v2147 = stablehlo.add %v2145, %v2146 : tensor<256x197x192xf32>
    %v2148 = stablehlo.reshape %v2147 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v2149 = stablehlo.reshape %v2148 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v2150 = stablehlo.dot_general %v2149, %b10_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x192xf32>, tensor<192x768xf32>) -> tensor<256x197x768xf32>
    %v2151 = stablehlo.broadcast_in_dim %b10_bfc1, dims = [2] : (tensor<768xf32>) -> tensor<256x197x768xf32>
    %v2152 = stablehlo.add %v2150, %v2151 : tensor<256x197x768xf32>
    %v2153 = stablehlo.reshape %v2152 : (tensor<256x197x768xf32>) -> tensor<256x151296xf32>
    %v2154 = stablehlo.constant dense<0.5> : tensor<256x151296xf32>
    %v2155 = stablehlo.multiply %v2154, %v2153 : tensor<256x151296xf32>
    %v2156 = stablehlo.negate %v2153 : tensor<256x151296xf32>
    %v2157 = stablehlo.constant dense<0.7071067811865476> : tensor<256x151296xf32>
    %v2158 = stablehlo.multiply %v2156, %v2157 : tensor<256x151296xf32>
    %v2159 = chlo.erfc %v2158 : tensor<256x151296xf32> -> tensor<256x151296xf32>
    %v2160 = stablehlo.multiply %v2155, %v2159 : tensor<256x151296xf32>
    %v2161 = stablehlo.reshape %v2160 : (tensor<256x151296xf32>) -> tensor<256x197x768xf32>
    %v2162 = stablehlo.dot_general %v2161, %b10_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x768xf32>, tensor<768x192xf32>) -> tensor<256x197x192xf32>
    %v2163 = stablehlo.broadcast_in_dim %b10_bfc2, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v2164 = stablehlo.add %v2162, %v2163 : tensor<256x197x192xf32>
    %v2165 = stablehlo.reshape %v2164 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v2166 = stablehlo.add %v2120, %v2165 : tensor<256x37824xf32>
    %v2167 = stablehlo.reshape %v2166 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v2168 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2169 = stablehlo.constant dense<192.0> : tensor<256x197x192xf32>
    %v2170 = stablehlo.constant dense<1.0e-5> : tensor<256x197x192xf32>
    %v2171 = stablehlo.reduce(%v2167 init: %v2168) applies stablehlo.add across dimensions = [2] : (tensor<256x197x192xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v2172 = stablehlo.broadcast_in_dim %v2171, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x192xf32>
    %v2173 = stablehlo.divide %v2172, %v2169 : tensor<256x197x192xf32>
    %v2174 = stablehlo.subtract %v2167, %v2173 : tensor<256x197x192xf32>
    %v2175 = stablehlo.multiply %v2174, %v2174 : tensor<256x197x192xf32>
    %v2176 = stablehlo.reduce(%v2175 init: %v2168) applies stablehlo.add across dimensions = [2] : (tensor<256x197x192xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v2177 = stablehlo.broadcast_in_dim %v2176, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x192xf32>
    %v2178 = stablehlo.divide %v2177, %v2169 : tensor<256x197x192xf32>
    %v2179 = stablehlo.add %v2178, %v2170 : tensor<256x197x192xf32>
    %v2180 = stablehlo.rsqrt %v2179 : tensor<256x197x192xf32>
    %v2181 = stablehlo.multiply %v2174, %v2180 : tensor<256x197x192xf32>
    %v2182 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<256x197x192xf32>
    %v2183 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<256x197x192xf32>
    %v2184 = stablehlo.multiply %v2181, %v2182 : tensor<256x197x192xf32>
    %v2185 = stablehlo.add %v2184, %v2183 : tensor<256x197x192xf32>
    %v2186 = stablehlo.reshape %v2185 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v2187 = stablehlo.reshape %v2186 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v2188 = stablehlo.broadcast_in_dim %b11_g1, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v2189 = stablehlo.multiply %v2187, %v2188 : tensor<256x197x192xf32>
    %v2190 = stablehlo.reshape %v2189 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v2191 = stablehlo.reshape %v2190 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v2192 = stablehlo.broadcast_in_dim %b11_bt1, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v2193 = stablehlo.add %v2191, %v2192 : tensor<256x197x192xf32>
    %v2194 = stablehlo.reshape %v2193 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v2195 = stablehlo.reshape %v2194 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v2196 = stablehlo.dot_general %v2195, %b11_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x192xf32>, tensor<192x192xf32>) -> tensor<256x197x192xf32>
    %v2197 = stablehlo.broadcast_in_dim %b11_bq, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v2198 = stablehlo.add %v2196, %v2197 : tensor<256x197x192xf32>
    %v2199 = stablehlo.reshape %v2198 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v2200 = stablehlo.reshape %v2194 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v2201 = stablehlo.dot_general %v2200, %b11_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x192xf32>, tensor<192x192xf32>) -> tensor<256x197x192xf32>
    %v2202 = stablehlo.broadcast_in_dim %b11_bk, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v2203 = stablehlo.add %v2201, %v2202 : tensor<256x197x192xf32>
    %v2204 = stablehlo.reshape %v2203 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v2205 = stablehlo.reshape %v2194 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v2206 = stablehlo.dot_general %v2205, %b11_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x192xf32>, tensor<192x192xf32>) -> tensor<256x197x192xf32>
    %v2207 = stablehlo.broadcast_in_dim %b11_bv, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v2208 = stablehlo.add %v2206, %v2207 : tensor<256x197x192xf32>
    %v2209 = stablehlo.reshape %v2208 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v2210 = stablehlo.reshape %v2199 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v2211 = stablehlo.slice %v2210 [0:256, 0:197, 0:64] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v2212 = stablehlo.reshape %v2211 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2213 = stablehlo.reshape %v2204 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v2214 = stablehlo.slice %v2213 [0:256, 0:197, 0:64] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v2215 = stablehlo.reshape %v2214 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2216 = stablehlo.reshape %v2209 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v2217 = stablehlo.slice %v2216 [0:256, 0:197, 0:64] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v2218 = stablehlo.reshape %v2217 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2219 = stablehlo.reshape %v2215 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2220 = stablehlo.transpose %v2219, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v2221 = stablehlo.reshape %v2220 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v2222 = stablehlo.reshape %v2212 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2223 = stablehlo.reshape %v2221 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v2224 = stablehlo.dot_general %v2222, %v2223, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v2225 = stablehlo.reshape %v2224 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v2226 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v2227 = stablehlo.multiply %v2225, %v2226 : tensor<256x38809xf32>
    %v2228 = stablehlo.reshape %v2227 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v2229 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2230 = stablehlo.exponential %v2228 : tensor<256x197x197xf32>
    %v2231 = stablehlo.reduce(%v2230 init: %v2229) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v2232 = stablehlo.broadcast_in_dim %v2231, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v2233 = stablehlo.divide %v2230, %v2232 : tensor<256x197x197xf32>
    %v2234 = stablehlo.reshape %v2233 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v2235 = stablehlo.reshape %v2234 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v2236 = stablehlo.reshape %v2218 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2237 = stablehlo.dot_general %v2235, %v2236, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v2238 = stablehlo.reshape %v2237 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2239 = stablehlo.reshape %v2238 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2240 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2241 = stablehlo.pad %v2239, %v2240, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x192xf32>
    %v2242 = stablehlo.reshape %v2241 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v2243 = stablehlo.reshape %v2199 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v2244 = stablehlo.slice %v2243 [0:256, 0:197, 64:128] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v2245 = stablehlo.reshape %v2244 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2246 = stablehlo.reshape %v2204 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v2247 = stablehlo.slice %v2246 [0:256, 0:197, 64:128] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v2248 = stablehlo.reshape %v2247 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2249 = stablehlo.reshape %v2209 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v2250 = stablehlo.slice %v2249 [0:256, 0:197, 64:128] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v2251 = stablehlo.reshape %v2250 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2252 = stablehlo.reshape %v2248 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2253 = stablehlo.transpose %v2252, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v2254 = stablehlo.reshape %v2253 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v2255 = stablehlo.reshape %v2245 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2256 = stablehlo.reshape %v2254 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v2257 = stablehlo.dot_general %v2255, %v2256, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v2258 = stablehlo.reshape %v2257 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v2259 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v2260 = stablehlo.multiply %v2258, %v2259 : tensor<256x38809xf32>
    %v2261 = stablehlo.reshape %v2260 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v2262 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2263 = stablehlo.exponential %v2261 : tensor<256x197x197xf32>
    %v2264 = stablehlo.reduce(%v2263 init: %v2262) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v2265 = stablehlo.broadcast_in_dim %v2264, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v2266 = stablehlo.divide %v2263, %v2265 : tensor<256x197x197xf32>
    %v2267 = stablehlo.reshape %v2266 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v2268 = stablehlo.reshape %v2267 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v2269 = stablehlo.reshape %v2251 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2270 = stablehlo.dot_general %v2268, %v2269, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v2271 = stablehlo.reshape %v2270 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2272 = stablehlo.reshape %v2271 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2273 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2274 = stablehlo.pad %v2272, %v2273, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x192xf32>
    %v2275 = stablehlo.reshape %v2274 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v2276 = stablehlo.add %v2242, %v2275 : tensor<256x37824xf32>
    %v2277 = stablehlo.reshape %v2199 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v2278 = stablehlo.slice %v2277 [0:256, 0:197, 128:192] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v2279 = stablehlo.reshape %v2278 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2280 = stablehlo.reshape %v2204 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v2281 = stablehlo.slice %v2280 [0:256, 0:197, 128:192] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v2282 = stablehlo.reshape %v2281 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2283 = stablehlo.reshape %v2209 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v2284 = stablehlo.slice %v2283 [0:256, 0:197, 128:192] : (tensor<256x197x192xf32>) -> tensor<256x197x64xf32>
    %v2285 = stablehlo.reshape %v2284 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2286 = stablehlo.reshape %v2282 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2287 = stablehlo.transpose %v2286, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v2288 = stablehlo.reshape %v2287 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v2289 = stablehlo.reshape %v2279 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2290 = stablehlo.reshape %v2288 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v2291 = stablehlo.dot_general %v2289, %v2290, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v2292 = stablehlo.reshape %v2291 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v2293 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v2294 = stablehlo.multiply %v2292, %v2293 : tensor<256x38809xf32>
    %v2295 = stablehlo.reshape %v2294 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v2296 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2297 = stablehlo.exponential %v2295 : tensor<256x197x197xf32>
    %v2298 = stablehlo.reduce(%v2297 init: %v2296) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v2299 = stablehlo.broadcast_in_dim %v2298, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v2300 = stablehlo.divide %v2297, %v2299 : tensor<256x197x197xf32>
    %v2301 = stablehlo.reshape %v2300 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v2302 = stablehlo.reshape %v2301 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v2303 = stablehlo.reshape %v2285 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2304 = stablehlo.dot_general %v2302, %v2303, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v2305 = stablehlo.reshape %v2304 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2306 = stablehlo.reshape %v2305 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2307 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2308 = stablehlo.pad %v2306, %v2307, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x192xf32>
    %v2309 = stablehlo.reshape %v2308 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v2310 = stablehlo.add %v2276, %v2309 : tensor<256x37824xf32>
    %v2311 = stablehlo.reshape %v2310 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v2312 = stablehlo.dot_general %v2311, %b11_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x192xf32>, tensor<192x192xf32>) -> tensor<256x197x192xf32>
    %v2313 = stablehlo.broadcast_in_dim %b11_bo, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v2314 = stablehlo.add %v2312, %v2313 : tensor<256x197x192xf32>
    %v2315 = stablehlo.reshape %v2314 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v2316 = stablehlo.add %v2166, %v2315 : tensor<256x37824xf32>
    %v2317 = stablehlo.reshape %v2316 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v2318 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2319 = stablehlo.constant dense<192.0> : tensor<256x197x192xf32>
    %v2320 = stablehlo.constant dense<1.0e-5> : tensor<256x197x192xf32>
    %v2321 = stablehlo.reduce(%v2317 init: %v2318) applies stablehlo.add across dimensions = [2] : (tensor<256x197x192xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v2322 = stablehlo.broadcast_in_dim %v2321, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x192xf32>
    %v2323 = stablehlo.divide %v2322, %v2319 : tensor<256x197x192xf32>
    %v2324 = stablehlo.subtract %v2317, %v2323 : tensor<256x197x192xf32>
    %v2325 = stablehlo.multiply %v2324, %v2324 : tensor<256x197x192xf32>
    %v2326 = stablehlo.reduce(%v2325 init: %v2318) applies stablehlo.add across dimensions = [2] : (tensor<256x197x192xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v2327 = stablehlo.broadcast_in_dim %v2326, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x192xf32>
    %v2328 = stablehlo.divide %v2327, %v2319 : tensor<256x197x192xf32>
    %v2329 = stablehlo.add %v2328, %v2320 : tensor<256x197x192xf32>
    %v2330 = stablehlo.rsqrt %v2329 : tensor<256x197x192xf32>
    %v2331 = stablehlo.multiply %v2324, %v2330 : tensor<256x197x192xf32>
    %v2332 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<256x197x192xf32>
    %v2333 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<256x197x192xf32>
    %v2334 = stablehlo.multiply %v2331, %v2332 : tensor<256x197x192xf32>
    %v2335 = stablehlo.add %v2334, %v2333 : tensor<256x197x192xf32>
    %v2336 = stablehlo.reshape %v2335 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v2337 = stablehlo.reshape %v2336 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v2338 = stablehlo.broadcast_in_dim %b11_g2, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v2339 = stablehlo.multiply %v2337, %v2338 : tensor<256x197x192xf32>
    %v2340 = stablehlo.reshape %v2339 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v2341 = stablehlo.reshape %v2340 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v2342 = stablehlo.broadcast_in_dim %b11_bt2, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v2343 = stablehlo.add %v2341, %v2342 : tensor<256x197x192xf32>
    %v2344 = stablehlo.reshape %v2343 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v2345 = stablehlo.reshape %v2344 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v2346 = stablehlo.dot_general %v2345, %b11_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x192xf32>, tensor<192x768xf32>) -> tensor<256x197x768xf32>
    %v2347 = stablehlo.broadcast_in_dim %b11_bfc1, dims = [2] : (tensor<768xf32>) -> tensor<256x197x768xf32>
    %v2348 = stablehlo.add %v2346, %v2347 : tensor<256x197x768xf32>
    %v2349 = stablehlo.reshape %v2348 : (tensor<256x197x768xf32>) -> tensor<256x151296xf32>
    %v2350 = stablehlo.constant dense<0.5> : tensor<256x151296xf32>
    %v2351 = stablehlo.multiply %v2350, %v2349 : tensor<256x151296xf32>
    %v2352 = stablehlo.negate %v2349 : tensor<256x151296xf32>
    %v2353 = stablehlo.constant dense<0.7071067811865476> : tensor<256x151296xf32>
    %v2354 = stablehlo.multiply %v2352, %v2353 : tensor<256x151296xf32>
    %v2355 = chlo.erfc %v2354 : tensor<256x151296xf32> -> tensor<256x151296xf32>
    %v2356 = stablehlo.multiply %v2351, %v2355 : tensor<256x151296xf32>
    %v2357 = stablehlo.reshape %v2356 : (tensor<256x151296xf32>) -> tensor<256x197x768xf32>
    %v2358 = stablehlo.dot_general %v2357, %b11_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x768xf32>, tensor<768x192xf32>) -> tensor<256x197x192xf32>
    %v2359 = stablehlo.broadcast_in_dim %b11_bfc2, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v2360 = stablehlo.add %v2358, %v2359 : tensor<256x197x192xf32>
    %v2361 = stablehlo.reshape %v2360 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v2362 = stablehlo.add %v2316, %v2361 : tensor<256x37824xf32>
    %v2363 = stablehlo.reshape %v2362 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v2364 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2365 = stablehlo.constant dense<192.0> : tensor<256x197x192xf32>
    %v2366 = stablehlo.constant dense<1.0e-5> : tensor<256x197x192xf32>
    %v2367 = stablehlo.reduce(%v2363 init: %v2364) applies stablehlo.add across dimensions = [2] : (tensor<256x197x192xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v2368 = stablehlo.broadcast_in_dim %v2367, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x192xf32>
    %v2369 = stablehlo.divide %v2368, %v2365 : tensor<256x197x192xf32>
    %v2370 = stablehlo.subtract %v2363, %v2369 : tensor<256x197x192xf32>
    %v2371 = stablehlo.multiply %v2370, %v2370 : tensor<256x197x192xf32>
    %v2372 = stablehlo.reduce(%v2371 init: %v2364) applies stablehlo.add across dimensions = [2] : (tensor<256x197x192xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v2373 = stablehlo.broadcast_in_dim %v2372, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x192xf32>
    %v2374 = stablehlo.divide %v2373, %v2365 : tensor<256x197x192xf32>
    %v2375 = stablehlo.add %v2374, %v2366 : tensor<256x197x192xf32>
    %v2376 = stablehlo.rsqrt %v2375 : tensor<256x197x192xf32>
    %v2377 = stablehlo.multiply %v2370, %v2376 : tensor<256x197x192xf32>
    %v2378 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<256x197x192xf32>
    %v2379 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<256x197x192xf32>
    %v2380 = stablehlo.multiply %v2377, %v2378 : tensor<256x197x192xf32>
    %v2381 = stablehlo.add %v2380, %v2379 : tensor<256x197x192xf32>
    %v2382 = stablehlo.reshape %v2381 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v2383 = stablehlo.reshape %v2382 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v2384 = stablehlo.broadcast_in_dim %gF, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v2385 = stablehlo.multiply %v2383, %v2384 : tensor<256x197x192xf32>
    %v2386 = stablehlo.reshape %v2385 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v2387 = stablehlo.reshape %v2386 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v2388 = stablehlo.broadcast_in_dim %btF, dims = [2] : (tensor<192xf32>) -> tensor<256x197x192xf32>
    %v2389 = stablehlo.add %v2387, %v2388 : tensor<256x197x192xf32>
    %v2390 = stablehlo.reshape %v2389 : (tensor<256x197x192xf32>) -> tensor<256x37824xf32>
    %v2391 = stablehlo.reshape %v2390 : (tensor<256x37824xf32>) -> tensor<256x197x192xf32>
    %v2392 = stablehlo.slice %v2391 [0:256, 0:1, 0:192] : (tensor<256x197x192xf32>) -> tensor<256x1x192xf32>
    %v2393 = stablehlo.reshape %v2392 : (tensor<256x1x192xf32>) -> tensor<256x192xf32>
    %v2394 = stablehlo.dot_general %v2393, %Wc, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x192xf32>, tensor<192x1000xf32>) -> tensor<256x1000xf32>
    %v2395 = stablehlo.broadcast_in_dim %bc, dims = [1] : (tensor<1000xf32>) -> tensor<256x1000xf32>
    %v2396 = stablehlo.add %v2394, %v2395 : tensor<256x1000xf32>
    return %v2396 : tensor<256x1000xf32>
  }
}

module @m {
  func.func @vit_fwd(%x: tensor<32x150528xf32>, %wConv: tensor<192x3x16x16xf32>, %bConv: tensor<192xf32>, %cls: tensor<192xf32>, %pos: tensor<197x192xf32>, %b0_g1: tensor<192xf32>, %b0_bt1: tensor<192xf32>, %b0_Wq: tensor<192x192xf32>, %b0_bq: tensor<192xf32>, %b0_Wk: tensor<192x192xf32>, %b0_bk: tensor<192xf32>, %b0_Wv: tensor<192x192xf32>, %b0_bv: tensor<192xf32>, %b0_Wo: tensor<192x192xf32>, %b0_bo: tensor<192xf32>, %b0_g2: tensor<192xf32>, %b0_bt2: tensor<192xf32>, %b0_Wfc1: tensor<192x768xf32>, %b0_bfc1: tensor<768xf32>, %b0_Wfc2: tensor<768x192xf32>, %b0_bfc2: tensor<192xf32>, %b1_g1: tensor<192xf32>, %b1_bt1: tensor<192xf32>, %b1_Wq: tensor<192x192xf32>, %b1_bq: tensor<192xf32>, %b1_Wk: tensor<192x192xf32>, %b1_bk: tensor<192xf32>, %b1_Wv: tensor<192x192xf32>, %b1_bv: tensor<192xf32>, %b1_Wo: tensor<192x192xf32>, %b1_bo: tensor<192xf32>, %b1_g2: tensor<192xf32>, %b1_bt2: tensor<192xf32>, %b1_Wfc1: tensor<192x768xf32>, %b1_bfc1: tensor<768xf32>, %b1_Wfc2: tensor<768x192xf32>, %b1_bfc2: tensor<192xf32>, %b2_g1: tensor<192xf32>, %b2_bt1: tensor<192xf32>, %b2_Wq: tensor<192x192xf32>, %b2_bq: tensor<192xf32>, %b2_Wk: tensor<192x192xf32>, %b2_bk: tensor<192xf32>, %b2_Wv: tensor<192x192xf32>, %b2_bv: tensor<192xf32>, %b2_Wo: tensor<192x192xf32>, %b2_bo: tensor<192xf32>, %b2_g2: tensor<192xf32>, %b2_bt2: tensor<192xf32>, %b2_Wfc1: tensor<192x768xf32>, %b2_bfc1: tensor<768xf32>, %b2_Wfc2: tensor<768x192xf32>, %b2_bfc2: tensor<192xf32>, %b3_g1: tensor<192xf32>, %b3_bt1: tensor<192xf32>, %b3_Wq: tensor<192x192xf32>, %b3_bq: tensor<192xf32>, %b3_Wk: tensor<192x192xf32>, %b3_bk: tensor<192xf32>, %b3_Wv: tensor<192x192xf32>, %b3_bv: tensor<192xf32>, %b3_Wo: tensor<192x192xf32>, %b3_bo: tensor<192xf32>, %b3_g2: tensor<192xf32>, %b3_bt2: tensor<192xf32>, %b3_Wfc1: tensor<192x768xf32>, %b3_bfc1: tensor<768xf32>, %b3_Wfc2: tensor<768x192xf32>, %b3_bfc2: tensor<192xf32>, %b4_g1: tensor<192xf32>, %b4_bt1: tensor<192xf32>, %b4_Wq: tensor<192x192xf32>, %b4_bq: tensor<192xf32>, %b4_Wk: tensor<192x192xf32>, %b4_bk: tensor<192xf32>, %b4_Wv: tensor<192x192xf32>, %b4_bv: tensor<192xf32>, %b4_Wo: tensor<192x192xf32>, %b4_bo: tensor<192xf32>, %b4_g2: tensor<192xf32>, %b4_bt2: tensor<192xf32>, %b4_Wfc1: tensor<192x768xf32>, %b4_bfc1: tensor<768xf32>, %b4_Wfc2: tensor<768x192xf32>, %b4_bfc2: tensor<192xf32>, %b5_g1: tensor<192xf32>, %b5_bt1: tensor<192xf32>, %b5_Wq: tensor<192x192xf32>, %b5_bq: tensor<192xf32>, %b5_Wk: tensor<192x192xf32>, %b5_bk: tensor<192xf32>, %b5_Wv: tensor<192x192xf32>, %b5_bv: tensor<192xf32>, %b5_Wo: tensor<192x192xf32>, %b5_bo: tensor<192xf32>, %b5_g2: tensor<192xf32>, %b5_bt2: tensor<192xf32>, %b5_Wfc1: tensor<192x768xf32>, %b5_bfc1: tensor<768xf32>, %b5_Wfc2: tensor<768x192xf32>, %b5_bfc2: tensor<192xf32>, %b6_g1: tensor<192xf32>, %b6_bt1: tensor<192xf32>, %b6_Wq: tensor<192x192xf32>, %b6_bq: tensor<192xf32>, %b6_Wk: tensor<192x192xf32>, %b6_bk: tensor<192xf32>, %b6_Wv: tensor<192x192xf32>, %b6_bv: tensor<192xf32>, %b6_Wo: tensor<192x192xf32>, %b6_bo: tensor<192xf32>, %b6_g2: tensor<192xf32>, %b6_bt2: tensor<192xf32>, %b6_Wfc1: tensor<192x768xf32>, %b6_bfc1: tensor<768xf32>, %b6_Wfc2: tensor<768x192xf32>, %b6_bfc2: tensor<192xf32>, %b7_g1: tensor<192xf32>, %b7_bt1: tensor<192xf32>, %b7_Wq: tensor<192x192xf32>, %b7_bq: tensor<192xf32>, %b7_Wk: tensor<192x192xf32>, %b7_bk: tensor<192xf32>, %b7_Wv: tensor<192x192xf32>, %b7_bv: tensor<192xf32>, %b7_Wo: tensor<192x192xf32>, %b7_bo: tensor<192xf32>, %b7_g2: tensor<192xf32>, %b7_bt2: tensor<192xf32>, %b7_Wfc1: tensor<192x768xf32>, %b7_bfc1: tensor<768xf32>, %b7_Wfc2: tensor<768x192xf32>, %b7_bfc2: tensor<192xf32>, %b8_g1: tensor<192xf32>, %b8_bt1: tensor<192xf32>, %b8_Wq: tensor<192x192xf32>, %b8_bq: tensor<192xf32>, %b8_Wk: tensor<192x192xf32>, %b8_bk: tensor<192xf32>, %b8_Wv: tensor<192x192xf32>, %b8_bv: tensor<192xf32>, %b8_Wo: tensor<192x192xf32>, %b8_bo: tensor<192xf32>, %b8_g2: tensor<192xf32>, %b8_bt2: tensor<192xf32>, %b8_Wfc1: tensor<192x768xf32>, %b8_bfc1: tensor<768xf32>, %b8_Wfc2: tensor<768x192xf32>, %b8_bfc2: tensor<192xf32>, %b9_g1: tensor<192xf32>, %b9_bt1: tensor<192xf32>, %b9_Wq: tensor<192x192xf32>, %b9_bq: tensor<192xf32>, %b9_Wk: tensor<192x192xf32>, %b9_bk: tensor<192xf32>, %b9_Wv: tensor<192x192xf32>, %b9_bv: tensor<192xf32>, %b9_Wo: tensor<192x192xf32>, %b9_bo: tensor<192xf32>, %b9_g2: tensor<192xf32>, %b9_bt2: tensor<192xf32>, %b9_Wfc1: tensor<192x768xf32>, %b9_bfc1: tensor<768xf32>, %b9_Wfc2: tensor<768x192xf32>, %b9_bfc2: tensor<192xf32>, %b10_g1: tensor<192xf32>, %b10_bt1: tensor<192xf32>, %b10_Wq: tensor<192x192xf32>, %b10_bq: tensor<192xf32>, %b10_Wk: tensor<192x192xf32>, %b10_bk: tensor<192xf32>, %b10_Wv: tensor<192x192xf32>, %b10_bv: tensor<192xf32>, %b10_Wo: tensor<192x192xf32>, %b10_bo: tensor<192xf32>, %b10_g2: tensor<192xf32>, %b10_bt2: tensor<192xf32>, %b10_Wfc1: tensor<192x768xf32>, %b10_bfc1: tensor<768xf32>, %b10_Wfc2: tensor<768x192xf32>, %b10_bfc2: tensor<192xf32>, %b11_g1: tensor<192xf32>, %b11_bt1: tensor<192xf32>, %b11_Wq: tensor<192x192xf32>, %b11_bq: tensor<192xf32>, %b11_Wk: tensor<192x192xf32>, %b11_bk: tensor<192xf32>, %b11_Wv: tensor<192x192xf32>, %b11_bv: tensor<192xf32>, %b11_Wo: tensor<192x192xf32>, %b11_bo: tensor<192xf32>, %b11_g2: tensor<192xf32>, %b11_bt2: tensor<192xf32>, %b11_Wfc1: tensor<192x768xf32>, %b11_bfc1: tensor<768xf32>, %b11_Wfc2: tensor<768x192xf32>, %b11_bfc2: tensor<192xf32>, %gF: tensor<192xf32>, %btF: tensor<192xf32>, %Wc: tensor<192x10xf32>, %bc: tensor<10xf32>) -> tensor<32x10xf32> {
    %one = stablehlo.constant dense<1.0> : tensor<f32>
    %zero = stablehlo.constant dense<0.0> : tensor<f32>
    %sc = stablehlo.constant dense<0.0> : tensor<f32>
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
    return %v2468 : tensor<32x10xf32>
  }
}

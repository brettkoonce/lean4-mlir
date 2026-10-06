module @m {
  func.func @vitbin_erf_fwd(%x: tensor<32x150528xf32>, %wConv: tensor<768x3x16x16xf32>, %bConv: tensor<768xf32>, %cls: tensor<768xf32>, %pos: tensor<197x768xf32>, %b0_g1: tensor<768xf32>, %b0_bt1: tensor<768xf32>, %b0_Wq: tensor<768x768xf32>, %b0_bq: tensor<768xf32>, %b0_Wk: tensor<768x768xf32>, %b0_bk: tensor<768xf32>, %b0_Wv: tensor<768x768xf32>, %b0_bv: tensor<768xf32>, %b0_Wo: tensor<768x768xf32>, %b0_bo: tensor<768xf32>, %b0_g2: tensor<768xf32>, %b0_bt2: tensor<768xf32>, %b0_Wfc1: tensor<768x3072xf32>, %b0_bfc1: tensor<3072xf32>, %b0_Wfc2: tensor<3072x768xf32>, %b0_bfc2: tensor<768xf32>, %b1_g1: tensor<768xf32>, %b1_bt1: tensor<768xf32>, %b1_Wq: tensor<768x768xf32>, %b1_bq: tensor<768xf32>, %b1_Wk: tensor<768x768xf32>, %b1_bk: tensor<768xf32>, %b1_Wv: tensor<768x768xf32>, %b1_bv: tensor<768xf32>, %b1_Wo: tensor<768x768xf32>, %b1_bo: tensor<768xf32>, %b1_g2: tensor<768xf32>, %b1_bt2: tensor<768xf32>, %b1_Wfc1: tensor<768x3072xf32>, %b1_bfc1: tensor<3072xf32>, %b1_Wfc2: tensor<3072x768xf32>, %b1_bfc2: tensor<768xf32>, %b2_g1: tensor<768xf32>, %b2_bt1: tensor<768xf32>, %b2_Wq: tensor<768x768xf32>, %b2_bq: tensor<768xf32>, %b2_Wk: tensor<768x768xf32>, %b2_bk: tensor<768xf32>, %b2_Wv: tensor<768x768xf32>, %b2_bv: tensor<768xf32>, %b2_Wo: tensor<768x768xf32>, %b2_bo: tensor<768xf32>, %b2_g2: tensor<768xf32>, %b2_bt2: tensor<768xf32>, %b2_Wfc1: tensor<768x3072xf32>, %b2_bfc1: tensor<3072xf32>, %b2_Wfc2: tensor<3072x768xf32>, %b2_bfc2: tensor<768xf32>, %b3_g1: tensor<768xf32>, %b3_bt1: tensor<768xf32>, %b3_Wq: tensor<768x768xf32>, %b3_bq: tensor<768xf32>, %b3_Wk: tensor<768x768xf32>, %b3_bk: tensor<768xf32>, %b3_Wv: tensor<768x768xf32>, %b3_bv: tensor<768xf32>, %b3_Wo: tensor<768x768xf32>, %b3_bo: tensor<768xf32>, %b3_g2: tensor<768xf32>, %b3_bt2: tensor<768xf32>, %b3_Wfc1: tensor<768x3072xf32>, %b3_bfc1: tensor<3072xf32>, %b3_Wfc2: tensor<3072x768xf32>, %b3_bfc2: tensor<768xf32>, %b4_g1: tensor<768xf32>, %b4_bt1: tensor<768xf32>, %b4_Wq: tensor<768x768xf32>, %b4_bq: tensor<768xf32>, %b4_Wk: tensor<768x768xf32>, %b4_bk: tensor<768xf32>, %b4_Wv: tensor<768x768xf32>, %b4_bv: tensor<768xf32>, %b4_Wo: tensor<768x768xf32>, %b4_bo: tensor<768xf32>, %b4_g2: tensor<768xf32>, %b4_bt2: tensor<768xf32>, %b4_Wfc1: tensor<768x3072xf32>, %b4_bfc1: tensor<3072xf32>, %b4_Wfc2: tensor<3072x768xf32>, %b4_bfc2: tensor<768xf32>, %b5_g1: tensor<768xf32>, %b5_bt1: tensor<768xf32>, %b5_Wq: tensor<768x768xf32>, %b5_bq: tensor<768xf32>, %b5_Wk: tensor<768x768xf32>, %b5_bk: tensor<768xf32>, %b5_Wv: tensor<768x768xf32>, %b5_bv: tensor<768xf32>, %b5_Wo: tensor<768x768xf32>, %b5_bo: tensor<768xf32>, %b5_g2: tensor<768xf32>, %b5_bt2: tensor<768xf32>, %b5_Wfc1: tensor<768x3072xf32>, %b5_bfc1: tensor<3072xf32>, %b5_Wfc2: tensor<3072x768xf32>, %b5_bfc2: tensor<768xf32>, %b6_g1: tensor<768xf32>, %b6_bt1: tensor<768xf32>, %b6_Wq: tensor<768x768xf32>, %b6_bq: tensor<768xf32>, %b6_Wk: tensor<768x768xf32>, %b6_bk: tensor<768xf32>, %b6_Wv: tensor<768x768xf32>, %b6_bv: tensor<768xf32>, %b6_Wo: tensor<768x768xf32>, %b6_bo: tensor<768xf32>, %b6_g2: tensor<768xf32>, %b6_bt2: tensor<768xf32>, %b6_Wfc1: tensor<768x3072xf32>, %b6_bfc1: tensor<3072xf32>, %b6_Wfc2: tensor<3072x768xf32>, %b6_bfc2: tensor<768xf32>, %b7_g1: tensor<768xf32>, %b7_bt1: tensor<768xf32>, %b7_Wq: tensor<768x768xf32>, %b7_bq: tensor<768xf32>, %b7_Wk: tensor<768x768xf32>, %b7_bk: tensor<768xf32>, %b7_Wv: tensor<768x768xf32>, %b7_bv: tensor<768xf32>, %b7_Wo: tensor<768x768xf32>, %b7_bo: tensor<768xf32>, %b7_g2: tensor<768xf32>, %b7_bt2: tensor<768xf32>, %b7_Wfc1: tensor<768x3072xf32>, %b7_bfc1: tensor<3072xf32>, %b7_Wfc2: tensor<3072x768xf32>, %b7_bfc2: tensor<768xf32>, %b8_g1: tensor<768xf32>, %b8_bt1: tensor<768xf32>, %b8_Wq: tensor<768x768xf32>, %b8_bq: tensor<768xf32>, %b8_Wk: tensor<768x768xf32>, %b8_bk: tensor<768xf32>, %b8_Wv: tensor<768x768xf32>, %b8_bv: tensor<768xf32>, %b8_Wo: tensor<768x768xf32>, %b8_bo: tensor<768xf32>, %b8_g2: tensor<768xf32>, %b8_bt2: tensor<768xf32>, %b8_Wfc1: tensor<768x3072xf32>, %b8_bfc1: tensor<3072xf32>, %b8_Wfc2: tensor<3072x768xf32>, %b8_bfc2: tensor<768xf32>, %b9_g1: tensor<768xf32>, %b9_bt1: tensor<768xf32>, %b9_Wq: tensor<768x768xf32>, %b9_bq: tensor<768xf32>, %b9_Wk: tensor<768x768xf32>, %b9_bk: tensor<768xf32>, %b9_Wv: tensor<768x768xf32>, %b9_bv: tensor<768xf32>, %b9_Wo: tensor<768x768xf32>, %b9_bo: tensor<768xf32>, %b9_g2: tensor<768xf32>, %b9_bt2: tensor<768xf32>, %b9_Wfc1: tensor<768x3072xf32>, %b9_bfc1: tensor<3072xf32>, %b9_Wfc2: tensor<3072x768xf32>, %b9_bfc2: tensor<768xf32>, %b10_g1: tensor<768xf32>, %b10_bt1: tensor<768xf32>, %b10_Wq: tensor<768x768xf32>, %b10_bq: tensor<768xf32>, %b10_Wk: tensor<768x768xf32>, %b10_bk: tensor<768xf32>, %b10_Wv: tensor<768x768xf32>, %b10_bv: tensor<768xf32>, %b10_Wo: tensor<768x768xf32>, %b10_bo: tensor<768xf32>, %b10_g2: tensor<768xf32>, %b10_bt2: tensor<768xf32>, %b10_Wfc1: tensor<768x3072xf32>, %b10_bfc1: tensor<3072xf32>, %b10_Wfc2: tensor<3072x768xf32>, %b10_bfc2: tensor<768xf32>, %b11_g1: tensor<768xf32>, %b11_bt1: tensor<768xf32>, %b11_Wq: tensor<768x768xf32>, %b11_bq: tensor<768xf32>, %b11_Wk: tensor<768x768xf32>, %b11_bk: tensor<768xf32>, %b11_Wv: tensor<768x768xf32>, %b11_bv: tensor<768xf32>, %b11_Wo: tensor<768x768xf32>, %b11_bo: tensor<768xf32>, %b11_g2: tensor<768xf32>, %b11_bt2: tensor<768xf32>, %b11_Wfc1: tensor<768x3072xf32>, %b11_bfc1: tensor<3072xf32>, %b11_Wfc2: tensor<3072x768xf32>, %b11_bfc2: tensor<768xf32>, %gF: tensor<768xf32>, %btF: tensor<768xf32>, %Wc: tensor<768x1000xf32>, %bc: tensor<1000xf32>) -> tensor<32x1000xf32> {
    %one = stablehlo.constant dense<1.0> : tensor<f32>
    %zero = stablehlo.constant dense<0.0> : tensor<f32>
    %sc = stablehlo.constant dense<0.0> : tensor<f32>
    %v0 = stablehlo.reshape %x : (tensor<32x150528xf32>) -> tensor<32x3x224x224xf32>
    %v1 = stablehlo.convolution(%v0, %wConv)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [16, 16], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3x224x224xf32>, tensor<768x3x16x16xf32>) -> tensor<32x768x14x14xf32>
    %v2 = stablehlo.broadcast_in_dim %bConv, dims = [1] : (tensor<768xf32>) -> tensor<32x768x14x14xf32>
    %v3 = stablehlo.add %v1, %v2 : tensor<32x768x14x14xf32>
    %v4 = stablehlo.transpose %v3, dims = [0, 2, 3, 1] : (tensor<32x768x14x14xf32>) -> tensor<32x14x14x768xf32>
    %v5 = stablehlo.reshape %v4 : (tensor<32x14x14x768xf32>) -> tensor<32x196x768xf32>
    %v6 = stablehlo.broadcast_in_dim %cls, dims = [2] : (tensor<768xf32>) -> tensor<32x1x768xf32>
    %v7 = stablehlo.concatenate %v6, %v5, dim = 1 : (tensor<32x1x768xf32>, tensor<32x196x768xf32>) -> tensor<32x197x768xf32>
    %v8 = stablehlo.broadcast_in_dim %pos, dims = [1, 2] : (tensor<197x768xf32>) -> tensor<32x197x768xf32>
    %v9 = stablehlo.add %v7, %v8 : tensor<32x197x768xf32>
    %v10 = stablehlo.reshape %v9 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v11 = stablehlo.reshape %v10 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v12 = stablehlo.constant dense<0.0> : tensor<f32>
    %v13 = stablehlo.constant dense<768.0> : tensor<32x197x768xf32>
    %v14 = stablehlo.constant dense<1.0e-5> : tensor<32x197x768xf32>
    %v15 = stablehlo.reduce(%v11 init: %v12) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v16 = stablehlo.broadcast_in_dim %v15, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v17 = stablehlo.divide %v16, %v13 : tensor<32x197x768xf32>
    %v18 = stablehlo.subtract %v11, %v17 : tensor<32x197x768xf32>
    %v19 = stablehlo.multiply %v18, %v18 : tensor<32x197x768xf32>
    %v20 = stablehlo.reduce(%v19 init: %v12) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v21 = stablehlo.broadcast_in_dim %v20, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v22 = stablehlo.divide %v21, %v13 : tensor<32x197x768xf32>
    %v23 = stablehlo.add %v22, %v14 : tensor<32x197x768xf32>
    %v24 = stablehlo.rsqrt %v23 : tensor<32x197x768xf32>
    %v25 = stablehlo.multiply %v18, %v24 : tensor<32x197x768xf32>
    %v26 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v27 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v28 = stablehlo.multiply %v25, %v26 : tensor<32x197x768xf32>
    %v29 = stablehlo.add %v28, %v27 : tensor<32x197x768xf32>
    %v30 = stablehlo.reshape %v29 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v31 = stablehlo.reshape %v30 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v32 = stablehlo.broadcast_in_dim %b0_g1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v33 = stablehlo.multiply %v31, %v32 : tensor<32x197x768xf32>
    %v34 = stablehlo.reshape %v33 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v35 = stablehlo.reshape %v34 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v36 = stablehlo.broadcast_in_dim %b0_bt1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v37 = stablehlo.add %v35, %v36 : tensor<32x197x768xf32>
    %v38 = stablehlo.reshape %v37 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v39 = stablehlo.reshape %v38 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v40 = stablehlo.dot_general %v39, %b0_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v41 = stablehlo.broadcast_in_dim %b0_bq, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v42 = stablehlo.add %v40, %v41 : tensor<32x197x768xf32>
    %v43 = stablehlo.reshape %v42 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v44 = stablehlo.reshape %v38 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v45 = stablehlo.dot_general %v44, %b0_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v46 = stablehlo.broadcast_in_dim %b0_bk, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v47 = stablehlo.add %v45, %v46 : tensor<32x197x768xf32>
    %v48 = stablehlo.reshape %v47 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v49 = stablehlo.reshape %v38 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v50 = stablehlo.dot_general %v49, %b0_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v51 = stablehlo.broadcast_in_dim %b0_bv, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v52 = stablehlo.add %v50, %v51 : tensor<32x197x768xf32>
    %v53 = stablehlo.reshape %v52 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v54 = stablehlo.reshape %v43 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v55 = stablehlo.slice %v54 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v56 = stablehlo.reshape %v55 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v57 = stablehlo.reshape %v48 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v58 = stablehlo.slice %v57 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v59 = stablehlo.reshape %v58 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v60 = stablehlo.reshape %v53 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v61 = stablehlo.slice %v60 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
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
    %v85 = stablehlo.pad %v83, %v84, low = [0, 0, 0], high = [0, 0, 704], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v86 = stablehlo.reshape %v85 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v87 = stablehlo.reshape %v43 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v88 = stablehlo.slice %v87 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v89 = stablehlo.reshape %v88 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v90 = stablehlo.reshape %v48 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v91 = stablehlo.slice %v90 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v92 = stablehlo.reshape %v91 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v93 = stablehlo.reshape %v53 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v94 = stablehlo.slice %v93 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
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
    %v118 = stablehlo.pad %v116, %v117, low = [0, 0, 64], high = [0, 0, 640], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v119 = stablehlo.reshape %v118 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v120 = stablehlo.add %v86, %v119 : tensor<32x151296xf32>
    %v121 = stablehlo.reshape %v43 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v122 = stablehlo.slice %v121 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v123 = stablehlo.reshape %v122 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v124 = stablehlo.reshape %v48 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v125 = stablehlo.slice %v124 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v126 = stablehlo.reshape %v125 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v127 = stablehlo.reshape %v53 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v128 = stablehlo.slice %v127 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
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
    %v152 = stablehlo.pad %v150, %v151, low = [0, 0, 128], high = [0, 0, 576], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v153 = stablehlo.reshape %v152 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v154 = stablehlo.add %v120, %v153 : tensor<32x151296xf32>
    %v155 = stablehlo.reshape %v43 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v156 = stablehlo.slice %v155 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v157 = stablehlo.reshape %v156 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v158 = stablehlo.reshape %v48 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v159 = stablehlo.slice %v158 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v160 = stablehlo.reshape %v159 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v161 = stablehlo.reshape %v53 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v162 = stablehlo.slice %v161 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v163 = stablehlo.reshape %v162 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v164 = stablehlo.reshape %v160 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v165 = stablehlo.transpose %v164, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v166 = stablehlo.reshape %v165 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v167 = stablehlo.reshape %v157 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v168 = stablehlo.reshape %v166 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v169 = stablehlo.dot_general %v167, %v168, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v170 = stablehlo.reshape %v169 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v171 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v172 = stablehlo.multiply %v170, %v171 : tensor<32x38809xf32>
    %v173 = stablehlo.reshape %v172 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v174 = stablehlo.constant dense<0.0> : tensor<f32>
    %v175 = stablehlo.exponential %v173 : tensor<32x197x197xf32>
    %v176 = stablehlo.reduce(%v175 init: %v174) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v177 = stablehlo.broadcast_in_dim %v176, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v178 = stablehlo.divide %v175, %v177 : tensor<32x197x197xf32>
    %v179 = stablehlo.reshape %v178 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v180 = stablehlo.reshape %v179 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v181 = stablehlo.reshape %v163 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v182 = stablehlo.dot_general %v180, %v181, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v183 = stablehlo.reshape %v182 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v184 = stablehlo.reshape %v183 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v185 = stablehlo.constant dense<0.0> : tensor<f32>
    %v186 = stablehlo.pad %v184, %v185, low = [0, 0, 192], high = [0, 0, 512], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v187 = stablehlo.reshape %v186 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v188 = stablehlo.add %v154, %v187 : tensor<32x151296xf32>
    %v189 = stablehlo.reshape %v43 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v190 = stablehlo.slice %v189 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v191 = stablehlo.reshape %v190 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v192 = stablehlo.reshape %v48 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v193 = stablehlo.slice %v192 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v194 = stablehlo.reshape %v193 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v195 = stablehlo.reshape %v53 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v196 = stablehlo.slice %v195 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v197 = stablehlo.reshape %v196 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v198 = stablehlo.reshape %v194 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v199 = stablehlo.transpose %v198, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v200 = stablehlo.reshape %v199 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v201 = stablehlo.reshape %v191 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v202 = stablehlo.reshape %v200 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v203 = stablehlo.dot_general %v201, %v202, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v204 = stablehlo.reshape %v203 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v205 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v206 = stablehlo.multiply %v204, %v205 : tensor<32x38809xf32>
    %v207 = stablehlo.reshape %v206 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v208 = stablehlo.constant dense<0.0> : tensor<f32>
    %v209 = stablehlo.exponential %v207 : tensor<32x197x197xf32>
    %v210 = stablehlo.reduce(%v209 init: %v208) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v211 = stablehlo.broadcast_in_dim %v210, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v212 = stablehlo.divide %v209, %v211 : tensor<32x197x197xf32>
    %v213 = stablehlo.reshape %v212 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v214 = stablehlo.reshape %v213 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v215 = stablehlo.reshape %v197 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v216 = stablehlo.dot_general %v214, %v215, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v217 = stablehlo.reshape %v216 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v218 = stablehlo.reshape %v217 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v219 = stablehlo.constant dense<0.0> : tensor<f32>
    %v220 = stablehlo.pad %v218, %v219, low = [0, 0, 256], high = [0, 0, 448], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v221 = stablehlo.reshape %v220 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v222 = stablehlo.add %v188, %v221 : tensor<32x151296xf32>
    %v223 = stablehlo.reshape %v43 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v224 = stablehlo.slice %v223 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v225 = stablehlo.reshape %v224 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v226 = stablehlo.reshape %v48 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v227 = stablehlo.slice %v226 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v228 = stablehlo.reshape %v227 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v229 = stablehlo.reshape %v53 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v230 = stablehlo.slice %v229 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v231 = stablehlo.reshape %v230 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v232 = stablehlo.reshape %v228 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v233 = stablehlo.transpose %v232, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v234 = stablehlo.reshape %v233 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v235 = stablehlo.reshape %v225 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v236 = stablehlo.reshape %v234 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v237 = stablehlo.dot_general %v235, %v236, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v238 = stablehlo.reshape %v237 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v239 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v240 = stablehlo.multiply %v238, %v239 : tensor<32x38809xf32>
    %v241 = stablehlo.reshape %v240 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v242 = stablehlo.constant dense<0.0> : tensor<f32>
    %v243 = stablehlo.exponential %v241 : tensor<32x197x197xf32>
    %v244 = stablehlo.reduce(%v243 init: %v242) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v245 = stablehlo.broadcast_in_dim %v244, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v246 = stablehlo.divide %v243, %v245 : tensor<32x197x197xf32>
    %v247 = stablehlo.reshape %v246 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v248 = stablehlo.reshape %v247 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v249 = stablehlo.reshape %v231 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v250 = stablehlo.dot_general %v248, %v249, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v251 = stablehlo.reshape %v250 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v252 = stablehlo.reshape %v251 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v253 = stablehlo.constant dense<0.0> : tensor<f32>
    %v254 = stablehlo.pad %v252, %v253, low = [0, 0, 320], high = [0, 0, 384], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v255 = stablehlo.reshape %v254 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v256 = stablehlo.add %v222, %v255 : tensor<32x151296xf32>
    %v257 = stablehlo.reshape %v43 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v258 = stablehlo.slice %v257 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v259 = stablehlo.reshape %v258 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v260 = stablehlo.reshape %v48 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v261 = stablehlo.slice %v260 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v262 = stablehlo.reshape %v261 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v263 = stablehlo.reshape %v53 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v264 = stablehlo.slice %v263 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v265 = stablehlo.reshape %v264 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v266 = stablehlo.reshape %v262 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v267 = stablehlo.transpose %v266, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v268 = stablehlo.reshape %v267 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v269 = stablehlo.reshape %v259 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v270 = stablehlo.reshape %v268 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v271 = stablehlo.dot_general %v269, %v270, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v272 = stablehlo.reshape %v271 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v273 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v274 = stablehlo.multiply %v272, %v273 : tensor<32x38809xf32>
    %v275 = stablehlo.reshape %v274 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v276 = stablehlo.constant dense<0.0> : tensor<f32>
    %v277 = stablehlo.exponential %v275 : tensor<32x197x197xf32>
    %v278 = stablehlo.reduce(%v277 init: %v276) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v279 = stablehlo.broadcast_in_dim %v278, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v280 = stablehlo.divide %v277, %v279 : tensor<32x197x197xf32>
    %v281 = stablehlo.reshape %v280 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v282 = stablehlo.reshape %v281 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v283 = stablehlo.reshape %v265 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v284 = stablehlo.dot_general %v282, %v283, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v285 = stablehlo.reshape %v284 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v286 = stablehlo.reshape %v285 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v287 = stablehlo.constant dense<0.0> : tensor<f32>
    %v288 = stablehlo.pad %v286, %v287, low = [0, 0, 384], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v289 = stablehlo.reshape %v288 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v290 = stablehlo.add %v256, %v289 : tensor<32x151296xf32>
    %v291 = stablehlo.reshape %v43 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v292 = stablehlo.slice %v291 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v293 = stablehlo.reshape %v292 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v294 = stablehlo.reshape %v48 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v295 = stablehlo.slice %v294 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v296 = stablehlo.reshape %v295 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v297 = stablehlo.reshape %v53 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v298 = stablehlo.slice %v297 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v299 = stablehlo.reshape %v298 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v300 = stablehlo.reshape %v296 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v301 = stablehlo.transpose %v300, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v302 = stablehlo.reshape %v301 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v303 = stablehlo.reshape %v293 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v304 = stablehlo.reshape %v302 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v305 = stablehlo.dot_general %v303, %v304, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v306 = stablehlo.reshape %v305 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v307 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v308 = stablehlo.multiply %v306, %v307 : tensor<32x38809xf32>
    %v309 = stablehlo.reshape %v308 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v310 = stablehlo.constant dense<0.0> : tensor<f32>
    %v311 = stablehlo.exponential %v309 : tensor<32x197x197xf32>
    %v312 = stablehlo.reduce(%v311 init: %v310) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v313 = stablehlo.broadcast_in_dim %v312, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v314 = stablehlo.divide %v311, %v313 : tensor<32x197x197xf32>
    %v315 = stablehlo.reshape %v314 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v316 = stablehlo.reshape %v315 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v317 = stablehlo.reshape %v299 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v318 = stablehlo.dot_general %v316, %v317, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v319 = stablehlo.reshape %v318 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v320 = stablehlo.reshape %v319 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v321 = stablehlo.constant dense<0.0> : tensor<f32>
    %v322 = stablehlo.pad %v320, %v321, low = [0, 0, 448], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v323 = stablehlo.reshape %v322 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v324 = stablehlo.add %v290, %v323 : tensor<32x151296xf32>
    %v325 = stablehlo.reshape %v43 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v326 = stablehlo.slice %v325 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v327 = stablehlo.reshape %v326 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v328 = stablehlo.reshape %v48 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v329 = stablehlo.slice %v328 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v330 = stablehlo.reshape %v329 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v331 = stablehlo.reshape %v53 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v332 = stablehlo.slice %v331 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v333 = stablehlo.reshape %v332 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v334 = stablehlo.reshape %v330 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v335 = stablehlo.transpose %v334, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v336 = stablehlo.reshape %v335 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v337 = stablehlo.reshape %v327 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v338 = stablehlo.reshape %v336 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v339 = stablehlo.dot_general %v337, %v338, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v340 = stablehlo.reshape %v339 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v341 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v342 = stablehlo.multiply %v340, %v341 : tensor<32x38809xf32>
    %v343 = stablehlo.reshape %v342 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v344 = stablehlo.constant dense<0.0> : tensor<f32>
    %v345 = stablehlo.exponential %v343 : tensor<32x197x197xf32>
    %v346 = stablehlo.reduce(%v345 init: %v344) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v347 = stablehlo.broadcast_in_dim %v346, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v348 = stablehlo.divide %v345, %v347 : tensor<32x197x197xf32>
    %v349 = stablehlo.reshape %v348 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v350 = stablehlo.reshape %v349 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v351 = stablehlo.reshape %v333 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v352 = stablehlo.dot_general %v350, %v351, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v353 = stablehlo.reshape %v352 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v354 = stablehlo.reshape %v353 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v355 = stablehlo.constant dense<0.0> : tensor<f32>
    %v356 = stablehlo.pad %v354, %v355, low = [0, 0, 512], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v357 = stablehlo.reshape %v356 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v358 = stablehlo.add %v324, %v357 : tensor<32x151296xf32>
    %v359 = stablehlo.reshape %v43 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v360 = stablehlo.slice %v359 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v361 = stablehlo.reshape %v360 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v362 = stablehlo.reshape %v48 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v363 = stablehlo.slice %v362 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v364 = stablehlo.reshape %v363 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v365 = stablehlo.reshape %v53 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v366 = stablehlo.slice %v365 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v367 = stablehlo.reshape %v366 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v368 = stablehlo.reshape %v364 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v369 = stablehlo.transpose %v368, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v370 = stablehlo.reshape %v369 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v371 = stablehlo.reshape %v361 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v372 = stablehlo.reshape %v370 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v373 = stablehlo.dot_general %v371, %v372, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v374 = stablehlo.reshape %v373 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v375 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v376 = stablehlo.multiply %v374, %v375 : tensor<32x38809xf32>
    %v377 = stablehlo.reshape %v376 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v378 = stablehlo.constant dense<0.0> : tensor<f32>
    %v379 = stablehlo.exponential %v377 : tensor<32x197x197xf32>
    %v380 = stablehlo.reduce(%v379 init: %v378) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v381 = stablehlo.broadcast_in_dim %v380, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v382 = stablehlo.divide %v379, %v381 : tensor<32x197x197xf32>
    %v383 = stablehlo.reshape %v382 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v384 = stablehlo.reshape %v383 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v385 = stablehlo.reshape %v367 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v386 = stablehlo.dot_general %v384, %v385, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v387 = stablehlo.reshape %v386 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v388 = stablehlo.reshape %v387 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v389 = stablehlo.constant dense<0.0> : tensor<f32>
    %v390 = stablehlo.pad %v388, %v389, low = [0, 0, 576], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v391 = stablehlo.reshape %v390 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v392 = stablehlo.add %v358, %v391 : tensor<32x151296xf32>
    %v393 = stablehlo.reshape %v43 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v394 = stablehlo.slice %v393 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v395 = stablehlo.reshape %v394 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v396 = stablehlo.reshape %v48 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v397 = stablehlo.slice %v396 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v398 = stablehlo.reshape %v397 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v399 = stablehlo.reshape %v53 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v400 = stablehlo.slice %v399 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v401 = stablehlo.reshape %v400 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v402 = stablehlo.reshape %v398 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v403 = stablehlo.transpose %v402, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v404 = stablehlo.reshape %v403 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v405 = stablehlo.reshape %v395 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v406 = stablehlo.reshape %v404 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v407 = stablehlo.dot_general %v405, %v406, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v408 = stablehlo.reshape %v407 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v409 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v410 = stablehlo.multiply %v408, %v409 : tensor<32x38809xf32>
    %v411 = stablehlo.reshape %v410 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v412 = stablehlo.constant dense<0.0> : tensor<f32>
    %v413 = stablehlo.exponential %v411 : tensor<32x197x197xf32>
    %v414 = stablehlo.reduce(%v413 init: %v412) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v415 = stablehlo.broadcast_in_dim %v414, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v416 = stablehlo.divide %v413, %v415 : tensor<32x197x197xf32>
    %v417 = stablehlo.reshape %v416 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v418 = stablehlo.reshape %v417 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v419 = stablehlo.reshape %v401 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v420 = stablehlo.dot_general %v418, %v419, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v421 = stablehlo.reshape %v420 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v422 = stablehlo.reshape %v421 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v423 = stablehlo.constant dense<0.0> : tensor<f32>
    %v424 = stablehlo.pad %v422, %v423, low = [0, 0, 640], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v425 = stablehlo.reshape %v424 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v426 = stablehlo.add %v392, %v425 : tensor<32x151296xf32>
    %v427 = stablehlo.reshape %v43 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v428 = stablehlo.slice %v427 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v429 = stablehlo.reshape %v428 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v430 = stablehlo.reshape %v48 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v431 = stablehlo.slice %v430 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v432 = stablehlo.reshape %v431 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v433 = stablehlo.reshape %v53 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v434 = stablehlo.slice %v433 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v435 = stablehlo.reshape %v434 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v436 = stablehlo.reshape %v432 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v437 = stablehlo.transpose %v436, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v438 = stablehlo.reshape %v437 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v439 = stablehlo.reshape %v429 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v440 = stablehlo.reshape %v438 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v441 = stablehlo.dot_general %v439, %v440, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v442 = stablehlo.reshape %v441 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v443 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v444 = stablehlo.multiply %v442, %v443 : tensor<32x38809xf32>
    %v445 = stablehlo.reshape %v444 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v446 = stablehlo.constant dense<0.0> : tensor<f32>
    %v447 = stablehlo.exponential %v445 : tensor<32x197x197xf32>
    %v448 = stablehlo.reduce(%v447 init: %v446) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v449 = stablehlo.broadcast_in_dim %v448, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v450 = stablehlo.divide %v447, %v449 : tensor<32x197x197xf32>
    %v451 = stablehlo.reshape %v450 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v452 = stablehlo.reshape %v451 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v453 = stablehlo.reshape %v435 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v454 = stablehlo.dot_general %v452, %v453, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v455 = stablehlo.reshape %v454 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v456 = stablehlo.reshape %v455 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v457 = stablehlo.constant dense<0.0> : tensor<f32>
    %v458 = stablehlo.pad %v456, %v457, low = [0, 0, 704], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v459 = stablehlo.reshape %v458 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v460 = stablehlo.add %v426, %v459 : tensor<32x151296xf32>
    %v461 = stablehlo.reshape %v460 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v462 = stablehlo.dot_general %v461, %b0_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v463 = stablehlo.broadcast_in_dim %b0_bo, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v464 = stablehlo.add %v462, %v463 : tensor<32x197x768xf32>
    %v465 = stablehlo.reshape %v464 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v466 = stablehlo.add %v10, %v465 : tensor<32x151296xf32>
    %v467 = stablehlo.reshape %v466 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v468 = stablehlo.constant dense<0.0> : tensor<f32>
    %v469 = stablehlo.constant dense<768.0> : tensor<32x197x768xf32>
    %v470 = stablehlo.constant dense<1.0e-5> : tensor<32x197x768xf32>
    %v471 = stablehlo.reduce(%v467 init: %v468) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v472 = stablehlo.broadcast_in_dim %v471, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v473 = stablehlo.divide %v472, %v469 : tensor<32x197x768xf32>
    %v474 = stablehlo.subtract %v467, %v473 : tensor<32x197x768xf32>
    %v475 = stablehlo.multiply %v474, %v474 : tensor<32x197x768xf32>
    %v476 = stablehlo.reduce(%v475 init: %v468) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v477 = stablehlo.broadcast_in_dim %v476, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v478 = stablehlo.divide %v477, %v469 : tensor<32x197x768xf32>
    %v479 = stablehlo.add %v478, %v470 : tensor<32x197x768xf32>
    %v480 = stablehlo.rsqrt %v479 : tensor<32x197x768xf32>
    %v481 = stablehlo.multiply %v474, %v480 : tensor<32x197x768xf32>
    %v482 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v483 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v484 = stablehlo.multiply %v481, %v482 : tensor<32x197x768xf32>
    %v485 = stablehlo.add %v484, %v483 : tensor<32x197x768xf32>
    %v486 = stablehlo.reshape %v485 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v487 = stablehlo.reshape %v486 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v488 = stablehlo.broadcast_in_dim %b0_g2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v489 = stablehlo.multiply %v487, %v488 : tensor<32x197x768xf32>
    %v490 = stablehlo.reshape %v489 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v491 = stablehlo.reshape %v490 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v492 = stablehlo.broadcast_in_dim %b0_bt2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v493 = stablehlo.add %v491, %v492 : tensor<32x197x768xf32>
    %v494 = stablehlo.reshape %v493 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v495 = stablehlo.reshape %v494 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v496 = stablehlo.dot_general %v495, %b0_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x3072xf32>) -> tensor<32x197x3072xf32>
    %v497 = stablehlo.broadcast_in_dim %b0_bfc1, dims = [2] : (tensor<3072xf32>) -> tensor<32x197x3072xf32>
    %v498 = stablehlo.add %v496, %v497 : tensor<32x197x3072xf32>
    %v499 = stablehlo.reshape %v498 : (tensor<32x197x3072xf32>) -> tensor<32x605184xf32>
    %v500 = stablehlo.constant dense<0.5> : tensor<32x605184xf32>
    %v501 = stablehlo.multiply %v500, %v499 : tensor<32x605184xf32>
    %v502 = stablehlo.negate %v499 : tensor<32x605184xf32>
    %v503 = stablehlo.constant dense<0.7071067811865476> : tensor<32x605184xf32>
    %v504 = stablehlo.multiply %v502, %v503 : tensor<32x605184xf32>
    %v505 = chlo.erfc %v504 : tensor<32x605184xf32> -> tensor<32x605184xf32>
    %v506 = stablehlo.multiply %v501, %v505 : tensor<32x605184xf32>
    %v507 = stablehlo.reshape %v506 : (tensor<32x605184xf32>) -> tensor<32x197x3072xf32>
    %v508 = stablehlo.dot_general %v507, %b0_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x3072xf32>, tensor<3072x768xf32>) -> tensor<32x197x768xf32>
    %v509 = stablehlo.broadcast_in_dim %b0_bfc2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v510 = stablehlo.add %v508, %v509 : tensor<32x197x768xf32>
    %v511 = stablehlo.reshape %v510 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v512 = stablehlo.add %v466, %v511 : tensor<32x151296xf32>
    %v513 = stablehlo.reshape %v512 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v514 = stablehlo.constant dense<0.0> : tensor<f32>
    %v515 = stablehlo.constant dense<768.0> : tensor<32x197x768xf32>
    %v516 = stablehlo.constant dense<1.0e-5> : tensor<32x197x768xf32>
    %v517 = stablehlo.reduce(%v513 init: %v514) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v518 = stablehlo.broadcast_in_dim %v517, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v519 = stablehlo.divide %v518, %v515 : tensor<32x197x768xf32>
    %v520 = stablehlo.subtract %v513, %v519 : tensor<32x197x768xf32>
    %v521 = stablehlo.multiply %v520, %v520 : tensor<32x197x768xf32>
    %v522 = stablehlo.reduce(%v521 init: %v514) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v523 = stablehlo.broadcast_in_dim %v522, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v524 = stablehlo.divide %v523, %v515 : tensor<32x197x768xf32>
    %v525 = stablehlo.add %v524, %v516 : tensor<32x197x768xf32>
    %v526 = stablehlo.rsqrt %v525 : tensor<32x197x768xf32>
    %v527 = stablehlo.multiply %v520, %v526 : tensor<32x197x768xf32>
    %v528 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v529 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v530 = stablehlo.multiply %v527, %v528 : tensor<32x197x768xf32>
    %v531 = stablehlo.add %v530, %v529 : tensor<32x197x768xf32>
    %v532 = stablehlo.reshape %v531 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v533 = stablehlo.reshape %v532 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v534 = stablehlo.broadcast_in_dim %b1_g1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v535 = stablehlo.multiply %v533, %v534 : tensor<32x197x768xf32>
    %v536 = stablehlo.reshape %v535 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v537 = stablehlo.reshape %v536 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v538 = stablehlo.broadcast_in_dim %b1_bt1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v539 = stablehlo.add %v537, %v538 : tensor<32x197x768xf32>
    %v540 = stablehlo.reshape %v539 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v541 = stablehlo.reshape %v540 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v542 = stablehlo.dot_general %v541, %b1_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v543 = stablehlo.broadcast_in_dim %b1_bq, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v544 = stablehlo.add %v542, %v543 : tensor<32x197x768xf32>
    %v545 = stablehlo.reshape %v544 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v546 = stablehlo.reshape %v540 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v547 = stablehlo.dot_general %v546, %b1_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v548 = stablehlo.broadcast_in_dim %b1_bk, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v549 = stablehlo.add %v547, %v548 : tensor<32x197x768xf32>
    %v550 = stablehlo.reshape %v549 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v551 = stablehlo.reshape %v540 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v552 = stablehlo.dot_general %v551, %b1_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v553 = stablehlo.broadcast_in_dim %b1_bv, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v554 = stablehlo.add %v552, %v553 : tensor<32x197x768xf32>
    %v555 = stablehlo.reshape %v554 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v556 = stablehlo.reshape %v545 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v557 = stablehlo.slice %v556 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v558 = stablehlo.reshape %v557 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v559 = stablehlo.reshape %v550 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v560 = stablehlo.slice %v559 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v561 = stablehlo.reshape %v560 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v562 = stablehlo.reshape %v555 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v563 = stablehlo.slice %v562 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v564 = stablehlo.reshape %v563 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v565 = stablehlo.reshape %v561 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v566 = stablehlo.transpose %v565, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v567 = stablehlo.reshape %v566 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v568 = stablehlo.reshape %v558 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v569 = stablehlo.reshape %v567 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v570 = stablehlo.dot_general %v568, %v569, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v571 = stablehlo.reshape %v570 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v572 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v573 = stablehlo.multiply %v571, %v572 : tensor<32x38809xf32>
    %v574 = stablehlo.reshape %v573 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v575 = stablehlo.constant dense<0.0> : tensor<f32>
    %v576 = stablehlo.exponential %v574 : tensor<32x197x197xf32>
    %v577 = stablehlo.reduce(%v576 init: %v575) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v578 = stablehlo.broadcast_in_dim %v577, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v579 = stablehlo.divide %v576, %v578 : tensor<32x197x197xf32>
    %v580 = stablehlo.reshape %v579 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v581 = stablehlo.reshape %v580 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v582 = stablehlo.reshape %v564 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v583 = stablehlo.dot_general %v581, %v582, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v584 = stablehlo.reshape %v583 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v585 = stablehlo.reshape %v584 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v586 = stablehlo.constant dense<0.0> : tensor<f32>
    %v587 = stablehlo.pad %v585, %v586, low = [0, 0, 0], high = [0, 0, 704], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v588 = stablehlo.reshape %v587 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v589 = stablehlo.reshape %v545 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v590 = stablehlo.slice %v589 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v591 = stablehlo.reshape %v590 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v592 = stablehlo.reshape %v550 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v593 = stablehlo.slice %v592 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v594 = stablehlo.reshape %v593 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v595 = stablehlo.reshape %v555 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v596 = stablehlo.slice %v595 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v597 = stablehlo.reshape %v596 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v598 = stablehlo.reshape %v594 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v599 = stablehlo.transpose %v598, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v600 = stablehlo.reshape %v599 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v601 = stablehlo.reshape %v591 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v602 = stablehlo.reshape %v600 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v603 = stablehlo.dot_general %v601, %v602, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v604 = stablehlo.reshape %v603 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v605 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v606 = stablehlo.multiply %v604, %v605 : tensor<32x38809xf32>
    %v607 = stablehlo.reshape %v606 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v608 = stablehlo.constant dense<0.0> : tensor<f32>
    %v609 = stablehlo.exponential %v607 : tensor<32x197x197xf32>
    %v610 = stablehlo.reduce(%v609 init: %v608) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v611 = stablehlo.broadcast_in_dim %v610, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v612 = stablehlo.divide %v609, %v611 : tensor<32x197x197xf32>
    %v613 = stablehlo.reshape %v612 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v614 = stablehlo.reshape %v613 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v615 = stablehlo.reshape %v597 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v616 = stablehlo.dot_general %v614, %v615, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v617 = stablehlo.reshape %v616 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v618 = stablehlo.reshape %v617 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v619 = stablehlo.constant dense<0.0> : tensor<f32>
    %v620 = stablehlo.pad %v618, %v619, low = [0, 0, 64], high = [0, 0, 640], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v621 = stablehlo.reshape %v620 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v622 = stablehlo.add %v588, %v621 : tensor<32x151296xf32>
    %v623 = stablehlo.reshape %v545 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v624 = stablehlo.slice %v623 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v625 = stablehlo.reshape %v624 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v626 = stablehlo.reshape %v550 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v627 = stablehlo.slice %v626 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v628 = stablehlo.reshape %v627 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v629 = stablehlo.reshape %v555 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v630 = stablehlo.slice %v629 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v631 = stablehlo.reshape %v630 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v632 = stablehlo.reshape %v628 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v633 = stablehlo.transpose %v632, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v634 = stablehlo.reshape %v633 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v635 = stablehlo.reshape %v625 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v636 = stablehlo.reshape %v634 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v637 = stablehlo.dot_general %v635, %v636, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v638 = stablehlo.reshape %v637 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v639 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v640 = stablehlo.multiply %v638, %v639 : tensor<32x38809xf32>
    %v641 = stablehlo.reshape %v640 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v642 = stablehlo.constant dense<0.0> : tensor<f32>
    %v643 = stablehlo.exponential %v641 : tensor<32x197x197xf32>
    %v644 = stablehlo.reduce(%v643 init: %v642) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v645 = stablehlo.broadcast_in_dim %v644, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v646 = stablehlo.divide %v643, %v645 : tensor<32x197x197xf32>
    %v647 = stablehlo.reshape %v646 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v648 = stablehlo.reshape %v647 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v649 = stablehlo.reshape %v631 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v650 = stablehlo.dot_general %v648, %v649, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v651 = stablehlo.reshape %v650 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v652 = stablehlo.reshape %v651 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v653 = stablehlo.constant dense<0.0> : tensor<f32>
    %v654 = stablehlo.pad %v652, %v653, low = [0, 0, 128], high = [0, 0, 576], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v655 = stablehlo.reshape %v654 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v656 = stablehlo.add %v622, %v655 : tensor<32x151296xf32>
    %v657 = stablehlo.reshape %v545 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v658 = stablehlo.slice %v657 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v659 = stablehlo.reshape %v658 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v660 = stablehlo.reshape %v550 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v661 = stablehlo.slice %v660 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v662 = stablehlo.reshape %v661 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v663 = stablehlo.reshape %v555 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v664 = stablehlo.slice %v663 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v665 = stablehlo.reshape %v664 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v666 = stablehlo.reshape %v662 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v667 = stablehlo.transpose %v666, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v668 = stablehlo.reshape %v667 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v669 = stablehlo.reshape %v659 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v670 = stablehlo.reshape %v668 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v671 = stablehlo.dot_general %v669, %v670, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v672 = stablehlo.reshape %v671 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v673 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v674 = stablehlo.multiply %v672, %v673 : tensor<32x38809xf32>
    %v675 = stablehlo.reshape %v674 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v676 = stablehlo.constant dense<0.0> : tensor<f32>
    %v677 = stablehlo.exponential %v675 : tensor<32x197x197xf32>
    %v678 = stablehlo.reduce(%v677 init: %v676) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v679 = stablehlo.broadcast_in_dim %v678, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v680 = stablehlo.divide %v677, %v679 : tensor<32x197x197xf32>
    %v681 = stablehlo.reshape %v680 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v682 = stablehlo.reshape %v681 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v683 = stablehlo.reshape %v665 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v684 = stablehlo.dot_general %v682, %v683, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v685 = stablehlo.reshape %v684 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v686 = stablehlo.reshape %v685 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v687 = stablehlo.constant dense<0.0> : tensor<f32>
    %v688 = stablehlo.pad %v686, %v687, low = [0, 0, 192], high = [0, 0, 512], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v689 = stablehlo.reshape %v688 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v690 = stablehlo.add %v656, %v689 : tensor<32x151296xf32>
    %v691 = stablehlo.reshape %v545 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v692 = stablehlo.slice %v691 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v693 = stablehlo.reshape %v692 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v694 = stablehlo.reshape %v550 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v695 = stablehlo.slice %v694 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v696 = stablehlo.reshape %v695 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v697 = stablehlo.reshape %v555 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v698 = stablehlo.slice %v697 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v699 = stablehlo.reshape %v698 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v700 = stablehlo.reshape %v696 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v701 = stablehlo.transpose %v700, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v702 = stablehlo.reshape %v701 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v703 = stablehlo.reshape %v693 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v704 = stablehlo.reshape %v702 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v705 = stablehlo.dot_general %v703, %v704, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v706 = stablehlo.reshape %v705 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v707 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v708 = stablehlo.multiply %v706, %v707 : tensor<32x38809xf32>
    %v709 = stablehlo.reshape %v708 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v710 = stablehlo.constant dense<0.0> : tensor<f32>
    %v711 = stablehlo.exponential %v709 : tensor<32x197x197xf32>
    %v712 = stablehlo.reduce(%v711 init: %v710) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v713 = stablehlo.broadcast_in_dim %v712, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v714 = stablehlo.divide %v711, %v713 : tensor<32x197x197xf32>
    %v715 = stablehlo.reshape %v714 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v716 = stablehlo.reshape %v715 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v717 = stablehlo.reshape %v699 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v718 = stablehlo.dot_general %v716, %v717, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v719 = stablehlo.reshape %v718 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v720 = stablehlo.reshape %v719 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v721 = stablehlo.constant dense<0.0> : tensor<f32>
    %v722 = stablehlo.pad %v720, %v721, low = [0, 0, 256], high = [0, 0, 448], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v723 = stablehlo.reshape %v722 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v724 = stablehlo.add %v690, %v723 : tensor<32x151296xf32>
    %v725 = stablehlo.reshape %v545 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v726 = stablehlo.slice %v725 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v727 = stablehlo.reshape %v726 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v728 = stablehlo.reshape %v550 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v729 = stablehlo.slice %v728 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v730 = stablehlo.reshape %v729 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v731 = stablehlo.reshape %v555 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v732 = stablehlo.slice %v731 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v733 = stablehlo.reshape %v732 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v734 = stablehlo.reshape %v730 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v735 = stablehlo.transpose %v734, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v736 = stablehlo.reshape %v735 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v737 = stablehlo.reshape %v727 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v738 = stablehlo.reshape %v736 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v739 = stablehlo.dot_general %v737, %v738, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v740 = stablehlo.reshape %v739 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v741 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v742 = stablehlo.multiply %v740, %v741 : tensor<32x38809xf32>
    %v743 = stablehlo.reshape %v742 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v744 = stablehlo.constant dense<0.0> : tensor<f32>
    %v745 = stablehlo.exponential %v743 : tensor<32x197x197xf32>
    %v746 = stablehlo.reduce(%v745 init: %v744) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v747 = stablehlo.broadcast_in_dim %v746, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v748 = stablehlo.divide %v745, %v747 : tensor<32x197x197xf32>
    %v749 = stablehlo.reshape %v748 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v750 = stablehlo.reshape %v749 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v751 = stablehlo.reshape %v733 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v752 = stablehlo.dot_general %v750, %v751, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v753 = stablehlo.reshape %v752 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v754 = stablehlo.reshape %v753 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v755 = stablehlo.constant dense<0.0> : tensor<f32>
    %v756 = stablehlo.pad %v754, %v755, low = [0, 0, 320], high = [0, 0, 384], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v757 = stablehlo.reshape %v756 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v758 = stablehlo.add %v724, %v757 : tensor<32x151296xf32>
    %v759 = stablehlo.reshape %v545 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v760 = stablehlo.slice %v759 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v761 = stablehlo.reshape %v760 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v762 = stablehlo.reshape %v550 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v763 = stablehlo.slice %v762 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v764 = stablehlo.reshape %v763 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v765 = stablehlo.reshape %v555 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v766 = stablehlo.slice %v765 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v767 = stablehlo.reshape %v766 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v768 = stablehlo.reshape %v764 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v769 = stablehlo.transpose %v768, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v770 = stablehlo.reshape %v769 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v771 = stablehlo.reshape %v761 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v772 = stablehlo.reshape %v770 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v773 = stablehlo.dot_general %v771, %v772, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v774 = stablehlo.reshape %v773 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v775 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v776 = stablehlo.multiply %v774, %v775 : tensor<32x38809xf32>
    %v777 = stablehlo.reshape %v776 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v778 = stablehlo.constant dense<0.0> : tensor<f32>
    %v779 = stablehlo.exponential %v777 : tensor<32x197x197xf32>
    %v780 = stablehlo.reduce(%v779 init: %v778) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v781 = stablehlo.broadcast_in_dim %v780, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v782 = stablehlo.divide %v779, %v781 : tensor<32x197x197xf32>
    %v783 = stablehlo.reshape %v782 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v784 = stablehlo.reshape %v783 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v785 = stablehlo.reshape %v767 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v786 = stablehlo.dot_general %v784, %v785, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v787 = stablehlo.reshape %v786 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v788 = stablehlo.reshape %v787 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v789 = stablehlo.constant dense<0.0> : tensor<f32>
    %v790 = stablehlo.pad %v788, %v789, low = [0, 0, 384], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v791 = stablehlo.reshape %v790 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v792 = stablehlo.add %v758, %v791 : tensor<32x151296xf32>
    %v793 = stablehlo.reshape %v545 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v794 = stablehlo.slice %v793 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v795 = stablehlo.reshape %v794 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v796 = stablehlo.reshape %v550 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v797 = stablehlo.slice %v796 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v798 = stablehlo.reshape %v797 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v799 = stablehlo.reshape %v555 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v800 = stablehlo.slice %v799 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v801 = stablehlo.reshape %v800 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v802 = stablehlo.reshape %v798 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v803 = stablehlo.transpose %v802, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v804 = stablehlo.reshape %v803 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v805 = stablehlo.reshape %v795 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v806 = stablehlo.reshape %v804 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v807 = stablehlo.dot_general %v805, %v806, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v808 = stablehlo.reshape %v807 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v809 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v810 = stablehlo.multiply %v808, %v809 : tensor<32x38809xf32>
    %v811 = stablehlo.reshape %v810 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v812 = stablehlo.constant dense<0.0> : tensor<f32>
    %v813 = stablehlo.exponential %v811 : tensor<32x197x197xf32>
    %v814 = stablehlo.reduce(%v813 init: %v812) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v815 = stablehlo.broadcast_in_dim %v814, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v816 = stablehlo.divide %v813, %v815 : tensor<32x197x197xf32>
    %v817 = stablehlo.reshape %v816 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v818 = stablehlo.reshape %v817 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v819 = stablehlo.reshape %v801 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v820 = stablehlo.dot_general %v818, %v819, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v821 = stablehlo.reshape %v820 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v822 = stablehlo.reshape %v821 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v823 = stablehlo.constant dense<0.0> : tensor<f32>
    %v824 = stablehlo.pad %v822, %v823, low = [0, 0, 448], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v825 = stablehlo.reshape %v824 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v826 = stablehlo.add %v792, %v825 : tensor<32x151296xf32>
    %v827 = stablehlo.reshape %v545 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v828 = stablehlo.slice %v827 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v829 = stablehlo.reshape %v828 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v830 = stablehlo.reshape %v550 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v831 = stablehlo.slice %v830 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v832 = stablehlo.reshape %v831 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v833 = stablehlo.reshape %v555 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v834 = stablehlo.slice %v833 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v835 = stablehlo.reshape %v834 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v836 = stablehlo.reshape %v832 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v837 = stablehlo.transpose %v836, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v838 = stablehlo.reshape %v837 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v839 = stablehlo.reshape %v829 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v840 = stablehlo.reshape %v838 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v841 = stablehlo.dot_general %v839, %v840, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v842 = stablehlo.reshape %v841 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v843 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v844 = stablehlo.multiply %v842, %v843 : tensor<32x38809xf32>
    %v845 = stablehlo.reshape %v844 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v846 = stablehlo.constant dense<0.0> : tensor<f32>
    %v847 = stablehlo.exponential %v845 : tensor<32x197x197xf32>
    %v848 = stablehlo.reduce(%v847 init: %v846) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v849 = stablehlo.broadcast_in_dim %v848, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v850 = stablehlo.divide %v847, %v849 : tensor<32x197x197xf32>
    %v851 = stablehlo.reshape %v850 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v852 = stablehlo.reshape %v851 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v853 = stablehlo.reshape %v835 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v854 = stablehlo.dot_general %v852, %v853, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v855 = stablehlo.reshape %v854 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v856 = stablehlo.reshape %v855 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v857 = stablehlo.constant dense<0.0> : tensor<f32>
    %v858 = stablehlo.pad %v856, %v857, low = [0, 0, 512], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v859 = stablehlo.reshape %v858 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v860 = stablehlo.add %v826, %v859 : tensor<32x151296xf32>
    %v861 = stablehlo.reshape %v545 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v862 = stablehlo.slice %v861 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v863 = stablehlo.reshape %v862 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v864 = stablehlo.reshape %v550 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v865 = stablehlo.slice %v864 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v866 = stablehlo.reshape %v865 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v867 = stablehlo.reshape %v555 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v868 = stablehlo.slice %v867 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v869 = stablehlo.reshape %v868 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v870 = stablehlo.reshape %v866 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v871 = stablehlo.transpose %v870, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v872 = stablehlo.reshape %v871 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v873 = stablehlo.reshape %v863 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v874 = stablehlo.reshape %v872 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v875 = stablehlo.dot_general %v873, %v874, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v876 = stablehlo.reshape %v875 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v877 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v878 = stablehlo.multiply %v876, %v877 : tensor<32x38809xf32>
    %v879 = stablehlo.reshape %v878 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v880 = stablehlo.constant dense<0.0> : tensor<f32>
    %v881 = stablehlo.exponential %v879 : tensor<32x197x197xf32>
    %v882 = stablehlo.reduce(%v881 init: %v880) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v883 = stablehlo.broadcast_in_dim %v882, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v884 = stablehlo.divide %v881, %v883 : tensor<32x197x197xf32>
    %v885 = stablehlo.reshape %v884 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v886 = stablehlo.reshape %v885 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v887 = stablehlo.reshape %v869 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v888 = stablehlo.dot_general %v886, %v887, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v889 = stablehlo.reshape %v888 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v890 = stablehlo.reshape %v889 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v891 = stablehlo.constant dense<0.0> : tensor<f32>
    %v892 = stablehlo.pad %v890, %v891, low = [0, 0, 576], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v893 = stablehlo.reshape %v892 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v894 = stablehlo.add %v860, %v893 : tensor<32x151296xf32>
    %v895 = stablehlo.reshape %v545 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v896 = stablehlo.slice %v895 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v897 = stablehlo.reshape %v896 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v898 = stablehlo.reshape %v550 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v899 = stablehlo.slice %v898 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v900 = stablehlo.reshape %v899 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v901 = stablehlo.reshape %v555 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v902 = stablehlo.slice %v901 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
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
    %v926 = stablehlo.pad %v924, %v925, low = [0, 0, 640], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v927 = stablehlo.reshape %v926 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v928 = stablehlo.add %v894, %v927 : tensor<32x151296xf32>
    %v929 = stablehlo.reshape %v545 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v930 = stablehlo.slice %v929 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v931 = stablehlo.reshape %v930 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v932 = stablehlo.reshape %v550 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v933 = stablehlo.slice %v932 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v934 = stablehlo.reshape %v933 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v935 = stablehlo.reshape %v555 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v936 = stablehlo.slice %v935 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
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
    %v960 = stablehlo.pad %v958, %v959, low = [0, 0, 704], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v961 = stablehlo.reshape %v960 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v962 = stablehlo.add %v928, %v961 : tensor<32x151296xf32>
    %v963 = stablehlo.reshape %v962 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v964 = stablehlo.dot_general %v963, %b1_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v965 = stablehlo.broadcast_in_dim %b1_bo, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v966 = stablehlo.add %v964, %v965 : tensor<32x197x768xf32>
    %v967 = stablehlo.reshape %v966 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v968 = stablehlo.add %v512, %v967 : tensor<32x151296xf32>
    %v969 = stablehlo.reshape %v968 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v970 = stablehlo.constant dense<0.0> : tensor<f32>
    %v971 = stablehlo.constant dense<768.0> : tensor<32x197x768xf32>
    %v972 = stablehlo.constant dense<1.0e-5> : tensor<32x197x768xf32>
    %v973 = stablehlo.reduce(%v969 init: %v970) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v974 = stablehlo.broadcast_in_dim %v973, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v975 = stablehlo.divide %v974, %v971 : tensor<32x197x768xf32>
    %v976 = stablehlo.subtract %v969, %v975 : tensor<32x197x768xf32>
    %v977 = stablehlo.multiply %v976, %v976 : tensor<32x197x768xf32>
    %v978 = stablehlo.reduce(%v977 init: %v970) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v979 = stablehlo.broadcast_in_dim %v978, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v980 = stablehlo.divide %v979, %v971 : tensor<32x197x768xf32>
    %v981 = stablehlo.add %v980, %v972 : tensor<32x197x768xf32>
    %v982 = stablehlo.rsqrt %v981 : tensor<32x197x768xf32>
    %v983 = stablehlo.multiply %v976, %v982 : tensor<32x197x768xf32>
    %v984 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v985 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v986 = stablehlo.multiply %v983, %v984 : tensor<32x197x768xf32>
    %v987 = stablehlo.add %v986, %v985 : tensor<32x197x768xf32>
    %v988 = stablehlo.reshape %v987 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v989 = stablehlo.reshape %v988 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v990 = stablehlo.broadcast_in_dim %b1_g2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v991 = stablehlo.multiply %v989, %v990 : tensor<32x197x768xf32>
    %v992 = stablehlo.reshape %v991 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v993 = stablehlo.reshape %v992 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v994 = stablehlo.broadcast_in_dim %b1_bt2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v995 = stablehlo.add %v993, %v994 : tensor<32x197x768xf32>
    %v996 = stablehlo.reshape %v995 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v997 = stablehlo.reshape %v996 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v998 = stablehlo.dot_general %v997, %b1_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x3072xf32>) -> tensor<32x197x3072xf32>
    %v999 = stablehlo.broadcast_in_dim %b1_bfc1, dims = [2] : (tensor<3072xf32>) -> tensor<32x197x3072xf32>
    %v1000 = stablehlo.add %v998, %v999 : tensor<32x197x3072xf32>
    %v1001 = stablehlo.reshape %v1000 : (tensor<32x197x3072xf32>) -> tensor<32x605184xf32>
    %v1002 = stablehlo.constant dense<0.5> : tensor<32x605184xf32>
    %v1003 = stablehlo.multiply %v1002, %v1001 : tensor<32x605184xf32>
    %v1004 = stablehlo.negate %v1001 : tensor<32x605184xf32>
    %v1005 = stablehlo.constant dense<0.7071067811865476> : tensor<32x605184xf32>
    %v1006 = stablehlo.multiply %v1004, %v1005 : tensor<32x605184xf32>
    %v1007 = chlo.erfc %v1006 : tensor<32x605184xf32> -> tensor<32x605184xf32>
    %v1008 = stablehlo.multiply %v1003, %v1007 : tensor<32x605184xf32>
    %v1009 = stablehlo.reshape %v1008 : (tensor<32x605184xf32>) -> tensor<32x197x3072xf32>
    %v1010 = stablehlo.dot_general %v1009, %b1_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x3072xf32>, tensor<3072x768xf32>) -> tensor<32x197x768xf32>
    %v1011 = stablehlo.broadcast_in_dim %b1_bfc2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v1012 = stablehlo.add %v1010, %v1011 : tensor<32x197x768xf32>
    %v1013 = stablehlo.reshape %v1012 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1014 = stablehlo.add %v968, %v1013 : tensor<32x151296xf32>
    %v1015 = stablehlo.reshape %v1014 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1016 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1017 = stablehlo.constant dense<768.0> : tensor<32x197x768xf32>
    %v1018 = stablehlo.constant dense<1.0e-5> : tensor<32x197x768xf32>
    %v1019 = stablehlo.reduce(%v1015 init: %v1016) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1020 = stablehlo.broadcast_in_dim %v1019, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v1021 = stablehlo.divide %v1020, %v1017 : tensor<32x197x768xf32>
    %v1022 = stablehlo.subtract %v1015, %v1021 : tensor<32x197x768xf32>
    %v1023 = stablehlo.multiply %v1022, %v1022 : tensor<32x197x768xf32>
    %v1024 = stablehlo.reduce(%v1023 init: %v1016) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1025 = stablehlo.broadcast_in_dim %v1024, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v1026 = stablehlo.divide %v1025, %v1017 : tensor<32x197x768xf32>
    %v1027 = stablehlo.add %v1026, %v1018 : tensor<32x197x768xf32>
    %v1028 = stablehlo.rsqrt %v1027 : tensor<32x197x768xf32>
    %v1029 = stablehlo.multiply %v1022, %v1028 : tensor<32x197x768xf32>
    %v1030 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v1031 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v1032 = stablehlo.multiply %v1029, %v1030 : tensor<32x197x768xf32>
    %v1033 = stablehlo.add %v1032, %v1031 : tensor<32x197x768xf32>
    %v1034 = stablehlo.reshape %v1033 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1035 = stablehlo.reshape %v1034 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1036 = stablehlo.broadcast_in_dim %b2_g1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v1037 = stablehlo.multiply %v1035, %v1036 : tensor<32x197x768xf32>
    %v1038 = stablehlo.reshape %v1037 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1039 = stablehlo.reshape %v1038 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1040 = stablehlo.broadcast_in_dim %b2_bt1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v1041 = stablehlo.add %v1039, %v1040 : tensor<32x197x768xf32>
    %v1042 = stablehlo.reshape %v1041 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1043 = stablehlo.reshape %v1042 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1044 = stablehlo.dot_general %v1043, %b2_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v1045 = stablehlo.broadcast_in_dim %b2_bq, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v1046 = stablehlo.add %v1044, %v1045 : tensor<32x197x768xf32>
    %v1047 = stablehlo.reshape %v1046 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1048 = stablehlo.reshape %v1042 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1049 = stablehlo.dot_general %v1048, %b2_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v1050 = stablehlo.broadcast_in_dim %b2_bk, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v1051 = stablehlo.add %v1049, %v1050 : tensor<32x197x768xf32>
    %v1052 = stablehlo.reshape %v1051 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1053 = stablehlo.reshape %v1042 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1054 = stablehlo.dot_general %v1053, %b2_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v1055 = stablehlo.broadcast_in_dim %b2_bv, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v1056 = stablehlo.add %v1054, %v1055 : tensor<32x197x768xf32>
    %v1057 = stablehlo.reshape %v1056 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1058 = stablehlo.reshape %v1047 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1059 = stablehlo.slice %v1058 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1060 = stablehlo.reshape %v1059 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1061 = stablehlo.reshape %v1052 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1062 = stablehlo.slice %v1061 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1063 = stablehlo.reshape %v1062 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1064 = stablehlo.reshape %v1057 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1065 = stablehlo.slice %v1064 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1066 = stablehlo.reshape %v1065 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1067 = stablehlo.reshape %v1063 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1068 = stablehlo.transpose %v1067, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1069 = stablehlo.reshape %v1068 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1070 = stablehlo.reshape %v1060 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1071 = stablehlo.reshape %v1069 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1072 = stablehlo.dot_general %v1070, %v1071, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1073 = stablehlo.reshape %v1072 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1074 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1075 = stablehlo.multiply %v1073, %v1074 : tensor<32x38809xf32>
    %v1076 = stablehlo.reshape %v1075 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1077 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1078 = stablehlo.exponential %v1076 : tensor<32x197x197xf32>
    %v1079 = stablehlo.reduce(%v1078 init: %v1077) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1080 = stablehlo.broadcast_in_dim %v1079, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1081 = stablehlo.divide %v1078, %v1080 : tensor<32x197x197xf32>
    %v1082 = stablehlo.reshape %v1081 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1083 = stablehlo.reshape %v1082 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1084 = stablehlo.reshape %v1066 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1085 = stablehlo.dot_general %v1083, %v1084, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1086 = stablehlo.reshape %v1085 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1087 = stablehlo.reshape %v1086 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1088 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1089 = stablehlo.pad %v1087, %v1088, low = [0, 0, 0], high = [0, 0, 704], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v1090 = stablehlo.reshape %v1089 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1091 = stablehlo.reshape %v1047 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1092 = stablehlo.slice %v1091 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1093 = stablehlo.reshape %v1092 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1094 = stablehlo.reshape %v1052 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1095 = stablehlo.slice %v1094 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1096 = stablehlo.reshape %v1095 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1097 = stablehlo.reshape %v1057 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1098 = stablehlo.slice %v1097 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1099 = stablehlo.reshape %v1098 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1100 = stablehlo.reshape %v1096 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1101 = stablehlo.transpose %v1100, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1102 = stablehlo.reshape %v1101 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1103 = stablehlo.reshape %v1093 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1104 = stablehlo.reshape %v1102 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1105 = stablehlo.dot_general %v1103, %v1104, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1106 = stablehlo.reshape %v1105 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1107 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1108 = stablehlo.multiply %v1106, %v1107 : tensor<32x38809xf32>
    %v1109 = stablehlo.reshape %v1108 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1110 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1111 = stablehlo.exponential %v1109 : tensor<32x197x197xf32>
    %v1112 = stablehlo.reduce(%v1111 init: %v1110) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1113 = stablehlo.broadcast_in_dim %v1112, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1114 = stablehlo.divide %v1111, %v1113 : tensor<32x197x197xf32>
    %v1115 = stablehlo.reshape %v1114 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1116 = stablehlo.reshape %v1115 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1117 = stablehlo.reshape %v1099 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1118 = stablehlo.dot_general %v1116, %v1117, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1119 = stablehlo.reshape %v1118 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1120 = stablehlo.reshape %v1119 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1121 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1122 = stablehlo.pad %v1120, %v1121, low = [0, 0, 64], high = [0, 0, 640], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v1123 = stablehlo.reshape %v1122 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1124 = stablehlo.add %v1090, %v1123 : tensor<32x151296xf32>
    %v1125 = stablehlo.reshape %v1047 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1126 = stablehlo.slice %v1125 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1127 = stablehlo.reshape %v1126 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1128 = stablehlo.reshape %v1052 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1129 = stablehlo.slice %v1128 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1130 = stablehlo.reshape %v1129 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1131 = stablehlo.reshape %v1057 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1132 = stablehlo.slice %v1131 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1133 = stablehlo.reshape %v1132 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1134 = stablehlo.reshape %v1130 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1135 = stablehlo.transpose %v1134, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1136 = stablehlo.reshape %v1135 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1137 = stablehlo.reshape %v1127 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1138 = stablehlo.reshape %v1136 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1139 = stablehlo.dot_general %v1137, %v1138, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1140 = stablehlo.reshape %v1139 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1141 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1142 = stablehlo.multiply %v1140, %v1141 : tensor<32x38809xf32>
    %v1143 = stablehlo.reshape %v1142 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1144 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1145 = stablehlo.exponential %v1143 : tensor<32x197x197xf32>
    %v1146 = stablehlo.reduce(%v1145 init: %v1144) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1147 = stablehlo.broadcast_in_dim %v1146, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1148 = stablehlo.divide %v1145, %v1147 : tensor<32x197x197xf32>
    %v1149 = stablehlo.reshape %v1148 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1150 = stablehlo.reshape %v1149 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1151 = stablehlo.reshape %v1133 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1152 = stablehlo.dot_general %v1150, %v1151, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1153 = stablehlo.reshape %v1152 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1154 = stablehlo.reshape %v1153 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1155 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1156 = stablehlo.pad %v1154, %v1155, low = [0, 0, 128], high = [0, 0, 576], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v1157 = stablehlo.reshape %v1156 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1158 = stablehlo.add %v1124, %v1157 : tensor<32x151296xf32>
    %v1159 = stablehlo.reshape %v1047 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1160 = stablehlo.slice %v1159 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1161 = stablehlo.reshape %v1160 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1162 = stablehlo.reshape %v1052 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1163 = stablehlo.slice %v1162 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1164 = stablehlo.reshape %v1163 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1165 = stablehlo.reshape %v1057 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1166 = stablehlo.slice %v1165 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1167 = stablehlo.reshape %v1166 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1168 = stablehlo.reshape %v1164 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1169 = stablehlo.transpose %v1168, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1170 = stablehlo.reshape %v1169 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1171 = stablehlo.reshape %v1161 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1172 = stablehlo.reshape %v1170 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1173 = stablehlo.dot_general %v1171, %v1172, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1174 = stablehlo.reshape %v1173 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1175 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1176 = stablehlo.multiply %v1174, %v1175 : tensor<32x38809xf32>
    %v1177 = stablehlo.reshape %v1176 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1178 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1179 = stablehlo.exponential %v1177 : tensor<32x197x197xf32>
    %v1180 = stablehlo.reduce(%v1179 init: %v1178) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1181 = stablehlo.broadcast_in_dim %v1180, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1182 = stablehlo.divide %v1179, %v1181 : tensor<32x197x197xf32>
    %v1183 = stablehlo.reshape %v1182 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1184 = stablehlo.reshape %v1183 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1185 = stablehlo.reshape %v1167 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1186 = stablehlo.dot_general %v1184, %v1185, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1187 = stablehlo.reshape %v1186 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1188 = stablehlo.reshape %v1187 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1189 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1190 = stablehlo.pad %v1188, %v1189, low = [0, 0, 192], high = [0, 0, 512], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v1191 = stablehlo.reshape %v1190 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1192 = stablehlo.add %v1158, %v1191 : tensor<32x151296xf32>
    %v1193 = stablehlo.reshape %v1047 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1194 = stablehlo.slice %v1193 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1195 = stablehlo.reshape %v1194 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1196 = stablehlo.reshape %v1052 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1197 = stablehlo.slice %v1196 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1198 = stablehlo.reshape %v1197 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1199 = stablehlo.reshape %v1057 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1200 = stablehlo.slice %v1199 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1201 = stablehlo.reshape %v1200 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1202 = stablehlo.reshape %v1198 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1203 = stablehlo.transpose %v1202, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1204 = stablehlo.reshape %v1203 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1205 = stablehlo.reshape %v1195 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1206 = stablehlo.reshape %v1204 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1207 = stablehlo.dot_general %v1205, %v1206, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1208 = stablehlo.reshape %v1207 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1209 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1210 = stablehlo.multiply %v1208, %v1209 : tensor<32x38809xf32>
    %v1211 = stablehlo.reshape %v1210 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1212 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1213 = stablehlo.exponential %v1211 : tensor<32x197x197xf32>
    %v1214 = stablehlo.reduce(%v1213 init: %v1212) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1215 = stablehlo.broadcast_in_dim %v1214, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1216 = stablehlo.divide %v1213, %v1215 : tensor<32x197x197xf32>
    %v1217 = stablehlo.reshape %v1216 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1218 = stablehlo.reshape %v1217 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1219 = stablehlo.reshape %v1201 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1220 = stablehlo.dot_general %v1218, %v1219, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1221 = stablehlo.reshape %v1220 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1222 = stablehlo.reshape %v1221 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1223 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1224 = stablehlo.pad %v1222, %v1223, low = [0, 0, 256], high = [0, 0, 448], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v1225 = stablehlo.reshape %v1224 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1226 = stablehlo.add %v1192, %v1225 : tensor<32x151296xf32>
    %v1227 = stablehlo.reshape %v1047 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1228 = stablehlo.slice %v1227 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1229 = stablehlo.reshape %v1228 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1230 = stablehlo.reshape %v1052 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1231 = stablehlo.slice %v1230 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1232 = stablehlo.reshape %v1231 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1233 = stablehlo.reshape %v1057 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1234 = stablehlo.slice %v1233 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1235 = stablehlo.reshape %v1234 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1236 = stablehlo.reshape %v1232 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1237 = stablehlo.transpose %v1236, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1238 = stablehlo.reshape %v1237 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1239 = stablehlo.reshape %v1229 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1240 = stablehlo.reshape %v1238 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1241 = stablehlo.dot_general %v1239, %v1240, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1242 = stablehlo.reshape %v1241 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1243 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1244 = stablehlo.multiply %v1242, %v1243 : tensor<32x38809xf32>
    %v1245 = stablehlo.reshape %v1244 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1246 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1247 = stablehlo.exponential %v1245 : tensor<32x197x197xf32>
    %v1248 = stablehlo.reduce(%v1247 init: %v1246) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1249 = stablehlo.broadcast_in_dim %v1248, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1250 = stablehlo.divide %v1247, %v1249 : tensor<32x197x197xf32>
    %v1251 = stablehlo.reshape %v1250 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1252 = stablehlo.reshape %v1251 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1253 = stablehlo.reshape %v1235 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1254 = stablehlo.dot_general %v1252, %v1253, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1255 = stablehlo.reshape %v1254 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1256 = stablehlo.reshape %v1255 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1257 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1258 = stablehlo.pad %v1256, %v1257, low = [0, 0, 320], high = [0, 0, 384], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v1259 = stablehlo.reshape %v1258 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1260 = stablehlo.add %v1226, %v1259 : tensor<32x151296xf32>
    %v1261 = stablehlo.reshape %v1047 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1262 = stablehlo.slice %v1261 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1263 = stablehlo.reshape %v1262 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1264 = stablehlo.reshape %v1052 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1265 = stablehlo.slice %v1264 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1266 = stablehlo.reshape %v1265 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1267 = stablehlo.reshape %v1057 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1268 = stablehlo.slice %v1267 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1269 = stablehlo.reshape %v1268 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1270 = stablehlo.reshape %v1266 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1271 = stablehlo.transpose %v1270, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1272 = stablehlo.reshape %v1271 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1273 = stablehlo.reshape %v1263 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1274 = stablehlo.reshape %v1272 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1275 = stablehlo.dot_general %v1273, %v1274, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1276 = stablehlo.reshape %v1275 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1277 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1278 = stablehlo.multiply %v1276, %v1277 : tensor<32x38809xf32>
    %v1279 = stablehlo.reshape %v1278 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1280 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1281 = stablehlo.exponential %v1279 : tensor<32x197x197xf32>
    %v1282 = stablehlo.reduce(%v1281 init: %v1280) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1283 = stablehlo.broadcast_in_dim %v1282, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1284 = stablehlo.divide %v1281, %v1283 : tensor<32x197x197xf32>
    %v1285 = stablehlo.reshape %v1284 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1286 = stablehlo.reshape %v1285 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1287 = stablehlo.reshape %v1269 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1288 = stablehlo.dot_general %v1286, %v1287, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1289 = stablehlo.reshape %v1288 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1290 = stablehlo.reshape %v1289 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1291 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1292 = stablehlo.pad %v1290, %v1291, low = [0, 0, 384], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v1293 = stablehlo.reshape %v1292 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1294 = stablehlo.add %v1260, %v1293 : tensor<32x151296xf32>
    %v1295 = stablehlo.reshape %v1047 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1296 = stablehlo.slice %v1295 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1297 = stablehlo.reshape %v1296 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1298 = stablehlo.reshape %v1052 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1299 = stablehlo.slice %v1298 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1300 = stablehlo.reshape %v1299 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1301 = stablehlo.reshape %v1057 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1302 = stablehlo.slice %v1301 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1303 = stablehlo.reshape %v1302 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1304 = stablehlo.reshape %v1300 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1305 = stablehlo.transpose %v1304, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1306 = stablehlo.reshape %v1305 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1307 = stablehlo.reshape %v1297 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1308 = stablehlo.reshape %v1306 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1309 = stablehlo.dot_general %v1307, %v1308, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1310 = stablehlo.reshape %v1309 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1311 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1312 = stablehlo.multiply %v1310, %v1311 : tensor<32x38809xf32>
    %v1313 = stablehlo.reshape %v1312 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1314 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1315 = stablehlo.exponential %v1313 : tensor<32x197x197xf32>
    %v1316 = stablehlo.reduce(%v1315 init: %v1314) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1317 = stablehlo.broadcast_in_dim %v1316, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1318 = stablehlo.divide %v1315, %v1317 : tensor<32x197x197xf32>
    %v1319 = stablehlo.reshape %v1318 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1320 = stablehlo.reshape %v1319 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1321 = stablehlo.reshape %v1303 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1322 = stablehlo.dot_general %v1320, %v1321, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1323 = stablehlo.reshape %v1322 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1324 = stablehlo.reshape %v1323 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1325 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1326 = stablehlo.pad %v1324, %v1325, low = [0, 0, 448], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v1327 = stablehlo.reshape %v1326 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1328 = stablehlo.add %v1294, %v1327 : tensor<32x151296xf32>
    %v1329 = stablehlo.reshape %v1047 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1330 = stablehlo.slice %v1329 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1331 = stablehlo.reshape %v1330 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1332 = stablehlo.reshape %v1052 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1333 = stablehlo.slice %v1332 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1334 = stablehlo.reshape %v1333 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1335 = stablehlo.reshape %v1057 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1336 = stablehlo.slice %v1335 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1337 = stablehlo.reshape %v1336 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1338 = stablehlo.reshape %v1334 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1339 = stablehlo.transpose %v1338, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1340 = stablehlo.reshape %v1339 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1341 = stablehlo.reshape %v1331 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1342 = stablehlo.reshape %v1340 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1343 = stablehlo.dot_general %v1341, %v1342, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1344 = stablehlo.reshape %v1343 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1345 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1346 = stablehlo.multiply %v1344, %v1345 : tensor<32x38809xf32>
    %v1347 = stablehlo.reshape %v1346 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1348 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1349 = stablehlo.exponential %v1347 : tensor<32x197x197xf32>
    %v1350 = stablehlo.reduce(%v1349 init: %v1348) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1351 = stablehlo.broadcast_in_dim %v1350, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1352 = stablehlo.divide %v1349, %v1351 : tensor<32x197x197xf32>
    %v1353 = stablehlo.reshape %v1352 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1354 = stablehlo.reshape %v1353 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1355 = stablehlo.reshape %v1337 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1356 = stablehlo.dot_general %v1354, %v1355, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1357 = stablehlo.reshape %v1356 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1358 = stablehlo.reshape %v1357 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1359 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1360 = stablehlo.pad %v1358, %v1359, low = [0, 0, 512], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v1361 = stablehlo.reshape %v1360 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1362 = stablehlo.add %v1328, %v1361 : tensor<32x151296xf32>
    %v1363 = stablehlo.reshape %v1047 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1364 = stablehlo.slice %v1363 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1365 = stablehlo.reshape %v1364 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1366 = stablehlo.reshape %v1052 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1367 = stablehlo.slice %v1366 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1368 = stablehlo.reshape %v1367 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1369 = stablehlo.reshape %v1057 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1370 = stablehlo.slice %v1369 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1371 = stablehlo.reshape %v1370 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1372 = stablehlo.reshape %v1368 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1373 = stablehlo.transpose %v1372, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1374 = stablehlo.reshape %v1373 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1375 = stablehlo.reshape %v1365 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1376 = stablehlo.reshape %v1374 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1377 = stablehlo.dot_general %v1375, %v1376, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1378 = stablehlo.reshape %v1377 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1379 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1380 = stablehlo.multiply %v1378, %v1379 : tensor<32x38809xf32>
    %v1381 = stablehlo.reshape %v1380 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1382 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1383 = stablehlo.exponential %v1381 : tensor<32x197x197xf32>
    %v1384 = stablehlo.reduce(%v1383 init: %v1382) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1385 = stablehlo.broadcast_in_dim %v1384, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1386 = stablehlo.divide %v1383, %v1385 : tensor<32x197x197xf32>
    %v1387 = stablehlo.reshape %v1386 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1388 = stablehlo.reshape %v1387 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1389 = stablehlo.reshape %v1371 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1390 = stablehlo.dot_general %v1388, %v1389, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1391 = stablehlo.reshape %v1390 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1392 = stablehlo.reshape %v1391 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1393 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1394 = stablehlo.pad %v1392, %v1393, low = [0, 0, 576], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v1395 = stablehlo.reshape %v1394 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1396 = stablehlo.add %v1362, %v1395 : tensor<32x151296xf32>
    %v1397 = stablehlo.reshape %v1047 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1398 = stablehlo.slice %v1397 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1399 = stablehlo.reshape %v1398 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1400 = stablehlo.reshape %v1052 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1401 = stablehlo.slice %v1400 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1402 = stablehlo.reshape %v1401 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1403 = stablehlo.reshape %v1057 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1404 = stablehlo.slice %v1403 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1405 = stablehlo.reshape %v1404 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1406 = stablehlo.reshape %v1402 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1407 = stablehlo.transpose %v1406, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1408 = stablehlo.reshape %v1407 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1409 = stablehlo.reshape %v1399 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1410 = stablehlo.reshape %v1408 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1411 = stablehlo.dot_general %v1409, %v1410, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1412 = stablehlo.reshape %v1411 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1413 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1414 = stablehlo.multiply %v1412, %v1413 : tensor<32x38809xf32>
    %v1415 = stablehlo.reshape %v1414 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1416 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1417 = stablehlo.exponential %v1415 : tensor<32x197x197xf32>
    %v1418 = stablehlo.reduce(%v1417 init: %v1416) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1419 = stablehlo.broadcast_in_dim %v1418, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1420 = stablehlo.divide %v1417, %v1419 : tensor<32x197x197xf32>
    %v1421 = stablehlo.reshape %v1420 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1422 = stablehlo.reshape %v1421 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1423 = stablehlo.reshape %v1405 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1424 = stablehlo.dot_general %v1422, %v1423, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1425 = stablehlo.reshape %v1424 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1426 = stablehlo.reshape %v1425 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1427 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1428 = stablehlo.pad %v1426, %v1427, low = [0, 0, 640], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v1429 = stablehlo.reshape %v1428 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1430 = stablehlo.add %v1396, %v1429 : tensor<32x151296xf32>
    %v1431 = stablehlo.reshape %v1047 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1432 = stablehlo.slice %v1431 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1433 = stablehlo.reshape %v1432 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1434 = stablehlo.reshape %v1052 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1435 = stablehlo.slice %v1434 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1436 = stablehlo.reshape %v1435 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1437 = stablehlo.reshape %v1057 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1438 = stablehlo.slice %v1437 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1439 = stablehlo.reshape %v1438 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1440 = stablehlo.reshape %v1436 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1441 = stablehlo.transpose %v1440, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1442 = stablehlo.reshape %v1441 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1443 = stablehlo.reshape %v1433 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1444 = stablehlo.reshape %v1442 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1445 = stablehlo.dot_general %v1443, %v1444, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1446 = stablehlo.reshape %v1445 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1447 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1448 = stablehlo.multiply %v1446, %v1447 : tensor<32x38809xf32>
    %v1449 = stablehlo.reshape %v1448 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1450 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1451 = stablehlo.exponential %v1449 : tensor<32x197x197xf32>
    %v1452 = stablehlo.reduce(%v1451 init: %v1450) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1453 = stablehlo.broadcast_in_dim %v1452, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1454 = stablehlo.divide %v1451, %v1453 : tensor<32x197x197xf32>
    %v1455 = stablehlo.reshape %v1454 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1456 = stablehlo.reshape %v1455 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1457 = stablehlo.reshape %v1439 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1458 = stablehlo.dot_general %v1456, %v1457, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1459 = stablehlo.reshape %v1458 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1460 = stablehlo.reshape %v1459 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1461 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1462 = stablehlo.pad %v1460, %v1461, low = [0, 0, 704], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v1463 = stablehlo.reshape %v1462 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1464 = stablehlo.add %v1430, %v1463 : tensor<32x151296xf32>
    %v1465 = stablehlo.reshape %v1464 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1466 = stablehlo.dot_general %v1465, %b2_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v1467 = stablehlo.broadcast_in_dim %b2_bo, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v1468 = stablehlo.add %v1466, %v1467 : tensor<32x197x768xf32>
    %v1469 = stablehlo.reshape %v1468 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1470 = stablehlo.add %v1014, %v1469 : tensor<32x151296xf32>
    %v1471 = stablehlo.reshape %v1470 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1472 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1473 = stablehlo.constant dense<768.0> : tensor<32x197x768xf32>
    %v1474 = stablehlo.constant dense<1.0e-5> : tensor<32x197x768xf32>
    %v1475 = stablehlo.reduce(%v1471 init: %v1472) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1476 = stablehlo.broadcast_in_dim %v1475, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v1477 = stablehlo.divide %v1476, %v1473 : tensor<32x197x768xf32>
    %v1478 = stablehlo.subtract %v1471, %v1477 : tensor<32x197x768xf32>
    %v1479 = stablehlo.multiply %v1478, %v1478 : tensor<32x197x768xf32>
    %v1480 = stablehlo.reduce(%v1479 init: %v1472) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1481 = stablehlo.broadcast_in_dim %v1480, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v1482 = stablehlo.divide %v1481, %v1473 : tensor<32x197x768xf32>
    %v1483 = stablehlo.add %v1482, %v1474 : tensor<32x197x768xf32>
    %v1484 = stablehlo.rsqrt %v1483 : tensor<32x197x768xf32>
    %v1485 = stablehlo.multiply %v1478, %v1484 : tensor<32x197x768xf32>
    %v1486 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v1487 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v1488 = stablehlo.multiply %v1485, %v1486 : tensor<32x197x768xf32>
    %v1489 = stablehlo.add %v1488, %v1487 : tensor<32x197x768xf32>
    %v1490 = stablehlo.reshape %v1489 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1491 = stablehlo.reshape %v1490 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1492 = stablehlo.broadcast_in_dim %b2_g2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v1493 = stablehlo.multiply %v1491, %v1492 : tensor<32x197x768xf32>
    %v1494 = stablehlo.reshape %v1493 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1495 = stablehlo.reshape %v1494 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1496 = stablehlo.broadcast_in_dim %b2_bt2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v1497 = stablehlo.add %v1495, %v1496 : tensor<32x197x768xf32>
    %v1498 = stablehlo.reshape %v1497 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1499 = stablehlo.reshape %v1498 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1500 = stablehlo.dot_general %v1499, %b2_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x3072xf32>) -> tensor<32x197x3072xf32>
    %v1501 = stablehlo.broadcast_in_dim %b2_bfc1, dims = [2] : (tensor<3072xf32>) -> tensor<32x197x3072xf32>
    %v1502 = stablehlo.add %v1500, %v1501 : tensor<32x197x3072xf32>
    %v1503 = stablehlo.reshape %v1502 : (tensor<32x197x3072xf32>) -> tensor<32x605184xf32>
    %v1504 = stablehlo.constant dense<0.5> : tensor<32x605184xf32>
    %v1505 = stablehlo.multiply %v1504, %v1503 : tensor<32x605184xf32>
    %v1506 = stablehlo.negate %v1503 : tensor<32x605184xf32>
    %v1507 = stablehlo.constant dense<0.7071067811865476> : tensor<32x605184xf32>
    %v1508 = stablehlo.multiply %v1506, %v1507 : tensor<32x605184xf32>
    %v1509 = chlo.erfc %v1508 : tensor<32x605184xf32> -> tensor<32x605184xf32>
    %v1510 = stablehlo.multiply %v1505, %v1509 : tensor<32x605184xf32>
    %v1511 = stablehlo.reshape %v1510 : (tensor<32x605184xf32>) -> tensor<32x197x3072xf32>
    %v1512 = stablehlo.dot_general %v1511, %b2_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x3072xf32>, tensor<3072x768xf32>) -> tensor<32x197x768xf32>
    %v1513 = stablehlo.broadcast_in_dim %b2_bfc2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v1514 = stablehlo.add %v1512, %v1513 : tensor<32x197x768xf32>
    %v1515 = stablehlo.reshape %v1514 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1516 = stablehlo.add %v1470, %v1515 : tensor<32x151296xf32>
    %v1517 = stablehlo.reshape %v1516 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1518 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1519 = stablehlo.constant dense<768.0> : tensor<32x197x768xf32>
    %v1520 = stablehlo.constant dense<1.0e-5> : tensor<32x197x768xf32>
    %v1521 = stablehlo.reduce(%v1517 init: %v1518) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1522 = stablehlo.broadcast_in_dim %v1521, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v1523 = stablehlo.divide %v1522, %v1519 : tensor<32x197x768xf32>
    %v1524 = stablehlo.subtract %v1517, %v1523 : tensor<32x197x768xf32>
    %v1525 = stablehlo.multiply %v1524, %v1524 : tensor<32x197x768xf32>
    %v1526 = stablehlo.reduce(%v1525 init: %v1518) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1527 = stablehlo.broadcast_in_dim %v1526, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v1528 = stablehlo.divide %v1527, %v1519 : tensor<32x197x768xf32>
    %v1529 = stablehlo.add %v1528, %v1520 : tensor<32x197x768xf32>
    %v1530 = stablehlo.rsqrt %v1529 : tensor<32x197x768xf32>
    %v1531 = stablehlo.multiply %v1524, %v1530 : tensor<32x197x768xf32>
    %v1532 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v1533 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v1534 = stablehlo.multiply %v1531, %v1532 : tensor<32x197x768xf32>
    %v1535 = stablehlo.add %v1534, %v1533 : tensor<32x197x768xf32>
    %v1536 = stablehlo.reshape %v1535 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1537 = stablehlo.reshape %v1536 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1538 = stablehlo.broadcast_in_dim %b3_g1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v1539 = stablehlo.multiply %v1537, %v1538 : tensor<32x197x768xf32>
    %v1540 = stablehlo.reshape %v1539 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1541 = stablehlo.reshape %v1540 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1542 = stablehlo.broadcast_in_dim %b3_bt1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v1543 = stablehlo.add %v1541, %v1542 : tensor<32x197x768xf32>
    %v1544 = stablehlo.reshape %v1543 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1545 = stablehlo.reshape %v1544 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1546 = stablehlo.dot_general %v1545, %b3_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v1547 = stablehlo.broadcast_in_dim %b3_bq, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v1548 = stablehlo.add %v1546, %v1547 : tensor<32x197x768xf32>
    %v1549 = stablehlo.reshape %v1548 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1550 = stablehlo.reshape %v1544 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1551 = stablehlo.dot_general %v1550, %b3_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v1552 = stablehlo.broadcast_in_dim %b3_bk, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v1553 = stablehlo.add %v1551, %v1552 : tensor<32x197x768xf32>
    %v1554 = stablehlo.reshape %v1553 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1555 = stablehlo.reshape %v1544 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1556 = stablehlo.dot_general %v1555, %b3_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v1557 = stablehlo.broadcast_in_dim %b3_bv, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v1558 = stablehlo.add %v1556, %v1557 : tensor<32x197x768xf32>
    %v1559 = stablehlo.reshape %v1558 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1560 = stablehlo.reshape %v1549 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1561 = stablehlo.slice %v1560 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1562 = stablehlo.reshape %v1561 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1563 = stablehlo.reshape %v1554 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1564 = stablehlo.slice %v1563 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1565 = stablehlo.reshape %v1564 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1566 = stablehlo.reshape %v1559 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1567 = stablehlo.slice %v1566 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1568 = stablehlo.reshape %v1567 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1569 = stablehlo.reshape %v1565 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1570 = stablehlo.transpose %v1569, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1571 = stablehlo.reshape %v1570 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1572 = stablehlo.reshape %v1562 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1573 = stablehlo.reshape %v1571 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1574 = stablehlo.dot_general %v1572, %v1573, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1575 = stablehlo.reshape %v1574 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1576 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1577 = stablehlo.multiply %v1575, %v1576 : tensor<32x38809xf32>
    %v1578 = stablehlo.reshape %v1577 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1579 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1580 = stablehlo.exponential %v1578 : tensor<32x197x197xf32>
    %v1581 = stablehlo.reduce(%v1580 init: %v1579) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1582 = stablehlo.broadcast_in_dim %v1581, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1583 = stablehlo.divide %v1580, %v1582 : tensor<32x197x197xf32>
    %v1584 = stablehlo.reshape %v1583 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1585 = stablehlo.reshape %v1584 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1586 = stablehlo.reshape %v1568 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1587 = stablehlo.dot_general %v1585, %v1586, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1588 = stablehlo.reshape %v1587 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1589 = stablehlo.reshape %v1588 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1590 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1591 = stablehlo.pad %v1589, %v1590, low = [0, 0, 0], high = [0, 0, 704], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v1592 = stablehlo.reshape %v1591 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1593 = stablehlo.reshape %v1549 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1594 = stablehlo.slice %v1593 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1595 = stablehlo.reshape %v1594 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1596 = stablehlo.reshape %v1554 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1597 = stablehlo.slice %v1596 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1598 = stablehlo.reshape %v1597 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1599 = stablehlo.reshape %v1559 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1600 = stablehlo.slice %v1599 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1601 = stablehlo.reshape %v1600 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1602 = stablehlo.reshape %v1598 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1603 = stablehlo.transpose %v1602, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1604 = stablehlo.reshape %v1603 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1605 = stablehlo.reshape %v1595 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1606 = stablehlo.reshape %v1604 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1607 = stablehlo.dot_general %v1605, %v1606, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1608 = stablehlo.reshape %v1607 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1609 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1610 = stablehlo.multiply %v1608, %v1609 : tensor<32x38809xf32>
    %v1611 = stablehlo.reshape %v1610 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1612 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1613 = stablehlo.exponential %v1611 : tensor<32x197x197xf32>
    %v1614 = stablehlo.reduce(%v1613 init: %v1612) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1615 = stablehlo.broadcast_in_dim %v1614, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1616 = stablehlo.divide %v1613, %v1615 : tensor<32x197x197xf32>
    %v1617 = stablehlo.reshape %v1616 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1618 = stablehlo.reshape %v1617 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1619 = stablehlo.reshape %v1601 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1620 = stablehlo.dot_general %v1618, %v1619, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1621 = stablehlo.reshape %v1620 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1622 = stablehlo.reshape %v1621 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1623 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1624 = stablehlo.pad %v1622, %v1623, low = [0, 0, 64], high = [0, 0, 640], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v1625 = stablehlo.reshape %v1624 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1626 = stablehlo.add %v1592, %v1625 : tensor<32x151296xf32>
    %v1627 = stablehlo.reshape %v1549 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1628 = stablehlo.slice %v1627 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1629 = stablehlo.reshape %v1628 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1630 = stablehlo.reshape %v1554 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1631 = stablehlo.slice %v1630 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1632 = stablehlo.reshape %v1631 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1633 = stablehlo.reshape %v1559 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1634 = stablehlo.slice %v1633 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1635 = stablehlo.reshape %v1634 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1636 = stablehlo.reshape %v1632 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1637 = stablehlo.transpose %v1636, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1638 = stablehlo.reshape %v1637 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1639 = stablehlo.reshape %v1629 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1640 = stablehlo.reshape %v1638 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1641 = stablehlo.dot_general %v1639, %v1640, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1642 = stablehlo.reshape %v1641 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1643 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1644 = stablehlo.multiply %v1642, %v1643 : tensor<32x38809xf32>
    %v1645 = stablehlo.reshape %v1644 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1646 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1647 = stablehlo.exponential %v1645 : tensor<32x197x197xf32>
    %v1648 = stablehlo.reduce(%v1647 init: %v1646) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1649 = stablehlo.broadcast_in_dim %v1648, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1650 = stablehlo.divide %v1647, %v1649 : tensor<32x197x197xf32>
    %v1651 = stablehlo.reshape %v1650 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1652 = stablehlo.reshape %v1651 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1653 = stablehlo.reshape %v1635 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1654 = stablehlo.dot_general %v1652, %v1653, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1655 = stablehlo.reshape %v1654 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1656 = stablehlo.reshape %v1655 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1657 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1658 = stablehlo.pad %v1656, %v1657, low = [0, 0, 128], high = [0, 0, 576], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v1659 = stablehlo.reshape %v1658 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1660 = stablehlo.add %v1626, %v1659 : tensor<32x151296xf32>
    %v1661 = stablehlo.reshape %v1549 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1662 = stablehlo.slice %v1661 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1663 = stablehlo.reshape %v1662 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1664 = stablehlo.reshape %v1554 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1665 = stablehlo.slice %v1664 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1666 = stablehlo.reshape %v1665 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1667 = stablehlo.reshape %v1559 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1668 = stablehlo.slice %v1667 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1669 = stablehlo.reshape %v1668 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1670 = stablehlo.reshape %v1666 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1671 = stablehlo.transpose %v1670, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1672 = stablehlo.reshape %v1671 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1673 = stablehlo.reshape %v1663 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1674 = stablehlo.reshape %v1672 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1675 = stablehlo.dot_general %v1673, %v1674, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1676 = stablehlo.reshape %v1675 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1677 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1678 = stablehlo.multiply %v1676, %v1677 : tensor<32x38809xf32>
    %v1679 = stablehlo.reshape %v1678 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1680 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1681 = stablehlo.exponential %v1679 : tensor<32x197x197xf32>
    %v1682 = stablehlo.reduce(%v1681 init: %v1680) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1683 = stablehlo.broadcast_in_dim %v1682, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1684 = stablehlo.divide %v1681, %v1683 : tensor<32x197x197xf32>
    %v1685 = stablehlo.reshape %v1684 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1686 = stablehlo.reshape %v1685 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1687 = stablehlo.reshape %v1669 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1688 = stablehlo.dot_general %v1686, %v1687, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1689 = stablehlo.reshape %v1688 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1690 = stablehlo.reshape %v1689 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1691 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1692 = stablehlo.pad %v1690, %v1691, low = [0, 0, 192], high = [0, 0, 512], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v1693 = stablehlo.reshape %v1692 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1694 = stablehlo.add %v1660, %v1693 : tensor<32x151296xf32>
    %v1695 = stablehlo.reshape %v1549 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1696 = stablehlo.slice %v1695 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1697 = stablehlo.reshape %v1696 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1698 = stablehlo.reshape %v1554 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1699 = stablehlo.slice %v1698 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1700 = stablehlo.reshape %v1699 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1701 = stablehlo.reshape %v1559 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1702 = stablehlo.slice %v1701 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1703 = stablehlo.reshape %v1702 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1704 = stablehlo.reshape %v1700 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1705 = stablehlo.transpose %v1704, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1706 = stablehlo.reshape %v1705 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1707 = stablehlo.reshape %v1697 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1708 = stablehlo.reshape %v1706 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1709 = stablehlo.dot_general %v1707, %v1708, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1710 = stablehlo.reshape %v1709 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1711 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1712 = stablehlo.multiply %v1710, %v1711 : tensor<32x38809xf32>
    %v1713 = stablehlo.reshape %v1712 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1714 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1715 = stablehlo.exponential %v1713 : tensor<32x197x197xf32>
    %v1716 = stablehlo.reduce(%v1715 init: %v1714) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1717 = stablehlo.broadcast_in_dim %v1716, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1718 = stablehlo.divide %v1715, %v1717 : tensor<32x197x197xf32>
    %v1719 = stablehlo.reshape %v1718 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1720 = stablehlo.reshape %v1719 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1721 = stablehlo.reshape %v1703 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1722 = stablehlo.dot_general %v1720, %v1721, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1723 = stablehlo.reshape %v1722 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1724 = stablehlo.reshape %v1723 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1725 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1726 = stablehlo.pad %v1724, %v1725, low = [0, 0, 256], high = [0, 0, 448], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v1727 = stablehlo.reshape %v1726 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1728 = stablehlo.add %v1694, %v1727 : tensor<32x151296xf32>
    %v1729 = stablehlo.reshape %v1549 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1730 = stablehlo.slice %v1729 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1731 = stablehlo.reshape %v1730 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1732 = stablehlo.reshape %v1554 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1733 = stablehlo.slice %v1732 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1734 = stablehlo.reshape %v1733 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1735 = stablehlo.reshape %v1559 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1736 = stablehlo.slice %v1735 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1737 = stablehlo.reshape %v1736 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1738 = stablehlo.reshape %v1734 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1739 = stablehlo.transpose %v1738, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1740 = stablehlo.reshape %v1739 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1741 = stablehlo.reshape %v1731 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1742 = stablehlo.reshape %v1740 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1743 = stablehlo.dot_general %v1741, %v1742, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1744 = stablehlo.reshape %v1743 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1745 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1746 = stablehlo.multiply %v1744, %v1745 : tensor<32x38809xf32>
    %v1747 = stablehlo.reshape %v1746 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1748 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1749 = stablehlo.exponential %v1747 : tensor<32x197x197xf32>
    %v1750 = stablehlo.reduce(%v1749 init: %v1748) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1751 = stablehlo.broadcast_in_dim %v1750, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1752 = stablehlo.divide %v1749, %v1751 : tensor<32x197x197xf32>
    %v1753 = stablehlo.reshape %v1752 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1754 = stablehlo.reshape %v1753 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1755 = stablehlo.reshape %v1737 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1756 = stablehlo.dot_general %v1754, %v1755, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1757 = stablehlo.reshape %v1756 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1758 = stablehlo.reshape %v1757 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1759 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1760 = stablehlo.pad %v1758, %v1759, low = [0, 0, 320], high = [0, 0, 384], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v1761 = stablehlo.reshape %v1760 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1762 = stablehlo.add %v1728, %v1761 : tensor<32x151296xf32>
    %v1763 = stablehlo.reshape %v1549 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1764 = stablehlo.slice %v1763 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1765 = stablehlo.reshape %v1764 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1766 = stablehlo.reshape %v1554 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1767 = stablehlo.slice %v1766 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1768 = stablehlo.reshape %v1767 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1769 = stablehlo.reshape %v1559 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1770 = stablehlo.slice %v1769 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1771 = stablehlo.reshape %v1770 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1772 = stablehlo.reshape %v1768 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1773 = stablehlo.transpose %v1772, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1774 = stablehlo.reshape %v1773 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1775 = stablehlo.reshape %v1765 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1776 = stablehlo.reshape %v1774 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1777 = stablehlo.dot_general %v1775, %v1776, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1778 = stablehlo.reshape %v1777 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1779 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1780 = stablehlo.multiply %v1778, %v1779 : tensor<32x38809xf32>
    %v1781 = stablehlo.reshape %v1780 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1782 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1783 = stablehlo.exponential %v1781 : tensor<32x197x197xf32>
    %v1784 = stablehlo.reduce(%v1783 init: %v1782) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1785 = stablehlo.broadcast_in_dim %v1784, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1786 = stablehlo.divide %v1783, %v1785 : tensor<32x197x197xf32>
    %v1787 = stablehlo.reshape %v1786 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1788 = stablehlo.reshape %v1787 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1789 = stablehlo.reshape %v1771 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1790 = stablehlo.dot_general %v1788, %v1789, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1791 = stablehlo.reshape %v1790 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1792 = stablehlo.reshape %v1791 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1793 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1794 = stablehlo.pad %v1792, %v1793, low = [0, 0, 384], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v1795 = stablehlo.reshape %v1794 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1796 = stablehlo.add %v1762, %v1795 : tensor<32x151296xf32>
    %v1797 = stablehlo.reshape %v1549 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1798 = stablehlo.slice %v1797 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1799 = stablehlo.reshape %v1798 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1800 = stablehlo.reshape %v1554 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1801 = stablehlo.slice %v1800 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1802 = stablehlo.reshape %v1801 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1803 = stablehlo.reshape %v1559 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1804 = stablehlo.slice %v1803 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1805 = stablehlo.reshape %v1804 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1806 = stablehlo.reshape %v1802 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1807 = stablehlo.transpose %v1806, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1808 = stablehlo.reshape %v1807 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1809 = stablehlo.reshape %v1799 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1810 = stablehlo.reshape %v1808 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1811 = stablehlo.dot_general %v1809, %v1810, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1812 = stablehlo.reshape %v1811 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1813 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1814 = stablehlo.multiply %v1812, %v1813 : tensor<32x38809xf32>
    %v1815 = stablehlo.reshape %v1814 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1816 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1817 = stablehlo.exponential %v1815 : tensor<32x197x197xf32>
    %v1818 = stablehlo.reduce(%v1817 init: %v1816) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1819 = stablehlo.broadcast_in_dim %v1818, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1820 = stablehlo.divide %v1817, %v1819 : tensor<32x197x197xf32>
    %v1821 = stablehlo.reshape %v1820 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1822 = stablehlo.reshape %v1821 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1823 = stablehlo.reshape %v1805 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1824 = stablehlo.dot_general %v1822, %v1823, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1825 = stablehlo.reshape %v1824 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1826 = stablehlo.reshape %v1825 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1827 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1828 = stablehlo.pad %v1826, %v1827, low = [0, 0, 448], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v1829 = stablehlo.reshape %v1828 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1830 = stablehlo.add %v1796, %v1829 : tensor<32x151296xf32>
    %v1831 = stablehlo.reshape %v1549 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1832 = stablehlo.slice %v1831 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1833 = stablehlo.reshape %v1832 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1834 = stablehlo.reshape %v1554 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1835 = stablehlo.slice %v1834 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1836 = stablehlo.reshape %v1835 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1837 = stablehlo.reshape %v1559 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1838 = stablehlo.slice %v1837 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1839 = stablehlo.reshape %v1838 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1840 = stablehlo.reshape %v1836 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1841 = stablehlo.transpose %v1840, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1842 = stablehlo.reshape %v1841 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1843 = stablehlo.reshape %v1833 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1844 = stablehlo.reshape %v1842 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1845 = stablehlo.dot_general %v1843, %v1844, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1846 = stablehlo.reshape %v1845 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1847 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1848 = stablehlo.multiply %v1846, %v1847 : tensor<32x38809xf32>
    %v1849 = stablehlo.reshape %v1848 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1850 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1851 = stablehlo.exponential %v1849 : tensor<32x197x197xf32>
    %v1852 = stablehlo.reduce(%v1851 init: %v1850) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1853 = stablehlo.broadcast_in_dim %v1852, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1854 = stablehlo.divide %v1851, %v1853 : tensor<32x197x197xf32>
    %v1855 = stablehlo.reshape %v1854 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1856 = stablehlo.reshape %v1855 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1857 = stablehlo.reshape %v1839 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1858 = stablehlo.dot_general %v1856, %v1857, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1859 = stablehlo.reshape %v1858 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1860 = stablehlo.reshape %v1859 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1861 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1862 = stablehlo.pad %v1860, %v1861, low = [0, 0, 512], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v1863 = stablehlo.reshape %v1862 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1864 = stablehlo.add %v1830, %v1863 : tensor<32x151296xf32>
    %v1865 = stablehlo.reshape %v1549 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1866 = stablehlo.slice %v1865 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1867 = stablehlo.reshape %v1866 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1868 = stablehlo.reshape %v1554 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1869 = stablehlo.slice %v1868 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1870 = stablehlo.reshape %v1869 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1871 = stablehlo.reshape %v1559 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1872 = stablehlo.slice %v1871 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1873 = stablehlo.reshape %v1872 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1874 = stablehlo.reshape %v1870 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1875 = stablehlo.transpose %v1874, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1876 = stablehlo.reshape %v1875 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1877 = stablehlo.reshape %v1867 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1878 = stablehlo.reshape %v1876 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1879 = stablehlo.dot_general %v1877, %v1878, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1880 = stablehlo.reshape %v1879 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1881 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1882 = stablehlo.multiply %v1880, %v1881 : tensor<32x38809xf32>
    %v1883 = stablehlo.reshape %v1882 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1884 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1885 = stablehlo.exponential %v1883 : tensor<32x197x197xf32>
    %v1886 = stablehlo.reduce(%v1885 init: %v1884) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1887 = stablehlo.broadcast_in_dim %v1886, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1888 = stablehlo.divide %v1885, %v1887 : tensor<32x197x197xf32>
    %v1889 = stablehlo.reshape %v1888 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1890 = stablehlo.reshape %v1889 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1891 = stablehlo.reshape %v1873 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1892 = stablehlo.dot_general %v1890, %v1891, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1893 = stablehlo.reshape %v1892 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1894 = stablehlo.reshape %v1893 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1895 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1896 = stablehlo.pad %v1894, %v1895, low = [0, 0, 576], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v1897 = stablehlo.reshape %v1896 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1898 = stablehlo.add %v1864, %v1897 : tensor<32x151296xf32>
    %v1899 = stablehlo.reshape %v1549 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1900 = stablehlo.slice %v1899 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1901 = stablehlo.reshape %v1900 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1902 = stablehlo.reshape %v1554 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1903 = stablehlo.slice %v1902 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1904 = stablehlo.reshape %v1903 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1905 = stablehlo.reshape %v1559 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1906 = stablehlo.slice %v1905 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1907 = stablehlo.reshape %v1906 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1908 = stablehlo.reshape %v1904 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1909 = stablehlo.transpose %v1908, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1910 = stablehlo.reshape %v1909 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1911 = stablehlo.reshape %v1901 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1912 = stablehlo.reshape %v1910 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1913 = stablehlo.dot_general %v1911, %v1912, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1914 = stablehlo.reshape %v1913 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1915 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1916 = stablehlo.multiply %v1914, %v1915 : tensor<32x38809xf32>
    %v1917 = stablehlo.reshape %v1916 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1918 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1919 = stablehlo.exponential %v1917 : tensor<32x197x197xf32>
    %v1920 = stablehlo.reduce(%v1919 init: %v1918) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1921 = stablehlo.broadcast_in_dim %v1920, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1922 = stablehlo.divide %v1919, %v1921 : tensor<32x197x197xf32>
    %v1923 = stablehlo.reshape %v1922 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1924 = stablehlo.reshape %v1923 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1925 = stablehlo.reshape %v1907 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1926 = stablehlo.dot_general %v1924, %v1925, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1927 = stablehlo.reshape %v1926 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1928 = stablehlo.reshape %v1927 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1929 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1930 = stablehlo.pad %v1928, %v1929, low = [0, 0, 640], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v1931 = stablehlo.reshape %v1930 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1932 = stablehlo.add %v1898, %v1931 : tensor<32x151296xf32>
    %v1933 = stablehlo.reshape %v1549 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1934 = stablehlo.slice %v1933 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1935 = stablehlo.reshape %v1934 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1936 = stablehlo.reshape %v1554 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1937 = stablehlo.slice %v1936 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1938 = stablehlo.reshape %v1937 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1939 = stablehlo.reshape %v1559 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1940 = stablehlo.slice %v1939 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1941 = stablehlo.reshape %v1940 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1942 = stablehlo.reshape %v1938 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1943 = stablehlo.transpose %v1942, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1944 = stablehlo.reshape %v1943 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1945 = stablehlo.reshape %v1935 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1946 = stablehlo.reshape %v1944 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1947 = stablehlo.dot_general %v1945, %v1946, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1948 = stablehlo.reshape %v1947 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1949 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1950 = stablehlo.multiply %v1948, %v1949 : tensor<32x38809xf32>
    %v1951 = stablehlo.reshape %v1950 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1952 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1953 = stablehlo.exponential %v1951 : tensor<32x197x197xf32>
    %v1954 = stablehlo.reduce(%v1953 init: %v1952) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1955 = stablehlo.broadcast_in_dim %v1954, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1956 = stablehlo.divide %v1953, %v1955 : tensor<32x197x197xf32>
    %v1957 = stablehlo.reshape %v1956 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1958 = stablehlo.reshape %v1957 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1959 = stablehlo.reshape %v1941 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1960 = stablehlo.dot_general %v1958, %v1959, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1961 = stablehlo.reshape %v1960 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1962 = stablehlo.reshape %v1961 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1963 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1964 = stablehlo.pad %v1962, %v1963, low = [0, 0, 704], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v1965 = stablehlo.reshape %v1964 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1966 = stablehlo.add %v1932, %v1965 : tensor<32x151296xf32>
    %v1967 = stablehlo.reshape %v1966 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1968 = stablehlo.dot_general %v1967, %b3_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v1969 = stablehlo.broadcast_in_dim %b3_bo, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v1970 = stablehlo.add %v1968, %v1969 : tensor<32x197x768xf32>
    %v1971 = stablehlo.reshape %v1970 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1972 = stablehlo.add %v1516, %v1971 : tensor<32x151296xf32>
    %v1973 = stablehlo.reshape %v1972 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1974 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1975 = stablehlo.constant dense<768.0> : tensor<32x197x768xf32>
    %v1976 = stablehlo.constant dense<1.0e-5> : tensor<32x197x768xf32>
    %v1977 = stablehlo.reduce(%v1973 init: %v1974) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1978 = stablehlo.broadcast_in_dim %v1977, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v1979 = stablehlo.divide %v1978, %v1975 : tensor<32x197x768xf32>
    %v1980 = stablehlo.subtract %v1973, %v1979 : tensor<32x197x768xf32>
    %v1981 = stablehlo.multiply %v1980, %v1980 : tensor<32x197x768xf32>
    %v1982 = stablehlo.reduce(%v1981 init: %v1974) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1983 = stablehlo.broadcast_in_dim %v1982, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v1984 = stablehlo.divide %v1983, %v1975 : tensor<32x197x768xf32>
    %v1985 = stablehlo.add %v1984, %v1976 : tensor<32x197x768xf32>
    %v1986 = stablehlo.rsqrt %v1985 : tensor<32x197x768xf32>
    %v1987 = stablehlo.multiply %v1980, %v1986 : tensor<32x197x768xf32>
    %v1988 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v1989 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v1990 = stablehlo.multiply %v1987, %v1988 : tensor<32x197x768xf32>
    %v1991 = stablehlo.add %v1990, %v1989 : tensor<32x197x768xf32>
    %v1992 = stablehlo.reshape %v1991 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1993 = stablehlo.reshape %v1992 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1994 = stablehlo.broadcast_in_dim %b3_g2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v1995 = stablehlo.multiply %v1993, %v1994 : tensor<32x197x768xf32>
    %v1996 = stablehlo.reshape %v1995 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1997 = stablehlo.reshape %v1996 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1998 = stablehlo.broadcast_in_dim %b3_bt2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v1999 = stablehlo.add %v1997, %v1998 : tensor<32x197x768xf32>
    %v2000 = stablehlo.reshape %v1999 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2001 = stablehlo.reshape %v2000 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2002 = stablehlo.dot_general %v2001, %b3_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x3072xf32>) -> tensor<32x197x3072xf32>
    %v2003 = stablehlo.broadcast_in_dim %b3_bfc1, dims = [2] : (tensor<3072xf32>) -> tensor<32x197x3072xf32>
    %v2004 = stablehlo.add %v2002, %v2003 : tensor<32x197x3072xf32>
    %v2005 = stablehlo.reshape %v2004 : (tensor<32x197x3072xf32>) -> tensor<32x605184xf32>
    %v2006 = stablehlo.constant dense<0.5> : tensor<32x605184xf32>
    %v2007 = stablehlo.multiply %v2006, %v2005 : tensor<32x605184xf32>
    %v2008 = stablehlo.negate %v2005 : tensor<32x605184xf32>
    %v2009 = stablehlo.constant dense<0.7071067811865476> : tensor<32x605184xf32>
    %v2010 = stablehlo.multiply %v2008, %v2009 : tensor<32x605184xf32>
    %v2011 = chlo.erfc %v2010 : tensor<32x605184xf32> -> tensor<32x605184xf32>
    %v2012 = stablehlo.multiply %v2007, %v2011 : tensor<32x605184xf32>
    %v2013 = stablehlo.reshape %v2012 : (tensor<32x605184xf32>) -> tensor<32x197x3072xf32>
    %v2014 = stablehlo.dot_general %v2013, %b3_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x3072xf32>, tensor<3072x768xf32>) -> tensor<32x197x768xf32>
    %v2015 = stablehlo.broadcast_in_dim %b3_bfc2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v2016 = stablehlo.add %v2014, %v2015 : tensor<32x197x768xf32>
    %v2017 = stablehlo.reshape %v2016 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2018 = stablehlo.add %v1972, %v2017 : tensor<32x151296xf32>
    %v2019 = stablehlo.reshape %v2018 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2020 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2021 = stablehlo.constant dense<768.0> : tensor<32x197x768xf32>
    %v2022 = stablehlo.constant dense<1.0e-5> : tensor<32x197x768xf32>
    %v2023 = stablehlo.reduce(%v2019 init: %v2020) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2024 = stablehlo.broadcast_in_dim %v2023, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v2025 = stablehlo.divide %v2024, %v2021 : tensor<32x197x768xf32>
    %v2026 = stablehlo.subtract %v2019, %v2025 : tensor<32x197x768xf32>
    %v2027 = stablehlo.multiply %v2026, %v2026 : tensor<32x197x768xf32>
    %v2028 = stablehlo.reduce(%v2027 init: %v2020) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2029 = stablehlo.broadcast_in_dim %v2028, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v2030 = stablehlo.divide %v2029, %v2021 : tensor<32x197x768xf32>
    %v2031 = stablehlo.add %v2030, %v2022 : tensor<32x197x768xf32>
    %v2032 = stablehlo.rsqrt %v2031 : tensor<32x197x768xf32>
    %v2033 = stablehlo.multiply %v2026, %v2032 : tensor<32x197x768xf32>
    %v2034 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v2035 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v2036 = stablehlo.multiply %v2033, %v2034 : tensor<32x197x768xf32>
    %v2037 = stablehlo.add %v2036, %v2035 : tensor<32x197x768xf32>
    %v2038 = stablehlo.reshape %v2037 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2039 = stablehlo.reshape %v2038 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2040 = stablehlo.broadcast_in_dim %b4_g1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v2041 = stablehlo.multiply %v2039, %v2040 : tensor<32x197x768xf32>
    %v2042 = stablehlo.reshape %v2041 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2043 = stablehlo.reshape %v2042 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2044 = stablehlo.broadcast_in_dim %b4_bt1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v2045 = stablehlo.add %v2043, %v2044 : tensor<32x197x768xf32>
    %v2046 = stablehlo.reshape %v2045 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2047 = stablehlo.reshape %v2046 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2048 = stablehlo.dot_general %v2047, %b4_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v2049 = stablehlo.broadcast_in_dim %b4_bq, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v2050 = stablehlo.add %v2048, %v2049 : tensor<32x197x768xf32>
    %v2051 = stablehlo.reshape %v2050 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2052 = stablehlo.reshape %v2046 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2053 = stablehlo.dot_general %v2052, %b4_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v2054 = stablehlo.broadcast_in_dim %b4_bk, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v2055 = stablehlo.add %v2053, %v2054 : tensor<32x197x768xf32>
    %v2056 = stablehlo.reshape %v2055 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2057 = stablehlo.reshape %v2046 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2058 = stablehlo.dot_general %v2057, %b4_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v2059 = stablehlo.broadcast_in_dim %b4_bv, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v2060 = stablehlo.add %v2058, %v2059 : tensor<32x197x768xf32>
    %v2061 = stablehlo.reshape %v2060 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2062 = stablehlo.reshape %v2051 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2063 = stablehlo.slice %v2062 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2064 = stablehlo.reshape %v2063 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2065 = stablehlo.reshape %v2056 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2066 = stablehlo.slice %v2065 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2067 = stablehlo.reshape %v2066 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2068 = stablehlo.reshape %v2061 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2069 = stablehlo.slice %v2068 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2070 = stablehlo.reshape %v2069 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2071 = stablehlo.reshape %v2067 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2072 = stablehlo.transpose %v2071, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2073 = stablehlo.reshape %v2072 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2074 = stablehlo.reshape %v2064 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2075 = stablehlo.reshape %v2073 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2076 = stablehlo.dot_general %v2074, %v2075, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2077 = stablehlo.reshape %v2076 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2078 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2079 = stablehlo.multiply %v2077, %v2078 : tensor<32x38809xf32>
    %v2080 = stablehlo.reshape %v2079 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2081 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2082 = stablehlo.exponential %v2080 : tensor<32x197x197xf32>
    %v2083 = stablehlo.reduce(%v2082 init: %v2081) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2084 = stablehlo.broadcast_in_dim %v2083, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2085 = stablehlo.divide %v2082, %v2084 : tensor<32x197x197xf32>
    %v2086 = stablehlo.reshape %v2085 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2087 = stablehlo.reshape %v2086 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2088 = stablehlo.reshape %v2070 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2089 = stablehlo.dot_general %v2087, %v2088, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2090 = stablehlo.reshape %v2089 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2091 = stablehlo.reshape %v2090 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2092 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2093 = stablehlo.pad %v2091, %v2092, low = [0, 0, 0], high = [0, 0, 704], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v2094 = stablehlo.reshape %v2093 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2095 = stablehlo.reshape %v2051 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2096 = stablehlo.slice %v2095 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2097 = stablehlo.reshape %v2096 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2098 = stablehlo.reshape %v2056 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2099 = stablehlo.slice %v2098 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2100 = stablehlo.reshape %v2099 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2101 = stablehlo.reshape %v2061 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2102 = stablehlo.slice %v2101 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2103 = stablehlo.reshape %v2102 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2104 = stablehlo.reshape %v2100 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2105 = stablehlo.transpose %v2104, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2106 = stablehlo.reshape %v2105 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2107 = stablehlo.reshape %v2097 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2108 = stablehlo.reshape %v2106 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2109 = stablehlo.dot_general %v2107, %v2108, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2110 = stablehlo.reshape %v2109 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2111 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2112 = stablehlo.multiply %v2110, %v2111 : tensor<32x38809xf32>
    %v2113 = stablehlo.reshape %v2112 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2114 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2115 = stablehlo.exponential %v2113 : tensor<32x197x197xf32>
    %v2116 = stablehlo.reduce(%v2115 init: %v2114) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2117 = stablehlo.broadcast_in_dim %v2116, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2118 = stablehlo.divide %v2115, %v2117 : tensor<32x197x197xf32>
    %v2119 = stablehlo.reshape %v2118 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2120 = stablehlo.reshape %v2119 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2121 = stablehlo.reshape %v2103 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2122 = stablehlo.dot_general %v2120, %v2121, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2123 = stablehlo.reshape %v2122 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2124 = stablehlo.reshape %v2123 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2125 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2126 = stablehlo.pad %v2124, %v2125, low = [0, 0, 64], high = [0, 0, 640], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v2127 = stablehlo.reshape %v2126 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2128 = stablehlo.add %v2094, %v2127 : tensor<32x151296xf32>
    %v2129 = stablehlo.reshape %v2051 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2130 = stablehlo.slice %v2129 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2131 = stablehlo.reshape %v2130 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2132 = stablehlo.reshape %v2056 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2133 = stablehlo.slice %v2132 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2134 = stablehlo.reshape %v2133 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2135 = stablehlo.reshape %v2061 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2136 = stablehlo.slice %v2135 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2137 = stablehlo.reshape %v2136 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2138 = stablehlo.reshape %v2134 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2139 = stablehlo.transpose %v2138, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2140 = stablehlo.reshape %v2139 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2141 = stablehlo.reshape %v2131 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2142 = stablehlo.reshape %v2140 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2143 = stablehlo.dot_general %v2141, %v2142, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2144 = stablehlo.reshape %v2143 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2145 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2146 = stablehlo.multiply %v2144, %v2145 : tensor<32x38809xf32>
    %v2147 = stablehlo.reshape %v2146 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2148 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2149 = stablehlo.exponential %v2147 : tensor<32x197x197xf32>
    %v2150 = stablehlo.reduce(%v2149 init: %v2148) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2151 = stablehlo.broadcast_in_dim %v2150, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2152 = stablehlo.divide %v2149, %v2151 : tensor<32x197x197xf32>
    %v2153 = stablehlo.reshape %v2152 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2154 = stablehlo.reshape %v2153 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2155 = stablehlo.reshape %v2137 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2156 = stablehlo.dot_general %v2154, %v2155, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2157 = stablehlo.reshape %v2156 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2158 = stablehlo.reshape %v2157 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2159 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2160 = stablehlo.pad %v2158, %v2159, low = [0, 0, 128], high = [0, 0, 576], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v2161 = stablehlo.reshape %v2160 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2162 = stablehlo.add %v2128, %v2161 : tensor<32x151296xf32>
    %v2163 = stablehlo.reshape %v2051 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2164 = stablehlo.slice %v2163 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2165 = stablehlo.reshape %v2164 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2166 = stablehlo.reshape %v2056 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2167 = stablehlo.slice %v2166 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2168 = stablehlo.reshape %v2167 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2169 = stablehlo.reshape %v2061 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2170 = stablehlo.slice %v2169 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2171 = stablehlo.reshape %v2170 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2172 = stablehlo.reshape %v2168 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2173 = stablehlo.transpose %v2172, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2174 = stablehlo.reshape %v2173 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2175 = stablehlo.reshape %v2165 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2176 = stablehlo.reshape %v2174 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2177 = stablehlo.dot_general %v2175, %v2176, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2178 = stablehlo.reshape %v2177 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2179 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2180 = stablehlo.multiply %v2178, %v2179 : tensor<32x38809xf32>
    %v2181 = stablehlo.reshape %v2180 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2182 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2183 = stablehlo.exponential %v2181 : tensor<32x197x197xf32>
    %v2184 = stablehlo.reduce(%v2183 init: %v2182) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2185 = stablehlo.broadcast_in_dim %v2184, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2186 = stablehlo.divide %v2183, %v2185 : tensor<32x197x197xf32>
    %v2187 = stablehlo.reshape %v2186 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2188 = stablehlo.reshape %v2187 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2189 = stablehlo.reshape %v2171 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2190 = stablehlo.dot_general %v2188, %v2189, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2191 = stablehlo.reshape %v2190 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2192 = stablehlo.reshape %v2191 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2193 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2194 = stablehlo.pad %v2192, %v2193, low = [0, 0, 192], high = [0, 0, 512], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v2195 = stablehlo.reshape %v2194 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2196 = stablehlo.add %v2162, %v2195 : tensor<32x151296xf32>
    %v2197 = stablehlo.reshape %v2051 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2198 = stablehlo.slice %v2197 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2199 = stablehlo.reshape %v2198 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2200 = stablehlo.reshape %v2056 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2201 = stablehlo.slice %v2200 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2202 = stablehlo.reshape %v2201 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2203 = stablehlo.reshape %v2061 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2204 = stablehlo.slice %v2203 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2205 = stablehlo.reshape %v2204 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2206 = stablehlo.reshape %v2202 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2207 = stablehlo.transpose %v2206, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2208 = stablehlo.reshape %v2207 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2209 = stablehlo.reshape %v2199 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2210 = stablehlo.reshape %v2208 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2211 = stablehlo.dot_general %v2209, %v2210, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2212 = stablehlo.reshape %v2211 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2213 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2214 = stablehlo.multiply %v2212, %v2213 : tensor<32x38809xf32>
    %v2215 = stablehlo.reshape %v2214 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2216 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2217 = stablehlo.exponential %v2215 : tensor<32x197x197xf32>
    %v2218 = stablehlo.reduce(%v2217 init: %v2216) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2219 = stablehlo.broadcast_in_dim %v2218, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2220 = stablehlo.divide %v2217, %v2219 : tensor<32x197x197xf32>
    %v2221 = stablehlo.reshape %v2220 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2222 = stablehlo.reshape %v2221 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2223 = stablehlo.reshape %v2205 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2224 = stablehlo.dot_general %v2222, %v2223, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2225 = stablehlo.reshape %v2224 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2226 = stablehlo.reshape %v2225 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2227 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2228 = stablehlo.pad %v2226, %v2227, low = [0, 0, 256], high = [0, 0, 448], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v2229 = stablehlo.reshape %v2228 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2230 = stablehlo.add %v2196, %v2229 : tensor<32x151296xf32>
    %v2231 = stablehlo.reshape %v2051 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2232 = stablehlo.slice %v2231 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2233 = stablehlo.reshape %v2232 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2234 = stablehlo.reshape %v2056 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2235 = stablehlo.slice %v2234 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2236 = stablehlo.reshape %v2235 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2237 = stablehlo.reshape %v2061 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2238 = stablehlo.slice %v2237 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2239 = stablehlo.reshape %v2238 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2240 = stablehlo.reshape %v2236 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2241 = stablehlo.transpose %v2240, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2242 = stablehlo.reshape %v2241 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2243 = stablehlo.reshape %v2233 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2244 = stablehlo.reshape %v2242 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2245 = stablehlo.dot_general %v2243, %v2244, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2246 = stablehlo.reshape %v2245 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2247 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2248 = stablehlo.multiply %v2246, %v2247 : tensor<32x38809xf32>
    %v2249 = stablehlo.reshape %v2248 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2250 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2251 = stablehlo.exponential %v2249 : tensor<32x197x197xf32>
    %v2252 = stablehlo.reduce(%v2251 init: %v2250) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2253 = stablehlo.broadcast_in_dim %v2252, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2254 = stablehlo.divide %v2251, %v2253 : tensor<32x197x197xf32>
    %v2255 = stablehlo.reshape %v2254 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2256 = stablehlo.reshape %v2255 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2257 = stablehlo.reshape %v2239 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2258 = stablehlo.dot_general %v2256, %v2257, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2259 = stablehlo.reshape %v2258 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2260 = stablehlo.reshape %v2259 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2261 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2262 = stablehlo.pad %v2260, %v2261, low = [0, 0, 320], high = [0, 0, 384], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v2263 = stablehlo.reshape %v2262 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2264 = stablehlo.add %v2230, %v2263 : tensor<32x151296xf32>
    %v2265 = stablehlo.reshape %v2051 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2266 = stablehlo.slice %v2265 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2267 = stablehlo.reshape %v2266 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2268 = stablehlo.reshape %v2056 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2269 = stablehlo.slice %v2268 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2270 = stablehlo.reshape %v2269 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2271 = stablehlo.reshape %v2061 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2272 = stablehlo.slice %v2271 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2273 = stablehlo.reshape %v2272 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2274 = stablehlo.reshape %v2270 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2275 = stablehlo.transpose %v2274, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2276 = stablehlo.reshape %v2275 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2277 = stablehlo.reshape %v2267 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2278 = stablehlo.reshape %v2276 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2279 = stablehlo.dot_general %v2277, %v2278, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2280 = stablehlo.reshape %v2279 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2281 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2282 = stablehlo.multiply %v2280, %v2281 : tensor<32x38809xf32>
    %v2283 = stablehlo.reshape %v2282 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2284 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2285 = stablehlo.exponential %v2283 : tensor<32x197x197xf32>
    %v2286 = stablehlo.reduce(%v2285 init: %v2284) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2287 = stablehlo.broadcast_in_dim %v2286, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2288 = stablehlo.divide %v2285, %v2287 : tensor<32x197x197xf32>
    %v2289 = stablehlo.reshape %v2288 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2290 = stablehlo.reshape %v2289 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2291 = stablehlo.reshape %v2273 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2292 = stablehlo.dot_general %v2290, %v2291, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2293 = stablehlo.reshape %v2292 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2294 = stablehlo.reshape %v2293 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2295 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2296 = stablehlo.pad %v2294, %v2295, low = [0, 0, 384], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v2297 = stablehlo.reshape %v2296 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2298 = stablehlo.add %v2264, %v2297 : tensor<32x151296xf32>
    %v2299 = stablehlo.reshape %v2051 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2300 = stablehlo.slice %v2299 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2301 = stablehlo.reshape %v2300 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2302 = stablehlo.reshape %v2056 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2303 = stablehlo.slice %v2302 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2304 = stablehlo.reshape %v2303 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2305 = stablehlo.reshape %v2061 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2306 = stablehlo.slice %v2305 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2307 = stablehlo.reshape %v2306 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2308 = stablehlo.reshape %v2304 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2309 = stablehlo.transpose %v2308, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2310 = stablehlo.reshape %v2309 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2311 = stablehlo.reshape %v2301 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2312 = stablehlo.reshape %v2310 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2313 = stablehlo.dot_general %v2311, %v2312, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2314 = stablehlo.reshape %v2313 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2315 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2316 = stablehlo.multiply %v2314, %v2315 : tensor<32x38809xf32>
    %v2317 = stablehlo.reshape %v2316 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2318 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2319 = stablehlo.exponential %v2317 : tensor<32x197x197xf32>
    %v2320 = stablehlo.reduce(%v2319 init: %v2318) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2321 = stablehlo.broadcast_in_dim %v2320, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2322 = stablehlo.divide %v2319, %v2321 : tensor<32x197x197xf32>
    %v2323 = stablehlo.reshape %v2322 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2324 = stablehlo.reshape %v2323 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2325 = stablehlo.reshape %v2307 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2326 = stablehlo.dot_general %v2324, %v2325, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2327 = stablehlo.reshape %v2326 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2328 = stablehlo.reshape %v2327 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2329 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2330 = stablehlo.pad %v2328, %v2329, low = [0, 0, 448], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v2331 = stablehlo.reshape %v2330 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2332 = stablehlo.add %v2298, %v2331 : tensor<32x151296xf32>
    %v2333 = stablehlo.reshape %v2051 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2334 = stablehlo.slice %v2333 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2335 = stablehlo.reshape %v2334 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2336 = stablehlo.reshape %v2056 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2337 = stablehlo.slice %v2336 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2338 = stablehlo.reshape %v2337 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2339 = stablehlo.reshape %v2061 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2340 = stablehlo.slice %v2339 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2341 = stablehlo.reshape %v2340 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2342 = stablehlo.reshape %v2338 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2343 = stablehlo.transpose %v2342, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2344 = stablehlo.reshape %v2343 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2345 = stablehlo.reshape %v2335 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2346 = stablehlo.reshape %v2344 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2347 = stablehlo.dot_general %v2345, %v2346, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2348 = stablehlo.reshape %v2347 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2349 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2350 = stablehlo.multiply %v2348, %v2349 : tensor<32x38809xf32>
    %v2351 = stablehlo.reshape %v2350 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2352 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2353 = stablehlo.exponential %v2351 : tensor<32x197x197xf32>
    %v2354 = stablehlo.reduce(%v2353 init: %v2352) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2355 = stablehlo.broadcast_in_dim %v2354, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2356 = stablehlo.divide %v2353, %v2355 : tensor<32x197x197xf32>
    %v2357 = stablehlo.reshape %v2356 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2358 = stablehlo.reshape %v2357 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2359 = stablehlo.reshape %v2341 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2360 = stablehlo.dot_general %v2358, %v2359, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2361 = stablehlo.reshape %v2360 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2362 = stablehlo.reshape %v2361 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2363 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2364 = stablehlo.pad %v2362, %v2363, low = [0, 0, 512], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v2365 = stablehlo.reshape %v2364 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2366 = stablehlo.add %v2332, %v2365 : tensor<32x151296xf32>
    %v2367 = stablehlo.reshape %v2051 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2368 = stablehlo.slice %v2367 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2369 = stablehlo.reshape %v2368 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2370 = stablehlo.reshape %v2056 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2371 = stablehlo.slice %v2370 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2372 = stablehlo.reshape %v2371 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2373 = stablehlo.reshape %v2061 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2374 = stablehlo.slice %v2373 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2375 = stablehlo.reshape %v2374 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2376 = stablehlo.reshape %v2372 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2377 = stablehlo.transpose %v2376, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2378 = stablehlo.reshape %v2377 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2379 = stablehlo.reshape %v2369 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2380 = stablehlo.reshape %v2378 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2381 = stablehlo.dot_general %v2379, %v2380, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2382 = stablehlo.reshape %v2381 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2383 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2384 = stablehlo.multiply %v2382, %v2383 : tensor<32x38809xf32>
    %v2385 = stablehlo.reshape %v2384 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2386 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2387 = stablehlo.exponential %v2385 : tensor<32x197x197xf32>
    %v2388 = stablehlo.reduce(%v2387 init: %v2386) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2389 = stablehlo.broadcast_in_dim %v2388, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2390 = stablehlo.divide %v2387, %v2389 : tensor<32x197x197xf32>
    %v2391 = stablehlo.reshape %v2390 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2392 = stablehlo.reshape %v2391 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2393 = stablehlo.reshape %v2375 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2394 = stablehlo.dot_general %v2392, %v2393, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2395 = stablehlo.reshape %v2394 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2396 = stablehlo.reshape %v2395 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2397 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2398 = stablehlo.pad %v2396, %v2397, low = [0, 0, 576], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v2399 = stablehlo.reshape %v2398 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2400 = stablehlo.add %v2366, %v2399 : tensor<32x151296xf32>
    %v2401 = stablehlo.reshape %v2051 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2402 = stablehlo.slice %v2401 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2403 = stablehlo.reshape %v2402 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2404 = stablehlo.reshape %v2056 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2405 = stablehlo.slice %v2404 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2406 = stablehlo.reshape %v2405 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2407 = stablehlo.reshape %v2061 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2408 = stablehlo.slice %v2407 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2409 = stablehlo.reshape %v2408 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2410 = stablehlo.reshape %v2406 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2411 = stablehlo.transpose %v2410, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2412 = stablehlo.reshape %v2411 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2413 = stablehlo.reshape %v2403 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2414 = stablehlo.reshape %v2412 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2415 = stablehlo.dot_general %v2413, %v2414, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2416 = stablehlo.reshape %v2415 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2417 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2418 = stablehlo.multiply %v2416, %v2417 : tensor<32x38809xf32>
    %v2419 = stablehlo.reshape %v2418 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2420 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2421 = stablehlo.exponential %v2419 : tensor<32x197x197xf32>
    %v2422 = stablehlo.reduce(%v2421 init: %v2420) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2423 = stablehlo.broadcast_in_dim %v2422, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2424 = stablehlo.divide %v2421, %v2423 : tensor<32x197x197xf32>
    %v2425 = stablehlo.reshape %v2424 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2426 = stablehlo.reshape %v2425 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2427 = stablehlo.reshape %v2409 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2428 = stablehlo.dot_general %v2426, %v2427, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2429 = stablehlo.reshape %v2428 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2430 = stablehlo.reshape %v2429 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2431 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2432 = stablehlo.pad %v2430, %v2431, low = [0, 0, 640], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v2433 = stablehlo.reshape %v2432 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2434 = stablehlo.add %v2400, %v2433 : tensor<32x151296xf32>
    %v2435 = stablehlo.reshape %v2051 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2436 = stablehlo.slice %v2435 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2437 = stablehlo.reshape %v2436 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2438 = stablehlo.reshape %v2056 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2439 = stablehlo.slice %v2438 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2440 = stablehlo.reshape %v2439 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2441 = stablehlo.reshape %v2061 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2442 = stablehlo.slice %v2441 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2443 = stablehlo.reshape %v2442 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2444 = stablehlo.reshape %v2440 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2445 = stablehlo.transpose %v2444, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2446 = stablehlo.reshape %v2445 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2447 = stablehlo.reshape %v2437 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2448 = stablehlo.reshape %v2446 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2449 = stablehlo.dot_general %v2447, %v2448, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2450 = stablehlo.reshape %v2449 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2451 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2452 = stablehlo.multiply %v2450, %v2451 : tensor<32x38809xf32>
    %v2453 = stablehlo.reshape %v2452 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2454 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2455 = stablehlo.exponential %v2453 : tensor<32x197x197xf32>
    %v2456 = stablehlo.reduce(%v2455 init: %v2454) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2457 = stablehlo.broadcast_in_dim %v2456, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2458 = stablehlo.divide %v2455, %v2457 : tensor<32x197x197xf32>
    %v2459 = stablehlo.reshape %v2458 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2460 = stablehlo.reshape %v2459 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2461 = stablehlo.reshape %v2443 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2462 = stablehlo.dot_general %v2460, %v2461, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2463 = stablehlo.reshape %v2462 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2464 = stablehlo.reshape %v2463 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2465 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2466 = stablehlo.pad %v2464, %v2465, low = [0, 0, 704], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v2467 = stablehlo.reshape %v2466 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2468 = stablehlo.add %v2434, %v2467 : tensor<32x151296xf32>
    %v2469 = stablehlo.reshape %v2468 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2470 = stablehlo.dot_general %v2469, %b4_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v2471 = stablehlo.broadcast_in_dim %b4_bo, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v2472 = stablehlo.add %v2470, %v2471 : tensor<32x197x768xf32>
    %v2473 = stablehlo.reshape %v2472 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2474 = stablehlo.add %v2018, %v2473 : tensor<32x151296xf32>
    %v2475 = stablehlo.reshape %v2474 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2476 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2477 = stablehlo.constant dense<768.0> : tensor<32x197x768xf32>
    %v2478 = stablehlo.constant dense<1.0e-5> : tensor<32x197x768xf32>
    %v2479 = stablehlo.reduce(%v2475 init: %v2476) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2480 = stablehlo.broadcast_in_dim %v2479, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v2481 = stablehlo.divide %v2480, %v2477 : tensor<32x197x768xf32>
    %v2482 = stablehlo.subtract %v2475, %v2481 : tensor<32x197x768xf32>
    %v2483 = stablehlo.multiply %v2482, %v2482 : tensor<32x197x768xf32>
    %v2484 = stablehlo.reduce(%v2483 init: %v2476) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2485 = stablehlo.broadcast_in_dim %v2484, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v2486 = stablehlo.divide %v2485, %v2477 : tensor<32x197x768xf32>
    %v2487 = stablehlo.add %v2486, %v2478 : tensor<32x197x768xf32>
    %v2488 = stablehlo.rsqrt %v2487 : tensor<32x197x768xf32>
    %v2489 = stablehlo.multiply %v2482, %v2488 : tensor<32x197x768xf32>
    %v2490 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v2491 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v2492 = stablehlo.multiply %v2489, %v2490 : tensor<32x197x768xf32>
    %v2493 = stablehlo.add %v2492, %v2491 : tensor<32x197x768xf32>
    %v2494 = stablehlo.reshape %v2493 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2495 = stablehlo.reshape %v2494 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2496 = stablehlo.broadcast_in_dim %b4_g2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v2497 = stablehlo.multiply %v2495, %v2496 : tensor<32x197x768xf32>
    %v2498 = stablehlo.reshape %v2497 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2499 = stablehlo.reshape %v2498 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2500 = stablehlo.broadcast_in_dim %b4_bt2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v2501 = stablehlo.add %v2499, %v2500 : tensor<32x197x768xf32>
    %v2502 = stablehlo.reshape %v2501 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2503 = stablehlo.reshape %v2502 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2504 = stablehlo.dot_general %v2503, %b4_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x3072xf32>) -> tensor<32x197x3072xf32>
    %v2505 = stablehlo.broadcast_in_dim %b4_bfc1, dims = [2] : (tensor<3072xf32>) -> tensor<32x197x3072xf32>
    %v2506 = stablehlo.add %v2504, %v2505 : tensor<32x197x3072xf32>
    %v2507 = stablehlo.reshape %v2506 : (tensor<32x197x3072xf32>) -> tensor<32x605184xf32>
    %v2508 = stablehlo.constant dense<0.5> : tensor<32x605184xf32>
    %v2509 = stablehlo.multiply %v2508, %v2507 : tensor<32x605184xf32>
    %v2510 = stablehlo.negate %v2507 : tensor<32x605184xf32>
    %v2511 = stablehlo.constant dense<0.7071067811865476> : tensor<32x605184xf32>
    %v2512 = stablehlo.multiply %v2510, %v2511 : tensor<32x605184xf32>
    %v2513 = chlo.erfc %v2512 : tensor<32x605184xf32> -> tensor<32x605184xf32>
    %v2514 = stablehlo.multiply %v2509, %v2513 : tensor<32x605184xf32>
    %v2515 = stablehlo.reshape %v2514 : (tensor<32x605184xf32>) -> tensor<32x197x3072xf32>
    %v2516 = stablehlo.dot_general %v2515, %b4_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x3072xf32>, tensor<3072x768xf32>) -> tensor<32x197x768xf32>
    %v2517 = stablehlo.broadcast_in_dim %b4_bfc2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v2518 = stablehlo.add %v2516, %v2517 : tensor<32x197x768xf32>
    %v2519 = stablehlo.reshape %v2518 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2520 = stablehlo.add %v2474, %v2519 : tensor<32x151296xf32>
    %v2521 = stablehlo.reshape %v2520 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2522 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2523 = stablehlo.constant dense<768.0> : tensor<32x197x768xf32>
    %v2524 = stablehlo.constant dense<1.0e-5> : tensor<32x197x768xf32>
    %v2525 = stablehlo.reduce(%v2521 init: %v2522) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2526 = stablehlo.broadcast_in_dim %v2525, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v2527 = stablehlo.divide %v2526, %v2523 : tensor<32x197x768xf32>
    %v2528 = stablehlo.subtract %v2521, %v2527 : tensor<32x197x768xf32>
    %v2529 = stablehlo.multiply %v2528, %v2528 : tensor<32x197x768xf32>
    %v2530 = stablehlo.reduce(%v2529 init: %v2522) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2531 = stablehlo.broadcast_in_dim %v2530, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v2532 = stablehlo.divide %v2531, %v2523 : tensor<32x197x768xf32>
    %v2533 = stablehlo.add %v2532, %v2524 : tensor<32x197x768xf32>
    %v2534 = stablehlo.rsqrt %v2533 : tensor<32x197x768xf32>
    %v2535 = stablehlo.multiply %v2528, %v2534 : tensor<32x197x768xf32>
    %v2536 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v2537 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v2538 = stablehlo.multiply %v2535, %v2536 : tensor<32x197x768xf32>
    %v2539 = stablehlo.add %v2538, %v2537 : tensor<32x197x768xf32>
    %v2540 = stablehlo.reshape %v2539 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2541 = stablehlo.reshape %v2540 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2542 = stablehlo.broadcast_in_dim %b5_g1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v2543 = stablehlo.multiply %v2541, %v2542 : tensor<32x197x768xf32>
    %v2544 = stablehlo.reshape %v2543 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2545 = stablehlo.reshape %v2544 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2546 = stablehlo.broadcast_in_dim %b5_bt1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v2547 = stablehlo.add %v2545, %v2546 : tensor<32x197x768xf32>
    %v2548 = stablehlo.reshape %v2547 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2549 = stablehlo.reshape %v2548 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2550 = stablehlo.dot_general %v2549, %b5_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v2551 = stablehlo.broadcast_in_dim %b5_bq, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v2552 = stablehlo.add %v2550, %v2551 : tensor<32x197x768xf32>
    %v2553 = stablehlo.reshape %v2552 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2554 = stablehlo.reshape %v2548 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2555 = stablehlo.dot_general %v2554, %b5_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v2556 = stablehlo.broadcast_in_dim %b5_bk, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v2557 = stablehlo.add %v2555, %v2556 : tensor<32x197x768xf32>
    %v2558 = stablehlo.reshape %v2557 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2559 = stablehlo.reshape %v2548 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2560 = stablehlo.dot_general %v2559, %b5_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v2561 = stablehlo.broadcast_in_dim %b5_bv, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v2562 = stablehlo.add %v2560, %v2561 : tensor<32x197x768xf32>
    %v2563 = stablehlo.reshape %v2562 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2564 = stablehlo.reshape %v2553 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2565 = stablehlo.slice %v2564 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2566 = stablehlo.reshape %v2565 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2567 = stablehlo.reshape %v2558 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2568 = stablehlo.slice %v2567 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2569 = stablehlo.reshape %v2568 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2570 = stablehlo.reshape %v2563 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2571 = stablehlo.slice %v2570 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2572 = stablehlo.reshape %v2571 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2573 = stablehlo.reshape %v2569 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2574 = stablehlo.transpose %v2573, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2575 = stablehlo.reshape %v2574 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2576 = stablehlo.reshape %v2566 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2577 = stablehlo.reshape %v2575 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2578 = stablehlo.dot_general %v2576, %v2577, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2579 = stablehlo.reshape %v2578 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2580 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2581 = stablehlo.multiply %v2579, %v2580 : tensor<32x38809xf32>
    %v2582 = stablehlo.reshape %v2581 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2583 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2584 = stablehlo.exponential %v2582 : tensor<32x197x197xf32>
    %v2585 = stablehlo.reduce(%v2584 init: %v2583) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2586 = stablehlo.broadcast_in_dim %v2585, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2587 = stablehlo.divide %v2584, %v2586 : tensor<32x197x197xf32>
    %v2588 = stablehlo.reshape %v2587 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2589 = stablehlo.reshape %v2588 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2590 = stablehlo.reshape %v2572 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2591 = stablehlo.dot_general %v2589, %v2590, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2592 = stablehlo.reshape %v2591 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2593 = stablehlo.reshape %v2592 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2594 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2595 = stablehlo.pad %v2593, %v2594, low = [0, 0, 0], high = [0, 0, 704], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v2596 = stablehlo.reshape %v2595 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2597 = stablehlo.reshape %v2553 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2598 = stablehlo.slice %v2597 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2599 = stablehlo.reshape %v2598 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2600 = stablehlo.reshape %v2558 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2601 = stablehlo.slice %v2600 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2602 = stablehlo.reshape %v2601 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2603 = stablehlo.reshape %v2563 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2604 = stablehlo.slice %v2603 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2605 = stablehlo.reshape %v2604 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2606 = stablehlo.reshape %v2602 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2607 = stablehlo.transpose %v2606, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2608 = stablehlo.reshape %v2607 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2609 = stablehlo.reshape %v2599 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2610 = stablehlo.reshape %v2608 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2611 = stablehlo.dot_general %v2609, %v2610, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2612 = stablehlo.reshape %v2611 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2613 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2614 = stablehlo.multiply %v2612, %v2613 : tensor<32x38809xf32>
    %v2615 = stablehlo.reshape %v2614 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2616 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2617 = stablehlo.exponential %v2615 : tensor<32x197x197xf32>
    %v2618 = stablehlo.reduce(%v2617 init: %v2616) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2619 = stablehlo.broadcast_in_dim %v2618, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2620 = stablehlo.divide %v2617, %v2619 : tensor<32x197x197xf32>
    %v2621 = stablehlo.reshape %v2620 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2622 = stablehlo.reshape %v2621 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2623 = stablehlo.reshape %v2605 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2624 = stablehlo.dot_general %v2622, %v2623, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2625 = stablehlo.reshape %v2624 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2626 = stablehlo.reshape %v2625 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2627 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2628 = stablehlo.pad %v2626, %v2627, low = [0, 0, 64], high = [0, 0, 640], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v2629 = stablehlo.reshape %v2628 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2630 = stablehlo.add %v2596, %v2629 : tensor<32x151296xf32>
    %v2631 = stablehlo.reshape %v2553 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2632 = stablehlo.slice %v2631 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2633 = stablehlo.reshape %v2632 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2634 = stablehlo.reshape %v2558 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2635 = stablehlo.slice %v2634 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2636 = stablehlo.reshape %v2635 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2637 = stablehlo.reshape %v2563 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2638 = stablehlo.slice %v2637 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2639 = stablehlo.reshape %v2638 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2640 = stablehlo.reshape %v2636 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2641 = stablehlo.transpose %v2640, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2642 = stablehlo.reshape %v2641 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2643 = stablehlo.reshape %v2633 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2644 = stablehlo.reshape %v2642 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2645 = stablehlo.dot_general %v2643, %v2644, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2646 = stablehlo.reshape %v2645 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2647 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2648 = stablehlo.multiply %v2646, %v2647 : tensor<32x38809xf32>
    %v2649 = stablehlo.reshape %v2648 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2650 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2651 = stablehlo.exponential %v2649 : tensor<32x197x197xf32>
    %v2652 = stablehlo.reduce(%v2651 init: %v2650) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2653 = stablehlo.broadcast_in_dim %v2652, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2654 = stablehlo.divide %v2651, %v2653 : tensor<32x197x197xf32>
    %v2655 = stablehlo.reshape %v2654 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2656 = stablehlo.reshape %v2655 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2657 = stablehlo.reshape %v2639 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2658 = stablehlo.dot_general %v2656, %v2657, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2659 = stablehlo.reshape %v2658 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2660 = stablehlo.reshape %v2659 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2661 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2662 = stablehlo.pad %v2660, %v2661, low = [0, 0, 128], high = [0, 0, 576], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v2663 = stablehlo.reshape %v2662 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2664 = stablehlo.add %v2630, %v2663 : tensor<32x151296xf32>
    %v2665 = stablehlo.reshape %v2553 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2666 = stablehlo.slice %v2665 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2667 = stablehlo.reshape %v2666 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2668 = stablehlo.reshape %v2558 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2669 = stablehlo.slice %v2668 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2670 = stablehlo.reshape %v2669 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2671 = stablehlo.reshape %v2563 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2672 = stablehlo.slice %v2671 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2673 = stablehlo.reshape %v2672 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2674 = stablehlo.reshape %v2670 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2675 = stablehlo.transpose %v2674, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2676 = stablehlo.reshape %v2675 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2677 = stablehlo.reshape %v2667 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2678 = stablehlo.reshape %v2676 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2679 = stablehlo.dot_general %v2677, %v2678, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2680 = stablehlo.reshape %v2679 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2681 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2682 = stablehlo.multiply %v2680, %v2681 : tensor<32x38809xf32>
    %v2683 = stablehlo.reshape %v2682 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2684 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2685 = stablehlo.exponential %v2683 : tensor<32x197x197xf32>
    %v2686 = stablehlo.reduce(%v2685 init: %v2684) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2687 = stablehlo.broadcast_in_dim %v2686, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2688 = stablehlo.divide %v2685, %v2687 : tensor<32x197x197xf32>
    %v2689 = stablehlo.reshape %v2688 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2690 = stablehlo.reshape %v2689 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2691 = stablehlo.reshape %v2673 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2692 = stablehlo.dot_general %v2690, %v2691, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2693 = stablehlo.reshape %v2692 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2694 = stablehlo.reshape %v2693 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2695 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2696 = stablehlo.pad %v2694, %v2695, low = [0, 0, 192], high = [0, 0, 512], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v2697 = stablehlo.reshape %v2696 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2698 = stablehlo.add %v2664, %v2697 : tensor<32x151296xf32>
    %v2699 = stablehlo.reshape %v2553 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2700 = stablehlo.slice %v2699 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2701 = stablehlo.reshape %v2700 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2702 = stablehlo.reshape %v2558 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2703 = stablehlo.slice %v2702 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2704 = stablehlo.reshape %v2703 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2705 = stablehlo.reshape %v2563 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2706 = stablehlo.slice %v2705 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2707 = stablehlo.reshape %v2706 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2708 = stablehlo.reshape %v2704 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2709 = stablehlo.transpose %v2708, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2710 = stablehlo.reshape %v2709 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2711 = stablehlo.reshape %v2701 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2712 = stablehlo.reshape %v2710 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2713 = stablehlo.dot_general %v2711, %v2712, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2714 = stablehlo.reshape %v2713 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2715 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2716 = stablehlo.multiply %v2714, %v2715 : tensor<32x38809xf32>
    %v2717 = stablehlo.reshape %v2716 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2718 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2719 = stablehlo.exponential %v2717 : tensor<32x197x197xf32>
    %v2720 = stablehlo.reduce(%v2719 init: %v2718) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2721 = stablehlo.broadcast_in_dim %v2720, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2722 = stablehlo.divide %v2719, %v2721 : tensor<32x197x197xf32>
    %v2723 = stablehlo.reshape %v2722 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2724 = stablehlo.reshape %v2723 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2725 = stablehlo.reshape %v2707 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2726 = stablehlo.dot_general %v2724, %v2725, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2727 = stablehlo.reshape %v2726 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2728 = stablehlo.reshape %v2727 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2729 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2730 = stablehlo.pad %v2728, %v2729, low = [0, 0, 256], high = [0, 0, 448], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v2731 = stablehlo.reshape %v2730 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2732 = stablehlo.add %v2698, %v2731 : tensor<32x151296xf32>
    %v2733 = stablehlo.reshape %v2553 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2734 = stablehlo.slice %v2733 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2735 = stablehlo.reshape %v2734 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2736 = stablehlo.reshape %v2558 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2737 = stablehlo.slice %v2736 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2738 = stablehlo.reshape %v2737 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2739 = stablehlo.reshape %v2563 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2740 = stablehlo.slice %v2739 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2741 = stablehlo.reshape %v2740 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2742 = stablehlo.reshape %v2738 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2743 = stablehlo.transpose %v2742, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2744 = stablehlo.reshape %v2743 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2745 = stablehlo.reshape %v2735 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2746 = stablehlo.reshape %v2744 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2747 = stablehlo.dot_general %v2745, %v2746, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2748 = stablehlo.reshape %v2747 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2749 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2750 = stablehlo.multiply %v2748, %v2749 : tensor<32x38809xf32>
    %v2751 = stablehlo.reshape %v2750 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2752 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2753 = stablehlo.exponential %v2751 : tensor<32x197x197xf32>
    %v2754 = stablehlo.reduce(%v2753 init: %v2752) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2755 = stablehlo.broadcast_in_dim %v2754, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2756 = stablehlo.divide %v2753, %v2755 : tensor<32x197x197xf32>
    %v2757 = stablehlo.reshape %v2756 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2758 = stablehlo.reshape %v2757 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2759 = stablehlo.reshape %v2741 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2760 = stablehlo.dot_general %v2758, %v2759, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2761 = stablehlo.reshape %v2760 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2762 = stablehlo.reshape %v2761 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2763 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2764 = stablehlo.pad %v2762, %v2763, low = [0, 0, 320], high = [0, 0, 384], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v2765 = stablehlo.reshape %v2764 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2766 = stablehlo.add %v2732, %v2765 : tensor<32x151296xf32>
    %v2767 = stablehlo.reshape %v2553 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2768 = stablehlo.slice %v2767 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2769 = stablehlo.reshape %v2768 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2770 = stablehlo.reshape %v2558 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2771 = stablehlo.slice %v2770 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2772 = stablehlo.reshape %v2771 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2773 = stablehlo.reshape %v2563 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2774 = stablehlo.slice %v2773 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2775 = stablehlo.reshape %v2774 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2776 = stablehlo.reshape %v2772 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2777 = stablehlo.transpose %v2776, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2778 = stablehlo.reshape %v2777 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2779 = stablehlo.reshape %v2769 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2780 = stablehlo.reshape %v2778 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2781 = stablehlo.dot_general %v2779, %v2780, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2782 = stablehlo.reshape %v2781 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2783 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2784 = stablehlo.multiply %v2782, %v2783 : tensor<32x38809xf32>
    %v2785 = stablehlo.reshape %v2784 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2786 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2787 = stablehlo.exponential %v2785 : tensor<32x197x197xf32>
    %v2788 = stablehlo.reduce(%v2787 init: %v2786) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2789 = stablehlo.broadcast_in_dim %v2788, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2790 = stablehlo.divide %v2787, %v2789 : tensor<32x197x197xf32>
    %v2791 = stablehlo.reshape %v2790 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2792 = stablehlo.reshape %v2791 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2793 = stablehlo.reshape %v2775 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2794 = stablehlo.dot_general %v2792, %v2793, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2795 = stablehlo.reshape %v2794 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2796 = stablehlo.reshape %v2795 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2797 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2798 = stablehlo.pad %v2796, %v2797, low = [0, 0, 384], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v2799 = stablehlo.reshape %v2798 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2800 = stablehlo.add %v2766, %v2799 : tensor<32x151296xf32>
    %v2801 = stablehlo.reshape %v2553 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2802 = stablehlo.slice %v2801 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2803 = stablehlo.reshape %v2802 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2804 = stablehlo.reshape %v2558 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2805 = stablehlo.slice %v2804 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2806 = stablehlo.reshape %v2805 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2807 = stablehlo.reshape %v2563 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2808 = stablehlo.slice %v2807 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2809 = stablehlo.reshape %v2808 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2810 = stablehlo.reshape %v2806 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2811 = stablehlo.transpose %v2810, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2812 = stablehlo.reshape %v2811 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2813 = stablehlo.reshape %v2803 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2814 = stablehlo.reshape %v2812 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2815 = stablehlo.dot_general %v2813, %v2814, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2816 = stablehlo.reshape %v2815 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2817 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2818 = stablehlo.multiply %v2816, %v2817 : tensor<32x38809xf32>
    %v2819 = stablehlo.reshape %v2818 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2820 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2821 = stablehlo.exponential %v2819 : tensor<32x197x197xf32>
    %v2822 = stablehlo.reduce(%v2821 init: %v2820) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2823 = stablehlo.broadcast_in_dim %v2822, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2824 = stablehlo.divide %v2821, %v2823 : tensor<32x197x197xf32>
    %v2825 = stablehlo.reshape %v2824 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2826 = stablehlo.reshape %v2825 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2827 = stablehlo.reshape %v2809 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2828 = stablehlo.dot_general %v2826, %v2827, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2829 = stablehlo.reshape %v2828 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2830 = stablehlo.reshape %v2829 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2831 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2832 = stablehlo.pad %v2830, %v2831, low = [0, 0, 448], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v2833 = stablehlo.reshape %v2832 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2834 = stablehlo.add %v2800, %v2833 : tensor<32x151296xf32>
    %v2835 = stablehlo.reshape %v2553 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2836 = stablehlo.slice %v2835 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2837 = stablehlo.reshape %v2836 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2838 = stablehlo.reshape %v2558 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2839 = stablehlo.slice %v2838 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2840 = stablehlo.reshape %v2839 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2841 = stablehlo.reshape %v2563 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2842 = stablehlo.slice %v2841 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2843 = stablehlo.reshape %v2842 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2844 = stablehlo.reshape %v2840 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2845 = stablehlo.transpose %v2844, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2846 = stablehlo.reshape %v2845 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2847 = stablehlo.reshape %v2837 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2848 = stablehlo.reshape %v2846 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2849 = stablehlo.dot_general %v2847, %v2848, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2850 = stablehlo.reshape %v2849 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2851 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2852 = stablehlo.multiply %v2850, %v2851 : tensor<32x38809xf32>
    %v2853 = stablehlo.reshape %v2852 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2854 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2855 = stablehlo.exponential %v2853 : tensor<32x197x197xf32>
    %v2856 = stablehlo.reduce(%v2855 init: %v2854) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2857 = stablehlo.broadcast_in_dim %v2856, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2858 = stablehlo.divide %v2855, %v2857 : tensor<32x197x197xf32>
    %v2859 = stablehlo.reshape %v2858 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2860 = stablehlo.reshape %v2859 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2861 = stablehlo.reshape %v2843 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2862 = stablehlo.dot_general %v2860, %v2861, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2863 = stablehlo.reshape %v2862 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2864 = stablehlo.reshape %v2863 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2865 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2866 = stablehlo.pad %v2864, %v2865, low = [0, 0, 512], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v2867 = stablehlo.reshape %v2866 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2868 = stablehlo.add %v2834, %v2867 : tensor<32x151296xf32>
    %v2869 = stablehlo.reshape %v2553 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2870 = stablehlo.slice %v2869 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2871 = stablehlo.reshape %v2870 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2872 = stablehlo.reshape %v2558 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2873 = stablehlo.slice %v2872 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2874 = stablehlo.reshape %v2873 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2875 = stablehlo.reshape %v2563 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2876 = stablehlo.slice %v2875 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2877 = stablehlo.reshape %v2876 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2878 = stablehlo.reshape %v2874 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2879 = stablehlo.transpose %v2878, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2880 = stablehlo.reshape %v2879 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2881 = stablehlo.reshape %v2871 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2882 = stablehlo.reshape %v2880 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2883 = stablehlo.dot_general %v2881, %v2882, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2884 = stablehlo.reshape %v2883 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2885 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2886 = stablehlo.multiply %v2884, %v2885 : tensor<32x38809xf32>
    %v2887 = stablehlo.reshape %v2886 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2888 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2889 = stablehlo.exponential %v2887 : tensor<32x197x197xf32>
    %v2890 = stablehlo.reduce(%v2889 init: %v2888) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2891 = stablehlo.broadcast_in_dim %v2890, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2892 = stablehlo.divide %v2889, %v2891 : tensor<32x197x197xf32>
    %v2893 = stablehlo.reshape %v2892 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2894 = stablehlo.reshape %v2893 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2895 = stablehlo.reshape %v2877 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2896 = stablehlo.dot_general %v2894, %v2895, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2897 = stablehlo.reshape %v2896 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2898 = stablehlo.reshape %v2897 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2899 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2900 = stablehlo.pad %v2898, %v2899, low = [0, 0, 576], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v2901 = stablehlo.reshape %v2900 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2902 = stablehlo.add %v2868, %v2901 : tensor<32x151296xf32>
    %v2903 = stablehlo.reshape %v2553 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2904 = stablehlo.slice %v2903 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2905 = stablehlo.reshape %v2904 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2906 = stablehlo.reshape %v2558 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2907 = stablehlo.slice %v2906 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2908 = stablehlo.reshape %v2907 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2909 = stablehlo.reshape %v2563 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2910 = stablehlo.slice %v2909 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2911 = stablehlo.reshape %v2910 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2912 = stablehlo.reshape %v2908 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2913 = stablehlo.transpose %v2912, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2914 = stablehlo.reshape %v2913 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2915 = stablehlo.reshape %v2905 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2916 = stablehlo.reshape %v2914 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2917 = stablehlo.dot_general %v2915, %v2916, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2918 = stablehlo.reshape %v2917 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2919 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2920 = stablehlo.multiply %v2918, %v2919 : tensor<32x38809xf32>
    %v2921 = stablehlo.reshape %v2920 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2922 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2923 = stablehlo.exponential %v2921 : tensor<32x197x197xf32>
    %v2924 = stablehlo.reduce(%v2923 init: %v2922) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2925 = stablehlo.broadcast_in_dim %v2924, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2926 = stablehlo.divide %v2923, %v2925 : tensor<32x197x197xf32>
    %v2927 = stablehlo.reshape %v2926 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2928 = stablehlo.reshape %v2927 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2929 = stablehlo.reshape %v2911 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2930 = stablehlo.dot_general %v2928, %v2929, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2931 = stablehlo.reshape %v2930 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2932 = stablehlo.reshape %v2931 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2933 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2934 = stablehlo.pad %v2932, %v2933, low = [0, 0, 640], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v2935 = stablehlo.reshape %v2934 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2936 = stablehlo.add %v2902, %v2935 : tensor<32x151296xf32>
    %v2937 = stablehlo.reshape %v2553 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2938 = stablehlo.slice %v2937 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2939 = stablehlo.reshape %v2938 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2940 = stablehlo.reshape %v2558 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2941 = stablehlo.slice %v2940 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2942 = stablehlo.reshape %v2941 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2943 = stablehlo.reshape %v2563 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2944 = stablehlo.slice %v2943 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2945 = stablehlo.reshape %v2944 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2946 = stablehlo.reshape %v2942 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2947 = stablehlo.transpose %v2946, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2948 = stablehlo.reshape %v2947 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2949 = stablehlo.reshape %v2939 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2950 = stablehlo.reshape %v2948 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2951 = stablehlo.dot_general %v2949, %v2950, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2952 = stablehlo.reshape %v2951 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2953 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2954 = stablehlo.multiply %v2952, %v2953 : tensor<32x38809xf32>
    %v2955 = stablehlo.reshape %v2954 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2956 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2957 = stablehlo.exponential %v2955 : tensor<32x197x197xf32>
    %v2958 = stablehlo.reduce(%v2957 init: %v2956) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2959 = stablehlo.broadcast_in_dim %v2958, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2960 = stablehlo.divide %v2957, %v2959 : tensor<32x197x197xf32>
    %v2961 = stablehlo.reshape %v2960 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2962 = stablehlo.reshape %v2961 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2963 = stablehlo.reshape %v2945 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2964 = stablehlo.dot_general %v2962, %v2963, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2965 = stablehlo.reshape %v2964 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2966 = stablehlo.reshape %v2965 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2967 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2968 = stablehlo.pad %v2966, %v2967, low = [0, 0, 704], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v2969 = stablehlo.reshape %v2968 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2970 = stablehlo.add %v2936, %v2969 : tensor<32x151296xf32>
    %v2971 = stablehlo.reshape %v2970 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2972 = stablehlo.dot_general %v2971, %b5_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v2973 = stablehlo.broadcast_in_dim %b5_bo, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v2974 = stablehlo.add %v2972, %v2973 : tensor<32x197x768xf32>
    %v2975 = stablehlo.reshape %v2974 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2976 = stablehlo.add %v2520, %v2975 : tensor<32x151296xf32>
    %v2977 = stablehlo.reshape %v2976 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2978 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2979 = stablehlo.constant dense<768.0> : tensor<32x197x768xf32>
    %v2980 = stablehlo.constant dense<1.0e-5> : tensor<32x197x768xf32>
    %v2981 = stablehlo.reduce(%v2977 init: %v2978) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2982 = stablehlo.broadcast_in_dim %v2981, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v2983 = stablehlo.divide %v2982, %v2979 : tensor<32x197x768xf32>
    %v2984 = stablehlo.subtract %v2977, %v2983 : tensor<32x197x768xf32>
    %v2985 = stablehlo.multiply %v2984, %v2984 : tensor<32x197x768xf32>
    %v2986 = stablehlo.reduce(%v2985 init: %v2978) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2987 = stablehlo.broadcast_in_dim %v2986, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v2988 = stablehlo.divide %v2987, %v2979 : tensor<32x197x768xf32>
    %v2989 = stablehlo.add %v2988, %v2980 : tensor<32x197x768xf32>
    %v2990 = stablehlo.rsqrt %v2989 : tensor<32x197x768xf32>
    %v2991 = stablehlo.multiply %v2984, %v2990 : tensor<32x197x768xf32>
    %v2992 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v2993 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v2994 = stablehlo.multiply %v2991, %v2992 : tensor<32x197x768xf32>
    %v2995 = stablehlo.add %v2994, %v2993 : tensor<32x197x768xf32>
    %v2996 = stablehlo.reshape %v2995 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2997 = stablehlo.reshape %v2996 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2998 = stablehlo.broadcast_in_dim %b5_g2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v2999 = stablehlo.multiply %v2997, %v2998 : tensor<32x197x768xf32>
    %v3000 = stablehlo.reshape %v2999 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3001 = stablehlo.reshape %v3000 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3002 = stablehlo.broadcast_in_dim %b5_bt2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v3003 = stablehlo.add %v3001, %v3002 : tensor<32x197x768xf32>
    %v3004 = stablehlo.reshape %v3003 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3005 = stablehlo.reshape %v3004 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3006 = stablehlo.dot_general %v3005, %b5_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x3072xf32>) -> tensor<32x197x3072xf32>
    %v3007 = stablehlo.broadcast_in_dim %b5_bfc1, dims = [2] : (tensor<3072xf32>) -> tensor<32x197x3072xf32>
    %v3008 = stablehlo.add %v3006, %v3007 : tensor<32x197x3072xf32>
    %v3009 = stablehlo.reshape %v3008 : (tensor<32x197x3072xf32>) -> tensor<32x605184xf32>
    %v3010 = stablehlo.constant dense<0.5> : tensor<32x605184xf32>
    %v3011 = stablehlo.multiply %v3010, %v3009 : tensor<32x605184xf32>
    %v3012 = stablehlo.negate %v3009 : tensor<32x605184xf32>
    %v3013 = stablehlo.constant dense<0.7071067811865476> : tensor<32x605184xf32>
    %v3014 = stablehlo.multiply %v3012, %v3013 : tensor<32x605184xf32>
    %v3015 = chlo.erfc %v3014 : tensor<32x605184xf32> -> tensor<32x605184xf32>
    %v3016 = stablehlo.multiply %v3011, %v3015 : tensor<32x605184xf32>
    %v3017 = stablehlo.reshape %v3016 : (tensor<32x605184xf32>) -> tensor<32x197x3072xf32>
    %v3018 = stablehlo.dot_general %v3017, %b5_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x3072xf32>, tensor<3072x768xf32>) -> tensor<32x197x768xf32>
    %v3019 = stablehlo.broadcast_in_dim %b5_bfc2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v3020 = stablehlo.add %v3018, %v3019 : tensor<32x197x768xf32>
    %v3021 = stablehlo.reshape %v3020 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3022 = stablehlo.add %v2976, %v3021 : tensor<32x151296xf32>
    %v3023 = stablehlo.reshape %v3022 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3024 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3025 = stablehlo.constant dense<768.0> : tensor<32x197x768xf32>
    %v3026 = stablehlo.constant dense<1.0e-5> : tensor<32x197x768xf32>
    %v3027 = stablehlo.reduce(%v3023 init: %v3024) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3028 = stablehlo.broadcast_in_dim %v3027, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v3029 = stablehlo.divide %v3028, %v3025 : tensor<32x197x768xf32>
    %v3030 = stablehlo.subtract %v3023, %v3029 : tensor<32x197x768xf32>
    %v3031 = stablehlo.multiply %v3030, %v3030 : tensor<32x197x768xf32>
    %v3032 = stablehlo.reduce(%v3031 init: %v3024) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3033 = stablehlo.broadcast_in_dim %v3032, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v3034 = stablehlo.divide %v3033, %v3025 : tensor<32x197x768xf32>
    %v3035 = stablehlo.add %v3034, %v3026 : tensor<32x197x768xf32>
    %v3036 = stablehlo.rsqrt %v3035 : tensor<32x197x768xf32>
    %v3037 = stablehlo.multiply %v3030, %v3036 : tensor<32x197x768xf32>
    %v3038 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v3039 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v3040 = stablehlo.multiply %v3037, %v3038 : tensor<32x197x768xf32>
    %v3041 = stablehlo.add %v3040, %v3039 : tensor<32x197x768xf32>
    %v3042 = stablehlo.reshape %v3041 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3043 = stablehlo.reshape %v3042 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3044 = stablehlo.broadcast_in_dim %b6_g1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v3045 = stablehlo.multiply %v3043, %v3044 : tensor<32x197x768xf32>
    %v3046 = stablehlo.reshape %v3045 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3047 = stablehlo.reshape %v3046 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3048 = stablehlo.broadcast_in_dim %b6_bt1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v3049 = stablehlo.add %v3047, %v3048 : tensor<32x197x768xf32>
    %v3050 = stablehlo.reshape %v3049 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3051 = stablehlo.reshape %v3050 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3052 = stablehlo.dot_general %v3051, %b6_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v3053 = stablehlo.broadcast_in_dim %b6_bq, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v3054 = stablehlo.add %v3052, %v3053 : tensor<32x197x768xf32>
    %v3055 = stablehlo.reshape %v3054 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3056 = stablehlo.reshape %v3050 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3057 = stablehlo.dot_general %v3056, %b6_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v3058 = stablehlo.broadcast_in_dim %b6_bk, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v3059 = stablehlo.add %v3057, %v3058 : tensor<32x197x768xf32>
    %v3060 = stablehlo.reshape %v3059 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3061 = stablehlo.reshape %v3050 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3062 = stablehlo.dot_general %v3061, %b6_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v3063 = stablehlo.broadcast_in_dim %b6_bv, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v3064 = stablehlo.add %v3062, %v3063 : tensor<32x197x768xf32>
    %v3065 = stablehlo.reshape %v3064 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3066 = stablehlo.reshape %v3055 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3067 = stablehlo.slice %v3066 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3068 = stablehlo.reshape %v3067 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3069 = stablehlo.reshape %v3060 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3070 = stablehlo.slice %v3069 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3071 = stablehlo.reshape %v3070 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3072 = stablehlo.reshape %v3065 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3073 = stablehlo.slice %v3072 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3074 = stablehlo.reshape %v3073 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3075 = stablehlo.reshape %v3071 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3076 = stablehlo.transpose %v3075, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3077 = stablehlo.reshape %v3076 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3078 = stablehlo.reshape %v3068 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3079 = stablehlo.reshape %v3077 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3080 = stablehlo.dot_general %v3078, %v3079, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3081 = stablehlo.reshape %v3080 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3082 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3083 = stablehlo.multiply %v3081, %v3082 : tensor<32x38809xf32>
    %v3084 = stablehlo.reshape %v3083 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3085 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3086 = stablehlo.exponential %v3084 : tensor<32x197x197xf32>
    %v3087 = stablehlo.reduce(%v3086 init: %v3085) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3088 = stablehlo.broadcast_in_dim %v3087, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3089 = stablehlo.divide %v3086, %v3088 : tensor<32x197x197xf32>
    %v3090 = stablehlo.reshape %v3089 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3091 = stablehlo.reshape %v3090 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3092 = stablehlo.reshape %v3074 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3093 = stablehlo.dot_general %v3091, %v3092, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3094 = stablehlo.reshape %v3093 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3095 = stablehlo.reshape %v3094 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3096 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3097 = stablehlo.pad %v3095, %v3096, low = [0, 0, 0], high = [0, 0, 704], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v3098 = stablehlo.reshape %v3097 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3099 = stablehlo.reshape %v3055 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3100 = stablehlo.slice %v3099 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3101 = stablehlo.reshape %v3100 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3102 = stablehlo.reshape %v3060 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3103 = stablehlo.slice %v3102 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3104 = stablehlo.reshape %v3103 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3105 = stablehlo.reshape %v3065 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3106 = stablehlo.slice %v3105 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3107 = stablehlo.reshape %v3106 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3108 = stablehlo.reshape %v3104 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3109 = stablehlo.transpose %v3108, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3110 = stablehlo.reshape %v3109 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3111 = stablehlo.reshape %v3101 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3112 = stablehlo.reshape %v3110 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3113 = stablehlo.dot_general %v3111, %v3112, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3114 = stablehlo.reshape %v3113 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3115 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3116 = stablehlo.multiply %v3114, %v3115 : tensor<32x38809xf32>
    %v3117 = stablehlo.reshape %v3116 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3118 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3119 = stablehlo.exponential %v3117 : tensor<32x197x197xf32>
    %v3120 = stablehlo.reduce(%v3119 init: %v3118) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3121 = stablehlo.broadcast_in_dim %v3120, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3122 = stablehlo.divide %v3119, %v3121 : tensor<32x197x197xf32>
    %v3123 = stablehlo.reshape %v3122 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3124 = stablehlo.reshape %v3123 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3125 = stablehlo.reshape %v3107 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3126 = stablehlo.dot_general %v3124, %v3125, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3127 = stablehlo.reshape %v3126 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3128 = stablehlo.reshape %v3127 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3129 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3130 = stablehlo.pad %v3128, %v3129, low = [0, 0, 64], high = [0, 0, 640], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v3131 = stablehlo.reshape %v3130 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3132 = stablehlo.add %v3098, %v3131 : tensor<32x151296xf32>
    %v3133 = stablehlo.reshape %v3055 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3134 = stablehlo.slice %v3133 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3135 = stablehlo.reshape %v3134 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3136 = stablehlo.reshape %v3060 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3137 = stablehlo.slice %v3136 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3138 = stablehlo.reshape %v3137 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3139 = stablehlo.reshape %v3065 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3140 = stablehlo.slice %v3139 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3141 = stablehlo.reshape %v3140 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3142 = stablehlo.reshape %v3138 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3143 = stablehlo.transpose %v3142, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3144 = stablehlo.reshape %v3143 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3145 = stablehlo.reshape %v3135 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3146 = stablehlo.reshape %v3144 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3147 = stablehlo.dot_general %v3145, %v3146, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3148 = stablehlo.reshape %v3147 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3149 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3150 = stablehlo.multiply %v3148, %v3149 : tensor<32x38809xf32>
    %v3151 = stablehlo.reshape %v3150 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3152 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3153 = stablehlo.exponential %v3151 : tensor<32x197x197xf32>
    %v3154 = stablehlo.reduce(%v3153 init: %v3152) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3155 = stablehlo.broadcast_in_dim %v3154, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3156 = stablehlo.divide %v3153, %v3155 : tensor<32x197x197xf32>
    %v3157 = stablehlo.reshape %v3156 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3158 = stablehlo.reshape %v3157 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3159 = stablehlo.reshape %v3141 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3160 = stablehlo.dot_general %v3158, %v3159, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3161 = stablehlo.reshape %v3160 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3162 = stablehlo.reshape %v3161 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3163 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3164 = stablehlo.pad %v3162, %v3163, low = [0, 0, 128], high = [0, 0, 576], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v3165 = stablehlo.reshape %v3164 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3166 = stablehlo.add %v3132, %v3165 : tensor<32x151296xf32>
    %v3167 = stablehlo.reshape %v3055 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3168 = stablehlo.slice %v3167 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3169 = stablehlo.reshape %v3168 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3170 = stablehlo.reshape %v3060 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3171 = stablehlo.slice %v3170 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3172 = stablehlo.reshape %v3171 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3173 = stablehlo.reshape %v3065 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3174 = stablehlo.slice %v3173 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3175 = stablehlo.reshape %v3174 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3176 = stablehlo.reshape %v3172 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3177 = stablehlo.transpose %v3176, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3178 = stablehlo.reshape %v3177 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3179 = stablehlo.reshape %v3169 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3180 = stablehlo.reshape %v3178 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3181 = stablehlo.dot_general %v3179, %v3180, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3182 = stablehlo.reshape %v3181 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3183 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3184 = stablehlo.multiply %v3182, %v3183 : tensor<32x38809xf32>
    %v3185 = stablehlo.reshape %v3184 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3186 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3187 = stablehlo.exponential %v3185 : tensor<32x197x197xf32>
    %v3188 = stablehlo.reduce(%v3187 init: %v3186) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3189 = stablehlo.broadcast_in_dim %v3188, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3190 = stablehlo.divide %v3187, %v3189 : tensor<32x197x197xf32>
    %v3191 = stablehlo.reshape %v3190 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3192 = stablehlo.reshape %v3191 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3193 = stablehlo.reshape %v3175 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3194 = stablehlo.dot_general %v3192, %v3193, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3195 = stablehlo.reshape %v3194 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3196 = stablehlo.reshape %v3195 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3197 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3198 = stablehlo.pad %v3196, %v3197, low = [0, 0, 192], high = [0, 0, 512], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v3199 = stablehlo.reshape %v3198 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3200 = stablehlo.add %v3166, %v3199 : tensor<32x151296xf32>
    %v3201 = stablehlo.reshape %v3055 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3202 = stablehlo.slice %v3201 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3203 = stablehlo.reshape %v3202 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3204 = stablehlo.reshape %v3060 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3205 = stablehlo.slice %v3204 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3206 = stablehlo.reshape %v3205 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3207 = stablehlo.reshape %v3065 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3208 = stablehlo.slice %v3207 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3209 = stablehlo.reshape %v3208 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3210 = stablehlo.reshape %v3206 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3211 = stablehlo.transpose %v3210, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3212 = stablehlo.reshape %v3211 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3213 = stablehlo.reshape %v3203 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3214 = stablehlo.reshape %v3212 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3215 = stablehlo.dot_general %v3213, %v3214, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3216 = stablehlo.reshape %v3215 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3217 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3218 = stablehlo.multiply %v3216, %v3217 : tensor<32x38809xf32>
    %v3219 = stablehlo.reshape %v3218 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3220 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3221 = stablehlo.exponential %v3219 : tensor<32x197x197xf32>
    %v3222 = stablehlo.reduce(%v3221 init: %v3220) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3223 = stablehlo.broadcast_in_dim %v3222, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3224 = stablehlo.divide %v3221, %v3223 : tensor<32x197x197xf32>
    %v3225 = stablehlo.reshape %v3224 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3226 = stablehlo.reshape %v3225 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3227 = stablehlo.reshape %v3209 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3228 = stablehlo.dot_general %v3226, %v3227, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3229 = stablehlo.reshape %v3228 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3230 = stablehlo.reshape %v3229 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3231 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3232 = stablehlo.pad %v3230, %v3231, low = [0, 0, 256], high = [0, 0, 448], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v3233 = stablehlo.reshape %v3232 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3234 = stablehlo.add %v3200, %v3233 : tensor<32x151296xf32>
    %v3235 = stablehlo.reshape %v3055 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3236 = stablehlo.slice %v3235 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3237 = stablehlo.reshape %v3236 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3238 = stablehlo.reshape %v3060 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3239 = stablehlo.slice %v3238 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3240 = stablehlo.reshape %v3239 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3241 = stablehlo.reshape %v3065 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3242 = stablehlo.slice %v3241 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3243 = stablehlo.reshape %v3242 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3244 = stablehlo.reshape %v3240 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3245 = stablehlo.transpose %v3244, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3246 = stablehlo.reshape %v3245 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3247 = stablehlo.reshape %v3237 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3248 = stablehlo.reshape %v3246 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3249 = stablehlo.dot_general %v3247, %v3248, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3250 = stablehlo.reshape %v3249 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3251 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3252 = stablehlo.multiply %v3250, %v3251 : tensor<32x38809xf32>
    %v3253 = stablehlo.reshape %v3252 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3254 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3255 = stablehlo.exponential %v3253 : tensor<32x197x197xf32>
    %v3256 = stablehlo.reduce(%v3255 init: %v3254) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3257 = stablehlo.broadcast_in_dim %v3256, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3258 = stablehlo.divide %v3255, %v3257 : tensor<32x197x197xf32>
    %v3259 = stablehlo.reshape %v3258 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3260 = stablehlo.reshape %v3259 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3261 = stablehlo.reshape %v3243 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3262 = stablehlo.dot_general %v3260, %v3261, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3263 = stablehlo.reshape %v3262 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3264 = stablehlo.reshape %v3263 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3265 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3266 = stablehlo.pad %v3264, %v3265, low = [0, 0, 320], high = [0, 0, 384], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v3267 = stablehlo.reshape %v3266 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3268 = stablehlo.add %v3234, %v3267 : tensor<32x151296xf32>
    %v3269 = stablehlo.reshape %v3055 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3270 = stablehlo.slice %v3269 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3271 = stablehlo.reshape %v3270 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3272 = stablehlo.reshape %v3060 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3273 = stablehlo.slice %v3272 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3274 = stablehlo.reshape %v3273 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3275 = stablehlo.reshape %v3065 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3276 = stablehlo.slice %v3275 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3277 = stablehlo.reshape %v3276 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3278 = stablehlo.reshape %v3274 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3279 = stablehlo.transpose %v3278, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3280 = stablehlo.reshape %v3279 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3281 = stablehlo.reshape %v3271 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3282 = stablehlo.reshape %v3280 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3283 = stablehlo.dot_general %v3281, %v3282, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3284 = stablehlo.reshape %v3283 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3285 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3286 = stablehlo.multiply %v3284, %v3285 : tensor<32x38809xf32>
    %v3287 = stablehlo.reshape %v3286 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3288 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3289 = stablehlo.exponential %v3287 : tensor<32x197x197xf32>
    %v3290 = stablehlo.reduce(%v3289 init: %v3288) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3291 = stablehlo.broadcast_in_dim %v3290, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3292 = stablehlo.divide %v3289, %v3291 : tensor<32x197x197xf32>
    %v3293 = stablehlo.reshape %v3292 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3294 = stablehlo.reshape %v3293 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3295 = stablehlo.reshape %v3277 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3296 = stablehlo.dot_general %v3294, %v3295, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3297 = stablehlo.reshape %v3296 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3298 = stablehlo.reshape %v3297 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3299 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3300 = stablehlo.pad %v3298, %v3299, low = [0, 0, 384], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v3301 = stablehlo.reshape %v3300 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3302 = stablehlo.add %v3268, %v3301 : tensor<32x151296xf32>
    %v3303 = stablehlo.reshape %v3055 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3304 = stablehlo.slice %v3303 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3305 = stablehlo.reshape %v3304 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3306 = stablehlo.reshape %v3060 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3307 = stablehlo.slice %v3306 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3308 = stablehlo.reshape %v3307 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3309 = stablehlo.reshape %v3065 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3310 = stablehlo.slice %v3309 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3311 = stablehlo.reshape %v3310 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3312 = stablehlo.reshape %v3308 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3313 = stablehlo.transpose %v3312, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3314 = stablehlo.reshape %v3313 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3315 = stablehlo.reshape %v3305 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3316 = stablehlo.reshape %v3314 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3317 = stablehlo.dot_general %v3315, %v3316, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3318 = stablehlo.reshape %v3317 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3319 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3320 = stablehlo.multiply %v3318, %v3319 : tensor<32x38809xf32>
    %v3321 = stablehlo.reshape %v3320 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3322 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3323 = stablehlo.exponential %v3321 : tensor<32x197x197xf32>
    %v3324 = stablehlo.reduce(%v3323 init: %v3322) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3325 = stablehlo.broadcast_in_dim %v3324, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3326 = stablehlo.divide %v3323, %v3325 : tensor<32x197x197xf32>
    %v3327 = stablehlo.reshape %v3326 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3328 = stablehlo.reshape %v3327 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3329 = stablehlo.reshape %v3311 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3330 = stablehlo.dot_general %v3328, %v3329, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3331 = stablehlo.reshape %v3330 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3332 = stablehlo.reshape %v3331 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3333 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3334 = stablehlo.pad %v3332, %v3333, low = [0, 0, 448], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v3335 = stablehlo.reshape %v3334 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3336 = stablehlo.add %v3302, %v3335 : tensor<32x151296xf32>
    %v3337 = stablehlo.reshape %v3055 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3338 = stablehlo.slice %v3337 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3339 = stablehlo.reshape %v3338 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3340 = stablehlo.reshape %v3060 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3341 = stablehlo.slice %v3340 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3342 = stablehlo.reshape %v3341 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3343 = stablehlo.reshape %v3065 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3344 = stablehlo.slice %v3343 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3345 = stablehlo.reshape %v3344 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3346 = stablehlo.reshape %v3342 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3347 = stablehlo.transpose %v3346, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3348 = stablehlo.reshape %v3347 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3349 = stablehlo.reshape %v3339 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3350 = stablehlo.reshape %v3348 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3351 = stablehlo.dot_general %v3349, %v3350, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3352 = stablehlo.reshape %v3351 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3353 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3354 = stablehlo.multiply %v3352, %v3353 : tensor<32x38809xf32>
    %v3355 = stablehlo.reshape %v3354 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3356 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3357 = stablehlo.exponential %v3355 : tensor<32x197x197xf32>
    %v3358 = stablehlo.reduce(%v3357 init: %v3356) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3359 = stablehlo.broadcast_in_dim %v3358, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3360 = stablehlo.divide %v3357, %v3359 : tensor<32x197x197xf32>
    %v3361 = stablehlo.reshape %v3360 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3362 = stablehlo.reshape %v3361 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3363 = stablehlo.reshape %v3345 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3364 = stablehlo.dot_general %v3362, %v3363, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3365 = stablehlo.reshape %v3364 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3366 = stablehlo.reshape %v3365 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3367 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3368 = stablehlo.pad %v3366, %v3367, low = [0, 0, 512], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v3369 = stablehlo.reshape %v3368 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3370 = stablehlo.add %v3336, %v3369 : tensor<32x151296xf32>
    %v3371 = stablehlo.reshape %v3055 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3372 = stablehlo.slice %v3371 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3373 = stablehlo.reshape %v3372 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3374 = stablehlo.reshape %v3060 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3375 = stablehlo.slice %v3374 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3376 = stablehlo.reshape %v3375 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3377 = stablehlo.reshape %v3065 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3378 = stablehlo.slice %v3377 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3379 = stablehlo.reshape %v3378 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3380 = stablehlo.reshape %v3376 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3381 = stablehlo.transpose %v3380, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3382 = stablehlo.reshape %v3381 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3383 = stablehlo.reshape %v3373 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3384 = stablehlo.reshape %v3382 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3385 = stablehlo.dot_general %v3383, %v3384, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3386 = stablehlo.reshape %v3385 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3387 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3388 = stablehlo.multiply %v3386, %v3387 : tensor<32x38809xf32>
    %v3389 = stablehlo.reshape %v3388 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3390 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3391 = stablehlo.exponential %v3389 : tensor<32x197x197xf32>
    %v3392 = stablehlo.reduce(%v3391 init: %v3390) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3393 = stablehlo.broadcast_in_dim %v3392, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3394 = stablehlo.divide %v3391, %v3393 : tensor<32x197x197xf32>
    %v3395 = stablehlo.reshape %v3394 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3396 = stablehlo.reshape %v3395 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3397 = stablehlo.reshape %v3379 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3398 = stablehlo.dot_general %v3396, %v3397, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3399 = stablehlo.reshape %v3398 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3400 = stablehlo.reshape %v3399 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3401 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3402 = stablehlo.pad %v3400, %v3401, low = [0, 0, 576], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v3403 = stablehlo.reshape %v3402 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3404 = stablehlo.add %v3370, %v3403 : tensor<32x151296xf32>
    %v3405 = stablehlo.reshape %v3055 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3406 = stablehlo.slice %v3405 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3407 = stablehlo.reshape %v3406 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3408 = stablehlo.reshape %v3060 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3409 = stablehlo.slice %v3408 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3410 = stablehlo.reshape %v3409 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3411 = stablehlo.reshape %v3065 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3412 = stablehlo.slice %v3411 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3413 = stablehlo.reshape %v3412 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3414 = stablehlo.reshape %v3410 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3415 = stablehlo.transpose %v3414, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3416 = stablehlo.reshape %v3415 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3417 = stablehlo.reshape %v3407 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3418 = stablehlo.reshape %v3416 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3419 = stablehlo.dot_general %v3417, %v3418, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3420 = stablehlo.reshape %v3419 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3421 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3422 = stablehlo.multiply %v3420, %v3421 : tensor<32x38809xf32>
    %v3423 = stablehlo.reshape %v3422 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3424 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3425 = stablehlo.exponential %v3423 : tensor<32x197x197xf32>
    %v3426 = stablehlo.reduce(%v3425 init: %v3424) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3427 = stablehlo.broadcast_in_dim %v3426, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3428 = stablehlo.divide %v3425, %v3427 : tensor<32x197x197xf32>
    %v3429 = stablehlo.reshape %v3428 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3430 = stablehlo.reshape %v3429 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3431 = stablehlo.reshape %v3413 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3432 = stablehlo.dot_general %v3430, %v3431, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3433 = stablehlo.reshape %v3432 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3434 = stablehlo.reshape %v3433 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3435 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3436 = stablehlo.pad %v3434, %v3435, low = [0, 0, 640], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v3437 = stablehlo.reshape %v3436 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3438 = stablehlo.add %v3404, %v3437 : tensor<32x151296xf32>
    %v3439 = stablehlo.reshape %v3055 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3440 = stablehlo.slice %v3439 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3441 = stablehlo.reshape %v3440 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3442 = stablehlo.reshape %v3060 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3443 = stablehlo.slice %v3442 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3444 = stablehlo.reshape %v3443 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3445 = stablehlo.reshape %v3065 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3446 = stablehlo.slice %v3445 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3447 = stablehlo.reshape %v3446 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3448 = stablehlo.reshape %v3444 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3449 = stablehlo.transpose %v3448, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3450 = stablehlo.reshape %v3449 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3451 = stablehlo.reshape %v3441 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3452 = stablehlo.reshape %v3450 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3453 = stablehlo.dot_general %v3451, %v3452, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3454 = stablehlo.reshape %v3453 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3455 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3456 = stablehlo.multiply %v3454, %v3455 : tensor<32x38809xf32>
    %v3457 = stablehlo.reshape %v3456 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3458 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3459 = stablehlo.exponential %v3457 : tensor<32x197x197xf32>
    %v3460 = stablehlo.reduce(%v3459 init: %v3458) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3461 = stablehlo.broadcast_in_dim %v3460, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3462 = stablehlo.divide %v3459, %v3461 : tensor<32x197x197xf32>
    %v3463 = stablehlo.reshape %v3462 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3464 = stablehlo.reshape %v3463 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3465 = stablehlo.reshape %v3447 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3466 = stablehlo.dot_general %v3464, %v3465, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3467 = stablehlo.reshape %v3466 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3468 = stablehlo.reshape %v3467 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3469 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3470 = stablehlo.pad %v3468, %v3469, low = [0, 0, 704], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v3471 = stablehlo.reshape %v3470 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3472 = stablehlo.add %v3438, %v3471 : tensor<32x151296xf32>
    %v3473 = stablehlo.reshape %v3472 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3474 = stablehlo.dot_general %v3473, %b6_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v3475 = stablehlo.broadcast_in_dim %b6_bo, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v3476 = stablehlo.add %v3474, %v3475 : tensor<32x197x768xf32>
    %v3477 = stablehlo.reshape %v3476 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3478 = stablehlo.add %v3022, %v3477 : tensor<32x151296xf32>
    %v3479 = stablehlo.reshape %v3478 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3480 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3481 = stablehlo.constant dense<768.0> : tensor<32x197x768xf32>
    %v3482 = stablehlo.constant dense<1.0e-5> : tensor<32x197x768xf32>
    %v3483 = stablehlo.reduce(%v3479 init: %v3480) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3484 = stablehlo.broadcast_in_dim %v3483, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v3485 = stablehlo.divide %v3484, %v3481 : tensor<32x197x768xf32>
    %v3486 = stablehlo.subtract %v3479, %v3485 : tensor<32x197x768xf32>
    %v3487 = stablehlo.multiply %v3486, %v3486 : tensor<32x197x768xf32>
    %v3488 = stablehlo.reduce(%v3487 init: %v3480) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3489 = stablehlo.broadcast_in_dim %v3488, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v3490 = stablehlo.divide %v3489, %v3481 : tensor<32x197x768xf32>
    %v3491 = stablehlo.add %v3490, %v3482 : tensor<32x197x768xf32>
    %v3492 = stablehlo.rsqrt %v3491 : tensor<32x197x768xf32>
    %v3493 = stablehlo.multiply %v3486, %v3492 : tensor<32x197x768xf32>
    %v3494 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v3495 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v3496 = stablehlo.multiply %v3493, %v3494 : tensor<32x197x768xf32>
    %v3497 = stablehlo.add %v3496, %v3495 : tensor<32x197x768xf32>
    %v3498 = stablehlo.reshape %v3497 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3499 = stablehlo.reshape %v3498 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3500 = stablehlo.broadcast_in_dim %b6_g2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v3501 = stablehlo.multiply %v3499, %v3500 : tensor<32x197x768xf32>
    %v3502 = stablehlo.reshape %v3501 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3503 = stablehlo.reshape %v3502 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3504 = stablehlo.broadcast_in_dim %b6_bt2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v3505 = stablehlo.add %v3503, %v3504 : tensor<32x197x768xf32>
    %v3506 = stablehlo.reshape %v3505 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3507 = stablehlo.reshape %v3506 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3508 = stablehlo.dot_general %v3507, %b6_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x3072xf32>) -> tensor<32x197x3072xf32>
    %v3509 = stablehlo.broadcast_in_dim %b6_bfc1, dims = [2] : (tensor<3072xf32>) -> tensor<32x197x3072xf32>
    %v3510 = stablehlo.add %v3508, %v3509 : tensor<32x197x3072xf32>
    %v3511 = stablehlo.reshape %v3510 : (tensor<32x197x3072xf32>) -> tensor<32x605184xf32>
    %v3512 = stablehlo.constant dense<0.5> : tensor<32x605184xf32>
    %v3513 = stablehlo.multiply %v3512, %v3511 : tensor<32x605184xf32>
    %v3514 = stablehlo.negate %v3511 : tensor<32x605184xf32>
    %v3515 = stablehlo.constant dense<0.7071067811865476> : tensor<32x605184xf32>
    %v3516 = stablehlo.multiply %v3514, %v3515 : tensor<32x605184xf32>
    %v3517 = chlo.erfc %v3516 : tensor<32x605184xf32> -> tensor<32x605184xf32>
    %v3518 = stablehlo.multiply %v3513, %v3517 : tensor<32x605184xf32>
    %v3519 = stablehlo.reshape %v3518 : (tensor<32x605184xf32>) -> tensor<32x197x3072xf32>
    %v3520 = stablehlo.dot_general %v3519, %b6_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x3072xf32>, tensor<3072x768xf32>) -> tensor<32x197x768xf32>
    %v3521 = stablehlo.broadcast_in_dim %b6_bfc2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v3522 = stablehlo.add %v3520, %v3521 : tensor<32x197x768xf32>
    %v3523 = stablehlo.reshape %v3522 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3524 = stablehlo.add %v3478, %v3523 : tensor<32x151296xf32>
    %v3525 = stablehlo.reshape %v3524 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3526 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3527 = stablehlo.constant dense<768.0> : tensor<32x197x768xf32>
    %v3528 = stablehlo.constant dense<1.0e-5> : tensor<32x197x768xf32>
    %v3529 = stablehlo.reduce(%v3525 init: %v3526) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3530 = stablehlo.broadcast_in_dim %v3529, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v3531 = stablehlo.divide %v3530, %v3527 : tensor<32x197x768xf32>
    %v3532 = stablehlo.subtract %v3525, %v3531 : tensor<32x197x768xf32>
    %v3533 = stablehlo.multiply %v3532, %v3532 : tensor<32x197x768xf32>
    %v3534 = stablehlo.reduce(%v3533 init: %v3526) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3535 = stablehlo.broadcast_in_dim %v3534, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v3536 = stablehlo.divide %v3535, %v3527 : tensor<32x197x768xf32>
    %v3537 = stablehlo.add %v3536, %v3528 : tensor<32x197x768xf32>
    %v3538 = stablehlo.rsqrt %v3537 : tensor<32x197x768xf32>
    %v3539 = stablehlo.multiply %v3532, %v3538 : tensor<32x197x768xf32>
    %v3540 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v3541 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v3542 = stablehlo.multiply %v3539, %v3540 : tensor<32x197x768xf32>
    %v3543 = stablehlo.add %v3542, %v3541 : tensor<32x197x768xf32>
    %v3544 = stablehlo.reshape %v3543 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3545 = stablehlo.reshape %v3544 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3546 = stablehlo.broadcast_in_dim %b7_g1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v3547 = stablehlo.multiply %v3545, %v3546 : tensor<32x197x768xf32>
    %v3548 = stablehlo.reshape %v3547 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3549 = stablehlo.reshape %v3548 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3550 = stablehlo.broadcast_in_dim %b7_bt1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v3551 = stablehlo.add %v3549, %v3550 : tensor<32x197x768xf32>
    %v3552 = stablehlo.reshape %v3551 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3553 = stablehlo.reshape %v3552 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3554 = stablehlo.dot_general %v3553, %b7_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v3555 = stablehlo.broadcast_in_dim %b7_bq, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v3556 = stablehlo.add %v3554, %v3555 : tensor<32x197x768xf32>
    %v3557 = stablehlo.reshape %v3556 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3558 = stablehlo.reshape %v3552 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3559 = stablehlo.dot_general %v3558, %b7_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v3560 = stablehlo.broadcast_in_dim %b7_bk, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v3561 = stablehlo.add %v3559, %v3560 : tensor<32x197x768xf32>
    %v3562 = stablehlo.reshape %v3561 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3563 = stablehlo.reshape %v3552 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3564 = stablehlo.dot_general %v3563, %b7_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v3565 = stablehlo.broadcast_in_dim %b7_bv, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v3566 = stablehlo.add %v3564, %v3565 : tensor<32x197x768xf32>
    %v3567 = stablehlo.reshape %v3566 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3568 = stablehlo.reshape %v3557 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3569 = stablehlo.slice %v3568 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3570 = stablehlo.reshape %v3569 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3571 = stablehlo.reshape %v3562 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3572 = stablehlo.slice %v3571 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3573 = stablehlo.reshape %v3572 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3574 = stablehlo.reshape %v3567 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3575 = stablehlo.slice %v3574 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3576 = stablehlo.reshape %v3575 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3577 = stablehlo.reshape %v3573 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3578 = stablehlo.transpose %v3577, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3579 = stablehlo.reshape %v3578 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3580 = stablehlo.reshape %v3570 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3581 = stablehlo.reshape %v3579 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3582 = stablehlo.dot_general %v3580, %v3581, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3583 = stablehlo.reshape %v3582 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3584 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3585 = stablehlo.multiply %v3583, %v3584 : tensor<32x38809xf32>
    %v3586 = stablehlo.reshape %v3585 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3587 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3588 = stablehlo.exponential %v3586 : tensor<32x197x197xf32>
    %v3589 = stablehlo.reduce(%v3588 init: %v3587) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3590 = stablehlo.broadcast_in_dim %v3589, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3591 = stablehlo.divide %v3588, %v3590 : tensor<32x197x197xf32>
    %v3592 = stablehlo.reshape %v3591 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3593 = stablehlo.reshape %v3592 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3594 = stablehlo.reshape %v3576 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3595 = stablehlo.dot_general %v3593, %v3594, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3596 = stablehlo.reshape %v3595 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3597 = stablehlo.reshape %v3596 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3598 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3599 = stablehlo.pad %v3597, %v3598, low = [0, 0, 0], high = [0, 0, 704], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v3600 = stablehlo.reshape %v3599 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3601 = stablehlo.reshape %v3557 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3602 = stablehlo.slice %v3601 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3603 = stablehlo.reshape %v3602 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3604 = stablehlo.reshape %v3562 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3605 = stablehlo.slice %v3604 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3606 = stablehlo.reshape %v3605 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3607 = stablehlo.reshape %v3567 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3608 = stablehlo.slice %v3607 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3609 = stablehlo.reshape %v3608 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3610 = stablehlo.reshape %v3606 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3611 = stablehlo.transpose %v3610, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3612 = stablehlo.reshape %v3611 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3613 = stablehlo.reshape %v3603 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3614 = stablehlo.reshape %v3612 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3615 = stablehlo.dot_general %v3613, %v3614, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3616 = stablehlo.reshape %v3615 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3617 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3618 = stablehlo.multiply %v3616, %v3617 : tensor<32x38809xf32>
    %v3619 = stablehlo.reshape %v3618 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3620 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3621 = stablehlo.exponential %v3619 : tensor<32x197x197xf32>
    %v3622 = stablehlo.reduce(%v3621 init: %v3620) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3623 = stablehlo.broadcast_in_dim %v3622, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3624 = stablehlo.divide %v3621, %v3623 : tensor<32x197x197xf32>
    %v3625 = stablehlo.reshape %v3624 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3626 = stablehlo.reshape %v3625 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3627 = stablehlo.reshape %v3609 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3628 = stablehlo.dot_general %v3626, %v3627, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3629 = stablehlo.reshape %v3628 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3630 = stablehlo.reshape %v3629 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3631 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3632 = stablehlo.pad %v3630, %v3631, low = [0, 0, 64], high = [0, 0, 640], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v3633 = stablehlo.reshape %v3632 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3634 = stablehlo.add %v3600, %v3633 : tensor<32x151296xf32>
    %v3635 = stablehlo.reshape %v3557 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3636 = stablehlo.slice %v3635 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3637 = stablehlo.reshape %v3636 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3638 = stablehlo.reshape %v3562 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3639 = stablehlo.slice %v3638 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3640 = stablehlo.reshape %v3639 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3641 = stablehlo.reshape %v3567 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3642 = stablehlo.slice %v3641 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3643 = stablehlo.reshape %v3642 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3644 = stablehlo.reshape %v3640 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3645 = stablehlo.transpose %v3644, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3646 = stablehlo.reshape %v3645 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3647 = stablehlo.reshape %v3637 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3648 = stablehlo.reshape %v3646 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3649 = stablehlo.dot_general %v3647, %v3648, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3650 = stablehlo.reshape %v3649 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3651 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3652 = stablehlo.multiply %v3650, %v3651 : tensor<32x38809xf32>
    %v3653 = stablehlo.reshape %v3652 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3654 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3655 = stablehlo.exponential %v3653 : tensor<32x197x197xf32>
    %v3656 = stablehlo.reduce(%v3655 init: %v3654) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3657 = stablehlo.broadcast_in_dim %v3656, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3658 = stablehlo.divide %v3655, %v3657 : tensor<32x197x197xf32>
    %v3659 = stablehlo.reshape %v3658 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3660 = stablehlo.reshape %v3659 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3661 = stablehlo.reshape %v3643 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3662 = stablehlo.dot_general %v3660, %v3661, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3663 = stablehlo.reshape %v3662 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3664 = stablehlo.reshape %v3663 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3665 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3666 = stablehlo.pad %v3664, %v3665, low = [0, 0, 128], high = [0, 0, 576], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v3667 = stablehlo.reshape %v3666 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3668 = stablehlo.add %v3634, %v3667 : tensor<32x151296xf32>
    %v3669 = stablehlo.reshape %v3557 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3670 = stablehlo.slice %v3669 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3671 = stablehlo.reshape %v3670 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3672 = stablehlo.reshape %v3562 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3673 = stablehlo.slice %v3672 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3674 = stablehlo.reshape %v3673 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3675 = stablehlo.reshape %v3567 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3676 = stablehlo.slice %v3675 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3677 = stablehlo.reshape %v3676 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3678 = stablehlo.reshape %v3674 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3679 = stablehlo.transpose %v3678, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3680 = stablehlo.reshape %v3679 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3681 = stablehlo.reshape %v3671 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3682 = stablehlo.reshape %v3680 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3683 = stablehlo.dot_general %v3681, %v3682, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3684 = stablehlo.reshape %v3683 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3685 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3686 = stablehlo.multiply %v3684, %v3685 : tensor<32x38809xf32>
    %v3687 = stablehlo.reshape %v3686 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3688 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3689 = stablehlo.exponential %v3687 : tensor<32x197x197xf32>
    %v3690 = stablehlo.reduce(%v3689 init: %v3688) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3691 = stablehlo.broadcast_in_dim %v3690, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3692 = stablehlo.divide %v3689, %v3691 : tensor<32x197x197xf32>
    %v3693 = stablehlo.reshape %v3692 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3694 = stablehlo.reshape %v3693 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3695 = stablehlo.reshape %v3677 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3696 = stablehlo.dot_general %v3694, %v3695, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3697 = stablehlo.reshape %v3696 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3698 = stablehlo.reshape %v3697 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3699 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3700 = stablehlo.pad %v3698, %v3699, low = [0, 0, 192], high = [0, 0, 512], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v3701 = stablehlo.reshape %v3700 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3702 = stablehlo.add %v3668, %v3701 : tensor<32x151296xf32>
    %v3703 = stablehlo.reshape %v3557 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3704 = stablehlo.slice %v3703 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3705 = stablehlo.reshape %v3704 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3706 = stablehlo.reshape %v3562 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3707 = stablehlo.slice %v3706 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3708 = stablehlo.reshape %v3707 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3709 = stablehlo.reshape %v3567 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3710 = stablehlo.slice %v3709 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3711 = stablehlo.reshape %v3710 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3712 = stablehlo.reshape %v3708 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3713 = stablehlo.transpose %v3712, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3714 = stablehlo.reshape %v3713 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3715 = stablehlo.reshape %v3705 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3716 = stablehlo.reshape %v3714 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3717 = stablehlo.dot_general %v3715, %v3716, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3718 = stablehlo.reshape %v3717 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3719 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3720 = stablehlo.multiply %v3718, %v3719 : tensor<32x38809xf32>
    %v3721 = stablehlo.reshape %v3720 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3722 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3723 = stablehlo.exponential %v3721 : tensor<32x197x197xf32>
    %v3724 = stablehlo.reduce(%v3723 init: %v3722) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3725 = stablehlo.broadcast_in_dim %v3724, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3726 = stablehlo.divide %v3723, %v3725 : tensor<32x197x197xf32>
    %v3727 = stablehlo.reshape %v3726 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3728 = stablehlo.reshape %v3727 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3729 = stablehlo.reshape %v3711 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3730 = stablehlo.dot_general %v3728, %v3729, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3731 = stablehlo.reshape %v3730 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3732 = stablehlo.reshape %v3731 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3733 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3734 = stablehlo.pad %v3732, %v3733, low = [0, 0, 256], high = [0, 0, 448], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v3735 = stablehlo.reshape %v3734 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3736 = stablehlo.add %v3702, %v3735 : tensor<32x151296xf32>
    %v3737 = stablehlo.reshape %v3557 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3738 = stablehlo.slice %v3737 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3739 = stablehlo.reshape %v3738 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3740 = stablehlo.reshape %v3562 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3741 = stablehlo.slice %v3740 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3742 = stablehlo.reshape %v3741 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3743 = stablehlo.reshape %v3567 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3744 = stablehlo.slice %v3743 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3745 = stablehlo.reshape %v3744 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3746 = stablehlo.reshape %v3742 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3747 = stablehlo.transpose %v3746, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3748 = stablehlo.reshape %v3747 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3749 = stablehlo.reshape %v3739 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3750 = stablehlo.reshape %v3748 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3751 = stablehlo.dot_general %v3749, %v3750, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3752 = stablehlo.reshape %v3751 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3753 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3754 = stablehlo.multiply %v3752, %v3753 : tensor<32x38809xf32>
    %v3755 = stablehlo.reshape %v3754 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3756 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3757 = stablehlo.exponential %v3755 : tensor<32x197x197xf32>
    %v3758 = stablehlo.reduce(%v3757 init: %v3756) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3759 = stablehlo.broadcast_in_dim %v3758, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3760 = stablehlo.divide %v3757, %v3759 : tensor<32x197x197xf32>
    %v3761 = stablehlo.reshape %v3760 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3762 = stablehlo.reshape %v3761 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3763 = stablehlo.reshape %v3745 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3764 = stablehlo.dot_general %v3762, %v3763, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3765 = stablehlo.reshape %v3764 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3766 = stablehlo.reshape %v3765 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3767 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3768 = stablehlo.pad %v3766, %v3767, low = [0, 0, 320], high = [0, 0, 384], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v3769 = stablehlo.reshape %v3768 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3770 = stablehlo.add %v3736, %v3769 : tensor<32x151296xf32>
    %v3771 = stablehlo.reshape %v3557 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3772 = stablehlo.slice %v3771 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3773 = stablehlo.reshape %v3772 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3774 = stablehlo.reshape %v3562 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3775 = stablehlo.slice %v3774 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3776 = stablehlo.reshape %v3775 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3777 = stablehlo.reshape %v3567 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3778 = stablehlo.slice %v3777 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3779 = stablehlo.reshape %v3778 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3780 = stablehlo.reshape %v3776 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3781 = stablehlo.transpose %v3780, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3782 = stablehlo.reshape %v3781 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3783 = stablehlo.reshape %v3773 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3784 = stablehlo.reshape %v3782 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3785 = stablehlo.dot_general %v3783, %v3784, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3786 = stablehlo.reshape %v3785 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3787 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3788 = stablehlo.multiply %v3786, %v3787 : tensor<32x38809xf32>
    %v3789 = stablehlo.reshape %v3788 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3790 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3791 = stablehlo.exponential %v3789 : tensor<32x197x197xf32>
    %v3792 = stablehlo.reduce(%v3791 init: %v3790) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3793 = stablehlo.broadcast_in_dim %v3792, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3794 = stablehlo.divide %v3791, %v3793 : tensor<32x197x197xf32>
    %v3795 = stablehlo.reshape %v3794 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3796 = stablehlo.reshape %v3795 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3797 = stablehlo.reshape %v3779 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3798 = stablehlo.dot_general %v3796, %v3797, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3799 = stablehlo.reshape %v3798 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3800 = stablehlo.reshape %v3799 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3801 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3802 = stablehlo.pad %v3800, %v3801, low = [0, 0, 384], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v3803 = stablehlo.reshape %v3802 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3804 = stablehlo.add %v3770, %v3803 : tensor<32x151296xf32>
    %v3805 = stablehlo.reshape %v3557 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3806 = stablehlo.slice %v3805 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3807 = stablehlo.reshape %v3806 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3808 = stablehlo.reshape %v3562 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3809 = stablehlo.slice %v3808 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3810 = stablehlo.reshape %v3809 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3811 = stablehlo.reshape %v3567 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3812 = stablehlo.slice %v3811 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3813 = stablehlo.reshape %v3812 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3814 = stablehlo.reshape %v3810 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3815 = stablehlo.transpose %v3814, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3816 = stablehlo.reshape %v3815 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3817 = stablehlo.reshape %v3807 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3818 = stablehlo.reshape %v3816 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3819 = stablehlo.dot_general %v3817, %v3818, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3820 = stablehlo.reshape %v3819 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3821 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3822 = stablehlo.multiply %v3820, %v3821 : tensor<32x38809xf32>
    %v3823 = stablehlo.reshape %v3822 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3824 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3825 = stablehlo.exponential %v3823 : tensor<32x197x197xf32>
    %v3826 = stablehlo.reduce(%v3825 init: %v3824) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3827 = stablehlo.broadcast_in_dim %v3826, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3828 = stablehlo.divide %v3825, %v3827 : tensor<32x197x197xf32>
    %v3829 = stablehlo.reshape %v3828 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3830 = stablehlo.reshape %v3829 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3831 = stablehlo.reshape %v3813 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3832 = stablehlo.dot_general %v3830, %v3831, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3833 = stablehlo.reshape %v3832 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3834 = stablehlo.reshape %v3833 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3835 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3836 = stablehlo.pad %v3834, %v3835, low = [0, 0, 448], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v3837 = stablehlo.reshape %v3836 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3838 = stablehlo.add %v3804, %v3837 : tensor<32x151296xf32>
    %v3839 = stablehlo.reshape %v3557 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3840 = stablehlo.slice %v3839 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3841 = stablehlo.reshape %v3840 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3842 = stablehlo.reshape %v3562 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3843 = stablehlo.slice %v3842 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3844 = stablehlo.reshape %v3843 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3845 = stablehlo.reshape %v3567 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3846 = stablehlo.slice %v3845 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3847 = stablehlo.reshape %v3846 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3848 = stablehlo.reshape %v3844 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3849 = stablehlo.transpose %v3848, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3850 = stablehlo.reshape %v3849 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3851 = stablehlo.reshape %v3841 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3852 = stablehlo.reshape %v3850 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3853 = stablehlo.dot_general %v3851, %v3852, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3854 = stablehlo.reshape %v3853 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3855 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3856 = stablehlo.multiply %v3854, %v3855 : tensor<32x38809xf32>
    %v3857 = stablehlo.reshape %v3856 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3858 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3859 = stablehlo.exponential %v3857 : tensor<32x197x197xf32>
    %v3860 = stablehlo.reduce(%v3859 init: %v3858) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3861 = stablehlo.broadcast_in_dim %v3860, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3862 = stablehlo.divide %v3859, %v3861 : tensor<32x197x197xf32>
    %v3863 = stablehlo.reshape %v3862 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3864 = stablehlo.reshape %v3863 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3865 = stablehlo.reshape %v3847 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3866 = stablehlo.dot_general %v3864, %v3865, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3867 = stablehlo.reshape %v3866 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3868 = stablehlo.reshape %v3867 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3869 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3870 = stablehlo.pad %v3868, %v3869, low = [0, 0, 512], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v3871 = stablehlo.reshape %v3870 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3872 = stablehlo.add %v3838, %v3871 : tensor<32x151296xf32>
    %v3873 = stablehlo.reshape %v3557 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3874 = stablehlo.slice %v3873 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3875 = stablehlo.reshape %v3874 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3876 = stablehlo.reshape %v3562 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3877 = stablehlo.slice %v3876 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3878 = stablehlo.reshape %v3877 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3879 = stablehlo.reshape %v3567 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3880 = stablehlo.slice %v3879 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3881 = stablehlo.reshape %v3880 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3882 = stablehlo.reshape %v3878 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3883 = stablehlo.transpose %v3882, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3884 = stablehlo.reshape %v3883 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3885 = stablehlo.reshape %v3875 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3886 = stablehlo.reshape %v3884 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3887 = stablehlo.dot_general %v3885, %v3886, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3888 = stablehlo.reshape %v3887 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3889 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3890 = stablehlo.multiply %v3888, %v3889 : tensor<32x38809xf32>
    %v3891 = stablehlo.reshape %v3890 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3892 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3893 = stablehlo.exponential %v3891 : tensor<32x197x197xf32>
    %v3894 = stablehlo.reduce(%v3893 init: %v3892) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3895 = stablehlo.broadcast_in_dim %v3894, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3896 = stablehlo.divide %v3893, %v3895 : tensor<32x197x197xf32>
    %v3897 = stablehlo.reshape %v3896 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3898 = stablehlo.reshape %v3897 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3899 = stablehlo.reshape %v3881 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3900 = stablehlo.dot_general %v3898, %v3899, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3901 = stablehlo.reshape %v3900 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3902 = stablehlo.reshape %v3901 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3903 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3904 = stablehlo.pad %v3902, %v3903, low = [0, 0, 576], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v3905 = stablehlo.reshape %v3904 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3906 = stablehlo.add %v3872, %v3905 : tensor<32x151296xf32>
    %v3907 = stablehlo.reshape %v3557 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3908 = stablehlo.slice %v3907 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3909 = stablehlo.reshape %v3908 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3910 = stablehlo.reshape %v3562 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3911 = stablehlo.slice %v3910 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3912 = stablehlo.reshape %v3911 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3913 = stablehlo.reshape %v3567 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3914 = stablehlo.slice %v3913 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3915 = stablehlo.reshape %v3914 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3916 = stablehlo.reshape %v3912 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3917 = stablehlo.transpose %v3916, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3918 = stablehlo.reshape %v3917 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3919 = stablehlo.reshape %v3909 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3920 = stablehlo.reshape %v3918 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3921 = stablehlo.dot_general %v3919, %v3920, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3922 = stablehlo.reshape %v3921 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3923 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3924 = stablehlo.multiply %v3922, %v3923 : tensor<32x38809xf32>
    %v3925 = stablehlo.reshape %v3924 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3926 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3927 = stablehlo.exponential %v3925 : tensor<32x197x197xf32>
    %v3928 = stablehlo.reduce(%v3927 init: %v3926) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3929 = stablehlo.broadcast_in_dim %v3928, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3930 = stablehlo.divide %v3927, %v3929 : tensor<32x197x197xf32>
    %v3931 = stablehlo.reshape %v3930 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3932 = stablehlo.reshape %v3931 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3933 = stablehlo.reshape %v3915 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3934 = stablehlo.dot_general %v3932, %v3933, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3935 = stablehlo.reshape %v3934 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3936 = stablehlo.reshape %v3935 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3937 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3938 = stablehlo.pad %v3936, %v3937, low = [0, 0, 640], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v3939 = stablehlo.reshape %v3938 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3940 = stablehlo.add %v3906, %v3939 : tensor<32x151296xf32>
    %v3941 = stablehlo.reshape %v3557 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3942 = stablehlo.slice %v3941 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3943 = stablehlo.reshape %v3942 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3944 = stablehlo.reshape %v3562 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3945 = stablehlo.slice %v3944 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3946 = stablehlo.reshape %v3945 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3947 = stablehlo.reshape %v3567 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3948 = stablehlo.slice %v3947 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3949 = stablehlo.reshape %v3948 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3950 = stablehlo.reshape %v3946 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3951 = stablehlo.transpose %v3950, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3952 = stablehlo.reshape %v3951 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3953 = stablehlo.reshape %v3943 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3954 = stablehlo.reshape %v3952 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3955 = stablehlo.dot_general %v3953, %v3954, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3956 = stablehlo.reshape %v3955 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3957 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3958 = stablehlo.multiply %v3956, %v3957 : tensor<32x38809xf32>
    %v3959 = stablehlo.reshape %v3958 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3960 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3961 = stablehlo.exponential %v3959 : tensor<32x197x197xf32>
    %v3962 = stablehlo.reduce(%v3961 init: %v3960) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3963 = stablehlo.broadcast_in_dim %v3962, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3964 = stablehlo.divide %v3961, %v3963 : tensor<32x197x197xf32>
    %v3965 = stablehlo.reshape %v3964 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3966 = stablehlo.reshape %v3965 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3967 = stablehlo.reshape %v3949 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3968 = stablehlo.dot_general %v3966, %v3967, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3969 = stablehlo.reshape %v3968 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3970 = stablehlo.reshape %v3969 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3971 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3972 = stablehlo.pad %v3970, %v3971, low = [0, 0, 704], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v3973 = stablehlo.reshape %v3972 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3974 = stablehlo.add %v3940, %v3973 : tensor<32x151296xf32>
    %v3975 = stablehlo.reshape %v3974 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3976 = stablehlo.dot_general %v3975, %b7_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v3977 = stablehlo.broadcast_in_dim %b7_bo, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v3978 = stablehlo.add %v3976, %v3977 : tensor<32x197x768xf32>
    %v3979 = stablehlo.reshape %v3978 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3980 = stablehlo.add %v3524, %v3979 : tensor<32x151296xf32>
    %v3981 = stablehlo.reshape %v3980 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3982 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3983 = stablehlo.constant dense<768.0> : tensor<32x197x768xf32>
    %v3984 = stablehlo.constant dense<1.0e-5> : tensor<32x197x768xf32>
    %v3985 = stablehlo.reduce(%v3981 init: %v3982) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3986 = stablehlo.broadcast_in_dim %v3985, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v3987 = stablehlo.divide %v3986, %v3983 : tensor<32x197x768xf32>
    %v3988 = stablehlo.subtract %v3981, %v3987 : tensor<32x197x768xf32>
    %v3989 = stablehlo.multiply %v3988, %v3988 : tensor<32x197x768xf32>
    %v3990 = stablehlo.reduce(%v3989 init: %v3982) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3991 = stablehlo.broadcast_in_dim %v3990, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v3992 = stablehlo.divide %v3991, %v3983 : tensor<32x197x768xf32>
    %v3993 = stablehlo.add %v3992, %v3984 : tensor<32x197x768xf32>
    %v3994 = stablehlo.rsqrt %v3993 : tensor<32x197x768xf32>
    %v3995 = stablehlo.multiply %v3988, %v3994 : tensor<32x197x768xf32>
    %v3996 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v3997 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v3998 = stablehlo.multiply %v3995, %v3996 : tensor<32x197x768xf32>
    %v3999 = stablehlo.add %v3998, %v3997 : tensor<32x197x768xf32>
    %v4000 = stablehlo.reshape %v3999 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4001 = stablehlo.reshape %v4000 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4002 = stablehlo.broadcast_in_dim %b7_g2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v4003 = stablehlo.multiply %v4001, %v4002 : tensor<32x197x768xf32>
    %v4004 = stablehlo.reshape %v4003 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4005 = stablehlo.reshape %v4004 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4006 = stablehlo.broadcast_in_dim %b7_bt2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v4007 = stablehlo.add %v4005, %v4006 : tensor<32x197x768xf32>
    %v4008 = stablehlo.reshape %v4007 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4009 = stablehlo.reshape %v4008 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4010 = stablehlo.dot_general %v4009, %b7_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x3072xf32>) -> tensor<32x197x3072xf32>
    %v4011 = stablehlo.broadcast_in_dim %b7_bfc1, dims = [2] : (tensor<3072xf32>) -> tensor<32x197x3072xf32>
    %v4012 = stablehlo.add %v4010, %v4011 : tensor<32x197x3072xf32>
    %v4013 = stablehlo.reshape %v4012 : (tensor<32x197x3072xf32>) -> tensor<32x605184xf32>
    %v4014 = stablehlo.constant dense<0.5> : tensor<32x605184xf32>
    %v4015 = stablehlo.multiply %v4014, %v4013 : tensor<32x605184xf32>
    %v4016 = stablehlo.negate %v4013 : tensor<32x605184xf32>
    %v4017 = stablehlo.constant dense<0.7071067811865476> : tensor<32x605184xf32>
    %v4018 = stablehlo.multiply %v4016, %v4017 : tensor<32x605184xf32>
    %v4019 = chlo.erfc %v4018 : tensor<32x605184xf32> -> tensor<32x605184xf32>
    %v4020 = stablehlo.multiply %v4015, %v4019 : tensor<32x605184xf32>
    %v4021 = stablehlo.reshape %v4020 : (tensor<32x605184xf32>) -> tensor<32x197x3072xf32>
    %v4022 = stablehlo.dot_general %v4021, %b7_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x3072xf32>, tensor<3072x768xf32>) -> tensor<32x197x768xf32>
    %v4023 = stablehlo.broadcast_in_dim %b7_bfc2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v4024 = stablehlo.add %v4022, %v4023 : tensor<32x197x768xf32>
    %v4025 = stablehlo.reshape %v4024 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4026 = stablehlo.add %v3980, %v4025 : tensor<32x151296xf32>
    %v4027 = stablehlo.reshape %v4026 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4028 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4029 = stablehlo.constant dense<768.0> : tensor<32x197x768xf32>
    %v4030 = stablehlo.constant dense<1.0e-5> : tensor<32x197x768xf32>
    %v4031 = stablehlo.reduce(%v4027 init: %v4028) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4032 = stablehlo.broadcast_in_dim %v4031, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v4033 = stablehlo.divide %v4032, %v4029 : tensor<32x197x768xf32>
    %v4034 = stablehlo.subtract %v4027, %v4033 : tensor<32x197x768xf32>
    %v4035 = stablehlo.multiply %v4034, %v4034 : tensor<32x197x768xf32>
    %v4036 = stablehlo.reduce(%v4035 init: %v4028) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4037 = stablehlo.broadcast_in_dim %v4036, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v4038 = stablehlo.divide %v4037, %v4029 : tensor<32x197x768xf32>
    %v4039 = stablehlo.add %v4038, %v4030 : tensor<32x197x768xf32>
    %v4040 = stablehlo.rsqrt %v4039 : tensor<32x197x768xf32>
    %v4041 = stablehlo.multiply %v4034, %v4040 : tensor<32x197x768xf32>
    %v4042 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v4043 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v4044 = stablehlo.multiply %v4041, %v4042 : tensor<32x197x768xf32>
    %v4045 = stablehlo.add %v4044, %v4043 : tensor<32x197x768xf32>
    %v4046 = stablehlo.reshape %v4045 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4047 = stablehlo.reshape %v4046 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4048 = stablehlo.broadcast_in_dim %b8_g1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v4049 = stablehlo.multiply %v4047, %v4048 : tensor<32x197x768xf32>
    %v4050 = stablehlo.reshape %v4049 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4051 = stablehlo.reshape %v4050 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4052 = stablehlo.broadcast_in_dim %b8_bt1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v4053 = stablehlo.add %v4051, %v4052 : tensor<32x197x768xf32>
    %v4054 = stablehlo.reshape %v4053 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4055 = stablehlo.reshape %v4054 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4056 = stablehlo.dot_general %v4055, %b8_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v4057 = stablehlo.broadcast_in_dim %b8_bq, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v4058 = stablehlo.add %v4056, %v4057 : tensor<32x197x768xf32>
    %v4059 = stablehlo.reshape %v4058 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4060 = stablehlo.reshape %v4054 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4061 = stablehlo.dot_general %v4060, %b8_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v4062 = stablehlo.broadcast_in_dim %b8_bk, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v4063 = stablehlo.add %v4061, %v4062 : tensor<32x197x768xf32>
    %v4064 = stablehlo.reshape %v4063 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4065 = stablehlo.reshape %v4054 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4066 = stablehlo.dot_general %v4065, %b8_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v4067 = stablehlo.broadcast_in_dim %b8_bv, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v4068 = stablehlo.add %v4066, %v4067 : tensor<32x197x768xf32>
    %v4069 = stablehlo.reshape %v4068 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4070 = stablehlo.reshape %v4059 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4071 = stablehlo.slice %v4070 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4072 = stablehlo.reshape %v4071 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4073 = stablehlo.reshape %v4064 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4074 = stablehlo.slice %v4073 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4075 = stablehlo.reshape %v4074 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4076 = stablehlo.reshape %v4069 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4077 = stablehlo.slice %v4076 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4078 = stablehlo.reshape %v4077 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4079 = stablehlo.reshape %v4075 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4080 = stablehlo.transpose %v4079, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v4081 = stablehlo.reshape %v4080 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4082 = stablehlo.reshape %v4072 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4083 = stablehlo.reshape %v4081 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4084 = stablehlo.dot_general %v4082, %v4083, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v4085 = stablehlo.reshape %v4084 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4086 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v4087 = stablehlo.multiply %v4085, %v4086 : tensor<32x38809xf32>
    %v4088 = stablehlo.reshape %v4087 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4089 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4090 = stablehlo.exponential %v4088 : tensor<32x197x197xf32>
    %v4091 = stablehlo.reduce(%v4090 init: %v4089) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4092 = stablehlo.broadcast_in_dim %v4091, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4093 = stablehlo.divide %v4090, %v4092 : tensor<32x197x197xf32>
    %v4094 = stablehlo.reshape %v4093 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4095 = stablehlo.reshape %v4094 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4096 = stablehlo.reshape %v4078 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4097 = stablehlo.dot_general %v4095, %v4096, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4098 = stablehlo.reshape %v4097 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4099 = stablehlo.reshape %v4098 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4100 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4101 = stablehlo.pad %v4099, %v4100, low = [0, 0, 0], high = [0, 0, 704], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v4102 = stablehlo.reshape %v4101 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4103 = stablehlo.reshape %v4059 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4104 = stablehlo.slice %v4103 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4105 = stablehlo.reshape %v4104 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4106 = stablehlo.reshape %v4064 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4107 = stablehlo.slice %v4106 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4108 = stablehlo.reshape %v4107 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4109 = stablehlo.reshape %v4069 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4110 = stablehlo.slice %v4109 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4111 = stablehlo.reshape %v4110 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4112 = stablehlo.reshape %v4108 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4113 = stablehlo.transpose %v4112, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v4114 = stablehlo.reshape %v4113 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4115 = stablehlo.reshape %v4105 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4116 = stablehlo.reshape %v4114 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4117 = stablehlo.dot_general %v4115, %v4116, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v4118 = stablehlo.reshape %v4117 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4119 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v4120 = stablehlo.multiply %v4118, %v4119 : tensor<32x38809xf32>
    %v4121 = stablehlo.reshape %v4120 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4122 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4123 = stablehlo.exponential %v4121 : tensor<32x197x197xf32>
    %v4124 = stablehlo.reduce(%v4123 init: %v4122) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4125 = stablehlo.broadcast_in_dim %v4124, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4126 = stablehlo.divide %v4123, %v4125 : tensor<32x197x197xf32>
    %v4127 = stablehlo.reshape %v4126 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4128 = stablehlo.reshape %v4127 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4129 = stablehlo.reshape %v4111 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4130 = stablehlo.dot_general %v4128, %v4129, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4131 = stablehlo.reshape %v4130 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4132 = stablehlo.reshape %v4131 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4133 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4134 = stablehlo.pad %v4132, %v4133, low = [0, 0, 64], high = [0, 0, 640], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v4135 = stablehlo.reshape %v4134 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4136 = stablehlo.add %v4102, %v4135 : tensor<32x151296xf32>
    %v4137 = stablehlo.reshape %v4059 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4138 = stablehlo.slice %v4137 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4139 = stablehlo.reshape %v4138 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4140 = stablehlo.reshape %v4064 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4141 = stablehlo.slice %v4140 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4142 = stablehlo.reshape %v4141 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4143 = stablehlo.reshape %v4069 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4144 = stablehlo.slice %v4143 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4145 = stablehlo.reshape %v4144 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4146 = stablehlo.reshape %v4142 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4147 = stablehlo.transpose %v4146, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v4148 = stablehlo.reshape %v4147 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4149 = stablehlo.reshape %v4139 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4150 = stablehlo.reshape %v4148 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4151 = stablehlo.dot_general %v4149, %v4150, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v4152 = stablehlo.reshape %v4151 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4153 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v4154 = stablehlo.multiply %v4152, %v4153 : tensor<32x38809xf32>
    %v4155 = stablehlo.reshape %v4154 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4156 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4157 = stablehlo.exponential %v4155 : tensor<32x197x197xf32>
    %v4158 = stablehlo.reduce(%v4157 init: %v4156) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4159 = stablehlo.broadcast_in_dim %v4158, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4160 = stablehlo.divide %v4157, %v4159 : tensor<32x197x197xf32>
    %v4161 = stablehlo.reshape %v4160 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4162 = stablehlo.reshape %v4161 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4163 = stablehlo.reshape %v4145 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4164 = stablehlo.dot_general %v4162, %v4163, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4165 = stablehlo.reshape %v4164 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4166 = stablehlo.reshape %v4165 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4167 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4168 = stablehlo.pad %v4166, %v4167, low = [0, 0, 128], high = [0, 0, 576], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v4169 = stablehlo.reshape %v4168 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4170 = stablehlo.add %v4136, %v4169 : tensor<32x151296xf32>
    %v4171 = stablehlo.reshape %v4059 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4172 = stablehlo.slice %v4171 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4173 = stablehlo.reshape %v4172 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4174 = stablehlo.reshape %v4064 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4175 = stablehlo.slice %v4174 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4176 = stablehlo.reshape %v4175 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4177 = stablehlo.reshape %v4069 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4178 = stablehlo.slice %v4177 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4179 = stablehlo.reshape %v4178 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4180 = stablehlo.reshape %v4176 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4181 = stablehlo.transpose %v4180, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v4182 = stablehlo.reshape %v4181 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4183 = stablehlo.reshape %v4173 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4184 = stablehlo.reshape %v4182 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4185 = stablehlo.dot_general %v4183, %v4184, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v4186 = stablehlo.reshape %v4185 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4187 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v4188 = stablehlo.multiply %v4186, %v4187 : tensor<32x38809xf32>
    %v4189 = stablehlo.reshape %v4188 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4190 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4191 = stablehlo.exponential %v4189 : tensor<32x197x197xf32>
    %v4192 = stablehlo.reduce(%v4191 init: %v4190) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4193 = stablehlo.broadcast_in_dim %v4192, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4194 = stablehlo.divide %v4191, %v4193 : tensor<32x197x197xf32>
    %v4195 = stablehlo.reshape %v4194 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4196 = stablehlo.reshape %v4195 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4197 = stablehlo.reshape %v4179 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4198 = stablehlo.dot_general %v4196, %v4197, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4199 = stablehlo.reshape %v4198 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4200 = stablehlo.reshape %v4199 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4201 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4202 = stablehlo.pad %v4200, %v4201, low = [0, 0, 192], high = [0, 0, 512], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v4203 = stablehlo.reshape %v4202 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4204 = stablehlo.add %v4170, %v4203 : tensor<32x151296xf32>
    %v4205 = stablehlo.reshape %v4059 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4206 = stablehlo.slice %v4205 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4207 = stablehlo.reshape %v4206 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4208 = stablehlo.reshape %v4064 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4209 = stablehlo.slice %v4208 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4210 = stablehlo.reshape %v4209 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4211 = stablehlo.reshape %v4069 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4212 = stablehlo.slice %v4211 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4213 = stablehlo.reshape %v4212 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4214 = stablehlo.reshape %v4210 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4215 = stablehlo.transpose %v4214, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v4216 = stablehlo.reshape %v4215 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4217 = stablehlo.reshape %v4207 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4218 = stablehlo.reshape %v4216 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4219 = stablehlo.dot_general %v4217, %v4218, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v4220 = stablehlo.reshape %v4219 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4221 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v4222 = stablehlo.multiply %v4220, %v4221 : tensor<32x38809xf32>
    %v4223 = stablehlo.reshape %v4222 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4224 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4225 = stablehlo.exponential %v4223 : tensor<32x197x197xf32>
    %v4226 = stablehlo.reduce(%v4225 init: %v4224) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4227 = stablehlo.broadcast_in_dim %v4226, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4228 = stablehlo.divide %v4225, %v4227 : tensor<32x197x197xf32>
    %v4229 = stablehlo.reshape %v4228 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4230 = stablehlo.reshape %v4229 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4231 = stablehlo.reshape %v4213 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4232 = stablehlo.dot_general %v4230, %v4231, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4233 = stablehlo.reshape %v4232 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4234 = stablehlo.reshape %v4233 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4235 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4236 = stablehlo.pad %v4234, %v4235, low = [0, 0, 256], high = [0, 0, 448], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v4237 = stablehlo.reshape %v4236 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4238 = stablehlo.add %v4204, %v4237 : tensor<32x151296xf32>
    %v4239 = stablehlo.reshape %v4059 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4240 = stablehlo.slice %v4239 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4241 = stablehlo.reshape %v4240 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4242 = stablehlo.reshape %v4064 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4243 = stablehlo.slice %v4242 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4244 = stablehlo.reshape %v4243 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4245 = stablehlo.reshape %v4069 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4246 = stablehlo.slice %v4245 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4247 = stablehlo.reshape %v4246 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4248 = stablehlo.reshape %v4244 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4249 = stablehlo.transpose %v4248, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v4250 = stablehlo.reshape %v4249 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4251 = stablehlo.reshape %v4241 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4252 = stablehlo.reshape %v4250 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4253 = stablehlo.dot_general %v4251, %v4252, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v4254 = stablehlo.reshape %v4253 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4255 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v4256 = stablehlo.multiply %v4254, %v4255 : tensor<32x38809xf32>
    %v4257 = stablehlo.reshape %v4256 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4258 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4259 = stablehlo.exponential %v4257 : tensor<32x197x197xf32>
    %v4260 = stablehlo.reduce(%v4259 init: %v4258) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4261 = stablehlo.broadcast_in_dim %v4260, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4262 = stablehlo.divide %v4259, %v4261 : tensor<32x197x197xf32>
    %v4263 = stablehlo.reshape %v4262 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4264 = stablehlo.reshape %v4263 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4265 = stablehlo.reshape %v4247 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4266 = stablehlo.dot_general %v4264, %v4265, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4267 = stablehlo.reshape %v4266 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4268 = stablehlo.reshape %v4267 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4269 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4270 = stablehlo.pad %v4268, %v4269, low = [0, 0, 320], high = [0, 0, 384], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v4271 = stablehlo.reshape %v4270 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4272 = stablehlo.add %v4238, %v4271 : tensor<32x151296xf32>
    %v4273 = stablehlo.reshape %v4059 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4274 = stablehlo.slice %v4273 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4275 = stablehlo.reshape %v4274 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4276 = stablehlo.reshape %v4064 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4277 = stablehlo.slice %v4276 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4278 = stablehlo.reshape %v4277 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4279 = stablehlo.reshape %v4069 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4280 = stablehlo.slice %v4279 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4281 = stablehlo.reshape %v4280 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4282 = stablehlo.reshape %v4278 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4283 = stablehlo.transpose %v4282, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v4284 = stablehlo.reshape %v4283 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4285 = stablehlo.reshape %v4275 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4286 = stablehlo.reshape %v4284 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4287 = stablehlo.dot_general %v4285, %v4286, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v4288 = stablehlo.reshape %v4287 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4289 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v4290 = stablehlo.multiply %v4288, %v4289 : tensor<32x38809xf32>
    %v4291 = stablehlo.reshape %v4290 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4292 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4293 = stablehlo.exponential %v4291 : tensor<32x197x197xf32>
    %v4294 = stablehlo.reduce(%v4293 init: %v4292) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4295 = stablehlo.broadcast_in_dim %v4294, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4296 = stablehlo.divide %v4293, %v4295 : tensor<32x197x197xf32>
    %v4297 = stablehlo.reshape %v4296 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4298 = stablehlo.reshape %v4297 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4299 = stablehlo.reshape %v4281 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4300 = stablehlo.dot_general %v4298, %v4299, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4301 = stablehlo.reshape %v4300 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4302 = stablehlo.reshape %v4301 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4303 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4304 = stablehlo.pad %v4302, %v4303, low = [0, 0, 384], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v4305 = stablehlo.reshape %v4304 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4306 = stablehlo.add %v4272, %v4305 : tensor<32x151296xf32>
    %v4307 = stablehlo.reshape %v4059 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4308 = stablehlo.slice %v4307 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4309 = stablehlo.reshape %v4308 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4310 = stablehlo.reshape %v4064 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4311 = stablehlo.slice %v4310 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4312 = stablehlo.reshape %v4311 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4313 = stablehlo.reshape %v4069 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4314 = stablehlo.slice %v4313 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4315 = stablehlo.reshape %v4314 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4316 = stablehlo.reshape %v4312 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4317 = stablehlo.transpose %v4316, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v4318 = stablehlo.reshape %v4317 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4319 = stablehlo.reshape %v4309 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4320 = stablehlo.reshape %v4318 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4321 = stablehlo.dot_general %v4319, %v4320, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v4322 = stablehlo.reshape %v4321 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4323 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v4324 = stablehlo.multiply %v4322, %v4323 : tensor<32x38809xf32>
    %v4325 = stablehlo.reshape %v4324 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4326 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4327 = stablehlo.exponential %v4325 : tensor<32x197x197xf32>
    %v4328 = stablehlo.reduce(%v4327 init: %v4326) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4329 = stablehlo.broadcast_in_dim %v4328, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4330 = stablehlo.divide %v4327, %v4329 : tensor<32x197x197xf32>
    %v4331 = stablehlo.reshape %v4330 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4332 = stablehlo.reshape %v4331 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4333 = stablehlo.reshape %v4315 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4334 = stablehlo.dot_general %v4332, %v4333, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4335 = stablehlo.reshape %v4334 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4336 = stablehlo.reshape %v4335 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4337 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4338 = stablehlo.pad %v4336, %v4337, low = [0, 0, 448], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v4339 = stablehlo.reshape %v4338 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4340 = stablehlo.add %v4306, %v4339 : tensor<32x151296xf32>
    %v4341 = stablehlo.reshape %v4059 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4342 = stablehlo.slice %v4341 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4343 = stablehlo.reshape %v4342 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4344 = stablehlo.reshape %v4064 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4345 = stablehlo.slice %v4344 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4346 = stablehlo.reshape %v4345 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4347 = stablehlo.reshape %v4069 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4348 = stablehlo.slice %v4347 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4349 = stablehlo.reshape %v4348 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4350 = stablehlo.reshape %v4346 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4351 = stablehlo.transpose %v4350, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v4352 = stablehlo.reshape %v4351 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4353 = stablehlo.reshape %v4343 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4354 = stablehlo.reshape %v4352 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4355 = stablehlo.dot_general %v4353, %v4354, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v4356 = stablehlo.reshape %v4355 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4357 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v4358 = stablehlo.multiply %v4356, %v4357 : tensor<32x38809xf32>
    %v4359 = stablehlo.reshape %v4358 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4360 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4361 = stablehlo.exponential %v4359 : tensor<32x197x197xf32>
    %v4362 = stablehlo.reduce(%v4361 init: %v4360) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4363 = stablehlo.broadcast_in_dim %v4362, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4364 = stablehlo.divide %v4361, %v4363 : tensor<32x197x197xf32>
    %v4365 = stablehlo.reshape %v4364 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4366 = stablehlo.reshape %v4365 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4367 = stablehlo.reshape %v4349 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4368 = stablehlo.dot_general %v4366, %v4367, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4369 = stablehlo.reshape %v4368 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4370 = stablehlo.reshape %v4369 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4371 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4372 = stablehlo.pad %v4370, %v4371, low = [0, 0, 512], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v4373 = stablehlo.reshape %v4372 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4374 = stablehlo.add %v4340, %v4373 : tensor<32x151296xf32>
    %v4375 = stablehlo.reshape %v4059 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4376 = stablehlo.slice %v4375 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4377 = stablehlo.reshape %v4376 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4378 = stablehlo.reshape %v4064 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4379 = stablehlo.slice %v4378 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4380 = stablehlo.reshape %v4379 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4381 = stablehlo.reshape %v4069 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4382 = stablehlo.slice %v4381 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4383 = stablehlo.reshape %v4382 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4384 = stablehlo.reshape %v4380 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4385 = stablehlo.transpose %v4384, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v4386 = stablehlo.reshape %v4385 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4387 = stablehlo.reshape %v4377 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4388 = stablehlo.reshape %v4386 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4389 = stablehlo.dot_general %v4387, %v4388, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v4390 = stablehlo.reshape %v4389 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4391 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v4392 = stablehlo.multiply %v4390, %v4391 : tensor<32x38809xf32>
    %v4393 = stablehlo.reshape %v4392 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4394 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4395 = stablehlo.exponential %v4393 : tensor<32x197x197xf32>
    %v4396 = stablehlo.reduce(%v4395 init: %v4394) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4397 = stablehlo.broadcast_in_dim %v4396, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4398 = stablehlo.divide %v4395, %v4397 : tensor<32x197x197xf32>
    %v4399 = stablehlo.reshape %v4398 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4400 = stablehlo.reshape %v4399 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4401 = stablehlo.reshape %v4383 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4402 = stablehlo.dot_general %v4400, %v4401, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4403 = stablehlo.reshape %v4402 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4404 = stablehlo.reshape %v4403 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4405 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4406 = stablehlo.pad %v4404, %v4405, low = [0, 0, 576], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v4407 = stablehlo.reshape %v4406 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4408 = stablehlo.add %v4374, %v4407 : tensor<32x151296xf32>
    %v4409 = stablehlo.reshape %v4059 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4410 = stablehlo.slice %v4409 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4411 = stablehlo.reshape %v4410 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4412 = stablehlo.reshape %v4064 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4413 = stablehlo.slice %v4412 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4414 = stablehlo.reshape %v4413 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4415 = stablehlo.reshape %v4069 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4416 = stablehlo.slice %v4415 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4417 = stablehlo.reshape %v4416 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4418 = stablehlo.reshape %v4414 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4419 = stablehlo.transpose %v4418, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v4420 = stablehlo.reshape %v4419 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4421 = stablehlo.reshape %v4411 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4422 = stablehlo.reshape %v4420 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4423 = stablehlo.dot_general %v4421, %v4422, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v4424 = stablehlo.reshape %v4423 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4425 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v4426 = stablehlo.multiply %v4424, %v4425 : tensor<32x38809xf32>
    %v4427 = stablehlo.reshape %v4426 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4428 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4429 = stablehlo.exponential %v4427 : tensor<32x197x197xf32>
    %v4430 = stablehlo.reduce(%v4429 init: %v4428) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4431 = stablehlo.broadcast_in_dim %v4430, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4432 = stablehlo.divide %v4429, %v4431 : tensor<32x197x197xf32>
    %v4433 = stablehlo.reshape %v4432 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4434 = stablehlo.reshape %v4433 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4435 = stablehlo.reshape %v4417 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4436 = stablehlo.dot_general %v4434, %v4435, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4437 = stablehlo.reshape %v4436 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4438 = stablehlo.reshape %v4437 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4439 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4440 = stablehlo.pad %v4438, %v4439, low = [0, 0, 640], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v4441 = stablehlo.reshape %v4440 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4442 = stablehlo.add %v4408, %v4441 : tensor<32x151296xf32>
    %v4443 = stablehlo.reshape %v4059 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4444 = stablehlo.slice %v4443 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4445 = stablehlo.reshape %v4444 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4446 = stablehlo.reshape %v4064 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4447 = stablehlo.slice %v4446 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4448 = stablehlo.reshape %v4447 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4449 = stablehlo.reshape %v4069 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4450 = stablehlo.slice %v4449 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4451 = stablehlo.reshape %v4450 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4452 = stablehlo.reshape %v4448 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4453 = stablehlo.transpose %v4452, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v4454 = stablehlo.reshape %v4453 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4455 = stablehlo.reshape %v4445 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4456 = stablehlo.reshape %v4454 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4457 = stablehlo.dot_general %v4455, %v4456, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v4458 = stablehlo.reshape %v4457 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4459 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v4460 = stablehlo.multiply %v4458, %v4459 : tensor<32x38809xf32>
    %v4461 = stablehlo.reshape %v4460 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4462 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4463 = stablehlo.exponential %v4461 : tensor<32x197x197xf32>
    %v4464 = stablehlo.reduce(%v4463 init: %v4462) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4465 = stablehlo.broadcast_in_dim %v4464, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4466 = stablehlo.divide %v4463, %v4465 : tensor<32x197x197xf32>
    %v4467 = stablehlo.reshape %v4466 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4468 = stablehlo.reshape %v4467 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4469 = stablehlo.reshape %v4451 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4470 = stablehlo.dot_general %v4468, %v4469, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4471 = stablehlo.reshape %v4470 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4472 = stablehlo.reshape %v4471 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4473 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4474 = stablehlo.pad %v4472, %v4473, low = [0, 0, 704], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v4475 = stablehlo.reshape %v4474 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4476 = stablehlo.add %v4442, %v4475 : tensor<32x151296xf32>
    %v4477 = stablehlo.reshape %v4476 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4478 = stablehlo.dot_general %v4477, %b8_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v4479 = stablehlo.broadcast_in_dim %b8_bo, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v4480 = stablehlo.add %v4478, %v4479 : tensor<32x197x768xf32>
    %v4481 = stablehlo.reshape %v4480 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4482 = stablehlo.add %v4026, %v4481 : tensor<32x151296xf32>
    %v4483 = stablehlo.reshape %v4482 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4484 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4485 = stablehlo.constant dense<768.0> : tensor<32x197x768xf32>
    %v4486 = stablehlo.constant dense<1.0e-5> : tensor<32x197x768xf32>
    %v4487 = stablehlo.reduce(%v4483 init: %v4484) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4488 = stablehlo.broadcast_in_dim %v4487, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v4489 = stablehlo.divide %v4488, %v4485 : tensor<32x197x768xf32>
    %v4490 = stablehlo.subtract %v4483, %v4489 : tensor<32x197x768xf32>
    %v4491 = stablehlo.multiply %v4490, %v4490 : tensor<32x197x768xf32>
    %v4492 = stablehlo.reduce(%v4491 init: %v4484) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4493 = stablehlo.broadcast_in_dim %v4492, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v4494 = stablehlo.divide %v4493, %v4485 : tensor<32x197x768xf32>
    %v4495 = stablehlo.add %v4494, %v4486 : tensor<32x197x768xf32>
    %v4496 = stablehlo.rsqrt %v4495 : tensor<32x197x768xf32>
    %v4497 = stablehlo.multiply %v4490, %v4496 : tensor<32x197x768xf32>
    %v4498 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v4499 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v4500 = stablehlo.multiply %v4497, %v4498 : tensor<32x197x768xf32>
    %v4501 = stablehlo.add %v4500, %v4499 : tensor<32x197x768xf32>
    %v4502 = stablehlo.reshape %v4501 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4503 = stablehlo.reshape %v4502 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4504 = stablehlo.broadcast_in_dim %b8_g2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v4505 = stablehlo.multiply %v4503, %v4504 : tensor<32x197x768xf32>
    %v4506 = stablehlo.reshape %v4505 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4507 = stablehlo.reshape %v4506 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4508 = stablehlo.broadcast_in_dim %b8_bt2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v4509 = stablehlo.add %v4507, %v4508 : tensor<32x197x768xf32>
    %v4510 = stablehlo.reshape %v4509 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4511 = stablehlo.reshape %v4510 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4512 = stablehlo.dot_general %v4511, %b8_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x3072xf32>) -> tensor<32x197x3072xf32>
    %v4513 = stablehlo.broadcast_in_dim %b8_bfc1, dims = [2] : (tensor<3072xf32>) -> tensor<32x197x3072xf32>
    %v4514 = stablehlo.add %v4512, %v4513 : tensor<32x197x3072xf32>
    %v4515 = stablehlo.reshape %v4514 : (tensor<32x197x3072xf32>) -> tensor<32x605184xf32>
    %v4516 = stablehlo.constant dense<0.5> : tensor<32x605184xf32>
    %v4517 = stablehlo.multiply %v4516, %v4515 : tensor<32x605184xf32>
    %v4518 = stablehlo.negate %v4515 : tensor<32x605184xf32>
    %v4519 = stablehlo.constant dense<0.7071067811865476> : tensor<32x605184xf32>
    %v4520 = stablehlo.multiply %v4518, %v4519 : tensor<32x605184xf32>
    %v4521 = chlo.erfc %v4520 : tensor<32x605184xf32> -> tensor<32x605184xf32>
    %v4522 = stablehlo.multiply %v4517, %v4521 : tensor<32x605184xf32>
    %v4523 = stablehlo.reshape %v4522 : (tensor<32x605184xf32>) -> tensor<32x197x3072xf32>
    %v4524 = stablehlo.dot_general %v4523, %b8_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x3072xf32>, tensor<3072x768xf32>) -> tensor<32x197x768xf32>
    %v4525 = stablehlo.broadcast_in_dim %b8_bfc2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v4526 = stablehlo.add %v4524, %v4525 : tensor<32x197x768xf32>
    %v4527 = stablehlo.reshape %v4526 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4528 = stablehlo.add %v4482, %v4527 : tensor<32x151296xf32>
    %v4529 = stablehlo.reshape %v4528 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4530 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4531 = stablehlo.constant dense<768.0> : tensor<32x197x768xf32>
    %v4532 = stablehlo.constant dense<1.0e-5> : tensor<32x197x768xf32>
    %v4533 = stablehlo.reduce(%v4529 init: %v4530) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4534 = stablehlo.broadcast_in_dim %v4533, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v4535 = stablehlo.divide %v4534, %v4531 : tensor<32x197x768xf32>
    %v4536 = stablehlo.subtract %v4529, %v4535 : tensor<32x197x768xf32>
    %v4537 = stablehlo.multiply %v4536, %v4536 : tensor<32x197x768xf32>
    %v4538 = stablehlo.reduce(%v4537 init: %v4530) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4539 = stablehlo.broadcast_in_dim %v4538, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v4540 = stablehlo.divide %v4539, %v4531 : tensor<32x197x768xf32>
    %v4541 = stablehlo.add %v4540, %v4532 : tensor<32x197x768xf32>
    %v4542 = stablehlo.rsqrt %v4541 : tensor<32x197x768xf32>
    %v4543 = stablehlo.multiply %v4536, %v4542 : tensor<32x197x768xf32>
    %v4544 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v4545 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v4546 = stablehlo.multiply %v4543, %v4544 : tensor<32x197x768xf32>
    %v4547 = stablehlo.add %v4546, %v4545 : tensor<32x197x768xf32>
    %v4548 = stablehlo.reshape %v4547 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4549 = stablehlo.reshape %v4548 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4550 = stablehlo.broadcast_in_dim %b9_g1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v4551 = stablehlo.multiply %v4549, %v4550 : tensor<32x197x768xf32>
    %v4552 = stablehlo.reshape %v4551 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4553 = stablehlo.reshape %v4552 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4554 = stablehlo.broadcast_in_dim %b9_bt1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v4555 = stablehlo.add %v4553, %v4554 : tensor<32x197x768xf32>
    %v4556 = stablehlo.reshape %v4555 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4557 = stablehlo.reshape %v4556 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4558 = stablehlo.dot_general %v4557, %b9_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v4559 = stablehlo.broadcast_in_dim %b9_bq, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v4560 = stablehlo.add %v4558, %v4559 : tensor<32x197x768xf32>
    %v4561 = stablehlo.reshape %v4560 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4562 = stablehlo.reshape %v4556 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4563 = stablehlo.dot_general %v4562, %b9_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v4564 = stablehlo.broadcast_in_dim %b9_bk, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v4565 = stablehlo.add %v4563, %v4564 : tensor<32x197x768xf32>
    %v4566 = stablehlo.reshape %v4565 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4567 = stablehlo.reshape %v4556 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4568 = stablehlo.dot_general %v4567, %b9_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v4569 = stablehlo.broadcast_in_dim %b9_bv, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v4570 = stablehlo.add %v4568, %v4569 : tensor<32x197x768xf32>
    %v4571 = stablehlo.reshape %v4570 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4572 = stablehlo.reshape %v4561 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4573 = stablehlo.slice %v4572 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4574 = stablehlo.reshape %v4573 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4575 = stablehlo.reshape %v4566 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4576 = stablehlo.slice %v4575 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4577 = stablehlo.reshape %v4576 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4578 = stablehlo.reshape %v4571 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4579 = stablehlo.slice %v4578 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4580 = stablehlo.reshape %v4579 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4581 = stablehlo.reshape %v4577 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4582 = stablehlo.transpose %v4581, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v4583 = stablehlo.reshape %v4582 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4584 = stablehlo.reshape %v4574 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4585 = stablehlo.reshape %v4583 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4586 = stablehlo.dot_general %v4584, %v4585, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v4587 = stablehlo.reshape %v4586 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4588 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v4589 = stablehlo.multiply %v4587, %v4588 : tensor<32x38809xf32>
    %v4590 = stablehlo.reshape %v4589 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4591 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4592 = stablehlo.exponential %v4590 : tensor<32x197x197xf32>
    %v4593 = stablehlo.reduce(%v4592 init: %v4591) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4594 = stablehlo.broadcast_in_dim %v4593, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4595 = stablehlo.divide %v4592, %v4594 : tensor<32x197x197xf32>
    %v4596 = stablehlo.reshape %v4595 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4597 = stablehlo.reshape %v4596 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4598 = stablehlo.reshape %v4580 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4599 = stablehlo.dot_general %v4597, %v4598, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4600 = stablehlo.reshape %v4599 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4601 = stablehlo.reshape %v4600 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4602 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4603 = stablehlo.pad %v4601, %v4602, low = [0, 0, 0], high = [0, 0, 704], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v4604 = stablehlo.reshape %v4603 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4605 = stablehlo.reshape %v4561 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4606 = stablehlo.slice %v4605 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4607 = stablehlo.reshape %v4606 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4608 = stablehlo.reshape %v4566 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4609 = stablehlo.slice %v4608 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4610 = stablehlo.reshape %v4609 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4611 = stablehlo.reshape %v4571 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4612 = stablehlo.slice %v4611 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4613 = stablehlo.reshape %v4612 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4614 = stablehlo.reshape %v4610 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4615 = stablehlo.transpose %v4614, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v4616 = stablehlo.reshape %v4615 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4617 = stablehlo.reshape %v4607 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4618 = stablehlo.reshape %v4616 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4619 = stablehlo.dot_general %v4617, %v4618, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v4620 = stablehlo.reshape %v4619 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4621 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v4622 = stablehlo.multiply %v4620, %v4621 : tensor<32x38809xf32>
    %v4623 = stablehlo.reshape %v4622 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4624 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4625 = stablehlo.exponential %v4623 : tensor<32x197x197xf32>
    %v4626 = stablehlo.reduce(%v4625 init: %v4624) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4627 = stablehlo.broadcast_in_dim %v4626, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4628 = stablehlo.divide %v4625, %v4627 : tensor<32x197x197xf32>
    %v4629 = stablehlo.reshape %v4628 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4630 = stablehlo.reshape %v4629 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4631 = stablehlo.reshape %v4613 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4632 = stablehlo.dot_general %v4630, %v4631, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4633 = stablehlo.reshape %v4632 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4634 = stablehlo.reshape %v4633 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4635 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4636 = stablehlo.pad %v4634, %v4635, low = [0, 0, 64], high = [0, 0, 640], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v4637 = stablehlo.reshape %v4636 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4638 = stablehlo.add %v4604, %v4637 : tensor<32x151296xf32>
    %v4639 = stablehlo.reshape %v4561 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4640 = stablehlo.slice %v4639 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4641 = stablehlo.reshape %v4640 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4642 = stablehlo.reshape %v4566 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4643 = stablehlo.slice %v4642 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4644 = stablehlo.reshape %v4643 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4645 = stablehlo.reshape %v4571 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4646 = stablehlo.slice %v4645 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4647 = stablehlo.reshape %v4646 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4648 = stablehlo.reshape %v4644 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4649 = stablehlo.transpose %v4648, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v4650 = stablehlo.reshape %v4649 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4651 = stablehlo.reshape %v4641 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4652 = stablehlo.reshape %v4650 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4653 = stablehlo.dot_general %v4651, %v4652, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v4654 = stablehlo.reshape %v4653 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4655 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v4656 = stablehlo.multiply %v4654, %v4655 : tensor<32x38809xf32>
    %v4657 = stablehlo.reshape %v4656 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4658 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4659 = stablehlo.exponential %v4657 : tensor<32x197x197xf32>
    %v4660 = stablehlo.reduce(%v4659 init: %v4658) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4661 = stablehlo.broadcast_in_dim %v4660, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4662 = stablehlo.divide %v4659, %v4661 : tensor<32x197x197xf32>
    %v4663 = stablehlo.reshape %v4662 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4664 = stablehlo.reshape %v4663 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4665 = stablehlo.reshape %v4647 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4666 = stablehlo.dot_general %v4664, %v4665, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4667 = stablehlo.reshape %v4666 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4668 = stablehlo.reshape %v4667 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4669 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4670 = stablehlo.pad %v4668, %v4669, low = [0, 0, 128], high = [0, 0, 576], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v4671 = stablehlo.reshape %v4670 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4672 = stablehlo.add %v4638, %v4671 : tensor<32x151296xf32>
    %v4673 = stablehlo.reshape %v4561 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4674 = stablehlo.slice %v4673 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4675 = stablehlo.reshape %v4674 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4676 = stablehlo.reshape %v4566 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4677 = stablehlo.slice %v4676 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4678 = stablehlo.reshape %v4677 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4679 = stablehlo.reshape %v4571 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4680 = stablehlo.slice %v4679 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4681 = stablehlo.reshape %v4680 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4682 = stablehlo.reshape %v4678 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4683 = stablehlo.transpose %v4682, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v4684 = stablehlo.reshape %v4683 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4685 = stablehlo.reshape %v4675 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4686 = stablehlo.reshape %v4684 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4687 = stablehlo.dot_general %v4685, %v4686, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v4688 = stablehlo.reshape %v4687 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4689 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v4690 = stablehlo.multiply %v4688, %v4689 : tensor<32x38809xf32>
    %v4691 = stablehlo.reshape %v4690 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4692 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4693 = stablehlo.exponential %v4691 : tensor<32x197x197xf32>
    %v4694 = stablehlo.reduce(%v4693 init: %v4692) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4695 = stablehlo.broadcast_in_dim %v4694, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4696 = stablehlo.divide %v4693, %v4695 : tensor<32x197x197xf32>
    %v4697 = stablehlo.reshape %v4696 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4698 = stablehlo.reshape %v4697 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4699 = stablehlo.reshape %v4681 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4700 = stablehlo.dot_general %v4698, %v4699, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4701 = stablehlo.reshape %v4700 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4702 = stablehlo.reshape %v4701 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4703 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4704 = stablehlo.pad %v4702, %v4703, low = [0, 0, 192], high = [0, 0, 512], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v4705 = stablehlo.reshape %v4704 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4706 = stablehlo.add %v4672, %v4705 : tensor<32x151296xf32>
    %v4707 = stablehlo.reshape %v4561 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4708 = stablehlo.slice %v4707 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4709 = stablehlo.reshape %v4708 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4710 = stablehlo.reshape %v4566 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4711 = stablehlo.slice %v4710 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4712 = stablehlo.reshape %v4711 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4713 = stablehlo.reshape %v4571 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4714 = stablehlo.slice %v4713 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4715 = stablehlo.reshape %v4714 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4716 = stablehlo.reshape %v4712 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4717 = stablehlo.transpose %v4716, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v4718 = stablehlo.reshape %v4717 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4719 = stablehlo.reshape %v4709 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4720 = stablehlo.reshape %v4718 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4721 = stablehlo.dot_general %v4719, %v4720, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v4722 = stablehlo.reshape %v4721 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4723 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v4724 = stablehlo.multiply %v4722, %v4723 : tensor<32x38809xf32>
    %v4725 = stablehlo.reshape %v4724 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4726 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4727 = stablehlo.exponential %v4725 : tensor<32x197x197xf32>
    %v4728 = stablehlo.reduce(%v4727 init: %v4726) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4729 = stablehlo.broadcast_in_dim %v4728, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4730 = stablehlo.divide %v4727, %v4729 : tensor<32x197x197xf32>
    %v4731 = stablehlo.reshape %v4730 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4732 = stablehlo.reshape %v4731 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4733 = stablehlo.reshape %v4715 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4734 = stablehlo.dot_general %v4732, %v4733, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4735 = stablehlo.reshape %v4734 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4736 = stablehlo.reshape %v4735 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4737 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4738 = stablehlo.pad %v4736, %v4737, low = [0, 0, 256], high = [0, 0, 448], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v4739 = stablehlo.reshape %v4738 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4740 = stablehlo.add %v4706, %v4739 : tensor<32x151296xf32>
    %v4741 = stablehlo.reshape %v4561 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4742 = stablehlo.slice %v4741 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4743 = stablehlo.reshape %v4742 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4744 = stablehlo.reshape %v4566 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4745 = stablehlo.slice %v4744 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4746 = stablehlo.reshape %v4745 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4747 = stablehlo.reshape %v4571 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4748 = stablehlo.slice %v4747 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4749 = stablehlo.reshape %v4748 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4750 = stablehlo.reshape %v4746 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4751 = stablehlo.transpose %v4750, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v4752 = stablehlo.reshape %v4751 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4753 = stablehlo.reshape %v4743 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4754 = stablehlo.reshape %v4752 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4755 = stablehlo.dot_general %v4753, %v4754, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v4756 = stablehlo.reshape %v4755 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4757 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v4758 = stablehlo.multiply %v4756, %v4757 : tensor<32x38809xf32>
    %v4759 = stablehlo.reshape %v4758 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4760 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4761 = stablehlo.exponential %v4759 : tensor<32x197x197xf32>
    %v4762 = stablehlo.reduce(%v4761 init: %v4760) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4763 = stablehlo.broadcast_in_dim %v4762, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4764 = stablehlo.divide %v4761, %v4763 : tensor<32x197x197xf32>
    %v4765 = stablehlo.reshape %v4764 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4766 = stablehlo.reshape %v4765 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4767 = stablehlo.reshape %v4749 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4768 = stablehlo.dot_general %v4766, %v4767, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4769 = stablehlo.reshape %v4768 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4770 = stablehlo.reshape %v4769 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4771 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4772 = stablehlo.pad %v4770, %v4771, low = [0, 0, 320], high = [0, 0, 384], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v4773 = stablehlo.reshape %v4772 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4774 = stablehlo.add %v4740, %v4773 : tensor<32x151296xf32>
    %v4775 = stablehlo.reshape %v4561 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4776 = stablehlo.slice %v4775 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4777 = stablehlo.reshape %v4776 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4778 = stablehlo.reshape %v4566 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4779 = stablehlo.slice %v4778 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4780 = stablehlo.reshape %v4779 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4781 = stablehlo.reshape %v4571 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4782 = stablehlo.slice %v4781 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4783 = stablehlo.reshape %v4782 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4784 = stablehlo.reshape %v4780 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4785 = stablehlo.transpose %v4784, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v4786 = stablehlo.reshape %v4785 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4787 = stablehlo.reshape %v4777 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4788 = stablehlo.reshape %v4786 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4789 = stablehlo.dot_general %v4787, %v4788, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v4790 = stablehlo.reshape %v4789 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4791 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v4792 = stablehlo.multiply %v4790, %v4791 : tensor<32x38809xf32>
    %v4793 = stablehlo.reshape %v4792 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4794 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4795 = stablehlo.exponential %v4793 : tensor<32x197x197xf32>
    %v4796 = stablehlo.reduce(%v4795 init: %v4794) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4797 = stablehlo.broadcast_in_dim %v4796, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4798 = stablehlo.divide %v4795, %v4797 : tensor<32x197x197xf32>
    %v4799 = stablehlo.reshape %v4798 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4800 = stablehlo.reshape %v4799 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4801 = stablehlo.reshape %v4783 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4802 = stablehlo.dot_general %v4800, %v4801, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4803 = stablehlo.reshape %v4802 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4804 = stablehlo.reshape %v4803 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4805 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4806 = stablehlo.pad %v4804, %v4805, low = [0, 0, 384], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v4807 = stablehlo.reshape %v4806 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4808 = stablehlo.add %v4774, %v4807 : tensor<32x151296xf32>
    %v4809 = stablehlo.reshape %v4561 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4810 = stablehlo.slice %v4809 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4811 = stablehlo.reshape %v4810 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4812 = stablehlo.reshape %v4566 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4813 = stablehlo.slice %v4812 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4814 = stablehlo.reshape %v4813 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4815 = stablehlo.reshape %v4571 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4816 = stablehlo.slice %v4815 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4817 = stablehlo.reshape %v4816 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4818 = stablehlo.reshape %v4814 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4819 = stablehlo.transpose %v4818, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v4820 = stablehlo.reshape %v4819 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4821 = stablehlo.reshape %v4811 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4822 = stablehlo.reshape %v4820 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4823 = stablehlo.dot_general %v4821, %v4822, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v4824 = stablehlo.reshape %v4823 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4825 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v4826 = stablehlo.multiply %v4824, %v4825 : tensor<32x38809xf32>
    %v4827 = stablehlo.reshape %v4826 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4828 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4829 = stablehlo.exponential %v4827 : tensor<32x197x197xf32>
    %v4830 = stablehlo.reduce(%v4829 init: %v4828) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4831 = stablehlo.broadcast_in_dim %v4830, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4832 = stablehlo.divide %v4829, %v4831 : tensor<32x197x197xf32>
    %v4833 = stablehlo.reshape %v4832 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4834 = stablehlo.reshape %v4833 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4835 = stablehlo.reshape %v4817 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4836 = stablehlo.dot_general %v4834, %v4835, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4837 = stablehlo.reshape %v4836 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4838 = stablehlo.reshape %v4837 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4839 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4840 = stablehlo.pad %v4838, %v4839, low = [0, 0, 448], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v4841 = stablehlo.reshape %v4840 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4842 = stablehlo.add %v4808, %v4841 : tensor<32x151296xf32>
    %v4843 = stablehlo.reshape %v4561 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4844 = stablehlo.slice %v4843 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4845 = stablehlo.reshape %v4844 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4846 = stablehlo.reshape %v4566 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4847 = stablehlo.slice %v4846 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4848 = stablehlo.reshape %v4847 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4849 = stablehlo.reshape %v4571 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4850 = stablehlo.slice %v4849 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4851 = stablehlo.reshape %v4850 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4852 = stablehlo.reshape %v4848 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4853 = stablehlo.transpose %v4852, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v4854 = stablehlo.reshape %v4853 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4855 = stablehlo.reshape %v4845 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4856 = stablehlo.reshape %v4854 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4857 = stablehlo.dot_general %v4855, %v4856, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v4858 = stablehlo.reshape %v4857 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4859 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v4860 = stablehlo.multiply %v4858, %v4859 : tensor<32x38809xf32>
    %v4861 = stablehlo.reshape %v4860 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4862 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4863 = stablehlo.exponential %v4861 : tensor<32x197x197xf32>
    %v4864 = stablehlo.reduce(%v4863 init: %v4862) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4865 = stablehlo.broadcast_in_dim %v4864, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4866 = stablehlo.divide %v4863, %v4865 : tensor<32x197x197xf32>
    %v4867 = stablehlo.reshape %v4866 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4868 = stablehlo.reshape %v4867 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4869 = stablehlo.reshape %v4851 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4870 = stablehlo.dot_general %v4868, %v4869, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4871 = stablehlo.reshape %v4870 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4872 = stablehlo.reshape %v4871 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4873 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4874 = stablehlo.pad %v4872, %v4873, low = [0, 0, 512], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v4875 = stablehlo.reshape %v4874 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4876 = stablehlo.add %v4842, %v4875 : tensor<32x151296xf32>
    %v4877 = stablehlo.reshape %v4561 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4878 = stablehlo.slice %v4877 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4879 = stablehlo.reshape %v4878 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4880 = stablehlo.reshape %v4566 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4881 = stablehlo.slice %v4880 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4882 = stablehlo.reshape %v4881 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4883 = stablehlo.reshape %v4571 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4884 = stablehlo.slice %v4883 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4885 = stablehlo.reshape %v4884 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4886 = stablehlo.reshape %v4882 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4887 = stablehlo.transpose %v4886, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v4888 = stablehlo.reshape %v4887 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4889 = stablehlo.reshape %v4879 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4890 = stablehlo.reshape %v4888 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4891 = stablehlo.dot_general %v4889, %v4890, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v4892 = stablehlo.reshape %v4891 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4893 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v4894 = stablehlo.multiply %v4892, %v4893 : tensor<32x38809xf32>
    %v4895 = stablehlo.reshape %v4894 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4896 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4897 = stablehlo.exponential %v4895 : tensor<32x197x197xf32>
    %v4898 = stablehlo.reduce(%v4897 init: %v4896) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4899 = stablehlo.broadcast_in_dim %v4898, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4900 = stablehlo.divide %v4897, %v4899 : tensor<32x197x197xf32>
    %v4901 = stablehlo.reshape %v4900 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4902 = stablehlo.reshape %v4901 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4903 = stablehlo.reshape %v4885 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4904 = stablehlo.dot_general %v4902, %v4903, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4905 = stablehlo.reshape %v4904 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4906 = stablehlo.reshape %v4905 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4907 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4908 = stablehlo.pad %v4906, %v4907, low = [0, 0, 576], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v4909 = stablehlo.reshape %v4908 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4910 = stablehlo.add %v4876, %v4909 : tensor<32x151296xf32>
    %v4911 = stablehlo.reshape %v4561 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4912 = stablehlo.slice %v4911 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4913 = stablehlo.reshape %v4912 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4914 = stablehlo.reshape %v4566 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4915 = stablehlo.slice %v4914 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4916 = stablehlo.reshape %v4915 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4917 = stablehlo.reshape %v4571 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4918 = stablehlo.slice %v4917 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4919 = stablehlo.reshape %v4918 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4920 = stablehlo.reshape %v4916 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4921 = stablehlo.transpose %v4920, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v4922 = stablehlo.reshape %v4921 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4923 = stablehlo.reshape %v4913 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4924 = stablehlo.reshape %v4922 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4925 = stablehlo.dot_general %v4923, %v4924, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v4926 = stablehlo.reshape %v4925 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4927 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v4928 = stablehlo.multiply %v4926, %v4927 : tensor<32x38809xf32>
    %v4929 = stablehlo.reshape %v4928 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4930 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4931 = stablehlo.exponential %v4929 : tensor<32x197x197xf32>
    %v4932 = stablehlo.reduce(%v4931 init: %v4930) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4933 = stablehlo.broadcast_in_dim %v4932, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4934 = stablehlo.divide %v4931, %v4933 : tensor<32x197x197xf32>
    %v4935 = stablehlo.reshape %v4934 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4936 = stablehlo.reshape %v4935 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4937 = stablehlo.reshape %v4919 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4938 = stablehlo.dot_general %v4936, %v4937, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4939 = stablehlo.reshape %v4938 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4940 = stablehlo.reshape %v4939 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4941 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4942 = stablehlo.pad %v4940, %v4941, low = [0, 0, 640], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v4943 = stablehlo.reshape %v4942 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4944 = stablehlo.add %v4910, %v4943 : tensor<32x151296xf32>
    %v4945 = stablehlo.reshape %v4561 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4946 = stablehlo.slice %v4945 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4947 = stablehlo.reshape %v4946 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4948 = stablehlo.reshape %v4566 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4949 = stablehlo.slice %v4948 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4950 = stablehlo.reshape %v4949 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4951 = stablehlo.reshape %v4571 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4952 = stablehlo.slice %v4951 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4953 = stablehlo.reshape %v4952 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4954 = stablehlo.reshape %v4950 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4955 = stablehlo.transpose %v4954, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v4956 = stablehlo.reshape %v4955 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4957 = stablehlo.reshape %v4947 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4958 = stablehlo.reshape %v4956 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4959 = stablehlo.dot_general %v4957, %v4958, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v4960 = stablehlo.reshape %v4959 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4961 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v4962 = stablehlo.multiply %v4960, %v4961 : tensor<32x38809xf32>
    %v4963 = stablehlo.reshape %v4962 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4964 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4965 = stablehlo.exponential %v4963 : tensor<32x197x197xf32>
    %v4966 = stablehlo.reduce(%v4965 init: %v4964) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4967 = stablehlo.broadcast_in_dim %v4966, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4968 = stablehlo.divide %v4965, %v4967 : tensor<32x197x197xf32>
    %v4969 = stablehlo.reshape %v4968 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4970 = stablehlo.reshape %v4969 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4971 = stablehlo.reshape %v4953 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4972 = stablehlo.dot_general %v4970, %v4971, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4973 = stablehlo.reshape %v4972 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4974 = stablehlo.reshape %v4973 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4975 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4976 = stablehlo.pad %v4974, %v4975, low = [0, 0, 704], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v4977 = stablehlo.reshape %v4976 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4978 = stablehlo.add %v4944, %v4977 : tensor<32x151296xf32>
    %v4979 = stablehlo.reshape %v4978 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4980 = stablehlo.dot_general %v4979, %b9_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v4981 = stablehlo.broadcast_in_dim %b9_bo, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v4982 = stablehlo.add %v4980, %v4981 : tensor<32x197x768xf32>
    %v4983 = stablehlo.reshape %v4982 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4984 = stablehlo.add %v4528, %v4983 : tensor<32x151296xf32>
    %v4985 = stablehlo.reshape %v4984 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4986 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4987 = stablehlo.constant dense<768.0> : tensor<32x197x768xf32>
    %v4988 = stablehlo.constant dense<1.0e-5> : tensor<32x197x768xf32>
    %v4989 = stablehlo.reduce(%v4985 init: %v4986) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4990 = stablehlo.broadcast_in_dim %v4989, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v4991 = stablehlo.divide %v4990, %v4987 : tensor<32x197x768xf32>
    %v4992 = stablehlo.subtract %v4985, %v4991 : tensor<32x197x768xf32>
    %v4993 = stablehlo.multiply %v4992, %v4992 : tensor<32x197x768xf32>
    %v4994 = stablehlo.reduce(%v4993 init: %v4986) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4995 = stablehlo.broadcast_in_dim %v4994, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v4996 = stablehlo.divide %v4995, %v4987 : tensor<32x197x768xf32>
    %v4997 = stablehlo.add %v4996, %v4988 : tensor<32x197x768xf32>
    %v4998 = stablehlo.rsqrt %v4997 : tensor<32x197x768xf32>
    %v4999 = stablehlo.multiply %v4992, %v4998 : tensor<32x197x768xf32>
    %v5000 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v5001 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v5002 = stablehlo.multiply %v4999, %v5000 : tensor<32x197x768xf32>
    %v5003 = stablehlo.add %v5002, %v5001 : tensor<32x197x768xf32>
    %v5004 = stablehlo.reshape %v5003 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5005 = stablehlo.reshape %v5004 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5006 = stablehlo.broadcast_in_dim %b9_g2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v5007 = stablehlo.multiply %v5005, %v5006 : tensor<32x197x768xf32>
    %v5008 = stablehlo.reshape %v5007 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5009 = stablehlo.reshape %v5008 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5010 = stablehlo.broadcast_in_dim %b9_bt2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v5011 = stablehlo.add %v5009, %v5010 : tensor<32x197x768xf32>
    %v5012 = stablehlo.reshape %v5011 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5013 = stablehlo.reshape %v5012 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5014 = stablehlo.dot_general %v5013, %b9_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x3072xf32>) -> tensor<32x197x3072xf32>
    %v5015 = stablehlo.broadcast_in_dim %b9_bfc1, dims = [2] : (tensor<3072xf32>) -> tensor<32x197x3072xf32>
    %v5016 = stablehlo.add %v5014, %v5015 : tensor<32x197x3072xf32>
    %v5017 = stablehlo.reshape %v5016 : (tensor<32x197x3072xf32>) -> tensor<32x605184xf32>
    %v5018 = stablehlo.constant dense<0.5> : tensor<32x605184xf32>
    %v5019 = stablehlo.multiply %v5018, %v5017 : tensor<32x605184xf32>
    %v5020 = stablehlo.negate %v5017 : tensor<32x605184xf32>
    %v5021 = stablehlo.constant dense<0.7071067811865476> : tensor<32x605184xf32>
    %v5022 = stablehlo.multiply %v5020, %v5021 : tensor<32x605184xf32>
    %v5023 = chlo.erfc %v5022 : tensor<32x605184xf32> -> tensor<32x605184xf32>
    %v5024 = stablehlo.multiply %v5019, %v5023 : tensor<32x605184xf32>
    %v5025 = stablehlo.reshape %v5024 : (tensor<32x605184xf32>) -> tensor<32x197x3072xf32>
    %v5026 = stablehlo.dot_general %v5025, %b9_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x3072xf32>, tensor<3072x768xf32>) -> tensor<32x197x768xf32>
    %v5027 = stablehlo.broadcast_in_dim %b9_bfc2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v5028 = stablehlo.add %v5026, %v5027 : tensor<32x197x768xf32>
    %v5029 = stablehlo.reshape %v5028 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5030 = stablehlo.add %v4984, %v5029 : tensor<32x151296xf32>
    %v5031 = stablehlo.reshape %v5030 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5032 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5033 = stablehlo.constant dense<768.0> : tensor<32x197x768xf32>
    %v5034 = stablehlo.constant dense<1.0e-5> : tensor<32x197x768xf32>
    %v5035 = stablehlo.reduce(%v5031 init: %v5032) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5036 = stablehlo.broadcast_in_dim %v5035, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v5037 = stablehlo.divide %v5036, %v5033 : tensor<32x197x768xf32>
    %v5038 = stablehlo.subtract %v5031, %v5037 : tensor<32x197x768xf32>
    %v5039 = stablehlo.multiply %v5038, %v5038 : tensor<32x197x768xf32>
    %v5040 = stablehlo.reduce(%v5039 init: %v5032) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5041 = stablehlo.broadcast_in_dim %v5040, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v5042 = stablehlo.divide %v5041, %v5033 : tensor<32x197x768xf32>
    %v5043 = stablehlo.add %v5042, %v5034 : tensor<32x197x768xf32>
    %v5044 = stablehlo.rsqrt %v5043 : tensor<32x197x768xf32>
    %v5045 = stablehlo.multiply %v5038, %v5044 : tensor<32x197x768xf32>
    %v5046 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v5047 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v5048 = stablehlo.multiply %v5045, %v5046 : tensor<32x197x768xf32>
    %v5049 = stablehlo.add %v5048, %v5047 : tensor<32x197x768xf32>
    %v5050 = stablehlo.reshape %v5049 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5051 = stablehlo.reshape %v5050 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5052 = stablehlo.broadcast_in_dim %b10_g1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v5053 = stablehlo.multiply %v5051, %v5052 : tensor<32x197x768xf32>
    %v5054 = stablehlo.reshape %v5053 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5055 = stablehlo.reshape %v5054 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5056 = stablehlo.broadcast_in_dim %b10_bt1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v5057 = stablehlo.add %v5055, %v5056 : tensor<32x197x768xf32>
    %v5058 = stablehlo.reshape %v5057 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5059 = stablehlo.reshape %v5058 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5060 = stablehlo.dot_general %v5059, %b10_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v5061 = stablehlo.broadcast_in_dim %b10_bq, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v5062 = stablehlo.add %v5060, %v5061 : tensor<32x197x768xf32>
    %v5063 = stablehlo.reshape %v5062 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5064 = stablehlo.reshape %v5058 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5065 = stablehlo.dot_general %v5064, %b10_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v5066 = stablehlo.broadcast_in_dim %b10_bk, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v5067 = stablehlo.add %v5065, %v5066 : tensor<32x197x768xf32>
    %v5068 = stablehlo.reshape %v5067 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5069 = stablehlo.reshape %v5058 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5070 = stablehlo.dot_general %v5069, %b10_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v5071 = stablehlo.broadcast_in_dim %b10_bv, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v5072 = stablehlo.add %v5070, %v5071 : tensor<32x197x768xf32>
    %v5073 = stablehlo.reshape %v5072 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5074 = stablehlo.reshape %v5063 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5075 = stablehlo.slice %v5074 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5076 = stablehlo.reshape %v5075 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5077 = stablehlo.reshape %v5068 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5078 = stablehlo.slice %v5077 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5079 = stablehlo.reshape %v5078 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5080 = stablehlo.reshape %v5073 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5081 = stablehlo.slice %v5080 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5082 = stablehlo.reshape %v5081 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5083 = stablehlo.reshape %v5079 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5084 = stablehlo.transpose %v5083, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5085 = stablehlo.reshape %v5084 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5086 = stablehlo.reshape %v5076 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5087 = stablehlo.reshape %v5085 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5088 = stablehlo.dot_general %v5086, %v5087, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v5089 = stablehlo.reshape %v5088 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5090 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v5091 = stablehlo.multiply %v5089, %v5090 : tensor<32x38809xf32>
    %v5092 = stablehlo.reshape %v5091 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5093 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5094 = stablehlo.exponential %v5092 : tensor<32x197x197xf32>
    %v5095 = stablehlo.reduce(%v5094 init: %v5093) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5096 = stablehlo.broadcast_in_dim %v5095, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5097 = stablehlo.divide %v5094, %v5096 : tensor<32x197x197xf32>
    %v5098 = stablehlo.reshape %v5097 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5099 = stablehlo.reshape %v5098 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5100 = stablehlo.reshape %v5082 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5101 = stablehlo.dot_general %v5099, %v5100, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5102 = stablehlo.reshape %v5101 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5103 = stablehlo.reshape %v5102 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5104 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5105 = stablehlo.pad %v5103, %v5104, low = [0, 0, 0], high = [0, 0, 704], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v5106 = stablehlo.reshape %v5105 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5107 = stablehlo.reshape %v5063 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5108 = stablehlo.slice %v5107 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5109 = stablehlo.reshape %v5108 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5110 = stablehlo.reshape %v5068 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5111 = stablehlo.slice %v5110 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5112 = stablehlo.reshape %v5111 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5113 = stablehlo.reshape %v5073 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5114 = stablehlo.slice %v5113 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5115 = stablehlo.reshape %v5114 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5116 = stablehlo.reshape %v5112 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5117 = stablehlo.transpose %v5116, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5118 = stablehlo.reshape %v5117 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5119 = stablehlo.reshape %v5109 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5120 = stablehlo.reshape %v5118 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5121 = stablehlo.dot_general %v5119, %v5120, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v5122 = stablehlo.reshape %v5121 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5123 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v5124 = stablehlo.multiply %v5122, %v5123 : tensor<32x38809xf32>
    %v5125 = stablehlo.reshape %v5124 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5126 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5127 = stablehlo.exponential %v5125 : tensor<32x197x197xf32>
    %v5128 = stablehlo.reduce(%v5127 init: %v5126) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5129 = stablehlo.broadcast_in_dim %v5128, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5130 = stablehlo.divide %v5127, %v5129 : tensor<32x197x197xf32>
    %v5131 = stablehlo.reshape %v5130 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5132 = stablehlo.reshape %v5131 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5133 = stablehlo.reshape %v5115 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5134 = stablehlo.dot_general %v5132, %v5133, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5135 = stablehlo.reshape %v5134 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5136 = stablehlo.reshape %v5135 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5137 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5138 = stablehlo.pad %v5136, %v5137, low = [0, 0, 64], high = [0, 0, 640], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v5139 = stablehlo.reshape %v5138 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5140 = stablehlo.add %v5106, %v5139 : tensor<32x151296xf32>
    %v5141 = stablehlo.reshape %v5063 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5142 = stablehlo.slice %v5141 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5143 = stablehlo.reshape %v5142 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5144 = stablehlo.reshape %v5068 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5145 = stablehlo.slice %v5144 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5146 = stablehlo.reshape %v5145 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5147 = stablehlo.reshape %v5073 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5148 = stablehlo.slice %v5147 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5149 = stablehlo.reshape %v5148 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5150 = stablehlo.reshape %v5146 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5151 = stablehlo.transpose %v5150, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5152 = stablehlo.reshape %v5151 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5153 = stablehlo.reshape %v5143 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5154 = stablehlo.reshape %v5152 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5155 = stablehlo.dot_general %v5153, %v5154, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v5156 = stablehlo.reshape %v5155 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5157 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v5158 = stablehlo.multiply %v5156, %v5157 : tensor<32x38809xf32>
    %v5159 = stablehlo.reshape %v5158 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5160 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5161 = stablehlo.exponential %v5159 : tensor<32x197x197xf32>
    %v5162 = stablehlo.reduce(%v5161 init: %v5160) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5163 = stablehlo.broadcast_in_dim %v5162, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5164 = stablehlo.divide %v5161, %v5163 : tensor<32x197x197xf32>
    %v5165 = stablehlo.reshape %v5164 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5166 = stablehlo.reshape %v5165 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5167 = stablehlo.reshape %v5149 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5168 = stablehlo.dot_general %v5166, %v5167, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5169 = stablehlo.reshape %v5168 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5170 = stablehlo.reshape %v5169 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5171 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5172 = stablehlo.pad %v5170, %v5171, low = [0, 0, 128], high = [0, 0, 576], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v5173 = stablehlo.reshape %v5172 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5174 = stablehlo.add %v5140, %v5173 : tensor<32x151296xf32>
    %v5175 = stablehlo.reshape %v5063 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5176 = stablehlo.slice %v5175 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5177 = stablehlo.reshape %v5176 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5178 = stablehlo.reshape %v5068 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5179 = stablehlo.slice %v5178 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5180 = stablehlo.reshape %v5179 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5181 = stablehlo.reshape %v5073 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5182 = stablehlo.slice %v5181 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5183 = stablehlo.reshape %v5182 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5184 = stablehlo.reshape %v5180 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5185 = stablehlo.transpose %v5184, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5186 = stablehlo.reshape %v5185 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5187 = stablehlo.reshape %v5177 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5188 = stablehlo.reshape %v5186 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5189 = stablehlo.dot_general %v5187, %v5188, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v5190 = stablehlo.reshape %v5189 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5191 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v5192 = stablehlo.multiply %v5190, %v5191 : tensor<32x38809xf32>
    %v5193 = stablehlo.reshape %v5192 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5194 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5195 = stablehlo.exponential %v5193 : tensor<32x197x197xf32>
    %v5196 = stablehlo.reduce(%v5195 init: %v5194) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5197 = stablehlo.broadcast_in_dim %v5196, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5198 = stablehlo.divide %v5195, %v5197 : tensor<32x197x197xf32>
    %v5199 = stablehlo.reshape %v5198 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5200 = stablehlo.reshape %v5199 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5201 = stablehlo.reshape %v5183 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5202 = stablehlo.dot_general %v5200, %v5201, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5203 = stablehlo.reshape %v5202 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5204 = stablehlo.reshape %v5203 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5205 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5206 = stablehlo.pad %v5204, %v5205, low = [0, 0, 192], high = [0, 0, 512], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v5207 = stablehlo.reshape %v5206 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5208 = stablehlo.add %v5174, %v5207 : tensor<32x151296xf32>
    %v5209 = stablehlo.reshape %v5063 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5210 = stablehlo.slice %v5209 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5211 = stablehlo.reshape %v5210 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5212 = stablehlo.reshape %v5068 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5213 = stablehlo.slice %v5212 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5214 = stablehlo.reshape %v5213 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5215 = stablehlo.reshape %v5073 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5216 = stablehlo.slice %v5215 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5217 = stablehlo.reshape %v5216 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5218 = stablehlo.reshape %v5214 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5219 = stablehlo.transpose %v5218, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5220 = stablehlo.reshape %v5219 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5221 = stablehlo.reshape %v5211 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5222 = stablehlo.reshape %v5220 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5223 = stablehlo.dot_general %v5221, %v5222, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v5224 = stablehlo.reshape %v5223 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5225 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v5226 = stablehlo.multiply %v5224, %v5225 : tensor<32x38809xf32>
    %v5227 = stablehlo.reshape %v5226 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5228 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5229 = stablehlo.exponential %v5227 : tensor<32x197x197xf32>
    %v5230 = stablehlo.reduce(%v5229 init: %v5228) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5231 = stablehlo.broadcast_in_dim %v5230, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5232 = stablehlo.divide %v5229, %v5231 : tensor<32x197x197xf32>
    %v5233 = stablehlo.reshape %v5232 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5234 = stablehlo.reshape %v5233 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5235 = stablehlo.reshape %v5217 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5236 = stablehlo.dot_general %v5234, %v5235, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5237 = stablehlo.reshape %v5236 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5238 = stablehlo.reshape %v5237 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5239 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5240 = stablehlo.pad %v5238, %v5239, low = [0, 0, 256], high = [0, 0, 448], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v5241 = stablehlo.reshape %v5240 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5242 = stablehlo.add %v5208, %v5241 : tensor<32x151296xf32>
    %v5243 = stablehlo.reshape %v5063 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5244 = stablehlo.slice %v5243 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5245 = stablehlo.reshape %v5244 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5246 = stablehlo.reshape %v5068 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5247 = stablehlo.slice %v5246 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5248 = stablehlo.reshape %v5247 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5249 = stablehlo.reshape %v5073 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5250 = stablehlo.slice %v5249 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5251 = stablehlo.reshape %v5250 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5252 = stablehlo.reshape %v5248 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5253 = stablehlo.transpose %v5252, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5254 = stablehlo.reshape %v5253 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5255 = stablehlo.reshape %v5245 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5256 = stablehlo.reshape %v5254 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5257 = stablehlo.dot_general %v5255, %v5256, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v5258 = stablehlo.reshape %v5257 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5259 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v5260 = stablehlo.multiply %v5258, %v5259 : tensor<32x38809xf32>
    %v5261 = stablehlo.reshape %v5260 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5262 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5263 = stablehlo.exponential %v5261 : tensor<32x197x197xf32>
    %v5264 = stablehlo.reduce(%v5263 init: %v5262) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5265 = stablehlo.broadcast_in_dim %v5264, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5266 = stablehlo.divide %v5263, %v5265 : tensor<32x197x197xf32>
    %v5267 = stablehlo.reshape %v5266 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5268 = stablehlo.reshape %v5267 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5269 = stablehlo.reshape %v5251 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5270 = stablehlo.dot_general %v5268, %v5269, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5271 = stablehlo.reshape %v5270 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5272 = stablehlo.reshape %v5271 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5273 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5274 = stablehlo.pad %v5272, %v5273, low = [0, 0, 320], high = [0, 0, 384], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v5275 = stablehlo.reshape %v5274 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5276 = stablehlo.add %v5242, %v5275 : tensor<32x151296xf32>
    %v5277 = stablehlo.reshape %v5063 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5278 = stablehlo.slice %v5277 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5279 = stablehlo.reshape %v5278 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5280 = stablehlo.reshape %v5068 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5281 = stablehlo.slice %v5280 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5282 = stablehlo.reshape %v5281 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5283 = stablehlo.reshape %v5073 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5284 = stablehlo.slice %v5283 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5285 = stablehlo.reshape %v5284 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5286 = stablehlo.reshape %v5282 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5287 = stablehlo.transpose %v5286, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5288 = stablehlo.reshape %v5287 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5289 = stablehlo.reshape %v5279 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5290 = stablehlo.reshape %v5288 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5291 = stablehlo.dot_general %v5289, %v5290, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v5292 = stablehlo.reshape %v5291 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5293 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v5294 = stablehlo.multiply %v5292, %v5293 : tensor<32x38809xf32>
    %v5295 = stablehlo.reshape %v5294 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5296 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5297 = stablehlo.exponential %v5295 : tensor<32x197x197xf32>
    %v5298 = stablehlo.reduce(%v5297 init: %v5296) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5299 = stablehlo.broadcast_in_dim %v5298, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5300 = stablehlo.divide %v5297, %v5299 : tensor<32x197x197xf32>
    %v5301 = stablehlo.reshape %v5300 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5302 = stablehlo.reshape %v5301 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5303 = stablehlo.reshape %v5285 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5304 = stablehlo.dot_general %v5302, %v5303, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5305 = stablehlo.reshape %v5304 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5306 = stablehlo.reshape %v5305 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5307 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5308 = stablehlo.pad %v5306, %v5307, low = [0, 0, 384], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v5309 = stablehlo.reshape %v5308 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5310 = stablehlo.add %v5276, %v5309 : tensor<32x151296xf32>
    %v5311 = stablehlo.reshape %v5063 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5312 = stablehlo.slice %v5311 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5313 = stablehlo.reshape %v5312 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5314 = stablehlo.reshape %v5068 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5315 = stablehlo.slice %v5314 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5316 = stablehlo.reshape %v5315 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5317 = stablehlo.reshape %v5073 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5318 = stablehlo.slice %v5317 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5319 = stablehlo.reshape %v5318 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5320 = stablehlo.reshape %v5316 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5321 = stablehlo.transpose %v5320, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5322 = stablehlo.reshape %v5321 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5323 = stablehlo.reshape %v5313 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5324 = stablehlo.reshape %v5322 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5325 = stablehlo.dot_general %v5323, %v5324, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v5326 = stablehlo.reshape %v5325 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5327 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v5328 = stablehlo.multiply %v5326, %v5327 : tensor<32x38809xf32>
    %v5329 = stablehlo.reshape %v5328 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5330 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5331 = stablehlo.exponential %v5329 : tensor<32x197x197xf32>
    %v5332 = stablehlo.reduce(%v5331 init: %v5330) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5333 = stablehlo.broadcast_in_dim %v5332, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5334 = stablehlo.divide %v5331, %v5333 : tensor<32x197x197xf32>
    %v5335 = stablehlo.reshape %v5334 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5336 = stablehlo.reshape %v5335 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5337 = stablehlo.reshape %v5319 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5338 = stablehlo.dot_general %v5336, %v5337, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5339 = stablehlo.reshape %v5338 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5340 = stablehlo.reshape %v5339 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5341 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5342 = stablehlo.pad %v5340, %v5341, low = [0, 0, 448], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v5343 = stablehlo.reshape %v5342 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5344 = stablehlo.add %v5310, %v5343 : tensor<32x151296xf32>
    %v5345 = stablehlo.reshape %v5063 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5346 = stablehlo.slice %v5345 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5347 = stablehlo.reshape %v5346 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5348 = stablehlo.reshape %v5068 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5349 = stablehlo.slice %v5348 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5350 = stablehlo.reshape %v5349 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5351 = stablehlo.reshape %v5073 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5352 = stablehlo.slice %v5351 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5353 = stablehlo.reshape %v5352 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5354 = stablehlo.reshape %v5350 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5355 = stablehlo.transpose %v5354, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5356 = stablehlo.reshape %v5355 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5357 = stablehlo.reshape %v5347 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5358 = stablehlo.reshape %v5356 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5359 = stablehlo.dot_general %v5357, %v5358, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v5360 = stablehlo.reshape %v5359 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5361 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v5362 = stablehlo.multiply %v5360, %v5361 : tensor<32x38809xf32>
    %v5363 = stablehlo.reshape %v5362 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5364 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5365 = stablehlo.exponential %v5363 : tensor<32x197x197xf32>
    %v5366 = stablehlo.reduce(%v5365 init: %v5364) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5367 = stablehlo.broadcast_in_dim %v5366, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5368 = stablehlo.divide %v5365, %v5367 : tensor<32x197x197xf32>
    %v5369 = stablehlo.reshape %v5368 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5370 = stablehlo.reshape %v5369 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5371 = stablehlo.reshape %v5353 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5372 = stablehlo.dot_general %v5370, %v5371, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5373 = stablehlo.reshape %v5372 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5374 = stablehlo.reshape %v5373 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5375 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5376 = stablehlo.pad %v5374, %v5375, low = [0, 0, 512], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v5377 = stablehlo.reshape %v5376 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5378 = stablehlo.add %v5344, %v5377 : tensor<32x151296xf32>
    %v5379 = stablehlo.reshape %v5063 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5380 = stablehlo.slice %v5379 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5381 = stablehlo.reshape %v5380 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5382 = stablehlo.reshape %v5068 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5383 = stablehlo.slice %v5382 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5384 = stablehlo.reshape %v5383 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5385 = stablehlo.reshape %v5073 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5386 = stablehlo.slice %v5385 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5387 = stablehlo.reshape %v5386 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5388 = stablehlo.reshape %v5384 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5389 = stablehlo.transpose %v5388, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5390 = stablehlo.reshape %v5389 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5391 = stablehlo.reshape %v5381 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5392 = stablehlo.reshape %v5390 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5393 = stablehlo.dot_general %v5391, %v5392, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v5394 = stablehlo.reshape %v5393 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5395 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v5396 = stablehlo.multiply %v5394, %v5395 : tensor<32x38809xf32>
    %v5397 = stablehlo.reshape %v5396 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5398 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5399 = stablehlo.exponential %v5397 : tensor<32x197x197xf32>
    %v5400 = stablehlo.reduce(%v5399 init: %v5398) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5401 = stablehlo.broadcast_in_dim %v5400, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5402 = stablehlo.divide %v5399, %v5401 : tensor<32x197x197xf32>
    %v5403 = stablehlo.reshape %v5402 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5404 = stablehlo.reshape %v5403 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5405 = stablehlo.reshape %v5387 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5406 = stablehlo.dot_general %v5404, %v5405, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5407 = stablehlo.reshape %v5406 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5408 = stablehlo.reshape %v5407 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5409 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5410 = stablehlo.pad %v5408, %v5409, low = [0, 0, 576], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v5411 = stablehlo.reshape %v5410 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5412 = stablehlo.add %v5378, %v5411 : tensor<32x151296xf32>
    %v5413 = stablehlo.reshape %v5063 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5414 = stablehlo.slice %v5413 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5415 = stablehlo.reshape %v5414 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5416 = stablehlo.reshape %v5068 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5417 = stablehlo.slice %v5416 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5418 = stablehlo.reshape %v5417 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5419 = stablehlo.reshape %v5073 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5420 = stablehlo.slice %v5419 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5421 = stablehlo.reshape %v5420 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5422 = stablehlo.reshape %v5418 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5423 = stablehlo.transpose %v5422, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5424 = stablehlo.reshape %v5423 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5425 = stablehlo.reshape %v5415 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5426 = stablehlo.reshape %v5424 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5427 = stablehlo.dot_general %v5425, %v5426, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v5428 = stablehlo.reshape %v5427 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5429 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v5430 = stablehlo.multiply %v5428, %v5429 : tensor<32x38809xf32>
    %v5431 = stablehlo.reshape %v5430 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5432 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5433 = stablehlo.exponential %v5431 : tensor<32x197x197xf32>
    %v5434 = stablehlo.reduce(%v5433 init: %v5432) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5435 = stablehlo.broadcast_in_dim %v5434, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5436 = stablehlo.divide %v5433, %v5435 : tensor<32x197x197xf32>
    %v5437 = stablehlo.reshape %v5436 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5438 = stablehlo.reshape %v5437 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5439 = stablehlo.reshape %v5421 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5440 = stablehlo.dot_general %v5438, %v5439, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5441 = stablehlo.reshape %v5440 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5442 = stablehlo.reshape %v5441 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5443 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5444 = stablehlo.pad %v5442, %v5443, low = [0, 0, 640], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v5445 = stablehlo.reshape %v5444 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5446 = stablehlo.add %v5412, %v5445 : tensor<32x151296xf32>
    %v5447 = stablehlo.reshape %v5063 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5448 = stablehlo.slice %v5447 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5449 = stablehlo.reshape %v5448 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5450 = stablehlo.reshape %v5068 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5451 = stablehlo.slice %v5450 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5452 = stablehlo.reshape %v5451 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5453 = stablehlo.reshape %v5073 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5454 = stablehlo.slice %v5453 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5455 = stablehlo.reshape %v5454 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5456 = stablehlo.reshape %v5452 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5457 = stablehlo.transpose %v5456, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5458 = stablehlo.reshape %v5457 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5459 = stablehlo.reshape %v5449 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5460 = stablehlo.reshape %v5458 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5461 = stablehlo.dot_general %v5459, %v5460, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v5462 = stablehlo.reshape %v5461 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5463 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v5464 = stablehlo.multiply %v5462, %v5463 : tensor<32x38809xf32>
    %v5465 = stablehlo.reshape %v5464 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5466 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5467 = stablehlo.exponential %v5465 : tensor<32x197x197xf32>
    %v5468 = stablehlo.reduce(%v5467 init: %v5466) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5469 = stablehlo.broadcast_in_dim %v5468, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5470 = stablehlo.divide %v5467, %v5469 : tensor<32x197x197xf32>
    %v5471 = stablehlo.reshape %v5470 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5472 = stablehlo.reshape %v5471 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5473 = stablehlo.reshape %v5455 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5474 = stablehlo.dot_general %v5472, %v5473, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5475 = stablehlo.reshape %v5474 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5476 = stablehlo.reshape %v5475 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5477 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5478 = stablehlo.pad %v5476, %v5477, low = [0, 0, 704], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v5479 = stablehlo.reshape %v5478 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5480 = stablehlo.add %v5446, %v5479 : tensor<32x151296xf32>
    %v5481 = stablehlo.reshape %v5480 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5482 = stablehlo.dot_general %v5481, %b10_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v5483 = stablehlo.broadcast_in_dim %b10_bo, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v5484 = stablehlo.add %v5482, %v5483 : tensor<32x197x768xf32>
    %v5485 = stablehlo.reshape %v5484 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5486 = stablehlo.add %v5030, %v5485 : tensor<32x151296xf32>
    %v5487 = stablehlo.reshape %v5486 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5488 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5489 = stablehlo.constant dense<768.0> : tensor<32x197x768xf32>
    %v5490 = stablehlo.constant dense<1.0e-5> : tensor<32x197x768xf32>
    %v5491 = stablehlo.reduce(%v5487 init: %v5488) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5492 = stablehlo.broadcast_in_dim %v5491, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v5493 = stablehlo.divide %v5492, %v5489 : tensor<32x197x768xf32>
    %v5494 = stablehlo.subtract %v5487, %v5493 : tensor<32x197x768xf32>
    %v5495 = stablehlo.multiply %v5494, %v5494 : tensor<32x197x768xf32>
    %v5496 = stablehlo.reduce(%v5495 init: %v5488) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5497 = stablehlo.broadcast_in_dim %v5496, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v5498 = stablehlo.divide %v5497, %v5489 : tensor<32x197x768xf32>
    %v5499 = stablehlo.add %v5498, %v5490 : tensor<32x197x768xf32>
    %v5500 = stablehlo.rsqrt %v5499 : tensor<32x197x768xf32>
    %v5501 = stablehlo.multiply %v5494, %v5500 : tensor<32x197x768xf32>
    %v5502 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v5503 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v5504 = stablehlo.multiply %v5501, %v5502 : tensor<32x197x768xf32>
    %v5505 = stablehlo.add %v5504, %v5503 : tensor<32x197x768xf32>
    %v5506 = stablehlo.reshape %v5505 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5507 = stablehlo.reshape %v5506 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5508 = stablehlo.broadcast_in_dim %b10_g2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v5509 = stablehlo.multiply %v5507, %v5508 : tensor<32x197x768xf32>
    %v5510 = stablehlo.reshape %v5509 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5511 = stablehlo.reshape %v5510 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5512 = stablehlo.broadcast_in_dim %b10_bt2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v5513 = stablehlo.add %v5511, %v5512 : tensor<32x197x768xf32>
    %v5514 = stablehlo.reshape %v5513 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5515 = stablehlo.reshape %v5514 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5516 = stablehlo.dot_general %v5515, %b10_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x3072xf32>) -> tensor<32x197x3072xf32>
    %v5517 = stablehlo.broadcast_in_dim %b10_bfc1, dims = [2] : (tensor<3072xf32>) -> tensor<32x197x3072xf32>
    %v5518 = stablehlo.add %v5516, %v5517 : tensor<32x197x3072xf32>
    %v5519 = stablehlo.reshape %v5518 : (tensor<32x197x3072xf32>) -> tensor<32x605184xf32>
    %v5520 = stablehlo.constant dense<0.5> : tensor<32x605184xf32>
    %v5521 = stablehlo.multiply %v5520, %v5519 : tensor<32x605184xf32>
    %v5522 = stablehlo.negate %v5519 : tensor<32x605184xf32>
    %v5523 = stablehlo.constant dense<0.7071067811865476> : tensor<32x605184xf32>
    %v5524 = stablehlo.multiply %v5522, %v5523 : tensor<32x605184xf32>
    %v5525 = chlo.erfc %v5524 : tensor<32x605184xf32> -> tensor<32x605184xf32>
    %v5526 = stablehlo.multiply %v5521, %v5525 : tensor<32x605184xf32>
    %v5527 = stablehlo.reshape %v5526 : (tensor<32x605184xf32>) -> tensor<32x197x3072xf32>
    %v5528 = stablehlo.dot_general %v5527, %b10_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x3072xf32>, tensor<3072x768xf32>) -> tensor<32x197x768xf32>
    %v5529 = stablehlo.broadcast_in_dim %b10_bfc2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v5530 = stablehlo.add %v5528, %v5529 : tensor<32x197x768xf32>
    %v5531 = stablehlo.reshape %v5530 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5532 = stablehlo.add %v5486, %v5531 : tensor<32x151296xf32>
    %v5533 = stablehlo.reshape %v5532 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5534 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5535 = stablehlo.constant dense<768.0> : tensor<32x197x768xf32>
    %v5536 = stablehlo.constant dense<1.0e-5> : tensor<32x197x768xf32>
    %v5537 = stablehlo.reduce(%v5533 init: %v5534) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5538 = stablehlo.broadcast_in_dim %v5537, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v5539 = stablehlo.divide %v5538, %v5535 : tensor<32x197x768xf32>
    %v5540 = stablehlo.subtract %v5533, %v5539 : tensor<32x197x768xf32>
    %v5541 = stablehlo.multiply %v5540, %v5540 : tensor<32x197x768xf32>
    %v5542 = stablehlo.reduce(%v5541 init: %v5534) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5543 = stablehlo.broadcast_in_dim %v5542, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v5544 = stablehlo.divide %v5543, %v5535 : tensor<32x197x768xf32>
    %v5545 = stablehlo.add %v5544, %v5536 : tensor<32x197x768xf32>
    %v5546 = stablehlo.rsqrt %v5545 : tensor<32x197x768xf32>
    %v5547 = stablehlo.multiply %v5540, %v5546 : tensor<32x197x768xf32>
    %v5548 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v5549 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v5550 = stablehlo.multiply %v5547, %v5548 : tensor<32x197x768xf32>
    %v5551 = stablehlo.add %v5550, %v5549 : tensor<32x197x768xf32>
    %v5552 = stablehlo.reshape %v5551 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5553 = stablehlo.reshape %v5552 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5554 = stablehlo.broadcast_in_dim %b11_g1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v5555 = stablehlo.multiply %v5553, %v5554 : tensor<32x197x768xf32>
    %v5556 = stablehlo.reshape %v5555 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5557 = stablehlo.reshape %v5556 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5558 = stablehlo.broadcast_in_dim %b11_bt1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v5559 = stablehlo.add %v5557, %v5558 : tensor<32x197x768xf32>
    %v5560 = stablehlo.reshape %v5559 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5561 = stablehlo.reshape %v5560 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5562 = stablehlo.dot_general %v5561, %b11_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v5563 = stablehlo.broadcast_in_dim %b11_bq, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v5564 = stablehlo.add %v5562, %v5563 : tensor<32x197x768xf32>
    %v5565 = stablehlo.reshape %v5564 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5566 = stablehlo.reshape %v5560 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5567 = stablehlo.dot_general %v5566, %b11_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v5568 = stablehlo.broadcast_in_dim %b11_bk, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v5569 = stablehlo.add %v5567, %v5568 : tensor<32x197x768xf32>
    %v5570 = stablehlo.reshape %v5569 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5571 = stablehlo.reshape %v5560 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5572 = stablehlo.dot_general %v5571, %b11_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v5573 = stablehlo.broadcast_in_dim %b11_bv, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v5574 = stablehlo.add %v5572, %v5573 : tensor<32x197x768xf32>
    %v5575 = stablehlo.reshape %v5574 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5576 = stablehlo.reshape %v5565 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5577 = stablehlo.slice %v5576 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5578 = stablehlo.reshape %v5577 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5579 = stablehlo.reshape %v5570 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5580 = stablehlo.slice %v5579 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5581 = stablehlo.reshape %v5580 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5582 = stablehlo.reshape %v5575 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5583 = stablehlo.slice %v5582 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5584 = stablehlo.reshape %v5583 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5585 = stablehlo.reshape %v5581 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5586 = stablehlo.transpose %v5585, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5587 = stablehlo.reshape %v5586 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5588 = stablehlo.reshape %v5578 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5589 = stablehlo.reshape %v5587 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5590 = stablehlo.dot_general %v5588, %v5589, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v5591 = stablehlo.reshape %v5590 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5592 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v5593 = stablehlo.multiply %v5591, %v5592 : tensor<32x38809xf32>
    %v5594 = stablehlo.reshape %v5593 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5595 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5596 = stablehlo.exponential %v5594 : tensor<32x197x197xf32>
    %v5597 = stablehlo.reduce(%v5596 init: %v5595) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5598 = stablehlo.broadcast_in_dim %v5597, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5599 = stablehlo.divide %v5596, %v5598 : tensor<32x197x197xf32>
    %v5600 = stablehlo.reshape %v5599 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5601 = stablehlo.reshape %v5600 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5602 = stablehlo.reshape %v5584 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5603 = stablehlo.dot_general %v5601, %v5602, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5604 = stablehlo.reshape %v5603 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5605 = stablehlo.reshape %v5604 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5606 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5607 = stablehlo.pad %v5605, %v5606, low = [0, 0, 0], high = [0, 0, 704], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v5608 = stablehlo.reshape %v5607 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5609 = stablehlo.reshape %v5565 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5610 = stablehlo.slice %v5609 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5611 = stablehlo.reshape %v5610 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5612 = stablehlo.reshape %v5570 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5613 = stablehlo.slice %v5612 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5614 = stablehlo.reshape %v5613 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5615 = stablehlo.reshape %v5575 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5616 = stablehlo.slice %v5615 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5617 = stablehlo.reshape %v5616 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5618 = stablehlo.reshape %v5614 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5619 = stablehlo.transpose %v5618, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5620 = stablehlo.reshape %v5619 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5621 = stablehlo.reshape %v5611 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5622 = stablehlo.reshape %v5620 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5623 = stablehlo.dot_general %v5621, %v5622, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v5624 = stablehlo.reshape %v5623 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5625 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v5626 = stablehlo.multiply %v5624, %v5625 : tensor<32x38809xf32>
    %v5627 = stablehlo.reshape %v5626 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5628 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5629 = stablehlo.exponential %v5627 : tensor<32x197x197xf32>
    %v5630 = stablehlo.reduce(%v5629 init: %v5628) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5631 = stablehlo.broadcast_in_dim %v5630, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5632 = stablehlo.divide %v5629, %v5631 : tensor<32x197x197xf32>
    %v5633 = stablehlo.reshape %v5632 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5634 = stablehlo.reshape %v5633 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5635 = stablehlo.reshape %v5617 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5636 = stablehlo.dot_general %v5634, %v5635, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5637 = stablehlo.reshape %v5636 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5638 = stablehlo.reshape %v5637 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5639 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5640 = stablehlo.pad %v5638, %v5639, low = [0, 0, 64], high = [0, 0, 640], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v5641 = stablehlo.reshape %v5640 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5642 = stablehlo.add %v5608, %v5641 : tensor<32x151296xf32>
    %v5643 = stablehlo.reshape %v5565 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5644 = stablehlo.slice %v5643 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5645 = stablehlo.reshape %v5644 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5646 = stablehlo.reshape %v5570 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5647 = stablehlo.slice %v5646 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5648 = stablehlo.reshape %v5647 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5649 = stablehlo.reshape %v5575 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5650 = stablehlo.slice %v5649 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5651 = stablehlo.reshape %v5650 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5652 = stablehlo.reshape %v5648 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5653 = stablehlo.transpose %v5652, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5654 = stablehlo.reshape %v5653 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5655 = stablehlo.reshape %v5645 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5656 = stablehlo.reshape %v5654 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5657 = stablehlo.dot_general %v5655, %v5656, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v5658 = stablehlo.reshape %v5657 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5659 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v5660 = stablehlo.multiply %v5658, %v5659 : tensor<32x38809xf32>
    %v5661 = stablehlo.reshape %v5660 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5662 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5663 = stablehlo.exponential %v5661 : tensor<32x197x197xf32>
    %v5664 = stablehlo.reduce(%v5663 init: %v5662) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5665 = stablehlo.broadcast_in_dim %v5664, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5666 = stablehlo.divide %v5663, %v5665 : tensor<32x197x197xf32>
    %v5667 = stablehlo.reshape %v5666 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5668 = stablehlo.reshape %v5667 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5669 = stablehlo.reshape %v5651 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5670 = stablehlo.dot_general %v5668, %v5669, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5671 = stablehlo.reshape %v5670 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5672 = stablehlo.reshape %v5671 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5673 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5674 = stablehlo.pad %v5672, %v5673, low = [0, 0, 128], high = [0, 0, 576], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v5675 = stablehlo.reshape %v5674 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5676 = stablehlo.add %v5642, %v5675 : tensor<32x151296xf32>
    %v5677 = stablehlo.reshape %v5565 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5678 = stablehlo.slice %v5677 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5679 = stablehlo.reshape %v5678 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5680 = stablehlo.reshape %v5570 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5681 = stablehlo.slice %v5680 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5682 = stablehlo.reshape %v5681 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5683 = stablehlo.reshape %v5575 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5684 = stablehlo.slice %v5683 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5685 = stablehlo.reshape %v5684 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5686 = stablehlo.reshape %v5682 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5687 = stablehlo.transpose %v5686, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5688 = stablehlo.reshape %v5687 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5689 = stablehlo.reshape %v5679 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5690 = stablehlo.reshape %v5688 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5691 = stablehlo.dot_general %v5689, %v5690, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v5692 = stablehlo.reshape %v5691 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5693 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v5694 = stablehlo.multiply %v5692, %v5693 : tensor<32x38809xf32>
    %v5695 = stablehlo.reshape %v5694 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5696 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5697 = stablehlo.exponential %v5695 : tensor<32x197x197xf32>
    %v5698 = stablehlo.reduce(%v5697 init: %v5696) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5699 = stablehlo.broadcast_in_dim %v5698, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5700 = stablehlo.divide %v5697, %v5699 : tensor<32x197x197xf32>
    %v5701 = stablehlo.reshape %v5700 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5702 = stablehlo.reshape %v5701 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5703 = stablehlo.reshape %v5685 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5704 = stablehlo.dot_general %v5702, %v5703, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5705 = stablehlo.reshape %v5704 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5706 = stablehlo.reshape %v5705 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5707 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5708 = stablehlo.pad %v5706, %v5707, low = [0, 0, 192], high = [0, 0, 512], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v5709 = stablehlo.reshape %v5708 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5710 = stablehlo.add %v5676, %v5709 : tensor<32x151296xf32>
    %v5711 = stablehlo.reshape %v5565 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5712 = stablehlo.slice %v5711 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5713 = stablehlo.reshape %v5712 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5714 = stablehlo.reshape %v5570 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5715 = stablehlo.slice %v5714 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5716 = stablehlo.reshape %v5715 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5717 = stablehlo.reshape %v5575 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5718 = stablehlo.slice %v5717 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5719 = stablehlo.reshape %v5718 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5720 = stablehlo.reshape %v5716 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5721 = stablehlo.transpose %v5720, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5722 = stablehlo.reshape %v5721 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5723 = stablehlo.reshape %v5713 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5724 = stablehlo.reshape %v5722 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5725 = stablehlo.dot_general %v5723, %v5724, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v5726 = stablehlo.reshape %v5725 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5727 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v5728 = stablehlo.multiply %v5726, %v5727 : tensor<32x38809xf32>
    %v5729 = stablehlo.reshape %v5728 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5730 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5731 = stablehlo.exponential %v5729 : tensor<32x197x197xf32>
    %v5732 = stablehlo.reduce(%v5731 init: %v5730) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5733 = stablehlo.broadcast_in_dim %v5732, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5734 = stablehlo.divide %v5731, %v5733 : tensor<32x197x197xf32>
    %v5735 = stablehlo.reshape %v5734 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5736 = stablehlo.reshape %v5735 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5737 = stablehlo.reshape %v5719 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5738 = stablehlo.dot_general %v5736, %v5737, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5739 = stablehlo.reshape %v5738 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5740 = stablehlo.reshape %v5739 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5741 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5742 = stablehlo.pad %v5740, %v5741, low = [0, 0, 256], high = [0, 0, 448], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v5743 = stablehlo.reshape %v5742 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5744 = stablehlo.add %v5710, %v5743 : tensor<32x151296xf32>
    %v5745 = stablehlo.reshape %v5565 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5746 = stablehlo.slice %v5745 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5747 = stablehlo.reshape %v5746 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5748 = stablehlo.reshape %v5570 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5749 = stablehlo.slice %v5748 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5750 = stablehlo.reshape %v5749 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5751 = stablehlo.reshape %v5575 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5752 = stablehlo.slice %v5751 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5753 = stablehlo.reshape %v5752 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5754 = stablehlo.reshape %v5750 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5755 = stablehlo.transpose %v5754, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5756 = stablehlo.reshape %v5755 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5757 = stablehlo.reshape %v5747 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5758 = stablehlo.reshape %v5756 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5759 = stablehlo.dot_general %v5757, %v5758, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v5760 = stablehlo.reshape %v5759 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5761 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v5762 = stablehlo.multiply %v5760, %v5761 : tensor<32x38809xf32>
    %v5763 = stablehlo.reshape %v5762 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5764 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5765 = stablehlo.exponential %v5763 : tensor<32x197x197xf32>
    %v5766 = stablehlo.reduce(%v5765 init: %v5764) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5767 = stablehlo.broadcast_in_dim %v5766, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5768 = stablehlo.divide %v5765, %v5767 : tensor<32x197x197xf32>
    %v5769 = stablehlo.reshape %v5768 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5770 = stablehlo.reshape %v5769 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5771 = stablehlo.reshape %v5753 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5772 = stablehlo.dot_general %v5770, %v5771, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5773 = stablehlo.reshape %v5772 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5774 = stablehlo.reshape %v5773 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5775 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5776 = stablehlo.pad %v5774, %v5775, low = [0, 0, 320], high = [0, 0, 384], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v5777 = stablehlo.reshape %v5776 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5778 = stablehlo.add %v5744, %v5777 : tensor<32x151296xf32>
    %v5779 = stablehlo.reshape %v5565 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5780 = stablehlo.slice %v5779 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5781 = stablehlo.reshape %v5780 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5782 = stablehlo.reshape %v5570 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5783 = stablehlo.slice %v5782 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5784 = stablehlo.reshape %v5783 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5785 = stablehlo.reshape %v5575 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5786 = stablehlo.slice %v5785 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5787 = stablehlo.reshape %v5786 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5788 = stablehlo.reshape %v5784 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5789 = stablehlo.transpose %v5788, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5790 = stablehlo.reshape %v5789 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5791 = stablehlo.reshape %v5781 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5792 = stablehlo.reshape %v5790 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5793 = stablehlo.dot_general %v5791, %v5792, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v5794 = stablehlo.reshape %v5793 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5795 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v5796 = stablehlo.multiply %v5794, %v5795 : tensor<32x38809xf32>
    %v5797 = stablehlo.reshape %v5796 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5798 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5799 = stablehlo.exponential %v5797 : tensor<32x197x197xf32>
    %v5800 = stablehlo.reduce(%v5799 init: %v5798) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5801 = stablehlo.broadcast_in_dim %v5800, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5802 = stablehlo.divide %v5799, %v5801 : tensor<32x197x197xf32>
    %v5803 = stablehlo.reshape %v5802 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5804 = stablehlo.reshape %v5803 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5805 = stablehlo.reshape %v5787 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5806 = stablehlo.dot_general %v5804, %v5805, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5807 = stablehlo.reshape %v5806 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5808 = stablehlo.reshape %v5807 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5809 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5810 = stablehlo.pad %v5808, %v5809, low = [0, 0, 384], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v5811 = stablehlo.reshape %v5810 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5812 = stablehlo.add %v5778, %v5811 : tensor<32x151296xf32>
    %v5813 = stablehlo.reshape %v5565 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5814 = stablehlo.slice %v5813 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5815 = stablehlo.reshape %v5814 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5816 = stablehlo.reshape %v5570 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5817 = stablehlo.slice %v5816 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5818 = stablehlo.reshape %v5817 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5819 = stablehlo.reshape %v5575 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5820 = stablehlo.slice %v5819 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5821 = stablehlo.reshape %v5820 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5822 = stablehlo.reshape %v5818 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5823 = stablehlo.transpose %v5822, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5824 = stablehlo.reshape %v5823 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5825 = stablehlo.reshape %v5815 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5826 = stablehlo.reshape %v5824 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5827 = stablehlo.dot_general %v5825, %v5826, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v5828 = stablehlo.reshape %v5827 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5829 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v5830 = stablehlo.multiply %v5828, %v5829 : tensor<32x38809xf32>
    %v5831 = stablehlo.reshape %v5830 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5832 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5833 = stablehlo.exponential %v5831 : tensor<32x197x197xf32>
    %v5834 = stablehlo.reduce(%v5833 init: %v5832) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5835 = stablehlo.broadcast_in_dim %v5834, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5836 = stablehlo.divide %v5833, %v5835 : tensor<32x197x197xf32>
    %v5837 = stablehlo.reshape %v5836 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5838 = stablehlo.reshape %v5837 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5839 = stablehlo.reshape %v5821 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5840 = stablehlo.dot_general %v5838, %v5839, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5841 = stablehlo.reshape %v5840 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5842 = stablehlo.reshape %v5841 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5843 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5844 = stablehlo.pad %v5842, %v5843, low = [0, 0, 448], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v5845 = stablehlo.reshape %v5844 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5846 = stablehlo.add %v5812, %v5845 : tensor<32x151296xf32>
    %v5847 = stablehlo.reshape %v5565 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5848 = stablehlo.slice %v5847 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5849 = stablehlo.reshape %v5848 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5850 = stablehlo.reshape %v5570 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5851 = stablehlo.slice %v5850 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5852 = stablehlo.reshape %v5851 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5853 = stablehlo.reshape %v5575 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5854 = stablehlo.slice %v5853 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5855 = stablehlo.reshape %v5854 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5856 = stablehlo.reshape %v5852 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5857 = stablehlo.transpose %v5856, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5858 = stablehlo.reshape %v5857 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5859 = stablehlo.reshape %v5849 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5860 = stablehlo.reshape %v5858 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5861 = stablehlo.dot_general %v5859, %v5860, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v5862 = stablehlo.reshape %v5861 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5863 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v5864 = stablehlo.multiply %v5862, %v5863 : tensor<32x38809xf32>
    %v5865 = stablehlo.reshape %v5864 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5866 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5867 = stablehlo.exponential %v5865 : tensor<32x197x197xf32>
    %v5868 = stablehlo.reduce(%v5867 init: %v5866) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5869 = stablehlo.broadcast_in_dim %v5868, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5870 = stablehlo.divide %v5867, %v5869 : tensor<32x197x197xf32>
    %v5871 = stablehlo.reshape %v5870 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5872 = stablehlo.reshape %v5871 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5873 = stablehlo.reshape %v5855 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5874 = stablehlo.dot_general %v5872, %v5873, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5875 = stablehlo.reshape %v5874 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5876 = stablehlo.reshape %v5875 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5877 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5878 = stablehlo.pad %v5876, %v5877, low = [0, 0, 512], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v5879 = stablehlo.reshape %v5878 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5880 = stablehlo.add %v5846, %v5879 : tensor<32x151296xf32>
    %v5881 = stablehlo.reshape %v5565 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5882 = stablehlo.slice %v5881 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5883 = stablehlo.reshape %v5882 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5884 = stablehlo.reshape %v5570 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5885 = stablehlo.slice %v5884 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5886 = stablehlo.reshape %v5885 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5887 = stablehlo.reshape %v5575 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5888 = stablehlo.slice %v5887 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5889 = stablehlo.reshape %v5888 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5890 = stablehlo.reshape %v5886 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5891 = stablehlo.transpose %v5890, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5892 = stablehlo.reshape %v5891 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5893 = stablehlo.reshape %v5883 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5894 = stablehlo.reshape %v5892 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5895 = stablehlo.dot_general %v5893, %v5894, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v5896 = stablehlo.reshape %v5895 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5897 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v5898 = stablehlo.multiply %v5896, %v5897 : tensor<32x38809xf32>
    %v5899 = stablehlo.reshape %v5898 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5900 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5901 = stablehlo.exponential %v5899 : tensor<32x197x197xf32>
    %v5902 = stablehlo.reduce(%v5901 init: %v5900) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5903 = stablehlo.broadcast_in_dim %v5902, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5904 = stablehlo.divide %v5901, %v5903 : tensor<32x197x197xf32>
    %v5905 = stablehlo.reshape %v5904 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5906 = stablehlo.reshape %v5905 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5907 = stablehlo.reshape %v5889 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5908 = stablehlo.dot_general %v5906, %v5907, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5909 = stablehlo.reshape %v5908 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5910 = stablehlo.reshape %v5909 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5911 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5912 = stablehlo.pad %v5910, %v5911, low = [0, 0, 576], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v5913 = stablehlo.reshape %v5912 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5914 = stablehlo.add %v5880, %v5913 : tensor<32x151296xf32>
    %v5915 = stablehlo.reshape %v5565 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5916 = stablehlo.slice %v5915 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5917 = stablehlo.reshape %v5916 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5918 = stablehlo.reshape %v5570 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5919 = stablehlo.slice %v5918 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5920 = stablehlo.reshape %v5919 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5921 = stablehlo.reshape %v5575 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5922 = stablehlo.slice %v5921 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5923 = stablehlo.reshape %v5922 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5924 = stablehlo.reshape %v5920 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5925 = stablehlo.transpose %v5924, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5926 = stablehlo.reshape %v5925 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5927 = stablehlo.reshape %v5917 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5928 = stablehlo.reshape %v5926 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5929 = stablehlo.dot_general %v5927, %v5928, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v5930 = stablehlo.reshape %v5929 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5931 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v5932 = stablehlo.multiply %v5930, %v5931 : tensor<32x38809xf32>
    %v5933 = stablehlo.reshape %v5932 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5934 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5935 = stablehlo.exponential %v5933 : tensor<32x197x197xf32>
    %v5936 = stablehlo.reduce(%v5935 init: %v5934) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5937 = stablehlo.broadcast_in_dim %v5936, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5938 = stablehlo.divide %v5935, %v5937 : tensor<32x197x197xf32>
    %v5939 = stablehlo.reshape %v5938 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5940 = stablehlo.reshape %v5939 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5941 = stablehlo.reshape %v5923 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5942 = stablehlo.dot_general %v5940, %v5941, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5943 = stablehlo.reshape %v5942 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5944 = stablehlo.reshape %v5943 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5945 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5946 = stablehlo.pad %v5944, %v5945, low = [0, 0, 640], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v5947 = stablehlo.reshape %v5946 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5948 = stablehlo.add %v5914, %v5947 : tensor<32x151296xf32>
    %v5949 = stablehlo.reshape %v5565 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5950 = stablehlo.slice %v5949 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5951 = stablehlo.reshape %v5950 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5952 = stablehlo.reshape %v5570 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5953 = stablehlo.slice %v5952 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5954 = stablehlo.reshape %v5953 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5955 = stablehlo.reshape %v5575 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5956 = stablehlo.slice %v5955 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5957 = stablehlo.reshape %v5956 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5958 = stablehlo.reshape %v5954 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5959 = stablehlo.transpose %v5958, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5960 = stablehlo.reshape %v5959 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5961 = stablehlo.reshape %v5951 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5962 = stablehlo.reshape %v5960 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5963 = stablehlo.dot_general %v5961, %v5962, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v5964 = stablehlo.reshape %v5963 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5965 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v5966 = stablehlo.multiply %v5964, %v5965 : tensor<32x38809xf32>
    %v5967 = stablehlo.reshape %v5966 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5968 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5969 = stablehlo.exponential %v5967 : tensor<32x197x197xf32>
    %v5970 = stablehlo.reduce(%v5969 init: %v5968) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5971 = stablehlo.broadcast_in_dim %v5970, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5972 = stablehlo.divide %v5969, %v5971 : tensor<32x197x197xf32>
    %v5973 = stablehlo.reshape %v5972 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5974 = stablehlo.reshape %v5973 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5975 = stablehlo.reshape %v5957 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5976 = stablehlo.dot_general %v5974, %v5975, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5977 = stablehlo.reshape %v5976 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5978 = stablehlo.reshape %v5977 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5979 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5980 = stablehlo.pad %v5978, %v5979, low = [0, 0, 704], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v5981 = stablehlo.reshape %v5980 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5982 = stablehlo.add %v5948, %v5981 : tensor<32x151296xf32>
    %v5983 = stablehlo.reshape %v5982 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5984 = stablehlo.dot_general %v5983, %b11_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v5985 = stablehlo.broadcast_in_dim %b11_bo, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v5986 = stablehlo.add %v5984, %v5985 : tensor<32x197x768xf32>
    %v5987 = stablehlo.reshape %v5986 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5988 = stablehlo.add %v5532, %v5987 : tensor<32x151296xf32>
    %v5989 = stablehlo.reshape %v5988 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5990 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5991 = stablehlo.constant dense<768.0> : tensor<32x197x768xf32>
    %v5992 = stablehlo.constant dense<1.0e-5> : tensor<32x197x768xf32>
    %v5993 = stablehlo.reduce(%v5989 init: %v5990) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5994 = stablehlo.broadcast_in_dim %v5993, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v5995 = stablehlo.divide %v5994, %v5991 : tensor<32x197x768xf32>
    %v5996 = stablehlo.subtract %v5989, %v5995 : tensor<32x197x768xf32>
    %v5997 = stablehlo.multiply %v5996, %v5996 : tensor<32x197x768xf32>
    %v5998 = stablehlo.reduce(%v5997 init: %v5990) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5999 = stablehlo.broadcast_in_dim %v5998, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v6000 = stablehlo.divide %v5999, %v5991 : tensor<32x197x768xf32>
    %v6001 = stablehlo.add %v6000, %v5992 : tensor<32x197x768xf32>
    %v6002 = stablehlo.rsqrt %v6001 : tensor<32x197x768xf32>
    %v6003 = stablehlo.multiply %v5996, %v6002 : tensor<32x197x768xf32>
    %v6004 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v6005 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v6006 = stablehlo.multiply %v6003, %v6004 : tensor<32x197x768xf32>
    %v6007 = stablehlo.add %v6006, %v6005 : tensor<32x197x768xf32>
    %v6008 = stablehlo.reshape %v6007 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v6009 = stablehlo.reshape %v6008 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6010 = stablehlo.broadcast_in_dim %b11_g2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v6011 = stablehlo.multiply %v6009, %v6010 : tensor<32x197x768xf32>
    %v6012 = stablehlo.reshape %v6011 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v6013 = stablehlo.reshape %v6012 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6014 = stablehlo.broadcast_in_dim %b11_bt2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v6015 = stablehlo.add %v6013, %v6014 : tensor<32x197x768xf32>
    %v6016 = stablehlo.reshape %v6015 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v6017 = stablehlo.reshape %v6016 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6018 = stablehlo.dot_general %v6017, %b11_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x3072xf32>) -> tensor<32x197x3072xf32>
    %v6019 = stablehlo.broadcast_in_dim %b11_bfc1, dims = [2] : (tensor<3072xf32>) -> tensor<32x197x3072xf32>
    %v6020 = stablehlo.add %v6018, %v6019 : tensor<32x197x3072xf32>
    %v6021 = stablehlo.reshape %v6020 : (tensor<32x197x3072xf32>) -> tensor<32x605184xf32>
    %v6022 = stablehlo.constant dense<0.5> : tensor<32x605184xf32>
    %v6023 = stablehlo.multiply %v6022, %v6021 : tensor<32x605184xf32>
    %v6024 = stablehlo.negate %v6021 : tensor<32x605184xf32>
    %v6025 = stablehlo.constant dense<0.7071067811865476> : tensor<32x605184xf32>
    %v6026 = stablehlo.multiply %v6024, %v6025 : tensor<32x605184xf32>
    %v6027 = chlo.erfc %v6026 : tensor<32x605184xf32> -> tensor<32x605184xf32>
    %v6028 = stablehlo.multiply %v6023, %v6027 : tensor<32x605184xf32>
    %v6029 = stablehlo.reshape %v6028 : (tensor<32x605184xf32>) -> tensor<32x197x3072xf32>
    %v6030 = stablehlo.dot_general %v6029, %b11_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x3072xf32>, tensor<3072x768xf32>) -> tensor<32x197x768xf32>
    %v6031 = stablehlo.broadcast_in_dim %b11_bfc2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v6032 = stablehlo.add %v6030, %v6031 : tensor<32x197x768xf32>
    %v6033 = stablehlo.reshape %v6032 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v6034 = stablehlo.add %v5988, %v6033 : tensor<32x151296xf32>
    %v6035 = stablehlo.reshape %v6034 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6036 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6037 = stablehlo.constant dense<768.0> : tensor<32x197x768xf32>
    %v6038 = stablehlo.constant dense<1.0e-5> : tensor<32x197x768xf32>
    %v6039 = stablehlo.reduce(%v6035 init: %v6036) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6040 = stablehlo.broadcast_in_dim %v6039, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v6041 = stablehlo.divide %v6040, %v6037 : tensor<32x197x768xf32>
    %v6042 = stablehlo.subtract %v6035, %v6041 : tensor<32x197x768xf32>
    %v6043 = stablehlo.multiply %v6042, %v6042 : tensor<32x197x768xf32>
    %v6044 = stablehlo.reduce(%v6043 init: %v6036) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6045 = stablehlo.broadcast_in_dim %v6044, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v6046 = stablehlo.divide %v6045, %v6037 : tensor<32x197x768xf32>
    %v6047 = stablehlo.add %v6046, %v6038 : tensor<32x197x768xf32>
    %v6048 = stablehlo.rsqrt %v6047 : tensor<32x197x768xf32>
    %v6049 = stablehlo.multiply %v6042, %v6048 : tensor<32x197x768xf32>
    %v6050 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v6051 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v6052 = stablehlo.multiply %v6049, %v6050 : tensor<32x197x768xf32>
    %v6053 = stablehlo.add %v6052, %v6051 : tensor<32x197x768xf32>
    %v6054 = stablehlo.reshape %v6053 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v6055 = stablehlo.reshape %v6054 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6056 = stablehlo.broadcast_in_dim %gF, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v6057 = stablehlo.multiply %v6055, %v6056 : tensor<32x197x768xf32>
    %v6058 = stablehlo.reshape %v6057 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v6059 = stablehlo.reshape %v6058 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6060 = stablehlo.broadcast_in_dim %btF, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v6061 = stablehlo.add %v6059, %v6060 : tensor<32x197x768xf32>
    %v6062 = stablehlo.reshape %v6061 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v6063 = stablehlo.reshape %v6062 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6064 = stablehlo.slice %v6063 [0:32, 0:1, 0:768] : (tensor<32x197x768xf32>) -> tensor<32x1x768xf32>
    %v6065 = stablehlo.reshape %v6064 : (tensor<32x1x768xf32>) -> tensor<32x768xf32>
    %v6066 = stablehlo.dot_general %v6065, %Wc, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x768xf32>, tensor<768x1000xf32>) -> tensor<32x1000xf32>
    %v6067 = stablehlo.broadcast_in_dim %bc, dims = [1] : (tensor<1000xf32>) -> tensor<32x1000xf32>
    %v6068 = stablehlo.add %v6066, %v6067 : tensor<32x1000xf32>
    return %v6068 : tensor<32x1000xf32>
  }
}

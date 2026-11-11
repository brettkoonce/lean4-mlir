module @m {
  func.func @vitbin_emaaccdp2x256wxdropeps0000001smxerfbf16_fwd(%x: tensor<32x150528xf32>, %wConv: tensor<768x3x16x16xf32>, %bConv: tensor<768xf32>, %cls: tensor<768xf32>, %pos: tensor<197x768xf32>, %b0_g1: tensor<768xf32>, %b0_bt1: tensor<768xf32>, %b0_Wq: tensor<768x768xf32>, %b0_bq: tensor<768xf32>, %b0_Wk: tensor<768x768xf32>, %b0_bk: tensor<768xf32>, %b0_Wv: tensor<768x768xf32>, %b0_bv: tensor<768xf32>, %b0_Wo: tensor<768x768xf32>, %b0_bo: tensor<768xf32>, %b0_g2: tensor<768xf32>, %b0_bt2: tensor<768xf32>, %b0_Wfc1: tensor<768x3072xf32>, %b0_bfc1: tensor<3072xf32>, %b0_Wfc2: tensor<3072x768xf32>, %b0_bfc2: tensor<768xf32>, %b1_g1: tensor<768xf32>, %b1_bt1: tensor<768xf32>, %b1_Wq: tensor<768x768xf32>, %b1_bq: tensor<768xf32>, %b1_Wk: tensor<768x768xf32>, %b1_bk: tensor<768xf32>, %b1_Wv: tensor<768x768xf32>, %b1_bv: tensor<768xf32>, %b1_Wo: tensor<768x768xf32>, %b1_bo: tensor<768xf32>, %b1_g2: tensor<768xf32>, %b1_bt2: tensor<768xf32>, %b1_Wfc1: tensor<768x3072xf32>, %b1_bfc1: tensor<3072xf32>, %b1_Wfc2: tensor<3072x768xf32>, %b1_bfc2: tensor<768xf32>, %b2_g1: tensor<768xf32>, %b2_bt1: tensor<768xf32>, %b2_Wq: tensor<768x768xf32>, %b2_bq: tensor<768xf32>, %b2_Wk: tensor<768x768xf32>, %b2_bk: tensor<768xf32>, %b2_Wv: tensor<768x768xf32>, %b2_bv: tensor<768xf32>, %b2_Wo: tensor<768x768xf32>, %b2_bo: tensor<768xf32>, %b2_g2: tensor<768xf32>, %b2_bt2: tensor<768xf32>, %b2_Wfc1: tensor<768x3072xf32>, %b2_bfc1: tensor<3072xf32>, %b2_Wfc2: tensor<3072x768xf32>, %b2_bfc2: tensor<768xf32>, %b3_g1: tensor<768xf32>, %b3_bt1: tensor<768xf32>, %b3_Wq: tensor<768x768xf32>, %b3_bq: tensor<768xf32>, %b3_Wk: tensor<768x768xf32>, %b3_bk: tensor<768xf32>, %b3_Wv: tensor<768x768xf32>, %b3_bv: tensor<768xf32>, %b3_Wo: tensor<768x768xf32>, %b3_bo: tensor<768xf32>, %b3_g2: tensor<768xf32>, %b3_bt2: tensor<768xf32>, %b3_Wfc1: tensor<768x3072xf32>, %b3_bfc1: tensor<3072xf32>, %b3_Wfc2: tensor<3072x768xf32>, %b3_bfc2: tensor<768xf32>, %b4_g1: tensor<768xf32>, %b4_bt1: tensor<768xf32>, %b4_Wq: tensor<768x768xf32>, %b4_bq: tensor<768xf32>, %b4_Wk: tensor<768x768xf32>, %b4_bk: tensor<768xf32>, %b4_Wv: tensor<768x768xf32>, %b4_bv: tensor<768xf32>, %b4_Wo: tensor<768x768xf32>, %b4_bo: tensor<768xf32>, %b4_g2: tensor<768xf32>, %b4_bt2: tensor<768xf32>, %b4_Wfc1: tensor<768x3072xf32>, %b4_bfc1: tensor<3072xf32>, %b4_Wfc2: tensor<3072x768xf32>, %b4_bfc2: tensor<768xf32>, %b5_g1: tensor<768xf32>, %b5_bt1: tensor<768xf32>, %b5_Wq: tensor<768x768xf32>, %b5_bq: tensor<768xf32>, %b5_Wk: tensor<768x768xf32>, %b5_bk: tensor<768xf32>, %b5_Wv: tensor<768x768xf32>, %b5_bv: tensor<768xf32>, %b5_Wo: tensor<768x768xf32>, %b5_bo: tensor<768xf32>, %b5_g2: tensor<768xf32>, %b5_bt2: tensor<768xf32>, %b5_Wfc1: tensor<768x3072xf32>, %b5_bfc1: tensor<3072xf32>, %b5_Wfc2: tensor<3072x768xf32>, %b5_bfc2: tensor<768xf32>, %b6_g1: tensor<768xf32>, %b6_bt1: tensor<768xf32>, %b6_Wq: tensor<768x768xf32>, %b6_bq: tensor<768xf32>, %b6_Wk: tensor<768x768xf32>, %b6_bk: tensor<768xf32>, %b6_Wv: tensor<768x768xf32>, %b6_bv: tensor<768xf32>, %b6_Wo: tensor<768x768xf32>, %b6_bo: tensor<768xf32>, %b6_g2: tensor<768xf32>, %b6_bt2: tensor<768xf32>, %b6_Wfc1: tensor<768x3072xf32>, %b6_bfc1: tensor<3072xf32>, %b6_Wfc2: tensor<3072x768xf32>, %b6_bfc2: tensor<768xf32>, %b7_g1: tensor<768xf32>, %b7_bt1: tensor<768xf32>, %b7_Wq: tensor<768x768xf32>, %b7_bq: tensor<768xf32>, %b7_Wk: tensor<768x768xf32>, %b7_bk: tensor<768xf32>, %b7_Wv: tensor<768x768xf32>, %b7_bv: tensor<768xf32>, %b7_Wo: tensor<768x768xf32>, %b7_bo: tensor<768xf32>, %b7_g2: tensor<768xf32>, %b7_bt2: tensor<768xf32>, %b7_Wfc1: tensor<768x3072xf32>, %b7_bfc1: tensor<3072xf32>, %b7_Wfc2: tensor<3072x768xf32>, %b7_bfc2: tensor<768xf32>, %b8_g1: tensor<768xf32>, %b8_bt1: tensor<768xf32>, %b8_Wq: tensor<768x768xf32>, %b8_bq: tensor<768xf32>, %b8_Wk: tensor<768x768xf32>, %b8_bk: tensor<768xf32>, %b8_Wv: tensor<768x768xf32>, %b8_bv: tensor<768xf32>, %b8_Wo: tensor<768x768xf32>, %b8_bo: tensor<768xf32>, %b8_g2: tensor<768xf32>, %b8_bt2: tensor<768xf32>, %b8_Wfc1: tensor<768x3072xf32>, %b8_bfc1: tensor<3072xf32>, %b8_Wfc2: tensor<3072x768xf32>, %b8_bfc2: tensor<768xf32>, %b9_g1: tensor<768xf32>, %b9_bt1: tensor<768xf32>, %b9_Wq: tensor<768x768xf32>, %b9_bq: tensor<768xf32>, %b9_Wk: tensor<768x768xf32>, %b9_bk: tensor<768xf32>, %b9_Wv: tensor<768x768xf32>, %b9_bv: tensor<768xf32>, %b9_Wo: tensor<768x768xf32>, %b9_bo: tensor<768xf32>, %b9_g2: tensor<768xf32>, %b9_bt2: tensor<768xf32>, %b9_Wfc1: tensor<768x3072xf32>, %b9_bfc1: tensor<3072xf32>, %b9_Wfc2: tensor<3072x768xf32>, %b9_bfc2: tensor<768xf32>, %b10_g1: tensor<768xf32>, %b10_bt1: tensor<768xf32>, %b10_Wq: tensor<768x768xf32>, %b10_bq: tensor<768xf32>, %b10_Wk: tensor<768x768xf32>, %b10_bk: tensor<768xf32>, %b10_Wv: tensor<768x768xf32>, %b10_bv: tensor<768xf32>, %b10_Wo: tensor<768x768xf32>, %b10_bo: tensor<768xf32>, %b10_g2: tensor<768xf32>, %b10_bt2: tensor<768xf32>, %b10_Wfc1: tensor<768x3072xf32>, %b10_bfc1: tensor<3072xf32>, %b10_Wfc2: tensor<3072x768xf32>, %b10_bfc2: tensor<768xf32>, %b11_g1: tensor<768xf32>, %b11_bt1: tensor<768xf32>, %b11_Wq: tensor<768x768xf32>, %b11_bq: tensor<768xf32>, %b11_Wk: tensor<768x768xf32>, %b11_bk: tensor<768xf32>, %b11_Wv: tensor<768x768xf32>, %b11_bv: tensor<768xf32>, %b11_Wo: tensor<768x768xf32>, %b11_bo: tensor<768xf32>, %b11_g2: tensor<768xf32>, %b11_bt2: tensor<768xf32>, %b11_Wfc1: tensor<768x3072xf32>, %b11_bfc1: tensor<3072xf32>, %b11_Wfc2: tensor<3072x768xf32>, %b11_bfc2: tensor<768xf32>, %gF: tensor<768xf32>, %btF: tensor<768xf32>, %Wc: tensor<768x1000xf32>, %bc: tensor<1000xf32>) -> tensor<32x1000xf32> {
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
    %v14 = stablehlo.constant dense<1.0e-6> : tensor<32x197x768xf32>
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
    %v73 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v74 = stablehlo.reduce(%v72 init: %v73) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v75 = stablehlo.broadcast_in_dim %v74, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v76 = stablehlo.subtract %v72, %v75 : tensor<32x197x197xf32>
    %v77 = stablehlo.constant dense<0.0> : tensor<f32>
    %v78 = stablehlo.exponential %v76 : tensor<32x197x197xf32>
    %v79 = stablehlo.reduce(%v78 init: %v77) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v80 = stablehlo.broadcast_in_dim %v79, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v81 = stablehlo.divide %v78, %v80 : tensor<32x197x197xf32>
    %v82 = stablehlo.reshape %v81 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v83 = stablehlo.reshape %v82 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v84 = stablehlo.reshape %v62 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v85 = stablehlo.dot_general %v83, %v84, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v86 = stablehlo.reshape %v85 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v87 = stablehlo.reshape %v86 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v88 = stablehlo.constant dense<0.0> : tensor<f32>
    %v89 = stablehlo.pad %v87, %v88, low = [0, 0, 0], high = [0, 0, 704], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v90 = stablehlo.reshape %v89 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v91 = stablehlo.reshape %v43 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v92 = stablehlo.slice %v91 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v93 = stablehlo.reshape %v92 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v94 = stablehlo.reshape %v48 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v95 = stablehlo.slice %v94 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v96 = stablehlo.reshape %v95 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v97 = stablehlo.reshape %v53 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v98 = stablehlo.slice %v97 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v99 = stablehlo.reshape %v98 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v100 = stablehlo.reshape %v96 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v101 = stablehlo.transpose %v100, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v102 = stablehlo.reshape %v101 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v103 = stablehlo.reshape %v93 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v104 = stablehlo.reshape %v102 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v105 = stablehlo.dot_general %v103, %v104, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v106 = stablehlo.reshape %v105 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v107 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v108 = stablehlo.multiply %v106, %v107 : tensor<32x38809xf32>
    %v109 = stablehlo.reshape %v108 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v110 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v111 = stablehlo.reduce(%v109 init: %v110) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v112 = stablehlo.broadcast_in_dim %v111, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v113 = stablehlo.subtract %v109, %v112 : tensor<32x197x197xf32>
    %v114 = stablehlo.constant dense<0.0> : tensor<f32>
    %v115 = stablehlo.exponential %v113 : tensor<32x197x197xf32>
    %v116 = stablehlo.reduce(%v115 init: %v114) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v117 = stablehlo.broadcast_in_dim %v116, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v118 = stablehlo.divide %v115, %v117 : tensor<32x197x197xf32>
    %v119 = stablehlo.reshape %v118 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v120 = stablehlo.reshape %v119 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v121 = stablehlo.reshape %v99 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v122 = stablehlo.dot_general %v120, %v121, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v123 = stablehlo.reshape %v122 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v124 = stablehlo.reshape %v123 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v125 = stablehlo.constant dense<0.0> : tensor<f32>
    %v126 = stablehlo.pad %v124, %v125, low = [0, 0, 64], high = [0, 0, 640], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v127 = stablehlo.reshape %v126 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v128 = stablehlo.add %v90, %v127 : tensor<32x151296xf32>
    %v129 = stablehlo.reshape %v43 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v130 = stablehlo.slice %v129 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v131 = stablehlo.reshape %v130 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v132 = stablehlo.reshape %v48 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v133 = stablehlo.slice %v132 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v134 = stablehlo.reshape %v133 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v135 = stablehlo.reshape %v53 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v136 = stablehlo.slice %v135 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v137 = stablehlo.reshape %v136 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v138 = stablehlo.reshape %v134 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v139 = stablehlo.transpose %v138, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v140 = stablehlo.reshape %v139 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v141 = stablehlo.reshape %v131 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v142 = stablehlo.reshape %v140 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v143 = stablehlo.dot_general %v141, %v142, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v144 = stablehlo.reshape %v143 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v145 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v146 = stablehlo.multiply %v144, %v145 : tensor<32x38809xf32>
    %v147 = stablehlo.reshape %v146 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v148 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v149 = stablehlo.reduce(%v147 init: %v148) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v150 = stablehlo.broadcast_in_dim %v149, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v151 = stablehlo.subtract %v147, %v150 : tensor<32x197x197xf32>
    %v152 = stablehlo.constant dense<0.0> : tensor<f32>
    %v153 = stablehlo.exponential %v151 : tensor<32x197x197xf32>
    %v154 = stablehlo.reduce(%v153 init: %v152) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v155 = stablehlo.broadcast_in_dim %v154, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v156 = stablehlo.divide %v153, %v155 : tensor<32x197x197xf32>
    %v157 = stablehlo.reshape %v156 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v158 = stablehlo.reshape %v157 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v159 = stablehlo.reshape %v137 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v160 = stablehlo.dot_general %v158, %v159, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v161 = stablehlo.reshape %v160 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v162 = stablehlo.reshape %v161 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v163 = stablehlo.constant dense<0.0> : tensor<f32>
    %v164 = stablehlo.pad %v162, %v163, low = [0, 0, 128], high = [0, 0, 576], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v165 = stablehlo.reshape %v164 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v166 = stablehlo.add %v128, %v165 : tensor<32x151296xf32>
    %v167 = stablehlo.reshape %v43 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v168 = stablehlo.slice %v167 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v169 = stablehlo.reshape %v168 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v170 = stablehlo.reshape %v48 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v171 = stablehlo.slice %v170 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v172 = stablehlo.reshape %v171 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v173 = stablehlo.reshape %v53 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v174 = stablehlo.slice %v173 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v175 = stablehlo.reshape %v174 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v176 = stablehlo.reshape %v172 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v177 = stablehlo.transpose %v176, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v178 = stablehlo.reshape %v177 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v179 = stablehlo.reshape %v169 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v180 = stablehlo.reshape %v178 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v181 = stablehlo.dot_general %v179, %v180, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v182 = stablehlo.reshape %v181 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v183 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v184 = stablehlo.multiply %v182, %v183 : tensor<32x38809xf32>
    %v185 = stablehlo.reshape %v184 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v186 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v187 = stablehlo.reduce(%v185 init: %v186) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v188 = stablehlo.broadcast_in_dim %v187, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v189 = stablehlo.subtract %v185, %v188 : tensor<32x197x197xf32>
    %v190 = stablehlo.constant dense<0.0> : tensor<f32>
    %v191 = stablehlo.exponential %v189 : tensor<32x197x197xf32>
    %v192 = stablehlo.reduce(%v191 init: %v190) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v193 = stablehlo.broadcast_in_dim %v192, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v194 = stablehlo.divide %v191, %v193 : tensor<32x197x197xf32>
    %v195 = stablehlo.reshape %v194 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v196 = stablehlo.reshape %v195 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v197 = stablehlo.reshape %v175 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v198 = stablehlo.dot_general %v196, %v197, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v199 = stablehlo.reshape %v198 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v200 = stablehlo.reshape %v199 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v201 = stablehlo.constant dense<0.0> : tensor<f32>
    %v202 = stablehlo.pad %v200, %v201, low = [0, 0, 192], high = [0, 0, 512], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v203 = stablehlo.reshape %v202 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v204 = stablehlo.add %v166, %v203 : tensor<32x151296xf32>
    %v205 = stablehlo.reshape %v43 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v206 = stablehlo.slice %v205 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v207 = stablehlo.reshape %v206 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v208 = stablehlo.reshape %v48 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v209 = stablehlo.slice %v208 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v210 = stablehlo.reshape %v209 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v211 = stablehlo.reshape %v53 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v212 = stablehlo.slice %v211 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v213 = stablehlo.reshape %v212 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v214 = stablehlo.reshape %v210 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v215 = stablehlo.transpose %v214, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v216 = stablehlo.reshape %v215 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v217 = stablehlo.reshape %v207 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v218 = stablehlo.reshape %v216 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v219 = stablehlo.dot_general %v217, %v218, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v220 = stablehlo.reshape %v219 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v221 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v222 = stablehlo.multiply %v220, %v221 : tensor<32x38809xf32>
    %v223 = stablehlo.reshape %v222 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v224 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v225 = stablehlo.reduce(%v223 init: %v224) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v226 = stablehlo.broadcast_in_dim %v225, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v227 = stablehlo.subtract %v223, %v226 : tensor<32x197x197xf32>
    %v228 = stablehlo.constant dense<0.0> : tensor<f32>
    %v229 = stablehlo.exponential %v227 : tensor<32x197x197xf32>
    %v230 = stablehlo.reduce(%v229 init: %v228) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v231 = stablehlo.broadcast_in_dim %v230, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v232 = stablehlo.divide %v229, %v231 : tensor<32x197x197xf32>
    %v233 = stablehlo.reshape %v232 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v234 = stablehlo.reshape %v233 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v235 = stablehlo.reshape %v213 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v236 = stablehlo.dot_general %v234, %v235, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v237 = stablehlo.reshape %v236 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v238 = stablehlo.reshape %v237 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v239 = stablehlo.constant dense<0.0> : tensor<f32>
    %v240 = stablehlo.pad %v238, %v239, low = [0, 0, 256], high = [0, 0, 448], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v241 = stablehlo.reshape %v240 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v242 = stablehlo.add %v204, %v241 : tensor<32x151296xf32>
    %v243 = stablehlo.reshape %v43 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v244 = stablehlo.slice %v243 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v245 = stablehlo.reshape %v244 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v246 = stablehlo.reshape %v48 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v247 = stablehlo.slice %v246 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v248 = stablehlo.reshape %v247 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v249 = stablehlo.reshape %v53 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v250 = stablehlo.slice %v249 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v251 = stablehlo.reshape %v250 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v252 = stablehlo.reshape %v248 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v253 = stablehlo.transpose %v252, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v254 = stablehlo.reshape %v253 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v255 = stablehlo.reshape %v245 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v256 = stablehlo.reshape %v254 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v257 = stablehlo.dot_general %v255, %v256, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v258 = stablehlo.reshape %v257 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v259 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v260 = stablehlo.multiply %v258, %v259 : tensor<32x38809xf32>
    %v261 = stablehlo.reshape %v260 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v262 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v263 = stablehlo.reduce(%v261 init: %v262) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v264 = stablehlo.broadcast_in_dim %v263, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v265 = stablehlo.subtract %v261, %v264 : tensor<32x197x197xf32>
    %v266 = stablehlo.constant dense<0.0> : tensor<f32>
    %v267 = stablehlo.exponential %v265 : tensor<32x197x197xf32>
    %v268 = stablehlo.reduce(%v267 init: %v266) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v269 = stablehlo.broadcast_in_dim %v268, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v270 = stablehlo.divide %v267, %v269 : tensor<32x197x197xf32>
    %v271 = stablehlo.reshape %v270 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v272 = stablehlo.reshape %v271 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v273 = stablehlo.reshape %v251 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v274 = stablehlo.dot_general %v272, %v273, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v275 = stablehlo.reshape %v274 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v276 = stablehlo.reshape %v275 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v277 = stablehlo.constant dense<0.0> : tensor<f32>
    %v278 = stablehlo.pad %v276, %v277, low = [0, 0, 320], high = [0, 0, 384], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v279 = stablehlo.reshape %v278 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v280 = stablehlo.add %v242, %v279 : tensor<32x151296xf32>
    %v281 = stablehlo.reshape %v43 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v282 = stablehlo.slice %v281 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v283 = stablehlo.reshape %v282 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v284 = stablehlo.reshape %v48 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v285 = stablehlo.slice %v284 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v286 = stablehlo.reshape %v285 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v287 = stablehlo.reshape %v53 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v288 = stablehlo.slice %v287 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v289 = stablehlo.reshape %v288 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v290 = stablehlo.reshape %v286 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v291 = stablehlo.transpose %v290, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v292 = stablehlo.reshape %v291 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v293 = stablehlo.reshape %v283 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v294 = stablehlo.reshape %v292 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v295 = stablehlo.dot_general %v293, %v294, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v296 = stablehlo.reshape %v295 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v297 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v298 = stablehlo.multiply %v296, %v297 : tensor<32x38809xf32>
    %v299 = stablehlo.reshape %v298 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v300 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v301 = stablehlo.reduce(%v299 init: %v300) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v302 = stablehlo.broadcast_in_dim %v301, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v303 = stablehlo.subtract %v299, %v302 : tensor<32x197x197xf32>
    %v304 = stablehlo.constant dense<0.0> : tensor<f32>
    %v305 = stablehlo.exponential %v303 : tensor<32x197x197xf32>
    %v306 = stablehlo.reduce(%v305 init: %v304) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v307 = stablehlo.broadcast_in_dim %v306, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v308 = stablehlo.divide %v305, %v307 : tensor<32x197x197xf32>
    %v309 = stablehlo.reshape %v308 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v310 = stablehlo.reshape %v309 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v311 = stablehlo.reshape %v289 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v312 = stablehlo.dot_general %v310, %v311, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v313 = stablehlo.reshape %v312 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v314 = stablehlo.reshape %v313 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v315 = stablehlo.constant dense<0.0> : tensor<f32>
    %v316 = stablehlo.pad %v314, %v315, low = [0, 0, 384], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v317 = stablehlo.reshape %v316 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v318 = stablehlo.add %v280, %v317 : tensor<32x151296xf32>
    %v319 = stablehlo.reshape %v43 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v320 = stablehlo.slice %v319 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v321 = stablehlo.reshape %v320 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v322 = stablehlo.reshape %v48 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v323 = stablehlo.slice %v322 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v324 = stablehlo.reshape %v323 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v325 = stablehlo.reshape %v53 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v326 = stablehlo.slice %v325 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v327 = stablehlo.reshape %v326 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v328 = stablehlo.reshape %v324 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v329 = stablehlo.transpose %v328, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v330 = stablehlo.reshape %v329 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v331 = stablehlo.reshape %v321 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v332 = stablehlo.reshape %v330 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v333 = stablehlo.dot_general %v331, %v332, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v334 = stablehlo.reshape %v333 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v335 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v336 = stablehlo.multiply %v334, %v335 : tensor<32x38809xf32>
    %v337 = stablehlo.reshape %v336 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v338 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v339 = stablehlo.reduce(%v337 init: %v338) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v340 = stablehlo.broadcast_in_dim %v339, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v341 = stablehlo.subtract %v337, %v340 : tensor<32x197x197xf32>
    %v342 = stablehlo.constant dense<0.0> : tensor<f32>
    %v343 = stablehlo.exponential %v341 : tensor<32x197x197xf32>
    %v344 = stablehlo.reduce(%v343 init: %v342) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v345 = stablehlo.broadcast_in_dim %v344, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v346 = stablehlo.divide %v343, %v345 : tensor<32x197x197xf32>
    %v347 = stablehlo.reshape %v346 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v348 = stablehlo.reshape %v347 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v349 = stablehlo.reshape %v327 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v350 = stablehlo.dot_general %v348, %v349, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v351 = stablehlo.reshape %v350 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v352 = stablehlo.reshape %v351 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v353 = stablehlo.constant dense<0.0> : tensor<f32>
    %v354 = stablehlo.pad %v352, %v353, low = [0, 0, 448], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v355 = stablehlo.reshape %v354 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v356 = stablehlo.add %v318, %v355 : tensor<32x151296xf32>
    %v357 = stablehlo.reshape %v43 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v358 = stablehlo.slice %v357 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v359 = stablehlo.reshape %v358 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v360 = stablehlo.reshape %v48 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v361 = stablehlo.slice %v360 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v362 = stablehlo.reshape %v361 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v363 = stablehlo.reshape %v53 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v364 = stablehlo.slice %v363 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v365 = stablehlo.reshape %v364 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v366 = stablehlo.reshape %v362 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v367 = stablehlo.transpose %v366, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v368 = stablehlo.reshape %v367 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v369 = stablehlo.reshape %v359 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v370 = stablehlo.reshape %v368 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v371 = stablehlo.dot_general %v369, %v370, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v372 = stablehlo.reshape %v371 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v373 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v374 = stablehlo.multiply %v372, %v373 : tensor<32x38809xf32>
    %v375 = stablehlo.reshape %v374 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v376 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v377 = stablehlo.reduce(%v375 init: %v376) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v378 = stablehlo.broadcast_in_dim %v377, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v379 = stablehlo.subtract %v375, %v378 : tensor<32x197x197xf32>
    %v380 = stablehlo.constant dense<0.0> : tensor<f32>
    %v381 = stablehlo.exponential %v379 : tensor<32x197x197xf32>
    %v382 = stablehlo.reduce(%v381 init: %v380) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v383 = stablehlo.broadcast_in_dim %v382, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v384 = stablehlo.divide %v381, %v383 : tensor<32x197x197xf32>
    %v385 = stablehlo.reshape %v384 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v386 = stablehlo.reshape %v385 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v387 = stablehlo.reshape %v365 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v388 = stablehlo.dot_general %v386, %v387, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v389 = stablehlo.reshape %v388 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v390 = stablehlo.reshape %v389 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v391 = stablehlo.constant dense<0.0> : tensor<f32>
    %v392 = stablehlo.pad %v390, %v391, low = [0, 0, 512], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v393 = stablehlo.reshape %v392 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v394 = stablehlo.add %v356, %v393 : tensor<32x151296xf32>
    %v395 = stablehlo.reshape %v43 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v396 = stablehlo.slice %v395 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v397 = stablehlo.reshape %v396 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v398 = stablehlo.reshape %v48 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v399 = stablehlo.slice %v398 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v400 = stablehlo.reshape %v399 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v401 = stablehlo.reshape %v53 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v402 = stablehlo.slice %v401 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v403 = stablehlo.reshape %v402 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v404 = stablehlo.reshape %v400 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v405 = stablehlo.transpose %v404, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v406 = stablehlo.reshape %v405 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v407 = stablehlo.reshape %v397 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v408 = stablehlo.reshape %v406 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v409 = stablehlo.dot_general %v407, %v408, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v410 = stablehlo.reshape %v409 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v411 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v412 = stablehlo.multiply %v410, %v411 : tensor<32x38809xf32>
    %v413 = stablehlo.reshape %v412 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v414 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v415 = stablehlo.reduce(%v413 init: %v414) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v416 = stablehlo.broadcast_in_dim %v415, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v417 = stablehlo.subtract %v413, %v416 : tensor<32x197x197xf32>
    %v418 = stablehlo.constant dense<0.0> : tensor<f32>
    %v419 = stablehlo.exponential %v417 : tensor<32x197x197xf32>
    %v420 = stablehlo.reduce(%v419 init: %v418) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v421 = stablehlo.broadcast_in_dim %v420, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v422 = stablehlo.divide %v419, %v421 : tensor<32x197x197xf32>
    %v423 = stablehlo.reshape %v422 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v424 = stablehlo.reshape %v423 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v425 = stablehlo.reshape %v403 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v426 = stablehlo.dot_general %v424, %v425, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v427 = stablehlo.reshape %v426 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v428 = stablehlo.reshape %v427 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v429 = stablehlo.constant dense<0.0> : tensor<f32>
    %v430 = stablehlo.pad %v428, %v429, low = [0, 0, 576], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v431 = stablehlo.reshape %v430 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v432 = stablehlo.add %v394, %v431 : tensor<32x151296xf32>
    %v433 = stablehlo.reshape %v43 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v434 = stablehlo.slice %v433 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v435 = stablehlo.reshape %v434 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v436 = stablehlo.reshape %v48 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v437 = stablehlo.slice %v436 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v438 = stablehlo.reshape %v437 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v439 = stablehlo.reshape %v53 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v440 = stablehlo.slice %v439 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v441 = stablehlo.reshape %v440 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v442 = stablehlo.reshape %v438 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v443 = stablehlo.transpose %v442, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v444 = stablehlo.reshape %v443 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v445 = stablehlo.reshape %v435 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v446 = stablehlo.reshape %v444 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v447 = stablehlo.dot_general %v445, %v446, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v448 = stablehlo.reshape %v447 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v449 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v450 = stablehlo.multiply %v448, %v449 : tensor<32x38809xf32>
    %v451 = stablehlo.reshape %v450 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v452 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v453 = stablehlo.reduce(%v451 init: %v452) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v454 = stablehlo.broadcast_in_dim %v453, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v455 = stablehlo.subtract %v451, %v454 : tensor<32x197x197xf32>
    %v456 = stablehlo.constant dense<0.0> : tensor<f32>
    %v457 = stablehlo.exponential %v455 : tensor<32x197x197xf32>
    %v458 = stablehlo.reduce(%v457 init: %v456) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v459 = stablehlo.broadcast_in_dim %v458, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v460 = stablehlo.divide %v457, %v459 : tensor<32x197x197xf32>
    %v461 = stablehlo.reshape %v460 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v462 = stablehlo.reshape %v461 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v463 = stablehlo.reshape %v441 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v464 = stablehlo.dot_general %v462, %v463, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v465 = stablehlo.reshape %v464 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v466 = stablehlo.reshape %v465 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v467 = stablehlo.constant dense<0.0> : tensor<f32>
    %v468 = stablehlo.pad %v466, %v467, low = [0, 0, 640], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v469 = stablehlo.reshape %v468 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v470 = stablehlo.add %v432, %v469 : tensor<32x151296xf32>
    %v471 = stablehlo.reshape %v43 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v472 = stablehlo.slice %v471 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v473 = stablehlo.reshape %v472 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v474 = stablehlo.reshape %v48 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v475 = stablehlo.slice %v474 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v476 = stablehlo.reshape %v475 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v477 = stablehlo.reshape %v53 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v478 = stablehlo.slice %v477 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v479 = stablehlo.reshape %v478 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v480 = stablehlo.reshape %v476 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v481 = stablehlo.transpose %v480, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v482 = stablehlo.reshape %v481 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v483 = stablehlo.reshape %v473 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v484 = stablehlo.reshape %v482 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v485 = stablehlo.dot_general %v483, %v484, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v486 = stablehlo.reshape %v485 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v487 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v488 = stablehlo.multiply %v486, %v487 : tensor<32x38809xf32>
    %v489 = stablehlo.reshape %v488 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v490 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v491 = stablehlo.reduce(%v489 init: %v490) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v492 = stablehlo.broadcast_in_dim %v491, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v493 = stablehlo.subtract %v489, %v492 : tensor<32x197x197xf32>
    %v494 = stablehlo.constant dense<0.0> : tensor<f32>
    %v495 = stablehlo.exponential %v493 : tensor<32x197x197xf32>
    %v496 = stablehlo.reduce(%v495 init: %v494) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v497 = stablehlo.broadcast_in_dim %v496, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v498 = stablehlo.divide %v495, %v497 : tensor<32x197x197xf32>
    %v499 = stablehlo.reshape %v498 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v500 = stablehlo.reshape %v499 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v501 = stablehlo.reshape %v479 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v502 = stablehlo.dot_general %v500, %v501, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v503 = stablehlo.reshape %v502 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v504 = stablehlo.reshape %v503 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v505 = stablehlo.constant dense<0.0> : tensor<f32>
    %v506 = stablehlo.pad %v504, %v505, low = [0, 0, 704], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v507 = stablehlo.reshape %v506 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v508 = stablehlo.add %v470, %v507 : tensor<32x151296xf32>
    %v509 = stablehlo.reshape %v508 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v510 = stablehlo.dot_general %v509, %b0_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v511 = stablehlo.broadcast_in_dim %b0_bo, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v512 = stablehlo.add %v510, %v511 : tensor<32x197x768xf32>
    %v513 = stablehlo.reshape %v512 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v514 = stablehlo.add %v10, %v513 : tensor<32x151296xf32>
    %v515 = stablehlo.reshape %v514 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v516 = stablehlo.constant dense<0.0> : tensor<f32>
    %v517 = stablehlo.constant dense<768.0> : tensor<32x197x768xf32>
    %v518 = stablehlo.constant dense<1.0e-6> : tensor<32x197x768xf32>
    %v519 = stablehlo.reduce(%v515 init: %v516) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v520 = stablehlo.broadcast_in_dim %v519, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v521 = stablehlo.divide %v520, %v517 : tensor<32x197x768xf32>
    %v522 = stablehlo.subtract %v515, %v521 : tensor<32x197x768xf32>
    %v523 = stablehlo.multiply %v522, %v522 : tensor<32x197x768xf32>
    %v524 = stablehlo.reduce(%v523 init: %v516) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v525 = stablehlo.broadcast_in_dim %v524, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v526 = stablehlo.divide %v525, %v517 : tensor<32x197x768xf32>
    %v527 = stablehlo.add %v526, %v518 : tensor<32x197x768xf32>
    %v528 = stablehlo.rsqrt %v527 : tensor<32x197x768xf32>
    %v529 = stablehlo.multiply %v522, %v528 : tensor<32x197x768xf32>
    %v530 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v531 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v532 = stablehlo.multiply %v529, %v530 : tensor<32x197x768xf32>
    %v533 = stablehlo.add %v532, %v531 : tensor<32x197x768xf32>
    %v534 = stablehlo.reshape %v533 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v535 = stablehlo.reshape %v534 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v536 = stablehlo.broadcast_in_dim %b0_g2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v537 = stablehlo.multiply %v535, %v536 : tensor<32x197x768xf32>
    %v538 = stablehlo.reshape %v537 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v539 = stablehlo.reshape %v538 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v540 = stablehlo.broadcast_in_dim %b0_bt2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v541 = stablehlo.add %v539, %v540 : tensor<32x197x768xf32>
    %v542 = stablehlo.reshape %v541 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v543 = stablehlo.reshape %v542 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v544 = stablehlo.dot_general %v543, %b0_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x3072xf32>) -> tensor<32x197x3072xf32>
    %v545 = stablehlo.broadcast_in_dim %b0_bfc1, dims = [2] : (tensor<3072xf32>) -> tensor<32x197x3072xf32>
    %v546 = stablehlo.add %v544, %v545 : tensor<32x197x3072xf32>
    %v547 = stablehlo.reshape %v546 : (tensor<32x197x3072xf32>) -> tensor<32x605184xf32>
    %v548 = stablehlo.constant dense<0.5> : tensor<32x605184xf32>
    %v549 = stablehlo.multiply %v548, %v547 : tensor<32x605184xf32>
    %v550 = stablehlo.negate %v547 : tensor<32x605184xf32>
    %v551 = stablehlo.constant dense<0.7071067811865476> : tensor<32x605184xf32>
    %v552 = stablehlo.multiply %v550, %v551 : tensor<32x605184xf32>
    %v553 = chlo.erfc %v552 : tensor<32x605184xf32> -> tensor<32x605184xf32>
    %v554 = stablehlo.multiply %v549, %v553 : tensor<32x605184xf32>
    %v555 = stablehlo.reshape %v554 : (tensor<32x605184xf32>) -> tensor<32x197x3072xf32>
    %v556 = stablehlo.dot_general %v555, %b0_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x3072xf32>, tensor<3072x768xf32>) -> tensor<32x197x768xf32>
    %v557 = stablehlo.broadcast_in_dim %b0_bfc2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v558 = stablehlo.add %v556, %v557 : tensor<32x197x768xf32>
    %v559 = stablehlo.reshape %v558 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v560 = stablehlo.add %v514, %v559 : tensor<32x151296xf32>
    %v561 = stablehlo.reshape %v560 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v562 = stablehlo.constant dense<0.0> : tensor<f32>
    %v563 = stablehlo.constant dense<768.0> : tensor<32x197x768xf32>
    %v564 = stablehlo.constant dense<1.0e-6> : tensor<32x197x768xf32>
    %v565 = stablehlo.reduce(%v561 init: %v562) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v566 = stablehlo.broadcast_in_dim %v565, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v567 = stablehlo.divide %v566, %v563 : tensor<32x197x768xf32>
    %v568 = stablehlo.subtract %v561, %v567 : tensor<32x197x768xf32>
    %v569 = stablehlo.multiply %v568, %v568 : tensor<32x197x768xf32>
    %v570 = stablehlo.reduce(%v569 init: %v562) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v571 = stablehlo.broadcast_in_dim %v570, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v572 = stablehlo.divide %v571, %v563 : tensor<32x197x768xf32>
    %v573 = stablehlo.add %v572, %v564 : tensor<32x197x768xf32>
    %v574 = stablehlo.rsqrt %v573 : tensor<32x197x768xf32>
    %v575 = stablehlo.multiply %v568, %v574 : tensor<32x197x768xf32>
    %v576 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v577 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v578 = stablehlo.multiply %v575, %v576 : tensor<32x197x768xf32>
    %v579 = stablehlo.add %v578, %v577 : tensor<32x197x768xf32>
    %v580 = stablehlo.reshape %v579 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v581 = stablehlo.reshape %v580 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v582 = stablehlo.broadcast_in_dim %b1_g1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v583 = stablehlo.multiply %v581, %v582 : tensor<32x197x768xf32>
    %v584 = stablehlo.reshape %v583 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v585 = stablehlo.reshape %v584 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v586 = stablehlo.broadcast_in_dim %b1_bt1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v587 = stablehlo.add %v585, %v586 : tensor<32x197x768xf32>
    %v588 = stablehlo.reshape %v587 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v589 = stablehlo.reshape %v588 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v590 = stablehlo.dot_general %v589, %b1_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v591 = stablehlo.broadcast_in_dim %b1_bq, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v592 = stablehlo.add %v590, %v591 : tensor<32x197x768xf32>
    %v593 = stablehlo.reshape %v592 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v594 = stablehlo.reshape %v588 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v595 = stablehlo.dot_general %v594, %b1_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v596 = stablehlo.broadcast_in_dim %b1_bk, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v597 = stablehlo.add %v595, %v596 : tensor<32x197x768xf32>
    %v598 = stablehlo.reshape %v597 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v599 = stablehlo.reshape %v588 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v600 = stablehlo.dot_general %v599, %b1_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v601 = stablehlo.broadcast_in_dim %b1_bv, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v602 = stablehlo.add %v600, %v601 : tensor<32x197x768xf32>
    %v603 = stablehlo.reshape %v602 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v604 = stablehlo.reshape %v593 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v605 = stablehlo.slice %v604 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v606 = stablehlo.reshape %v605 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v607 = stablehlo.reshape %v598 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v608 = stablehlo.slice %v607 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v609 = stablehlo.reshape %v608 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v610 = stablehlo.reshape %v603 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v611 = stablehlo.slice %v610 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v612 = stablehlo.reshape %v611 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v613 = stablehlo.reshape %v609 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v614 = stablehlo.transpose %v613, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v615 = stablehlo.reshape %v614 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v616 = stablehlo.reshape %v606 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v617 = stablehlo.reshape %v615 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v618 = stablehlo.dot_general %v616, %v617, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v619 = stablehlo.reshape %v618 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v620 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v621 = stablehlo.multiply %v619, %v620 : tensor<32x38809xf32>
    %v622 = stablehlo.reshape %v621 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v623 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v624 = stablehlo.reduce(%v622 init: %v623) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v625 = stablehlo.broadcast_in_dim %v624, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v626 = stablehlo.subtract %v622, %v625 : tensor<32x197x197xf32>
    %v627 = stablehlo.constant dense<0.0> : tensor<f32>
    %v628 = stablehlo.exponential %v626 : tensor<32x197x197xf32>
    %v629 = stablehlo.reduce(%v628 init: %v627) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v630 = stablehlo.broadcast_in_dim %v629, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v631 = stablehlo.divide %v628, %v630 : tensor<32x197x197xf32>
    %v632 = stablehlo.reshape %v631 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v633 = stablehlo.reshape %v632 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v634 = stablehlo.reshape %v612 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v635 = stablehlo.dot_general %v633, %v634, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v636 = stablehlo.reshape %v635 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v637 = stablehlo.reshape %v636 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v638 = stablehlo.constant dense<0.0> : tensor<f32>
    %v639 = stablehlo.pad %v637, %v638, low = [0, 0, 0], high = [0, 0, 704], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v640 = stablehlo.reshape %v639 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v641 = stablehlo.reshape %v593 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v642 = stablehlo.slice %v641 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v643 = stablehlo.reshape %v642 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v644 = stablehlo.reshape %v598 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v645 = stablehlo.slice %v644 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v646 = stablehlo.reshape %v645 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v647 = stablehlo.reshape %v603 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v648 = stablehlo.slice %v647 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v649 = stablehlo.reshape %v648 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v650 = stablehlo.reshape %v646 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v651 = stablehlo.transpose %v650, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v652 = stablehlo.reshape %v651 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v653 = stablehlo.reshape %v643 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v654 = stablehlo.reshape %v652 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v655 = stablehlo.dot_general %v653, %v654, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v656 = stablehlo.reshape %v655 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v657 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v658 = stablehlo.multiply %v656, %v657 : tensor<32x38809xf32>
    %v659 = stablehlo.reshape %v658 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v660 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v661 = stablehlo.reduce(%v659 init: %v660) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v662 = stablehlo.broadcast_in_dim %v661, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v663 = stablehlo.subtract %v659, %v662 : tensor<32x197x197xf32>
    %v664 = stablehlo.constant dense<0.0> : tensor<f32>
    %v665 = stablehlo.exponential %v663 : tensor<32x197x197xf32>
    %v666 = stablehlo.reduce(%v665 init: %v664) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v667 = stablehlo.broadcast_in_dim %v666, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v668 = stablehlo.divide %v665, %v667 : tensor<32x197x197xf32>
    %v669 = stablehlo.reshape %v668 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v670 = stablehlo.reshape %v669 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v671 = stablehlo.reshape %v649 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v672 = stablehlo.dot_general %v670, %v671, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v673 = stablehlo.reshape %v672 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v674 = stablehlo.reshape %v673 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v675 = stablehlo.constant dense<0.0> : tensor<f32>
    %v676 = stablehlo.pad %v674, %v675, low = [0, 0, 64], high = [0, 0, 640], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v677 = stablehlo.reshape %v676 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v678 = stablehlo.add %v640, %v677 : tensor<32x151296xf32>
    %v679 = stablehlo.reshape %v593 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v680 = stablehlo.slice %v679 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v681 = stablehlo.reshape %v680 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v682 = stablehlo.reshape %v598 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v683 = stablehlo.slice %v682 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v684 = stablehlo.reshape %v683 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v685 = stablehlo.reshape %v603 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v686 = stablehlo.slice %v685 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v687 = stablehlo.reshape %v686 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v688 = stablehlo.reshape %v684 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v689 = stablehlo.transpose %v688, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v690 = stablehlo.reshape %v689 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v691 = stablehlo.reshape %v681 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v692 = stablehlo.reshape %v690 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v693 = stablehlo.dot_general %v691, %v692, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v694 = stablehlo.reshape %v693 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v695 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v696 = stablehlo.multiply %v694, %v695 : tensor<32x38809xf32>
    %v697 = stablehlo.reshape %v696 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v698 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v699 = stablehlo.reduce(%v697 init: %v698) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v700 = stablehlo.broadcast_in_dim %v699, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v701 = stablehlo.subtract %v697, %v700 : tensor<32x197x197xf32>
    %v702 = stablehlo.constant dense<0.0> : tensor<f32>
    %v703 = stablehlo.exponential %v701 : tensor<32x197x197xf32>
    %v704 = stablehlo.reduce(%v703 init: %v702) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v705 = stablehlo.broadcast_in_dim %v704, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v706 = stablehlo.divide %v703, %v705 : tensor<32x197x197xf32>
    %v707 = stablehlo.reshape %v706 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v708 = stablehlo.reshape %v707 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v709 = stablehlo.reshape %v687 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v710 = stablehlo.dot_general %v708, %v709, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v711 = stablehlo.reshape %v710 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v712 = stablehlo.reshape %v711 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v713 = stablehlo.constant dense<0.0> : tensor<f32>
    %v714 = stablehlo.pad %v712, %v713, low = [0, 0, 128], high = [0, 0, 576], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v715 = stablehlo.reshape %v714 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v716 = stablehlo.add %v678, %v715 : tensor<32x151296xf32>
    %v717 = stablehlo.reshape %v593 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v718 = stablehlo.slice %v717 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v719 = stablehlo.reshape %v718 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v720 = stablehlo.reshape %v598 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v721 = stablehlo.slice %v720 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v722 = stablehlo.reshape %v721 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v723 = stablehlo.reshape %v603 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v724 = stablehlo.slice %v723 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v725 = stablehlo.reshape %v724 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v726 = stablehlo.reshape %v722 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v727 = stablehlo.transpose %v726, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v728 = stablehlo.reshape %v727 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v729 = stablehlo.reshape %v719 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v730 = stablehlo.reshape %v728 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v731 = stablehlo.dot_general %v729, %v730, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v732 = stablehlo.reshape %v731 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v733 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v734 = stablehlo.multiply %v732, %v733 : tensor<32x38809xf32>
    %v735 = stablehlo.reshape %v734 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v736 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v737 = stablehlo.reduce(%v735 init: %v736) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v738 = stablehlo.broadcast_in_dim %v737, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v739 = stablehlo.subtract %v735, %v738 : tensor<32x197x197xf32>
    %v740 = stablehlo.constant dense<0.0> : tensor<f32>
    %v741 = stablehlo.exponential %v739 : tensor<32x197x197xf32>
    %v742 = stablehlo.reduce(%v741 init: %v740) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v743 = stablehlo.broadcast_in_dim %v742, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v744 = stablehlo.divide %v741, %v743 : tensor<32x197x197xf32>
    %v745 = stablehlo.reshape %v744 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v746 = stablehlo.reshape %v745 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v747 = stablehlo.reshape %v725 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v748 = stablehlo.dot_general %v746, %v747, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v749 = stablehlo.reshape %v748 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v750 = stablehlo.reshape %v749 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v751 = stablehlo.constant dense<0.0> : tensor<f32>
    %v752 = stablehlo.pad %v750, %v751, low = [0, 0, 192], high = [0, 0, 512], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v753 = stablehlo.reshape %v752 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v754 = stablehlo.add %v716, %v753 : tensor<32x151296xf32>
    %v755 = stablehlo.reshape %v593 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v756 = stablehlo.slice %v755 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v757 = stablehlo.reshape %v756 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v758 = stablehlo.reshape %v598 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v759 = stablehlo.slice %v758 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v760 = stablehlo.reshape %v759 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v761 = stablehlo.reshape %v603 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v762 = stablehlo.slice %v761 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v763 = stablehlo.reshape %v762 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v764 = stablehlo.reshape %v760 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v765 = stablehlo.transpose %v764, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v766 = stablehlo.reshape %v765 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v767 = stablehlo.reshape %v757 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v768 = stablehlo.reshape %v766 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v769 = stablehlo.dot_general %v767, %v768, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v770 = stablehlo.reshape %v769 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v771 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v772 = stablehlo.multiply %v770, %v771 : tensor<32x38809xf32>
    %v773 = stablehlo.reshape %v772 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v774 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v775 = stablehlo.reduce(%v773 init: %v774) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v776 = stablehlo.broadcast_in_dim %v775, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v777 = stablehlo.subtract %v773, %v776 : tensor<32x197x197xf32>
    %v778 = stablehlo.constant dense<0.0> : tensor<f32>
    %v779 = stablehlo.exponential %v777 : tensor<32x197x197xf32>
    %v780 = stablehlo.reduce(%v779 init: %v778) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v781 = stablehlo.broadcast_in_dim %v780, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v782 = stablehlo.divide %v779, %v781 : tensor<32x197x197xf32>
    %v783 = stablehlo.reshape %v782 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v784 = stablehlo.reshape %v783 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v785 = stablehlo.reshape %v763 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v786 = stablehlo.dot_general %v784, %v785, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v787 = stablehlo.reshape %v786 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v788 = stablehlo.reshape %v787 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v789 = stablehlo.constant dense<0.0> : tensor<f32>
    %v790 = stablehlo.pad %v788, %v789, low = [0, 0, 256], high = [0, 0, 448], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v791 = stablehlo.reshape %v790 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v792 = stablehlo.add %v754, %v791 : tensor<32x151296xf32>
    %v793 = stablehlo.reshape %v593 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v794 = stablehlo.slice %v793 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v795 = stablehlo.reshape %v794 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v796 = stablehlo.reshape %v598 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v797 = stablehlo.slice %v796 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v798 = stablehlo.reshape %v797 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v799 = stablehlo.reshape %v603 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v800 = stablehlo.slice %v799 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
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
    %v812 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v813 = stablehlo.reduce(%v811 init: %v812) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v814 = stablehlo.broadcast_in_dim %v813, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v815 = stablehlo.subtract %v811, %v814 : tensor<32x197x197xf32>
    %v816 = stablehlo.constant dense<0.0> : tensor<f32>
    %v817 = stablehlo.exponential %v815 : tensor<32x197x197xf32>
    %v818 = stablehlo.reduce(%v817 init: %v816) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v819 = stablehlo.broadcast_in_dim %v818, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v820 = stablehlo.divide %v817, %v819 : tensor<32x197x197xf32>
    %v821 = stablehlo.reshape %v820 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v822 = stablehlo.reshape %v821 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v823 = stablehlo.reshape %v801 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v824 = stablehlo.dot_general %v822, %v823, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v825 = stablehlo.reshape %v824 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v826 = stablehlo.reshape %v825 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v827 = stablehlo.constant dense<0.0> : tensor<f32>
    %v828 = stablehlo.pad %v826, %v827, low = [0, 0, 320], high = [0, 0, 384], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v829 = stablehlo.reshape %v828 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v830 = stablehlo.add %v792, %v829 : tensor<32x151296xf32>
    %v831 = stablehlo.reshape %v593 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v832 = stablehlo.slice %v831 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v833 = stablehlo.reshape %v832 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v834 = stablehlo.reshape %v598 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v835 = stablehlo.slice %v834 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v836 = stablehlo.reshape %v835 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v837 = stablehlo.reshape %v603 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v838 = stablehlo.slice %v837 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v839 = stablehlo.reshape %v838 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v840 = stablehlo.reshape %v836 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v841 = stablehlo.transpose %v840, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v842 = stablehlo.reshape %v841 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v843 = stablehlo.reshape %v833 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v844 = stablehlo.reshape %v842 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v845 = stablehlo.dot_general %v843, %v844, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v846 = stablehlo.reshape %v845 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v847 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v848 = stablehlo.multiply %v846, %v847 : tensor<32x38809xf32>
    %v849 = stablehlo.reshape %v848 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v850 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v851 = stablehlo.reduce(%v849 init: %v850) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v852 = stablehlo.broadcast_in_dim %v851, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v853 = stablehlo.subtract %v849, %v852 : tensor<32x197x197xf32>
    %v854 = stablehlo.constant dense<0.0> : tensor<f32>
    %v855 = stablehlo.exponential %v853 : tensor<32x197x197xf32>
    %v856 = stablehlo.reduce(%v855 init: %v854) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v857 = stablehlo.broadcast_in_dim %v856, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v858 = stablehlo.divide %v855, %v857 : tensor<32x197x197xf32>
    %v859 = stablehlo.reshape %v858 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v860 = stablehlo.reshape %v859 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v861 = stablehlo.reshape %v839 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v862 = stablehlo.dot_general %v860, %v861, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v863 = stablehlo.reshape %v862 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v864 = stablehlo.reshape %v863 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v865 = stablehlo.constant dense<0.0> : tensor<f32>
    %v866 = stablehlo.pad %v864, %v865, low = [0, 0, 384], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v867 = stablehlo.reshape %v866 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v868 = stablehlo.add %v830, %v867 : tensor<32x151296xf32>
    %v869 = stablehlo.reshape %v593 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v870 = stablehlo.slice %v869 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v871 = stablehlo.reshape %v870 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v872 = stablehlo.reshape %v598 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v873 = stablehlo.slice %v872 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v874 = stablehlo.reshape %v873 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v875 = stablehlo.reshape %v603 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v876 = stablehlo.slice %v875 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v877 = stablehlo.reshape %v876 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v878 = stablehlo.reshape %v874 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v879 = stablehlo.transpose %v878, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v880 = stablehlo.reshape %v879 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v881 = stablehlo.reshape %v871 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v882 = stablehlo.reshape %v880 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v883 = stablehlo.dot_general %v881, %v882, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v884 = stablehlo.reshape %v883 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v885 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v886 = stablehlo.multiply %v884, %v885 : tensor<32x38809xf32>
    %v887 = stablehlo.reshape %v886 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v888 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v889 = stablehlo.reduce(%v887 init: %v888) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v890 = stablehlo.broadcast_in_dim %v889, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v891 = stablehlo.subtract %v887, %v890 : tensor<32x197x197xf32>
    %v892 = stablehlo.constant dense<0.0> : tensor<f32>
    %v893 = stablehlo.exponential %v891 : tensor<32x197x197xf32>
    %v894 = stablehlo.reduce(%v893 init: %v892) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v895 = stablehlo.broadcast_in_dim %v894, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v896 = stablehlo.divide %v893, %v895 : tensor<32x197x197xf32>
    %v897 = stablehlo.reshape %v896 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v898 = stablehlo.reshape %v897 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v899 = stablehlo.reshape %v877 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v900 = stablehlo.dot_general %v898, %v899, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v901 = stablehlo.reshape %v900 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v902 = stablehlo.reshape %v901 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v903 = stablehlo.constant dense<0.0> : tensor<f32>
    %v904 = stablehlo.pad %v902, %v903, low = [0, 0, 448], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v905 = stablehlo.reshape %v904 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v906 = stablehlo.add %v868, %v905 : tensor<32x151296xf32>
    %v907 = stablehlo.reshape %v593 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v908 = stablehlo.slice %v907 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v909 = stablehlo.reshape %v908 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v910 = stablehlo.reshape %v598 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v911 = stablehlo.slice %v910 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v912 = stablehlo.reshape %v911 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v913 = stablehlo.reshape %v603 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v914 = stablehlo.slice %v913 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v915 = stablehlo.reshape %v914 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v916 = stablehlo.reshape %v912 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v917 = stablehlo.transpose %v916, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v918 = stablehlo.reshape %v917 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v919 = stablehlo.reshape %v909 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v920 = stablehlo.reshape %v918 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v921 = stablehlo.dot_general %v919, %v920, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v922 = stablehlo.reshape %v921 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v923 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v924 = stablehlo.multiply %v922, %v923 : tensor<32x38809xf32>
    %v925 = stablehlo.reshape %v924 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v926 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v927 = stablehlo.reduce(%v925 init: %v926) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v928 = stablehlo.broadcast_in_dim %v927, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v929 = stablehlo.subtract %v925, %v928 : tensor<32x197x197xf32>
    %v930 = stablehlo.constant dense<0.0> : tensor<f32>
    %v931 = stablehlo.exponential %v929 : tensor<32x197x197xf32>
    %v932 = stablehlo.reduce(%v931 init: %v930) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v933 = stablehlo.broadcast_in_dim %v932, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v934 = stablehlo.divide %v931, %v933 : tensor<32x197x197xf32>
    %v935 = stablehlo.reshape %v934 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v936 = stablehlo.reshape %v935 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v937 = stablehlo.reshape %v915 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v938 = stablehlo.dot_general %v936, %v937, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v939 = stablehlo.reshape %v938 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v940 = stablehlo.reshape %v939 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v941 = stablehlo.constant dense<0.0> : tensor<f32>
    %v942 = stablehlo.pad %v940, %v941, low = [0, 0, 512], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v943 = stablehlo.reshape %v942 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v944 = stablehlo.add %v906, %v943 : tensor<32x151296xf32>
    %v945 = stablehlo.reshape %v593 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v946 = stablehlo.slice %v945 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v947 = stablehlo.reshape %v946 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v948 = stablehlo.reshape %v598 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v949 = stablehlo.slice %v948 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v950 = stablehlo.reshape %v949 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v951 = stablehlo.reshape %v603 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v952 = stablehlo.slice %v951 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v953 = stablehlo.reshape %v952 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v954 = stablehlo.reshape %v950 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v955 = stablehlo.transpose %v954, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v956 = stablehlo.reshape %v955 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v957 = stablehlo.reshape %v947 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v958 = stablehlo.reshape %v956 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v959 = stablehlo.dot_general %v957, %v958, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v960 = stablehlo.reshape %v959 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v961 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v962 = stablehlo.multiply %v960, %v961 : tensor<32x38809xf32>
    %v963 = stablehlo.reshape %v962 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v964 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v965 = stablehlo.reduce(%v963 init: %v964) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v966 = stablehlo.broadcast_in_dim %v965, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v967 = stablehlo.subtract %v963, %v966 : tensor<32x197x197xf32>
    %v968 = stablehlo.constant dense<0.0> : tensor<f32>
    %v969 = stablehlo.exponential %v967 : tensor<32x197x197xf32>
    %v970 = stablehlo.reduce(%v969 init: %v968) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v971 = stablehlo.broadcast_in_dim %v970, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v972 = stablehlo.divide %v969, %v971 : tensor<32x197x197xf32>
    %v973 = stablehlo.reshape %v972 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v974 = stablehlo.reshape %v973 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v975 = stablehlo.reshape %v953 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v976 = stablehlo.dot_general %v974, %v975, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v977 = stablehlo.reshape %v976 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v978 = stablehlo.reshape %v977 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v979 = stablehlo.constant dense<0.0> : tensor<f32>
    %v980 = stablehlo.pad %v978, %v979, low = [0, 0, 576], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v981 = stablehlo.reshape %v980 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v982 = stablehlo.add %v944, %v981 : tensor<32x151296xf32>
    %v983 = stablehlo.reshape %v593 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v984 = stablehlo.slice %v983 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v985 = stablehlo.reshape %v984 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v986 = stablehlo.reshape %v598 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v987 = stablehlo.slice %v986 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v988 = stablehlo.reshape %v987 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v989 = stablehlo.reshape %v603 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v990 = stablehlo.slice %v989 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v991 = stablehlo.reshape %v990 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v992 = stablehlo.reshape %v988 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v993 = stablehlo.transpose %v992, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v994 = stablehlo.reshape %v993 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v995 = stablehlo.reshape %v985 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v996 = stablehlo.reshape %v994 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v997 = stablehlo.dot_general %v995, %v996, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v998 = stablehlo.reshape %v997 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v999 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1000 = stablehlo.multiply %v998, %v999 : tensor<32x38809xf32>
    %v1001 = stablehlo.reshape %v1000 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1002 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v1003 = stablehlo.reduce(%v1001 init: %v1002) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1004 = stablehlo.broadcast_in_dim %v1003, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1005 = stablehlo.subtract %v1001, %v1004 : tensor<32x197x197xf32>
    %v1006 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1007 = stablehlo.exponential %v1005 : tensor<32x197x197xf32>
    %v1008 = stablehlo.reduce(%v1007 init: %v1006) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1009 = stablehlo.broadcast_in_dim %v1008, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1010 = stablehlo.divide %v1007, %v1009 : tensor<32x197x197xf32>
    %v1011 = stablehlo.reshape %v1010 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1012 = stablehlo.reshape %v1011 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1013 = stablehlo.reshape %v991 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1014 = stablehlo.dot_general %v1012, %v1013, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1015 = stablehlo.reshape %v1014 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1016 = stablehlo.reshape %v1015 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1017 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1018 = stablehlo.pad %v1016, %v1017, low = [0, 0, 640], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v1019 = stablehlo.reshape %v1018 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1020 = stablehlo.add %v982, %v1019 : tensor<32x151296xf32>
    %v1021 = stablehlo.reshape %v593 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1022 = stablehlo.slice %v1021 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1023 = stablehlo.reshape %v1022 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1024 = stablehlo.reshape %v598 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1025 = stablehlo.slice %v1024 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1026 = stablehlo.reshape %v1025 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1027 = stablehlo.reshape %v603 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1028 = stablehlo.slice %v1027 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1029 = stablehlo.reshape %v1028 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1030 = stablehlo.reshape %v1026 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1031 = stablehlo.transpose %v1030, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1032 = stablehlo.reshape %v1031 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1033 = stablehlo.reshape %v1023 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1034 = stablehlo.reshape %v1032 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1035 = stablehlo.dot_general %v1033, %v1034, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1036 = stablehlo.reshape %v1035 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1037 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1038 = stablehlo.multiply %v1036, %v1037 : tensor<32x38809xf32>
    %v1039 = stablehlo.reshape %v1038 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1040 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v1041 = stablehlo.reduce(%v1039 init: %v1040) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1042 = stablehlo.broadcast_in_dim %v1041, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1043 = stablehlo.subtract %v1039, %v1042 : tensor<32x197x197xf32>
    %v1044 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1045 = stablehlo.exponential %v1043 : tensor<32x197x197xf32>
    %v1046 = stablehlo.reduce(%v1045 init: %v1044) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1047 = stablehlo.broadcast_in_dim %v1046, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1048 = stablehlo.divide %v1045, %v1047 : tensor<32x197x197xf32>
    %v1049 = stablehlo.reshape %v1048 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1050 = stablehlo.reshape %v1049 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1051 = stablehlo.reshape %v1029 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1052 = stablehlo.dot_general %v1050, %v1051, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1053 = stablehlo.reshape %v1052 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1054 = stablehlo.reshape %v1053 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1055 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1056 = stablehlo.pad %v1054, %v1055, low = [0, 0, 704], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v1057 = stablehlo.reshape %v1056 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1058 = stablehlo.add %v1020, %v1057 : tensor<32x151296xf32>
    %v1059 = stablehlo.reshape %v1058 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1060 = stablehlo.dot_general %v1059, %b1_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v1061 = stablehlo.broadcast_in_dim %b1_bo, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v1062 = stablehlo.add %v1060, %v1061 : tensor<32x197x768xf32>
    %v1063 = stablehlo.reshape %v1062 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1064 = stablehlo.add %v560, %v1063 : tensor<32x151296xf32>
    %v1065 = stablehlo.reshape %v1064 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1066 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1067 = stablehlo.constant dense<768.0> : tensor<32x197x768xf32>
    %v1068 = stablehlo.constant dense<1.0e-6> : tensor<32x197x768xf32>
    %v1069 = stablehlo.reduce(%v1065 init: %v1066) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1070 = stablehlo.broadcast_in_dim %v1069, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v1071 = stablehlo.divide %v1070, %v1067 : tensor<32x197x768xf32>
    %v1072 = stablehlo.subtract %v1065, %v1071 : tensor<32x197x768xf32>
    %v1073 = stablehlo.multiply %v1072, %v1072 : tensor<32x197x768xf32>
    %v1074 = stablehlo.reduce(%v1073 init: %v1066) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1075 = stablehlo.broadcast_in_dim %v1074, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v1076 = stablehlo.divide %v1075, %v1067 : tensor<32x197x768xf32>
    %v1077 = stablehlo.add %v1076, %v1068 : tensor<32x197x768xf32>
    %v1078 = stablehlo.rsqrt %v1077 : tensor<32x197x768xf32>
    %v1079 = stablehlo.multiply %v1072, %v1078 : tensor<32x197x768xf32>
    %v1080 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v1081 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v1082 = stablehlo.multiply %v1079, %v1080 : tensor<32x197x768xf32>
    %v1083 = stablehlo.add %v1082, %v1081 : tensor<32x197x768xf32>
    %v1084 = stablehlo.reshape %v1083 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1085 = stablehlo.reshape %v1084 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1086 = stablehlo.broadcast_in_dim %b1_g2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v1087 = stablehlo.multiply %v1085, %v1086 : tensor<32x197x768xf32>
    %v1088 = stablehlo.reshape %v1087 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1089 = stablehlo.reshape %v1088 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1090 = stablehlo.broadcast_in_dim %b1_bt2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v1091 = stablehlo.add %v1089, %v1090 : tensor<32x197x768xf32>
    %v1092 = stablehlo.reshape %v1091 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1093 = stablehlo.reshape %v1092 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1094 = stablehlo.dot_general %v1093, %b1_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x3072xf32>) -> tensor<32x197x3072xf32>
    %v1095 = stablehlo.broadcast_in_dim %b1_bfc1, dims = [2] : (tensor<3072xf32>) -> tensor<32x197x3072xf32>
    %v1096 = stablehlo.add %v1094, %v1095 : tensor<32x197x3072xf32>
    %v1097 = stablehlo.reshape %v1096 : (tensor<32x197x3072xf32>) -> tensor<32x605184xf32>
    %v1098 = stablehlo.constant dense<0.5> : tensor<32x605184xf32>
    %v1099 = stablehlo.multiply %v1098, %v1097 : tensor<32x605184xf32>
    %v1100 = stablehlo.negate %v1097 : tensor<32x605184xf32>
    %v1101 = stablehlo.constant dense<0.7071067811865476> : tensor<32x605184xf32>
    %v1102 = stablehlo.multiply %v1100, %v1101 : tensor<32x605184xf32>
    %v1103 = chlo.erfc %v1102 : tensor<32x605184xf32> -> tensor<32x605184xf32>
    %v1104 = stablehlo.multiply %v1099, %v1103 : tensor<32x605184xf32>
    %v1105 = stablehlo.reshape %v1104 : (tensor<32x605184xf32>) -> tensor<32x197x3072xf32>
    %v1106 = stablehlo.dot_general %v1105, %b1_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x3072xf32>, tensor<3072x768xf32>) -> tensor<32x197x768xf32>
    %v1107 = stablehlo.broadcast_in_dim %b1_bfc2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v1108 = stablehlo.add %v1106, %v1107 : tensor<32x197x768xf32>
    %v1109 = stablehlo.reshape %v1108 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1110 = stablehlo.add %v1064, %v1109 : tensor<32x151296xf32>
    %v1111 = stablehlo.reshape %v1110 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1112 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1113 = stablehlo.constant dense<768.0> : tensor<32x197x768xf32>
    %v1114 = stablehlo.constant dense<1.0e-6> : tensor<32x197x768xf32>
    %v1115 = stablehlo.reduce(%v1111 init: %v1112) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1116 = stablehlo.broadcast_in_dim %v1115, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v1117 = stablehlo.divide %v1116, %v1113 : tensor<32x197x768xf32>
    %v1118 = stablehlo.subtract %v1111, %v1117 : tensor<32x197x768xf32>
    %v1119 = stablehlo.multiply %v1118, %v1118 : tensor<32x197x768xf32>
    %v1120 = stablehlo.reduce(%v1119 init: %v1112) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1121 = stablehlo.broadcast_in_dim %v1120, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v1122 = stablehlo.divide %v1121, %v1113 : tensor<32x197x768xf32>
    %v1123 = stablehlo.add %v1122, %v1114 : tensor<32x197x768xf32>
    %v1124 = stablehlo.rsqrt %v1123 : tensor<32x197x768xf32>
    %v1125 = stablehlo.multiply %v1118, %v1124 : tensor<32x197x768xf32>
    %v1126 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v1127 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v1128 = stablehlo.multiply %v1125, %v1126 : tensor<32x197x768xf32>
    %v1129 = stablehlo.add %v1128, %v1127 : tensor<32x197x768xf32>
    %v1130 = stablehlo.reshape %v1129 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1131 = stablehlo.reshape %v1130 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1132 = stablehlo.broadcast_in_dim %b2_g1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v1133 = stablehlo.multiply %v1131, %v1132 : tensor<32x197x768xf32>
    %v1134 = stablehlo.reshape %v1133 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1135 = stablehlo.reshape %v1134 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1136 = stablehlo.broadcast_in_dim %b2_bt1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v1137 = stablehlo.add %v1135, %v1136 : tensor<32x197x768xf32>
    %v1138 = stablehlo.reshape %v1137 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1139 = stablehlo.reshape %v1138 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1140 = stablehlo.dot_general %v1139, %b2_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v1141 = stablehlo.broadcast_in_dim %b2_bq, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v1142 = stablehlo.add %v1140, %v1141 : tensor<32x197x768xf32>
    %v1143 = stablehlo.reshape %v1142 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1144 = stablehlo.reshape %v1138 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1145 = stablehlo.dot_general %v1144, %b2_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v1146 = stablehlo.broadcast_in_dim %b2_bk, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v1147 = stablehlo.add %v1145, %v1146 : tensor<32x197x768xf32>
    %v1148 = stablehlo.reshape %v1147 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1149 = stablehlo.reshape %v1138 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1150 = stablehlo.dot_general %v1149, %b2_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v1151 = stablehlo.broadcast_in_dim %b2_bv, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v1152 = stablehlo.add %v1150, %v1151 : tensor<32x197x768xf32>
    %v1153 = stablehlo.reshape %v1152 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1154 = stablehlo.reshape %v1143 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1155 = stablehlo.slice %v1154 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1156 = stablehlo.reshape %v1155 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1157 = stablehlo.reshape %v1148 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1158 = stablehlo.slice %v1157 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1159 = stablehlo.reshape %v1158 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1160 = stablehlo.reshape %v1153 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1161 = stablehlo.slice %v1160 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1162 = stablehlo.reshape %v1161 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1163 = stablehlo.reshape %v1159 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1164 = stablehlo.transpose %v1163, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1165 = stablehlo.reshape %v1164 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1166 = stablehlo.reshape %v1156 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1167 = stablehlo.reshape %v1165 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1168 = stablehlo.dot_general %v1166, %v1167, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1169 = stablehlo.reshape %v1168 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1170 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1171 = stablehlo.multiply %v1169, %v1170 : tensor<32x38809xf32>
    %v1172 = stablehlo.reshape %v1171 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1173 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v1174 = stablehlo.reduce(%v1172 init: %v1173) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1175 = stablehlo.broadcast_in_dim %v1174, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1176 = stablehlo.subtract %v1172, %v1175 : tensor<32x197x197xf32>
    %v1177 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1178 = stablehlo.exponential %v1176 : tensor<32x197x197xf32>
    %v1179 = stablehlo.reduce(%v1178 init: %v1177) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1180 = stablehlo.broadcast_in_dim %v1179, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1181 = stablehlo.divide %v1178, %v1180 : tensor<32x197x197xf32>
    %v1182 = stablehlo.reshape %v1181 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1183 = stablehlo.reshape %v1182 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1184 = stablehlo.reshape %v1162 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1185 = stablehlo.dot_general %v1183, %v1184, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1186 = stablehlo.reshape %v1185 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1187 = stablehlo.reshape %v1186 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1188 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1189 = stablehlo.pad %v1187, %v1188, low = [0, 0, 0], high = [0, 0, 704], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v1190 = stablehlo.reshape %v1189 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1191 = stablehlo.reshape %v1143 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1192 = stablehlo.slice %v1191 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1193 = stablehlo.reshape %v1192 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1194 = stablehlo.reshape %v1148 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1195 = stablehlo.slice %v1194 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1196 = stablehlo.reshape %v1195 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1197 = stablehlo.reshape %v1153 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1198 = stablehlo.slice %v1197 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1199 = stablehlo.reshape %v1198 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1200 = stablehlo.reshape %v1196 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1201 = stablehlo.transpose %v1200, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1202 = stablehlo.reshape %v1201 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1203 = stablehlo.reshape %v1193 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1204 = stablehlo.reshape %v1202 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1205 = stablehlo.dot_general %v1203, %v1204, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1206 = stablehlo.reshape %v1205 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1207 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1208 = stablehlo.multiply %v1206, %v1207 : tensor<32x38809xf32>
    %v1209 = stablehlo.reshape %v1208 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1210 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v1211 = stablehlo.reduce(%v1209 init: %v1210) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1212 = stablehlo.broadcast_in_dim %v1211, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1213 = stablehlo.subtract %v1209, %v1212 : tensor<32x197x197xf32>
    %v1214 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1215 = stablehlo.exponential %v1213 : tensor<32x197x197xf32>
    %v1216 = stablehlo.reduce(%v1215 init: %v1214) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1217 = stablehlo.broadcast_in_dim %v1216, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1218 = stablehlo.divide %v1215, %v1217 : tensor<32x197x197xf32>
    %v1219 = stablehlo.reshape %v1218 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1220 = stablehlo.reshape %v1219 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1221 = stablehlo.reshape %v1199 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1222 = stablehlo.dot_general %v1220, %v1221, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1223 = stablehlo.reshape %v1222 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1224 = stablehlo.reshape %v1223 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1225 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1226 = stablehlo.pad %v1224, %v1225, low = [0, 0, 64], high = [0, 0, 640], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v1227 = stablehlo.reshape %v1226 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1228 = stablehlo.add %v1190, %v1227 : tensor<32x151296xf32>
    %v1229 = stablehlo.reshape %v1143 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1230 = stablehlo.slice %v1229 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1231 = stablehlo.reshape %v1230 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1232 = stablehlo.reshape %v1148 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1233 = stablehlo.slice %v1232 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1234 = stablehlo.reshape %v1233 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1235 = stablehlo.reshape %v1153 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1236 = stablehlo.slice %v1235 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1237 = stablehlo.reshape %v1236 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1238 = stablehlo.reshape %v1234 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1239 = stablehlo.transpose %v1238, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1240 = stablehlo.reshape %v1239 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1241 = stablehlo.reshape %v1231 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1242 = stablehlo.reshape %v1240 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1243 = stablehlo.dot_general %v1241, %v1242, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1244 = stablehlo.reshape %v1243 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1245 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1246 = stablehlo.multiply %v1244, %v1245 : tensor<32x38809xf32>
    %v1247 = stablehlo.reshape %v1246 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1248 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v1249 = stablehlo.reduce(%v1247 init: %v1248) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1250 = stablehlo.broadcast_in_dim %v1249, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1251 = stablehlo.subtract %v1247, %v1250 : tensor<32x197x197xf32>
    %v1252 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1253 = stablehlo.exponential %v1251 : tensor<32x197x197xf32>
    %v1254 = stablehlo.reduce(%v1253 init: %v1252) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1255 = stablehlo.broadcast_in_dim %v1254, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1256 = stablehlo.divide %v1253, %v1255 : tensor<32x197x197xf32>
    %v1257 = stablehlo.reshape %v1256 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1258 = stablehlo.reshape %v1257 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1259 = stablehlo.reshape %v1237 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1260 = stablehlo.dot_general %v1258, %v1259, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1261 = stablehlo.reshape %v1260 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1262 = stablehlo.reshape %v1261 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1263 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1264 = stablehlo.pad %v1262, %v1263, low = [0, 0, 128], high = [0, 0, 576], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v1265 = stablehlo.reshape %v1264 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1266 = stablehlo.add %v1228, %v1265 : tensor<32x151296xf32>
    %v1267 = stablehlo.reshape %v1143 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1268 = stablehlo.slice %v1267 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1269 = stablehlo.reshape %v1268 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1270 = stablehlo.reshape %v1148 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1271 = stablehlo.slice %v1270 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1272 = stablehlo.reshape %v1271 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1273 = stablehlo.reshape %v1153 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1274 = stablehlo.slice %v1273 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1275 = stablehlo.reshape %v1274 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1276 = stablehlo.reshape %v1272 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1277 = stablehlo.transpose %v1276, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1278 = stablehlo.reshape %v1277 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1279 = stablehlo.reshape %v1269 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1280 = stablehlo.reshape %v1278 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1281 = stablehlo.dot_general %v1279, %v1280, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1282 = stablehlo.reshape %v1281 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1283 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1284 = stablehlo.multiply %v1282, %v1283 : tensor<32x38809xf32>
    %v1285 = stablehlo.reshape %v1284 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1286 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v1287 = stablehlo.reduce(%v1285 init: %v1286) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1288 = stablehlo.broadcast_in_dim %v1287, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1289 = stablehlo.subtract %v1285, %v1288 : tensor<32x197x197xf32>
    %v1290 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1291 = stablehlo.exponential %v1289 : tensor<32x197x197xf32>
    %v1292 = stablehlo.reduce(%v1291 init: %v1290) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1293 = stablehlo.broadcast_in_dim %v1292, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1294 = stablehlo.divide %v1291, %v1293 : tensor<32x197x197xf32>
    %v1295 = stablehlo.reshape %v1294 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1296 = stablehlo.reshape %v1295 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1297 = stablehlo.reshape %v1275 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1298 = stablehlo.dot_general %v1296, %v1297, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1299 = stablehlo.reshape %v1298 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1300 = stablehlo.reshape %v1299 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1301 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1302 = stablehlo.pad %v1300, %v1301, low = [0, 0, 192], high = [0, 0, 512], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v1303 = stablehlo.reshape %v1302 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1304 = stablehlo.add %v1266, %v1303 : tensor<32x151296xf32>
    %v1305 = stablehlo.reshape %v1143 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1306 = stablehlo.slice %v1305 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1307 = stablehlo.reshape %v1306 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1308 = stablehlo.reshape %v1148 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1309 = stablehlo.slice %v1308 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1310 = stablehlo.reshape %v1309 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1311 = stablehlo.reshape %v1153 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1312 = stablehlo.slice %v1311 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1313 = stablehlo.reshape %v1312 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1314 = stablehlo.reshape %v1310 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1315 = stablehlo.transpose %v1314, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1316 = stablehlo.reshape %v1315 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1317 = stablehlo.reshape %v1307 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1318 = stablehlo.reshape %v1316 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1319 = stablehlo.dot_general %v1317, %v1318, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1320 = stablehlo.reshape %v1319 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1321 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1322 = stablehlo.multiply %v1320, %v1321 : tensor<32x38809xf32>
    %v1323 = stablehlo.reshape %v1322 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1324 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v1325 = stablehlo.reduce(%v1323 init: %v1324) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1326 = stablehlo.broadcast_in_dim %v1325, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1327 = stablehlo.subtract %v1323, %v1326 : tensor<32x197x197xf32>
    %v1328 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1329 = stablehlo.exponential %v1327 : tensor<32x197x197xf32>
    %v1330 = stablehlo.reduce(%v1329 init: %v1328) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1331 = stablehlo.broadcast_in_dim %v1330, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1332 = stablehlo.divide %v1329, %v1331 : tensor<32x197x197xf32>
    %v1333 = stablehlo.reshape %v1332 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1334 = stablehlo.reshape %v1333 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1335 = stablehlo.reshape %v1313 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1336 = stablehlo.dot_general %v1334, %v1335, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1337 = stablehlo.reshape %v1336 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1338 = stablehlo.reshape %v1337 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1339 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1340 = stablehlo.pad %v1338, %v1339, low = [0, 0, 256], high = [0, 0, 448], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v1341 = stablehlo.reshape %v1340 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1342 = stablehlo.add %v1304, %v1341 : tensor<32x151296xf32>
    %v1343 = stablehlo.reshape %v1143 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1344 = stablehlo.slice %v1343 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1345 = stablehlo.reshape %v1344 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1346 = stablehlo.reshape %v1148 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1347 = stablehlo.slice %v1346 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1348 = stablehlo.reshape %v1347 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1349 = stablehlo.reshape %v1153 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1350 = stablehlo.slice %v1349 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1351 = stablehlo.reshape %v1350 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1352 = stablehlo.reshape %v1348 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1353 = stablehlo.transpose %v1352, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1354 = stablehlo.reshape %v1353 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1355 = stablehlo.reshape %v1345 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1356 = stablehlo.reshape %v1354 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1357 = stablehlo.dot_general %v1355, %v1356, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1358 = stablehlo.reshape %v1357 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1359 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1360 = stablehlo.multiply %v1358, %v1359 : tensor<32x38809xf32>
    %v1361 = stablehlo.reshape %v1360 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1362 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v1363 = stablehlo.reduce(%v1361 init: %v1362) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1364 = stablehlo.broadcast_in_dim %v1363, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1365 = stablehlo.subtract %v1361, %v1364 : tensor<32x197x197xf32>
    %v1366 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1367 = stablehlo.exponential %v1365 : tensor<32x197x197xf32>
    %v1368 = stablehlo.reduce(%v1367 init: %v1366) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1369 = stablehlo.broadcast_in_dim %v1368, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1370 = stablehlo.divide %v1367, %v1369 : tensor<32x197x197xf32>
    %v1371 = stablehlo.reshape %v1370 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1372 = stablehlo.reshape %v1371 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1373 = stablehlo.reshape %v1351 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1374 = stablehlo.dot_general %v1372, %v1373, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1375 = stablehlo.reshape %v1374 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1376 = stablehlo.reshape %v1375 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1377 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1378 = stablehlo.pad %v1376, %v1377, low = [0, 0, 320], high = [0, 0, 384], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v1379 = stablehlo.reshape %v1378 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1380 = stablehlo.add %v1342, %v1379 : tensor<32x151296xf32>
    %v1381 = stablehlo.reshape %v1143 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1382 = stablehlo.slice %v1381 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1383 = stablehlo.reshape %v1382 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1384 = stablehlo.reshape %v1148 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1385 = stablehlo.slice %v1384 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1386 = stablehlo.reshape %v1385 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1387 = stablehlo.reshape %v1153 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1388 = stablehlo.slice %v1387 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1389 = stablehlo.reshape %v1388 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1390 = stablehlo.reshape %v1386 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1391 = stablehlo.transpose %v1390, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1392 = stablehlo.reshape %v1391 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1393 = stablehlo.reshape %v1383 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1394 = stablehlo.reshape %v1392 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1395 = stablehlo.dot_general %v1393, %v1394, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1396 = stablehlo.reshape %v1395 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1397 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1398 = stablehlo.multiply %v1396, %v1397 : tensor<32x38809xf32>
    %v1399 = stablehlo.reshape %v1398 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1400 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v1401 = stablehlo.reduce(%v1399 init: %v1400) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1402 = stablehlo.broadcast_in_dim %v1401, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1403 = stablehlo.subtract %v1399, %v1402 : tensor<32x197x197xf32>
    %v1404 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1405 = stablehlo.exponential %v1403 : tensor<32x197x197xf32>
    %v1406 = stablehlo.reduce(%v1405 init: %v1404) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1407 = stablehlo.broadcast_in_dim %v1406, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1408 = stablehlo.divide %v1405, %v1407 : tensor<32x197x197xf32>
    %v1409 = stablehlo.reshape %v1408 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1410 = stablehlo.reshape %v1409 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1411 = stablehlo.reshape %v1389 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1412 = stablehlo.dot_general %v1410, %v1411, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1413 = stablehlo.reshape %v1412 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1414 = stablehlo.reshape %v1413 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1415 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1416 = stablehlo.pad %v1414, %v1415, low = [0, 0, 384], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v1417 = stablehlo.reshape %v1416 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1418 = stablehlo.add %v1380, %v1417 : tensor<32x151296xf32>
    %v1419 = stablehlo.reshape %v1143 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1420 = stablehlo.slice %v1419 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1421 = stablehlo.reshape %v1420 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1422 = stablehlo.reshape %v1148 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1423 = stablehlo.slice %v1422 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1424 = stablehlo.reshape %v1423 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1425 = stablehlo.reshape %v1153 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1426 = stablehlo.slice %v1425 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1427 = stablehlo.reshape %v1426 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1428 = stablehlo.reshape %v1424 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1429 = stablehlo.transpose %v1428, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1430 = stablehlo.reshape %v1429 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1431 = stablehlo.reshape %v1421 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1432 = stablehlo.reshape %v1430 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1433 = stablehlo.dot_general %v1431, %v1432, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1434 = stablehlo.reshape %v1433 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1435 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1436 = stablehlo.multiply %v1434, %v1435 : tensor<32x38809xf32>
    %v1437 = stablehlo.reshape %v1436 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1438 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v1439 = stablehlo.reduce(%v1437 init: %v1438) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1440 = stablehlo.broadcast_in_dim %v1439, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1441 = stablehlo.subtract %v1437, %v1440 : tensor<32x197x197xf32>
    %v1442 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1443 = stablehlo.exponential %v1441 : tensor<32x197x197xf32>
    %v1444 = stablehlo.reduce(%v1443 init: %v1442) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1445 = stablehlo.broadcast_in_dim %v1444, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1446 = stablehlo.divide %v1443, %v1445 : tensor<32x197x197xf32>
    %v1447 = stablehlo.reshape %v1446 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1448 = stablehlo.reshape %v1447 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1449 = stablehlo.reshape %v1427 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1450 = stablehlo.dot_general %v1448, %v1449, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1451 = stablehlo.reshape %v1450 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1452 = stablehlo.reshape %v1451 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1453 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1454 = stablehlo.pad %v1452, %v1453, low = [0, 0, 448], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v1455 = stablehlo.reshape %v1454 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1456 = stablehlo.add %v1418, %v1455 : tensor<32x151296xf32>
    %v1457 = stablehlo.reshape %v1143 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1458 = stablehlo.slice %v1457 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1459 = stablehlo.reshape %v1458 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1460 = stablehlo.reshape %v1148 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1461 = stablehlo.slice %v1460 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1462 = stablehlo.reshape %v1461 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1463 = stablehlo.reshape %v1153 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1464 = stablehlo.slice %v1463 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1465 = stablehlo.reshape %v1464 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1466 = stablehlo.reshape %v1462 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1467 = stablehlo.transpose %v1466, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1468 = stablehlo.reshape %v1467 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1469 = stablehlo.reshape %v1459 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1470 = stablehlo.reshape %v1468 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1471 = stablehlo.dot_general %v1469, %v1470, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1472 = stablehlo.reshape %v1471 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1473 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1474 = stablehlo.multiply %v1472, %v1473 : tensor<32x38809xf32>
    %v1475 = stablehlo.reshape %v1474 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1476 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v1477 = stablehlo.reduce(%v1475 init: %v1476) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1478 = stablehlo.broadcast_in_dim %v1477, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1479 = stablehlo.subtract %v1475, %v1478 : tensor<32x197x197xf32>
    %v1480 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1481 = stablehlo.exponential %v1479 : tensor<32x197x197xf32>
    %v1482 = stablehlo.reduce(%v1481 init: %v1480) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1483 = stablehlo.broadcast_in_dim %v1482, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1484 = stablehlo.divide %v1481, %v1483 : tensor<32x197x197xf32>
    %v1485 = stablehlo.reshape %v1484 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1486 = stablehlo.reshape %v1485 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1487 = stablehlo.reshape %v1465 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1488 = stablehlo.dot_general %v1486, %v1487, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1489 = stablehlo.reshape %v1488 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1490 = stablehlo.reshape %v1489 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1491 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1492 = stablehlo.pad %v1490, %v1491, low = [0, 0, 512], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v1493 = stablehlo.reshape %v1492 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1494 = stablehlo.add %v1456, %v1493 : tensor<32x151296xf32>
    %v1495 = stablehlo.reshape %v1143 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1496 = stablehlo.slice %v1495 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1497 = stablehlo.reshape %v1496 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1498 = stablehlo.reshape %v1148 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1499 = stablehlo.slice %v1498 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1500 = stablehlo.reshape %v1499 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1501 = stablehlo.reshape %v1153 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1502 = stablehlo.slice %v1501 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1503 = stablehlo.reshape %v1502 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1504 = stablehlo.reshape %v1500 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1505 = stablehlo.transpose %v1504, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1506 = stablehlo.reshape %v1505 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1507 = stablehlo.reshape %v1497 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1508 = stablehlo.reshape %v1506 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1509 = stablehlo.dot_general %v1507, %v1508, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1510 = stablehlo.reshape %v1509 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1511 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1512 = stablehlo.multiply %v1510, %v1511 : tensor<32x38809xf32>
    %v1513 = stablehlo.reshape %v1512 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1514 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v1515 = stablehlo.reduce(%v1513 init: %v1514) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1516 = stablehlo.broadcast_in_dim %v1515, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1517 = stablehlo.subtract %v1513, %v1516 : tensor<32x197x197xf32>
    %v1518 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1519 = stablehlo.exponential %v1517 : tensor<32x197x197xf32>
    %v1520 = stablehlo.reduce(%v1519 init: %v1518) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1521 = stablehlo.broadcast_in_dim %v1520, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1522 = stablehlo.divide %v1519, %v1521 : tensor<32x197x197xf32>
    %v1523 = stablehlo.reshape %v1522 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1524 = stablehlo.reshape %v1523 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1525 = stablehlo.reshape %v1503 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1526 = stablehlo.dot_general %v1524, %v1525, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1527 = stablehlo.reshape %v1526 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1528 = stablehlo.reshape %v1527 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1529 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1530 = stablehlo.pad %v1528, %v1529, low = [0, 0, 576], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v1531 = stablehlo.reshape %v1530 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1532 = stablehlo.add %v1494, %v1531 : tensor<32x151296xf32>
    %v1533 = stablehlo.reshape %v1143 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1534 = stablehlo.slice %v1533 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1535 = stablehlo.reshape %v1534 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1536 = stablehlo.reshape %v1148 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1537 = stablehlo.slice %v1536 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1538 = stablehlo.reshape %v1537 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1539 = stablehlo.reshape %v1153 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1540 = stablehlo.slice %v1539 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1541 = stablehlo.reshape %v1540 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1542 = stablehlo.reshape %v1538 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1543 = stablehlo.transpose %v1542, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1544 = stablehlo.reshape %v1543 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1545 = stablehlo.reshape %v1535 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1546 = stablehlo.reshape %v1544 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1547 = stablehlo.dot_general %v1545, %v1546, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1548 = stablehlo.reshape %v1547 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1549 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1550 = stablehlo.multiply %v1548, %v1549 : tensor<32x38809xf32>
    %v1551 = stablehlo.reshape %v1550 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1552 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v1553 = stablehlo.reduce(%v1551 init: %v1552) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1554 = stablehlo.broadcast_in_dim %v1553, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1555 = stablehlo.subtract %v1551, %v1554 : tensor<32x197x197xf32>
    %v1556 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1557 = stablehlo.exponential %v1555 : tensor<32x197x197xf32>
    %v1558 = stablehlo.reduce(%v1557 init: %v1556) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1559 = stablehlo.broadcast_in_dim %v1558, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1560 = stablehlo.divide %v1557, %v1559 : tensor<32x197x197xf32>
    %v1561 = stablehlo.reshape %v1560 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1562 = stablehlo.reshape %v1561 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1563 = stablehlo.reshape %v1541 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1564 = stablehlo.dot_general %v1562, %v1563, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1565 = stablehlo.reshape %v1564 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1566 = stablehlo.reshape %v1565 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1567 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1568 = stablehlo.pad %v1566, %v1567, low = [0, 0, 640], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v1569 = stablehlo.reshape %v1568 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1570 = stablehlo.add %v1532, %v1569 : tensor<32x151296xf32>
    %v1571 = stablehlo.reshape %v1143 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1572 = stablehlo.slice %v1571 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1573 = stablehlo.reshape %v1572 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1574 = stablehlo.reshape %v1148 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1575 = stablehlo.slice %v1574 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1576 = stablehlo.reshape %v1575 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1577 = stablehlo.reshape %v1153 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1578 = stablehlo.slice %v1577 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1579 = stablehlo.reshape %v1578 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1580 = stablehlo.reshape %v1576 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1581 = stablehlo.transpose %v1580, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1582 = stablehlo.reshape %v1581 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1583 = stablehlo.reshape %v1573 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1584 = stablehlo.reshape %v1582 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1585 = stablehlo.dot_general %v1583, %v1584, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1586 = stablehlo.reshape %v1585 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1587 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1588 = stablehlo.multiply %v1586, %v1587 : tensor<32x38809xf32>
    %v1589 = stablehlo.reshape %v1588 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1590 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v1591 = stablehlo.reduce(%v1589 init: %v1590) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1592 = stablehlo.broadcast_in_dim %v1591, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1593 = stablehlo.subtract %v1589, %v1592 : tensor<32x197x197xf32>
    %v1594 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1595 = stablehlo.exponential %v1593 : tensor<32x197x197xf32>
    %v1596 = stablehlo.reduce(%v1595 init: %v1594) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1597 = stablehlo.broadcast_in_dim %v1596, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1598 = stablehlo.divide %v1595, %v1597 : tensor<32x197x197xf32>
    %v1599 = stablehlo.reshape %v1598 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1600 = stablehlo.reshape %v1599 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1601 = stablehlo.reshape %v1579 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1602 = stablehlo.dot_general %v1600, %v1601, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1603 = stablehlo.reshape %v1602 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1604 = stablehlo.reshape %v1603 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1605 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1606 = stablehlo.pad %v1604, %v1605, low = [0, 0, 704], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v1607 = stablehlo.reshape %v1606 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1608 = stablehlo.add %v1570, %v1607 : tensor<32x151296xf32>
    %v1609 = stablehlo.reshape %v1608 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1610 = stablehlo.dot_general %v1609, %b2_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v1611 = stablehlo.broadcast_in_dim %b2_bo, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v1612 = stablehlo.add %v1610, %v1611 : tensor<32x197x768xf32>
    %v1613 = stablehlo.reshape %v1612 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1614 = stablehlo.add %v1110, %v1613 : tensor<32x151296xf32>
    %v1615 = stablehlo.reshape %v1614 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1616 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1617 = stablehlo.constant dense<768.0> : tensor<32x197x768xf32>
    %v1618 = stablehlo.constant dense<1.0e-6> : tensor<32x197x768xf32>
    %v1619 = stablehlo.reduce(%v1615 init: %v1616) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1620 = stablehlo.broadcast_in_dim %v1619, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v1621 = stablehlo.divide %v1620, %v1617 : tensor<32x197x768xf32>
    %v1622 = stablehlo.subtract %v1615, %v1621 : tensor<32x197x768xf32>
    %v1623 = stablehlo.multiply %v1622, %v1622 : tensor<32x197x768xf32>
    %v1624 = stablehlo.reduce(%v1623 init: %v1616) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1625 = stablehlo.broadcast_in_dim %v1624, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v1626 = stablehlo.divide %v1625, %v1617 : tensor<32x197x768xf32>
    %v1627 = stablehlo.add %v1626, %v1618 : tensor<32x197x768xf32>
    %v1628 = stablehlo.rsqrt %v1627 : tensor<32x197x768xf32>
    %v1629 = stablehlo.multiply %v1622, %v1628 : tensor<32x197x768xf32>
    %v1630 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v1631 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v1632 = stablehlo.multiply %v1629, %v1630 : tensor<32x197x768xf32>
    %v1633 = stablehlo.add %v1632, %v1631 : tensor<32x197x768xf32>
    %v1634 = stablehlo.reshape %v1633 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1635 = stablehlo.reshape %v1634 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1636 = stablehlo.broadcast_in_dim %b2_g2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v1637 = stablehlo.multiply %v1635, %v1636 : tensor<32x197x768xf32>
    %v1638 = stablehlo.reshape %v1637 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1639 = stablehlo.reshape %v1638 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1640 = stablehlo.broadcast_in_dim %b2_bt2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v1641 = stablehlo.add %v1639, %v1640 : tensor<32x197x768xf32>
    %v1642 = stablehlo.reshape %v1641 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1643 = stablehlo.reshape %v1642 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1644 = stablehlo.dot_general %v1643, %b2_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x3072xf32>) -> tensor<32x197x3072xf32>
    %v1645 = stablehlo.broadcast_in_dim %b2_bfc1, dims = [2] : (tensor<3072xf32>) -> tensor<32x197x3072xf32>
    %v1646 = stablehlo.add %v1644, %v1645 : tensor<32x197x3072xf32>
    %v1647 = stablehlo.reshape %v1646 : (tensor<32x197x3072xf32>) -> tensor<32x605184xf32>
    %v1648 = stablehlo.constant dense<0.5> : tensor<32x605184xf32>
    %v1649 = stablehlo.multiply %v1648, %v1647 : tensor<32x605184xf32>
    %v1650 = stablehlo.negate %v1647 : tensor<32x605184xf32>
    %v1651 = stablehlo.constant dense<0.7071067811865476> : tensor<32x605184xf32>
    %v1652 = stablehlo.multiply %v1650, %v1651 : tensor<32x605184xf32>
    %v1653 = chlo.erfc %v1652 : tensor<32x605184xf32> -> tensor<32x605184xf32>
    %v1654 = stablehlo.multiply %v1649, %v1653 : tensor<32x605184xf32>
    %v1655 = stablehlo.reshape %v1654 : (tensor<32x605184xf32>) -> tensor<32x197x3072xf32>
    %v1656 = stablehlo.dot_general %v1655, %b2_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x3072xf32>, tensor<3072x768xf32>) -> tensor<32x197x768xf32>
    %v1657 = stablehlo.broadcast_in_dim %b2_bfc2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v1658 = stablehlo.add %v1656, %v1657 : tensor<32x197x768xf32>
    %v1659 = stablehlo.reshape %v1658 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1660 = stablehlo.add %v1614, %v1659 : tensor<32x151296xf32>
    %v1661 = stablehlo.reshape %v1660 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1662 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1663 = stablehlo.constant dense<768.0> : tensor<32x197x768xf32>
    %v1664 = stablehlo.constant dense<1.0e-6> : tensor<32x197x768xf32>
    %v1665 = stablehlo.reduce(%v1661 init: %v1662) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1666 = stablehlo.broadcast_in_dim %v1665, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v1667 = stablehlo.divide %v1666, %v1663 : tensor<32x197x768xf32>
    %v1668 = stablehlo.subtract %v1661, %v1667 : tensor<32x197x768xf32>
    %v1669 = stablehlo.multiply %v1668, %v1668 : tensor<32x197x768xf32>
    %v1670 = stablehlo.reduce(%v1669 init: %v1662) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1671 = stablehlo.broadcast_in_dim %v1670, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v1672 = stablehlo.divide %v1671, %v1663 : tensor<32x197x768xf32>
    %v1673 = stablehlo.add %v1672, %v1664 : tensor<32x197x768xf32>
    %v1674 = stablehlo.rsqrt %v1673 : tensor<32x197x768xf32>
    %v1675 = stablehlo.multiply %v1668, %v1674 : tensor<32x197x768xf32>
    %v1676 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v1677 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v1678 = stablehlo.multiply %v1675, %v1676 : tensor<32x197x768xf32>
    %v1679 = stablehlo.add %v1678, %v1677 : tensor<32x197x768xf32>
    %v1680 = stablehlo.reshape %v1679 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1681 = stablehlo.reshape %v1680 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1682 = stablehlo.broadcast_in_dim %b3_g1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v1683 = stablehlo.multiply %v1681, %v1682 : tensor<32x197x768xf32>
    %v1684 = stablehlo.reshape %v1683 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1685 = stablehlo.reshape %v1684 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1686 = stablehlo.broadcast_in_dim %b3_bt1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v1687 = stablehlo.add %v1685, %v1686 : tensor<32x197x768xf32>
    %v1688 = stablehlo.reshape %v1687 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1689 = stablehlo.reshape %v1688 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1690 = stablehlo.dot_general %v1689, %b3_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v1691 = stablehlo.broadcast_in_dim %b3_bq, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v1692 = stablehlo.add %v1690, %v1691 : tensor<32x197x768xf32>
    %v1693 = stablehlo.reshape %v1692 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1694 = stablehlo.reshape %v1688 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1695 = stablehlo.dot_general %v1694, %b3_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v1696 = stablehlo.broadcast_in_dim %b3_bk, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v1697 = stablehlo.add %v1695, %v1696 : tensor<32x197x768xf32>
    %v1698 = stablehlo.reshape %v1697 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1699 = stablehlo.reshape %v1688 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1700 = stablehlo.dot_general %v1699, %b3_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v1701 = stablehlo.broadcast_in_dim %b3_bv, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v1702 = stablehlo.add %v1700, %v1701 : tensor<32x197x768xf32>
    %v1703 = stablehlo.reshape %v1702 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1704 = stablehlo.reshape %v1693 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1705 = stablehlo.slice %v1704 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1706 = stablehlo.reshape %v1705 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1707 = stablehlo.reshape %v1698 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1708 = stablehlo.slice %v1707 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1709 = stablehlo.reshape %v1708 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1710 = stablehlo.reshape %v1703 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1711 = stablehlo.slice %v1710 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1712 = stablehlo.reshape %v1711 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1713 = stablehlo.reshape %v1709 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1714 = stablehlo.transpose %v1713, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1715 = stablehlo.reshape %v1714 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1716 = stablehlo.reshape %v1706 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1717 = stablehlo.reshape %v1715 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1718 = stablehlo.dot_general %v1716, %v1717, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1719 = stablehlo.reshape %v1718 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1720 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1721 = stablehlo.multiply %v1719, %v1720 : tensor<32x38809xf32>
    %v1722 = stablehlo.reshape %v1721 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1723 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v1724 = stablehlo.reduce(%v1722 init: %v1723) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1725 = stablehlo.broadcast_in_dim %v1724, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1726 = stablehlo.subtract %v1722, %v1725 : tensor<32x197x197xf32>
    %v1727 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1728 = stablehlo.exponential %v1726 : tensor<32x197x197xf32>
    %v1729 = stablehlo.reduce(%v1728 init: %v1727) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1730 = stablehlo.broadcast_in_dim %v1729, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1731 = stablehlo.divide %v1728, %v1730 : tensor<32x197x197xf32>
    %v1732 = stablehlo.reshape %v1731 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1733 = stablehlo.reshape %v1732 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1734 = stablehlo.reshape %v1712 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1735 = stablehlo.dot_general %v1733, %v1734, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1736 = stablehlo.reshape %v1735 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1737 = stablehlo.reshape %v1736 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1738 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1739 = stablehlo.pad %v1737, %v1738, low = [0, 0, 0], high = [0, 0, 704], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v1740 = stablehlo.reshape %v1739 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1741 = stablehlo.reshape %v1693 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1742 = stablehlo.slice %v1741 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1743 = stablehlo.reshape %v1742 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1744 = stablehlo.reshape %v1698 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1745 = stablehlo.slice %v1744 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1746 = stablehlo.reshape %v1745 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1747 = stablehlo.reshape %v1703 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1748 = stablehlo.slice %v1747 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1749 = stablehlo.reshape %v1748 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1750 = stablehlo.reshape %v1746 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1751 = stablehlo.transpose %v1750, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1752 = stablehlo.reshape %v1751 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1753 = stablehlo.reshape %v1743 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1754 = stablehlo.reshape %v1752 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1755 = stablehlo.dot_general %v1753, %v1754, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1756 = stablehlo.reshape %v1755 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1757 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1758 = stablehlo.multiply %v1756, %v1757 : tensor<32x38809xf32>
    %v1759 = stablehlo.reshape %v1758 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1760 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v1761 = stablehlo.reduce(%v1759 init: %v1760) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1762 = stablehlo.broadcast_in_dim %v1761, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1763 = stablehlo.subtract %v1759, %v1762 : tensor<32x197x197xf32>
    %v1764 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1765 = stablehlo.exponential %v1763 : tensor<32x197x197xf32>
    %v1766 = stablehlo.reduce(%v1765 init: %v1764) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1767 = stablehlo.broadcast_in_dim %v1766, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1768 = stablehlo.divide %v1765, %v1767 : tensor<32x197x197xf32>
    %v1769 = stablehlo.reshape %v1768 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1770 = stablehlo.reshape %v1769 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1771 = stablehlo.reshape %v1749 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1772 = stablehlo.dot_general %v1770, %v1771, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1773 = stablehlo.reshape %v1772 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1774 = stablehlo.reshape %v1773 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1775 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1776 = stablehlo.pad %v1774, %v1775, low = [0, 0, 64], high = [0, 0, 640], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v1777 = stablehlo.reshape %v1776 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1778 = stablehlo.add %v1740, %v1777 : tensor<32x151296xf32>
    %v1779 = stablehlo.reshape %v1693 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1780 = stablehlo.slice %v1779 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1781 = stablehlo.reshape %v1780 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1782 = stablehlo.reshape %v1698 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1783 = stablehlo.slice %v1782 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1784 = stablehlo.reshape %v1783 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1785 = stablehlo.reshape %v1703 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1786 = stablehlo.slice %v1785 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1787 = stablehlo.reshape %v1786 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1788 = stablehlo.reshape %v1784 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1789 = stablehlo.transpose %v1788, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1790 = stablehlo.reshape %v1789 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1791 = stablehlo.reshape %v1781 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1792 = stablehlo.reshape %v1790 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1793 = stablehlo.dot_general %v1791, %v1792, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1794 = stablehlo.reshape %v1793 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1795 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1796 = stablehlo.multiply %v1794, %v1795 : tensor<32x38809xf32>
    %v1797 = stablehlo.reshape %v1796 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1798 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v1799 = stablehlo.reduce(%v1797 init: %v1798) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1800 = stablehlo.broadcast_in_dim %v1799, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1801 = stablehlo.subtract %v1797, %v1800 : tensor<32x197x197xf32>
    %v1802 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1803 = stablehlo.exponential %v1801 : tensor<32x197x197xf32>
    %v1804 = stablehlo.reduce(%v1803 init: %v1802) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1805 = stablehlo.broadcast_in_dim %v1804, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1806 = stablehlo.divide %v1803, %v1805 : tensor<32x197x197xf32>
    %v1807 = stablehlo.reshape %v1806 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1808 = stablehlo.reshape %v1807 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1809 = stablehlo.reshape %v1787 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1810 = stablehlo.dot_general %v1808, %v1809, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1811 = stablehlo.reshape %v1810 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1812 = stablehlo.reshape %v1811 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1813 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1814 = stablehlo.pad %v1812, %v1813, low = [0, 0, 128], high = [0, 0, 576], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v1815 = stablehlo.reshape %v1814 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1816 = stablehlo.add %v1778, %v1815 : tensor<32x151296xf32>
    %v1817 = stablehlo.reshape %v1693 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1818 = stablehlo.slice %v1817 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1819 = stablehlo.reshape %v1818 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1820 = stablehlo.reshape %v1698 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1821 = stablehlo.slice %v1820 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1822 = stablehlo.reshape %v1821 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1823 = stablehlo.reshape %v1703 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1824 = stablehlo.slice %v1823 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1825 = stablehlo.reshape %v1824 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1826 = stablehlo.reshape %v1822 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1827 = stablehlo.transpose %v1826, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1828 = stablehlo.reshape %v1827 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1829 = stablehlo.reshape %v1819 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1830 = stablehlo.reshape %v1828 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1831 = stablehlo.dot_general %v1829, %v1830, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1832 = stablehlo.reshape %v1831 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1833 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1834 = stablehlo.multiply %v1832, %v1833 : tensor<32x38809xf32>
    %v1835 = stablehlo.reshape %v1834 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1836 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v1837 = stablehlo.reduce(%v1835 init: %v1836) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1838 = stablehlo.broadcast_in_dim %v1837, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1839 = stablehlo.subtract %v1835, %v1838 : tensor<32x197x197xf32>
    %v1840 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1841 = stablehlo.exponential %v1839 : tensor<32x197x197xf32>
    %v1842 = stablehlo.reduce(%v1841 init: %v1840) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1843 = stablehlo.broadcast_in_dim %v1842, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1844 = stablehlo.divide %v1841, %v1843 : tensor<32x197x197xf32>
    %v1845 = stablehlo.reshape %v1844 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1846 = stablehlo.reshape %v1845 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1847 = stablehlo.reshape %v1825 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1848 = stablehlo.dot_general %v1846, %v1847, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1849 = stablehlo.reshape %v1848 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1850 = stablehlo.reshape %v1849 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1851 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1852 = stablehlo.pad %v1850, %v1851, low = [0, 0, 192], high = [0, 0, 512], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v1853 = stablehlo.reshape %v1852 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1854 = stablehlo.add %v1816, %v1853 : tensor<32x151296xf32>
    %v1855 = stablehlo.reshape %v1693 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1856 = stablehlo.slice %v1855 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1857 = stablehlo.reshape %v1856 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1858 = stablehlo.reshape %v1698 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1859 = stablehlo.slice %v1858 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1860 = stablehlo.reshape %v1859 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1861 = stablehlo.reshape %v1703 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1862 = stablehlo.slice %v1861 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1863 = stablehlo.reshape %v1862 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1864 = stablehlo.reshape %v1860 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1865 = stablehlo.transpose %v1864, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1866 = stablehlo.reshape %v1865 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1867 = stablehlo.reshape %v1857 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1868 = stablehlo.reshape %v1866 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1869 = stablehlo.dot_general %v1867, %v1868, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1870 = stablehlo.reshape %v1869 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1871 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1872 = stablehlo.multiply %v1870, %v1871 : tensor<32x38809xf32>
    %v1873 = stablehlo.reshape %v1872 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1874 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v1875 = stablehlo.reduce(%v1873 init: %v1874) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1876 = stablehlo.broadcast_in_dim %v1875, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1877 = stablehlo.subtract %v1873, %v1876 : tensor<32x197x197xf32>
    %v1878 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1879 = stablehlo.exponential %v1877 : tensor<32x197x197xf32>
    %v1880 = stablehlo.reduce(%v1879 init: %v1878) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1881 = stablehlo.broadcast_in_dim %v1880, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1882 = stablehlo.divide %v1879, %v1881 : tensor<32x197x197xf32>
    %v1883 = stablehlo.reshape %v1882 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1884 = stablehlo.reshape %v1883 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1885 = stablehlo.reshape %v1863 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1886 = stablehlo.dot_general %v1884, %v1885, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1887 = stablehlo.reshape %v1886 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1888 = stablehlo.reshape %v1887 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1889 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1890 = stablehlo.pad %v1888, %v1889, low = [0, 0, 256], high = [0, 0, 448], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v1891 = stablehlo.reshape %v1890 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1892 = stablehlo.add %v1854, %v1891 : tensor<32x151296xf32>
    %v1893 = stablehlo.reshape %v1693 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1894 = stablehlo.slice %v1893 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1895 = stablehlo.reshape %v1894 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1896 = stablehlo.reshape %v1698 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1897 = stablehlo.slice %v1896 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1898 = stablehlo.reshape %v1897 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1899 = stablehlo.reshape %v1703 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1900 = stablehlo.slice %v1899 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1901 = stablehlo.reshape %v1900 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1902 = stablehlo.reshape %v1898 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1903 = stablehlo.transpose %v1902, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1904 = stablehlo.reshape %v1903 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1905 = stablehlo.reshape %v1895 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1906 = stablehlo.reshape %v1904 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1907 = stablehlo.dot_general %v1905, %v1906, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1908 = stablehlo.reshape %v1907 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1909 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1910 = stablehlo.multiply %v1908, %v1909 : tensor<32x38809xf32>
    %v1911 = stablehlo.reshape %v1910 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1912 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v1913 = stablehlo.reduce(%v1911 init: %v1912) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1914 = stablehlo.broadcast_in_dim %v1913, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1915 = stablehlo.subtract %v1911, %v1914 : tensor<32x197x197xf32>
    %v1916 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1917 = stablehlo.exponential %v1915 : tensor<32x197x197xf32>
    %v1918 = stablehlo.reduce(%v1917 init: %v1916) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1919 = stablehlo.broadcast_in_dim %v1918, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1920 = stablehlo.divide %v1917, %v1919 : tensor<32x197x197xf32>
    %v1921 = stablehlo.reshape %v1920 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1922 = stablehlo.reshape %v1921 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1923 = stablehlo.reshape %v1901 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1924 = stablehlo.dot_general %v1922, %v1923, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1925 = stablehlo.reshape %v1924 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1926 = stablehlo.reshape %v1925 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1927 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1928 = stablehlo.pad %v1926, %v1927, low = [0, 0, 320], high = [0, 0, 384], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v1929 = stablehlo.reshape %v1928 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1930 = stablehlo.add %v1892, %v1929 : tensor<32x151296xf32>
    %v1931 = stablehlo.reshape %v1693 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1932 = stablehlo.slice %v1931 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1933 = stablehlo.reshape %v1932 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1934 = stablehlo.reshape %v1698 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1935 = stablehlo.slice %v1934 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1936 = stablehlo.reshape %v1935 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1937 = stablehlo.reshape %v1703 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1938 = stablehlo.slice %v1937 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1939 = stablehlo.reshape %v1938 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1940 = stablehlo.reshape %v1936 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1941 = stablehlo.transpose %v1940, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1942 = stablehlo.reshape %v1941 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1943 = stablehlo.reshape %v1933 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1944 = stablehlo.reshape %v1942 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1945 = stablehlo.dot_general %v1943, %v1944, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1946 = stablehlo.reshape %v1945 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1947 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1948 = stablehlo.multiply %v1946, %v1947 : tensor<32x38809xf32>
    %v1949 = stablehlo.reshape %v1948 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1950 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v1951 = stablehlo.reduce(%v1949 init: %v1950) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1952 = stablehlo.broadcast_in_dim %v1951, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1953 = stablehlo.subtract %v1949, %v1952 : tensor<32x197x197xf32>
    %v1954 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1955 = stablehlo.exponential %v1953 : tensor<32x197x197xf32>
    %v1956 = stablehlo.reduce(%v1955 init: %v1954) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1957 = stablehlo.broadcast_in_dim %v1956, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1958 = stablehlo.divide %v1955, %v1957 : tensor<32x197x197xf32>
    %v1959 = stablehlo.reshape %v1958 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1960 = stablehlo.reshape %v1959 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1961 = stablehlo.reshape %v1939 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1962 = stablehlo.dot_general %v1960, %v1961, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1963 = stablehlo.reshape %v1962 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1964 = stablehlo.reshape %v1963 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1965 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1966 = stablehlo.pad %v1964, %v1965, low = [0, 0, 384], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v1967 = stablehlo.reshape %v1966 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1968 = stablehlo.add %v1930, %v1967 : tensor<32x151296xf32>
    %v1969 = stablehlo.reshape %v1693 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1970 = stablehlo.slice %v1969 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1971 = stablehlo.reshape %v1970 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1972 = stablehlo.reshape %v1698 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1973 = stablehlo.slice %v1972 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1974 = stablehlo.reshape %v1973 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1975 = stablehlo.reshape %v1703 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1976 = stablehlo.slice %v1975 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1977 = stablehlo.reshape %v1976 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1978 = stablehlo.reshape %v1974 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1979 = stablehlo.transpose %v1978, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1980 = stablehlo.reshape %v1979 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1981 = stablehlo.reshape %v1971 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1982 = stablehlo.reshape %v1980 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1983 = stablehlo.dot_general %v1981, %v1982, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1984 = stablehlo.reshape %v1983 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1985 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1986 = stablehlo.multiply %v1984, %v1985 : tensor<32x38809xf32>
    %v1987 = stablehlo.reshape %v1986 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1988 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v1989 = stablehlo.reduce(%v1987 init: %v1988) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1990 = stablehlo.broadcast_in_dim %v1989, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1991 = stablehlo.subtract %v1987, %v1990 : tensor<32x197x197xf32>
    %v1992 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1993 = stablehlo.exponential %v1991 : tensor<32x197x197xf32>
    %v1994 = stablehlo.reduce(%v1993 init: %v1992) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1995 = stablehlo.broadcast_in_dim %v1994, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1996 = stablehlo.divide %v1993, %v1995 : tensor<32x197x197xf32>
    %v1997 = stablehlo.reshape %v1996 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1998 = stablehlo.reshape %v1997 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1999 = stablehlo.reshape %v1977 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2000 = stablehlo.dot_general %v1998, %v1999, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2001 = stablehlo.reshape %v2000 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2002 = stablehlo.reshape %v2001 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2003 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2004 = stablehlo.pad %v2002, %v2003, low = [0, 0, 448], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v2005 = stablehlo.reshape %v2004 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2006 = stablehlo.add %v1968, %v2005 : tensor<32x151296xf32>
    %v2007 = stablehlo.reshape %v1693 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2008 = stablehlo.slice %v2007 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2009 = stablehlo.reshape %v2008 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2010 = stablehlo.reshape %v1698 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2011 = stablehlo.slice %v2010 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2012 = stablehlo.reshape %v2011 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2013 = stablehlo.reshape %v1703 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2014 = stablehlo.slice %v2013 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2015 = stablehlo.reshape %v2014 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2016 = stablehlo.reshape %v2012 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2017 = stablehlo.transpose %v2016, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2018 = stablehlo.reshape %v2017 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2019 = stablehlo.reshape %v2009 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2020 = stablehlo.reshape %v2018 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2021 = stablehlo.dot_general %v2019, %v2020, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2022 = stablehlo.reshape %v2021 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2023 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2024 = stablehlo.multiply %v2022, %v2023 : tensor<32x38809xf32>
    %v2025 = stablehlo.reshape %v2024 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2026 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v2027 = stablehlo.reduce(%v2025 init: %v2026) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2028 = stablehlo.broadcast_in_dim %v2027, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2029 = stablehlo.subtract %v2025, %v2028 : tensor<32x197x197xf32>
    %v2030 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2031 = stablehlo.exponential %v2029 : tensor<32x197x197xf32>
    %v2032 = stablehlo.reduce(%v2031 init: %v2030) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2033 = stablehlo.broadcast_in_dim %v2032, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2034 = stablehlo.divide %v2031, %v2033 : tensor<32x197x197xf32>
    %v2035 = stablehlo.reshape %v2034 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2036 = stablehlo.reshape %v2035 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2037 = stablehlo.reshape %v2015 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2038 = stablehlo.dot_general %v2036, %v2037, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2039 = stablehlo.reshape %v2038 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2040 = stablehlo.reshape %v2039 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2041 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2042 = stablehlo.pad %v2040, %v2041, low = [0, 0, 512], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v2043 = stablehlo.reshape %v2042 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2044 = stablehlo.add %v2006, %v2043 : tensor<32x151296xf32>
    %v2045 = stablehlo.reshape %v1693 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2046 = stablehlo.slice %v2045 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2047 = stablehlo.reshape %v2046 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2048 = stablehlo.reshape %v1698 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2049 = stablehlo.slice %v2048 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2050 = stablehlo.reshape %v2049 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2051 = stablehlo.reshape %v1703 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2052 = stablehlo.slice %v2051 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2053 = stablehlo.reshape %v2052 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2054 = stablehlo.reshape %v2050 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2055 = stablehlo.transpose %v2054, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2056 = stablehlo.reshape %v2055 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2057 = stablehlo.reshape %v2047 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2058 = stablehlo.reshape %v2056 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2059 = stablehlo.dot_general %v2057, %v2058, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2060 = stablehlo.reshape %v2059 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2061 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2062 = stablehlo.multiply %v2060, %v2061 : tensor<32x38809xf32>
    %v2063 = stablehlo.reshape %v2062 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2064 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v2065 = stablehlo.reduce(%v2063 init: %v2064) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2066 = stablehlo.broadcast_in_dim %v2065, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2067 = stablehlo.subtract %v2063, %v2066 : tensor<32x197x197xf32>
    %v2068 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2069 = stablehlo.exponential %v2067 : tensor<32x197x197xf32>
    %v2070 = stablehlo.reduce(%v2069 init: %v2068) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2071 = stablehlo.broadcast_in_dim %v2070, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2072 = stablehlo.divide %v2069, %v2071 : tensor<32x197x197xf32>
    %v2073 = stablehlo.reshape %v2072 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2074 = stablehlo.reshape %v2073 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2075 = stablehlo.reshape %v2053 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2076 = stablehlo.dot_general %v2074, %v2075, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2077 = stablehlo.reshape %v2076 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2078 = stablehlo.reshape %v2077 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2079 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2080 = stablehlo.pad %v2078, %v2079, low = [0, 0, 576], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v2081 = stablehlo.reshape %v2080 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2082 = stablehlo.add %v2044, %v2081 : tensor<32x151296xf32>
    %v2083 = stablehlo.reshape %v1693 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2084 = stablehlo.slice %v2083 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2085 = stablehlo.reshape %v2084 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2086 = stablehlo.reshape %v1698 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2087 = stablehlo.slice %v2086 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2088 = stablehlo.reshape %v2087 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2089 = stablehlo.reshape %v1703 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2090 = stablehlo.slice %v2089 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2091 = stablehlo.reshape %v2090 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2092 = stablehlo.reshape %v2088 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2093 = stablehlo.transpose %v2092, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2094 = stablehlo.reshape %v2093 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2095 = stablehlo.reshape %v2085 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2096 = stablehlo.reshape %v2094 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2097 = stablehlo.dot_general %v2095, %v2096, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2098 = stablehlo.reshape %v2097 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2099 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2100 = stablehlo.multiply %v2098, %v2099 : tensor<32x38809xf32>
    %v2101 = stablehlo.reshape %v2100 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2102 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v2103 = stablehlo.reduce(%v2101 init: %v2102) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2104 = stablehlo.broadcast_in_dim %v2103, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2105 = stablehlo.subtract %v2101, %v2104 : tensor<32x197x197xf32>
    %v2106 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2107 = stablehlo.exponential %v2105 : tensor<32x197x197xf32>
    %v2108 = stablehlo.reduce(%v2107 init: %v2106) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2109 = stablehlo.broadcast_in_dim %v2108, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2110 = stablehlo.divide %v2107, %v2109 : tensor<32x197x197xf32>
    %v2111 = stablehlo.reshape %v2110 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2112 = stablehlo.reshape %v2111 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2113 = stablehlo.reshape %v2091 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2114 = stablehlo.dot_general %v2112, %v2113, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2115 = stablehlo.reshape %v2114 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2116 = stablehlo.reshape %v2115 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2117 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2118 = stablehlo.pad %v2116, %v2117, low = [0, 0, 640], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v2119 = stablehlo.reshape %v2118 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2120 = stablehlo.add %v2082, %v2119 : tensor<32x151296xf32>
    %v2121 = stablehlo.reshape %v1693 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2122 = stablehlo.slice %v2121 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2123 = stablehlo.reshape %v2122 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2124 = stablehlo.reshape %v1698 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2125 = stablehlo.slice %v2124 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2126 = stablehlo.reshape %v2125 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2127 = stablehlo.reshape %v1703 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2128 = stablehlo.slice %v2127 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2129 = stablehlo.reshape %v2128 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2130 = stablehlo.reshape %v2126 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2131 = stablehlo.transpose %v2130, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2132 = stablehlo.reshape %v2131 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2133 = stablehlo.reshape %v2123 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2134 = stablehlo.reshape %v2132 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2135 = stablehlo.dot_general %v2133, %v2134, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2136 = stablehlo.reshape %v2135 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2137 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2138 = stablehlo.multiply %v2136, %v2137 : tensor<32x38809xf32>
    %v2139 = stablehlo.reshape %v2138 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2140 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v2141 = stablehlo.reduce(%v2139 init: %v2140) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2142 = stablehlo.broadcast_in_dim %v2141, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2143 = stablehlo.subtract %v2139, %v2142 : tensor<32x197x197xf32>
    %v2144 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2145 = stablehlo.exponential %v2143 : tensor<32x197x197xf32>
    %v2146 = stablehlo.reduce(%v2145 init: %v2144) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2147 = stablehlo.broadcast_in_dim %v2146, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2148 = stablehlo.divide %v2145, %v2147 : tensor<32x197x197xf32>
    %v2149 = stablehlo.reshape %v2148 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2150 = stablehlo.reshape %v2149 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2151 = stablehlo.reshape %v2129 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2152 = stablehlo.dot_general %v2150, %v2151, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2153 = stablehlo.reshape %v2152 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2154 = stablehlo.reshape %v2153 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2155 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2156 = stablehlo.pad %v2154, %v2155, low = [0, 0, 704], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v2157 = stablehlo.reshape %v2156 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2158 = stablehlo.add %v2120, %v2157 : tensor<32x151296xf32>
    %v2159 = stablehlo.reshape %v2158 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2160 = stablehlo.dot_general %v2159, %b3_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v2161 = stablehlo.broadcast_in_dim %b3_bo, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v2162 = stablehlo.add %v2160, %v2161 : tensor<32x197x768xf32>
    %v2163 = stablehlo.reshape %v2162 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2164 = stablehlo.add %v1660, %v2163 : tensor<32x151296xf32>
    %v2165 = stablehlo.reshape %v2164 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2166 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2167 = stablehlo.constant dense<768.0> : tensor<32x197x768xf32>
    %v2168 = stablehlo.constant dense<1.0e-6> : tensor<32x197x768xf32>
    %v2169 = stablehlo.reduce(%v2165 init: %v2166) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2170 = stablehlo.broadcast_in_dim %v2169, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v2171 = stablehlo.divide %v2170, %v2167 : tensor<32x197x768xf32>
    %v2172 = stablehlo.subtract %v2165, %v2171 : tensor<32x197x768xf32>
    %v2173 = stablehlo.multiply %v2172, %v2172 : tensor<32x197x768xf32>
    %v2174 = stablehlo.reduce(%v2173 init: %v2166) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2175 = stablehlo.broadcast_in_dim %v2174, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v2176 = stablehlo.divide %v2175, %v2167 : tensor<32x197x768xf32>
    %v2177 = stablehlo.add %v2176, %v2168 : tensor<32x197x768xf32>
    %v2178 = stablehlo.rsqrt %v2177 : tensor<32x197x768xf32>
    %v2179 = stablehlo.multiply %v2172, %v2178 : tensor<32x197x768xf32>
    %v2180 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v2181 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v2182 = stablehlo.multiply %v2179, %v2180 : tensor<32x197x768xf32>
    %v2183 = stablehlo.add %v2182, %v2181 : tensor<32x197x768xf32>
    %v2184 = stablehlo.reshape %v2183 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2185 = stablehlo.reshape %v2184 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2186 = stablehlo.broadcast_in_dim %b3_g2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v2187 = stablehlo.multiply %v2185, %v2186 : tensor<32x197x768xf32>
    %v2188 = stablehlo.reshape %v2187 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2189 = stablehlo.reshape %v2188 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2190 = stablehlo.broadcast_in_dim %b3_bt2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v2191 = stablehlo.add %v2189, %v2190 : tensor<32x197x768xf32>
    %v2192 = stablehlo.reshape %v2191 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2193 = stablehlo.reshape %v2192 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2194 = stablehlo.dot_general %v2193, %b3_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x3072xf32>) -> tensor<32x197x3072xf32>
    %v2195 = stablehlo.broadcast_in_dim %b3_bfc1, dims = [2] : (tensor<3072xf32>) -> tensor<32x197x3072xf32>
    %v2196 = stablehlo.add %v2194, %v2195 : tensor<32x197x3072xf32>
    %v2197 = stablehlo.reshape %v2196 : (tensor<32x197x3072xf32>) -> tensor<32x605184xf32>
    %v2198 = stablehlo.constant dense<0.5> : tensor<32x605184xf32>
    %v2199 = stablehlo.multiply %v2198, %v2197 : tensor<32x605184xf32>
    %v2200 = stablehlo.negate %v2197 : tensor<32x605184xf32>
    %v2201 = stablehlo.constant dense<0.7071067811865476> : tensor<32x605184xf32>
    %v2202 = stablehlo.multiply %v2200, %v2201 : tensor<32x605184xf32>
    %v2203 = chlo.erfc %v2202 : tensor<32x605184xf32> -> tensor<32x605184xf32>
    %v2204 = stablehlo.multiply %v2199, %v2203 : tensor<32x605184xf32>
    %v2205 = stablehlo.reshape %v2204 : (tensor<32x605184xf32>) -> tensor<32x197x3072xf32>
    %v2206 = stablehlo.dot_general %v2205, %b3_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x3072xf32>, tensor<3072x768xf32>) -> tensor<32x197x768xf32>
    %v2207 = stablehlo.broadcast_in_dim %b3_bfc2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v2208 = stablehlo.add %v2206, %v2207 : tensor<32x197x768xf32>
    %v2209 = stablehlo.reshape %v2208 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2210 = stablehlo.add %v2164, %v2209 : tensor<32x151296xf32>
    %v2211 = stablehlo.reshape %v2210 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2212 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2213 = stablehlo.constant dense<768.0> : tensor<32x197x768xf32>
    %v2214 = stablehlo.constant dense<1.0e-6> : tensor<32x197x768xf32>
    %v2215 = stablehlo.reduce(%v2211 init: %v2212) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2216 = stablehlo.broadcast_in_dim %v2215, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v2217 = stablehlo.divide %v2216, %v2213 : tensor<32x197x768xf32>
    %v2218 = stablehlo.subtract %v2211, %v2217 : tensor<32x197x768xf32>
    %v2219 = stablehlo.multiply %v2218, %v2218 : tensor<32x197x768xf32>
    %v2220 = stablehlo.reduce(%v2219 init: %v2212) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2221 = stablehlo.broadcast_in_dim %v2220, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v2222 = stablehlo.divide %v2221, %v2213 : tensor<32x197x768xf32>
    %v2223 = stablehlo.add %v2222, %v2214 : tensor<32x197x768xf32>
    %v2224 = stablehlo.rsqrt %v2223 : tensor<32x197x768xf32>
    %v2225 = stablehlo.multiply %v2218, %v2224 : tensor<32x197x768xf32>
    %v2226 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v2227 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v2228 = stablehlo.multiply %v2225, %v2226 : tensor<32x197x768xf32>
    %v2229 = stablehlo.add %v2228, %v2227 : tensor<32x197x768xf32>
    %v2230 = stablehlo.reshape %v2229 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2231 = stablehlo.reshape %v2230 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2232 = stablehlo.broadcast_in_dim %b4_g1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v2233 = stablehlo.multiply %v2231, %v2232 : tensor<32x197x768xf32>
    %v2234 = stablehlo.reshape %v2233 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2235 = stablehlo.reshape %v2234 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2236 = stablehlo.broadcast_in_dim %b4_bt1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v2237 = stablehlo.add %v2235, %v2236 : tensor<32x197x768xf32>
    %v2238 = stablehlo.reshape %v2237 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2239 = stablehlo.reshape %v2238 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2240 = stablehlo.dot_general %v2239, %b4_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v2241 = stablehlo.broadcast_in_dim %b4_bq, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v2242 = stablehlo.add %v2240, %v2241 : tensor<32x197x768xf32>
    %v2243 = stablehlo.reshape %v2242 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2244 = stablehlo.reshape %v2238 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2245 = stablehlo.dot_general %v2244, %b4_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v2246 = stablehlo.broadcast_in_dim %b4_bk, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v2247 = stablehlo.add %v2245, %v2246 : tensor<32x197x768xf32>
    %v2248 = stablehlo.reshape %v2247 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2249 = stablehlo.reshape %v2238 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2250 = stablehlo.dot_general %v2249, %b4_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v2251 = stablehlo.broadcast_in_dim %b4_bv, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v2252 = stablehlo.add %v2250, %v2251 : tensor<32x197x768xf32>
    %v2253 = stablehlo.reshape %v2252 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2254 = stablehlo.reshape %v2243 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2255 = stablehlo.slice %v2254 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2256 = stablehlo.reshape %v2255 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2257 = stablehlo.reshape %v2248 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2258 = stablehlo.slice %v2257 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2259 = stablehlo.reshape %v2258 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2260 = stablehlo.reshape %v2253 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2261 = stablehlo.slice %v2260 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2262 = stablehlo.reshape %v2261 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2263 = stablehlo.reshape %v2259 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2264 = stablehlo.transpose %v2263, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2265 = stablehlo.reshape %v2264 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2266 = stablehlo.reshape %v2256 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2267 = stablehlo.reshape %v2265 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2268 = stablehlo.dot_general %v2266, %v2267, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2269 = stablehlo.reshape %v2268 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2270 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2271 = stablehlo.multiply %v2269, %v2270 : tensor<32x38809xf32>
    %v2272 = stablehlo.reshape %v2271 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2273 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v2274 = stablehlo.reduce(%v2272 init: %v2273) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2275 = stablehlo.broadcast_in_dim %v2274, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2276 = stablehlo.subtract %v2272, %v2275 : tensor<32x197x197xf32>
    %v2277 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2278 = stablehlo.exponential %v2276 : tensor<32x197x197xf32>
    %v2279 = stablehlo.reduce(%v2278 init: %v2277) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2280 = stablehlo.broadcast_in_dim %v2279, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2281 = stablehlo.divide %v2278, %v2280 : tensor<32x197x197xf32>
    %v2282 = stablehlo.reshape %v2281 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2283 = stablehlo.reshape %v2282 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2284 = stablehlo.reshape %v2262 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2285 = stablehlo.dot_general %v2283, %v2284, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2286 = stablehlo.reshape %v2285 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2287 = stablehlo.reshape %v2286 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2288 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2289 = stablehlo.pad %v2287, %v2288, low = [0, 0, 0], high = [0, 0, 704], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v2290 = stablehlo.reshape %v2289 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2291 = stablehlo.reshape %v2243 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2292 = stablehlo.slice %v2291 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2293 = stablehlo.reshape %v2292 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2294 = stablehlo.reshape %v2248 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2295 = stablehlo.slice %v2294 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2296 = stablehlo.reshape %v2295 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2297 = stablehlo.reshape %v2253 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2298 = stablehlo.slice %v2297 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2299 = stablehlo.reshape %v2298 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2300 = stablehlo.reshape %v2296 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2301 = stablehlo.transpose %v2300, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2302 = stablehlo.reshape %v2301 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2303 = stablehlo.reshape %v2293 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2304 = stablehlo.reshape %v2302 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2305 = stablehlo.dot_general %v2303, %v2304, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2306 = stablehlo.reshape %v2305 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2307 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2308 = stablehlo.multiply %v2306, %v2307 : tensor<32x38809xf32>
    %v2309 = stablehlo.reshape %v2308 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2310 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v2311 = stablehlo.reduce(%v2309 init: %v2310) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2312 = stablehlo.broadcast_in_dim %v2311, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2313 = stablehlo.subtract %v2309, %v2312 : tensor<32x197x197xf32>
    %v2314 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2315 = stablehlo.exponential %v2313 : tensor<32x197x197xf32>
    %v2316 = stablehlo.reduce(%v2315 init: %v2314) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2317 = stablehlo.broadcast_in_dim %v2316, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2318 = stablehlo.divide %v2315, %v2317 : tensor<32x197x197xf32>
    %v2319 = stablehlo.reshape %v2318 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2320 = stablehlo.reshape %v2319 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2321 = stablehlo.reshape %v2299 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2322 = stablehlo.dot_general %v2320, %v2321, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2323 = stablehlo.reshape %v2322 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2324 = stablehlo.reshape %v2323 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2325 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2326 = stablehlo.pad %v2324, %v2325, low = [0, 0, 64], high = [0, 0, 640], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v2327 = stablehlo.reshape %v2326 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2328 = stablehlo.add %v2290, %v2327 : tensor<32x151296xf32>
    %v2329 = stablehlo.reshape %v2243 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2330 = stablehlo.slice %v2329 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2331 = stablehlo.reshape %v2330 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2332 = stablehlo.reshape %v2248 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2333 = stablehlo.slice %v2332 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2334 = stablehlo.reshape %v2333 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2335 = stablehlo.reshape %v2253 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2336 = stablehlo.slice %v2335 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2337 = stablehlo.reshape %v2336 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2338 = stablehlo.reshape %v2334 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2339 = stablehlo.transpose %v2338, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2340 = stablehlo.reshape %v2339 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2341 = stablehlo.reshape %v2331 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2342 = stablehlo.reshape %v2340 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2343 = stablehlo.dot_general %v2341, %v2342, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2344 = stablehlo.reshape %v2343 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2345 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2346 = stablehlo.multiply %v2344, %v2345 : tensor<32x38809xf32>
    %v2347 = stablehlo.reshape %v2346 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2348 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v2349 = stablehlo.reduce(%v2347 init: %v2348) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2350 = stablehlo.broadcast_in_dim %v2349, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2351 = stablehlo.subtract %v2347, %v2350 : tensor<32x197x197xf32>
    %v2352 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2353 = stablehlo.exponential %v2351 : tensor<32x197x197xf32>
    %v2354 = stablehlo.reduce(%v2353 init: %v2352) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2355 = stablehlo.broadcast_in_dim %v2354, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2356 = stablehlo.divide %v2353, %v2355 : tensor<32x197x197xf32>
    %v2357 = stablehlo.reshape %v2356 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2358 = stablehlo.reshape %v2357 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2359 = stablehlo.reshape %v2337 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2360 = stablehlo.dot_general %v2358, %v2359, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2361 = stablehlo.reshape %v2360 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2362 = stablehlo.reshape %v2361 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2363 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2364 = stablehlo.pad %v2362, %v2363, low = [0, 0, 128], high = [0, 0, 576], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v2365 = stablehlo.reshape %v2364 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2366 = stablehlo.add %v2328, %v2365 : tensor<32x151296xf32>
    %v2367 = stablehlo.reshape %v2243 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2368 = stablehlo.slice %v2367 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2369 = stablehlo.reshape %v2368 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2370 = stablehlo.reshape %v2248 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2371 = stablehlo.slice %v2370 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2372 = stablehlo.reshape %v2371 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2373 = stablehlo.reshape %v2253 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2374 = stablehlo.slice %v2373 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
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
    %v2386 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v2387 = stablehlo.reduce(%v2385 init: %v2386) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2388 = stablehlo.broadcast_in_dim %v2387, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2389 = stablehlo.subtract %v2385, %v2388 : tensor<32x197x197xf32>
    %v2390 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2391 = stablehlo.exponential %v2389 : tensor<32x197x197xf32>
    %v2392 = stablehlo.reduce(%v2391 init: %v2390) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2393 = stablehlo.broadcast_in_dim %v2392, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2394 = stablehlo.divide %v2391, %v2393 : tensor<32x197x197xf32>
    %v2395 = stablehlo.reshape %v2394 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2396 = stablehlo.reshape %v2395 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2397 = stablehlo.reshape %v2375 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2398 = stablehlo.dot_general %v2396, %v2397, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2399 = stablehlo.reshape %v2398 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2400 = stablehlo.reshape %v2399 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2401 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2402 = stablehlo.pad %v2400, %v2401, low = [0, 0, 192], high = [0, 0, 512], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v2403 = stablehlo.reshape %v2402 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2404 = stablehlo.add %v2366, %v2403 : tensor<32x151296xf32>
    %v2405 = stablehlo.reshape %v2243 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2406 = stablehlo.slice %v2405 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2407 = stablehlo.reshape %v2406 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2408 = stablehlo.reshape %v2248 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2409 = stablehlo.slice %v2408 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2410 = stablehlo.reshape %v2409 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2411 = stablehlo.reshape %v2253 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2412 = stablehlo.slice %v2411 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2413 = stablehlo.reshape %v2412 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2414 = stablehlo.reshape %v2410 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2415 = stablehlo.transpose %v2414, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2416 = stablehlo.reshape %v2415 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2417 = stablehlo.reshape %v2407 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2418 = stablehlo.reshape %v2416 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2419 = stablehlo.dot_general %v2417, %v2418, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2420 = stablehlo.reshape %v2419 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2421 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2422 = stablehlo.multiply %v2420, %v2421 : tensor<32x38809xf32>
    %v2423 = stablehlo.reshape %v2422 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2424 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v2425 = stablehlo.reduce(%v2423 init: %v2424) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2426 = stablehlo.broadcast_in_dim %v2425, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2427 = stablehlo.subtract %v2423, %v2426 : tensor<32x197x197xf32>
    %v2428 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2429 = stablehlo.exponential %v2427 : tensor<32x197x197xf32>
    %v2430 = stablehlo.reduce(%v2429 init: %v2428) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2431 = stablehlo.broadcast_in_dim %v2430, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2432 = stablehlo.divide %v2429, %v2431 : tensor<32x197x197xf32>
    %v2433 = stablehlo.reshape %v2432 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2434 = stablehlo.reshape %v2433 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2435 = stablehlo.reshape %v2413 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2436 = stablehlo.dot_general %v2434, %v2435, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2437 = stablehlo.reshape %v2436 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2438 = stablehlo.reshape %v2437 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2439 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2440 = stablehlo.pad %v2438, %v2439, low = [0, 0, 256], high = [0, 0, 448], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v2441 = stablehlo.reshape %v2440 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2442 = stablehlo.add %v2404, %v2441 : tensor<32x151296xf32>
    %v2443 = stablehlo.reshape %v2243 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2444 = stablehlo.slice %v2443 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2445 = stablehlo.reshape %v2444 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2446 = stablehlo.reshape %v2248 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2447 = stablehlo.slice %v2446 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2448 = stablehlo.reshape %v2447 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2449 = stablehlo.reshape %v2253 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2450 = stablehlo.slice %v2449 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2451 = stablehlo.reshape %v2450 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2452 = stablehlo.reshape %v2448 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2453 = stablehlo.transpose %v2452, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2454 = stablehlo.reshape %v2453 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2455 = stablehlo.reshape %v2445 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2456 = stablehlo.reshape %v2454 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2457 = stablehlo.dot_general %v2455, %v2456, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2458 = stablehlo.reshape %v2457 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2459 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2460 = stablehlo.multiply %v2458, %v2459 : tensor<32x38809xf32>
    %v2461 = stablehlo.reshape %v2460 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2462 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v2463 = stablehlo.reduce(%v2461 init: %v2462) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2464 = stablehlo.broadcast_in_dim %v2463, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2465 = stablehlo.subtract %v2461, %v2464 : tensor<32x197x197xf32>
    %v2466 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2467 = stablehlo.exponential %v2465 : tensor<32x197x197xf32>
    %v2468 = stablehlo.reduce(%v2467 init: %v2466) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2469 = stablehlo.broadcast_in_dim %v2468, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2470 = stablehlo.divide %v2467, %v2469 : tensor<32x197x197xf32>
    %v2471 = stablehlo.reshape %v2470 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2472 = stablehlo.reshape %v2471 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2473 = stablehlo.reshape %v2451 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2474 = stablehlo.dot_general %v2472, %v2473, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2475 = stablehlo.reshape %v2474 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2476 = stablehlo.reshape %v2475 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2477 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2478 = stablehlo.pad %v2476, %v2477, low = [0, 0, 320], high = [0, 0, 384], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v2479 = stablehlo.reshape %v2478 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2480 = stablehlo.add %v2442, %v2479 : tensor<32x151296xf32>
    %v2481 = stablehlo.reshape %v2243 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2482 = stablehlo.slice %v2481 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2483 = stablehlo.reshape %v2482 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2484 = stablehlo.reshape %v2248 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2485 = stablehlo.slice %v2484 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2486 = stablehlo.reshape %v2485 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2487 = stablehlo.reshape %v2253 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2488 = stablehlo.slice %v2487 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2489 = stablehlo.reshape %v2488 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2490 = stablehlo.reshape %v2486 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2491 = stablehlo.transpose %v2490, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2492 = stablehlo.reshape %v2491 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2493 = stablehlo.reshape %v2483 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2494 = stablehlo.reshape %v2492 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2495 = stablehlo.dot_general %v2493, %v2494, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2496 = stablehlo.reshape %v2495 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2497 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2498 = stablehlo.multiply %v2496, %v2497 : tensor<32x38809xf32>
    %v2499 = stablehlo.reshape %v2498 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2500 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v2501 = stablehlo.reduce(%v2499 init: %v2500) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2502 = stablehlo.broadcast_in_dim %v2501, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2503 = stablehlo.subtract %v2499, %v2502 : tensor<32x197x197xf32>
    %v2504 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2505 = stablehlo.exponential %v2503 : tensor<32x197x197xf32>
    %v2506 = stablehlo.reduce(%v2505 init: %v2504) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2507 = stablehlo.broadcast_in_dim %v2506, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2508 = stablehlo.divide %v2505, %v2507 : tensor<32x197x197xf32>
    %v2509 = stablehlo.reshape %v2508 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2510 = stablehlo.reshape %v2509 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2511 = stablehlo.reshape %v2489 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2512 = stablehlo.dot_general %v2510, %v2511, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2513 = stablehlo.reshape %v2512 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2514 = stablehlo.reshape %v2513 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2515 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2516 = stablehlo.pad %v2514, %v2515, low = [0, 0, 384], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v2517 = stablehlo.reshape %v2516 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2518 = stablehlo.add %v2480, %v2517 : tensor<32x151296xf32>
    %v2519 = stablehlo.reshape %v2243 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2520 = stablehlo.slice %v2519 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2521 = stablehlo.reshape %v2520 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2522 = stablehlo.reshape %v2248 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2523 = stablehlo.slice %v2522 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2524 = stablehlo.reshape %v2523 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2525 = stablehlo.reshape %v2253 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2526 = stablehlo.slice %v2525 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2527 = stablehlo.reshape %v2526 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2528 = stablehlo.reshape %v2524 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2529 = stablehlo.transpose %v2528, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2530 = stablehlo.reshape %v2529 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2531 = stablehlo.reshape %v2521 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2532 = stablehlo.reshape %v2530 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2533 = stablehlo.dot_general %v2531, %v2532, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2534 = stablehlo.reshape %v2533 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2535 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2536 = stablehlo.multiply %v2534, %v2535 : tensor<32x38809xf32>
    %v2537 = stablehlo.reshape %v2536 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2538 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v2539 = stablehlo.reduce(%v2537 init: %v2538) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2540 = stablehlo.broadcast_in_dim %v2539, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2541 = stablehlo.subtract %v2537, %v2540 : tensor<32x197x197xf32>
    %v2542 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2543 = stablehlo.exponential %v2541 : tensor<32x197x197xf32>
    %v2544 = stablehlo.reduce(%v2543 init: %v2542) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2545 = stablehlo.broadcast_in_dim %v2544, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2546 = stablehlo.divide %v2543, %v2545 : tensor<32x197x197xf32>
    %v2547 = stablehlo.reshape %v2546 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2548 = stablehlo.reshape %v2547 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2549 = stablehlo.reshape %v2527 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2550 = stablehlo.dot_general %v2548, %v2549, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2551 = stablehlo.reshape %v2550 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2552 = stablehlo.reshape %v2551 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2553 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2554 = stablehlo.pad %v2552, %v2553, low = [0, 0, 448], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v2555 = stablehlo.reshape %v2554 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2556 = stablehlo.add %v2518, %v2555 : tensor<32x151296xf32>
    %v2557 = stablehlo.reshape %v2243 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2558 = stablehlo.slice %v2557 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2559 = stablehlo.reshape %v2558 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2560 = stablehlo.reshape %v2248 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2561 = stablehlo.slice %v2560 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2562 = stablehlo.reshape %v2561 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2563 = stablehlo.reshape %v2253 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2564 = stablehlo.slice %v2563 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2565 = stablehlo.reshape %v2564 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2566 = stablehlo.reshape %v2562 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2567 = stablehlo.transpose %v2566, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2568 = stablehlo.reshape %v2567 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2569 = stablehlo.reshape %v2559 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2570 = stablehlo.reshape %v2568 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2571 = stablehlo.dot_general %v2569, %v2570, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2572 = stablehlo.reshape %v2571 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2573 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2574 = stablehlo.multiply %v2572, %v2573 : tensor<32x38809xf32>
    %v2575 = stablehlo.reshape %v2574 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2576 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v2577 = stablehlo.reduce(%v2575 init: %v2576) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2578 = stablehlo.broadcast_in_dim %v2577, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2579 = stablehlo.subtract %v2575, %v2578 : tensor<32x197x197xf32>
    %v2580 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2581 = stablehlo.exponential %v2579 : tensor<32x197x197xf32>
    %v2582 = stablehlo.reduce(%v2581 init: %v2580) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2583 = stablehlo.broadcast_in_dim %v2582, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2584 = stablehlo.divide %v2581, %v2583 : tensor<32x197x197xf32>
    %v2585 = stablehlo.reshape %v2584 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2586 = stablehlo.reshape %v2585 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2587 = stablehlo.reshape %v2565 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2588 = stablehlo.dot_general %v2586, %v2587, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2589 = stablehlo.reshape %v2588 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2590 = stablehlo.reshape %v2589 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2591 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2592 = stablehlo.pad %v2590, %v2591, low = [0, 0, 512], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v2593 = stablehlo.reshape %v2592 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2594 = stablehlo.add %v2556, %v2593 : tensor<32x151296xf32>
    %v2595 = stablehlo.reshape %v2243 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2596 = stablehlo.slice %v2595 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2597 = stablehlo.reshape %v2596 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2598 = stablehlo.reshape %v2248 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2599 = stablehlo.slice %v2598 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2600 = stablehlo.reshape %v2599 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2601 = stablehlo.reshape %v2253 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2602 = stablehlo.slice %v2601 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2603 = stablehlo.reshape %v2602 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2604 = stablehlo.reshape %v2600 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2605 = stablehlo.transpose %v2604, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2606 = stablehlo.reshape %v2605 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2607 = stablehlo.reshape %v2597 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2608 = stablehlo.reshape %v2606 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2609 = stablehlo.dot_general %v2607, %v2608, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2610 = stablehlo.reshape %v2609 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2611 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2612 = stablehlo.multiply %v2610, %v2611 : tensor<32x38809xf32>
    %v2613 = stablehlo.reshape %v2612 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2614 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v2615 = stablehlo.reduce(%v2613 init: %v2614) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2616 = stablehlo.broadcast_in_dim %v2615, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2617 = stablehlo.subtract %v2613, %v2616 : tensor<32x197x197xf32>
    %v2618 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2619 = stablehlo.exponential %v2617 : tensor<32x197x197xf32>
    %v2620 = stablehlo.reduce(%v2619 init: %v2618) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2621 = stablehlo.broadcast_in_dim %v2620, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2622 = stablehlo.divide %v2619, %v2621 : tensor<32x197x197xf32>
    %v2623 = stablehlo.reshape %v2622 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2624 = stablehlo.reshape %v2623 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2625 = stablehlo.reshape %v2603 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2626 = stablehlo.dot_general %v2624, %v2625, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2627 = stablehlo.reshape %v2626 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2628 = stablehlo.reshape %v2627 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2629 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2630 = stablehlo.pad %v2628, %v2629, low = [0, 0, 576], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v2631 = stablehlo.reshape %v2630 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2632 = stablehlo.add %v2594, %v2631 : tensor<32x151296xf32>
    %v2633 = stablehlo.reshape %v2243 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2634 = stablehlo.slice %v2633 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2635 = stablehlo.reshape %v2634 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2636 = stablehlo.reshape %v2248 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2637 = stablehlo.slice %v2636 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2638 = stablehlo.reshape %v2637 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2639 = stablehlo.reshape %v2253 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2640 = stablehlo.slice %v2639 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2641 = stablehlo.reshape %v2640 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2642 = stablehlo.reshape %v2638 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2643 = stablehlo.transpose %v2642, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2644 = stablehlo.reshape %v2643 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2645 = stablehlo.reshape %v2635 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2646 = stablehlo.reshape %v2644 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2647 = stablehlo.dot_general %v2645, %v2646, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2648 = stablehlo.reshape %v2647 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2649 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2650 = stablehlo.multiply %v2648, %v2649 : tensor<32x38809xf32>
    %v2651 = stablehlo.reshape %v2650 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2652 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v2653 = stablehlo.reduce(%v2651 init: %v2652) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2654 = stablehlo.broadcast_in_dim %v2653, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2655 = stablehlo.subtract %v2651, %v2654 : tensor<32x197x197xf32>
    %v2656 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2657 = stablehlo.exponential %v2655 : tensor<32x197x197xf32>
    %v2658 = stablehlo.reduce(%v2657 init: %v2656) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2659 = stablehlo.broadcast_in_dim %v2658, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2660 = stablehlo.divide %v2657, %v2659 : tensor<32x197x197xf32>
    %v2661 = stablehlo.reshape %v2660 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2662 = stablehlo.reshape %v2661 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2663 = stablehlo.reshape %v2641 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2664 = stablehlo.dot_general %v2662, %v2663, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2665 = stablehlo.reshape %v2664 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2666 = stablehlo.reshape %v2665 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2667 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2668 = stablehlo.pad %v2666, %v2667, low = [0, 0, 640], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v2669 = stablehlo.reshape %v2668 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2670 = stablehlo.add %v2632, %v2669 : tensor<32x151296xf32>
    %v2671 = stablehlo.reshape %v2243 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2672 = stablehlo.slice %v2671 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2673 = stablehlo.reshape %v2672 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2674 = stablehlo.reshape %v2248 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2675 = stablehlo.slice %v2674 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2676 = stablehlo.reshape %v2675 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2677 = stablehlo.reshape %v2253 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2678 = stablehlo.slice %v2677 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2679 = stablehlo.reshape %v2678 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2680 = stablehlo.reshape %v2676 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2681 = stablehlo.transpose %v2680, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2682 = stablehlo.reshape %v2681 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2683 = stablehlo.reshape %v2673 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2684 = stablehlo.reshape %v2682 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2685 = stablehlo.dot_general %v2683, %v2684, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2686 = stablehlo.reshape %v2685 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2687 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2688 = stablehlo.multiply %v2686, %v2687 : tensor<32x38809xf32>
    %v2689 = stablehlo.reshape %v2688 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2690 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v2691 = stablehlo.reduce(%v2689 init: %v2690) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2692 = stablehlo.broadcast_in_dim %v2691, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2693 = stablehlo.subtract %v2689, %v2692 : tensor<32x197x197xf32>
    %v2694 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2695 = stablehlo.exponential %v2693 : tensor<32x197x197xf32>
    %v2696 = stablehlo.reduce(%v2695 init: %v2694) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2697 = stablehlo.broadcast_in_dim %v2696, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2698 = stablehlo.divide %v2695, %v2697 : tensor<32x197x197xf32>
    %v2699 = stablehlo.reshape %v2698 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2700 = stablehlo.reshape %v2699 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2701 = stablehlo.reshape %v2679 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2702 = stablehlo.dot_general %v2700, %v2701, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2703 = stablehlo.reshape %v2702 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2704 = stablehlo.reshape %v2703 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2705 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2706 = stablehlo.pad %v2704, %v2705, low = [0, 0, 704], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v2707 = stablehlo.reshape %v2706 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2708 = stablehlo.add %v2670, %v2707 : tensor<32x151296xf32>
    %v2709 = stablehlo.reshape %v2708 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2710 = stablehlo.dot_general %v2709, %b4_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v2711 = stablehlo.broadcast_in_dim %b4_bo, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v2712 = stablehlo.add %v2710, %v2711 : tensor<32x197x768xf32>
    %v2713 = stablehlo.reshape %v2712 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2714 = stablehlo.add %v2210, %v2713 : tensor<32x151296xf32>
    %v2715 = stablehlo.reshape %v2714 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2716 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2717 = stablehlo.constant dense<768.0> : tensor<32x197x768xf32>
    %v2718 = stablehlo.constant dense<1.0e-6> : tensor<32x197x768xf32>
    %v2719 = stablehlo.reduce(%v2715 init: %v2716) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2720 = stablehlo.broadcast_in_dim %v2719, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v2721 = stablehlo.divide %v2720, %v2717 : tensor<32x197x768xf32>
    %v2722 = stablehlo.subtract %v2715, %v2721 : tensor<32x197x768xf32>
    %v2723 = stablehlo.multiply %v2722, %v2722 : tensor<32x197x768xf32>
    %v2724 = stablehlo.reduce(%v2723 init: %v2716) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2725 = stablehlo.broadcast_in_dim %v2724, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v2726 = stablehlo.divide %v2725, %v2717 : tensor<32x197x768xf32>
    %v2727 = stablehlo.add %v2726, %v2718 : tensor<32x197x768xf32>
    %v2728 = stablehlo.rsqrt %v2727 : tensor<32x197x768xf32>
    %v2729 = stablehlo.multiply %v2722, %v2728 : tensor<32x197x768xf32>
    %v2730 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v2731 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v2732 = stablehlo.multiply %v2729, %v2730 : tensor<32x197x768xf32>
    %v2733 = stablehlo.add %v2732, %v2731 : tensor<32x197x768xf32>
    %v2734 = stablehlo.reshape %v2733 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2735 = stablehlo.reshape %v2734 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2736 = stablehlo.broadcast_in_dim %b4_g2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v2737 = stablehlo.multiply %v2735, %v2736 : tensor<32x197x768xf32>
    %v2738 = stablehlo.reshape %v2737 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2739 = stablehlo.reshape %v2738 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2740 = stablehlo.broadcast_in_dim %b4_bt2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v2741 = stablehlo.add %v2739, %v2740 : tensor<32x197x768xf32>
    %v2742 = stablehlo.reshape %v2741 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2743 = stablehlo.reshape %v2742 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2744 = stablehlo.dot_general %v2743, %b4_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x3072xf32>) -> tensor<32x197x3072xf32>
    %v2745 = stablehlo.broadcast_in_dim %b4_bfc1, dims = [2] : (tensor<3072xf32>) -> tensor<32x197x3072xf32>
    %v2746 = stablehlo.add %v2744, %v2745 : tensor<32x197x3072xf32>
    %v2747 = stablehlo.reshape %v2746 : (tensor<32x197x3072xf32>) -> tensor<32x605184xf32>
    %v2748 = stablehlo.constant dense<0.5> : tensor<32x605184xf32>
    %v2749 = stablehlo.multiply %v2748, %v2747 : tensor<32x605184xf32>
    %v2750 = stablehlo.negate %v2747 : tensor<32x605184xf32>
    %v2751 = stablehlo.constant dense<0.7071067811865476> : tensor<32x605184xf32>
    %v2752 = stablehlo.multiply %v2750, %v2751 : tensor<32x605184xf32>
    %v2753 = chlo.erfc %v2752 : tensor<32x605184xf32> -> tensor<32x605184xf32>
    %v2754 = stablehlo.multiply %v2749, %v2753 : tensor<32x605184xf32>
    %v2755 = stablehlo.reshape %v2754 : (tensor<32x605184xf32>) -> tensor<32x197x3072xf32>
    %v2756 = stablehlo.dot_general %v2755, %b4_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x3072xf32>, tensor<3072x768xf32>) -> tensor<32x197x768xf32>
    %v2757 = stablehlo.broadcast_in_dim %b4_bfc2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v2758 = stablehlo.add %v2756, %v2757 : tensor<32x197x768xf32>
    %v2759 = stablehlo.reshape %v2758 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2760 = stablehlo.add %v2714, %v2759 : tensor<32x151296xf32>
    %v2761 = stablehlo.reshape %v2760 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2762 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2763 = stablehlo.constant dense<768.0> : tensor<32x197x768xf32>
    %v2764 = stablehlo.constant dense<1.0e-6> : tensor<32x197x768xf32>
    %v2765 = stablehlo.reduce(%v2761 init: %v2762) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2766 = stablehlo.broadcast_in_dim %v2765, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v2767 = stablehlo.divide %v2766, %v2763 : tensor<32x197x768xf32>
    %v2768 = stablehlo.subtract %v2761, %v2767 : tensor<32x197x768xf32>
    %v2769 = stablehlo.multiply %v2768, %v2768 : tensor<32x197x768xf32>
    %v2770 = stablehlo.reduce(%v2769 init: %v2762) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2771 = stablehlo.broadcast_in_dim %v2770, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v2772 = stablehlo.divide %v2771, %v2763 : tensor<32x197x768xf32>
    %v2773 = stablehlo.add %v2772, %v2764 : tensor<32x197x768xf32>
    %v2774 = stablehlo.rsqrt %v2773 : tensor<32x197x768xf32>
    %v2775 = stablehlo.multiply %v2768, %v2774 : tensor<32x197x768xf32>
    %v2776 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v2777 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v2778 = stablehlo.multiply %v2775, %v2776 : tensor<32x197x768xf32>
    %v2779 = stablehlo.add %v2778, %v2777 : tensor<32x197x768xf32>
    %v2780 = stablehlo.reshape %v2779 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2781 = stablehlo.reshape %v2780 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2782 = stablehlo.broadcast_in_dim %b5_g1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v2783 = stablehlo.multiply %v2781, %v2782 : tensor<32x197x768xf32>
    %v2784 = stablehlo.reshape %v2783 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2785 = stablehlo.reshape %v2784 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2786 = stablehlo.broadcast_in_dim %b5_bt1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v2787 = stablehlo.add %v2785, %v2786 : tensor<32x197x768xf32>
    %v2788 = stablehlo.reshape %v2787 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2789 = stablehlo.reshape %v2788 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2790 = stablehlo.dot_general %v2789, %b5_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v2791 = stablehlo.broadcast_in_dim %b5_bq, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v2792 = stablehlo.add %v2790, %v2791 : tensor<32x197x768xf32>
    %v2793 = stablehlo.reshape %v2792 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2794 = stablehlo.reshape %v2788 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2795 = stablehlo.dot_general %v2794, %b5_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v2796 = stablehlo.broadcast_in_dim %b5_bk, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v2797 = stablehlo.add %v2795, %v2796 : tensor<32x197x768xf32>
    %v2798 = stablehlo.reshape %v2797 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2799 = stablehlo.reshape %v2788 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2800 = stablehlo.dot_general %v2799, %b5_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v2801 = stablehlo.broadcast_in_dim %b5_bv, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v2802 = stablehlo.add %v2800, %v2801 : tensor<32x197x768xf32>
    %v2803 = stablehlo.reshape %v2802 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2804 = stablehlo.reshape %v2793 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2805 = stablehlo.slice %v2804 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2806 = stablehlo.reshape %v2805 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2807 = stablehlo.reshape %v2798 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2808 = stablehlo.slice %v2807 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2809 = stablehlo.reshape %v2808 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2810 = stablehlo.reshape %v2803 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2811 = stablehlo.slice %v2810 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2812 = stablehlo.reshape %v2811 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2813 = stablehlo.reshape %v2809 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2814 = stablehlo.transpose %v2813, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2815 = stablehlo.reshape %v2814 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2816 = stablehlo.reshape %v2806 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2817 = stablehlo.reshape %v2815 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2818 = stablehlo.dot_general %v2816, %v2817, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2819 = stablehlo.reshape %v2818 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2820 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2821 = stablehlo.multiply %v2819, %v2820 : tensor<32x38809xf32>
    %v2822 = stablehlo.reshape %v2821 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2823 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v2824 = stablehlo.reduce(%v2822 init: %v2823) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2825 = stablehlo.broadcast_in_dim %v2824, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2826 = stablehlo.subtract %v2822, %v2825 : tensor<32x197x197xf32>
    %v2827 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2828 = stablehlo.exponential %v2826 : tensor<32x197x197xf32>
    %v2829 = stablehlo.reduce(%v2828 init: %v2827) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2830 = stablehlo.broadcast_in_dim %v2829, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2831 = stablehlo.divide %v2828, %v2830 : tensor<32x197x197xf32>
    %v2832 = stablehlo.reshape %v2831 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2833 = stablehlo.reshape %v2832 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2834 = stablehlo.reshape %v2812 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2835 = stablehlo.dot_general %v2833, %v2834, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2836 = stablehlo.reshape %v2835 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2837 = stablehlo.reshape %v2836 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2838 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2839 = stablehlo.pad %v2837, %v2838, low = [0, 0, 0], high = [0, 0, 704], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v2840 = stablehlo.reshape %v2839 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2841 = stablehlo.reshape %v2793 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2842 = stablehlo.slice %v2841 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2843 = stablehlo.reshape %v2842 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2844 = stablehlo.reshape %v2798 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2845 = stablehlo.slice %v2844 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2846 = stablehlo.reshape %v2845 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2847 = stablehlo.reshape %v2803 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2848 = stablehlo.slice %v2847 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2849 = stablehlo.reshape %v2848 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2850 = stablehlo.reshape %v2846 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2851 = stablehlo.transpose %v2850, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2852 = stablehlo.reshape %v2851 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2853 = stablehlo.reshape %v2843 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2854 = stablehlo.reshape %v2852 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2855 = stablehlo.dot_general %v2853, %v2854, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2856 = stablehlo.reshape %v2855 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2857 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2858 = stablehlo.multiply %v2856, %v2857 : tensor<32x38809xf32>
    %v2859 = stablehlo.reshape %v2858 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2860 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v2861 = stablehlo.reduce(%v2859 init: %v2860) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2862 = stablehlo.broadcast_in_dim %v2861, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2863 = stablehlo.subtract %v2859, %v2862 : tensor<32x197x197xf32>
    %v2864 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2865 = stablehlo.exponential %v2863 : tensor<32x197x197xf32>
    %v2866 = stablehlo.reduce(%v2865 init: %v2864) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2867 = stablehlo.broadcast_in_dim %v2866, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2868 = stablehlo.divide %v2865, %v2867 : tensor<32x197x197xf32>
    %v2869 = stablehlo.reshape %v2868 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2870 = stablehlo.reshape %v2869 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2871 = stablehlo.reshape %v2849 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2872 = stablehlo.dot_general %v2870, %v2871, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2873 = stablehlo.reshape %v2872 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2874 = stablehlo.reshape %v2873 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2875 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2876 = stablehlo.pad %v2874, %v2875, low = [0, 0, 64], high = [0, 0, 640], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v2877 = stablehlo.reshape %v2876 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2878 = stablehlo.add %v2840, %v2877 : tensor<32x151296xf32>
    %v2879 = stablehlo.reshape %v2793 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2880 = stablehlo.slice %v2879 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2881 = stablehlo.reshape %v2880 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2882 = stablehlo.reshape %v2798 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2883 = stablehlo.slice %v2882 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2884 = stablehlo.reshape %v2883 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2885 = stablehlo.reshape %v2803 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2886 = stablehlo.slice %v2885 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2887 = stablehlo.reshape %v2886 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2888 = stablehlo.reshape %v2884 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2889 = stablehlo.transpose %v2888, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2890 = stablehlo.reshape %v2889 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2891 = stablehlo.reshape %v2881 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2892 = stablehlo.reshape %v2890 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2893 = stablehlo.dot_general %v2891, %v2892, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2894 = stablehlo.reshape %v2893 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2895 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2896 = stablehlo.multiply %v2894, %v2895 : tensor<32x38809xf32>
    %v2897 = stablehlo.reshape %v2896 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2898 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v2899 = stablehlo.reduce(%v2897 init: %v2898) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2900 = stablehlo.broadcast_in_dim %v2899, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2901 = stablehlo.subtract %v2897, %v2900 : tensor<32x197x197xf32>
    %v2902 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2903 = stablehlo.exponential %v2901 : tensor<32x197x197xf32>
    %v2904 = stablehlo.reduce(%v2903 init: %v2902) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2905 = stablehlo.broadcast_in_dim %v2904, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2906 = stablehlo.divide %v2903, %v2905 : tensor<32x197x197xf32>
    %v2907 = stablehlo.reshape %v2906 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2908 = stablehlo.reshape %v2907 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2909 = stablehlo.reshape %v2887 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2910 = stablehlo.dot_general %v2908, %v2909, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2911 = stablehlo.reshape %v2910 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2912 = stablehlo.reshape %v2911 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2913 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2914 = stablehlo.pad %v2912, %v2913, low = [0, 0, 128], high = [0, 0, 576], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v2915 = stablehlo.reshape %v2914 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2916 = stablehlo.add %v2878, %v2915 : tensor<32x151296xf32>
    %v2917 = stablehlo.reshape %v2793 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2918 = stablehlo.slice %v2917 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2919 = stablehlo.reshape %v2918 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2920 = stablehlo.reshape %v2798 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2921 = stablehlo.slice %v2920 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2922 = stablehlo.reshape %v2921 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2923 = stablehlo.reshape %v2803 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2924 = stablehlo.slice %v2923 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2925 = stablehlo.reshape %v2924 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2926 = stablehlo.reshape %v2922 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2927 = stablehlo.transpose %v2926, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2928 = stablehlo.reshape %v2927 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2929 = stablehlo.reshape %v2919 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2930 = stablehlo.reshape %v2928 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2931 = stablehlo.dot_general %v2929, %v2930, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2932 = stablehlo.reshape %v2931 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2933 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2934 = stablehlo.multiply %v2932, %v2933 : tensor<32x38809xf32>
    %v2935 = stablehlo.reshape %v2934 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2936 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v2937 = stablehlo.reduce(%v2935 init: %v2936) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2938 = stablehlo.broadcast_in_dim %v2937, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2939 = stablehlo.subtract %v2935, %v2938 : tensor<32x197x197xf32>
    %v2940 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2941 = stablehlo.exponential %v2939 : tensor<32x197x197xf32>
    %v2942 = stablehlo.reduce(%v2941 init: %v2940) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2943 = stablehlo.broadcast_in_dim %v2942, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2944 = stablehlo.divide %v2941, %v2943 : tensor<32x197x197xf32>
    %v2945 = stablehlo.reshape %v2944 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2946 = stablehlo.reshape %v2945 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2947 = stablehlo.reshape %v2925 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2948 = stablehlo.dot_general %v2946, %v2947, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2949 = stablehlo.reshape %v2948 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2950 = stablehlo.reshape %v2949 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2951 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2952 = stablehlo.pad %v2950, %v2951, low = [0, 0, 192], high = [0, 0, 512], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v2953 = stablehlo.reshape %v2952 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2954 = stablehlo.add %v2916, %v2953 : tensor<32x151296xf32>
    %v2955 = stablehlo.reshape %v2793 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2956 = stablehlo.slice %v2955 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2957 = stablehlo.reshape %v2956 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2958 = stablehlo.reshape %v2798 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2959 = stablehlo.slice %v2958 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2960 = stablehlo.reshape %v2959 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2961 = stablehlo.reshape %v2803 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2962 = stablehlo.slice %v2961 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2963 = stablehlo.reshape %v2962 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2964 = stablehlo.reshape %v2960 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2965 = stablehlo.transpose %v2964, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2966 = stablehlo.reshape %v2965 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2967 = stablehlo.reshape %v2957 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2968 = stablehlo.reshape %v2966 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2969 = stablehlo.dot_general %v2967, %v2968, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2970 = stablehlo.reshape %v2969 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2971 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2972 = stablehlo.multiply %v2970, %v2971 : tensor<32x38809xf32>
    %v2973 = stablehlo.reshape %v2972 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2974 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v2975 = stablehlo.reduce(%v2973 init: %v2974) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2976 = stablehlo.broadcast_in_dim %v2975, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2977 = stablehlo.subtract %v2973, %v2976 : tensor<32x197x197xf32>
    %v2978 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2979 = stablehlo.exponential %v2977 : tensor<32x197x197xf32>
    %v2980 = stablehlo.reduce(%v2979 init: %v2978) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2981 = stablehlo.broadcast_in_dim %v2980, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2982 = stablehlo.divide %v2979, %v2981 : tensor<32x197x197xf32>
    %v2983 = stablehlo.reshape %v2982 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2984 = stablehlo.reshape %v2983 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2985 = stablehlo.reshape %v2963 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2986 = stablehlo.dot_general %v2984, %v2985, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2987 = stablehlo.reshape %v2986 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2988 = stablehlo.reshape %v2987 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2989 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2990 = stablehlo.pad %v2988, %v2989, low = [0, 0, 256], high = [0, 0, 448], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v2991 = stablehlo.reshape %v2990 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2992 = stablehlo.add %v2954, %v2991 : tensor<32x151296xf32>
    %v2993 = stablehlo.reshape %v2793 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2994 = stablehlo.slice %v2993 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2995 = stablehlo.reshape %v2994 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2996 = stablehlo.reshape %v2798 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2997 = stablehlo.slice %v2996 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2998 = stablehlo.reshape %v2997 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2999 = stablehlo.reshape %v2803 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3000 = stablehlo.slice %v2999 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3001 = stablehlo.reshape %v3000 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3002 = stablehlo.reshape %v2998 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3003 = stablehlo.transpose %v3002, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3004 = stablehlo.reshape %v3003 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3005 = stablehlo.reshape %v2995 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3006 = stablehlo.reshape %v3004 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3007 = stablehlo.dot_general %v3005, %v3006, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3008 = stablehlo.reshape %v3007 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3009 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3010 = stablehlo.multiply %v3008, %v3009 : tensor<32x38809xf32>
    %v3011 = stablehlo.reshape %v3010 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3012 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v3013 = stablehlo.reduce(%v3011 init: %v3012) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3014 = stablehlo.broadcast_in_dim %v3013, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3015 = stablehlo.subtract %v3011, %v3014 : tensor<32x197x197xf32>
    %v3016 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3017 = stablehlo.exponential %v3015 : tensor<32x197x197xf32>
    %v3018 = stablehlo.reduce(%v3017 init: %v3016) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3019 = stablehlo.broadcast_in_dim %v3018, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3020 = stablehlo.divide %v3017, %v3019 : tensor<32x197x197xf32>
    %v3021 = stablehlo.reshape %v3020 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3022 = stablehlo.reshape %v3021 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3023 = stablehlo.reshape %v3001 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3024 = stablehlo.dot_general %v3022, %v3023, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3025 = stablehlo.reshape %v3024 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3026 = stablehlo.reshape %v3025 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3027 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3028 = stablehlo.pad %v3026, %v3027, low = [0, 0, 320], high = [0, 0, 384], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v3029 = stablehlo.reshape %v3028 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3030 = stablehlo.add %v2992, %v3029 : tensor<32x151296xf32>
    %v3031 = stablehlo.reshape %v2793 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3032 = stablehlo.slice %v3031 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3033 = stablehlo.reshape %v3032 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3034 = stablehlo.reshape %v2798 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3035 = stablehlo.slice %v3034 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3036 = stablehlo.reshape %v3035 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3037 = stablehlo.reshape %v2803 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3038 = stablehlo.slice %v3037 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3039 = stablehlo.reshape %v3038 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3040 = stablehlo.reshape %v3036 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3041 = stablehlo.transpose %v3040, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3042 = stablehlo.reshape %v3041 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3043 = stablehlo.reshape %v3033 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3044 = stablehlo.reshape %v3042 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3045 = stablehlo.dot_general %v3043, %v3044, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3046 = stablehlo.reshape %v3045 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3047 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3048 = stablehlo.multiply %v3046, %v3047 : tensor<32x38809xf32>
    %v3049 = stablehlo.reshape %v3048 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3050 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v3051 = stablehlo.reduce(%v3049 init: %v3050) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3052 = stablehlo.broadcast_in_dim %v3051, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3053 = stablehlo.subtract %v3049, %v3052 : tensor<32x197x197xf32>
    %v3054 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3055 = stablehlo.exponential %v3053 : tensor<32x197x197xf32>
    %v3056 = stablehlo.reduce(%v3055 init: %v3054) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3057 = stablehlo.broadcast_in_dim %v3056, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3058 = stablehlo.divide %v3055, %v3057 : tensor<32x197x197xf32>
    %v3059 = stablehlo.reshape %v3058 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3060 = stablehlo.reshape %v3059 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3061 = stablehlo.reshape %v3039 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3062 = stablehlo.dot_general %v3060, %v3061, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3063 = stablehlo.reshape %v3062 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3064 = stablehlo.reshape %v3063 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3065 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3066 = stablehlo.pad %v3064, %v3065, low = [0, 0, 384], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v3067 = stablehlo.reshape %v3066 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3068 = stablehlo.add %v3030, %v3067 : tensor<32x151296xf32>
    %v3069 = stablehlo.reshape %v2793 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3070 = stablehlo.slice %v3069 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3071 = stablehlo.reshape %v3070 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3072 = stablehlo.reshape %v2798 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3073 = stablehlo.slice %v3072 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3074 = stablehlo.reshape %v3073 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3075 = stablehlo.reshape %v2803 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3076 = stablehlo.slice %v3075 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3077 = stablehlo.reshape %v3076 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3078 = stablehlo.reshape %v3074 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3079 = stablehlo.transpose %v3078, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3080 = stablehlo.reshape %v3079 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3081 = stablehlo.reshape %v3071 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3082 = stablehlo.reshape %v3080 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3083 = stablehlo.dot_general %v3081, %v3082, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3084 = stablehlo.reshape %v3083 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3085 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3086 = stablehlo.multiply %v3084, %v3085 : tensor<32x38809xf32>
    %v3087 = stablehlo.reshape %v3086 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3088 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v3089 = stablehlo.reduce(%v3087 init: %v3088) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3090 = stablehlo.broadcast_in_dim %v3089, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3091 = stablehlo.subtract %v3087, %v3090 : tensor<32x197x197xf32>
    %v3092 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3093 = stablehlo.exponential %v3091 : tensor<32x197x197xf32>
    %v3094 = stablehlo.reduce(%v3093 init: %v3092) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3095 = stablehlo.broadcast_in_dim %v3094, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3096 = stablehlo.divide %v3093, %v3095 : tensor<32x197x197xf32>
    %v3097 = stablehlo.reshape %v3096 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3098 = stablehlo.reshape %v3097 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3099 = stablehlo.reshape %v3077 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3100 = stablehlo.dot_general %v3098, %v3099, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3101 = stablehlo.reshape %v3100 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3102 = stablehlo.reshape %v3101 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3103 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3104 = stablehlo.pad %v3102, %v3103, low = [0, 0, 448], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v3105 = stablehlo.reshape %v3104 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3106 = stablehlo.add %v3068, %v3105 : tensor<32x151296xf32>
    %v3107 = stablehlo.reshape %v2793 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3108 = stablehlo.slice %v3107 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3109 = stablehlo.reshape %v3108 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3110 = stablehlo.reshape %v2798 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3111 = stablehlo.slice %v3110 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3112 = stablehlo.reshape %v3111 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3113 = stablehlo.reshape %v2803 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3114 = stablehlo.slice %v3113 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3115 = stablehlo.reshape %v3114 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3116 = stablehlo.reshape %v3112 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3117 = stablehlo.transpose %v3116, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3118 = stablehlo.reshape %v3117 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3119 = stablehlo.reshape %v3109 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3120 = stablehlo.reshape %v3118 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3121 = stablehlo.dot_general %v3119, %v3120, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3122 = stablehlo.reshape %v3121 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3123 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3124 = stablehlo.multiply %v3122, %v3123 : tensor<32x38809xf32>
    %v3125 = stablehlo.reshape %v3124 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3126 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v3127 = stablehlo.reduce(%v3125 init: %v3126) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3128 = stablehlo.broadcast_in_dim %v3127, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3129 = stablehlo.subtract %v3125, %v3128 : tensor<32x197x197xf32>
    %v3130 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3131 = stablehlo.exponential %v3129 : tensor<32x197x197xf32>
    %v3132 = stablehlo.reduce(%v3131 init: %v3130) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3133 = stablehlo.broadcast_in_dim %v3132, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3134 = stablehlo.divide %v3131, %v3133 : tensor<32x197x197xf32>
    %v3135 = stablehlo.reshape %v3134 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3136 = stablehlo.reshape %v3135 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3137 = stablehlo.reshape %v3115 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3138 = stablehlo.dot_general %v3136, %v3137, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3139 = stablehlo.reshape %v3138 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3140 = stablehlo.reshape %v3139 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3141 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3142 = stablehlo.pad %v3140, %v3141, low = [0, 0, 512], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v3143 = stablehlo.reshape %v3142 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3144 = stablehlo.add %v3106, %v3143 : tensor<32x151296xf32>
    %v3145 = stablehlo.reshape %v2793 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3146 = stablehlo.slice %v3145 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3147 = stablehlo.reshape %v3146 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3148 = stablehlo.reshape %v2798 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3149 = stablehlo.slice %v3148 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3150 = stablehlo.reshape %v3149 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3151 = stablehlo.reshape %v2803 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3152 = stablehlo.slice %v3151 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3153 = stablehlo.reshape %v3152 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3154 = stablehlo.reshape %v3150 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3155 = stablehlo.transpose %v3154, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3156 = stablehlo.reshape %v3155 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3157 = stablehlo.reshape %v3147 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3158 = stablehlo.reshape %v3156 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3159 = stablehlo.dot_general %v3157, %v3158, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3160 = stablehlo.reshape %v3159 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3161 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3162 = stablehlo.multiply %v3160, %v3161 : tensor<32x38809xf32>
    %v3163 = stablehlo.reshape %v3162 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3164 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v3165 = stablehlo.reduce(%v3163 init: %v3164) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3166 = stablehlo.broadcast_in_dim %v3165, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3167 = stablehlo.subtract %v3163, %v3166 : tensor<32x197x197xf32>
    %v3168 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3169 = stablehlo.exponential %v3167 : tensor<32x197x197xf32>
    %v3170 = stablehlo.reduce(%v3169 init: %v3168) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3171 = stablehlo.broadcast_in_dim %v3170, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3172 = stablehlo.divide %v3169, %v3171 : tensor<32x197x197xf32>
    %v3173 = stablehlo.reshape %v3172 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3174 = stablehlo.reshape %v3173 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3175 = stablehlo.reshape %v3153 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3176 = stablehlo.dot_general %v3174, %v3175, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3177 = stablehlo.reshape %v3176 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3178 = stablehlo.reshape %v3177 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3179 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3180 = stablehlo.pad %v3178, %v3179, low = [0, 0, 576], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v3181 = stablehlo.reshape %v3180 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3182 = stablehlo.add %v3144, %v3181 : tensor<32x151296xf32>
    %v3183 = stablehlo.reshape %v2793 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3184 = stablehlo.slice %v3183 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3185 = stablehlo.reshape %v3184 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3186 = stablehlo.reshape %v2798 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3187 = stablehlo.slice %v3186 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3188 = stablehlo.reshape %v3187 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3189 = stablehlo.reshape %v2803 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3190 = stablehlo.slice %v3189 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3191 = stablehlo.reshape %v3190 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3192 = stablehlo.reshape %v3188 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3193 = stablehlo.transpose %v3192, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3194 = stablehlo.reshape %v3193 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3195 = stablehlo.reshape %v3185 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3196 = stablehlo.reshape %v3194 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3197 = stablehlo.dot_general %v3195, %v3196, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3198 = stablehlo.reshape %v3197 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3199 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3200 = stablehlo.multiply %v3198, %v3199 : tensor<32x38809xf32>
    %v3201 = stablehlo.reshape %v3200 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3202 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v3203 = stablehlo.reduce(%v3201 init: %v3202) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3204 = stablehlo.broadcast_in_dim %v3203, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3205 = stablehlo.subtract %v3201, %v3204 : tensor<32x197x197xf32>
    %v3206 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3207 = stablehlo.exponential %v3205 : tensor<32x197x197xf32>
    %v3208 = stablehlo.reduce(%v3207 init: %v3206) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3209 = stablehlo.broadcast_in_dim %v3208, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3210 = stablehlo.divide %v3207, %v3209 : tensor<32x197x197xf32>
    %v3211 = stablehlo.reshape %v3210 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3212 = stablehlo.reshape %v3211 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3213 = stablehlo.reshape %v3191 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3214 = stablehlo.dot_general %v3212, %v3213, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3215 = stablehlo.reshape %v3214 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3216 = stablehlo.reshape %v3215 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3217 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3218 = stablehlo.pad %v3216, %v3217, low = [0, 0, 640], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v3219 = stablehlo.reshape %v3218 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3220 = stablehlo.add %v3182, %v3219 : tensor<32x151296xf32>
    %v3221 = stablehlo.reshape %v2793 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3222 = stablehlo.slice %v3221 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3223 = stablehlo.reshape %v3222 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3224 = stablehlo.reshape %v2798 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3225 = stablehlo.slice %v3224 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3226 = stablehlo.reshape %v3225 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3227 = stablehlo.reshape %v2803 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3228 = stablehlo.slice %v3227 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3229 = stablehlo.reshape %v3228 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3230 = stablehlo.reshape %v3226 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3231 = stablehlo.transpose %v3230, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3232 = stablehlo.reshape %v3231 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3233 = stablehlo.reshape %v3223 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3234 = stablehlo.reshape %v3232 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3235 = stablehlo.dot_general %v3233, %v3234, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3236 = stablehlo.reshape %v3235 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3237 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3238 = stablehlo.multiply %v3236, %v3237 : tensor<32x38809xf32>
    %v3239 = stablehlo.reshape %v3238 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3240 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v3241 = stablehlo.reduce(%v3239 init: %v3240) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3242 = stablehlo.broadcast_in_dim %v3241, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3243 = stablehlo.subtract %v3239, %v3242 : tensor<32x197x197xf32>
    %v3244 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3245 = stablehlo.exponential %v3243 : tensor<32x197x197xf32>
    %v3246 = stablehlo.reduce(%v3245 init: %v3244) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3247 = stablehlo.broadcast_in_dim %v3246, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3248 = stablehlo.divide %v3245, %v3247 : tensor<32x197x197xf32>
    %v3249 = stablehlo.reshape %v3248 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3250 = stablehlo.reshape %v3249 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3251 = stablehlo.reshape %v3229 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3252 = stablehlo.dot_general %v3250, %v3251, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3253 = stablehlo.reshape %v3252 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3254 = stablehlo.reshape %v3253 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3255 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3256 = stablehlo.pad %v3254, %v3255, low = [0, 0, 704], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v3257 = stablehlo.reshape %v3256 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3258 = stablehlo.add %v3220, %v3257 : tensor<32x151296xf32>
    %v3259 = stablehlo.reshape %v3258 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3260 = stablehlo.dot_general %v3259, %b5_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v3261 = stablehlo.broadcast_in_dim %b5_bo, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v3262 = stablehlo.add %v3260, %v3261 : tensor<32x197x768xf32>
    %v3263 = stablehlo.reshape %v3262 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3264 = stablehlo.add %v2760, %v3263 : tensor<32x151296xf32>
    %v3265 = stablehlo.reshape %v3264 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3266 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3267 = stablehlo.constant dense<768.0> : tensor<32x197x768xf32>
    %v3268 = stablehlo.constant dense<1.0e-6> : tensor<32x197x768xf32>
    %v3269 = stablehlo.reduce(%v3265 init: %v3266) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3270 = stablehlo.broadcast_in_dim %v3269, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v3271 = stablehlo.divide %v3270, %v3267 : tensor<32x197x768xf32>
    %v3272 = stablehlo.subtract %v3265, %v3271 : tensor<32x197x768xf32>
    %v3273 = stablehlo.multiply %v3272, %v3272 : tensor<32x197x768xf32>
    %v3274 = stablehlo.reduce(%v3273 init: %v3266) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3275 = stablehlo.broadcast_in_dim %v3274, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v3276 = stablehlo.divide %v3275, %v3267 : tensor<32x197x768xf32>
    %v3277 = stablehlo.add %v3276, %v3268 : tensor<32x197x768xf32>
    %v3278 = stablehlo.rsqrt %v3277 : tensor<32x197x768xf32>
    %v3279 = stablehlo.multiply %v3272, %v3278 : tensor<32x197x768xf32>
    %v3280 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v3281 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v3282 = stablehlo.multiply %v3279, %v3280 : tensor<32x197x768xf32>
    %v3283 = stablehlo.add %v3282, %v3281 : tensor<32x197x768xf32>
    %v3284 = stablehlo.reshape %v3283 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3285 = stablehlo.reshape %v3284 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3286 = stablehlo.broadcast_in_dim %b5_g2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v3287 = stablehlo.multiply %v3285, %v3286 : tensor<32x197x768xf32>
    %v3288 = stablehlo.reshape %v3287 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3289 = stablehlo.reshape %v3288 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3290 = stablehlo.broadcast_in_dim %b5_bt2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v3291 = stablehlo.add %v3289, %v3290 : tensor<32x197x768xf32>
    %v3292 = stablehlo.reshape %v3291 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3293 = stablehlo.reshape %v3292 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3294 = stablehlo.dot_general %v3293, %b5_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x3072xf32>) -> tensor<32x197x3072xf32>
    %v3295 = stablehlo.broadcast_in_dim %b5_bfc1, dims = [2] : (tensor<3072xf32>) -> tensor<32x197x3072xf32>
    %v3296 = stablehlo.add %v3294, %v3295 : tensor<32x197x3072xf32>
    %v3297 = stablehlo.reshape %v3296 : (tensor<32x197x3072xf32>) -> tensor<32x605184xf32>
    %v3298 = stablehlo.constant dense<0.5> : tensor<32x605184xf32>
    %v3299 = stablehlo.multiply %v3298, %v3297 : tensor<32x605184xf32>
    %v3300 = stablehlo.negate %v3297 : tensor<32x605184xf32>
    %v3301 = stablehlo.constant dense<0.7071067811865476> : tensor<32x605184xf32>
    %v3302 = stablehlo.multiply %v3300, %v3301 : tensor<32x605184xf32>
    %v3303 = chlo.erfc %v3302 : tensor<32x605184xf32> -> tensor<32x605184xf32>
    %v3304 = stablehlo.multiply %v3299, %v3303 : tensor<32x605184xf32>
    %v3305 = stablehlo.reshape %v3304 : (tensor<32x605184xf32>) -> tensor<32x197x3072xf32>
    %v3306 = stablehlo.dot_general %v3305, %b5_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x3072xf32>, tensor<3072x768xf32>) -> tensor<32x197x768xf32>
    %v3307 = stablehlo.broadcast_in_dim %b5_bfc2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v3308 = stablehlo.add %v3306, %v3307 : tensor<32x197x768xf32>
    %v3309 = stablehlo.reshape %v3308 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3310 = stablehlo.add %v3264, %v3309 : tensor<32x151296xf32>
    %v3311 = stablehlo.reshape %v3310 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3312 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3313 = stablehlo.constant dense<768.0> : tensor<32x197x768xf32>
    %v3314 = stablehlo.constant dense<1.0e-6> : tensor<32x197x768xf32>
    %v3315 = stablehlo.reduce(%v3311 init: %v3312) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3316 = stablehlo.broadcast_in_dim %v3315, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v3317 = stablehlo.divide %v3316, %v3313 : tensor<32x197x768xf32>
    %v3318 = stablehlo.subtract %v3311, %v3317 : tensor<32x197x768xf32>
    %v3319 = stablehlo.multiply %v3318, %v3318 : tensor<32x197x768xf32>
    %v3320 = stablehlo.reduce(%v3319 init: %v3312) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3321 = stablehlo.broadcast_in_dim %v3320, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v3322 = stablehlo.divide %v3321, %v3313 : tensor<32x197x768xf32>
    %v3323 = stablehlo.add %v3322, %v3314 : tensor<32x197x768xf32>
    %v3324 = stablehlo.rsqrt %v3323 : tensor<32x197x768xf32>
    %v3325 = stablehlo.multiply %v3318, %v3324 : tensor<32x197x768xf32>
    %v3326 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v3327 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v3328 = stablehlo.multiply %v3325, %v3326 : tensor<32x197x768xf32>
    %v3329 = stablehlo.add %v3328, %v3327 : tensor<32x197x768xf32>
    %v3330 = stablehlo.reshape %v3329 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3331 = stablehlo.reshape %v3330 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3332 = stablehlo.broadcast_in_dim %b6_g1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v3333 = stablehlo.multiply %v3331, %v3332 : tensor<32x197x768xf32>
    %v3334 = stablehlo.reshape %v3333 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3335 = stablehlo.reshape %v3334 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3336 = stablehlo.broadcast_in_dim %b6_bt1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v3337 = stablehlo.add %v3335, %v3336 : tensor<32x197x768xf32>
    %v3338 = stablehlo.reshape %v3337 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3339 = stablehlo.reshape %v3338 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3340 = stablehlo.dot_general %v3339, %b6_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v3341 = stablehlo.broadcast_in_dim %b6_bq, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v3342 = stablehlo.add %v3340, %v3341 : tensor<32x197x768xf32>
    %v3343 = stablehlo.reshape %v3342 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3344 = stablehlo.reshape %v3338 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3345 = stablehlo.dot_general %v3344, %b6_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v3346 = stablehlo.broadcast_in_dim %b6_bk, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v3347 = stablehlo.add %v3345, %v3346 : tensor<32x197x768xf32>
    %v3348 = stablehlo.reshape %v3347 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3349 = stablehlo.reshape %v3338 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3350 = stablehlo.dot_general %v3349, %b6_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v3351 = stablehlo.broadcast_in_dim %b6_bv, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v3352 = stablehlo.add %v3350, %v3351 : tensor<32x197x768xf32>
    %v3353 = stablehlo.reshape %v3352 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3354 = stablehlo.reshape %v3343 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3355 = stablehlo.slice %v3354 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3356 = stablehlo.reshape %v3355 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3357 = stablehlo.reshape %v3348 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3358 = stablehlo.slice %v3357 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3359 = stablehlo.reshape %v3358 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3360 = stablehlo.reshape %v3353 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3361 = stablehlo.slice %v3360 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3362 = stablehlo.reshape %v3361 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3363 = stablehlo.reshape %v3359 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3364 = stablehlo.transpose %v3363, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3365 = stablehlo.reshape %v3364 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3366 = stablehlo.reshape %v3356 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3367 = stablehlo.reshape %v3365 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3368 = stablehlo.dot_general %v3366, %v3367, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3369 = stablehlo.reshape %v3368 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3370 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3371 = stablehlo.multiply %v3369, %v3370 : tensor<32x38809xf32>
    %v3372 = stablehlo.reshape %v3371 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3373 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v3374 = stablehlo.reduce(%v3372 init: %v3373) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3375 = stablehlo.broadcast_in_dim %v3374, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3376 = stablehlo.subtract %v3372, %v3375 : tensor<32x197x197xf32>
    %v3377 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3378 = stablehlo.exponential %v3376 : tensor<32x197x197xf32>
    %v3379 = stablehlo.reduce(%v3378 init: %v3377) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3380 = stablehlo.broadcast_in_dim %v3379, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3381 = stablehlo.divide %v3378, %v3380 : tensor<32x197x197xf32>
    %v3382 = stablehlo.reshape %v3381 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3383 = stablehlo.reshape %v3382 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3384 = stablehlo.reshape %v3362 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3385 = stablehlo.dot_general %v3383, %v3384, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3386 = stablehlo.reshape %v3385 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3387 = stablehlo.reshape %v3386 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3388 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3389 = stablehlo.pad %v3387, %v3388, low = [0, 0, 0], high = [0, 0, 704], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v3390 = stablehlo.reshape %v3389 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3391 = stablehlo.reshape %v3343 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3392 = stablehlo.slice %v3391 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3393 = stablehlo.reshape %v3392 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3394 = stablehlo.reshape %v3348 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3395 = stablehlo.slice %v3394 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3396 = stablehlo.reshape %v3395 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3397 = stablehlo.reshape %v3353 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3398 = stablehlo.slice %v3397 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3399 = stablehlo.reshape %v3398 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3400 = stablehlo.reshape %v3396 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3401 = stablehlo.transpose %v3400, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3402 = stablehlo.reshape %v3401 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3403 = stablehlo.reshape %v3393 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3404 = stablehlo.reshape %v3402 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3405 = stablehlo.dot_general %v3403, %v3404, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3406 = stablehlo.reshape %v3405 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3407 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3408 = stablehlo.multiply %v3406, %v3407 : tensor<32x38809xf32>
    %v3409 = stablehlo.reshape %v3408 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3410 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v3411 = stablehlo.reduce(%v3409 init: %v3410) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3412 = stablehlo.broadcast_in_dim %v3411, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3413 = stablehlo.subtract %v3409, %v3412 : tensor<32x197x197xf32>
    %v3414 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3415 = stablehlo.exponential %v3413 : tensor<32x197x197xf32>
    %v3416 = stablehlo.reduce(%v3415 init: %v3414) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3417 = stablehlo.broadcast_in_dim %v3416, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3418 = stablehlo.divide %v3415, %v3417 : tensor<32x197x197xf32>
    %v3419 = stablehlo.reshape %v3418 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3420 = stablehlo.reshape %v3419 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3421 = stablehlo.reshape %v3399 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3422 = stablehlo.dot_general %v3420, %v3421, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3423 = stablehlo.reshape %v3422 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3424 = stablehlo.reshape %v3423 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3425 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3426 = stablehlo.pad %v3424, %v3425, low = [0, 0, 64], high = [0, 0, 640], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v3427 = stablehlo.reshape %v3426 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3428 = stablehlo.add %v3390, %v3427 : tensor<32x151296xf32>
    %v3429 = stablehlo.reshape %v3343 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3430 = stablehlo.slice %v3429 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3431 = stablehlo.reshape %v3430 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3432 = stablehlo.reshape %v3348 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3433 = stablehlo.slice %v3432 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3434 = stablehlo.reshape %v3433 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3435 = stablehlo.reshape %v3353 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3436 = stablehlo.slice %v3435 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3437 = stablehlo.reshape %v3436 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3438 = stablehlo.reshape %v3434 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3439 = stablehlo.transpose %v3438, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3440 = stablehlo.reshape %v3439 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3441 = stablehlo.reshape %v3431 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3442 = stablehlo.reshape %v3440 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3443 = stablehlo.dot_general %v3441, %v3442, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3444 = stablehlo.reshape %v3443 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3445 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3446 = stablehlo.multiply %v3444, %v3445 : tensor<32x38809xf32>
    %v3447 = stablehlo.reshape %v3446 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3448 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v3449 = stablehlo.reduce(%v3447 init: %v3448) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3450 = stablehlo.broadcast_in_dim %v3449, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3451 = stablehlo.subtract %v3447, %v3450 : tensor<32x197x197xf32>
    %v3452 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3453 = stablehlo.exponential %v3451 : tensor<32x197x197xf32>
    %v3454 = stablehlo.reduce(%v3453 init: %v3452) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3455 = stablehlo.broadcast_in_dim %v3454, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3456 = stablehlo.divide %v3453, %v3455 : tensor<32x197x197xf32>
    %v3457 = stablehlo.reshape %v3456 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3458 = stablehlo.reshape %v3457 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3459 = stablehlo.reshape %v3437 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3460 = stablehlo.dot_general %v3458, %v3459, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3461 = stablehlo.reshape %v3460 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3462 = stablehlo.reshape %v3461 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3463 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3464 = stablehlo.pad %v3462, %v3463, low = [0, 0, 128], high = [0, 0, 576], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v3465 = stablehlo.reshape %v3464 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3466 = stablehlo.add %v3428, %v3465 : tensor<32x151296xf32>
    %v3467 = stablehlo.reshape %v3343 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3468 = stablehlo.slice %v3467 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3469 = stablehlo.reshape %v3468 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3470 = stablehlo.reshape %v3348 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3471 = stablehlo.slice %v3470 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3472 = stablehlo.reshape %v3471 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3473 = stablehlo.reshape %v3353 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3474 = stablehlo.slice %v3473 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3475 = stablehlo.reshape %v3474 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3476 = stablehlo.reshape %v3472 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3477 = stablehlo.transpose %v3476, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3478 = stablehlo.reshape %v3477 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3479 = stablehlo.reshape %v3469 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3480 = stablehlo.reshape %v3478 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3481 = stablehlo.dot_general %v3479, %v3480, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3482 = stablehlo.reshape %v3481 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3483 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3484 = stablehlo.multiply %v3482, %v3483 : tensor<32x38809xf32>
    %v3485 = stablehlo.reshape %v3484 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3486 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v3487 = stablehlo.reduce(%v3485 init: %v3486) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3488 = stablehlo.broadcast_in_dim %v3487, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3489 = stablehlo.subtract %v3485, %v3488 : tensor<32x197x197xf32>
    %v3490 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3491 = stablehlo.exponential %v3489 : tensor<32x197x197xf32>
    %v3492 = stablehlo.reduce(%v3491 init: %v3490) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3493 = stablehlo.broadcast_in_dim %v3492, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3494 = stablehlo.divide %v3491, %v3493 : tensor<32x197x197xf32>
    %v3495 = stablehlo.reshape %v3494 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3496 = stablehlo.reshape %v3495 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3497 = stablehlo.reshape %v3475 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3498 = stablehlo.dot_general %v3496, %v3497, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3499 = stablehlo.reshape %v3498 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3500 = stablehlo.reshape %v3499 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3501 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3502 = stablehlo.pad %v3500, %v3501, low = [0, 0, 192], high = [0, 0, 512], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v3503 = stablehlo.reshape %v3502 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3504 = stablehlo.add %v3466, %v3503 : tensor<32x151296xf32>
    %v3505 = stablehlo.reshape %v3343 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3506 = stablehlo.slice %v3505 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3507 = stablehlo.reshape %v3506 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3508 = stablehlo.reshape %v3348 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3509 = stablehlo.slice %v3508 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3510 = stablehlo.reshape %v3509 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3511 = stablehlo.reshape %v3353 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3512 = stablehlo.slice %v3511 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3513 = stablehlo.reshape %v3512 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3514 = stablehlo.reshape %v3510 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3515 = stablehlo.transpose %v3514, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3516 = stablehlo.reshape %v3515 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3517 = stablehlo.reshape %v3507 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3518 = stablehlo.reshape %v3516 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3519 = stablehlo.dot_general %v3517, %v3518, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3520 = stablehlo.reshape %v3519 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3521 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3522 = stablehlo.multiply %v3520, %v3521 : tensor<32x38809xf32>
    %v3523 = stablehlo.reshape %v3522 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3524 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v3525 = stablehlo.reduce(%v3523 init: %v3524) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3526 = stablehlo.broadcast_in_dim %v3525, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3527 = stablehlo.subtract %v3523, %v3526 : tensor<32x197x197xf32>
    %v3528 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3529 = stablehlo.exponential %v3527 : tensor<32x197x197xf32>
    %v3530 = stablehlo.reduce(%v3529 init: %v3528) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3531 = stablehlo.broadcast_in_dim %v3530, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3532 = stablehlo.divide %v3529, %v3531 : tensor<32x197x197xf32>
    %v3533 = stablehlo.reshape %v3532 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3534 = stablehlo.reshape %v3533 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3535 = stablehlo.reshape %v3513 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3536 = stablehlo.dot_general %v3534, %v3535, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3537 = stablehlo.reshape %v3536 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3538 = stablehlo.reshape %v3537 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3539 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3540 = stablehlo.pad %v3538, %v3539, low = [0, 0, 256], high = [0, 0, 448], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v3541 = stablehlo.reshape %v3540 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3542 = stablehlo.add %v3504, %v3541 : tensor<32x151296xf32>
    %v3543 = stablehlo.reshape %v3343 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3544 = stablehlo.slice %v3543 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3545 = stablehlo.reshape %v3544 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3546 = stablehlo.reshape %v3348 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3547 = stablehlo.slice %v3546 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3548 = stablehlo.reshape %v3547 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3549 = stablehlo.reshape %v3353 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3550 = stablehlo.slice %v3549 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3551 = stablehlo.reshape %v3550 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3552 = stablehlo.reshape %v3548 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3553 = stablehlo.transpose %v3552, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3554 = stablehlo.reshape %v3553 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3555 = stablehlo.reshape %v3545 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3556 = stablehlo.reshape %v3554 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3557 = stablehlo.dot_general %v3555, %v3556, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3558 = stablehlo.reshape %v3557 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3559 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3560 = stablehlo.multiply %v3558, %v3559 : tensor<32x38809xf32>
    %v3561 = stablehlo.reshape %v3560 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3562 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v3563 = stablehlo.reduce(%v3561 init: %v3562) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3564 = stablehlo.broadcast_in_dim %v3563, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3565 = stablehlo.subtract %v3561, %v3564 : tensor<32x197x197xf32>
    %v3566 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3567 = stablehlo.exponential %v3565 : tensor<32x197x197xf32>
    %v3568 = stablehlo.reduce(%v3567 init: %v3566) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3569 = stablehlo.broadcast_in_dim %v3568, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3570 = stablehlo.divide %v3567, %v3569 : tensor<32x197x197xf32>
    %v3571 = stablehlo.reshape %v3570 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3572 = stablehlo.reshape %v3571 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3573 = stablehlo.reshape %v3551 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3574 = stablehlo.dot_general %v3572, %v3573, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3575 = stablehlo.reshape %v3574 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3576 = stablehlo.reshape %v3575 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3577 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3578 = stablehlo.pad %v3576, %v3577, low = [0, 0, 320], high = [0, 0, 384], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v3579 = stablehlo.reshape %v3578 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3580 = stablehlo.add %v3542, %v3579 : tensor<32x151296xf32>
    %v3581 = stablehlo.reshape %v3343 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3582 = stablehlo.slice %v3581 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3583 = stablehlo.reshape %v3582 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3584 = stablehlo.reshape %v3348 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3585 = stablehlo.slice %v3584 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3586 = stablehlo.reshape %v3585 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3587 = stablehlo.reshape %v3353 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3588 = stablehlo.slice %v3587 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3589 = stablehlo.reshape %v3588 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3590 = stablehlo.reshape %v3586 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3591 = stablehlo.transpose %v3590, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3592 = stablehlo.reshape %v3591 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3593 = stablehlo.reshape %v3583 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3594 = stablehlo.reshape %v3592 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3595 = stablehlo.dot_general %v3593, %v3594, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3596 = stablehlo.reshape %v3595 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3597 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3598 = stablehlo.multiply %v3596, %v3597 : tensor<32x38809xf32>
    %v3599 = stablehlo.reshape %v3598 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3600 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v3601 = stablehlo.reduce(%v3599 init: %v3600) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3602 = stablehlo.broadcast_in_dim %v3601, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3603 = stablehlo.subtract %v3599, %v3602 : tensor<32x197x197xf32>
    %v3604 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3605 = stablehlo.exponential %v3603 : tensor<32x197x197xf32>
    %v3606 = stablehlo.reduce(%v3605 init: %v3604) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3607 = stablehlo.broadcast_in_dim %v3606, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3608 = stablehlo.divide %v3605, %v3607 : tensor<32x197x197xf32>
    %v3609 = stablehlo.reshape %v3608 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3610 = stablehlo.reshape %v3609 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3611 = stablehlo.reshape %v3589 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3612 = stablehlo.dot_general %v3610, %v3611, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3613 = stablehlo.reshape %v3612 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3614 = stablehlo.reshape %v3613 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3615 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3616 = stablehlo.pad %v3614, %v3615, low = [0, 0, 384], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v3617 = stablehlo.reshape %v3616 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3618 = stablehlo.add %v3580, %v3617 : tensor<32x151296xf32>
    %v3619 = stablehlo.reshape %v3343 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3620 = stablehlo.slice %v3619 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3621 = stablehlo.reshape %v3620 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3622 = stablehlo.reshape %v3348 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3623 = stablehlo.slice %v3622 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3624 = stablehlo.reshape %v3623 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3625 = stablehlo.reshape %v3353 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3626 = stablehlo.slice %v3625 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3627 = stablehlo.reshape %v3626 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3628 = stablehlo.reshape %v3624 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3629 = stablehlo.transpose %v3628, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3630 = stablehlo.reshape %v3629 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3631 = stablehlo.reshape %v3621 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3632 = stablehlo.reshape %v3630 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3633 = stablehlo.dot_general %v3631, %v3632, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3634 = stablehlo.reshape %v3633 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3635 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3636 = stablehlo.multiply %v3634, %v3635 : tensor<32x38809xf32>
    %v3637 = stablehlo.reshape %v3636 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3638 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v3639 = stablehlo.reduce(%v3637 init: %v3638) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3640 = stablehlo.broadcast_in_dim %v3639, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3641 = stablehlo.subtract %v3637, %v3640 : tensor<32x197x197xf32>
    %v3642 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3643 = stablehlo.exponential %v3641 : tensor<32x197x197xf32>
    %v3644 = stablehlo.reduce(%v3643 init: %v3642) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3645 = stablehlo.broadcast_in_dim %v3644, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3646 = stablehlo.divide %v3643, %v3645 : tensor<32x197x197xf32>
    %v3647 = stablehlo.reshape %v3646 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3648 = stablehlo.reshape %v3647 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3649 = stablehlo.reshape %v3627 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3650 = stablehlo.dot_general %v3648, %v3649, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3651 = stablehlo.reshape %v3650 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3652 = stablehlo.reshape %v3651 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3653 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3654 = stablehlo.pad %v3652, %v3653, low = [0, 0, 448], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v3655 = stablehlo.reshape %v3654 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3656 = stablehlo.add %v3618, %v3655 : tensor<32x151296xf32>
    %v3657 = stablehlo.reshape %v3343 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3658 = stablehlo.slice %v3657 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3659 = stablehlo.reshape %v3658 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3660 = stablehlo.reshape %v3348 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3661 = stablehlo.slice %v3660 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3662 = stablehlo.reshape %v3661 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3663 = stablehlo.reshape %v3353 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3664 = stablehlo.slice %v3663 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3665 = stablehlo.reshape %v3664 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3666 = stablehlo.reshape %v3662 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3667 = stablehlo.transpose %v3666, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3668 = stablehlo.reshape %v3667 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3669 = stablehlo.reshape %v3659 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3670 = stablehlo.reshape %v3668 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3671 = stablehlo.dot_general %v3669, %v3670, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3672 = stablehlo.reshape %v3671 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3673 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3674 = stablehlo.multiply %v3672, %v3673 : tensor<32x38809xf32>
    %v3675 = stablehlo.reshape %v3674 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3676 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v3677 = stablehlo.reduce(%v3675 init: %v3676) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3678 = stablehlo.broadcast_in_dim %v3677, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3679 = stablehlo.subtract %v3675, %v3678 : tensor<32x197x197xf32>
    %v3680 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3681 = stablehlo.exponential %v3679 : tensor<32x197x197xf32>
    %v3682 = stablehlo.reduce(%v3681 init: %v3680) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3683 = stablehlo.broadcast_in_dim %v3682, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3684 = stablehlo.divide %v3681, %v3683 : tensor<32x197x197xf32>
    %v3685 = stablehlo.reshape %v3684 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3686 = stablehlo.reshape %v3685 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3687 = stablehlo.reshape %v3665 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3688 = stablehlo.dot_general %v3686, %v3687, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3689 = stablehlo.reshape %v3688 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3690 = stablehlo.reshape %v3689 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3691 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3692 = stablehlo.pad %v3690, %v3691, low = [0, 0, 512], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v3693 = stablehlo.reshape %v3692 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3694 = stablehlo.add %v3656, %v3693 : tensor<32x151296xf32>
    %v3695 = stablehlo.reshape %v3343 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3696 = stablehlo.slice %v3695 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3697 = stablehlo.reshape %v3696 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3698 = stablehlo.reshape %v3348 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3699 = stablehlo.slice %v3698 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3700 = stablehlo.reshape %v3699 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3701 = stablehlo.reshape %v3353 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3702 = stablehlo.slice %v3701 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3703 = stablehlo.reshape %v3702 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3704 = stablehlo.reshape %v3700 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3705 = stablehlo.transpose %v3704, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3706 = stablehlo.reshape %v3705 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3707 = stablehlo.reshape %v3697 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3708 = stablehlo.reshape %v3706 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3709 = stablehlo.dot_general %v3707, %v3708, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3710 = stablehlo.reshape %v3709 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3711 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3712 = stablehlo.multiply %v3710, %v3711 : tensor<32x38809xf32>
    %v3713 = stablehlo.reshape %v3712 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3714 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v3715 = stablehlo.reduce(%v3713 init: %v3714) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3716 = stablehlo.broadcast_in_dim %v3715, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3717 = stablehlo.subtract %v3713, %v3716 : tensor<32x197x197xf32>
    %v3718 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3719 = stablehlo.exponential %v3717 : tensor<32x197x197xf32>
    %v3720 = stablehlo.reduce(%v3719 init: %v3718) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3721 = stablehlo.broadcast_in_dim %v3720, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3722 = stablehlo.divide %v3719, %v3721 : tensor<32x197x197xf32>
    %v3723 = stablehlo.reshape %v3722 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3724 = stablehlo.reshape %v3723 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3725 = stablehlo.reshape %v3703 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3726 = stablehlo.dot_general %v3724, %v3725, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3727 = stablehlo.reshape %v3726 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3728 = stablehlo.reshape %v3727 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3729 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3730 = stablehlo.pad %v3728, %v3729, low = [0, 0, 576], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v3731 = stablehlo.reshape %v3730 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3732 = stablehlo.add %v3694, %v3731 : tensor<32x151296xf32>
    %v3733 = stablehlo.reshape %v3343 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3734 = stablehlo.slice %v3733 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3735 = stablehlo.reshape %v3734 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3736 = stablehlo.reshape %v3348 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3737 = stablehlo.slice %v3736 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3738 = stablehlo.reshape %v3737 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3739 = stablehlo.reshape %v3353 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3740 = stablehlo.slice %v3739 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3741 = stablehlo.reshape %v3740 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3742 = stablehlo.reshape %v3738 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3743 = stablehlo.transpose %v3742, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3744 = stablehlo.reshape %v3743 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3745 = stablehlo.reshape %v3735 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3746 = stablehlo.reshape %v3744 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3747 = stablehlo.dot_general %v3745, %v3746, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3748 = stablehlo.reshape %v3747 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3749 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3750 = stablehlo.multiply %v3748, %v3749 : tensor<32x38809xf32>
    %v3751 = stablehlo.reshape %v3750 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3752 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v3753 = stablehlo.reduce(%v3751 init: %v3752) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3754 = stablehlo.broadcast_in_dim %v3753, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3755 = stablehlo.subtract %v3751, %v3754 : tensor<32x197x197xf32>
    %v3756 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3757 = stablehlo.exponential %v3755 : tensor<32x197x197xf32>
    %v3758 = stablehlo.reduce(%v3757 init: %v3756) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3759 = stablehlo.broadcast_in_dim %v3758, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3760 = stablehlo.divide %v3757, %v3759 : tensor<32x197x197xf32>
    %v3761 = stablehlo.reshape %v3760 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3762 = stablehlo.reshape %v3761 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3763 = stablehlo.reshape %v3741 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3764 = stablehlo.dot_general %v3762, %v3763, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3765 = stablehlo.reshape %v3764 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3766 = stablehlo.reshape %v3765 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3767 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3768 = stablehlo.pad %v3766, %v3767, low = [0, 0, 640], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v3769 = stablehlo.reshape %v3768 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3770 = stablehlo.add %v3732, %v3769 : tensor<32x151296xf32>
    %v3771 = stablehlo.reshape %v3343 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3772 = stablehlo.slice %v3771 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3773 = stablehlo.reshape %v3772 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3774 = stablehlo.reshape %v3348 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3775 = stablehlo.slice %v3774 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3776 = stablehlo.reshape %v3775 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3777 = stablehlo.reshape %v3353 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3778 = stablehlo.slice %v3777 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
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
    %v3790 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v3791 = stablehlo.reduce(%v3789 init: %v3790) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3792 = stablehlo.broadcast_in_dim %v3791, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3793 = stablehlo.subtract %v3789, %v3792 : tensor<32x197x197xf32>
    %v3794 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3795 = stablehlo.exponential %v3793 : tensor<32x197x197xf32>
    %v3796 = stablehlo.reduce(%v3795 init: %v3794) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3797 = stablehlo.broadcast_in_dim %v3796, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3798 = stablehlo.divide %v3795, %v3797 : tensor<32x197x197xf32>
    %v3799 = stablehlo.reshape %v3798 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3800 = stablehlo.reshape %v3799 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3801 = stablehlo.reshape %v3779 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3802 = stablehlo.dot_general %v3800, %v3801, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3803 = stablehlo.reshape %v3802 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3804 = stablehlo.reshape %v3803 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3805 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3806 = stablehlo.pad %v3804, %v3805, low = [0, 0, 704], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v3807 = stablehlo.reshape %v3806 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3808 = stablehlo.add %v3770, %v3807 : tensor<32x151296xf32>
    %v3809 = stablehlo.reshape %v3808 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3810 = stablehlo.dot_general %v3809, %b6_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v3811 = stablehlo.broadcast_in_dim %b6_bo, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v3812 = stablehlo.add %v3810, %v3811 : tensor<32x197x768xf32>
    %v3813 = stablehlo.reshape %v3812 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3814 = stablehlo.add %v3310, %v3813 : tensor<32x151296xf32>
    %v3815 = stablehlo.reshape %v3814 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3816 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3817 = stablehlo.constant dense<768.0> : tensor<32x197x768xf32>
    %v3818 = stablehlo.constant dense<1.0e-6> : tensor<32x197x768xf32>
    %v3819 = stablehlo.reduce(%v3815 init: %v3816) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3820 = stablehlo.broadcast_in_dim %v3819, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v3821 = stablehlo.divide %v3820, %v3817 : tensor<32x197x768xf32>
    %v3822 = stablehlo.subtract %v3815, %v3821 : tensor<32x197x768xf32>
    %v3823 = stablehlo.multiply %v3822, %v3822 : tensor<32x197x768xf32>
    %v3824 = stablehlo.reduce(%v3823 init: %v3816) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3825 = stablehlo.broadcast_in_dim %v3824, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v3826 = stablehlo.divide %v3825, %v3817 : tensor<32x197x768xf32>
    %v3827 = stablehlo.add %v3826, %v3818 : tensor<32x197x768xf32>
    %v3828 = stablehlo.rsqrt %v3827 : tensor<32x197x768xf32>
    %v3829 = stablehlo.multiply %v3822, %v3828 : tensor<32x197x768xf32>
    %v3830 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v3831 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v3832 = stablehlo.multiply %v3829, %v3830 : tensor<32x197x768xf32>
    %v3833 = stablehlo.add %v3832, %v3831 : tensor<32x197x768xf32>
    %v3834 = stablehlo.reshape %v3833 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3835 = stablehlo.reshape %v3834 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3836 = stablehlo.broadcast_in_dim %b6_g2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v3837 = stablehlo.multiply %v3835, %v3836 : tensor<32x197x768xf32>
    %v3838 = stablehlo.reshape %v3837 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3839 = stablehlo.reshape %v3838 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3840 = stablehlo.broadcast_in_dim %b6_bt2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v3841 = stablehlo.add %v3839, %v3840 : tensor<32x197x768xf32>
    %v3842 = stablehlo.reshape %v3841 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3843 = stablehlo.reshape %v3842 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3844 = stablehlo.dot_general %v3843, %b6_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x3072xf32>) -> tensor<32x197x3072xf32>
    %v3845 = stablehlo.broadcast_in_dim %b6_bfc1, dims = [2] : (tensor<3072xf32>) -> tensor<32x197x3072xf32>
    %v3846 = stablehlo.add %v3844, %v3845 : tensor<32x197x3072xf32>
    %v3847 = stablehlo.reshape %v3846 : (tensor<32x197x3072xf32>) -> tensor<32x605184xf32>
    %v3848 = stablehlo.constant dense<0.5> : tensor<32x605184xf32>
    %v3849 = stablehlo.multiply %v3848, %v3847 : tensor<32x605184xf32>
    %v3850 = stablehlo.negate %v3847 : tensor<32x605184xf32>
    %v3851 = stablehlo.constant dense<0.7071067811865476> : tensor<32x605184xf32>
    %v3852 = stablehlo.multiply %v3850, %v3851 : tensor<32x605184xf32>
    %v3853 = chlo.erfc %v3852 : tensor<32x605184xf32> -> tensor<32x605184xf32>
    %v3854 = stablehlo.multiply %v3849, %v3853 : tensor<32x605184xf32>
    %v3855 = stablehlo.reshape %v3854 : (tensor<32x605184xf32>) -> tensor<32x197x3072xf32>
    %v3856 = stablehlo.dot_general %v3855, %b6_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x3072xf32>, tensor<3072x768xf32>) -> tensor<32x197x768xf32>
    %v3857 = stablehlo.broadcast_in_dim %b6_bfc2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v3858 = stablehlo.add %v3856, %v3857 : tensor<32x197x768xf32>
    %v3859 = stablehlo.reshape %v3858 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3860 = stablehlo.add %v3814, %v3859 : tensor<32x151296xf32>
    %v3861 = stablehlo.reshape %v3860 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3862 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3863 = stablehlo.constant dense<768.0> : tensor<32x197x768xf32>
    %v3864 = stablehlo.constant dense<1.0e-6> : tensor<32x197x768xf32>
    %v3865 = stablehlo.reduce(%v3861 init: %v3862) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3866 = stablehlo.broadcast_in_dim %v3865, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v3867 = stablehlo.divide %v3866, %v3863 : tensor<32x197x768xf32>
    %v3868 = stablehlo.subtract %v3861, %v3867 : tensor<32x197x768xf32>
    %v3869 = stablehlo.multiply %v3868, %v3868 : tensor<32x197x768xf32>
    %v3870 = stablehlo.reduce(%v3869 init: %v3862) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3871 = stablehlo.broadcast_in_dim %v3870, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v3872 = stablehlo.divide %v3871, %v3863 : tensor<32x197x768xf32>
    %v3873 = stablehlo.add %v3872, %v3864 : tensor<32x197x768xf32>
    %v3874 = stablehlo.rsqrt %v3873 : tensor<32x197x768xf32>
    %v3875 = stablehlo.multiply %v3868, %v3874 : tensor<32x197x768xf32>
    %v3876 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v3877 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v3878 = stablehlo.multiply %v3875, %v3876 : tensor<32x197x768xf32>
    %v3879 = stablehlo.add %v3878, %v3877 : tensor<32x197x768xf32>
    %v3880 = stablehlo.reshape %v3879 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3881 = stablehlo.reshape %v3880 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3882 = stablehlo.broadcast_in_dim %b7_g1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v3883 = stablehlo.multiply %v3881, %v3882 : tensor<32x197x768xf32>
    %v3884 = stablehlo.reshape %v3883 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3885 = stablehlo.reshape %v3884 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3886 = stablehlo.broadcast_in_dim %b7_bt1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v3887 = stablehlo.add %v3885, %v3886 : tensor<32x197x768xf32>
    %v3888 = stablehlo.reshape %v3887 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3889 = stablehlo.reshape %v3888 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3890 = stablehlo.dot_general %v3889, %b7_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v3891 = stablehlo.broadcast_in_dim %b7_bq, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v3892 = stablehlo.add %v3890, %v3891 : tensor<32x197x768xf32>
    %v3893 = stablehlo.reshape %v3892 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3894 = stablehlo.reshape %v3888 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3895 = stablehlo.dot_general %v3894, %b7_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v3896 = stablehlo.broadcast_in_dim %b7_bk, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v3897 = stablehlo.add %v3895, %v3896 : tensor<32x197x768xf32>
    %v3898 = stablehlo.reshape %v3897 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3899 = stablehlo.reshape %v3888 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3900 = stablehlo.dot_general %v3899, %b7_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v3901 = stablehlo.broadcast_in_dim %b7_bv, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v3902 = stablehlo.add %v3900, %v3901 : tensor<32x197x768xf32>
    %v3903 = stablehlo.reshape %v3902 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3904 = stablehlo.reshape %v3893 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3905 = stablehlo.slice %v3904 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3906 = stablehlo.reshape %v3905 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3907 = stablehlo.reshape %v3898 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3908 = stablehlo.slice %v3907 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3909 = stablehlo.reshape %v3908 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3910 = stablehlo.reshape %v3903 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3911 = stablehlo.slice %v3910 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3912 = stablehlo.reshape %v3911 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3913 = stablehlo.reshape %v3909 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3914 = stablehlo.transpose %v3913, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3915 = stablehlo.reshape %v3914 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3916 = stablehlo.reshape %v3906 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3917 = stablehlo.reshape %v3915 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3918 = stablehlo.dot_general %v3916, %v3917, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3919 = stablehlo.reshape %v3918 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3920 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3921 = stablehlo.multiply %v3919, %v3920 : tensor<32x38809xf32>
    %v3922 = stablehlo.reshape %v3921 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3923 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v3924 = stablehlo.reduce(%v3922 init: %v3923) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3925 = stablehlo.broadcast_in_dim %v3924, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3926 = stablehlo.subtract %v3922, %v3925 : tensor<32x197x197xf32>
    %v3927 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3928 = stablehlo.exponential %v3926 : tensor<32x197x197xf32>
    %v3929 = stablehlo.reduce(%v3928 init: %v3927) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3930 = stablehlo.broadcast_in_dim %v3929, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3931 = stablehlo.divide %v3928, %v3930 : tensor<32x197x197xf32>
    %v3932 = stablehlo.reshape %v3931 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3933 = stablehlo.reshape %v3932 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3934 = stablehlo.reshape %v3912 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3935 = stablehlo.dot_general %v3933, %v3934, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3936 = stablehlo.reshape %v3935 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3937 = stablehlo.reshape %v3936 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3938 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3939 = stablehlo.pad %v3937, %v3938, low = [0, 0, 0], high = [0, 0, 704], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v3940 = stablehlo.reshape %v3939 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3941 = stablehlo.reshape %v3893 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3942 = stablehlo.slice %v3941 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3943 = stablehlo.reshape %v3942 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3944 = stablehlo.reshape %v3898 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3945 = stablehlo.slice %v3944 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3946 = stablehlo.reshape %v3945 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3947 = stablehlo.reshape %v3903 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3948 = stablehlo.slice %v3947 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
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
    %v3960 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v3961 = stablehlo.reduce(%v3959 init: %v3960) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3962 = stablehlo.broadcast_in_dim %v3961, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3963 = stablehlo.subtract %v3959, %v3962 : tensor<32x197x197xf32>
    %v3964 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3965 = stablehlo.exponential %v3963 : tensor<32x197x197xf32>
    %v3966 = stablehlo.reduce(%v3965 init: %v3964) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3967 = stablehlo.broadcast_in_dim %v3966, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3968 = stablehlo.divide %v3965, %v3967 : tensor<32x197x197xf32>
    %v3969 = stablehlo.reshape %v3968 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3970 = stablehlo.reshape %v3969 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3971 = stablehlo.reshape %v3949 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3972 = stablehlo.dot_general %v3970, %v3971, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3973 = stablehlo.reshape %v3972 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3974 = stablehlo.reshape %v3973 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3975 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3976 = stablehlo.pad %v3974, %v3975, low = [0, 0, 64], high = [0, 0, 640], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v3977 = stablehlo.reshape %v3976 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3978 = stablehlo.add %v3940, %v3977 : tensor<32x151296xf32>
    %v3979 = stablehlo.reshape %v3893 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3980 = stablehlo.slice %v3979 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3981 = stablehlo.reshape %v3980 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3982 = stablehlo.reshape %v3898 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3983 = stablehlo.slice %v3982 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3984 = stablehlo.reshape %v3983 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3985 = stablehlo.reshape %v3903 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3986 = stablehlo.slice %v3985 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3987 = stablehlo.reshape %v3986 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3988 = stablehlo.reshape %v3984 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3989 = stablehlo.transpose %v3988, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3990 = stablehlo.reshape %v3989 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3991 = stablehlo.reshape %v3981 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3992 = stablehlo.reshape %v3990 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3993 = stablehlo.dot_general %v3991, %v3992, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3994 = stablehlo.reshape %v3993 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3995 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3996 = stablehlo.multiply %v3994, %v3995 : tensor<32x38809xf32>
    %v3997 = stablehlo.reshape %v3996 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3998 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v3999 = stablehlo.reduce(%v3997 init: %v3998) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4000 = stablehlo.broadcast_in_dim %v3999, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4001 = stablehlo.subtract %v3997, %v4000 : tensor<32x197x197xf32>
    %v4002 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4003 = stablehlo.exponential %v4001 : tensor<32x197x197xf32>
    %v4004 = stablehlo.reduce(%v4003 init: %v4002) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4005 = stablehlo.broadcast_in_dim %v4004, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4006 = stablehlo.divide %v4003, %v4005 : tensor<32x197x197xf32>
    %v4007 = stablehlo.reshape %v4006 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4008 = stablehlo.reshape %v4007 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4009 = stablehlo.reshape %v3987 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4010 = stablehlo.dot_general %v4008, %v4009, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4011 = stablehlo.reshape %v4010 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4012 = stablehlo.reshape %v4011 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4013 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4014 = stablehlo.pad %v4012, %v4013, low = [0, 0, 128], high = [0, 0, 576], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v4015 = stablehlo.reshape %v4014 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4016 = stablehlo.add %v3978, %v4015 : tensor<32x151296xf32>
    %v4017 = stablehlo.reshape %v3893 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4018 = stablehlo.slice %v4017 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4019 = stablehlo.reshape %v4018 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4020 = stablehlo.reshape %v3898 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4021 = stablehlo.slice %v4020 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4022 = stablehlo.reshape %v4021 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4023 = stablehlo.reshape %v3903 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4024 = stablehlo.slice %v4023 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4025 = stablehlo.reshape %v4024 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4026 = stablehlo.reshape %v4022 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4027 = stablehlo.transpose %v4026, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v4028 = stablehlo.reshape %v4027 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4029 = stablehlo.reshape %v4019 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4030 = stablehlo.reshape %v4028 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4031 = stablehlo.dot_general %v4029, %v4030, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v4032 = stablehlo.reshape %v4031 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4033 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v4034 = stablehlo.multiply %v4032, %v4033 : tensor<32x38809xf32>
    %v4035 = stablehlo.reshape %v4034 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4036 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v4037 = stablehlo.reduce(%v4035 init: %v4036) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4038 = stablehlo.broadcast_in_dim %v4037, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4039 = stablehlo.subtract %v4035, %v4038 : tensor<32x197x197xf32>
    %v4040 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4041 = stablehlo.exponential %v4039 : tensor<32x197x197xf32>
    %v4042 = stablehlo.reduce(%v4041 init: %v4040) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4043 = stablehlo.broadcast_in_dim %v4042, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4044 = stablehlo.divide %v4041, %v4043 : tensor<32x197x197xf32>
    %v4045 = stablehlo.reshape %v4044 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4046 = stablehlo.reshape %v4045 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4047 = stablehlo.reshape %v4025 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4048 = stablehlo.dot_general %v4046, %v4047, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4049 = stablehlo.reshape %v4048 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4050 = stablehlo.reshape %v4049 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4051 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4052 = stablehlo.pad %v4050, %v4051, low = [0, 0, 192], high = [0, 0, 512], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v4053 = stablehlo.reshape %v4052 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4054 = stablehlo.add %v4016, %v4053 : tensor<32x151296xf32>
    %v4055 = stablehlo.reshape %v3893 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4056 = stablehlo.slice %v4055 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4057 = stablehlo.reshape %v4056 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4058 = stablehlo.reshape %v3898 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4059 = stablehlo.slice %v4058 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4060 = stablehlo.reshape %v4059 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4061 = stablehlo.reshape %v3903 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4062 = stablehlo.slice %v4061 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4063 = stablehlo.reshape %v4062 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4064 = stablehlo.reshape %v4060 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4065 = stablehlo.transpose %v4064, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v4066 = stablehlo.reshape %v4065 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4067 = stablehlo.reshape %v4057 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4068 = stablehlo.reshape %v4066 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4069 = stablehlo.dot_general %v4067, %v4068, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v4070 = stablehlo.reshape %v4069 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4071 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v4072 = stablehlo.multiply %v4070, %v4071 : tensor<32x38809xf32>
    %v4073 = stablehlo.reshape %v4072 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4074 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v4075 = stablehlo.reduce(%v4073 init: %v4074) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4076 = stablehlo.broadcast_in_dim %v4075, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4077 = stablehlo.subtract %v4073, %v4076 : tensor<32x197x197xf32>
    %v4078 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4079 = stablehlo.exponential %v4077 : tensor<32x197x197xf32>
    %v4080 = stablehlo.reduce(%v4079 init: %v4078) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4081 = stablehlo.broadcast_in_dim %v4080, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4082 = stablehlo.divide %v4079, %v4081 : tensor<32x197x197xf32>
    %v4083 = stablehlo.reshape %v4082 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4084 = stablehlo.reshape %v4083 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4085 = stablehlo.reshape %v4063 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4086 = stablehlo.dot_general %v4084, %v4085, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4087 = stablehlo.reshape %v4086 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4088 = stablehlo.reshape %v4087 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4089 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4090 = stablehlo.pad %v4088, %v4089, low = [0, 0, 256], high = [0, 0, 448], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v4091 = stablehlo.reshape %v4090 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4092 = stablehlo.add %v4054, %v4091 : tensor<32x151296xf32>
    %v4093 = stablehlo.reshape %v3893 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4094 = stablehlo.slice %v4093 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4095 = stablehlo.reshape %v4094 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4096 = stablehlo.reshape %v3898 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4097 = stablehlo.slice %v4096 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4098 = stablehlo.reshape %v4097 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4099 = stablehlo.reshape %v3903 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4100 = stablehlo.slice %v4099 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4101 = stablehlo.reshape %v4100 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4102 = stablehlo.reshape %v4098 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4103 = stablehlo.transpose %v4102, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v4104 = stablehlo.reshape %v4103 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4105 = stablehlo.reshape %v4095 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4106 = stablehlo.reshape %v4104 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4107 = stablehlo.dot_general %v4105, %v4106, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v4108 = stablehlo.reshape %v4107 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4109 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v4110 = stablehlo.multiply %v4108, %v4109 : tensor<32x38809xf32>
    %v4111 = stablehlo.reshape %v4110 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4112 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v4113 = stablehlo.reduce(%v4111 init: %v4112) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4114 = stablehlo.broadcast_in_dim %v4113, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4115 = stablehlo.subtract %v4111, %v4114 : tensor<32x197x197xf32>
    %v4116 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4117 = stablehlo.exponential %v4115 : tensor<32x197x197xf32>
    %v4118 = stablehlo.reduce(%v4117 init: %v4116) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4119 = stablehlo.broadcast_in_dim %v4118, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4120 = stablehlo.divide %v4117, %v4119 : tensor<32x197x197xf32>
    %v4121 = stablehlo.reshape %v4120 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4122 = stablehlo.reshape %v4121 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4123 = stablehlo.reshape %v4101 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4124 = stablehlo.dot_general %v4122, %v4123, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4125 = stablehlo.reshape %v4124 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4126 = stablehlo.reshape %v4125 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4127 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4128 = stablehlo.pad %v4126, %v4127, low = [0, 0, 320], high = [0, 0, 384], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v4129 = stablehlo.reshape %v4128 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4130 = stablehlo.add %v4092, %v4129 : tensor<32x151296xf32>
    %v4131 = stablehlo.reshape %v3893 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4132 = stablehlo.slice %v4131 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4133 = stablehlo.reshape %v4132 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4134 = stablehlo.reshape %v3898 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4135 = stablehlo.slice %v4134 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4136 = stablehlo.reshape %v4135 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4137 = stablehlo.reshape %v3903 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4138 = stablehlo.slice %v4137 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4139 = stablehlo.reshape %v4138 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4140 = stablehlo.reshape %v4136 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4141 = stablehlo.transpose %v4140, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v4142 = stablehlo.reshape %v4141 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4143 = stablehlo.reshape %v4133 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4144 = stablehlo.reshape %v4142 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4145 = stablehlo.dot_general %v4143, %v4144, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v4146 = stablehlo.reshape %v4145 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4147 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v4148 = stablehlo.multiply %v4146, %v4147 : tensor<32x38809xf32>
    %v4149 = stablehlo.reshape %v4148 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4150 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v4151 = stablehlo.reduce(%v4149 init: %v4150) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4152 = stablehlo.broadcast_in_dim %v4151, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4153 = stablehlo.subtract %v4149, %v4152 : tensor<32x197x197xf32>
    %v4154 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4155 = stablehlo.exponential %v4153 : tensor<32x197x197xf32>
    %v4156 = stablehlo.reduce(%v4155 init: %v4154) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4157 = stablehlo.broadcast_in_dim %v4156, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4158 = stablehlo.divide %v4155, %v4157 : tensor<32x197x197xf32>
    %v4159 = stablehlo.reshape %v4158 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4160 = stablehlo.reshape %v4159 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4161 = stablehlo.reshape %v4139 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4162 = stablehlo.dot_general %v4160, %v4161, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4163 = stablehlo.reshape %v4162 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4164 = stablehlo.reshape %v4163 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4165 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4166 = stablehlo.pad %v4164, %v4165, low = [0, 0, 384], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v4167 = stablehlo.reshape %v4166 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4168 = stablehlo.add %v4130, %v4167 : tensor<32x151296xf32>
    %v4169 = stablehlo.reshape %v3893 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4170 = stablehlo.slice %v4169 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4171 = stablehlo.reshape %v4170 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4172 = stablehlo.reshape %v3898 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4173 = stablehlo.slice %v4172 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4174 = stablehlo.reshape %v4173 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4175 = stablehlo.reshape %v3903 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4176 = stablehlo.slice %v4175 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4177 = stablehlo.reshape %v4176 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4178 = stablehlo.reshape %v4174 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4179 = stablehlo.transpose %v4178, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v4180 = stablehlo.reshape %v4179 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4181 = stablehlo.reshape %v4171 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4182 = stablehlo.reshape %v4180 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4183 = stablehlo.dot_general %v4181, %v4182, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v4184 = stablehlo.reshape %v4183 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4185 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v4186 = stablehlo.multiply %v4184, %v4185 : tensor<32x38809xf32>
    %v4187 = stablehlo.reshape %v4186 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4188 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v4189 = stablehlo.reduce(%v4187 init: %v4188) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4190 = stablehlo.broadcast_in_dim %v4189, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4191 = stablehlo.subtract %v4187, %v4190 : tensor<32x197x197xf32>
    %v4192 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4193 = stablehlo.exponential %v4191 : tensor<32x197x197xf32>
    %v4194 = stablehlo.reduce(%v4193 init: %v4192) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4195 = stablehlo.broadcast_in_dim %v4194, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4196 = stablehlo.divide %v4193, %v4195 : tensor<32x197x197xf32>
    %v4197 = stablehlo.reshape %v4196 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4198 = stablehlo.reshape %v4197 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4199 = stablehlo.reshape %v4177 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4200 = stablehlo.dot_general %v4198, %v4199, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4201 = stablehlo.reshape %v4200 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4202 = stablehlo.reshape %v4201 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4203 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4204 = stablehlo.pad %v4202, %v4203, low = [0, 0, 448], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v4205 = stablehlo.reshape %v4204 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4206 = stablehlo.add %v4168, %v4205 : tensor<32x151296xf32>
    %v4207 = stablehlo.reshape %v3893 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4208 = stablehlo.slice %v4207 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4209 = stablehlo.reshape %v4208 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4210 = stablehlo.reshape %v3898 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4211 = stablehlo.slice %v4210 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4212 = stablehlo.reshape %v4211 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4213 = stablehlo.reshape %v3903 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4214 = stablehlo.slice %v4213 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4215 = stablehlo.reshape %v4214 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4216 = stablehlo.reshape %v4212 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4217 = stablehlo.transpose %v4216, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v4218 = stablehlo.reshape %v4217 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4219 = stablehlo.reshape %v4209 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4220 = stablehlo.reshape %v4218 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4221 = stablehlo.dot_general %v4219, %v4220, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v4222 = stablehlo.reshape %v4221 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4223 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v4224 = stablehlo.multiply %v4222, %v4223 : tensor<32x38809xf32>
    %v4225 = stablehlo.reshape %v4224 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4226 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v4227 = stablehlo.reduce(%v4225 init: %v4226) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4228 = stablehlo.broadcast_in_dim %v4227, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4229 = stablehlo.subtract %v4225, %v4228 : tensor<32x197x197xf32>
    %v4230 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4231 = stablehlo.exponential %v4229 : tensor<32x197x197xf32>
    %v4232 = stablehlo.reduce(%v4231 init: %v4230) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4233 = stablehlo.broadcast_in_dim %v4232, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4234 = stablehlo.divide %v4231, %v4233 : tensor<32x197x197xf32>
    %v4235 = stablehlo.reshape %v4234 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4236 = stablehlo.reshape %v4235 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4237 = stablehlo.reshape %v4215 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4238 = stablehlo.dot_general %v4236, %v4237, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4239 = stablehlo.reshape %v4238 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4240 = stablehlo.reshape %v4239 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4241 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4242 = stablehlo.pad %v4240, %v4241, low = [0, 0, 512], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v4243 = stablehlo.reshape %v4242 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4244 = stablehlo.add %v4206, %v4243 : tensor<32x151296xf32>
    %v4245 = stablehlo.reshape %v3893 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4246 = stablehlo.slice %v4245 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4247 = stablehlo.reshape %v4246 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4248 = stablehlo.reshape %v3898 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4249 = stablehlo.slice %v4248 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4250 = stablehlo.reshape %v4249 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4251 = stablehlo.reshape %v3903 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4252 = stablehlo.slice %v4251 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4253 = stablehlo.reshape %v4252 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4254 = stablehlo.reshape %v4250 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4255 = stablehlo.transpose %v4254, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v4256 = stablehlo.reshape %v4255 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4257 = stablehlo.reshape %v4247 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4258 = stablehlo.reshape %v4256 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4259 = stablehlo.dot_general %v4257, %v4258, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v4260 = stablehlo.reshape %v4259 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4261 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v4262 = stablehlo.multiply %v4260, %v4261 : tensor<32x38809xf32>
    %v4263 = stablehlo.reshape %v4262 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4264 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v4265 = stablehlo.reduce(%v4263 init: %v4264) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4266 = stablehlo.broadcast_in_dim %v4265, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4267 = stablehlo.subtract %v4263, %v4266 : tensor<32x197x197xf32>
    %v4268 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4269 = stablehlo.exponential %v4267 : tensor<32x197x197xf32>
    %v4270 = stablehlo.reduce(%v4269 init: %v4268) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4271 = stablehlo.broadcast_in_dim %v4270, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4272 = stablehlo.divide %v4269, %v4271 : tensor<32x197x197xf32>
    %v4273 = stablehlo.reshape %v4272 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4274 = stablehlo.reshape %v4273 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4275 = stablehlo.reshape %v4253 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4276 = stablehlo.dot_general %v4274, %v4275, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4277 = stablehlo.reshape %v4276 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4278 = stablehlo.reshape %v4277 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4279 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4280 = stablehlo.pad %v4278, %v4279, low = [0, 0, 576], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v4281 = stablehlo.reshape %v4280 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4282 = stablehlo.add %v4244, %v4281 : tensor<32x151296xf32>
    %v4283 = stablehlo.reshape %v3893 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4284 = stablehlo.slice %v4283 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4285 = stablehlo.reshape %v4284 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4286 = stablehlo.reshape %v3898 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4287 = stablehlo.slice %v4286 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4288 = stablehlo.reshape %v4287 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4289 = stablehlo.reshape %v3903 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4290 = stablehlo.slice %v4289 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4291 = stablehlo.reshape %v4290 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4292 = stablehlo.reshape %v4288 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4293 = stablehlo.transpose %v4292, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v4294 = stablehlo.reshape %v4293 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4295 = stablehlo.reshape %v4285 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4296 = stablehlo.reshape %v4294 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4297 = stablehlo.dot_general %v4295, %v4296, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v4298 = stablehlo.reshape %v4297 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4299 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v4300 = stablehlo.multiply %v4298, %v4299 : tensor<32x38809xf32>
    %v4301 = stablehlo.reshape %v4300 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4302 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v4303 = stablehlo.reduce(%v4301 init: %v4302) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4304 = stablehlo.broadcast_in_dim %v4303, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4305 = stablehlo.subtract %v4301, %v4304 : tensor<32x197x197xf32>
    %v4306 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4307 = stablehlo.exponential %v4305 : tensor<32x197x197xf32>
    %v4308 = stablehlo.reduce(%v4307 init: %v4306) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4309 = stablehlo.broadcast_in_dim %v4308, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4310 = stablehlo.divide %v4307, %v4309 : tensor<32x197x197xf32>
    %v4311 = stablehlo.reshape %v4310 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4312 = stablehlo.reshape %v4311 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4313 = stablehlo.reshape %v4291 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4314 = stablehlo.dot_general %v4312, %v4313, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4315 = stablehlo.reshape %v4314 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4316 = stablehlo.reshape %v4315 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4317 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4318 = stablehlo.pad %v4316, %v4317, low = [0, 0, 640], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v4319 = stablehlo.reshape %v4318 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4320 = stablehlo.add %v4282, %v4319 : tensor<32x151296xf32>
    %v4321 = stablehlo.reshape %v3893 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4322 = stablehlo.slice %v4321 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4323 = stablehlo.reshape %v4322 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4324 = stablehlo.reshape %v3898 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4325 = stablehlo.slice %v4324 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4326 = stablehlo.reshape %v4325 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4327 = stablehlo.reshape %v3903 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4328 = stablehlo.slice %v4327 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4329 = stablehlo.reshape %v4328 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4330 = stablehlo.reshape %v4326 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4331 = stablehlo.transpose %v4330, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v4332 = stablehlo.reshape %v4331 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4333 = stablehlo.reshape %v4323 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4334 = stablehlo.reshape %v4332 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4335 = stablehlo.dot_general %v4333, %v4334, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v4336 = stablehlo.reshape %v4335 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4337 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v4338 = stablehlo.multiply %v4336, %v4337 : tensor<32x38809xf32>
    %v4339 = stablehlo.reshape %v4338 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4340 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v4341 = stablehlo.reduce(%v4339 init: %v4340) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4342 = stablehlo.broadcast_in_dim %v4341, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4343 = stablehlo.subtract %v4339, %v4342 : tensor<32x197x197xf32>
    %v4344 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4345 = stablehlo.exponential %v4343 : tensor<32x197x197xf32>
    %v4346 = stablehlo.reduce(%v4345 init: %v4344) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4347 = stablehlo.broadcast_in_dim %v4346, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4348 = stablehlo.divide %v4345, %v4347 : tensor<32x197x197xf32>
    %v4349 = stablehlo.reshape %v4348 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4350 = stablehlo.reshape %v4349 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4351 = stablehlo.reshape %v4329 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4352 = stablehlo.dot_general %v4350, %v4351, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4353 = stablehlo.reshape %v4352 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4354 = stablehlo.reshape %v4353 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4355 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4356 = stablehlo.pad %v4354, %v4355, low = [0, 0, 704], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v4357 = stablehlo.reshape %v4356 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4358 = stablehlo.add %v4320, %v4357 : tensor<32x151296xf32>
    %v4359 = stablehlo.reshape %v4358 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4360 = stablehlo.dot_general %v4359, %b7_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v4361 = stablehlo.broadcast_in_dim %b7_bo, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v4362 = stablehlo.add %v4360, %v4361 : tensor<32x197x768xf32>
    %v4363 = stablehlo.reshape %v4362 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4364 = stablehlo.add %v3860, %v4363 : tensor<32x151296xf32>
    %v4365 = stablehlo.reshape %v4364 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4366 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4367 = stablehlo.constant dense<768.0> : tensor<32x197x768xf32>
    %v4368 = stablehlo.constant dense<1.0e-6> : tensor<32x197x768xf32>
    %v4369 = stablehlo.reduce(%v4365 init: %v4366) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4370 = stablehlo.broadcast_in_dim %v4369, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v4371 = stablehlo.divide %v4370, %v4367 : tensor<32x197x768xf32>
    %v4372 = stablehlo.subtract %v4365, %v4371 : tensor<32x197x768xf32>
    %v4373 = stablehlo.multiply %v4372, %v4372 : tensor<32x197x768xf32>
    %v4374 = stablehlo.reduce(%v4373 init: %v4366) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4375 = stablehlo.broadcast_in_dim %v4374, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v4376 = stablehlo.divide %v4375, %v4367 : tensor<32x197x768xf32>
    %v4377 = stablehlo.add %v4376, %v4368 : tensor<32x197x768xf32>
    %v4378 = stablehlo.rsqrt %v4377 : tensor<32x197x768xf32>
    %v4379 = stablehlo.multiply %v4372, %v4378 : tensor<32x197x768xf32>
    %v4380 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v4381 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v4382 = stablehlo.multiply %v4379, %v4380 : tensor<32x197x768xf32>
    %v4383 = stablehlo.add %v4382, %v4381 : tensor<32x197x768xf32>
    %v4384 = stablehlo.reshape %v4383 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4385 = stablehlo.reshape %v4384 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4386 = stablehlo.broadcast_in_dim %b7_g2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v4387 = stablehlo.multiply %v4385, %v4386 : tensor<32x197x768xf32>
    %v4388 = stablehlo.reshape %v4387 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4389 = stablehlo.reshape %v4388 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4390 = stablehlo.broadcast_in_dim %b7_bt2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v4391 = stablehlo.add %v4389, %v4390 : tensor<32x197x768xf32>
    %v4392 = stablehlo.reshape %v4391 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4393 = stablehlo.reshape %v4392 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4394 = stablehlo.dot_general %v4393, %b7_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x3072xf32>) -> tensor<32x197x3072xf32>
    %v4395 = stablehlo.broadcast_in_dim %b7_bfc1, dims = [2] : (tensor<3072xf32>) -> tensor<32x197x3072xf32>
    %v4396 = stablehlo.add %v4394, %v4395 : tensor<32x197x3072xf32>
    %v4397 = stablehlo.reshape %v4396 : (tensor<32x197x3072xf32>) -> tensor<32x605184xf32>
    %v4398 = stablehlo.constant dense<0.5> : tensor<32x605184xf32>
    %v4399 = stablehlo.multiply %v4398, %v4397 : tensor<32x605184xf32>
    %v4400 = stablehlo.negate %v4397 : tensor<32x605184xf32>
    %v4401 = stablehlo.constant dense<0.7071067811865476> : tensor<32x605184xf32>
    %v4402 = stablehlo.multiply %v4400, %v4401 : tensor<32x605184xf32>
    %v4403 = chlo.erfc %v4402 : tensor<32x605184xf32> -> tensor<32x605184xf32>
    %v4404 = stablehlo.multiply %v4399, %v4403 : tensor<32x605184xf32>
    %v4405 = stablehlo.reshape %v4404 : (tensor<32x605184xf32>) -> tensor<32x197x3072xf32>
    %v4406 = stablehlo.dot_general %v4405, %b7_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x3072xf32>, tensor<3072x768xf32>) -> tensor<32x197x768xf32>
    %v4407 = stablehlo.broadcast_in_dim %b7_bfc2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v4408 = stablehlo.add %v4406, %v4407 : tensor<32x197x768xf32>
    %v4409 = stablehlo.reshape %v4408 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4410 = stablehlo.add %v4364, %v4409 : tensor<32x151296xf32>
    %v4411 = stablehlo.reshape %v4410 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4412 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4413 = stablehlo.constant dense<768.0> : tensor<32x197x768xf32>
    %v4414 = stablehlo.constant dense<1.0e-6> : tensor<32x197x768xf32>
    %v4415 = stablehlo.reduce(%v4411 init: %v4412) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4416 = stablehlo.broadcast_in_dim %v4415, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v4417 = stablehlo.divide %v4416, %v4413 : tensor<32x197x768xf32>
    %v4418 = stablehlo.subtract %v4411, %v4417 : tensor<32x197x768xf32>
    %v4419 = stablehlo.multiply %v4418, %v4418 : tensor<32x197x768xf32>
    %v4420 = stablehlo.reduce(%v4419 init: %v4412) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4421 = stablehlo.broadcast_in_dim %v4420, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v4422 = stablehlo.divide %v4421, %v4413 : tensor<32x197x768xf32>
    %v4423 = stablehlo.add %v4422, %v4414 : tensor<32x197x768xf32>
    %v4424 = stablehlo.rsqrt %v4423 : tensor<32x197x768xf32>
    %v4425 = stablehlo.multiply %v4418, %v4424 : tensor<32x197x768xf32>
    %v4426 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v4427 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v4428 = stablehlo.multiply %v4425, %v4426 : tensor<32x197x768xf32>
    %v4429 = stablehlo.add %v4428, %v4427 : tensor<32x197x768xf32>
    %v4430 = stablehlo.reshape %v4429 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4431 = stablehlo.reshape %v4430 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4432 = stablehlo.broadcast_in_dim %b8_g1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v4433 = stablehlo.multiply %v4431, %v4432 : tensor<32x197x768xf32>
    %v4434 = stablehlo.reshape %v4433 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4435 = stablehlo.reshape %v4434 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4436 = stablehlo.broadcast_in_dim %b8_bt1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v4437 = stablehlo.add %v4435, %v4436 : tensor<32x197x768xf32>
    %v4438 = stablehlo.reshape %v4437 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4439 = stablehlo.reshape %v4438 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4440 = stablehlo.dot_general %v4439, %b8_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v4441 = stablehlo.broadcast_in_dim %b8_bq, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v4442 = stablehlo.add %v4440, %v4441 : tensor<32x197x768xf32>
    %v4443 = stablehlo.reshape %v4442 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4444 = stablehlo.reshape %v4438 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4445 = stablehlo.dot_general %v4444, %b8_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v4446 = stablehlo.broadcast_in_dim %b8_bk, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v4447 = stablehlo.add %v4445, %v4446 : tensor<32x197x768xf32>
    %v4448 = stablehlo.reshape %v4447 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4449 = stablehlo.reshape %v4438 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4450 = stablehlo.dot_general %v4449, %b8_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v4451 = stablehlo.broadcast_in_dim %b8_bv, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v4452 = stablehlo.add %v4450, %v4451 : tensor<32x197x768xf32>
    %v4453 = stablehlo.reshape %v4452 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4454 = stablehlo.reshape %v4443 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4455 = stablehlo.slice %v4454 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4456 = stablehlo.reshape %v4455 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4457 = stablehlo.reshape %v4448 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4458 = stablehlo.slice %v4457 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4459 = stablehlo.reshape %v4458 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4460 = stablehlo.reshape %v4453 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4461 = stablehlo.slice %v4460 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4462 = stablehlo.reshape %v4461 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4463 = stablehlo.reshape %v4459 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4464 = stablehlo.transpose %v4463, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v4465 = stablehlo.reshape %v4464 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4466 = stablehlo.reshape %v4456 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4467 = stablehlo.reshape %v4465 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4468 = stablehlo.dot_general %v4466, %v4467, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v4469 = stablehlo.reshape %v4468 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4470 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v4471 = stablehlo.multiply %v4469, %v4470 : tensor<32x38809xf32>
    %v4472 = stablehlo.reshape %v4471 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4473 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v4474 = stablehlo.reduce(%v4472 init: %v4473) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4475 = stablehlo.broadcast_in_dim %v4474, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4476 = stablehlo.subtract %v4472, %v4475 : tensor<32x197x197xf32>
    %v4477 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4478 = stablehlo.exponential %v4476 : tensor<32x197x197xf32>
    %v4479 = stablehlo.reduce(%v4478 init: %v4477) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4480 = stablehlo.broadcast_in_dim %v4479, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4481 = stablehlo.divide %v4478, %v4480 : tensor<32x197x197xf32>
    %v4482 = stablehlo.reshape %v4481 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4483 = stablehlo.reshape %v4482 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4484 = stablehlo.reshape %v4462 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4485 = stablehlo.dot_general %v4483, %v4484, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4486 = stablehlo.reshape %v4485 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4487 = stablehlo.reshape %v4486 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4488 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4489 = stablehlo.pad %v4487, %v4488, low = [0, 0, 0], high = [0, 0, 704], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v4490 = stablehlo.reshape %v4489 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4491 = stablehlo.reshape %v4443 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4492 = stablehlo.slice %v4491 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4493 = stablehlo.reshape %v4492 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4494 = stablehlo.reshape %v4448 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4495 = stablehlo.slice %v4494 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4496 = stablehlo.reshape %v4495 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4497 = stablehlo.reshape %v4453 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4498 = stablehlo.slice %v4497 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4499 = stablehlo.reshape %v4498 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4500 = stablehlo.reshape %v4496 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4501 = stablehlo.transpose %v4500, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v4502 = stablehlo.reshape %v4501 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4503 = stablehlo.reshape %v4493 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4504 = stablehlo.reshape %v4502 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4505 = stablehlo.dot_general %v4503, %v4504, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v4506 = stablehlo.reshape %v4505 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4507 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v4508 = stablehlo.multiply %v4506, %v4507 : tensor<32x38809xf32>
    %v4509 = stablehlo.reshape %v4508 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4510 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v4511 = stablehlo.reduce(%v4509 init: %v4510) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4512 = stablehlo.broadcast_in_dim %v4511, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4513 = stablehlo.subtract %v4509, %v4512 : tensor<32x197x197xf32>
    %v4514 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4515 = stablehlo.exponential %v4513 : tensor<32x197x197xf32>
    %v4516 = stablehlo.reduce(%v4515 init: %v4514) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4517 = stablehlo.broadcast_in_dim %v4516, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4518 = stablehlo.divide %v4515, %v4517 : tensor<32x197x197xf32>
    %v4519 = stablehlo.reshape %v4518 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4520 = stablehlo.reshape %v4519 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4521 = stablehlo.reshape %v4499 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4522 = stablehlo.dot_general %v4520, %v4521, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4523 = stablehlo.reshape %v4522 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4524 = stablehlo.reshape %v4523 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4525 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4526 = stablehlo.pad %v4524, %v4525, low = [0, 0, 64], high = [0, 0, 640], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v4527 = stablehlo.reshape %v4526 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4528 = stablehlo.add %v4490, %v4527 : tensor<32x151296xf32>
    %v4529 = stablehlo.reshape %v4443 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4530 = stablehlo.slice %v4529 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4531 = stablehlo.reshape %v4530 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4532 = stablehlo.reshape %v4448 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4533 = stablehlo.slice %v4532 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4534 = stablehlo.reshape %v4533 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4535 = stablehlo.reshape %v4453 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4536 = stablehlo.slice %v4535 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4537 = stablehlo.reshape %v4536 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4538 = stablehlo.reshape %v4534 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4539 = stablehlo.transpose %v4538, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v4540 = stablehlo.reshape %v4539 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4541 = stablehlo.reshape %v4531 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4542 = stablehlo.reshape %v4540 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4543 = stablehlo.dot_general %v4541, %v4542, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v4544 = stablehlo.reshape %v4543 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4545 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v4546 = stablehlo.multiply %v4544, %v4545 : tensor<32x38809xf32>
    %v4547 = stablehlo.reshape %v4546 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4548 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v4549 = stablehlo.reduce(%v4547 init: %v4548) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4550 = stablehlo.broadcast_in_dim %v4549, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4551 = stablehlo.subtract %v4547, %v4550 : tensor<32x197x197xf32>
    %v4552 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4553 = stablehlo.exponential %v4551 : tensor<32x197x197xf32>
    %v4554 = stablehlo.reduce(%v4553 init: %v4552) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4555 = stablehlo.broadcast_in_dim %v4554, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4556 = stablehlo.divide %v4553, %v4555 : tensor<32x197x197xf32>
    %v4557 = stablehlo.reshape %v4556 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4558 = stablehlo.reshape %v4557 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4559 = stablehlo.reshape %v4537 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4560 = stablehlo.dot_general %v4558, %v4559, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4561 = stablehlo.reshape %v4560 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4562 = stablehlo.reshape %v4561 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4563 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4564 = stablehlo.pad %v4562, %v4563, low = [0, 0, 128], high = [0, 0, 576], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v4565 = stablehlo.reshape %v4564 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4566 = stablehlo.add %v4528, %v4565 : tensor<32x151296xf32>
    %v4567 = stablehlo.reshape %v4443 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4568 = stablehlo.slice %v4567 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4569 = stablehlo.reshape %v4568 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4570 = stablehlo.reshape %v4448 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4571 = stablehlo.slice %v4570 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4572 = stablehlo.reshape %v4571 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4573 = stablehlo.reshape %v4453 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4574 = stablehlo.slice %v4573 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4575 = stablehlo.reshape %v4574 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4576 = stablehlo.reshape %v4572 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4577 = stablehlo.transpose %v4576, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v4578 = stablehlo.reshape %v4577 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4579 = stablehlo.reshape %v4569 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4580 = stablehlo.reshape %v4578 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4581 = stablehlo.dot_general %v4579, %v4580, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v4582 = stablehlo.reshape %v4581 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4583 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v4584 = stablehlo.multiply %v4582, %v4583 : tensor<32x38809xf32>
    %v4585 = stablehlo.reshape %v4584 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4586 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v4587 = stablehlo.reduce(%v4585 init: %v4586) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4588 = stablehlo.broadcast_in_dim %v4587, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4589 = stablehlo.subtract %v4585, %v4588 : tensor<32x197x197xf32>
    %v4590 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4591 = stablehlo.exponential %v4589 : tensor<32x197x197xf32>
    %v4592 = stablehlo.reduce(%v4591 init: %v4590) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4593 = stablehlo.broadcast_in_dim %v4592, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4594 = stablehlo.divide %v4591, %v4593 : tensor<32x197x197xf32>
    %v4595 = stablehlo.reshape %v4594 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4596 = stablehlo.reshape %v4595 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4597 = stablehlo.reshape %v4575 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4598 = stablehlo.dot_general %v4596, %v4597, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4599 = stablehlo.reshape %v4598 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4600 = stablehlo.reshape %v4599 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4601 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4602 = stablehlo.pad %v4600, %v4601, low = [0, 0, 192], high = [0, 0, 512], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v4603 = stablehlo.reshape %v4602 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4604 = stablehlo.add %v4566, %v4603 : tensor<32x151296xf32>
    %v4605 = stablehlo.reshape %v4443 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4606 = stablehlo.slice %v4605 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4607 = stablehlo.reshape %v4606 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4608 = stablehlo.reshape %v4448 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4609 = stablehlo.slice %v4608 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4610 = stablehlo.reshape %v4609 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4611 = stablehlo.reshape %v4453 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4612 = stablehlo.slice %v4611 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
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
    %v4624 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v4625 = stablehlo.reduce(%v4623 init: %v4624) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4626 = stablehlo.broadcast_in_dim %v4625, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4627 = stablehlo.subtract %v4623, %v4626 : tensor<32x197x197xf32>
    %v4628 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4629 = stablehlo.exponential %v4627 : tensor<32x197x197xf32>
    %v4630 = stablehlo.reduce(%v4629 init: %v4628) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4631 = stablehlo.broadcast_in_dim %v4630, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4632 = stablehlo.divide %v4629, %v4631 : tensor<32x197x197xf32>
    %v4633 = stablehlo.reshape %v4632 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4634 = stablehlo.reshape %v4633 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4635 = stablehlo.reshape %v4613 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4636 = stablehlo.dot_general %v4634, %v4635, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4637 = stablehlo.reshape %v4636 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4638 = stablehlo.reshape %v4637 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4639 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4640 = stablehlo.pad %v4638, %v4639, low = [0, 0, 256], high = [0, 0, 448], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v4641 = stablehlo.reshape %v4640 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4642 = stablehlo.add %v4604, %v4641 : tensor<32x151296xf32>
    %v4643 = stablehlo.reshape %v4443 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4644 = stablehlo.slice %v4643 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4645 = stablehlo.reshape %v4644 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4646 = stablehlo.reshape %v4448 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4647 = stablehlo.slice %v4646 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4648 = stablehlo.reshape %v4647 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4649 = stablehlo.reshape %v4453 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4650 = stablehlo.slice %v4649 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4651 = stablehlo.reshape %v4650 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4652 = stablehlo.reshape %v4648 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4653 = stablehlo.transpose %v4652, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v4654 = stablehlo.reshape %v4653 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4655 = stablehlo.reshape %v4645 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4656 = stablehlo.reshape %v4654 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4657 = stablehlo.dot_general %v4655, %v4656, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v4658 = stablehlo.reshape %v4657 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4659 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v4660 = stablehlo.multiply %v4658, %v4659 : tensor<32x38809xf32>
    %v4661 = stablehlo.reshape %v4660 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4662 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v4663 = stablehlo.reduce(%v4661 init: %v4662) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4664 = stablehlo.broadcast_in_dim %v4663, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4665 = stablehlo.subtract %v4661, %v4664 : tensor<32x197x197xf32>
    %v4666 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4667 = stablehlo.exponential %v4665 : tensor<32x197x197xf32>
    %v4668 = stablehlo.reduce(%v4667 init: %v4666) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4669 = stablehlo.broadcast_in_dim %v4668, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4670 = stablehlo.divide %v4667, %v4669 : tensor<32x197x197xf32>
    %v4671 = stablehlo.reshape %v4670 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4672 = stablehlo.reshape %v4671 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4673 = stablehlo.reshape %v4651 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4674 = stablehlo.dot_general %v4672, %v4673, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4675 = stablehlo.reshape %v4674 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4676 = stablehlo.reshape %v4675 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4677 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4678 = stablehlo.pad %v4676, %v4677, low = [0, 0, 320], high = [0, 0, 384], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v4679 = stablehlo.reshape %v4678 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4680 = stablehlo.add %v4642, %v4679 : tensor<32x151296xf32>
    %v4681 = stablehlo.reshape %v4443 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4682 = stablehlo.slice %v4681 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4683 = stablehlo.reshape %v4682 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4684 = stablehlo.reshape %v4448 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4685 = stablehlo.slice %v4684 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4686 = stablehlo.reshape %v4685 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4687 = stablehlo.reshape %v4453 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4688 = stablehlo.slice %v4687 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4689 = stablehlo.reshape %v4688 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4690 = stablehlo.reshape %v4686 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4691 = stablehlo.transpose %v4690, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v4692 = stablehlo.reshape %v4691 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4693 = stablehlo.reshape %v4683 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4694 = stablehlo.reshape %v4692 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4695 = stablehlo.dot_general %v4693, %v4694, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v4696 = stablehlo.reshape %v4695 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4697 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v4698 = stablehlo.multiply %v4696, %v4697 : tensor<32x38809xf32>
    %v4699 = stablehlo.reshape %v4698 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4700 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v4701 = stablehlo.reduce(%v4699 init: %v4700) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4702 = stablehlo.broadcast_in_dim %v4701, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4703 = stablehlo.subtract %v4699, %v4702 : tensor<32x197x197xf32>
    %v4704 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4705 = stablehlo.exponential %v4703 : tensor<32x197x197xf32>
    %v4706 = stablehlo.reduce(%v4705 init: %v4704) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4707 = stablehlo.broadcast_in_dim %v4706, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4708 = stablehlo.divide %v4705, %v4707 : tensor<32x197x197xf32>
    %v4709 = stablehlo.reshape %v4708 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4710 = stablehlo.reshape %v4709 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4711 = stablehlo.reshape %v4689 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4712 = stablehlo.dot_general %v4710, %v4711, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4713 = stablehlo.reshape %v4712 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4714 = stablehlo.reshape %v4713 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4715 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4716 = stablehlo.pad %v4714, %v4715, low = [0, 0, 384], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v4717 = stablehlo.reshape %v4716 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4718 = stablehlo.add %v4680, %v4717 : tensor<32x151296xf32>
    %v4719 = stablehlo.reshape %v4443 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4720 = stablehlo.slice %v4719 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4721 = stablehlo.reshape %v4720 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4722 = stablehlo.reshape %v4448 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4723 = stablehlo.slice %v4722 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4724 = stablehlo.reshape %v4723 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4725 = stablehlo.reshape %v4453 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4726 = stablehlo.slice %v4725 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4727 = stablehlo.reshape %v4726 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4728 = stablehlo.reshape %v4724 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4729 = stablehlo.transpose %v4728, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v4730 = stablehlo.reshape %v4729 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4731 = stablehlo.reshape %v4721 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4732 = stablehlo.reshape %v4730 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4733 = stablehlo.dot_general %v4731, %v4732, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v4734 = stablehlo.reshape %v4733 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4735 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v4736 = stablehlo.multiply %v4734, %v4735 : tensor<32x38809xf32>
    %v4737 = stablehlo.reshape %v4736 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4738 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v4739 = stablehlo.reduce(%v4737 init: %v4738) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4740 = stablehlo.broadcast_in_dim %v4739, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4741 = stablehlo.subtract %v4737, %v4740 : tensor<32x197x197xf32>
    %v4742 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4743 = stablehlo.exponential %v4741 : tensor<32x197x197xf32>
    %v4744 = stablehlo.reduce(%v4743 init: %v4742) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4745 = stablehlo.broadcast_in_dim %v4744, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4746 = stablehlo.divide %v4743, %v4745 : tensor<32x197x197xf32>
    %v4747 = stablehlo.reshape %v4746 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4748 = stablehlo.reshape %v4747 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4749 = stablehlo.reshape %v4727 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4750 = stablehlo.dot_general %v4748, %v4749, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4751 = stablehlo.reshape %v4750 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4752 = stablehlo.reshape %v4751 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4753 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4754 = stablehlo.pad %v4752, %v4753, low = [0, 0, 448], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v4755 = stablehlo.reshape %v4754 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4756 = stablehlo.add %v4718, %v4755 : tensor<32x151296xf32>
    %v4757 = stablehlo.reshape %v4443 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4758 = stablehlo.slice %v4757 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4759 = stablehlo.reshape %v4758 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4760 = stablehlo.reshape %v4448 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4761 = stablehlo.slice %v4760 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4762 = stablehlo.reshape %v4761 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4763 = stablehlo.reshape %v4453 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4764 = stablehlo.slice %v4763 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4765 = stablehlo.reshape %v4764 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4766 = stablehlo.reshape %v4762 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4767 = stablehlo.transpose %v4766, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v4768 = stablehlo.reshape %v4767 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4769 = stablehlo.reshape %v4759 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4770 = stablehlo.reshape %v4768 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4771 = stablehlo.dot_general %v4769, %v4770, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v4772 = stablehlo.reshape %v4771 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4773 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v4774 = stablehlo.multiply %v4772, %v4773 : tensor<32x38809xf32>
    %v4775 = stablehlo.reshape %v4774 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4776 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v4777 = stablehlo.reduce(%v4775 init: %v4776) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4778 = stablehlo.broadcast_in_dim %v4777, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4779 = stablehlo.subtract %v4775, %v4778 : tensor<32x197x197xf32>
    %v4780 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4781 = stablehlo.exponential %v4779 : tensor<32x197x197xf32>
    %v4782 = stablehlo.reduce(%v4781 init: %v4780) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4783 = stablehlo.broadcast_in_dim %v4782, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4784 = stablehlo.divide %v4781, %v4783 : tensor<32x197x197xf32>
    %v4785 = stablehlo.reshape %v4784 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4786 = stablehlo.reshape %v4785 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4787 = stablehlo.reshape %v4765 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4788 = stablehlo.dot_general %v4786, %v4787, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4789 = stablehlo.reshape %v4788 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4790 = stablehlo.reshape %v4789 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4791 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4792 = stablehlo.pad %v4790, %v4791, low = [0, 0, 512], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v4793 = stablehlo.reshape %v4792 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4794 = stablehlo.add %v4756, %v4793 : tensor<32x151296xf32>
    %v4795 = stablehlo.reshape %v4443 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4796 = stablehlo.slice %v4795 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4797 = stablehlo.reshape %v4796 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4798 = stablehlo.reshape %v4448 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4799 = stablehlo.slice %v4798 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4800 = stablehlo.reshape %v4799 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4801 = stablehlo.reshape %v4453 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4802 = stablehlo.slice %v4801 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4803 = stablehlo.reshape %v4802 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4804 = stablehlo.reshape %v4800 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4805 = stablehlo.transpose %v4804, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v4806 = stablehlo.reshape %v4805 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4807 = stablehlo.reshape %v4797 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4808 = stablehlo.reshape %v4806 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4809 = stablehlo.dot_general %v4807, %v4808, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v4810 = stablehlo.reshape %v4809 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4811 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v4812 = stablehlo.multiply %v4810, %v4811 : tensor<32x38809xf32>
    %v4813 = stablehlo.reshape %v4812 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4814 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v4815 = stablehlo.reduce(%v4813 init: %v4814) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4816 = stablehlo.broadcast_in_dim %v4815, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4817 = stablehlo.subtract %v4813, %v4816 : tensor<32x197x197xf32>
    %v4818 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4819 = stablehlo.exponential %v4817 : tensor<32x197x197xf32>
    %v4820 = stablehlo.reduce(%v4819 init: %v4818) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4821 = stablehlo.broadcast_in_dim %v4820, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4822 = stablehlo.divide %v4819, %v4821 : tensor<32x197x197xf32>
    %v4823 = stablehlo.reshape %v4822 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4824 = stablehlo.reshape %v4823 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4825 = stablehlo.reshape %v4803 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4826 = stablehlo.dot_general %v4824, %v4825, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4827 = stablehlo.reshape %v4826 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4828 = stablehlo.reshape %v4827 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4829 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4830 = stablehlo.pad %v4828, %v4829, low = [0, 0, 576], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v4831 = stablehlo.reshape %v4830 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4832 = stablehlo.add %v4794, %v4831 : tensor<32x151296xf32>
    %v4833 = stablehlo.reshape %v4443 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4834 = stablehlo.slice %v4833 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4835 = stablehlo.reshape %v4834 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4836 = stablehlo.reshape %v4448 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4837 = stablehlo.slice %v4836 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4838 = stablehlo.reshape %v4837 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4839 = stablehlo.reshape %v4453 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4840 = stablehlo.slice %v4839 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4841 = stablehlo.reshape %v4840 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4842 = stablehlo.reshape %v4838 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4843 = stablehlo.transpose %v4842, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v4844 = stablehlo.reshape %v4843 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4845 = stablehlo.reshape %v4835 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4846 = stablehlo.reshape %v4844 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4847 = stablehlo.dot_general %v4845, %v4846, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v4848 = stablehlo.reshape %v4847 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4849 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v4850 = stablehlo.multiply %v4848, %v4849 : tensor<32x38809xf32>
    %v4851 = stablehlo.reshape %v4850 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4852 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v4853 = stablehlo.reduce(%v4851 init: %v4852) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4854 = stablehlo.broadcast_in_dim %v4853, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4855 = stablehlo.subtract %v4851, %v4854 : tensor<32x197x197xf32>
    %v4856 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4857 = stablehlo.exponential %v4855 : tensor<32x197x197xf32>
    %v4858 = stablehlo.reduce(%v4857 init: %v4856) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4859 = stablehlo.broadcast_in_dim %v4858, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4860 = stablehlo.divide %v4857, %v4859 : tensor<32x197x197xf32>
    %v4861 = stablehlo.reshape %v4860 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4862 = stablehlo.reshape %v4861 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4863 = stablehlo.reshape %v4841 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4864 = stablehlo.dot_general %v4862, %v4863, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4865 = stablehlo.reshape %v4864 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4866 = stablehlo.reshape %v4865 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4867 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4868 = stablehlo.pad %v4866, %v4867, low = [0, 0, 640], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v4869 = stablehlo.reshape %v4868 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4870 = stablehlo.add %v4832, %v4869 : tensor<32x151296xf32>
    %v4871 = stablehlo.reshape %v4443 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4872 = stablehlo.slice %v4871 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4873 = stablehlo.reshape %v4872 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4874 = stablehlo.reshape %v4448 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4875 = stablehlo.slice %v4874 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4876 = stablehlo.reshape %v4875 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4877 = stablehlo.reshape %v4453 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4878 = stablehlo.slice %v4877 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4879 = stablehlo.reshape %v4878 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4880 = stablehlo.reshape %v4876 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4881 = stablehlo.transpose %v4880, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v4882 = stablehlo.reshape %v4881 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4883 = stablehlo.reshape %v4873 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4884 = stablehlo.reshape %v4882 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4885 = stablehlo.dot_general %v4883, %v4884, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v4886 = stablehlo.reshape %v4885 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4887 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v4888 = stablehlo.multiply %v4886, %v4887 : tensor<32x38809xf32>
    %v4889 = stablehlo.reshape %v4888 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4890 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v4891 = stablehlo.reduce(%v4889 init: %v4890) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4892 = stablehlo.broadcast_in_dim %v4891, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4893 = stablehlo.subtract %v4889, %v4892 : tensor<32x197x197xf32>
    %v4894 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4895 = stablehlo.exponential %v4893 : tensor<32x197x197xf32>
    %v4896 = stablehlo.reduce(%v4895 init: %v4894) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4897 = stablehlo.broadcast_in_dim %v4896, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4898 = stablehlo.divide %v4895, %v4897 : tensor<32x197x197xf32>
    %v4899 = stablehlo.reshape %v4898 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4900 = stablehlo.reshape %v4899 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4901 = stablehlo.reshape %v4879 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4902 = stablehlo.dot_general %v4900, %v4901, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4903 = stablehlo.reshape %v4902 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4904 = stablehlo.reshape %v4903 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4905 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4906 = stablehlo.pad %v4904, %v4905, low = [0, 0, 704], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v4907 = stablehlo.reshape %v4906 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4908 = stablehlo.add %v4870, %v4907 : tensor<32x151296xf32>
    %v4909 = stablehlo.reshape %v4908 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4910 = stablehlo.dot_general %v4909, %b8_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v4911 = stablehlo.broadcast_in_dim %b8_bo, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v4912 = stablehlo.add %v4910, %v4911 : tensor<32x197x768xf32>
    %v4913 = stablehlo.reshape %v4912 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4914 = stablehlo.add %v4410, %v4913 : tensor<32x151296xf32>
    %v4915 = stablehlo.reshape %v4914 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4916 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4917 = stablehlo.constant dense<768.0> : tensor<32x197x768xf32>
    %v4918 = stablehlo.constant dense<1.0e-6> : tensor<32x197x768xf32>
    %v4919 = stablehlo.reduce(%v4915 init: %v4916) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4920 = stablehlo.broadcast_in_dim %v4919, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v4921 = stablehlo.divide %v4920, %v4917 : tensor<32x197x768xf32>
    %v4922 = stablehlo.subtract %v4915, %v4921 : tensor<32x197x768xf32>
    %v4923 = stablehlo.multiply %v4922, %v4922 : tensor<32x197x768xf32>
    %v4924 = stablehlo.reduce(%v4923 init: %v4916) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4925 = stablehlo.broadcast_in_dim %v4924, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v4926 = stablehlo.divide %v4925, %v4917 : tensor<32x197x768xf32>
    %v4927 = stablehlo.add %v4926, %v4918 : tensor<32x197x768xf32>
    %v4928 = stablehlo.rsqrt %v4927 : tensor<32x197x768xf32>
    %v4929 = stablehlo.multiply %v4922, %v4928 : tensor<32x197x768xf32>
    %v4930 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v4931 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v4932 = stablehlo.multiply %v4929, %v4930 : tensor<32x197x768xf32>
    %v4933 = stablehlo.add %v4932, %v4931 : tensor<32x197x768xf32>
    %v4934 = stablehlo.reshape %v4933 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4935 = stablehlo.reshape %v4934 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4936 = stablehlo.broadcast_in_dim %b8_g2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v4937 = stablehlo.multiply %v4935, %v4936 : tensor<32x197x768xf32>
    %v4938 = stablehlo.reshape %v4937 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4939 = stablehlo.reshape %v4938 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4940 = stablehlo.broadcast_in_dim %b8_bt2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v4941 = stablehlo.add %v4939, %v4940 : tensor<32x197x768xf32>
    %v4942 = stablehlo.reshape %v4941 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4943 = stablehlo.reshape %v4942 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4944 = stablehlo.dot_general %v4943, %b8_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x3072xf32>) -> tensor<32x197x3072xf32>
    %v4945 = stablehlo.broadcast_in_dim %b8_bfc1, dims = [2] : (tensor<3072xf32>) -> tensor<32x197x3072xf32>
    %v4946 = stablehlo.add %v4944, %v4945 : tensor<32x197x3072xf32>
    %v4947 = stablehlo.reshape %v4946 : (tensor<32x197x3072xf32>) -> tensor<32x605184xf32>
    %v4948 = stablehlo.constant dense<0.5> : tensor<32x605184xf32>
    %v4949 = stablehlo.multiply %v4948, %v4947 : tensor<32x605184xf32>
    %v4950 = stablehlo.negate %v4947 : tensor<32x605184xf32>
    %v4951 = stablehlo.constant dense<0.7071067811865476> : tensor<32x605184xf32>
    %v4952 = stablehlo.multiply %v4950, %v4951 : tensor<32x605184xf32>
    %v4953 = chlo.erfc %v4952 : tensor<32x605184xf32> -> tensor<32x605184xf32>
    %v4954 = stablehlo.multiply %v4949, %v4953 : tensor<32x605184xf32>
    %v4955 = stablehlo.reshape %v4954 : (tensor<32x605184xf32>) -> tensor<32x197x3072xf32>
    %v4956 = stablehlo.dot_general %v4955, %b8_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x3072xf32>, tensor<3072x768xf32>) -> tensor<32x197x768xf32>
    %v4957 = stablehlo.broadcast_in_dim %b8_bfc2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v4958 = stablehlo.add %v4956, %v4957 : tensor<32x197x768xf32>
    %v4959 = stablehlo.reshape %v4958 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4960 = stablehlo.add %v4914, %v4959 : tensor<32x151296xf32>
    %v4961 = stablehlo.reshape %v4960 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4962 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4963 = stablehlo.constant dense<768.0> : tensor<32x197x768xf32>
    %v4964 = stablehlo.constant dense<1.0e-6> : tensor<32x197x768xf32>
    %v4965 = stablehlo.reduce(%v4961 init: %v4962) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4966 = stablehlo.broadcast_in_dim %v4965, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v4967 = stablehlo.divide %v4966, %v4963 : tensor<32x197x768xf32>
    %v4968 = stablehlo.subtract %v4961, %v4967 : tensor<32x197x768xf32>
    %v4969 = stablehlo.multiply %v4968, %v4968 : tensor<32x197x768xf32>
    %v4970 = stablehlo.reduce(%v4969 init: %v4962) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4971 = stablehlo.broadcast_in_dim %v4970, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v4972 = stablehlo.divide %v4971, %v4963 : tensor<32x197x768xf32>
    %v4973 = stablehlo.add %v4972, %v4964 : tensor<32x197x768xf32>
    %v4974 = stablehlo.rsqrt %v4973 : tensor<32x197x768xf32>
    %v4975 = stablehlo.multiply %v4968, %v4974 : tensor<32x197x768xf32>
    %v4976 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v4977 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v4978 = stablehlo.multiply %v4975, %v4976 : tensor<32x197x768xf32>
    %v4979 = stablehlo.add %v4978, %v4977 : tensor<32x197x768xf32>
    %v4980 = stablehlo.reshape %v4979 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4981 = stablehlo.reshape %v4980 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4982 = stablehlo.broadcast_in_dim %b9_g1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v4983 = stablehlo.multiply %v4981, %v4982 : tensor<32x197x768xf32>
    %v4984 = stablehlo.reshape %v4983 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4985 = stablehlo.reshape %v4984 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4986 = stablehlo.broadcast_in_dim %b9_bt1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v4987 = stablehlo.add %v4985, %v4986 : tensor<32x197x768xf32>
    %v4988 = stablehlo.reshape %v4987 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4989 = stablehlo.reshape %v4988 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4990 = stablehlo.dot_general %v4989, %b9_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v4991 = stablehlo.broadcast_in_dim %b9_bq, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v4992 = stablehlo.add %v4990, %v4991 : tensor<32x197x768xf32>
    %v4993 = stablehlo.reshape %v4992 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4994 = stablehlo.reshape %v4988 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4995 = stablehlo.dot_general %v4994, %b9_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v4996 = stablehlo.broadcast_in_dim %b9_bk, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v4997 = stablehlo.add %v4995, %v4996 : tensor<32x197x768xf32>
    %v4998 = stablehlo.reshape %v4997 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4999 = stablehlo.reshape %v4988 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5000 = stablehlo.dot_general %v4999, %b9_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v5001 = stablehlo.broadcast_in_dim %b9_bv, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v5002 = stablehlo.add %v5000, %v5001 : tensor<32x197x768xf32>
    %v5003 = stablehlo.reshape %v5002 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5004 = stablehlo.reshape %v4993 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5005 = stablehlo.slice %v5004 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5006 = stablehlo.reshape %v5005 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5007 = stablehlo.reshape %v4998 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5008 = stablehlo.slice %v5007 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5009 = stablehlo.reshape %v5008 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5010 = stablehlo.reshape %v5003 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5011 = stablehlo.slice %v5010 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5012 = stablehlo.reshape %v5011 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5013 = stablehlo.reshape %v5009 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5014 = stablehlo.transpose %v5013, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5015 = stablehlo.reshape %v5014 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5016 = stablehlo.reshape %v5006 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5017 = stablehlo.reshape %v5015 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5018 = stablehlo.dot_general %v5016, %v5017, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v5019 = stablehlo.reshape %v5018 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5020 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v5021 = stablehlo.multiply %v5019, %v5020 : tensor<32x38809xf32>
    %v5022 = stablehlo.reshape %v5021 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5023 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v5024 = stablehlo.reduce(%v5022 init: %v5023) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5025 = stablehlo.broadcast_in_dim %v5024, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5026 = stablehlo.subtract %v5022, %v5025 : tensor<32x197x197xf32>
    %v5027 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5028 = stablehlo.exponential %v5026 : tensor<32x197x197xf32>
    %v5029 = stablehlo.reduce(%v5028 init: %v5027) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5030 = stablehlo.broadcast_in_dim %v5029, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5031 = stablehlo.divide %v5028, %v5030 : tensor<32x197x197xf32>
    %v5032 = stablehlo.reshape %v5031 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5033 = stablehlo.reshape %v5032 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5034 = stablehlo.reshape %v5012 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5035 = stablehlo.dot_general %v5033, %v5034, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5036 = stablehlo.reshape %v5035 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5037 = stablehlo.reshape %v5036 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5038 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5039 = stablehlo.pad %v5037, %v5038, low = [0, 0, 0], high = [0, 0, 704], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v5040 = stablehlo.reshape %v5039 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5041 = stablehlo.reshape %v4993 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5042 = stablehlo.slice %v5041 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5043 = stablehlo.reshape %v5042 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5044 = stablehlo.reshape %v4998 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5045 = stablehlo.slice %v5044 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5046 = stablehlo.reshape %v5045 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5047 = stablehlo.reshape %v5003 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5048 = stablehlo.slice %v5047 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5049 = stablehlo.reshape %v5048 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5050 = stablehlo.reshape %v5046 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5051 = stablehlo.transpose %v5050, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5052 = stablehlo.reshape %v5051 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5053 = stablehlo.reshape %v5043 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5054 = stablehlo.reshape %v5052 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5055 = stablehlo.dot_general %v5053, %v5054, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v5056 = stablehlo.reshape %v5055 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5057 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v5058 = stablehlo.multiply %v5056, %v5057 : tensor<32x38809xf32>
    %v5059 = stablehlo.reshape %v5058 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5060 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v5061 = stablehlo.reduce(%v5059 init: %v5060) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5062 = stablehlo.broadcast_in_dim %v5061, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5063 = stablehlo.subtract %v5059, %v5062 : tensor<32x197x197xf32>
    %v5064 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5065 = stablehlo.exponential %v5063 : tensor<32x197x197xf32>
    %v5066 = stablehlo.reduce(%v5065 init: %v5064) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5067 = stablehlo.broadcast_in_dim %v5066, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5068 = stablehlo.divide %v5065, %v5067 : tensor<32x197x197xf32>
    %v5069 = stablehlo.reshape %v5068 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5070 = stablehlo.reshape %v5069 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5071 = stablehlo.reshape %v5049 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5072 = stablehlo.dot_general %v5070, %v5071, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5073 = stablehlo.reshape %v5072 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5074 = stablehlo.reshape %v5073 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5075 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5076 = stablehlo.pad %v5074, %v5075, low = [0, 0, 64], high = [0, 0, 640], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v5077 = stablehlo.reshape %v5076 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5078 = stablehlo.add %v5040, %v5077 : tensor<32x151296xf32>
    %v5079 = stablehlo.reshape %v4993 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5080 = stablehlo.slice %v5079 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5081 = stablehlo.reshape %v5080 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5082 = stablehlo.reshape %v4998 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5083 = stablehlo.slice %v5082 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5084 = stablehlo.reshape %v5083 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5085 = stablehlo.reshape %v5003 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5086 = stablehlo.slice %v5085 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5087 = stablehlo.reshape %v5086 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5088 = stablehlo.reshape %v5084 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5089 = stablehlo.transpose %v5088, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5090 = stablehlo.reshape %v5089 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5091 = stablehlo.reshape %v5081 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5092 = stablehlo.reshape %v5090 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5093 = stablehlo.dot_general %v5091, %v5092, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v5094 = stablehlo.reshape %v5093 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5095 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v5096 = stablehlo.multiply %v5094, %v5095 : tensor<32x38809xf32>
    %v5097 = stablehlo.reshape %v5096 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5098 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v5099 = stablehlo.reduce(%v5097 init: %v5098) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5100 = stablehlo.broadcast_in_dim %v5099, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5101 = stablehlo.subtract %v5097, %v5100 : tensor<32x197x197xf32>
    %v5102 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5103 = stablehlo.exponential %v5101 : tensor<32x197x197xf32>
    %v5104 = stablehlo.reduce(%v5103 init: %v5102) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5105 = stablehlo.broadcast_in_dim %v5104, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5106 = stablehlo.divide %v5103, %v5105 : tensor<32x197x197xf32>
    %v5107 = stablehlo.reshape %v5106 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5108 = stablehlo.reshape %v5107 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5109 = stablehlo.reshape %v5087 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5110 = stablehlo.dot_general %v5108, %v5109, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5111 = stablehlo.reshape %v5110 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5112 = stablehlo.reshape %v5111 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5113 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5114 = stablehlo.pad %v5112, %v5113, low = [0, 0, 128], high = [0, 0, 576], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v5115 = stablehlo.reshape %v5114 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5116 = stablehlo.add %v5078, %v5115 : tensor<32x151296xf32>
    %v5117 = stablehlo.reshape %v4993 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5118 = stablehlo.slice %v5117 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5119 = stablehlo.reshape %v5118 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5120 = stablehlo.reshape %v4998 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5121 = stablehlo.slice %v5120 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5122 = stablehlo.reshape %v5121 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5123 = stablehlo.reshape %v5003 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5124 = stablehlo.slice %v5123 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5125 = stablehlo.reshape %v5124 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5126 = stablehlo.reshape %v5122 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5127 = stablehlo.transpose %v5126, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5128 = stablehlo.reshape %v5127 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5129 = stablehlo.reshape %v5119 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5130 = stablehlo.reshape %v5128 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5131 = stablehlo.dot_general %v5129, %v5130, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v5132 = stablehlo.reshape %v5131 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5133 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v5134 = stablehlo.multiply %v5132, %v5133 : tensor<32x38809xf32>
    %v5135 = stablehlo.reshape %v5134 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5136 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v5137 = stablehlo.reduce(%v5135 init: %v5136) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5138 = stablehlo.broadcast_in_dim %v5137, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5139 = stablehlo.subtract %v5135, %v5138 : tensor<32x197x197xf32>
    %v5140 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5141 = stablehlo.exponential %v5139 : tensor<32x197x197xf32>
    %v5142 = stablehlo.reduce(%v5141 init: %v5140) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5143 = stablehlo.broadcast_in_dim %v5142, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5144 = stablehlo.divide %v5141, %v5143 : tensor<32x197x197xf32>
    %v5145 = stablehlo.reshape %v5144 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5146 = stablehlo.reshape %v5145 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5147 = stablehlo.reshape %v5125 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5148 = stablehlo.dot_general %v5146, %v5147, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5149 = stablehlo.reshape %v5148 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5150 = stablehlo.reshape %v5149 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5151 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5152 = stablehlo.pad %v5150, %v5151, low = [0, 0, 192], high = [0, 0, 512], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v5153 = stablehlo.reshape %v5152 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5154 = stablehlo.add %v5116, %v5153 : tensor<32x151296xf32>
    %v5155 = stablehlo.reshape %v4993 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5156 = stablehlo.slice %v5155 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5157 = stablehlo.reshape %v5156 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5158 = stablehlo.reshape %v4998 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5159 = stablehlo.slice %v5158 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5160 = stablehlo.reshape %v5159 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5161 = stablehlo.reshape %v5003 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5162 = stablehlo.slice %v5161 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5163 = stablehlo.reshape %v5162 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5164 = stablehlo.reshape %v5160 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5165 = stablehlo.transpose %v5164, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5166 = stablehlo.reshape %v5165 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5167 = stablehlo.reshape %v5157 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5168 = stablehlo.reshape %v5166 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5169 = stablehlo.dot_general %v5167, %v5168, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v5170 = stablehlo.reshape %v5169 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5171 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v5172 = stablehlo.multiply %v5170, %v5171 : tensor<32x38809xf32>
    %v5173 = stablehlo.reshape %v5172 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5174 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v5175 = stablehlo.reduce(%v5173 init: %v5174) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5176 = stablehlo.broadcast_in_dim %v5175, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5177 = stablehlo.subtract %v5173, %v5176 : tensor<32x197x197xf32>
    %v5178 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5179 = stablehlo.exponential %v5177 : tensor<32x197x197xf32>
    %v5180 = stablehlo.reduce(%v5179 init: %v5178) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5181 = stablehlo.broadcast_in_dim %v5180, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5182 = stablehlo.divide %v5179, %v5181 : tensor<32x197x197xf32>
    %v5183 = stablehlo.reshape %v5182 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5184 = stablehlo.reshape %v5183 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5185 = stablehlo.reshape %v5163 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5186 = stablehlo.dot_general %v5184, %v5185, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5187 = stablehlo.reshape %v5186 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5188 = stablehlo.reshape %v5187 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5189 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5190 = stablehlo.pad %v5188, %v5189, low = [0, 0, 256], high = [0, 0, 448], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v5191 = stablehlo.reshape %v5190 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5192 = stablehlo.add %v5154, %v5191 : tensor<32x151296xf32>
    %v5193 = stablehlo.reshape %v4993 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5194 = stablehlo.slice %v5193 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5195 = stablehlo.reshape %v5194 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5196 = stablehlo.reshape %v4998 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5197 = stablehlo.slice %v5196 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5198 = stablehlo.reshape %v5197 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5199 = stablehlo.reshape %v5003 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5200 = stablehlo.slice %v5199 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5201 = stablehlo.reshape %v5200 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5202 = stablehlo.reshape %v5198 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5203 = stablehlo.transpose %v5202, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5204 = stablehlo.reshape %v5203 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5205 = stablehlo.reshape %v5195 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5206 = stablehlo.reshape %v5204 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5207 = stablehlo.dot_general %v5205, %v5206, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v5208 = stablehlo.reshape %v5207 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5209 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v5210 = stablehlo.multiply %v5208, %v5209 : tensor<32x38809xf32>
    %v5211 = stablehlo.reshape %v5210 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5212 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v5213 = stablehlo.reduce(%v5211 init: %v5212) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5214 = stablehlo.broadcast_in_dim %v5213, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5215 = stablehlo.subtract %v5211, %v5214 : tensor<32x197x197xf32>
    %v5216 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5217 = stablehlo.exponential %v5215 : tensor<32x197x197xf32>
    %v5218 = stablehlo.reduce(%v5217 init: %v5216) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5219 = stablehlo.broadcast_in_dim %v5218, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5220 = stablehlo.divide %v5217, %v5219 : tensor<32x197x197xf32>
    %v5221 = stablehlo.reshape %v5220 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5222 = stablehlo.reshape %v5221 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5223 = stablehlo.reshape %v5201 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5224 = stablehlo.dot_general %v5222, %v5223, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5225 = stablehlo.reshape %v5224 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5226 = stablehlo.reshape %v5225 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5227 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5228 = stablehlo.pad %v5226, %v5227, low = [0, 0, 320], high = [0, 0, 384], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v5229 = stablehlo.reshape %v5228 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5230 = stablehlo.add %v5192, %v5229 : tensor<32x151296xf32>
    %v5231 = stablehlo.reshape %v4993 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5232 = stablehlo.slice %v5231 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5233 = stablehlo.reshape %v5232 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5234 = stablehlo.reshape %v4998 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5235 = stablehlo.slice %v5234 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5236 = stablehlo.reshape %v5235 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5237 = stablehlo.reshape %v5003 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5238 = stablehlo.slice %v5237 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5239 = stablehlo.reshape %v5238 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5240 = stablehlo.reshape %v5236 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5241 = stablehlo.transpose %v5240, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5242 = stablehlo.reshape %v5241 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5243 = stablehlo.reshape %v5233 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5244 = stablehlo.reshape %v5242 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5245 = stablehlo.dot_general %v5243, %v5244, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v5246 = stablehlo.reshape %v5245 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5247 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v5248 = stablehlo.multiply %v5246, %v5247 : tensor<32x38809xf32>
    %v5249 = stablehlo.reshape %v5248 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5250 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v5251 = stablehlo.reduce(%v5249 init: %v5250) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5252 = stablehlo.broadcast_in_dim %v5251, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5253 = stablehlo.subtract %v5249, %v5252 : tensor<32x197x197xf32>
    %v5254 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5255 = stablehlo.exponential %v5253 : tensor<32x197x197xf32>
    %v5256 = stablehlo.reduce(%v5255 init: %v5254) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5257 = stablehlo.broadcast_in_dim %v5256, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5258 = stablehlo.divide %v5255, %v5257 : tensor<32x197x197xf32>
    %v5259 = stablehlo.reshape %v5258 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5260 = stablehlo.reshape %v5259 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5261 = stablehlo.reshape %v5239 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5262 = stablehlo.dot_general %v5260, %v5261, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5263 = stablehlo.reshape %v5262 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5264 = stablehlo.reshape %v5263 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5265 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5266 = stablehlo.pad %v5264, %v5265, low = [0, 0, 384], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v5267 = stablehlo.reshape %v5266 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5268 = stablehlo.add %v5230, %v5267 : tensor<32x151296xf32>
    %v5269 = stablehlo.reshape %v4993 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5270 = stablehlo.slice %v5269 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5271 = stablehlo.reshape %v5270 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5272 = stablehlo.reshape %v4998 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5273 = stablehlo.slice %v5272 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5274 = stablehlo.reshape %v5273 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5275 = stablehlo.reshape %v5003 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5276 = stablehlo.slice %v5275 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5277 = stablehlo.reshape %v5276 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5278 = stablehlo.reshape %v5274 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5279 = stablehlo.transpose %v5278, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5280 = stablehlo.reshape %v5279 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5281 = stablehlo.reshape %v5271 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5282 = stablehlo.reshape %v5280 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5283 = stablehlo.dot_general %v5281, %v5282, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v5284 = stablehlo.reshape %v5283 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5285 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v5286 = stablehlo.multiply %v5284, %v5285 : tensor<32x38809xf32>
    %v5287 = stablehlo.reshape %v5286 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5288 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v5289 = stablehlo.reduce(%v5287 init: %v5288) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5290 = stablehlo.broadcast_in_dim %v5289, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5291 = stablehlo.subtract %v5287, %v5290 : tensor<32x197x197xf32>
    %v5292 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5293 = stablehlo.exponential %v5291 : tensor<32x197x197xf32>
    %v5294 = stablehlo.reduce(%v5293 init: %v5292) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5295 = stablehlo.broadcast_in_dim %v5294, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5296 = stablehlo.divide %v5293, %v5295 : tensor<32x197x197xf32>
    %v5297 = stablehlo.reshape %v5296 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5298 = stablehlo.reshape %v5297 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5299 = stablehlo.reshape %v5277 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5300 = stablehlo.dot_general %v5298, %v5299, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5301 = stablehlo.reshape %v5300 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5302 = stablehlo.reshape %v5301 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5303 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5304 = stablehlo.pad %v5302, %v5303, low = [0, 0, 448], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v5305 = stablehlo.reshape %v5304 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5306 = stablehlo.add %v5268, %v5305 : tensor<32x151296xf32>
    %v5307 = stablehlo.reshape %v4993 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5308 = stablehlo.slice %v5307 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5309 = stablehlo.reshape %v5308 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5310 = stablehlo.reshape %v4998 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5311 = stablehlo.slice %v5310 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5312 = stablehlo.reshape %v5311 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5313 = stablehlo.reshape %v5003 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5314 = stablehlo.slice %v5313 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5315 = stablehlo.reshape %v5314 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5316 = stablehlo.reshape %v5312 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5317 = stablehlo.transpose %v5316, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5318 = stablehlo.reshape %v5317 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5319 = stablehlo.reshape %v5309 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5320 = stablehlo.reshape %v5318 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5321 = stablehlo.dot_general %v5319, %v5320, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v5322 = stablehlo.reshape %v5321 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5323 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v5324 = stablehlo.multiply %v5322, %v5323 : tensor<32x38809xf32>
    %v5325 = stablehlo.reshape %v5324 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5326 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v5327 = stablehlo.reduce(%v5325 init: %v5326) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5328 = stablehlo.broadcast_in_dim %v5327, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5329 = stablehlo.subtract %v5325, %v5328 : tensor<32x197x197xf32>
    %v5330 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5331 = stablehlo.exponential %v5329 : tensor<32x197x197xf32>
    %v5332 = stablehlo.reduce(%v5331 init: %v5330) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5333 = stablehlo.broadcast_in_dim %v5332, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5334 = stablehlo.divide %v5331, %v5333 : tensor<32x197x197xf32>
    %v5335 = stablehlo.reshape %v5334 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5336 = stablehlo.reshape %v5335 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5337 = stablehlo.reshape %v5315 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5338 = stablehlo.dot_general %v5336, %v5337, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5339 = stablehlo.reshape %v5338 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5340 = stablehlo.reshape %v5339 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5341 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5342 = stablehlo.pad %v5340, %v5341, low = [0, 0, 512], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v5343 = stablehlo.reshape %v5342 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5344 = stablehlo.add %v5306, %v5343 : tensor<32x151296xf32>
    %v5345 = stablehlo.reshape %v4993 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5346 = stablehlo.slice %v5345 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5347 = stablehlo.reshape %v5346 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5348 = stablehlo.reshape %v4998 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5349 = stablehlo.slice %v5348 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5350 = stablehlo.reshape %v5349 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5351 = stablehlo.reshape %v5003 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5352 = stablehlo.slice %v5351 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
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
    %v5364 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v5365 = stablehlo.reduce(%v5363 init: %v5364) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5366 = stablehlo.broadcast_in_dim %v5365, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5367 = stablehlo.subtract %v5363, %v5366 : tensor<32x197x197xf32>
    %v5368 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5369 = stablehlo.exponential %v5367 : tensor<32x197x197xf32>
    %v5370 = stablehlo.reduce(%v5369 init: %v5368) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5371 = stablehlo.broadcast_in_dim %v5370, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5372 = stablehlo.divide %v5369, %v5371 : tensor<32x197x197xf32>
    %v5373 = stablehlo.reshape %v5372 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5374 = stablehlo.reshape %v5373 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5375 = stablehlo.reshape %v5353 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5376 = stablehlo.dot_general %v5374, %v5375, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5377 = stablehlo.reshape %v5376 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5378 = stablehlo.reshape %v5377 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5379 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5380 = stablehlo.pad %v5378, %v5379, low = [0, 0, 576], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v5381 = stablehlo.reshape %v5380 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5382 = stablehlo.add %v5344, %v5381 : tensor<32x151296xf32>
    %v5383 = stablehlo.reshape %v4993 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5384 = stablehlo.slice %v5383 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5385 = stablehlo.reshape %v5384 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5386 = stablehlo.reshape %v4998 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5387 = stablehlo.slice %v5386 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5388 = stablehlo.reshape %v5387 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5389 = stablehlo.reshape %v5003 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5390 = stablehlo.slice %v5389 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5391 = stablehlo.reshape %v5390 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5392 = stablehlo.reshape %v5388 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5393 = stablehlo.transpose %v5392, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5394 = stablehlo.reshape %v5393 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5395 = stablehlo.reshape %v5385 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5396 = stablehlo.reshape %v5394 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5397 = stablehlo.dot_general %v5395, %v5396, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v5398 = stablehlo.reshape %v5397 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5399 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v5400 = stablehlo.multiply %v5398, %v5399 : tensor<32x38809xf32>
    %v5401 = stablehlo.reshape %v5400 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5402 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v5403 = stablehlo.reduce(%v5401 init: %v5402) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5404 = stablehlo.broadcast_in_dim %v5403, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5405 = stablehlo.subtract %v5401, %v5404 : tensor<32x197x197xf32>
    %v5406 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5407 = stablehlo.exponential %v5405 : tensor<32x197x197xf32>
    %v5408 = stablehlo.reduce(%v5407 init: %v5406) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5409 = stablehlo.broadcast_in_dim %v5408, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5410 = stablehlo.divide %v5407, %v5409 : tensor<32x197x197xf32>
    %v5411 = stablehlo.reshape %v5410 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5412 = stablehlo.reshape %v5411 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5413 = stablehlo.reshape %v5391 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5414 = stablehlo.dot_general %v5412, %v5413, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5415 = stablehlo.reshape %v5414 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5416 = stablehlo.reshape %v5415 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5417 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5418 = stablehlo.pad %v5416, %v5417, low = [0, 0, 640], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v5419 = stablehlo.reshape %v5418 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5420 = stablehlo.add %v5382, %v5419 : tensor<32x151296xf32>
    %v5421 = stablehlo.reshape %v4993 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5422 = stablehlo.slice %v5421 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5423 = stablehlo.reshape %v5422 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5424 = stablehlo.reshape %v4998 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5425 = stablehlo.slice %v5424 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5426 = stablehlo.reshape %v5425 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5427 = stablehlo.reshape %v5003 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5428 = stablehlo.slice %v5427 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5429 = stablehlo.reshape %v5428 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5430 = stablehlo.reshape %v5426 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5431 = stablehlo.transpose %v5430, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5432 = stablehlo.reshape %v5431 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5433 = stablehlo.reshape %v5423 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5434 = stablehlo.reshape %v5432 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5435 = stablehlo.dot_general %v5433, %v5434, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v5436 = stablehlo.reshape %v5435 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5437 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v5438 = stablehlo.multiply %v5436, %v5437 : tensor<32x38809xf32>
    %v5439 = stablehlo.reshape %v5438 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5440 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v5441 = stablehlo.reduce(%v5439 init: %v5440) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5442 = stablehlo.broadcast_in_dim %v5441, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5443 = stablehlo.subtract %v5439, %v5442 : tensor<32x197x197xf32>
    %v5444 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5445 = stablehlo.exponential %v5443 : tensor<32x197x197xf32>
    %v5446 = stablehlo.reduce(%v5445 init: %v5444) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5447 = stablehlo.broadcast_in_dim %v5446, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5448 = stablehlo.divide %v5445, %v5447 : tensor<32x197x197xf32>
    %v5449 = stablehlo.reshape %v5448 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5450 = stablehlo.reshape %v5449 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5451 = stablehlo.reshape %v5429 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5452 = stablehlo.dot_general %v5450, %v5451, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5453 = stablehlo.reshape %v5452 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5454 = stablehlo.reshape %v5453 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5455 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5456 = stablehlo.pad %v5454, %v5455, low = [0, 0, 704], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v5457 = stablehlo.reshape %v5456 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5458 = stablehlo.add %v5420, %v5457 : tensor<32x151296xf32>
    %v5459 = stablehlo.reshape %v5458 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5460 = stablehlo.dot_general %v5459, %b9_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v5461 = stablehlo.broadcast_in_dim %b9_bo, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v5462 = stablehlo.add %v5460, %v5461 : tensor<32x197x768xf32>
    %v5463 = stablehlo.reshape %v5462 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5464 = stablehlo.add %v4960, %v5463 : tensor<32x151296xf32>
    %v5465 = stablehlo.reshape %v5464 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5466 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5467 = stablehlo.constant dense<768.0> : tensor<32x197x768xf32>
    %v5468 = stablehlo.constant dense<1.0e-6> : tensor<32x197x768xf32>
    %v5469 = stablehlo.reduce(%v5465 init: %v5466) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5470 = stablehlo.broadcast_in_dim %v5469, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v5471 = stablehlo.divide %v5470, %v5467 : tensor<32x197x768xf32>
    %v5472 = stablehlo.subtract %v5465, %v5471 : tensor<32x197x768xf32>
    %v5473 = stablehlo.multiply %v5472, %v5472 : tensor<32x197x768xf32>
    %v5474 = stablehlo.reduce(%v5473 init: %v5466) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5475 = stablehlo.broadcast_in_dim %v5474, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v5476 = stablehlo.divide %v5475, %v5467 : tensor<32x197x768xf32>
    %v5477 = stablehlo.add %v5476, %v5468 : tensor<32x197x768xf32>
    %v5478 = stablehlo.rsqrt %v5477 : tensor<32x197x768xf32>
    %v5479 = stablehlo.multiply %v5472, %v5478 : tensor<32x197x768xf32>
    %v5480 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v5481 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v5482 = stablehlo.multiply %v5479, %v5480 : tensor<32x197x768xf32>
    %v5483 = stablehlo.add %v5482, %v5481 : tensor<32x197x768xf32>
    %v5484 = stablehlo.reshape %v5483 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5485 = stablehlo.reshape %v5484 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5486 = stablehlo.broadcast_in_dim %b9_g2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v5487 = stablehlo.multiply %v5485, %v5486 : tensor<32x197x768xf32>
    %v5488 = stablehlo.reshape %v5487 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5489 = stablehlo.reshape %v5488 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5490 = stablehlo.broadcast_in_dim %b9_bt2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v5491 = stablehlo.add %v5489, %v5490 : tensor<32x197x768xf32>
    %v5492 = stablehlo.reshape %v5491 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5493 = stablehlo.reshape %v5492 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5494 = stablehlo.dot_general %v5493, %b9_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x3072xf32>) -> tensor<32x197x3072xf32>
    %v5495 = stablehlo.broadcast_in_dim %b9_bfc1, dims = [2] : (tensor<3072xf32>) -> tensor<32x197x3072xf32>
    %v5496 = stablehlo.add %v5494, %v5495 : tensor<32x197x3072xf32>
    %v5497 = stablehlo.reshape %v5496 : (tensor<32x197x3072xf32>) -> tensor<32x605184xf32>
    %v5498 = stablehlo.constant dense<0.5> : tensor<32x605184xf32>
    %v5499 = stablehlo.multiply %v5498, %v5497 : tensor<32x605184xf32>
    %v5500 = stablehlo.negate %v5497 : tensor<32x605184xf32>
    %v5501 = stablehlo.constant dense<0.7071067811865476> : tensor<32x605184xf32>
    %v5502 = stablehlo.multiply %v5500, %v5501 : tensor<32x605184xf32>
    %v5503 = chlo.erfc %v5502 : tensor<32x605184xf32> -> tensor<32x605184xf32>
    %v5504 = stablehlo.multiply %v5499, %v5503 : tensor<32x605184xf32>
    %v5505 = stablehlo.reshape %v5504 : (tensor<32x605184xf32>) -> tensor<32x197x3072xf32>
    %v5506 = stablehlo.dot_general %v5505, %b9_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x3072xf32>, tensor<3072x768xf32>) -> tensor<32x197x768xf32>
    %v5507 = stablehlo.broadcast_in_dim %b9_bfc2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v5508 = stablehlo.add %v5506, %v5507 : tensor<32x197x768xf32>
    %v5509 = stablehlo.reshape %v5508 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5510 = stablehlo.add %v5464, %v5509 : tensor<32x151296xf32>
    %v5511 = stablehlo.reshape %v5510 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5512 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5513 = stablehlo.constant dense<768.0> : tensor<32x197x768xf32>
    %v5514 = stablehlo.constant dense<1.0e-6> : tensor<32x197x768xf32>
    %v5515 = stablehlo.reduce(%v5511 init: %v5512) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5516 = stablehlo.broadcast_in_dim %v5515, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v5517 = stablehlo.divide %v5516, %v5513 : tensor<32x197x768xf32>
    %v5518 = stablehlo.subtract %v5511, %v5517 : tensor<32x197x768xf32>
    %v5519 = stablehlo.multiply %v5518, %v5518 : tensor<32x197x768xf32>
    %v5520 = stablehlo.reduce(%v5519 init: %v5512) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5521 = stablehlo.broadcast_in_dim %v5520, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v5522 = stablehlo.divide %v5521, %v5513 : tensor<32x197x768xf32>
    %v5523 = stablehlo.add %v5522, %v5514 : tensor<32x197x768xf32>
    %v5524 = stablehlo.rsqrt %v5523 : tensor<32x197x768xf32>
    %v5525 = stablehlo.multiply %v5518, %v5524 : tensor<32x197x768xf32>
    %v5526 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v5527 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v5528 = stablehlo.multiply %v5525, %v5526 : tensor<32x197x768xf32>
    %v5529 = stablehlo.add %v5528, %v5527 : tensor<32x197x768xf32>
    %v5530 = stablehlo.reshape %v5529 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5531 = stablehlo.reshape %v5530 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5532 = stablehlo.broadcast_in_dim %b10_g1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v5533 = stablehlo.multiply %v5531, %v5532 : tensor<32x197x768xf32>
    %v5534 = stablehlo.reshape %v5533 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5535 = stablehlo.reshape %v5534 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5536 = stablehlo.broadcast_in_dim %b10_bt1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v5537 = stablehlo.add %v5535, %v5536 : tensor<32x197x768xf32>
    %v5538 = stablehlo.reshape %v5537 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5539 = stablehlo.reshape %v5538 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5540 = stablehlo.dot_general %v5539, %b10_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v5541 = stablehlo.broadcast_in_dim %b10_bq, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v5542 = stablehlo.add %v5540, %v5541 : tensor<32x197x768xf32>
    %v5543 = stablehlo.reshape %v5542 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5544 = stablehlo.reshape %v5538 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5545 = stablehlo.dot_general %v5544, %b10_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v5546 = stablehlo.broadcast_in_dim %b10_bk, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v5547 = stablehlo.add %v5545, %v5546 : tensor<32x197x768xf32>
    %v5548 = stablehlo.reshape %v5547 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5549 = stablehlo.reshape %v5538 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5550 = stablehlo.dot_general %v5549, %b10_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v5551 = stablehlo.broadcast_in_dim %b10_bv, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v5552 = stablehlo.add %v5550, %v5551 : tensor<32x197x768xf32>
    %v5553 = stablehlo.reshape %v5552 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5554 = stablehlo.reshape %v5543 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5555 = stablehlo.slice %v5554 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5556 = stablehlo.reshape %v5555 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5557 = stablehlo.reshape %v5548 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5558 = stablehlo.slice %v5557 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5559 = stablehlo.reshape %v5558 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5560 = stablehlo.reshape %v5553 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5561 = stablehlo.slice %v5560 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5562 = stablehlo.reshape %v5561 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5563 = stablehlo.reshape %v5559 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5564 = stablehlo.transpose %v5563, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5565 = stablehlo.reshape %v5564 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5566 = stablehlo.reshape %v5556 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5567 = stablehlo.reshape %v5565 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5568 = stablehlo.dot_general %v5566, %v5567, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v5569 = stablehlo.reshape %v5568 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5570 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v5571 = stablehlo.multiply %v5569, %v5570 : tensor<32x38809xf32>
    %v5572 = stablehlo.reshape %v5571 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5573 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v5574 = stablehlo.reduce(%v5572 init: %v5573) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5575 = stablehlo.broadcast_in_dim %v5574, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5576 = stablehlo.subtract %v5572, %v5575 : tensor<32x197x197xf32>
    %v5577 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5578 = stablehlo.exponential %v5576 : tensor<32x197x197xf32>
    %v5579 = stablehlo.reduce(%v5578 init: %v5577) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5580 = stablehlo.broadcast_in_dim %v5579, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5581 = stablehlo.divide %v5578, %v5580 : tensor<32x197x197xf32>
    %v5582 = stablehlo.reshape %v5581 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5583 = stablehlo.reshape %v5582 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5584 = stablehlo.reshape %v5562 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5585 = stablehlo.dot_general %v5583, %v5584, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5586 = stablehlo.reshape %v5585 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5587 = stablehlo.reshape %v5586 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5588 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5589 = stablehlo.pad %v5587, %v5588, low = [0, 0, 0], high = [0, 0, 704], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v5590 = stablehlo.reshape %v5589 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5591 = stablehlo.reshape %v5543 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5592 = stablehlo.slice %v5591 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5593 = stablehlo.reshape %v5592 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5594 = stablehlo.reshape %v5548 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5595 = stablehlo.slice %v5594 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5596 = stablehlo.reshape %v5595 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5597 = stablehlo.reshape %v5553 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5598 = stablehlo.slice %v5597 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5599 = stablehlo.reshape %v5598 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5600 = stablehlo.reshape %v5596 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5601 = stablehlo.transpose %v5600, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5602 = stablehlo.reshape %v5601 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5603 = stablehlo.reshape %v5593 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5604 = stablehlo.reshape %v5602 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5605 = stablehlo.dot_general %v5603, %v5604, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v5606 = stablehlo.reshape %v5605 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5607 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v5608 = stablehlo.multiply %v5606, %v5607 : tensor<32x38809xf32>
    %v5609 = stablehlo.reshape %v5608 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5610 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v5611 = stablehlo.reduce(%v5609 init: %v5610) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5612 = stablehlo.broadcast_in_dim %v5611, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5613 = stablehlo.subtract %v5609, %v5612 : tensor<32x197x197xf32>
    %v5614 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5615 = stablehlo.exponential %v5613 : tensor<32x197x197xf32>
    %v5616 = stablehlo.reduce(%v5615 init: %v5614) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5617 = stablehlo.broadcast_in_dim %v5616, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5618 = stablehlo.divide %v5615, %v5617 : tensor<32x197x197xf32>
    %v5619 = stablehlo.reshape %v5618 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5620 = stablehlo.reshape %v5619 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5621 = stablehlo.reshape %v5599 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5622 = stablehlo.dot_general %v5620, %v5621, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5623 = stablehlo.reshape %v5622 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5624 = stablehlo.reshape %v5623 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5625 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5626 = stablehlo.pad %v5624, %v5625, low = [0, 0, 64], high = [0, 0, 640], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v5627 = stablehlo.reshape %v5626 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5628 = stablehlo.add %v5590, %v5627 : tensor<32x151296xf32>
    %v5629 = stablehlo.reshape %v5543 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5630 = stablehlo.slice %v5629 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5631 = stablehlo.reshape %v5630 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5632 = stablehlo.reshape %v5548 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5633 = stablehlo.slice %v5632 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5634 = stablehlo.reshape %v5633 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5635 = stablehlo.reshape %v5553 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5636 = stablehlo.slice %v5635 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5637 = stablehlo.reshape %v5636 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5638 = stablehlo.reshape %v5634 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5639 = stablehlo.transpose %v5638, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5640 = stablehlo.reshape %v5639 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5641 = stablehlo.reshape %v5631 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5642 = stablehlo.reshape %v5640 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5643 = stablehlo.dot_general %v5641, %v5642, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v5644 = stablehlo.reshape %v5643 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5645 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v5646 = stablehlo.multiply %v5644, %v5645 : tensor<32x38809xf32>
    %v5647 = stablehlo.reshape %v5646 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5648 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v5649 = stablehlo.reduce(%v5647 init: %v5648) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5650 = stablehlo.broadcast_in_dim %v5649, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5651 = stablehlo.subtract %v5647, %v5650 : tensor<32x197x197xf32>
    %v5652 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5653 = stablehlo.exponential %v5651 : tensor<32x197x197xf32>
    %v5654 = stablehlo.reduce(%v5653 init: %v5652) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5655 = stablehlo.broadcast_in_dim %v5654, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5656 = stablehlo.divide %v5653, %v5655 : tensor<32x197x197xf32>
    %v5657 = stablehlo.reshape %v5656 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5658 = stablehlo.reshape %v5657 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5659 = stablehlo.reshape %v5637 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5660 = stablehlo.dot_general %v5658, %v5659, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5661 = stablehlo.reshape %v5660 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5662 = stablehlo.reshape %v5661 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5663 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5664 = stablehlo.pad %v5662, %v5663, low = [0, 0, 128], high = [0, 0, 576], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v5665 = stablehlo.reshape %v5664 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5666 = stablehlo.add %v5628, %v5665 : tensor<32x151296xf32>
    %v5667 = stablehlo.reshape %v5543 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5668 = stablehlo.slice %v5667 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5669 = stablehlo.reshape %v5668 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5670 = stablehlo.reshape %v5548 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5671 = stablehlo.slice %v5670 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5672 = stablehlo.reshape %v5671 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5673 = stablehlo.reshape %v5553 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5674 = stablehlo.slice %v5673 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5675 = stablehlo.reshape %v5674 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5676 = stablehlo.reshape %v5672 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5677 = stablehlo.transpose %v5676, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5678 = stablehlo.reshape %v5677 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5679 = stablehlo.reshape %v5669 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5680 = stablehlo.reshape %v5678 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5681 = stablehlo.dot_general %v5679, %v5680, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v5682 = stablehlo.reshape %v5681 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5683 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v5684 = stablehlo.multiply %v5682, %v5683 : tensor<32x38809xf32>
    %v5685 = stablehlo.reshape %v5684 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5686 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v5687 = stablehlo.reduce(%v5685 init: %v5686) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5688 = stablehlo.broadcast_in_dim %v5687, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5689 = stablehlo.subtract %v5685, %v5688 : tensor<32x197x197xf32>
    %v5690 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5691 = stablehlo.exponential %v5689 : tensor<32x197x197xf32>
    %v5692 = stablehlo.reduce(%v5691 init: %v5690) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5693 = stablehlo.broadcast_in_dim %v5692, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5694 = stablehlo.divide %v5691, %v5693 : tensor<32x197x197xf32>
    %v5695 = stablehlo.reshape %v5694 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5696 = stablehlo.reshape %v5695 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5697 = stablehlo.reshape %v5675 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5698 = stablehlo.dot_general %v5696, %v5697, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5699 = stablehlo.reshape %v5698 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5700 = stablehlo.reshape %v5699 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5701 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5702 = stablehlo.pad %v5700, %v5701, low = [0, 0, 192], high = [0, 0, 512], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v5703 = stablehlo.reshape %v5702 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5704 = stablehlo.add %v5666, %v5703 : tensor<32x151296xf32>
    %v5705 = stablehlo.reshape %v5543 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5706 = stablehlo.slice %v5705 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5707 = stablehlo.reshape %v5706 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5708 = stablehlo.reshape %v5548 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5709 = stablehlo.slice %v5708 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5710 = stablehlo.reshape %v5709 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5711 = stablehlo.reshape %v5553 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5712 = stablehlo.slice %v5711 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5713 = stablehlo.reshape %v5712 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5714 = stablehlo.reshape %v5710 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5715 = stablehlo.transpose %v5714, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5716 = stablehlo.reshape %v5715 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5717 = stablehlo.reshape %v5707 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5718 = stablehlo.reshape %v5716 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5719 = stablehlo.dot_general %v5717, %v5718, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v5720 = stablehlo.reshape %v5719 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5721 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v5722 = stablehlo.multiply %v5720, %v5721 : tensor<32x38809xf32>
    %v5723 = stablehlo.reshape %v5722 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5724 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v5725 = stablehlo.reduce(%v5723 init: %v5724) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5726 = stablehlo.broadcast_in_dim %v5725, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5727 = stablehlo.subtract %v5723, %v5726 : tensor<32x197x197xf32>
    %v5728 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5729 = stablehlo.exponential %v5727 : tensor<32x197x197xf32>
    %v5730 = stablehlo.reduce(%v5729 init: %v5728) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5731 = stablehlo.broadcast_in_dim %v5730, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5732 = stablehlo.divide %v5729, %v5731 : tensor<32x197x197xf32>
    %v5733 = stablehlo.reshape %v5732 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5734 = stablehlo.reshape %v5733 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5735 = stablehlo.reshape %v5713 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5736 = stablehlo.dot_general %v5734, %v5735, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5737 = stablehlo.reshape %v5736 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5738 = stablehlo.reshape %v5737 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5739 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5740 = stablehlo.pad %v5738, %v5739, low = [0, 0, 256], high = [0, 0, 448], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v5741 = stablehlo.reshape %v5740 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5742 = stablehlo.add %v5704, %v5741 : tensor<32x151296xf32>
    %v5743 = stablehlo.reshape %v5543 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5744 = stablehlo.slice %v5743 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5745 = stablehlo.reshape %v5744 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5746 = stablehlo.reshape %v5548 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5747 = stablehlo.slice %v5746 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5748 = stablehlo.reshape %v5747 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5749 = stablehlo.reshape %v5553 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5750 = stablehlo.slice %v5749 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5751 = stablehlo.reshape %v5750 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5752 = stablehlo.reshape %v5748 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5753 = stablehlo.transpose %v5752, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5754 = stablehlo.reshape %v5753 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5755 = stablehlo.reshape %v5745 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5756 = stablehlo.reshape %v5754 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5757 = stablehlo.dot_general %v5755, %v5756, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v5758 = stablehlo.reshape %v5757 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5759 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v5760 = stablehlo.multiply %v5758, %v5759 : tensor<32x38809xf32>
    %v5761 = stablehlo.reshape %v5760 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5762 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v5763 = stablehlo.reduce(%v5761 init: %v5762) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5764 = stablehlo.broadcast_in_dim %v5763, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5765 = stablehlo.subtract %v5761, %v5764 : tensor<32x197x197xf32>
    %v5766 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5767 = stablehlo.exponential %v5765 : tensor<32x197x197xf32>
    %v5768 = stablehlo.reduce(%v5767 init: %v5766) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5769 = stablehlo.broadcast_in_dim %v5768, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5770 = stablehlo.divide %v5767, %v5769 : tensor<32x197x197xf32>
    %v5771 = stablehlo.reshape %v5770 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5772 = stablehlo.reshape %v5771 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5773 = stablehlo.reshape %v5751 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5774 = stablehlo.dot_general %v5772, %v5773, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5775 = stablehlo.reshape %v5774 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5776 = stablehlo.reshape %v5775 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5777 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5778 = stablehlo.pad %v5776, %v5777, low = [0, 0, 320], high = [0, 0, 384], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v5779 = stablehlo.reshape %v5778 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5780 = stablehlo.add %v5742, %v5779 : tensor<32x151296xf32>
    %v5781 = stablehlo.reshape %v5543 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5782 = stablehlo.slice %v5781 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5783 = stablehlo.reshape %v5782 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5784 = stablehlo.reshape %v5548 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5785 = stablehlo.slice %v5784 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5786 = stablehlo.reshape %v5785 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5787 = stablehlo.reshape %v5553 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5788 = stablehlo.slice %v5787 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5789 = stablehlo.reshape %v5788 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5790 = stablehlo.reshape %v5786 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5791 = stablehlo.transpose %v5790, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5792 = stablehlo.reshape %v5791 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5793 = stablehlo.reshape %v5783 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5794 = stablehlo.reshape %v5792 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5795 = stablehlo.dot_general %v5793, %v5794, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v5796 = stablehlo.reshape %v5795 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5797 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v5798 = stablehlo.multiply %v5796, %v5797 : tensor<32x38809xf32>
    %v5799 = stablehlo.reshape %v5798 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5800 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v5801 = stablehlo.reduce(%v5799 init: %v5800) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5802 = stablehlo.broadcast_in_dim %v5801, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5803 = stablehlo.subtract %v5799, %v5802 : tensor<32x197x197xf32>
    %v5804 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5805 = stablehlo.exponential %v5803 : tensor<32x197x197xf32>
    %v5806 = stablehlo.reduce(%v5805 init: %v5804) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5807 = stablehlo.broadcast_in_dim %v5806, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5808 = stablehlo.divide %v5805, %v5807 : tensor<32x197x197xf32>
    %v5809 = stablehlo.reshape %v5808 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5810 = stablehlo.reshape %v5809 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5811 = stablehlo.reshape %v5789 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5812 = stablehlo.dot_general %v5810, %v5811, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5813 = stablehlo.reshape %v5812 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5814 = stablehlo.reshape %v5813 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5815 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5816 = stablehlo.pad %v5814, %v5815, low = [0, 0, 384], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v5817 = stablehlo.reshape %v5816 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5818 = stablehlo.add %v5780, %v5817 : tensor<32x151296xf32>
    %v5819 = stablehlo.reshape %v5543 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5820 = stablehlo.slice %v5819 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5821 = stablehlo.reshape %v5820 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5822 = stablehlo.reshape %v5548 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5823 = stablehlo.slice %v5822 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5824 = stablehlo.reshape %v5823 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5825 = stablehlo.reshape %v5553 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5826 = stablehlo.slice %v5825 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5827 = stablehlo.reshape %v5826 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5828 = stablehlo.reshape %v5824 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5829 = stablehlo.transpose %v5828, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5830 = stablehlo.reshape %v5829 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5831 = stablehlo.reshape %v5821 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5832 = stablehlo.reshape %v5830 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5833 = stablehlo.dot_general %v5831, %v5832, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v5834 = stablehlo.reshape %v5833 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5835 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v5836 = stablehlo.multiply %v5834, %v5835 : tensor<32x38809xf32>
    %v5837 = stablehlo.reshape %v5836 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5838 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v5839 = stablehlo.reduce(%v5837 init: %v5838) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5840 = stablehlo.broadcast_in_dim %v5839, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5841 = stablehlo.subtract %v5837, %v5840 : tensor<32x197x197xf32>
    %v5842 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5843 = stablehlo.exponential %v5841 : tensor<32x197x197xf32>
    %v5844 = stablehlo.reduce(%v5843 init: %v5842) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5845 = stablehlo.broadcast_in_dim %v5844, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5846 = stablehlo.divide %v5843, %v5845 : tensor<32x197x197xf32>
    %v5847 = stablehlo.reshape %v5846 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5848 = stablehlo.reshape %v5847 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5849 = stablehlo.reshape %v5827 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5850 = stablehlo.dot_general %v5848, %v5849, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5851 = stablehlo.reshape %v5850 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5852 = stablehlo.reshape %v5851 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5853 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5854 = stablehlo.pad %v5852, %v5853, low = [0, 0, 448], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v5855 = stablehlo.reshape %v5854 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5856 = stablehlo.add %v5818, %v5855 : tensor<32x151296xf32>
    %v5857 = stablehlo.reshape %v5543 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5858 = stablehlo.slice %v5857 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5859 = stablehlo.reshape %v5858 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5860 = stablehlo.reshape %v5548 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5861 = stablehlo.slice %v5860 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5862 = stablehlo.reshape %v5861 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5863 = stablehlo.reshape %v5553 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5864 = stablehlo.slice %v5863 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5865 = stablehlo.reshape %v5864 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5866 = stablehlo.reshape %v5862 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5867 = stablehlo.transpose %v5866, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5868 = stablehlo.reshape %v5867 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5869 = stablehlo.reshape %v5859 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5870 = stablehlo.reshape %v5868 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5871 = stablehlo.dot_general %v5869, %v5870, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v5872 = stablehlo.reshape %v5871 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5873 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v5874 = stablehlo.multiply %v5872, %v5873 : tensor<32x38809xf32>
    %v5875 = stablehlo.reshape %v5874 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5876 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v5877 = stablehlo.reduce(%v5875 init: %v5876) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5878 = stablehlo.broadcast_in_dim %v5877, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5879 = stablehlo.subtract %v5875, %v5878 : tensor<32x197x197xf32>
    %v5880 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5881 = stablehlo.exponential %v5879 : tensor<32x197x197xf32>
    %v5882 = stablehlo.reduce(%v5881 init: %v5880) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5883 = stablehlo.broadcast_in_dim %v5882, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5884 = stablehlo.divide %v5881, %v5883 : tensor<32x197x197xf32>
    %v5885 = stablehlo.reshape %v5884 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5886 = stablehlo.reshape %v5885 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5887 = stablehlo.reshape %v5865 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5888 = stablehlo.dot_general %v5886, %v5887, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5889 = stablehlo.reshape %v5888 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5890 = stablehlo.reshape %v5889 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5891 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5892 = stablehlo.pad %v5890, %v5891, low = [0, 0, 512], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v5893 = stablehlo.reshape %v5892 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5894 = stablehlo.add %v5856, %v5893 : tensor<32x151296xf32>
    %v5895 = stablehlo.reshape %v5543 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5896 = stablehlo.slice %v5895 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5897 = stablehlo.reshape %v5896 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5898 = stablehlo.reshape %v5548 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5899 = stablehlo.slice %v5898 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5900 = stablehlo.reshape %v5899 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5901 = stablehlo.reshape %v5553 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5902 = stablehlo.slice %v5901 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5903 = stablehlo.reshape %v5902 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5904 = stablehlo.reshape %v5900 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5905 = stablehlo.transpose %v5904, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5906 = stablehlo.reshape %v5905 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5907 = stablehlo.reshape %v5897 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5908 = stablehlo.reshape %v5906 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5909 = stablehlo.dot_general %v5907, %v5908, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v5910 = stablehlo.reshape %v5909 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5911 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v5912 = stablehlo.multiply %v5910, %v5911 : tensor<32x38809xf32>
    %v5913 = stablehlo.reshape %v5912 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5914 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v5915 = stablehlo.reduce(%v5913 init: %v5914) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5916 = stablehlo.broadcast_in_dim %v5915, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5917 = stablehlo.subtract %v5913, %v5916 : tensor<32x197x197xf32>
    %v5918 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5919 = stablehlo.exponential %v5917 : tensor<32x197x197xf32>
    %v5920 = stablehlo.reduce(%v5919 init: %v5918) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5921 = stablehlo.broadcast_in_dim %v5920, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5922 = stablehlo.divide %v5919, %v5921 : tensor<32x197x197xf32>
    %v5923 = stablehlo.reshape %v5922 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5924 = stablehlo.reshape %v5923 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5925 = stablehlo.reshape %v5903 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5926 = stablehlo.dot_general %v5924, %v5925, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5927 = stablehlo.reshape %v5926 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5928 = stablehlo.reshape %v5927 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5929 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5930 = stablehlo.pad %v5928, %v5929, low = [0, 0, 576], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v5931 = stablehlo.reshape %v5930 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5932 = stablehlo.add %v5894, %v5931 : tensor<32x151296xf32>
    %v5933 = stablehlo.reshape %v5543 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5934 = stablehlo.slice %v5933 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5935 = stablehlo.reshape %v5934 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5936 = stablehlo.reshape %v5548 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5937 = stablehlo.slice %v5936 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5938 = stablehlo.reshape %v5937 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5939 = stablehlo.reshape %v5553 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5940 = stablehlo.slice %v5939 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5941 = stablehlo.reshape %v5940 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5942 = stablehlo.reshape %v5938 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5943 = stablehlo.transpose %v5942, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5944 = stablehlo.reshape %v5943 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5945 = stablehlo.reshape %v5935 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5946 = stablehlo.reshape %v5944 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5947 = stablehlo.dot_general %v5945, %v5946, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v5948 = stablehlo.reshape %v5947 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5949 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v5950 = stablehlo.multiply %v5948, %v5949 : tensor<32x38809xf32>
    %v5951 = stablehlo.reshape %v5950 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5952 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v5953 = stablehlo.reduce(%v5951 init: %v5952) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5954 = stablehlo.broadcast_in_dim %v5953, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5955 = stablehlo.subtract %v5951, %v5954 : tensor<32x197x197xf32>
    %v5956 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5957 = stablehlo.exponential %v5955 : tensor<32x197x197xf32>
    %v5958 = stablehlo.reduce(%v5957 init: %v5956) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5959 = stablehlo.broadcast_in_dim %v5958, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5960 = stablehlo.divide %v5957, %v5959 : tensor<32x197x197xf32>
    %v5961 = stablehlo.reshape %v5960 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5962 = stablehlo.reshape %v5961 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5963 = stablehlo.reshape %v5941 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5964 = stablehlo.dot_general %v5962, %v5963, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5965 = stablehlo.reshape %v5964 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5966 = stablehlo.reshape %v5965 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5967 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5968 = stablehlo.pad %v5966, %v5967, low = [0, 0, 640], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v5969 = stablehlo.reshape %v5968 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5970 = stablehlo.add %v5932, %v5969 : tensor<32x151296xf32>
    %v5971 = stablehlo.reshape %v5543 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5972 = stablehlo.slice %v5971 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5973 = stablehlo.reshape %v5972 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5974 = stablehlo.reshape %v5548 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5975 = stablehlo.slice %v5974 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5976 = stablehlo.reshape %v5975 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5977 = stablehlo.reshape %v5553 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5978 = stablehlo.slice %v5977 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5979 = stablehlo.reshape %v5978 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5980 = stablehlo.reshape %v5976 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5981 = stablehlo.transpose %v5980, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5982 = stablehlo.reshape %v5981 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5983 = stablehlo.reshape %v5973 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5984 = stablehlo.reshape %v5982 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5985 = stablehlo.dot_general %v5983, %v5984, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v5986 = stablehlo.reshape %v5985 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5987 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v5988 = stablehlo.multiply %v5986, %v5987 : tensor<32x38809xf32>
    %v5989 = stablehlo.reshape %v5988 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5990 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v5991 = stablehlo.reduce(%v5989 init: %v5990) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5992 = stablehlo.broadcast_in_dim %v5991, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5993 = stablehlo.subtract %v5989, %v5992 : tensor<32x197x197xf32>
    %v5994 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5995 = stablehlo.exponential %v5993 : tensor<32x197x197xf32>
    %v5996 = stablehlo.reduce(%v5995 init: %v5994) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5997 = stablehlo.broadcast_in_dim %v5996, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5998 = stablehlo.divide %v5995, %v5997 : tensor<32x197x197xf32>
    %v5999 = stablehlo.reshape %v5998 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v6000 = stablehlo.reshape %v5999 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v6001 = stablehlo.reshape %v5979 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6002 = stablehlo.dot_general %v6000, %v6001, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v6003 = stablehlo.reshape %v6002 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6004 = stablehlo.reshape %v6003 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6005 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6006 = stablehlo.pad %v6004, %v6005, low = [0, 0, 704], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v6007 = stablehlo.reshape %v6006 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v6008 = stablehlo.add %v5970, %v6007 : tensor<32x151296xf32>
    %v6009 = stablehlo.reshape %v6008 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6010 = stablehlo.dot_general %v6009, %b10_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v6011 = stablehlo.broadcast_in_dim %b10_bo, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v6012 = stablehlo.add %v6010, %v6011 : tensor<32x197x768xf32>
    %v6013 = stablehlo.reshape %v6012 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v6014 = stablehlo.add %v5510, %v6013 : tensor<32x151296xf32>
    %v6015 = stablehlo.reshape %v6014 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6016 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6017 = stablehlo.constant dense<768.0> : tensor<32x197x768xf32>
    %v6018 = stablehlo.constant dense<1.0e-6> : tensor<32x197x768xf32>
    %v6019 = stablehlo.reduce(%v6015 init: %v6016) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6020 = stablehlo.broadcast_in_dim %v6019, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v6021 = stablehlo.divide %v6020, %v6017 : tensor<32x197x768xf32>
    %v6022 = stablehlo.subtract %v6015, %v6021 : tensor<32x197x768xf32>
    %v6023 = stablehlo.multiply %v6022, %v6022 : tensor<32x197x768xf32>
    %v6024 = stablehlo.reduce(%v6023 init: %v6016) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6025 = stablehlo.broadcast_in_dim %v6024, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v6026 = stablehlo.divide %v6025, %v6017 : tensor<32x197x768xf32>
    %v6027 = stablehlo.add %v6026, %v6018 : tensor<32x197x768xf32>
    %v6028 = stablehlo.rsqrt %v6027 : tensor<32x197x768xf32>
    %v6029 = stablehlo.multiply %v6022, %v6028 : tensor<32x197x768xf32>
    %v6030 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v6031 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v6032 = stablehlo.multiply %v6029, %v6030 : tensor<32x197x768xf32>
    %v6033 = stablehlo.add %v6032, %v6031 : tensor<32x197x768xf32>
    %v6034 = stablehlo.reshape %v6033 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v6035 = stablehlo.reshape %v6034 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6036 = stablehlo.broadcast_in_dim %b10_g2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v6037 = stablehlo.multiply %v6035, %v6036 : tensor<32x197x768xf32>
    %v6038 = stablehlo.reshape %v6037 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v6039 = stablehlo.reshape %v6038 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6040 = stablehlo.broadcast_in_dim %b10_bt2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v6041 = stablehlo.add %v6039, %v6040 : tensor<32x197x768xf32>
    %v6042 = stablehlo.reshape %v6041 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v6043 = stablehlo.reshape %v6042 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6044 = stablehlo.dot_general %v6043, %b10_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x3072xf32>) -> tensor<32x197x3072xf32>
    %v6045 = stablehlo.broadcast_in_dim %b10_bfc1, dims = [2] : (tensor<3072xf32>) -> tensor<32x197x3072xf32>
    %v6046 = stablehlo.add %v6044, %v6045 : tensor<32x197x3072xf32>
    %v6047 = stablehlo.reshape %v6046 : (tensor<32x197x3072xf32>) -> tensor<32x605184xf32>
    %v6048 = stablehlo.constant dense<0.5> : tensor<32x605184xf32>
    %v6049 = stablehlo.multiply %v6048, %v6047 : tensor<32x605184xf32>
    %v6050 = stablehlo.negate %v6047 : tensor<32x605184xf32>
    %v6051 = stablehlo.constant dense<0.7071067811865476> : tensor<32x605184xf32>
    %v6052 = stablehlo.multiply %v6050, %v6051 : tensor<32x605184xf32>
    %v6053 = chlo.erfc %v6052 : tensor<32x605184xf32> -> tensor<32x605184xf32>
    %v6054 = stablehlo.multiply %v6049, %v6053 : tensor<32x605184xf32>
    %v6055 = stablehlo.reshape %v6054 : (tensor<32x605184xf32>) -> tensor<32x197x3072xf32>
    %v6056 = stablehlo.dot_general %v6055, %b10_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x3072xf32>, tensor<3072x768xf32>) -> tensor<32x197x768xf32>
    %v6057 = stablehlo.broadcast_in_dim %b10_bfc2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v6058 = stablehlo.add %v6056, %v6057 : tensor<32x197x768xf32>
    %v6059 = stablehlo.reshape %v6058 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v6060 = stablehlo.add %v6014, %v6059 : tensor<32x151296xf32>
    %v6061 = stablehlo.reshape %v6060 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6062 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6063 = stablehlo.constant dense<768.0> : tensor<32x197x768xf32>
    %v6064 = stablehlo.constant dense<1.0e-6> : tensor<32x197x768xf32>
    %v6065 = stablehlo.reduce(%v6061 init: %v6062) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6066 = stablehlo.broadcast_in_dim %v6065, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v6067 = stablehlo.divide %v6066, %v6063 : tensor<32x197x768xf32>
    %v6068 = stablehlo.subtract %v6061, %v6067 : tensor<32x197x768xf32>
    %v6069 = stablehlo.multiply %v6068, %v6068 : tensor<32x197x768xf32>
    %v6070 = stablehlo.reduce(%v6069 init: %v6062) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6071 = stablehlo.broadcast_in_dim %v6070, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v6072 = stablehlo.divide %v6071, %v6063 : tensor<32x197x768xf32>
    %v6073 = stablehlo.add %v6072, %v6064 : tensor<32x197x768xf32>
    %v6074 = stablehlo.rsqrt %v6073 : tensor<32x197x768xf32>
    %v6075 = stablehlo.multiply %v6068, %v6074 : tensor<32x197x768xf32>
    %v6076 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v6077 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v6078 = stablehlo.multiply %v6075, %v6076 : tensor<32x197x768xf32>
    %v6079 = stablehlo.add %v6078, %v6077 : tensor<32x197x768xf32>
    %v6080 = stablehlo.reshape %v6079 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v6081 = stablehlo.reshape %v6080 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6082 = stablehlo.broadcast_in_dim %b11_g1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v6083 = stablehlo.multiply %v6081, %v6082 : tensor<32x197x768xf32>
    %v6084 = stablehlo.reshape %v6083 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v6085 = stablehlo.reshape %v6084 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6086 = stablehlo.broadcast_in_dim %b11_bt1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v6087 = stablehlo.add %v6085, %v6086 : tensor<32x197x768xf32>
    %v6088 = stablehlo.reshape %v6087 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v6089 = stablehlo.reshape %v6088 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6090 = stablehlo.dot_general %v6089, %b11_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v6091 = stablehlo.broadcast_in_dim %b11_bq, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v6092 = stablehlo.add %v6090, %v6091 : tensor<32x197x768xf32>
    %v6093 = stablehlo.reshape %v6092 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v6094 = stablehlo.reshape %v6088 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6095 = stablehlo.dot_general %v6094, %b11_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v6096 = stablehlo.broadcast_in_dim %b11_bk, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v6097 = stablehlo.add %v6095, %v6096 : tensor<32x197x768xf32>
    %v6098 = stablehlo.reshape %v6097 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v6099 = stablehlo.reshape %v6088 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6100 = stablehlo.dot_general %v6099, %b11_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v6101 = stablehlo.broadcast_in_dim %b11_bv, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v6102 = stablehlo.add %v6100, %v6101 : tensor<32x197x768xf32>
    %v6103 = stablehlo.reshape %v6102 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v6104 = stablehlo.reshape %v6093 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6105 = stablehlo.slice %v6104 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v6106 = stablehlo.reshape %v6105 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6107 = stablehlo.reshape %v6098 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6108 = stablehlo.slice %v6107 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v6109 = stablehlo.reshape %v6108 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6110 = stablehlo.reshape %v6103 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6111 = stablehlo.slice %v6110 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v6112 = stablehlo.reshape %v6111 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6113 = stablehlo.reshape %v6109 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6114 = stablehlo.transpose %v6113, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v6115 = stablehlo.reshape %v6114 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v6116 = stablehlo.reshape %v6106 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6117 = stablehlo.reshape %v6115 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v6118 = stablehlo.dot_general %v6116, %v6117, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v6119 = stablehlo.reshape %v6118 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v6120 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v6121 = stablehlo.multiply %v6119, %v6120 : tensor<32x38809xf32>
    %v6122 = stablehlo.reshape %v6121 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v6123 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v6124 = stablehlo.reduce(%v6122 init: %v6123) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6125 = stablehlo.broadcast_in_dim %v6124, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v6126 = stablehlo.subtract %v6122, %v6125 : tensor<32x197x197xf32>
    %v6127 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6128 = stablehlo.exponential %v6126 : tensor<32x197x197xf32>
    %v6129 = stablehlo.reduce(%v6128 init: %v6127) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6130 = stablehlo.broadcast_in_dim %v6129, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v6131 = stablehlo.divide %v6128, %v6130 : tensor<32x197x197xf32>
    %v6132 = stablehlo.reshape %v6131 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v6133 = stablehlo.reshape %v6132 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v6134 = stablehlo.reshape %v6112 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6135 = stablehlo.dot_general %v6133, %v6134, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v6136 = stablehlo.reshape %v6135 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6137 = stablehlo.reshape %v6136 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6138 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6139 = stablehlo.pad %v6137, %v6138, low = [0, 0, 0], high = [0, 0, 704], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v6140 = stablehlo.reshape %v6139 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v6141 = stablehlo.reshape %v6093 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6142 = stablehlo.slice %v6141 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v6143 = stablehlo.reshape %v6142 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6144 = stablehlo.reshape %v6098 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6145 = stablehlo.slice %v6144 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v6146 = stablehlo.reshape %v6145 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6147 = stablehlo.reshape %v6103 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6148 = stablehlo.slice %v6147 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v6149 = stablehlo.reshape %v6148 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6150 = stablehlo.reshape %v6146 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6151 = stablehlo.transpose %v6150, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v6152 = stablehlo.reshape %v6151 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v6153 = stablehlo.reshape %v6143 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6154 = stablehlo.reshape %v6152 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v6155 = stablehlo.dot_general %v6153, %v6154, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v6156 = stablehlo.reshape %v6155 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v6157 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v6158 = stablehlo.multiply %v6156, %v6157 : tensor<32x38809xf32>
    %v6159 = stablehlo.reshape %v6158 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v6160 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v6161 = stablehlo.reduce(%v6159 init: %v6160) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6162 = stablehlo.broadcast_in_dim %v6161, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v6163 = stablehlo.subtract %v6159, %v6162 : tensor<32x197x197xf32>
    %v6164 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6165 = stablehlo.exponential %v6163 : tensor<32x197x197xf32>
    %v6166 = stablehlo.reduce(%v6165 init: %v6164) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6167 = stablehlo.broadcast_in_dim %v6166, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v6168 = stablehlo.divide %v6165, %v6167 : tensor<32x197x197xf32>
    %v6169 = stablehlo.reshape %v6168 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v6170 = stablehlo.reshape %v6169 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v6171 = stablehlo.reshape %v6149 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6172 = stablehlo.dot_general %v6170, %v6171, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v6173 = stablehlo.reshape %v6172 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6174 = stablehlo.reshape %v6173 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6175 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6176 = stablehlo.pad %v6174, %v6175, low = [0, 0, 64], high = [0, 0, 640], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v6177 = stablehlo.reshape %v6176 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v6178 = stablehlo.add %v6140, %v6177 : tensor<32x151296xf32>
    %v6179 = stablehlo.reshape %v6093 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6180 = stablehlo.slice %v6179 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v6181 = stablehlo.reshape %v6180 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6182 = stablehlo.reshape %v6098 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6183 = stablehlo.slice %v6182 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v6184 = stablehlo.reshape %v6183 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6185 = stablehlo.reshape %v6103 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6186 = stablehlo.slice %v6185 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v6187 = stablehlo.reshape %v6186 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6188 = stablehlo.reshape %v6184 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6189 = stablehlo.transpose %v6188, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v6190 = stablehlo.reshape %v6189 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v6191 = stablehlo.reshape %v6181 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6192 = stablehlo.reshape %v6190 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v6193 = stablehlo.dot_general %v6191, %v6192, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v6194 = stablehlo.reshape %v6193 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v6195 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v6196 = stablehlo.multiply %v6194, %v6195 : tensor<32x38809xf32>
    %v6197 = stablehlo.reshape %v6196 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v6198 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v6199 = stablehlo.reduce(%v6197 init: %v6198) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6200 = stablehlo.broadcast_in_dim %v6199, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v6201 = stablehlo.subtract %v6197, %v6200 : tensor<32x197x197xf32>
    %v6202 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6203 = stablehlo.exponential %v6201 : tensor<32x197x197xf32>
    %v6204 = stablehlo.reduce(%v6203 init: %v6202) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6205 = stablehlo.broadcast_in_dim %v6204, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v6206 = stablehlo.divide %v6203, %v6205 : tensor<32x197x197xf32>
    %v6207 = stablehlo.reshape %v6206 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v6208 = stablehlo.reshape %v6207 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v6209 = stablehlo.reshape %v6187 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6210 = stablehlo.dot_general %v6208, %v6209, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v6211 = stablehlo.reshape %v6210 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6212 = stablehlo.reshape %v6211 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6213 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6214 = stablehlo.pad %v6212, %v6213, low = [0, 0, 128], high = [0, 0, 576], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v6215 = stablehlo.reshape %v6214 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v6216 = stablehlo.add %v6178, %v6215 : tensor<32x151296xf32>
    %v6217 = stablehlo.reshape %v6093 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6218 = stablehlo.slice %v6217 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v6219 = stablehlo.reshape %v6218 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6220 = stablehlo.reshape %v6098 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6221 = stablehlo.slice %v6220 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v6222 = stablehlo.reshape %v6221 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6223 = stablehlo.reshape %v6103 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6224 = stablehlo.slice %v6223 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v6225 = stablehlo.reshape %v6224 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6226 = stablehlo.reshape %v6222 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6227 = stablehlo.transpose %v6226, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v6228 = stablehlo.reshape %v6227 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v6229 = stablehlo.reshape %v6219 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6230 = stablehlo.reshape %v6228 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v6231 = stablehlo.dot_general %v6229, %v6230, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v6232 = stablehlo.reshape %v6231 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v6233 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v6234 = stablehlo.multiply %v6232, %v6233 : tensor<32x38809xf32>
    %v6235 = stablehlo.reshape %v6234 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v6236 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v6237 = stablehlo.reduce(%v6235 init: %v6236) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6238 = stablehlo.broadcast_in_dim %v6237, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v6239 = stablehlo.subtract %v6235, %v6238 : tensor<32x197x197xf32>
    %v6240 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6241 = stablehlo.exponential %v6239 : tensor<32x197x197xf32>
    %v6242 = stablehlo.reduce(%v6241 init: %v6240) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6243 = stablehlo.broadcast_in_dim %v6242, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v6244 = stablehlo.divide %v6241, %v6243 : tensor<32x197x197xf32>
    %v6245 = stablehlo.reshape %v6244 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v6246 = stablehlo.reshape %v6245 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v6247 = stablehlo.reshape %v6225 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6248 = stablehlo.dot_general %v6246, %v6247, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v6249 = stablehlo.reshape %v6248 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6250 = stablehlo.reshape %v6249 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6251 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6252 = stablehlo.pad %v6250, %v6251, low = [0, 0, 192], high = [0, 0, 512], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v6253 = stablehlo.reshape %v6252 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v6254 = stablehlo.add %v6216, %v6253 : tensor<32x151296xf32>
    %v6255 = stablehlo.reshape %v6093 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6256 = stablehlo.slice %v6255 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v6257 = stablehlo.reshape %v6256 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6258 = stablehlo.reshape %v6098 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6259 = stablehlo.slice %v6258 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v6260 = stablehlo.reshape %v6259 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6261 = stablehlo.reshape %v6103 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6262 = stablehlo.slice %v6261 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v6263 = stablehlo.reshape %v6262 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6264 = stablehlo.reshape %v6260 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6265 = stablehlo.transpose %v6264, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v6266 = stablehlo.reshape %v6265 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v6267 = stablehlo.reshape %v6257 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6268 = stablehlo.reshape %v6266 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v6269 = stablehlo.dot_general %v6267, %v6268, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v6270 = stablehlo.reshape %v6269 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v6271 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v6272 = stablehlo.multiply %v6270, %v6271 : tensor<32x38809xf32>
    %v6273 = stablehlo.reshape %v6272 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v6274 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v6275 = stablehlo.reduce(%v6273 init: %v6274) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6276 = stablehlo.broadcast_in_dim %v6275, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v6277 = stablehlo.subtract %v6273, %v6276 : tensor<32x197x197xf32>
    %v6278 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6279 = stablehlo.exponential %v6277 : tensor<32x197x197xf32>
    %v6280 = stablehlo.reduce(%v6279 init: %v6278) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6281 = stablehlo.broadcast_in_dim %v6280, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v6282 = stablehlo.divide %v6279, %v6281 : tensor<32x197x197xf32>
    %v6283 = stablehlo.reshape %v6282 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v6284 = stablehlo.reshape %v6283 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v6285 = stablehlo.reshape %v6263 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6286 = stablehlo.dot_general %v6284, %v6285, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v6287 = stablehlo.reshape %v6286 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6288 = stablehlo.reshape %v6287 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6289 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6290 = stablehlo.pad %v6288, %v6289, low = [0, 0, 256], high = [0, 0, 448], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v6291 = stablehlo.reshape %v6290 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v6292 = stablehlo.add %v6254, %v6291 : tensor<32x151296xf32>
    %v6293 = stablehlo.reshape %v6093 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6294 = stablehlo.slice %v6293 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v6295 = stablehlo.reshape %v6294 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6296 = stablehlo.reshape %v6098 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6297 = stablehlo.slice %v6296 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v6298 = stablehlo.reshape %v6297 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6299 = stablehlo.reshape %v6103 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6300 = stablehlo.slice %v6299 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v6301 = stablehlo.reshape %v6300 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6302 = stablehlo.reshape %v6298 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6303 = stablehlo.transpose %v6302, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v6304 = stablehlo.reshape %v6303 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v6305 = stablehlo.reshape %v6295 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6306 = stablehlo.reshape %v6304 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v6307 = stablehlo.dot_general %v6305, %v6306, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v6308 = stablehlo.reshape %v6307 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v6309 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v6310 = stablehlo.multiply %v6308, %v6309 : tensor<32x38809xf32>
    %v6311 = stablehlo.reshape %v6310 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v6312 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v6313 = stablehlo.reduce(%v6311 init: %v6312) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6314 = stablehlo.broadcast_in_dim %v6313, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v6315 = stablehlo.subtract %v6311, %v6314 : tensor<32x197x197xf32>
    %v6316 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6317 = stablehlo.exponential %v6315 : tensor<32x197x197xf32>
    %v6318 = stablehlo.reduce(%v6317 init: %v6316) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6319 = stablehlo.broadcast_in_dim %v6318, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v6320 = stablehlo.divide %v6317, %v6319 : tensor<32x197x197xf32>
    %v6321 = stablehlo.reshape %v6320 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v6322 = stablehlo.reshape %v6321 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v6323 = stablehlo.reshape %v6301 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6324 = stablehlo.dot_general %v6322, %v6323, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v6325 = stablehlo.reshape %v6324 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6326 = stablehlo.reshape %v6325 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6327 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6328 = stablehlo.pad %v6326, %v6327, low = [0, 0, 320], high = [0, 0, 384], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v6329 = stablehlo.reshape %v6328 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v6330 = stablehlo.add %v6292, %v6329 : tensor<32x151296xf32>
    %v6331 = stablehlo.reshape %v6093 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6332 = stablehlo.slice %v6331 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v6333 = stablehlo.reshape %v6332 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6334 = stablehlo.reshape %v6098 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6335 = stablehlo.slice %v6334 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v6336 = stablehlo.reshape %v6335 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6337 = stablehlo.reshape %v6103 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6338 = stablehlo.slice %v6337 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v6339 = stablehlo.reshape %v6338 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6340 = stablehlo.reshape %v6336 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6341 = stablehlo.transpose %v6340, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v6342 = stablehlo.reshape %v6341 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v6343 = stablehlo.reshape %v6333 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6344 = stablehlo.reshape %v6342 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v6345 = stablehlo.dot_general %v6343, %v6344, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v6346 = stablehlo.reshape %v6345 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v6347 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v6348 = stablehlo.multiply %v6346, %v6347 : tensor<32x38809xf32>
    %v6349 = stablehlo.reshape %v6348 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v6350 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v6351 = stablehlo.reduce(%v6349 init: %v6350) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6352 = stablehlo.broadcast_in_dim %v6351, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v6353 = stablehlo.subtract %v6349, %v6352 : tensor<32x197x197xf32>
    %v6354 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6355 = stablehlo.exponential %v6353 : tensor<32x197x197xf32>
    %v6356 = stablehlo.reduce(%v6355 init: %v6354) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6357 = stablehlo.broadcast_in_dim %v6356, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v6358 = stablehlo.divide %v6355, %v6357 : tensor<32x197x197xf32>
    %v6359 = stablehlo.reshape %v6358 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v6360 = stablehlo.reshape %v6359 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v6361 = stablehlo.reshape %v6339 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6362 = stablehlo.dot_general %v6360, %v6361, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v6363 = stablehlo.reshape %v6362 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6364 = stablehlo.reshape %v6363 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6365 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6366 = stablehlo.pad %v6364, %v6365, low = [0, 0, 384], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v6367 = stablehlo.reshape %v6366 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v6368 = stablehlo.add %v6330, %v6367 : tensor<32x151296xf32>
    %v6369 = stablehlo.reshape %v6093 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6370 = stablehlo.slice %v6369 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v6371 = stablehlo.reshape %v6370 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6372 = stablehlo.reshape %v6098 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6373 = stablehlo.slice %v6372 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v6374 = stablehlo.reshape %v6373 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6375 = stablehlo.reshape %v6103 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6376 = stablehlo.slice %v6375 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v6377 = stablehlo.reshape %v6376 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6378 = stablehlo.reshape %v6374 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6379 = stablehlo.transpose %v6378, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v6380 = stablehlo.reshape %v6379 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v6381 = stablehlo.reshape %v6371 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6382 = stablehlo.reshape %v6380 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v6383 = stablehlo.dot_general %v6381, %v6382, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v6384 = stablehlo.reshape %v6383 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v6385 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v6386 = stablehlo.multiply %v6384, %v6385 : tensor<32x38809xf32>
    %v6387 = stablehlo.reshape %v6386 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v6388 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v6389 = stablehlo.reduce(%v6387 init: %v6388) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6390 = stablehlo.broadcast_in_dim %v6389, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v6391 = stablehlo.subtract %v6387, %v6390 : tensor<32x197x197xf32>
    %v6392 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6393 = stablehlo.exponential %v6391 : tensor<32x197x197xf32>
    %v6394 = stablehlo.reduce(%v6393 init: %v6392) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6395 = stablehlo.broadcast_in_dim %v6394, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v6396 = stablehlo.divide %v6393, %v6395 : tensor<32x197x197xf32>
    %v6397 = stablehlo.reshape %v6396 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v6398 = stablehlo.reshape %v6397 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v6399 = stablehlo.reshape %v6377 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6400 = stablehlo.dot_general %v6398, %v6399, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v6401 = stablehlo.reshape %v6400 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6402 = stablehlo.reshape %v6401 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6403 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6404 = stablehlo.pad %v6402, %v6403, low = [0, 0, 448], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v6405 = stablehlo.reshape %v6404 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v6406 = stablehlo.add %v6368, %v6405 : tensor<32x151296xf32>
    %v6407 = stablehlo.reshape %v6093 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6408 = stablehlo.slice %v6407 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v6409 = stablehlo.reshape %v6408 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6410 = stablehlo.reshape %v6098 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6411 = stablehlo.slice %v6410 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v6412 = stablehlo.reshape %v6411 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6413 = stablehlo.reshape %v6103 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6414 = stablehlo.slice %v6413 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v6415 = stablehlo.reshape %v6414 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6416 = stablehlo.reshape %v6412 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6417 = stablehlo.transpose %v6416, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v6418 = stablehlo.reshape %v6417 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v6419 = stablehlo.reshape %v6409 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6420 = stablehlo.reshape %v6418 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v6421 = stablehlo.dot_general %v6419, %v6420, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v6422 = stablehlo.reshape %v6421 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v6423 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v6424 = stablehlo.multiply %v6422, %v6423 : tensor<32x38809xf32>
    %v6425 = stablehlo.reshape %v6424 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v6426 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v6427 = stablehlo.reduce(%v6425 init: %v6426) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6428 = stablehlo.broadcast_in_dim %v6427, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v6429 = stablehlo.subtract %v6425, %v6428 : tensor<32x197x197xf32>
    %v6430 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6431 = stablehlo.exponential %v6429 : tensor<32x197x197xf32>
    %v6432 = stablehlo.reduce(%v6431 init: %v6430) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6433 = stablehlo.broadcast_in_dim %v6432, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v6434 = stablehlo.divide %v6431, %v6433 : tensor<32x197x197xf32>
    %v6435 = stablehlo.reshape %v6434 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v6436 = stablehlo.reshape %v6435 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v6437 = stablehlo.reshape %v6415 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6438 = stablehlo.dot_general %v6436, %v6437, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v6439 = stablehlo.reshape %v6438 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6440 = stablehlo.reshape %v6439 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6441 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6442 = stablehlo.pad %v6440, %v6441, low = [0, 0, 512], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v6443 = stablehlo.reshape %v6442 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v6444 = stablehlo.add %v6406, %v6443 : tensor<32x151296xf32>
    %v6445 = stablehlo.reshape %v6093 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6446 = stablehlo.slice %v6445 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v6447 = stablehlo.reshape %v6446 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6448 = stablehlo.reshape %v6098 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6449 = stablehlo.slice %v6448 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v6450 = stablehlo.reshape %v6449 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6451 = stablehlo.reshape %v6103 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6452 = stablehlo.slice %v6451 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v6453 = stablehlo.reshape %v6452 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6454 = stablehlo.reshape %v6450 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6455 = stablehlo.transpose %v6454, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v6456 = stablehlo.reshape %v6455 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v6457 = stablehlo.reshape %v6447 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6458 = stablehlo.reshape %v6456 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v6459 = stablehlo.dot_general %v6457, %v6458, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v6460 = stablehlo.reshape %v6459 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v6461 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v6462 = stablehlo.multiply %v6460, %v6461 : tensor<32x38809xf32>
    %v6463 = stablehlo.reshape %v6462 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v6464 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v6465 = stablehlo.reduce(%v6463 init: %v6464) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6466 = stablehlo.broadcast_in_dim %v6465, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v6467 = stablehlo.subtract %v6463, %v6466 : tensor<32x197x197xf32>
    %v6468 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6469 = stablehlo.exponential %v6467 : tensor<32x197x197xf32>
    %v6470 = stablehlo.reduce(%v6469 init: %v6468) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6471 = stablehlo.broadcast_in_dim %v6470, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v6472 = stablehlo.divide %v6469, %v6471 : tensor<32x197x197xf32>
    %v6473 = stablehlo.reshape %v6472 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v6474 = stablehlo.reshape %v6473 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v6475 = stablehlo.reshape %v6453 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6476 = stablehlo.dot_general %v6474, %v6475, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v6477 = stablehlo.reshape %v6476 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6478 = stablehlo.reshape %v6477 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6479 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6480 = stablehlo.pad %v6478, %v6479, low = [0, 0, 576], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v6481 = stablehlo.reshape %v6480 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v6482 = stablehlo.add %v6444, %v6481 : tensor<32x151296xf32>
    %v6483 = stablehlo.reshape %v6093 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6484 = stablehlo.slice %v6483 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v6485 = stablehlo.reshape %v6484 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6486 = stablehlo.reshape %v6098 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6487 = stablehlo.slice %v6486 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v6488 = stablehlo.reshape %v6487 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6489 = stablehlo.reshape %v6103 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6490 = stablehlo.slice %v6489 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v6491 = stablehlo.reshape %v6490 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6492 = stablehlo.reshape %v6488 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6493 = stablehlo.transpose %v6492, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v6494 = stablehlo.reshape %v6493 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v6495 = stablehlo.reshape %v6485 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6496 = stablehlo.reshape %v6494 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v6497 = stablehlo.dot_general %v6495, %v6496, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v6498 = stablehlo.reshape %v6497 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v6499 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v6500 = stablehlo.multiply %v6498, %v6499 : tensor<32x38809xf32>
    %v6501 = stablehlo.reshape %v6500 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v6502 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v6503 = stablehlo.reduce(%v6501 init: %v6502) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6504 = stablehlo.broadcast_in_dim %v6503, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v6505 = stablehlo.subtract %v6501, %v6504 : tensor<32x197x197xf32>
    %v6506 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6507 = stablehlo.exponential %v6505 : tensor<32x197x197xf32>
    %v6508 = stablehlo.reduce(%v6507 init: %v6506) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6509 = stablehlo.broadcast_in_dim %v6508, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v6510 = stablehlo.divide %v6507, %v6509 : tensor<32x197x197xf32>
    %v6511 = stablehlo.reshape %v6510 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v6512 = stablehlo.reshape %v6511 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v6513 = stablehlo.reshape %v6491 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6514 = stablehlo.dot_general %v6512, %v6513, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v6515 = stablehlo.reshape %v6514 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6516 = stablehlo.reshape %v6515 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6517 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6518 = stablehlo.pad %v6516, %v6517, low = [0, 0, 640], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v6519 = stablehlo.reshape %v6518 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v6520 = stablehlo.add %v6482, %v6519 : tensor<32x151296xf32>
    %v6521 = stablehlo.reshape %v6093 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6522 = stablehlo.slice %v6521 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v6523 = stablehlo.reshape %v6522 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6524 = stablehlo.reshape %v6098 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6525 = stablehlo.slice %v6524 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v6526 = stablehlo.reshape %v6525 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6527 = stablehlo.reshape %v6103 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6528 = stablehlo.slice %v6527 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v6529 = stablehlo.reshape %v6528 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6530 = stablehlo.reshape %v6526 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6531 = stablehlo.transpose %v6530, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v6532 = stablehlo.reshape %v6531 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v6533 = stablehlo.reshape %v6523 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6534 = stablehlo.reshape %v6532 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v6535 = stablehlo.dot_general %v6533, %v6534, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v6536 = stablehlo.reshape %v6535 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v6537 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v6538 = stablehlo.multiply %v6536, %v6537 : tensor<32x38809xf32>
    %v6539 = stablehlo.reshape %v6538 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v6540 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v6541 = stablehlo.reduce(%v6539 init: %v6540) applies stablehlo.maximum across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6542 = stablehlo.broadcast_in_dim %v6541, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v6543 = stablehlo.subtract %v6539, %v6542 : tensor<32x197x197xf32>
    %v6544 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6545 = stablehlo.exponential %v6543 : tensor<32x197x197xf32>
    %v6546 = stablehlo.reduce(%v6545 init: %v6544) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6547 = stablehlo.broadcast_in_dim %v6546, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v6548 = stablehlo.divide %v6545, %v6547 : tensor<32x197x197xf32>
    %v6549 = stablehlo.reshape %v6548 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v6550 = stablehlo.reshape %v6549 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v6551 = stablehlo.reshape %v6529 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6552 = stablehlo.dot_general %v6550, %v6551, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v6553 = stablehlo.reshape %v6552 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6554 = stablehlo.reshape %v6553 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6555 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6556 = stablehlo.pad %v6554, %v6555, low = [0, 0, 704], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v6557 = stablehlo.reshape %v6556 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v6558 = stablehlo.add %v6520, %v6557 : tensor<32x151296xf32>
    %v6559 = stablehlo.reshape %v6558 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6560 = stablehlo.dot_general %v6559, %b11_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v6561 = stablehlo.broadcast_in_dim %b11_bo, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v6562 = stablehlo.add %v6560, %v6561 : tensor<32x197x768xf32>
    %v6563 = stablehlo.reshape %v6562 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v6564 = stablehlo.add %v6060, %v6563 : tensor<32x151296xf32>
    %v6565 = stablehlo.reshape %v6564 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6566 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6567 = stablehlo.constant dense<768.0> : tensor<32x197x768xf32>
    %v6568 = stablehlo.constant dense<1.0e-6> : tensor<32x197x768xf32>
    %v6569 = stablehlo.reduce(%v6565 init: %v6566) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6570 = stablehlo.broadcast_in_dim %v6569, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v6571 = stablehlo.divide %v6570, %v6567 : tensor<32x197x768xf32>
    %v6572 = stablehlo.subtract %v6565, %v6571 : tensor<32x197x768xf32>
    %v6573 = stablehlo.multiply %v6572, %v6572 : tensor<32x197x768xf32>
    %v6574 = stablehlo.reduce(%v6573 init: %v6566) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6575 = stablehlo.broadcast_in_dim %v6574, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v6576 = stablehlo.divide %v6575, %v6567 : tensor<32x197x768xf32>
    %v6577 = stablehlo.add %v6576, %v6568 : tensor<32x197x768xf32>
    %v6578 = stablehlo.rsqrt %v6577 : tensor<32x197x768xf32>
    %v6579 = stablehlo.multiply %v6572, %v6578 : tensor<32x197x768xf32>
    %v6580 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v6581 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v6582 = stablehlo.multiply %v6579, %v6580 : tensor<32x197x768xf32>
    %v6583 = stablehlo.add %v6582, %v6581 : tensor<32x197x768xf32>
    %v6584 = stablehlo.reshape %v6583 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v6585 = stablehlo.reshape %v6584 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6586 = stablehlo.broadcast_in_dim %b11_g2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v6587 = stablehlo.multiply %v6585, %v6586 : tensor<32x197x768xf32>
    %v6588 = stablehlo.reshape %v6587 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v6589 = stablehlo.reshape %v6588 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6590 = stablehlo.broadcast_in_dim %b11_bt2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v6591 = stablehlo.add %v6589, %v6590 : tensor<32x197x768xf32>
    %v6592 = stablehlo.reshape %v6591 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v6593 = stablehlo.reshape %v6592 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6594 = stablehlo.dot_general %v6593, %b11_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x3072xf32>) -> tensor<32x197x3072xf32>
    %v6595 = stablehlo.broadcast_in_dim %b11_bfc1, dims = [2] : (tensor<3072xf32>) -> tensor<32x197x3072xf32>
    %v6596 = stablehlo.add %v6594, %v6595 : tensor<32x197x3072xf32>
    %v6597 = stablehlo.reshape %v6596 : (tensor<32x197x3072xf32>) -> tensor<32x605184xf32>
    %v6598 = stablehlo.constant dense<0.5> : tensor<32x605184xf32>
    %v6599 = stablehlo.multiply %v6598, %v6597 : tensor<32x605184xf32>
    %v6600 = stablehlo.negate %v6597 : tensor<32x605184xf32>
    %v6601 = stablehlo.constant dense<0.7071067811865476> : tensor<32x605184xf32>
    %v6602 = stablehlo.multiply %v6600, %v6601 : tensor<32x605184xf32>
    %v6603 = chlo.erfc %v6602 : tensor<32x605184xf32> -> tensor<32x605184xf32>
    %v6604 = stablehlo.multiply %v6599, %v6603 : tensor<32x605184xf32>
    %v6605 = stablehlo.reshape %v6604 : (tensor<32x605184xf32>) -> tensor<32x197x3072xf32>
    %v6606 = stablehlo.dot_general %v6605, %b11_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x3072xf32>, tensor<3072x768xf32>) -> tensor<32x197x768xf32>
    %v6607 = stablehlo.broadcast_in_dim %b11_bfc2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v6608 = stablehlo.add %v6606, %v6607 : tensor<32x197x768xf32>
    %v6609 = stablehlo.reshape %v6608 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v6610 = stablehlo.add %v6564, %v6609 : tensor<32x151296xf32>
    %v6611 = stablehlo.reshape %v6610 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6612 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6613 = stablehlo.constant dense<768.0> : tensor<32x197x768xf32>
    %v6614 = stablehlo.constant dense<1.0e-6> : tensor<32x197x768xf32>
    %v6615 = stablehlo.reduce(%v6611 init: %v6612) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6616 = stablehlo.broadcast_in_dim %v6615, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v6617 = stablehlo.divide %v6616, %v6613 : tensor<32x197x768xf32>
    %v6618 = stablehlo.subtract %v6611, %v6617 : tensor<32x197x768xf32>
    %v6619 = stablehlo.multiply %v6618, %v6618 : tensor<32x197x768xf32>
    %v6620 = stablehlo.reduce(%v6619 init: %v6612) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6621 = stablehlo.broadcast_in_dim %v6620, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v6622 = stablehlo.divide %v6621, %v6613 : tensor<32x197x768xf32>
    %v6623 = stablehlo.add %v6622, %v6614 : tensor<32x197x768xf32>
    %v6624 = stablehlo.rsqrt %v6623 : tensor<32x197x768xf32>
    %v6625 = stablehlo.multiply %v6618, %v6624 : tensor<32x197x768xf32>
    %v6626 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v6627 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v6628 = stablehlo.multiply %v6625, %v6626 : tensor<32x197x768xf32>
    %v6629 = stablehlo.add %v6628, %v6627 : tensor<32x197x768xf32>
    %v6630 = stablehlo.reshape %v6629 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v6631 = stablehlo.reshape %v6630 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6632 = stablehlo.broadcast_in_dim %gF, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v6633 = stablehlo.multiply %v6631, %v6632 : tensor<32x197x768xf32>
    %v6634 = stablehlo.reshape %v6633 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v6635 = stablehlo.reshape %v6634 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6636 = stablehlo.broadcast_in_dim %btF, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v6637 = stablehlo.add %v6635, %v6636 : tensor<32x197x768xf32>
    %v6638 = stablehlo.reshape %v6637 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v6639 = stablehlo.reshape %v6638 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6640 = stablehlo.slice %v6639 [0:32, 0:1, 0:768] : (tensor<32x197x768xf32>) -> tensor<32x1x768xf32>
    %v6641 = stablehlo.reshape %v6640 : (tensor<32x1x768xf32>) -> tensor<32x768xf32>
    %v6642 = stablehlo.dot_general %v6641, %Wc, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x768xf32>, tensor<768x1000xf32>) -> tensor<32x1000xf32>
    %v6643 = stablehlo.broadcast_in_dim %bc, dims = [1] : (tensor<1000xf32>) -> tensor<32x1000xf32>
    %v6644 = stablehlo.add %v6642, %v6643 : tensor<32x1000xf32>
    return %v6644 : tensor<32x1000xf32>
  }
}

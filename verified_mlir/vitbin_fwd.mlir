module @m {
  func.func @vitbin_fwd(%x: tensor<32x150528xf32>, %wConv: tensor<768x3x16x16xf32>, %bConv: tensor<768xf32>, %cls: tensor<768xf32>, %pos: tensor<197x768xf32>, %b0_g1: tensor<768xf32>, %b0_bt1: tensor<768xf32>, %b0_Wq: tensor<768x768xf32>, %b0_bq: tensor<768xf32>, %b0_Wk: tensor<768x768xf32>, %b0_bk: tensor<768xf32>, %b0_Wv: tensor<768x768xf32>, %b0_bv: tensor<768xf32>, %b0_Wo: tensor<768x768xf32>, %b0_bo: tensor<768xf32>, %b0_g2: tensor<768xf32>, %b0_bt2: tensor<768xf32>, %b0_Wfc1: tensor<768x3072xf32>, %b0_bfc1: tensor<3072xf32>, %b0_Wfc2: tensor<3072x768xf32>, %b0_bfc2: tensor<768xf32>, %b1_g1: tensor<768xf32>, %b1_bt1: tensor<768xf32>, %b1_Wq: tensor<768x768xf32>, %b1_bq: tensor<768xf32>, %b1_Wk: tensor<768x768xf32>, %b1_bk: tensor<768xf32>, %b1_Wv: tensor<768x768xf32>, %b1_bv: tensor<768xf32>, %b1_Wo: tensor<768x768xf32>, %b1_bo: tensor<768xf32>, %b1_g2: tensor<768xf32>, %b1_bt2: tensor<768xf32>, %b1_Wfc1: tensor<768x3072xf32>, %b1_bfc1: tensor<3072xf32>, %b1_Wfc2: tensor<3072x768xf32>, %b1_bfc2: tensor<768xf32>, %b2_g1: tensor<768xf32>, %b2_bt1: tensor<768xf32>, %b2_Wq: tensor<768x768xf32>, %b2_bq: tensor<768xf32>, %b2_Wk: tensor<768x768xf32>, %b2_bk: tensor<768xf32>, %b2_Wv: tensor<768x768xf32>, %b2_bv: tensor<768xf32>, %b2_Wo: tensor<768x768xf32>, %b2_bo: tensor<768xf32>, %b2_g2: tensor<768xf32>, %b2_bt2: tensor<768xf32>, %b2_Wfc1: tensor<768x3072xf32>, %b2_bfc1: tensor<3072xf32>, %b2_Wfc2: tensor<3072x768xf32>, %b2_bfc2: tensor<768xf32>, %b3_g1: tensor<768xf32>, %b3_bt1: tensor<768xf32>, %b3_Wq: tensor<768x768xf32>, %b3_bq: tensor<768xf32>, %b3_Wk: tensor<768x768xf32>, %b3_bk: tensor<768xf32>, %b3_Wv: tensor<768x768xf32>, %b3_bv: tensor<768xf32>, %b3_Wo: tensor<768x768xf32>, %b3_bo: tensor<768xf32>, %b3_g2: tensor<768xf32>, %b3_bt2: tensor<768xf32>, %b3_Wfc1: tensor<768x3072xf32>, %b3_bfc1: tensor<3072xf32>, %b3_Wfc2: tensor<3072x768xf32>, %b3_bfc2: tensor<768xf32>, %b4_g1: tensor<768xf32>, %b4_bt1: tensor<768xf32>, %b4_Wq: tensor<768x768xf32>, %b4_bq: tensor<768xf32>, %b4_Wk: tensor<768x768xf32>, %b4_bk: tensor<768xf32>, %b4_Wv: tensor<768x768xf32>, %b4_bv: tensor<768xf32>, %b4_Wo: tensor<768x768xf32>, %b4_bo: tensor<768xf32>, %b4_g2: tensor<768xf32>, %b4_bt2: tensor<768xf32>, %b4_Wfc1: tensor<768x3072xf32>, %b4_bfc1: tensor<3072xf32>, %b4_Wfc2: tensor<3072x768xf32>, %b4_bfc2: tensor<768xf32>, %b5_g1: tensor<768xf32>, %b5_bt1: tensor<768xf32>, %b5_Wq: tensor<768x768xf32>, %b5_bq: tensor<768xf32>, %b5_Wk: tensor<768x768xf32>, %b5_bk: tensor<768xf32>, %b5_Wv: tensor<768x768xf32>, %b5_bv: tensor<768xf32>, %b5_Wo: tensor<768x768xf32>, %b5_bo: tensor<768xf32>, %b5_g2: tensor<768xf32>, %b5_bt2: tensor<768xf32>, %b5_Wfc1: tensor<768x3072xf32>, %b5_bfc1: tensor<3072xf32>, %b5_Wfc2: tensor<3072x768xf32>, %b5_bfc2: tensor<768xf32>, %b6_g1: tensor<768xf32>, %b6_bt1: tensor<768xf32>, %b6_Wq: tensor<768x768xf32>, %b6_bq: tensor<768xf32>, %b6_Wk: tensor<768x768xf32>, %b6_bk: tensor<768xf32>, %b6_Wv: tensor<768x768xf32>, %b6_bv: tensor<768xf32>, %b6_Wo: tensor<768x768xf32>, %b6_bo: tensor<768xf32>, %b6_g2: tensor<768xf32>, %b6_bt2: tensor<768xf32>, %b6_Wfc1: tensor<768x3072xf32>, %b6_bfc1: tensor<3072xf32>, %b6_Wfc2: tensor<3072x768xf32>, %b6_bfc2: tensor<768xf32>, %b7_g1: tensor<768xf32>, %b7_bt1: tensor<768xf32>, %b7_Wq: tensor<768x768xf32>, %b7_bq: tensor<768xf32>, %b7_Wk: tensor<768x768xf32>, %b7_bk: tensor<768xf32>, %b7_Wv: tensor<768x768xf32>, %b7_bv: tensor<768xf32>, %b7_Wo: tensor<768x768xf32>, %b7_bo: tensor<768xf32>, %b7_g2: tensor<768xf32>, %b7_bt2: tensor<768xf32>, %b7_Wfc1: tensor<768x3072xf32>, %b7_bfc1: tensor<3072xf32>, %b7_Wfc2: tensor<3072x768xf32>, %b7_bfc2: tensor<768xf32>, %b8_g1: tensor<768xf32>, %b8_bt1: tensor<768xf32>, %b8_Wq: tensor<768x768xf32>, %b8_bq: tensor<768xf32>, %b8_Wk: tensor<768x768xf32>, %b8_bk: tensor<768xf32>, %b8_Wv: tensor<768x768xf32>, %b8_bv: tensor<768xf32>, %b8_Wo: tensor<768x768xf32>, %b8_bo: tensor<768xf32>, %b8_g2: tensor<768xf32>, %b8_bt2: tensor<768xf32>, %b8_Wfc1: tensor<768x3072xf32>, %b8_bfc1: tensor<3072xf32>, %b8_Wfc2: tensor<3072x768xf32>, %b8_bfc2: tensor<768xf32>, %b9_g1: tensor<768xf32>, %b9_bt1: tensor<768xf32>, %b9_Wq: tensor<768x768xf32>, %b9_bq: tensor<768xf32>, %b9_Wk: tensor<768x768xf32>, %b9_bk: tensor<768xf32>, %b9_Wv: tensor<768x768xf32>, %b9_bv: tensor<768xf32>, %b9_Wo: tensor<768x768xf32>, %b9_bo: tensor<768xf32>, %b9_g2: tensor<768xf32>, %b9_bt2: tensor<768xf32>, %b9_Wfc1: tensor<768x3072xf32>, %b9_bfc1: tensor<3072xf32>, %b9_Wfc2: tensor<3072x768xf32>, %b9_bfc2: tensor<768xf32>, %b10_g1: tensor<768xf32>, %b10_bt1: tensor<768xf32>, %b10_Wq: tensor<768x768xf32>, %b10_bq: tensor<768xf32>, %b10_Wk: tensor<768x768xf32>, %b10_bk: tensor<768xf32>, %b10_Wv: tensor<768x768xf32>, %b10_bv: tensor<768xf32>, %b10_Wo: tensor<768x768xf32>, %b10_bo: tensor<768xf32>, %b10_g2: tensor<768xf32>, %b10_bt2: tensor<768xf32>, %b10_Wfc1: tensor<768x3072xf32>, %b10_bfc1: tensor<3072xf32>, %b10_Wfc2: tensor<3072x768xf32>, %b10_bfc2: tensor<768xf32>, %b11_g1: tensor<768xf32>, %b11_bt1: tensor<768xf32>, %b11_Wq: tensor<768x768xf32>, %b11_bq: tensor<768xf32>, %b11_Wk: tensor<768x768xf32>, %b11_bk: tensor<768xf32>, %b11_Wv: tensor<768x768xf32>, %b11_bv: tensor<768xf32>, %b11_Wo: tensor<768x768xf32>, %b11_bo: tensor<768xf32>, %b11_g2: tensor<768xf32>, %b11_bt2: tensor<768xf32>, %b11_Wfc1: tensor<768x3072xf32>, %b11_bfc1: tensor<3072xf32>, %b11_Wfc2: tensor<3072x768xf32>, %b11_bfc2: tensor<768xf32>, %gF: tensor<768xf32>, %btF: tensor<768xf32>, %Wc: tensor<768x1000xf32>, %bc: tensor<1000xf32>) -> tensor<32x1000xf32> {
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
    %v500 = stablehlo.multiply %v499, %v499 : tensor<32x605184xf32>
    %v501 = stablehlo.multiply %v500, %v499 : tensor<32x605184xf32>
    %v502 = stablehlo.constant dense<0.044715> : tensor<32x605184xf32>
    %v503 = stablehlo.multiply %v502, %v501 : tensor<32x605184xf32>
    %v504 = stablehlo.add %v499, %v503 : tensor<32x605184xf32>
    %v505 = stablehlo.constant dense<0.7978845608028654> : tensor<32x605184xf32>
    %v506 = stablehlo.multiply %v505, %v504 : tensor<32x605184xf32>
    %v507 = stablehlo.tanh %v506 : tensor<32x605184xf32>
    %v508 = stablehlo.constant dense<1.0> : tensor<32x605184xf32>
    %v509 = stablehlo.add %v508, %v507 : tensor<32x605184xf32>
    %v510 = stablehlo.constant dense<0.5> : tensor<32x605184xf32>
    %v511 = stablehlo.multiply %v510, %v499 : tensor<32x605184xf32>
    %v512 = stablehlo.multiply %v511, %v509 : tensor<32x605184xf32>
    %v513 = stablehlo.reshape %v512 : (tensor<32x605184xf32>) -> tensor<32x197x3072xf32>
    %v514 = stablehlo.dot_general %v513, %b0_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x3072xf32>, tensor<3072x768xf32>) -> tensor<32x197x768xf32>
    %v515 = stablehlo.broadcast_in_dim %b0_bfc2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v516 = stablehlo.add %v514, %v515 : tensor<32x197x768xf32>
    %v517 = stablehlo.reshape %v516 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v518 = stablehlo.add %v466, %v517 : tensor<32x151296xf32>
    %v519 = stablehlo.reshape %v518 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v520 = stablehlo.constant dense<0.0> : tensor<f32>
    %v521 = stablehlo.constant dense<768.0> : tensor<32x197x768xf32>
    %v522 = stablehlo.constant dense<1.0e-5> : tensor<32x197x768xf32>
    %v523 = stablehlo.reduce(%v519 init: %v520) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v524 = stablehlo.broadcast_in_dim %v523, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v525 = stablehlo.divide %v524, %v521 : tensor<32x197x768xf32>
    %v526 = stablehlo.subtract %v519, %v525 : tensor<32x197x768xf32>
    %v527 = stablehlo.multiply %v526, %v526 : tensor<32x197x768xf32>
    %v528 = stablehlo.reduce(%v527 init: %v520) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v529 = stablehlo.broadcast_in_dim %v528, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v530 = stablehlo.divide %v529, %v521 : tensor<32x197x768xf32>
    %v531 = stablehlo.add %v530, %v522 : tensor<32x197x768xf32>
    %v532 = stablehlo.rsqrt %v531 : tensor<32x197x768xf32>
    %v533 = stablehlo.multiply %v526, %v532 : tensor<32x197x768xf32>
    %v534 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v535 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v536 = stablehlo.multiply %v533, %v534 : tensor<32x197x768xf32>
    %v537 = stablehlo.add %v536, %v535 : tensor<32x197x768xf32>
    %v538 = stablehlo.reshape %v537 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v539 = stablehlo.reshape %v538 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v540 = stablehlo.broadcast_in_dim %b1_g1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v541 = stablehlo.multiply %v539, %v540 : tensor<32x197x768xf32>
    %v542 = stablehlo.reshape %v541 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v543 = stablehlo.reshape %v542 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v544 = stablehlo.broadcast_in_dim %b1_bt1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v545 = stablehlo.add %v543, %v544 : tensor<32x197x768xf32>
    %v546 = stablehlo.reshape %v545 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v547 = stablehlo.reshape %v546 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v548 = stablehlo.dot_general %v547, %b1_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v549 = stablehlo.broadcast_in_dim %b1_bq, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v550 = stablehlo.add %v548, %v549 : tensor<32x197x768xf32>
    %v551 = stablehlo.reshape %v550 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v552 = stablehlo.reshape %v546 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v553 = stablehlo.dot_general %v552, %b1_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v554 = stablehlo.broadcast_in_dim %b1_bk, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v555 = stablehlo.add %v553, %v554 : tensor<32x197x768xf32>
    %v556 = stablehlo.reshape %v555 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v557 = stablehlo.reshape %v546 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v558 = stablehlo.dot_general %v557, %b1_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v559 = stablehlo.broadcast_in_dim %b1_bv, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v560 = stablehlo.add %v558, %v559 : tensor<32x197x768xf32>
    %v561 = stablehlo.reshape %v560 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v562 = stablehlo.reshape %v551 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v563 = stablehlo.slice %v562 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v564 = stablehlo.reshape %v563 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v565 = stablehlo.reshape %v556 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v566 = stablehlo.slice %v565 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v567 = stablehlo.reshape %v566 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v568 = stablehlo.reshape %v561 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v569 = stablehlo.slice %v568 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v570 = stablehlo.reshape %v569 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v571 = stablehlo.reshape %v567 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v572 = stablehlo.transpose %v571, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v573 = stablehlo.reshape %v572 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v574 = stablehlo.reshape %v564 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v575 = stablehlo.reshape %v573 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v576 = stablehlo.dot_general %v574, %v575, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v577 = stablehlo.reshape %v576 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v578 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v579 = stablehlo.multiply %v577, %v578 : tensor<32x38809xf32>
    %v580 = stablehlo.reshape %v579 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v581 = stablehlo.constant dense<0.0> : tensor<f32>
    %v582 = stablehlo.exponential %v580 : tensor<32x197x197xf32>
    %v583 = stablehlo.reduce(%v582 init: %v581) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v584 = stablehlo.broadcast_in_dim %v583, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v585 = stablehlo.divide %v582, %v584 : tensor<32x197x197xf32>
    %v586 = stablehlo.reshape %v585 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v587 = stablehlo.reshape %v586 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v588 = stablehlo.reshape %v570 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v589 = stablehlo.dot_general %v587, %v588, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v590 = stablehlo.reshape %v589 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v591 = stablehlo.reshape %v590 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v592 = stablehlo.constant dense<0.0> : tensor<f32>
    %v593 = stablehlo.pad %v591, %v592, low = [0, 0, 0], high = [0, 0, 704], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v594 = stablehlo.reshape %v593 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v595 = stablehlo.reshape %v551 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v596 = stablehlo.slice %v595 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v597 = stablehlo.reshape %v596 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v598 = stablehlo.reshape %v556 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v599 = stablehlo.slice %v598 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v600 = stablehlo.reshape %v599 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v601 = stablehlo.reshape %v561 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v602 = stablehlo.slice %v601 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v603 = stablehlo.reshape %v602 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v604 = stablehlo.reshape %v600 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v605 = stablehlo.transpose %v604, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v606 = stablehlo.reshape %v605 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v607 = stablehlo.reshape %v597 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v608 = stablehlo.reshape %v606 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v609 = stablehlo.dot_general %v607, %v608, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v610 = stablehlo.reshape %v609 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v611 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v612 = stablehlo.multiply %v610, %v611 : tensor<32x38809xf32>
    %v613 = stablehlo.reshape %v612 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v614 = stablehlo.constant dense<0.0> : tensor<f32>
    %v615 = stablehlo.exponential %v613 : tensor<32x197x197xf32>
    %v616 = stablehlo.reduce(%v615 init: %v614) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v617 = stablehlo.broadcast_in_dim %v616, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v618 = stablehlo.divide %v615, %v617 : tensor<32x197x197xf32>
    %v619 = stablehlo.reshape %v618 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v620 = stablehlo.reshape %v619 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v621 = stablehlo.reshape %v603 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v622 = stablehlo.dot_general %v620, %v621, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v623 = stablehlo.reshape %v622 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v624 = stablehlo.reshape %v623 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v625 = stablehlo.constant dense<0.0> : tensor<f32>
    %v626 = stablehlo.pad %v624, %v625, low = [0, 0, 64], high = [0, 0, 640], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v627 = stablehlo.reshape %v626 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v628 = stablehlo.add %v594, %v627 : tensor<32x151296xf32>
    %v629 = stablehlo.reshape %v551 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v630 = stablehlo.slice %v629 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v631 = stablehlo.reshape %v630 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v632 = stablehlo.reshape %v556 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v633 = stablehlo.slice %v632 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v634 = stablehlo.reshape %v633 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v635 = stablehlo.reshape %v561 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v636 = stablehlo.slice %v635 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v637 = stablehlo.reshape %v636 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v638 = stablehlo.reshape %v634 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v639 = stablehlo.transpose %v638, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v640 = stablehlo.reshape %v639 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v641 = stablehlo.reshape %v631 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v642 = stablehlo.reshape %v640 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v643 = stablehlo.dot_general %v641, %v642, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v644 = stablehlo.reshape %v643 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v645 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v646 = stablehlo.multiply %v644, %v645 : tensor<32x38809xf32>
    %v647 = stablehlo.reshape %v646 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v648 = stablehlo.constant dense<0.0> : tensor<f32>
    %v649 = stablehlo.exponential %v647 : tensor<32x197x197xf32>
    %v650 = stablehlo.reduce(%v649 init: %v648) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v651 = stablehlo.broadcast_in_dim %v650, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v652 = stablehlo.divide %v649, %v651 : tensor<32x197x197xf32>
    %v653 = stablehlo.reshape %v652 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v654 = stablehlo.reshape %v653 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v655 = stablehlo.reshape %v637 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v656 = stablehlo.dot_general %v654, %v655, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v657 = stablehlo.reshape %v656 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v658 = stablehlo.reshape %v657 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v659 = stablehlo.constant dense<0.0> : tensor<f32>
    %v660 = stablehlo.pad %v658, %v659, low = [0, 0, 128], high = [0, 0, 576], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v661 = stablehlo.reshape %v660 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v662 = stablehlo.add %v628, %v661 : tensor<32x151296xf32>
    %v663 = stablehlo.reshape %v551 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v664 = stablehlo.slice %v663 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v665 = stablehlo.reshape %v664 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v666 = stablehlo.reshape %v556 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v667 = stablehlo.slice %v666 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v668 = stablehlo.reshape %v667 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v669 = stablehlo.reshape %v561 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v670 = stablehlo.slice %v669 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v671 = stablehlo.reshape %v670 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v672 = stablehlo.reshape %v668 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v673 = stablehlo.transpose %v672, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v674 = stablehlo.reshape %v673 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v675 = stablehlo.reshape %v665 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v676 = stablehlo.reshape %v674 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v677 = stablehlo.dot_general %v675, %v676, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v678 = stablehlo.reshape %v677 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v679 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v680 = stablehlo.multiply %v678, %v679 : tensor<32x38809xf32>
    %v681 = stablehlo.reshape %v680 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v682 = stablehlo.constant dense<0.0> : tensor<f32>
    %v683 = stablehlo.exponential %v681 : tensor<32x197x197xf32>
    %v684 = stablehlo.reduce(%v683 init: %v682) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v685 = stablehlo.broadcast_in_dim %v684, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v686 = stablehlo.divide %v683, %v685 : tensor<32x197x197xf32>
    %v687 = stablehlo.reshape %v686 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v688 = stablehlo.reshape %v687 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v689 = stablehlo.reshape %v671 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v690 = stablehlo.dot_general %v688, %v689, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v691 = stablehlo.reshape %v690 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v692 = stablehlo.reshape %v691 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v693 = stablehlo.constant dense<0.0> : tensor<f32>
    %v694 = stablehlo.pad %v692, %v693, low = [0, 0, 192], high = [0, 0, 512], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v695 = stablehlo.reshape %v694 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v696 = stablehlo.add %v662, %v695 : tensor<32x151296xf32>
    %v697 = stablehlo.reshape %v551 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v698 = stablehlo.slice %v697 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v699 = stablehlo.reshape %v698 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v700 = stablehlo.reshape %v556 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v701 = stablehlo.slice %v700 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v702 = stablehlo.reshape %v701 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v703 = stablehlo.reshape %v561 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v704 = stablehlo.slice %v703 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v705 = stablehlo.reshape %v704 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v706 = stablehlo.reshape %v702 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v707 = stablehlo.transpose %v706, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v708 = stablehlo.reshape %v707 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v709 = stablehlo.reshape %v699 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v710 = stablehlo.reshape %v708 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v711 = stablehlo.dot_general %v709, %v710, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v712 = stablehlo.reshape %v711 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v713 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v714 = stablehlo.multiply %v712, %v713 : tensor<32x38809xf32>
    %v715 = stablehlo.reshape %v714 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v716 = stablehlo.constant dense<0.0> : tensor<f32>
    %v717 = stablehlo.exponential %v715 : tensor<32x197x197xf32>
    %v718 = stablehlo.reduce(%v717 init: %v716) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v719 = stablehlo.broadcast_in_dim %v718, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v720 = stablehlo.divide %v717, %v719 : tensor<32x197x197xf32>
    %v721 = stablehlo.reshape %v720 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v722 = stablehlo.reshape %v721 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v723 = stablehlo.reshape %v705 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v724 = stablehlo.dot_general %v722, %v723, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v725 = stablehlo.reshape %v724 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v726 = stablehlo.reshape %v725 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v727 = stablehlo.constant dense<0.0> : tensor<f32>
    %v728 = stablehlo.pad %v726, %v727, low = [0, 0, 256], high = [0, 0, 448], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v729 = stablehlo.reshape %v728 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v730 = stablehlo.add %v696, %v729 : tensor<32x151296xf32>
    %v731 = stablehlo.reshape %v551 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v732 = stablehlo.slice %v731 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v733 = stablehlo.reshape %v732 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v734 = stablehlo.reshape %v556 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v735 = stablehlo.slice %v734 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v736 = stablehlo.reshape %v735 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v737 = stablehlo.reshape %v561 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v738 = stablehlo.slice %v737 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v739 = stablehlo.reshape %v738 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v740 = stablehlo.reshape %v736 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v741 = stablehlo.transpose %v740, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v742 = stablehlo.reshape %v741 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v743 = stablehlo.reshape %v733 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v744 = stablehlo.reshape %v742 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v745 = stablehlo.dot_general %v743, %v744, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v746 = stablehlo.reshape %v745 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v747 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v748 = stablehlo.multiply %v746, %v747 : tensor<32x38809xf32>
    %v749 = stablehlo.reshape %v748 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v750 = stablehlo.constant dense<0.0> : tensor<f32>
    %v751 = stablehlo.exponential %v749 : tensor<32x197x197xf32>
    %v752 = stablehlo.reduce(%v751 init: %v750) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v753 = stablehlo.broadcast_in_dim %v752, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v754 = stablehlo.divide %v751, %v753 : tensor<32x197x197xf32>
    %v755 = stablehlo.reshape %v754 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v756 = stablehlo.reshape %v755 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v757 = stablehlo.reshape %v739 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v758 = stablehlo.dot_general %v756, %v757, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v759 = stablehlo.reshape %v758 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v760 = stablehlo.reshape %v759 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v761 = stablehlo.constant dense<0.0> : tensor<f32>
    %v762 = stablehlo.pad %v760, %v761, low = [0, 0, 320], high = [0, 0, 384], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v763 = stablehlo.reshape %v762 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v764 = stablehlo.add %v730, %v763 : tensor<32x151296xf32>
    %v765 = stablehlo.reshape %v551 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v766 = stablehlo.slice %v765 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v767 = stablehlo.reshape %v766 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v768 = stablehlo.reshape %v556 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v769 = stablehlo.slice %v768 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v770 = stablehlo.reshape %v769 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v771 = stablehlo.reshape %v561 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v772 = stablehlo.slice %v771 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v773 = stablehlo.reshape %v772 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v774 = stablehlo.reshape %v770 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v775 = stablehlo.transpose %v774, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v776 = stablehlo.reshape %v775 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v777 = stablehlo.reshape %v767 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v778 = stablehlo.reshape %v776 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v779 = stablehlo.dot_general %v777, %v778, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v780 = stablehlo.reshape %v779 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v781 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v782 = stablehlo.multiply %v780, %v781 : tensor<32x38809xf32>
    %v783 = stablehlo.reshape %v782 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v784 = stablehlo.constant dense<0.0> : tensor<f32>
    %v785 = stablehlo.exponential %v783 : tensor<32x197x197xf32>
    %v786 = stablehlo.reduce(%v785 init: %v784) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v787 = stablehlo.broadcast_in_dim %v786, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v788 = stablehlo.divide %v785, %v787 : tensor<32x197x197xf32>
    %v789 = stablehlo.reshape %v788 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v790 = stablehlo.reshape %v789 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v791 = stablehlo.reshape %v773 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v792 = stablehlo.dot_general %v790, %v791, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v793 = stablehlo.reshape %v792 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v794 = stablehlo.reshape %v793 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v795 = stablehlo.constant dense<0.0> : tensor<f32>
    %v796 = stablehlo.pad %v794, %v795, low = [0, 0, 384], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v797 = stablehlo.reshape %v796 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v798 = stablehlo.add %v764, %v797 : tensor<32x151296xf32>
    %v799 = stablehlo.reshape %v551 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v800 = stablehlo.slice %v799 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v801 = stablehlo.reshape %v800 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v802 = stablehlo.reshape %v556 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v803 = stablehlo.slice %v802 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v804 = stablehlo.reshape %v803 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v805 = stablehlo.reshape %v561 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v806 = stablehlo.slice %v805 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v807 = stablehlo.reshape %v806 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v808 = stablehlo.reshape %v804 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v809 = stablehlo.transpose %v808, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v810 = stablehlo.reshape %v809 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v811 = stablehlo.reshape %v801 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v812 = stablehlo.reshape %v810 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v813 = stablehlo.dot_general %v811, %v812, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v814 = stablehlo.reshape %v813 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v815 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v816 = stablehlo.multiply %v814, %v815 : tensor<32x38809xf32>
    %v817 = stablehlo.reshape %v816 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v818 = stablehlo.constant dense<0.0> : tensor<f32>
    %v819 = stablehlo.exponential %v817 : tensor<32x197x197xf32>
    %v820 = stablehlo.reduce(%v819 init: %v818) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v821 = stablehlo.broadcast_in_dim %v820, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v822 = stablehlo.divide %v819, %v821 : tensor<32x197x197xf32>
    %v823 = stablehlo.reshape %v822 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v824 = stablehlo.reshape %v823 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v825 = stablehlo.reshape %v807 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v826 = stablehlo.dot_general %v824, %v825, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v827 = stablehlo.reshape %v826 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v828 = stablehlo.reshape %v827 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v829 = stablehlo.constant dense<0.0> : tensor<f32>
    %v830 = stablehlo.pad %v828, %v829, low = [0, 0, 448], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v831 = stablehlo.reshape %v830 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v832 = stablehlo.add %v798, %v831 : tensor<32x151296xf32>
    %v833 = stablehlo.reshape %v551 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v834 = stablehlo.slice %v833 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v835 = stablehlo.reshape %v834 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v836 = stablehlo.reshape %v556 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v837 = stablehlo.slice %v836 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v838 = stablehlo.reshape %v837 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v839 = stablehlo.reshape %v561 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v840 = stablehlo.slice %v839 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v841 = stablehlo.reshape %v840 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v842 = stablehlo.reshape %v838 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v843 = stablehlo.transpose %v842, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v844 = stablehlo.reshape %v843 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v845 = stablehlo.reshape %v835 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v846 = stablehlo.reshape %v844 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v847 = stablehlo.dot_general %v845, %v846, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v848 = stablehlo.reshape %v847 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v849 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v850 = stablehlo.multiply %v848, %v849 : tensor<32x38809xf32>
    %v851 = stablehlo.reshape %v850 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v852 = stablehlo.constant dense<0.0> : tensor<f32>
    %v853 = stablehlo.exponential %v851 : tensor<32x197x197xf32>
    %v854 = stablehlo.reduce(%v853 init: %v852) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v855 = stablehlo.broadcast_in_dim %v854, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v856 = stablehlo.divide %v853, %v855 : tensor<32x197x197xf32>
    %v857 = stablehlo.reshape %v856 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v858 = stablehlo.reshape %v857 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v859 = stablehlo.reshape %v841 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v860 = stablehlo.dot_general %v858, %v859, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v861 = stablehlo.reshape %v860 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v862 = stablehlo.reshape %v861 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v863 = stablehlo.constant dense<0.0> : tensor<f32>
    %v864 = stablehlo.pad %v862, %v863, low = [0, 0, 512], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v865 = stablehlo.reshape %v864 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v866 = stablehlo.add %v832, %v865 : tensor<32x151296xf32>
    %v867 = stablehlo.reshape %v551 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v868 = stablehlo.slice %v867 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v869 = stablehlo.reshape %v868 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v870 = stablehlo.reshape %v556 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v871 = stablehlo.slice %v870 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v872 = stablehlo.reshape %v871 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v873 = stablehlo.reshape %v561 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v874 = stablehlo.slice %v873 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v875 = stablehlo.reshape %v874 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v876 = stablehlo.reshape %v872 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v877 = stablehlo.transpose %v876, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v878 = stablehlo.reshape %v877 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v879 = stablehlo.reshape %v869 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v880 = stablehlo.reshape %v878 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v881 = stablehlo.dot_general %v879, %v880, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v882 = stablehlo.reshape %v881 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v883 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v884 = stablehlo.multiply %v882, %v883 : tensor<32x38809xf32>
    %v885 = stablehlo.reshape %v884 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v886 = stablehlo.constant dense<0.0> : tensor<f32>
    %v887 = stablehlo.exponential %v885 : tensor<32x197x197xf32>
    %v888 = stablehlo.reduce(%v887 init: %v886) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v889 = stablehlo.broadcast_in_dim %v888, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v890 = stablehlo.divide %v887, %v889 : tensor<32x197x197xf32>
    %v891 = stablehlo.reshape %v890 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v892 = stablehlo.reshape %v891 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v893 = stablehlo.reshape %v875 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v894 = stablehlo.dot_general %v892, %v893, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v895 = stablehlo.reshape %v894 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v896 = stablehlo.reshape %v895 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v897 = stablehlo.constant dense<0.0> : tensor<f32>
    %v898 = stablehlo.pad %v896, %v897, low = [0, 0, 576], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v899 = stablehlo.reshape %v898 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v900 = stablehlo.add %v866, %v899 : tensor<32x151296xf32>
    %v901 = stablehlo.reshape %v551 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v902 = stablehlo.slice %v901 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v903 = stablehlo.reshape %v902 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v904 = stablehlo.reshape %v556 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v905 = stablehlo.slice %v904 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v906 = stablehlo.reshape %v905 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v907 = stablehlo.reshape %v561 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v908 = stablehlo.slice %v907 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v909 = stablehlo.reshape %v908 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v910 = stablehlo.reshape %v906 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v911 = stablehlo.transpose %v910, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v912 = stablehlo.reshape %v911 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v913 = stablehlo.reshape %v903 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v914 = stablehlo.reshape %v912 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v915 = stablehlo.dot_general %v913, %v914, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v916 = stablehlo.reshape %v915 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v917 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v918 = stablehlo.multiply %v916, %v917 : tensor<32x38809xf32>
    %v919 = stablehlo.reshape %v918 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v920 = stablehlo.constant dense<0.0> : tensor<f32>
    %v921 = stablehlo.exponential %v919 : tensor<32x197x197xf32>
    %v922 = stablehlo.reduce(%v921 init: %v920) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v923 = stablehlo.broadcast_in_dim %v922, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v924 = stablehlo.divide %v921, %v923 : tensor<32x197x197xf32>
    %v925 = stablehlo.reshape %v924 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v926 = stablehlo.reshape %v925 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v927 = stablehlo.reshape %v909 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v928 = stablehlo.dot_general %v926, %v927, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v929 = stablehlo.reshape %v928 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v930 = stablehlo.reshape %v929 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v931 = stablehlo.constant dense<0.0> : tensor<f32>
    %v932 = stablehlo.pad %v930, %v931, low = [0, 0, 640], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v933 = stablehlo.reshape %v932 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v934 = stablehlo.add %v900, %v933 : tensor<32x151296xf32>
    %v935 = stablehlo.reshape %v551 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v936 = stablehlo.slice %v935 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v937 = stablehlo.reshape %v936 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v938 = stablehlo.reshape %v556 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v939 = stablehlo.slice %v938 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v940 = stablehlo.reshape %v939 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v941 = stablehlo.reshape %v561 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v942 = stablehlo.slice %v941 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v943 = stablehlo.reshape %v942 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v944 = stablehlo.reshape %v940 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v945 = stablehlo.transpose %v944, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v946 = stablehlo.reshape %v945 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v947 = stablehlo.reshape %v937 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v948 = stablehlo.reshape %v946 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v949 = stablehlo.dot_general %v947, %v948, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v950 = stablehlo.reshape %v949 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v951 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v952 = stablehlo.multiply %v950, %v951 : tensor<32x38809xf32>
    %v953 = stablehlo.reshape %v952 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v954 = stablehlo.constant dense<0.0> : tensor<f32>
    %v955 = stablehlo.exponential %v953 : tensor<32x197x197xf32>
    %v956 = stablehlo.reduce(%v955 init: %v954) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v957 = stablehlo.broadcast_in_dim %v956, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v958 = stablehlo.divide %v955, %v957 : tensor<32x197x197xf32>
    %v959 = stablehlo.reshape %v958 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v960 = stablehlo.reshape %v959 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v961 = stablehlo.reshape %v943 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v962 = stablehlo.dot_general %v960, %v961, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v963 = stablehlo.reshape %v962 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v964 = stablehlo.reshape %v963 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v965 = stablehlo.constant dense<0.0> : tensor<f32>
    %v966 = stablehlo.pad %v964, %v965, low = [0, 0, 704], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v967 = stablehlo.reshape %v966 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v968 = stablehlo.add %v934, %v967 : tensor<32x151296xf32>
    %v969 = stablehlo.reshape %v968 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v970 = stablehlo.dot_general %v969, %b1_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v971 = stablehlo.broadcast_in_dim %b1_bo, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v972 = stablehlo.add %v970, %v971 : tensor<32x197x768xf32>
    %v973 = stablehlo.reshape %v972 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v974 = stablehlo.add %v518, %v973 : tensor<32x151296xf32>
    %v975 = stablehlo.reshape %v974 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v976 = stablehlo.constant dense<0.0> : tensor<f32>
    %v977 = stablehlo.constant dense<768.0> : tensor<32x197x768xf32>
    %v978 = stablehlo.constant dense<1.0e-5> : tensor<32x197x768xf32>
    %v979 = stablehlo.reduce(%v975 init: %v976) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v980 = stablehlo.broadcast_in_dim %v979, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v981 = stablehlo.divide %v980, %v977 : tensor<32x197x768xf32>
    %v982 = stablehlo.subtract %v975, %v981 : tensor<32x197x768xf32>
    %v983 = stablehlo.multiply %v982, %v982 : tensor<32x197x768xf32>
    %v984 = stablehlo.reduce(%v983 init: %v976) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v985 = stablehlo.broadcast_in_dim %v984, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v986 = stablehlo.divide %v985, %v977 : tensor<32x197x768xf32>
    %v987 = stablehlo.add %v986, %v978 : tensor<32x197x768xf32>
    %v988 = stablehlo.rsqrt %v987 : tensor<32x197x768xf32>
    %v989 = stablehlo.multiply %v982, %v988 : tensor<32x197x768xf32>
    %v990 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v991 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v992 = stablehlo.multiply %v989, %v990 : tensor<32x197x768xf32>
    %v993 = stablehlo.add %v992, %v991 : tensor<32x197x768xf32>
    %v994 = stablehlo.reshape %v993 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v995 = stablehlo.reshape %v994 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v996 = stablehlo.broadcast_in_dim %b1_g2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v997 = stablehlo.multiply %v995, %v996 : tensor<32x197x768xf32>
    %v998 = stablehlo.reshape %v997 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v999 = stablehlo.reshape %v998 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1000 = stablehlo.broadcast_in_dim %b1_bt2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v1001 = stablehlo.add %v999, %v1000 : tensor<32x197x768xf32>
    %v1002 = stablehlo.reshape %v1001 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1003 = stablehlo.reshape %v1002 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1004 = stablehlo.dot_general %v1003, %b1_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x3072xf32>) -> tensor<32x197x3072xf32>
    %v1005 = stablehlo.broadcast_in_dim %b1_bfc1, dims = [2] : (tensor<3072xf32>) -> tensor<32x197x3072xf32>
    %v1006 = stablehlo.add %v1004, %v1005 : tensor<32x197x3072xf32>
    %v1007 = stablehlo.reshape %v1006 : (tensor<32x197x3072xf32>) -> tensor<32x605184xf32>
    %v1008 = stablehlo.multiply %v1007, %v1007 : tensor<32x605184xf32>
    %v1009 = stablehlo.multiply %v1008, %v1007 : tensor<32x605184xf32>
    %v1010 = stablehlo.constant dense<0.044715> : tensor<32x605184xf32>
    %v1011 = stablehlo.multiply %v1010, %v1009 : tensor<32x605184xf32>
    %v1012 = stablehlo.add %v1007, %v1011 : tensor<32x605184xf32>
    %v1013 = stablehlo.constant dense<0.7978845608028654> : tensor<32x605184xf32>
    %v1014 = stablehlo.multiply %v1013, %v1012 : tensor<32x605184xf32>
    %v1015 = stablehlo.tanh %v1014 : tensor<32x605184xf32>
    %v1016 = stablehlo.constant dense<1.0> : tensor<32x605184xf32>
    %v1017 = stablehlo.add %v1016, %v1015 : tensor<32x605184xf32>
    %v1018 = stablehlo.constant dense<0.5> : tensor<32x605184xf32>
    %v1019 = stablehlo.multiply %v1018, %v1007 : tensor<32x605184xf32>
    %v1020 = stablehlo.multiply %v1019, %v1017 : tensor<32x605184xf32>
    %v1021 = stablehlo.reshape %v1020 : (tensor<32x605184xf32>) -> tensor<32x197x3072xf32>
    %v1022 = stablehlo.dot_general %v1021, %b1_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x3072xf32>, tensor<3072x768xf32>) -> tensor<32x197x768xf32>
    %v1023 = stablehlo.broadcast_in_dim %b1_bfc2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v1024 = stablehlo.add %v1022, %v1023 : tensor<32x197x768xf32>
    %v1025 = stablehlo.reshape %v1024 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1026 = stablehlo.add %v974, %v1025 : tensor<32x151296xf32>
    %v1027 = stablehlo.reshape %v1026 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1028 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1029 = stablehlo.constant dense<768.0> : tensor<32x197x768xf32>
    %v1030 = stablehlo.constant dense<1.0e-5> : tensor<32x197x768xf32>
    %v1031 = stablehlo.reduce(%v1027 init: %v1028) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1032 = stablehlo.broadcast_in_dim %v1031, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v1033 = stablehlo.divide %v1032, %v1029 : tensor<32x197x768xf32>
    %v1034 = stablehlo.subtract %v1027, %v1033 : tensor<32x197x768xf32>
    %v1035 = stablehlo.multiply %v1034, %v1034 : tensor<32x197x768xf32>
    %v1036 = stablehlo.reduce(%v1035 init: %v1028) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1037 = stablehlo.broadcast_in_dim %v1036, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v1038 = stablehlo.divide %v1037, %v1029 : tensor<32x197x768xf32>
    %v1039 = stablehlo.add %v1038, %v1030 : tensor<32x197x768xf32>
    %v1040 = stablehlo.rsqrt %v1039 : tensor<32x197x768xf32>
    %v1041 = stablehlo.multiply %v1034, %v1040 : tensor<32x197x768xf32>
    %v1042 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v1043 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v1044 = stablehlo.multiply %v1041, %v1042 : tensor<32x197x768xf32>
    %v1045 = stablehlo.add %v1044, %v1043 : tensor<32x197x768xf32>
    %v1046 = stablehlo.reshape %v1045 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1047 = stablehlo.reshape %v1046 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1048 = stablehlo.broadcast_in_dim %b2_g1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v1049 = stablehlo.multiply %v1047, %v1048 : tensor<32x197x768xf32>
    %v1050 = stablehlo.reshape %v1049 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1051 = stablehlo.reshape %v1050 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1052 = stablehlo.broadcast_in_dim %b2_bt1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v1053 = stablehlo.add %v1051, %v1052 : tensor<32x197x768xf32>
    %v1054 = stablehlo.reshape %v1053 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1055 = stablehlo.reshape %v1054 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1056 = stablehlo.dot_general %v1055, %b2_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v1057 = stablehlo.broadcast_in_dim %b2_bq, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v1058 = stablehlo.add %v1056, %v1057 : tensor<32x197x768xf32>
    %v1059 = stablehlo.reshape %v1058 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1060 = stablehlo.reshape %v1054 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1061 = stablehlo.dot_general %v1060, %b2_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v1062 = stablehlo.broadcast_in_dim %b2_bk, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v1063 = stablehlo.add %v1061, %v1062 : tensor<32x197x768xf32>
    %v1064 = stablehlo.reshape %v1063 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1065 = stablehlo.reshape %v1054 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1066 = stablehlo.dot_general %v1065, %b2_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v1067 = stablehlo.broadcast_in_dim %b2_bv, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v1068 = stablehlo.add %v1066, %v1067 : tensor<32x197x768xf32>
    %v1069 = stablehlo.reshape %v1068 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1070 = stablehlo.reshape %v1059 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1071 = stablehlo.slice %v1070 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1072 = stablehlo.reshape %v1071 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1073 = stablehlo.reshape %v1064 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1074 = stablehlo.slice %v1073 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1075 = stablehlo.reshape %v1074 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1076 = stablehlo.reshape %v1069 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1077 = stablehlo.slice %v1076 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1078 = stablehlo.reshape %v1077 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1079 = stablehlo.reshape %v1075 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1080 = stablehlo.transpose %v1079, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1081 = stablehlo.reshape %v1080 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1082 = stablehlo.reshape %v1072 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1083 = stablehlo.reshape %v1081 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1084 = stablehlo.dot_general %v1082, %v1083, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1085 = stablehlo.reshape %v1084 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1086 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1087 = stablehlo.multiply %v1085, %v1086 : tensor<32x38809xf32>
    %v1088 = stablehlo.reshape %v1087 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1089 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1090 = stablehlo.exponential %v1088 : tensor<32x197x197xf32>
    %v1091 = stablehlo.reduce(%v1090 init: %v1089) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1092 = stablehlo.broadcast_in_dim %v1091, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1093 = stablehlo.divide %v1090, %v1092 : tensor<32x197x197xf32>
    %v1094 = stablehlo.reshape %v1093 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1095 = stablehlo.reshape %v1094 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1096 = stablehlo.reshape %v1078 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1097 = stablehlo.dot_general %v1095, %v1096, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1098 = stablehlo.reshape %v1097 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1099 = stablehlo.reshape %v1098 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1100 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1101 = stablehlo.pad %v1099, %v1100, low = [0, 0, 0], high = [0, 0, 704], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v1102 = stablehlo.reshape %v1101 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1103 = stablehlo.reshape %v1059 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1104 = stablehlo.slice %v1103 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1105 = stablehlo.reshape %v1104 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1106 = stablehlo.reshape %v1064 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1107 = stablehlo.slice %v1106 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1108 = stablehlo.reshape %v1107 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1109 = stablehlo.reshape %v1069 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1110 = stablehlo.slice %v1109 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1111 = stablehlo.reshape %v1110 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1112 = stablehlo.reshape %v1108 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1113 = stablehlo.transpose %v1112, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1114 = stablehlo.reshape %v1113 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1115 = stablehlo.reshape %v1105 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1116 = stablehlo.reshape %v1114 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1117 = stablehlo.dot_general %v1115, %v1116, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1118 = stablehlo.reshape %v1117 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1119 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1120 = stablehlo.multiply %v1118, %v1119 : tensor<32x38809xf32>
    %v1121 = stablehlo.reshape %v1120 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1122 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1123 = stablehlo.exponential %v1121 : tensor<32x197x197xf32>
    %v1124 = stablehlo.reduce(%v1123 init: %v1122) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1125 = stablehlo.broadcast_in_dim %v1124, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1126 = stablehlo.divide %v1123, %v1125 : tensor<32x197x197xf32>
    %v1127 = stablehlo.reshape %v1126 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1128 = stablehlo.reshape %v1127 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1129 = stablehlo.reshape %v1111 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1130 = stablehlo.dot_general %v1128, %v1129, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1131 = stablehlo.reshape %v1130 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1132 = stablehlo.reshape %v1131 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1133 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1134 = stablehlo.pad %v1132, %v1133, low = [0, 0, 64], high = [0, 0, 640], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v1135 = stablehlo.reshape %v1134 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1136 = stablehlo.add %v1102, %v1135 : tensor<32x151296xf32>
    %v1137 = stablehlo.reshape %v1059 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1138 = stablehlo.slice %v1137 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1139 = stablehlo.reshape %v1138 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1140 = stablehlo.reshape %v1064 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1141 = stablehlo.slice %v1140 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1142 = stablehlo.reshape %v1141 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1143 = stablehlo.reshape %v1069 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1144 = stablehlo.slice %v1143 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1145 = stablehlo.reshape %v1144 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1146 = stablehlo.reshape %v1142 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1147 = stablehlo.transpose %v1146, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1148 = stablehlo.reshape %v1147 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1149 = stablehlo.reshape %v1139 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1150 = stablehlo.reshape %v1148 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1151 = stablehlo.dot_general %v1149, %v1150, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1152 = stablehlo.reshape %v1151 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1153 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1154 = stablehlo.multiply %v1152, %v1153 : tensor<32x38809xf32>
    %v1155 = stablehlo.reshape %v1154 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1156 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1157 = stablehlo.exponential %v1155 : tensor<32x197x197xf32>
    %v1158 = stablehlo.reduce(%v1157 init: %v1156) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1159 = stablehlo.broadcast_in_dim %v1158, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1160 = stablehlo.divide %v1157, %v1159 : tensor<32x197x197xf32>
    %v1161 = stablehlo.reshape %v1160 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1162 = stablehlo.reshape %v1161 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1163 = stablehlo.reshape %v1145 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1164 = stablehlo.dot_general %v1162, %v1163, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1165 = stablehlo.reshape %v1164 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1166 = stablehlo.reshape %v1165 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1167 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1168 = stablehlo.pad %v1166, %v1167, low = [0, 0, 128], high = [0, 0, 576], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v1169 = stablehlo.reshape %v1168 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1170 = stablehlo.add %v1136, %v1169 : tensor<32x151296xf32>
    %v1171 = stablehlo.reshape %v1059 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1172 = stablehlo.slice %v1171 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1173 = stablehlo.reshape %v1172 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1174 = stablehlo.reshape %v1064 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1175 = stablehlo.slice %v1174 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1176 = stablehlo.reshape %v1175 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1177 = stablehlo.reshape %v1069 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1178 = stablehlo.slice %v1177 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1179 = stablehlo.reshape %v1178 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1180 = stablehlo.reshape %v1176 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1181 = stablehlo.transpose %v1180, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1182 = stablehlo.reshape %v1181 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1183 = stablehlo.reshape %v1173 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1184 = stablehlo.reshape %v1182 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1185 = stablehlo.dot_general %v1183, %v1184, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1186 = stablehlo.reshape %v1185 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1187 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1188 = stablehlo.multiply %v1186, %v1187 : tensor<32x38809xf32>
    %v1189 = stablehlo.reshape %v1188 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1190 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1191 = stablehlo.exponential %v1189 : tensor<32x197x197xf32>
    %v1192 = stablehlo.reduce(%v1191 init: %v1190) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1193 = stablehlo.broadcast_in_dim %v1192, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1194 = stablehlo.divide %v1191, %v1193 : tensor<32x197x197xf32>
    %v1195 = stablehlo.reshape %v1194 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1196 = stablehlo.reshape %v1195 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1197 = stablehlo.reshape %v1179 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1198 = stablehlo.dot_general %v1196, %v1197, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1199 = stablehlo.reshape %v1198 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1200 = stablehlo.reshape %v1199 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1201 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1202 = stablehlo.pad %v1200, %v1201, low = [0, 0, 192], high = [0, 0, 512], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v1203 = stablehlo.reshape %v1202 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1204 = stablehlo.add %v1170, %v1203 : tensor<32x151296xf32>
    %v1205 = stablehlo.reshape %v1059 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1206 = stablehlo.slice %v1205 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1207 = stablehlo.reshape %v1206 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1208 = stablehlo.reshape %v1064 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1209 = stablehlo.slice %v1208 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1210 = stablehlo.reshape %v1209 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1211 = stablehlo.reshape %v1069 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1212 = stablehlo.slice %v1211 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1213 = stablehlo.reshape %v1212 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1214 = stablehlo.reshape %v1210 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1215 = stablehlo.transpose %v1214, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1216 = stablehlo.reshape %v1215 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1217 = stablehlo.reshape %v1207 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1218 = stablehlo.reshape %v1216 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1219 = stablehlo.dot_general %v1217, %v1218, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1220 = stablehlo.reshape %v1219 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1221 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1222 = stablehlo.multiply %v1220, %v1221 : tensor<32x38809xf32>
    %v1223 = stablehlo.reshape %v1222 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1224 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1225 = stablehlo.exponential %v1223 : tensor<32x197x197xf32>
    %v1226 = stablehlo.reduce(%v1225 init: %v1224) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1227 = stablehlo.broadcast_in_dim %v1226, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1228 = stablehlo.divide %v1225, %v1227 : tensor<32x197x197xf32>
    %v1229 = stablehlo.reshape %v1228 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1230 = stablehlo.reshape %v1229 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1231 = stablehlo.reshape %v1213 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1232 = stablehlo.dot_general %v1230, %v1231, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1233 = stablehlo.reshape %v1232 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1234 = stablehlo.reshape %v1233 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1235 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1236 = stablehlo.pad %v1234, %v1235, low = [0, 0, 256], high = [0, 0, 448], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v1237 = stablehlo.reshape %v1236 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1238 = stablehlo.add %v1204, %v1237 : tensor<32x151296xf32>
    %v1239 = stablehlo.reshape %v1059 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1240 = stablehlo.slice %v1239 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1241 = stablehlo.reshape %v1240 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1242 = stablehlo.reshape %v1064 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1243 = stablehlo.slice %v1242 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1244 = stablehlo.reshape %v1243 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1245 = stablehlo.reshape %v1069 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1246 = stablehlo.slice %v1245 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1247 = stablehlo.reshape %v1246 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1248 = stablehlo.reshape %v1244 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1249 = stablehlo.transpose %v1248, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1250 = stablehlo.reshape %v1249 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1251 = stablehlo.reshape %v1241 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1252 = stablehlo.reshape %v1250 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1253 = stablehlo.dot_general %v1251, %v1252, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1254 = stablehlo.reshape %v1253 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1255 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1256 = stablehlo.multiply %v1254, %v1255 : tensor<32x38809xf32>
    %v1257 = stablehlo.reshape %v1256 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1258 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1259 = stablehlo.exponential %v1257 : tensor<32x197x197xf32>
    %v1260 = stablehlo.reduce(%v1259 init: %v1258) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1261 = stablehlo.broadcast_in_dim %v1260, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1262 = stablehlo.divide %v1259, %v1261 : tensor<32x197x197xf32>
    %v1263 = stablehlo.reshape %v1262 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1264 = stablehlo.reshape %v1263 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1265 = stablehlo.reshape %v1247 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1266 = stablehlo.dot_general %v1264, %v1265, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1267 = stablehlo.reshape %v1266 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1268 = stablehlo.reshape %v1267 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1269 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1270 = stablehlo.pad %v1268, %v1269, low = [0, 0, 320], high = [0, 0, 384], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v1271 = stablehlo.reshape %v1270 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1272 = stablehlo.add %v1238, %v1271 : tensor<32x151296xf32>
    %v1273 = stablehlo.reshape %v1059 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1274 = stablehlo.slice %v1273 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1275 = stablehlo.reshape %v1274 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1276 = stablehlo.reshape %v1064 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1277 = stablehlo.slice %v1276 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1278 = stablehlo.reshape %v1277 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1279 = stablehlo.reshape %v1069 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1280 = stablehlo.slice %v1279 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1281 = stablehlo.reshape %v1280 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1282 = stablehlo.reshape %v1278 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1283 = stablehlo.transpose %v1282, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1284 = stablehlo.reshape %v1283 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1285 = stablehlo.reshape %v1275 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1286 = stablehlo.reshape %v1284 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1287 = stablehlo.dot_general %v1285, %v1286, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1288 = stablehlo.reshape %v1287 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1289 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1290 = stablehlo.multiply %v1288, %v1289 : tensor<32x38809xf32>
    %v1291 = stablehlo.reshape %v1290 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1292 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1293 = stablehlo.exponential %v1291 : tensor<32x197x197xf32>
    %v1294 = stablehlo.reduce(%v1293 init: %v1292) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1295 = stablehlo.broadcast_in_dim %v1294, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1296 = stablehlo.divide %v1293, %v1295 : tensor<32x197x197xf32>
    %v1297 = stablehlo.reshape %v1296 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1298 = stablehlo.reshape %v1297 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1299 = stablehlo.reshape %v1281 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1300 = stablehlo.dot_general %v1298, %v1299, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1301 = stablehlo.reshape %v1300 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1302 = stablehlo.reshape %v1301 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1303 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1304 = stablehlo.pad %v1302, %v1303, low = [0, 0, 384], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v1305 = stablehlo.reshape %v1304 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1306 = stablehlo.add %v1272, %v1305 : tensor<32x151296xf32>
    %v1307 = stablehlo.reshape %v1059 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1308 = stablehlo.slice %v1307 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1309 = stablehlo.reshape %v1308 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1310 = stablehlo.reshape %v1064 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1311 = stablehlo.slice %v1310 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1312 = stablehlo.reshape %v1311 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1313 = stablehlo.reshape %v1069 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1314 = stablehlo.slice %v1313 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1315 = stablehlo.reshape %v1314 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1316 = stablehlo.reshape %v1312 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1317 = stablehlo.transpose %v1316, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1318 = stablehlo.reshape %v1317 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1319 = stablehlo.reshape %v1309 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1320 = stablehlo.reshape %v1318 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1321 = stablehlo.dot_general %v1319, %v1320, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1322 = stablehlo.reshape %v1321 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1323 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1324 = stablehlo.multiply %v1322, %v1323 : tensor<32x38809xf32>
    %v1325 = stablehlo.reshape %v1324 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1326 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1327 = stablehlo.exponential %v1325 : tensor<32x197x197xf32>
    %v1328 = stablehlo.reduce(%v1327 init: %v1326) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1329 = stablehlo.broadcast_in_dim %v1328, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1330 = stablehlo.divide %v1327, %v1329 : tensor<32x197x197xf32>
    %v1331 = stablehlo.reshape %v1330 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1332 = stablehlo.reshape %v1331 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1333 = stablehlo.reshape %v1315 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1334 = stablehlo.dot_general %v1332, %v1333, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1335 = stablehlo.reshape %v1334 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1336 = stablehlo.reshape %v1335 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1337 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1338 = stablehlo.pad %v1336, %v1337, low = [0, 0, 448], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v1339 = stablehlo.reshape %v1338 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1340 = stablehlo.add %v1306, %v1339 : tensor<32x151296xf32>
    %v1341 = stablehlo.reshape %v1059 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1342 = stablehlo.slice %v1341 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1343 = stablehlo.reshape %v1342 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1344 = stablehlo.reshape %v1064 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1345 = stablehlo.slice %v1344 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1346 = stablehlo.reshape %v1345 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1347 = stablehlo.reshape %v1069 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1348 = stablehlo.slice %v1347 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1349 = stablehlo.reshape %v1348 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1350 = stablehlo.reshape %v1346 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1351 = stablehlo.transpose %v1350, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1352 = stablehlo.reshape %v1351 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1353 = stablehlo.reshape %v1343 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1354 = stablehlo.reshape %v1352 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1355 = stablehlo.dot_general %v1353, %v1354, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1356 = stablehlo.reshape %v1355 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1357 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1358 = stablehlo.multiply %v1356, %v1357 : tensor<32x38809xf32>
    %v1359 = stablehlo.reshape %v1358 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1360 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1361 = stablehlo.exponential %v1359 : tensor<32x197x197xf32>
    %v1362 = stablehlo.reduce(%v1361 init: %v1360) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1363 = stablehlo.broadcast_in_dim %v1362, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1364 = stablehlo.divide %v1361, %v1363 : tensor<32x197x197xf32>
    %v1365 = stablehlo.reshape %v1364 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1366 = stablehlo.reshape %v1365 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1367 = stablehlo.reshape %v1349 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1368 = stablehlo.dot_general %v1366, %v1367, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1369 = stablehlo.reshape %v1368 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1370 = stablehlo.reshape %v1369 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1371 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1372 = stablehlo.pad %v1370, %v1371, low = [0, 0, 512], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v1373 = stablehlo.reshape %v1372 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1374 = stablehlo.add %v1340, %v1373 : tensor<32x151296xf32>
    %v1375 = stablehlo.reshape %v1059 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1376 = stablehlo.slice %v1375 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1377 = stablehlo.reshape %v1376 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1378 = stablehlo.reshape %v1064 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1379 = stablehlo.slice %v1378 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1380 = stablehlo.reshape %v1379 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1381 = stablehlo.reshape %v1069 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1382 = stablehlo.slice %v1381 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1383 = stablehlo.reshape %v1382 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1384 = stablehlo.reshape %v1380 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1385 = stablehlo.transpose %v1384, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1386 = stablehlo.reshape %v1385 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1387 = stablehlo.reshape %v1377 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1388 = stablehlo.reshape %v1386 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1389 = stablehlo.dot_general %v1387, %v1388, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1390 = stablehlo.reshape %v1389 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1391 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1392 = stablehlo.multiply %v1390, %v1391 : tensor<32x38809xf32>
    %v1393 = stablehlo.reshape %v1392 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1394 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1395 = stablehlo.exponential %v1393 : tensor<32x197x197xf32>
    %v1396 = stablehlo.reduce(%v1395 init: %v1394) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1397 = stablehlo.broadcast_in_dim %v1396, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1398 = stablehlo.divide %v1395, %v1397 : tensor<32x197x197xf32>
    %v1399 = stablehlo.reshape %v1398 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1400 = stablehlo.reshape %v1399 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1401 = stablehlo.reshape %v1383 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1402 = stablehlo.dot_general %v1400, %v1401, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1403 = stablehlo.reshape %v1402 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1404 = stablehlo.reshape %v1403 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1405 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1406 = stablehlo.pad %v1404, %v1405, low = [0, 0, 576], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v1407 = stablehlo.reshape %v1406 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1408 = stablehlo.add %v1374, %v1407 : tensor<32x151296xf32>
    %v1409 = stablehlo.reshape %v1059 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1410 = stablehlo.slice %v1409 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1411 = stablehlo.reshape %v1410 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1412 = stablehlo.reshape %v1064 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1413 = stablehlo.slice %v1412 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1414 = stablehlo.reshape %v1413 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1415 = stablehlo.reshape %v1069 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1416 = stablehlo.slice %v1415 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1417 = stablehlo.reshape %v1416 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1418 = stablehlo.reshape %v1414 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1419 = stablehlo.transpose %v1418, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1420 = stablehlo.reshape %v1419 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1421 = stablehlo.reshape %v1411 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1422 = stablehlo.reshape %v1420 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1423 = stablehlo.dot_general %v1421, %v1422, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1424 = stablehlo.reshape %v1423 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1425 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1426 = stablehlo.multiply %v1424, %v1425 : tensor<32x38809xf32>
    %v1427 = stablehlo.reshape %v1426 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1428 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1429 = stablehlo.exponential %v1427 : tensor<32x197x197xf32>
    %v1430 = stablehlo.reduce(%v1429 init: %v1428) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1431 = stablehlo.broadcast_in_dim %v1430, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1432 = stablehlo.divide %v1429, %v1431 : tensor<32x197x197xf32>
    %v1433 = stablehlo.reshape %v1432 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1434 = stablehlo.reshape %v1433 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1435 = stablehlo.reshape %v1417 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1436 = stablehlo.dot_general %v1434, %v1435, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1437 = stablehlo.reshape %v1436 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1438 = stablehlo.reshape %v1437 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1439 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1440 = stablehlo.pad %v1438, %v1439, low = [0, 0, 640], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v1441 = stablehlo.reshape %v1440 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1442 = stablehlo.add %v1408, %v1441 : tensor<32x151296xf32>
    %v1443 = stablehlo.reshape %v1059 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1444 = stablehlo.slice %v1443 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1445 = stablehlo.reshape %v1444 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1446 = stablehlo.reshape %v1064 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1447 = stablehlo.slice %v1446 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1448 = stablehlo.reshape %v1447 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1449 = stablehlo.reshape %v1069 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1450 = stablehlo.slice %v1449 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1451 = stablehlo.reshape %v1450 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1452 = stablehlo.reshape %v1448 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1453 = stablehlo.transpose %v1452, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1454 = stablehlo.reshape %v1453 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1455 = stablehlo.reshape %v1445 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1456 = stablehlo.reshape %v1454 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1457 = stablehlo.dot_general %v1455, %v1456, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1458 = stablehlo.reshape %v1457 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1459 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1460 = stablehlo.multiply %v1458, %v1459 : tensor<32x38809xf32>
    %v1461 = stablehlo.reshape %v1460 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1462 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1463 = stablehlo.exponential %v1461 : tensor<32x197x197xf32>
    %v1464 = stablehlo.reduce(%v1463 init: %v1462) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1465 = stablehlo.broadcast_in_dim %v1464, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1466 = stablehlo.divide %v1463, %v1465 : tensor<32x197x197xf32>
    %v1467 = stablehlo.reshape %v1466 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1468 = stablehlo.reshape %v1467 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1469 = stablehlo.reshape %v1451 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1470 = stablehlo.dot_general %v1468, %v1469, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1471 = stablehlo.reshape %v1470 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1472 = stablehlo.reshape %v1471 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1473 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1474 = stablehlo.pad %v1472, %v1473, low = [0, 0, 704], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v1475 = stablehlo.reshape %v1474 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1476 = stablehlo.add %v1442, %v1475 : tensor<32x151296xf32>
    %v1477 = stablehlo.reshape %v1476 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1478 = stablehlo.dot_general %v1477, %b2_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v1479 = stablehlo.broadcast_in_dim %b2_bo, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v1480 = stablehlo.add %v1478, %v1479 : tensor<32x197x768xf32>
    %v1481 = stablehlo.reshape %v1480 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1482 = stablehlo.add %v1026, %v1481 : tensor<32x151296xf32>
    %v1483 = stablehlo.reshape %v1482 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1484 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1485 = stablehlo.constant dense<768.0> : tensor<32x197x768xf32>
    %v1486 = stablehlo.constant dense<1.0e-5> : tensor<32x197x768xf32>
    %v1487 = stablehlo.reduce(%v1483 init: %v1484) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1488 = stablehlo.broadcast_in_dim %v1487, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v1489 = stablehlo.divide %v1488, %v1485 : tensor<32x197x768xf32>
    %v1490 = stablehlo.subtract %v1483, %v1489 : tensor<32x197x768xf32>
    %v1491 = stablehlo.multiply %v1490, %v1490 : tensor<32x197x768xf32>
    %v1492 = stablehlo.reduce(%v1491 init: %v1484) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1493 = stablehlo.broadcast_in_dim %v1492, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v1494 = stablehlo.divide %v1493, %v1485 : tensor<32x197x768xf32>
    %v1495 = stablehlo.add %v1494, %v1486 : tensor<32x197x768xf32>
    %v1496 = stablehlo.rsqrt %v1495 : tensor<32x197x768xf32>
    %v1497 = stablehlo.multiply %v1490, %v1496 : tensor<32x197x768xf32>
    %v1498 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v1499 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v1500 = stablehlo.multiply %v1497, %v1498 : tensor<32x197x768xf32>
    %v1501 = stablehlo.add %v1500, %v1499 : tensor<32x197x768xf32>
    %v1502 = stablehlo.reshape %v1501 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1503 = stablehlo.reshape %v1502 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1504 = stablehlo.broadcast_in_dim %b2_g2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v1505 = stablehlo.multiply %v1503, %v1504 : tensor<32x197x768xf32>
    %v1506 = stablehlo.reshape %v1505 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1507 = stablehlo.reshape %v1506 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1508 = stablehlo.broadcast_in_dim %b2_bt2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v1509 = stablehlo.add %v1507, %v1508 : tensor<32x197x768xf32>
    %v1510 = stablehlo.reshape %v1509 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1511 = stablehlo.reshape %v1510 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1512 = stablehlo.dot_general %v1511, %b2_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x3072xf32>) -> tensor<32x197x3072xf32>
    %v1513 = stablehlo.broadcast_in_dim %b2_bfc1, dims = [2] : (tensor<3072xf32>) -> tensor<32x197x3072xf32>
    %v1514 = stablehlo.add %v1512, %v1513 : tensor<32x197x3072xf32>
    %v1515 = stablehlo.reshape %v1514 : (tensor<32x197x3072xf32>) -> tensor<32x605184xf32>
    %v1516 = stablehlo.multiply %v1515, %v1515 : tensor<32x605184xf32>
    %v1517 = stablehlo.multiply %v1516, %v1515 : tensor<32x605184xf32>
    %v1518 = stablehlo.constant dense<0.044715> : tensor<32x605184xf32>
    %v1519 = stablehlo.multiply %v1518, %v1517 : tensor<32x605184xf32>
    %v1520 = stablehlo.add %v1515, %v1519 : tensor<32x605184xf32>
    %v1521 = stablehlo.constant dense<0.7978845608028654> : tensor<32x605184xf32>
    %v1522 = stablehlo.multiply %v1521, %v1520 : tensor<32x605184xf32>
    %v1523 = stablehlo.tanh %v1522 : tensor<32x605184xf32>
    %v1524 = stablehlo.constant dense<1.0> : tensor<32x605184xf32>
    %v1525 = stablehlo.add %v1524, %v1523 : tensor<32x605184xf32>
    %v1526 = stablehlo.constant dense<0.5> : tensor<32x605184xf32>
    %v1527 = stablehlo.multiply %v1526, %v1515 : tensor<32x605184xf32>
    %v1528 = stablehlo.multiply %v1527, %v1525 : tensor<32x605184xf32>
    %v1529 = stablehlo.reshape %v1528 : (tensor<32x605184xf32>) -> tensor<32x197x3072xf32>
    %v1530 = stablehlo.dot_general %v1529, %b2_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x3072xf32>, tensor<3072x768xf32>) -> tensor<32x197x768xf32>
    %v1531 = stablehlo.broadcast_in_dim %b2_bfc2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v1532 = stablehlo.add %v1530, %v1531 : tensor<32x197x768xf32>
    %v1533 = stablehlo.reshape %v1532 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1534 = stablehlo.add %v1482, %v1533 : tensor<32x151296xf32>
    %v1535 = stablehlo.reshape %v1534 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1536 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1537 = stablehlo.constant dense<768.0> : tensor<32x197x768xf32>
    %v1538 = stablehlo.constant dense<1.0e-5> : tensor<32x197x768xf32>
    %v1539 = stablehlo.reduce(%v1535 init: %v1536) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1540 = stablehlo.broadcast_in_dim %v1539, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v1541 = stablehlo.divide %v1540, %v1537 : tensor<32x197x768xf32>
    %v1542 = stablehlo.subtract %v1535, %v1541 : tensor<32x197x768xf32>
    %v1543 = stablehlo.multiply %v1542, %v1542 : tensor<32x197x768xf32>
    %v1544 = stablehlo.reduce(%v1543 init: %v1536) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1545 = stablehlo.broadcast_in_dim %v1544, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v1546 = stablehlo.divide %v1545, %v1537 : tensor<32x197x768xf32>
    %v1547 = stablehlo.add %v1546, %v1538 : tensor<32x197x768xf32>
    %v1548 = stablehlo.rsqrt %v1547 : tensor<32x197x768xf32>
    %v1549 = stablehlo.multiply %v1542, %v1548 : tensor<32x197x768xf32>
    %v1550 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v1551 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v1552 = stablehlo.multiply %v1549, %v1550 : tensor<32x197x768xf32>
    %v1553 = stablehlo.add %v1552, %v1551 : tensor<32x197x768xf32>
    %v1554 = stablehlo.reshape %v1553 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1555 = stablehlo.reshape %v1554 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1556 = stablehlo.broadcast_in_dim %b3_g1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v1557 = stablehlo.multiply %v1555, %v1556 : tensor<32x197x768xf32>
    %v1558 = stablehlo.reshape %v1557 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1559 = stablehlo.reshape %v1558 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1560 = stablehlo.broadcast_in_dim %b3_bt1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v1561 = stablehlo.add %v1559, %v1560 : tensor<32x197x768xf32>
    %v1562 = stablehlo.reshape %v1561 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1563 = stablehlo.reshape %v1562 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1564 = stablehlo.dot_general %v1563, %b3_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v1565 = stablehlo.broadcast_in_dim %b3_bq, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v1566 = stablehlo.add %v1564, %v1565 : tensor<32x197x768xf32>
    %v1567 = stablehlo.reshape %v1566 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1568 = stablehlo.reshape %v1562 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1569 = stablehlo.dot_general %v1568, %b3_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v1570 = stablehlo.broadcast_in_dim %b3_bk, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v1571 = stablehlo.add %v1569, %v1570 : tensor<32x197x768xf32>
    %v1572 = stablehlo.reshape %v1571 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1573 = stablehlo.reshape %v1562 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1574 = stablehlo.dot_general %v1573, %b3_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v1575 = stablehlo.broadcast_in_dim %b3_bv, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v1576 = stablehlo.add %v1574, %v1575 : tensor<32x197x768xf32>
    %v1577 = stablehlo.reshape %v1576 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1578 = stablehlo.reshape %v1567 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1579 = stablehlo.slice %v1578 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1580 = stablehlo.reshape %v1579 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1581 = stablehlo.reshape %v1572 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1582 = stablehlo.slice %v1581 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1583 = stablehlo.reshape %v1582 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1584 = stablehlo.reshape %v1577 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1585 = stablehlo.slice %v1584 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1586 = stablehlo.reshape %v1585 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1587 = stablehlo.reshape %v1583 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1588 = stablehlo.transpose %v1587, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1589 = stablehlo.reshape %v1588 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1590 = stablehlo.reshape %v1580 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1591 = stablehlo.reshape %v1589 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1592 = stablehlo.dot_general %v1590, %v1591, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1593 = stablehlo.reshape %v1592 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1594 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1595 = stablehlo.multiply %v1593, %v1594 : tensor<32x38809xf32>
    %v1596 = stablehlo.reshape %v1595 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1597 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1598 = stablehlo.exponential %v1596 : tensor<32x197x197xf32>
    %v1599 = stablehlo.reduce(%v1598 init: %v1597) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1600 = stablehlo.broadcast_in_dim %v1599, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1601 = stablehlo.divide %v1598, %v1600 : tensor<32x197x197xf32>
    %v1602 = stablehlo.reshape %v1601 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1603 = stablehlo.reshape %v1602 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1604 = stablehlo.reshape %v1586 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1605 = stablehlo.dot_general %v1603, %v1604, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1606 = stablehlo.reshape %v1605 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1607 = stablehlo.reshape %v1606 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1608 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1609 = stablehlo.pad %v1607, %v1608, low = [0, 0, 0], high = [0, 0, 704], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v1610 = stablehlo.reshape %v1609 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1611 = stablehlo.reshape %v1567 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1612 = stablehlo.slice %v1611 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1613 = stablehlo.reshape %v1612 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1614 = stablehlo.reshape %v1572 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1615 = stablehlo.slice %v1614 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1616 = stablehlo.reshape %v1615 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1617 = stablehlo.reshape %v1577 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1618 = stablehlo.slice %v1617 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1619 = stablehlo.reshape %v1618 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1620 = stablehlo.reshape %v1616 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1621 = stablehlo.transpose %v1620, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1622 = stablehlo.reshape %v1621 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1623 = stablehlo.reshape %v1613 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1624 = stablehlo.reshape %v1622 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1625 = stablehlo.dot_general %v1623, %v1624, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1626 = stablehlo.reshape %v1625 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1627 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1628 = stablehlo.multiply %v1626, %v1627 : tensor<32x38809xf32>
    %v1629 = stablehlo.reshape %v1628 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1630 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1631 = stablehlo.exponential %v1629 : tensor<32x197x197xf32>
    %v1632 = stablehlo.reduce(%v1631 init: %v1630) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1633 = stablehlo.broadcast_in_dim %v1632, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1634 = stablehlo.divide %v1631, %v1633 : tensor<32x197x197xf32>
    %v1635 = stablehlo.reshape %v1634 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1636 = stablehlo.reshape %v1635 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1637 = stablehlo.reshape %v1619 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1638 = stablehlo.dot_general %v1636, %v1637, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1639 = stablehlo.reshape %v1638 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1640 = stablehlo.reshape %v1639 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1641 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1642 = stablehlo.pad %v1640, %v1641, low = [0, 0, 64], high = [0, 0, 640], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v1643 = stablehlo.reshape %v1642 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1644 = stablehlo.add %v1610, %v1643 : tensor<32x151296xf32>
    %v1645 = stablehlo.reshape %v1567 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1646 = stablehlo.slice %v1645 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1647 = stablehlo.reshape %v1646 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1648 = stablehlo.reshape %v1572 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1649 = stablehlo.slice %v1648 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1650 = stablehlo.reshape %v1649 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1651 = stablehlo.reshape %v1577 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1652 = stablehlo.slice %v1651 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1653 = stablehlo.reshape %v1652 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1654 = stablehlo.reshape %v1650 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1655 = stablehlo.transpose %v1654, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1656 = stablehlo.reshape %v1655 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1657 = stablehlo.reshape %v1647 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1658 = stablehlo.reshape %v1656 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1659 = stablehlo.dot_general %v1657, %v1658, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1660 = stablehlo.reshape %v1659 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1661 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1662 = stablehlo.multiply %v1660, %v1661 : tensor<32x38809xf32>
    %v1663 = stablehlo.reshape %v1662 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1664 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1665 = stablehlo.exponential %v1663 : tensor<32x197x197xf32>
    %v1666 = stablehlo.reduce(%v1665 init: %v1664) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1667 = stablehlo.broadcast_in_dim %v1666, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1668 = stablehlo.divide %v1665, %v1667 : tensor<32x197x197xf32>
    %v1669 = stablehlo.reshape %v1668 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1670 = stablehlo.reshape %v1669 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1671 = stablehlo.reshape %v1653 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1672 = stablehlo.dot_general %v1670, %v1671, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1673 = stablehlo.reshape %v1672 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1674 = stablehlo.reshape %v1673 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1675 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1676 = stablehlo.pad %v1674, %v1675, low = [0, 0, 128], high = [0, 0, 576], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v1677 = stablehlo.reshape %v1676 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1678 = stablehlo.add %v1644, %v1677 : tensor<32x151296xf32>
    %v1679 = stablehlo.reshape %v1567 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1680 = stablehlo.slice %v1679 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1681 = stablehlo.reshape %v1680 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1682 = stablehlo.reshape %v1572 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1683 = stablehlo.slice %v1682 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1684 = stablehlo.reshape %v1683 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1685 = stablehlo.reshape %v1577 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1686 = stablehlo.slice %v1685 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1687 = stablehlo.reshape %v1686 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1688 = stablehlo.reshape %v1684 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1689 = stablehlo.transpose %v1688, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1690 = stablehlo.reshape %v1689 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1691 = stablehlo.reshape %v1681 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1692 = stablehlo.reshape %v1690 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1693 = stablehlo.dot_general %v1691, %v1692, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1694 = stablehlo.reshape %v1693 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1695 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1696 = stablehlo.multiply %v1694, %v1695 : tensor<32x38809xf32>
    %v1697 = stablehlo.reshape %v1696 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1698 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1699 = stablehlo.exponential %v1697 : tensor<32x197x197xf32>
    %v1700 = stablehlo.reduce(%v1699 init: %v1698) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1701 = stablehlo.broadcast_in_dim %v1700, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1702 = stablehlo.divide %v1699, %v1701 : tensor<32x197x197xf32>
    %v1703 = stablehlo.reshape %v1702 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1704 = stablehlo.reshape %v1703 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1705 = stablehlo.reshape %v1687 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1706 = stablehlo.dot_general %v1704, %v1705, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1707 = stablehlo.reshape %v1706 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1708 = stablehlo.reshape %v1707 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1709 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1710 = stablehlo.pad %v1708, %v1709, low = [0, 0, 192], high = [0, 0, 512], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v1711 = stablehlo.reshape %v1710 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1712 = stablehlo.add %v1678, %v1711 : tensor<32x151296xf32>
    %v1713 = stablehlo.reshape %v1567 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1714 = stablehlo.slice %v1713 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1715 = stablehlo.reshape %v1714 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1716 = stablehlo.reshape %v1572 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1717 = stablehlo.slice %v1716 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1718 = stablehlo.reshape %v1717 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1719 = stablehlo.reshape %v1577 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1720 = stablehlo.slice %v1719 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1721 = stablehlo.reshape %v1720 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1722 = stablehlo.reshape %v1718 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1723 = stablehlo.transpose %v1722, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1724 = stablehlo.reshape %v1723 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1725 = stablehlo.reshape %v1715 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1726 = stablehlo.reshape %v1724 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1727 = stablehlo.dot_general %v1725, %v1726, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1728 = stablehlo.reshape %v1727 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1729 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1730 = stablehlo.multiply %v1728, %v1729 : tensor<32x38809xf32>
    %v1731 = stablehlo.reshape %v1730 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1732 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1733 = stablehlo.exponential %v1731 : tensor<32x197x197xf32>
    %v1734 = stablehlo.reduce(%v1733 init: %v1732) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1735 = stablehlo.broadcast_in_dim %v1734, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1736 = stablehlo.divide %v1733, %v1735 : tensor<32x197x197xf32>
    %v1737 = stablehlo.reshape %v1736 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1738 = stablehlo.reshape %v1737 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1739 = stablehlo.reshape %v1721 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1740 = stablehlo.dot_general %v1738, %v1739, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1741 = stablehlo.reshape %v1740 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1742 = stablehlo.reshape %v1741 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1743 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1744 = stablehlo.pad %v1742, %v1743, low = [0, 0, 256], high = [0, 0, 448], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v1745 = stablehlo.reshape %v1744 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1746 = stablehlo.add %v1712, %v1745 : tensor<32x151296xf32>
    %v1747 = stablehlo.reshape %v1567 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1748 = stablehlo.slice %v1747 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1749 = stablehlo.reshape %v1748 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1750 = stablehlo.reshape %v1572 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1751 = stablehlo.slice %v1750 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1752 = stablehlo.reshape %v1751 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1753 = stablehlo.reshape %v1577 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1754 = stablehlo.slice %v1753 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1755 = stablehlo.reshape %v1754 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1756 = stablehlo.reshape %v1752 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1757 = stablehlo.transpose %v1756, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1758 = stablehlo.reshape %v1757 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1759 = stablehlo.reshape %v1749 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1760 = stablehlo.reshape %v1758 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1761 = stablehlo.dot_general %v1759, %v1760, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1762 = stablehlo.reshape %v1761 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1763 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1764 = stablehlo.multiply %v1762, %v1763 : tensor<32x38809xf32>
    %v1765 = stablehlo.reshape %v1764 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1766 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1767 = stablehlo.exponential %v1765 : tensor<32x197x197xf32>
    %v1768 = stablehlo.reduce(%v1767 init: %v1766) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1769 = stablehlo.broadcast_in_dim %v1768, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1770 = stablehlo.divide %v1767, %v1769 : tensor<32x197x197xf32>
    %v1771 = stablehlo.reshape %v1770 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1772 = stablehlo.reshape %v1771 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1773 = stablehlo.reshape %v1755 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1774 = stablehlo.dot_general %v1772, %v1773, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1775 = stablehlo.reshape %v1774 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1776 = stablehlo.reshape %v1775 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1777 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1778 = stablehlo.pad %v1776, %v1777, low = [0, 0, 320], high = [0, 0, 384], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v1779 = stablehlo.reshape %v1778 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1780 = stablehlo.add %v1746, %v1779 : tensor<32x151296xf32>
    %v1781 = stablehlo.reshape %v1567 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1782 = stablehlo.slice %v1781 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1783 = stablehlo.reshape %v1782 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1784 = stablehlo.reshape %v1572 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1785 = stablehlo.slice %v1784 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1786 = stablehlo.reshape %v1785 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1787 = stablehlo.reshape %v1577 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1788 = stablehlo.slice %v1787 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1789 = stablehlo.reshape %v1788 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1790 = stablehlo.reshape %v1786 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1791 = stablehlo.transpose %v1790, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1792 = stablehlo.reshape %v1791 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1793 = stablehlo.reshape %v1783 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1794 = stablehlo.reshape %v1792 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1795 = stablehlo.dot_general %v1793, %v1794, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1796 = stablehlo.reshape %v1795 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1797 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1798 = stablehlo.multiply %v1796, %v1797 : tensor<32x38809xf32>
    %v1799 = stablehlo.reshape %v1798 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1800 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1801 = stablehlo.exponential %v1799 : tensor<32x197x197xf32>
    %v1802 = stablehlo.reduce(%v1801 init: %v1800) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1803 = stablehlo.broadcast_in_dim %v1802, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1804 = stablehlo.divide %v1801, %v1803 : tensor<32x197x197xf32>
    %v1805 = stablehlo.reshape %v1804 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1806 = stablehlo.reshape %v1805 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1807 = stablehlo.reshape %v1789 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1808 = stablehlo.dot_general %v1806, %v1807, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1809 = stablehlo.reshape %v1808 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1810 = stablehlo.reshape %v1809 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1811 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1812 = stablehlo.pad %v1810, %v1811, low = [0, 0, 384], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v1813 = stablehlo.reshape %v1812 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1814 = stablehlo.add %v1780, %v1813 : tensor<32x151296xf32>
    %v1815 = stablehlo.reshape %v1567 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1816 = stablehlo.slice %v1815 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1817 = stablehlo.reshape %v1816 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1818 = stablehlo.reshape %v1572 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1819 = stablehlo.slice %v1818 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1820 = stablehlo.reshape %v1819 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1821 = stablehlo.reshape %v1577 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1822 = stablehlo.slice %v1821 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1823 = stablehlo.reshape %v1822 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1824 = stablehlo.reshape %v1820 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1825 = stablehlo.transpose %v1824, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1826 = stablehlo.reshape %v1825 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1827 = stablehlo.reshape %v1817 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1828 = stablehlo.reshape %v1826 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1829 = stablehlo.dot_general %v1827, %v1828, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1830 = stablehlo.reshape %v1829 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1831 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1832 = stablehlo.multiply %v1830, %v1831 : tensor<32x38809xf32>
    %v1833 = stablehlo.reshape %v1832 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1834 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1835 = stablehlo.exponential %v1833 : tensor<32x197x197xf32>
    %v1836 = stablehlo.reduce(%v1835 init: %v1834) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1837 = stablehlo.broadcast_in_dim %v1836, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1838 = stablehlo.divide %v1835, %v1837 : tensor<32x197x197xf32>
    %v1839 = stablehlo.reshape %v1838 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1840 = stablehlo.reshape %v1839 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1841 = stablehlo.reshape %v1823 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1842 = stablehlo.dot_general %v1840, %v1841, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1843 = stablehlo.reshape %v1842 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1844 = stablehlo.reshape %v1843 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1845 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1846 = stablehlo.pad %v1844, %v1845, low = [0, 0, 448], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v1847 = stablehlo.reshape %v1846 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1848 = stablehlo.add %v1814, %v1847 : tensor<32x151296xf32>
    %v1849 = stablehlo.reshape %v1567 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1850 = stablehlo.slice %v1849 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1851 = stablehlo.reshape %v1850 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1852 = stablehlo.reshape %v1572 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1853 = stablehlo.slice %v1852 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1854 = stablehlo.reshape %v1853 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1855 = stablehlo.reshape %v1577 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1856 = stablehlo.slice %v1855 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1857 = stablehlo.reshape %v1856 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1858 = stablehlo.reshape %v1854 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1859 = stablehlo.transpose %v1858, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1860 = stablehlo.reshape %v1859 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1861 = stablehlo.reshape %v1851 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1862 = stablehlo.reshape %v1860 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1863 = stablehlo.dot_general %v1861, %v1862, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1864 = stablehlo.reshape %v1863 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1865 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1866 = stablehlo.multiply %v1864, %v1865 : tensor<32x38809xf32>
    %v1867 = stablehlo.reshape %v1866 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1868 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1869 = stablehlo.exponential %v1867 : tensor<32x197x197xf32>
    %v1870 = stablehlo.reduce(%v1869 init: %v1868) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1871 = stablehlo.broadcast_in_dim %v1870, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1872 = stablehlo.divide %v1869, %v1871 : tensor<32x197x197xf32>
    %v1873 = stablehlo.reshape %v1872 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1874 = stablehlo.reshape %v1873 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1875 = stablehlo.reshape %v1857 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1876 = stablehlo.dot_general %v1874, %v1875, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1877 = stablehlo.reshape %v1876 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1878 = stablehlo.reshape %v1877 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1879 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1880 = stablehlo.pad %v1878, %v1879, low = [0, 0, 512], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v1881 = stablehlo.reshape %v1880 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1882 = stablehlo.add %v1848, %v1881 : tensor<32x151296xf32>
    %v1883 = stablehlo.reshape %v1567 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1884 = stablehlo.slice %v1883 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1885 = stablehlo.reshape %v1884 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1886 = stablehlo.reshape %v1572 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1887 = stablehlo.slice %v1886 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1888 = stablehlo.reshape %v1887 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1889 = stablehlo.reshape %v1577 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1890 = stablehlo.slice %v1889 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1891 = stablehlo.reshape %v1890 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1892 = stablehlo.reshape %v1888 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1893 = stablehlo.transpose %v1892, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1894 = stablehlo.reshape %v1893 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1895 = stablehlo.reshape %v1885 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1896 = stablehlo.reshape %v1894 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1897 = stablehlo.dot_general %v1895, %v1896, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1898 = stablehlo.reshape %v1897 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1899 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1900 = stablehlo.multiply %v1898, %v1899 : tensor<32x38809xf32>
    %v1901 = stablehlo.reshape %v1900 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1902 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1903 = stablehlo.exponential %v1901 : tensor<32x197x197xf32>
    %v1904 = stablehlo.reduce(%v1903 init: %v1902) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1905 = stablehlo.broadcast_in_dim %v1904, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1906 = stablehlo.divide %v1903, %v1905 : tensor<32x197x197xf32>
    %v1907 = stablehlo.reshape %v1906 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1908 = stablehlo.reshape %v1907 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1909 = stablehlo.reshape %v1891 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1910 = stablehlo.dot_general %v1908, %v1909, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1911 = stablehlo.reshape %v1910 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1912 = stablehlo.reshape %v1911 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1913 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1914 = stablehlo.pad %v1912, %v1913, low = [0, 0, 576], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v1915 = stablehlo.reshape %v1914 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1916 = stablehlo.add %v1882, %v1915 : tensor<32x151296xf32>
    %v1917 = stablehlo.reshape %v1567 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1918 = stablehlo.slice %v1917 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1919 = stablehlo.reshape %v1918 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1920 = stablehlo.reshape %v1572 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1921 = stablehlo.slice %v1920 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1922 = stablehlo.reshape %v1921 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1923 = stablehlo.reshape %v1577 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1924 = stablehlo.slice %v1923 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1925 = stablehlo.reshape %v1924 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1926 = stablehlo.reshape %v1922 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1927 = stablehlo.transpose %v1926, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1928 = stablehlo.reshape %v1927 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1929 = stablehlo.reshape %v1919 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1930 = stablehlo.reshape %v1928 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1931 = stablehlo.dot_general %v1929, %v1930, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1932 = stablehlo.reshape %v1931 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1933 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1934 = stablehlo.multiply %v1932, %v1933 : tensor<32x38809xf32>
    %v1935 = stablehlo.reshape %v1934 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1936 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1937 = stablehlo.exponential %v1935 : tensor<32x197x197xf32>
    %v1938 = stablehlo.reduce(%v1937 init: %v1936) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1939 = stablehlo.broadcast_in_dim %v1938, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1940 = stablehlo.divide %v1937, %v1939 : tensor<32x197x197xf32>
    %v1941 = stablehlo.reshape %v1940 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1942 = stablehlo.reshape %v1941 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1943 = stablehlo.reshape %v1925 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1944 = stablehlo.dot_general %v1942, %v1943, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1945 = stablehlo.reshape %v1944 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1946 = stablehlo.reshape %v1945 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1947 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1948 = stablehlo.pad %v1946, %v1947, low = [0, 0, 640], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v1949 = stablehlo.reshape %v1948 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1950 = stablehlo.add %v1916, %v1949 : tensor<32x151296xf32>
    %v1951 = stablehlo.reshape %v1567 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1952 = stablehlo.slice %v1951 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1953 = stablehlo.reshape %v1952 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1954 = stablehlo.reshape %v1572 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1955 = stablehlo.slice %v1954 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1956 = stablehlo.reshape %v1955 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1957 = stablehlo.reshape %v1577 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1958 = stablehlo.slice %v1957 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v1959 = stablehlo.reshape %v1958 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1960 = stablehlo.reshape %v1956 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1961 = stablehlo.transpose %v1960, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1962 = stablehlo.reshape %v1961 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1963 = stablehlo.reshape %v1953 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1964 = stablehlo.reshape %v1962 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1965 = stablehlo.dot_general %v1963, %v1964, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1966 = stablehlo.reshape %v1965 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1967 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1968 = stablehlo.multiply %v1966, %v1967 : tensor<32x38809xf32>
    %v1969 = stablehlo.reshape %v1968 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1970 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1971 = stablehlo.exponential %v1969 : tensor<32x197x197xf32>
    %v1972 = stablehlo.reduce(%v1971 init: %v1970) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1973 = stablehlo.broadcast_in_dim %v1972, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1974 = stablehlo.divide %v1971, %v1973 : tensor<32x197x197xf32>
    %v1975 = stablehlo.reshape %v1974 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1976 = stablehlo.reshape %v1975 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1977 = stablehlo.reshape %v1959 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1978 = stablehlo.dot_general %v1976, %v1977, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1979 = stablehlo.reshape %v1978 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1980 = stablehlo.reshape %v1979 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1981 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1982 = stablehlo.pad %v1980, %v1981, low = [0, 0, 704], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v1983 = stablehlo.reshape %v1982 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1984 = stablehlo.add %v1950, %v1983 : tensor<32x151296xf32>
    %v1985 = stablehlo.reshape %v1984 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1986 = stablehlo.dot_general %v1985, %b3_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v1987 = stablehlo.broadcast_in_dim %b3_bo, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v1988 = stablehlo.add %v1986, %v1987 : tensor<32x197x768xf32>
    %v1989 = stablehlo.reshape %v1988 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1990 = stablehlo.add %v1534, %v1989 : tensor<32x151296xf32>
    %v1991 = stablehlo.reshape %v1990 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1992 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1993 = stablehlo.constant dense<768.0> : tensor<32x197x768xf32>
    %v1994 = stablehlo.constant dense<1.0e-5> : tensor<32x197x768xf32>
    %v1995 = stablehlo.reduce(%v1991 init: %v1992) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1996 = stablehlo.broadcast_in_dim %v1995, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v1997 = stablehlo.divide %v1996, %v1993 : tensor<32x197x768xf32>
    %v1998 = stablehlo.subtract %v1991, %v1997 : tensor<32x197x768xf32>
    %v1999 = stablehlo.multiply %v1998, %v1998 : tensor<32x197x768xf32>
    %v2000 = stablehlo.reduce(%v1999 init: %v1992) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2001 = stablehlo.broadcast_in_dim %v2000, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v2002 = stablehlo.divide %v2001, %v1993 : tensor<32x197x768xf32>
    %v2003 = stablehlo.add %v2002, %v1994 : tensor<32x197x768xf32>
    %v2004 = stablehlo.rsqrt %v2003 : tensor<32x197x768xf32>
    %v2005 = stablehlo.multiply %v1998, %v2004 : tensor<32x197x768xf32>
    %v2006 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v2007 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v2008 = stablehlo.multiply %v2005, %v2006 : tensor<32x197x768xf32>
    %v2009 = stablehlo.add %v2008, %v2007 : tensor<32x197x768xf32>
    %v2010 = stablehlo.reshape %v2009 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2011 = stablehlo.reshape %v2010 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2012 = stablehlo.broadcast_in_dim %b3_g2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v2013 = stablehlo.multiply %v2011, %v2012 : tensor<32x197x768xf32>
    %v2014 = stablehlo.reshape %v2013 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2015 = stablehlo.reshape %v2014 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2016 = stablehlo.broadcast_in_dim %b3_bt2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v2017 = stablehlo.add %v2015, %v2016 : tensor<32x197x768xf32>
    %v2018 = stablehlo.reshape %v2017 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2019 = stablehlo.reshape %v2018 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2020 = stablehlo.dot_general %v2019, %b3_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x3072xf32>) -> tensor<32x197x3072xf32>
    %v2021 = stablehlo.broadcast_in_dim %b3_bfc1, dims = [2] : (tensor<3072xf32>) -> tensor<32x197x3072xf32>
    %v2022 = stablehlo.add %v2020, %v2021 : tensor<32x197x3072xf32>
    %v2023 = stablehlo.reshape %v2022 : (tensor<32x197x3072xf32>) -> tensor<32x605184xf32>
    %v2024 = stablehlo.multiply %v2023, %v2023 : tensor<32x605184xf32>
    %v2025 = stablehlo.multiply %v2024, %v2023 : tensor<32x605184xf32>
    %v2026 = stablehlo.constant dense<0.044715> : tensor<32x605184xf32>
    %v2027 = stablehlo.multiply %v2026, %v2025 : tensor<32x605184xf32>
    %v2028 = stablehlo.add %v2023, %v2027 : tensor<32x605184xf32>
    %v2029 = stablehlo.constant dense<0.7978845608028654> : tensor<32x605184xf32>
    %v2030 = stablehlo.multiply %v2029, %v2028 : tensor<32x605184xf32>
    %v2031 = stablehlo.tanh %v2030 : tensor<32x605184xf32>
    %v2032 = stablehlo.constant dense<1.0> : tensor<32x605184xf32>
    %v2033 = stablehlo.add %v2032, %v2031 : tensor<32x605184xf32>
    %v2034 = stablehlo.constant dense<0.5> : tensor<32x605184xf32>
    %v2035 = stablehlo.multiply %v2034, %v2023 : tensor<32x605184xf32>
    %v2036 = stablehlo.multiply %v2035, %v2033 : tensor<32x605184xf32>
    %v2037 = stablehlo.reshape %v2036 : (tensor<32x605184xf32>) -> tensor<32x197x3072xf32>
    %v2038 = stablehlo.dot_general %v2037, %b3_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x3072xf32>, tensor<3072x768xf32>) -> tensor<32x197x768xf32>
    %v2039 = stablehlo.broadcast_in_dim %b3_bfc2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v2040 = stablehlo.add %v2038, %v2039 : tensor<32x197x768xf32>
    %v2041 = stablehlo.reshape %v2040 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2042 = stablehlo.add %v1990, %v2041 : tensor<32x151296xf32>
    %v2043 = stablehlo.reshape %v2042 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2044 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2045 = stablehlo.constant dense<768.0> : tensor<32x197x768xf32>
    %v2046 = stablehlo.constant dense<1.0e-5> : tensor<32x197x768xf32>
    %v2047 = stablehlo.reduce(%v2043 init: %v2044) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2048 = stablehlo.broadcast_in_dim %v2047, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v2049 = stablehlo.divide %v2048, %v2045 : tensor<32x197x768xf32>
    %v2050 = stablehlo.subtract %v2043, %v2049 : tensor<32x197x768xf32>
    %v2051 = stablehlo.multiply %v2050, %v2050 : tensor<32x197x768xf32>
    %v2052 = stablehlo.reduce(%v2051 init: %v2044) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2053 = stablehlo.broadcast_in_dim %v2052, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v2054 = stablehlo.divide %v2053, %v2045 : tensor<32x197x768xf32>
    %v2055 = stablehlo.add %v2054, %v2046 : tensor<32x197x768xf32>
    %v2056 = stablehlo.rsqrt %v2055 : tensor<32x197x768xf32>
    %v2057 = stablehlo.multiply %v2050, %v2056 : tensor<32x197x768xf32>
    %v2058 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v2059 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v2060 = stablehlo.multiply %v2057, %v2058 : tensor<32x197x768xf32>
    %v2061 = stablehlo.add %v2060, %v2059 : tensor<32x197x768xf32>
    %v2062 = stablehlo.reshape %v2061 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2063 = stablehlo.reshape %v2062 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2064 = stablehlo.broadcast_in_dim %b4_g1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v2065 = stablehlo.multiply %v2063, %v2064 : tensor<32x197x768xf32>
    %v2066 = stablehlo.reshape %v2065 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2067 = stablehlo.reshape %v2066 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2068 = stablehlo.broadcast_in_dim %b4_bt1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v2069 = stablehlo.add %v2067, %v2068 : tensor<32x197x768xf32>
    %v2070 = stablehlo.reshape %v2069 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2071 = stablehlo.reshape %v2070 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2072 = stablehlo.dot_general %v2071, %b4_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v2073 = stablehlo.broadcast_in_dim %b4_bq, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v2074 = stablehlo.add %v2072, %v2073 : tensor<32x197x768xf32>
    %v2075 = stablehlo.reshape %v2074 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2076 = stablehlo.reshape %v2070 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2077 = stablehlo.dot_general %v2076, %b4_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v2078 = stablehlo.broadcast_in_dim %b4_bk, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v2079 = stablehlo.add %v2077, %v2078 : tensor<32x197x768xf32>
    %v2080 = stablehlo.reshape %v2079 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2081 = stablehlo.reshape %v2070 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2082 = stablehlo.dot_general %v2081, %b4_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v2083 = stablehlo.broadcast_in_dim %b4_bv, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v2084 = stablehlo.add %v2082, %v2083 : tensor<32x197x768xf32>
    %v2085 = stablehlo.reshape %v2084 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2086 = stablehlo.reshape %v2075 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2087 = stablehlo.slice %v2086 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2088 = stablehlo.reshape %v2087 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2089 = stablehlo.reshape %v2080 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2090 = stablehlo.slice %v2089 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2091 = stablehlo.reshape %v2090 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2092 = stablehlo.reshape %v2085 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2093 = stablehlo.slice %v2092 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2094 = stablehlo.reshape %v2093 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2095 = stablehlo.reshape %v2091 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2096 = stablehlo.transpose %v2095, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2097 = stablehlo.reshape %v2096 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2098 = stablehlo.reshape %v2088 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2099 = stablehlo.reshape %v2097 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2100 = stablehlo.dot_general %v2098, %v2099, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2101 = stablehlo.reshape %v2100 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2102 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2103 = stablehlo.multiply %v2101, %v2102 : tensor<32x38809xf32>
    %v2104 = stablehlo.reshape %v2103 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2105 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2106 = stablehlo.exponential %v2104 : tensor<32x197x197xf32>
    %v2107 = stablehlo.reduce(%v2106 init: %v2105) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2108 = stablehlo.broadcast_in_dim %v2107, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2109 = stablehlo.divide %v2106, %v2108 : tensor<32x197x197xf32>
    %v2110 = stablehlo.reshape %v2109 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2111 = stablehlo.reshape %v2110 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2112 = stablehlo.reshape %v2094 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2113 = stablehlo.dot_general %v2111, %v2112, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2114 = stablehlo.reshape %v2113 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2115 = stablehlo.reshape %v2114 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2116 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2117 = stablehlo.pad %v2115, %v2116, low = [0, 0, 0], high = [0, 0, 704], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v2118 = stablehlo.reshape %v2117 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2119 = stablehlo.reshape %v2075 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2120 = stablehlo.slice %v2119 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2121 = stablehlo.reshape %v2120 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2122 = stablehlo.reshape %v2080 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2123 = stablehlo.slice %v2122 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2124 = stablehlo.reshape %v2123 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2125 = stablehlo.reshape %v2085 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2126 = stablehlo.slice %v2125 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2127 = stablehlo.reshape %v2126 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2128 = stablehlo.reshape %v2124 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2129 = stablehlo.transpose %v2128, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2130 = stablehlo.reshape %v2129 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2131 = stablehlo.reshape %v2121 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2132 = stablehlo.reshape %v2130 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2133 = stablehlo.dot_general %v2131, %v2132, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2134 = stablehlo.reshape %v2133 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2135 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2136 = stablehlo.multiply %v2134, %v2135 : tensor<32x38809xf32>
    %v2137 = stablehlo.reshape %v2136 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2138 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2139 = stablehlo.exponential %v2137 : tensor<32x197x197xf32>
    %v2140 = stablehlo.reduce(%v2139 init: %v2138) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2141 = stablehlo.broadcast_in_dim %v2140, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2142 = stablehlo.divide %v2139, %v2141 : tensor<32x197x197xf32>
    %v2143 = stablehlo.reshape %v2142 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2144 = stablehlo.reshape %v2143 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2145 = stablehlo.reshape %v2127 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2146 = stablehlo.dot_general %v2144, %v2145, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2147 = stablehlo.reshape %v2146 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2148 = stablehlo.reshape %v2147 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2149 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2150 = stablehlo.pad %v2148, %v2149, low = [0, 0, 64], high = [0, 0, 640], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v2151 = stablehlo.reshape %v2150 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2152 = stablehlo.add %v2118, %v2151 : tensor<32x151296xf32>
    %v2153 = stablehlo.reshape %v2075 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2154 = stablehlo.slice %v2153 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2155 = stablehlo.reshape %v2154 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2156 = stablehlo.reshape %v2080 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2157 = stablehlo.slice %v2156 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2158 = stablehlo.reshape %v2157 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2159 = stablehlo.reshape %v2085 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2160 = stablehlo.slice %v2159 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2161 = stablehlo.reshape %v2160 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2162 = stablehlo.reshape %v2158 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2163 = stablehlo.transpose %v2162, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2164 = stablehlo.reshape %v2163 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2165 = stablehlo.reshape %v2155 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2166 = stablehlo.reshape %v2164 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2167 = stablehlo.dot_general %v2165, %v2166, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2168 = stablehlo.reshape %v2167 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2169 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2170 = stablehlo.multiply %v2168, %v2169 : tensor<32x38809xf32>
    %v2171 = stablehlo.reshape %v2170 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2172 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2173 = stablehlo.exponential %v2171 : tensor<32x197x197xf32>
    %v2174 = stablehlo.reduce(%v2173 init: %v2172) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2175 = stablehlo.broadcast_in_dim %v2174, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2176 = stablehlo.divide %v2173, %v2175 : tensor<32x197x197xf32>
    %v2177 = stablehlo.reshape %v2176 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2178 = stablehlo.reshape %v2177 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2179 = stablehlo.reshape %v2161 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2180 = stablehlo.dot_general %v2178, %v2179, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2181 = stablehlo.reshape %v2180 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2182 = stablehlo.reshape %v2181 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2183 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2184 = stablehlo.pad %v2182, %v2183, low = [0, 0, 128], high = [0, 0, 576], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v2185 = stablehlo.reshape %v2184 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2186 = stablehlo.add %v2152, %v2185 : tensor<32x151296xf32>
    %v2187 = stablehlo.reshape %v2075 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2188 = stablehlo.slice %v2187 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2189 = stablehlo.reshape %v2188 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2190 = stablehlo.reshape %v2080 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2191 = stablehlo.slice %v2190 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2192 = stablehlo.reshape %v2191 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2193 = stablehlo.reshape %v2085 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2194 = stablehlo.slice %v2193 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2195 = stablehlo.reshape %v2194 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2196 = stablehlo.reshape %v2192 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2197 = stablehlo.transpose %v2196, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2198 = stablehlo.reshape %v2197 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2199 = stablehlo.reshape %v2189 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2200 = stablehlo.reshape %v2198 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2201 = stablehlo.dot_general %v2199, %v2200, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2202 = stablehlo.reshape %v2201 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2203 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2204 = stablehlo.multiply %v2202, %v2203 : tensor<32x38809xf32>
    %v2205 = stablehlo.reshape %v2204 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2206 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2207 = stablehlo.exponential %v2205 : tensor<32x197x197xf32>
    %v2208 = stablehlo.reduce(%v2207 init: %v2206) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2209 = stablehlo.broadcast_in_dim %v2208, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2210 = stablehlo.divide %v2207, %v2209 : tensor<32x197x197xf32>
    %v2211 = stablehlo.reshape %v2210 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2212 = stablehlo.reshape %v2211 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2213 = stablehlo.reshape %v2195 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2214 = stablehlo.dot_general %v2212, %v2213, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2215 = stablehlo.reshape %v2214 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2216 = stablehlo.reshape %v2215 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2217 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2218 = stablehlo.pad %v2216, %v2217, low = [0, 0, 192], high = [0, 0, 512], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v2219 = stablehlo.reshape %v2218 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2220 = stablehlo.add %v2186, %v2219 : tensor<32x151296xf32>
    %v2221 = stablehlo.reshape %v2075 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2222 = stablehlo.slice %v2221 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2223 = stablehlo.reshape %v2222 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2224 = stablehlo.reshape %v2080 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2225 = stablehlo.slice %v2224 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2226 = stablehlo.reshape %v2225 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2227 = stablehlo.reshape %v2085 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2228 = stablehlo.slice %v2227 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2229 = stablehlo.reshape %v2228 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2230 = stablehlo.reshape %v2226 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2231 = stablehlo.transpose %v2230, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2232 = stablehlo.reshape %v2231 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2233 = stablehlo.reshape %v2223 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2234 = stablehlo.reshape %v2232 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2235 = stablehlo.dot_general %v2233, %v2234, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2236 = stablehlo.reshape %v2235 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2237 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2238 = stablehlo.multiply %v2236, %v2237 : tensor<32x38809xf32>
    %v2239 = stablehlo.reshape %v2238 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2240 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2241 = stablehlo.exponential %v2239 : tensor<32x197x197xf32>
    %v2242 = stablehlo.reduce(%v2241 init: %v2240) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2243 = stablehlo.broadcast_in_dim %v2242, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2244 = stablehlo.divide %v2241, %v2243 : tensor<32x197x197xf32>
    %v2245 = stablehlo.reshape %v2244 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2246 = stablehlo.reshape %v2245 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2247 = stablehlo.reshape %v2229 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2248 = stablehlo.dot_general %v2246, %v2247, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2249 = stablehlo.reshape %v2248 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2250 = stablehlo.reshape %v2249 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2251 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2252 = stablehlo.pad %v2250, %v2251, low = [0, 0, 256], high = [0, 0, 448], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v2253 = stablehlo.reshape %v2252 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2254 = stablehlo.add %v2220, %v2253 : tensor<32x151296xf32>
    %v2255 = stablehlo.reshape %v2075 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2256 = stablehlo.slice %v2255 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2257 = stablehlo.reshape %v2256 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2258 = stablehlo.reshape %v2080 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2259 = stablehlo.slice %v2258 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2260 = stablehlo.reshape %v2259 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2261 = stablehlo.reshape %v2085 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2262 = stablehlo.slice %v2261 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2263 = stablehlo.reshape %v2262 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2264 = stablehlo.reshape %v2260 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2265 = stablehlo.transpose %v2264, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2266 = stablehlo.reshape %v2265 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2267 = stablehlo.reshape %v2257 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2268 = stablehlo.reshape %v2266 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2269 = stablehlo.dot_general %v2267, %v2268, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2270 = stablehlo.reshape %v2269 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2271 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2272 = stablehlo.multiply %v2270, %v2271 : tensor<32x38809xf32>
    %v2273 = stablehlo.reshape %v2272 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2274 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2275 = stablehlo.exponential %v2273 : tensor<32x197x197xf32>
    %v2276 = stablehlo.reduce(%v2275 init: %v2274) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2277 = stablehlo.broadcast_in_dim %v2276, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2278 = stablehlo.divide %v2275, %v2277 : tensor<32x197x197xf32>
    %v2279 = stablehlo.reshape %v2278 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2280 = stablehlo.reshape %v2279 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2281 = stablehlo.reshape %v2263 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2282 = stablehlo.dot_general %v2280, %v2281, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2283 = stablehlo.reshape %v2282 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2284 = stablehlo.reshape %v2283 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2285 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2286 = stablehlo.pad %v2284, %v2285, low = [0, 0, 320], high = [0, 0, 384], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v2287 = stablehlo.reshape %v2286 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2288 = stablehlo.add %v2254, %v2287 : tensor<32x151296xf32>
    %v2289 = stablehlo.reshape %v2075 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2290 = stablehlo.slice %v2289 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2291 = stablehlo.reshape %v2290 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2292 = stablehlo.reshape %v2080 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2293 = stablehlo.slice %v2292 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2294 = stablehlo.reshape %v2293 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2295 = stablehlo.reshape %v2085 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2296 = stablehlo.slice %v2295 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2297 = stablehlo.reshape %v2296 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2298 = stablehlo.reshape %v2294 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2299 = stablehlo.transpose %v2298, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2300 = stablehlo.reshape %v2299 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2301 = stablehlo.reshape %v2291 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2302 = stablehlo.reshape %v2300 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2303 = stablehlo.dot_general %v2301, %v2302, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2304 = stablehlo.reshape %v2303 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2305 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2306 = stablehlo.multiply %v2304, %v2305 : tensor<32x38809xf32>
    %v2307 = stablehlo.reshape %v2306 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2308 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2309 = stablehlo.exponential %v2307 : tensor<32x197x197xf32>
    %v2310 = stablehlo.reduce(%v2309 init: %v2308) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2311 = stablehlo.broadcast_in_dim %v2310, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2312 = stablehlo.divide %v2309, %v2311 : tensor<32x197x197xf32>
    %v2313 = stablehlo.reshape %v2312 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2314 = stablehlo.reshape %v2313 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2315 = stablehlo.reshape %v2297 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2316 = stablehlo.dot_general %v2314, %v2315, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2317 = stablehlo.reshape %v2316 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2318 = stablehlo.reshape %v2317 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2319 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2320 = stablehlo.pad %v2318, %v2319, low = [0, 0, 384], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v2321 = stablehlo.reshape %v2320 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2322 = stablehlo.add %v2288, %v2321 : tensor<32x151296xf32>
    %v2323 = stablehlo.reshape %v2075 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2324 = stablehlo.slice %v2323 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2325 = stablehlo.reshape %v2324 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2326 = stablehlo.reshape %v2080 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2327 = stablehlo.slice %v2326 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2328 = stablehlo.reshape %v2327 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2329 = stablehlo.reshape %v2085 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2330 = stablehlo.slice %v2329 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2331 = stablehlo.reshape %v2330 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2332 = stablehlo.reshape %v2328 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2333 = stablehlo.transpose %v2332, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2334 = stablehlo.reshape %v2333 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2335 = stablehlo.reshape %v2325 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2336 = stablehlo.reshape %v2334 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2337 = stablehlo.dot_general %v2335, %v2336, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2338 = stablehlo.reshape %v2337 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2339 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2340 = stablehlo.multiply %v2338, %v2339 : tensor<32x38809xf32>
    %v2341 = stablehlo.reshape %v2340 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2342 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2343 = stablehlo.exponential %v2341 : tensor<32x197x197xf32>
    %v2344 = stablehlo.reduce(%v2343 init: %v2342) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2345 = stablehlo.broadcast_in_dim %v2344, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2346 = stablehlo.divide %v2343, %v2345 : tensor<32x197x197xf32>
    %v2347 = stablehlo.reshape %v2346 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2348 = stablehlo.reshape %v2347 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2349 = stablehlo.reshape %v2331 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2350 = stablehlo.dot_general %v2348, %v2349, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2351 = stablehlo.reshape %v2350 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2352 = stablehlo.reshape %v2351 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2353 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2354 = stablehlo.pad %v2352, %v2353, low = [0, 0, 448], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v2355 = stablehlo.reshape %v2354 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2356 = stablehlo.add %v2322, %v2355 : tensor<32x151296xf32>
    %v2357 = stablehlo.reshape %v2075 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2358 = stablehlo.slice %v2357 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2359 = stablehlo.reshape %v2358 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2360 = stablehlo.reshape %v2080 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2361 = stablehlo.slice %v2360 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2362 = stablehlo.reshape %v2361 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2363 = stablehlo.reshape %v2085 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2364 = stablehlo.slice %v2363 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2365 = stablehlo.reshape %v2364 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2366 = stablehlo.reshape %v2362 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2367 = stablehlo.transpose %v2366, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2368 = stablehlo.reshape %v2367 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2369 = stablehlo.reshape %v2359 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2370 = stablehlo.reshape %v2368 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2371 = stablehlo.dot_general %v2369, %v2370, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2372 = stablehlo.reshape %v2371 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2373 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2374 = stablehlo.multiply %v2372, %v2373 : tensor<32x38809xf32>
    %v2375 = stablehlo.reshape %v2374 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2376 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2377 = stablehlo.exponential %v2375 : tensor<32x197x197xf32>
    %v2378 = stablehlo.reduce(%v2377 init: %v2376) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2379 = stablehlo.broadcast_in_dim %v2378, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2380 = stablehlo.divide %v2377, %v2379 : tensor<32x197x197xf32>
    %v2381 = stablehlo.reshape %v2380 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2382 = stablehlo.reshape %v2381 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2383 = stablehlo.reshape %v2365 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2384 = stablehlo.dot_general %v2382, %v2383, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2385 = stablehlo.reshape %v2384 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2386 = stablehlo.reshape %v2385 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2387 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2388 = stablehlo.pad %v2386, %v2387, low = [0, 0, 512], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v2389 = stablehlo.reshape %v2388 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2390 = stablehlo.add %v2356, %v2389 : tensor<32x151296xf32>
    %v2391 = stablehlo.reshape %v2075 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2392 = stablehlo.slice %v2391 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2393 = stablehlo.reshape %v2392 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2394 = stablehlo.reshape %v2080 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2395 = stablehlo.slice %v2394 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2396 = stablehlo.reshape %v2395 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2397 = stablehlo.reshape %v2085 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2398 = stablehlo.slice %v2397 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2399 = stablehlo.reshape %v2398 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2400 = stablehlo.reshape %v2396 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2401 = stablehlo.transpose %v2400, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2402 = stablehlo.reshape %v2401 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2403 = stablehlo.reshape %v2393 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2404 = stablehlo.reshape %v2402 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2405 = stablehlo.dot_general %v2403, %v2404, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2406 = stablehlo.reshape %v2405 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2407 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2408 = stablehlo.multiply %v2406, %v2407 : tensor<32x38809xf32>
    %v2409 = stablehlo.reshape %v2408 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2410 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2411 = stablehlo.exponential %v2409 : tensor<32x197x197xf32>
    %v2412 = stablehlo.reduce(%v2411 init: %v2410) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2413 = stablehlo.broadcast_in_dim %v2412, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2414 = stablehlo.divide %v2411, %v2413 : tensor<32x197x197xf32>
    %v2415 = stablehlo.reshape %v2414 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2416 = stablehlo.reshape %v2415 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2417 = stablehlo.reshape %v2399 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2418 = stablehlo.dot_general %v2416, %v2417, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2419 = stablehlo.reshape %v2418 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2420 = stablehlo.reshape %v2419 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2421 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2422 = stablehlo.pad %v2420, %v2421, low = [0, 0, 576], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v2423 = stablehlo.reshape %v2422 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2424 = stablehlo.add %v2390, %v2423 : tensor<32x151296xf32>
    %v2425 = stablehlo.reshape %v2075 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2426 = stablehlo.slice %v2425 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2427 = stablehlo.reshape %v2426 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2428 = stablehlo.reshape %v2080 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2429 = stablehlo.slice %v2428 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2430 = stablehlo.reshape %v2429 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2431 = stablehlo.reshape %v2085 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2432 = stablehlo.slice %v2431 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2433 = stablehlo.reshape %v2432 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2434 = stablehlo.reshape %v2430 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2435 = stablehlo.transpose %v2434, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2436 = stablehlo.reshape %v2435 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2437 = stablehlo.reshape %v2427 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2438 = stablehlo.reshape %v2436 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2439 = stablehlo.dot_general %v2437, %v2438, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2440 = stablehlo.reshape %v2439 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2441 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2442 = stablehlo.multiply %v2440, %v2441 : tensor<32x38809xf32>
    %v2443 = stablehlo.reshape %v2442 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2444 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2445 = stablehlo.exponential %v2443 : tensor<32x197x197xf32>
    %v2446 = stablehlo.reduce(%v2445 init: %v2444) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2447 = stablehlo.broadcast_in_dim %v2446, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2448 = stablehlo.divide %v2445, %v2447 : tensor<32x197x197xf32>
    %v2449 = stablehlo.reshape %v2448 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2450 = stablehlo.reshape %v2449 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2451 = stablehlo.reshape %v2433 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2452 = stablehlo.dot_general %v2450, %v2451, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2453 = stablehlo.reshape %v2452 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2454 = stablehlo.reshape %v2453 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2455 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2456 = stablehlo.pad %v2454, %v2455, low = [0, 0, 640], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v2457 = stablehlo.reshape %v2456 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2458 = stablehlo.add %v2424, %v2457 : tensor<32x151296xf32>
    %v2459 = stablehlo.reshape %v2075 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2460 = stablehlo.slice %v2459 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2461 = stablehlo.reshape %v2460 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2462 = stablehlo.reshape %v2080 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2463 = stablehlo.slice %v2462 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2464 = stablehlo.reshape %v2463 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2465 = stablehlo.reshape %v2085 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2466 = stablehlo.slice %v2465 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2467 = stablehlo.reshape %v2466 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2468 = stablehlo.reshape %v2464 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2469 = stablehlo.transpose %v2468, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2470 = stablehlo.reshape %v2469 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2471 = stablehlo.reshape %v2461 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2472 = stablehlo.reshape %v2470 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2473 = stablehlo.dot_general %v2471, %v2472, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2474 = stablehlo.reshape %v2473 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2475 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2476 = stablehlo.multiply %v2474, %v2475 : tensor<32x38809xf32>
    %v2477 = stablehlo.reshape %v2476 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2478 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2479 = stablehlo.exponential %v2477 : tensor<32x197x197xf32>
    %v2480 = stablehlo.reduce(%v2479 init: %v2478) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2481 = stablehlo.broadcast_in_dim %v2480, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2482 = stablehlo.divide %v2479, %v2481 : tensor<32x197x197xf32>
    %v2483 = stablehlo.reshape %v2482 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2484 = stablehlo.reshape %v2483 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2485 = stablehlo.reshape %v2467 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2486 = stablehlo.dot_general %v2484, %v2485, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2487 = stablehlo.reshape %v2486 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2488 = stablehlo.reshape %v2487 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2489 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2490 = stablehlo.pad %v2488, %v2489, low = [0, 0, 704], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v2491 = stablehlo.reshape %v2490 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2492 = stablehlo.add %v2458, %v2491 : tensor<32x151296xf32>
    %v2493 = stablehlo.reshape %v2492 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2494 = stablehlo.dot_general %v2493, %b4_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v2495 = stablehlo.broadcast_in_dim %b4_bo, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v2496 = stablehlo.add %v2494, %v2495 : tensor<32x197x768xf32>
    %v2497 = stablehlo.reshape %v2496 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2498 = stablehlo.add %v2042, %v2497 : tensor<32x151296xf32>
    %v2499 = stablehlo.reshape %v2498 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2500 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2501 = stablehlo.constant dense<768.0> : tensor<32x197x768xf32>
    %v2502 = stablehlo.constant dense<1.0e-5> : tensor<32x197x768xf32>
    %v2503 = stablehlo.reduce(%v2499 init: %v2500) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2504 = stablehlo.broadcast_in_dim %v2503, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v2505 = stablehlo.divide %v2504, %v2501 : tensor<32x197x768xf32>
    %v2506 = stablehlo.subtract %v2499, %v2505 : tensor<32x197x768xf32>
    %v2507 = stablehlo.multiply %v2506, %v2506 : tensor<32x197x768xf32>
    %v2508 = stablehlo.reduce(%v2507 init: %v2500) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2509 = stablehlo.broadcast_in_dim %v2508, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v2510 = stablehlo.divide %v2509, %v2501 : tensor<32x197x768xf32>
    %v2511 = stablehlo.add %v2510, %v2502 : tensor<32x197x768xf32>
    %v2512 = stablehlo.rsqrt %v2511 : tensor<32x197x768xf32>
    %v2513 = stablehlo.multiply %v2506, %v2512 : tensor<32x197x768xf32>
    %v2514 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v2515 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v2516 = stablehlo.multiply %v2513, %v2514 : tensor<32x197x768xf32>
    %v2517 = stablehlo.add %v2516, %v2515 : tensor<32x197x768xf32>
    %v2518 = stablehlo.reshape %v2517 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2519 = stablehlo.reshape %v2518 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2520 = stablehlo.broadcast_in_dim %b4_g2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v2521 = stablehlo.multiply %v2519, %v2520 : tensor<32x197x768xf32>
    %v2522 = stablehlo.reshape %v2521 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2523 = stablehlo.reshape %v2522 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2524 = stablehlo.broadcast_in_dim %b4_bt2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v2525 = stablehlo.add %v2523, %v2524 : tensor<32x197x768xf32>
    %v2526 = stablehlo.reshape %v2525 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2527 = stablehlo.reshape %v2526 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2528 = stablehlo.dot_general %v2527, %b4_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x3072xf32>) -> tensor<32x197x3072xf32>
    %v2529 = stablehlo.broadcast_in_dim %b4_bfc1, dims = [2] : (tensor<3072xf32>) -> tensor<32x197x3072xf32>
    %v2530 = stablehlo.add %v2528, %v2529 : tensor<32x197x3072xf32>
    %v2531 = stablehlo.reshape %v2530 : (tensor<32x197x3072xf32>) -> tensor<32x605184xf32>
    %v2532 = stablehlo.multiply %v2531, %v2531 : tensor<32x605184xf32>
    %v2533 = stablehlo.multiply %v2532, %v2531 : tensor<32x605184xf32>
    %v2534 = stablehlo.constant dense<0.044715> : tensor<32x605184xf32>
    %v2535 = stablehlo.multiply %v2534, %v2533 : tensor<32x605184xf32>
    %v2536 = stablehlo.add %v2531, %v2535 : tensor<32x605184xf32>
    %v2537 = stablehlo.constant dense<0.7978845608028654> : tensor<32x605184xf32>
    %v2538 = stablehlo.multiply %v2537, %v2536 : tensor<32x605184xf32>
    %v2539 = stablehlo.tanh %v2538 : tensor<32x605184xf32>
    %v2540 = stablehlo.constant dense<1.0> : tensor<32x605184xf32>
    %v2541 = stablehlo.add %v2540, %v2539 : tensor<32x605184xf32>
    %v2542 = stablehlo.constant dense<0.5> : tensor<32x605184xf32>
    %v2543 = stablehlo.multiply %v2542, %v2531 : tensor<32x605184xf32>
    %v2544 = stablehlo.multiply %v2543, %v2541 : tensor<32x605184xf32>
    %v2545 = stablehlo.reshape %v2544 : (tensor<32x605184xf32>) -> tensor<32x197x3072xf32>
    %v2546 = stablehlo.dot_general %v2545, %b4_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x3072xf32>, tensor<3072x768xf32>) -> tensor<32x197x768xf32>
    %v2547 = stablehlo.broadcast_in_dim %b4_bfc2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v2548 = stablehlo.add %v2546, %v2547 : tensor<32x197x768xf32>
    %v2549 = stablehlo.reshape %v2548 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2550 = stablehlo.add %v2498, %v2549 : tensor<32x151296xf32>
    %v2551 = stablehlo.reshape %v2550 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2552 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2553 = stablehlo.constant dense<768.0> : tensor<32x197x768xf32>
    %v2554 = stablehlo.constant dense<1.0e-5> : tensor<32x197x768xf32>
    %v2555 = stablehlo.reduce(%v2551 init: %v2552) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2556 = stablehlo.broadcast_in_dim %v2555, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v2557 = stablehlo.divide %v2556, %v2553 : tensor<32x197x768xf32>
    %v2558 = stablehlo.subtract %v2551, %v2557 : tensor<32x197x768xf32>
    %v2559 = stablehlo.multiply %v2558, %v2558 : tensor<32x197x768xf32>
    %v2560 = stablehlo.reduce(%v2559 init: %v2552) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2561 = stablehlo.broadcast_in_dim %v2560, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v2562 = stablehlo.divide %v2561, %v2553 : tensor<32x197x768xf32>
    %v2563 = stablehlo.add %v2562, %v2554 : tensor<32x197x768xf32>
    %v2564 = stablehlo.rsqrt %v2563 : tensor<32x197x768xf32>
    %v2565 = stablehlo.multiply %v2558, %v2564 : tensor<32x197x768xf32>
    %v2566 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v2567 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v2568 = stablehlo.multiply %v2565, %v2566 : tensor<32x197x768xf32>
    %v2569 = stablehlo.add %v2568, %v2567 : tensor<32x197x768xf32>
    %v2570 = stablehlo.reshape %v2569 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2571 = stablehlo.reshape %v2570 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2572 = stablehlo.broadcast_in_dim %b5_g1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v2573 = stablehlo.multiply %v2571, %v2572 : tensor<32x197x768xf32>
    %v2574 = stablehlo.reshape %v2573 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2575 = stablehlo.reshape %v2574 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2576 = stablehlo.broadcast_in_dim %b5_bt1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v2577 = stablehlo.add %v2575, %v2576 : tensor<32x197x768xf32>
    %v2578 = stablehlo.reshape %v2577 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2579 = stablehlo.reshape %v2578 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2580 = stablehlo.dot_general %v2579, %b5_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v2581 = stablehlo.broadcast_in_dim %b5_bq, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v2582 = stablehlo.add %v2580, %v2581 : tensor<32x197x768xf32>
    %v2583 = stablehlo.reshape %v2582 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2584 = stablehlo.reshape %v2578 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2585 = stablehlo.dot_general %v2584, %b5_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v2586 = stablehlo.broadcast_in_dim %b5_bk, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v2587 = stablehlo.add %v2585, %v2586 : tensor<32x197x768xf32>
    %v2588 = stablehlo.reshape %v2587 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2589 = stablehlo.reshape %v2578 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2590 = stablehlo.dot_general %v2589, %b5_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v2591 = stablehlo.broadcast_in_dim %b5_bv, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v2592 = stablehlo.add %v2590, %v2591 : tensor<32x197x768xf32>
    %v2593 = stablehlo.reshape %v2592 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2594 = stablehlo.reshape %v2583 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2595 = stablehlo.slice %v2594 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2596 = stablehlo.reshape %v2595 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2597 = stablehlo.reshape %v2588 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2598 = stablehlo.slice %v2597 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2599 = stablehlo.reshape %v2598 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2600 = stablehlo.reshape %v2593 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2601 = stablehlo.slice %v2600 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2602 = stablehlo.reshape %v2601 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2603 = stablehlo.reshape %v2599 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2604 = stablehlo.transpose %v2603, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2605 = stablehlo.reshape %v2604 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2606 = stablehlo.reshape %v2596 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2607 = stablehlo.reshape %v2605 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2608 = stablehlo.dot_general %v2606, %v2607, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2609 = stablehlo.reshape %v2608 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2610 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2611 = stablehlo.multiply %v2609, %v2610 : tensor<32x38809xf32>
    %v2612 = stablehlo.reshape %v2611 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2613 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2614 = stablehlo.exponential %v2612 : tensor<32x197x197xf32>
    %v2615 = stablehlo.reduce(%v2614 init: %v2613) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2616 = stablehlo.broadcast_in_dim %v2615, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2617 = stablehlo.divide %v2614, %v2616 : tensor<32x197x197xf32>
    %v2618 = stablehlo.reshape %v2617 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2619 = stablehlo.reshape %v2618 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2620 = stablehlo.reshape %v2602 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2621 = stablehlo.dot_general %v2619, %v2620, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2622 = stablehlo.reshape %v2621 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2623 = stablehlo.reshape %v2622 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2624 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2625 = stablehlo.pad %v2623, %v2624, low = [0, 0, 0], high = [0, 0, 704], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v2626 = stablehlo.reshape %v2625 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2627 = stablehlo.reshape %v2583 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2628 = stablehlo.slice %v2627 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2629 = stablehlo.reshape %v2628 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2630 = stablehlo.reshape %v2588 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2631 = stablehlo.slice %v2630 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2632 = stablehlo.reshape %v2631 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2633 = stablehlo.reshape %v2593 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2634 = stablehlo.slice %v2633 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2635 = stablehlo.reshape %v2634 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2636 = stablehlo.reshape %v2632 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2637 = stablehlo.transpose %v2636, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2638 = stablehlo.reshape %v2637 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2639 = stablehlo.reshape %v2629 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2640 = stablehlo.reshape %v2638 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2641 = stablehlo.dot_general %v2639, %v2640, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2642 = stablehlo.reshape %v2641 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2643 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2644 = stablehlo.multiply %v2642, %v2643 : tensor<32x38809xf32>
    %v2645 = stablehlo.reshape %v2644 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2646 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2647 = stablehlo.exponential %v2645 : tensor<32x197x197xf32>
    %v2648 = stablehlo.reduce(%v2647 init: %v2646) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2649 = stablehlo.broadcast_in_dim %v2648, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2650 = stablehlo.divide %v2647, %v2649 : tensor<32x197x197xf32>
    %v2651 = stablehlo.reshape %v2650 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2652 = stablehlo.reshape %v2651 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2653 = stablehlo.reshape %v2635 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2654 = stablehlo.dot_general %v2652, %v2653, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2655 = stablehlo.reshape %v2654 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2656 = stablehlo.reshape %v2655 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2657 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2658 = stablehlo.pad %v2656, %v2657, low = [0, 0, 64], high = [0, 0, 640], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v2659 = stablehlo.reshape %v2658 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2660 = stablehlo.add %v2626, %v2659 : tensor<32x151296xf32>
    %v2661 = stablehlo.reshape %v2583 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2662 = stablehlo.slice %v2661 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2663 = stablehlo.reshape %v2662 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2664 = stablehlo.reshape %v2588 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2665 = stablehlo.slice %v2664 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2666 = stablehlo.reshape %v2665 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2667 = stablehlo.reshape %v2593 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2668 = stablehlo.slice %v2667 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2669 = stablehlo.reshape %v2668 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2670 = stablehlo.reshape %v2666 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2671 = stablehlo.transpose %v2670, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2672 = stablehlo.reshape %v2671 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2673 = stablehlo.reshape %v2663 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2674 = stablehlo.reshape %v2672 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2675 = stablehlo.dot_general %v2673, %v2674, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2676 = stablehlo.reshape %v2675 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2677 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2678 = stablehlo.multiply %v2676, %v2677 : tensor<32x38809xf32>
    %v2679 = stablehlo.reshape %v2678 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2680 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2681 = stablehlo.exponential %v2679 : tensor<32x197x197xf32>
    %v2682 = stablehlo.reduce(%v2681 init: %v2680) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2683 = stablehlo.broadcast_in_dim %v2682, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2684 = stablehlo.divide %v2681, %v2683 : tensor<32x197x197xf32>
    %v2685 = stablehlo.reshape %v2684 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2686 = stablehlo.reshape %v2685 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2687 = stablehlo.reshape %v2669 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2688 = stablehlo.dot_general %v2686, %v2687, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2689 = stablehlo.reshape %v2688 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2690 = stablehlo.reshape %v2689 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2691 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2692 = stablehlo.pad %v2690, %v2691, low = [0, 0, 128], high = [0, 0, 576], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v2693 = stablehlo.reshape %v2692 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2694 = stablehlo.add %v2660, %v2693 : tensor<32x151296xf32>
    %v2695 = stablehlo.reshape %v2583 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2696 = stablehlo.slice %v2695 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2697 = stablehlo.reshape %v2696 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2698 = stablehlo.reshape %v2588 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2699 = stablehlo.slice %v2698 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2700 = stablehlo.reshape %v2699 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2701 = stablehlo.reshape %v2593 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2702 = stablehlo.slice %v2701 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2703 = stablehlo.reshape %v2702 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2704 = stablehlo.reshape %v2700 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2705 = stablehlo.transpose %v2704, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2706 = stablehlo.reshape %v2705 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2707 = stablehlo.reshape %v2697 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2708 = stablehlo.reshape %v2706 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2709 = stablehlo.dot_general %v2707, %v2708, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2710 = stablehlo.reshape %v2709 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2711 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2712 = stablehlo.multiply %v2710, %v2711 : tensor<32x38809xf32>
    %v2713 = stablehlo.reshape %v2712 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2714 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2715 = stablehlo.exponential %v2713 : tensor<32x197x197xf32>
    %v2716 = stablehlo.reduce(%v2715 init: %v2714) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2717 = stablehlo.broadcast_in_dim %v2716, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2718 = stablehlo.divide %v2715, %v2717 : tensor<32x197x197xf32>
    %v2719 = stablehlo.reshape %v2718 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2720 = stablehlo.reshape %v2719 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2721 = stablehlo.reshape %v2703 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2722 = stablehlo.dot_general %v2720, %v2721, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2723 = stablehlo.reshape %v2722 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2724 = stablehlo.reshape %v2723 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2725 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2726 = stablehlo.pad %v2724, %v2725, low = [0, 0, 192], high = [0, 0, 512], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v2727 = stablehlo.reshape %v2726 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2728 = stablehlo.add %v2694, %v2727 : tensor<32x151296xf32>
    %v2729 = stablehlo.reshape %v2583 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2730 = stablehlo.slice %v2729 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2731 = stablehlo.reshape %v2730 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2732 = stablehlo.reshape %v2588 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2733 = stablehlo.slice %v2732 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2734 = stablehlo.reshape %v2733 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2735 = stablehlo.reshape %v2593 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2736 = stablehlo.slice %v2735 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2737 = stablehlo.reshape %v2736 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2738 = stablehlo.reshape %v2734 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2739 = stablehlo.transpose %v2738, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2740 = stablehlo.reshape %v2739 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2741 = stablehlo.reshape %v2731 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2742 = stablehlo.reshape %v2740 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2743 = stablehlo.dot_general %v2741, %v2742, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2744 = stablehlo.reshape %v2743 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2745 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2746 = stablehlo.multiply %v2744, %v2745 : tensor<32x38809xf32>
    %v2747 = stablehlo.reshape %v2746 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2748 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2749 = stablehlo.exponential %v2747 : tensor<32x197x197xf32>
    %v2750 = stablehlo.reduce(%v2749 init: %v2748) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2751 = stablehlo.broadcast_in_dim %v2750, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2752 = stablehlo.divide %v2749, %v2751 : tensor<32x197x197xf32>
    %v2753 = stablehlo.reshape %v2752 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2754 = stablehlo.reshape %v2753 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2755 = stablehlo.reshape %v2737 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2756 = stablehlo.dot_general %v2754, %v2755, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2757 = stablehlo.reshape %v2756 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2758 = stablehlo.reshape %v2757 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2759 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2760 = stablehlo.pad %v2758, %v2759, low = [0, 0, 256], high = [0, 0, 448], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v2761 = stablehlo.reshape %v2760 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2762 = stablehlo.add %v2728, %v2761 : tensor<32x151296xf32>
    %v2763 = stablehlo.reshape %v2583 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2764 = stablehlo.slice %v2763 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2765 = stablehlo.reshape %v2764 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2766 = stablehlo.reshape %v2588 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2767 = stablehlo.slice %v2766 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2768 = stablehlo.reshape %v2767 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2769 = stablehlo.reshape %v2593 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2770 = stablehlo.slice %v2769 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2771 = stablehlo.reshape %v2770 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2772 = stablehlo.reshape %v2768 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2773 = stablehlo.transpose %v2772, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2774 = stablehlo.reshape %v2773 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2775 = stablehlo.reshape %v2765 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2776 = stablehlo.reshape %v2774 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2777 = stablehlo.dot_general %v2775, %v2776, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2778 = stablehlo.reshape %v2777 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2779 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2780 = stablehlo.multiply %v2778, %v2779 : tensor<32x38809xf32>
    %v2781 = stablehlo.reshape %v2780 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2782 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2783 = stablehlo.exponential %v2781 : tensor<32x197x197xf32>
    %v2784 = stablehlo.reduce(%v2783 init: %v2782) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2785 = stablehlo.broadcast_in_dim %v2784, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2786 = stablehlo.divide %v2783, %v2785 : tensor<32x197x197xf32>
    %v2787 = stablehlo.reshape %v2786 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2788 = stablehlo.reshape %v2787 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2789 = stablehlo.reshape %v2771 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2790 = stablehlo.dot_general %v2788, %v2789, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2791 = stablehlo.reshape %v2790 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2792 = stablehlo.reshape %v2791 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2793 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2794 = stablehlo.pad %v2792, %v2793, low = [0, 0, 320], high = [0, 0, 384], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v2795 = stablehlo.reshape %v2794 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2796 = stablehlo.add %v2762, %v2795 : tensor<32x151296xf32>
    %v2797 = stablehlo.reshape %v2583 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2798 = stablehlo.slice %v2797 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2799 = stablehlo.reshape %v2798 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2800 = stablehlo.reshape %v2588 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2801 = stablehlo.slice %v2800 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2802 = stablehlo.reshape %v2801 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2803 = stablehlo.reshape %v2593 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2804 = stablehlo.slice %v2803 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2805 = stablehlo.reshape %v2804 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2806 = stablehlo.reshape %v2802 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2807 = stablehlo.transpose %v2806, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2808 = stablehlo.reshape %v2807 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2809 = stablehlo.reshape %v2799 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2810 = stablehlo.reshape %v2808 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2811 = stablehlo.dot_general %v2809, %v2810, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2812 = stablehlo.reshape %v2811 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2813 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2814 = stablehlo.multiply %v2812, %v2813 : tensor<32x38809xf32>
    %v2815 = stablehlo.reshape %v2814 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2816 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2817 = stablehlo.exponential %v2815 : tensor<32x197x197xf32>
    %v2818 = stablehlo.reduce(%v2817 init: %v2816) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2819 = stablehlo.broadcast_in_dim %v2818, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2820 = stablehlo.divide %v2817, %v2819 : tensor<32x197x197xf32>
    %v2821 = stablehlo.reshape %v2820 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2822 = stablehlo.reshape %v2821 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2823 = stablehlo.reshape %v2805 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2824 = stablehlo.dot_general %v2822, %v2823, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2825 = stablehlo.reshape %v2824 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2826 = stablehlo.reshape %v2825 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2827 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2828 = stablehlo.pad %v2826, %v2827, low = [0, 0, 384], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v2829 = stablehlo.reshape %v2828 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2830 = stablehlo.add %v2796, %v2829 : tensor<32x151296xf32>
    %v2831 = stablehlo.reshape %v2583 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2832 = stablehlo.slice %v2831 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2833 = stablehlo.reshape %v2832 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2834 = stablehlo.reshape %v2588 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2835 = stablehlo.slice %v2834 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2836 = stablehlo.reshape %v2835 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2837 = stablehlo.reshape %v2593 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2838 = stablehlo.slice %v2837 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2839 = stablehlo.reshape %v2838 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2840 = stablehlo.reshape %v2836 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2841 = stablehlo.transpose %v2840, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2842 = stablehlo.reshape %v2841 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2843 = stablehlo.reshape %v2833 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2844 = stablehlo.reshape %v2842 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2845 = stablehlo.dot_general %v2843, %v2844, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2846 = stablehlo.reshape %v2845 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2847 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2848 = stablehlo.multiply %v2846, %v2847 : tensor<32x38809xf32>
    %v2849 = stablehlo.reshape %v2848 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2850 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2851 = stablehlo.exponential %v2849 : tensor<32x197x197xf32>
    %v2852 = stablehlo.reduce(%v2851 init: %v2850) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2853 = stablehlo.broadcast_in_dim %v2852, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2854 = stablehlo.divide %v2851, %v2853 : tensor<32x197x197xf32>
    %v2855 = stablehlo.reshape %v2854 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2856 = stablehlo.reshape %v2855 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2857 = stablehlo.reshape %v2839 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2858 = stablehlo.dot_general %v2856, %v2857, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2859 = stablehlo.reshape %v2858 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2860 = stablehlo.reshape %v2859 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2861 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2862 = stablehlo.pad %v2860, %v2861, low = [0, 0, 448], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v2863 = stablehlo.reshape %v2862 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2864 = stablehlo.add %v2830, %v2863 : tensor<32x151296xf32>
    %v2865 = stablehlo.reshape %v2583 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2866 = stablehlo.slice %v2865 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2867 = stablehlo.reshape %v2866 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2868 = stablehlo.reshape %v2588 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2869 = stablehlo.slice %v2868 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2870 = stablehlo.reshape %v2869 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2871 = stablehlo.reshape %v2593 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2872 = stablehlo.slice %v2871 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2873 = stablehlo.reshape %v2872 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2874 = stablehlo.reshape %v2870 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2875 = stablehlo.transpose %v2874, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2876 = stablehlo.reshape %v2875 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2877 = stablehlo.reshape %v2867 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2878 = stablehlo.reshape %v2876 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2879 = stablehlo.dot_general %v2877, %v2878, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2880 = stablehlo.reshape %v2879 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2881 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2882 = stablehlo.multiply %v2880, %v2881 : tensor<32x38809xf32>
    %v2883 = stablehlo.reshape %v2882 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2884 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2885 = stablehlo.exponential %v2883 : tensor<32x197x197xf32>
    %v2886 = stablehlo.reduce(%v2885 init: %v2884) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2887 = stablehlo.broadcast_in_dim %v2886, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2888 = stablehlo.divide %v2885, %v2887 : tensor<32x197x197xf32>
    %v2889 = stablehlo.reshape %v2888 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2890 = stablehlo.reshape %v2889 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2891 = stablehlo.reshape %v2873 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2892 = stablehlo.dot_general %v2890, %v2891, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2893 = stablehlo.reshape %v2892 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2894 = stablehlo.reshape %v2893 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2895 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2896 = stablehlo.pad %v2894, %v2895, low = [0, 0, 512], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v2897 = stablehlo.reshape %v2896 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2898 = stablehlo.add %v2864, %v2897 : tensor<32x151296xf32>
    %v2899 = stablehlo.reshape %v2583 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2900 = stablehlo.slice %v2899 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2901 = stablehlo.reshape %v2900 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2902 = stablehlo.reshape %v2588 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2903 = stablehlo.slice %v2902 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2904 = stablehlo.reshape %v2903 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2905 = stablehlo.reshape %v2593 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2906 = stablehlo.slice %v2905 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2907 = stablehlo.reshape %v2906 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2908 = stablehlo.reshape %v2904 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2909 = stablehlo.transpose %v2908, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2910 = stablehlo.reshape %v2909 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2911 = stablehlo.reshape %v2901 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2912 = stablehlo.reshape %v2910 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2913 = stablehlo.dot_general %v2911, %v2912, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2914 = stablehlo.reshape %v2913 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2915 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2916 = stablehlo.multiply %v2914, %v2915 : tensor<32x38809xf32>
    %v2917 = stablehlo.reshape %v2916 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2918 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2919 = stablehlo.exponential %v2917 : tensor<32x197x197xf32>
    %v2920 = stablehlo.reduce(%v2919 init: %v2918) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2921 = stablehlo.broadcast_in_dim %v2920, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2922 = stablehlo.divide %v2919, %v2921 : tensor<32x197x197xf32>
    %v2923 = stablehlo.reshape %v2922 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2924 = stablehlo.reshape %v2923 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2925 = stablehlo.reshape %v2907 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2926 = stablehlo.dot_general %v2924, %v2925, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2927 = stablehlo.reshape %v2926 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2928 = stablehlo.reshape %v2927 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2929 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2930 = stablehlo.pad %v2928, %v2929, low = [0, 0, 576], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v2931 = stablehlo.reshape %v2930 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2932 = stablehlo.add %v2898, %v2931 : tensor<32x151296xf32>
    %v2933 = stablehlo.reshape %v2583 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2934 = stablehlo.slice %v2933 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2935 = stablehlo.reshape %v2934 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2936 = stablehlo.reshape %v2588 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2937 = stablehlo.slice %v2936 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2938 = stablehlo.reshape %v2937 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2939 = stablehlo.reshape %v2593 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2940 = stablehlo.slice %v2939 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2941 = stablehlo.reshape %v2940 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2942 = stablehlo.reshape %v2938 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2943 = stablehlo.transpose %v2942, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2944 = stablehlo.reshape %v2943 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2945 = stablehlo.reshape %v2935 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2946 = stablehlo.reshape %v2944 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2947 = stablehlo.dot_general %v2945, %v2946, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2948 = stablehlo.reshape %v2947 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2949 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2950 = stablehlo.multiply %v2948, %v2949 : tensor<32x38809xf32>
    %v2951 = stablehlo.reshape %v2950 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2952 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2953 = stablehlo.exponential %v2951 : tensor<32x197x197xf32>
    %v2954 = stablehlo.reduce(%v2953 init: %v2952) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2955 = stablehlo.broadcast_in_dim %v2954, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2956 = stablehlo.divide %v2953, %v2955 : tensor<32x197x197xf32>
    %v2957 = stablehlo.reshape %v2956 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2958 = stablehlo.reshape %v2957 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2959 = stablehlo.reshape %v2941 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2960 = stablehlo.dot_general %v2958, %v2959, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2961 = stablehlo.reshape %v2960 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2962 = stablehlo.reshape %v2961 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2963 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2964 = stablehlo.pad %v2962, %v2963, low = [0, 0, 640], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v2965 = stablehlo.reshape %v2964 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2966 = stablehlo.add %v2932, %v2965 : tensor<32x151296xf32>
    %v2967 = stablehlo.reshape %v2583 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2968 = stablehlo.slice %v2967 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2969 = stablehlo.reshape %v2968 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2970 = stablehlo.reshape %v2588 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2971 = stablehlo.slice %v2970 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2972 = stablehlo.reshape %v2971 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2973 = stablehlo.reshape %v2593 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2974 = stablehlo.slice %v2973 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v2975 = stablehlo.reshape %v2974 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2976 = stablehlo.reshape %v2972 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2977 = stablehlo.transpose %v2976, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2978 = stablehlo.reshape %v2977 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2979 = stablehlo.reshape %v2969 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2980 = stablehlo.reshape %v2978 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2981 = stablehlo.dot_general %v2979, %v2980, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2982 = stablehlo.reshape %v2981 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2983 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2984 = stablehlo.multiply %v2982, %v2983 : tensor<32x38809xf32>
    %v2985 = stablehlo.reshape %v2984 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2986 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2987 = stablehlo.exponential %v2985 : tensor<32x197x197xf32>
    %v2988 = stablehlo.reduce(%v2987 init: %v2986) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2989 = stablehlo.broadcast_in_dim %v2988, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2990 = stablehlo.divide %v2987, %v2989 : tensor<32x197x197xf32>
    %v2991 = stablehlo.reshape %v2990 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2992 = stablehlo.reshape %v2991 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2993 = stablehlo.reshape %v2975 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2994 = stablehlo.dot_general %v2992, %v2993, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2995 = stablehlo.reshape %v2994 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2996 = stablehlo.reshape %v2995 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2997 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2998 = stablehlo.pad %v2996, %v2997, low = [0, 0, 704], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v2999 = stablehlo.reshape %v2998 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3000 = stablehlo.add %v2966, %v2999 : tensor<32x151296xf32>
    %v3001 = stablehlo.reshape %v3000 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3002 = stablehlo.dot_general %v3001, %b5_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v3003 = stablehlo.broadcast_in_dim %b5_bo, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v3004 = stablehlo.add %v3002, %v3003 : tensor<32x197x768xf32>
    %v3005 = stablehlo.reshape %v3004 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3006 = stablehlo.add %v2550, %v3005 : tensor<32x151296xf32>
    %v3007 = stablehlo.reshape %v3006 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3008 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3009 = stablehlo.constant dense<768.0> : tensor<32x197x768xf32>
    %v3010 = stablehlo.constant dense<1.0e-5> : tensor<32x197x768xf32>
    %v3011 = stablehlo.reduce(%v3007 init: %v3008) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3012 = stablehlo.broadcast_in_dim %v3011, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v3013 = stablehlo.divide %v3012, %v3009 : tensor<32x197x768xf32>
    %v3014 = stablehlo.subtract %v3007, %v3013 : tensor<32x197x768xf32>
    %v3015 = stablehlo.multiply %v3014, %v3014 : tensor<32x197x768xf32>
    %v3016 = stablehlo.reduce(%v3015 init: %v3008) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3017 = stablehlo.broadcast_in_dim %v3016, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v3018 = stablehlo.divide %v3017, %v3009 : tensor<32x197x768xf32>
    %v3019 = stablehlo.add %v3018, %v3010 : tensor<32x197x768xf32>
    %v3020 = stablehlo.rsqrt %v3019 : tensor<32x197x768xf32>
    %v3021 = stablehlo.multiply %v3014, %v3020 : tensor<32x197x768xf32>
    %v3022 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v3023 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v3024 = stablehlo.multiply %v3021, %v3022 : tensor<32x197x768xf32>
    %v3025 = stablehlo.add %v3024, %v3023 : tensor<32x197x768xf32>
    %v3026 = stablehlo.reshape %v3025 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3027 = stablehlo.reshape %v3026 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3028 = stablehlo.broadcast_in_dim %b5_g2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v3029 = stablehlo.multiply %v3027, %v3028 : tensor<32x197x768xf32>
    %v3030 = stablehlo.reshape %v3029 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3031 = stablehlo.reshape %v3030 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3032 = stablehlo.broadcast_in_dim %b5_bt2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v3033 = stablehlo.add %v3031, %v3032 : tensor<32x197x768xf32>
    %v3034 = stablehlo.reshape %v3033 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3035 = stablehlo.reshape %v3034 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3036 = stablehlo.dot_general %v3035, %b5_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x3072xf32>) -> tensor<32x197x3072xf32>
    %v3037 = stablehlo.broadcast_in_dim %b5_bfc1, dims = [2] : (tensor<3072xf32>) -> tensor<32x197x3072xf32>
    %v3038 = stablehlo.add %v3036, %v3037 : tensor<32x197x3072xf32>
    %v3039 = stablehlo.reshape %v3038 : (tensor<32x197x3072xf32>) -> tensor<32x605184xf32>
    %v3040 = stablehlo.multiply %v3039, %v3039 : tensor<32x605184xf32>
    %v3041 = stablehlo.multiply %v3040, %v3039 : tensor<32x605184xf32>
    %v3042 = stablehlo.constant dense<0.044715> : tensor<32x605184xf32>
    %v3043 = stablehlo.multiply %v3042, %v3041 : tensor<32x605184xf32>
    %v3044 = stablehlo.add %v3039, %v3043 : tensor<32x605184xf32>
    %v3045 = stablehlo.constant dense<0.7978845608028654> : tensor<32x605184xf32>
    %v3046 = stablehlo.multiply %v3045, %v3044 : tensor<32x605184xf32>
    %v3047 = stablehlo.tanh %v3046 : tensor<32x605184xf32>
    %v3048 = stablehlo.constant dense<1.0> : tensor<32x605184xf32>
    %v3049 = stablehlo.add %v3048, %v3047 : tensor<32x605184xf32>
    %v3050 = stablehlo.constant dense<0.5> : tensor<32x605184xf32>
    %v3051 = stablehlo.multiply %v3050, %v3039 : tensor<32x605184xf32>
    %v3052 = stablehlo.multiply %v3051, %v3049 : tensor<32x605184xf32>
    %v3053 = stablehlo.reshape %v3052 : (tensor<32x605184xf32>) -> tensor<32x197x3072xf32>
    %v3054 = stablehlo.dot_general %v3053, %b5_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x3072xf32>, tensor<3072x768xf32>) -> tensor<32x197x768xf32>
    %v3055 = stablehlo.broadcast_in_dim %b5_bfc2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v3056 = stablehlo.add %v3054, %v3055 : tensor<32x197x768xf32>
    %v3057 = stablehlo.reshape %v3056 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3058 = stablehlo.add %v3006, %v3057 : tensor<32x151296xf32>
    %v3059 = stablehlo.reshape %v3058 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3060 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3061 = stablehlo.constant dense<768.0> : tensor<32x197x768xf32>
    %v3062 = stablehlo.constant dense<1.0e-5> : tensor<32x197x768xf32>
    %v3063 = stablehlo.reduce(%v3059 init: %v3060) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3064 = stablehlo.broadcast_in_dim %v3063, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v3065 = stablehlo.divide %v3064, %v3061 : tensor<32x197x768xf32>
    %v3066 = stablehlo.subtract %v3059, %v3065 : tensor<32x197x768xf32>
    %v3067 = stablehlo.multiply %v3066, %v3066 : tensor<32x197x768xf32>
    %v3068 = stablehlo.reduce(%v3067 init: %v3060) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3069 = stablehlo.broadcast_in_dim %v3068, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v3070 = stablehlo.divide %v3069, %v3061 : tensor<32x197x768xf32>
    %v3071 = stablehlo.add %v3070, %v3062 : tensor<32x197x768xf32>
    %v3072 = stablehlo.rsqrt %v3071 : tensor<32x197x768xf32>
    %v3073 = stablehlo.multiply %v3066, %v3072 : tensor<32x197x768xf32>
    %v3074 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v3075 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v3076 = stablehlo.multiply %v3073, %v3074 : tensor<32x197x768xf32>
    %v3077 = stablehlo.add %v3076, %v3075 : tensor<32x197x768xf32>
    %v3078 = stablehlo.reshape %v3077 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3079 = stablehlo.reshape %v3078 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3080 = stablehlo.broadcast_in_dim %b6_g1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v3081 = stablehlo.multiply %v3079, %v3080 : tensor<32x197x768xf32>
    %v3082 = stablehlo.reshape %v3081 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3083 = stablehlo.reshape %v3082 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3084 = stablehlo.broadcast_in_dim %b6_bt1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v3085 = stablehlo.add %v3083, %v3084 : tensor<32x197x768xf32>
    %v3086 = stablehlo.reshape %v3085 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3087 = stablehlo.reshape %v3086 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3088 = stablehlo.dot_general %v3087, %b6_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v3089 = stablehlo.broadcast_in_dim %b6_bq, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v3090 = stablehlo.add %v3088, %v3089 : tensor<32x197x768xf32>
    %v3091 = stablehlo.reshape %v3090 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3092 = stablehlo.reshape %v3086 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3093 = stablehlo.dot_general %v3092, %b6_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v3094 = stablehlo.broadcast_in_dim %b6_bk, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v3095 = stablehlo.add %v3093, %v3094 : tensor<32x197x768xf32>
    %v3096 = stablehlo.reshape %v3095 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3097 = stablehlo.reshape %v3086 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3098 = stablehlo.dot_general %v3097, %b6_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v3099 = stablehlo.broadcast_in_dim %b6_bv, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v3100 = stablehlo.add %v3098, %v3099 : tensor<32x197x768xf32>
    %v3101 = stablehlo.reshape %v3100 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3102 = stablehlo.reshape %v3091 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3103 = stablehlo.slice %v3102 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3104 = stablehlo.reshape %v3103 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3105 = stablehlo.reshape %v3096 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3106 = stablehlo.slice %v3105 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3107 = stablehlo.reshape %v3106 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3108 = stablehlo.reshape %v3101 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3109 = stablehlo.slice %v3108 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3110 = stablehlo.reshape %v3109 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3111 = stablehlo.reshape %v3107 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3112 = stablehlo.transpose %v3111, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3113 = stablehlo.reshape %v3112 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3114 = stablehlo.reshape %v3104 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3115 = stablehlo.reshape %v3113 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3116 = stablehlo.dot_general %v3114, %v3115, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3117 = stablehlo.reshape %v3116 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3118 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3119 = stablehlo.multiply %v3117, %v3118 : tensor<32x38809xf32>
    %v3120 = stablehlo.reshape %v3119 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3121 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3122 = stablehlo.exponential %v3120 : tensor<32x197x197xf32>
    %v3123 = stablehlo.reduce(%v3122 init: %v3121) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3124 = stablehlo.broadcast_in_dim %v3123, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3125 = stablehlo.divide %v3122, %v3124 : tensor<32x197x197xf32>
    %v3126 = stablehlo.reshape %v3125 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3127 = stablehlo.reshape %v3126 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3128 = stablehlo.reshape %v3110 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3129 = stablehlo.dot_general %v3127, %v3128, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3130 = stablehlo.reshape %v3129 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3131 = stablehlo.reshape %v3130 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3132 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3133 = stablehlo.pad %v3131, %v3132, low = [0, 0, 0], high = [0, 0, 704], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v3134 = stablehlo.reshape %v3133 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3135 = stablehlo.reshape %v3091 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3136 = stablehlo.slice %v3135 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3137 = stablehlo.reshape %v3136 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3138 = stablehlo.reshape %v3096 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3139 = stablehlo.slice %v3138 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3140 = stablehlo.reshape %v3139 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3141 = stablehlo.reshape %v3101 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3142 = stablehlo.slice %v3141 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3143 = stablehlo.reshape %v3142 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3144 = stablehlo.reshape %v3140 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3145 = stablehlo.transpose %v3144, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3146 = stablehlo.reshape %v3145 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3147 = stablehlo.reshape %v3137 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3148 = stablehlo.reshape %v3146 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3149 = stablehlo.dot_general %v3147, %v3148, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3150 = stablehlo.reshape %v3149 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3151 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3152 = stablehlo.multiply %v3150, %v3151 : tensor<32x38809xf32>
    %v3153 = stablehlo.reshape %v3152 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3154 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3155 = stablehlo.exponential %v3153 : tensor<32x197x197xf32>
    %v3156 = stablehlo.reduce(%v3155 init: %v3154) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3157 = stablehlo.broadcast_in_dim %v3156, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3158 = stablehlo.divide %v3155, %v3157 : tensor<32x197x197xf32>
    %v3159 = stablehlo.reshape %v3158 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3160 = stablehlo.reshape %v3159 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3161 = stablehlo.reshape %v3143 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3162 = stablehlo.dot_general %v3160, %v3161, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3163 = stablehlo.reshape %v3162 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3164 = stablehlo.reshape %v3163 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3165 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3166 = stablehlo.pad %v3164, %v3165, low = [0, 0, 64], high = [0, 0, 640], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v3167 = stablehlo.reshape %v3166 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3168 = stablehlo.add %v3134, %v3167 : tensor<32x151296xf32>
    %v3169 = stablehlo.reshape %v3091 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3170 = stablehlo.slice %v3169 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3171 = stablehlo.reshape %v3170 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3172 = stablehlo.reshape %v3096 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3173 = stablehlo.slice %v3172 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3174 = stablehlo.reshape %v3173 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3175 = stablehlo.reshape %v3101 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3176 = stablehlo.slice %v3175 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3177 = stablehlo.reshape %v3176 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3178 = stablehlo.reshape %v3174 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3179 = stablehlo.transpose %v3178, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3180 = stablehlo.reshape %v3179 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3181 = stablehlo.reshape %v3171 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3182 = stablehlo.reshape %v3180 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3183 = stablehlo.dot_general %v3181, %v3182, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3184 = stablehlo.reshape %v3183 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3185 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3186 = stablehlo.multiply %v3184, %v3185 : tensor<32x38809xf32>
    %v3187 = stablehlo.reshape %v3186 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3188 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3189 = stablehlo.exponential %v3187 : tensor<32x197x197xf32>
    %v3190 = stablehlo.reduce(%v3189 init: %v3188) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3191 = stablehlo.broadcast_in_dim %v3190, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3192 = stablehlo.divide %v3189, %v3191 : tensor<32x197x197xf32>
    %v3193 = stablehlo.reshape %v3192 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3194 = stablehlo.reshape %v3193 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3195 = stablehlo.reshape %v3177 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3196 = stablehlo.dot_general %v3194, %v3195, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3197 = stablehlo.reshape %v3196 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3198 = stablehlo.reshape %v3197 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3199 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3200 = stablehlo.pad %v3198, %v3199, low = [0, 0, 128], high = [0, 0, 576], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v3201 = stablehlo.reshape %v3200 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3202 = stablehlo.add %v3168, %v3201 : tensor<32x151296xf32>
    %v3203 = stablehlo.reshape %v3091 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3204 = stablehlo.slice %v3203 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3205 = stablehlo.reshape %v3204 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3206 = stablehlo.reshape %v3096 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3207 = stablehlo.slice %v3206 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3208 = stablehlo.reshape %v3207 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3209 = stablehlo.reshape %v3101 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3210 = stablehlo.slice %v3209 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3211 = stablehlo.reshape %v3210 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3212 = stablehlo.reshape %v3208 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3213 = stablehlo.transpose %v3212, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3214 = stablehlo.reshape %v3213 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3215 = stablehlo.reshape %v3205 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3216 = stablehlo.reshape %v3214 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3217 = stablehlo.dot_general %v3215, %v3216, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3218 = stablehlo.reshape %v3217 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3219 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3220 = stablehlo.multiply %v3218, %v3219 : tensor<32x38809xf32>
    %v3221 = stablehlo.reshape %v3220 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3222 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3223 = stablehlo.exponential %v3221 : tensor<32x197x197xf32>
    %v3224 = stablehlo.reduce(%v3223 init: %v3222) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3225 = stablehlo.broadcast_in_dim %v3224, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3226 = stablehlo.divide %v3223, %v3225 : tensor<32x197x197xf32>
    %v3227 = stablehlo.reshape %v3226 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3228 = stablehlo.reshape %v3227 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3229 = stablehlo.reshape %v3211 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3230 = stablehlo.dot_general %v3228, %v3229, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3231 = stablehlo.reshape %v3230 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3232 = stablehlo.reshape %v3231 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3233 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3234 = stablehlo.pad %v3232, %v3233, low = [0, 0, 192], high = [0, 0, 512], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v3235 = stablehlo.reshape %v3234 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3236 = stablehlo.add %v3202, %v3235 : tensor<32x151296xf32>
    %v3237 = stablehlo.reshape %v3091 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3238 = stablehlo.slice %v3237 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3239 = stablehlo.reshape %v3238 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3240 = stablehlo.reshape %v3096 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3241 = stablehlo.slice %v3240 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3242 = stablehlo.reshape %v3241 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3243 = stablehlo.reshape %v3101 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3244 = stablehlo.slice %v3243 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3245 = stablehlo.reshape %v3244 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3246 = stablehlo.reshape %v3242 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3247 = stablehlo.transpose %v3246, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3248 = stablehlo.reshape %v3247 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3249 = stablehlo.reshape %v3239 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3250 = stablehlo.reshape %v3248 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3251 = stablehlo.dot_general %v3249, %v3250, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3252 = stablehlo.reshape %v3251 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3253 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3254 = stablehlo.multiply %v3252, %v3253 : tensor<32x38809xf32>
    %v3255 = stablehlo.reshape %v3254 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3256 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3257 = stablehlo.exponential %v3255 : tensor<32x197x197xf32>
    %v3258 = stablehlo.reduce(%v3257 init: %v3256) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3259 = stablehlo.broadcast_in_dim %v3258, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3260 = stablehlo.divide %v3257, %v3259 : tensor<32x197x197xf32>
    %v3261 = stablehlo.reshape %v3260 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3262 = stablehlo.reshape %v3261 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3263 = stablehlo.reshape %v3245 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3264 = stablehlo.dot_general %v3262, %v3263, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3265 = stablehlo.reshape %v3264 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3266 = stablehlo.reshape %v3265 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3267 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3268 = stablehlo.pad %v3266, %v3267, low = [0, 0, 256], high = [0, 0, 448], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v3269 = stablehlo.reshape %v3268 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3270 = stablehlo.add %v3236, %v3269 : tensor<32x151296xf32>
    %v3271 = stablehlo.reshape %v3091 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3272 = stablehlo.slice %v3271 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3273 = stablehlo.reshape %v3272 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3274 = stablehlo.reshape %v3096 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3275 = stablehlo.slice %v3274 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3276 = stablehlo.reshape %v3275 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3277 = stablehlo.reshape %v3101 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3278 = stablehlo.slice %v3277 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3279 = stablehlo.reshape %v3278 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3280 = stablehlo.reshape %v3276 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3281 = stablehlo.transpose %v3280, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3282 = stablehlo.reshape %v3281 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3283 = stablehlo.reshape %v3273 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3284 = stablehlo.reshape %v3282 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3285 = stablehlo.dot_general %v3283, %v3284, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3286 = stablehlo.reshape %v3285 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3287 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3288 = stablehlo.multiply %v3286, %v3287 : tensor<32x38809xf32>
    %v3289 = stablehlo.reshape %v3288 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3290 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3291 = stablehlo.exponential %v3289 : tensor<32x197x197xf32>
    %v3292 = stablehlo.reduce(%v3291 init: %v3290) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3293 = stablehlo.broadcast_in_dim %v3292, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3294 = stablehlo.divide %v3291, %v3293 : tensor<32x197x197xf32>
    %v3295 = stablehlo.reshape %v3294 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3296 = stablehlo.reshape %v3295 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3297 = stablehlo.reshape %v3279 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3298 = stablehlo.dot_general %v3296, %v3297, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3299 = stablehlo.reshape %v3298 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3300 = stablehlo.reshape %v3299 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3301 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3302 = stablehlo.pad %v3300, %v3301, low = [0, 0, 320], high = [0, 0, 384], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v3303 = stablehlo.reshape %v3302 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3304 = stablehlo.add %v3270, %v3303 : tensor<32x151296xf32>
    %v3305 = stablehlo.reshape %v3091 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3306 = stablehlo.slice %v3305 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3307 = stablehlo.reshape %v3306 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3308 = stablehlo.reshape %v3096 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3309 = stablehlo.slice %v3308 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3310 = stablehlo.reshape %v3309 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3311 = stablehlo.reshape %v3101 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3312 = stablehlo.slice %v3311 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3313 = stablehlo.reshape %v3312 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3314 = stablehlo.reshape %v3310 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3315 = stablehlo.transpose %v3314, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3316 = stablehlo.reshape %v3315 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3317 = stablehlo.reshape %v3307 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3318 = stablehlo.reshape %v3316 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3319 = stablehlo.dot_general %v3317, %v3318, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3320 = stablehlo.reshape %v3319 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3321 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3322 = stablehlo.multiply %v3320, %v3321 : tensor<32x38809xf32>
    %v3323 = stablehlo.reshape %v3322 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3324 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3325 = stablehlo.exponential %v3323 : tensor<32x197x197xf32>
    %v3326 = stablehlo.reduce(%v3325 init: %v3324) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3327 = stablehlo.broadcast_in_dim %v3326, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3328 = stablehlo.divide %v3325, %v3327 : tensor<32x197x197xf32>
    %v3329 = stablehlo.reshape %v3328 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3330 = stablehlo.reshape %v3329 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3331 = stablehlo.reshape %v3313 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3332 = stablehlo.dot_general %v3330, %v3331, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3333 = stablehlo.reshape %v3332 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3334 = stablehlo.reshape %v3333 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3335 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3336 = stablehlo.pad %v3334, %v3335, low = [0, 0, 384], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v3337 = stablehlo.reshape %v3336 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3338 = stablehlo.add %v3304, %v3337 : tensor<32x151296xf32>
    %v3339 = stablehlo.reshape %v3091 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3340 = stablehlo.slice %v3339 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3341 = stablehlo.reshape %v3340 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3342 = stablehlo.reshape %v3096 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3343 = stablehlo.slice %v3342 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3344 = stablehlo.reshape %v3343 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3345 = stablehlo.reshape %v3101 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3346 = stablehlo.slice %v3345 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3347 = stablehlo.reshape %v3346 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3348 = stablehlo.reshape %v3344 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3349 = stablehlo.transpose %v3348, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3350 = stablehlo.reshape %v3349 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3351 = stablehlo.reshape %v3341 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3352 = stablehlo.reshape %v3350 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3353 = stablehlo.dot_general %v3351, %v3352, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3354 = stablehlo.reshape %v3353 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3355 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3356 = stablehlo.multiply %v3354, %v3355 : tensor<32x38809xf32>
    %v3357 = stablehlo.reshape %v3356 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3358 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3359 = stablehlo.exponential %v3357 : tensor<32x197x197xf32>
    %v3360 = stablehlo.reduce(%v3359 init: %v3358) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3361 = stablehlo.broadcast_in_dim %v3360, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3362 = stablehlo.divide %v3359, %v3361 : tensor<32x197x197xf32>
    %v3363 = stablehlo.reshape %v3362 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3364 = stablehlo.reshape %v3363 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3365 = stablehlo.reshape %v3347 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3366 = stablehlo.dot_general %v3364, %v3365, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3367 = stablehlo.reshape %v3366 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3368 = stablehlo.reshape %v3367 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3369 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3370 = stablehlo.pad %v3368, %v3369, low = [0, 0, 448], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v3371 = stablehlo.reshape %v3370 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3372 = stablehlo.add %v3338, %v3371 : tensor<32x151296xf32>
    %v3373 = stablehlo.reshape %v3091 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3374 = stablehlo.slice %v3373 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3375 = stablehlo.reshape %v3374 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3376 = stablehlo.reshape %v3096 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3377 = stablehlo.slice %v3376 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3378 = stablehlo.reshape %v3377 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3379 = stablehlo.reshape %v3101 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3380 = stablehlo.slice %v3379 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3381 = stablehlo.reshape %v3380 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3382 = stablehlo.reshape %v3378 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3383 = stablehlo.transpose %v3382, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3384 = stablehlo.reshape %v3383 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3385 = stablehlo.reshape %v3375 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3386 = stablehlo.reshape %v3384 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3387 = stablehlo.dot_general %v3385, %v3386, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3388 = stablehlo.reshape %v3387 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3389 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3390 = stablehlo.multiply %v3388, %v3389 : tensor<32x38809xf32>
    %v3391 = stablehlo.reshape %v3390 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3392 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3393 = stablehlo.exponential %v3391 : tensor<32x197x197xf32>
    %v3394 = stablehlo.reduce(%v3393 init: %v3392) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3395 = stablehlo.broadcast_in_dim %v3394, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3396 = stablehlo.divide %v3393, %v3395 : tensor<32x197x197xf32>
    %v3397 = stablehlo.reshape %v3396 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3398 = stablehlo.reshape %v3397 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3399 = stablehlo.reshape %v3381 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3400 = stablehlo.dot_general %v3398, %v3399, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3401 = stablehlo.reshape %v3400 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3402 = stablehlo.reshape %v3401 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3403 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3404 = stablehlo.pad %v3402, %v3403, low = [0, 0, 512], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v3405 = stablehlo.reshape %v3404 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3406 = stablehlo.add %v3372, %v3405 : tensor<32x151296xf32>
    %v3407 = stablehlo.reshape %v3091 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3408 = stablehlo.slice %v3407 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3409 = stablehlo.reshape %v3408 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3410 = stablehlo.reshape %v3096 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3411 = stablehlo.slice %v3410 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3412 = stablehlo.reshape %v3411 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3413 = stablehlo.reshape %v3101 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3414 = stablehlo.slice %v3413 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3415 = stablehlo.reshape %v3414 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3416 = stablehlo.reshape %v3412 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3417 = stablehlo.transpose %v3416, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3418 = stablehlo.reshape %v3417 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3419 = stablehlo.reshape %v3409 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3420 = stablehlo.reshape %v3418 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3421 = stablehlo.dot_general %v3419, %v3420, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3422 = stablehlo.reshape %v3421 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3423 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3424 = stablehlo.multiply %v3422, %v3423 : tensor<32x38809xf32>
    %v3425 = stablehlo.reshape %v3424 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3426 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3427 = stablehlo.exponential %v3425 : tensor<32x197x197xf32>
    %v3428 = stablehlo.reduce(%v3427 init: %v3426) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3429 = stablehlo.broadcast_in_dim %v3428, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3430 = stablehlo.divide %v3427, %v3429 : tensor<32x197x197xf32>
    %v3431 = stablehlo.reshape %v3430 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3432 = stablehlo.reshape %v3431 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3433 = stablehlo.reshape %v3415 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3434 = stablehlo.dot_general %v3432, %v3433, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3435 = stablehlo.reshape %v3434 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3436 = stablehlo.reshape %v3435 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3437 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3438 = stablehlo.pad %v3436, %v3437, low = [0, 0, 576], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v3439 = stablehlo.reshape %v3438 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3440 = stablehlo.add %v3406, %v3439 : tensor<32x151296xf32>
    %v3441 = stablehlo.reshape %v3091 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3442 = stablehlo.slice %v3441 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3443 = stablehlo.reshape %v3442 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3444 = stablehlo.reshape %v3096 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3445 = stablehlo.slice %v3444 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3446 = stablehlo.reshape %v3445 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3447 = stablehlo.reshape %v3101 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3448 = stablehlo.slice %v3447 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3449 = stablehlo.reshape %v3448 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3450 = stablehlo.reshape %v3446 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3451 = stablehlo.transpose %v3450, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3452 = stablehlo.reshape %v3451 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3453 = stablehlo.reshape %v3443 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3454 = stablehlo.reshape %v3452 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3455 = stablehlo.dot_general %v3453, %v3454, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3456 = stablehlo.reshape %v3455 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3457 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3458 = stablehlo.multiply %v3456, %v3457 : tensor<32x38809xf32>
    %v3459 = stablehlo.reshape %v3458 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3460 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3461 = stablehlo.exponential %v3459 : tensor<32x197x197xf32>
    %v3462 = stablehlo.reduce(%v3461 init: %v3460) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3463 = stablehlo.broadcast_in_dim %v3462, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3464 = stablehlo.divide %v3461, %v3463 : tensor<32x197x197xf32>
    %v3465 = stablehlo.reshape %v3464 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3466 = stablehlo.reshape %v3465 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3467 = stablehlo.reshape %v3449 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3468 = stablehlo.dot_general %v3466, %v3467, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3469 = stablehlo.reshape %v3468 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3470 = stablehlo.reshape %v3469 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3471 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3472 = stablehlo.pad %v3470, %v3471, low = [0, 0, 640], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v3473 = stablehlo.reshape %v3472 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3474 = stablehlo.add %v3440, %v3473 : tensor<32x151296xf32>
    %v3475 = stablehlo.reshape %v3091 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3476 = stablehlo.slice %v3475 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3477 = stablehlo.reshape %v3476 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3478 = stablehlo.reshape %v3096 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3479 = stablehlo.slice %v3478 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3480 = stablehlo.reshape %v3479 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3481 = stablehlo.reshape %v3101 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3482 = stablehlo.slice %v3481 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3483 = stablehlo.reshape %v3482 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3484 = stablehlo.reshape %v3480 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3485 = stablehlo.transpose %v3484, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3486 = stablehlo.reshape %v3485 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3487 = stablehlo.reshape %v3477 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3488 = stablehlo.reshape %v3486 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3489 = stablehlo.dot_general %v3487, %v3488, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3490 = stablehlo.reshape %v3489 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3491 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3492 = stablehlo.multiply %v3490, %v3491 : tensor<32x38809xf32>
    %v3493 = stablehlo.reshape %v3492 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3494 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3495 = stablehlo.exponential %v3493 : tensor<32x197x197xf32>
    %v3496 = stablehlo.reduce(%v3495 init: %v3494) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3497 = stablehlo.broadcast_in_dim %v3496, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3498 = stablehlo.divide %v3495, %v3497 : tensor<32x197x197xf32>
    %v3499 = stablehlo.reshape %v3498 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3500 = stablehlo.reshape %v3499 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3501 = stablehlo.reshape %v3483 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3502 = stablehlo.dot_general %v3500, %v3501, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3503 = stablehlo.reshape %v3502 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3504 = stablehlo.reshape %v3503 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3505 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3506 = stablehlo.pad %v3504, %v3505, low = [0, 0, 704], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v3507 = stablehlo.reshape %v3506 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3508 = stablehlo.add %v3474, %v3507 : tensor<32x151296xf32>
    %v3509 = stablehlo.reshape %v3508 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3510 = stablehlo.dot_general %v3509, %b6_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v3511 = stablehlo.broadcast_in_dim %b6_bo, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v3512 = stablehlo.add %v3510, %v3511 : tensor<32x197x768xf32>
    %v3513 = stablehlo.reshape %v3512 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3514 = stablehlo.add %v3058, %v3513 : tensor<32x151296xf32>
    %v3515 = stablehlo.reshape %v3514 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3516 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3517 = stablehlo.constant dense<768.0> : tensor<32x197x768xf32>
    %v3518 = stablehlo.constant dense<1.0e-5> : tensor<32x197x768xf32>
    %v3519 = stablehlo.reduce(%v3515 init: %v3516) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3520 = stablehlo.broadcast_in_dim %v3519, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v3521 = stablehlo.divide %v3520, %v3517 : tensor<32x197x768xf32>
    %v3522 = stablehlo.subtract %v3515, %v3521 : tensor<32x197x768xf32>
    %v3523 = stablehlo.multiply %v3522, %v3522 : tensor<32x197x768xf32>
    %v3524 = stablehlo.reduce(%v3523 init: %v3516) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3525 = stablehlo.broadcast_in_dim %v3524, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v3526 = stablehlo.divide %v3525, %v3517 : tensor<32x197x768xf32>
    %v3527 = stablehlo.add %v3526, %v3518 : tensor<32x197x768xf32>
    %v3528 = stablehlo.rsqrt %v3527 : tensor<32x197x768xf32>
    %v3529 = stablehlo.multiply %v3522, %v3528 : tensor<32x197x768xf32>
    %v3530 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v3531 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v3532 = stablehlo.multiply %v3529, %v3530 : tensor<32x197x768xf32>
    %v3533 = stablehlo.add %v3532, %v3531 : tensor<32x197x768xf32>
    %v3534 = stablehlo.reshape %v3533 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3535 = stablehlo.reshape %v3534 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3536 = stablehlo.broadcast_in_dim %b6_g2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v3537 = stablehlo.multiply %v3535, %v3536 : tensor<32x197x768xf32>
    %v3538 = stablehlo.reshape %v3537 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3539 = stablehlo.reshape %v3538 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3540 = stablehlo.broadcast_in_dim %b6_bt2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v3541 = stablehlo.add %v3539, %v3540 : tensor<32x197x768xf32>
    %v3542 = stablehlo.reshape %v3541 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3543 = stablehlo.reshape %v3542 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3544 = stablehlo.dot_general %v3543, %b6_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x3072xf32>) -> tensor<32x197x3072xf32>
    %v3545 = stablehlo.broadcast_in_dim %b6_bfc1, dims = [2] : (tensor<3072xf32>) -> tensor<32x197x3072xf32>
    %v3546 = stablehlo.add %v3544, %v3545 : tensor<32x197x3072xf32>
    %v3547 = stablehlo.reshape %v3546 : (tensor<32x197x3072xf32>) -> tensor<32x605184xf32>
    %v3548 = stablehlo.multiply %v3547, %v3547 : tensor<32x605184xf32>
    %v3549 = stablehlo.multiply %v3548, %v3547 : tensor<32x605184xf32>
    %v3550 = stablehlo.constant dense<0.044715> : tensor<32x605184xf32>
    %v3551 = stablehlo.multiply %v3550, %v3549 : tensor<32x605184xf32>
    %v3552 = stablehlo.add %v3547, %v3551 : tensor<32x605184xf32>
    %v3553 = stablehlo.constant dense<0.7978845608028654> : tensor<32x605184xf32>
    %v3554 = stablehlo.multiply %v3553, %v3552 : tensor<32x605184xf32>
    %v3555 = stablehlo.tanh %v3554 : tensor<32x605184xf32>
    %v3556 = stablehlo.constant dense<1.0> : tensor<32x605184xf32>
    %v3557 = stablehlo.add %v3556, %v3555 : tensor<32x605184xf32>
    %v3558 = stablehlo.constant dense<0.5> : tensor<32x605184xf32>
    %v3559 = stablehlo.multiply %v3558, %v3547 : tensor<32x605184xf32>
    %v3560 = stablehlo.multiply %v3559, %v3557 : tensor<32x605184xf32>
    %v3561 = stablehlo.reshape %v3560 : (tensor<32x605184xf32>) -> tensor<32x197x3072xf32>
    %v3562 = stablehlo.dot_general %v3561, %b6_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x3072xf32>, tensor<3072x768xf32>) -> tensor<32x197x768xf32>
    %v3563 = stablehlo.broadcast_in_dim %b6_bfc2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v3564 = stablehlo.add %v3562, %v3563 : tensor<32x197x768xf32>
    %v3565 = stablehlo.reshape %v3564 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3566 = stablehlo.add %v3514, %v3565 : tensor<32x151296xf32>
    %v3567 = stablehlo.reshape %v3566 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3568 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3569 = stablehlo.constant dense<768.0> : tensor<32x197x768xf32>
    %v3570 = stablehlo.constant dense<1.0e-5> : tensor<32x197x768xf32>
    %v3571 = stablehlo.reduce(%v3567 init: %v3568) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3572 = stablehlo.broadcast_in_dim %v3571, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v3573 = stablehlo.divide %v3572, %v3569 : tensor<32x197x768xf32>
    %v3574 = stablehlo.subtract %v3567, %v3573 : tensor<32x197x768xf32>
    %v3575 = stablehlo.multiply %v3574, %v3574 : tensor<32x197x768xf32>
    %v3576 = stablehlo.reduce(%v3575 init: %v3568) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3577 = stablehlo.broadcast_in_dim %v3576, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v3578 = stablehlo.divide %v3577, %v3569 : tensor<32x197x768xf32>
    %v3579 = stablehlo.add %v3578, %v3570 : tensor<32x197x768xf32>
    %v3580 = stablehlo.rsqrt %v3579 : tensor<32x197x768xf32>
    %v3581 = stablehlo.multiply %v3574, %v3580 : tensor<32x197x768xf32>
    %v3582 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v3583 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v3584 = stablehlo.multiply %v3581, %v3582 : tensor<32x197x768xf32>
    %v3585 = stablehlo.add %v3584, %v3583 : tensor<32x197x768xf32>
    %v3586 = stablehlo.reshape %v3585 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3587 = stablehlo.reshape %v3586 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3588 = stablehlo.broadcast_in_dim %b7_g1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v3589 = stablehlo.multiply %v3587, %v3588 : tensor<32x197x768xf32>
    %v3590 = stablehlo.reshape %v3589 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3591 = stablehlo.reshape %v3590 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3592 = stablehlo.broadcast_in_dim %b7_bt1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v3593 = stablehlo.add %v3591, %v3592 : tensor<32x197x768xf32>
    %v3594 = stablehlo.reshape %v3593 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3595 = stablehlo.reshape %v3594 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3596 = stablehlo.dot_general %v3595, %b7_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v3597 = stablehlo.broadcast_in_dim %b7_bq, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v3598 = stablehlo.add %v3596, %v3597 : tensor<32x197x768xf32>
    %v3599 = stablehlo.reshape %v3598 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3600 = stablehlo.reshape %v3594 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3601 = stablehlo.dot_general %v3600, %b7_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v3602 = stablehlo.broadcast_in_dim %b7_bk, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v3603 = stablehlo.add %v3601, %v3602 : tensor<32x197x768xf32>
    %v3604 = stablehlo.reshape %v3603 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3605 = stablehlo.reshape %v3594 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3606 = stablehlo.dot_general %v3605, %b7_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v3607 = stablehlo.broadcast_in_dim %b7_bv, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v3608 = stablehlo.add %v3606, %v3607 : tensor<32x197x768xf32>
    %v3609 = stablehlo.reshape %v3608 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3610 = stablehlo.reshape %v3599 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3611 = stablehlo.slice %v3610 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3612 = stablehlo.reshape %v3611 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3613 = stablehlo.reshape %v3604 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3614 = stablehlo.slice %v3613 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3615 = stablehlo.reshape %v3614 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3616 = stablehlo.reshape %v3609 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3617 = stablehlo.slice %v3616 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3618 = stablehlo.reshape %v3617 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3619 = stablehlo.reshape %v3615 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3620 = stablehlo.transpose %v3619, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3621 = stablehlo.reshape %v3620 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3622 = stablehlo.reshape %v3612 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3623 = stablehlo.reshape %v3621 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3624 = stablehlo.dot_general %v3622, %v3623, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3625 = stablehlo.reshape %v3624 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3626 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3627 = stablehlo.multiply %v3625, %v3626 : tensor<32x38809xf32>
    %v3628 = stablehlo.reshape %v3627 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3629 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3630 = stablehlo.exponential %v3628 : tensor<32x197x197xf32>
    %v3631 = stablehlo.reduce(%v3630 init: %v3629) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3632 = stablehlo.broadcast_in_dim %v3631, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3633 = stablehlo.divide %v3630, %v3632 : tensor<32x197x197xf32>
    %v3634 = stablehlo.reshape %v3633 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3635 = stablehlo.reshape %v3634 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3636 = stablehlo.reshape %v3618 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3637 = stablehlo.dot_general %v3635, %v3636, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3638 = stablehlo.reshape %v3637 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3639 = stablehlo.reshape %v3638 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3640 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3641 = stablehlo.pad %v3639, %v3640, low = [0, 0, 0], high = [0, 0, 704], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v3642 = stablehlo.reshape %v3641 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3643 = stablehlo.reshape %v3599 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3644 = stablehlo.slice %v3643 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3645 = stablehlo.reshape %v3644 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3646 = stablehlo.reshape %v3604 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3647 = stablehlo.slice %v3646 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3648 = stablehlo.reshape %v3647 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3649 = stablehlo.reshape %v3609 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3650 = stablehlo.slice %v3649 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3651 = stablehlo.reshape %v3650 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3652 = stablehlo.reshape %v3648 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3653 = stablehlo.transpose %v3652, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3654 = stablehlo.reshape %v3653 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3655 = stablehlo.reshape %v3645 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3656 = stablehlo.reshape %v3654 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3657 = stablehlo.dot_general %v3655, %v3656, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3658 = stablehlo.reshape %v3657 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3659 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3660 = stablehlo.multiply %v3658, %v3659 : tensor<32x38809xf32>
    %v3661 = stablehlo.reshape %v3660 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3662 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3663 = stablehlo.exponential %v3661 : tensor<32x197x197xf32>
    %v3664 = stablehlo.reduce(%v3663 init: %v3662) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3665 = stablehlo.broadcast_in_dim %v3664, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3666 = stablehlo.divide %v3663, %v3665 : tensor<32x197x197xf32>
    %v3667 = stablehlo.reshape %v3666 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3668 = stablehlo.reshape %v3667 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3669 = stablehlo.reshape %v3651 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3670 = stablehlo.dot_general %v3668, %v3669, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3671 = stablehlo.reshape %v3670 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3672 = stablehlo.reshape %v3671 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3673 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3674 = stablehlo.pad %v3672, %v3673, low = [0, 0, 64], high = [0, 0, 640], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v3675 = stablehlo.reshape %v3674 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3676 = stablehlo.add %v3642, %v3675 : tensor<32x151296xf32>
    %v3677 = stablehlo.reshape %v3599 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3678 = stablehlo.slice %v3677 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3679 = stablehlo.reshape %v3678 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3680 = stablehlo.reshape %v3604 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3681 = stablehlo.slice %v3680 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3682 = stablehlo.reshape %v3681 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3683 = stablehlo.reshape %v3609 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3684 = stablehlo.slice %v3683 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3685 = stablehlo.reshape %v3684 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3686 = stablehlo.reshape %v3682 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3687 = stablehlo.transpose %v3686, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3688 = stablehlo.reshape %v3687 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3689 = stablehlo.reshape %v3679 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3690 = stablehlo.reshape %v3688 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3691 = stablehlo.dot_general %v3689, %v3690, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3692 = stablehlo.reshape %v3691 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3693 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3694 = stablehlo.multiply %v3692, %v3693 : tensor<32x38809xf32>
    %v3695 = stablehlo.reshape %v3694 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3696 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3697 = stablehlo.exponential %v3695 : tensor<32x197x197xf32>
    %v3698 = stablehlo.reduce(%v3697 init: %v3696) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3699 = stablehlo.broadcast_in_dim %v3698, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3700 = stablehlo.divide %v3697, %v3699 : tensor<32x197x197xf32>
    %v3701 = stablehlo.reshape %v3700 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3702 = stablehlo.reshape %v3701 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3703 = stablehlo.reshape %v3685 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3704 = stablehlo.dot_general %v3702, %v3703, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3705 = stablehlo.reshape %v3704 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3706 = stablehlo.reshape %v3705 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3707 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3708 = stablehlo.pad %v3706, %v3707, low = [0, 0, 128], high = [0, 0, 576], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v3709 = stablehlo.reshape %v3708 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3710 = stablehlo.add %v3676, %v3709 : tensor<32x151296xf32>
    %v3711 = stablehlo.reshape %v3599 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3712 = stablehlo.slice %v3711 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3713 = stablehlo.reshape %v3712 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3714 = stablehlo.reshape %v3604 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3715 = stablehlo.slice %v3714 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3716 = stablehlo.reshape %v3715 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3717 = stablehlo.reshape %v3609 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3718 = stablehlo.slice %v3717 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3719 = stablehlo.reshape %v3718 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3720 = stablehlo.reshape %v3716 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3721 = stablehlo.transpose %v3720, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3722 = stablehlo.reshape %v3721 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3723 = stablehlo.reshape %v3713 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3724 = stablehlo.reshape %v3722 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3725 = stablehlo.dot_general %v3723, %v3724, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3726 = stablehlo.reshape %v3725 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3727 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3728 = stablehlo.multiply %v3726, %v3727 : tensor<32x38809xf32>
    %v3729 = stablehlo.reshape %v3728 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3730 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3731 = stablehlo.exponential %v3729 : tensor<32x197x197xf32>
    %v3732 = stablehlo.reduce(%v3731 init: %v3730) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3733 = stablehlo.broadcast_in_dim %v3732, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3734 = stablehlo.divide %v3731, %v3733 : tensor<32x197x197xf32>
    %v3735 = stablehlo.reshape %v3734 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3736 = stablehlo.reshape %v3735 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3737 = stablehlo.reshape %v3719 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3738 = stablehlo.dot_general %v3736, %v3737, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3739 = stablehlo.reshape %v3738 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3740 = stablehlo.reshape %v3739 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3741 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3742 = stablehlo.pad %v3740, %v3741, low = [0, 0, 192], high = [0, 0, 512], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v3743 = stablehlo.reshape %v3742 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3744 = stablehlo.add %v3710, %v3743 : tensor<32x151296xf32>
    %v3745 = stablehlo.reshape %v3599 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3746 = stablehlo.slice %v3745 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3747 = stablehlo.reshape %v3746 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3748 = stablehlo.reshape %v3604 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3749 = stablehlo.slice %v3748 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3750 = stablehlo.reshape %v3749 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3751 = stablehlo.reshape %v3609 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3752 = stablehlo.slice %v3751 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3753 = stablehlo.reshape %v3752 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3754 = stablehlo.reshape %v3750 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3755 = stablehlo.transpose %v3754, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3756 = stablehlo.reshape %v3755 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3757 = stablehlo.reshape %v3747 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3758 = stablehlo.reshape %v3756 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3759 = stablehlo.dot_general %v3757, %v3758, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3760 = stablehlo.reshape %v3759 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3761 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3762 = stablehlo.multiply %v3760, %v3761 : tensor<32x38809xf32>
    %v3763 = stablehlo.reshape %v3762 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3764 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3765 = stablehlo.exponential %v3763 : tensor<32x197x197xf32>
    %v3766 = stablehlo.reduce(%v3765 init: %v3764) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3767 = stablehlo.broadcast_in_dim %v3766, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3768 = stablehlo.divide %v3765, %v3767 : tensor<32x197x197xf32>
    %v3769 = stablehlo.reshape %v3768 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3770 = stablehlo.reshape %v3769 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3771 = stablehlo.reshape %v3753 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3772 = stablehlo.dot_general %v3770, %v3771, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3773 = stablehlo.reshape %v3772 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3774 = stablehlo.reshape %v3773 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3775 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3776 = stablehlo.pad %v3774, %v3775, low = [0, 0, 256], high = [0, 0, 448], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v3777 = stablehlo.reshape %v3776 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3778 = stablehlo.add %v3744, %v3777 : tensor<32x151296xf32>
    %v3779 = stablehlo.reshape %v3599 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3780 = stablehlo.slice %v3779 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3781 = stablehlo.reshape %v3780 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3782 = stablehlo.reshape %v3604 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3783 = stablehlo.slice %v3782 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3784 = stablehlo.reshape %v3783 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3785 = stablehlo.reshape %v3609 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3786 = stablehlo.slice %v3785 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3787 = stablehlo.reshape %v3786 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3788 = stablehlo.reshape %v3784 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3789 = stablehlo.transpose %v3788, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3790 = stablehlo.reshape %v3789 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3791 = stablehlo.reshape %v3781 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3792 = stablehlo.reshape %v3790 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3793 = stablehlo.dot_general %v3791, %v3792, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3794 = stablehlo.reshape %v3793 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3795 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3796 = stablehlo.multiply %v3794, %v3795 : tensor<32x38809xf32>
    %v3797 = stablehlo.reshape %v3796 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3798 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3799 = stablehlo.exponential %v3797 : tensor<32x197x197xf32>
    %v3800 = stablehlo.reduce(%v3799 init: %v3798) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3801 = stablehlo.broadcast_in_dim %v3800, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3802 = stablehlo.divide %v3799, %v3801 : tensor<32x197x197xf32>
    %v3803 = stablehlo.reshape %v3802 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3804 = stablehlo.reshape %v3803 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3805 = stablehlo.reshape %v3787 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3806 = stablehlo.dot_general %v3804, %v3805, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3807 = stablehlo.reshape %v3806 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3808 = stablehlo.reshape %v3807 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3809 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3810 = stablehlo.pad %v3808, %v3809, low = [0, 0, 320], high = [0, 0, 384], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v3811 = stablehlo.reshape %v3810 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3812 = stablehlo.add %v3778, %v3811 : tensor<32x151296xf32>
    %v3813 = stablehlo.reshape %v3599 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3814 = stablehlo.slice %v3813 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3815 = stablehlo.reshape %v3814 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3816 = stablehlo.reshape %v3604 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3817 = stablehlo.slice %v3816 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3818 = stablehlo.reshape %v3817 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3819 = stablehlo.reshape %v3609 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3820 = stablehlo.slice %v3819 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3821 = stablehlo.reshape %v3820 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3822 = stablehlo.reshape %v3818 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3823 = stablehlo.transpose %v3822, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3824 = stablehlo.reshape %v3823 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3825 = stablehlo.reshape %v3815 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3826 = stablehlo.reshape %v3824 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3827 = stablehlo.dot_general %v3825, %v3826, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3828 = stablehlo.reshape %v3827 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3829 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3830 = stablehlo.multiply %v3828, %v3829 : tensor<32x38809xf32>
    %v3831 = stablehlo.reshape %v3830 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3832 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3833 = stablehlo.exponential %v3831 : tensor<32x197x197xf32>
    %v3834 = stablehlo.reduce(%v3833 init: %v3832) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3835 = stablehlo.broadcast_in_dim %v3834, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3836 = stablehlo.divide %v3833, %v3835 : tensor<32x197x197xf32>
    %v3837 = stablehlo.reshape %v3836 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3838 = stablehlo.reshape %v3837 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3839 = stablehlo.reshape %v3821 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3840 = stablehlo.dot_general %v3838, %v3839, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3841 = stablehlo.reshape %v3840 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3842 = stablehlo.reshape %v3841 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3843 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3844 = stablehlo.pad %v3842, %v3843, low = [0, 0, 384], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v3845 = stablehlo.reshape %v3844 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3846 = stablehlo.add %v3812, %v3845 : tensor<32x151296xf32>
    %v3847 = stablehlo.reshape %v3599 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3848 = stablehlo.slice %v3847 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3849 = stablehlo.reshape %v3848 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3850 = stablehlo.reshape %v3604 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3851 = stablehlo.slice %v3850 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3852 = stablehlo.reshape %v3851 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3853 = stablehlo.reshape %v3609 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3854 = stablehlo.slice %v3853 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3855 = stablehlo.reshape %v3854 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3856 = stablehlo.reshape %v3852 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3857 = stablehlo.transpose %v3856, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3858 = stablehlo.reshape %v3857 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3859 = stablehlo.reshape %v3849 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3860 = stablehlo.reshape %v3858 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3861 = stablehlo.dot_general %v3859, %v3860, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3862 = stablehlo.reshape %v3861 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3863 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3864 = stablehlo.multiply %v3862, %v3863 : tensor<32x38809xf32>
    %v3865 = stablehlo.reshape %v3864 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3866 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3867 = stablehlo.exponential %v3865 : tensor<32x197x197xf32>
    %v3868 = stablehlo.reduce(%v3867 init: %v3866) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3869 = stablehlo.broadcast_in_dim %v3868, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3870 = stablehlo.divide %v3867, %v3869 : tensor<32x197x197xf32>
    %v3871 = stablehlo.reshape %v3870 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3872 = stablehlo.reshape %v3871 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3873 = stablehlo.reshape %v3855 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3874 = stablehlo.dot_general %v3872, %v3873, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3875 = stablehlo.reshape %v3874 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3876 = stablehlo.reshape %v3875 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3877 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3878 = stablehlo.pad %v3876, %v3877, low = [0, 0, 448], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v3879 = stablehlo.reshape %v3878 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3880 = stablehlo.add %v3846, %v3879 : tensor<32x151296xf32>
    %v3881 = stablehlo.reshape %v3599 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3882 = stablehlo.slice %v3881 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3883 = stablehlo.reshape %v3882 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3884 = stablehlo.reshape %v3604 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3885 = stablehlo.slice %v3884 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3886 = stablehlo.reshape %v3885 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3887 = stablehlo.reshape %v3609 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3888 = stablehlo.slice %v3887 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3889 = stablehlo.reshape %v3888 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3890 = stablehlo.reshape %v3886 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3891 = stablehlo.transpose %v3890, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3892 = stablehlo.reshape %v3891 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3893 = stablehlo.reshape %v3883 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3894 = stablehlo.reshape %v3892 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3895 = stablehlo.dot_general %v3893, %v3894, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3896 = stablehlo.reshape %v3895 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3897 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3898 = stablehlo.multiply %v3896, %v3897 : tensor<32x38809xf32>
    %v3899 = stablehlo.reshape %v3898 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3900 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3901 = stablehlo.exponential %v3899 : tensor<32x197x197xf32>
    %v3902 = stablehlo.reduce(%v3901 init: %v3900) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3903 = stablehlo.broadcast_in_dim %v3902, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3904 = stablehlo.divide %v3901, %v3903 : tensor<32x197x197xf32>
    %v3905 = stablehlo.reshape %v3904 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3906 = stablehlo.reshape %v3905 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3907 = stablehlo.reshape %v3889 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3908 = stablehlo.dot_general %v3906, %v3907, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3909 = stablehlo.reshape %v3908 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3910 = stablehlo.reshape %v3909 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3911 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3912 = stablehlo.pad %v3910, %v3911, low = [0, 0, 512], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v3913 = stablehlo.reshape %v3912 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3914 = stablehlo.add %v3880, %v3913 : tensor<32x151296xf32>
    %v3915 = stablehlo.reshape %v3599 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3916 = stablehlo.slice %v3915 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3917 = stablehlo.reshape %v3916 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3918 = stablehlo.reshape %v3604 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3919 = stablehlo.slice %v3918 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3920 = stablehlo.reshape %v3919 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3921 = stablehlo.reshape %v3609 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3922 = stablehlo.slice %v3921 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3923 = stablehlo.reshape %v3922 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3924 = stablehlo.reshape %v3920 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3925 = stablehlo.transpose %v3924, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3926 = stablehlo.reshape %v3925 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3927 = stablehlo.reshape %v3917 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3928 = stablehlo.reshape %v3926 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3929 = stablehlo.dot_general %v3927, %v3928, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3930 = stablehlo.reshape %v3929 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3931 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3932 = stablehlo.multiply %v3930, %v3931 : tensor<32x38809xf32>
    %v3933 = stablehlo.reshape %v3932 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3934 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3935 = stablehlo.exponential %v3933 : tensor<32x197x197xf32>
    %v3936 = stablehlo.reduce(%v3935 init: %v3934) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3937 = stablehlo.broadcast_in_dim %v3936, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3938 = stablehlo.divide %v3935, %v3937 : tensor<32x197x197xf32>
    %v3939 = stablehlo.reshape %v3938 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3940 = stablehlo.reshape %v3939 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3941 = stablehlo.reshape %v3923 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3942 = stablehlo.dot_general %v3940, %v3941, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3943 = stablehlo.reshape %v3942 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3944 = stablehlo.reshape %v3943 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3945 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3946 = stablehlo.pad %v3944, %v3945, low = [0, 0, 576], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v3947 = stablehlo.reshape %v3946 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3948 = stablehlo.add %v3914, %v3947 : tensor<32x151296xf32>
    %v3949 = stablehlo.reshape %v3599 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3950 = stablehlo.slice %v3949 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3951 = stablehlo.reshape %v3950 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3952 = stablehlo.reshape %v3604 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3953 = stablehlo.slice %v3952 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3954 = stablehlo.reshape %v3953 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3955 = stablehlo.reshape %v3609 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3956 = stablehlo.slice %v3955 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3957 = stablehlo.reshape %v3956 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3958 = stablehlo.reshape %v3954 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3959 = stablehlo.transpose %v3958, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3960 = stablehlo.reshape %v3959 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3961 = stablehlo.reshape %v3951 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3962 = stablehlo.reshape %v3960 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3963 = stablehlo.dot_general %v3961, %v3962, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3964 = stablehlo.reshape %v3963 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3965 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3966 = stablehlo.multiply %v3964, %v3965 : tensor<32x38809xf32>
    %v3967 = stablehlo.reshape %v3966 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3968 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3969 = stablehlo.exponential %v3967 : tensor<32x197x197xf32>
    %v3970 = stablehlo.reduce(%v3969 init: %v3968) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3971 = stablehlo.broadcast_in_dim %v3970, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3972 = stablehlo.divide %v3969, %v3971 : tensor<32x197x197xf32>
    %v3973 = stablehlo.reshape %v3972 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3974 = stablehlo.reshape %v3973 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3975 = stablehlo.reshape %v3957 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3976 = stablehlo.dot_general %v3974, %v3975, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3977 = stablehlo.reshape %v3976 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3978 = stablehlo.reshape %v3977 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3979 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3980 = stablehlo.pad %v3978, %v3979, low = [0, 0, 640], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v3981 = stablehlo.reshape %v3980 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v3982 = stablehlo.add %v3948, %v3981 : tensor<32x151296xf32>
    %v3983 = stablehlo.reshape %v3599 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3984 = stablehlo.slice %v3983 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3985 = stablehlo.reshape %v3984 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3986 = stablehlo.reshape %v3604 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3987 = stablehlo.slice %v3986 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3988 = stablehlo.reshape %v3987 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3989 = stablehlo.reshape %v3609 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v3990 = stablehlo.slice %v3989 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v3991 = stablehlo.reshape %v3990 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3992 = stablehlo.reshape %v3988 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3993 = stablehlo.transpose %v3992, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3994 = stablehlo.reshape %v3993 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3995 = stablehlo.reshape %v3985 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3996 = stablehlo.reshape %v3994 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3997 = stablehlo.dot_general %v3995, %v3996, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3998 = stablehlo.reshape %v3997 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3999 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v4000 = stablehlo.multiply %v3998, %v3999 : tensor<32x38809xf32>
    %v4001 = stablehlo.reshape %v4000 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4002 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4003 = stablehlo.exponential %v4001 : tensor<32x197x197xf32>
    %v4004 = stablehlo.reduce(%v4003 init: %v4002) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4005 = stablehlo.broadcast_in_dim %v4004, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4006 = stablehlo.divide %v4003, %v4005 : tensor<32x197x197xf32>
    %v4007 = stablehlo.reshape %v4006 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4008 = stablehlo.reshape %v4007 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4009 = stablehlo.reshape %v3991 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4010 = stablehlo.dot_general %v4008, %v4009, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4011 = stablehlo.reshape %v4010 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4012 = stablehlo.reshape %v4011 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4013 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4014 = stablehlo.pad %v4012, %v4013, low = [0, 0, 704], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v4015 = stablehlo.reshape %v4014 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4016 = stablehlo.add %v3982, %v4015 : tensor<32x151296xf32>
    %v4017 = stablehlo.reshape %v4016 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4018 = stablehlo.dot_general %v4017, %b7_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v4019 = stablehlo.broadcast_in_dim %b7_bo, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v4020 = stablehlo.add %v4018, %v4019 : tensor<32x197x768xf32>
    %v4021 = stablehlo.reshape %v4020 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4022 = stablehlo.add %v3566, %v4021 : tensor<32x151296xf32>
    %v4023 = stablehlo.reshape %v4022 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4024 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4025 = stablehlo.constant dense<768.0> : tensor<32x197x768xf32>
    %v4026 = stablehlo.constant dense<1.0e-5> : tensor<32x197x768xf32>
    %v4027 = stablehlo.reduce(%v4023 init: %v4024) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4028 = stablehlo.broadcast_in_dim %v4027, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v4029 = stablehlo.divide %v4028, %v4025 : tensor<32x197x768xf32>
    %v4030 = stablehlo.subtract %v4023, %v4029 : tensor<32x197x768xf32>
    %v4031 = stablehlo.multiply %v4030, %v4030 : tensor<32x197x768xf32>
    %v4032 = stablehlo.reduce(%v4031 init: %v4024) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4033 = stablehlo.broadcast_in_dim %v4032, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v4034 = stablehlo.divide %v4033, %v4025 : tensor<32x197x768xf32>
    %v4035 = stablehlo.add %v4034, %v4026 : tensor<32x197x768xf32>
    %v4036 = stablehlo.rsqrt %v4035 : tensor<32x197x768xf32>
    %v4037 = stablehlo.multiply %v4030, %v4036 : tensor<32x197x768xf32>
    %v4038 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v4039 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v4040 = stablehlo.multiply %v4037, %v4038 : tensor<32x197x768xf32>
    %v4041 = stablehlo.add %v4040, %v4039 : tensor<32x197x768xf32>
    %v4042 = stablehlo.reshape %v4041 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4043 = stablehlo.reshape %v4042 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4044 = stablehlo.broadcast_in_dim %b7_g2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v4045 = stablehlo.multiply %v4043, %v4044 : tensor<32x197x768xf32>
    %v4046 = stablehlo.reshape %v4045 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4047 = stablehlo.reshape %v4046 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4048 = stablehlo.broadcast_in_dim %b7_bt2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v4049 = stablehlo.add %v4047, %v4048 : tensor<32x197x768xf32>
    %v4050 = stablehlo.reshape %v4049 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4051 = stablehlo.reshape %v4050 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4052 = stablehlo.dot_general %v4051, %b7_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x3072xf32>) -> tensor<32x197x3072xf32>
    %v4053 = stablehlo.broadcast_in_dim %b7_bfc1, dims = [2] : (tensor<3072xf32>) -> tensor<32x197x3072xf32>
    %v4054 = stablehlo.add %v4052, %v4053 : tensor<32x197x3072xf32>
    %v4055 = stablehlo.reshape %v4054 : (tensor<32x197x3072xf32>) -> tensor<32x605184xf32>
    %v4056 = stablehlo.multiply %v4055, %v4055 : tensor<32x605184xf32>
    %v4057 = stablehlo.multiply %v4056, %v4055 : tensor<32x605184xf32>
    %v4058 = stablehlo.constant dense<0.044715> : tensor<32x605184xf32>
    %v4059 = stablehlo.multiply %v4058, %v4057 : tensor<32x605184xf32>
    %v4060 = stablehlo.add %v4055, %v4059 : tensor<32x605184xf32>
    %v4061 = stablehlo.constant dense<0.7978845608028654> : tensor<32x605184xf32>
    %v4062 = stablehlo.multiply %v4061, %v4060 : tensor<32x605184xf32>
    %v4063 = stablehlo.tanh %v4062 : tensor<32x605184xf32>
    %v4064 = stablehlo.constant dense<1.0> : tensor<32x605184xf32>
    %v4065 = stablehlo.add %v4064, %v4063 : tensor<32x605184xf32>
    %v4066 = stablehlo.constant dense<0.5> : tensor<32x605184xf32>
    %v4067 = stablehlo.multiply %v4066, %v4055 : tensor<32x605184xf32>
    %v4068 = stablehlo.multiply %v4067, %v4065 : tensor<32x605184xf32>
    %v4069 = stablehlo.reshape %v4068 : (tensor<32x605184xf32>) -> tensor<32x197x3072xf32>
    %v4070 = stablehlo.dot_general %v4069, %b7_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x3072xf32>, tensor<3072x768xf32>) -> tensor<32x197x768xf32>
    %v4071 = stablehlo.broadcast_in_dim %b7_bfc2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v4072 = stablehlo.add %v4070, %v4071 : tensor<32x197x768xf32>
    %v4073 = stablehlo.reshape %v4072 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4074 = stablehlo.add %v4022, %v4073 : tensor<32x151296xf32>
    %v4075 = stablehlo.reshape %v4074 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4076 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4077 = stablehlo.constant dense<768.0> : tensor<32x197x768xf32>
    %v4078 = stablehlo.constant dense<1.0e-5> : tensor<32x197x768xf32>
    %v4079 = stablehlo.reduce(%v4075 init: %v4076) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4080 = stablehlo.broadcast_in_dim %v4079, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v4081 = stablehlo.divide %v4080, %v4077 : tensor<32x197x768xf32>
    %v4082 = stablehlo.subtract %v4075, %v4081 : tensor<32x197x768xf32>
    %v4083 = stablehlo.multiply %v4082, %v4082 : tensor<32x197x768xf32>
    %v4084 = stablehlo.reduce(%v4083 init: %v4076) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4085 = stablehlo.broadcast_in_dim %v4084, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v4086 = stablehlo.divide %v4085, %v4077 : tensor<32x197x768xf32>
    %v4087 = stablehlo.add %v4086, %v4078 : tensor<32x197x768xf32>
    %v4088 = stablehlo.rsqrt %v4087 : tensor<32x197x768xf32>
    %v4089 = stablehlo.multiply %v4082, %v4088 : tensor<32x197x768xf32>
    %v4090 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v4091 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v4092 = stablehlo.multiply %v4089, %v4090 : tensor<32x197x768xf32>
    %v4093 = stablehlo.add %v4092, %v4091 : tensor<32x197x768xf32>
    %v4094 = stablehlo.reshape %v4093 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4095 = stablehlo.reshape %v4094 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4096 = stablehlo.broadcast_in_dim %b8_g1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v4097 = stablehlo.multiply %v4095, %v4096 : tensor<32x197x768xf32>
    %v4098 = stablehlo.reshape %v4097 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4099 = stablehlo.reshape %v4098 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4100 = stablehlo.broadcast_in_dim %b8_bt1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v4101 = stablehlo.add %v4099, %v4100 : tensor<32x197x768xf32>
    %v4102 = stablehlo.reshape %v4101 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4103 = stablehlo.reshape %v4102 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4104 = stablehlo.dot_general %v4103, %b8_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v4105 = stablehlo.broadcast_in_dim %b8_bq, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v4106 = stablehlo.add %v4104, %v4105 : tensor<32x197x768xf32>
    %v4107 = stablehlo.reshape %v4106 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4108 = stablehlo.reshape %v4102 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4109 = stablehlo.dot_general %v4108, %b8_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v4110 = stablehlo.broadcast_in_dim %b8_bk, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v4111 = stablehlo.add %v4109, %v4110 : tensor<32x197x768xf32>
    %v4112 = stablehlo.reshape %v4111 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4113 = stablehlo.reshape %v4102 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4114 = stablehlo.dot_general %v4113, %b8_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v4115 = stablehlo.broadcast_in_dim %b8_bv, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v4116 = stablehlo.add %v4114, %v4115 : tensor<32x197x768xf32>
    %v4117 = stablehlo.reshape %v4116 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4118 = stablehlo.reshape %v4107 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4119 = stablehlo.slice %v4118 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4120 = stablehlo.reshape %v4119 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4121 = stablehlo.reshape %v4112 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4122 = stablehlo.slice %v4121 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4123 = stablehlo.reshape %v4122 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4124 = stablehlo.reshape %v4117 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4125 = stablehlo.slice %v4124 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4126 = stablehlo.reshape %v4125 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4127 = stablehlo.reshape %v4123 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4128 = stablehlo.transpose %v4127, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v4129 = stablehlo.reshape %v4128 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4130 = stablehlo.reshape %v4120 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4131 = stablehlo.reshape %v4129 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4132 = stablehlo.dot_general %v4130, %v4131, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v4133 = stablehlo.reshape %v4132 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4134 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v4135 = stablehlo.multiply %v4133, %v4134 : tensor<32x38809xf32>
    %v4136 = stablehlo.reshape %v4135 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4137 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4138 = stablehlo.exponential %v4136 : tensor<32x197x197xf32>
    %v4139 = stablehlo.reduce(%v4138 init: %v4137) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4140 = stablehlo.broadcast_in_dim %v4139, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4141 = stablehlo.divide %v4138, %v4140 : tensor<32x197x197xf32>
    %v4142 = stablehlo.reshape %v4141 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4143 = stablehlo.reshape %v4142 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4144 = stablehlo.reshape %v4126 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4145 = stablehlo.dot_general %v4143, %v4144, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4146 = stablehlo.reshape %v4145 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4147 = stablehlo.reshape %v4146 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4148 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4149 = stablehlo.pad %v4147, %v4148, low = [0, 0, 0], high = [0, 0, 704], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v4150 = stablehlo.reshape %v4149 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4151 = stablehlo.reshape %v4107 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4152 = stablehlo.slice %v4151 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4153 = stablehlo.reshape %v4152 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4154 = stablehlo.reshape %v4112 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4155 = stablehlo.slice %v4154 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4156 = stablehlo.reshape %v4155 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4157 = stablehlo.reshape %v4117 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4158 = stablehlo.slice %v4157 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4159 = stablehlo.reshape %v4158 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4160 = stablehlo.reshape %v4156 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4161 = stablehlo.transpose %v4160, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v4162 = stablehlo.reshape %v4161 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4163 = stablehlo.reshape %v4153 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4164 = stablehlo.reshape %v4162 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4165 = stablehlo.dot_general %v4163, %v4164, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v4166 = stablehlo.reshape %v4165 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4167 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v4168 = stablehlo.multiply %v4166, %v4167 : tensor<32x38809xf32>
    %v4169 = stablehlo.reshape %v4168 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4170 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4171 = stablehlo.exponential %v4169 : tensor<32x197x197xf32>
    %v4172 = stablehlo.reduce(%v4171 init: %v4170) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4173 = stablehlo.broadcast_in_dim %v4172, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4174 = stablehlo.divide %v4171, %v4173 : tensor<32x197x197xf32>
    %v4175 = stablehlo.reshape %v4174 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4176 = stablehlo.reshape %v4175 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4177 = stablehlo.reshape %v4159 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4178 = stablehlo.dot_general %v4176, %v4177, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4179 = stablehlo.reshape %v4178 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4180 = stablehlo.reshape %v4179 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4181 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4182 = stablehlo.pad %v4180, %v4181, low = [0, 0, 64], high = [0, 0, 640], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v4183 = stablehlo.reshape %v4182 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4184 = stablehlo.add %v4150, %v4183 : tensor<32x151296xf32>
    %v4185 = stablehlo.reshape %v4107 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4186 = stablehlo.slice %v4185 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4187 = stablehlo.reshape %v4186 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4188 = stablehlo.reshape %v4112 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4189 = stablehlo.slice %v4188 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4190 = stablehlo.reshape %v4189 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4191 = stablehlo.reshape %v4117 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4192 = stablehlo.slice %v4191 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4193 = stablehlo.reshape %v4192 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4194 = stablehlo.reshape %v4190 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4195 = stablehlo.transpose %v4194, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v4196 = stablehlo.reshape %v4195 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4197 = stablehlo.reshape %v4187 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4198 = stablehlo.reshape %v4196 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4199 = stablehlo.dot_general %v4197, %v4198, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v4200 = stablehlo.reshape %v4199 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4201 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v4202 = stablehlo.multiply %v4200, %v4201 : tensor<32x38809xf32>
    %v4203 = stablehlo.reshape %v4202 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4204 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4205 = stablehlo.exponential %v4203 : tensor<32x197x197xf32>
    %v4206 = stablehlo.reduce(%v4205 init: %v4204) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4207 = stablehlo.broadcast_in_dim %v4206, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4208 = stablehlo.divide %v4205, %v4207 : tensor<32x197x197xf32>
    %v4209 = stablehlo.reshape %v4208 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4210 = stablehlo.reshape %v4209 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4211 = stablehlo.reshape %v4193 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4212 = stablehlo.dot_general %v4210, %v4211, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4213 = stablehlo.reshape %v4212 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4214 = stablehlo.reshape %v4213 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4215 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4216 = stablehlo.pad %v4214, %v4215, low = [0, 0, 128], high = [0, 0, 576], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v4217 = stablehlo.reshape %v4216 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4218 = stablehlo.add %v4184, %v4217 : tensor<32x151296xf32>
    %v4219 = stablehlo.reshape %v4107 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4220 = stablehlo.slice %v4219 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4221 = stablehlo.reshape %v4220 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4222 = stablehlo.reshape %v4112 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4223 = stablehlo.slice %v4222 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4224 = stablehlo.reshape %v4223 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4225 = stablehlo.reshape %v4117 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4226 = stablehlo.slice %v4225 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4227 = stablehlo.reshape %v4226 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4228 = stablehlo.reshape %v4224 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4229 = stablehlo.transpose %v4228, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v4230 = stablehlo.reshape %v4229 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4231 = stablehlo.reshape %v4221 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4232 = stablehlo.reshape %v4230 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4233 = stablehlo.dot_general %v4231, %v4232, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v4234 = stablehlo.reshape %v4233 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4235 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v4236 = stablehlo.multiply %v4234, %v4235 : tensor<32x38809xf32>
    %v4237 = stablehlo.reshape %v4236 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4238 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4239 = stablehlo.exponential %v4237 : tensor<32x197x197xf32>
    %v4240 = stablehlo.reduce(%v4239 init: %v4238) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4241 = stablehlo.broadcast_in_dim %v4240, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4242 = stablehlo.divide %v4239, %v4241 : tensor<32x197x197xf32>
    %v4243 = stablehlo.reshape %v4242 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4244 = stablehlo.reshape %v4243 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4245 = stablehlo.reshape %v4227 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4246 = stablehlo.dot_general %v4244, %v4245, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4247 = stablehlo.reshape %v4246 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4248 = stablehlo.reshape %v4247 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4249 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4250 = stablehlo.pad %v4248, %v4249, low = [0, 0, 192], high = [0, 0, 512], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v4251 = stablehlo.reshape %v4250 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4252 = stablehlo.add %v4218, %v4251 : tensor<32x151296xf32>
    %v4253 = stablehlo.reshape %v4107 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4254 = stablehlo.slice %v4253 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4255 = stablehlo.reshape %v4254 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4256 = stablehlo.reshape %v4112 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4257 = stablehlo.slice %v4256 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4258 = stablehlo.reshape %v4257 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4259 = stablehlo.reshape %v4117 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4260 = stablehlo.slice %v4259 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4261 = stablehlo.reshape %v4260 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4262 = stablehlo.reshape %v4258 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4263 = stablehlo.transpose %v4262, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v4264 = stablehlo.reshape %v4263 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4265 = stablehlo.reshape %v4255 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4266 = stablehlo.reshape %v4264 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4267 = stablehlo.dot_general %v4265, %v4266, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v4268 = stablehlo.reshape %v4267 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4269 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v4270 = stablehlo.multiply %v4268, %v4269 : tensor<32x38809xf32>
    %v4271 = stablehlo.reshape %v4270 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4272 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4273 = stablehlo.exponential %v4271 : tensor<32x197x197xf32>
    %v4274 = stablehlo.reduce(%v4273 init: %v4272) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4275 = stablehlo.broadcast_in_dim %v4274, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4276 = stablehlo.divide %v4273, %v4275 : tensor<32x197x197xf32>
    %v4277 = stablehlo.reshape %v4276 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4278 = stablehlo.reshape %v4277 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4279 = stablehlo.reshape %v4261 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4280 = stablehlo.dot_general %v4278, %v4279, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4281 = stablehlo.reshape %v4280 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4282 = stablehlo.reshape %v4281 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4283 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4284 = stablehlo.pad %v4282, %v4283, low = [0, 0, 256], high = [0, 0, 448], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v4285 = stablehlo.reshape %v4284 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4286 = stablehlo.add %v4252, %v4285 : tensor<32x151296xf32>
    %v4287 = stablehlo.reshape %v4107 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4288 = stablehlo.slice %v4287 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4289 = stablehlo.reshape %v4288 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4290 = stablehlo.reshape %v4112 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4291 = stablehlo.slice %v4290 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4292 = stablehlo.reshape %v4291 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4293 = stablehlo.reshape %v4117 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4294 = stablehlo.slice %v4293 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4295 = stablehlo.reshape %v4294 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4296 = stablehlo.reshape %v4292 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4297 = stablehlo.transpose %v4296, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v4298 = stablehlo.reshape %v4297 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4299 = stablehlo.reshape %v4289 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4300 = stablehlo.reshape %v4298 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4301 = stablehlo.dot_general %v4299, %v4300, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v4302 = stablehlo.reshape %v4301 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4303 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v4304 = stablehlo.multiply %v4302, %v4303 : tensor<32x38809xf32>
    %v4305 = stablehlo.reshape %v4304 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4306 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4307 = stablehlo.exponential %v4305 : tensor<32x197x197xf32>
    %v4308 = stablehlo.reduce(%v4307 init: %v4306) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4309 = stablehlo.broadcast_in_dim %v4308, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4310 = stablehlo.divide %v4307, %v4309 : tensor<32x197x197xf32>
    %v4311 = stablehlo.reshape %v4310 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4312 = stablehlo.reshape %v4311 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4313 = stablehlo.reshape %v4295 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4314 = stablehlo.dot_general %v4312, %v4313, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4315 = stablehlo.reshape %v4314 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4316 = stablehlo.reshape %v4315 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4317 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4318 = stablehlo.pad %v4316, %v4317, low = [0, 0, 320], high = [0, 0, 384], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v4319 = stablehlo.reshape %v4318 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4320 = stablehlo.add %v4286, %v4319 : tensor<32x151296xf32>
    %v4321 = stablehlo.reshape %v4107 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4322 = stablehlo.slice %v4321 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4323 = stablehlo.reshape %v4322 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4324 = stablehlo.reshape %v4112 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4325 = stablehlo.slice %v4324 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4326 = stablehlo.reshape %v4325 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4327 = stablehlo.reshape %v4117 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4328 = stablehlo.slice %v4327 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
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
    %v4340 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4341 = stablehlo.exponential %v4339 : tensor<32x197x197xf32>
    %v4342 = stablehlo.reduce(%v4341 init: %v4340) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4343 = stablehlo.broadcast_in_dim %v4342, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4344 = stablehlo.divide %v4341, %v4343 : tensor<32x197x197xf32>
    %v4345 = stablehlo.reshape %v4344 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4346 = stablehlo.reshape %v4345 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4347 = stablehlo.reshape %v4329 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4348 = stablehlo.dot_general %v4346, %v4347, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4349 = stablehlo.reshape %v4348 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4350 = stablehlo.reshape %v4349 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4351 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4352 = stablehlo.pad %v4350, %v4351, low = [0, 0, 384], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v4353 = stablehlo.reshape %v4352 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4354 = stablehlo.add %v4320, %v4353 : tensor<32x151296xf32>
    %v4355 = stablehlo.reshape %v4107 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4356 = stablehlo.slice %v4355 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4357 = stablehlo.reshape %v4356 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4358 = stablehlo.reshape %v4112 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4359 = stablehlo.slice %v4358 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4360 = stablehlo.reshape %v4359 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4361 = stablehlo.reshape %v4117 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4362 = stablehlo.slice %v4361 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4363 = stablehlo.reshape %v4362 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4364 = stablehlo.reshape %v4360 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4365 = stablehlo.transpose %v4364, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v4366 = stablehlo.reshape %v4365 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4367 = stablehlo.reshape %v4357 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4368 = stablehlo.reshape %v4366 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4369 = stablehlo.dot_general %v4367, %v4368, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v4370 = stablehlo.reshape %v4369 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4371 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v4372 = stablehlo.multiply %v4370, %v4371 : tensor<32x38809xf32>
    %v4373 = stablehlo.reshape %v4372 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4374 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4375 = stablehlo.exponential %v4373 : tensor<32x197x197xf32>
    %v4376 = stablehlo.reduce(%v4375 init: %v4374) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4377 = stablehlo.broadcast_in_dim %v4376, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4378 = stablehlo.divide %v4375, %v4377 : tensor<32x197x197xf32>
    %v4379 = stablehlo.reshape %v4378 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4380 = stablehlo.reshape %v4379 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4381 = stablehlo.reshape %v4363 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4382 = stablehlo.dot_general %v4380, %v4381, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4383 = stablehlo.reshape %v4382 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4384 = stablehlo.reshape %v4383 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4385 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4386 = stablehlo.pad %v4384, %v4385, low = [0, 0, 448], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v4387 = stablehlo.reshape %v4386 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4388 = stablehlo.add %v4354, %v4387 : tensor<32x151296xf32>
    %v4389 = stablehlo.reshape %v4107 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4390 = stablehlo.slice %v4389 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4391 = stablehlo.reshape %v4390 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4392 = stablehlo.reshape %v4112 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4393 = stablehlo.slice %v4392 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4394 = stablehlo.reshape %v4393 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4395 = stablehlo.reshape %v4117 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4396 = stablehlo.slice %v4395 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4397 = stablehlo.reshape %v4396 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4398 = stablehlo.reshape %v4394 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4399 = stablehlo.transpose %v4398, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v4400 = stablehlo.reshape %v4399 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4401 = stablehlo.reshape %v4391 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4402 = stablehlo.reshape %v4400 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4403 = stablehlo.dot_general %v4401, %v4402, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v4404 = stablehlo.reshape %v4403 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4405 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v4406 = stablehlo.multiply %v4404, %v4405 : tensor<32x38809xf32>
    %v4407 = stablehlo.reshape %v4406 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4408 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4409 = stablehlo.exponential %v4407 : tensor<32x197x197xf32>
    %v4410 = stablehlo.reduce(%v4409 init: %v4408) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4411 = stablehlo.broadcast_in_dim %v4410, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4412 = stablehlo.divide %v4409, %v4411 : tensor<32x197x197xf32>
    %v4413 = stablehlo.reshape %v4412 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4414 = stablehlo.reshape %v4413 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4415 = stablehlo.reshape %v4397 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4416 = stablehlo.dot_general %v4414, %v4415, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4417 = stablehlo.reshape %v4416 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4418 = stablehlo.reshape %v4417 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4419 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4420 = stablehlo.pad %v4418, %v4419, low = [0, 0, 512], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v4421 = stablehlo.reshape %v4420 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4422 = stablehlo.add %v4388, %v4421 : tensor<32x151296xf32>
    %v4423 = stablehlo.reshape %v4107 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4424 = stablehlo.slice %v4423 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4425 = stablehlo.reshape %v4424 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4426 = stablehlo.reshape %v4112 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4427 = stablehlo.slice %v4426 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4428 = stablehlo.reshape %v4427 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4429 = stablehlo.reshape %v4117 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4430 = stablehlo.slice %v4429 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4431 = stablehlo.reshape %v4430 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4432 = stablehlo.reshape %v4428 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4433 = stablehlo.transpose %v4432, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v4434 = stablehlo.reshape %v4433 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4435 = stablehlo.reshape %v4425 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4436 = stablehlo.reshape %v4434 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4437 = stablehlo.dot_general %v4435, %v4436, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v4438 = stablehlo.reshape %v4437 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4439 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v4440 = stablehlo.multiply %v4438, %v4439 : tensor<32x38809xf32>
    %v4441 = stablehlo.reshape %v4440 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4442 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4443 = stablehlo.exponential %v4441 : tensor<32x197x197xf32>
    %v4444 = stablehlo.reduce(%v4443 init: %v4442) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4445 = stablehlo.broadcast_in_dim %v4444, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4446 = stablehlo.divide %v4443, %v4445 : tensor<32x197x197xf32>
    %v4447 = stablehlo.reshape %v4446 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4448 = stablehlo.reshape %v4447 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4449 = stablehlo.reshape %v4431 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4450 = stablehlo.dot_general %v4448, %v4449, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4451 = stablehlo.reshape %v4450 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4452 = stablehlo.reshape %v4451 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4453 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4454 = stablehlo.pad %v4452, %v4453, low = [0, 0, 576], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v4455 = stablehlo.reshape %v4454 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4456 = stablehlo.add %v4422, %v4455 : tensor<32x151296xf32>
    %v4457 = stablehlo.reshape %v4107 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4458 = stablehlo.slice %v4457 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4459 = stablehlo.reshape %v4458 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4460 = stablehlo.reshape %v4112 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4461 = stablehlo.slice %v4460 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4462 = stablehlo.reshape %v4461 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4463 = stablehlo.reshape %v4117 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4464 = stablehlo.slice %v4463 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4465 = stablehlo.reshape %v4464 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4466 = stablehlo.reshape %v4462 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4467 = stablehlo.transpose %v4466, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v4468 = stablehlo.reshape %v4467 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4469 = stablehlo.reshape %v4459 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4470 = stablehlo.reshape %v4468 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4471 = stablehlo.dot_general %v4469, %v4470, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v4472 = stablehlo.reshape %v4471 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4473 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v4474 = stablehlo.multiply %v4472, %v4473 : tensor<32x38809xf32>
    %v4475 = stablehlo.reshape %v4474 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4476 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4477 = stablehlo.exponential %v4475 : tensor<32x197x197xf32>
    %v4478 = stablehlo.reduce(%v4477 init: %v4476) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4479 = stablehlo.broadcast_in_dim %v4478, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4480 = stablehlo.divide %v4477, %v4479 : tensor<32x197x197xf32>
    %v4481 = stablehlo.reshape %v4480 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4482 = stablehlo.reshape %v4481 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4483 = stablehlo.reshape %v4465 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4484 = stablehlo.dot_general %v4482, %v4483, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4485 = stablehlo.reshape %v4484 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4486 = stablehlo.reshape %v4485 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4487 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4488 = stablehlo.pad %v4486, %v4487, low = [0, 0, 640], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v4489 = stablehlo.reshape %v4488 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4490 = stablehlo.add %v4456, %v4489 : tensor<32x151296xf32>
    %v4491 = stablehlo.reshape %v4107 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4492 = stablehlo.slice %v4491 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4493 = stablehlo.reshape %v4492 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4494 = stablehlo.reshape %v4112 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4495 = stablehlo.slice %v4494 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4496 = stablehlo.reshape %v4495 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4497 = stablehlo.reshape %v4117 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4498 = stablehlo.slice %v4497 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
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
    %v4510 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4511 = stablehlo.exponential %v4509 : tensor<32x197x197xf32>
    %v4512 = stablehlo.reduce(%v4511 init: %v4510) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4513 = stablehlo.broadcast_in_dim %v4512, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4514 = stablehlo.divide %v4511, %v4513 : tensor<32x197x197xf32>
    %v4515 = stablehlo.reshape %v4514 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4516 = stablehlo.reshape %v4515 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4517 = stablehlo.reshape %v4499 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4518 = stablehlo.dot_general %v4516, %v4517, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4519 = stablehlo.reshape %v4518 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4520 = stablehlo.reshape %v4519 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4521 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4522 = stablehlo.pad %v4520, %v4521, low = [0, 0, 704], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v4523 = stablehlo.reshape %v4522 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4524 = stablehlo.add %v4490, %v4523 : tensor<32x151296xf32>
    %v4525 = stablehlo.reshape %v4524 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4526 = stablehlo.dot_general %v4525, %b8_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v4527 = stablehlo.broadcast_in_dim %b8_bo, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v4528 = stablehlo.add %v4526, %v4527 : tensor<32x197x768xf32>
    %v4529 = stablehlo.reshape %v4528 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4530 = stablehlo.add %v4074, %v4529 : tensor<32x151296xf32>
    %v4531 = stablehlo.reshape %v4530 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4532 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4533 = stablehlo.constant dense<768.0> : tensor<32x197x768xf32>
    %v4534 = stablehlo.constant dense<1.0e-5> : tensor<32x197x768xf32>
    %v4535 = stablehlo.reduce(%v4531 init: %v4532) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4536 = stablehlo.broadcast_in_dim %v4535, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v4537 = stablehlo.divide %v4536, %v4533 : tensor<32x197x768xf32>
    %v4538 = stablehlo.subtract %v4531, %v4537 : tensor<32x197x768xf32>
    %v4539 = stablehlo.multiply %v4538, %v4538 : tensor<32x197x768xf32>
    %v4540 = stablehlo.reduce(%v4539 init: %v4532) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4541 = stablehlo.broadcast_in_dim %v4540, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v4542 = stablehlo.divide %v4541, %v4533 : tensor<32x197x768xf32>
    %v4543 = stablehlo.add %v4542, %v4534 : tensor<32x197x768xf32>
    %v4544 = stablehlo.rsqrt %v4543 : tensor<32x197x768xf32>
    %v4545 = stablehlo.multiply %v4538, %v4544 : tensor<32x197x768xf32>
    %v4546 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v4547 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v4548 = stablehlo.multiply %v4545, %v4546 : tensor<32x197x768xf32>
    %v4549 = stablehlo.add %v4548, %v4547 : tensor<32x197x768xf32>
    %v4550 = stablehlo.reshape %v4549 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4551 = stablehlo.reshape %v4550 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4552 = stablehlo.broadcast_in_dim %b8_g2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v4553 = stablehlo.multiply %v4551, %v4552 : tensor<32x197x768xf32>
    %v4554 = stablehlo.reshape %v4553 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4555 = stablehlo.reshape %v4554 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4556 = stablehlo.broadcast_in_dim %b8_bt2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v4557 = stablehlo.add %v4555, %v4556 : tensor<32x197x768xf32>
    %v4558 = stablehlo.reshape %v4557 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4559 = stablehlo.reshape %v4558 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4560 = stablehlo.dot_general %v4559, %b8_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x3072xf32>) -> tensor<32x197x3072xf32>
    %v4561 = stablehlo.broadcast_in_dim %b8_bfc1, dims = [2] : (tensor<3072xf32>) -> tensor<32x197x3072xf32>
    %v4562 = stablehlo.add %v4560, %v4561 : tensor<32x197x3072xf32>
    %v4563 = stablehlo.reshape %v4562 : (tensor<32x197x3072xf32>) -> tensor<32x605184xf32>
    %v4564 = stablehlo.multiply %v4563, %v4563 : tensor<32x605184xf32>
    %v4565 = stablehlo.multiply %v4564, %v4563 : tensor<32x605184xf32>
    %v4566 = stablehlo.constant dense<0.044715> : tensor<32x605184xf32>
    %v4567 = stablehlo.multiply %v4566, %v4565 : tensor<32x605184xf32>
    %v4568 = stablehlo.add %v4563, %v4567 : tensor<32x605184xf32>
    %v4569 = stablehlo.constant dense<0.7978845608028654> : tensor<32x605184xf32>
    %v4570 = stablehlo.multiply %v4569, %v4568 : tensor<32x605184xf32>
    %v4571 = stablehlo.tanh %v4570 : tensor<32x605184xf32>
    %v4572 = stablehlo.constant dense<1.0> : tensor<32x605184xf32>
    %v4573 = stablehlo.add %v4572, %v4571 : tensor<32x605184xf32>
    %v4574 = stablehlo.constant dense<0.5> : tensor<32x605184xf32>
    %v4575 = stablehlo.multiply %v4574, %v4563 : tensor<32x605184xf32>
    %v4576 = stablehlo.multiply %v4575, %v4573 : tensor<32x605184xf32>
    %v4577 = stablehlo.reshape %v4576 : (tensor<32x605184xf32>) -> tensor<32x197x3072xf32>
    %v4578 = stablehlo.dot_general %v4577, %b8_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x3072xf32>, tensor<3072x768xf32>) -> tensor<32x197x768xf32>
    %v4579 = stablehlo.broadcast_in_dim %b8_bfc2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v4580 = stablehlo.add %v4578, %v4579 : tensor<32x197x768xf32>
    %v4581 = stablehlo.reshape %v4580 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4582 = stablehlo.add %v4530, %v4581 : tensor<32x151296xf32>
    %v4583 = stablehlo.reshape %v4582 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4584 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4585 = stablehlo.constant dense<768.0> : tensor<32x197x768xf32>
    %v4586 = stablehlo.constant dense<1.0e-5> : tensor<32x197x768xf32>
    %v4587 = stablehlo.reduce(%v4583 init: %v4584) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4588 = stablehlo.broadcast_in_dim %v4587, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v4589 = stablehlo.divide %v4588, %v4585 : tensor<32x197x768xf32>
    %v4590 = stablehlo.subtract %v4583, %v4589 : tensor<32x197x768xf32>
    %v4591 = stablehlo.multiply %v4590, %v4590 : tensor<32x197x768xf32>
    %v4592 = stablehlo.reduce(%v4591 init: %v4584) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4593 = stablehlo.broadcast_in_dim %v4592, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v4594 = stablehlo.divide %v4593, %v4585 : tensor<32x197x768xf32>
    %v4595 = stablehlo.add %v4594, %v4586 : tensor<32x197x768xf32>
    %v4596 = stablehlo.rsqrt %v4595 : tensor<32x197x768xf32>
    %v4597 = stablehlo.multiply %v4590, %v4596 : tensor<32x197x768xf32>
    %v4598 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v4599 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v4600 = stablehlo.multiply %v4597, %v4598 : tensor<32x197x768xf32>
    %v4601 = stablehlo.add %v4600, %v4599 : tensor<32x197x768xf32>
    %v4602 = stablehlo.reshape %v4601 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4603 = stablehlo.reshape %v4602 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4604 = stablehlo.broadcast_in_dim %b9_g1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v4605 = stablehlo.multiply %v4603, %v4604 : tensor<32x197x768xf32>
    %v4606 = stablehlo.reshape %v4605 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4607 = stablehlo.reshape %v4606 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4608 = stablehlo.broadcast_in_dim %b9_bt1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v4609 = stablehlo.add %v4607, %v4608 : tensor<32x197x768xf32>
    %v4610 = stablehlo.reshape %v4609 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4611 = stablehlo.reshape %v4610 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4612 = stablehlo.dot_general %v4611, %b9_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v4613 = stablehlo.broadcast_in_dim %b9_bq, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v4614 = stablehlo.add %v4612, %v4613 : tensor<32x197x768xf32>
    %v4615 = stablehlo.reshape %v4614 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4616 = stablehlo.reshape %v4610 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4617 = stablehlo.dot_general %v4616, %b9_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v4618 = stablehlo.broadcast_in_dim %b9_bk, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v4619 = stablehlo.add %v4617, %v4618 : tensor<32x197x768xf32>
    %v4620 = stablehlo.reshape %v4619 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4621 = stablehlo.reshape %v4610 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4622 = stablehlo.dot_general %v4621, %b9_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v4623 = stablehlo.broadcast_in_dim %b9_bv, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v4624 = stablehlo.add %v4622, %v4623 : tensor<32x197x768xf32>
    %v4625 = stablehlo.reshape %v4624 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4626 = stablehlo.reshape %v4615 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4627 = stablehlo.slice %v4626 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4628 = stablehlo.reshape %v4627 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4629 = stablehlo.reshape %v4620 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4630 = stablehlo.slice %v4629 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4631 = stablehlo.reshape %v4630 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4632 = stablehlo.reshape %v4625 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4633 = stablehlo.slice %v4632 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4634 = stablehlo.reshape %v4633 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4635 = stablehlo.reshape %v4631 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4636 = stablehlo.transpose %v4635, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v4637 = stablehlo.reshape %v4636 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4638 = stablehlo.reshape %v4628 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4639 = stablehlo.reshape %v4637 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4640 = stablehlo.dot_general %v4638, %v4639, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v4641 = stablehlo.reshape %v4640 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4642 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v4643 = stablehlo.multiply %v4641, %v4642 : tensor<32x38809xf32>
    %v4644 = stablehlo.reshape %v4643 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4645 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4646 = stablehlo.exponential %v4644 : tensor<32x197x197xf32>
    %v4647 = stablehlo.reduce(%v4646 init: %v4645) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4648 = stablehlo.broadcast_in_dim %v4647, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4649 = stablehlo.divide %v4646, %v4648 : tensor<32x197x197xf32>
    %v4650 = stablehlo.reshape %v4649 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4651 = stablehlo.reshape %v4650 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4652 = stablehlo.reshape %v4634 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4653 = stablehlo.dot_general %v4651, %v4652, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4654 = stablehlo.reshape %v4653 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4655 = stablehlo.reshape %v4654 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4656 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4657 = stablehlo.pad %v4655, %v4656, low = [0, 0, 0], high = [0, 0, 704], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v4658 = stablehlo.reshape %v4657 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4659 = stablehlo.reshape %v4615 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4660 = stablehlo.slice %v4659 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4661 = stablehlo.reshape %v4660 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4662 = stablehlo.reshape %v4620 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4663 = stablehlo.slice %v4662 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4664 = stablehlo.reshape %v4663 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4665 = stablehlo.reshape %v4625 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4666 = stablehlo.slice %v4665 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4667 = stablehlo.reshape %v4666 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4668 = stablehlo.reshape %v4664 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4669 = stablehlo.transpose %v4668, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v4670 = stablehlo.reshape %v4669 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4671 = stablehlo.reshape %v4661 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4672 = stablehlo.reshape %v4670 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4673 = stablehlo.dot_general %v4671, %v4672, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v4674 = stablehlo.reshape %v4673 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4675 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v4676 = stablehlo.multiply %v4674, %v4675 : tensor<32x38809xf32>
    %v4677 = stablehlo.reshape %v4676 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4678 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4679 = stablehlo.exponential %v4677 : tensor<32x197x197xf32>
    %v4680 = stablehlo.reduce(%v4679 init: %v4678) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4681 = stablehlo.broadcast_in_dim %v4680, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4682 = stablehlo.divide %v4679, %v4681 : tensor<32x197x197xf32>
    %v4683 = stablehlo.reshape %v4682 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4684 = stablehlo.reshape %v4683 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4685 = stablehlo.reshape %v4667 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4686 = stablehlo.dot_general %v4684, %v4685, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4687 = stablehlo.reshape %v4686 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4688 = stablehlo.reshape %v4687 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4689 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4690 = stablehlo.pad %v4688, %v4689, low = [0, 0, 64], high = [0, 0, 640], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v4691 = stablehlo.reshape %v4690 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4692 = stablehlo.add %v4658, %v4691 : tensor<32x151296xf32>
    %v4693 = stablehlo.reshape %v4615 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4694 = stablehlo.slice %v4693 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4695 = stablehlo.reshape %v4694 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4696 = stablehlo.reshape %v4620 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4697 = stablehlo.slice %v4696 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4698 = stablehlo.reshape %v4697 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4699 = stablehlo.reshape %v4625 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4700 = stablehlo.slice %v4699 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4701 = stablehlo.reshape %v4700 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4702 = stablehlo.reshape %v4698 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4703 = stablehlo.transpose %v4702, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v4704 = stablehlo.reshape %v4703 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4705 = stablehlo.reshape %v4695 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4706 = stablehlo.reshape %v4704 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4707 = stablehlo.dot_general %v4705, %v4706, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v4708 = stablehlo.reshape %v4707 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4709 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v4710 = stablehlo.multiply %v4708, %v4709 : tensor<32x38809xf32>
    %v4711 = stablehlo.reshape %v4710 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4712 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4713 = stablehlo.exponential %v4711 : tensor<32x197x197xf32>
    %v4714 = stablehlo.reduce(%v4713 init: %v4712) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4715 = stablehlo.broadcast_in_dim %v4714, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4716 = stablehlo.divide %v4713, %v4715 : tensor<32x197x197xf32>
    %v4717 = stablehlo.reshape %v4716 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4718 = stablehlo.reshape %v4717 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4719 = stablehlo.reshape %v4701 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4720 = stablehlo.dot_general %v4718, %v4719, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4721 = stablehlo.reshape %v4720 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4722 = stablehlo.reshape %v4721 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4723 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4724 = stablehlo.pad %v4722, %v4723, low = [0, 0, 128], high = [0, 0, 576], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v4725 = stablehlo.reshape %v4724 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4726 = stablehlo.add %v4692, %v4725 : tensor<32x151296xf32>
    %v4727 = stablehlo.reshape %v4615 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4728 = stablehlo.slice %v4727 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4729 = stablehlo.reshape %v4728 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4730 = stablehlo.reshape %v4620 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4731 = stablehlo.slice %v4730 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4732 = stablehlo.reshape %v4731 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4733 = stablehlo.reshape %v4625 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4734 = stablehlo.slice %v4733 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4735 = stablehlo.reshape %v4734 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4736 = stablehlo.reshape %v4732 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4737 = stablehlo.transpose %v4736, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v4738 = stablehlo.reshape %v4737 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4739 = stablehlo.reshape %v4729 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4740 = stablehlo.reshape %v4738 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4741 = stablehlo.dot_general %v4739, %v4740, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v4742 = stablehlo.reshape %v4741 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4743 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v4744 = stablehlo.multiply %v4742, %v4743 : tensor<32x38809xf32>
    %v4745 = stablehlo.reshape %v4744 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4746 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4747 = stablehlo.exponential %v4745 : tensor<32x197x197xf32>
    %v4748 = stablehlo.reduce(%v4747 init: %v4746) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4749 = stablehlo.broadcast_in_dim %v4748, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4750 = stablehlo.divide %v4747, %v4749 : tensor<32x197x197xf32>
    %v4751 = stablehlo.reshape %v4750 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4752 = stablehlo.reshape %v4751 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4753 = stablehlo.reshape %v4735 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4754 = stablehlo.dot_general %v4752, %v4753, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4755 = stablehlo.reshape %v4754 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4756 = stablehlo.reshape %v4755 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4757 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4758 = stablehlo.pad %v4756, %v4757, low = [0, 0, 192], high = [0, 0, 512], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v4759 = stablehlo.reshape %v4758 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4760 = stablehlo.add %v4726, %v4759 : tensor<32x151296xf32>
    %v4761 = stablehlo.reshape %v4615 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4762 = stablehlo.slice %v4761 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4763 = stablehlo.reshape %v4762 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4764 = stablehlo.reshape %v4620 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4765 = stablehlo.slice %v4764 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4766 = stablehlo.reshape %v4765 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4767 = stablehlo.reshape %v4625 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4768 = stablehlo.slice %v4767 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4769 = stablehlo.reshape %v4768 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4770 = stablehlo.reshape %v4766 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4771 = stablehlo.transpose %v4770, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v4772 = stablehlo.reshape %v4771 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4773 = stablehlo.reshape %v4763 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4774 = stablehlo.reshape %v4772 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4775 = stablehlo.dot_general %v4773, %v4774, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v4776 = stablehlo.reshape %v4775 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4777 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v4778 = stablehlo.multiply %v4776, %v4777 : tensor<32x38809xf32>
    %v4779 = stablehlo.reshape %v4778 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4780 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4781 = stablehlo.exponential %v4779 : tensor<32x197x197xf32>
    %v4782 = stablehlo.reduce(%v4781 init: %v4780) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4783 = stablehlo.broadcast_in_dim %v4782, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4784 = stablehlo.divide %v4781, %v4783 : tensor<32x197x197xf32>
    %v4785 = stablehlo.reshape %v4784 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4786 = stablehlo.reshape %v4785 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4787 = stablehlo.reshape %v4769 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4788 = stablehlo.dot_general %v4786, %v4787, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4789 = stablehlo.reshape %v4788 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4790 = stablehlo.reshape %v4789 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4791 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4792 = stablehlo.pad %v4790, %v4791, low = [0, 0, 256], high = [0, 0, 448], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v4793 = stablehlo.reshape %v4792 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4794 = stablehlo.add %v4760, %v4793 : tensor<32x151296xf32>
    %v4795 = stablehlo.reshape %v4615 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4796 = stablehlo.slice %v4795 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4797 = stablehlo.reshape %v4796 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4798 = stablehlo.reshape %v4620 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4799 = stablehlo.slice %v4798 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4800 = stablehlo.reshape %v4799 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4801 = stablehlo.reshape %v4625 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4802 = stablehlo.slice %v4801 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
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
    %v4814 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4815 = stablehlo.exponential %v4813 : tensor<32x197x197xf32>
    %v4816 = stablehlo.reduce(%v4815 init: %v4814) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4817 = stablehlo.broadcast_in_dim %v4816, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4818 = stablehlo.divide %v4815, %v4817 : tensor<32x197x197xf32>
    %v4819 = stablehlo.reshape %v4818 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4820 = stablehlo.reshape %v4819 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4821 = stablehlo.reshape %v4803 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4822 = stablehlo.dot_general %v4820, %v4821, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4823 = stablehlo.reshape %v4822 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4824 = stablehlo.reshape %v4823 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4825 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4826 = stablehlo.pad %v4824, %v4825, low = [0, 0, 320], high = [0, 0, 384], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v4827 = stablehlo.reshape %v4826 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4828 = stablehlo.add %v4794, %v4827 : tensor<32x151296xf32>
    %v4829 = stablehlo.reshape %v4615 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4830 = stablehlo.slice %v4829 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4831 = stablehlo.reshape %v4830 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4832 = stablehlo.reshape %v4620 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4833 = stablehlo.slice %v4832 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4834 = stablehlo.reshape %v4833 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4835 = stablehlo.reshape %v4625 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4836 = stablehlo.slice %v4835 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4837 = stablehlo.reshape %v4836 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4838 = stablehlo.reshape %v4834 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4839 = stablehlo.transpose %v4838, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v4840 = stablehlo.reshape %v4839 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4841 = stablehlo.reshape %v4831 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4842 = stablehlo.reshape %v4840 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4843 = stablehlo.dot_general %v4841, %v4842, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v4844 = stablehlo.reshape %v4843 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4845 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v4846 = stablehlo.multiply %v4844, %v4845 : tensor<32x38809xf32>
    %v4847 = stablehlo.reshape %v4846 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4848 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4849 = stablehlo.exponential %v4847 : tensor<32x197x197xf32>
    %v4850 = stablehlo.reduce(%v4849 init: %v4848) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4851 = stablehlo.broadcast_in_dim %v4850, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4852 = stablehlo.divide %v4849, %v4851 : tensor<32x197x197xf32>
    %v4853 = stablehlo.reshape %v4852 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4854 = stablehlo.reshape %v4853 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4855 = stablehlo.reshape %v4837 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4856 = stablehlo.dot_general %v4854, %v4855, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4857 = stablehlo.reshape %v4856 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4858 = stablehlo.reshape %v4857 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4859 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4860 = stablehlo.pad %v4858, %v4859, low = [0, 0, 384], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v4861 = stablehlo.reshape %v4860 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4862 = stablehlo.add %v4828, %v4861 : tensor<32x151296xf32>
    %v4863 = stablehlo.reshape %v4615 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4864 = stablehlo.slice %v4863 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4865 = stablehlo.reshape %v4864 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4866 = stablehlo.reshape %v4620 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4867 = stablehlo.slice %v4866 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4868 = stablehlo.reshape %v4867 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4869 = stablehlo.reshape %v4625 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4870 = stablehlo.slice %v4869 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4871 = stablehlo.reshape %v4870 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4872 = stablehlo.reshape %v4868 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4873 = stablehlo.transpose %v4872, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v4874 = stablehlo.reshape %v4873 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4875 = stablehlo.reshape %v4865 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4876 = stablehlo.reshape %v4874 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4877 = stablehlo.dot_general %v4875, %v4876, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v4878 = stablehlo.reshape %v4877 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4879 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v4880 = stablehlo.multiply %v4878, %v4879 : tensor<32x38809xf32>
    %v4881 = stablehlo.reshape %v4880 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4882 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4883 = stablehlo.exponential %v4881 : tensor<32x197x197xf32>
    %v4884 = stablehlo.reduce(%v4883 init: %v4882) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4885 = stablehlo.broadcast_in_dim %v4884, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4886 = stablehlo.divide %v4883, %v4885 : tensor<32x197x197xf32>
    %v4887 = stablehlo.reshape %v4886 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4888 = stablehlo.reshape %v4887 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4889 = stablehlo.reshape %v4871 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4890 = stablehlo.dot_general %v4888, %v4889, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4891 = stablehlo.reshape %v4890 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4892 = stablehlo.reshape %v4891 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4893 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4894 = stablehlo.pad %v4892, %v4893, low = [0, 0, 448], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v4895 = stablehlo.reshape %v4894 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4896 = stablehlo.add %v4862, %v4895 : tensor<32x151296xf32>
    %v4897 = stablehlo.reshape %v4615 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4898 = stablehlo.slice %v4897 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4899 = stablehlo.reshape %v4898 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4900 = stablehlo.reshape %v4620 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4901 = stablehlo.slice %v4900 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4902 = stablehlo.reshape %v4901 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4903 = stablehlo.reshape %v4625 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4904 = stablehlo.slice %v4903 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4905 = stablehlo.reshape %v4904 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4906 = stablehlo.reshape %v4902 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4907 = stablehlo.transpose %v4906, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v4908 = stablehlo.reshape %v4907 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4909 = stablehlo.reshape %v4899 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4910 = stablehlo.reshape %v4908 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4911 = stablehlo.dot_general %v4909, %v4910, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v4912 = stablehlo.reshape %v4911 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4913 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v4914 = stablehlo.multiply %v4912, %v4913 : tensor<32x38809xf32>
    %v4915 = stablehlo.reshape %v4914 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4916 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4917 = stablehlo.exponential %v4915 : tensor<32x197x197xf32>
    %v4918 = stablehlo.reduce(%v4917 init: %v4916) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4919 = stablehlo.broadcast_in_dim %v4918, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4920 = stablehlo.divide %v4917, %v4919 : tensor<32x197x197xf32>
    %v4921 = stablehlo.reshape %v4920 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4922 = stablehlo.reshape %v4921 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4923 = stablehlo.reshape %v4905 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4924 = stablehlo.dot_general %v4922, %v4923, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4925 = stablehlo.reshape %v4924 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4926 = stablehlo.reshape %v4925 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4927 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4928 = stablehlo.pad %v4926, %v4927, low = [0, 0, 512], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v4929 = stablehlo.reshape %v4928 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4930 = stablehlo.add %v4896, %v4929 : tensor<32x151296xf32>
    %v4931 = stablehlo.reshape %v4615 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4932 = stablehlo.slice %v4931 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4933 = stablehlo.reshape %v4932 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4934 = stablehlo.reshape %v4620 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4935 = stablehlo.slice %v4934 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4936 = stablehlo.reshape %v4935 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4937 = stablehlo.reshape %v4625 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4938 = stablehlo.slice %v4937 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4939 = stablehlo.reshape %v4938 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4940 = stablehlo.reshape %v4936 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4941 = stablehlo.transpose %v4940, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v4942 = stablehlo.reshape %v4941 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4943 = stablehlo.reshape %v4933 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4944 = stablehlo.reshape %v4942 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4945 = stablehlo.dot_general %v4943, %v4944, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v4946 = stablehlo.reshape %v4945 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4947 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v4948 = stablehlo.multiply %v4946, %v4947 : tensor<32x38809xf32>
    %v4949 = stablehlo.reshape %v4948 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4950 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4951 = stablehlo.exponential %v4949 : tensor<32x197x197xf32>
    %v4952 = stablehlo.reduce(%v4951 init: %v4950) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4953 = stablehlo.broadcast_in_dim %v4952, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4954 = stablehlo.divide %v4951, %v4953 : tensor<32x197x197xf32>
    %v4955 = stablehlo.reshape %v4954 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4956 = stablehlo.reshape %v4955 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4957 = stablehlo.reshape %v4939 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4958 = stablehlo.dot_general %v4956, %v4957, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4959 = stablehlo.reshape %v4958 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4960 = stablehlo.reshape %v4959 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4961 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4962 = stablehlo.pad %v4960, %v4961, low = [0, 0, 576], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v4963 = stablehlo.reshape %v4962 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4964 = stablehlo.add %v4930, %v4963 : tensor<32x151296xf32>
    %v4965 = stablehlo.reshape %v4615 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4966 = stablehlo.slice %v4965 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4967 = stablehlo.reshape %v4966 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4968 = stablehlo.reshape %v4620 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4969 = stablehlo.slice %v4968 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4970 = stablehlo.reshape %v4969 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4971 = stablehlo.reshape %v4625 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v4972 = stablehlo.slice %v4971 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v4973 = stablehlo.reshape %v4972 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4974 = stablehlo.reshape %v4970 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4975 = stablehlo.transpose %v4974, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v4976 = stablehlo.reshape %v4975 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v4977 = stablehlo.reshape %v4967 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4978 = stablehlo.reshape %v4976 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v4979 = stablehlo.dot_general %v4977, %v4978, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v4980 = stablehlo.reshape %v4979 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4981 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v4982 = stablehlo.multiply %v4980, %v4981 : tensor<32x38809xf32>
    %v4983 = stablehlo.reshape %v4982 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4984 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4985 = stablehlo.exponential %v4983 : tensor<32x197x197xf32>
    %v4986 = stablehlo.reduce(%v4985 init: %v4984) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v4987 = stablehlo.broadcast_in_dim %v4986, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v4988 = stablehlo.divide %v4985, %v4987 : tensor<32x197x197xf32>
    %v4989 = stablehlo.reshape %v4988 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v4990 = stablehlo.reshape %v4989 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v4991 = stablehlo.reshape %v4973 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4992 = stablehlo.dot_general %v4990, %v4991, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v4993 = stablehlo.reshape %v4992 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v4994 = stablehlo.reshape %v4993 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v4995 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4996 = stablehlo.pad %v4994, %v4995, low = [0, 0, 640], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v4997 = stablehlo.reshape %v4996 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v4998 = stablehlo.add %v4964, %v4997 : tensor<32x151296xf32>
    %v4999 = stablehlo.reshape %v4615 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5000 = stablehlo.slice %v4999 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5001 = stablehlo.reshape %v5000 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5002 = stablehlo.reshape %v4620 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5003 = stablehlo.slice %v5002 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5004 = stablehlo.reshape %v5003 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5005 = stablehlo.reshape %v4625 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5006 = stablehlo.slice %v5005 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5007 = stablehlo.reshape %v5006 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5008 = stablehlo.reshape %v5004 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5009 = stablehlo.transpose %v5008, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5010 = stablehlo.reshape %v5009 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5011 = stablehlo.reshape %v5001 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5012 = stablehlo.reshape %v5010 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5013 = stablehlo.dot_general %v5011, %v5012, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v5014 = stablehlo.reshape %v5013 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5015 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v5016 = stablehlo.multiply %v5014, %v5015 : tensor<32x38809xf32>
    %v5017 = stablehlo.reshape %v5016 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5018 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5019 = stablehlo.exponential %v5017 : tensor<32x197x197xf32>
    %v5020 = stablehlo.reduce(%v5019 init: %v5018) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5021 = stablehlo.broadcast_in_dim %v5020, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5022 = stablehlo.divide %v5019, %v5021 : tensor<32x197x197xf32>
    %v5023 = stablehlo.reshape %v5022 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5024 = stablehlo.reshape %v5023 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5025 = stablehlo.reshape %v5007 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5026 = stablehlo.dot_general %v5024, %v5025, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5027 = stablehlo.reshape %v5026 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5028 = stablehlo.reshape %v5027 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5029 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5030 = stablehlo.pad %v5028, %v5029, low = [0, 0, 704], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v5031 = stablehlo.reshape %v5030 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5032 = stablehlo.add %v4998, %v5031 : tensor<32x151296xf32>
    %v5033 = stablehlo.reshape %v5032 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5034 = stablehlo.dot_general %v5033, %b9_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v5035 = stablehlo.broadcast_in_dim %b9_bo, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v5036 = stablehlo.add %v5034, %v5035 : tensor<32x197x768xf32>
    %v5037 = stablehlo.reshape %v5036 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5038 = stablehlo.add %v4582, %v5037 : tensor<32x151296xf32>
    %v5039 = stablehlo.reshape %v5038 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5040 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5041 = stablehlo.constant dense<768.0> : tensor<32x197x768xf32>
    %v5042 = stablehlo.constant dense<1.0e-5> : tensor<32x197x768xf32>
    %v5043 = stablehlo.reduce(%v5039 init: %v5040) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5044 = stablehlo.broadcast_in_dim %v5043, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v5045 = stablehlo.divide %v5044, %v5041 : tensor<32x197x768xf32>
    %v5046 = stablehlo.subtract %v5039, %v5045 : tensor<32x197x768xf32>
    %v5047 = stablehlo.multiply %v5046, %v5046 : tensor<32x197x768xf32>
    %v5048 = stablehlo.reduce(%v5047 init: %v5040) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5049 = stablehlo.broadcast_in_dim %v5048, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v5050 = stablehlo.divide %v5049, %v5041 : tensor<32x197x768xf32>
    %v5051 = stablehlo.add %v5050, %v5042 : tensor<32x197x768xf32>
    %v5052 = stablehlo.rsqrt %v5051 : tensor<32x197x768xf32>
    %v5053 = stablehlo.multiply %v5046, %v5052 : tensor<32x197x768xf32>
    %v5054 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v5055 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v5056 = stablehlo.multiply %v5053, %v5054 : tensor<32x197x768xf32>
    %v5057 = stablehlo.add %v5056, %v5055 : tensor<32x197x768xf32>
    %v5058 = stablehlo.reshape %v5057 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5059 = stablehlo.reshape %v5058 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5060 = stablehlo.broadcast_in_dim %b9_g2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v5061 = stablehlo.multiply %v5059, %v5060 : tensor<32x197x768xf32>
    %v5062 = stablehlo.reshape %v5061 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5063 = stablehlo.reshape %v5062 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5064 = stablehlo.broadcast_in_dim %b9_bt2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v5065 = stablehlo.add %v5063, %v5064 : tensor<32x197x768xf32>
    %v5066 = stablehlo.reshape %v5065 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5067 = stablehlo.reshape %v5066 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5068 = stablehlo.dot_general %v5067, %b9_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x3072xf32>) -> tensor<32x197x3072xf32>
    %v5069 = stablehlo.broadcast_in_dim %b9_bfc1, dims = [2] : (tensor<3072xf32>) -> tensor<32x197x3072xf32>
    %v5070 = stablehlo.add %v5068, %v5069 : tensor<32x197x3072xf32>
    %v5071 = stablehlo.reshape %v5070 : (tensor<32x197x3072xf32>) -> tensor<32x605184xf32>
    %v5072 = stablehlo.multiply %v5071, %v5071 : tensor<32x605184xf32>
    %v5073 = stablehlo.multiply %v5072, %v5071 : tensor<32x605184xf32>
    %v5074 = stablehlo.constant dense<0.044715> : tensor<32x605184xf32>
    %v5075 = stablehlo.multiply %v5074, %v5073 : tensor<32x605184xf32>
    %v5076 = stablehlo.add %v5071, %v5075 : tensor<32x605184xf32>
    %v5077 = stablehlo.constant dense<0.7978845608028654> : tensor<32x605184xf32>
    %v5078 = stablehlo.multiply %v5077, %v5076 : tensor<32x605184xf32>
    %v5079 = stablehlo.tanh %v5078 : tensor<32x605184xf32>
    %v5080 = stablehlo.constant dense<1.0> : tensor<32x605184xf32>
    %v5081 = stablehlo.add %v5080, %v5079 : tensor<32x605184xf32>
    %v5082 = stablehlo.constant dense<0.5> : tensor<32x605184xf32>
    %v5083 = stablehlo.multiply %v5082, %v5071 : tensor<32x605184xf32>
    %v5084 = stablehlo.multiply %v5083, %v5081 : tensor<32x605184xf32>
    %v5085 = stablehlo.reshape %v5084 : (tensor<32x605184xf32>) -> tensor<32x197x3072xf32>
    %v5086 = stablehlo.dot_general %v5085, %b9_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x3072xf32>, tensor<3072x768xf32>) -> tensor<32x197x768xf32>
    %v5087 = stablehlo.broadcast_in_dim %b9_bfc2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v5088 = stablehlo.add %v5086, %v5087 : tensor<32x197x768xf32>
    %v5089 = stablehlo.reshape %v5088 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5090 = stablehlo.add %v5038, %v5089 : tensor<32x151296xf32>
    %v5091 = stablehlo.reshape %v5090 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5092 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5093 = stablehlo.constant dense<768.0> : tensor<32x197x768xf32>
    %v5094 = stablehlo.constant dense<1.0e-5> : tensor<32x197x768xf32>
    %v5095 = stablehlo.reduce(%v5091 init: %v5092) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5096 = stablehlo.broadcast_in_dim %v5095, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v5097 = stablehlo.divide %v5096, %v5093 : tensor<32x197x768xf32>
    %v5098 = stablehlo.subtract %v5091, %v5097 : tensor<32x197x768xf32>
    %v5099 = stablehlo.multiply %v5098, %v5098 : tensor<32x197x768xf32>
    %v5100 = stablehlo.reduce(%v5099 init: %v5092) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5101 = stablehlo.broadcast_in_dim %v5100, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v5102 = stablehlo.divide %v5101, %v5093 : tensor<32x197x768xf32>
    %v5103 = stablehlo.add %v5102, %v5094 : tensor<32x197x768xf32>
    %v5104 = stablehlo.rsqrt %v5103 : tensor<32x197x768xf32>
    %v5105 = stablehlo.multiply %v5098, %v5104 : tensor<32x197x768xf32>
    %v5106 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v5107 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v5108 = stablehlo.multiply %v5105, %v5106 : tensor<32x197x768xf32>
    %v5109 = stablehlo.add %v5108, %v5107 : tensor<32x197x768xf32>
    %v5110 = stablehlo.reshape %v5109 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5111 = stablehlo.reshape %v5110 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5112 = stablehlo.broadcast_in_dim %b10_g1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v5113 = stablehlo.multiply %v5111, %v5112 : tensor<32x197x768xf32>
    %v5114 = stablehlo.reshape %v5113 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5115 = stablehlo.reshape %v5114 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5116 = stablehlo.broadcast_in_dim %b10_bt1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v5117 = stablehlo.add %v5115, %v5116 : tensor<32x197x768xf32>
    %v5118 = stablehlo.reshape %v5117 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5119 = stablehlo.reshape %v5118 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5120 = stablehlo.dot_general %v5119, %b10_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v5121 = stablehlo.broadcast_in_dim %b10_bq, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v5122 = stablehlo.add %v5120, %v5121 : tensor<32x197x768xf32>
    %v5123 = stablehlo.reshape %v5122 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5124 = stablehlo.reshape %v5118 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5125 = stablehlo.dot_general %v5124, %b10_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v5126 = stablehlo.broadcast_in_dim %b10_bk, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v5127 = stablehlo.add %v5125, %v5126 : tensor<32x197x768xf32>
    %v5128 = stablehlo.reshape %v5127 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5129 = stablehlo.reshape %v5118 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5130 = stablehlo.dot_general %v5129, %b10_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v5131 = stablehlo.broadcast_in_dim %b10_bv, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v5132 = stablehlo.add %v5130, %v5131 : tensor<32x197x768xf32>
    %v5133 = stablehlo.reshape %v5132 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5134 = stablehlo.reshape %v5123 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5135 = stablehlo.slice %v5134 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5136 = stablehlo.reshape %v5135 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5137 = stablehlo.reshape %v5128 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5138 = stablehlo.slice %v5137 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5139 = stablehlo.reshape %v5138 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5140 = stablehlo.reshape %v5133 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5141 = stablehlo.slice %v5140 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5142 = stablehlo.reshape %v5141 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5143 = stablehlo.reshape %v5139 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5144 = stablehlo.transpose %v5143, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5145 = stablehlo.reshape %v5144 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5146 = stablehlo.reshape %v5136 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5147 = stablehlo.reshape %v5145 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5148 = stablehlo.dot_general %v5146, %v5147, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v5149 = stablehlo.reshape %v5148 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5150 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v5151 = stablehlo.multiply %v5149, %v5150 : tensor<32x38809xf32>
    %v5152 = stablehlo.reshape %v5151 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5153 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5154 = stablehlo.exponential %v5152 : tensor<32x197x197xf32>
    %v5155 = stablehlo.reduce(%v5154 init: %v5153) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5156 = stablehlo.broadcast_in_dim %v5155, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5157 = stablehlo.divide %v5154, %v5156 : tensor<32x197x197xf32>
    %v5158 = stablehlo.reshape %v5157 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5159 = stablehlo.reshape %v5158 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5160 = stablehlo.reshape %v5142 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5161 = stablehlo.dot_general %v5159, %v5160, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5162 = stablehlo.reshape %v5161 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5163 = stablehlo.reshape %v5162 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5164 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5165 = stablehlo.pad %v5163, %v5164, low = [0, 0, 0], high = [0, 0, 704], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v5166 = stablehlo.reshape %v5165 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5167 = stablehlo.reshape %v5123 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5168 = stablehlo.slice %v5167 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5169 = stablehlo.reshape %v5168 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5170 = stablehlo.reshape %v5128 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5171 = stablehlo.slice %v5170 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5172 = stablehlo.reshape %v5171 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5173 = stablehlo.reshape %v5133 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5174 = stablehlo.slice %v5173 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5175 = stablehlo.reshape %v5174 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5176 = stablehlo.reshape %v5172 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5177 = stablehlo.transpose %v5176, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5178 = stablehlo.reshape %v5177 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5179 = stablehlo.reshape %v5169 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5180 = stablehlo.reshape %v5178 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5181 = stablehlo.dot_general %v5179, %v5180, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v5182 = stablehlo.reshape %v5181 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5183 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v5184 = stablehlo.multiply %v5182, %v5183 : tensor<32x38809xf32>
    %v5185 = stablehlo.reshape %v5184 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5186 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5187 = stablehlo.exponential %v5185 : tensor<32x197x197xf32>
    %v5188 = stablehlo.reduce(%v5187 init: %v5186) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5189 = stablehlo.broadcast_in_dim %v5188, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5190 = stablehlo.divide %v5187, %v5189 : tensor<32x197x197xf32>
    %v5191 = stablehlo.reshape %v5190 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5192 = stablehlo.reshape %v5191 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5193 = stablehlo.reshape %v5175 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5194 = stablehlo.dot_general %v5192, %v5193, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5195 = stablehlo.reshape %v5194 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5196 = stablehlo.reshape %v5195 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5197 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5198 = stablehlo.pad %v5196, %v5197, low = [0, 0, 64], high = [0, 0, 640], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v5199 = stablehlo.reshape %v5198 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5200 = stablehlo.add %v5166, %v5199 : tensor<32x151296xf32>
    %v5201 = stablehlo.reshape %v5123 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5202 = stablehlo.slice %v5201 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5203 = stablehlo.reshape %v5202 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5204 = stablehlo.reshape %v5128 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5205 = stablehlo.slice %v5204 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5206 = stablehlo.reshape %v5205 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5207 = stablehlo.reshape %v5133 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5208 = stablehlo.slice %v5207 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5209 = stablehlo.reshape %v5208 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5210 = stablehlo.reshape %v5206 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5211 = stablehlo.transpose %v5210, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5212 = stablehlo.reshape %v5211 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5213 = stablehlo.reshape %v5203 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5214 = stablehlo.reshape %v5212 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5215 = stablehlo.dot_general %v5213, %v5214, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v5216 = stablehlo.reshape %v5215 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5217 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v5218 = stablehlo.multiply %v5216, %v5217 : tensor<32x38809xf32>
    %v5219 = stablehlo.reshape %v5218 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5220 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5221 = stablehlo.exponential %v5219 : tensor<32x197x197xf32>
    %v5222 = stablehlo.reduce(%v5221 init: %v5220) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5223 = stablehlo.broadcast_in_dim %v5222, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5224 = stablehlo.divide %v5221, %v5223 : tensor<32x197x197xf32>
    %v5225 = stablehlo.reshape %v5224 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5226 = stablehlo.reshape %v5225 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5227 = stablehlo.reshape %v5209 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5228 = stablehlo.dot_general %v5226, %v5227, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5229 = stablehlo.reshape %v5228 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5230 = stablehlo.reshape %v5229 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5231 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5232 = stablehlo.pad %v5230, %v5231, low = [0, 0, 128], high = [0, 0, 576], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v5233 = stablehlo.reshape %v5232 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5234 = stablehlo.add %v5200, %v5233 : tensor<32x151296xf32>
    %v5235 = stablehlo.reshape %v5123 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5236 = stablehlo.slice %v5235 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5237 = stablehlo.reshape %v5236 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5238 = stablehlo.reshape %v5128 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5239 = stablehlo.slice %v5238 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5240 = stablehlo.reshape %v5239 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5241 = stablehlo.reshape %v5133 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5242 = stablehlo.slice %v5241 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5243 = stablehlo.reshape %v5242 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5244 = stablehlo.reshape %v5240 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5245 = stablehlo.transpose %v5244, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5246 = stablehlo.reshape %v5245 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5247 = stablehlo.reshape %v5237 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5248 = stablehlo.reshape %v5246 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5249 = stablehlo.dot_general %v5247, %v5248, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v5250 = stablehlo.reshape %v5249 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5251 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v5252 = stablehlo.multiply %v5250, %v5251 : tensor<32x38809xf32>
    %v5253 = stablehlo.reshape %v5252 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5254 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5255 = stablehlo.exponential %v5253 : tensor<32x197x197xf32>
    %v5256 = stablehlo.reduce(%v5255 init: %v5254) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5257 = stablehlo.broadcast_in_dim %v5256, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5258 = stablehlo.divide %v5255, %v5257 : tensor<32x197x197xf32>
    %v5259 = stablehlo.reshape %v5258 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5260 = stablehlo.reshape %v5259 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5261 = stablehlo.reshape %v5243 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5262 = stablehlo.dot_general %v5260, %v5261, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5263 = stablehlo.reshape %v5262 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5264 = stablehlo.reshape %v5263 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5265 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5266 = stablehlo.pad %v5264, %v5265, low = [0, 0, 192], high = [0, 0, 512], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v5267 = stablehlo.reshape %v5266 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5268 = stablehlo.add %v5234, %v5267 : tensor<32x151296xf32>
    %v5269 = stablehlo.reshape %v5123 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5270 = stablehlo.slice %v5269 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5271 = stablehlo.reshape %v5270 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5272 = stablehlo.reshape %v5128 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5273 = stablehlo.slice %v5272 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5274 = stablehlo.reshape %v5273 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5275 = stablehlo.reshape %v5133 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5276 = stablehlo.slice %v5275 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
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
    %v5288 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5289 = stablehlo.exponential %v5287 : tensor<32x197x197xf32>
    %v5290 = stablehlo.reduce(%v5289 init: %v5288) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5291 = stablehlo.broadcast_in_dim %v5290, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5292 = stablehlo.divide %v5289, %v5291 : tensor<32x197x197xf32>
    %v5293 = stablehlo.reshape %v5292 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5294 = stablehlo.reshape %v5293 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5295 = stablehlo.reshape %v5277 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5296 = stablehlo.dot_general %v5294, %v5295, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5297 = stablehlo.reshape %v5296 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5298 = stablehlo.reshape %v5297 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5299 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5300 = stablehlo.pad %v5298, %v5299, low = [0, 0, 256], high = [0, 0, 448], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v5301 = stablehlo.reshape %v5300 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5302 = stablehlo.add %v5268, %v5301 : tensor<32x151296xf32>
    %v5303 = stablehlo.reshape %v5123 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5304 = stablehlo.slice %v5303 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5305 = stablehlo.reshape %v5304 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5306 = stablehlo.reshape %v5128 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5307 = stablehlo.slice %v5306 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5308 = stablehlo.reshape %v5307 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5309 = stablehlo.reshape %v5133 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5310 = stablehlo.slice %v5309 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5311 = stablehlo.reshape %v5310 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5312 = stablehlo.reshape %v5308 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5313 = stablehlo.transpose %v5312, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5314 = stablehlo.reshape %v5313 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5315 = stablehlo.reshape %v5305 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5316 = stablehlo.reshape %v5314 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5317 = stablehlo.dot_general %v5315, %v5316, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v5318 = stablehlo.reshape %v5317 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5319 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v5320 = stablehlo.multiply %v5318, %v5319 : tensor<32x38809xf32>
    %v5321 = stablehlo.reshape %v5320 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5322 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5323 = stablehlo.exponential %v5321 : tensor<32x197x197xf32>
    %v5324 = stablehlo.reduce(%v5323 init: %v5322) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5325 = stablehlo.broadcast_in_dim %v5324, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5326 = stablehlo.divide %v5323, %v5325 : tensor<32x197x197xf32>
    %v5327 = stablehlo.reshape %v5326 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5328 = stablehlo.reshape %v5327 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5329 = stablehlo.reshape %v5311 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5330 = stablehlo.dot_general %v5328, %v5329, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5331 = stablehlo.reshape %v5330 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5332 = stablehlo.reshape %v5331 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5333 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5334 = stablehlo.pad %v5332, %v5333, low = [0, 0, 320], high = [0, 0, 384], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v5335 = stablehlo.reshape %v5334 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5336 = stablehlo.add %v5302, %v5335 : tensor<32x151296xf32>
    %v5337 = stablehlo.reshape %v5123 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5338 = stablehlo.slice %v5337 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5339 = stablehlo.reshape %v5338 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5340 = stablehlo.reshape %v5128 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5341 = stablehlo.slice %v5340 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5342 = stablehlo.reshape %v5341 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5343 = stablehlo.reshape %v5133 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5344 = stablehlo.slice %v5343 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5345 = stablehlo.reshape %v5344 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5346 = stablehlo.reshape %v5342 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5347 = stablehlo.transpose %v5346, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5348 = stablehlo.reshape %v5347 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5349 = stablehlo.reshape %v5339 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5350 = stablehlo.reshape %v5348 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5351 = stablehlo.dot_general %v5349, %v5350, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v5352 = stablehlo.reshape %v5351 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5353 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v5354 = stablehlo.multiply %v5352, %v5353 : tensor<32x38809xf32>
    %v5355 = stablehlo.reshape %v5354 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5356 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5357 = stablehlo.exponential %v5355 : tensor<32x197x197xf32>
    %v5358 = stablehlo.reduce(%v5357 init: %v5356) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5359 = stablehlo.broadcast_in_dim %v5358, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5360 = stablehlo.divide %v5357, %v5359 : tensor<32x197x197xf32>
    %v5361 = stablehlo.reshape %v5360 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5362 = stablehlo.reshape %v5361 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5363 = stablehlo.reshape %v5345 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5364 = stablehlo.dot_general %v5362, %v5363, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5365 = stablehlo.reshape %v5364 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5366 = stablehlo.reshape %v5365 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5367 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5368 = stablehlo.pad %v5366, %v5367, low = [0, 0, 384], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v5369 = stablehlo.reshape %v5368 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5370 = stablehlo.add %v5336, %v5369 : tensor<32x151296xf32>
    %v5371 = stablehlo.reshape %v5123 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5372 = stablehlo.slice %v5371 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5373 = stablehlo.reshape %v5372 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5374 = stablehlo.reshape %v5128 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5375 = stablehlo.slice %v5374 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5376 = stablehlo.reshape %v5375 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5377 = stablehlo.reshape %v5133 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5378 = stablehlo.slice %v5377 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5379 = stablehlo.reshape %v5378 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5380 = stablehlo.reshape %v5376 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5381 = stablehlo.transpose %v5380, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5382 = stablehlo.reshape %v5381 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5383 = stablehlo.reshape %v5373 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5384 = stablehlo.reshape %v5382 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5385 = stablehlo.dot_general %v5383, %v5384, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v5386 = stablehlo.reshape %v5385 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5387 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v5388 = stablehlo.multiply %v5386, %v5387 : tensor<32x38809xf32>
    %v5389 = stablehlo.reshape %v5388 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5390 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5391 = stablehlo.exponential %v5389 : tensor<32x197x197xf32>
    %v5392 = stablehlo.reduce(%v5391 init: %v5390) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5393 = stablehlo.broadcast_in_dim %v5392, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5394 = stablehlo.divide %v5391, %v5393 : tensor<32x197x197xf32>
    %v5395 = stablehlo.reshape %v5394 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5396 = stablehlo.reshape %v5395 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5397 = stablehlo.reshape %v5379 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5398 = stablehlo.dot_general %v5396, %v5397, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5399 = stablehlo.reshape %v5398 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5400 = stablehlo.reshape %v5399 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5401 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5402 = stablehlo.pad %v5400, %v5401, low = [0, 0, 448], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v5403 = stablehlo.reshape %v5402 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5404 = stablehlo.add %v5370, %v5403 : tensor<32x151296xf32>
    %v5405 = stablehlo.reshape %v5123 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5406 = stablehlo.slice %v5405 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5407 = stablehlo.reshape %v5406 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5408 = stablehlo.reshape %v5128 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5409 = stablehlo.slice %v5408 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5410 = stablehlo.reshape %v5409 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5411 = stablehlo.reshape %v5133 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5412 = stablehlo.slice %v5411 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5413 = stablehlo.reshape %v5412 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5414 = stablehlo.reshape %v5410 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5415 = stablehlo.transpose %v5414, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5416 = stablehlo.reshape %v5415 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5417 = stablehlo.reshape %v5407 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5418 = stablehlo.reshape %v5416 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5419 = stablehlo.dot_general %v5417, %v5418, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v5420 = stablehlo.reshape %v5419 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5421 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v5422 = stablehlo.multiply %v5420, %v5421 : tensor<32x38809xf32>
    %v5423 = stablehlo.reshape %v5422 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5424 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5425 = stablehlo.exponential %v5423 : tensor<32x197x197xf32>
    %v5426 = stablehlo.reduce(%v5425 init: %v5424) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5427 = stablehlo.broadcast_in_dim %v5426, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5428 = stablehlo.divide %v5425, %v5427 : tensor<32x197x197xf32>
    %v5429 = stablehlo.reshape %v5428 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5430 = stablehlo.reshape %v5429 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5431 = stablehlo.reshape %v5413 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5432 = stablehlo.dot_general %v5430, %v5431, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5433 = stablehlo.reshape %v5432 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5434 = stablehlo.reshape %v5433 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5435 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5436 = stablehlo.pad %v5434, %v5435, low = [0, 0, 512], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v5437 = stablehlo.reshape %v5436 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5438 = stablehlo.add %v5404, %v5437 : tensor<32x151296xf32>
    %v5439 = stablehlo.reshape %v5123 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5440 = stablehlo.slice %v5439 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5441 = stablehlo.reshape %v5440 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5442 = stablehlo.reshape %v5128 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5443 = stablehlo.slice %v5442 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5444 = stablehlo.reshape %v5443 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5445 = stablehlo.reshape %v5133 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5446 = stablehlo.slice %v5445 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5447 = stablehlo.reshape %v5446 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5448 = stablehlo.reshape %v5444 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5449 = stablehlo.transpose %v5448, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5450 = stablehlo.reshape %v5449 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5451 = stablehlo.reshape %v5441 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5452 = stablehlo.reshape %v5450 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5453 = stablehlo.dot_general %v5451, %v5452, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v5454 = stablehlo.reshape %v5453 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5455 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v5456 = stablehlo.multiply %v5454, %v5455 : tensor<32x38809xf32>
    %v5457 = stablehlo.reshape %v5456 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5458 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5459 = stablehlo.exponential %v5457 : tensor<32x197x197xf32>
    %v5460 = stablehlo.reduce(%v5459 init: %v5458) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5461 = stablehlo.broadcast_in_dim %v5460, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5462 = stablehlo.divide %v5459, %v5461 : tensor<32x197x197xf32>
    %v5463 = stablehlo.reshape %v5462 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5464 = stablehlo.reshape %v5463 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5465 = stablehlo.reshape %v5447 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5466 = stablehlo.dot_general %v5464, %v5465, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5467 = stablehlo.reshape %v5466 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5468 = stablehlo.reshape %v5467 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5469 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5470 = stablehlo.pad %v5468, %v5469, low = [0, 0, 576], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v5471 = stablehlo.reshape %v5470 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5472 = stablehlo.add %v5438, %v5471 : tensor<32x151296xf32>
    %v5473 = stablehlo.reshape %v5123 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5474 = stablehlo.slice %v5473 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5475 = stablehlo.reshape %v5474 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5476 = stablehlo.reshape %v5128 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5477 = stablehlo.slice %v5476 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5478 = stablehlo.reshape %v5477 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5479 = stablehlo.reshape %v5133 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5480 = stablehlo.slice %v5479 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5481 = stablehlo.reshape %v5480 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5482 = stablehlo.reshape %v5478 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5483 = stablehlo.transpose %v5482, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5484 = stablehlo.reshape %v5483 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5485 = stablehlo.reshape %v5475 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5486 = stablehlo.reshape %v5484 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5487 = stablehlo.dot_general %v5485, %v5486, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v5488 = stablehlo.reshape %v5487 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5489 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v5490 = stablehlo.multiply %v5488, %v5489 : tensor<32x38809xf32>
    %v5491 = stablehlo.reshape %v5490 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5492 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5493 = stablehlo.exponential %v5491 : tensor<32x197x197xf32>
    %v5494 = stablehlo.reduce(%v5493 init: %v5492) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5495 = stablehlo.broadcast_in_dim %v5494, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5496 = stablehlo.divide %v5493, %v5495 : tensor<32x197x197xf32>
    %v5497 = stablehlo.reshape %v5496 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5498 = stablehlo.reshape %v5497 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5499 = stablehlo.reshape %v5481 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5500 = stablehlo.dot_general %v5498, %v5499, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5501 = stablehlo.reshape %v5500 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5502 = stablehlo.reshape %v5501 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5503 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5504 = stablehlo.pad %v5502, %v5503, low = [0, 0, 640], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v5505 = stablehlo.reshape %v5504 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5506 = stablehlo.add %v5472, %v5505 : tensor<32x151296xf32>
    %v5507 = stablehlo.reshape %v5123 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5508 = stablehlo.slice %v5507 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5509 = stablehlo.reshape %v5508 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5510 = stablehlo.reshape %v5128 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5511 = stablehlo.slice %v5510 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5512 = stablehlo.reshape %v5511 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5513 = stablehlo.reshape %v5133 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5514 = stablehlo.slice %v5513 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5515 = stablehlo.reshape %v5514 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5516 = stablehlo.reshape %v5512 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5517 = stablehlo.transpose %v5516, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5518 = stablehlo.reshape %v5517 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5519 = stablehlo.reshape %v5509 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5520 = stablehlo.reshape %v5518 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5521 = stablehlo.dot_general %v5519, %v5520, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v5522 = stablehlo.reshape %v5521 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5523 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v5524 = stablehlo.multiply %v5522, %v5523 : tensor<32x38809xf32>
    %v5525 = stablehlo.reshape %v5524 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5526 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5527 = stablehlo.exponential %v5525 : tensor<32x197x197xf32>
    %v5528 = stablehlo.reduce(%v5527 init: %v5526) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5529 = stablehlo.broadcast_in_dim %v5528, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5530 = stablehlo.divide %v5527, %v5529 : tensor<32x197x197xf32>
    %v5531 = stablehlo.reshape %v5530 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5532 = stablehlo.reshape %v5531 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5533 = stablehlo.reshape %v5515 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5534 = stablehlo.dot_general %v5532, %v5533, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5535 = stablehlo.reshape %v5534 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5536 = stablehlo.reshape %v5535 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5537 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5538 = stablehlo.pad %v5536, %v5537, low = [0, 0, 704], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v5539 = stablehlo.reshape %v5538 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5540 = stablehlo.add %v5506, %v5539 : tensor<32x151296xf32>
    %v5541 = stablehlo.reshape %v5540 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5542 = stablehlo.dot_general %v5541, %b10_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v5543 = stablehlo.broadcast_in_dim %b10_bo, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v5544 = stablehlo.add %v5542, %v5543 : tensor<32x197x768xf32>
    %v5545 = stablehlo.reshape %v5544 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5546 = stablehlo.add %v5090, %v5545 : tensor<32x151296xf32>
    %v5547 = stablehlo.reshape %v5546 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5548 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5549 = stablehlo.constant dense<768.0> : tensor<32x197x768xf32>
    %v5550 = stablehlo.constant dense<1.0e-5> : tensor<32x197x768xf32>
    %v5551 = stablehlo.reduce(%v5547 init: %v5548) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5552 = stablehlo.broadcast_in_dim %v5551, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v5553 = stablehlo.divide %v5552, %v5549 : tensor<32x197x768xf32>
    %v5554 = stablehlo.subtract %v5547, %v5553 : tensor<32x197x768xf32>
    %v5555 = stablehlo.multiply %v5554, %v5554 : tensor<32x197x768xf32>
    %v5556 = stablehlo.reduce(%v5555 init: %v5548) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5557 = stablehlo.broadcast_in_dim %v5556, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v5558 = stablehlo.divide %v5557, %v5549 : tensor<32x197x768xf32>
    %v5559 = stablehlo.add %v5558, %v5550 : tensor<32x197x768xf32>
    %v5560 = stablehlo.rsqrt %v5559 : tensor<32x197x768xf32>
    %v5561 = stablehlo.multiply %v5554, %v5560 : tensor<32x197x768xf32>
    %v5562 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v5563 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v5564 = stablehlo.multiply %v5561, %v5562 : tensor<32x197x768xf32>
    %v5565 = stablehlo.add %v5564, %v5563 : tensor<32x197x768xf32>
    %v5566 = stablehlo.reshape %v5565 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5567 = stablehlo.reshape %v5566 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5568 = stablehlo.broadcast_in_dim %b10_g2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v5569 = stablehlo.multiply %v5567, %v5568 : tensor<32x197x768xf32>
    %v5570 = stablehlo.reshape %v5569 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5571 = stablehlo.reshape %v5570 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5572 = stablehlo.broadcast_in_dim %b10_bt2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v5573 = stablehlo.add %v5571, %v5572 : tensor<32x197x768xf32>
    %v5574 = stablehlo.reshape %v5573 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5575 = stablehlo.reshape %v5574 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5576 = stablehlo.dot_general %v5575, %b10_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x3072xf32>) -> tensor<32x197x3072xf32>
    %v5577 = stablehlo.broadcast_in_dim %b10_bfc1, dims = [2] : (tensor<3072xf32>) -> tensor<32x197x3072xf32>
    %v5578 = stablehlo.add %v5576, %v5577 : tensor<32x197x3072xf32>
    %v5579 = stablehlo.reshape %v5578 : (tensor<32x197x3072xf32>) -> tensor<32x605184xf32>
    %v5580 = stablehlo.multiply %v5579, %v5579 : tensor<32x605184xf32>
    %v5581 = stablehlo.multiply %v5580, %v5579 : tensor<32x605184xf32>
    %v5582 = stablehlo.constant dense<0.044715> : tensor<32x605184xf32>
    %v5583 = stablehlo.multiply %v5582, %v5581 : tensor<32x605184xf32>
    %v5584 = stablehlo.add %v5579, %v5583 : tensor<32x605184xf32>
    %v5585 = stablehlo.constant dense<0.7978845608028654> : tensor<32x605184xf32>
    %v5586 = stablehlo.multiply %v5585, %v5584 : tensor<32x605184xf32>
    %v5587 = stablehlo.tanh %v5586 : tensor<32x605184xf32>
    %v5588 = stablehlo.constant dense<1.0> : tensor<32x605184xf32>
    %v5589 = stablehlo.add %v5588, %v5587 : tensor<32x605184xf32>
    %v5590 = stablehlo.constant dense<0.5> : tensor<32x605184xf32>
    %v5591 = stablehlo.multiply %v5590, %v5579 : tensor<32x605184xf32>
    %v5592 = stablehlo.multiply %v5591, %v5589 : tensor<32x605184xf32>
    %v5593 = stablehlo.reshape %v5592 : (tensor<32x605184xf32>) -> tensor<32x197x3072xf32>
    %v5594 = stablehlo.dot_general %v5593, %b10_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x3072xf32>, tensor<3072x768xf32>) -> tensor<32x197x768xf32>
    %v5595 = stablehlo.broadcast_in_dim %b10_bfc2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v5596 = stablehlo.add %v5594, %v5595 : tensor<32x197x768xf32>
    %v5597 = stablehlo.reshape %v5596 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5598 = stablehlo.add %v5546, %v5597 : tensor<32x151296xf32>
    %v5599 = stablehlo.reshape %v5598 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5600 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5601 = stablehlo.constant dense<768.0> : tensor<32x197x768xf32>
    %v5602 = stablehlo.constant dense<1.0e-5> : tensor<32x197x768xf32>
    %v5603 = stablehlo.reduce(%v5599 init: %v5600) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5604 = stablehlo.broadcast_in_dim %v5603, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v5605 = stablehlo.divide %v5604, %v5601 : tensor<32x197x768xf32>
    %v5606 = stablehlo.subtract %v5599, %v5605 : tensor<32x197x768xf32>
    %v5607 = stablehlo.multiply %v5606, %v5606 : tensor<32x197x768xf32>
    %v5608 = stablehlo.reduce(%v5607 init: %v5600) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5609 = stablehlo.broadcast_in_dim %v5608, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v5610 = stablehlo.divide %v5609, %v5601 : tensor<32x197x768xf32>
    %v5611 = stablehlo.add %v5610, %v5602 : tensor<32x197x768xf32>
    %v5612 = stablehlo.rsqrt %v5611 : tensor<32x197x768xf32>
    %v5613 = stablehlo.multiply %v5606, %v5612 : tensor<32x197x768xf32>
    %v5614 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v5615 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v5616 = stablehlo.multiply %v5613, %v5614 : tensor<32x197x768xf32>
    %v5617 = stablehlo.add %v5616, %v5615 : tensor<32x197x768xf32>
    %v5618 = stablehlo.reshape %v5617 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5619 = stablehlo.reshape %v5618 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5620 = stablehlo.broadcast_in_dim %b11_g1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v5621 = stablehlo.multiply %v5619, %v5620 : tensor<32x197x768xf32>
    %v5622 = stablehlo.reshape %v5621 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5623 = stablehlo.reshape %v5622 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5624 = stablehlo.broadcast_in_dim %b11_bt1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v5625 = stablehlo.add %v5623, %v5624 : tensor<32x197x768xf32>
    %v5626 = stablehlo.reshape %v5625 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5627 = stablehlo.reshape %v5626 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5628 = stablehlo.dot_general %v5627, %b11_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v5629 = stablehlo.broadcast_in_dim %b11_bq, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v5630 = stablehlo.add %v5628, %v5629 : tensor<32x197x768xf32>
    %v5631 = stablehlo.reshape %v5630 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5632 = stablehlo.reshape %v5626 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5633 = stablehlo.dot_general %v5632, %b11_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v5634 = stablehlo.broadcast_in_dim %b11_bk, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v5635 = stablehlo.add %v5633, %v5634 : tensor<32x197x768xf32>
    %v5636 = stablehlo.reshape %v5635 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5637 = stablehlo.reshape %v5626 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5638 = stablehlo.dot_general %v5637, %b11_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v5639 = stablehlo.broadcast_in_dim %b11_bv, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v5640 = stablehlo.add %v5638, %v5639 : tensor<32x197x768xf32>
    %v5641 = stablehlo.reshape %v5640 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5642 = stablehlo.reshape %v5631 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5643 = stablehlo.slice %v5642 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5644 = stablehlo.reshape %v5643 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5645 = stablehlo.reshape %v5636 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5646 = stablehlo.slice %v5645 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5647 = stablehlo.reshape %v5646 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5648 = stablehlo.reshape %v5641 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5649 = stablehlo.slice %v5648 [0:32, 0:197, 0:64] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5650 = stablehlo.reshape %v5649 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5651 = stablehlo.reshape %v5647 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5652 = stablehlo.transpose %v5651, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5653 = stablehlo.reshape %v5652 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5654 = stablehlo.reshape %v5644 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5655 = stablehlo.reshape %v5653 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5656 = stablehlo.dot_general %v5654, %v5655, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v5657 = stablehlo.reshape %v5656 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5658 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v5659 = stablehlo.multiply %v5657, %v5658 : tensor<32x38809xf32>
    %v5660 = stablehlo.reshape %v5659 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5661 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5662 = stablehlo.exponential %v5660 : tensor<32x197x197xf32>
    %v5663 = stablehlo.reduce(%v5662 init: %v5661) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5664 = stablehlo.broadcast_in_dim %v5663, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5665 = stablehlo.divide %v5662, %v5664 : tensor<32x197x197xf32>
    %v5666 = stablehlo.reshape %v5665 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5667 = stablehlo.reshape %v5666 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5668 = stablehlo.reshape %v5650 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5669 = stablehlo.dot_general %v5667, %v5668, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5670 = stablehlo.reshape %v5669 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5671 = stablehlo.reshape %v5670 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5672 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5673 = stablehlo.pad %v5671, %v5672, low = [0, 0, 0], high = [0, 0, 704], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v5674 = stablehlo.reshape %v5673 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5675 = stablehlo.reshape %v5631 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5676 = stablehlo.slice %v5675 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5677 = stablehlo.reshape %v5676 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5678 = stablehlo.reshape %v5636 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5679 = stablehlo.slice %v5678 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5680 = stablehlo.reshape %v5679 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5681 = stablehlo.reshape %v5641 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5682 = stablehlo.slice %v5681 [0:32, 0:197, 64:128] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5683 = stablehlo.reshape %v5682 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5684 = stablehlo.reshape %v5680 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5685 = stablehlo.transpose %v5684, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5686 = stablehlo.reshape %v5685 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5687 = stablehlo.reshape %v5677 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5688 = stablehlo.reshape %v5686 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5689 = stablehlo.dot_general %v5687, %v5688, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v5690 = stablehlo.reshape %v5689 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5691 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v5692 = stablehlo.multiply %v5690, %v5691 : tensor<32x38809xf32>
    %v5693 = stablehlo.reshape %v5692 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5694 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5695 = stablehlo.exponential %v5693 : tensor<32x197x197xf32>
    %v5696 = stablehlo.reduce(%v5695 init: %v5694) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5697 = stablehlo.broadcast_in_dim %v5696, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5698 = stablehlo.divide %v5695, %v5697 : tensor<32x197x197xf32>
    %v5699 = stablehlo.reshape %v5698 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5700 = stablehlo.reshape %v5699 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5701 = stablehlo.reshape %v5683 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5702 = stablehlo.dot_general %v5700, %v5701, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5703 = stablehlo.reshape %v5702 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5704 = stablehlo.reshape %v5703 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5705 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5706 = stablehlo.pad %v5704, %v5705, low = [0, 0, 64], high = [0, 0, 640], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v5707 = stablehlo.reshape %v5706 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5708 = stablehlo.add %v5674, %v5707 : tensor<32x151296xf32>
    %v5709 = stablehlo.reshape %v5631 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5710 = stablehlo.slice %v5709 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5711 = stablehlo.reshape %v5710 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5712 = stablehlo.reshape %v5636 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5713 = stablehlo.slice %v5712 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5714 = stablehlo.reshape %v5713 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5715 = stablehlo.reshape %v5641 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5716 = stablehlo.slice %v5715 [0:32, 0:197, 128:192] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5717 = stablehlo.reshape %v5716 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5718 = stablehlo.reshape %v5714 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5719 = stablehlo.transpose %v5718, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5720 = stablehlo.reshape %v5719 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5721 = stablehlo.reshape %v5711 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5722 = stablehlo.reshape %v5720 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5723 = stablehlo.dot_general %v5721, %v5722, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v5724 = stablehlo.reshape %v5723 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5725 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v5726 = stablehlo.multiply %v5724, %v5725 : tensor<32x38809xf32>
    %v5727 = stablehlo.reshape %v5726 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5728 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5729 = stablehlo.exponential %v5727 : tensor<32x197x197xf32>
    %v5730 = stablehlo.reduce(%v5729 init: %v5728) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5731 = stablehlo.broadcast_in_dim %v5730, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5732 = stablehlo.divide %v5729, %v5731 : tensor<32x197x197xf32>
    %v5733 = stablehlo.reshape %v5732 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5734 = stablehlo.reshape %v5733 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5735 = stablehlo.reshape %v5717 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5736 = stablehlo.dot_general %v5734, %v5735, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5737 = stablehlo.reshape %v5736 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5738 = stablehlo.reshape %v5737 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5739 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5740 = stablehlo.pad %v5738, %v5739, low = [0, 0, 128], high = [0, 0, 576], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v5741 = stablehlo.reshape %v5740 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5742 = stablehlo.add %v5708, %v5741 : tensor<32x151296xf32>
    %v5743 = stablehlo.reshape %v5631 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5744 = stablehlo.slice %v5743 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5745 = stablehlo.reshape %v5744 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5746 = stablehlo.reshape %v5636 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5747 = stablehlo.slice %v5746 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5748 = stablehlo.reshape %v5747 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5749 = stablehlo.reshape %v5641 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5750 = stablehlo.slice %v5749 [0:32, 0:197, 192:256] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
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
    %v5762 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5763 = stablehlo.exponential %v5761 : tensor<32x197x197xf32>
    %v5764 = stablehlo.reduce(%v5763 init: %v5762) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5765 = stablehlo.broadcast_in_dim %v5764, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5766 = stablehlo.divide %v5763, %v5765 : tensor<32x197x197xf32>
    %v5767 = stablehlo.reshape %v5766 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5768 = stablehlo.reshape %v5767 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5769 = stablehlo.reshape %v5751 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5770 = stablehlo.dot_general %v5768, %v5769, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5771 = stablehlo.reshape %v5770 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5772 = stablehlo.reshape %v5771 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5773 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5774 = stablehlo.pad %v5772, %v5773, low = [0, 0, 192], high = [0, 0, 512], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v5775 = stablehlo.reshape %v5774 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5776 = stablehlo.add %v5742, %v5775 : tensor<32x151296xf32>
    %v5777 = stablehlo.reshape %v5631 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5778 = stablehlo.slice %v5777 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5779 = stablehlo.reshape %v5778 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5780 = stablehlo.reshape %v5636 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5781 = stablehlo.slice %v5780 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5782 = stablehlo.reshape %v5781 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5783 = stablehlo.reshape %v5641 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5784 = stablehlo.slice %v5783 [0:32, 0:197, 256:320] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5785 = stablehlo.reshape %v5784 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5786 = stablehlo.reshape %v5782 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5787 = stablehlo.transpose %v5786, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5788 = stablehlo.reshape %v5787 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5789 = stablehlo.reshape %v5779 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5790 = stablehlo.reshape %v5788 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5791 = stablehlo.dot_general %v5789, %v5790, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v5792 = stablehlo.reshape %v5791 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5793 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v5794 = stablehlo.multiply %v5792, %v5793 : tensor<32x38809xf32>
    %v5795 = stablehlo.reshape %v5794 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5796 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5797 = stablehlo.exponential %v5795 : tensor<32x197x197xf32>
    %v5798 = stablehlo.reduce(%v5797 init: %v5796) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5799 = stablehlo.broadcast_in_dim %v5798, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5800 = stablehlo.divide %v5797, %v5799 : tensor<32x197x197xf32>
    %v5801 = stablehlo.reshape %v5800 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5802 = stablehlo.reshape %v5801 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5803 = stablehlo.reshape %v5785 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5804 = stablehlo.dot_general %v5802, %v5803, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5805 = stablehlo.reshape %v5804 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5806 = stablehlo.reshape %v5805 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5807 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5808 = stablehlo.pad %v5806, %v5807, low = [0, 0, 256], high = [0, 0, 448], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v5809 = stablehlo.reshape %v5808 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5810 = stablehlo.add %v5776, %v5809 : tensor<32x151296xf32>
    %v5811 = stablehlo.reshape %v5631 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5812 = stablehlo.slice %v5811 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5813 = stablehlo.reshape %v5812 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5814 = stablehlo.reshape %v5636 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5815 = stablehlo.slice %v5814 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5816 = stablehlo.reshape %v5815 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5817 = stablehlo.reshape %v5641 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5818 = stablehlo.slice %v5817 [0:32, 0:197, 320:384] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5819 = stablehlo.reshape %v5818 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5820 = stablehlo.reshape %v5816 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5821 = stablehlo.transpose %v5820, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5822 = stablehlo.reshape %v5821 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5823 = stablehlo.reshape %v5813 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5824 = stablehlo.reshape %v5822 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5825 = stablehlo.dot_general %v5823, %v5824, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v5826 = stablehlo.reshape %v5825 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5827 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v5828 = stablehlo.multiply %v5826, %v5827 : tensor<32x38809xf32>
    %v5829 = stablehlo.reshape %v5828 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5830 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5831 = stablehlo.exponential %v5829 : tensor<32x197x197xf32>
    %v5832 = stablehlo.reduce(%v5831 init: %v5830) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5833 = stablehlo.broadcast_in_dim %v5832, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5834 = stablehlo.divide %v5831, %v5833 : tensor<32x197x197xf32>
    %v5835 = stablehlo.reshape %v5834 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5836 = stablehlo.reshape %v5835 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5837 = stablehlo.reshape %v5819 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5838 = stablehlo.dot_general %v5836, %v5837, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5839 = stablehlo.reshape %v5838 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5840 = stablehlo.reshape %v5839 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5841 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5842 = stablehlo.pad %v5840, %v5841, low = [0, 0, 320], high = [0, 0, 384], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v5843 = stablehlo.reshape %v5842 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5844 = stablehlo.add %v5810, %v5843 : tensor<32x151296xf32>
    %v5845 = stablehlo.reshape %v5631 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5846 = stablehlo.slice %v5845 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5847 = stablehlo.reshape %v5846 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5848 = stablehlo.reshape %v5636 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5849 = stablehlo.slice %v5848 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5850 = stablehlo.reshape %v5849 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5851 = stablehlo.reshape %v5641 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5852 = stablehlo.slice %v5851 [0:32, 0:197, 384:448] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5853 = stablehlo.reshape %v5852 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5854 = stablehlo.reshape %v5850 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5855 = stablehlo.transpose %v5854, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5856 = stablehlo.reshape %v5855 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5857 = stablehlo.reshape %v5847 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5858 = stablehlo.reshape %v5856 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5859 = stablehlo.dot_general %v5857, %v5858, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v5860 = stablehlo.reshape %v5859 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5861 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v5862 = stablehlo.multiply %v5860, %v5861 : tensor<32x38809xf32>
    %v5863 = stablehlo.reshape %v5862 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5864 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5865 = stablehlo.exponential %v5863 : tensor<32x197x197xf32>
    %v5866 = stablehlo.reduce(%v5865 init: %v5864) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5867 = stablehlo.broadcast_in_dim %v5866, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5868 = stablehlo.divide %v5865, %v5867 : tensor<32x197x197xf32>
    %v5869 = stablehlo.reshape %v5868 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5870 = stablehlo.reshape %v5869 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5871 = stablehlo.reshape %v5853 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5872 = stablehlo.dot_general %v5870, %v5871, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5873 = stablehlo.reshape %v5872 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5874 = stablehlo.reshape %v5873 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5875 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5876 = stablehlo.pad %v5874, %v5875, low = [0, 0, 384], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v5877 = stablehlo.reshape %v5876 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5878 = stablehlo.add %v5844, %v5877 : tensor<32x151296xf32>
    %v5879 = stablehlo.reshape %v5631 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5880 = stablehlo.slice %v5879 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5881 = stablehlo.reshape %v5880 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5882 = stablehlo.reshape %v5636 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5883 = stablehlo.slice %v5882 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5884 = stablehlo.reshape %v5883 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5885 = stablehlo.reshape %v5641 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5886 = stablehlo.slice %v5885 [0:32, 0:197, 448:512] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5887 = stablehlo.reshape %v5886 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5888 = stablehlo.reshape %v5884 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5889 = stablehlo.transpose %v5888, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5890 = stablehlo.reshape %v5889 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5891 = stablehlo.reshape %v5881 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5892 = stablehlo.reshape %v5890 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5893 = stablehlo.dot_general %v5891, %v5892, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v5894 = stablehlo.reshape %v5893 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5895 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v5896 = stablehlo.multiply %v5894, %v5895 : tensor<32x38809xf32>
    %v5897 = stablehlo.reshape %v5896 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5898 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5899 = stablehlo.exponential %v5897 : tensor<32x197x197xf32>
    %v5900 = stablehlo.reduce(%v5899 init: %v5898) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5901 = stablehlo.broadcast_in_dim %v5900, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5902 = stablehlo.divide %v5899, %v5901 : tensor<32x197x197xf32>
    %v5903 = stablehlo.reshape %v5902 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5904 = stablehlo.reshape %v5903 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5905 = stablehlo.reshape %v5887 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5906 = stablehlo.dot_general %v5904, %v5905, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5907 = stablehlo.reshape %v5906 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5908 = stablehlo.reshape %v5907 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5909 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5910 = stablehlo.pad %v5908, %v5909, low = [0, 0, 448], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v5911 = stablehlo.reshape %v5910 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5912 = stablehlo.add %v5878, %v5911 : tensor<32x151296xf32>
    %v5913 = stablehlo.reshape %v5631 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5914 = stablehlo.slice %v5913 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5915 = stablehlo.reshape %v5914 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5916 = stablehlo.reshape %v5636 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5917 = stablehlo.slice %v5916 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5918 = stablehlo.reshape %v5917 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5919 = stablehlo.reshape %v5641 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5920 = stablehlo.slice %v5919 [0:32, 0:197, 512:576] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5921 = stablehlo.reshape %v5920 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5922 = stablehlo.reshape %v5918 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5923 = stablehlo.transpose %v5922, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5924 = stablehlo.reshape %v5923 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5925 = stablehlo.reshape %v5915 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5926 = stablehlo.reshape %v5924 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5927 = stablehlo.dot_general %v5925, %v5926, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v5928 = stablehlo.reshape %v5927 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5929 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v5930 = stablehlo.multiply %v5928, %v5929 : tensor<32x38809xf32>
    %v5931 = stablehlo.reshape %v5930 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5932 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5933 = stablehlo.exponential %v5931 : tensor<32x197x197xf32>
    %v5934 = stablehlo.reduce(%v5933 init: %v5932) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5935 = stablehlo.broadcast_in_dim %v5934, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5936 = stablehlo.divide %v5933, %v5935 : tensor<32x197x197xf32>
    %v5937 = stablehlo.reshape %v5936 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5938 = stablehlo.reshape %v5937 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5939 = stablehlo.reshape %v5921 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5940 = stablehlo.dot_general %v5938, %v5939, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5941 = stablehlo.reshape %v5940 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5942 = stablehlo.reshape %v5941 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5943 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5944 = stablehlo.pad %v5942, %v5943, low = [0, 0, 512], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v5945 = stablehlo.reshape %v5944 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5946 = stablehlo.add %v5912, %v5945 : tensor<32x151296xf32>
    %v5947 = stablehlo.reshape %v5631 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5948 = stablehlo.slice %v5947 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5949 = stablehlo.reshape %v5948 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5950 = stablehlo.reshape %v5636 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5951 = stablehlo.slice %v5950 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5952 = stablehlo.reshape %v5951 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5953 = stablehlo.reshape %v5641 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5954 = stablehlo.slice %v5953 [0:32, 0:197, 576:640] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5955 = stablehlo.reshape %v5954 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5956 = stablehlo.reshape %v5952 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5957 = stablehlo.transpose %v5956, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5958 = stablehlo.reshape %v5957 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5959 = stablehlo.reshape %v5949 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5960 = stablehlo.reshape %v5958 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5961 = stablehlo.dot_general %v5959, %v5960, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v5962 = stablehlo.reshape %v5961 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5963 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v5964 = stablehlo.multiply %v5962, %v5963 : tensor<32x38809xf32>
    %v5965 = stablehlo.reshape %v5964 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5966 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5967 = stablehlo.exponential %v5965 : tensor<32x197x197xf32>
    %v5968 = stablehlo.reduce(%v5967 init: %v5966) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v5969 = stablehlo.broadcast_in_dim %v5968, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v5970 = stablehlo.divide %v5967, %v5969 : tensor<32x197x197xf32>
    %v5971 = stablehlo.reshape %v5970 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5972 = stablehlo.reshape %v5971 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v5973 = stablehlo.reshape %v5955 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5974 = stablehlo.dot_general %v5972, %v5973, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v5975 = stablehlo.reshape %v5974 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5976 = stablehlo.reshape %v5975 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5977 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5978 = stablehlo.pad %v5976, %v5977, low = [0, 0, 576], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v5979 = stablehlo.reshape %v5978 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v5980 = stablehlo.add %v5946, %v5979 : tensor<32x151296xf32>
    %v5981 = stablehlo.reshape %v5631 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5982 = stablehlo.slice %v5981 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5983 = stablehlo.reshape %v5982 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5984 = stablehlo.reshape %v5636 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5985 = stablehlo.slice %v5984 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5986 = stablehlo.reshape %v5985 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5987 = stablehlo.reshape %v5641 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v5988 = stablehlo.slice %v5987 [0:32, 0:197, 640:704] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v5989 = stablehlo.reshape %v5988 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v5990 = stablehlo.reshape %v5986 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5991 = stablehlo.transpose %v5990, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v5992 = stablehlo.reshape %v5991 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v5993 = stablehlo.reshape %v5983 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v5994 = stablehlo.reshape %v5992 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v5995 = stablehlo.dot_general %v5993, %v5994, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v5996 = stablehlo.reshape %v5995 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v5997 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v5998 = stablehlo.multiply %v5996, %v5997 : tensor<32x38809xf32>
    %v5999 = stablehlo.reshape %v5998 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v6000 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6001 = stablehlo.exponential %v5999 : tensor<32x197x197xf32>
    %v6002 = stablehlo.reduce(%v6001 init: %v6000) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6003 = stablehlo.broadcast_in_dim %v6002, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v6004 = stablehlo.divide %v6001, %v6003 : tensor<32x197x197xf32>
    %v6005 = stablehlo.reshape %v6004 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v6006 = stablehlo.reshape %v6005 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v6007 = stablehlo.reshape %v5989 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6008 = stablehlo.dot_general %v6006, %v6007, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v6009 = stablehlo.reshape %v6008 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6010 = stablehlo.reshape %v6009 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6011 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6012 = stablehlo.pad %v6010, %v6011, low = [0, 0, 640], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v6013 = stablehlo.reshape %v6012 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v6014 = stablehlo.add %v5980, %v6013 : tensor<32x151296xf32>
    %v6015 = stablehlo.reshape %v5631 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6016 = stablehlo.slice %v6015 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v6017 = stablehlo.reshape %v6016 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6018 = stablehlo.reshape %v5636 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6019 = stablehlo.slice %v6018 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v6020 = stablehlo.reshape %v6019 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6021 = stablehlo.reshape %v5641 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6022 = stablehlo.slice %v6021 [0:32, 0:197, 704:768] : (tensor<32x197x768xf32>) -> tensor<32x197x64xf32>
    %v6023 = stablehlo.reshape %v6022 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6024 = stablehlo.reshape %v6020 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6025 = stablehlo.transpose %v6024, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v6026 = stablehlo.reshape %v6025 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v6027 = stablehlo.reshape %v6017 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6028 = stablehlo.reshape %v6026 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v6029 = stablehlo.dot_general %v6027, %v6028, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v6030 = stablehlo.reshape %v6029 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v6031 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v6032 = stablehlo.multiply %v6030, %v6031 : tensor<32x38809xf32>
    %v6033 = stablehlo.reshape %v6032 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v6034 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6035 = stablehlo.exponential %v6033 : tensor<32x197x197xf32>
    %v6036 = stablehlo.reduce(%v6035 init: %v6034) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6037 = stablehlo.broadcast_in_dim %v6036, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v6038 = stablehlo.divide %v6035, %v6037 : tensor<32x197x197xf32>
    %v6039 = stablehlo.reshape %v6038 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v6040 = stablehlo.reshape %v6039 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v6041 = stablehlo.reshape %v6023 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6042 = stablehlo.dot_general %v6040, %v6041, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v6043 = stablehlo.reshape %v6042 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v6044 = stablehlo.reshape %v6043 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v6045 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6046 = stablehlo.pad %v6044, %v6045, low = [0, 0, 704], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x768xf32>
    %v6047 = stablehlo.reshape %v6046 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v6048 = stablehlo.add %v6014, %v6047 : tensor<32x151296xf32>
    %v6049 = stablehlo.reshape %v6048 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6050 = stablehlo.dot_general %v6049, %b11_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x768xf32>) -> tensor<32x197x768xf32>
    %v6051 = stablehlo.broadcast_in_dim %b11_bo, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v6052 = stablehlo.add %v6050, %v6051 : tensor<32x197x768xf32>
    %v6053 = stablehlo.reshape %v6052 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v6054 = stablehlo.add %v5598, %v6053 : tensor<32x151296xf32>
    %v6055 = stablehlo.reshape %v6054 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6056 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6057 = stablehlo.constant dense<768.0> : tensor<32x197x768xf32>
    %v6058 = stablehlo.constant dense<1.0e-5> : tensor<32x197x768xf32>
    %v6059 = stablehlo.reduce(%v6055 init: %v6056) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6060 = stablehlo.broadcast_in_dim %v6059, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v6061 = stablehlo.divide %v6060, %v6057 : tensor<32x197x768xf32>
    %v6062 = stablehlo.subtract %v6055, %v6061 : tensor<32x197x768xf32>
    %v6063 = stablehlo.multiply %v6062, %v6062 : tensor<32x197x768xf32>
    %v6064 = stablehlo.reduce(%v6063 init: %v6056) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6065 = stablehlo.broadcast_in_dim %v6064, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v6066 = stablehlo.divide %v6065, %v6057 : tensor<32x197x768xf32>
    %v6067 = stablehlo.add %v6066, %v6058 : tensor<32x197x768xf32>
    %v6068 = stablehlo.rsqrt %v6067 : tensor<32x197x768xf32>
    %v6069 = stablehlo.multiply %v6062, %v6068 : tensor<32x197x768xf32>
    %v6070 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v6071 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v6072 = stablehlo.multiply %v6069, %v6070 : tensor<32x197x768xf32>
    %v6073 = stablehlo.add %v6072, %v6071 : tensor<32x197x768xf32>
    %v6074 = stablehlo.reshape %v6073 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v6075 = stablehlo.reshape %v6074 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6076 = stablehlo.broadcast_in_dim %b11_g2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v6077 = stablehlo.multiply %v6075, %v6076 : tensor<32x197x768xf32>
    %v6078 = stablehlo.reshape %v6077 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v6079 = stablehlo.reshape %v6078 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6080 = stablehlo.broadcast_in_dim %b11_bt2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v6081 = stablehlo.add %v6079, %v6080 : tensor<32x197x768xf32>
    %v6082 = stablehlo.reshape %v6081 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v6083 = stablehlo.reshape %v6082 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6084 = stablehlo.dot_general %v6083, %b11_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x3072xf32>) -> tensor<32x197x3072xf32>
    %v6085 = stablehlo.broadcast_in_dim %b11_bfc1, dims = [2] : (tensor<3072xf32>) -> tensor<32x197x3072xf32>
    %v6086 = stablehlo.add %v6084, %v6085 : tensor<32x197x3072xf32>
    %v6087 = stablehlo.reshape %v6086 : (tensor<32x197x3072xf32>) -> tensor<32x605184xf32>
    %v6088 = stablehlo.multiply %v6087, %v6087 : tensor<32x605184xf32>
    %v6089 = stablehlo.multiply %v6088, %v6087 : tensor<32x605184xf32>
    %v6090 = stablehlo.constant dense<0.044715> : tensor<32x605184xf32>
    %v6091 = stablehlo.multiply %v6090, %v6089 : tensor<32x605184xf32>
    %v6092 = stablehlo.add %v6087, %v6091 : tensor<32x605184xf32>
    %v6093 = stablehlo.constant dense<0.7978845608028654> : tensor<32x605184xf32>
    %v6094 = stablehlo.multiply %v6093, %v6092 : tensor<32x605184xf32>
    %v6095 = stablehlo.tanh %v6094 : tensor<32x605184xf32>
    %v6096 = stablehlo.constant dense<1.0> : tensor<32x605184xf32>
    %v6097 = stablehlo.add %v6096, %v6095 : tensor<32x605184xf32>
    %v6098 = stablehlo.constant dense<0.5> : tensor<32x605184xf32>
    %v6099 = stablehlo.multiply %v6098, %v6087 : tensor<32x605184xf32>
    %v6100 = stablehlo.multiply %v6099, %v6097 : tensor<32x605184xf32>
    %v6101 = stablehlo.reshape %v6100 : (tensor<32x605184xf32>) -> tensor<32x197x3072xf32>
    %v6102 = stablehlo.dot_general %v6101, %b11_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x3072xf32>, tensor<3072x768xf32>) -> tensor<32x197x768xf32>
    %v6103 = stablehlo.broadcast_in_dim %b11_bfc2, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v6104 = stablehlo.add %v6102, %v6103 : tensor<32x197x768xf32>
    %v6105 = stablehlo.reshape %v6104 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v6106 = stablehlo.add %v6054, %v6105 : tensor<32x151296xf32>
    %v6107 = stablehlo.reshape %v6106 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6108 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6109 = stablehlo.constant dense<768.0> : tensor<32x197x768xf32>
    %v6110 = stablehlo.constant dense<1.0e-5> : tensor<32x197x768xf32>
    %v6111 = stablehlo.reduce(%v6107 init: %v6108) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6112 = stablehlo.broadcast_in_dim %v6111, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v6113 = stablehlo.divide %v6112, %v6109 : tensor<32x197x768xf32>
    %v6114 = stablehlo.subtract %v6107, %v6113 : tensor<32x197x768xf32>
    %v6115 = stablehlo.multiply %v6114, %v6114 : tensor<32x197x768xf32>
    %v6116 = stablehlo.reduce(%v6115 init: %v6108) applies stablehlo.add across dimensions = [2] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v6117 = stablehlo.broadcast_in_dim %v6116, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x768xf32>
    %v6118 = stablehlo.divide %v6117, %v6109 : tensor<32x197x768xf32>
    %v6119 = stablehlo.add %v6118, %v6110 : tensor<32x197x768xf32>
    %v6120 = stablehlo.rsqrt %v6119 : tensor<32x197x768xf32>
    %v6121 = stablehlo.multiply %v6114, %v6120 : tensor<32x197x768xf32>
    %v6122 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v6123 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x768xf32>
    %v6124 = stablehlo.multiply %v6121, %v6122 : tensor<32x197x768xf32>
    %v6125 = stablehlo.add %v6124, %v6123 : tensor<32x197x768xf32>
    %v6126 = stablehlo.reshape %v6125 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v6127 = stablehlo.reshape %v6126 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6128 = stablehlo.broadcast_in_dim %gF, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v6129 = stablehlo.multiply %v6127, %v6128 : tensor<32x197x768xf32>
    %v6130 = stablehlo.reshape %v6129 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v6131 = stablehlo.reshape %v6130 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6132 = stablehlo.broadcast_in_dim %btF, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v6133 = stablehlo.add %v6131, %v6132 : tensor<32x197x768xf32>
    %v6134 = stablehlo.reshape %v6133 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v6135 = stablehlo.reshape %v6134 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v6136 = stablehlo.slice %v6135 [0:32, 0:1, 0:768] : (tensor<32x197x768xf32>) -> tensor<32x1x768xf32>
    %v6137 = stablehlo.reshape %v6136 : (tensor<32x1x768xf32>) -> tensor<32x768xf32>
    %v6138 = stablehlo.dot_general %v6137, %Wc, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x768xf32>, tensor<768x1000xf32>) -> tensor<32x1000xf32>
    %v6139 = stablehlo.broadcast_in_dim %bc, dims = [1] : (tensor<1000xf32>) -> tensor<32x1000xf32>
    %v6140 = stablehlo.add %v6138, %v6139 : tensor<32x1000xf32>
    return %v6140 : tensor<32x1000xf32>
  }
}

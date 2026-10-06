module @m {
  func.func @vitsin_erf_fwd(%x: tensor<32x150528xf32>, %wConv: tensor<384x3x16x16xf32>, %bConv: tensor<384xf32>, %cls: tensor<384xf32>, %pos: tensor<197x384xf32>, %b0_g1: tensor<384xf32>, %b0_bt1: tensor<384xf32>, %b0_Wq: tensor<384x384xf32>, %b0_bq: tensor<384xf32>, %b0_Wk: tensor<384x384xf32>, %b0_bk: tensor<384xf32>, %b0_Wv: tensor<384x384xf32>, %b0_bv: tensor<384xf32>, %b0_Wo: tensor<384x384xf32>, %b0_bo: tensor<384xf32>, %b0_g2: tensor<384xf32>, %b0_bt2: tensor<384xf32>, %b0_Wfc1: tensor<384x1536xf32>, %b0_bfc1: tensor<1536xf32>, %b0_Wfc2: tensor<1536x384xf32>, %b0_bfc2: tensor<384xf32>, %b1_g1: tensor<384xf32>, %b1_bt1: tensor<384xf32>, %b1_Wq: tensor<384x384xf32>, %b1_bq: tensor<384xf32>, %b1_Wk: tensor<384x384xf32>, %b1_bk: tensor<384xf32>, %b1_Wv: tensor<384x384xf32>, %b1_bv: tensor<384xf32>, %b1_Wo: tensor<384x384xf32>, %b1_bo: tensor<384xf32>, %b1_g2: tensor<384xf32>, %b1_bt2: tensor<384xf32>, %b1_Wfc1: tensor<384x1536xf32>, %b1_bfc1: tensor<1536xf32>, %b1_Wfc2: tensor<1536x384xf32>, %b1_bfc2: tensor<384xf32>, %b2_g1: tensor<384xf32>, %b2_bt1: tensor<384xf32>, %b2_Wq: tensor<384x384xf32>, %b2_bq: tensor<384xf32>, %b2_Wk: tensor<384x384xf32>, %b2_bk: tensor<384xf32>, %b2_Wv: tensor<384x384xf32>, %b2_bv: tensor<384xf32>, %b2_Wo: tensor<384x384xf32>, %b2_bo: tensor<384xf32>, %b2_g2: tensor<384xf32>, %b2_bt2: tensor<384xf32>, %b2_Wfc1: tensor<384x1536xf32>, %b2_bfc1: tensor<1536xf32>, %b2_Wfc2: tensor<1536x384xf32>, %b2_bfc2: tensor<384xf32>, %b3_g1: tensor<384xf32>, %b3_bt1: tensor<384xf32>, %b3_Wq: tensor<384x384xf32>, %b3_bq: tensor<384xf32>, %b3_Wk: tensor<384x384xf32>, %b3_bk: tensor<384xf32>, %b3_Wv: tensor<384x384xf32>, %b3_bv: tensor<384xf32>, %b3_Wo: tensor<384x384xf32>, %b3_bo: tensor<384xf32>, %b3_g2: tensor<384xf32>, %b3_bt2: tensor<384xf32>, %b3_Wfc1: tensor<384x1536xf32>, %b3_bfc1: tensor<1536xf32>, %b3_Wfc2: tensor<1536x384xf32>, %b3_bfc2: tensor<384xf32>, %b4_g1: tensor<384xf32>, %b4_bt1: tensor<384xf32>, %b4_Wq: tensor<384x384xf32>, %b4_bq: tensor<384xf32>, %b4_Wk: tensor<384x384xf32>, %b4_bk: tensor<384xf32>, %b4_Wv: tensor<384x384xf32>, %b4_bv: tensor<384xf32>, %b4_Wo: tensor<384x384xf32>, %b4_bo: tensor<384xf32>, %b4_g2: tensor<384xf32>, %b4_bt2: tensor<384xf32>, %b4_Wfc1: tensor<384x1536xf32>, %b4_bfc1: tensor<1536xf32>, %b4_Wfc2: tensor<1536x384xf32>, %b4_bfc2: tensor<384xf32>, %b5_g1: tensor<384xf32>, %b5_bt1: tensor<384xf32>, %b5_Wq: tensor<384x384xf32>, %b5_bq: tensor<384xf32>, %b5_Wk: tensor<384x384xf32>, %b5_bk: tensor<384xf32>, %b5_Wv: tensor<384x384xf32>, %b5_bv: tensor<384xf32>, %b5_Wo: tensor<384x384xf32>, %b5_bo: tensor<384xf32>, %b5_g2: tensor<384xf32>, %b5_bt2: tensor<384xf32>, %b5_Wfc1: tensor<384x1536xf32>, %b5_bfc1: tensor<1536xf32>, %b5_Wfc2: tensor<1536x384xf32>, %b5_bfc2: tensor<384xf32>, %b6_g1: tensor<384xf32>, %b6_bt1: tensor<384xf32>, %b6_Wq: tensor<384x384xf32>, %b6_bq: tensor<384xf32>, %b6_Wk: tensor<384x384xf32>, %b6_bk: tensor<384xf32>, %b6_Wv: tensor<384x384xf32>, %b6_bv: tensor<384xf32>, %b6_Wo: tensor<384x384xf32>, %b6_bo: tensor<384xf32>, %b6_g2: tensor<384xf32>, %b6_bt2: tensor<384xf32>, %b6_Wfc1: tensor<384x1536xf32>, %b6_bfc1: tensor<1536xf32>, %b6_Wfc2: tensor<1536x384xf32>, %b6_bfc2: tensor<384xf32>, %b7_g1: tensor<384xf32>, %b7_bt1: tensor<384xf32>, %b7_Wq: tensor<384x384xf32>, %b7_bq: tensor<384xf32>, %b7_Wk: tensor<384x384xf32>, %b7_bk: tensor<384xf32>, %b7_Wv: tensor<384x384xf32>, %b7_bv: tensor<384xf32>, %b7_Wo: tensor<384x384xf32>, %b7_bo: tensor<384xf32>, %b7_g2: tensor<384xf32>, %b7_bt2: tensor<384xf32>, %b7_Wfc1: tensor<384x1536xf32>, %b7_bfc1: tensor<1536xf32>, %b7_Wfc2: tensor<1536x384xf32>, %b7_bfc2: tensor<384xf32>, %b8_g1: tensor<384xf32>, %b8_bt1: tensor<384xf32>, %b8_Wq: tensor<384x384xf32>, %b8_bq: tensor<384xf32>, %b8_Wk: tensor<384x384xf32>, %b8_bk: tensor<384xf32>, %b8_Wv: tensor<384x384xf32>, %b8_bv: tensor<384xf32>, %b8_Wo: tensor<384x384xf32>, %b8_bo: tensor<384xf32>, %b8_g2: tensor<384xf32>, %b8_bt2: tensor<384xf32>, %b8_Wfc1: tensor<384x1536xf32>, %b8_bfc1: tensor<1536xf32>, %b8_Wfc2: tensor<1536x384xf32>, %b8_bfc2: tensor<384xf32>, %b9_g1: tensor<384xf32>, %b9_bt1: tensor<384xf32>, %b9_Wq: tensor<384x384xf32>, %b9_bq: tensor<384xf32>, %b9_Wk: tensor<384x384xf32>, %b9_bk: tensor<384xf32>, %b9_Wv: tensor<384x384xf32>, %b9_bv: tensor<384xf32>, %b9_Wo: tensor<384x384xf32>, %b9_bo: tensor<384xf32>, %b9_g2: tensor<384xf32>, %b9_bt2: tensor<384xf32>, %b9_Wfc1: tensor<384x1536xf32>, %b9_bfc1: tensor<1536xf32>, %b9_Wfc2: tensor<1536x384xf32>, %b9_bfc2: tensor<384xf32>, %b10_g1: tensor<384xf32>, %b10_bt1: tensor<384xf32>, %b10_Wq: tensor<384x384xf32>, %b10_bq: tensor<384xf32>, %b10_Wk: tensor<384x384xf32>, %b10_bk: tensor<384xf32>, %b10_Wv: tensor<384x384xf32>, %b10_bv: tensor<384xf32>, %b10_Wo: tensor<384x384xf32>, %b10_bo: tensor<384xf32>, %b10_g2: tensor<384xf32>, %b10_bt2: tensor<384xf32>, %b10_Wfc1: tensor<384x1536xf32>, %b10_bfc1: tensor<1536xf32>, %b10_Wfc2: tensor<1536x384xf32>, %b10_bfc2: tensor<384xf32>, %b11_g1: tensor<384xf32>, %b11_bt1: tensor<384xf32>, %b11_Wq: tensor<384x384xf32>, %b11_bq: tensor<384xf32>, %b11_Wk: tensor<384x384xf32>, %b11_bk: tensor<384xf32>, %b11_Wv: tensor<384x384xf32>, %b11_bv: tensor<384xf32>, %b11_Wo: tensor<384x384xf32>, %b11_bo: tensor<384xf32>, %b11_g2: tensor<384xf32>, %b11_bt2: tensor<384xf32>, %b11_Wfc1: tensor<384x1536xf32>, %b11_bfc1: tensor<1536xf32>, %b11_Wfc2: tensor<1536x384xf32>, %b11_bfc2: tensor<384xf32>, %gF: tensor<384xf32>, %btF: tensor<384xf32>, %Wc: tensor<384x1000xf32>, %bc: tensor<1000xf32>) -> tensor<32x1000xf32> {
    %one = stablehlo.constant dense<1.0> : tensor<f32>
    %zero = stablehlo.constant dense<0.0> : tensor<f32>
    %sc = stablehlo.constant dense<0.0> : tensor<f32>
    %v0 = stablehlo.reshape %x : (tensor<32x150528xf32>) -> tensor<32x3x224x224xf32>
    %v1 = stablehlo.convolution(%v0, %wConv)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [16, 16], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3x224x224xf32>, tensor<384x3x16x16xf32>) -> tensor<32x384x14x14xf32>
    %v2 = stablehlo.broadcast_in_dim %bConv, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3 = stablehlo.add %v1, %v2 : tensor<32x384x14x14xf32>
    %v4 = stablehlo.transpose %v3, dims = [0, 2, 3, 1] : (tensor<32x384x14x14xf32>) -> tensor<32x14x14x384xf32>
    %v5 = stablehlo.reshape %v4 : (tensor<32x14x14x384xf32>) -> tensor<32x196x384xf32>
    %v6 = stablehlo.broadcast_in_dim %cls, dims = [2] : (tensor<384xf32>) -> tensor<32x1x384xf32>
    %v7 = stablehlo.concatenate %v6, %v5, dim = 1 : (tensor<32x1x384xf32>, tensor<32x196x384xf32>) -> tensor<32x197x384xf32>
    %v8 = stablehlo.broadcast_in_dim %pos, dims = [1, 2] : (tensor<197x384xf32>) -> tensor<32x197x384xf32>
    %v9 = stablehlo.add %v7, %v8 : tensor<32x197x384xf32>
    %v10 = stablehlo.reshape %v9 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v11 = stablehlo.reshape %v10 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v12 = stablehlo.constant dense<0.0> : tensor<f32>
    %v13 = stablehlo.constant dense<384.0> : tensor<32x197x384xf32>
    %v14 = stablehlo.constant dense<1.0e-5> : tensor<32x197x384xf32>
    %v15 = stablehlo.reduce(%v11 init: %v12) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v16 = stablehlo.broadcast_in_dim %v15, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v17 = stablehlo.divide %v16, %v13 : tensor<32x197x384xf32>
    %v18 = stablehlo.subtract %v11, %v17 : tensor<32x197x384xf32>
    %v19 = stablehlo.multiply %v18, %v18 : tensor<32x197x384xf32>
    %v20 = stablehlo.reduce(%v19 init: %v12) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v21 = stablehlo.broadcast_in_dim %v20, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v22 = stablehlo.divide %v21, %v13 : tensor<32x197x384xf32>
    %v23 = stablehlo.add %v22, %v14 : tensor<32x197x384xf32>
    %v24 = stablehlo.rsqrt %v23 : tensor<32x197x384xf32>
    %v25 = stablehlo.multiply %v18, %v24 : tensor<32x197x384xf32>
    %v26 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v27 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v28 = stablehlo.multiply %v25, %v26 : tensor<32x197x384xf32>
    %v29 = stablehlo.add %v28, %v27 : tensor<32x197x384xf32>
    %v30 = stablehlo.reshape %v29 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v31 = stablehlo.reshape %v30 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v32 = stablehlo.broadcast_in_dim %b0_g1, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v33 = stablehlo.multiply %v31, %v32 : tensor<32x197x384xf32>
    %v34 = stablehlo.reshape %v33 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v35 = stablehlo.reshape %v34 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v36 = stablehlo.broadcast_in_dim %b0_bt1, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v37 = stablehlo.add %v35, %v36 : tensor<32x197x384xf32>
    %v38 = stablehlo.reshape %v37 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v39 = stablehlo.reshape %v38 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v40 = stablehlo.dot_general %v39, %b0_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v41 = stablehlo.broadcast_in_dim %b0_bq, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v42 = stablehlo.add %v40, %v41 : tensor<32x197x384xf32>
    %v43 = stablehlo.reshape %v42 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v44 = stablehlo.reshape %v38 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v45 = stablehlo.dot_general %v44, %b0_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v46 = stablehlo.broadcast_in_dim %b0_bk, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v47 = stablehlo.add %v45, %v46 : tensor<32x197x384xf32>
    %v48 = stablehlo.reshape %v47 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v49 = stablehlo.reshape %v38 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v50 = stablehlo.dot_general %v49, %b0_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v51 = stablehlo.broadcast_in_dim %b0_bv, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v52 = stablehlo.add %v50, %v51 : tensor<32x197x384xf32>
    %v53 = stablehlo.reshape %v52 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v54 = stablehlo.reshape %v43 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v55 = stablehlo.slice %v54 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v56 = stablehlo.reshape %v55 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v57 = stablehlo.reshape %v48 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v58 = stablehlo.slice %v57 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v59 = stablehlo.reshape %v58 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v60 = stablehlo.reshape %v53 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v61 = stablehlo.slice %v60 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
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
    %v85 = stablehlo.pad %v83, %v84, low = [0, 0, 0], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v86 = stablehlo.reshape %v85 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v87 = stablehlo.reshape %v43 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v88 = stablehlo.slice %v87 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v89 = stablehlo.reshape %v88 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v90 = stablehlo.reshape %v48 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v91 = stablehlo.slice %v90 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v92 = stablehlo.reshape %v91 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v93 = stablehlo.reshape %v53 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v94 = stablehlo.slice %v93 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
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
    %v118 = stablehlo.pad %v116, %v117, low = [0, 0, 64], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v119 = stablehlo.reshape %v118 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v120 = stablehlo.add %v86, %v119 : tensor<32x75648xf32>
    %v121 = stablehlo.reshape %v43 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v122 = stablehlo.slice %v121 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v123 = stablehlo.reshape %v122 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v124 = stablehlo.reshape %v48 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v125 = stablehlo.slice %v124 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v126 = stablehlo.reshape %v125 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v127 = stablehlo.reshape %v53 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v128 = stablehlo.slice %v127 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
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
    %v152 = stablehlo.pad %v150, %v151, low = [0, 0, 128], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v153 = stablehlo.reshape %v152 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v154 = stablehlo.add %v120, %v153 : tensor<32x75648xf32>
    %v155 = stablehlo.reshape %v43 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v156 = stablehlo.slice %v155 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v157 = stablehlo.reshape %v156 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v158 = stablehlo.reshape %v48 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v159 = stablehlo.slice %v158 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v160 = stablehlo.reshape %v159 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v161 = stablehlo.reshape %v53 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v162 = stablehlo.slice %v161 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
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
    %v186 = stablehlo.pad %v184, %v185, low = [0, 0, 192], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v187 = stablehlo.reshape %v186 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v188 = stablehlo.add %v154, %v187 : tensor<32x75648xf32>
    %v189 = stablehlo.reshape %v43 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v190 = stablehlo.slice %v189 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v191 = stablehlo.reshape %v190 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v192 = stablehlo.reshape %v48 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v193 = stablehlo.slice %v192 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v194 = stablehlo.reshape %v193 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v195 = stablehlo.reshape %v53 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v196 = stablehlo.slice %v195 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
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
    %v220 = stablehlo.pad %v218, %v219, low = [0, 0, 256], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v221 = stablehlo.reshape %v220 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v222 = stablehlo.add %v188, %v221 : tensor<32x75648xf32>
    %v223 = stablehlo.reshape %v43 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v224 = stablehlo.slice %v223 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v225 = stablehlo.reshape %v224 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v226 = stablehlo.reshape %v48 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v227 = stablehlo.slice %v226 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v228 = stablehlo.reshape %v227 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v229 = stablehlo.reshape %v53 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v230 = stablehlo.slice %v229 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
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
    %v254 = stablehlo.pad %v252, %v253, low = [0, 0, 320], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v255 = stablehlo.reshape %v254 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v256 = stablehlo.add %v222, %v255 : tensor<32x75648xf32>
    %v257 = stablehlo.reshape %v256 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v258 = stablehlo.dot_general %v257, %b0_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v259 = stablehlo.broadcast_in_dim %b0_bo, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v260 = stablehlo.add %v258, %v259 : tensor<32x197x384xf32>
    %v261 = stablehlo.reshape %v260 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v262 = stablehlo.add %v10, %v261 : tensor<32x75648xf32>
    %v263 = stablehlo.reshape %v262 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v264 = stablehlo.constant dense<0.0> : tensor<f32>
    %v265 = stablehlo.constant dense<384.0> : tensor<32x197x384xf32>
    %v266 = stablehlo.constant dense<1.0e-5> : tensor<32x197x384xf32>
    %v267 = stablehlo.reduce(%v263 init: %v264) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v268 = stablehlo.broadcast_in_dim %v267, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v269 = stablehlo.divide %v268, %v265 : tensor<32x197x384xf32>
    %v270 = stablehlo.subtract %v263, %v269 : tensor<32x197x384xf32>
    %v271 = stablehlo.multiply %v270, %v270 : tensor<32x197x384xf32>
    %v272 = stablehlo.reduce(%v271 init: %v264) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v273 = stablehlo.broadcast_in_dim %v272, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v274 = stablehlo.divide %v273, %v265 : tensor<32x197x384xf32>
    %v275 = stablehlo.add %v274, %v266 : tensor<32x197x384xf32>
    %v276 = stablehlo.rsqrt %v275 : tensor<32x197x384xf32>
    %v277 = stablehlo.multiply %v270, %v276 : tensor<32x197x384xf32>
    %v278 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v279 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v280 = stablehlo.multiply %v277, %v278 : tensor<32x197x384xf32>
    %v281 = stablehlo.add %v280, %v279 : tensor<32x197x384xf32>
    %v282 = stablehlo.reshape %v281 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v283 = stablehlo.reshape %v282 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v284 = stablehlo.broadcast_in_dim %b0_g2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v285 = stablehlo.multiply %v283, %v284 : tensor<32x197x384xf32>
    %v286 = stablehlo.reshape %v285 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v287 = stablehlo.reshape %v286 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v288 = stablehlo.broadcast_in_dim %b0_bt2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v289 = stablehlo.add %v287, %v288 : tensor<32x197x384xf32>
    %v290 = stablehlo.reshape %v289 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v291 = stablehlo.reshape %v290 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v292 = stablehlo.dot_general %v291, %b0_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x1536xf32>) -> tensor<32x197x1536xf32>
    %v293 = stablehlo.broadcast_in_dim %b0_bfc1, dims = [2] : (tensor<1536xf32>) -> tensor<32x197x1536xf32>
    %v294 = stablehlo.add %v292, %v293 : tensor<32x197x1536xf32>
    %v295 = stablehlo.reshape %v294 : (tensor<32x197x1536xf32>) -> tensor<32x302592xf32>
    %v296 = stablehlo.constant dense<0.5> : tensor<32x302592xf32>
    %v297 = stablehlo.multiply %v296, %v295 : tensor<32x302592xf32>
    %v298 = stablehlo.negate %v295 : tensor<32x302592xf32>
    %v299 = stablehlo.constant dense<0.7071067811865476> : tensor<32x302592xf32>
    %v300 = stablehlo.multiply %v298, %v299 : tensor<32x302592xf32>
    %v301 = chlo.erfc %v300 : tensor<32x302592xf32> -> tensor<32x302592xf32>
    %v302 = stablehlo.multiply %v297, %v301 : tensor<32x302592xf32>
    %v303 = stablehlo.reshape %v302 : (tensor<32x302592xf32>) -> tensor<32x197x1536xf32>
    %v304 = stablehlo.dot_general %v303, %b0_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x1536xf32>, tensor<1536x384xf32>) -> tensor<32x197x384xf32>
    %v305 = stablehlo.broadcast_in_dim %b0_bfc2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v306 = stablehlo.add %v304, %v305 : tensor<32x197x384xf32>
    %v307 = stablehlo.reshape %v306 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v308 = stablehlo.add %v262, %v307 : tensor<32x75648xf32>
    %v309 = stablehlo.reshape %v308 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v310 = stablehlo.constant dense<0.0> : tensor<f32>
    %v311 = stablehlo.constant dense<384.0> : tensor<32x197x384xf32>
    %v312 = stablehlo.constant dense<1.0e-5> : tensor<32x197x384xf32>
    %v313 = stablehlo.reduce(%v309 init: %v310) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v314 = stablehlo.broadcast_in_dim %v313, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v315 = stablehlo.divide %v314, %v311 : tensor<32x197x384xf32>
    %v316 = stablehlo.subtract %v309, %v315 : tensor<32x197x384xf32>
    %v317 = stablehlo.multiply %v316, %v316 : tensor<32x197x384xf32>
    %v318 = stablehlo.reduce(%v317 init: %v310) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v319 = stablehlo.broadcast_in_dim %v318, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v320 = stablehlo.divide %v319, %v311 : tensor<32x197x384xf32>
    %v321 = stablehlo.add %v320, %v312 : tensor<32x197x384xf32>
    %v322 = stablehlo.rsqrt %v321 : tensor<32x197x384xf32>
    %v323 = stablehlo.multiply %v316, %v322 : tensor<32x197x384xf32>
    %v324 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v325 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v326 = stablehlo.multiply %v323, %v324 : tensor<32x197x384xf32>
    %v327 = stablehlo.add %v326, %v325 : tensor<32x197x384xf32>
    %v328 = stablehlo.reshape %v327 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v329 = stablehlo.reshape %v328 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v330 = stablehlo.broadcast_in_dim %b1_g1, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v331 = stablehlo.multiply %v329, %v330 : tensor<32x197x384xf32>
    %v332 = stablehlo.reshape %v331 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v333 = stablehlo.reshape %v332 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v334 = stablehlo.broadcast_in_dim %b1_bt1, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v335 = stablehlo.add %v333, %v334 : tensor<32x197x384xf32>
    %v336 = stablehlo.reshape %v335 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v337 = stablehlo.reshape %v336 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v338 = stablehlo.dot_general %v337, %b1_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v339 = stablehlo.broadcast_in_dim %b1_bq, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v340 = stablehlo.add %v338, %v339 : tensor<32x197x384xf32>
    %v341 = stablehlo.reshape %v340 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v342 = stablehlo.reshape %v336 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v343 = stablehlo.dot_general %v342, %b1_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v344 = stablehlo.broadcast_in_dim %b1_bk, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v345 = stablehlo.add %v343, %v344 : tensor<32x197x384xf32>
    %v346 = stablehlo.reshape %v345 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v347 = stablehlo.reshape %v336 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v348 = stablehlo.dot_general %v347, %b1_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v349 = stablehlo.broadcast_in_dim %b1_bv, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v350 = stablehlo.add %v348, %v349 : tensor<32x197x384xf32>
    %v351 = stablehlo.reshape %v350 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v352 = stablehlo.reshape %v341 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v353 = stablehlo.slice %v352 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v354 = stablehlo.reshape %v353 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v355 = stablehlo.reshape %v346 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v356 = stablehlo.slice %v355 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v357 = stablehlo.reshape %v356 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v358 = stablehlo.reshape %v351 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v359 = stablehlo.slice %v358 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v360 = stablehlo.reshape %v359 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v361 = stablehlo.reshape %v357 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v362 = stablehlo.transpose %v361, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v363 = stablehlo.reshape %v362 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v364 = stablehlo.reshape %v354 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v365 = stablehlo.reshape %v363 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v366 = stablehlo.dot_general %v364, %v365, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v367 = stablehlo.reshape %v366 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v368 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v369 = stablehlo.multiply %v367, %v368 : tensor<32x38809xf32>
    %v370 = stablehlo.reshape %v369 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v371 = stablehlo.constant dense<0.0> : tensor<f32>
    %v372 = stablehlo.exponential %v370 : tensor<32x197x197xf32>
    %v373 = stablehlo.reduce(%v372 init: %v371) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v374 = stablehlo.broadcast_in_dim %v373, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v375 = stablehlo.divide %v372, %v374 : tensor<32x197x197xf32>
    %v376 = stablehlo.reshape %v375 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v377 = stablehlo.reshape %v376 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v378 = stablehlo.reshape %v360 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v379 = stablehlo.dot_general %v377, %v378, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v380 = stablehlo.reshape %v379 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v381 = stablehlo.reshape %v380 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v382 = stablehlo.constant dense<0.0> : tensor<f32>
    %v383 = stablehlo.pad %v381, %v382, low = [0, 0, 0], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v384 = stablehlo.reshape %v383 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v385 = stablehlo.reshape %v341 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v386 = stablehlo.slice %v385 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v387 = stablehlo.reshape %v386 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v388 = stablehlo.reshape %v346 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v389 = stablehlo.slice %v388 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v390 = stablehlo.reshape %v389 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v391 = stablehlo.reshape %v351 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v392 = stablehlo.slice %v391 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v393 = stablehlo.reshape %v392 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v394 = stablehlo.reshape %v390 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v395 = stablehlo.transpose %v394, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v396 = stablehlo.reshape %v395 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v397 = stablehlo.reshape %v387 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v398 = stablehlo.reshape %v396 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v399 = stablehlo.dot_general %v397, %v398, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v400 = stablehlo.reshape %v399 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v401 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v402 = stablehlo.multiply %v400, %v401 : tensor<32x38809xf32>
    %v403 = stablehlo.reshape %v402 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v404 = stablehlo.constant dense<0.0> : tensor<f32>
    %v405 = stablehlo.exponential %v403 : tensor<32x197x197xf32>
    %v406 = stablehlo.reduce(%v405 init: %v404) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v407 = stablehlo.broadcast_in_dim %v406, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v408 = stablehlo.divide %v405, %v407 : tensor<32x197x197xf32>
    %v409 = stablehlo.reshape %v408 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v410 = stablehlo.reshape %v409 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v411 = stablehlo.reshape %v393 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v412 = stablehlo.dot_general %v410, %v411, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v413 = stablehlo.reshape %v412 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v414 = stablehlo.reshape %v413 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v415 = stablehlo.constant dense<0.0> : tensor<f32>
    %v416 = stablehlo.pad %v414, %v415, low = [0, 0, 64], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v417 = stablehlo.reshape %v416 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v418 = stablehlo.add %v384, %v417 : tensor<32x75648xf32>
    %v419 = stablehlo.reshape %v341 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v420 = stablehlo.slice %v419 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v421 = stablehlo.reshape %v420 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v422 = stablehlo.reshape %v346 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v423 = stablehlo.slice %v422 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v424 = stablehlo.reshape %v423 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v425 = stablehlo.reshape %v351 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v426 = stablehlo.slice %v425 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v427 = stablehlo.reshape %v426 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v428 = stablehlo.reshape %v424 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v429 = stablehlo.transpose %v428, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v430 = stablehlo.reshape %v429 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v431 = stablehlo.reshape %v421 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v432 = stablehlo.reshape %v430 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v433 = stablehlo.dot_general %v431, %v432, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v434 = stablehlo.reshape %v433 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v435 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v436 = stablehlo.multiply %v434, %v435 : tensor<32x38809xf32>
    %v437 = stablehlo.reshape %v436 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v438 = stablehlo.constant dense<0.0> : tensor<f32>
    %v439 = stablehlo.exponential %v437 : tensor<32x197x197xf32>
    %v440 = stablehlo.reduce(%v439 init: %v438) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v441 = stablehlo.broadcast_in_dim %v440, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v442 = stablehlo.divide %v439, %v441 : tensor<32x197x197xf32>
    %v443 = stablehlo.reshape %v442 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v444 = stablehlo.reshape %v443 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v445 = stablehlo.reshape %v427 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v446 = stablehlo.dot_general %v444, %v445, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v447 = stablehlo.reshape %v446 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v448 = stablehlo.reshape %v447 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v449 = stablehlo.constant dense<0.0> : tensor<f32>
    %v450 = stablehlo.pad %v448, %v449, low = [0, 0, 128], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v451 = stablehlo.reshape %v450 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v452 = stablehlo.add %v418, %v451 : tensor<32x75648xf32>
    %v453 = stablehlo.reshape %v341 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v454 = stablehlo.slice %v453 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v455 = stablehlo.reshape %v454 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v456 = stablehlo.reshape %v346 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v457 = stablehlo.slice %v456 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v458 = stablehlo.reshape %v457 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v459 = stablehlo.reshape %v351 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v460 = stablehlo.slice %v459 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v461 = stablehlo.reshape %v460 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v462 = stablehlo.reshape %v458 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v463 = stablehlo.transpose %v462, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v464 = stablehlo.reshape %v463 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v465 = stablehlo.reshape %v455 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v466 = stablehlo.reshape %v464 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v467 = stablehlo.dot_general %v465, %v466, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v468 = stablehlo.reshape %v467 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v469 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v470 = stablehlo.multiply %v468, %v469 : tensor<32x38809xf32>
    %v471 = stablehlo.reshape %v470 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v472 = stablehlo.constant dense<0.0> : tensor<f32>
    %v473 = stablehlo.exponential %v471 : tensor<32x197x197xf32>
    %v474 = stablehlo.reduce(%v473 init: %v472) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v475 = stablehlo.broadcast_in_dim %v474, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v476 = stablehlo.divide %v473, %v475 : tensor<32x197x197xf32>
    %v477 = stablehlo.reshape %v476 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v478 = stablehlo.reshape %v477 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v479 = stablehlo.reshape %v461 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v480 = stablehlo.dot_general %v478, %v479, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v481 = stablehlo.reshape %v480 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v482 = stablehlo.reshape %v481 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v483 = stablehlo.constant dense<0.0> : tensor<f32>
    %v484 = stablehlo.pad %v482, %v483, low = [0, 0, 192], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v485 = stablehlo.reshape %v484 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v486 = stablehlo.add %v452, %v485 : tensor<32x75648xf32>
    %v487 = stablehlo.reshape %v341 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v488 = stablehlo.slice %v487 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v489 = stablehlo.reshape %v488 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v490 = stablehlo.reshape %v346 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v491 = stablehlo.slice %v490 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v492 = stablehlo.reshape %v491 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v493 = stablehlo.reshape %v351 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v494 = stablehlo.slice %v493 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v495 = stablehlo.reshape %v494 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v496 = stablehlo.reshape %v492 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v497 = stablehlo.transpose %v496, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v498 = stablehlo.reshape %v497 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v499 = stablehlo.reshape %v489 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v500 = stablehlo.reshape %v498 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v501 = stablehlo.dot_general %v499, %v500, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v502 = stablehlo.reshape %v501 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v503 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v504 = stablehlo.multiply %v502, %v503 : tensor<32x38809xf32>
    %v505 = stablehlo.reshape %v504 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v506 = stablehlo.constant dense<0.0> : tensor<f32>
    %v507 = stablehlo.exponential %v505 : tensor<32x197x197xf32>
    %v508 = stablehlo.reduce(%v507 init: %v506) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v509 = stablehlo.broadcast_in_dim %v508, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v510 = stablehlo.divide %v507, %v509 : tensor<32x197x197xf32>
    %v511 = stablehlo.reshape %v510 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v512 = stablehlo.reshape %v511 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v513 = stablehlo.reshape %v495 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v514 = stablehlo.dot_general %v512, %v513, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v515 = stablehlo.reshape %v514 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v516 = stablehlo.reshape %v515 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v517 = stablehlo.constant dense<0.0> : tensor<f32>
    %v518 = stablehlo.pad %v516, %v517, low = [0, 0, 256], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v519 = stablehlo.reshape %v518 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v520 = stablehlo.add %v486, %v519 : tensor<32x75648xf32>
    %v521 = stablehlo.reshape %v341 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v522 = stablehlo.slice %v521 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v523 = stablehlo.reshape %v522 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v524 = stablehlo.reshape %v346 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v525 = stablehlo.slice %v524 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v526 = stablehlo.reshape %v525 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v527 = stablehlo.reshape %v351 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v528 = stablehlo.slice %v527 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v529 = stablehlo.reshape %v528 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v530 = stablehlo.reshape %v526 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v531 = stablehlo.transpose %v530, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v532 = stablehlo.reshape %v531 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v533 = stablehlo.reshape %v523 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v534 = stablehlo.reshape %v532 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v535 = stablehlo.dot_general %v533, %v534, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v536 = stablehlo.reshape %v535 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v537 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v538 = stablehlo.multiply %v536, %v537 : tensor<32x38809xf32>
    %v539 = stablehlo.reshape %v538 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v540 = stablehlo.constant dense<0.0> : tensor<f32>
    %v541 = stablehlo.exponential %v539 : tensor<32x197x197xf32>
    %v542 = stablehlo.reduce(%v541 init: %v540) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v543 = stablehlo.broadcast_in_dim %v542, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v544 = stablehlo.divide %v541, %v543 : tensor<32x197x197xf32>
    %v545 = stablehlo.reshape %v544 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v546 = stablehlo.reshape %v545 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v547 = stablehlo.reshape %v529 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v548 = stablehlo.dot_general %v546, %v547, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v549 = stablehlo.reshape %v548 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v550 = stablehlo.reshape %v549 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v551 = stablehlo.constant dense<0.0> : tensor<f32>
    %v552 = stablehlo.pad %v550, %v551, low = [0, 0, 320], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v553 = stablehlo.reshape %v552 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v554 = stablehlo.add %v520, %v553 : tensor<32x75648xf32>
    %v555 = stablehlo.reshape %v554 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v556 = stablehlo.dot_general %v555, %b1_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v557 = stablehlo.broadcast_in_dim %b1_bo, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v558 = stablehlo.add %v556, %v557 : tensor<32x197x384xf32>
    %v559 = stablehlo.reshape %v558 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v560 = stablehlo.add %v308, %v559 : tensor<32x75648xf32>
    %v561 = stablehlo.reshape %v560 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v562 = stablehlo.constant dense<0.0> : tensor<f32>
    %v563 = stablehlo.constant dense<384.0> : tensor<32x197x384xf32>
    %v564 = stablehlo.constant dense<1.0e-5> : tensor<32x197x384xf32>
    %v565 = stablehlo.reduce(%v561 init: %v562) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v566 = stablehlo.broadcast_in_dim %v565, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v567 = stablehlo.divide %v566, %v563 : tensor<32x197x384xf32>
    %v568 = stablehlo.subtract %v561, %v567 : tensor<32x197x384xf32>
    %v569 = stablehlo.multiply %v568, %v568 : tensor<32x197x384xf32>
    %v570 = stablehlo.reduce(%v569 init: %v562) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v571 = stablehlo.broadcast_in_dim %v570, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v572 = stablehlo.divide %v571, %v563 : tensor<32x197x384xf32>
    %v573 = stablehlo.add %v572, %v564 : tensor<32x197x384xf32>
    %v574 = stablehlo.rsqrt %v573 : tensor<32x197x384xf32>
    %v575 = stablehlo.multiply %v568, %v574 : tensor<32x197x384xf32>
    %v576 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v577 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v578 = stablehlo.multiply %v575, %v576 : tensor<32x197x384xf32>
    %v579 = stablehlo.add %v578, %v577 : tensor<32x197x384xf32>
    %v580 = stablehlo.reshape %v579 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v581 = stablehlo.reshape %v580 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v582 = stablehlo.broadcast_in_dim %b1_g2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v583 = stablehlo.multiply %v581, %v582 : tensor<32x197x384xf32>
    %v584 = stablehlo.reshape %v583 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v585 = stablehlo.reshape %v584 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v586 = stablehlo.broadcast_in_dim %b1_bt2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v587 = stablehlo.add %v585, %v586 : tensor<32x197x384xf32>
    %v588 = stablehlo.reshape %v587 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v589 = stablehlo.reshape %v588 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v590 = stablehlo.dot_general %v589, %b1_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x1536xf32>) -> tensor<32x197x1536xf32>
    %v591 = stablehlo.broadcast_in_dim %b1_bfc1, dims = [2] : (tensor<1536xf32>) -> tensor<32x197x1536xf32>
    %v592 = stablehlo.add %v590, %v591 : tensor<32x197x1536xf32>
    %v593 = stablehlo.reshape %v592 : (tensor<32x197x1536xf32>) -> tensor<32x302592xf32>
    %v594 = stablehlo.constant dense<0.5> : tensor<32x302592xf32>
    %v595 = stablehlo.multiply %v594, %v593 : tensor<32x302592xf32>
    %v596 = stablehlo.negate %v593 : tensor<32x302592xf32>
    %v597 = stablehlo.constant dense<0.7071067811865476> : tensor<32x302592xf32>
    %v598 = stablehlo.multiply %v596, %v597 : tensor<32x302592xf32>
    %v599 = chlo.erfc %v598 : tensor<32x302592xf32> -> tensor<32x302592xf32>
    %v600 = stablehlo.multiply %v595, %v599 : tensor<32x302592xf32>
    %v601 = stablehlo.reshape %v600 : (tensor<32x302592xf32>) -> tensor<32x197x1536xf32>
    %v602 = stablehlo.dot_general %v601, %b1_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x1536xf32>, tensor<1536x384xf32>) -> tensor<32x197x384xf32>
    %v603 = stablehlo.broadcast_in_dim %b1_bfc2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v604 = stablehlo.add %v602, %v603 : tensor<32x197x384xf32>
    %v605 = stablehlo.reshape %v604 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v606 = stablehlo.add %v560, %v605 : tensor<32x75648xf32>
    %v607 = stablehlo.reshape %v606 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v608 = stablehlo.constant dense<0.0> : tensor<f32>
    %v609 = stablehlo.constant dense<384.0> : tensor<32x197x384xf32>
    %v610 = stablehlo.constant dense<1.0e-5> : tensor<32x197x384xf32>
    %v611 = stablehlo.reduce(%v607 init: %v608) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v612 = stablehlo.broadcast_in_dim %v611, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v613 = stablehlo.divide %v612, %v609 : tensor<32x197x384xf32>
    %v614 = stablehlo.subtract %v607, %v613 : tensor<32x197x384xf32>
    %v615 = stablehlo.multiply %v614, %v614 : tensor<32x197x384xf32>
    %v616 = stablehlo.reduce(%v615 init: %v608) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v617 = stablehlo.broadcast_in_dim %v616, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v618 = stablehlo.divide %v617, %v609 : tensor<32x197x384xf32>
    %v619 = stablehlo.add %v618, %v610 : tensor<32x197x384xf32>
    %v620 = stablehlo.rsqrt %v619 : tensor<32x197x384xf32>
    %v621 = stablehlo.multiply %v614, %v620 : tensor<32x197x384xf32>
    %v622 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v623 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v624 = stablehlo.multiply %v621, %v622 : tensor<32x197x384xf32>
    %v625 = stablehlo.add %v624, %v623 : tensor<32x197x384xf32>
    %v626 = stablehlo.reshape %v625 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v627 = stablehlo.reshape %v626 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v628 = stablehlo.broadcast_in_dim %b2_g1, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v629 = stablehlo.multiply %v627, %v628 : tensor<32x197x384xf32>
    %v630 = stablehlo.reshape %v629 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v631 = stablehlo.reshape %v630 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v632 = stablehlo.broadcast_in_dim %b2_bt1, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v633 = stablehlo.add %v631, %v632 : tensor<32x197x384xf32>
    %v634 = stablehlo.reshape %v633 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v635 = stablehlo.reshape %v634 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v636 = stablehlo.dot_general %v635, %b2_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v637 = stablehlo.broadcast_in_dim %b2_bq, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v638 = stablehlo.add %v636, %v637 : tensor<32x197x384xf32>
    %v639 = stablehlo.reshape %v638 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v640 = stablehlo.reshape %v634 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v641 = stablehlo.dot_general %v640, %b2_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v642 = stablehlo.broadcast_in_dim %b2_bk, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v643 = stablehlo.add %v641, %v642 : tensor<32x197x384xf32>
    %v644 = stablehlo.reshape %v643 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v645 = stablehlo.reshape %v634 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v646 = stablehlo.dot_general %v645, %b2_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v647 = stablehlo.broadcast_in_dim %b2_bv, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v648 = stablehlo.add %v646, %v647 : tensor<32x197x384xf32>
    %v649 = stablehlo.reshape %v648 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v650 = stablehlo.reshape %v639 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v651 = stablehlo.slice %v650 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v652 = stablehlo.reshape %v651 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v653 = stablehlo.reshape %v644 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v654 = stablehlo.slice %v653 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v655 = stablehlo.reshape %v654 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v656 = stablehlo.reshape %v649 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v657 = stablehlo.slice %v656 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v658 = stablehlo.reshape %v657 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v659 = stablehlo.reshape %v655 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v660 = stablehlo.transpose %v659, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v661 = stablehlo.reshape %v660 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v662 = stablehlo.reshape %v652 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v663 = stablehlo.reshape %v661 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v664 = stablehlo.dot_general %v662, %v663, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v665 = stablehlo.reshape %v664 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v666 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v667 = stablehlo.multiply %v665, %v666 : tensor<32x38809xf32>
    %v668 = stablehlo.reshape %v667 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v669 = stablehlo.constant dense<0.0> : tensor<f32>
    %v670 = stablehlo.exponential %v668 : tensor<32x197x197xf32>
    %v671 = stablehlo.reduce(%v670 init: %v669) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v672 = stablehlo.broadcast_in_dim %v671, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v673 = stablehlo.divide %v670, %v672 : tensor<32x197x197xf32>
    %v674 = stablehlo.reshape %v673 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v675 = stablehlo.reshape %v674 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v676 = stablehlo.reshape %v658 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v677 = stablehlo.dot_general %v675, %v676, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v678 = stablehlo.reshape %v677 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v679 = stablehlo.reshape %v678 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v680 = stablehlo.constant dense<0.0> : tensor<f32>
    %v681 = stablehlo.pad %v679, %v680, low = [0, 0, 0], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v682 = stablehlo.reshape %v681 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v683 = stablehlo.reshape %v639 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v684 = stablehlo.slice %v683 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v685 = stablehlo.reshape %v684 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v686 = stablehlo.reshape %v644 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v687 = stablehlo.slice %v686 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v688 = stablehlo.reshape %v687 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v689 = stablehlo.reshape %v649 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v690 = stablehlo.slice %v689 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v691 = stablehlo.reshape %v690 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v692 = stablehlo.reshape %v688 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v693 = stablehlo.transpose %v692, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v694 = stablehlo.reshape %v693 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v695 = stablehlo.reshape %v685 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v696 = stablehlo.reshape %v694 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v697 = stablehlo.dot_general %v695, %v696, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v698 = stablehlo.reshape %v697 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v699 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v700 = stablehlo.multiply %v698, %v699 : tensor<32x38809xf32>
    %v701 = stablehlo.reshape %v700 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v702 = stablehlo.constant dense<0.0> : tensor<f32>
    %v703 = stablehlo.exponential %v701 : tensor<32x197x197xf32>
    %v704 = stablehlo.reduce(%v703 init: %v702) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v705 = stablehlo.broadcast_in_dim %v704, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v706 = stablehlo.divide %v703, %v705 : tensor<32x197x197xf32>
    %v707 = stablehlo.reshape %v706 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v708 = stablehlo.reshape %v707 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v709 = stablehlo.reshape %v691 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v710 = stablehlo.dot_general %v708, %v709, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v711 = stablehlo.reshape %v710 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v712 = stablehlo.reshape %v711 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v713 = stablehlo.constant dense<0.0> : tensor<f32>
    %v714 = stablehlo.pad %v712, %v713, low = [0, 0, 64], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v715 = stablehlo.reshape %v714 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v716 = stablehlo.add %v682, %v715 : tensor<32x75648xf32>
    %v717 = stablehlo.reshape %v639 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v718 = stablehlo.slice %v717 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v719 = stablehlo.reshape %v718 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v720 = stablehlo.reshape %v644 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v721 = stablehlo.slice %v720 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v722 = stablehlo.reshape %v721 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v723 = stablehlo.reshape %v649 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v724 = stablehlo.slice %v723 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
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
    %v736 = stablehlo.constant dense<0.0> : tensor<f32>
    %v737 = stablehlo.exponential %v735 : tensor<32x197x197xf32>
    %v738 = stablehlo.reduce(%v737 init: %v736) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v739 = stablehlo.broadcast_in_dim %v738, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v740 = stablehlo.divide %v737, %v739 : tensor<32x197x197xf32>
    %v741 = stablehlo.reshape %v740 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v742 = stablehlo.reshape %v741 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v743 = stablehlo.reshape %v725 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v744 = stablehlo.dot_general %v742, %v743, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v745 = stablehlo.reshape %v744 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v746 = stablehlo.reshape %v745 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v747 = stablehlo.constant dense<0.0> : tensor<f32>
    %v748 = stablehlo.pad %v746, %v747, low = [0, 0, 128], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v749 = stablehlo.reshape %v748 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v750 = stablehlo.add %v716, %v749 : tensor<32x75648xf32>
    %v751 = stablehlo.reshape %v639 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v752 = stablehlo.slice %v751 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v753 = stablehlo.reshape %v752 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v754 = stablehlo.reshape %v644 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v755 = stablehlo.slice %v754 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v756 = stablehlo.reshape %v755 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v757 = stablehlo.reshape %v649 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v758 = stablehlo.slice %v757 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v759 = stablehlo.reshape %v758 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v760 = stablehlo.reshape %v756 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v761 = stablehlo.transpose %v760, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v762 = stablehlo.reshape %v761 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v763 = stablehlo.reshape %v753 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v764 = stablehlo.reshape %v762 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v765 = stablehlo.dot_general %v763, %v764, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v766 = stablehlo.reshape %v765 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v767 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v768 = stablehlo.multiply %v766, %v767 : tensor<32x38809xf32>
    %v769 = stablehlo.reshape %v768 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v770 = stablehlo.constant dense<0.0> : tensor<f32>
    %v771 = stablehlo.exponential %v769 : tensor<32x197x197xf32>
    %v772 = stablehlo.reduce(%v771 init: %v770) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v773 = stablehlo.broadcast_in_dim %v772, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v774 = stablehlo.divide %v771, %v773 : tensor<32x197x197xf32>
    %v775 = stablehlo.reshape %v774 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v776 = stablehlo.reshape %v775 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v777 = stablehlo.reshape %v759 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v778 = stablehlo.dot_general %v776, %v777, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v779 = stablehlo.reshape %v778 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v780 = stablehlo.reshape %v779 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v781 = stablehlo.constant dense<0.0> : tensor<f32>
    %v782 = stablehlo.pad %v780, %v781, low = [0, 0, 192], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v783 = stablehlo.reshape %v782 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v784 = stablehlo.add %v750, %v783 : tensor<32x75648xf32>
    %v785 = stablehlo.reshape %v639 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v786 = stablehlo.slice %v785 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v787 = stablehlo.reshape %v786 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v788 = stablehlo.reshape %v644 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v789 = stablehlo.slice %v788 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v790 = stablehlo.reshape %v789 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v791 = stablehlo.reshape %v649 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v792 = stablehlo.slice %v791 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v793 = stablehlo.reshape %v792 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v794 = stablehlo.reshape %v790 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v795 = stablehlo.transpose %v794, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v796 = stablehlo.reshape %v795 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v797 = stablehlo.reshape %v787 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v798 = stablehlo.reshape %v796 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v799 = stablehlo.dot_general %v797, %v798, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v800 = stablehlo.reshape %v799 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v801 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v802 = stablehlo.multiply %v800, %v801 : tensor<32x38809xf32>
    %v803 = stablehlo.reshape %v802 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v804 = stablehlo.constant dense<0.0> : tensor<f32>
    %v805 = stablehlo.exponential %v803 : tensor<32x197x197xf32>
    %v806 = stablehlo.reduce(%v805 init: %v804) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v807 = stablehlo.broadcast_in_dim %v806, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v808 = stablehlo.divide %v805, %v807 : tensor<32x197x197xf32>
    %v809 = stablehlo.reshape %v808 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v810 = stablehlo.reshape %v809 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v811 = stablehlo.reshape %v793 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v812 = stablehlo.dot_general %v810, %v811, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v813 = stablehlo.reshape %v812 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v814 = stablehlo.reshape %v813 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v815 = stablehlo.constant dense<0.0> : tensor<f32>
    %v816 = stablehlo.pad %v814, %v815, low = [0, 0, 256], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v817 = stablehlo.reshape %v816 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v818 = stablehlo.add %v784, %v817 : tensor<32x75648xf32>
    %v819 = stablehlo.reshape %v639 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v820 = stablehlo.slice %v819 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v821 = stablehlo.reshape %v820 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v822 = stablehlo.reshape %v644 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v823 = stablehlo.slice %v822 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v824 = stablehlo.reshape %v823 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v825 = stablehlo.reshape %v649 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v826 = stablehlo.slice %v825 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v827 = stablehlo.reshape %v826 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v828 = stablehlo.reshape %v824 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v829 = stablehlo.transpose %v828, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v830 = stablehlo.reshape %v829 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v831 = stablehlo.reshape %v821 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v832 = stablehlo.reshape %v830 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v833 = stablehlo.dot_general %v831, %v832, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v834 = stablehlo.reshape %v833 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v835 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v836 = stablehlo.multiply %v834, %v835 : tensor<32x38809xf32>
    %v837 = stablehlo.reshape %v836 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v838 = stablehlo.constant dense<0.0> : tensor<f32>
    %v839 = stablehlo.exponential %v837 : tensor<32x197x197xf32>
    %v840 = stablehlo.reduce(%v839 init: %v838) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v841 = stablehlo.broadcast_in_dim %v840, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v842 = stablehlo.divide %v839, %v841 : tensor<32x197x197xf32>
    %v843 = stablehlo.reshape %v842 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v844 = stablehlo.reshape %v843 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v845 = stablehlo.reshape %v827 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v846 = stablehlo.dot_general %v844, %v845, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v847 = stablehlo.reshape %v846 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v848 = stablehlo.reshape %v847 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v849 = stablehlo.constant dense<0.0> : tensor<f32>
    %v850 = stablehlo.pad %v848, %v849, low = [0, 0, 320], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v851 = stablehlo.reshape %v850 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v852 = stablehlo.add %v818, %v851 : tensor<32x75648xf32>
    %v853 = stablehlo.reshape %v852 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v854 = stablehlo.dot_general %v853, %b2_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v855 = stablehlo.broadcast_in_dim %b2_bo, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v856 = stablehlo.add %v854, %v855 : tensor<32x197x384xf32>
    %v857 = stablehlo.reshape %v856 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v858 = stablehlo.add %v606, %v857 : tensor<32x75648xf32>
    %v859 = stablehlo.reshape %v858 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v860 = stablehlo.constant dense<0.0> : tensor<f32>
    %v861 = stablehlo.constant dense<384.0> : tensor<32x197x384xf32>
    %v862 = stablehlo.constant dense<1.0e-5> : tensor<32x197x384xf32>
    %v863 = stablehlo.reduce(%v859 init: %v860) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v864 = stablehlo.broadcast_in_dim %v863, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v865 = stablehlo.divide %v864, %v861 : tensor<32x197x384xf32>
    %v866 = stablehlo.subtract %v859, %v865 : tensor<32x197x384xf32>
    %v867 = stablehlo.multiply %v866, %v866 : tensor<32x197x384xf32>
    %v868 = stablehlo.reduce(%v867 init: %v860) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v869 = stablehlo.broadcast_in_dim %v868, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v870 = stablehlo.divide %v869, %v861 : tensor<32x197x384xf32>
    %v871 = stablehlo.add %v870, %v862 : tensor<32x197x384xf32>
    %v872 = stablehlo.rsqrt %v871 : tensor<32x197x384xf32>
    %v873 = stablehlo.multiply %v866, %v872 : tensor<32x197x384xf32>
    %v874 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v875 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v876 = stablehlo.multiply %v873, %v874 : tensor<32x197x384xf32>
    %v877 = stablehlo.add %v876, %v875 : tensor<32x197x384xf32>
    %v878 = stablehlo.reshape %v877 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v879 = stablehlo.reshape %v878 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v880 = stablehlo.broadcast_in_dim %b2_g2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v881 = stablehlo.multiply %v879, %v880 : tensor<32x197x384xf32>
    %v882 = stablehlo.reshape %v881 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v883 = stablehlo.reshape %v882 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v884 = stablehlo.broadcast_in_dim %b2_bt2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v885 = stablehlo.add %v883, %v884 : tensor<32x197x384xf32>
    %v886 = stablehlo.reshape %v885 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v887 = stablehlo.reshape %v886 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v888 = stablehlo.dot_general %v887, %b2_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x1536xf32>) -> tensor<32x197x1536xf32>
    %v889 = stablehlo.broadcast_in_dim %b2_bfc1, dims = [2] : (tensor<1536xf32>) -> tensor<32x197x1536xf32>
    %v890 = stablehlo.add %v888, %v889 : tensor<32x197x1536xf32>
    %v891 = stablehlo.reshape %v890 : (tensor<32x197x1536xf32>) -> tensor<32x302592xf32>
    %v892 = stablehlo.constant dense<0.5> : tensor<32x302592xf32>
    %v893 = stablehlo.multiply %v892, %v891 : tensor<32x302592xf32>
    %v894 = stablehlo.negate %v891 : tensor<32x302592xf32>
    %v895 = stablehlo.constant dense<0.7071067811865476> : tensor<32x302592xf32>
    %v896 = stablehlo.multiply %v894, %v895 : tensor<32x302592xf32>
    %v897 = chlo.erfc %v896 : tensor<32x302592xf32> -> tensor<32x302592xf32>
    %v898 = stablehlo.multiply %v893, %v897 : tensor<32x302592xf32>
    %v899 = stablehlo.reshape %v898 : (tensor<32x302592xf32>) -> tensor<32x197x1536xf32>
    %v900 = stablehlo.dot_general %v899, %b2_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x1536xf32>, tensor<1536x384xf32>) -> tensor<32x197x384xf32>
    %v901 = stablehlo.broadcast_in_dim %b2_bfc2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v902 = stablehlo.add %v900, %v901 : tensor<32x197x384xf32>
    %v903 = stablehlo.reshape %v902 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v904 = stablehlo.add %v858, %v903 : tensor<32x75648xf32>
    %v905 = stablehlo.reshape %v904 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v906 = stablehlo.constant dense<0.0> : tensor<f32>
    %v907 = stablehlo.constant dense<384.0> : tensor<32x197x384xf32>
    %v908 = stablehlo.constant dense<1.0e-5> : tensor<32x197x384xf32>
    %v909 = stablehlo.reduce(%v905 init: %v906) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v910 = stablehlo.broadcast_in_dim %v909, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v911 = stablehlo.divide %v910, %v907 : tensor<32x197x384xf32>
    %v912 = stablehlo.subtract %v905, %v911 : tensor<32x197x384xf32>
    %v913 = stablehlo.multiply %v912, %v912 : tensor<32x197x384xf32>
    %v914 = stablehlo.reduce(%v913 init: %v906) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v915 = stablehlo.broadcast_in_dim %v914, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v916 = stablehlo.divide %v915, %v907 : tensor<32x197x384xf32>
    %v917 = stablehlo.add %v916, %v908 : tensor<32x197x384xf32>
    %v918 = stablehlo.rsqrt %v917 : tensor<32x197x384xf32>
    %v919 = stablehlo.multiply %v912, %v918 : tensor<32x197x384xf32>
    %v920 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v921 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v922 = stablehlo.multiply %v919, %v920 : tensor<32x197x384xf32>
    %v923 = stablehlo.add %v922, %v921 : tensor<32x197x384xf32>
    %v924 = stablehlo.reshape %v923 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v925 = stablehlo.reshape %v924 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v926 = stablehlo.broadcast_in_dim %b3_g1, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v927 = stablehlo.multiply %v925, %v926 : tensor<32x197x384xf32>
    %v928 = stablehlo.reshape %v927 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v929 = stablehlo.reshape %v928 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v930 = stablehlo.broadcast_in_dim %b3_bt1, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v931 = stablehlo.add %v929, %v930 : tensor<32x197x384xf32>
    %v932 = stablehlo.reshape %v931 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v933 = stablehlo.reshape %v932 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v934 = stablehlo.dot_general %v933, %b3_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v935 = stablehlo.broadcast_in_dim %b3_bq, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v936 = stablehlo.add %v934, %v935 : tensor<32x197x384xf32>
    %v937 = stablehlo.reshape %v936 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v938 = stablehlo.reshape %v932 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v939 = stablehlo.dot_general %v938, %b3_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v940 = stablehlo.broadcast_in_dim %b3_bk, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v941 = stablehlo.add %v939, %v940 : tensor<32x197x384xf32>
    %v942 = stablehlo.reshape %v941 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v943 = stablehlo.reshape %v932 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v944 = stablehlo.dot_general %v943, %b3_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v945 = stablehlo.broadcast_in_dim %b3_bv, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v946 = stablehlo.add %v944, %v945 : tensor<32x197x384xf32>
    %v947 = stablehlo.reshape %v946 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v948 = stablehlo.reshape %v937 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v949 = stablehlo.slice %v948 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v950 = stablehlo.reshape %v949 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v951 = stablehlo.reshape %v942 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v952 = stablehlo.slice %v951 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v953 = stablehlo.reshape %v952 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v954 = stablehlo.reshape %v947 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v955 = stablehlo.slice %v954 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v956 = stablehlo.reshape %v955 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v957 = stablehlo.reshape %v953 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v958 = stablehlo.transpose %v957, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v959 = stablehlo.reshape %v958 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v960 = stablehlo.reshape %v950 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v961 = stablehlo.reshape %v959 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v962 = stablehlo.dot_general %v960, %v961, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v963 = stablehlo.reshape %v962 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v964 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v965 = stablehlo.multiply %v963, %v964 : tensor<32x38809xf32>
    %v966 = stablehlo.reshape %v965 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v967 = stablehlo.constant dense<0.0> : tensor<f32>
    %v968 = stablehlo.exponential %v966 : tensor<32x197x197xf32>
    %v969 = stablehlo.reduce(%v968 init: %v967) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v970 = stablehlo.broadcast_in_dim %v969, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v971 = stablehlo.divide %v968, %v970 : tensor<32x197x197xf32>
    %v972 = stablehlo.reshape %v971 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v973 = stablehlo.reshape %v972 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v974 = stablehlo.reshape %v956 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v975 = stablehlo.dot_general %v973, %v974, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v976 = stablehlo.reshape %v975 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v977 = stablehlo.reshape %v976 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v978 = stablehlo.constant dense<0.0> : tensor<f32>
    %v979 = stablehlo.pad %v977, %v978, low = [0, 0, 0], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v980 = stablehlo.reshape %v979 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v981 = stablehlo.reshape %v937 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v982 = stablehlo.slice %v981 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v983 = stablehlo.reshape %v982 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v984 = stablehlo.reshape %v942 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v985 = stablehlo.slice %v984 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v986 = stablehlo.reshape %v985 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v987 = stablehlo.reshape %v947 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v988 = stablehlo.slice %v987 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v989 = stablehlo.reshape %v988 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v990 = stablehlo.reshape %v986 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v991 = stablehlo.transpose %v990, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v992 = stablehlo.reshape %v991 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v993 = stablehlo.reshape %v983 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v994 = stablehlo.reshape %v992 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v995 = stablehlo.dot_general %v993, %v994, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v996 = stablehlo.reshape %v995 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v997 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v998 = stablehlo.multiply %v996, %v997 : tensor<32x38809xf32>
    %v999 = stablehlo.reshape %v998 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1000 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1001 = stablehlo.exponential %v999 : tensor<32x197x197xf32>
    %v1002 = stablehlo.reduce(%v1001 init: %v1000) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1003 = stablehlo.broadcast_in_dim %v1002, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1004 = stablehlo.divide %v1001, %v1003 : tensor<32x197x197xf32>
    %v1005 = stablehlo.reshape %v1004 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1006 = stablehlo.reshape %v1005 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1007 = stablehlo.reshape %v989 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1008 = stablehlo.dot_general %v1006, %v1007, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1009 = stablehlo.reshape %v1008 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1010 = stablehlo.reshape %v1009 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1011 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1012 = stablehlo.pad %v1010, %v1011, low = [0, 0, 64], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v1013 = stablehlo.reshape %v1012 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1014 = stablehlo.add %v980, %v1013 : tensor<32x75648xf32>
    %v1015 = stablehlo.reshape %v937 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1016 = stablehlo.slice %v1015 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1017 = stablehlo.reshape %v1016 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1018 = stablehlo.reshape %v942 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1019 = stablehlo.slice %v1018 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1020 = stablehlo.reshape %v1019 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1021 = stablehlo.reshape %v947 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1022 = stablehlo.slice %v1021 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1023 = stablehlo.reshape %v1022 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1024 = stablehlo.reshape %v1020 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1025 = stablehlo.transpose %v1024, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1026 = stablehlo.reshape %v1025 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1027 = stablehlo.reshape %v1017 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1028 = stablehlo.reshape %v1026 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1029 = stablehlo.dot_general %v1027, %v1028, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1030 = stablehlo.reshape %v1029 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1031 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1032 = stablehlo.multiply %v1030, %v1031 : tensor<32x38809xf32>
    %v1033 = stablehlo.reshape %v1032 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1034 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1035 = stablehlo.exponential %v1033 : tensor<32x197x197xf32>
    %v1036 = stablehlo.reduce(%v1035 init: %v1034) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1037 = stablehlo.broadcast_in_dim %v1036, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1038 = stablehlo.divide %v1035, %v1037 : tensor<32x197x197xf32>
    %v1039 = stablehlo.reshape %v1038 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1040 = stablehlo.reshape %v1039 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1041 = stablehlo.reshape %v1023 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1042 = stablehlo.dot_general %v1040, %v1041, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1043 = stablehlo.reshape %v1042 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1044 = stablehlo.reshape %v1043 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1045 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1046 = stablehlo.pad %v1044, %v1045, low = [0, 0, 128], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v1047 = stablehlo.reshape %v1046 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1048 = stablehlo.add %v1014, %v1047 : tensor<32x75648xf32>
    %v1049 = stablehlo.reshape %v937 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1050 = stablehlo.slice %v1049 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1051 = stablehlo.reshape %v1050 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1052 = stablehlo.reshape %v942 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1053 = stablehlo.slice %v1052 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1054 = stablehlo.reshape %v1053 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1055 = stablehlo.reshape %v947 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1056 = stablehlo.slice %v1055 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1057 = stablehlo.reshape %v1056 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1058 = stablehlo.reshape %v1054 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1059 = stablehlo.transpose %v1058, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1060 = stablehlo.reshape %v1059 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1061 = stablehlo.reshape %v1051 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1062 = stablehlo.reshape %v1060 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1063 = stablehlo.dot_general %v1061, %v1062, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1064 = stablehlo.reshape %v1063 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1065 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1066 = stablehlo.multiply %v1064, %v1065 : tensor<32x38809xf32>
    %v1067 = stablehlo.reshape %v1066 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1068 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1069 = stablehlo.exponential %v1067 : tensor<32x197x197xf32>
    %v1070 = stablehlo.reduce(%v1069 init: %v1068) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1071 = stablehlo.broadcast_in_dim %v1070, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1072 = stablehlo.divide %v1069, %v1071 : tensor<32x197x197xf32>
    %v1073 = stablehlo.reshape %v1072 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1074 = stablehlo.reshape %v1073 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1075 = stablehlo.reshape %v1057 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1076 = stablehlo.dot_general %v1074, %v1075, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1077 = stablehlo.reshape %v1076 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1078 = stablehlo.reshape %v1077 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1079 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1080 = stablehlo.pad %v1078, %v1079, low = [0, 0, 192], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v1081 = stablehlo.reshape %v1080 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1082 = stablehlo.add %v1048, %v1081 : tensor<32x75648xf32>
    %v1083 = stablehlo.reshape %v937 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1084 = stablehlo.slice %v1083 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1085 = stablehlo.reshape %v1084 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1086 = stablehlo.reshape %v942 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1087 = stablehlo.slice %v1086 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1088 = stablehlo.reshape %v1087 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1089 = stablehlo.reshape %v947 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1090 = stablehlo.slice %v1089 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1091 = stablehlo.reshape %v1090 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1092 = stablehlo.reshape %v1088 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1093 = stablehlo.transpose %v1092, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1094 = stablehlo.reshape %v1093 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1095 = stablehlo.reshape %v1085 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1096 = stablehlo.reshape %v1094 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1097 = stablehlo.dot_general %v1095, %v1096, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1098 = stablehlo.reshape %v1097 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1099 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1100 = stablehlo.multiply %v1098, %v1099 : tensor<32x38809xf32>
    %v1101 = stablehlo.reshape %v1100 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1102 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1103 = stablehlo.exponential %v1101 : tensor<32x197x197xf32>
    %v1104 = stablehlo.reduce(%v1103 init: %v1102) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1105 = stablehlo.broadcast_in_dim %v1104, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1106 = stablehlo.divide %v1103, %v1105 : tensor<32x197x197xf32>
    %v1107 = stablehlo.reshape %v1106 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1108 = stablehlo.reshape %v1107 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1109 = stablehlo.reshape %v1091 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1110 = stablehlo.dot_general %v1108, %v1109, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1111 = stablehlo.reshape %v1110 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1112 = stablehlo.reshape %v1111 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1113 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1114 = stablehlo.pad %v1112, %v1113, low = [0, 0, 256], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v1115 = stablehlo.reshape %v1114 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1116 = stablehlo.add %v1082, %v1115 : tensor<32x75648xf32>
    %v1117 = stablehlo.reshape %v937 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1118 = stablehlo.slice %v1117 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1119 = stablehlo.reshape %v1118 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1120 = stablehlo.reshape %v942 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1121 = stablehlo.slice %v1120 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1122 = stablehlo.reshape %v1121 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1123 = stablehlo.reshape %v947 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1124 = stablehlo.slice %v1123 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1125 = stablehlo.reshape %v1124 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1126 = stablehlo.reshape %v1122 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1127 = stablehlo.transpose %v1126, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1128 = stablehlo.reshape %v1127 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1129 = stablehlo.reshape %v1119 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1130 = stablehlo.reshape %v1128 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1131 = stablehlo.dot_general %v1129, %v1130, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1132 = stablehlo.reshape %v1131 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1133 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1134 = stablehlo.multiply %v1132, %v1133 : tensor<32x38809xf32>
    %v1135 = stablehlo.reshape %v1134 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1136 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1137 = stablehlo.exponential %v1135 : tensor<32x197x197xf32>
    %v1138 = stablehlo.reduce(%v1137 init: %v1136) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1139 = stablehlo.broadcast_in_dim %v1138, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1140 = stablehlo.divide %v1137, %v1139 : tensor<32x197x197xf32>
    %v1141 = stablehlo.reshape %v1140 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1142 = stablehlo.reshape %v1141 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1143 = stablehlo.reshape %v1125 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1144 = stablehlo.dot_general %v1142, %v1143, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1145 = stablehlo.reshape %v1144 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1146 = stablehlo.reshape %v1145 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1147 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1148 = stablehlo.pad %v1146, %v1147, low = [0, 0, 320], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v1149 = stablehlo.reshape %v1148 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1150 = stablehlo.add %v1116, %v1149 : tensor<32x75648xf32>
    %v1151 = stablehlo.reshape %v1150 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1152 = stablehlo.dot_general %v1151, %b3_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v1153 = stablehlo.broadcast_in_dim %b3_bo, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1154 = stablehlo.add %v1152, %v1153 : tensor<32x197x384xf32>
    %v1155 = stablehlo.reshape %v1154 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1156 = stablehlo.add %v904, %v1155 : tensor<32x75648xf32>
    %v1157 = stablehlo.reshape %v1156 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1158 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1159 = stablehlo.constant dense<384.0> : tensor<32x197x384xf32>
    %v1160 = stablehlo.constant dense<1.0e-5> : tensor<32x197x384xf32>
    %v1161 = stablehlo.reduce(%v1157 init: %v1158) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1162 = stablehlo.broadcast_in_dim %v1161, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v1163 = stablehlo.divide %v1162, %v1159 : tensor<32x197x384xf32>
    %v1164 = stablehlo.subtract %v1157, %v1163 : tensor<32x197x384xf32>
    %v1165 = stablehlo.multiply %v1164, %v1164 : tensor<32x197x384xf32>
    %v1166 = stablehlo.reduce(%v1165 init: %v1158) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1167 = stablehlo.broadcast_in_dim %v1166, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v1168 = stablehlo.divide %v1167, %v1159 : tensor<32x197x384xf32>
    %v1169 = stablehlo.add %v1168, %v1160 : tensor<32x197x384xf32>
    %v1170 = stablehlo.rsqrt %v1169 : tensor<32x197x384xf32>
    %v1171 = stablehlo.multiply %v1164, %v1170 : tensor<32x197x384xf32>
    %v1172 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v1173 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v1174 = stablehlo.multiply %v1171, %v1172 : tensor<32x197x384xf32>
    %v1175 = stablehlo.add %v1174, %v1173 : tensor<32x197x384xf32>
    %v1176 = stablehlo.reshape %v1175 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1177 = stablehlo.reshape %v1176 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1178 = stablehlo.broadcast_in_dim %b3_g2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1179 = stablehlo.multiply %v1177, %v1178 : tensor<32x197x384xf32>
    %v1180 = stablehlo.reshape %v1179 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1181 = stablehlo.reshape %v1180 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1182 = stablehlo.broadcast_in_dim %b3_bt2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1183 = stablehlo.add %v1181, %v1182 : tensor<32x197x384xf32>
    %v1184 = stablehlo.reshape %v1183 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1185 = stablehlo.reshape %v1184 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1186 = stablehlo.dot_general %v1185, %b3_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x1536xf32>) -> tensor<32x197x1536xf32>
    %v1187 = stablehlo.broadcast_in_dim %b3_bfc1, dims = [2] : (tensor<1536xf32>) -> tensor<32x197x1536xf32>
    %v1188 = stablehlo.add %v1186, %v1187 : tensor<32x197x1536xf32>
    %v1189 = stablehlo.reshape %v1188 : (tensor<32x197x1536xf32>) -> tensor<32x302592xf32>
    %v1190 = stablehlo.constant dense<0.5> : tensor<32x302592xf32>
    %v1191 = stablehlo.multiply %v1190, %v1189 : tensor<32x302592xf32>
    %v1192 = stablehlo.negate %v1189 : tensor<32x302592xf32>
    %v1193 = stablehlo.constant dense<0.7071067811865476> : tensor<32x302592xf32>
    %v1194 = stablehlo.multiply %v1192, %v1193 : tensor<32x302592xf32>
    %v1195 = chlo.erfc %v1194 : tensor<32x302592xf32> -> tensor<32x302592xf32>
    %v1196 = stablehlo.multiply %v1191, %v1195 : tensor<32x302592xf32>
    %v1197 = stablehlo.reshape %v1196 : (tensor<32x302592xf32>) -> tensor<32x197x1536xf32>
    %v1198 = stablehlo.dot_general %v1197, %b3_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x1536xf32>, tensor<1536x384xf32>) -> tensor<32x197x384xf32>
    %v1199 = stablehlo.broadcast_in_dim %b3_bfc2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1200 = stablehlo.add %v1198, %v1199 : tensor<32x197x384xf32>
    %v1201 = stablehlo.reshape %v1200 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1202 = stablehlo.add %v1156, %v1201 : tensor<32x75648xf32>
    %v1203 = stablehlo.reshape %v1202 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1204 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1205 = stablehlo.constant dense<384.0> : tensor<32x197x384xf32>
    %v1206 = stablehlo.constant dense<1.0e-5> : tensor<32x197x384xf32>
    %v1207 = stablehlo.reduce(%v1203 init: %v1204) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1208 = stablehlo.broadcast_in_dim %v1207, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v1209 = stablehlo.divide %v1208, %v1205 : tensor<32x197x384xf32>
    %v1210 = stablehlo.subtract %v1203, %v1209 : tensor<32x197x384xf32>
    %v1211 = stablehlo.multiply %v1210, %v1210 : tensor<32x197x384xf32>
    %v1212 = stablehlo.reduce(%v1211 init: %v1204) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1213 = stablehlo.broadcast_in_dim %v1212, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v1214 = stablehlo.divide %v1213, %v1205 : tensor<32x197x384xf32>
    %v1215 = stablehlo.add %v1214, %v1206 : tensor<32x197x384xf32>
    %v1216 = stablehlo.rsqrt %v1215 : tensor<32x197x384xf32>
    %v1217 = stablehlo.multiply %v1210, %v1216 : tensor<32x197x384xf32>
    %v1218 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v1219 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v1220 = stablehlo.multiply %v1217, %v1218 : tensor<32x197x384xf32>
    %v1221 = stablehlo.add %v1220, %v1219 : tensor<32x197x384xf32>
    %v1222 = stablehlo.reshape %v1221 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1223 = stablehlo.reshape %v1222 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1224 = stablehlo.broadcast_in_dim %b4_g1, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1225 = stablehlo.multiply %v1223, %v1224 : tensor<32x197x384xf32>
    %v1226 = stablehlo.reshape %v1225 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1227 = stablehlo.reshape %v1226 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1228 = stablehlo.broadcast_in_dim %b4_bt1, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1229 = stablehlo.add %v1227, %v1228 : tensor<32x197x384xf32>
    %v1230 = stablehlo.reshape %v1229 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1231 = stablehlo.reshape %v1230 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1232 = stablehlo.dot_general %v1231, %b4_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v1233 = stablehlo.broadcast_in_dim %b4_bq, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1234 = stablehlo.add %v1232, %v1233 : tensor<32x197x384xf32>
    %v1235 = stablehlo.reshape %v1234 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1236 = stablehlo.reshape %v1230 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1237 = stablehlo.dot_general %v1236, %b4_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v1238 = stablehlo.broadcast_in_dim %b4_bk, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1239 = stablehlo.add %v1237, %v1238 : tensor<32x197x384xf32>
    %v1240 = stablehlo.reshape %v1239 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1241 = stablehlo.reshape %v1230 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1242 = stablehlo.dot_general %v1241, %b4_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v1243 = stablehlo.broadcast_in_dim %b4_bv, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1244 = stablehlo.add %v1242, %v1243 : tensor<32x197x384xf32>
    %v1245 = stablehlo.reshape %v1244 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1246 = stablehlo.reshape %v1235 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1247 = stablehlo.slice %v1246 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1248 = stablehlo.reshape %v1247 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1249 = stablehlo.reshape %v1240 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1250 = stablehlo.slice %v1249 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1251 = stablehlo.reshape %v1250 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1252 = stablehlo.reshape %v1245 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1253 = stablehlo.slice %v1252 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1254 = stablehlo.reshape %v1253 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1255 = stablehlo.reshape %v1251 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1256 = stablehlo.transpose %v1255, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1257 = stablehlo.reshape %v1256 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1258 = stablehlo.reshape %v1248 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1259 = stablehlo.reshape %v1257 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1260 = stablehlo.dot_general %v1258, %v1259, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1261 = stablehlo.reshape %v1260 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1262 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1263 = stablehlo.multiply %v1261, %v1262 : tensor<32x38809xf32>
    %v1264 = stablehlo.reshape %v1263 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1265 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1266 = stablehlo.exponential %v1264 : tensor<32x197x197xf32>
    %v1267 = stablehlo.reduce(%v1266 init: %v1265) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1268 = stablehlo.broadcast_in_dim %v1267, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1269 = stablehlo.divide %v1266, %v1268 : tensor<32x197x197xf32>
    %v1270 = stablehlo.reshape %v1269 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1271 = stablehlo.reshape %v1270 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1272 = stablehlo.reshape %v1254 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1273 = stablehlo.dot_general %v1271, %v1272, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1274 = stablehlo.reshape %v1273 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1275 = stablehlo.reshape %v1274 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1276 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1277 = stablehlo.pad %v1275, %v1276, low = [0, 0, 0], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v1278 = stablehlo.reshape %v1277 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1279 = stablehlo.reshape %v1235 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1280 = stablehlo.slice %v1279 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1281 = stablehlo.reshape %v1280 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1282 = stablehlo.reshape %v1240 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1283 = stablehlo.slice %v1282 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1284 = stablehlo.reshape %v1283 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1285 = stablehlo.reshape %v1245 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1286 = stablehlo.slice %v1285 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1287 = stablehlo.reshape %v1286 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1288 = stablehlo.reshape %v1284 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1289 = stablehlo.transpose %v1288, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1290 = stablehlo.reshape %v1289 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1291 = stablehlo.reshape %v1281 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1292 = stablehlo.reshape %v1290 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1293 = stablehlo.dot_general %v1291, %v1292, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1294 = stablehlo.reshape %v1293 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1295 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1296 = stablehlo.multiply %v1294, %v1295 : tensor<32x38809xf32>
    %v1297 = stablehlo.reshape %v1296 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1298 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1299 = stablehlo.exponential %v1297 : tensor<32x197x197xf32>
    %v1300 = stablehlo.reduce(%v1299 init: %v1298) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1301 = stablehlo.broadcast_in_dim %v1300, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1302 = stablehlo.divide %v1299, %v1301 : tensor<32x197x197xf32>
    %v1303 = stablehlo.reshape %v1302 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1304 = stablehlo.reshape %v1303 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1305 = stablehlo.reshape %v1287 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1306 = stablehlo.dot_general %v1304, %v1305, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1307 = stablehlo.reshape %v1306 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1308 = stablehlo.reshape %v1307 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1309 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1310 = stablehlo.pad %v1308, %v1309, low = [0, 0, 64], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v1311 = stablehlo.reshape %v1310 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1312 = stablehlo.add %v1278, %v1311 : tensor<32x75648xf32>
    %v1313 = stablehlo.reshape %v1235 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1314 = stablehlo.slice %v1313 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1315 = stablehlo.reshape %v1314 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1316 = stablehlo.reshape %v1240 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1317 = stablehlo.slice %v1316 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1318 = stablehlo.reshape %v1317 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1319 = stablehlo.reshape %v1245 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1320 = stablehlo.slice %v1319 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1321 = stablehlo.reshape %v1320 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1322 = stablehlo.reshape %v1318 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1323 = stablehlo.transpose %v1322, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1324 = stablehlo.reshape %v1323 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1325 = stablehlo.reshape %v1315 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1326 = stablehlo.reshape %v1324 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1327 = stablehlo.dot_general %v1325, %v1326, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1328 = stablehlo.reshape %v1327 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1329 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1330 = stablehlo.multiply %v1328, %v1329 : tensor<32x38809xf32>
    %v1331 = stablehlo.reshape %v1330 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1332 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1333 = stablehlo.exponential %v1331 : tensor<32x197x197xf32>
    %v1334 = stablehlo.reduce(%v1333 init: %v1332) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1335 = stablehlo.broadcast_in_dim %v1334, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1336 = stablehlo.divide %v1333, %v1335 : tensor<32x197x197xf32>
    %v1337 = stablehlo.reshape %v1336 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1338 = stablehlo.reshape %v1337 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1339 = stablehlo.reshape %v1321 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1340 = stablehlo.dot_general %v1338, %v1339, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1341 = stablehlo.reshape %v1340 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1342 = stablehlo.reshape %v1341 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1343 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1344 = stablehlo.pad %v1342, %v1343, low = [0, 0, 128], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v1345 = stablehlo.reshape %v1344 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1346 = stablehlo.add %v1312, %v1345 : tensor<32x75648xf32>
    %v1347 = stablehlo.reshape %v1235 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1348 = stablehlo.slice %v1347 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1349 = stablehlo.reshape %v1348 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1350 = stablehlo.reshape %v1240 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1351 = stablehlo.slice %v1350 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1352 = stablehlo.reshape %v1351 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1353 = stablehlo.reshape %v1245 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1354 = stablehlo.slice %v1353 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1355 = stablehlo.reshape %v1354 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1356 = stablehlo.reshape %v1352 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1357 = stablehlo.transpose %v1356, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1358 = stablehlo.reshape %v1357 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1359 = stablehlo.reshape %v1349 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1360 = stablehlo.reshape %v1358 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1361 = stablehlo.dot_general %v1359, %v1360, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1362 = stablehlo.reshape %v1361 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1363 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1364 = stablehlo.multiply %v1362, %v1363 : tensor<32x38809xf32>
    %v1365 = stablehlo.reshape %v1364 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1366 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1367 = stablehlo.exponential %v1365 : tensor<32x197x197xf32>
    %v1368 = stablehlo.reduce(%v1367 init: %v1366) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1369 = stablehlo.broadcast_in_dim %v1368, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1370 = stablehlo.divide %v1367, %v1369 : tensor<32x197x197xf32>
    %v1371 = stablehlo.reshape %v1370 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1372 = stablehlo.reshape %v1371 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1373 = stablehlo.reshape %v1355 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1374 = stablehlo.dot_general %v1372, %v1373, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1375 = stablehlo.reshape %v1374 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1376 = stablehlo.reshape %v1375 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1377 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1378 = stablehlo.pad %v1376, %v1377, low = [0, 0, 192], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v1379 = stablehlo.reshape %v1378 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1380 = stablehlo.add %v1346, %v1379 : tensor<32x75648xf32>
    %v1381 = stablehlo.reshape %v1235 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1382 = stablehlo.slice %v1381 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1383 = stablehlo.reshape %v1382 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1384 = stablehlo.reshape %v1240 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1385 = stablehlo.slice %v1384 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1386 = stablehlo.reshape %v1385 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1387 = stablehlo.reshape %v1245 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1388 = stablehlo.slice %v1387 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
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
    %v1400 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1401 = stablehlo.exponential %v1399 : tensor<32x197x197xf32>
    %v1402 = stablehlo.reduce(%v1401 init: %v1400) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1403 = stablehlo.broadcast_in_dim %v1402, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1404 = stablehlo.divide %v1401, %v1403 : tensor<32x197x197xf32>
    %v1405 = stablehlo.reshape %v1404 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1406 = stablehlo.reshape %v1405 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1407 = stablehlo.reshape %v1389 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1408 = stablehlo.dot_general %v1406, %v1407, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1409 = stablehlo.reshape %v1408 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1410 = stablehlo.reshape %v1409 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1411 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1412 = stablehlo.pad %v1410, %v1411, low = [0, 0, 256], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v1413 = stablehlo.reshape %v1412 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1414 = stablehlo.add %v1380, %v1413 : tensor<32x75648xf32>
    %v1415 = stablehlo.reshape %v1235 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1416 = stablehlo.slice %v1415 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1417 = stablehlo.reshape %v1416 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1418 = stablehlo.reshape %v1240 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1419 = stablehlo.slice %v1418 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1420 = stablehlo.reshape %v1419 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1421 = stablehlo.reshape %v1245 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1422 = stablehlo.slice %v1421 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1423 = stablehlo.reshape %v1422 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1424 = stablehlo.reshape %v1420 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1425 = stablehlo.transpose %v1424, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1426 = stablehlo.reshape %v1425 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1427 = stablehlo.reshape %v1417 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1428 = stablehlo.reshape %v1426 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1429 = stablehlo.dot_general %v1427, %v1428, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1430 = stablehlo.reshape %v1429 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1431 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1432 = stablehlo.multiply %v1430, %v1431 : tensor<32x38809xf32>
    %v1433 = stablehlo.reshape %v1432 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1434 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1435 = stablehlo.exponential %v1433 : tensor<32x197x197xf32>
    %v1436 = stablehlo.reduce(%v1435 init: %v1434) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1437 = stablehlo.broadcast_in_dim %v1436, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1438 = stablehlo.divide %v1435, %v1437 : tensor<32x197x197xf32>
    %v1439 = stablehlo.reshape %v1438 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1440 = stablehlo.reshape %v1439 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1441 = stablehlo.reshape %v1423 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1442 = stablehlo.dot_general %v1440, %v1441, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1443 = stablehlo.reshape %v1442 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1444 = stablehlo.reshape %v1443 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1445 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1446 = stablehlo.pad %v1444, %v1445, low = [0, 0, 320], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v1447 = stablehlo.reshape %v1446 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1448 = stablehlo.add %v1414, %v1447 : tensor<32x75648xf32>
    %v1449 = stablehlo.reshape %v1448 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1450 = stablehlo.dot_general %v1449, %b4_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v1451 = stablehlo.broadcast_in_dim %b4_bo, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1452 = stablehlo.add %v1450, %v1451 : tensor<32x197x384xf32>
    %v1453 = stablehlo.reshape %v1452 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1454 = stablehlo.add %v1202, %v1453 : tensor<32x75648xf32>
    %v1455 = stablehlo.reshape %v1454 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1456 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1457 = stablehlo.constant dense<384.0> : tensor<32x197x384xf32>
    %v1458 = stablehlo.constant dense<1.0e-5> : tensor<32x197x384xf32>
    %v1459 = stablehlo.reduce(%v1455 init: %v1456) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1460 = stablehlo.broadcast_in_dim %v1459, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v1461 = stablehlo.divide %v1460, %v1457 : tensor<32x197x384xf32>
    %v1462 = stablehlo.subtract %v1455, %v1461 : tensor<32x197x384xf32>
    %v1463 = stablehlo.multiply %v1462, %v1462 : tensor<32x197x384xf32>
    %v1464 = stablehlo.reduce(%v1463 init: %v1456) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1465 = stablehlo.broadcast_in_dim %v1464, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v1466 = stablehlo.divide %v1465, %v1457 : tensor<32x197x384xf32>
    %v1467 = stablehlo.add %v1466, %v1458 : tensor<32x197x384xf32>
    %v1468 = stablehlo.rsqrt %v1467 : tensor<32x197x384xf32>
    %v1469 = stablehlo.multiply %v1462, %v1468 : tensor<32x197x384xf32>
    %v1470 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v1471 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v1472 = stablehlo.multiply %v1469, %v1470 : tensor<32x197x384xf32>
    %v1473 = stablehlo.add %v1472, %v1471 : tensor<32x197x384xf32>
    %v1474 = stablehlo.reshape %v1473 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1475 = stablehlo.reshape %v1474 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1476 = stablehlo.broadcast_in_dim %b4_g2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1477 = stablehlo.multiply %v1475, %v1476 : tensor<32x197x384xf32>
    %v1478 = stablehlo.reshape %v1477 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1479 = stablehlo.reshape %v1478 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1480 = stablehlo.broadcast_in_dim %b4_bt2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1481 = stablehlo.add %v1479, %v1480 : tensor<32x197x384xf32>
    %v1482 = stablehlo.reshape %v1481 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1483 = stablehlo.reshape %v1482 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1484 = stablehlo.dot_general %v1483, %b4_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x1536xf32>) -> tensor<32x197x1536xf32>
    %v1485 = stablehlo.broadcast_in_dim %b4_bfc1, dims = [2] : (tensor<1536xf32>) -> tensor<32x197x1536xf32>
    %v1486 = stablehlo.add %v1484, %v1485 : tensor<32x197x1536xf32>
    %v1487 = stablehlo.reshape %v1486 : (tensor<32x197x1536xf32>) -> tensor<32x302592xf32>
    %v1488 = stablehlo.constant dense<0.5> : tensor<32x302592xf32>
    %v1489 = stablehlo.multiply %v1488, %v1487 : tensor<32x302592xf32>
    %v1490 = stablehlo.negate %v1487 : tensor<32x302592xf32>
    %v1491 = stablehlo.constant dense<0.7071067811865476> : tensor<32x302592xf32>
    %v1492 = stablehlo.multiply %v1490, %v1491 : tensor<32x302592xf32>
    %v1493 = chlo.erfc %v1492 : tensor<32x302592xf32> -> tensor<32x302592xf32>
    %v1494 = stablehlo.multiply %v1489, %v1493 : tensor<32x302592xf32>
    %v1495 = stablehlo.reshape %v1494 : (tensor<32x302592xf32>) -> tensor<32x197x1536xf32>
    %v1496 = stablehlo.dot_general %v1495, %b4_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x1536xf32>, tensor<1536x384xf32>) -> tensor<32x197x384xf32>
    %v1497 = stablehlo.broadcast_in_dim %b4_bfc2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1498 = stablehlo.add %v1496, %v1497 : tensor<32x197x384xf32>
    %v1499 = stablehlo.reshape %v1498 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1500 = stablehlo.add %v1454, %v1499 : tensor<32x75648xf32>
    %v1501 = stablehlo.reshape %v1500 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1502 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1503 = stablehlo.constant dense<384.0> : tensor<32x197x384xf32>
    %v1504 = stablehlo.constant dense<1.0e-5> : tensor<32x197x384xf32>
    %v1505 = stablehlo.reduce(%v1501 init: %v1502) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1506 = stablehlo.broadcast_in_dim %v1505, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v1507 = stablehlo.divide %v1506, %v1503 : tensor<32x197x384xf32>
    %v1508 = stablehlo.subtract %v1501, %v1507 : tensor<32x197x384xf32>
    %v1509 = stablehlo.multiply %v1508, %v1508 : tensor<32x197x384xf32>
    %v1510 = stablehlo.reduce(%v1509 init: %v1502) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1511 = stablehlo.broadcast_in_dim %v1510, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v1512 = stablehlo.divide %v1511, %v1503 : tensor<32x197x384xf32>
    %v1513 = stablehlo.add %v1512, %v1504 : tensor<32x197x384xf32>
    %v1514 = stablehlo.rsqrt %v1513 : tensor<32x197x384xf32>
    %v1515 = stablehlo.multiply %v1508, %v1514 : tensor<32x197x384xf32>
    %v1516 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v1517 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v1518 = stablehlo.multiply %v1515, %v1516 : tensor<32x197x384xf32>
    %v1519 = stablehlo.add %v1518, %v1517 : tensor<32x197x384xf32>
    %v1520 = stablehlo.reshape %v1519 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1521 = stablehlo.reshape %v1520 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1522 = stablehlo.broadcast_in_dim %b5_g1, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1523 = stablehlo.multiply %v1521, %v1522 : tensor<32x197x384xf32>
    %v1524 = stablehlo.reshape %v1523 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1525 = stablehlo.reshape %v1524 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1526 = stablehlo.broadcast_in_dim %b5_bt1, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1527 = stablehlo.add %v1525, %v1526 : tensor<32x197x384xf32>
    %v1528 = stablehlo.reshape %v1527 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1529 = stablehlo.reshape %v1528 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1530 = stablehlo.dot_general %v1529, %b5_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v1531 = stablehlo.broadcast_in_dim %b5_bq, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1532 = stablehlo.add %v1530, %v1531 : tensor<32x197x384xf32>
    %v1533 = stablehlo.reshape %v1532 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1534 = stablehlo.reshape %v1528 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1535 = stablehlo.dot_general %v1534, %b5_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v1536 = stablehlo.broadcast_in_dim %b5_bk, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1537 = stablehlo.add %v1535, %v1536 : tensor<32x197x384xf32>
    %v1538 = stablehlo.reshape %v1537 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1539 = stablehlo.reshape %v1528 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1540 = stablehlo.dot_general %v1539, %b5_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v1541 = stablehlo.broadcast_in_dim %b5_bv, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1542 = stablehlo.add %v1540, %v1541 : tensor<32x197x384xf32>
    %v1543 = stablehlo.reshape %v1542 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1544 = stablehlo.reshape %v1533 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1545 = stablehlo.slice %v1544 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1546 = stablehlo.reshape %v1545 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1547 = stablehlo.reshape %v1538 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1548 = stablehlo.slice %v1547 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1549 = stablehlo.reshape %v1548 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1550 = stablehlo.reshape %v1543 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1551 = stablehlo.slice %v1550 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1552 = stablehlo.reshape %v1551 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1553 = stablehlo.reshape %v1549 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1554 = stablehlo.transpose %v1553, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1555 = stablehlo.reshape %v1554 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1556 = stablehlo.reshape %v1546 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1557 = stablehlo.reshape %v1555 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1558 = stablehlo.dot_general %v1556, %v1557, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1559 = stablehlo.reshape %v1558 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1560 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1561 = stablehlo.multiply %v1559, %v1560 : tensor<32x38809xf32>
    %v1562 = stablehlo.reshape %v1561 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1563 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1564 = stablehlo.exponential %v1562 : tensor<32x197x197xf32>
    %v1565 = stablehlo.reduce(%v1564 init: %v1563) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1566 = stablehlo.broadcast_in_dim %v1565, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1567 = stablehlo.divide %v1564, %v1566 : tensor<32x197x197xf32>
    %v1568 = stablehlo.reshape %v1567 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1569 = stablehlo.reshape %v1568 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1570 = stablehlo.reshape %v1552 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1571 = stablehlo.dot_general %v1569, %v1570, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1572 = stablehlo.reshape %v1571 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1573 = stablehlo.reshape %v1572 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1574 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1575 = stablehlo.pad %v1573, %v1574, low = [0, 0, 0], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v1576 = stablehlo.reshape %v1575 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1577 = stablehlo.reshape %v1533 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1578 = stablehlo.slice %v1577 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1579 = stablehlo.reshape %v1578 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1580 = stablehlo.reshape %v1538 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1581 = stablehlo.slice %v1580 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1582 = stablehlo.reshape %v1581 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1583 = stablehlo.reshape %v1543 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1584 = stablehlo.slice %v1583 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1585 = stablehlo.reshape %v1584 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1586 = stablehlo.reshape %v1582 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1587 = stablehlo.transpose %v1586, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1588 = stablehlo.reshape %v1587 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1589 = stablehlo.reshape %v1579 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1590 = stablehlo.reshape %v1588 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1591 = stablehlo.dot_general %v1589, %v1590, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1592 = stablehlo.reshape %v1591 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1593 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1594 = stablehlo.multiply %v1592, %v1593 : tensor<32x38809xf32>
    %v1595 = stablehlo.reshape %v1594 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1596 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1597 = stablehlo.exponential %v1595 : tensor<32x197x197xf32>
    %v1598 = stablehlo.reduce(%v1597 init: %v1596) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1599 = stablehlo.broadcast_in_dim %v1598, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1600 = stablehlo.divide %v1597, %v1599 : tensor<32x197x197xf32>
    %v1601 = stablehlo.reshape %v1600 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1602 = stablehlo.reshape %v1601 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1603 = stablehlo.reshape %v1585 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1604 = stablehlo.dot_general %v1602, %v1603, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1605 = stablehlo.reshape %v1604 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1606 = stablehlo.reshape %v1605 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1607 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1608 = stablehlo.pad %v1606, %v1607, low = [0, 0, 64], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v1609 = stablehlo.reshape %v1608 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1610 = stablehlo.add %v1576, %v1609 : tensor<32x75648xf32>
    %v1611 = stablehlo.reshape %v1533 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1612 = stablehlo.slice %v1611 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1613 = stablehlo.reshape %v1612 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1614 = stablehlo.reshape %v1538 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1615 = stablehlo.slice %v1614 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1616 = stablehlo.reshape %v1615 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1617 = stablehlo.reshape %v1543 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1618 = stablehlo.slice %v1617 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
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
    %v1642 = stablehlo.pad %v1640, %v1641, low = [0, 0, 128], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v1643 = stablehlo.reshape %v1642 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1644 = stablehlo.add %v1610, %v1643 : tensor<32x75648xf32>
    %v1645 = stablehlo.reshape %v1533 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1646 = stablehlo.slice %v1645 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1647 = stablehlo.reshape %v1646 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1648 = stablehlo.reshape %v1538 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1649 = stablehlo.slice %v1648 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1650 = stablehlo.reshape %v1649 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1651 = stablehlo.reshape %v1543 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1652 = stablehlo.slice %v1651 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
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
    %v1676 = stablehlo.pad %v1674, %v1675, low = [0, 0, 192], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v1677 = stablehlo.reshape %v1676 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1678 = stablehlo.add %v1644, %v1677 : tensor<32x75648xf32>
    %v1679 = stablehlo.reshape %v1533 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1680 = stablehlo.slice %v1679 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1681 = stablehlo.reshape %v1680 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1682 = stablehlo.reshape %v1538 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1683 = stablehlo.slice %v1682 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1684 = stablehlo.reshape %v1683 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1685 = stablehlo.reshape %v1543 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1686 = stablehlo.slice %v1685 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
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
    %v1710 = stablehlo.pad %v1708, %v1709, low = [0, 0, 256], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v1711 = stablehlo.reshape %v1710 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1712 = stablehlo.add %v1678, %v1711 : tensor<32x75648xf32>
    %v1713 = stablehlo.reshape %v1533 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1714 = stablehlo.slice %v1713 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1715 = stablehlo.reshape %v1714 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1716 = stablehlo.reshape %v1538 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1717 = stablehlo.slice %v1716 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1718 = stablehlo.reshape %v1717 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1719 = stablehlo.reshape %v1543 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1720 = stablehlo.slice %v1719 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
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
    %v1744 = stablehlo.pad %v1742, %v1743, low = [0, 0, 320], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v1745 = stablehlo.reshape %v1744 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1746 = stablehlo.add %v1712, %v1745 : tensor<32x75648xf32>
    %v1747 = stablehlo.reshape %v1746 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1748 = stablehlo.dot_general %v1747, %b5_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v1749 = stablehlo.broadcast_in_dim %b5_bo, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1750 = stablehlo.add %v1748, %v1749 : tensor<32x197x384xf32>
    %v1751 = stablehlo.reshape %v1750 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1752 = stablehlo.add %v1500, %v1751 : tensor<32x75648xf32>
    %v1753 = stablehlo.reshape %v1752 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1754 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1755 = stablehlo.constant dense<384.0> : tensor<32x197x384xf32>
    %v1756 = stablehlo.constant dense<1.0e-5> : tensor<32x197x384xf32>
    %v1757 = stablehlo.reduce(%v1753 init: %v1754) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1758 = stablehlo.broadcast_in_dim %v1757, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v1759 = stablehlo.divide %v1758, %v1755 : tensor<32x197x384xf32>
    %v1760 = stablehlo.subtract %v1753, %v1759 : tensor<32x197x384xf32>
    %v1761 = stablehlo.multiply %v1760, %v1760 : tensor<32x197x384xf32>
    %v1762 = stablehlo.reduce(%v1761 init: %v1754) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1763 = stablehlo.broadcast_in_dim %v1762, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v1764 = stablehlo.divide %v1763, %v1755 : tensor<32x197x384xf32>
    %v1765 = stablehlo.add %v1764, %v1756 : tensor<32x197x384xf32>
    %v1766 = stablehlo.rsqrt %v1765 : tensor<32x197x384xf32>
    %v1767 = stablehlo.multiply %v1760, %v1766 : tensor<32x197x384xf32>
    %v1768 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v1769 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v1770 = stablehlo.multiply %v1767, %v1768 : tensor<32x197x384xf32>
    %v1771 = stablehlo.add %v1770, %v1769 : tensor<32x197x384xf32>
    %v1772 = stablehlo.reshape %v1771 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1773 = stablehlo.reshape %v1772 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1774 = stablehlo.broadcast_in_dim %b5_g2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1775 = stablehlo.multiply %v1773, %v1774 : tensor<32x197x384xf32>
    %v1776 = stablehlo.reshape %v1775 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1777 = stablehlo.reshape %v1776 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1778 = stablehlo.broadcast_in_dim %b5_bt2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1779 = stablehlo.add %v1777, %v1778 : tensor<32x197x384xf32>
    %v1780 = stablehlo.reshape %v1779 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1781 = stablehlo.reshape %v1780 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1782 = stablehlo.dot_general %v1781, %b5_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x1536xf32>) -> tensor<32x197x1536xf32>
    %v1783 = stablehlo.broadcast_in_dim %b5_bfc1, dims = [2] : (tensor<1536xf32>) -> tensor<32x197x1536xf32>
    %v1784 = stablehlo.add %v1782, %v1783 : tensor<32x197x1536xf32>
    %v1785 = stablehlo.reshape %v1784 : (tensor<32x197x1536xf32>) -> tensor<32x302592xf32>
    %v1786 = stablehlo.constant dense<0.5> : tensor<32x302592xf32>
    %v1787 = stablehlo.multiply %v1786, %v1785 : tensor<32x302592xf32>
    %v1788 = stablehlo.negate %v1785 : tensor<32x302592xf32>
    %v1789 = stablehlo.constant dense<0.7071067811865476> : tensor<32x302592xf32>
    %v1790 = stablehlo.multiply %v1788, %v1789 : tensor<32x302592xf32>
    %v1791 = chlo.erfc %v1790 : tensor<32x302592xf32> -> tensor<32x302592xf32>
    %v1792 = stablehlo.multiply %v1787, %v1791 : tensor<32x302592xf32>
    %v1793 = stablehlo.reshape %v1792 : (tensor<32x302592xf32>) -> tensor<32x197x1536xf32>
    %v1794 = stablehlo.dot_general %v1793, %b5_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x1536xf32>, tensor<1536x384xf32>) -> tensor<32x197x384xf32>
    %v1795 = stablehlo.broadcast_in_dim %b5_bfc2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1796 = stablehlo.add %v1794, %v1795 : tensor<32x197x384xf32>
    %v1797 = stablehlo.reshape %v1796 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1798 = stablehlo.add %v1752, %v1797 : tensor<32x75648xf32>
    %v1799 = stablehlo.reshape %v1798 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1800 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1801 = stablehlo.constant dense<384.0> : tensor<32x197x384xf32>
    %v1802 = stablehlo.constant dense<1.0e-5> : tensor<32x197x384xf32>
    %v1803 = stablehlo.reduce(%v1799 init: %v1800) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1804 = stablehlo.broadcast_in_dim %v1803, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v1805 = stablehlo.divide %v1804, %v1801 : tensor<32x197x384xf32>
    %v1806 = stablehlo.subtract %v1799, %v1805 : tensor<32x197x384xf32>
    %v1807 = stablehlo.multiply %v1806, %v1806 : tensor<32x197x384xf32>
    %v1808 = stablehlo.reduce(%v1807 init: %v1800) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1809 = stablehlo.broadcast_in_dim %v1808, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v1810 = stablehlo.divide %v1809, %v1801 : tensor<32x197x384xf32>
    %v1811 = stablehlo.add %v1810, %v1802 : tensor<32x197x384xf32>
    %v1812 = stablehlo.rsqrt %v1811 : tensor<32x197x384xf32>
    %v1813 = stablehlo.multiply %v1806, %v1812 : tensor<32x197x384xf32>
    %v1814 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v1815 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v1816 = stablehlo.multiply %v1813, %v1814 : tensor<32x197x384xf32>
    %v1817 = stablehlo.add %v1816, %v1815 : tensor<32x197x384xf32>
    %v1818 = stablehlo.reshape %v1817 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1819 = stablehlo.reshape %v1818 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1820 = stablehlo.broadcast_in_dim %b6_g1, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1821 = stablehlo.multiply %v1819, %v1820 : tensor<32x197x384xf32>
    %v1822 = stablehlo.reshape %v1821 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1823 = stablehlo.reshape %v1822 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1824 = stablehlo.broadcast_in_dim %b6_bt1, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1825 = stablehlo.add %v1823, %v1824 : tensor<32x197x384xf32>
    %v1826 = stablehlo.reshape %v1825 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1827 = stablehlo.reshape %v1826 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1828 = stablehlo.dot_general %v1827, %b6_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v1829 = stablehlo.broadcast_in_dim %b6_bq, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1830 = stablehlo.add %v1828, %v1829 : tensor<32x197x384xf32>
    %v1831 = stablehlo.reshape %v1830 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1832 = stablehlo.reshape %v1826 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1833 = stablehlo.dot_general %v1832, %b6_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v1834 = stablehlo.broadcast_in_dim %b6_bk, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1835 = stablehlo.add %v1833, %v1834 : tensor<32x197x384xf32>
    %v1836 = stablehlo.reshape %v1835 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1837 = stablehlo.reshape %v1826 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1838 = stablehlo.dot_general %v1837, %b6_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v1839 = stablehlo.broadcast_in_dim %b6_bv, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1840 = stablehlo.add %v1838, %v1839 : tensor<32x197x384xf32>
    %v1841 = stablehlo.reshape %v1840 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1842 = stablehlo.reshape %v1831 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1843 = stablehlo.slice %v1842 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1844 = stablehlo.reshape %v1843 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1845 = stablehlo.reshape %v1836 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1846 = stablehlo.slice %v1845 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1847 = stablehlo.reshape %v1846 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1848 = stablehlo.reshape %v1841 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1849 = stablehlo.slice %v1848 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1850 = stablehlo.reshape %v1849 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1851 = stablehlo.reshape %v1847 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1852 = stablehlo.transpose %v1851, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1853 = stablehlo.reshape %v1852 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1854 = stablehlo.reshape %v1844 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1855 = stablehlo.reshape %v1853 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1856 = stablehlo.dot_general %v1854, %v1855, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1857 = stablehlo.reshape %v1856 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1858 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1859 = stablehlo.multiply %v1857, %v1858 : tensor<32x38809xf32>
    %v1860 = stablehlo.reshape %v1859 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1861 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1862 = stablehlo.exponential %v1860 : tensor<32x197x197xf32>
    %v1863 = stablehlo.reduce(%v1862 init: %v1861) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1864 = stablehlo.broadcast_in_dim %v1863, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1865 = stablehlo.divide %v1862, %v1864 : tensor<32x197x197xf32>
    %v1866 = stablehlo.reshape %v1865 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1867 = stablehlo.reshape %v1866 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1868 = stablehlo.reshape %v1850 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1869 = stablehlo.dot_general %v1867, %v1868, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1870 = stablehlo.reshape %v1869 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1871 = stablehlo.reshape %v1870 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1872 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1873 = stablehlo.pad %v1871, %v1872, low = [0, 0, 0], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v1874 = stablehlo.reshape %v1873 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1875 = stablehlo.reshape %v1831 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1876 = stablehlo.slice %v1875 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1877 = stablehlo.reshape %v1876 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1878 = stablehlo.reshape %v1836 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1879 = stablehlo.slice %v1878 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1880 = stablehlo.reshape %v1879 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1881 = stablehlo.reshape %v1841 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1882 = stablehlo.slice %v1881 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1883 = stablehlo.reshape %v1882 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1884 = stablehlo.reshape %v1880 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1885 = stablehlo.transpose %v1884, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1886 = stablehlo.reshape %v1885 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1887 = stablehlo.reshape %v1877 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1888 = stablehlo.reshape %v1886 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1889 = stablehlo.dot_general %v1887, %v1888, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1890 = stablehlo.reshape %v1889 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1891 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1892 = stablehlo.multiply %v1890, %v1891 : tensor<32x38809xf32>
    %v1893 = stablehlo.reshape %v1892 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1894 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1895 = stablehlo.exponential %v1893 : tensor<32x197x197xf32>
    %v1896 = stablehlo.reduce(%v1895 init: %v1894) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1897 = stablehlo.broadcast_in_dim %v1896, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1898 = stablehlo.divide %v1895, %v1897 : tensor<32x197x197xf32>
    %v1899 = stablehlo.reshape %v1898 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1900 = stablehlo.reshape %v1899 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1901 = stablehlo.reshape %v1883 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1902 = stablehlo.dot_general %v1900, %v1901, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1903 = stablehlo.reshape %v1902 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1904 = stablehlo.reshape %v1903 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1905 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1906 = stablehlo.pad %v1904, %v1905, low = [0, 0, 64], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v1907 = stablehlo.reshape %v1906 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1908 = stablehlo.add %v1874, %v1907 : tensor<32x75648xf32>
    %v1909 = stablehlo.reshape %v1831 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1910 = stablehlo.slice %v1909 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1911 = stablehlo.reshape %v1910 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1912 = stablehlo.reshape %v1836 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1913 = stablehlo.slice %v1912 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1914 = stablehlo.reshape %v1913 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1915 = stablehlo.reshape %v1841 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1916 = stablehlo.slice %v1915 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1917 = stablehlo.reshape %v1916 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1918 = stablehlo.reshape %v1914 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1919 = stablehlo.transpose %v1918, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1920 = stablehlo.reshape %v1919 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1921 = stablehlo.reshape %v1911 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1922 = stablehlo.reshape %v1920 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1923 = stablehlo.dot_general %v1921, %v1922, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1924 = stablehlo.reshape %v1923 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1925 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1926 = stablehlo.multiply %v1924, %v1925 : tensor<32x38809xf32>
    %v1927 = stablehlo.reshape %v1926 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1928 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1929 = stablehlo.exponential %v1927 : tensor<32x197x197xf32>
    %v1930 = stablehlo.reduce(%v1929 init: %v1928) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1931 = stablehlo.broadcast_in_dim %v1930, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1932 = stablehlo.divide %v1929, %v1931 : tensor<32x197x197xf32>
    %v1933 = stablehlo.reshape %v1932 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1934 = stablehlo.reshape %v1933 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1935 = stablehlo.reshape %v1917 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1936 = stablehlo.dot_general %v1934, %v1935, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1937 = stablehlo.reshape %v1936 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1938 = stablehlo.reshape %v1937 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1939 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1940 = stablehlo.pad %v1938, %v1939, low = [0, 0, 128], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v1941 = stablehlo.reshape %v1940 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1942 = stablehlo.add %v1908, %v1941 : tensor<32x75648xf32>
    %v1943 = stablehlo.reshape %v1831 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1944 = stablehlo.slice %v1943 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1945 = stablehlo.reshape %v1944 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1946 = stablehlo.reshape %v1836 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1947 = stablehlo.slice %v1946 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1948 = stablehlo.reshape %v1947 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1949 = stablehlo.reshape %v1841 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1950 = stablehlo.slice %v1949 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1951 = stablehlo.reshape %v1950 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1952 = stablehlo.reshape %v1948 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1953 = stablehlo.transpose %v1952, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1954 = stablehlo.reshape %v1953 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1955 = stablehlo.reshape %v1945 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1956 = stablehlo.reshape %v1954 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1957 = stablehlo.dot_general %v1955, %v1956, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1958 = stablehlo.reshape %v1957 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1959 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1960 = stablehlo.multiply %v1958, %v1959 : tensor<32x38809xf32>
    %v1961 = stablehlo.reshape %v1960 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1962 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1963 = stablehlo.exponential %v1961 : tensor<32x197x197xf32>
    %v1964 = stablehlo.reduce(%v1963 init: %v1962) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1965 = stablehlo.broadcast_in_dim %v1964, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1966 = stablehlo.divide %v1963, %v1965 : tensor<32x197x197xf32>
    %v1967 = stablehlo.reshape %v1966 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1968 = stablehlo.reshape %v1967 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1969 = stablehlo.reshape %v1951 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1970 = stablehlo.dot_general %v1968, %v1969, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1971 = stablehlo.reshape %v1970 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1972 = stablehlo.reshape %v1971 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1973 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1974 = stablehlo.pad %v1972, %v1973, low = [0, 0, 192], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v1975 = stablehlo.reshape %v1974 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1976 = stablehlo.add %v1942, %v1975 : tensor<32x75648xf32>
    %v1977 = stablehlo.reshape %v1831 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1978 = stablehlo.slice %v1977 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1979 = stablehlo.reshape %v1978 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1980 = stablehlo.reshape %v1836 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1981 = stablehlo.slice %v1980 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1982 = stablehlo.reshape %v1981 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1983 = stablehlo.reshape %v1841 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1984 = stablehlo.slice %v1983 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1985 = stablehlo.reshape %v1984 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1986 = stablehlo.reshape %v1982 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1987 = stablehlo.transpose %v1986, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1988 = stablehlo.reshape %v1987 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1989 = stablehlo.reshape %v1979 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1990 = stablehlo.reshape %v1988 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1991 = stablehlo.dot_general %v1989, %v1990, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1992 = stablehlo.reshape %v1991 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1993 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1994 = stablehlo.multiply %v1992, %v1993 : tensor<32x38809xf32>
    %v1995 = stablehlo.reshape %v1994 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1996 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1997 = stablehlo.exponential %v1995 : tensor<32x197x197xf32>
    %v1998 = stablehlo.reduce(%v1997 init: %v1996) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1999 = stablehlo.broadcast_in_dim %v1998, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2000 = stablehlo.divide %v1997, %v1999 : tensor<32x197x197xf32>
    %v2001 = stablehlo.reshape %v2000 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2002 = stablehlo.reshape %v2001 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2003 = stablehlo.reshape %v1985 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2004 = stablehlo.dot_general %v2002, %v2003, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2005 = stablehlo.reshape %v2004 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2006 = stablehlo.reshape %v2005 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2007 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2008 = stablehlo.pad %v2006, %v2007, low = [0, 0, 256], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v2009 = stablehlo.reshape %v2008 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2010 = stablehlo.add %v1976, %v2009 : tensor<32x75648xf32>
    %v2011 = stablehlo.reshape %v1831 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2012 = stablehlo.slice %v2011 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2013 = stablehlo.reshape %v2012 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2014 = stablehlo.reshape %v1836 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2015 = stablehlo.slice %v2014 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2016 = stablehlo.reshape %v2015 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2017 = stablehlo.reshape %v1841 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2018 = stablehlo.slice %v2017 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2019 = stablehlo.reshape %v2018 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2020 = stablehlo.reshape %v2016 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2021 = stablehlo.transpose %v2020, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2022 = stablehlo.reshape %v2021 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2023 = stablehlo.reshape %v2013 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2024 = stablehlo.reshape %v2022 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2025 = stablehlo.dot_general %v2023, %v2024, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2026 = stablehlo.reshape %v2025 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2027 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2028 = stablehlo.multiply %v2026, %v2027 : tensor<32x38809xf32>
    %v2029 = stablehlo.reshape %v2028 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2030 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2031 = stablehlo.exponential %v2029 : tensor<32x197x197xf32>
    %v2032 = stablehlo.reduce(%v2031 init: %v2030) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2033 = stablehlo.broadcast_in_dim %v2032, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2034 = stablehlo.divide %v2031, %v2033 : tensor<32x197x197xf32>
    %v2035 = stablehlo.reshape %v2034 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2036 = stablehlo.reshape %v2035 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2037 = stablehlo.reshape %v2019 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2038 = stablehlo.dot_general %v2036, %v2037, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2039 = stablehlo.reshape %v2038 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2040 = stablehlo.reshape %v2039 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2041 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2042 = stablehlo.pad %v2040, %v2041, low = [0, 0, 320], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v2043 = stablehlo.reshape %v2042 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2044 = stablehlo.add %v2010, %v2043 : tensor<32x75648xf32>
    %v2045 = stablehlo.reshape %v2044 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2046 = stablehlo.dot_general %v2045, %b6_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v2047 = stablehlo.broadcast_in_dim %b6_bo, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2048 = stablehlo.add %v2046, %v2047 : tensor<32x197x384xf32>
    %v2049 = stablehlo.reshape %v2048 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2050 = stablehlo.add %v1798, %v2049 : tensor<32x75648xf32>
    %v2051 = stablehlo.reshape %v2050 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2052 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2053 = stablehlo.constant dense<384.0> : tensor<32x197x384xf32>
    %v2054 = stablehlo.constant dense<1.0e-5> : tensor<32x197x384xf32>
    %v2055 = stablehlo.reduce(%v2051 init: %v2052) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2056 = stablehlo.broadcast_in_dim %v2055, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v2057 = stablehlo.divide %v2056, %v2053 : tensor<32x197x384xf32>
    %v2058 = stablehlo.subtract %v2051, %v2057 : tensor<32x197x384xf32>
    %v2059 = stablehlo.multiply %v2058, %v2058 : tensor<32x197x384xf32>
    %v2060 = stablehlo.reduce(%v2059 init: %v2052) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2061 = stablehlo.broadcast_in_dim %v2060, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v2062 = stablehlo.divide %v2061, %v2053 : tensor<32x197x384xf32>
    %v2063 = stablehlo.add %v2062, %v2054 : tensor<32x197x384xf32>
    %v2064 = stablehlo.rsqrt %v2063 : tensor<32x197x384xf32>
    %v2065 = stablehlo.multiply %v2058, %v2064 : tensor<32x197x384xf32>
    %v2066 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v2067 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v2068 = stablehlo.multiply %v2065, %v2066 : tensor<32x197x384xf32>
    %v2069 = stablehlo.add %v2068, %v2067 : tensor<32x197x384xf32>
    %v2070 = stablehlo.reshape %v2069 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2071 = stablehlo.reshape %v2070 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2072 = stablehlo.broadcast_in_dim %b6_g2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2073 = stablehlo.multiply %v2071, %v2072 : tensor<32x197x384xf32>
    %v2074 = stablehlo.reshape %v2073 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2075 = stablehlo.reshape %v2074 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2076 = stablehlo.broadcast_in_dim %b6_bt2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2077 = stablehlo.add %v2075, %v2076 : tensor<32x197x384xf32>
    %v2078 = stablehlo.reshape %v2077 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2079 = stablehlo.reshape %v2078 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2080 = stablehlo.dot_general %v2079, %b6_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x1536xf32>) -> tensor<32x197x1536xf32>
    %v2081 = stablehlo.broadcast_in_dim %b6_bfc1, dims = [2] : (tensor<1536xf32>) -> tensor<32x197x1536xf32>
    %v2082 = stablehlo.add %v2080, %v2081 : tensor<32x197x1536xf32>
    %v2083 = stablehlo.reshape %v2082 : (tensor<32x197x1536xf32>) -> tensor<32x302592xf32>
    %v2084 = stablehlo.constant dense<0.5> : tensor<32x302592xf32>
    %v2085 = stablehlo.multiply %v2084, %v2083 : tensor<32x302592xf32>
    %v2086 = stablehlo.negate %v2083 : tensor<32x302592xf32>
    %v2087 = stablehlo.constant dense<0.7071067811865476> : tensor<32x302592xf32>
    %v2088 = stablehlo.multiply %v2086, %v2087 : tensor<32x302592xf32>
    %v2089 = chlo.erfc %v2088 : tensor<32x302592xf32> -> tensor<32x302592xf32>
    %v2090 = stablehlo.multiply %v2085, %v2089 : tensor<32x302592xf32>
    %v2091 = stablehlo.reshape %v2090 : (tensor<32x302592xf32>) -> tensor<32x197x1536xf32>
    %v2092 = stablehlo.dot_general %v2091, %b6_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x1536xf32>, tensor<1536x384xf32>) -> tensor<32x197x384xf32>
    %v2093 = stablehlo.broadcast_in_dim %b6_bfc2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2094 = stablehlo.add %v2092, %v2093 : tensor<32x197x384xf32>
    %v2095 = stablehlo.reshape %v2094 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2096 = stablehlo.add %v2050, %v2095 : tensor<32x75648xf32>
    %v2097 = stablehlo.reshape %v2096 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2098 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2099 = stablehlo.constant dense<384.0> : tensor<32x197x384xf32>
    %v2100 = stablehlo.constant dense<1.0e-5> : tensor<32x197x384xf32>
    %v2101 = stablehlo.reduce(%v2097 init: %v2098) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2102 = stablehlo.broadcast_in_dim %v2101, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v2103 = stablehlo.divide %v2102, %v2099 : tensor<32x197x384xf32>
    %v2104 = stablehlo.subtract %v2097, %v2103 : tensor<32x197x384xf32>
    %v2105 = stablehlo.multiply %v2104, %v2104 : tensor<32x197x384xf32>
    %v2106 = stablehlo.reduce(%v2105 init: %v2098) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2107 = stablehlo.broadcast_in_dim %v2106, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v2108 = stablehlo.divide %v2107, %v2099 : tensor<32x197x384xf32>
    %v2109 = stablehlo.add %v2108, %v2100 : tensor<32x197x384xf32>
    %v2110 = stablehlo.rsqrt %v2109 : tensor<32x197x384xf32>
    %v2111 = stablehlo.multiply %v2104, %v2110 : tensor<32x197x384xf32>
    %v2112 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v2113 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v2114 = stablehlo.multiply %v2111, %v2112 : tensor<32x197x384xf32>
    %v2115 = stablehlo.add %v2114, %v2113 : tensor<32x197x384xf32>
    %v2116 = stablehlo.reshape %v2115 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2117 = stablehlo.reshape %v2116 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2118 = stablehlo.broadcast_in_dim %b7_g1, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2119 = stablehlo.multiply %v2117, %v2118 : tensor<32x197x384xf32>
    %v2120 = stablehlo.reshape %v2119 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2121 = stablehlo.reshape %v2120 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2122 = stablehlo.broadcast_in_dim %b7_bt1, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2123 = stablehlo.add %v2121, %v2122 : tensor<32x197x384xf32>
    %v2124 = stablehlo.reshape %v2123 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2125 = stablehlo.reshape %v2124 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2126 = stablehlo.dot_general %v2125, %b7_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v2127 = stablehlo.broadcast_in_dim %b7_bq, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2128 = stablehlo.add %v2126, %v2127 : tensor<32x197x384xf32>
    %v2129 = stablehlo.reshape %v2128 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2130 = stablehlo.reshape %v2124 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2131 = stablehlo.dot_general %v2130, %b7_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v2132 = stablehlo.broadcast_in_dim %b7_bk, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2133 = stablehlo.add %v2131, %v2132 : tensor<32x197x384xf32>
    %v2134 = stablehlo.reshape %v2133 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2135 = stablehlo.reshape %v2124 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2136 = stablehlo.dot_general %v2135, %b7_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v2137 = stablehlo.broadcast_in_dim %b7_bv, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2138 = stablehlo.add %v2136, %v2137 : tensor<32x197x384xf32>
    %v2139 = stablehlo.reshape %v2138 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2140 = stablehlo.reshape %v2129 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2141 = stablehlo.slice %v2140 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2142 = stablehlo.reshape %v2141 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2143 = stablehlo.reshape %v2134 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2144 = stablehlo.slice %v2143 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2145 = stablehlo.reshape %v2144 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2146 = stablehlo.reshape %v2139 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2147 = stablehlo.slice %v2146 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2148 = stablehlo.reshape %v2147 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2149 = stablehlo.reshape %v2145 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2150 = stablehlo.transpose %v2149, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2151 = stablehlo.reshape %v2150 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2152 = stablehlo.reshape %v2142 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2153 = stablehlo.reshape %v2151 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2154 = stablehlo.dot_general %v2152, %v2153, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2155 = stablehlo.reshape %v2154 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2156 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2157 = stablehlo.multiply %v2155, %v2156 : tensor<32x38809xf32>
    %v2158 = stablehlo.reshape %v2157 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2159 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2160 = stablehlo.exponential %v2158 : tensor<32x197x197xf32>
    %v2161 = stablehlo.reduce(%v2160 init: %v2159) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2162 = stablehlo.broadcast_in_dim %v2161, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2163 = stablehlo.divide %v2160, %v2162 : tensor<32x197x197xf32>
    %v2164 = stablehlo.reshape %v2163 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2165 = stablehlo.reshape %v2164 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2166 = stablehlo.reshape %v2148 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2167 = stablehlo.dot_general %v2165, %v2166, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2168 = stablehlo.reshape %v2167 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2169 = stablehlo.reshape %v2168 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2170 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2171 = stablehlo.pad %v2169, %v2170, low = [0, 0, 0], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v2172 = stablehlo.reshape %v2171 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2173 = stablehlo.reshape %v2129 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2174 = stablehlo.slice %v2173 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2175 = stablehlo.reshape %v2174 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2176 = stablehlo.reshape %v2134 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2177 = stablehlo.slice %v2176 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2178 = stablehlo.reshape %v2177 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2179 = stablehlo.reshape %v2139 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2180 = stablehlo.slice %v2179 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2181 = stablehlo.reshape %v2180 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2182 = stablehlo.reshape %v2178 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2183 = stablehlo.transpose %v2182, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2184 = stablehlo.reshape %v2183 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2185 = stablehlo.reshape %v2175 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2186 = stablehlo.reshape %v2184 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2187 = stablehlo.dot_general %v2185, %v2186, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2188 = stablehlo.reshape %v2187 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2189 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2190 = stablehlo.multiply %v2188, %v2189 : tensor<32x38809xf32>
    %v2191 = stablehlo.reshape %v2190 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2192 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2193 = stablehlo.exponential %v2191 : tensor<32x197x197xf32>
    %v2194 = stablehlo.reduce(%v2193 init: %v2192) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2195 = stablehlo.broadcast_in_dim %v2194, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2196 = stablehlo.divide %v2193, %v2195 : tensor<32x197x197xf32>
    %v2197 = stablehlo.reshape %v2196 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2198 = stablehlo.reshape %v2197 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2199 = stablehlo.reshape %v2181 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2200 = stablehlo.dot_general %v2198, %v2199, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2201 = stablehlo.reshape %v2200 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2202 = stablehlo.reshape %v2201 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2203 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2204 = stablehlo.pad %v2202, %v2203, low = [0, 0, 64], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v2205 = stablehlo.reshape %v2204 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2206 = stablehlo.add %v2172, %v2205 : tensor<32x75648xf32>
    %v2207 = stablehlo.reshape %v2129 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2208 = stablehlo.slice %v2207 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2209 = stablehlo.reshape %v2208 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2210 = stablehlo.reshape %v2134 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2211 = stablehlo.slice %v2210 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2212 = stablehlo.reshape %v2211 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2213 = stablehlo.reshape %v2139 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2214 = stablehlo.slice %v2213 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2215 = stablehlo.reshape %v2214 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2216 = stablehlo.reshape %v2212 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2217 = stablehlo.transpose %v2216, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2218 = stablehlo.reshape %v2217 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2219 = stablehlo.reshape %v2209 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2220 = stablehlo.reshape %v2218 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2221 = stablehlo.dot_general %v2219, %v2220, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2222 = stablehlo.reshape %v2221 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2223 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2224 = stablehlo.multiply %v2222, %v2223 : tensor<32x38809xf32>
    %v2225 = stablehlo.reshape %v2224 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2226 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2227 = stablehlo.exponential %v2225 : tensor<32x197x197xf32>
    %v2228 = stablehlo.reduce(%v2227 init: %v2226) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2229 = stablehlo.broadcast_in_dim %v2228, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2230 = stablehlo.divide %v2227, %v2229 : tensor<32x197x197xf32>
    %v2231 = stablehlo.reshape %v2230 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2232 = stablehlo.reshape %v2231 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2233 = stablehlo.reshape %v2215 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2234 = stablehlo.dot_general %v2232, %v2233, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2235 = stablehlo.reshape %v2234 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2236 = stablehlo.reshape %v2235 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2237 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2238 = stablehlo.pad %v2236, %v2237, low = [0, 0, 128], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v2239 = stablehlo.reshape %v2238 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2240 = stablehlo.add %v2206, %v2239 : tensor<32x75648xf32>
    %v2241 = stablehlo.reshape %v2129 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2242 = stablehlo.slice %v2241 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2243 = stablehlo.reshape %v2242 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2244 = stablehlo.reshape %v2134 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2245 = stablehlo.slice %v2244 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2246 = stablehlo.reshape %v2245 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2247 = stablehlo.reshape %v2139 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2248 = stablehlo.slice %v2247 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2249 = stablehlo.reshape %v2248 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2250 = stablehlo.reshape %v2246 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2251 = stablehlo.transpose %v2250, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2252 = stablehlo.reshape %v2251 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2253 = stablehlo.reshape %v2243 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2254 = stablehlo.reshape %v2252 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2255 = stablehlo.dot_general %v2253, %v2254, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2256 = stablehlo.reshape %v2255 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2257 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2258 = stablehlo.multiply %v2256, %v2257 : tensor<32x38809xf32>
    %v2259 = stablehlo.reshape %v2258 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2260 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2261 = stablehlo.exponential %v2259 : tensor<32x197x197xf32>
    %v2262 = stablehlo.reduce(%v2261 init: %v2260) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2263 = stablehlo.broadcast_in_dim %v2262, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2264 = stablehlo.divide %v2261, %v2263 : tensor<32x197x197xf32>
    %v2265 = stablehlo.reshape %v2264 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2266 = stablehlo.reshape %v2265 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2267 = stablehlo.reshape %v2249 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2268 = stablehlo.dot_general %v2266, %v2267, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2269 = stablehlo.reshape %v2268 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2270 = stablehlo.reshape %v2269 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2271 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2272 = stablehlo.pad %v2270, %v2271, low = [0, 0, 192], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v2273 = stablehlo.reshape %v2272 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2274 = stablehlo.add %v2240, %v2273 : tensor<32x75648xf32>
    %v2275 = stablehlo.reshape %v2129 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2276 = stablehlo.slice %v2275 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2277 = stablehlo.reshape %v2276 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2278 = stablehlo.reshape %v2134 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2279 = stablehlo.slice %v2278 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2280 = stablehlo.reshape %v2279 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2281 = stablehlo.reshape %v2139 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2282 = stablehlo.slice %v2281 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2283 = stablehlo.reshape %v2282 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2284 = stablehlo.reshape %v2280 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2285 = stablehlo.transpose %v2284, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2286 = stablehlo.reshape %v2285 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2287 = stablehlo.reshape %v2277 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2288 = stablehlo.reshape %v2286 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2289 = stablehlo.dot_general %v2287, %v2288, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2290 = stablehlo.reshape %v2289 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2291 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2292 = stablehlo.multiply %v2290, %v2291 : tensor<32x38809xf32>
    %v2293 = stablehlo.reshape %v2292 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2294 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2295 = stablehlo.exponential %v2293 : tensor<32x197x197xf32>
    %v2296 = stablehlo.reduce(%v2295 init: %v2294) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2297 = stablehlo.broadcast_in_dim %v2296, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2298 = stablehlo.divide %v2295, %v2297 : tensor<32x197x197xf32>
    %v2299 = stablehlo.reshape %v2298 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2300 = stablehlo.reshape %v2299 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2301 = stablehlo.reshape %v2283 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2302 = stablehlo.dot_general %v2300, %v2301, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2303 = stablehlo.reshape %v2302 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2304 = stablehlo.reshape %v2303 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2305 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2306 = stablehlo.pad %v2304, %v2305, low = [0, 0, 256], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v2307 = stablehlo.reshape %v2306 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2308 = stablehlo.add %v2274, %v2307 : tensor<32x75648xf32>
    %v2309 = stablehlo.reshape %v2129 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2310 = stablehlo.slice %v2309 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2311 = stablehlo.reshape %v2310 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2312 = stablehlo.reshape %v2134 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2313 = stablehlo.slice %v2312 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2314 = stablehlo.reshape %v2313 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2315 = stablehlo.reshape %v2139 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2316 = stablehlo.slice %v2315 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
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
    %v2340 = stablehlo.pad %v2338, %v2339, low = [0, 0, 320], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v2341 = stablehlo.reshape %v2340 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2342 = stablehlo.add %v2308, %v2341 : tensor<32x75648xf32>
    %v2343 = stablehlo.reshape %v2342 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2344 = stablehlo.dot_general %v2343, %b7_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v2345 = stablehlo.broadcast_in_dim %b7_bo, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2346 = stablehlo.add %v2344, %v2345 : tensor<32x197x384xf32>
    %v2347 = stablehlo.reshape %v2346 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2348 = stablehlo.add %v2096, %v2347 : tensor<32x75648xf32>
    %v2349 = stablehlo.reshape %v2348 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2350 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2351 = stablehlo.constant dense<384.0> : tensor<32x197x384xf32>
    %v2352 = stablehlo.constant dense<1.0e-5> : tensor<32x197x384xf32>
    %v2353 = stablehlo.reduce(%v2349 init: %v2350) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2354 = stablehlo.broadcast_in_dim %v2353, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v2355 = stablehlo.divide %v2354, %v2351 : tensor<32x197x384xf32>
    %v2356 = stablehlo.subtract %v2349, %v2355 : tensor<32x197x384xf32>
    %v2357 = stablehlo.multiply %v2356, %v2356 : tensor<32x197x384xf32>
    %v2358 = stablehlo.reduce(%v2357 init: %v2350) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2359 = stablehlo.broadcast_in_dim %v2358, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v2360 = stablehlo.divide %v2359, %v2351 : tensor<32x197x384xf32>
    %v2361 = stablehlo.add %v2360, %v2352 : tensor<32x197x384xf32>
    %v2362 = stablehlo.rsqrt %v2361 : tensor<32x197x384xf32>
    %v2363 = stablehlo.multiply %v2356, %v2362 : tensor<32x197x384xf32>
    %v2364 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v2365 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v2366 = stablehlo.multiply %v2363, %v2364 : tensor<32x197x384xf32>
    %v2367 = stablehlo.add %v2366, %v2365 : tensor<32x197x384xf32>
    %v2368 = stablehlo.reshape %v2367 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2369 = stablehlo.reshape %v2368 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2370 = stablehlo.broadcast_in_dim %b7_g2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2371 = stablehlo.multiply %v2369, %v2370 : tensor<32x197x384xf32>
    %v2372 = stablehlo.reshape %v2371 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2373 = stablehlo.reshape %v2372 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2374 = stablehlo.broadcast_in_dim %b7_bt2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2375 = stablehlo.add %v2373, %v2374 : tensor<32x197x384xf32>
    %v2376 = stablehlo.reshape %v2375 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2377 = stablehlo.reshape %v2376 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2378 = stablehlo.dot_general %v2377, %b7_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x1536xf32>) -> tensor<32x197x1536xf32>
    %v2379 = stablehlo.broadcast_in_dim %b7_bfc1, dims = [2] : (tensor<1536xf32>) -> tensor<32x197x1536xf32>
    %v2380 = stablehlo.add %v2378, %v2379 : tensor<32x197x1536xf32>
    %v2381 = stablehlo.reshape %v2380 : (tensor<32x197x1536xf32>) -> tensor<32x302592xf32>
    %v2382 = stablehlo.constant dense<0.5> : tensor<32x302592xf32>
    %v2383 = stablehlo.multiply %v2382, %v2381 : tensor<32x302592xf32>
    %v2384 = stablehlo.negate %v2381 : tensor<32x302592xf32>
    %v2385 = stablehlo.constant dense<0.7071067811865476> : tensor<32x302592xf32>
    %v2386 = stablehlo.multiply %v2384, %v2385 : tensor<32x302592xf32>
    %v2387 = chlo.erfc %v2386 : tensor<32x302592xf32> -> tensor<32x302592xf32>
    %v2388 = stablehlo.multiply %v2383, %v2387 : tensor<32x302592xf32>
    %v2389 = stablehlo.reshape %v2388 : (tensor<32x302592xf32>) -> tensor<32x197x1536xf32>
    %v2390 = stablehlo.dot_general %v2389, %b7_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x1536xf32>, tensor<1536x384xf32>) -> tensor<32x197x384xf32>
    %v2391 = stablehlo.broadcast_in_dim %b7_bfc2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2392 = stablehlo.add %v2390, %v2391 : tensor<32x197x384xf32>
    %v2393 = stablehlo.reshape %v2392 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2394 = stablehlo.add %v2348, %v2393 : tensor<32x75648xf32>
    %v2395 = stablehlo.reshape %v2394 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2396 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2397 = stablehlo.constant dense<384.0> : tensor<32x197x384xf32>
    %v2398 = stablehlo.constant dense<1.0e-5> : tensor<32x197x384xf32>
    %v2399 = stablehlo.reduce(%v2395 init: %v2396) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2400 = stablehlo.broadcast_in_dim %v2399, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v2401 = stablehlo.divide %v2400, %v2397 : tensor<32x197x384xf32>
    %v2402 = stablehlo.subtract %v2395, %v2401 : tensor<32x197x384xf32>
    %v2403 = stablehlo.multiply %v2402, %v2402 : tensor<32x197x384xf32>
    %v2404 = stablehlo.reduce(%v2403 init: %v2396) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2405 = stablehlo.broadcast_in_dim %v2404, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v2406 = stablehlo.divide %v2405, %v2397 : tensor<32x197x384xf32>
    %v2407 = stablehlo.add %v2406, %v2398 : tensor<32x197x384xf32>
    %v2408 = stablehlo.rsqrt %v2407 : tensor<32x197x384xf32>
    %v2409 = stablehlo.multiply %v2402, %v2408 : tensor<32x197x384xf32>
    %v2410 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v2411 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v2412 = stablehlo.multiply %v2409, %v2410 : tensor<32x197x384xf32>
    %v2413 = stablehlo.add %v2412, %v2411 : tensor<32x197x384xf32>
    %v2414 = stablehlo.reshape %v2413 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2415 = stablehlo.reshape %v2414 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2416 = stablehlo.broadcast_in_dim %b8_g1, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2417 = stablehlo.multiply %v2415, %v2416 : tensor<32x197x384xf32>
    %v2418 = stablehlo.reshape %v2417 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2419 = stablehlo.reshape %v2418 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2420 = stablehlo.broadcast_in_dim %b8_bt1, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2421 = stablehlo.add %v2419, %v2420 : tensor<32x197x384xf32>
    %v2422 = stablehlo.reshape %v2421 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2423 = stablehlo.reshape %v2422 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2424 = stablehlo.dot_general %v2423, %b8_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v2425 = stablehlo.broadcast_in_dim %b8_bq, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2426 = stablehlo.add %v2424, %v2425 : tensor<32x197x384xf32>
    %v2427 = stablehlo.reshape %v2426 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2428 = stablehlo.reshape %v2422 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2429 = stablehlo.dot_general %v2428, %b8_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v2430 = stablehlo.broadcast_in_dim %b8_bk, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2431 = stablehlo.add %v2429, %v2430 : tensor<32x197x384xf32>
    %v2432 = stablehlo.reshape %v2431 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2433 = stablehlo.reshape %v2422 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2434 = stablehlo.dot_general %v2433, %b8_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v2435 = stablehlo.broadcast_in_dim %b8_bv, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2436 = stablehlo.add %v2434, %v2435 : tensor<32x197x384xf32>
    %v2437 = stablehlo.reshape %v2436 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2438 = stablehlo.reshape %v2427 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2439 = stablehlo.slice %v2438 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2440 = stablehlo.reshape %v2439 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2441 = stablehlo.reshape %v2432 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2442 = stablehlo.slice %v2441 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2443 = stablehlo.reshape %v2442 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2444 = stablehlo.reshape %v2437 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2445 = stablehlo.slice %v2444 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2446 = stablehlo.reshape %v2445 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2447 = stablehlo.reshape %v2443 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2448 = stablehlo.transpose %v2447, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2449 = stablehlo.reshape %v2448 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2450 = stablehlo.reshape %v2440 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2451 = stablehlo.reshape %v2449 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2452 = stablehlo.dot_general %v2450, %v2451, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2453 = stablehlo.reshape %v2452 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2454 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2455 = stablehlo.multiply %v2453, %v2454 : tensor<32x38809xf32>
    %v2456 = stablehlo.reshape %v2455 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2457 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2458 = stablehlo.exponential %v2456 : tensor<32x197x197xf32>
    %v2459 = stablehlo.reduce(%v2458 init: %v2457) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2460 = stablehlo.broadcast_in_dim %v2459, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2461 = stablehlo.divide %v2458, %v2460 : tensor<32x197x197xf32>
    %v2462 = stablehlo.reshape %v2461 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2463 = stablehlo.reshape %v2462 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2464 = stablehlo.reshape %v2446 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2465 = stablehlo.dot_general %v2463, %v2464, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2466 = stablehlo.reshape %v2465 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2467 = stablehlo.reshape %v2466 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2468 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2469 = stablehlo.pad %v2467, %v2468, low = [0, 0, 0], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v2470 = stablehlo.reshape %v2469 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2471 = stablehlo.reshape %v2427 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2472 = stablehlo.slice %v2471 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2473 = stablehlo.reshape %v2472 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2474 = stablehlo.reshape %v2432 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2475 = stablehlo.slice %v2474 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2476 = stablehlo.reshape %v2475 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2477 = stablehlo.reshape %v2437 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2478 = stablehlo.slice %v2477 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2479 = stablehlo.reshape %v2478 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2480 = stablehlo.reshape %v2476 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2481 = stablehlo.transpose %v2480, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2482 = stablehlo.reshape %v2481 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2483 = stablehlo.reshape %v2473 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2484 = stablehlo.reshape %v2482 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2485 = stablehlo.dot_general %v2483, %v2484, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2486 = stablehlo.reshape %v2485 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2487 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2488 = stablehlo.multiply %v2486, %v2487 : tensor<32x38809xf32>
    %v2489 = stablehlo.reshape %v2488 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2490 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2491 = stablehlo.exponential %v2489 : tensor<32x197x197xf32>
    %v2492 = stablehlo.reduce(%v2491 init: %v2490) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2493 = stablehlo.broadcast_in_dim %v2492, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2494 = stablehlo.divide %v2491, %v2493 : tensor<32x197x197xf32>
    %v2495 = stablehlo.reshape %v2494 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2496 = stablehlo.reshape %v2495 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2497 = stablehlo.reshape %v2479 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2498 = stablehlo.dot_general %v2496, %v2497, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2499 = stablehlo.reshape %v2498 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2500 = stablehlo.reshape %v2499 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2501 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2502 = stablehlo.pad %v2500, %v2501, low = [0, 0, 64], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v2503 = stablehlo.reshape %v2502 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2504 = stablehlo.add %v2470, %v2503 : tensor<32x75648xf32>
    %v2505 = stablehlo.reshape %v2427 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2506 = stablehlo.slice %v2505 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2507 = stablehlo.reshape %v2506 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2508 = stablehlo.reshape %v2432 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2509 = stablehlo.slice %v2508 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2510 = stablehlo.reshape %v2509 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2511 = stablehlo.reshape %v2437 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2512 = stablehlo.slice %v2511 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2513 = stablehlo.reshape %v2512 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2514 = stablehlo.reshape %v2510 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2515 = stablehlo.transpose %v2514, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2516 = stablehlo.reshape %v2515 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2517 = stablehlo.reshape %v2507 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2518 = stablehlo.reshape %v2516 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2519 = stablehlo.dot_general %v2517, %v2518, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2520 = stablehlo.reshape %v2519 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2521 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2522 = stablehlo.multiply %v2520, %v2521 : tensor<32x38809xf32>
    %v2523 = stablehlo.reshape %v2522 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2524 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2525 = stablehlo.exponential %v2523 : tensor<32x197x197xf32>
    %v2526 = stablehlo.reduce(%v2525 init: %v2524) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2527 = stablehlo.broadcast_in_dim %v2526, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2528 = stablehlo.divide %v2525, %v2527 : tensor<32x197x197xf32>
    %v2529 = stablehlo.reshape %v2528 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2530 = stablehlo.reshape %v2529 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2531 = stablehlo.reshape %v2513 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2532 = stablehlo.dot_general %v2530, %v2531, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2533 = stablehlo.reshape %v2532 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2534 = stablehlo.reshape %v2533 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2535 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2536 = stablehlo.pad %v2534, %v2535, low = [0, 0, 128], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v2537 = stablehlo.reshape %v2536 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2538 = stablehlo.add %v2504, %v2537 : tensor<32x75648xf32>
    %v2539 = stablehlo.reshape %v2427 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2540 = stablehlo.slice %v2539 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2541 = stablehlo.reshape %v2540 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2542 = stablehlo.reshape %v2432 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2543 = stablehlo.slice %v2542 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2544 = stablehlo.reshape %v2543 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2545 = stablehlo.reshape %v2437 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2546 = stablehlo.slice %v2545 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2547 = stablehlo.reshape %v2546 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2548 = stablehlo.reshape %v2544 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2549 = stablehlo.transpose %v2548, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2550 = stablehlo.reshape %v2549 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2551 = stablehlo.reshape %v2541 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2552 = stablehlo.reshape %v2550 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2553 = stablehlo.dot_general %v2551, %v2552, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2554 = stablehlo.reshape %v2553 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2555 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2556 = stablehlo.multiply %v2554, %v2555 : tensor<32x38809xf32>
    %v2557 = stablehlo.reshape %v2556 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2558 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2559 = stablehlo.exponential %v2557 : tensor<32x197x197xf32>
    %v2560 = stablehlo.reduce(%v2559 init: %v2558) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2561 = stablehlo.broadcast_in_dim %v2560, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2562 = stablehlo.divide %v2559, %v2561 : tensor<32x197x197xf32>
    %v2563 = stablehlo.reshape %v2562 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2564 = stablehlo.reshape %v2563 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2565 = stablehlo.reshape %v2547 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2566 = stablehlo.dot_general %v2564, %v2565, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2567 = stablehlo.reshape %v2566 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2568 = stablehlo.reshape %v2567 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2569 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2570 = stablehlo.pad %v2568, %v2569, low = [0, 0, 192], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v2571 = stablehlo.reshape %v2570 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2572 = stablehlo.add %v2538, %v2571 : tensor<32x75648xf32>
    %v2573 = stablehlo.reshape %v2427 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2574 = stablehlo.slice %v2573 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2575 = stablehlo.reshape %v2574 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2576 = stablehlo.reshape %v2432 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2577 = stablehlo.slice %v2576 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2578 = stablehlo.reshape %v2577 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2579 = stablehlo.reshape %v2437 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2580 = stablehlo.slice %v2579 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2581 = stablehlo.reshape %v2580 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2582 = stablehlo.reshape %v2578 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2583 = stablehlo.transpose %v2582, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2584 = stablehlo.reshape %v2583 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2585 = stablehlo.reshape %v2575 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2586 = stablehlo.reshape %v2584 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2587 = stablehlo.dot_general %v2585, %v2586, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2588 = stablehlo.reshape %v2587 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2589 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2590 = stablehlo.multiply %v2588, %v2589 : tensor<32x38809xf32>
    %v2591 = stablehlo.reshape %v2590 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2592 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2593 = stablehlo.exponential %v2591 : tensor<32x197x197xf32>
    %v2594 = stablehlo.reduce(%v2593 init: %v2592) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2595 = stablehlo.broadcast_in_dim %v2594, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2596 = stablehlo.divide %v2593, %v2595 : tensor<32x197x197xf32>
    %v2597 = stablehlo.reshape %v2596 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2598 = stablehlo.reshape %v2597 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2599 = stablehlo.reshape %v2581 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2600 = stablehlo.dot_general %v2598, %v2599, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2601 = stablehlo.reshape %v2600 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2602 = stablehlo.reshape %v2601 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2603 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2604 = stablehlo.pad %v2602, %v2603, low = [0, 0, 256], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v2605 = stablehlo.reshape %v2604 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2606 = stablehlo.add %v2572, %v2605 : tensor<32x75648xf32>
    %v2607 = stablehlo.reshape %v2427 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2608 = stablehlo.slice %v2607 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2609 = stablehlo.reshape %v2608 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2610 = stablehlo.reshape %v2432 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2611 = stablehlo.slice %v2610 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2612 = stablehlo.reshape %v2611 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2613 = stablehlo.reshape %v2437 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2614 = stablehlo.slice %v2613 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2615 = stablehlo.reshape %v2614 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2616 = stablehlo.reshape %v2612 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2617 = stablehlo.transpose %v2616, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2618 = stablehlo.reshape %v2617 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2619 = stablehlo.reshape %v2609 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2620 = stablehlo.reshape %v2618 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2621 = stablehlo.dot_general %v2619, %v2620, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2622 = stablehlo.reshape %v2621 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2623 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2624 = stablehlo.multiply %v2622, %v2623 : tensor<32x38809xf32>
    %v2625 = stablehlo.reshape %v2624 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2626 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2627 = stablehlo.exponential %v2625 : tensor<32x197x197xf32>
    %v2628 = stablehlo.reduce(%v2627 init: %v2626) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2629 = stablehlo.broadcast_in_dim %v2628, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2630 = stablehlo.divide %v2627, %v2629 : tensor<32x197x197xf32>
    %v2631 = stablehlo.reshape %v2630 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2632 = stablehlo.reshape %v2631 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2633 = stablehlo.reshape %v2615 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2634 = stablehlo.dot_general %v2632, %v2633, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2635 = stablehlo.reshape %v2634 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2636 = stablehlo.reshape %v2635 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2637 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2638 = stablehlo.pad %v2636, %v2637, low = [0, 0, 320], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v2639 = stablehlo.reshape %v2638 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2640 = stablehlo.add %v2606, %v2639 : tensor<32x75648xf32>
    %v2641 = stablehlo.reshape %v2640 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2642 = stablehlo.dot_general %v2641, %b8_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v2643 = stablehlo.broadcast_in_dim %b8_bo, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2644 = stablehlo.add %v2642, %v2643 : tensor<32x197x384xf32>
    %v2645 = stablehlo.reshape %v2644 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2646 = stablehlo.add %v2394, %v2645 : tensor<32x75648xf32>
    %v2647 = stablehlo.reshape %v2646 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2648 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2649 = stablehlo.constant dense<384.0> : tensor<32x197x384xf32>
    %v2650 = stablehlo.constant dense<1.0e-5> : tensor<32x197x384xf32>
    %v2651 = stablehlo.reduce(%v2647 init: %v2648) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2652 = stablehlo.broadcast_in_dim %v2651, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v2653 = stablehlo.divide %v2652, %v2649 : tensor<32x197x384xf32>
    %v2654 = stablehlo.subtract %v2647, %v2653 : tensor<32x197x384xf32>
    %v2655 = stablehlo.multiply %v2654, %v2654 : tensor<32x197x384xf32>
    %v2656 = stablehlo.reduce(%v2655 init: %v2648) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2657 = stablehlo.broadcast_in_dim %v2656, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v2658 = stablehlo.divide %v2657, %v2649 : tensor<32x197x384xf32>
    %v2659 = stablehlo.add %v2658, %v2650 : tensor<32x197x384xf32>
    %v2660 = stablehlo.rsqrt %v2659 : tensor<32x197x384xf32>
    %v2661 = stablehlo.multiply %v2654, %v2660 : tensor<32x197x384xf32>
    %v2662 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v2663 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v2664 = stablehlo.multiply %v2661, %v2662 : tensor<32x197x384xf32>
    %v2665 = stablehlo.add %v2664, %v2663 : tensor<32x197x384xf32>
    %v2666 = stablehlo.reshape %v2665 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2667 = stablehlo.reshape %v2666 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2668 = stablehlo.broadcast_in_dim %b8_g2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2669 = stablehlo.multiply %v2667, %v2668 : tensor<32x197x384xf32>
    %v2670 = stablehlo.reshape %v2669 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2671 = stablehlo.reshape %v2670 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2672 = stablehlo.broadcast_in_dim %b8_bt2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2673 = stablehlo.add %v2671, %v2672 : tensor<32x197x384xf32>
    %v2674 = stablehlo.reshape %v2673 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2675 = stablehlo.reshape %v2674 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2676 = stablehlo.dot_general %v2675, %b8_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x1536xf32>) -> tensor<32x197x1536xf32>
    %v2677 = stablehlo.broadcast_in_dim %b8_bfc1, dims = [2] : (tensor<1536xf32>) -> tensor<32x197x1536xf32>
    %v2678 = stablehlo.add %v2676, %v2677 : tensor<32x197x1536xf32>
    %v2679 = stablehlo.reshape %v2678 : (tensor<32x197x1536xf32>) -> tensor<32x302592xf32>
    %v2680 = stablehlo.constant dense<0.5> : tensor<32x302592xf32>
    %v2681 = stablehlo.multiply %v2680, %v2679 : tensor<32x302592xf32>
    %v2682 = stablehlo.negate %v2679 : tensor<32x302592xf32>
    %v2683 = stablehlo.constant dense<0.7071067811865476> : tensor<32x302592xf32>
    %v2684 = stablehlo.multiply %v2682, %v2683 : tensor<32x302592xf32>
    %v2685 = chlo.erfc %v2684 : tensor<32x302592xf32> -> tensor<32x302592xf32>
    %v2686 = stablehlo.multiply %v2681, %v2685 : tensor<32x302592xf32>
    %v2687 = stablehlo.reshape %v2686 : (tensor<32x302592xf32>) -> tensor<32x197x1536xf32>
    %v2688 = stablehlo.dot_general %v2687, %b8_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x1536xf32>, tensor<1536x384xf32>) -> tensor<32x197x384xf32>
    %v2689 = stablehlo.broadcast_in_dim %b8_bfc2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2690 = stablehlo.add %v2688, %v2689 : tensor<32x197x384xf32>
    %v2691 = stablehlo.reshape %v2690 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2692 = stablehlo.add %v2646, %v2691 : tensor<32x75648xf32>
    %v2693 = stablehlo.reshape %v2692 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2694 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2695 = stablehlo.constant dense<384.0> : tensor<32x197x384xf32>
    %v2696 = stablehlo.constant dense<1.0e-5> : tensor<32x197x384xf32>
    %v2697 = stablehlo.reduce(%v2693 init: %v2694) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2698 = stablehlo.broadcast_in_dim %v2697, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v2699 = stablehlo.divide %v2698, %v2695 : tensor<32x197x384xf32>
    %v2700 = stablehlo.subtract %v2693, %v2699 : tensor<32x197x384xf32>
    %v2701 = stablehlo.multiply %v2700, %v2700 : tensor<32x197x384xf32>
    %v2702 = stablehlo.reduce(%v2701 init: %v2694) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2703 = stablehlo.broadcast_in_dim %v2702, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v2704 = stablehlo.divide %v2703, %v2695 : tensor<32x197x384xf32>
    %v2705 = stablehlo.add %v2704, %v2696 : tensor<32x197x384xf32>
    %v2706 = stablehlo.rsqrt %v2705 : tensor<32x197x384xf32>
    %v2707 = stablehlo.multiply %v2700, %v2706 : tensor<32x197x384xf32>
    %v2708 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v2709 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v2710 = stablehlo.multiply %v2707, %v2708 : tensor<32x197x384xf32>
    %v2711 = stablehlo.add %v2710, %v2709 : tensor<32x197x384xf32>
    %v2712 = stablehlo.reshape %v2711 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2713 = stablehlo.reshape %v2712 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2714 = stablehlo.broadcast_in_dim %b9_g1, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2715 = stablehlo.multiply %v2713, %v2714 : tensor<32x197x384xf32>
    %v2716 = stablehlo.reshape %v2715 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2717 = stablehlo.reshape %v2716 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2718 = stablehlo.broadcast_in_dim %b9_bt1, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2719 = stablehlo.add %v2717, %v2718 : tensor<32x197x384xf32>
    %v2720 = stablehlo.reshape %v2719 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2721 = stablehlo.reshape %v2720 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2722 = stablehlo.dot_general %v2721, %b9_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v2723 = stablehlo.broadcast_in_dim %b9_bq, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2724 = stablehlo.add %v2722, %v2723 : tensor<32x197x384xf32>
    %v2725 = stablehlo.reshape %v2724 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2726 = stablehlo.reshape %v2720 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2727 = stablehlo.dot_general %v2726, %b9_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v2728 = stablehlo.broadcast_in_dim %b9_bk, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2729 = stablehlo.add %v2727, %v2728 : tensor<32x197x384xf32>
    %v2730 = stablehlo.reshape %v2729 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2731 = stablehlo.reshape %v2720 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2732 = stablehlo.dot_general %v2731, %b9_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v2733 = stablehlo.broadcast_in_dim %b9_bv, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2734 = stablehlo.add %v2732, %v2733 : tensor<32x197x384xf32>
    %v2735 = stablehlo.reshape %v2734 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2736 = stablehlo.reshape %v2725 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2737 = stablehlo.slice %v2736 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2738 = stablehlo.reshape %v2737 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2739 = stablehlo.reshape %v2730 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2740 = stablehlo.slice %v2739 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2741 = stablehlo.reshape %v2740 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2742 = stablehlo.reshape %v2735 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2743 = stablehlo.slice %v2742 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2744 = stablehlo.reshape %v2743 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2745 = stablehlo.reshape %v2741 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2746 = stablehlo.transpose %v2745, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2747 = stablehlo.reshape %v2746 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2748 = stablehlo.reshape %v2738 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2749 = stablehlo.reshape %v2747 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2750 = stablehlo.dot_general %v2748, %v2749, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2751 = stablehlo.reshape %v2750 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2752 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2753 = stablehlo.multiply %v2751, %v2752 : tensor<32x38809xf32>
    %v2754 = stablehlo.reshape %v2753 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2755 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2756 = stablehlo.exponential %v2754 : tensor<32x197x197xf32>
    %v2757 = stablehlo.reduce(%v2756 init: %v2755) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2758 = stablehlo.broadcast_in_dim %v2757, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2759 = stablehlo.divide %v2756, %v2758 : tensor<32x197x197xf32>
    %v2760 = stablehlo.reshape %v2759 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2761 = stablehlo.reshape %v2760 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2762 = stablehlo.reshape %v2744 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2763 = stablehlo.dot_general %v2761, %v2762, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2764 = stablehlo.reshape %v2763 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2765 = stablehlo.reshape %v2764 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2766 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2767 = stablehlo.pad %v2765, %v2766, low = [0, 0, 0], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v2768 = stablehlo.reshape %v2767 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2769 = stablehlo.reshape %v2725 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2770 = stablehlo.slice %v2769 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2771 = stablehlo.reshape %v2770 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2772 = stablehlo.reshape %v2730 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2773 = stablehlo.slice %v2772 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2774 = stablehlo.reshape %v2773 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2775 = stablehlo.reshape %v2735 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2776 = stablehlo.slice %v2775 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2777 = stablehlo.reshape %v2776 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2778 = stablehlo.reshape %v2774 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2779 = stablehlo.transpose %v2778, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2780 = stablehlo.reshape %v2779 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2781 = stablehlo.reshape %v2771 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2782 = stablehlo.reshape %v2780 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2783 = stablehlo.dot_general %v2781, %v2782, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2784 = stablehlo.reshape %v2783 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2785 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2786 = stablehlo.multiply %v2784, %v2785 : tensor<32x38809xf32>
    %v2787 = stablehlo.reshape %v2786 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2788 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2789 = stablehlo.exponential %v2787 : tensor<32x197x197xf32>
    %v2790 = stablehlo.reduce(%v2789 init: %v2788) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2791 = stablehlo.broadcast_in_dim %v2790, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2792 = stablehlo.divide %v2789, %v2791 : tensor<32x197x197xf32>
    %v2793 = stablehlo.reshape %v2792 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2794 = stablehlo.reshape %v2793 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2795 = stablehlo.reshape %v2777 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2796 = stablehlo.dot_general %v2794, %v2795, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2797 = stablehlo.reshape %v2796 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2798 = stablehlo.reshape %v2797 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2799 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2800 = stablehlo.pad %v2798, %v2799, low = [0, 0, 64], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v2801 = stablehlo.reshape %v2800 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2802 = stablehlo.add %v2768, %v2801 : tensor<32x75648xf32>
    %v2803 = stablehlo.reshape %v2725 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2804 = stablehlo.slice %v2803 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2805 = stablehlo.reshape %v2804 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2806 = stablehlo.reshape %v2730 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2807 = stablehlo.slice %v2806 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2808 = stablehlo.reshape %v2807 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2809 = stablehlo.reshape %v2735 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2810 = stablehlo.slice %v2809 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2811 = stablehlo.reshape %v2810 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2812 = stablehlo.reshape %v2808 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2813 = stablehlo.transpose %v2812, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2814 = stablehlo.reshape %v2813 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2815 = stablehlo.reshape %v2805 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2816 = stablehlo.reshape %v2814 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2817 = stablehlo.dot_general %v2815, %v2816, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2818 = stablehlo.reshape %v2817 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2819 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2820 = stablehlo.multiply %v2818, %v2819 : tensor<32x38809xf32>
    %v2821 = stablehlo.reshape %v2820 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2822 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2823 = stablehlo.exponential %v2821 : tensor<32x197x197xf32>
    %v2824 = stablehlo.reduce(%v2823 init: %v2822) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2825 = stablehlo.broadcast_in_dim %v2824, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2826 = stablehlo.divide %v2823, %v2825 : tensor<32x197x197xf32>
    %v2827 = stablehlo.reshape %v2826 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2828 = stablehlo.reshape %v2827 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2829 = stablehlo.reshape %v2811 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2830 = stablehlo.dot_general %v2828, %v2829, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2831 = stablehlo.reshape %v2830 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2832 = stablehlo.reshape %v2831 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2833 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2834 = stablehlo.pad %v2832, %v2833, low = [0, 0, 128], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v2835 = stablehlo.reshape %v2834 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2836 = stablehlo.add %v2802, %v2835 : tensor<32x75648xf32>
    %v2837 = stablehlo.reshape %v2725 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2838 = stablehlo.slice %v2837 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2839 = stablehlo.reshape %v2838 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2840 = stablehlo.reshape %v2730 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2841 = stablehlo.slice %v2840 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2842 = stablehlo.reshape %v2841 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2843 = stablehlo.reshape %v2735 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2844 = stablehlo.slice %v2843 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2845 = stablehlo.reshape %v2844 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2846 = stablehlo.reshape %v2842 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2847 = stablehlo.transpose %v2846, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2848 = stablehlo.reshape %v2847 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2849 = stablehlo.reshape %v2839 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2850 = stablehlo.reshape %v2848 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2851 = stablehlo.dot_general %v2849, %v2850, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2852 = stablehlo.reshape %v2851 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2853 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2854 = stablehlo.multiply %v2852, %v2853 : tensor<32x38809xf32>
    %v2855 = stablehlo.reshape %v2854 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2856 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2857 = stablehlo.exponential %v2855 : tensor<32x197x197xf32>
    %v2858 = stablehlo.reduce(%v2857 init: %v2856) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2859 = stablehlo.broadcast_in_dim %v2858, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2860 = stablehlo.divide %v2857, %v2859 : tensor<32x197x197xf32>
    %v2861 = stablehlo.reshape %v2860 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2862 = stablehlo.reshape %v2861 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2863 = stablehlo.reshape %v2845 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2864 = stablehlo.dot_general %v2862, %v2863, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2865 = stablehlo.reshape %v2864 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2866 = stablehlo.reshape %v2865 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2867 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2868 = stablehlo.pad %v2866, %v2867, low = [0, 0, 192], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v2869 = stablehlo.reshape %v2868 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2870 = stablehlo.add %v2836, %v2869 : tensor<32x75648xf32>
    %v2871 = stablehlo.reshape %v2725 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2872 = stablehlo.slice %v2871 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2873 = stablehlo.reshape %v2872 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2874 = stablehlo.reshape %v2730 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2875 = stablehlo.slice %v2874 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2876 = stablehlo.reshape %v2875 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2877 = stablehlo.reshape %v2735 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2878 = stablehlo.slice %v2877 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2879 = stablehlo.reshape %v2878 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2880 = stablehlo.reshape %v2876 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2881 = stablehlo.transpose %v2880, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2882 = stablehlo.reshape %v2881 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2883 = stablehlo.reshape %v2873 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2884 = stablehlo.reshape %v2882 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2885 = stablehlo.dot_general %v2883, %v2884, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2886 = stablehlo.reshape %v2885 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2887 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2888 = stablehlo.multiply %v2886, %v2887 : tensor<32x38809xf32>
    %v2889 = stablehlo.reshape %v2888 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2890 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2891 = stablehlo.exponential %v2889 : tensor<32x197x197xf32>
    %v2892 = stablehlo.reduce(%v2891 init: %v2890) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2893 = stablehlo.broadcast_in_dim %v2892, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2894 = stablehlo.divide %v2891, %v2893 : tensor<32x197x197xf32>
    %v2895 = stablehlo.reshape %v2894 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2896 = stablehlo.reshape %v2895 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2897 = stablehlo.reshape %v2879 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2898 = stablehlo.dot_general %v2896, %v2897, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2899 = stablehlo.reshape %v2898 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2900 = stablehlo.reshape %v2899 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2901 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2902 = stablehlo.pad %v2900, %v2901, low = [0, 0, 256], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v2903 = stablehlo.reshape %v2902 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2904 = stablehlo.add %v2870, %v2903 : tensor<32x75648xf32>
    %v2905 = stablehlo.reshape %v2725 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2906 = stablehlo.slice %v2905 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2907 = stablehlo.reshape %v2906 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2908 = stablehlo.reshape %v2730 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2909 = stablehlo.slice %v2908 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2910 = stablehlo.reshape %v2909 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2911 = stablehlo.reshape %v2735 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2912 = stablehlo.slice %v2911 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2913 = stablehlo.reshape %v2912 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2914 = stablehlo.reshape %v2910 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2915 = stablehlo.transpose %v2914, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2916 = stablehlo.reshape %v2915 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2917 = stablehlo.reshape %v2907 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2918 = stablehlo.reshape %v2916 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2919 = stablehlo.dot_general %v2917, %v2918, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2920 = stablehlo.reshape %v2919 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2921 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2922 = stablehlo.multiply %v2920, %v2921 : tensor<32x38809xf32>
    %v2923 = stablehlo.reshape %v2922 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2924 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2925 = stablehlo.exponential %v2923 : tensor<32x197x197xf32>
    %v2926 = stablehlo.reduce(%v2925 init: %v2924) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2927 = stablehlo.broadcast_in_dim %v2926, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2928 = stablehlo.divide %v2925, %v2927 : tensor<32x197x197xf32>
    %v2929 = stablehlo.reshape %v2928 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2930 = stablehlo.reshape %v2929 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2931 = stablehlo.reshape %v2913 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2932 = stablehlo.dot_general %v2930, %v2931, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2933 = stablehlo.reshape %v2932 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2934 = stablehlo.reshape %v2933 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2935 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2936 = stablehlo.pad %v2934, %v2935, low = [0, 0, 320], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v2937 = stablehlo.reshape %v2936 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2938 = stablehlo.add %v2904, %v2937 : tensor<32x75648xf32>
    %v2939 = stablehlo.reshape %v2938 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2940 = stablehlo.dot_general %v2939, %b9_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v2941 = stablehlo.broadcast_in_dim %b9_bo, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2942 = stablehlo.add %v2940, %v2941 : tensor<32x197x384xf32>
    %v2943 = stablehlo.reshape %v2942 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2944 = stablehlo.add %v2692, %v2943 : tensor<32x75648xf32>
    %v2945 = stablehlo.reshape %v2944 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2946 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2947 = stablehlo.constant dense<384.0> : tensor<32x197x384xf32>
    %v2948 = stablehlo.constant dense<1.0e-5> : tensor<32x197x384xf32>
    %v2949 = stablehlo.reduce(%v2945 init: %v2946) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2950 = stablehlo.broadcast_in_dim %v2949, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v2951 = stablehlo.divide %v2950, %v2947 : tensor<32x197x384xf32>
    %v2952 = stablehlo.subtract %v2945, %v2951 : tensor<32x197x384xf32>
    %v2953 = stablehlo.multiply %v2952, %v2952 : tensor<32x197x384xf32>
    %v2954 = stablehlo.reduce(%v2953 init: %v2946) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2955 = stablehlo.broadcast_in_dim %v2954, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v2956 = stablehlo.divide %v2955, %v2947 : tensor<32x197x384xf32>
    %v2957 = stablehlo.add %v2956, %v2948 : tensor<32x197x384xf32>
    %v2958 = stablehlo.rsqrt %v2957 : tensor<32x197x384xf32>
    %v2959 = stablehlo.multiply %v2952, %v2958 : tensor<32x197x384xf32>
    %v2960 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v2961 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v2962 = stablehlo.multiply %v2959, %v2960 : tensor<32x197x384xf32>
    %v2963 = stablehlo.add %v2962, %v2961 : tensor<32x197x384xf32>
    %v2964 = stablehlo.reshape %v2963 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2965 = stablehlo.reshape %v2964 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2966 = stablehlo.broadcast_in_dim %b9_g2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2967 = stablehlo.multiply %v2965, %v2966 : tensor<32x197x384xf32>
    %v2968 = stablehlo.reshape %v2967 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2969 = stablehlo.reshape %v2968 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2970 = stablehlo.broadcast_in_dim %b9_bt2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2971 = stablehlo.add %v2969, %v2970 : tensor<32x197x384xf32>
    %v2972 = stablehlo.reshape %v2971 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2973 = stablehlo.reshape %v2972 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2974 = stablehlo.dot_general %v2973, %b9_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x1536xf32>) -> tensor<32x197x1536xf32>
    %v2975 = stablehlo.broadcast_in_dim %b9_bfc1, dims = [2] : (tensor<1536xf32>) -> tensor<32x197x1536xf32>
    %v2976 = stablehlo.add %v2974, %v2975 : tensor<32x197x1536xf32>
    %v2977 = stablehlo.reshape %v2976 : (tensor<32x197x1536xf32>) -> tensor<32x302592xf32>
    %v2978 = stablehlo.constant dense<0.5> : tensor<32x302592xf32>
    %v2979 = stablehlo.multiply %v2978, %v2977 : tensor<32x302592xf32>
    %v2980 = stablehlo.negate %v2977 : tensor<32x302592xf32>
    %v2981 = stablehlo.constant dense<0.7071067811865476> : tensor<32x302592xf32>
    %v2982 = stablehlo.multiply %v2980, %v2981 : tensor<32x302592xf32>
    %v2983 = chlo.erfc %v2982 : tensor<32x302592xf32> -> tensor<32x302592xf32>
    %v2984 = stablehlo.multiply %v2979, %v2983 : tensor<32x302592xf32>
    %v2985 = stablehlo.reshape %v2984 : (tensor<32x302592xf32>) -> tensor<32x197x1536xf32>
    %v2986 = stablehlo.dot_general %v2985, %b9_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x1536xf32>, tensor<1536x384xf32>) -> tensor<32x197x384xf32>
    %v2987 = stablehlo.broadcast_in_dim %b9_bfc2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2988 = stablehlo.add %v2986, %v2987 : tensor<32x197x384xf32>
    %v2989 = stablehlo.reshape %v2988 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2990 = stablehlo.add %v2944, %v2989 : tensor<32x75648xf32>
    %v2991 = stablehlo.reshape %v2990 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2992 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2993 = stablehlo.constant dense<384.0> : tensor<32x197x384xf32>
    %v2994 = stablehlo.constant dense<1.0e-5> : tensor<32x197x384xf32>
    %v2995 = stablehlo.reduce(%v2991 init: %v2992) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2996 = stablehlo.broadcast_in_dim %v2995, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v2997 = stablehlo.divide %v2996, %v2993 : tensor<32x197x384xf32>
    %v2998 = stablehlo.subtract %v2991, %v2997 : tensor<32x197x384xf32>
    %v2999 = stablehlo.multiply %v2998, %v2998 : tensor<32x197x384xf32>
    %v3000 = stablehlo.reduce(%v2999 init: %v2992) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3001 = stablehlo.broadcast_in_dim %v3000, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v3002 = stablehlo.divide %v3001, %v2993 : tensor<32x197x384xf32>
    %v3003 = stablehlo.add %v3002, %v2994 : tensor<32x197x384xf32>
    %v3004 = stablehlo.rsqrt %v3003 : tensor<32x197x384xf32>
    %v3005 = stablehlo.multiply %v2998, %v3004 : tensor<32x197x384xf32>
    %v3006 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v3007 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v3008 = stablehlo.multiply %v3005, %v3006 : tensor<32x197x384xf32>
    %v3009 = stablehlo.add %v3008, %v3007 : tensor<32x197x384xf32>
    %v3010 = stablehlo.reshape %v3009 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3011 = stablehlo.reshape %v3010 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3012 = stablehlo.broadcast_in_dim %b10_g1, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v3013 = stablehlo.multiply %v3011, %v3012 : tensor<32x197x384xf32>
    %v3014 = stablehlo.reshape %v3013 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3015 = stablehlo.reshape %v3014 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3016 = stablehlo.broadcast_in_dim %b10_bt1, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v3017 = stablehlo.add %v3015, %v3016 : tensor<32x197x384xf32>
    %v3018 = stablehlo.reshape %v3017 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3019 = stablehlo.reshape %v3018 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3020 = stablehlo.dot_general %v3019, %b10_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v3021 = stablehlo.broadcast_in_dim %b10_bq, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v3022 = stablehlo.add %v3020, %v3021 : tensor<32x197x384xf32>
    %v3023 = stablehlo.reshape %v3022 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3024 = stablehlo.reshape %v3018 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3025 = stablehlo.dot_general %v3024, %b10_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v3026 = stablehlo.broadcast_in_dim %b10_bk, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v3027 = stablehlo.add %v3025, %v3026 : tensor<32x197x384xf32>
    %v3028 = stablehlo.reshape %v3027 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3029 = stablehlo.reshape %v3018 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3030 = stablehlo.dot_general %v3029, %b10_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v3031 = stablehlo.broadcast_in_dim %b10_bv, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v3032 = stablehlo.add %v3030, %v3031 : tensor<32x197x384xf32>
    %v3033 = stablehlo.reshape %v3032 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3034 = stablehlo.reshape %v3023 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3035 = stablehlo.slice %v3034 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3036 = stablehlo.reshape %v3035 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3037 = stablehlo.reshape %v3028 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3038 = stablehlo.slice %v3037 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3039 = stablehlo.reshape %v3038 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3040 = stablehlo.reshape %v3033 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3041 = stablehlo.slice %v3040 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3042 = stablehlo.reshape %v3041 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3043 = stablehlo.reshape %v3039 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3044 = stablehlo.transpose %v3043, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3045 = stablehlo.reshape %v3044 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3046 = stablehlo.reshape %v3036 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3047 = stablehlo.reshape %v3045 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3048 = stablehlo.dot_general %v3046, %v3047, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3049 = stablehlo.reshape %v3048 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3050 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3051 = stablehlo.multiply %v3049, %v3050 : tensor<32x38809xf32>
    %v3052 = stablehlo.reshape %v3051 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3053 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3054 = stablehlo.exponential %v3052 : tensor<32x197x197xf32>
    %v3055 = stablehlo.reduce(%v3054 init: %v3053) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3056 = stablehlo.broadcast_in_dim %v3055, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3057 = stablehlo.divide %v3054, %v3056 : tensor<32x197x197xf32>
    %v3058 = stablehlo.reshape %v3057 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3059 = stablehlo.reshape %v3058 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3060 = stablehlo.reshape %v3042 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3061 = stablehlo.dot_general %v3059, %v3060, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3062 = stablehlo.reshape %v3061 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3063 = stablehlo.reshape %v3062 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3064 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3065 = stablehlo.pad %v3063, %v3064, low = [0, 0, 0], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v3066 = stablehlo.reshape %v3065 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3067 = stablehlo.reshape %v3023 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3068 = stablehlo.slice %v3067 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3069 = stablehlo.reshape %v3068 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3070 = stablehlo.reshape %v3028 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3071 = stablehlo.slice %v3070 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3072 = stablehlo.reshape %v3071 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3073 = stablehlo.reshape %v3033 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3074 = stablehlo.slice %v3073 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3075 = stablehlo.reshape %v3074 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3076 = stablehlo.reshape %v3072 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3077 = stablehlo.transpose %v3076, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3078 = stablehlo.reshape %v3077 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3079 = stablehlo.reshape %v3069 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3080 = stablehlo.reshape %v3078 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3081 = stablehlo.dot_general %v3079, %v3080, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3082 = stablehlo.reshape %v3081 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3083 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3084 = stablehlo.multiply %v3082, %v3083 : tensor<32x38809xf32>
    %v3085 = stablehlo.reshape %v3084 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3086 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3087 = stablehlo.exponential %v3085 : tensor<32x197x197xf32>
    %v3088 = stablehlo.reduce(%v3087 init: %v3086) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3089 = stablehlo.broadcast_in_dim %v3088, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3090 = stablehlo.divide %v3087, %v3089 : tensor<32x197x197xf32>
    %v3091 = stablehlo.reshape %v3090 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3092 = stablehlo.reshape %v3091 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3093 = stablehlo.reshape %v3075 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3094 = stablehlo.dot_general %v3092, %v3093, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3095 = stablehlo.reshape %v3094 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3096 = stablehlo.reshape %v3095 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3097 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3098 = stablehlo.pad %v3096, %v3097, low = [0, 0, 64], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v3099 = stablehlo.reshape %v3098 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3100 = stablehlo.add %v3066, %v3099 : tensor<32x75648xf32>
    %v3101 = stablehlo.reshape %v3023 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3102 = stablehlo.slice %v3101 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3103 = stablehlo.reshape %v3102 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3104 = stablehlo.reshape %v3028 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3105 = stablehlo.slice %v3104 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3106 = stablehlo.reshape %v3105 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3107 = stablehlo.reshape %v3033 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3108 = stablehlo.slice %v3107 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3109 = stablehlo.reshape %v3108 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3110 = stablehlo.reshape %v3106 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3111 = stablehlo.transpose %v3110, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3112 = stablehlo.reshape %v3111 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3113 = stablehlo.reshape %v3103 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3114 = stablehlo.reshape %v3112 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3115 = stablehlo.dot_general %v3113, %v3114, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3116 = stablehlo.reshape %v3115 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3117 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3118 = stablehlo.multiply %v3116, %v3117 : tensor<32x38809xf32>
    %v3119 = stablehlo.reshape %v3118 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3120 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3121 = stablehlo.exponential %v3119 : tensor<32x197x197xf32>
    %v3122 = stablehlo.reduce(%v3121 init: %v3120) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3123 = stablehlo.broadcast_in_dim %v3122, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3124 = stablehlo.divide %v3121, %v3123 : tensor<32x197x197xf32>
    %v3125 = stablehlo.reshape %v3124 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3126 = stablehlo.reshape %v3125 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3127 = stablehlo.reshape %v3109 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3128 = stablehlo.dot_general %v3126, %v3127, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3129 = stablehlo.reshape %v3128 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3130 = stablehlo.reshape %v3129 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3131 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3132 = stablehlo.pad %v3130, %v3131, low = [0, 0, 128], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v3133 = stablehlo.reshape %v3132 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3134 = stablehlo.add %v3100, %v3133 : tensor<32x75648xf32>
    %v3135 = stablehlo.reshape %v3023 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3136 = stablehlo.slice %v3135 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3137 = stablehlo.reshape %v3136 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3138 = stablehlo.reshape %v3028 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3139 = stablehlo.slice %v3138 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3140 = stablehlo.reshape %v3139 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3141 = stablehlo.reshape %v3033 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3142 = stablehlo.slice %v3141 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
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
    %v3166 = stablehlo.pad %v3164, %v3165, low = [0, 0, 192], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v3167 = stablehlo.reshape %v3166 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3168 = stablehlo.add %v3134, %v3167 : tensor<32x75648xf32>
    %v3169 = stablehlo.reshape %v3023 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3170 = stablehlo.slice %v3169 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3171 = stablehlo.reshape %v3170 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3172 = stablehlo.reshape %v3028 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3173 = stablehlo.slice %v3172 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3174 = stablehlo.reshape %v3173 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3175 = stablehlo.reshape %v3033 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3176 = stablehlo.slice %v3175 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
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
    %v3200 = stablehlo.pad %v3198, %v3199, low = [0, 0, 256], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v3201 = stablehlo.reshape %v3200 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3202 = stablehlo.add %v3168, %v3201 : tensor<32x75648xf32>
    %v3203 = stablehlo.reshape %v3023 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3204 = stablehlo.slice %v3203 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3205 = stablehlo.reshape %v3204 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3206 = stablehlo.reshape %v3028 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3207 = stablehlo.slice %v3206 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3208 = stablehlo.reshape %v3207 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3209 = stablehlo.reshape %v3033 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3210 = stablehlo.slice %v3209 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
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
    %v3234 = stablehlo.pad %v3232, %v3233, low = [0, 0, 320], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v3235 = stablehlo.reshape %v3234 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3236 = stablehlo.add %v3202, %v3235 : tensor<32x75648xf32>
    %v3237 = stablehlo.reshape %v3236 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3238 = stablehlo.dot_general %v3237, %b10_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v3239 = stablehlo.broadcast_in_dim %b10_bo, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v3240 = stablehlo.add %v3238, %v3239 : tensor<32x197x384xf32>
    %v3241 = stablehlo.reshape %v3240 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3242 = stablehlo.add %v2990, %v3241 : tensor<32x75648xf32>
    %v3243 = stablehlo.reshape %v3242 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3244 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3245 = stablehlo.constant dense<384.0> : tensor<32x197x384xf32>
    %v3246 = stablehlo.constant dense<1.0e-5> : tensor<32x197x384xf32>
    %v3247 = stablehlo.reduce(%v3243 init: %v3244) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3248 = stablehlo.broadcast_in_dim %v3247, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v3249 = stablehlo.divide %v3248, %v3245 : tensor<32x197x384xf32>
    %v3250 = stablehlo.subtract %v3243, %v3249 : tensor<32x197x384xf32>
    %v3251 = stablehlo.multiply %v3250, %v3250 : tensor<32x197x384xf32>
    %v3252 = stablehlo.reduce(%v3251 init: %v3244) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3253 = stablehlo.broadcast_in_dim %v3252, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v3254 = stablehlo.divide %v3253, %v3245 : tensor<32x197x384xf32>
    %v3255 = stablehlo.add %v3254, %v3246 : tensor<32x197x384xf32>
    %v3256 = stablehlo.rsqrt %v3255 : tensor<32x197x384xf32>
    %v3257 = stablehlo.multiply %v3250, %v3256 : tensor<32x197x384xf32>
    %v3258 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v3259 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v3260 = stablehlo.multiply %v3257, %v3258 : tensor<32x197x384xf32>
    %v3261 = stablehlo.add %v3260, %v3259 : tensor<32x197x384xf32>
    %v3262 = stablehlo.reshape %v3261 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3263 = stablehlo.reshape %v3262 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3264 = stablehlo.broadcast_in_dim %b10_g2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v3265 = stablehlo.multiply %v3263, %v3264 : tensor<32x197x384xf32>
    %v3266 = stablehlo.reshape %v3265 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3267 = stablehlo.reshape %v3266 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3268 = stablehlo.broadcast_in_dim %b10_bt2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v3269 = stablehlo.add %v3267, %v3268 : tensor<32x197x384xf32>
    %v3270 = stablehlo.reshape %v3269 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3271 = stablehlo.reshape %v3270 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3272 = stablehlo.dot_general %v3271, %b10_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x1536xf32>) -> tensor<32x197x1536xf32>
    %v3273 = stablehlo.broadcast_in_dim %b10_bfc1, dims = [2] : (tensor<1536xf32>) -> tensor<32x197x1536xf32>
    %v3274 = stablehlo.add %v3272, %v3273 : tensor<32x197x1536xf32>
    %v3275 = stablehlo.reshape %v3274 : (tensor<32x197x1536xf32>) -> tensor<32x302592xf32>
    %v3276 = stablehlo.constant dense<0.5> : tensor<32x302592xf32>
    %v3277 = stablehlo.multiply %v3276, %v3275 : tensor<32x302592xf32>
    %v3278 = stablehlo.negate %v3275 : tensor<32x302592xf32>
    %v3279 = stablehlo.constant dense<0.7071067811865476> : tensor<32x302592xf32>
    %v3280 = stablehlo.multiply %v3278, %v3279 : tensor<32x302592xf32>
    %v3281 = chlo.erfc %v3280 : tensor<32x302592xf32> -> tensor<32x302592xf32>
    %v3282 = stablehlo.multiply %v3277, %v3281 : tensor<32x302592xf32>
    %v3283 = stablehlo.reshape %v3282 : (tensor<32x302592xf32>) -> tensor<32x197x1536xf32>
    %v3284 = stablehlo.dot_general %v3283, %b10_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x1536xf32>, tensor<1536x384xf32>) -> tensor<32x197x384xf32>
    %v3285 = stablehlo.broadcast_in_dim %b10_bfc2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v3286 = stablehlo.add %v3284, %v3285 : tensor<32x197x384xf32>
    %v3287 = stablehlo.reshape %v3286 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3288 = stablehlo.add %v3242, %v3287 : tensor<32x75648xf32>
    %v3289 = stablehlo.reshape %v3288 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3290 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3291 = stablehlo.constant dense<384.0> : tensor<32x197x384xf32>
    %v3292 = stablehlo.constant dense<1.0e-5> : tensor<32x197x384xf32>
    %v3293 = stablehlo.reduce(%v3289 init: %v3290) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3294 = stablehlo.broadcast_in_dim %v3293, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v3295 = stablehlo.divide %v3294, %v3291 : tensor<32x197x384xf32>
    %v3296 = stablehlo.subtract %v3289, %v3295 : tensor<32x197x384xf32>
    %v3297 = stablehlo.multiply %v3296, %v3296 : tensor<32x197x384xf32>
    %v3298 = stablehlo.reduce(%v3297 init: %v3290) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3299 = stablehlo.broadcast_in_dim %v3298, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v3300 = stablehlo.divide %v3299, %v3291 : tensor<32x197x384xf32>
    %v3301 = stablehlo.add %v3300, %v3292 : tensor<32x197x384xf32>
    %v3302 = stablehlo.rsqrt %v3301 : tensor<32x197x384xf32>
    %v3303 = stablehlo.multiply %v3296, %v3302 : tensor<32x197x384xf32>
    %v3304 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v3305 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v3306 = stablehlo.multiply %v3303, %v3304 : tensor<32x197x384xf32>
    %v3307 = stablehlo.add %v3306, %v3305 : tensor<32x197x384xf32>
    %v3308 = stablehlo.reshape %v3307 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3309 = stablehlo.reshape %v3308 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3310 = stablehlo.broadcast_in_dim %b11_g1, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v3311 = stablehlo.multiply %v3309, %v3310 : tensor<32x197x384xf32>
    %v3312 = stablehlo.reshape %v3311 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3313 = stablehlo.reshape %v3312 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3314 = stablehlo.broadcast_in_dim %b11_bt1, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v3315 = stablehlo.add %v3313, %v3314 : tensor<32x197x384xf32>
    %v3316 = stablehlo.reshape %v3315 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3317 = stablehlo.reshape %v3316 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3318 = stablehlo.dot_general %v3317, %b11_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v3319 = stablehlo.broadcast_in_dim %b11_bq, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v3320 = stablehlo.add %v3318, %v3319 : tensor<32x197x384xf32>
    %v3321 = stablehlo.reshape %v3320 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3322 = stablehlo.reshape %v3316 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3323 = stablehlo.dot_general %v3322, %b11_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v3324 = stablehlo.broadcast_in_dim %b11_bk, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v3325 = stablehlo.add %v3323, %v3324 : tensor<32x197x384xf32>
    %v3326 = stablehlo.reshape %v3325 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3327 = stablehlo.reshape %v3316 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3328 = stablehlo.dot_general %v3327, %b11_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v3329 = stablehlo.broadcast_in_dim %b11_bv, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v3330 = stablehlo.add %v3328, %v3329 : tensor<32x197x384xf32>
    %v3331 = stablehlo.reshape %v3330 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3332 = stablehlo.reshape %v3321 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3333 = stablehlo.slice %v3332 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3334 = stablehlo.reshape %v3333 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3335 = stablehlo.reshape %v3326 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3336 = stablehlo.slice %v3335 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3337 = stablehlo.reshape %v3336 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3338 = stablehlo.reshape %v3331 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3339 = stablehlo.slice %v3338 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3340 = stablehlo.reshape %v3339 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3341 = stablehlo.reshape %v3337 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3342 = stablehlo.transpose %v3341, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3343 = stablehlo.reshape %v3342 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3344 = stablehlo.reshape %v3334 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3345 = stablehlo.reshape %v3343 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3346 = stablehlo.dot_general %v3344, %v3345, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3347 = stablehlo.reshape %v3346 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3348 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3349 = stablehlo.multiply %v3347, %v3348 : tensor<32x38809xf32>
    %v3350 = stablehlo.reshape %v3349 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3351 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3352 = stablehlo.exponential %v3350 : tensor<32x197x197xf32>
    %v3353 = stablehlo.reduce(%v3352 init: %v3351) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3354 = stablehlo.broadcast_in_dim %v3353, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3355 = stablehlo.divide %v3352, %v3354 : tensor<32x197x197xf32>
    %v3356 = stablehlo.reshape %v3355 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3357 = stablehlo.reshape %v3356 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3358 = stablehlo.reshape %v3340 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3359 = stablehlo.dot_general %v3357, %v3358, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3360 = stablehlo.reshape %v3359 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3361 = stablehlo.reshape %v3360 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3362 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3363 = stablehlo.pad %v3361, %v3362, low = [0, 0, 0], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v3364 = stablehlo.reshape %v3363 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3365 = stablehlo.reshape %v3321 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3366 = stablehlo.slice %v3365 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3367 = stablehlo.reshape %v3366 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3368 = stablehlo.reshape %v3326 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3369 = stablehlo.slice %v3368 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3370 = stablehlo.reshape %v3369 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3371 = stablehlo.reshape %v3331 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3372 = stablehlo.slice %v3371 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3373 = stablehlo.reshape %v3372 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3374 = stablehlo.reshape %v3370 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3375 = stablehlo.transpose %v3374, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3376 = stablehlo.reshape %v3375 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3377 = stablehlo.reshape %v3367 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3378 = stablehlo.reshape %v3376 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3379 = stablehlo.dot_general %v3377, %v3378, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3380 = stablehlo.reshape %v3379 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3381 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3382 = stablehlo.multiply %v3380, %v3381 : tensor<32x38809xf32>
    %v3383 = stablehlo.reshape %v3382 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3384 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3385 = stablehlo.exponential %v3383 : tensor<32x197x197xf32>
    %v3386 = stablehlo.reduce(%v3385 init: %v3384) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3387 = stablehlo.broadcast_in_dim %v3386, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3388 = stablehlo.divide %v3385, %v3387 : tensor<32x197x197xf32>
    %v3389 = stablehlo.reshape %v3388 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3390 = stablehlo.reshape %v3389 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3391 = stablehlo.reshape %v3373 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3392 = stablehlo.dot_general %v3390, %v3391, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3393 = stablehlo.reshape %v3392 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3394 = stablehlo.reshape %v3393 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3395 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3396 = stablehlo.pad %v3394, %v3395, low = [0, 0, 64], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v3397 = stablehlo.reshape %v3396 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3398 = stablehlo.add %v3364, %v3397 : tensor<32x75648xf32>
    %v3399 = stablehlo.reshape %v3321 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3400 = stablehlo.slice %v3399 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3401 = stablehlo.reshape %v3400 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3402 = stablehlo.reshape %v3326 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3403 = stablehlo.slice %v3402 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3404 = stablehlo.reshape %v3403 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3405 = stablehlo.reshape %v3331 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3406 = stablehlo.slice %v3405 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3407 = stablehlo.reshape %v3406 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3408 = stablehlo.reshape %v3404 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3409 = stablehlo.transpose %v3408, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3410 = stablehlo.reshape %v3409 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3411 = stablehlo.reshape %v3401 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3412 = stablehlo.reshape %v3410 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3413 = stablehlo.dot_general %v3411, %v3412, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3414 = stablehlo.reshape %v3413 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3415 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3416 = stablehlo.multiply %v3414, %v3415 : tensor<32x38809xf32>
    %v3417 = stablehlo.reshape %v3416 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3418 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3419 = stablehlo.exponential %v3417 : tensor<32x197x197xf32>
    %v3420 = stablehlo.reduce(%v3419 init: %v3418) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3421 = stablehlo.broadcast_in_dim %v3420, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3422 = stablehlo.divide %v3419, %v3421 : tensor<32x197x197xf32>
    %v3423 = stablehlo.reshape %v3422 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3424 = stablehlo.reshape %v3423 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3425 = stablehlo.reshape %v3407 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3426 = stablehlo.dot_general %v3424, %v3425, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3427 = stablehlo.reshape %v3426 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3428 = stablehlo.reshape %v3427 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3429 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3430 = stablehlo.pad %v3428, %v3429, low = [0, 0, 128], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v3431 = stablehlo.reshape %v3430 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3432 = stablehlo.add %v3398, %v3431 : tensor<32x75648xf32>
    %v3433 = stablehlo.reshape %v3321 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3434 = stablehlo.slice %v3433 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3435 = stablehlo.reshape %v3434 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3436 = stablehlo.reshape %v3326 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3437 = stablehlo.slice %v3436 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3438 = stablehlo.reshape %v3437 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3439 = stablehlo.reshape %v3331 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3440 = stablehlo.slice %v3439 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3441 = stablehlo.reshape %v3440 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3442 = stablehlo.reshape %v3438 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3443 = stablehlo.transpose %v3442, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3444 = stablehlo.reshape %v3443 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3445 = stablehlo.reshape %v3435 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3446 = stablehlo.reshape %v3444 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3447 = stablehlo.dot_general %v3445, %v3446, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3448 = stablehlo.reshape %v3447 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3449 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3450 = stablehlo.multiply %v3448, %v3449 : tensor<32x38809xf32>
    %v3451 = stablehlo.reshape %v3450 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3452 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3453 = stablehlo.exponential %v3451 : tensor<32x197x197xf32>
    %v3454 = stablehlo.reduce(%v3453 init: %v3452) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3455 = stablehlo.broadcast_in_dim %v3454, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3456 = stablehlo.divide %v3453, %v3455 : tensor<32x197x197xf32>
    %v3457 = stablehlo.reshape %v3456 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3458 = stablehlo.reshape %v3457 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3459 = stablehlo.reshape %v3441 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3460 = stablehlo.dot_general %v3458, %v3459, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3461 = stablehlo.reshape %v3460 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3462 = stablehlo.reshape %v3461 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3463 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3464 = stablehlo.pad %v3462, %v3463, low = [0, 0, 192], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v3465 = stablehlo.reshape %v3464 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3466 = stablehlo.add %v3432, %v3465 : tensor<32x75648xf32>
    %v3467 = stablehlo.reshape %v3321 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3468 = stablehlo.slice %v3467 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3469 = stablehlo.reshape %v3468 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3470 = stablehlo.reshape %v3326 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3471 = stablehlo.slice %v3470 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3472 = stablehlo.reshape %v3471 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3473 = stablehlo.reshape %v3331 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3474 = stablehlo.slice %v3473 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
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
    %v3486 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3487 = stablehlo.exponential %v3485 : tensor<32x197x197xf32>
    %v3488 = stablehlo.reduce(%v3487 init: %v3486) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3489 = stablehlo.broadcast_in_dim %v3488, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3490 = stablehlo.divide %v3487, %v3489 : tensor<32x197x197xf32>
    %v3491 = stablehlo.reshape %v3490 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3492 = stablehlo.reshape %v3491 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3493 = stablehlo.reshape %v3475 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3494 = stablehlo.dot_general %v3492, %v3493, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3495 = stablehlo.reshape %v3494 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3496 = stablehlo.reshape %v3495 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3497 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3498 = stablehlo.pad %v3496, %v3497, low = [0, 0, 256], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v3499 = stablehlo.reshape %v3498 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3500 = stablehlo.add %v3466, %v3499 : tensor<32x75648xf32>
    %v3501 = stablehlo.reshape %v3321 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3502 = stablehlo.slice %v3501 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3503 = stablehlo.reshape %v3502 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3504 = stablehlo.reshape %v3326 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3505 = stablehlo.slice %v3504 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3506 = stablehlo.reshape %v3505 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3507 = stablehlo.reshape %v3331 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3508 = stablehlo.slice %v3507 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3509 = stablehlo.reshape %v3508 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3510 = stablehlo.reshape %v3506 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3511 = stablehlo.transpose %v3510, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3512 = stablehlo.reshape %v3511 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3513 = stablehlo.reshape %v3503 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3514 = stablehlo.reshape %v3512 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3515 = stablehlo.dot_general %v3513, %v3514, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3516 = stablehlo.reshape %v3515 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3517 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3518 = stablehlo.multiply %v3516, %v3517 : tensor<32x38809xf32>
    %v3519 = stablehlo.reshape %v3518 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3520 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3521 = stablehlo.exponential %v3519 : tensor<32x197x197xf32>
    %v3522 = stablehlo.reduce(%v3521 init: %v3520) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3523 = stablehlo.broadcast_in_dim %v3522, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3524 = stablehlo.divide %v3521, %v3523 : tensor<32x197x197xf32>
    %v3525 = stablehlo.reshape %v3524 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3526 = stablehlo.reshape %v3525 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3527 = stablehlo.reshape %v3509 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3528 = stablehlo.dot_general %v3526, %v3527, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3529 = stablehlo.reshape %v3528 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3530 = stablehlo.reshape %v3529 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3531 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3532 = stablehlo.pad %v3530, %v3531, low = [0, 0, 320], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v3533 = stablehlo.reshape %v3532 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3534 = stablehlo.add %v3500, %v3533 : tensor<32x75648xf32>
    %v3535 = stablehlo.reshape %v3534 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3536 = stablehlo.dot_general %v3535, %b11_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v3537 = stablehlo.broadcast_in_dim %b11_bo, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v3538 = stablehlo.add %v3536, %v3537 : tensor<32x197x384xf32>
    %v3539 = stablehlo.reshape %v3538 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3540 = stablehlo.add %v3288, %v3539 : tensor<32x75648xf32>
    %v3541 = stablehlo.reshape %v3540 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3542 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3543 = stablehlo.constant dense<384.0> : tensor<32x197x384xf32>
    %v3544 = stablehlo.constant dense<1.0e-5> : tensor<32x197x384xf32>
    %v3545 = stablehlo.reduce(%v3541 init: %v3542) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3546 = stablehlo.broadcast_in_dim %v3545, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v3547 = stablehlo.divide %v3546, %v3543 : tensor<32x197x384xf32>
    %v3548 = stablehlo.subtract %v3541, %v3547 : tensor<32x197x384xf32>
    %v3549 = stablehlo.multiply %v3548, %v3548 : tensor<32x197x384xf32>
    %v3550 = stablehlo.reduce(%v3549 init: %v3542) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3551 = stablehlo.broadcast_in_dim %v3550, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v3552 = stablehlo.divide %v3551, %v3543 : tensor<32x197x384xf32>
    %v3553 = stablehlo.add %v3552, %v3544 : tensor<32x197x384xf32>
    %v3554 = stablehlo.rsqrt %v3553 : tensor<32x197x384xf32>
    %v3555 = stablehlo.multiply %v3548, %v3554 : tensor<32x197x384xf32>
    %v3556 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v3557 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v3558 = stablehlo.multiply %v3555, %v3556 : tensor<32x197x384xf32>
    %v3559 = stablehlo.add %v3558, %v3557 : tensor<32x197x384xf32>
    %v3560 = stablehlo.reshape %v3559 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3561 = stablehlo.reshape %v3560 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3562 = stablehlo.broadcast_in_dim %b11_g2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v3563 = stablehlo.multiply %v3561, %v3562 : tensor<32x197x384xf32>
    %v3564 = stablehlo.reshape %v3563 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3565 = stablehlo.reshape %v3564 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3566 = stablehlo.broadcast_in_dim %b11_bt2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v3567 = stablehlo.add %v3565, %v3566 : tensor<32x197x384xf32>
    %v3568 = stablehlo.reshape %v3567 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3569 = stablehlo.reshape %v3568 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3570 = stablehlo.dot_general %v3569, %b11_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x1536xf32>) -> tensor<32x197x1536xf32>
    %v3571 = stablehlo.broadcast_in_dim %b11_bfc1, dims = [2] : (tensor<1536xf32>) -> tensor<32x197x1536xf32>
    %v3572 = stablehlo.add %v3570, %v3571 : tensor<32x197x1536xf32>
    %v3573 = stablehlo.reshape %v3572 : (tensor<32x197x1536xf32>) -> tensor<32x302592xf32>
    %v3574 = stablehlo.constant dense<0.5> : tensor<32x302592xf32>
    %v3575 = stablehlo.multiply %v3574, %v3573 : tensor<32x302592xf32>
    %v3576 = stablehlo.negate %v3573 : tensor<32x302592xf32>
    %v3577 = stablehlo.constant dense<0.7071067811865476> : tensor<32x302592xf32>
    %v3578 = stablehlo.multiply %v3576, %v3577 : tensor<32x302592xf32>
    %v3579 = chlo.erfc %v3578 : tensor<32x302592xf32> -> tensor<32x302592xf32>
    %v3580 = stablehlo.multiply %v3575, %v3579 : tensor<32x302592xf32>
    %v3581 = stablehlo.reshape %v3580 : (tensor<32x302592xf32>) -> tensor<32x197x1536xf32>
    %v3582 = stablehlo.dot_general %v3581, %b11_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x1536xf32>, tensor<1536x384xf32>) -> tensor<32x197x384xf32>
    %v3583 = stablehlo.broadcast_in_dim %b11_bfc2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v3584 = stablehlo.add %v3582, %v3583 : tensor<32x197x384xf32>
    %v3585 = stablehlo.reshape %v3584 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3586 = stablehlo.add %v3540, %v3585 : tensor<32x75648xf32>
    %v3587 = stablehlo.reshape %v3586 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3588 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3589 = stablehlo.constant dense<384.0> : tensor<32x197x384xf32>
    %v3590 = stablehlo.constant dense<1.0e-5> : tensor<32x197x384xf32>
    %v3591 = stablehlo.reduce(%v3587 init: %v3588) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3592 = stablehlo.broadcast_in_dim %v3591, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v3593 = stablehlo.divide %v3592, %v3589 : tensor<32x197x384xf32>
    %v3594 = stablehlo.subtract %v3587, %v3593 : tensor<32x197x384xf32>
    %v3595 = stablehlo.multiply %v3594, %v3594 : tensor<32x197x384xf32>
    %v3596 = stablehlo.reduce(%v3595 init: %v3588) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3597 = stablehlo.broadcast_in_dim %v3596, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v3598 = stablehlo.divide %v3597, %v3589 : tensor<32x197x384xf32>
    %v3599 = stablehlo.add %v3598, %v3590 : tensor<32x197x384xf32>
    %v3600 = stablehlo.rsqrt %v3599 : tensor<32x197x384xf32>
    %v3601 = stablehlo.multiply %v3594, %v3600 : tensor<32x197x384xf32>
    %v3602 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v3603 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v3604 = stablehlo.multiply %v3601, %v3602 : tensor<32x197x384xf32>
    %v3605 = stablehlo.add %v3604, %v3603 : tensor<32x197x384xf32>
    %v3606 = stablehlo.reshape %v3605 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3607 = stablehlo.reshape %v3606 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3608 = stablehlo.broadcast_in_dim %gF, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v3609 = stablehlo.multiply %v3607, %v3608 : tensor<32x197x384xf32>
    %v3610 = stablehlo.reshape %v3609 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3611 = stablehlo.reshape %v3610 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3612 = stablehlo.broadcast_in_dim %btF, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v3613 = stablehlo.add %v3611, %v3612 : tensor<32x197x384xf32>
    %v3614 = stablehlo.reshape %v3613 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3615 = stablehlo.reshape %v3614 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3616 = stablehlo.slice %v3615 [0:32, 0:1, 0:384] : (tensor<32x197x384xf32>) -> tensor<32x1x384xf32>
    %v3617 = stablehlo.reshape %v3616 : (tensor<32x1x384xf32>) -> tensor<32x384xf32>
    %v3618 = stablehlo.dot_general %v3617, %Wc, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x384xf32>, tensor<384x1000xf32>) -> tensor<32x1000xf32>
    %v3619 = stablehlo.broadcast_in_dim %bc, dims = [1] : (tensor<1000xf32>) -> tensor<32x1000xf32>
    %v3620 = stablehlo.add %v3618, %v3619 : tensor<32x1000xf32>
    return %v3620 : tensor<32x1000xf32>
  }
}

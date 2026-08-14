module @m {
  func.func @vitsin_fwd(%x: tensor<32x150528xf32>, %wConv: tensor<384x3x16x16xf32>, %bConv: tensor<384xf32>, %cls: tensor<384xf32>, %pos: tensor<197x384xf32>, %b0_g1: tensor<384xf32>, %b0_bt1: tensor<384xf32>, %b0_Wq: tensor<384x384xf32>, %b0_bq: tensor<384xf32>, %b0_Wk: tensor<384x384xf32>, %b0_bk: tensor<384xf32>, %b0_Wv: tensor<384x384xf32>, %b0_bv: tensor<384xf32>, %b0_Wo: tensor<384x384xf32>, %b0_bo: tensor<384xf32>, %b0_g2: tensor<384xf32>, %b0_bt2: tensor<384xf32>, %b0_Wfc1: tensor<384x1536xf32>, %b0_bfc1: tensor<1536xf32>, %b0_Wfc2: tensor<1536x384xf32>, %b0_bfc2: tensor<384xf32>, %b1_g1: tensor<384xf32>, %b1_bt1: tensor<384xf32>, %b1_Wq: tensor<384x384xf32>, %b1_bq: tensor<384xf32>, %b1_Wk: tensor<384x384xf32>, %b1_bk: tensor<384xf32>, %b1_Wv: tensor<384x384xf32>, %b1_bv: tensor<384xf32>, %b1_Wo: tensor<384x384xf32>, %b1_bo: tensor<384xf32>, %b1_g2: tensor<384xf32>, %b1_bt2: tensor<384xf32>, %b1_Wfc1: tensor<384x1536xf32>, %b1_bfc1: tensor<1536xf32>, %b1_Wfc2: tensor<1536x384xf32>, %b1_bfc2: tensor<384xf32>, %b2_g1: tensor<384xf32>, %b2_bt1: tensor<384xf32>, %b2_Wq: tensor<384x384xf32>, %b2_bq: tensor<384xf32>, %b2_Wk: tensor<384x384xf32>, %b2_bk: tensor<384xf32>, %b2_Wv: tensor<384x384xf32>, %b2_bv: tensor<384xf32>, %b2_Wo: tensor<384x384xf32>, %b2_bo: tensor<384xf32>, %b2_g2: tensor<384xf32>, %b2_bt2: tensor<384xf32>, %b2_Wfc1: tensor<384x1536xf32>, %b2_bfc1: tensor<1536xf32>, %b2_Wfc2: tensor<1536x384xf32>, %b2_bfc2: tensor<384xf32>, %b3_g1: tensor<384xf32>, %b3_bt1: tensor<384xf32>, %b3_Wq: tensor<384x384xf32>, %b3_bq: tensor<384xf32>, %b3_Wk: tensor<384x384xf32>, %b3_bk: tensor<384xf32>, %b3_Wv: tensor<384x384xf32>, %b3_bv: tensor<384xf32>, %b3_Wo: tensor<384x384xf32>, %b3_bo: tensor<384xf32>, %b3_g2: tensor<384xf32>, %b3_bt2: tensor<384xf32>, %b3_Wfc1: tensor<384x1536xf32>, %b3_bfc1: tensor<1536xf32>, %b3_Wfc2: tensor<1536x384xf32>, %b3_bfc2: tensor<384xf32>, %b4_g1: tensor<384xf32>, %b4_bt1: tensor<384xf32>, %b4_Wq: tensor<384x384xf32>, %b4_bq: tensor<384xf32>, %b4_Wk: tensor<384x384xf32>, %b4_bk: tensor<384xf32>, %b4_Wv: tensor<384x384xf32>, %b4_bv: tensor<384xf32>, %b4_Wo: tensor<384x384xf32>, %b4_bo: tensor<384xf32>, %b4_g2: tensor<384xf32>, %b4_bt2: tensor<384xf32>, %b4_Wfc1: tensor<384x1536xf32>, %b4_bfc1: tensor<1536xf32>, %b4_Wfc2: tensor<1536x384xf32>, %b4_bfc2: tensor<384xf32>, %b5_g1: tensor<384xf32>, %b5_bt1: tensor<384xf32>, %b5_Wq: tensor<384x384xf32>, %b5_bq: tensor<384xf32>, %b5_Wk: tensor<384x384xf32>, %b5_bk: tensor<384xf32>, %b5_Wv: tensor<384x384xf32>, %b5_bv: tensor<384xf32>, %b5_Wo: tensor<384x384xf32>, %b5_bo: tensor<384xf32>, %b5_g2: tensor<384xf32>, %b5_bt2: tensor<384xf32>, %b5_Wfc1: tensor<384x1536xf32>, %b5_bfc1: tensor<1536xf32>, %b5_Wfc2: tensor<1536x384xf32>, %b5_bfc2: tensor<384xf32>, %b6_g1: tensor<384xf32>, %b6_bt1: tensor<384xf32>, %b6_Wq: tensor<384x384xf32>, %b6_bq: tensor<384xf32>, %b6_Wk: tensor<384x384xf32>, %b6_bk: tensor<384xf32>, %b6_Wv: tensor<384x384xf32>, %b6_bv: tensor<384xf32>, %b6_Wo: tensor<384x384xf32>, %b6_bo: tensor<384xf32>, %b6_g2: tensor<384xf32>, %b6_bt2: tensor<384xf32>, %b6_Wfc1: tensor<384x1536xf32>, %b6_bfc1: tensor<1536xf32>, %b6_Wfc2: tensor<1536x384xf32>, %b6_bfc2: tensor<384xf32>, %b7_g1: tensor<384xf32>, %b7_bt1: tensor<384xf32>, %b7_Wq: tensor<384x384xf32>, %b7_bq: tensor<384xf32>, %b7_Wk: tensor<384x384xf32>, %b7_bk: tensor<384xf32>, %b7_Wv: tensor<384x384xf32>, %b7_bv: tensor<384xf32>, %b7_Wo: tensor<384x384xf32>, %b7_bo: tensor<384xf32>, %b7_g2: tensor<384xf32>, %b7_bt2: tensor<384xf32>, %b7_Wfc1: tensor<384x1536xf32>, %b7_bfc1: tensor<1536xf32>, %b7_Wfc2: tensor<1536x384xf32>, %b7_bfc2: tensor<384xf32>, %b8_g1: tensor<384xf32>, %b8_bt1: tensor<384xf32>, %b8_Wq: tensor<384x384xf32>, %b8_bq: tensor<384xf32>, %b8_Wk: tensor<384x384xf32>, %b8_bk: tensor<384xf32>, %b8_Wv: tensor<384x384xf32>, %b8_bv: tensor<384xf32>, %b8_Wo: tensor<384x384xf32>, %b8_bo: tensor<384xf32>, %b8_g2: tensor<384xf32>, %b8_bt2: tensor<384xf32>, %b8_Wfc1: tensor<384x1536xf32>, %b8_bfc1: tensor<1536xf32>, %b8_Wfc2: tensor<1536x384xf32>, %b8_bfc2: tensor<384xf32>, %b9_g1: tensor<384xf32>, %b9_bt1: tensor<384xf32>, %b9_Wq: tensor<384x384xf32>, %b9_bq: tensor<384xf32>, %b9_Wk: tensor<384x384xf32>, %b9_bk: tensor<384xf32>, %b9_Wv: tensor<384x384xf32>, %b9_bv: tensor<384xf32>, %b9_Wo: tensor<384x384xf32>, %b9_bo: tensor<384xf32>, %b9_g2: tensor<384xf32>, %b9_bt2: tensor<384xf32>, %b9_Wfc1: tensor<384x1536xf32>, %b9_bfc1: tensor<1536xf32>, %b9_Wfc2: tensor<1536x384xf32>, %b9_bfc2: tensor<384xf32>, %b10_g1: tensor<384xf32>, %b10_bt1: tensor<384xf32>, %b10_Wq: tensor<384x384xf32>, %b10_bq: tensor<384xf32>, %b10_Wk: tensor<384x384xf32>, %b10_bk: tensor<384xf32>, %b10_Wv: tensor<384x384xf32>, %b10_bv: tensor<384xf32>, %b10_Wo: tensor<384x384xf32>, %b10_bo: tensor<384xf32>, %b10_g2: tensor<384xf32>, %b10_bt2: tensor<384xf32>, %b10_Wfc1: tensor<384x1536xf32>, %b10_bfc1: tensor<1536xf32>, %b10_Wfc2: tensor<1536x384xf32>, %b10_bfc2: tensor<384xf32>, %b11_g1: tensor<384xf32>, %b11_bt1: tensor<384xf32>, %b11_Wq: tensor<384x384xf32>, %b11_bq: tensor<384xf32>, %b11_Wk: tensor<384x384xf32>, %b11_bk: tensor<384xf32>, %b11_Wv: tensor<384x384xf32>, %b11_bv: tensor<384xf32>, %b11_Wo: tensor<384x384xf32>, %b11_bo: tensor<384xf32>, %b11_g2: tensor<384xf32>, %b11_bt2: tensor<384xf32>, %b11_Wfc1: tensor<384x1536xf32>, %b11_bfc1: tensor<1536xf32>, %b11_Wfc2: tensor<1536x384xf32>, %b11_bfc2: tensor<384xf32>, %gF: tensor<384xf32>, %btF: tensor<384xf32>, %Wc: tensor<384x1000xf32>, %bc: tensor<1000xf32>) -> tensor<32x1000xf32> {
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
    %v296 = stablehlo.multiply %v295, %v295 : tensor<32x302592xf32>
    %v297 = stablehlo.multiply %v296, %v295 : tensor<32x302592xf32>
    %v298 = stablehlo.constant dense<0.044715> : tensor<32x302592xf32>
    %v299 = stablehlo.multiply %v298, %v297 : tensor<32x302592xf32>
    %v300 = stablehlo.add %v295, %v299 : tensor<32x302592xf32>
    %v301 = stablehlo.constant dense<0.7978845608028654> : tensor<32x302592xf32>
    %v302 = stablehlo.multiply %v301, %v300 : tensor<32x302592xf32>
    %v303 = stablehlo.tanh %v302 : tensor<32x302592xf32>
    %v304 = stablehlo.constant dense<1.0> : tensor<32x302592xf32>
    %v305 = stablehlo.add %v304, %v303 : tensor<32x302592xf32>
    %v306 = stablehlo.constant dense<0.5> : tensor<32x302592xf32>
    %v307 = stablehlo.multiply %v306, %v295 : tensor<32x302592xf32>
    %v308 = stablehlo.multiply %v307, %v305 : tensor<32x302592xf32>
    %v309 = stablehlo.reshape %v308 : (tensor<32x302592xf32>) -> tensor<32x197x1536xf32>
    %v310 = stablehlo.dot_general %v309, %b0_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x1536xf32>, tensor<1536x384xf32>) -> tensor<32x197x384xf32>
    %v311 = stablehlo.broadcast_in_dim %b0_bfc2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v312 = stablehlo.add %v310, %v311 : tensor<32x197x384xf32>
    %v313 = stablehlo.reshape %v312 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v314 = stablehlo.add %v262, %v313 : tensor<32x75648xf32>
    %v315 = stablehlo.reshape %v314 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v316 = stablehlo.constant dense<0.0> : tensor<f32>
    %v317 = stablehlo.constant dense<384.0> : tensor<32x197x384xf32>
    %v318 = stablehlo.constant dense<1.0e-5> : tensor<32x197x384xf32>
    %v319 = stablehlo.reduce(%v315 init: %v316) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v320 = stablehlo.broadcast_in_dim %v319, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v321 = stablehlo.divide %v320, %v317 : tensor<32x197x384xf32>
    %v322 = stablehlo.subtract %v315, %v321 : tensor<32x197x384xf32>
    %v323 = stablehlo.multiply %v322, %v322 : tensor<32x197x384xf32>
    %v324 = stablehlo.reduce(%v323 init: %v316) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v325 = stablehlo.broadcast_in_dim %v324, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v326 = stablehlo.divide %v325, %v317 : tensor<32x197x384xf32>
    %v327 = stablehlo.add %v326, %v318 : tensor<32x197x384xf32>
    %v328 = stablehlo.rsqrt %v327 : tensor<32x197x384xf32>
    %v329 = stablehlo.multiply %v322, %v328 : tensor<32x197x384xf32>
    %v330 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v331 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v332 = stablehlo.multiply %v329, %v330 : tensor<32x197x384xf32>
    %v333 = stablehlo.add %v332, %v331 : tensor<32x197x384xf32>
    %v334 = stablehlo.reshape %v333 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v335 = stablehlo.reshape %v334 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v336 = stablehlo.broadcast_in_dim %b1_g1, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v337 = stablehlo.multiply %v335, %v336 : tensor<32x197x384xf32>
    %v338 = stablehlo.reshape %v337 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v339 = stablehlo.reshape %v338 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v340 = stablehlo.broadcast_in_dim %b1_bt1, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v341 = stablehlo.add %v339, %v340 : tensor<32x197x384xf32>
    %v342 = stablehlo.reshape %v341 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v343 = stablehlo.reshape %v342 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v344 = stablehlo.dot_general %v343, %b1_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v345 = stablehlo.broadcast_in_dim %b1_bq, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v346 = stablehlo.add %v344, %v345 : tensor<32x197x384xf32>
    %v347 = stablehlo.reshape %v346 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v348 = stablehlo.reshape %v342 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v349 = stablehlo.dot_general %v348, %b1_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v350 = stablehlo.broadcast_in_dim %b1_bk, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v351 = stablehlo.add %v349, %v350 : tensor<32x197x384xf32>
    %v352 = stablehlo.reshape %v351 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v353 = stablehlo.reshape %v342 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v354 = stablehlo.dot_general %v353, %b1_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v355 = stablehlo.broadcast_in_dim %b1_bv, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v356 = stablehlo.add %v354, %v355 : tensor<32x197x384xf32>
    %v357 = stablehlo.reshape %v356 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v358 = stablehlo.reshape %v347 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v359 = stablehlo.slice %v358 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v360 = stablehlo.reshape %v359 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v361 = stablehlo.reshape %v352 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v362 = stablehlo.slice %v361 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v363 = stablehlo.reshape %v362 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v364 = stablehlo.reshape %v357 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v365 = stablehlo.slice %v364 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v366 = stablehlo.reshape %v365 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v367 = stablehlo.reshape %v363 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v368 = stablehlo.transpose %v367, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v369 = stablehlo.reshape %v368 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v370 = stablehlo.reshape %v360 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v371 = stablehlo.reshape %v369 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v372 = stablehlo.dot_general %v370, %v371, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v373 = stablehlo.reshape %v372 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v374 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v375 = stablehlo.multiply %v373, %v374 : tensor<32x38809xf32>
    %v376 = stablehlo.reshape %v375 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v377 = stablehlo.constant dense<0.0> : tensor<f32>
    %v378 = stablehlo.exponential %v376 : tensor<32x197x197xf32>
    %v379 = stablehlo.reduce(%v378 init: %v377) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v380 = stablehlo.broadcast_in_dim %v379, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v381 = stablehlo.divide %v378, %v380 : tensor<32x197x197xf32>
    %v382 = stablehlo.reshape %v381 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v383 = stablehlo.reshape %v382 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v384 = stablehlo.reshape %v366 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v385 = stablehlo.dot_general %v383, %v384, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v386 = stablehlo.reshape %v385 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v387 = stablehlo.reshape %v386 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v388 = stablehlo.constant dense<0.0> : tensor<f32>
    %v389 = stablehlo.pad %v387, %v388, low = [0, 0, 0], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v390 = stablehlo.reshape %v389 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v391 = stablehlo.reshape %v347 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v392 = stablehlo.slice %v391 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v393 = stablehlo.reshape %v392 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v394 = stablehlo.reshape %v352 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v395 = stablehlo.slice %v394 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v396 = stablehlo.reshape %v395 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v397 = stablehlo.reshape %v357 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v398 = stablehlo.slice %v397 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v399 = stablehlo.reshape %v398 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v400 = stablehlo.reshape %v396 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v401 = stablehlo.transpose %v400, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v402 = stablehlo.reshape %v401 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v403 = stablehlo.reshape %v393 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v404 = stablehlo.reshape %v402 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v405 = stablehlo.dot_general %v403, %v404, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v406 = stablehlo.reshape %v405 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v407 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v408 = stablehlo.multiply %v406, %v407 : tensor<32x38809xf32>
    %v409 = stablehlo.reshape %v408 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v410 = stablehlo.constant dense<0.0> : tensor<f32>
    %v411 = stablehlo.exponential %v409 : tensor<32x197x197xf32>
    %v412 = stablehlo.reduce(%v411 init: %v410) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v413 = stablehlo.broadcast_in_dim %v412, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v414 = stablehlo.divide %v411, %v413 : tensor<32x197x197xf32>
    %v415 = stablehlo.reshape %v414 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v416 = stablehlo.reshape %v415 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v417 = stablehlo.reshape %v399 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v418 = stablehlo.dot_general %v416, %v417, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v419 = stablehlo.reshape %v418 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v420 = stablehlo.reshape %v419 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v421 = stablehlo.constant dense<0.0> : tensor<f32>
    %v422 = stablehlo.pad %v420, %v421, low = [0, 0, 64], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v423 = stablehlo.reshape %v422 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v424 = stablehlo.add %v390, %v423 : tensor<32x75648xf32>
    %v425 = stablehlo.reshape %v347 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v426 = stablehlo.slice %v425 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v427 = stablehlo.reshape %v426 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v428 = stablehlo.reshape %v352 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v429 = stablehlo.slice %v428 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v430 = stablehlo.reshape %v429 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v431 = stablehlo.reshape %v357 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v432 = stablehlo.slice %v431 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v433 = stablehlo.reshape %v432 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v434 = stablehlo.reshape %v430 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v435 = stablehlo.transpose %v434, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v436 = stablehlo.reshape %v435 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v437 = stablehlo.reshape %v427 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v438 = stablehlo.reshape %v436 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v439 = stablehlo.dot_general %v437, %v438, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v440 = stablehlo.reshape %v439 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v441 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v442 = stablehlo.multiply %v440, %v441 : tensor<32x38809xf32>
    %v443 = stablehlo.reshape %v442 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v444 = stablehlo.constant dense<0.0> : tensor<f32>
    %v445 = stablehlo.exponential %v443 : tensor<32x197x197xf32>
    %v446 = stablehlo.reduce(%v445 init: %v444) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v447 = stablehlo.broadcast_in_dim %v446, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v448 = stablehlo.divide %v445, %v447 : tensor<32x197x197xf32>
    %v449 = stablehlo.reshape %v448 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v450 = stablehlo.reshape %v449 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v451 = stablehlo.reshape %v433 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v452 = stablehlo.dot_general %v450, %v451, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v453 = stablehlo.reshape %v452 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v454 = stablehlo.reshape %v453 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v455 = stablehlo.constant dense<0.0> : tensor<f32>
    %v456 = stablehlo.pad %v454, %v455, low = [0, 0, 128], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v457 = stablehlo.reshape %v456 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v458 = stablehlo.add %v424, %v457 : tensor<32x75648xf32>
    %v459 = stablehlo.reshape %v347 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v460 = stablehlo.slice %v459 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v461 = stablehlo.reshape %v460 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v462 = stablehlo.reshape %v352 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v463 = stablehlo.slice %v462 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v464 = stablehlo.reshape %v463 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v465 = stablehlo.reshape %v357 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v466 = stablehlo.slice %v465 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v467 = stablehlo.reshape %v466 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v468 = stablehlo.reshape %v464 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v469 = stablehlo.transpose %v468, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v470 = stablehlo.reshape %v469 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v471 = stablehlo.reshape %v461 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v472 = stablehlo.reshape %v470 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v473 = stablehlo.dot_general %v471, %v472, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v474 = stablehlo.reshape %v473 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v475 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v476 = stablehlo.multiply %v474, %v475 : tensor<32x38809xf32>
    %v477 = stablehlo.reshape %v476 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v478 = stablehlo.constant dense<0.0> : tensor<f32>
    %v479 = stablehlo.exponential %v477 : tensor<32x197x197xf32>
    %v480 = stablehlo.reduce(%v479 init: %v478) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v481 = stablehlo.broadcast_in_dim %v480, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v482 = stablehlo.divide %v479, %v481 : tensor<32x197x197xf32>
    %v483 = stablehlo.reshape %v482 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v484 = stablehlo.reshape %v483 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v485 = stablehlo.reshape %v467 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v486 = stablehlo.dot_general %v484, %v485, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v487 = stablehlo.reshape %v486 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v488 = stablehlo.reshape %v487 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v489 = stablehlo.constant dense<0.0> : tensor<f32>
    %v490 = stablehlo.pad %v488, %v489, low = [0, 0, 192], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v491 = stablehlo.reshape %v490 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v492 = stablehlo.add %v458, %v491 : tensor<32x75648xf32>
    %v493 = stablehlo.reshape %v347 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v494 = stablehlo.slice %v493 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v495 = stablehlo.reshape %v494 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v496 = stablehlo.reshape %v352 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v497 = stablehlo.slice %v496 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v498 = stablehlo.reshape %v497 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v499 = stablehlo.reshape %v357 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v500 = stablehlo.slice %v499 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v501 = stablehlo.reshape %v500 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v502 = stablehlo.reshape %v498 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v503 = stablehlo.transpose %v502, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v504 = stablehlo.reshape %v503 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v505 = stablehlo.reshape %v495 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v506 = stablehlo.reshape %v504 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v507 = stablehlo.dot_general %v505, %v506, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v508 = stablehlo.reshape %v507 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v509 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v510 = stablehlo.multiply %v508, %v509 : tensor<32x38809xf32>
    %v511 = stablehlo.reshape %v510 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v512 = stablehlo.constant dense<0.0> : tensor<f32>
    %v513 = stablehlo.exponential %v511 : tensor<32x197x197xf32>
    %v514 = stablehlo.reduce(%v513 init: %v512) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v515 = stablehlo.broadcast_in_dim %v514, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v516 = stablehlo.divide %v513, %v515 : tensor<32x197x197xf32>
    %v517 = stablehlo.reshape %v516 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v518 = stablehlo.reshape %v517 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v519 = stablehlo.reshape %v501 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v520 = stablehlo.dot_general %v518, %v519, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v521 = stablehlo.reshape %v520 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v522 = stablehlo.reshape %v521 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v523 = stablehlo.constant dense<0.0> : tensor<f32>
    %v524 = stablehlo.pad %v522, %v523, low = [0, 0, 256], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v525 = stablehlo.reshape %v524 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v526 = stablehlo.add %v492, %v525 : tensor<32x75648xf32>
    %v527 = stablehlo.reshape %v347 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v528 = stablehlo.slice %v527 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v529 = stablehlo.reshape %v528 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v530 = stablehlo.reshape %v352 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v531 = stablehlo.slice %v530 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v532 = stablehlo.reshape %v531 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v533 = stablehlo.reshape %v357 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v534 = stablehlo.slice %v533 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v535 = stablehlo.reshape %v534 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v536 = stablehlo.reshape %v532 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v537 = stablehlo.transpose %v536, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v538 = stablehlo.reshape %v537 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v539 = stablehlo.reshape %v529 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v540 = stablehlo.reshape %v538 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v541 = stablehlo.dot_general %v539, %v540, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v542 = stablehlo.reshape %v541 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v543 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v544 = stablehlo.multiply %v542, %v543 : tensor<32x38809xf32>
    %v545 = stablehlo.reshape %v544 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v546 = stablehlo.constant dense<0.0> : tensor<f32>
    %v547 = stablehlo.exponential %v545 : tensor<32x197x197xf32>
    %v548 = stablehlo.reduce(%v547 init: %v546) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v549 = stablehlo.broadcast_in_dim %v548, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v550 = stablehlo.divide %v547, %v549 : tensor<32x197x197xf32>
    %v551 = stablehlo.reshape %v550 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v552 = stablehlo.reshape %v551 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v553 = stablehlo.reshape %v535 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v554 = stablehlo.dot_general %v552, %v553, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v555 = stablehlo.reshape %v554 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v556 = stablehlo.reshape %v555 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v557 = stablehlo.constant dense<0.0> : tensor<f32>
    %v558 = stablehlo.pad %v556, %v557, low = [0, 0, 320], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v559 = stablehlo.reshape %v558 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v560 = stablehlo.add %v526, %v559 : tensor<32x75648xf32>
    %v561 = stablehlo.reshape %v560 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v562 = stablehlo.dot_general %v561, %b1_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v563 = stablehlo.broadcast_in_dim %b1_bo, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v564 = stablehlo.add %v562, %v563 : tensor<32x197x384xf32>
    %v565 = stablehlo.reshape %v564 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v566 = stablehlo.add %v314, %v565 : tensor<32x75648xf32>
    %v567 = stablehlo.reshape %v566 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v568 = stablehlo.constant dense<0.0> : tensor<f32>
    %v569 = stablehlo.constant dense<384.0> : tensor<32x197x384xf32>
    %v570 = stablehlo.constant dense<1.0e-5> : tensor<32x197x384xf32>
    %v571 = stablehlo.reduce(%v567 init: %v568) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v572 = stablehlo.broadcast_in_dim %v571, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v573 = stablehlo.divide %v572, %v569 : tensor<32x197x384xf32>
    %v574 = stablehlo.subtract %v567, %v573 : tensor<32x197x384xf32>
    %v575 = stablehlo.multiply %v574, %v574 : tensor<32x197x384xf32>
    %v576 = stablehlo.reduce(%v575 init: %v568) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v577 = stablehlo.broadcast_in_dim %v576, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v578 = stablehlo.divide %v577, %v569 : tensor<32x197x384xf32>
    %v579 = stablehlo.add %v578, %v570 : tensor<32x197x384xf32>
    %v580 = stablehlo.rsqrt %v579 : tensor<32x197x384xf32>
    %v581 = stablehlo.multiply %v574, %v580 : tensor<32x197x384xf32>
    %v582 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v583 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v584 = stablehlo.multiply %v581, %v582 : tensor<32x197x384xf32>
    %v585 = stablehlo.add %v584, %v583 : tensor<32x197x384xf32>
    %v586 = stablehlo.reshape %v585 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v587 = stablehlo.reshape %v586 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v588 = stablehlo.broadcast_in_dim %b1_g2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v589 = stablehlo.multiply %v587, %v588 : tensor<32x197x384xf32>
    %v590 = stablehlo.reshape %v589 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v591 = stablehlo.reshape %v590 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v592 = stablehlo.broadcast_in_dim %b1_bt2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v593 = stablehlo.add %v591, %v592 : tensor<32x197x384xf32>
    %v594 = stablehlo.reshape %v593 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v595 = stablehlo.reshape %v594 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v596 = stablehlo.dot_general %v595, %b1_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x1536xf32>) -> tensor<32x197x1536xf32>
    %v597 = stablehlo.broadcast_in_dim %b1_bfc1, dims = [2] : (tensor<1536xf32>) -> tensor<32x197x1536xf32>
    %v598 = stablehlo.add %v596, %v597 : tensor<32x197x1536xf32>
    %v599 = stablehlo.reshape %v598 : (tensor<32x197x1536xf32>) -> tensor<32x302592xf32>
    %v600 = stablehlo.multiply %v599, %v599 : tensor<32x302592xf32>
    %v601 = stablehlo.multiply %v600, %v599 : tensor<32x302592xf32>
    %v602 = stablehlo.constant dense<0.044715> : tensor<32x302592xf32>
    %v603 = stablehlo.multiply %v602, %v601 : tensor<32x302592xf32>
    %v604 = stablehlo.add %v599, %v603 : tensor<32x302592xf32>
    %v605 = stablehlo.constant dense<0.7978845608028654> : tensor<32x302592xf32>
    %v606 = stablehlo.multiply %v605, %v604 : tensor<32x302592xf32>
    %v607 = stablehlo.tanh %v606 : tensor<32x302592xf32>
    %v608 = stablehlo.constant dense<1.0> : tensor<32x302592xf32>
    %v609 = stablehlo.add %v608, %v607 : tensor<32x302592xf32>
    %v610 = stablehlo.constant dense<0.5> : tensor<32x302592xf32>
    %v611 = stablehlo.multiply %v610, %v599 : tensor<32x302592xf32>
    %v612 = stablehlo.multiply %v611, %v609 : tensor<32x302592xf32>
    %v613 = stablehlo.reshape %v612 : (tensor<32x302592xf32>) -> tensor<32x197x1536xf32>
    %v614 = stablehlo.dot_general %v613, %b1_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x1536xf32>, tensor<1536x384xf32>) -> tensor<32x197x384xf32>
    %v615 = stablehlo.broadcast_in_dim %b1_bfc2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v616 = stablehlo.add %v614, %v615 : tensor<32x197x384xf32>
    %v617 = stablehlo.reshape %v616 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v618 = stablehlo.add %v566, %v617 : tensor<32x75648xf32>
    %v619 = stablehlo.reshape %v618 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v620 = stablehlo.constant dense<0.0> : tensor<f32>
    %v621 = stablehlo.constant dense<384.0> : tensor<32x197x384xf32>
    %v622 = stablehlo.constant dense<1.0e-5> : tensor<32x197x384xf32>
    %v623 = stablehlo.reduce(%v619 init: %v620) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v624 = stablehlo.broadcast_in_dim %v623, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v625 = stablehlo.divide %v624, %v621 : tensor<32x197x384xf32>
    %v626 = stablehlo.subtract %v619, %v625 : tensor<32x197x384xf32>
    %v627 = stablehlo.multiply %v626, %v626 : tensor<32x197x384xf32>
    %v628 = stablehlo.reduce(%v627 init: %v620) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v629 = stablehlo.broadcast_in_dim %v628, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v630 = stablehlo.divide %v629, %v621 : tensor<32x197x384xf32>
    %v631 = stablehlo.add %v630, %v622 : tensor<32x197x384xf32>
    %v632 = stablehlo.rsqrt %v631 : tensor<32x197x384xf32>
    %v633 = stablehlo.multiply %v626, %v632 : tensor<32x197x384xf32>
    %v634 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v635 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v636 = stablehlo.multiply %v633, %v634 : tensor<32x197x384xf32>
    %v637 = stablehlo.add %v636, %v635 : tensor<32x197x384xf32>
    %v638 = stablehlo.reshape %v637 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v639 = stablehlo.reshape %v638 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v640 = stablehlo.broadcast_in_dim %b2_g1, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v641 = stablehlo.multiply %v639, %v640 : tensor<32x197x384xf32>
    %v642 = stablehlo.reshape %v641 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v643 = stablehlo.reshape %v642 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v644 = stablehlo.broadcast_in_dim %b2_bt1, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v645 = stablehlo.add %v643, %v644 : tensor<32x197x384xf32>
    %v646 = stablehlo.reshape %v645 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v647 = stablehlo.reshape %v646 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v648 = stablehlo.dot_general %v647, %b2_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v649 = stablehlo.broadcast_in_dim %b2_bq, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v650 = stablehlo.add %v648, %v649 : tensor<32x197x384xf32>
    %v651 = stablehlo.reshape %v650 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v652 = stablehlo.reshape %v646 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v653 = stablehlo.dot_general %v652, %b2_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v654 = stablehlo.broadcast_in_dim %b2_bk, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v655 = stablehlo.add %v653, %v654 : tensor<32x197x384xf32>
    %v656 = stablehlo.reshape %v655 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v657 = stablehlo.reshape %v646 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v658 = stablehlo.dot_general %v657, %b2_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v659 = stablehlo.broadcast_in_dim %b2_bv, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v660 = stablehlo.add %v658, %v659 : tensor<32x197x384xf32>
    %v661 = stablehlo.reshape %v660 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v662 = stablehlo.reshape %v651 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v663 = stablehlo.slice %v662 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v664 = stablehlo.reshape %v663 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v665 = stablehlo.reshape %v656 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v666 = stablehlo.slice %v665 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v667 = stablehlo.reshape %v666 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v668 = stablehlo.reshape %v661 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v669 = stablehlo.slice %v668 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v670 = stablehlo.reshape %v669 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v671 = stablehlo.reshape %v667 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v672 = stablehlo.transpose %v671, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v673 = stablehlo.reshape %v672 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v674 = stablehlo.reshape %v664 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v675 = stablehlo.reshape %v673 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v676 = stablehlo.dot_general %v674, %v675, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v677 = stablehlo.reshape %v676 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v678 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v679 = stablehlo.multiply %v677, %v678 : tensor<32x38809xf32>
    %v680 = stablehlo.reshape %v679 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v681 = stablehlo.constant dense<0.0> : tensor<f32>
    %v682 = stablehlo.exponential %v680 : tensor<32x197x197xf32>
    %v683 = stablehlo.reduce(%v682 init: %v681) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v684 = stablehlo.broadcast_in_dim %v683, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v685 = stablehlo.divide %v682, %v684 : tensor<32x197x197xf32>
    %v686 = stablehlo.reshape %v685 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v687 = stablehlo.reshape %v686 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v688 = stablehlo.reshape %v670 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v689 = stablehlo.dot_general %v687, %v688, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v690 = stablehlo.reshape %v689 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v691 = stablehlo.reshape %v690 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v692 = stablehlo.constant dense<0.0> : tensor<f32>
    %v693 = stablehlo.pad %v691, %v692, low = [0, 0, 0], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v694 = stablehlo.reshape %v693 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v695 = stablehlo.reshape %v651 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v696 = stablehlo.slice %v695 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v697 = stablehlo.reshape %v696 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v698 = stablehlo.reshape %v656 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v699 = stablehlo.slice %v698 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v700 = stablehlo.reshape %v699 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v701 = stablehlo.reshape %v661 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v702 = stablehlo.slice %v701 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v703 = stablehlo.reshape %v702 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v704 = stablehlo.reshape %v700 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v705 = stablehlo.transpose %v704, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v706 = stablehlo.reshape %v705 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v707 = stablehlo.reshape %v697 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v708 = stablehlo.reshape %v706 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v709 = stablehlo.dot_general %v707, %v708, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v710 = stablehlo.reshape %v709 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v711 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v712 = stablehlo.multiply %v710, %v711 : tensor<32x38809xf32>
    %v713 = stablehlo.reshape %v712 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v714 = stablehlo.constant dense<0.0> : tensor<f32>
    %v715 = stablehlo.exponential %v713 : tensor<32x197x197xf32>
    %v716 = stablehlo.reduce(%v715 init: %v714) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v717 = stablehlo.broadcast_in_dim %v716, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v718 = stablehlo.divide %v715, %v717 : tensor<32x197x197xf32>
    %v719 = stablehlo.reshape %v718 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v720 = stablehlo.reshape %v719 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v721 = stablehlo.reshape %v703 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v722 = stablehlo.dot_general %v720, %v721, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v723 = stablehlo.reshape %v722 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v724 = stablehlo.reshape %v723 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v725 = stablehlo.constant dense<0.0> : tensor<f32>
    %v726 = stablehlo.pad %v724, %v725, low = [0, 0, 64], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v727 = stablehlo.reshape %v726 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v728 = stablehlo.add %v694, %v727 : tensor<32x75648xf32>
    %v729 = stablehlo.reshape %v651 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v730 = stablehlo.slice %v729 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v731 = stablehlo.reshape %v730 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v732 = stablehlo.reshape %v656 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v733 = stablehlo.slice %v732 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v734 = stablehlo.reshape %v733 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v735 = stablehlo.reshape %v661 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v736 = stablehlo.slice %v735 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v737 = stablehlo.reshape %v736 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v738 = stablehlo.reshape %v734 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v739 = stablehlo.transpose %v738, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v740 = stablehlo.reshape %v739 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v741 = stablehlo.reshape %v731 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v742 = stablehlo.reshape %v740 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v743 = stablehlo.dot_general %v741, %v742, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v744 = stablehlo.reshape %v743 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v745 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v746 = stablehlo.multiply %v744, %v745 : tensor<32x38809xf32>
    %v747 = stablehlo.reshape %v746 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v748 = stablehlo.constant dense<0.0> : tensor<f32>
    %v749 = stablehlo.exponential %v747 : tensor<32x197x197xf32>
    %v750 = stablehlo.reduce(%v749 init: %v748) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v751 = stablehlo.broadcast_in_dim %v750, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v752 = stablehlo.divide %v749, %v751 : tensor<32x197x197xf32>
    %v753 = stablehlo.reshape %v752 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v754 = stablehlo.reshape %v753 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v755 = stablehlo.reshape %v737 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v756 = stablehlo.dot_general %v754, %v755, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v757 = stablehlo.reshape %v756 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v758 = stablehlo.reshape %v757 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v759 = stablehlo.constant dense<0.0> : tensor<f32>
    %v760 = stablehlo.pad %v758, %v759, low = [0, 0, 128], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v761 = stablehlo.reshape %v760 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v762 = stablehlo.add %v728, %v761 : tensor<32x75648xf32>
    %v763 = stablehlo.reshape %v651 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v764 = stablehlo.slice %v763 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v765 = stablehlo.reshape %v764 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v766 = stablehlo.reshape %v656 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v767 = stablehlo.slice %v766 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v768 = stablehlo.reshape %v767 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v769 = stablehlo.reshape %v661 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v770 = stablehlo.slice %v769 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v771 = stablehlo.reshape %v770 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v772 = stablehlo.reshape %v768 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v773 = stablehlo.transpose %v772, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v774 = stablehlo.reshape %v773 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v775 = stablehlo.reshape %v765 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v776 = stablehlo.reshape %v774 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v777 = stablehlo.dot_general %v775, %v776, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v778 = stablehlo.reshape %v777 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v779 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v780 = stablehlo.multiply %v778, %v779 : tensor<32x38809xf32>
    %v781 = stablehlo.reshape %v780 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v782 = stablehlo.constant dense<0.0> : tensor<f32>
    %v783 = stablehlo.exponential %v781 : tensor<32x197x197xf32>
    %v784 = stablehlo.reduce(%v783 init: %v782) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v785 = stablehlo.broadcast_in_dim %v784, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v786 = stablehlo.divide %v783, %v785 : tensor<32x197x197xf32>
    %v787 = stablehlo.reshape %v786 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v788 = stablehlo.reshape %v787 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v789 = stablehlo.reshape %v771 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v790 = stablehlo.dot_general %v788, %v789, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v791 = stablehlo.reshape %v790 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v792 = stablehlo.reshape %v791 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v793 = stablehlo.constant dense<0.0> : tensor<f32>
    %v794 = stablehlo.pad %v792, %v793, low = [0, 0, 192], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v795 = stablehlo.reshape %v794 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v796 = stablehlo.add %v762, %v795 : tensor<32x75648xf32>
    %v797 = stablehlo.reshape %v651 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v798 = stablehlo.slice %v797 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v799 = stablehlo.reshape %v798 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v800 = stablehlo.reshape %v656 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v801 = stablehlo.slice %v800 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v802 = stablehlo.reshape %v801 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v803 = stablehlo.reshape %v661 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v804 = stablehlo.slice %v803 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v805 = stablehlo.reshape %v804 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v806 = stablehlo.reshape %v802 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v807 = stablehlo.transpose %v806, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v808 = stablehlo.reshape %v807 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v809 = stablehlo.reshape %v799 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v810 = stablehlo.reshape %v808 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v811 = stablehlo.dot_general %v809, %v810, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v812 = stablehlo.reshape %v811 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v813 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v814 = stablehlo.multiply %v812, %v813 : tensor<32x38809xf32>
    %v815 = stablehlo.reshape %v814 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v816 = stablehlo.constant dense<0.0> : tensor<f32>
    %v817 = stablehlo.exponential %v815 : tensor<32x197x197xf32>
    %v818 = stablehlo.reduce(%v817 init: %v816) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v819 = stablehlo.broadcast_in_dim %v818, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v820 = stablehlo.divide %v817, %v819 : tensor<32x197x197xf32>
    %v821 = stablehlo.reshape %v820 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v822 = stablehlo.reshape %v821 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v823 = stablehlo.reshape %v805 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v824 = stablehlo.dot_general %v822, %v823, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v825 = stablehlo.reshape %v824 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v826 = stablehlo.reshape %v825 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v827 = stablehlo.constant dense<0.0> : tensor<f32>
    %v828 = stablehlo.pad %v826, %v827, low = [0, 0, 256], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v829 = stablehlo.reshape %v828 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v830 = stablehlo.add %v796, %v829 : tensor<32x75648xf32>
    %v831 = stablehlo.reshape %v651 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v832 = stablehlo.slice %v831 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v833 = stablehlo.reshape %v832 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v834 = stablehlo.reshape %v656 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v835 = stablehlo.slice %v834 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v836 = stablehlo.reshape %v835 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v837 = stablehlo.reshape %v661 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v838 = stablehlo.slice %v837 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
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
    %v850 = stablehlo.constant dense<0.0> : tensor<f32>
    %v851 = stablehlo.exponential %v849 : tensor<32x197x197xf32>
    %v852 = stablehlo.reduce(%v851 init: %v850) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v853 = stablehlo.broadcast_in_dim %v852, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v854 = stablehlo.divide %v851, %v853 : tensor<32x197x197xf32>
    %v855 = stablehlo.reshape %v854 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v856 = stablehlo.reshape %v855 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v857 = stablehlo.reshape %v839 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v858 = stablehlo.dot_general %v856, %v857, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v859 = stablehlo.reshape %v858 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v860 = stablehlo.reshape %v859 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v861 = stablehlo.constant dense<0.0> : tensor<f32>
    %v862 = stablehlo.pad %v860, %v861, low = [0, 0, 320], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v863 = stablehlo.reshape %v862 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v864 = stablehlo.add %v830, %v863 : tensor<32x75648xf32>
    %v865 = stablehlo.reshape %v864 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v866 = stablehlo.dot_general %v865, %b2_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v867 = stablehlo.broadcast_in_dim %b2_bo, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v868 = stablehlo.add %v866, %v867 : tensor<32x197x384xf32>
    %v869 = stablehlo.reshape %v868 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v870 = stablehlo.add %v618, %v869 : tensor<32x75648xf32>
    %v871 = stablehlo.reshape %v870 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v872 = stablehlo.constant dense<0.0> : tensor<f32>
    %v873 = stablehlo.constant dense<384.0> : tensor<32x197x384xf32>
    %v874 = stablehlo.constant dense<1.0e-5> : tensor<32x197x384xf32>
    %v875 = stablehlo.reduce(%v871 init: %v872) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v876 = stablehlo.broadcast_in_dim %v875, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v877 = stablehlo.divide %v876, %v873 : tensor<32x197x384xf32>
    %v878 = stablehlo.subtract %v871, %v877 : tensor<32x197x384xf32>
    %v879 = stablehlo.multiply %v878, %v878 : tensor<32x197x384xf32>
    %v880 = stablehlo.reduce(%v879 init: %v872) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v881 = stablehlo.broadcast_in_dim %v880, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v882 = stablehlo.divide %v881, %v873 : tensor<32x197x384xf32>
    %v883 = stablehlo.add %v882, %v874 : tensor<32x197x384xf32>
    %v884 = stablehlo.rsqrt %v883 : tensor<32x197x384xf32>
    %v885 = stablehlo.multiply %v878, %v884 : tensor<32x197x384xf32>
    %v886 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v887 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v888 = stablehlo.multiply %v885, %v886 : tensor<32x197x384xf32>
    %v889 = stablehlo.add %v888, %v887 : tensor<32x197x384xf32>
    %v890 = stablehlo.reshape %v889 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v891 = stablehlo.reshape %v890 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v892 = stablehlo.broadcast_in_dim %b2_g2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v893 = stablehlo.multiply %v891, %v892 : tensor<32x197x384xf32>
    %v894 = stablehlo.reshape %v893 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v895 = stablehlo.reshape %v894 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v896 = stablehlo.broadcast_in_dim %b2_bt2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v897 = stablehlo.add %v895, %v896 : tensor<32x197x384xf32>
    %v898 = stablehlo.reshape %v897 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v899 = stablehlo.reshape %v898 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v900 = stablehlo.dot_general %v899, %b2_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x1536xf32>) -> tensor<32x197x1536xf32>
    %v901 = stablehlo.broadcast_in_dim %b2_bfc1, dims = [2] : (tensor<1536xf32>) -> tensor<32x197x1536xf32>
    %v902 = stablehlo.add %v900, %v901 : tensor<32x197x1536xf32>
    %v903 = stablehlo.reshape %v902 : (tensor<32x197x1536xf32>) -> tensor<32x302592xf32>
    %v904 = stablehlo.multiply %v903, %v903 : tensor<32x302592xf32>
    %v905 = stablehlo.multiply %v904, %v903 : tensor<32x302592xf32>
    %v906 = stablehlo.constant dense<0.044715> : tensor<32x302592xf32>
    %v907 = stablehlo.multiply %v906, %v905 : tensor<32x302592xf32>
    %v908 = stablehlo.add %v903, %v907 : tensor<32x302592xf32>
    %v909 = stablehlo.constant dense<0.7978845608028654> : tensor<32x302592xf32>
    %v910 = stablehlo.multiply %v909, %v908 : tensor<32x302592xf32>
    %v911 = stablehlo.tanh %v910 : tensor<32x302592xf32>
    %v912 = stablehlo.constant dense<1.0> : tensor<32x302592xf32>
    %v913 = stablehlo.add %v912, %v911 : tensor<32x302592xf32>
    %v914 = stablehlo.constant dense<0.5> : tensor<32x302592xf32>
    %v915 = stablehlo.multiply %v914, %v903 : tensor<32x302592xf32>
    %v916 = stablehlo.multiply %v915, %v913 : tensor<32x302592xf32>
    %v917 = stablehlo.reshape %v916 : (tensor<32x302592xf32>) -> tensor<32x197x1536xf32>
    %v918 = stablehlo.dot_general %v917, %b2_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x1536xf32>, tensor<1536x384xf32>) -> tensor<32x197x384xf32>
    %v919 = stablehlo.broadcast_in_dim %b2_bfc2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v920 = stablehlo.add %v918, %v919 : tensor<32x197x384xf32>
    %v921 = stablehlo.reshape %v920 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v922 = stablehlo.add %v870, %v921 : tensor<32x75648xf32>
    %v923 = stablehlo.reshape %v922 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v924 = stablehlo.constant dense<0.0> : tensor<f32>
    %v925 = stablehlo.constant dense<384.0> : tensor<32x197x384xf32>
    %v926 = stablehlo.constant dense<1.0e-5> : tensor<32x197x384xf32>
    %v927 = stablehlo.reduce(%v923 init: %v924) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v928 = stablehlo.broadcast_in_dim %v927, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v929 = stablehlo.divide %v928, %v925 : tensor<32x197x384xf32>
    %v930 = stablehlo.subtract %v923, %v929 : tensor<32x197x384xf32>
    %v931 = stablehlo.multiply %v930, %v930 : tensor<32x197x384xf32>
    %v932 = stablehlo.reduce(%v931 init: %v924) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v933 = stablehlo.broadcast_in_dim %v932, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v934 = stablehlo.divide %v933, %v925 : tensor<32x197x384xf32>
    %v935 = stablehlo.add %v934, %v926 : tensor<32x197x384xf32>
    %v936 = stablehlo.rsqrt %v935 : tensor<32x197x384xf32>
    %v937 = stablehlo.multiply %v930, %v936 : tensor<32x197x384xf32>
    %v938 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v939 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v940 = stablehlo.multiply %v937, %v938 : tensor<32x197x384xf32>
    %v941 = stablehlo.add %v940, %v939 : tensor<32x197x384xf32>
    %v942 = stablehlo.reshape %v941 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v943 = stablehlo.reshape %v942 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v944 = stablehlo.broadcast_in_dim %b3_g1, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v945 = stablehlo.multiply %v943, %v944 : tensor<32x197x384xf32>
    %v946 = stablehlo.reshape %v945 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v947 = stablehlo.reshape %v946 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v948 = stablehlo.broadcast_in_dim %b3_bt1, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v949 = stablehlo.add %v947, %v948 : tensor<32x197x384xf32>
    %v950 = stablehlo.reshape %v949 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v951 = stablehlo.reshape %v950 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v952 = stablehlo.dot_general %v951, %b3_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v953 = stablehlo.broadcast_in_dim %b3_bq, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v954 = stablehlo.add %v952, %v953 : tensor<32x197x384xf32>
    %v955 = stablehlo.reshape %v954 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v956 = stablehlo.reshape %v950 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v957 = stablehlo.dot_general %v956, %b3_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v958 = stablehlo.broadcast_in_dim %b3_bk, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v959 = stablehlo.add %v957, %v958 : tensor<32x197x384xf32>
    %v960 = stablehlo.reshape %v959 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v961 = stablehlo.reshape %v950 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v962 = stablehlo.dot_general %v961, %b3_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v963 = stablehlo.broadcast_in_dim %b3_bv, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v964 = stablehlo.add %v962, %v963 : tensor<32x197x384xf32>
    %v965 = stablehlo.reshape %v964 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v966 = stablehlo.reshape %v955 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v967 = stablehlo.slice %v966 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v968 = stablehlo.reshape %v967 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v969 = stablehlo.reshape %v960 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v970 = stablehlo.slice %v969 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v971 = stablehlo.reshape %v970 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v972 = stablehlo.reshape %v965 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v973 = stablehlo.slice %v972 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v974 = stablehlo.reshape %v973 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v975 = stablehlo.reshape %v971 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v976 = stablehlo.transpose %v975, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v977 = stablehlo.reshape %v976 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v978 = stablehlo.reshape %v968 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v979 = stablehlo.reshape %v977 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v980 = stablehlo.dot_general %v978, %v979, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v981 = stablehlo.reshape %v980 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v982 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v983 = stablehlo.multiply %v981, %v982 : tensor<32x38809xf32>
    %v984 = stablehlo.reshape %v983 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v985 = stablehlo.constant dense<0.0> : tensor<f32>
    %v986 = stablehlo.exponential %v984 : tensor<32x197x197xf32>
    %v987 = stablehlo.reduce(%v986 init: %v985) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v988 = stablehlo.broadcast_in_dim %v987, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v989 = stablehlo.divide %v986, %v988 : tensor<32x197x197xf32>
    %v990 = stablehlo.reshape %v989 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v991 = stablehlo.reshape %v990 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v992 = stablehlo.reshape %v974 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v993 = stablehlo.dot_general %v991, %v992, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v994 = stablehlo.reshape %v993 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v995 = stablehlo.reshape %v994 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v996 = stablehlo.constant dense<0.0> : tensor<f32>
    %v997 = stablehlo.pad %v995, %v996, low = [0, 0, 0], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v998 = stablehlo.reshape %v997 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v999 = stablehlo.reshape %v955 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1000 = stablehlo.slice %v999 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1001 = stablehlo.reshape %v1000 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1002 = stablehlo.reshape %v960 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1003 = stablehlo.slice %v1002 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1004 = stablehlo.reshape %v1003 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1005 = stablehlo.reshape %v965 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1006 = stablehlo.slice %v1005 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1007 = stablehlo.reshape %v1006 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1008 = stablehlo.reshape %v1004 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1009 = stablehlo.transpose %v1008, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1010 = stablehlo.reshape %v1009 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1011 = stablehlo.reshape %v1001 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1012 = stablehlo.reshape %v1010 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1013 = stablehlo.dot_general %v1011, %v1012, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1014 = stablehlo.reshape %v1013 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1015 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1016 = stablehlo.multiply %v1014, %v1015 : tensor<32x38809xf32>
    %v1017 = stablehlo.reshape %v1016 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1018 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1019 = stablehlo.exponential %v1017 : tensor<32x197x197xf32>
    %v1020 = stablehlo.reduce(%v1019 init: %v1018) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1021 = stablehlo.broadcast_in_dim %v1020, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1022 = stablehlo.divide %v1019, %v1021 : tensor<32x197x197xf32>
    %v1023 = stablehlo.reshape %v1022 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1024 = stablehlo.reshape %v1023 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1025 = stablehlo.reshape %v1007 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1026 = stablehlo.dot_general %v1024, %v1025, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1027 = stablehlo.reshape %v1026 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1028 = stablehlo.reshape %v1027 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1029 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1030 = stablehlo.pad %v1028, %v1029, low = [0, 0, 64], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v1031 = stablehlo.reshape %v1030 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1032 = stablehlo.add %v998, %v1031 : tensor<32x75648xf32>
    %v1033 = stablehlo.reshape %v955 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1034 = stablehlo.slice %v1033 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1035 = stablehlo.reshape %v1034 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1036 = stablehlo.reshape %v960 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1037 = stablehlo.slice %v1036 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1038 = stablehlo.reshape %v1037 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1039 = stablehlo.reshape %v965 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1040 = stablehlo.slice %v1039 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1041 = stablehlo.reshape %v1040 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1042 = stablehlo.reshape %v1038 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1043 = stablehlo.transpose %v1042, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1044 = stablehlo.reshape %v1043 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1045 = stablehlo.reshape %v1035 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1046 = stablehlo.reshape %v1044 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1047 = stablehlo.dot_general %v1045, %v1046, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1048 = stablehlo.reshape %v1047 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1049 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1050 = stablehlo.multiply %v1048, %v1049 : tensor<32x38809xf32>
    %v1051 = stablehlo.reshape %v1050 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1052 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1053 = stablehlo.exponential %v1051 : tensor<32x197x197xf32>
    %v1054 = stablehlo.reduce(%v1053 init: %v1052) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1055 = stablehlo.broadcast_in_dim %v1054, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1056 = stablehlo.divide %v1053, %v1055 : tensor<32x197x197xf32>
    %v1057 = stablehlo.reshape %v1056 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1058 = stablehlo.reshape %v1057 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1059 = stablehlo.reshape %v1041 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1060 = stablehlo.dot_general %v1058, %v1059, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1061 = stablehlo.reshape %v1060 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1062 = stablehlo.reshape %v1061 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1063 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1064 = stablehlo.pad %v1062, %v1063, low = [0, 0, 128], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v1065 = stablehlo.reshape %v1064 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1066 = stablehlo.add %v1032, %v1065 : tensor<32x75648xf32>
    %v1067 = stablehlo.reshape %v955 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1068 = stablehlo.slice %v1067 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1069 = stablehlo.reshape %v1068 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1070 = stablehlo.reshape %v960 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1071 = stablehlo.slice %v1070 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1072 = stablehlo.reshape %v1071 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1073 = stablehlo.reshape %v965 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1074 = stablehlo.slice %v1073 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1075 = stablehlo.reshape %v1074 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1076 = stablehlo.reshape %v1072 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1077 = stablehlo.transpose %v1076, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1078 = stablehlo.reshape %v1077 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1079 = stablehlo.reshape %v1069 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1080 = stablehlo.reshape %v1078 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1081 = stablehlo.dot_general %v1079, %v1080, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1082 = stablehlo.reshape %v1081 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1083 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1084 = stablehlo.multiply %v1082, %v1083 : tensor<32x38809xf32>
    %v1085 = stablehlo.reshape %v1084 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1086 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1087 = stablehlo.exponential %v1085 : tensor<32x197x197xf32>
    %v1088 = stablehlo.reduce(%v1087 init: %v1086) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1089 = stablehlo.broadcast_in_dim %v1088, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1090 = stablehlo.divide %v1087, %v1089 : tensor<32x197x197xf32>
    %v1091 = stablehlo.reshape %v1090 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1092 = stablehlo.reshape %v1091 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1093 = stablehlo.reshape %v1075 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1094 = stablehlo.dot_general %v1092, %v1093, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1095 = stablehlo.reshape %v1094 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1096 = stablehlo.reshape %v1095 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1097 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1098 = stablehlo.pad %v1096, %v1097, low = [0, 0, 192], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v1099 = stablehlo.reshape %v1098 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1100 = stablehlo.add %v1066, %v1099 : tensor<32x75648xf32>
    %v1101 = stablehlo.reshape %v955 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1102 = stablehlo.slice %v1101 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1103 = stablehlo.reshape %v1102 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1104 = stablehlo.reshape %v960 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1105 = stablehlo.slice %v1104 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1106 = stablehlo.reshape %v1105 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1107 = stablehlo.reshape %v965 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1108 = stablehlo.slice %v1107 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1109 = stablehlo.reshape %v1108 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1110 = stablehlo.reshape %v1106 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1111 = stablehlo.transpose %v1110, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1112 = stablehlo.reshape %v1111 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1113 = stablehlo.reshape %v1103 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1114 = stablehlo.reshape %v1112 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1115 = stablehlo.dot_general %v1113, %v1114, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1116 = stablehlo.reshape %v1115 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1117 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1118 = stablehlo.multiply %v1116, %v1117 : tensor<32x38809xf32>
    %v1119 = stablehlo.reshape %v1118 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1120 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1121 = stablehlo.exponential %v1119 : tensor<32x197x197xf32>
    %v1122 = stablehlo.reduce(%v1121 init: %v1120) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1123 = stablehlo.broadcast_in_dim %v1122, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1124 = stablehlo.divide %v1121, %v1123 : tensor<32x197x197xf32>
    %v1125 = stablehlo.reshape %v1124 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1126 = stablehlo.reshape %v1125 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1127 = stablehlo.reshape %v1109 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1128 = stablehlo.dot_general %v1126, %v1127, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1129 = stablehlo.reshape %v1128 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1130 = stablehlo.reshape %v1129 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1131 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1132 = stablehlo.pad %v1130, %v1131, low = [0, 0, 256], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v1133 = stablehlo.reshape %v1132 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1134 = stablehlo.add %v1100, %v1133 : tensor<32x75648xf32>
    %v1135 = stablehlo.reshape %v955 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1136 = stablehlo.slice %v1135 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1137 = stablehlo.reshape %v1136 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1138 = stablehlo.reshape %v960 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1139 = stablehlo.slice %v1138 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1140 = stablehlo.reshape %v1139 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1141 = stablehlo.reshape %v965 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1142 = stablehlo.slice %v1141 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1143 = stablehlo.reshape %v1142 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1144 = stablehlo.reshape %v1140 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1145 = stablehlo.transpose %v1144, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1146 = stablehlo.reshape %v1145 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1147 = stablehlo.reshape %v1137 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1148 = stablehlo.reshape %v1146 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1149 = stablehlo.dot_general %v1147, %v1148, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1150 = stablehlo.reshape %v1149 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1151 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1152 = stablehlo.multiply %v1150, %v1151 : tensor<32x38809xf32>
    %v1153 = stablehlo.reshape %v1152 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1154 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1155 = stablehlo.exponential %v1153 : tensor<32x197x197xf32>
    %v1156 = stablehlo.reduce(%v1155 init: %v1154) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1157 = stablehlo.broadcast_in_dim %v1156, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1158 = stablehlo.divide %v1155, %v1157 : tensor<32x197x197xf32>
    %v1159 = stablehlo.reshape %v1158 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1160 = stablehlo.reshape %v1159 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1161 = stablehlo.reshape %v1143 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1162 = stablehlo.dot_general %v1160, %v1161, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1163 = stablehlo.reshape %v1162 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1164 = stablehlo.reshape %v1163 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1165 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1166 = stablehlo.pad %v1164, %v1165, low = [0, 0, 320], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v1167 = stablehlo.reshape %v1166 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1168 = stablehlo.add %v1134, %v1167 : tensor<32x75648xf32>
    %v1169 = stablehlo.reshape %v1168 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1170 = stablehlo.dot_general %v1169, %b3_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v1171 = stablehlo.broadcast_in_dim %b3_bo, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1172 = stablehlo.add %v1170, %v1171 : tensor<32x197x384xf32>
    %v1173 = stablehlo.reshape %v1172 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1174 = stablehlo.add %v922, %v1173 : tensor<32x75648xf32>
    %v1175 = stablehlo.reshape %v1174 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1176 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1177 = stablehlo.constant dense<384.0> : tensor<32x197x384xf32>
    %v1178 = stablehlo.constant dense<1.0e-5> : tensor<32x197x384xf32>
    %v1179 = stablehlo.reduce(%v1175 init: %v1176) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1180 = stablehlo.broadcast_in_dim %v1179, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v1181 = stablehlo.divide %v1180, %v1177 : tensor<32x197x384xf32>
    %v1182 = stablehlo.subtract %v1175, %v1181 : tensor<32x197x384xf32>
    %v1183 = stablehlo.multiply %v1182, %v1182 : tensor<32x197x384xf32>
    %v1184 = stablehlo.reduce(%v1183 init: %v1176) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1185 = stablehlo.broadcast_in_dim %v1184, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v1186 = stablehlo.divide %v1185, %v1177 : tensor<32x197x384xf32>
    %v1187 = stablehlo.add %v1186, %v1178 : tensor<32x197x384xf32>
    %v1188 = stablehlo.rsqrt %v1187 : tensor<32x197x384xf32>
    %v1189 = stablehlo.multiply %v1182, %v1188 : tensor<32x197x384xf32>
    %v1190 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v1191 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v1192 = stablehlo.multiply %v1189, %v1190 : tensor<32x197x384xf32>
    %v1193 = stablehlo.add %v1192, %v1191 : tensor<32x197x384xf32>
    %v1194 = stablehlo.reshape %v1193 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1195 = stablehlo.reshape %v1194 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1196 = stablehlo.broadcast_in_dim %b3_g2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1197 = stablehlo.multiply %v1195, %v1196 : tensor<32x197x384xf32>
    %v1198 = stablehlo.reshape %v1197 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1199 = stablehlo.reshape %v1198 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1200 = stablehlo.broadcast_in_dim %b3_bt2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1201 = stablehlo.add %v1199, %v1200 : tensor<32x197x384xf32>
    %v1202 = stablehlo.reshape %v1201 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1203 = stablehlo.reshape %v1202 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1204 = stablehlo.dot_general %v1203, %b3_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x1536xf32>) -> tensor<32x197x1536xf32>
    %v1205 = stablehlo.broadcast_in_dim %b3_bfc1, dims = [2] : (tensor<1536xf32>) -> tensor<32x197x1536xf32>
    %v1206 = stablehlo.add %v1204, %v1205 : tensor<32x197x1536xf32>
    %v1207 = stablehlo.reshape %v1206 : (tensor<32x197x1536xf32>) -> tensor<32x302592xf32>
    %v1208 = stablehlo.multiply %v1207, %v1207 : tensor<32x302592xf32>
    %v1209 = stablehlo.multiply %v1208, %v1207 : tensor<32x302592xf32>
    %v1210 = stablehlo.constant dense<0.044715> : tensor<32x302592xf32>
    %v1211 = stablehlo.multiply %v1210, %v1209 : tensor<32x302592xf32>
    %v1212 = stablehlo.add %v1207, %v1211 : tensor<32x302592xf32>
    %v1213 = stablehlo.constant dense<0.7978845608028654> : tensor<32x302592xf32>
    %v1214 = stablehlo.multiply %v1213, %v1212 : tensor<32x302592xf32>
    %v1215 = stablehlo.tanh %v1214 : tensor<32x302592xf32>
    %v1216 = stablehlo.constant dense<1.0> : tensor<32x302592xf32>
    %v1217 = stablehlo.add %v1216, %v1215 : tensor<32x302592xf32>
    %v1218 = stablehlo.constant dense<0.5> : tensor<32x302592xf32>
    %v1219 = stablehlo.multiply %v1218, %v1207 : tensor<32x302592xf32>
    %v1220 = stablehlo.multiply %v1219, %v1217 : tensor<32x302592xf32>
    %v1221 = stablehlo.reshape %v1220 : (tensor<32x302592xf32>) -> tensor<32x197x1536xf32>
    %v1222 = stablehlo.dot_general %v1221, %b3_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x1536xf32>, tensor<1536x384xf32>) -> tensor<32x197x384xf32>
    %v1223 = stablehlo.broadcast_in_dim %b3_bfc2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1224 = stablehlo.add %v1222, %v1223 : tensor<32x197x384xf32>
    %v1225 = stablehlo.reshape %v1224 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1226 = stablehlo.add %v1174, %v1225 : tensor<32x75648xf32>
    %v1227 = stablehlo.reshape %v1226 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1228 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1229 = stablehlo.constant dense<384.0> : tensor<32x197x384xf32>
    %v1230 = stablehlo.constant dense<1.0e-5> : tensor<32x197x384xf32>
    %v1231 = stablehlo.reduce(%v1227 init: %v1228) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1232 = stablehlo.broadcast_in_dim %v1231, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v1233 = stablehlo.divide %v1232, %v1229 : tensor<32x197x384xf32>
    %v1234 = stablehlo.subtract %v1227, %v1233 : tensor<32x197x384xf32>
    %v1235 = stablehlo.multiply %v1234, %v1234 : tensor<32x197x384xf32>
    %v1236 = stablehlo.reduce(%v1235 init: %v1228) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1237 = stablehlo.broadcast_in_dim %v1236, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v1238 = stablehlo.divide %v1237, %v1229 : tensor<32x197x384xf32>
    %v1239 = stablehlo.add %v1238, %v1230 : tensor<32x197x384xf32>
    %v1240 = stablehlo.rsqrt %v1239 : tensor<32x197x384xf32>
    %v1241 = stablehlo.multiply %v1234, %v1240 : tensor<32x197x384xf32>
    %v1242 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v1243 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v1244 = stablehlo.multiply %v1241, %v1242 : tensor<32x197x384xf32>
    %v1245 = stablehlo.add %v1244, %v1243 : tensor<32x197x384xf32>
    %v1246 = stablehlo.reshape %v1245 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1247 = stablehlo.reshape %v1246 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1248 = stablehlo.broadcast_in_dim %b4_g1, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1249 = stablehlo.multiply %v1247, %v1248 : tensor<32x197x384xf32>
    %v1250 = stablehlo.reshape %v1249 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1251 = stablehlo.reshape %v1250 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1252 = stablehlo.broadcast_in_dim %b4_bt1, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1253 = stablehlo.add %v1251, %v1252 : tensor<32x197x384xf32>
    %v1254 = stablehlo.reshape %v1253 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1255 = stablehlo.reshape %v1254 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1256 = stablehlo.dot_general %v1255, %b4_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v1257 = stablehlo.broadcast_in_dim %b4_bq, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1258 = stablehlo.add %v1256, %v1257 : tensor<32x197x384xf32>
    %v1259 = stablehlo.reshape %v1258 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1260 = stablehlo.reshape %v1254 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1261 = stablehlo.dot_general %v1260, %b4_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v1262 = stablehlo.broadcast_in_dim %b4_bk, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1263 = stablehlo.add %v1261, %v1262 : tensor<32x197x384xf32>
    %v1264 = stablehlo.reshape %v1263 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1265 = stablehlo.reshape %v1254 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1266 = stablehlo.dot_general %v1265, %b4_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v1267 = stablehlo.broadcast_in_dim %b4_bv, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1268 = stablehlo.add %v1266, %v1267 : tensor<32x197x384xf32>
    %v1269 = stablehlo.reshape %v1268 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1270 = stablehlo.reshape %v1259 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1271 = stablehlo.slice %v1270 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1272 = stablehlo.reshape %v1271 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1273 = stablehlo.reshape %v1264 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1274 = stablehlo.slice %v1273 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1275 = stablehlo.reshape %v1274 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1276 = stablehlo.reshape %v1269 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1277 = stablehlo.slice %v1276 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1278 = stablehlo.reshape %v1277 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1279 = stablehlo.reshape %v1275 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1280 = stablehlo.transpose %v1279, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1281 = stablehlo.reshape %v1280 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1282 = stablehlo.reshape %v1272 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1283 = stablehlo.reshape %v1281 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1284 = stablehlo.dot_general %v1282, %v1283, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1285 = stablehlo.reshape %v1284 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1286 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1287 = stablehlo.multiply %v1285, %v1286 : tensor<32x38809xf32>
    %v1288 = stablehlo.reshape %v1287 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1289 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1290 = stablehlo.exponential %v1288 : tensor<32x197x197xf32>
    %v1291 = stablehlo.reduce(%v1290 init: %v1289) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1292 = stablehlo.broadcast_in_dim %v1291, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1293 = stablehlo.divide %v1290, %v1292 : tensor<32x197x197xf32>
    %v1294 = stablehlo.reshape %v1293 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1295 = stablehlo.reshape %v1294 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1296 = stablehlo.reshape %v1278 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1297 = stablehlo.dot_general %v1295, %v1296, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1298 = stablehlo.reshape %v1297 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1299 = stablehlo.reshape %v1298 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1300 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1301 = stablehlo.pad %v1299, %v1300, low = [0, 0, 0], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v1302 = stablehlo.reshape %v1301 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1303 = stablehlo.reshape %v1259 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1304 = stablehlo.slice %v1303 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1305 = stablehlo.reshape %v1304 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1306 = stablehlo.reshape %v1264 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1307 = stablehlo.slice %v1306 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1308 = stablehlo.reshape %v1307 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1309 = stablehlo.reshape %v1269 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1310 = stablehlo.slice %v1309 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1311 = stablehlo.reshape %v1310 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1312 = stablehlo.reshape %v1308 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1313 = stablehlo.transpose %v1312, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1314 = stablehlo.reshape %v1313 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1315 = stablehlo.reshape %v1305 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1316 = stablehlo.reshape %v1314 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1317 = stablehlo.dot_general %v1315, %v1316, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1318 = stablehlo.reshape %v1317 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1319 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1320 = stablehlo.multiply %v1318, %v1319 : tensor<32x38809xf32>
    %v1321 = stablehlo.reshape %v1320 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1322 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1323 = stablehlo.exponential %v1321 : tensor<32x197x197xf32>
    %v1324 = stablehlo.reduce(%v1323 init: %v1322) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1325 = stablehlo.broadcast_in_dim %v1324, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1326 = stablehlo.divide %v1323, %v1325 : tensor<32x197x197xf32>
    %v1327 = stablehlo.reshape %v1326 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1328 = stablehlo.reshape %v1327 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1329 = stablehlo.reshape %v1311 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1330 = stablehlo.dot_general %v1328, %v1329, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1331 = stablehlo.reshape %v1330 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1332 = stablehlo.reshape %v1331 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1333 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1334 = stablehlo.pad %v1332, %v1333, low = [0, 0, 64], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v1335 = stablehlo.reshape %v1334 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1336 = stablehlo.add %v1302, %v1335 : tensor<32x75648xf32>
    %v1337 = stablehlo.reshape %v1259 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1338 = stablehlo.slice %v1337 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1339 = stablehlo.reshape %v1338 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1340 = stablehlo.reshape %v1264 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1341 = stablehlo.slice %v1340 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1342 = stablehlo.reshape %v1341 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1343 = stablehlo.reshape %v1269 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1344 = stablehlo.slice %v1343 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1345 = stablehlo.reshape %v1344 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1346 = stablehlo.reshape %v1342 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1347 = stablehlo.transpose %v1346, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1348 = stablehlo.reshape %v1347 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1349 = stablehlo.reshape %v1339 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1350 = stablehlo.reshape %v1348 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1351 = stablehlo.dot_general %v1349, %v1350, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1352 = stablehlo.reshape %v1351 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1353 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1354 = stablehlo.multiply %v1352, %v1353 : tensor<32x38809xf32>
    %v1355 = stablehlo.reshape %v1354 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1356 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1357 = stablehlo.exponential %v1355 : tensor<32x197x197xf32>
    %v1358 = stablehlo.reduce(%v1357 init: %v1356) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1359 = stablehlo.broadcast_in_dim %v1358, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1360 = stablehlo.divide %v1357, %v1359 : tensor<32x197x197xf32>
    %v1361 = stablehlo.reshape %v1360 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1362 = stablehlo.reshape %v1361 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1363 = stablehlo.reshape %v1345 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1364 = stablehlo.dot_general %v1362, %v1363, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1365 = stablehlo.reshape %v1364 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1366 = stablehlo.reshape %v1365 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1367 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1368 = stablehlo.pad %v1366, %v1367, low = [0, 0, 128], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v1369 = stablehlo.reshape %v1368 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1370 = stablehlo.add %v1336, %v1369 : tensor<32x75648xf32>
    %v1371 = stablehlo.reshape %v1259 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1372 = stablehlo.slice %v1371 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1373 = stablehlo.reshape %v1372 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1374 = stablehlo.reshape %v1264 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1375 = stablehlo.slice %v1374 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1376 = stablehlo.reshape %v1375 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1377 = stablehlo.reshape %v1269 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1378 = stablehlo.slice %v1377 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1379 = stablehlo.reshape %v1378 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1380 = stablehlo.reshape %v1376 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1381 = stablehlo.transpose %v1380, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1382 = stablehlo.reshape %v1381 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1383 = stablehlo.reshape %v1373 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1384 = stablehlo.reshape %v1382 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1385 = stablehlo.dot_general %v1383, %v1384, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1386 = stablehlo.reshape %v1385 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1387 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1388 = stablehlo.multiply %v1386, %v1387 : tensor<32x38809xf32>
    %v1389 = stablehlo.reshape %v1388 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1390 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1391 = stablehlo.exponential %v1389 : tensor<32x197x197xf32>
    %v1392 = stablehlo.reduce(%v1391 init: %v1390) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1393 = stablehlo.broadcast_in_dim %v1392, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1394 = stablehlo.divide %v1391, %v1393 : tensor<32x197x197xf32>
    %v1395 = stablehlo.reshape %v1394 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1396 = stablehlo.reshape %v1395 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1397 = stablehlo.reshape %v1379 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1398 = stablehlo.dot_general %v1396, %v1397, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1399 = stablehlo.reshape %v1398 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1400 = stablehlo.reshape %v1399 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1401 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1402 = stablehlo.pad %v1400, %v1401, low = [0, 0, 192], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v1403 = stablehlo.reshape %v1402 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1404 = stablehlo.add %v1370, %v1403 : tensor<32x75648xf32>
    %v1405 = stablehlo.reshape %v1259 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1406 = stablehlo.slice %v1405 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1407 = stablehlo.reshape %v1406 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1408 = stablehlo.reshape %v1264 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1409 = stablehlo.slice %v1408 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1410 = stablehlo.reshape %v1409 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1411 = stablehlo.reshape %v1269 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1412 = stablehlo.slice %v1411 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1413 = stablehlo.reshape %v1412 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1414 = stablehlo.reshape %v1410 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1415 = stablehlo.transpose %v1414, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1416 = stablehlo.reshape %v1415 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1417 = stablehlo.reshape %v1407 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1418 = stablehlo.reshape %v1416 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1419 = stablehlo.dot_general %v1417, %v1418, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1420 = stablehlo.reshape %v1419 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1421 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1422 = stablehlo.multiply %v1420, %v1421 : tensor<32x38809xf32>
    %v1423 = stablehlo.reshape %v1422 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1424 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1425 = stablehlo.exponential %v1423 : tensor<32x197x197xf32>
    %v1426 = stablehlo.reduce(%v1425 init: %v1424) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1427 = stablehlo.broadcast_in_dim %v1426, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1428 = stablehlo.divide %v1425, %v1427 : tensor<32x197x197xf32>
    %v1429 = stablehlo.reshape %v1428 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1430 = stablehlo.reshape %v1429 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1431 = stablehlo.reshape %v1413 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1432 = stablehlo.dot_general %v1430, %v1431, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1433 = stablehlo.reshape %v1432 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1434 = stablehlo.reshape %v1433 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1435 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1436 = stablehlo.pad %v1434, %v1435, low = [0, 0, 256], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v1437 = stablehlo.reshape %v1436 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1438 = stablehlo.add %v1404, %v1437 : tensor<32x75648xf32>
    %v1439 = stablehlo.reshape %v1259 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1440 = stablehlo.slice %v1439 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1441 = stablehlo.reshape %v1440 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1442 = stablehlo.reshape %v1264 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1443 = stablehlo.slice %v1442 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1444 = stablehlo.reshape %v1443 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1445 = stablehlo.reshape %v1269 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1446 = stablehlo.slice %v1445 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1447 = stablehlo.reshape %v1446 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1448 = stablehlo.reshape %v1444 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1449 = stablehlo.transpose %v1448, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1450 = stablehlo.reshape %v1449 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1451 = stablehlo.reshape %v1441 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1452 = stablehlo.reshape %v1450 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1453 = stablehlo.dot_general %v1451, %v1452, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1454 = stablehlo.reshape %v1453 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1455 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1456 = stablehlo.multiply %v1454, %v1455 : tensor<32x38809xf32>
    %v1457 = stablehlo.reshape %v1456 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1458 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1459 = stablehlo.exponential %v1457 : tensor<32x197x197xf32>
    %v1460 = stablehlo.reduce(%v1459 init: %v1458) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1461 = stablehlo.broadcast_in_dim %v1460, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1462 = stablehlo.divide %v1459, %v1461 : tensor<32x197x197xf32>
    %v1463 = stablehlo.reshape %v1462 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1464 = stablehlo.reshape %v1463 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1465 = stablehlo.reshape %v1447 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1466 = stablehlo.dot_general %v1464, %v1465, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1467 = stablehlo.reshape %v1466 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1468 = stablehlo.reshape %v1467 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1469 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1470 = stablehlo.pad %v1468, %v1469, low = [0, 0, 320], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v1471 = stablehlo.reshape %v1470 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1472 = stablehlo.add %v1438, %v1471 : tensor<32x75648xf32>
    %v1473 = stablehlo.reshape %v1472 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1474 = stablehlo.dot_general %v1473, %b4_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v1475 = stablehlo.broadcast_in_dim %b4_bo, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1476 = stablehlo.add %v1474, %v1475 : tensor<32x197x384xf32>
    %v1477 = stablehlo.reshape %v1476 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1478 = stablehlo.add %v1226, %v1477 : tensor<32x75648xf32>
    %v1479 = stablehlo.reshape %v1478 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1480 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1481 = stablehlo.constant dense<384.0> : tensor<32x197x384xf32>
    %v1482 = stablehlo.constant dense<1.0e-5> : tensor<32x197x384xf32>
    %v1483 = stablehlo.reduce(%v1479 init: %v1480) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1484 = stablehlo.broadcast_in_dim %v1483, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v1485 = stablehlo.divide %v1484, %v1481 : tensor<32x197x384xf32>
    %v1486 = stablehlo.subtract %v1479, %v1485 : tensor<32x197x384xf32>
    %v1487 = stablehlo.multiply %v1486, %v1486 : tensor<32x197x384xf32>
    %v1488 = stablehlo.reduce(%v1487 init: %v1480) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1489 = stablehlo.broadcast_in_dim %v1488, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v1490 = stablehlo.divide %v1489, %v1481 : tensor<32x197x384xf32>
    %v1491 = stablehlo.add %v1490, %v1482 : tensor<32x197x384xf32>
    %v1492 = stablehlo.rsqrt %v1491 : tensor<32x197x384xf32>
    %v1493 = stablehlo.multiply %v1486, %v1492 : tensor<32x197x384xf32>
    %v1494 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v1495 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v1496 = stablehlo.multiply %v1493, %v1494 : tensor<32x197x384xf32>
    %v1497 = stablehlo.add %v1496, %v1495 : tensor<32x197x384xf32>
    %v1498 = stablehlo.reshape %v1497 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1499 = stablehlo.reshape %v1498 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1500 = stablehlo.broadcast_in_dim %b4_g2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1501 = stablehlo.multiply %v1499, %v1500 : tensor<32x197x384xf32>
    %v1502 = stablehlo.reshape %v1501 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1503 = stablehlo.reshape %v1502 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1504 = stablehlo.broadcast_in_dim %b4_bt2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1505 = stablehlo.add %v1503, %v1504 : tensor<32x197x384xf32>
    %v1506 = stablehlo.reshape %v1505 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1507 = stablehlo.reshape %v1506 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1508 = stablehlo.dot_general %v1507, %b4_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x1536xf32>) -> tensor<32x197x1536xf32>
    %v1509 = stablehlo.broadcast_in_dim %b4_bfc1, dims = [2] : (tensor<1536xf32>) -> tensor<32x197x1536xf32>
    %v1510 = stablehlo.add %v1508, %v1509 : tensor<32x197x1536xf32>
    %v1511 = stablehlo.reshape %v1510 : (tensor<32x197x1536xf32>) -> tensor<32x302592xf32>
    %v1512 = stablehlo.multiply %v1511, %v1511 : tensor<32x302592xf32>
    %v1513 = stablehlo.multiply %v1512, %v1511 : tensor<32x302592xf32>
    %v1514 = stablehlo.constant dense<0.044715> : tensor<32x302592xf32>
    %v1515 = stablehlo.multiply %v1514, %v1513 : tensor<32x302592xf32>
    %v1516 = stablehlo.add %v1511, %v1515 : tensor<32x302592xf32>
    %v1517 = stablehlo.constant dense<0.7978845608028654> : tensor<32x302592xf32>
    %v1518 = stablehlo.multiply %v1517, %v1516 : tensor<32x302592xf32>
    %v1519 = stablehlo.tanh %v1518 : tensor<32x302592xf32>
    %v1520 = stablehlo.constant dense<1.0> : tensor<32x302592xf32>
    %v1521 = stablehlo.add %v1520, %v1519 : tensor<32x302592xf32>
    %v1522 = stablehlo.constant dense<0.5> : tensor<32x302592xf32>
    %v1523 = stablehlo.multiply %v1522, %v1511 : tensor<32x302592xf32>
    %v1524 = stablehlo.multiply %v1523, %v1521 : tensor<32x302592xf32>
    %v1525 = stablehlo.reshape %v1524 : (tensor<32x302592xf32>) -> tensor<32x197x1536xf32>
    %v1526 = stablehlo.dot_general %v1525, %b4_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x1536xf32>, tensor<1536x384xf32>) -> tensor<32x197x384xf32>
    %v1527 = stablehlo.broadcast_in_dim %b4_bfc2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1528 = stablehlo.add %v1526, %v1527 : tensor<32x197x384xf32>
    %v1529 = stablehlo.reshape %v1528 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1530 = stablehlo.add %v1478, %v1529 : tensor<32x75648xf32>
    %v1531 = stablehlo.reshape %v1530 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1532 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1533 = stablehlo.constant dense<384.0> : tensor<32x197x384xf32>
    %v1534 = stablehlo.constant dense<1.0e-5> : tensor<32x197x384xf32>
    %v1535 = stablehlo.reduce(%v1531 init: %v1532) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1536 = stablehlo.broadcast_in_dim %v1535, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v1537 = stablehlo.divide %v1536, %v1533 : tensor<32x197x384xf32>
    %v1538 = stablehlo.subtract %v1531, %v1537 : tensor<32x197x384xf32>
    %v1539 = stablehlo.multiply %v1538, %v1538 : tensor<32x197x384xf32>
    %v1540 = stablehlo.reduce(%v1539 init: %v1532) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1541 = stablehlo.broadcast_in_dim %v1540, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v1542 = stablehlo.divide %v1541, %v1533 : tensor<32x197x384xf32>
    %v1543 = stablehlo.add %v1542, %v1534 : tensor<32x197x384xf32>
    %v1544 = stablehlo.rsqrt %v1543 : tensor<32x197x384xf32>
    %v1545 = stablehlo.multiply %v1538, %v1544 : tensor<32x197x384xf32>
    %v1546 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v1547 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v1548 = stablehlo.multiply %v1545, %v1546 : tensor<32x197x384xf32>
    %v1549 = stablehlo.add %v1548, %v1547 : tensor<32x197x384xf32>
    %v1550 = stablehlo.reshape %v1549 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1551 = stablehlo.reshape %v1550 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1552 = stablehlo.broadcast_in_dim %b5_g1, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1553 = stablehlo.multiply %v1551, %v1552 : tensor<32x197x384xf32>
    %v1554 = stablehlo.reshape %v1553 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1555 = stablehlo.reshape %v1554 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1556 = stablehlo.broadcast_in_dim %b5_bt1, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1557 = stablehlo.add %v1555, %v1556 : tensor<32x197x384xf32>
    %v1558 = stablehlo.reshape %v1557 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1559 = stablehlo.reshape %v1558 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1560 = stablehlo.dot_general %v1559, %b5_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v1561 = stablehlo.broadcast_in_dim %b5_bq, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1562 = stablehlo.add %v1560, %v1561 : tensor<32x197x384xf32>
    %v1563 = stablehlo.reshape %v1562 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1564 = stablehlo.reshape %v1558 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1565 = stablehlo.dot_general %v1564, %b5_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v1566 = stablehlo.broadcast_in_dim %b5_bk, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1567 = stablehlo.add %v1565, %v1566 : tensor<32x197x384xf32>
    %v1568 = stablehlo.reshape %v1567 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1569 = stablehlo.reshape %v1558 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1570 = stablehlo.dot_general %v1569, %b5_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v1571 = stablehlo.broadcast_in_dim %b5_bv, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1572 = stablehlo.add %v1570, %v1571 : tensor<32x197x384xf32>
    %v1573 = stablehlo.reshape %v1572 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1574 = stablehlo.reshape %v1563 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1575 = stablehlo.slice %v1574 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1576 = stablehlo.reshape %v1575 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1577 = stablehlo.reshape %v1568 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1578 = stablehlo.slice %v1577 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1579 = stablehlo.reshape %v1578 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1580 = stablehlo.reshape %v1573 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1581 = stablehlo.slice %v1580 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1582 = stablehlo.reshape %v1581 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1583 = stablehlo.reshape %v1579 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1584 = stablehlo.transpose %v1583, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1585 = stablehlo.reshape %v1584 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1586 = stablehlo.reshape %v1576 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1587 = stablehlo.reshape %v1585 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1588 = stablehlo.dot_general %v1586, %v1587, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1589 = stablehlo.reshape %v1588 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1590 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1591 = stablehlo.multiply %v1589, %v1590 : tensor<32x38809xf32>
    %v1592 = stablehlo.reshape %v1591 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1593 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1594 = stablehlo.exponential %v1592 : tensor<32x197x197xf32>
    %v1595 = stablehlo.reduce(%v1594 init: %v1593) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1596 = stablehlo.broadcast_in_dim %v1595, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1597 = stablehlo.divide %v1594, %v1596 : tensor<32x197x197xf32>
    %v1598 = stablehlo.reshape %v1597 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1599 = stablehlo.reshape %v1598 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1600 = stablehlo.reshape %v1582 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1601 = stablehlo.dot_general %v1599, %v1600, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1602 = stablehlo.reshape %v1601 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1603 = stablehlo.reshape %v1602 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1604 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1605 = stablehlo.pad %v1603, %v1604, low = [0, 0, 0], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v1606 = stablehlo.reshape %v1605 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1607 = stablehlo.reshape %v1563 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1608 = stablehlo.slice %v1607 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1609 = stablehlo.reshape %v1608 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1610 = stablehlo.reshape %v1568 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1611 = stablehlo.slice %v1610 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1612 = stablehlo.reshape %v1611 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1613 = stablehlo.reshape %v1573 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1614 = stablehlo.slice %v1613 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1615 = stablehlo.reshape %v1614 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1616 = stablehlo.reshape %v1612 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1617 = stablehlo.transpose %v1616, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1618 = stablehlo.reshape %v1617 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1619 = stablehlo.reshape %v1609 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1620 = stablehlo.reshape %v1618 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1621 = stablehlo.dot_general %v1619, %v1620, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1622 = stablehlo.reshape %v1621 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1623 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1624 = stablehlo.multiply %v1622, %v1623 : tensor<32x38809xf32>
    %v1625 = stablehlo.reshape %v1624 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1626 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1627 = stablehlo.exponential %v1625 : tensor<32x197x197xf32>
    %v1628 = stablehlo.reduce(%v1627 init: %v1626) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1629 = stablehlo.broadcast_in_dim %v1628, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1630 = stablehlo.divide %v1627, %v1629 : tensor<32x197x197xf32>
    %v1631 = stablehlo.reshape %v1630 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1632 = stablehlo.reshape %v1631 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1633 = stablehlo.reshape %v1615 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1634 = stablehlo.dot_general %v1632, %v1633, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1635 = stablehlo.reshape %v1634 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1636 = stablehlo.reshape %v1635 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1637 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1638 = stablehlo.pad %v1636, %v1637, low = [0, 0, 64], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v1639 = stablehlo.reshape %v1638 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1640 = stablehlo.add %v1606, %v1639 : tensor<32x75648xf32>
    %v1641 = stablehlo.reshape %v1563 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1642 = stablehlo.slice %v1641 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1643 = stablehlo.reshape %v1642 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1644 = stablehlo.reshape %v1568 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1645 = stablehlo.slice %v1644 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1646 = stablehlo.reshape %v1645 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1647 = stablehlo.reshape %v1573 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1648 = stablehlo.slice %v1647 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1649 = stablehlo.reshape %v1648 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1650 = stablehlo.reshape %v1646 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1651 = stablehlo.transpose %v1650, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1652 = stablehlo.reshape %v1651 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1653 = stablehlo.reshape %v1643 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1654 = stablehlo.reshape %v1652 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1655 = stablehlo.dot_general %v1653, %v1654, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1656 = stablehlo.reshape %v1655 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1657 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1658 = stablehlo.multiply %v1656, %v1657 : tensor<32x38809xf32>
    %v1659 = stablehlo.reshape %v1658 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1660 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1661 = stablehlo.exponential %v1659 : tensor<32x197x197xf32>
    %v1662 = stablehlo.reduce(%v1661 init: %v1660) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1663 = stablehlo.broadcast_in_dim %v1662, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1664 = stablehlo.divide %v1661, %v1663 : tensor<32x197x197xf32>
    %v1665 = stablehlo.reshape %v1664 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1666 = stablehlo.reshape %v1665 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1667 = stablehlo.reshape %v1649 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1668 = stablehlo.dot_general %v1666, %v1667, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1669 = stablehlo.reshape %v1668 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1670 = stablehlo.reshape %v1669 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1671 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1672 = stablehlo.pad %v1670, %v1671, low = [0, 0, 128], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v1673 = stablehlo.reshape %v1672 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1674 = stablehlo.add %v1640, %v1673 : tensor<32x75648xf32>
    %v1675 = stablehlo.reshape %v1563 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1676 = stablehlo.slice %v1675 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1677 = stablehlo.reshape %v1676 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1678 = stablehlo.reshape %v1568 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1679 = stablehlo.slice %v1678 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1680 = stablehlo.reshape %v1679 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1681 = stablehlo.reshape %v1573 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1682 = stablehlo.slice %v1681 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1683 = stablehlo.reshape %v1682 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1684 = stablehlo.reshape %v1680 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1685 = stablehlo.transpose %v1684, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1686 = stablehlo.reshape %v1685 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1687 = stablehlo.reshape %v1677 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1688 = stablehlo.reshape %v1686 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1689 = stablehlo.dot_general %v1687, %v1688, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1690 = stablehlo.reshape %v1689 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1691 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1692 = stablehlo.multiply %v1690, %v1691 : tensor<32x38809xf32>
    %v1693 = stablehlo.reshape %v1692 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1694 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1695 = stablehlo.exponential %v1693 : tensor<32x197x197xf32>
    %v1696 = stablehlo.reduce(%v1695 init: %v1694) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1697 = stablehlo.broadcast_in_dim %v1696, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1698 = stablehlo.divide %v1695, %v1697 : tensor<32x197x197xf32>
    %v1699 = stablehlo.reshape %v1698 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1700 = stablehlo.reshape %v1699 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1701 = stablehlo.reshape %v1683 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1702 = stablehlo.dot_general %v1700, %v1701, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1703 = stablehlo.reshape %v1702 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1704 = stablehlo.reshape %v1703 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1705 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1706 = stablehlo.pad %v1704, %v1705, low = [0, 0, 192], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v1707 = stablehlo.reshape %v1706 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1708 = stablehlo.add %v1674, %v1707 : tensor<32x75648xf32>
    %v1709 = stablehlo.reshape %v1563 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1710 = stablehlo.slice %v1709 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1711 = stablehlo.reshape %v1710 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1712 = stablehlo.reshape %v1568 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1713 = stablehlo.slice %v1712 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1714 = stablehlo.reshape %v1713 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1715 = stablehlo.reshape %v1573 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1716 = stablehlo.slice %v1715 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1717 = stablehlo.reshape %v1716 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1718 = stablehlo.reshape %v1714 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1719 = stablehlo.transpose %v1718, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1720 = stablehlo.reshape %v1719 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1721 = stablehlo.reshape %v1711 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1722 = stablehlo.reshape %v1720 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1723 = stablehlo.dot_general %v1721, %v1722, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1724 = stablehlo.reshape %v1723 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1725 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1726 = stablehlo.multiply %v1724, %v1725 : tensor<32x38809xf32>
    %v1727 = stablehlo.reshape %v1726 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1728 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1729 = stablehlo.exponential %v1727 : tensor<32x197x197xf32>
    %v1730 = stablehlo.reduce(%v1729 init: %v1728) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1731 = stablehlo.broadcast_in_dim %v1730, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1732 = stablehlo.divide %v1729, %v1731 : tensor<32x197x197xf32>
    %v1733 = stablehlo.reshape %v1732 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1734 = stablehlo.reshape %v1733 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1735 = stablehlo.reshape %v1717 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1736 = stablehlo.dot_general %v1734, %v1735, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1737 = stablehlo.reshape %v1736 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1738 = stablehlo.reshape %v1737 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1739 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1740 = stablehlo.pad %v1738, %v1739, low = [0, 0, 256], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v1741 = stablehlo.reshape %v1740 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1742 = stablehlo.add %v1708, %v1741 : tensor<32x75648xf32>
    %v1743 = stablehlo.reshape %v1563 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1744 = stablehlo.slice %v1743 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1745 = stablehlo.reshape %v1744 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1746 = stablehlo.reshape %v1568 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1747 = stablehlo.slice %v1746 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1748 = stablehlo.reshape %v1747 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1749 = stablehlo.reshape %v1573 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1750 = stablehlo.slice %v1749 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1751 = stablehlo.reshape %v1750 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1752 = stablehlo.reshape %v1748 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1753 = stablehlo.transpose %v1752, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1754 = stablehlo.reshape %v1753 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1755 = stablehlo.reshape %v1745 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1756 = stablehlo.reshape %v1754 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1757 = stablehlo.dot_general %v1755, %v1756, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1758 = stablehlo.reshape %v1757 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1759 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1760 = stablehlo.multiply %v1758, %v1759 : tensor<32x38809xf32>
    %v1761 = stablehlo.reshape %v1760 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1762 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1763 = stablehlo.exponential %v1761 : tensor<32x197x197xf32>
    %v1764 = stablehlo.reduce(%v1763 init: %v1762) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1765 = stablehlo.broadcast_in_dim %v1764, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1766 = stablehlo.divide %v1763, %v1765 : tensor<32x197x197xf32>
    %v1767 = stablehlo.reshape %v1766 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1768 = stablehlo.reshape %v1767 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1769 = stablehlo.reshape %v1751 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1770 = stablehlo.dot_general %v1768, %v1769, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1771 = stablehlo.reshape %v1770 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1772 = stablehlo.reshape %v1771 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1773 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1774 = stablehlo.pad %v1772, %v1773, low = [0, 0, 320], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v1775 = stablehlo.reshape %v1774 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1776 = stablehlo.add %v1742, %v1775 : tensor<32x75648xf32>
    %v1777 = stablehlo.reshape %v1776 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1778 = stablehlo.dot_general %v1777, %b5_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v1779 = stablehlo.broadcast_in_dim %b5_bo, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1780 = stablehlo.add %v1778, %v1779 : tensor<32x197x384xf32>
    %v1781 = stablehlo.reshape %v1780 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1782 = stablehlo.add %v1530, %v1781 : tensor<32x75648xf32>
    %v1783 = stablehlo.reshape %v1782 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1784 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1785 = stablehlo.constant dense<384.0> : tensor<32x197x384xf32>
    %v1786 = stablehlo.constant dense<1.0e-5> : tensor<32x197x384xf32>
    %v1787 = stablehlo.reduce(%v1783 init: %v1784) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1788 = stablehlo.broadcast_in_dim %v1787, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v1789 = stablehlo.divide %v1788, %v1785 : tensor<32x197x384xf32>
    %v1790 = stablehlo.subtract %v1783, %v1789 : tensor<32x197x384xf32>
    %v1791 = stablehlo.multiply %v1790, %v1790 : tensor<32x197x384xf32>
    %v1792 = stablehlo.reduce(%v1791 init: %v1784) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1793 = stablehlo.broadcast_in_dim %v1792, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v1794 = stablehlo.divide %v1793, %v1785 : tensor<32x197x384xf32>
    %v1795 = stablehlo.add %v1794, %v1786 : tensor<32x197x384xf32>
    %v1796 = stablehlo.rsqrt %v1795 : tensor<32x197x384xf32>
    %v1797 = stablehlo.multiply %v1790, %v1796 : tensor<32x197x384xf32>
    %v1798 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v1799 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v1800 = stablehlo.multiply %v1797, %v1798 : tensor<32x197x384xf32>
    %v1801 = stablehlo.add %v1800, %v1799 : tensor<32x197x384xf32>
    %v1802 = stablehlo.reshape %v1801 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1803 = stablehlo.reshape %v1802 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1804 = stablehlo.broadcast_in_dim %b5_g2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1805 = stablehlo.multiply %v1803, %v1804 : tensor<32x197x384xf32>
    %v1806 = stablehlo.reshape %v1805 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1807 = stablehlo.reshape %v1806 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1808 = stablehlo.broadcast_in_dim %b5_bt2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1809 = stablehlo.add %v1807, %v1808 : tensor<32x197x384xf32>
    %v1810 = stablehlo.reshape %v1809 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1811 = stablehlo.reshape %v1810 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1812 = stablehlo.dot_general %v1811, %b5_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x1536xf32>) -> tensor<32x197x1536xf32>
    %v1813 = stablehlo.broadcast_in_dim %b5_bfc1, dims = [2] : (tensor<1536xf32>) -> tensor<32x197x1536xf32>
    %v1814 = stablehlo.add %v1812, %v1813 : tensor<32x197x1536xf32>
    %v1815 = stablehlo.reshape %v1814 : (tensor<32x197x1536xf32>) -> tensor<32x302592xf32>
    %v1816 = stablehlo.multiply %v1815, %v1815 : tensor<32x302592xf32>
    %v1817 = stablehlo.multiply %v1816, %v1815 : tensor<32x302592xf32>
    %v1818 = stablehlo.constant dense<0.044715> : tensor<32x302592xf32>
    %v1819 = stablehlo.multiply %v1818, %v1817 : tensor<32x302592xf32>
    %v1820 = stablehlo.add %v1815, %v1819 : tensor<32x302592xf32>
    %v1821 = stablehlo.constant dense<0.7978845608028654> : tensor<32x302592xf32>
    %v1822 = stablehlo.multiply %v1821, %v1820 : tensor<32x302592xf32>
    %v1823 = stablehlo.tanh %v1822 : tensor<32x302592xf32>
    %v1824 = stablehlo.constant dense<1.0> : tensor<32x302592xf32>
    %v1825 = stablehlo.add %v1824, %v1823 : tensor<32x302592xf32>
    %v1826 = stablehlo.constant dense<0.5> : tensor<32x302592xf32>
    %v1827 = stablehlo.multiply %v1826, %v1815 : tensor<32x302592xf32>
    %v1828 = stablehlo.multiply %v1827, %v1825 : tensor<32x302592xf32>
    %v1829 = stablehlo.reshape %v1828 : (tensor<32x302592xf32>) -> tensor<32x197x1536xf32>
    %v1830 = stablehlo.dot_general %v1829, %b5_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x1536xf32>, tensor<1536x384xf32>) -> tensor<32x197x384xf32>
    %v1831 = stablehlo.broadcast_in_dim %b5_bfc2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1832 = stablehlo.add %v1830, %v1831 : tensor<32x197x384xf32>
    %v1833 = stablehlo.reshape %v1832 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1834 = stablehlo.add %v1782, %v1833 : tensor<32x75648xf32>
    %v1835 = stablehlo.reshape %v1834 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1836 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1837 = stablehlo.constant dense<384.0> : tensor<32x197x384xf32>
    %v1838 = stablehlo.constant dense<1.0e-5> : tensor<32x197x384xf32>
    %v1839 = stablehlo.reduce(%v1835 init: %v1836) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1840 = stablehlo.broadcast_in_dim %v1839, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v1841 = stablehlo.divide %v1840, %v1837 : tensor<32x197x384xf32>
    %v1842 = stablehlo.subtract %v1835, %v1841 : tensor<32x197x384xf32>
    %v1843 = stablehlo.multiply %v1842, %v1842 : tensor<32x197x384xf32>
    %v1844 = stablehlo.reduce(%v1843 init: %v1836) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1845 = stablehlo.broadcast_in_dim %v1844, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v1846 = stablehlo.divide %v1845, %v1837 : tensor<32x197x384xf32>
    %v1847 = stablehlo.add %v1846, %v1838 : tensor<32x197x384xf32>
    %v1848 = stablehlo.rsqrt %v1847 : tensor<32x197x384xf32>
    %v1849 = stablehlo.multiply %v1842, %v1848 : tensor<32x197x384xf32>
    %v1850 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v1851 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v1852 = stablehlo.multiply %v1849, %v1850 : tensor<32x197x384xf32>
    %v1853 = stablehlo.add %v1852, %v1851 : tensor<32x197x384xf32>
    %v1854 = stablehlo.reshape %v1853 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1855 = stablehlo.reshape %v1854 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1856 = stablehlo.broadcast_in_dim %b6_g1, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1857 = stablehlo.multiply %v1855, %v1856 : tensor<32x197x384xf32>
    %v1858 = stablehlo.reshape %v1857 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1859 = stablehlo.reshape %v1858 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1860 = stablehlo.broadcast_in_dim %b6_bt1, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1861 = stablehlo.add %v1859, %v1860 : tensor<32x197x384xf32>
    %v1862 = stablehlo.reshape %v1861 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1863 = stablehlo.reshape %v1862 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1864 = stablehlo.dot_general %v1863, %b6_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v1865 = stablehlo.broadcast_in_dim %b6_bq, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1866 = stablehlo.add %v1864, %v1865 : tensor<32x197x384xf32>
    %v1867 = stablehlo.reshape %v1866 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1868 = stablehlo.reshape %v1862 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1869 = stablehlo.dot_general %v1868, %b6_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v1870 = stablehlo.broadcast_in_dim %b6_bk, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1871 = stablehlo.add %v1869, %v1870 : tensor<32x197x384xf32>
    %v1872 = stablehlo.reshape %v1871 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1873 = stablehlo.reshape %v1862 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1874 = stablehlo.dot_general %v1873, %b6_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v1875 = stablehlo.broadcast_in_dim %b6_bv, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1876 = stablehlo.add %v1874, %v1875 : tensor<32x197x384xf32>
    %v1877 = stablehlo.reshape %v1876 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1878 = stablehlo.reshape %v1867 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1879 = stablehlo.slice %v1878 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1880 = stablehlo.reshape %v1879 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1881 = stablehlo.reshape %v1872 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1882 = stablehlo.slice %v1881 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1883 = stablehlo.reshape %v1882 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1884 = stablehlo.reshape %v1877 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1885 = stablehlo.slice %v1884 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1886 = stablehlo.reshape %v1885 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1887 = stablehlo.reshape %v1883 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1888 = stablehlo.transpose %v1887, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1889 = stablehlo.reshape %v1888 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1890 = stablehlo.reshape %v1880 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1891 = stablehlo.reshape %v1889 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1892 = stablehlo.dot_general %v1890, %v1891, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1893 = stablehlo.reshape %v1892 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1894 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1895 = stablehlo.multiply %v1893, %v1894 : tensor<32x38809xf32>
    %v1896 = stablehlo.reshape %v1895 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1897 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1898 = stablehlo.exponential %v1896 : tensor<32x197x197xf32>
    %v1899 = stablehlo.reduce(%v1898 init: %v1897) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1900 = stablehlo.broadcast_in_dim %v1899, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1901 = stablehlo.divide %v1898, %v1900 : tensor<32x197x197xf32>
    %v1902 = stablehlo.reshape %v1901 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1903 = stablehlo.reshape %v1902 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1904 = stablehlo.reshape %v1886 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1905 = stablehlo.dot_general %v1903, %v1904, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1906 = stablehlo.reshape %v1905 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1907 = stablehlo.reshape %v1906 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1908 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1909 = stablehlo.pad %v1907, %v1908, low = [0, 0, 0], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v1910 = stablehlo.reshape %v1909 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1911 = stablehlo.reshape %v1867 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1912 = stablehlo.slice %v1911 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1913 = stablehlo.reshape %v1912 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1914 = stablehlo.reshape %v1872 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1915 = stablehlo.slice %v1914 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1916 = stablehlo.reshape %v1915 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1917 = stablehlo.reshape %v1877 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1918 = stablehlo.slice %v1917 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1919 = stablehlo.reshape %v1918 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1920 = stablehlo.reshape %v1916 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1921 = stablehlo.transpose %v1920, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1922 = stablehlo.reshape %v1921 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1923 = stablehlo.reshape %v1913 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1924 = stablehlo.reshape %v1922 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1925 = stablehlo.dot_general %v1923, %v1924, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1926 = stablehlo.reshape %v1925 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1927 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1928 = stablehlo.multiply %v1926, %v1927 : tensor<32x38809xf32>
    %v1929 = stablehlo.reshape %v1928 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1930 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1931 = stablehlo.exponential %v1929 : tensor<32x197x197xf32>
    %v1932 = stablehlo.reduce(%v1931 init: %v1930) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1933 = stablehlo.broadcast_in_dim %v1932, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1934 = stablehlo.divide %v1931, %v1933 : tensor<32x197x197xf32>
    %v1935 = stablehlo.reshape %v1934 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1936 = stablehlo.reshape %v1935 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1937 = stablehlo.reshape %v1919 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1938 = stablehlo.dot_general %v1936, %v1937, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1939 = stablehlo.reshape %v1938 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1940 = stablehlo.reshape %v1939 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1941 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1942 = stablehlo.pad %v1940, %v1941, low = [0, 0, 64], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v1943 = stablehlo.reshape %v1942 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1944 = stablehlo.add %v1910, %v1943 : tensor<32x75648xf32>
    %v1945 = stablehlo.reshape %v1867 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1946 = stablehlo.slice %v1945 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1947 = stablehlo.reshape %v1946 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1948 = stablehlo.reshape %v1872 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1949 = stablehlo.slice %v1948 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1950 = stablehlo.reshape %v1949 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1951 = stablehlo.reshape %v1877 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1952 = stablehlo.slice %v1951 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1953 = stablehlo.reshape %v1952 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1954 = stablehlo.reshape %v1950 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1955 = stablehlo.transpose %v1954, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1956 = stablehlo.reshape %v1955 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1957 = stablehlo.reshape %v1947 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1958 = stablehlo.reshape %v1956 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1959 = stablehlo.dot_general %v1957, %v1958, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1960 = stablehlo.reshape %v1959 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1961 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1962 = stablehlo.multiply %v1960, %v1961 : tensor<32x38809xf32>
    %v1963 = stablehlo.reshape %v1962 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1964 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1965 = stablehlo.exponential %v1963 : tensor<32x197x197xf32>
    %v1966 = stablehlo.reduce(%v1965 init: %v1964) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1967 = stablehlo.broadcast_in_dim %v1966, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1968 = stablehlo.divide %v1965, %v1967 : tensor<32x197x197xf32>
    %v1969 = stablehlo.reshape %v1968 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1970 = stablehlo.reshape %v1969 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1971 = stablehlo.reshape %v1953 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1972 = stablehlo.dot_general %v1970, %v1971, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1973 = stablehlo.reshape %v1972 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1974 = stablehlo.reshape %v1973 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1975 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1976 = stablehlo.pad %v1974, %v1975, low = [0, 0, 128], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v1977 = stablehlo.reshape %v1976 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1978 = stablehlo.add %v1944, %v1977 : tensor<32x75648xf32>
    %v1979 = stablehlo.reshape %v1867 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1980 = stablehlo.slice %v1979 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1981 = stablehlo.reshape %v1980 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1982 = stablehlo.reshape %v1872 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1983 = stablehlo.slice %v1982 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1984 = stablehlo.reshape %v1983 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1985 = stablehlo.reshape %v1877 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1986 = stablehlo.slice %v1985 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1987 = stablehlo.reshape %v1986 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1988 = stablehlo.reshape %v1984 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1989 = stablehlo.transpose %v1988, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1990 = stablehlo.reshape %v1989 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1991 = stablehlo.reshape %v1981 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1992 = stablehlo.reshape %v1990 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1993 = stablehlo.dot_general %v1991, %v1992, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1994 = stablehlo.reshape %v1993 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1995 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1996 = stablehlo.multiply %v1994, %v1995 : tensor<32x38809xf32>
    %v1997 = stablehlo.reshape %v1996 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1998 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1999 = stablehlo.exponential %v1997 : tensor<32x197x197xf32>
    %v2000 = stablehlo.reduce(%v1999 init: %v1998) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2001 = stablehlo.broadcast_in_dim %v2000, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2002 = stablehlo.divide %v1999, %v2001 : tensor<32x197x197xf32>
    %v2003 = stablehlo.reshape %v2002 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2004 = stablehlo.reshape %v2003 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2005 = stablehlo.reshape %v1987 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2006 = stablehlo.dot_general %v2004, %v2005, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2007 = stablehlo.reshape %v2006 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2008 = stablehlo.reshape %v2007 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2009 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2010 = stablehlo.pad %v2008, %v2009, low = [0, 0, 192], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v2011 = stablehlo.reshape %v2010 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2012 = stablehlo.add %v1978, %v2011 : tensor<32x75648xf32>
    %v2013 = stablehlo.reshape %v1867 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2014 = stablehlo.slice %v2013 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2015 = stablehlo.reshape %v2014 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2016 = stablehlo.reshape %v1872 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2017 = stablehlo.slice %v2016 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2018 = stablehlo.reshape %v2017 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2019 = stablehlo.reshape %v1877 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2020 = stablehlo.slice %v2019 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2021 = stablehlo.reshape %v2020 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2022 = stablehlo.reshape %v2018 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2023 = stablehlo.transpose %v2022, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2024 = stablehlo.reshape %v2023 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2025 = stablehlo.reshape %v2015 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2026 = stablehlo.reshape %v2024 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2027 = stablehlo.dot_general %v2025, %v2026, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2028 = stablehlo.reshape %v2027 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2029 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2030 = stablehlo.multiply %v2028, %v2029 : tensor<32x38809xf32>
    %v2031 = stablehlo.reshape %v2030 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2032 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2033 = stablehlo.exponential %v2031 : tensor<32x197x197xf32>
    %v2034 = stablehlo.reduce(%v2033 init: %v2032) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2035 = stablehlo.broadcast_in_dim %v2034, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2036 = stablehlo.divide %v2033, %v2035 : tensor<32x197x197xf32>
    %v2037 = stablehlo.reshape %v2036 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2038 = stablehlo.reshape %v2037 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2039 = stablehlo.reshape %v2021 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2040 = stablehlo.dot_general %v2038, %v2039, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2041 = stablehlo.reshape %v2040 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2042 = stablehlo.reshape %v2041 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2043 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2044 = stablehlo.pad %v2042, %v2043, low = [0, 0, 256], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v2045 = stablehlo.reshape %v2044 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2046 = stablehlo.add %v2012, %v2045 : tensor<32x75648xf32>
    %v2047 = stablehlo.reshape %v1867 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2048 = stablehlo.slice %v2047 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2049 = stablehlo.reshape %v2048 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2050 = stablehlo.reshape %v1872 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2051 = stablehlo.slice %v2050 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2052 = stablehlo.reshape %v2051 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2053 = stablehlo.reshape %v1877 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2054 = stablehlo.slice %v2053 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2055 = stablehlo.reshape %v2054 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2056 = stablehlo.reshape %v2052 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2057 = stablehlo.transpose %v2056, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2058 = stablehlo.reshape %v2057 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2059 = stablehlo.reshape %v2049 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2060 = stablehlo.reshape %v2058 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2061 = stablehlo.dot_general %v2059, %v2060, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2062 = stablehlo.reshape %v2061 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2063 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2064 = stablehlo.multiply %v2062, %v2063 : tensor<32x38809xf32>
    %v2065 = stablehlo.reshape %v2064 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2066 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2067 = stablehlo.exponential %v2065 : tensor<32x197x197xf32>
    %v2068 = stablehlo.reduce(%v2067 init: %v2066) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2069 = stablehlo.broadcast_in_dim %v2068, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2070 = stablehlo.divide %v2067, %v2069 : tensor<32x197x197xf32>
    %v2071 = stablehlo.reshape %v2070 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2072 = stablehlo.reshape %v2071 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2073 = stablehlo.reshape %v2055 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2074 = stablehlo.dot_general %v2072, %v2073, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2075 = stablehlo.reshape %v2074 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2076 = stablehlo.reshape %v2075 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2077 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2078 = stablehlo.pad %v2076, %v2077, low = [0, 0, 320], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v2079 = stablehlo.reshape %v2078 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2080 = stablehlo.add %v2046, %v2079 : tensor<32x75648xf32>
    %v2081 = stablehlo.reshape %v2080 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2082 = stablehlo.dot_general %v2081, %b6_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v2083 = stablehlo.broadcast_in_dim %b6_bo, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2084 = stablehlo.add %v2082, %v2083 : tensor<32x197x384xf32>
    %v2085 = stablehlo.reshape %v2084 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2086 = stablehlo.add %v1834, %v2085 : tensor<32x75648xf32>
    %v2087 = stablehlo.reshape %v2086 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2088 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2089 = stablehlo.constant dense<384.0> : tensor<32x197x384xf32>
    %v2090 = stablehlo.constant dense<1.0e-5> : tensor<32x197x384xf32>
    %v2091 = stablehlo.reduce(%v2087 init: %v2088) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2092 = stablehlo.broadcast_in_dim %v2091, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v2093 = stablehlo.divide %v2092, %v2089 : tensor<32x197x384xf32>
    %v2094 = stablehlo.subtract %v2087, %v2093 : tensor<32x197x384xf32>
    %v2095 = stablehlo.multiply %v2094, %v2094 : tensor<32x197x384xf32>
    %v2096 = stablehlo.reduce(%v2095 init: %v2088) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2097 = stablehlo.broadcast_in_dim %v2096, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v2098 = stablehlo.divide %v2097, %v2089 : tensor<32x197x384xf32>
    %v2099 = stablehlo.add %v2098, %v2090 : tensor<32x197x384xf32>
    %v2100 = stablehlo.rsqrt %v2099 : tensor<32x197x384xf32>
    %v2101 = stablehlo.multiply %v2094, %v2100 : tensor<32x197x384xf32>
    %v2102 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v2103 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v2104 = stablehlo.multiply %v2101, %v2102 : tensor<32x197x384xf32>
    %v2105 = stablehlo.add %v2104, %v2103 : tensor<32x197x384xf32>
    %v2106 = stablehlo.reshape %v2105 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2107 = stablehlo.reshape %v2106 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2108 = stablehlo.broadcast_in_dim %b6_g2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2109 = stablehlo.multiply %v2107, %v2108 : tensor<32x197x384xf32>
    %v2110 = stablehlo.reshape %v2109 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2111 = stablehlo.reshape %v2110 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2112 = stablehlo.broadcast_in_dim %b6_bt2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2113 = stablehlo.add %v2111, %v2112 : tensor<32x197x384xf32>
    %v2114 = stablehlo.reshape %v2113 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2115 = stablehlo.reshape %v2114 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2116 = stablehlo.dot_general %v2115, %b6_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x1536xf32>) -> tensor<32x197x1536xf32>
    %v2117 = stablehlo.broadcast_in_dim %b6_bfc1, dims = [2] : (tensor<1536xf32>) -> tensor<32x197x1536xf32>
    %v2118 = stablehlo.add %v2116, %v2117 : tensor<32x197x1536xf32>
    %v2119 = stablehlo.reshape %v2118 : (tensor<32x197x1536xf32>) -> tensor<32x302592xf32>
    %v2120 = stablehlo.multiply %v2119, %v2119 : tensor<32x302592xf32>
    %v2121 = stablehlo.multiply %v2120, %v2119 : tensor<32x302592xf32>
    %v2122 = stablehlo.constant dense<0.044715> : tensor<32x302592xf32>
    %v2123 = stablehlo.multiply %v2122, %v2121 : tensor<32x302592xf32>
    %v2124 = stablehlo.add %v2119, %v2123 : tensor<32x302592xf32>
    %v2125 = stablehlo.constant dense<0.7978845608028654> : tensor<32x302592xf32>
    %v2126 = stablehlo.multiply %v2125, %v2124 : tensor<32x302592xf32>
    %v2127 = stablehlo.tanh %v2126 : tensor<32x302592xf32>
    %v2128 = stablehlo.constant dense<1.0> : tensor<32x302592xf32>
    %v2129 = stablehlo.add %v2128, %v2127 : tensor<32x302592xf32>
    %v2130 = stablehlo.constant dense<0.5> : tensor<32x302592xf32>
    %v2131 = stablehlo.multiply %v2130, %v2119 : tensor<32x302592xf32>
    %v2132 = stablehlo.multiply %v2131, %v2129 : tensor<32x302592xf32>
    %v2133 = stablehlo.reshape %v2132 : (tensor<32x302592xf32>) -> tensor<32x197x1536xf32>
    %v2134 = stablehlo.dot_general %v2133, %b6_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x1536xf32>, tensor<1536x384xf32>) -> tensor<32x197x384xf32>
    %v2135 = stablehlo.broadcast_in_dim %b6_bfc2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2136 = stablehlo.add %v2134, %v2135 : tensor<32x197x384xf32>
    %v2137 = stablehlo.reshape %v2136 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2138 = stablehlo.add %v2086, %v2137 : tensor<32x75648xf32>
    %v2139 = stablehlo.reshape %v2138 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2140 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2141 = stablehlo.constant dense<384.0> : tensor<32x197x384xf32>
    %v2142 = stablehlo.constant dense<1.0e-5> : tensor<32x197x384xf32>
    %v2143 = stablehlo.reduce(%v2139 init: %v2140) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2144 = stablehlo.broadcast_in_dim %v2143, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v2145 = stablehlo.divide %v2144, %v2141 : tensor<32x197x384xf32>
    %v2146 = stablehlo.subtract %v2139, %v2145 : tensor<32x197x384xf32>
    %v2147 = stablehlo.multiply %v2146, %v2146 : tensor<32x197x384xf32>
    %v2148 = stablehlo.reduce(%v2147 init: %v2140) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2149 = stablehlo.broadcast_in_dim %v2148, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v2150 = stablehlo.divide %v2149, %v2141 : tensor<32x197x384xf32>
    %v2151 = stablehlo.add %v2150, %v2142 : tensor<32x197x384xf32>
    %v2152 = stablehlo.rsqrt %v2151 : tensor<32x197x384xf32>
    %v2153 = stablehlo.multiply %v2146, %v2152 : tensor<32x197x384xf32>
    %v2154 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v2155 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v2156 = stablehlo.multiply %v2153, %v2154 : tensor<32x197x384xf32>
    %v2157 = stablehlo.add %v2156, %v2155 : tensor<32x197x384xf32>
    %v2158 = stablehlo.reshape %v2157 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2159 = stablehlo.reshape %v2158 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2160 = stablehlo.broadcast_in_dim %b7_g1, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2161 = stablehlo.multiply %v2159, %v2160 : tensor<32x197x384xf32>
    %v2162 = stablehlo.reshape %v2161 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2163 = stablehlo.reshape %v2162 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2164 = stablehlo.broadcast_in_dim %b7_bt1, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2165 = stablehlo.add %v2163, %v2164 : tensor<32x197x384xf32>
    %v2166 = stablehlo.reshape %v2165 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2167 = stablehlo.reshape %v2166 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2168 = stablehlo.dot_general %v2167, %b7_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v2169 = stablehlo.broadcast_in_dim %b7_bq, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2170 = stablehlo.add %v2168, %v2169 : tensor<32x197x384xf32>
    %v2171 = stablehlo.reshape %v2170 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2172 = stablehlo.reshape %v2166 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2173 = stablehlo.dot_general %v2172, %b7_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v2174 = stablehlo.broadcast_in_dim %b7_bk, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2175 = stablehlo.add %v2173, %v2174 : tensor<32x197x384xf32>
    %v2176 = stablehlo.reshape %v2175 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2177 = stablehlo.reshape %v2166 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2178 = stablehlo.dot_general %v2177, %b7_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v2179 = stablehlo.broadcast_in_dim %b7_bv, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2180 = stablehlo.add %v2178, %v2179 : tensor<32x197x384xf32>
    %v2181 = stablehlo.reshape %v2180 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2182 = stablehlo.reshape %v2171 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2183 = stablehlo.slice %v2182 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2184 = stablehlo.reshape %v2183 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2185 = stablehlo.reshape %v2176 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2186 = stablehlo.slice %v2185 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2187 = stablehlo.reshape %v2186 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2188 = stablehlo.reshape %v2181 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2189 = stablehlo.slice %v2188 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2190 = stablehlo.reshape %v2189 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2191 = stablehlo.reshape %v2187 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2192 = stablehlo.transpose %v2191, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2193 = stablehlo.reshape %v2192 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2194 = stablehlo.reshape %v2184 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2195 = stablehlo.reshape %v2193 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2196 = stablehlo.dot_general %v2194, %v2195, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2197 = stablehlo.reshape %v2196 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2198 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2199 = stablehlo.multiply %v2197, %v2198 : tensor<32x38809xf32>
    %v2200 = stablehlo.reshape %v2199 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2201 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2202 = stablehlo.exponential %v2200 : tensor<32x197x197xf32>
    %v2203 = stablehlo.reduce(%v2202 init: %v2201) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2204 = stablehlo.broadcast_in_dim %v2203, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2205 = stablehlo.divide %v2202, %v2204 : tensor<32x197x197xf32>
    %v2206 = stablehlo.reshape %v2205 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2207 = stablehlo.reshape %v2206 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2208 = stablehlo.reshape %v2190 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2209 = stablehlo.dot_general %v2207, %v2208, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2210 = stablehlo.reshape %v2209 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2211 = stablehlo.reshape %v2210 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2212 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2213 = stablehlo.pad %v2211, %v2212, low = [0, 0, 0], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v2214 = stablehlo.reshape %v2213 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2215 = stablehlo.reshape %v2171 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2216 = stablehlo.slice %v2215 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2217 = stablehlo.reshape %v2216 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2218 = stablehlo.reshape %v2176 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2219 = stablehlo.slice %v2218 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2220 = stablehlo.reshape %v2219 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2221 = stablehlo.reshape %v2181 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2222 = stablehlo.slice %v2221 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2223 = stablehlo.reshape %v2222 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2224 = stablehlo.reshape %v2220 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2225 = stablehlo.transpose %v2224, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2226 = stablehlo.reshape %v2225 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2227 = stablehlo.reshape %v2217 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2228 = stablehlo.reshape %v2226 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2229 = stablehlo.dot_general %v2227, %v2228, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2230 = stablehlo.reshape %v2229 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2231 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2232 = stablehlo.multiply %v2230, %v2231 : tensor<32x38809xf32>
    %v2233 = stablehlo.reshape %v2232 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2234 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2235 = stablehlo.exponential %v2233 : tensor<32x197x197xf32>
    %v2236 = stablehlo.reduce(%v2235 init: %v2234) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2237 = stablehlo.broadcast_in_dim %v2236, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2238 = stablehlo.divide %v2235, %v2237 : tensor<32x197x197xf32>
    %v2239 = stablehlo.reshape %v2238 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2240 = stablehlo.reshape %v2239 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2241 = stablehlo.reshape %v2223 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2242 = stablehlo.dot_general %v2240, %v2241, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2243 = stablehlo.reshape %v2242 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2244 = stablehlo.reshape %v2243 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2245 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2246 = stablehlo.pad %v2244, %v2245, low = [0, 0, 64], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v2247 = stablehlo.reshape %v2246 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2248 = stablehlo.add %v2214, %v2247 : tensor<32x75648xf32>
    %v2249 = stablehlo.reshape %v2171 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2250 = stablehlo.slice %v2249 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2251 = stablehlo.reshape %v2250 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2252 = stablehlo.reshape %v2176 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2253 = stablehlo.slice %v2252 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2254 = stablehlo.reshape %v2253 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2255 = stablehlo.reshape %v2181 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2256 = stablehlo.slice %v2255 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2257 = stablehlo.reshape %v2256 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2258 = stablehlo.reshape %v2254 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2259 = stablehlo.transpose %v2258, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2260 = stablehlo.reshape %v2259 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2261 = stablehlo.reshape %v2251 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2262 = stablehlo.reshape %v2260 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2263 = stablehlo.dot_general %v2261, %v2262, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2264 = stablehlo.reshape %v2263 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2265 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2266 = stablehlo.multiply %v2264, %v2265 : tensor<32x38809xf32>
    %v2267 = stablehlo.reshape %v2266 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2268 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2269 = stablehlo.exponential %v2267 : tensor<32x197x197xf32>
    %v2270 = stablehlo.reduce(%v2269 init: %v2268) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2271 = stablehlo.broadcast_in_dim %v2270, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2272 = stablehlo.divide %v2269, %v2271 : tensor<32x197x197xf32>
    %v2273 = stablehlo.reshape %v2272 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2274 = stablehlo.reshape %v2273 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2275 = stablehlo.reshape %v2257 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2276 = stablehlo.dot_general %v2274, %v2275, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2277 = stablehlo.reshape %v2276 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2278 = stablehlo.reshape %v2277 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2279 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2280 = stablehlo.pad %v2278, %v2279, low = [0, 0, 128], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v2281 = stablehlo.reshape %v2280 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2282 = stablehlo.add %v2248, %v2281 : tensor<32x75648xf32>
    %v2283 = stablehlo.reshape %v2171 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2284 = stablehlo.slice %v2283 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2285 = stablehlo.reshape %v2284 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2286 = stablehlo.reshape %v2176 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2287 = stablehlo.slice %v2286 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2288 = stablehlo.reshape %v2287 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2289 = stablehlo.reshape %v2181 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2290 = stablehlo.slice %v2289 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2291 = stablehlo.reshape %v2290 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2292 = stablehlo.reshape %v2288 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2293 = stablehlo.transpose %v2292, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2294 = stablehlo.reshape %v2293 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2295 = stablehlo.reshape %v2285 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2296 = stablehlo.reshape %v2294 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2297 = stablehlo.dot_general %v2295, %v2296, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2298 = stablehlo.reshape %v2297 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2299 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2300 = stablehlo.multiply %v2298, %v2299 : tensor<32x38809xf32>
    %v2301 = stablehlo.reshape %v2300 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2302 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2303 = stablehlo.exponential %v2301 : tensor<32x197x197xf32>
    %v2304 = stablehlo.reduce(%v2303 init: %v2302) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2305 = stablehlo.broadcast_in_dim %v2304, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2306 = stablehlo.divide %v2303, %v2305 : tensor<32x197x197xf32>
    %v2307 = stablehlo.reshape %v2306 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2308 = stablehlo.reshape %v2307 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2309 = stablehlo.reshape %v2291 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2310 = stablehlo.dot_general %v2308, %v2309, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2311 = stablehlo.reshape %v2310 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2312 = stablehlo.reshape %v2311 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2313 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2314 = stablehlo.pad %v2312, %v2313, low = [0, 0, 192], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v2315 = stablehlo.reshape %v2314 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2316 = stablehlo.add %v2282, %v2315 : tensor<32x75648xf32>
    %v2317 = stablehlo.reshape %v2171 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2318 = stablehlo.slice %v2317 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2319 = stablehlo.reshape %v2318 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2320 = stablehlo.reshape %v2176 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2321 = stablehlo.slice %v2320 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2322 = stablehlo.reshape %v2321 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2323 = stablehlo.reshape %v2181 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2324 = stablehlo.slice %v2323 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2325 = stablehlo.reshape %v2324 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2326 = stablehlo.reshape %v2322 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2327 = stablehlo.transpose %v2326, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2328 = stablehlo.reshape %v2327 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2329 = stablehlo.reshape %v2319 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2330 = stablehlo.reshape %v2328 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2331 = stablehlo.dot_general %v2329, %v2330, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2332 = stablehlo.reshape %v2331 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2333 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2334 = stablehlo.multiply %v2332, %v2333 : tensor<32x38809xf32>
    %v2335 = stablehlo.reshape %v2334 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2336 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2337 = stablehlo.exponential %v2335 : tensor<32x197x197xf32>
    %v2338 = stablehlo.reduce(%v2337 init: %v2336) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2339 = stablehlo.broadcast_in_dim %v2338, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2340 = stablehlo.divide %v2337, %v2339 : tensor<32x197x197xf32>
    %v2341 = stablehlo.reshape %v2340 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2342 = stablehlo.reshape %v2341 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2343 = stablehlo.reshape %v2325 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2344 = stablehlo.dot_general %v2342, %v2343, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2345 = stablehlo.reshape %v2344 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2346 = stablehlo.reshape %v2345 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2347 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2348 = stablehlo.pad %v2346, %v2347, low = [0, 0, 256], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v2349 = stablehlo.reshape %v2348 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2350 = stablehlo.add %v2316, %v2349 : tensor<32x75648xf32>
    %v2351 = stablehlo.reshape %v2171 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2352 = stablehlo.slice %v2351 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2353 = stablehlo.reshape %v2352 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2354 = stablehlo.reshape %v2176 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2355 = stablehlo.slice %v2354 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2356 = stablehlo.reshape %v2355 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2357 = stablehlo.reshape %v2181 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2358 = stablehlo.slice %v2357 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2359 = stablehlo.reshape %v2358 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2360 = stablehlo.reshape %v2356 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2361 = stablehlo.transpose %v2360, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2362 = stablehlo.reshape %v2361 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2363 = stablehlo.reshape %v2353 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2364 = stablehlo.reshape %v2362 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2365 = stablehlo.dot_general %v2363, %v2364, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2366 = stablehlo.reshape %v2365 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2367 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2368 = stablehlo.multiply %v2366, %v2367 : tensor<32x38809xf32>
    %v2369 = stablehlo.reshape %v2368 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2370 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2371 = stablehlo.exponential %v2369 : tensor<32x197x197xf32>
    %v2372 = stablehlo.reduce(%v2371 init: %v2370) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2373 = stablehlo.broadcast_in_dim %v2372, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2374 = stablehlo.divide %v2371, %v2373 : tensor<32x197x197xf32>
    %v2375 = stablehlo.reshape %v2374 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2376 = stablehlo.reshape %v2375 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2377 = stablehlo.reshape %v2359 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2378 = stablehlo.dot_general %v2376, %v2377, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2379 = stablehlo.reshape %v2378 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2380 = stablehlo.reshape %v2379 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2381 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2382 = stablehlo.pad %v2380, %v2381, low = [0, 0, 320], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v2383 = stablehlo.reshape %v2382 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2384 = stablehlo.add %v2350, %v2383 : tensor<32x75648xf32>
    %v2385 = stablehlo.reshape %v2384 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2386 = stablehlo.dot_general %v2385, %b7_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v2387 = stablehlo.broadcast_in_dim %b7_bo, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2388 = stablehlo.add %v2386, %v2387 : tensor<32x197x384xf32>
    %v2389 = stablehlo.reshape %v2388 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2390 = stablehlo.add %v2138, %v2389 : tensor<32x75648xf32>
    %v2391 = stablehlo.reshape %v2390 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2392 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2393 = stablehlo.constant dense<384.0> : tensor<32x197x384xf32>
    %v2394 = stablehlo.constant dense<1.0e-5> : tensor<32x197x384xf32>
    %v2395 = stablehlo.reduce(%v2391 init: %v2392) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2396 = stablehlo.broadcast_in_dim %v2395, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v2397 = stablehlo.divide %v2396, %v2393 : tensor<32x197x384xf32>
    %v2398 = stablehlo.subtract %v2391, %v2397 : tensor<32x197x384xf32>
    %v2399 = stablehlo.multiply %v2398, %v2398 : tensor<32x197x384xf32>
    %v2400 = stablehlo.reduce(%v2399 init: %v2392) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2401 = stablehlo.broadcast_in_dim %v2400, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v2402 = stablehlo.divide %v2401, %v2393 : tensor<32x197x384xf32>
    %v2403 = stablehlo.add %v2402, %v2394 : tensor<32x197x384xf32>
    %v2404 = stablehlo.rsqrt %v2403 : tensor<32x197x384xf32>
    %v2405 = stablehlo.multiply %v2398, %v2404 : tensor<32x197x384xf32>
    %v2406 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v2407 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v2408 = stablehlo.multiply %v2405, %v2406 : tensor<32x197x384xf32>
    %v2409 = stablehlo.add %v2408, %v2407 : tensor<32x197x384xf32>
    %v2410 = stablehlo.reshape %v2409 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2411 = stablehlo.reshape %v2410 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2412 = stablehlo.broadcast_in_dim %b7_g2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2413 = stablehlo.multiply %v2411, %v2412 : tensor<32x197x384xf32>
    %v2414 = stablehlo.reshape %v2413 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2415 = stablehlo.reshape %v2414 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2416 = stablehlo.broadcast_in_dim %b7_bt2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2417 = stablehlo.add %v2415, %v2416 : tensor<32x197x384xf32>
    %v2418 = stablehlo.reshape %v2417 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2419 = stablehlo.reshape %v2418 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2420 = stablehlo.dot_general %v2419, %b7_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x1536xf32>) -> tensor<32x197x1536xf32>
    %v2421 = stablehlo.broadcast_in_dim %b7_bfc1, dims = [2] : (tensor<1536xf32>) -> tensor<32x197x1536xf32>
    %v2422 = stablehlo.add %v2420, %v2421 : tensor<32x197x1536xf32>
    %v2423 = stablehlo.reshape %v2422 : (tensor<32x197x1536xf32>) -> tensor<32x302592xf32>
    %v2424 = stablehlo.multiply %v2423, %v2423 : tensor<32x302592xf32>
    %v2425 = stablehlo.multiply %v2424, %v2423 : tensor<32x302592xf32>
    %v2426 = stablehlo.constant dense<0.044715> : tensor<32x302592xf32>
    %v2427 = stablehlo.multiply %v2426, %v2425 : tensor<32x302592xf32>
    %v2428 = stablehlo.add %v2423, %v2427 : tensor<32x302592xf32>
    %v2429 = stablehlo.constant dense<0.7978845608028654> : tensor<32x302592xf32>
    %v2430 = stablehlo.multiply %v2429, %v2428 : tensor<32x302592xf32>
    %v2431 = stablehlo.tanh %v2430 : tensor<32x302592xf32>
    %v2432 = stablehlo.constant dense<1.0> : tensor<32x302592xf32>
    %v2433 = stablehlo.add %v2432, %v2431 : tensor<32x302592xf32>
    %v2434 = stablehlo.constant dense<0.5> : tensor<32x302592xf32>
    %v2435 = stablehlo.multiply %v2434, %v2423 : tensor<32x302592xf32>
    %v2436 = stablehlo.multiply %v2435, %v2433 : tensor<32x302592xf32>
    %v2437 = stablehlo.reshape %v2436 : (tensor<32x302592xf32>) -> tensor<32x197x1536xf32>
    %v2438 = stablehlo.dot_general %v2437, %b7_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x1536xf32>, tensor<1536x384xf32>) -> tensor<32x197x384xf32>
    %v2439 = stablehlo.broadcast_in_dim %b7_bfc2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2440 = stablehlo.add %v2438, %v2439 : tensor<32x197x384xf32>
    %v2441 = stablehlo.reshape %v2440 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2442 = stablehlo.add %v2390, %v2441 : tensor<32x75648xf32>
    %v2443 = stablehlo.reshape %v2442 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2444 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2445 = stablehlo.constant dense<384.0> : tensor<32x197x384xf32>
    %v2446 = stablehlo.constant dense<1.0e-5> : tensor<32x197x384xf32>
    %v2447 = stablehlo.reduce(%v2443 init: %v2444) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2448 = stablehlo.broadcast_in_dim %v2447, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v2449 = stablehlo.divide %v2448, %v2445 : tensor<32x197x384xf32>
    %v2450 = stablehlo.subtract %v2443, %v2449 : tensor<32x197x384xf32>
    %v2451 = stablehlo.multiply %v2450, %v2450 : tensor<32x197x384xf32>
    %v2452 = stablehlo.reduce(%v2451 init: %v2444) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2453 = stablehlo.broadcast_in_dim %v2452, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v2454 = stablehlo.divide %v2453, %v2445 : tensor<32x197x384xf32>
    %v2455 = stablehlo.add %v2454, %v2446 : tensor<32x197x384xf32>
    %v2456 = stablehlo.rsqrt %v2455 : tensor<32x197x384xf32>
    %v2457 = stablehlo.multiply %v2450, %v2456 : tensor<32x197x384xf32>
    %v2458 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v2459 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v2460 = stablehlo.multiply %v2457, %v2458 : tensor<32x197x384xf32>
    %v2461 = stablehlo.add %v2460, %v2459 : tensor<32x197x384xf32>
    %v2462 = stablehlo.reshape %v2461 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2463 = stablehlo.reshape %v2462 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2464 = stablehlo.broadcast_in_dim %b8_g1, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2465 = stablehlo.multiply %v2463, %v2464 : tensor<32x197x384xf32>
    %v2466 = stablehlo.reshape %v2465 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2467 = stablehlo.reshape %v2466 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2468 = stablehlo.broadcast_in_dim %b8_bt1, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2469 = stablehlo.add %v2467, %v2468 : tensor<32x197x384xf32>
    %v2470 = stablehlo.reshape %v2469 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2471 = stablehlo.reshape %v2470 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2472 = stablehlo.dot_general %v2471, %b8_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v2473 = stablehlo.broadcast_in_dim %b8_bq, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2474 = stablehlo.add %v2472, %v2473 : tensor<32x197x384xf32>
    %v2475 = stablehlo.reshape %v2474 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2476 = stablehlo.reshape %v2470 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2477 = stablehlo.dot_general %v2476, %b8_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v2478 = stablehlo.broadcast_in_dim %b8_bk, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2479 = stablehlo.add %v2477, %v2478 : tensor<32x197x384xf32>
    %v2480 = stablehlo.reshape %v2479 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2481 = stablehlo.reshape %v2470 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2482 = stablehlo.dot_general %v2481, %b8_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v2483 = stablehlo.broadcast_in_dim %b8_bv, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2484 = stablehlo.add %v2482, %v2483 : tensor<32x197x384xf32>
    %v2485 = stablehlo.reshape %v2484 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2486 = stablehlo.reshape %v2475 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2487 = stablehlo.slice %v2486 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2488 = stablehlo.reshape %v2487 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2489 = stablehlo.reshape %v2480 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2490 = stablehlo.slice %v2489 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2491 = stablehlo.reshape %v2490 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2492 = stablehlo.reshape %v2485 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2493 = stablehlo.slice %v2492 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2494 = stablehlo.reshape %v2493 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2495 = stablehlo.reshape %v2491 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2496 = stablehlo.transpose %v2495, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2497 = stablehlo.reshape %v2496 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2498 = stablehlo.reshape %v2488 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2499 = stablehlo.reshape %v2497 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2500 = stablehlo.dot_general %v2498, %v2499, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2501 = stablehlo.reshape %v2500 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2502 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2503 = stablehlo.multiply %v2501, %v2502 : tensor<32x38809xf32>
    %v2504 = stablehlo.reshape %v2503 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2505 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2506 = stablehlo.exponential %v2504 : tensor<32x197x197xf32>
    %v2507 = stablehlo.reduce(%v2506 init: %v2505) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2508 = stablehlo.broadcast_in_dim %v2507, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2509 = stablehlo.divide %v2506, %v2508 : tensor<32x197x197xf32>
    %v2510 = stablehlo.reshape %v2509 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2511 = stablehlo.reshape %v2510 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2512 = stablehlo.reshape %v2494 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2513 = stablehlo.dot_general %v2511, %v2512, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2514 = stablehlo.reshape %v2513 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2515 = stablehlo.reshape %v2514 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2516 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2517 = stablehlo.pad %v2515, %v2516, low = [0, 0, 0], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v2518 = stablehlo.reshape %v2517 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2519 = stablehlo.reshape %v2475 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2520 = stablehlo.slice %v2519 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2521 = stablehlo.reshape %v2520 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2522 = stablehlo.reshape %v2480 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2523 = stablehlo.slice %v2522 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2524 = stablehlo.reshape %v2523 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2525 = stablehlo.reshape %v2485 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2526 = stablehlo.slice %v2525 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
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
    %v2538 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2539 = stablehlo.exponential %v2537 : tensor<32x197x197xf32>
    %v2540 = stablehlo.reduce(%v2539 init: %v2538) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2541 = stablehlo.broadcast_in_dim %v2540, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2542 = stablehlo.divide %v2539, %v2541 : tensor<32x197x197xf32>
    %v2543 = stablehlo.reshape %v2542 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2544 = stablehlo.reshape %v2543 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2545 = stablehlo.reshape %v2527 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2546 = stablehlo.dot_general %v2544, %v2545, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2547 = stablehlo.reshape %v2546 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2548 = stablehlo.reshape %v2547 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2549 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2550 = stablehlo.pad %v2548, %v2549, low = [0, 0, 64], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v2551 = stablehlo.reshape %v2550 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2552 = stablehlo.add %v2518, %v2551 : tensor<32x75648xf32>
    %v2553 = stablehlo.reshape %v2475 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2554 = stablehlo.slice %v2553 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2555 = stablehlo.reshape %v2554 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2556 = stablehlo.reshape %v2480 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2557 = stablehlo.slice %v2556 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2558 = stablehlo.reshape %v2557 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2559 = stablehlo.reshape %v2485 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2560 = stablehlo.slice %v2559 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2561 = stablehlo.reshape %v2560 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2562 = stablehlo.reshape %v2558 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2563 = stablehlo.transpose %v2562, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2564 = stablehlo.reshape %v2563 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2565 = stablehlo.reshape %v2555 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2566 = stablehlo.reshape %v2564 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2567 = stablehlo.dot_general %v2565, %v2566, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2568 = stablehlo.reshape %v2567 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2569 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2570 = stablehlo.multiply %v2568, %v2569 : tensor<32x38809xf32>
    %v2571 = stablehlo.reshape %v2570 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2572 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2573 = stablehlo.exponential %v2571 : tensor<32x197x197xf32>
    %v2574 = stablehlo.reduce(%v2573 init: %v2572) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2575 = stablehlo.broadcast_in_dim %v2574, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2576 = stablehlo.divide %v2573, %v2575 : tensor<32x197x197xf32>
    %v2577 = stablehlo.reshape %v2576 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2578 = stablehlo.reshape %v2577 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2579 = stablehlo.reshape %v2561 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2580 = stablehlo.dot_general %v2578, %v2579, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2581 = stablehlo.reshape %v2580 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2582 = stablehlo.reshape %v2581 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2583 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2584 = stablehlo.pad %v2582, %v2583, low = [0, 0, 128], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v2585 = stablehlo.reshape %v2584 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2586 = stablehlo.add %v2552, %v2585 : tensor<32x75648xf32>
    %v2587 = stablehlo.reshape %v2475 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2588 = stablehlo.slice %v2587 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2589 = stablehlo.reshape %v2588 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2590 = stablehlo.reshape %v2480 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2591 = stablehlo.slice %v2590 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2592 = stablehlo.reshape %v2591 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2593 = stablehlo.reshape %v2485 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2594 = stablehlo.slice %v2593 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2595 = stablehlo.reshape %v2594 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2596 = stablehlo.reshape %v2592 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2597 = stablehlo.transpose %v2596, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2598 = stablehlo.reshape %v2597 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2599 = stablehlo.reshape %v2589 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2600 = stablehlo.reshape %v2598 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2601 = stablehlo.dot_general %v2599, %v2600, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2602 = stablehlo.reshape %v2601 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2603 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2604 = stablehlo.multiply %v2602, %v2603 : tensor<32x38809xf32>
    %v2605 = stablehlo.reshape %v2604 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2606 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2607 = stablehlo.exponential %v2605 : tensor<32x197x197xf32>
    %v2608 = stablehlo.reduce(%v2607 init: %v2606) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2609 = stablehlo.broadcast_in_dim %v2608, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2610 = stablehlo.divide %v2607, %v2609 : tensor<32x197x197xf32>
    %v2611 = stablehlo.reshape %v2610 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2612 = stablehlo.reshape %v2611 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2613 = stablehlo.reshape %v2595 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2614 = stablehlo.dot_general %v2612, %v2613, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2615 = stablehlo.reshape %v2614 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2616 = stablehlo.reshape %v2615 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2617 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2618 = stablehlo.pad %v2616, %v2617, low = [0, 0, 192], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v2619 = stablehlo.reshape %v2618 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2620 = stablehlo.add %v2586, %v2619 : tensor<32x75648xf32>
    %v2621 = stablehlo.reshape %v2475 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2622 = stablehlo.slice %v2621 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2623 = stablehlo.reshape %v2622 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2624 = stablehlo.reshape %v2480 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2625 = stablehlo.slice %v2624 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2626 = stablehlo.reshape %v2625 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2627 = stablehlo.reshape %v2485 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2628 = stablehlo.slice %v2627 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2629 = stablehlo.reshape %v2628 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2630 = stablehlo.reshape %v2626 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2631 = stablehlo.transpose %v2630, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2632 = stablehlo.reshape %v2631 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2633 = stablehlo.reshape %v2623 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2634 = stablehlo.reshape %v2632 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2635 = stablehlo.dot_general %v2633, %v2634, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2636 = stablehlo.reshape %v2635 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2637 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2638 = stablehlo.multiply %v2636, %v2637 : tensor<32x38809xf32>
    %v2639 = stablehlo.reshape %v2638 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2640 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2641 = stablehlo.exponential %v2639 : tensor<32x197x197xf32>
    %v2642 = stablehlo.reduce(%v2641 init: %v2640) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2643 = stablehlo.broadcast_in_dim %v2642, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2644 = stablehlo.divide %v2641, %v2643 : tensor<32x197x197xf32>
    %v2645 = stablehlo.reshape %v2644 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2646 = stablehlo.reshape %v2645 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2647 = stablehlo.reshape %v2629 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2648 = stablehlo.dot_general %v2646, %v2647, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2649 = stablehlo.reshape %v2648 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2650 = stablehlo.reshape %v2649 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2651 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2652 = stablehlo.pad %v2650, %v2651, low = [0, 0, 256], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v2653 = stablehlo.reshape %v2652 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2654 = stablehlo.add %v2620, %v2653 : tensor<32x75648xf32>
    %v2655 = stablehlo.reshape %v2475 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2656 = stablehlo.slice %v2655 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2657 = stablehlo.reshape %v2656 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2658 = stablehlo.reshape %v2480 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2659 = stablehlo.slice %v2658 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2660 = stablehlo.reshape %v2659 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2661 = stablehlo.reshape %v2485 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2662 = stablehlo.slice %v2661 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2663 = stablehlo.reshape %v2662 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2664 = stablehlo.reshape %v2660 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2665 = stablehlo.transpose %v2664, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2666 = stablehlo.reshape %v2665 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2667 = stablehlo.reshape %v2657 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2668 = stablehlo.reshape %v2666 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2669 = stablehlo.dot_general %v2667, %v2668, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2670 = stablehlo.reshape %v2669 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2671 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2672 = stablehlo.multiply %v2670, %v2671 : tensor<32x38809xf32>
    %v2673 = stablehlo.reshape %v2672 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2674 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2675 = stablehlo.exponential %v2673 : tensor<32x197x197xf32>
    %v2676 = stablehlo.reduce(%v2675 init: %v2674) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2677 = stablehlo.broadcast_in_dim %v2676, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2678 = stablehlo.divide %v2675, %v2677 : tensor<32x197x197xf32>
    %v2679 = stablehlo.reshape %v2678 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2680 = stablehlo.reshape %v2679 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2681 = stablehlo.reshape %v2663 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2682 = stablehlo.dot_general %v2680, %v2681, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2683 = stablehlo.reshape %v2682 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2684 = stablehlo.reshape %v2683 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2685 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2686 = stablehlo.pad %v2684, %v2685, low = [0, 0, 320], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v2687 = stablehlo.reshape %v2686 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2688 = stablehlo.add %v2654, %v2687 : tensor<32x75648xf32>
    %v2689 = stablehlo.reshape %v2688 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2690 = stablehlo.dot_general %v2689, %b8_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v2691 = stablehlo.broadcast_in_dim %b8_bo, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2692 = stablehlo.add %v2690, %v2691 : tensor<32x197x384xf32>
    %v2693 = stablehlo.reshape %v2692 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2694 = stablehlo.add %v2442, %v2693 : tensor<32x75648xf32>
    %v2695 = stablehlo.reshape %v2694 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2696 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2697 = stablehlo.constant dense<384.0> : tensor<32x197x384xf32>
    %v2698 = stablehlo.constant dense<1.0e-5> : tensor<32x197x384xf32>
    %v2699 = stablehlo.reduce(%v2695 init: %v2696) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2700 = stablehlo.broadcast_in_dim %v2699, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v2701 = stablehlo.divide %v2700, %v2697 : tensor<32x197x384xf32>
    %v2702 = stablehlo.subtract %v2695, %v2701 : tensor<32x197x384xf32>
    %v2703 = stablehlo.multiply %v2702, %v2702 : tensor<32x197x384xf32>
    %v2704 = stablehlo.reduce(%v2703 init: %v2696) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2705 = stablehlo.broadcast_in_dim %v2704, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v2706 = stablehlo.divide %v2705, %v2697 : tensor<32x197x384xf32>
    %v2707 = stablehlo.add %v2706, %v2698 : tensor<32x197x384xf32>
    %v2708 = stablehlo.rsqrt %v2707 : tensor<32x197x384xf32>
    %v2709 = stablehlo.multiply %v2702, %v2708 : tensor<32x197x384xf32>
    %v2710 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v2711 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v2712 = stablehlo.multiply %v2709, %v2710 : tensor<32x197x384xf32>
    %v2713 = stablehlo.add %v2712, %v2711 : tensor<32x197x384xf32>
    %v2714 = stablehlo.reshape %v2713 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2715 = stablehlo.reshape %v2714 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2716 = stablehlo.broadcast_in_dim %b8_g2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2717 = stablehlo.multiply %v2715, %v2716 : tensor<32x197x384xf32>
    %v2718 = stablehlo.reshape %v2717 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2719 = stablehlo.reshape %v2718 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2720 = stablehlo.broadcast_in_dim %b8_bt2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2721 = stablehlo.add %v2719, %v2720 : tensor<32x197x384xf32>
    %v2722 = stablehlo.reshape %v2721 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2723 = stablehlo.reshape %v2722 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2724 = stablehlo.dot_general %v2723, %b8_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x1536xf32>) -> tensor<32x197x1536xf32>
    %v2725 = stablehlo.broadcast_in_dim %b8_bfc1, dims = [2] : (tensor<1536xf32>) -> tensor<32x197x1536xf32>
    %v2726 = stablehlo.add %v2724, %v2725 : tensor<32x197x1536xf32>
    %v2727 = stablehlo.reshape %v2726 : (tensor<32x197x1536xf32>) -> tensor<32x302592xf32>
    %v2728 = stablehlo.multiply %v2727, %v2727 : tensor<32x302592xf32>
    %v2729 = stablehlo.multiply %v2728, %v2727 : tensor<32x302592xf32>
    %v2730 = stablehlo.constant dense<0.044715> : tensor<32x302592xf32>
    %v2731 = stablehlo.multiply %v2730, %v2729 : tensor<32x302592xf32>
    %v2732 = stablehlo.add %v2727, %v2731 : tensor<32x302592xf32>
    %v2733 = stablehlo.constant dense<0.7978845608028654> : tensor<32x302592xf32>
    %v2734 = stablehlo.multiply %v2733, %v2732 : tensor<32x302592xf32>
    %v2735 = stablehlo.tanh %v2734 : tensor<32x302592xf32>
    %v2736 = stablehlo.constant dense<1.0> : tensor<32x302592xf32>
    %v2737 = stablehlo.add %v2736, %v2735 : tensor<32x302592xf32>
    %v2738 = stablehlo.constant dense<0.5> : tensor<32x302592xf32>
    %v2739 = stablehlo.multiply %v2738, %v2727 : tensor<32x302592xf32>
    %v2740 = stablehlo.multiply %v2739, %v2737 : tensor<32x302592xf32>
    %v2741 = stablehlo.reshape %v2740 : (tensor<32x302592xf32>) -> tensor<32x197x1536xf32>
    %v2742 = stablehlo.dot_general %v2741, %b8_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x1536xf32>, tensor<1536x384xf32>) -> tensor<32x197x384xf32>
    %v2743 = stablehlo.broadcast_in_dim %b8_bfc2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2744 = stablehlo.add %v2742, %v2743 : tensor<32x197x384xf32>
    %v2745 = stablehlo.reshape %v2744 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2746 = stablehlo.add %v2694, %v2745 : tensor<32x75648xf32>
    %v2747 = stablehlo.reshape %v2746 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2748 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2749 = stablehlo.constant dense<384.0> : tensor<32x197x384xf32>
    %v2750 = stablehlo.constant dense<1.0e-5> : tensor<32x197x384xf32>
    %v2751 = stablehlo.reduce(%v2747 init: %v2748) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2752 = stablehlo.broadcast_in_dim %v2751, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v2753 = stablehlo.divide %v2752, %v2749 : tensor<32x197x384xf32>
    %v2754 = stablehlo.subtract %v2747, %v2753 : tensor<32x197x384xf32>
    %v2755 = stablehlo.multiply %v2754, %v2754 : tensor<32x197x384xf32>
    %v2756 = stablehlo.reduce(%v2755 init: %v2748) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2757 = stablehlo.broadcast_in_dim %v2756, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v2758 = stablehlo.divide %v2757, %v2749 : tensor<32x197x384xf32>
    %v2759 = stablehlo.add %v2758, %v2750 : tensor<32x197x384xf32>
    %v2760 = stablehlo.rsqrt %v2759 : tensor<32x197x384xf32>
    %v2761 = stablehlo.multiply %v2754, %v2760 : tensor<32x197x384xf32>
    %v2762 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v2763 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v2764 = stablehlo.multiply %v2761, %v2762 : tensor<32x197x384xf32>
    %v2765 = stablehlo.add %v2764, %v2763 : tensor<32x197x384xf32>
    %v2766 = stablehlo.reshape %v2765 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2767 = stablehlo.reshape %v2766 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2768 = stablehlo.broadcast_in_dim %b9_g1, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2769 = stablehlo.multiply %v2767, %v2768 : tensor<32x197x384xf32>
    %v2770 = stablehlo.reshape %v2769 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2771 = stablehlo.reshape %v2770 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2772 = stablehlo.broadcast_in_dim %b9_bt1, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2773 = stablehlo.add %v2771, %v2772 : tensor<32x197x384xf32>
    %v2774 = stablehlo.reshape %v2773 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2775 = stablehlo.reshape %v2774 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2776 = stablehlo.dot_general %v2775, %b9_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v2777 = stablehlo.broadcast_in_dim %b9_bq, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2778 = stablehlo.add %v2776, %v2777 : tensor<32x197x384xf32>
    %v2779 = stablehlo.reshape %v2778 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2780 = stablehlo.reshape %v2774 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2781 = stablehlo.dot_general %v2780, %b9_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v2782 = stablehlo.broadcast_in_dim %b9_bk, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2783 = stablehlo.add %v2781, %v2782 : tensor<32x197x384xf32>
    %v2784 = stablehlo.reshape %v2783 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2785 = stablehlo.reshape %v2774 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2786 = stablehlo.dot_general %v2785, %b9_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v2787 = stablehlo.broadcast_in_dim %b9_bv, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2788 = stablehlo.add %v2786, %v2787 : tensor<32x197x384xf32>
    %v2789 = stablehlo.reshape %v2788 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2790 = stablehlo.reshape %v2779 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2791 = stablehlo.slice %v2790 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2792 = stablehlo.reshape %v2791 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2793 = stablehlo.reshape %v2784 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2794 = stablehlo.slice %v2793 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2795 = stablehlo.reshape %v2794 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2796 = stablehlo.reshape %v2789 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2797 = stablehlo.slice %v2796 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2798 = stablehlo.reshape %v2797 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2799 = stablehlo.reshape %v2795 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2800 = stablehlo.transpose %v2799, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2801 = stablehlo.reshape %v2800 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2802 = stablehlo.reshape %v2792 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2803 = stablehlo.reshape %v2801 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2804 = stablehlo.dot_general %v2802, %v2803, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2805 = stablehlo.reshape %v2804 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2806 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2807 = stablehlo.multiply %v2805, %v2806 : tensor<32x38809xf32>
    %v2808 = stablehlo.reshape %v2807 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2809 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2810 = stablehlo.exponential %v2808 : tensor<32x197x197xf32>
    %v2811 = stablehlo.reduce(%v2810 init: %v2809) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2812 = stablehlo.broadcast_in_dim %v2811, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2813 = stablehlo.divide %v2810, %v2812 : tensor<32x197x197xf32>
    %v2814 = stablehlo.reshape %v2813 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2815 = stablehlo.reshape %v2814 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2816 = stablehlo.reshape %v2798 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2817 = stablehlo.dot_general %v2815, %v2816, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2818 = stablehlo.reshape %v2817 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2819 = stablehlo.reshape %v2818 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2820 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2821 = stablehlo.pad %v2819, %v2820, low = [0, 0, 0], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v2822 = stablehlo.reshape %v2821 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2823 = stablehlo.reshape %v2779 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2824 = stablehlo.slice %v2823 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2825 = stablehlo.reshape %v2824 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2826 = stablehlo.reshape %v2784 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2827 = stablehlo.slice %v2826 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2828 = stablehlo.reshape %v2827 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2829 = stablehlo.reshape %v2789 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2830 = stablehlo.slice %v2829 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2831 = stablehlo.reshape %v2830 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2832 = stablehlo.reshape %v2828 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2833 = stablehlo.transpose %v2832, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2834 = stablehlo.reshape %v2833 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2835 = stablehlo.reshape %v2825 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2836 = stablehlo.reshape %v2834 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2837 = stablehlo.dot_general %v2835, %v2836, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2838 = stablehlo.reshape %v2837 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2839 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2840 = stablehlo.multiply %v2838, %v2839 : tensor<32x38809xf32>
    %v2841 = stablehlo.reshape %v2840 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2842 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2843 = stablehlo.exponential %v2841 : tensor<32x197x197xf32>
    %v2844 = stablehlo.reduce(%v2843 init: %v2842) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2845 = stablehlo.broadcast_in_dim %v2844, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2846 = stablehlo.divide %v2843, %v2845 : tensor<32x197x197xf32>
    %v2847 = stablehlo.reshape %v2846 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2848 = stablehlo.reshape %v2847 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2849 = stablehlo.reshape %v2831 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2850 = stablehlo.dot_general %v2848, %v2849, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2851 = stablehlo.reshape %v2850 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2852 = stablehlo.reshape %v2851 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2853 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2854 = stablehlo.pad %v2852, %v2853, low = [0, 0, 64], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v2855 = stablehlo.reshape %v2854 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2856 = stablehlo.add %v2822, %v2855 : tensor<32x75648xf32>
    %v2857 = stablehlo.reshape %v2779 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2858 = stablehlo.slice %v2857 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2859 = stablehlo.reshape %v2858 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2860 = stablehlo.reshape %v2784 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2861 = stablehlo.slice %v2860 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2862 = stablehlo.reshape %v2861 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2863 = stablehlo.reshape %v2789 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2864 = stablehlo.slice %v2863 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2865 = stablehlo.reshape %v2864 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2866 = stablehlo.reshape %v2862 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2867 = stablehlo.transpose %v2866, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2868 = stablehlo.reshape %v2867 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2869 = stablehlo.reshape %v2859 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2870 = stablehlo.reshape %v2868 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2871 = stablehlo.dot_general %v2869, %v2870, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2872 = stablehlo.reshape %v2871 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2873 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2874 = stablehlo.multiply %v2872, %v2873 : tensor<32x38809xf32>
    %v2875 = stablehlo.reshape %v2874 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2876 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2877 = stablehlo.exponential %v2875 : tensor<32x197x197xf32>
    %v2878 = stablehlo.reduce(%v2877 init: %v2876) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2879 = stablehlo.broadcast_in_dim %v2878, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2880 = stablehlo.divide %v2877, %v2879 : tensor<32x197x197xf32>
    %v2881 = stablehlo.reshape %v2880 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2882 = stablehlo.reshape %v2881 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2883 = stablehlo.reshape %v2865 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2884 = stablehlo.dot_general %v2882, %v2883, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2885 = stablehlo.reshape %v2884 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2886 = stablehlo.reshape %v2885 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2887 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2888 = stablehlo.pad %v2886, %v2887, low = [0, 0, 128], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v2889 = stablehlo.reshape %v2888 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2890 = stablehlo.add %v2856, %v2889 : tensor<32x75648xf32>
    %v2891 = stablehlo.reshape %v2779 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2892 = stablehlo.slice %v2891 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2893 = stablehlo.reshape %v2892 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2894 = stablehlo.reshape %v2784 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2895 = stablehlo.slice %v2894 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2896 = stablehlo.reshape %v2895 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2897 = stablehlo.reshape %v2789 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2898 = stablehlo.slice %v2897 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2899 = stablehlo.reshape %v2898 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2900 = stablehlo.reshape %v2896 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2901 = stablehlo.transpose %v2900, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2902 = stablehlo.reshape %v2901 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2903 = stablehlo.reshape %v2893 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2904 = stablehlo.reshape %v2902 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2905 = stablehlo.dot_general %v2903, %v2904, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2906 = stablehlo.reshape %v2905 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2907 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2908 = stablehlo.multiply %v2906, %v2907 : tensor<32x38809xf32>
    %v2909 = stablehlo.reshape %v2908 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2910 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2911 = stablehlo.exponential %v2909 : tensor<32x197x197xf32>
    %v2912 = stablehlo.reduce(%v2911 init: %v2910) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2913 = stablehlo.broadcast_in_dim %v2912, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2914 = stablehlo.divide %v2911, %v2913 : tensor<32x197x197xf32>
    %v2915 = stablehlo.reshape %v2914 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2916 = stablehlo.reshape %v2915 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2917 = stablehlo.reshape %v2899 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2918 = stablehlo.dot_general %v2916, %v2917, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2919 = stablehlo.reshape %v2918 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2920 = stablehlo.reshape %v2919 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2921 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2922 = stablehlo.pad %v2920, %v2921, low = [0, 0, 192], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v2923 = stablehlo.reshape %v2922 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2924 = stablehlo.add %v2890, %v2923 : tensor<32x75648xf32>
    %v2925 = stablehlo.reshape %v2779 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2926 = stablehlo.slice %v2925 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2927 = stablehlo.reshape %v2926 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2928 = stablehlo.reshape %v2784 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2929 = stablehlo.slice %v2928 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2930 = stablehlo.reshape %v2929 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2931 = stablehlo.reshape %v2789 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2932 = stablehlo.slice %v2931 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2933 = stablehlo.reshape %v2932 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2934 = stablehlo.reshape %v2930 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2935 = stablehlo.transpose %v2934, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2936 = stablehlo.reshape %v2935 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2937 = stablehlo.reshape %v2927 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2938 = stablehlo.reshape %v2936 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2939 = stablehlo.dot_general %v2937, %v2938, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2940 = stablehlo.reshape %v2939 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2941 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2942 = stablehlo.multiply %v2940, %v2941 : tensor<32x38809xf32>
    %v2943 = stablehlo.reshape %v2942 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2944 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2945 = stablehlo.exponential %v2943 : tensor<32x197x197xf32>
    %v2946 = stablehlo.reduce(%v2945 init: %v2944) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2947 = stablehlo.broadcast_in_dim %v2946, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2948 = stablehlo.divide %v2945, %v2947 : tensor<32x197x197xf32>
    %v2949 = stablehlo.reshape %v2948 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2950 = stablehlo.reshape %v2949 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2951 = stablehlo.reshape %v2933 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2952 = stablehlo.dot_general %v2950, %v2951, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2953 = stablehlo.reshape %v2952 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2954 = stablehlo.reshape %v2953 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2955 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2956 = stablehlo.pad %v2954, %v2955, low = [0, 0, 256], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v2957 = stablehlo.reshape %v2956 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2958 = stablehlo.add %v2924, %v2957 : tensor<32x75648xf32>
    %v2959 = stablehlo.reshape %v2779 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2960 = stablehlo.slice %v2959 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2961 = stablehlo.reshape %v2960 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2962 = stablehlo.reshape %v2784 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2963 = stablehlo.slice %v2962 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2964 = stablehlo.reshape %v2963 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2965 = stablehlo.reshape %v2789 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2966 = stablehlo.slice %v2965 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2967 = stablehlo.reshape %v2966 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2968 = stablehlo.reshape %v2964 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2969 = stablehlo.transpose %v2968, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2970 = stablehlo.reshape %v2969 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2971 = stablehlo.reshape %v2961 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2972 = stablehlo.reshape %v2970 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2973 = stablehlo.dot_general %v2971, %v2972, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2974 = stablehlo.reshape %v2973 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2975 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2976 = stablehlo.multiply %v2974, %v2975 : tensor<32x38809xf32>
    %v2977 = stablehlo.reshape %v2976 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2978 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2979 = stablehlo.exponential %v2977 : tensor<32x197x197xf32>
    %v2980 = stablehlo.reduce(%v2979 init: %v2978) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2981 = stablehlo.broadcast_in_dim %v2980, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2982 = stablehlo.divide %v2979, %v2981 : tensor<32x197x197xf32>
    %v2983 = stablehlo.reshape %v2982 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2984 = stablehlo.reshape %v2983 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2985 = stablehlo.reshape %v2967 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2986 = stablehlo.dot_general %v2984, %v2985, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2987 = stablehlo.reshape %v2986 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2988 = stablehlo.reshape %v2987 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2989 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2990 = stablehlo.pad %v2988, %v2989, low = [0, 0, 320], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v2991 = stablehlo.reshape %v2990 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2992 = stablehlo.add %v2958, %v2991 : tensor<32x75648xf32>
    %v2993 = stablehlo.reshape %v2992 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2994 = stablehlo.dot_general %v2993, %b9_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v2995 = stablehlo.broadcast_in_dim %b9_bo, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2996 = stablehlo.add %v2994, %v2995 : tensor<32x197x384xf32>
    %v2997 = stablehlo.reshape %v2996 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2998 = stablehlo.add %v2746, %v2997 : tensor<32x75648xf32>
    %v2999 = stablehlo.reshape %v2998 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3000 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3001 = stablehlo.constant dense<384.0> : tensor<32x197x384xf32>
    %v3002 = stablehlo.constant dense<1.0e-5> : tensor<32x197x384xf32>
    %v3003 = stablehlo.reduce(%v2999 init: %v3000) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3004 = stablehlo.broadcast_in_dim %v3003, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v3005 = stablehlo.divide %v3004, %v3001 : tensor<32x197x384xf32>
    %v3006 = stablehlo.subtract %v2999, %v3005 : tensor<32x197x384xf32>
    %v3007 = stablehlo.multiply %v3006, %v3006 : tensor<32x197x384xf32>
    %v3008 = stablehlo.reduce(%v3007 init: %v3000) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3009 = stablehlo.broadcast_in_dim %v3008, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v3010 = stablehlo.divide %v3009, %v3001 : tensor<32x197x384xf32>
    %v3011 = stablehlo.add %v3010, %v3002 : tensor<32x197x384xf32>
    %v3012 = stablehlo.rsqrt %v3011 : tensor<32x197x384xf32>
    %v3013 = stablehlo.multiply %v3006, %v3012 : tensor<32x197x384xf32>
    %v3014 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v3015 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v3016 = stablehlo.multiply %v3013, %v3014 : tensor<32x197x384xf32>
    %v3017 = stablehlo.add %v3016, %v3015 : tensor<32x197x384xf32>
    %v3018 = stablehlo.reshape %v3017 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3019 = stablehlo.reshape %v3018 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3020 = stablehlo.broadcast_in_dim %b9_g2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v3021 = stablehlo.multiply %v3019, %v3020 : tensor<32x197x384xf32>
    %v3022 = stablehlo.reshape %v3021 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3023 = stablehlo.reshape %v3022 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3024 = stablehlo.broadcast_in_dim %b9_bt2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v3025 = stablehlo.add %v3023, %v3024 : tensor<32x197x384xf32>
    %v3026 = stablehlo.reshape %v3025 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3027 = stablehlo.reshape %v3026 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3028 = stablehlo.dot_general %v3027, %b9_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x1536xf32>) -> tensor<32x197x1536xf32>
    %v3029 = stablehlo.broadcast_in_dim %b9_bfc1, dims = [2] : (tensor<1536xf32>) -> tensor<32x197x1536xf32>
    %v3030 = stablehlo.add %v3028, %v3029 : tensor<32x197x1536xf32>
    %v3031 = stablehlo.reshape %v3030 : (tensor<32x197x1536xf32>) -> tensor<32x302592xf32>
    %v3032 = stablehlo.multiply %v3031, %v3031 : tensor<32x302592xf32>
    %v3033 = stablehlo.multiply %v3032, %v3031 : tensor<32x302592xf32>
    %v3034 = stablehlo.constant dense<0.044715> : tensor<32x302592xf32>
    %v3035 = stablehlo.multiply %v3034, %v3033 : tensor<32x302592xf32>
    %v3036 = stablehlo.add %v3031, %v3035 : tensor<32x302592xf32>
    %v3037 = stablehlo.constant dense<0.7978845608028654> : tensor<32x302592xf32>
    %v3038 = stablehlo.multiply %v3037, %v3036 : tensor<32x302592xf32>
    %v3039 = stablehlo.tanh %v3038 : tensor<32x302592xf32>
    %v3040 = stablehlo.constant dense<1.0> : tensor<32x302592xf32>
    %v3041 = stablehlo.add %v3040, %v3039 : tensor<32x302592xf32>
    %v3042 = stablehlo.constant dense<0.5> : tensor<32x302592xf32>
    %v3043 = stablehlo.multiply %v3042, %v3031 : tensor<32x302592xf32>
    %v3044 = stablehlo.multiply %v3043, %v3041 : tensor<32x302592xf32>
    %v3045 = stablehlo.reshape %v3044 : (tensor<32x302592xf32>) -> tensor<32x197x1536xf32>
    %v3046 = stablehlo.dot_general %v3045, %b9_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x1536xf32>, tensor<1536x384xf32>) -> tensor<32x197x384xf32>
    %v3047 = stablehlo.broadcast_in_dim %b9_bfc2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v3048 = stablehlo.add %v3046, %v3047 : tensor<32x197x384xf32>
    %v3049 = stablehlo.reshape %v3048 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3050 = stablehlo.add %v2998, %v3049 : tensor<32x75648xf32>
    %v3051 = stablehlo.reshape %v3050 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3052 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3053 = stablehlo.constant dense<384.0> : tensor<32x197x384xf32>
    %v3054 = stablehlo.constant dense<1.0e-5> : tensor<32x197x384xf32>
    %v3055 = stablehlo.reduce(%v3051 init: %v3052) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3056 = stablehlo.broadcast_in_dim %v3055, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v3057 = stablehlo.divide %v3056, %v3053 : tensor<32x197x384xf32>
    %v3058 = stablehlo.subtract %v3051, %v3057 : tensor<32x197x384xf32>
    %v3059 = stablehlo.multiply %v3058, %v3058 : tensor<32x197x384xf32>
    %v3060 = stablehlo.reduce(%v3059 init: %v3052) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3061 = stablehlo.broadcast_in_dim %v3060, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v3062 = stablehlo.divide %v3061, %v3053 : tensor<32x197x384xf32>
    %v3063 = stablehlo.add %v3062, %v3054 : tensor<32x197x384xf32>
    %v3064 = stablehlo.rsqrt %v3063 : tensor<32x197x384xf32>
    %v3065 = stablehlo.multiply %v3058, %v3064 : tensor<32x197x384xf32>
    %v3066 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v3067 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v3068 = stablehlo.multiply %v3065, %v3066 : tensor<32x197x384xf32>
    %v3069 = stablehlo.add %v3068, %v3067 : tensor<32x197x384xf32>
    %v3070 = stablehlo.reshape %v3069 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3071 = stablehlo.reshape %v3070 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3072 = stablehlo.broadcast_in_dim %b10_g1, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v3073 = stablehlo.multiply %v3071, %v3072 : tensor<32x197x384xf32>
    %v3074 = stablehlo.reshape %v3073 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3075 = stablehlo.reshape %v3074 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3076 = stablehlo.broadcast_in_dim %b10_bt1, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v3077 = stablehlo.add %v3075, %v3076 : tensor<32x197x384xf32>
    %v3078 = stablehlo.reshape %v3077 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3079 = stablehlo.reshape %v3078 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3080 = stablehlo.dot_general %v3079, %b10_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v3081 = stablehlo.broadcast_in_dim %b10_bq, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v3082 = stablehlo.add %v3080, %v3081 : tensor<32x197x384xf32>
    %v3083 = stablehlo.reshape %v3082 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3084 = stablehlo.reshape %v3078 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3085 = stablehlo.dot_general %v3084, %b10_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v3086 = stablehlo.broadcast_in_dim %b10_bk, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v3087 = stablehlo.add %v3085, %v3086 : tensor<32x197x384xf32>
    %v3088 = stablehlo.reshape %v3087 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3089 = stablehlo.reshape %v3078 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3090 = stablehlo.dot_general %v3089, %b10_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v3091 = stablehlo.broadcast_in_dim %b10_bv, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v3092 = stablehlo.add %v3090, %v3091 : tensor<32x197x384xf32>
    %v3093 = stablehlo.reshape %v3092 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3094 = stablehlo.reshape %v3083 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3095 = stablehlo.slice %v3094 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3096 = stablehlo.reshape %v3095 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3097 = stablehlo.reshape %v3088 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3098 = stablehlo.slice %v3097 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3099 = stablehlo.reshape %v3098 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3100 = stablehlo.reshape %v3093 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3101 = stablehlo.slice %v3100 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3102 = stablehlo.reshape %v3101 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3103 = stablehlo.reshape %v3099 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3104 = stablehlo.transpose %v3103, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3105 = stablehlo.reshape %v3104 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3106 = stablehlo.reshape %v3096 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3107 = stablehlo.reshape %v3105 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3108 = stablehlo.dot_general %v3106, %v3107, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3109 = stablehlo.reshape %v3108 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3110 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3111 = stablehlo.multiply %v3109, %v3110 : tensor<32x38809xf32>
    %v3112 = stablehlo.reshape %v3111 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3113 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3114 = stablehlo.exponential %v3112 : tensor<32x197x197xf32>
    %v3115 = stablehlo.reduce(%v3114 init: %v3113) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3116 = stablehlo.broadcast_in_dim %v3115, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3117 = stablehlo.divide %v3114, %v3116 : tensor<32x197x197xf32>
    %v3118 = stablehlo.reshape %v3117 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3119 = stablehlo.reshape %v3118 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3120 = stablehlo.reshape %v3102 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3121 = stablehlo.dot_general %v3119, %v3120, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3122 = stablehlo.reshape %v3121 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3123 = stablehlo.reshape %v3122 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3124 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3125 = stablehlo.pad %v3123, %v3124, low = [0, 0, 0], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v3126 = stablehlo.reshape %v3125 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3127 = stablehlo.reshape %v3083 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3128 = stablehlo.slice %v3127 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3129 = stablehlo.reshape %v3128 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3130 = stablehlo.reshape %v3088 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3131 = stablehlo.slice %v3130 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3132 = stablehlo.reshape %v3131 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3133 = stablehlo.reshape %v3093 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3134 = stablehlo.slice %v3133 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3135 = stablehlo.reshape %v3134 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3136 = stablehlo.reshape %v3132 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3137 = stablehlo.transpose %v3136, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3138 = stablehlo.reshape %v3137 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3139 = stablehlo.reshape %v3129 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3140 = stablehlo.reshape %v3138 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3141 = stablehlo.dot_general %v3139, %v3140, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3142 = stablehlo.reshape %v3141 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3143 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3144 = stablehlo.multiply %v3142, %v3143 : tensor<32x38809xf32>
    %v3145 = stablehlo.reshape %v3144 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3146 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3147 = stablehlo.exponential %v3145 : tensor<32x197x197xf32>
    %v3148 = stablehlo.reduce(%v3147 init: %v3146) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3149 = stablehlo.broadcast_in_dim %v3148, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3150 = stablehlo.divide %v3147, %v3149 : tensor<32x197x197xf32>
    %v3151 = stablehlo.reshape %v3150 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3152 = stablehlo.reshape %v3151 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3153 = stablehlo.reshape %v3135 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3154 = stablehlo.dot_general %v3152, %v3153, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3155 = stablehlo.reshape %v3154 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3156 = stablehlo.reshape %v3155 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3157 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3158 = stablehlo.pad %v3156, %v3157, low = [0, 0, 64], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v3159 = stablehlo.reshape %v3158 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3160 = stablehlo.add %v3126, %v3159 : tensor<32x75648xf32>
    %v3161 = stablehlo.reshape %v3083 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3162 = stablehlo.slice %v3161 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3163 = stablehlo.reshape %v3162 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3164 = stablehlo.reshape %v3088 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3165 = stablehlo.slice %v3164 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3166 = stablehlo.reshape %v3165 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3167 = stablehlo.reshape %v3093 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3168 = stablehlo.slice %v3167 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3169 = stablehlo.reshape %v3168 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3170 = stablehlo.reshape %v3166 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3171 = stablehlo.transpose %v3170, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3172 = stablehlo.reshape %v3171 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3173 = stablehlo.reshape %v3163 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3174 = stablehlo.reshape %v3172 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3175 = stablehlo.dot_general %v3173, %v3174, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3176 = stablehlo.reshape %v3175 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3177 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3178 = stablehlo.multiply %v3176, %v3177 : tensor<32x38809xf32>
    %v3179 = stablehlo.reshape %v3178 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3180 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3181 = stablehlo.exponential %v3179 : tensor<32x197x197xf32>
    %v3182 = stablehlo.reduce(%v3181 init: %v3180) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3183 = stablehlo.broadcast_in_dim %v3182, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3184 = stablehlo.divide %v3181, %v3183 : tensor<32x197x197xf32>
    %v3185 = stablehlo.reshape %v3184 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3186 = stablehlo.reshape %v3185 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3187 = stablehlo.reshape %v3169 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3188 = stablehlo.dot_general %v3186, %v3187, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3189 = stablehlo.reshape %v3188 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3190 = stablehlo.reshape %v3189 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3191 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3192 = stablehlo.pad %v3190, %v3191, low = [0, 0, 128], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v3193 = stablehlo.reshape %v3192 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3194 = stablehlo.add %v3160, %v3193 : tensor<32x75648xf32>
    %v3195 = stablehlo.reshape %v3083 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3196 = stablehlo.slice %v3195 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3197 = stablehlo.reshape %v3196 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3198 = stablehlo.reshape %v3088 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3199 = stablehlo.slice %v3198 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3200 = stablehlo.reshape %v3199 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3201 = stablehlo.reshape %v3093 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3202 = stablehlo.slice %v3201 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3203 = stablehlo.reshape %v3202 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3204 = stablehlo.reshape %v3200 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3205 = stablehlo.transpose %v3204, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3206 = stablehlo.reshape %v3205 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3207 = stablehlo.reshape %v3197 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3208 = stablehlo.reshape %v3206 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3209 = stablehlo.dot_general %v3207, %v3208, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3210 = stablehlo.reshape %v3209 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3211 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3212 = stablehlo.multiply %v3210, %v3211 : tensor<32x38809xf32>
    %v3213 = stablehlo.reshape %v3212 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3214 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3215 = stablehlo.exponential %v3213 : tensor<32x197x197xf32>
    %v3216 = stablehlo.reduce(%v3215 init: %v3214) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3217 = stablehlo.broadcast_in_dim %v3216, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3218 = stablehlo.divide %v3215, %v3217 : tensor<32x197x197xf32>
    %v3219 = stablehlo.reshape %v3218 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3220 = stablehlo.reshape %v3219 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3221 = stablehlo.reshape %v3203 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3222 = stablehlo.dot_general %v3220, %v3221, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3223 = stablehlo.reshape %v3222 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3224 = stablehlo.reshape %v3223 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3225 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3226 = stablehlo.pad %v3224, %v3225, low = [0, 0, 192], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v3227 = stablehlo.reshape %v3226 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3228 = stablehlo.add %v3194, %v3227 : tensor<32x75648xf32>
    %v3229 = stablehlo.reshape %v3083 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3230 = stablehlo.slice %v3229 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3231 = stablehlo.reshape %v3230 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3232 = stablehlo.reshape %v3088 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3233 = stablehlo.slice %v3232 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3234 = stablehlo.reshape %v3233 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3235 = stablehlo.reshape %v3093 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3236 = stablehlo.slice %v3235 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3237 = stablehlo.reshape %v3236 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3238 = stablehlo.reshape %v3234 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3239 = stablehlo.transpose %v3238, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3240 = stablehlo.reshape %v3239 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3241 = stablehlo.reshape %v3231 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3242 = stablehlo.reshape %v3240 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3243 = stablehlo.dot_general %v3241, %v3242, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3244 = stablehlo.reshape %v3243 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3245 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3246 = stablehlo.multiply %v3244, %v3245 : tensor<32x38809xf32>
    %v3247 = stablehlo.reshape %v3246 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3248 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3249 = stablehlo.exponential %v3247 : tensor<32x197x197xf32>
    %v3250 = stablehlo.reduce(%v3249 init: %v3248) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3251 = stablehlo.broadcast_in_dim %v3250, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3252 = stablehlo.divide %v3249, %v3251 : tensor<32x197x197xf32>
    %v3253 = stablehlo.reshape %v3252 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3254 = stablehlo.reshape %v3253 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3255 = stablehlo.reshape %v3237 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3256 = stablehlo.dot_general %v3254, %v3255, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3257 = stablehlo.reshape %v3256 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3258 = stablehlo.reshape %v3257 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3259 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3260 = stablehlo.pad %v3258, %v3259, low = [0, 0, 256], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v3261 = stablehlo.reshape %v3260 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3262 = stablehlo.add %v3228, %v3261 : tensor<32x75648xf32>
    %v3263 = stablehlo.reshape %v3083 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3264 = stablehlo.slice %v3263 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3265 = stablehlo.reshape %v3264 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3266 = stablehlo.reshape %v3088 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3267 = stablehlo.slice %v3266 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3268 = stablehlo.reshape %v3267 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3269 = stablehlo.reshape %v3093 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3270 = stablehlo.slice %v3269 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3271 = stablehlo.reshape %v3270 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3272 = stablehlo.reshape %v3268 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3273 = stablehlo.transpose %v3272, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3274 = stablehlo.reshape %v3273 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3275 = stablehlo.reshape %v3265 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3276 = stablehlo.reshape %v3274 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3277 = stablehlo.dot_general %v3275, %v3276, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3278 = stablehlo.reshape %v3277 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3279 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3280 = stablehlo.multiply %v3278, %v3279 : tensor<32x38809xf32>
    %v3281 = stablehlo.reshape %v3280 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3282 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3283 = stablehlo.exponential %v3281 : tensor<32x197x197xf32>
    %v3284 = stablehlo.reduce(%v3283 init: %v3282) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3285 = stablehlo.broadcast_in_dim %v3284, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3286 = stablehlo.divide %v3283, %v3285 : tensor<32x197x197xf32>
    %v3287 = stablehlo.reshape %v3286 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3288 = stablehlo.reshape %v3287 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3289 = stablehlo.reshape %v3271 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3290 = stablehlo.dot_general %v3288, %v3289, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3291 = stablehlo.reshape %v3290 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3292 = stablehlo.reshape %v3291 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3293 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3294 = stablehlo.pad %v3292, %v3293, low = [0, 0, 320], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v3295 = stablehlo.reshape %v3294 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3296 = stablehlo.add %v3262, %v3295 : tensor<32x75648xf32>
    %v3297 = stablehlo.reshape %v3296 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3298 = stablehlo.dot_general %v3297, %b10_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v3299 = stablehlo.broadcast_in_dim %b10_bo, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v3300 = stablehlo.add %v3298, %v3299 : tensor<32x197x384xf32>
    %v3301 = stablehlo.reshape %v3300 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3302 = stablehlo.add %v3050, %v3301 : tensor<32x75648xf32>
    %v3303 = stablehlo.reshape %v3302 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3304 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3305 = stablehlo.constant dense<384.0> : tensor<32x197x384xf32>
    %v3306 = stablehlo.constant dense<1.0e-5> : tensor<32x197x384xf32>
    %v3307 = stablehlo.reduce(%v3303 init: %v3304) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3308 = stablehlo.broadcast_in_dim %v3307, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v3309 = stablehlo.divide %v3308, %v3305 : tensor<32x197x384xf32>
    %v3310 = stablehlo.subtract %v3303, %v3309 : tensor<32x197x384xf32>
    %v3311 = stablehlo.multiply %v3310, %v3310 : tensor<32x197x384xf32>
    %v3312 = stablehlo.reduce(%v3311 init: %v3304) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3313 = stablehlo.broadcast_in_dim %v3312, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v3314 = stablehlo.divide %v3313, %v3305 : tensor<32x197x384xf32>
    %v3315 = stablehlo.add %v3314, %v3306 : tensor<32x197x384xf32>
    %v3316 = stablehlo.rsqrt %v3315 : tensor<32x197x384xf32>
    %v3317 = stablehlo.multiply %v3310, %v3316 : tensor<32x197x384xf32>
    %v3318 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v3319 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v3320 = stablehlo.multiply %v3317, %v3318 : tensor<32x197x384xf32>
    %v3321 = stablehlo.add %v3320, %v3319 : tensor<32x197x384xf32>
    %v3322 = stablehlo.reshape %v3321 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3323 = stablehlo.reshape %v3322 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3324 = stablehlo.broadcast_in_dim %b10_g2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v3325 = stablehlo.multiply %v3323, %v3324 : tensor<32x197x384xf32>
    %v3326 = stablehlo.reshape %v3325 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3327 = stablehlo.reshape %v3326 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3328 = stablehlo.broadcast_in_dim %b10_bt2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v3329 = stablehlo.add %v3327, %v3328 : tensor<32x197x384xf32>
    %v3330 = stablehlo.reshape %v3329 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3331 = stablehlo.reshape %v3330 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3332 = stablehlo.dot_general %v3331, %b10_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x1536xf32>) -> tensor<32x197x1536xf32>
    %v3333 = stablehlo.broadcast_in_dim %b10_bfc1, dims = [2] : (tensor<1536xf32>) -> tensor<32x197x1536xf32>
    %v3334 = stablehlo.add %v3332, %v3333 : tensor<32x197x1536xf32>
    %v3335 = stablehlo.reshape %v3334 : (tensor<32x197x1536xf32>) -> tensor<32x302592xf32>
    %v3336 = stablehlo.multiply %v3335, %v3335 : tensor<32x302592xf32>
    %v3337 = stablehlo.multiply %v3336, %v3335 : tensor<32x302592xf32>
    %v3338 = stablehlo.constant dense<0.044715> : tensor<32x302592xf32>
    %v3339 = stablehlo.multiply %v3338, %v3337 : tensor<32x302592xf32>
    %v3340 = stablehlo.add %v3335, %v3339 : tensor<32x302592xf32>
    %v3341 = stablehlo.constant dense<0.7978845608028654> : tensor<32x302592xf32>
    %v3342 = stablehlo.multiply %v3341, %v3340 : tensor<32x302592xf32>
    %v3343 = stablehlo.tanh %v3342 : tensor<32x302592xf32>
    %v3344 = stablehlo.constant dense<1.0> : tensor<32x302592xf32>
    %v3345 = stablehlo.add %v3344, %v3343 : tensor<32x302592xf32>
    %v3346 = stablehlo.constant dense<0.5> : tensor<32x302592xf32>
    %v3347 = stablehlo.multiply %v3346, %v3335 : tensor<32x302592xf32>
    %v3348 = stablehlo.multiply %v3347, %v3345 : tensor<32x302592xf32>
    %v3349 = stablehlo.reshape %v3348 : (tensor<32x302592xf32>) -> tensor<32x197x1536xf32>
    %v3350 = stablehlo.dot_general %v3349, %b10_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x1536xf32>, tensor<1536x384xf32>) -> tensor<32x197x384xf32>
    %v3351 = stablehlo.broadcast_in_dim %b10_bfc2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v3352 = stablehlo.add %v3350, %v3351 : tensor<32x197x384xf32>
    %v3353 = stablehlo.reshape %v3352 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3354 = stablehlo.add %v3302, %v3353 : tensor<32x75648xf32>
    %v3355 = stablehlo.reshape %v3354 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3356 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3357 = stablehlo.constant dense<384.0> : tensor<32x197x384xf32>
    %v3358 = stablehlo.constant dense<1.0e-5> : tensor<32x197x384xf32>
    %v3359 = stablehlo.reduce(%v3355 init: %v3356) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3360 = stablehlo.broadcast_in_dim %v3359, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v3361 = stablehlo.divide %v3360, %v3357 : tensor<32x197x384xf32>
    %v3362 = stablehlo.subtract %v3355, %v3361 : tensor<32x197x384xf32>
    %v3363 = stablehlo.multiply %v3362, %v3362 : tensor<32x197x384xf32>
    %v3364 = stablehlo.reduce(%v3363 init: %v3356) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3365 = stablehlo.broadcast_in_dim %v3364, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v3366 = stablehlo.divide %v3365, %v3357 : tensor<32x197x384xf32>
    %v3367 = stablehlo.add %v3366, %v3358 : tensor<32x197x384xf32>
    %v3368 = stablehlo.rsqrt %v3367 : tensor<32x197x384xf32>
    %v3369 = stablehlo.multiply %v3362, %v3368 : tensor<32x197x384xf32>
    %v3370 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v3371 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v3372 = stablehlo.multiply %v3369, %v3370 : tensor<32x197x384xf32>
    %v3373 = stablehlo.add %v3372, %v3371 : tensor<32x197x384xf32>
    %v3374 = stablehlo.reshape %v3373 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3375 = stablehlo.reshape %v3374 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3376 = stablehlo.broadcast_in_dim %b11_g1, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v3377 = stablehlo.multiply %v3375, %v3376 : tensor<32x197x384xf32>
    %v3378 = stablehlo.reshape %v3377 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3379 = stablehlo.reshape %v3378 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3380 = stablehlo.broadcast_in_dim %b11_bt1, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v3381 = stablehlo.add %v3379, %v3380 : tensor<32x197x384xf32>
    %v3382 = stablehlo.reshape %v3381 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3383 = stablehlo.reshape %v3382 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3384 = stablehlo.dot_general %v3383, %b11_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v3385 = stablehlo.broadcast_in_dim %b11_bq, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v3386 = stablehlo.add %v3384, %v3385 : tensor<32x197x384xf32>
    %v3387 = stablehlo.reshape %v3386 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3388 = stablehlo.reshape %v3382 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3389 = stablehlo.dot_general %v3388, %b11_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v3390 = stablehlo.broadcast_in_dim %b11_bk, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v3391 = stablehlo.add %v3389, %v3390 : tensor<32x197x384xf32>
    %v3392 = stablehlo.reshape %v3391 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3393 = stablehlo.reshape %v3382 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3394 = stablehlo.dot_general %v3393, %b11_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v3395 = stablehlo.broadcast_in_dim %b11_bv, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v3396 = stablehlo.add %v3394, %v3395 : tensor<32x197x384xf32>
    %v3397 = stablehlo.reshape %v3396 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3398 = stablehlo.reshape %v3387 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3399 = stablehlo.slice %v3398 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3400 = stablehlo.reshape %v3399 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3401 = stablehlo.reshape %v3392 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3402 = stablehlo.slice %v3401 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3403 = stablehlo.reshape %v3402 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3404 = stablehlo.reshape %v3397 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3405 = stablehlo.slice %v3404 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3406 = stablehlo.reshape %v3405 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3407 = stablehlo.reshape %v3403 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3408 = stablehlo.transpose %v3407, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3409 = stablehlo.reshape %v3408 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3410 = stablehlo.reshape %v3400 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3411 = stablehlo.reshape %v3409 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3412 = stablehlo.dot_general %v3410, %v3411, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3413 = stablehlo.reshape %v3412 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3414 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3415 = stablehlo.multiply %v3413, %v3414 : tensor<32x38809xf32>
    %v3416 = stablehlo.reshape %v3415 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3417 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3418 = stablehlo.exponential %v3416 : tensor<32x197x197xf32>
    %v3419 = stablehlo.reduce(%v3418 init: %v3417) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3420 = stablehlo.broadcast_in_dim %v3419, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3421 = stablehlo.divide %v3418, %v3420 : tensor<32x197x197xf32>
    %v3422 = stablehlo.reshape %v3421 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3423 = stablehlo.reshape %v3422 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3424 = stablehlo.reshape %v3406 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3425 = stablehlo.dot_general %v3423, %v3424, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3426 = stablehlo.reshape %v3425 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3427 = stablehlo.reshape %v3426 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3428 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3429 = stablehlo.pad %v3427, %v3428, low = [0, 0, 0], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v3430 = stablehlo.reshape %v3429 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3431 = stablehlo.reshape %v3387 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3432 = stablehlo.slice %v3431 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3433 = stablehlo.reshape %v3432 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3434 = stablehlo.reshape %v3392 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3435 = stablehlo.slice %v3434 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3436 = stablehlo.reshape %v3435 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3437 = stablehlo.reshape %v3397 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3438 = stablehlo.slice %v3437 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3439 = stablehlo.reshape %v3438 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3440 = stablehlo.reshape %v3436 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3441 = stablehlo.transpose %v3440, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3442 = stablehlo.reshape %v3441 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3443 = stablehlo.reshape %v3433 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3444 = stablehlo.reshape %v3442 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3445 = stablehlo.dot_general %v3443, %v3444, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3446 = stablehlo.reshape %v3445 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3447 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3448 = stablehlo.multiply %v3446, %v3447 : tensor<32x38809xf32>
    %v3449 = stablehlo.reshape %v3448 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3450 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3451 = stablehlo.exponential %v3449 : tensor<32x197x197xf32>
    %v3452 = stablehlo.reduce(%v3451 init: %v3450) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3453 = stablehlo.broadcast_in_dim %v3452, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3454 = stablehlo.divide %v3451, %v3453 : tensor<32x197x197xf32>
    %v3455 = stablehlo.reshape %v3454 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3456 = stablehlo.reshape %v3455 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3457 = stablehlo.reshape %v3439 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3458 = stablehlo.dot_general %v3456, %v3457, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3459 = stablehlo.reshape %v3458 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3460 = stablehlo.reshape %v3459 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3461 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3462 = stablehlo.pad %v3460, %v3461, low = [0, 0, 64], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v3463 = stablehlo.reshape %v3462 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3464 = stablehlo.add %v3430, %v3463 : tensor<32x75648xf32>
    %v3465 = stablehlo.reshape %v3387 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3466 = stablehlo.slice %v3465 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3467 = stablehlo.reshape %v3466 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3468 = stablehlo.reshape %v3392 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3469 = stablehlo.slice %v3468 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3470 = stablehlo.reshape %v3469 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3471 = stablehlo.reshape %v3397 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3472 = stablehlo.slice %v3471 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3473 = stablehlo.reshape %v3472 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3474 = stablehlo.reshape %v3470 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3475 = stablehlo.transpose %v3474, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3476 = stablehlo.reshape %v3475 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3477 = stablehlo.reshape %v3467 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3478 = stablehlo.reshape %v3476 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3479 = stablehlo.dot_general %v3477, %v3478, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3480 = stablehlo.reshape %v3479 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3481 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3482 = stablehlo.multiply %v3480, %v3481 : tensor<32x38809xf32>
    %v3483 = stablehlo.reshape %v3482 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3484 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3485 = stablehlo.exponential %v3483 : tensor<32x197x197xf32>
    %v3486 = stablehlo.reduce(%v3485 init: %v3484) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3487 = stablehlo.broadcast_in_dim %v3486, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3488 = stablehlo.divide %v3485, %v3487 : tensor<32x197x197xf32>
    %v3489 = stablehlo.reshape %v3488 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3490 = stablehlo.reshape %v3489 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3491 = stablehlo.reshape %v3473 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3492 = stablehlo.dot_general %v3490, %v3491, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3493 = stablehlo.reshape %v3492 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3494 = stablehlo.reshape %v3493 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3495 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3496 = stablehlo.pad %v3494, %v3495, low = [0, 0, 128], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v3497 = stablehlo.reshape %v3496 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3498 = stablehlo.add %v3464, %v3497 : tensor<32x75648xf32>
    %v3499 = stablehlo.reshape %v3387 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3500 = stablehlo.slice %v3499 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3501 = stablehlo.reshape %v3500 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3502 = stablehlo.reshape %v3392 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3503 = stablehlo.slice %v3502 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3504 = stablehlo.reshape %v3503 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3505 = stablehlo.reshape %v3397 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3506 = stablehlo.slice %v3505 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3507 = stablehlo.reshape %v3506 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3508 = stablehlo.reshape %v3504 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3509 = stablehlo.transpose %v3508, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3510 = stablehlo.reshape %v3509 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3511 = stablehlo.reshape %v3501 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3512 = stablehlo.reshape %v3510 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3513 = stablehlo.dot_general %v3511, %v3512, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3514 = stablehlo.reshape %v3513 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3515 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3516 = stablehlo.multiply %v3514, %v3515 : tensor<32x38809xf32>
    %v3517 = stablehlo.reshape %v3516 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3518 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3519 = stablehlo.exponential %v3517 : tensor<32x197x197xf32>
    %v3520 = stablehlo.reduce(%v3519 init: %v3518) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3521 = stablehlo.broadcast_in_dim %v3520, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3522 = stablehlo.divide %v3519, %v3521 : tensor<32x197x197xf32>
    %v3523 = stablehlo.reshape %v3522 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3524 = stablehlo.reshape %v3523 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3525 = stablehlo.reshape %v3507 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3526 = stablehlo.dot_general %v3524, %v3525, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3527 = stablehlo.reshape %v3526 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3528 = stablehlo.reshape %v3527 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3529 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3530 = stablehlo.pad %v3528, %v3529, low = [0, 0, 192], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v3531 = stablehlo.reshape %v3530 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3532 = stablehlo.add %v3498, %v3531 : tensor<32x75648xf32>
    %v3533 = stablehlo.reshape %v3387 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3534 = stablehlo.slice %v3533 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3535 = stablehlo.reshape %v3534 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3536 = stablehlo.reshape %v3392 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3537 = stablehlo.slice %v3536 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3538 = stablehlo.reshape %v3537 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3539 = stablehlo.reshape %v3397 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3540 = stablehlo.slice %v3539 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3541 = stablehlo.reshape %v3540 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3542 = stablehlo.reshape %v3538 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3543 = stablehlo.transpose %v3542, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3544 = stablehlo.reshape %v3543 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3545 = stablehlo.reshape %v3535 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3546 = stablehlo.reshape %v3544 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3547 = stablehlo.dot_general %v3545, %v3546, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3548 = stablehlo.reshape %v3547 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3549 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3550 = stablehlo.multiply %v3548, %v3549 : tensor<32x38809xf32>
    %v3551 = stablehlo.reshape %v3550 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3552 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3553 = stablehlo.exponential %v3551 : tensor<32x197x197xf32>
    %v3554 = stablehlo.reduce(%v3553 init: %v3552) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3555 = stablehlo.broadcast_in_dim %v3554, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3556 = stablehlo.divide %v3553, %v3555 : tensor<32x197x197xf32>
    %v3557 = stablehlo.reshape %v3556 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3558 = stablehlo.reshape %v3557 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3559 = stablehlo.reshape %v3541 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3560 = stablehlo.dot_general %v3558, %v3559, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3561 = stablehlo.reshape %v3560 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3562 = stablehlo.reshape %v3561 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3563 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3564 = stablehlo.pad %v3562, %v3563, low = [0, 0, 256], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v3565 = stablehlo.reshape %v3564 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3566 = stablehlo.add %v3532, %v3565 : tensor<32x75648xf32>
    %v3567 = stablehlo.reshape %v3387 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3568 = stablehlo.slice %v3567 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3569 = stablehlo.reshape %v3568 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3570 = stablehlo.reshape %v3392 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3571 = stablehlo.slice %v3570 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3572 = stablehlo.reshape %v3571 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3573 = stablehlo.reshape %v3397 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3574 = stablehlo.slice %v3573 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3575 = stablehlo.reshape %v3574 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3576 = stablehlo.reshape %v3572 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3577 = stablehlo.transpose %v3576, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3578 = stablehlo.reshape %v3577 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3579 = stablehlo.reshape %v3569 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3580 = stablehlo.reshape %v3578 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3581 = stablehlo.dot_general %v3579, %v3580, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3582 = stablehlo.reshape %v3581 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3583 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3584 = stablehlo.multiply %v3582, %v3583 : tensor<32x38809xf32>
    %v3585 = stablehlo.reshape %v3584 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3586 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3587 = stablehlo.exponential %v3585 : tensor<32x197x197xf32>
    %v3588 = stablehlo.reduce(%v3587 init: %v3586) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3589 = stablehlo.broadcast_in_dim %v3588, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3590 = stablehlo.divide %v3587, %v3589 : tensor<32x197x197xf32>
    %v3591 = stablehlo.reshape %v3590 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3592 = stablehlo.reshape %v3591 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3593 = stablehlo.reshape %v3575 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3594 = stablehlo.dot_general %v3592, %v3593, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3595 = stablehlo.reshape %v3594 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3596 = stablehlo.reshape %v3595 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3597 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3598 = stablehlo.pad %v3596, %v3597, low = [0, 0, 320], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v3599 = stablehlo.reshape %v3598 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3600 = stablehlo.add %v3566, %v3599 : tensor<32x75648xf32>
    %v3601 = stablehlo.reshape %v3600 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3602 = stablehlo.dot_general %v3601, %b11_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v3603 = stablehlo.broadcast_in_dim %b11_bo, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v3604 = stablehlo.add %v3602, %v3603 : tensor<32x197x384xf32>
    %v3605 = stablehlo.reshape %v3604 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3606 = stablehlo.add %v3354, %v3605 : tensor<32x75648xf32>
    %v3607 = stablehlo.reshape %v3606 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3608 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3609 = stablehlo.constant dense<384.0> : tensor<32x197x384xf32>
    %v3610 = stablehlo.constant dense<1.0e-5> : tensor<32x197x384xf32>
    %v3611 = stablehlo.reduce(%v3607 init: %v3608) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3612 = stablehlo.broadcast_in_dim %v3611, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v3613 = stablehlo.divide %v3612, %v3609 : tensor<32x197x384xf32>
    %v3614 = stablehlo.subtract %v3607, %v3613 : tensor<32x197x384xf32>
    %v3615 = stablehlo.multiply %v3614, %v3614 : tensor<32x197x384xf32>
    %v3616 = stablehlo.reduce(%v3615 init: %v3608) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3617 = stablehlo.broadcast_in_dim %v3616, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v3618 = stablehlo.divide %v3617, %v3609 : tensor<32x197x384xf32>
    %v3619 = stablehlo.add %v3618, %v3610 : tensor<32x197x384xf32>
    %v3620 = stablehlo.rsqrt %v3619 : tensor<32x197x384xf32>
    %v3621 = stablehlo.multiply %v3614, %v3620 : tensor<32x197x384xf32>
    %v3622 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v3623 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v3624 = stablehlo.multiply %v3621, %v3622 : tensor<32x197x384xf32>
    %v3625 = stablehlo.add %v3624, %v3623 : tensor<32x197x384xf32>
    %v3626 = stablehlo.reshape %v3625 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3627 = stablehlo.reshape %v3626 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3628 = stablehlo.broadcast_in_dim %b11_g2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v3629 = stablehlo.multiply %v3627, %v3628 : tensor<32x197x384xf32>
    %v3630 = stablehlo.reshape %v3629 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3631 = stablehlo.reshape %v3630 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3632 = stablehlo.broadcast_in_dim %b11_bt2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v3633 = stablehlo.add %v3631, %v3632 : tensor<32x197x384xf32>
    %v3634 = stablehlo.reshape %v3633 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3635 = stablehlo.reshape %v3634 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3636 = stablehlo.dot_general %v3635, %b11_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x1536xf32>) -> tensor<32x197x1536xf32>
    %v3637 = stablehlo.broadcast_in_dim %b11_bfc1, dims = [2] : (tensor<1536xf32>) -> tensor<32x197x1536xf32>
    %v3638 = stablehlo.add %v3636, %v3637 : tensor<32x197x1536xf32>
    %v3639 = stablehlo.reshape %v3638 : (tensor<32x197x1536xf32>) -> tensor<32x302592xf32>
    %v3640 = stablehlo.multiply %v3639, %v3639 : tensor<32x302592xf32>
    %v3641 = stablehlo.multiply %v3640, %v3639 : tensor<32x302592xf32>
    %v3642 = stablehlo.constant dense<0.044715> : tensor<32x302592xf32>
    %v3643 = stablehlo.multiply %v3642, %v3641 : tensor<32x302592xf32>
    %v3644 = stablehlo.add %v3639, %v3643 : tensor<32x302592xf32>
    %v3645 = stablehlo.constant dense<0.7978845608028654> : tensor<32x302592xf32>
    %v3646 = stablehlo.multiply %v3645, %v3644 : tensor<32x302592xf32>
    %v3647 = stablehlo.tanh %v3646 : tensor<32x302592xf32>
    %v3648 = stablehlo.constant dense<1.0> : tensor<32x302592xf32>
    %v3649 = stablehlo.add %v3648, %v3647 : tensor<32x302592xf32>
    %v3650 = stablehlo.constant dense<0.5> : tensor<32x302592xf32>
    %v3651 = stablehlo.multiply %v3650, %v3639 : tensor<32x302592xf32>
    %v3652 = stablehlo.multiply %v3651, %v3649 : tensor<32x302592xf32>
    %v3653 = stablehlo.reshape %v3652 : (tensor<32x302592xf32>) -> tensor<32x197x1536xf32>
    %v3654 = stablehlo.dot_general %v3653, %b11_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x1536xf32>, tensor<1536x384xf32>) -> tensor<32x197x384xf32>
    %v3655 = stablehlo.broadcast_in_dim %b11_bfc2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v3656 = stablehlo.add %v3654, %v3655 : tensor<32x197x384xf32>
    %v3657 = stablehlo.reshape %v3656 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3658 = stablehlo.add %v3606, %v3657 : tensor<32x75648xf32>
    %v3659 = stablehlo.reshape %v3658 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3660 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3661 = stablehlo.constant dense<384.0> : tensor<32x197x384xf32>
    %v3662 = stablehlo.constant dense<1.0e-5> : tensor<32x197x384xf32>
    %v3663 = stablehlo.reduce(%v3659 init: %v3660) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3664 = stablehlo.broadcast_in_dim %v3663, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v3665 = stablehlo.divide %v3664, %v3661 : tensor<32x197x384xf32>
    %v3666 = stablehlo.subtract %v3659, %v3665 : tensor<32x197x384xf32>
    %v3667 = stablehlo.multiply %v3666, %v3666 : tensor<32x197x384xf32>
    %v3668 = stablehlo.reduce(%v3667 init: %v3660) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3669 = stablehlo.broadcast_in_dim %v3668, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v3670 = stablehlo.divide %v3669, %v3661 : tensor<32x197x384xf32>
    %v3671 = stablehlo.add %v3670, %v3662 : tensor<32x197x384xf32>
    %v3672 = stablehlo.rsqrt %v3671 : tensor<32x197x384xf32>
    %v3673 = stablehlo.multiply %v3666, %v3672 : tensor<32x197x384xf32>
    %v3674 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v3675 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v3676 = stablehlo.multiply %v3673, %v3674 : tensor<32x197x384xf32>
    %v3677 = stablehlo.add %v3676, %v3675 : tensor<32x197x384xf32>
    %v3678 = stablehlo.reshape %v3677 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3679 = stablehlo.reshape %v3678 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3680 = stablehlo.broadcast_in_dim %gF, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v3681 = stablehlo.multiply %v3679, %v3680 : tensor<32x197x384xf32>
    %v3682 = stablehlo.reshape %v3681 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3683 = stablehlo.reshape %v3682 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3684 = stablehlo.broadcast_in_dim %btF, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v3685 = stablehlo.add %v3683, %v3684 : tensor<32x197x384xf32>
    %v3686 = stablehlo.reshape %v3685 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3687 = stablehlo.reshape %v3686 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3688 = stablehlo.slice %v3687 [0:32, 0:1, 0:384] : (tensor<32x197x384xf32>) -> tensor<32x1x384xf32>
    %v3689 = stablehlo.reshape %v3688 : (tensor<32x1x384xf32>) -> tensor<32x384xf32>
    %v3690 = stablehlo.dot_general %v3689, %Wc, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x384xf32>, tensor<384x1000xf32>) -> tensor<32x1000xf32>
    %v3691 = stablehlo.broadcast_in_dim %bc, dims = [1] : (tensor<1000xf32>) -> tensor<32x1000xf32>
    %v3692 = stablehlo.add %v3690, %v3691 : tensor<32x1000xf32>
    return %v3692 : tensor<32x1000xf32>
  }
}

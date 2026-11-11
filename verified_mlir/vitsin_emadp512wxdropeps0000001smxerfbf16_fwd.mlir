module @m {
  func.func @vitsin_emadp512wxdropeps0000001smxerfbf16_fwd(%x: tensor<256x150528xf32>, %wConv: tensor<384x3x16x16xf32>, %bConv: tensor<384xf32>, %cls: tensor<384xf32>, %pos: tensor<197x384xf32>, %b0_g1: tensor<384xf32>, %b0_bt1: tensor<384xf32>, %b0_Wq: tensor<384x384xf32>, %b0_bq: tensor<384xf32>, %b0_Wk: tensor<384x384xf32>, %b0_bk: tensor<384xf32>, %b0_Wv: tensor<384x384xf32>, %b0_bv: tensor<384xf32>, %b0_Wo: tensor<384x384xf32>, %b0_bo: tensor<384xf32>, %b0_g2: tensor<384xf32>, %b0_bt2: tensor<384xf32>, %b0_Wfc1: tensor<384x1536xf32>, %b0_bfc1: tensor<1536xf32>, %b0_Wfc2: tensor<1536x384xf32>, %b0_bfc2: tensor<384xf32>, %b1_g1: tensor<384xf32>, %b1_bt1: tensor<384xf32>, %b1_Wq: tensor<384x384xf32>, %b1_bq: tensor<384xf32>, %b1_Wk: tensor<384x384xf32>, %b1_bk: tensor<384xf32>, %b1_Wv: tensor<384x384xf32>, %b1_bv: tensor<384xf32>, %b1_Wo: tensor<384x384xf32>, %b1_bo: tensor<384xf32>, %b1_g2: tensor<384xf32>, %b1_bt2: tensor<384xf32>, %b1_Wfc1: tensor<384x1536xf32>, %b1_bfc1: tensor<1536xf32>, %b1_Wfc2: tensor<1536x384xf32>, %b1_bfc2: tensor<384xf32>, %b2_g1: tensor<384xf32>, %b2_bt1: tensor<384xf32>, %b2_Wq: tensor<384x384xf32>, %b2_bq: tensor<384xf32>, %b2_Wk: tensor<384x384xf32>, %b2_bk: tensor<384xf32>, %b2_Wv: tensor<384x384xf32>, %b2_bv: tensor<384xf32>, %b2_Wo: tensor<384x384xf32>, %b2_bo: tensor<384xf32>, %b2_g2: tensor<384xf32>, %b2_bt2: tensor<384xf32>, %b2_Wfc1: tensor<384x1536xf32>, %b2_bfc1: tensor<1536xf32>, %b2_Wfc2: tensor<1536x384xf32>, %b2_bfc2: tensor<384xf32>, %b3_g1: tensor<384xf32>, %b3_bt1: tensor<384xf32>, %b3_Wq: tensor<384x384xf32>, %b3_bq: tensor<384xf32>, %b3_Wk: tensor<384x384xf32>, %b3_bk: tensor<384xf32>, %b3_Wv: tensor<384x384xf32>, %b3_bv: tensor<384xf32>, %b3_Wo: tensor<384x384xf32>, %b3_bo: tensor<384xf32>, %b3_g2: tensor<384xf32>, %b3_bt2: tensor<384xf32>, %b3_Wfc1: tensor<384x1536xf32>, %b3_bfc1: tensor<1536xf32>, %b3_Wfc2: tensor<1536x384xf32>, %b3_bfc2: tensor<384xf32>, %b4_g1: tensor<384xf32>, %b4_bt1: tensor<384xf32>, %b4_Wq: tensor<384x384xf32>, %b4_bq: tensor<384xf32>, %b4_Wk: tensor<384x384xf32>, %b4_bk: tensor<384xf32>, %b4_Wv: tensor<384x384xf32>, %b4_bv: tensor<384xf32>, %b4_Wo: tensor<384x384xf32>, %b4_bo: tensor<384xf32>, %b4_g2: tensor<384xf32>, %b4_bt2: tensor<384xf32>, %b4_Wfc1: tensor<384x1536xf32>, %b4_bfc1: tensor<1536xf32>, %b4_Wfc2: tensor<1536x384xf32>, %b4_bfc2: tensor<384xf32>, %b5_g1: tensor<384xf32>, %b5_bt1: tensor<384xf32>, %b5_Wq: tensor<384x384xf32>, %b5_bq: tensor<384xf32>, %b5_Wk: tensor<384x384xf32>, %b5_bk: tensor<384xf32>, %b5_Wv: tensor<384x384xf32>, %b5_bv: tensor<384xf32>, %b5_Wo: tensor<384x384xf32>, %b5_bo: tensor<384xf32>, %b5_g2: tensor<384xf32>, %b5_bt2: tensor<384xf32>, %b5_Wfc1: tensor<384x1536xf32>, %b5_bfc1: tensor<1536xf32>, %b5_Wfc2: tensor<1536x384xf32>, %b5_bfc2: tensor<384xf32>, %b6_g1: tensor<384xf32>, %b6_bt1: tensor<384xf32>, %b6_Wq: tensor<384x384xf32>, %b6_bq: tensor<384xf32>, %b6_Wk: tensor<384x384xf32>, %b6_bk: tensor<384xf32>, %b6_Wv: tensor<384x384xf32>, %b6_bv: tensor<384xf32>, %b6_Wo: tensor<384x384xf32>, %b6_bo: tensor<384xf32>, %b6_g2: tensor<384xf32>, %b6_bt2: tensor<384xf32>, %b6_Wfc1: tensor<384x1536xf32>, %b6_bfc1: tensor<1536xf32>, %b6_Wfc2: tensor<1536x384xf32>, %b6_bfc2: tensor<384xf32>, %b7_g1: tensor<384xf32>, %b7_bt1: tensor<384xf32>, %b7_Wq: tensor<384x384xf32>, %b7_bq: tensor<384xf32>, %b7_Wk: tensor<384x384xf32>, %b7_bk: tensor<384xf32>, %b7_Wv: tensor<384x384xf32>, %b7_bv: tensor<384xf32>, %b7_Wo: tensor<384x384xf32>, %b7_bo: tensor<384xf32>, %b7_g2: tensor<384xf32>, %b7_bt2: tensor<384xf32>, %b7_Wfc1: tensor<384x1536xf32>, %b7_bfc1: tensor<1536xf32>, %b7_Wfc2: tensor<1536x384xf32>, %b7_bfc2: tensor<384xf32>, %b8_g1: tensor<384xf32>, %b8_bt1: tensor<384xf32>, %b8_Wq: tensor<384x384xf32>, %b8_bq: tensor<384xf32>, %b8_Wk: tensor<384x384xf32>, %b8_bk: tensor<384xf32>, %b8_Wv: tensor<384x384xf32>, %b8_bv: tensor<384xf32>, %b8_Wo: tensor<384x384xf32>, %b8_bo: tensor<384xf32>, %b8_g2: tensor<384xf32>, %b8_bt2: tensor<384xf32>, %b8_Wfc1: tensor<384x1536xf32>, %b8_bfc1: tensor<1536xf32>, %b8_Wfc2: tensor<1536x384xf32>, %b8_bfc2: tensor<384xf32>, %b9_g1: tensor<384xf32>, %b9_bt1: tensor<384xf32>, %b9_Wq: tensor<384x384xf32>, %b9_bq: tensor<384xf32>, %b9_Wk: tensor<384x384xf32>, %b9_bk: tensor<384xf32>, %b9_Wv: tensor<384x384xf32>, %b9_bv: tensor<384xf32>, %b9_Wo: tensor<384x384xf32>, %b9_bo: tensor<384xf32>, %b9_g2: tensor<384xf32>, %b9_bt2: tensor<384xf32>, %b9_Wfc1: tensor<384x1536xf32>, %b9_bfc1: tensor<1536xf32>, %b9_Wfc2: tensor<1536x384xf32>, %b9_bfc2: tensor<384xf32>, %b10_g1: tensor<384xf32>, %b10_bt1: tensor<384xf32>, %b10_Wq: tensor<384x384xf32>, %b10_bq: tensor<384xf32>, %b10_Wk: tensor<384x384xf32>, %b10_bk: tensor<384xf32>, %b10_Wv: tensor<384x384xf32>, %b10_bv: tensor<384xf32>, %b10_Wo: tensor<384x384xf32>, %b10_bo: tensor<384xf32>, %b10_g2: tensor<384xf32>, %b10_bt2: tensor<384xf32>, %b10_Wfc1: tensor<384x1536xf32>, %b10_bfc1: tensor<1536xf32>, %b10_Wfc2: tensor<1536x384xf32>, %b10_bfc2: tensor<384xf32>, %b11_g1: tensor<384xf32>, %b11_bt1: tensor<384xf32>, %b11_Wq: tensor<384x384xf32>, %b11_bq: tensor<384xf32>, %b11_Wk: tensor<384x384xf32>, %b11_bk: tensor<384xf32>, %b11_Wv: tensor<384x384xf32>, %b11_bv: tensor<384xf32>, %b11_Wo: tensor<384x384xf32>, %b11_bo: tensor<384xf32>, %b11_g2: tensor<384xf32>, %b11_bt2: tensor<384xf32>, %b11_Wfc1: tensor<384x1536xf32>, %b11_bfc1: tensor<1536xf32>, %b11_Wfc2: tensor<1536x384xf32>, %b11_bfc2: tensor<384xf32>, %gF: tensor<384xf32>, %btF: tensor<384xf32>, %Wc: tensor<384x1000xf32>, %bc: tensor<1000xf32>) -> tensor<256x1000xf32> {
    %one = stablehlo.constant dense<1.0> : tensor<f32>
    %zero = stablehlo.constant dense<0.0> : tensor<f32>
    %sc = stablehlo.constant dense<0.0> : tensor<f32>
    %v0 = stablehlo.reshape %x : (tensor<256x150528xf32>) -> tensor<256x3x224x224xf32>
    %v1 = stablehlo.convolution(%v0, %wConv)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [16, 16], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x3x224x224xf32>, tensor<384x3x16x16xf32>) -> tensor<256x384x14x14xf32>
    %v2 = stablehlo.broadcast_in_dim %bConv, dims = [1] : (tensor<384xf32>) -> tensor<256x384x14x14xf32>
    %v3 = stablehlo.add %v1, %v2 : tensor<256x384x14x14xf32>
    %v4 = stablehlo.transpose %v3, dims = [0, 2, 3, 1] : (tensor<256x384x14x14xf32>) -> tensor<256x14x14x384xf32>
    %v5 = stablehlo.reshape %v4 : (tensor<256x14x14x384xf32>) -> tensor<256x196x384xf32>
    %v6 = stablehlo.broadcast_in_dim %cls, dims = [2] : (tensor<384xf32>) -> tensor<256x1x384xf32>
    %v7 = stablehlo.concatenate %v6, %v5, dim = 1 : (tensor<256x1x384xf32>, tensor<256x196x384xf32>) -> tensor<256x197x384xf32>
    %v8 = stablehlo.broadcast_in_dim %pos, dims = [1, 2] : (tensor<197x384xf32>) -> tensor<256x197x384xf32>
    %v9 = stablehlo.add %v7, %v8 : tensor<256x197x384xf32>
    %v10 = stablehlo.reshape %v9 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v11 = stablehlo.reshape %v10 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v12 = stablehlo.constant dense<0.0> : tensor<f32>
    %v13 = stablehlo.constant dense<384.0> : tensor<256x197x384xf32>
    %v14 = stablehlo.constant dense<1.0e-6> : tensor<256x197x384xf32>
    %v15 = stablehlo.reduce(%v11 init: %v12) applies stablehlo.add across dimensions = [2] : (tensor<256x197x384xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v16 = stablehlo.broadcast_in_dim %v15, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x384xf32>
    %v17 = stablehlo.divide %v16, %v13 : tensor<256x197x384xf32>
    %v18 = stablehlo.subtract %v11, %v17 : tensor<256x197x384xf32>
    %v19 = stablehlo.multiply %v18, %v18 : tensor<256x197x384xf32>
    %v20 = stablehlo.reduce(%v19 init: %v12) applies stablehlo.add across dimensions = [2] : (tensor<256x197x384xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v21 = stablehlo.broadcast_in_dim %v20, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x384xf32>
    %v22 = stablehlo.divide %v21, %v13 : tensor<256x197x384xf32>
    %v23 = stablehlo.add %v22, %v14 : tensor<256x197x384xf32>
    %v24 = stablehlo.rsqrt %v23 : tensor<256x197x384xf32>
    %v25 = stablehlo.multiply %v18, %v24 : tensor<256x197x384xf32>
    %v26 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<256x197x384xf32>
    %v27 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<256x197x384xf32>
    %v28 = stablehlo.multiply %v25, %v26 : tensor<256x197x384xf32>
    %v29 = stablehlo.add %v28, %v27 : tensor<256x197x384xf32>
    %v30 = stablehlo.reshape %v29 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v31 = stablehlo.reshape %v30 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v32 = stablehlo.broadcast_in_dim %b0_g1, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v33 = stablehlo.multiply %v31, %v32 : tensor<256x197x384xf32>
    %v34 = stablehlo.reshape %v33 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v35 = stablehlo.reshape %v34 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v36 = stablehlo.broadcast_in_dim %b0_bt1, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v37 = stablehlo.add %v35, %v36 : tensor<256x197x384xf32>
    %v38 = stablehlo.reshape %v37 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v39 = stablehlo.reshape %v38 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v40 = stablehlo.dot_general %v39, %b0_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x384xf32>, tensor<384x384xf32>) -> tensor<256x197x384xf32>
    %v41 = stablehlo.broadcast_in_dim %b0_bq, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v42 = stablehlo.add %v40, %v41 : tensor<256x197x384xf32>
    %v43 = stablehlo.reshape %v42 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v44 = stablehlo.reshape %v38 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v45 = stablehlo.dot_general %v44, %b0_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x384xf32>, tensor<384x384xf32>) -> tensor<256x197x384xf32>
    %v46 = stablehlo.broadcast_in_dim %b0_bk, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v47 = stablehlo.add %v45, %v46 : tensor<256x197x384xf32>
    %v48 = stablehlo.reshape %v47 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v49 = stablehlo.reshape %v38 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v50 = stablehlo.dot_general %v49, %b0_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x384xf32>, tensor<384x384xf32>) -> tensor<256x197x384xf32>
    %v51 = stablehlo.broadcast_in_dim %b0_bv, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v52 = stablehlo.add %v50, %v51 : tensor<256x197x384xf32>
    %v53 = stablehlo.reshape %v52 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v54 = stablehlo.reshape %v43 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v55 = stablehlo.slice %v54 [0:256, 0:197, 0:64] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v56 = stablehlo.reshape %v55 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v57 = stablehlo.reshape %v48 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v58 = stablehlo.slice %v57 [0:256, 0:197, 0:64] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v59 = stablehlo.reshape %v58 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v60 = stablehlo.reshape %v53 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v61 = stablehlo.slice %v60 [0:256, 0:197, 0:64] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
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
    %v73 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v74 = stablehlo.reduce(%v72 init: %v73) applies stablehlo.maximum across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v75 = stablehlo.broadcast_in_dim %v74, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v76 = stablehlo.subtract %v72, %v75 : tensor<256x197x197xf32>
    %v77 = stablehlo.constant dense<0.0> : tensor<f32>
    %v78 = stablehlo.exponential %v76 : tensor<256x197x197xf32>
    %v79 = stablehlo.reduce(%v78 init: %v77) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v80 = stablehlo.broadcast_in_dim %v79, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v81 = stablehlo.divide %v78, %v80 : tensor<256x197x197xf32>
    %v82 = stablehlo.reshape %v81 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v83 = stablehlo.reshape %v82 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v84 = stablehlo.reshape %v62 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v85 = stablehlo.dot_general %v83, %v84, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v86 = stablehlo.reshape %v85 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v87 = stablehlo.reshape %v86 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v88 = stablehlo.constant dense<0.0> : tensor<f32>
    %v89 = stablehlo.pad %v87, %v88, low = [0, 0, 0], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x384xf32>
    %v90 = stablehlo.reshape %v89 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v91 = stablehlo.reshape %v43 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v92 = stablehlo.slice %v91 [0:256, 0:197, 64:128] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v93 = stablehlo.reshape %v92 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v94 = stablehlo.reshape %v48 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v95 = stablehlo.slice %v94 [0:256, 0:197, 64:128] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v96 = stablehlo.reshape %v95 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v97 = stablehlo.reshape %v53 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v98 = stablehlo.slice %v97 [0:256, 0:197, 64:128] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v99 = stablehlo.reshape %v98 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v100 = stablehlo.reshape %v96 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v101 = stablehlo.transpose %v100, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v102 = stablehlo.reshape %v101 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v103 = stablehlo.reshape %v93 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v104 = stablehlo.reshape %v102 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v105 = stablehlo.dot_general %v103, %v104, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v106 = stablehlo.reshape %v105 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v107 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v108 = stablehlo.multiply %v106, %v107 : tensor<256x38809xf32>
    %v109 = stablehlo.reshape %v108 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v110 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v111 = stablehlo.reduce(%v109 init: %v110) applies stablehlo.maximum across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v112 = stablehlo.broadcast_in_dim %v111, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v113 = stablehlo.subtract %v109, %v112 : tensor<256x197x197xf32>
    %v114 = stablehlo.constant dense<0.0> : tensor<f32>
    %v115 = stablehlo.exponential %v113 : tensor<256x197x197xf32>
    %v116 = stablehlo.reduce(%v115 init: %v114) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v117 = stablehlo.broadcast_in_dim %v116, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v118 = stablehlo.divide %v115, %v117 : tensor<256x197x197xf32>
    %v119 = stablehlo.reshape %v118 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v120 = stablehlo.reshape %v119 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v121 = stablehlo.reshape %v99 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v122 = stablehlo.dot_general %v120, %v121, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v123 = stablehlo.reshape %v122 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v124 = stablehlo.reshape %v123 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v125 = stablehlo.constant dense<0.0> : tensor<f32>
    %v126 = stablehlo.pad %v124, %v125, low = [0, 0, 64], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x384xf32>
    %v127 = stablehlo.reshape %v126 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v128 = stablehlo.add %v90, %v127 : tensor<256x75648xf32>
    %v129 = stablehlo.reshape %v43 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v130 = stablehlo.slice %v129 [0:256, 0:197, 128:192] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v131 = stablehlo.reshape %v130 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v132 = stablehlo.reshape %v48 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v133 = stablehlo.slice %v132 [0:256, 0:197, 128:192] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v134 = stablehlo.reshape %v133 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v135 = stablehlo.reshape %v53 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v136 = stablehlo.slice %v135 [0:256, 0:197, 128:192] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v137 = stablehlo.reshape %v136 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v138 = stablehlo.reshape %v134 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v139 = stablehlo.transpose %v138, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v140 = stablehlo.reshape %v139 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v141 = stablehlo.reshape %v131 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v142 = stablehlo.reshape %v140 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v143 = stablehlo.dot_general %v141, %v142, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v144 = stablehlo.reshape %v143 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v145 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v146 = stablehlo.multiply %v144, %v145 : tensor<256x38809xf32>
    %v147 = stablehlo.reshape %v146 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v148 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v149 = stablehlo.reduce(%v147 init: %v148) applies stablehlo.maximum across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v150 = stablehlo.broadcast_in_dim %v149, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v151 = stablehlo.subtract %v147, %v150 : tensor<256x197x197xf32>
    %v152 = stablehlo.constant dense<0.0> : tensor<f32>
    %v153 = stablehlo.exponential %v151 : tensor<256x197x197xf32>
    %v154 = stablehlo.reduce(%v153 init: %v152) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v155 = stablehlo.broadcast_in_dim %v154, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v156 = stablehlo.divide %v153, %v155 : tensor<256x197x197xf32>
    %v157 = stablehlo.reshape %v156 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v158 = stablehlo.reshape %v157 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v159 = stablehlo.reshape %v137 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v160 = stablehlo.dot_general %v158, %v159, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v161 = stablehlo.reshape %v160 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v162 = stablehlo.reshape %v161 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v163 = stablehlo.constant dense<0.0> : tensor<f32>
    %v164 = stablehlo.pad %v162, %v163, low = [0, 0, 128], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x384xf32>
    %v165 = stablehlo.reshape %v164 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v166 = stablehlo.add %v128, %v165 : tensor<256x75648xf32>
    %v167 = stablehlo.reshape %v43 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v168 = stablehlo.slice %v167 [0:256, 0:197, 192:256] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v169 = stablehlo.reshape %v168 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v170 = stablehlo.reshape %v48 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v171 = stablehlo.slice %v170 [0:256, 0:197, 192:256] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v172 = stablehlo.reshape %v171 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v173 = stablehlo.reshape %v53 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v174 = stablehlo.slice %v173 [0:256, 0:197, 192:256] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v175 = stablehlo.reshape %v174 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v176 = stablehlo.reshape %v172 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v177 = stablehlo.transpose %v176, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v178 = stablehlo.reshape %v177 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v179 = stablehlo.reshape %v169 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v180 = stablehlo.reshape %v178 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v181 = stablehlo.dot_general %v179, %v180, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v182 = stablehlo.reshape %v181 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v183 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v184 = stablehlo.multiply %v182, %v183 : tensor<256x38809xf32>
    %v185 = stablehlo.reshape %v184 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v186 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v187 = stablehlo.reduce(%v185 init: %v186) applies stablehlo.maximum across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v188 = stablehlo.broadcast_in_dim %v187, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v189 = stablehlo.subtract %v185, %v188 : tensor<256x197x197xf32>
    %v190 = stablehlo.constant dense<0.0> : tensor<f32>
    %v191 = stablehlo.exponential %v189 : tensor<256x197x197xf32>
    %v192 = stablehlo.reduce(%v191 init: %v190) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v193 = stablehlo.broadcast_in_dim %v192, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v194 = stablehlo.divide %v191, %v193 : tensor<256x197x197xf32>
    %v195 = stablehlo.reshape %v194 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v196 = stablehlo.reshape %v195 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v197 = stablehlo.reshape %v175 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v198 = stablehlo.dot_general %v196, %v197, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v199 = stablehlo.reshape %v198 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v200 = stablehlo.reshape %v199 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v201 = stablehlo.constant dense<0.0> : tensor<f32>
    %v202 = stablehlo.pad %v200, %v201, low = [0, 0, 192], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x384xf32>
    %v203 = stablehlo.reshape %v202 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v204 = stablehlo.add %v166, %v203 : tensor<256x75648xf32>
    %v205 = stablehlo.reshape %v43 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v206 = stablehlo.slice %v205 [0:256, 0:197, 256:320] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v207 = stablehlo.reshape %v206 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v208 = stablehlo.reshape %v48 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v209 = stablehlo.slice %v208 [0:256, 0:197, 256:320] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v210 = stablehlo.reshape %v209 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v211 = stablehlo.reshape %v53 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v212 = stablehlo.slice %v211 [0:256, 0:197, 256:320] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v213 = stablehlo.reshape %v212 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v214 = stablehlo.reshape %v210 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v215 = stablehlo.transpose %v214, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v216 = stablehlo.reshape %v215 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v217 = stablehlo.reshape %v207 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v218 = stablehlo.reshape %v216 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v219 = stablehlo.dot_general %v217, %v218, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v220 = stablehlo.reshape %v219 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v221 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v222 = stablehlo.multiply %v220, %v221 : tensor<256x38809xf32>
    %v223 = stablehlo.reshape %v222 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v224 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v225 = stablehlo.reduce(%v223 init: %v224) applies stablehlo.maximum across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v226 = stablehlo.broadcast_in_dim %v225, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v227 = stablehlo.subtract %v223, %v226 : tensor<256x197x197xf32>
    %v228 = stablehlo.constant dense<0.0> : tensor<f32>
    %v229 = stablehlo.exponential %v227 : tensor<256x197x197xf32>
    %v230 = stablehlo.reduce(%v229 init: %v228) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v231 = stablehlo.broadcast_in_dim %v230, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v232 = stablehlo.divide %v229, %v231 : tensor<256x197x197xf32>
    %v233 = stablehlo.reshape %v232 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v234 = stablehlo.reshape %v233 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v235 = stablehlo.reshape %v213 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v236 = stablehlo.dot_general %v234, %v235, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v237 = stablehlo.reshape %v236 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v238 = stablehlo.reshape %v237 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v239 = stablehlo.constant dense<0.0> : tensor<f32>
    %v240 = stablehlo.pad %v238, %v239, low = [0, 0, 256], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x384xf32>
    %v241 = stablehlo.reshape %v240 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v242 = stablehlo.add %v204, %v241 : tensor<256x75648xf32>
    %v243 = stablehlo.reshape %v43 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v244 = stablehlo.slice %v243 [0:256, 0:197, 320:384] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v245 = stablehlo.reshape %v244 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v246 = stablehlo.reshape %v48 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v247 = stablehlo.slice %v246 [0:256, 0:197, 320:384] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v248 = stablehlo.reshape %v247 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v249 = stablehlo.reshape %v53 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v250 = stablehlo.slice %v249 [0:256, 0:197, 320:384] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v251 = stablehlo.reshape %v250 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v252 = stablehlo.reshape %v248 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v253 = stablehlo.transpose %v252, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v254 = stablehlo.reshape %v253 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v255 = stablehlo.reshape %v245 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v256 = stablehlo.reshape %v254 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v257 = stablehlo.dot_general %v255, %v256, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v258 = stablehlo.reshape %v257 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v259 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v260 = stablehlo.multiply %v258, %v259 : tensor<256x38809xf32>
    %v261 = stablehlo.reshape %v260 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v262 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v263 = stablehlo.reduce(%v261 init: %v262) applies stablehlo.maximum across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v264 = stablehlo.broadcast_in_dim %v263, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v265 = stablehlo.subtract %v261, %v264 : tensor<256x197x197xf32>
    %v266 = stablehlo.constant dense<0.0> : tensor<f32>
    %v267 = stablehlo.exponential %v265 : tensor<256x197x197xf32>
    %v268 = stablehlo.reduce(%v267 init: %v266) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v269 = stablehlo.broadcast_in_dim %v268, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v270 = stablehlo.divide %v267, %v269 : tensor<256x197x197xf32>
    %v271 = stablehlo.reshape %v270 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v272 = stablehlo.reshape %v271 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v273 = stablehlo.reshape %v251 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v274 = stablehlo.dot_general %v272, %v273, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v275 = stablehlo.reshape %v274 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v276 = stablehlo.reshape %v275 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v277 = stablehlo.constant dense<0.0> : tensor<f32>
    %v278 = stablehlo.pad %v276, %v277, low = [0, 0, 320], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x384xf32>
    %v279 = stablehlo.reshape %v278 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v280 = stablehlo.add %v242, %v279 : tensor<256x75648xf32>
    %v281 = stablehlo.reshape %v280 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v282 = stablehlo.dot_general %v281, %b0_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x384xf32>, tensor<384x384xf32>) -> tensor<256x197x384xf32>
    %v283 = stablehlo.broadcast_in_dim %b0_bo, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v284 = stablehlo.add %v282, %v283 : tensor<256x197x384xf32>
    %v285 = stablehlo.reshape %v284 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v286 = stablehlo.add %v10, %v285 : tensor<256x75648xf32>
    %v287 = stablehlo.reshape %v286 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v288 = stablehlo.constant dense<0.0> : tensor<f32>
    %v289 = stablehlo.constant dense<384.0> : tensor<256x197x384xf32>
    %v290 = stablehlo.constant dense<1.0e-6> : tensor<256x197x384xf32>
    %v291 = stablehlo.reduce(%v287 init: %v288) applies stablehlo.add across dimensions = [2] : (tensor<256x197x384xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v292 = stablehlo.broadcast_in_dim %v291, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x384xf32>
    %v293 = stablehlo.divide %v292, %v289 : tensor<256x197x384xf32>
    %v294 = stablehlo.subtract %v287, %v293 : tensor<256x197x384xf32>
    %v295 = stablehlo.multiply %v294, %v294 : tensor<256x197x384xf32>
    %v296 = stablehlo.reduce(%v295 init: %v288) applies stablehlo.add across dimensions = [2] : (tensor<256x197x384xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v297 = stablehlo.broadcast_in_dim %v296, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x384xf32>
    %v298 = stablehlo.divide %v297, %v289 : tensor<256x197x384xf32>
    %v299 = stablehlo.add %v298, %v290 : tensor<256x197x384xf32>
    %v300 = stablehlo.rsqrt %v299 : tensor<256x197x384xf32>
    %v301 = stablehlo.multiply %v294, %v300 : tensor<256x197x384xf32>
    %v302 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<256x197x384xf32>
    %v303 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<256x197x384xf32>
    %v304 = stablehlo.multiply %v301, %v302 : tensor<256x197x384xf32>
    %v305 = stablehlo.add %v304, %v303 : tensor<256x197x384xf32>
    %v306 = stablehlo.reshape %v305 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v307 = stablehlo.reshape %v306 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v308 = stablehlo.broadcast_in_dim %b0_g2, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v309 = stablehlo.multiply %v307, %v308 : tensor<256x197x384xf32>
    %v310 = stablehlo.reshape %v309 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v311 = stablehlo.reshape %v310 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v312 = stablehlo.broadcast_in_dim %b0_bt2, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v313 = stablehlo.add %v311, %v312 : tensor<256x197x384xf32>
    %v314 = stablehlo.reshape %v313 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v315 = stablehlo.reshape %v314 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v316 = stablehlo.dot_general %v315, %b0_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x384xf32>, tensor<384x1536xf32>) -> tensor<256x197x1536xf32>
    %v317 = stablehlo.broadcast_in_dim %b0_bfc1, dims = [2] : (tensor<1536xf32>) -> tensor<256x197x1536xf32>
    %v318 = stablehlo.add %v316, %v317 : tensor<256x197x1536xf32>
    %v319 = stablehlo.reshape %v318 : (tensor<256x197x1536xf32>) -> tensor<256x302592xf32>
    %v320 = stablehlo.constant dense<0.5> : tensor<256x302592xf32>
    %v321 = stablehlo.multiply %v320, %v319 : tensor<256x302592xf32>
    %v322 = stablehlo.negate %v319 : tensor<256x302592xf32>
    %v323 = stablehlo.constant dense<0.7071067811865476> : tensor<256x302592xf32>
    %v324 = stablehlo.multiply %v322, %v323 : tensor<256x302592xf32>
    %v325 = chlo.erfc %v324 : tensor<256x302592xf32> -> tensor<256x302592xf32>
    %v326 = stablehlo.multiply %v321, %v325 : tensor<256x302592xf32>
    %v327 = stablehlo.reshape %v326 : (tensor<256x302592xf32>) -> tensor<256x197x1536xf32>
    %v328 = stablehlo.dot_general %v327, %b0_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x1536xf32>, tensor<1536x384xf32>) -> tensor<256x197x384xf32>
    %v329 = stablehlo.broadcast_in_dim %b0_bfc2, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v330 = stablehlo.add %v328, %v329 : tensor<256x197x384xf32>
    %v331 = stablehlo.reshape %v330 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v332 = stablehlo.add %v286, %v331 : tensor<256x75648xf32>
    %v333 = stablehlo.reshape %v332 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v334 = stablehlo.constant dense<0.0> : tensor<f32>
    %v335 = stablehlo.constant dense<384.0> : tensor<256x197x384xf32>
    %v336 = stablehlo.constant dense<1.0e-6> : tensor<256x197x384xf32>
    %v337 = stablehlo.reduce(%v333 init: %v334) applies stablehlo.add across dimensions = [2] : (tensor<256x197x384xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v338 = stablehlo.broadcast_in_dim %v337, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x384xf32>
    %v339 = stablehlo.divide %v338, %v335 : tensor<256x197x384xf32>
    %v340 = stablehlo.subtract %v333, %v339 : tensor<256x197x384xf32>
    %v341 = stablehlo.multiply %v340, %v340 : tensor<256x197x384xf32>
    %v342 = stablehlo.reduce(%v341 init: %v334) applies stablehlo.add across dimensions = [2] : (tensor<256x197x384xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v343 = stablehlo.broadcast_in_dim %v342, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x384xf32>
    %v344 = stablehlo.divide %v343, %v335 : tensor<256x197x384xf32>
    %v345 = stablehlo.add %v344, %v336 : tensor<256x197x384xf32>
    %v346 = stablehlo.rsqrt %v345 : tensor<256x197x384xf32>
    %v347 = stablehlo.multiply %v340, %v346 : tensor<256x197x384xf32>
    %v348 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<256x197x384xf32>
    %v349 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<256x197x384xf32>
    %v350 = stablehlo.multiply %v347, %v348 : tensor<256x197x384xf32>
    %v351 = stablehlo.add %v350, %v349 : tensor<256x197x384xf32>
    %v352 = stablehlo.reshape %v351 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v353 = stablehlo.reshape %v352 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v354 = stablehlo.broadcast_in_dim %b1_g1, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v355 = stablehlo.multiply %v353, %v354 : tensor<256x197x384xf32>
    %v356 = stablehlo.reshape %v355 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v357 = stablehlo.reshape %v356 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v358 = stablehlo.broadcast_in_dim %b1_bt1, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v359 = stablehlo.add %v357, %v358 : tensor<256x197x384xf32>
    %v360 = stablehlo.reshape %v359 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v361 = stablehlo.reshape %v360 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v362 = stablehlo.dot_general %v361, %b1_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x384xf32>, tensor<384x384xf32>) -> tensor<256x197x384xf32>
    %v363 = stablehlo.broadcast_in_dim %b1_bq, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v364 = stablehlo.add %v362, %v363 : tensor<256x197x384xf32>
    %v365 = stablehlo.reshape %v364 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v366 = stablehlo.reshape %v360 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v367 = stablehlo.dot_general %v366, %b1_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x384xf32>, tensor<384x384xf32>) -> tensor<256x197x384xf32>
    %v368 = stablehlo.broadcast_in_dim %b1_bk, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v369 = stablehlo.add %v367, %v368 : tensor<256x197x384xf32>
    %v370 = stablehlo.reshape %v369 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v371 = stablehlo.reshape %v360 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v372 = stablehlo.dot_general %v371, %b1_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x384xf32>, tensor<384x384xf32>) -> tensor<256x197x384xf32>
    %v373 = stablehlo.broadcast_in_dim %b1_bv, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v374 = stablehlo.add %v372, %v373 : tensor<256x197x384xf32>
    %v375 = stablehlo.reshape %v374 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v376 = stablehlo.reshape %v365 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v377 = stablehlo.slice %v376 [0:256, 0:197, 0:64] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v378 = stablehlo.reshape %v377 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v379 = stablehlo.reshape %v370 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v380 = stablehlo.slice %v379 [0:256, 0:197, 0:64] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v381 = stablehlo.reshape %v380 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v382 = stablehlo.reshape %v375 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v383 = stablehlo.slice %v382 [0:256, 0:197, 0:64] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v384 = stablehlo.reshape %v383 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v385 = stablehlo.reshape %v381 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v386 = stablehlo.transpose %v385, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v387 = stablehlo.reshape %v386 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v388 = stablehlo.reshape %v378 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v389 = stablehlo.reshape %v387 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v390 = stablehlo.dot_general %v388, %v389, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v391 = stablehlo.reshape %v390 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v392 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v393 = stablehlo.multiply %v391, %v392 : tensor<256x38809xf32>
    %v394 = stablehlo.reshape %v393 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v395 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v396 = stablehlo.reduce(%v394 init: %v395) applies stablehlo.maximum across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v397 = stablehlo.broadcast_in_dim %v396, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v398 = stablehlo.subtract %v394, %v397 : tensor<256x197x197xf32>
    %v399 = stablehlo.constant dense<0.0> : tensor<f32>
    %v400 = stablehlo.exponential %v398 : tensor<256x197x197xf32>
    %v401 = stablehlo.reduce(%v400 init: %v399) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v402 = stablehlo.broadcast_in_dim %v401, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v403 = stablehlo.divide %v400, %v402 : tensor<256x197x197xf32>
    %v404 = stablehlo.reshape %v403 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v405 = stablehlo.reshape %v404 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v406 = stablehlo.reshape %v384 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v407 = stablehlo.dot_general %v405, %v406, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v408 = stablehlo.reshape %v407 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v409 = stablehlo.reshape %v408 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v410 = stablehlo.constant dense<0.0> : tensor<f32>
    %v411 = stablehlo.pad %v409, %v410, low = [0, 0, 0], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x384xf32>
    %v412 = stablehlo.reshape %v411 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v413 = stablehlo.reshape %v365 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v414 = stablehlo.slice %v413 [0:256, 0:197, 64:128] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v415 = stablehlo.reshape %v414 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v416 = stablehlo.reshape %v370 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v417 = stablehlo.slice %v416 [0:256, 0:197, 64:128] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v418 = stablehlo.reshape %v417 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v419 = stablehlo.reshape %v375 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v420 = stablehlo.slice %v419 [0:256, 0:197, 64:128] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v421 = stablehlo.reshape %v420 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v422 = stablehlo.reshape %v418 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v423 = stablehlo.transpose %v422, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v424 = stablehlo.reshape %v423 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v425 = stablehlo.reshape %v415 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v426 = stablehlo.reshape %v424 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v427 = stablehlo.dot_general %v425, %v426, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v428 = stablehlo.reshape %v427 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v429 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v430 = stablehlo.multiply %v428, %v429 : tensor<256x38809xf32>
    %v431 = stablehlo.reshape %v430 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v432 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v433 = stablehlo.reduce(%v431 init: %v432) applies stablehlo.maximum across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v434 = stablehlo.broadcast_in_dim %v433, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v435 = stablehlo.subtract %v431, %v434 : tensor<256x197x197xf32>
    %v436 = stablehlo.constant dense<0.0> : tensor<f32>
    %v437 = stablehlo.exponential %v435 : tensor<256x197x197xf32>
    %v438 = stablehlo.reduce(%v437 init: %v436) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v439 = stablehlo.broadcast_in_dim %v438, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v440 = stablehlo.divide %v437, %v439 : tensor<256x197x197xf32>
    %v441 = stablehlo.reshape %v440 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v442 = stablehlo.reshape %v441 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v443 = stablehlo.reshape %v421 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v444 = stablehlo.dot_general %v442, %v443, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v445 = stablehlo.reshape %v444 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v446 = stablehlo.reshape %v445 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v447 = stablehlo.constant dense<0.0> : tensor<f32>
    %v448 = stablehlo.pad %v446, %v447, low = [0, 0, 64], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x384xf32>
    %v449 = stablehlo.reshape %v448 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v450 = stablehlo.add %v412, %v449 : tensor<256x75648xf32>
    %v451 = stablehlo.reshape %v365 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v452 = stablehlo.slice %v451 [0:256, 0:197, 128:192] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v453 = stablehlo.reshape %v452 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v454 = stablehlo.reshape %v370 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v455 = stablehlo.slice %v454 [0:256, 0:197, 128:192] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v456 = stablehlo.reshape %v455 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v457 = stablehlo.reshape %v375 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v458 = stablehlo.slice %v457 [0:256, 0:197, 128:192] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v459 = stablehlo.reshape %v458 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v460 = stablehlo.reshape %v456 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v461 = stablehlo.transpose %v460, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v462 = stablehlo.reshape %v461 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v463 = stablehlo.reshape %v453 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v464 = stablehlo.reshape %v462 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v465 = stablehlo.dot_general %v463, %v464, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v466 = stablehlo.reshape %v465 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v467 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v468 = stablehlo.multiply %v466, %v467 : tensor<256x38809xf32>
    %v469 = stablehlo.reshape %v468 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v470 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v471 = stablehlo.reduce(%v469 init: %v470) applies stablehlo.maximum across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v472 = stablehlo.broadcast_in_dim %v471, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v473 = stablehlo.subtract %v469, %v472 : tensor<256x197x197xf32>
    %v474 = stablehlo.constant dense<0.0> : tensor<f32>
    %v475 = stablehlo.exponential %v473 : tensor<256x197x197xf32>
    %v476 = stablehlo.reduce(%v475 init: %v474) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v477 = stablehlo.broadcast_in_dim %v476, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v478 = stablehlo.divide %v475, %v477 : tensor<256x197x197xf32>
    %v479 = stablehlo.reshape %v478 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v480 = stablehlo.reshape %v479 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v481 = stablehlo.reshape %v459 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v482 = stablehlo.dot_general %v480, %v481, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v483 = stablehlo.reshape %v482 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v484 = stablehlo.reshape %v483 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v485 = stablehlo.constant dense<0.0> : tensor<f32>
    %v486 = stablehlo.pad %v484, %v485, low = [0, 0, 128], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x384xf32>
    %v487 = stablehlo.reshape %v486 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v488 = stablehlo.add %v450, %v487 : tensor<256x75648xf32>
    %v489 = stablehlo.reshape %v365 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v490 = stablehlo.slice %v489 [0:256, 0:197, 192:256] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v491 = stablehlo.reshape %v490 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v492 = stablehlo.reshape %v370 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v493 = stablehlo.slice %v492 [0:256, 0:197, 192:256] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v494 = stablehlo.reshape %v493 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v495 = stablehlo.reshape %v375 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v496 = stablehlo.slice %v495 [0:256, 0:197, 192:256] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v497 = stablehlo.reshape %v496 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v498 = stablehlo.reshape %v494 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v499 = stablehlo.transpose %v498, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v500 = stablehlo.reshape %v499 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v501 = stablehlo.reshape %v491 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v502 = stablehlo.reshape %v500 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v503 = stablehlo.dot_general %v501, %v502, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v504 = stablehlo.reshape %v503 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v505 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v506 = stablehlo.multiply %v504, %v505 : tensor<256x38809xf32>
    %v507 = stablehlo.reshape %v506 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v508 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v509 = stablehlo.reduce(%v507 init: %v508) applies stablehlo.maximum across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v510 = stablehlo.broadcast_in_dim %v509, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v511 = stablehlo.subtract %v507, %v510 : tensor<256x197x197xf32>
    %v512 = stablehlo.constant dense<0.0> : tensor<f32>
    %v513 = stablehlo.exponential %v511 : tensor<256x197x197xf32>
    %v514 = stablehlo.reduce(%v513 init: %v512) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v515 = stablehlo.broadcast_in_dim %v514, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v516 = stablehlo.divide %v513, %v515 : tensor<256x197x197xf32>
    %v517 = stablehlo.reshape %v516 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v518 = stablehlo.reshape %v517 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v519 = stablehlo.reshape %v497 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v520 = stablehlo.dot_general %v518, %v519, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v521 = stablehlo.reshape %v520 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v522 = stablehlo.reshape %v521 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v523 = stablehlo.constant dense<0.0> : tensor<f32>
    %v524 = stablehlo.pad %v522, %v523, low = [0, 0, 192], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x384xf32>
    %v525 = stablehlo.reshape %v524 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v526 = stablehlo.add %v488, %v525 : tensor<256x75648xf32>
    %v527 = stablehlo.reshape %v365 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v528 = stablehlo.slice %v527 [0:256, 0:197, 256:320] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v529 = stablehlo.reshape %v528 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v530 = stablehlo.reshape %v370 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v531 = stablehlo.slice %v530 [0:256, 0:197, 256:320] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v532 = stablehlo.reshape %v531 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v533 = stablehlo.reshape %v375 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v534 = stablehlo.slice %v533 [0:256, 0:197, 256:320] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v535 = stablehlo.reshape %v534 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v536 = stablehlo.reshape %v532 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v537 = stablehlo.transpose %v536, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v538 = stablehlo.reshape %v537 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v539 = stablehlo.reshape %v529 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v540 = stablehlo.reshape %v538 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v541 = stablehlo.dot_general %v539, %v540, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v542 = stablehlo.reshape %v541 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v543 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v544 = stablehlo.multiply %v542, %v543 : tensor<256x38809xf32>
    %v545 = stablehlo.reshape %v544 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v546 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v547 = stablehlo.reduce(%v545 init: %v546) applies stablehlo.maximum across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v548 = stablehlo.broadcast_in_dim %v547, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v549 = stablehlo.subtract %v545, %v548 : tensor<256x197x197xf32>
    %v550 = stablehlo.constant dense<0.0> : tensor<f32>
    %v551 = stablehlo.exponential %v549 : tensor<256x197x197xf32>
    %v552 = stablehlo.reduce(%v551 init: %v550) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v553 = stablehlo.broadcast_in_dim %v552, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v554 = stablehlo.divide %v551, %v553 : tensor<256x197x197xf32>
    %v555 = stablehlo.reshape %v554 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v556 = stablehlo.reshape %v555 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v557 = stablehlo.reshape %v535 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v558 = stablehlo.dot_general %v556, %v557, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v559 = stablehlo.reshape %v558 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v560 = stablehlo.reshape %v559 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v561 = stablehlo.constant dense<0.0> : tensor<f32>
    %v562 = stablehlo.pad %v560, %v561, low = [0, 0, 256], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x384xf32>
    %v563 = stablehlo.reshape %v562 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v564 = stablehlo.add %v526, %v563 : tensor<256x75648xf32>
    %v565 = stablehlo.reshape %v365 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v566 = stablehlo.slice %v565 [0:256, 0:197, 320:384] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v567 = stablehlo.reshape %v566 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v568 = stablehlo.reshape %v370 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v569 = stablehlo.slice %v568 [0:256, 0:197, 320:384] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v570 = stablehlo.reshape %v569 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v571 = stablehlo.reshape %v375 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v572 = stablehlo.slice %v571 [0:256, 0:197, 320:384] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v573 = stablehlo.reshape %v572 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v574 = stablehlo.reshape %v570 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v575 = stablehlo.transpose %v574, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v576 = stablehlo.reshape %v575 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v577 = stablehlo.reshape %v567 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v578 = stablehlo.reshape %v576 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v579 = stablehlo.dot_general %v577, %v578, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v580 = stablehlo.reshape %v579 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v581 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v582 = stablehlo.multiply %v580, %v581 : tensor<256x38809xf32>
    %v583 = stablehlo.reshape %v582 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v584 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v585 = stablehlo.reduce(%v583 init: %v584) applies stablehlo.maximum across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v586 = stablehlo.broadcast_in_dim %v585, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v587 = stablehlo.subtract %v583, %v586 : tensor<256x197x197xf32>
    %v588 = stablehlo.constant dense<0.0> : tensor<f32>
    %v589 = stablehlo.exponential %v587 : tensor<256x197x197xf32>
    %v590 = stablehlo.reduce(%v589 init: %v588) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v591 = stablehlo.broadcast_in_dim %v590, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v592 = stablehlo.divide %v589, %v591 : tensor<256x197x197xf32>
    %v593 = stablehlo.reshape %v592 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v594 = stablehlo.reshape %v593 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v595 = stablehlo.reshape %v573 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v596 = stablehlo.dot_general %v594, %v595, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v597 = stablehlo.reshape %v596 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v598 = stablehlo.reshape %v597 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v599 = stablehlo.constant dense<0.0> : tensor<f32>
    %v600 = stablehlo.pad %v598, %v599, low = [0, 0, 320], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x384xf32>
    %v601 = stablehlo.reshape %v600 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v602 = stablehlo.add %v564, %v601 : tensor<256x75648xf32>
    %v603 = stablehlo.reshape %v602 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v604 = stablehlo.dot_general %v603, %b1_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x384xf32>, tensor<384x384xf32>) -> tensor<256x197x384xf32>
    %v605 = stablehlo.broadcast_in_dim %b1_bo, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v606 = stablehlo.add %v604, %v605 : tensor<256x197x384xf32>
    %v607 = stablehlo.reshape %v606 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v608 = stablehlo.add %v332, %v607 : tensor<256x75648xf32>
    %v609 = stablehlo.reshape %v608 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v610 = stablehlo.constant dense<0.0> : tensor<f32>
    %v611 = stablehlo.constant dense<384.0> : tensor<256x197x384xf32>
    %v612 = stablehlo.constant dense<1.0e-6> : tensor<256x197x384xf32>
    %v613 = stablehlo.reduce(%v609 init: %v610) applies stablehlo.add across dimensions = [2] : (tensor<256x197x384xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v614 = stablehlo.broadcast_in_dim %v613, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x384xf32>
    %v615 = stablehlo.divide %v614, %v611 : tensor<256x197x384xf32>
    %v616 = stablehlo.subtract %v609, %v615 : tensor<256x197x384xf32>
    %v617 = stablehlo.multiply %v616, %v616 : tensor<256x197x384xf32>
    %v618 = stablehlo.reduce(%v617 init: %v610) applies stablehlo.add across dimensions = [2] : (tensor<256x197x384xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v619 = stablehlo.broadcast_in_dim %v618, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x384xf32>
    %v620 = stablehlo.divide %v619, %v611 : tensor<256x197x384xf32>
    %v621 = stablehlo.add %v620, %v612 : tensor<256x197x384xf32>
    %v622 = stablehlo.rsqrt %v621 : tensor<256x197x384xf32>
    %v623 = stablehlo.multiply %v616, %v622 : tensor<256x197x384xf32>
    %v624 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<256x197x384xf32>
    %v625 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<256x197x384xf32>
    %v626 = stablehlo.multiply %v623, %v624 : tensor<256x197x384xf32>
    %v627 = stablehlo.add %v626, %v625 : tensor<256x197x384xf32>
    %v628 = stablehlo.reshape %v627 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v629 = stablehlo.reshape %v628 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v630 = stablehlo.broadcast_in_dim %b1_g2, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v631 = stablehlo.multiply %v629, %v630 : tensor<256x197x384xf32>
    %v632 = stablehlo.reshape %v631 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v633 = stablehlo.reshape %v632 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v634 = stablehlo.broadcast_in_dim %b1_bt2, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v635 = stablehlo.add %v633, %v634 : tensor<256x197x384xf32>
    %v636 = stablehlo.reshape %v635 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v637 = stablehlo.reshape %v636 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v638 = stablehlo.dot_general %v637, %b1_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x384xf32>, tensor<384x1536xf32>) -> tensor<256x197x1536xf32>
    %v639 = stablehlo.broadcast_in_dim %b1_bfc1, dims = [2] : (tensor<1536xf32>) -> tensor<256x197x1536xf32>
    %v640 = stablehlo.add %v638, %v639 : tensor<256x197x1536xf32>
    %v641 = stablehlo.reshape %v640 : (tensor<256x197x1536xf32>) -> tensor<256x302592xf32>
    %v642 = stablehlo.constant dense<0.5> : tensor<256x302592xf32>
    %v643 = stablehlo.multiply %v642, %v641 : tensor<256x302592xf32>
    %v644 = stablehlo.negate %v641 : tensor<256x302592xf32>
    %v645 = stablehlo.constant dense<0.7071067811865476> : tensor<256x302592xf32>
    %v646 = stablehlo.multiply %v644, %v645 : tensor<256x302592xf32>
    %v647 = chlo.erfc %v646 : tensor<256x302592xf32> -> tensor<256x302592xf32>
    %v648 = stablehlo.multiply %v643, %v647 : tensor<256x302592xf32>
    %v649 = stablehlo.reshape %v648 : (tensor<256x302592xf32>) -> tensor<256x197x1536xf32>
    %v650 = stablehlo.dot_general %v649, %b1_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x1536xf32>, tensor<1536x384xf32>) -> tensor<256x197x384xf32>
    %v651 = stablehlo.broadcast_in_dim %b1_bfc2, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v652 = stablehlo.add %v650, %v651 : tensor<256x197x384xf32>
    %v653 = stablehlo.reshape %v652 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v654 = stablehlo.add %v608, %v653 : tensor<256x75648xf32>
    %v655 = stablehlo.reshape %v654 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v656 = stablehlo.constant dense<0.0> : tensor<f32>
    %v657 = stablehlo.constant dense<384.0> : tensor<256x197x384xf32>
    %v658 = stablehlo.constant dense<1.0e-6> : tensor<256x197x384xf32>
    %v659 = stablehlo.reduce(%v655 init: %v656) applies stablehlo.add across dimensions = [2] : (tensor<256x197x384xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v660 = stablehlo.broadcast_in_dim %v659, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x384xf32>
    %v661 = stablehlo.divide %v660, %v657 : tensor<256x197x384xf32>
    %v662 = stablehlo.subtract %v655, %v661 : tensor<256x197x384xf32>
    %v663 = stablehlo.multiply %v662, %v662 : tensor<256x197x384xf32>
    %v664 = stablehlo.reduce(%v663 init: %v656) applies stablehlo.add across dimensions = [2] : (tensor<256x197x384xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v665 = stablehlo.broadcast_in_dim %v664, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x384xf32>
    %v666 = stablehlo.divide %v665, %v657 : tensor<256x197x384xf32>
    %v667 = stablehlo.add %v666, %v658 : tensor<256x197x384xf32>
    %v668 = stablehlo.rsqrt %v667 : tensor<256x197x384xf32>
    %v669 = stablehlo.multiply %v662, %v668 : tensor<256x197x384xf32>
    %v670 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<256x197x384xf32>
    %v671 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<256x197x384xf32>
    %v672 = stablehlo.multiply %v669, %v670 : tensor<256x197x384xf32>
    %v673 = stablehlo.add %v672, %v671 : tensor<256x197x384xf32>
    %v674 = stablehlo.reshape %v673 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v675 = stablehlo.reshape %v674 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v676 = stablehlo.broadcast_in_dim %b2_g1, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v677 = stablehlo.multiply %v675, %v676 : tensor<256x197x384xf32>
    %v678 = stablehlo.reshape %v677 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v679 = stablehlo.reshape %v678 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v680 = stablehlo.broadcast_in_dim %b2_bt1, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v681 = stablehlo.add %v679, %v680 : tensor<256x197x384xf32>
    %v682 = stablehlo.reshape %v681 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v683 = stablehlo.reshape %v682 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v684 = stablehlo.dot_general %v683, %b2_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x384xf32>, tensor<384x384xf32>) -> tensor<256x197x384xf32>
    %v685 = stablehlo.broadcast_in_dim %b2_bq, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v686 = stablehlo.add %v684, %v685 : tensor<256x197x384xf32>
    %v687 = stablehlo.reshape %v686 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v688 = stablehlo.reshape %v682 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v689 = stablehlo.dot_general %v688, %b2_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x384xf32>, tensor<384x384xf32>) -> tensor<256x197x384xf32>
    %v690 = stablehlo.broadcast_in_dim %b2_bk, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v691 = stablehlo.add %v689, %v690 : tensor<256x197x384xf32>
    %v692 = stablehlo.reshape %v691 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v693 = stablehlo.reshape %v682 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v694 = stablehlo.dot_general %v693, %b2_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x384xf32>, tensor<384x384xf32>) -> tensor<256x197x384xf32>
    %v695 = stablehlo.broadcast_in_dim %b2_bv, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v696 = stablehlo.add %v694, %v695 : tensor<256x197x384xf32>
    %v697 = stablehlo.reshape %v696 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v698 = stablehlo.reshape %v687 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v699 = stablehlo.slice %v698 [0:256, 0:197, 0:64] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v700 = stablehlo.reshape %v699 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v701 = stablehlo.reshape %v692 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v702 = stablehlo.slice %v701 [0:256, 0:197, 0:64] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v703 = stablehlo.reshape %v702 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v704 = stablehlo.reshape %v697 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v705 = stablehlo.slice %v704 [0:256, 0:197, 0:64] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v706 = stablehlo.reshape %v705 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v707 = stablehlo.reshape %v703 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v708 = stablehlo.transpose %v707, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v709 = stablehlo.reshape %v708 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v710 = stablehlo.reshape %v700 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v711 = stablehlo.reshape %v709 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v712 = stablehlo.dot_general %v710, %v711, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v713 = stablehlo.reshape %v712 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v714 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v715 = stablehlo.multiply %v713, %v714 : tensor<256x38809xf32>
    %v716 = stablehlo.reshape %v715 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v717 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v718 = stablehlo.reduce(%v716 init: %v717) applies stablehlo.maximum across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v719 = stablehlo.broadcast_in_dim %v718, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v720 = stablehlo.subtract %v716, %v719 : tensor<256x197x197xf32>
    %v721 = stablehlo.constant dense<0.0> : tensor<f32>
    %v722 = stablehlo.exponential %v720 : tensor<256x197x197xf32>
    %v723 = stablehlo.reduce(%v722 init: %v721) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v724 = stablehlo.broadcast_in_dim %v723, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v725 = stablehlo.divide %v722, %v724 : tensor<256x197x197xf32>
    %v726 = stablehlo.reshape %v725 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v727 = stablehlo.reshape %v726 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v728 = stablehlo.reshape %v706 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v729 = stablehlo.dot_general %v727, %v728, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v730 = stablehlo.reshape %v729 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v731 = stablehlo.reshape %v730 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v732 = stablehlo.constant dense<0.0> : tensor<f32>
    %v733 = stablehlo.pad %v731, %v732, low = [0, 0, 0], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x384xf32>
    %v734 = stablehlo.reshape %v733 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v735 = stablehlo.reshape %v687 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v736 = stablehlo.slice %v735 [0:256, 0:197, 64:128] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v737 = stablehlo.reshape %v736 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v738 = stablehlo.reshape %v692 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v739 = stablehlo.slice %v738 [0:256, 0:197, 64:128] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v740 = stablehlo.reshape %v739 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v741 = stablehlo.reshape %v697 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v742 = stablehlo.slice %v741 [0:256, 0:197, 64:128] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v743 = stablehlo.reshape %v742 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v744 = stablehlo.reshape %v740 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v745 = stablehlo.transpose %v744, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v746 = stablehlo.reshape %v745 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v747 = stablehlo.reshape %v737 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v748 = stablehlo.reshape %v746 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v749 = stablehlo.dot_general %v747, %v748, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v750 = stablehlo.reshape %v749 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v751 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v752 = stablehlo.multiply %v750, %v751 : tensor<256x38809xf32>
    %v753 = stablehlo.reshape %v752 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v754 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v755 = stablehlo.reduce(%v753 init: %v754) applies stablehlo.maximum across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v756 = stablehlo.broadcast_in_dim %v755, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v757 = stablehlo.subtract %v753, %v756 : tensor<256x197x197xf32>
    %v758 = stablehlo.constant dense<0.0> : tensor<f32>
    %v759 = stablehlo.exponential %v757 : tensor<256x197x197xf32>
    %v760 = stablehlo.reduce(%v759 init: %v758) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v761 = stablehlo.broadcast_in_dim %v760, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v762 = stablehlo.divide %v759, %v761 : tensor<256x197x197xf32>
    %v763 = stablehlo.reshape %v762 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v764 = stablehlo.reshape %v763 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v765 = stablehlo.reshape %v743 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v766 = stablehlo.dot_general %v764, %v765, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v767 = stablehlo.reshape %v766 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v768 = stablehlo.reshape %v767 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v769 = stablehlo.constant dense<0.0> : tensor<f32>
    %v770 = stablehlo.pad %v768, %v769, low = [0, 0, 64], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x384xf32>
    %v771 = stablehlo.reshape %v770 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v772 = stablehlo.add %v734, %v771 : tensor<256x75648xf32>
    %v773 = stablehlo.reshape %v687 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v774 = stablehlo.slice %v773 [0:256, 0:197, 128:192] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v775 = stablehlo.reshape %v774 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v776 = stablehlo.reshape %v692 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v777 = stablehlo.slice %v776 [0:256, 0:197, 128:192] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v778 = stablehlo.reshape %v777 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v779 = stablehlo.reshape %v697 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v780 = stablehlo.slice %v779 [0:256, 0:197, 128:192] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v781 = stablehlo.reshape %v780 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v782 = stablehlo.reshape %v778 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v783 = stablehlo.transpose %v782, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v784 = stablehlo.reshape %v783 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v785 = stablehlo.reshape %v775 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v786 = stablehlo.reshape %v784 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v787 = stablehlo.dot_general %v785, %v786, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v788 = stablehlo.reshape %v787 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v789 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v790 = stablehlo.multiply %v788, %v789 : tensor<256x38809xf32>
    %v791 = stablehlo.reshape %v790 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v792 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v793 = stablehlo.reduce(%v791 init: %v792) applies stablehlo.maximum across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v794 = stablehlo.broadcast_in_dim %v793, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v795 = stablehlo.subtract %v791, %v794 : tensor<256x197x197xf32>
    %v796 = stablehlo.constant dense<0.0> : tensor<f32>
    %v797 = stablehlo.exponential %v795 : tensor<256x197x197xf32>
    %v798 = stablehlo.reduce(%v797 init: %v796) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v799 = stablehlo.broadcast_in_dim %v798, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v800 = stablehlo.divide %v797, %v799 : tensor<256x197x197xf32>
    %v801 = stablehlo.reshape %v800 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v802 = stablehlo.reshape %v801 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v803 = stablehlo.reshape %v781 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v804 = stablehlo.dot_general %v802, %v803, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v805 = stablehlo.reshape %v804 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v806 = stablehlo.reshape %v805 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v807 = stablehlo.constant dense<0.0> : tensor<f32>
    %v808 = stablehlo.pad %v806, %v807, low = [0, 0, 128], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x384xf32>
    %v809 = stablehlo.reshape %v808 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v810 = stablehlo.add %v772, %v809 : tensor<256x75648xf32>
    %v811 = stablehlo.reshape %v687 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v812 = stablehlo.slice %v811 [0:256, 0:197, 192:256] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v813 = stablehlo.reshape %v812 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v814 = stablehlo.reshape %v692 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v815 = stablehlo.slice %v814 [0:256, 0:197, 192:256] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v816 = stablehlo.reshape %v815 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v817 = stablehlo.reshape %v697 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v818 = stablehlo.slice %v817 [0:256, 0:197, 192:256] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v819 = stablehlo.reshape %v818 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v820 = stablehlo.reshape %v816 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v821 = stablehlo.transpose %v820, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v822 = stablehlo.reshape %v821 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v823 = stablehlo.reshape %v813 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v824 = stablehlo.reshape %v822 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v825 = stablehlo.dot_general %v823, %v824, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v826 = stablehlo.reshape %v825 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v827 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v828 = stablehlo.multiply %v826, %v827 : tensor<256x38809xf32>
    %v829 = stablehlo.reshape %v828 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v830 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v831 = stablehlo.reduce(%v829 init: %v830) applies stablehlo.maximum across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v832 = stablehlo.broadcast_in_dim %v831, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v833 = stablehlo.subtract %v829, %v832 : tensor<256x197x197xf32>
    %v834 = stablehlo.constant dense<0.0> : tensor<f32>
    %v835 = stablehlo.exponential %v833 : tensor<256x197x197xf32>
    %v836 = stablehlo.reduce(%v835 init: %v834) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v837 = stablehlo.broadcast_in_dim %v836, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v838 = stablehlo.divide %v835, %v837 : tensor<256x197x197xf32>
    %v839 = stablehlo.reshape %v838 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v840 = stablehlo.reshape %v839 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v841 = stablehlo.reshape %v819 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v842 = stablehlo.dot_general %v840, %v841, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v843 = stablehlo.reshape %v842 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v844 = stablehlo.reshape %v843 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v845 = stablehlo.constant dense<0.0> : tensor<f32>
    %v846 = stablehlo.pad %v844, %v845, low = [0, 0, 192], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x384xf32>
    %v847 = stablehlo.reshape %v846 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v848 = stablehlo.add %v810, %v847 : tensor<256x75648xf32>
    %v849 = stablehlo.reshape %v687 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v850 = stablehlo.slice %v849 [0:256, 0:197, 256:320] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v851 = stablehlo.reshape %v850 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v852 = stablehlo.reshape %v692 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v853 = stablehlo.slice %v852 [0:256, 0:197, 256:320] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v854 = stablehlo.reshape %v853 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v855 = stablehlo.reshape %v697 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v856 = stablehlo.slice %v855 [0:256, 0:197, 256:320] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v857 = stablehlo.reshape %v856 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v858 = stablehlo.reshape %v854 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v859 = stablehlo.transpose %v858, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v860 = stablehlo.reshape %v859 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v861 = stablehlo.reshape %v851 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v862 = stablehlo.reshape %v860 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v863 = stablehlo.dot_general %v861, %v862, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v864 = stablehlo.reshape %v863 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v865 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v866 = stablehlo.multiply %v864, %v865 : tensor<256x38809xf32>
    %v867 = stablehlo.reshape %v866 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v868 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v869 = stablehlo.reduce(%v867 init: %v868) applies stablehlo.maximum across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v870 = stablehlo.broadcast_in_dim %v869, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v871 = stablehlo.subtract %v867, %v870 : tensor<256x197x197xf32>
    %v872 = stablehlo.constant dense<0.0> : tensor<f32>
    %v873 = stablehlo.exponential %v871 : tensor<256x197x197xf32>
    %v874 = stablehlo.reduce(%v873 init: %v872) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v875 = stablehlo.broadcast_in_dim %v874, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v876 = stablehlo.divide %v873, %v875 : tensor<256x197x197xf32>
    %v877 = stablehlo.reshape %v876 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v878 = stablehlo.reshape %v877 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v879 = stablehlo.reshape %v857 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v880 = stablehlo.dot_general %v878, %v879, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v881 = stablehlo.reshape %v880 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v882 = stablehlo.reshape %v881 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v883 = stablehlo.constant dense<0.0> : tensor<f32>
    %v884 = stablehlo.pad %v882, %v883, low = [0, 0, 256], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x384xf32>
    %v885 = stablehlo.reshape %v884 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v886 = stablehlo.add %v848, %v885 : tensor<256x75648xf32>
    %v887 = stablehlo.reshape %v687 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v888 = stablehlo.slice %v887 [0:256, 0:197, 320:384] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v889 = stablehlo.reshape %v888 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v890 = stablehlo.reshape %v692 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v891 = stablehlo.slice %v890 [0:256, 0:197, 320:384] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v892 = stablehlo.reshape %v891 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v893 = stablehlo.reshape %v697 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v894 = stablehlo.slice %v893 [0:256, 0:197, 320:384] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v895 = stablehlo.reshape %v894 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v896 = stablehlo.reshape %v892 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v897 = stablehlo.transpose %v896, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v898 = stablehlo.reshape %v897 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v899 = stablehlo.reshape %v889 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v900 = stablehlo.reshape %v898 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v901 = stablehlo.dot_general %v899, %v900, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v902 = stablehlo.reshape %v901 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v903 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v904 = stablehlo.multiply %v902, %v903 : tensor<256x38809xf32>
    %v905 = stablehlo.reshape %v904 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v906 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v907 = stablehlo.reduce(%v905 init: %v906) applies stablehlo.maximum across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v908 = stablehlo.broadcast_in_dim %v907, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v909 = stablehlo.subtract %v905, %v908 : tensor<256x197x197xf32>
    %v910 = stablehlo.constant dense<0.0> : tensor<f32>
    %v911 = stablehlo.exponential %v909 : tensor<256x197x197xf32>
    %v912 = stablehlo.reduce(%v911 init: %v910) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v913 = stablehlo.broadcast_in_dim %v912, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v914 = stablehlo.divide %v911, %v913 : tensor<256x197x197xf32>
    %v915 = stablehlo.reshape %v914 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v916 = stablehlo.reshape %v915 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v917 = stablehlo.reshape %v895 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v918 = stablehlo.dot_general %v916, %v917, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v919 = stablehlo.reshape %v918 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v920 = stablehlo.reshape %v919 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v921 = stablehlo.constant dense<0.0> : tensor<f32>
    %v922 = stablehlo.pad %v920, %v921, low = [0, 0, 320], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x384xf32>
    %v923 = stablehlo.reshape %v922 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v924 = stablehlo.add %v886, %v923 : tensor<256x75648xf32>
    %v925 = stablehlo.reshape %v924 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v926 = stablehlo.dot_general %v925, %b2_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x384xf32>, tensor<384x384xf32>) -> tensor<256x197x384xf32>
    %v927 = stablehlo.broadcast_in_dim %b2_bo, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v928 = stablehlo.add %v926, %v927 : tensor<256x197x384xf32>
    %v929 = stablehlo.reshape %v928 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v930 = stablehlo.add %v654, %v929 : tensor<256x75648xf32>
    %v931 = stablehlo.reshape %v930 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v932 = stablehlo.constant dense<0.0> : tensor<f32>
    %v933 = stablehlo.constant dense<384.0> : tensor<256x197x384xf32>
    %v934 = stablehlo.constant dense<1.0e-6> : tensor<256x197x384xf32>
    %v935 = stablehlo.reduce(%v931 init: %v932) applies stablehlo.add across dimensions = [2] : (tensor<256x197x384xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v936 = stablehlo.broadcast_in_dim %v935, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x384xf32>
    %v937 = stablehlo.divide %v936, %v933 : tensor<256x197x384xf32>
    %v938 = stablehlo.subtract %v931, %v937 : tensor<256x197x384xf32>
    %v939 = stablehlo.multiply %v938, %v938 : tensor<256x197x384xf32>
    %v940 = stablehlo.reduce(%v939 init: %v932) applies stablehlo.add across dimensions = [2] : (tensor<256x197x384xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v941 = stablehlo.broadcast_in_dim %v940, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x384xf32>
    %v942 = stablehlo.divide %v941, %v933 : tensor<256x197x384xf32>
    %v943 = stablehlo.add %v942, %v934 : tensor<256x197x384xf32>
    %v944 = stablehlo.rsqrt %v943 : tensor<256x197x384xf32>
    %v945 = stablehlo.multiply %v938, %v944 : tensor<256x197x384xf32>
    %v946 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<256x197x384xf32>
    %v947 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<256x197x384xf32>
    %v948 = stablehlo.multiply %v945, %v946 : tensor<256x197x384xf32>
    %v949 = stablehlo.add %v948, %v947 : tensor<256x197x384xf32>
    %v950 = stablehlo.reshape %v949 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v951 = stablehlo.reshape %v950 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v952 = stablehlo.broadcast_in_dim %b2_g2, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v953 = stablehlo.multiply %v951, %v952 : tensor<256x197x384xf32>
    %v954 = stablehlo.reshape %v953 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v955 = stablehlo.reshape %v954 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v956 = stablehlo.broadcast_in_dim %b2_bt2, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v957 = stablehlo.add %v955, %v956 : tensor<256x197x384xf32>
    %v958 = stablehlo.reshape %v957 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v959 = stablehlo.reshape %v958 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v960 = stablehlo.dot_general %v959, %b2_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x384xf32>, tensor<384x1536xf32>) -> tensor<256x197x1536xf32>
    %v961 = stablehlo.broadcast_in_dim %b2_bfc1, dims = [2] : (tensor<1536xf32>) -> tensor<256x197x1536xf32>
    %v962 = stablehlo.add %v960, %v961 : tensor<256x197x1536xf32>
    %v963 = stablehlo.reshape %v962 : (tensor<256x197x1536xf32>) -> tensor<256x302592xf32>
    %v964 = stablehlo.constant dense<0.5> : tensor<256x302592xf32>
    %v965 = stablehlo.multiply %v964, %v963 : tensor<256x302592xf32>
    %v966 = stablehlo.negate %v963 : tensor<256x302592xf32>
    %v967 = stablehlo.constant dense<0.7071067811865476> : tensor<256x302592xf32>
    %v968 = stablehlo.multiply %v966, %v967 : tensor<256x302592xf32>
    %v969 = chlo.erfc %v968 : tensor<256x302592xf32> -> tensor<256x302592xf32>
    %v970 = stablehlo.multiply %v965, %v969 : tensor<256x302592xf32>
    %v971 = stablehlo.reshape %v970 : (tensor<256x302592xf32>) -> tensor<256x197x1536xf32>
    %v972 = stablehlo.dot_general %v971, %b2_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x1536xf32>, tensor<1536x384xf32>) -> tensor<256x197x384xf32>
    %v973 = stablehlo.broadcast_in_dim %b2_bfc2, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v974 = stablehlo.add %v972, %v973 : tensor<256x197x384xf32>
    %v975 = stablehlo.reshape %v974 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v976 = stablehlo.add %v930, %v975 : tensor<256x75648xf32>
    %v977 = stablehlo.reshape %v976 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v978 = stablehlo.constant dense<0.0> : tensor<f32>
    %v979 = stablehlo.constant dense<384.0> : tensor<256x197x384xf32>
    %v980 = stablehlo.constant dense<1.0e-6> : tensor<256x197x384xf32>
    %v981 = stablehlo.reduce(%v977 init: %v978) applies stablehlo.add across dimensions = [2] : (tensor<256x197x384xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v982 = stablehlo.broadcast_in_dim %v981, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x384xf32>
    %v983 = stablehlo.divide %v982, %v979 : tensor<256x197x384xf32>
    %v984 = stablehlo.subtract %v977, %v983 : tensor<256x197x384xf32>
    %v985 = stablehlo.multiply %v984, %v984 : tensor<256x197x384xf32>
    %v986 = stablehlo.reduce(%v985 init: %v978) applies stablehlo.add across dimensions = [2] : (tensor<256x197x384xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v987 = stablehlo.broadcast_in_dim %v986, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x384xf32>
    %v988 = stablehlo.divide %v987, %v979 : tensor<256x197x384xf32>
    %v989 = stablehlo.add %v988, %v980 : tensor<256x197x384xf32>
    %v990 = stablehlo.rsqrt %v989 : tensor<256x197x384xf32>
    %v991 = stablehlo.multiply %v984, %v990 : tensor<256x197x384xf32>
    %v992 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<256x197x384xf32>
    %v993 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<256x197x384xf32>
    %v994 = stablehlo.multiply %v991, %v992 : tensor<256x197x384xf32>
    %v995 = stablehlo.add %v994, %v993 : tensor<256x197x384xf32>
    %v996 = stablehlo.reshape %v995 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v997 = stablehlo.reshape %v996 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v998 = stablehlo.broadcast_in_dim %b3_g1, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v999 = stablehlo.multiply %v997, %v998 : tensor<256x197x384xf32>
    %v1000 = stablehlo.reshape %v999 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v1001 = stablehlo.reshape %v1000 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1002 = stablehlo.broadcast_in_dim %b3_bt1, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v1003 = stablehlo.add %v1001, %v1002 : tensor<256x197x384xf32>
    %v1004 = stablehlo.reshape %v1003 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v1005 = stablehlo.reshape %v1004 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1006 = stablehlo.dot_general %v1005, %b3_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x384xf32>, tensor<384x384xf32>) -> tensor<256x197x384xf32>
    %v1007 = stablehlo.broadcast_in_dim %b3_bq, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v1008 = stablehlo.add %v1006, %v1007 : tensor<256x197x384xf32>
    %v1009 = stablehlo.reshape %v1008 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v1010 = stablehlo.reshape %v1004 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1011 = stablehlo.dot_general %v1010, %b3_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x384xf32>, tensor<384x384xf32>) -> tensor<256x197x384xf32>
    %v1012 = stablehlo.broadcast_in_dim %b3_bk, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v1013 = stablehlo.add %v1011, %v1012 : tensor<256x197x384xf32>
    %v1014 = stablehlo.reshape %v1013 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v1015 = stablehlo.reshape %v1004 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1016 = stablehlo.dot_general %v1015, %b3_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x384xf32>, tensor<384x384xf32>) -> tensor<256x197x384xf32>
    %v1017 = stablehlo.broadcast_in_dim %b3_bv, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v1018 = stablehlo.add %v1016, %v1017 : tensor<256x197x384xf32>
    %v1019 = stablehlo.reshape %v1018 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v1020 = stablehlo.reshape %v1009 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1021 = stablehlo.slice %v1020 [0:256, 0:197, 0:64] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v1022 = stablehlo.reshape %v1021 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1023 = stablehlo.reshape %v1014 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1024 = stablehlo.slice %v1023 [0:256, 0:197, 0:64] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v1025 = stablehlo.reshape %v1024 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1026 = stablehlo.reshape %v1019 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1027 = stablehlo.slice %v1026 [0:256, 0:197, 0:64] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v1028 = stablehlo.reshape %v1027 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1029 = stablehlo.reshape %v1025 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1030 = stablehlo.transpose %v1029, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v1031 = stablehlo.reshape %v1030 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v1032 = stablehlo.reshape %v1022 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1033 = stablehlo.reshape %v1031 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v1034 = stablehlo.dot_general %v1032, %v1033, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v1035 = stablehlo.reshape %v1034 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v1036 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v1037 = stablehlo.multiply %v1035, %v1036 : tensor<256x38809xf32>
    %v1038 = stablehlo.reshape %v1037 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v1039 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v1040 = stablehlo.reduce(%v1038 init: %v1039) applies stablehlo.maximum across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1041 = stablehlo.broadcast_in_dim %v1040, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v1042 = stablehlo.subtract %v1038, %v1041 : tensor<256x197x197xf32>
    %v1043 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1044 = stablehlo.exponential %v1042 : tensor<256x197x197xf32>
    %v1045 = stablehlo.reduce(%v1044 init: %v1043) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1046 = stablehlo.broadcast_in_dim %v1045, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v1047 = stablehlo.divide %v1044, %v1046 : tensor<256x197x197xf32>
    %v1048 = stablehlo.reshape %v1047 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v1049 = stablehlo.reshape %v1048 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v1050 = stablehlo.reshape %v1028 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1051 = stablehlo.dot_general %v1049, %v1050, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v1052 = stablehlo.reshape %v1051 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1053 = stablehlo.reshape %v1052 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1054 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1055 = stablehlo.pad %v1053, %v1054, low = [0, 0, 0], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x384xf32>
    %v1056 = stablehlo.reshape %v1055 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v1057 = stablehlo.reshape %v1009 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1058 = stablehlo.slice %v1057 [0:256, 0:197, 64:128] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v1059 = stablehlo.reshape %v1058 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1060 = stablehlo.reshape %v1014 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1061 = stablehlo.slice %v1060 [0:256, 0:197, 64:128] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v1062 = stablehlo.reshape %v1061 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1063 = stablehlo.reshape %v1019 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1064 = stablehlo.slice %v1063 [0:256, 0:197, 64:128] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v1065 = stablehlo.reshape %v1064 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1066 = stablehlo.reshape %v1062 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1067 = stablehlo.transpose %v1066, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v1068 = stablehlo.reshape %v1067 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v1069 = stablehlo.reshape %v1059 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1070 = stablehlo.reshape %v1068 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v1071 = stablehlo.dot_general %v1069, %v1070, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v1072 = stablehlo.reshape %v1071 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v1073 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v1074 = stablehlo.multiply %v1072, %v1073 : tensor<256x38809xf32>
    %v1075 = stablehlo.reshape %v1074 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v1076 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v1077 = stablehlo.reduce(%v1075 init: %v1076) applies stablehlo.maximum across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1078 = stablehlo.broadcast_in_dim %v1077, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v1079 = stablehlo.subtract %v1075, %v1078 : tensor<256x197x197xf32>
    %v1080 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1081 = stablehlo.exponential %v1079 : tensor<256x197x197xf32>
    %v1082 = stablehlo.reduce(%v1081 init: %v1080) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1083 = stablehlo.broadcast_in_dim %v1082, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v1084 = stablehlo.divide %v1081, %v1083 : tensor<256x197x197xf32>
    %v1085 = stablehlo.reshape %v1084 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v1086 = stablehlo.reshape %v1085 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v1087 = stablehlo.reshape %v1065 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1088 = stablehlo.dot_general %v1086, %v1087, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v1089 = stablehlo.reshape %v1088 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1090 = stablehlo.reshape %v1089 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1091 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1092 = stablehlo.pad %v1090, %v1091, low = [0, 0, 64], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x384xf32>
    %v1093 = stablehlo.reshape %v1092 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v1094 = stablehlo.add %v1056, %v1093 : tensor<256x75648xf32>
    %v1095 = stablehlo.reshape %v1009 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1096 = stablehlo.slice %v1095 [0:256, 0:197, 128:192] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v1097 = stablehlo.reshape %v1096 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1098 = stablehlo.reshape %v1014 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1099 = stablehlo.slice %v1098 [0:256, 0:197, 128:192] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v1100 = stablehlo.reshape %v1099 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1101 = stablehlo.reshape %v1019 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1102 = stablehlo.slice %v1101 [0:256, 0:197, 128:192] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v1103 = stablehlo.reshape %v1102 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1104 = stablehlo.reshape %v1100 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1105 = stablehlo.transpose %v1104, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v1106 = stablehlo.reshape %v1105 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v1107 = stablehlo.reshape %v1097 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1108 = stablehlo.reshape %v1106 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v1109 = stablehlo.dot_general %v1107, %v1108, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v1110 = stablehlo.reshape %v1109 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v1111 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v1112 = stablehlo.multiply %v1110, %v1111 : tensor<256x38809xf32>
    %v1113 = stablehlo.reshape %v1112 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v1114 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v1115 = stablehlo.reduce(%v1113 init: %v1114) applies stablehlo.maximum across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1116 = stablehlo.broadcast_in_dim %v1115, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v1117 = stablehlo.subtract %v1113, %v1116 : tensor<256x197x197xf32>
    %v1118 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1119 = stablehlo.exponential %v1117 : tensor<256x197x197xf32>
    %v1120 = stablehlo.reduce(%v1119 init: %v1118) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1121 = stablehlo.broadcast_in_dim %v1120, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v1122 = stablehlo.divide %v1119, %v1121 : tensor<256x197x197xf32>
    %v1123 = stablehlo.reshape %v1122 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v1124 = stablehlo.reshape %v1123 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v1125 = stablehlo.reshape %v1103 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1126 = stablehlo.dot_general %v1124, %v1125, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v1127 = stablehlo.reshape %v1126 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1128 = stablehlo.reshape %v1127 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1129 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1130 = stablehlo.pad %v1128, %v1129, low = [0, 0, 128], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x384xf32>
    %v1131 = stablehlo.reshape %v1130 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v1132 = stablehlo.add %v1094, %v1131 : tensor<256x75648xf32>
    %v1133 = stablehlo.reshape %v1009 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1134 = stablehlo.slice %v1133 [0:256, 0:197, 192:256] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v1135 = stablehlo.reshape %v1134 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1136 = stablehlo.reshape %v1014 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1137 = stablehlo.slice %v1136 [0:256, 0:197, 192:256] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v1138 = stablehlo.reshape %v1137 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1139 = stablehlo.reshape %v1019 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1140 = stablehlo.slice %v1139 [0:256, 0:197, 192:256] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v1141 = stablehlo.reshape %v1140 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1142 = stablehlo.reshape %v1138 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1143 = stablehlo.transpose %v1142, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v1144 = stablehlo.reshape %v1143 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v1145 = stablehlo.reshape %v1135 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1146 = stablehlo.reshape %v1144 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v1147 = stablehlo.dot_general %v1145, %v1146, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v1148 = stablehlo.reshape %v1147 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v1149 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v1150 = stablehlo.multiply %v1148, %v1149 : tensor<256x38809xf32>
    %v1151 = stablehlo.reshape %v1150 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v1152 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v1153 = stablehlo.reduce(%v1151 init: %v1152) applies stablehlo.maximum across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1154 = stablehlo.broadcast_in_dim %v1153, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v1155 = stablehlo.subtract %v1151, %v1154 : tensor<256x197x197xf32>
    %v1156 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1157 = stablehlo.exponential %v1155 : tensor<256x197x197xf32>
    %v1158 = stablehlo.reduce(%v1157 init: %v1156) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1159 = stablehlo.broadcast_in_dim %v1158, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v1160 = stablehlo.divide %v1157, %v1159 : tensor<256x197x197xf32>
    %v1161 = stablehlo.reshape %v1160 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v1162 = stablehlo.reshape %v1161 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v1163 = stablehlo.reshape %v1141 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1164 = stablehlo.dot_general %v1162, %v1163, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v1165 = stablehlo.reshape %v1164 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1166 = stablehlo.reshape %v1165 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1167 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1168 = stablehlo.pad %v1166, %v1167, low = [0, 0, 192], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x384xf32>
    %v1169 = stablehlo.reshape %v1168 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v1170 = stablehlo.add %v1132, %v1169 : tensor<256x75648xf32>
    %v1171 = stablehlo.reshape %v1009 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1172 = stablehlo.slice %v1171 [0:256, 0:197, 256:320] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v1173 = stablehlo.reshape %v1172 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1174 = stablehlo.reshape %v1014 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1175 = stablehlo.slice %v1174 [0:256, 0:197, 256:320] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v1176 = stablehlo.reshape %v1175 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1177 = stablehlo.reshape %v1019 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1178 = stablehlo.slice %v1177 [0:256, 0:197, 256:320] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v1179 = stablehlo.reshape %v1178 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1180 = stablehlo.reshape %v1176 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1181 = stablehlo.transpose %v1180, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v1182 = stablehlo.reshape %v1181 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v1183 = stablehlo.reshape %v1173 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1184 = stablehlo.reshape %v1182 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v1185 = stablehlo.dot_general %v1183, %v1184, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v1186 = stablehlo.reshape %v1185 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v1187 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v1188 = stablehlo.multiply %v1186, %v1187 : tensor<256x38809xf32>
    %v1189 = stablehlo.reshape %v1188 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v1190 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v1191 = stablehlo.reduce(%v1189 init: %v1190) applies stablehlo.maximum across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1192 = stablehlo.broadcast_in_dim %v1191, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v1193 = stablehlo.subtract %v1189, %v1192 : tensor<256x197x197xf32>
    %v1194 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1195 = stablehlo.exponential %v1193 : tensor<256x197x197xf32>
    %v1196 = stablehlo.reduce(%v1195 init: %v1194) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1197 = stablehlo.broadcast_in_dim %v1196, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v1198 = stablehlo.divide %v1195, %v1197 : tensor<256x197x197xf32>
    %v1199 = stablehlo.reshape %v1198 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v1200 = stablehlo.reshape %v1199 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v1201 = stablehlo.reshape %v1179 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1202 = stablehlo.dot_general %v1200, %v1201, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v1203 = stablehlo.reshape %v1202 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1204 = stablehlo.reshape %v1203 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1205 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1206 = stablehlo.pad %v1204, %v1205, low = [0, 0, 256], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x384xf32>
    %v1207 = stablehlo.reshape %v1206 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v1208 = stablehlo.add %v1170, %v1207 : tensor<256x75648xf32>
    %v1209 = stablehlo.reshape %v1009 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1210 = stablehlo.slice %v1209 [0:256, 0:197, 320:384] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v1211 = stablehlo.reshape %v1210 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1212 = stablehlo.reshape %v1014 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1213 = stablehlo.slice %v1212 [0:256, 0:197, 320:384] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v1214 = stablehlo.reshape %v1213 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1215 = stablehlo.reshape %v1019 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1216 = stablehlo.slice %v1215 [0:256, 0:197, 320:384] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v1217 = stablehlo.reshape %v1216 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1218 = stablehlo.reshape %v1214 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1219 = stablehlo.transpose %v1218, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v1220 = stablehlo.reshape %v1219 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v1221 = stablehlo.reshape %v1211 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1222 = stablehlo.reshape %v1220 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v1223 = stablehlo.dot_general %v1221, %v1222, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v1224 = stablehlo.reshape %v1223 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v1225 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v1226 = stablehlo.multiply %v1224, %v1225 : tensor<256x38809xf32>
    %v1227 = stablehlo.reshape %v1226 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v1228 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v1229 = stablehlo.reduce(%v1227 init: %v1228) applies stablehlo.maximum across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1230 = stablehlo.broadcast_in_dim %v1229, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v1231 = stablehlo.subtract %v1227, %v1230 : tensor<256x197x197xf32>
    %v1232 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1233 = stablehlo.exponential %v1231 : tensor<256x197x197xf32>
    %v1234 = stablehlo.reduce(%v1233 init: %v1232) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1235 = stablehlo.broadcast_in_dim %v1234, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v1236 = stablehlo.divide %v1233, %v1235 : tensor<256x197x197xf32>
    %v1237 = stablehlo.reshape %v1236 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v1238 = stablehlo.reshape %v1237 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v1239 = stablehlo.reshape %v1217 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1240 = stablehlo.dot_general %v1238, %v1239, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v1241 = stablehlo.reshape %v1240 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1242 = stablehlo.reshape %v1241 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1243 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1244 = stablehlo.pad %v1242, %v1243, low = [0, 0, 320], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x384xf32>
    %v1245 = stablehlo.reshape %v1244 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v1246 = stablehlo.add %v1208, %v1245 : tensor<256x75648xf32>
    %v1247 = stablehlo.reshape %v1246 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1248 = stablehlo.dot_general %v1247, %b3_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x384xf32>, tensor<384x384xf32>) -> tensor<256x197x384xf32>
    %v1249 = stablehlo.broadcast_in_dim %b3_bo, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v1250 = stablehlo.add %v1248, %v1249 : tensor<256x197x384xf32>
    %v1251 = stablehlo.reshape %v1250 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v1252 = stablehlo.add %v976, %v1251 : tensor<256x75648xf32>
    %v1253 = stablehlo.reshape %v1252 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1254 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1255 = stablehlo.constant dense<384.0> : tensor<256x197x384xf32>
    %v1256 = stablehlo.constant dense<1.0e-6> : tensor<256x197x384xf32>
    %v1257 = stablehlo.reduce(%v1253 init: %v1254) applies stablehlo.add across dimensions = [2] : (tensor<256x197x384xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1258 = stablehlo.broadcast_in_dim %v1257, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x384xf32>
    %v1259 = stablehlo.divide %v1258, %v1255 : tensor<256x197x384xf32>
    %v1260 = stablehlo.subtract %v1253, %v1259 : tensor<256x197x384xf32>
    %v1261 = stablehlo.multiply %v1260, %v1260 : tensor<256x197x384xf32>
    %v1262 = stablehlo.reduce(%v1261 init: %v1254) applies stablehlo.add across dimensions = [2] : (tensor<256x197x384xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1263 = stablehlo.broadcast_in_dim %v1262, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x384xf32>
    %v1264 = stablehlo.divide %v1263, %v1255 : tensor<256x197x384xf32>
    %v1265 = stablehlo.add %v1264, %v1256 : tensor<256x197x384xf32>
    %v1266 = stablehlo.rsqrt %v1265 : tensor<256x197x384xf32>
    %v1267 = stablehlo.multiply %v1260, %v1266 : tensor<256x197x384xf32>
    %v1268 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<256x197x384xf32>
    %v1269 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<256x197x384xf32>
    %v1270 = stablehlo.multiply %v1267, %v1268 : tensor<256x197x384xf32>
    %v1271 = stablehlo.add %v1270, %v1269 : tensor<256x197x384xf32>
    %v1272 = stablehlo.reshape %v1271 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v1273 = stablehlo.reshape %v1272 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1274 = stablehlo.broadcast_in_dim %b3_g2, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v1275 = stablehlo.multiply %v1273, %v1274 : tensor<256x197x384xf32>
    %v1276 = stablehlo.reshape %v1275 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v1277 = stablehlo.reshape %v1276 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1278 = stablehlo.broadcast_in_dim %b3_bt2, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v1279 = stablehlo.add %v1277, %v1278 : tensor<256x197x384xf32>
    %v1280 = stablehlo.reshape %v1279 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v1281 = stablehlo.reshape %v1280 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1282 = stablehlo.dot_general %v1281, %b3_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x384xf32>, tensor<384x1536xf32>) -> tensor<256x197x1536xf32>
    %v1283 = stablehlo.broadcast_in_dim %b3_bfc1, dims = [2] : (tensor<1536xf32>) -> tensor<256x197x1536xf32>
    %v1284 = stablehlo.add %v1282, %v1283 : tensor<256x197x1536xf32>
    %v1285 = stablehlo.reshape %v1284 : (tensor<256x197x1536xf32>) -> tensor<256x302592xf32>
    %v1286 = stablehlo.constant dense<0.5> : tensor<256x302592xf32>
    %v1287 = stablehlo.multiply %v1286, %v1285 : tensor<256x302592xf32>
    %v1288 = stablehlo.negate %v1285 : tensor<256x302592xf32>
    %v1289 = stablehlo.constant dense<0.7071067811865476> : tensor<256x302592xf32>
    %v1290 = stablehlo.multiply %v1288, %v1289 : tensor<256x302592xf32>
    %v1291 = chlo.erfc %v1290 : tensor<256x302592xf32> -> tensor<256x302592xf32>
    %v1292 = stablehlo.multiply %v1287, %v1291 : tensor<256x302592xf32>
    %v1293 = stablehlo.reshape %v1292 : (tensor<256x302592xf32>) -> tensor<256x197x1536xf32>
    %v1294 = stablehlo.dot_general %v1293, %b3_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x1536xf32>, tensor<1536x384xf32>) -> tensor<256x197x384xf32>
    %v1295 = stablehlo.broadcast_in_dim %b3_bfc2, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v1296 = stablehlo.add %v1294, %v1295 : tensor<256x197x384xf32>
    %v1297 = stablehlo.reshape %v1296 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v1298 = stablehlo.add %v1252, %v1297 : tensor<256x75648xf32>
    %v1299 = stablehlo.reshape %v1298 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1300 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1301 = stablehlo.constant dense<384.0> : tensor<256x197x384xf32>
    %v1302 = stablehlo.constant dense<1.0e-6> : tensor<256x197x384xf32>
    %v1303 = stablehlo.reduce(%v1299 init: %v1300) applies stablehlo.add across dimensions = [2] : (tensor<256x197x384xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1304 = stablehlo.broadcast_in_dim %v1303, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x384xf32>
    %v1305 = stablehlo.divide %v1304, %v1301 : tensor<256x197x384xf32>
    %v1306 = stablehlo.subtract %v1299, %v1305 : tensor<256x197x384xf32>
    %v1307 = stablehlo.multiply %v1306, %v1306 : tensor<256x197x384xf32>
    %v1308 = stablehlo.reduce(%v1307 init: %v1300) applies stablehlo.add across dimensions = [2] : (tensor<256x197x384xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1309 = stablehlo.broadcast_in_dim %v1308, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x384xf32>
    %v1310 = stablehlo.divide %v1309, %v1301 : tensor<256x197x384xf32>
    %v1311 = stablehlo.add %v1310, %v1302 : tensor<256x197x384xf32>
    %v1312 = stablehlo.rsqrt %v1311 : tensor<256x197x384xf32>
    %v1313 = stablehlo.multiply %v1306, %v1312 : tensor<256x197x384xf32>
    %v1314 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<256x197x384xf32>
    %v1315 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<256x197x384xf32>
    %v1316 = stablehlo.multiply %v1313, %v1314 : tensor<256x197x384xf32>
    %v1317 = stablehlo.add %v1316, %v1315 : tensor<256x197x384xf32>
    %v1318 = stablehlo.reshape %v1317 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v1319 = stablehlo.reshape %v1318 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1320 = stablehlo.broadcast_in_dim %b4_g1, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v1321 = stablehlo.multiply %v1319, %v1320 : tensor<256x197x384xf32>
    %v1322 = stablehlo.reshape %v1321 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v1323 = stablehlo.reshape %v1322 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1324 = stablehlo.broadcast_in_dim %b4_bt1, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v1325 = stablehlo.add %v1323, %v1324 : tensor<256x197x384xf32>
    %v1326 = stablehlo.reshape %v1325 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v1327 = stablehlo.reshape %v1326 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1328 = stablehlo.dot_general %v1327, %b4_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x384xf32>, tensor<384x384xf32>) -> tensor<256x197x384xf32>
    %v1329 = stablehlo.broadcast_in_dim %b4_bq, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v1330 = stablehlo.add %v1328, %v1329 : tensor<256x197x384xf32>
    %v1331 = stablehlo.reshape %v1330 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v1332 = stablehlo.reshape %v1326 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1333 = stablehlo.dot_general %v1332, %b4_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x384xf32>, tensor<384x384xf32>) -> tensor<256x197x384xf32>
    %v1334 = stablehlo.broadcast_in_dim %b4_bk, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v1335 = stablehlo.add %v1333, %v1334 : tensor<256x197x384xf32>
    %v1336 = stablehlo.reshape %v1335 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v1337 = stablehlo.reshape %v1326 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1338 = stablehlo.dot_general %v1337, %b4_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x384xf32>, tensor<384x384xf32>) -> tensor<256x197x384xf32>
    %v1339 = stablehlo.broadcast_in_dim %b4_bv, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v1340 = stablehlo.add %v1338, %v1339 : tensor<256x197x384xf32>
    %v1341 = stablehlo.reshape %v1340 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v1342 = stablehlo.reshape %v1331 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1343 = stablehlo.slice %v1342 [0:256, 0:197, 0:64] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v1344 = stablehlo.reshape %v1343 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1345 = stablehlo.reshape %v1336 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1346 = stablehlo.slice %v1345 [0:256, 0:197, 0:64] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v1347 = stablehlo.reshape %v1346 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1348 = stablehlo.reshape %v1341 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1349 = stablehlo.slice %v1348 [0:256, 0:197, 0:64] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v1350 = stablehlo.reshape %v1349 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1351 = stablehlo.reshape %v1347 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1352 = stablehlo.transpose %v1351, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v1353 = stablehlo.reshape %v1352 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v1354 = stablehlo.reshape %v1344 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1355 = stablehlo.reshape %v1353 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v1356 = stablehlo.dot_general %v1354, %v1355, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v1357 = stablehlo.reshape %v1356 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v1358 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v1359 = stablehlo.multiply %v1357, %v1358 : tensor<256x38809xf32>
    %v1360 = stablehlo.reshape %v1359 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v1361 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v1362 = stablehlo.reduce(%v1360 init: %v1361) applies stablehlo.maximum across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1363 = stablehlo.broadcast_in_dim %v1362, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v1364 = stablehlo.subtract %v1360, %v1363 : tensor<256x197x197xf32>
    %v1365 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1366 = stablehlo.exponential %v1364 : tensor<256x197x197xf32>
    %v1367 = stablehlo.reduce(%v1366 init: %v1365) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1368 = stablehlo.broadcast_in_dim %v1367, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v1369 = stablehlo.divide %v1366, %v1368 : tensor<256x197x197xf32>
    %v1370 = stablehlo.reshape %v1369 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v1371 = stablehlo.reshape %v1370 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v1372 = stablehlo.reshape %v1350 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1373 = stablehlo.dot_general %v1371, %v1372, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v1374 = stablehlo.reshape %v1373 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1375 = stablehlo.reshape %v1374 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1376 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1377 = stablehlo.pad %v1375, %v1376, low = [0, 0, 0], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x384xf32>
    %v1378 = stablehlo.reshape %v1377 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v1379 = stablehlo.reshape %v1331 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1380 = stablehlo.slice %v1379 [0:256, 0:197, 64:128] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v1381 = stablehlo.reshape %v1380 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1382 = stablehlo.reshape %v1336 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1383 = stablehlo.slice %v1382 [0:256, 0:197, 64:128] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v1384 = stablehlo.reshape %v1383 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1385 = stablehlo.reshape %v1341 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1386 = stablehlo.slice %v1385 [0:256, 0:197, 64:128] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v1387 = stablehlo.reshape %v1386 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1388 = stablehlo.reshape %v1384 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1389 = stablehlo.transpose %v1388, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v1390 = stablehlo.reshape %v1389 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v1391 = stablehlo.reshape %v1381 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1392 = stablehlo.reshape %v1390 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v1393 = stablehlo.dot_general %v1391, %v1392, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v1394 = stablehlo.reshape %v1393 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v1395 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v1396 = stablehlo.multiply %v1394, %v1395 : tensor<256x38809xf32>
    %v1397 = stablehlo.reshape %v1396 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v1398 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v1399 = stablehlo.reduce(%v1397 init: %v1398) applies stablehlo.maximum across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1400 = stablehlo.broadcast_in_dim %v1399, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v1401 = stablehlo.subtract %v1397, %v1400 : tensor<256x197x197xf32>
    %v1402 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1403 = stablehlo.exponential %v1401 : tensor<256x197x197xf32>
    %v1404 = stablehlo.reduce(%v1403 init: %v1402) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1405 = stablehlo.broadcast_in_dim %v1404, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v1406 = stablehlo.divide %v1403, %v1405 : tensor<256x197x197xf32>
    %v1407 = stablehlo.reshape %v1406 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v1408 = stablehlo.reshape %v1407 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v1409 = stablehlo.reshape %v1387 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1410 = stablehlo.dot_general %v1408, %v1409, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v1411 = stablehlo.reshape %v1410 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1412 = stablehlo.reshape %v1411 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1413 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1414 = stablehlo.pad %v1412, %v1413, low = [0, 0, 64], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x384xf32>
    %v1415 = stablehlo.reshape %v1414 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v1416 = stablehlo.add %v1378, %v1415 : tensor<256x75648xf32>
    %v1417 = stablehlo.reshape %v1331 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1418 = stablehlo.slice %v1417 [0:256, 0:197, 128:192] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v1419 = stablehlo.reshape %v1418 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1420 = stablehlo.reshape %v1336 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1421 = stablehlo.slice %v1420 [0:256, 0:197, 128:192] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v1422 = stablehlo.reshape %v1421 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1423 = stablehlo.reshape %v1341 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1424 = stablehlo.slice %v1423 [0:256, 0:197, 128:192] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v1425 = stablehlo.reshape %v1424 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1426 = stablehlo.reshape %v1422 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1427 = stablehlo.transpose %v1426, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v1428 = stablehlo.reshape %v1427 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v1429 = stablehlo.reshape %v1419 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1430 = stablehlo.reshape %v1428 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v1431 = stablehlo.dot_general %v1429, %v1430, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v1432 = stablehlo.reshape %v1431 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v1433 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v1434 = stablehlo.multiply %v1432, %v1433 : tensor<256x38809xf32>
    %v1435 = stablehlo.reshape %v1434 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v1436 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v1437 = stablehlo.reduce(%v1435 init: %v1436) applies stablehlo.maximum across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1438 = stablehlo.broadcast_in_dim %v1437, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v1439 = stablehlo.subtract %v1435, %v1438 : tensor<256x197x197xf32>
    %v1440 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1441 = stablehlo.exponential %v1439 : tensor<256x197x197xf32>
    %v1442 = stablehlo.reduce(%v1441 init: %v1440) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1443 = stablehlo.broadcast_in_dim %v1442, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v1444 = stablehlo.divide %v1441, %v1443 : tensor<256x197x197xf32>
    %v1445 = stablehlo.reshape %v1444 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v1446 = stablehlo.reshape %v1445 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v1447 = stablehlo.reshape %v1425 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1448 = stablehlo.dot_general %v1446, %v1447, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v1449 = stablehlo.reshape %v1448 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1450 = stablehlo.reshape %v1449 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1451 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1452 = stablehlo.pad %v1450, %v1451, low = [0, 0, 128], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x384xf32>
    %v1453 = stablehlo.reshape %v1452 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v1454 = stablehlo.add %v1416, %v1453 : tensor<256x75648xf32>
    %v1455 = stablehlo.reshape %v1331 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1456 = stablehlo.slice %v1455 [0:256, 0:197, 192:256] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v1457 = stablehlo.reshape %v1456 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1458 = stablehlo.reshape %v1336 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1459 = stablehlo.slice %v1458 [0:256, 0:197, 192:256] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v1460 = stablehlo.reshape %v1459 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1461 = stablehlo.reshape %v1341 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1462 = stablehlo.slice %v1461 [0:256, 0:197, 192:256] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v1463 = stablehlo.reshape %v1462 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1464 = stablehlo.reshape %v1460 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1465 = stablehlo.transpose %v1464, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v1466 = stablehlo.reshape %v1465 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v1467 = stablehlo.reshape %v1457 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1468 = stablehlo.reshape %v1466 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v1469 = stablehlo.dot_general %v1467, %v1468, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v1470 = stablehlo.reshape %v1469 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v1471 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v1472 = stablehlo.multiply %v1470, %v1471 : tensor<256x38809xf32>
    %v1473 = stablehlo.reshape %v1472 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v1474 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v1475 = stablehlo.reduce(%v1473 init: %v1474) applies stablehlo.maximum across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1476 = stablehlo.broadcast_in_dim %v1475, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v1477 = stablehlo.subtract %v1473, %v1476 : tensor<256x197x197xf32>
    %v1478 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1479 = stablehlo.exponential %v1477 : tensor<256x197x197xf32>
    %v1480 = stablehlo.reduce(%v1479 init: %v1478) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1481 = stablehlo.broadcast_in_dim %v1480, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v1482 = stablehlo.divide %v1479, %v1481 : tensor<256x197x197xf32>
    %v1483 = stablehlo.reshape %v1482 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v1484 = stablehlo.reshape %v1483 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v1485 = stablehlo.reshape %v1463 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1486 = stablehlo.dot_general %v1484, %v1485, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v1487 = stablehlo.reshape %v1486 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1488 = stablehlo.reshape %v1487 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1489 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1490 = stablehlo.pad %v1488, %v1489, low = [0, 0, 192], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x384xf32>
    %v1491 = stablehlo.reshape %v1490 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v1492 = stablehlo.add %v1454, %v1491 : tensor<256x75648xf32>
    %v1493 = stablehlo.reshape %v1331 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1494 = stablehlo.slice %v1493 [0:256, 0:197, 256:320] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v1495 = stablehlo.reshape %v1494 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1496 = stablehlo.reshape %v1336 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1497 = stablehlo.slice %v1496 [0:256, 0:197, 256:320] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v1498 = stablehlo.reshape %v1497 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1499 = stablehlo.reshape %v1341 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1500 = stablehlo.slice %v1499 [0:256, 0:197, 256:320] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
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
    %v1512 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v1513 = stablehlo.reduce(%v1511 init: %v1512) applies stablehlo.maximum across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1514 = stablehlo.broadcast_in_dim %v1513, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v1515 = stablehlo.subtract %v1511, %v1514 : tensor<256x197x197xf32>
    %v1516 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1517 = stablehlo.exponential %v1515 : tensor<256x197x197xf32>
    %v1518 = stablehlo.reduce(%v1517 init: %v1516) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1519 = stablehlo.broadcast_in_dim %v1518, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v1520 = stablehlo.divide %v1517, %v1519 : tensor<256x197x197xf32>
    %v1521 = stablehlo.reshape %v1520 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v1522 = stablehlo.reshape %v1521 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v1523 = stablehlo.reshape %v1501 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1524 = stablehlo.dot_general %v1522, %v1523, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v1525 = stablehlo.reshape %v1524 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1526 = stablehlo.reshape %v1525 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1527 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1528 = stablehlo.pad %v1526, %v1527, low = [0, 0, 256], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x384xf32>
    %v1529 = stablehlo.reshape %v1528 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v1530 = stablehlo.add %v1492, %v1529 : tensor<256x75648xf32>
    %v1531 = stablehlo.reshape %v1331 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1532 = stablehlo.slice %v1531 [0:256, 0:197, 320:384] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v1533 = stablehlo.reshape %v1532 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1534 = stablehlo.reshape %v1336 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1535 = stablehlo.slice %v1534 [0:256, 0:197, 320:384] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v1536 = stablehlo.reshape %v1535 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1537 = stablehlo.reshape %v1341 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1538 = stablehlo.slice %v1537 [0:256, 0:197, 320:384] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v1539 = stablehlo.reshape %v1538 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1540 = stablehlo.reshape %v1536 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1541 = stablehlo.transpose %v1540, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v1542 = stablehlo.reshape %v1541 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v1543 = stablehlo.reshape %v1533 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1544 = stablehlo.reshape %v1542 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v1545 = stablehlo.dot_general %v1543, %v1544, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v1546 = stablehlo.reshape %v1545 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v1547 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v1548 = stablehlo.multiply %v1546, %v1547 : tensor<256x38809xf32>
    %v1549 = stablehlo.reshape %v1548 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v1550 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v1551 = stablehlo.reduce(%v1549 init: %v1550) applies stablehlo.maximum across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1552 = stablehlo.broadcast_in_dim %v1551, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v1553 = stablehlo.subtract %v1549, %v1552 : tensor<256x197x197xf32>
    %v1554 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1555 = stablehlo.exponential %v1553 : tensor<256x197x197xf32>
    %v1556 = stablehlo.reduce(%v1555 init: %v1554) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1557 = stablehlo.broadcast_in_dim %v1556, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v1558 = stablehlo.divide %v1555, %v1557 : tensor<256x197x197xf32>
    %v1559 = stablehlo.reshape %v1558 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v1560 = stablehlo.reshape %v1559 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v1561 = stablehlo.reshape %v1539 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1562 = stablehlo.dot_general %v1560, %v1561, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v1563 = stablehlo.reshape %v1562 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1564 = stablehlo.reshape %v1563 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1565 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1566 = stablehlo.pad %v1564, %v1565, low = [0, 0, 320], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x384xf32>
    %v1567 = stablehlo.reshape %v1566 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v1568 = stablehlo.add %v1530, %v1567 : tensor<256x75648xf32>
    %v1569 = stablehlo.reshape %v1568 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1570 = stablehlo.dot_general %v1569, %b4_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x384xf32>, tensor<384x384xf32>) -> tensor<256x197x384xf32>
    %v1571 = stablehlo.broadcast_in_dim %b4_bo, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v1572 = stablehlo.add %v1570, %v1571 : tensor<256x197x384xf32>
    %v1573 = stablehlo.reshape %v1572 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v1574 = stablehlo.add %v1298, %v1573 : tensor<256x75648xf32>
    %v1575 = stablehlo.reshape %v1574 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1576 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1577 = stablehlo.constant dense<384.0> : tensor<256x197x384xf32>
    %v1578 = stablehlo.constant dense<1.0e-6> : tensor<256x197x384xf32>
    %v1579 = stablehlo.reduce(%v1575 init: %v1576) applies stablehlo.add across dimensions = [2] : (tensor<256x197x384xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1580 = stablehlo.broadcast_in_dim %v1579, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x384xf32>
    %v1581 = stablehlo.divide %v1580, %v1577 : tensor<256x197x384xf32>
    %v1582 = stablehlo.subtract %v1575, %v1581 : tensor<256x197x384xf32>
    %v1583 = stablehlo.multiply %v1582, %v1582 : tensor<256x197x384xf32>
    %v1584 = stablehlo.reduce(%v1583 init: %v1576) applies stablehlo.add across dimensions = [2] : (tensor<256x197x384xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1585 = stablehlo.broadcast_in_dim %v1584, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x384xf32>
    %v1586 = stablehlo.divide %v1585, %v1577 : tensor<256x197x384xf32>
    %v1587 = stablehlo.add %v1586, %v1578 : tensor<256x197x384xf32>
    %v1588 = stablehlo.rsqrt %v1587 : tensor<256x197x384xf32>
    %v1589 = stablehlo.multiply %v1582, %v1588 : tensor<256x197x384xf32>
    %v1590 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<256x197x384xf32>
    %v1591 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<256x197x384xf32>
    %v1592 = stablehlo.multiply %v1589, %v1590 : tensor<256x197x384xf32>
    %v1593 = stablehlo.add %v1592, %v1591 : tensor<256x197x384xf32>
    %v1594 = stablehlo.reshape %v1593 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v1595 = stablehlo.reshape %v1594 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1596 = stablehlo.broadcast_in_dim %b4_g2, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v1597 = stablehlo.multiply %v1595, %v1596 : tensor<256x197x384xf32>
    %v1598 = stablehlo.reshape %v1597 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v1599 = stablehlo.reshape %v1598 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1600 = stablehlo.broadcast_in_dim %b4_bt2, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v1601 = stablehlo.add %v1599, %v1600 : tensor<256x197x384xf32>
    %v1602 = stablehlo.reshape %v1601 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v1603 = stablehlo.reshape %v1602 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1604 = stablehlo.dot_general %v1603, %b4_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x384xf32>, tensor<384x1536xf32>) -> tensor<256x197x1536xf32>
    %v1605 = stablehlo.broadcast_in_dim %b4_bfc1, dims = [2] : (tensor<1536xf32>) -> tensor<256x197x1536xf32>
    %v1606 = stablehlo.add %v1604, %v1605 : tensor<256x197x1536xf32>
    %v1607 = stablehlo.reshape %v1606 : (tensor<256x197x1536xf32>) -> tensor<256x302592xf32>
    %v1608 = stablehlo.constant dense<0.5> : tensor<256x302592xf32>
    %v1609 = stablehlo.multiply %v1608, %v1607 : tensor<256x302592xf32>
    %v1610 = stablehlo.negate %v1607 : tensor<256x302592xf32>
    %v1611 = stablehlo.constant dense<0.7071067811865476> : tensor<256x302592xf32>
    %v1612 = stablehlo.multiply %v1610, %v1611 : tensor<256x302592xf32>
    %v1613 = chlo.erfc %v1612 : tensor<256x302592xf32> -> tensor<256x302592xf32>
    %v1614 = stablehlo.multiply %v1609, %v1613 : tensor<256x302592xf32>
    %v1615 = stablehlo.reshape %v1614 : (tensor<256x302592xf32>) -> tensor<256x197x1536xf32>
    %v1616 = stablehlo.dot_general %v1615, %b4_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x1536xf32>, tensor<1536x384xf32>) -> tensor<256x197x384xf32>
    %v1617 = stablehlo.broadcast_in_dim %b4_bfc2, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v1618 = stablehlo.add %v1616, %v1617 : tensor<256x197x384xf32>
    %v1619 = stablehlo.reshape %v1618 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v1620 = stablehlo.add %v1574, %v1619 : tensor<256x75648xf32>
    %v1621 = stablehlo.reshape %v1620 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1622 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1623 = stablehlo.constant dense<384.0> : tensor<256x197x384xf32>
    %v1624 = stablehlo.constant dense<1.0e-6> : tensor<256x197x384xf32>
    %v1625 = stablehlo.reduce(%v1621 init: %v1622) applies stablehlo.add across dimensions = [2] : (tensor<256x197x384xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1626 = stablehlo.broadcast_in_dim %v1625, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x384xf32>
    %v1627 = stablehlo.divide %v1626, %v1623 : tensor<256x197x384xf32>
    %v1628 = stablehlo.subtract %v1621, %v1627 : tensor<256x197x384xf32>
    %v1629 = stablehlo.multiply %v1628, %v1628 : tensor<256x197x384xf32>
    %v1630 = stablehlo.reduce(%v1629 init: %v1622) applies stablehlo.add across dimensions = [2] : (tensor<256x197x384xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1631 = stablehlo.broadcast_in_dim %v1630, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x384xf32>
    %v1632 = stablehlo.divide %v1631, %v1623 : tensor<256x197x384xf32>
    %v1633 = stablehlo.add %v1632, %v1624 : tensor<256x197x384xf32>
    %v1634 = stablehlo.rsqrt %v1633 : tensor<256x197x384xf32>
    %v1635 = stablehlo.multiply %v1628, %v1634 : tensor<256x197x384xf32>
    %v1636 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<256x197x384xf32>
    %v1637 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<256x197x384xf32>
    %v1638 = stablehlo.multiply %v1635, %v1636 : tensor<256x197x384xf32>
    %v1639 = stablehlo.add %v1638, %v1637 : tensor<256x197x384xf32>
    %v1640 = stablehlo.reshape %v1639 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v1641 = stablehlo.reshape %v1640 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1642 = stablehlo.broadcast_in_dim %b5_g1, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v1643 = stablehlo.multiply %v1641, %v1642 : tensor<256x197x384xf32>
    %v1644 = stablehlo.reshape %v1643 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v1645 = stablehlo.reshape %v1644 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1646 = stablehlo.broadcast_in_dim %b5_bt1, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v1647 = stablehlo.add %v1645, %v1646 : tensor<256x197x384xf32>
    %v1648 = stablehlo.reshape %v1647 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v1649 = stablehlo.reshape %v1648 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1650 = stablehlo.dot_general %v1649, %b5_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x384xf32>, tensor<384x384xf32>) -> tensor<256x197x384xf32>
    %v1651 = stablehlo.broadcast_in_dim %b5_bq, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v1652 = stablehlo.add %v1650, %v1651 : tensor<256x197x384xf32>
    %v1653 = stablehlo.reshape %v1652 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v1654 = stablehlo.reshape %v1648 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1655 = stablehlo.dot_general %v1654, %b5_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x384xf32>, tensor<384x384xf32>) -> tensor<256x197x384xf32>
    %v1656 = stablehlo.broadcast_in_dim %b5_bk, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v1657 = stablehlo.add %v1655, %v1656 : tensor<256x197x384xf32>
    %v1658 = stablehlo.reshape %v1657 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v1659 = stablehlo.reshape %v1648 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1660 = stablehlo.dot_general %v1659, %b5_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x384xf32>, tensor<384x384xf32>) -> tensor<256x197x384xf32>
    %v1661 = stablehlo.broadcast_in_dim %b5_bv, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v1662 = stablehlo.add %v1660, %v1661 : tensor<256x197x384xf32>
    %v1663 = stablehlo.reshape %v1662 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v1664 = stablehlo.reshape %v1653 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1665 = stablehlo.slice %v1664 [0:256, 0:197, 0:64] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v1666 = stablehlo.reshape %v1665 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1667 = stablehlo.reshape %v1658 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1668 = stablehlo.slice %v1667 [0:256, 0:197, 0:64] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v1669 = stablehlo.reshape %v1668 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1670 = stablehlo.reshape %v1663 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1671 = stablehlo.slice %v1670 [0:256, 0:197, 0:64] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v1672 = stablehlo.reshape %v1671 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1673 = stablehlo.reshape %v1669 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1674 = stablehlo.transpose %v1673, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v1675 = stablehlo.reshape %v1674 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v1676 = stablehlo.reshape %v1666 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1677 = stablehlo.reshape %v1675 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v1678 = stablehlo.dot_general %v1676, %v1677, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v1679 = stablehlo.reshape %v1678 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v1680 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v1681 = stablehlo.multiply %v1679, %v1680 : tensor<256x38809xf32>
    %v1682 = stablehlo.reshape %v1681 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v1683 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v1684 = stablehlo.reduce(%v1682 init: %v1683) applies stablehlo.maximum across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1685 = stablehlo.broadcast_in_dim %v1684, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v1686 = stablehlo.subtract %v1682, %v1685 : tensor<256x197x197xf32>
    %v1687 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1688 = stablehlo.exponential %v1686 : tensor<256x197x197xf32>
    %v1689 = stablehlo.reduce(%v1688 init: %v1687) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1690 = stablehlo.broadcast_in_dim %v1689, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v1691 = stablehlo.divide %v1688, %v1690 : tensor<256x197x197xf32>
    %v1692 = stablehlo.reshape %v1691 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v1693 = stablehlo.reshape %v1692 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v1694 = stablehlo.reshape %v1672 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1695 = stablehlo.dot_general %v1693, %v1694, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v1696 = stablehlo.reshape %v1695 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1697 = stablehlo.reshape %v1696 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1698 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1699 = stablehlo.pad %v1697, %v1698, low = [0, 0, 0], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x384xf32>
    %v1700 = stablehlo.reshape %v1699 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v1701 = stablehlo.reshape %v1653 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1702 = stablehlo.slice %v1701 [0:256, 0:197, 64:128] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v1703 = stablehlo.reshape %v1702 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1704 = stablehlo.reshape %v1658 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1705 = stablehlo.slice %v1704 [0:256, 0:197, 64:128] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v1706 = stablehlo.reshape %v1705 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1707 = stablehlo.reshape %v1663 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1708 = stablehlo.slice %v1707 [0:256, 0:197, 64:128] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v1709 = stablehlo.reshape %v1708 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1710 = stablehlo.reshape %v1706 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1711 = stablehlo.transpose %v1710, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v1712 = stablehlo.reshape %v1711 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v1713 = stablehlo.reshape %v1703 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1714 = stablehlo.reshape %v1712 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v1715 = stablehlo.dot_general %v1713, %v1714, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v1716 = stablehlo.reshape %v1715 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v1717 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v1718 = stablehlo.multiply %v1716, %v1717 : tensor<256x38809xf32>
    %v1719 = stablehlo.reshape %v1718 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v1720 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v1721 = stablehlo.reduce(%v1719 init: %v1720) applies stablehlo.maximum across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1722 = stablehlo.broadcast_in_dim %v1721, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v1723 = stablehlo.subtract %v1719, %v1722 : tensor<256x197x197xf32>
    %v1724 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1725 = stablehlo.exponential %v1723 : tensor<256x197x197xf32>
    %v1726 = stablehlo.reduce(%v1725 init: %v1724) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1727 = stablehlo.broadcast_in_dim %v1726, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v1728 = stablehlo.divide %v1725, %v1727 : tensor<256x197x197xf32>
    %v1729 = stablehlo.reshape %v1728 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v1730 = stablehlo.reshape %v1729 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v1731 = stablehlo.reshape %v1709 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1732 = stablehlo.dot_general %v1730, %v1731, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v1733 = stablehlo.reshape %v1732 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1734 = stablehlo.reshape %v1733 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1735 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1736 = stablehlo.pad %v1734, %v1735, low = [0, 0, 64], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x384xf32>
    %v1737 = stablehlo.reshape %v1736 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v1738 = stablehlo.add %v1700, %v1737 : tensor<256x75648xf32>
    %v1739 = stablehlo.reshape %v1653 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1740 = stablehlo.slice %v1739 [0:256, 0:197, 128:192] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v1741 = stablehlo.reshape %v1740 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1742 = stablehlo.reshape %v1658 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1743 = stablehlo.slice %v1742 [0:256, 0:197, 128:192] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v1744 = stablehlo.reshape %v1743 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1745 = stablehlo.reshape %v1663 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1746 = stablehlo.slice %v1745 [0:256, 0:197, 128:192] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v1747 = stablehlo.reshape %v1746 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1748 = stablehlo.reshape %v1744 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1749 = stablehlo.transpose %v1748, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v1750 = stablehlo.reshape %v1749 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v1751 = stablehlo.reshape %v1741 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1752 = stablehlo.reshape %v1750 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v1753 = stablehlo.dot_general %v1751, %v1752, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v1754 = stablehlo.reshape %v1753 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v1755 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v1756 = stablehlo.multiply %v1754, %v1755 : tensor<256x38809xf32>
    %v1757 = stablehlo.reshape %v1756 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v1758 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v1759 = stablehlo.reduce(%v1757 init: %v1758) applies stablehlo.maximum across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1760 = stablehlo.broadcast_in_dim %v1759, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v1761 = stablehlo.subtract %v1757, %v1760 : tensor<256x197x197xf32>
    %v1762 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1763 = stablehlo.exponential %v1761 : tensor<256x197x197xf32>
    %v1764 = stablehlo.reduce(%v1763 init: %v1762) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1765 = stablehlo.broadcast_in_dim %v1764, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v1766 = stablehlo.divide %v1763, %v1765 : tensor<256x197x197xf32>
    %v1767 = stablehlo.reshape %v1766 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v1768 = stablehlo.reshape %v1767 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v1769 = stablehlo.reshape %v1747 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1770 = stablehlo.dot_general %v1768, %v1769, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v1771 = stablehlo.reshape %v1770 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1772 = stablehlo.reshape %v1771 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1773 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1774 = stablehlo.pad %v1772, %v1773, low = [0, 0, 128], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x384xf32>
    %v1775 = stablehlo.reshape %v1774 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v1776 = stablehlo.add %v1738, %v1775 : tensor<256x75648xf32>
    %v1777 = stablehlo.reshape %v1653 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1778 = stablehlo.slice %v1777 [0:256, 0:197, 192:256] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v1779 = stablehlo.reshape %v1778 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1780 = stablehlo.reshape %v1658 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1781 = stablehlo.slice %v1780 [0:256, 0:197, 192:256] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v1782 = stablehlo.reshape %v1781 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1783 = stablehlo.reshape %v1663 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1784 = stablehlo.slice %v1783 [0:256, 0:197, 192:256] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v1785 = stablehlo.reshape %v1784 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1786 = stablehlo.reshape %v1782 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1787 = stablehlo.transpose %v1786, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v1788 = stablehlo.reshape %v1787 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v1789 = stablehlo.reshape %v1779 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1790 = stablehlo.reshape %v1788 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v1791 = stablehlo.dot_general %v1789, %v1790, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v1792 = stablehlo.reshape %v1791 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v1793 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v1794 = stablehlo.multiply %v1792, %v1793 : tensor<256x38809xf32>
    %v1795 = stablehlo.reshape %v1794 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v1796 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v1797 = stablehlo.reduce(%v1795 init: %v1796) applies stablehlo.maximum across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1798 = stablehlo.broadcast_in_dim %v1797, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v1799 = stablehlo.subtract %v1795, %v1798 : tensor<256x197x197xf32>
    %v1800 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1801 = stablehlo.exponential %v1799 : tensor<256x197x197xf32>
    %v1802 = stablehlo.reduce(%v1801 init: %v1800) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1803 = stablehlo.broadcast_in_dim %v1802, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v1804 = stablehlo.divide %v1801, %v1803 : tensor<256x197x197xf32>
    %v1805 = stablehlo.reshape %v1804 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v1806 = stablehlo.reshape %v1805 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v1807 = stablehlo.reshape %v1785 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1808 = stablehlo.dot_general %v1806, %v1807, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v1809 = stablehlo.reshape %v1808 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1810 = stablehlo.reshape %v1809 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1811 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1812 = stablehlo.pad %v1810, %v1811, low = [0, 0, 192], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x384xf32>
    %v1813 = stablehlo.reshape %v1812 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v1814 = stablehlo.add %v1776, %v1813 : tensor<256x75648xf32>
    %v1815 = stablehlo.reshape %v1653 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1816 = stablehlo.slice %v1815 [0:256, 0:197, 256:320] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v1817 = stablehlo.reshape %v1816 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1818 = stablehlo.reshape %v1658 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1819 = stablehlo.slice %v1818 [0:256, 0:197, 256:320] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v1820 = stablehlo.reshape %v1819 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1821 = stablehlo.reshape %v1663 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1822 = stablehlo.slice %v1821 [0:256, 0:197, 256:320] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v1823 = stablehlo.reshape %v1822 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1824 = stablehlo.reshape %v1820 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1825 = stablehlo.transpose %v1824, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v1826 = stablehlo.reshape %v1825 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v1827 = stablehlo.reshape %v1817 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1828 = stablehlo.reshape %v1826 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v1829 = stablehlo.dot_general %v1827, %v1828, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v1830 = stablehlo.reshape %v1829 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v1831 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v1832 = stablehlo.multiply %v1830, %v1831 : tensor<256x38809xf32>
    %v1833 = stablehlo.reshape %v1832 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v1834 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v1835 = stablehlo.reduce(%v1833 init: %v1834) applies stablehlo.maximum across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1836 = stablehlo.broadcast_in_dim %v1835, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v1837 = stablehlo.subtract %v1833, %v1836 : tensor<256x197x197xf32>
    %v1838 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1839 = stablehlo.exponential %v1837 : tensor<256x197x197xf32>
    %v1840 = stablehlo.reduce(%v1839 init: %v1838) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1841 = stablehlo.broadcast_in_dim %v1840, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v1842 = stablehlo.divide %v1839, %v1841 : tensor<256x197x197xf32>
    %v1843 = stablehlo.reshape %v1842 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v1844 = stablehlo.reshape %v1843 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v1845 = stablehlo.reshape %v1823 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1846 = stablehlo.dot_general %v1844, %v1845, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v1847 = stablehlo.reshape %v1846 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1848 = stablehlo.reshape %v1847 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1849 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1850 = stablehlo.pad %v1848, %v1849, low = [0, 0, 256], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x384xf32>
    %v1851 = stablehlo.reshape %v1850 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v1852 = stablehlo.add %v1814, %v1851 : tensor<256x75648xf32>
    %v1853 = stablehlo.reshape %v1653 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1854 = stablehlo.slice %v1853 [0:256, 0:197, 320:384] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v1855 = stablehlo.reshape %v1854 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1856 = stablehlo.reshape %v1658 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1857 = stablehlo.slice %v1856 [0:256, 0:197, 320:384] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v1858 = stablehlo.reshape %v1857 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1859 = stablehlo.reshape %v1663 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1860 = stablehlo.slice %v1859 [0:256, 0:197, 320:384] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v1861 = stablehlo.reshape %v1860 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1862 = stablehlo.reshape %v1858 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1863 = stablehlo.transpose %v1862, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v1864 = stablehlo.reshape %v1863 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v1865 = stablehlo.reshape %v1855 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1866 = stablehlo.reshape %v1864 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v1867 = stablehlo.dot_general %v1865, %v1866, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v1868 = stablehlo.reshape %v1867 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v1869 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v1870 = stablehlo.multiply %v1868, %v1869 : tensor<256x38809xf32>
    %v1871 = stablehlo.reshape %v1870 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v1872 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v1873 = stablehlo.reduce(%v1871 init: %v1872) applies stablehlo.maximum across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1874 = stablehlo.broadcast_in_dim %v1873, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v1875 = stablehlo.subtract %v1871, %v1874 : tensor<256x197x197xf32>
    %v1876 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1877 = stablehlo.exponential %v1875 : tensor<256x197x197xf32>
    %v1878 = stablehlo.reduce(%v1877 init: %v1876) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1879 = stablehlo.broadcast_in_dim %v1878, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v1880 = stablehlo.divide %v1877, %v1879 : tensor<256x197x197xf32>
    %v1881 = stablehlo.reshape %v1880 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v1882 = stablehlo.reshape %v1881 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v1883 = stablehlo.reshape %v1861 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1884 = stablehlo.dot_general %v1882, %v1883, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v1885 = stablehlo.reshape %v1884 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1886 = stablehlo.reshape %v1885 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1887 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1888 = stablehlo.pad %v1886, %v1887, low = [0, 0, 320], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x384xf32>
    %v1889 = stablehlo.reshape %v1888 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v1890 = stablehlo.add %v1852, %v1889 : tensor<256x75648xf32>
    %v1891 = stablehlo.reshape %v1890 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1892 = stablehlo.dot_general %v1891, %b5_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x384xf32>, tensor<384x384xf32>) -> tensor<256x197x384xf32>
    %v1893 = stablehlo.broadcast_in_dim %b5_bo, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v1894 = stablehlo.add %v1892, %v1893 : tensor<256x197x384xf32>
    %v1895 = stablehlo.reshape %v1894 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v1896 = stablehlo.add %v1620, %v1895 : tensor<256x75648xf32>
    %v1897 = stablehlo.reshape %v1896 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1898 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1899 = stablehlo.constant dense<384.0> : tensor<256x197x384xf32>
    %v1900 = stablehlo.constant dense<1.0e-6> : tensor<256x197x384xf32>
    %v1901 = stablehlo.reduce(%v1897 init: %v1898) applies stablehlo.add across dimensions = [2] : (tensor<256x197x384xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1902 = stablehlo.broadcast_in_dim %v1901, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x384xf32>
    %v1903 = stablehlo.divide %v1902, %v1899 : tensor<256x197x384xf32>
    %v1904 = stablehlo.subtract %v1897, %v1903 : tensor<256x197x384xf32>
    %v1905 = stablehlo.multiply %v1904, %v1904 : tensor<256x197x384xf32>
    %v1906 = stablehlo.reduce(%v1905 init: %v1898) applies stablehlo.add across dimensions = [2] : (tensor<256x197x384xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1907 = stablehlo.broadcast_in_dim %v1906, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x384xf32>
    %v1908 = stablehlo.divide %v1907, %v1899 : tensor<256x197x384xf32>
    %v1909 = stablehlo.add %v1908, %v1900 : tensor<256x197x384xf32>
    %v1910 = stablehlo.rsqrt %v1909 : tensor<256x197x384xf32>
    %v1911 = stablehlo.multiply %v1904, %v1910 : tensor<256x197x384xf32>
    %v1912 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<256x197x384xf32>
    %v1913 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<256x197x384xf32>
    %v1914 = stablehlo.multiply %v1911, %v1912 : tensor<256x197x384xf32>
    %v1915 = stablehlo.add %v1914, %v1913 : tensor<256x197x384xf32>
    %v1916 = stablehlo.reshape %v1915 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v1917 = stablehlo.reshape %v1916 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1918 = stablehlo.broadcast_in_dim %b5_g2, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v1919 = stablehlo.multiply %v1917, %v1918 : tensor<256x197x384xf32>
    %v1920 = stablehlo.reshape %v1919 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v1921 = stablehlo.reshape %v1920 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1922 = stablehlo.broadcast_in_dim %b5_bt2, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v1923 = stablehlo.add %v1921, %v1922 : tensor<256x197x384xf32>
    %v1924 = stablehlo.reshape %v1923 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v1925 = stablehlo.reshape %v1924 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1926 = stablehlo.dot_general %v1925, %b5_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x384xf32>, tensor<384x1536xf32>) -> tensor<256x197x1536xf32>
    %v1927 = stablehlo.broadcast_in_dim %b5_bfc1, dims = [2] : (tensor<1536xf32>) -> tensor<256x197x1536xf32>
    %v1928 = stablehlo.add %v1926, %v1927 : tensor<256x197x1536xf32>
    %v1929 = stablehlo.reshape %v1928 : (tensor<256x197x1536xf32>) -> tensor<256x302592xf32>
    %v1930 = stablehlo.constant dense<0.5> : tensor<256x302592xf32>
    %v1931 = stablehlo.multiply %v1930, %v1929 : tensor<256x302592xf32>
    %v1932 = stablehlo.negate %v1929 : tensor<256x302592xf32>
    %v1933 = stablehlo.constant dense<0.7071067811865476> : tensor<256x302592xf32>
    %v1934 = stablehlo.multiply %v1932, %v1933 : tensor<256x302592xf32>
    %v1935 = chlo.erfc %v1934 : tensor<256x302592xf32> -> tensor<256x302592xf32>
    %v1936 = stablehlo.multiply %v1931, %v1935 : tensor<256x302592xf32>
    %v1937 = stablehlo.reshape %v1936 : (tensor<256x302592xf32>) -> tensor<256x197x1536xf32>
    %v1938 = stablehlo.dot_general %v1937, %b5_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x1536xf32>, tensor<1536x384xf32>) -> tensor<256x197x384xf32>
    %v1939 = stablehlo.broadcast_in_dim %b5_bfc2, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v1940 = stablehlo.add %v1938, %v1939 : tensor<256x197x384xf32>
    %v1941 = stablehlo.reshape %v1940 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v1942 = stablehlo.add %v1896, %v1941 : tensor<256x75648xf32>
    %v1943 = stablehlo.reshape %v1942 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1944 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1945 = stablehlo.constant dense<384.0> : tensor<256x197x384xf32>
    %v1946 = stablehlo.constant dense<1.0e-6> : tensor<256x197x384xf32>
    %v1947 = stablehlo.reduce(%v1943 init: %v1944) applies stablehlo.add across dimensions = [2] : (tensor<256x197x384xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1948 = stablehlo.broadcast_in_dim %v1947, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x384xf32>
    %v1949 = stablehlo.divide %v1948, %v1945 : tensor<256x197x384xf32>
    %v1950 = stablehlo.subtract %v1943, %v1949 : tensor<256x197x384xf32>
    %v1951 = stablehlo.multiply %v1950, %v1950 : tensor<256x197x384xf32>
    %v1952 = stablehlo.reduce(%v1951 init: %v1944) applies stablehlo.add across dimensions = [2] : (tensor<256x197x384xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v1953 = stablehlo.broadcast_in_dim %v1952, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x384xf32>
    %v1954 = stablehlo.divide %v1953, %v1945 : tensor<256x197x384xf32>
    %v1955 = stablehlo.add %v1954, %v1946 : tensor<256x197x384xf32>
    %v1956 = stablehlo.rsqrt %v1955 : tensor<256x197x384xf32>
    %v1957 = stablehlo.multiply %v1950, %v1956 : tensor<256x197x384xf32>
    %v1958 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<256x197x384xf32>
    %v1959 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<256x197x384xf32>
    %v1960 = stablehlo.multiply %v1957, %v1958 : tensor<256x197x384xf32>
    %v1961 = stablehlo.add %v1960, %v1959 : tensor<256x197x384xf32>
    %v1962 = stablehlo.reshape %v1961 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v1963 = stablehlo.reshape %v1962 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1964 = stablehlo.broadcast_in_dim %b6_g1, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v1965 = stablehlo.multiply %v1963, %v1964 : tensor<256x197x384xf32>
    %v1966 = stablehlo.reshape %v1965 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v1967 = stablehlo.reshape %v1966 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1968 = stablehlo.broadcast_in_dim %b6_bt1, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v1969 = stablehlo.add %v1967, %v1968 : tensor<256x197x384xf32>
    %v1970 = stablehlo.reshape %v1969 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v1971 = stablehlo.reshape %v1970 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1972 = stablehlo.dot_general %v1971, %b6_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x384xf32>, tensor<384x384xf32>) -> tensor<256x197x384xf32>
    %v1973 = stablehlo.broadcast_in_dim %b6_bq, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v1974 = stablehlo.add %v1972, %v1973 : tensor<256x197x384xf32>
    %v1975 = stablehlo.reshape %v1974 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v1976 = stablehlo.reshape %v1970 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1977 = stablehlo.dot_general %v1976, %b6_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x384xf32>, tensor<384x384xf32>) -> tensor<256x197x384xf32>
    %v1978 = stablehlo.broadcast_in_dim %b6_bk, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v1979 = stablehlo.add %v1977, %v1978 : tensor<256x197x384xf32>
    %v1980 = stablehlo.reshape %v1979 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v1981 = stablehlo.reshape %v1970 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1982 = stablehlo.dot_general %v1981, %b6_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x384xf32>, tensor<384x384xf32>) -> tensor<256x197x384xf32>
    %v1983 = stablehlo.broadcast_in_dim %b6_bv, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v1984 = stablehlo.add %v1982, %v1983 : tensor<256x197x384xf32>
    %v1985 = stablehlo.reshape %v1984 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v1986 = stablehlo.reshape %v1975 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1987 = stablehlo.slice %v1986 [0:256, 0:197, 0:64] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v1988 = stablehlo.reshape %v1987 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1989 = stablehlo.reshape %v1980 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1990 = stablehlo.slice %v1989 [0:256, 0:197, 0:64] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v1991 = stablehlo.reshape %v1990 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1992 = stablehlo.reshape %v1985 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v1993 = stablehlo.slice %v1992 [0:256, 0:197, 0:64] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v1994 = stablehlo.reshape %v1993 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v1995 = stablehlo.reshape %v1991 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1996 = stablehlo.transpose %v1995, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v1997 = stablehlo.reshape %v1996 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v1998 = stablehlo.reshape %v1988 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v1999 = stablehlo.reshape %v1997 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v2000 = stablehlo.dot_general %v1998, %v1999, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v2001 = stablehlo.reshape %v2000 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v2002 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v2003 = stablehlo.multiply %v2001, %v2002 : tensor<256x38809xf32>
    %v2004 = stablehlo.reshape %v2003 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v2005 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v2006 = stablehlo.reduce(%v2004 init: %v2005) applies stablehlo.maximum across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v2007 = stablehlo.broadcast_in_dim %v2006, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v2008 = stablehlo.subtract %v2004, %v2007 : tensor<256x197x197xf32>
    %v2009 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2010 = stablehlo.exponential %v2008 : tensor<256x197x197xf32>
    %v2011 = stablehlo.reduce(%v2010 init: %v2009) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v2012 = stablehlo.broadcast_in_dim %v2011, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v2013 = stablehlo.divide %v2010, %v2012 : tensor<256x197x197xf32>
    %v2014 = stablehlo.reshape %v2013 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v2015 = stablehlo.reshape %v2014 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v2016 = stablehlo.reshape %v1994 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2017 = stablehlo.dot_general %v2015, %v2016, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v2018 = stablehlo.reshape %v2017 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2019 = stablehlo.reshape %v2018 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2020 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2021 = stablehlo.pad %v2019, %v2020, low = [0, 0, 0], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x384xf32>
    %v2022 = stablehlo.reshape %v2021 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v2023 = stablehlo.reshape %v1975 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2024 = stablehlo.slice %v2023 [0:256, 0:197, 64:128] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v2025 = stablehlo.reshape %v2024 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2026 = stablehlo.reshape %v1980 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2027 = stablehlo.slice %v2026 [0:256, 0:197, 64:128] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v2028 = stablehlo.reshape %v2027 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2029 = stablehlo.reshape %v1985 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2030 = stablehlo.slice %v2029 [0:256, 0:197, 64:128] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v2031 = stablehlo.reshape %v2030 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2032 = stablehlo.reshape %v2028 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2033 = stablehlo.transpose %v2032, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v2034 = stablehlo.reshape %v2033 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v2035 = stablehlo.reshape %v2025 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2036 = stablehlo.reshape %v2034 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v2037 = stablehlo.dot_general %v2035, %v2036, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v2038 = stablehlo.reshape %v2037 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v2039 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v2040 = stablehlo.multiply %v2038, %v2039 : tensor<256x38809xf32>
    %v2041 = stablehlo.reshape %v2040 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v2042 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v2043 = stablehlo.reduce(%v2041 init: %v2042) applies stablehlo.maximum across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v2044 = stablehlo.broadcast_in_dim %v2043, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v2045 = stablehlo.subtract %v2041, %v2044 : tensor<256x197x197xf32>
    %v2046 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2047 = stablehlo.exponential %v2045 : tensor<256x197x197xf32>
    %v2048 = stablehlo.reduce(%v2047 init: %v2046) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v2049 = stablehlo.broadcast_in_dim %v2048, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v2050 = stablehlo.divide %v2047, %v2049 : tensor<256x197x197xf32>
    %v2051 = stablehlo.reshape %v2050 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v2052 = stablehlo.reshape %v2051 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v2053 = stablehlo.reshape %v2031 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2054 = stablehlo.dot_general %v2052, %v2053, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v2055 = stablehlo.reshape %v2054 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2056 = stablehlo.reshape %v2055 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2057 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2058 = stablehlo.pad %v2056, %v2057, low = [0, 0, 64], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x384xf32>
    %v2059 = stablehlo.reshape %v2058 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v2060 = stablehlo.add %v2022, %v2059 : tensor<256x75648xf32>
    %v2061 = stablehlo.reshape %v1975 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2062 = stablehlo.slice %v2061 [0:256, 0:197, 128:192] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v2063 = stablehlo.reshape %v2062 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2064 = stablehlo.reshape %v1980 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2065 = stablehlo.slice %v2064 [0:256, 0:197, 128:192] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v2066 = stablehlo.reshape %v2065 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2067 = stablehlo.reshape %v1985 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2068 = stablehlo.slice %v2067 [0:256, 0:197, 128:192] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v2069 = stablehlo.reshape %v2068 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2070 = stablehlo.reshape %v2066 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2071 = stablehlo.transpose %v2070, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v2072 = stablehlo.reshape %v2071 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v2073 = stablehlo.reshape %v2063 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2074 = stablehlo.reshape %v2072 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v2075 = stablehlo.dot_general %v2073, %v2074, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v2076 = stablehlo.reshape %v2075 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v2077 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v2078 = stablehlo.multiply %v2076, %v2077 : tensor<256x38809xf32>
    %v2079 = stablehlo.reshape %v2078 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v2080 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v2081 = stablehlo.reduce(%v2079 init: %v2080) applies stablehlo.maximum across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v2082 = stablehlo.broadcast_in_dim %v2081, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v2083 = stablehlo.subtract %v2079, %v2082 : tensor<256x197x197xf32>
    %v2084 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2085 = stablehlo.exponential %v2083 : tensor<256x197x197xf32>
    %v2086 = stablehlo.reduce(%v2085 init: %v2084) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v2087 = stablehlo.broadcast_in_dim %v2086, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v2088 = stablehlo.divide %v2085, %v2087 : tensor<256x197x197xf32>
    %v2089 = stablehlo.reshape %v2088 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v2090 = stablehlo.reshape %v2089 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v2091 = stablehlo.reshape %v2069 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2092 = stablehlo.dot_general %v2090, %v2091, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v2093 = stablehlo.reshape %v2092 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2094 = stablehlo.reshape %v2093 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2095 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2096 = stablehlo.pad %v2094, %v2095, low = [0, 0, 128], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x384xf32>
    %v2097 = stablehlo.reshape %v2096 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v2098 = stablehlo.add %v2060, %v2097 : tensor<256x75648xf32>
    %v2099 = stablehlo.reshape %v1975 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2100 = stablehlo.slice %v2099 [0:256, 0:197, 192:256] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v2101 = stablehlo.reshape %v2100 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2102 = stablehlo.reshape %v1980 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2103 = stablehlo.slice %v2102 [0:256, 0:197, 192:256] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v2104 = stablehlo.reshape %v2103 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2105 = stablehlo.reshape %v1985 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2106 = stablehlo.slice %v2105 [0:256, 0:197, 192:256] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v2107 = stablehlo.reshape %v2106 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2108 = stablehlo.reshape %v2104 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2109 = stablehlo.transpose %v2108, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v2110 = stablehlo.reshape %v2109 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v2111 = stablehlo.reshape %v2101 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2112 = stablehlo.reshape %v2110 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v2113 = stablehlo.dot_general %v2111, %v2112, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v2114 = stablehlo.reshape %v2113 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v2115 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v2116 = stablehlo.multiply %v2114, %v2115 : tensor<256x38809xf32>
    %v2117 = stablehlo.reshape %v2116 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v2118 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v2119 = stablehlo.reduce(%v2117 init: %v2118) applies stablehlo.maximum across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v2120 = stablehlo.broadcast_in_dim %v2119, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v2121 = stablehlo.subtract %v2117, %v2120 : tensor<256x197x197xf32>
    %v2122 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2123 = stablehlo.exponential %v2121 : tensor<256x197x197xf32>
    %v2124 = stablehlo.reduce(%v2123 init: %v2122) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v2125 = stablehlo.broadcast_in_dim %v2124, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v2126 = stablehlo.divide %v2123, %v2125 : tensor<256x197x197xf32>
    %v2127 = stablehlo.reshape %v2126 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v2128 = stablehlo.reshape %v2127 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v2129 = stablehlo.reshape %v2107 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2130 = stablehlo.dot_general %v2128, %v2129, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v2131 = stablehlo.reshape %v2130 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2132 = stablehlo.reshape %v2131 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2133 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2134 = stablehlo.pad %v2132, %v2133, low = [0, 0, 192], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x384xf32>
    %v2135 = stablehlo.reshape %v2134 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v2136 = stablehlo.add %v2098, %v2135 : tensor<256x75648xf32>
    %v2137 = stablehlo.reshape %v1975 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2138 = stablehlo.slice %v2137 [0:256, 0:197, 256:320] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v2139 = stablehlo.reshape %v2138 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2140 = stablehlo.reshape %v1980 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2141 = stablehlo.slice %v2140 [0:256, 0:197, 256:320] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v2142 = stablehlo.reshape %v2141 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2143 = stablehlo.reshape %v1985 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2144 = stablehlo.slice %v2143 [0:256, 0:197, 256:320] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v2145 = stablehlo.reshape %v2144 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2146 = stablehlo.reshape %v2142 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2147 = stablehlo.transpose %v2146, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v2148 = stablehlo.reshape %v2147 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v2149 = stablehlo.reshape %v2139 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2150 = stablehlo.reshape %v2148 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v2151 = stablehlo.dot_general %v2149, %v2150, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v2152 = stablehlo.reshape %v2151 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v2153 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v2154 = stablehlo.multiply %v2152, %v2153 : tensor<256x38809xf32>
    %v2155 = stablehlo.reshape %v2154 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v2156 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v2157 = stablehlo.reduce(%v2155 init: %v2156) applies stablehlo.maximum across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v2158 = stablehlo.broadcast_in_dim %v2157, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v2159 = stablehlo.subtract %v2155, %v2158 : tensor<256x197x197xf32>
    %v2160 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2161 = stablehlo.exponential %v2159 : tensor<256x197x197xf32>
    %v2162 = stablehlo.reduce(%v2161 init: %v2160) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v2163 = stablehlo.broadcast_in_dim %v2162, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v2164 = stablehlo.divide %v2161, %v2163 : tensor<256x197x197xf32>
    %v2165 = stablehlo.reshape %v2164 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v2166 = stablehlo.reshape %v2165 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v2167 = stablehlo.reshape %v2145 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2168 = stablehlo.dot_general %v2166, %v2167, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v2169 = stablehlo.reshape %v2168 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2170 = stablehlo.reshape %v2169 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2171 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2172 = stablehlo.pad %v2170, %v2171, low = [0, 0, 256], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x384xf32>
    %v2173 = stablehlo.reshape %v2172 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v2174 = stablehlo.add %v2136, %v2173 : tensor<256x75648xf32>
    %v2175 = stablehlo.reshape %v1975 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2176 = stablehlo.slice %v2175 [0:256, 0:197, 320:384] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v2177 = stablehlo.reshape %v2176 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2178 = stablehlo.reshape %v1980 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2179 = stablehlo.slice %v2178 [0:256, 0:197, 320:384] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v2180 = stablehlo.reshape %v2179 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2181 = stablehlo.reshape %v1985 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2182 = stablehlo.slice %v2181 [0:256, 0:197, 320:384] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v2183 = stablehlo.reshape %v2182 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2184 = stablehlo.reshape %v2180 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2185 = stablehlo.transpose %v2184, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v2186 = stablehlo.reshape %v2185 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v2187 = stablehlo.reshape %v2177 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2188 = stablehlo.reshape %v2186 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v2189 = stablehlo.dot_general %v2187, %v2188, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v2190 = stablehlo.reshape %v2189 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v2191 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v2192 = stablehlo.multiply %v2190, %v2191 : tensor<256x38809xf32>
    %v2193 = stablehlo.reshape %v2192 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v2194 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v2195 = stablehlo.reduce(%v2193 init: %v2194) applies stablehlo.maximum across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v2196 = stablehlo.broadcast_in_dim %v2195, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v2197 = stablehlo.subtract %v2193, %v2196 : tensor<256x197x197xf32>
    %v2198 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2199 = stablehlo.exponential %v2197 : tensor<256x197x197xf32>
    %v2200 = stablehlo.reduce(%v2199 init: %v2198) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v2201 = stablehlo.broadcast_in_dim %v2200, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v2202 = stablehlo.divide %v2199, %v2201 : tensor<256x197x197xf32>
    %v2203 = stablehlo.reshape %v2202 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v2204 = stablehlo.reshape %v2203 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v2205 = stablehlo.reshape %v2183 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2206 = stablehlo.dot_general %v2204, %v2205, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v2207 = stablehlo.reshape %v2206 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2208 = stablehlo.reshape %v2207 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2209 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2210 = stablehlo.pad %v2208, %v2209, low = [0, 0, 320], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x384xf32>
    %v2211 = stablehlo.reshape %v2210 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v2212 = stablehlo.add %v2174, %v2211 : tensor<256x75648xf32>
    %v2213 = stablehlo.reshape %v2212 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2214 = stablehlo.dot_general %v2213, %b6_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x384xf32>, tensor<384x384xf32>) -> tensor<256x197x384xf32>
    %v2215 = stablehlo.broadcast_in_dim %b6_bo, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v2216 = stablehlo.add %v2214, %v2215 : tensor<256x197x384xf32>
    %v2217 = stablehlo.reshape %v2216 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v2218 = stablehlo.add %v1942, %v2217 : tensor<256x75648xf32>
    %v2219 = stablehlo.reshape %v2218 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2220 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2221 = stablehlo.constant dense<384.0> : tensor<256x197x384xf32>
    %v2222 = stablehlo.constant dense<1.0e-6> : tensor<256x197x384xf32>
    %v2223 = stablehlo.reduce(%v2219 init: %v2220) applies stablehlo.add across dimensions = [2] : (tensor<256x197x384xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v2224 = stablehlo.broadcast_in_dim %v2223, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x384xf32>
    %v2225 = stablehlo.divide %v2224, %v2221 : tensor<256x197x384xf32>
    %v2226 = stablehlo.subtract %v2219, %v2225 : tensor<256x197x384xf32>
    %v2227 = stablehlo.multiply %v2226, %v2226 : tensor<256x197x384xf32>
    %v2228 = stablehlo.reduce(%v2227 init: %v2220) applies stablehlo.add across dimensions = [2] : (tensor<256x197x384xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v2229 = stablehlo.broadcast_in_dim %v2228, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x384xf32>
    %v2230 = stablehlo.divide %v2229, %v2221 : tensor<256x197x384xf32>
    %v2231 = stablehlo.add %v2230, %v2222 : tensor<256x197x384xf32>
    %v2232 = stablehlo.rsqrt %v2231 : tensor<256x197x384xf32>
    %v2233 = stablehlo.multiply %v2226, %v2232 : tensor<256x197x384xf32>
    %v2234 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<256x197x384xf32>
    %v2235 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<256x197x384xf32>
    %v2236 = stablehlo.multiply %v2233, %v2234 : tensor<256x197x384xf32>
    %v2237 = stablehlo.add %v2236, %v2235 : tensor<256x197x384xf32>
    %v2238 = stablehlo.reshape %v2237 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v2239 = stablehlo.reshape %v2238 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2240 = stablehlo.broadcast_in_dim %b6_g2, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v2241 = stablehlo.multiply %v2239, %v2240 : tensor<256x197x384xf32>
    %v2242 = stablehlo.reshape %v2241 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v2243 = stablehlo.reshape %v2242 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2244 = stablehlo.broadcast_in_dim %b6_bt2, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v2245 = stablehlo.add %v2243, %v2244 : tensor<256x197x384xf32>
    %v2246 = stablehlo.reshape %v2245 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v2247 = stablehlo.reshape %v2246 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2248 = stablehlo.dot_general %v2247, %b6_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x384xf32>, tensor<384x1536xf32>) -> tensor<256x197x1536xf32>
    %v2249 = stablehlo.broadcast_in_dim %b6_bfc1, dims = [2] : (tensor<1536xf32>) -> tensor<256x197x1536xf32>
    %v2250 = stablehlo.add %v2248, %v2249 : tensor<256x197x1536xf32>
    %v2251 = stablehlo.reshape %v2250 : (tensor<256x197x1536xf32>) -> tensor<256x302592xf32>
    %v2252 = stablehlo.constant dense<0.5> : tensor<256x302592xf32>
    %v2253 = stablehlo.multiply %v2252, %v2251 : tensor<256x302592xf32>
    %v2254 = stablehlo.negate %v2251 : tensor<256x302592xf32>
    %v2255 = stablehlo.constant dense<0.7071067811865476> : tensor<256x302592xf32>
    %v2256 = stablehlo.multiply %v2254, %v2255 : tensor<256x302592xf32>
    %v2257 = chlo.erfc %v2256 : tensor<256x302592xf32> -> tensor<256x302592xf32>
    %v2258 = stablehlo.multiply %v2253, %v2257 : tensor<256x302592xf32>
    %v2259 = stablehlo.reshape %v2258 : (tensor<256x302592xf32>) -> tensor<256x197x1536xf32>
    %v2260 = stablehlo.dot_general %v2259, %b6_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x1536xf32>, tensor<1536x384xf32>) -> tensor<256x197x384xf32>
    %v2261 = stablehlo.broadcast_in_dim %b6_bfc2, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v2262 = stablehlo.add %v2260, %v2261 : tensor<256x197x384xf32>
    %v2263 = stablehlo.reshape %v2262 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v2264 = stablehlo.add %v2218, %v2263 : tensor<256x75648xf32>
    %v2265 = stablehlo.reshape %v2264 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2266 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2267 = stablehlo.constant dense<384.0> : tensor<256x197x384xf32>
    %v2268 = stablehlo.constant dense<1.0e-6> : tensor<256x197x384xf32>
    %v2269 = stablehlo.reduce(%v2265 init: %v2266) applies stablehlo.add across dimensions = [2] : (tensor<256x197x384xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v2270 = stablehlo.broadcast_in_dim %v2269, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x384xf32>
    %v2271 = stablehlo.divide %v2270, %v2267 : tensor<256x197x384xf32>
    %v2272 = stablehlo.subtract %v2265, %v2271 : tensor<256x197x384xf32>
    %v2273 = stablehlo.multiply %v2272, %v2272 : tensor<256x197x384xf32>
    %v2274 = stablehlo.reduce(%v2273 init: %v2266) applies stablehlo.add across dimensions = [2] : (tensor<256x197x384xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v2275 = stablehlo.broadcast_in_dim %v2274, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x384xf32>
    %v2276 = stablehlo.divide %v2275, %v2267 : tensor<256x197x384xf32>
    %v2277 = stablehlo.add %v2276, %v2268 : tensor<256x197x384xf32>
    %v2278 = stablehlo.rsqrt %v2277 : tensor<256x197x384xf32>
    %v2279 = stablehlo.multiply %v2272, %v2278 : tensor<256x197x384xf32>
    %v2280 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<256x197x384xf32>
    %v2281 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<256x197x384xf32>
    %v2282 = stablehlo.multiply %v2279, %v2280 : tensor<256x197x384xf32>
    %v2283 = stablehlo.add %v2282, %v2281 : tensor<256x197x384xf32>
    %v2284 = stablehlo.reshape %v2283 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v2285 = stablehlo.reshape %v2284 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2286 = stablehlo.broadcast_in_dim %b7_g1, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v2287 = stablehlo.multiply %v2285, %v2286 : tensor<256x197x384xf32>
    %v2288 = stablehlo.reshape %v2287 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v2289 = stablehlo.reshape %v2288 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2290 = stablehlo.broadcast_in_dim %b7_bt1, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v2291 = stablehlo.add %v2289, %v2290 : tensor<256x197x384xf32>
    %v2292 = stablehlo.reshape %v2291 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v2293 = stablehlo.reshape %v2292 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2294 = stablehlo.dot_general %v2293, %b7_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x384xf32>, tensor<384x384xf32>) -> tensor<256x197x384xf32>
    %v2295 = stablehlo.broadcast_in_dim %b7_bq, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v2296 = stablehlo.add %v2294, %v2295 : tensor<256x197x384xf32>
    %v2297 = stablehlo.reshape %v2296 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v2298 = stablehlo.reshape %v2292 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2299 = stablehlo.dot_general %v2298, %b7_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x384xf32>, tensor<384x384xf32>) -> tensor<256x197x384xf32>
    %v2300 = stablehlo.broadcast_in_dim %b7_bk, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v2301 = stablehlo.add %v2299, %v2300 : tensor<256x197x384xf32>
    %v2302 = stablehlo.reshape %v2301 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v2303 = stablehlo.reshape %v2292 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2304 = stablehlo.dot_general %v2303, %b7_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x384xf32>, tensor<384x384xf32>) -> tensor<256x197x384xf32>
    %v2305 = stablehlo.broadcast_in_dim %b7_bv, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v2306 = stablehlo.add %v2304, %v2305 : tensor<256x197x384xf32>
    %v2307 = stablehlo.reshape %v2306 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v2308 = stablehlo.reshape %v2297 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2309 = stablehlo.slice %v2308 [0:256, 0:197, 0:64] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v2310 = stablehlo.reshape %v2309 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2311 = stablehlo.reshape %v2302 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2312 = stablehlo.slice %v2311 [0:256, 0:197, 0:64] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v2313 = stablehlo.reshape %v2312 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2314 = stablehlo.reshape %v2307 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2315 = stablehlo.slice %v2314 [0:256, 0:197, 0:64] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v2316 = stablehlo.reshape %v2315 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2317 = stablehlo.reshape %v2313 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2318 = stablehlo.transpose %v2317, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v2319 = stablehlo.reshape %v2318 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v2320 = stablehlo.reshape %v2310 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2321 = stablehlo.reshape %v2319 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v2322 = stablehlo.dot_general %v2320, %v2321, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v2323 = stablehlo.reshape %v2322 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v2324 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v2325 = stablehlo.multiply %v2323, %v2324 : tensor<256x38809xf32>
    %v2326 = stablehlo.reshape %v2325 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v2327 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v2328 = stablehlo.reduce(%v2326 init: %v2327) applies stablehlo.maximum across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v2329 = stablehlo.broadcast_in_dim %v2328, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v2330 = stablehlo.subtract %v2326, %v2329 : tensor<256x197x197xf32>
    %v2331 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2332 = stablehlo.exponential %v2330 : tensor<256x197x197xf32>
    %v2333 = stablehlo.reduce(%v2332 init: %v2331) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v2334 = stablehlo.broadcast_in_dim %v2333, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v2335 = stablehlo.divide %v2332, %v2334 : tensor<256x197x197xf32>
    %v2336 = stablehlo.reshape %v2335 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v2337 = stablehlo.reshape %v2336 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v2338 = stablehlo.reshape %v2316 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2339 = stablehlo.dot_general %v2337, %v2338, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v2340 = stablehlo.reshape %v2339 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2341 = stablehlo.reshape %v2340 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2342 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2343 = stablehlo.pad %v2341, %v2342, low = [0, 0, 0], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x384xf32>
    %v2344 = stablehlo.reshape %v2343 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v2345 = stablehlo.reshape %v2297 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2346 = stablehlo.slice %v2345 [0:256, 0:197, 64:128] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v2347 = stablehlo.reshape %v2346 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2348 = stablehlo.reshape %v2302 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2349 = stablehlo.slice %v2348 [0:256, 0:197, 64:128] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v2350 = stablehlo.reshape %v2349 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2351 = stablehlo.reshape %v2307 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2352 = stablehlo.slice %v2351 [0:256, 0:197, 64:128] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v2353 = stablehlo.reshape %v2352 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2354 = stablehlo.reshape %v2350 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2355 = stablehlo.transpose %v2354, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v2356 = stablehlo.reshape %v2355 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v2357 = stablehlo.reshape %v2347 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2358 = stablehlo.reshape %v2356 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v2359 = stablehlo.dot_general %v2357, %v2358, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v2360 = stablehlo.reshape %v2359 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v2361 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v2362 = stablehlo.multiply %v2360, %v2361 : tensor<256x38809xf32>
    %v2363 = stablehlo.reshape %v2362 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v2364 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v2365 = stablehlo.reduce(%v2363 init: %v2364) applies stablehlo.maximum across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v2366 = stablehlo.broadcast_in_dim %v2365, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v2367 = stablehlo.subtract %v2363, %v2366 : tensor<256x197x197xf32>
    %v2368 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2369 = stablehlo.exponential %v2367 : tensor<256x197x197xf32>
    %v2370 = stablehlo.reduce(%v2369 init: %v2368) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v2371 = stablehlo.broadcast_in_dim %v2370, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v2372 = stablehlo.divide %v2369, %v2371 : tensor<256x197x197xf32>
    %v2373 = stablehlo.reshape %v2372 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v2374 = stablehlo.reshape %v2373 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v2375 = stablehlo.reshape %v2353 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2376 = stablehlo.dot_general %v2374, %v2375, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v2377 = stablehlo.reshape %v2376 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2378 = stablehlo.reshape %v2377 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2379 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2380 = stablehlo.pad %v2378, %v2379, low = [0, 0, 64], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x384xf32>
    %v2381 = stablehlo.reshape %v2380 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v2382 = stablehlo.add %v2344, %v2381 : tensor<256x75648xf32>
    %v2383 = stablehlo.reshape %v2297 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2384 = stablehlo.slice %v2383 [0:256, 0:197, 128:192] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v2385 = stablehlo.reshape %v2384 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2386 = stablehlo.reshape %v2302 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2387 = stablehlo.slice %v2386 [0:256, 0:197, 128:192] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v2388 = stablehlo.reshape %v2387 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2389 = stablehlo.reshape %v2307 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2390 = stablehlo.slice %v2389 [0:256, 0:197, 128:192] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v2391 = stablehlo.reshape %v2390 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2392 = stablehlo.reshape %v2388 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2393 = stablehlo.transpose %v2392, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v2394 = stablehlo.reshape %v2393 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v2395 = stablehlo.reshape %v2385 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2396 = stablehlo.reshape %v2394 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v2397 = stablehlo.dot_general %v2395, %v2396, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v2398 = stablehlo.reshape %v2397 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v2399 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v2400 = stablehlo.multiply %v2398, %v2399 : tensor<256x38809xf32>
    %v2401 = stablehlo.reshape %v2400 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v2402 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v2403 = stablehlo.reduce(%v2401 init: %v2402) applies stablehlo.maximum across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v2404 = stablehlo.broadcast_in_dim %v2403, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v2405 = stablehlo.subtract %v2401, %v2404 : tensor<256x197x197xf32>
    %v2406 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2407 = stablehlo.exponential %v2405 : tensor<256x197x197xf32>
    %v2408 = stablehlo.reduce(%v2407 init: %v2406) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v2409 = stablehlo.broadcast_in_dim %v2408, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v2410 = stablehlo.divide %v2407, %v2409 : tensor<256x197x197xf32>
    %v2411 = stablehlo.reshape %v2410 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v2412 = stablehlo.reshape %v2411 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v2413 = stablehlo.reshape %v2391 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2414 = stablehlo.dot_general %v2412, %v2413, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v2415 = stablehlo.reshape %v2414 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2416 = stablehlo.reshape %v2415 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2417 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2418 = stablehlo.pad %v2416, %v2417, low = [0, 0, 128], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x384xf32>
    %v2419 = stablehlo.reshape %v2418 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v2420 = stablehlo.add %v2382, %v2419 : tensor<256x75648xf32>
    %v2421 = stablehlo.reshape %v2297 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2422 = stablehlo.slice %v2421 [0:256, 0:197, 192:256] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v2423 = stablehlo.reshape %v2422 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2424 = stablehlo.reshape %v2302 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2425 = stablehlo.slice %v2424 [0:256, 0:197, 192:256] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v2426 = stablehlo.reshape %v2425 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2427 = stablehlo.reshape %v2307 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2428 = stablehlo.slice %v2427 [0:256, 0:197, 192:256] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v2429 = stablehlo.reshape %v2428 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2430 = stablehlo.reshape %v2426 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2431 = stablehlo.transpose %v2430, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v2432 = stablehlo.reshape %v2431 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v2433 = stablehlo.reshape %v2423 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2434 = stablehlo.reshape %v2432 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v2435 = stablehlo.dot_general %v2433, %v2434, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v2436 = stablehlo.reshape %v2435 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v2437 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v2438 = stablehlo.multiply %v2436, %v2437 : tensor<256x38809xf32>
    %v2439 = stablehlo.reshape %v2438 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v2440 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v2441 = stablehlo.reduce(%v2439 init: %v2440) applies stablehlo.maximum across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v2442 = stablehlo.broadcast_in_dim %v2441, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v2443 = stablehlo.subtract %v2439, %v2442 : tensor<256x197x197xf32>
    %v2444 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2445 = stablehlo.exponential %v2443 : tensor<256x197x197xf32>
    %v2446 = stablehlo.reduce(%v2445 init: %v2444) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v2447 = stablehlo.broadcast_in_dim %v2446, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v2448 = stablehlo.divide %v2445, %v2447 : tensor<256x197x197xf32>
    %v2449 = stablehlo.reshape %v2448 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v2450 = stablehlo.reshape %v2449 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v2451 = stablehlo.reshape %v2429 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2452 = stablehlo.dot_general %v2450, %v2451, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v2453 = stablehlo.reshape %v2452 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2454 = stablehlo.reshape %v2453 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2455 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2456 = stablehlo.pad %v2454, %v2455, low = [0, 0, 192], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x384xf32>
    %v2457 = stablehlo.reshape %v2456 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v2458 = stablehlo.add %v2420, %v2457 : tensor<256x75648xf32>
    %v2459 = stablehlo.reshape %v2297 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2460 = stablehlo.slice %v2459 [0:256, 0:197, 256:320] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v2461 = stablehlo.reshape %v2460 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2462 = stablehlo.reshape %v2302 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2463 = stablehlo.slice %v2462 [0:256, 0:197, 256:320] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v2464 = stablehlo.reshape %v2463 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2465 = stablehlo.reshape %v2307 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2466 = stablehlo.slice %v2465 [0:256, 0:197, 256:320] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v2467 = stablehlo.reshape %v2466 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2468 = stablehlo.reshape %v2464 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2469 = stablehlo.transpose %v2468, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v2470 = stablehlo.reshape %v2469 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v2471 = stablehlo.reshape %v2461 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2472 = stablehlo.reshape %v2470 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v2473 = stablehlo.dot_general %v2471, %v2472, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v2474 = stablehlo.reshape %v2473 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v2475 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v2476 = stablehlo.multiply %v2474, %v2475 : tensor<256x38809xf32>
    %v2477 = stablehlo.reshape %v2476 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v2478 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v2479 = stablehlo.reduce(%v2477 init: %v2478) applies stablehlo.maximum across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v2480 = stablehlo.broadcast_in_dim %v2479, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v2481 = stablehlo.subtract %v2477, %v2480 : tensor<256x197x197xf32>
    %v2482 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2483 = stablehlo.exponential %v2481 : tensor<256x197x197xf32>
    %v2484 = stablehlo.reduce(%v2483 init: %v2482) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v2485 = stablehlo.broadcast_in_dim %v2484, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v2486 = stablehlo.divide %v2483, %v2485 : tensor<256x197x197xf32>
    %v2487 = stablehlo.reshape %v2486 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v2488 = stablehlo.reshape %v2487 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v2489 = stablehlo.reshape %v2467 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2490 = stablehlo.dot_general %v2488, %v2489, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v2491 = stablehlo.reshape %v2490 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2492 = stablehlo.reshape %v2491 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2493 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2494 = stablehlo.pad %v2492, %v2493, low = [0, 0, 256], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x384xf32>
    %v2495 = stablehlo.reshape %v2494 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v2496 = stablehlo.add %v2458, %v2495 : tensor<256x75648xf32>
    %v2497 = stablehlo.reshape %v2297 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2498 = stablehlo.slice %v2497 [0:256, 0:197, 320:384] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v2499 = stablehlo.reshape %v2498 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2500 = stablehlo.reshape %v2302 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2501 = stablehlo.slice %v2500 [0:256, 0:197, 320:384] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v2502 = stablehlo.reshape %v2501 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2503 = stablehlo.reshape %v2307 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2504 = stablehlo.slice %v2503 [0:256, 0:197, 320:384] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v2505 = stablehlo.reshape %v2504 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2506 = stablehlo.reshape %v2502 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2507 = stablehlo.transpose %v2506, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v2508 = stablehlo.reshape %v2507 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v2509 = stablehlo.reshape %v2499 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2510 = stablehlo.reshape %v2508 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v2511 = stablehlo.dot_general %v2509, %v2510, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v2512 = stablehlo.reshape %v2511 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v2513 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v2514 = stablehlo.multiply %v2512, %v2513 : tensor<256x38809xf32>
    %v2515 = stablehlo.reshape %v2514 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v2516 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v2517 = stablehlo.reduce(%v2515 init: %v2516) applies stablehlo.maximum across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v2518 = stablehlo.broadcast_in_dim %v2517, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v2519 = stablehlo.subtract %v2515, %v2518 : tensor<256x197x197xf32>
    %v2520 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2521 = stablehlo.exponential %v2519 : tensor<256x197x197xf32>
    %v2522 = stablehlo.reduce(%v2521 init: %v2520) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v2523 = stablehlo.broadcast_in_dim %v2522, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v2524 = stablehlo.divide %v2521, %v2523 : tensor<256x197x197xf32>
    %v2525 = stablehlo.reshape %v2524 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v2526 = stablehlo.reshape %v2525 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v2527 = stablehlo.reshape %v2505 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2528 = stablehlo.dot_general %v2526, %v2527, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v2529 = stablehlo.reshape %v2528 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2530 = stablehlo.reshape %v2529 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2531 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2532 = stablehlo.pad %v2530, %v2531, low = [0, 0, 320], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x384xf32>
    %v2533 = stablehlo.reshape %v2532 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v2534 = stablehlo.add %v2496, %v2533 : tensor<256x75648xf32>
    %v2535 = stablehlo.reshape %v2534 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2536 = stablehlo.dot_general %v2535, %b7_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x384xf32>, tensor<384x384xf32>) -> tensor<256x197x384xf32>
    %v2537 = stablehlo.broadcast_in_dim %b7_bo, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v2538 = stablehlo.add %v2536, %v2537 : tensor<256x197x384xf32>
    %v2539 = stablehlo.reshape %v2538 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v2540 = stablehlo.add %v2264, %v2539 : tensor<256x75648xf32>
    %v2541 = stablehlo.reshape %v2540 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2542 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2543 = stablehlo.constant dense<384.0> : tensor<256x197x384xf32>
    %v2544 = stablehlo.constant dense<1.0e-6> : tensor<256x197x384xf32>
    %v2545 = stablehlo.reduce(%v2541 init: %v2542) applies stablehlo.add across dimensions = [2] : (tensor<256x197x384xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v2546 = stablehlo.broadcast_in_dim %v2545, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x384xf32>
    %v2547 = stablehlo.divide %v2546, %v2543 : tensor<256x197x384xf32>
    %v2548 = stablehlo.subtract %v2541, %v2547 : tensor<256x197x384xf32>
    %v2549 = stablehlo.multiply %v2548, %v2548 : tensor<256x197x384xf32>
    %v2550 = stablehlo.reduce(%v2549 init: %v2542) applies stablehlo.add across dimensions = [2] : (tensor<256x197x384xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v2551 = stablehlo.broadcast_in_dim %v2550, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x384xf32>
    %v2552 = stablehlo.divide %v2551, %v2543 : tensor<256x197x384xf32>
    %v2553 = stablehlo.add %v2552, %v2544 : tensor<256x197x384xf32>
    %v2554 = stablehlo.rsqrt %v2553 : tensor<256x197x384xf32>
    %v2555 = stablehlo.multiply %v2548, %v2554 : tensor<256x197x384xf32>
    %v2556 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<256x197x384xf32>
    %v2557 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<256x197x384xf32>
    %v2558 = stablehlo.multiply %v2555, %v2556 : tensor<256x197x384xf32>
    %v2559 = stablehlo.add %v2558, %v2557 : tensor<256x197x384xf32>
    %v2560 = stablehlo.reshape %v2559 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v2561 = stablehlo.reshape %v2560 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2562 = stablehlo.broadcast_in_dim %b7_g2, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v2563 = stablehlo.multiply %v2561, %v2562 : tensor<256x197x384xf32>
    %v2564 = stablehlo.reshape %v2563 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v2565 = stablehlo.reshape %v2564 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2566 = stablehlo.broadcast_in_dim %b7_bt2, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v2567 = stablehlo.add %v2565, %v2566 : tensor<256x197x384xf32>
    %v2568 = stablehlo.reshape %v2567 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v2569 = stablehlo.reshape %v2568 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2570 = stablehlo.dot_general %v2569, %b7_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x384xf32>, tensor<384x1536xf32>) -> tensor<256x197x1536xf32>
    %v2571 = stablehlo.broadcast_in_dim %b7_bfc1, dims = [2] : (tensor<1536xf32>) -> tensor<256x197x1536xf32>
    %v2572 = stablehlo.add %v2570, %v2571 : tensor<256x197x1536xf32>
    %v2573 = stablehlo.reshape %v2572 : (tensor<256x197x1536xf32>) -> tensor<256x302592xf32>
    %v2574 = stablehlo.constant dense<0.5> : tensor<256x302592xf32>
    %v2575 = stablehlo.multiply %v2574, %v2573 : tensor<256x302592xf32>
    %v2576 = stablehlo.negate %v2573 : tensor<256x302592xf32>
    %v2577 = stablehlo.constant dense<0.7071067811865476> : tensor<256x302592xf32>
    %v2578 = stablehlo.multiply %v2576, %v2577 : tensor<256x302592xf32>
    %v2579 = chlo.erfc %v2578 : tensor<256x302592xf32> -> tensor<256x302592xf32>
    %v2580 = stablehlo.multiply %v2575, %v2579 : tensor<256x302592xf32>
    %v2581 = stablehlo.reshape %v2580 : (tensor<256x302592xf32>) -> tensor<256x197x1536xf32>
    %v2582 = stablehlo.dot_general %v2581, %b7_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x1536xf32>, tensor<1536x384xf32>) -> tensor<256x197x384xf32>
    %v2583 = stablehlo.broadcast_in_dim %b7_bfc2, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v2584 = stablehlo.add %v2582, %v2583 : tensor<256x197x384xf32>
    %v2585 = stablehlo.reshape %v2584 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v2586 = stablehlo.add %v2540, %v2585 : tensor<256x75648xf32>
    %v2587 = stablehlo.reshape %v2586 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2588 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2589 = stablehlo.constant dense<384.0> : tensor<256x197x384xf32>
    %v2590 = stablehlo.constant dense<1.0e-6> : tensor<256x197x384xf32>
    %v2591 = stablehlo.reduce(%v2587 init: %v2588) applies stablehlo.add across dimensions = [2] : (tensor<256x197x384xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v2592 = stablehlo.broadcast_in_dim %v2591, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x384xf32>
    %v2593 = stablehlo.divide %v2592, %v2589 : tensor<256x197x384xf32>
    %v2594 = stablehlo.subtract %v2587, %v2593 : tensor<256x197x384xf32>
    %v2595 = stablehlo.multiply %v2594, %v2594 : tensor<256x197x384xf32>
    %v2596 = stablehlo.reduce(%v2595 init: %v2588) applies stablehlo.add across dimensions = [2] : (tensor<256x197x384xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v2597 = stablehlo.broadcast_in_dim %v2596, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x384xf32>
    %v2598 = stablehlo.divide %v2597, %v2589 : tensor<256x197x384xf32>
    %v2599 = stablehlo.add %v2598, %v2590 : tensor<256x197x384xf32>
    %v2600 = stablehlo.rsqrt %v2599 : tensor<256x197x384xf32>
    %v2601 = stablehlo.multiply %v2594, %v2600 : tensor<256x197x384xf32>
    %v2602 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<256x197x384xf32>
    %v2603 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<256x197x384xf32>
    %v2604 = stablehlo.multiply %v2601, %v2602 : tensor<256x197x384xf32>
    %v2605 = stablehlo.add %v2604, %v2603 : tensor<256x197x384xf32>
    %v2606 = stablehlo.reshape %v2605 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v2607 = stablehlo.reshape %v2606 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2608 = stablehlo.broadcast_in_dim %b8_g1, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v2609 = stablehlo.multiply %v2607, %v2608 : tensor<256x197x384xf32>
    %v2610 = stablehlo.reshape %v2609 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v2611 = stablehlo.reshape %v2610 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2612 = stablehlo.broadcast_in_dim %b8_bt1, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v2613 = stablehlo.add %v2611, %v2612 : tensor<256x197x384xf32>
    %v2614 = stablehlo.reshape %v2613 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v2615 = stablehlo.reshape %v2614 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2616 = stablehlo.dot_general %v2615, %b8_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x384xf32>, tensor<384x384xf32>) -> tensor<256x197x384xf32>
    %v2617 = stablehlo.broadcast_in_dim %b8_bq, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v2618 = stablehlo.add %v2616, %v2617 : tensor<256x197x384xf32>
    %v2619 = stablehlo.reshape %v2618 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v2620 = stablehlo.reshape %v2614 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2621 = stablehlo.dot_general %v2620, %b8_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x384xf32>, tensor<384x384xf32>) -> tensor<256x197x384xf32>
    %v2622 = stablehlo.broadcast_in_dim %b8_bk, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v2623 = stablehlo.add %v2621, %v2622 : tensor<256x197x384xf32>
    %v2624 = stablehlo.reshape %v2623 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v2625 = stablehlo.reshape %v2614 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2626 = stablehlo.dot_general %v2625, %b8_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x384xf32>, tensor<384x384xf32>) -> tensor<256x197x384xf32>
    %v2627 = stablehlo.broadcast_in_dim %b8_bv, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v2628 = stablehlo.add %v2626, %v2627 : tensor<256x197x384xf32>
    %v2629 = stablehlo.reshape %v2628 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v2630 = stablehlo.reshape %v2619 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2631 = stablehlo.slice %v2630 [0:256, 0:197, 0:64] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v2632 = stablehlo.reshape %v2631 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2633 = stablehlo.reshape %v2624 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2634 = stablehlo.slice %v2633 [0:256, 0:197, 0:64] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v2635 = stablehlo.reshape %v2634 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2636 = stablehlo.reshape %v2629 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2637 = stablehlo.slice %v2636 [0:256, 0:197, 0:64] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v2638 = stablehlo.reshape %v2637 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2639 = stablehlo.reshape %v2635 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2640 = stablehlo.transpose %v2639, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v2641 = stablehlo.reshape %v2640 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v2642 = stablehlo.reshape %v2632 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2643 = stablehlo.reshape %v2641 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v2644 = stablehlo.dot_general %v2642, %v2643, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v2645 = stablehlo.reshape %v2644 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v2646 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v2647 = stablehlo.multiply %v2645, %v2646 : tensor<256x38809xf32>
    %v2648 = stablehlo.reshape %v2647 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v2649 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v2650 = stablehlo.reduce(%v2648 init: %v2649) applies stablehlo.maximum across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v2651 = stablehlo.broadcast_in_dim %v2650, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v2652 = stablehlo.subtract %v2648, %v2651 : tensor<256x197x197xf32>
    %v2653 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2654 = stablehlo.exponential %v2652 : tensor<256x197x197xf32>
    %v2655 = stablehlo.reduce(%v2654 init: %v2653) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v2656 = stablehlo.broadcast_in_dim %v2655, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v2657 = stablehlo.divide %v2654, %v2656 : tensor<256x197x197xf32>
    %v2658 = stablehlo.reshape %v2657 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v2659 = stablehlo.reshape %v2658 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v2660 = stablehlo.reshape %v2638 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2661 = stablehlo.dot_general %v2659, %v2660, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v2662 = stablehlo.reshape %v2661 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2663 = stablehlo.reshape %v2662 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2664 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2665 = stablehlo.pad %v2663, %v2664, low = [0, 0, 0], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x384xf32>
    %v2666 = stablehlo.reshape %v2665 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v2667 = stablehlo.reshape %v2619 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2668 = stablehlo.slice %v2667 [0:256, 0:197, 64:128] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v2669 = stablehlo.reshape %v2668 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2670 = stablehlo.reshape %v2624 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2671 = stablehlo.slice %v2670 [0:256, 0:197, 64:128] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v2672 = stablehlo.reshape %v2671 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2673 = stablehlo.reshape %v2629 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2674 = stablehlo.slice %v2673 [0:256, 0:197, 64:128] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v2675 = stablehlo.reshape %v2674 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2676 = stablehlo.reshape %v2672 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2677 = stablehlo.transpose %v2676, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v2678 = stablehlo.reshape %v2677 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v2679 = stablehlo.reshape %v2669 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2680 = stablehlo.reshape %v2678 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v2681 = stablehlo.dot_general %v2679, %v2680, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v2682 = stablehlo.reshape %v2681 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v2683 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v2684 = stablehlo.multiply %v2682, %v2683 : tensor<256x38809xf32>
    %v2685 = stablehlo.reshape %v2684 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v2686 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v2687 = stablehlo.reduce(%v2685 init: %v2686) applies stablehlo.maximum across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v2688 = stablehlo.broadcast_in_dim %v2687, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v2689 = stablehlo.subtract %v2685, %v2688 : tensor<256x197x197xf32>
    %v2690 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2691 = stablehlo.exponential %v2689 : tensor<256x197x197xf32>
    %v2692 = stablehlo.reduce(%v2691 init: %v2690) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v2693 = stablehlo.broadcast_in_dim %v2692, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v2694 = stablehlo.divide %v2691, %v2693 : tensor<256x197x197xf32>
    %v2695 = stablehlo.reshape %v2694 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v2696 = stablehlo.reshape %v2695 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v2697 = stablehlo.reshape %v2675 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2698 = stablehlo.dot_general %v2696, %v2697, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v2699 = stablehlo.reshape %v2698 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2700 = stablehlo.reshape %v2699 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2701 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2702 = stablehlo.pad %v2700, %v2701, low = [0, 0, 64], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x384xf32>
    %v2703 = stablehlo.reshape %v2702 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v2704 = stablehlo.add %v2666, %v2703 : tensor<256x75648xf32>
    %v2705 = stablehlo.reshape %v2619 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2706 = stablehlo.slice %v2705 [0:256, 0:197, 128:192] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v2707 = stablehlo.reshape %v2706 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2708 = stablehlo.reshape %v2624 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2709 = stablehlo.slice %v2708 [0:256, 0:197, 128:192] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v2710 = stablehlo.reshape %v2709 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2711 = stablehlo.reshape %v2629 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2712 = stablehlo.slice %v2711 [0:256, 0:197, 128:192] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v2713 = stablehlo.reshape %v2712 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2714 = stablehlo.reshape %v2710 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2715 = stablehlo.transpose %v2714, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v2716 = stablehlo.reshape %v2715 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v2717 = stablehlo.reshape %v2707 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2718 = stablehlo.reshape %v2716 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v2719 = stablehlo.dot_general %v2717, %v2718, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v2720 = stablehlo.reshape %v2719 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v2721 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v2722 = stablehlo.multiply %v2720, %v2721 : tensor<256x38809xf32>
    %v2723 = stablehlo.reshape %v2722 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v2724 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v2725 = stablehlo.reduce(%v2723 init: %v2724) applies stablehlo.maximum across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v2726 = stablehlo.broadcast_in_dim %v2725, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v2727 = stablehlo.subtract %v2723, %v2726 : tensor<256x197x197xf32>
    %v2728 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2729 = stablehlo.exponential %v2727 : tensor<256x197x197xf32>
    %v2730 = stablehlo.reduce(%v2729 init: %v2728) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v2731 = stablehlo.broadcast_in_dim %v2730, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v2732 = stablehlo.divide %v2729, %v2731 : tensor<256x197x197xf32>
    %v2733 = stablehlo.reshape %v2732 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v2734 = stablehlo.reshape %v2733 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v2735 = stablehlo.reshape %v2713 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2736 = stablehlo.dot_general %v2734, %v2735, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v2737 = stablehlo.reshape %v2736 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2738 = stablehlo.reshape %v2737 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2739 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2740 = stablehlo.pad %v2738, %v2739, low = [0, 0, 128], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x384xf32>
    %v2741 = stablehlo.reshape %v2740 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v2742 = stablehlo.add %v2704, %v2741 : tensor<256x75648xf32>
    %v2743 = stablehlo.reshape %v2619 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2744 = stablehlo.slice %v2743 [0:256, 0:197, 192:256] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v2745 = stablehlo.reshape %v2744 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2746 = stablehlo.reshape %v2624 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2747 = stablehlo.slice %v2746 [0:256, 0:197, 192:256] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v2748 = stablehlo.reshape %v2747 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2749 = stablehlo.reshape %v2629 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2750 = stablehlo.slice %v2749 [0:256, 0:197, 192:256] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v2751 = stablehlo.reshape %v2750 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2752 = stablehlo.reshape %v2748 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2753 = stablehlo.transpose %v2752, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v2754 = stablehlo.reshape %v2753 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v2755 = stablehlo.reshape %v2745 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2756 = stablehlo.reshape %v2754 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v2757 = stablehlo.dot_general %v2755, %v2756, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v2758 = stablehlo.reshape %v2757 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v2759 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v2760 = stablehlo.multiply %v2758, %v2759 : tensor<256x38809xf32>
    %v2761 = stablehlo.reshape %v2760 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v2762 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v2763 = stablehlo.reduce(%v2761 init: %v2762) applies stablehlo.maximum across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v2764 = stablehlo.broadcast_in_dim %v2763, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v2765 = stablehlo.subtract %v2761, %v2764 : tensor<256x197x197xf32>
    %v2766 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2767 = stablehlo.exponential %v2765 : tensor<256x197x197xf32>
    %v2768 = stablehlo.reduce(%v2767 init: %v2766) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v2769 = stablehlo.broadcast_in_dim %v2768, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v2770 = stablehlo.divide %v2767, %v2769 : tensor<256x197x197xf32>
    %v2771 = stablehlo.reshape %v2770 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v2772 = stablehlo.reshape %v2771 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v2773 = stablehlo.reshape %v2751 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2774 = stablehlo.dot_general %v2772, %v2773, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v2775 = stablehlo.reshape %v2774 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2776 = stablehlo.reshape %v2775 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2777 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2778 = stablehlo.pad %v2776, %v2777, low = [0, 0, 192], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x384xf32>
    %v2779 = stablehlo.reshape %v2778 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v2780 = stablehlo.add %v2742, %v2779 : tensor<256x75648xf32>
    %v2781 = stablehlo.reshape %v2619 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2782 = stablehlo.slice %v2781 [0:256, 0:197, 256:320] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v2783 = stablehlo.reshape %v2782 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2784 = stablehlo.reshape %v2624 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2785 = stablehlo.slice %v2784 [0:256, 0:197, 256:320] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v2786 = stablehlo.reshape %v2785 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2787 = stablehlo.reshape %v2629 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2788 = stablehlo.slice %v2787 [0:256, 0:197, 256:320] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v2789 = stablehlo.reshape %v2788 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2790 = stablehlo.reshape %v2786 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2791 = stablehlo.transpose %v2790, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v2792 = stablehlo.reshape %v2791 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v2793 = stablehlo.reshape %v2783 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2794 = stablehlo.reshape %v2792 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v2795 = stablehlo.dot_general %v2793, %v2794, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v2796 = stablehlo.reshape %v2795 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v2797 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v2798 = stablehlo.multiply %v2796, %v2797 : tensor<256x38809xf32>
    %v2799 = stablehlo.reshape %v2798 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v2800 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v2801 = stablehlo.reduce(%v2799 init: %v2800) applies stablehlo.maximum across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v2802 = stablehlo.broadcast_in_dim %v2801, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v2803 = stablehlo.subtract %v2799, %v2802 : tensor<256x197x197xf32>
    %v2804 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2805 = stablehlo.exponential %v2803 : tensor<256x197x197xf32>
    %v2806 = stablehlo.reduce(%v2805 init: %v2804) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v2807 = stablehlo.broadcast_in_dim %v2806, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v2808 = stablehlo.divide %v2805, %v2807 : tensor<256x197x197xf32>
    %v2809 = stablehlo.reshape %v2808 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v2810 = stablehlo.reshape %v2809 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v2811 = stablehlo.reshape %v2789 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2812 = stablehlo.dot_general %v2810, %v2811, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v2813 = stablehlo.reshape %v2812 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2814 = stablehlo.reshape %v2813 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2815 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2816 = stablehlo.pad %v2814, %v2815, low = [0, 0, 256], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x384xf32>
    %v2817 = stablehlo.reshape %v2816 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v2818 = stablehlo.add %v2780, %v2817 : tensor<256x75648xf32>
    %v2819 = stablehlo.reshape %v2619 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2820 = stablehlo.slice %v2819 [0:256, 0:197, 320:384] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v2821 = stablehlo.reshape %v2820 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2822 = stablehlo.reshape %v2624 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2823 = stablehlo.slice %v2822 [0:256, 0:197, 320:384] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v2824 = stablehlo.reshape %v2823 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2825 = stablehlo.reshape %v2629 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2826 = stablehlo.slice %v2825 [0:256, 0:197, 320:384] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v2827 = stablehlo.reshape %v2826 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2828 = stablehlo.reshape %v2824 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2829 = stablehlo.transpose %v2828, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v2830 = stablehlo.reshape %v2829 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v2831 = stablehlo.reshape %v2821 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2832 = stablehlo.reshape %v2830 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v2833 = stablehlo.dot_general %v2831, %v2832, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v2834 = stablehlo.reshape %v2833 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v2835 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v2836 = stablehlo.multiply %v2834, %v2835 : tensor<256x38809xf32>
    %v2837 = stablehlo.reshape %v2836 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v2838 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v2839 = stablehlo.reduce(%v2837 init: %v2838) applies stablehlo.maximum across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v2840 = stablehlo.broadcast_in_dim %v2839, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v2841 = stablehlo.subtract %v2837, %v2840 : tensor<256x197x197xf32>
    %v2842 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2843 = stablehlo.exponential %v2841 : tensor<256x197x197xf32>
    %v2844 = stablehlo.reduce(%v2843 init: %v2842) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v2845 = stablehlo.broadcast_in_dim %v2844, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v2846 = stablehlo.divide %v2843, %v2845 : tensor<256x197x197xf32>
    %v2847 = stablehlo.reshape %v2846 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v2848 = stablehlo.reshape %v2847 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v2849 = stablehlo.reshape %v2827 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2850 = stablehlo.dot_general %v2848, %v2849, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v2851 = stablehlo.reshape %v2850 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2852 = stablehlo.reshape %v2851 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2853 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2854 = stablehlo.pad %v2852, %v2853, low = [0, 0, 320], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x384xf32>
    %v2855 = stablehlo.reshape %v2854 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v2856 = stablehlo.add %v2818, %v2855 : tensor<256x75648xf32>
    %v2857 = stablehlo.reshape %v2856 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2858 = stablehlo.dot_general %v2857, %b8_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x384xf32>, tensor<384x384xf32>) -> tensor<256x197x384xf32>
    %v2859 = stablehlo.broadcast_in_dim %b8_bo, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v2860 = stablehlo.add %v2858, %v2859 : tensor<256x197x384xf32>
    %v2861 = stablehlo.reshape %v2860 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v2862 = stablehlo.add %v2586, %v2861 : tensor<256x75648xf32>
    %v2863 = stablehlo.reshape %v2862 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2864 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2865 = stablehlo.constant dense<384.0> : tensor<256x197x384xf32>
    %v2866 = stablehlo.constant dense<1.0e-6> : tensor<256x197x384xf32>
    %v2867 = stablehlo.reduce(%v2863 init: %v2864) applies stablehlo.add across dimensions = [2] : (tensor<256x197x384xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v2868 = stablehlo.broadcast_in_dim %v2867, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x384xf32>
    %v2869 = stablehlo.divide %v2868, %v2865 : tensor<256x197x384xf32>
    %v2870 = stablehlo.subtract %v2863, %v2869 : tensor<256x197x384xf32>
    %v2871 = stablehlo.multiply %v2870, %v2870 : tensor<256x197x384xf32>
    %v2872 = stablehlo.reduce(%v2871 init: %v2864) applies stablehlo.add across dimensions = [2] : (tensor<256x197x384xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v2873 = stablehlo.broadcast_in_dim %v2872, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x384xf32>
    %v2874 = stablehlo.divide %v2873, %v2865 : tensor<256x197x384xf32>
    %v2875 = stablehlo.add %v2874, %v2866 : tensor<256x197x384xf32>
    %v2876 = stablehlo.rsqrt %v2875 : tensor<256x197x384xf32>
    %v2877 = stablehlo.multiply %v2870, %v2876 : tensor<256x197x384xf32>
    %v2878 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<256x197x384xf32>
    %v2879 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<256x197x384xf32>
    %v2880 = stablehlo.multiply %v2877, %v2878 : tensor<256x197x384xf32>
    %v2881 = stablehlo.add %v2880, %v2879 : tensor<256x197x384xf32>
    %v2882 = stablehlo.reshape %v2881 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v2883 = stablehlo.reshape %v2882 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2884 = stablehlo.broadcast_in_dim %b8_g2, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v2885 = stablehlo.multiply %v2883, %v2884 : tensor<256x197x384xf32>
    %v2886 = stablehlo.reshape %v2885 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v2887 = stablehlo.reshape %v2886 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2888 = stablehlo.broadcast_in_dim %b8_bt2, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v2889 = stablehlo.add %v2887, %v2888 : tensor<256x197x384xf32>
    %v2890 = stablehlo.reshape %v2889 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v2891 = stablehlo.reshape %v2890 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2892 = stablehlo.dot_general %v2891, %b8_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x384xf32>, tensor<384x1536xf32>) -> tensor<256x197x1536xf32>
    %v2893 = stablehlo.broadcast_in_dim %b8_bfc1, dims = [2] : (tensor<1536xf32>) -> tensor<256x197x1536xf32>
    %v2894 = stablehlo.add %v2892, %v2893 : tensor<256x197x1536xf32>
    %v2895 = stablehlo.reshape %v2894 : (tensor<256x197x1536xf32>) -> tensor<256x302592xf32>
    %v2896 = stablehlo.constant dense<0.5> : tensor<256x302592xf32>
    %v2897 = stablehlo.multiply %v2896, %v2895 : tensor<256x302592xf32>
    %v2898 = stablehlo.negate %v2895 : tensor<256x302592xf32>
    %v2899 = stablehlo.constant dense<0.7071067811865476> : tensor<256x302592xf32>
    %v2900 = stablehlo.multiply %v2898, %v2899 : tensor<256x302592xf32>
    %v2901 = chlo.erfc %v2900 : tensor<256x302592xf32> -> tensor<256x302592xf32>
    %v2902 = stablehlo.multiply %v2897, %v2901 : tensor<256x302592xf32>
    %v2903 = stablehlo.reshape %v2902 : (tensor<256x302592xf32>) -> tensor<256x197x1536xf32>
    %v2904 = stablehlo.dot_general %v2903, %b8_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x1536xf32>, tensor<1536x384xf32>) -> tensor<256x197x384xf32>
    %v2905 = stablehlo.broadcast_in_dim %b8_bfc2, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v2906 = stablehlo.add %v2904, %v2905 : tensor<256x197x384xf32>
    %v2907 = stablehlo.reshape %v2906 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v2908 = stablehlo.add %v2862, %v2907 : tensor<256x75648xf32>
    %v2909 = stablehlo.reshape %v2908 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2910 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2911 = stablehlo.constant dense<384.0> : tensor<256x197x384xf32>
    %v2912 = stablehlo.constant dense<1.0e-6> : tensor<256x197x384xf32>
    %v2913 = stablehlo.reduce(%v2909 init: %v2910) applies stablehlo.add across dimensions = [2] : (tensor<256x197x384xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v2914 = stablehlo.broadcast_in_dim %v2913, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x384xf32>
    %v2915 = stablehlo.divide %v2914, %v2911 : tensor<256x197x384xf32>
    %v2916 = stablehlo.subtract %v2909, %v2915 : tensor<256x197x384xf32>
    %v2917 = stablehlo.multiply %v2916, %v2916 : tensor<256x197x384xf32>
    %v2918 = stablehlo.reduce(%v2917 init: %v2910) applies stablehlo.add across dimensions = [2] : (tensor<256x197x384xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v2919 = stablehlo.broadcast_in_dim %v2918, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x384xf32>
    %v2920 = stablehlo.divide %v2919, %v2911 : tensor<256x197x384xf32>
    %v2921 = stablehlo.add %v2920, %v2912 : tensor<256x197x384xf32>
    %v2922 = stablehlo.rsqrt %v2921 : tensor<256x197x384xf32>
    %v2923 = stablehlo.multiply %v2916, %v2922 : tensor<256x197x384xf32>
    %v2924 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<256x197x384xf32>
    %v2925 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<256x197x384xf32>
    %v2926 = stablehlo.multiply %v2923, %v2924 : tensor<256x197x384xf32>
    %v2927 = stablehlo.add %v2926, %v2925 : tensor<256x197x384xf32>
    %v2928 = stablehlo.reshape %v2927 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v2929 = stablehlo.reshape %v2928 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2930 = stablehlo.broadcast_in_dim %b9_g1, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v2931 = stablehlo.multiply %v2929, %v2930 : tensor<256x197x384xf32>
    %v2932 = stablehlo.reshape %v2931 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v2933 = stablehlo.reshape %v2932 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2934 = stablehlo.broadcast_in_dim %b9_bt1, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v2935 = stablehlo.add %v2933, %v2934 : tensor<256x197x384xf32>
    %v2936 = stablehlo.reshape %v2935 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v2937 = stablehlo.reshape %v2936 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2938 = stablehlo.dot_general %v2937, %b9_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x384xf32>, tensor<384x384xf32>) -> tensor<256x197x384xf32>
    %v2939 = stablehlo.broadcast_in_dim %b9_bq, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v2940 = stablehlo.add %v2938, %v2939 : tensor<256x197x384xf32>
    %v2941 = stablehlo.reshape %v2940 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v2942 = stablehlo.reshape %v2936 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2943 = stablehlo.dot_general %v2942, %b9_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x384xf32>, tensor<384x384xf32>) -> tensor<256x197x384xf32>
    %v2944 = stablehlo.broadcast_in_dim %b9_bk, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v2945 = stablehlo.add %v2943, %v2944 : tensor<256x197x384xf32>
    %v2946 = stablehlo.reshape %v2945 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v2947 = stablehlo.reshape %v2936 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2948 = stablehlo.dot_general %v2947, %b9_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x384xf32>, tensor<384x384xf32>) -> tensor<256x197x384xf32>
    %v2949 = stablehlo.broadcast_in_dim %b9_bv, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v2950 = stablehlo.add %v2948, %v2949 : tensor<256x197x384xf32>
    %v2951 = stablehlo.reshape %v2950 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v2952 = stablehlo.reshape %v2941 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2953 = stablehlo.slice %v2952 [0:256, 0:197, 0:64] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v2954 = stablehlo.reshape %v2953 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2955 = stablehlo.reshape %v2946 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2956 = stablehlo.slice %v2955 [0:256, 0:197, 0:64] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v2957 = stablehlo.reshape %v2956 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2958 = stablehlo.reshape %v2951 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2959 = stablehlo.slice %v2958 [0:256, 0:197, 0:64] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v2960 = stablehlo.reshape %v2959 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2961 = stablehlo.reshape %v2957 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2962 = stablehlo.transpose %v2961, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v2963 = stablehlo.reshape %v2962 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v2964 = stablehlo.reshape %v2954 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2965 = stablehlo.reshape %v2963 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v2966 = stablehlo.dot_general %v2964, %v2965, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v2967 = stablehlo.reshape %v2966 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v2968 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v2969 = stablehlo.multiply %v2967, %v2968 : tensor<256x38809xf32>
    %v2970 = stablehlo.reshape %v2969 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v2971 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v2972 = stablehlo.reduce(%v2970 init: %v2971) applies stablehlo.maximum across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v2973 = stablehlo.broadcast_in_dim %v2972, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v2974 = stablehlo.subtract %v2970, %v2973 : tensor<256x197x197xf32>
    %v2975 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2976 = stablehlo.exponential %v2974 : tensor<256x197x197xf32>
    %v2977 = stablehlo.reduce(%v2976 init: %v2975) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v2978 = stablehlo.broadcast_in_dim %v2977, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v2979 = stablehlo.divide %v2976, %v2978 : tensor<256x197x197xf32>
    %v2980 = stablehlo.reshape %v2979 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v2981 = stablehlo.reshape %v2980 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v2982 = stablehlo.reshape %v2960 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2983 = stablehlo.dot_general %v2981, %v2982, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v2984 = stablehlo.reshape %v2983 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2985 = stablehlo.reshape %v2984 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2986 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2987 = stablehlo.pad %v2985, %v2986, low = [0, 0, 0], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x384xf32>
    %v2988 = stablehlo.reshape %v2987 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v2989 = stablehlo.reshape %v2941 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2990 = stablehlo.slice %v2989 [0:256, 0:197, 64:128] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v2991 = stablehlo.reshape %v2990 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2992 = stablehlo.reshape %v2946 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2993 = stablehlo.slice %v2992 [0:256, 0:197, 64:128] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v2994 = stablehlo.reshape %v2993 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2995 = stablehlo.reshape %v2951 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v2996 = stablehlo.slice %v2995 [0:256, 0:197, 64:128] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v2997 = stablehlo.reshape %v2996 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v2998 = stablehlo.reshape %v2994 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v2999 = stablehlo.transpose %v2998, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v3000 = stablehlo.reshape %v2999 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v3001 = stablehlo.reshape %v2991 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v3002 = stablehlo.reshape %v3000 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v3003 = stablehlo.dot_general %v3001, %v3002, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v3004 = stablehlo.reshape %v3003 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v3005 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v3006 = stablehlo.multiply %v3004, %v3005 : tensor<256x38809xf32>
    %v3007 = stablehlo.reshape %v3006 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v3008 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v3009 = stablehlo.reduce(%v3007 init: %v3008) applies stablehlo.maximum across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v3010 = stablehlo.broadcast_in_dim %v3009, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v3011 = stablehlo.subtract %v3007, %v3010 : tensor<256x197x197xf32>
    %v3012 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3013 = stablehlo.exponential %v3011 : tensor<256x197x197xf32>
    %v3014 = stablehlo.reduce(%v3013 init: %v3012) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v3015 = stablehlo.broadcast_in_dim %v3014, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v3016 = stablehlo.divide %v3013, %v3015 : tensor<256x197x197xf32>
    %v3017 = stablehlo.reshape %v3016 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v3018 = stablehlo.reshape %v3017 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v3019 = stablehlo.reshape %v2997 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v3020 = stablehlo.dot_general %v3018, %v3019, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v3021 = stablehlo.reshape %v3020 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v3022 = stablehlo.reshape %v3021 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v3023 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3024 = stablehlo.pad %v3022, %v3023, low = [0, 0, 64], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x384xf32>
    %v3025 = stablehlo.reshape %v3024 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v3026 = stablehlo.add %v2988, %v3025 : tensor<256x75648xf32>
    %v3027 = stablehlo.reshape %v2941 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v3028 = stablehlo.slice %v3027 [0:256, 0:197, 128:192] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v3029 = stablehlo.reshape %v3028 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v3030 = stablehlo.reshape %v2946 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v3031 = stablehlo.slice %v3030 [0:256, 0:197, 128:192] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v3032 = stablehlo.reshape %v3031 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v3033 = stablehlo.reshape %v2951 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v3034 = stablehlo.slice %v3033 [0:256, 0:197, 128:192] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v3035 = stablehlo.reshape %v3034 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v3036 = stablehlo.reshape %v3032 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v3037 = stablehlo.transpose %v3036, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v3038 = stablehlo.reshape %v3037 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v3039 = stablehlo.reshape %v3029 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v3040 = stablehlo.reshape %v3038 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v3041 = stablehlo.dot_general %v3039, %v3040, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v3042 = stablehlo.reshape %v3041 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v3043 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v3044 = stablehlo.multiply %v3042, %v3043 : tensor<256x38809xf32>
    %v3045 = stablehlo.reshape %v3044 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v3046 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v3047 = stablehlo.reduce(%v3045 init: %v3046) applies stablehlo.maximum across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v3048 = stablehlo.broadcast_in_dim %v3047, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v3049 = stablehlo.subtract %v3045, %v3048 : tensor<256x197x197xf32>
    %v3050 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3051 = stablehlo.exponential %v3049 : tensor<256x197x197xf32>
    %v3052 = stablehlo.reduce(%v3051 init: %v3050) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v3053 = stablehlo.broadcast_in_dim %v3052, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v3054 = stablehlo.divide %v3051, %v3053 : tensor<256x197x197xf32>
    %v3055 = stablehlo.reshape %v3054 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v3056 = stablehlo.reshape %v3055 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v3057 = stablehlo.reshape %v3035 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v3058 = stablehlo.dot_general %v3056, %v3057, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v3059 = stablehlo.reshape %v3058 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v3060 = stablehlo.reshape %v3059 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v3061 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3062 = stablehlo.pad %v3060, %v3061, low = [0, 0, 128], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x384xf32>
    %v3063 = stablehlo.reshape %v3062 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v3064 = stablehlo.add %v3026, %v3063 : tensor<256x75648xf32>
    %v3065 = stablehlo.reshape %v2941 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v3066 = stablehlo.slice %v3065 [0:256, 0:197, 192:256] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v3067 = stablehlo.reshape %v3066 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v3068 = stablehlo.reshape %v2946 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v3069 = stablehlo.slice %v3068 [0:256, 0:197, 192:256] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v3070 = stablehlo.reshape %v3069 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v3071 = stablehlo.reshape %v2951 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v3072 = stablehlo.slice %v3071 [0:256, 0:197, 192:256] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v3073 = stablehlo.reshape %v3072 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v3074 = stablehlo.reshape %v3070 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v3075 = stablehlo.transpose %v3074, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v3076 = stablehlo.reshape %v3075 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v3077 = stablehlo.reshape %v3067 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v3078 = stablehlo.reshape %v3076 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v3079 = stablehlo.dot_general %v3077, %v3078, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v3080 = stablehlo.reshape %v3079 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v3081 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v3082 = stablehlo.multiply %v3080, %v3081 : tensor<256x38809xf32>
    %v3083 = stablehlo.reshape %v3082 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v3084 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v3085 = stablehlo.reduce(%v3083 init: %v3084) applies stablehlo.maximum across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v3086 = stablehlo.broadcast_in_dim %v3085, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v3087 = stablehlo.subtract %v3083, %v3086 : tensor<256x197x197xf32>
    %v3088 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3089 = stablehlo.exponential %v3087 : tensor<256x197x197xf32>
    %v3090 = stablehlo.reduce(%v3089 init: %v3088) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v3091 = stablehlo.broadcast_in_dim %v3090, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v3092 = stablehlo.divide %v3089, %v3091 : tensor<256x197x197xf32>
    %v3093 = stablehlo.reshape %v3092 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v3094 = stablehlo.reshape %v3093 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v3095 = stablehlo.reshape %v3073 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v3096 = stablehlo.dot_general %v3094, %v3095, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v3097 = stablehlo.reshape %v3096 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v3098 = stablehlo.reshape %v3097 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v3099 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3100 = stablehlo.pad %v3098, %v3099, low = [0, 0, 192], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x384xf32>
    %v3101 = stablehlo.reshape %v3100 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v3102 = stablehlo.add %v3064, %v3101 : tensor<256x75648xf32>
    %v3103 = stablehlo.reshape %v2941 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v3104 = stablehlo.slice %v3103 [0:256, 0:197, 256:320] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v3105 = stablehlo.reshape %v3104 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v3106 = stablehlo.reshape %v2946 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v3107 = stablehlo.slice %v3106 [0:256, 0:197, 256:320] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v3108 = stablehlo.reshape %v3107 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v3109 = stablehlo.reshape %v2951 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v3110 = stablehlo.slice %v3109 [0:256, 0:197, 256:320] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v3111 = stablehlo.reshape %v3110 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v3112 = stablehlo.reshape %v3108 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v3113 = stablehlo.transpose %v3112, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v3114 = stablehlo.reshape %v3113 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v3115 = stablehlo.reshape %v3105 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v3116 = stablehlo.reshape %v3114 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v3117 = stablehlo.dot_general %v3115, %v3116, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v3118 = stablehlo.reshape %v3117 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v3119 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v3120 = stablehlo.multiply %v3118, %v3119 : tensor<256x38809xf32>
    %v3121 = stablehlo.reshape %v3120 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v3122 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v3123 = stablehlo.reduce(%v3121 init: %v3122) applies stablehlo.maximum across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v3124 = stablehlo.broadcast_in_dim %v3123, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v3125 = stablehlo.subtract %v3121, %v3124 : tensor<256x197x197xf32>
    %v3126 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3127 = stablehlo.exponential %v3125 : tensor<256x197x197xf32>
    %v3128 = stablehlo.reduce(%v3127 init: %v3126) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v3129 = stablehlo.broadcast_in_dim %v3128, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v3130 = stablehlo.divide %v3127, %v3129 : tensor<256x197x197xf32>
    %v3131 = stablehlo.reshape %v3130 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v3132 = stablehlo.reshape %v3131 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v3133 = stablehlo.reshape %v3111 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v3134 = stablehlo.dot_general %v3132, %v3133, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v3135 = stablehlo.reshape %v3134 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v3136 = stablehlo.reshape %v3135 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v3137 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3138 = stablehlo.pad %v3136, %v3137, low = [0, 0, 256], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x384xf32>
    %v3139 = stablehlo.reshape %v3138 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v3140 = stablehlo.add %v3102, %v3139 : tensor<256x75648xf32>
    %v3141 = stablehlo.reshape %v2941 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v3142 = stablehlo.slice %v3141 [0:256, 0:197, 320:384] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v3143 = stablehlo.reshape %v3142 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v3144 = stablehlo.reshape %v2946 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v3145 = stablehlo.slice %v3144 [0:256, 0:197, 320:384] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v3146 = stablehlo.reshape %v3145 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v3147 = stablehlo.reshape %v2951 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v3148 = stablehlo.slice %v3147 [0:256, 0:197, 320:384] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v3149 = stablehlo.reshape %v3148 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v3150 = stablehlo.reshape %v3146 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v3151 = stablehlo.transpose %v3150, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v3152 = stablehlo.reshape %v3151 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v3153 = stablehlo.reshape %v3143 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v3154 = stablehlo.reshape %v3152 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v3155 = stablehlo.dot_general %v3153, %v3154, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v3156 = stablehlo.reshape %v3155 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v3157 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v3158 = stablehlo.multiply %v3156, %v3157 : tensor<256x38809xf32>
    %v3159 = stablehlo.reshape %v3158 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v3160 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v3161 = stablehlo.reduce(%v3159 init: %v3160) applies stablehlo.maximum across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v3162 = stablehlo.broadcast_in_dim %v3161, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v3163 = stablehlo.subtract %v3159, %v3162 : tensor<256x197x197xf32>
    %v3164 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3165 = stablehlo.exponential %v3163 : tensor<256x197x197xf32>
    %v3166 = stablehlo.reduce(%v3165 init: %v3164) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v3167 = stablehlo.broadcast_in_dim %v3166, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v3168 = stablehlo.divide %v3165, %v3167 : tensor<256x197x197xf32>
    %v3169 = stablehlo.reshape %v3168 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v3170 = stablehlo.reshape %v3169 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v3171 = stablehlo.reshape %v3149 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v3172 = stablehlo.dot_general %v3170, %v3171, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v3173 = stablehlo.reshape %v3172 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v3174 = stablehlo.reshape %v3173 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v3175 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3176 = stablehlo.pad %v3174, %v3175, low = [0, 0, 320], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x384xf32>
    %v3177 = stablehlo.reshape %v3176 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v3178 = stablehlo.add %v3140, %v3177 : tensor<256x75648xf32>
    %v3179 = stablehlo.reshape %v3178 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v3180 = stablehlo.dot_general %v3179, %b9_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x384xf32>, tensor<384x384xf32>) -> tensor<256x197x384xf32>
    %v3181 = stablehlo.broadcast_in_dim %b9_bo, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v3182 = stablehlo.add %v3180, %v3181 : tensor<256x197x384xf32>
    %v3183 = stablehlo.reshape %v3182 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v3184 = stablehlo.add %v2908, %v3183 : tensor<256x75648xf32>
    %v3185 = stablehlo.reshape %v3184 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v3186 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3187 = stablehlo.constant dense<384.0> : tensor<256x197x384xf32>
    %v3188 = stablehlo.constant dense<1.0e-6> : tensor<256x197x384xf32>
    %v3189 = stablehlo.reduce(%v3185 init: %v3186) applies stablehlo.add across dimensions = [2] : (tensor<256x197x384xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v3190 = stablehlo.broadcast_in_dim %v3189, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x384xf32>
    %v3191 = stablehlo.divide %v3190, %v3187 : tensor<256x197x384xf32>
    %v3192 = stablehlo.subtract %v3185, %v3191 : tensor<256x197x384xf32>
    %v3193 = stablehlo.multiply %v3192, %v3192 : tensor<256x197x384xf32>
    %v3194 = stablehlo.reduce(%v3193 init: %v3186) applies stablehlo.add across dimensions = [2] : (tensor<256x197x384xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v3195 = stablehlo.broadcast_in_dim %v3194, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x384xf32>
    %v3196 = stablehlo.divide %v3195, %v3187 : tensor<256x197x384xf32>
    %v3197 = stablehlo.add %v3196, %v3188 : tensor<256x197x384xf32>
    %v3198 = stablehlo.rsqrt %v3197 : tensor<256x197x384xf32>
    %v3199 = stablehlo.multiply %v3192, %v3198 : tensor<256x197x384xf32>
    %v3200 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<256x197x384xf32>
    %v3201 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<256x197x384xf32>
    %v3202 = stablehlo.multiply %v3199, %v3200 : tensor<256x197x384xf32>
    %v3203 = stablehlo.add %v3202, %v3201 : tensor<256x197x384xf32>
    %v3204 = stablehlo.reshape %v3203 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v3205 = stablehlo.reshape %v3204 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v3206 = stablehlo.broadcast_in_dim %b9_g2, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v3207 = stablehlo.multiply %v3205, %v3206 : tensor<256x197x384xf32>
    %v3208 = stablehlo.reshape %v3207 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v3209 = stablehlo.reshape %v3208 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v3210 = stablehlo.broadcast_in_dim %b9_bt2, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v3211 = stablehlo.add %v3209, %v3210 : tensor<256x197x384xf32>
    %v3212 = stablehlo.reshape %v3211 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v3213 = stablehlo.reshape %v3212 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v3214 = stablehlo.dot_general %v3213, %b9_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x384xf32>, tensor<384x1536xf32>) -> tensor<256x197x1536xf32>
    %v3215 = stablehlo.broadcast_in_dim %b9_bfc1, dims = [2] : (tensor<1536xf32>) -> tensor<256x197x1536xf32>
    %v3216 = stablehlo.add %v3214, %v3215 : tensor<256x197x1536xf32>
    %v3217 = stablehlo.reshape %v3216 : (tensor<256x197x1536xf32>) -> tensor<256x302592xf32>
    %v3218 = stablehlo.constant dense<0.5> : tensor<256x302592xf32>
    %v3219 = stablehlo.multiply %v3218, %v3217 : tensor<256x302592xf32>
    %v3220 = stablehlo.negate %v3217 : tensor<256x302592xf32>
    %v3221 = stablehlo.constant dense<0.7071067811865476> : tensor<256x302592xf32>
    %v3222 = stablehlo.multiply %v3220, %v3221 : tensor<256x302592xf32>
    %v3223 = chlo.erfc %v3222 : tensor<256x302592xf32> -> tensor<256x302592xf32>
    %v3224 = stablehlo.multiply %v3219, %v3223 : tensor<256x302592xf32>
    %v3225 = stablehlo.reshape %v3224 : (tensor<256x302592xf32>) -> tensor<256x197x1536xf32>
    %v3226 = stablehlo.dot_general %v3225, %b9_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x1536xf32>, tensor<1536x384xf32>) -> tensor<256x197x384xf32>
    %v3227 = stablehlo.broadcast_in_dim %b9_bfc2, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v3228 = stablehlo.add %v3226, %v3227 : tensor<256x197x384xf32>
    %v3229 = stablehlo.reshape %v3228 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v3230 = stablehlo.add %v3184, %v3229 : tensor<256x75648xf32>
    %v3231 = stablehlo.reshape %v3230 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v3232 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3233 = stablehlo.constant dense<384.0> : tensor<256x197x384xf32>
    %v3234 = stablehlo.constant dense<1.0e-6> : tensor<256x197x384xf32>
    %v3235 = stablehlo.reduce(%v3231 init: %v3232) applies stablehlo.add across dimensions = [2] : (tensor<256x197x384xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v3236 = stablehlo.broadcast_in_dim %v3235, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x384xf32>
    %v3237 = stablehlo.divide %v3236, %v3233 : tensor<256x197x384xf32>
    %v3238 = stablehlo.subtract %v3231, %v3237 : tensor<256x197x384xf32>
    %v3239 = stablehlo.multiply %v3238, %v3238 : tensor<256x197x384xf32>
    %v3240 = stablehlo.reduce(%v3239 init: %v3232) applies stablehlo.add across dimensions = [2] : (tensor<256x197x384xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v3241 = stablehlo.broadcast_in_dim %v3240, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x384xf32>
    %v3242 = stablehlo.divide %v3241, %v3233 : tensor<256x197x384xf32>
    %v3243 = stablehlo.add %v3242, %v3234 : tensor<256x197x384xf32>
    %v3244 = stablehlo.rsqrt %v3243 : tensor<256x197x384xf32>
    %v3245 = stablehlo.multiply %v3238, %v3244 : tensor<256x197x384xf32>
    %v3246 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<256x197x384xf32>
    %v3247 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<256x197x384xf32>
    %v3248 = stablehlo.multiply %v3245, %v3246 : tensor<256x197x384xf32>
    %v3249 = stablehlo.add %v3248, %v3247 : tensor<256x197x384xf32>
    %v3250 = stablehlo.reshape %v3249 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v3251 = stablehlo.reshape %v3250 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v3252 = stablehlo.broadcast_in_dim %b10_g1, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v3253 = stablehlo.multiply %v3251, %v3252 : tensor<256x197x384xf32>
    %v3254 = stablehlo.reshape %v3253 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v3255 = stablehlo.reshape %v3254 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v3256 = stablehlo.broadcast_in_dim %b10_bt1, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v3257 = stablehlo.add %v3255, %v3256 : tensor<256x197x384xf32>
    %v3258 = stablehlo.reshape %v3257 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v3259 = stablehlo.reshape %v3258 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v3260 = stablehlo.dot_general %v3259, %b10_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x384xf32>, tensor<384x384xf32>) -> tensor<256x197x384xf32>
    %v3261 = stablehlo.broadcast_in_dim %b10_bq, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v3262 = stablehlo.add %v3260, %v3261 : tensor<256x197x384xf32>
    %v3263 = stablehlo.reshape %v3262 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v3264 = stablehlo.reshape %v3258 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v3265 = stablehlo.dot_general %v3264, %b10_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x384xf32>, tensor<384x384xf32>) -> tensor<256x197x384xf32>
    %v3266 = stablehlo.broadcast_in_dim %b10_bk, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v3267 = stablehlo.add %v3265, %v3266 : tensor<256x197x384xf32>
    %v3268 = stablehlo.reshape %v3267 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v3269 = stablehlo.reshape %v3258 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v3270 = stablehlo.dot_general %v3269, %b10_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x384xf32>, tensor<384x384xf32>) -> tensor<256x197x384xf32>
    %v3271 = stablehlo.broadcast_in_dim %b10_bv, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v3272 = stablehlo.add %v3270, %v3271 : tensor<256x197x384xf32>
    %v3273 = stablehlo.reshape %v3272 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v3274 = stablehlo.reshape %v3263 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v3275 = stablehlo.slice %v3274 [0:256, 0:197, 0:64] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v3276 = stablehlo.reshape %v3275 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v3277 = stablehlo.reshape %v3268 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v3278 = stablehlo.slice %v3277 [0:256, 0:197, 0:64] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v3279 = stablehlo.reshape %v3278 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v3280 = stablehlo.reshape %v3273 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v3281 = stablehlo.slice %v3280 [0:256, 0:197, 0:64] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v3282 = stablehlo.reshape %v3281 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v3283 = stablehlo.reshape %v3279 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v3284 = stablehlo.transpose %v3283, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v3285 = stablehlo.reshape %v3284 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v3286 = stablehlo.reshape %v3276 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v3287 = stablehlo.reshape %v3285 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v3288 = stablehlo.dot_general %v3286, %v3287, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v3289 = stablehlo.reshape %v3288 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v3290 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v3291 = stablehlo.multiply %v3289, %v3290 : tensor<256x38809xf32>
    %v3292 = stablehlo.reshape %v3291 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v3293 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v3294 = stablehlo.reduce(%v3292 init: %v3293) applies stablehlo.maximum across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v3295 = stablehlo.broadcast_in_dim %v3294, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v3296 = stablehlo.subtract %v3292, %v3295 : tensor<256x197x197xf32>
    %v3297 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3298 = stablehlo.exponential %v3296 : tensor<256x197x197xf32>
    %v3299 = stablehlo.reduce(%v3298 init: %v3297) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v3300 = stablehlo.broadcast_in_dim %v3299, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v3301 = stablehlo.divide %v3298, %v3300 : tensor<256x197x197xf32>
    %v3302 = stablehlo.reshape %v3301 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v3303 = stablehlo.reshape %v3302 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v3304 = stablehlo.reshape %v3282 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v3305 = stablehlo.dot_general %v3303, %v3304, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v3306 = stablehlo.reshape %v3305 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v3307 = stablehlo.reshape %v3306 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v3308 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3309 = stablehlo.pad %v3307, %v3308, low = [0, 0, 0], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x384xf32>
    %v3310 = stablehlo.reshape %v3309 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v3311 = stablehlo.reshape %v3263 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v3312 = stablehlo.slice %v3311 [0:256, 0:197, 64:128] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v3313 = stablehlo.reshape %v3312 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v3314 = stablehlo.reshape %v3268 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v3315 = stablehlo.slice %v3314 [0:256, 0:197, 64:128] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v3316 = stablehlo.reshape %v3315 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v3317 = stablehlo.reshape %v3273 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v3318 = stablehlo.slice %v3317 [0:256, 0:197, 64:128] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v3319 = stablehlo.reshape %v3318 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v3320 = stablehlo.reshape %v3316 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v3321 = stablehlo.transpose %v3320, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v3322 = stablehlo.reshape %v3321 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v3323 = stablehlo.reshape %v3313 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v3324 = stablehlo.reshape %v3322 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v3325 = stablehlo.dot_general %v3323, %v3324, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v3326 = stablehlo.reshape %v3325 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v3327 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v3328 = stablehlo.multiply %v3326, %v3327 : tensor<256x38809xf32>
    %v3329 = stablehlo.reshape %v3328 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v3330 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v3331 = stablehlo.reduce(%v3329 init: %v3330) applies stablehlo.maximum across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v3332 = stablehlo.broadcast_in_dim %v3331, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v3333 = stablehlo.subtract %v3329, %v3332 : tensor<256x197x197xf32>
    %v3334 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3335 = stablehlo.exponential %v3333 : tensor<256x197x197xf32>
    %v3336 = stablehlo.reduce(%v3335 init: %v3334) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v3337 = stablehlo.broadcast_in_dim %v3336, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v3338 = stablehlo.divide %v3335, %v3337 : tensor<256x197x197xf32>
    %v3339 = stablehlo.reshape %v3338 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v3340 = stablehlo.reshape %v3339 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v3341 = stablehlo.reshape %v3319 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v3342 = stablehlo.dot_general %v3340, %v3341, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v3343 = stablehlo.reshape %v3342 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v3344 = stablehlo.reshape %v3343 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v3345 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3346 = stablehlo.pad %v3344, %v3345, low = [0, 0, 64], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x384xf32>
    %v3347 = stablehlo.reshape %v3346 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v3348 = stablehlo.add %v3310, %v3347 : tensor<256x75648xf32>
    %v3349 = stablehlo.reshape %v3263 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v3350 = stablehlo.slice %v3349 [0:256, 0:197, 128:192] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v3351 = stablehlo.reshape %v3350 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v3352 = stablehlo.reshape %v3268 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v3353 = stablehlo.slice %v3352 [0:256, 0:197, 128:192] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v3354 = stablehlo.reshape %v3353 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v3355 = stablehlo.reshape %v3273 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v3356 = stablehlo.slice %v3355 [0:256, 0:197, 128:192] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v3357 = stablehlo.reshape %v3356 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v3358 = stablehlo.reshape %v3354 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v3359 = stablehlo.transpose %v3358, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v3360 = stablehlo.reshape %v3359 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v3361 = stablehlo.reshape %v3351 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v3362 = stablehlo.reshape %v3360 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v3363 = stablehlo.dot_general %v3361, %v3362, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v3364 = stablehlo.reshape %v3363 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v3365 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v3366 = stablehlo.multiply %v3364, %v3365 : tensor<256x38809xf32>
    %v3367 = stablehlo.reshape %v3366 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v3368 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v3369 = stablehlo.reduce(%v3367 init: %v3368) applies stablehlo.maximum across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v3370 = stablehlo.broadcast_in_dim %v3369, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v3371 = stablehlo.subtract %v3367, %v3370 : tensor<256x197x197xf32>
    %v3372 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3373 = stablehlo.exponential %v3371 : tensor<256x197x197xf32>
    %v3374 = stablehlo.reduce(%v3373 init: %v3372) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v3375 = stablehlo.broadcast_in_dim %v3374, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v3376 = stablehlo.divide %v3373, %v3375 : tensor<256x197x197xf32>
    %v3377 = stablehlo.reshape %v3376 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v3378 = stablehlo.reshape %v3377 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v3379 = stablehlo.reshape %v3357 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v3380 = stablehlo.dot_general %v3378, %v3379, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v3381 = stablehlo.reshape %v3380 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v3382 = stablehlo.reshape %v3381 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v3383 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3384 = stablehlo.pad %v3382, %v3383, low = [0, 0, 128], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x384xf32>
    %v3385 = stablehlo.reshape %v3384 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v3386 = stablehlo.add %v3348, %v3385 : tensor<256x75648xf32>
    %v3387 = stablehlo.reshape %v3263 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v3388 = stablehlo.slice %v3387 [0:256, 0:197, 192:256] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v3389 = stablehlo.reshape %v3388 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v3390 = stablehlo.reshape %v3268 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v3391 = stablehlo.slice %v3390 [0:256, 0:197, 192:256] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v3392 = stablehlo.reshape %v3391 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v3393 = stablehlo.reshape %v3273 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v3394 = stablehlo.slice %v3393 [0:256, 0:197, 192:256] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v3395 = stablehlo.reshape %v3394 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v3396 = stablehlo.reshape %v3392 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v3397 = stablehlo.transpose %v3396, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v3398 = stablehlo.reshape %v3397 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v3399 = stablehlo.reshape %v3389 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v3400 = stablehlo.reshape %v3398 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v3401 = stablehlo.dot_general %v3399, %v3400, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v3402 = stablehlo.reshape %v3401 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v3403 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v3404 = stablehlo.multiply %v3402, %v3403 : tensor<256x38809xf32>
    %v3405 = stablehlo.reshape %v3404 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v3406 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v3407 = stablehlo.reduce(%v3405 init: %v3406) applies stablehlo.maximum across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v3408 = stablehlo.broadcast_in_dim %v3407, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v3409 = stablehlo.subtract %v3405, %v3408 : tensor<256x197x197xf32>
    %v3410 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3411 = stablehlo.exponential %v3409 : tensor<256x197x197xf32>
    %v3412 = stablehlo.reduce(%v3411 init: %v3410) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v3413 = stablehlo.broadcast_in_dim %v3412, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v3414 = stablehlo.divide %v3411, %v3413 : tensor<256x197x197xf32>
    %v3415 = stablehlo.reshape %v3414 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v3416 = stablehlo.reshape %v3415 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v3417 = stablehlo.reshape %v3395 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v3418 = stablehlo.dot_general %v3416, %v3417, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v3419 = stablehlo.reshape %v3418 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v3420 = stablehlo.reshape %v3419 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v3421 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3422 = stablehlo.pad %v3420, %v3421, low = [0, 0, 192], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x384xf32>
    %v3423 = stablehlo.reshape %v3422 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v3424 = stablehlo.add %v3386, %v3423 : tensor<256x75648xf32>
    %v3425 = stablehlo.reshape %v3263 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v3426 = stablehlo.slice %v3425 [0:256, 0:197, 256:320] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v3427 = stablehlo.reshape %v3426 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v3428 = stablehlo.reshape %v3268 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v3429 = stablehlo.slice %v3428 [0:256, 0:197, 256:320] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v3430 = stablehlo.reshape %v3429 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v3431 = stablehlo.reshape %v3273 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v3432 = stablehlo.slice %v3431 [0:256, 0:197, 256:320] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v3433 = stablehlo.reshape %v3432 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v3434 = stablehlo.reshape %v3430 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v3435 = stablehlo.transpose %v3434, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v3436 = stablehlo.reshape %v3435 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v3437 = stablehlo.reshape %v3427 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v3438 = stablehlo.reshape %v3436 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v3439 = stablehlo.dot_general %v3437, %v3438, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v3440 = stablehlo.reshape %v3439 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v3441 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v3442 = stablehlo.multiply %v3440, %v3441 : tensor<256x38809xf32>
    %v3443 = stablehlo.reshape %v3442 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v3444 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v3445 = stablehlo.reduce(%v3443 init: %v3444) applies stablehlo.maximum across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v3446 = stablehlo.broadcast_in_dim %v3445, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v3447 = stablehlo.subtract %v3443, %v3446 : tensor<256x197x197xf32>
    %v3448 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3449 = stablehlo.exponential %v3447 : tensor<256x197x197xf32>
    %v3450 = stablehlo.reduce(%v3449 init: %v3448) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v3451 = stablehlo.broadcast_in_dim %v3450, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v3452 = stablehlo.divide %v3449, %v3451 : tensor<256x197x197xf32>
    %v3453 = stablehlo.reshape %v3452 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v3454 = stablehlo.reshape %v3453 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v3455 = stablehlo.reshape %v3433 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v3456 = stablehlo.dot_general %v3454, %v3455, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v3457 = stablehlo.reshape %v3456 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v3458 = stablehlo.reshape %v3457 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v3459 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3460 = stablehlo.pad %v3458, %v3459, low = [0, 0, 256], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x384xf32>
    %v3461 = stablehlo.reshape %v3460 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v3462 = stablehlo.add %v3424, %v3461 : tensor<256x75648xf32>
    %v3463 = stablehlo.reshape %v3263 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v3464 = stablehlo.slice %v3463 [0:256, 0:197, 320:384] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v3465 = stablehlo.reshape %v3464 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v3466 = stablehlo.reshape %v3268 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v3467 = stablehlo.slice %v3466 [0:256, 0:197, 320:384] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v3468 = stablehlo.reshape %v3467 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v3469 = stablehlo.reshape %v3273 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v3470 = stablehlo.slice %v3469 [0:256, 0:197, 320:384] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v3471 = stablehlo.reshape %v3470 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v3472 = stablehlo.reshape %v3468 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v3473 = stablehlo.transpose %v3472, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v3474 = stablehlo.reshape %v3473 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v3475 = stablehlo.reshape %v3465 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v3476 = stablehlo.reshape %v3474 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v3477 = stablehlo.dot_general %v3475, %v3476, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v3478 = stablehlo.reshape %v3477 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v3479 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v3480 = stablehlo.multiply %v3478, %v3479 : tensor<256x38809xf32>
    %v3481 = stablehlo.reshape %v3480 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v3482 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v3483 = stablehlo.reduce(%v3481 init: %v3482) applies stablehlo.maximum across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v3484 = stablehlo.broadcast_in_dim %v3483, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v3485 = stablehlo.subtract %v3481, %v3484 : tensor<256x197x197xf32>
    %v3486 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3487 = stablehlo.exponential %v3485 : tensor<256x197x197xf32>
    %v3488 = stablehlo.reduce(%v3487 init: %v3486) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v3489 = stablehlo.broadcast_in_dim %v3488, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v3490 = stablehlo.divide %v3487, %v3489 : tensor<256x197x197xf32>
    %v3491 = stablehlo.reshape %v3490 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v3492 = stablehlo.reshape %v3491 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v3493 = stablehlo.reshape %v3471 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v3494 = stablehlo.dot_general %v3492, %v3493, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v3495 = stablehlo.reshape %v3494 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v3496 = stablehlo.reshape %v3495 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v3497 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3498 = stablehlo.pad %v3496, %v3497, low = [0, 0, 320], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x384xf32>
    %v3499 = stablehlo.reshape %v3498 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v3500 = stablehlo.add %v3462, %v3499 : tensor<256x75648xf32>
    %v3501 = stablehlo.reshape %v3500 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v3502 = stablehlo.dot_general %v3501, %b10_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x384xf32>, tensor<384x384xf32>) -> tensor<256x197x384xf32>
    %v3503 = stablehlo.broadcast_in_dim %b10_bo, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v3504 = stablehlo.add %v3502, %v3503 : tensor<256x197x384xf32>
    %v3505 = stablehlo.reshape %v3504 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v3506 = stablehlo.add %v3230, %v3505 : tensor<256x75648xf32>
    %v3507 = stablehlo.reshape %v3506 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v3508 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3509 = stablehlo.constant dense<384.0> : tensor<256x197x384xf32>
    %v3510 = stablehlo.constant dense<1.0e-6> : tensor<256x197x384xf32>
    %v3511 = stablehlo.reduce(%v3507 init: %v3508) applies stablehlo.add across dimensions = [2] : (tensor<256x197x384xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v3512 = stablehlo.broadcast_in_dim %v3511, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x384xf32>
    %v3513 = stablehlo.divide %v3512, %v3509 : tensor<256x197x384xf32>
    %v3514 = stablehlo.subtract %v3507, %v3513 : tensor<256x197x384xf32>
    %v3515 = stablehlo.multiply %v3514, %v3514 : tensor<256x197x384xf32>
    %v3516 = stablehlo.reduce(%v3515 init: %v3508) applies stablehlo.add across dimensions = [2] : (tensor<256x197x384xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v3517 = stablehlo.broadcast_in_dim %v3516, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x384xf32>
    %v3518 = stablehlo.divide %v3517, %v3509 : tensor<256x197x384xf32>
    %v3519 = stablehlo.add %v3518, %v3510 : tensor<256x197x384xf32>
    %v3520 = stablehlo.rsqrt %v3519 : tensor<256x197x384xf32>
    %v3521 = stablehlo.multiply %v3514, %v3520 : tensor<256x197x384xf32>
    %v3522 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<256x197x384xf32>
    %v3523 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<256x197x384xf32>
    %v3524 = stablehlo.multiply %v3521, %v3522 : tensor<256x197x384xf32>
    %v3525 = stablehlo.add %v3524, %v3523 : tensor<256x197x384xf32>
    %v3526 = stablehlo.reshape %v3525 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v3527 = stablehlo.reshape %v3526 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v3528 = stablehlo.broadcast_in_dim %b10_g2, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v3529 = stablehlo.multiply %v3527, %v3528 : tensor<256x197x384xf32>
    %v3530 = stablehlo.reshape %v3529 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v3531 = stablehlo.reshape %v3530 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v3532 = stablehlo.broadcast_in_dim %b10_bt2, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v3533 = stablehlo.add %v3531, %v3532 : tensor<256x197x384xf32>
    %v3534 = stablehlo.reshape %v3533 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v3535 = stablehlo.reshape %v3534 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v3536 = stablehlo.dot_general %v3535, %b10_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x384xf32>, tensor<384x1536xf32>) -> tensor<256x197x1536xf32>
    %v3537 = stablehlo.broadcast_in_dim %b10_bfc1, dims = [2] : (tensor<1536xf32>) -> tensor<256x197x1536xf32>
    %v3538 = stablehlo.add %v3536, %v3537 : tensor<256x197x1536xf32>
    %v3539 = stablehlo.reshape %v3538 : (tensor<256x197x1536xf32>) -> tensor<256x302592xf32>
    %v3540 = stablehlo.constant dense<0.5> : tensor<256x302592xf32>
    %v3541 = stablehlo.multiply %v3540, %v3539 : tensor<256x302592xf32>
    %v3542 = stablehlo.negate %v3539 : tensor<256x302592xf32>
    %v3543 = stablehlo.constant dense<0.7071067811865476> : tensor<256x302592xf32>
    %v3544 = stablehlo.multiply %v3542, %v3543 : tensor<256x302592xf32>
    %v3545 = chlo.erfc %v3544 : tensor<256x302592xf32> -> tensor<256x302592xf32>
    %v3546 = stablehlo.multiply %v3541, %v3545 : tensor<256x302592xf32>
    %v3547 = stablehlo.reshape %v3546 : (tensor<256x302592xf32>) -> tensor<256x197x1536xf32>
    %v3548 = stablehlo.dot_general %v3547, %b10_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x1536xf32>, tensor<1536x384xf32>) -> tensor<256x197x384xf32>
    %v3549 = stablehlo.broadcast_in_dim %b10_bfc2, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v3550 = stablehlo.add %v3548, %v3549 : tensor<256x197x384xf32>
    %v3551 = stablehlo.reshape %v3550 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v3552 = stablehlo.add %v3506, %v3551 : tensor<256x75648xf32>
    %v3553 = stablehlo.reshape %v3552 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v3554 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3555 = stablehlo.constant dense<384.0> : tensor<256x197x384xf32>
    %v3556 = stablehlo.constant dense<1.0e-6> : tensor<256x197x384xf32>
    %v3557 = stablehlo.reduce(%v3553 init: %v3554) applies stablehlo.add across dimensions = [2] : (tensor<256x197x384xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v3558 = stablehlo.broadcast_in_dim %v3557, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x384xf32>
    %v3559 = stablehlo.divide %v3558, %v3555 : tensor<256x197x384xf32>
    %v3560 = stablehlo.subtract %v3553, %v3559 : tensor<256x197x384xf32>
    %v3561 = stablehlo.multiply %v3560, %v3560 : tensor<256x197x384xf32>
    %v3562 = stablehlo.reduce(%v3561 init: %v3554) applies stablehlo.add across dimensions = [2] : (tensor<256x197x384xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v3563 = stablehlo.broadcast_in_dim %v3562, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x384xf32>
    %v3564 = stablehlo.divide %v3563, %v3555 : tensor<256x197x384xf32>
    %v3565 = stablehlo.add %v3564, %v3556 : tensor<256x197x384xf32>
    %v3566 = stablehlo.rsqrt %v3565 : tensor<256x197x384xf32>
    %v3567 = stablehlo.multiply %v3560, %v3566 : tensor<256x197x384xf32>
    %v3568 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<256x197x384xf32>
    %v3569 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<256x197x384xf32>
    %v3570 = stablehlo.multiply %v3567, %v3568 : tensor<256x197x384xf32>
    %v3571 = stablehlo.add %v3570, %v3569 : tensor<256x197x384xf32>
    %v3572 = stablehlo.reshape %v3571 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v3573 = stablehlo.reshape %v3572 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v3574 = stablehlo.broadcast_in_dim %b11_g1, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v3575 = stablehlo.multiply %v3573, %v3574 : tensor<256x197x384xf32>
    %v3576 = stablehlo.reshape %v3575 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v3577 = stablehlo.reshape %v3576 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v3578 = stablehlo.broadcast_in_dim %b11_bt1, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v3579 = stablehlo.add %v3577, %v3578 : tensor<256x197x384xf32>
    %v3580 = stablehlo.reshape %v3579 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v3581 = stablehlo.reshape %v3580 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v3582 = stablehlo.dot_general %v3581, %b11_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x384xf32>, tensor<384x384xf32>) -> tensor<256x197x384xf32>
    %v3583 = stablehlo.broadcast_in_dim %b11_bq, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v3584 = stablehlo.add %v3582, %v3583 : tensor<256x197x384xf32>
    %v3585 = stablehlo.reshape %v3584 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v3586 = stablehlo.reshape %v3580 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v3587 = stablehlo.dot_general %v3586, %b11_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x384xf32>, tensor<384x384xf32>) -> tensor<256x197x384xf32>
    %v3588 = stablehlo.broadcast_in_dim %b11_bk, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v3589 = stablehlo.add %v3587, %v3588 : tensor<256x197x384xf32>
    %v3590 = stablehlo.reshape %v3589 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v3591 = stablehlo.reshape %v3580 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v3592 = stablehlo.dot_general %v3591, %b11_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x384xf32>, tensor<384x384xf32>) -> tensor<256x197x384xf32>
    %v3593 = stablehlo.broadcast_in_dim %b11_bv, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v3594 = stablehlo.add %v3592, %v3593 : tensor<256x197x384xf32>
    %v3595 = stablehlo.reshape %v3594 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v3596 = stablehlo.reshape %v3585 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v3597 = stablehlo.slice %v3596 [0:256, 0:197, 0:64] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v3598 = stablehlo.reshape %v3597 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v3599 = stablehlo.reshape %v3590 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v3600 = stablehlo.slice %v3599 [0:256, 0:197, 0:64] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v3601 = stablehlo.reshape %v3600 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v3602 = stablehlo.reshape %v3595 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v3603 = stablehlo.slice %v3602 [0:256, 0:197, 0:64] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v3604 = stablehlo.reshape %v3603 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v3605 = stablehlo.reshape %v3601 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v3606 = stablehlo.transpose %v3605, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v3607 = stablehlo.reshape %v3606 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v3608 = stablehlo.reshape %v3598 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v3609 = stablehlo.reshape %v3607 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v3610 = stablehlo.dot_general %v3608, %v3609, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v3611 = stablehlo.reshape %v3610 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v3612 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v3613 = stablehlo.multiply %v3611, %v3612 : tensor<256x38809xf32>
    %v3614 = stablehlo.reshape %v3613 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v3615 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v3616 = stablehlo.reduce(%v3614 init: %v3615) applies stablehlo.maximum across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v3617 = stablehlo.broadcast_in_dim %v3616, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v3618 = stablehlo.subtract %v3614, %v3617 : tensor<256x197x197xf32>
    %v3619 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3620 = stablehlo.exponential %v3618 : tensor<256x197x197xf32>
    %v3621 = stablehlo.reduce(%v3620 init: %v3619) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v3622 = stablehlo.broadcast_in_dim %v3621, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v3623 = stablehlo.divide %v3620, %v3622 : tensor<256x197x197xf32>
    %v3624 = stablehlo.reshape %v3623 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v3625 = stablehlo.reshape %v3624 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v3626 = stablehlo.reshape %v3604 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v3627 = stablehlo.dot_general %v3625, %v3626, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v3628 = stablehlo.reshape %v3627 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v3629 = stablehlo.reshape %v3628 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v3630 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3631 = stablehlo.pad %v3629, %v3630, low = [0, 0, 0], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x384xf32>
    %v3632 = stablehlo.reshape %v3631 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v3633 = stablehlo.reshape %v3585 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v3634 = stablehlo.slice %v3633 [0:256, 0:197, 64:128] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v3635 = stablehlo.reshape %v3634 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v3636 = stablehlo.reshape %v3590 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v3637 = stablehlo.slice %v3636 [0:256, 0:197, 64:128] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v3638 = stablehlo.reshape %v3637 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v3639 = stablehlo.reshape %v3595 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v3640 = stablehlo.slice %v3639 [0:256, 0:197, 64:128] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v3641 = stablehlo.reshape %v3640 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v3642 = stablehlo.reshape %v3638 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v3643 = stablehlo.transpose %v3642, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v3644 = stablehlo.reshape %v3643 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v3645 = stablehlo.reshape %v3635 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v3646 = stablehlo.reshape %v3644 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v3647 = stablehlo.dot_general %v3645, %v3646, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v3648 = stablehlo.reshape %v3647 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v3649 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v3650 = stablehlo.multiply %v3648, %v3649 : tensor<256x38809xf32>
    %v3651 = stablehlo.reshape %v3650 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v3652 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v3653 = stablehlo.reduce(%v3651 init: %v3652) applies stablehlo.maximum across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v3654 = stablehlo.broadcast_in_dim %v3653, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v3655 = stablehlo.subtract %v3651, %v3654 : tensor<256x197x197xf32>
    %v3656 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3657 = stablehlo.exponential %v3655 : tensor<256x197x197xf32>
    %v3658 = stablehlo.reduce(%v3657 init: %v3656) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v3659 = stablehlo.broadcast_in_dim %v3658, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v3660 = stablehlo.divide %v3657, %v3659 : tensor<256x197x197xf32>
    %v3661 = stablehlo.reshape %v3660 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v3662 = stablehlo.reshape %v3661 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v3663 = stablehlo.reshape %v3641 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v3664 = stablehlo.dot_general %v3662, %v3663, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v3665 = stablehlo.reshape %v3664 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v3666 = stablehlo.reshape %v3665 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v3667 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3668 = stablehlo.pad %v3666, %v3667, low = [0, 0, 64], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x384xf32>
    %v3669 = stablehlo.reshape %v3668 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v3670 = stablehlo.add %v3632, %v3669 : tensor<256x75648xf32>
    %v3671 = stablehlo.reshape %v3585 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v3672 = stablehlo.slice %v3671 [0:256, 0:197, 128:192] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v3673 = stablehlo.reshape %v3672 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v3674 = stablehlo.reshape %v3590 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v3675 = stablehlo.slice %v3674 [0:256, 0:197, 128:192] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v3676 = stablehlo.reshape %v3675 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v3677 = stablehlo.reshape %v3595 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v3678 = stablehlo.slice %v3677 [0:256, 0:197, 128:192] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v3679 = stablehlo.reshape %v3678 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v3680 = stablehlo.reshape %v3676 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v3681 = stablehlo.transpose %v3680, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v3682 = stablehlo.reshape %v3681 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v3683 = stablehlo.reshape %v3673 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v3684 = stablehlo.reshape %v3682 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v3685 = stablehlo.dot_general %v3683, %v3684, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v3686 = stablehlo.reshape %v3685 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v3687 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v3688 = stablehlo.multiply %v3686, %v3687 : tensor<256x38809xf32>
    %v3689 = stablehlo.reshape %v3688 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v3690 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v3691 = stablehlo.reduce(%v3689 init: %v3690) applies stablehlo.maximum across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v3692 = stablehlo.broadcast_in_dim %v3691, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v3693 = stablehlo.subtract %v3689, %v3692 : tensor<256x197x197xf32>
    %v3694 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3695 = stablehlo.exponential %v3693 : tensor<256x197x197xf32>
    %v3696 = stablehlo.reduce(%v3695 init: %v3694) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v3697 = stablehlo.broadcast_in_dim %v3696, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v3698 = stablehlo.divide %v3695, %v3697 : tensor<256x197x197xf32>
    %v3699 = stablehlo.reshape %v3698 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v3700 = stablehlo.reshape %v3699 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v3701 = stablehlo.reshape %v3679 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v3702 = stablehlo.dot_general %v3700, %v3701, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v3703 = stablehlo.reshape %v3702 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v3704 = stablehlo.reshape %v3703 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v3705 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3706 = stablehlo.pad %v3704, %v3705, low = [0, 0, 128], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x384xf32>
    %v3707 = stablehlo.reshape %v3706 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v3708 = stablehlo.add %v3670, %v3707 : tensor<256x75648xf32>
    %v3709 = stablehlo.reshape %v3585 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v3710 = stablehlo.slice %v3709 [0:256, 0:197, 192:256] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v3711 = stablehlo.reshape %v3710 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v3712 = stablehlo.reshape %v3590 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v3713 = stablehlo.slice %v3712 [0:256, 0:197, 192:256] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v3714 = stablehlo.reshape %v3713 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v3715 = stablehlo.reshape %v3595 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v3716 = stablehlo.slice %v3715 [0:256, 0:197, 192:256] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v3717 = stablehlo.reshape %v3716 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v3718 = stablehlo.reshape %v3714 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v3719 = stablehlo.transpose %v3718, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v3720 = stablehlo.reshape %v3719 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v3721 = stablehlo.reshape %v3711 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v3722 = stablehlo.reshape %v3720 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v3723 = stablehlo.dot_general %v3721, %v3722, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v3724 = stablehlo.reshape %v3723 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v3725 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v3726 = stablehlo.multiply %v3724, %v3725 : tensor<256x38809xf32>
    %v3727 = stablehlo.reshape %v3726 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v3728 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v3729 = stablehlo.reduce(%v3727 init: %v3728) applies stablehlo.maximum across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v3730 = stablehlo.broadcast_in_dim %v3729, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v3731 = stablehlo.subtract %v3727, %v3730 : tensor<256x197x197xf32>
    %v3732 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3733 = stablehlo.exponential %v3731 : tensor<256x197x197xf32>
    %v3734 = stablehlo.reduce(%v3733 init: %v3732) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v3735 = stablehlo.broadcast_in_dim %v3734, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v3736 = stablehlo.divide %v3733, %v3735 : tensor<256x197x197xf32>
    %v3737 = stablehlo.reshape %v3736 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v3738 = stablehlo.reshape %v3737 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v3739 = stablehlo.reshape %v3717 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v3740 = stablehlo.dot_general %v3738, %v3739, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v3741 = stablehlo.reshape %v3740 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v3742 = stablehlo.reshape %v3741 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v3743 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3744 = stablehlo.pad %v3742, %v3743, low = [0, 0, 192], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x384xf32>
    %v3745 = stablehlo.reshape %v3744 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v3746 = stablehlo.add %v3708, %v3745 : tensor<256x75648xf32>
    %v3747 = stablehlo.reshape %v3585 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v3748 = stablehlo.slice %v3747 [0:256, 0:197, 256:320] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v3749 = stablehlo.reshape %v3748 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v3750 = stablehlo.reshape %v3590 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v3751 = stablehlo.slice %v3750 [0:256, 0:197, 256:320] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v3752 = stablehlo.reshape %v3751 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v3753 = stablehlo.reshape %v3595 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v3754 = stablehlo.slice %v3753 [0:256, 0:197, 256:320] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v3755 = stablehlo.reshape %v3754 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v3756 = stablehlo.reshape %v3752 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v3757 = stablehlo.transpose %v3756, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v3758 = stablehlo.reshape %v3757 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v3759 = stablehlo.reshape %v3749 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v3760 = stablehlo.reshape %v3758 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v3761 = stablehlo.dot_general %v3759, %v3760, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v3762 = stablehlo.reshape %v3761 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v3763 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v3764 = stablehlo.multiply %v3762, %v3763 : tensor<256x38809xf32>
    %v3765 = stablehlo.reshape %v3764 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v3766 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v3767 = stablehlo.reduce(%v3765 init: %v3766) applies stablehlo.maximum across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v3768 = stablehlo.broadcast_in_dim %v3767, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v3769 = stablehlo.subtract %v3765, %v3768 : tensor<256x197x197xf32>
    %v3770 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3771 = stablehlo.exponential %v3769 : tensor<256x197x197xf32>
    %v3772 = stablehlo.reduce(%v3771 init: %v3770) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v3773 = stablehlo.broadcast_in_dim %v3772, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v3774 = stablehlo.divide %v3771, %v3773 : tensor<256x197x197xf32>
    %v3775 = stablehlo.reshape %v3774 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v3776 = stablehlo.reshape %v3775 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v3777 = stablehlo.reshape %v3755 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v3778 = stablehlo.dot_general %v3776, %v3777, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v3779 = stablehlo.reshape %v3778 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v3780 = stablehlo.reshape %v3779 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v3781 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3782 = stablehlo.pad %v3780, %v3781, low = [0, 0, 256], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x384xf32>
    %v3783 = stablehlo.reshape %v3782 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v3784 = stablehlo.add %v3746, %v3783 : tensor<256x75648xf32>
    %v3785 = stablehlo.reshape %v3585 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v3786 = stablehlo.slice %v3785 [0:256, 0:197, 320:384] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v3787 = stablehlo.reshape %v3786 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v3788 = stablehlo.reshape %v3590 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v3789 = stablehlo.slice %v3788 [0:256, 0:197, 320:384] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v3790 = stablehlo.reshape %v3789 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v3791 = stablehlo.reshape %v3595 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v3792 = stablehlo.slice %v3791 [0:256, 0:197, 320:384] : (tensor<256x197x384xf32>) -> tensor<256x197x64xf32>
    %v3793 = stablehlo.reshape %v3792 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v3794 = stablehlo.reshape %v3790 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v3795 = stablehlo.transpose %v3794, dims = [0, 2, 1] : (tensor<256x197x64xf32>) -> tensor<256x64x197xf32>
    %v3796 = stablehlo.reshape %v3795 : (tensor<256x64x197xf32>) -> tensor<256x12608xf32>
    %v3797 = stablehlo.reshape %v3787 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v3798 = stablehlo.reshape %v3796 : (tensor<256x12608xf32>) -> tensor<256x64x197xf32>
    %v3799 = stablehlo.dot_general %v3797, %v3798, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x64xf32>, tensor<256x64x197xf32>) -> tensor<256x197x197xf32>
    %v3800 = stablehlo.reshape %v3799 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v3801 = stablehlo.constant dense<0.125> : tensor<256x38809xf32>
    %v3802 = stablehlo.multiply %v3800, %v3801 : tensor<256x38809xf32>
    %v3803 = stablehlo.reshape %v3802 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v3804 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v3805 = stablehlo.reduce(%v3803 init: %v3804) applies stablehlo.maximum across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v3806 = stablehlo.broadcast_in_dim %v3805, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v3807 = stablehlo.subtract %v3803, %v3806 : tensor<256x197x197xf32>
    %v3808 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3809 = stablehlo.exponential %v3807 : tensor<256x197x197xf32>
    %v3810 = stablehlo.reduce(%v3809 init: %v3808) applies stablehlo.add across dimensions = [2] : (tensor<256x197x197xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v3811 = stablehlo.broadcast_in_dim %v3810, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x197xf32>
    %v3812 = stablehlo.divide %v3809, %v3811 : tensor<256x197x197xf32>
    %v3813 = stablehlo.reshape %v3812 : (tensor<256x197x197xf32>) -> tensor<256x38809xf32>
    %v3814 = stablehlo.reshape %v3813 : (tensor<256x38809xf32>) -> tensor<256x197x197xf32>
    %v3815 = stablehlo.reshape %v3793 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v3816 = stablehlo.dot_general %v3814, %v3815, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x197x197xf32>, tensor<256x197x64xf32>) -> tensor<256x197x64xf32>
    %v3817 = stablehlo.reshape %v3816 : (tensor<256x197x64xf32>) -> tensor<256x12608xf32>
    %v3818 = stablehlo.reshape %v3817 : (tensor<256x12608xf32>) -> tensor<256x197x64xf32>
    %v3819 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3820 = stablehlo.pad %v3818, %v3819, low = [0, 0, 320], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<256x197x64xf32>, tensor<f32>) -> tensor<256x197x384xf32>
    %v3821 = stablehlo.reshape %v3820 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v3822 = stablehlo.add %v3784, %v3821 : tensor<256x75648xf32>
    %v3823 = stablehlo.reshape %v3822 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v3824 = stablehlo.dot_general %v3823, %b11_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x384xf32>, tensor<384x384xf32>) -> tensor<256x197x384xf32>
    %v3825 = stablehlo.broadcast_in_dim %b11_bo, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v3826 = stablehlo.add %v3824, %v3825 : tensor<256x197x384xf32>
    %v3827 = stablehlo.reshape %v3826 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v3828 = stablehlo.add %v3552, %v3827 : tensor<256x75648xf32>
    %v3829 = stablehlo.reshape %v3828 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v3830 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3831 = stablehlo.constant dense<384.0> : tensor<256x197x384xf32>
    %v3832 = stablehlo.constant dense<1.0e-6> : tensor<256x197x384xf32>
    %v3833 = stablehlo.reduce(%v3829 init: %v3830) applies stablehlo.add across dimensions = [2] : (tensor<256x197x384xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v3834 = stablehlo.broadcast_in_dim %v3833, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x384xf32>
    %v3835 = stablehlo.divide %v3834, %v3831 : tensor<256x197x384xf32>
    %v3836 = stablehlo.subtract %v3829, %v3835 : tensor<256x197x384xf32>
    %v3837 = stablehlo.multiply %v3836, %v3836 : tensor<256x197x384xf32>
    %v3838 = stablehlo.reduce(%v3837 init: %v3830) applies stablehlo.add across dimensions = [2] : (tensor<256x197x384xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v3839 = stablehlo.broadcast_in_dim %v3838, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x384xf32>
    %v3840 = stablehlo.divide %v3839, %v3831 : tensor<256x197x384xf32>
    %v3841 = stablehlo.add %v3840, %v3832 : tensor<256x197x384xf32>
    %v3842 = stablehlo.rsqrt %v3841 : tensor<256x197x384xf32>
    %v3843 = stablehlo.multiply %v3836, %v3842 : tensor<256x197x384xf32>
    %v3844 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<256x197x384xf32>
    %v3845 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<256x197x384xf32>
    %v3846 = stablehlo.multiply %v3843, %v3844 : tensor<256x197x384xf32>
    %v3847 = stablehlo.add %v3846, %v3845 : tensor<256x197x384xf32>
    %v3848 = stablehlo.reshape %v3847 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v3849 = stablehlo.reshape %v3848 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v3850 = stablehlo.broadcast_in_dim %b11_g2, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v3851 = stablehlo.multiply %v3849, %v3850 : tensor<256x197x384xf32>
    %v3852 = stablehlo.reshape %v3851 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v3853 = stablehlo.reshape %v3852 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v3854 = stablehlo.broadcast_in_dim %b11_bt2, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v3855 = stablehlo.add %v3853, %v3854 : tensor<256x197x384xf32>
    %v3856 = stablehlo.reshape %v3855 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v3857 = stablehlo.reshape %v3856 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v3858 = stablehlo.dot_general %v3857, %b11_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x384xf32>, tensor<384x1536xf32>) -> tensor<256x197x1536xf32>
    %v3859 = stablehlo.broadcast_in_dim %b11_bfc1, dims = [2] : (tensor<1536xf32>) -> tensor<256x197x1536xf32>
    %v3860 = stablehlo.add %v3858, %v3859 : tensor<256x197x1536xf32>
    %v3861 = stablehlo.reshape %v3860 : (tensor<256x197x1536xf32>) -> tensor<256x302592xf32>
    %v3862 = stablehlo.constant dense<0.5> : tensor<256x302592xf32>
    %v3863 = stablehlo.multiply %v3862, %v3861 : tensor<256x302592xf32>
    %v3864 = stablehlo.negate %v3861 : tensor<256x302592xf32>
    %v3865 = stablehlo.constant dense<0.7071067811865476> : tensor<256x302592xf32>
    %v3866 = stablehlo.multiply %v3864, %v3865 : tensor<256x302592xf32>
    %v3867 = chlo.erfc %v3866 : tensor<256x302592xf32> -> tensor<256x302592xf32>
    %v3868 = stablehlo.multiply %v3863, %v3867 : tensor<256x302592xf32>
    %v3869 = stablehlo.reshape %v3868 : (tensor<256x302592xf32>) -> tensor<256x197x1536xf32>
    %v3870 = stablehlo.dot_general %v3869, %b11_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x197x1536xf32>, tensor<1536x384xf32>) -> tensor<256x197x384xf32>
    %v3871 = stablehlo.broadcast_in_dim %b11_bfc2, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v3872 = stablehlo.add %v3870, %v3871 : tensor<256x197x384xf32>
    %v3873 = stablehlo.reshape %v3872 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v3874 = stablehlo.add %v3828, %v3873 : tensor<256x75648xf32>
    %v3875 = stablehlo.reshape %v3874 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v3876 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3877 = stablehlo.constant dense<384.0> : tensor<256x197x384xf32>
    %v3878 = stablehlo.constant dense<1.0e-6> : tensor<256x197x384xf32>
    %v3879 = stablehlo.reduce(%v3875 init: %v3876) applies stablehlo.add across dimensions = [2] : (tensor<256x197x384xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v3880 = stablehlo.broadcast_in_dim %v3879, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x384xf32>
    %v3881 = stablehlo.divide %v3880, %v3877 : tensor<256x197x384xf32>
    %v3882 = stablehlo.subtract %v3875, %v3881 : tensor<256x197x384xf32>
    %v3883 = stablehlo.multiply %v3882, %v3882 : tensor<256x197x384xf32>
    %v3884 = stablehlo.reduce(%v3883 init: %v3876) applies stablehlo.add across dimensions = [2] : (tensor<256x197x384xf32>, tensor<f32>) -> tensor<256x197xf32>
    %v3885 = stablehlo.broadcast_in_dim %v3884, dims = [0, 1] : (tensor<256x197xf32>) -> tensor<256x197x384xf32>
    %v3886 = stablehlo.divide %v3885, %v3877 : tensor<256x197x384xf32>
    %v3887 = stablehlo.add %v3886, %v3878 : tensor<256x197x384xf32>
    %v3888 = stablehlo.rsqrt %v3887 : tensor<256x197x384xf32>
    %v3889 = stablehlo.multiply %v3882, %v3888 : tensor<256x197x384xf32>
    %v3890 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<256x197x384xf32>
    %v3891 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<256x197x384xf32>
    %v3892 = stablehlo.multiply %v3889, %v3890 : tensor<256x197x384xf32>
    %v3893 = stablehlo.add %v3892, %v3891 : tensor<256x197x384xf32>
    %v3894 = stablehlo.reshape %v3893 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v3895 = stablehlo.reshape %v3894 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v3896 = stablehlo.broadcast_in_dim %gF, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v3897 = stablehlo.multiply %v3895, %v3896 : tensor<256x197x384xf32>
    %v3898 = stablehlo.reshape %v3897 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v3899 = stablehlo.reshape %v3898 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v3900 = stablehlo.broadcast_in_dim %btF, dims = [2] : (tensor<384xf32>) -> tensor<256x197x384xf32>
    %v3901 = stablehlo.add %v3899, %v3900 : tensor<256x197x384xf32>
    %v3902 = stablehlo.reshape %v3901 : (tensor<256x197x384xf32>) -> tensor<256x75648xf32>
    %v3903 = stablehlo.reshape %v3902 : (tensor<256x75648xf32>) -> tensor<256x197x384xf32>
    %v3904 = stablehlo.slice %v3903 [0:256, 0:1, 0:384] : (tensor<256x197x384xf32>) -> tensor<256x1x384xf32>
    %v3905 = stablehlo.reshape %v3904 : (tensor<256x1x384xf32>) -> tensor<256x384xf32>
    %v3906 = stablehlo.dot_general %v3905, %Wc, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x384xf32>, tensor<384x1000xf32>) -> tensor<256x1000xf32>
    %v3907 = stablehlo.broadcast_in_dim %bc, dims = [1] : (tensor<1000xf32>) -> tensor<256x1000xf32>
    %v3908 = stablehlo.add %v3906, %v3907 : tensor<256x1000xf32>
    return %v3908 : tensor<256x1000xf32>
  }
}

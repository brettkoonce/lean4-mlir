module @m {
  func.func @vitsin_drop_fwd(%x: tensor<32x150528xf32>, %wConv: tensor<384x3x16x16xf32>, %bConv: tensor<384xf32>, %cls: tensor<384xf32>, %pos: tensor<197x384xf32>, %b0_g1: tensor<384xf32>, %b0_bt1: tensor<384xf32>, %b0_Wq: tensor<384x384xf32>, %b0_bq: tensor<384xf32>, %b0_Wk: tensor<384x384xf32>, %b0_bk: tensor<384xf32>, %b0_Wv: tensor<384x384xf32>, %b0_bv: tensor<384xf32>, %b0_Wo: tensor<384x384xf32>, %b0_bo: tensor<384xf32>, %b0_g2: tensor<384xf32>, %b0_bt2: tensor<384xf32>, %b0_Wfc1: tensor<384x1536xf32>, %b0_bfc1: tensor<1536xf32>, %b0_Wfc2: tensor<1536x384xf32>, %b0_bfc2: tensor<384xf32>, %b1_g1: tensor<384xf32>, %b1_bt1: tensor<384xf32>, %b1_Wq: tensor<384x384xf32>, %b1_bq: tensor<384xf32>, %b1_Wk: tensor<384x384xf32>, %b1_bk: tensor<384xf32>, %b1_Wv: tensor<384x384xf32>, %b1_bv: tensor<384xf32>, %b1_Wo: tensor<384x384xf32>, %b1_bo: tensor<384xf32>, %b1_g2: tensor<384xf32>, %b1_bt2: tensor<384xf32>, %b1_Wfc1: tensor<384x1536xf32>, %b1_bfc1: tensor<1536xf32>, %b1_Wfc2: tensor<1536x384xf32>, %b1_bfc2: tensor<384xf32>, %b2_g1: tensor<384xf32>, %b2_bt1: tensor<384xf32>, %b2_Wq: tensor<384x384xf32>, %b2_bq: tensor<384xf32>, %b2_Wk: tensor<384x384xf32>, %b2_bk: tensor<384xf32>, %b2_Wv: tensor<384x384xf32>, %b2_bv: tensor<384xf32>, %b2_Wo: tensor<384x384xf32>, %b2_bo: tensor<384xf32>, %b2_g2: tensor<384xf32>, %b2_bt2: tensor<384xf32>, %b2_Wfc1: tensor<384x1536xf32>, %b2_bfc1: tensor<1536xf32>, %b2_Wfc2: tensor<1536x384xf32>, %b2_bfc2: tensor<384xf32>, %b3_g1: tensor<384xf32>, %b3_bt1: tensor<384xf32>, %b3_Wq: tensor<384x384xf32>, %b3_bq: tensor<384xf32>, %b3_Wk: tensor<384x384xf32>, %b3_bk: tensor<384xf32>, %b3_Wv: tensor<384x384xf32>, %b3_bv: tensor<384xf32>, %b3_Wo: tensor<384x384xf32>, %b3_bo: tensor<384xf32>, %b3_g2: tensor<384xf32>, %b3_bt2: tensor<384xf32>, %b3_Wfc1: tensor<384x1536xf32>, %b3_bfc1: tensor<1536xf32>, %b3_Wfc2: tensor<1536x384xf32>, %b3_bfc2: tensor<384xf32>, %b4_g1: tensor<384xf32>, %b4_bt1: tensor<384xf32>, %b4_Wq: tensor<384x384xf32>, %b4_bq: tensor<384xf32>, %b4_Wk: tensor<384x384xf32>, %b4_bk: tensor<384xf32>, %b4_Wv: tensor<384x384xf32>, %b4_bv: tensor<384xf32>, %b4_Wo: tensor<384x384xf32>, %b4_bo: tensor<384xf32>, %b4_g2: tensor<384xf32>, %b4_bt2: tensor<384xf32>, %b4_Wfc1: tensor<384x1536xf32>, %b4_bfc1: tensor<1536xf32>, %b4_Wfc2: tensor<1536x384xf32>, %b4_bfc2: tensor<384xf32>, %b5_g1: tensor<384xf32>, %b5_bt1: tensor<384xf32>, %b5_Wq: tensor<384x384xf32>, %b5_bq: tensor<384xf32>, %b5_Wk: tensor<384x384xf32>, %b5_bk: tensor<384xf32>, %b5_Wv: tensor<384x384xf32>, %b5_bv: tensor<384xf32>, %b5_Wo: tensor<384x384xf32>, %b5_bo: tensor<384xf32>, %b5_g2: tensor<384xf32>, %b5_bt2: tensor<384xf32>, %b5_Wfc1: tensor<384x1536xf32>, %b5_bfc1: tensor<1536xf32>, %b5_Wfc2: tensor<1536x384xf32>, %b5_bfc2: tensor<384xf32>, %b6_g1: tensor<384xf32>, %b6_bt1: tensor<384xf32>, %b6_Wq: tensor<384x384xf32>, %b6_bq: tensor<384xf32>, %b6_Wk: tensor<384x384xf32>, %b6_bk: tensor<384xf32>, %b6_Wv: tensor<384x384xf32>, %b6_bv: tensor<384xf32>, %b6_Wo: tensor<384x384xf32>, %b6_bo: tensor<384xf32>, %b6_g2: tensor<384xf32>, %b6_bt2: tensor<384xf32>, %b6_Wfc1: tensor<384x1536xf32>, %b6_bfc1: tensor<1536xf32>, %b6_Wfc2: tensor<1536x384xf32>, %b6_bfc2: tensor<384xf32>, %b7_g1: tensor<384xf32>, %b7_bt1: tensor<384xf32>, %b7_Wq: tensor<384x384xf32>, %b7_bq: tensor<384xf32>, %b7_Wk: tensor<384x384xf32>, %b7_bk: tensor<384xf32>, %b7_Wv: tensor<384x384xf32>, %b7_bv: tensor<384xf32>, %b7_Wo: tensor<384x384xf32>, %b7_bo: tensor<384xf32>, %b7_g2: tensor<384xf32>, %b7_bt2: tensor<384xf32>, %b7_Wfc1: tensor<384x1536xf32>, %b7_bfc1: tensor<1536xf32>, %b7_Wfc2: tensor<1536x384xf32>, %b7_bfc2: tensor<384xf32>, %b8_g1: tensor<384xf32>, %b8_bt1: tensor<384xf32>, %b8_Wq: tensor<384x384xf32>, %b8_bq: tensor<384xf32>, %b8_Wk: tensor<384x384xf32>, %b8_bk: tensor<384xf32>, %b8_Wv: tensor<384x384xf32>, %b8_bv: tensor<384xf32>, %b8_Wo: tensor<384x384xf32>, %b8_bo: tensor<384xf32>, %b8_g2: tensor<384xf32>, %b8_bt2: tensor<384xf32>, %b8_Wfc1: tensor<384x1536xf32>, %b8_bfc1: tensor<1536xf32>, %b8_Wfc2: tensor<1536x384xf32>, %b8_bfc2: tensor<384xf32>, %b9_g1: tensor<384xf32>, %b9_bt1: tensor<384xf32>, %b9_Wq: tensor<384x384xf32>, %b9_bq: tensor<384xf32>, %b9_Wk: tensor<384x384xf32>, %b9_bk: tensor<384xf32>, %b9_Wv: tensor<384x384xf32>, %b9_bv: tensor<384xf32>, %b9_Wo: tensor<384x384xf32>, %b9_bo: tensor<384xf32>, %b9_g2: tensor<384xf32>, %b9_bt2: tensor<384xf32>, %b9_Wfc1: tensor<384x1536xf32>, %b9_bfc1: tensor<1536xf32>, %b9_Wfc2: tensor<1536x384xf32>, %b9_bfc2: tensor<384xf32>, %b10_g1: tensor<384xf32>, %b10_bt1: tensor<384xf32>, %b10_Wq: tensor<384x384xf32>, %b10_bq: tensor<384xf32>, %b10_Wk: tensor<384x384xf32>, %b10_bk: tensor<384xf32>, %b10_Wv: tensor<384x384xf32>, %b10_bv: tensor<384xf32>, %b10_Wo: tensor<384x384xf32>, %b10_bo: tensor<384xf32>, %b10_g2: tensor<384xf32>, %b10_bt2: tensor<384xf32>, %b10_Wfc1: tensor<384x1536xf32>, %b10_bfc1: tensor<1536xf32>, %b10_Wfc2: tensor<1536x384xf32>, %b10_bfc2: tensor<384xf32>, %b11_g1: tensor<384xf32>, %b11_bt1: tensor<384xf32>, %b11_Wq: tensor<384x384xf32>, %b11_bq: tensor<384xf32>, %b11_Wk: tensor<384x384xf32>, %b11_bk: tensor<384xf32>, %b11_Wv: tensor<384x384xf32>, %b11_bv: tensor<384xf32>, %b11_Wo: tensor<384x384xf32>, %b11_bo: tensor<384xf32>, %b11_g2: tensor<384xf32>, %b11_bt2: tensor<384xf32>, %b11_Wfc1: tensor<384x1536xf32>, %b11_bfc1: tensor<1536xf32>, %b11_Wfc2: tensor<1536x384xf32>, %b11_bfc2: tensor<384xf32>, %gF: tensor<384xf32>, %btF: tensor<384xf32>, %Wc: tensor<384x1000xf32>, %bc: tensor<1000xf32>, %dp0: tensor<32xf32>, %dp1: tensor<32xf32>, %dp2: tensor<32xf32>, %dp3: tensor<32xf32>, %dp4: tensor<32xf32>, %dp5: tensor<32xf32>, %dp6: tensor<32xf32>, %dp7: tensor<32xf32>, %dp8: tensor<32xf32>, %dp9: tensor<32xf32>, %dp10: tensor<32xf32>, %dp11: tensor<32xf32>, %dp12: tensor<32xf32>, %dp13: tensor<32xf32>, %dp14: tensor<32xf32>, %dp15: tensor<32xf32>, %dp16: tensor<32xf32>, %dp17: tensor<32xf32>, %dp18: tensor<32xf32>, %dp19: tensor<32xf32>, %dp20: tensor<32xf32>, %dp21: tensor<32xf32>, %dp22: tensor<32xf32>, %dp23: tensor<32xf32>) -> tensor<32x1000xf32> {
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
    %v262 = stablehlo.broadcast_in_dim %dp0, dims = [0] : (tensor<32xf32>) -> tensor<32x75648xf32>
    %v263 = stablehlo.multiply %v262, %v261 : tensor<32x75648xf32>
    %v264 = stablehlo.add %v10, %v263 : tensor<32x75648xf32>
    %v265 = stablehlo.reshape %v264 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v266 = stablehlo.constant dense<0.0> : tensor<f32>
    %v267 = stablehlo.constant dense<384.0> : tensor<32x197x384xf32>
    %v268 = stablehlo.constant dense<1.0e-5> : tensor<32x197x384xf32>
    %v269 = stablehlo.reduce(%v265 init: %v266) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v270 = stablehlo.broadcast_in_dim %v269, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v271 = stablehlo.divide %v270, %v267 : tensor<32x197x384xf32>
    %v272 = stablehlo.subtract %v265, %v271 : tensor<32x197x384xf32>
    %v273 = stablehlo.multiply %v272, %v272 : tensor<32x197x384xf32>
    %v274 = stablehlo.reduce(%v273 init: %v266) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v275 = stablehlo.broadcast_in_dim %v274, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v276 = stablehlo.divide %v275, %v267 : tensor<32x197x384xf32>
    %v277 = stablehlo.add %v276, %v268 : tensor<32x197x384xf32>
    %v278 = stablehlo.rsqrt %v277 : tensor<32x197x384xf32>
    %v279 = stablehlo.multiply %v272, %v278 : tensor<32x197x384xf32>
    %v280 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v281 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v282 = stablehlo.multiply %v279, %v280 : tensor<32x197x384xf32>
    %v283 = stablehlo.add %v282, %v281 : tensor<32x197x384xf32>
    %v284 = stablehlo.reshape %v283 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v285 = stablehlo.reshape %v284 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v286 = stablehlo.broadcast_in_dim %b0_g2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v287 = stablehlo.multiply %v285, %v286 : tensor<32x197x384xf32>
    %v288 = stablehlo.reshape %v287 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v289 = stablehlo.reshape %v288 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v290 = stablehlo.broadcast_in_dim %b0_bt2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v291 = stablehlo.add %v289, %v290 : tensor<32x197x384xf32>
    %v292 = stablehlo.reshape %v291 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v293 = stablehlo.reshape %v292 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v294 = stablehlo.dot_general %v293, %b0_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x1536xf32>) -> tensor<32x197x1536xf32>
    %v295 = stablehlo.broadcast_in_dim %b0_bfc1, dims = [2] : (tensor<1536xf32>) -> tensor<32x197x1536xf32>
    %v296 = stablehlo.add %v294, %v295 : tensor<32x197x1536xf32>
    %v297 = stablehlo.reshape %v296 : (tensor<32x197x1536xf32>) -> tensor<32x302592xf32>
    %v298 = stablehlo.multiply %v297, %v297 : tensor<32x302592xf32>
    %v299 = stablehlo.multiply %v298, %v297 : tensor<32x302592xf32>
    %v300 = stablehlo.constant dense<0.044715> : tensor<32x302592xf32>
    %v301 = stablehlo.multiply %v300, %v299 : tensor<32x302592xf32>
    %v302 = stablehlo.add %v297, %v301 : tensor<32x302592xf32>
    %v303 = stablehlo.constant dense<0.7978845608028654> : tensor<32x302592xf32>
    %v304 = stablehlo.multiply %v303, %v302 : tensor<32x302592xf32>
    %v305 = stablehlo.tanh %v304 : tensor<32x302592xf32>
    %v306 = stablehlo.constant dense<1.0> : tensor<32x302592xf32>
    %v307 = stablehlo.add %v306, %v305 : tensor<32x302592xf32>
    %v308 = stablehlo.constant dense<0.5> : tensor<32x302592xf32>
    %v309 = stablehlo.multiply %v308, %v297 : tensor<32x302592xf32>
    %v310 = stablehlo.multiply %v309, %v307 : tensor<32x302592xf32>
    %v311 = stablehlo.reshape %v310 : (tensor<32x302592xf32>) -> tensor<32x197x1536xf32>
    %v312 = stablehlo.dot_general %v311, %b0_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x1536xf32>, tensor<1536x384xf32>) -> tensor<32x197x384xf32>
    %v313 = stablehlo.broadcast_in_dim %b0_bfc2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v314 = stablehlo.add %v312, %v313 : tensor<32x197x384xf32>
    %v315 = stablehlo.reshape %v314 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v316 = stablehlo.broadcast_in_dim %dp1, dims = [0] : (tensor<32xf32>) -> tensor<32x75648xf32>
    %v317 = stablehlo.multiply %v316, %v315 : tensor<32x75648xf32>
    %v318 = stablehlo.add %v264, %v317 : tensor<32x75648xf32>
    %v319 = stablehlo.reshape %v318 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v320 = stablehlo.constant dense<0.0> : tensor<f32>
    %v321 = stablehlo.constant dense<384.0> : tensor<32x197x384xf32>
    %v322 = stablehlo.constant dense<1.0e-5> : tensor<32x197x384xf32>
    %v323 = stablehlo.reduce(%v319 init: %v320) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v324 = stablehlo.broadcast_in_dim %v323, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v325 = stablehlo.divide %v324, %v321 : tensor<32x197x384xf32>
    %v326 = stablehlo.subtract %v319, %v325 : tensor<32x197x384xf32>
    %v327 = stablehlo.multiply %v326, %v326 : tensor<32x197x384xf32>
    %v328 = stablehlo.reduce(%v327 init: %v320) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v329 = stablehlo.broadcast_in_dim %v328, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v330 = stablehlo.divide %v329, %v321 : tensor<32x197x384xf32>
    %v331 = stablehlo.add %v330, %v322 : tensor<32x197x384xf32>
    %v332 = stablehlo.rsqrt %v331 : tensor<32x197x384xf32>
    %v333 = stablehlo.multiply %v326, %v332 : tensor<32x197x384xf32>
    %v334 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v335 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v336 = stablehlo.multiply %v333, %v334 : tensor<32x197x384xf32>
    %v337 = stablehlo.add %v336, %v335 : tensor<32x197x384xf32>
    %v338 = stablehlo.reshape %v337 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v339 = stablehlo.reshape %v338 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v340 = stablehlo.broadcast_in_dim %b1_g1, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v341 = stablehlo.multiply %v339, %v340 : tensor<32x197x384xf32>
    %v342 = stablehlo.reshape %v341 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v343 = stablehlo.reshape %v342 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v344 = stablehlo.broadcast_in_dim %b1_bt1, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v345 = stablehlo.add %v343, %v344 : tensor<32x197x384xf32>
    %v346 = stablehlo.reshape %v345 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v347 = stablehlo.reshape %v346 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v348 = stablehlo.dot_general %v347, %b1_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v349 = stablehlo.broadcast_in_dim %b1_bq, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v350 = stablehlo.add %v348, %v349 : tensor<32x197x384xf32>
    %v351 = stablehlo.reshape %v350 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v352 = stablehlo.reshape %v346 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v353 = stablehlo.dot_general %v352, %b1_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v354 = stablehlo.broadcast_in_dim %b1_bk, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v355 = stablehlo.add %v353, %v354 : tensor<32x197x384xf32>
    %v356 = stablehlo.reshape %v355 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v357 = stablehlo.reshape %v346 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v358 = stablehlo.dot_general %v357, %b1_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v359 = stablehlo.broadcast_in_dim %b1_bv, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v360 = stablehlo.add %v358, %v359 : tensor<32x197x384xf32>
    %v361 = stablehlo.reshape %v360 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v362 = stablehlo.reshape %v351 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v363 = stablehlo.slice %v362 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v364 = stablehlo.reshape %v363 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v365 = stablehlo.reshape %v356 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v366 = stablehlo.slice %v365 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v367 = stablehlo.reshape %v366 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v368 = stablehlo.reshape %v361 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v369 = stablehlo.slice %v368 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v370 = stablehlo.reshape %v369 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v371 = stablehlo.reshape %v367 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v372 = stablehlo.transpose %v371, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v373 = stablehlo.reshape %v372 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v374 = stablehlo.reshape %v364 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v375 = stablehlo.reshape %v373 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v376 = stablehlo.dot_general %v374, %v375, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v377 = stablehlo.reshape %v376 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v378 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v379 = stablehlo.multiply %v377, %v378 : tensor<32x38809xf32>
    %v380 = stablehlo.reshape %v379 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v381 = stablehlo.constant dense<0.0> : tensor<f32>
    %v382 = stablehlo.exponential %v380 : tensor<32x197x197xf32>
    %v383 = stablehlo.reduce(%v382 init: %v381) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v384 = stablehlo.broadcast_in_dim %v383, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v385 = stablehlo.divide %v382, %v384 : tensor<32x197x197xf32>
    %v386 = stablehlo.reshape %v385 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v387 = stablehlo.reshape %v386 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v388 = stablehlo.reshape %v370 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v389 = stablehlo.dot_general %v387, %v388, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v390 = stablehlo.reshape %v389 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v391 = stablehlo.reshape %v390 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v392 = stablehlo.constant dense<0.0> : tensor<f32>
    %v393 = stablehlo.pad %v391, %v392, low = [0, 0, 0], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v394 = stablehlo.reshape %v393 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v395 = stablehlo.reshape %v351 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v396 = stablehlo.slice %v395 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v397 = stablehlo.reshape %v396 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v398 = stablehlo.reshape %v356 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v399 = stablehlo.slice %v398 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v400 = stablehlo.reshape %v399 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v401 = stablehlo.reshape %v361 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v402 = stablehlo.slice %v401 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
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
    %v414 = stablehlo.constant dense<0.0> : tensor<f32>
    %v415 = stablehlo.exponential %v413 : tensor<32x197x197xf32>
    %v416 = stablehlo.reduce(%v415 init: %v414) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v417 = stablehlo.broadcast_in_dim %v416, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v418 = stablehlo.divide %v415, %v417 : tensor<32x197x197xf32>
    %v419 = stablehlo.reshape %v418 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v420 = stablehlo.reshape %v419 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v421 = stablehlo.reshape %v403 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v422 = stablehlo.dot_general %v420, %v421, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v423 = stablehlo.reshape %v422 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v424 = stablehlo.reshape %v423 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v425 = stablehlo.constant dense<0.0> : tensor<f32>
    %v426 = stablehlo.pad %v424, %v425, low = [0, 0, 64], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v427 = stablehlo.reshape %v426 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v428 = stablehlo.add %v394, %v427 : tensor<32x75648xf32>
    %v429 = stablehlo.reshape %v351 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v430 = stablehlo.slice %v429 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v431 = stablehlo.reshape %v430 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v432 = stablehlo.reshape %v356 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v433 = stablehlo.slice %v432 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v434 = stablehlo.reshape %v433 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v435 = stablehlo.reshape %v361 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v436 = stablehlo.slice %v435 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v437 = stablehlo.reshape %v436 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v438 = stablehlo.reshape %v434 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v439 = stablehlo.transpose %v438, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v440 = stablehlo.reshape %v439 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v441 = stablehlo.reshape %v431 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v442 = stablehlo.reshape %v440 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v443 = stablehlo.dot_general %v441, %v442, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v444 = stablehlo.reshape %v443 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v445 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v446 = stablehlo.multiply %v444, %v445 : tensor<32x38809xf32>
    %v447 = stablehlo.reshape %v446 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v448 = stablehlo.constant dense<0.0> : tensor<f32>
    %v449 = stablehlo.exponential %v447 : tensor<32x197x197xf32>
    %v450 = stablehlo.reduce(%v449 init: %v448) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v451 = stablehlo.broadcast_in_dim %v450, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v452 = stablehlo.divide %v449, %v451 : tensor<32x197x197xf32>
    %v453 = stablehlo.reshape %v452 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v454 = stablehlo.reshape %v453 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v455 = stablehlo.reshape %v437 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v456 = stablehlo.dot_general %v454, %v455, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v457 = stablehlo.reshape %v456 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v458 = stablehlo.reshape %v457 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v459 = stablehlo.constant dense<0.0> : tensor<f32>
    %v460 = stablehlo.pad %v458, %v459, low = [0, 0, 128], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v461 = stablehlo.reshape %v460 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v462 = stablehlo.add %v428, %v461 : tensor<32x75648xf32>
    %v463 = stablehlo.reshape %v351 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v464 = stablehlo.slice %v463 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v465 = stablehlo.reshape %v464 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v466 = stablehlo.reshape %v356 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v467 = stablehlo.slice %v466 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v468 = stablehlo.reshape %v467 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v469 = stablehlo.reshape %v361 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v470 = stablehlo.slice %v469 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v471 = stablehlo.reshape %v470 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v472 = stablehlo.reshape %v468 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v473 = stablehlo.transpose %v472, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v474 = stablehlo.reshape %v473 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v475 = stablehlo.reshape %v465 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v476 = stablehlo.reshape %v474 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v477 = stablehlo.dot_general %v475, %v476, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v478 = stablehlo.reshape %v477 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v479 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v480 = stablehlo.multiply %v478, %v479 : tensor<32x38809xf32>
    %v481 = stablehlo.reshape %v480 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v482 = stablehlo.constant dense<0.0> : tensor<f32>
    %v483 = stablehlo.exponential %v481 : tensor<32x197x197xf32>
    %v484 = stablehlo.reduce(%v483 init: %v482) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v485 = stablehlo.broadcast_in_dim %v484, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v486 = stablehlo.divide %v483, %v485 : tensor<32x197x197xf32>
    %v487 = stablehlo.reshape %v486 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v488 = stablehlo.reshape %v487 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v489 = stablehlo.reshape %v471 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v490 = stablehlo.dot_general %v488, %v489, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v491 = stablehlo.reshape %v490 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v492 = stablehlo.reshape %v491 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v493 = stablehlo.constant dense<0.0> : tensor<f32>
    %v494 = stablehlo.pad %v492, %v493, low = [0, 0, 192], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v495 = stablehlo.reshape %v494 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v496 = stablehlo.add %v462, %v495 : tensor<32x75648xf32>
    %v497 = stablehlo.reshape %v351 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v498 = stablehlo.slice %v497 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v499 = stablehlo.reshape %v498 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v500 = stablehlo.reshape %v356 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v501 = stablehlo.slice %v500 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v502 = stablehlo.reshape %v501 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v503 = stablehlo.reshape %v361 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v504 = stablehlo.slice %v503 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v505 = stablehlo.reshape %v504 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v506 = stablehlo.reshape %v502 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v507 = stablehlo.transpose %v506, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v508 = stablehlo.reshape %v507 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v509 = stablehlo.reshape %v499 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v510 = stablehlo.reshape %v508 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v511 = stablehlo.dot_general %v509, %v510, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v512 = stablehlo.reshape %v511 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v513 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v514 = stablehlo.multiply %v512, %v513 : tensor<32x38809xf32>
    %v515 = stablehlo.reshape %v514 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v516 = stablehlo.constant dense<0.0> : tensor<f32>
    %v517 = stablehlo.exponential %v515 : tensor<32x197x197xf32>
    %v518 = stablehlo.reduce(%v517 init: %v516) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v519 = stablehlo.broadcast_in_dim %v518, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v520 = stablehlo.divide %v517, %v519 : tensor<32x197x197xf32>
    %v521 = stablehlo.reshape %v520 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v522 = stablehlo.reshape %v521 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v523 = stablehlo.reshape %v505 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v524 = stablehlo.dot_general %v522, %v523, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v525 = stablehlo.reshape %v524 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v526 = stablehlo.reshape %v525 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v527 = stablehlo.constant dense<0.0> : tensor<f32>
    %v528 = stablehlo.pad %v526, %v527, low = [0, 0, 256], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v529 = stablehlo.reshape %v528 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v530 = stablehlo.add %v496, %v529 : tensor<32x75648xf32>
    %v531 = stablehlo.reshape %v351 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v532 = stablehlo.slice %v531 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v533 = stablehlo.reshape %v532 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v534 = stablehlo.reshape %v356 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v535 = stablehlo.slice %v534 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v536 = stablehlo.reshape %v535 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v537 = stablehlo.reshape %v361 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v538 = stablehlo.slice %v537 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v539 = stablehlo.reshape %v538 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v540 = stablehlo.reshape %v536 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v541 = stablehlo.transpose %v540, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v542 = stablehlo.reshape %v541 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v543 = stablehlo.reshape %v533 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v544 = stablehlo.reshape %v542 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v545 = stablehlo.dot_general %v543, %v544, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v546 = stablehlo.reshape %v545 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v547 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v548 = stablehlo.multiply %v546, %v547 : tensor<32x38809xf32>
    %v549 = stablehlo.reshape %v548 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v550 = stablehlo.constant dense<0.0> : tensor<f32>
    %v551 = stablehlo.exponential %v549 : tensor<32x197x197xf32>
    %v552 = stablehlo.reduce(%v551 init: %v550) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v553 = stablehlo.broadcast_in_dim %v552, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v554 = stablehlo.divide %v551, %v553 : tensor<32x197x197xf32>
    %v555 = stablehlo.reshape %v554 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v556 = stablehlo.reshape %v555 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v557 = stablehlo.reshape %v539 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v558 = stablehlo.dot_general %v556, %v557, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v559 = stablehlo.reshape %v558 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v560 = stablehlo.reshape %v559 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v561 = stablehlo.constant dense<0.0> : tensor<f32>
    %v562 = stablehlo.pad %v560, %v561, low = [0, 0, 320], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v563 = stablehlo.reshape %v562 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v564 = stablehlo.add %v530, %v563 : tensor<32x75648xf32>
    %v565 = stablehlo.reshape %v564 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v566 = stablehlo.dot_general %v565, %b1_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v567 = stablehlo.broadcast_in_dim %b1_bo, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v568 = stablehlo.add %v566, %v567 : tensor<32x197x384xf32>
    %v569 = stablehlo.reshape %v568 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v570 = stablehlo.broadcast_in_dim %dp2, dims = [0] : (tensor<32xf32>) -> tensor<32x75648xf32>
    %v571 = stablehlo.multiply %v570, %v569 : tensor<32x75648xf32>
    %v572 = stablehlo.add %v318, %v571 : tensor<32x75648xf32>
    %v573 = stablehlo.reshape %v572 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v574 = stablehlo.constant dense<0.0> : tensor<f32>
    %v575 = stablehlo.constant dense<384.0> : tensor<32x197x384xf32>
    %v576 = stablehlo.constant dense<1.0e-5> : tensor<32x197x384xf32>
    %v577 = stablehlo.reduce(%v573 init: %v574) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v578 = stablehlo.broadcast_in_dim %v577, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v579 = stablehlo.divide %v578, %v575 : tensor<32x197x384xf32>
    %v580 = stablehlo.subtract %v573, %v579 : tensor<32x197x384xf32>
    %v581 = stablehlo.multiply %v580, %v580 : tensor<32x197x384xf32>
    %v582 = stablehlo.reduce(%v581 init: %v574) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v583 = stablehlo.broadcast_in_dim %v582, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v584 = stablehlo.divide %v583, %v575 : tensor<32x197x384xf32>
    %v585 = stablehlo.add %v584, %v576 : tensor<32x197x384xf32>
    %v586 = stablehlo.rsqrt %v585 : tensor<32x197x384xf32>
    %v587 = stablehlo.multiply %v580, %v586 : tensor<32x197x384xf32>
    %v588 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v589 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v590 = stablehlo.multiply %v587, %v588 : tensor<32x197x384xf32>
    %v591 = stablehlo.add %v590, %v589 : tensor<32x197x384xf32>
    %v592 = stablehlo.reshape %v591 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v593 = stablehlo.reshape %v592 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v594 = stablehlo.broadcast_in_dim %b1_g2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v595 = stablehlo.multiply %v593, %v594 : tensor<32x197x384xf32>
    %v596 = stablehlo.reshape %v595 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v597 = stablehlo.reshape %v596 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v598 = stablehlo.broadcast_in_dim %b1_bt2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v599 = stablehlo.add %v597, %v598 : tensor<32x197x384xf32>
    %v600 = stablehlo.reshape %v599 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v601 = stablehlo.reshape %v600 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v602 = stablehlo.dot_general %v601, %b1_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x1536xf32>) -> tensor<32x197x1536xf32>
    %v603 = stablehlo.broadcast_in_dim %b1_bfc1, dims = [2] : (tensor<1536xf32>) -> tensor<32x197x1536xf32>
    %v604 = stablehlo.add %v602, %v603 : tensor<32x197x1536xf32>
    %v605 = stablehlo.reshape %v604 : (tensor<32x197x1536xf32>) -> tensor<32x302592xf32>
    %v606 = stablehlo.multiply %v605, %v605 : tensor<32x302592xf32>
    %v607 = stablehlo.multiply %v606, %v605 : tensor<32x302592xf32>
    %v608 = stablehlo.constant dense<0.044715> : tensor<32x302592xf32>
    %v609 = stablehlo.multiply %v608, %v607 : tensor<32x302592xf32>
    %v610 = stablehlo.add %v605, %v609 : tensor<32x302592xf32>
    %v611 = stablehlo.constant dense<0.7978845608028654> : tensor<32x302592xf32>
    %v612 = stablehlo.multiply %v611, %v610 : tensor<32x302592xf32>
    %v613 = stablehlo.tanh %v612 : tensor<32x302592xf32>
    %v614 = stablehlo.constant dense<1.0> : tensor<32x302592xf32>
    %v615 = stablehlo.add %v614, %v613 : tensor<32x302592xf32>
    %v616 = stablehlo.constant dense<0.5> : tensor<32x302592xf32>
    %v617 = stablehlo.multiply %v616, %v605 : tensor<32x302592xf32>
    %v618 = stablehlo.multiply %v617, %v615 : tensor<32x302592xf32>
    %v619 = stablehlo.reshape %v618 : (tensor<32x302592xf32>) -> tensor<32x197x1536xf32>
    %v620 = stablehlo.dot_general %v619, %b1_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x1536xf32>, tensor<1536x384xf32>) -> tensor<32x197x384xf32>
    %v621 = stablehlo.broadcast_in_dim %b1_bfc2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v622 = stablehlo.add %v620, %v621 : tensor<32x197x384xf32>
    %v623 = stablehlo.reshape %v622 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v624 = stablehlo.broadcast_in_dim %dp3, dims = [0] : (tensor<32xf32>) -> tensor<32x75648xf32>
    %v625 = stablehlo.multiply %v624, %v623 : tensor<32x75648xf32>
    %v626 = stablehlo.add %v572, %v625 : tensor<32x75648xf32>
    %v627 = stablehlo.reshape %v626 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v628 = stablehlo.constant dense<0.0> : tensor<f32>
    %v629 = stablehlo.constant dense<384.0> : tensor<32x197x384xf32>
    %v630 = stablehlo.constant dense<1.0e-5> : tensor<32x197x384xf32>
    %v631 = stablehlo.reduce(%v627 init: %v628) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v632 = stablehlo.broadcast_in_dim %v631, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v633 = stablehlo.divide %v632, %v629 : tensor<32x197x384xf32>
    %v634 = stablehlo.subtract %v627, %v633 : tensor<32x197x384xf32>
    %v635 = stablehlo.multiply %v634, %v634 : tensor<32x197x384xf32>
    %v636 = stablehlo.reduce(%v635 init: %v628) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v637 = stablehlo.broadcast_in_dim %v636, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v638 = stablehlo.divide %v637, %v629 : tensor<32x197x384xf32>
    %v639 = stablehlo.add %v638, %v630 : tensor<32x197x384xf32>
    %v640 = stablehlo.rsqrt %v639 : tensor<32x197x384xf32>
    %v641 = stablehlo.multiply %v634, %v640 : tensor<32x197x384xf32>
    %v642 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v643 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v644 = stablehlo.multiply %v641, %v642 : tensor<32x197x384xf32>
    %v645 = stablehlo.add %v644, %v643 : tensor<32x197x384xf32>
    %v646 = stablehlo.reshape %v645 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v647 = stablehlo.reshape %v646 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v648 = stablehlo.broadcast_in_dim %b2_g1, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v649 = stablehlo.multiply %v647, %v648 : tensor<32x197x384xf32>
    %v650 = stablehlo.reshape %v649 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v651 = stablehlo.reshape %v650 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v652 = stablehlo.broadcast_in_dim %b2_bt1, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v653 = stablehlo.add %v651, %v652 : tensor<32x197x384xf32>
    %v654 = stablehlo.reshape %v653 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v655 = stablehlo.reshape %v654 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v656 = stablehlo.dot_general %v655, %b2_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v657 = stablehlo.broadcast_in_dim %b2_bq, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v658 = stablehlo.add %v656, %v657 : tensor<32x197x384xf32>
    %v659 = stablehlo.reshape %v658 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v660 = stablehlo.reshape %v654 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v661 = stablehlo.dot_general %v660, %b2_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v662 = stablehlo.broadcast_in_dim %b2_bk, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v663 = stablehlo.add %v661, %v662 : tensor<32x197x384xf32>
    %v664 = stablehlo.reshape %v663 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v665 = stablehlo.reshape %v654 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v666 = stablehlo.dot_general %v665, %b2_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v667 = stablehlo.broadcast_in_dim %b2_bv, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v668 = stablehlo.add %v666, %v667 : tensor<32x197x384xf32>
    %v669 = stablehlo.reshape %v668 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v670 = stablehlo.reshape %v659 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v671 = stablehlo.slice %v670 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v672 = stablehlo.reshape %v671 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v673 = stablehlo.reshape %v664 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v674 = stablehlo.slice %v673 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v675 = stablehlo.reshape %v674 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v676 = stablehlo.reshape %v669 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v677 = stablehlo.slice %v676 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v678 = stablehlo.reshape %v677 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v679 = stablehlo.reshape %v675 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v680 = stablehlo.transpose %v679, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v681 = stablehlo.reshape %v680 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v682 = stablehlo.reshape %v672 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v683 = stablehlo.reshape %v681 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v684 = stablehlo.dot_general %v682, %v683, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v685 = stablehlo.reshape %v684 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v686 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v687 = stablehlo.multiply %v685, %v686 : tensor<32x38809xf32>
    %v688 = stablehlo.reshape %v687 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v689 = stablehlo.constant dense<0.0> : tensor<f32>
    %v690 = stablehlo.exponential %v688 : tensor<32x197x197xf32>
    %v691 = stablehlo.reduce(%v690 init: %v689) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v692 = stablehlo.broadcast_in_dim %v691, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v693 = stablehlo.divide %v690, %v692 : tensor<32x197x197xf32>
    %v694 = stablehlo.reshape %v693 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v695 = stablehlo.reshape %v694 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v696 = stablehlo.reshape %v678 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v697 = stablehlo.dot_general %v695, %v696, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v698 = stablehlo.reshape %v697 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v699 = stablehlo.reshape %v698 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v700 = stablehlo.constant dense<0.0> : tensor<f32>
    %v701 = stablehlo.pad %v699, %v700, low = [0, 0, 0], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v702 = stablehlo.reshape %v701 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v703 = stablehlo.reshape %v659 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v704 = stablehlo.slice %v703 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v705 = stablehlo.reshape %v704 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v706 = stablehlo.reshape %v664 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v707 = stablehlo.slice %v706 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v708 = stablehlo.reshape %v707 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v709 = stablehlo.reshape %v669 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v710 = stablehlo.slice %v709 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v711 = stablehlo.reshape %v710 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v712 = stablehlo.reshape %v708 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v713 = stablehlo.transpose %v712, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v714 = stablehlo.reshape %v713 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v715 = stablehlo.reshape %v705 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v716 = stablehlo.reshape %v714 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v717 = stablehlo.dot_general %v715, %v716, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v718 = stablehlo.reshape %v717 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v719 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v720 = stablehlo.multiply %v718, %v719 : tensor<32x38809xf32>
    %v721 = stablehlo.reshape %v720 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v722 = stablehlo.constant dense<0.0> : tensor<f32>
    %v723 = stablehlo.exponential %v721 : tensor<32x197x197xf32>
    %v724 = stablehlo.reduce(%v723 init: %v722) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v725 = stablehlo.broadcast_in_dim %v724, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v726 = stablehlo.divide %v723, %v725 : tensor<32x197x197xf32>
    %v727 = stablehlo.reshape %v726 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v728 = stablehlo.reshape %v727 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v729 = stablehlo.reshape %v711 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v730 = stablehlo.dot_general %v728, %v729, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v731 = stablehlo.reshape %v730 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v732 = stablehlo.reshape %v731 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v733 = stablehlo.constant dense<0.0> : tensor<f32>
    %v734 = stablehlo.pad %v732, %v733, low = [0, 0, 64], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v735 = stablehlo.reshape %v734 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v736 = stablehlo.add %v702, %v735 : tensor<32x75648xf32>
    %v737 = stablehlo.reshape %v659 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v738 = stablehlo.slice %v737 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v739 = stablehlo.reshape %v738 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v740 = stablehlo.reshape %v664 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v741 = stablehlo.slice %v740 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v742 = stablehlo.reshape %v741 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v743 = stablehlo.reshape %v669 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v744 = stablehlo.slice %v743 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v745 = stablehlo.reshape %v744 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v746 = stablehlo.reshape %v742 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v747 = stablehlo.transpose %v746, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v748 = stablehlo.reshape %v747 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v749 = stablehlo.reshape %v739 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v750 = stablehlo.reshape %v748 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v751 = stablehlo.dot_general %v749, %v750, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v752 = stablehlo.reshape %v751 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v753 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v754 = stablehlo.multiply %v752, %v753 : tensor<32x38809xf32>
    %v755 = stablehlo.reshape %v754 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v756 = stablehlo.constant dense<0.0> : tensor<f32>
    %v757 = stablehlo.exponential %v755 : tensor<32x197x197xf32>
    %v758 = stablehlo.reduce(%v757 init: %v756) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v759 = stablehlo.broadcast_in_dim %v758, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v760 = stablehlo.divide %v757, %v759 : tensor<32x197x197xf32>
    %v761 = stablehlo.reshape %v760 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v762 = stablehlo.reshape %v761 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v763 = stablehlo.reshape %v745 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v764 = stablehlo.dot_general %v762, %v763, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v765 = stablehlo.reshape %v764 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v766 = stablehlo.reshape %v765 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v767 = stablehlo.constant dense<0.0> : tensor<f32>
    %v768 = stablehlo.pad %v766, %v767, low = [0, 0, 128], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v769 = stablehlo.reshape %v768 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v770 = stablehlo.add %v736, %v769 : tensor<32x75648xf32>
    %v771 = stablehlo.reshape %v659 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v772 = stablehlo.slice %v771 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v773 = stablehlo.reshape %v772 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v774 = stablehlo.reshape %v664 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v775 = stablehlo.slice %v774 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v776 = stablehlo.reshape %v775 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v777 = stablehlo.reshape %v669 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v778 = stablehlo.slice %v777 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v779 = stablehlo.reshape %v778 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v780 = stablehlo.reshape %v776 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v781 = stablehlo.transpose %v780, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v782 = stablehlo.reshape %v781 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v783 = stablehlo.reshape %v773 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v784 = stablehlo.reshape %v782 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v785 = stablehlo.dot_general %v783, %v784, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v786 = stablehlo.reshape %v785 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v787 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v788 = stablehlo.multiply %v786, %v787 : tensor<32x38809xf32>
    %v789 = stablehlo.reshape %v788 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v790 = stablehlo.constant dense<0.0> : tensor<f32>
    %v791 = stablehlo.exponential %v789 : tensor<32x197x197xf32>
    %v792 = stablehlo.reduce(%v791 init: %v790) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v793 = stablehlo.broadcast_in_dim %v792, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v794 = stablehlo.divide %v791, %v793 : tensor<32x197x197xf32>
    %v795 = stablehlo.reshape %v794 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v796 = stablehlo.reshape %v795 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v797 = stablehlo.reshape %v779 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v798 = stablehlo.dot_general %v796, %v797, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v799 = stablehlo.reshape %v798 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v800 = stablehlo.reshape %v799 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v801 = stablehlo.constant dense<0.0> : tensor<f32>
    %v802 = stablehlo.pad %v800, %v801, low = [0, 0, 192], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v803 = stablehlo.reshape %v802 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v804 = stablehlo.add %v770, %v803 : tensor<32x75648xf32>
    %v805 = stablehlo.reshape %v659 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v806 = stablehlo.slice %v805 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v807 = stablehlo.reshape %v806 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v808 = stablehlo.reshape %v664 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v809 = stablehlo.slice %v808 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v810 = stablehlo.reshape %v809 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v811 = stablehlo.reshape %v669 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v812 = stablehlo.slice %v811 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v813 = stablehlo.reshape %v812 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v814 = stablehlo.reshape %v810 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v815 = stablehlo.transpose %v814, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v816 = stablehlo.reshape %v815 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v817 = stablehlo.reshape %v807 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v818 = stablehlo.reshape %v816 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v819 = stablehlo.dot_general %v817, %v818, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v820 = stablehlo.reshape %v819 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v821 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v822 = stablehlo.multiply %v820, %v821 : tensor<32x38809xf32>
    %v823 = stablehlo.reshape %v822 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v824 = stablehlo.constant dense<0.0> : tensor<f32>
    %v825 = stablehlo.exponential %v823 : tensor<32x197x197xf32>
    %v826 = stablehlo.reduce(%v825 init: %v824) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v827 = stablehlo.broadcast_in_dim %v826, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v828 = stablehlo.divide %v825, %v827 : tensor<32x197x197xf32>
    %v829 = stablehlo.reshape %v828 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v830 = stablehlo.reshape %v829 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v831 = stablehlo.reshape %v813 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v832 = stablehlo.dot_general %v830, %v831, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v833 = stablehlo.reshape %v832 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v834 = stablehlo.reshape %v833 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v835 = stablehlo.constant dense<0.0> : tensor<f32>
    %v836 = stablehlo.pad %v834, %v835, low = [0, 0, 256], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v837 = stablehlo.reshape %v836 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v838 = stablehlo.add %v804, %v837 : tensor<32x75648xf32>
    %v839 = stablehlo.reshape %v659 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v840 = stablehlo.slice %v839 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v841 = stablehlo.reshape %v840 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v842 = stablehlo.reshape %v664 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v843 = stablehlo.slice %v842 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v844 = stablehlo.reshape %v843 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v845 = stablehlo.reshape %v669 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v846 = stablehlo.slice %v845 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v847 = stablehlo.reshape %v846 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v848 = stablehlo.reshape %v844 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v849 = stablehlo.transpose %v848, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v850 = stablehlo.reshape %v849 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v851 = stablehlo.reshape %v841 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v852 = stablehlo.reshape %v850 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v853 = stablehlo.dot_general %v851, %v852, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v854 = stablehlo.reshape %v853 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v855 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v856 = stablehlo.multiply %v854, %v855 : tensor<32x38809xf32>
    %v857 = stablehlo.reshape %v856 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v858 = stablehlo.constant dense<0.0> : tensor<f32>
    %v859 = stablehlo.exponential %v857 : tensor<32x197x197xf32>
    %v860 = stablehlo.reduce(%v859 init: %v858) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v861 = stablehlo.broadcast_in_dim %v860, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v862 = stablehlo.divide %v859, %v861 : tensor<32x197x197xf32>
    %v863 = stablehlo.reshape %v862 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v864 = stablehlo.reshape %v863 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v865 = stablehlo.reshape %v847 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v866 = stablehlo.dot_general %v864, %v865, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v867 = stablehlo.reshape %v866 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v868 = stablehlo.reshape %v867 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v869 = stablehlo.constant dense<0.0> : tensor<f32>
    %v870 = stablehlo.pad %v868, %v869, low = [0, 0, 320], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v871 = stablehlo.reshape %v870 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v872 = stablehlo.add %v838, %v871 : tensor<32x75648xf32>
    %v873 = stablehlo.reshape %v872 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v874 = stablehlo.dot_general %v873, %b2_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v875 = stablehlo.broadcast_in_dim %b2_bo, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v876 = stablehlo.add %v874, %v875 : tensor<32x197x384xf32>
    %v877 = stablehlo.reshape %v876 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v878 = stablehlo.broadcast_in_dim %dp4, dims = [0] : (tensor<32xf32>) -> tensor<32x75648xf32>
    %v879 = stablehlo.multiply %v878, %v877 : tensor<32x75648xf32>
    %v880 = stablehlo.add %v626, %v879 : tensor<32x75648xf32>
    %v881 = stablehlo.reshape %v880 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v882 = stablehlo.constant dense<0.0> : tensor<f32>
    %v883 = stablehlo.constant dense<384.0> : tensor<32x197x384xf32>
    %v884 = stablehlo.constant dense<1.0e-5> : tensor<32x197x384xf32>
    %v885 = stablehlo.reduce(%v881 init: %v882) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v886 = stablehlo.broadcast_in_dim %v885, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v887 = stablehlo.divide %v886, %v883 : tensor<32x197x384xf32>
    %v888 = stablehlo.subtract %v881, %v887 : tensor<32x197x384xf32>
    %v889 = stablehlo.multiply %v888, %v888 : tensor<32x197x384xf32>
    %v890 = stablehlo.reduce(%v889 init: %v882) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v891 = stablehlo.broadcast_in_dim %v890, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v892 = stablehlo.divide %v891, %v883 : tensor<32x197x384xf32>
    %v893 = stablehlo.add %v892, %v884 : tensor<32x197x384xf32>
    %v894 = stablehlo.rsqrt %v893 : tensor<32x197x384xf32>
    %v895 = stablehlo.multiply %v888, %v894 : tensor<32x197x384xf32>
    %v896 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v897 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v898 = stablehlo.multiply %v895, %v896 : tensor<32x197x384xf32>
    %v899 = stablehlo.add %v898, %v897 : tensor<32x197x384xf32>
    %v900 = stablehlo.reshape %v899 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v901 = stablehlo.reshape %v900 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v902 = stablehlo.broadcast_in_dim %b2_g2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v903 = stablehlo.multiply %v901, %v902 : tensor<32x197x384xf32>
    %v904 = stablehlo.reshape %v903 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v905 = stablehlo.reshape %v904 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v906 = stablehlo.broadcast_in_dim %b2_bt2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v907 = stablehlo.add %v905, %v906 : tensor<32x197x384xf32>
    %v908 = stablehlo.reshape %v907 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v909 = stablehlo.reshape %v908 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v910 = stablehlo.dot_general %v909, %b2_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x1536xf32>) -> tensor<32x197x1536xf32>
    %v911 = stablehlo.broadcast_in_dim %b2_bfc1, dims = [2] : (tensor<1536xf32>) -> tensor<32x197x1536xf32>
    %v912 = stablehlo.add %v910, %v911 : tensor<32x197x1536xf32>
    %v913 = stablehlo.reshape %v912 : (tensor<32x197x1536xf32>) -> tensor<32x302592xf32>
    %v914 = stablehlo.multiply %v913, %v913 : tensor<32x302592xf32>
    %v915 = stablehlo.multiply %v914, %v913 : tensor<32x302592xf32>
    %v916 = stablehlo.constant dense<0.044715> : tensor<32x302592xf32>
    %v917 = stablehlo.multiply %v916, %v915 : tensor<32x302592xf32>
    %v918 = stablehlo.add %v913, %v917 : tensor<32x302592xf32>
    %v919 = stablehlo.constant dense<0.7978845608028654> : tensor<32x302592xf32>
    %v920 = stablehlo.multiply %v919, %v918 : tensor<32x302592xf32>
    %v921 = stablehlo.tanh %v920 : tensor<32x302592xf32>
    %v922 = stablehlo.constant dense<1.0> : tensor<32x302592xf32>
    %v923 = stablehlo.add %v922, %v921 : tensor<32x302592xf32>
    %v924 = stablehlo.constant dense<0.5> : tensor<32x302592xf32>
    %v925 = stablehlo.multiply %v924, %v913 : tensor<32x302592xf32>
    %v926 = stablehlo.multiply %v925, %v923 : tensor<32x302592xf32>
    %v927 = stablehlo.reshape %v926 : (tensor<32x302592xf32>) -> tensor<32x197x1536xf32>
    %v928 = stablehlo.dot_general %v927, %b2_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x1536xf32>, tensor<1536x384xf32>) -> tensor<32x197x384xf32>
    %v929 = stablehlo.broadcast_in_dim %b2_bfc2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v930 = stablehlo.add %v928, %v929 : tensor<32x197x384xf32>
    %v931 = stablehlo.reshape %v930 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v932 = stablehlo.broadcast_in_dim %dp5, dims = [0] : (tensor<32xf32>) -> tensor<32x75648xf32>
    %v933 = stablehlo.multiply %v932, %v931 : tensor<32x75648xf32>
    %v934 = stablehlo.add %v880, %v933 : tensor<32x75648xf32>
    %v935 = stablehlo.reshape %v934 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v936 = stablehlo.constant dense<0.0> : tensor<f32>
    %v937 = stablehlo.constant dense<384.0> : tensor<32x197x384xf32>
    %v938 = stablehlo.constant dense<1.0e-5> : tensor<32x197x384xf32>
    %v939 = stablehlo.reduce(%v935 init: %v936) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v940 = stablehlo.broadcast_in_dim %v939, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v941 = stablehlo.divide %v940, %v937 : tensor<32x197x384xf32>
    %v942 = stablehlo.subtract %v935, %v941 : tensor<32x197x384xf32>
    %v943 = stablehlo.multiply %v942, %v942 : tensor<32x197x384xf32>
    %v944 = stablehlo.reduce(%v943 init: %v936) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v945 = stablehlo.broadcast_in_dim %v944, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v946 = stablehlo.divide %v945, %v937 : tensor<32x197x384xf32>
    %v947 = stablehlo.add %v946, %v938 : tensor<32x197x384xf32>
    %v948 = stablehlo.rsqrt %v947 : tensor<32x197x384xf32>
    %v949 = stablehlo.multiply %v942, %v948 : tensor<32x197x384xf32>
    %v950 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v951 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v952 = stablehlo.multiply %v949, %v950 : tensor<32x197x384xf32>
    %v953 = stablehlo.add %v952, %v951 : tensor<32x197x384xf32>
    %v954 = stablehlo.reshape %v953 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v955 = stablehlo.reshape %v954 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v956 = stablehlo.broadcast_in_dim %b3_g1, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v957 = stablehlo.multiply %v955, %v956 : tensor<32x197x384xf32>
    %v958 = stablehlo.reshape %v957 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v959 = stablehlo.reshape %v958 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v960 = stablehlo.broadcast_in_dim %b3_bt1, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v961 = stablehlo.add %v959, %v960 : tensor<32x197x384xf32>
    %v962 = stablehlo.reshape %v961 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v963 = stablehlo.reshape %v962 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v964 = stablehlo.dot_general %v963, %b3_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v965 = stablehlo.broadcast_in_dim %b3_bq, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v966 = stablehlo.add %v964, %v965 : tensor<32x197x384xf32>
    %v967 = stablehlo.reshape %v966 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v968 = stablehlo.reshape %v962 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v969 = stablehlo.dot_general %v968, %b3_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v970 = stablehlo.broadcast_in_dim %b3_bk, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v971 = stablehlo.add %v969, %v970 : tensor<32x197x384xf32>
    %v972 = stablehlo.reshape %v971 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v973 = stablehlo.reshape %v962 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v974 = stablehlo.dot_general %v973, %b3_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v975 = stablehlo.broadcast_in_dim %b3_bv, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v976 = stablehlo.add %v974, %v975 : tensor<32x197x384xf32>
    %v977 = stablehlo.reshape %v976 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v978 = stablehlo.reshape %v967 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v979 = stablehlo.slice %v978 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v980 = stablehlo.reshape %v979 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v981 = stablehlo.reshape %v972 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v982 = stablehlo.slice %v981 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v983 = stablehlo.reshape %v982 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v984 = stablehlo.reshape %v977 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v985 = stablehlo.slice %v984 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v986 = stablehlo.reshape %v985 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v987 = stablehlo.reshape %v983 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v988 = stablehlo.transpose %v987, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v989 = stablehlo.reshape %v988 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v990 = stablehlo.reshape %v980 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v991 = stablehlo.reshape %v989 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v992 = stablehlo.dot_general %v990, %v991, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v993 = stablehlo.reshape %v992 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v994 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v995 = stablehlo.multiply %v993, %v994 : tensor<32x38809xf32>
    %v996 = stablehlo.reshape %v995 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v997 = stablehlo.constant dense<0.0> : tensor<f32>
    %v998 = stablehlo.exponential %v996 : tensor<32x197x197xf32>
    %v999 = stablehlo.reduce(%v998 init: %v997) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1000 = stablehlo.broadcast_in_dim %v999, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1001 = stablehlo.divide %v998, %v1000 : tensor<32x197x197xf32>
    %v1002 = stablehlo.reshape %v1001 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1003 = stablehlo.reshape %v1002 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1004 = stablehlo.reshape %v986 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1005 = stablehlo.dot_general %v1003, %v1004, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1006 = stablehlo.reshape %v1005 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1007 = stablehlo.reshape %v1006 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1008 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1009 = stablehlo.pad %v1007, %v1008, low = [0, 0, 0], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v1010 = stablehlo.reshape %v1009 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1011 = stablehlo.reshape %v967 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1012 = stablehlo.slice %v1011 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1013 = stablehlo.reshape %v1012 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1014 = stablehlo.reshape %v972 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1015 = stablehlo.slice %v1014 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1016 = stablehlo.reshape %v1015 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1017 = stablehlo.reshape %v977 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1018 = stablehlo.slice %v1017 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1019 = stablehlo.reshape %v1018 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1020 = stablehlo.reshape %v1016 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1021 = stablehlo.transpose %v1020, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1022 = stablehlo.reshape %v1021 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1023 = stablehlo.reshape %v1013 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1024 = stablehlo.reshape %v1022 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1025 = stablehlo.dot_general %v1023, %v1024, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1026 = stablehlo.reshape %v1025 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1027 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1028 = stablehlo.multiply %v1026, %v1027 : tensor<32x38809xf32>
    %v1029 = stablehlo.reshape %v1028 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1030 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1031 = stablehlo.exponential %v1029 : tensor<32x197x197xf32>
    %v1032 = stablehlo.reduce(%v1031 init: %v1030) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1033 = stablehlo.broadcast_in_dim %v1032, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1034 = stablehlo.divide %v1031, %v1033 : tensor<32x197x197xf32>
    %v1035 = stablehlo.reshape %v1034 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1036 = stablehlo.reshape %v1035 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1037 = stablehlo.reshape %v1019 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1038 = stablehlo.dot_general %v1036, %v1037, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1039 = stablehlo.reshape %v1038 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1040 = stablehlo.reshape %v1039 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1041 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1042 = stablehlo.pad %v1040, %v1041, low = [0, 0, 64], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v1043 = stablehlo.reshape %v1042 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1044 = stablehlo.add %v1010, %v1043 : tensor<32x75648xf32>
    %v1045 = stablehlo.reshape %v967 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1046 = stablehlo.slice %v1045 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1047 = stablehlo.reshape %v1046 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1048 = stablehlo.reshape %v972 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1049 = stablehlo.slice %v1048 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1050 = stablehlo.reshape %v1049 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1051 = stablehlo.reshape %v977 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1052 = stablehlo.slice %v1051 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1053 = stablehlo.reshape %v1052 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1054 = stablehlo.reshape %v1050 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1055 = stablehlo.transpose %v1054, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1056 = stablehlo.reshape %v1055 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1057 = stablehlo.reshape %v1047 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1058 = stablehlo.reshape %v1056 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1059 = stablehlo.dot_general %v1057, %v1058, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1060 = stablehlo.reshape %v1059 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1061 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1062 = stablehlo.multiply %v1060, %v1061 : tensor<32x38809xf32>
    %v1063 = stablehlo.reshape %v1062 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1064 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1065 = stablehlo.exponential %v1063 : tensor<32x197x197xf32>
    %v1066 = stablehlo.reduce(%v1065 init: %v1064) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1067 = stablehlo.broadcast_in_dim %v1066, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1068 = stablehlo.divide %v1065, %v1067 : tensor<32x197x197xf32>
    %v1069 = stablehlo.reshape %v1068 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1070 = stablehlo.reshape %v1069 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1071 = stablehlo.reshape %v1053 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1072 = stablehlo.dot_general %v1070, %v1071, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1073 = stablehlo.reshape %v1072 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1074 = stablehlo.reshape %v1073 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1075 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1076 = stablehlo.pad %v1074, %v1075, low = [0, 0, 128], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v1077 = stablehlo.reshape %v1076 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1078 = stablehlo.add %v1044, %v1077 : tensor<32x75648xf32>
    %v1079 = stablehlo.reshape %v967 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1080 = stablehlo.slice %v1079 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1081 = stablehlo.reshape %v1080 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1082 = stablehlo.reshape %v972 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1083 = stablehlo.slice %v1082 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1084 = stablehlo.reshape %v1083 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1085 = stablehlo.reshape %v977 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1086 = stablehlo.slice %v1085 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1087 = stablehlo.reshape %v1086 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1088 = stablehlo.reshape %v1084 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1089 = stablehlo.transpose %v1088, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1090 = stablehlo.reshape %v1089 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1091 = stablehlo.reshape %v1081 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1092 = stablehlo.reshape %v1090 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1093 = stablehlo.dot_general %v1091, %v1092, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1094 = stablehlo.reshape %v1093 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1095 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1096 = stablehlo.multiply %v1094, %v1095 : tensor<32x38809xf32>
    %v1097 = stablehlo.reshape %v1096 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1098 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1099 = stablehlo.exponential %v1097 : tensor<32x197x197xf32>
    %v1100 = stablehlo.reduce(%v1099 init: %v1098) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1101 = stablehlo.broadcast_in_dim %v1100, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1102 = stablehlo.divide %v1099, %v1101 : tensor<32x197x197xf32>
    %v1103 = stablehlo.reshape %v1102 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1104 = stablehlo.reshape %v1103 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1105 = stablehlo.reshape %v1087 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1106 = stablehlo.dot_general %v1104, %v1105, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1107 = stablehlo.reshape %v1106 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1108 = stablehlo.reshape %v1107 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1109 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1110 = stablehlo.pad %v1108, %v1109, low = [0, 0, 192], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v1111 = stablehlo.reshape %v1110 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1112 = stablehlo.add %v1078, %v1111 : tensor<32x75648xf32>
    %v1113 = stablehlo.reshape %v967 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1114 = stablehlo.slice %v1113 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1115 = stablehlo.reshape %v1114 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1116 = stablehlo.reshape %v972 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1117 = stablehlo.slice %v1116 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1118 = stablehlo.reshape %v1117 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1119 = stablehlo.reshape %v977 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1120 = stablehlo.slice %v1119 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1121 = stablehlo.reshape %v1120 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1122 = stablehlo.reshape %v1118 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1123 = stablehlo.transpose %v1122, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1124 = stablehlo.reshape %v1123 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1125 = stablehlo.reshape %v1115 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1126 = stablehlo.reshape %v1124 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1127 = stablehlo.dot_general %v1125, %v1126, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1128 = stablehlo.reshape %v1127 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1129 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1130 = stablehlo.multiply %v1128, %v1129 : tensor<32x38809xf32>
    %v1131 = stablehlo.reshape %v1130 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1132 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1133 = stablehlo.exponential %v1131 : tensor<32x197x197xf32>
    %v1134 = stablehlo.reduce(%v1133 init: %v1132) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1135 = stablehlo.broadcast_in_dim %v1134, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1136 = stablehlo.divide %v1133, %v1135 : tensor<32x197x197xf32>
    %v1137 = stablehlo.reshape %v1136 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1138 = stablehlo.reshape %v1137 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1139 = stablehlo.reshape %v1121 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1140 = stablehlo.dot_general %v1138, %v1139, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1141 = stablehlo.reshape %v1140 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1142 = stablehlo.reshape %v1141 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1143 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1144 = stablehlo.pad %v1142, %v1143, low = [0, 0, 256], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v1145 = stablehlo.reshape %v1144 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1146 = stablehlo.add %v1112, %v1145 : tensor<32x75648xf32>
    %v1147 = stablehlo.reshape %v967 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1148 = stablehlo.slice %v1147 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1149 = stablehlo.reshape %v1148 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1150 = stablehlo.reshape %v972 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1151 = stablehlo.slice %v1150 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1152 = stablehlo.reshape %v1151 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1153 = stablehlo.reshape %v977 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1154 = stablehlo.slice %v1153 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1155 = stablehlo.reshape %v1154 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1156 = stablehlo.reshape %v1152 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1157 = stablehlo.transpose %v1156, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1158 = stablehlo.reshape %v1157 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1159 = stablehlo.reshape %v1149 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1160 = stablehlo.reshape %v1158 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1161 = stablehlo.dot_general %v1159, %v1160, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1162 = stablehlo.reshape %v1161 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1163 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1164 = stablehlo.multiply %v1162, %v1163 : tensor<32x38809xf32>
    %v1165 = stablehlo.reshape %v1164 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1166 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1167 = stablehlo.exponential %v1165 : tensor<32x197x197xf32>
    %v1168 = stablehlo.reduce(%v1167 init: %v1166) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1169 = stablehlo.broadcast_in_dim %v1168, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1170 = stablehlo.divide %v1167, %v1169 : tensor<32x197x197xf32>
    %v1171 = stablehlo.reshape %v1170 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1172 = stablehlo.reshape %v1171 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1173 = stablehlo.reshape %v1155 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1174 = stablehlo.dot_general %v1172, %v1173, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1175 = stablehlo.reshape %v1174 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1176 = stablehlo.reshape %v1175 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1177 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1178 = stablehlo.pad %v1176, %v1177, low = [0, 0, 320], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v1179 = stablehlo.reshape %v1178 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1180 = stablehlo.add %v1146, %v1179 : tensor<32x75648xf32>
    %v1181 = stablehlo.reshape %v1180 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1182 = stablehlo.dot_general %v1181, %b3_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v1183 = stablehlo.broadcast_in_dim %b3_bo, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1184 = stablehlo.add %v1182, %v1183 : tensor<32x197x384xf32>
    %v1185 = stablehlo.reshape %v1184 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1186 = stablehlo.broadcast_in_dim %dp6, dims = [0] : (tensor<32xf32>) -> tensor<32x75648xf32>
    %v1187 = stablehlo.multiply %v1186, %v1185 : tensor<32x75648xf32>
    %v1188 = stablehlo.add %v934, %v1187 : tensor<32x75648xf32>
    %v1189 = stablehlo.reshape %v1188 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1190 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1191 = stablehlo.constant dense<384.0> : tensor<32x197x384xf32>
    %v1192 = stablehlo.constant dense<1.0e-5> : tensor<32x197x384xf32>
    %v1193 = stablehlo.reduce(%v1189 init: %v1190) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1194 = stablehlo.broadcast_in_dim %v1193, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v1195 = stablehlo.divide %v1194, %v1191 : tensor<32x197x384xf32>
    %v1196 = stablehlo.subtract %v1189, %v1195 : tensor<32x197x384xf32>
    %v1197 = stablehlo.multiply %v1196, %v1196 : tensor<32x197x384xf32>
    %v1198 = stablehlo.reduce(%v1197 init: %v1190) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1199 = stablehlo.broadcast_in_dim %v1198, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v1200 = stablehlo.divide %v1199, %v1191 : tensor<32x197x384xf32>
    %v1201 = stablehlo.add %v1200, %v1192 : tensor<32x197x384xf32>
    %v1202 = stablehlo.rsqrt %v1201 : tensor<32x197x384xf32>
    %v1203 = stablehlo.multiply %v1196, %v1202 : tensor<32x197x384xf32>
    %v1204 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v1205 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v1206 = stablehlo.multiply %v1203, %v1204 : tensor<32x197x384xf32>
    %v1207 = stablehlo.add %v1206, %v1205 : tensor<32x197x384xf32>
    %v1208 = stablehlo.reshape %v1207 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1209 = stablehlo.reshape %v1208 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1210 = stablehlo.broadcast_in_dim %b3_g2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1211 = stablehlo.multiply %v1209, %v1210 : tensor<32x197x384xf32>
    %v1212 = stablehlo.reshape %v1211 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1213 = stablehlo.reshape %v1212 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1214 = stablehlo.broadcast_in_dim %b3_bt2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1215 = stablehlo.add %v1213, %v1214 : tensor<32x197x384xf32>
    %v1216 = stablehlo.reshape %v1215 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1217 = stablehlo.reshape %v1216 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1218 = stablehlo.dot_general %v1217, %b3_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x1536xf32>) -> tensor<32x197x1536xf32>
    %v1219 = stablehlo.broadcast_in_dim %b3_bfc1, dims = [2] : (tensor<1536xf32>) -> tensor<32x197x1536xf32>
    %v1220 = stablehlo.add %v1218, %v1219 : tensor<32x197x1536xf32>
    %v1221 = stablehlo.reshape %v1220 : (tensor<32x197x1536xf32>) -> tensor<32x302592xf32>
    %v1222 = stablehlo.multiply %v1221, %v1221 : tensor<32x302592xf32>
    %v1223 = stablehlo.multiply %v1222, %v1221 : tensor<32x302592xf32>
    %v1224 = stablehlo.constant dense<0.044715> : tensor<32x302592xf32>
    %v1225 = stablehlo.multiply %v1224, %v1223 : tensor<32x302592xf32>
    %v1226 = stablehlo.add %v1221, %v1225 : tensor<32x302592xf32>
    %v1227 = stablehlo.constant dense<0.7978845608028654> : tensor<32x302592xf32>
    %v1228 = stablehlo.multiply %v1227, %v1226 : tensor<32x302592xf32>
    %v1229 = stablehlo.tanh %v1228 : tensor<32x302592xf32>
    %v1230 = stablehlo.constant dense<1.0> : tensor<32x302592xf32>
    %v1231 = stablehlo.add %v1230, %v1229 : tensor<32x302592xf32>
    %v1232 = stablehlo.constant dense<0.5> : tensor<32x302592xf32>
    %v1233 = stablehlo.multiply %v1232, %v1221 : tensor<32x302592xf32>
    %v1234 = stablehlo.multiply %v1233, %v1231 : tensor<32x302592xf32>
    %v1235 = stablehlo.reshape %v1234 : (tensor<32x302592xf32>) -> tensor<32x197x1536xf32>
    %v1236 = stablehlo.dot_general %v1235, %b3_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x1536xf32>, tensor<1536x384xf32>) -> tensor<32x197x384xf32>
    %v1237 = stablehlo.broadcast_in_dim %b3_bfc2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1238 = stablehlo.add %v1236, %v1237 : tensor<32x197x384xf32>
    %v1239 = stablehlo.reshape %v1238 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1240 = stablehlo.broadcast_in_dim %dp7, dims = [0] : (tensor<32xf32>) -> tensor<32x75648xf32>
    %v1241 = stablehlo.multiply %v1240, %v1239 : tensor<32x75648xf32>
    %v1242 = stablehlo.add %v1188, %v1241 : tensor<32x75648xf32>
    %v1243 = stablehlo.reshape %v1242 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1244 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1245 = stablehlo.constant dense<384.0> : tensor<32x197x384xf32>
    %v1246 = stablehlo.constant dense<1.0e-5> : tensor<32x197x384xf32>
    %v1247 = stablehlo.reduce(%v1243 init: %v1244) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1248 = stablehlo.broadcast_in_dim %v1247, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v1249 = stablehlo.divide %v1248, %v1245 : tensor<32x197x384xf32>
    %v1250 = stablehlo.subtract %v1243, %v1249 : tensor<32x197x384xf32>
    %v1251 = stablehlo.multiply %v1250, %v1250 : tensor<32x197x384xf32>
    %v1252 = stablehlo.reduce(%v1251 init: %v1244) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1253 = stablehlo.broadcast_in_dim %v1252, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v1254 = stablehlo.divide %v1253, %v1245 : tensor<32x197x384xf32>
    %v1255 = stablehlo.add %v1254, %v1246 : tensor<32x197x384xf32>
    %v1256 = stablehlo.rsqrt %v1255 : tensor<32x197x384xf32>
    %v1257 = stablehlo.multiply %v1250, %v1256 : tensor<32x197x384xf32>
    %v1258 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v1259 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v1260 = stablehlo.multiply %v1257, %v1258 : tensor<32x197x384xf32>
    %v1261 = stablehlo.add %v1260, %v1259 : tensor<32x197x384xf32>
    %v1262 = stablehlo.reshape %v1261 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1263 = stablehlo.reshape %v1262 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1264 = stablehlo.broadcast_in_dim %b4_g1, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1265 = stablehlo.multiply %v1263, %v1264 : tensor<32x197x384xf32>
    %v1266 = stablehlo.reshape %v1265 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1267 = stablehlo.reshape %v1266 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1268 = stablehlo.broadcast_in_dim %b4_bt1, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1269 = stablehlo.add %v1267, %v1268 : tensor<32x197x384xf32>
    %v1270 = stablehlo.reshape %v1269 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1271 = stablehlo.reshape %v1270 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1272 = stablehlo.dot_general %v1271, %b4_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v1273 = stablehlo.broadcast_in_dim %b4_bq, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1274 = stablehlo.add %v1272, %v1273 : tensor<32x197x384xf32>
    %v1275 = stablehlo.reshape %v1274 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1276 = stablehlo.reshape %v1270 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1277 = stablehlo.dot_general %v1276, %b4_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v1278 = stablehlo.broadcast_in_dim %b4_bk, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1279 = stablehlo.add %v1277, %v1278 : tensor<32x197x384xf32>
    %v1280 = stablehlo.reshape %v1279 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1281 = stablehlo.reshape %v1270 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1282 = stablehlo.dot_general %v1281, %b4_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v1283 = stablehlo.broadcast_in_dim %b4_bv, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1284 = stablehlo.add %v1282, %v1283 : tensor<32x197x384xf32>
    %v1285 = stablehlo.reshape %v1284 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1286 = stablehlo.reshape %v1275 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1287 = stablehlo.slice %v1286 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1288 = stablehlo.reshape %v1287 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1289 = stablehlo.reshape %v1280 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1290 = stablehlo.slice %v1289 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1291 = stablehlo.reshape %v1290 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1292 = stablehlo.reshape %v1285 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1293 = stablehlo.slice %v1292 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1294 = stablehlo.reshape %v1293 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1295 = stablehlo.reshape %v1291 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1296 = stablehlo.transpose %v1295, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1297 = stablehlo.reshape %v1296 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1298 = stablehlo.reshape %v1288 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1299 = stablehlo.reshape %v1297 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1300 = stablehlo.dot_general %v1298, %v1299, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1301 = stablehlo.reshape %v1300 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1302 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1303 = stablehlo.multiply %v1301, %v1302 : tensor<32x38809xf32>
    %v1304 = stablehlo.reshape %v1303 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1305 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1306 = stablehlo.exponential %v1304 : tensor<32x197x197xf32>
    %v1307 = stablehlo.reduce(%v1306 init: %v1305) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1308 = stablehlo.broadcast_in_dim %v1307, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1309 = stablehlo.divide %v1306, %v1308 : tensor<32x197x197xf32>
    %v1310 = stablehlo.reshape %v1309 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1311 = stablehlo.reshape %v1310 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1312 = stablehlo.reshape %v1294 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1313 = stablehlo.dot_general %v1311, %v1312, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1314 = stablehlo.reshape %v1313 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1315 = stablehlo.reshape %v1314 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1316 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1317 = stablehlo.pad %v1315, %v1316, low = [0, 0, 0], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v1318 = stablehlo.reshape %v1317 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1319 = stablehlo.reshape %v1275 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1320 = stablehlo.slice %v1319 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1321 = stablehlo.reshape %v1320 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1322 = stablehlo.reshape %v1280 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1323 = stablehlo.slice %v1322 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1324 = stablehlo.reshape %v1323 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1325 = stablehlo.reshape %v1285 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1326 = stablehlo.slice %v1325 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1327 = stablehlo.reshape %v1326 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1328 = stablehlo.reshape %v1324 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1329 = stablehlo.transpose %v1328, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1330 = stablehlo.reshape %v1329 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1331 = stablehlo.reshape %v1321 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1332 = stablehlo.reshape %v1330 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1333 = stablehlo.dot_general %v1331, %v1332, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1334 = stablehlo.reshape %v1333 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1335 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1336 = stablehlo.multiply %v1334, %v1335 : tensor<32x38809xf32>
    %v1337 = stablehlo.reshape %v1336 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1338 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1339 = stablehlo.exponential %v1337 : tensor<32x197x197xf32>
    %v1340 = stablehlo.reduce(%v1339 init: %v1338) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1341 = stablehlo.broadcast_in_dim %v1340, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1342 = stablehlo.divide %v1339, %v1341 : tensor<32x197x197xf32>
    %v1343 = stablehlo.reshape %v1342 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1344 = stablehlo.reshape %v1343 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1345 = stablehlo.reshape %v1327 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1346 = stablehlo.dot_general %v1344, %v1345, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1347 = stablehlo.reshape %v1346 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1348 = stablehlo.reshape %v1347 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1349 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1350 = stablehlo.pad %v1348, %v1349, low = [0, 0, 64], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v1351 = stablehlo.reshape %v1350 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1352 = stablehlo.add %v1318, %v1351 : tensor<32x75648xf32>
    %v1353 = stablehlo.reshape %v1275 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1354 = stablehlo.slice %v1353 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1355 = stablehlo.reshape %v1354 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1356 = stablehlo.reshape %v1280 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1357 = stablehlo.slice %v1356 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1358 = stablehlo.reshape %v1357 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1359 = stablehlo.reshape %v1285 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1360 = stablehlo.slice %v1359 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1361 = stablehlo.reshape %v1360 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1362 = stablehlo.reshape %v1358 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1363 = stablehlo.transpose %v1362, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1364 = stablehlo.reshape %v1363 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1365 = stablehlo.reshape %v1355 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1366 = stablehlo.reshape %v1364 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1367 = stablehlo.dot_general %v1365, %v1366, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1368 = stablehlo.reshape %v1367 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1369 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1370 = stablehlo.multiply %v1368, %v1369 : tensor<32x38809xf32>
    %v1371 = stablehlo.reshape %v1370 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1372 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1373 = stablehlo.exponential %v1371 : tensor<32x197x197xf32>
    %v1374 = stablehlo.reduce(%v1373 init: %v1372) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1375 = stablehlo.broadcast_in_dim %v1374, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1376 = stablehlo.divide %v1373, %v1375 : tensor<32x197x197xf32>
    %v1377 = stablehlo.reshape %v1376 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1378 = stablehlo.reshape %v1377 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1379 = stablehlo.reshape %v1361 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1380 = stablehlo.dot_general %v1378, %v1379, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1381 = stablehlo.reshape %v1380 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1382 = stablehlo.reshape %v1381 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1383 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1384 = stablehlo.pad %v1382, %v1383, low = [0, 0, 128], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v1385 = stablehlo.reshape %v1384 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1386 = stablehlo.add %v1352, %v1385 : tensor<32x75648xf32>
    %v1387 = stablehlo.reshape %v1275 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1388 = stablehlo.slice %v1387 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1389 = stablehlo.reshape %v1388 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1390 = stablehlo.reshape %v1280 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1391 = stablehlo.slice %v1390 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1392 = stablehlo.reshape %v1391 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1393 = stablehlo.reshape %v1285 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1394 = stablehlo.slice %v1393 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1395 = stablehlo.reshape %v1394 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1396 = stablehlo.reshape %v1392 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1397 = stablehlo.transpose %v1396, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1398 = stablehlo.reshape %v1397 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1399 = stablehlo.reshape %v1389 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1400 = stablehlo.reshape %v1398 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1401 = stablehlo.dot_general %v1399, %v1400, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1402 = stablehlo.reshape %v1401 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1403 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1404 = stablehlo.multiply %v1402, %v1403 : tensor<32x38809xf32>
    %v1405 = stablehlo.reshape %v1404 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1406 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1407 = stablehlo.exponential %v1405 : tensor<32x197x197xf32>
    %v1408 = stablehlo.reduce(%v1407 init: %v1406) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1409 = stablehlo.broadcast_in_dim %v1408, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1410 = stablehlo.divide %v1407, %v1409 : tensor<32x197x197xf32>
    %v1411 = stablehlo.reshape %v1410 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1412 = stablehlo.reshape %v1411 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1413 = stablehlo.reshape %v1395 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1414 = stablehlo.dot_general %v1412, %v1413, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1415 = stablehlo.reshape %v1414 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1416 = stablehlo.reshape %v1415 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1417 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1418 = stablehlo.pad %v1416, %v1417, low = [0, 0, 192], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v1419 = stablehlo.reshape %v1418 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1420 = stablehlo.add %v1386, %v1419 : tensor<32x75648xf32>
    %v1421 = stablehlo.reshape %v1275 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1422 = stablehlo.slice %v1421 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1423 = stablehlo.reshape %v1422 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1424 = stablehlo.reshape %v1280 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1425 = stablehlo.slice %v1424 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1426 = stablehlo.reshape %v1425 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1427 = stablehlo.reshape %v1285 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1428 = stablehlo.slice %v1427 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1429 = stablehlo.reshape %v1428 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1430 = stablehlo.reshape %v1426 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1431 = stablehlo.transpose %v1430, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1432 = stablehlo.reshape %v1431 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1433 = stablehlo.reshape %v1423 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1434 = stablehlo.reshape %v1432 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1435 = stablehlo.dot_general %v1433, %v1434, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1436 = stablehlo.reshape %v1435 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1437 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1438 = stablehlo.multiply %v1436, %v1437 : tensor<32x38809xf32>
    %v1439 = stablehlo.reshape %v1438 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1440 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1441 = stablehlo.exponential %v1439 : tensor<32x197x197xf32>
    %v1442 = stablehlo.reduce(%v1441 init: %v1440) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1443 = stablehlo.broadcast_in_dim %v1442, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1444 = stablehlo.divide %v1441, %v1443 : tensor<32x197x197xf32>
    %v1445 = stablehlo.reshape %v1444 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1446 = stablehlo.reshape %v1445 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1447 = stablehlo.reshape %v1429 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1448 = stablehlo.dot_general %v1446, %v1447, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1449 = stablehlo.reshape %v1448 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1450 = stablehlo.reshape %v1449 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1451 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1452 = stablehlo.pad %v1450, %v1451, low = [0, 0, 256], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v1453 = stablehlo.reshape %v1452 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1454 = stablehlo.add %v1420, %v1453 : tensor<32x75648xf32>
    %v1455 = stablehlo.reshape %v1275 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1456 = stablehlo.slice %v1455 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1457 = stablehlo.reshape %v1456 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1458 = stablehlo.reshape %v1280 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1459 = stablehlo.slice %v1458 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1460 = stablehlo.reshape %v1459 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1461 = stablehlo.reshape %v1285 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1462 = stablehlo.slice %v1461 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1463 = stablehlo.reshape %v1462 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1464 = stablehlo.reshape %v1460 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1465 = stablehlo.transpose %v1464, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1466 = stablehlo.reshape %v1465 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1467 = stablehlo.reshape %v1457 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1468 = stablehlo.reshape %v1466 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1469 = stablehlo.dot_general %v1467, %v1468, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1470 = stablehlo.reshape %v1469 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1471 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1472 = stablehlo.multiply %v1470, %v1471 : tensor<32x38809xf32>
    %v1473 = stablehlo.reshape %v1472 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1474 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1475 = stablehlo.exponential %v1473 : tensor<32x197x197xf32>
    %v1476 = stablehlo.reduce(%v1475 init: %v1474) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1477 = stablehlo.broadcast_in_dim %v1476, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1478 = stablehlo.divide %v1475, %v1477 : tensor<32x197x197xf32>
    %v1479 = stablehlo.reshape %v1478 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1480 = stablehlo.reshape %v1479 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1481 = stablehlo.reshape %v1463 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1482 = stablehlo.dot_general %v1480, %v1481, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1483 = stablehlo.reshape %v1482 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1484 = stablehlo.reshape %v1483 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1485 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1486 = stablehlo.pad %v1484, %v1485, low = [0, 0, 320], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v1487 = stablehlo.reshape %v1486 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1488 = stablehlo.add %v1454, %v1487 : tensor<32x75648xf32>
    %v1489 = stablehlo.reshape %v1488 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1490 = stablehlo.dot_general %v1489, %b4_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v1491 = stablehlo.broadcast_in_dim %b4_bo, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1492 = stablehlo.add %v1490, %v1491 : tensor<32x197x384xf32>
    %v1493 = stablehlo.reshape %v1492 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1494 = stablehlo.broadcast_in_dim %dp8, dims = [0] : (tensor<32xf32>) -> tensor<32x75648xf32>
    %v1495 = stablehlo.multiply %v1494, %v1493 : tensor<32x75648xf32>
    %v1496 = stablehlo.add %v1242, %v1495 : tensor<32x75648xf32>
    %v1497 = stablehlo.reshape %v1496 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1498 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1499 = stablehlo.constant dense<384.0> : tensor<32x197x384xf32>
    %v1500 = stablehlo.constant dense<1.0e-5> : tensor<32x197x384xf32>
    %v1501 = stablehlo.reduce(%v1497 init: %v1498) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1502 = stablehlo.broadcast_in_dim %v1501, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v1503 = stablehlo.divide %v1502, %v1499 : tensor<32x197x384xf32>
    %v1504 = stablehlo.subtract %v1497, %v1503 : tensor<32x197x384xf32>
    %v1505 = stablehlo.multiply %v1504, %v1504 : tensor<32x197x384xf32>
    %v1506 = stablehlo.reduce(%v1505 init: %v1498) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1507 = stablehlo.broadcast_in_dim %v1506, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v1508 = stablehlo.divide %v1507, %v1499 : tensor<32x197x384xf32>
    %v1509 = stablehlo.add %v1508, %v1500 : tensor<32x197x384xf32>
    %v1510 = stablehlo.rsqrt %v1509 : tensor<32x197x384xf32>
    %v1511 = stablehlo.multiply %v1504, %v1510 : tensor<32x197x384xf32>
    %v1512 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v1513 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v1514 = stablehlo.multiply %v1511, %v1512 : tensor<32x197x384xf32>
    %v1515 = stablehlo.add %v1514, %v1513 : tensor<32x197x384xf32>
    %v1516 = stablehlo.reshape %v1515 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1517 = stablehlo.reshape %v1516 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1518 = stablehlo.broadcast_in_dim %b4_g2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1519 = stablehlo.multiply %v1517, %v1518 : tensor<32x197x384xf32>
    %v1520 = stablehlo.reshape %v1519 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1521 = stablehlo.reshape %v1520 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1522 = stablehlo.broadcast_in_dim %b4_bt2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1523 = stablehlo.add %v1521, %v1522 : tensor<32x197x384xf32>
    %v1524 = stablehlo.reshape %v1523 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1525 = stablehlo.reshape %v1524 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1526 = stablehlo.dot_general %v1525, %b4_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x1536xf32>) -> tensor<32x197x1536xf32>
    %v1527 = stablehlo.broadcast_in_dim %b4_bfc1, dims = [2] : (tensor<1536xf32>) -> tensor<32x197x1536xf32>
    %v1528 = stablehlo.add %v1526, %v1527 : tensor<32x197x1536xf32>
    %v1529 = stablehlo.reshape %v1528 : (tensor<32x197x1536xf32>) -> tensor<32x302592xf32>
    %v1530 = stablehlo.multiply %v1529, %v1529 : tensor<32x302592xf32>
    %v1531 = stablehlo.multiply %v1530, %v1529 : tensor<32x302592xf32>
    %v1532 = stablehlo.constant dense<0.044715> : tensor<32x302592xf32>
    %v1533 = stablehlo.multiply %v1532, %v1531 : tensor<32x302592xf32>
    %v1534 = stablehlo.add %v1529, %v1533 : tensor<32x302592xf32>
    %v1535 = stablehlo.constant dense<0.7978845608028654> : tensor<32x302592xf32>
    %v1536 = stablehlo.multiply %v1535, %v1534 : tensor<32x302592xf32>
    %v1537 = stablehlo.tanh %v1536 : tensor<32x302592xf32>
    %v1538 = stablehlo.constant dense<1.0> : tensor<32x302592xf32>
    %v1539 = stablehlo.add %v1538, %v1537 : tensor<32x302592xf32>
    %v1540 = stablehlo.constant dense<0.5> : tensor<32x302592xf32>
    %v1541 = stablehlo.multiply %v1540, %v1529 : tensor<32x302592xf32>
    %v1542 = stablehlo.multiply %v1541, %v1539 : tensor<32x302592xf32>
    %v1543 = stablehlo.reshape %v1542 : (tensor<32x302592xf32>) -> tensor<32x197x1536xf32>
    %v1544 = stablehlo.dot_general %v1543, %b4_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x1536xf32>, tensor<1536x384xf32>) -> tensor<32x197x384xf32>
    %v1545 = stablehlo.broadcast_in_dim %b4_bfc2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1546 = stablehlo.add %v1544, %v1545 : tensor<32x197x384xf32>
    %v1547 = stablehlo.reshape %v1546 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1548 = stablehlo.broadcast_in_dim %dp9, dims = [0] : (tensor<32xf32>) -> tensor<32x75648xf32>
    %v1549 = stablehlo.multiply %v1548, %v1547 : tensor<32x75648xf32>
    %v1550 = stablehlo.add %v1496, %v1549 : tensor<32x75648xf32>
    %v1551 = stablehlo.reshape %v1550 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1552 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1553 = stablehlo.constant dense<384.0> : tensor<32x197x384xf32>
    %v1554 = stablehlo.constant dense<1.0e-5> : tensor<32x197x384xf32>
    %v1555 = stablehlo.reduce(%v1551 init: %v1552) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1556 = stablehlo.broadcast_in_dim %v1555, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v1557 = stablehlo.divide %v1556, %v1553 : tensor<32x197x384xf32>
    %v1558 = stablehlo.subtract %v1551, %v1557 : tensor<32x197x384xf32>
    %v1559 = stablehlo.multiply %v1558, %v1558 : tensor<32x197x384xf32>
    %v1560 = stablehlo.reduce(%v1559 init: %v1552) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1561 = stablehlo.broadcast_in_dim %v1560, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v1562 = stablehlo.divide %v1561, %v1553 : tensor<32x197x384xf32>
    %v1563 = stablehlo.add %v1562, %v1554 : tensor<32x197x384xf32>
    %v1564 = stablehlo.rsqrt %v1563 : tensor<32x197x384xf32>
    %v1565 = stablehlo.multiply %v1558, %v1564 : tensor<32x197x384xf32>
    %v1566 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v1567 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v1568 = stablehlo.multiply %v1565, %v1566 : tensor<32x197x384xf32>
    %v1569 = stablehlo.add %v1568, %v1567 : tensor<32x197x384xf32>
    %v1570 = stablehlo.reshape %v1569 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1571 = stablehlo.reshape %v1570 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1572 = stablehlo.broadcast_in_dim %b5_g1, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1573 = stablehlo.multiply %v1571, %v1572 : tensor<32x197x384xf32>
    %v1574 = stablehlo.reshape %v1573 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1575 = stablehlo.reshape %v1574 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1576 = stablehlo.broadcast_in_dim %b5_bt1, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1577 = stablehlo.add %v1575, %v1576 : tensor<32x197x384xf32>
    %v1578 = stablehlo.reshape %v1577 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1579 = stablehlo.reshape %v1578 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1580 = stablehlo.dot_general %v1579, %b5_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v1581 = stablehlo.broadcast_in_dim %b5_bq, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1582 = stablehlo.add %v1580, %v1581 : tensor<32x197x384xf32>
    %v1583 = stablehlo.reshape %v1582 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1584 = stablehlo.reshape %v1578 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1585 = stablehlo.dot_general %v1584, %b5_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v1586 = stablehlo.broadcast_in_dim %b5_bk, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1587 = stablehlo.add %v1585, %v1586 : tensor<32x197x384xf32>
    %v1588 = stablehlo.reshape %v1587 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1589 = stablehlo.reshape %v1578 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1590 = stablehlo.dot_general %v1589, %b5_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v1591 = stablehlo.broadcast_in_dim %b5_bv, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1592 = stablehlo.add %v1590, %v1591 : tensor<32x197x384xf32>
    %v1593 = stablehlo.reshape %v1592 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1594 = stablehlo.reshape %v1583 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1595 = stablehlo.slice %v1594 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1596 = stablehlo.reshape %v1595 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1597 = stablehlo.reshape %v1588 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1598 = stablehlo.slice %v1597 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1599 = stablehlo.reshape %v1598 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1600 = stablehlo.reshape %v1593 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1601 = stablehlo.slice %v1600 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1602 = stablehlo.reshape %v1601 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1603 = stablehlo.reshape %v1599 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1604 = stablehlo.transpose %v1603, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1605 = stablehlo.reshape %v1604 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1606 = stablehlo.reshape %v1596 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1607 = stablehlo.reshape %v1605 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1608 = stablehlo.dot_general %v1606, %v1607, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1609 = stablehlo.reshape %v1608 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1610 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1611 = stablehlo.multiply %v1609, %v1610 : tensor<32x38809xf32>
    %v1612 = stablehlo.reshape %v1611 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1613 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1614 = stablehlo.exponential %v1612 : tensor<32x197x197xf32>
    %v1615 = stablehlo.reduce(%v1614 init: %v1613) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1616 = stablehlo.broadcast_in_dim %v1615, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1617 = stablehlo.divide %v1614, %v1616 : tensor<32x197x197xf32>
    %v1618 = stablehlo.reshape %v1617 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1619 = stablehlo.reshape %v1618 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1620 = stablehlo.reshape %v1602 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1621 = stablehlo.dot_general %v1619, %v1620, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1622 = stablehlo.reshape %v1621 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1623 = stablehlo.reshape %v1622 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1624 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1625 = stablehlo.pad %v1623, %v1624, low = [0, 0, 0], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v1626 = stablehlo.reshape %v1625 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1627 = stablehlo.reshape %v1583 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1628 = stablehlo.slice %v1627 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1629 = stablehlo.reshape %v1628 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1630 = stablehlo.reshape %v1588 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1631 = stablehlo.slice %v1630 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1632 = stablehlo.reshape %v1631 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1633 = stablehlo.reshape %v1593 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1634 = stablehlo.slice %v1633 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
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
    %v1658 = stablehlo.pad %v1656, %v1657, low = [0, 0, 64], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v1659 = stablehlo.reshape %v1658 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1660 = stablehlo.add %v1626, %v1659 : tensor<32x75648xf32>
    %v1661 = stablehlo.reshape %v1583 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1662 = stablehlo.slice %v1661 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1663 = stablehlo.reshape %v1662 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1664 = stablehlo.reshape %v1588 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1665 = stablehlo.slice %v1664 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1666 = stablehlo.reshape %v1665 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1667 = stablehlo.reshape %v1593 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1668 = stablehlo.slice %v1667 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
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
    %v1692 = stablehlo.pad %v1690, %v1691, low = [0, 0, 128], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v1693 = stablehlo.reshape %v1692 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1694 = stablehlo.add %v1660, %v1693 : tensor<32x75648xf32>
    %v1695 = stablehlo.reshape %v1583 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1696 = stablehlo.slice %v1695 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1697 = stablehlo.reshape %v1696 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1698 = stablehlo.reshape %v1588 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1699 = stablehlo.slice %v1698 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1700 = stablehlo.reshape %v1699 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1701 = stablehlo.reshape %v1593 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1702 = stablehlo.slice %v1701 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
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
    %v1726 = stablehlo.pad %v1724, %v1725, low = [0, 0, 192], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v1727 = stablehlo.reshape %v1726 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1728 = stablehlo.add %v1694, %v1727 : tensor<32x75648xf32>
    %v1729 = stablehlo.reshape %v1583 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1730 = stablehlo.slice %v1729 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1731 = stablehlo.reshape %v1730 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1732 = stablehlo.reshape %v1588 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1733 = stablehlo.slice %v1732 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1734 = stablehlo.reshape %v1733 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1735 = stablehlo.reshape %v1593 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1736 = stablehlo.slice %v1735 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
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
    %v1760 = stablehlo.pad %v1758, %v1759, low = [0, 0, 256], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v1761 = stablehlo.reshape %v1760 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1762 = stablehlo.add %v1728, %v1761 : tensor<32x75648xf32>
    %v1763 = stablehlo.reshape %v1583 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1764 = stablehlo.slice %v1763 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1765 = stablehlo.reshape %v1764 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1766 = stablehlo.reshape %v1588 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1767 = stablehlo.slice %v1766 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1768 = stablehlo.reshape %v1767 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1769 = stablehlo.reshape %v1593 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1770 = stablehlo.slice %v1769 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
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
    %v1794 = stablehlo.pad %v1792, %v1793, low = [0, 0, 320], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v1795 = stablehlo.reshape %v1794 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1796 = stablehlo.add %v1762, %v1795 : tensor<32x75648xf32>
    %v1797 = stablehlo.reshape %v1796 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1798 = stablehlo.dot_general %v1797, %b5_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v1799 = stablehlo.broadcast_in_dim %b5_bo, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1800 = stablehlo.add %v1798, %v1799 : tensor<32x197x384xf32>
    %v1801 = stablehlo.reshape %v1800 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1802 = stablehlo.broadcast_in_dim %dp10, dims = [0] : (tensor<32xf32>) -> tensor<32x75648xf32>
    %v1803 = stablehlo.multiply %v1802, %v1801 : tensor<32x75648xf32>
    %v1804 = stablehlo.add %v1550, %v1803 : tensor<32x75648xf32>
    %v1805 = stablehlo.reshape %v1804 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1806 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1807 = stablehlo.constant dense<384.0> : tensor<32x197x384xf32>
    %v1808 = stablehlo.constant dense<1.0e-5> : tensor<32x197x384xf32>
    %v1809 = stablehlo.reduce(%v1805 init: %v1806) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1810 = stablehlo.broadcast_in_dim %v1809, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v1811 = stablehlo.divide %v1810, %v1807 : tensor<32x197x384xf32>
    %v1812 = stablehlo.subtract %v1805, %v1811 : tensor<32x197x384xf32>
    %v1813 = stablehlo.multiply %v1812, %v1812 : tensor<32x197x384xf32>
    %v1814 = stablehlo.reduce(%v1813 init: %v1806) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1815 = stablehlo.broadcast_in_dim %v1814, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v1816 = stablehlo.divide %v1815, %v1807 : tensor<32x197x384xf32>
    %v1817 = stablehlo.add %v1816, %v1808 : tensor<32x197x384xf32>
    %v1818 = stablehlo.rsqrt %v1817 : tensor<32x197x384xf32>
    %v1819 = stablehlo.multiply %v1812, %v1818 : tensor<32x197x384xf32>
    %v1820 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v1821 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v1822 = stablehlo.multiply %v1819, %v1820 : tensor<32x197x384xf32>
    %v1823 = stablehlo.add %v1822, %v1821 : tensor<32x197x384xf32>
    %v1824 = stablehlo.reshape %v1823 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1825 = stablehlo.reshape %v1824 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1826 = stablehlo.broadcast_in_dim %b5_g2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1827 = stablehlo.multiply %v1825, %v1826 : tensor<32x197x384xf32>
    %v1828 = stablehlo.reshape %v1827 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1829 = stablehlo.reshape %v1828 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1830 = stablehlo.broadcast_in_dim %b5_bt2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1831 = stablehlo.add %v1829, %v1830 : tensor<32x197x384xf32>
    %v1832 = stablehlo.reshape %v1831 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1833 = stablehlo.reshape %v1832 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1834 = stablehlo.dot_general %v1833, %b5_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x1536xf32>) -> tensor<32x197x1536xf32>
    %v1835 = stablehlo.broadcast_in_dim %b5_bfc1, dims = [2] : (tensor<1536xf32>) -> tensor<32x197x1536xf32>
    %v1836 = stablehlo.add %v1834, %v1835 : tensor<32x197x1536xf32>
    %v1837 = stablehlo.reshape %v1836 : (tensor<32x197x1536xf32>) -> tensor<32x302592xf32>
    %v1838 = stablehlo.multiply %v1837, %v1837 : tensor<32x302592xf32>
    %v1839 = stablehlo.multiply %v1838, %v1837 : tensor<32x302592xf32>
    %v1840 = stablehlo.constant dense<0.044715> : tensor<32x302592xf32>
    %v1841 = stablehlo.multiply %v1840, %v1839 : tensor<32x302592xf32>
    %v1842 = stablehlo.add %v1837, %v1841 : tensor<32x302592xf32>
    %v1843 = stablehlo.constant dense<0.7978845608028654> : tensor<32x302592xf32>
    %v1844 = stablehlo.multiply %v1843, %v1842 : tensor<32x302592xf32>
    %v1845 = stablehlo.tanh %v1844 : tensor<32x302592xf32>
    %v1846 = stablehlo.constant dense<1.0> : tensor<32x302592xf32>
    %v1847 = stablehlo.add %v1846, %v1845 : tensor<32x302592xf32>
    %v1848 = stablehlo.constant dense<0.5> : tensor<32x302592xf32>
    %v1849 = stablehlo.multiply %v1848, %v1837 : tensor<32x302592xf32>
    %v1850 = stablehlo.multiply %v1849, %v1847 : tensor<32x302592xf32>
    %v1851 = stablehlo.reshape %v1850 : (tensor<32x302592xf32>) -> tensor<32x197x1536xf32>
    %v1852 = stablehlo.dot_general %v1851, %b5_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x1536xf32>, tensor<1536x384xf32>) -> tensor<32x197x384xf32>
    %v1853 = stablehlo.broadcast_in_dim %b5_bfc2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1854 = stablehlo.add %v1852, %v1853 : tensor<32x197x384xf32>
    %v1855 = stablehlo.reshape %v1854 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1856 = stablehlo.broadcast_in_dim %dp11, dims = [0] : (tensor<32xf32>) -> tensor<32x75648xf32>
    %v1857 = stablehlo.multiply %v1856, %v1855 : tensor<32x75648xf32>
    %v1858 = stablehlo.add %v1804, %v1857 : tensor<32x75648xf32>
    %v1859 = stablehlo.reshape %v1858 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1860 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1861 = stablehlo.constant dense<384.0> : tensor<32x197x384xf32>
    %v1862 = stablehlo.constant dense<1.0e-5> : tensor<32x197x384xf32>
    %v1863 = stablehlo.reduce(%v1859 init: %v1860) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1864 = stablehlo.broadcast_in_dim %v1863, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v1865 = stablehlo.divide %v1864, %v1861 : tensor<32x197x384xf32>
    %v1866 = stablehlo.subtract %v1859, %v1865 : tensor<32x197x384xf32>
    %v1867 = stablehlo.multiply %v1866, %v1866 : tensor<32x197x384xf32>
    %v1868 = stablehlo.reduce(%v1867 init: %v1860) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1869 = stablehlo.broadcast_in_dim %v1868, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v1870 = stablehlo.divide %v1869, %v1861 : tensor<32x197x384xf32>
    %v1871 = stablehlo.add %v1870, %v1862 : tensor<32x197x384xf32>
    %v1872 = stablehlo.rsqrt %v1871 : tensor<32x197x384xf32>
    %v1873 = stablehlo.multiply %v1866, %v1872 : tensor<32x197x384xf32>
    %v1874 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v1875 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v1876 = stablehlo.multiply %v1873, %v1874 : tensor<32x197x384xf32>
    %v1877 = stablehlo.add %v1876, %v1875 : tensor<32x197x384xf32>
    %v1878 = stablehlo.reshape %v1877 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1879 = stablehlo.reshape %v1878 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1880 = stablehlo.broadcast_in_dim %b6_g1, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1881 = stablehlo.multiply %v1879, %v1880 : tensor<32x197x384xf32>
    %v1882 = stablehlo.reshape %v1881 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1883 = stablehlo.reshape %v1882 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1884 = stablehlo.broadcast_in_dim %b6_bt1, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1885 = stablehlo.add %v1883, %v1884 : tensor<32x197x384xf32>
    %v1886 = stablehlo.reshape %v1885 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1887 = stablehlo.reshape %v1886 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1888 = stablehlo.dot_general %v1887, %b6_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v1889 = stablehlo.broadcast_in_dim %b6_bq, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1890 = stablehlo.add %v1888, %v1889 : tensor<32x197x384xf32>
    %v1891 = stablehlo.reshape %v1890 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1892 = stablehlo.reshape %v1886 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1893 = stablehlo.dot_general %v1892, %b6_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v1894 = stablehlo.broadcast_in_dim %b6_bk, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1895 = stablehlo.add %v1893, %v1894 : tensor<32x197x384xf32>
    %v1896 = stablehlo.reshape %v1895 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1897 = stablehlo.reshape %v1886 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1898 = stablehlo.dot_general %v1897, %b6_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v1899 = stablehlo.broadcast_in_dim %b6_bv, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v1900 = stablehlo.add %v1898, %v1899 : tensor<32x197x384xf32>
    %v1901 = stablehlo.reshape %v1900 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1902 = stablehlo.reshape %v1891 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1903 = stablehlo.slice %v1902 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1904 = stablehlo.reshape %v1903 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1905 = stablehlo.reshape %v1896 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1906 = stablehlo.slice %v1905 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1907 = stablehlo.reshape %v1906 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1908 = stablehlo.reshape %v1901 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1909 = stablehlo.slice %v1908 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1910 = stablehlo.reshape %v1909 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1911 = stablehlo.reshape %v1907 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1912 = stablehlo.transpose %v1911, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1913 = stablehlo.reshape %v1912 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1914 = stablehlo.reshape %v1904 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1915 = stablehlo.reshape %v1913 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1916 = stablehlo.dot_general %v1914, %v1915, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1917 = stablehlo.reshape %v1916 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1918 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1919 = stablehlo.multiply %v1917, %v1918 : tensor<32x38809xf32>
    %v1920 = stablehlo.reshape %v1919 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1921 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1922 = stablehlo.exponential %v1920 : tensor<32x197x197xf32>
    %v1923 = stablehlo.reduce(%v1922 init: %v1921) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1924 = stablehlo.broadcast_in_dim %v1923, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1925 = stablehlo.divide %v1922, %v1924 : tensor<32x197x197xf32>
    %v1926 = stablehlo.reshape %v1925 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1927 = stablehlo.reshape %v1926 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1928 = stablehlo.reshape %v1910 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1929 = stablehlo.dot_general %v1927, %v1928, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1930 = stablehlo.reshape %v1929 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1931 = stablehlo.reshape %v1930 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1932 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1933 = stablehlo.pad %v1931, %v1932, low = [0, 0, 0], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v1934 = stablehlo.reshape %v1933 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1935 = stablehlo.reshape %v1891 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1936 = stablehlo.slice %v1935 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1937 = stablehlo.reshape %v1936 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1938 = stablehlo.reshape %v1896 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1939 = stablehlo.slice %v1938 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1940 = stablehlo.reshape %v1939 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1941 = stablehlo.reshape %v1901 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1942 = stablehlo.slice %v1941 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1943 = stablehlo.reshape %v1942 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1944 = stablehlo.reshape %v1940 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1945 = stablehlo.transpose %v1944, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1946 = stablehlo.reshape %v1945 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1947 = stablehlo.reshape %v1937 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1948 = stablehlo.reshape %v1946 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1949 = stablehlo.dot_general %v1947, %v1948, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1950 = stablehlo.reshape %v1949 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1951 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1952 = stablehlo.multiply %v1950, %v1951 : tensor<32x38809xf32>
    %v1953 = stablehlo.reshape %v1952 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1954 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1955 = stablehlo.exponential %v1953 : tensor<32x197x197xf32>
    %v1956 = stablehlo.reduce(%v1955 init: %v1954) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1957 = stablehlo.broadcast_in_dim %v1956, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1958 = stablehlo.divide %v1955, %v1957 : tensor<32x197x197xf32>
    %v1959 = stablehlo.reshape %v1958 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1960 = stablehlo.reshape %v1959 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1961 = stablehlo.reshape %v1943 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1962 = stablehlo.dot_general %v1960, %v1961, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1963 = stablehlo.reshape %v1962 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1964 = stablehlo.reshape %v1963 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1965 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1966 = stablehlo.pad %v1964, %v1965, low = [0, 0, 64], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v1967 = stablehlo.reshape %v1966 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v1968 = stablehlo.add %v1934, %v1967 : tensor<32x75648xf32>
    %v1969 = stablehlo.reshape %v1891 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1970 = stablehlo.slice %v1969 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1971 = stablehlo.reshape %v1970 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1972 = stablehlo.reshape %v1896 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1973 = stablehlo.slice %v1972 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v1974 = stablehlo.reshape %v1973 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1975 = stablehlo.reshape %v1901 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v1976 = stablehlo.slice %v1975 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
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
    %v1988 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1989 = stablehlo.exponential %v1987 : tensor<32x197x197xf32>
    %v1990 = stablehlo.reduce(%v1989 init: %v1988) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1991 = stablehlo.broadcast_in_dim %v1990, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1992 = stablehlo.divide %v1989, %v1991 : tensor<32x197x197xf32>
    %v1993 = stablehlo.reshape %v1992 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1994 = stablehlo.reshape %v1993 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1995 = stablehlo.reshape %v1977 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1996 = stablehlo.dot_general %v1994, %v1995, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1997 = stablehlo.reshape %v1996 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1998 = stablehlo.reshape %v1997 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1999 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2000 = stablehlo.pad %v1998, %v1999, low = [0, 0, 128], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v2001 = stablehlo.reshape %v2000 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2002 = stablehlo.add %v1968, %v2001 : tensor<32x75648xf32>
    %v2003 = stablehlo.reshape %v1891 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2004 = stablehlo.slice %v2003 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2005 = stablehlo.reshape %v2004 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2006 = stablehlo.reshape %v1896 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2007 = stablehlo.slice %v2006 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2008 = stablehlo.reshape %v2007 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2009 = stablehlo.reshape %v1901 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2010 = stablehlo.slice %v2009 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2011 = stablehlo.reshape %v2010 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2012 = stablehlo.reshape %v2008 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2013 = stablehlo.transpose %v2012, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2014 = stablehlo.reshape %v2013 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2015 = stablehlo.reshape %v2005 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2016 = stablehlo.reshape %v2014 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2017 = stablehlo.dot_general %v2015, %v2016, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2018 = stablehlo.reshape %v2017 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2019 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2020 = stablehlo.multiply %v2018, %v2019 : tensor<32x38809xf32>
    %v2021 = stablehlo.reshape %v2020 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2022 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2023 = stablehlo.exponential %v2021 : tensor<32x197x197xf32>
    %v2024 = stablehlo.reduce(%v2023 init: %v2022) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2025 = stablehlo.broadcast_in_dim %v2024, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2026 = stablehlo.divide %v2023, %v2025 : tensor<32x197x197xf32>
    %v2027 = stablehlo.reshape %v2026 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2028 = stablehlo.reshape %v2027 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2029 = stablehlo.reshape %v2011 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2030 = stablehlo.dot_general %v2028, %v2029, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2031 = stablehlo.reshape %v2030 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2032 = stablehlo.reshape %v2031 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2033 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2034 = stablehlo.pad %v2032, %v2033, low = [0, 0, 192], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v2035 = stablehlo.reshape %v2034 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2036 = stablehlo.add %v2002, %v2035 : tensor<32x75648xf32>
    %v2037 = stablehlo.reshape %v1891 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2038 = stablehlo.slice %v2037 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2039 = stablehlo.reshape %v2038 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2040 = stablehlo.reshape %v1896 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2041 = stablehlo.slice %v2040 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2042 = stablehlo.reshape %v2041 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2043 = stablehlo.reshape %v1901 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2044 = stablehlo.slice %v2043 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2045 = stablehlo.reshape %v2044 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2046 = stablehlo.reshape %v2042 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2047 = stablehlo.transpose %v2046, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2048 = stablehlo.reshape %v2047 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2049 = stablehlo.reshape %v2039 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2050 = stablehlo.reshape %v2048 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2051 = stablehlo.dot_general %v2049, %v2050, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2052 = stablehlo.reshape %v2051 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2053 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2054 = stablehlo.multiply %v2052, %v2053 : tensor<32x38809xf32>
    %v2055 = stablehlo.reshape %v2054 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2056 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2057 = stablehlo.exponential %v2055 : tensor<32x197x197xf32>
    %v2058 = stablehlo.reduce(%v2057 init: %v2056) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2059 = stablehlo.broadcast_in_dim %v2058, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2060 = stablehlo.divide %v2057, %v2059 : tensor<32x197x197xf32>
    %v2061 = stablehlo.reshape %v2060 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2062 = stablehlo.reshape %v2061 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2063 = stablehlo.reshape %v2045 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2064 = stablehlo.dot_general %v2062, %v2063, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2065 = stablehlo.reshape %v2064 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2066 = stablehlo.reshape %v2065 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2067 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2068 = stablehlo.pad %v2066, %v2067, low = [0, 0, 256], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v2069 = stablehlo.reshape %v2068 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2070 = stablehlo.add %v2036, %v2069 : tensor<32x75648xf32>
    %v2071 = stablehlo.reshape %v1891 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2072 = stablehlo.slice %v2071 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2073 = stablehlo.reshape %v2072 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2074 = stablehlo.reshape %v1896 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2075 = stablehlo.slice %v2074 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2076 = stablehlo.reshape %v2075 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2077 = stablehlo.reshape %v1901 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2078 = stablehlo.slice %v2077 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2079 = stablehlo.reshape %v2078 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2080 = stablehlo.reshape %v2076 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2081 = stablehlo.transpose %v2080, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2082 = stablehlo.reshape %v2081 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2083 = stablehlo.reshape %v2073 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2084 = stablehlo.reshape %v2082 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2085 = stablehlo.dot_general %v2083, %v2084, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2086 = stablehlo.reshape %v2085 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2087 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2088 = stablehlo.multiply %v2086, %v2087 : tensor<32x38809xf32>
    %v2089 = stablehlo.reshape %v2088 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2090 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2091 = stablehlo.exponential %v2089 : tensor<32x197x197xf32>
    %v2092 = stablehlo.reduce(%v2091 init: %v2090) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2093 = stablehlo.broadcast_in_dim %v2092, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2094 = stablehlo.divide %v2091, %v2093 : tensor<32x197x197xf32>
    %v2095 = stablehlo.reshape %v2094 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2096 = stablehlo.reshape %v2095 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2097 = stablehlo.reshape %v2079 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2098 = stablehlo.dot_general %v2096, %v2097, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2099 = stablehlo.reshape %v2098 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2100 = stablehlo.reshape %v2099 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2101 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2102 = stablehlo.pad %v2100, %v2101, low = [0, 0, 320], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v2103 = stablehlo.reshape %v2102 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2104 = stablehlo.add %v2070, %v2103 : tensor<32x75648xf32>
    %v2105 = stablehlo.reshape %v2104 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2106 = stablehlo.dot_general %v2105, %b6_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v2107 = stablehlo.broadcast_in_dim %b6_bo, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2108 = stablehlo.add %v2106, %v2107 : tensor<32x197x384xf32>
    %v2109 = stablehlo.reshape %v2108 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2110 = stablehlo.broadcast_in_dim %dp12, dims = [0] : (tensor<32xf32>) -> tensor<32x75648xf32>
    %v2111 = stablehlo.multiply %v2110, %v2109 : tensor<32x75648xf32>
    %v2112 = stablehlo.add %v1858, %v2111 : tensor<32x75648xf32>
    %v2113 = stablehlo.reshape %v2112 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2114 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2115 = stablehlo.constant dense<384.0> : tensor<32x197x384xf32>
    %v2116 = stablehlo.constant dense<1.0e-5> : tensor<32x197x384xf32>
    %v2117 = stablehlo.reduce(%v2113 init: %v2114) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2118 = stablehlo.broadcast_in_dim %v2117, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v2119 = stablehlo.divide %v2118, %v2115 : tensor<32x197x384xf32>
    %v2120 = stablehlo.subtract %v2113, %v2119 : tensor<32x197x384xf32>
    %v2121 = stablehlo.multiply %v2120, %v2120 : tensor<32x197x384xf32>
    %v2122 = stablehlo.reduce(%v2121 init: %v2114) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2123 = stablehlo.broadcast_in_dim %v2122, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v2124 = stablehlo.divide %v2123, %v2115 : tensor<32x197x384xf32>
    %v2125 = stablehlo.add %v2124, %v2116 : tensor<32x197x384xf32>
    %v2126 = stablehlo.rsqrt %v2125 : tensor<32x197x384xf32>
    %v2127 = stablehlo.multiply %v2120, %v2126 : tensor<32x197x384xf32>
    %v2128 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v2129 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v2130 = stablehlo.multiply %v2127, %v2128 : tensor<32x197x384xf32>
    %v2131 = stablehlo.add %v2130, %v2129 : tensor<32x197x384xf32>
    %v2132 = stablehlo.reshape %v2131 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2133 = stablehlo.reshape %v2132 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2134 = stablehlo.broadcast_in_dim %b6_g2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2135 = stablehlo.multiply %v2133, %v2134 : tensor<32x197x384xf32>
    %v2136 = stablehlo.reshape %v2135 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2137 = stablehlo.reshape %v2136 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2138 = stablehlo.broadcast_in_dim %b6_bt2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2139 = stablehlo.add %v2137, %v2138 : tensor<32x197x384xf32>
    %v2140 = stablehlo.reshape %v2139 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2141 = stablehlo.reshape %v2140 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2142 = stablehlo.dot_general %v2141, %b6_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x1536xf32>) -> tensor<32x197x1536xf32>
    %v2143 = stablehlo.broadcast_in_dim %b6_bfc1, dims = [2] : (tensor<1536xf32>) -> tensor<32x197x1536xf32>
    %v2144 = stablehlo.add %v2142, %v2143 : tensor<32x197x1536xf32>
    %v2145 = stablehlo.reshape %v2144 : (tensor<32x197x1536xf32>) -> tensor<32x302592xf32>
    %v2146 = stablehlo.multiply %v2145, %v2145 : tensor<32x302592xf32>
    %v2147 = stablehlo.multiply %v2146, %v2145 : tensor<32x302592xf32>
    %v2148 = stablehlo.constant dense<0.044715> : tensor<32x302592xf32>
    %v2149 = stablehlo.multiply %v2148, %v2147 : tensor<32x302592xf32>
    %v2150 = stablehlo.add %v2145, %v2149 : tensor<32x302592xf32>
    %v2151 = stablehlo.constant dense<0.7978845608028654> : tensor<32x302592xf32>
    %v2152 = stablehlo.multiply %v2151, %v2150 : tensor<32x302592xf32>
    %v2153 = stablehlo.tanh %v2152 : tensor<32x302592xf32>
    %v2154 = stablehlo.constant dense<1.0> : tensor<32x302592xf32>
    %v2155 = stablehlo.add %v2154, %v2153 : tensor<32x302592xf32>
    %v2156 = stablehlo.constant dense<0.5> : tensor<32x302592xf32>
    %v2157 = stablehlo.multiply %v2156, %v2145 : tensor<32x302592xf32>
    %v2158 = stablehlo.multiply %v2157, %v2155 : tensor<32x302592xf32>
    %v2159 = stablehlo.reshape %v2158 : (tensor<32x302592xf32>) -> tensor<32x197x1536xf32>
    %v2160 = stablehlo.dot_general %v2159, %b6_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x1536xf32>, tensor<1536x384xf32>) -> tensor<32x197x384xf32>
    %v2161 = stablehlo.broadcast_in_dim %b6_bfc2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2162 = stablehlo.add %v2160, %v2161 : tensor<32x197x384xf32>
    %v2163 = stablehlo.reshape %v2162 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2164 = stablehlo.broadcast_in_dim %dp13, dims = [0] : (tensor<32xf32>) -> tensor<32x75648xf32>
    %v2165 = stablehlo.multiply %v2164, %v2163 : tensor<32x75648xf32>
    %v2166 = stablehlo.add %v2112, %v2165 : tensor<32x75648xf32>
    %v2167 = stablehlo.reshape %v2166 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2168 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2169 = stablehlo.constant dense<384.0> : tensor<32x197x384xf32>
    %v2170 = stablehlo.constant dense<1.0e-5> : tensor<32x197x384xf32>
    %v2171 = stablehlo.reduce(%v2167 init: %v2168) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2172 = stablehlo.broadcast_in_dim %v2171, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v2173 = stablehlo.divide %v2172, %v2169 : tensor<32x197x384xf32>
    %v2174 = stablehlo.subtract %v2167, %v2173 : tensor<32x197x384xf32>
    %v2175 = stablehlo.multiply %v2174, %v2174 : tensor<32x197x384xf32>
    %v2176 = stablehlo.reduce(%v2175 init: %v2168) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2177 = stablehlo.broadcast_in_dim %v2176, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v2178 = stablehlo.divide %v2177, %v2169 : tensor<32x197x384xf32>
    %v2179 = stablehlo.add %v2178, %v2170 : tensor<32x197x384xf32>
    %v2180 = stablehlo.rsqrt %v2179 : tensor<32x197x384xf32>
    %v2181 = stablehlo.multiply %v2174, %v2180 : tensor<32x197x384xf32>
    %v2182 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v2183 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v2184 = stablehlo.multiply %v2181, %v2182 : tensor<32x197x384xf32>
    %v2185 = stablehlo.add %v2184, %v2183 : tensor<32x197x384xf32>
    %v2186 = stablehlo.reshape %v2185 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2187 = stablehlo.reshape %v2186 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2188 = stablehlo.broadcast_in_dim %b7_g1, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2189 = stablehlo.multiply %v2187, %v2188 : tensor<32x197x384xf32>
    %v2190 = stablehlo.reshape %v2189 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2191 = stablehlo.reshape %v2190 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2192 = stablehlo.broadcast_in_dim %b7_bt1, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2193 = stablehlo.add %v2191, %v2192 : tensor<32x197x384xf32>
    %v2194 = stablehlo.reshape %v2193 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2195 = stablehlo.reshape %v2194 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2196 = stablehlo.dot_general %v2195, %b7_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v2197 = stablehlo.broadcast_in_dim %b7_bq, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2198 = stablehlo.add %v2196, %v2197 : tensor<32x197x384xf32>
    %v2199 = stablehlo.reshape %v2198 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2200 = stablehlo.reshape %v2194 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2201 = stablehlo.dot_general %v2200, %b7_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v2202 = stablehlo.broadcast_in_dim %b7_bk, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2203 = stablehlo.add %v2201, %v2202 : tensor<32x197x384xf32>
    %v2204 = stablehlo.reshape %v2203 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2205 = stablehlo.reshape %v2194 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2206 = stablehlo.dot_general %v2205, %b7_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v2207 = stablehlo.broadcast_in_dim %b7_bv, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2208 = stablehlo.add %v2206, %v2207 : tensor<32x197x384xf32>
    %v2209 = stablehlo.reshape %v2208 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2210 = stablehlo.reshape %v2199 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2211 = stablehlo.slice %v2210 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2212 = stablehlo.reshape %v2211 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2213 = stablehlo.reshape %v2204 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2214 = stablehlo.slice %v2213 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2215 = stablehlo.reshape %v2214 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2216 = stablehlo.reshape %v2209 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2217 = stablehlo.slice %v2216 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2218 = stablehlo.reshape %v2217 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2219 = stablehlo.reshape %v2215 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2220 = stablehlo.transpose %v2219, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2221 = stablehlo.reshape %v2220 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2222 = stablehlo.reshape %v2212 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2223 = stablehlo.reshape %v2221 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2224 = stablehlo.dot_general %v2222, %v2223, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2225 = stablehlo.reshape %v2224 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2226 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2227 = stablehlo.multiply %v2225, %v2226 : tensor<32x38809xf32>
    %v2228 = stablehlo.reshape %v2227 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2229 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2230 = stablehlo.exponential %v2228 : tensor<32x197x197xf32>
    %v2231 = stablehlo.reduce(%v2230 init: %v2229) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2232 = stablehlo.broadcast_in_dim %v2231, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2233 = stablehlo.divide %v2230, %v2232 : tensor<32x197x197xf32>
    %v2234 = stablehlo.reshape %v2233 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2235 = stablehlo.reshape %v2234 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2236 = stablehlo.reshape %v2218 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2237 = stablehlo.dot_general %v2235, %v2236, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2238 = stablehlo.reshape %v2237 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2239 = stablehlo.reshape %v2238 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2240 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2241 = stablehlo.pad %v2239, %v2240, low = [0, 0, 0], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v2242 = stablehlo.reshape %v2241 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2243 = stablehlo.reshape %v2199 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2244 = stablehlo.slice %v2243 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2245 = stablehlo.reshape %v2244 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2246 = stablehlo.reshape %v2204 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2247 = stablehlo.slice %v2246 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2248 = stablehlo.reshape %v2247 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2249 = stablehlo.reshape %v2209 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2250 = stablehlo.slice %v2249 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2251 = stablehlo.reshape %v2250 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2252 = stablehlo.reshape %v2248 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2253 = stablehlo.transpose %v2252, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2254 = stablehlo.reshape %v2253 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2255 = stablehlo.reshape %v2245 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2256 = stablehlo.reshape %v2254 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2257 = stablehlo.dot_general %v2255, %v2256, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2258 = stablehlo.reshape %v2257 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2259 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2260 = stablehlo.multiply %v2258, %v2259 : tensor<32x38809xf32>
    %v2261 = stablehlo.reshape %v2260 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2262 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2263 = stablehlo.exponential %v2261 : tensor<32x197x197xf32>
    %v2264 = stablehlo.reduce(%v2263 init: %v2262) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2265 = stablehlo.broadcast_in_dim %v2264, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2266 = stablehlo.divide %v2263, %v2265 : tensor<32x197x197xf32>
    %v2267 = stablehlo.reshape %v2266 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2268 = stablehlo.reshape %v2267 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2269 = stablehlo.reshape %v2251 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2270 = stablehlo.dot_general %v2268, %v2269, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2271 = stablehlo.reshape %v2270 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2272 = stablehlo.reshape %v2271 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2273 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2274 = stablehlo.pad %v2272, %v2273, low = [0, 0, 64], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v2275 = stablehlo.reshape %v2274 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2276 = stablehlo.add %v2242, %v2275 : tensor<32x75648xf32>
    %v2277 = stablehlo.reshape %v2199 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2278 = stablehlo.slice %v2277 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2279 = stablehlo.reshape %v2278 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2280 = stablehlo.reshape %v2204 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2281 = stablehlo.slice %v2280 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2282 = stablehlo.reshape %v2281 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2283 = stablehlo.reshape %v2209 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2284 = stablehlo.slice %v2283 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2285 = stablehlo.reshape %v2284 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2286 = stablehlo.reshape %v2282 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2287 = stablehlo.transpose %v2286, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2288 = stablehlo.reshape %v2287 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2289 = stablehlo.reshape %v2279 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2290 = stablehlo.reshape %v2288 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2291 = stablehlo.dot_general %v2289, %v2290, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2292 = stablehlo.reshape %v2291 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2293 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2294 = stablehlo.multiply %v2292, %v2293 : tensor<32x38809xf32>
    %v2295 = stablehlo.reshape %v2294 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2296 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2297 = stablehlo.exponential %v2295 : tensor<32x197x197xf32>
    %v2298 = stablehlo.reduce(%v2297 init: %v2296) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2299 = stablehlo.broadcast_in_dim %v2298, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2300 = stablehlo.divide %v2297, %v2299 : tensor<32x197x197xf32>
    %v2301 = stablehlo.reshape %v2300 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2302 = stablehlo.reshape %v2301 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2303 = stablehlo.reshape %v2285 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2304 = stablehlo.dot_general %v2302, %v2303, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2305 = stablehlo.reshape %v2304 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2306 = stablehlo.reshape %v2305 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2307 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2308 = stablehlo.pad %v2306, %v2307, low = [0, 0, 128], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v2309 = stablehlo.reshape %v2308 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2310 = stablehlo.add %v2276, %v2309 : tensor<32x75648xf32>
    %v2311 = stablehlo.reshape %v2199 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2312 = stablehlo.slice %v2311 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2313 = stablehlo.reshape %v2312 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2314 = stablehlo.reshape %v2204 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2315 = stablehlo.slice %v2314 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2316 = stablehlo.reshape %v2315 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2317 = stablehlo.reshape %v2209 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2318 = stablehlo.slice %v2317 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2319 = stablehlo.reshape %v2318 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2320 = stablehlo.reshape %v2316 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2321 = stablehlo.transpose %v2320, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2322 = stablehlo.reshape %v2321 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2323 = stablehlo.reshape %v2313 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2324 = stablehlo.reshape %v2322 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2325 = stablehlo.dot_general %v2323, %v2324, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2326 = stablehlo.reshape %v2325 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2327 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2328 = stablehlo.multiply %v2326, %v2327 : tensor<32x38809xf32>
    %v2329 = stablehlo.reshape %v2328 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2330 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2331 = stablehlo.exponential %v2329 : tensor<32x197x197xf32>
    %v2332 = stablehlo.reduce(%v2331 init: %v2330) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2333 = stablehlo.broadcast_in_dim %v2332, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2334 = stablehlo.divide %v2331, %v2333 : tensor<32x197x197xf32>
    %v2335 = stablehlo.reshape %v2334 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2336 = stablehlo.reshape %v2335 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2337 = stablehlo.reshape %v2319 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2338 = stablehlo.dot_general %v2336, %v2337, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2339 = stablehlo.reshape %v2338 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2340 = stablehlo.reshape %v2339 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2341 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2342 = stablehlo.pad %v2340, %v2341, low = [0, 0, 192], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v2343 = stablehlo.reshape %v2342 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2344 = stablehlo.add %v2310, %v2343 : tensor<32x75648xf32>
    %v2345 = stablehlo.reshape %v2199 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2346 = stablehlo.slice %v2345 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2347 = stablehlo.reshape %v2346 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2348 = stablehlo.reshape %v2204 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2349 = stablehlo.slice %v2348 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2350 = stablehlo.reshape %v2349 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2351 = stablehlo.reshape %v2209 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2352 = stablehlo.slice %v2351 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2353 = stablehlo.reshape %v2352 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2354 = stablehlo.reshape %v2350 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2355 = stablehlo.transpose %v2354, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2356 = stablehlo.reshape %v2355 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2357 = stablehlo.reshape %v2347 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2358 = stablehlo.reshape %v2356 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2359 = stablehlo.dot_general %v2357, %v2358, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2360 = stablehlo.reshape %v2359 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2361 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2362 = stablehlo.multiply %v2360, %v2361 : tensor<32x38809xf32>
    %v2363 = stablehlo.reshape %v2362 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2364 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2365 = stablehlo.exponential %v2363 : tensor<32x197x197xf32>
    %v2366 = stablehlo.reduce(%v2365 init: %v2364) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2367 = stablehlo.broadcast_in_dim %v2366, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2368 = stablehlo.divide %v2365, %v2367 : tensor<32x197x197xf32>
    %v2369 = stablehlo.reshape %v2368 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2370 = stablehlo.reshape %v2369 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2371 = stablehlo.reshape %v2353 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2372 = stablehlo.dot_general %v2370, %v2371, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2373 = stablehlo.reshape %v2372 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2374 = stablehlo.reshape %v2373 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2375 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2376 = stablehlo.pad %v2374, %v2375, low = [0, 0, 256], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v2377 = stablehlo.reshape %v2376 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2378 = stablehlo.add %v2344, %v2377 : tensor<32x75648xf32>
    %v2379 = stablehlo.reshape %v2199 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2380 = stablehlo.slice %v2379 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2381 = stablehlo.reshape %v2380 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2382 = stablehlo.reshape %v2204 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2383 = stablehlo.slice %v2382 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2384 = stablehlo.reshape %v2383 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2385 = stablehlo.reshape %v2209 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2386 = stablehlo.slice %v2385 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2387 = stablehlo.reshape %v2386 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2388 = stablehlo.reshape %v2384 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2389 = stablehlo.transpose %v2388, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2390 = stablehlo.reshape %v2389 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2391 = stablehlo.reshape %v2381 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2392 = stablehlo.reshape %v2390 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2393 = stablehlo.dot_general %v2391, %v2392, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2394 = stablehlo.reshape %v2393 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2395 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2396 = stablehlo.multiply %v2394, %v2395 : tensor<32x38809xf32>
    %v2397 = stablehlo.reshape %v2396 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2398 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2399 = stablehlo.exponential %v2397 : tensor<32x197x197xf32>
    %v2400 = stablehlo.reduce(%v2399 init: %v2398) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2401 = stablehlo.broadcast_in_dim %v2400, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2402 = stablehlo.divide %v2399, %v2401 : tensor<32x197x197xf32>
    %v2403 = stablehlo.reshape %v2402 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2404 = stablehlo.reshape %v2403 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2405 = stablehlo.reshape %v2387 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2406 = stablehlo.dot_general %v2404, %v2405, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2407 = stablehlo.reshape %v2406 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2408 = stablehlo.reshape %v2407 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2409 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2410 = stablehlo.pad %v2408, %v2409, low = [0, 0, 320], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v2411 = stablehlo.reshape %v2410 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2412 = stablehlo.add %v2378, %v2411 : tensor<32x75648xf32>
    %v2413 = stablehlo.reshape %v2412 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2414 = stablehlo.dot_general %v2413, %b7_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v2415 = stablehlo.broadcast_in_dim %b7_bo, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2416 = stablehlo.add %v2414, %v2415 : tensor<32x197x384xf32>
    %v2417 = stablehlo.reshape %v2416 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2418 = stablehlo.broadcast_in_dim %dp14, dims = [0] : (tensor<32xf32>) -> tensor<32x75648xf32>
    %v2419 = stablehlo.multiply %v2418, %v2417 : tensor<32x75648xf32>
    %v2420 = stablehlo.add %v2166, %v2419 : tensor<32x75648xf32>
    %v2421 = stablehlo.reshape %v2420 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2422 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2423 = stablehlo.constant dense<384.0> : tensor<32x197x384xf32>
    %v2424 = stablehlo.constant dense<1.0e-5> : tensor<32x197x384xf32>
    %v2425 = stablehlo.reduce(%v2421 init: %v2422) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2426 = stablehlo.broadcast_in_dim %v2425, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v2427 = stablehlo.divide %v2426, %v2423 : tensor<32x197x384xf32>
    %v2428 = stablehlo.subtract %v2421, %v2427 : tensor<32x197x384xf32>
    %v2429 = stablehlo.multiply %v2428, %v2428 : tensor<32x197x384xf32>
    %v2430 = stablehlo.reduce(%v2429 init: %v2422) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2431 = stablehlo.broadcast_in_dim %v2430, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v2432 = stablehlo.divide %v2431, %v2423 : tensor<32x197x384xf32>
    %v2433 = stablehlo.add %v2432, %v2424 : tensor<32x197x384xf32>
    %v2434 = stablehlo.rsqrt %v2433 : tensor<32x197x384xf32>
    %v2435 = stablehlo.multiply %v2428, %v2434 : tensor<32x197x384xf32>
    %v2436 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v2437 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v2438 = stablehlo.multiply %v2435, %v2436 : tensor<32x197x384xf32>
    %v2439 = stablehlo.add %v2438, %v2437 : tensor<32x197x384xf32>
    %v2440 = stablehlo.reshape %v2439 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2441 = stablehlo.reshape %v2440 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2442 = stablehlo.broadcast_in_dim %b7_g2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2443 = stablehlo.multiply %v2441, %v2442 : tensor<32x197x384xf32>
    %v2444 = stablehlo.reshape %v2443 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2445 = stablehlo.reshape %v2444 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2446 = stablehlo.broadcast_in_dim %b7_bt2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2447 = stablehlo.add %v2445, %v2446 : tensor<32x197x384xf32>
    %v2448 = stablehlo.reshape %v2447 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2449 = stablehlo.reshape %v2448 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2450 = stablehlo.dot_general %v2449, %b7_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x1536xf32>) -> tensor<32x197x1536xf32>
    %v2451 = stablehlo.broadcast_in_dim %b7_bfc1, dims = [2] : (tensor<1536xf32>) -> tensor<32x197x1536xf32>
    %v2452 = stablehlo.add %v2450, %v2451 : tensor<32x197x1536xf32>
    %v2453 = stablehlo.reshape %v2452 : (tensor<32x197x1536xf32>) -> tensor<32x302592xf32>
    %v2454 = stablehlo.multiply %v2453, %v2453 : tensor<32x302592xf32>
    %v2455 = stablehlo.multiply %v2454, %v2453 : tensor<32x302592xf32>
    %v2456 = stablehlo.constant dense<0.044715> : tensor<32x302592xf32>
    %v2457 = stablehlo.multiply %v2456, %v2455 : tensor<32x302592xf32>
    %v2458 = stablehlo.add %v2453, %v2457 : tensor<32x302592xf32>
    %v2459 = stablehlo.constant dense<0.7978845608028654> : tensor<32x302592xf32>
    %v2460 = stablehlo.multiply %v2459, %v2458 : tensor<32x302592xf32>
    %v2461 = stablehlo.tanh %v2460 : tensor<32x302592xf32>
    %v2462 = stablehlo.constant dense<1.0> : tensor<32x302592xf32>
    %v2463 = stablehlo.add %v2462, %v2461 : tensor<32x302592xf32>
    %v2464 = stablehlo.constant dense<0.5> : tensor<32x302592xf32>
    %v2465 = stablehlo.multiply %v2464, %v2453 : tensor<32x302592xf32>
    %v2466 = stablehlo.multiply %v2465, %v2463 : tensor<32x302592xf32>
    %v2467 = stablehlo.reshape %v2466 : (tensor<32x302592xf32>) -> tensor<32x197x1536xf32>
    %v2468 = stablehlo.dot_general %v2467, %b7_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x1536xf32>, tensor<1536x384xf32>) -> tensor<32x197x384xf32>
    %v2469 = stablehlo.broadcast_in_dim %b7_bfc2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2470 = stablehlo.add %v2468, %v2469 : tensor<32x197x384xf32>
    %v2471 = stablehlo.reshape %v2470 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2472 = stablehlo.broadcast_in_dim %dp15, dims = [0] : (tensor<32xf32>) -> tensor<32x75648xf32>
    %v2473 = stablehlo.multiply %v2472, %v2471 : tensor<32x75648xf32>
    %v2474 = stablehlo.add %v2420, %v2473 : tensor<32x75648xf32>
    %v2475 = stablehlo.reshape %v2474 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2476 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2477 = stablehlo.constant dense<384.0> : tensor<32x197x384xf32>
    %v2478 = stablehlo.constant dense<1.0e-5> : tensor<32x197x384xf32>
    %v2479 = stablehlo.reduce(%v2475 init: %v2476) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2480 = stablehlo.broadcast_in_dim %v2479, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v2481 = stablehlo.divide %v2480, %v2477 : tensor<32x197x384xf32>
    %v2482 = stablehlo.subtract %v2475, %v2481 : tensor<32x197x384xf32>
    %v2483 = stablehlo.multiply %v2482, %v2482 : tensor<32x197x384xf32>
    %v2484 = stablehlo.reduce(%v2483 init: %v2476) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2485 = stablehlo.broadcast_in_dim %v2484, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v2486 = stablehlo.divide %v2485, %v2477 : tensor<32x197x384xf32>
    %v2487 = stablehlo.add %v2486, %v2478 : tensor<32x197x384xf32>
    %v2488 = stablehlo.rsqrt %v2487 : tensor<32x197x384xf32>
    %v2489 = stablehlo.multiply %v2482, %v2488 : tensor<32x197x384xf32>
    %v2490 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v2491 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v2492 = stablehlo.multiply %v2489, %v2490 : tensor<32x197x384xf32>
    %v2493 = stablehlo.add %v2492, %v2491 : tensor<32x197x384xf32>
    %v2494 = stablehlo.reshape %v2493 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2495 = stablehlo.reshape %v2494 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2496 = stablehlo.broadcast_in_dim %b8_g1, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2497 = stablehlo.multiply %v2495, %v2496 : tensor<32x197x384xf32>
    %v2498 = stablehlo.reshape %v2497 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2499 = stablehlo.reshape %v2498 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2500 = stablehlo.broadcast_in_dim %b8_bt1, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2501 = stablehlo.add %v2499, %v2500 : tensor<32x197x384xf32>
    %v2502 = stablehlo.reshape %v2501 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2503 = stablehlo.reshape %v2502 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2504 = stablehlo.dot_general %v2503, %b8_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v2505 = stablehlo.broadcast_in_dim %b8_bq, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2506 = stablehlo.add %v2504, %v2505 : tensor<32x197x384xf32>
    %v2507 = stablehlo.reshape %v2506 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2508 = stablehlo.reshape %v2502 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2509 = stablehlo.dot_general %v2508, %b8_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v2510 = stablehlo.broadcast_in_dim %b8_bk, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2511 = stablehlo.add %v2509, %v2510 : tensor<32x197x384xf32>
    %v2512 = stablehlo.reshape %v2511 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2513 = stablehlo.reshape %v2502 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2514 = stablehlo.dot_general %v2513, %b8_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v2515 = stablehlo.broadcast_in_dim %b8_bv, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2516 = stablehlo.add %v2514, %v2515 : tensor<32x197x384xf32>
    %v2517 = stablehlo.reshape %v2516 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2518 = stablehlo.reshape %v2507 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2519 = stablehlo.slice %v2518 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2520 = stablehlo.reshape %v2519 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2521 = stablehlo.reshape %v2512 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2522 = stablehlo.slice %v2521 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2523 = stablehlo.reshape %v2522 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2524 = stablehlo.reshape %v2517 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2525 = stablehlo.slice %v2524 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2526 = stablehlo.reshape %v2525 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2527 = stablehlo.reshape %v2523 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2528 = stablehlo.transpose %v2527, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2529 = stablehlo.reshape %v2528 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2530 = stablehlo.reshape %v2520 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2531 = stablehlo.reshape %v2529 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2532 = stablehlo.dot_general %v2530, %v2531, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2533 = stablehlo.reshape %v2532 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2534 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2535 = stablehlo.multiply %v2533, %v2534 : tensor<32x38809xf32>
    %v2536 = stablehlo.reshape %v2535 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2537 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2538 = stablehlo.exponential %v2536 : tensor<32x197x197xf32>
    %v2539 = stablehlo.reduce(%v2538 init: %v2537) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2540 = stablehlo.broadcast_in_dim %v2539, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2541 = stablehlo.divide %v2538, %v2540 : tensor<32x197x197xf32>
    %v2542 = stablehlo.reshape %v2541 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2543 = stablehlo.reshape %v2542 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2544 = stablehlo.reshape %v2526 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2545 = stablehlo.dot_general %v2543, %v2544, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2546 = stablehlo.reshape %v2545 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2547 = stablehlo.reshape %v2546 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2548 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2549 = stablehlo.pad %v2547, %v2548, low = [0, 0, 0], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v2550 = stablehlo.reshape %v2549 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2551 = stablehlo.reshape %v2507 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2552 = stablehlo.slice %v2551 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2553 = stablehlo.reshape %v2552 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2554 = stablehlo.reshape %v2512 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2555 = stablehlo.slice %v2554 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2556 = stablehlo.reshape %v2555 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2557 = stablehlo.reshape %v2517 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2558 = stablehlo.slice %v2557 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2559 = stablehlo.reshape %v2558 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2560 = stablehlo.reshape %v2556 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2561 = stablehlo.transpose %v2560, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2562 = stablehlo.reshape %v2561 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2563 = stablehlo.reshape %v2553 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2564 = stablehlo.reshape %v2562 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2565 = stablehlo.dot_general %v2563, %v2564, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2566 = stablehlo.reshape %v2565 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2567 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2568 = stablehlo.multiply %v2566, %v2567 : tensor<32x38809xf32>
    %v2569 = stablehlo.reshape %v2568 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2570 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2571 = stablehlo.exponential %v2569 : tensor<32x197x197xf32>
    %v2572 = stablehlo.reduce(%v2571 init: %v2570) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2573 = stablehlo.broadcast_in_dim %v2572, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2574 = stablehlo.divide %v2571, %v2573 : tensor<32x197x197xf32>
    %v2575 = stablehlo.reshape %v2574 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2576 = stablehlo.reshape %v2575 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2577 = stablehlo.reshape %v2559 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2578 = stablehlo.dot_general %v2576, %v2577, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2579 = stablehlo.reshape %v2578 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2580 = stablehlo.reshape %v2579 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2581 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2582 = stablehlo.pad %v2580, %v2581, low = [0, 0, 64], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v2583 = stablehlo.reshape %v2582 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2584 = stablehlo.add %v2550, %v2583 : tensor<32x75648xf32>
    %v2585 = stablehlo.reshape %v2507 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2586 = stablehlo.slice %v2585 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2587 = stablehlo.reshape %v2586 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2588 = stablehlo.reshape %v2512 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2589 = stablehlo.slice %v2588 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2590 = stablehlo.reshape %v2589 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2591 = stablehlo.reshape %v2517 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2592 = stablehlo.slice %v2591 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2593 = stablehlo.reshape %v2592 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2594 = stablehlo.reshape %v2590 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2595 = stablehlo.transpose %v2594, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2596 = stablehlo.reshape %v2595 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2597 = stablehlo.reshape %v2587 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2598 = stablehlo.reshape %v2596 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2599 = stablehlo.dot_general %v2597, %v2598, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2600 = stablehlo.reshape %v2599 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2601 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2602 = stablehlo.multiply %v2600, %v2601 : tensor<32x38809xf32>
    %v2603 = stablehlo.reshape %v2602 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2604 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2605 = stablehlo.exponential %v2603 : tensor<32x197x197xf32>
    %v2606 = stablehlo.reduce(%v2605 init: %v2604) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2607 = stablehlo.broadcast_in_dim %v2606, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2608 = stablehlo.divide %v2605, %v2607 : tensor<32x197x197xf32>
    %v2609 = stablehlo.reshape %v2608 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2610 = stablehlo.reshape %v2609 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2611 = stablehlo.reshape %v2593 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2612 = stablehlo.dot_general %v2610, %v2611, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2613 = stablehlo.reshape %v2612 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2614 = stablehlo.reshape %v2613 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2615 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2616 = stablehlo.pad %v2614, %v2615, low = [0, 0, 128], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v2617 = stablehlo.reshape %v2616 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2618 = stablehlo.add %v2584, %v2617 : tensor<32x75648xf32>
    %v2619 = stablehlo.reshape %v2507 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2620 = stablehlo.slice %v2619 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2621 = stablehlo.reshape %v2620 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2622 = stablehlo.reshape %v2512 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2623 = stablehlo.slice %v2622 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2624 = stablehlo.reshape %v2623 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2625 = stablehlo.reshape %v2517 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2626 = stablehlo.slice %v2625 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2627 = stablehlo.reshape %v2626 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2628 = stablehlo.reshape %v2624 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2629 = stablehlo.transpose %v2628, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2630 = stablehlo.reshape %v2629 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2631 = stablehlo.reshape %v2621 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2632 = stablehlo.reshape %v2630 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2633 = stablehlo.dot_general %v2631, %v2632, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2634 = stablehlo.reshape %v2633 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2635 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2636 = stablehlo.multiply %v2634, %v2635 : tensor<32x38809xf32>
    %v2637 = stablehlo.reshape %v2636 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2638 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2639 = stablehlo.exponential %v2637 : tensor<32x197x197xf32>
    %v2640 = stablehlo.reduce(%v2639 init: %v2638) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2641 = stablehlo.broadcast_in_dim %v2640, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2642 = stablehlo.divide %v2639, %v2641 : tensor<32x197x197xf32>
    %v2643 = stablehlo.reshape %v2642 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2644 = stablehlo.reshape %v2643 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2645 = stablehlo.reshape %v2627 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2646 = stablehlo.dot_general %v2644, %v2645, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2647 = stablehlo.reshape %v2646 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2648 = stablehlo.reshape %v2647 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2649 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2650 = stablehlo.pad %v2648, %v2649, low = [0, 0, 192], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v2651 = stablehlo.reshape %v2650 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2652 = stablehlo.add %v2618, %v2651 : tensor<32x75648xf32>
    %v2653 = stablehlo.reshape %v2507 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2654 = stablehlo.slice %v2653 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2655 = stablehlo.reshape %v2654 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2656 = stablehlo.reshape %v2512 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2657 = stablehlo.slice %v2656 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2658 = stablehlo.reshape %v2657 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2659 = stablehlo.reshape %v2517 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2660 = stablehlo.slice %v2659 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2661 = stablehlo.reshape %v2660 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2662 = stablehlo.reshape %v2658 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2663 = stablehlo.transpose %v2662, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2664 = stablehlo.reshape %v2663 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2665 = stablehlo.reshape %v2655 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2666 = stablehlo.reshape %v2664 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2667 = stablehlo.dot_general %v2665, %v2666, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2668 = stablehlo.reshape %v2667 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2669 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2670 = stablehlo.multiply %v2668, %v2669 : tensor<32x38809xf32>
    %v2671 = stablehlo.reshape %v2670 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2672 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2673 = stablehlo.exponential %v2671 : tensor<32x197x197xf32>
    %v2674 = stablehlo.reduce(%v2673 init: %v2672) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2675 = stablehlo.broadcast_in_dim %v2674, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2676 = stablehlo.divide %v2673, %v2675 : tensor<32x197x197xf32>
    %v2677 = stablehlo.reshape %v2676 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2678 = stablehlo.reshape %v2677 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2679 = stablehlo.reshape %v2661 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2680 = stablehlo.dot_general %v2678, %v2679, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2681 = stablehlo.reshape %v2680 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2682 = stablehlo.reshape %v2681 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2683 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2684 = stablehlo.pad %v2682, %v2683, low = [0, 0, 256], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v2685 = stablehlo.reshape %v2684 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2686 = stablehlo.add %v2652, %v2685 : tensor<32x75648xf32>
    %v2687 = stablehlo.reshape %v2507 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2688 = stablehlo.slice %v2687 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2689 = stablehlo.reshape %v2688 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2690 = stablehlo.reshape %v2512 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2691 = stablehlo.slice %v2690 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2692 = stablehlo.reshape %v2691 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2693 = stablehlo.reshape %v2517 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2694 = stablehlo.slice %v2693 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2695 = stablehlo.reshape %v2694 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2696 = stablehlo.reshape %v2692 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2697 = stablehlo.transpose %v2696, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2698 = stablehlo.reshape %v2697 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2699 = stablehlo.reshape %v2689 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2700 = stablehlo.reshape %v2698 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2701 = stablehlo.dot_general %v2699, %v2700, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2702 = stablehlo.reshape %v2701 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2703 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2704 = stablehlo.multiply %v2702, %v2703 : tensor<32x38809xf32>
    %v2705 = stablehlo.reshape %v2704 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2706 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2707 = stablehlo.exponential %v2705 : tensor<32x197x197xf32>
    %v2708 = stablehlo.reduce(%v2707 init: %v2706) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2709 = stablehlo.broadcast_in_dim %v2708, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2710 = stablehlo.divide %v2707, %v2709 : tensor<32x197x197xf32>
    %v2711 = stablehlo.reshape %v2710 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2712 = stablehlo.reshape %v2711 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2713 = stablehlo.reshape %v2695 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2714 = stablehlo.dot_general %v2712, %v2713, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2715 = stablehlo.reshape %v2714 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2716 = stablehlo.reshape %v2715 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2717 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2718 = stablehlo.pad %v2716, %v2717, low = [0, 0, 320], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v2719 = stablehlo.reshape %v2718 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2720 = stablehlo.add %v2686, %v2719 : tensor<32x75648xf32>
    %v2721 = stablehlo.reshape %v2720 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2722 = stablehlo.dot_general %v2721, %b8_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v2723 = stablehlo.broadcast_in_dim %b8_bo, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2724 = stablehlo.add %v2722, %v2723 : tensor<32x197x384xf32>
    %v2725 = stablehlo.reshape %v2724 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2726 = stablehlo.broadcast_in_dim %dp16, dims = [0] : (tensor<32xf32>) -> tensor<32x75648xf32>
    %v2727 = stablehlo.multiply %v2726, %v2725 : tensor<32x75648xf32>
    %v2728 = stablehlo.add %v2474, %v2727 : tensor<32x75648xf32>
    %v2729 = stablehlo.reshape %v2728 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2730 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2731 = stablehlo.constant dense<384.0> : tensor<32x197x384xf32>
    %v2732 = stablehlo.constant dense<1.0e-5> : tensor<32x197x384xf32>
    %v2733 = stablehlo.reduce(%v2729 init: %v2730) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2734 = stablehlo.broadcast_in_dim %v2733, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v2735 = stablehlo.divide %v2734, %v2731 : tensor<32x197x384xf32>
    %v2736 = stablehlo.subtract %v2729, %v2735 : tensor<32x197x384xf32>
    %v2737 = stablehlo.multiply %v2736, %v2736 : tensor<32x197x384xf32>
    %v2738 = stablehlo.reduce(%v2737 init: %v2730) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2739 = stablehlo.broadcast_in_dim %v2738, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v2740 = stablehlo.divide %v2739, %v2731 : tensor<32x197x384xf32>
    %v2741 = stablehlo.add %v2740, %v2732 : tensor<32x197x384xf32>
    %v2742 = stablehlo.rsqrt %v2741 : tensor<32x197x384xf32>
    %v2743 = stablehlo.multiply %v2736, %v2742 : tensor<32x197x384xf32>
    %v2744 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v2745 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v2746 = stablehlo.multiply %v2743, %v2744 : tensor<32x197x384xf32>
    %v2747 = stablehlo.add %v2746, %v2745 : tensor<32x197x384xf32>
    %v2748 = stablehlo.reshape %v2747 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2749 = stablehlo.reshape %v2748 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2750 = stablehlo.broadcast_in_dim %b8_g2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2751 = stablehlo.multiply %v2749, %v2750 : tensor<32x197x384xf32>
    %v2752 = stablehlo.reshape %v2751 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2753 = stablehlo.reshape %v2752 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2754 = stablehlo.broadcast_in_dim %b8_bt2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2755 = stablehlo.add %v2753, %v2754 : tensor<32x197x384xf32>
    %v2756 = stablehlo.reshape %v2755 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2757 = stablehlo.reshape %v2756 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2758 = stablehlo.dot_general %v2757, %b8_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x1536xf32>) -> tensor<32x197x1536xf32>
    %v2759 = stablehlo.broadcast_in_dim %b8_bfc1, dims = [2] : (tensor<1536xf32>) -> tensor<32x197x1536xf32>
    %v2760 = stablehlo.add %v2758, %v2759 : tensor<32x197x1536xf32>
    %v2761 = stablehlo.reshape %v2760 : (tensor<32x197x1536xf32>) -> tensor<32x302592xf32>
    %v2762 = stablehlo.multiply %v2761, %v2761 : tensor<32x302592xf32>
    %v2763 = stablehlo.multiply %v2762, %v2761 : tensor<32x302592xf32>
    %v2764 = stablehlo.constant dense<0.044715> : tensor<32x302592xf32>
    %v2765 = stablehlo.multiply %v2764, %v2763 : tensor<32x302592xf32>
    %v2766 = stablehlo.add %v2761, %v2765 : tensor<32x302592xf32>
    %v2767 = stablehlo.constant dense<0.7978845608028654> : tensor<32x302592xf32>
    %v2768 = stablehlo.multiply %v2767, %v2766 : tensor<32x302592xf32>
    %v2769 = stablehlo.tanh %v2768 : tensor<32x302592xf32>
    %v2770 = stablehlo.constant dense<1.0> : tensor<32x302592xf32>
    %v2771 = stablehlo.add %v2770, %v2769 : tensor<32x302592xf32>
    %v2772 = stablehlo.constant dense<0.5> : tensor<32x302592xf32>
    %v2773 = stablehlo.multiply %v2772, %v2761 : tensor<32x302592xf32>
    %v2774 = stablehlo.multiply %v2773, %v2771 : tensor<32x302592xf32>
    %v2775 = stablehlo.reshape %v2774 : (tensor<32x302592xf32>) -> tensor<32x197x1536xf32>
    %v2776 = stablehlo.dot_general %v2775, %b8_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x1536xf32>, tensor<1536x384xf32>) -> tensor<32x197x384xf32>
    %v2777 = stablehlo.broadcast_in_dim %b8_bfc2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2778 = stablehlo.add %v2776, %v2777 : tensor<32x197x384xf32>
    %v2779 = stablehlo.reshape %v2778 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2780 = stablehlo.broadcast_in_dim %dp17, dims = [0] : (tensor<32xf32>) -> tensor<32x75648xf32>
    %v2781 = stablehlo.multiply %v2780, %v2779 : tensor<32x75648xf32>
    %v2782 = stablehlo.add %v2728, %v2781 : tensor<32x75648xf32>
    %v2783 = stablehlo.reshape %v2782 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2784 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2785 = stablehlo.constant dense<384.0> : tensor<32x197x384xf32>
    %v2786 = stablehlo.constant dense<1.0e-5> : tensor<32x197x384xf32>
    %v2787 = stablehlo.reduce(%v2783 init: %v2784) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2788 = stablehlo.broadcast_in_dim %v2787, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v2789 = stablehlo.divide %v2788, %v2785 : tensor<32x197x384xf32>
    %v2790 = stablehlo.subtract %v2783, %v2789 : tensor<32x197x384xf32>
    %v2791 = stablehlo.multiply %v2790, %v2790 : tensor<32x197x384xf32>
    %v2792 = stablehlo.reduce(%v2791 init: %v2784) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2793 = stablehlo.broadcast_in_dim %v2792, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v2794 = stablehlo.divide %v2793, %v2785 : tensor<32x197x384xf32>
    %v2795 = stablehlo.add %v2794, %v2786 : tensor<32x197x384xf32>
    %v2796 = stablehlo.rsqrt %v2795 : tensor<32x197x384xf32>
    %v2797 = stablehlo.multiply %v2790, %v2796 : tensor<32x197x384xf32>
    %v2798 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v2799 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v2800 = stablehlo.multiply %v2797, %v2798 : tensor<32x197x384xf32>
    %v2801 = stablehlo.add %v2800, %v2799 : tensor<32x197x384xf32>
    %v2802 = stablehlo.reshape %v2801 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2803 = stablehlo.reshape %v2802 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2804 = stablehlo.broadcast_in_dim %b9_g1, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2805 = stablehlo.multiply %v2803, %v2804 : tensor<32x197x384xf32>
    %v2806 = stablehlo.reshape %v2805 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2807 = stablehlo.reshape %v2806 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2808 = stablehlo.broadcast_in_dim %b9_bt1, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2809 = stablehlo.add %v2807, %v2808 : tensor<32x197x384xf32>
    %v2810 = stablehlo.reshape %v2809 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2811 = stablehlo.reshape %v2810 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2812 = stablehlo.dot_general %v2811, %b9_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v2813 = stablehlo.broadcast_in_dim %b9_bq, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2814 = stablehlo.add %v2812, %v2813 : tensor<32x197x384xf32>
    %v2815 = stablehlo.reshape %v2814 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2816 = stablehlo.reshape %v2810 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2817 = stablehlo.dot_general %v2816, %b9_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v2818 = stablehlo.broadcast_in_dim %b9_bk, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2819 = stablehlo.add %v2817, %v2818 : tensor<32x197x384xf32>
    %v2820 = stablehlo.reshape %v2819 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2821 = stablehlo.reshape %v2810 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2822 = stablehlo.dot_general %v2821, %b9_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v2823 = stablehlo.broadcast_in_dim %b9_bv, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v2824 = stablehlo.add %v2822, %v2823 : tensor<32x197x384xf32>
    %v2825 = stablehlo.reshape %v2824 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2826 = stablehlo.reshape %v2815 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2827 = stablehlo.slice %v2826 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2828 = stablehlo.reshape %v2827 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2829 = stablehlo.reshape %v2820 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2830 = stablehlo.slice %v2829 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2831 = stablehlo.reshape %v2830 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2832 = stablehlo.reshape %v2825 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2833 = stablehlo.slice %v2832 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2834 = stablehlo.reshape %v2833 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2835 = stablehlo.reshape %v2831 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2836 = stablehlo.transpose %v2835, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2837 = stablehlo.reshape %v2836 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2838 = stablehlo.reshape %v2828 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2839 = stablehlo.reshape %v2837 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2840 = stablehlo.dot_general %v2838, %v2839, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2841 = stablehlo.reshape %v2840 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2842 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2843 = stablehlo.multiply %v2841, %v2842 : tensor<32x38809xf32>
    %v2844 = stablehlo.reshape %v2843 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2845 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2846 = stablehlo.exponential %v2844 : tensor<32x197x197xf32>
    %v2847 = stablehlo.reduce(%v2846 init: %v2845) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2848 = stablehlo.broadcast_in_dim %v2847, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2849 = stablehlo.divide %v2846, %v2848 : tensor<32x197x197xf32>
    %v2850 = stablehlo.reshape %v2849 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2851 = stablehlo.reshape %v2850 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2852 = stablehlo.reshape %v2834 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2853 = stablehlo.dot_general %v2851, %v2852, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2854 = stablehlo.reshape %v2853 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2855 = stablehlo.reshape %v2854 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2856 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2857 = stablehlo.pad %v2855, %v2856, low = [0, 0, 0], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v2858 = stablehlo.reshape %v2857 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2859 = stablehlo.reshape %v2815 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2860 = stablehlo.slice %v2859 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2861 = stablehlo.reshape %v2860 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2862 = stablehlo.reshape %v2820 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2863 = stablehlo.slice %v2862 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2864 = stablehlo.reshape %v2863 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2865 = stablehlo.reshape %v2825 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2866 = stablehlo.slice %v2865 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2867 = stablehlo.reshape %v2866 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2868 = stablehlo.reshape %v2864 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2869 = stablehlo.transpose %v2868, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2870 = stablehlo.reshape %v2869 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2871 = stablehlo.reshape %v2861 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2872 = stablehlo.reshape %v2870 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2873 = stablehlo.dot_general %v2871, %v2872, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2874 = stablehlo.reshape %v2873 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2875 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2876 = stablehlo.multiply %v2874, %v2875 : tensor<32x38809xf32>
    %v2877 = stablehlo.reshape %v2876 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2878 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2879 = stablehlo.exponential %v2877 : tensor<32x197x197xf32>
    %v2880 = stablehlo.reduce(%v2879 init: %v2878) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2881 = stablehlo.broadcast_in_dim %v2880, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2882 = stablehlo.divide %v2879, %v2881 : tensor<32x197x197xf32>
    %v2883 = stablehlo.reshape %v2882 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2884 = stablehlo.reshape %v2883 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2885 = stablehlo.reshape %v2867 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2886 = stablehlo.dot_general %v2884, %v2885, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2887 = stablehlo.reshape %v2886 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2888 = stablehlo.reshape %v2887 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2889 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2890 = stablehlo.pad %v2888, %v2889, low = [0, 0, 64], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v2891 = stablehlo.reshape %v2890 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2892 = stablehlo.add %v2858, %v2891 : tensor<32x75648xf32>
    %v2893 = stablehlo.reshape %v2815 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2894 = stablehlo.slice %v2893 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2895 = stablehlo.reshape %v2894 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2896 = stablehlo.reshape %v2820 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2897 = stablehlo.slice %v2896 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2898 = stablehlo.reshape %v2897 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2899 = stablehlo.reshape %v2825 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2900 = stablehlo.slice %v2899 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2901 = stablehlo.reshape %v2900 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2902 = stablehlo.reshape %v2898 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2903 = stablehlo.transpose %v2902, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2904 = stablehlo.reshape %v2903 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2905 = stablehlo.reshape %v2895 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2906 = stablehlo.reshape %v2904 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2907 = stablehlo.dot_general %v2905, %v2906, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2908 = stablehlo.reshape %v2907 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2909 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2910 = stablehlo.multiply %v2908, %v2909 : tensor<32x38809xf32>
    %v2911 = stablehlo.reshape %v2910 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2912 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2913 = stablehlo.exponential %v2911 : tensor<32x197x197xf32>
    %v2914 = stablehlo.reduce(%v2913 init: %v2912) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2915 = stablehlo.broadcast_in_dim %v2914, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2916 = stablehlo.divide %v2913, %v2915 : tensor<32x197x197xf32>
    %v2917 = stablehlo.reshape %v2916 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2918 = stablehlo.reshape %v2917 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2919 = stablehlo.reshape %v2901 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2920 = stablehlo.dot_general %v2918, %v2919, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2921 = stablehlo.reshape %v2920 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2922 = stablehlo.reshape %v2921 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2923 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2924 = stablehlo.pad %v2922, %v2923, low = [0, 0, 128], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v2925 = stablehlo.reshape %v2924 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2926 = stablehlo.add %v2892, %v2925 : tensor<32x75648xf32>
    %v2927 = stablehlo.reshape %v2815 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2928 = stablehlo.slice %v2927 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2929 = stablehlo.reshape %v2928 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2930 = stablehlo.reshape %v2820 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2931 = stablehlo.slice %v2930 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2932 = stablehlo.reshape %v2931 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2933 = stablehlo.reshape %v2825 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2934 = stablehlo.slice %v2933 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2935 = stablehlo.reshape %v2934 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2936 = stablehlo.reshape %v2932 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2937 = stablehlo.transpose %v2936, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2938 = stablehlo.reshape %v2937 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2939 = stablehlo.reshape %v2929 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2940 = stablehlo.reshape %v2938 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2941 = stablehlo.dot_general %v2939, %v2940, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2942 = stablehlo.reshape %v2941 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2943 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2944 = stablehlo.multiply %v2942, %v2943 : tensor<32x38809xf32>
    %v2945 = stablehlo.reshape %v2944 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2946 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2947 = stablehlo.exponential %v2945 : tensor<32x197x197xf32>
    %v2948 = stablehlo.reduce(%v2947 init: %v2946) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2949 = stablehlo.broadcast_in_dim %v2948, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2950 = stablehlo.divide %v2947, %v2949 : tensor<32x197x197xf32>
    %v2951 = stablehlo.reshape %v2950 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2952 = stablehlo.reshape %v2951 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2953 = stablehlo.reshape %v2935 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2954 = stablehlo.dot_general %v2952, %v2953, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2955 = stablehlo.reshape %v2954 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2956 = stablehlo.reshape %v2955 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2957 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2958 = stablehlo.pad %v2956, %v2957, low = [0, 0, 192], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v2959 = stablehlo.reshape %v2958 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2960 = stablehlo.add %v2926, %v2959 : tensor<32x75648xf32>
    %v2961 = stablehlo.reshape %v2815 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2962 = stablehlo.slice %v2961 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2963 = stablehlo.reshape %v2962 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2964 = stablehlo.reshape %v2820 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2965 = stablehlo.slice %v2964 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2966 = stablehlo.reshape %v2965 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2967 = stablehlo.reshape %v2825 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2968 = stablehlo.slice %v2967 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2969 = stablehlo.reshape %v2968 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2970 = stablehlo.reshape %v2966 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2971 = stablehlo.transpose %v2970, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2972 = stablehlo.reshape %v2971 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2973 = stablehlo.reshape %v2963 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2974 = stablehlo.reshape %v2972 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2975 = stablehlo.dot_general %v2973, %v2974, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2976 = stablehlo.reshape %v2975 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2977 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2978 = stablehlo.multiply %v2976, %v2977 : tensor<32x38809xf32>
    %v2979 = stablehlo.reshape %v2978 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2980 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2981 = stablehlo.exponential %v2979 : tensor<32x197x197xf32>
    %v2982 = stablehlo.reduce(%v2981 init: %v2980) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2983 = stablehlo.broadcast_in_dim %v2982, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2984 = stablehlo.divide %v2981, %v2983 : tensor<32x197x197xf32>
    %v2985 = stablehlo.reshape %v2984 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2986 = stablehlo.reshape %v2985 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2987 = stablehlo.reshape %v2969 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2988 = stablehlo.dot_general %v2986, %v2987, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2989 = stablehlo.reshape %v2988 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2990 = stablehlo.reshape %v2989 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2991 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2992 = stablehlo.pad %v2990, %v2991, low = [0, 0, 256], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v2993 = stablehlo.reshape %v2992 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v2994 = stablehlo.add %v2960, %v2993 : tensor<32x75648xf32>
    %v2995 = stablehlo.reshape %v2815 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2996 = stablehlo.slice %v2995 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v2997 = stablehlo.reshape %v2996 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2998 = stablehlo.reshape %v2820 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v2999 = stablehlo.slice %v2998 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3000 = stablehlo.reshape %v2999 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3001 = stablehlo.reshape %v2825 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3002 = stablehlo.slice %v3001 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3003 = stablehlo.reshape %v3002 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3004 = stablehlo.reshape %v3000 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3005 = stablehlo.transpose %v3004, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3006 = stablehlo.reshape %v3005 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3007 = stablehlo.reshape %v2997 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3008 = stablehlo.reshape %v3006 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3009 = stablehlo.dot_general %v3007, %v3008, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3010 = stablehlo.reshape %v3009 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3011 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3012 = stablehlo.multiply %v3010, %v3011 : tensor<32x38809xf32>
    %v3013 = stablehlo.reshape %v3012 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3014 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3015 = stablehlo.exponential %v3013 : tensor<32x197x197xf32>
    %v3016 = stablehlo.reduce(%v3015 init: %v3014) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3017 = stablehlo.broadcast_in_dim %v3016, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3018 = stablehlo.divide %v3015, %v3017 : tensor<32x197x197xf32>
    %v3019 = stablehlo.reshape %v3018 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3020 = stablehlo.reshape %v3019 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3021 = stablehlo.reshape %v3003 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3022 = stablehlo.dot_general %v3020, %v3021, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3023 = stablehlo.reshape %v3022 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3024 = stablehlo.reshape %v3023 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3025 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3026 = stablehlo.pad %v3024, %v3025, low = [0, 0, 320], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v3027 = stablehlo.reshape %v3026 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3028 = stablehlo.add %v2994, %v3027 : tensor<32x75648xf32>
    %v3029 = stablehlo.reshape %v3028 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3030 = stablehlo.dot_general %v3029, %b9_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v3031 = stablehlo.broadcast_in_dim %b9_bo, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v3032 = stablehlo.add %v3030, %v3031 : tensor<32x197x384xf32>
    %v3033 = stablehlo.reshape %v3032 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3034 = stablehlo.broadcast_in_dim %dp18, dims = [0] : (tensor<32xf32>) -> tensor<32x75648xf32>
    %v3035 = stablehlo.multiply %v3034, %v3033 : tensor<32x75648xf32>
    %v3036 = stablehlo.add %v2782, %v3035 : tensor<32x75648xf32>
    %v3037 = stablehlo.reshape %v3036 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3038 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3039 = stablehlo.constant dense<384.0> : tensor<32x197x384xf32>
    %v3040 = stablehlo.constant dense<1.0e-5> : tensor<32x197x384xf32>
    %v3041 = stablehlo.reduce(%v3037 init: %v3038) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3042 = stablehlo.broadcast_in_dim %v3041, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v3043 = stablehlo.divide %v3042, %v3039 : tensor<32x197x384xf32>
    %v3044 = stablehlo.subtract %v3037, %v3043 : tensor<32x197x384xf32>
    %v3045 = stablehlo.multiply %v3044, %v3044 : tensor<32x197x384xf32>
    %v3046 = stablehlo.reduce(%v3045 init: %v3038) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3047 = stablehlo.broadcast_in_dim %v3046, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v3048 = stablehlo.divide %v3047, %v3039 : tensor<32x197x384xf32>
    %v3049 = stablehlo.add %v3048, %v3040 : tensor<32x197x384xf32>
    %v3050 = stablehlo.rsqrt %v3049 : tensor<32x197x384xf32>
    %v3051 = stablehlo.multiply %v3044, %v3050 : tensor<32x197x384xf32>
    %v3052 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v3053 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v3054 = stablehlo.multiply %v3051, %v3052 : tensor<32x197x384xf32>
    %v3055 = stablehlo.add %v3054, %v3053 : tensor<32x197x384xf32>
    %v3056 = stablehlo.reshape %v3055 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3057 = stablehlo.reshape %v3056 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3058 = stablehlo.broadcast_in_dim %b9_g2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v3059 = stablehlo.multiply %v3057, %v3058 : tensor<32x197x384xf32>
    %v3060 = stablehlo.reshape %v3059 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3061 = stablehlo.reshape %v3060 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3062 = stablehlo.broadcast_in_dim %b9_bt2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v3063 = stablehlo.add %v3061, %v3062 : tensor<32x197x384xf32>
    %v3064 = stablehlo.reshape %v3063 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3065 = stablehlo.reshape %v3064 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3066 = stablehlo.dot_general %v3065, %b9_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x1536xf32>) -> tensor<32x197x1536xf32>
    %v3067 = stablehlo.broadcast_in_dim %b9_bfc1, dims = [2] : (tensor<1536xf32>) -> tensor<32x197x1536xf32>
    %v3068 = stablehlo.add %v3066, %v3067 : tensor<32x197x1536xf32>
    %v3069 = stablehlo.reshape %v3068 : (tensor<32x197x1536xf32>) -> tensor<32x302592xf32>
    %v3070 = stablehlo.multiply %v3069, %v3069 : tensor<32x302592xf32>
    %v3071 = stablehlo.multiply %v3070, %v3069 : tensor<32x302592xf32>
    %v3072 = stablehlo.constant dense<0.044715> : tensor<32x302592xf32>
    %v3073 = stablehlo.multiply %v3072, %v3071 : tensor<32x302592xf32>
    %v3074 = stablehlo.add %v3069, %v3073 : tensor<32x302592xf32>
    %v3075 = stablehlo.constant dense<0.7978845608028654> : tensor<32x302592xf32>
    %v3076 = stablehlo.multiply %v3075, %v3074 : tensor<32x302592xf32>
    %v3077 = stablehlo.tanh %v3076 : tensor<32x302592xf32>
    %v3078 = stablehlo.constant dense<1.0> : tensor<32x302592xf32>
    %v3079 = stablehlo.add %v3078, %v3077 : tensor<32x302592xf32>
    %v3080 = stablehlo.constant dense<0.5> : tensor<32x302592xf32>
    %v3081 = stablehlo.multiply %v3080, %v3069 : tensor<32x302592xf32>
    %v3082 = stablehlo.multiply %v3081, %v3079 : tensor<32x302592xf32>
    %v3083 = stablehlo.reshape %v3082 : (tensor<32x302592xf32>) -> tensor<32x197x1536xf32>
    %v3084 = stablehlo.dot_general %v3083, %b9_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x1536xf32>, tensor<1536x384xf32>) -> tensor<32x197x384xf32>
    %v3085 = stablehlo.broadcast_in_dim %b9_bfc2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v3086 = stablehlo.add %v3084, %v3085 : tensor<32x197x384xf32>
    %v3087 = stablehlo.reshape %v3086 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3088 = stablehlo.broadcast_in_dim %dp19, dims = [0] : (tensor<32xf32>) -> tensor<32x75648xf32>
    %v3089 = stablehlo.multiply %v3088, %v3087 : tensor<32x75648xf32>
    %v3090 = stablehlo.add %v3036, %v3089 : tensor<32x75648xf32>
    %v3091 = stablehlo.reshape %v3090 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3092 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3093 = stablehlo.constant dense<384.0> : tensor<32x197x384xf32>
    %v3094 = stablehlo.constant dense<1.0e-5> : tensor<32x197x384xf32>
    %v3095 = stablehlo.reduce(%v3091 init: %v3092) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3096 = stablehlo.broadcast_in_dim %v3095, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v3097 = stablehlo.divide %v3096, %v3093 : tensor<32x197x384xf32>
    %v3098 = stablehlo.subtract %v3091, %v3097 : tensor<32x197x384xf32>
    %v3099 = stablehlo.multiply %v3098, %v3098 : tensor<32x197x384xf32>
    %v3100 = stablehlo.reduce(%v3099 init: %v3092) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3101 = stablehlo.broadcast_in_dim %v3100, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v3102 = stablehlo.divide %v3101, %v3093 : tensor<32x197x384xf32>
    %v3103 = stablehlo.add %v3102, %v3094 : tensor<32x197x384xf32>
    %v3104 = stablehlo.rsqrt %v3103 : tensor<32x197x384xf32>
    %v3105 = stablehlo.multiply %v3098, %v3104 : tensor<32x197x384xf32>
    %v3106 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v3107 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v3108 = stablehlo.multiply %v3105, %v3106 : tensor<32x197x384xf32>
    %v3109 = stablehlo.add %v3108, %v3107 : tensor<32x197x384xf32>
    %v3110 = stablehlo.reshape %v3109 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3111 = stablehlo.reshape %v3110 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3112 = stablehlo.broadcast_in_dim %b10_g1, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v3113 = stablehlo.multiply %v3111, %v3112 : tensor<32x197x384xf32>
    %v3114 = stablehlo.reshape %v3113 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3115 = stablehlo.reshape %v3114 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3116 = stablehlo.broadcast_in_dim %b10_bt1, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v3117 = stablehlo.add %v3115, %v3116 : tensor<32x197x384xf32>
    %v3118 = stablehlo.reshape %v3117 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3119 = stablehlo.reshape %v3118 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3120 = stablehlo.dot_general %v3119, %b10_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v3121 = stablehlo.broadcast_in_dim %b10_bq, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v3122 = stablehlo.add %v3120, %v3121 : tensor<32x197x384xf32>
    %v3123 = stablehlo.reshape %v3122 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3124 = stablehlo.reshape %v3118 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3125 = stablehlo.dot_general %v3124, %b10_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v3126 = stablehlo.broadcast_in_dim %b10_bk, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v3127 = stablehlo.add %v3125, %v3126 : tensor<32x197x384xf32>
    %v3128 = stablehlo.reshape %v3127 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3129 = stablehlo.reshape %v3118 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3130 = stablehlo.dot_general %v3129, %b10_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v3131 = stablehlo.broadcast_in_dim %b10_bv, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v3132 = stablehlo.add %v3130, %v3131 : tensor<32x197x384xf32>
    %v3133 = stablehlo.reshape %v3132 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3134 = stablehlo.reshape %v3123 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3135 = stablehlo.slice %v3134 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3136 = stablehlo.reshape %v3135 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3137 = stablehlo.reshape %v3128 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3138 = stablehlo.slice %v3137 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3139 = stablehlo.reshape %v3138 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3140 = stablehlo.reshape %v3133 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3141 = stablehlo.slice %v3140 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3142 = stablehlo.reshape %v3141 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3143 = stablehlo.reshape %v3139 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3144 = stablehlo.transpose %v3143, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3145 = stablehlo.reshape %v3144 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3146 = stablehlo.reshape %v3136 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3147 = stablehlo.reshape %v3145 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3148 = stablehlo.dot_general %v3146, %v3147, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3149 = stablehlo.reshape %v3148 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3150 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3151 = stablehlo.multiply %v3149, %v3150 : tensor<32x38809xf32>
    %v3152 = stablehlo.reshape %v3151 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3153 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3154 = stablehlo.exponential %v3152 : tensor<32x197x197xf32>
    %v3155 = stablehlo.reduce(%v3154 init: %v3153) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3156 = stablehlo.broadcast_in_dim %v3155, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3157 = stablehlo.divide %v3154, %v3156 : tensor<32x197x197xf32>
    %v3158 = stablehlo.reshape %v3157 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3159 = stablehlo.reshape %v3158 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3160 = stablehlo.reshape %v3142 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3161 = stablehlo.dot_general %v3159, %v3160, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3162 = stablehlo.reshape %v3161 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3163 = stablehlo.reshape %v3162 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3164 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3165 = stablehlo.pad %v3163, %v3164, low = [0, 0, 0], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v3166 = stablehlo.reshape %v3165 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3167 = stablehlo.reshape %v3123 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3168 = stablehlo.slice %v3167 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3169 = stablehlo.reshape %v3168 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3170 = stablehlo.reshape %v3128 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3171 = stablehlo.slice %v3170 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3172 = stablehlo.reshape %v3171 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3173 = stablehlo.reshape %v3133 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3174 = stablehlo.slice %v3173 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
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
    %v3198 = stablehlo.pad %v3196, %v3197, low = [0, 0, 64], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v3199 = stablehlo.reshape %v3198 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3200 = stablehlo.add %v3166, %v3199 : tensor<32x75648xf32>
    %v3201 = stablehlo.reshape %v3123 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3202 = stablehlo.slice %v3201 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3203 = stablehlo.reshape %v3202 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3204 = stablehlo.reshape %v3128 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3205 = stablehlo.slice %v3204 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3206 = stablehlo.reshape %v3205 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3207 = stablehlo.reshape %v3133 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3208 = stablehlo.slice %v3207 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
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
    %v3232 = stablehlo.pad %v3230, %v3231, low = [0, 0, 128], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v3233 = stablehlo.reshape %v3232 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3234 = stablehlo.add %v3200, %v3233 : tensor<32x75648xf32>
    %v3235 = stablehlo.reshape %v3123 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3236 = stablehlo.slice %v3235 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3237 = stablehlo.reshape %v3236 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3238 = stablehlo.reshape %v3128 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3239 = stablehlo.slice %v3238 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3240 = stablehlo.reshape %v3239 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3241 = stablehlo.reshape %v3133 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3242 = stablehlo.slice %v3241 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
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
    %v3266 = stablehlo.pad %v3264, %v3265, low = [0, 0, 192], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v3267 = stablehlo.reshape %v3266 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3268 = stablehlo.add %v3234, %v3267 : tensor<32x75648xf32>
    %v3269 = stablehlo.reshape %v3123 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3270 = stablehlo.slice %v3269 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3271 = stablehlo.reshape %v3270 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3272 = stablehlo.reshape %v3128 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3273 = stablehlo.slice %v3272 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3274 = stablehlo.reshape %v3273 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3275 = stablehlo.reshape %v3133 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3276 = stablehlo.slice %v3275 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
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
    %v3300 = stablehlo.pad %v3298, %v3299, low = [0, 0, 256], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v3301 = stablehlo.reshape %v3300 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3302 = stablehlo.add %v3268, %v3301 : tensor<32x75648xf32>
    %v3303 = stablehlo.reshape %v3123 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3304 = stablehlo.slice %v3303 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3305 = stablehlo.reshape %v3304 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3306 = stablehlo.reshape %v3128 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3307 = stablehlo.slice %v3306 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3308 = stablehlo.reshape %v3307 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3309 = stablehlo.reshape %v3133 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3310 = stablehlo.slice %v3309 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
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
    %v3334 = stablehlo.pad %v3332, %v3333, low = [0, 0, 320], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v3335 = stablehlo.reshape %v3334 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3336 = stablehlo.add %v3302, %v3335 : tensor<32x75648xf32>
    %v3337 = stablehlo.reshape %v3336 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3338 = stablehlo.dot_general %v3337, %b10_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v3339 = stablehlo.broadcast_in_dim %b10_bo, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v3340 = stablehlo.add %v3338, %v3339 : tensor<32x197x384xf32>
    %v3341 = stablehlo.reshape %v3340 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3342 = stablehlo.broadcast_in_dim %dp20, dims = [0] : (tensor<32xf32>) -> tensor<32x75648xf32>
    %v3343 = stablehlo.multiply %v3342, %v3341 : tensor<32x75648xf32>
    %v3344 = stablehlo.add %v3090, %v3343 : tensor<32x75648xf32>
    %v3345 = stablehlo.reshape %v3344 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3346 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3347 = stablehlo.constant dense<384.0> : tensor<32x197x384xf32>
    %v3348 = stablehlo.constant dense<1.0e-5> : tensor<32x197x384xf32>
    %v3349 = stablehlo.reduce(%v3345 init: %v3346) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3350 = stablehlo.broadcast_in_dim %v3349, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v3351 = stablehlo.divide %v3350, %v3347 : tensor<32x197x384xf32>
    %v3352 = stablehlo.subtract %v3345, %v3351 : tensor<32x197x384xf32>
    %v3353 = stablehlo.multiply %v3352, %v3352 : tensor<32x197x384xf32>
    %v3354 = stablehlo.reduce(%v3353 init: %v3346) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3355 = stablehlo.broadcast_in_dim %v3354, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v3356 = stablehlo.divide %v3355, %v3347 : tensor<32x197x384xf32>
    %v3357 = stablehlo.add %v3356, %v3348 : tensor<32x197x384xf32>
    %v3358 = stablehlo.rsqrt %v3357 : tensor<32x197x384xf32>
    %v3359 = stablehlo.multiply %v3352, %v3358 : tensor<32x197x384xf32>
    %v3360 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v3361 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v3362 = stablehlo.multiply %v3359, %v3360 : tensor<32x197x384xf32>
    %v3363 = stablehlo.add %v3362, %v3361 : tensor<32x197x384xf32>
    %v3364 = stablehlo.reshape %v3363 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3365 = stablehlo.reshape %v3364 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3366 = stablehlo.broadcast_in_dim %b10_g2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v3367 = stablehlo.multiply %v3365, %v3366 : tensor<32x197x384xf32>
    %v3368 = stablehlo.reshape %v3367 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3369 = stablehlo.reshape %v3368 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3370 = stablehlo.broadcast_in_dim %b10_bt2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v3371 = stablehlo.add %v3369, %v3370 : tensor<32x197x384xf32>
    %v3372 = stablehlo.reshape %v3371 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3373 = stablehlo.reshape %v3372 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3374 = stablehlo.dot_general %v3373, %b10_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x1536xf32>) -> tensor<32x197x1536xf32>
    %v3375 = stablehlo.broadcast_in_dim %b10_bfc1, dims = [2] : (tensor<1536xf32>) -> tensor<32x197x1536xf32>
    %v3376 = stablehlo.add %v3374, %v3375 : tensor<32x197x1536xf32>
    %v3377 = stablehlo.reshape %v3376 : (tensor<32x197x1536xf32>) -> tensor<32x302592xf32>
    %v3378 = stablehlo.multiply %v3377, %v3377 : tensor<32x302592xf32>
    %v3379 = stablehlo.multiply %v3378, %v3377 : tensor<32x302592xf32>
    %v3380 = stablehlo.constant dense<0.044715> : tensor<32x302592xf32>
    %v3381 = stablehlo.multiply %v3380, %v3379 : tensor<32x302592xf32>
    %v3382 = stablehlo.add %v3377, %v3381 : tensor<32x302592xf32>
    %v3383 = stablehlo.constant dense<0.7978845608028654> : tensor<32x302592xf32>
    %v3384 = stablehlo.multiply %v3383, %v3382 : tensor<32x302592xf32>
    %v3385 = stablehlo.tanh %v3384 : tensor<32x302592xf32>
    %v3386 = stablehlo.constant dense<1.0> : tensor<32x302592xf32>
    %v3387 = stablehlo.add %v3386, %v3385 : tensor<32x302592xf32>
    %v3388 = stablehlo.constant dense<0.5> : tensor<32x302592xf32>
    %v3389 = stablehlo.multiply %v3388, %v3377 : tensor<32x302592xf32>
    %v3390 = stablehlo.multiply %v3389, %v3387 : tensor<32x302592xf32>
    %v3391 = stablehlo.reshape %v3390 : (tensor<32x302592xf32>) -> tensor<32x197x1536xf32>
    %v3392 = stablehlo.dot_general %v3391, %b10_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x1536xf32>, tensor<1536x384xf32>) -> tensor<32x197x384xf32>
    %v3393 = stablehlo.broadcast_in_dim %b10_bfc2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v3394 = stablehlo.add %v3392, %v3393 : tensor<32x197x384xf32>
    %v3395 = stablehlo.reshape %v3394 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3396 = stablehlo.broadcast_in_dim %dp21, dims = [0] : (tensor<32xf32>) -> tensor<32x75648xf32>
    %v3397 = stablehlo.multiply %v3396, %v3395 : tensor<32x75648xf32>
    %v3398 = stablehlo.add %v3344, %v3397 : tensor<32x75648xf32>
    %v3399 = stablehlo.reshape %v3398 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3400 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3401 = stablehlo.constant dense<384.0> : tensor<32x197x384xf32>
    %v3402 = stablehlo.constant dense<1.0e-5> : tensor<32x197x384xf32>
    %v3403 = stablehlo.reduce(%v3399 init: %v3400) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3404 = stablehlo.broadcast_in_dim %v3403, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v3405 = stablehlo.divide %v3404, %v3401 : tensor<32x197x384xf32>
    %v3406 = stablehlo.subtract %v3399, %v3405 : tensor<32x197x384xf32>
    %v3407 = stablehlo.multiply %v3406, %v3406 : tensor<32x197x384xf32>
    %v3408 = stablehlo.reduce(%v3407 init: %v3400) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3409 = stablehlo.broadcast_in_dim %v3408, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v3410 = stablehlo.divide %v3409, %v3401 : tensor<32x197x384xf32>
    %v3411 = stablehlo.add %v3410, %v3402 : tensor<32x197x384xf32>
    %v3412 = stablehlo.rsqrt %v3411 : tensor<32x197x384xf32>
    %v3413 = stablehlo.multiply %v3406, %v3412 : tensor<32x197x384xf32>
    %v3414 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v3415 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v3416 = stablehlo.multiply %v3413, %v3414 : tensor<32x197x384xf32>
    %v3417 = stablehlo.add %v3416, %v3415 : tensor<32x197x384xf32>
    %v3418 = stablehlo.reshape %v3417 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3419 = stablehlo.reshape %v3418 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3420 = stablehlo.broadcast_in_dim %b11_g1, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v3421 = stablehlo.multiply %v3419, %v3420 : tensor<32x197x384xf32>
    %v3422 = stablehlo.reshape %v3421 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3423 = stablehlo.reshape %v3422 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3424 = stablehlo.broadcast_in_dim %b11_bt1, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v3425 = stablehlo.add %v3423, %v3424 : tensor<32x197x384xf32>
    %v3426 = stablehlo.reshape %v3425 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3427 = stablehlo.reshape %v3426 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3428 = stablehlo.dot_general %v3427, %b11_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v3429 = stablehlo.broadcast_in_dim %b11_bq, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v3430 = stablehlo.add %v3428, %v3429 : tensor<32x197x384xf32>
    %v3431 = stablehlo.reshape %v3430 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3432 = stablehlo.reshape %v3426 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3433 = stablehlo.dot_general %v3432, %b11_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v3434 = stablehlo.broadcast_in_dim %b11_bk, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v3435 = stablehlo.add %v3433, %v3434 : tensor<32x197x384xf32>
    %v3436 = stablehlo.reshape %v3435 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3437 = stablehlo.reshape %v3426 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3438 = stablehlo.dot_general %v3437, %b11_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v3439 = stablehlo.broadcast_in_dim %b11_bv, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v3440 = stablehlo.add %v3438, %v3439 : tensor<32x197x384xf32>
    %v3441 = stablehlo.reshape %v3440 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3442 = stablehlo.reshape %v3431 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3443 = stablehlo.slice %v3442 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3444 = stablehlo.reshape %v3443 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3445 = stablehlo.reshape %v3436 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3446 = stablehlo.slice %v3445 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3447 = stablehlo.reshape %v3446 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3448 = stablehlo.reshape %v3441 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3449 = stablehlo.slice %v3448 [0:32, 0:197, 0:64] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3450 = stablehlo.reshape %v3449 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3451 = stablehlo.reshape %v3447 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3452 = stablehlo.transpose %v3451, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3453 = stablehlo.reshape %v3452 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3454 = stablehlo.reshape %v3444 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3455 = stablehlo.reshape %v3453 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3456 = stablehlo.dot_general %v3454, %v3455, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3457 = stablehlo.reshape %v3456 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3458 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3459 = stablehlo.multiply %v3457, %v3458 : tensor<32x38809xf32>
    %v3460 = stablehlo.reshape %v3459 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3461 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3462 = stablehlo.exponential %v3460 : tensor<32x197x197xf32>
    %v3463 = stablehlo.reduce(%v3462 init: %v3461) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3464 = stablehlo.broadcast_in_dim %v3463, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3465 = stablehlo.divide %v3462, %v3464 : tensor<32x197x197xf32>
    %v3466 = stablehlo.reshape %v3465 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3467 = stablehlo.reshape %v3466 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3468 = stablehlo.reshape %v3450 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3469 = stablehlo.dot_general %v3467, %v3468, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3470 = stablehlo.reshape %v3469 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3471 = stablehlo.reshape %v3470 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3472 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3473 = stablehlo.pad %v3471, %v3472, low = [0, 0, 0], high = [0, 0, 320], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v3474 = stablehlo.reshape %v3473 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3475 = stablehlo.reshape %v3431 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3476 = stablehlo.slice %v3475 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3477 = stablehlo.reshape %v3476 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3478 = stablehlo.reshape %v3436 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3479 = stablehlo.slice %v3478 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3480 = stablehlo.reshape %v3479 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3481 = stablehlo.reshape %v3441 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3482 = stablehlo.slice %v3481 [0:32, 0:197, 64:128] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
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
    %v3506 = stablehlo.pad %v3504, %v3505, low = [0, 0, 64], high = [0, 0, 256], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v3507 = stablehlo.reshape %v3506 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3508 = stablehlo.add %v3474, %v3507 : tensor<32x75648xf32>
    %v3509 = stablehlo.reshape %v3431 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3510 = stablehlo.slice %v3509 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3511 = stablehlo.reshape %v3510 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3512 = stablehlo.reshape %v3436 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3513 = stablehlo.slice %v3512 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3514 = stablehlo.reshape %v3513 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3515 = stablehlo.reshape %v3441 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3516 = stablehlo.slice %v3515 [0:32, 0:197, 128:192] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3517 = stablehlo.reshape %v3516 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3518 = stablehlo.reshape %v3514 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3519 = stablehlo.transpose %v3518, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3520 = stablehlo.reshape %v3519 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3521 = stablehlo.reshape %v3511 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3522 = stablehlo.reshape %v3520 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3523 = stablehlo.dot_general %v3521, %v3522, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3524 = stablehlo.reshape %v3523 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3525 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3526 = stablehlo.multiply %v3524, %v3525 : tensor<32x38809xf32>
    %v3527 = stablehlo.reshape %v3526 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3528 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3529 = stablehlo.exponential %v3527 : tensor<32x197x197xf32>
    %v3530 = stablehlo.reduce(%v3529 init: %v3528) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3531 = stablehlo.broadcast_in_dim %v3530, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3532 = stablehlo.divide %v3529, %v3531 : tensor<32x197x197xf32>
    %v3533 = stablehlo.reshape %v3532 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3534 = stablehlo.reshape %v3533 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3535 = stablehlo.reshape %v3517 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3536 = stablehlo.dot_general %v3534, %v3535, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3537 = stablehlo.reshape %v3536 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3538 = stablehlo.reshape %v3537 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3539 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3540 = stablehlo.pad %v3538, %v3539, low = [0, 0, 128], high = [0, 0, 192], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v3541 = stablehlo.reshape %v3540 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3542 = stablehlo.add %v3508, %v3541 : tensor<32x75648xf32>
    %v3543 = stablehlo.reshape %v3431 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3544 = stablehlo.slice %v3543 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3545 = stablehlo.reshape %v3544 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3546 = stablehlo.reshape %v3436 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3547 = stablehlo.slice %v3546 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3548 = stablehlo.reshape %v3547 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3549 = stablehlo.reshape %v3441 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3550 = stablehlo.slice %v3549 [0:32, 0:197, 192:256] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
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
    %v3562 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3563 = stablehlo.exponential %v3561 : tensor<32x197x197xf32>
    %v3564 = stablehlo.reduce(%v3563 init: %v3562) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3565 = stablehlo.broadcast_in_dim %v3564, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3566 = stablehlo.divide %v3563, %v3565 : tensor<32x197x197xf32>
    %v3567 = stablehlo.reshape %v3566 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3568 = stablehlo.reshape %v3567 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3569 = stablehlo.reshape %v3551 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3570 = stablehlo.dot_general %v3568, %v3569, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3571 = stablehlo.reshape %v3570 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3572 = stablehlo.reshape %v3571 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3573 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3574 = stablehlo.pad %v3572, %v3573, low = [0, 0, 192], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v3575 = stablehlo.reshape %v3574 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3576 = stablehlo.add %v3542, %v3575 : tensor<32x75648xf32>
    %v3577 = stablehlo.reshape %v3431 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3578 = stablehlo.slice %v3577 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3579 = stablehlo.reshape %v3578 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3580 = stablehlo.reshape %v3436 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3581 = stablehlo.slice %v3580 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3582 = stablehlo.reshape %v3581 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3583 = stablehlo.reshape %v3441 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3584 = stablehlo.slice %v3583 [0:32, 0:197, 256:320] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3585 = stablehlo.reshape %v3584 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3586 = stablehlo.reshape %v3582 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3587 = stablehlo.transpose %v3586, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3588 = stablehlo.reshape %v3587 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3589 = stablehlo.reshape %v3579 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3590 = stablehlo.reshape %v3588 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3591 = stablehlo.dot_general %v3589, %v3590, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3592 = stablehlo.reshape %v3591 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3593 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3594 = stablehlo.multiply %v3592, %v3593 : tensor<32x38809xf32>
    %v3595 = stablehlo.reshape %v3594 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3596 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3597 = stablehlo.exponential %v3595 : tensor<32x197x197xf32>
    %v3598 = stablehlo.reduce(%v3597 init: %v3596) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3599 = stablehlo.broadcast_in_dim %v3598, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3600 = stablehlo.divide %v3597, %v3599 : tensor<32x197x197xf32>
    %v3601 = stablehlo.reshape %v3600 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3602 = stablehlo.reshape %v3601 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3603 = stablehlo.reshape %v3585 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3604 = stablehlo.dot_general %v3602, %v3603, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3605 = stablehlo.reshape %v3604 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3606 = stablehlo.reshape %v3605 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3607 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3608 = stablehlo.pad %v3606, %v3607, low = [0, 0, 256], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v3609 = stablehlo.reshape %v3608 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3610 = stablehlo.add %v3576, %v3609 : tensor<32x75648xf32>
    %v3611 = stablehlo.reshape %v3431 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3612 = stablehlo.slice %v3611 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3613 = stablehlo.reshape %v3612 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3614 = stablehlo.reshape %v3436 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3615 = stablehlo.slice %v3614 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3616 = stablehlo.reshape %v3615 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3617 = stablehlo.reshape %v3441 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3618 = stablehlo.slice %v3617 [0:32, 0:197, 320:384] : (tensor<32x197x384xf32>) -> tensor<32x197x64xf32>
    %v3619 = stablehlo.reshape %v3618 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3620 = stablehlo.reshape %v3616 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3621 = stablehlo.transpose %v3620, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v3622 = stablehlo.reshape %v3621 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v3623 = stablehlo.reshape %v3613 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3624 = stablehlo.reshape %v3622 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v3625 = stablehlo.dot_general %v3623, %v3624, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v3626 = stablehlo.reshape %v3625 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3627 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v3628 = stablehlo.multiply %v3626, %v3627 : tensor<32x38809xf32>
    %v3629 = stablehlo.reshape %v3628 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3630 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3631 = stablehlo.exponential %v3629 : tensor<32x197x197xf32>
    %v3632 = stablehlo.reduce(%v3631 init: %v3630) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3633 = stablehlo.broadcast_in_dim %v3632, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v3634 = stablehlo.divide %v3631, %v3633 : tensor<32x197x197xf32>
    %v3635 = stablehlo.reshape %v3634 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v3636 = stablehlo.reshape %v3635 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v3637 = stablehlo.reshape %v3619 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3638 = stablehlo.dot_general %v3636, %v3637, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v3639 = stablehlo.reshape %v3638 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v3640 = stablehlo.reshape %v3639 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v3641 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3642 = stablehlo.pad %v3640, %v3641, low = [0, 0, 320], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x384xf32>
    %v3643 = stablehlo.reshape %v3642 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3644 = stablehlo.add %v3610, %v3643 : tensor<32x75648xf32>
    %v3645 = stablehlo.reshape %v3644 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3646 = stablehlo.dot_general %v3645, %b11_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x384xf32>) -> tensor<32x197x384xf32>
    %v3647 = stablehlo.broadcast_in_dim %b11_bo, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v3648 = stablehlo.add %v3646, %v3647 : tensor<32x197x384xf32>
    %v3649 = stablehlo.reshape %v3648 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3650 = stablehlo.broadcast_in_dim %dp22, dims = [0] : (tensor<32xf32>) -> tensor<32x75648xf32>
    %v3651 = stablehlo.multiply %v3650, %v3649 : tensor<32x75648xf32>
    %v3652 = stablehlo.add %v3398, %v3651 : tensor<32x75648xf32>
    %v3653 = stablehlo.reshape %v3652 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3654 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3655 = stablehlo.constant dense<384.0> : tensor<32x197x384xf32>
    %v3656 = stablehlo.constant dense<1.0e-5> : tensor<32x197x384xf32>
    %v3657 = stablehlo.reduce(%v3653 init: %v3654) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3658 = stablehlo.broadcast_in_dim %v3657, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v3659 = stablehlo.divide %v3658, %v3655 : tensor<32x197x384xf32>
    %v3660 = stablehlo.subtract %v3653, %v3659 : tensor<32x197x384xf32>
    %v3661 = stablehlo.multiply %v3660, %v3660 : tensor<32x197x384xf32>
    %v3662 = stablehlo.reduce(%v3661 init: %v3654) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3663 = stablehlo.broadcast_in_dim %v3662, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v3664 = stablehlo.divide %v3663, %v3655 : tensor<32x197x384xf32>
    %v3665 = stablehlo.add %v3664, %v3656 : tensor<32x197x384xf32>
    %v3666 = stablehlo.rsqrt %v3665 : tensor<32x197x384xf32>
    %v3667 = stablehlo.multiply %v3660, %v3666 : tensor<32x197x384xf32>
    %v3668 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v3669 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v3670 = stablehlo.multiply %v3667, %v3668 : tensor<32x197x384xf32>
    %v3671 = stablehlo.add %v3670, %v3669 : tensor<32x197x384xf32>
    %v3672 = stablehlo.reshape %v3671 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3673 = stablehlo.reshape %v3672 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3674 = stablehlo.broadcast_in_dim %b11_g2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v3675 = stablehlo.multiply %v3673, %v3674 : tensor<32x197x384xf32>
    %v3676 = stablehlo.reshape %v3675 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3677 = stablehlo.reshape %v3676 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3678 = stablehlo.broadcast_in_dim %b11_bt2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v3679 = stablehlo.add %v3677, %v3678 : tensor<32x197x384xf32>
    %v3680 = stablehlo.reshape %v3679 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3681 = stablehlo.reshape %v3680 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3682 = stablehlo.dot_general %v3681, %b11_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x384xf32>, tensor<384x1536xf32>) -> tensor<32x197x1536xf32>
    %v3683 = stablehlo.broadcast_in_dim %b11_bfc1, dims = [2] : (tensor<1536xf32>) -> tensor<32x197x1536xf32>
    %v3684 = stablehlo.add %v3682, %v3683 : tensor<32x197x1536xf32>
    %v3685 = stablehlo.reshape %v3684 : (tensor<32x197x1536xf32>) -> tensor<32x302592xf32>
    %v3686 = stablehlo.multiply %v3685, %v3685 : tensor<32x302592xf32>
    %v3687 = stablehlo.multiply %v3686, %v3685 : tensor<32x302592xf32>
    %v3688 = stablehlo.constant dense<0.044715> : tensor<32x302592xf32>
    %v3689 = stablehlo.multiply %v3688, %v3687 : tensor<32x302592xf32>
    %v3690 = stablehlo.add %v3685, %v3689 : tensor<32x302592xf32>
    %v3691 = stablehlo.constant dense<0.7978845608028654> : tensor<32x302592xf32>
    %v3692 = stablehlo.multiply %v3691, %v3690 : tensor<32x302592xf32>
    %v3693 = stablehlo.tanh %v3692 : tensor<32x302592xf32>
    %v3694 = stablehlo.constant dense<1.0> : tensor<32x302592xf32>
    %v3695 = stablehlo.add %v3694, %v3693 : tensor<32x302592xf32>
    %v3696 = stablehlo.constant dense<0.5> : tensor<32x302592xf32>
    %v3697 = stablehlo.multiply %v3696, %v3685 : tensor<32x302592xf32>
    %v3698 = stablehlo.multiply %v3697, %v3695 : tensor<32x302592xf32>
    %v3699 = stablehlo.reshape %v3698 : (tensor<32x302592xf32>) -> tensor<32x197x1536xf32>
    %v3700 = stablehlo.dot_general %v3699, %b11_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x1536xf32>, tensor<1536x384xf32>) -> tensor<32x197x384xf32>
    %v3701 = stablehlo.broadcast_in_dim %b11_bfc2, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v3702 = stablehlo.add %v3700, %v3701 : tensor<32x197x384xf32>
    %v3703 = stablehlo.reshape %v3702 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3704 = stablehlo.broadcast_in_dim %dp23, dims = [0] : (tensor<32xf32>) -> tensor<32x75648xf32>
    %v3705 = stablehlo.multiply %v3704, %v3703 : tensor<32x75648xf32>
    %v3706 = stablehlo.add %v3652, %v3705 : tensor<32x75648xf32>
    %v3707 = stablehlo.reshape %v3706 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3708 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3709 = stablehlo.constant dense<384.0> : tensor<32x197x384xf32>
    %v3710 = stablehlo.constant dense<1.0e-5> : tensor<32x197x384xf32>
    %v3711 = stablehlo.reduce(%v3707 init: %v3708) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3712 = stablehlo.broadcast_in_dim %v3711, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v3713 = stablehlo.divide %v3712, %v3709 : tensor<32x197x384xf32>
    %v3714 = stablehlo.subtract %v3707, %v3713 : tensor<32x197x384xf32>
    %v3715 = stablehlo.multiply %v3714, %v3714 : tensor<32x197x384xf32>
    %v3716 = stablehlo.reduce(%v3715 init: %v3708) applies stablehlo.add across dimensions = [2] : (tensor<32x197x384xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v3717 = stablehlo.broadcast_in_dim %v3716, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x384xf32>
    %v3718 = stablehlo.divide %v3717, %v3709 : tensor<32x197x384xf32>
    %v3719 = stablehlo.add %v3718, %v3710 : tensor<32x197x384xf32>
    %v3720 = stablehlo.rsqrt %v3719 : tensor<32x197x384xf32>
    %v3721 = stablehlo.multiply %v3714, %v3720 : tensor<32x197x384xf32>
    %v3722 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v3723 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x384xf32>
    %v3724 = stablehlo.multiply %v3721, %v3722 : tensor<32x197x384xf32>
    %v3725 = stablehlo.add %v3724, %v3723 : tensor<32x197x384xf32>
    %v3726 = stablehlo.reshape %v3725 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3727 = stablehlo.reshape %v3726 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3728 = stablehlo.broadcast_in_dim %gF, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v3729 = stablehlo.multiply %v3727, %v3728 : tensor<32x197x384xf32>
    %v3730 = stablehlo.reshape %v3729 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3731 = stablehlo.reshape %v3730 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3732 = stablehlo.broadcast_in_dim %btF, dims = [2] : (tensor<384xf32>) -> tensor<32x197x384xf32>
    %v3733 = stablehlo.add %v3731, %v3732 : tensor<32x197x384xf32>
    %v3734 = stablehlo.reshape %v3733 : (tensor<32x197x384xf32>) -> tensor<32x75648xf32>
    %v3735 = stablehlo.reshape %v3734 : (tensor<32x75648xf32>) -> tensor<32x197x384xf32>
    %v3736 = stablehlo.slice %v3735 [0:32, 0:1, 0:384] : (tensor<32x197x384xf32>) -> tensor<32x1x384xf32>
    %v3737 = stablehlo.reshape %v3736 : (tensor<32x1x384xf32>) -> tensor<32x384xf32>
    %v3738 = stablehlo.dot_general %v3737, %Wc, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x384xf32>, tensor<384x1000xf32>) -> tensor<32x1000xf32>
    %v3739 = stablehlo.broadcast_in_dim %bc, dims = [1] : (tensor<1000xf32>) -> tensor<32x1000xf32>
    %v3740 = stablehlo.add %v3738, %v3739 : tensor<32x1000xf32>
    return %v3740 : tensor<32x1000xf32>
  }
}

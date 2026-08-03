module @m {
  func.func @vit_drop_fwd(%x: tensor<32x150528xf32>, %wConv: tensor<192x3x16x16xf32>, %bConv: tensor<192xf32>, %cls: tensor<192xf32>, %pos: tensor<197x192xf32>, %b0_g1: tensor<192xf32>, %b0_bt1: tensor<192xf32>, %b0_Wq: tensor<192x192xf32>, %b0_bq: tensor<192xf32>, %b0_Wk: tensor<192x192xf32>, %b0_bk: tensor<192xf32>, %b0_Wv: tensor<192x192xf32>, %b0_bv: tensor<192xf32>, %b0_Wo: tensor<192x192xf32>, %b0_bo: tensor<192xf32>, %b0_g2: tensor<192xf32>, %b0_bt2: tensor<192xf32>, %b0_Wfc1: tensor<192x768xf32>, %b0_bfc1: tensor<768xf32>, %b0_Wfc2: tensor<768x192xf32>, %b0_bfc2: tensor<192xf32>, %b1_g1: tensor<192xf32>, %b1_bt1: tensor<192xf32>, %b1_Wq: tensor<192x192xf32>, %b1_bq: tensor<192xf32>, %b1_Wk: tensor<192x192xf32>, %b1_bk: tensor<192xf32>, %b1_Wv: tensor<192x192xf32>, %b1_bv: tensor<192xf32>, %b1_Wo: tensor<192x192xf32>, %b1_bo: tensor<192xf32>, %b1_g2: tensor<192xf32>, %b1_bt2: tensor<192xf32>, %b1_Wfc1: tensor<192x768xf32>, %b1_bfc1: tensor<768xf32>, %b1_Wfc2: tensor<768x192xf32>, %b1_bfc2: tensor<192xf32>, %b2_g1: tensor<192xf32>, %b2_bt1: tensor<192xf32>, %b2_Wq: tensor<192x192xf32>, %b2_bq: tensor<192xf32>, %b2_Wk: tensor<192x192xf32>, %b2_bk: tensor<192xf32>, %b2_Wv: tensor<192x192xf32>, %b2_bv: tensor<192xf32>, %b2_Wo: tensor<192x192xf32>, %b2_bo: tensor<192xf32>, %b2_g2: tensor<192xf32>, %b2_bt2: tensor<192xf32>, %b2_Wfc1: tensor<192x768xf32>, %b2_bfc1: tensor<768xf32>, %b2_Wfc2: tensor<768x192xf32>, %b2_bfc2: tensor<192xf32>, %b3_g1: tensor<192xf32>, %b3_bt1: tensor<192xf32>, %b3_Wq: tensor<192x192xf32>, %b3_bq: tensor<192xf32>, %b3_Wk: tensor<192x192xf32>, %b3_bk: tensor<192xf32>, %b3_Wv: tensor<192x192xf32>, %b3_bv: tensor<192xf32>, %b3_Wo: tensor<192x192xf32>, %b3_bo: tensor<192xf32>, %b3_g2: tensor<192xf32>, %b3_bt2: tensor<192xf32>, %b3_Wfc1: tensor<192x768xf32>, %b3_bfc1: tensor<768xf32>, %b3_Wfc2: tensor<768x192xf32>, %b3_bfc2: tensor<192xf32>, %b4_g1: tensor<192xf32>, %b4_bt1: tensor<192xf32>, %b4_Wq: tensor<192x192xf32>, %b4_bq: tensor<192xf32>, %b4_Wk: tensor<192x192xf32>, %b4_bk: tensor<192xf32>, %b4_Wv: tensor<192x192xf32>, %b4_bv: tensor<192xf32>, %b4_Wo: tensor<192x192xf32>, %b4_bo: tensor<192xf32>, %b4_g2: tensor<192xf32>, %b4_bt2: tensor<192xf32>, %b4_Wfc1: tensor<192x768xf32>, %b4_bfc1: tensor<768xf32>, %b4_Wfc2: tensor<768x192xf32>, %b4_bfc2: tensor<192xf32>, %b5_g1: tensor<192xf32>, %b5_bt1: tensor<192xf32>, %b5_Wq: tensor<192x192xf32>, %b5_bq: tensor<192xf32>, %b5_Wk: tensor<192x192xf32>, %b5_bk: tensor<192xf32>, %b5_Wv: tensor<192x192xf32>, %b5_bv: tensor<192xf32>, %b5_Wo: tensor<192x192xf32>, %b5_bo: tensor<192xf32>, %b5_g2: tensor<192xf32>, %b5_bt2: tensor<192xf32>, %b5_Wfc1: tensor<192x768xf32>, %b5_bfc1: tensor<768xf32>, %b5_Wfc2: tensor<768x192xf32>, %b5_bfc2: tensor<192xf32>, %b6_g1: tensor<192xf32>, %b6_bt1: tensor<192xf32>, %b6_Wq: tensor<192x192xf32>, %b6_bq: tensor<192xf32>, %b6_Wk: tensor<192x192xf32>, %b6_bk: tensor<192xf32>, %b6_Wv: tensor<192x192xf32>, %b6_bv: tensor<192xf32>, %b6_Wo: tensor<192x192xf32>, %b6_bo: tensor<192xf32>, %b6_g2: tensor<192xf32>, %b6_bt2: tensor<192xf32>, %b6_Wfc1: tensor<192x768xf32>, %b6_bfc1: tensor<768xf32>, %b6_Wfc2: tensor<768x192xf32>, %b6_bfc2: tensor<192xf32>, %b7_g1: tensor<192xf32>, %b7_bt1: tensor<192xf32>, %b7_Wq: tensor<192x192xf32>, %b7_bq: tensor<192xf32>, %b7_Wk: tensor<192x192xf32>, %b7_bk: tensor<192xf32>, %b7_Wv: tensor<192x192xf32>, %b7_bv: tensor<192xf32>, %b7_Wo: tensor<192x192xf32>, %b7_bo: tensor<192xf32>, %b7_g2: tensor<192xf32>, %b7_bt2: tensor<192xf32>, %b7_Wfc1: tensor<192x768xf32>, %b7_bfc1: tensor<768xf32>, %b7_Wfc2: tensor<768x192xf32>, %b7_bfc2: tensor<192xf32>, %b8_g1: tensor<192xf32>, %b8_bt1: tensor<192xf32>, %b8_Wq: tensor<192x192xf32>, %b8_bq: tensor<192xf32>, %b8_Wk: tensor<192x192xf32>, %b8_bk: tensor<192xf32>, %b8_Wv: tensor<192x192xf32>, %b8_bv: tensor<192xf32>, %b8_Wo: tensor<192x192xf32>, %b8_bo: tensor<192xf32>, %b8_g2: tensor<192xf32>, %b8_bt2: tensor<192xf32>, %b8_Wfc1: tensor<192x768xf32>, %b8_bfc1: tensor<768xf32>, %b8_Wfc2: tensor<768x192xf32>, %b8_bfc2: tensor<192xf32>, %b9_g1: tensor<192xf32>, %b9_bt1: tensor<192xf32>, %b9_Wq: tensor<192x192xf32>, %b9_bq: tensor<192xf32>, %b9_Wk: tensor<192x192xf32>, %b9_bk: tensor<192xf32>, %b9_Wv: tensor<192x192xf32>, %b9_bv: tensor<192xf32>, %b9_Wo: tensor<192x192xf32>, %b9_bo: tensor<192xf32>, %b9_g2: tensor<192xf32>, %b9_bt2: tensor<192xf32>, %b9_Wfc1: tensor<192x768xf32>, %b9_bfc1: tensor<768xf32>, %b9_Wfc2: tensor<768x192xf32>, %b9_bfc2: tensor<192xf32>, %b10_g1: tensor<192xf32>, %b10_bt1: tensor<192xf32>, %b10_Wq: tensor<192x192xf32>, %b10_bq: tensor<192xf32>, %b10_Wk: tensor<192x192xf32>, %b10_bk: tensor<192xf32>, %b10_Wv: tensor<192x192xf32>, %b10_bv: tensor<192xf32>, %b10_Wo: tensor<192x192xf32>, %b10_bo: tensor<192xf32>, %b10_g2: tensor<192xf32>, %b10_bt2: tensor<192xf32>, %b10_Wfc1: tensor<192x768xf32>, %b10_bfc1: tensor<768xf32>, %b10_Wfc2: tensor<768x192xf32>, %b10_bfc2: tensor<192xf32>, %b11_g1: tensor<192xf32>, %b11_bt1: tensor<192xf32>, %b11_Wq: tensor<192x192xf32>, %b11_bq: tensor<192xf32>, %b11_Wk: tensor<192x192xf32>, %b11_bk: tensor<192xf32>, %b11_Wv: tensor<192x192xf32>, %b11_bv: tensor<192xf32>, %b11_Wo: tensor<192x192xf32>, %b11_bo: tensor<192xf32>, %b11_g2: tensor<192xf32>, %b11_bt2: tensor<192xf32>, %b11_Wfc1: tensor<192x768xf32>, %b11_bfc1: tensor<768xf32>, %b11_Wfc2: tensor<768x192xf32>, %b11_bfc2: tensor<192xf32>, %gF: tensor<192xf32>, %btF: tensor<192xf32>, %Wc: tensor<192x10xf32>, %bc: tensor<10xf32>, %dp0: tensor<32xf32>, %dp1: tensor<32xf32>, %dp2: tensor<32xf32>, %dp3: tensor<32xf32>, %dp4: tensor<32xf32>, %dp5: tensor<32xf32>, %dp6: tensor<32xf32>, %dp7: tensor<32xf32>, %dp8: tensor<32xf32>, %dp9: tensor<32xf32>, %dp10: tensor<32xf32>, %dp11: tensor<32xf32>, %dp12: tensor<32xf32>, %dp13: tensor<32xf32>, %dp14: tensor<32xf32>, %dp15: tensor<32xf32>, %dp16: tensor<32xf32>, %dp17: tensor<32xf32>, %dp18: tensor<32xf32>, %dp19: tensor<32xf32>, %dp20: tensor<32xf32>, %dp21: tensor<32xf32>, %dp22: tensor<32xf32>, %dp23: tensor<32xf32>) -> tensor<32x10xf32> {
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
    %v160 = stablehlo.broadcast_in_dim %dp0, dims = [0] : (tensor<32xf32>) -> tensor<32x37824xf32>
    %v161 = stablehlo.multiply %v160, %v159 : tensor<32x37824xf32>
    %v162 = stablehlo.add %v10, %v161 : tensor<32x37824xf32>
    %v163 = stablehlo.reshape %v162 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v164 = stablehlo.constant dense<0.0> : tensor<f32>
    %v165 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v166 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v167 = stablehlo.reduce(%v163 init: %v164) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v168 = stablehlo.broadcast_in_dim %v167, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v169 = stablehlo.divide %v168, %v165 : tensor<32x197x192xf32>
    %v170 = stablehlo.subtract %v163, %v169 : tensor<32x197x192xf32>
    %v171 = stablehlo.multiply %v170, %v170 : tensor<32x197x192xf32>
    %v172 = stablehlo.reduce(%v171 init: %v164) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v173 = stablehlo.broadcast_in_dim %v172, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v174 = stablehlo.divide %v173, %v165 : tensor<32x197x192xf32>
    %v175 = stablehlo.add %v174, %v166 : tensor<32x197x192xf32>
    %v176 = stablehlo.rsqrt %v175 : tensor<32x197x192xf32>
    %v177 = stablehlo.multiply %v170, %v176 : tensor<32x197x192xf32>
    %v178 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v179 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v180 = stablehlo.multiply %v177, %v178 : tensor<32x197x192xf32>
    %v181 = stablehlo.add %v180, %v179 : tensor<32x197x192xf32>
    %v182 = stablehlo.reshape %v181 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v183 = stablehlo.reshape %v182 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v184 = stablehlo.broadcast_in_dim %b0_g2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v185 = stablehlo.multiply %v183, %v184 : tensor<32x197x192xf32>
    %v186 = stablehlo.reshape %v185 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v187 = stablehlo.reshape %v186 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v188 = stablehlo.broadcast_in_dim %b0_bt2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v189 = stablehlo.add %v187, %v188 : tensor<32x197x192xf32>
    %v190 = stablehlo.reshape %v189 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v191 = stablehlo.reshape %v190 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v192 = stablehlo.dot_general %v191, %b0_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x768xf32>) -> tensor<32x197x768xf32>
    %v193 = stablehlo.broadcast_in_dim %b0_bfc1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v194 = stablehlo.add %v192, %v193 : tensor<32x197x768xf32>
    %v195 = stablehlo.reshape %v194 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v196 = stablehlo.multiply %v195, %v195 : tensor<32x151296xf32>
    %v197 = stablehlo.multiply %v196, %v195 : tensor<32x151296xf32>
    %v198 = stablehlo.constant dense<0.044715> : tensor<32x151296xf32>
    %v199 = stablehlo.multiply %v198, %v197 : tensor<32x151296xf32>
    %v200 = stablehlo.add %v195, %v199 : tensor<32x151296xf32>
    %v201 = stablehlo.constant dense<0.7978845608028654> : tensor<32x151296xf32>
    %v202 = stablehlo.multiply %v201, %v200 : tensor<32x151296xf32>
    %v203 = stablehlo.tanh %v202 : tensor<32x151296xf32>
    %v204 = stablehlo.constant dense<1.0> : tensor<32x151296xf32>
    %v205 = stablehlo.add %v204, %v203 : tensor<32x151296xf32>
    %v206 = stablehlo.constant dense<0.5> : tensor<32x151296xf32>
    %v207 = stablehlo.multiply %v206, %v195 : tensor<32x151296xf32>
    %v208 = stablehlo.multiply %v207, %v205 : tensor<32x151296xf32>
    %v209 = stablehlo.reshape %v208 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v210 = stablehlo.dot_general %v209, %b0_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x192xf32>) -> tensor<32x197x192xf32>
    %v211 = stablehlo.broadcast_in_dim %b0_bfc2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v212 = stablehlo.add %v210, %v211 : tensor<32x197x192xf32>
    %v213 = stablehlo.reshape %v212 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v214 = stablehlo.broadcast_in_dim %dp1, dims = [0] : (tensor<32xf32>) -> tensor<32x37824xf32>
    %v215 = stablehlo.multiply %v214, %v213 : tensor<32x37824xf32>
    %v216 = stablehlo.add %v162, %v215 : tensor<32x37824xf32>
    %v217 = stablehlo.reshape %v216 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v218 = stablehlo.constant dense<0.0> : tensor<f32>
    %v219 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v220 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v221 = stablehlo.reduce(%v217 init: %v218) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v222 = stablehlo.broadcast_in_dim %v221, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v223 = stablehlo.divide %v222, %v219 : tensor<32x197x192xf32>
    %v224 = stablehlo.subtract %v217, %v223 : tensor<32x197x192xf32>
    %v225 = stablehlo.multiply %v224, %v224 : tensor<32x197x192xf32>
    %v226 = stablehlo.reduce(%v225 init: %v218) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v227 = stablehlo.broadcast_in_dim %v226, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v228 = stablehlo.divide %v227, %v219 : tensor<32x197x192xf32>
    %v229 = stablehlo.add %v228, %v220 : tensor<32x197x192xf32>
    %v230 = stablehlo.rsqrt %v229 : tensor<32x197x192xf32>
    %v231 = stablehlo.multiply %v224, %v230 : tensor<32x197x192xf32>
    %v232 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v233 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v234 = stablehlo.multiply %v231, %v232 : tensor<32x197x192xf32>
    %v235 = stablehlo.add %v234, %v233 : tensor<32x197x192xf32>
    %v236 = stablehlo.reshape %v235 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v237 = stablehlo.reshape %v236 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v238 = stablehlo.broadcast_in_dim %b1_g1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v239 = stablehlo.multiply %v237, %v238 : tensor<32x197x192xf32>
    %v240 = stablehlo.reshape %v239 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v241 = stablehlo.reshape %v240 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v242 = stablehlo.broadcast_in_dim %b1_bt1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v243 = stablehlo.add %v241, %v242 : tensor<32x197x192xf32>
    %v244 = stablehlo.reshape %v243 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v245 = stablehlo.reshape %v244 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v246 = stablehlo.dot_general %v245, %b1_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v247 = stablehlo.broadcast_in_dim %b1_bq, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v248 = stablehlo.add %v246, %v247 : tensor<32x197x192xf32>
    %v249 = stablehlo.reshape %v248 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v250 = stablehlo.reshape %v244 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v251 = stablehlo.dot_general %v250, %b1_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v252 = stablehlo.broadcast_in_dim %b1_bk, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v253 = stablehlo.add %v251, %v252 : tensor<32x197x192xf32>
    %v254 = stablehlo.reshape %v253 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v255 = stablehlo.reshape %v244 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v256 = stablehlo.dot_general %v255, %b1_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v257 = stablehlo.broadcast_in_dim %b1_bv, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v258 = stablehlo.add %v256, %v257 : tensor<32x197x192xf32>
    %v259 = stablehlo.reshape %v258 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v260 = stablehlo.reshape %v249 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v261 = stablehlo.slice %v260 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v262 = stablehlo.reshape %v261 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v263 = stablehlo.reshape %v254 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v264 = stablehlo.slice %v263 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v265 = stablehlo.reshape %v264 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v266 = stablehlo.reshape %v259 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v267 = stablehlo.slice %v266 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v268 = stablehlo.reshape %v267 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v269 = stablehlo.reshape %v265 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v270 = stablehlo.transpose %v269, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v271 = stablehlo.reshape %v270 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v272 = stablehlo.reshape %v262 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v273 = stablehlo.reshape %v271 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v274 = stablehlo.dot_general %v272, %v273, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v275 = stablehlo.reshape %v274 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v276 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v277 = stablehlo.multiply %v275, %v276 : tensor<32x38809xf32>
    %v278 = stablehlo.reshape %v277 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v279 = stablehlo.constant dense<0.0> : tensor<f32>
    %v280 = stablehlo.exponential %v278 : tensor<32x197x197xf32>
    %v281 = stablehlo.reduce(%v280 init: %v279) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v282 = stablehlo.broadcast_in_dim %v281, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v283 = stablehlo.divide %v280, %v282 : tensor<32x197x197xf32>
    %v284 = stablehlo.reshape %v283 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v285 = stablehlo.reshape %v284 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v286 = stablehlo.reshape %v268 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v287 = stablehlo.dot_general %v285, %v286, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v288 = stablehlo.reshape %v287 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v289 = stablehlo.reshape %v288 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v290 = stablehlo.constant dense<0.0> : tensor<f32>
    %v291 = stablehlo.pad %v289, %v290, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v292 = stablehlo.reshape %v291 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v293 = stablehlo.reshape %v249 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v294 = stablehlo.slice %v293 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v295 = stablehlo.reshape %v294 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v296 = stablehlo.reshape %v254 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v297 = stablehlo.slice %v296 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v298 = stablehlo.reshape %v297 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v299 = stablehlo.reshape %v259 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v300 = stablehlo.slice %v299 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v301 = stablehlo.reshape %v300 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v302 = stablehlo.reshape %v298 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v303 = stablehlo.transpose %v302, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v304 = stablehlo.reshape %v303 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v305 = stablehlo.reshape %v295 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v306 = stablehlo.reshape %v304 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v307 = stablehlo.dot_general %v305, %v306, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v308 = stablehlo.reshape %v307 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v309 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v310 = stablehlo.multiply %v308, %v309 : tensor<32x38809xf32>
    %v311 = stablehlo.reshape %v310 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v312 = stablehlo.constant dense<0.0> : tensor<f32>
    %v313 = stablehlo.exponential %v311 : tensor<32x197x197xf32>
    %v314 = stablehlo.reduce(%v313 init: %v312) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v315 = stablehlo.broadcast_in_dim %v314, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v316 = stablehlo.divide %v313, %v315 : tensor<32x197x197xf32>
    %v317 = stablehlo.reshape %v316 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v318 = stablehlo.reshape %v317 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v319 = stablehlo.reshape %v301 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v320 = stablehlo.dot_general %v318, %v319, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v321 = stablehlo.reshape %v320 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v322 = stablehlo.reshape %v321 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v323 = stablehlo.constant dense<0.0> : tensor<f32>
    %v324 = stablehlo.pad %v322, %v323, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v325 = stablehlo.reshape %v324 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v326 = stablehlo.add %v292, %v325 : tensor<32x37824xf32>
    %v327 = stablehlo.reshape %v249 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v328 = stablehlo.slice %v327 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v329 = stablehlo.reshape %v328 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v330 = stablehlo.reshape %v254 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v331 = stablehlo.slice %v330 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v332 = stablehlo.reshape %v331 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v333 = stablehlo.reshape %v259 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v334 = stablehlo.slice %v333 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v335 = stablehlo.reshape %v334 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v336 = stablehlo.reshape %v332 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v337 = stablehlo.transpose %v336, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v338 = stablehlo.reshape %v337 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v339 = stablehlo.reshape %v329 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v340 = stablehlo.reshape %v338 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v341 = stablehlo.dot_general %v339, %v340, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v342 = stablehlo.reshape %v341 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v343 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v344 = stablehlo.multiply %v342, %v343 : tensor<32x38809xf32>
    %v345 = stablehlo.reshape %v344 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v346 = stablehlo.constant dense<0.0> : tensor<f32>
    %v347 = stablehlo.exponential %v345 : tensor<32x197x197xf32>
    %v348 = stablehlo.reduce(%v347 init: %v346) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v349 = stablehlo.broadcast_in_dim %v348, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v350 = stablehlo.divide %v347, %v349 : tensor<32x197x197xf32>
    %v351 = stablehlo.reshape %v350 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v352 = stablehlo.reshape %v351 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v353 = stablehlo.reshape %v335 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v354 = stablehlo.dot_general %v352, %v353, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v355 = stablehlo.reshape %v354 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v356 = stablehlo.reshape %v355 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v357 = stablehlo.constant dense<0.0> : tensor<f32>
    %v358 = stablehlo.pad %v356, %v357, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v359 = stablehlo.reshape %v358 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v360 = stablehlo.add %v326, %v359 : tensor<32x37824xf32>
    %v361 = stablehlo.reshape %v360 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v362 = stablehlo.dot_general %v361, %b1_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v363 = stablehlo.broadcast_in_dim %b1_bo, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v364 = stablehlo.add %v362, %v363 : tensor<32x197x192xf32>
    %v365 = stablehlo.reshape %v364 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v366 = stablehlo.broadcast_in_dim %dp2, dims = [0] : (tensor<32xf32>) -> tensor<32x37824xf32>
    %v367 = stablehlo.multiply %v366, %v365 : tensor<32x37824xf32>
    %v368 = stablehlo.add %v216, %v367 : tensor<32x37824xf32>
    %v369 = stablehlo.reshape %v368 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v370 = stablehlo.constant dense<0.0> : tensor<f32>
    %v371 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v372 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v373 = stablehlo.reduce(%v369 init: %v370) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v374 = stablehlo.broadcast_in_dim %v373, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v375 = stablehlo.divide %v374, %v371 : tensor<32x197x192xf32>
    %v376 = stablehlo.subtract %v369, %v375 : tensor<32x197x192xf32>
    %v377 = stablehlo.multiply %v376, %v376 : tensor<32x197x192xf32>
    %v378 = stablehlo.reduce(%v377 init: %v370) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v379 = stablehlo.broadcast_in_dim %v378, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v380 = stablehlo.divide %v379, %v371 : tensor<32x197x192xf32>
    %v381 = stablehlo.add %v380, %v372 : tensor<32x197x192xf32>
    %v382 = stablehlo.rsqrt %v381 : tensor<32x197x192xf32>
    %v383 = stablehlo.multiply %v376, %v382 : tensor<32x197x192xf32>
    %v384 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v385 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v386 = stablehlo.multiply %v383, %v384 : tensor<32x197x192xf32>
    %v387 = stablehlo.add %v386, %v385 : tensor<32x197x192xf32>
    %v388 = stablehlo.reshape %v387 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v389 = stablehlo.reshape %v388 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v390 = stablehlo.broadcast_in_dim %b1_g2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v391 = stablehlo.multiply %v389, %v390 : tensor<32x197x192xf32>
    %v392 = stablehlo.reshape %v391 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v393 = stablehlo.reshape %v392 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v394 = stablehlo.broadcast_in_dim %b1_bt2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v395 = stablehlo.add %v393, %v394 : tensor<32x197x192xf32>
    %v396 = stablehlo.reshape %v395 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v397 = stablehlo.reshape %v396 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v398 = stablehlo.dot_general %v397, %b1_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x768xf32>) -> tensor<32x197x768xf32>
    %v399 = stablehlo.broadcast_in_dim %b1_bfc1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v400 = stablehlo.add %v398, %v399 : tensor<32x197x768xf32>
    %v401 = stablehlo.reshape %v400 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v402 = stablehlo.multiply %v401, %v401 : tensor<32x151296xf32>
    %v403 = stablehlo.multiply %v402, %v401 : tensor<32x151296xf32>
    %v404 = stablehlo.constant dense<0.044715> : tensor<32x151296xf32>
    %v405 = stablehlo.multiply %v404, %v403 : tensor<32x151296xf32>
    %v406 = stablehlo.add %v401, %v405 : tensor<32x151296xf32>
    %v407 = stablehlo.constant dense<0.7978845608028654> : tensor<32x151296xf32>
    %v408 = stablehlo.multiply %v407, %v406 : tensor<32x151296xf32>
    %v409 = stablehlo.tanh %v408 : tensor<32x151296xf32>
    %v410 = stablehlo.constant dense<1.0> : tensor<32x151296xf32>
    %v411 = stablehlo.add %v410, %v409 : tensor<32x151296xf32>
    %v412 = stablehlo.constant dense<0.5> : tensor<32x151296xf32>
    %v413 = stablehlo.multiply %v412, %v401 : tensor<32x151296xf32>
    %v414 = stablehlo.multiply %v413, %v411 : tensor<32x151296xf32>
    %v415 = stablehlo.reshape %v414 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v416 = stablehlo.dot_general %v415, %b1_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x192xf32>) -> tensor<32x197x192xf32>
    %v417 = stablehlo.broadcast_in_dim %b1_bfc2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v418 = stablehlo.add %v416, %v417 : tensor<32x197x192xf32>
    %v419 = stablehlo.reshape %v418 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v420 = stablehlo.broadcast_in_dim %dp3, dims = [0] : (tensor<32xf32>) -> tensor<32x37824xf32>
    %v421 = stablehlo.multiply %v420, %v419 : tensor<32x37824xf32>
    %v422 = stablehlo.add %v368, %v421 : tensor<32x37824xf32>
    %v423 = stablehlo.reshape %v422 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v424 = stablehlo.constant dense<0.0> : tensor<f32>
    %v425 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v426 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v427 = stablehlo.reduce(%v423 init: %v424) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v428 = stablehlo.broadcast_in_dim %v427, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v429 = stablehlo.divide %v428, %v425 : tensor<32x197x192xf32>
    %v430 = stablehlo.subtract %v423, %v429 : tensor<32x197x192xf32>
    %v431 = stablehlo.multiply %v430, %v430 : tensor<32x197x192xf32>
    %v432 = stablehlo.reduce(%v431 init: %v424) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v433 = stablehlo.broadcast_in_dim %v432, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v434 = stablehlo.divide %v433, %v425 : tensor<32x197x192xf32>
    %v435 = stablehlo.add %v434, %v426 : tensor<32x197x192xf32>
    %v436 = stablehlo.rsqrt %v435 : tensor<32x197x192xf32>
    %v437 = stablehlo.multiply %v430, %v436 : tensor<32x197x192xf32>
    %v438 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v439 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v440 = stablehlo.multiply %v437, %v438 : tensor<32x197x192xf32>
    %v441 = stablehlo.add %v440, %v439 : tensor<32x197x192xf32>
    %v442 = stablehlo.reshape %v441 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v443 = stablehlo.reshape %v442 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v444 = stablehlo.broadcast_in_dim %b2_g1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v445 = stablehlo.multiply %v443, %v444 : tensor<32x197x192xf32>
    %v446 = stablehlo.reshape %v445 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v447 = stablehlo.reshape %v446 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v448 = stablehlo.broadcast_in_dim %b2_bt1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v449 = stablehlo.add %v447, %v448 : tensor<32x197x192xf32>
    %v450 = stablehlo.reshape %v449 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v451 = stablehlo.reshape %v450 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v452 = stablehlo.dot_general %v451, %b2_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v453 = stablehlo.broadcast_in_dim %b2_bq, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v454 = stablehlo.add %v452, %v453 : tensor<32x197x192xf32>
    %v455 = stablehlo.reshape %v454 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v456 = stablehlo.reshape %v450 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v457 = stablehlo.dot_general %v456, %b2_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v458 = stablehlo.broadcast_in_dim %b2_bk, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v459 = stablehlo.add %v457, %v458 : tensor<32x197x192xf32>
    %v460 = stablehlo.reshape %v459 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v461 = stablehlo.reshape %v450 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v462 = stablehlo.dot_general %v461, %b2_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v463 = stablehlo.broadcast_in_dim %b2_bv, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v464 = stablehlo.add %v462, %v463 : tensor<32x197x192xf32>
    %v465 = stablehlo.reshape %v464 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v466 = stablehlo.reshape %v455 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v467 = stablehlo.slice %v466 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v468 = stablehlo.reshape %v467 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v469 = stablehlo.reshape %v460 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v470 = stablehlo.slice %v469 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v471 = stablehlo.reshape %v470 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v472 = stablehlo.reshape %v465 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v473 = stablehlo.slice %v472 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v474 = stablehlo.reshape %v473 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v475 = stablehlo.reshape %v471 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v476 = stablehlo.transpose %v475, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v477 = stablehlo.reshape %v476 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v478 = stablehlo.reshape %v468 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v479 = stablehlo.reshape %v477 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v480 = stablehlo.dot_general %v478, %v479, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v481 = stablehlo.reshape %v480 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v482 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v483 = stablehlo.multiply %v481, %v482 : tensor<32x38809xf32>
    %v484 = stablehlo.reshape %v483 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v485 = stablehlo.constant dense<0.0> : tensor<f32>
    %v486 = stablehlo.exponential %v484 : tensor<32x197x197xf32>
    %v487 = stablehlo.reduce(%v486 init: %v485) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v488 = stablehlo.broadcast_in_dim %v487, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v489 = stablehlo.divide %v486, %v488 : tensor<32x197x197xf32>
    %v490 = stablehlo.reshape %v489 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v491 = stablehlo.reshape %v490 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v492 = stablehlo.reshape %v474 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v493 = stablehlo.dot_general %v491, %v492, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v494 = stablehlo.reshape %v493 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v495 = stablehlo.reshape %v494 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v496 = stablehlo.constant dense<0.0> : tensor<f32>
    %v497 = stablehlo.pad %v495, %v496, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v498 = stablehlo.reshape %v497 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v499 = stablehlo.reshape %v455 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v500 = stablehlo.slice %v499 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v501 = stablehlo.reshape %v500 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v502 = stablehlo.reshape %v460 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v503 = stablehlo.slice %v502 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v504 = stablehlo.reshape %v503 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v505 = stablehlo.reshape %v465 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v506 = stablehlo.slice %v505 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v507 = stablehlo.reshape %v506 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v508 = stablehlo.reshape %v504 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v509 = stablehlo.transpose %v508, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v510 = stablehlo.reshape %v509 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v511 = stablehlo.reshape %v501 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v512 = stablehlo.reshape %v510 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v513 = stablehlo.dot_general %v511, %v512, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v514 = stablehlo.reshape %v513 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v515 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v516 = stablehlo.multiply %v514, %v515 : tensor<32x38809xf32>
    %v517 = stablehlo.reshape %v516 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v518 = stablehlo.constant dense<0.0> : tensor<f32>
    %v519 = stablehlo.exponential %v517 : tensor<32x197x197xf32>
    %v520 = stablehlo.reduce(%v519 init: %v518) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v521 = stablehlo.broadcast_in_dim %v520, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v522 = stablehlo.divide %v519, %v521 : tensor<32x197x197xf32>
    %v523 = stablehlo.reshape %v522 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v524 = stablehlo.reshape %v523 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v525 = stablehlo.reshape %v507 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v526 = stablehlo.dot_general %v524, %v525, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v527 = stablehlo.reshape %v526 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v528 = stablehlo.reshape %v527 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v529 = stablehlo.constant dense<0.0> : tensor<f32>
    %v530 = stablehlo.pad %v528, %v529, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v531 = stablehlo.reshape %v530 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v532 = stablehlo.add %v498, %v531 : tensor<32x37824xf32>
    %v533 = stablehlo.reshape %v455 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v534 = stablehlo.slice %v533 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v535 = stablehlo.reshape %v534 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v536 = stablehlo.reshape %v460 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v537 = stablehlo.slice %v536 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v538 = stablehlo.reshape %v537 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v539 = stablehlo.reshape %v465 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v540 = stablehlo.slice %v539 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v541 = stablehlo.reshape %v540 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v542 = stablehlo.reshape %v538 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v543 = stablehlo.transpose %v542, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v544 = stablehlo.reshape %v543 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v545 = stablehlo.reshape %v535 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v546 = stablehlo.reshape %v544 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v547 = stablehlo.dot_general %v545, %v546, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v548 = stablehlo.reshape %v547 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v549 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v550 = stablehlo.multiply %v548, %v549 : tensor<32x38809xf32>
    %v551 = stablehlo.reshape %v550 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v552 = stablehlo.constant dense<0.0> : tensor<f32>
    %v553 = stablehlo.exponential %v551 : tensor<32x197x197xf32>
    %v554 = stablehlo.reduce(%v553 init: %v552) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v555 = stablehlo.broadcast_in_dim %v554, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v556 = stablehlo.divide %v553, %v555 : tensor<32x197x197xf32>
    %v557 = stablehlo.reshape %v556 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v558 = stablehlo.reshape %v557 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v559 = stablehlo.reshape %v541 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v560 = stablehlo.dot_general %v558, %v559, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v561 = stablehlo.reshape %v560 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v562 = stablehlo.reshape %v561 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v563 = stablehlo.constant dense<0.0> : tensor<f32>
    %v564 = stablehlo.pad %v562, %v563, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v565 = stablehlo.reshape %v564 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v566 = stablehlo.add %v532, %v565 : tensor<32x37824xf32>
    %v567 = stablehlo.reshape %v566 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v568 = stablehlo.dot_general %v567, %b2_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v569 = stablehlo.broadcast_in_dim %b2_bo, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v570 = stablehlo.add %v568, %v569 : tensor<32x197x192xf32>
    %v571 = stablehlo.reshape %v570 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v572 = stablehlo.broadcast_in_dim %dp4, dims = [0] : (tensor<32xf32>) -> tensor<32x37824xf32>
    %v573 = stablehlo.multiply %v572, %v571 : tensor<32x37824xf32>
    %v574 = stablehlo.add %v422, %v573 : tensor<32x37824xf32>
    %v575 = stablehlo.reshape %v574 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v576 = stablehlo.constant dense<0.0> : tensor<f32>
    %v577 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v578 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v579 = stablehlo.reduce(%v575 init: %v576) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v580 = stablehlo.broadcast_in_dim %v579, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v581 = stablehlo.divide %v580, %v577 : tensor<32x197x192xf32>
    %v582 = stablehlo.subtract %v575, %v581 : tensor<32x197x192xf32>
    %v583 = stablehlo.multiply %v582, %v582 : tensor<32x197x192xf32>
    %v584 = stablehlo.reduce(%v583 init: %v576) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v585 = stablehlo.broadcast_in_dim %v584, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v586 = stablehlo.divide %v585, %v577 : tensor<32x197x192xf32>
    %v587 = stablehlo.add %v586, %v578 : tensor<32x197x192xf32>
    %v588 = stablehlo.rsqrt %v587 : tensor<32x197x192xf32>
    %v589 = stablehlo.multiply %v582, %v588 : tensor<32x197x192xf32>
    %v590 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v591 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v592 = stablehlo.multiply %v589, %v590 : tensor<32x197x192xf32>
    %v593 = stablehlo.add %v592, %v591 : tensor<32x197x192xf32>
    %v594 = stablehlo.reshape %v593 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v595 = stablehlo.reshape %v594 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v596 = stablehlo.broadcast_in_dim %b2_g2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v597 = stablehlo.multiply %v595, %v596 : tensor<32x197x192xf32>
    %v598 = stablehlo.reshape %v597 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v599 = stablehlo.reshape %v598 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v600 = stablehlo.broadcast_in_dim %b2_bt2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v601 = stablehlo.add %v599, %v600 : tensor<32x197x192xf32>
    %v602 = stablehlo.reshape %v601 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v603 = stablehlo.reshape %v602 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v604 = stablehlo.dot_general %v603, %b2_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x768xf32>) -> tensor<32x197x768xf32>
    %v605 = stablehlo.broadcast_in_dim %b2_bfc1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v606 = stablehlo.add %v604, %v605 : tensor<32x197x768xf32>
    %v607 = stablehlo.reshape %v606 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v608 = stablehlo.multiply %v607, %v607 : tensor<32x151296xf32>
    %v609 = stablehlo.multiply %v608, %v607 : tensor<32x151296xf32>
    %v610 = stablehlo.constant dense<0.044715> : tensor<32x151296xf32>
    %v611 = stablehlo.multiply %v610, %v609 : tensor<32x151296xf32>
    %v612 = stablehlo.add %v607, %v611 : tensor<32x151296xf32>
    %v613 = stablehlo.constant dense<0.7978845608028654> : tensor<32x151296xf32>
    %v614 = stablehlo.multiply %v613, %v612 : tensor<32x151296xf32>
    %v615 = stablehlo.tanh %v614 : tensor<32x151296xf32>
    %v616 = stablehlo.constant dense<1.0> : tensor<32x151296xf32>
    %v617 = stablehlo.add %v616, %v615 : tensor<32x151296xf32>
    %v618 = stablehlo.constant dense<0.5> : tensor<32x151296xf32>
    %v619 = stablehlo.multiply %v618, %v607 : tensor<32x151296xf32>
    %v620 = stablehlo.multiply %v619, %v617 : tensor<32x151296xf32>
    %v621 = stablehlo.reshape %v620 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v622 = stablehlo.dot_general %v621, %b2_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x192xf32>) -> tensor<32x197x192xf32>
    %v623 = stablehlo.broadcast_in_dim %b2_bfc2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v624 = stablehlo.add %v622, %v623 : tensor<32x197x192xf32>
    %v625 = stablehlo.reshape %v624 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v626 = stablehlo.broadcast_in_dim %dp5, dims = [0] : (tensor<32xf32>) -> tensor<32x37824xf32>
    %v627 = stablehlo.multiply %v626, %v625 : tensor<32x37824xf32>
    %v628 = stablehlo.add %v574, %v627 : tensor<32x37824xf32>
    %v629 = stablehlo.reshape %v628 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v630 = stablehlo.constant dense<0.0> : tensor<f32>
    %v631 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v632 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v633 = stablehlo.reduce(%v629 init: %v630) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v634 = stablehlo.broadcast_in_dim %v633, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v635 = stablehlo.divide %v634, %v631 : tensor<32x197x192xf32>
    %v636 = stablehlo.subtract %v629, %v635 : tensor<32x197x192xf32>
    %v637 = stablehlo.multiply %v636, %v636 : tensor<32x197x192xf32>
    %v638 = stablehlo.reduce(%v637 init: %v630) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v639 = stablehlo.broadcast_in_dim %v638, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v640 = stablehlo.divide %v639, %v631 : tensor<32x197x192xf32>
    %v641 = stablehlo.add %v640, %v632 : tensor<32x197x192xf32>
    %v642 = stablehlo.rsqrt %v641 : tensor<32x197x192xf32>
    %v643 = stablehlo.multiply %v636, %v642 : tensor<32x197x192xf32>
    %v644 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v645 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v646 = stablehlo.multiply %v643, %v644 : tensor<32x197x192xf32>
    %v647 = stablehlo.add %v646, %v645 : tensor<32x197x192xf32>
    %v648 = stablehlo.reshape %v647 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v649 = stablehlo.reshape %v648 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v650 = stablehlo.broadcast_in_dim %b3_g1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v651 = stablehlo.multiply %v649, %v650 : tensor<32x197x192xf32>
    %v652 = stablehlo.reshape %v651 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v653 = stablehlo.reshape %v652 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v654 = stablehlo.broadcast_in_dim %b3_bt1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v655 = stablehlo.add %v653, %v654 : tensor<32x197x192xf32>
    %v656 = stablehlo.reshape %v655 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v657 = stablehlo.reshape %v656 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v658 = stablehlo.dot_general %v657, %b3_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v659 = stablehlo.broadcast_in_dim %b3_bq, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v660 = stablehlo.add %v658, %v659 : tensor<32x197x192xf32>
    %v661 = stablehlo.reshape %v660 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v662 = stablehlo.reshape %v656 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v663 = stablehlo.dot_general %v662, %b3_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v664 = stablehlo.broadcast_in_dim %b3_bk, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v665 = stablehlo.add %v663, %v664 : tensor<32x197x192xf32>
    %v666 = stablehlo.reshape %v665 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v667 = stablehlo.reshape %v656 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v668 = stablehlo.dot_general %v667, %b3_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v669 = stablehlo.broadcast_in_dim %b3_bv, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v670 = stablehlo.add %v668, %v669 : tensor<32x197x192xf32>
    %v671 = stablehlo.reshape %v670 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v672 = stablehlo.reshape %v661 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v673 = stablehlo.slice %v672 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v674 = stablehlo.reshape %v673 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v675 = stablehlo.reshape %v666 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v676 = stablehlo.slice %v675 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v677 = stablehlo.reshape %v676 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v678 = stablehlo.reshape %v671 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v679 = stablehlo.slice %v678 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v680 = stablehlo.reshape %v679 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v681 = stablehlo.reshape %v677 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v682 = stablehlo.transpose %v681, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v683 = stablehlo.reshape %v682 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v684 = stablehlo.reshape %v674 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v685 = stablehlo.reshape %v683 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v686 = stablehlo.dot_general %v684, %v685, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v687 = stablehlo.reshape %v686 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v688 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v689 = stablehlo.multiply %v687, %v688 : tensor<32x38809xf32>
    %v690 = stablehlo.reshape %v689 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v691 = stablehlo.constant dense<0.0> : tensor<f32>
    %v692 = stablehlo.exponential %v690 : tensor<32x197x197xf32>
    %v693 = stablehlo.reduce(%v692 init: %v691) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v694 = stablehlo.broadcast_in_dim %v693, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v695 = stablehlo.divide %v692, %v694 : tensor<32x197x197xf32>
    %v696 = stablehlo.reshape %v695 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v697 = stablehlo.reshape %v696 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v698 = stablehlo.reshape %v680 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v699 = stablehlo.dot_general %v697, %v698, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v700 = stablehlo.reshape %v699 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v701 = stablehlo.reshape %v700 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v702 = stablehlo.constant dense<0.0> : tensor<f32>
    %v703 = stablehlo.pad %v701, %v702, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v704 = stablehlo.reshape %v703 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v705 = stablehlo.reshape %v661 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v706 = stablehlo.slice %v705 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v707 = stablehlo.reshape %v706 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v708 = stablehlo.reshape %v666 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v709 = stablehlo.slice %v708 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v710 = stablehlo.reshape %v709 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v711 = stablehlo.reshape %v671 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v712 = stablehlo.slice %v711 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v713 = stablehlo.reshape %v712 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v714 = stablehlo.reshape %v710 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v715 = stablehlo.transpose %v714, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v716 = stablehlo.reshape %v715 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v717 = stablehlo.reshape %v707 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v718 = stablehlo.reshape %v716 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v719 = stablehlo.dot_general %v717, %v718, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v720 = stablehlo.reshape %v719 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v721 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v722 = stablehlo.multiply %v720, %v721 : tensor<32x38809xf32>
    %v723 = stablehlo.reshape %v722 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v724 = stablehlo.constant dense<0.0> : tensor<f32>
    %v725 = stablehlo.exponential %v723 : tensor<32x197x197xf32>
    %v726 = stablehlo.reduce(%v725 init: %v724) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v727 = stablehlo.broadcast_in_dim %v726, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v728 = stablehlo.divide %v725, %v727 : tensor<32x197x197xf32>
    %v729 = stablehlo.reshape %v728 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v730 = stablehlo.reshape %v729 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v731 = stablehlo.reshape %v713 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v732 = stablehlo.dot_general %v730, %v731, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v733 = stablehlo.reshape %v732 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v734 = stablehlo.reshape %v733 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v735 = stablehlo.constant dense<0.0> : tensor<f32>
    %v736 = stablehlo.pad %v734, %v735, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v737 = stablehlo.reshape %v736 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v738 = stablehlo.add %v704, %v737 : tensor<32x37824xf32>
    %v739 = stablehlo.reshape %v661 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v740 = stablehlo.slice %v739 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v741 = stablehlo.reshape %v740 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v742 = stablehlo.reshape %v666 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v743 = stablehlo.slice %v742 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v744 = stablehlo.reshape %v743 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v745 = stablehlo.reshape %v671 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v746 = stablehlo.slice %v745 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v747 = stablehlo.reshape %v746 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v748 = stablehlo.reshape %v744 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v749 = stablehlo.transpose %v748, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v750 = stablehlo.reshape %v749 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v751 = stablehlo.reshape %v741 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v752 = stablehlo.reshape %v750 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v753 = stablehlo.dot_general %v751, %v752, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v754 = stablehlo.reshape %v753 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v755 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v756 = stablehlo.multiply %v754, %v755 : tensor<32x38809xf32>
    %v757 = stablehlo.reshape %v756 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v758 = stablehlo.constant dense<0.0> : tensor<f32>
    %v759 = stablehlo.exponential %v757 : tensor<32x197x197xf32>
    %v760 = stablehlo.reduce(%v759 init: %v758) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v761 = stablehlo.broadcast_in_dim %v760, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v762 = stablehlo.divide %v759, %v761 : tensor<32x197x197xf32>
    %v763 = stablehlo.reshape %v762 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v764 = stablehlo.reshape %v763 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v765 = stablehlo.reshape %v747 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v766 = stablehlo.dot_general %v764, %v765, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v767 = stablehlo.reshape %v766 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v768 = stablehlo.reshape %v767 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v769 = stablehlo.constant dense<0.0> : tensor<f32>
    %v770 = stablehlo.pad %v768, %v769, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v771 = stablehlo.reshape %v770 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v772 = stablehlo.add %v738, %v771 : tensor<32x37824xf32>
    %v773 = stablehlo.reshape %v772 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v774 = stablehlo.dot_general %v773, %b3_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v775 = stablehlo.broadcast_in_dim %b3_bo, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v776 = stablehlo.add %v774, %v775 : tensor<32x197x192xf32>
    %v777 = stablehlo.reshape %v776 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v778 = stablehlo.broadcast_in_dim %dp6, dims = [0] : (tensor<32xf32>) -> tensor<32x37824xf32>
    %v779 = stablehlo.multiply %v778, %v777 : tensor<32x37824xf32>
    %v780 = stablehlo.add %v628, %v779 : tensor<32x37824xf32>
    %v781 = stablehlo.reshape %v780 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v782 = stablehlo.constant dense<0.0> : tensor<f32>
    %v783 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v784 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v785 = stablehlo.reduce(%v781 init: %v782) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v786 = stablehlo.broadcast_in_dim %v785, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v787 = stablehlo.divide %v786, %v783 : tensor<32x197x192xf32>
    %v788 = stablehlo.subtract %v781, %v787 : tensor<32x197x192xf32>
    %v789 = stablehlo.multiply %v788, %v788 : tensor<32x197x192xf32>
    %v790 = stablehlo.reduce(%v789 init: %v782) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v791 = stablehlo.broadcast_in_dim %v790, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v792 = stablehlo.divide %v791, %v783 : tensor<32x197x192xf32>
    %v793 = stablehlo.add %v792, %v784 : tensor<32x197x192xf32>
    %v794 = stablehlo.rsqrt %v793 : tensor<32x197x192xf32>
    %v795 = stablehlo.multiply %v788, %v794 : tensor<32x197x192xf32>
    %v796 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v797 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v798 = stablehlo.multiply %v795, %v796 : tensor<32x197x192xf32>
    %v799 = stablehlo.add %v798, %v797 : tensor<32x197x192xf32>
    %v800 = stablehlo.reshape %v799 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v801 = stablehlo.reshape %v800 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v802 = stablehlo.broadcast_in_dim %b3_g2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v803 = stablehlo.multiply %v801, %v802 : tensor<32x197x192xf32>
    %v804 = stablehlo.reshape %v803 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v805 = stablehlo.reshape %v804 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v806 = stablehlo.broadcast_in_dim %b3_bt2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v807 = stablehlo.add %v805, %v806 : tensor<32x197x192xf32>
    %v808 = stablehlo.reshape %v807 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v809 = stablehlo.reshape %v808 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v810 = stablehlo.dot_general %v809, %b3_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x768xf32>) -> tensor<32x197x768xf32>
    %v811 = stablehlo.broadcast_in_dim %b3_bfc1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v812 = stablehlo.add %v810, %v811 : tensor<32x197x768xf32>
    %v813 = stablehlo.reshape %v812 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v814 = stablehlo.multiply %v813, %v813 : tensor<32x151296xf32>
    %v815 = stablehlo.multiply %v814, %v813 : tensor<32x151296xf32>
    %v816 = stablehlo.constant dense<0.044715> : tensor<32x151296xf32>
    %v817 = stablehlo.multiply %v816, %v815 : tensor<32x151296xf32>
    %v818 = stablehlo.add %v813, %v817 : tensor<32x151296xf32>
    %v819 = stablehlo.constant dense<0.7978845608028654> : tensor<32x151296xf32>
    %v820 = stablehlo.multiply %v819, %v818 : tensor<32x151296xf32>
    %v821 = stablehlo.tanh %v820 : tensor<32x151296xf32>
    %v822 = stablehlo.constant dense<1.0> : tensor<32x151296xf32>
    %v823 = stablehlo.add %v822, %v821 : tensor<32x151296xf32>
    %v824 = stablehlo.constant dense<0.5> : tensor<32x151296xf32>
    %v825 = stablehlo.multiply %v824, %v813 : tensor<32x151296xf32>
    %v826 = stablehlo.multiply %v825, %v823 : tensor<32x151296xf32>
    %v827 = stablehlo.reshape %v826 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v828 = stablehlo.dot_general %v827, %b3_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x192xf32>) -> tensor<32x197x192xf32>
    %v829 = stablehlo.broadcast_in_dim %b3_bfc2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v830 = stablehlo.add %v828, %v829 : tensor<32x197x192xf32>
    %v831 = stablehlo.reshape %v830 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v832 = stablehlo.broadcast_in_dim %dp7, dims = [0] : (tensor<32xf32>) -> tensor<32x37824xf32>
    %v833 = stablehlo.multiply %v832, %v831 : tensor<32x37824xf32>
    %v834 = stablehlo.add %v780, %v833 : tensor<32x37824xf32>
    %v835 = stablehlo.reshape %v834 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v836 = stablehlo.constant dense<0.0> : tensor<f32>
    %v837 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v838 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v839 = stablehlo.reduce(%v835 init: %v836) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v840 = stablehlo.broadcast_in_dim %v839, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v841 = stablehlo.divide %v840, %v837 : tensor<32x197x192xf32>
    %v842 = stablehlo.subtract %v835, %v841 : tensor<32x197x192xf32>
    %v843 = stablehlo.multiply %v842, %v842 : tensor<32x197x192xf32>
    %v844 = stablehlo.reduce(%v843 init: %v836) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v845 = stablehlo.broadcast_in_dim %v844, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v846 = stablehlo.divide %v845, %v837 : tensor<32x197x192xf32>
    %v847 = stablehlo.add %v846, %v838 : tensor<32x197x192xf32>
    %v848 = stablehlo.rsqrt %v847 : tensor<32x197x192xf32>
    %v849 = stablehlo.multiply %v842, %v848 : tensor<32x197x192xf32>
    %v850 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v851 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v852 = stablehlo.multiply %v849, %v850 : tensor<32x197x192xf32>
    %v853 = stablehlo.add %v852, %v851 : tensor<32x197x192xf32>
    %v854 = stablehlo.reshape %v853 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v855 = stablehlo.reshape %v854 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v856 = stablehlo.broadcast_in_dim %b4_g1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v857 = stablehlo.multiply %v855, %v856 : tensor<32x197x192xf32>
    %v858 = stablehlo.reshape %v857 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v859 = stablehlo.reshape %v858 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v860 = stablehlo.broadcast_in_dim %b4_bt1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v861 = stablehlo.add %v859, %v860 : tensor<32x197x192xf32>
    %v862 = stablehlo.reshape %v861 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v863 = stablehlo.reshape %v862 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v864 = stablehlo.dot_general %v863, %b4_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v865 = stablehlo.broadcast_in_dim %b4_bq, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v866 = stablehlo.add %v864, %v865 : tensor<32x197x192xf32>
    %v867 = stablehlo.reshape %v866 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v868 = stablehlo.reshape %v862 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v869 = stablehlo.dot_general %v868, %b4_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v870 = stablehlo.broadcast_in_dim %b4_bk, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v871 = stablehlo.add %v869, %v870 : tensor<32x197x192xf32>
    %v872 = stablehlo.reshape %v871 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v873 = stablehlo.reshape %v862 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v874 = stablehlo.dot_general %v873, %b4_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v875 = stablehlo.broadcast_in_dim %b4_bv, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v876 = stablehlo.add %v874, %v875 : tensor<32x197x192xf32>
    %v877 = stablehlo.reshape %v876 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v878 = stablehlo.reshape %v867 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v879 = stablehlo.slice %v878 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v880 = stablehlo.reshape %v879 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v881 = stablehlo.reshape %v872 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v882 = stablehlo.slice %v881 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v883 = stablehlo.reshape %v882 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v884 = stablehlo.reshape %v877 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v885 = stablehlo.slice %v884 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v886 = stablehlo.reshape %v885 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v887 = stablehlo.reshape %v883 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v888 = stablehlo.transpose %v887, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v889 = stablehlo.reshape %v888 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v890 = stablehlo.reshape %v880 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v891 = stablehlo.reshape %v889 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v892 = stablehlo.dot_general %v890, %v891, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v893 = stablehlo.reshape %v892 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v894 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v895 = stablehlo.multiply %v893, %v894 : tensor<32x38809xf32>
    %v896 = stablehlo.reshape %v895 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v897 = stablehlo.constant dense<0.0> : tensor<f32>
    %v898 = stablehlo.exponential %v896 : tensor<32x197x197xf32>
    %v899 = stablehlo.reduce(%v898 init: %v897) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v900 = stablehlo.broadcast_in_dim %v899, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v901 = stablehlo.divide %v898, %v900 : tensor<32x197x197xf32>
    %v902 = stablehlo.reshape %v901 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v903 = stablehlo.reshape %v902 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v904 = stablehlo.reshape %v886 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v905 = stablehlo.dot_general %v903, %v904, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v906 = stablehlo.reshape %v905 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v907 = stablehlo.reshape %v906 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v908 = stablehlo.constant dense<0.0> : tensor<f32>
    %v909 = stablehlo.pad %v907, %v908, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v910 = stablehlo.reshape %v909 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v911 = stablehlo.reshape %v867 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v912 = stablehlo.slice %v911 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v913 = stablehlo.reshape %v912 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v914 = stablehlo.reshape %v872 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v915 = stablehlo.slice %v914 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v916 = stablehlo.reshape %v915 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v917 = stablehlo.reshape %v877 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v918 = stablehlo.slice %v917 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v919 = stablehlo.reshape %v918 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v920 = stablehlo.reshape %v916 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v921 = stablehlo.transpose %v920, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v922 = stablehlo.reshape %v921 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v923 = stablehlo.reshape %v913 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v924 = stablehlo.reshape %v922 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v925 = stablehlo.dot_general %v923, %v924, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v926 = stablehlo.reshape %v925 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v927 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v928 = stablehlo.multiply %v926, %v927 : tensor<32x38809xf32>
    %v929 = stablehlo.reshape %v928 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v930 = stablehlo.constant dense<0.0> : tensor<f32>
    %v931 = stablehlo.exponential %v929 : tensor<32x197x197xf32>
    %v932 = stablehlo.reduce(%v931 init: %v930) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v933 = stablehlo.broadcast_in_dim %v932, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v934 = stablehlo.divide %v931, %v933 : tensor<32x197x197xf32>
    %v935 = stablehlo.reshape %v934 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v936 = stablehlo.reshape %v935 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v937 = stablehlo.reshape %v919 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v938 = stablehlo.dot_general %v936, %v937, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v939 = stablehlo.reshape %v938 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v940 = stablehlo.reshape %v939 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v941 = stablehlo.constant dense<0.0> : tensor<f32>
    %v942 = stablehlo.pad %v940, %v941, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v943 = stablehlo.reshape %v942 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v944 = stablehlo.add %v910, %v943 : tensor<32x37824xf32>
    %v945 = stablehlo.reshape %v867 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v946 = stablehlo.slice %v945 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v947 = stablehlo.reshape %v946 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v948 = stablehlo.reshape %v872 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v949 = stablehlo.slice %v948 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v950 = stablehlo.reshape %v949 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v951 = stablehlo.reshape %v877 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v952 = stablehlo.slice %v951 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
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
    %v964 = stablehlo.constant dense<0.0> : tensor<f32>
    %v965 = stablehlo.exponential %v963 : tensor<32x197x197xf32>
    %v966 = stablehlo.reduce(%v965 init: %v964) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v967 = stablehlo.broadcast_in_dim %v966, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v968 = stablehlo.divide %v965, %v967 : tensor<32x197x197xf32>
    %v969 = stablehlo.reshape %v968 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v970 = stablehlo.reshape %v969 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v971 = stablehlo.reshape %v953 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v972 = stablehlo.dot_general %v970, %v971, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v973 = stablehlo.reshape %v972 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v974 = stablehlo.reshape %v973 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v975 = stablehlo.constant dense<0.0> : tensor<f32>
    %v976 = stablehlo.pad %v974, %v975, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v977 = stablehlo.reshape %v976 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v978 = stablehlo.add %v944, %v977 : tensor<32x37824xf32>
    %v979 = stablehlo.reshape %v978 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v980 = stablehlo.dot_general %v979, %b4_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v981 = stablehlo.broadcast_in_dim %b4_bo, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v982 = stablehlo.add %v980, %v981 : tensor<32x197x192xf32>
    %v983 = stablehlo.reshape %v982 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v984 = stablehlo.broadcast_in_dim %dp8, dims = [0] : (tensor<32xf32>) -> tensor<32x37824xf32>
    %v985 = stablehlo.multiply %v984, %v983 : tensor<32x37824xf32>
    %v986 = stablehlo.add %v834, %v985 : tensor<32x37824xf32>
    %v987 = stablehlo.reshape %v986 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v988 = stablehlo.constant dense<0.0> : tensor<f32>
    %v989 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v990 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v991 = stablehlo.reduce(%v987 init: %v988) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v992 = stablehlo.broadcast_in_dim %v991, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v993 = stablehlo.divide %v992, %v989 : tensor<32x197x192xf32>
    %v994 = stablehlo.subtract %v987, %v993 : tensor<32x197x192xf32>
    %v995 = stablehlo.multiply %v994, %v994 : tensor<32x197x192xf32>
    %v996 = stablehlo.reduce(%v995 init: %v988) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v997 = stablehlo.broadcast_in_dim %v996, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v998 = stablehlo.divide %v997, %v989 : tensor<32x197x192xf32>
    %v999 = stablehlo.add %v998, %v990 : tensor<32x197x192xf32>
    %v1000 = stablehlo.rsqrt %v999 : tensor<32x197x192xf32>
    %v1001 = stablehlo.multiply %v994, %v1000 : tensor<32x197x192xf32>
    %v1002 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v1003 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v1004 = stablehlo.multiply %v1001, %v1002 : tensor<32x197x192xf32>
    %v1005 = stablehlo.add %v1004, %v1003 : tensor<32x197x192xf32>
    %v1006 = stablehlo.reshape %v1005 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1007 = stablehlo.reshape %v1006 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1008 = stablehlo.broadcast_in_dim %b4_g2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1009 = stablehlo.multiply %v1007, %v1008 : tensor<32x197x192xf32>
    %v1010 = stablehlo.reshape %v1009 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1011 = stablehlo.reshape %v1010 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1012 = stablehlo.broadcast_in_dim %b4_bt2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1013 = stablehlo.add %v1011, %v1012 : tensor<32x197x192xf32>
    %v1014 = stablehlo.reshape %v1013 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1015 = stablehlo.reshape %v1014 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1016 = stablehlo.dot_general %v1015, %b4_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x768xf32>) -> tensor<32x197x768xf32>
    %v1017 = stablehlo.broadcast_in_dim %b4_bfc1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v1018 = stablehlo.add %v1016, %v1017 : tensor<32x197x768xf32>
    %v1019 = stablehlo.reshape %v1018 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1020 = stablehlo.multiply %v1019, %v1019 : tensor<32x151296xf32>
    %v1021 = stablehlo.multiply %v1020, %v1019 : tensor<32x151296xf32>
    %v1022 = stablehlo.constant dense<0.044715> : tensor<32x151296xf32>
    %v1023 = stablehlo.multiply %v1022, %v1021 : tensor<32x151296xf32>
    %v1024 = stablehlo.add %v1019, %v1023 : tensor<32x151296xf32>
    %v1025 = stablehlo.constant dense<0.7978845608028654> : tensor<32x151296xf32>
    %v1026 = stablehlo.multiply %v1025, %v1024 : tensor<32x151296xf32>
    %v1027 = stablehlo.tanh %v1026 : tensor<32x151296xf32>
    %v1028 = stablehlo.constant dense<1.0> : tensor<32x151296xf32>
    %v1029 = stablehlo.add %v1028, %v1027 : tensor<32x151296xf32>
    %v1030 = stablehlo.constant dense<0.5> : tensor<32x151296xf32>
    %v1031 = stablehlo.multiply %v1030, %v1019 : tensor<32x151296xf32>
    %v1032 = stablehlo.multiply %v1031, %v1029 : tensor<32x151296xf32>
    %v1033 = stablehlo.reshape %v1032 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1034 = stablehlo.dot_general %v1033, %b4_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x192xf32>) -> tensor<32x197x192xf32>
    %v1035 = stablehlo.broadcast_in_dim %b4_bfc2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1036 = stablehlo.add %v1034, %v1035 : tensor<32x197x192xf32>
    %v1037 = stablehlo.reshape %v1036 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1038 = stablehlo.broadcast_in_dim %dp9, dims = [0] : (tensor<32xf32>) -> tensor<32x37824xf32>
    %v1039 = stablehlo.multiply %v1038, %v1037 : tensor<32x37824xf32>
    %v1040 = stablehlo.add %v986, %v1039 : tensor<32x37824xf32>
    %v1041 = stablehlo.reshape %v1040 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1042 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1043 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v1044 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v1045 = stablehlo.reduce(%v1041 init: %v1042) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1046 = stablehlo.broadcast_in_dim %v1045, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v1047 = stablehlo.divide %v1046, %v1043 : tensor<32x197x192xf32>
    %v1048 = stablehlo.subtract %v1041, %v1047 : tensor<32x197x192xf32>
    %v1049 = stablehlo.multiply %v1048, %v1048 : tensor<32x197x192xf32>
    %v1050 = stablehlo.reduce(%v1049 init: %v1042) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1051 = stablehlo.broadcast_in_dim %v1050, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v1052 = stablehlo.divide %v1051, %v1043 : tensor<32x197x192xf32>
    %v1053 = stablehlo.add %v1052, %v1044 : tensor<32x197x192xf32>
    %v1054 = stablehlo.rsqrt %v1053 : tensor<32x197x192xf32>
    %v1055 = stablehlo.multiply %v1048, %v1054 : tensor<32x197x192xf32>
    %v1056 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v1057 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v1058 = stablehlo.multiply %v1055, %v1056 : tensor<32x197x192xf32>
    %v1059 = stablehlo.add %v1058, %v1057 : tensor<32x197x192xf32>
    %v1060 = stablehlo.reshape %v1059 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1061 = stablehlo.reshape %v1060 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1062 = stablehlo.broadcast_in_dim %b5_g1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1063 = stablehlo.multiply %v1061, %v1062 : tensor<32x197x192xf32>
    %v1064 = stablehlo.reshape %v1063 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1065 = stablehlo.reshape %v1064 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1066 = stablehlo.broadcast_in_dim %b5_bt1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1067 = stablehlo.add %v1065, %v1066 : tensor<32x197x192xf32>
    %v1068 = stablehlo.reshape %v1067 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1069 = stablehlo.reshape %v1068 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1070 = stablehlo.dot_general %v1069, %b5_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v1071 = stablehlo.broadcast_in_dim %b5_bq, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1072 = stablehlo.add %v1070, %v1071 : tensor<32x197x192xf32>
    %v1073 = stablehlo.reshape %v1072 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1074 = stablehlo.reshape %v1068 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1075 = stablehlo.dot_general %v1074, %b5_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v1076 = stablehlo.broadcast_in_dim %b5_bk, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1077 = stablehlo.add %v1075, %v1076 : tensor<32x197x192xf32>
    %v1078 = stablehlo.reshape %v1077 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1079 = stablehlo.reshape %v1068 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1080 = stablehlo.dot_general %v1079, %b5_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v1081 = stablehlo.broadcast_in_dim %b5_bv, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1082 = stablehlo.add %v1080, %v1081 : tensor<32x197x192xf32>
    %v1083 = stablehlo.reshape %v1082 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1084 = stablehlo.reshape %v1073 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1085 = stablehlo.slice %v1084 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1086 = stablehlo.reshape %v1085 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1087 = stablehlo.reshape %v1078 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1088 = stablehlo.slice %v1087 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1089 = stablehlo.reshape %v1088 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1090 = stablehlo.reshape %v1083 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1091 = stablehlo.slice %v1090 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1092 = stablehlo.reshape %v1091 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1093 = stablehlo.reshape %v1089 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1094 = stablehlo.transpose %v1093, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1095 = stablehlo.reshape %v1094 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1096 = stablehlo.reshape %v1086 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1097 = stablehlo.reshape %v1095 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1098 = stablehlo.dot_general %v1096, %v1097, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1099 = stablehlo.reshape %v1098 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1100 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1101 = stablehlo.multiply %v1099, %v1100 : tensor<32x38809xf32>
    %v1102 = stablehlo.reshape %v1101 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1103 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1104 = stablehlo.exponential %v1102 : tensor<32x197x197xf32>
    %v1105 = stablehlo.reduce(%v1104 init: %v1103) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1106 = stablehlo.broadcast_in_dim %v1105, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1107 = stablehlo.divide %v1104, %v1106 : tensor<32x197x197xf32>
    %v1108 = stablehlo.reshape %v1107 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1109 = stablehlo.reshape %v1108 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1110 = stablehlo.reshape %v1092 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1111 = stablehlo.dot_general %v1109, %v1110, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1112 = stablehlo.reshape %v1111 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1113 = stablehlo.reshape %v1112 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1114 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1115 = stablehlo.pad %v1113, %v1114, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v1116 = stablehlo.reshape %v1115 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1117 = stablehlo.reshape %v1073 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1118 = stablehlo.slice %v1117 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1119 = stablehlo.reshape %v1118 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1120 = stablehlo.reshape %v1078 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1121 = stablehlo.slice %v1120 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1122 = stablehlo.reshape %v1121 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1123 = stablehlo.reshape %v1083 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1124 = stablehlo.slice %v1123 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
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
    %v1148 = stablehlo.pad %v1146, %v1147, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v1149 = stablehlo.reshape %v1148 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1150 = stablehlo.add %v1116, %v1149 : tensor<32x37824xf32>
    %v1151 = stablehlo.reshape %v1073 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1152 = stablehlo.slice %v1151 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1153 = stablehlo.reshape %v1152 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1154 = stablehlo.reshape %v1078 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1155 = stablehlo.slice %v1154 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1156 = stablehlo.reshape %v1155 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1157 = stablehlo.reshape %v1083 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1158 = stablehlo.slice %v1157 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1159 = stablehlo.reshape %v1158 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1160 = stablehlo.reshape %v1156 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1161 = stablehlo.transpose %v1160, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1162 = stablehlo.reshape %v1161 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1163 = stablehlo.reshape %v1153 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1164 = stablehlo.reshape %v1162 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1165 = stablehlo.dot_general %v1163, %v1164, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1166 = stablehlo.reshape %v1165 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1167 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1168 = stablehlo.multiply %v1166, %v1167 : tensor<32x38809xf32>
    %v1169 = stablehlo.reshape %v1168 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1170 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1171 = stablehlo.exponential %v1169 : tensor<32x197x197xf32>
    %v1172 = stablehlo.reduce(%v1171 init: %v1170) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1173 = stablehlo.broadcast_in_dim %v1172, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1174 = stablehlo.divide %v1171, %v1173 : tensor<32x197x197xf32>
    %v1175 = stablehlo.reshape %v1174 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1176 = stablehlo.reshape %v1175 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1177 = stablehlo.reshape %v1159 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1178 = stablehlo.dot_general %v1176, %v1177, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1179 = stablehlo.reshape %v1178 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1180 = stablehlo.reshape %v1179 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1181 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1182 = stablehlo.pad %v1180, %v1181, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v1183 = stablehlo.reshape %v1182 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1184 = stablehlo.add %v1150, %v1183 : tensor<32x37824xf32>
    %v1185 = stablehlo.reshape %v1184 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1186 = stablehlo.dot_general %v1185, %b5_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v1187 = stablehlo.broadcast_in_dim %b5_bo, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1188 = stablehlo.add %v1186, %v1187 : tensor<32x197x192xf32>
    %v1189 = stablehlo.reshape %v1188 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1190 = stablehlo.broadcast_in_dim %dp10, dims = [0] : (tensor<32xf32>) -> tensor<32x37824xf32>
    %v1191 = stablehlo.multiply %v1190, %v1189 : tensor<32x37824xf32>
    %v1192 = stablehlo.add %v1040, %v1191 : tensor<32x37824xf32>
    %v1193 = stablehlo.reshape %v1192 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1194 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1195 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v1196 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v1197 = stablehlo.reduce(%v1193 init: %v1194) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1198 = stablehlo.broadcast_in_dim %v1197, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v1199 = stablehlo.divide %v1198, %v1195 : tensor<32x197x192xf32>
    %v1200 = stablehlo.subtract %v1193, %v1199 : tensor<32x197x192xf32>
    %v1201 = stablehlo.multiply %v1200, %v1200 : tensor<32x197x192xf32>
    %v1202 = stablehlo.reduce(%v1201 init: %v1194) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1203 = stablehlo.broadcast_in_dim %v1202, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v1204 = stablehlo.divide %v1203, %v1195 : tensor<32x197x192xf32>
    %v1205 = stablehlo.add %v1204, %v1196 : tensor<32x197x192xf32>
    %v1206 = stablehlo.rsqrt %v1205 : tensor<32x197x192xf32>
    %v1207 = stablehlo.multiply %v1200, %v1206 : tensor<32x197x192xf32>
    %v1208 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v1209 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v1210 = stablehlo.multiply %v1207, %v1208 : tensor<32x197x192xf32>
    %v1211 = stablehlo.add %v1210, %v1209 : tensor<32x197x192xf32>
    %v1212 = stablehlo.reshape %v1211 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1213 = stablehlo.reshape %v1212 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1214 = stablehlo.broadcast_in_dim %b5_g2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1215 = stablehlo.multiply %v1213, %v1214 : tensor<32x197x192xf32>
    %v1216 = stablehlo.reshape %v1215 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1217 = stablehlo.reshape %v1216 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1218 = stablehlo.broadcast_in_dim %b5_bt2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1219 = stablehlo.add %v1217, %v1218 : tensor<32x197x192xf32>
    %v1220 = stablehlo.reshape %v1219 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1221 = stablehlo.reshape %v1220 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1222 = stablehlo.dot_general %v1221, %b5_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x768xf32>) -> tensor<32x197x768xf32>
    %v1223 = stablehlo.broadcast_in_dim %b5_bfc1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v1224 = stablehlo.add %v1222, %v1223 : tensor<32x197x768xf32>
    %v1225 = stablehlo.reshape %v1224 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1226 = stablehlo.multiply %v1225, %v1225 : tensor<32x151296xf32>
    %v1227 = stablehlo.multiply %v1226, %v1225 : tensor<32x151296xf32>
    %v1228 = stablehlo.constant dense<0.044715> : tensor<32x151296xf32>
    %v1229 = stablehlo.multiply %v1228, %v1227 : tensor<32x151296xf32>
    %v1230 = stablehlo.add %v1225, %v1229 : tensor<32x151296xf32>
    %v1231 = stablehlo.constant dense<0.7978845608028654> : tensor<32x151296xf32>
    %v1232 = stablehlo.multiply %v1231, %v1230 : tensor<32x151296xf32>
    %v1233 = stablehlo.tanh %v1232 : tensor<32x151296xf32>
    %v1234 = stablehlo.constant dense<1.0> : tensor<32x151296xf32>
    %v1235 = stablehlo.add %v1234, %v1233 : tensor<32x151296xf32>
    %v1236 = stablehlo.constant dense<0.5> : tensor<32x151296xf32>
    %v1237 = stablehlo.multiply %v1236, %v1225 : tensor<32x151296xf32>
    %v1238 = stablehlo.multiply %v1237, %v1235 : tensor<32x151296xf32>
    %v1239 = stablehlo.reshape %v1238 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1240 = stablehlo.dot_general %v1239, %b5_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x192xf32>) -> tensor<32x197x192xf32>
    %v1241 = stablehlo.broadcast_in_dim %b5_bfc2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1242 = stablehlo.add %v1240, %v1241 : tensor<32x197x192xf32>
    %v1243 = stablehlo.reshape %v1242 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1244 = stablehlo.broadcast_in_dim %dp11, dims = [0] : (tensor<32xf32>) -> tensor<32x37824xf32>
    %v1245 = stablehlo.multiply %v1244, %v1243 : tensor<32x37824xf32>
    %v1246 = stablehlo.add %v1192, %v1245 : tensor<32x37824xf32>
    %v1247 = stablehlo.reshape %v1246 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1248 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1249 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v1250 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v1251 = stablehlo.reduce(%v1247 init: %v1248) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1252 = stablehlo.broadcast_in_dim %v1251, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v1253 = stablehlo.divide %v1252, %v1249 : tensor<32x197x192xf32>
    %v1254 = stablehlo.subtract %v1247, %v1253 : tensor<32x197x192xf32>
    %v1255 = stablehlo.multiply %v1254, %v1254 : tensor<32x197x192xf32>
    %v1256 = stablehlo.reduce(%v1255 init: %v1248) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1257 = stablehlo.broadcast_in_dim %v1256, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v1258 = stablehlo.divide %v1257, %v1249 : tensor<32x197x192xf32>
    %v1259 = stablehlo.add %v1258, %v1250 : tensor<32x197x192xf32>
    %v1260 = stablehlo.rsqrt %v1259 : tensor<32x197x192xf32>
    %v1261 = stablehlo.multiply %v1254, %v1260 : tensor<32x197x192xf32>
    %v1262 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v1263 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v1264 = stablehlo.multiply %v1261, %v1262 : tensor<32x197x192xf32>
    %v1265 = stablehlo.add %v1264, %v1263 : tensor<32x197x192xf32>
    %v1266 = stablehlo.reshape %v1265 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1267 = stablehlo.reshape %v1266 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1268 = stablehlo.broadcast_in_dim %b6_g1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1269 = stablehlo.multiply %v1267, %v1268 : tensor<32x197x192xf32>
    %v1270 = stablehlo.reshape %v1269 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1271 = stablehlo.reshape %v1270 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1272 = stablehlo.broadcast_in_dim %b6_bt1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1273 = stablehlo.add %v1271, %v1272 : tensor<32x197x192xf32>
    %v1274 = stablehlo.reshape %v1273 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1275 = stablehlo.reshape %v1274 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1276 = stablehlo.dot_general %v1275, %b6_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v1277 = stablehlo.broadcast_in_dim %b6_bq, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1278 = stablehlo.add %v1276, %v1277 : tensor<32x197x192xf32>
    %v1279 = stablehlo.reshape %v1278 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1280 = stablehlo.reshape %v1274 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1281 = stablehlo.dot_general %v1280, %b6_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v1282 = stablehlo.broadcast_in_dim %b6_bk, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1283 = stablehlo.add %v1281, %v1282 : tensor<32x197x192xf32>
    %v1284 = stablehlo.reshape %v1283 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1285 = stablehlo.reshape %v1274 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1286 = stablehlo.dot_general %v1285, %b6_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v1287 = stablehlo.broadcast_in_dim %b6_bv, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1288 = stablehlo.add %v1286, %v1287 : tensor<32x197x192xf32>
    %v1289 = stablehlo.reshape %v1288 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1290 = stablehlo.reshape %v1279 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1291 = stablehlo.slice %v1290 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1292 = stablehlo.reshape %v1291 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1293 = stablehlo.reshape %v1284 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1294 = stablehlo.slice %v1293 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1295 = stablehlo.reshape %v1294 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1296 = stablehlo.reshape %v1289 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1297 = stablehlo.slice %v1296 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1298 = stablehlo.reshape %v1297 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1299 = stablehlo.reshape %v1295 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1300 = stablehlo.transpose %v1299, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1301 = stablehlo.reshape %v1300 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1302 = stablehlo.reshape %v1292 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1303 = stablehlo.reshape %v1301 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1304 = stablehlo.dot_general %v1302, %v1303, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1305 = stablehlo.reshape %v1304 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1306 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1307 = stablehlo.multiply %v1305, %v1306 : tensor<32x38809xf32>
    %v1308 = stablehlo.reshape %v1307 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1309 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1310 = stablehlo.exponential %v1308 : tensor<32x197x197xf32>
    %v1311 = stablehlo.reduce(%v1310 init: %v1309) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1312 = stablehlo.broadcast_in_dim %v1311, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1313 = stablehlo.divide %v1310, %v1312 : tensor<32x197x197xf32>
    %v1314 = stablehlo.reshape %v1313 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1315 = stablehlo.reshape %v1314 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1316 = stablehlo.reshape %v1298 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1317 = stablehlo.dot_general %v1315, %v1316, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1318 = stablehlo.reshape %v1317 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1319 = stablehlo.reshape %v1318 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1320 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1321 = stablehlo.pad %v1319, %v1320, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v1322 = stablehlo.reshape %v1321 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1323 = stablehlo.reshape %v1279 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1324 = stablehlo.slice %v1323 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1325 = stablehlo.reshape %v1324 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1326 = stablehlo.reshape %v1284 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1327 = stablehlo.slice %v1326 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1328 = stablehlo.reshape %v1327 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1329 = stablehlo.reshape %v1289 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1330 = stablehlo.slice %v1329 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1331 = stablehlo.reshape %v1330 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1332 = stablehlo.reshape %v1328 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1333 = stablehlo.transpose %v1332, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1334 = stablehlo.reshape %v1333 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1335 = stablehlo.reshape %v1325 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1336 = stablehlo.reshape %v1334 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1337 = stablehlo.dot_general %v1335, %v1336, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1338 = stablehlo.reshape %v1337 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1339 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1340 = stablehlo.multiply %v1338, %v1339 : tensor<32x38809xf32>
    %v1341 = stablehlo.reshape %v1340 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1342 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1343 = stablehlo.exponential %v1341 : tensor<32x197x197xf32>
    %v1344 = stablehlo.reduce(%v1343 init: %v1342) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1345 = stablehlo.broadcast_in_dim %v1344, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1346 = stablehlo.divide %v1343, %v1345 : tensor<32x197x197xf32>
    %v1347 = stablehlo.reshape %v1346 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1348 = stablehlo.reshape %v1347 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1349 = stablehlo.reshape %v1331 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1350 = stablehlo.dot_general %v1348, %v1349, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1351 = stablehlo.reshape %v1350 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1352 = stablehlo.reshape %v1351 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1353 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1354 = stablehlo.pad %v1352, %v1353, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v1355 = stablehlo.reshape %v1354 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1356 = stablehlo.add %v1322, %v1355 : tensor<32x37824xf32>
    %v1357 = stablehlo.reshape %v1279 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1358 = stablehlo.slice %v1357 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1359 = stablehlo.reshape %v1358 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1360 = stablehlo.reshape %v1284 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1361 = stablehlo.slice %v1360 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1362 = stablehlo.reshape %v1361 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1363 = stablehlo.reshape %v1289 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1364 = stablehlo.slice %v1363 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1365 = stablehlo.reshape %v1364 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1366 = stablehlo.reshape %v1362 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1367 = stablehlo.transpose %v1366, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1368 = stablehlo.reshape %v1367 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1369 = stablehlo.reshape %v1359 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1370 = stablehlo.reshape %v1368 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1371 = stablehlo.dot_general %v1369, %v1370, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1372 = stablehlo.reshape %v1371 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1373 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1374 = stablehlo.multiply %v1372, %v1373 : tensor<32x38809xf32>
    %v1375 = stablehlo.reshape %v1374 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1376 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1377 = stablehlo.exponential %v1375 : tensor<32x197x197xf32>
    %v1378 = stablehlo.reduce(%v1377 init: %v1376) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1379 = stablehlo.broadcast_in_dim %v1378, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1380 = stablehlo.divide %v1377, %v1379 : tensor<32x197x197xf32>
    %v1381 = stablehlo.reshape %v1380 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1382 = stablehlo.reshape %v1381 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1383 = stablehlo.reshape %v1365 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1384 = stablehlo.dot_general %v1382, %v1383, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1385 = stablehlo.reshape %v1384 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1386 = stablehlo.reshape %v1385 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1387 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1388 = stablehlo.pad %v1386, %v1387, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v1389 = stablehlo.reshape %v1388 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1390 = stablehlo.add %v1356, %v1389 : tensor<32x37824xf32>
    %v1391 = stablehlo.reshape %v1390 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1392 = stablehlo.dot_general %v1391, %b6_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v1393 = stablehlo.broadcast_in_dim %b6_bo, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1394 = stablehlo.add %v1392, %v1393 : tensor<32x197x192xf32>
    %v1395 = stablehlo.reshape %v1394 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1396 = stablehlo.broadcast_in_dim %dp12, dims = [0] : (tensor<32xf32>) -> tensor<32x37824xf32>
    %v1397 = stablehlo.multiply %v1396, %v1395 : tensor<32x37824xf32>
    %v1398 = stablehlo.add %v1246, %v1397 : tensor<32x37824xf32>
    %v1399 = stablehlo.reshape %v1398 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1400 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1401 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v1402 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v1403 = stablehlo.reduce(%v1399 init: %v1400) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1404 = stablehlo.broadcast_in_dim %v1403, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v1405 = stablehlo.divide %v1404, %v1401 : tensor<32x197x192xf32>
    %v1406 = stablehlo.subtract %v1399, %v1405 : tensor<32x197x192xf32>
    %v1407 = stablehlo.multiply %v1406, %v1406 : tensor<32x197x192xf32>
    %v1408 = stablehlo.reduce(%v1407 init: %v1400) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1409 = stablehlo.broadcast_in_dim %v1408, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v1410 = stablehlo.divide %v1409, %v1401 : tensor<32x197x192xf32>
    %v1411 = stablehlo.add %v1410, %v1402 : tensor<32x197x192xf32>
    %v1412 = stablehlo.rsqrt %v1411 : tensor<32x197x192xf32>
    %v1413 = stablehlo.multiply %v1406, %v1412 : tensor<32x197x192xf32>
    %v1414 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v1415 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v1416 = stablehlo.multiply %v1413, %v1414 : tensor<32x197x192xf32>
    %v1417 = stablehlo.add %v1416, %v1415 : tensor<32x197x192xf32>
    %v1418 = stablehlo.reshape %v1417 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1419 = stablehlo.reshape %v1418 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1420 = stablehlo.broadcast_in_dim %b6_g2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1421 = stablehlo.multiply %v1419, %v1420 : tensor<32x197x192xf32>
    %v1422 = stablehlo.reshape %v1421 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1423 = stablehlo.reshape %v1422 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1424 = stablehlo.broadcast_in_dim %b6_bt2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1425 = stablehlo.add %v1423, %v1424 : tensor<32x197x192xf32>
    %v1426 = stablehlo.reshape %v1425 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1427 = stablehlo.reshape %v1426 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1428 = stablehlo.dot_general %v1427, %b6_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x768xf32>) -> tensor<32x197x768xf32>
    %v1429 = stablehlo.broadcast_in_dim %b6_bfc1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v1430 = stablehlo.add %v1428, %v1429 : tensor<32x197x768xf32>
    %v1431 = stablehlo.reshape %v1430 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1432 = stablehlo.multiply %v1431, %v1431 : tensor<32x151296xf32>
    %v1433 = stablehlo.multiply %v1432, %v1431 : tensor<32x151296xf32>
    %v1434 = stablehlo.constant dense<0.044715> : tensor<32x151296xf32>
    %v1435 = stablehlo.multiply %v1434, %v1433 : tensor<32x151296xf32>
    %v1436 = stablehlo.add %v1431, %v1435 : tensor<32x151296xf32>
    %v1437 = stablehlo.constant dense<0.7978845608028654> : tensor<32x151296xf32>
    %v1438 = stablehlo.multiply %v1437, %v1436 : tensor<32x151296xf32>
    %v1439 = stablehlo.tanh %v1438 : tensor<32x151296xf32>
    %v1440 = stablehlo.constant dense<1.0> : tensor<32x151296xf32>
    %v1441 = stablehlo.add %v1440, %v1439 : tensor<32x151296xf32>
    %v1442 = stablehlo.constant dense<0.5> : tensor<32x151296xf32>
    %v1443 = stablehlo.multiply %v1442, %v1431 : tensor<32x151296xf32>
    %v1444 = stablehlo.multiply %v1443, %v1441 : tensor<32x151296xf32>
    %v1445 = stablehlo.reshape %v1444 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1446 = stablehlo.dot_general %v1445, %b6_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x192xf32>) -> tensor<32x197x192xf32>
    %v1447 = stablehlo.broadcast_in_dim %b6_bfc2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1448 = stablehlo.add %v1446, %v1447 : tensor<32x197x192xf32>
    %v1449 = stablehlo.reshape %v1448 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1450 = stablehlo.broadcast_in_dim %dp13, dims = [0] : (tensor<32xf32>) -> tensor<32x37824xf32>
    %v1451 = stablehlo.multiply %v1450, %v1449 : tensor<32x37824xf32>
    %v1452 = stablehlo.add %v1398, %v1451 : tensor<32x37824xf32>
    %v1453 = stablehlo.reshape %v1452 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1454 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1455 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v1456 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v1457 = stablehlo.reduce(%v1453 init: %v1454) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1458 = stablehlo.broadcast_in_dim %v1457, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v1459 = stablehlo.divide %v1458, %v1455 : tensor<32x197x192xf32>
    %v1460 = stablehlo.subtract %v1453, %v1459 : tensor<32x197x192xf32>
    %v1461 = stablehlo.multiply %v1460, %v1460 : tensor<32x197x192xf32>
    %v1462 = stablehlo.reduce(%v1461 init: %v1454) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1463 = stablehlo.broadcast_in_dim %v1462, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v1464 = stablehlo.divide %v1463, %v1455 : tensor<32x197x192xf32>
    %v1465 = stablehlo.add %v1464, %v1456 : tensor<32x197x192xf32>
    %v1466 = stablehlo.rsqrt %v1465 : tensor<32x197x192xf32>
    %v1467 = stablehlo.multiply %v1460, %v1466 : tensor<32x197x192xf32>
    %v1468 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v1469 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v1470 = stablehlo.multiply %v1467, %v1468 : tensor<32x197x192xf32>
    %v1471 = stablehlo.add %v1470, %v1469 : tensor<32x197x192xf32>
    %v1472 = stablehlo.reshape %v1471 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1473 = stablehlo.reshape %v1472 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1474 = stablehlo.broadcast_in_dim %b7_g1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1475 = stablehlo.multiply %v1473, %v1474 : tensor<32x197x192xf32>
    %v1476 = stablehlo.reshape %v1475 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1477 = stablehlo.reshape %v1476 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1478 = stablehlo.broadcast_in_dim %b7_bt1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1479 = stablehlo.add %v1477, %v1478 : tensor<32x197x192xf32>
    %v1480 = stablehlo.reshape %v1479 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1481 = stablehlo.reshape %v1480 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1482 = stablehlo.dot_general %v1481, %b7_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v1483 = stablehlo.broadcast_in_dim %b7_bq, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1484 = stablehlo.add %v1482, %v1483 : tensor<32x197x192xf32>
    %v1485 = stablehlo.reshape %v1484 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1486 = stablehlo.reshape %v1480 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1487 = stablehlo.dot_general %v1486, %b7_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v1488 = stablehlo.broadcast_in_dim %b7_bk, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1489 = stablehlo.add %v1487, %v1488 : tensor<32x197x192xf32>
    %v1490 = stablehlo.reshape %v1489 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1491 = stablehlo.reshape %v1480 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1492 = stablehlo.dot_general %v1491, %b7_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v1493 = stablehlo.broadcast_in_dim %b7_bv, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1494 = stablehlo.add %v1492, %v1493 : tensor<32x197x192xf32>
    %v1495 = stablehlo.reshape %v1494 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1496 = stablehlo.reshape %v1485 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1497 = stablehlo.slice %v1496 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1498 = stablehlo.reshape %v1497 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1499 = stablehlo.reshape %v1490 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1500 = stablehlo.slice %v1499 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1501 = stablehlo.reshape %v1500 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1502 = stablehlo.reshape %v1495 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1503 = stablehlo.slice %v1502 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1504 = stablehlo.reshape %v1503 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1505 = stablehlo.reshape %v1501 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1506 = stablehlo.transpose %v1505, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1507 = stablehlo.reshape %v1506 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1508 = stablehlo.reshape %v1498 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1509 = stablehlo.reshape %v1507 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1510 = stablehlo.dot_general %v1508, %v1509, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1511 = stablehlo.reshape %v1510 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1512 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1513 = stablehlo.multiply %v1511, %v1512 : tensor<32x38809xf32>
    %v1514 = stablehlo.reshape %v1513 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1515 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1516 = stablehlo.exponential %v1514 : tensor<32x197x197xf32>
    %v1517 = stablehlo.reduce(%v1516 init: %v1515) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1518 = stablehlo.broadcast_in_dim %v1517, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1519 = stablehlo.divide %v1516, %v1518 : tensor<32x197x197xf32>
    %v1520 = stablehlo.reshape %v1519 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1521 = stablehlo.reshape %v1520 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1522 = stablehlo.reshape %v1504 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1523 = stablehlo.dot_general %v1521, %v1522, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1524 = stablehlo.reshape %v1523 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1525 = stablehlo.reshape %v1524 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1526 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1527 = stablehlo.pad %v1525, %v1526, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v1528 = stablehlo.reshape %v1527 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1529 = stablehlo.reshape %v1485 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1530 = stablehlo.slice %v1529 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1531 = stablehlo.reshape %v1530 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1532 = stablehlo.reshape %v1490 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1533 = stablehlo.slice %v1532 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1534 = stablehlo.reshape %v1533 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1535 = stablehlo.reshape %v1495 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1536 = stablehlo.slice %v1535 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1537 = stablehlo.reshape %v1536 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1538 = stablehlo.reshape %v1534 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1539 = stablehlo.transpose %v1538, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1540 = stablehlo.reshape %v1539 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1541 = stablehlo.reshape %v1531 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1542 = stablehlo.reshape %v1540 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1543 = stablehlo.dot_general %v1541, %v1542, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1544 = stablehlo.reshape %v1543 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1545 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1546 = stablehlo.multiply %v1544, %v1545 : tensor<32x38809xf32>
    %v1547 = stablehlo.reshape %v1546 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1548 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1549 = stablehlo.exponential %v1547 : tensor<32x197x197xf32>
    %v1550 = stablehlo.reduce(%v1549 init: %v1548) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1551 = stablehlo.broadcast_in_dim %v1550, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1552 = stablehlo.divide %v1549, %v1551 : tensor<32x197x197xf32>
    %v1553 = stablehlo.reshape %v1552 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1554 = stablehlo.reshape %v1553 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1555 = stablehlo.reshape %v1537 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1556 = stablehlo.dot_general %v1554, %v1555, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1557 = stablehlo.reshape %v1556 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1558 = stablehlo.reshape %v1557 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1559 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1560 = stablehlo.pad %v1558, %v1559, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v1561 = stablehlo.reshape %v1560 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1562 = stablehlo.add %v1528, %v1561 : tensor<32x37824xf32>
    %v1563 = stablehlo.reshape %v1485 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1564 = stablehlo.slice %v1563 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1565 = stablehlo.reshape %v1564 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1566 = stablehlo.reshape %v1490 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1567 = stablehlo.slice %v1566 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1568 = stablehlo.reshape %v1567 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1569 = stablehlo.reshape %v1495 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1570 = stablehlo.slice %v1569 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1571 = stablehlo.reshape %v1570 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1572 = stablehlo.reshape %v1568 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1573 = stablehlo.transpose %v1572, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1574 = stablehlo.reshape %v1573 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1575 = stablehlo.reshape %v1565 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1576 = stablehlo.reshape %v1574 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1577 = stablehlo.dot_general %v1575, %v1576, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1578 = stablehlo.reshape %v1577 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1579 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1580 = stablehlo.multiply %v1578, %v1579 : tensor<32x38809xf32>
    %v1581 = stablehlo.reshape %v1580 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1582 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1583 = stablehlo.exponential %v1581 : tensor<32x197x197xf32>
    %v1584 = stablehlo.reduce(%v1583 init: %v1582) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1585 = stablehlo.broadcast_in_dim %v1584, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1586 = stablehlo.divide %v1583, %v1585 : tensor<32x197x197xf32>
    %v1587 = stablehlo.reshape %v1586 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1588 = stablehlo.reshape %v1587 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1589 = stablehlo.reshape %v1571 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1590 = stablehlo.dot_general %v1588, %v1589, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1591 = stablehlo.reshape %v1590 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1592 = stablehlo.reshape %v1591 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1593 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1594 = stablehlo.pad %v1592, %v1593, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v1595 = stablehlo.reshape %v1594 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1596 = stablehlo.add %v1562, %v1595 : tensor<32x37824xf32>
    %v1597 = stablehlo.reshape %v1596 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1598 = stablehlo.dot_general %v1597, %b7_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v1599 = stablehlo.broadcast_in_dim %b7_bo, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1600 = stablehlo.add %v1598, %v1599 : tensor<32x197x192xf32>
    %v1601 = stablehlo.reshape %v1600 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1602 = stablehlo.broadcast_in_dim %dp14, dims = [0] : (tensor<32xf32>) -> tensor<32x37824xf32>
    %v1603 = stablehlo.multiply %v1602, %v1601 : tensor<32x37824xf32>
    %v1604 = stablehlo.add %v1452, %v1603 : tensor<32x37824xf32>
    %v1605 = stablehlo.reshape %v1604 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1606 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1607 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v1608 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v1609 = stablehlo.reduce(%v1605 init: %v1606) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1610 = stablehlo.broadcast_in_dim %v1609, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v1611 = stablehlo.divide %v1610, %v1607 : tensor<32x197x192xf32>
    %v1612 = stablehlo.subtract %v1605, %v1611 : tensor<32x197x192xf32>
    %v1613 = stablehlo.multiply %v1612, %v1612 : tensor<32x197x192xf32>
    %v1614 = stablehlo.reduce(%v1613 init: %v1606) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1615 = stablehlo.broadcast_in_dim %v1614, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v1616 = stablehlo.divide %v1615, %v1607 : tensor<32x197x192xf32>
    %v1617 = stablehlo.add %v1616, %v1608 : tensor<32x197x192xf32>
    %v1618 = stablehlo.rsqrt %v1617 : tensor<32x197x192xf32>
    %v1619 = stablehlo.multiply %v1612, %v1618 : tensor<32x197x192xf32>
    %v1620 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v1621 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v1622 = stablehlo.multiply %v1619, %v1620 : tensor<32x197x192xf32>
    %v1623 = stablehlo.add %v1622, %v1621 : tensor<32x197x192xf32>
    %v1624 = stablehlo.reshape %v1623 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1625 = stablehlo.reshape %v1624 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1626 = stablehlo.broadcast_in_dim %b7_g2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1627 = stablehlo.multiply %v1625, %v1626 : tensor<32x197x192xf32>
    %v1628 = stablehlo.reshape %v1627 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1629 = stablehlo.reshape %v1628 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1630 = stablehlo.broadcast_in_dim %b7_bt2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1631 = stablehlo.add %v1629, %v1630 : tensor<32x197x192xf32>
    %v1632 = stablehlo.reshape %v1631 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1633 = stablehlo.reshape %v1632 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1634 = stablehlo.dot_general %v1633, %b7_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x768xf32>) -> tensor<32x197x768xf32>
    %v1635 = stablehlo.broadcast_in_dim %b7_bfc1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v1636 = stablehlo.add %v1634, %v1635 : tensor<32x197x768xf32>
    %v1637 = stablehlo.reshape %v1636 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1638 = stablehlo.multiply %v1637, %v1637 : tensor<32x151296xf32>
    %v1639 = stablehlo.multiply %v1638, %v1637 : tensor<32x151296xf32>
    %v1640 = stablehlo.constant dense<0.044715> : tensor<32x151296xf32>
    %v1641 = stablehlo.multiply %v1640, %v1639 : tensor<32x151296xf32>
    %v1642 = stablehlo.add %v1637, %v1641 : tensor<32x151296xf32>
    %v1643 = stablehlo.constant dense<0.7978845608028654> : tensor<32x151296xf32>
    %v1644 = stablehlo.multiply %v1643, %v1642 : tensor<32x151296xf32>
    %v1645 = stablehlo.tanh %v1644 : tensor<32x151296xf32>
    %v1646 = stablehlo.constant dense<1.0> : tensor<32x151296xf32>
    %v1647 = stablehlo.add %v1646, %v1645 : tensor<32x151296xf32>
    %v1648 = stablehlo.constant dense<0.5> : tensor<32x151296xf32>
    %v1649 = stablehlo.multiply %v1648, %v1637 : tensor<32x151296xf32>
    %v1650 = stablehlo.multiply %v1649, %v1647 : tensor<32x151296xf32>
    %v1651 = stablehlo.reshape %v1650 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1652 = stablehlo.dot_general %v1651, %b7_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x192xf32>) -> tensor<32x197x192xf32>
    %v1653 = stablehlo.broadcast_in_dim %b7_bfc2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1654 = stablehlo.add %v1652, %v1653 : tensor<32x197x192xf32>
    %v1655 = stablehlo.reshape %v1654 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1656 = stablehlo.broadcast_in_dim %dp15, dims = [0] : (tensor<32xf32>) -> tensor<32x37824xf32>
    %v1657 = stablehlo.multiply %v1656, %v1655 : tensor<32x37824xf32>
    %v1658 = stablehlo.add %v1604, %v1657 : tensor<32x37824xf32>
    %v1659 = stablehlo.reshape %v1658 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1660 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1661 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v1662 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v1663 = stablehlo.reduce(%v1659 init: %v1660) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1664 = stablehlo.broadcast_in_dim %v1663, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v1665 = stablehlo.divide %v1664, %v1661 : tensor<32x197x192xf32>
    %v1666 = stablehlo.subtract %v1659, %v1665 : tensor<32x197x192xf32>
    %v1667 = stablehlo.multiply %v1666, %v1666 : tensor<32x197x192xf32>
    %v1668 = stablehlo.reduce(%v1667 init: %v1660) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1669 = stablehlo.broadcast_in_dim %v1668, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v1670 = stablehlo.divide %v1669, %v1661 : tensor<32x197x192xf32>
    %v1671 = stablehlo.add %v1670, %v1662 : tensor<32x197x192xf32>
    %v1672 = stablehlo.rsqrt %v1671 : tensor<32x197x192xf32>
    %v1673 = stablehlo.multiply %v1666, %v1672 : tensor<32x197x192xf32>
    %v1674 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v1675 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v1676 = stablehlo.multiply %v1673, %v1674 : tensor<32x197x192xf32>
    %v1677 = stablehlo.add %v1676, %v1675 : tensor<32x197x192xf32>
    %v1678 = stablehlo.reshape %v1677 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1679 = stablehlo.reshape %v1678 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1680 = stablehlo.broadcast_in_dim %b8_g1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1681 = stablehlo.multiply %v1679, %v1680 : tensor<32x197x192xf32>
    %v1682 = stablehlo.reshape %v1681 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1683 = stablehlo.reshape %v1682 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1684 = stablehlo.broadcast_in_dim %b8_bt1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1685 = stablehlo.add %v1683, %v1684 : tensor<32x197x192xf32>
    %v1686 = stablehlo.reshape %v1685 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1687 = stablehlo.reshape %v1686 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1688 = stablehlo.dot_general %v1687, %b8_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v1689 = stablehlo.broadcast_in_dim %b8_bq, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1690 = stablehlo.add %v1688, %v1689 : tensor<32x197x192xf32>
    %v1691 = stablehlo.reshape %v1690 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1692 = stablehlo.reshape %v1686 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1693 = stablehlo.dot_general %v1692, %b8_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v1694 = stablehlo.broadcast_in_dim %b8_bk, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1695 = stablehlo.add %v1693, %v1694 : tensor<32x197x192xf32>
    %v1696 = stablehlo.reshape %v1695 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1697 = stablehlo.reshape %v1686 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1698 = stablehlo.dot_general %v1697, %b8_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v1699 = stablehlo.broadcast_in_dim %b8_bv, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1700 = stablehlo.add %v1698, %v1699 : tensor<32x197x192xf32>
    %v1701 = stablehlo.reshape %v1700 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1702 = stablehlo.reshape %v1691 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1703 = stablehlo.slice %v1702 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1704 = stablehlo.reshape %v1703 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1705 = stablehlo.reshape %v1696 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1706 = stablehlo.slice %v1705 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1707 = stablehlo.reshape %v1706 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1708 = stablehlo.reshape %v1701 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1709 = stablehlo.slice %v1708 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1710 = stablehlo.reshape %v1709 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1711 = stablehlo.reshape %v1707 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1712 = stablehlo.transpose %v1711, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1713 = stablehlo.reshape %v1712 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1714 = stablehlo.reshape %v1704 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1715 = stablehlo.reshape %v1713 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1716 = stablehlo.dot_general %v1714, %v1715, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1717 = stablehlo.reshape %v1716 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1718 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1719 = stablehlo.multiply %v1717, %v1718 : tensor<32x38809xf32>
    %v1720 = stablehlo.reshape %v1719 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1721 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1722 = stablehlo.exponential %v1720 : tensor<32x197x197xf32>
    %v1723 = stablehlo.reduce(%v1722 init: %v1721) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1724 = stablehlo.broadcast_in_dim %v1723, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1725 = stablehlo.divide %v1722, %v1724 : tensor<32x197x197xf32>
    %v1726 = stablehlo.reshape %v1725 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1727 = stablehlo.reshape %v1726 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1728 = stablehlo.reshape %v1710 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1729 = stablehlo.dot_general %v1727, %v1728, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1730 = stablehlo.reshape %v1729 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1731 = stablehlo.reshape %v1730 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1732 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1733 = stablehlo.pad %v1731, %v1732, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v1734 = stablehlo.reshape %v1733 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1735 = stablehlo.reshape %v1691 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1736 = stablehlo.slice %v1735 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1737 = stablehlo.reshape %v1736 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1738 = stablehlo.reshape %v1696 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1739 = stablehlo.slice %v1738 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1740 = stablehlo.reshape %v1739 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1741 = stablehlo.reshape %v1701 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1742 = stablehlo.slice %v1741 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1743 = stablehlo.reshape %v1742 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1744 = stablehlo.reshape %v1740 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1745 = stablehlo.transpose %v1744, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1746 = stablehlo.reshape %v1745 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1747 = stablehlo.reshape %v1737 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1748 = stablehlo.reshape %v1746 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1749 = stablehlo.dot_general %v1747, %v1748, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1750 = stablehlo.reshape %v1749 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1751 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1752 = stablehlo.multiply %v1750, %v1751 : tensor<32x38809xf32>
    %v1753 = stablehlo.reshape %v1752 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1754 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1755 = stablehlo.exponential %v1753 : tensor<32x197x197xf32>
    %v1756 = stablehlo.reduce(%v1755 init: %v1754) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1757 = stablehlo.broadcast_in_dim %v1756, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1758 = stablehlo.divide %v1755, %v1757 : tensor<32x197x197xf32>
    %v1759 = stablehlo.reshape %v1758 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1760 = stablehlo.reshape %v1759 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1761 = stablehlo.reshape %v1743 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1762 = stablehlo.dot_general %v1760, %v1761, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1763 = stablehlo.reshape %v1762 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1764 = stablehlo.reshape %v1763 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1765 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1766 = stablehlo.pad %v1764, %v1765, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v1767 = stablehlo.reshape %v1766 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1768 = stablehlo.add %v1734, %v1767 : tensor<32x37824xf32>
    %v1769 = stablehlo.reshape %v1691 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1770 = stablehlo.slice %v1769 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1771 = stablehlo.reshape %v1770 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1772 = stablehlo.reshape %v1696 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1773 = stablehlo.slice %v1772 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1774 = stablehlo.reshape %v1773 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1775 = stablehlo.reshape %v1701 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1776 = stablehlo.slice %v1775 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1777 = stablehlo.reshape %v1776 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1778 = stablehlo.reshape %v1774 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1779 = stablehlo.transpose %v1778, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1780 = stablehlo.reshape %v1779 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1781 = stablehlo.reshape %v1771 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1782 = stablehlo.reshape %v1780 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1783 = stablehlo.dot_general %v1781, %v1782, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1784 = stablehlo.reshape %v1783 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1785 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1786 = stablehlo.multiply %v1784, %v1785 : tensor<32x38809xf32>
    %v1787 = stablehlo.reshape %v1786 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1788 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1789 = stablehlo.exponential %v1787 : tensor<32x197x197xf32>
    %v1790 = stablehlo.reduce(%v1789 init: %v1788) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1791 = stablehlo.broadcast_in_dim %v1790, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1792 = stablehlo.divide %v1789, %v1791 : tensor<32x197x197xf32>
    %v1793 = stablehlo.reshape %v1792 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1794 = stablehlo.reshape %v1793 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1795 = stablehlo.reshape %v1777 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1796 = stablehlo.dot_general %v1794, %v1795, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1797 = stablehlo.reshape %v1796 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1798 = stablehlo.reshape %v1797 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1799 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1800 = stablehlo.pad %v1798, %v1799, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v1801 = stablehlo.reshape %v1800 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1802 = stablehlo.add %v1768, %v1801 : tensor<32x37824xf32>
    %v1803 = stablehlo.reshape %v1802 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1804 = stablehlo.dot_general %v1803, %b8_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v1805 = stablehlo.broadcast_in_dim %b8_bo, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1806 = stablehlo.add %v1804, %v1805 : tensor<32x197x192xf32>
    %v1807 = stablehlo.reshape %v1806 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1808 = stablehlo.broadcast_in_dim %dp16, dims = [0] : (tensor<32xf32>) -> tensor<32x37824xf32>
    %v1809 = stablehlo.multiply %v1808, %v1807 : tensor<32x37824xf32>
    %v1810 = stablehlo.add %v1658, %v1809 : tensor<32x37824xf32>
    %v1811 = stablehlo.reshape %v1810 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1812 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1813 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v1814 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v1815 = stablehlo.reduce(%v1811 init: %v1812) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1816 = stablehlo.broadcast_in_dim %v1815, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v1817 = stablehlo.divide %v1816, %v1813 : tensor<32x197x192xf32>
    %v1818 = stablehlo.subtract %v1811, %v1817 : tensor<32x197x192xf32>
    %v1819 = stablehlo.multiply %v1818, %v1818 : tensor<32x197x192xf32>
    %v1820 = stablehlo.reduce(%v1819 init: %v1812) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1821 = stablehlo.broadcast_in_dim %v1820, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v1822 = stablehlo.divide %v1821, %v1813 : tensor<32x197x192xf32>
    %v1823 = stablehlo.add %v1822, %v1814 : tensor<32x197x192xf32>
    %v1824 = stablehlo.rsqrt %v1823 : tensor<32x197x192xf32>
    %v1825 = stablehlo.multiply %v1818, %v1824 : tensor<32x197x192xf32>
    %v1826 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v1827 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v1828 = stablehlo.multiply %v1825, %v1826 : tensor<32x197x192xf32>
    %v1829 = stablehlo.add %v1828, %v1827 : tensor<32x197x192xf32>
    %v1830 = stablehlo.reshape %v1829 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1831 = stablehlo.reshape %v1830 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1832 = stablehlo.broadcast_in_dim %b8_g2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1833 = stablehlo.multiply %v1831, %v1832 : tensor<32x197x192xf32>
    %v1834 = stablehlo.reshape %v1833 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1835 = stablehlo.reshape %v1834 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1836 = stablehlo.broadcast_in_dim %b8_bt2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1837 = stablehlo.add %v1835, %v1836 : tensor<32x197x192xf32>
    %v1838 = stablehlo.reshape %v1837 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1839 = stablehlo.reshape %v1838 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1840 = stablehlo.dot_general %v1839, %b8_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x768xf32>) -> tensor<32x197x768xf32>
    %v1841 = stablehlo.broadcast_in_dim %b8_bfc1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v1842 = stablehlo.add %v1840, %v1841 : tensor<32x197x768xf32>
    %v1843 = stablehlo.reshape %v1842 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1844 = stablehlo.multiply %v1843, %v1843 : tensor<32x151296xf32>
    %v1845 = stablehlo.multiply %v1844, %v1843 : tensor<32x151296xf32>
    %v1846 = stablehlo.constant dense<0.044715> : tensor<32x151296xf32>
    %v1847 = stablehlo.multiply %v1846, %v1845 : tensor<32x151296xf32>
    %v1848 = stablehlo.add %v1843, %v1847 : tensor<32x151296xf32>
    %v1849 = stablehlo.constant dense<0.7978845608028654> : tensor<32x151296xf32>
    %v1850 = stablehlo.multiply %v1849, %v1848 : tensor<32x151296xf32>
    %v1851 = stablehlo.tanh %v1850 : tensor<32x151296xf32>
    %v1852 = stablehlo.constant dense<1.0> : tensor<32x151296xf32>
    %v1853 = stablehlo.add %v1852, %v1851 : tensor<32x151296xf32>
    %v1854 = stablehlo.constant dense<0.5> : tensor<32x151296xf32>
    %v1855 = stablehlo.multiply %v1854, %v1843 : tensor<32x151296xf32>
    %v1856 = stablehlo.multiply %v1855, %v1853 : tensor<32x151296xf32>
    %v1857 = stablehlo.reshape %v1856 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1858 = stablehlo.dot_general %v1857, %b8_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x192xf32>) -> tensor<32x197x192xf32>
    %v1859 = stablehlo.broadcast_in_dim %b8_bfc2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1860 = stablehlo.add %v1858, %v1859 : tensor<32x197x192xf32>
    %v1861 = stablehlo.reshape %v1860 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1862 = stablehlo.broadcast_in_dim %dp17, dims = [0] : (tensor<32xf32>) -> tensor<32x37824xf32>
    %v1863 = stablehlo.multiply %v1862, %v1861 : tensor<32x37824xf32>
    %v1864 = stablehlo.add %v1810, %v1863 : tensor<32x37824xf32>
    %v1865 = stablehlo.reshape %v1864 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1866 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1867 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v1868 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v1869 = stablehlo.reduce(%v1865 init: %v1866) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1870 = stablehlo.broadcast_in_dim %v1869, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v1871 = stablehlo.divide %v1870, %v1867 : tensor<32x197x192xf32>
    %v1872 = stablehlo.subtract %v1865, %v1871 : tensor<32x197x192xf32>
    %v1873 = stablehlo.multiply %v1872, %v1872 : tensor<32x197x192xf32>
    %v1874 = stablehlo.reduce(%v1873 init: %v1866) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1875 = stablehlo.broadcast_in_dim %v1874, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v1876 = stablehlo.divide %v1875, %v1867 : tensor<32x197x192xf32>
    %v1877 = stablehlo.add %v1876, %v1868 : tensor<32x197x192xf32>
    %v1878 = stablehlo.rsqrt %v1877 : tensor<32x197x192xf32>
    %v1879 = stablehlo.multiply %v1872, %v1878 : tensor<32x197x192xf32>
    %v1880 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v1881 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v1882 = stablehlo.multiply %v1879, %v1880 : tensor<32x197x192xf32>
    %v1883 = stablehlo.add %v1882, %v1881 : tensor<32x197x192xf32>
    %v1884 = stablehlo.reshape %v1883 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1885 = stablehlo.reshape %v1884 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1886 = stablehlo.broadcast_in_dim %b9_g1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1887 = stablehlo.multiply %v1885, %v1886 : tensor<32x197x192xf32>
    %v1888 = stablehlo.reshape %v1887 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1889 = stablehlo.reshape %v1888 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1890 = stablehlo.broadcast_in_dim %b9_bt1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1891 = stablehlo.add %v1889, %v1890 : tensor<32x197x192xf32>
    %v1892 = stablehlo.reshape %v1891 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1893 = stablehlo.reshape %v1892 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1894 = stablehlo.dot_general %v1893, %b9_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v1895 = stablehlo.broadcast_in_dim %b9_bq, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1896 = stablehlo.add %v1894, %v1895 : tensor<32x197x192xf32>
    %v1897 = stablehlo.reshape %v1896 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1898 = stablehlo.reshape %v1892 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1899 = stablehlo.dot_general %v1898, %b9_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v1900 = stablehlo.broadcast_in_dim %b9_bk, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1901 = stablehlo.add %v1899, %v1900 : tensor<32x197x192xf32>
    %v1902 = stablehlo.reshape %v1901 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1903 = stablehlo.reshape %v1892 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1904 = stablehlo.dot_general %v1903, %b9_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v1905 = stablehlo.broadcast_in_dim %b9_bv, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1906 = stablehlo.add %v1904, %v1905 : tensor<32x197x192xf32>
    %v1907 = stablehlo.reshape %v1906 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1908 = stablehlo.reshape %v1897 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1909 = stablehlo.slice %v1908 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1910 = stablehlo.reshape %v1909 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1911 = stablehlo.reshape %v1902 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1912 = stablehlo.slice %v1911 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1913 = stablehlo.reshape %v1912 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1914 = stablehlo.reshape %v1907 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1915 = stablehlo.slice %v1914 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1916 = stablehlo.reshape %v1915 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1917 = stablehlo.reshape %v1913 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1918 = stablehlo.transpose %v1917, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1919 = stablehlo.reshape %v1918 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1920 = stablehlo.reshape %v1910 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1921 = stablehlo.reshape %v1919 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1922 = stablehlo.dot_general %v1920, %v1921, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1923 = stablehlo.reshape %v1922 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1924 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1925 = stablehlo.multiply %v1923, %v1924 : tensor<32x38809xf32>
    %v1926 = stablehlo.reshape %v1925 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1927 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1928 = stablehlo.exponential %v1926 : tensor<32x197x197xf32>
    %v1929 = stablehlo.reduce(%v1928 init: %v1927) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1930 = stablehlo.broadcast_in_dim %v1929, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1931 = stablehlo.divide %v1928, %v1930 : tensor<32x197x197xf32>
    %v1932 = stablehlo.reshape %v1931 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1933 = stablehlo.reshape %v1932 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1934 = stablehlo.reshape %v1916 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1935 = stablehlo.dot_general %v1933, %v1934, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1936 = stablehlo.reshape %v1935 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1937 = stablehlo.reshape %v1936 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1938 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1939 = stablehlo.pad %v1937, %v1938, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v1940 = stablehlo.reshape %v1939 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1941 = stablehlo.reshape %v1897 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1942 = stablehlo.slice %v1941 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1943 = stablehlo.reshape %v1942 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1944 = stablehlo.reshape %v1902 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1945 = stablehlo.slice %v1944 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1946 = stablehlo.reshape %v1945 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1947 = stablehlo.reshape %v1907 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1948 = stablehlo.slice %v1947 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1949 = stablehlo.reshape %v1948 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1950 = stablehlo.reshape %v1946 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1951 = stablehlo.transpose %v1950, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1952 = stablehlo.reshape %v1951 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1953 = stablehlo.reshape %v1943 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1954 = stablehlo.reshape %v1952 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1955 = stablehlo.dot_general %v1953, %v1954, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1956 = stablehlo.reshape %v1955 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1957 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1958 = stablehlo.multiply %v1956, %v1957 : tensor<32x38809xf32>
    %v1959 = stablehlo.reshape %v1958 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1960 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1961 = stablehlo.exponential %v1959 : tensor<32x197x197xf32>
    %v1962 = stablehlo.reduce(%v1961 init: %v1960) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1963 = stablehlo.broadcast_in_dim %v1962, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1964 = stablehlo.divide %v1961, %v1963 : tensor<32x197x197xf32>
    %v1965 = stablehlo.reshape %v1964 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1966 = stablehlo.reshape %v1965 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1967 = stablehlo.reshape %v1949 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1968 = stablehlo.dot_general %v1966, %v1967, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1969 = stablehlo.reshape %v1968 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1970 = stablehlo.reshape %v1969 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1971 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1972 = stablehlo.pad %v1970, %v1971, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v1973 = stablehlo.reshape %v1972 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1974 = stablehlo.add %v1940, %v1973 : tensor<32x37824xf32>
    %v1975 = stablehlo.reshape %v1897 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1976 = stablehlo.slice %v1975 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1977 = stablehlo.reshape %v1976 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1978 = stablehlo.reshape %v1902 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1979 = stablehlo.slice %v1978 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1980 = stablehlo.reshape %v1979 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1981 = stablehlo.reshape %v1907 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1982 = stablehlo.slice %v1981 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1983 = stablehlo.reshape %v1982 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1984 = stablehlo.reshape %v1980 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1985 = stablehlo.transpose %v1984, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1986 = stablehlo.reshape %v1985 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1987 = stablehlo.reshape %v1977 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1988 = stablehlo.reshape %v1986 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1989 = stablehlo.dot_general %v1987, %v1988, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1990 = stablehlo.reshape %v1989 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1991 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1992 = stablehlo.multiply %v1990, %v1991 : tensor<32x38809xf32>
    %v1993 = stablehlo.reshape %v1992 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1994 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1995 = stablehlo.exponential %v1993 : tensor<32x197x197xf32>
    %v1996 = stablehlo.reduce(%v1995 init: %v1994) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1997 = stablehlo.broadcast_in_dim %v1996, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1998 = stablehlo.divide %v1995, %v1997 : tensor<32x197x197xf32>
    %v1999 = stablehlo.reshape %v1998 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2000 = stablehlo.reshape %v1999 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2001 = stablehlo.reshape %v1983 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2002 = stablehlo.dot_general %v2000, %v2001, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2003 = stablehlo.reshape %v2002 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2004 = stablehlo.reshape %v2003 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2005 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2006 = stablehlo.pad %v2004, %v2005, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v2007 = stablehlo.reshape %v2006 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2008 = stablehlo.add %v1974, %v2007 : tensor<32x37824xf32>
    %v2009 = stablehlo.reshape %v2008 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2010 = stablehlo.dot_general %v2009, %b9_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v2011 = stablehlo.broadcast_in_dim %b9_bo, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v2012 = stablehlo.add %v2010, %v2011 : tensor<32x197x192xf32>
    %v2013 = stablehlo.reshape %v2012 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2014 = stablehlo.broadcast_in_dim %dp18, dims = [0] : (tensor<32xf32>) -> tensor<32x37824xf32>
    %v2015 = stablehlo.multiply %v2014, %v2013 : tensor<32x37824xf32>
    %v2016 = stablehlo.add %v1864, %v2015 : tensor<32x37824xf32>
    %v2017 = stablehlo.reshape %v2016 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2018 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2019 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v2020 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v2021 = stablehlo.reduce(%v2017 init: %v2018) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2022 = stablehlo.broadcast_in_dim %v2021, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v2023 = stablehlo.divide %v2022, %v2019 : tensor<32x197x192xf32>
    %v2024 = stablehlo.subtract %v2017, %v2023 : tensor<32x197x192xf32>
    %v2025 = stablehlo.multiply %v2024, %v2024 : tensor<32x197x192xf32>
    %v2026 = stablehlo.reduce(%v2025 init: %v2018) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2027 = stablehlo.broadcast_in_dim %v2026, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v2028 = stablehlo.divide %v2027, %v2019 : tensor<32x197x192xf32>
    %v2029 = stablehlo.add %v2028, %v2020 : tensor<32x197x192xf32>
    %v2030 = stablehlo.rsqrt %v2029 : tensor<32x197x192xf32>
    %v2031 = stablehlo.multiply %v2024, %v2030 : tensor<32x197x192xf32>
    %v2032 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v2033 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v2034 = stablehlo.multiply %v2031, %v2032 : tensor<32x197x192xf32>
    %v2035 = stablehlo.add %v2034, %v2033 : tensor<32x197x192xf32>
    %v2036 = stablehlo.reshape %v2035 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2037 = stablehlo.reshape %v2036 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2038 = stablehlo.broadcast_in_dim %b9_g2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v2039 = stablehlo.multiply %v2037, %v2038 : tensor<32x197x192xf32>
    %v2040 = stablehlo.reshape %v2039 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2041 = stablehlo.reshape %v2040 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2042 = stablehlo.broadcast_in_dim %b9_bt2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v2043 = stablehlo.add %v2041, %v2042 : tensor<32x197x192xf32>
    %v2044 = stablehlo.reshape %v2043 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2045 = stablehlo.reshape %v2044 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2046 = stablehlo.dot_general %v2045, %b9_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x768xf32>) -> tensor<32x197x768xf32>
    %v2047 = stablehlo.broadcast_in_dim %b9_bfc1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v2048 = stablehlo.add %v2046, %v2047 : tensor<32x197x768xf32>
    %v2049 = stablehlo.reshape %v2048 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2050 = stablehlo.multiply %v2049, %v2049 : tensor<32x151296xf32>
    %v2051 = stablehlo.multiply %v2050, %v2049 : tensor<32x151296xf32>
    %v2052 = stablehlo.constant dense<0.044715> : tensor<32x151296xf32>
    %v2053 = stablehlo.multiply %v2052, %v2051 : tensor<32x151296xf32>
    %v2054 = stablehlo.add %v2049, %v2053 : tensor<32x151296xf32>
    %v2055 = stablehlo.constant dense<0.7978845608028654> : tensor<32x151296xf32>
    %v2056 = stablehlo.multiply %v2055, %v2054 : tensor<32x151296xf32>
    %v2057 = stablehlo.tanh %v2056 : tensor<32x151296xf32>
    %v2058 = stablehlo.constant dense<1.0> : tensor<32x151296xf32>
    %v2059 = stablehlo.add %v2058, %v2057 : tensor<32x151296xf32>
    %v2060 = stablehlo.constant dense<0.5> : tensor<32x151296xf32>
    %v2061 = stablehlo.multiply %v2060, %v2049 : tensor<32x151296xf32>
    %v2062 = stablehlo.multiply %v2061, %v2059 : tensor<32x151296xf32>
    %v2063 = stablehlo.reshape %v2062 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2064 = stablehlo.dot_general %v2063, %b9_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x192xf32>) -> tensor<32x197x192xf32>
    %v2065 = stablehlo.broadcast_in_dim %b9_bfc2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v2066 = stablehlo.add %v2064, %v2065 : tensor<32x197x192xf32>
    %v2067 = stablehlo.reshape %v2066 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2068 = stablehlo.broadcast_in_dim %dp19, dims = [0] : (tensor<32xf32>) -> tensor<32x37824xf32>
    %v2069 = stablehlo.multiply %v2068, %v2067 : tensor<32x37824xf32>
    %v2070 = stablehlo.add %v2016, %v2069 : tensor<32x37824xf32>
    %v2071 = stablehlo.reshape %v2070 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2072 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2073 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v2074 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v2075 = stablehlo.reduce(%v2071 init: %v2072) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2076 = stablehlo.broadcast_in_dim %v2075, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v2077 = stablehlo.divide %v2076, %v2073 : tensor<32x197x192xf32>
    %v2078 = stablehlo.subtract %v2071, %v2077 : tensor<32x197x192xf32>
    %v2079 = stablehlo.multiply %v2078, %v2078 : tensor<32x197x192xf32>
    %v2080 = stablehlo.reduce(%v2079 init: %v2072) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2081 = stablehlo.broadcast_in_dim %v2080, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v2082 = stablehlo.divide %v2081, %v2073 : tensor<32x197x192xf32>
    %v2083 = stablehlo.add %v2082, %v2074 : tensor<32x197x192xf32>
    %v2084 = stablehlo.rsqrt %v2083 : tensor<32x197x192xf32>
    %v2085 = stablehlo.multiply %v2078, %v2084 : tensor<32x197x192xf32>
    %v2086 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v2087 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v2088 = stablehlo.multiply %v2085, %v2086 : tensor<32x197x192xf32>
    %v2089 = stablehlo.add %v2088, %v2087 : tensor<32x197x192xf32>
    %v2090 = stablehlo.reshape %v2089 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2091 = stablehlo.reshape %v2090 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2092 = stablehlo.broadcast_in_dim %b10_g1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v2093 = stablehlo.multiply %v2091, %v2092 : tensor<32x197x192xf32>
    %v2094 = stablehlo.reshape %v2093 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2095 = stablehlo.reshape %v2094 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2096 = stablehlo.broadcast_in_dim %b10_bt1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v2097 = stablehlo.add %v2095, %v2096 : tensor<32x197x192xf32>
    %v2098 = stablehlo.reshape %v2097 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2099 = stablehlo.reshape %v2098 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2100 = stablehlo.dot_general %v2099, %b10_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v2101 = stablehlo.broadcast_in_dim %b10_bq, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v2102 = stablehlo.add %v2100, %v2101 : tensor<32x197x192xf32>
    %v2103 = stablehlo.reshape %v2102 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2104 = stablehlo.reshape %v2098 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2105 = stablehlo.dot_general %v2104, %b10_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v2106 = stablehlo.broadcast_in_dim %b10_bk, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v2107 = stablehlo.add %v2105, %v2106 : tensor<32x197x192xf32>
    %v2108 = stablehlo.reshape %v2107 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2109 = stablehlo.reshape %v2098 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2110 = stablehlo.dot_general %v2109, %b10_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v2111 = stablehlo.broadcast_in_dim %b10_bv, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v2112 = stablehlo.add %v2110, %v2111 : tensor<32x197x192xf32>
    %v2113 = stablehlo.reshape %v2112 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2114 = stablehlo.reshape %v2103 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2115 = stablehlo.slice %v2114 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v2116 = stablehlo.reshape %v2115 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2117 = stablehlo.reshape %v2108 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2118 = stablehlo.slice %v2117 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v2119 = stablehlo.reshape %v2118 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2120 = stablehlo.reshape %v2113 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2121 = stablehlo.slice %v2120 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v2122 = stablehlo.reshape %v2121 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2123 = stablehlo.reshape %v2119 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2124 = stablehlo.transpose %v2123, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2125 = stablehlo.reshape %v2124 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2126 = stablehlo.reshape %v2116 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2127 = stablehlo.reshape %v2125 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2128 = stablehlo.dot_general %v2126, %v2127, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2129 = stablehlo.reshape %v2128 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2130 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2131 = stablehlo.multiply %v2129, %v2130 : tensor<32x38809xf32>
    %v2132 = stablehlo.reshape %v2131 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2133 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2134 = stablehlo.exponential %v2132 : tensor<32x197x197xf32>
    %v2135 = stablehlo.reduce(%v2134 init: %v2133) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2136 = stablehlo.broadcast_in_dim %v2135, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2137 = stablehlo.divide %v2134, %v2136 : tensor<32x197x197xf32>
    %v2138 = stablehlo.reshape %v2137 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2139 = stablehlo.reshape %v2138 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2140 = stablehlo.reshape %v2122 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2141 = stablehlo.dot_general %v2139, %v2140, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2142 = stablehlo.reshape %v2141 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2143 = stablehlo.reshape %v2142 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2144 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2145 = stablehlo.pad %v2143, %v2144, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v2146 = stablehlo.reshape %v2145 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2147 = stablehlo.reshape %v2103 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2148 = stablehlo.slice %v2147 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v2149 = stablehlo.reshape %v2148 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2150 = stablehlo.reshape %v2108 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2151 = stablehlo.slice %v2150 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v2152 = stablehlo.reshape %v2151 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2153 = stablehlo.reshape %v2113 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2154 = stablehlo.slice %v2153 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v2155 = stablehlo.reshape %v2154 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2156 = stablehlo.reshape %v2152 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2157 = stablehlo.transpose %v2156, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2158 = stablehlo.reshape %v2157 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2159 = stablehlo.reshape %v2149 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2160 = stablehlo.reshape %v2158 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2161 = stablehlo.dot_general %v2159, %v2160, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2162 = stablehlo.reshape %v2161 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2163 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2164 = stablehlo.multiply %v2162, %v2163 : tensor<32x38809xf32>
    %v2165 = stablehlo.reshape %v2164 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2166 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2167 = stablehlo.exponential %v2165 : tensor<32x197x197xf32>
    %v2168 = stablehlo.reduce(%v2167 init: %v2166) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2169 = stablehlo.broadcast_in_dim %v2168, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2170 = stablehlo.divide %v2167, %v2169 : tensor<32x197x197xf32>
    %v2171 = stablehlo.reshape %v2170 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2172 = stablehlo.reshape %v2171 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2173 = stablehlo.reshape %v2155 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2174 = stablehlo.dot_general %v2172, %v2173, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2175 = stablehlo.reshape %v2174 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2176 = stablehlo.reshape %v2175 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2177 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2178 = stablehlo.pad %v2176, %v2177, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v2179 = stablehlo.reshape %v2178 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2180 = stablehlo.add %v2146, %v2179 : tensor<32x37824xf32>
    %v2181 = stablehlo.reshape %v2103 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2182 = stablehlo.slice %v2181 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v2183 = stablehlo.reshape %v2182 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2184 = stablehlo.reshape %v2108 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2185 = stablehlo.slice %v2184 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v2186 = stablehlo.reshape %v2185 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2187 = stablehlo.reshape %v2113 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2188 = stablehlo.slice %v2187 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v2189 = stablehlo.reshape %v2188 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2190 = stablehlo.reshape %v2186 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2191 = stablehlo.transpose %v2190, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2192 = stablehlo.reshape %v2191 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2193 = stablehlo.reshape %v2183 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2194 = stablehlo.reshape %v2192 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2195 = stablehlo.dot_general %v2193, %v2194, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2196 = stablehlo.reshape %v2195 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2197 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2198 = stablehlo.multiply %v2196, %v2197 : tensor<32x38809xf32>
    %v2199 = stablehlo.reshape %v2198 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2200 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2201 = stablehlo.exponential %v2199 : tensor<32x197x197xf32>
    %v2202 = stablehlo.reduce(%v2201 init: %v2200) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2203 = stablehlo.broadcast_in_dim %v2202, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2204 = stablehlo.divide %v2201, %v2203 : tensor<32x197x197xf32>
    %v2205 = stablehlo.reshape %v2204 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2206 = stablehlo.reshape %v2205 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2207 = stablehlo.reshape %v2189 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2208 = stablehlo.dot_general %v2206, %v2207, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2209 = stablehlo.reshape %v2208 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2210 = stablehlo.reshape %v2209 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2211 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2212 = stablehlo.pad %v2210, %v2211, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v2213 = stablehlo.reshape %v2212 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2214 = stablehlo.add %v2180, %v2213 : tensor<32x37824xf32>
    %v2215 = stablehlo.reshape %v2214 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2216 = stablehlo.dot_general %v2215, %b10_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v2217 = stablehlo.broadcast_in_dim %b10_bo, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v2218 = stablehlo.add %v2216, %v2217 : tensor<32x197x192xf32>
    %v2219 = stablehlo.reshape %v2218 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2220 = stablehlo.broadcast_in_dim %dp20, dims = [0] : (tensor<32xf32>) -> tensor<32x37824xf32>
    %v2221 = stablehlo.multiply %v2220, %v2219 : tensor<32x37824xf32>
    %v2222 = stablehlo.add %v2070, %v2221 : tensor<32x37824xf32>
    %v2223 = stablehlo.reshape %v2222 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2224 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2225 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v2226 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v2227 = stablehlo.reduce(%v2223 init: %v2224) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2228 = stablehlo.broadcast_in_dim %v2227, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v2229 = stablehlo.divide %v2228, %v2225 : tensor<32x197x192xf32>
    %v2230 = stablehlo.subtract %v2223, %v2229 : tensor<32x197x192xf32>
    %v2231 = stablehlo.multiply %v2230, %v2230 : tensor<32x197x192xf32>
    %v2232 = stablehlo.reduce(%v2231 init: %v2224) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2233 = stablehlo.broadcast_in_dim %v2232, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v2234 = stablehlo.divide %v2233, %v2225 : tensor<32x197x192xf32>
    %v2235 = stablehlo.add %v2234, %v2226 : tensor<32x197x192xf32>
    %v2236 = stablehlo.rsqrt %v2235 : tensor<32x197x192xf32>
    %v2237 = stablehlo.multiply %v2230, %v2236 : tensor<32x197x192xf32>
    %v2238 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v2239 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v2240 = stablehlo.multiply %v2237, %v2238 : tensor<32x197x192xf32>
    %v2241 = stablehlo.add %v2240, %v2239 : tensor<32x197x192xf32>
    %v2242 = stablehlo.reshape %v2241 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2243 = stablehlo.reshape %v2242 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2244 = stablehlo.broadcast_in_dim %b10_g2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v2245 = stablehlo.multiply %v2243, %v2244 : tensor<32x197x192xf32>
    %v2246 = stablehlo.reshape %v2245 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2247 = stablehlo.reshape %v2246 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2248 = stablehlo.broadcast_in_dim %b10_bt2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v2249 = stablehlo.add %v2247, %v2248 : tensor<32x197x192xf32>
    %v2250 = stablehlo.reshape %v2249 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2251 = stablehlo.reshape %v2250 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2252 = stablehlo.dot_general %v2251, %b10_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x768xf32>) -> tensor<32x197x768xf32>
    %v2253 = stablehlo.broadcast_in_dim %b10_bfc1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v2254 = stablehlo.add %v2252, %v2253 : tensor<32x197x768xf32>
    %v2255 = stablehlo.reshape %v2254 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2256 = stablehlo.multiply %v2255, %v2255 : tensor<32x151296xf32>
    %v2257 = stablehlo.multiply %v2256, %v2255 : tensor<32x151296xf32>
    %v2258 = stablehlo.constant dense<0.044715> : tensor<32x151296xf32>
    %v2259 = stablehlo.multiply %v2258, %v2257 : tensor<32x151296xf32>
    %v2260 = stablehlo.add %v2255, %v2259 : tensor<32x151296xf32>
    %v2261 = stablehlo.constant dense<0.7978845608028654> : tensor<32x151296xf32>
    %v2262 = stablehlo.multiply %v2261, %v2260 : tensor<32x151296xf32>
    %v2263 = stablehlo.tanh %v2262 : tensor<32x151296xf32>
    %v2264 = stablehlo.constant dense<1.0> : tensor<32x151296xf32>
    %v2265 = stablehlo.add %v2264, %v2263 : tensor<32x151296xf32>
    %v2266 = stablehlo.constant dense<0.5> : tensor<32x151296xf32>
    %v2267 = stablehlo.multiply %v2266, %v2255 : tensor<32x151296xf32>
    %v2268 = stablehlo.multiply %v2267, %v2265 : tensor<32x151296xf32>
    %v2269 = stablehlo.reshape %v2268 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2270 = stablehlo.dot_general %v2269, %b10_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x192xf32>) -> tensor<32x197x192xf32>
    %v2271 = stablehlo.broadcast_in_dim %b10_bfc2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v2272 = stablehlo.add %v2270, %v2271 : tensor<32x197x192xf32>
    %v2273 = stablehlo.reshape %v2272 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2274 = stablehlo.broadcast_in_dim %dp21, dims = [0] : (tensor<32xf32>) -> tensor<32x37824xf32>
    %v2275 = stablehlo.multiply %v2274, %v2273 : tensor<32x37824xf32>
    %v2276 = stablehlo.add %v2222, %v2275 : tensor<32x37824xf32>
    %v2277 = stablehlo.reshape %v2276 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2278 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2279 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v2280 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v2281 = stablehlo.reduce(%v2277 init: %v2278) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2282 = stablehlo.broadcast_in_dim %v2281, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v2283 = stablehlo.divide %v2282, %v2279 : tensor<32x197x192xf32>
    %v2284 = stablehlo.subtract %v2277, %v2283 : tensor<32x197x192xf32>
    %v2285 = stablehlo.multiply %v2284, %v2284 : tensor<32x197x192xf32>
    %v2286 = stablehlo.reduce(%v2285 init: %v2278) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2287 = stablehlo.broadcast_in_dim %v2286, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v2288 = stablehlo.divide %v2287, %v2279 : tensor<32x197x192xf32>
    %v2289 = stablehlo.add %v2288, %v2280 : tensor<32x197x192xf32>
    %v2290 = stablehlo.rsqrt %v2289 : tensor<32x197x192xf32>
    %v2291 = stablehlo.multiply %v2284, %v2290 : tensor<32x197x192xf32>
    %v2292 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v2293 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v2294 = stablehlo.multiply %v2291, %v2292 : tensor<32x197x192xf32>
    %v2295 = stablehlo.add %v2294, %v2293 : tensor<32x197x192xf32>
    %v2296 = stablehlo.reshape %v2295 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2297 = stablehlo.reshape %v2296 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2298 = stablehlo.broadcast_in_dim %b11_g1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v2299 = stablehlo.multiply %v2297, %v2298 : tensor<32x197x192xf32>
    %v2300 = stablehlo.reshape %v2299 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2301 = stablehlo.reshape %v2300 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2302 = stablehlo.broadcast_in_dim %b11_bt1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v2303 = stablehlo.add %v2301, %v2302 : tensor<32x197x192xf32>
    %v2304 = stablehlo.reshape %v2303 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2305 = stablehlo.reshape %v2304 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2306 = stablehlo.dot_general %v2305, %b11_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v2307 = stablehlo.broadcast_in_dim %b11_bq, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v2308 = stablehlo.add %v2306, %v2307 : tensor<32x197x192xf32>
    %v2309 = stablehlo.reshape %v2308 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2310 = stablehlo.reshape %v2304 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2311 = stablehlo.dot_general %v2310, %b11_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v2312 = stablehlo.broadcast_in_dim %b11_bk, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v2313 = stablehlo.add %v2311, %v2312 : tensor<32x197x192xf32>
    %v2314 = stablehlo.reshape %v2313 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2315 = stablehlo.reshape %v2304 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2316 = stablehlo.dot_general %v2315, %b11_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v2317 = stablehlo.broadcast_in_dim %b11_bv, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v2318 = stablehlo.add %v2316, %v2317 : tensor<32x197x192xf32>
    %v2319 = stablehlo.reshape %v2318 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2320 = stablehlo.reshape %v2309 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2321 = stablehlo.slice %v2320 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v2322 = stablehlo.reshape %v2321 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2323 = stablehlo.reshape %v2314 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2324 = stablehlo.slice %v2323 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v2325 = stablehlo.reshape %v2324 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2326 = stablehlo.reshape %v2319 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2327 = stablehlo.slice %v2326 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v2328 = stablehlo.reshape %v2327 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2329 = stablehlo.reshape %v2325 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2330 = stablehlo.transpose %v2329, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2331 = stablehlo.reshape %v2330 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2332 = stablehlo.reshape %v2322 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2333 = stablehlo.reshape %v2331 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2334 = stablehlo.dot_general %v2332, %v2333, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2335 = stablehlo.reshape %v2334 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2336 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2337 = stablehlo.multiply %v2335, %v2336 : tensor<32x38809xf32>
    %v2338 = stablehlo.reshape %v2337 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2339 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2340 = stablehlo.exponential %v2338 : tensor<32x197x197xf32>
    %v2341 = stablehlo.reduce(%v2340 init: %v2339) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2342 = stablehlo.broadcast_in_dim %v2341, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2343 = stablehlo.divide %v2340, %v2342 : tensor<32x197x197xf32>
    %v2344 = stablehlo.reshape %v2343 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2345 = stablehlo.reshape %v2344 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2346 = stablehlo.reshape %v2328 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2347 = stablehlo.dot_general %v2345, %v2346, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2348 = stablehlo.reshape %v2347 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2349 = stablehlo.reshape %v2348 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2350 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2351 = stablehlo.pad %v2349, %v2350, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v2352 = stablehlo.reshape %v2351 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2353 = stablehlo.reshape %v2309 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2354 = stablehlo.slice %v2353 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v2355 = stablehlo.reshape %v2354 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2356 = stablehlo.reshape %v2314 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2357 = stablehlo.slice %v2356 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v2358 = stablehlo.reshape %v2357 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2359 = stablehlo.reshape %v2319 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2360 = stablehlo.slice %v2359 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v2361 = stablehlo.reshape %v2360 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2362 = stablehlo.reshape %v2358 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2363 = stablehlo.transpose %v2362, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2364 = stablehlo.reshape %v2363 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2365 = stablehlo.reshape %v2355 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2366 = stablehlo.reshape %v2364 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2367 = stablehlo.dot_general %v2365, %v2366, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2368 = stablehlo.reshape %v2367 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2369 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2370 = stablehlo.multiply %v2368, %v2369 : tensor<32x38809xf32>
    %v2371 = stablehlo.reshape %v2370 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2372 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2373 = stablehlo.exponential %v2371 : tensor<32x197x197xf32>
    %v2374 = stablehlo.reduce(%v2373 init: %v2372) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2375 = stablehlo.broadcast_in_dim %v2374, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2376 = stablehlo.divide %v2373, %v2375 : tensor<32x197x197xf32>
    %v2377 = stablehlo.reshape %v2376 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2378 = stablehlo.reshape %v2377 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2379 = stablehlo.reshape %v2361 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2380 = stablehlo.dot_general %v2378, %v2379, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2381 = stablehlo.reshape %v2380 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2382 = stablehlo.reshape %v2381 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2383 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2384 = stablehlo.pad %v2382, %v2383, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v2385 = stablehlo.reshape %v2384 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2386 = stablehlo.add %v2352, %v2385 : tensor<32x37824xf32>
    %v2387 = stablehlo.reshape %v2309 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2388 = stablehlo.slice %v2387 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v2389 = stablehlo.reshape %v2388 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2390 = stablehlo.reshape %v2314 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2391 = stablehlo.slice %v2390 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v2392 = stablehlo.reshape %v2391 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2393 = stablehlo.reshape %v2319 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2394 = stablehlo.slice %v2393 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v2395 = stablehlo.reshape %v2394 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2396 = stablehlo.reshape %v2392 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2397 = stablehlo.transpose %v2396, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2398 = stablehlo.reshape %v2397 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2399 = stablehlo.reshape %v2389 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2400 = stablehlo.reshape %v2398 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2401 = stablehlo.dot_general %v2399, %v2400, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2402 = stablehlo.reshape %v2401 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2403 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2404 = stablehlo.multiply %v2402, %v2403 : tensor<32x38809xf32>
    %v2405 = stablehlo.reshape %v2404 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2406 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2407 = stablehlo.exponential %v2405 : tensor<32x197x197xf32>
    %v2408 = stablehlo.reduce(%v2407 init: %v2406) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2409 = stablehlo.broadcast_in_dim %v2408, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2410 = stablehlo.divide %v2407, %v2409 : tensor<32x197x197xf32>
    %v2411 = stablehlo.reshape %v2410 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2412 = stablehlo.reshape %v2411 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2413 = stablehlo.reshape %v2395 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2414 = stablehlo.dot_general %v2412, %v2413, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2415 = stablehlo.reshape %v2414 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2416 = stablehlo.reshape %v2415 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2417 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2418 = stablehlo.pad %v2416, %v2417, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v2419 = stablehlo.reshape %v2418 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2420 = stablehlo.add %v2386, %v2419 : tensor<32x37824xf32>
    %v2421 = stablehlo.reshape %v2420 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2422 = stablehlo.dot_general %v2421, %b11_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v2423 = stablehlo.broadcast_in_dim %b11_bo, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v2424 = stablehlo.add %v2422, %v2423 : tensor<32x197x192xf32>
    %v2425 = stablehlo.reshape %v2424 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2426 = stablehlo.broadcast_in_dim %dp22, dims = [0] : (tensor<32xf32>) -> tensor<32x37824xf32>
    %v2427 = stablehlo.multiply %v2426, %v2425 : tensor<32x37824xf32>
    %v2428 = stablehlo.add %v2276, %v2427 : tensor<32x37824xf32>
    %v2429 = stablehlo.reshape %v2428 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2430 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2431 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v2432 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v2433 = stablehlo.reduce(%v2429 init: %v2430) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2434 = stablehlo.broadcast_in_dim %v2433, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v2435 = stablehlo.divide %v2434, %v2431 : tensor<32x197x192xf32>
    %v2436 = stablehlo.subtract %v2429, %v2435 : tensor<32x197x192xf32>
    %v2437 = stablehlo.multiply %v2436, %v2436 : tensor<32x197x192xf32>
    %v2438 = stablehlo.reduce(%v2437 init: %v2430) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2439 = stablehlo.broadcast_in_dim %v2438, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v2440 = stablehlo.divide %v2439, %v2431 : tensor<32x197x192xf32>
    %v2441 = stablehlo.add %v2440, %v2432 : tensor<32x197x192xf32>
    %v2442 = stablehlo.rsqrt %v2441 : tensor<32x197x192xf32>
    %v2443 = stablehlo.multiply %v2436, %v2442 : tensor<32x197x192xf32>
    %v2444 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v2445 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v2446 = stablehlo.multiply %v2443, %v2444 : tensor<32x197x192xf32>
    %v2447 = stablehlo.add %v2446, %v2445 : tensor<32x197x192xf32>
    %v2448 = stablehlo.reshape %v2447 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2449 = stablehlo.reshape %v2448 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2450 = stablehlo.broadcast_in_dim %b11_g2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v2451 = stablehlo.multiply %v2449, %v2450 : tensor<32x197x192xf32>
    %v2452 = stablehlo.reshape %v2451 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2453 = stablehlo.reshape %v2452 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2454 = stablehlo.broadcast_in_dim %b11_bt2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v2455 = stablehlo.add %v2453, %v2454 : tensor<32x197x192xf32>
    %v2456 = stablehlo.reshape %v2455 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2457 = stablehlo.reshape %v2456 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2458 = stablehlo.dot_general %v2457, %b11_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x768xf32>) -> tensor<32x197x768xf32>
    %v2459 = stablehlo.broadcast_in_dim %b11_bfc1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v2460 = stablehlo.add %v2458, %v2459 : tensor<32x197x768xf32>
    %v2461 = stablehlo.reshape %v2460 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2462 = stablehlo.multiply %v2461, %v2461 : tensor<32x151296xf32>
    %v2463 = stablehlo.multiply %v2462, %v2461 : tensor<32x151296xf32>
    %v2464 = stablehlo.constant dense<0.044715> : tensor<32x151296xf32>
    %v2465 = stablehlo.multiply %v2464, %v2463 : tensor<32x151296xf32>
    %v2466 = stablehlo.add %v2461, %v2465 : tensor<32x151296xf32>
    %v2467 = stablehlo.constant dense<0.7978845608028654> : tensor<32x151296xf32>
    %v2468 = stablehlo.multiply %v2467, %v2466 : tensor<32x151296xf32>
    %v2469 = stablehlo.tanh %v2468 : tensor<32x151296xf32>
    %v2470 = stablehlo.constant dense<1.0> : tensor<32x151296xf32>
    %v2471 = stablehlo.add %v2470, %v2469 : tensor<32x151296xf32>
    %v2472 = stablehlo.constant dense<0.5> : tensor<32x151296xf32>
    %v2473 = stablehlo.multiply %v2472, %v2461 : tensor<32x151296xf32>
    %v2474 = stablehlo.multiply %v2473, %v2471 : tensor<32x151296xf32>
    %v2475 = stablehlo.reshape %v2474 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2476 = stablehlo.dot_general %v2475, %b11_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x192xf32>) -> tensor<32x197x192xf32>
    %v2477 = stablehlo.broadcast_in_dim %b11_bfc2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v2478 = stablehlo.add %v2476, %v2477 : tensor<32x197x192xf32>
    %v2479 = stablehlo.reshape %v2478 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2480 = stablehlo.broadcast_in_dim %dp23, dims = [0] : (tensor<32xf32>) -> tensor<32x37824xf32>
    %v2481 = stablehlo.multiply %v2480, %v2479 : tensor<32x37824xf32>
    %v2482 = stablehlo.add %v2428, %v2481 : tensor<32x37824xf32>
    %v2483 = stablehlo.reshape %v2482 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2484 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2485 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v2486 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v2487 = stablehlo.reduce(%v2483 init: %v2484) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2488 = stablehlo.broadcast_in_dim %v2487, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v2489 = stablehlo.divide %v2488, %v2485 : tensor<32x197x192xf32>
    %v2490 = stablehlo.subtract %v2483, %v2489 : tensor<32x197x192xf32>
    %v2491 = stablehlo.multiply %v2490, %v2490 : tensor<32x197x192xf32>
    %v2492 = stablehlo.reduce(%v2491 init: %v2484) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2493 = stablehlo.broadcast_in_dim %v2492, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v2494 = stablehlo.divide %v2493, %v2485 : tensor<32x197x192xf32>
    %v2495 = stablehlo.add %v2494, %v2486 : tensor<32x197x192xf32>
    %v2496 = stablehlo.rsqrt %v2495 : tensor<32x197x192xf32>
    %v2497 = stablehlo.multiply %v2490, %v2496 : tensor<32x197x192xf32>
    %v2498 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v2499 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v2500 = stablehlo.multiply %v2497, %v2498 : tensor<32x197x192xf32>
    %v2501 = stablehlo.add %v2500, %v2499 : tensor<32x197x192xf32>
    %v2502 = stablehlo.reshape %v2501 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2503 = stablehlo.reshape %v2502 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2504 = stablehlo.broadcast_in_dim %gF, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v2505 = stablehlo.multiply %v2503, %v2504 : tensor<32x197x192xf32>
    %v2506 = stablehlo.reshape %v2505 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2507 = stablehlo.reshape %v2506 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2508 = stablehlo.broadcast_in_dim %btF, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v2509 = stablehlo.add %v2507, %v2508 : tensor<32x197x192xf32>
    %v2510 = stablehlo.reshape %v2509 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2511 = stablehlo.reshape %v2510 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2512 = stablehlo.slice %v2511 [0:32, 0:1, 0:192] : (tensor<32x197x192xf32>) -> tensor<32x1x192xf32>
    %v2513 = stablehlo.reshape %v2512 : (tensor<32x1x192xf32>) -> tensor<32x192xf32>
    %v2514 = stablehlo.dot_general %v2513, %Wc, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x192xf32>, tensor<192x10xf32>) -> tensor<32x10xf32>
    %v2515 = stablehlo.broadcast_in_dim %bc, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v2516 = stablehlo.add %v2514, %v2515 : tensor<32x10xf32>
    return %v2516 : tensor<32x10xf32>
  }
}

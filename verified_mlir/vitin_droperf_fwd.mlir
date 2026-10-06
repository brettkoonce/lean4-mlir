module @m {
  func.func @vitin_droperf_fwd(%x: tensor<32x150528xf32>, %wConv: tensor<192x3x16x16xf32>, %bConv: tensor<192xf32>, %cls: tensor<192xf32>, %pos: tensor<197x192xf32>, %b0_g1: tensor<192xf32>, %b0_bt1: tensor<192xf32>, %b0_Wq: tensor<192x192xf32>, %b0_bq: tensor<192xf32>, %b0_Wk: tensor<192x192xf32>, %b0_bk: tensor<192xf32>, %b0_Wv: tensor<192x192xf32>, %b0_bv: tensor<192xf32>, %b0_Wo: tensor<192x192xf32>, %b0_bo: tensor<192xf32>, %b0_g2: tensor<192xf32>, %b0_bt2: tensor<192xf32>, %b0_Wfc1: tensor<192x768xf32>, %b0_bfc1: tensor<768xf32>, %b0_Wfc2: tensor<768x192xf32>, %b0_bfc2: tensor<192xf32>, %b1_g1: tensor<192xf32>, %b1_bt1: tensor<192xf32>, %b1_Wq: tensor<192x192xf32>, %b1_bq: tensor<192xf32>, %b1_Wk: tensor<192x192xf32>, %b1_bk: tensor<192xf32>, %b1_Wv: tensor<192x192xf32>, %b1_bv: tensor<192xf32>, %b1_Wo: tensor<192x192xf32>, %b1_bo: tensor<192xf32>, %b1_g2: tensor<192xf32>, %b1_bt2: tensor<192xf32>, %b1_Wfc1: tensor<192x768xf32>, %b1_bfc1: tensor<768xf32>, %b1_Wfc2: tensor<768x192xf32>, %b1_bfc2: tensor<192xf32>, %b2_g1: tensor<192xf32>, %b2_bt1: tensor<192xf32>, %b2_Wq: tensor<192x192xf32>, %b2_bq: tensor<192xf32>, %b2_Wk: tensor<192x192xf32>, %b2_bk: tensor<192xf32>, %b2_Wv: tensor<192x192xf32>, %b2_bv: tensor<192xf32>, %b2_Wo: tensor<192x192xf32>, %b2_bo: tensor<192xf32>, %b2_g2: tensor<192xf32>, %b2_bt2: tensor<192xf32>, %b2_Wfc1: tensor<192x768xf32>, %b2_bfc1: tensor<768xf32>, %b2_Wfc2: tensor<768x192xf32>, %b2_bfc2: tensor<192xf32>, %b3_g1: tensor<192xf32>, %b3_bt1: tensor<192xf32>, %b3_Wq: tensor<192x192xf32>, %b3_bq: tensor<192xf32>, %b3_Wk: tensor<192x192xf32>, %b3_bk: tensor<192xf32>, %b3_Wv: tensor<192x192xf32>, %b3_bv: tensor<192xf32>, %b3_Wo: tensor<192x192xf32>, %b3_bo: tensor<192xf32>, %b3_g2: tensor<192xf32>, %b3_bt2: tensor<192xf32>, %b3_Wfc1: tensor<192x768xf32>, %b3_bfc1: tensor<768xf32>, %b3_Wfc2: tensor<768x192xf32>, %b3_bfc2: tensor<192xf32>, %b4_g1: tensor<192xf32>, %b4_bt1: tensor<192xf32>, %b4_Wq: tensor<192x192xf32>, %b4_bq: tensor<192xf32>, %b4_Wk: tensor<192x192xf32>, %b4_bk: tensor<192xf32>, %b4_Wv: tensor<192x192xf32>, %b4_bv: tensor<192xf32>, %b4_Wo: tensor<192x192xf32>, %b4_bo: tensor<192xf32>, %b4_g2: tensor<192xf32>, %b4_bt2: tensor<192xf32>, %b4_Wfc1: tensor<192x768xf32>, %b4_bfc1: tensor<768xf32>, %b4_Wfc2: tensor<768x192xf32>, %b4_bfc2: tensor<192xf32>, %b5_g1: tensor<192xf32>, %b5_bt1: tensor<192xf32>, %b5_Wq: tensor<192x192xf32>, %b5_bq: tensor<192xf32>, %b5_Wk: tensor<192x192xf32>, %b5_bk: tensor<192xf32>, %b5_Wv: tensor<192x192xf32>, %b5_bv: tensor<192xf32>, %b5_Wo: tensor<192x192xf32>, %b5_bo: tensor<192xf32>, %b5_g2: tensor<192xf32>, %b5_bt2: tensor<192xf32>, %b5_Wfc1: tensor<192x768xf32>, %b5_bfc1: tensor<768xf32>, %b5_Wfc2: tensor<768x192xf32>, %b5_bfc2: tensor<192xf32>, %b6_g1: tensor<192xf32>, %b6_bt1: tensor<192xf32>, %b6_Wq: tensor<192x192xf32>, %b6_bq: tensor<192xf32>, %b6_Wk: tensor<192x192xf32>, %b6_bk: tensor<192xf32>, %b6_Wv: tensor<192x192xf32>, %b6_bv: tensor<192xf32>, %b6_Wo: tensor<192x192xf32>, %b6_bo: tensor<192xf32>, %b6_g2: tensor<192xf32>, %b6_bt2: tensor<192xf32>, %b6_Wfc1: tensor<192x768xf32>, %b6_bfc1: tensor<768xf32>, %b6_Wfc2: tensor<768x192xf32>, %b6_bfc2: tensor<192xf32>, %b7_g1: tensor<192xf32>, %b7_bt1: tensor<192xf32>, %b7_Wq: tensor<192x192xf32>, %b7_bq: tensor<192xf32>, %b7_Wk: tensor<192x192xf32>, %b7_bk: tensor<192xf32>, %b7_Wv: tensor<192x192xf32>, %b7_bv: tensor<192xf32>, %b7_Wo: tensor<192x192xf32>, %b7_bo: tensor<192xf32>, %b7_g2: tensor<192xf32>, %b7_bt2: tensor<192xf32>, %b7_Wfc1: tensor<192x768xf32>, %b7_bfc1: tensor<768xf32>, %b7_Wfc2: tensor<768x192xf32>, %b7_bfc2: tensor<192xf32>, %b8_g1: tensor<192xf32>, %b8_bt1: tensor<192xf32>, %b8_Wq: tensor<192x192xf32>, %b8_bq: tensor<192xf32>, %b8_Wk: tensor<192x192xf32>, %b8_bk: tensor<192xf32>, %b8_Wv: tensor<192x192xf32>, %b8_bv: tensor<192xf32>, %b8_Wo: tensor<192x192xf32>, %b8_bo: tensor<192xf32>, %b8_g2: tensor<192xf32>, %b8_bt2: tensor<192xf32>, %b8_Wfc1: tensor<192x768xf32>, %b8_bfc1: tensor<768xf32>, %b8_Wfc2: tensor<768x192xf32>, %b8_bfc2: tensor<192xf32>, %b9_g1: tensor<192xf32>, %b9_bt1: tensor<192xf32>, %b9_Wq: tensor<192x192xf32>, %b9_bq: tensor<192xf32>, %b9_Wk: tensor<192x192xf32>, %b9_bk: tensor<192xf32>, %b9_Wv: tensor<192x192xf32>, %b9_bv: tensor<192xf32>, %b9_Wo: tensor<192x192xf32>, %b9_bo: tensor<192xf32>, %b9_g2: tensor<192xf32>, %b9_bt2: tensor<192xf32>, %b9_Wfc1: tensor<192x768xf32>, %b9_bfc1: tensor<768xf32>, %b9_Wfc2: tensor<768x192xf32>, %b9_bfc2: tensor<192xf32>, %b10_g1: tensor<192xf32>, %b10_bt1: tensor<192xf32>, %b10_Wq: tensor<192x192xf32>, %b10_bq: tensor<192xf32>, %b10_Wk: tensor<192x192xf32>, %b10_bk: tensor<192xf32>, %b10_Wv: tensor<192x192xf32>, %b10_bv: tensor<192xf32>, %b10_Wo: tensor<192x192xf32>, %b10_bo: tensor<192xf32>, %b10_g2: tensor<192xf32>, %b10_bt2: tensor<192xf32>, %b10_Wfc1: tensor<192x768xf32>, %b10_bfc1: tensor<768xf32>, %b10_Wfc2: tensor<768x192xf32>, %b10_bfc2: tensor<192xf32>, %b11_g1: tensor<192xf32>, %b11_bt1: tensor<192xf32>, %b11_Wq: tensor<192x192xf32>, %b11_bq: tensor<192xf32>, %b11_Wk: tensor<192x192xf32>, %b11_bk: tensor<192xf32>, %b11_Wv: tensor<192x192xf32>, %b11_bv: tensor<192xf32>, %b11_Wo: tensor<192x192xf32>, %b11_bo: tensor<192xf32>, %b11_g2: tensor<192xf32>, %b11_bt2: tensor<192xf32>, %b11_Wfc1: tensor<192x768xf32>, %b11_bfc1: tensor<768xf32>, %b11_Wfc2: tensor<768x192xf32>, %b11_bfc2: tensor<192xf32>, %gF: tensor<192xf32>, %btF: tensor<192xf32>, %Wc: tensor<192x1000xf32>, %bc: tensor<1000xf32>, %dp0: tensor<32xf32>, %dp1: tensor<32xf32>, %dp2: tensor<32xf32>, %dp3: tensor<32xf32>, %dp4: tensor<32xf32>, %dp5: tensor<32xf32>, %dp6: tensor<32xf32>, %dp7: tensor<32xf32>, %dp8: tensor<32xf32>, %dp9: tensor<32xf32>, %dp10: tensor<32xf32>, %dp11: tensor<32xf32>, %dp12: tensor<32xf32>, %dp13: tensor<32xf32>, %dp14: tensor<32xf32>, %dp15: tensor<32xf32>, %dp16: tensor<32xf32>, %dp17: tensor<32xf32>, %dp18: tensor<32xf32>, %dp19: tensor<32xf32>, %dp20: tensor<32xf32>, %dp21: tensor<32xf32>, %dp22: tensor<32xf32>, %dp23: tensor<32xf32>) -> tensor<32x1000xf32> {
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
    %v196 = stablehlo.constant dense<0.5> : tensor<32x151296xf32>
    %v197 = stablehlo.multiply %v196, %v195 : tensor<32x151296xf32>
    %v198 = stablehlo.negate %v195 : tensor<32x151296xf32>
    %v199 = stablehlo.constant dense<0.7071067811865476> : tensor<32x151296xf32>
    %v200 = stablehlo.multiply %v198, %v199 : tensor<32x151296xf32>
    %v201 = chlo.erfc %v200 : tensor<32x151296xf32> -> tensor<32x151296xf32>
    %v202 = stablehlo.multiply %v197, %v201 : tensor<32x151296xf32>
    %v203 = stablehlo.reshape %v202 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v204 = stablehlo.dot_general %v203, %b0_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x192xf32>) -> tensor<32x197x192xf32>
    %v205 = stablehlo.broadcast_in_dim %b0_bfc2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v206 = stablehlo.add %v204, %v205 : tensor<32x197x192xf32>
    %v207 = stablehlo.reshape %v206 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v208 = stablehlo.broadcast_in_dim %dp1, dims = [0] : (tensor<32xf32>) -> tensor<32x37824xf32>
    %v209 = stablehlo.multiply %v208, %v207 : tensor<32x37824xf32>
    %v210 = stablehlo.add %v162, %v209 : tensor<32x37824xf32>
    %v211 = stablehlo.reshape %v210 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v212 = stablehlo.constant dense<0.0> : tensor<f32>
    %v213 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v214 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v215 = stablehlo.reduce(%v211 init: %v212) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v216 = stablehlo.broadcast_in_dim %v215, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v217 = stablehlo.divide %v216, %v213 : tensor<32x197x192xf32>
    %v218 = stablehlo.subtract %v211, %v217 : tensor<32x197x192xf32>
    %v219 = stablehlo.multiply %v218, %v218 : tensor<32x197x192xf32>
    %v220 = stablehlo.reduce(%v219 init: %v212) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v221 = stablehlo.broadcast_in_dim %v220, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v222 = stablehlo.divide %v221, %v213 : tensor<32x197x192xf32>
    %v223 = stablehlo.add %v222, %v214 : tensor<32x197x192xf32>
    %v224 = stablehlo.rsqrt %v223 : tensor<32x197x192xf32>
    %v225 = stablehlo.multiply %v218, %v224 : tensor<32x197x192xf32>
    %v226 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v227 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v228 = stablehlo.multiply %v225, %v226 : tensor<32x197x192xf32>
    %v229 = stablehlo.add %v228, %v227 : tensor<32x197x192xf32>
    %v230 = stablehlo.reshape %v229 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v231 = stablehlo.reshape %v230 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v232 = stablehlo.broadcast_in_dim %b1_g1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v233 = stablehlo.multiply %v231, %v232 : tensor<32x197x192xf32>
    %v234 = stablehlo.reshape %v233 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v235 = stablehlo.reshape %v234 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v236 = stablehlo.broadcast_in_dim %b1_bt1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v237 = stablehlo.add %v235, %v236 : tensor<32x197x192xf32>
    %v238 = stablehlo.reshape %v237 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v239 = stablehlo.reshape %v238 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v240 = stablehlo.dot_general %v239, %b1_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v241 = stablehlo.broadcast_in_dim %b1_bq, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v242 = stablehlo.add %v240, %v241 : tensor<32x197x192xf32>
    %v243 = stablehlo.reshape %v242 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v244 = stablehlo.reshape %v238 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v245 = stablehlo.dot_general %v244, %b1_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v246 = stablehlo.broadcast_in_dim %b1_bk, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v247 = stablehlo.add %v245, %v246 : tensor<32x197x192xf32>
    %v248 = stablehlo.reshape %v247 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v249 = stablehlo.reshape %v238 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v250 = stablehlo.dot_general %v249, %b1_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v251 = stablehlo.broadcast_in_dim %b1_bv, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v252 = stablehlo.add %v250, %v251 : tensor<32x197x192xf32>
    %v253 = stablehlo.reshape %v252 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v254 = stablehlo.reshape %v243 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v255 = stablehlo.slice %v254 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v256 = stablehlo.reshape %v255 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v257 = stablehlo.reshape %v248 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v258 = stablehlo.slice %v257 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v259 = stablehlo.reshape %v258 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v260 = stablehlo.reshape %v253 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v261 = stablehlo.slice %v260 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v262 = stablehlo.reshape %v261 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v263 = stablehlo.reshape %v259 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v264 = stablehlo.transpose %v263, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v265 = stablehlo.reshape %v264 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v266 = stablehlo.reshape %v256 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v267 = stablehlo.reshape %v265 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v268 = stablehlo.dot_general %v266, %v267, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v269 = stablehlo.reshape %v268 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v270 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v271 = stablehlo.multiply %v269, %v270 : tensor<32x38809xf32>
    %v272 = stablehlo.reshape %v271 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v273 = stablehlo.constant dense<0.0> : tensor<f32>
    %v274 = stablehlo.exponential %v272 : tensor<32x197x197xf32>
    %v275 = stablehlo.reduce(%v274 init: %v273) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v276 = stablehlo.broadcast_in_dim %v275, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v277 = stablehlo.divide %v274, %v276 : tensor<32x197x197xf32>
    %v278 = stablehlo.reshape %v277 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v279 = stablehlo.reshape %v278 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v280 = stablehlo.reshape %v262 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v281 = stablehlo.dot_general %v279, %v280, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v282 = stablehlo.reshape %v281 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v283 = stablehlo.reshape %v282 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v284 = stablehlo.constant dense<0.0> : tensor<f32>
    %v285 = stablehlo.pad %v283, %v284, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v286 = stablehlo.reshape %v285 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v287 = stablehlo.reshape %v243 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v288 = stablehlo.slice %v287 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v289 = stablehlo.reshape %v288 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v290 = stablehlo.reshape %v248 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v291 = stablehlo.slice %v290 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v292 = stablehlo.reshape %v291 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v293 = stablehlo.reshape %v253 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v294 = stablehlo.slice %v293 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v295 = stablehlo.reshape %v294 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v296 = stablehlo.reshape %v292 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v297 = stablehlo.transpose %v296, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v298 = stablehlo.reshape %v297 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v299 = stablehlo.reshape %v289 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v300 = stablehlo.reshape %v298 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v301 = stablehlo.dot_general %v299, %v300, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v302 = stablehlo.reshape %v301 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v303 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v304 = stablehlo.multiply %v302, %v303 : tensor<32x38809xf32>
    %v305 = stablehlo.reshape %v304 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v306 = stablehlo.constant dense<0.0> : tensor<f32>
    %v307 = stablehlo.exponential %v305 : tensor<32x197x197xf32>
    %v308 = stablehlo.reduce(%v307 init: %v306) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v309 = stablehlo.broadcast_in_dim %v308, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v310 = stablehlo.divide %v307, %v309 : tensor<32x197x197xf32>
    %v311 = stablehlo.reshape %v310 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v312 = stablehlo.reshape %v311 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v313 = stablehlo.reshape %v295 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v314 = stablehlo.dot_general %v312, %v313, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v315 = stablehlo.reshape %v314 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v316 = stablehlo.reshape %v315 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v317 = stablehlo.constant dense<0.0> : tensor<f32>
    %v318 = stablehlo.pad %v316, %v317, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v319 = stablehlo.reshape %v318 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v320 = stablehlo.add %v286, %v319 : tensor<32x37824xf32>
    %v321 = stablehlo.reshape %v243 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v322 = stablehlo.slice %v321 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v323 = stablehlo.reshape %v322 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v324 = stablehlo.reshape %v248 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v325 = stablehlo.slice %v324 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v326 = stablehlo.reshape %v325 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v327 = stablehlo.reshape %v253 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v328 = stablehlo.slice %v327 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v329 = stablehlo.reshape %v328 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v330 = stablehlo.reshape %v326 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v331 = stablehlo.transpose %v330, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v332 = stablehlo.reshape %v331 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v333 = stablehlo.reshape %v323 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v334 = stablehlo.reshape %v332 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v335 = stablehlo.dot_general %v333, %v334, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v336 = stablehlo.reshape %v335 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v337 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v338 = stablehlo.multiply %v336, %v337 : tensor<32x38809xf32>
    %v339 = stablehlo.reshape %v338 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v340 = stablehlo.constant dense<0.0> : tensor<f32>
    %v341 = stablehlo.exponential %v339 : tensor<32x197x197xf32>
    %v342 = stablehlo.reduce(%v341 init: %v340) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v343 = stablehlo.broadcast_in_dim %v342, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v344 = stablehlo.divide %v341, %v343 : tensor<32x197x197xf32>
    %v345 = stablehlo.reshape %v344 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v346 = stablehlo.reshape %v345 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v347 = stablehlo.reshape %v329 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v348 = stablehlo.dot_general %v346, %v347, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v349 = stablehlo.reshape %v348 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v350 = stablehlo.reshape %v349 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v351 = stablehlo.constant dense<0.0> : tensor<f32>
    %v352 = stablehlo.pad %v350, %v351, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v353 = stablehlo.reshape %v352 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v354 = stablehlo.add %v320, %v353 : tensor<32x37824xf32>
    %v355 = stablehlo.reshape %v354 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v356 = stablehlo.dot_general %v355, %b1_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v357 = stablehlo.broadcast_in_dim %b1_bo, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v358 = stablehlo.add %v356, %v357 : tensor<32x197x192xf32>
    %v359 = stablehlo.reshape %v358 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v360 = stablehlo.broadcast_in_dim %dp2, dims = [0] : (tensor<32xf32>) -> tensor<32x37824xf32>
    %v361 = stablehlo.multiply %v360, %v359 : tensor<32x37824xf32>
    %v362 = stablehlo.add %v210, %v361 : tensor<32x37824xf32>
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
    %v396 = stablehlo.constant dense<0.5> : tensor<32x151296xf32>
    %v397 = stablehlo.multiply %v396, %v395 : tensor<32x151296xf32>
    %v398 = stablehlo.negate %v395 : tensor<32x151296xf32>
    %v399 = stablehlo.constant dense<0.7071067811865476> : tensor<32x151296xf32>
    %v400 = stablehlo.multiply %v398, %v399 : tensor<32x151296xf32>
    %v401 = chlo.erfc %v400 : tensor<32x151296xf32> -> tensor<32x151296xf32>
    %v402 = stablehlo.multiply %v397, %v401 : tensor<32x151296xf32>
    %v403 = stablehlo.reshape %v402 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v404 = stablehlo.dot_general %v403, %b1_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x192xf32>) -> tensor<32x197x192xf32>
    %v405 = stablehlo.broadcast_in_dim %b1_bfc2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v406 = stablehlo.add %v404, %v405 : tensor<32x197x192xf32>
    %v407 = stablehlo.reshape %v406 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v408 = stablehlo.broadcast_in_dim %dp3, dims = [0] : (tensor<32xf32>) -> tensor<32x37824xf32>
    %v409 = stablehlo.multiply %v408, %v407 : tensor<32x37824xf32>
    %v410 = stablehlo.add %v362, %v409 : tensor<32x37824xf32>
    %v411 = stablehlo.reshape %v410 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v412 = stablehlo.constant dense<0.0> : tensor<f32>
    %v413 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v414 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v415 = stablehlo.reduce(%v411 init: %v412) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v416 = stablehlo.broadcast_in_dim %v415, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v417 = stablehlo.divide %v416, %v413 : tensor<32x197x192xf32>
    %v418 = stablehlo.subtract %v411, %v417 : tensor<32x197x192xf32>
    %v419 = stablehlo.multiply %v418, %v418 : tensor<32x197x192xf32>
    %v420 = stablehlo.reduce(%v419 init: %v412) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v421 = stablehlo.broadcast_in_dim %v420, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v422 = stablehlo.divide %v421, %v413 : tensor<32x197x192xf32>
    %v423 = stablehlo.add %v422, %v414 : tensor<32x197x192xf32>
    %v424 = stablehlo.rsqrt %v423 : tensor<32x197x192xf32>
    %v425 = stablehlo.multiply %v418, %v424 : tensor<32x197x192xf32>
    %v426 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v427 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v428 = stablehlo.multiply %v425, %v426 : tensor<32x197x192xf32>
    %v429 = stablehlo.add %v428, %v427 : tensor<32x197x192xf32>
    %v430 = stablehlo.reshape %v429 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v431 = stablehlo.reshape %v430 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v432 = stablehlo.broadcast_in_dim %b2_g1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v433 = stablehlo.multiply %v431, %v432 : tensor<32x197x192xf32>
    %v434 = stablehlo.reshape %v433 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v435 = stablehlo.reshape %v434 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v436 = stablehlo.broadcast_in_dim %b2_bt1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v437 = stablehlo.add %v435, %v436 : tensor<32x197x192xf32>
    %v438 = stablehlo.reshape %v437 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v439 = stablehlo.reshape %v438 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v440 = stablehlo.dot_general %v439, %b2_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v441 = stablehlo.broadcast_in_dim %b2_bq, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v442 = stablehlo.add %v440, %v441 : tensor<32x197x192xf32>
    %v443 = stablehlo.reshape %v442 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v444 = stablehlo.reshape %v438 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v445 = stablehlo.dot_general %v444, %b2_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v446 = stablehlo.broadcast_in_dim %b2_bk, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v447 = stablehlo.add %v445, %v446 : tensor<32x197x192xf32>
    %v448 = stablehlo.reshape %v447 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v449 = stablehlo.reshape %v438 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v450 = stablehlo.dot_general %v449, %b2_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v451 = stablehlo.broadcast_in_dim %b2_bv, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v452 = stablehlo.add %v450, %v451 : tensor<32x197x192xf32>
    %v453 = stablehlo.reshape %v452 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v454 = stablehlo.reshape %v443 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v455 = stablehlo.slice %v454 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v456 = stablehlo.reshape %v455 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v457 = stablehlo.reshape %v448 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v458 = stablehlo.slice %v457 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v459 = stablehlo.reshape %v458 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v460 = stablehlo.reshape %v453 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v461 = stablehlo.slice %v460 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v462 = stablehlo.reshape %v461 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v463 = stablehlo.reshape %v459 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v464 = stablehlo.transpose %v463, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v465 = stablehlo.reshape %v464 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v466 = stablehlo.reshape %v456 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v467 = stablehlo.reshape %v465 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v468 = stablehlo.dot_general %v466, %v467, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v469 = stablehlo.reshape %v468 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v470 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v471 = stablehlo.multiply %v469, %v470 : tensor<32x38809xf32>
    %v472 = stablehlo.reshape %v471 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v473 = stablehlo.constant dense<0.0> : tensor<f32>
    %v474 = stablehlo.exponential %v472 : tensor<32x197x197xf32>
    %v475 = stablehlo.reduce(%v474 init: %v473) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v476 = stablehlo.broadcast_in_dim %v475, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v477 = stablehlo.divide %v474, %v476 : tensor<32x197x197xf32>
    %v478 = stablehlo.reshape %v477 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v479 = stablehlo.reshape %v478 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v480 = stablehlo.reshape %v462 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v481 = stablehlo.dot_general %v479, %v480, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v482 = stablehlo.reshape %v481 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v483 = stablehlo.reshape %v482 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v484 = stablehlo.constant dense<0.0> : tensor<f32>
    %v485 = stablehlo.pad %v483, %v484, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v486 = stablehlo.reshape %v485 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v487 = stablehlo.reshape %v443 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v488 = stablehlo.slice %v487 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v489 = stablehlo.reshape %v488 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v490 = stablehlo.reshape %v448 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v491 = stablehlo.slice %v490 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v492 = stablehlo.reshape %v491 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v493 = stablehlo.reshape %v453 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v494 = stablehlo.slice %v493 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
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
    %v518 = stablehlo.pad %v516, %v517, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v519 = stablehlo.reshape %v518 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v520 = stablehlo.add %v486, %v519 : tensor<32x37824xf32>
    %v521 = stablehlo.reshape %v443 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v522 = stablehlo.slice %v521 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v523 = stablehlo.reshape %v522 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v524 = stablehlo.reshape %v448 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v525 = stablehlo.slice %v524 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v526 = stablehlo.reshape %v525 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v527 = stablehlo.reshape %v453 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v528 = stablehlo.slice %v527 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
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
    %v552 = stablehlo.pad %v550, %v551, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v553 = stablehlo.reshape %v552 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v554 = stablehlo.add %v520, %v553 : tensor<32x37824xf32>
    %v555 = stablehlo.reshape %v554 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v556 = stablehlo.dot_general %v555, %b2_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v557 = stablehlo.broadcast_in_dim %b2_bo, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v558 = stablehlo.add %v556, %v557 : tensor<32x197x192xf32>
    %v559 = stablehlo.reshape %v558 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v560 = stablehlo.broadcast_in_dim %dp4, dims = [0] : (tensor<32xf32>) -> tensor<32x37824xf32>
    %v561 = stablehlo.multiply %v560, %v559 : tensor<32x37824xf32>
    %v562 = stablehlo.add %v410, %v561 : tensor<32x37824xf32>
    %v563 = stablehlo.reshape %v562 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v564 = stablehlo.constant dense<0.0> : tensor<f32>
    %v565 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v566 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v567 = stablehlo.reduce(%v563 init: %v564) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v568 = stablehlo.broadcast_in_dim %v567, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v569 = stablehlo.divide %v568, %v565 : tensor<32x197x192xf32>
    %v570 = stablehlo.subtract %v563, %v569 : tensor<32x197x192xf32>
    %v571 = stablehlo.multiply %v570, %v570 : tensor<32x197x192xf32>
    %v572 = stablehlo.reduce(%v571 init: %v564) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v573 = stablehlo.broadcast_in_dim %v572, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v574 = stablehlo.divide %v573, %v565 : tensor<32x197x192xf32>
    %v575 = stablehlo.add %v574, %v566 : tensor<32x197x192xf32>
    %v576 = stablehlo.rsqrt %v575 : tensor<32x197x192xf32>
    %v577 = stablehlo.multiply %v570, %v576 : tensor<32x197x192xf32>
    %v578 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v579 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v580 = stablehlo.multiply %v577, %v578 : tensor<32x197x192xf32>
    %v581 = stablehlo.add %v580, %v579 : tensor<32x197x192xf32>
    %v582 = stablehlo.reshape %v581 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v583 = stablehlo.reshape %v582 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v584 = stablehlo.broadcast_in_dim %b2_g2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v585 = stablehlo.multiply %v583, %v584 : tensor<32x197x192xf32>
    %v586 = stablehlo.reshape %v585 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v587 = stablehlo.reshape %v586 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v588 = stablehlo.broadcast_in_dim %b2_bt2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v589 = stablehlo.add %v587, %v588 : tensor<32x197x192xf32>
    %v590 = stablehlo.reshape %v589 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v591 = stablehlo.reshape %v590 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v592 = stablehlo.dot_general %v591, %b2_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x768xf32>) -> tensor<32x197x768xf32>
    %v593 = stablehlo.broadcast_in_dim %b2_bfc1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v594 = stablehlo.add %v592, %v593 : tensor<32x197x768xf32>
    %v595 = stablehlo.reshape %v594 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v596 = stablehlo.constant dense<0.5> : tensor<32x151296xf32>
    %v597 = stablehlo.multiply %v596, %v595 : tensor<32x151296xf32>
    %v598 = stablehlo.negate %v595 : tensor<32x151296xf32>
    %v599 = stablehlo.constant dense<0.7071067811865476> : tensor<32x151296xf32>
    %v600 = stablehlo.multiply %v598, %v599 : tensor<32x151296xf32>
    %v601 = chlo.erfc %v600 : tensor<32x151296xf32> -> tensor<32x151296xf32>
    %v602 = stablehlo.multiply %v597, %v601 : tensor<32x151296xf32>
    %v603 = stablehlo.reshape %v602 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v604 = stablehlo.dot_general %v603, %b2_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x192xf32>) -> tensor<32x197x192xf32>
    %v605 = stablehlo.broadcast_in_dim %b2_bfc2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v606 = stablehlo.add %v604, %v605 : tensor<32x197x192xf32>
    %v607 = stablehlo.reshape %v606 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v608 = stablehlo.broadcast_in_dim %dp5, dims = [0] : (tensor<32xf32>) -> tensor<32x37824xf32>
    %v609 = stablehlo.multiply %v608, %v607 : tensor<32x37824xf32>
    %v610 = stablehlo.add %v562, %v609 : tensor<32x37824xf32>
    %v611 = stablehlo.reshape %v610 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v612 = stablehlo.constant dense<0.0> : tensor<f32>
    %v613 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v614 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v615 = stablehlo.reduce(%v611 init: %v612) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v616 = stablehlo.broadcast_in_dim %v615, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v617 = stablehlo.divide %v616, %v613 : tensor<32x197x192xf32>
    %v618 = stablehlo.subtract %v611, %v617 : tensor<32x197x192xf32>
    %v619 = stablehlo.multiply %v618, %v618 : tensor<32x197x192xf32>
    %v620 = stablehlo.reduce(%v619 init: %v612) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v621 = stablehlo.broadcast_in_dim %v620, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v622 = stablehlo.divide %v621, %v613 : tensor<32x197x192xf32>
    %v623 = stablehlo.add %v622, %v614 : tensor<32x197x192xf32>
    %v624 = stablehlo.rsqrt %v623 : tensor<32x197x192xf32>
    %v625 = stablehlo.multiply %v618, %v624 : tensor<32x197x192xf32>
    %v626 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v627 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v628 = stablehlo.multiply %v625, %v626 : tensor<32x197x192xf32>
    %v629 = stablehlo.add %v628, %v627 : tensor<32x197x192xf32>
    %v630 = stablehlo.reshape %v629 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v631 = stablehlo.reshape %v630 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v632 = stablehlo.broadcast_in_dim %b3_g1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v633 = stablehlo.multiply %v631, %v632 : tensor<32x197x192xf32>
    %v634 = stablehlo.reshape %v633 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v635 = stablehlo.reshape %v634 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v636 = stablehlo.broadcast_in_dim %b3_bt1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v637 = stablehlo.add %v635, %v636 : tensor<32x197x192xf32>
    %v638 = stablehlo.reshape %v637 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v639 = stablehlo.reshape %v638 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v640 = stablehlo.dot_general %v639, %b3_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v641 = stablehlo.broadcast_in_dim %b3_bq, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v642 = stablehlo.add %v640, %v641 : tensor<32x197x192xf32>
    %v643 = stablehlo.reshape %v642 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v644 = stablehlo.reshape %v638 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v645 = stablehlo.dot_general %v644, %b3_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v646 = stablehlo.broadcast_in_dim %b3_bk, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v647 = stablehlo.add %v645, %v646 : tensor<32x197x192xf32>
    %v648 = stablehlo.reshape %v647 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v649 = stablehlo.reshape %v638 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v650 = stablehlo.dot_general %v649, %b3_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v651 = stablehlo.broadcast_in_dim %b3_bv, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v652 = stablehlo.add %v650, %v651 : tensor<32x197x192xf32>
    %v653 = stablehlo.reshape %v652 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v654 = stablehlo.reshape %v643 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v655 = stablehlo.slice %v654 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v656 = stablehlo.reshape %v655 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v657 = stablehlo.reshape %v648 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v658 = stablehlo.slice %v657 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v659 = stablehlo.reshape %v658 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v660 = stablehlo.reshape %v653 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v661 = stablehlo.slice %v660 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v662 = stablehlo.reshape %v661 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v663 = stablehlo.reshape %v659 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v664 = stablehlo.transpose %v663, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v665 = stablehlo.reshape %v664 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v666 = stablehlo.reshape %v656 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v667 = stablehlo.reshape %v665 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v668 = stablehlo.dot_general %v666, %v667, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v669 = stablehlo.reshape %v668 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v670 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v671 = stablehlo.multiply %v669, %v670 : tensor<32x38809xf32>
    %v672 = stablehlo.reshape %v671 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v673 = stablehlo.constant dense<0.0> : tensor<f32>
    %v674 = stablehlo.exponential %v672 : tensor<32x197x197xf32>
    %v675 = stablehlo.reduce(%v674 init: %v673) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v676 = stablehlo.broadcast_in_dim %v675, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v677 = stablehlo.divide %v674, %v676 : tensor<32x197x197xf32>
    %v678 = stablehlo.reshape %v677 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v679 = stablehlo.reshape %v678 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v680 = stablehlo.reshape %v662 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v681 = stablehlo.dot_general %v679, %v680, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v682 = stablehlo.reshape %v681 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v683 = stablehlo.reshape %v682 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v684 = stablehlo.constant dense<0.0> : tensor<f32>
    %v685 = stablehlo.pad %v683, %v684, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v686 = stablehlo.reshape %v685 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v687 = stablehlo.reshape %v643 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v688 = stablehlo.slice %v687 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v689 = stablehlo.reshape %v688 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v690 = stablehlo.reshape %v648 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v691 = stablehlo.slice %v690 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v692 = stablehlo.reshape %v691 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v693 = stablehlo.reshape %v653 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v694 = stablehlo.slice %v693 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v695 = stablehlo.reshape %v694 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v696 = stablehlo.reshape %v692 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v697 = stablehlo.transpose %v696, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v698 = stablehlo.reshape %v697 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v699 = stablehlo.reshape %v689 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v700 = stablehlo.reshape %v698 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v701 = stablehlo.dot_general %v699, %v700, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v702 = stablehlo.reshape %v701 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v703 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v704 = stablehlo.multiply %v702, %v703 : tensor<32x38809xf32>
    %v705 = stablehlo.reshape %v704 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v706 = stablehlo.constant dense<0.0> : tensor<f32>
    %v707 = stablehlo.exponential %v705 : tensor<32x197x197xf32>
    %v708 = stablehlo.reduce(%v707 init: %v706) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v709 = stablehlo.broadcast_in_dim %v708, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v710 = stablehlo.divide %v707, %v709 : tensor<32x197x197xf32>
    %v711 = stablehlo.reshape %v710 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v712 = stablehlo.reshape %v711 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v713 = stablehlo.reshape %v695 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v714 = stablehlo.dot_general %v712, %v713, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v715 = stablehlo.reshape %v714 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v716 = stablehlo.reshape %v715 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v717 = stablehlo.constant dense<0.0> : tensor<f32>
    %v718 = stablehlo.pad %v716, %v717, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v719 = stablehlo.reshape %v718 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v720 = stablehlo.add %v686, %v719 : tensor<32x37824xf32>
    %v721 = stablehlo.reshape %v643 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v722 = stablehlo.slice %v721 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v723 = stablehlo.reshape %v722 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v724 = stablehlo.reshape %v648 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v725 = stablehlo.slice %v724 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v726 = stablehlo.reshape %v725 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v727 = stablehlo.reshape %v653 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v728 = stablehlo.slice %v727 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v729 = stablehlo.reshape %v728 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v730 = stablehlo.reshape %v726 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v731 = stablehlo.transpose %v730, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v732 = stablehlo.reshape %v731 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v733 = stablehlo.reshape %v723 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v734 = stablehlo.reshape %v732 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v735 = stablehlo.dot_general %v733, %v734, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v736 = stablehlo.reshape %v735 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v737 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v738 = stablehlo.multiply %v736, %v737 : tensor<32x38809xf32>
    %v739 = stablehlo.reshape %v738 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v740 = stablehlo.constant dense<0.0> : tensor<f32>
    %v741 = stablehlo.exponential %v739 : tensor<32x197x197xf32>
    %v742 = stablehlo.reduce(%v741 init: %v740) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v743 = stablehlo.broadcast_in_dim %v742, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v744 = stablehlo.divide %v741, %v743 : tensor<32x197x197xf32>
    %v745 = stablehlo.reshape %v744 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v746 = stablehlo.reshape %v745 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v747 = stablehlo.reshape %v729 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v748 = stablehlo.dot_general %v746, %v747, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v749 = stablehlo.reshape %v748 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v750 = stablehlo.reshape %v749 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v751 = stablehlo.constant dense<0.0> : tensor<f32>
    %v752 = stablehlo.pad %v750, %v751, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v753 = stablehlo.reshape %v752 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v754 = stablehlo.add %v720, %v753 : tensor<32x37824xf32>
    %v755 = stablehlo.reshape %v754 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v756 = stablehlo.dot_general %v755, %b3_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v757 = stablehlo.broadcast_in_dim %b3_bo, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v758 = stablehlo.add %v756, %v757 : tensor<32x197x192xf32>
    %v759 = stablehlo.reshape %v758 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v760 = stablehlo.broadcast_in_dim %dp6, dims = [0] : (tensor<32xf32>) -> tensor<32x37824xf32>
    %v761 = stablehlo.multiply %v760, %v759 : tensor<32x37824xf32>
    %v762 = stablehlo.add %v610, %v761 : tensor<32x37824xf32>
    %v763 = stablehlo.reshape %v762 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v764 = stablehlo.constant dense<0.0> : tensor<f32>
    %v765 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v766 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v767 = stablehlo.reduce(%v763 init: %v764) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v768 = stablehlo.broadcast_in_dim %v767, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v769 = stablehlo.divide %v768, %v765 : tensor<32x197x192xf32>
    %v770 = stablehlo.subtract %v763, %v769 : tensor<32x197x192xf32>
    %v771 = stablehlo.multiply %v770, %v770 : tensor<32x197x192xf32>
    %v772 = stablehlo.reduce(%v771 init: %v764) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v773 = stablehlo.broadcast_in_dim %v772, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v774 = stablehlo.divide %v773, %v765 : tensor<32x197x192xf32>
    %v775 = stablehlo.add %v774, %v766 : tensor<32x197x192xf32>
    %v776 = stablehlo.rsqrt %v775 : tensor<32x197x192xf32>
    %v777 = stablehlo.multiply %v770, %v776 : tensor<32x197x192xf32>
    %v778 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v779 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v780 = stablehlo.multiply %v777, %v778 : tensor<32x197x192xf32>
    %v781 = stablehlo.add %v780, %v779 : tensor<32x197x192xf32>
    %v782 = stablehlo.reshape %v781 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v783 = stablehlo.reshape %v782 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v784 = stablehlo.broadcast_in_dim %b3_g2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v785 = stablehlo.multiply %v783, %v784 : tensor<32x197x192xf32>
    %v786 = stablehlo.reshape %v785 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v787 = stablehlo.reshape %v786 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v788 = stablehlo.broadcast_in_dim %b3_bt2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v789 = stablehlo.add %v787, %v788 : tensor<32x197x192xf32>
    %v790 = stablehlo.reshape %v789 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v791 = stablehlo.reshape %v790 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v792 = stablehlo.dot_general %v791, %b3_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x768xf32>) -> tensor<32x197x768xf32>
    %v793 = stablehlo.broadcast_in_dim %b3_bfc1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v794 = stablehlo.add %v792, %v793 : tensor<32x197x768xf32>
    %v795 = stablehlo.reshape %v794 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v796 = stablehlo.constant dense<0.5> : tensor<32x151296xf32>
    %v797 = stablehlo.multiply %v796, %v795 : tensor<32x151296xf32>
    %v798 = stablehlo.negate %v795 : tensor<32x151296xf32>
    %v799 = stablehlo.constant dense<0.7071067811865476> : tensor<32x151296xf32>
    %v800 = stablehlo.multiply %v798, %v799 : tensor<32x151296xf32>
    %v801 = chlo.erfc %v800 : tensor<32x151296xf32> -> tensor<32x151296xf32>
    %v802 = stablehlo.multiply %v797, %v801 : tensor<32x151296xf32>
    %v803 = stablehlo.reshape %v802 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v804 = stablehlo.dot_general %v803, %b3_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x192xf32>) -> tensor<32x197x192xf32>
    %v805 = stablehlo.broadcast_in_dim %b3_bfc2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v806 = stablehlo.add %v804, %v805 : tensor<32x197x192xf32>
    %v807 = stablehlo.reshape %v806 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v808 = stablehlo.broadcast_in_dim %dp7, dims = [0] : (tensor<32xf32>) -> tensor<32x37824xf32>
    %v809 = stablehlo.multiply %v808, %v807 : tensor<32x37824xf32>
    %v810 = stablehlo.add %v762, %v809 : tensor<32x37824xf32>
    %v811 = stablehlo.reshape %v810 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v812 = stablehlo.constant dense<0.0> : tensor<f32>
    %v813 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v814 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v815 = stablehlo.reduce(%v811 init: %v812) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v816 = stablehlo.broadcast_in_dim %v815, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v817 = stablehlo.divide %v816, %v813 : tensor<32x197x192xf32>
    %v818 = stablehlo.subtract %v811, %v817 : tensor<32x197x192xf32>
    %v819 = stablehlo.multiply %v818, %v818 : tensor<32x197x192xf32>
    %v820 = stablehlo.reduce(%v819 init: %v812) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v821 = stablehlo.broadcast_in_dim %v820, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v822 = stablehlo.divide %v821, %v813 : tensor<32x197x192xf32>
    %v823 = stablehlo.add %v822, %v814 : tensor<32x197x192xf32>
    %v824 = stablehlo.rsqrt %v823 : tensor<32x197x192xf32>
    %v825 = stablehlo.multiply %v818, %v824 : tensor<32x197x192xf32>
    %v826 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v827 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v828 = stablehlo.multiply %v825, %v826 : tensor<32x197x192xf32>
    %v829 = stablehlo.add %v828, %v827 : tensor<32x197x192xf32>
    %v830 = stablehlo.reshape %v829 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v831 = stablehlo.reshape %v830 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v832 = stablehlo.broadcast_in_dim %b4_g1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v833 = stablehlo.multiply %v831, %v832 : tensor<32x197x192xf32>
    %v834 = stablehlo.reshape %v833 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v835 = stablehlo.reshape %v834 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v836 = stablehlo.broadcast_in_dim %b4_bt1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v837 = stablehlo.add %v835, %v836 : tensor<32x197x192xf32>
    %v838 = stablehlo.reshape %v837 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v839 = stablehlo.reshape %v838 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v840 = stablehlo.dot_general %v839, %b4_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v841 = stablehlo.broadcast_in_dim %b4_bq, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v842 = stablehlo.add %v840, %v841 : tensor<32x197x192xf32>
    %v843 = stablehlo.reshape %v842 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v844 = stablehlo.reshape %v838 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v845 = stablehlo.dot_general %v844, %b4_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v846 = stablehlo.broadcast_in_dim %b4_bk, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v847 = stablehlo.add %v845, %v846 : tensor<32x197x192xf32>
    %v848 = stablehlo.reshape %v847 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v849 = stablehlo.reshape %v838 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v850 = stablehlo.dot_general %v849, %b4_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v851 = stablehlo.broadcast_in_dim %b4_bv, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v852 = stablehlo.add %v850, %v851 : tensor<32x197x192xf32>
    %v853 = stablehlo.reshape %v852 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v854 = stablehlo.reshape %v843 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v855 = stablehlo.slice %v854 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v856 = stablehlo.reshape %v855 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v857 = stablehlo.reshape %v848 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v858 = stablehlo.slice %v857 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v859 = stablehlo.reshape %v858 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v860 = stablehlo.reshape %v853 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v861 = stablehlo.slice %v860 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v862 = stablehlo.reshape %v861 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v863 = stablehlo.reshape %v859 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v864 = stablehlo.transpose %v863, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v865 = stablehlo.reshape %v864 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v866 = stablehlo.reshape %v856 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v867 = stablehlo.reshape %v865 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v868 = stablehlo.dot_general %v866, %v867, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v869 = stablehlo.reshape %v868 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v870 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v871 = stablehlo.multiply %v869, %v870 : tensor<32x38809xf32>
    %v872 = stablehlo.reshape %v871 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v873 = stablehlo.constant dense<0.0> : tensor<f32>
    %v874 = stablehlo.exponential %v872 : tensor<32x197x197xf32>
    %v875 = stablehlo.reduce(%v874 init: %v873) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v876 = stablehlo.broadcast_in_dim %v875, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v877 = stablehlo.divide %v874, %v876 : tensor<32x197x197xf32>
    %v878 = stablehlo.reshape %v877 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v879 = stablehlo.reshape %v878 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v880 = stablehlo.reshape %v862 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v881 = stablehlo.dot_general %v879, %v880, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v882 = stablehlo.reshape %v881 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v883 = stablehlo.reshape %v882 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v884 = stablehlo.constant dense<0.0> : tensor<f32>
    %v885 = stablehlo.pad %v883, %v884, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v886 = stablehlo.reshape %v885 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v887 = stablehlo.reshape %v843 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v888 = stablehlo.slice %v887 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v889 = stablehlo.reshape %v888 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v890 = stablehlo.reshape %v848 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v891 = stablehlo.slice %v890 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v892 = stablehlo.reshape %v891 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v893 = stablehlo.reshape %v853 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v894 = stablehlo.slice %v893 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v895 = stablehlo.reshape %v894 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v896 = stablehlo.reshape %v892 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v897 = stablehlo.transpose %v896, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v898 = stablehlo.reshape %v897 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v899 = stablehlo.reshape %v889 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v900 = stablehlo.reshape %v898 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v901 = stablehlo.dot_general %v899, %v900, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v902 = stablehlo.reshape %v901 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v903 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v904 = stablehlo.multiply %v902, %v903 : tensor<32x38809xf32>
    %v905 = stablehlo.reshape %v904 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v906 = stablehlo.constant dense<0.0> : tensor<f32>
    %v907 = stablehlo.exponential %v905 : tensor<32x197x197xf32>
    %v908 = stablehlo.reduce(%v907 init: %v906) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v909 = stablehlo.broadcast_in_dim %v908, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v910 = stablehlo.divide %v907, %v909 : tensor<32x197x197xf32>
    %v911 = stablehlo.reshape %v910 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v912 = stablehlo.reshape %v911 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v913 = stablehlo.reshape %v895 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v914 = stablehlo.dot_general %v912, %v913, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v915 = stablehlo.reshape %v914 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v916 = stablehlo.reshape %v915 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v917 = stablehlo.constant dense<0.0> : tensor<f32>
    %v918 = stablehlo.pad %v916, %v917, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v919 = stablehlo.reshape %v918 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v920 = stablehlo.add %v886, %v919 : tensor<32x37824xf32>
    %v921 = stablehlo.reshape %v843 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v922 = stablehlo.slice %v921 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v923 = stablehlo.reshape %v922 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v924 = stablehlo.reshape %v848 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v925 = stablehlo.slice %v924 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v926 = stablehlo.reshape %v925 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v927 = stablehlo.reshape %v853 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v928 = stablehlo.slice %v927 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v929 = stablehlo.reshape %v928 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v930 = stablehlo.reshape %v926 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v931 = stablehlo.transpose %v930, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v932 = stablehlo.reshape %v931 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v933 = stablehlo.reshape %v923 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v934 = stablehlo.reshape %v932 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v935 = stablehlo.dot_general %v933, %v934, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v936 = stablehlo.reshape %v935 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v937 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v938 = stablehlo.multiply %v936, %v937 : tensor<32x38809xf32>
    %v939 = stablehlo.reshape %v938 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v940 = stablehlo.constant dense<0.0> : tensor<f32>
    %v941 = stablehlo.exponential %v939 : tensor<32x197x197xf32>
    %v942 = stablehlo.reduce(%v941 init: %v940) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v943 = stablehlo.broadcast_in_dim %v942, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v944 = stablehlo.divide %v941, %v943 : tensor<32x197x197xf32>
    %v945 = stablehlo.reshape %v944 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v946 = stablehlo.reshape %v945 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v947 = stablehlo.reshape %v929 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v948 = stablehlo.dot_general %v946, %v947, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v949 = stablehlo.reshape %v948 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v950 = stablehlo.reshape %v949 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v951 = stablehlo.constant dense<0.0> : tensor<f32>
    %v952 = stablehlo.pad %v950, %v951, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v953 = stablehlo.reshape %v952 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v954 = stablehlo.add %v920, %v953 : tensor<32x37824xf32>
    %v955 = stablehlo.reshape %v954 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v956 = stablehlo.dot_general %v955, %b4_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v957 = stablehlo.broadcast_in_dim %b4_bo, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v958 = stablehlo.add %v956, %v957 : tensor<32x197x192xf32>
    %v959 = stablehlo.reshape %v958 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v960 = stablehlo.broadcast_in_dim %dp8, dims = [0] : (tensor<32xf32>) -> tensor<32x37824xf32>
    %v961 = stablehlo.multiply %v960, %v959 : tensor<32x37824xf32>
    %v962 = stablehlo.add %v810, %v961 : tensor<32x37824xf32>
    %v963 = stablehlo.reshape %v962 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v964 = stablehlo.constant dense<0.0> : tensor<f32>
    %v965 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v966 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v967 = stablehlo.reduce(%v963 init: %v964) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v968 = stablehlo.broadcast_in_dim %v967, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v969 = stablehlo.divide %v968, %v965 : tensor<32x197x192xf32>
    %v970 = stablehlo.subtract %v963, %v969 : tensor<32x197x192xf32>
    %v971 = stablehlo.multiply %v970, %v970 : tensor<32x197x192xf32>
    %v972 = stablehlo.reduce(%v971 init: %v964) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v973 = stablehlo.broadcast_in_dim %v972, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v974 = stablehlo.divide %v973, %v965 : tensor<32x197x192xf32>
    %v975 = stablehlo.add %v974, %v966 : tensor<32x197x192xf32>
    %v976 = stablehlo.rsqrt %v975 : tensor<32x197x192xf32>
    %v977 = stablehlo.multiply %v970, %v976 : tensor<32x197x192xf32>
    %v978 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v979 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v980 = stablehlo.multiply %v977, %v978 : tensor<32x197x192xf32>
    %v981 = stablehlo.add %v980, %v979 : tensor<32x197x192xf32>
    %v982 = stablehlo.reshape %v981 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v983 = stablehlo.reshape %v982 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v984 = stablehlo.broadcast_in_dim %b4_g2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v985 = stablehlo.multiply %v983, %v984 : tensor<32x197x192xf32>
    %v986 = stablehlo.reshape %v985 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v987 = stablehlo.reshape %v986 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v988 = stablehlo.broadcast_in_dim %b4_bt2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v989 = stablehlo.add %v987, %v988 : tensor<32x197x192xf32>
    %v990 = stablehlo.reshape %v989 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v991 = stablehlo.reshape %v990 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v992 = stablehlo.dot_general %v991, %b4_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x768xf32>) -> tensor<32x197x768xf32>
    %v993 = stablehlo.broadcast_in_dim %b4_bfc1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v994 = stablehlo.add %v992, %v993 : tensor<32x197x768xf32>
    %v995 = stablehlo.reshape %v994 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v996 = stablehlo.constant dense<0.5> : tensor<32x151296xf32>
    %v997 = stablehlo.multiply %v996, %v995 : tensor<32x151296xf32>
    %v998 = stablehlo.negate %v995 : tensor<32x151296xf32>
    %v999 = stablehlo.constant dense<0.7071067811865476> : tensor<32x151296xf32>
    %v1000 = stablehlo.multiply %v998, %v999 : tensor<32x151296xf32>
    %v1001 = chlo.erfc %v1000 : tensor<32x151296xf32> -> tensor<32x151296xf32>
    %v1002 = stablehlo.multiply %v997, %v1001 : tensor<32x151296xf32>
    %v1003 = stablehlo.reshape %v1002 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1004 = stablehlo.dot_general %v1003, %b4_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x192xf32>) -> tensor<32x197x192xf32>
    %v1005 = stablehlo.broadcast_in_dim %b4_bfc2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1006 = stablehlo.add %v1004, %v1005 : tensor<32x197x192xf32>
    %v1007 = stablehlo.reshape %v1006 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1008 = stablehlo.broadcast_in_dim %dp9, dims = [0] : (tensor<32xf32>) -> tensor<32x37824xf32>
    %v1009 = stablehlo.multiply %v1008, %v1007 : tensor<32x37824xf32>
    %v1010 = stablehlo.add %v962, %v1009 : tensor<32x37824xf32>
    %v1011 = stablehlo.reshape %v1010 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1012 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1013 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v1014 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v1015 = stablehlo.reduce(%v1011 init: %v1012) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1016 = stablehlo.broadcast_in_dim %v1015, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v1017 = stablehlo.divide %v1016, %v1013 : tensor<32x197x192xf32>
    %v1018 = stablehlo.subtract %v1011, %v1017 : tensor<32x197x192xf32>
    %v1019 = stablehlo.multiply %v1018, %v1018 : tensor<32x197x192xf32>
    %v1020 = stablehlo.reduce(%v1019 init: %v1012) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1021 = stablehlo.broadcast_in_dim %v1020, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v1022 = stablehlo.divide %v1021, %v1013 : tensor<32x197x192xf32>
    %v1023 = stablehlo.add %v1022, %v1014 : tensor<32x197x192xf32>
    %v1024 = stablehlo.rsqrt %v1023 : tensor<32x197x192xf32>
    %v1025 = stablehlo.multiply %v1018, %v1024 : tensor<32x197x192xf32>
    %v1026 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v1027 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v1028 = stablehlo.multiply %v1025, %v1026 : tensor<32x197x192xf32>
    %v1029 = stablehlo.add %v1028, %v1027 : tensor<32x197x192xf32>
    %v1030 = stablehlo.reshape %v1029 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1031 = stablehlo.reshape %v1030 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1032 = stablehlo.broadcast_in_dim %b5_g1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1033 = stablehlo.multiply %v1031, %v1032 : tensor<32x197x192xf32>
    %v1034 = stablehlo.reshape %v1033 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1035 = stablehlo.reshape %v1034 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1036 = stablehlo.broadcast_in_dim %b5_bt1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1037 = stablehlo.add %v1035, %v1036 : tensor<32x197x192xf32>
    %v1038 = stablehlo.reshape %v1037 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1039 = stablehlo.reshape %v1038 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1040 = stablehlo.dot_general %v1039, %b5_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v1041 = stablehlo.broadcast_in_dim %b5_bq, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1042 = stablehlo.add %v1040, %v1041 : tensor<32x197x192xf32>
    %v1043 = stablehlo.reshape %v1042 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1044 = stablehlo.reshape %v1038 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1045 = stablehlo.dot_general %v1044, %b5_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v1046 = stablehlo.broadcast_in_dim %b5_bk, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1047 = stablehlo.add %v1045, %v1046 : tensor<32x197x192xf32>
    %v1048 = stablehlo.reshape %v1047 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1049 = stablehlo.reshape %v1038 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1050 = stablehlo.dot_general %v1049, %b5_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v1051 = stablehlo.broadcast_in_dim %b5_bv, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1052 = stablehlo.add %v1050, %v1051 : tensor<32x197x192xf32>
    %v1053 = stablehlo.reshape %v1052 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1054 = stablehlo.reshape %v1043 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1055 = stablehlo.slice %v1054 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1056 = stablehlo.reshape %v1055 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1057 = stablehlo.reshape %v1048 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1058 = stablehlo.slice %v1057 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1059 = stablehlo.reshape %v1058 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1060 = stablehlo.reshape %v1053 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1061 = stablehlo.slice %v1060 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1062 = stablehlo.reshape %v1061 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1063 = stablehlo.reshape %v1059 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1064 = stablehlo.transpose %v1063, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1065 = stablehlo.reshape %v1064 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1066 = stablehlo.reshape %v1056 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1067 = stablehlo.reshape %v1065 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1068 = stablehlo.dot_general %v1066, %v1067, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1069 = stablehlo.reshape %v1068 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1070 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1071 = stablehlo.multiply %v1069, %v1070 : tensor<32x38809xf32>
    %v1072 = stablehlo.reshape %v1071 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1073 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1074 = stablehlo.exponential %v1072 : tensor<32x197x197xf32>
    %v1075 = stablehlo.reduce(%v1074 init: %v1073) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1076 = stablehlo.broadcast_in_dim %v1075, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1077 = stablehlo.divide %v1074, %v1076 : tensor<32x197x197xf32>
    %v1078 = stablehlo.reshape %v1077 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1079 = stablehlo.reshape %v1078 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1080 = stablehlo.reshape %v1062 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1081 = stablehlo.dot_general %v1079, %v1080, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1082 = stablehlo.reshape %v1081 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1083 = stablehlo.reshape %v1082 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1084 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1085 = stablehlo.pad %v1083, %v1084, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v1086 = stablehlo.reshape %v1085 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1087 = stablehlo.reshape %v1043 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1088 = stablehlo.slice %v1087 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1089 = stablehlo.reshape %v1088 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1090 = stablehlo.reshape %v1048 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1091 = stablehlo.slice %v1090 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1092 = stablehlo.reshape %v1091 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1093 = stablehlo.reshape %v1053 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1094 = stablehlo.slice %v1093 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1095 = stablehlo.reshape %v1094 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1096 = stablehlo.reshape %v1092 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1097 = stablehlo.transpose %v1096, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1098 = stablehlo.reshape %v1097 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1099 = stablehlo.reshape %v1089 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1100 = stablehlo.reshape %v1098 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1101 = stablehlo.dot_general %v1099, %v1100, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1102 = stablehlo.reshape %v1101 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1103 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1104 = stablehlo.multiply %v1102, %v1103 : tensor<32x38809xf32>
    %v1105 = stablehlo.reshape %v1104 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1106 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1107 = stablehlo.exponential %v1105 : tensor<32x197x197xf32>
    %v1108 = stablehlo.reduce(%v1107 init: %v1106) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1109 = stablehlo.broadcast_in_dim %v1108, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1110 = stablehlo.divide %v1107, %v1109 : tensor<32x197x197xf32>
    %v1111 = stablehlo.reshape %v1110 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1112 = stablehlo.reshape %v1111 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1113 = stablehlo.reshape %v1095 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1114 = stablehlo.dot_general %v1112, %v1113, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1115 = stablehlo.reshape %v1114 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1116 = stablehlo.reshape %v1115 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1117 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1118 = stablehlo.pad %v1116, %v1117, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v1119 = stablehlo.reshape %v1118 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1120 = stablehlo.add %v1086, %v1119 : tensor<32x37824xf32>
    %v1121 = stablehlo.reshape %v1043 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1122 = stablehlo.slice %v1121 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1123 = stablehlo.reshape %v1122 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1124 = stablehlo.reshape %v1048 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1125 = stablehlo.slice %v1124 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1126 = stablehlo.reshape %v1125 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1127 = stablehlo.reshape %v1053 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1128 = stablehlo.slice %v1127 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1129 = stablehlo.reshape %v1128 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1130 = stablehlo.reshape %v1126 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1131 = stablehlo.transpose %v1130, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1132 = stablehlo.reshape %v1131 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1133 = stablehlo.reshape %v1123 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1134 = stablehlo.reshape %v1132 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1135 = stablehlo.dot_general %v1133, %v1134, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1136 = stablehlo.reshape %v1135 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1137 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1138 = stablehlo.multiply %v1136, %v1137 : tensor<32x38809xf32>
    %v1139 = stablehlo.reshape %v1138 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1140 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1141 = stablehlo.exponential %v1139 : tensor<32x197x197xf32>
    %v1142 = stablehlo.reduce(%v1141 init: %v1140) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1143 = stablehlo.broadcast_in_dim %v1142, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1144 = stablehlo.divide %v1141, %v1143 : tensor<32x197x197xf32>
    %v1145 = stablehlo.reshape %v1144 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1146 = stablehlo.reshape %v1145 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1147 = stablehlo.reshape %v1129 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1148 = stablehlo.dot_general %v1146, %v1147, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1149 = stablehlo.reshape %v1148 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1150 = stablehlo.reshape %v1149 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1151 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1152 = stablehlo.pad %v1150, %v1151, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v1153 = stablehlo.reshape %v1152 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1154 = stablehlo.add %v1120, %v1153 : tensor<32x37824xf32>
    %v1155 = stablehlo.reshape %v1154 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1156 = stablehlo.dot_general %v1155, %b5_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v1157 = stablehlo.broadcast_in_dim %b5_bo, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1158 = stablehlo.add %v1156, %v1157 : tensor<32x197x192xf32>
    %v1159 = stablehlo.reshape %v1158 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1160 = stablehlo.broadcast_in_dim %dp10, dims = [0] : (tensor<32xf32>) -> tensor<32x37824xf32>
    %v1161 = stablehlo.multiply %v1160, %v1159 : tensor<32x37824xf32>
    %v1162 = stablehlo.add %v1010, %v1161 : tensor<32x37824xf32>
    %v1163 = stablehlo.reshape %v1162 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1164 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1165 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v1166 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v1167 = stablehlo.reduce(%v1163 init: %v1164) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1168 = stablehlo.broadcast_in_dim %v1167, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v1169 = stablehlo.divide %v1168, %v1165 : tensor<32x197x192xf32>
    %v1170 = stablehlo.subtract %v1163, %v1169 : tensor<32x197x192xf32>
    %v1171 = stablehlo.multiply %v1170, %v1170 : tensor<32x197x192xf32>
    %v1172 = stablehlo.reduce(%v1171 init: %v1164) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1173 = stablehlo.broadcast_in_dim %v1172, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v1174 = stablehlo.divide %v1173, %v1165 : tensor<32x197x192xf32>
    %v1175 = stablehlo.add %v1174, %v1166 : tensor<32x197x192xf32>
    %v1176 = stablehlo.rsqrt %v1175 : tensor<32x197x192xf32>
    %v1177 = stablehlo.multiply %v1170, %v1176 : tensor<32x197x192xf32>
    %v1178 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v1179 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v1180 = stablehlo.multiply %v1177, %v1178 : tensor<32x197x192xf32>
    %v1181 = stablehlo.add %v1180, %v1179 : tensor<32x197x192xf32>
    %v1182 = stablehlo.reshape %v1181 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1183 = stablehlo.reshape %v1182 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1184 = stablehlo.broadcast_in_dim %b5_g2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1185 = stablehlo.multiply %v1183, %v1184 : tensor<32x197x192xf32>
    %v1186 = stablehlo.reshape %v1185 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1187 = stablehlo.reshape %v1186 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1188 = stablehlo.broadcast_in_dim %b5_bt2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1189 = stablehlo.add %v1187, %v1188 : tensor<32x197x192xf32>
    %v1190 = stablehlo.reshape %v1189 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1191 = stablehlo.reshape %v1190 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1192 = stablehlo.dot_general %v1191, %b5_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x768xf32>) -> tensor<32x197x768xf32>
    %v1193 = stablehlo.broadcast_in_dim %b5_bfc1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v1194 = stablehlo.add %v1192, %v1193 : tensor<32x197x768xf32>
    %v1195 = stablehlo.reshape %v1194 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1196 = stablehlo.constant dense<0.5> : tensor<32x151296xf32>
    %v1197 = stablehlo.multiply %v1196, %v1195 : tensor<32x151296xf32>
    %v1198 = stablehlo.negate %v1195 : tensor<32x151296xf32>
    %v1199 = stablehlo.constant dense<0.7071067811865476> : tensor<32x151296xf32>
    %v1200 = stablehlo.multiply %v1198, %v1199 : tensor<32x151296xf32>
    %v1201 = chlo.erfc %v1200 : tensor<32x151296xf32> -> tensor<32x151296xf32>
    %v1202 = stablehlo.multiply %v1197, %v1201 : tensor<32x151296xf32>
    %v1203 = stablehlo.reshape %v1202 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1204 = stablehlo.dot_general %v1203, %b5_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x192xf32>) -> tensor<32x197x192xf32>
    %v1205 = stablehlo.broadcast_in_dim %b5_bfc2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1206 = stablehlo.add %v1204, %v1205 : tensor<32x197x192xf32>
    %v1207 = stablehlo.reshape %v1206 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1208 = stablehlo.broadcast_in_dim %dp11, dims = [0] : (tensor<32xf32>) -> tensor<32x37824xf32>
    %v1209 = stablehlo.multiply %v1208, %v1207 : tensor<32x37824xf32>
    %v1210 = stablehlo.add %v1162, %v1209 : tensor<32x37824xf32>
    %v1211 = stablehlo.reshape %v1210 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1212 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1213 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v1214 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v1215 = stablehlo.reduce(%v1211 init: %v1212) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1216 = stablehlo.broadcast_in_dim %v1215, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v1217 = stablehlo.divide %v1216, %v1213 : tensor<32x197x192xf32>
    %v1218 = stablehlo.subtract %v1211, %v1217 : tensor<32x197x192xf32>
    %v1219 = stablehlo.multiply %v1218, %v1218 : tensor<32x197x192xf32>
    %v1220 = stablehlo.reduce(%v1219 init: %v1212) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1221 = stablehlo.broadcast_in_dim %v1220, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v1222 = stablehlo.divide %v1221, %v1213 : tensor<32x197x192xf32>
    %v1223 = stablehlo.add %v1222, %v1214 : tensor<32x197x192xf32>
    %v1224 = stablehlo.rsqrt %v1223 : tensor<32x197x192xf32>
    %v1225 = stablehlo.multiply %v1218, %v1224 : tensor<32x197x192xf32>
    %v1226 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v1227 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v1228 = stablehlo.multiply %v1225, %v1226 : tensor<32x197x192xf32>
    %v1229 = stablehlo.add %v1228, %v1227 : tensor<32x197x192xf32>
    %v1230 = stablehlo.reshape %v1229 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1231 = stablehlo.reshape %v1230 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1232 = stablehlo.broadcast_in_dim %b6_g1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1233 = stablehlo.multiply %v1231, %v1232 : tensor<32x197x192xf32>
    %v1234 = stablehlo.reshape %v1233 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1235 = stablehlo.reshape %v1234 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1236 = stablehlo.broadcast_in_dim %b6_bt1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1237 = stablehlo.add %v1235, %v1236 : tensor<32x197x192xf32>
    %v1238 = stablehlo.reshape %v1237 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1239 = stablehlo.reshape %v1238 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1240 = stablehlo.dot_general %v1239, %b6_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v1241 = stablehlo.broadcast_in_dim %b6_bq, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1242 = stablehlo.add %v1240, %v1241 : tensor<32x197x192xf32>
    %v1243 = stablehlo.reshape %v1242 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1244 = stablehlo.reshape %v1238 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1245 = stablehlo.dot_general %v1244, %b6_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v1246 = stablehlo.broadcast_in_dim %b6_bk, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1247 = stablehlo.add %v1245, %v1246 : tensor<32x197x192xf32>
    %v1248 = stablehlo.reshape %v1247 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1249 = stablehlo.reshape %v1238 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1250 = stablehlo.dot_general %v1249, %b6_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v1251 = stablehlo.broadcast_in_dim %b6_bv, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1252 = stablehlo.add %v1250, %v1251 : tensor<32x197x192xf32>
    %v1253 = stablehlo.reshape %v1252 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1254 = stablehlo.reshape %v1243 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1255 = stablehlo.slice %v1254 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1256 = stablehlo.reshape %v1255 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1257 = stablehlo.reshape %v1248 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1258 = stablehlo.slice %v1257 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1259 = stablehlo.reshape %v1258 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1260 = stablehlo.reshape %v1253 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1261 = stablehlo.slice %v1260 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1262 = stablehlo.reshape %v1261 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1263 = stablehlo.reshape %v1259 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1264 = stablehlo.transpose %v1263, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1265 = stablehlo.reshape %v1264 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1266 = stablehlo.reshape %v1256 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1267 = stablehlo.reshape %v1265 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1268 = stablehlo.dot_general %v1266, %v1267, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1269 = stablehlo.reshape %v1268 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1270 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1271 = stablehlo.multiply %v1269, %v1270 : tensor<32x38809xf32>
    %v1272 = stablehlo.reshape %v1271 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1273 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1274 = stablehlo.exponential %v1272 : tensor<32x197x197xf32>
    %v1275 = stablehlo.reduce(%v1274 init: %v1273) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1276 = stablehlo.broadcast_in_dim %v1275, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1277 = stablehlo.divide %v1274, %v1276 : tensor<32x197x197xf32>
    %v1278 = stablehlo.reshape %v1277 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1279 = stablehlo.reshape %v1278 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1280 = stablehlo.reshape %v1262 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1281 = stablehlo.dot_general %v1279, %v1280, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1282 = stablehlo.reshape %v1281 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1283 = stablehlo.reshape %v1282 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1284 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1285 = stablehlo.pad %v1283, %v1284, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v1286 = stablehlo.reshape %v1285 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1287 = stablehlo.reshape %v1243 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1288 = stablehlo.slice %v1287 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1289 = stablehlo.reshape %v1288 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1290 = stablehlo.reshape %v1248 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1291 = stablehlo.slice %v1290 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1292 = stablehlo.reshape %v1291 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1293 = stablehlo.reshape %v1253 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1294 = stablehlo.slice %v1293 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1295 = stablehlo.reshape %v1294 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1296 = stablehlo.reshape %v1292 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1297 = stablehlo.transpose %v1296, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1298 = stablehlo.reshape %v1297 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1299 = stablehlo.reshape %v1289 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1300 = stablehlo.reshape %v1298 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1301 = stablehlo.dot_general %v1299, %v1300, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1302 = stablehlo.reshape %v1301 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1303 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1304 = stablehlo.multiply %v1302, %v1303 : tensor<32x38809xf32>
    %v1305 = stablehlo.reshape %v1304 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1306 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1307 = stablehlo.exponential %v1305 : tensor<32x197x197xf32>
    %v1308 = stablehlo.reduce(%v1307 init: %v1306) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1309 = stablehlo.broadcast_in_dim %v1308, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1310 = stablehlo.divide %v1307, %v1309 : tensor<32x197x197xf32>
    %v1311 = stablehlo.reshape %v1310 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1312 = stablehlo.reshape %v1311 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1313 = stablehlo.reshape %v1295 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1314 = stablehlo.dot_general %v1312, %v1313, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1315 = stablehlo.reshape %v1314 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1316 = stablehlo.reshape %v1315 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1317 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1318 = stablehlo.pad %v1316, %v1317, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v1319 = stablehlo.reshape %v1318 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1320 = stablehlo.add %v1286, %v1319 : tensor<32x37824xf32>
    %v1321 = stablehlo.reshape %v1243 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1322 = stablehlo.slice %v1321 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1323 = stablehlo.reshape %v1322 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1324 = stablehlo.reshape %v1248 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1325 = stablehlo.slice %v1324 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1326 = stablehlo.reshape %v1325 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1327 = stablehlo.reshape %v1253 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1328 = stablehlo.slice %v1327 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1329 = stablehlo.reshape %v1328 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1330 = stablehlo.reshape %v1326 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1331 = stablehlo.transpose %v1330, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1332 = stablehlo.reshape %v1331 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1333 = stablehlo.reshape %v1323 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1334 = stablehlo.reshape %v1332 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1335 = stablehlo.dot_general %v1333, %v1334, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1336 = stablehlo.reshape %v1335 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1337 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1338 = stablehlo.multiply %v1336, %v1337 : tensor<32x38809xf32>
    %v1339 = stablehlo.reshape %v1338 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1340 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1341 = stablehlo.exponential %v1339 : tensor<32x197x197xf32>
    %v1342 = stablehlo.reduce(%v1341 init: %v1340) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1343 = stablehlo.broadcast_in_dim %v1342, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1344 = stablehlo.divide %v1341, %v1343 : tensor<32x197x197xf32>
    %v1345 = stablehlo.reshape %v1344 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1346 = stablehlo.reshape %v1345 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1347 = stablehlo.reshape %v1329 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1348 = stablehlo.dot_general %v1346, %v1347, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1349 = stablehlo.reshape %v1348 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1350 = stablehlo.reshape %v1349 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1351 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1352 = stablehlo.pad %v1350, %v1351, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v1353 = stablehlo.reshape %v1352 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1354 = stablehlo.add %v1320, %v1353 : tensor<32x37824xf32>
    %v1355 = stablehlo.reshape %v1354 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1356 = stablehlo.dot_general %v1355, %b6_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v1357 = stablehlo.broadcast_in_dim %b6_bo, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1358 = stablehlo.add %v1356, %v1357 : tensor<32x197x192xf32>
    %v1359 = stablehlo.reshape %v1358 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1360 = stablehlo.broadcast_in_dim %dp12, dims = [0] : (tensor<32xf32>) -> tensor<32x37824xf32>
    %v1361 = stablehlo.multiply %v1360, %v1359 : tensor<32x37824xf32>
    %v1362 = stablehlo.add %v1210, %v1361 : tensor<32x37824xf32>
    %v1363 = stablehlo.reshape %v1362 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1364 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1365 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v1366 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v1367 = stablehlo.reduce(%v1363 init: %v1364) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1368 = stablehlo.broadcast_in_dim %v1367, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v1369 = stablehlo.divide %v1368, %v1365 : tensor<32x197x192xf32>
    %v1370 = stablehlo.subtract %v1363, %v1369 : tensor<32x197x192xf32>
    %v1371 = stablehlo.multiply %v1370, %v1370 : tensor<32x197x192xf32>
    %v1372 = stablehlo.reduce(%v1371 init: %v1364) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1373 = stablehlo.broadcast_in_dim %v1372, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v1374 = stablehlo.divide %v1373, %v1365 : tensor<32x197x192xf32>
    %v1375 = stablehlo.add %v1374, %v1366 : tensor<32x197x192xf32>
    %v1376 = stablehlo.rsqrt %v1375 : tensor<32x197x192xf32>
    %v1377 = stablehlo.multiply %v1370, %v1376 : tensor<32x197x192xf32>
    %v1378 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v1379 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v1380 = stablehlo.multiply %v1377, %v1378 : tensor<32x197x192xf32>
    %v1381 = stablehlo.add %v1380, %v1379 : tensor<32x197x192xf32>
    %v1382 = stablehlo.reshape %v1381 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1383 = stablehlo.reshape %v1382 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1384 = stablehlo.broadcast_in_dim %b6_g2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1385 = stablehlo.multiply %v1383, %v1384 : tensor<32x197x192xf32>
    %v1386 = stablehlo.reshape %v1385 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1387 = stablehlo.reshape %v1386 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1388 = stablehlo.broadcast_in_dim %b6_bt2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1389 = stablehlo.add %v1387, %v1388 : tensor<32x197x192xf32>
    %v1390 = stablehlo.reshape %v1389 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1391 = stablehlo.reshape %v1390 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1392 = stablehlo.dot_general %v1391, %b6_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x768xf32>) -> tensor<32x197x768xf32>
    %v1393 = stablehlo.broadcast_in_dim %b6_bfc1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v1394 = stablehlo.add %v1392, %v1393 : tensor<32x197x768xf32>
    %v1395 = stablehlo.reshape %v1394 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1396 = stablehlo.constant dense<0.5> : tensor<32x151296xf32>
    %v1397 = stablehlo.multiply %v1396, %v1395 : tensor<32x151296xf32>
    %v1398 = stablehlo.negate %v1395 : tensor<32x151296xf32>
    %v1399 = stablehlo.constant dense<0.7071067811865476> : tensor<32x151296xf32>
    %v1400 = stablehlo.multiply %v1398, %v1399 : tensor<32x151296xf32>
    %v1401 = chlo.erfc %v1400 : tensor<32x151296xf32> -> tensor<32x151296xf32>
    %v1402 = stablehlo.multiply %v1397, %v1401 : tensor<32x151296xf32>
    %v1403 = stablehlo.reshape %v1402 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1404 = stablehlo.dot_general %v1403, %b6_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x192xf32>) -> tensor<32x197x192xf32>
    %v1405 = stablehlo.broadcast_in_dim %b6_bfc2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1406 = stablehlo.add %v1404, %v1405 : tensor<32x197x192xf32>
    %v1407 = stablehlo.reshape %v1406 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1408 = stablehlo.broadcast_in_dim %dp13, dims = [0] : (tensor<32xf32>) -> tensor<32x37824xf32>
    %v1409 = stablehlo.multiply %v1408, %v1407 : tensor<32x37824xf32>
    %v1410 = stablehlo.add %v1362, %v1409 : tensor<32x37824xf32>
    %v1411 = stablehlo.reshape %v1410 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1412 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1413 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v1414 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v1415 = stablehlo.reduce(%v1411 init: %v1412) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1416 = stablehlo.broadcast_in_dim %v1415, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v1417 = stablehlo.divide %v1416, %v1413 : tensor<32x197x192xf32>
    %v1418 = stablehlo.subtract %v1411, %v1417 : tensor<32x197x192xf32>
    %v1419 = stablehlo.multiply %v1418, %v1418 : tensor<32x197x192xf32>
    %v1420 = stablehlo.reduce(%v1419 init: %v1412) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1421 = stablehlo.broadcast_in_dim %v1420, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v1422 = stablehlo.divide %v1421, %v1413 : tensor<32x197x192xf32>
    %v1423 = stablehlo.add %v1422, %v1414 : tensor<32x197x192xf32>
    %v1424 = stablehlo.rsqrt %v1423 : tensor<32x197x192xf32>
    %v1425 = stablehlo.multiply %v1418, %v1424 : tensor<32x197x192xf32>
    %v1426 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v1427 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v1428 = stablehlo.multiply %v1425, %v1426 : tensor<32x197x192xf32>
    %v1429 = stablehlo.add %v1428, %v1427 : tensor<32x197x192xf32>
    %v1430 = stablehlo.reshape %v1429 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1431 = stablehlo.reshape %v1430 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1432 = stablehlo.broadcast_in_dim %b7_g1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1433 = stablehlo.multiply %v1431, %v1432 : tensor<32x197x192xf32>
    %v1434 = stablehlo.reshape %v1433 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1435 = stablehlo.reshape %v1434 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1436 = stablehlo.broadcast_in_dim %b7_bt1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1437 = stablehlo.add %v1435, %v1436 : tensor<32x197x192xf32>
    %v1438 = stablehlo.reshape %v1437 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1439 = stablehlo.reshape %v1438 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1440 = stablehlo.dot_general %v1439, %b7_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v1441 = stablehlo.broadcast_in_dim %b7_bq, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1442 = stablehlo.add %v1440, %v1441 : tensor<32x197x192xf32>
    %v1443 = stablehlo.reshape %v1442 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1444 = stablehlo.reshape %v1438 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1445 = stablehlo.dot_general %v1444, %b7_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v1446 = stablehlo.broadcast_in_dim %b7_bk, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1447 = stablehlo.add %v1445, %v1446 : tensor<32x197x192xf32>
    %v1448 = stablehlo.reshape %v1447 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1449 = stablehlo.reshape %v1438 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1450 = stablehlo.dot_general %v1449, %b7_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v1451 = stablehlo.broadcast_in_dim %b7_bv, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1452 = stablehlo.add %v1450, %v1451 : tensor<32x197x192xf32>
    %v1453 = stablehlo.reshape %v1452 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1454 = stablehlo.reshape %v1443 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1455 = stablehlo.slice %v1454 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1456 = stablehlo.reshape %v1455 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1457 = stablehlo.reshape %v1448 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1458 = stablehlo.slice %v1457 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1459 = stablehlo.reshape %v1458 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1460 = stablehlo.reshape %v1453 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1461 = stablehlo.slice %v1460 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1462 = stablehlo.reshape %v1461 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1463 = stablehlo.reshape %v1459 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1464 = stablehlo.transpose %v1463, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1465 = stablehlo.reshape %v1464 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1466 = stablehlo.reshape %v1456 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1467 = stablehlo.reshape %v1465 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1468 = stablehlo.dot_general %v1466, %v1467, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1469 = stablehlo.reshape %v1468 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1470 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1471 = stablehlo.multiply %v1469, %v1470 : tensor<32x38809xf32>
    %v1472 = stablehlo.reshape %v1471 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1473 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1474 = stablehlo.exponential %v1472 : tensor<32x197x197xf32>
    %v1475 = stablehlo.reduce(%v1474 init: %v1473) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1476 = stablehlo.broadcast_in_dim %v1475, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1477 = stablehlo.divide %v1474, %v1476 : tensor<32x197x197xf32>
    %v1478 = stablehlo.reshape %v1477 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1479 = stablehlo.reshape %v1478 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1480 = stablehlo.reshape %v1462 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1481 = stablehlo.dot_general %v1479, %v1480, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1482 = stablehlo.reshape %v1481 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1483 = stablehlo.reshape %v1482 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1484 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1485 = stablehlo.pad %v1483, %v1484, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v1486 = stablehlo.reshape %v1485 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1487 = stablehlo.reshape %v1443 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1488 = stablehlo.slice %v1487 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1489 = stablehlo.reshape %v1488 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1490 = stablehlo.reshape %v1448 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1491 = stablehlo.slice %v1490 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1492 = stablehlo.reshape %v1491 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1493 = stablehlo.reshape %v1453 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1494 = stablehlo.slice %v1493 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1495 = stablehlo.reshape %v1494 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1496 = stablehlo.reshape %v1492 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1497 = stablehlo.transpose %v1496, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1498 = stablehlo.reshape %v1497 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1499 = stablehlo.reshape %v1489 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1500 = stablehlo.reshape %v1498 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1501 = stablehlo.dot_general %v1499, %v1500, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1502 = stablehlo.reshape %v1501 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1503 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1504 = stablehlo.multiply %v1502, %v1503 : tensor<32x38809xf32>
    %v1505 = stablehlo.reshape %v1504 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1506 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1507 = stablehlo.exponential %v1505 : tensor<32x197x197xf32>
    %v1508 = stablehlo.reduce(%v1507 init: %v1506) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1509 = stablehlo.broadcast_in_dim %v1508, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1510 = stablehlo.divide %v1507, %v1509 : tensor<32x197x197xf32>
    %v1511 = stablehlo.reshape %v1510 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1512 = stablehlo.reshape %v1511 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1513 = stablehlo.reshape %v1495 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1514 = stablehlo.dot_general %v1512, %v1513, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1515 = stablehlo.reshape %v1514 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1516 = stablehlo.reshape %v1515 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1517 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1518 = stablehlo.pad %v1516, %v1517, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v1519 = stablehlo.reshape %v1518 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1520 = stablehlo.add %v1486, %v1519 : tensor<32x37824xf32>
    %v1521 = stablehlo.reshape %v1443 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1522 = stablehlo.slice %v1521 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1523 = stablehlo.reshape %v1522 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1524 = stablehlo.reshape %v1448 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1525 = stablehlo.slice %v1524 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1526 = stablehlo.reshape %v1525 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1527 = stablehlo.reshape %v1453 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1528 = stablehlo.slice %v1527 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1529 = stablehlo.reshape %v1528 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1530 = stablehlo.reshape %v1526 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1531 = stablehlo.transpose %v1530, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1532 = stablehlo.reshape %v1531 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1533 = stablehlo.reshape %v1523 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1534 = stablehlo.reshape %v1532 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1535 = stablehlo.dot_general %v1533, %v1534, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1536 = stablehlo.reshape %v1535 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1537 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1538 = stablehlo.multiply %v1536, %v1537 : tensor<32x38809xf32>
    %v1539 = stablehlo.reshape %v1538 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1540 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1541 = stablehlo.exponential %v1539 : tensor<32x197x197xf32>
    %v1542 = stablehlo.reduce(%v1541 init: %v1540) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1543 = stablehlo.broadcast_in_dim %v1542, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1544 = stablehlo.divide %v1541, %v1543 : tensor<32x197x197xf32>
    %v1545 = stablehlo.reshape %v1544 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1546 = stablehlo.reshape %v1545 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1547 = stablehlo.reshape %v1529 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1548 = stablehlo.dot_general %v1546, %v1547, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1549 = stablehlo.reshape %v1548 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1550 = stablehlo.reshape %v1549 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1551 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1552 = stablehlo.pad %v1550, %v1551, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v1553 = stablehlo.reshape %v1552 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1554 = stablehlo.add %v1520, %v1553 : tensor<32x37824xf32>
    %v1555 = stablehlo.reshape %v1554 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1556 = stablehlo.dot_general %v1555, %b7_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v1557 = stablehlo.broadcast_in_dim %b7_bo, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1558 = stablehlo.add %v1556, %v1557 : tensor<32x197x192xf32>
    %v1559 = stablehlo.reshape %v1558 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1560 = stablehlo.broadcast_in_dim %dp14, dims = [0] : (tensor<32xf32>) -> tensor<32x37824xf32>
    %v1561 = stablehlo.multiply %v1560, %v1559 : tensor<32x37824xf32>
    %v1562 = stablehlo.add %v1410, %v1561 : tensor<32x37824xf32>
    %v1563 = stablehlo.reshape %v1562 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1564 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1565 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v1566 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v1567 = stablehlo.reduce(%v1563 init: %v1564) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1568 = stablehlo.broadcast_in_dim %v1567, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v1569 = stablehlo.divide %v1568, %v1565 : tensor<32x197x192xf32>
    %v1570 = stablehlo.subtract %v1563, %v1569 : tensor<32x197x192xf32>
    %v1571 = stablehlo.multiply %v1570, %v1570 : tensor<32x197x192xf32>
    %v1572 = stablehlo.reduce(%v1571 init: %v1564) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1573 = stablehlo.broadcast_in_dim %v1572, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v1574 = stablehlo.divide %v1573, %v1565 : tensor<32x197x192xf32>
    %v1575 = stablehlo.add %v1574, %v1566 : tensor<32x197x192xf32>
    %v1576 = stablehlo.rsqrt %v1575 : tensor<32x197x192xf32>
    %v1577 = stablehlo.multiply %v1570, %v1576 : tensor<32x197x192xf32>
    %v1578 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v1579 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v1580 = stablehlo.multiply %v1577, %v1578 : tensor<32x197x192xf32>
    %v1581 = stablehlo.add %v1580, %v1579 : tensor<32x197x192xf32>
    %v1582 = stablehlo.reshape %v1581 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1583 = stablehlo.reshape %v1582 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1584 = stablehlo.broadcast_in_dim %b7_g2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1585 = stablehlo.multiply %v1583, %v1584 : tensor<32x197x192xf32>
    %v1586 = stablehlo.reshape %v1585 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1587 = stablehlo.reshape %v1586 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1588 = stablehlo.broadcast_in_dim %b7_bt2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1589 = stablehlo.add %v1587, %v1588 : tensor<32x197x192xf32>
    %v1590 = stablehlo.reshape %v1589 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1591 = stablehlo.reshape %v1590 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1592 = stablehlo.dot_general %v1591, %b7_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x768xf32>) -> tensor<32x197x768xf32>
    %v1593 = stablehlo.broadcast_in_dim %b7_bfc1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v1594 = stablehlo.add %v1592, %v1593 : tensor<32x197x768xf32>
    %v1595 = stablehlo.reshape %v1594 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1596 = stablehlo.constant dense<0.5> : tensor<32x151296xf32>
    %v1597 = stablehlo.multiply %v1596, %v1595 : tensor<32x151296xf32>
    %v1598 = stablehlo.negate %v1595 : tensor<32x151296xf32>
    %v1599 = stablehlo.constant dense<0.7071067811865476> : tensor<32x151296xf32>
    %v1600 = stablehlo.multiply %v1598, %v1599 : tensor<32x151296xf32>
    %v1601 = chlo.erfc %v1600 : tensor<32x151296xf32> -> tensor<32x151296xf32>
    %v1602 = stablehlo.multiply %v1597, %v1601 : tensor<32x151296xf32>
    %v1603 = stablehlo.reshape %v1602 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1604 = stablehlo.dot_general %v1603, %b7_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x192xf32>) -> tensor<32x197x192xf32>
    %v1605 = stablehlo.broadcast_in_dim %b7_bfc2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1606 = stablehlo.add %v1604, %v1605 : tensor<32x197x192xf32>
    %v1607 = stablehlo.reshape %v1606 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1608 = stablehlo.broadcast_in_dim %dp15, dims = [0] : (tensor<32xf32>) -> tensor<32x37824xf32>
    %v1609 = stablehlo.multiply %v1608, %v1607 : tensor<32x37824xf32>
    %v1610 = stablehlo.add %v1562, %v1609 : tensor<32x37824xf32>
    %v1611 = stablehlo.reshape %v1610 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1612 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1613 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v1614 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v1615 = stablehlo.reduce(%v1611 init: %v1612) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1616 = stablehlo.broadcast_in_dim %v1615, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v1617 = stablehlo.divide %v1616, %v1613 : tensor<32x197x192xf32>
    %v1618 = stablehlo.subtract %v1611, %v1617 : tensor<32x197x192xf32>
    %v1619 = stablehlo.multiply %v1618, %v1618 : tensor<32x197x192xf32>
    %v1620 = stablehlo.reduce(%v1619 init: %v1612) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1621 = stablehlo.broadcast_in_dim %v1620, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v1622 = stablehlo.divide %v1621, %v1613 : tensor<32x197x192xf32>
    %v1623 = stablehlo.add %v1622, %v1614 : tensor<32x197x192xf32>
    %v1624 = stablehlo.rsqrt %v1623 : tensor<32x197x192xf32>
    %v1625 = stablehlo.multiply %v1618, %v1624 : tensor<32x197x192xf32>
    %v1626 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v1627 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v1628 = stablehlo.multiply %v1625, %v1626 : tensor<32x197x192xf32>
    %v1629 = stablehlo.add %v1628, %v1627 : tensor<32x197x192xf32>
    %v1630 = stablehlo.reshape %v1629 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1631 = stablehlo.reshape %v1630 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1632 = stablehlo.broadcast_in_dim %b8_g1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1633 = stablehlo.multiply %v1631, %v1632 : tensor<32x197x192xf32>
    %v1634 = stablehlo.reshape %v1633 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1635 = stablehlo.reshape %v1634 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1636 = stablehlo.broadcast_in_dim %b8_bt1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1637 = stablehlo.add %v1635, %v1636 : tensor<32x197x192xf32>
    %v1638 = stablehlo.reshape %v1637 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1639 = stablehlo.reshape %v1638 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1640 = stablehlo.dot_general %v1639, %b8_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v1641 = stablehlo.broadcast_in_dim %b8_bq, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1642 = stablehlo.add %v1640, %v1641 : tensor<32x197x192xf32>
    %v1643 = stablehlo.reshape %v1642 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1644 = stablehlo.reshape %v1638 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1645 = stablehlo.dot_general %v1644, %b8_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v1646 = stablehlo.broadcast_in_dim %b8_bk, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1647 = stablehlo.add %v1645, %v1646 : tensor<32x197x192xf32>
    %v1648 = stablehlo.reshape %v1647 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1649 = stablehlo.reshape %v1638 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1650 = stablehlo.dot_general %v1649, %b8_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v1651 = stablehlo.broadcast_in_dim %b8_bv, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1652 = stablehlo.add %v1650, %v1651 : tensor<32x197x192xf32>
    %v1653 = stablehlo.reshape %v1652 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1654 = stablehlo.reshape %v1643 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1655 = stablehlo.slice %v1654 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1656 = stablehlo.reshape %v1655 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1657 = stablehlo.reshape %v1648 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1658 = stablehlo.slice %v1657 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1659 = stablehlo.reshape %v1658 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1660 = stablehlo.reshape %v1653 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1661 = stablehlo.slice %v1660 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1662 = stablehlo.reshape %v1661 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1663 = stablehlo.reshape %v1659 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1664 = stablehlo.transpose %v1663, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1665 = stablehlo.reshape %v1664 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1666 = stablehlo.reshape %v1656 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1667 = stablehlo.reshape %v1665 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1668 = stablehlo.dot_general %v1666, %v1667, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1669 = stablehlo.reshape %v1668 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1670 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1671 = stablehlo.multiply %v1669, %v1670 : tensor<32x38809xf32>
    %v1672 = stablehlo.reshape %v1671 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1673 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1674 = stablehlo.exponential %v1672 : tensor<32x197x197xf32>
    %v1675 = stablehlo.reduce(%v1674 init: %v1673) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1676 = stablehlo.broadcast_in_dim %v1675, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1677 = stablehlo.divide %v1674, %v1676 : tensor<32x197x197xf32>
    %v1678 = stablehlo.reshape %v1677 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1679 = stablehlo.reshape %v1678 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1680 = stablehlo.reshape %v1662 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1681 = stablehlo.dot_general %v1679, %v1680, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1682 = stablehlo.reshape %v1681 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1683 = stablehlo.reshape %v1682 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1684 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1685 = stablehlo.pad %v1683, %v1684, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v1686 = stablehlo.reshape %v1685 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1687 = stablehlo.reshape %v1643 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1688 = stablehlo.slice %v1687 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1689 = stablehlo.reshape %v1688 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1690 = stablehlo.reshape %v1648 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1691 = stablehlo.slice %v1690 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1692 = stablehlo.reshape %v1691 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1693 = stablehlo.reshape %v1653 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1694 = stablehlo.slice %v1693 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1695 = stablehlo.reshape %v1694 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1696 = stablehlo.reshape %v1692 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1697 = stablehlo.transpose %v1696, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1698 = stablehlo.reshape %v1697 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1699 = stablehlo.reshape %v1689 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1700 = stablehlo.reshape %v1698 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1701 = stablehlo.dot_general %v1699, %v1700, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1702 = stablehlo.reshape %v1701 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1703 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1704 = stablehlo.multiply %v1702, %v1703 : tensor<32x38809xf32>
    %v1705 = stablehlo.reshape %v1704 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1706 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1707 = stablehlo.exponential %v1705 : tensor<32x197x197xf32>
    %v1708 = stablehlo.reduce(%v1707 init: %v1706) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1709 = stablehlo.broadcast_in_dim %v1708, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1710 = stablehlo.divide %v1707, %v1709 : tensor<32x197x197xf32>
    %v1711 = stablehlo.reshape %v1710 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1712 = stablehlo.reshape %v1711 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1713 = stablehlo.reshape %v1695 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1714 = stablehlo.dot_general %v1712, %v1713, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1715 = stablehlo.reshape %v1714 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1716 = stablehlo.reshape %v1715 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1717 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1718 = stablehlo.pad %v1716, %v1717, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v1719 = stablehlo.reshape %v1718 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1720 = stablehlo.add %v1686, %v1719 : tensor<32x37824xf32>
    %v1721 = stablehlo.reshape %v1643 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1722 = stablehlo.slice %v1721 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1723 = stablehlo.reshape %v1722 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1724 = stablehlo.reshape %v1648 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1725 = stablehlo.slice %v1724 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1726 = stablehlo.reshape %v1725 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1727 = stablehlo.reshape %v1653 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1728 = stablehlo.slice %v1727 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1729 = stablehlo.reshape %v1728 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1730 = stablehlo.reshape %v1726 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1731 = stablehlo.transpose %v1730, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1732 = stablehlo.reshape %v1731 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1733 = stablehlo.reshape %v1723 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1734 = stablehlo.reshape %v1732 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1735 = stablehlo.dot_general %v1733, %v1734, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1736 = stablehlo.reshape %v1735 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1737 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1738 = stablehlo.multiply %v1736, %v1737 : tensor<32x38809xf32>
    %v1739 = stablehlo.reshape %v1738 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1740 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1741 = stablehlo.exponential %v1739 : tensor<32x197x197xf32>
    %v1742 = stablehlo.reduce(%v1741 init: %v1740) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1743 = stablehlo.broadcast_in_dim %v1742, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1744 = stablehlo.divide %v1741, %v1743 : tensor<32x197x197xf32>
    %v1745 = stablehlo.reshape %v1744 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1746 = stablehlo.reshape %v1745 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1747 = stablehlo.reshape %v1729 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1748 = stablehlo.dot_general %v1746, %v1747, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1749 = stablehlo.reshape %v1748 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1750 = stablehlo.reshape %v1749 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1751 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1752 = stablehlo.pad %v1750, %v1751, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v1753 = stablehlo.reshape %v1752 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1754 = stablehlo.add %v1720, %v1753 : tensor<32x37824xf32>
    %v1755 = stablehlo.reshape %v1754 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1756 = stablehlo.dot_general %v1755, %b8_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v1757 = stablehlo.broadcast_in_dim %b8_bo, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1758 = stablehlo.add %v1756, %v1757 : tensor<32x197x192xf32>
    %v1759 = stablehlo.reshape %v1758 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1760 = stablehlo.broadcast_in_dim %dp16, dims = [0] : (tensor<32xf32>) -> tensor<32x37824xf32>
    %v1761 = stablehlo.multiply %v1760, %v1759 : tensor<32x37824xf32>
    %v1762 = stablehlo.add %v1610, %v1761 : tensor<32x37824xf32>
    %v1763 = stablehlo.reshape %v1762 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1764 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1765 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v1766 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v1767 = stablehlo.reduce(%v1763 init: %v1764) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1768 = stablehlo.broadcast_in_dim %v1767, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v1769 = stablehlo.divide %v1768, %v1765 : tensor<32x197x192xf32>
    %v1770 = stablehlo.subtract %v1763, %v1769 : tensor<32x197x192xf32>
    %v1771 = stablehlo.multiply %v1770, %v1770 : tensor<32x197x192xf32>
    %v1772 = stablehlo.reduce(%v1771 init: %v1764) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1773 = stablehlo.broadcast_in_dim %v1772, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v1774 = stablehlo.divide %v1773, %v1765 : tensor<32x197x192xf32>
    %v1775 = stablehlo.add %v1774, %v1766 : tensor<32x197x192xf32>
    %v1776 = stablehlo.rsqrt %v1775 : tensor<32x197x192xf32>
    %v1777 = stablehlo.multiply %v1770, %v1776 : tensor<32x197x192xf32>
    %v1778 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v1779 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v1780 = stablehlo.multiply %v1777, %v1778 : tensor<32x197x192xf32>
    %v1781 = stablehlo.add %v1780, %v1779 : tensor<32x197x192xf32>
    %v1782 = stablehlo.reshape %v1781 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1783 = stablehlo.reshape %v1782 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1784 = stablehlo.broadcast_in_dim %b8_g2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1785 = stablehlo.multiply %v1783, %v1784 : tensor<32x197x192xf32>
    %v1786 = stablehlo.reshape %v1785 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1787 = stablehlo.reshape %v1786 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1788 = stablehlo.broadcast_in_dim %b8_bt2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1789 = stablehlo.add %v1787, %v1788 : tensor<32x197x192xf32>
    %v1790 = stablehlo.reshape %v1789 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1791 = stablehlo.reshape %v1790 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1792 = stablehlo.dot_general %v1791, %b8_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x768xf32>) -> tensor<32x197x768xf32>
    %v1793 = stablehlo.broadcast_in_dim %b8_bfc1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v1794 = stablehlo.add %v1792, %v1793 : tensor<32x197x768xf32>
    %v1795 = stablehlo.reshape %v1794 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1796 = stablehlo.constant dense<0.5> : tensor<32x151296xf32>
    %v1797 = stablehlo.multiply %v1796, %v1795 : tensor<32x151296xf32>
    %v1798 = stablehlo.negate %v1795 : tensor<32x151296xf32>
    %v1799 = stablehlo.constant dense<0.7071067811865476> : tensor<32x151296xf32>
    %v1800 = stablehlo.multiply %v1798, %v1799 : tensor<32x151296xf32>
    %v1801 = chlo.erfc %v1800 : tensor<32x151296xf32> -> tensor<32x151296xf32>
    %v1802 = stablehlo.multiply %v1797, %v1801 : tensor<32x151296xf32>
    %v1803 = stablehlo.reshape %v1802 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v1804 = stablehlo.dot_general %v1803, %b8_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x192xf32>) -> tensor<32x197x192xf32>
    %v1805 = stablehlo.broadcast_in_dim %b8_bfc2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1806 = stablehlo.add %v1804, %v1805 : tensor<32x197x192xf32>
    %v1807 = stablehlo.reshape %v1806 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1808 = stablehlo.broadcast_in_dim %dp17, dims = [0] : (tensor<32xf32>) -> tensor<32x37824xf32>
    %v1809 = stablehlo.multiply %v1808, %v1807 : tensor<32x37824xf32>
    %v1810 = stablehlo.add %v1762, %v1809 : tensor<32x37824xf32>
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
    %v1832 = stablehlo.broadcast_in_dim %b9_g1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1833 = stablehlo.multiply %v1831, %v1832 : tensor<32x197x192xf32>
    %v1834 = stablehlo.reshape %v1833 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1835 = stablehlo.reshape %v1834 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1836 = stablehlo.broadcast_in_dim %b9_bt1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1837 = stablehlo.add %v1835, %v1836 : tensor<32x197x192xf32>
    %v1838 = stablehlo.reshape %v1837 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1839 = stablehlo.reshape %v1838 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1840 = stablehlo.dot_general %v1839, %b9_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v1841 = stablehlo.broadcast_in_dim %b9_bq, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1842 = stablehlo.add %v1840, %v1841 : tensor<32x197x192xf32>
    %v1843 = stablehlo.reshape %v1842 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1844 = stablehlo.reshape %v1838 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1845 = stablehlo.dot_general %v1844, %b9_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v1846 = stablehlo.broadcast_in_dim %b9_bk, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1847 = stablehlo.add %v1845, %v1846 : tensor<32x197x192xf32>
    %v1848 = stablehlo.reshape %v1847 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1849 = stablehlo.reshape %v1838 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1850 = stablehlo.dot_general %v1849, %b9_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v1851 = stablehlo.broadcast_in_dim %b9_bv, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1852 = stablehlo.add %v1850, %v1851 : tensor<32x197x192xf32>
    %v1853 = stablehlo.reshape %v1852 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1854 = stablehlo.reshape %v1843 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1855 = stablehlo.slice %v1854 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1856 = stablehlo.reshape %v1855 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1857 = stablehlo.reshape %v1848 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1858 = stablehlo.slice %v1857 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1859 = stablehlo.reshape %v1858 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1860 = stablehlo.reshape %v1853 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1861 = stablehlo.slice %v1860 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1862 = stablehlo.reshape %v1861 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1863 = stablehlo.reshape %v1859 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1864 = stablehlo.transpose %v1863, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1865 = stablehlo.reshape %v1864 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1866 = stablehlo.reshape %v1856 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1867 = stablehlo.reshape %v1865 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1868 = stablehlo.dot_general %v1866, %v1867, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1869 = stablehlo.reshape %v1868 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1870 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1871 = stablehlo.multiply %v1869, %v1870 : tensor<32x38809xf32>
    %v1872 = stablehlo.reshape %v1871 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1873 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1874 = stablehlo.exponential %v1872 : tensor<32x197x197xf32>
    %v1875 = stablehlo.reduce(%v1874 init: %v1873) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1876 = stablehlo.broadcast_in_dim %v1875, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1877 = stablehlo.divide %v1874, %v1876 : tensor<32x197x197xf32>
    %v1878 = stablehlo.reshape %v1877 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1879 = stablehlo.reshape %v1878 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1880 = stablehlo.reshape %v1862 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1881 = stablehlo.dot_general %v1879, %v1880, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1882 = stablehlo.reshape %v1881 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1883 = stablehlo.reshape %v1882 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1884 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1885 = stablehlo.pad %v1883, %v1884, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v1886 = stablehlo.reshape %v1885 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1887 = stablehlo.reshape %v1843 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1888 = stablehlo.slice %v1887 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1889 = stablehlo.reshape %v1888 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1890 = stablehlo.reshape %v1848 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1891 = stablehlo.slice %v1890 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1892 = stablehlo.reshape %v1891 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1893 = stablehlo.reshape %v1853 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1894 = stablehlo.slice %v1893 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1895 = stablehlo.reshape %v1894 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1896 = stablehlo.reshape %v1892 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1897 = stablehlo.transpose %v1896, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1898 = stablehlo.reshape %v1897 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1899 = stablehlo.reshape %v1889 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1900 = stablehlo.reshape %v1898 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1901 = stablehlo.dot_general %v1899, %v1900, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1902 = stablehlo.reshape %v1901 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1903 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1904 = stablehlo.multiply %v1902, %v1903 : tensor<32x38809xf32>
    %v1905 = stablehlo.reshape %v1904 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1906 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1907 = stablehlo.exponential %v1905 : tensor<32x197x197xf32>
    %v1908 = stablehlo.reduce(%v1907 init: %v1906) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1909 = stablehlo.broadcast_in_dim %v1908, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1910 = stablehlo.divide %v1907, %v1909 : tensor<32x197x197xf32>
    %v1911 = stablehlo.reshape %v1910 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1912 = stablehlo.reshape %v1911 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1913 = stablehlo.reshape %v1895 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1914 = stablehlo.dot_general %v1912, %v1913, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1915 = stablehlo.reshape %v1914 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1916 = stablehlo.reshape %v1915 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1917 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1918 = stablehlo.pad %v1916, %v1917, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v1919 = stablehlo.reshape %v1918 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1920 = stablehlo.add %v1886, %v1919 : tensor<32x37824xf32>
    %v1921 = stablehlo.reshape %v1843 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1922 = stablehlo.slice %v1921 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1923 = stablehlo.reshape %v1922 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1924 = stablehlo.reshape %v1848 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1925 = stablehlo.slice %v1924 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1926 = stablehlo.reshape %v1925 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1927 = stablehlo.reshape %v1853 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1928 = stablehlo.slice %v1927 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v1929 = stablehlo.reshape %v1928 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1930 = stablehlo.reshape %v1926 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1931 = stablehlo.transpose %v1930, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v1932 = stablehlo.reshape %v1931 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v1933 = stablehlo.reshape %v1923 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1934 = stablehlo.reshape %v1932 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v1935 = stablehlo.dot_general %v1933, %v1934, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v1936 = stablehlo.reshape %v1935 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1937 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v1938 = stablehlo.multiply %v1936, %v1937 : tensor<32x38809xf32>
    %v1939 = stablehlo.reshape %v1938 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1940 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1941 = stablehlo.exponential %v1939 : tensor<32x197x197xf32>
    %v1942 = stablehlo.reduce(%v1941 init: %v1940) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1943 = stablehlo.broadcast_in_dim %v1942, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v1944 = stablehlo.divide %v1941, %v1943 : tensor<32x197x197xf32>
    %v1945 = stablehlo.reshape %v1944 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v1946 = stablehlo.reshape %v1945 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v1947 = stablehlo.reshape %v1929 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1948 = stablehlo.dot_general %v1946, %v1947, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v1949 = stablehlo.reshape %v1948 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v1950 = stablehlo.reshape %v1949 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v1951 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1952 = stablehlo.pad %v1950, %v1951, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v1953 = stablehlo.reshape %v1952 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1954 = stablehlo.add %v1920, %v1953 : tensor<32x37824xf32>
    %v1955 = stablehlo.reshape %v1954 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1956 = stablehlo.dot_general %v1955, %b9_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v1957 = stablehlo.broadcast_in_dim %b9_bo, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1958 = stablehlo.add %v1956, %v1957 : tensor<32x197x192xf32>
    %v1959 = stablehlo.reshape %v1958 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1960 = stablehlo.broadcast_in_dim %dp18, dims = [0] : (tensor<32xf32>) -> tensor<32x37824xf32>
    %v1961 = stablehlo.multiply %v1960, %v1959 : tensor<32x37824xf32>
    %v1962 = stablehlo.add %v1810, %v1961 : tensor<32x37824xf32>
    %v1963 = stablehlo.reshape %v1962 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1964 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1965 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v1966 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v1967 = stablehlo.reduce(%v1963 init: %v1964) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1968 = stablehlo.broadcast_in_dim %v1967, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v1969 = stablehlo.divide %v1968, %v1965 : tensor<32x197x192xf32>
    %v1970 = stablehlo.subtract %v1963, %v1969 : tensor<32x197x192xf32>
    %v1971 = stablehlo.multiply %v1970, %v1970 : tensor<32x197x192xf32>
    %v1972 = stablehlo.reduce(%v1971 init: %v1964) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v1973 = stablehlo.broadcast_in_dim %v1972, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v1974 = stablehlo.divide %v1973, %v1965 : tensor<32x197x192xf32>
    %v1975 = stablehlo.add %v1974, %v1966 : tensor<32x197x192xf32>
    %v1976 = stablehlo.rsqrt %v1975 : tensor<32x197x192xf32>
    %v1977 = stablehlo.multiply %v1970, %v1976 : tensor<32x197x192xf32>
    %v1978 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v1979 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v1980 = stablehlo.multiply %v1977, %v1978 : tensor<32x197x192xf32>
    %v1981 = stablehlo.add %v1980, %v1979 : tensor<32x197x192xf32>
    %v1982 = stablehlo.reshape %v1981 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1983 = stablehlo.reshape %v1982 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1984 = stablehlo.broadcast_in_dim %b9_g2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1985 = stablehlo.multiply %v1983, %v1984 : tensor<32x197x192xf32>
    %v1986 = stablehlo.reshape %v1985 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1987 = stablehlo.reshape %v1986 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1988 = stablehlo.broadcast_in_dim %b9_bt2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v1989 = stablehlo.add %v1987, %v1988 : tensor<32x197x192xf32>
    %v1990 = stablehlo.reshape %v1989 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v1991 = stablehlo.reshape %v1990 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v1992 = stablehlo.dot_general %v1991, %b9_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x768xf32>) -> tensor<32x197x768xf32>
    %v1993 = stablehlo.broadcast_in_dim %b9_bfc1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v1994 = stablehlo.add %v1992, %v1993 : tensor<32x197x768xf32>
    %v1995 = stablehlo.reshape %v1994 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v1996 = stablehlo.constant dense<0.5> : tensor<32x151296xf32>
    %v1997 = stablehlo.multiply %v1996, %v1995 : tensor<32x151296xf32>
    %v1998 = stablehlo.negate %v1995 : tensor<32x151296xf32>
    %v1999 = stablehlo.constant dense<0.7071067811865476> : tensor<32x151296xf32>
    %v2000 = stablehlo.multiply %v1998, %v1999 : tensor<32x151296xf32>
    %v2001 = chlo.erfc %v2000 : tensor<32x151296xf32> -> tensor<32x151296xf32>
    %v2002 = stablehlo.multiply %v1997, %v2001 : tensor<32x151296xf32>
    %v2003 = stablehlo.reshape %v2002 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2004 = stablehlo.dot_general %v2003, %b9_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x192xf32>) -> tensor<32x197x192xf32>
    %v2005 = stablehlo.broadcast_in_dim %b9_bfc2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v2006 = stablehlo.add %v2004, %v2005 : tensor<32x197x192xf32>
    %v2007 = stablehlo.reshape %v2006 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2008 = stablehlo.broadcast_in_dim %dp19, dims = [0] : (tensor<32xf32>) -> tensor<32x37824xf32>
    %v2009 = stablehlo.multiply %v2008, %v2007 : tensor<32x37824xf32>
    %v2010 = stablehlo.add %v1962, %v2009 : tensor<32x37824xf32>
    %v2011 = stablehlo.reshape %v2010 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2012 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2013 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v2014 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v2015 = stablehlo.reduce(%v2011 init: %v2012) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2016 = stablehlo.broadcast_in_dim %v2015, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v2017 = stablehlo.divide %v2016, %v2013 : tensor<32x197x192xf32>
    %v2018 = stablehlo.subtract %v2011, %v2017 : tensor<32x197x192xf32>
    %v2019 = stablehlo.multiply %v2018, %v2018 : tensor<32x197x192xf32>
    %v2020 = stablehlo.reduce(%v2019 init: %v2012) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2021 = stablehlo.broadcast_in_dim %v2020, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v2022 = stablehlo.divide %v2021, %v2013 : tensor<32x197x192xf32>
    %v2023 = stablehlo.add %v2022, %v2014 : tensor<32x197x192xf32>
    %v2024 = stablehlo.rsqrt %v2023 : tensor<32x197x192xf32>
    %v2025 = stablehlo.multiply %v2018, %v2024 : tensor<32x197x192xf32>
    %v2026 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v2027 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v2028 = stablehlo.multiply %v2025, %v2026 : tensor<32x197x192xf32>
    %v2029 = stablehlo.add %v2028, %v2027 : tensor<32x197x192xf32>
    %v2030 = stablehlo.reshape %v2029 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2031 = stablehlo.reshape %v2030 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2032 = stablehlo.broadcast_in_dim %b10_g1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v2033 = stablehlo.multiply %v2031, %v2032 : tensor<32x197x192xf32>
    %v2034 = stablehlo.reshape %v2033 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2035 = stablehlo.reshape %v2034 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2036 = stablehlo.broadcast_in_dim %b10_bt1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v2037 = stablehlo.add %v2035, %v2036 : tensor<32x197x192xf32>
    %v2038 = stablehlo.reshape %v2037 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2039 = stablehlo.reshape %v2038 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2040 = stablehlo.dot_general %v2039, %b10_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v2041 = stablehlo.broadcast_in_dim %b10_bq, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v2042 = stablehlo.add %v2040, %v2041 : tensor<32x197x192xf32>
    %v2043 = stablehlo.reshape %v2042 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2044 = stablehlo.reshape %v2038 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2045 = stablehlo.dot_general %v2044, %b10_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v2046 = stablehlo.broadcast_in_dim %b10_bk, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v2047 = stablehlo.add %v2045, %v2046 : tensor<32x197x192xf32>
    %v2048 = stablehlo.reshape %v2047 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2049 = stablehlo.reshape %v2038 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2050 = stablehlo.dot_general %v2049, %b10_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v2051 = stablehlo.broadcast_in_dim %b10_bv, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v2052 = stablehlo.add %v2050, %v2051 : tensor<32x197x192xf32>
    %v2053 = stablehlo.reshape %v2052 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2054 = stablehlo.reshape %v2043 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2055 = stablehlo.slice %v2054 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v2056 = stablehlo.reshape %v2055 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2057 = stablehlo.reshape %v2048 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2058 = stablehlo.slice %v2057 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v2059 = stablehlo.reshape %v2058 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2060 = stablehlo.reshape %v2053 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2061 = stablehlo.slice %v2060 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v2062 = stablehlo.reshape %v2061 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2063 = stablehlo.reshape %v2059 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2064 = stablehlo.transpose %v2063, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2065 = stablehlo.reshape %v2064 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2066 = stablehlo.reshape %v2056 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2067 = stablehlo.reshape %v2065 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2068 = stablehlo.dot_general %v2066, %v2067, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2069 = stablehlo.reshape %v2068 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2070 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2071 = stablehlo.multiply %v2069, %v2070 : tensor<32x38809xf32>
    %v2072 = stablehlo.reshape %v2071 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2073 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2074 = stablehlo.exponential %v2072 : tensor<32x197x197xf32>
    %v2075 = stablehlo.reduce(%v2074 init: %v2073) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2076 = stablehlo.broadcast_in_dim %v2075, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2077 = stablehlo.divide %v2074, %v2076 : tensor<32x197x197xf32>
    %v2078 = stablehlo.reshape %v2077 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2079 = stablehlo.reshape %v2078 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2080 = stablehlo.reshape %v2062 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2081 = stablehlo.dot_general %v2079, %v2080, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2082 = stablehlo.reshape %v2081 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2083 = stablehlo.reshape %v2082 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2084 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2085 = stablehlo.pad %v2083, %v2084, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v2086 = stablehlo.reshape %v2085 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2087 = stablehlo.reshape %v2043 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2088 = stablehlo.slice %v2087 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v2089 = stablehlo.reshape %v2088 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2090 = stablehlo.reshape %v2048 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2091 = stablehlo.slice %v2090 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v2092 = stablehlo.reshape %v2091 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2093 = stablehlo.reshape %v2053 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2094 = stablehlo.slice %v2093 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v2095 = stablehlo.reshape %v2094 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2096 = stablehlo.reshape %v2092 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2097 = stablehlo.transpose %v2096, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2098 = stablehlo.reshape %v2097 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2099 = stablehlo.reshape %v2089 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2100 = stablehlo.reshape %v2098 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2101 = stablehlo.dot_general %v2099, %v2100, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2102 = stablehlo.reshape %v2101 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2103 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2104 = stablehlo.multiply %v2102, %v2103 : tensor<32x38809xf32>
    %v2105 = stablehlo.reshape %v2104 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2106 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2107 = stablehlo.exponential %v2105 : tensor<32x197x197xf32>
    %v2108 = stablehlo.reduce(%v2107 init: %v2106) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2109 = stablehlo.broadcast_in_dim %v2108, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2110 = stablehlo.divide %v2107, %v2109 : tensor<32x197x197xf32>
    %v2111 = stablehlo.reshape %v2110 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2112 = stablehlo.reshape %v2111 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2113 = stablehlo.reshape %v2095 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2114 = stablehlo.dot_general %v2112, %v2113, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2115 = stablehlo.reshape %v2114 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2116 = stablehlo.reshape %v2115 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2117 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2118 = stablehlo.pad %v2116, %v2117, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v2119 = stablehlo.reshape %v2118 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2120 = stablehlo.add %v2086, %v2119 : tensor<32x37824xf32>
    %v2121 = stablehlo.reshape %v2043 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2122 = stablehlo.slice %v2121 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v2123 = stablehlo.reshape %v2122 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2124 = stablehlo.reshape %v2048 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2125 = stablehlo.slice %v2124 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v2126 = stablehlo.reshape %v2125 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2127 = stablehlo.reshape %v2053 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2128 = stablehlo.slice %v2127 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
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
    %v2140 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2141 = stablehlo.exponential %v2139 : tensor<32x197x197xf32>
    %v2142 = stablehlo.reduce(%v2141 init: %v2140) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2143 = stablehlo.broadcast_in_dim %v2142, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2144 = stablehlo.divide %v2141, %v2143 : tensor<32x197x197xf32>
    %v2145 = stablehlo.reshape %v2144 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2146 = stablehlo.reshape %v2145 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2147 = stablehlo.reshape %v2129 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2148 = stablehlo.dot_general %v2146, %v2147, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2149 = stablehlo.reshape %v2148 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2150 = stablehlo.reshape %v2149 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2151 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2152 = stablehlo.pad %v2150, %v2151, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v2153 = stablehlo.reshape %v2152 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2154 = stablehlo.add %v2120, %v2153 : tensor<32x37824xf32>
    %v2155 = stablehlo.reshape %v2154 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2156 = stablehlo.dot_general %v2155, %b10_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v2157 = stablehlo.broadcast_in_dim %b10_bo, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v2158 = stablehlo.add %v2156, %v2157 : tensor<32x197x192xf32>
    %v2159 = stablehlo.reshape %v2158 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2160 = stablehlo.broadcast_in_dim %dp20, dims = [0] : (tensor<32xf32>) -> tensor<32x37824xf32>
    %v2161 = stablehlo.multiply %v2160, %v2159 : tensor<32x37824xf32>
    %v2162 = stablehlo.add %v2010, %v2161 : tensor<32x37824xf32>
    %v2163 = stablehlo.reshape %v2162 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2164 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2165 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v2166 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v2167 = stablehlo.reduce(%v2163 init: %v2164) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2168 = stablehlo.broadcast_in_dim %v2167, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v2169 = stablehlo.divide %v2168, %v2165 : tensor<32x197x192xf32>
    %v2170 = stablehlo.subtract %v2163, %v2169 : tensor<32x197x192xf32>
    %v2171 = stablehlo.multiply %v2170, %v2170 : tensor<32x197x192xf32>
    %v2172 = stablehlo.reduce(%v2171 init: %v2164) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2173 = stablehlo.broadcast_in_dim %v2172, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v2174 = stablehlo.divide %v2173, %v2165 : tensor<32x197x192xf32>
    %v2175 = stablehlo.add %v2174, %v2166 : tensor<32x197x192xf32>
    %v2176 = stablehlo.rsqrt %v2175 : tensor<32x197x192xf32>
    %v2177 = stablehlo.multiply %v2170, %v2176 : tensor<32x197x192xf32>
    %v2178 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v2179 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v2180 = stablehlo.multiply %v2177, %v2178 : tensor<32x197x192xf32>
    %v2181 = stablehlo.add %v2180, %v2179 : tensor<32x197x192xf32>
    %v2182 = stablehlo.reshape %v2181 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2183 = stablehlo.reshape %v2182 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2184 = stablehlo.broadcast_in_dim %b10_g2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v2185 = stablehlo.multiply %v2183, %v2184 : tensor<32x197x192xf32>
    %v2186 = stablehlo.reshape %v2185 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2187 = stablehlo.reshape %v2186 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2188 = stablehlo.broadcast_in_dim %b10_bt2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v2189 = stablehlo.add %v2187, %v2188 : tensor<32x197x192xf32>
    %v2190 = stablehlo.reshape %v2189 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2191 = stablehlo.reshape %v2190 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2192 = stablehlo.dot_general %v2191, %b10_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x768xf32>) -> tensor<32x197x768xf32>
    %v2193 = stablehlo.broadcast_in_dim %b10_bfc1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v2194 = stablehlo.add %v2192, %v2193 : tensor<32x197x768xf32>
    %v2195 = stablehlo.reshape %v2194 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2196 = stablehlo.constant dense<0.5> : tensor<32x151296xf32>
    %v2197 = stablehlo.multiply %v2196, %v2195 : tensor<32x151296xf32>
    %v2198 = stablehlo.negate %v2195 : tensor<32x151296xf32>
    %v2199 = stablehlo.constant dense<0.7071067811865476> : tensor<32x151296xf32>
    %v2200 = stablehlo.multiply %v2198, %v2199 : tensor<32x151296xf32>
    %v2201 = chlo.erfc %v2200 : tensor<32x151296xf32> -> tensor<32x151296xf32>
    %v2202 = stablehlo.multiply %v2197, %v2201 : tensor<32x151296xf32>
    %v2203 = stablehlo.reshape %v2202 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2204 = stablehlo.dot_general %v2203, %b10_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x192xf32>) -> tensor<32x197x192xf32>
    %v2205 = stablehlo.broadcast_in_dim %b10_bfc2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v2206 = stablehlo.add %v2204, %v2205 : tensor<32x197x192xf32>
    %v2207 = stablehlo.reshape %v2206 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2208 = stablehlo.broadcast_in_dim %dp21, dims = [0] : (tensor<32xf32>) -> tensor<32x37824xf32>
    %v2209 = stablehlo.multiply %v2208, %v2207 : tensor<32x37824xf32>
    %v2210 = stablehlo.add %v2162, %v2209 : tensor<32x37824xf32>
    %v2211 = stablehlo.reshape %v2210 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2212 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2213 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v2214 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v2215 = stablehlo.reduce(%v2211 init: %v2212) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2216 = stablehlo.broadcast_in_dim %v2215, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v2217 = stablehlo.divide %v2216, %v2213 : tensor<32x197x192xf32>
    %v2218 = stablehlo.subtract %v2211, %v2217 : tensor<32x197x192xf32>
    %v2219 = stablehlo.multiply %v2218, %v2218 : tensor<32x197x192xf32>
    %v2220 = stablehlo.reduce(%v2219 init: %v2212) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2221 = stablehlo.broadcast_in_dim %v2220, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v2222 = stablehlo.divide %v2221, %v2213 : tensor<32x197x192xf32>
    %v2223 = stablehlo.add %v2222, %v2214 : tensor<32x197x192xf32>
    %v2224 = stablehlo.rsqrt %v2223 : tensor<32x197x192xf32>
    %v2225 = stablehlo.multiply %v2218, %v2224 : tensor<32x197x192xf32>
    %v2226 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v2227 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v2228 = stablehlo.multiply %v2225, %v2226 : tensor<32x197x192xf32>
    %v2229 = stablehlo.add %v2228, %v2227 : tensor<32x197x192xf32>
    %v2230 = stablehlo.reshape %v2229 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2231 = stablehlo.reshape %v2230 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2232 = stablehlo.broadcast_in_dim %b11_g1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v2233 = stablehlo.multiply %v2231, %v2232 : tensor<32x197x192xf32>
    %v2234 = stablehlo.reshape %v2233 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2235 = stablehlo.reshape %v2234 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2236 = stablehlo.broadcast_in_dim %b11_bt1, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v2237 = stablehlo.add %v2235, %v2236 : tensor<32x197x192xf32>
    %v2238 = stablehlo.reshape %v2237 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2239 = stablehlo.reshape %v2238 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2240 = stablehlo.dot_general %v2239, %b11_Wq, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v2241 = stablehlo.broadcast_in_dim %b11_bq, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v2242 = stablehlo.add %v2240, %v2241 : tensor<32x197x192xf32>
    %v2243 = stablehlo.reshape %v2242 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2244 = stablehlo.reshape %v2238 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2245 = stablehlo.dot_general %v2244, %b11_Wk, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v2246 = stablehlo.broadcast_in_dim %b11_bk, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v2247 = stablehlo.add %v2245, %v2246 : tensor<32x197x192xf32>
    %v2248 = stablehlo.reshape %v2247 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2249 = stablehlo.reshape %v2238 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2250 = stablehlo.dot_general %v2249, %b11_Wv, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v2251 = stablehlo.broadcast_in_dim %b11_bv, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v2252 = stablehlo.add %v2250, %v2251 : tensor<32x197x192xf32>
    %v2253 = stablehlo.reshape %v2252 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2254 = stablehlo.reshape %v2243 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2255 = stablehlo.slice %v2254 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v2256 = stablehlo.reshape %v2255 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2257 = stablehlo.reshape %v2248 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2258 = stablehlo.slice %v2257 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v2259 = stablehlo.reshape %v2258 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2260 = stablehlo.reshape %v2253 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2261 = stablehlo.slice %v2260 [0:32, 0:197, 0:64] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
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
    %v2273 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2274 = stablehlo.exponential %v2272 : tensor<32x197x197xf32>
    %v2275 = stablehlo.reduce(%v2274 init: %v2273) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2276 = stablehlo.broadcast_in_dim %v2275, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2277 = stablehlo.divide %v2274, %v2276 : tensor<32x197x197xf32>
    %v2278 = stablehlo.reshape %v2277 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2279 = stablehlo.reshape %v2278 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2280 = stablehlo.reshape %v2262 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2281 = stablehlo.dot_general %v2279, %v2280, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2282 = stablehlo.reshape %v2281 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2283 = stablehlo.reshape %v2282 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2284 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2285 = stablehlo.pad %v2283, %v2284, low = [0, 0, 0], high = [0, 0, 128], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v2286 = stablehlo.reshape %v2285 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2287 = stablehlo.reshape %v2243 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2288 = stablehlo.slice %v2287 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v2289 = stablehlo.reshape %v2288 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2290 = stablehlo.reshape %v2248 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2291 = stablehlo.slice %v2290 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v2292 = stablehlo.reshape %v2291 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2293 = stablehlo.reshape %v2253 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2294 = stablehlo.slice %v2293 [0:32, 0:197, 64:128] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v2295 = stablehlo.reshape %v2294 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2296 = stablehlo.reshape %v2292 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2297 = stablehlo.transpose %v2296, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2298 = stablehlo.reshape %v2297 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2299 = stablehlo.reshape %v2289 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2300 = stablehlo.reshape %v2298 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2301 = stablehlo.dot_general %v2299, %v2300, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2302 = stablehlo.reshape %v2301 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2303 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2304 = stablehlo.multiply %v2302, %v2303 : tensor<32x38809xf32>
    %v2305 = stablehlo.reshape %v2304 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2306 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2307 = stablehlo.exponential %v2305 : tensor<32x197x197xf32>
    %v2308 = stablehlo.reduce(%v2307 init: %v2306) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2309 = stablehlo.broadcast_in_dim %v2308, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2310 = stablehlo.divide %v2307, %v2309 : tensor<32x197x197xf32>
    %v2311 = stablehlo.reshape %v2310 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2312 = stablehlo.reshape %v2311 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2313 = stablehlo.reshape %v2295 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2314 = stablehlo.dot_general %v2312, %v2313, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2315 = stablehlo.reshape %v2314 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2316 = stablehlo.reshape %v2315 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2317 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2318 = stablehlo.pad %v2316, %v2317, low = [0, 0, 64], high = [0, 0, 64], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v2319 = stablehlo.reshape %v2318 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2320 = stablehlo.add %v2286, %v2319 : tensor<32x37824xf32>
    %v2321 = stablehlo.reshape %v2243 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2322 = stablehlo.slice %v2321 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v2323 = stablehlo.reshape %v2322 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2324 = stablehlo.reshape %v2248 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2325 = stablehlo.slice %v2324 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v2326 = stablehlo.reshape %v2325 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2327 = stablehlo.reshape %v2253 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2328 = stablehlo.slice %v2327 [0:32, 0:197, 128:192] : (tensor<32x197x192xf32>) -> tensor<32x197x64xf32>
    %v2329 = stablehlo.reshape %v2328 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2330 = stablehlo.reshape %v2326 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2331 = stablehlo.transpose %v2330, dims = [0, 2, 1] : (tensor<32x197x64xf32>) -> tensor<32x64x197xf32>
    %v2332 = stablehlo.reshape %v2331 : (tensor<32x64x197xf32>) -> tensor<32x12608xf32>
    %v2333 = stablehlo.reshape %v2323 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2334 = stablehlo.reshape %v2332 : (tensor<32x12608xf32>) -> tensor<32x64x197xf32>
    %v2335 = stablehlo.dot_general %v2333, %v2334, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x64xf32>, tensor<32x64x197xf32>) -> tensor<32x197x197xf32>
    %v2336 = stablehlo.reshape %v2335 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2337 = stablehlo.constant dense<0.125> : tensor<32x38809xf32>
    %v2338 = stablehlo.multiply %v2336, %v2337 : tensor<32x38809xf32>
    %v2339 = stablehlo.reshape %v2338 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2340 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2341 = stablehlo.exponential %v2339 : tensor<32x197x197xf32>
    %v2342 = stablehlo.reduce(%v2341 init: %v2340) applies stablehlo.add across dimensions = [2] : (tensor<32x197x197xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2343 = stablehlo.broadcast_in_dim %v2342, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x197xf32>
    %v2344 = stablehlo.divide %v2341, %v2343 : tensor<32x197x197xf32>
    %v2345 = stablehlo.reshape %v2344 : (tensor<32x197x197xf32>) -> tensor<32x38809xf32>
    %v2346 = stablehlo.reshape %v2345 : (tensor<32x38809xf32>) -> tensor<32x197x197xf32>
    %v2347 = stablehlo.reshape %v2329 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2348 = stablehlo.dot_general %v2346, %v2347, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x197x197xf32>, tensor<32x197x64xf32>) -> tensor<32x197x64xf32>
    %v2349 = stablehlo.reshape %v2348 : (tensor<32x197x64xf32>) -> tensor<32x12608xf32>
    %v2350 = stablehlo.reshape %v2349 : (tensor<32x12608xf32>) -> tensor<32x197x64xf32>
    %v2351 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2352 = stablehlo.pad %v2350, %v2351, low = [0, 0, 128], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<32x197x64xf32>, tensor<f32>) -> tensor<32x197x192xf32>
    %v2353 = stablehlo.reshape %v2352 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2354 = stablehlo.add %v2320, %v2353 : tensor<32x37824xf32>
    %v2355 = stablehlo.reshape %v2354 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2356 = stablehlo.dot_general %v2355, %b11_Wo, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x192xf32>) -> tensor<32x197x192xf32>
    %v2357 = stablehlo.broadcast_in_dim %b11_bo, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v2358 = stablehlo.add %v2356, %v2357 : tensor<32x197x192xf32>
    %v2359 = stablehlo.reshape %v2358 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2360 = stablehlo.broadcast_in_dim %dp22, dims = [0] : (tensor<32xf32>) -> tensor<32x37824xf32>
    %v2361 = stablehlo.multiply %v2360, %v2359 : tensor<32x37824xf32>
    %v2362 = stablehlo.add %v2210, %v2361 : tensor<32x37824xf32>
    %v2363 = stablehlo.reshape %v2362 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2364 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2365 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v2366 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v2367 = stablehlo.reduce(%v2363 init: %v2364) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2368 = stablehlo.broadcast_in_dim %v2367, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v2369 = stablehlo.divide %v2368, %v2365 : tensor<32x197x192xf32>
    %v2370 = stablehlo.subtract %v2363, %v2369 : tensor<32x197x192xf32>
    %v2371 = stablehlo.multiply %v2370, %v2370 : tensor<32x197x192xf32>
    %v2372 = stablehlo.reduce(%v2371 init: %v2364) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2373 = stablehlo.broadcast_in_dim %v2372, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v2374 = stablehlo.divide %v2373, %v2365 : tensor<32x197x192xf32>
    %v2375 = stablehlo.add %v2374, %v2366 : tensor<32x197x192xf32>
    %v2376 = stablehlo.rsqrt %v2375 : tensor<32x197x192xf32>
    %v2377 = stablehlo.multiply %v2370, %v2376 : tensor<32x197x192xf32>
    %v2378 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v2379 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v2380 = stablehlo.multiply %v2377, %v2378 : tensor<32x197x192xf32>
    %v2381 = stablehlo.add %v2380, %v2379 : tensor<32x197x192xf32>
    %v2382 = stablehlo.reshape %v2381 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2383 = stablehlo.reshape %v2382 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2384 = stablehlo.broadcast_in_dim %b11_g2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v2385 = stablehlo.multiply %v2383, %v2384 : tensor<32x197x192xf32>
    %v2386 = stablehlo.reshape %v2385 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2387 = stablehlo.reshape %v2386 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2388 = stablehlo.broadcast_in_dim %b11_bt2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v2389 = stablehlo.add %v2387, %v2388 : tensor<32x197x192xf32>
    %v2390 = stablehlo.reshape %v2389 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2391 = stablehlo.reshape %v2390 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2392 = stablehlo.dot_general %v2391, %b11_Wfc1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x192xf32>, tensor<192x768xf32>) -> tensor<32x197x768xf32>
    %v2393 = stablehlo.broadcast_in_dim %b11_bfc1, dims = [2] : (tensor<768xf32>) -> tensor<32x197x768xf32>
    %v2394 = stablehlo.add %v2392, %v2393 : tensor<32x197x768xf32>
    %v2395 = stablehlo.reshape %v2394 : (tensor<32x197x768xf32>) -> tensor<32x151296xf32>
    %v2396 = stablehlo.constant dense<0.5> : tensor<32x151296xf32>
    %v2397 = stablehlo.multiply %v2396, %v2395 : tensor<32x151296xf32>
    %v2398 = stablehlo.negate %v2395 : tensor<32x151296xf32>
    %v2399 = stablehlo.constant dense<0.7071067811865476> : tensor<32x151296xf32>
    %v2400 = stablehlo.multiply %v2398, %v2399 : tensor<32x151296xf32>
    %v2401 = chlo.erfc %v2400 : tensor<32x151296xf32> -> tensor<32x151296xf32>
    %v2402 = stablehlo.multiply %v2397, %v2401 : tensor<32x151296xf32>
    %v2403 = stablehlo.reshape %v2402 : (tensor<32x151296xf32>) -> tensor<32x197x768xf32>
    %v2404 = stablehlo.dot_general %v2403, %b11_Wfc2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x197x768xf32>, tensor<768x192xf32>) -> tensor<32x197x192xf32>
    %v2405 = stablehlo.broadcast_in_dim %b11_bfc2, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v2406 = stablehlo.add %v2404, %v2405 : tensor<32x197x192xf32>
    %v2407 = stablehlo.reshape %v2406 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2408 = stablehlo.broadcast_in_dim %dp23, dims = [0] : (tensor<32xf32>) -> tensor<32x37824xf32>
    %v2409 = stablehlo.multiply %v2408, %v2407 : tensor<32x37824xf32>
    %v2410 = stablehlo.add %v2362, %v2409 : tensor<32x37824xf32>
    %v2411 = stablehlo.reshape %v2410 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2412 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2413 = stablehlo.constant dense<192.0> : tensor<32x197x192xf32>
    %v2414 = stablehlo.constant dense<1.0e-5> : tensor<32x197x192xf32>
    %v2415 = stablehlo.reduce(%v2411 init: %v2412) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2416 = stablehlo.broadcast_in_dim %v2415, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v2417 = stablehlo.divide %v2416, %v2413 : tensor<32x197x192xf32>
    %v2418 = stablehlo.subtract %v2411, %v2417 : tensor<32x197x192xf32>
    %v2419 = stablehlo.multiply %v2418, %v2418 : tensor<32x197x192xf32>
    %v2420 = stablehlo.reduce(%v2419 init: %v2412) applies stablehlo.add across dimensions = [2] : (tensor<32x197x192xf32>, tensor<f32>) -> tensor<32x197xf32>
    %v2421 = stablehlo.broadcast_in_dim %v2420, dims = [0, 1] : (tensor<32x197xf32>) -> tensor<32x197x192xf32>
    %v2422 = stablehlo.divide %v2421, %v2413 : tensor<32x197x192xf32>
    %v2423 = stablehlo.add %v2422, %v2414 : tensor<32x197x192xf32>
    %v2424 = stablehlo.rsqrt %v2423 : tensor<32x197x192xf32>
    %v2425 = stablehlo.multiply %v2418, %v2424 : tensor<32x197x192xf32>
    %v2426 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v2427 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x197x192xf32>
    %v2428 = stablehlo.multiply %v2425, %v2426 : tensor<32x197x192xf32>
    %v2429 = stablehlo.add %v2428, %v2427 : tensor<32x197x192xf32>
    %v2430 = stablehlo.reshape %v2429 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2431 = stablehlo.reshape %v2430 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2432 = stablehlo.broadcast_in_dim %gF, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v2433 = stablehlo.multiply %v2431, %v2432 : tensor<32x197x192xf32>
    %v2434 = stablehlo.reshape %v2433 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2435 = stablehlo.reshape %v2434 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2436 = stablehlo.broadcast_in_dim %btF, dims = [2] : (tensor<192xf32>) -> tensor<32x197x192xf32>
    %v2437 = stablehlo.add %v2435, %v2436 : tensor<32x197x192xf32>
    %v2438 = stablehlo.reshape %v2437 : (tensor<32x197x192xf32>) -> tensor<32x37824xf32>
    %v2439 = stablehlo.reshape %v2438 : (tensor<32x37824xf32>) -> tensor<32x197x192xf32>
    %v2440 = stablehlo.slice %v2439 [0:32, 0:1, 0:192] : (tensor<32x197x192xf32>) -> tensor<32x1x192xf32>
    %v2441 = stablehlo.reshape %v2440 : (tensor<32x1x192xf32>) -> tensor<32x192xf32>
    %v2442 = stablehlo.dot_general %v2441, %Wc, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x192xf32>, tensor<192x1000xf32>) -> tensor<32x1000xf32>
    %v2443 = stablehlo.broadcast_in_dim %bc, dims = [1] : (tensor<1000xf32>) -> tensor<32x1000xf32>
    %v2444 = stablehlo.add %v2442, %v2443 : tensor<32x1000xf32>
    return %v2444 : tensor<32x1000xf32>
  }
}

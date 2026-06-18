module @m {
  func.func @mobilenetv2_train_step(%x: tensor<32x150528xf32>, %Ws: tensor<16x3x3x3xf32>, %bs: tensor<16xf32>, %gs: tensor<16xf32>, %bts: tensor<16xf32>, %We1: tensor<64x16x1x1xf32>, %be1: tensor<64xf32>, %ge1: tensor<64xf32>, %bte1: tensor<64xf32>, %Wd1: tensor<64x1x3x3xf32>, %bd1: tensor<64xf32>, %gd1: tensor<64xf32>, %btd1: tensor<64xf32>, %Wp1: tensor<24x64x1x1xf32>, %bp1: tensor<24xf32>, %gp1: tensor<24xf32>, %btp1: tensor<24xf32>, %We2: tensor<96x24x1x1xf32>, %be2: tensor<96xf32>, %ge2: tensor<96xf32>, %bte2: tensor<96xf32>, %Wd2: tensor<96x1x3x3xf32>, %bd2: tensor<96xf32>, %gd2: tensor<96xf32>, %btd2: tensor<96xf32>, %Wp2: tensor<24x96x1x1xf32>, %bp2: tensor<24xf32>, %gp2: tensor<24xf32>, %btp2: tensor<24xf32>, %We3: tensor<96x24x1x1xf32>, %be3: tensor<96xf32>, %ge3: tensor<96xf32>, %bte3: tensor<96xf32>, %Wd3: tensor<96x1x3x3xf32>, %bd3: tensor<96xf32>, %gd3: tensor<96xf32>, %btd3: tensor<96xf32>, %Wp3: tensor<32x96x1x1xf32>, %bp3: tensor<32xf32>, %gp3: tensor<32xf32>, %btp3: tensor<32xf32>, %We4: tensor<128x32x1x1xf32>, %be4: tensor<128xf32>, %ge4: tensor<128xf32>, %bte4: tensor<128xf32>, %Wd4: tensor<128x1x3x3xf32>, %bd4: tensor<128xf32>, %gd4: tensor<128xf32>, %btd4: tensor<128xf32>, %Wp4: tensor<32x128x1x1xf32>, %bp4: tensor<32xf32>, %gp4: tensor<32xf32>, %btp4: tensor<32xf32>, %We5: tensor<128x32x1x1xf32>, %be5: tensor<128xf32>, %ge5: tensor<128xf32>, %bte5: tensor<128xf32>, %Wd5: tensor<128x1x3x3xf32>, %bd5: tensor<128xf32>, %gd5: tensor<128xf32>, %btd5: tensor<128xf32>, %Wp5: tensor<64x128x1x1xf32>, %bp5: tensor<64xf32>, %gp5: tensor<64xf32>, %btp5: tensor<64xf32>, %We6: tensor<256x64x1x1xf32>, %be6: tensor<256xf32>, %ge6: tensor<256xf32>, %bte6: tensor<256xf32>, %Wd6: tensor<256x1x3x3xf32>, %bd6: tensor<256xf32>, %gd6: tensor<256xf32>, %btd6: tensor<256xf32>, %Wp6: tensor<64x256x1x1xf32>, %bp6: tensor<64xf32>, %gp6: tensor<64xf32>, %btp6: tensor<64xf32>, %Wh: tensor<128x64x1x1xf32>, %bh: tensor<128xf32>, %gh: tensor<128xf32>, %bth: tensor<128xf32>, %Wfc: tensor<128x10xf32>, %bfc: tensor<10xf32>, %onehot: tensor<32x10xf32>) -> (tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<64x16x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x1x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<24x64x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<96x24x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x3x3xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<24x96x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<96x24x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x3x3xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<32x96x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x32x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x1x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<32x128x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x32x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x1x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<64x128x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x1x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x10xf32>, tensor<10xf32>) {
    // ── MobileNetV2 train step: every line is pretty(verified AST node) ──
    %v0 = stablehlo.reshape %x : (tensor<32x150528xf32>) -> tensor<32x3x224x224xf32>
    %v1 = stablehlo.convolution(%v0, %Ws)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3x224x224xf32>, tensor<16x3x3x3xf32>) -> tensor<32x16x112x112xf32>
    %v2 = stablehlo.broadcast_in_dim %bs, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v3 = stablehlo.add %v1, %v2 : tensor<32x16x112x112xf32>
    %v4 = stablehlo.reshape %v3 : (tensor<32x16x112x112xf32>) -> tensor<32x200704xf32>
    %v5 = stablehlo.reshape %v4 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v6 = stablehlo.constant dense<0.0> : tensor<f32>
    %v7 = stablehlo.constant dense<12544.0> : tensor<32x16x112x112xf32>
    %v8 = stablehlo.constant dense<1.0e-5> : tensor<32x16x112x112xf32>
    %v9 = stablehlo.reduce(%v5 init: %v6) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<32x16xf32>
    %v10 = stablehlo.broadcast_in_dim %v9, dims = [0, 1] : (tensor<32x16xf32>) -> tensor<32x16x112x112xf32>
    %v11 = stablehlo.divide %v10, %v7 : tensor<32x16x112x112xf32>
    %v12 = stablehlo.subtract %v5, %v11 : tensor<32x16x112x112xf32>
    %v13 = stablehlo.multiply %v12, %v12 : tensor<32x16x112x112xf32>
    %v14 = stablehlo.reduce(%v13 init: %v6) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<32x16xf32>
    %v15 = stablehlo.broadcast_in_dim %v14, dims = [0, 1] : (tensor<32x16xf32>) -> tensor<32x16x112x112xf32>
    %v16 = stablehlo.divide %v15, %v7 : tensor<32x16x112x112xf32>
    %v17 = stablehlo.add %v16, %v8 : tensor<32x16x112x112xf32>
    %v18 = stablehlo.rsqrt %v17 : tensor<32x16x112x112xf32>
    %v19 = stablehlo.multiply %v12, %v18 : tensor<32x16x112x112xf32>
    %v20 = stablehlo.broadcast_in_dim %gs, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v21 = stablehlo.broadcast_in_dim %bts, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v22 = stablehlo.multiply %v19, %v20 : tensor<32x16x112x112xf32>
    %v23 = stablehlo.add %v22, %v21 : tensor<32x16x112x112xf32>
    %v24 = stablehlo.reshape %v23 : (tensor<32x16x112x112xf32>) -> tensor<32x200704xf32>
    %v25 = stablehlo.constant dense<0.0> : tensor<32x200704xf32>
    %v26 = stablehlo.constant dense<6.0> : tensor<32x200704xf32>
    %v27 = stablehlo.maximum %v24, %v25 : tensor<32x200704xf32>
    %v28 = stablehlo.minimum %v27, %v26 : tensor<32x200704xf32>
    %v29 = stablehlo.reshape %v28 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v30 = stablehlo.convolution(%v29, %We1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x16x112x112xf32>, tensor<64x16x1x1xf32>) -> tensor<32x64x112x112xf32>
    %v31 = stablehlo.broadcast_in_dim %be1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v32 = stablehlo.add %v30, %v31 : tensor<32x64x112x112xf32>
    %v33 = stablehlo.reshape %v32 : (tensor<32x64x112x112xf32>) -> tensor<32x802816xf32>
    %v34 = stablehlo.reshape %v33 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v35 = stablehlo.constant dense<0.0> : tensor<f32>
    %v36 = stablehlo.constant dense<12544.0> : tensor<32x64x112x112xf32>
    %v37 = stablehlo.constant dense<1.0e-5> : tensor<32x64x112x112xf32>
    %v38 = stablehlo.reduce(%v34 init: %v35) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v39 = stablehlo.broadcast_in_dim %v38, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x112x112xf32>
    %v40 = stablehlo.divide %v39, %v36 : tensor<32x64x112x112xf32>
    %v41 = stablehlo.subtract %v34, %v40 : tensor<32x64x112x112xf32>
    %v42 = stablehlo.multiply %v41, %v41 : tensor<32x64x112x112xf32>
    %v43 = stablehlo.reduce(%v42 init: %v35) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v44 = stablehlo.broadcast_in_dim %v43, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x112x112xf32>
    %v45 = stablehlo.divide %v44, %v36 : tensor<32x64x112x112xf32>
    %v46 = stablehlo.add %v45, %v37 : tensor<32x64x112x112xf32>
    %v47 = stablehlo.rsqrt %v46 : tensor<32x64x112x112xf32>
    %v48 = stablehlo.multiply %v41, %v47 : tensor<32x64x112x112xf32>
    %v49 = stablehlo.broadcast_in_dim %ge1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v50 = stablehlo.broadcast_in_dim %bte1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v51 = stablehlo.multiply %v48, %v49 : tensor<32x64x112x112xf32>
    %v52 = stablehlo.add %v51, %v50 : tensor<32x64x112x112xf32>
    %v53 = stablehlo.reshape %v52 : (tensor<32x64x112x112xf32>) -> tensor<32x802816xf32>
    %v54 = stablehlo.constant dense<0.0> : tensor<32x802816xf32>
    %v55 = stablehlo.constant dense<6.0> : tensor<32x802816xf32>
    %v56 = stablehlo.maximum %v53, %v54 : tensor<32x802816xf32>
    %v57 = stablehlo.minimum %v56, %v55 : tensor<32x802816xf32>
    %v58 = stablehlo.reshape %v57 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v59 = stablehlo.convolution(%v58, %Wd1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 64 : i64} : (tensor<32x64x112x112xf32>, tensor<64x1x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v60 = stablehlo.broadcast_in_dim %bd1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v61 = stablehlo.add %v59, %v60 : tensor<32x64x56x56xf32>
    %v62 = stablehlo.reshape %v61 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v63 = stablehlo.reshape %v62 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v64 = stablehlo.constant dense<0.0> : tensor<f32>
    %v65 = stablehlo.constant dense<3136.0> : tensor<32x64x56x56xf32>
    %v66 = stablehlo.constant dense<1.0e-5> : tensor<32x64x56x56xf32>
    %v67 = stablehlo.reduce(%v63 init: %v64) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v68 = stablehlo.broadcast_in_dim %v67, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v69 = stablehlo.divide %v68, %v65 : tensor<32x64x56x56xf32>
    %v70 = stablehlo.subtract %v63, %v69 : tensor<32x64x56x56xf32>
    %v71 = stablehlo.multiply %v70, %v70 : tensor<32x64x56x56xf32>
    %v72 = stablehlo.reduce(%v71 init: %v64) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v73 = stablehlo.broadcast_in_dim %v72, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v74 = stablehlo.divide %v73, %v65 : tensor<32x64x56x56xf32>
    %v75 = stablehlo.add %v74, %v66 : tensor<32x64x56x56xf32>
    %v76 = stablehlo.rsqrt %v75 : tensor<32x64x56x56xf32>
    %v77 = stablehlo.multiply %v70, %v76 : tensor<32x64x56x56xf32>
    %v78 = stablehlo.broadcast_in_dim %gd1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v79 = stablehlo.broadcast_in_dim %btd1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v80 = stablehlo.multiply %v77, %v78 : tensor<32x64x56x56xf32>
    %v81 = stablehlo.add %v80, %v79 : tensor<32x64x56x56xf32>
    %v82 = stablehlo.reshape %v81 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v83 = stablehlo.constant dense<0.0> : tensor<32x200704xf32>
    %v84 = stablehlo.constant dense<6.0> : tensor<32x200704xf32>
    %v85 = stablehlo.maximum %v82, %v83 : tensor<32x200704xf32>
    %v86 = stablehlo.minimum %v85, %v84 : tensor<32x200704xf32>
    %v87 = stablehlo.reshape %v86 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v88 = stablehlo.convolution(%v87, %Wp1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<24x64x1x1xf32>) -> tensor<32x24x56x56xf32>
    %v89 = stablehlo.broadcast_in_dim %bp1, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v90 = stablehlo.add %v88, %v89 : tensor<32x24x56x56xf32>
    %v91 = stablehlo.reshape %v90 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v92 = stablehlo.reshape %v91 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v93 = stablehlo.constant dense<0.0> : tensor<f32>
    %v94 = stablehlo.constant dense<3136.0> : tensor<32x24x56x56xf32>
    %v95 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v96 = stablehlo.reduce(%v92 init: %v93) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v97 = stablehlo.broadcast_in_dim %v96, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v98 = stablehlo.divide %v97, %v94 : tensor<32x24x56x56xf32>
    %v99 = stablehlo.subtract %v92, %v98 : tensor<32x24x56x56xf32>
    %v100 = stablehlo.multiply %v99, %v99 : tensor<32x24x56x56xf32>
    %v101 = stablehlo.reduce(%v100 init: %v93) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v102 = stablehlo.broadcast_in_dim %v101, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v103 = stablehlo.divide %v102, %v94 : tensor<32x24x56x56xf32>
    %v104 = stablehlo.add %v103, %v95 : tensor<32x24x56x56xf32>
    %v105 = stablehlo.rsqrt %v104 : tensor<32x24x56x56xf32>
    %v106 = stablehlo.multiply %v99, %v105 : tensor<32x24x56x56xf32>
    %v107 = stablehlo.broadcast_in_dim %gp1, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v108 = stablehlo.broadcast_in_dim %btp1, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v109 = stablehlo.multiply %v106, %v107 : tensor<32x24x56x56xf32>
    %v110 = stablehlo.add %v109, %v108 : tensor<32x24x56x56xf32>
    %v111 = stablehlo.reshape %v110 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v112 = stablehlo.reshape %v111 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v113 = stablehlo.convolution(%v112, %We2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<96x24x1x1xf32>) -> tensor<32x96x56x56xf32>
    %v114 = stablehlo.broadcast_in_dim %be2, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v115 = stablehlo.add %v113, %v114 : tensor<32x96x56x56xf32>
    %v116 = stablehlo.reshape %v115 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v117 = stablehlo.reshape %v116 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v118 = stablehlo.constant dense<0.0> : tensor<f32>
    %v119 = stablehlo.constant dense<3136.0> : tensor<32x96x56x56xf32>
    %v120 = stablehlo.constant dense<1.0e-5> : tensor<32x96x56x56xf32>
    %v121 = stablehlo.reduce(%v117 init: %v118) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v122 = stablehlo.broadcast_in_dim %v121, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v123 = stablehlo.divide %v122, %v119 : tensor<32x96x56x56xf32>
    %v124 = stablehlo.subtract %v117, %v123 : tensor<32x96x56x56xf32>
    %v125 = stablehlo.multiply %v124, %v124 : tensor<32x96x56x56xf32>
    %v126 = stablehlo.reduce(%v125 init: %v118) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v127 = stablehlo.broadcast_in_dim %v126, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v128 = stablehlo.divide %v127, %v119 : tensor<32x96x56x56xf32>
    %v129 = stablehlo.add %v128, %v120 : tensor<32x96x56x56xf32>
    %v130 = stablehlo.rsqrt %v129 : tensor<32x96x56x56xf32>
    %v131 = stablehlo.multiply %v124, %v130 : tensor<32x96x56x56xf32>
    %v132 = stablehlo.broadcast_in_dim %ge2, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v133 = stablehlo.broadcast_in_dim %bte2, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v134 = stablehlo.multiply %v131, %v132 : tensor<32x96x56x56xf32>
    %v135 = stablehlo.add %v134, %v133 : tensor<32x96x56x56xf32>
    %v136 = stablehlo.reshape %v135 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v137 = stablehlo.constant dense<0.0> : tensor<32x301056xf32>
    %v138 = stablehlo.constant dense<6.0> : tensor<32x301056xf32>
    %v139 = stablehlo.maximum %v136, %v137 : tensor<32x301056xf32>
    %v140 = stablehlo.minimum %v139, %v138 : tensor<32x301056xf32>
    %v141 = stablehlo.reshape %v140 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v142 = stablehlo.convolution(%v141, %Wd2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x3x3xf32>) -> tensor<32x96x56x56xf32>
    %v143 = stablehlo.broadcast_in_dim %bd2, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v144 = stablehlo.add %v142, %v143 : tensor<32x96x56x56xf32>
    %v145 = stablehlo.reshape %v144 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v146 = stablehlo.reshape %v145 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v147 = stablehlo.constant dense<0.0> : tensor<f32>
    %v148 = stablehlo.constant dense<3136.0> : tensor<32x96x56x56xf32>
    %v149 = stablehlo.constant dense<1.0e-5> : tensor<32x96x56x56xf32>
    %v150 = stablehlo.reduce(%v146 init: %v147) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v151 = stablehlo.broadcast_in_dim %v150, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v152 = stablehlo.divide %v151, %v148 : tensor<32x96x56x56xf32>
    %v153 = stablehlo.subtract %v146, %v152 : tensor<32x96x56x56xf32>
    %v154 = stablehlo.multiply %v153, %v153 : tensor<32x96x56x56xf32>
    %v155 = stablehlo.reduce(%v154 init: %v147) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v156 = stablehlo.broadcast_in_dim %v155, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v157 = stablehlo.divide %v156, %v148 : tensor<32x96x56x56xf32>
    %v158 = stablehlo.add %v157, %v149 : tensor<32x96x56x56xf32>
    %v159 = stablehlo.rsqrt %v158 : tensor<32x96x56x56xf32>
    %v160 = stablehlo.multiply %v153, %v159 : tensor<32x96x56x56xf32>
    %v161 = stablehlo.broadcast_in_dim %gd2, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v162 = stablehlo.broadcast_in_dim %btd2, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v163 = stablehlo.multiply %v160, %v161 : tensor<32x96x56x56xf32>
    %v164 = stablehlo.add %v163, %v162 : tensor<32x96x56x56xf32>
    %v165 = stablehlo.reshape %v164 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v166 = stablehlo.constant dense<0.0> : tensor<32x301056xf32>
    %v167 = stablehlo.constant dense<6.0> : tensor<32x301056xf32>
    %v168 = stablehlo.maximum %v165, %v166 : tensor<32x301056xf32>
    %v169 = stablehlo.minimum %v168, %v167 : tensor<32x301056xf32>
    %v170 = stablehlo.reshape %v169 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v171 = stablehlo.convolution(%v170, %Wp2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<24x96x1x1xf32>) -> tensor<32x24x56x56xf32>
    %v172 = stablehlo.broadcast_in_dim %bp2, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v173 = stablehlo.add %v171, %v172 : tensor<32x24x56x56xf32>
    %v174 = stablehlo.reshape %v173 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v175 = stablehlo.reshape %v174 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v176 = stablehlo.constant dense<0.0> : tensor<f32>
    %v177 = stablehlo.constant dense<3136.0> : tensor<32x24x56x56xf32>
    %v178 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v179 = stablehlo.reduce(%v175 init: %v176) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v180 = stablehlo.broadcast_in_dim %v179, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v181 = stablehlo.divide %v180, %v177 : tensor<32x24x56x56xf32>
    %v182 = stablehlo.subtract %v175, %v181 : tensor<32x24x56x56xf32>
    %v183 = stablehlo.multiply %v182, %v182 : tensor<32x24x56x56xf32>
    %v184 = stablehlo.reduce(%v183 init: %v176) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v185 = stablehlo.broadcast_in_dim %v184, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v186 = stablehlo.divide %v185, %v177 : tensor<32x24x56x56xf32>
    %v187 = stablehlo.add %v186, %v178 : tensor<32x24x56x56xf32>
    %v188 = stablehlo.rsqrt %v187 : tensor<32x24x56x56xf32>
    %v189 = stablehlo.multiply %v182, %v188 : tensor<32x24x56x56xf32>
    %v190 = stablehlo.broadcast_in_dim %gp2, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v191 = stablehlo.broadcast_in_dim %btp2, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v192 = stablehlo.multiply %v189, %v190 : tensor<32x24x56x56xf32>
    %v193 = stablehlo.add %v192, %v191 : tensor<32x24x56x56xf32>
    %v194 = stablehlo.reshape %v193 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v195 = stablehlo.add %v194, %v111 : tensor<32x75264xf32>
    %v196 = stablehlo.reshape %v195 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v197 = stablehlo.convolution(%v196, %We3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<96x24x1x1xf32>) -> tensor<32x96x56x56xf32>
    %v198 = stablehlo.broadcast_in_dim %be3, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v199 = stablehlo.add %v197, %v198 : tensor<32x96x56x56xf32>
    %v200 = stablehlo.reshape %v199 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v201 = stablehlo.reshape %v200 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v202 = stablehlo.constant dense<0.0> : tensor<f32>
    %v203 = stablehlo.constant dense<3136.0> : tensor<32x96x56x56xf32>
    %v204 = stablehlo.constant dense<1.0e-5> : tensor<32x96x56x56xf32>
    %v205 = stablehlo.reduce(%v201 init: %v202) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v206 = stablehlo.broadcast_in_dim %v205, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v207 = stablehlo.divide %v206, %v203 : tensor<32x96x56x56xf32>
    %v208 = stablehlo.subtract %v201, %v207 : tensor<32x96x56x56xf32>
    %v209 = stablehlo.multiply %v208, %v208 : tensor<32x96x56x56xf32>
    %v210 = stablehlo.reduce(%v209 init: %v202) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v211 = stablehlo.broadcast_in_dim %v210, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v212 = stablehlo.divide %v211, %v203 : tensor<32x96x56x56xf32>
    %v213 = stablehlo.add %v212, %v204 : tensor<32x96x56x56xf32>
    %v214 = stablehlo.rsqrt %v213 : tensor<32x96x56x56xf32>
    %v215 = stablehlo.multiply %v208, %v214 : tensor<32x96x56x56xf32>
    %v216 = stablehlo.broadcast_in_dim %ge3, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v217 = stablehlo.broadcast_in_dim %bte3, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v218 = stablehlo.multiply %v215, %v216 : tensor<32x96x56x56xf32>
    %v219 = stablehlo.add %v218, %v217 : tensor<32x96x56x56xf32>
    %v220 = stablehlo.reshape %v219 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v221 = stablehlo.constant dense<0.0> : tensor<32x301056xf32>
    %v222 = stablehlo.constant dense<6.0> : tensor<32x301056xf32>
    %v223 = stablehlo.maximum %v220, %v221 : tensor<32x301056xf32>
    %v224 = stablehlo.minimum %v223, %v222 : tensor<32x301056xf32>
    %v225 = stablehlo.reshape %v224 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v226 = stablehlo.convolution(%v225, %Wd3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x3x3xf32>) -> tensor<32x96x28x28xf32>
    %v227 = stablehlo.broadcast_in_dim %bd3, dims = [1] : (tensor<96xf32>) -> tensor<32x96x28x28xf32>
    %v228 = stablehlo.add %v226, %v227 : tensor<32x96x28x28xf32>
    %v229 = stablehlo.reshape %v228 : (tensor<32x96x28x28xf32>) -> tensor<32x75264xf32>
    %v230 = stablehlo.reshape %v229 : (tensor<32x75264xf32>) -> tensor<32x96x28x28xf32>
    %v231 = stablehlo.constant dense<0.0> : tensor<f32>
    %v232 = stablehlo.constant dense<784.0> : tensor<32x96x28x28xf32>
    %v233 = stablehlo.constant dense<1.0e-5> : tensor<32x96x28x28xf32>
    %v234 = stablehlo.reduce(%v230 init: %v231) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x28x28xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v235 = stablehlo.broadcast_in_dim %v234, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x28x28xf32>
    %v236 = stablehlo.divide %v235, %v232 : tensor<32x96x28x28xf32>
    %v237 = stablehlo.subtract %v230, %v236 : tensor<32x96x28x28xf32>
    %v238 = stablehlo.multiply %v237, %v237 : tensor<32x96x28x28xf32>
    %v239 = stablehlo.reduce(%v238 init: %v231) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x28x28xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v240 = stablehlo.broadcast_in_dim %v239, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x28x28xf32>
    %v241 = stablehlo.divide %v240, %v232 : tensor<32x96x28x28xf32>
    %v242 = stablehlo.add %v241, %v233 : tensor<32x96x28x28xf32>
    %v243 = stablehlo.rsqrt %v242 : tensor<32x96x28x28xf32>
    %v244 = stablehlo.multiply %v237, %v243 : tensor<32x96x28x28xf32>
    %v245 = stablehlo.broadcast_in_dim %gd3, dims = [1] : (tensor<96xf32>) -> tensor<32x96x28x28xf32>
    %v246 = stablehlo.broadcast_in_dim %btd3, dims = [1] : (tensor<96xf32>) -> tensor<32x96x28x28xf32>
    %v247 = stablehlo.multiply %v244, %v245 : tensor<32x96x28x28xf32>
    %v248 = stablehlo.add %v247, %v246 : tensor<32x96x28x28xf32>
    %v249 = stablehlo.reshape %v248 : (tensor<32x96x28x28xf32>) -> tensor<32x75264xf32>
    %v250 = stablehlo.constant dense<0.0> : tensor<32x75264xf32>
    %v251 = stablehlo.constant dense<6.0> : tensor<32x75264xf32>
    %v252 = stablehlo.maximum %v249, %v250 : tensor<32x75264xf32>
    %v253 = stablehlo.minimum %v252, %v251 : tensor<32x75264xf32>
    %v254 = stablehlo.reshape %v253 : (tensor<32x75264xf32>) -> tensor<32x96x28x28xf32>
    %v255 = stablehlo.convolution(%v254, %Wp3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x28x28xf32>, tensor<32x96x1x1xf32>) -> tensor<32x32x28x28xf32>
    %v256 = stablehlo.broadcast_in_dim %bp3, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v257 = stablehlo.add %v255, %v256 : tensor<32x32x28x28xf32>
    %v258 = stablehlo.reshape %v257 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v259 = stablehlo.reshape %v258 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v260 = stablehlo.constant dense<0.0> : tensor<f32>
    %v261 = stablehlo.constant dense<784.0> : tensor<32x32x28x28xf32>
    %v262 = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %v263 = stablehlo.reduce(%v259 init: %v260) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v264 = stablehlo.broadcast_in_dim %v263, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v265 = stablehlo.divide %v264, %v261 : tensor<32x32x28x28xf32>
    %v266 = stablehlo.subtract %v259, %v265 : tensor<32x32x28x28xf32>
    %v267 = stablehlo.multiply %v266, %v266 : tensor<32x32x28x28xf32>
    %v268 = stablehlo.reduce(%v267 init: %v260) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v269 = stablehlo.broadcast_in_dim %v268, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v270 = stablehlo.divide %v269, %v261 : tensor<32x32x28x28xf32>
    %v271 = stablehlo.add %v270, %v262 : tensor<32x32x28x28xf32>
    %v272 = stablehlo.rsqrt %v271 : tensor<32x32x28x28xf32>
    %v273 = stablehlo.multiply %v266, %v272 : tensor<32x32x28x28xf32>
    %v274 = stablehlo.broadcast_in_dim %gp3, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v275 = stablehlo.broadcast_in_dim %btp3, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v276 = stablehlo.multiply %v273, %v274 : tensor<32x32x28x28xf32>
    %v277 = stablehlo.add %v276, %v275 : tensor<32x32x28x28xf32>
    %v278 = stablehlo.reshape %v277 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v279 = stablehlo.reshape %v278 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v280 = stablehlo.convolution(%v279, %We4)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<128x32x1x1xf32>) -> tensor<32x128x28x28xf32>
    %v281 = stablehlo.broadcast_in_dim %be4, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v282 = stablehlo.add %v280, %v281 : tensor<32x128x28x28xf32>
    %v283 = stablehlo.reshape %v282 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v284 = stablehlo.reshape %v283 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v285 = stablehlo.constant dense<0.0> : tensor<f32>
    %v286 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v287 = stablehlo.constant dense<1.0e-5> : tensor<32x128x28x28xf32>
    %v288 = stablehlo.reduce(%v284 init: %v285) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v289 = stablehlo.broadcast_in_dim %v288, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v290 = stablehlo.divide %v289, %v286 : tensor<32x128x28x28xf32>
    %v291 = stablehlo.subtract %v284, %v290 : tensor<32x128x28x28xf32>
    %v292 = stablehlo.multiply %v291, %v291 : tensor<32x128x28x28xf32>
    %v293 = stablehlo.reduce(%v292 init: %v285) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v294 = stablehlo.broadcast_in_dim %v293, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v295 = stablehlo.divide %v294, %v286 : tensor<32x128x28x28xf32>
    %v296 = stablehlo.add %v295, %v287 : tensor<32x128x28x28xf32>
    %v297 = stablehlo.rsqrt %v296 : tensor<32x128x28x28xf32>
    %v298 = stablehlo.multiply %v291, %v297 : tensor<32x128x28x28xf32>
    %v299 = stablehlo.broadcast_in_dim %ge4, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v300 = stablehlo.broadcast_in_dim %bte4, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v301 = stablehlo.multiply %v298, %v299 : tensor<32x128x28x28xf32>
    %v302 = stablehlo.add %v301, %v300 : tensor<32x128x28x28xf32>
    %v303 = stablehlo.reshape %v302 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v304 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v305 = stablehlo.constant dense<6.0> : tensor<32x100352xf32>
    %v306 = stablehlo.maximum %v303, %v304 : tensor<32x100352xf32>
    %v307 = stablehlo.minimum %v306, %v305 : tensor<32x100352xf32>
    %v308 = stablehlo.reshape %v307 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v309 = stablehlo.convolution(%v308, %Wd4)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 128 : i64} : (tensor<32x128x28x28xf32>, tensor<128x1x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v310 = stablehlo.broadcast_in_dim %bd4, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v311 = stablehlo.add %v309, %v310 : tensor<32x128x28x28xf32>
    %v312 = stablehlo.reshape %v311 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v313 = stablehlo.reshape %v312 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v314 = stablehlo.constant dense<0.0> : tensor<f32>
    %v315 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v316 = stablehlo.constant dense<1.0e-5> : tensor<32x128x28x28xf32>
    %v317 = stablehlo.reduce(%v313 init: %v314) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v318 = stablehlo.broadcast_in_dim %v317, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v319 = stablehlo.divide %v318, %v315 : tensor<32x128x28x28xf32>
    %v320 = stablehlo.subtract %v313, %v319 : tensor<32x128x28x28xf32>
    %v321 = stablehlo.multiply %v320, %v320 : tensor<32x128x28x28xf32>
    %v322 = stablehlo.reduce(%v321 init: %v314) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v323 = stablehlo.broadcast_in_dim %v322, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v324 = stablehlo.divide %v323, %v315 : tensor<32x128x28x28xf32>
    %v325 = stablehlo.add %v324, %v316 : tensor<32x128x28x28xf32>
    %v326 = stablehlo.rsqrt %v325 : tensor<32x128x28x28xf32>
    %v327 = stablehlo.multiply %v320, %v326 : tensor<32x128x28x28xf32>
    %v328 = stablehlo.broadcast_in_dim %gd4, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v329 = stablehlo.broadcast_in_dim %btd4, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v330 = stablehlo.multiply %v327, %v328 : tensor<32x128x28x28xf32>
    %v331 = stablehlo.add %v330, %v329 : tensor<32x128x28x28xf32>
    %v332 = stablehlo.reshape %v331 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v333 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v334 = stablehlo.constant dense<6.0> : tensor<32x100352xf32>
    %v335 = stablehlo.maximum %v332, %v333 : tensor<32x100352xf32>
    %v336 = stablehlo.minimum %v335, %v334 : tensor<32x100352xf32>
    %v337 = stablehlo.reshape %v336 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v338 = stablehlo.convolution(%v337, %Wp4)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<32x128x1x1xf32>) -> tensor<32x32x28x28xf32>
    %v339 = stablehlo.broadcast_in_dim %bp4, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v340 = stablehlo.add %v338, %v339 : tensor<32x32x28x28xf32>
    %v341 = stablehlo.reshape %v340 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v342 = stablehlo.reshape %v341 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v343 = stablehlo.constant dense<0.0> : tensor<f32>
    %v344 = stablehlo.constant dense<784.0> : tensor<32x32x28x28xf32>
    %v345 = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %v346 = stablehlo.reduce(%v342 init: %v343) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v347 = stablehlo.broadcast_in_dim %v346, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v348 = stablehlo.divide %v347, %v344 : tensor<32x32x28x28xf32>
    %v349 = stablehlo.subtract %v342, %v348 : tensor<32x32x28x28xf32>
    %v350 = stablehlo.multiply %v349, %v349 : tensor<32x32x28x28xf32>
    %v351 = stablehlo.reduce(%v350 init: %v343) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v352 = stablehlo.broadcast_in_dim %v351, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v353 = stablehlo.divide %v352, %v344 : tensor<32x32x28x28xf32>
    %v354 = stablehlo.add %v353, %v345 : tensor<32x32x28x28xf32>
    %v355 = stablehlo.rsqrt %v354 : tensor<32x32x28x28xf32>
    %v356 = stablehlo.multiply %v349, %v355 : tensor<32x32x28x28xf32>
    %v357 = stablehlo.broadcast_in_dim %gp4, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v358 = stablehlo.broadcast_in_dim %btp4, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v359 = stablehlo.multiply %v356, %v357 : tensor<32x32x28x28xf32>
    %v360 = stablehlo.add %v359, %v358 : tensor<32x32x28x28xf32>
    %v361 = stablehlo.reshape %v360 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v362 = stablehlo.add %v361, %v278 : tensor<32x25088xf32>
    %v363 = stablehlo.reshape %v362 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v364 = stablehlo.convolution(%v363, %We5)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<128x32x1x1xf32>) -> tensor<32x128x28x28xf32>
    %v365 = stablehlo.broadcast_in_dim %be5, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v366 = stablehlo.add %v364, %v365 : tensor<32x128x28x28xf32>
    %v367 = stablehlo.reshape %v366 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v368 = stablehlo.reshape %v367 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v369 = stablehlo.constant dense<0.0> : tensor<f32>
    %v370 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v371 = stablehlo.constant dense<1.0e-5> : tensor<32x128x28x28xf32>
    %v372 = stablehlo.reduce(%v368 init: %v369) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v373 = stablehlo.broadcast_in_dim %v372, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v374 = stablehlo.divide %v373, %v370 : tensor<32x128x28x28xf32>
    %v375 = stablehlo.subtract %v368, %v374 : tensor<32x128x28x28xf32>
    %v376 = stablehlo.multiply %v375, %v375 : tensor<32x128x28x28xf32>
    %v377 = stablehlo.reduce(%v376 init: %v369) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v378 = stablehlo.broadcast_in_dim %v377, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v379 = stablehlo.divide %v378, %v370 : tensor<32x128x28x28xf32>
    %v380 = stablehlo.add %v379, %v371 : tensor<32x128x28x28xf32>
    %v381 = stablehlo.rsqrt %v380 : tensor<32x128x28x28xf32>
    %v382 = stablehlo.multiply %v375, %v381 : tensor<32x128x28x28xf32>
    %v383 = stablehlo.broadcast_in_dim %ge5, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v384 = stablehlo.broadcast_in_dim %bte5, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v385 = stablehlo.multiply %v382, %v383 : tensor<32x128x28x28xf32>
    %v386 = stablehlo.add %v385, %v384 : tensor<32x128x28x28xf32>
    %v387 = stablehlo.reshape %v386 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v388 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v389 = stablehlo.constant dense<6.0> : tensor<32x100352xf32>
    %v390 = stablehlo.maximum %v387, %v388 : tensor<32x100352xf32>
    %v391 = stablehlo.minimum %v390, %v389 : tensor<32x100352xf32>
    %v392 = stablehlo.reshape %v391 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v393 = stablehlo.convolution(%v392, %Wd5)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 128 : i64} : (tensor<32x128x28x28xf32>, tensor<128x1x3x3xf32>) -> tensor<32x128x14x14xf32>
    %v394 = stablehlo.broadcast_in_dim %bd5, dims = [1] : (tensor<128xf32>) -> tensor<32x128x14x14xf32>
    %v395 = stablehlo.add %v393, %v394 : tensor<32x128x14x14xf32>
    %v396 = stablehlo.reshape %v395 : (tensor<32x128x14x14xf32>) -> tensor<32x25088xf32>
    %v397 = stablehlo.reshape %v396 : (tensor<32x25088xf32>) -> tensor<32x128x14x14xf32>
    %v398 = stablehlo.constant dense<0.0> : tensor<f32>
    %v399 = stablehlo.constant dense<196.0> : tensor<32x128x14x14xf32>
    %v400 = stablehlo.constant dense<1.0e-5> : tensor<32x128x14x14xf32>
    %v401 = stablehlo.reduce(%v397 init: %v398) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x14x14xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v402 = stablehlo.broadcast_in_dim %v401, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x14x14xf32>
    %v403 = stablehlo.divide %v402, %v399 : tensor<32x128x14x14xf32>
    %v404 = stablehlo.subtract %v397, %v403 : tensor<32x128x14x14xf32>
    %v405 = stablehlo.multiply %v404, %v404 : tensor<32x128x14x14xf32>
    %v406 = stablehlo.reduce(%v405 init: %v398) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x14x14xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v407 = stablehlo.broadcast_in_dim %v406, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x14x14xf32>
    %v408 = stablehlo.divide %v407, %v399 : tensor<32x128x14x14xf32>
    %v409 = stablehlo.add %v408, %v400 : tensor<32x128x14x14xf32>
    %v410 = stablehlo.rsqrt %v409 : tensor<32x128x14x14xf32>
    %v411 = stablehlo.multiply %v404, %v410 : tensor<32x128x14x14xf32>
    %v412 = stablehlo.broadcast_in_dim %gd5, dims = [1] : (tensor<128xf32>) -> tensor<32x128x14x14xf32>
    %v413 = stablehlo.broadcast_in_dim %btd5, dims = [1] : (tensor<128xf32>) -> tensor<32x128x14x14xf32>
    %v414 = stablehlo.multiply %v411, %v412 : tensor<32x128x14x14xf32>
    %v415 = stablehlo.add %v414, %v413 : tensor<32x128x14x14xf32>
    %v416 = stablehlo.reshape %v415 : (tensor<32x128x14x14xf32>) -> tensor<32x25088xf32>
    %v417 = stablehlo.constant dense<0.0> : tensor<32x25088xf32>
    %v418 = stablehlo.constant dense<6.0> : tensor<32x25088xf32>
    %v419 = stablehlo.maximum %v416, %v417 : tensor<32x25088xf32>
    %v420 = stablehlo.minimum %v419, %v418 : tensor<32x25088xf32>
    %v421 = stablehlo.reshape %v420 : (tensor<32x25088xf32>) -> tensor<32x128x14x14xf32>
    %v422 = stablehlo.convolution(%v421, %Wp5)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x14x14xf32>, tensor<64x128x1x1xf32>) -> tensor<32x64x14x14xf32>
    %v423 = stablehlo.broadcast_in_dim %bp5, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v424 = stablehlo.add %v422, %v423 : tensor<32x64x14x14xf32>
    %v425 = stablehlo.reshape %v424 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v426 = stablehlo.reshape %v425 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v427 = stablehlo.constant dense<0.0> : tensor<f32>
    %v428 = stablehlo.constant dense<196.0> : tensor<32x64x14x14xf32>
    %v429 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v430 = stablehlo.reduce(%v426 init: %v427) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v431 = stablehlo.broadcast_in_dim %v430, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v432 = stablehlo.divide %v431, %v428 : tensor<32x64x14x14xf32>
    %v433 = stablehlo.subtract %v426, %v432 : tensor<32x64x14x14xf32>
    %v434 = stablehlo.multiply %v433, %v433 : tensor<32x64x14x14xf32>
    %v435 = stablehlo.reduce(%v434 init: %v427) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v436 = stablehlo.broadcast_in_dim %v435, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v437 = stablehlo.divide %v436, %v428 : tensor<32x64x14x14xf32>
    %v438 = stablehlo.add %v437, %v429 : tensor<32x64x14x14xf32>
    %v439 = stablehlo.rsqrt %v438 : tensor<32x64x14x14xf32>
    %v440 = stablehlo.multiply %v433, %v439 : tensor<32x64x14x14xf32>
    %v441 = stablehlo.broadcast_in_dim %gp5, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v442 = stablehlo.broadcast_in_dim %btp5, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v443 = stablehlo.multiply %v440, %v441 : tensor<32x64x14x14xf32>
    %v444 = stablehlo.add %v443, %v442 : tensor<32x64x14x14xf32>
    %v445 = stablehlo.reshape %v444 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v446 = stablehlo.reshape %v445 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v447 = stablehlo.convolution(%v446, %We6)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x14x14xf32>, tensor<256x64x1x1xf32>) -> tensor<32x256x14x14xf32>
    %v448 = stablehlo.broadcast_in_dim %be6, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v449 = stablehlo.add %v447, %v448 : tensor<32x256x14x14xf32>
    %v450 = stablehlo.reshape %v449 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v451 = stablehlo.reshape %v450 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v452 = stablehlo.constant dense<0.0> : tensor<f32>
    %v453 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v454 = stablehlo.constant dense<1.0e-5> : tensor<32x256x14x14xf32>
    %v455 = stablehlo.reduce(%v451 init: %v452) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v456 = stablehlo.broadcast_in_dim %v455, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v457 = stablehlo.divide %v456, %v453 : tensor<32x256x14x14xf32>
    %v458 = stablehlo.subtract %v451, %v457 : tensor<32x256x14x14xf32>
    %v459 = stablehlo.multiply %v458, %v458 : tensor<32x256x14x14xf32>
    %v460 = stablehlo.reduce(%v459 init: %v452) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v461 = stablehlo.broadcast_in_dim %v460, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v462 = stablehlo.divide %v461, %v453 : tensor<32x256x14x14xf32>
    %v463 = stablehlo.add %v462, %v454 : tensor<32x256x14x14xf32>
    %v464 = stablehlo.rsqrt %v463 : tensor<32x256x14x14xf32>
    %v465 = stablehlo.multiply %v458, %v464 : tensor<32x256x14x14xf32>
    %v466 = stablehlo.broadcast_in_dim %ge6, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v467 = stablehlo.broadcast_in_dim %bte6, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v468 = stablehlo.multiply %v465, %v466 : tensor<32x256x14x14xf32>
    %v469 = stablehlo.add %v468, %v467 : tensor<32x256x14x14xf32>
    %v470 = stablehlo.reshape %v469 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v471 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v472 = stablehlo.constant dense<6.0> : tensor<32x50176xf32>
    %v473 = stablehlo.maximum %v470, %v471 : tensor<32x50176xf32>
    %v474 = stablehlo.minimum %v473, %v472 : tensor<32x50176xf32>
    %v475 = stablehlo.reshape %v474 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v476 = stablehlo.convolution(%v475, %Wd6)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 256 : i64} : (tensor<32x256x14x14xf32>, tensor<256x1x3x3xf32>) -> tensor<32x256x7x7xf32>
    %v477 = stablehlo.broadcast_in_dim %bd6, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v478 = stablehlo.add %v476, %v477 : tensor<32x256x7x7xf32>
    %v479 = stablehlo.reshape %v478 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v480 = stablehlo.reshape %v479 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v481 = stablehlo.constant dense<0.0> : tensor<f32>
    %v482 = stablehlo.constant dense<49.0> : tensor<32x256x7x7xf32>
    %v483 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v484 = stablehlo.reduce(%v480 init: %v481) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v485 = stablehlo.broadcast_in_dim %v484, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x7x7xf32>
    %v486 = stablehlo.divide %v485, %v482 : tensor<32x256x7x7xf32>
    %v487 = stablehlo.subtract %v480, %v486 : tensor<32x256x7x7xf32>
    %v488 = stablehlo.multiply %v487, %v487 : tensor<32x256x7x7xf32>
    %v489 = stablehlo.reduce(%v488 init: %v481) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v490 = stablehlo.broadcast_in_dim %v489, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x7x7xf32>
    %v491 = stablehlo.divide %v490, %v482 : tensor<32x256x7x7xf32>
    %v492 = stablehlo.add %v491, %v483 : tensor<32x256x7x7xf32>
    %v493 = stablehlo.rsqrt %v492 : tensor<32x256x7x7xf32>
    %v494 = stablehlo.multiply %v487, %v493 : tensor<32x256x7x7xf32>
    %v495 = stablehlo.broadcast_in_dim %gd6, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v496 = stablehlo.broadcast_in_dim %btd6, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v497 = stablehlo.multiply %v494, %v495 : tensor<32x256x7x7xf32>
    %v498 = stablehlo.add %v497, %v496 : tensor<32x256x7x7xf32>
    %v499 = stablehlo.reshape %v498 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v500 = stablehlo.constant dense<0.0> : tensor<32x12544xf32>
    %v501 = stablehlo.constant dense<6.0> : tensor<32x12544xf32>
    %v502 = stablehlo.maximum %v499, %v500 : tensor<32x12544xf32>
    %v503 = stablehlo.minimum %v502, %v501 : tensor<32x12544xf32>
    %v504 = stablehlo.reshape %v503 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v505 = stablehlo.convolution(%v504, %Wp6)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x7x7xf32>, tensor<64x256x1x1xf32>) -> tensor<32x64x7x7xf32>
    %v506 = stablehlo.broadcast_in_dim %bp6, dims = [1] : (tensor<64xf32>) -> tensor<32x64x7x7xf32>
    %v507 = stablehlo.add %v505, %v506 : tensor<32x64x7x7xf32>
    %v508 = stablehlo.reshape %v507 : (tensor<32x64x7x7xf32>) -> tensor<32x3136xf32>
    %v509 = stablehlo.reshape %v508 : (tensor<32x3136xf32>) -> tensor<32x64x7x7xf32>
    %v510 = stablehlo.constant dense<0.0> : tensor<f32>
    %v511 = stablehlo.constant dense<49.0> : tensor<32x64x7x7xf32>
    %v512 = stablehlo.constant dense<1.0e-5> : tensor<32x64x7x7xf32>
    %v513 = stablehlo.reduce(%v509 init: %v510) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x7x7xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v514 = stablehlo.broadcast_in_dim %v513, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x7x7xf32>
    %v515 = stablehlo.divide %v514, %v511 : tensor<32x64x7x7xf32>
    %v516 = stablehlo.subtract %v509, %v515 : tensor<32x64x7x7xf32>
    %v517 = stablehlo.multiply %v516, %v516 : tensor<32x64x7x7xf32>
    %v518 = stablehlo.reduce(%v517 init: %v510) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x7x7xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v519 = stablehlo.broadcast_in_dim %v518, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x7x7xf32>
    %v520 = stablehlo.divide %v519, %v511 : tensor<32x64x7x7xf32>
    %v521 = stablehlo.add %v520, %v512 : tensor<32x64x7x7xf32>
    %v522 = stablehlo.rsqrt %v521 : tensor<32x64x7x7xf32>
    %v523 = stablehlo.multiply %v516, %v522 : tensor<32x64x7x7xf32>
    %v524 = stablehlo.broadcast_in_dim %gp6, dims = [1] : (tensor<64xf32>) -> tensor<32x64x7x7xf32>
    %v525 = stablehlo.broadcast_in_dim %btp6, dims = [1] : (tensor<64xf32>) -> tensor<32x64x7x7xf32>
    %v526 = stablehlo.multiply %v523, %v524 : tensor<32x64x7x7xf32>
    %v527 = stablehlo.add %v526, %v525 : tensor<32x64x7x7xf32>
    %v528 = stablehlo.reshape %v527 : (tensor<32x64x7x7xf32>) -> tensor<32x3136xf32>
    %v529 = stablehlo.reshape %v528 : (tensor<32x3136xf32>) -> tensor<32x64x7x7xf32>
    %v530 = stablehlo.convolution(%v529, %Wh)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x7x7xf32>, tensor<128x64x1x1xf32>) -> tensor<32x128x7x7xf32>
    %v531 = stablehlo.broadcast_in_dim %bh, dims = [1] : (tensor<128xf32>) -> tensor<32x128x7x7xf32>
    %v532 = stablehlo.add %v530, %v531 : tensor<32x128x7x7xf32>
    %v533 = stablehlo.reshape %v532 : (tensor<32x128x7x7xf32>) -> tensor<32x6272xf32>
    %v534 = stablehlo.reshape %v533 : (tensor<32x6272xf32>) -> tensor<32x128x7x7xf32>
    %v535 = stablehlo.constant dense<0.0> : tensor<f32>
    %v536 = stablehlo.constant dense<49.0> : tensor<32x128x7x7xf32>
    %v537 = stablehlo.constant dense<1.0e-5> : tensor<32x128x7x7xf32>
    %v538 = stablehlo.reduce(%v534 init: %v535) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x7x7xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v539 = stablehlo.broadcast_in_dim %v538, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x7x7xf32>
    %v540 = stablehlo.divide %v539, %v536 : tensor<32x128x7x7xf32>
    %v541 = stablehlo.subtract %v534, %v540 : tensor<32x128x7x7xf32>
    %v542 = stablehlo.multiply %v541, %v541 : tensor<32x128x7x7xf32>
    %v543 = stablehlo.reduce(%v542 init: %v535) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x7x7xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v544 = stablehlo.broadcast_in_dim %v543, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x7x7xf32>
    %v545 = stablehlo.divide %v544, %v536 : tensor<32x128x7x7xf32>
    %v546 = stablehlo.add %v545, %v537 : tensor<32x128x7x7xf32>
    %v547 = stablehlo.rsqrt %v546 : tensor<32x128x7x7xf32>
    %v548 = stablehlo.multiply %v541, %v547 : tensor<32x128x7x7xf32>
    %v549 = stablehlo.broadcast_in_dim %gh, dims = [1] : (tensor<128xf32>) -> tensor<32x128x7x7xf32>
    %v550 = stablehlo.broadcast_in_dim %bth, dims = [1] : (tensor<128xf32>) -> tensor<32x128x7x7xf32>
    %v551 = stablehlo.multiply %v548, %v549 : tensor<32x128x7x7xf32>
    %v552 = stablehlo.add %v551, %v550 : tensor<32x128x7x7xf32>
    %v553 = stablehlo.reshape %v552 : (tensor<32x128x7x7xf32>) -> tensor<32x6272xf32>
    %v554 = stablehlo.constant dense<0.0> : tensor<32x6272xf32>
    %v555 = stablehlo.constant dense<6.0> : tensor<32x6272xf32>
    %v556 = stablehlo.maximum %v553, %v554 : tensor<32x6272xf32>
    %v557 = stablehlo.minimum %v556, %v555 : tensor<32x6272xf32>
    %v558 = stablehlo.reshape %v557 : (tensor<32x6272xf32>) -> tensor<32x128x7x7xf32>
    %v559 = stablehlo.constant dense<0.0> : tensor<f32>
    %v560 = stablehlo.reduce(%v558 init: %v559) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x7x7xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v561 = stablehlo.constant dense<49.0> : tensor<32x128xf32>
    %v562 = stablehlo.divide %v560, %v561 : tensor<32x128xf32>
    %v563 = stablehlo.dot_general %v562, %Wfc, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x128xf32>, tensor<128x10xf32>) -> tensor<32x10xf32>
    %v564 = stablehlo.broadcast_in_dim %bfc, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v565 = stablehlo.add %v563, %v564 : tensor<32x10xf32>
    %v566 = stablehlo.exponential %v565 : tensor<32x10xf32>
    %v567 = stablehlo.constant dense<0.0> : tensor<f32>
    %v568 = stablehlo.reduce(%v566 init: %v567) applies stablehlo.add across dimensions = [1] : (tensor<32x10xf32>, tensor<f32>) -> tensor<32xf32>
    %v569 = stablehlo.broadcast_in_dim %v568, dims = [0] : (tensor<32xf32>) -> tensor<32x10xf32>
    %v570 = stablehlo.divide %v566, %v569 : tensor<32x10xf32>
    %v571 = stablehlo.subtract %v570, %onehot : tensor<32x10xf32>
    %v572 = stablehlo.dot_general %v571, %Wfc, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<128x10xf32>) -> tensor<32x128xf32>
    %v573 = stablehlo.constant dense<49.0> : tensor<32x128xf32>
    %v574 = stablehlo.divide %v572, %v573 : tensor<32x128xf32>
    %v575 = stablehlo.broadcast_in_dim %v574, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x7x7xf32>
    %v576 = stablehlo.reshape %v575 : (tensor<32x128x7x7xf32>) -> tensor<32x6272xf32>
    %v577 = stablehlo.dot_general %v562, %v571, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x128xf32>, tensor<32x10xf32>) -> tensor<128x10xf32>
    %v578 = stablehlo.constant dense<0.3> : tensor<128x10xf32>
    %v579 = stablehlo.multiply %v577, %v578 : tensor<128x10xf32>
    %v580 = stablehlo.subtract %Wfc, %v579 : tensor<128x10xf32>
    %v581 = stablehlo.constant dense<0.0> : tensor<f32>
    %v582 = stablehlo.reduce(%v571 init: %v581) applies stablehlo.add across dimensions = [0] : (tensor<32x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v583 = stablehlo.constant dense<0.3> : tensor<10xf32>
    %v584 = stablehlo.multiply %v582, %v583 : tensor<10xf32>
    %v585 = stablehlo.subtract %bfc, %v584 : tensor<10xf32>
    %v586 = stablehlo.constant dense<0.0> : tensor<32x6272xf32>
    %v587 = stablehlo.constant dense<6.0> : tensor<32x6272xf32>
    %v588 = stablehlo.compare GT, %v553, %v586 : (tensor<32x6272xf32>, tensor<32x6272xf32>) -> tensor<32x6272xi1>
    %v589 = stablehlo.compare LT, %v553, %v587 : (tensor<32x6272xf32>, tensor<32x6272xf32>) -> tensor<32x6272xi1>
    %v590 = stablehlo.and %v588, %v589 : tensor<32x6272xi1>
    %v591 = stablehlo.select %v590, %v576, %v586 : tensor<32x6272xi1>, tensor<32x6272xf32>
    %v592 = stablehlo.reshape %v591 : (tensor<32x6272xf32>) -> tensor<32x128x7x7xf32>
    %v593 = stablehlo.reshape %v533 : (tensor<32x6272xf32>) -> tensor<32x128x7x7xf32>
    %v594 = stablehlo.constant dense<0.0> : tensor<f32>
    %v595 = stablehlo.constant dense<49.0> : tensor<32x128x7x7xf32>
    %v596 = stablehlo.constant dense<1.0e-5> : tensor<32x128x7x7xf32>
    %v597 = stablehlo.reduce(%v593 init: %v594) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x7x7xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v598 = stablehlo.broadcast_in_dim %v597, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x7x7xf32>
    %v599 = stablehlo.divide %v598, %v595 : tensor<32x128x7x7xf32>
    %v600 = stablehlo.subtract %v593, %v599 : tensor<32x128x7x7xf32>
    %v601 = stablehlo.multiply %v600, %v600 : tensor<32x128x7x7xf32>
    %v602 = stablehlo.reduce(%v601 init: %v594) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x7x7xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v603 = stablehlo.broadcast_in_dim %v602, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x7x7xf32>
    %v604 = stablehlo.divide %v603, %v595 : tensor<32x128x7x7xf32>
    %v605 = stablehlo.add %v604, %v596 : tensor<32x128x7x7xf32>
    %v606 = stablehlo.rsqrt %v605 : tensor<32x128x7x7xf32>
    %v607 = stablehlo.multiply %v600, %v606 : tensor<32x128x7x7xf32>
    %v608 = stablehlo.broadcast_in_dim %gh, dims = [1] : (tensor<128xf32>) -> tensor<32x128x7x7xf32>
    %v609 = stablehlo.multiply %v608, %v592 : tensor<32x128x7x7xf32>
    %v610 = stablehlo.reduce(%v609 init: %v594) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x7x7xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v611 = stablehlo.broadcast_in_dim %v610, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x7x7xf32>
    %v612 = stablehlo.multiply %v607, %v609 : tensor<32x128x7x7xf32>
    %v613 = stablehlo.reduce(%v612 init: %v594) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x7x7xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v614 = stablehlo.broadcast_in_dim %v613, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x7x7xf32>
    %v615 = stablehlo.multiply %v609, %v595 : tensor<32x128x7x7xf32>
    %v616 = stablehlo.subtract %v615, %v611 : tensor<32x128x7x7xf32>
    %v617 = stablehlo.multiply %v607, %v614 : tensor<32x128x7x7xf32>
    %v618 = stablehlo.subtract %v616, %v617 : tensor<32x128x7x7xf32>
    %v619 = stablehlo.divide %v606, %v595 : tensor<32x128x7x7xf32>
    %v620 = stablehlo.multiply %v619, %v618 : tensor<32x128x7x7xf32>
    %v621 = stablehlo.reshape %v620 : (tensor<32x128x7x7xf32>) -> tensor<32x6272xf32>
    %v622 = stablehlo.reshape %v621 : (tensor<32x6272xf32>) -> tensor<32x128x7x7xf32>
    %v623 = stablehlo.transpose %Wh, dims = [1, 0, 2, 3] : (tensor<128x64x1x1xf32>) -> tensor<64x128x1x1xf32>
    %v624 = stablehlo.reverse %v623, dims = [2, 3] : tensor<64x128x1x1xf32>
    %v625 = stablehlo.convolution(%v622, %v624)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x7x7xf32>, tensor<64x128x1x1xf32>) -> tensor<32x64x7x7xf32>
    %v626 = stablehlo.reshape %v625 : (tensor<32x64x7x7xf32>) -> tensor<32x3136xf32>
    %v627 = stablehlo.reshape %v528 : (tensor<32x3136xf32>) -> tensor<32x64x7x7xf32>
    %v628 = stablehlo.reshape %v621 : (tensor<32x6272xf32>) -> tensor<32x128x7x7xf32>
    %v629 = stablehlo.transpose %v627, dims = [1, 0, 2, 3] : (tensor<32x64x7x7xf32>) -> tensor<64x32x7x7xf32>
    %v630 = stablehlo.transpose %v628, dims = [1, 0, 2, 3] : (tensor<32x128x7x7xf32>) -> tensor<128x32x7x7xf32>
    %v631 = stablehlo.convolution(%v629, %v630)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x7x7xf32>, tensor<128x32x7x7xf32>) -> tensor<64x128x1x1xf32>
    %v632 = stablehlo.transpose %v631, dims = [1, 0, 2, 3] : (tensor<64x128x1x1xf32>) -> tensor<128x64x1x1xf32>
    %v633 = stablehlo.constant dense<0.3> : tensor<128x64x1x1xf32>
    %v634 = stablehlo.multiply %v632, %v633 : tensor<128x64x1x1xf32>
    %v635 = stablehlo.subtract %Wh, %v634 : tensor<128x64x1x1xf32>
    %v636 = stablehlo.reshape %v621 : (tensor<32x6272xf32>) -> tensor<32x128x7x7xf32>
    %v637 = stablehlo.constant dense<0.0> : tensor<f32>
    %v638 = stablehlo.reduce(%v636 init: %v637) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x7x7xf32>, tensor<f32>) -> tensor<128xf32>
    %v639 = stablehlo.constant dense<0.3> : tensor<128xf32>
    %v640 = stablehlo.multiply %v638, %v639 : tensor<128xf32>
    %v641 = stablehlo.subtract %bh, %v640 : tensor<128xf32>
    %v642 = stablehlo.constant dense<0.0> : tensor<f32>
    %v643 = stablehlo.reshape %v533 : (tensor<32x6272xf32>) -> tensor<32x128x7x7xf32>
    %v644 = stablehlo.constant dense<49.0> : tensor<32x128x7x7xf32>
    %v645 = stablehlo.constant dense<1.0e-5> : tensor<32x128x7x7xf32>
    %v646 = stablehlo.reduce(%v643 init: %v642) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x7x7xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v647 = stablehlo.broadcast_in_dim %v646, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x7x7xf32>
    %v648 = stablehlo.divide %v647, %v644 : tensor<32x128x7x7xf32>
    %v649 = stablehlo.subtract %v643, %v648 : tensor<32x128x7x7xf32>
    %v650 = stablehlo.multiply %v649, %v649 : tensor<32x128x7x7xf32>
    %v651 = stablehlo.reduce(%v650 init: %v642) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x7x7xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v652 = stablehlo.broadcast_in_dim %v651, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x7x7xf32>
    %v653 = stablehlo.divide %v652, %v644 : tensor<32x128x7x7xf32>
    %v654 = stablehlo.add %v653, %v645 : tensor<32x128x7x7xf32>
    %v655 = stablehlo.rsqrt %v654 : tensor<32x128x7x7xf32>
    %v656 = stablehlo.multiply %v649, %v655 : tensor<32x128x7x7xf32>
    %v657 = stablehlo.reshape %v591 : (tensor<32x6272xf32>) -> tensor<32x128x7x7xf32>
    %v658 = stablehlo.multiply %v657, %v656 : tensor<32x128x7x7xf32>
    %v659 = stablehlo.reduce(%v658 init: %v642) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x7x7xf32>, tensor<f32>) -> tensor<128xf32>
    %v660 = stablehlo.constant dense<0.3> : tensor<128xf32>
    %v661 = stablehlo.multiply %v659, %v660 : tensor<128xf32>
    %v662 = stablehlo.subtract %gh, %v661 : tensor<128xf32>
    %v663 = stablehlo.constant dense<0.0> : tensor<f32>
    %v664 = stablehlo.reshape %v591 : (tensor<32x6272xf32>) -> tensor<32x128x7x7xf32>
    %v665 = stablehlo.reduce(%v664 init: %v663) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x7x7xf32>, tensor<f32>) -> tensor<128xf32>
    %v666 = stablehlo.constant dense<0.3> : tensor<128xf32>
    %v667 = stablehlo.multiply %v665, %v666 : tensor<128xf32>
    %v668 = stablehlo.subtract %bth, %v667 : tensor<128xf32>
    %v669 = stablehlo.reshape %v626 : (tensor<32x3136xf32>) -> tensor<32x64x7x7xf32>
    %v670 = stablehlo.reshape %v508 : (tensor<32x3136xf32>) -> tensor<32x64x7x7xf32>
    %v671 = stablehlo.constant dense<0.0> : tensor<f32>
    %v672 = stablehlo.constant dense<49.0> : tensor<32x64x7x7xf32>
    %v673 = stablehlo.constant dense<1.0e-5> : tensor<32x64x7x7xf32>
    %v674 = stablehlo.reduce(%v670 init: %v671) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x7x7xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v675 = stablehlo.broadcast_in_dim %v674, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x7x7xf32>
    %v676 = stablehlo.divide %v675, %v672 : tensor<32x64x7x7xf32>
    %v677 = stablehlo.subtract %v670, %v676 : tensor<32x64x7x7xf32>
    %v678 = stablehlo.multiply %v677, %v677 : tensor<32x64x7x7xf32>
    %v679 = stablehlo.reduce(%v678 init: %v671) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x7x7xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v680 = stablehlo.broadcast_in_dim %v679, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x7x7xf32>
    %v681 = stablehlo.divide %v680, %v672 : tensor<32x64x7x7xf32>
    %v682 = stablehlo.add %v681, %v673 : tensor<32x64x7x7xf32>
    %v683 = stablehlo.rsqrt %v682 : tensor<32x64x7x7xf32>
    %v684 = stablehlo.multiply %v677, %v683 : tensor<32x64x7x7xf32>
    %v685 = stablehlo.broadcast_in_dim %gp6, dims = [1] : (tensor<64xf32>) -> tensor<32x64x7x7xf32>
    %v686 = stablehlo.multiply %v685, %v669 : tensor<32x64x7x7xf32>
    %v687 = stablehlo.reduce(%v686 init: %v671) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x7x7xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v688 = stablehlo.broadcast_in_dim %v687, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x7x7xf32>
    %v689 = stablehlo.multiply %v684, %v686 : tensor<32x64x7x7xf32>
    %v690 = stablehlo.reduce(%v689 init: %v671) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x7x7xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v691 = stablehlo.broadcast_in_dim %v690, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x7x7xf32>
    %v692 = stablehlo.multiply %v686, %v672 : tensor<32x64x7x7xf32>
    %v693 = stablehlo.subtract %v692, %v688 : tensor<32x64x7x7xf32>
    %v694 = stablehlo.multiply %v684, %v691 : tensor<32x64x7x7xf32>
    %v695 = stablehlo.subtract %v693, %v694 : tensor<32x64x7x7xf32>
    %v696 = stablehlo.divide %v683, %v672 : tensor<32x64x7x7xf32>
    %v697 = stablehlo.multiply %v696, %v695 : tensor<32x64x7x7xf32>
    %v698 = stablehlo.reshape %v697 : (tensor<32x64x7x7xf32>) -> tensor<32x3136xf32>
    %v699 = stablehlo.reshape %v698 : (tensor<32x3136xf32>) -> tensor<32x64x7x7xf32>
    %v700 = stablehlo.transpose %Wp6, dims = [1, 0, 2, 3] : (tensor<64x256x1x1xf32>) -> tensor<256x64x1x1xf32>
    %v701 = stablehlo.reverse %v700, dims = [2, 3] : tensor<256x64x1x1xf32>
    %v702 = stablehlo.convolution(%v699, %v701)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x7x7xf32>, tensor<256x64x1x1xf32>) -> tensor<32x256x7x7xf32>
    %v703 = stablehlo.reshape %v702 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v704 = stablehlo.constant dense<0.0> : tensor<32x12544xf32>
    %v705 = stablehlo.constant dense<6.0> : tensor<32x12544xf32>
    %v706 = stablehlo.compare GT, %v499, %v704 : (tensor<32x12544xf32>, tensor<32x12544xf32>) -> tensor<32x12544xi1>
    %v707 = stablehlo.compare LT, %v499, %v705 : (tensor<32x12544xf32>, tensor<32x12544xf32>) -> tensor<32x12544xi1>
    %v708 = stablehlo.and %v706, %v707 : tensor<32x12544xi1>
    %v709 = stablehlo.select %v708, %v703, %v704 : tensor<32x12544xi1>, tensor<32x12544xf32>
    %v710 = stablehlo.reshape %v709 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v711 = stablehlo.reshape %v479 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v712 = stablehlo.constant dense<0.0> : tensor<f32>
    %v713 = stablehlo.constant dense<49.0> : tensor<32x256x7x7xf32>
    %v714 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v715 = stablehlo.reduce(%v711 init: %v712) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v716 = stablehlo.broadcast_in_dim %v715, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x7x7xf32>
    %v717 = stablehlo.divide %v716, %v713 : tensor<32x256x7x7xf32>
    %v718 = stablehlo.subtract %v711, %v717 : tensor<32x256x7x7xf32>
    %v719 = stablehlo.multiply %v718, %v718 : tensor<32x256x7x7xf32>
    %v720 = stablehlo.reduce(%v719 init: %v712) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v721 = stablehlo.broadcast_in_dim %v720, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x7x7xf32>
    %v722 = stablehlo.divide %v721, %v713 : tensor<32x256x7x7xf32>
    %v723 = stablehlo.add %v722, %v714 : tensor<32x256x7x7xf32>
    %v724 = stablehlo.rsqrt %v723 : tensor<32x256x7x7xf32>
    %v725 = stablehlo.multiply %v718, %v724 : tensor<32x256x7x7xf32>
    %v726 = stablehlo.broadcast_in_dim %gd6, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v727 = stablehlo.multiply %v726, %v710 : tensor<32x256x7x7xf32>
    %v728 = stablehlo.reduce(%v727 init: %v712) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v729 = stablehlo.broadcast_in_dim %v728, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x7x7xf32>
    %v730 = stablehlo.multiply %v725, %v727 : tensor<32x256x7x7xf32>
    %v731 = stablehlo.reduce(%v730 init: %v712) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v732 = stablehlo.broadcast_in_dim %v731, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x7x7xf32>
    %v733 = stablehlo.multiply %v727, %v713 : tensor<32x256x7x7xf32>
    %v734 = stablehlo.subtract %v733, %v729 : tensor<32x256x7x7xf32>
    %v735 = stablehlo.multiply %v725, %v732 : tensor<32x256x7x7xf32>
    %v736 = stablehlo.subtract %v734, %v735 : tensor<32x256x7x7xf32>
    %v737 = stablehlo.divide %v724, %v713 : tensor<32x256x7x7xf32>
    %v738 = stablehlo.multiply %v737, %v736 : tensor<32x256x7x7xf32>
    %v739 = stablehlo.reshape %v738 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v740 = stablehlo.reshape %v739 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v741 = stablehlo.constant dense<0.0> : tensor<f32>
    %v742 = stablehlo.pad %v740, %v741, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<32x256x14x14xf32>
    %v743 = stablehlo.reverse %Wd6, dims = [2, 3] : tensor<256x1x3x3xf32>
    %v744 = stablehlo.convolution(%v742, %v743)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 256 : i64} : (tensor<32x256x14x14xf32>, tensor<256x1x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v745 = stablehlo.reshape %v744 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v746 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v747 = stablehlo.constant dense<6.0> : tensor<32x50176xf32>
    %v748 = stablehlo.compare GT, %v470, %v746 : (tensor<32x50176xf32>, tensor<32x50176xf32>) -> tensor<32x50176xi1>
    %v749 = stablehlo.compare LT, %v470, %v747 : (tensor<32x50176xf32>, tensor<32x50176xf32>) -> tensor<32x50176xi1>
    %v750 = stablehlo.and %v748, %v749 : tensor<32x50176xi1>
    %v751 = stablehlo.select %v750, %v745, %v746 : tensor<32x50176xi1>, tensor<32x50176xf32>
    %v752 = stablehlo.reshape %v751 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v753 = stablehlo.reshape %v450 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v754 = stablehlo.constant dense<0.0> : tensor<f32>
    %v755 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v756 = stablehlo.constant dense<1.0e-5> : tensor<32x256x14x14xf32>
    %v757 = stablehlo.reduce(%v753 init: %v754) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v758 = stablehlo.broadcast_in_dim %v757, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v759 = stablehlo.divide %v758, %v755 : tensor<32x256x14x14xf32>
    %v760 = stablehlo.subtract %v753, %v759 : tensor<32x256x14x14xf32>
    %v761 = stablehlo.multiply %v760, %v760 : tensor<32x256x14x14xf32>
    %v762 = stablehlo.reduce(%v761 init: %v754) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v763 = stablehlo.broadcast_in_dim %v762, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v764 = stablehlo.divide %v763, %v755 : tensor<32x256x14x14xf32>
    %v765 = stablehlo.add %v764, %v756 : tensor<32x256x14x14xf32>
    %v766 = stablehlo.rsqrt %v765 : tensor<32x256x14x14xf32>
    %v767 = stablehlo.multiply %v760, %v766 : tensor<32x256x14x14xf32>
    %v768 = stablehlo.broadcast_in_dim %ge6, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v769 = stablehlo.multiply %v768, %v752 : tensor<32x256x14x14xf32>
    %v770 = stablehlo.reduce(%v769 init: %v754) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v771 = stablehlo.broadcast_in_dim %v770, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v772 = stablehlo.multiply %v767, %v769 : tensor<32x256x14x14xf32>
    %v773 = stablehlo.reduce(%v772 init: %v754) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v774 = stablehlo.broadcast_in_dim %v773, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v775 = stablehlo.multiply %v769, %v755 : tensor<32x256x14x14xf32>
    %v776 = stablehlo.subtract %v775, %v771 : tensor<32x256x14x14xf32>
    %v777 = stablehlo.multiply %v767, %v774 : tensor<32x256x14x14xf32>
    %v778 = stablehlo.subtract %v776, %v777 : tensor<32x256x14x14xf32>
    %v779 = stablehlo.divide %v766, %v755 : tensor<32x256x14x14xf32>
    %v780 = stablehlo.multiply %v779, %v778 : tensor<32x256x14x14xf32>
    %v781 = stablehlo.reshape %v780 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v782 = stablehlo.reshape %v781 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v783 = stablehlo.transpose %We6, dims = [1, 0, 2, 3] : (tensor<256x64x1x1xf32>) -> tensor<64x256x1x1xf32>
    %v784 = stablehlo.reverse %v783, dims = [2, 3] : tensor<64x256x1x1xf32>
    %v785 = stablehlo.convolution(%v782, %v784)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<64x256x1x1xf32>) -> tensor<32x64x14x14xf32>
    %v786 = stablehlo.reshape %v785 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v787 = stablehlo.reshape %v445 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v788 = stablehlo.reshape %v781 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v789 = stablehlo.transpose %v787, dims = [1, 0, 2, 3] : (tensor<32x64x14x14xf32>) -> tensor<64x32x14x14xf32>
    %v790 = stablehlo.transpose %v788, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v791 = stablehlo.convolution(%v789, %v790)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<64x256x1x1xf32>
    %v792 = stablehlo.transpose %v791, dims = [1, 0, 2, 3] : (tensor<64x256x1x1xf32>) -> tensor<256x64x1x1xf32>
    %v793 = stablehlo.constant dense<0.3> : tensor<256x64x1x1xf32>
    %v794 = stablehlo.multiply %v792, %v793 : tensor<256x64x1x1xf32>
    %v795 = stablehlo.subtract %We6, %v794 : tensor<256x64x1x1xf32>
    %v796 = stablehlo.reshape %v781 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v797 = stablehlo.constant dense<0.0> : tensor<f32>
    %v798 = stablehlo.reduce(%v796 init: %v797) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v799 = stablehlo.constant dense<0.3> : tensor<256xf32>
    %v800 = stablehlo.multiply %v798, %v799 : tensor<256xf32>
    %v801 = stablehlo.subtract %be6, %v800 : tensor<256xf32>
    %v802 = stablehlo.constant dense<0.0> : tensor<f32>
    %v803 = stablehlo.reshape %v450 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v804 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v805 = stablehlo.constant dense<1.0e-5> : tensor<32x256x14x14xf32>
    %v806 = stablehlo.reduce(%v803 init: %v802) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v807 = stablehlo.broadcast_in_dim %v806, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v808 = stablehlo.divide %v807, %v804 : tensor<32x256x14x14xf32>
    %v809 = stablehlo.subtract %v803, %v808 : tensor<32x256x14x14xf32>
    %v810 = stablehlo.multiply %v809, %v809 : tensor<32x256x14x14xf32>
    %v811 = stablehlo.reduce(%v810 init: %v802) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v812 = stablehlo.broadcast_in_dim %v811, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v813 = stablehlo.divide %v812, %v804 : tensor<32x256x14x14xf32>
    %v814 = stablehlo.add %v813, %v805 : tensor<32x256x14x14xf32>
    %v815 = stablehlo.rsqrt %v814 : tensor<32x256x14x14xf32>
    %v816 = stablehlo.multiply %v809, %v815 : tensor<32x256x14x14xf32>
    %v817 = stablehlo.reshape %v751 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v818 = stablehlo.multiply %v817, %v816 : tensor<32x256x14x14xf32>
    %v819 = stablehlo.reduce(%v818 init: %v802) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v820 = stablehlo.constant dense<0.3> : tensor<256xf32>
    %v821 = stablehlo.multiply %v819, %v820 : tensor<256xf32>
    %v822 = stablehlo.subtract %ge6, %v821 : tensor<256xf32>
    %v823 = stablehlo.constant dense<0.0> : tensor<f32>
    %v824 = stablehlo.reshape %v751 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v825 = stablehlo.reduce(%v824 init: %v823) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v826 = stablehlo.constant dense<0.3> : tensor<256xf32>
    %v827 = stablehlo.multiply %v825, %v826 : tensor<256xf32>
    %v828 = stablehlo.subtract %bte6, %v827 : tensor<256xf32>
    %v829 = stablehlo.reshape %v474 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v830 = stablehlo.reshape %v739 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v831 = stablehlo.constant dense<0.0> : tensor<f32>
    %v832 = stablehlo.pad %v830, %v831, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<32x256x14x14xf32>
    %v833 = stablehlo.transpose %v829, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v834 = stablehlo.transpose %v832, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v835 = stablehlo.convolution(%v833, %v834)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 256 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<1x256x3x3xf32>
    %v836 = stablehlo.reshape %v835 : (tensor<1x256x3x3xf32>) -> tensor<256x1x3x3xf32>
    %v837 = stablehlo.constant dense<0.3> : tensor<256x1x3x3xf32>
    %v838 = stablehlo.multiply %v836, %v837 : tensor<256x1x3x3xf32>
    %v839 = stablehlo.subtract %Wd6, %v838 : tensor<256x1x3x3xf32>
    %v840 = stablehlo.reshape %v739 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v841 = stablehlo.constant dense<0.0> : tensor<f32>
    %v842 = stablehlo.reduce(%v840 init: %v841) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v843 = stablehlo.constant dense<0.3> : tensor<256xf32>
    %v844 = stablehlo.multiply %v842, %v843 : tensor<256xf32>
    %v845 = stablehlo.subtract %bd6, %v844 : tensor<256xf32>
    %v846 = stablehlo.constant dense<0.0> : tensor<f32>
    %v847 = stablehlo.reshape %v479 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v848 = stablehlo.constant dense<49.0> : tensor<32x256x7x7xf32>
    %v849 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v850 = stablehlo.reduce(%v847 init: %v846) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v851 = stablehlo.broadcast_in_dim %v850, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x7x7xf32>
    %v852 = stablehlo.divide %v851, %v848 : tensor<32x256x7x7xf32>
    %v853 = stablehlo.subtract %v847, %v852 : tensor<32x256x7x7xf32>
    %v854 = stablehlo.multiply %v853, %v853 : tensor<32x256x7x7xf32>
    %v855 = stablehlo.reduce(%v854 init: %v846) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v856 = stablehlo.broadcast_in_dim %v855, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x7x7xf32>
    %v857 = stablehlo.divide %v856, %v848 : tensor<32x256x7x7xf32>
    %v858 = stablehlo.add %v857, %v849 : tensor<32x256x7x7xf32>
    %v859 = stablehlo.rsqrt %v858 : tensor<32x256x7x7xf32>
    %v860 = stablehlo.multiply %v853, %v859 : tensor<32x256x7x7xf32>
    %v861 = stablehlo.reshape %v709 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v862 = stablehlo.multiply %v861, %v860 : tensor<32x256x7x7xf32>
    %v863 = stablehlo.reduce(%v862 init: %v846) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v864 = stablehlo.constant dense<0.3> : tensor<256xf32>
    %v865 = stablehlo.multiply %v863, %v864 : tensor<256xf32>
    %v866 = stablehlo.subtract %gd6, %v865 : tensor<256xf32>
    %v867 = stablehlo.constant dense<0.0> : tensor<f32>
    %v868 = stablehlo.reshape %v709 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v869 = stablehlo.reduce(%v868 init: %v867) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v870 = stablehlo.constant dense<0.3> : tensor<256xf32>
    %v871 = stablehlo.multiply %v869, %v870 : tensor<256xf32>
    %v872 = stablehlo.subtract %btd6, %v871 : tensor<256xf32>
    %v873 = stablehlo.reshape %v503 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v874 = stablehlo.reshape %v698 : (tensor<32x3136xf32>) -> tensor<32x64x7x7xf32>
    %v875 = stablehlo.transpose %v873, dims = [1, 0, 2, 3] : (tensor<32x256x7x7xf32>) -> tensor<256x32x7x7xf32>
    %v876 = stablehlo.transpose %v874, dims = [1, 0, 2, 3] : (tensor<32x64x7x7xf32>) -> tensor<64x32x7x7xf32>
    %v877 = stablehlo.convolution(%v875, %v876)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x7x7xf32>, tensor<64x32x7x7xf32>) -> tensor<256x64x1x1xf32>
    %v878 = stablehlo.transpose %v877, dims = [1, 0, 2, 3] : (tensor<256x64x1x1xf32>) -> tensor<64x256x1x1xf32>
    %v879 = stablehlo.constant dense<0.3> : tensor<64x256x1x1xf32>
    %v880 = stablehlo.multiply %v878, %v879 : tensor<64x256x1x1xf32>
    %v881 = stablehlo.subtract %Wp6, %v880 : tensor<64x256x1x1xf32>
    %v882 = stablehlo.reshape %v698 : (tensor<32x3136xf32>) -> tensor<32x64x7x7xf32>
    %v883 = stablehlo.constant dense<0.0> : tensor<f32>
    %v884 = stablehlo.reduce(%v882 init: %v883) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x7x7xf32>, tensor<f32>) -> tensor<64xf32>
    %v885 = stablehlo.constant dense<0.3> : tensor<64xf32>
    %v886 = stablehlo.multiply %v884, %v885 : tensor<64xf32>
    %v887 = stablehlo.subtract %bp6, %v886 : tensor<64xf32>
    %v888 = stablehlo.constant dense<0.0> : tensor<f32>
    %v889 = stablehlo.reshape %v508 : (tensor<32x3136xf32>) -> tensor<32x64x7x7xf32>
    %v890 = stablehlo.constant dense<49.0> : tensor<32x64x7x7xf32>
    %v891 = stablehlo.constant dense<1.0e-5> : tensor<32x64x7x7xf32>
    %v892 = stablehlo.reduce(%v889 init: %v888) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x7x7xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v893 = stablehlo.broadcast_in_dim %v892, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x7x7xf32>
    %v894 = stablehlo.divide %v893, %v890 : tensor<32x64x7x7xf32>
    %v895 = stablehlo.subtract %v889, %v894 : tensor<32x64x7x7xf32>
    %v896 = stablehlo.multiply %v895, %v895 : tensor<32x64x7x7xf32>
    %v897 = stablehlo.reduce(%v896 init: %v888) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x7x7xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v898 = stablehlo.broadcast_in_dim %v897, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x7x7xf32>
    %v899 = stablehlo.divide %v898, %v890 : tensor<32x64x7x7xf32>
    %v900 = stablehlo.add %v899, %v891 : tensor<32x64x7x7xf32>
    %v901 = stablehlo.rsqrt %v900 : tensor<32x64x7x7xf32>
    %v902 = stablehlo.multiply %v895, %v901 : tensor<32x64x7x7xf32>
    %v903 = stablehlo.reshape %v626 : (tensor<32x3136xf32>) -> tensor<32x64x7x7xf32>
    %v904 = stablehlo.multiply %v903, %v902 : tensor<32x64x7x7xf32>
    %v905 = stablehlo.reduce(%v904 init: %v888) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x7x7xf32>, tensor<f32>) -> tensor<64xf32>
    %v906 = stablehlo.constant dense<0.3> : tensor<64xf32>
    %v907 = stablehlo.multiply %v905, %v906 : tensor<64xf32>
    %v908 = stablehlo.subtract %gp6, %v907 : tensor<64xf32>
    %v909 = stablehlo.constant dense<0.0> : tensor<f32>
    %v910 = stablehlo.reshape %v626 : (tensor<32x3136xf32>) -> tensor<32x64x7x7xf32>
    %v911 = stablehlo.reduce(%v910 init: %v909) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x7x7xf32>, tensor<f32>) -> tensor<64xf32>
    %v912 = stablehlo.constant dense<0.3> : tensor<64xf32>
    %v913 = stablehlo.multiply %v911, %v912 : tensor<64xf32>
    %v914 = stablehlo.subtract %btp6, %v913 : tensor<64xf32>
    %v915 = stablehlo.reshape %v786 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v916 = stablehlo.reshape %v425 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v917 = stablehlo.constant dense<0.0> : tensor<f32>
    %v918 = stablehlo.constant dense<196.0> : tensor<32x64x14x14xf32>
    %v919 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v920 = stablehlo.reduce(%v916 init: %v917) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v921 = stablehlo.broadcast_in_dim %v920, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v922 = stablehlo.divide %v921, %v918 : tensor<32x64x14x14xf32>
    %v923 = stablehlo.subtract %v916, %v922 : tensor<32x64x14x14xf32>
    %v924 = stablehlo.multiply %v923, %v923 : tensor<32x64x14x14xf32>
    %v925 = stablehlo.reduce(%v924 init: %v917) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v926 = stablehlo.broadcast_in_dim %v925, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v927 = stablehlo.divide %v926, %v918 : tensor<32x64x14x14xf32>
    %v928 = stablehlo.add %v927, %v919 : tensor<32x64x14x14xf32>
    %v929 = stablehlo.rsqrt %v928 : tensor<32x64x14x14xf32>
    %v930 = stablehlo.multiply %v923, %v929 : tensor<32x64x14x14xf32>
    %v931 = stablehlo.broadcast_in_dim %gp5, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v932 = stablehlo.multiply %v931, %v915 : tensor<32x64x14x14xf32>
    %v933 = stablehlo.reduce(%v932 init: %v917) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v934 = stablehlo.broadcast_in_dim %v933, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v935 = stablehlo.multiply %v930, %v932 : tensor<32x64x14x14xf32>
    %v936 = stablehlo.reduce(%v935 init: %v917) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v937 = stablehlo.broadcast_in_dim %v936, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v938 = stablehlo.multiply %v932, %v918 : tensor<32x64x14x14xf32>
    %v939 = stablehlo.subtract %v938, %v934 : tensor<32x64x14x14xf32>
    %v940 = stablehlo.multiply %v930, %v937 : tensor<32x64x14x14xf32>
    %v941 = stablehlo.subtract %v939, %v940 : tensor<32x64x14x14xf32>
    %v942 = stablehlo.divide %v929, %v918 : tensor<32x64x14x14xf32>
    %v943 = stablehlo.multiply %v942, %v941 : tensor<32x64x14x14xf32>
    %v944 = stablehlo.reshape %v943 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v945 = stablehlo.reshape %v944 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v946 = stablehlo.transpose %Wp5, dims = [1, 0, 2, 3] : (tensor<64x128x1x1xf32>) -> tensor<128x64x1x1xf32>
    %v947 = stablehlo.reverse %v946, dims = [2, 3] : tensor<128x64x1x1xf32>
    %v948 = stablehlo.convolution(%v945, %v947)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x14x14xf32>, tensor<128x64x1x1xf32>) -> tensor<32x128x14x14xf32>
    %v949 = stablehlo.reshape %v948 : (tensor<32x128x14x14xf32>) -> tensor<32x25088xf32>
    %v950 = stablehlo.constant dense<0.0> : tensor<32x25088xf32>
    %v951 = stablehlo.constant dense<6.0> : tensor<32x25088xf32>
    %v952 = stablehlo.compare GT, %v416, %v950 : (tensor<32x25088xf32>, tensor<32x25088xf32>) -> tensor<32x25088xi1>
    %v953 = stablehlo.compare LT, %v416, %v951 : (tensor<32x25088xf32>, tensor<32x25088xf32>) -> tensor<32x25088xi1>
    %v954 = stablehlo.and %v952, %v953 : tensor<32x25088xi1>
    %v955 = stablehlo.select %v954, %v949, %v950 : tensor<32x25088xi1>, tensor<32x25088xf32>
    %v956 = stablehlo.reshape %v955 : (tensor<32x25088xf32>) -> tensor<32x128x14x14xf32>
    %v957 = stablehlo.reshape %v396 : (tensor<32x25088xf32>) -> tensor<32x128x14x14xf32>
    %v958 = stablehlo.constant dense<0.0> : tensor<f32>
    %v959 = stablehlo.constant dense<196.0> : tensor<32x128x14x14xf32>
    %v960 = stablehlo.constant dense<1.0e-5> : tensor<32x128x14x14xf32>
    %v961 = stablehlo.reduce(%v957 init: %v958) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x14x14xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v962 = stablehlo.broadcast_in_dim %v961, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x14x14xf32>
    %v963 = stablehlo.divide %v962, %v959 : tensor<32x128x14x14xf32>
    %v964 = stablehlo.subtract %v957, %v963 : tensor<32x128x14x14xf32>
    %v965 = stablehlo.multiply %v964, %v964 : tensor<32x128x14x14xf32>
    %v966 = stablehlo.reduce(%v965 init: %v958) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x14x14xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v967 = stablehlo.broadcast_in_dim %v966, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x14x14xf32>
    %v968 = stablehlo.divide %v967, %v959 : tensor<32x128x14x14xf32>
    %v969 = stablehlo.add %v968, %v960 : tensor<32x128x14x14xf32>
    %v970 = stablehlo.rsqrt %v969 : tensor<32x128x14x14xf32>
    %v971 = stablehlo.multiply %v964, %v970 : tensor<32x128x14x14xf32>
    %v972 = stablehlo.broadcast_in_dim %gd5, dims = [1] : (tensor<128xf32>) -> tensor<32x128x14x14xf32>
    %v973 = stablehlo.multiply %v972, %v956 : tensor<32x128x14x14xf32>
    %v974 = stablehlo.reduce(%v973 init: %v958) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x14x14xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v975 = stablehlo.broadcast_in_dim %v974, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x14x14xf32>
    %v976 = stablehlo.multiply %v971, %v973 : tensor<32x128x14x14xf32>
    %v977 = stablehlo.reduce(%v976 init: %v958) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x14x14xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v978 = stablehlo.broadcast_in_dim %v977, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x14x14xf32>
    %v979 = stablehlo.multiply %v973, %v959 : tensor<32x128x14x14xf32>
    %v980 = stablehlo.subtract %v979, %v975 : tensor<32x128x14x14xf32>
    %v981 = stablehlo.multiply %v971, %v978 : tensor<32x128x14x14xf32>
    %v982 = stablehlo.subtract %v980, %v981 : tensor<32x128x14x14xf32>
    %v983 = stablehlo.divide %v970, %v959 : tensor<32x128x14x14xf32>
    %v984 = stablehlo.multiply %v983, %v982 : tensor<32x128x14x14xf32>
    %v985 = stablehlo.reshape %v984 : (tensor<32x128x14x14xf32>) -> tensor<32x25088xf32>
    %v986 = stablehlo.reshape %v985 : (tensor<32x25088xf32>) -> tensor<32x128x14x14xf32>
    %v987 = stablehlo.constant dense<0.0> : tensor<f32>
    %v988 = stablehlo.pad %v986, %v987, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x128x14x14xf32>, tensor<f32>) -> tensor<32x128x28x28xf32>
    %v989 = stablehlo.reverse %Wd5, dims = [2, 3] : tensor<128x1x3x3xf32>
    %v990 = stablehlo.convolution(%v988, %v989)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 128 : i64} : (tensor<32x128x28x28xf32>, tensor<128x1x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v991 = stablehlo.reshape %v990 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v992 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v993 = stablehlo.constant dense<6.0> : tensor<32x100352xf32>
    %v994 = stablehlo.compare GT, %v387, %v992 : (tensor<32x100352xf32>, tensor<32x100352xf32>) -> tensor<32x100352xi1>
    %v995 = stablehlo.compare LT, %v387, %v993 : (tensor<32x100352xf32>, tensor<32x100352xf32>) -> tensor<32x100352xi1>
    %v996 = stablehlo.and %v994, %v995 : tensor<32x100352xi1>
    %v997 = stablehlo.select %v996, %v991, %v992 : tensor<32x100352xi1>, tensor<32x100352xf32>
    %v998 = stablehlo.reshape %v997 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v999 = stablehlo.reshape %v367 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v1000 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1001 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v1002 = stablehlo.constant dense<1.0e-5> : tensor<32x128x28x28xf32>
    %v1003 = stablehlo.reduce(%v999 init: %v1000) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v1004 = stablehlo.broadcast_in_dim %v1003, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v1005 = stablehlo.divide %v1004, %v1001 : tensor<32x128x28x28xf32>
    %v1006 = stablehlo.subtract %v999, %v1005 : tensor<32x128x28x28xf32>
    %v1007 = stablehlo.multiply %v1006, %v1006 : tensor<32x128x28x28xf32>
    %v1008 = stablehlo.reduce(%v1007 init: %v1000) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v1009 = stablehlo.broadcast_in_dim %v1008, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v1010 = stablehlo.divide %v1009, %v1001 : tensor<32x128x28x28xf32>
    %v1011 = stablehlo.add %v1010, %v1002 : tensor<32x128x28x28xf32>
    %v1012 = stablehlo.rsqrt %v1011 : tensor<32x128x28x28xf32>
    %v1013 = stablehlo.multiply %v1006, %v1012 : tensor<32x128x28x28xf32>
    %v1014 = stablehlo.broadcast_in_dim %ge5, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v1015 = stablehlo.multiply %v1014, %v998 : tensor<32x128x28x28xf32>
    %v1016 = stablehlo.reduce(%v1015 init: %v1000) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v1017 = stablehlo.broadcast_in_dim %v1016, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v1018 = stablehlo.multiply %v1013, %v1015 : tensor<32x128x28x28xf32>
    %v1019 = stablehlo.reduce(%v1018 init: %v1000) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v1020 = stablehlo.broadcast_in_dim %v1019, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v1021 = stablehlo.multiply %v1015, %v1001 : tensor<32x128x28x28xf32>
    %v1022 = stablehlo.subtract %v1021, %v1017 : tensor<32x128x28x28xf32>
    %v1023 = stablehlo.multiply %v1013, %v1020 : tensor<32x128x28x28xf32>
    %v1024 = stablehlo.subtract %v1022, %v1023 : tensor<32x128x28x28xf32>
    %v1025 = stablehlo.divide %v1012, %v1001 : tensor<32x128x28x28xf32>
    %v1026 = stablehlo.multiply %v1025, %v1024 : tensor<32x128x28x28xf32>
    %v1027 = stablehlo.reshape %v1026 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v1028 = stablehlo.reshape %v1027 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v1029 = stablehlo.transpose %We5, dims = [1, 0, 2, 3] : (tensor<128x32x1x1xf32>) -> tensor<32x128x1x1xf32>
    %v1030 = stablehlo.reverse %v1029, dims = [2, 3] : tensor<32x128x1x1xf32>
    %v1031 = stablehlo.convolution(%v1028, %v1030)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<32x128x1x1xf32>) -> tensor<32x32x28x28xf32>
    %v1032 = stablehlo.reshape %v1031 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v1033 = stablehlo.reshape %v362 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v1034 = stablehlo.reshape %v1027 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v1035 = stablehlo.transpose %v1033, dims = [1, 0, 2, 3] : (tensor<32x32x28x28xf32>) -> tensor<32x32x28x28xf32>
    %v1036 = stablehlo.transpose %v1034, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v1037 = stablehlo.convolution(%v1035, %v1036)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<32x128x1x1xf32>
    %v1038 = stablehlo.transpose %v1037, dims = [1, 0, 2, 3] : (tensor<32x128x1x1xf32>) -> tensor<128x32x1x1xf32>
    %v1039 = stablehlo.constant dense<0.3> : tensor<128x32x1x1xf32>
    %v1040 = stablehlo.multiply %v1038, %v1039 : tensor<128x32x1x1xf32>
    %v1041 = stablehlo.subtract %We5, %v1040 : tensor<128x32x1x1xf32>
    %v1042 = stablehlo.reshape %v1027 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v1043 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1044 = stablehlo.reduce(%v1042 init: %v1043) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v1045 = stablehlo.constant dense<0.3> : tensor<128xf32>
    %v1046 = stablehlo.multiply %v1044, %v1045 : tensor<128xf32>
    %v1047 = stablehlo.subtract %be5, %v1046 : tensor<128xf32>
    %v1048 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1049 = stablehlo.reshape %v367 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v1050 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v1051 = stablehlo.constant dense<1.0e-5> : tensor<32x128x28x28xf32>
    %v1052 = stablehlo.reduce(%v1049 init: %v1048) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v1053 = stablehlo.broadcast_in_dim %v1052, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v1054 = stablehlo.divide %v1053, %v1050 : tensor<32x128x28x28xf32>
    %v1055 = stablehlo.subtract %v1049, %v1054 : tensor<32x128x28x28xf32>
    %v1056 = stablehlo.multiply %v1055, %v1055 : tensor<32x128x28x28xf32>
    %v1057 = stablehlo.reduce(%v1056 init: %v1048) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v1058 = stablehlo.broadcast_in_dim %v1057, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v1059 = stablehlo.divide %v1058, %v1050 : tensor<32x128x28x28xf32>
    %v1060 = stablehlo.add %v1059, %v1051 : tensor<32x128x28x28xf32>
    %v1061 = stablehlo.rsqrt %v1060 : tensor<32x128x28x28xf32>
    %v1062 = stablehlo.multiply %v1055, %v1061 : tensor<32x128x28x28xf32>
    %v1063 = stablehlo.reshape %v997 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v1064 = stablehlo.multiply %v1063, %v1062 : tensor<32x128x28x28xf32>
    %v1065 = stablehlo.reduce(%v1064 init: %v1048) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v1066 = stablehlo.constant dense<0.3> : tensor<128xf32>
    %v1067 = stablehlo.multiply %v1065, %v1066 : tensor<128xf32>
    %v1068 = stablehlo.subtract %ge5, %v1067 : tensor<128xf32>
    %v1069 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1070 = stablehlo.reshape %v997 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v1071 = stablehlo.reduce(%v1070 init: %v1069) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v1072 = stablehlo.constant dense<0.3> : tensor<128xf32>
    %v1073 = stablehlo.multiply %v1071, %v1072 : tensor<128xf32>
    %v1074 = stablehlo.subtract %bte5, %v1073 : tensor<128xf32>
    %v1075 = stablehlo.reshape %v391 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v1076 = stablehlo.reshape %v985 : (tensor<32x25088xf32>) -> tensor<32x128x14x14xf32>
    %v1077 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1078 = stablehlo.pad %v1076, %v1077, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x128x14x14xf32>, tensor<f32>) -> tensor<32x128x28x28xf32>
    %v1079 = stablehlo.transpose %v1075, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v1080 = stablehlo.transpose %v1078, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v1081 = stablehlo.convolution(%v1079, %v1080)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 128 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<1x128x3x3xf32>
    %v1082 = stablehlo.reshape %v1081 : (tensor<1x128x3x3xf32>) -> tensor<128x1x3x3xf32>
    %v1083 = stablehlo.constant dense<0.3> : tensor<128x1x3x3xf32>
    %v1084 = stablehlo.multiply %v1082, %v1083 : tensor<128x1x3x3xf32>
    %v1085 = stablehlo.subtract %Wd5, %v1084 : tensor<128x1x3x3xf32>
    %v1086 = stablehlo.reshape %v985 : (tensor<32x25088xf32>) -> tensor<32x128x14x14xf32>
    %v1087 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1088 = stablehlo.reduce(%v1086 init: %v1087) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x14x14xf32>, tensor<f32>) -> tensor<128xf32>
    %v1089 = stablehlo.constant dense<0.3> : tensor<128xf32>
    %v1090 = stablehlo.multiply %v1088, %v1089 : tensor<128xf32>
    %v1091 = stablehlo.subtract %bd5, %v1090 : tensor<128xf32>
    %v1092 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1093 = stablehlo.reshape %v396 : (tensor<32x25088xf32>) -> tensor<32x128x14x14xf32>
    %v1094 = stablehlo.constant dense<196.0> : tensor<32x128x14x14xf32>
    %v1095 = stablehlo.constant dense<1.0e-5> : tensor<32x128x14x14xf32>
    %v1096 = stablehlo.reduce(%v1093 init: %v1092) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x14x14xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v1097 = stablehlo.broadcast_in_dim %v1096, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x14x14xf32>
    %v1098 = stablehlo.divide %v1097, %v1094 : tensor<32x128x14x14xf32>
    %v1099 = stablehlo.subtract %v1093, %v1098 : tensor<32x128x14x14xf32>
    %v1100 = stablehlo.multiply %v1099, %v1099 : tensor<32x128x14x14xf32>
    %v1101 = stablehlo.reduce(%v1100 init: %v1092) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x14x14xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v1102 = stablehlo.broadcast_in_dim %v1101, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x14x14xf32>
    %v1103 = stablehlo.divide %v1102, %v1094 : tensor<32x128x14x14xf32>
    %v1104 = stablehlo.add %v1103, %v1095 : tensor<32x128x14x14xf32>
    %v1105 = stablehlo.rsqrt %v1104 : tensor<32x128x14x14xf32>
    %v1106 = stablehlo.multiply %v1099, %v1105 : tensor<32x128x14x14xf32>
    %v1107 = stablehlo.reshape %v955 : (tensor<32x25088xf32>) -> tensor<32x128x14x14xf32>
    %v1108 = stablehlo.multiply %v1107, %v1106 : tensor<32x128x14x14xf32>
    %v1109 = stablehlo.reduce(%v1108 init: %v1092) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x14x14xf32>, tensor<f32>) -> tensor<128xf32>
    %v1110 = stablehlo.constant dense<0.3> : tensor<128xf32>
    %v1111 = stablehlo.multiply %v1109, %v1110 : tensor<128xf32>
    %v1112 = stablehlo.subtract %gd5, %v1111 : tensor<128xf32>
    %v1113 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1114 = stablehlo.reshape %v955 : (tensor<32x25088xf32>) -> tensor<32x128x14x14xf32>
    %v1115 = stablehlo.reduce(%v1114 init: %v1113) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x14x14xf32>, tensor<f32>) -> tensor<128xf32>
    %v1116 = stablehlo.constant dense<0.3> : tensor<128xf32>
    %v1117 = stablehlo.multiply %v1115, %v1116 : tensor<128xf32>
    %v1118 = stablehlo.subtract %btd5, %v1117 : tensor<128xf32>
    %v1119 = stablehlo.reshape %v420 : (tensor<32x25088xf32>) -> tensor<32x128x14x14xf32>
    %v1120 = stablehlo.reshape %v944 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v1121 = stablehlo.transpose %v1119, dims = [1, 0, 2, 3] : (tensor<32x128x14x14xf32>) -> tensor<128x32x14x14xf32>
    %v1122 = stablehlo.transpose %v1120, dims = [1, 0, 2, 3] : (tensor<32x64x14x14xf32>) -> tensor<64x32x14x14xf32>
    %v1123 = stablehlo.convolution(%v1121, %v1122)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x14x14xf32>, tensor<64x32x14x14xf32>) -> tensor<128x64x1x1xf32>
    %v1124 = stablehlo.transpose %v1123, dims = [1, 0, 2, 3] : (tensor<128x64x1x1xf32>) -> tensor<64x128x1x1xf32>
    %v1125 = stablehlo.constant dense<0.3> : tensor<64x128x1x1xf32>
    %v1126 = stablehlo.multiply %v1124, %v1125 : tensor<64x128x1x1xf32>
    %v1127 = stablehlo.subtract %Wp5, %v1126 : tensor<64x128x1x1xf32>
    %v1128 = stablehlo.reshape %v944 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v1129 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1130 = stablehlo.reduce(%v1128 init: %v1129) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v1131 = stablehlo.constant dense<0.3> : tensor<64xf32>
    %v1132 = stablehlo.multiply %v1130, %v1131 : tensor<64xf32>
    %v1133 = stablehlo.subtract %bp5, %v1132 : tensor<64xf32>
    %v1134 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1135 = stablehlo.reshape %v425 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v1136 = stablehlo.constant dense<196.0> : tensor<32x64x14x14xf32>
    %v1137 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v1138 = stablehlo.reduce(%v1135 init: %v1134) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v1139 = stablehlo.broadcast_in_dim %v1138, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v1140 = stablehlo.divide %v1139, %v1136 : tensor<32x64x14x14xf32>
    %v1141 = stablehlo.subtract %v1135, %v1140 : tensor<32x64x14x14xf32>
    %v1142 = stablehlo.multiply %v1141, %v1141 : tensor<32x64x14x14xf32>
    %v1143 = stablehlo.reduce(%v1142 init: %v1134) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v1144 = stablehlo.broadcast_in_dim %v1143, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v1145 = stablehlo.divide %v1144, %v1136 : tensor<32x64x14x14xf32>
    %v1146 = stablehlo.add %v1145, %v1137 : tensor<32x64x14x14xf32>
    %v1147 = stablehlo.rsqrt %v1146 : tensor<32x64x14x14xf32>
    %v1148 = stablehlo.multiply %v1141, %v1147 : tensor<32x64x14x14xf32>
    %v1149 = stablehlo.reshape %v786 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v1150 = stablehlo.multiply %v1149, %v1148 : tensor<32x64x14x14xf32>
    %v1151 = stablehlo.reduce(%v1150 init: %v1134) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v1152 = stablehlo.constant dense<0.3> : tensor<64xf32>
    %v1153 = stablehlo.multiply %v1151, %v1152 : tensor<64xf32>
    %v1154 = stablehlo.subtract %gp5, %v1153 : tensor<64xf32>
    %v1155 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1156 = stablehlo.reshape %v786 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v1157 = stablehlo.reduce(%v1156 init: %v1155) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v1158 = stablehlo.constant dense<0.3> : tensor<64xf32>
    %v1159 = stablehlo.multiply %v1157, %v1158 : tensor<64xf32>
    %v1160 = stablehlo.subtract %btp5, %v1159 : tensor<64xf32>
    %v1161 = stablehlo.reshape %v1032 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v1162 = stablehlo.reshape %v341 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v1163 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1164 = stablehlo.constant dense<784.0> : tensor<32x32x28x28xf32>
    %v1165 = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %v1166 = stablehlo.reduce(%v1162 init: %v1163) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v1167 = stablehlo.broadcast_in_dim %v1166, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v1168 = stablehlo.divide %v1167, %v1164 : tensor<32x32x28x28xf32>
    %v1169 = stablehlo.subtract %v1162, %v1168 : tensor<32x32x28x28xf32>
    %v1170 = stablehlo.multiply %v1169, %v1169 : tensor<32x32x28x28xf32>
    %v1171 = stablehlo.reduce(%v1170 init: %v1163) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v1172 = stablehlo.broadcast_in_dim %v1171, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v1173 = stablehlo.divide %v1172, %v1164 : tensor<32x32x28x28xf32>
    %v1174 = stablehlo.add %v1173, %v1165 : tensor<32x32x28x28xf32>
    %v1175 = stablehlo.rsqrt %v1174 : tensor<32x32x28x28xf32>
    %v1176 = stablehlo.multiply %v1169, %v1175 : tensor<32x32x28x28xf32>
    %v1177 = stablehlo.broadcast_in_dim %gp4, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v1178 = stablehlo.multiply %v1177, %v1161 : tensor<32x32x28x28xf32>
    %v1179 = stablehlo.reduce(%v1178 init: %v1163) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v1180 = stablehlo.broadcast_in_dim %v1179, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v1181 = stablehlo.multiply %v1176, %v1178 : tensor<32x32x28x28xf32>
    %v1182 = stablehlo.reduce(%v1181 init: %v1163) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v1183 = stablehlo.broadcast_in_dim %v1182, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v1184 = stablehlo.multiply %v1178, %v1164 : tensor<32x32x28x28xf32>
    %v1185 = stablehlo.subtract %v1184, %v1180 : tensor<32x32x28x28xf32>
    %v1186 = stablehlo.multiply %v1176, %v1183 : tensor<32x32x28x28xf32>
    %v1187 = stablehlo.subtract %v1185, %v1186 : tensor<32x32x28x28xf32>
    %v1188 = stablehlo.divide %v1175, %v1164 : tensor<32x32x28x28xf32>
    %v1189 = stablehlo.multiply %v1188, %v1187 : tensor<32x32x28x28xf32>
    %v1190 = stablehlo.reshape %v1189 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v1191 = stablehlo.reshape %v1190 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v1192 = stablehlo.transpose %Wp4, dims = [1, 0, 2, 3] : (tensor<32x128x1x1xf32>) -> tensor<128x32x1x1xf32>
    %v1193 = stablehlo.reverse %v1192, dims = [2, 3] : tensor<128x32x1x1xf32>
    %v1194 = stablehlo.convolution(%v1191, %v1193)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<128x32x1x1xf32>) -> tensor<32x128x28x28xf32>
    %v1195 = stablehlo.reshape %v1194 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v1196 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v1197 = stablehlo.constant dense<6.0> : tensor<32x100352xf32>
    %v1198 = stablehlo.compare GT, %v332, %v1196 : (tensor<32x100352xf32>, tensor<32x100352xf32>) -> tensor<32x100352xi1>
    %v1199 = stablehlo.compare LT, %v332, %v1197 : (tensor<32x100352xf32>, tensor<32x100352xf32>) -> tensor<32x100352xi1>
    %v1200 = stablehlo.and %v1198, %v1199 : tensor<32x100352xi1>
    %v1201 = stablehlo.select %v1200, %v1195, %v1196 : tensor<32x100352xi1>, tensor<32x100352xf32>
    %v1202 = stablehlo.reshape %v1201 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v1203 = stablehlo.reshape %v312 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v1204 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1205 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v1206 = stablehlo.constant dense<1.0e-5> : tensor<32x128x28x28xf32>
    %v1207 = stablehlo.reduce(%v1203 init: %v1204) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v1208 = stablehlo.broadcast_in_dim %v1207, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v1209 = stablehlo.divide %v1208, %v1205 : tensor<32x128x28x28xf32>
    %v1210 = stablehlo.subtract %v1203, %v1209 : tensor<32x128x28x28xf32>
    %v1211 = stablehlo.multiply %v1210, %v1210 : tensor<32x128x28x28xf32>
    %v1212 = stablehlo.reduce(%v1211 init: %v1204) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v1213 = stablehlo.broadcast_in_dim %v1212, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v1214 = stablehlo.divide %v1213, %v1205 : tensor<32x128x28x28xf32>
    %v1215 = stablehlo.add %v1214, %v1206 : tensor<32x128x28x28xf32>
    %v1216 = stablehlo.rsqrt %v1215 : tensor<32x128x28x28xf32>
    %v1217 = stablehlo.multiply %v1210, %v1216 : tensor<32x128x28x28xf32>
    %v1218 = stablehlo.broadcast_in_dim %gd4, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v1219 = stablehlo.multiply %v1218, %v1202 : tensor<32x128x28x28xf32>
    %v1220 = stablehlo.reduce(%v1219 init: %v1204) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v1221 = stablehlo.broadcast_in_dim %v1220, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v1222 = stablehlo.multiply %v1217, %v1219 : tensor<32x128x28x28xf32>
    %v1223 = stablehlo.reduce(%v1222 init: %v1204) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v1224 = stablehlo.broadcast_in_dim %v1223, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v1225 = stablehlo.multiply %v1219, %v1205 : tensor<32x128x28x28xf32>
    %v1226 = stablehlo.subtract %v1225, %v1221 : tensor<32x128x28x28xf32>
    %v1227 = stablehlo.multiply %v1217, %v1224 : tensor<32x128x28x28xf32>
    %v1228 = stablehlo.subtract %v1226, %v1227 : tensor<32x128x28x28xf32>
    %v1229 = stablehlo.divide %v1216, %v1205 : tensor<32x128x28x28xf32>
    %v1230 = stablehlo.multiply %v1229, %v1228 : tensor<32x128x28x28xf32>
    %v1231 = stablehlo.reshape %v1230 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v1232 = stablehlo.reshape %v1231 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v1233 = stablehlo.reverse %Wd4, dims = [2, 3] : tensor<128x1x3x3xf32>
    %v1234 = stablehlo.convolution(%v1232, %v1233)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 128 : i64} : (tensor<32x128x28x28xf32>, tensor<128x1x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v1235 = stablehlo.reshape %v1234 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v1236 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v1237 = stablehlo.constant dense<6.0> : tensor<32x100352xf32>
    %v1238 = stablehlo.compare GT, %v303, %v1236 : (tensor<32x100352xf32>, tensor<32x100352xf32>) -> tensor<32x100352xi1>
    %v1239 = stablehlo.compare LT, %v303, %v1237 : (tensor<32x100352xf32>, tensor<32x100352xf32>) -> tensor<32x100352xi1>
    %v1240 = stablehlo.and %v1238, %v1239 : tensor<32x100352xi1>
    %v1241 = stablehlo.select %v1240, %v1235, %v1236 : tensor<32x100352xi1>, tensor<32x100352xf32>
    %v1242 = stablehlo.reshape %v1241 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v1243 = stablehlo.reshape %v283 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v1244 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1245 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v1246 = stablehlo.constant dense<1.0e-5> : tensor<32x128x28x28xf32>
    %v1247 = stablehlo.reduce(%v1243 init: %v1244) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v1248 = stablehlo.broadcast_in_dim %v1247, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v1249 = stablehlo.divide %v1248, %v1245 : tensor<32x128x28x28xf32>
    %v1250 = stablehlo.subtract %v1243, %v1249 : tensor<32x128x28x28xf32>
    %v1251 = stablehlo.multiply %v1250, %v1250 : tensor<32x128x28x28xf32>
    %v1252 = stablehlo.reduce(%v1251 init: %v1244) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v1253 = stablehlo.broadcast_in_dim %v1252, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v1254 = stablehlo.divide %v1253, %v1245 : tensor<32x128x28x28xf32>
    %v1255 = stablehlo.add %v1254, %v1246 : tensor<32x128x28x28xf32>
    %v1256 = stablehlo.rsqrt %v1255 : tensor<32x128x28x28xf32>
    %v1257 = stablehlo.multiply %v1250, %v1256 : tensor<32x128x28x28xf32>
    %v1258 = stablehlo.broadcast_in_dim %ge4, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v1259 = stablehlo.multiply %v1258, %v1242 : tensor<32x128x28x28xf32>
    %v1260 = stablehlo.reduce(%v1259 init: %v1244) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v1261 = stablehlo.broadcast_in_dim %v1260, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v1262 = stablehlo.multiply %v1257, %v1259 : tensor<32x128x28x28xf32>
    %v1263 = stablehlo.reduce(%v1262 init: %v1244) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v1264 = stablehlo.broadcast_in_dim %v1263, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v1265 = stablehlo.multiply %v1259, %v1245 : tensor<32x128x28x28xf32>
    %v1266 = stablehlo.subtract %v1265, %v1261 : tensor<32x128x28x28xf32>
    %v1267 = stablehlo.multiply %v1257, %v1264 : tensor<32x128x28x28xf32>
    %v1268 = stablehlo.subtract %v1266, %v1267 : tensor<32x128x28x28xf32>
    %v1269 = stablehlo.divide %v1256, %v1245 : tensor<32x128x28x28xf32>
    %v1270 = stablehlo.multiply %v1269, %v1268 : tensor<32x128x28x28xf32>
    %v1271 = stablehlo.reshape %v1270 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v1272 = stablehlo.reshape %v1271 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v1273 = stablehlo.transpose %We4, dims = [1, 0, 2, 3] : (tensor<128x32x1x1xf32>) -> tensor<32x128x1x1xf32>
    %v1274 = stablehlo.reverse %v1273, dims = [2, 3] : tensor<32x128x1x1xf32>
    %v1275 = stablehlo.convolution(%v1272, %v1274)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<32x128x1x1xf32>) -> tensor<32x32x28x28xf32>
    %v1276 = stablehlo.reshape %v1275 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v1277 = stablehlo.add %v1276, %v1032 : tensor<32x25088xf32>
    %v1278 = stablehlo.reshape %v278 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v1279 = stablehlo.reshape %v1271 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v1280 = stablehlo.transpose %v1278, dims = [1, 0, 2, 3] : (tensor<32x32x28x28xf32>) -> tensor<32x32x28x28xf32>
    %v1281 = stablehlo.transpose %v1279, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v1282 = stablehlo.convolution(%v1280, %v1281)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<32x128x1x1xf32>
    %v1283 = stablehlo.transpose %v1282, dims = [1, 0, 2, 3] : (tensor<32x128x1x1xf32>) -> tensor<128x32x1x1xf32>
    %v1284 = stablehlo.constant dense<0.3> : tensor<128x32x1x1xf32>
    %v1285 = stablehlo.multiply %v1283, %v1284 : tensor<128x32x1x1xf32>
    %v1286 = stablehlo.subtract %We4, %v1285 : tensor<128x32x1x1xf32>
    %v1287 = stablehlo.reshape %v1271 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v1288 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1289 = stablehlo.reduce(%v1287 init: %v1288) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v1290 = stablehlo.constant dense<0.3> : tensor<128xf32>
    %v1291 = stablehlo.multiply %v1289, %v1290 : tensor<128xf32>
    %v1292 = stablehlo.subtract %be4, %v1291 : tensor<128xf32>
    %v1293 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1294 = stablehlo.reshape %v283 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v1295 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v1296 = stablehlo.constant dense<1.0e-5> : tensor<32x128x28x28xf32>
    %v1297 = stablehlo.reduce(%v1294 init: %v1293) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v1298 = stablehlo.broadcast_in_dim %v1297, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v1299 = stablehlo.divide %v1298, %v1295 : tensor<32x128x28x28xf32>
    %v1300 = stablehlo.subtract %v1294, %v1299 : tensor<32x128x28x28xf32>
    %v1301 = stablehlo.multiply %v1300, %v1300 : tensor<32x128x28x28xf32>
    %v1302 = stablehlo.reduce(%v1301 init: %v1293) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v1303 = stablehlo.broadcast_in_dim %v1302, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v1304 = stablehlo.divide %v1303, %v1295 : tensor<32x128x28x28xf32>
    %v1305 = stablehlo.add %v1304, %v1296 : tensor<32x128x28x28xf32>
    %v1306 = stablehlo.rsqrt %v1305 : tensor<32x128x28x28xf32>
    %v1307 = stablehlo.multiply %v1300, %v1306 : tensor<32x128x28x28xf32>
    %v1308 = stablehlo.reshape %v1241 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v1309 = stablehlo.multiply %v1308, %v1307 : tensor<32x128x28x28xf32>
    %v1310 = stablehlo.reduce(%v1309 init: %v1293) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v1311 = stablehlo.constant dense<0.3> : tensor<128xf32>
    %v1312 = stablehlo.multiply %v1310, %v1311 : tensor<128xf32>
    %v1313 = stablehlo.subtract %ge4, %v1312 : tensor<128xf32>
    %v1314 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1315 = stablehlo.reshape %v1241 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v1316 = stablehlo.reduce(%v1315 init: %v1314) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v1317 = stablehlo.constant dense<0.3> : tensor<128xf32>
    %v1318 = stablehlo.multiply %v1316, %v1317 : tensor<128xf32>
    %v1319 = stablehlo.subtract %bte4, %v1318 : tensor<128xf32>
    %v1320 = stablehlo.reshape %v307 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v1321 = stablehlo.reshape %v1231 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v1322 = stablehlo.transpose %v1320, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v1323 = stablehlo.transpose %v1321, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v1324 = stablehlo.convolution(%v1322, %v1323)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 128 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<1x128x3x3xf32>
    %v1325 = stablehlo.reshape %v1324 : (tensor<1x128x3x3xf32>) -> tensor<128x1x3x3xf32>
    %v1326 = stablehlo.constant dense<0.3> : tensor<128x1x3x3xf32>
    %v1327 = stablehlo.multiply %v1325, %v1326 : tensor<128x1x3x3xf32>
    %v1328 = stablehlo.subtract %Wd4, %v1327 : tensor<128x1x3x3xf32>
    %v1329 = stablehlo.reshape %v1231 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v1330 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1331 = stablehlo.reduce(%v1329 init: %v1330) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v1332 = stablehlo.constant dense<0.3> : tensor<128xf32>
    %v1333 = stablehlo.multiply %v1331, %v1332 : tensor<128xf32>
    %v1334 = stablehlo.subtract %bd4, %v1333 : tensor<128xf32>
    %v1335 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1336 = stablehlo.reshape %v312 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v1337 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v1338 = stablehlo.constant dense<1.0e-5> : tensor<32x128x28x28xf32>
    %v1339 = stablehlo.reduce(%v1336 init: %v1335) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v1340 = stablehlo.broadcast_in_dim %v1339, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v1341 = stablehlo.divide %v1340, %v1337 : tensor<32x128x28x28xf32>
    %v1342 = stablehlo.subtract %v1336, %v1341 : tensor<32x128x28x28xf32>
    %v1343 = stablehlo.multiply %v1342, %v1342 : tensor<32x128x28x28xf32>
    %v1344 = stablehlo.reduce(%v1343 init: %v1335) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v1345 = stablehlo.broadcast_in_dim %v1344, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v1346 = stablehlo.divide %v1345, %v1337 : tensor<32x128x28x28xf32>
    %v1347 = stablehlo.add %v1346, %v1338 : tensor<32x128x28x28xf32>
    %v1348 = stablehlo.rsqrt %v1347 : tensor<32x128x28x28xf32>
    %v1349 = stablehlo.multiply %v1342, %v1348 : tensor<32x128x28x28xf32>
    %v1350 = stablehlo.reshape %v1201 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v1351 = stablehlo.multiply %v1350, %v1349 : tensor<32x128x28x28xf32>
    %v1352 = stablehlo.reduce(%v1351 init: %v1335) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v1353 = stablehlo.constant dense<0.3> : tensor<128xf32>
    %v1354 = stablehlo.multiply %v1352, %v1353 : tensor<128xf32>
    %v1355 = stablehlo.subtract %gd4, %v1354 : tensor<128xf32>
    %v1356 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1357 = stablehlo.reshape %v1201 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v1358 = stablehlo.reduce(%v1357 init: %v1356) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v1359 = stablehlo.constant dense<0.3> : tensor<128xf32>
    %v1360 = stablehlo.multiply %v1358, %v1359 : tensor<128xf32>
    %v1361 = stablehlo.subtract %btd4, %v1360 : tensor<128xf32>
    %v1362 = stablehlo.reshape %v336 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v1363 = stablehlo.reshape %v1190 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v1364 = stablehlo.transpose %v1362, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v1365 = stablehlo.transpose %v1363, dims = [1, 0, 2, 3] : (tensor<32x32x28x28xf32>) -> tensor<32x32x28x28xf32>
    %v1366 = stablehlo.convolution(%v1364, %v1365)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<32x32x28x28xf32>) -> tensor<128x32x1x1xf32>
    %v1367 = stablehlo.transpose %v1366, dims = [1, 0, 2, 3] : (tensor<128x32x1x1xf32>) -> tensor<32x128x1x1xf32>
    %v1368 = stablehlo.constant dense<0.3> : tensor<32x128x1x1xf32>
    %v1369 = stablehlo.multiply %v1367, %v1368 : tensor<32x128x1x1xf32>
    %v1370 = stablehlo.subtract %Wp4, %v1369 : tensor<32x128x1x1xf32>
    %v1371 = stablehlo.reshape %v1190 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v1372 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1373 = stablehlo.reduce(%v1371 init: %v1372) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v1374 = stablehlo.constant dense<0.3> : tensor<32xf32>
    %v1375 = stablehlo.multiply %v1373, %v1374 : tensor<32xf32>
    %v1376 = stablehlo.subtract %bp4, %v1375 : tensor<32xf32>
    %v1377 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1378 = stablehlo.reshape %v341 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v1379 = stablehlo.constant dense<784.0> : tensor<32x32x28x28xf32>
    %v1380 = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %v1381 = stablehlo.reduce(%v1378 init: %v1377) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v1382 = stablehlo.broadcast_in_dim %v1381, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v1383 = stablehlo.divide %v1382, %v1379 : tensor<32x32x28x28xf32>
    %v1384 = stablehlo.subtract %v1378, %v1383 : tensor<32x32x28x28xf32>
    %v1385 = stablehlo.multiply %v1384, %v1384 : tensor<32x32x28x28xf32>
    %v1386 = stablehlo.reduce(%v1385 init: %v1377) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v1387 = stablehlo.broadcast_in_dim %v1386, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v1388 = stablehlo.divide %v1387, %v1379 : tensor<32x32x28x28xf32>
    %v1389 = stablehlo.add %v1388, %v1380 : tensor<32x32x28x28xf32>
    %v1390 = stablehlo.rsqrt %v1389 : tensor<32x32x28x28xf32>
    %v1391 = stablehlo.multiply %v1384, %v1390 : tensor<32x32x28x28xf32>
    %v1392 = stablehlo.reshape %v1032 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v1393 = stablehlo.multiply %v1392, %v1391 : tensor<32x32x28x28xf32>
    %v1394 = stablehlo.reduce(%v1393 init: %v1377) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v1395 = stablehlo.constant dense<0.3> : tensor<32xf32>
    %v1396 = stablehlo.multiply %v1394, %v1395 : tensor<32xf32>
    %v1397 = stablehlo.subtract %gp4, %v1396 : tensor<32xf32>
    %v1398 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1399 = stablehlo.reshape %v1032 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v1400 = stablehlo.reduce(%v1399 init: %v1398) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v1401 = stablehlo.constant dense<0.3> : tensor<32xf32>
    %v1402 = stablehlo.multiply %v1400, %v1401 : tensor<32xf32>
    %v1403 = stablehlo.subtract %btp4, %v1402 : tensor<32xf32>
    %v1404 = stablehlo.reshape %v1277 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v1405 = stablehlo.reshape %v258 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v1406 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1407 = stablehlo.constant dense<784.0> : tensor<32x32x28x28xf32>
    %v1408 = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %v1409 = stablehlo.reduce(%v1405 init: %v1406) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v1410 = stablehlo.broadcast_in_dim %v1409, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v1411 = stablehlo.divide %v1410, %v1407 : tensor<32x32x28x28xf32>
    %v1412 = stablehlo.subtract %v1405, %v1411 : tensor<32x32x28x28xf32>
    %v1413 = stablehlo.multiply %v1412, %v1412 : tensor<32x32x28x28xf32>
    %v1414 = stablehlo.reduce(%v1413 init: %v1406) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v1415 = stablehlo.broadcast_in_dim %v1414, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v1416 = stablehlo.divide %v1415, %v1407 : tensor<32x32x28x28xf32>
    %v1417 = stablehlo.add %v1416, %v1408 : tensor<32x32x28x28xf32>
    %v1418 = stablehlo.rsqrt %v1417 : tensor<32x32x28x28xf32>
    %v1419 = stablehlo.multiply %v1412, %v1418 : tensor<32x32x28x28xf32>
    %v1420 = stablehlo.broadcast_in_dim %gp3, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v1421 = stablehlo.multiply %v1420, %v1404 : tensor<32x32x28x28xf32>
    %v1422 = stablehlo.reduce(%v1421 init: %v1406) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v1423 = stablehlo.broadcast_in_dim %v1422, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v1424 = stablehlo.multiply %v1419, %v1421 : tensor<32x32x28x28xf32>
    %v1425 = stablehlo.reduce(%v1424 init: %v1406) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v1426 = stablehlo.broadcast_in_dim %v1425, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v1427 = stablehlo.multiply %v1421, %v1407 : tensor<32x32x28x28xf32>
    %v1428 = stablehlo.subtract %v1427, %v1423 : tensor<32x32x28x28xf32>
    %v1429 = stablehlo.multiply %v1419, %v1426 : tensor<32x32x28x28xf32>
    %v1430 = stablehlo.subtract %v1428, %v1429 : tensor<32x32x28x28xf32>
    %v1431 = stablehlo.divide %v1418, %v1407 : tensor<32x32x28x28xf32>
    %v1432 = stablehlo.multiply %v1431, %v1430 : tensor<32x32x28x28xf32>
    %v1433 = stablehlo.reshape %v1432 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v1434 = stablehlo.reshape %v1433 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v1435 = stablehlo.transpose %Wp3, dims = [1, 0, 2, 3] : (tensor<32x96x1x1xf32>) -> tensor<96x32x1x1xf32>
    %v1436 = stablehlo.reverse %v1435, dims = [2, 3] : tensor<96x32x1x1xf32>
    %v1437 = stablehlo.convolution(%v1434, %v1436)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<96x32x1x1xf32>) -> tensor<32x96x28x28xf32>
    %v1438 = stablehlo.reshape %v1437 : (tensor<32x96x28x28xf32>) -> tensor<32x75264xf32>
    %v1439 = stablehlo.constant dense<0.0> : tensor<32x75264xf32>
    %v1440 = stablehlo.constant dense<6.0> : tensor<32x75264xf32>
    %v1441 = stablehlo.compare GT, %v249, %v1439 : (tensor<32x75264xf32>, tensor<32x75264xf32>) -> tensor<32x75264xi1>
    %v1442 = stablehlo.compare LT, %v249, %v1440 : (tensor<32x75264xf32>, tensor<32x75264xf32>) -> tensor<32x75264xi1>
    %v1443 = stablehlo.and %v1441, %v1442 : tensor<32x75264xi1>
    %v1444 = stablehlo.select %v1443, %v1438, %v1439 : tensor<32x75264xi1>, tensor<32x75264xf32>
    %v1445 = stablehlo.reshape %v1444 : (tensor<32x75264xf32>) -> tensor<32x96x28x28xf32>
    %v1446 = stablehlo.reshape %v229 : (tensor<32x75264xf32>) -> tensor<32x96x28x28xf32>
    %v1447 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1448 = stablehlo.constant dense<784.0> : tensor<32x96x28x28xf32>
    %v1449 = stablehlo.constant dense<1.0e-5> : tensor<32x96x28x28xf32>
    %v1450 = stablehlo.reduce(%v1446 init: %v1447) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x28x28xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1451 = stablehlo.broadcast_in_dim %v1450, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x28x28xf32>
    %v1452 = stablehlo.divide %v1451, %v1448 : tensor<32x96x28x28xf32>
    %v1453 = stablehlo.subtract %v1446, %v1452 : tensor<32x96x28x28xf32>
    %v1454 = stablehlo.multiply %v1453, %v1453 : tensor<32x96x28x28xf32>
    %v1455 = stablehlo.reduce(%v1454 init: %v1447) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x28x28xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1456 = stablehlo.broadcast_in_dim %v1455, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x28x28xf32>
    %v1457 = stablehlo.divide %v1456, %v1448 : tensor<32x96x28x28xf32>
    %v1458 = stablehlo.add %v1457, %v1449 : tensor<32x96x28x28xf32>
    %v1459 = stablehlo.rsqrt %v1458 : tensor<32x96x28x28xf32>
    %v1460 = stablehlo.multiply %v1453, %v1459 : tensor<32x96x28x28xf32>
    %v1461 = stablehlo.broadcast_in_dim %gd3, dims = [1] : (tensor<96xf32>) -> tensor<32x96x28x28xf32>
    %v1462 = stablehlo.multiply %v1461, %v1445 : tensor<32x96x28x28xf32>
    %v1463 = stablehlo.reduce(%v1462 init: %v1447) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x28x28xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1464 = stablehlo.broadcast_in_dim %v1463, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x28x28xf32>
    %v1465 = stablehlo.multiply %v1460, %v1462 : tensor<32x96x28x28xf32>
    %v1466 = stablehlo.reduce(%v1465 init: %v1447) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x28x28xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1467 = stablehlo.broadcast_in_dim %v1466, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x28x28xf32>
    %v1468 = stablehlo.multiply %v1462, %v1448 : tensor<32x96x28x28xf32>
    %v1469 = stablehlo.subtract %v1468, %v1464 : tensor<32x96x28x28xf32>
    %v1470 = stablehlo.multiply %v1460, %v1467 : tensor<32x96x28x28xf32>
    %v1471 = stablehlo.subtract %v1469, %v1470 : tensor<32x96x28x28xf32>
    %v1472 = stablehlo.divide %v1459, %v1448 : tensor<32x96x28x28xf32>
    %v1473 = stablehlo.multiply %v1472, %v1471 : tensor<32x96x28x28xf32>
    %v1474 = stablehlo.reshape %v1473 : (tensor<32x96x28x28xf32>) -> tensor<32x75264xf32>
    %v1475 = stablehlo.reshape %v1474 : (tensor<32x75264xf32>) -> tensor<32x96x28x28xf32>
    %v1476 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1477 = stablehlo.pad %v1475, %v1476, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x96x28x28xf32>, tensor<f32>) -> tensor<32x96x56x56xf32>
    %v1478 = stablehlo.reverse %Wd3, dims = [2, 3] : tensor<96x1x3x3xf32>
    %v1479 = stablehlo.convolution(%v1477, %v1478)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x3x3xf32>) -> tensor<32x96x56x56xf32>
    %v1480 = stablehlo.reshape %v1479 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v1481 = stablehlo.constant dense<0.0> : tensor<32x301056xf32>
    %v1482 = stablehlo.constant dense<6.0> : tensor<32x301056xf32>
    %v1483 = stablehlo.compare GT, %v220, %v1481 : (tensor<32x301056xf32>, tensor<32x301056xf32>) -> tensor<32x301056xi1>
    %v1484 = stablehlo.compare LT, %v220, %v1482 : (tensor<32x301056xf32>, tensor<32x301056xf32>) -> tensor<32x301056xi1>
    %v1485 = stablehlo.and %v1483, %v1484 : tensor<32x301056xi1>
    %v1486 = stablehlo.select %v1485, %v1480, %v1481 : tensor<32x301056xi1>, tensor<32x301056xf32>
    %v1487 = stablehlo.reshape %v1486 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1488 = stablehlo.reshape %v200 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1489 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1490 = stablehlo.constant dense<3136.0> : tensor<32x96x56x56xf32>
    %v1491 = stablehlo.constant dense<1.0e-5> : tensor<32x96x56x56xf32>
    %v1492 = stablehlo.reduce(%v1488 init: %v1489) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1493 = stablehlo.broadcast_in_dim %v1492, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v1494 = stablehlo.divide %v1493, %v1490 : tensor<32x96x56x56xf32>
    %v1495 = stablehlo.subtract %v1488, %v1494 : tensor<32x96x56x56xf32>
    %v1496 = stablehlo.multiply %v1495, %v1495 : tensor<32x96x56x56xf32>
    %v1497 = stablehlo.reduce(%v1496 init: %v1489) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1498 = stablehlo.broadcast_in_dim %v1497, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v1499 = stablehlo.divide %v1498, %v1490 : tensor<32x96x56x56xf32>
    %v1500 = stablehlo.add %v1499, %v1491 : tensor<32x96x56x56xf32>
    %v1501 = stablehlo.rsqrt %v1500 : tensor<32x96x56x56xf32>
    %v1502 = stablehlo.multiply %v1495, %v1501 : tensor<32x96x56x56xf32>
    %v1503 = stablehlo.broadcast_in_dim %ge3, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v1504 = stablehlo.multiply %v1503, %v1487 : tensor<32x96x56x56xf32>
    %v1505 = stablehlo.reduce(%v1504 init: %v1489) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1506 = stablehlo.broadcast_in_dim %v1505, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v1507 = stablehlo.multiply %v1502, %v1504 : tensor<32x96x56x56xf32>
    %v1508 = stablehlo.reduce(%v1507 init: %v1489) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1509 = stablehlo.broadcast_in_dim %v1508, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v1510 = stablehlo.multiply %v1504, %v1490 : tensor<32x96x56x56xf32>
    %v1511 = stablehlo.subtract %v1510, %v1506 : tensor<32x96x56x56xf32>
    %v1512 = stablehlo.multiply %v1502, %v1509 : tensor<32x96x56x56xf32>
    %v1513 = stablehlo.subtract %v1511, %v1512 : tensor<32x96x56x56xf32>
    %v1514 = stablehlo.divide %v1501, %v1490 : tensor<32x96x56x56xf32>
    %v1515 = stablehlo.multiply %v1514, %v1513 : tensor<32x96x56x56xf32>
    %v1516 = stablehlo.reshape %v1515 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v1517 = stablehlo.reshape %v1516 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1518 = stablehlo.transpose %We3, dims = [1, 0, 2, 3] : (tensor<96x24x1x1xf32>) -> tensor<24x96x1x1xf32>
    %v1519 = stablehlo.reverse %v1518, dims = [2, 3] : tensor<24x96x1x1xf32>
    %v1520 = stablehlo.convolution(%v1517, %v1519)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<24x96x1x1xf32>) -> tensor<32x24x56x56xf32>
    %v1521 = stablehlo.reshape %v1520 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v1522 = stablehlo.reshape %v195 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v1523 = stablehlo.reshape %v1516 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1524 = stablehlo.transpose %v1522, dims = [1, 0, 2, 3] : (tensor<32x24x56x56xf32>) -> tensor<24x32x56x56xf32>
    %v1525 = stablehlo.transpose %v1523, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v1526 = stablehlo.convolution(%v1524, %v1525)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<24x32x56x56xf32>, tensor<96x32x56x56xf32>) -> tensor<24x96x1x1xf32>
    %v1527 = stablehlo.transpose %v1526, dims = [1, 0, 2, 3] : (tensor<24x96x1x1xf32>) -> tensor<96x24x1x1xf32>
    %v1528 = stablehlo.constant dense<0.3> : tensor<96x24x1x1xf32>
    %v1529 = stablehlo.multiply %v1527, %v1528 : tensor<96x24x1x1xf32>
    %v1530 = stablehlo.subtract %We3, %v1529 : tensor<96x24x1x1xf32>
    %v1531 = stablehlo.reshape %v1516 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1532 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1533 = stablehlo.reduce(%v1531 init: %v1532) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v1534 = stablehlo.constant dense<0.3> : tensor<96xf32>
    %v1535 = stablehlo.multiply %v1533, %v1534 : tensor<96xf32>
    %v1536 = stablehlo.subtract %be3, %v1535 : tensor<96xf32>
    %v1537 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1538 = stablehlo.reshape %v200 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1539 = stablehlo.constant dense<3136.0> : tensor<32x96x56x56xf32>
    %v1540 = stablehlo.constant dense<1.0e-5> : tensor<32x96x56x56xf32>
    %v1541 = stablehlo.reduce(%v1538 init: %v1537) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1542 = stablehlo.broadcast_in_dim %v1541, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v1543 = stablehlo.divide %v1542, %v1539 : tensor<32x96x56x56xf32>
    %v1544 = stablehlo.subtract %v1538, %v1543 : tensor<32x96x56x56xf32>
    %v1545 = stablehlo.multiply %v1544, %v1544 : tensor<32x96x56x56xf32>
    %v1546 = stablehlo.reduce(%v1545 init: %v1537) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1547 = stablehlo.broadcast_in_dim %v1546, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v1548 = stablehlo.divide %v1547, %v1539 : tensor<32x96x56x56xf32>
    %v1549 = stablehlo.add %v1548, %v1540 : tensor<32x96x56x56xf32>
    %v1550 = stablehlo.rsqrt %v1549 : tensor<32x96x56x56xf32>
    %v1551 = stablehlo.multiply %v1544, %v1550 : tensor<32x96x56x56xf32>
    %v1552 = stablehlo.reshape %v1486 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1553 = stablehlo.multiply %v1552, %v1551 : tensor<32x96x56x56xf32>
    %v1554 = stablehlo.reduce(%v1553 init: %v1537) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v1555 = stablehlo.constant dense<0.3> : tensor<96xf32>
    %v1556 = stablehlo.multiply %v1554, %v1555 : tensor<96xf32>
    %v1557 = stablehlo.subtract %ge3, %v1556 : tensor<96xf32>
    %v1558 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1559 = stablehlo.reshape %v1486 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1560 = stablehlo.reduce(%v1559 init: %v1558) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v1561 = stablehlo.constant dense<0.3> : tensor<96xf32>
    %v1562 = stablehlo.multiply %v1560, %v1561 : tensor<96xf32>
    %v1563 = stablehlo.subtract %bte3, %v1562 : tensor<96xf32>
    %v1564 = stablehlo.reshape %v224 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1565 = stablehlo.reshape %v1474 : (tensor<32x75264xf32>) -> tensor<32x96x28x28xf32>
    %v1566 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1567 = stablehlo.pad %v1565, %v1566, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x96x28x28xf32>, tensor<f32>) -> tensor<32x96x56x56xf32>
    %v1568 = stablehlo.transpose %v1564, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v1569 = stablehlo.transpose %v1567, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v1570 = stablehlo.convolution(%v1568, %v1569)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 96 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<96x32x56x56xf32>) -> tensor<1x96x3x3xf32>
    %v1571 = stablehlo.reshape %v1570 : (tensor<1x96x3x3xf32>) -> tensor<96x1x3x3xf32>
    %v1572 = stablehlo.constant dense<0.3> : tensor<96x1x3x3xf32>
    %v1573 = stablehlo.multiply %v1571, %v1572 : tensor<96x1x3x3xf32>
    %v1574 = stablehlo.subtract %Wd3, %v1573 : tensor<96x1x3x3xf32>
    %v1575 = stablehlo.reshape %v1474 : (tensor<32x75264xf32>) -> tensor<32x96x28x28xf32>
    %v1576 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1577 = stablehlo.reduce(%v1575 init: %v1576) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x28x28xf32>, tensor<f32>) -> tensor<96xf32>
    %v1578 = stablehlo.constant dense<0.3> : tensor<96xf32>
    %v1579 = stablehlo.multiply %v1577, %v1578 : tensor<96xf32>
    %v1580 = stablehlo.subtract %bd3, %v1579 : tensor<96xf32>
    %v1581 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1582 = stablehlo.reshape %v229 : (tensor<32x75264xf32>) -> tensor<32x96x28x28xf32>
    %v1583 = stablehlo.constant dense<784.0> : tensor<32x96x28x28xf32>
    %v1584 = stablehlo.constant dense<1.0e-5> : tensor<32x96x28x28xf32>
    %v1585 = stablehlo.reduce(%v1582 init: %v1581) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x28x28xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1586 = stablehlo.broadcast_in_dim %v1585, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x28x28xf32>
    %v1587 = stablehlo.divide %v1586, %v1583 : tensor<32x96x28x28xf32>
    %v1588 = stablehlo.subtract %v1582, %v1587 : tensor<32x96x28x28xf32>
    %v1589 = stablehlo.multiply %v1588, %v1588 : tensor<32x96x28x28xf32>
    %v1590 = stablehlo.reduce(%v1589 init: %v1581) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x28x28xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1591 = stablehlo.broadcast_in_dim %v1590, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x28x28xf32>
    %v1592 = stablehlo.divide %v1591, %v1583 : tensor<32x96x28x28xf32>
    %v1593 = stablehlo.add %v1592, %v1584 : tensor<32x96x28x28xf32>
    %v1594 = stablehlo.rsqrt %v1593 : tensor<32x96x28x28xf32>
    %v1595 = stablehlo.multiply %v1588, %v1594 : tensor<32x96x28x28xf32>
    %v1596 = stablehlo.reshape %v1444 : (tensor<32x75264xf32>) -> tensor<32x96x28x28xf32>
    %v1597 = stablehlo.multiply %v1596, %v1595 : tensor<32x96x28x28xf32>
    %v1598 = stablehlo.reduce(%v1597 init: %v1581) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x28x28xf32>, tensor<f32>) -> tensor<96xf32>
    %v1599 = stablehlo.constant dense<0.3> : tensor<96xf32>
    %v1600 = stablehlo.multiply %v1598, %v1599 : tensor<96xf32>
    %v1601 = stablehlo.subtract %gd3, %v1600 : tensor<96xf32>
    %v1602 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1603 = stablehlo.reshape %v1444 : (tensor<32x75264xf32>) -> tensor<32x96x28x28xf32>
    %v1604 = stablehlo.reduce(%v1603 init: %v1602) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x28x28xf32>, tensor<f32>) -> tensor<96xf32>
    %v1605 = stablehlo.constant dense<0.3> : tensor<96xf32>
    %v1606 = stablehlo.multiply %v1604, %v1605 : tensor<96xf32>
    %v1607 = stablehlo.subtract %btd3, %v1606 : tensor<96xf32>
    %v1608 = stablehlo.reshape %v253 : (tensor<32x75264xf32>) -> tensor<32x96x28x28xf32>
    %v1609 = stablehlo.reshape %v1433 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v1610 = stablehlo.transpose %v1608, dims = [1, 0, 2, 3] : (tensor<32x96x28x28xf32>) -> tensor<96x32x28x28xf32>
    %v1611 = stablehlo.transpose %v1609, dims = [1, 0, 2, 3] : (tensor<32x32x28x28xf32>) -> tensor<32x32x28x28xf32>
    %v1612 = stablehlo.convolution(%v1610, %v1611)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x28x28xf32>, tensor<32x32x28x28xf32>) -> tensor<96x32x1x1xf32>
    %v1613 = stablehlo.transpose %v1612, dims = [1, 0, 2, 3] : (tensor<96x32x1x1xf32>) -> tensor<32x96x1x1xf32>
    %v1614 = stablehlo.constant dense<0.3> : tensor<32x96x1x1xf32>
    %v1615 = stablehlo.multiply %v1613, %v1614 : tensor<32x96x1x1xf32>
    %v1616 = stablehlo.subtract %Wp3, %v1615 : tensor<32x96x1x1xf32>
    %v1617 = stablehlo.reshape %v1433 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v1618 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1619 = stablehlo.reduce(%v1617 init: %v1618) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v1620 = stablehlo.constant dense<0.3> : tensor<32xf32>
    %v1621 = stablehlo.multiply %v1619, %v1620 : tensor<32xf32>
    %v1622 = stablehlo.subtract %bp3, %v1621 : tensor<32xf32>
    %v1623 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1624 = stablehlo.reshape %v258 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v1625 = stablehlo.constant dense<784.0> : tensor<32x32x28x28xf32>
    %v1626 = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %v1627 = stablehlo.reduce(%v1624 init: %v1623) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v1628 = stablehlo.broadcast_in_dim %v1627, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v1629 = stablehlo.divide %v1628, %v1625 : tensor<32x32x28x28xf32>
    %v1630 = stablehlo.subtract %v1624, %v1629 : tensor<32x32x28x28xf32>
    %v1631 = stablehlo.multiply %v1630, %v1630 : tensor<32x32x28x28xf32>
    %v1632 = stablehlo.reduce(%v1631 init: %v1623) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v1633 = stablehlo.broadcast_in_dim %v1632, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v1634 = stablehlo.divide %v1633, %v1625 : tensor<32x32x28x28xf32>
    %v1635 = stablehlo.add %v1634, %v1626 : tensor<32x32x28x28xf32>
    %v1636 = stablehlo.rsqrt %v1635 : tensor<32x32x28x28xf32>
    %v1637 = stablehlo.multiply %v1630, %v1636 : tensor<32x32x28x28xf32>
    %v1638 = stablehlo.reshape %v1277 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v1639 = stablehlo.multiply %v1638, %v1637 : tensor<32x32x28x28xf32>
    %v1640 = stablehlo.reduce(%v1639 init: %v1623) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v1641 = stablehlo.constant dense<0.3> : tensor<32xf32>
    %v1642 = stablehlo.multiply %v1640, %v1641 : tensor<32xf32>
    %v1643 = stablehlo.subtract %gp3, %v1642 : tensor<32xf32>
    %v1644 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1645 = stablehlo.reshape %v1277 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v1646 = stablehlo.reduce(%v1645 init: %v1644) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v1647 = stablehlo.constant dense<0.3> : tensor<32xf32>
    %v1648 = stablehlo.multiply %v1646, %v1647 : tensor<32xf32>
    %v1649 = stablehlo.subtract %btp3, %v1648 : tensor<32xf32>
    %v1650 = stablehlo.reshape %v1521 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v1651 = stablehlo.reshape %v174 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v1652 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1653 = stablehlo.constant dense<3136.0> : tensor<32x24x56x56xf32>
    %v1654 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v1655 = stablehlo.reduce(%v1651 init: %v1652) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v1656 = stablehlo.broadcast_in_dim %v1655, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v1657 = stablehlo.divide %v1656, %v1653 : tensor<32x24x56x56xf32>
    %v1658 = stablehlo.subtract %v1651, %v1657 : tensor<32x24x56x56xf32>
    %v1659 = stablehlo.multiply %v1658, %v1658 : tensor<32x24x56x56xf32>
    %v1660 = stablehlo.reduce(%v1659 init: %v1652) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v1661 = stablehlo.broadcast_in_dim %v1660, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v1662 = stablehlo.divide %v1661, %v1653 : tensor<32x24x56x56xf32>
    %v1663 = stablehlo.add %v1662, %v1654 : tensor<32x24x56x56xf32>
    %v1664 = stablehlo.rsqrt %v1663 : tensor<32x24x56x56xf32>
    %v1665 = stablehlo.multiply %v1658, %v1664 : tensor<32x24x56x56xf32>
    %v1666 = stablehlo.broadcast_in_dim %gp2, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v1667 = stablehlo.multiply %v1666, %v1650 : tensor<32x24x56x56xf32>
    %v1668 = stablehlo.reduce(%v1667 init: %v1652) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v1669 = stablehlo.broadcast_in_dim %v1668, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v1670 = stablehlo.multiply %v1665, %v1667 : tensor<32x24x56x56xf32>
    %v1671 = stablehlo.reduce(%v1670 init: %v1652) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v1672 = stablehlo.broadcast_in_dim %v1671, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v1673 = stablehlo.multiply %v1667, %v1653 : tensor<32x24x56x56xf32>
    %v1674 = stablehlo.subtract %v1673, %v1669 : tensor<32x24x56x56xf32>
    %v1675 = stablehlo.multiply %v1665, %v1672 : tensor<32x24x56x56xf32>
    %v1676 = stablehlo.subtract %v1674, %v1675 : tensor<32x24x56x56xf32>
    %v1677 = stablehlo.divide %v1664, %v1653 : tensor<32x24x56x56xf32>
    %v1678 = stablehlo.multiply %v1677, %v1676 : tensor<32x24x56x56xf32>
    %v1679 = stablehlo.reshape %v1678 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v1680 = stablehlo.reshape %v1679 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v1681 = stablehlo.transpose %Wp2, dims = [1, 0, 2, 3] : (tensor<24x96x1x1xf32>) -> tensor<96x24x1x1xf32>
    %v1682 = stablehlo.reverse %v1681, dims = [2, 3] : tensor<96x24x1x1xf32>
    %v1683 = stablehlo.convolution(%v1680, %v1682)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<96x24x1x1xf32>) -> tensor<32x96x56x56xf32>
    %v1684 = stablehlo.reshape %v1683 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v1685 = stablehlo.constant dense<0.0> : tensor<32x301056xf32>
    %v1686 = stablehlo.constant dense<6.0> : tensor<32x301056xf32>
    %v1687 = stablehlo.compare GT, %v165, %v1685 : (tensor<32x301056xf32>, tensor<32x301056xf32>) -> tensor<32x301056xi1>
    %v1688 = stablehlo.compare LT, %v165, %v1686 : (tensor<32x301056xf32>, tensor<32x301056xf32>) -> tensor<32x301056xi1>
    %v1689 = stablehlo.and %v1687, %v1688 : tensor<32x301056xi1>
    %v1690 = stablehlo.select %v1689, %v1684, %v1685 : tensor<32x301056xi1>, tensor<32x301056xf32>
    %v1691 = stablehlo.reshape %v1690 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1692 = stablehlo.reshape %v145 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1693 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1694 = stablehlo.constant dense<3136.0> : tensor<32x96x56x56xf32>
    %v1695 = stablehlo.constant dense<1.0e-5> : tensor<32x96x56x56xf32>
    %v1696 = stablehlo.reduce(%v1692 init: %v1693) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1697 = stablehlo.broadcast_in_dim %v1696, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v1698 = stablehlo.divide %v1697, %v1694 : tensor<32x96x56x56xf32>
    %v1699 = stablehlo.subtract %v1692, %v1698 : tensor<32x96x56x56xf32>
    %v1700 = stablehlo.multiply %v1699, %v1699 : tensor<32x96x56x56xf32>
    %v1701 = stablehlo.reduce(%v1700 init: %v1693) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1702 = stablehlo.broadcast_in_dim %v1701, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v1703 = stablehlo.divide %v1702, %v1694 : tensor<32x96x56x56xf32>
    %v1704 = stablehlo.add %v1703, %v1695 : tensor<32x96x56x56xf32>
    %v1705 = stablehlo.rsqrt %v1704 : tensor<32x96x56x56xf32>
    %v1706 = stablehlo.multiply %v1699, %v1705 : tensor<32x96x56x56xf32>
    %v1707 = stablehlo.broadcast_in_dim %gd2, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v1708 = stablehlo.multiply %v1707, %v1691 : tensor<32x96x56x56xf32>
    %v1709 = stablehlo.reduce(%v1708 init: %v1693) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1710 = stablehlo.broadcast_in_dim %v1709, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v1711 = stablehlo.multiply %v1706, %v1708 : tensor<32x96x56x56xf32>
    %v1712 = stablehlo.reduce(%v1711 init: %v1693) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1713 = stablehlo.broadcast_in_dim %v1712, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v1714 = stablehlo.multiply %v1708, %v1694 : tensor<32x96x56x56xf32>
    %v1715 = stablehlo.subtract %v1714, %v1710 : tensor<32x96x56x56xf32>
    %v1716 = stablehlo.multiply %v1706, %v1713 : tensor<32x96x56x56xf32>
    %v1717 = stablehlo.subtract %v1715, %v1716 : tensor<32x96x56x56xf32>
    %v1718 = stablehlo.divide %v1705, %v1694 : tensor<32x96x56x56xf32>
    %v1719 = stablehlo.multiply %v1718, %v1717 : tensor<32x96x56x56xf32>
    %v1720 = stablehlo.reshape %v1719 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v1721 = stablehlo.reshape %v1720 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1722 = stablehlo.reverse %Wd2, dims = [2, 3] : tensor<96x1x3x3xf32>
    %v1723 = stablehlo.convolution(%v1721, %v1722)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x3x3xf32>) -> tensor<32x96x56x56xf32>
    %v1724 = stablehlo.reshape %v1723 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v1725 = stablehlo.constant dense<0.0> : tensor<32x301056xf32>
    %v1726 = stablehlo.constant dense<6.0> : tensor<32x301056xf32>
    %v1727 = stablehlo.compare GT, %v136, %v1725 : (tensor<32x301056xf32>, tensor<32x301056xf32>) -> tensor<32x301056xi1>
    %v1728 = stablehlo.compare LT, %v136, %v1726 : (tensor<32x301056xf32>, tensor<32x301056xf32>) -> tensor<32x301056xi1>
    %v1729 = stablehlo.and %v1727, %v1728 : tensor<32x301056xi1>
    %v1730 = stablehlo.select %v1729, %v1724, %v1725 : tensor<32x301056xi1>, tensor<32x301056xf32>
    %v1731 = stablehlo.reshape %v1730 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1732 = stablehlo.reshape %v116 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1733 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1734 = stablehlo.constant dense<3136.0> : tensor<32x96x56x56xf32>
    %v1735 = stablehlo.constant dense<1.0e-5> : tensor<32x96x56x56xf32>
    %v1736 = stablehlo.reduce(%v1732 init: %v1733) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1737 = stablehlo.broadcast_in_dim %v1736, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v1738 = stablehlo.divide %v1737, %v1734 : tensor<32x96x56x56xf32>
    %v1739 = stablehlo.subtract %v1732, %v1738 : tensor<32x96x56x56xf32>
    %v1740 = stablehlo.multiply %v1739, %v1739 : tensor<32x96x56x56xf32>
    %v1741 = stablehlo.reduce(%v1740 init: %v1733) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1742 = stablehlo.broadcast_in_dim %v1741, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v1743 = stablehlo.divide %v1742, %v1734 : tensor<32x96x56x56xf32>
    %v1744 = stablehlo.add %v1743, %v1735 : tensor<32x96x56x56xf32>
    %v1745 = stablehlo.rsqrt %v1744 : tensor<32x96x56x56xf32>
    %v1746 = stablehlo.multiply %v1739, %v1745 : tensor<32x96x56x56xf32>
    %v1747 = stablehlo.broadcast_in_dim %ge2, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v1748 = stablehlo.multiply %v1747, %v1731 : tensor<32x96x56x56xf32>
    %v1749 = stablehlo.reduce(%v1748 init: %v1733) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1750 = stablehlo.broadcast_in_dim %v1749, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v1751 = stablehlo.multiply %v1746, %v1748 : tensor<32x96x56x56xf32>
    %v1752 = stablehlo.reduce(%v1751 init: %v1733) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1753 = stablehlo.broadcast_in_dim %v1752, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v1754 = stablehlo.multiply %v1748, %v1734 : tensor<32x96x56x56xf32>
    %v1755 = stablehlo.subtract %v1754, %v1750 : tensor<32x96x56x56xf32>
    %v1756 = stablehlo.multiply %v1746, %v1753 : tensor<32x96x56x56xf32>
    %v1757 = stablehlo.subtract %v1755, %v1756 : tensor<32x96x56x56xf32>
    %v1758 = stablehlo.divide %v1745, %v1734 : tensor<32x96x56x56xf32>
    %v1759 = stablehlo.multiply %v1758, %v1757 : tensor<32x96x56x56xf32>
    %v1760 = stablehlo.reshape %v1759 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v1761 = stablehlo.reshape %v1760 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1762 = stablehlo.transpose %We2, dims = [1, 0, 2, 3] : (tensor<96x24x1x1xf32>) -> tensor<24x96x1x1xf32>
    %v1763 = stablehlo.reverse %v1762, dims = [2, 3] : tensor<24x96x1x1xf32>
    %v1764 = stablehlo.convolution(%v1761, %v1763)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<24x96x1x1xf32>) -> tensor<32x24x56x56xf32>
    %v1765 = stablehlo.reshape %v1764 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v1766 = stablehlo.add %v1765, %v1521 : tensor<32x75264xf32>
    %v1767 = stablehlo.reshape %v111 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v1768 = stablehlo.reshape %v1760 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1769 = stablehlo.transpose %v1767, dims = [1, 0, 2, 3] : (tensor<32x24x56x56xf32>) -> tensor<24x32x56x56xf32>
    %v1770 = stablehlo.transpose %v1768, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v1771 = stablehlo.convolution(%v1769, %v1770)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<24x32x56x56xf32>, tensor<96x32x56x56xf32>) -> tensor<24x96x1x1xf32>
    %v1772 = stablehlo.transpose %v1771, dims = [1, 0, 2, 3] : (tensor<24x96x1x1xf32>) -> tensor<96x24x1x1xf32>
    %v1773 = stablehlo.constant dense<0.3> : tensor<96x24x1x1xf32>
    %v1774 = stablehlo.multiply %v1772, %v1773 : tensor<96x24x1x1xf32>
    %v1775 = stablehlo.subtract %We2, %v1774 : tensor<96x24x1x1xf32>
    %v1776 = stablehlo.reshape %v1760 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1777 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1778 = stablehlo.reduce(%v1776 init: %v1777) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v1779 = stablehlo.constant dense<0.3> : tensor<96xf32>
    %v1780 = stablehlo.multiply %v1778, %v1779 : tensor<96xf32>
    %v1781 = stablehlo.subtract %be2, %v1780 : tensor<96xf32>
    %v1782 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1783 = stablehlo.reshape %v116 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1784 = stablehlo.constant dense<3136.0> : tensor<32x96x56x56xf32>
    %v1785 = stablehlo.constant dense<1.0e-5> : tensor<32x96x56x56xf32>
    %v1786 = stablehlo.reduce(%v1783 init: %v1782) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1787 = stablehlo.broadcast_in_dim %v1786, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v1788 = stablehlo.divide %v1787, %v1784 : tensor<32x96x56x56xf32>
    %v1789 = stablehlo.subtract %v1783, %v1788 : tensor<32x96x56x56xf32>
    %v1790 = stablehlo.multiply %v1789, %v1789 : tensor<32x96x56x56xf32>
    %v1791 = stablehlo.reduce(%v1790 init: %v1782) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1792 = stablehlo.broadcast_in_dim %v1791, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v1793 = stablehlo.divide %v1792, %v1784 : tensor<32x96x56x56xf32>
    %v1794 = stablehlo.add %v1793, %v1785 : tensor<32x96x56x56xf32>
    %v1795 = stablehlo.rsqrt %v1794 : tensor<32x96x56x56xf32>
    %v1796 = stablehlo.multiply %v1789, %v1795 : tensor<32x96x56x56xf32>
    %v1797 = stablehlo.reshape %v1730 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1798 = stablehlo.multiply %v1797, %v1796 : tensor<32x96x56x56xf32>
    %v1799 = stablehlo.reduce(%v1798 init: %v1782) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v1800 = stablehlo.constant dense<0.3> : tensor<96xf32>
    %v1801 = stablehlo.multiply %v1799, %v1800 : tensor<96xf32>
    %v1802 = stablehlo.subtract %ge2, %v1801 : tensor<96xf32>
    %v1803 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1804 = stablehlo.reshape %v1730 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1805 = stablehlo.reduce(%v1804 init: %v1803) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v1806 = stablehlo.constant dense<0.3> : tensor<96xf32>
    %v1807 = stablehlo.multiply %v1805, %v1806 : tensor<96xf32>
    %v1808 = stablehlo.subtract %bte2, %v1807 : tensor<96xf32>
    %v1809 = stablehlo.reshape %v140 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1810 = stablehlo.reshape %v1720 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1811 = stablehlo.transpose %v1809, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v1812 = stablehlo.transpose %v1810, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v1813 = stablehlo.convolution(%v1811, %v1812)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 96 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<96x32x56x56xf32>) -> tensor<1x96x3x3xf32>
    %v1814 = stablehlo.reshape %v1813 : (tensor<1x96x3x3xf32>) -> tensor<96x1x3x3xf32>
    %v1815 = stablehlo.constant dense<0.3> : tensor<96x1x3x3xf32>
    %v1816 = stablehlo.multiply %v1814, %v1815 : tensor<96x1x3x3xf32>
    %v1817 = stablehlo.subtract %Wd2, %v1816 : tensor<96x1x3x3xf32>
    %v1818 = stablehlo.reshape %v1720 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1819 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1820 = stablehlo.reduce(%v1818 init: %v1819) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v1821 = stablehlo.constant dense<0.3> : tensor<96xf32>
    %v1822 = stablehlo.multiply %v1820, %v1821 : tensor<96xf32>
    %v1823 = stablehlo.subtract %bd2, %v1822 : tensor<96xf32>
    %v1824 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1825 = stablehlo.reshape %v145 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1826 = stablehlo.constant dense<3136.0> : tensor<32x96x56x56xf32>
    %v1827 = stablehlo.constant dense<1.0e-5> : tensor<32x96x56x56xf32>
    %v1828 = stablehlo.reduce(%v1825 init: %v1824) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1829 = stablehlo.broadcast_in_dim %v1828, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v1830 = stablehlo.divide %v1829, %v1826 : tensor<32x96x56x56xf32>
    %v1831 = stablehlo.subtract %v1825, %v1830 : tensor<32x96x56x56xf32>
    %v1832 = stablehlo.multiply %v1831, %v1831 : tensor<32x96x56x56xf32>
    %v1833 = stablehlo.reduce(%v1832 init: %v1824) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1834 = stablehlo.broadcast_in_dim %v1833, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v1835 = stablehlo.divide %v1834, %v1826 : tensor<32x96x56x56xf32>
    %v1836 = stablehlo.add %v1835, %v1827 : tensor<32x96x56x56xf32>
    %v1837 = stablehlo.rsqrt %v1836 : tensor<32x96x56x56xf32>
    %v1838 = stablehlo.multiply %v1831, %v1837 : tensor<32x96x56x56xf32>
    %v1839 = stablehlo.reshape %v1690 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1840 = stablehlo.multiply %v1839, %v1838 : tensor<32x96x56x56xf32>
    %v1841 = stablehlo.reduce(%v1840 init: %v1824) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v1842 = stablehlo.constant dense<0.3> : tensor<96xf32>
    %v1843 = stablehlo.multiply %v1841, %v1842 : tensor<96xf32>
    %v1844 = stablehlo.subtract %gd2, %v1843 : tensor<96xf32>
    %v1845 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1846 = stablehlo.reshape %v1690 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1847 = stablehlo.reduce(%v1846 init: %v1845) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v1848 = stablehlo.constant dense<0.3> : tensor<96xf32>
    %v1849 = stablehlo.multiply %v1847, %v1848 : tensor<96xf32>
    %v1850 = stablehlo.subtract %btd2, %v1849 : tensor<96xf32>
    %v1851 = stablehlo.reshape %v169 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1852 = stablehlo.reshape %v1679 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v1853 = stablehlo.transpose %v1851, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v1854 = stablehlo.transpose %v1852, dims = [1, 0, 2, 3] : (tensor<32x24x56x56xf32>) -> tensor<24x32x56x56xf32>
    %v1855 = stablehlo.convolution(%v1853, %v1854)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<24x32x56x56xf32>) -> tensor<96x24x1x1xf32>
    %v1856 = stablehlo.transpose %v1855, dims = [1, 0, 2, 3] : (tensor<96x24x1x1xf32>) -> tensor<24x96x1x1xf32>
    %v1857 = stablehlo.constant dense<0.3> : tensor<24x96x1x1xf32>
    %v1858 = stablehlo.multiply %v1856, %v1857 : tensor<24x96x1x1xf32>
    %v1859 = stablehlo.subtract %Wp2, %v1858 : tensor<24x96x1x1xf32>
    %v1860 = stablehlo.reshape %v1679 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v1861 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1862 = stablehlo.reduce(%v1860 init: %v1861) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v1863 = stablehlo.constant dense<0.3> : tensor<24xf32>
    %v1864 = stablehlo.multiply %v1862, %v1863 : tensor<24xf32>
    %v1865 = stablehlo.subtract %bp2, %v1864 : tensor<24xf32>
    %v1866 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1867 = stablehlo.reshape %v174 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v1868 = stablehlo.constant dense<3136.0> : tensor<32x24x56x56xf32>
    %v1869 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v1870 = stablehlo.reduce(%v1867 init: %v1866) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v1871 = stablehlo.broadcast_in_dim %v1870, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v1872 = stablehlo.divide %v1871, %v1868 : tensor<32x24x56x56xf32>
    %v1873 = stablehlo.subtract %v1867, %v1872 : tensor<32x24x56x56xf32>
    %v1874 = stablehlo.multiply %v1873, %v1873 : tensor<32x24x56x56xf32>
    %v1875 = stablehlo.reduce(%v1874 init: %v1866) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v1876 = stablehlo.broadcast_in_dim %v1875, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v1877 = stablehlo.divide %v1876, %v1868 : tensor<32x24x56x56xf32>
    %v1878 = stablehlo.add %v1877, %v1869 : tensor<32x24x56x56xf32>
    %v1879 = stablehlo.rsqrt %v1878 : tensor<32x24x56x56xf32>
    %v1880 = stablehlo.multiply %v1873, %v1879 : tensor<32x24x56x56xf32>
    %v1881 = stablehlo.reshape %v1521 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v1882 = stablehlo.multiply %v1881, %v1880 : tensor<32x24x56x56xf32>
    %v1883 = stablehlo.reduce(%v1882 init: %v1866) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v1884 = stablehlo.constant dense<0.3> : tensor<24xf32>
    %v1885 = stablehlo.multiply %v1883, %v1884 : tensor<24xf32>
    %v1886 = stablehlo.subtract %gp2, %v1885 : tensor<24xf32>
    %v1887 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1888 = stablehlo.reshape %v1521 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v1889 = stablehlo.reduce(%v1888 init: %v1887) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v1890 = stablehlo.constant dense<0.3> : tensor<24xf32>
    %v1891 = stablehlo.multiply %v1889, %v1890 : tensor<24xf32>
    %v1892 = stablehlo.subtract %btp2, %v1891 : tensor<24xf32>
    %v1893 = stablehlo.reshape %v1766 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v1894 = stablehlo.reshape %v91 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v1895 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1896 = stablehlo.constant dense<3136.0> : tensor<32x24x56x56xf32>
    %v1897 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v1898 = stablehlo.reduce(%v1894 init: %v1895) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v1899 = stablehlo.broadcast_in_dim %v1898, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v1900 = stablehlo.divide %v1899, %v1896 : tensor<32x24x56x56xf32>
    %v1901 = stablehlo.subtract %v1894, %v1900 : tensor<32x24x56x56xf32>
    %v1902 = stablehlo.multiply %v1901, %v1901 : tensor<32x24x56x56xf32>
    %v1903 = stablehlo.reduce(%v1902 init: %v1895) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v1904 = stablehlo.broadcast_in_dim %v1903, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v1905 = stablehlo.divide %v1904, %v1896 : tensor<32x24x56x56xf32>
    %v1906 = stablehlo.add %v1905, %v1897 : tensor<32x24x56x56xf32>
    %v1907 = stablehlo.rsqrt %v1906 : tensor<32x24x56x56xf32>
    %v1908 = stablehlo.multiply %v1901, %v1907 : tensor<32x24x56x56xf32>
    %v1909 = stablehlo.broadcast_in_dim %gp1, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v1910 = stablehlo.multiply %v1909, %v1893 : tensor<32x24x56x56xf32>
    %v1911 = stablehlo.reduce(%v1910 init: %v1895) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v1912 = stablehlo.broadcast_in_dim %v1911, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v1913 = stablehlo.multiply %v1908, %v1910 : tensor<32x24x56x56xf32>
    %v1914 = stablehlo.reduce(%v1913 init: %v1895) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v1915 = stablehlo.broadcast_in_dim %v1914, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v1916 = stablehlo.multiply %v1910, %v1896 : tensor<32x24x56x56xf32>
    %v1917 = stablehlo.subtract %v1916, %v1912 : tensor<32x24x56x56xf32>
    %v1918 = stablehlo.multiply %v1908, %v1915 : tensor<32x24x56x56xf32>
    %v1919 = stablehlo.subtract %v1917, %v1918 : tensor<32x24x56x56xf32>
    %v1920 = stablehlo.divide %v1907, %v1896 : tensor<32x24x56x56xf32>
    %v1921 = stablehlo.multiply %v1920, %v1919 : tensor<32x24x56x56xf32>
    %v1922 = stablehlo.reshape %v1921 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v1923 = stablehlo.reshape %v1922 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v1924 = stablehlo.transpose %Wp1, dims = [1, 0, 2, 3] : (tensor<24x64x1x1xf32>) -> tensor<64x24x1x1xf32>
    %v1925 = stablehlo.reverse %v1924, dims = [2, 3] : tensor<64x24x1x1xf32>
    %v1926 = stablehlo.convolution(%v1923, %v1925)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<64x24x1x1xf32>) -> tensor<32x64x56x56xf32>
    %v1927 = stablehlo.reshape %v1926 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v1928 = stablehlo.constant dense<0.0> : tensor<32x200704xf32>
    %v1929 = stablehlo.constant dense<6.0> : tensor<32x200704xf32>
    %v1930 = stablehlo.compare GT, %v82, %v1928 : (tensor<32x200704xf32>, tensor<32x200704xf32>) -> tensor<32x200704xi1>
    %v1931 = stablehlo.compare LT, %v82, %v1929 : (tensor<32x200704xf32>, tensor<32x200704xf32>) -> tensor<32x200704xi1>
    %v1932 = stablehlo.and %v1930, %v1931 : tensor<32x200704xi1>
    %v1933 = stablehlo.select %v1932, %v1927, %v1928 : tensor<32x200704xi1>, tensor<32x200704xf32>
    %v1934 = stablehlo.reshape %v1933 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v1935 = stablehlo.reshape %v62 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v1936 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1937 = stablehlo.constant dense<3136.0> : tensor<32x64x56x56xf32>
    %v1938 = stablehlo.constant dense<1.0e-5> : tensor<32x64x56x56xf32>
    %v1939 = stablehlo.reduce(%v1935 init: %v1936) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v1940 = stablehlo.broadcast_in_dim %v1939, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v1941 = stablehlo.divide %v1940, %v1937 : tensor<32x64x56x56xf32>
    %v1942 = stablehlo.subtract %v1935, %v1941 : tensor<32x64x56x56xf32>
    %v1943 = stablehlo.multiply %v1942, %v1942 : tensor<32x64x56x56xf32>
    %v1944 = stablehlo.reduce(%v1943 init: %v1936) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v1945 = stablehlo.broadcast_in_dim %v1944, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v1946 = stablehlo.divide %v1945, %v1937 : tensor<32x64x56x56xf32>
    %v1947 = stablehlo.add %v1946, %v1938 : tensor<32x64x56x56xf32>
    %v1948 = stablehlo.rsqrt %v1947 : tensor<32x64x56x56xf32>
    %v1949 = stablehlo.multiply %v1942, %v1948 : tensor<32x64x56x56xf32>
    %v1950 = stablehlo.broadcast_in_dim %gd1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v1951 = stablehlo.multiply %v1950, %v1934 : tensor<32x64x56x56xf32>
    %v1952 = stablehlo.reduce(%v1951 init: %v1936) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v1953 = stablehlo.broadcast_in_dim %v1952, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v1954 = stablehlo.multiply %v1949, %v1951 : tensor<32x64x56x56xf32>
    %v1955 = stablehlo.reduce(%v1954 init: %v1936) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v1956 = stablehlo.broadcast_in_dim %v1955, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v1957 = stablehlo.multiply %v1951, %v1937 : tensor<32x64x56x56xf32>
    %v1958 = stablehlo.subtract %v1957, %v1953 : tensor<32x64x56x56xf32>
    %v1959 = stablehlo.multiply %v1949, %v1956 : tensor<32x64x56x56xf32>
    %v1960 = stablehlo.subtract %v1958, %v1959 : tensor<32x64x56x56xf32>
    %v1961 = stablehlo.divide %v1948, %v1937 : tensor<32x64x56x56xf32>
    %v1962 = stablehlo.multiply %v1961, %v1960 : tensor<32x64x56x56xf32>
    %v1963 = stablehlo.reshape %v1962 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v1964 = stablehlo.reshape %v1963 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v1965 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1966 = stablehlo.pad %v1964, %v1965, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64x112x112xf32>
    %v1967 = stablehlo.reverse %Wd1, dims = [2, 3] : tensor<64x1x3x3xf32>
    %v1968 = stablehlo.convolution(%v1966, %v1967)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 64 : i64} : (tensor<32x64x112x112xf32>, tensor<64x1x3x3xf32>) -> tensor<32x64x112x112xf32>
    %v1969 = stablehlo.reshape %v1968 : (tensor<32x64x112x112xf32>) -> tensor<32x802816xf32>
    %v1970 = stablehlo.constant dense<0.0> : tensor<32x802816xf32>
    %v1971 = stablehlo.constant dense<6.0> : tensor<32x802816xf32>
    %v1972 = stablehlo.compare GT, %v53, %v1970 : (tensor<32x802816xf32>, tensor<32x802816xf32>) -> tensor<32x802816xi1>
    %v1973 = stablehlo.compare LT, %v53, %v1971 : (tensor<32x802816xf32>, tensor<32x802816xf32>) -> tensor<32x802816xi1>
    %v1974 = stablehlo.and %v1972, %v1973 : tensor<32x802816xi1>
    %v1975 = stablehlo.select %v1974, %v1969, %v1970 : tensor<32x802816xi1>, tensor<32x802816xf32>
    %v1976 = stablehlo.reshape %v1975 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v1977 = stablehlo.reshape %v33 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v1978 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1979 = stablehlo.constant dense<12544.0> : tensor<32x64x112x112xf32>
    %v1980 = stablehlo.constant dense<1.0e-5> : tensor<32x64x112x112xf32>
    %v1981 = stablehlo.reduce(%v1977 init: %v1978) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v1982 = stablehlo.broadcast_in_dim %v1981, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x112x112xf32>
    %v1983 = stablehlo.divide %v1982, %v1979 : tensor<32x64x112x112xf32>
    %v1984 = stablehlo.subtract %v1977, %v1983 : tensor<32x64x112x112xf32>
    %v1985 = stablehlo.multiply %v1984, %v1984 : tensor<32x64x112x112xf32>
    %v1986 = stablehlo.reduce(%v1985 init: %v1978) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v1987 = stablehlo.broadcast_in_dim %v1986, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x112x112xf32>
    %v1988 = stablehlo.divide %v1987, %v1979 : tensor<32x64x112x112xf32>
    %v1989 = stablehlo.add %v1988, %v1980 : tensor<32x64x112x112xf32>
    %v1990 = stablehlo.rsqrt %v1989 : tensor<32x64x112x112xf32>
    %v1991 = stablehlo.multiply %v1984, %v1990 : tensor<32x64x112x112xf32>
    %v1992 = stablehlo.broadcast_in_dim %ge1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v1993 = stablehlo.multiply %v1992, %v1976 : tensor<32x64x112x112xf32>
    %v1994 = stablehlo.reduce(%v1993 init: %v1978) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v1995 = stablehlo.broadcast_in_dim %v1994, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x112x112xf32>
    %v1996 = stablehlo.multiply %v1991, %v1993 : tensor<32x64x112x112xf32>
    %v1997 = stablehlo.reduce(%v1996 init: %v1978) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v1998 = stablehlo.broadcast_in_dim %v1997, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x112x112xf32>
    %v1999 = stablehlo.multiply %v1993, %v1979 : tensor<32x64x112x112xf32>
    %v2000 = stablehlo.subtract %v1999, %v1995 : tensor<32x64x112x112xf32>
    %v2001 = stablehlo.multiply %v1991, %v1998 : tensor<32x64x112x112xf32>
    %v2002 = stablehlo.subtract %v2000, %v2001 : tensor<32x64x112x112xf32>
    %v2003 = stablehlo.divide %v1990, %v1979 : tensor<32x64x112x112xf32>
    %v2004 = stablehlo.multiply %v2003, %v2002 : tensor<32x64x112x112xf32>
    %v2005 = stablehlo.reshape %v2004 : (tensor<32x64x112x112xf32>) -> tensor<32x802816xf32>
    %v2006 = stablehlo.reshape %v2005 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v2007 = stablehlo.transpose %We1, dims = [1, 0, 2, 3] : (tensor<64x16x1x1xf32>) -> tensor<16x64x1x1xf32>
    %v2008 = stablehlo.reverse %v2007, dims = [2, 3] : tensor<16x64x1x1xf32>
    %v2009 = stablehlo.convolution(%v2006, %v2008)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x112x112xf32>, tensor<16x64x1x1xf32>) -> tensor<32x16x112x112xf32>
    %v2010 = stablehlo.reshape %v2009 : (tensor<32x16x112x112xf32>) -> tensor<32x200704xf32>
    %v2011 = stablehlo.reshape %v28 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v2012 = stablehlo.reshape %v2005 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v2013 = stablehlo.transpose %v2011, dims = [1, 0, 2, 3] : (tensor<32x16x112x112xf32>) -> tensor<16x32x112x112xf32>
    %v2014 = stablehlo.transpose %v2012, dims = [1, 0, 2, 3] : (tensor<32x64x112x112xf32>) -> tensor<64x32x112x112xf32>
    %v2015 = stablehlo.convolution(%v2013, %v2014)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x32x112x112xf32>, tensor<64x32x112x112xf32>) -> tensor<16x64x1x1xf32>
    %v2016 = stablehlo.transpose %v2015, dims = [1, 0, 2, 3] : (tensor<16x64x1x1xf32>) -> tensor<64x16x1x1xf32>
    %v2017 = stablehlo.constant dense<0.3> : tensor<64x16x1x1xf32>
    %v2018 = stablehlo.multiply %v2016, %v2017 : tensor<64x16x1x1xf32>
    %v2019 = stablehlo.subtract %We1, %v2018 : tensor<64x16x1x1xf32>
    %v2020 = stablehlo.reshape %v2005 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v2021 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2022 = stablehlo.reduce(%v2020 init: %v2021) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v2023 = stablehlo.constant dense<0.3> : tensor<64xf32>
    %v2024 = stablehlo.multiply %v2022, %v2023 : tensor<64xf32>
    %v2025 = stablehlo.subtract %be1, %v2024 : tensor<64xf32>
    %v2026 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2027 = stablehlo.reshape %v33 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v2028 = stablehlo.constant dense<12544.0> : tensor<32x64x112x112xf32>
    %v2029 = stablehlo.constant dense<1.0e-5> : tensor<32x64x112x112xf32>
    %v2030 = stablehlo.reduce(%v2027 init: %v2026) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v2031 = stablehlo.broadcast_in_dim %v2030, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x112x112xf32>
    %v2032 = stablehlo.divide %v2031, %v2028 : tensor<32x64x112x112xf32>
    %v2033 = stablehlo.subtract %v2027, %v2032 : tensor<32x64x112x112xf32>
    %v2034 = stablehlo.multiply %v2033, %v2033 : tensor<32x64x112x112xf32>
    %v2035 = stablehlo.reduce(%v2034 init: %v2026) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v2036 = stablehlo.broadcast_in_dim %v2035, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x112x112xf32>
    %v2037 = stablehlo.divide %v2036, %v2028 : tensor<32x64x112x112xf32>
    %v2038 = stablehlo.add %v2037, %v2029 : tensor<32x64x112x112xf32>
    %v2039 = stablehlo.rsqrt %v2038 : tensor<32x64x112x112xf32>
    %v2040 = stablehlo.multiply %v2033, %v2039 : tensor<32x64x112x112xf32>
    %v2041 = stablehlo.reshape %v1975 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v2042 = stablehlo.multiply %v2041, %v2040 : tensor<32x64x112x112xf32>
    %v2043 = stablehlo.reduce(%v2042 init: %v2026) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v2044 = stablehlo.constant dense<0.3> : tensor<64xf32>
    %v2045 = stablehlo.multiply %v2043, %v2044 : tensor<64xf32>
    %v2046 = stablehlo.subtract %ge1, %v2045 : tensor<64xf32>
    %v2047 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2048 = stablehlo.reshape %v1975 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v2049 = stablehlo.reduce(%v2048 init: %v2047) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v2050 = stablehlo.constant dense<0.3> : tensor<64xf32>
    %v2051 = stablehlo.multiply %v2049, %v2050 : tensor<64xf32>
    %v2052 = stablehlo.subtract %bte1, %v2051 : tensor<64xf32>
    %v2053 = stablehlo.reshape %v57 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v2054 = stablehlo.reshape %v1963 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v2055 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2056 = stablehlo.pad %v2054, %v2055, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64x112x112xf32>
    %v2057 = stablehlo.transpose %v2053, dims = [1, 0, 2, 3] : (tensor<32x64x112x112xf32>) -> tensor<64x32x112x112xf32>
    %v2058 = stablehlo.transpose %v2056, dims = [1, 0, 2, 3] : (tensor<32x64x112x112xf32>) -> tensor<64x32x112x112xf32>
    %v2059 = stablehlo.convolution(%v2057, %v2058)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 64 : i64, feature_group_count = 1 : i64} : (tensor<64x32x112x112xf32>, tensor<64x32x112x112xf32>) -> tensor<1x64x3x3xf32>
    %v2060 = stablehlo.reshape %v2059 : (tensor<1x64x3x3xf32>) -> tensor<64x1x3x3xf32>
    %v2061 = stablehlo.constant dense<0.3> : tensor<64x1x3x3xf32>
    %v2062 = stablehlo.multiply %v2060, %v2061 : tensor<64x1x3x3xf32>
    %v2063 = stablehlo.subtract %Wd1, %v2062 : tensor<64x1x3x3xf32>
    %v2064 = stablehlo.reshape %v1963 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v2065 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2066 = stablehlo.reduce(%v2064 init: %v2065) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v2067 = stablehlo.constant dense<0.3> : tensor<64xf32>
    %v2068 = stablehlo.multiply %v2066, %v2067 : tensor<64xf32>
    %v2069 = stablehlo.subtract %bd1, %v2068 : tensor<64xf32>
    %v2070 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2071 = stablehlo.reshape %v62 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v2072 = stablehlo.constant dense<3136.0> : tensor<32x64x56x56xf32>
    %v2073 = stablehlo.constant dense<1.0e-5> : tensor<32x64x56x56xf32>
    %v2074 = stablehlo.reduce(%v2071 init: %v2070) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v2075 = stablehlo.broadcast_in_dim %v2074, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v2076 = stablehlo.divide %v2075, %v2072 : tensor<32x64x56x56xf32>
    %v2077 = stablehlo.subtract %v2071, %v2076 : tensor<32x64x56x56xf32>
    %v2078 = stablehlo.multiply %v2077, %v2077 : tensor<32x64x56x56xf32>
    %v2079 = stablehlo.reduce(%v2078 init: %v2070) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v2080 = stablehlo.broadcast_in_dim %v2079, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v2081 = stablehlo.divide %v2080, %v2072 : tensor<32x64x56x56xf32>
    %v2082 = stablehlo.add %v2081, %v2073 : tensor<32x64x56x56xf32>
    %v2083 = stablehlo.rsqrt %v2082 : tensor<32x64x56x56xf32>
    %v2084 = stablehlo.multiply %v2077, %v2083 : tensor<32x64x56x56xf32>
    %v2085 = stablehlo.reshape %v1933 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v2086 = stablehlo.multiply %v2085, %v2084 : tensor<32x64x56x56xf32>
    %v2087 = stablehlo.reduce(%v2086 init: %v2070) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v2088 = stablehlo.constant dense<0.3> : tensor<64xf32>
    %v2089 = stablehlo.multiply %v2087, %v2088 : tensor<64xf32>
    %v2090 = stablehlo.subtract %gd1, %v2089 : tensor<64xf32>
    %v2091 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2092 = stablehlo.reshape %v1933 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v2093 = stablehlo.reduce(%v2092 init: %v2091) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v2094 = stablehlo.constant dense<0.3> : tensor<64xf32>
    %v2095 = stablehlo.multiply %v2093, %v2094 : tensor<64xf32>
    %v2096 = stablehlo.subtract %btd1, %v2095 : tensor<64xf32>
    %v2097 = stablehlo.reshape %v86 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v2098 = stablehlo.reshape %v1922 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v2099 = stablehlo.transpose %v2097, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v2100 = stablehlo.transpose %v2098, dims = [1, 0, 2, 3] : (tensor<32x24x56x56xf32>) -> tensor<24x32x56x56xf32>
    %v2101 = stablehlo.convolution(%v2099, %v2100)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<24x32x56x56xf32>) -> tensor<64x24x1x1xf32>
    %v2102 = stablehlo.transpose %v2101, dims = [1, 0, 2, 3] : (tensor<64x24x1x1xf32>) -> tensor<24x64x1x1xf32>
    %v2103 = stablehlo.constant dense<0.3> : tensor<24x64x1x1xf32>
    %v2104 = stablehlo.multiply %v2102, %v2103 : tensor<24x64x1x1xf32>
    %v2105 = stablehlo.subtract %Wp1, %v2104 : tensor<24x64x1x1xf32>
    %v2106 = stablehlo.reshape %v1922 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v2107 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2108 = stablehlo.reduce(%v2106 init: %v2107) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v2109 = stablehlo.constant dense<0.3> : tensor<24xf32>
    %v2110 = stablehlo.multiply %v2108, %v2109 : tensor<24xf32>
    %v2111 = stablehlo.subtract %bp1, %v2110 : tensor<24xf32>
    %v2112 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2113 = stablehlo.reshape %v91 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v2114 = stablehlo.constant dense<3136.0> : tensor<32x24x56x56xf32>
    %v2115 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v2116 = stablehlo.reduce(%v2113 init: %v2112) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v2117 = stablehlo.broadcast_in_dim %v2116, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v2118 = stablehlo.divide %v2117, %v2114 : tensor<32x24x56x56xf32>
    %v2119 = stablehlo.subtract %v2113, %v2118 : tensor<32x24x56x56xf32>
    %v2120 = stablehlo.multiply %v2119, %v2119 : tensor<32x24x56x56xf32>
    %v2121 = stablehlo.reduce(%v2120 init: %v2112) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v2122 = stablehlo.broadcast_in_dim %v2121, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v2123 = stablehlo.divide %v2122, %v2114 : tensor<32x24x56x56xf32>
    %v2124 = stablehlo.add %v2123, %v2115 : tensor<32x24x56x56xf32>
    %v2125 = stablehlo.rsqrt %v2124 : tensor<32x24x56x56xf32>
    %v2126 = stablehlo.multiply %v2119, %v2125 : tensor<32x24x56x56xf32>
    %v2127 = stablehlo.reshape %v1766 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v2128 = stablehlo.multiply %v2127, %v2126 : tensor<32x24x56x56xf32>
    %v2129 = stablehlo.reduce(%v2128 init: %v2112) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v2130 = stablehlo.constant dense<0.3> : tensor<24xf32>
    %v2131 = stablehlo.multiply %v2129, %v2130 : tensor<24xf32>
    %v2132 = stablehlo.subtract %gp1, %v2131 : tensor<24xf32>
    %v2133 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2134 = stablehlo.reshape %v1766 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v2135 = stablehlo.reduce(%v2134 init: %v2133) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v2136 = stablehlo.constant dense<0.3> : tensor<24xf32>
    %v2137 = stablehlo.multiply %v2135, %v2136 : tensor<24xf32>
    %v2138 = stablehlo.subtract %btp1, %v2137 : tensor<24xf32>
    %v2139 = stablehlo.constant dense<0.0> : tensor<32x200704xf32>
    %v2140 = stablehlo.constant dense<6.0> : tensor<32x200704xf32>
    %v2141 = stablehlo.compare GT, %v24, %v2139 : (tensor<32x200704xf32>, tensor<32x200704xf32>) -> tensor<32x200704xi1>
    %v2142 = stablehlo.compare LT, %v24, %v2140 : (tensor<32x200704xf32>, tensor<32x200704xf32>) -> tensor<32x200704xi1>
    %v2143 = stablehlo.and %v2141, %v2142 : tensor<32x200704xi1>
    %v2144 = stablehlo.select %v2143, %v2010, %v2139 : tensor<32x200704xi1>, tensor<32x200704xf32>
    %v2145 = stablehlo.reshape %v2144 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v2146 = stablehlo.reshape %v4 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v2147 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2148 = stablehlo.constant dense<12544.0> : tensor<32x16x112x112xf32>
    %v2149 = stablehlo.constant dense<1.0e-5> : tensor<32x16x112x112xf32>
    %v2150 = stablehlo.reduce(%v2146 init: %v2147) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<32x16xf32>
    %v2151 = stablehlo.broadcast_in_dim %v2150, dims = [0, 1] : (tensor<32x16xf32>) -> tensor<32x16x112x112xf32>
    %v2152 = stablehlo.divide %v2151, %v2148 : tensor<32x16x112x112xf32>
    %v2153 = stablehlo.subtract %v2146, %v2152 : tensor<32x16x112x112xf32>
    %v2154 = stablehlo.multiply %v2153, %v2153 : tensor<32x16x112x112xf32>
    %v2155 = stablehlo.reduce(%v2154 init: %v2147) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<32x16xf32>
    %v2156 = stablehlo.broadcast_in_dim %v2155, dims = [0, 1] : (tensor<32x16xf32>) -> tensor<32x16x112x112xf32>
    %v2157 = stablehlo.divide %v2156, %v2148 : tensor<32x16x112x112xf32>
    %v2158 = stablehlo.add %v2157, %v2149 : tensor<32x16x112x112xf32>
    %v2159 = stablehlo.rsqrt %v2158 : tensor<32x16x112x112xf32>
    %v2160 = stablehlo.multiply %v2153, %v2159 : tensor<32x16x112x112xf32>
    %v2161 = stablehlo.broadcast_in_dim %gs, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v2162 = stablehlo.multiply %v2161, %v2145 : tensor<32x16x112x112xf32>
    %v2163 = stablehlo.reduce(%v2162 init: %v2147) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<32x16xf32>
    %v2164 = stablehlo.broadcast_in_dim %v2163, dims = [0, 1] : (tensor<32x16xf32>) -> tensor<32x16x112x112xf32>
    %v2165 = stablehlo.multiply %v2160, %v2162 : tensor<32x16x112x112xf32>
    %v2166 = stablehlo.reduce(%v2165 init: %v2147) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<32x16xf32>
    %v2167 = stablehlo.broadcast_in_dim %v2166, dims = [0, 1] : (tensor<32x16xf32>) -> tensor<32x16x112x112xf32>
    %v2168 = stablehlo.multiply %v2162, %v2148 : tensor<32x16x112x112xf32>
    %v2169 = stablehlo.subtract %v2168, %v2164 : tensor<32x16x112x112xf32>
    %v2170 = stablehlo.multiply %v2160, %v2167 : tensor<32x16x112x112xf32>
    %v2171 = stablehlo.subtract %v2169, %v2170 : tensor<32x16x112x112xf32>
    %v2172 = stablehlo.divide %v2159, %v2148 : tensor<32x16x112x112xf32>
    %v2173 = stablehlo.multiply %v2172, %v2171 : tensor<32x16x112x112xf32>
    %v2174 = stablehlo.reshape %v2173 : (tensor<32x16x112x112xf32>) -> tensor<32x200704xf32>
    %v2175 = stablehlo.reshape %x : (tensor<32x150528xf32>) -> tensor<32x3x224x224xf32>
    %v2176 = stablehlo.reshape %v2174 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v2177 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2178 = stablehlo.pad %v2176, %v2177, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<32x16x224x224xf32>
    %v2179 = stablehlo.transpose %v2175, dims = [1, 0, 2, 3] : (tensor<32x3x224x224xf32>) -> tensor<3x32x224x224xf32>
    %v2180 = stablehlo.transpose %v2178, dims = [1, 0, 2, 3] : (tensor<32x16x224x224xf32>) -> tensor<16x32x224x224xf32>
    %v2181 = stablehlo.convolution(%v2179, %v2180)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x32x224x224xf32>, tensor<16x32x224x224xf32>) -> tensor<3x16x3x3xf32>
    %v2182 = stablehlo.transpose %v2181, dims = [1, 0, 2, 3] : (tensor<3x16x3x3xf32>) -> tensor<16x3x3x3xf32>
    %v2183 = stablehlo.constant dense<0.3> : tensor<16x3x3x3xf32>
    %v2184 = stablehlo.multiply %v2182, %v2183 : tensor<16x3x3x3xf32>
    %v2185 = stablehlo.subtract %Ws, %v2184 : tensor<16x3x3x3xf32>
    %v2186 = stablehlo.reshape %v2174 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v2187 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2188 = stablehlo.reduce(%v2186 init: %v2187) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v2189 = stablehlo.constant dense<0.3> : tensor<16xf32>
    %v2190 = stablehlo.multiply %v2188, %v2189 : tensor<16xf32>
    %v2191 = stablehlo.subtract %bs, %v2190 : tensor<16xf32>
    %v2192 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2193 = stablehlo.reshape %v4 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v2194 = stablehlo.constant dense<12544.0> : tensor<32x16x112x112xf32>
    %v2195 = stablehlo.constant dense<1.0e-5> : tensor<32x16x112x112xf32>
    %v2196 = stablehlo.reduce(%v2193 init: %v2192) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<32x16xf32>
    %v2197 = stablehlo.broadcast_in_dim %v2196, dims = [0, 1] : (tensor<32x16xf32>) -> tensor<32x16x112x112xf32>
    %v2198 = stablehlo.divide %v2197, %v2194 : tensor<32x16x112x112xf32>
    %v2199 = stablehlo.subtract %v2193, %v2198 : tensor<32x16x112x112xf32>
    %v2200 = stablehlo.multiply %v2199, %v2199 : tensor<32x16x112x112xf32>
    %v2201 = stablehlo.reduce(%v2200 init: %v2192) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<32x16xf32>
    %v2202 = stablehlo.broadcast_in_dim %v2201, dims = [0, 1] : (tensor<32x16xf32>) -> tensor<32x16x112x112xf32>
    %v2203 = stablehlo.divide %v2202, %v2194 : tensor<32x16x112x112xf32>
    %v2204 = stablehlo.add %v2203, %v2195 : tensor<32x16x112x112xf32>
    %v2205 = stablehlo.rsqrt %v2204 : tensor<32x16x112x112xf32>
    %v2206 = stablehlo.multiply %v2199, %v2205 : tensor<32x16x112x112xf32>
    %v2207 = stablehlo.reshape %v2144 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v2208 = stablehlo.multiply %v2207, %v2206 : tensor<32x16x112x112xf32>
    %v2209 = stablehlo.reduce(%v2208 init: %v2192) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v2210 = stablehlo.constant dense<0.3> : tensor<16xf32>
    %v2211 = stablehlo.multiply %v2209, %v2210 : tensor<16xf32>
    %v2212 = stablehlo.subtract %gs, %v2211 : tensor<16xf32>
    %v2213 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2214 = stablehlo.reshape %v2144 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v2215 = stablehlo.reduce(%v2214 init: %v2213) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v2216 = stablehlo.constant dense<0.3> : tensor<16xf32>
    %v2217 = stablehlo.multiply %v2215, %v2216 : tensor<16xf32>
    %v2218 = stablehlo.subtract %bts, %v2217 : tensor<16xf32>
    return %v2185, %v2191, %v2212, %v2218, %v2019, %v2025, %v2046, %v2052, %v2063, %v2069, %v2090, %v2096, %v2105, %v2111, %v2132, %v2138, %v1775, %v1781, %v1802, %v1808, %v1817, %v1823, %v1844, %v1850, %v1859, %v1865, %v1886, %v1892, %v1530, %v1536, %v1557, %v1563, %v1574, %v1580, %v1601, %v1607, %v1616, %v1622, %v1643, %v1649, %v1286, %v1292, %v1313, %v1319, %v1328, %v1334, %v1355, %v1361, %v1370, %v1376, %v1397, %v1403, %v1041, %v1047, %v1068, %v1074, %v1085, %v1091, %v1112, %v1118, %v1127, %v1133, %v1154, %v1160, %v795, %v801, %v822, %v828, %v839, %v845, %v866, %v872, %v881, %v887, %v908, %v914, %v635, %v641, %v662, %v668, %v580, %v585 : tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<64x16x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x1x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<24x64x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<96x24x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x3x3xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<24x96x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<96x24x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x3x3xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<32x96x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x32x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x1x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<32x128x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x32x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x1x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<64x128x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x1x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x10xf32>, tensor<10xf32>
  }
}

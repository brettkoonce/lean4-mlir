module @m {
  func.func @mobilenetv2_train_step(%x: tensor<32x150528xf32>, %Ws: tensor<16x3x3x3xf32>, %gs: tensor<16xf32>, %bts: tensor<16xf32>, %We1: tensor<64x16x1x1xf32>, %ge1: tensor<64xf32>, %bte1: tensor<64xf32>, %Wd1: tensor<64x1x3x3xf32>, %gd1: tensor<64xf32>, %btd1: tensor<64xf32>, %Wp1: tensor<24x64x1x1xf32>, %gp1: tensor<24xf32>, %btp1: tensor<24xf32>, %We2: tensor<96x24x1x1xf32>, %ge2: tensor<96xf32>, %bte2: tensor<96xf32>, %Wd2: tensor<96x1x3x3xf32>, %gd2: tensor<96xf32>, %btd2: tensor<96xf32>, %Wp2: tensor<24x96x1x1xf32>, %gp2: tensor<24xf32>, %btp2: tensor<24xf32>, %We3: tensor<96x24x1x1xf32>, %ge3: tensor<96xf32>, %bte3: tensor<96xf32>, %Wd3: tensor<96x1x3x3xf32>, %gd3: tensor<96xf32>, %btd3: tensor<96xf32>, %Wp3: tensor<32x96x1x1xf32>, %gp3: tensor<32xf32>, %btp3: tensor<32xf32>, %We4: tensor<128x32x1x1xf32>, %ge4: tensor<128xf32>, %bte4: tensor<128xf32>, %Wd4: tensor<128x1x3x3xf32>, %gd4: tensor<128xf32>, %btd4: tensor<128xf32>, %Wp4: tensor<32x128x1x1xf32>, %gp4: tensor<32xf32>, %btp4: tensor<32xf32>, %We5: tensor<128x32x1x1xf32>, %ge5: tensor<128xf32>, %bte5: tensor<128xf32>, %Wd5: tensor<128x1x3x3xf32>, %gd5: tensor<128xf32>, %btd5: tensor<128xf32>, %Wp5: tensor<64x128x1x1xf32>, %gp5: tensor<64xf32>, %btp5: tensor<64xf32>, %We6: tensor<256x64x1x1xf32>, %ge6: tensor<256xf32>, %bte6: tensor<256xf32>, %Wd6: tensor<256x1x3x3xf32>, %gd6: tensor<256xf32>, %btd6: tensor<256xf32>, %Wp6: tensor<64x256x1x1xf32>, %gp6: tensor<64xf32>, %btp6: tensor<64xf32>, %Wh: tensor<128x64x1x1xf32>, %gh: tensor<128xf32>, %bth: tensor<128xf32>, %Wfc: tensor<128x10xf32>, %bfc: tensor<10xf32>, %onehot: tensor<32x10xf32>) -> (tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<64x16x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x1x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<24x64x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<96x24x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x3x3xf32>, tensor<96xf32>, tensor<96xf32>, tensor<24x96x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<96x24x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x3x3xf32>, tensor<96xf32>, tensor<96xf32>, tensor<32x96x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x32x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x1x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<32x128x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x32x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x1x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<64x128x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x1x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x10xf32>, tensor<10xf32>) {
    // ── MobileNetV2 train step: every line is pretty(verified AST node) ──
    // §2l step B: the conv biases are gone from the signature (BN removes them; He et al.'s
    // `.convBn` has none). The proven conv ops still take a bias operand, so it is bound to a
    // zero constant here — same op, `bias = 0`, and `x + 0.0` is exact.
    %zb16 = stablehlo.constant dense<0.0> : tensor<16xf32>
    %zb24 = stablehlo.constant dense<0.0> : tensor<24xf32>
    %zb32 = stablehlo.constant dense<0.0> : tensor<32xf32>
    %zb64 = stablehlo.constant dense<0.0> : tensor<64xf32>
    %zb96 = stablehlo.constant dense<0.0> : tensor<96xf32>
    %zb128 = stablehlo.constant dense<0.0> : tensor<128xf32>
    %zb144 = stablehlo.constant dense<0.0> : tensor<144xf32>
    %zb160 = stablehlo.constant dense<0.0> : tensor<160xf32>
    %zb192 = stablehlo.constant dense<0.0> : tensor<192xf32>
    %zb256 = stablehlo.constant dense<0.0> : tensor<256xf32>
    %zb320 = stablehlo.constant dense<0.0> : tensor<320xf32>
    %zb384 = stablehlo.constant dense<0.0> : tensor<384xf32>
    %zb576 = stablehlo.constant dense<0.0> : tensor<576xf32>
    %zb960 = stablehlo.constant dense<0.0> : tensor<960xf32>
    %zb1280 = stablehlo.constant dense<0.0> : tensor<1280xf32>
    %v0 = stablehlo.reshape %x : (tensor<32x150528xf32>) -> tensor<32x3x224x224xf32>
    %v1 = stablehlo.convolution(%v0, %Ws)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3x224x224xf32>, tensor<16x3x3x3xf32>) -> tensor<32x16x112x112xf32>
    %v2 = stablehlo.broadcast_in_dim %zb16, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
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
    %v31 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
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
    %v60 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
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
    %v89 = stablehlo.broadcast_in_dim %zb24, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
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
    %v114 = stablehlo.broadcast_in_dim %zb96, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
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
    %v143 = stablehlo.broadcast_in_dim %zb96, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
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
    %v172 = stablehlo.broadcast_in_dim %zb24, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
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
    %v198 = stablehlo.broadcast_in_dim %zb96, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
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
    %v227 = stablehlo.broadcast_in_dim %zb96, dims = [1] : (tensor<96xf32>) -> tensor<32x96x28x28xf32>
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
    %v256 = stablehlo.broadcast_in_dim %zb32, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
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
    %v281 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
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
    %v310 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
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
    %v339 = stablehlo.broadcast_in_dim %zb32, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
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
    %v365 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
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
    %v394 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x14x14xf32>
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
    %v423 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
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
    %v448 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
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
    %v477 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
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
    %v506 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x7x7xf32>
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
    %v531 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x7x7xf32>
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
    %v636 = stablehlo.constant dense<0.0> : tensor<f32>
    %v637 = stablehlo.reshape %v533 : (tensor<32x6272xf32>) -> tensor<32x128x7x7xf32>
    %v638 = stablehlo.constant dense<49.0> : tensor<32x128x7x7xf32>
    %v639 = stablehlo.constant dense<1.0e-5> : tensor<32x128x7x7xf32>
    %v640 = stablehlo.reduce(%v637 init: %v636) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x7x7xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v641 = stablehlo.broadcast_in_dim %v640, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x7x7xf32>
    %v642 = stablehlo.divide %v641, %v638 : tensor<32x128x7x7xf32>
    %v643 = stablehlo.subtract %v637, %v642 : tensor<32x128x7x7xf32>
    %v644 = stablehlo.multiply %v643, %v643 : tensor<32x128x7x7xf32>
    %v645 = stablehlo.reduce(%v644 init: %v636) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x7x7xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v646 = stablehlo.broadcast_in_dim %v645, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x7x7xf32>
    %v647 = stablehlo.divide %v646, %v638 : tensor<32x128x7x7xf32>
    %v648 = stablehlo.add %v647, %v639 : tensor<32x128x7x7xf32>
    %v649 = stablehlo.rsqrt %v648 : tensor<32x128x7x7xf32>
    %v650 = stablehlo.multiply %v643, %v649 : tensor<32x128x7x7xf32>
    %v651 = stablehlo.reshape %v591 : (tensor<32x6272xf32>) -> tensor<32x128x7x7xf32>
    %v652 = stablehlo.multiply %v651, %v650 : tensor<32x128x7x7xf32>
    %v653 = stablehlo.reduce(%v652 init: %v636) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x7x7xf32>, tensor<f32>) -> tensor<128xf32>
    %v654 = stablehlo.constant dense<0.3> : tensor<128xf32>
    %v655 = stablehlo.multiply %v653, %v654 : tensor<128xf32>
    %v656 = stablehlo.subtract %gh, %v655 : tensor<128xf32>
    %v657 = stablehlo.constant dense<0.0> : tensor<f32>
    %v658 = stablehlo.reshape %v591 : (tensor<32x6272xf32>) -> tensor<32x128x7x7xf32>
    %v659 = stablehlo.reduce(%v658 init: %v657) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x7x7xf32>, tensor<f32>) -> tensor<128xf32>
    %v660 = stablehlo.constant dense<0.3> : tensor<128xf32>
    %v661 = stablehlo.multiply %v659, %v660 : tensor<128xf32>
    %v662 = stablehlo.subtract %bth, %v661 : tensor<128xf32>
    %v663 = stablehlo.reshape %v626 : (tensor<32x3136xf32>) -> tensor<32x64x7x7xf32>
    %v664 = stablehlo.reshape %v508 : (tensor<32x3136xf32>) -> tensor<32x64x7x7xf32>
    %v665 = stablehlo.constant dense<0.0> : tensor<f32>
    %v666 = stablehlo.constant dense<49.0> : tensor<32x64x7x7xf32>
    %v667 = stablehlo.constant dense<1.0e-5> : tensor<32x64x7x7xf32>
    %v668 = stablehlo.reduce(%v664 init: %v665) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x7x7xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v669 = stablehlo.broadcast_in_dim %v668, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x7x7xf32>
    %v670 = stablehlo.divide %v669, %v666 : tensor<32x64x7x7xf32>
    %v671 = stablehlo.subtract %v664, %v670 : tensor<32x64x7x7xf32>
    %v672 = stablehlo.multiply %v671, %v671 : tensor<32x64x7x7xf32>
    %v673 = stablehlo.reduce(%v672 init: %v665) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x7x7xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v674 = stablehlo.broadcast_in_dim %v673, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x7x7xf32>
    %v675 = stablehlo.divide %v674, %v666 : tensor<32x64x7x7xf32>
    %v676 = stablehlo.add %v675, %v667 : tensor<32x64x7x7xf32>
    %v677 = stablehlo.rsqrt %v676 : tensor<32x64x7x7xf32>
    %v678 = stablehlo.multiply %v671, %v677 : tensor<32x64x7x7xf32>
    %v679 = stablehlo.broadcast_in_dim %gp6, dims = [1] : (tensor<64xf32>) -> tensor<32x64x7x7xf32>
    %v680 = stablehlo.multiply %v679, %v663 : tensor<32x64x7x7xf32>
    %v681 = stablehlo.reduce(%v680 init: %v665) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x7x7xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v682 = stablehlo.broadcast_in_dim %v681, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x7x7xf32>
    %v683 = stablehlo.multiply %v678, %v680 : tensor<32x64x7x7xf32>
    %v684 = stablehlo.reduce(%v683 init: %v665) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x7x7xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v685 = stablehlo.broadcast_in_dim %v684, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x7x7xf32>
    %v686 = stablehlo.multiply %v680, %v666 : tensor<32x64x7x7xf32>
    %v687 = stablehlo.subtract %v686, %v682 : tensor<32x64x7x7xf32>
    %v688 = stablehlo.multiply %v678, %v685 : tensor<32x64x7x7xf32>
    %v689 = stablehlo.subtract %v687, %v688 : tensor<32x64x7x7xf32>
    %v690 = stablehlo.divide %v677, %v666 : tensor<32x64x7x7xf32>
    %v691 = stablehlo.multiply %v690, %v689 : tensor<32x64x7x7xf32>
    %v692 = stablehlo.reshape %v691 : (tensor<32x64x7x7xf32>) -> tensor<32x3136xf32>
    %v693 = stablehlo.reshape %v692 : (tensor<32x3136xf32>) -> tensor<32x64x7x7xf32>
    %v694 = stablehlo.transpose %Wp6, dims = [1, 0, 2, 3] : (tensor<64x256x1x1xf32>) -> tensor<256x64x1x1xf32>
    %v695 = stablehlo.reverse %v694, dims = [2, 3] : tensor<256x64x1x1xf32>
    %v696 = stablehlo.convolution(%v693, %v695)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x7x7xf32>, tensor<256x64x1x1xf32>) -> tensor<32x256x7x7xf32>
    %v697 = stablehlo.reshape %v696 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v698 = stablehlo.constant dense<0.0> : tensor<32x12544xf32>
    %v699 = stablehlo.constant dense<6.0> : tensor<32x12544xf32>
    %v700 = stablehlo.compare GT, %v499, %v698 : (tensor<32x12544xf32>, tensor<32x12544xf32>) -> tensor<32x12544xi1>
    %v701 = stablehlo.compare LT, %v499, %v699 : (tensor<32x12544xf32>, tensor<32x12544xf32>) -> tensor<32x12544xi1>
    %v702 = stablehlo.and %v700, %v701 : tensor<32x12544xi1>
    %v703 = stablehlo.select %v702, %v697, %v698 : tensor<32x12544xi1>, tensor<32x12544xf32>
    %v704 = stablehlo.reshape %v703 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v705 = stablehlo.reshape %v479 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v706 = stablehlo.constant dense<0.0> : tensor<f32>
    %v707 = stablehlo.constant dense<49.0> : tensor<32x256x7x7xf32>
    %v708 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v709 = stablehlo.reduce(%v705 init: %v706) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v710 = stablehlo.broadcast_in_dim %v709, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x7x7xf32>
    %v711 = stablehlo.divide %v710, %v707 : tensor<32x256x7x7xf32>
    %v712 = stablehlo.subtract %v705, %v711 : tensor<32x256x7x7xf32>
    %v713 = stablehlo.multiply %v712, %v712 : tensor<32x256x7x7xf32>
    %v714 = stablehlo.reduce(%v713 init: %v706) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v715 = stablehlo.broadcast_in_dim %v714, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x7x7xf32>
    %v716 = stablehlo.divide %v715, %v707 : tensor<32x256x7x7xf32>
    %v717 = stablehlo.add %v716, %v708 : tensor<32x256x7x7xf32>
    %v718 = stablehlo.rsqrt %v717 : tensor<32x256x7x7xf32>
    %v719 = stablehlo.multiply %v712, %v718 : tensor<32x256x7x7xf32>
    %v720 = stablehlo.broadcast_in_dim %gd6, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v721 = stablehlo.multiply %v720, %v704 : tensor<32x256x7x7xf32>
    %v722 = stablehlo.reduce(%v721 init: %v706) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v723 = stablehlo.broadcast_in_dim %v722, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x7x7xf32>
    %v724 = stablehlo.multiply %v719, %v721 : tensor<32x256x7x7xf32>
    %v725 = stablehlo.reduce(%v724 init: %v706) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v726 = stablehlo.broadcast_in_dim %v725, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x7x7xf32>
    %v727 = stablehlo.multiply %v721, %v707 : tensor<32x256x7x7xf32>
    %v728 = stablehlo.subtract %v727, %v723 : tensor<32x256x7x7xf32>
    %v729 = stablehlo.multiply %v719, %v726 : tensor<32x256x7x7xf32>
    %v730 = stablehlo.subtract %v728, %v729 : tensor<32x256x7x7xf32>
    %v731 = stablehlo.divide %v718, %v707 : tensor<32x256x7x7xf32>
    %v732 = stablehlo.multiply %v731, %v730 : tensor<32x256x7x7xf32>
    %v733 = stablehlo.reshape %v732 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v734 = stablehlo.reshape %v733 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v735 = stablehlo.constant dense<0.0> : tensor<f32>
    %v736 = stablehlo.pad %v734, %v735, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<32x256x14x14xf32>
    %v737 = stablehlo.reverse %Wd6, dims = [2, 3] : tensor<256x1x3x3xf32>
    %v738 = stablehlo.convolution(%v736, %v737)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 256 : i64} : (tensor<32x256x14x14xf32>, tensor<256x1x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v739 = stablehlo.reshape %v738 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v740 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v741 = stablehlo.constant dense<6.0> : tensor<32x50176xf32>
    %v742 = stablehlo.compare GT, %v470, %v740 : (tensor<32x50176xf32>, tensor<32x50176xf32>) -> tensor<32x50176xi1>
    %v743 = stablehlo.compare LT, %v470, %v741 : (tensor<32x50176xf32>, tensor<32x50176xf32>) -> tensor<32x50176xi1>
    %v744 = stablehlo.and %v742, %v743 : tensor<32x50176xi1>
    %v745 = stablehlo.select %v744, %v739, %v740 : tensor<32x50176xi1>, tensor<32x50176xf32>
    %v746 = stablehlo.reshape %v745 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v747 = stablehlo.reshape %v450 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v748 = stablehlo.constant dense<0.0> : tensor<f32>
    %v749 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v750 = stablehlo.constant dense<1.0e-5> : tensor<32x256x14x14xf32>
    %v751 = stablehlo.reduce(%v747 init: %v748) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v752 = stablehlo.broadcast_in_dim %v751, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v753 = stablehlo.divide %v752, %v749 : tensor<32x256x14x14xf32>
    %v754 = stablehlo.subtract %v747, %v753 : tensor<32x256x14x14xf32>
    %v755 = stablehlo.multiply %v754, %v754 : tensor<32x256x14x14xf32>
    %v756 = stablehlo.reduce(%v755 init: %v748) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v757 = stablehlo.broadcast_in_dim %v756, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v758 = stablehlo.divide %v757, %v749 : tensor<32x256x14x14xf32>
    %v759 = stablehlo.add %v758, %v750 : tensor<32x256x14x14xf32>
    %v760 = stablehlo.rsqrt %v759 : tensor<32x256x14x14xf32>
    %v761 = stablehlo.multiply %v754, %v760 : tensor<32x256x14x14xf32>
    %v762 = stablehlo.broadcast_in_dim %ge6, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v763 = stablehlo.multiply %v762, %v746 : tensor<32x256x14x14xf32>
    %v764 = stablehlo.reduce(%v763 init: %v748) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v765 = stablehlo.broadcast_in_dim %v764, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v766 = stablehlo.multiply %v761, %v763 : tensor<32x256x14x14xf32>
    %v767 = stablehlo.reduce(%v766 init: %v748) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v768 = stablehlo.broadcast_in_dim %v767, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v769 = stablehlo.multiply %v763, %v749 : tensor<32x256x14x14xf32>
    %v770 = stablehlo.subtract %v769, %v765 : tensor<32x256x14x14xf32>
    %v771 = stablehlo.multiply %v761, %v768 : tensor<32x256x14x14xf32>
    %v772 = stablehlo.subtract %v770, %v771 : tensor<32x256x14x14xf32>
    %v773 = stablehlo.divide %v760, %v749 : tensor<32x256x14x14xf32>
    %v774 = stablehlo.multiply %v773, %v772 : tensor<32x256x14x14xf32>
    %v775 = stablehlo.reshape %v774 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v776 = stablehlo.reshape %v775 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v777 = stablehlo.transpose %We6, dims = [1, 0, 2, 3] : (tensor<256x64x1x1xf32>) -> tensor<64x256x1x1xf32>
    %v778 = stablehlo.reverse %v777, dims = [2, 3] : tensor<64x256x1x1xf32>
    %v779 = stablehlo.convolution(%v776, %v778)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<64x256x1x1xf32>) -> tensor<32x64x14x14xf32>
    %v780 = stablehlo.reshape %v779 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v781 = stablehlo.reshape %v445 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v782 = stablehlo.reshape %v775 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v783 = stablehlo.transpose %v781, dims = [1, 0, 2, 3] : (tensor<32x64x14x14xf32>) -> tensor<64x32x14x14xf32>
    %v784 = stablehlo.transpose %v782, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v785 = stablehlo.convolution(%v783, %v784)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<64x256x1x1xf32>
    %v786 = stablehlo.transpose %v785, dims = [1, 0, 2, 3] : (tensor<64x256x1x1xf32>) -> tensor<256x64x1x1xf32>
    %v787 = stablehlo.constant dense<0.3> : tensor<256x64x1x1xf32>
    %v788 = stablehlo.multiply %v786, %v787 : tensor<256x64x1x1xf32>
    %v789 = stablehlo.subtract %We6, %v788 : tensor<256x64x1x1xf32>
    %v790 = stablehlo.constant dense<0.0> : tensor<f32>
    %v791 = stablehlo.reshape %v450 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v792 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v793 = stablehlo.constant dense<1.0e-5> : tensor<32x256x14x14xf32>
    %v794 = stablehlo.reduce(%v791 init: %v790) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v795 = stablehlo.broadcast_in_dim %v794, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v796 = stablehlo.divide %v795, %v792 : tensor<32x256x14x14xf32>
    %v797 = stablehlo.subtract %v791, %v796 : tensor<32x256x14x14xf32>
    %v798 = stablehlo.multiply %v797, %v797 : tensor<32x256x14x14xf32>
    %v799 = stablehlo.reduce(%v798 init: %v790) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v800 = stablehlo.broadcast_in_dim %v799, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v801 = stablehlo.divide %v800, %v792 : tensor<32x256x14x14xf32>
    %v802 = stablehlo.add %v801, %v793 : tensor<32x256x14x14xf32>
    %v803 = stablehlo.rsqrt %v802 : tensor<32x256x14x14xf32>
    %v804 = stablehlo.multiply %v797, %v803 : tensor<32x256x14x14xf32>
    %v805 = stablehlo.reshape %v745 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v806 = stablehlo.multiply %v805, %v804 : tensor<32x256x14x14xf32>
    %v807 = stablehlo.reduce(%v806 init: %v790) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v808 = stablehlo.constant dense<0.3> : tensor<256xf32>
    %v809 = stablehlo.multiply %v807, %v808 : tensor<256xf32>
    %v810 = stablehlo.subtract %ge6, %v809 : tensor<256xf32>
    %v811 = stablehlo.constant dense<0.0> : tensor<f32>
    %v812 = stablehlo.reshape %v745 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v813 = stablehlo.reduce(%v812 init: %v811) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v814 = stablehlo.constant dense<0.3> : tensor<256xf32>
    %v815 = stablehlo.multiply %v813, %v814 : tensor<256xf32>
    %v816 = stablehlo.subtract %bte6, %v815 : tensor<256xf32>
    %v817 = stablehlo.reshape %v474 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v818 = stablehlo.reshape %v733 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v819 = stablehlo.constant dense<0.0> : tensor<f32>
    %v820 = stablehlo.pad %v818, %v819, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<32x256x14x14xf32>
    %v821 = stablehlo.transpose %v817, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v822 = stablehlo.transpose %v820, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v823 = stablehlo.convolution(%v821, %v822)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 256 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<1x256x3x3xf32>
    %v824 = stablehlo.reshape %v823 : (tensor<1x256x3x3xf32>) -> tensor<256x1x3x3xf32>
    %v825 = stablehlo.constant dense<0.3> : tensor<256x1x3x3xf32>
    %v826 = stablehlo.multiply %v824, %v825 : tensor<256x1x3x3xf32>
    %v827 = stablehlo.subtract %Wd6, %v826 : tensor<256x1x3x3xf32>
    %v828 = stablehlo.constant dense<0.0> : tensor<f32>
    %v829 = stablehlo.reshape %v479 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v830 = stablehlo.constant dense<49.0> : tensor<32x256x7x7xf32>
    %v831 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v832 = stablehlo.reduce(%v829 init: %v828) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v833 = stablehlo.broadcast_in_dim %v832, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x7x7xf32>
    %v834 = stablehlo.divide %v833, %v830 : tensor<32x256x7x7xf32>
    %v835 = stablehlo.subtract %v829, %v834 : tensor<32x256x7x7xf32>
    %v836 = stablehlo.multiply %v835, %v835 : tensor<32x256x7x7xf32>
    %v837 = stablehlo.reduce(%v836 init: %v828) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v838 = stablehlo.broadcast_in_dim %v837, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x7x7xf32>
    %v839 = stablehlo.divide %v838, %v830 : tensor<32x256x7x7xf32>
    %v840 = stablehlo.add %v839, %v831 : tensor<32x256x7x7xf32>
    %v841 = stablehlo.rsqrt %v840 : tensor<32x256x7x7xf32>
    %v842 = stablehlo.multiply %v835, %v841 : tensor<32x256x7x7xf32>
    %v843 = stablehlo.reshape %v703 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v844 = stablehlo.multiply %v843, %v842 : tensor<32x256x7x7xf32>
    %v845 = stablehlo.reduce(%v844 init: %v828) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v846 = stablehlo.constant dense<0.3> : tensor<256xf32>
    %v847 = stablehlo.multiply %v845, %v846 : tensor<256xf32>
    %v848 = stablehlo.subtract %gd6, %v847 : tensor<256xf32>
    %v849 = stablehlo.constant dense<0.0> : tensor<f32>
    %v850 = stablehlo.reshape %v703 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v851 = stablehlo.reduce(%v850 init: %v849) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v852 = stablehlo.constant dense<0.3> : tensor<256xf32>
    %v853 = stablehlo.multiply %v851, %v852 : tensor<256xf32>
    %v854 = stablehlo.subtract %btd6, %v853 : tensor<256xf32>
    %v855 = stablehlo.reshape %v503 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v856 = stablehlo.reshape %v692 : (tensor<32x3136xf32>) -> tensor<32x64x7x7xf32>
    %v857 = stablehlo.transpose %v855, dims = [1, 0, 2, 3] : (tensor<32x256x7x7xf32>) -> tensor<256x32x7x7xf32>
    %v858 = stablehlo.transpose %v856, dims = [1, 0, 2, 3] : (tensor<32x64x7x7xf32>) -> tensor<64x32x7x7xf32>
    %v859 = stablehlo.convolution(%v857, %v858)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x7x7xf32>, tensor<64x32x7x7xf32>) -> tensor<256x64x1x1xf32>
    %v860 = stablehlo.transpose %v859, dims = [1, 0, 2, 3] : (tensor<256x64x1x1xf32>) -> tensor<64x256x1x1xf32>
    %v861 = stablehlo.constant dense<0.3> : tensor<64x256x1x1xf32>
    %v862 = stablehlo.multiply %v860, %v861 : tensor<64x256x1x1xf32>
    %v863 = stablehlo.subtract %Wp6, %v862 : tensor<64x256x1x1xf32>
    %v864 = stablehlo.constant dense<0.0> : tensor<f32>
    %v865 = stablehlo.reshape %v508 : (tensor<32x3136xf32>) -> tensor<32x64x7x7xf32>
    %v866 = stablehlo.constant dense<49.0> : tensor<32x64x7x7xf32>
    %v867 = stablehlo.constant dense<1.0e-5> : tensor<32x64x7x7xf32>
    %v868 = stablehlo.reduce(%v865 init: %v864) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x7x7xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v869 = stablehlo.broadcast_in_dim %v868, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x7x7xf32>
    %v870 = stablehlo.divide %v869, %v866 : tensor<32x64x7x7xf32>
    %v871 = stablehlo.subtract %v865, %v870 : tensor<32x64x7x7xf32>
    %v872 = stablehlo.multiply %v871, %v871 : tensor<32x64x7x7xf32>
    %v873 = stablehlo.reduce(%v872 init: %v864) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x7x7xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v874 = stablehlo.broadcast_in_dim %v873, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x7x7xf32>
    %v875 = stablehlo.divide %v874, %v866 : tensor<32x64x7x7xf32>
    %v876 = stablehlo.add %v875, %v867 : tensor<32x64x7x7xf32>
    %v877 = stablehlo.rsqrt %v876 : tensor<32x64x7x7xf32>
    %v878 = stablehlo.multiply %v871, %v877 : tensor<32x64x7x7xf32>
    %v879 = stablehlo.reshape %v626 : (tensor<32x3136xf32>) -> tensor<32x64x7x7xf32>
    %v880 = stablehlo.multiply %v879, %v878 : tensor<32x64x7x7xf32>
    %v881 = stablehlo.reduce(%v880 init: %v864) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x7x7xf32>, tensor<f32>) -> tensor<64xf32>
    %v882 = stablehlo.constant dense<0.3> : tensor<64xf32>
    %v883 = stablehlo.multiply %v881, %v882 : tensor<64xf32>
    %v884 = stablehlo.subtract %gp6, %v883 : tensor<64xf32>
    %v885 = stablehlo.constant dense<0.0> : tensor<f32>
    %v886 = stablehlo.reshape %v626 : (tensor<32x3136xf32>) -> tensor<32x64x7x7xf32>
    %v887 = stablehlo.reduce(%v886 init: %v885) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x7x7xf32>, tensor<f32>) -> tensor<64xf32>
    %v888 = stablehlo.constant dense<0.3> : tensor<64xf32>
    %v889 = stablehlo.multiply %v887, %v888 : tensor<64xf32>
    %v890 = stablehlo.subtract %btp6, %v889 : tensor<64xf32>
    %v891 = stablehlo.reshape %v780 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v892 = stablehlo.reshape %v425 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v893 = stablehlo.constant dense<0.0> : tensor<f32>
    %v894 = stablehlo.constant dense<196.0> : tensor<32x64x14x14xf32>
    %v895 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v896 = stablehlo.reduce(%v892 init: %v893) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v897 = stablehlo.broadcast_in_dim %v896, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v898 = stablehlo.divide %v897, %v894 : tensor<32x64x14x14xf32>
    %v899 = stablehlo.subtract %v892, %v898 : tensor<32x64x14x14xf32>
    %v900 = stablehlo.multiply %v899, %v899 : tensor<32x64x14x14xf32>
    %v901 = stablehlo.reduce(%v900 init: %v893) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v902 = stablehlo.broadcast_in_dim %v901, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v903 = stablehlo.divide %v902, %v894 : tensor<32x64x14x14xf32>
    %v904 = stablehlo.add %v903, %v895 : tensor<32x64x14x14xf32>
    %v905 = stablehlo.rsqrt %v904 : tensor<32x64x14x14xf32>
    %v906 = stablehlo.multiply %v899, %v905 : tensor<32x64x14x14xf32>
    %v907 = stablehlo.broadcast_in_dim %gp5, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v908 = stablehlo.multiply %v907, %v891 : tensor<32x64x14x14xf32>
    %v909 = stablehlo.reduce(%v908 init: %v893) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v910 = stablehlo.broadcast_in_dim %v909, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v911 = stablehlo.multiply %v906, %v908 : tensor<32x64x14x14xf32>
    %v912 = stablehlo.reduce(%v911 init: %v893) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v913 = stablehlo.broadcast_in_dim %v912, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v914 = stablehlo.multiply %v908, %v894 : tensor<32x64x14x14xf32>
    %v915 = stablehlo.subtract %v914, %v910 : tensor<32x64x14x14xf32>
    %v916 = stablehlo.multiply %v906, %v913 : tensor<32x64x14x14xf32>
    %v917 = stablehlo.subtract %v915, %v916 : tensor<32x64x14x14xf32>
    %v918 = stablehlo.divide %v905, %v894 : tensor<32x64x14x14xf32>
    %v919 = stablehlo.multiply %v918, %v917 : tensor<32x64x14x14xf32>
    %v920 = stablehlo.reshape %v919 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v921 = stablehlo.reshape %v920 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v922 = stablehlo.transpose %Wp5, dims = [1, 0, 2, 3] : (tensor<64x128x1x1xf32>) -> tensor<128x64x1x1xf32>
    %v923 = stablehlo.reverse %v922, dims = [2, 3] : tensor<128x64x1x1xf32>
    %v924 = stablehlo.convolution(%v921, %v923)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x14x14xf32>, tensor<128x64x1x1xf32>) -> tensor<32x128x14x14xf32>
    %v925 = stablehlo.reshape %v924 : (tensor<32x128x14x14xf32>) -> tensor<32x25088xf32>
    %v926 = stablehlo.constant dense<0.0> : tensor<32x25088xf32>
    %v927 = stablehlo.constant dense<6.0> : tensor<32x25088xf32>
    %v928 = stablehlo.compare GT, %v416, %v926 : (tensor<32x25088xf32>, tensor<32x25088xf32>) -> tensor<32x25088xi1>
    %v929 = stablehlo.compare LT, %v416, %v927 : (tensor<32x25088xf32>, tensor<32x25088xf32>) -> tensor<32x25088xi1>
    %v930 = stablehlo.and %v928, %v929 : tensor<32x25088xi1>
    %v931 = stablehlo.select %v930, %v925, %v926 : tensor<32x25088xi1>, tensor<32x25088xf32>
    %v932 = stablehlo.reshape %v931 : (tensor<32x25088xf32>) -> tensor<32x128x14x14xf32>
    %v933 = stablehlo.reshape %v396 : (tensor<32x25088xf32>) -> tensor<32x128x14x14xf32>
    %v934 = stablehlo.constant dense<0.0> : tensor<f32>
    %v935 = stablehlo.constant dense<196.0> : tensor<32x128x14x14xf32>
    %v936 = stablehlo.constant dense<1.0e-5> : tensor<32x128x14x14xf32>
    %v937 = stablehlo.reduce(%v933 init: %v934) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x14x14xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v938 = stablehlo.broadcast_in_dim %v937, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x14x14xf32>
    %v939 = stablehlo.divide %v938, %v935 : tensor<32x128x14x14xf32>
    %v940 = stablehlo.subtract %v933, %v939 : tensor<32x128x14x14xf32>
    %v941 = stablehlo.multiply %v940, %v940 : tensor<32x128x14x14xf32>
    %v942 = stablehlo.reduce(%v941 init: %v934) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x14x14xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v943 = stablehlo.broadcast_in_dim %v942, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x14x14xf32>
    %v944 = stablehlo.divide %v943, %v935 : tensor<32x128x14x14xf32>
    %v945 = stablehlo.add %v944, %v936 : tensor<32x128x14x14xf32>
    %v946 = stablehlo.rsqrt %v945 : tensor<32x128x14x14xf32>
    %v947 = stablehlo.multiply %v940, %v946 : tensor<32x128x14x14xf32>
    %v948 = stablehlo.broadcast_in_dim %gd5, dims = [1] : (tensor<128xf32>) -> tensor<32x128x14x14xf32>
    %v949 = stablehlo.multiply %v948, %v932 : tensor<32x128x14x14xf32>
    %v950 = stablehlo.reduce(%v949 init: %v934) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x14x14xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v951 = stablehlo.broadcast_in_dim %v950, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x14x14xf32>
    %v952 = stablehlo.multiply %v947, %v949 : tensor<32x128x14x14xf32>
    %v953 = stablehlo.reduce(%v952 init: %v934) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x14x14xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v954 = stablehlo.broadcast_in_dim %v953, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x14x14xf32>
    %v955 = stablehlo.multiply %v949, %v935 : tensor<32x128x14x14xf32>
    %v956 = stablehlo.subtract %v955, %v951 : tensor<32x128x14x14xf32>
    %v957 = stablehlo.multiply %v947, %v954 : tensor<32x128x14x14xf32>
    %v958 = stablehlo.subtract %v956, %v957 : tensor<32x128x14x14xf32>
    %v959 = stablehlo.divide %v946, %v935 : tensor<32x128x14x14xf32>
    %v960 = stablehlo.multiply %v959, %v958 : tensor<32x128x14x14xf32>
    %v961 = stablehlo.reshape %v960 : (tensor<32x128x14x14xf32>) -> tensor<32x25088xf32>
    %v962 = stablehlo.reshape %v961 : (tensor<32x25088xf32>) -> tensor<32x128x14x14xf32>
    %v963 = stablehlo.constant dense<0.0> : tensor<f32>
    %v964 = stablehlo.pad %v962, %v963, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x128x14x14xf32>, tensor<f32>) -> tensor<32x128x28x28xf32>
    %v965 = stablehlo.reverse %Wd5, dims = [2, 3] : tensor<128x1x3x3xf32>
    %v966 = stablehlo.convolution(%v964, %v965)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 128 : i64} : (tensor<32x128x28x28xf32>, tensor<128x1x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v967 = stablehlo.reshape %v966 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v968 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v969 = stablehlo.constant dense<6.0> : tensor<32x100352xf32>
    %v970 = stablehlo.compare GT, %v387, %v968 : (tensor<32x100352xf32>, tensor<32x100352xf32>) -> tensor<32x100352xi1>
    %v971 = stablehlo.compare LT, %v387, %v969 : (tensor<32x100352xf32>, tensor<32x100352xf32>) -> tensor<32x100352xi1>
    %v972 = stablehlo.and %v970, %v971 : tensor<32x100352xi1>
    %v973 = stablehlo.select %v972, %v967, %v968 : tensor<32x100352xi1>, tensor<32x100352xf32>
    %v974 = stablehlo.reshape %v973 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v975 = stablehlo.reshape %v367 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v976 = stablehlo.constant dense<0.0> : tensor<f32>
    %v977 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v978 = stablehlo.constant dense<1.0e-5> : tensor<32x128x28x28xf32>
    %v979 = stablehlo.reduce(%v975 init: %v976) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v980 = stablehlo.broadcast_in_dim %v979, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v981 = stablehlo.divide %v980, %v977 : tensor<32x128x28x28xf32>
    %v982 = stablehlo.subtract %v975, %v981 : tensor<32x128x28x28xf32>
    %v983 = stablehlo.multiply %v982, %v982 : tensor<32x128x28x28xf32>
    %v984 = stablehlo.reduce(%v983 init: %v976) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v985 = stablehlo.broadcast_in_dim %v984, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v986 = stablehlo.divide %v985, %v977 : tensor<32x128x28x28xf32>
    %v987 = stablehlo.add %v986, %v978 : tensor<32x128x28x28xf32>
    %v988 = stablehlo.rsqrt %v987 : tensor<32x128x28x28xf32>
    %v989 = stablehlo.multiply %v982, %v988 : tensor<32x128x28x28xf32>
    %v990 = stablehlo.broadcast_in_dim %ge5, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v991 = stablehlo.multiply %v990, %v974 : tensor<32x128x28x28xf32>
    %v992 = stablehlo.reduce(%v991 init: %v976) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v993 = stablehlo.broadcast_in_dim %v992, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v994 = stablehlo.multiply %v989, %v991 : tensor<32x128x28x28xf32>
    %v995 = stablehlo.reduce(%v994 init: %v976) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v996 = stablehlo.broadcast_in_dim %v995, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v997 = stablehlo.multiply %v991, %v977 : tensor<32x128x28x28xf32>
    %v998 = stablehlo.subtract %v997, %v993 : tensor<32x128x28x28xf32>
    %v999 = stablehlo.multiply %v989, %v996 : tensor<32x128x28x28xf32>
    %v1000 = stablehlo.subtract %v998, %v999 : tensor<32x128x28x28xf32>
    %v1001 = stablehlo.divide %v988, %v977 : tensor<32x128x28x28xf32>
    %v1002 = stablehlo.multiply %v1001, %v1000 : tensor<32x128x28x28xf32>
    %v1003 = stablehlo.reshape %v1002 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v1004 = stablehlo.reshape %v1003 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v1005 = stablehlo.transpose %We5, dims = [1, 0, 2, 3] : (tensor<128x32x1x1xf32>) -> tensor<32x128x1x1xf32>
    %v1006 = stablehlo.reverse %v1005, dims = [2, 3] : tensor<32x128x1x1xf32>
    %v1007 = stablehlo.convolution(%v1004, %v1006)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<32x128x1x1xf32>) -> tensor<32x32x28x28xf32>
    %v1008 = stablehlo.reshape %v1007 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v1009 = stablehlo.reshape %v362 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v1010 = stablehlo.reshape %v1003 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v1011 = stablehlo.transpose %v1009, dims = [1, 0, 2, 3] : (tensor<32x32x28x28xf32>) -> tensor<32x32x28x28xf32>
    %v1012 = stablehlo.transpose %v1010, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v1013 = stablehlo.convolution(%v1011, %v1012)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<32x128x1x1xf32>
    %v1014 = stablehlo.transpose %v1013, dims = [1, 0, 2, 3] : (tensor<32x128x1x1xf32>) -> tensor<128x32x1x1xf32>
    %v1015 = stablehlo.constant dense<0.3> : tensor<128x32x1x1xf32>
    %v1016 = stablehlo.multiply %v1014, %v1015 : tensor<128x32x1x1xf32>
    %v1017 = stablehlo.subtract %We5, %v1016 : tensor<128x32x1x1xf32>
    %v1018 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1019 = stablehlo.reshape %v367 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v1020 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v1021 = stablehlo.constant dense<1.0e-5> : tensor<32x128x28x28xf32>
    %v1022 = stablehlo.reduce(%v1019 init: %v1018) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v1023 = stablehlo.broadcast_in_dim %v1022, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v1024 = stablehlo.divide %v1023, %v1020 : tensor<32x128x28x28xf32>
    %v1025 = stablehlo.subtract %v1019, %v1024 : tensor<32x128x28x28xf32>
    %v1026 = stablehlo.multiply %v1025, %v1025 : tensor<32x128x28x28xf32>
    %v1027 = stablehlo.reduce(%v1026 init: %v1018) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v1028 = stablehlo.broadcast_in_dim %v1027, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v1029 = stablehlo.divide %v1028, %v1020 : tensor<32x128x28x28xf32>
    %v1030 = stablehlo.add %v1029, %v1021 : tensor<32x128x28x28xf32>
    %v1031 = stablehlo.rsqrt %v1030 : tensor<32x128x28x28xf32>
    %v1032 = stablehlo.multiply %v1025, %v1031 : tensor<32x128x28x28xf32>
    %v1033 = stablehlo.reshape %v973 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v1034 = stablehlo.multiply %v1033, %v1032 : tensor<32x128x28x28xf32>
    %v1035 = stablehlo.reduce(%v1034 init: %v1018) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v1036 = stablehlo.constant dense<0.3> : tensor<128xf32>
    %v1037 = stablehlo.multiply %v1035, %v1036 : tensor<128xf32>
    %v1038 = stablehlo.subtract %ge5, %v1037 : tensor<128xf32>
    %v1039 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1040 = stablehlo.reshape %v973 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v1041 = stablehlo.reduce(%v1040 init: %v1039) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v1042 = stablehlo.constant dense<0.3> : tensor<128xf32>
    %v1043 = stablehlo.multiply %v1041, %v1042 : tensor<128xf32>
    %v1044 = stablehlo.subtract %bte5, %v1043 : tensor<128xf32>
    %v1045 = stablehlo.reshape %v391 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v1046 = stablehlo.reshape %v961 : (tensor<32x25088xf32>) -> tensor<32x128x14x14xf32>
    %v1047 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1048 = stablehlo.pad %v1046, %v1047, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x128x14x14xf32>, tensor<f32>) -> tensor<32x128x28x28xf32>
    %v1049 = stablehlo.transpose %v1045, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v1050 = stablehlo.transpose %v1048, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v1051 = stablehlo.convolution(%v1049, %v1050)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 128 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<1x128x3x3xf32>
    %v1052 = stablehlo.reshape %v1051 : (tensor<1x128x3x3xf32>) -> tensor<128x1x3x3xf32>
    %v1053 = stablehlo.constant dense<0.3> : tensor<128x1x3x3xf32>
    %v1054 = stablehlo.multiply %v1052, %v1053 : tensor<128x1x3x3xf32>
    %v1055 = stablehlo.subtract %Wd5, %v1054 : tensor<128x1x3x3xf32>
    %v1056 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1057 = stablehlo.reshape %v396 : (tensor<32x25088xf32>) -> tensor<32x128x14x14xf32>
    %v1058 = stablehlo.constant dense<196.0> : tensor<32x128x14x14xf32>
    %v1059 = stablehlo.constant dense<1.0e-5> : tensor<32x128x14x14xf32>
    %v1060 = stablehlo.reduce(%v1057 init: %v1056) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x14x14xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v1061 = stablehlo.broadcast_in_dim %v1060, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x14x14xf32>
    %v1062 = stablehlo.divide %v1061, %v1058 : tensor<32x128x14x14xf32>
    %v1063 = stablehlo.subtract %v1057, %v1062 : tensor<32x128x14x14xf32>
    %v1064 = stablehlo.multiply %v1063, %v1063 : tensor<32x128x14x14xf32>
    %v1065 = stablehlo.reduce(%v1064 init: %v1056) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x14x14xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v1066 = stablehlo.broadcast_in_dim %v1065, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x14x14xf32>
    %v1067 = stablehlo.divide %v1066, %v1058 : tensor<32x128x14x14xf32>
    %v1068 = stablehlo.add %v1067, %v1059 : tensor<32x128x14x14xf32>
    %v1069 = stablehlo.rsqrt %v1068 : tensor<32x128x14x14xf32>
    %v1070 = stablehlo.multiply %v1063, %v1069 : tensor<32x128x14x14xf32>
    %v1071 = stablehlo.reshape %v931 : (tensor<32x25088xf32>) -> tensor<32x128x14x14xf32>
    %v1072 = stablehlo.multiply %v1071, %v1070 : tensor<32x128x14x14xf32>
    %v1073 = stablehlo.reduce(%v1072 init: %v1056) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x14x14xf32>, tensor<f32>) -> tensor<128xf32>
    %v1074 = stablehlo.constant dense<0.3> : tensor<128xf32>
    %v1075 = stablehlo.multiply %v1073, %v1074 : tensor<128xf32>
    %v1076 = stablehlo.subtract %gd5, %v1075 : tensor<128xf32>
    %v1077 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1078 = stablehlo.reshape %v931 : (tensor<32x25088xf32>) -> tensor<32x128x14x14xf32>
    %v1079 = stablehlo.reduce(%v1078 init: %v1077) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x14x14xf32>, tensor<f32>) -> tensor<128xf32>
    %v1080 = stablehlo.constant dense<0.3> : tensor<128xf32>
    %v1081 = stablehlo.multiply %v1079, %v1080 : tensor<128xf32>
    %v1082 = stablehlo.subtract %btd5, %v1081 : tensor<128xf32>
    %v1083 = stablehlo.reshape %v420 : (tensor<32x25088xf32>) -> tensor<32x128x14x14xf32>
    %v1084 = stablehlo.reshape %v920 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v1085 = stablehlo.transpose %v1083, dims = [1, 0, 2, 3] : (tensor<32x128x14x14xf32>) -> tensor<128x32x14x14xf32>
    %v1086 = stablehlo.transpose %v1084, dims = [1, 0, 2, 3] : (tensor<32x64x14x14xf32>) -> tensor<64x32x14x14xf32>
    %v1087 = stablehlo.convolution(%v1085, %v1086)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x14x14xf32>, tensor<64x32x14x14xf32>) -> tensor<128x64x1x1xf32>
    %v1088 = stablehlo.transpose %v1087, dims = [1, 0, 2, 3] : (tensor<128x64x1x1xf32>) -> tensor<64x128x1x1xf32>
    %v1089 = stablehlo.constant dense<0.3> : tensor<64x128x1x1xf32>
    %v1090 = stablehlo.multiply %v1088, %v1089 : tensor<64x128x1x1xf32>
    %v1091 = stablehlo.subtract %Wp5, %v1090 : tensor<64x128x1x1xf32>
    %v1092 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1093 = stablehlo.reshape %v425 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v1094 = stablehlo.constant dense<196.0> : tensor<32x64x14x14xf32>
    %v1095 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v1096 = stablehlo.reduce(%v1093 init: %v1092) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v1097 = stablehlo.broadcast_in_dim %v1096, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v1098 = stablehlo.divide %v1097, %v1094 : tensor<32x64x14x14xf32>
    %v1099 = stablehlo.subtract %v1093, %v1098 : tensor<32x64x14x14xf32>
    %v1100 = stablehlo.multiply %v1099, %v1099 : tensor<32x64x14x14xf32>
    %v1101 = stablehlo.reduce(%v1100 init: %v1092) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v1102 = stablehlo.broadcast_in_dim %v1101, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v1103 = stablehlo.divide %v1102, %v1094 : tensor<32x64x14x14xf32>
    %v1104 = stablehlo.add %v1103, %v1095 : tensor<32x64x14x14xf32>
    %v1105 = stablehlo.rsqrt %v1104 : tensor<32x64x14x14xf32>
    %v1106 = stablehlo.multiply %v1099, %v1105 : tensor<32x64x14x14xf32>
    %v1107 = stablehlo.reshape %v780 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v1108 = stablehlo.multiply %v1107, %v1106 : tensor<32x64x14x14xf32>
    %v1109 = stablehlo.reduce(%v1108 init: %v1092) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v1110 = stablehlo.constant dense<0.3> : tensor<64xf32>
    %v1111 = stablehlo.multiply %v1109, %v1110 : tensor<64xf32>
    %v1112 = stablehlo.subtract %gp5, %v1111 : tensor<64xf32>
    %v1113 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1114 = stablehlo.reshape %v780 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v1115 = stablehlo.reduce(%v1114 init: %v1113) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v1116 = stablehlo.constant dense<0.3> : tensor<64xf32>
    %v1117 = stablehlo.multiply %v1115, %v1116 : tensor<64xf32>
    %v1118 = stablehlo.subtract %btp5, %v1117 : tensor<64xf32>
    %v1119 = stablehlo.reshape %v1008 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v1120 = stablehlo.reshape %v341 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v1121 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1122 = stablehlo.constant dense<784.0> : tensor<32x32x28x28xf32>
    %v1123 = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %v1124 = stablehlo.reduce(%v1120 init: %v1121) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v1125 = stablehlo.broadcast_in_dim %v1124, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v1126 = stablehlo.divide %v1125, %v1122 : tensor<32x32x28x28xf32>
    %v1127 = stablehlo.subtract %v1120, %v1126 : tensor<32x32x28x28xf32>
    %v1128 = stablehlo.multiply %v1127, %v1127 : tensor<32x32x28x28xf32>
    %v1129 = stablehlo.reduce(%v1128 init: %v1121) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v1130 = stablehlo.broadcast_in_dim %v1129, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v1131 = stablehlo.divide %v1130, %v1122 : tensor<32x32x28x28xf32>
    %v1132 = stablehlo.add %v1131, %v1123 : tensor<32x32x28x28xf32>
    %v1133 = stablehlo.rsqrt %v1132 : tensor<32x32x28x28xf32>
    %v1134 = stablehlo.multiply %v1127, %v1133 : tensor<32x32x28x28xf32>
    %v1135 = stablehlo.broadcast_in_dim %gp4, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v1136 = stablehlo.multiply %v1135, %v1119 : tensor<32x32x28x28xf32>
    %v1137 = stablehlo.reduce(%v1136 init: %v1121) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v1138 = stablehlo.broadcast_in_dim %v1137, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v1139 = stablehlo.multiply %v1134, %v1136 : tensor<32x32x28x28xf32>
    %v1140 = stablehlo.reduce(%v1139 init: %v1121) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v1141 = stablehlo.broadcast_in_dim %v1140, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v1142 = stablehlo.multiply %v1136, %v1122 : tensor<32x32x28x28xf32>
    %v1143 = stablehlo.subtract %v1142, %v1138 : tensor<32x32x28x28xf32>
    %v1144 = stablehlo.multiply %v1134, %v1141 : tensor<32x32x28x28xf32>
    %v1145 = stablehlo.subtract %v1143, %v1144 : tensor<32x32x28x28xf32>
    %v1146 = stablehlo.divide %v1133, %v1122 : tensor<32x32x28x28xf32>
    %v1147 = stablehlo.multiply %v1146, %v1145 : tensor<32x32x28x28xf32>
    %v1148 = stablehlo.reshape %v1147 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v1149 = stablehlo.reshape %v1148 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v1150 = stablehlo.transpose %Wp4, dims = [1, 0, 2, 3] : (tensor<32x128x1x1xf32>) -> tensor<128x32x1x1xf32>
    %v1151 = stablehlo.reverse %v1150, dims = [2, 3] : tensor<128x32x1x1xf32>
    %v1152 = stablehlo.convolution(%v1149, %v1151)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<128x32x1x1xf32>) -> tensor<32x128x28x28xf32>
    %v1153 = stablehlo.reshape %v1152 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v1154 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v1155 = stablehlo.constant dense<6.0> : tensor<32x100352xf32>
    %v1156 = stablehlo.compare GT, %v332, %v1154 : (tensor<32x100352xf32>, tensor<32x100352xf32>) -> tensor<32x100352xi1>
    %v1157 = stablehlo.compare LT, %v332, %v1155 : (tensor<32x100352xf32>, tensor<32x100352xf32>) -> tensor<32x100352xi1>
    %v1158 = stablehlo.and %v1156, %v1157 : tensor<32x100352xi1>
    %v1159 = stablehlo.select %v1158, %v1153, %v1154 : tensor<32x100352xi1>, tensor<32x100352xf32>
    %v1160 = stablehlo.reshape %v1159 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v1161 = stablehlo.reshape %v312 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v1162 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1163 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v1164 = stablehlo.constant dense<1.0e-5> : tensor<32x128x28x28xf32>
    %v1165 = stablehlo.reduce(%v1161 init: %v1162) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v1166 = stablehlo.broadcast_in_dim %v1165, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v1167 = stablehlo.divide %v1166, %v1163 : tensor<32x128x28x28xf32>
    %v1168 = stablehlo.subtract %v1161, %v1167 : tensor<32x128x28x28xf32>
    %v1169 = stablehlo.multiply %v1168, %v1168 : tensor<32x128x28x28xf32>
    %v1170 = stablehlo.reduce(%v1169 init: %v1162) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v1171 = stablehlo.broadcast_in_dim %v1170, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v1172 = stablehlo.divide %v1171, %v1163 : tensor<32x128x28x28xf32>
    %v1173 = stablehlo.add %v1172, %v1164 : tensor<32x128x28x28xf32>
    %v1174 = stablehlo.rsqrt %v1173 : tensor<32x128x28x28xf32>
    %v1175 = stablehlo.multiply %v1168, %v1174 : tensor<32x128x28x28xf32>
    %v1176 = stablehlo.broadcast_in_dim %gd4, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v1177 = stablehlo.multiply %v1176, %v1160 : tensor<32x128x28x28xf32>
    %v1178 = stablehlo.reduce(%v1177 init: %v1162) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v1179 = stablehlo.broadcast_in_dim %v1178, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v1180 = stablehlo.multiply %v1175, %v1177 : tensor<32x128x28x28xf32>
    %v1181 = stablehlo.reduce(%v1180 init: %v1162) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v1182 = stablehlo.broadcast_in_dim %v1181, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v1183 = stablehlo.multiply %v1177, %v1163 : tensor<32x128x28x28xf32>
    %v1184 = stablehlo.subtract %v1183, %v1179 : tensor<32x128x28x28xf32>
    %v1185 = stablehlo.multiply %v1175, %v1182 : tensor<32x128x28x28xf32>
    %v1186 = stablehlo.subtract %v1184, %v1185 : tensor<32x128x28x28xf32>
    %v1187 = stablehlo.divide %v1174, %v1163 : tensor<32x128x28x28xf32>
    %v1188 = stablehlo.multiply %v1187, %v1186 : tensor<32x128x28x28xf32>
    %v1189 = stablehlo.reshape %v1188 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v1190 = stablehlo.reshape %v1189 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v1191 = stablehlo.reverse %Wd4, dims = [2, 3] : tensor<128x1x3x3xf32>
    %v1192 = stablehlo.convolution(%v1190, %v1191)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 128 : i64} : (tensor<32x128x28x28xf32>, tensor<128x1x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v1193 = stablehlo.reshape %v1192 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v1194 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v1195 = stablehlo.constant dense<6.0> : tensor<32x100352xf32>
    %v1196 = stablehlo.compare GT, %v303, %v1194 : (tensor<32x100352xf32>, tensor<32x100352xf32>) -> tensor<32x100352xi1>
    %v1197 = stablehlo.compare LT, %v303, %v1195 : (tensor<32x100352xf32>, tensor<32x100352xf32>) -> tensor<32x100352xi1>
    %v1198 = stablehlo.and %v1196, %v1197 : tensor<32x100352xi1>
    %v1199 = stablehlo.select %v1198, %v1193, %v1194 : tensor<32x100352xi1>, tensor<32x100352xf32>
    %v1200 = stablehlo.reshape %v1199 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v1201 = stablehlo.reshape %v283 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v1202 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1203 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v1204 = stablehlo.constant dense<1.0e-5> : tensor<32x128x28x28xf32>
    %v1205 = stablehlo.reduce(%v1201 init: %v1202) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v1206 = stablehlo.broadcast_in_dim %v1205, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v1207 = stablehlo.divide %v1206, %v1203 : tensor<32x128x28x28xf32>
    %v1208 = stablehlo.subtract %v1201, %v1207 : tensor<32x128x28x28xf32>
    %v1209 = stablehlo.multiply %v1208, %v1208 : tensor<32x128x28x28xf32>
    %v1210 = stablehlo.reduce(%v1209 init: %v1202) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v1211 = stablehlo.broadcast_in_dim %v1210, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v1212 = stablehlo.divide %v1211, %v1203 : tensor<32x128x28x28xf32>
    %v1213 = stablehlo.add %v1212, %v1204 : tensor<32x128x28x28xf32>
    %v1214 = stablehlo.rsqrt %v1213 : tensor<32x128x28x28xf32>
    %v1215 = stablehlo.multiply %v1208, %v1214 : tensor<32x128x28x28xf32>
    %v1216 = stablehlo.broadcast_in_dim %ge4, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v1217 = stablehlo.multiply %v1216, %v1200 : tensor<32x128x28x28xf32>
    %v1218 = stablehlo.reduce(%v1217 init: %v1202) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v1219 = stablehlo.broadcast_in_dim %v1218, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v1220 = stablehlo.multiply %v1215, %v1217 : tensor<32x128x28x28xf32>
    %v1221 = stablehlo.reduce(%v1220 init: %v1202) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v1222 = stablehlo.broadcast_in_dim %v1221, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v1223 = stablehlo.multiply %v1217, %v1203 : tensor<32x128x28x28xf32>
    %v1224 = stablehlo.subtract %v1223, %v1219 : tensor<32x128x28x28xf32>
    %v1225 = stablehlo.multiply %v1215, %v1222 : tensor<32x128x28x28xf32>
    %v1226 = stablehlo.subtract %v1224, %v1225 : tensor<32x128x28x28xf32>
    %v1227 = stablehlo.divide %v1214, %v1203 : tensor<32x128x28x28xf32>
    %v1228 = stablehlo.multiply %v1227, %v1226 : tensor<32x128x28x28xf32>
    %v1229 = stablehlo.reshape %v1228 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v1230 = stablehlo.reshape %v1229 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v1231 = stablehlo.transpose %We4, dims = [1, 0, 2, 3] : (tensor<128x32x1x1xf32>) -> tensor<32x128x1x1xf32>
    %v1232 = stablehlo.reverse %v1231, dims = [2, 3] : tensor<32x128x1x1xf32>
    %v1233 = stablehlo.convolution(%v1230, %v1232)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<32x128x1x1xf32>) -> tensor<32x32x28x28xf32>
    %v1234 = stablehlo.reshape %v1233 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v1235 = stablehlo.add %v1234, %v1008 : tensor<32x25088xf32>
    %v1236 = stablehlo.reshape %v278 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v1237 = stablehlo.reshape %v1229 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v1238 = stablehlo.transpose %v1236, dims = [1, 0, 2, 3] : (tensor<32x32x28x28xf32>) -> tensor<32x32x28x28xf32>
    %v1239 = stablehlo.transpose %v1237, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v1240 = stablehlo.convolution(%v1238, %v1239)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<32x128x1x1xf32>
    %v1241 = stablehlo.transpose %v1240, dims = [1, 0, 2, 3] : (tensor<32x128x1x1xf32>) -> tensor<128x32x1x1xf32>
    %v1242 = stablehlo.constant dense<0.3> : tensor<128x32x1x1xf32>
    %v1243 = stablehlo.multiply %v1241, %v1242 : tensor<128x32x1x1xf32>
    %v1244 = stablehlo.subtract %We4, %v1243 : tensor<128x32x1x1xf32>
    %v1245 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1246 = stablehlo.reshape %v283 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v1247 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v1248 = stablehlo.constant dense<1.0e-5> : tensor<32x128x28x28xf32>
    %v1249 = stablehlo.reduce(%v1246 init: %v1245) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v1250 = stablehlo.broadcast_in_dim %v1249, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v1251 = stablehlo.divide %v1250, %v1247 : tensor<32x128x28x28xf32>
    %v1252 = stablehlo.subtract %v1246, %v1251 : tensor<32x128x28x28xf32>
    %v1253 = stablehlo.multiply %v1252, %v1252 : tensor<32x128x28x28xf32>
    %v1254 = stablehlo.reduce(%v1253 init: %v1245) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v1255 = stablehlo.broadcast_in_dim %v1254, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v1256 = stablehlo.divide %v1255, %v1247 : tensor<32x128x28x28xf32>
    %v1257 = stablehlo.add %v1256, %v1248 : tensor<32x128x28x28xf32>
    %v1258 = stablehlo.rsqrt %v1257 : tensor<32x128x28x28xf32>
    %v1259 = stablehlo.multiply %v1252, %v1258 : tensor<32x128x28x28xf32>
    %v1260 = stablehlo.reshape %v1199 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v1261 = stablehlo.multiply %v1260, %v1259 : tensor<32x128x28x28xf32>
    %v1262 = stablehlo.reduce(%v1261 init: %v1245) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v1263 = stablehlo.constant dense<0.3> : tensor<128xf32>
    %v1264 = stablehlo.multiply %v1262, %v1263 : tensor<128xf32>
    %v1265 = stablehlo.subtract %ge4, %v1264 : tensor<128xf32>
    %v1266 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1267 = stablehlo.reshape %v1199 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v1268 = stablehlo.reduce(%v1267 init: %v1266) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v1269 = stablehlo.constant dense<0.3> : tensor<128xf32>
    %v1270 = stablehlo.multiply %v1268, %v1269 : tensor<128xf32>
    %v1271 = stablehlo.subtract %bte4, %v1270 : tensor<128xf32>
    %v1272 = stablehlo.reshape %v307 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v1273 = stablehlo.reshape %v1189 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v1274 = stablehlo.transpose %v1272, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v1275 = stablehlo.transpose %v1273, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v1276 = stablehlo.convolution(%v1274, %v1275)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 128 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<1x128x3x3xf32>
    %v1277 = stablehlo.reshape %v1276 : (tensor<1x128x3x3xf32>) -> tensor<128x1x3x3xf32>
    %v1278 = stablehlo.constant dense<0.3> : tensor<128x1x3x3xf32>
    %v1279 = stablehlo.multiply %v1277, %v1278 : tensor<128x1x3x3xf32>
    %v1280 = stablehlo.subtract %Wd4, %v1279 : tensor<128x1x3x3xf32>
    %v1281 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1282 = stablehlo.reshape %v312 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v1283 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v1284 = stablehlo.constant dense<1.0e-5> : tensor<32x128x28x28xf32>
    %v1285 = stablehlo.reduce(%v1282 init: %v1281) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v1286 = stablehlo.broadcast_in_dim %v1285, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v1287 = stablehlo.divide %v1286, %v1283 : tensor<32x128x28x28xf32>
    %v1288 = stablehlo.subtract %v1282, %v1287 : tensor<32x128x28x28xf32>
    %v1289 = stablehlo.multiply %v1288, %v1288 : tensor<32x128x28x28xf32>
    %v1290 = stablehlo.reduce(%v1289 init: %v1281) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v1291 = stablehlo.broadcast_in_dim %v1290, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v1292 = stablehlo.divide %v1291, %v1283 : tensor<32x128x28x28xf32>
    %v1293 = stablehlo.add %v1292, %v1284 : tensor<32x128x28x28xf32>
    %v1294 = stablehlo.rsqrt %v1293 : tensor<32x128x28x28xf32>
    %v1295 = stablehlo.multiply %v1288, %v1294 : tensor<32x128x28x28xf32>
    %v1296 = stablehlo.reshape %v1159 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v1297 = stablehlo.multiply %v1296, %v1295 : tensor<32x128x28x28xf32>
    %v1298 = stablehlo.reduce(%v1297 init: %v1281) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v1299 = stablehlo.constant dense<0.3> : tensor<128xf32>
    %v1300 = stablehlo.multiply %v1298, %v1299 : tensor<128xf32>
    %v1301 = stablehlo.subtract %gd4, %v1300 : tensor<128xf32>
    %v1302 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1303 = stablehlo.reshape %v1159 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v1304 = stablehlo.reduce(%v1303 init: %v1302) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v1305 = stablehlo.constant dense<0.3> : tensor<128xf32>
    %v1306 = stablehlo.multiply %v1304, %v1305 : tensor<128xf32>
    %v1307 = stablehlo.subtract %btd4, %v1306 : tensor<128xf32>
    %v1308 = stablehlo.reshape %v336 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v1309 = stablehlo.reshape %v1148 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v1310 = stablehlo.transpose %v1308, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v1311 = stablehlo.transpose %v1309, dims = [1, 0, 2, 3] : (tensor<32x32x28x28xf32>) -> tensor<32x32x28x28xf32>
    %v1312 = stablehlo.convolution(%v1310, %v1311)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<32x32x28x28xf32>) -> tensor<128x32x1x1xf32>
    %v1313 = stablehlo.transpose %v1312, dims = [1, 0, 2, 3] : (tensor<128x32x1x1xf32>) -> tensor<32x128x1x1xf32>
    %v1314 = stablehlo.constant dense<0.3> : tensor<32x128x1x1xf32>
    %v1315 = stablehlo.multiply %v1313, %v1314 : tensor<32x128x1x1xf32>
    %v1316 = stablehlo.subtract %Wp4, %v1315 : tensor<32x128x1x1xf32>
    %v1317 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1318 = stablehlo.reshape %v341 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v1319 = stablehlo.constant dense<784.0> : tensor<32x32x28x28xf32>
    %v1320 = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %v1321 = stablehlo.reduce(%v1318 init: %v1317) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v1322 = stablehlo.broadcast_in_dim %v1321, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v1323 = stablehlo.divide %v1322, %v1319 : tensor<32x32x28x28xf32>
    %v1324 = stablehlo.subtract %v1318, %v1323 : tensor<32x32x28x28xf32>
    %v1325 = stablehlo.multiply %v1324, %v1324 : tensor<32x32x28x28xf32>
    %v1326 = stablehlo.reduce(%v1325 init: %v1317) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v1327 = stablehlo.broadcast_in_dim %v1326, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v1328 = stablehlo.divide %v1327, %v1319 : tensor<32x32x28x28xf32>
    %v1329 = stablehlo.add %v1328, %v1320 : tensor<32x32x28x28xf32>
    %v1330 = stablehlo.rsqrt %v1329 : tensor<32x32x28x28xf32>
    %v1331 = stablehlo.multiply %v1324, %v1330 : tensor<32x32x28x28xf32>
    %v1332 = stablehlo.reshape %v1008 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v1333 = stablehlo.multiply %v1332, %v1331 : tensor<32x32x28x28xf32>
    %v1334 = stablehlo.reduce(%v1333 init: %v1317) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v1335 = stablehlo.constant dense<0.3> : tensor<32xf32>
    %v1336 = stablehlo.multiply %v1334, %v1335 : tensor<32xf32>
    %v1337 = stablehlo.subtract %gp4, %v1336 : tensor<32xf32>
    %v1338 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1339 = stablehlo.reshape %v1008 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v1340 = stablehlo.reduce(%v1339 init: %v1338) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v1341 = stablehlo.constant dense<0.3> : tensor<32xf32>
    %v1342 = stablehlo.multiply %v1340, %v1341 : tensor<32xf32>
    %v1343 = stablehlo.subtract %btp4, %v1342 : tensor<32xf32>
    %v1344 = stablehlo.reshape %v1235 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v1345 = stablehlo.reshape %v258 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v1346 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1347 = stablehlo.constant dense<784.0> : tensor<32x32x28x28xf32>
    %v1348 = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %v1349 = stablehlo.reduce(%v1345 init: %v1346) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v1350 = stablehlo.broadcast_in_dim %v1349, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v1351 = stablehlo.divide %v1350, %v1347 : tensor<32x32x28x28xf32>
    %v1352 = stablehlo.subtract %v1345, %v1351 : tensor<32x32x28x28xf32>
    %v1353 = stablehlo.multiply %v1352, %v1352 : tensor<32x32x28x28xf32>
    %v1354 = stablehlo.reduce(%v1353 init: %v1346) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v1355 = stablehlo.broadcast_in_dim %v1354, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v1356 = stablehlo.divide %v1355, %v1347 : tensor<32x32x28x28xf32>
    %v1357 = stablehlo.add %v1356, %v1348 : tensor<32x32x28x28xf32>
    %v1358 = stablehlo.rsqrt %v1357 : tensor<32x32x28x28xf32>
    %v1359 = stablehlo.multiply %v1352, %v1358 : tensor<32x32x28x28xf32>
    %v1360 = stablehlo.broadcast_in_dim %gp3, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v1361 = stablehlo.multiply %v1360, %v1344 : tensor<32x32x28x28xf32>
    %v1362 = stablehlo.reduce(%v1361 init: %v1346) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v1363 = stablehlo.broadcast_in_dim %v1362, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v1364 = stablehlo.multiply %v1359, %v1361 : tensor<32x32x28x28xf32>
    %v1365 = stablehlo.reduce(%v1364 init: %v1346) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v1366 = stablehlo.broadcast_in_dim %v1365, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v1367 = stablehlo.multiply %v1361, %v1347 : tensor<32x32x28x28xf32>
    %v1368 = stablehlo.subtract %v1367, %v1363 : tensor<32x32x28x28xf32>
    %v1369 = stablehlo.multiply %v1359, %v1366 : tensor<32x32x28x28xf32>
    %v1370 = stablehlo.subtract %v1368, %v1369 : tensor<32x32x28x28xf32>
    %v1371 = stablehlo.divide %v1358, %v1347 : tensor<32x32x28x28xf32>
    %v1372 = stablehlo.multiply %v1371, %v1370 : tensor<32x32x28x28xf32>
    %v1373 = stablehlo.reshape %v1372 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v1374 = stablehlo.reshape %v1373 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v1375 = stablehlo.transpose %Wp3, dims = [1, 0, 2, 3] : (tensor<32x96x1x1xf32>) -> tensor<96x32x1x1xf32>
    %v1376 = stablehlo.reverse %v1375, dims = [2, 3] : tensor<96x32x1x1xf32>
    %v1377 = stablehlo.convolution(%v1374, %v1376)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<96x32x1x1xf32>) -> tensor<32x96x28x28xf32>
    %v1378 = stablehlo.reshape %v1377 : (tensor<32x96x28x28xf32>) -> tensor<32x75264xf32>
    %v1379 = stablehlo.constant dense<0.0> : tensor<32x75264xf32>
    %v1380 = stablehlo.constant dense<6.0> : tensor<32x75264xf32>
    %v1381 = stablehlo.compare GT, %v249, %v1379 : (tensor<32x75264xf32>, tensor<32x75264xf32>) -> tensor<32x75264xi1>
    %v1382 = stablehlo.compare LT, %v249, %v1380 : (tensor<32x75264xf32>, tensor<32x75264xf32>) -> tensor<32x75264xi1>
    %v1383 = stablehlo.and %v1381, %v1382 : tensor<32x75264xi1>
    %v1384 = stablehlo.select %v1383, %v1378, %v1379 : tensor<32x75264xi1>, tensor<32x75264xf32>
    %v1385 = stablehlo.reshape %v1384 : (tensor<32x75264xf32>) -> tensor<32x96x28x28xf32>
    %v1386 = stablehlo.reshape %v229 : (tensor<32x75264xf32>) -> tensor<32x96x28x28xf32>
    %v1387 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1388 = stablehlo.constant dense<784.0> : tensor<32x96x28x28xf32>
    %v1389 = stablehlo.constant dense<1.0e-5> : tensor<32x96x28x28xf32>
    %v1390 = stablehlo.reduce(%v1386 init: %v1387) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x28x28xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1391 = stablehlo.broadcast_in_dim %v1390, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x28x28xf32>
    %v1392 = stablehlo.divide %v1391, %v1388 : tensor<32x96x28x28xf32>
    %v1393 = stablehlo.subtract %v1386, %v1392 : tensor<32x96x28x28xf32>
    %v1394 = stablehlo.multiply %v1393, %v1393 : tensor<32x96x28x28xf32>
    %v1395 = stablehlo.reduce(%v1394 init: %v1387) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x28x28xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1396 = stablehlo.broadcast_in_dim %v1395, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x28x28xf32>
    %v1397 = stablehlo.divide %v1396, %v1388 : tensor<32x96x28x28xf32>
    %v1398 = stablehlo.add %v1397, %v1389 : tensor<32x96x28x28xf32>
    %v1399 = stablehlo.rsqrt %v1398 : tensor<32x96x28x28xf32>
    %v1400 = stablehlo.multiply %v1393, %v1399 : tensor<32x96x28x28xf32>
    %v1401 = stablehlo.broadcast_in_dim %gd3, dims = [1] : (tensor<96xf32>) -> tensor<32x96x28x28xf32>
    %v1402 = stablehlo.multiply %v1401, %v1385 : tensor<32x96x28x28xf32>
    %v1403 = stablehlo.reduce(%v1402 init: %v1387) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x28x28xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1404 = stablehlo.broadcast_in_dim %v1403, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x28x28xf32>
    %v1405 = stablehlo.multiply %v1400, %v1402 : tensor<32x96x28x28xf32>
    %v1406 = stablehlo.reduce(%v1405 init: %v1387) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x28x28xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1407 = stablehlo.broadcast_in_dim %v1406, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x28x28xf32>
    %v1408 = stablehlo.multiply %v1402, %v1388 : tensor<32x96x28x28xf32>
    %v1409 = stablehlo.subtract %v1408, %v1404 : tensor<32x96x28x28xf32>
    %v1410 = stablehlo.multiply %v1400, %v1407 : tensor<32x96x28x28xf32>
    %v1411 = stablehlo.subtract %v1409, %v1410 : tensor<32x96x28x28xf32>
    %v1412 = stablehlo.divide %v1399, %v1388 : tensor<32x96x28x28xf32>
    %v1413 = stablehlo.multiply %v1412, %v1411 : tensor<32x96x28x28xf32>
    %v1414 = stablehlo.reshape %v1413 : (tensor<32x96x28x28xf32>) -> tensor<32x75264xf32>
    %v1415 = stablehlo.reshape %v1414 : (tensor<32x75264xf32>) -> tensor<32x96x28x28xf32>
    %v1416 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1417 = stablehlo.pad %v1415, %v1416, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x96x28x28xf32>, tensor<f32>) -> tensor<32x96x56x56xf32>
    %v1418 = stablehlo.reverse %Wd3, dims = [2, 3] : tensor<96x1x3x3xf32>
    %v1419 = stablehlo.convolution(%v1417, %v1418)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x3x3xf32>) -> tensor<32x96x56x56xf32>
    %v1420 = stablehlo.reshape %v1419 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v1421 = stablehlo.constant dense<0.0> : tensor<32x301056xf32>
    %v1422 = stablehlo.constant dense<6.0> : tensor<32x301056xf32>
    %v1423 = stablehlo.compare GT, %v220, %v1421 : (tensor<32x301056xf32>, tensor<32x301056xf32>) -> tensor<32x301056xi1>
    %v1424 = stablehlo.compare LT, %v220, %v1422 : (tensor<32x301056xf32>, tensor<32x301056xf32>) -> tensor<32x301056xi1>
    %v1425 = stablehlo.and %v1423, %v1424 : tensor<32x301056xi1>
    %v1426 = stablehlo.select %v1425, %v1420, %v1421 : tensor<32x301056xi1>, tensor<32x301056xf32>
    %v1427 = stablehlo.reshape %v1426 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1428 = stablehlo.reshape %v200 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1429 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1430 = stablehlo.constant dense<3136.0> : tensor<32x96x56x56xf32>
    %v1431 = stablehlo.constant dense<1.0e-5> : tensor<32x96x56x56xf32>
    %v1432 = stablehlo.reduce(%v1428 init: %v1429) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1433 = stablehlo.broadcast_in_dim %v1432, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v1434 = stablehlo.divide %v1433, %v1430 : tensor<32x96x56x56xf32>
    %v1435 = stablehlo.subtract %v1428, %v1434 : tensor<32x96x56x56xf32>
    %v1436 = stablehlo.multiply %v1435, %v1435 : tensor<32x96x56x56xf32>
    %v1437 = stablehlo.reduce(%v1436 init: %v1429) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1438 = stablehlo.broadcast_in_dim %v1437, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v1439 = stablehlo.divide %v1438, %v1430 : tensor<32x96x56x56xf32>
    %v1440 = stablehlo.add %v1439, %v1431 : tensor<32x96x56x56xf32>
    %v1441 = stablehlo.rsqrt %v1440 : tensor<32x96x56x56xf32>
    %v1442 = stablehlo.multiply %v1435, %v1441 : tensor<32x96x56x56xf32>
    %v1443 = stablehlo.broadcast_in_dim %ge3, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v1444 = stablehlo.multiply %v1443, %v1427 : tensor<32x96x56x56xf32>
    %v1445 = stablehlo.reduce(%v1444 init: %v1429) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1446 = stablehlo.broadcast_in_dim %v1445, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v1447 = stablehlo.multiply %v1442, %v1444 : tensor<32x96x56x56xf32>
    %v1448 = stablehlo.reduce(%v1447 init: %v1429) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1449 = stablehlo.broadcast_in_dim %v1448, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v1450 = stablehlo.multiply %v1444, %v1430 : tensor<32x96x56x56xf32>
    %v1451 = stablehlo.subtract %v1450, %v1446 : tensor<32x96x56x56xf32>
    %v1452 = stablehlo.multiply %v1442, %v1449 : tensor<32x96x56x56xf32>
    %v1453 = stablehlo.subtract %v1451, %v1452 : tensor<32x96x56x56xf32>
    %v1454 = stablehlo.divide %v1441, %v1430 : tensor<32x96x56x56xf32>
    %v1455 = stablehlo.multiply %v1454, %v1453 : tensor<32x96x56x56xf32>
    %v1456 = stablehlo.reshape %v1455 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v1457 = stablehlo.reshape %v1456 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1458 = stablehlo.transpose %We3, dims = [1, 0, 2, 3] : (tensor<96x24x1x1xf32>) -> tensor<24x96x1x1xf32>
    %v1459 = stablehlo.reverse %v1458, dims = [2, 3] : tensor<24x96x1x1xf32>
    %v1460 = stablehlo.convolution(%v1457, %v1459)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<24x96x1x1xf32>) -> tensor<32x24x56x56xf32>
    %v1461 = stablehlo.reshape %v1460 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v1462 = stablehlo.reshape %v195 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v1463 = stablehlo.reshape %v1456 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1464 = stablehlo.transpose %v1462, dims = [1, 0, 2, 3] : (tensor<32x24x56x56xf32>) -> tensor<24x32x56x56xf32>
    %v1465 = stablehlo.transpose %v1463, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v1466 = stablehlo.convolution(%v1464, %v1465)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<24x32x56x56xf32>, tensor<96x32x56x56xf32>) -> tensor<24x96x1x1xf32>
    %v1467 = stablehlo.transpose %v1466, dims = [1, 0, 2, 3] : (tensor<24x96x1x1xf32>) -> tensor<96x24x1x1xf32>
    %v1468 = stablehlo.constant dense<0.3> : tensor<96x24x1x1xf32>
    %v1469 = stablehlo.multiply %v1467, %v1468 : tensor<96x24x1x1xf32>
    %v1470 = stablehlo.subtract %We3, %v1469 : tensor<96x24x1x1xf32>
    %v1471 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1472 = stablehlo.reshape %v200 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1473 = stablehlo.constant dense<3136.0> : tensor<32x96x56x56xf32>
    %v1474 = stablehlo.constant dense<1.0e-5> : tensor<32x96x56x56xf32>
    %v1475 = stablehlo.reduce(%v1472 init: %v1471) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1476 = stablehlo.broadcast_in_dim %v1475, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v1477 = stablehlo.divide %v1476, %v1473 : tensor<32x96x56x56xf32>
    %v1478 = stablehlo.subtract %v1472, %v1477 : tensor<32x96x56x56xf32>
    %v1479 = stablehlo.multiply %v1478, %v1478 : tensor<32x96x56x56xf32>
    %v1480 = stablehlo.reduce(%v1479 init: %v1471) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1481 = stablehlo.broadcast_in_dim %v1480, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v1482 = stablehlo.divide %v1481, %v1473 : tensor<32x96x56x56xf32>
    %v1483 = stablehlo.add %v1482, %v1474 : tensor<32x96x56x56xf32>
    %v1484 = stablehlo.rsqrt %v1483 : tensor<32x96x56x56xf32>
    %v1485 = stablehlo.multiply %v1478, %v1484 : tensor<32x96x56x56xf32>
    %v1486 = stablehlo.reshape %v1426 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1487 = stablehlo.multiply %v1486, %v1485 : tensor<32x96x56x56xf32>
    %v1488 = stablehlo.reduce(%v1487 init: %v1471) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v1489 = stablehlo.constant dense<0.3> : tensor<96xf32>
    %v1490 = stablehlo.multiply %v1488, %v1489 : tensor<96xf32>
    %v1491 = stablehlo.subtract %ge3, %v1490 : tensor<96xf32>
    %v1492 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1493 = stablehlo.reshape %v1426 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1494 = stablehlo.reduce(%v1493 init: %v1492) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v1495 = stablehlo.constant dense<0.3> : tensor<96xf32>
    %v1496 = stablehlo.multiply %v1494, %v1495 : tensor<96xf32>
    %v1497 = stablehlo.subtract %bte3, %v1496 : tensor<96xf32>
    %v1498 = stablehlo.reshape %v224 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1499 = stablehlo.reshape %v1414 : (tensor<32x75264xf32>) -> tensor<32x96x28x28xf32>
    %v1500 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1501 = stablehlo.pad %v1499, %v1500, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x96x28x28xf32>, tensor<f32>) -> tensor<32x96x56x56xf32>
    %v1502 = stablehlo.transpose %v1498, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v1503 = stablehlo.transpose %v1501, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v1504 = stablehlo.convolution(%v1502, %v1503)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 96 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<96x32x56x56xf32>) -> tensor<1x96x3x3xf32>
    %v1505 = stablehlo.reshape %v1504 : (tensor<1x96x3x3xf32>) -> tensor<96x1x3x3xf32>
    %v1506 = stablehlo.constant dense<0.3> : tensor<96x1x3x3xf32>
    %v1507 = stablehlo.multiply %v1505, %v1506 : tensor<96x1x3x3xf32>
    %v1508 = stablehlo.subtract %Wd3, %v1507 : tensor<96x1x3x3xf32>
    %v1509 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1510 = stablehlo.reshape %v229 : (tensor<32x75264xf32>) -> tensor<32x96x28x28xf32>
    %v1511 = stablehlo.constant dense<784.0> : tensor<32x96x28x28xf32>
    %v1512 = stablehlo.constant dense<1.0e-5> : tensor<32x96x28x28xf32>
    %v1513 = stablehlo.reduce(%v1510 init: %v1509) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x28x28xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1514 = stablehlo.broadcast_in_dim %v1513, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x28x28xf32>
    %v1515 = stablehlo.divide %v1514, %v1511 : tensor<32x96x28x28xf32>
    %v1516 = stablehlo.subtract %v1510, %v1515 : tensor<32x96x28x28xf32>
    %v1517 = stablehlo.multiply %v1516, %v1516 : tensor<32x96x28x28xf32>
    %v1518 = stablehlo.reduce(%v1517 init: %v1509) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x28x28xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1519 = stablehlo.broadcast_in_dim %v1518, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x28x28xf32>
    %v1520 = stablehlo.divide %v1519, %v1511 : tensor<32x96x28x28xf32>
    %v1521 = stablehlo.add %v1520, %v1512 : tensor<32x96x28x28xf32>
    %v1522 = stablehlo.rsqrt %v1521 : tensor<32x96x28x28xf32>
    %v1523 = stablehlo.multiply %v1516, %v1522 : tensor<32x96x28x28xf32>
    %v1524 = stablehlo.reshape %v1384 : (tensor<32x75264xf32>) -> tensor<32x96x28x28xf32>
    %v1525 = stablehlo.multiply %v1524, %v1523 : tensor<32x96x28x28xf32>
    %v1526 = stablehlo.reduce(%v1525 init: %v1509) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x28x28xf32>, tensor<f32>) -> tensor<96xf32>
    %v1527 = stablehlo.constant dense<0.3> : tensor<96xf32>
    %v1528 = stablehlo.multiply %v1526, %v1527 : tensor<96xf32>
    %v1529 = stablehlo.subtract %gd3, %v1528 : tensor<96xf32>
    %v1530 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1531 = stablehlo.reshape %v1384 : (tensor<32x75264xf32>) -> tensor<32x96x28x28xf32>
    %v1532 = stablehlo.reduce(%v1531 init: %v1530) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x28x28xf32>, tensor<f32>) -> tensor<96xf32>
    %v1533 = stablehlo.constant dense<0.3> : tensor<96xf32>
    %v1534 = stablehlo.multiply %v1532, %v1533 : tensor<96xf32>
    %v1535 = stablehlo.subtract %btd3, %v1534 : tensor<96xf32>
    %v1536 = stablehlo.reshape %v253 : (tensor<32x75264xf32>) -> tensor<32x96x28x28xf32>
    %v1537 = stablehlo.reshape %v1373 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v1538 = stablehlo.transpose %v1536, dims = [1, 0, 2, 3] : (tensor<32x96x28x28xf32>) -> tensor<96x32x28x28xf32>
    %v1539 = stablehlo.transpose %v1537, dims = [1, 0, 2, 3] : (tensor<32x32x28x28xf32>) -> tensor<32x32x28x28xf32>
    %v1540 = stablehlo.convolution(%v1538, %v1539)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x28x28xf32>, tensor<32x32x28x28xf32>) -> tensor<96x32x1x1xf32>
    %v1541 = stablehlo.transpose %v1540, dims = [1, 0, 2, 3] : (tensor<96x32x1x1xf32>) -> tensor<32x96x1x1xf32>
    %v1542 = stablehlo.constant dense<0.3> : tensor<32x96x1x1xf32>
    %v1543 = stablehlo.multiply %v1541, %v1542 : tensor<32x96x1x1xf32>
    %v1544 = stablehlo.subtract %Wp3, %v1543 : tensor<32x96x1x1xf32>
    %v1545 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1546 = stablehlo.reshape %v258 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v1547 = stablehlo.constant dense<784.0> : tensor<32x32x28x28xf32>
    %v1548 = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %v1549 = stablehlo.reduce(%v1546 init: %v1545) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v1550 = stablehlo.broadcast_in_dim %v1549, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v1551 = stablehlo.divide %v1550, %v1547 : tensor<32x32x28x28xf32>
    %v1552 = stablehlo.subtract %v1546, %v1551 : tensor<32x32x28x28xf32>
    %v1553 = stablehlo.multiply %v1552, %v1552 : tensor<32x32x28x28xf32>
    %v1554 = stablehlo.reduce(%v1553 init: %v1545) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v1555 = stablehlo.broadcast_in_dim %v1554, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v1556 = stablehlo.divide %v1555, %v1547 : tensor<32x32x28x28xf32>
    %v1557 = stablehlo.add %v1556, %v1548 : tensor<32x32x28x28xf32>
    %v1558 = stablehlo.rsqrt %v1557 : tensor<32x32x28x28xf32>
    %v1559 = stablehlo.multiply %v1552, %v1558 : tensor<32x32x28x28xf32>
    %v1560 = stablehlo.reshape %v1235 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v1561 = stablehlo.multiply %v1560, %v1559 : tensor<32x32x28x28xf32>
    %v1562 = stablehlo.reduce(%v1561 init: %v1545) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v1563 = stablehlo.constant dense<0.3> : tensor<32xf32>
    %v1564 = stablehlo.multiply %v1562, %v1563 : tensor<32xf32>
    %v1565 = stablehlo.subtract %gp3, %v1564 : tensor<32xf32>
    %v1566 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1567 = stablehlo.reshape %v1235 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v1568 = stablehlo.reduce(%v1567 init: %v1566) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v1569 = stablehlo.constant dense<0.3> : tensor<32xf32>
    %v1570 = stablehlo.multiply %v1568, %v1569 : tensor<32xf32>
    %v1571 = stablehlo.subtract %btp3, %v1570 : tensor<32xf32>
    %v1572 = stablehlo.reshape %v1461 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v1573 = stablehlo.reshape %v174 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v1574 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1575 = stablehlo.constant dense<3136.0> : tensor<32x24x56x56xf32>
    %v1576 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v1577 = stablehlo.reduce(%v1573 init: %v1574) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v1578 = stablehlo.broadcast_in_dim %v1577, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v1579 = stablehlo.divide %v1578, %v1575 : tensor<32x24x56x56xf32>
    %v1580 = stablehlo.subtract %v1573, %v1579 : tensor<32x24x56x56xf32>
    %v1581 = stablehlo.multiply %v1580, %v1580 : tensor<32x24x56x56xf32>
    %v1582 = stablehlo.reduce(%v1581 init: %v1574) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v1583 = stablehlo.broadcast_in_dim %v1582, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v1584 = stablehlo.divide %v1583, %v1575 : tensor<32x24x56x56xf32>
    %v1585 = stablehlo.add %v1584, %v1576 : tensor<32x24x56x56xf32>
    %v1586 = stablehlo.rsqrt %v1585 : tensor<32x24x56x56xf32>
    %v1587 = stablehlo.multiply %v1580, %v1586 : tensor<32x24x56x56xf32>
    %v1588 = stablehlo.broadcast_in_dim %gp2, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v1589 = stablehlo.multiply %v1588, %v1572 : tensor<32x24x56x56xf32>
    %v1590 = stablehlo.reduce(%v1589 init: %v1574) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v1591 = stablehlo.broadcast_in_dim %v1590, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v1592 = stablehlo.multiply %v1587, %v1589 : tensor<32x24x56x56xf32>
    %v1593 = stablehlo.reduce(%v1592 init: %v1574) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v1594 = stablehlo.broadcast_in_dim %v1593, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v1595 = stablehlo.multiply %v1589, %v1575 : tensor<32x24x56x56xf32>
    %v1596 = stablehlo.subtract %v1595, %v1591 : tensor<32x24x56x56xf32>
    %v1597 = stablehlo.multiply %v1587, %v1594 : tensor<32x24x56x56xf32>
    %v1598 = stablehlo.subtract %v1596, %v1597 : tensor<32x24x56x56xf32>
    %v1599 = stablehlo.divide %v1586, %v1575 : tensor<32x24x56x56xf32>
    %v1600 = stablehlo.multiply %v1599, %v1598 : tensor<32x24x56x56xf32>
    %v1601 = stablehlo.reshape %v1600 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v1602 = stablehlo.reshape %v1601 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v1603 = stablehlo.transpose %Wp2, dims = [1, 0, 2, 3] : (tensor<24x96x1x1xf32>) -> tensor<96x24x1x1xf32>
    %v1604 = stablehlo.reverse %v1603, dims = [2, 3] : tensor<96x24x1x1xf32>
    %v1605 = stablehlo.convolution(%v1602, %v1604)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<96x24x1x1xf32>) -> tensor<32x96x56x56xf32>
    %v1606 = stablehlo.reshape %v1605 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v1607 = stablehlo.constant dense<0.0> : tensor<32x301056xf32>
    %v1608 = stablehlo.constant dense<6.0> : tensor<32x301056xf32>
    %v1609 = stablehlo.compare GT, %v165, %v1607 : (tensor<32x301056xf32>, tensor<32x301056xf32>) -> tensor<32x301056xi1>
    %v1610 = stablehlo.compare LT, %v165, %v1608 : (tensor<32x301056xf32>, tensor<32x301056xf32>) -> tensor<32x301056xi1>
    %v1611 = stablehlo.and %v1609, %v1610 : tensor<32x301056xi1>
    %v1612 = stablehlo.select %v1611, %v1606, %v1607 : tensor<32x301056xi1>, tensor<32x301056xf32>
    %v1613 = stablehlo.reshape %v1612 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1614 = stablehlo.reshape %v145 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1615 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1616 = stablehlo.constant dense<3136.0> : tensor<32x96x56x56xf32>
    %v1617 = stablehlo.constant dense<1.0e-5> : tensor<32x96x56x56xf32>
    %v1618 = stablehlo.reduce(%v1614 init: %v1615) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1619 = stablehlo.broadcast_in_dim %v1618, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v1620 = stablehlo.divide %v1619, %v1616 : tensor<32x96x56x56xf32>
    %v1621 = stablehlo.subtract %v1614, %v1620 : tensor<32x96x56x56xf32>
    %v1622 = stablehlo.multiply %v1621, %v1621 : tensor<32x96x56x56xf32>
    %v1623 = stablehlo.reduce(%v1622 init: %v1615) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1624 = stablehlo.broadcast_in_dim %v1623, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v1625 = stablehlo.divide %v1624, %v1616 : tensor<32x96x56x56xf32>
    %v1626 = stablehlo.add %v1625, %v1617 : tensor<32x96x56x56xf32>
    %v1627 = stablehlo.rsqrt %v1626 : tensor<32x96x56x56xf32>
    %v1628 = stablehlo.multiply %v1621, %v1627 : tensor<32x96x56x56xf32>
    %v1629 = stablehlo.broadcast_in_dim %gd2, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v1630 = stablehlo.multiply %v1629, %v1613 : tensor<32x96x56x56xf32>
    %v1631 = stablehlo.reduce(%v1630 init: %v1615) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1632 = stablehlo.broadcast_in_dim %v1631, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v1633 = stablehlo.multiply %v1628, %v1630 : tensor<32x96x56x56xf32>
    %v1634 = stablehlo.reduce(%v1633 init: %v1615) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1635 = stablehlo.broadcast_in_dim %v1634, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v1636 = stablehlo.multiply %v1630, %v1616 : tensor<32x96x56x56xf32>
    %v1637 = stablehlo.subtract %v1636, %v1632 : tensor<32x96x56x56xf32>
    %v1638 = stablehlo.multiply %v1628, %v1635 : tensor<32x96x56x56xf32>
    %v1639 = stablehlo.subtract %v1637, %v1638 : tensor<32x96x56x56xf32>
    %v1640 = stablehlo.divide %v1627, %v1616 : tensor<32x96x56x56xf32>
    %v1641 = stablehlo.multiply %v1640, %v1639 : tensor<32x96x56x56xf32>
    %v1642 = stablehlo.reshape %v1641 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v1643 = stablehlo.reshape %v1642 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1644 = stablehlo.reverse %Wd2, dims = [2, 3] : tensor<96x1x3x3xf32>
    %v1645 = stablehlo.convolution(%v1643, %v1644)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x3x3xf32>) -> tensor<32x96x56x56xf32>
    %v1646 = stablehlo.reshape %v1645 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v1647 = stablehlo.constant dense<0.0> : tensor<32x301056xf32>
    %v1648 = stablehlo.constant dense<6.0> : tensor<32x301056xf32>
    %v1649 = stablehlo.compare GT, %v136, %v1647 : (tensor<32x301056xf32>, tensor<32x301056xf32>) -> tensor<32x301056xi1>
    %v1650 = stablehlo.compare LT, %v136, %v1648 : (tensor<32x301056xf32>, tensor<32x301056xf32>) -> tensor<32x301056xi1>
    %v1651 = stablehlo.and %v1649, %v1650 : tensor<32x301056xi1>
    %v1652 = stablehlo.select %v1651, %v1646, %v1647 : tensor<32x301056xi1>, tensor<32x301056xf32>
    %v1653 = stablehlo.reshape %v1652 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1654 = stablehlo.reshape %v116 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1655 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1656 = stablehlo.constant dense<3136.0> : tensor<32x96x56x56xf32>
    %v1657 = stablehlo.constant dense<1.0e-5> : tensor<32x96x56x56xf32>
    %v1658 = stablehlo.reduce(%v1654 init: %v1655) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1659 = stablehlo.broadcast_in_dim %v1658, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v1660 = stablehlo.divide %v1659, %v1656 : tensor<32x96x56x56xf32>
    %v1661 = stablehlo.subtract %v1654, %v1660 : tensor<32x96x56x56xf32>
    %v1662 = stablehlo.multiply %v1661, %v1661 : tensor<32x96x56x56xf32>
    %v1663 = stablehlo.reduce(%v1662 init: %v1655) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1664 = stablehlo.broadcast_in_dim %v1663, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v1665 = stablehlo.divide %v1664, %v1656 : tensor<32x96x56x56xf32>
    %v1666 = stablehlo.add %v1665, %v1657 : tensor<32x96x56x56xf32>
    %v1667 = stablehlo.rsqrt %v1666 : tensor<32x96x56x56xf32>
    %v1668 = stablehlo.multiply %v1661, %v1667 : tensor<32x96x56x56xf32>
    %v1669 = stablehlo.broadcast_in_dim %ge2, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v1670 = stablehlo.multiply %v1669, %v1653 : tensor<32x96x56x56xf32>
    %v1671 = stablehlo.reduce(%v1670 init: %v1655) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1672 = stablehlo.broadcast_in_dim %v1671, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v1673 = stablehlo.multiply %v1668, %v1670 : tensor<32x96x56x56xf32>
    %v1674 = stablehlo.reduce(%v1673 init: %v1655) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1675 = stablehlo.broadcast_in_dim %v1674, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v1676 = stablehlo.multiply %v1670, %v1656 : tensor<32x96x56x56xf32>
    %v1677 = stablehlo.subtract %v1676, %v1672 : tensor<32x96x56x56xf32>
    %v1678 = stablehlo.multiply %v1668, %v1675 : tensor<32x96x56x56xf32>
    %v1679 = stablehlo.subtract %v1677, %v1678 : tensor<32x96x56x56xf32>
    %v1680 = stablehlo.divide %v1667, %v1656 : tensor<32x96x56x56xf32>
    %v1681 = stablehlo.multiply %v1680, %v1679 : tensor<32x96x56x56xf32>
    %v1682 = stablehlo.reshape %v1681 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v1683 = stablehlo.reshape %v1682 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1684 = stablehlo.transpose %We2, dims = [1, 0, 2, 3] : (tensor<96x24x1x1xf32>) -> tensor<24x96x1x1xf32>
    %v1685 = stablehlo.reverse %v1684, dims = [2, 3] : tensor<24x96x1x1xf32>
    %v1686 = stablehlo.convolution(%v1683, %v1685)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<24x96x1x1xf32>) -> tensor<32x24x56x56xf32>
    %v1687 = stablehlo.reshape %v1686 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v1688 = stablehlo.add %v1687, %v1461 : tensor<32x75264xf32>
    %v1689 = stablehlo.reshape %v111 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v1690 = stablehlo.reshape %v1682 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1691 = stablehlo.transpose %v1689, dims = [1, 0, 2, 3] : (tensor<32x24x56x56xf32>) -> tensor<24x32x56x56xf32>
    %v1692 = stablehlo.transpose %v1690, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v1693 = stablehlo.convolution(%v1691, %v1692)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<24x32x56x56xf32>, tensor<96x32x56x56xf32>) -> tensor<24x96x1x1xf32>
    %v1694 = stablehlo.transpose %v1693, dims = [1, 0, 2, 3] : (tensor<24x96x1x1xf32>) -> tensor<96x24x1x1xf32>
    %v1695 = stablehlo.constant dense<0.3> : tensor<96x24x1x1xf32>
    %v1696 = stablehlo.multiply %v1694, %v1695 : tensor<96x24x1x1xf32>
    %v1697 = stablehlo.subtract %We2, %v1696 : tensor<96x24x1x1xf32>
    %v1698 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1699 = stablehlo.reshape %v116 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1700 = stablehlo.constant dense<3136.0> : tensor<32x96x56x56xf32>
    %v1701 = stablehlo.constant dense<1.0e-5> : tensor<32x96x56x56xf32>
    %v1702 = stablehlo.reduce(%v1699 init: %v1698) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1703 = stablehlo.broadcast_in_dim %v1702, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v1704 = stablehlo.divide %v1703, %v1700 : tensor<32x96x56x56xf32>
    %v1705 = stablehlo.subtract %v1699, %v1704 : tensor<32x96x56x56xf32>
    %v1706 = stablehlo.multiply %v1705, %v1705 : tensor<32x96x56x56xf32>
    %v1707 = stablehlo.reduce(%v1706 init: %v1698) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1708 = stablehlo.broadcast_in_dim %v1707, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v1709 = stablehlo.divide %v1708, %v1700 : tensor<32x96x56x56xf32>
    %v1710 = stablehlo.add %v1709, %v1701 : tensor<32x96x56x56xf32>
    %v1711 = stablehlo.rsqrt %v1710 : tensor<32x96x56x56xf32>
    %v1712 = stablehlo.multiply %v1705, %v1711 : tensor<32x96x56x56xf32>
    %v1713 = stablehlo.reshape %v1652 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1714 = stablehlo.multiply %v1713, %v1712 : tensor<32x96x56x56xf32>
    %v1715 = stablehlo.reduce(%v1714 init: %v1698) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v1716 = stablehlo.constant dense<0.3> : tensor<96xf32>
    %v1717 = stablehlo.multiply %v1715, %v1716 : tensor<96xf32>
    %v1718 = stablehlo.subtract %ge2, %v1717 : tensor<96xf32>
    %v1719 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1720 = stablehlo.reshape %v1652 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1721 = stablehlo.reduce(%v1720 init: %v1719) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v1722 = stablehlo.constant dense<0.3> : tensor<96xf32>
    %v1723 = stablehlo.multiply %v1721, %v1722 : tensor<96xf32>
    %v1724 = stablehlo.subtract %bte2, %v1723 : tensor<96xf32>
    %v1725 = stablehlo.reshape %v140 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1726 = stablehlo.reshape %v1642 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1727 = stablehlo.transpose %v1725, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v1728 = stablehlo.transpose %v1726, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v1729 = stablehlo.convolution(%v1727, %v1728)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 96 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<96x32x56x56xf32>) -> tensor<1x96x3x3xf32>
    %v1730 = stablehlo.reshape %v1729 : (tensor<1x96x3x3xf32>) -> tensor<96x1x3x3xf32>
    %v1731 = stablehlo.constant dense<0.3> : tensor<96x1x3x3xf32>
    %v1732 = stablehlo.multiply %v1730, %v1731 : tensor<96x1x3x3xf32>
    %v1733 = stablehlo.subtract %Wd2, %v1732 : tensor<96x1x3x3xf32>
    %v1734 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1735 = stablehlo.reshape %v145 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1736 = stablehlo.constant dense<3136.0> : tensor<32x96x56x56xf32>
    %v1737 = stablehlo.constant dense<1.0e-5> : tensor<32x96x56x56xf32>
    %v1738 = stablehlo.reduce(%v1735 init: %v1734) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1739 = stablehlo.broadcast_in_dim %v1738, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v1740 = stablehlo.divide %v1739, %v1736 : tensor<32x96x56x56xf32>
    %v1741 = stablehlo.subtract %v1735, %v1740 : tensor<32x96x56x56xf32>
    %v1742 = stablehlo.multiply %v1741, %v1741 : tensor<32x96x56x56xf32>
    %v1743 = stablehlo.reduce(%v1742 init: %v1734) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1744 = stablehlo.broadcast_in_dim %v1743, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v1745 = stablehlo.divide %v1744, %v1736 : tensor<32x96x56x56xf32>
    %v1746 = stablehlo.add %v1745, %v1737 : tensor<32x96x56x56xf32>
    %v1747 = stablehlo.rsqrt %v1746 : tensor<32x96x56x56xf32>
    %v1748 = stablehlo.multiply %v1741, %v1747 : tensor<32x96x56x56xf32>
    %v1749 = stablehlo.reshape %v1612 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1750 = stablehlo.multiply %v1749, %v1748 : tensor<32x96x56x56xf32>
    %v1751 = stablehlo.reduce(%v1750 init: %v1734) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v1752 = stablehlo.constant dense<0.3> : tensor<96xf32>
    %v1753 = stablehlo.multiply %v1751, %v1752 : tensor<96xf32>
    %v1754 = stablehlo.subtract %gd2, %v1753 : tensor<96xf32>
    %v1755 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1756 = stablehlo.reshape %v1612 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1757 = stablehlo.reduce(%v1756 init: %v1755) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v1758 = stablehlo.constant dense<0.3> : tensor<96xf32>
    %v1759 = stablehlo.multiply %v1757, %v1758 : tensor<96xf32>
    %v1760 = stablehlo.subtract %btd2, %v1759 : tensor<96xf32>
    %v1761 = stablehlo.reshape %v169 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1762 = stablehlo.reshape %v1601 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v1763 = stablehlo.transpose %v1761, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v1764 = stablehlo.transpose %v1762, dims = [1, 0, 2, 3] : (tensor<32x24x56x56xf32>) -> tensor<24x32x56x56xf32>
    %v1765 = stablehlo.convolution(%v1763, %v1764)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<24x32x56x56xf32>) -> tensor<96x24x1x1xf32>
    %v1766 = stablehlo.transpose %v1765, dims = [1, 0, 2, 3] : (tensor<96x24x1x1xf32>) -> tensor<24x96x1x1xf32>
    %v1767 = stablehlo.constant dense<0.3> : tensor<24x96x1x1xf32>
    %v1768 = stablehlo.multiply %v1766, %v1767 : tensor<24x96x1x1xf32>
    %v1769 = stablehlo.subtract %Wp2, %v1768 : tensor<24x96x1x1xf32>
    %v1770 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1771 = stablehlo.reshape %v174 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v1772 = stablehlo.constant dense<3136.0> : tensor<32x24x56x56xf32>
    %v1773 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v1774 = stablehlo.reduce(%v1771 init: %v1770) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v1775 = stablehlo.broadcast_in_dim %v1774, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v1776 = stablehlo.divide %v1775, %v1772 : tensor<32x24x56x56xf32>
    %v1777 = stablehlo.subtract %v1771, %v1776 : tensor<32x24x56x56xf32>
    %v1778 = stablehlo.multiply %v1777, %v1777 : tensor<32x24x56x56xf32>
    %v1779 = stablehlo.reduce(%v1778 init: %v1770) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v1780 = stablehlo.broadcast_in_dim %v1779, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v1781 = stablehlo.divide %v1780, %v1772 : tensor<32x24x56x56xf32>
    %v1782 = stablehlo.add %v1781, %v1773 : tensor<32x24x56x56xf32>
    %v1783 = stablehlo.rsqrt %v1782 : tensor<32x24x56x56xf32>
    %v1784 = stablehlo.multiply %v1777, %v1783 : tensor<32x24x56x56xf32>
    %v1785 = stablehlo.reshape %v1461 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v1786 = stablehlo.multiply %v1785, %v1784 : tensor<32x24x56x56xf32>
    %v1787 = stablehlo.reduce(%v1786 init: %v1770) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v1788 = stablehlo.constant dense<0.3> : tensor<24xf32>
    %v1789 = stablehlo.multiply %v1787, %v1788 : tensor<24xf32>
    %v1790 = stablehlo.subtract %gp2, %v1789 : tensor<24xf32>
    %v1791 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1792 = stablehlo.reshape %v1461 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v1793 = stablehlo.reduce(%v1792 init: %v1791) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v1794 = stablehlo.constant dense<0.3> : tensor<24xf32>
    %v1795 = stablehlo.multiply %v1793, %v1794 : tensor<24xf32>
    %v1796 = stablehlo.subtract %btp2, %v1795 : tensor<24xf32>
    %v1797 = stablehlo.reshape %v1688 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v1798 = stablehlo.reshape %v91 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v1799 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1800 = stablehlo.constant dense<3136.0> : tensor<32x24x56x56xf32>
    %v1801 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v1802 = stablehlo.reduce(%v1798 init: %v1799) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v1803 = stablehlo.broadcast_in_dim %v1802, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v1804 = stablehlo.divide %v1803, %v1800 : tensor<32x24x56x56xf32>
    %v1805 = stablehlo.subtract %v1798, %v1804 : tensor<32x24x56x56xf32>
    %v1806 = stablehlo.multiply %v1805, %v1805 : tensor<32x24x56x56xf32>
    %v1807 = stablehlo.reduce(%v1806 init: %v1799) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v1808 = stablehlo.broadcast_in_dim %v1807, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v1809 = stablehlo.divide %v1808, %v1800 : tensor<32x24x56x56xf32>
    %v1810 = stablehlo.add %v1809, %v1801 : tensor<32x24x56x56xf32>
    %v1811 = stablehlo.rsqrt %v1810 : tensor<32x24x56x56xf32>
    %v1812 = stablehlo.multiply %v1805, %v1811 : tensor<32x24x56x56xf32>
    %v1813 = stablehlo.broadcast_in_dim %gp1, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v1814 = stablehlo.multiply %v1813, %v1797 : tensor<32x24x56x56xf32>
    %v1815 = stablehlo.reduce(%v1814 init: %v1799) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v1816 = stablehlo.broadcast_in_dim %v1815, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v1817 = stablehlo.multiply %v1812, %v1814 : tensor<32x24x56x56xf32>
    %v1818 = stablehlo.reduce(%v1817 init: %v1799) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v1819 = stablehlo.broadcast_in_dim %v1818, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v1820 = stablehlo.multiply %v1814, %v1800 : tensor<32x24x56x56xf32>
    %v1821 = stablehlo.subtract %v1820, %v1816 : tensor<32x24x56x56xf32>
    %v1822 = stablehlo.multiply %v1812, %v1819 : tensor<32x24x56x56xf32>
    %v1823 = stablehlo.subtract %v1821, %v1822 : tensor<32x24x56x56xf32>
    %v1824 = stablehlo.divide %v1811, %v1800 : tensor<32x24x56x56xf32>
    %v1825 = stablehlo.multiply %v1824, %v1823 : tensor<32x24x56x56xf32>
    %v1826 = stablehlo.reshape %v1825 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v1827 = stablehlo.reshape %v1826 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v1828 = stablehlo.transpose %Wp1, dims = [1, 0, 2, 3] : (tensor<24x64x1x1xf32>) -> tensor<64x24x1x1xf32>
    %v1829 = stablehlo.reverse %v1828, dims = [2, 3] : tensor<64x24x1x1xf32>
    %v1830 = stablehlo.convolution(%v1827, %v1829)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<64x24x1x1xf32>) -> tensor<32x64x56x56xf32>
    %v1831 = stablehlo.reshape %v1830 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v1832 = stablehlo.constant dense<0.0> : tensor<32x200704xf32>
    %v1833 = stablehlo.constant dense<6.0> : tensor<32x200704xf32>
    %v1834 = stablehlo.compare GT, %v82, %v1832 : (tensor<32x200704xf32>, tensor<32x200704xf32>) -> tensor<32x200704xi1>
    %v1835 = stablehlo.compare LT, %v82, %v1833 : (tensor<32x200704xf32>, tensor<32x200704xf32>) -> tensor<32x200704xi1>
    %v1836 = stablehlo.and %v1834, %v1835 : tensor<32x200704xi1>
    %v1837 = stablehlo.select %v1836, %v1831, %v1832 : tensor<32x200704xi1>, tensor<32x200704xf32>
    %v1838 = stablehlo.reshape %v1837 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v1839 = stablehlo.reshape %v62 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v1840 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1841 = stablehlo.constant dense<3136.0> : tensor<32x64x56x56xf32>
    %v1842 = stablehlo.constant dense<1.0e-5> : tensor<32x64x56x56xf32>
    %v1843 = stablehlo.reduce(%v1839 init: %v1840) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v1844 = stablehlo.broadcast_in_dim %v1843, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v1845 = stablehlo.divide %v1844, %v1841 : tensor<32x64x56x56xf32>
    %v1846 = stablehlo.subtract %v1839, %v1845 : tensor<32x64x56x56xf32>
    %v1847 = stablehlo.multiply %v1846, %v1846 : tensor<32x64x56x56xf32>
    %v1848 = stablehlo.reduce(%v1847 init: %v1840) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v1849 = stablehlo.broadcast_in_dim %v1848, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v1850 = stablehlo.divide %v1849, %v1841 : tensor<32x64x56x56xf32>
    %v1851 = stablehlo.add %v1850, %v1842 : tensor<32x64x56x56xf32>
    %v1852 = stablehlo.rsqrt %v1851 : tensor<32x64x56x56xf32>
    %v1853 = stablehlo.multiply %v1846, %v1852 : tensor<32x64x56x56xf32>
    %v1854 = stablehlo.broadcast_in_dim %gd1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v1855 = stablehlo.multiply %v1854, %v1838 : tensor<32x64x56x56xf32>
    %v1856 = stablehlo.reduce(%v1855 init: %v1840) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v1857 = stablehlo.broadcast_in_dim %v1856, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v1858 = stablehlo.multiply %v1853, %v1855 : tensor<32x64x56x56xf32>
    %v1859 = stablehlo.reduce(%v1858 init: %v1840) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v1860 = stablehlo.broadcast_in_dim %v1859, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v1861 = stablehlo.multiply %v1855, %v1841 : tensor<32x64x56x56xf32>
    %v1862 = stablehlo.subtract %v1861, %v1857 : tensor<32x64x56x56xf32>
    %v1863 = stablehlo.multiply %v1853, %v1860 : tensor<32x64x56x56xf32>
    %v1864 = stablehlo.subtract %v1862, %v1863 : tensor<32x64x56x56xf32>
    %v1865 = stablehlo.divide %v1852, %v1841 : tensor<32x64x56x56xf32>
    %v1866 = stablehlo.multiply %v1865, %v1864 : tensor<32x64x56x56xf32>
    %v1867 = stablehlo.reshape %v1866 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v1868 = stablehlo.reshape %v1867 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v1869 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1870 = stablehlo.pad %v1868, %v1869, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64x112x112xf32>
    %v1871 = stablehlo.reverse %Wd1, dims = [2, 3] : tensor<64x1x3x3xf32>
    %v1872 = stablehlo.convolution(%v1870, %v1871)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 64 : i64} : (tensor<32x64x112x112xf32>, tensor<64x1x3x3xf32>) -> tensor<32x64x112x112xf32>
    %v1873 = stablehlo.reshape %v1872 : (tensor<32x64x112x112xf32>) -> tensor<32x802816xf32>
    %v1874 = stablehlo.constant dense<0.0> : tensor<32x802816xf32>
    %v1875 = stablehlo.constant dense<6.0> : tensor<32x802816xf32>
    %v1876 = stablehlo.compare GT, %v53, %v1874 : (tensor<32x802816xf32>, tensor<32x802816xf32>) -> tensor<32x802816xi1>
    %v1877 = stablehlo.compare LT, %v53, %v1875 : (tensor<32x802816xf32>, tensor<32x802816xf32>) -> tensor<32x802816xi1>
    %v1878 = stablehlo.and %v1876, %v1877 : tensor<32x802816xi1>
    %v1879 = stablehlo.select %v1878, %v1873, %v1874 : tensor<32x802816xi1>, tensor<32x802816xf32>
    %v1880 = stablehlo.reshape %v1879 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v1881 = stablehlo.reshape %v33 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v1882 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1883 = stablehlo.constant dense<12544.0> : tensor<32x64x112x112xf32>
    %v1884 = stablehlo.constant dense<1.0e-5> : tensor<32x64x112x112xf32>
    %v1885 = stablehlo.reduce(%v1881 init: %v1882) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v1886 = stablehlo.broadcast_in_dim %v1885, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x112x112xf32>
    %v1887 = stablehlo.divide %v1886, %v1883 : tensor<32x64x112x112xf32>
    %v1888 = stablehlo.subtract %v1881, %v1887 : tensor<32x64x112x112xf32>
    %v1889 = stablehlo.multiply %v1888, %v1888 : tensor<32x64x112x112xf32>
    %v1890 = stablehlo.reduce(%v1889 init: %v1882) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v1891 = stablehlo.broadcast_in_dim %v1890, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x112x112xf32>
    %v1892 = stablehlo.divide %v1891, %v1883 : tensor<32x64x112x112xf32>
    %v1893 = stablehlo.add %v1892, %v1884 : tensor<32x64x112x112xf32>
    %v1894 = stablehlo.rsqrt %v1893 : tensor<32x64x112x112xf32>
    %v1895 = stablehlo.multiply %v1888, %v1894 : tensor<32x64x112x112xf32>
    %v1896 = stablehlo.broadcast_in_dim %ge1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v1897 = stablehlo.multiply %v1896, %v1880 : tensor<32x64x112x112xf32>
    %v1898 = stablehlo.reduce(%v1897 init: %v1882) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v1899 = stablehlo.broadcast_in_dim %v1898, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x112x112xf32>
    %v1900 = stablehlo.multiply %v1895, %v1897 : tensor<32x64x112x112xf32>
    %v1901 = stablehlo.reduce(%v1900 init: %v1882) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v1902 = stablehlo.broadcast_in_dim %v1901, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x112x112xf32>
    %v1903 = stablehlo.multiply %v1897, %v1883 : tensor<32x64x112x112xf32>
    %v1904 = stablehlo.subtract %v1903, %v1899 : tensor<32x64x112x112xf32>
    %v1905 = stablehlo.multiply %v1895, %v1902 : tensor<32x64x112x112xf32>
    %v1906 = stablehlo.subtract %v1904, %v1905 : tensor<32x64x112x112xf32>
    %v1907 = stablehlo.divide %v1894, %v1883 : tensor<32x64x112x112xf32>
    %v1908 = stablehlo.multiply %v1907, %v1906 : tensor<32x64x112x112xf32>
    %v1909 = stablehlo.reshape %v1908 : (tensor<32x64x112x112xf32>) -> tensor<32x802816xf32>
    %v1910 = stablehlo.reshape %v1909 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v1911 = stablehlo.transpose %We1, dims = [1, 0, 2, 3] : (tensor<64x16x1x1xf32>) -> tensor<16x64x1x1xf32>
    %v1912 = stablehlo.reverse %v1911, dims = [2, 3] : tensor<16x64x1x1xf32>
    %v1913 = stablehlo.convolution(%v1910, %v1912)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x112x112xf32>, tensor<16x64x1x1xf32>) -> tensor<32x16x112x112xf32>
    %v1914 = stablehlo.reshape %v1913 : (tensor<32x16x112x112xf32>) -> tensor<32x200704xf32>
    %v1915 = stablehlo.reshape %v28 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v1916 = stablehlo.reshape %v1909 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v1917 = stablehlo.transpose %v1915, dims = [1, 0, 2, 3] : (tensor<32x16x112x112xf32>) -> tensor<16x32x112x112xf32>
    %v1918 = stablehlo.transpose %v1916, dims = [1, 0, 2, 3] : (tensor<32x64x112x112xf32>) -> tensor<64x32x112x112xf32>
    %v1919 = stablehlo.convolution(%v1917, %v1918)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x32x112x112xf32>, tensor<64x32x112x112xf32>) -> tensor<16x64x1x1xf32>
    %v1920 = stablehlo.transpose %v1919, dims = [1, 0, 2, 3] : (tensor<16x64x1x1xf32>) -> tensor<64x16x1x1xf32>
    %v1921 = stablehlo.constant dense<0.3> : tensor<64x16x1x1xf32>
    %v1922 = stablehlo.multiply %v1920, %v1921 : tensor<64x16x1x1xf32>
    %v1923 = stablehlo.subtract %We1, %v1922 : tensor<64x16x1x1xf32>
    %v1924 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1925 = stablehlo.reshape %v33 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v1926 = stablehlo.constant dense<12544.0> : tensor<32x64x112x112xf32>
    %v1927 = stablehlo.constant dense<1.0e-5> : tensor<32x64x112x112xf32>
    %v1928 = stablehlo.reduce(%v1925 init: %v1924) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v1929 = stablehlo.broadcast_in_dim %v1928, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x112x112xf32>
    %v1930 = stablehlo.divide %v1929, %v1926 : tensor<32x64x112x112xf32>
    %v1931 = stablehlo.subtract %v1925, %v1930 : tensor<32x64x112x112xf32>
    %v1932 = stablehlo.multiply %v1931, %v1931 : tensor<32x64x112x112xf32>
    %v1933 = stablehlo.reduce(%v1932 init: %v1924) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v1934 = stablehlo.broadcast_in_dim %v1933, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x112x112xf32>
    %v1935 = stablehlo.divide %v1934, %v1926 : tensor<32x64x112x112xf32>
    %v1936 = stablehlo.add %v1935, %v1927 : tensor<32x64x112x112xf32>
    %v1937 = stablehlo.rsqrt %v1936 : tensor<32x64x112x112xf32>
    %v1938 = stablehlo.multiply %v1931, %v1937 : tensor<32x64x112x112xf32>
    %v1939 = stablehlo.reshape %v1879 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v1940 = stablehlo.multiply %v1939, %v1938 : tensor<32x64x112x112xf32>
    %v1941 = stablehlo.reduce(%v1940 init: %v1924) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v1942 = stablehlo.constant dense<0.3> : tensor<64xf32>
    %v1943 = stablehlo.multiply %v1941, %v1942 : tensor<64xf32>
    %v1944 = stablehlo.subtract %ge1, %v1943 : tensor<64xf32>
    %v1945 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1946 = stablehlo.reshape %v1879 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v1947 = stablehlo.reduce(%v1946 init: %v1945) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v1948 = stablehlo.constant dense<0.3> : tensor<64xf32>
    %v1949 = stablehlo.multiply %v1947, %v1948 : tensor<64xf32>
    %v1950 = stablehlo.subtract %bte1, %v1949 : tensor<64xf32>
    %v1951 = stablehlo.reshape %v57 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v1952 = stablehlo.reshape %v1867 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v1953 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1954 = stablehlo.pad %v1952, %v1953, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64x112x112xf32>
    %v1955 = stablehlo.transpose %v1951, dims = [1, 0, 2, 3] : (tensor<32x64x112x112xf32>) -> tensor<64x32x112x112xf32>
    %v1956 = stablehlo.transpose %v1954, dims = [1, 0, 2, 3] : (tensor<32x64x112x112xf32>) -> tensor<64x32x112x112xf32>
    %v1957 = stablehlo.convolution(%v1955, %v1956)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 64 : i64, feature_group_count = 1 : i64} : (tensor<64x32x112x112xf32>, tensor<64x32x112x112xf32>) -> tensor<1x64x3x3xf32>
    %v1958 = stablehlo.reshape %v1957 : (tensor<1x64x3x3xf32>) -> tensor<64x1x3x3xf32>
    %v1959 = stablehlo.constant dense<0.3> : tensor<64x1x3x3xf32>
    %v1960 = stablehlo.multiply %v1958, %v1959 : tensor<64x1x3x3xf32>
    %v1961 = stablehlo.subtract %Wd1, %v1960 : tensor<64x1x3x3xf32>
    %v1962 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1963 = stablehlo.reshape %v62 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v1964 = stablehlo.constant dense<3136.0> : tensor<32x64x56x56xf32>
    %v1965 = stablehlo.constant dense<1.0e-5> : tensor<32x64x56x56xf32>
    %v1966 = stablehlo.reduce(%v1963 init: %v1962) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v1967 = stablehlo.broadcast_in_dim %v1966, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v1968 = stablehlo.divide %v1967, %v1964 : tensor<32x64x56x56xf32>
    %v1969 = stablehlo.subtract %v1963, %v1968 : tensor<32x64x56x56xf32>
    %v1970 = stablehlo.multiply %v1969, %v1969 : tensor<32x64x56x56xf32>
    %v1971 = stablehlo.reduce(%v1970 init: %v1962) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v1972 = stablehlo.broadcast_in_dim %v1971, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v1973 = stablehlo.divide %v1972, %v1964 : tensor<32x64x56x56xf32>
    %v1974 = stablehlo.add %v1973, %v1965 : tensor<32x64x56x56xf32>
    %v1975 = stablehlo.rsqrt %v1974 : tensor<32x64x56x56xf32>
    %v1976 = stablehlo.multiply %v1969, %v1975 : tensor<32x64x56x56xf32>
    %v1977 = stablehlo.reshape %v1837 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v1978 = stablehlo.multiply %v1977, %v1976 : tensor<32x64x56x56xf32>
    %v1979 = stablehlo.reduce(%v1978 init: %v1962) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v1980 = stablehlo.constant dense<0.3> : tensor<64xf32>
    %v1981 = stablehlo.multiply %v1979, %v1980 : tensor<64xf32>
    %v1982 = stablehlo.subtract %gd1, %v1981 : tensor<64xf32>
    %v1983 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1984 = stablehlo.reshape %v1837 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v1985 = stablehlo.reduce(%v1984 init: %v1983) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v1986 = stablehlo.constant dense<0.3> : tensor<64xf32>
    %v1987 = stablehlo.multiply %v1985, %v1986 : tensor<64xf32>
    %v1988 = stablehlo.subtract %btd1, %v1987 : tensor<64xf32>
    %v1989 = stablehlo.reshape %v86 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v1990 = stablehlo.reshape %v1826 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v1991 = stablehlo.transpose %v1989, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v1992 = stablehlo.transpose %v1990, dims = [1, 0, 2, 3] : (tensor<32x24x56x56xf32>) -> tensor<24x32x56x56xf32>
    %v1993 = stablehlo.convolution(%v1991, %v1992)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<24x32x56x56xf32>) -> tensor<64x24x1x1xf32>
    %v1994 = stablehlo.transpose %v1993, dims = [1, 0, 2, 3] : (tensor<64x24x1x1xf32>) -> tensor<24x64x1x1xf32>
    %v1995 = stablehlo.constant dense<0.3> : tensor<24x64x1x1xf32>
    %v1996 = stablehlo.multiply %v1994, %v1995 : tensor<24x64x1x1xf32>
    %v1997 = stablehlo.subtract %Wp1, %v1996 : tensor<24x64x1x1xf32>
    %v1998 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1999 = stablehlo.reshape %v91 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v2000 = stablehlo.constant dense<3136.0> : tensor<32x24x56x56xf32>
    %v2001 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v2002 = stablehlo.reduce(%v1999 init: %v1998) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v2003 = stablehlo.broadcast_in_dim %v2002, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v2004 = stablehlo.divide %v2003, %v2000 : tensor<32x24x56x56xf32>
    %v2005 = stablehlo.subtract %v1999, %v2004 : tensor<32x24x56x56xf32>
    %v2006 = stablehlo.multiply %v2005, %v2005 : tensor<32x24x56x56xf32>
    %v2007 = stablehlo.reduce(%v2006 init: %v1998) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v2008 = stablehlo.broadcast_in_dim %v2007, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v2009 = stablehlo.divide %v2008, %v2000 : tensor<32x24x56x56xf32>
    %v2010 = stablehlo.add %v2009, %v2001 : tensor<32x24x56x56xf32>
    %v2011 = stablehlo.rsqrt %v2010 : tensor<32x24x56x56xf32>
    %v2012 = stablehlo.multiply %v2005, %v2011 : tensor<32x24x56x56xf32>
    %v2013 = stablehlo.reshape %v1688 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v2014 = stablehlo.multiply %v2013, %v2012 : tensor<32x24x56x56xf32>
    %v2015 = stablehlo.reduce(%v2014 init: %v1998) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v2016 = stablehlo.constant dense<0.3> : tensor<24xf32>
    %v2017 = stablehlo.multiply %v2015, %v2016 : tensor<24xf32>
    %v2018 = stablehlo.subtract %gp1, %v2017 : tensor<24xf32>
    %v2019 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2020 = stablehlo.reshape %v1688 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v2021 = stablehlo.reduce(%v2020 init: %v2019) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v2022 = stablehlo.constant dense<0.3> : tensor<24xf32>
    %v2023 = stablehlo.multiply %v2021, %v2022 : tensor<24xf32>
    %v2024 = stablehlo.subtract %btp1, %v2023 : tensor<24xf32>
    %v2025 = stablehlo.constant dense<0.0> : tensor<32x200704xf32>
    %v2026 = stablehlo.constant dense<6.0> : tensor<32x200704xf32>
    %v2027 = stablehlo.compare GT, %v24, %v2025 : (tensor<32x200704xf32>, tensor<32x200704xf32>) -> tensor<32x200704xi1>
    %v2028 = stablehlo.compare LT, %v24, %v2026 : (tensor<32x200704xf32>, tensor<32x200704xf32>) -> tensor<32x200704xi1>
    %v2029 = stablehlo.and %v2027, %v2028 : tensor<32x200704xi1>
    %v2030 = stablehlo.select %v2029, %v1914, %v2025 : tensor<32x200704xi1>, tensor<32x200704xf32>
    %v2031 = stablehlo.reshape %v2030 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v2032 = stablehlo.reshape %v4 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v2033 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2034 = stablehlo.constant dense<12544.0> : tensor<32x16x112x112xf32>
    %v2035 = stablehlo.constant dense<1.0e-5> : tensor<32x16x112x112xf32>
    %v2036 = stablehlo.reduce(%v2032 init: %v2033) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<32x16xf32>
    %v2037 = stablehlo.broadcast_in_dim %v2036, dims = [0, 1] : (tensor<32x16xf32>) -> tensor<32x16x112x112xf32>
    %v2038 = stablehlo.divide %v2037, %v2034 : tensor<32x16x112x112xf32>
    %v2039 = stablehlo.subtract %v2032, %v2038 : tensor<32x16x112x112xf32>
    %v2040 = stablehlo.multiply %v2039, %v2039 : tensor<32x16x112x112xf32>
    %v2041 = stablehlo.reduce(%v2040 init: %v2033) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<32x16xf32>
    %v2042 = stablehlo.broadcast_in_dim %v2041, dims = [0, 1] : (tensor<32x16xf32>) -> tensor<32x16x112x112xf32>
    %v2043 = stablehlo.divide %v2042, %v2034 : tensor<32x16x112x112xf32>
    %v2044 = stablehlo.add %v2043, %v2035 : tensor<32x16x112x112xf32>
    %v2045 = stablehlo.rsqrt %v2044 : tensor<32x16x112x112xf32>
    %v2046 = stablehlo.multiply %v2039, %v2045 : tensor<32x16x112x112xf32>
    %v2047 = stablehlo.broadcast_in_dim %gs, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v2048 = stablehlo.multiply %v2047, %v2031 : tensor<32x16x112x112xf32>
    %v2049 = stablehlo.reduce(%v2048 init: %v2033) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<32x16xf32>
    %v2050 = stablehlo.broadcast_in_dim %v2049, dims = [0, 1] : (tensor<32x16xf32>) -> tensor<32x16x112x112xf32>
    %v2051 = stablehlo.multiply %v2046, %v2048 : tensor<32x16x112x112xf32>
    %v2052 = stablehlo.reduce(%v2051 init: %v2033) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<32x16xf32>
    %v2053 = stablehlo.broadcast_in_dim %v2052, dims = [0, 1] : (tensor<32x16xf32>) -> tensor<32x16x112x112xf32>
    %v2054 = stablehlo.multiply %v2048, %v2034 : tensor<32x16x112x112xf32>
    %v2055 = stablehlo.subtract %v2054, %v2050 : tensor<32x16x112x112xf32>
    %v2056 = stablehlo.multiply %v2046, %v2053 : tensor<32x16x112x112xf32>
    %v2057 = stablehlo.subtract %v2055, %v2056 : tensor<32x16x112x112xf32>
    %v2058 = stablehlo.divide %v2045, %v2034 : tensor<32x16x112x112xf32>
    %v2059 = stablehlo.multiply %v2058, %v2057 : tensor<32x16x112x112xf32>
    %v2060 = stablehlo.reshape %v2059 : (tensor<32x16x112x112xf32>) -> tensor<32x200704xf32>
    %v2061 = stablehlo.reshape %x : (tensor<32x150528xf32>) -> tensor<32x3x224x224xf32>
    %v2062 = stablehlo.reshape %v2060 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v2063 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2064 = stablehlo.pad %v2062, %v2063, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<32x16x224x224xf32>
    %v2065 = stablehlo.transpose %v2061, dims = [1, 0, 2, 3] : (tensor<32x3x224x224xf32>) -> tensor<3x32x224x224xf32>
    %v2066 = stablehlo.transpose %v2064, dims = [1, 0, 2, 3] : (tensor<32x16x224x224xf32>) -> tensor<16x32x224x224xf32>
    %v2067 = stablehlo.convolution(%v2065, %v2066)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x32x224x224xf32>, tensor<16x32x224x224xf32>) -> tensor<3x16x3x3xf32>
    %v2068 = stablehlo.transpose %v2067, dims = [1, 0, 2, 3] : (tensor<3x16x3x3xf32>) -> tensor<16x3x3x3xf32>
    %v2069 = stablehlo.constant dense<0.3> : tensor<16x3x3x3xf32>
    %v2070 = stablehlo.multiply %v2068, %v2069 : tensor<16x3x3x3xf32>
    %v2071 = stablehlo.subtract %Ws, %v2070 : tensor<16x3x3x3xf32>
    %v2072 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2073 = stablehlo.reshape %v4 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v2074 = stablehlo.constant dense<12544.0> : tensor<32x16x112x112xf32>
    %v2075 = stablehlo.constant dense<1.0e-5> : tensor<32x16x112x112xf32>
    %v2076 = stablehlo.reduce(%v2073 init: %v2072) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<32x16xf32>
    %v2077 = stablehlo.broadcast_in_dim %v2076, dims = [0, 1] : (tensor<32x16xf32>) -> tensor<32x16x112x112xf32>
    %v2078 = stablehlo.divide %v2077, %v2074 : tensor<32x16x112x112xf32>
    %v2079 = stablehlo.subtract %v2073, %v2078 : tensor<32x16x112x112xf32>
    %v2080 = stablehlo.multiply %v2079, %v2079 : tensor<32x16x112x112xf32>
    %v2081 = stablehlo.reduce(%v2080 init: %v2072) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<32x16xf32>
    %v2082 = stablehlo.broadcast_in_dim %v2081, dims = [0, 1] : (tensor<32x16xf32>) -> tensor<32x16x112x112xf32>
    %v2083 = stablehlo.divide %v2082, %v2074 : tensor<32x16x112x112xf32>
    %v2084 = stablehlo.add %v2083, %v2075 : tensor<32x16x112x112xf32>
    %v2085 = stablehlo.rsqrt %v2084 : tensor<32x16x112x112xf32>
    %v2086 = stablehlo.multiply %v2079, %v2085 : tensor<32x16x112x112xf32>
    %v2087 = stablehlo.reshape %v2030 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v2088 = stablehlo.multiply %v2087, %v2086 : tensor<32x16x112x112xf32>
    %v2089 = stablehlo.reduce(%v2088 init: %v2072) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v2090 = stablehlo.constant dense<0.3> : tensor<16xf32>
    %v2091 = stablehlo.multiply %v2089, %v2090 : tensor<16xf32>
    %v2092 = stablehlo.subtract %gs, %v2091 : tensor<16xf32>
    %v2093 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2094 = stablehlo.reshape %v2030 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v2095 = stablehlo.reduce(%v2094 init: %v2093) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v2096 = stablehlo.constant dense<0.3> : tensor<16xf32>
    %v2097 = stablehlo.multiply %v2095, %v2096 : tensor<16xf32>
    %v2098 = stablehlo.subtract %bts, %v2097 : tensor<16xf32>
    return %v2071, %v2092, %v2098, %v1923, %v1944, %v1950, %v1961, %v1982, %v1988, %v1997, %v2018, %v2024, %v1697, %v1718, %v1724, %v1733, %v1754, %v1760, %v1769, %v1790, %v1796, %v1470, %v1491, %v1497, %v1508, %v1529, %v1535, %v1544, %v1565, %v1571, %v1244, %v1265, %v1271, %v1280, %v1301, %v1307, %v1316, %v1337, %v1343, %v1017, %v1038, %v1044, %v1055, %v1076, %v1082, %v1091, %v1112, %v1118, %v789, %v810, %v816, %v827, %v848, %v854, %v863, %v884, %v890, %v635, %v656, %v662, %v580, %v585 : tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<64x16x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x1x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<24x64x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<96x24x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x3x3xf32>, tensor<96xf32>, tensor<96xf32>, tensor<24x96x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<96x24x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x3x3xf32>, tensor<96xf32>, tensor<96xf32>, tensor<32x96x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x32x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x1x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<32x128x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x32x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x1x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<64x128x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x1x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x10xf32>, tensor<10xf32>
  }
}

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
    %v25 = stablehlo.reshape %v24 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v26 = stablehlo.constant dense<0.0> : tensor<32x16x112x112xf32>
    %v27 = stablehlo.constant dense<6.0> : tensor<32x16x112x112xf32>
    %v28 = stablehlo.maximum %v25, %v26 : tensor<32x16x112x112xf32>
    %v29 = stablehlo.minimum %v28, %v27 : tensor<32x16x112x112xf32>
    %v30 = stablehlo.reshape %v29 : (tensor<32x16x112x112xf32>) -> tensor<32x200704xf32>
    %v31 = stablehlo.reshape %v30 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v32 = stablehlo.convolution(%v31, %We1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x16x112x112xf32>, tensor<64x16x1x1xf32>) -> tensor<32x64x112x112xf32>
    %v33 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v34 = stablehlo.add %v32, %v33 : tensor<32x64x112x112xf32>
    %v35 = stablehlo.reshape %v34 : (tensor<32x64x112x112xf32>) -> tensor<32x802816xf32>
    %v36 = stablehlo.reshape %v35 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v37 = stablehlo.constant dense<0.0> : tensor<f32>
    %v38 = stablehlo.constant dense<12544.0> : tensor<32x64x112x112xf32>
    %v39 = stablehlo.constant dense<1.0e-5> : tensor<32x64x112x112xf32>
    %v40 = stablehlo.reduce(%v36 init: %v37) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v41 = stablehlo.broadcast_in_dim %v40, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x112x112xf32>
    %v42 = stablehlo.divide %v41, %v38 : tensor<32x64x112x112xf32>
    %v43 = stablehlo.subtract %v36, %v42 : tensor<32x64x112x112xf32>
    %v44 = stablehlo.multiply %v43, %v43 : tensor<32x64x112x112xf32>
    %v45 = stablehlo.reduce(%v44 init: %v37) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v46 = stablehlo.broadcast_in_dim %v45, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x112x112xf32>
    %v47 = stablehlo.divide %v46, %v38 : tensor<32x64x112x112xf32>
    %v48 = stablehlo.add %v47, %v39 : tensor<32x64x112x112xf32>
    %v49 = stablehlo.rsqrt %v48 : tensor<32x64x112x112xf32>
    %v50 = stablehlo.multiply %v43, %v49 : tensor<32x64x112x112xf32>
    %v51 = stablehlo.broadcast_in_dim %ge1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v52 = stablehlo.broadcast_in_dim %bte1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v53 = stablehlo.multiply %v50, %v51 : tensor<32x64x112x112xf32>
    %v54 = stablehlo.add %v53, %v52 : tensor<32x64x112x112xf32>
    %v55 = stablehlo.reshape %v54 : (tensor<32x64x112x112xf32>) -> tensor<32x802816xf32>
    %v56 = stablehlo.reshape %v55 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v57 = stablehlo.constant dense<0.0> : tensor<32x64x112x112xf32>
    %v58 = stablehlo.constant dense<6.0> : tensor<32x64x112x112xf32>
    %v59 = stablehlo.maximum %v56, %v57 : tensor<32x64x112x112xf32>
    %v60 = stablehlo.minimum %v59, %v58 : tensor<32x64x112x112xf32>
    %v61 = stablehlo.reshape %v60 : (tensor<32x64x112x112xf32>) -> tensor<32x802816xf32>
    %v62 = stablehlo.reshape %v61 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v63 = stablehlo.convolution(%v62, %Wd1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 64 : i64} : (tensor<32x64x112x112xf32>, tensor<64x1x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v64 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v65 = stablehlo.add %v63, %v64 : tensor<32x64x56x56xf32>
    %v66 = stablehlo.reshape %v65 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v67 = stablehlo.reshape %v66 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v68 = stablehlo.constant dense<0.0> : tensor<f32>
    %v69 = stablehlo.constant dense<3136.0> : tensor<32x64x56x56xf32>
    %v70 = stablehlo.constant dense<1.0e-5> : tensor<32x64x56x56xf32>
    %v71 = stablehlo.reduce(%v67 init: %v68) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v72 = stablehlo.broadcast_in_dim %v71, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v73 = stablehlo.divide %v72, %v69 : tensor<32x64x56x56xf32>
    %v74 = stablehlo.subtract %v67, %v73 : tensor<32x64x56x56xf32>
    %v75 = stablehlo.multiply %v74, %v74 : tensor<32x64x56x56xf32>
    %v76 = stablehlo.reduce(%v75 init: %v68) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v77 = stablehlo.broadcast_in_dim %v76, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v78 = stablehlo.divide %v77, %v69 : tensor<32x64x56x56xf32>
    %v79 = stablehlo.add %v78, %v70 : tensor<32x64x56x56xf32>
    %v80 = stablehlo.rsqrt %v79 : tensor<32x64x56x56xf32>
    %v81 = stablehlo.multiply %v74, %v80 : tensor<32x64x56x56xf32>
    %v82 = stablehlo.broadcast_in_dim %gd1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v83 = stablehlo.broadcast_in_dim %btd1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v84 = stablehlo.multiply %v81, %v82 : tensor<32x64x56x56xf32>
    %v85 = stablehlo.add %v84, %v83 : tensor<32x64x56x56xf32>
    %v86 = stablehlo.reshape %v85 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v87 = stablehlo.reshape %v86 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v88 = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %v89 = stablehlo.constant dense<6.0> : tensor<32x64x56x56xf32>
    %v90 = stablehlo.maximum %v87, %v88 : tensor<32x64x56x56xf32>
    %v91 = stablehlo.minimum %v90, %v89 : tensor<32x64x56x56xf32>
    %v92 = stablehlo.reshape %v91 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v93 = stablehlo.reshape %v92 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v94 = stablehlo.convolution(%v93, %Wp1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<24x64x1x1xf32>) -> tensor<32x24x56x56xf32>
    %v95 = stablehlo.broadcast_in_dim %zb24, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v96 = stablehlo.add %v94, %v95 : tensor<32x24x56x56xf32>
    %v97 = stablehlo.reshape %v96 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v98 = stablehlo.reshape %v97 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v99 = stablehlo.constant dense<0.0> : tensor<f32>
    %v100 = stablehlo.constant dense<3136.0> : tensor<32x24x56x56xf32>
    %v101 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v102 = stablehlo.reduce(%v98 init: %v99) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v103 = stablehlo.broadcast_in_dim %v102, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v104 = stablehlo.divide %v103, %v100 : tensor<32x24x56x56xf32>
    %v105 = stablehlo.subtract %v98, %v104 : tensor<32x24x56x56xf32>
    %v106 = stablehlo.multiply %v105, %v105 : tensor<32x24x56x56xf32>
    %v107 = stablehlo.reduce(%v106 init: %v99) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v108 = stablehlo.broadcast_in_dim %v107, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v109 = stablehlo.divide %v108, %v100 : tensor<32x24x56x56xf32>
    %v110 = stablehlo.add %v109, %v101 : tensor<32x24x56x56xf32>
    %v111 = stablehlo.rsqrt %v110 : tensor<32x24x56x56xf32>
    %v112 = stablehlo.multiply %v105, %v111 : tensor<32x24x56x56xf32>
    %v113 = stablehlo.broadcast_in_dim %gp1, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v114 = stablehlo.broadcast_in_dim %btp1, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v115 = stablehlo.multiply %v112, %v113 : tensor<32x24x56x56xf32>
    %v116 = stablehlo.add %v115, %v114 : tensor<32x24x56x56xf32>
    %v117 = stablehlo.reshape %v116 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v118 = stablehlo.reshape %v117 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v119 = stablehlo.convolution(%v118, %We2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<96x24x1x1xf32>) -> tensor<32x96x56x56xf32>
    %v120 = stablehlo.broadcast_in_dim %zb96, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v121 = stablehlo.add %v119, %v120 : tensor<32x96x56x56xf32>
    %v122 = stablehlo.reshape %v121 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v123 = stablehlo.reshape %v122 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v124 = stablehlo.constant dense<0.0> : tensor<f32>
    %v125 = stablehlo.constant dense<3136.0> : tensor<32x96x56x56xf32>
    %v126 = stablehlo.constant dense<1.0e-5> : tensor<32x96x56x56xf32>
    %v127 = stablehlo.reduce(%v123 init: %v124) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v128 = stablehlo.broadcast_in_dim %v127, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v129 = stablehlo.divide %v128, %v125 : tensor<32x96x56x56xf32>
    %v130 = stablehlo.subtract %v123, %v129 : tensor<32x96x56x56xf32>
    %v131 = stablehlo.multiply %v130, %v130 : tensor<32x96x56x56xf32>
    %v132 = stablehlo.reduce(%v131 init: %v124) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v133 = stablehlo.broadcast_in_dim %v132, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v134 = stablehlo.divide %v133, %v125 : tensor<32x96x56x56xf32>
    %v135 = stablehlo.add %v134, %v126 : tensor<32x96x56x56xf32>
    %v136 = stablehlo.rsqrt %v135 : tensor<32x96x56x56xf32>
    %v137 = stablehlo.multiply %v130, %v136 : tensor<32x96x56x56xf32>
    %v138 = stablehlo.broadcast_in_dim %ge2, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v139 = stablehlo.broadcast_in_dim %bte2, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v140 = stablehlo.multiply %v137, %v138 : tensor<32x96x56x56xf32>
    %v141 = stablehlo.add %v140, %v139 : tensor<32x96x56x56xf32>
    %v142 = stablehlo.reshape %v141 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v143 = stablehlo.reshape %v142 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v144 = stablehlo.constant dense<0.0> : tensor<32x96x56x56xf32>
    %v145 = stablehlo.constant dense<6.0> : tensor<32x96x56x56xf32>
    %v146 = stablehlo.maximum %v143, %v144 : tensor<32x96x56x56xf32>
    %v147 = stablehlo.minimum %v146, %v145 : tensor<32x96x56x56xf32>
    %v148 = stablehlo.reshape %v147 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v149 = stablehlo.reshape %v148 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v150 = stablehlo.convolution(%v149, %Wd2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x3x3xf32>) -> tensor<32x96x56x56xf32>
    %v151 = stablehlo.broadcast_in_dim %zb96, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v152 = stablehlo.add %v150, %v151 : tensor<32x96x56x56xf32>
    %v153 = stablehlo.reshape %v152 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v154 = stablehlo.reshape %v153 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v155 = stablehlo.constant dense<0.0> : tensor<f32>
    %v156 = stablehlo.constant dense<3136.0> : tensor<32x96x56x56xf32>
    %v157 = stablehlo.constant dense<1.0e-5> : tensor<32x96x56x56xf32>
    %v158 = stablehlo.reduce(%v154 init: %v155) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v159 = stablehlo.broadcast_in_dim %v158, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v160 = stablehlo.divide %v159, %v156 : tensor<32x96x56x56xf32>
    %v161 = stablehlo.subtract %v154, %v160 : tensor<32x96x56x56xf32>
    %v162 = stablehlo.multiply %v161, %v161 : tensor<32x96x56x56xf32>
    %v163 = stablehlo.reduce(%v162 init: %v155) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v164 = stablehlo.broadcast_in_dim %v163, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v165 = stablehlo.divide %v164, %v156 : tensor<32x96x56x56xf32>
    %v166 = stablehlo.add %v165, %v157 : tensor<32x96x56x56xf32>
    %v167 = stablehlo.rsqrt %v166 : tensor<32x96x56x56xf32>
    %v168 = stablehlo.multiply %v161, %v167 : tensor<32x96x56x56xf32>
    %v169 = stablehlo.broadcast_in_dim %gd2, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v170 = stablehlo.broadcast_in_dim %btd2, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v171 = stablehlo.multiply %v168, %v169 : tensor<32x96x56x56xf32>
    %v172 = stablehlo.add %v171, %v170 : tensor<32x96x56x56xf32>
    %v173 = stablehlo.reshape %v172 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v174 = stablehlo.reshape %v173 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v175 = stablehlo.constant dense<0.0> : tensor<32x96x56x56xf32>
    %v176 = stablehlo.constant dense<6.0> : tensor<32x96x56x56xf32>
    %v177 = stablehlo.maximum %v174, %v175 : tensor<32x96x56x56xf32>
    %v178 = stablehlo.minimum %v177, %v176 : tensor<32x96x56x56xf32>
    %v179 = stablehlo.reshape %v178 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v180 = stablehlo.reshape %v179 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v181 = stablehlo.convolution(%v180, %Wp2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<24x96x1x1xf32>) -> tensor<32x24x56x56xf32>
    %v182 = stablehlo.broadcast_in_dim %zb24, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v183 = stablehlo.add %v181, %v182 : tensor<32x24x56x56xf32>
    %v184 = stablehlo.reshape %v183 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v185 = stablehlo.reshape %v184 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v186 = stablehlo.constant dense<0.0> : tensor<f32>
    %v187 = stablehlo.constant dense<3136.0> : tensor<32x24x56x56xf32>
    %v188 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v189 = stablehlo.reduce(%v185 init: %v186) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v190 = stablehlo.broadcast_in_dim %v189, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v191 = stablehlo.divide %v190, %v187 : tensor<32x24x56x56xf32>
    %v192 = stablehlo.subtract %v185, %v191 : tensor<32x24x56x56xf32>
    %v193 = stablehlo.multiply %v192, %v192 : tensor<32x24x56x56xf32>
    %v194 = stablehlo.reduce(%v193 init: %v186) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v195 = stablehlo.broadcast_in_dim %v194, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v196 = stablehlo.divide %v195, %v187 : tensor<32x24x56x56xf32>
    %v197 = stablehlo.add %v196, %v188 : tensor<32x24x56x56xf32>
    %v198 = stablehlo.rsqrt %v197 : tensor<32x24x56x56xf32>
    %v199 = stablehlo.multiply %v192, %v198 : tensor<32x24x56x56xf32>
    %v200 = stablehlo.broadcast_in_dim %gp2, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v201 = stablehlo.broadcast_in_dim %btp2, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v202 = stablehlo.multiply %v199, %v200 : tensor<32x24x56x56xf32>
    %v203 = stablehlo.add %v202, %v201 : tensor<32x24x56x56xf32>
    %v204 = stablehlo.reshape %v203 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v205 = stablehlo.reshape %v204 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v206 = stablehlo.reshape %v117 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v207 = stablehlo.add %v205, %v206 : tensor<32x24x56x56xf32>
    %v208 = stablehlo.reshape %v207 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v209 = stablehlo.reshape %v208 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v210 = stablehlo.convolution(%v209, %We3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<96x24x1x1xf32>) -> tensor<32x96x56x56xf32>
    %v211 = stablehlo.broadcast_in_dim %zb96, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v212 = stablehlo.add %v210, %v211 : tensor<32x96x56x56xf32>
    %v213 = stablehlo.reshape %v212 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v214 = stablehlo.reshape %v213 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v215 = stablehlo.constant dense<0.0> : tensor<f32>
    %v216 = stablehlo.constant dense<3136.0> : tensor<32x96x56x56xf32>
    %v217 = stablehlo.constant dense<1.0e-5> : tensor<32x96x56x56xf32>
    %v218 = stablehlo.reduce(%v214 init: %v215) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v219 = stablehlo.broadcast_in_dim %v218, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v220 = stablehlo.divide %v219, %v216 : tensor<32x96x56x56xf32>
    %v221 = stablehlo.subtract %v214, %v220 : tensor<32x96x56x56xf32>
    %v222 = stablehlo.multiply %v221, %v221 : tensor<32x96x56x56xf32>
    %v223 = stablehlo.reduce(%v222 init: %v215) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v224 = stablehlo.broadcast_in_dim %v223, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v225 = stablehlo.divide %v224, %v216 : tensor<32x96x56x56xf32>
    %v226 = stablehlo.add %v225, %v217 : tensor<32x96x56x56xf32>
    %v227 = stablehlo.rsqrt %v226 : tensor<32x96x56x56xf32>
    %v228 = stablehlo.multiply %v221, %v227 : tensor<32x96x56x56xf32>
    %v229 = stablehlo.broadcast_in_dim %ge3, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v230 = stablehlo.broadcast_in_dim %bte3, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v231 = stablehlo.multiply %v228, %v229 : tensor<32x96x56x56xf32>
    %v232 = stablehlo.add %v231, %v230 : tensor<32x96x56x56xf32>
    %v233 = stablehlo.reshape %v232 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v234 = stablehlo.reshape %v233 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v235 = stablehlo.constant dense<0.0> : tensor<32x96x56x56xf32>
    %v236 = stablehlo.constant dense<6.0> : tensor<32x96x56x56xf32>
    %v237 = stablehlo.maximum %v234, %v235 : tensor<32x96x56x56xf32>
    %v238 = stablehlo.minimum %v237, %v236 : tensor<32x96x56x56xf32>
    %v239 = stablehlo.reshape %v238 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v240 = stablehlo.reshape %v239 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v241 = stablehlo.convolution(%v240, %Wd3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x3x3xf32>) -> tensor<32x96x28x28xf32>
    %v242 = stablehlo.broadcast_in_dim %zb96, dims = [1] : (tensor<96xf32>) -> tensor<32x96x28x28xf32>
    %v243 = stablehlo.add %v241, %v242 : tensor<32x96x28x28xf32>
    %v244 = stablehlo.reshape %v243 : (tensor<32x96x28x28xf32>) -> tensor<32x75264xf32>
    %v245 = stablehlo.reshape %v244 : (tensor<32x75264xf32>) -> tensor<32x96x28x28xf32>
    %v246 = stablehlo.constant dense<0.0> : tensor<f32>
    %v247 = stablehlo.constant dense<784.0> : tensor<32x96x28x28xf32>
    %v248 = stablehlo.constant dense<1.0e-5> : tensor<32x96x28x28xf32>
    %v249 = stablehlo.reduce(%v245 init: %v246) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x28x28xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v250 = stablehlo.broadcast_in_dim %v249, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x28x28xf32>
    %v251 = stablehlo.divide %v250, %v247 : tensor<32x96x28x28xf32>
    %v252 = stablehlo.subtract %v245, %v251 : tensor<32x96x28x28xf32>
    %v253 = stablehlo.multiply %v252, %v252 : tensor<32x96x28x28xf32>
    %v254 = stablehlo.reduce(%v253 init: %v246) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x28x28xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v255 = stablehlo.broadcast_in_dim %v254, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x28x28xf32>
    %v256 = stablehlo.divide %v255, %v247 : tensor<32x96x28x28xf32>
    %v257 = stablehlo.add %v256, %v248 : tensor<32x96x28x28xf32>
    %v258 = stablehlo.rsqrt %v257 : tensor<32x96x28x28xf32>
    %v259 = stablehlo.multiply %v252, %v258 : tensor<32x96x28x28xf32>
    %v260 = stablehlo.broadcast_in_dim %gd3, dims = [1] : (tensor<96xf32>) -> tensor<32x96x28x28xf32>
    %v261 = stablehlo.broadcast_in_dim %btd3, dims = [1] : (tensor<96xf32>) -> tensor<32x96x28x28xf32>
    %v262 = stablehlo.multiply %v259, %v260 : tensor<32x96x28x28xf32>
    %v263 = stablehlo.add %v262, %v261 : tensor<32x96x28x28xf32>
    %v264 = stablehlo.reshape %v263 : (tensor<32x96x28x28xf32>) -> tensor<32x75264xf32>
    %v265 = stablehlo.reshape %v264 : (tensor<32x75264xf32>) -> tensor<32x96x28x28xf32>
    %v266 = stablehlo.constant dense<0.0> : tensor<32x96x28x28xf32>
    %v267 = stablehlo.constant dense<6.0> : tensor<32x96x28x28xf32>
    %v268 = stablehlo.maximum %v265, %v266 : tensor<32x96x28x28xf32>
    %v269 = stablehlo.minimum %v268, %v267 : tensor<32x96x28x28xf32>
    %v270 = stablehlo.reshape %v269 : (tensor<32x96x28x28xf32>) -> tensor<32x75264xf32>
    %v271 = stablehlo.reshape %v270 : (tensor<32x75264xf32>) -> tensor<32x96x28x28xf32>
    %v272 = stablehlo.convolution(%v271, %Wp3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x28x28xf32>, tensor<32x96x1x1xf32>) -> tensor<32x32x28x28xf32>
    %v273 = stablehlo.broadcast_in_dim %zb32, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v274 = stablehlo.add %v272, %v273 : tensor<32x32x28x28xf32>
    %v275 = stablehlo.reshape %v274 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v276 = stablehlo.reshape %v275 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v277 = stablehlo.constant dense<0.0> : tensor<f32>
    %v278 = stablehlo.constant dense<784.0> : tensor<32x32x28x28xf32>
    %v279 = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %v280 = stablehlo.reduce(%v276 init: %v277) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v281 = stablehlo.broadcast_in_dim %v280, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v282 = stablehlo.divide %v281, %v278 : tensor<32x32x28x28xf32>
    %v283 = stablehlo.subtract %v276, %v282 : tensor<32x32x28x28xf32>
    %v284 = stablehlo.multiply %v283, %v283 : tensor<32x32x28x28xf32>
    %v285 = stablehlo.reduce(%v284 init: %v277) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v286 = stablehlo.broadcast_in_dim %v285, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v287 = stablehlo.divide %v286, %v278 : tensor<32x32x28x28xf32>
    %v288 = stablehlo.add %v287, %v279 : tensor<32x32x28x28xf32>
    %v289 = stablehlo.rsqrt %v288 : tensor<32x32x28x28xf32>
    %v290 = stablehlo.multiply %v283, %v289 : tensor<32x32x28x28xf32>
    %v291 = stablehlo.broadcast_in_dim %gp3, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v292 = stablehlo.broadcast_in_dim %btp3, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v293 = stablehlo.multiply %v290, %v291 : tensor<32x32x28x28xf32>
    %v294 = stablehlo.add %v293, %v292 : tensor<32x32x28x28xf32>
    %v295 = stablehlo.reshape %v294 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v296 = stablehlo.reshape %v295 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v297 = stablehlo.convolution(%v296, %We4)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<128x32x1x1xf32>) -> tensor<32x128x28x28xf32>
    %v298 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v299 = stablehlo.add %v297, %v298 : tensor<32x128x28x28xf32>
    %v300 = stablehlo.reshape %v299 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v301 = stablehlo.reshape %v300 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v302 = stablehlo.constant dense<0.0> : tensor<f32>
    %v303 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v304 = stablehlo.constant dense<1.0e-5> : tensor<32x128x28x28xf32>
    %v305 = stablehlo.reduce(%v301 init: %v302) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v306 = stablehlo.broadcast_in_dim %v305, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v307 = stablehlo.divide %v306, %v303 : tensor<32x128x28x28xf32>
    %v308 = stablehlo.subtract %v301, %v307 : tensor<32x128x28x28xf32>
    %v309 = stablehlo.multiply %v308, %v308 : tensor<32x128x28x28xf32>
    %v310 = stablehlo.reduce(%v309 init: %v302) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v311 = stablehlo.broadcast_in_dim %v310, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v312 = stablehlo.divide %v311, %v303 : tensor<32x128x28x28xf32>
    %v313 = stablehlo.add %v312, %v304 : tensor<32x128x28x28xf32>
    %v314 = stablehlo.rsqrt %v313 : tensor<32x128x28x28xf32>
    %v315 = stablehlo.multiply %v308, %v314 : tensor<32x128x28x28xf32>
    %v316 = stablehlo.broadcast_in_dim %ge4, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v317 = stablehlo.broadcast_in_dim %bte4, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v318 = stablehlo.multiply %v315, %v316 : tensor<32x128x28x28xf32>
    %v319 = stablehlo.add %v318, %v317 : tensor<32x128x28x28xf32>
    %v320 = stablehlo.reshape %v319 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v321 = stablehlo.reshape %v320 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v322 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v323 = stablehlo.constant dense<6.0> : tensor<32x128x28x28xf32>
    %v324 = stablehlo.maximum %v321, %v322 : tensor<32x128x28x28xf32>
    %v325 = stablehlo.minimum %v324, %v323 : tensor<32x128x28x28xf32>
    %v326 = stablehlo.reshape %v325 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v327 = stablehlo.reshape %v326 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v328 = stablehlo.convolution(%v327, %Wd4)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 128 : i64} : (tensor<32x128x28x28xf32>, tensor<128x1x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v329 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v330 = stablehlo.add %v328, %v329 : tensor<32x128x28x28xf32>
    %v331 = stablehlo.reshape %v330 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v332 = stablehlo.reshape %v331 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v333 = stablehlo.constant dense<0.0> : tensor<f32>
    %v334 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v335 = stablehlo.constant dense<1.0e-5> : tensor<32x128x28x28xf32>
    %v336 = stablehlo.reduce(%v332 init: %v333) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v337 = stablehlo.broadcast_in_dim %v336, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v338 = stablehlo.divide %v337, %v334 : tensor<32x128x28x28xf32>
    %v339 = stablehlo.subtract %v332, %v338 : tensor<32x128x28x28xf32>
    %v340 = stablehlo.multiply %v339, %v339 : tensor<32x128x28x28xf32>
    %v341 = stablehlo.reduce(%v340 init: %v333) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v342 = stablehlo.broadcast_in_dim %v341, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v343 = stablehlo.divide %v342, %v334 : tensor<32x128x28x28xf32>
    %v344 = stablehlo.add %v343, %v335 : tensor<32x128x28x28xf32>
    %v345 = stablehlo.rsqrt %v344 : tensor<32x128x28x28xf32>
    %v346 = stablehlo.multiply %v339, %v345 : tensor<32x128x28x28xf32>
    %v347 = stablehlo.broadcast_in_dim %gd4, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v348 = stablehlo.broadcast_in_dim %btd4, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v349 = stablehlo.multiply %v346, %v347 : tensor<32x128x28x28xf32>
    %v350 = stablehlo.add %v349, %v348 : tensor<32x128x28x28xf32>
    %v351 = stablehlo.reshape %v350 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v352 = stablehlo.reshape %v351 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v353 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v354 = stablehlo.constant dense<6.0> : tensor<32x128x28x28xf32>
    %v355 = stablehlo.maximum %v352, %v353 : tensor<32x128x28x28xf32>
    %v356 = stablehlo.minimum %v355, %v354 : tensor<32x128x28x28xf32>
    %v357 = stablehlo.reshape %v356 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v358 = stablehlo.reshape %v357 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v359 = stablehlo.convolution(%v358, %Wp4)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<32x128x1x1xf32>) -> tensor<32x32x28x28xf32>
    %v360 = stablehlo.broadcast_in_dim %zb32, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v361 = stablehlo.add %v359, %v360 : tensor<32x32x28x28xf32>
    %v362 = stablehlo.reshape %v361 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v363 = stablehlo.reshape %v362 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v364 = stablehlo.constant dense<0.0> : tensor<f32>
    %v365 = stablehlo.constant dense<784.0> : tensor<32x32x28x28xf32>
    %v366 = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %v367 = stablehlo.reduce(%v363 init: %v364) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v368 = stablehlo.broadcast_in_dim %v367, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v369 = stablehlo.divide %v368, %v365 : tensor<32x32x28x28xf32>
    %v370 = stablehlo.subtract %v363, %v369 : tensor<32x32x28x28xf32>
    %v371 = stablehlo.multiply %v370, %v370 : tensor<32x32x28x28xf32>
    %v372 = stablehlo.reduce(%v371 init: %v364) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v373 = stablehlo.broadcast_in_dim %v372, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v374 = stablehlo.divide %v373, %v365 : tensor<32x32x28x28xf32>
    %v375 = stablehlo.add %v374, %v366 : tensor<32x32x28x28xf32>
    %v376 = stablehlo.rsqrt %v375 : tensor<32x32x28x28xf32>
    %v377 = stablehlo.multiply %v370, %v376 : tensor<32x32x28x28xf32>
    %v378 = stablehlo.broadcast_in_dim %gp4, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v379 = stablehlo.broadcast_in_dim %btp4, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v380 = stablehlo.multiply %v377, %v378 : tensor<32x32x28x28xf32>
    %v381 = stablehlo.add %v380, %v379 : tensor<32x32x28x28xf32>
    %v382 = stablehlo.reshape %v381 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v383 = stablehlo.reshape %v382 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v384 = stablehlo.reshape %v295 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v385 = stablehlo.add %v383, %v384 : tensor<32x32x28x28xf32>
    %v386 = stablehlo.reshape %v385 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v387 = stablehlo.reshape %v386 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v388 = stablehlo.convolution(%v387, %We5)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<128x32x1x1xf32>) -> tensor<32x128x28x28xf32>
    %v389 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v390 = stablehlo.add %v388, %v389 : tensor<32x128x28x28xf32>
    %v391 = stablehlo.reshape %v390 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v392 = stablehlo.reshape %v391 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v393 = stablehlo.constant dense<0.0> : tensor<f32>
    %v394 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v395 = stablehlo.constant dense<1.0e-5> : tensor<32x128x28x28xf32>
    %v396 = stablehlo.reduce(%v392 init: %v393) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v397 = stablehlo.broadcast_in_dim %v396, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v398 = stablehlo.divide %v397, %v394 : tensor<32x128x28x28xf32>
    %v399 = stablehlo.subtract %v392, %v398 : tensor<32x128x28x28xf32>
    %v400 = stablehlo.multiply %v399, %v399 : tensor<32x128x28x28xf32>
    %v401 = stablehlo.reduce(%v400 init: %v393) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v402 = stablehlo.broadcast_in_dim %v401, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v403 = stablehlo.divide %v402, %v394 : tensor<32x128x28x28xf32>
    %v404 = stablehlo.add %v403, %v395 : tensor<32x128x28x28xf32>
    %v405 = stablehlo.rsqrt %v404 : tensor<32x128x28x28xf32>
    %v406 = stablehlo.multiply %v399, %v405 : tensor<32x128x28x28xf32>
    %v407 = stablehlo.broadcast_in_dim %ge5, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v408 = stablehlo.broadcast_in_dim %bte5, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v409 = stablehlo.multiply %v406, %v407 : tensor<32x128x28x28xf32>
    %v410 = stablehlo.add %v409, %v408 : tensor<32x128x28x28xf32>
    %v411 = stablehlo.reshape %v410 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v412 = stablehlo.reshape %v411 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v413 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v414 = stablehlo.constant dense<6.0> : tensor<32x128x28x28xf32>
    %v415 = stablehlo.maximum %v412, %v413 : tensor<32x128x28x28xf32>
    %v416 = stablehlo.minimum %v415, %v414 : tensor<32x128x28x28xf32>
    %v417 = stablehlo.reshape %v416 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v418 = stablehlo.reshape %v417 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v419 = stablehlo.convolution(%v418, %Wd5)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 128 : i64} : (tensor<32x128x28x28xf32>, tensor<128x1x3x3xf32>) -> tensor<32x128x14x14xf32>
    %v420 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x14x14xf32>
    %v421 = stablehlo.add %v419, %v420 : tensor<32x128x14x14xf32>
    %v422 = stablehlo.reshape %v421 : (tensor<32x128x14x14xf32>) -> tensor<32x25088xf32>
    %v423 = stablehlo.reshape %v422 : (tensor<32x25088xf32>) -> tensor<32x128x14x14xf32>
    %v424 = stablehlo.constant dense<0.0> : tensor<f32>
    %v425 = stablehlo.constant dense<196.0> : tensor<32x128x14x14xf32>
    %v426 = stablehlo.constant dense<1.0e-5> : tensor<32x128x14x14xf32>
    %v427 = stablehlo.reduce(%v423 init: %v424) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x14x14xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v428 = stablehlo.broadcast_in_dim %v427, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x14x14xf32>
    %v429 = stablehlo.divide %v428, %v425 : tensor<32x128x14x14xf32>
    %v430 = stablehlo.subtract %v423, %v429 : tensor<32x128x14x14xf32>
    %v431 = stablehlo.multiply %v430, %v430 : tensor<32x128x14x14xf32>
    %v432 = stablehlo.reduce(%v431 init: %v424) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x14x14xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v433 = stablehlo.broadcast_in_dim %v432, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x14x14xf32>
    %v434 = stablehlo.divide %v433, %v425 : tensor<32x128x14x14xf32>
    %v435 = stablehlo.add %v434, %v426 : tensor<32x128x14x14xf32>
    %v436 = stablehlo.rsqrt %v435 : tensor<32x128x14x14xf32>
    %v437 = stablehlo.multiply %v430, %v436 : tensor<32x128x14x14xf32>
    %v438 = stablehlo.broadcast_in_dim %gd5, dims = [1] : (tensor<128xf32>) -> tensor<32x128x14x14xf32>
    %v439 = stablehlo.broadcast_in_dim %btd5, dims = [1] : (tensor<128xf32>) -> tensor<32x128x14x14xf32>
    %v440 = stablehlo.multiply %v437, %v438 : tensor<32x128x14x14xf32>
    %v441 = stablehlo.add %v440, %v439 : tensor<32x128x14x14xf32>
    %v442 = stablehlo.reshape %v441 : (tensor<32x128x14x14xf32>) -> tensor<32x25088xf32>
    %v443 = stablehlo.reshape %v442 : (tensor<32x25088xf32>) -> tensor<32x128x14x14xf32>
    %v444 = stablehlo.constant dense<0.0> : tensor<32x128x14x14xf32>
    %v445 = stablehlo.constant dense<6.0> : tensor<32x128x14x14xf32>
    %v446 = stablehlo.maximum %v443, %v444 : tensor<32x128x14x14xf32>
    %v447 = stablehlo.minimum %v446, %v445 : tensor<32x128x14x14xf32>
    %v448 = stablehlo.reshape %v447 : (tensor<32x128x14x14xf32>) -> tensor<32x25088xf32>
    %v449 = stablehlo.reshape %v448 : (tensor<32x25088xf32>) -> tensor<32x128x14x14xf32>
    %v450 = stablehlo.convolution(%v449, %Wp5)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x14x14xf32>, tensor<64x128x1x1xf32>) -> tensor<32x64x14x14xf32>
    %v451 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v452 = stablehlo.add %v450, %v451 : tensor<32x64x14x14xf32>
    %v453 = stablehlo.reshape %v452 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v454 = stablehlo.reshape %v453 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v455 = stablehlo.constant dense<0.0> : tensor<f32>
    %v456 = stablehlo.constant dense<196.0> : tensor<32x64x14x14xf32>
    %v457 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v458 = stablehlo.reduce(%v454 init: %v455) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v459 = stablehlo.broadcast_in_dim %v458, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v460 = stablehlo.divide %v459, %v456 : tensor<32x64x14x14xf32>
    %v461 = stablehlo.subtract %v454, %v460 : tensor<32x64x14x14xf32>
    %v462 = stablehlo.multiply %v461, %v461 : tensor<32x64x14x14xf32>
    %v463 = stablehlo.reduce(%v462 init: %v455) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v464 = stablehlo.broadcast_in_dim %v463, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v465 = stablehlo.divide %v464, %v456 : tensor<32x64x14x14xf32>
    %v466 = stablehlo.add %v465, %v457 : tensor<32x64x14x14xf32>
    %v467 = stablehlo.rsqrt %v466 : tensor<32x64x14x14xf32>
    %v468 = stablehlo.multiply %v461, %v467 : tensor<32x64x14x14xf32>
    %v469 = stablehlo.broadcast_in_dim %gp5, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v470 = stablehlo.broadcast_in_dim %btp5, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v471 = stablehlo.multiply %v468, %v469 : tensor<32x64x14x14xf32>
    %v472 = stablehlo.add %v471, %v470 : tensor<32x64x14x14xf32>
    %v473 = stablehlo.reshape %v472 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v474 = stablehlo.reshape %v473 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v475 = stablehlo.convolution(%v474, %We6)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x14x14xf32>, tensor<256x64x1x1xf32>) -> tensor<32x256x14x14xf32>
    %v476 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v477 = stablehlo.add %v475, %v476 : tensor<32x256x14x14xf32>
    %v478 = stablehlo.reshape %v477 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v479 = stablehlo.reshape %v478 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v480 = stablehlo.constant dense<0.0> : tensor<f32>
    %v481 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v482 = stablehlo.constant dense<1.0e-5> : tensor<32x256x14x14xf32>
    %v483 = stablehlo.reduce(%v479 init: %v480) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v484 = stablehlo.broadcast_in_dim %v483, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v485 = stablehlo.divide %v484, %v481 : tensor<32x256x14x14xf32>
    %v486 = stablehlo.subtract %v479, %v485 : tensor<32x256x14x14xf32>
    %v487 = stablehlo.multiply %v486, %v486 : tensor<32x256x14x14xf32>
    %v488 = stablehlo.reduce(%v487 init: %v480) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v489 = stablehlo.broadcast_in_dim %v488, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v490 = stablehlo.divide %v489, %v481 : tensor<32x256x14x14xf32>
    %v491 = stablehlo.add %v490, %v482 : tensor<32x256x14x14xf32>
    %v492 = stablehlo.rsqrt %v491 : tensor<32x256x14x14xf32>
    %v493 = stablehlo.multiply %v486, %v492 : tensor<32x256x14x14xf32>
    %v494 = stablehlo.broadcast_in_dim %ge6, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v495 = stablehlo.broadcast_in_dim %bte6, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v496 = stablehlo.multiply %v493, %v494 : tensor<32x256x14x14xf32>
    %v497 = stablehlo.add %v496, %v495 : tensor<32x256x14x14xf32>
    %v498 = stablehlo.reshape %v497 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v499 = stablehlo.reshape %v498 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v500 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v501 = stablehlo.constant dense<6.0> : tensor<32x256x14x14xf32>
    %v502 = stablehlo.maximum %v499, %v500 : tensor<32x256x14x14xf32>
    %v503 = stablehlo.minimum %v502, %v501 : tensor<32x256x14x14xf32>
    %v504 = stablehlo.reshape %v503 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v505 = stablehlo.reshape %v504 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v506 = stablehlo.convolution(%v505, %Wd6)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 256 : i64} : (tensor<32x256x14x14xf32>, tensor<256x1x3x3xf32>) -> tensor<32x256x7x7xf32>
    %v507 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v508 = stablehlo.add %v506, %v507 : tensor<32x256x7x7xf32>
    %v509 = stablehlo.reshape %v508 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v510 = stablehlo.reshape %v509 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v511 = stablehlo.constant dense<0.0> : tensor<f32>
    %v512 = stablehlo.constant dense<49.0> : tensor<32x256x7x7xf32>
    %v513 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v514 = stablehlo.reduce(%v510 init: %v511) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v515 = stablehlo.broadcast_in_dim %v514, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x7x7xf32>
    %v516 = stablehlo.divide %v515, %v512 : tensor<32x256x7x7xf32>
    %v517 = stablehlo.subtract %v510, %v516 : tensor<32x256x7x7xf32>
    %v518 = stablehlo.multiply %v517, %v517 : tensor<32x256x7x7xf32>
    %v519 = stablehlo.reduce(%v518 init: %v511) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v520 = stablehlo.broadcast_in_dim %v519, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x7x7xf32>
    %v521 = stablehlo.divide %v520, %v512 : tensor<32x256x7x7xf32>
    %v522 = stablehlo.add %v521, %v513 : tensor<32x256x7x7xf32>
    %v523 = stablehlo.rsqrt %v522 : tensor<32x256x7x7xf32>
    %v524 = stablehlo.multiply %v517, %v523 : tensor<32x256x7x7xf32>
    %v525 = stablehlo.broadcast_in_dim %gd6, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v526 = stablehlo.broadcast_in_dim %btd6, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v527 = stablehlo.multiply %v524, %v525 : tensor<32x256x7x7xf32>
    %v528 = stablehlo.add %v527, %v526 : tensor<32x256x7x7xf32>
    %v529 = stablehlo.reshape %v528 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v530 = stablehlo.reshape %v529 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v531 = stablehlo.constant dense<0.0> : tensor<32x256x7x7xf32>
    %v532 = stablehlo.constant dense<6.0> : tensor<32x256x7x7xf32>
    %v533 = stablehlo.maximum %v530, %v531 : tensor<32x256x7x7xf32>
    %v534 = stablehlo.minimum %v533, %v532 : tensor<32x256x7x7xf32>
    %v535 = stablehlo.reshape %v534 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v536 = stablehlo.reshape %v535 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v537 = stablehlo.convolution(%v536, %Wp6)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x7x7xf32>, tensor<64x256x1x1xf32>) -> tensor<32x64x7x7xf32>
    %v538 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x7x7xf32>
    %v539 = stablehlo.add %v537, %v538 : tensor<32x64x7x7xf32>
    %v540 = stablehlo.reshape %v539 : (tensor<32x64x7x7xf32>) -> tensor<32x3136xf32>
    %v541 = stablehlo.reshape %v540 : (tensor<32x3136xf32>) -> tensor<32x64x7x7xf32>
    %v542 = stablehlo.constant dense<0.0> : tensor<f32>
    %v543 = stablehlo.constant dense<49.0> : tensor<32x64x7x7xf32>
    %v544 = stablehlo.constant dense<1.0e-5> : tensor<32x64x7x7xf32>
    %v545 = stablehlo.reduce(%v541 init: %v542) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x7x7xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v546 = stablehlo.broadcast_in_dim %v545, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x7x7xf32>
    %v547 = stablehlo.divide %v546, %v543 : tensor<32x64x7x7xf32>
    %v548 = stablehlo.subtract %v541, %v547 : tensor<32x64x7x7xf32>
    %v549 = stablehlo.multiply %v548, %v548 : tensor<32x64x7x7xf32>
    %v550 = stablehlo.reduce(%v549 init: %v542) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x7x7xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v551 = stablehlo.broadcast_in_dim %v550, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x7x7xf32>
    %v552 = stablehlo.divide %v551, %v543 : tensor<32x64x7x7xf32>
    %v553 = stablehlo.add %v552, %v544 : tensor<32x64x7x7xf32>
    %v554 = stablehlo.rsqrt %v553 : tensor<32x64x7x7xf32>
    %v555 = stablehlo.multiply %v548, %v554 : tensor<32x64x7x7xf32>
    %v556 = stablehlo.broadcast_in_dim %gp6, dims = [1] : (tensor<64xf32>) -> tensor<32x64x7x7xf32>
    %v557 = stablehlo.broadcast_in_dim %btp6, dims = [1] : (tensor<64xf32>) -> tensor<32x64x7x7xf32>
    %v558 = stablehlo.multiply %v555, %v556 : tensor<32x64x7x7xf32>
    %v559 = stablehlo.add %v558, %v557 : tensor<32x64x7x7xf32>
    %v560 = stablehlo.reshape %v559 : (tensor<32x64x7x7xf32>) -> tensor<32x3136xf32>
    %v561 = stablehlo.reshape %v560 : (tensor<32x3136xf32>) -> tensor<32x64x7x7xf32>
    %v562 = stablehlo.convolution(%v561, %Wh)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x7x7xf32>, tensor<128x64x1x1xf32>) -> tensor<32x128x7x7xf32>
    %v563 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x7x7xf32>
    %v564 = stablehlo.add %v562, %v563 : tensor<32x128x7x7xf32>
    %v565 = stablehlo.reshape %v564 : (tensor<32x128x7x7xf32>) -> tensor<32x6272xf32>
    %v566 = stablehlo.reshape %v565 : (tensor<32x6272xf32>) -> tensor<32x128x7x7xf32>
    %v567 = stablehlo.constant dense<0.0> : tensor<f32>
    %v568 = stablehlo.constant dense<49.0> : tensor<32x128x7x7xf32>
    %v569 = stablehlo.constant dense<1.0e-5> : tensor<32x128x7x7xf32>
    %v570 = stablehlo.reduce(%v566 init: %v567) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x7x7xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v571 = stablehlo.broadcast_in_dim %v570, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x7x7xf32>
    %v572 = stablehlo.divide %v571, %v568 : tensor<32x128x7x7xf32>
    %v573 = stablehlo.subtract %v566, %v572 : tensor<32x128x7x7xf32>
    %v574 = stablehlo.multiply %v573, %v573 : tensor<32x128x7x7xf32>
    %v575 = stablehlo.reduce(%v574 init: %v567) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x7x7xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v576 = stablehlo.broadcast_in_dim %v575, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x7x7xf32>
    %v577 = stablehlo.divide %v576, %v568 : tensor<32x128x7x7xf32>
    %v578 = stablehlo.add %v577, %v569 : tensor<32x128x7x7xf32>
    %v579 = stablehlo.rsqrt %v578 : tensor<32x128x7x7xf32>
    %v580 = stablehlo.multiply %v573, %v579 : tensor<32x128x7x7xf32>
    %v581 = stablehlo.broadcast_in_dim %gh, dims = [1] : (tensor<128xf32>) -> tensor<32x128x7x7xf32>
    %v582 = stablehlo.broadcast_in_dim %bth, dims = [1] : (tensor<128xf32>) -> tensor<32x128x7x7xf32>
    %v583 = stablehlo.multiply %v580, %v581 : tensor<32x128x7x7xf32>
    %v584 = stablehlo.add %v583, %v582 : tensor<32x128x7x7xf32>
    %v585 = stablehlo.reshape %v584 : (tensor<32x128x7x7xf32>) -> tensor<32x6272xf32>
    %v586 = stablehlo.reshape %v585 : (tensor<32x6272xf32>) -> tensor<32x128x7x7xf32>
    %v587 = stablehlo.constant dense<0.0> : tensor<32x128x7x7xf32>
    %v588 = stablehlo.constant dense<6.0> : tensor<32x128x7x7xf32>
    %v589 = stablehlo.maximum %v586, %v587 : tensor<32x128x7x7xf32>
    %v590 = stablehlo.minimum %v589, %v588 : tensor<32x128x7x7xf32>
    %v591 = stablehlo.reshape %v590 : (tensor<32x128x7x7xf32>) -> tensor<32x6272xf32>
    %v592 = stablehlo.reshape %v591 : (tensor<32x6272xf32>) -> tensor<32x128x7x7xf32>
    %v593 = stablehlo.constant dense<0.0> : tensor<f32>
    %v594 = stablehlo.reduce(%v592 init: %v593) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x7x7xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v595 = stablehlo.constant dense<49.0> : tensor<32x128xf32>
    %v596 = stablehlo.divide %v594, %v595 : tensor<32x128xf32>
    %v597 = stablehlo.dot_general %v596, %Wfc, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x128xf32>, tensor<128x10xf32>) -> tensor<32x10xf32>
    %v598 = stablehlo.broadcast_in_dim %bfc, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v599 = stablehlo.add %v597, %v598 : tensor<32x10xf32>
    %v600 = stablehlo.exponential %v599 : tensor<32x10xf32>
    %v601 = stablehlo.constant dense<0.0> : tensor<f32>
    %v602 = stablehlo.reduce(%v600 init: %v601) applies stablehlo.add across dimensions = [1] : (tensor<32x10xf32>, tensor<f32>) -> tensor<32xf32>
    %v603 = stablehlo.broadcast_in_dim %v602, dims = [0] : (tensor<32xf32>) -> tensor<32x10xf32>
    %v604 = stablehlo.divide %v600, %v603 : tensor<32x10xf32>
    %v605 = stablehlo.subtract %v604, %onehot : tensor<32x10xf32>
    %v606 = stablehlo.dot_general %v605, %Wfc, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<128x10xf32>) -> tensor<32x128xf32>
    %v607 = stablehlo.constant dense<49.0> : tensor<32x128xf32>
    %v608 = stablehlo.divide %v606, %v607 : tensor<32x128xf32>
    %v609 = stablehlo.broadcast_in_dim %v608, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x7x7xf32>
    %v610 = stablehlo.reshape %v609 : (tensor<32x128x7x7xf32>) -> tensor<32x6272xf32>
    %v611 = stablehlo.dot_general %v596, %v605, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x128xf32>, tensor<32x10xf32>) -> tensor<128x10xf32>
    %v612 = stablehlo.constant dense<0.3> : tensor<128x10xf32>
    %v613 = stablehlo.multiply %v611, %v612 : tensor<128x10xf32>
    %v614 = stablehlo.subtract %Wfc, %v613 : tensor<128x10xf32>
    %v615 = stablehlo.constant dense<0.0> : tensor<f32>
    %v616 = stablehlo.reduce(%v605 init: %v615) applies stablehlo.add across dimensions = [0] : (tensor<32x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v617 = stablehlo.constant dense<0.3> : tensor<10xf32>
    %v618 = stablehlo.multiply %v616, %v617 : tensor<10xf32>
    %v619 = stablehlo.subtract %bfc, %v618 : tensor<10xf32>
    %v620 = stablehlo.reshape %v610 : (tensor<32x6272xf32>) -> tensor<32x128x7x7xf32>
    %v621 = stablehlo.reshape %v585 : (tensor<32x6272xf32>) -> tensor<32x128x7x7xf32>
    %v622 = stablehlo.constant dense<0.0> : tensor<32x128x7x7xf32>
    %v623 = stablehlo.constant dense<6.0> : tensor<32x128x7x7xf32>
    %v624 = stablehlo.compare GT, %v621, %v622 : (tensor<32x128x7x7xf32>, tensor<32x128x7x7xf32>) -> tensor<32x128x7x7xi1>
    %v625 = stablehlo.compare LT, %v621, %v623 : (tensor<32x128x7x7xf32>, tensor<32x128x7x7xf32>) -> tensor<32x128x7x7xi1>
    %v626 = stablehlo.and %v624, %v625 : tensor<32x128x7x7xi1>
    %v627 = stablehlo.select %v626, %v620, %v622 : tensor<32x128x7x7xi1>, tensor<32x128x7x7xf32>
    %v628 = stablehlo.reshape %v627 : (tensor<32x128x7x7xf32>) -> tensor<32x6272xf32>
    %v629 = stablehlo.reshape %v628 : (tensor<32x6272xf32>) -> tensor<32x128x7x7xf32>
    %v630 = stablehlo.reshape %v565 : (tensor<32x6272xf32>) -> tensor<32x128x7x7xf32>
    %v631 = stablehlo.constant dense<0.0> : tensor<f32>
    %v632 = stablehlo.constant dense<49.0> : tensor<32x128x7x7xf32>
    %v633 = stablehlo.constant dense<1.0e-5> : tensor<32x128x7x7xf32>
    %v634 = stablehlo.reduce(%v630 init: %v631) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x7x7xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v635 = stablehlo.broadcast_in_dim %v634, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x7x7xf32>
    %v636 = stablehlo.divide %v635, %v632 : tensor<32x128x7x7xf32>
    %v637 = stablehlo.subtract %v630, %v636 : tensor<32x128x7x7xf32>
    %v638 = stablehlo.multiply %v637, %v637 : tensor<32x128x7x7xf32>
    %v639 = stablehlo.reduce(%v638 init: %v631) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x7x7xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v640 = stablehlo.broadcast_in_dim %v639, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x7x7xf32>
    %v641 = stablehlo.divide %v640, %v632 : tensor<32x128x7x7xf32>
    %v642 = stablehlo.add %v641, %v633 : tensor<32x128x7x7xf32>
    %v643 = stablehlo.rsqrt %v642 : tensor<32x128x7x7xf32>
    %v644 = stablehlo.multiply %v637, %v643 : tensor<32x128x7x7xf32>
    %v645 = stablehlo.broadcast_in_dim %gh, dims = [1] : (tensor<128xf32>) -> tensor<32x128x7x7xf32>
    %v646 = stablehlo.multiply %v645, %v629 : tensor<32x128x7x7xf32>
    %v647 = stablehlo.reduce(%v646 init: %v631) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x7x7xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v648 = stablehlo.broadcast_in_dim %v647, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x7x7xf32>
    %v649 = stablehlo.multiply %v644, %v646 : tensor<32x128x7x7xf32>
    %v650 = stablehlo.reduce(%v649 init: %v631) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x7x7xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v651 = stablehlo.broadcast_in_dim %v650, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x7x7xf32>
    %v652 = stablehlo.multiply %v646, %v632 : tensor<32x128x7x7xf32>
    %v653 = stablehlo.subtract %v652, %v648 : tensor<32x128x7x7xf32>
    %v654 = stablehlo.multiply %v644, %v651 : tensor<32x128x7x7xf32>
    %v655 = stablehlo.subtract %v653, %v654 : tensor<32x128x7x7xf32>
    %v656 = stablehlo.divide %v643, %v632 : tensor<32x128x7x7xf32>
    %v657 = stablehlo.multiply %v656, %v655 : tensor<32x128x7x7xf32>
    %v658 = stablehlo.reshape %v657 : (tensor<32x128x7x7xf32>) -> tensor<32x6272xf32>
    %v659 = stablehlo.reshape %v658 : (tensor<32x6272xf32>) -> tensor<32x128x7x7xf32>
    %v660 = stablehlo.transpose %Wh, dims = [1, 0, 2, 3] : (tensor<128x64x1x1xf32>) -> tensor<64x128x1x1xf32>
    %v661 = stablehlo.reverse %v660, dims = [2, 3] : tensor<64x128x1x1xf32>
    %v662 = stablehlo.convolution(%v659, %v661)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x7x7xf32>, tensor<64x128x1x1xf32>) -> tensor<32x64x7x7xf32>
    %v663 = stablehlo.reshape %v662 : (tensor<32x64x7x7xf32>) -> tensor<32x3136xf32>
    %v664 = stablehlo.reshape %v560 : (tensor<32x3136xf32>) -> tensor<32x64x7x7xf32>
    %v665 = stablehlo.reshape %v658 : (tensor<32x6272xf32>) -> tensor<32x128x7x7xf32>
    %v666 = stablehlo.transpose %v664, dims = [1, 0, 2, 3] : (tensor<32x64x7x7xf32>) -> tensor<64x32x7x7xf32>
    %v667 = stablehlo.transpose %v665, dims = [1, 0, 2, 3] : (tensor<32x128x7x7xf32>) -> tensor<128x32x7x7xf32>
    %v668 = stablehlo.convolution(%v666, %v667)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x7x7xf32>, tensor<128x32x7x7xf32>) -> tensor<64x128x1x1xf32>
    %v669 = stablehlo.transpose %v668, dims = [1, 0, 2, 3] : (tensor<64x128x1x1xf32>) -> tensor<128x64x1x1xf32>
    %v670 = stablehlo.constant dense<0.3> : tensor<128x64x1x1xf32>
    %v671 = stablehlo.multiply %v669, %v670 : tensor<128x64x1x1xf32>
    %v672 = stablehlo.subtract %Wh, %v671 : tensor<128x64x1x1xf32>
    %v673 = stablehlo.constant dense<0.0> : tensor<f32>
    %v674 = stablehlo.reshape %v565 : (tensor<32x6272xf32>) -> tensor<32x128x7x7xf32>
    %v675 = stablehlo.constant dense<49.0> : tensor<32x128x7x7xf32>
    %v676 = stablehlo.constant dense<1.0e-5> : tensor<32x128x7x7xf32>
    %v677 = stablehlo.reduce(%v674 init: %v673) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x7x7xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v678 = stablehlo.broadcast_in_dim %v677, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x7x7xf32>
    %v679 = stablehlo.divide %v678, %v675 : tensor<32x128x7x7xf32>
    %v680 = stablehlo.subtract %v674, %v679 : tensor<32x128x7x7xf32>
    %v681 = stablehlo.multiply %v680, %v680 : tensor<32x128x7x7xf32>
    %v682 = stablehlo.reduce(%v681 init: %v673) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x7x7xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v683 = stablehlo.broadcast_in_dim %v682, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x7x7xf32>
    %v684 = stablehlo.divide %v683, %v675 : tensor<32x128x7x7xf32>
    %v685 = stablehlo.add %v684, %v676 : tensor<32x128x7x7xf32>
    %v686 = stablehlo.rsqrt %v685 : tensor<32x128x7x7xf32>
    %v687 = stablehlo.multiply %v680, %v686 : tensor<32x128x7x7xf32>
    %v688 = stablehlo.reshape %v628 : (tensor<32x6272xf32>) -> tensor<32x128x7x7xf32>
    %v689 = stablehlo.multiply %v688, %v687 : tensor<32x128x7x7xf32>
    %v690 = stablehlo.reduce(%v689 init: %v673) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x7x7xf32>, tensor<f32>) -> tensor<128xf32>
    %v691 = stablehlo.constant dense<0.3> : tensor<128xf32>
    %v692 = stablehlo.multiply %v690, %v691 : tensor<128xf32>
    %v693 = stablehlo.subtract %gh, %v692 : tensor<128xf32>
    %v694 = stablehlo.constant dense<0.0> : tensor<f32>
    %v695 = stablehlo.reshape %v628 : (tensor<32x6272xf32>) -> tensor<32x128x7x7xf32>
    %v696 = stablehlo.reduce(%v695 init: %v694) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x7x7xf32>, tensor<f32>) -> tensor<128xf32>
    %v697 = stablehlo.constant dense<0.3> : tensor<128xf32>
    %v698 = stablehlo.multiply %v696, %v697 : tensor<128xf32>
    %v699 = stablehlo.subtract %bth, %v698 : tensor<128xf32>
    %v700 = stablehlo.reshape %v663 : (tensor<32x3136xf32>) -> tensor<32x64x7x7xf32>
    %v701 = stablehlo.reshape %v540 : (tensor<32x3136xf32>) -> tensor<32x64x7x7xf32>
    %v702 = stablehlo.constant dense<0.0> : tensor<f32>
    %v703 = stablehlo.constant dense<49.0> : tensor<32x64x7x7xf32>
    %v704 = stablehlo.constant dense<1.0e-5> : tensor<32x64x7x7xf32>
    %v705 = stablehlo.reduce(%v701 init: %v702) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x7x7xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v706 = stablehlo.broadcast_in_dim %v705, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x7x7xf32>
    %v707 = stablehlo.divide %v706, %v703 : tensor<32x64x7x7xf32>
    %v708 = stablehlo.subtract %v701, %v707 : tensor<32x64x7x7xf32>
    %v709 = stablehlo.multiply %v708, %v708 : tensor<32x64x7x7xf32>
    %v710 = stablehlo.reduce(%v709 init: %v702) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x7x7xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v711 = stablehlo.broadcast_in_dim %v710, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x7x7xf32>
    %v712 = stablehlo.divide %v711, %v703 : tensor<32x64x7x7xf32>
    %v713 = stablehlo.add %v712, %v704 : tensor<32x64x7x7xf32>
    %v714 = stablehlo.rsqrt %v713 : tensor<32x64x7x7xf32>
    %v715 = stablehlo.multiply %v708, %v714 : tensor<32x64x7x7xf32>
    %v716 = stablehlo.broadcast_in_dim %gp6, dims = [1] : (tensor<64xf32>) -> tensor<32x64x7x7xf32>
    %v717 = stablehlo.multiply %v716, %v700 : tensor<32x64x7x7xf32>
    %v718 = stablehlo.reduce(%v717 init: %v702) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x7x7xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v719 = stablehlo.broadcast_in_dim %v718, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x7x7xf32>
    %v720 = stablehlo.multiply %v715, %v717 : tensor<32x64x7x7xf32>
    %v721 = stablehlo.reduce(%v720 init: %v702) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x7x7xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v722 = stablehlo.broadcast_in_dim %v721, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x7x7xf32>
    %v723 = stablehlo.multiply %v717, %v703 : tensor<32x64x7x7xf32>
    %v724 = stablehlo.subtract %v723, %v719 : tensor<32x64x7x7xf32>
    %v725 = stablehlo.multiply %v715, %v722 : tensor<32x64x7x7xf32>
    %v726 = stablehlo.subtract %v724, %v725 : tensor<32x64x7x7xf32>
    %v727 = stablehlo.divide %v714, %v703 : tensor<32x64x7x7xf32>
    %v728 = stablehlo.multiply %v727, %v726 : tensor<32x64x7x7xf32>
    %v729 = stablehlo.reshape %v728 : (tensor<32x64x7x7xf32>) -> tensor<32x3136xf32>
    %v730 = stablehlo.reshape %v729 : (tensor<32x3136xf32>) -> tensor<32x64x7x7xf32>
    %v731 = stablehlo.transpose %Wp6, dims = [1, 0, 2, 3] : (tensor<64x256x1x1xf32>) -> tensor<256x64x1x1xf32>
    %v732 = stablehlo.reverse %v731, dims = [2, 3] : tensor<256x64x1x1xf32>
    %v733 = stablehlo.convolution(%v730, %v732)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x7x7xf32>, tensor<256x64x1x1xf32>) -> tensor<32x256x7x7xf32>
    %v734 = stablehlo.reshape %v733 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v735 = stablehlo.reshape %v734 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v736 = stablehlo.reshape %v529 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v737 = stablehlo.constant dense<0.0> : tensor<32x256x7x7xf32>
    %v738 = stablehlo.constant dense<6.0> : tensor<32x256x7x7xf32>
    %v739 = stablehlo.compare GT, %v736, %v737 : (tensor<32x256x7x7xf32>, tensor<32x256x7x7xf32>) -> tensor<32x256x7x7xi1>
    %v740 = stablehlo.compare LT, %v736, %v738 : (tensor<32x256x7x7xf32>, tensor<32x256x7x7xf32>) -> tensor<32x256x7x7xi1>
    %v741 = stablehlo.and %v739, %v740 : tensor<32x256x7x7xi1>
    %v742 = stablehlo.select %v741, %v735, %v737 : tensor<32x256x7x7xi1>, tensor<32x256x7x7xf32>
    %v743 = stablehlo.reshape %v742 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v744 = stablehlo.reshape %v743 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v745 = stablehlo.reshape %v509 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v746 = stablehlo.constant dense<0.0> : tensor<f32>
    %v747 = stablehlo.constant dense<49.0> : tensor<32x256x7x7xf32>
    %v748 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v749 = stablehlo.reduce(%v745 init: %v746) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v750 = stablehlo.broadcast_in_dim %v749, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x7x7xf32>
    %v751 = stablehlo.divide %v750, %v747 : tensor<32x256x7x7xf32>
    %v752 = stablehlo.subtract %v745, %v751 : tensor<32x256x7x7xf32>
    %v753 = stablehlo.multiply %v752, %v752 : tensor<32x256x7x7xf32>
    %v754 = stablehlo.reduce(%v753 init: %v746) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v755 = stablehlo.broadcast_in_dim %v754, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x7x7xf32>
    %v756 = stablehlo.divide %v755, %v747 : tensor<32x256x7x7xf32>
    %v757 = stablehlo.add %v756, %v748 : tensor<32x256x7x7xf32>
    %v758 = stablehlo.rsqrt %v757 : tensor<32x256x7x7xf32>
    %v759 = stablehlo.multiply %v752, %v758 : tensor<32x256x7x7xf32>
    %v760 = stablehlo.broadcast_in_dim %gd6, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v761 = stablehlo.multiply %v760, %v744 : tensor<32x256x7x7xf32>
    %v762 = stablehlo.reduce(%v761 init: %v746) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v763 = stablehlo.broadcast_in_dim %v762, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x7x7xf32>
    %v764 = stablehlo.multiply %v759, %v761 : tensor<32x256x7x7xf32>
    %v765 = stablehlo.reduce(%v764 init: %v746) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v766 = stablehlo.broadcast_in_dim %v765, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x7x7xf32>
    %v767 = stablehlo.multiply %v761, %v747 : tensor<32x256x7x7xf32>
    %v768 = stablehlo.subtract %v767, %v763 : tensor<32x256x7x7xf32>
    %v769 = stablehlo.multiply %v759, %v766 : tensor<32x256x7x7xf32>
    %v770 = stablehlo.subtract %v768, %v769 : tensor<32x256x7x7xf32>
    %v771 = stablehlo.divide %v758, %v747 : tensor<32x256x7x7xf32>
    %v772 = stablehlo.multiply %v771, %v770 : tensor<32x256x7x7xf32>
    %v773 = stablehlo.reshape %v772 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v774 = stablehlo.reshape %v773 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v775 = stablehlo.constant dense<0.0> : tensor<f32>
    %v776 = stablehlo.pad %v774, %v775, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<32x256x14x14xf32>
    %v777 = stablehlo.reverse %Wd6, dims = [2, 3] : tensor<256x1x3x3xf32>
    %v778 = stablehlo.convolution(%v776, %v777)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 256 : i64} : (tensor<32x256x14x14xf32>, tensor<256x1x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v779 = stablehlo.reshape %v778 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v780 = stablehlo.reshape %v779 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v781 = stablehlo.reshape %v498 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v782 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v783 = stablehlo.constant dense<6.0> : tensor<32x256x14x14xf32>
    %v784 = stablehlo.compare GT, %v781, %v782 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v785 = stablehlo.compare LT, %v781, %v783 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v786 = stablehlo.and %v784, %v785 : tensor<32x256x14x14xi1>
    %v787 = stablehlo.select %v786, %v780, %v782 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v788 = stablehlo.reshape %v787 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v789 = stablehlo.reshape %v788 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v790 = stablehlo.reshape %v478 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v791 = stablehlo.constant dense<0.0> : tensor<f32>
    %v792 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v793 = stablehlo.constant dense<1.0e-5> : tensor<32x256x14x14xf32>
    %v794 = stablehlo.reduce(%v790 init: %v791) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v795 = stablehlo.broadcast_in_dim %v794, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v796 = stablehlo.divide %v795, %v792 : tensor<32x256x14x14xf32>
    %v797 = stablehlo.subtract %v790, %v796 : tensor<32x256x14x14xf32>
    %v798 = stablehlo.multiply %v797, %v797 : tensor<32x256x14x14xf32>
    %v799 = stablehlo.reduce(%v798 init: %v791) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v800 = stablehlo.broadcast_in_dim %v799, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v801 = stablehlo.divide %v800, %v792 : tensor<32x256x14x14xf32>
    %v802 = stablehlo.add %v801, %v793 : tensor<32x256x14x14xf32>
    %v803 = stablehlo.rsqrt %v802 : tensor<32x256x14x14xf32>
    %v804 = stablehlo.multiply %v797, %v803 : tensor<32x256x14x14xf32>
    %v805 = stablehlo.broadcast_in_dim %ge6, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v806 = stablehlo.multiply %v805, %v789 : tensor<32x256x14x14xf32>
    %v807 = stablehlo.reduce(%v806 init: %v791) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v808 = stablehlo.broadcast_in_dim %v807, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v809 = stablehlo.multiply %v804, %v806 : tensor<32x256x14x14xf32>
    %v810 = stablehlo.reduce(%v809 init: %v791) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v811 = stablehlo.broadcast_in_dim %v810, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v812 = stablehlo.multiply %v806, %v792 : tensor<32x256x14x14xf32>
    %v813 = stablehlo.subtract %v812, %v808 : tensor<32x256x14x14xf32>
    %v814 = stablehlo.multiply %v804, %v811 : tensor<32x256x14x14xf32>
    %v815 = stablehlo.subtract %v813, %v814 : tensor<32x256x14x14xf32>
    %v816 = stablehlo.divide %v803, %v792 : tensor<32x256x14x14xf32>
    %v817 = stablehlo.multiply %v816, %v815 : tensor<32x256x14x14xf32>
    %v818 = stablehlo.reshape %v817 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v819 = stablehlo.reshape %v818 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v820 = stablehlo.transpose %We6, dims = [1, 0, 2, 3] : (tensor<256x64x1x1xf32>) -> tensor<64x256x1x1xf32>
    %v821 = stablehlo.reverse %v820, dims = [2, 3] : tensor<64x256x1x1xf32>
    %v822 = stablehlo.convolution(%v819, %v821)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<64x256x1x1xf32>) -> tensor<32x64x14x14xf32>
    %v823 = stablehlo.reshape %v822 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v824 = stablehlo.reshape %v473 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v825 = stablehlo.reshape %v818 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v826 = stablehlo.transpose %v824, dims = [1, 0, 2, 3] : (tensor<32x64x14x14xf32>) -> tensor<64x32x14x14xf32>
    %v827 = stablehlo.transpose %v825, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v828 = stablehlo.convolution(%v826, %v827)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<64x256x1x1xf32>
    %v829 = stablehlo.transpose %v828, dims = [1, 0, 2, 3] : (tensor<64x256x1x1xf32>) -> tensor<256x64x1x1xf32>
    %v830 = stablehlo.constant dense<0.3> : tensor<256x64x1x1xf32>
    %v831 = stablehlo.multiply %v829, %v830 : tensor<256x64x1x1xf32>
    %v832 = stablehlo.subtract %We6, %v831 : tensor<256x64x1x1xf32>
    %v833 = stablehlo.constant dense<0.0> : tensor<f32>
    %v834 = stablehlo.reshape %v478 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v835 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v836 = stablehlo.constant dense<1.0e-5> : tensor<32x256x14x14xf32>
    %v837 = stablehlo.reduce(%v834 init: %v833) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v838 = stablehlo.broadcast_in_dim %v837, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v839 = stablehlo.divide %v838, %v835 : tensor<32x256x14x14xf32>
    %v840 = stablehlo.subtract %v834, %v839 : tensor<32x256x14x14xf32>
    %v841 = stablehlo.multiply %v840, %v840 : tensor<32x256x14x14xf32>
    %v842 = stablehlo.reduce(%v841 init: %v833) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v843 = stablehlo.broadcast_in_dim %v842, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v844 = stablehlo.divide %v843, %v835 : tensor<32x256x14x14xf32>
    %v845 = stablehlo.add %v844, %v836 : tensor<32x256x14x14xf32>
    %v846 = stablehlo.rsqrt %v845 : tensor<32x256x14x14xf32>
    %v847 = stablehlo.multiply %v840, %v846 : tensor<32x256x14x14xf32>
    %v848 = stablehlo.reshape %v788 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v849 = stablehlo.multiply %v848, %v847 : tensor<32x256x14x14xf32>
    %v850 = stablehlo.reduce(%v849 init: %v833) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v851 = stablehlo.constant dense<0.3> : tensor<256xf32>
    %v852 = stablehlo.multiply %v850, %v851 : tensor<256xf32>
    %v853 = stablehlo.subtract %ge6, %v852 : tensor<256xf32>
    %v854 = stablehlo.constant dense<0.0> : tensor<f32>
    %v855 = stablehlo.reshape %v788 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v856 = stablehlo.reduce(%v855 init: %v854) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v857 = stablehlo.constant dense<0.3> : tensor<256xf32>
    %v858 = stablehlo.multiply %v856, %v857 : tensor<256xf32>
    %v859 = stablehlo.subtract %bte6, %v858 : tensor<256xf32>
    %v860 = stablehlo.reshape %v504 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v861 = stablehlo.reshape %v773 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v862 = stablehlo.constant dense<0.0> : tensor<f32>
    %v863 = stablehlo.pad %v861, %v862, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<32x256x14x14xf32>
    %v864 = stablehlo.transpose %v860, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v865 = stablehlo.transpose %v863, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v866 = stablehlo.convolution(%v864, %v865)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 256 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<1x256x3x3xf32>
    %v867 = stablehlo.reshape %v866 : (tensor<1x256x3x3xf32>) -> tensor<256x1x3x3xf32>
    %v868 = stablehlo.constant dense<0.3> : tensor<256x1x3x3xf32>
    %v869 = stablehlo.multiply %v867, %v868 : tensor<256x1x3x3xf32>
    %v870 = stablehlo.subtract %Wd6, %v869 : tensor<256x1x3x3xf32>
    %v871 = stablehlo.constant dense<0.0> : tensor<f32>
    %v872 = stablehlo.reshape %v509 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v873 = stablehlo.constant dense<49.0> : tensor<32x256x7x7xf32>
    %v874 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v875 = stablehlo.reduce(%v872 init: %v871) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v876 = stablehlo.broadcast_in_dim %v875, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x7x7xf32>
    %v877 = stablehlo.divide %v876, %v873 : tensor<32x256x7x7xf32>
    %v878 = stablehlo.subtract %v872, %v877 : tensor<32x256x7x7xf32>
    %v879 = stablehlo.multiply %v878, %v878 : tensor<32x256x7x7xf32>
    %v880 = stablehlo.reduce(%v879 init: %v871) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v881 = stablehlo.broadcast_in_dim %v880, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x7x7xf32>
    %v882 = stablehlo.divide %v881, %v873 : tensor<32x256x7x7xf32>
    %v883 = stablehlo.add %v882, %v874 : tensor<32x256x7x7xf32>
    %v884 = stablehlo.rsqrt %v883 : tensor<32x256x7x7xf32>
    %v885 = stablehlo.multiply %v878, %v884 : tensor<32x256x7x7xf32>
    %v886 = stablehlo.reshape %v743 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v887 = stablehlo.multiply %v886, %v885 : tensor<32x256x7x7xf32>
    %v888 = stablehlo.reduce(%v887 init: %v871) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v889 = stablehlo.constant dense<0.3> : tensor<256xf32>
    %v890 = stablehlo.multiply %v888, %v889 : tensor<256xf32>
    %v891 = stablehlo.subtract %gd6, %v890 : tensor<256xf32>
    %v892 = stablehlo.constant dense<0.0> : tensor<f32>
    %v893 = stablehlo.reshape %v743 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v894 = stablehlo.reduce(%v893 init: %v892) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v895 = stablehlo.constant dense<0.3> : tensor<256xf32>
    %v896 = stablehlo.multiply %v894, %v895 : tensor<256xf32>
    %v897 = stablehlo.subtract %btd6, %v896 : tensor<256xf32>
    %v898 = stablehlo.reshape %v535 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v899 = stablehlo.reshape %v729 : (tensor<32x3136xf32>) -> tensor<32x64x7x7xf32>
    %v900 = stablehlo.transpose %v898, dims = [1, 0, 2, 3] : (tensor<32x256x7x7xf32>) -> tensor<256x32x7x7xf32>
    %v901 = stablehlo.transpose %v899, dims = [1, 0, 2, 3] : (tensor<32x64x7x7xf32>) -> tensor<64x32x7x7xf32>
    %v902 = stablehlo.convolution(%v900, %v901)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x7x7xf32>, tensor<64x32x7x7xf32>) -> tensor<256x64x1x1xf32>
    %v903 = stablehlo.transpose %v902, dims = [1, 0, 2, 3] : (tensor<256x64x1x1xf32>) -> tensor<64x256x1x1xf32>
    %v904 = stablehlo.constant dense<0.3> : tensor<64x256x1x1xf32>
    %v905 = stablehlo.multiply %v903, %v904 : tensor<64x256x1x1xf32>
    %v906 = stablehlo.subtract %Wp6, %v905 : tensor<64x256x1x1xf32>
    %v907 = stablehlo.constant dense<0.0> : tensor<f32>
    %v908 = stablehlo.reshape %v540 : (tensor<32x3136xf32>) -> tensor<32x64x7x7xf32>
    %v909 = stablehlo.constant dense<49.0> : tensor<32x64x7x7xf32>
    %v910 = stablehlo.constant dense<1.0e-5> : tensor<32x64x7x7xf32>
    %v911 = stablehlo.reduce(%v908 init: %v907) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x7x7xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v912 = stablehlo.broadcast_in_dim %v911, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x7x7xf32>
    %v913 = stablehlo.divide %v912, %v909 : tensor<32x64x7x7xf32>
    %v914 = stablehlo.subtract %v908, %v913 : tensor<32x64x7x7xf32>
    %v915 = stablehlo.multiply %v914, %v914 : tensor<32x64x7x7xf32>
    %v916 = stablehlo.reduce(%v915 init: %v907) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x7x7xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v917 = stablehlo.broadcast_in_dim %v916, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x7x7xf32>
    %v918 = stablehlo.divide %v917, %v909 : tensor<32x64x7x7xf32>
    %v919 = stablehlo.add %v918, %v910 : tensor<32x64x7x7xf32>
    %v920 = stablehlo.rsqrt %v919 : tensor<32x64x7x7xf32>
    %v921 = stablehlo.multiply %v914, %v920 : tensor<32x64x7x7xf32>
    %v922 = stablehlo.reshape %v663 : (tensor<32x3136xf32>) -> tensor<32x64x7x7xf32>
    %v923 = stablehlo.multiply %v922, %v921 : tensor<32x64x7x7xf32>
    %v924 = stablehlo.reduce(%v923 init: %v907) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x7x7xf32>, tensor<f32>) -> tensor<64xf32>
    %v925 = stablehlo.constant dense<0.3> : tensor<64xf32>
    %v926 = stablehlo.multiply %v924, %v925 : tensor<64xf32>
    %v927 = stablehlo.subtract %gp6, %v926 : tensor<64xf32>
    %v928 = stablehlo.constant dense<0.0> : tensor<f32>
    %v929 = stablehlo.reshape %v663 : (tensor<32x3136xf32>) -> tensor<32x64x7x7xf32>
    %v930 = stablehlo.reduce(%v929 init: %v928) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x7x7xf32>, tensor<f32>) -> tensor<64xf32>
    %v931 = stablehlo.constant dense<0.3> : tensor<64xf32>
    %v932 = stablehlo.multiply %v930, %v931 : tensor<64xf32>
    %v933 = stablehlo.subtract %btp6, %v932 : tensor<64xf32>
    %v934 = stablehlo.reshape %v823 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v935 = stablehlo.reshape %v453 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v936 = stablehlo.constant dense<0.0> : tensor<f32>
    %v937 = stablehlo.constant dense<196.0> : tensor<32x64x14x14xf32>
    %v938 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v939 = stablehlo.reduce(%v935 init: %v936) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v940 = stablehlo.broadcast_in_dim %v939, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v941 = stablehlo.divide %v940, %v937 : tensor<32x64x14x14xf32>
    %v942 = stablehlo.subtract %v935, %v941 : tensor<32x64x14x14xf32>
    %v943 = stablehlo.multiply %v942, %v942 : tensor<32x64x14x14xf32>
    %v944 = stablehlo.reduce(%v943 init: %v936) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v945 = stablehlo.broadcast_in_dim %v944, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v946 = stablehlo.divide %v945, %v937 : tensor<32x64x14x14xf32>
    %v947 = stablehlo.add %v946, %v938 : tensor<32x64x14x14xf32>
    %v948 = stablehlo.rsqrt %v947 : tensor<32x64x14x14xf32>
    %v949 = stablehlo.multiply %v942, %v948 : tensor<32x64x14x14xf32>
    %v950 = stablehlo.broadcast_in_dim %gp5, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v951 = stablehlo.multiply %v950, %v934 : tensor<32x64x14x14xf32>
    %v952 = stablehlo.reduce(%v951 init: %v936) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v953 = stablehlo.broadcast_in_dim %v952, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v954 = stablehlo.multiply %v949, %v951 : tensor<32x64x14x14xf32>
    %v955 = stablehlo.reduce(%v954 init: %v936) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v956 = stablehlo.broadcast_in_dim %v955, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v957 = stablehlo.multiply %v951, %v937 : tensor<32x64x14x14xf32>
    %v958 = stablehlo.subtract %v957, %v953 : tensor<32x64x14x14xf32>
    %v959 = stablehlo.multiply %v949, %v956 : tensor<32x64x14x14xf32>
    %v960 = stablehlo.subtract %v958, %v959 : tensor<32x64x14x14xf32>
    %v961 = stablehlo.divide %v948, %v937 : tensor<32x64x14x14xf32>
    %v962 = stablehlo.multiply %v961, %v960 : tensor<32x64x14x14xf32>
    %v963 = stablehlo.reshape %v962 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v964 = stablehlo.reshape %v963 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v965 = stablehlo.transpose %Wp5, dims = [1, 0, 2, 3] : (tensor<64x128x1x1xf32>) -> tensor<128x64x1x1xf32>
    %v966 = stablehlo.reverse %v965, dims = [2, 3] : tensor<128x64x1x1xf32>
    %v967 = stablehlo.convolution(%v964, %v966)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x14x14xf32>, tensor<128x64x1x1xf32>) -> tensor<32x128x14x14xf32>
    %v968 = stablehlo.reshape %v967 : (tensor<32x128x14x14xf32>) -> tensor<32x25088xf32>
    %v969 = stablehlo.reshape %v968 : (tensor<32x25088xf32>) -> tensor<32x128x14x14xf32>
    %v970 = stablehlo.reshape %v442 : (tensor<32x25088xf32>) -> tensor<32x128x14x14xf32>
    %v971 = stablehlo.constant dense<0.0> : tensor<32x128x14x14xf32>
    %v972 = stablehlo.constant dense<6.0> : tensor<32x128x14x14xf32>
    %v973 = stablehlo.compare GT, %v970, %v971 : (tensor<32x128x14x14xf32>, tensor<32x128x14x14xf32>) -> tensor<32x128x14x14xi1>
    %v974 = stablehlo.compare LT, %v970, %v972 : (tensor<32x128x14x14xf32>, tensor<32x128x14x14xf32>) -> tensor<32x128x14x14xi1>
    %v975 = stablehlo.and %v973, %v974 : tensor<32x128x14x14xi1>
    %v976 = stablehlo.select %v975, %v969, %v971 : tensor<32x128x14x14xi1>, tensor<32x128x14x14xf32>
    %v977 = stablehlo.reshape %v976 : (tensor<32x128x14x14xf32>) -> tensor<32x25088xf32>
    %v978 = stablehlo.reshape %v977 : (tensor<32x25088xf32>) -> tensor<32x128x14x14xf32>
    %v979 = stablehlo.reshape %v422 : (tensor<32x25088xf32>) -> tensor<32x128x14x14xf32>
    %v980 = stablehlo.constant dense<0.0> : tensor<f32>
    %v981 = stablehlo.constant dense<196.0> : tensor<32x128x14x14xf32>
    %v982 = stablehlo.constant dense<1.0e-5> : tensor<32x128x14x14xf32>
    %v983 = stablehlo.reduce(%v979 init: %v980) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x14x14xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v984 = stablehlo.broadcast_in_dim %v983, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x14x14xf32>
    %v985 = stablehlo.divide %v984, %v981 : tensor<32x128x14x14xf32>
    %v986 = stablehlo.subtract %v979, %v985 : tensor<32x128x14x14xf32>
    %v987 = stablehlo.multiply %v986, %v986 : tensor<32x128x14x14xf32>
    %v988 = stablehlo.reduce(%v987 init: %v980) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x14x14xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v989 = stablehlo.broadcast_in_dim %v988, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x14x14xf32>
    %v990 = stablehlo.divide %v989, %v981 : tensor<32x128x14x14xf32>
    %v991 = stablehlo.add %v990, %v982 : tensor<32x128x14x14xf32>
    %v992 = stablehlo.rsqrt %v991 : tensor<32x128x14x14xf32>
    %v993 = stablehlo.multiply %v986, %v992 : tensor<32x128x14x14xf32>
    %v994 = stablehlo.broadcast_in_dim %gd5, dims = [1] : (tensor<128xf32>) -> tensor<32x128x14x14xf32>
    %v995 = stablehlo.multiply %v994, %v978 : tensor<32x128x14x14xf32>
    %v996 = stablehlo.reduce(%v995 init: %v980) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x14x14xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v997 = stablehlo.broadcast_in_dim %v996, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x14x14xf32>
    %v998 = stablehlo.multiply %v993, %v995 : tensor<32x128x14x14xf32>
    %v999 = stablehlo.reduce(%v998 init: %v980) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x14x14xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v1000 = stablehlo.broadcast_in_dim %v999, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x14x14xf32>
    %v1001 = stablehlo.multiply %v995, %v981 : tensor<32x128x14x14xf32>
    %v1002 = stablehlo.subtract %v1001, %v997 : tensor<32x128x14x14xf32>
    %v1003 = stablehlo.multiply %v993, %v1000 : tensor<32x128x14x14xf32>
    %v1004 = stablehlo.subtract %v1002, %v1003 : tensor<32x128x14x14xf32>
    %v1005 = stablehlo.divide %v992, %v981 : tensor<32x128x14x14xf32>
    %v1006 = stablehlo.multiply %v1005, %v1004 : tensor<32x128x14x14xf32>
    %v1007 = stablehlo.reshape %v1006 : (tensor<32x128x14x14xf32>) -> tensor<32x25088xf32>
    %v1008 = stablehlo.reshape %v1007 : (tensor<32x25088xf32>) -> tensor<32x128x14x14xf32>
    %v1009 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1010 = stablehlo.pad %v1008, %v1009, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x128x14x14xf32>, tensor<f32>) -> tensor<32x128x28x28xf32>
    %v1011 = stablehlo.reverse %Wd5, dims = [2, 3] : tensor<128x1x3x3xf32>
    %v1012 = stablehlo.convolution(%v1010, %v1011)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 128 : i64} : (tensor<32x128x28x28xf32>, tensor<128x1x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v1013 = stablehlo.reshape %v1012 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v1014 = stablehlo.reshape %v1013 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v1015 = stablehlo.reshape %v411 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v1016 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v1017 = stablehlo.constant dense<6.0> : tensor<32x128x28x28xf32>
    %v1018 = stablehlo.compare GT, %v1015, %v1016 : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xi1>
    %v1019 = stablehlo.compare LT, %v1015, %v1017 : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xi1>
    %v1020 = stablehlo.and %v1018, %v1019 : tensor<32x128x28x28xi1>
    %v1021 = stablehlo.select %v1020, %v1014, %v1016 : tensor<32x128x28x28xi1>, tensor<32x128x28x28xf32>
    %v1022 = stablehlo.reshape %v1021 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v1023 = stablehlo.reshape %v1022 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v1024 = stablehlo.reshape %v391 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v1025 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1026 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v1027 = stablehlo.constant dense<1.0e-5> : tensor<32x128x28x28xf32>
    %v1028 = stablehlo.reduce(%v1024 init: %v1025) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v1029 = stablehlo.broadcast_in_dim %v1028, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v1030 = stablehlo.divide %v1029, %v1026 : tensor<32x128x28x28xf32>
    %v1031 = stablehlo.subtract %v1024, %v1030 : tensor<32x128x28x28xf32>
    %v1032 = stablehlo.multiply %v1031, %v1031 : tensor<32x128x28x28xf32>
    %v1033 = stablehlo.reduce(%v1032 init: %v1025) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v1034 = stablehlo.broadcast_in_dim %v1033, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v1035 = stablehlo.divide %v1034, %v1026 : tensor<32x128x28x28xf32>
    %v1036 = stablehlo.add %v1035, %v1027 : tensor<32x128x28x28xf32>
    %v1037 = stablehlo.rsqrt %v1036 : tensor<32x128x28x28xf32>
    %v1038 = stablehlo.multiply %v1031, %v1037 : tensor<32x128x28x28xf32>
    %v1039 = stablehlo.broadcast_in_dim %ge5, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v1040 = stablehlo.multiply %v1039, %v1023 : tensor<32x128x28x28xf32>
    %v1041 = stablehlo.reduce(%v1040 init: %v1025) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v1042 = stablehlo.broadcast_in_dim %v1041, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v1043 = stablehlo.multiply %v1038, %v1040 : tensor<32x128x28x28xf32>
    %v1044 = stablehlo.reduce(%v1043 init: %v1025) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v1045 = stablehlo.broadcast_in_dim %v1044, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v1046 = stablehlo.multiply %v1040, %v1026 : tensor<32x128x28x28xf32>
    %v1047 = stablehlo.subtract %v1046, %v1042 : tensor<32x128x28x28xf32>
    %v1048 = stablehlo.multiply %v1038, %v1045 : tensor<32x128x28x28xf32>
    %v1049 = stablehlo.subtract %v1047, %v1048 : tensor<32x128x28x28xf32>
    %v1050 = stablehlo.divide %v1037, %v1026 : tensor<32x128x28x28xf32>
    %v1051 = stablehlo.multiply %v1050, %v1049 : tensor<32x128x28x28xf32>
    %v1052 = stablehlo.reshape %v1051 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v1053 = stablehlo.reshape %v1052 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v1054 = stablehlo.transpose %We5, dims = [1, 0, 2, 3] : (tensor<128x32x1x1xf32>) -> tensor<32x128x1x1xf32>
    %v1055 = stablehlo.reverse %v1054, dims = [2, 3] : tensor<32x128x1x1xf32>
    %v1056 = stablehlo.convolution(%v1053, %v1055)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<32x128x1x1xf32>) -> tensor<32x32x28x28xf32>
    %v1057 = stablehlo.reshape %v1056 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v1058 = stablehlo.reshape %v386 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v1059 = stablehlo.reshape %v1052 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v1060 = stablehlo.transpose %v1058, dims = [1, 0, 2, 3] : (tensor<32x32x28x28xf32>) -> tensor<32x32x28x28xf32>
    %v1061 = stablehlo.transpose %v1059, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v1062 = stablehlo.convolution(%v1060, %v1061)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<32x128x1x1xf32>
    %v1063 = stablehlo.transpose %v1062, dims = [1, 0, 2, 3] : (tensor<32x128x1x1xf32>) -> tensor<128x32x1x1xf32>
    %v1064 = stablehlo.constant dense<0.3> : tensor<128x32x1x1xf32>
    %v1065 = stablehlo.multiply %v1063, %v1064 : tensor<128x32x1x1xf32>
    %v1066 = stablehlo.subtract %We5, %v1065 : tensor<128x32x1x1xf32>
    %v1067 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1068 = stablehlo.reshape %v391 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v1069 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v1070 = stablehlo.constant dense<1.0e-5> : tensor<32x128x28x28xf32>
    %v1071 = stablehlo.reduce(%v1068 init: %v1067) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v1072 = stablehlo.broadcast_in_dim %v1071, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v1073 = stablehlo.divide %v1072, %v1069 : tensor<32x128x28x28xf32>
    %v1074 = stablehlo.subtract %v1068, %v1073 : tensor<32x128x28x28xf32>
    %v1075 = stablehlo.multiply %v1074, %v1074 : tensor<32x128x28x28xf32>
    %v1076 = stablehlo.reduce(%v1075 init: %v1067) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v1077 = stablehlo.broadcast_in_dim %v1076, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v1078 = stablehlo.divide %v1077, %v1069 : tensor<32x128x28x28xf32>
    %v1079 = stablehlo.add %v1078, %v1070 : tensor<32x128x28x28xf32>
    %v1080 = stablehlo.rsqrt %v1079 : tensor<32x128x28x28xf32>
    %v1081 = stablehlo.multiply %v1074, %v1080 : tensor<32x128x28x28xf32>
    %v1082 = stablehlo.reshape %v1022 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v1083 = stablehlo.multiply %v1082, %v1081 : tensor<32x128x28x28xf32>
    %v1084 = stablehlo.reduce(%v1083 init: %v1067) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v1085 = stablehlo.constant dense<0.3> : tensor<128xf32>
    %v1086 = stablehlo.multiply %v1084, %v1085 : tensor<128xf32>
    %v1087 = stablehlo.subtract %ge5, %v1086 : tensor<128xf32>
    %v1088 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1089 = stablehlo.reshape %v1022 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v1090 = stablehlo.reduce(%v1089 init: %v1088) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v1091 = stablehlo.constant dense<0.3> : tensor<128xf32>
    %v1092 = stablehlo.multiply %v1090, %v1091 : tensor<128xf32>
    %v1093 = stablehlo.subtract %bte5, %v1092 : tensor<128xf32>
    %v1094 = stablehlo.reshape %v417 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v1095 = stablehlo.reshape %v1007 : (tensor<32x25088xf32>) -> tensor<32x128x14x14xf32>
    %v1096 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1097 = stablehlo.pad %v1095, %v1096, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x128x14x14xf32>, tensor<f32>) -> tensor<32x128x28x28xf32>
    %v1098 = stablehlo.transpose %v1094, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v1099 = stablehlo.transpose %v1097, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v1100 = stablehlo.convolution(%v1098, %v1099)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 128 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<1x128x3x3xf32>
    %v1101 = stablehlo.reshape %v1100 : (tensor<1x128x3x3xf32>) -> tensor<128x1x3x3xf32>
    %v1102 = stablehlo.constant dense<0.3> : tensor<128x1x3x3xf32>
    %v1103 = stablehlo.multiply %v1101, %v1102 : tensor<128x1x3x3xf32>
    %v1104 = stablehlo.subtract %Wd5, %v1103 : tensor<128x1x3x3xf32>
    %v1105 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1106 = stablehlo.reshape %v422 : (tensor<32x25088xf32>) -> tensor<32x128x14x14xf32>
    %v1107 = stablehlo.constant dense<196.0> : tensor<32x128x14x14xf32>
    %v1108 = stablehlo.constant dense<1.0e-5> : tensor<32x128x14x14xf32>
    %v1109 = stablehlo.reduce(%v1106 init: %v1105) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x14x14xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v1110 = stablehlo.broadcast_in_dim %v1109, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x14x14xf32>
    %v1111 = stablehlo.divide %v1110, %v1107 : tensor<32x128x14x14xf32>
    %v1112 = stablehlo.subtract %v1106, %v1111 : tensor<32x128x14x14xf32>
    %v1113 = stablehlo.multiply %v1112, %v1112 : tensor<32x128x14x14xf32>
    %v1114 = stablehlo.reduce(%v1113 init: %v1105) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x14x14xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v1115 = stablehlo.broadcast_in_dim %v1114, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x14x14xf32>
    %v1116 = stablehlo.divide %v1115, %v1107 : tensor<32x128x14x14xf32>
    %v1117 = stablehlo.add %v1116, %v1108 : tensor<32x128x14x14xf32>
    %v1118 = stablehlo.rsqrt %v1117 : tensor<32x128x14x14xf32>
    %v1119 = stablehlo.multiply %v1112, %v1118 : tensor<32x128x14x14xf32>
    %v1120 = stablehlo.reshape %v977 : (tensor<32x25088xf32>) -> tensor<32x128x14x14xf32>
    %v1121 = stablehlo.multiply %v1120, %v1119 : tensor<32x128x14x14xf32>
    %v1122 = stablehlo.reduce(%v1121 init: %v1105) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x14x14xf32>, tensor<f32>) -> tensor<128xf32>
    %v1123 = stablehlo.constant dense<0.3> : tensor<128xf32>
    %v1124 = stablehlo.multiply %v1122, %v1123 : tensor<128xf32>
    %v1125 = stablehlo.subtract %gd5, %v1124 : tensor<128xf32>
    %v1126 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1127 = stablehlo.reshape %v977 : (tensor<32x25088xf32>) -> tensor<32x128x14x14xf32>
    %v1128 = stablehlo.reduce(%v1127 init: %v1126) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x14x14xf32>, tensor<f32>) -> tensor<128xf32>
    %v1129 = stablehlo.constant dense<0.3> : tensor<128xf32>
    %v1130 = stablehlo.multiply %v1128, %v1129 : tensor<128xf32>
    %v1131 = stablehlo.subtract %btd5, %v1130 : tensor<128xf32>
    %v1132 = stablehlo.reshape %v448 : (tensor<32x25088xf32>) -> tensor<32x128x14x14xf32>
    %v1133 = stablehlo.reshape %v963 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v1134 = stablehlo.transpose %v1132, dims = [1, 0, 2, 3] : (tensor<32x128x14x14xf32>) -> tensor<128x32x14x14xf32>
    %v1135 = stablehlo.transpose %v1133, dims = [1, 0, 2, 3] : (tensor<32x64x14x14xf32>) -> tensor<64x32x14x14xf32>
    %v1136 = stablehlo.convolution(%v1134, %v1135)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x14x14xf32>, tensor<64x32x14x14xf32>) -> tensor<128x64x1x1xf32>
    %v1137 = stablehlo.transpose %v1136, dims = [1, 0, 2, 3] : (tensor<128x64x1x1xf32>) -> tensor<64x128x1x1xf32>
    %v1138 = stablehlo.constant dense<0.3> : tensor<64x128x1x1xf32>
    %v1139 = stablehlo.multiply %v1137, %v1138 : tensor<64x128x1x1xf32>
    %v1140 = stablehlo.subtract %Wp5, %v1139 : tensor<64x128x1x1xf32>
    %v1141 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1142 = stablehlo.reshape %v453 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v1143 = stablehlo.constant dense<196.0> : tensor<32x64x14x14xf32>
    %v1144 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v1145 = stablehlo.reduce(%v1142 init: %v1141) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v1146 = stablehlo.broadcast_in_dim %v1145, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v1147 = stablehlo.divide %v1146, %v1143 : tensor<32x64x14x14xf32>
    %v1148 = stablehlo.subtract %v1142, %v1147 : tensor<32x64x14x14xf32>
    %v1149 = stablehlo.multiply %v1148, %v1148 : tensor<32x64x14x14xf32>
    %v1150 = stablehlo.reduce(%v1149 init: %v1141) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v1151 = stablehlo.broadcast_in_dim %v1150, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v1152 = stablehlo.divide %v1151, %v1143 : tensor<32x64x14x14xf32>
    %v1153 = stablehlo.add %v1152, %v1144 : tensor<32x64x14x14xf32>
    %v1154 = stablehlo.rsqrt %v1153 : tensor<32x64x14x14xf32>
    %v1155 = stablehlo.multiply %v1148, %v1154 : tensor<32x64x14x14xf32>
    %v1156 = stablehlo.reshape %v823 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v1157 = stablehlo.multiply %v1156, %v1155 : tensor<32x64x14x14xf32>
    %v1158 = stablehlo.reduce(%v1157 init: %v1141) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v1159 = stablehlo.constant dense<0.3> : tensor<64xf32>
    %v1160 = stablehlo.multiply %v1158, %v1159 : tensor<64xf32>
    %v1161 = stablehlo.subtract %gp5, %v1160 : tensor<64xf32>
    %v1162 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1163 = stablehlo.reshape %v823 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v1164 = stablehlo.reduce(%v1163 init: %v1162) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v1165 = stablehlo.constant dense<0.3> : tensor<64xf32>
    %v1166 = stablehlo.multiply %v1164, %v1165 : tensor<64xf32>
    %v1167 = stablehlo.subtract %btp5, %v1166 : tensor<64xf32>
    %v1168 = stablehlo.reshape %v1057 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v1169 = stablehlo.reshape %v362 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v1170 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1171 = stablehlo.constant dense<784.0> : tensor<32x32x28x28xf32>
    %v1172 = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %v1173 = stablehlo.reduce(%v1169 init: %v1170) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v1174 = stablehlo.broadcast_in_dim %v1173, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v1175 = stablehlo.divide %v1174, %v1171 : tensor<32x32x28x28xf32>
    %v1176 = stablehlo.subtract %v1169, %v1175 : tensor<32x32x28x28xf32>
    %v1177 = stablehlo.multiply %v1176, %v1176 : tensor<32x32x28x28xf32>
    %v1178 = stablehlo.reduce(%v1177 init: %v1170) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v1179 = stablehlo.broadcast_in_dim %v1178, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v1180 = stablehlo.divide %v1179, %v1171 : tensor<32x32x28x28xf32>
    %v1181 = stablehlo.add %v1180, %v1172 : tensor<32x32x28x28xf32>
    %v1182 = stablehlo.rsqrt %v1181 : tensor<32x32x28x28xf32>
    %v1183 = stablehlo.multiply %v1176, %v1182 : tensor<32x32x28x28xf32>
    %v1184 = stablehlo.broadcast_in_dim %gp4, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v1185 = stablehlo.multiply %v1184, %v1168 : tensor<32x32x28x28xf32>
    %v1186 = stablehlo.reduce(%v1185 init: %v1170) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v1187 = stablehlo.broadcast_in_dim %v1186, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v1188 = stablehlo.multiply %v1183, %v1185 : tensor<32x32x28x28xf32>
    %v1189 = stablehlo.reduce(%v1188 init: %v1170) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v1190 = stablehlo.broadcast_in_dim %v1189, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v1191 = stablehlo.multiply %v1185, %v1171 : tensor<32x32x28x28xf32>
    %v1192 = stablehlo.subtract %v1191, %v1187 : tensor<32x32x28x28xf32>
    %v1193 = stablehlo.multiply %v1183, %v1190 : tensor<32x32x28x28xf32>
    %v1194 = stablehlo.subtract %v1192, %v1193 : tensor<32x32x28x28xf32>
    %v1195 = stablehlo.divide %v1182, %v1171 : tensor<32x32x28x28xf32>
    %v1196 = stablehlo.multiply %v1195, %v1194 : tensor<32x32x28x28xf32>
    %v1197 = stablehlo.reshape %v1196 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v1198 = stablehlo.reshape %v1197 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v1199 = stablehlo.transpose %Wp4, dims = [1, 0, 2, 3] : (tensor<32x128x1x1xf32>) -> tensor<128x32x1x1xf32>
    %v1200 = stablehlo.reverse %v1199, dims = [2, 3] : tensor<128x32x1x1xf32>
    %v1201 = stablehlo.convolution(%v1198, %v1200)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<128x32x1x1xf32>) -> tensor<32x128x28x28xf32>
    %v1202 = stablehlo.reshape %v1201 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v1203 = stablehlo.reshape %v1202 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v1204 = stablehlo.reshape %v351 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v1205 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v1206 = stablehlo.constant dense<6.0> : tensor<32x128x28x28xf32>
    %v1207 = stablehlo.compare GT, %v1204, %v1205 : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xi1>
    %v1208 = stablehlo.compare LT, %v1204, %v1206 : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xi1>
    %v1209 = stablehlo.and %v1207, %v1208 : tensor<32x128x28x28xi1>
    %v1210 = stablehlo.select %v1209, %v1203, %v1205 : tensor<32x128x28x28xi1>, tensor<32x128x28x28xf32>
    %v1211 = stablehlo.reshape %v1210 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v1212 = stablehlo.reshape %v1211 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v1213 = stablehlo.reshape %v331 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v1214 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1215 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v1216 = stablehlo.constant dense<1.0e-5> : tensor<32x128x28x28xf32>
    %v1217 = stablehlo.reduce(%v1213 init: %v1214) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v1218 = stablehlo.broadcast_in_dim %v1217, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v1219 = stablehlo.divide %v1218, %v1215 : tensor<32x128x28x28xf32>
    %v1220 = stablehlo.subtract %v1213, %v1219 : tensor<32x128x28x28xf32>
    %v1221 = stablehlo.multiply %v1220, %v1220 : tensor<32x128x28x28xf32>
    %v1222 = stablehlo.reduce(%v1221 init: %v1214) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v1223 = stablehlo.broadcast_in_dim %v1222, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v1224 = stablehlo.divide %v1223, %v1215 : tensor<32x128x28x28xf32>
    %v1225 = stablehlo.add %v1224, %v1216 : tensor<32x128x28x28xf32>
    %v1226 = stablehlo.rsqrt %v1225 : tensor<32x128x28x28xf32>
    %v1227 = stablehlo.multiply %v1220, %v1226 : tensor<32x128x28x28xf32>
    %v1228 = stablehlo.broadcast_in_dim %gd4, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v1229 = stablehlo.multiply %v1228, %v1212 : tensor<32x128x28x28xf32>
    %v1230 = stablehlo.reduce(%v1229 init: %v1214) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v1231 = stablehlo.broadcast_in_dim %v1230, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v1232 = stablehlo.multiply %v1227, %v1229 : tensor<32x128x28x28xf32>
    %v1233 = stablehlo.reduce(%v1232 init: %v1214) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v1234 = stablehlo.broadcast_in_dim %v1233, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v1235 = stablehlo.multiply %v1229, %v1215 : tensor<32x128x28x28xf32>
    %v1236 = stablehlo.subtract %v1235, %v1231 : tensor<32x128x28x28xf32>
    %v1237 = stablehlo.multiply %v1227, %v1234 : tensor<32x128x28x28xf32>
    %v1238 = stablehlo.subtract %v1236, %v1237 : tensor<32x128x28x28xf32>
    %v1239 = stablehlo.divide %v1226, %v1215 : tensor<32x128x28x28xf32>
    %v1240 = stablehlo.multiply %v1239, %v1238 : tensor<32x128x28x28xf32>
    %v1241 = stablehlo.reshape %v1240 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v1242 = stablehlo.reshape %v1241 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v1243 = stablehlo.reverse %Wd4, dims = [2, 3] : tensor<128x1x3x3xf32>
    %v1244 = stablehlo.convolution(%v1242, %v1243)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 128 : i64} : (tensor<32x128x28x28xf32>, tensor<128x1x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v1245 = stablehlo.reshape %v1244 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v1246 = stablehlo.reshape %v1245 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v1247 = stablehlo.reshape %v320 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v1248 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v1249 = stablehlo.constant dense<6.0> : tensor<32x128x28x28xf32>
    %v1250 = stablehlo.compare GT, %v1247, %v1248 : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xi1>
    %v1251 = stablehlo.compare LT, %v1247, %v1249 : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xi1>
    %v1252 = stablehlo.and %v1250, %v1251 : tensor<32x128x28x28xi1>
    %v1253 = stablehlo.select %v1252, %v1246, %v1248 : tensor<32x128x28x28xi1>, tensor<32x128x28x28xf32>
    %v1254 = stablehlo.reshape %v1253 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v1255 = stablehlo.reshape %v1254 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v1256 = stablehlo.reshape %v300 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v1257 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1258 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v1259 = stablehlo.constant dense<1.0e-5> : tensor<32x128x28x28xf32>
    %v1260 = stablehlo.reduce(%v1256 init: %v1257) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v1261 = stablehlo.broadcast_in_dim %v1260, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v1262 = stablehlo.divide %v1261, %v1258 : tensor<32x128x28x28xf32>
    %v1263 = stablehlo.subtract %v1256, %v1262 : tensor<32x128x28x28xf32>
    %v1264 = stablehlo.multiply %v1263, %v1263 : tensor<32x128x28x28xf32>
    %v1265 = stablehlo.reduce(%v1264 init: %v1257) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v1266 = stablehlo.broadcast_in_dim %v1265, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v1267 = stablehlo.divide %v1266, %v1258 : tensor<32x128x28x28xf32>
    %v1268 = stablehlo.add %v1267, %v1259 : tensor<32x128x28x28xf32>
    %v1269 = stablehlo.rsqrt %v1268 : tensor<32x128x28x28xf32>
    %v1270 = stablehlo.multiply %v1263, %v1269 : tensor<32x128x28x28xf32>
    %v1271 = stablehlo.broadcast_in_dim %ge4, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v1272 = stablehlo.multiply %v1271, %v1255 : tensor<32x128x28x28xf32>
    %v1273 = stablehlo.reduce(%v1272 init: %v1257) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v1274 = stablehlo.broadcast_in_dim %v1273, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v1275 = stablehlo.multiply %v1270, %v1272 : tensor<32x128x28x28xf32>
    %v1276 = stablehlo.reduce(%v1275 init: %v1257) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v1277 = stablehlo.broadcast_in_dim %v1276, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v1278 = stablehlo.multiply %v1272, %v1258 : tensor<32x128x28x28xf32>
    %v1279 = stablehlo.subtract %v1278, %v1274 : tensor<32x128x28x28xf32>
    %v1280 = stablehlo.multiply %v1270, %v1277 : tensor<32x128x28x28xf32>
    %v1281 = stablehlo.subtract %v1279, %v1280 : tensor<32x128x28x28xf32>
    %v1282 = stablehlo.divide %v1269, %v1258 : tensor<32x128x28x28xf32>
    %v1283 = stablehlo.multiply %v1282, %v1281 : tensor<32x128x28x28xf32>
    %v1284 = stablehlo.reshape %v1283 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v1285 = stablehlo.reshape %v1284 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v1286 = stablehlo.transpose %We4, dims = [1, 0, 2, 3] : (tensor<128x32x1x1xf32>) -> tensor<32x128x1x1xf32>
    %v1287 = stablehlo.reverse %v1286, dims = [2, 3] : tensor<32x128x1x1xf32>
    %v1288 = stablehlo.convolution(%v1285, %v1287)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<32x128x1x1xf32>) -> tensor<32x32x28x28xf32>
    %v1289 = stablehlo.reshape %v1288 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v1290 = stablehlo.reshape %v1289 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v1291 = stablehlo.reshape %v1057 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v1292 = stablehlo.add %v1290, %v1291 : tensor<32x32x28x28xf32>
    %v1293 = stablehlo.reshape %v1292 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v1294 = stablehlo.reshape %v295 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v1295 = stablehlo.reshape %v1284 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v1296 = stablehlo.transpose %v1294, dims = [1, 0, 2, 3] : (tensor<32x32x28x28xf32>) -> tensor<32x32x28x28xf32>
    %v1297 = stablehlo.transpose %v1295, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v1298 = stablehlo.convolution(%v1296, %v1297)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<32x128x1x1xf32>
    %v1299 = stablehlo.transpose %v1298, dims = [1, 0, 2, 3] : (tensor<32x128x1x1xf32>) -> tensor<128x32x1x1xf32>
    %v1300 = stablehlo.constant dense<0.3> : tensor<128x32x1x1xf32>
    %v1301 = stablehlo.multiply %v1299, %v1300 : tensor<128x32x1x1xf32>
    %v1302 = stablehlo.subtract %We4, %v1301 : tensor<128x32x1x1xf32>
    %v1303 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1304 = stablehlo.reshape %v300 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v1305 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v1306 = stablehlo.constant dense<1.0e-5> : tensor<32x128x28x28xf32>
    %v1307 = stablehlo.reduce(%v1304 init: %v1303) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v1308 = stablehlo.broadcast_in_dim %v1307, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v1309 = stablehlo.divide %v1308, %v1305 : tensor<32x128x28x28xf32>
    %v1310 = stablehlo.subtract %v1304, %v1309 : tensor<32x128x28x28xf32>
    %v1311 = stablehlo.multiply %v1310, %v1310 : tensor<32x128x28x28xf32>
    %v1312 = stablehlo.reduce(%v1311 init: %v1303) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v1313 = stablehlo.broadcast_in_dim %v1312, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v1314 = stablehlo.divide %v1313, %v1305 : tensor<32x128x28x28xf32>
    %v1315 = stablehlo.add %v1314, %v1306 : tensor<32x128x28x28xf32>
    %v1316 = stablehlo.rsqrt %v1315 : tensor<32x128x28x28xf32>
    %v1317 = stablehlo.multiply %v1310, %v1316 : tensor<32x128x28x28xf32>
    %v1318 = stablehlo.reshape %v1254 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v1319 = stablehlo.multiply %v1318, %v1317 : tensor<32x128x28x28xf32>
    %v1320 = stablehlo.reduce(%v1319 init: %v1303) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v1321 = stablehlo.constant dense<0.3> : tensor<128xf32>
    %v1322 = stablehlo.multiply %v1320, %v1321 : tensor<128xf32>
    %v1323 = stablehlo.subtract %ge4, %v1322 : tensor<128xf32>
    %v1324 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1325 = stablehlo.reshape %v1254 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v1326 = stablehlo.reduce(%v1325 init: %v1324) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v1327 = stablehlo.constant dense<0.3> : tensor<128xf32>
    %v1328 = stablehlo.multiply %v1326, %v1327 : tensor<128xf32>
    %v1329 = stablehlo.subtract %bte4, %v1328 : tensor<128xf32>
    %v1330 = stablehlo.reshape %v326 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v1331 = stablehlo.reshape %v1241 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v1332 = stablehlo.transpose %v1330, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v1333 = stablehlo.transpose %v1331, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v1334 = stablehlo.convolution(%v1332, %v1333)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 128 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<1x128x3x3xf32>
    %v1335 = stablehlo.reshape %v1334 : (tensor<1x128x3x3xf32>) -> tensor<128x1x3x3xf32>
    %v1336 = stablehlo.constant dense<0.3> : tensor<128x1x3x3xf32>
    %v1337 = stablehlo.multiply %v1335, %v1336 : tensor<128x1x3x3xf32>
    %v1338 = stablehlo.subtract %Wd4, %v1337 : tensor<128x1x3x3xf32>
    %v1339 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1340 = stablehlo.reshape %v331 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v1341 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v1342 = stablehlo.constant dense<1.0e-5> : tensor<32x128x28x28xf32>
    %v1343 = stablehlo.reduce(%v1340 init: %v1339) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v1344 = stablehlo.broadcast_in_dim %v1343, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v1345 = stablehlo.divide %v1344, %v1341 : tensor<32x128x28x28xf32>
    %v1346 = stablehlo.subtract %v1340, %v1345 : tensor<32x128x28x28xf32>
    %v1347 = stablehlo.multiply %v1346, %v1346 : tensor<32x128x28x28xf32>
    %v1348 = stablehlo.reduce(%v1347 init: %v1339) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v1349 = stablehlo.broadcast_in_dim %v1348, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v1350 = stablehlo.divide %v1349, %v1341 : tensor<32x128x28x28xf32>
    %v1351 = stablehlo.add %v1350, %v1342 : tensor<32x128x28x28xf32>
    %v1352 = stablehlo.rsqrt %v1351 : tensor<32x128x28x28xf32>
    %v1353 = stablehlo.multiply %v1346, %v1352 : tensor<32x128x28x28xf32>
    %v1354 = stablehlo.reshape %v1211 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v1355 = stablehlo.multiply %v1354, %v1353 : tensor<32x128x28x28xf32>
    %v1356 = stablehlo.reduce(%v1355 init: %v1339) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v1357 = stablehlo.constant dense<0.3> : tensor<128xf32>
    %v1358 = stablehlo.multiply %v1356, %v1357 : tensor<128xf32>
    %v1359 = stablehlo.subtract %gd4, %v1358 : tensor<128xf32>
    %v1360 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1361 = stablehlo.reshape %v1211 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v1362 = stablehlo.reduce(%v1361 init: %v1360) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v1363 = stablehlo.constant dense<0.3> : tensor<128xf32>
    %v1364 = stablehlo.multiply %v1362, %v1363 : tensor<128xf32>
    %v1365 = stablehlo.subtract %btd4, %v1364 : tensor<128xf32>
    %v1366 = stablehlo.reshape %v357 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v1367 = stablehlo.reshape %v1197 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v1368 = stablehlo.transpose %v1366, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v1369 = stablehlo.transpose %v1367, dims = [1, 0, 2, 3] : (tensor<32x32x28x28xf32>) -> tensor<32x32x28x28xf32>
    %v1370 = stablehlo.convolution(%v1368, %v1369)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<32x32x28x28xf32>) -> tensor<128x32x1x1xf32>
    %v1371 = stablehlo.transpose %v1370, dims = [1, 0, 2, 3] : (tensor<128x32x1x1xf32>) -> tensor<32x128x1x1xf32>
    %v1372 = stablehlo.constant dense<0.3> : tensor<32x128x1x1xf32>
    %v1373 = stablehlo.multiply %v1371, %v1372 : tensor<32x128x1x1xf32>
    %v1374 = stablehlo.subtract %Wp4, %v1373 : tensor<32x128x1x1xf32>
    %v1375 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1376 = stablehlo.reshape %v362 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v1377 = stablehlo.constant dense<784.0> : tensor<32x32x28x28xf32>
    %v1378 = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %v1379 = stablehlo.reduce(%v1376 init: %v1375) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v1380 = stablehlo.broadcast_in_dim %v1379, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v1381 = stablehlo.divide %v1380, %v1377 : tensor<32x32x28x28xf32>
    %v1382 = stablehlo.subtract %v1376, %v1381 : tensor<32x32x28x28xf32>
    %v1383 = stablehlo.multiply %v1382, %v1382 : tensor<32x32x28x28xf32>
    %v1384 = stablehlo.reduce(%v1383 init: %v1375) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v1385 = stablehlo.broadcast_in_dim %v1384, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v1386 = stablehlo.divide %v1385, %v1377 : tensor<32x32x28x28xf32>
    %v1387 = stablehlo.add %v1386, %v1378 : tensor<32x32x28x28xf32>
    %v1388 = stablehlo.rsqrt %v1387 : tensor<32x32x28x28xf32>
    %v1389 = stablehlo.multiply %v1382, %v1388 : tensor<32x32x28x28xf32>
    %v1390 = stablehlo.reshape %v1057 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v1391 = stablehlo.multiply %v1390, %v1389 : tensor<32x32x28x28xf32>
    %v1392 = stablehlo.reduce(%v1391 init: %v1375) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v1393 = stablehlo.constant dense<0.3> : tensor<32xf32>
    %v1394 = stablehlo.multiply %v1392, %v1393 : tensor<32xf32>
    %v1395 = stablehlo.subtract %gp4, %v1394 : tensor<32xf32>
    %v1396 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1397 = stablehlo.reshape %v1057 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v1398 = stablehlo.reduce(%v1397 init: %v1396) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v1399 = stablehlo.constant dense<0.3> : tensor<32xf32>
    %v1400 = stablehlo.multiply %v1398, %v1399 : tensor<32xf32>
    %v1401 = stablehlo.subtract %btp4, %v1400 : tensor<32xf32>
    %v1402 = stablehlo.reshape %v1293 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v1403 = stablehlo.reshape %v275 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v1404 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1405 = stablehlo.constant dense<784.0> : tensor<32x32x28x28xf32>
    %v1406 = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %v1407 = stablehlo.reduce(%v1403 init: %v1404) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v1408 = stablehlo.broadcast_in_dim %v1407, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v1409 = stablehlo.divide %v1408, %v1405 : tensor<32x32x28x28xf32>
    %v1410 = stablehlo.subtract %v1403, %v1409 : tensor<32x32x28x28xf32>
    %v1411 = stablehlo.multiply %v1410, %v1410 : tensor<32x32x28x28xf32>
    %v1412 = stablehlo.reduce(%v1411 init: %v1404) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v1413 = stablehlo.broadcast_in_dim %v1412, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v1414 = stablehlo.divide %v1413, %v1405 : tensor<32x32x28x28xf32>
    %v1415 = stablehlo.add %v1414, %v1406 : tensor<32x32x28x28xf32>
    %v1416 = stablehlo.rsqrt %v1415 : tensor<32x32x28x28xf32>
    %v1417 = stablehlo.multiply %v1410, %v1416 : tensor<32x32x28x28xf32>
    %v1418 = stablehlo.broadcast_in_dim %gp3, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v1419 = stablehlo.multiply %v1418, %v1402 : tensor<32x32x28x28xf32>
    %v1420 = stablehlo.reduce(%v1419 init: %v1404) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v1421 = stablehlo.broadcast_in_dim %v1420, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v1422 = stablehlo.multiply %v1417, %v1419 : tensor<32x32x28x28xf32>
    %v1423 = stablehlo.reduce(%v1422 init: %v1404) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v1424 = stablehlo.broadcast_in_dim %v1423, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v1425 = stablehlo.multiply %v1419, %v1405 : tensor<32x32x28x28xf32>
    %v1426 = stablehlo.subtract %v1425, %v1421 : tensor<32x32x28x28xf32>
    %v1427 = stablehlo.multiply %v1417, %v1424 : tensor<32x32x28x28xf32>
    %v1428 = stablehlo.subtract %v1426, %v1427 : tensor<32x32x28x28xf32>
    %v1429 = stablehlo.divide %v1416, %v1405 : tensor<32x32x28x28xf32>
    %v1430 = stablehlo.multiply %v1429, %v1428 : tensor<32x32x28x28xf32>
    %v1431 = stablehlo.reshape %v1430 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v1432 = stablehlo.reshape %v1431 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v1433 = stablehlo.transpose %Wp3, dims = [1, 0, 2, 3] : (tensor<32x96x1x1xf32>) -> tensor<96x32x1x1xf32>
    %v1434 = stablehlo.reverse %v1433, dims = [2, 3] : tensor<96x32x1x1xf32>
    %v1435 = stablehlo.convolution(%v1432, %v1434)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<96x32x1x1xf32>) -> tensor<32x96x28x28xf32>
    %v1436 = stablehlo.reshape %v1435 : (tensor<32x96x28x28xf32>) -> tensor<32x75264xf32>
    %v1437 = stablehlo.reshape %v1436 : (tensor<32x75264xf32>) -> tensor<32x96x28x28xf32>
    %v1438 = stablehlo.reshape %v264 : (tensor<32x75264xf32>) -> tensor<32x96x28x28xf32>
    %v1439 = stablehlo.constant dense<0.0> : tensor<32x96x28x28xf32>
    %v1440 = stablehlo.constant dense<6.0> : tensor<32x96x28x28xf32>
    %v1441 = stablehlo.compare GT, %v1438, %v1439 : (tensor<32x96x28x28xf32>, tensor<32x96x28x28xf32>) -> tensor<32x96x28x28xi1>
    %v1442 = stablehlo.compare LT, %v1438, %v1440 : (tensor<32x96x28x28xf32>, tensor<32x96x28x28xf32>) -> tensor<32x96x28x28xi1>
    %v1443 = stablehlo.and %v1441, %v1442 : tensor<32x96x28x28xi1>
    %v1444 = stablehlo.select %v1443, %v1437, %v1439 : tensor<32x96x28x28xi1>, tensor<32x96x28x28xf32>
    %v1445 = stablehlo.reshape %v1444 : (tensor<32x96x28x28xf32>) -> tensor<32x75264xf32>
    %v1446 = stablehlo.reshape %v1445 : (tensor<32x75264xf32>) -> tensor<32x96x28x28xf32>
    %v1447 = stablehlo.reshape %v244 : (tensor<32x75264xf32>) -> tensor<32x96x28x28xf32>
    %v1448 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1449 = stablehlo.constant dense<784.0> : tensor<32x96x28x28xf32>
    %v1450 = stablehlo.constant dense<1.0e-5> : tensor<32x96x28x28xf32>
    %v1451 = stablehlo.reduce(%v1447 init: %v1448) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x28x28xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1452 = stablehlo.broadcast_in_dim %v1451, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x28x28xf32>
    %v1453 = stablehlo.divide %v1452, %v1449 : tensor<32x96x28x28xf32>
    %v1454 = stablehlo.subtract %v1447, %v1453 : tensor<32x96x28x28xf32>
    %v1455 = stablehlo.multiply %v1454, %v1454 : tensor<32x96x28x28xf32>
    %v1456 = stablehlo.reduce(%v1455 init: %v1448) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x28x28xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1457 = stablehlo.broadcast_in_dim %v1456, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x28x28xf32>
    %v1458 = stablehlo.divide %v1457, %v1449 : tensor<32x96x28x28xf32>
    %v1459 = stablehlo.add %v1458, %v1450 : tensor<32x96x28x28xf32>
    %v1460 = stablehlo.rsqrt %v1459 : tensor<32x96x28x28xf32>
    %v1461 = stablehlo.multiply %v1454, %v1460 : tensor<32x96x28x28xf32>
    %v1462 = stablehlo.broadcast_in_dim %gd3, dims = [1] : (tensor<96xf32>) -> tensor<32x96x28x28xf32>
    %v1463 = stablehlo.multiply %v1462, %v1446 : tensor<32x96x28x28xf32>
    %v1464 = stablehlo.reduce(%v1463 init: %v1448) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x28x28xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1465 = stablehlo.broadcast_in_dim %v1464, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x28x28xf32>
    %v1466 = stablehlo.multiply %v1461, %v1463 : tensor<32x96x28x28xf32>
    %v1467 = stablehlo.reduce(%v1466 init: %v1448) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x28x28xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1468 = stablehlo.broadcast_in_dim %v1467, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x28x28xf32>
    %v1469 = stablehlo.multiply %v1463, %v1449 : tensor<32x96x28x28xf32>
    %v1470 = stablehlo.subtract %v1469, %v1465 : tensor<32x96x28x28xf32>
    %v1471 = stablehlo.multiply %v1461, %v1468 : tensor<32x96x28x28xf32>
    %v1472 = stablehlo.subtract %v1470, %v1471 : tensor<32x96x28x28xf32>
    %v1473 = stablehlo.divide %v1460, %v1449 : tensor<32x96x28x28xf32>
    %v1474 = stablehlo.multiply %v1473, %v1472 : tensor<32x96x28x28xf32>
    %v1475 = stablehlo.reshape %v1474 : (tensor<32x96x28x28xf32>) -> tensor<32x75264xf32>
    %v1476 = stablehlo.reshape %v1475 : (tensor<32x75264xf32>) -> tensor<32x96x28x28xf32>
    %v1477 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1478 = stablehlo.pad %v1476, %v1477, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x96x28x28xf32>, tensor<f32>) -> tensor<32x96x56x56xf32>
    %v1479 = stablehlo.reverse %Wd3, dims = [2, 3] : tensor<96x1x3x3xf32>
    %v1480 = stablehlo.convolution(%v1478, %v1479)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x3x3xf32>) -> tensor<32x96x56x56xf32>
    %v1481 = stablehlo.reshape %v1480 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v1482 = stablehlo.reshape %v1481 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1483 = stablehlo.reshape %v233 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1484 = stablehlo.constant dense<0.0> : tensor<32x96x56x56xf32>
    %v1485 = stablehlo.constant dense<6.0> : tensor<32x96x56x56xf32>
    %v1486 = stablehlo.compare GT, %v1483, %v1484 : (tensor<32x96x56x56xf32>, tensor<32x96x56x56xf32>) -> tensor<32x96x56x56xi1>
    %v1487 = stablehlo.compare LT, %v1483, %v1485 : (tensor<32x96x56x56xf32>, tensor<32x96x56x56xf32>) -> tensor<32x96x56x56xi1>
    %v1488 = stablehlo.and %v1486, %v1487 : tensor<32x96x56x56xi1>
    %v1489 = stablehlo.select %v1488, %v1482, %v1484 : tensor<32x96x56x56xi1>, tensor<32x96x56x56xf32>
    %v1490 = stablehlo.reshape %v1489 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v1491 = stablehlo.reshape %v1490 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1492 = stablehlo.reshape %v213 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1493 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1494 = stablehlo.constant dense<3136.0> : tensor<32x96x56x56xf32>
    %v1495 = stablehlo.constant dense<1.0e-5> : tensor<32x96x56x56xf32>
    %v1496 = stablehlo.reduce(%v1492 init: %v1493) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1497 = stablehlo.broadcast_in_dim %v1496, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v1498 = stablehlo.divide %v1497, %v1494 : tensor<32x96x56x56xf32>
    %v1499 = stablehlo.subtract %v1492, %v1498 : tensor<32x96x56x56xf32>
    %v1500 = stablehlo.multiply %v1499, %v1499 : tensor<32x96x56x56xf32>
    %v1501 = stablehlo.reduce(%v1500 init: %v1493) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1502 = stablehlo.broadcast_in_dim %v1501, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v1503 = stablehlo.divide %v1502, %v1494 : tensor<32x96x56x56xf32>
    %v1504 = stablehlo.add %v1503, %v1495 : tensor<32x96x56x56xf32>
    %v1505 = stablehlo.rsqrt %v1504 : tensor<32x96x56x56xf32>
    %v1506 = stablehlo.multiply %v1499, %v1505 : tensor<32x96x56x56xf32>
    %v1507 = stablehlo.broadcast_in_dim %ge3, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v1508 = stablehlo.multiply %v1507, %v1491 : tensor<32x96x56x56xf32>
    %v1509 = stablehlo.reduce(%v1508 init: %v1493) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1510 = stablehlo.broadcast_in_dim %v1509, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v1511 = stablehlo.multiply %v1506, %v1508 : tensor<32x96x56x56xf32>
    %v1512 = stablehlo.reduce(%v1511 init: %v1493) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1513 = stablehlo.broadcast_in_dim %v1512, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v1514 = stablehlo.multiply %v1508, %v1494 : tensor<32x96x56x56xf32>
    %v1515 = stablehlo.subtract %v1514, %v1510 : tensor<32x96x56x56xf32>
    %v1516 = stablehlo.multiply %v1506, %v1513 : tensor<32x96x56x56xf32>
    %v1517 = stablehlo.subtract %v1515, %v1516 : tensor<32x96x56x56xf32>
    %v1518 = stablehlo.divide %v1505, %v1494 : tensor<32x96x56x56xf32>
    %v1519 = stablehlo.multiply %v1518, %v1517 : tensor<32x96x56x56xf32>
    %v1520 = stablehlo.reshape %v1519 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v1521 = stablehlo.reshape %v1520 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1522 = stablehlo.transpose %We3, dims = [1, 0, 2, 3] : (tensor<96x24x1x1xf32>) -> tensor<24x96x1x1xf32>
    %v1523 = stablehlo.reverse %v1522, dims = [2, 3] : tensor<24x96x1x1xf32>
    %v1524 = stablehlo.convolution(%v1521, %v1523)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<24x96x1x1xf32>) -> tensor<32x24x56x56xf32>
    %v1525 = stablehlo.reshape %v1524 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v1526 = stablehlo.reshape %v208 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v1527 = stablehlo.reshape %v1520 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1528 = stablehlo.transpose %v1526, dims = [1, 0, 2, 3] : (tensor<32x24x56x56xf32>) -> tensor<24x32x56x56xf32>
    %v1529 = stablehlo.transpose %v1527, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v1530 = stablehlo.convolution(%v1528, %v1529)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<24x32x56x56xf32>, tensor<96x32x56x56xf32>) -> tensor<24x96x1x1xf32>
    %v1531 = stablehlo.transpose %v1530, dims = [1, 0, 2, 3] : (tensor<24x96x1x1xf32>) -> tensor<96x24x1x1xf32>
    %v1532 = stablehlo.constant dense<0.3> : tensor<96x24x1x1xf32>
    %v1533 = stablehlo.multiply %v1531, %v1532 : tensor<96x24x1x1xf32>
    %v1534 = stablehlo.subtract %We3, %v1533 : tensor<96x24x1x1xf32>
    %v1535 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1536 = stablehlo.reshape %v213 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1537 = stablehlo.constant dense<3136.0> : tensor<32x96x56x56xf32>
    %v1538 = stablehlo.constant dense<1.0e-5> : tensor<32x96x56x56xf32>
    %v1539 = stablehlo.reduce(%v1536 init: %v1535) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1540 = stablehlo.broadcast_in_dim %v1539, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v1541 = stablehlo.divide %v1540, %v1537 : tensor<32x96x56x56xf32>
    %v1542 = stablehlo.subtract %v1536, %v1541 : tensor<32x96x56x56xf32>
    %v1543 = stablehlo.multiply %v1542, %v1542 : tensor<32x96x56x56xf32>
    %v1544 = stablehlo.reduce(%v1543 init: %v1535) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1545 = stablehlo.broadcast_in_dim %v1544, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v1546 = stablehlo.divide %v1545, %v1537 : tensor<32x96x56x56xf32>
    %v1547 = stablehlo.add %v1546, %v1538 : tensor<32x96x56x56xf32>
    %v1548 = stablehlo.rsqrt %v1547 : tensor<32x96x56x56xf32>
    %v1549 = stablehlo.multiply %v1542, %v1548 : tensor<32x96x56x56xf32>
    %v1550 = stablehlo.reshape %v1490 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1551 = stablehlo.multiply %v1550, %v1549 : tensor<32x96x56x56xf32>
    %v1552 = stablehlo.reduce(%v1551 init: %v1535) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v1553 = stablehlo.constant dense<0.3> : tensor<96xf32>
    %v1554 = stablehlo.multiply %v1552, %v1553 : tensor<96xf32>
    %v1555 = stablehlo.subtract %ge3, %v1554 : tensor<96xf32>
    %v1556 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1557 = stablehlo.reshape %v1490 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1558 = stablehlo.reduce(%v1557 init: %v1556) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v1559 = stablehlo.constant dense<0.3> : tensor<96xf32>
    %v1560 = stablehlo.multiply %v1558, %v1559 : tensor<96xf32>
    %v1561 = stablehlo.subtract %bte3, %v1560 : tensor<96xf32>
    %v1562 = stablehlo.reshape %v239 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1563 = stablehlo.reshape %v1475 : (tensor<32x75264xf32>) -> tensor<32x96x28x28xf32>
    %v1564 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1565 = stablehlo.pad %v1563, %v1564, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x96x28x28xf32>, tensor<f32>) -> tensor<32x96x56x56xf32>
    %v1566 = stablehlo.transpose %v1562, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v1567 = stablehlo.transpose %v1565, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v1568 = stablehlo.convolution(%v1566, %v1567)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 96 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<96x32x56x56xf32>) -> tensor<1x96x3x3xf32>
    %v1569 = stablehlo.reshape %v1568 : (tensor<1x96x3x3xf32>) -> tensor<96x1x3x3xf32>
    %v1570 = stablehlo.constant dense<0.3> : tensor<96x1x3x3xf32>
    %v1571 = stablehlo.multiply %v1569, %v1570 : tensor<96x1x3x3xf32>
    %v1572 = stablehlo.subtract %Wd3, %v1571 : tensor<96x1x3x3xf32>
    %v1573 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1574 = stablehlo.reshape %v244 : (tensor<32x75264xf32>) -> tensor<32x96x28x28xf32>
    %v1575 = stablehlo.constant dense<784.0> : tensor<32x96x28x28xf32>
    %v1576 = stablehlo.constant dense<1.0e-5> : tensor<32x96x28x28xf32>
    %v1577 = stablehlo.reduce(%v1574 init: %v1573) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x28x28xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1578 = stablehlo.broadcast_in_dim %v1577, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x28x28xf32>
    %v1579 = stablehlo.divide %v1578, %v1575 : tensor<32x96x28x28xf32>
    %v1580 = stablehlo.subtract %v1574, %v1579 : tensor<32x96x28x28xf32>
    %v1581 = stablehlo.multiply %v1580, %v1580 : tensor<32x96x28x28xf32>
    %v1582 = stablehlo.reduce(%v1581 init: %v1573) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x28x28xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1583 = stablehlo.broadcast_in_dim %v1582, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x28x28xf32>
    %v1584 = stablehlo.divide %v1583, %v1575 : tensor<32x96x28x28xf32>
    %v1585 = stablehlo.add %v1584, %v1576 : tensor<32x96x28x28xf32>
    %v1586 = stablehlo.rsqrt %v1585 : tensor<32x96x28x28xf32>
    %v1587 = stablehlo.multiply %v1580, %v1586 : tensor<32x96x28x28xf32>
    %v1588 = stablehlo.reshape %v1445 : (tensor<32x75264xf32>) -> tensor<32x96x28x28xf32>
    %v1589 = stablehlo.multiply %v1588, %v1587 : tensor<32x96x28x28xf32>
    %v1590 = stablehlo.reduce(%v1589 init: %v1573) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x28x28xf32>, tensor<f32>) -> tensor<96xf32>
    %v1591 = stablehlo.constant dense<0.3> : tensor<96xf32>
    %v1592 = stablehlo.multiply %v1590, %v1591 : tensor<96xf32>
    %v1593 = stablehlo.subtract %gd3, %v1592 : tensor<96xf32>
    %v1594 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1595 = stablehlo.reshape %v1445 : (tensor<32x75264xf32>) -> tensor<32x96x28x28xf32>
    %v1596 = stablehlo.reduce(%v1595 init: %v1594) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x28x28xf32>, tensor<f32>) -> tensor<96xf32>
    %v1597 = stablehlo.constant dense<0.3> : tensor<96xf32>
    %v1598 = stablehlo.multiply %v1596, %v1597 : tensor<96xf32>
    %v1599 = stablehlo.subtract %btd3, %v1598 : tensor<96xf32>
    %v1600 = stablehlo.reshape %v270 : (tensor<32x75264xf32>) -> tensor<32x96x28x28xf32>
    %v1601 = stablehlo.reshape %v1431 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v1602 = stablehlo.transpose %v1600, dims = [1, 0, 2, 3] : (tensor<32x96x28x28xf32>) -> tensor<96x32x28x28xf32>
    %v1603 = stablehlo.transpose %v1601, dims = [1, 0, 2, 3] : (tensor<32x32x28x28xf32>) -> tensor<32x32x28x28xf32>
    %v1604 = stablehlo.convolution(%v1602, %v1603)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x28x28xf32>, tensor<32x32x28x28xf32>) -> tensor<96x32x1x1xf32>
    %v1605 = stablehlo.transpose %v1604, dims = [1, 0, 2, 3] : (tensor<96x32x1x1xf32>) -> tensor<32x96x1x1xf32>
    %v1606 = stablehlo.constant dense<0.3> : tensor<32x96x1x1xf32>
    %v1607 = stablehlo.multiply %v1605, %v1606 : tensor<32x96x1x1xf32>
    %v1608 = stablehlo.subtract %Wp3, %v1607 : tensor<32x96x1x1xf32>
    %v1609 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1610 = stablehlo.reshape %v275 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v1611 = stablehlo.constant dense<784.0> : tensor<32x32x28x28xf32>
    %v1612 = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %v1613 = stablehlo.reduce(%v1610 init: %v1609) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v1614 = stablehlo.broadcast_in_dim %v1613, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v1615 = stablehlo.divide %v1614, %v1611 : tensor<32x32x28x28xf32>
    %v1616 = stablehlo.subtract %v1610, %v1615 : tensor<32x32x28x28xf32>
    %v1617 = stablehlo.multiply %v1616, %v1616 : tensor<32x32x28x28xf32>
    %v1618 = stablehlo.reduce(%v1617 init: %v1609) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v1619 = stablehlo.broadcast_in_dim %v1618, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v1620 = stablehlo.divide %v1619, %v1611 : tensor<32x32x28x28xf32>
    %v1621 = stablehlo.add %v1620, %v1612 : tensor<32x32x28x28xf32>
    %v1622 = stablehlo.rsqrt %v1621 : tensor<32x32x28x28xf32>
    %v1623 = stablehlo.multiply %v1616, %v1622 : tensor<32x32x28x28xf32>
    %v1624 = stablehlo.reshape %v1293 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v1625 = stablehlo.multiply %v1624, %v1623 : tensor<32x32x28x28xf32>
    %v1626 = stablehlo.reduce(%v1625 init: %v1609) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v1627 = stablehlo.constant dense<0.3> : tensor<32xf32>
    %v1628 = stablehlo.multiply %v1626, %v1627 : tensor<32xf32>
    %v1629 = stablehlo.subtract %gp3, %v1628 : tensor<32xf32>
    %v1630 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1631 = stablehlo.reshape %v1293 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v1632 = stablehlo.reduce(%v1631 init: %v1630) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v1633 = stablehlo.constant dense<0.3> : tensor<32xf32>
    %v1634 = stablehlo.multiply %v1632, %v1633 : tensor<32xf32>
    %v1635 = stablehlo.subtract %btp3, %v1634 : tensor<32xf32>
    %v1636 = stablehlo.reshape %v1525 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v1637 = stablehlo.reshape %v184 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v1638 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1639 = stablehlo.constant dense<3136.0> : tensor<32x24x56x56xf32>
    %v1640 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v1641 = stablehlo.reduce(%v1637 init: %v1638) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v1642 = stablehlo.broadcast_in_dim %v1641, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v1643 = stablehlo.divide %v1642, %v1639 : tensor<32x24x56x56xf32>
    %v1644 = stablehlo.subtract %v1637, %v1643 : tensor<32x24x56x56xf32>
    %v1645 = stablehlo.multiply %v1644, %v1644 : tensor<32x24x56x56xf32>
    %v1646 = stablehlo.reduce(%v1645 init: %v1638) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v1647 = stablehlo.broadcast_in_dim %v1646, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v1648 = stablehlo.divide %v1647, %v1639 : tensor<32x24x56x56xf32>
    %v1649 = stablehlo.add %v1648, %v1640 : tensor<32x24x56x56xf32>
    %v1650 = stablehlo.rsqrt %v1649 : tensor<32x24x56x56xf32>
    %v1651 = stablehlo.multiply %v1644, %v1650 : tensor<32x24x56x56xf32>
    %v1652 = stablehlo.broadcast_in_dim %gp2, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v1653 = stablehlo.multiply %v1652, %v1636 : tensor<32x24x56x56xf32>
    %v1654 = stablehlo.reduce(%v1653 init: %v1638) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v1655 = stablehlo.broadcast_in_dim %v1654, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v1656 = stablehlo.multiply %v1651, %v1653 : tensor<32x24x56x56xf32>
    %v1657 = stablehlo.reduce(%v1656 init: %v1638) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v1658 = stablehlo.broadcast_in_dim %v1657, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v1659 = stablehlo.multiply %v1653, %v1639 : tensor<32x24x56x56xf32>
    %v1660 = stablehlo.subtract %v1659, %v1655 : tensor<32x24x56x56xf32>
    %v1661 = stablehlo.multiply %v1651, %v1658 : tensor<32x24x56x56xf32>
    %v1662 = stablehlo.subtract %v1660, %v1661 : tensor<32x24x56x56xf32>
    %v1663 = stablehlo.divide %v1650, %v1639 : tensor<32x24x56x56xf32>
    %v1664 = stablehlo.multiply %v1663, %v1662 : tensor<32x24x56x56xf32>
    %v1665 = stablehlo.reshape %v1664 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v1666 = stablehlo.reshape %v1665 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v1667 = stablehlo.transpose %Wp2, dims = [1, 0, 2, 3] : (tensor<24x96x1x1xf32>) -> tensor<96x24x1x1xf32>
    %v1668 = stablehlo.reverse %v1667, dims = [2, 3] : tensor<96x24x1x1xf32>
    %v1669 = stablehlo.convolution(%v1666, %v1668)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<96x24x1x1xf32>) -> tensor<32x96x56x56xf32>
    %v1670 = stablehlo.reshape %v1669 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v1671 = stablehlo.reshape %v1670 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1672 = stablehlo.reshape %v173 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1673 = stablehlo.constant dense<0.0> : tensor<32x96x56x56xf32>
    %v1674 = stablehlo.constant dense<6.0> : tensor<32x96x56x56xf32>
    %v1675 = stablehlo.compare GT, %v1672, %v1673 : (tensor<32x96x56x56xf32>, tensor<32x96x56x56xf32>) -> tensor<32x96x56x56xi1>
    %v1676 = stablehlo.compare LT, %v1672, %v1674 : (tensor<32x96x56x56xf32>, tensor<32x96x56x56xf32>) -> tensor<32x96x56x56xi1>
    %v1677 = stablehlo.and %v1675, %v1676 : tensor<32x96x56x56xi1>
    %v1678 = stablehlo.select %v1677, %v1671, %v1673 : tensor<32x96x56x56xi1>, tensor<32x96x56x56xf32>
    %v1679 = stablehlo.reshape %v1678 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v1680 = stablehlo.reshape %v1679 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1681 = stablehlo.reshape %v153 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1682 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1683 = stablehlo.constant dense<3136.0> : tensor<32x96x56x56xf32>
    %v1684 = stablehlo.constant dense<1.0e-5> : tensor<32x96x56x56xf32>
    %v1685 = stablehlo.reduce(%v1681 init: %v1682) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1686 = stablehlo.broadcast_in_dim %v1685, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v1687 = stablehlo.divide %v1686, %v1683 : tensor<32x96x56x56xf32>
    %v1688 = stablehlo.subtract %v1681, %v1687 : tensor<32x96x56x56xf32>
    %v1689 = stablehlo.multiply %v1688, %v1688 : tensor<32x96x56x56xf32>
    %v1690 = stablehlo.reduce(%v1689 init: %v1682) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1691 = stablehlo.broadcast_in_dim %v1690, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v1692 = stablehlo.divide %v1691, %v1683 : tensor<32x96x56x56xf32>
    %v1693 = stablehlo.add %v1692, %v1684 : tensor<32x96x56x56xf32>
    %v1694 = stablehlo.rsqrt %v1693 : tensor<32x96x56x56xf32>
    %v1695 = stablehlo.multiply %v1688, %v1694 : tensor<32x96x56x56xf32>
    %v1696 = stablehlo.broadcast_in_dim %gd2, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v1697 = stablehlo.multiply %v1696, %v1680 : tensor<32x96x56x56xf32>
    %v1698 = stablehlo.reduce(%v1697 init: %v1682) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1699 = stablehlo.broadcast_in_dim %v1698, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v1700 = stablehlo.multiply %v1695, %v1697 : tensor<32x96x56x56xf32>
    %v1701 = stablehlo.reduce(%v1700 init: %v1682) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1702 = stablehlo.broadcast_in_dim %v1701, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v1703 = stablehlo.multiply %v1697, %v1683 : tensor<32x96x56x56xf32>
    %v1704 = stablehlo.subtract %v1703, %v1699 : tensor<32x96x56x56xf32>
    %v1705 = stablehlo.multiply %v1695, %v1702 : tensor<32x96x56x56xf32>
    %v1706 = stablehlo.subtract %v1704, %v1705 : tensor<32x96x56x56xf32>
    %v1707 = stablehlo.divide %v1694, %v1683 : tensor<32x96x56x56xf32>
    %v1708 = stablehlo.multiply %v1707, %v1706 : tensor<32x96x56x56xf32>
    %v1709 = stablehlo.reshape %v1708 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v1710 = stablehlo.reshape %v1709 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1711 = stablehlo.reverse %Wd2, dims = [2, 3] : tensor<96x1x3x3xf32>
    %v1712 = stablehlo.convolution(%v1710, %v1711)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x3x3xf32>) -> tensor<32x96x56x56xf32>
    %v1713 = stablehlo.reshape %v1712 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v1714 = stablehlo.reshape %v1713 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1715 = stablehlo.reshape %v142 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1716 = stablehlo.constant dense<0.0> : tensor<32x96x56x56xf32>
    %v1717 = stablehlo.constant dense<6.0> : tensor<32x96x56x56xf32>
    %v1718 = stablehlo.compare GT, %v1715, %v1716 : (tensor<32x96x56x56xf32>, tensor<32x96x56x56xf32>) -> tensor<32x96x56x56xi1>
    %v1719 = stablehlo.compare LT, %v1715, %v1717 : (tensor<32x96x56x56xf32>, tensor<32x96x56x56xf32>) -> tensor<32x96x56x56xi1>
    %v1720 = stablehlo.and %v1718, %v1719 : tensor<32x96x56x56xi1>
    %v1721 = stablehlo.select %v1720, %v1714, %v1716 : tensor<32x96x56x56xi1>, tensor<32x96x56x56xf32>
    %v1722 = stablehlo.reshape %v1721 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v1723 = stablehlo.reshape %v1722 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1724 = stablehlo.reshape %v122 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1725 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1726 = stablehlo.constant dense<3136.0> : tensor<32x96x56x56xf32>
    %v1727 = stablehlo.constant dense<1.0e-5> : tensor<32x96x56x56xf32>
    %v1728 = stablehlo.reduce(%v1724 init: %v1725) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1729 = stablehlo.broadcast_in_dim %v1728, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v1730 = stablehlo.divide %v1729, %v1726 : tensor<32x96x56x56xf32>
    %v1731 = stablehlo.subtract %v1724, %v1730 : tensor<32x96x56x56xf32>
    %v1732 = stablehlo.multiply %v1731, %v1731 : tensor<32x96x56x56xf32>
    %v1733 = stablehlo.reduce(%v1732 init: %v1725) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1734 = stablehlo.broadcast_in_dim %v1733, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v1735 = stablehlo.divide %v1734, %v1726 : tensor<32x96x56x56xf32>
    %v1736 = stablehlo.add %v1735, %v1727 : tensor<32x96x56x56xf32>
    %v1737 = stablehlo.rsqrt %v1736 : tensor<32x96x56x56xf32>
    %v1738 = stablehlo.multiply %v1731, %v1737 : tensor<32x96x56x56xf32>
    %v1739 = stablehlo.broadcast_in_dim %ge2, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v1740 = stablehlo.multiply %v1739, %v1723 : tensor<32x96x56x56xf32>
    %v1741 = stablehlo.reduce(%v1740 init: %v1725) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1742 = stablehlo.broadcast_in_dim %v1741, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v1743 = stablehlo.multiply %v1738, %v1740 : tensor<32x96x56x56xf32>
    %v1744 = stablehlo.reduce(%v1743 init: %v1725) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1745 = stablehlo.broadcast_in_dim %v1744, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v1746 = stablehlo.multiply %v1740, %v1726 : tensor<32x96x56x56xf32>
    %v1747 = stablehlo.subtract %v1746, %v1742 : tensor<32x96x56x56xf32>
    %v1748 = stablehlo.multiply %v1738, %v1745 : tensor<32x96x56x56xf32>
    %v1749 = stablehlo.subtract %v1747, %v1748 : tensor<32x96x56x56xf32>
    %v1750 = stablehlo.divide %v1737, %v1726 : tensor<32x96x56x56xf32>
    %v1751 = stablehlo.multiply %v1750, %v1749 : tensor<32x96x56x56xf32>
    %v1752 = stablehlo.reshape %v1751 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v1753 = stablehlo.reshape %v1752 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1754 = stablehlo.transpose %We2, dims = [1, 0, 2, 3] : (tensor<96x24x1x1xf32>) -> tensor<24x96x1x1xf32>
    %v1755 = stablehlo.reverse %v1754, dims = [2, 3] : tensor<24x96x1x1xf32>
    %v1756 = stablehlo.convolution(%v1753, %v1755)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<24x96x1x1xf32>) -> tensor<32x24x56x56xf32>
    %v1757 = stablehlo.reshape %v1756 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v1758 = stablehlo.reshape %v1757 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v1759 = stablehlo.reshape %v1525 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v1760 = stablehlo.add %v1758, %v1759 : tensor<32x24x56x56xf32>
    %v1761 = stablehlo.reshape %v1760 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v1762 = stablehlo.reshape %v117 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v1763 = stablehlo.reshape %v1752 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1764 = stablehlo.transpose %v1762, dims = [1, 0, 2, 3] : (tensor<32x24x56x56xf32>) -> tensor<24x32x56x56xf32>
    %v1765 = stablehlo.transpose %v1763, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v1766 = stablehlo.convolution(%v1764, %v1765)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<24x32x56x56xf32>, tensor<96x32x56x56xf32>) -> tensor<24x96x1x1xf32>
    %v1767 = stablehlo.transpose %v1766, dims = [1, 0, 2, 3] : (tensor<24x96x1x1xf32>) -> tensor<96x24x1x1xf32>
    %v1768 = stablehlo.constant dense<0.3> : tensor<96x24x1x1xf32>
    %v1769 = stablehlo.multiply %v1767, %v1768 : tensor<96x24x1x1xf32>
    %v1770 = stablehlo.subtract %We2, %v1769 : tensor<96x24x1x1xf32>
    %v1771 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1772 = stablehlo.reshape %v122 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1773 = stablehlo.constant dense<3136.0> : tensor<32x96x56x56xf32>
    %v1774 = stablehlo.constant dense<1.0e-5> : tensor<32x96x56x56xf32>
    %v1775 = stablehlo.reduce(%v1772 init: %v1771) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1776 = stablehlo.broadcast_in_dim %v1775, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v1777 = stablehlo.divide %v1776, %v1773 : tensor<32x96x56x56xf32>
    %v1778 = stablehlo.subtract %v1772, %v1777 : tensor<32x96x56x56xf32>
    %v1779 = stablehlo.multiply %v1778, %v1778 : tensor<32x96x56x56xf32>
    %v1780 = stablehlo.reduce(%v1779 init: %v1771) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1781 = stablehlo.broadcast_in_dim %v1780, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v1782 = stablehlo.divide %v1781, %v1773 : tensor<32x96x56x56xf32>
    %v1783 = stablehlo.add %v1782, %v1774 : tensor<32x96x56x56xf32>
    %v1784 = stablehlo.rsqrt %v1783 : tensor<32x96x56x56xf32>
    %v1785 = stablehlo.multiply %v1778, %v1784 : tensor<32x96x56x56xf32>
    %v1786 = stablehlo.reshape %v1722 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1787 = stablehlo.multiply %v1786, %v1785 : tensor<32x96x56x56xf32>
    %v1788 = stablehlo.reduce(%v1787 init: %v1771) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v1789 = stablehlo.constant dense<0.3> : tensor<96xf32>
    %v1790 = stablehlo.multiply %v1788, %v1789 : tensor<96xf32>
    %v1791 = stablehlo.subtract %ge2, %v1790 : tensor<96xf32>
    %v1792 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1793 = stablehlo.reshape %v1722 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1794 = stablehlo.reduce(%v1793 init: %v1792) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v1795 = stablehlo.constant dense<0.3> : tensor<96xf32>
    %v1796 = stablehlo.multiply %v1794, %v1795 : tensor<96xf32>
    %v1797 = stablehlo.subtract %bte2, %v1796 : tensor<96xf32>
    %v1798 = stablehlo.reshape %v148 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1799 = stablehlo.reshape %v1709 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1800 = stablehlo.transpose %v1798, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v1801 = stablehlo.transpose %v1799, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v1802 = stablehlo.convolution(%v1800, %v1801)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 96 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<96x32x56x56xf32>) -> tensor<1x96x3x3xf32>
    %v1803 = stablehlo.reshape %v1802 : (tensor<1x96x3x3xf32>) -> tensor<96x1x3x3xf32>
    %v1804 = stablehlo.constant dense<0.3> : tensor<96x1x3x3xf32>
    %v1805 = stablehlo.multiply %v1803, %v1804 : tensor<96x1x3x3xf32>
    %v1806 = stablehlo.subtract %Wd2, %v1805 : tensor<96x1x3x3xf32>
    %v1807 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1808 = stablehlo.reshape %v153 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1809 = stablehlo.constant dense<3136.0> : tensor<32x96x56x56xf32>
    %v1810 = stablehlo.constant dense<1.0e-5> : tensor<32x96x56x56xf32>
    %v1811 = stablehlo.reduce(%v1808 init: %v1807) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1812 = stablehlo.broadcast_in_dim %v1811, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v1813 = stablehlo.divide %v1812, %v1809 : tensor<32x96x56x56xf32>
    %v1814 = stablehlo.subtract %v1808, %v1813 : tensor<32x96x56x56xf32>
    %v1815 = stablehlo.multiply %v1814, %v1814 : tensor<32x96x56x56xf32>
    %v1816 = stablehlo.reduce(%v1815 init: %v1807) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1817 = stablehlo.broadcast_in_dim %v1816, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v1818 = stablehlo.divide %v1817, %v1809 : tensor<32x96x56x56xf32>
    %v1819 = stablehlo.add %v1818, %v1810 : tensor<32x96x56x56xf32>
    %v1820 = stablehlo.rsqrt %v1819 : tensor<32x96x56x56xf32>
    %v1821 = stablehlo.multiply %v1814, %v1820 : tensor<32x96x56x56xf32>
    %v1822 = stablehlo.reshape %v1679 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1823 = stablehlo.multiply %v1822, %v1821 : tensor<32x96x56x56xf32>
    %v1824 = stablehlo.reduce(%v1823 init: %v1807) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v1825 = stablehlo.constant dense<0.3> : tensor<96xf32>
    %v1826 = stablehlo.multiply %v1824, %v1825 : tensor<96xf32>
    %v1827 = stablehlo.subtract %gd2, %v1826 : tensor<96xf32>
    %v1828 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1829 = stablehlo.reshape %v1679 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1830 = stablehlo.reduce(%v1829 init: %v1828) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v1831 = stablehlo.constant dense<0.3> : tensor<96xf32>
    %v1832 = stablehlo.multiply %v1830, %v1831 : tensor<96xf32>
    %v1833 = stablehlo.subtract %btd2, %v1832 : tensor<96xf32>
    %v1834 = stablehlo.reshape %v179 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v1835 = stablehlo.reshape %v1665 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v1836 = stablehlo.transpose %v1834, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v1837 = stablehlo.transpose %v1835, dims = [1, 0, 2, 3] : (tensor<32x24x56x56xf32>) -> tensor<24x32x56x56xf32>
    %v1838 = stablehlo.convolution(%v1836, %v1837)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<24x32x56x56xf32>) -> tensor<96x24x1x1xf32>
    %v1839 = stablehlo.transpose %v1838, dims = [1, 0, 2, 3] : (tensor<96x24x1x1xf32>) -> tensor<24x96x1x1xf32>
    %v1840 = stablehlo.constant dense<0.3> : tensor<24x96x1x1xf32>
    %v1841 = stablehlo.multiply %v1839, %v1840 : tensor<24x96x1x1xf32>
    %v1842 = stablehlo.subtract %Wp2, %v1841 : tensor<24x96x1x1xf32>
    %v1843 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1844 = stablehlo.reshape %v184 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v1845 = stablehlo.constant dense<3136.0> : tensor<32x24x56x56xf32>
    %v1846 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v1847 = stablehlo.reduce(%v1844 init: %v1843) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v1848 = stablehlo.broadcast_in_dim %v1847, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v1849 = stablehlo.divide %v1848, %v1845 : tensor<32x24x56x56xf32>
    %v1850 = stablehlo.subtract %v1844, %v1849 : tensor<32x24x56x56xf32>
    %v1851 = stablehlo.multiply %v1850, %v1850 : tensor<32x24x56x56xf32>
    %v1852 = stablehlo.reduce(%v1851 init: %v1843) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v1853 = stablehlo.broadcast_in_dim %v1852, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v1854 = stablehlo.divide %v1853, %v1845 : tensor<32x24x56x56xf32>
    %v1855 = stablehlo.add %v1854, %v1846 : tensor<32x24x56x56xf32>
    %v1856 = stablehlo.rsqrt %v1855 : tensor<32x24x56x56xf32>
    %v1857 = stablehlo.multiply %v1850, %v1856 : tensor<32x24x56x56xf32>
    %v1858 = stablehlo.reshape %v1525 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v1859 = stablehlo.multiply %v1858, %v1857 : tensor<32x24x56x56xf32>
    %v1860 = stablehlo.reduce(%v1859 init: %v1843) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v1861 = stablehlo.constant dense<0.3> : tensor<24xf32>
    %v1862 = stablehlo.multiply %v1860, %v1861 : tensor<24xf32>
    %v1863 = stablehlo.subtract %gp2, %v1862 : tensor<24xf32>
    %v1864 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1865 = stablehlo.reshape %v1525 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v1866 = stablehlo.reduce(%v1865 init: %v1864) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v1867 = stablehlo.constant dense<0.3> : tensor<24xf32>
    %v1868 = stablehlo.multiply %v1866, %v1867 : tensor<24xf32>
    %v1869 = stablehlo.subtract %btp2, %v1868 : tensor<24xf32>
    %v1870 = stablehlo.reshape %v1761 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v1871 = stablehlo.reshape %v97 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v1872 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1873 = stablehlo.constant dense<3136.0> : tensor<32x24x56x56xf32>
    %v1874 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v1875 = stablehlo.reduce(%v1871 init: %v1872) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v1876 = stablehlo.broadcast_in_dim %v1875, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v1877 = stablehlo.divide %v1876, %v1873 : tensor<32x24x56x56xf32>
    %v1878 = stablehlo.subtract %v1871, %v1877 : tensor<32x24x56x56xf32>
    %v1879 = stablehlo.multiply %v1878, %v1878 : tensor<32x24x56x56xf32>
    %v1880 = stablehlo.reduce(%v1879 init: %v1872) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v1881 = stablehlo.broadcast_in_dim %v1880, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v1882 = stablehlo.divide %v1881, %v1873 : tensor<32x24x56x56xf32>
    %v1883 = stablehlo.add %v1882, %v1874 : tensor<32x24x56x56xf32>
    %v1884 = stablehlo.rsqrt %v1883 : tensor<32x24x56x56xf32>
    %v1885 = stablehlo.multiply %v1878, %v1884 : tensor<32x24x56x56xf32>
    %v1886 = stablehlo.broadcast_in_dim %gp1, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v1887 = stablehlo.multiply %v1886, %v1870 : tensor<32x24x56x56xf32>
    %v1888 = stablehlo.reduce(%v1887 init: %v1872) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v1889 = stablehlo.broadcast_in_dim %v1888, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v1890 = stablehlo.multiply %v1885, %v1887 : tensor<32x24x56x56xf32>
    %v1891 = stablehlo.reduce(%v1890 init: %v1872) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v1892 = stablehlo.broadcast_in_dim %v1891, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v1893 = stablehlo.multiply %v1887, %v1873 : tensor<32x24x56x56xf32>
    %v1894 = stablehlo.subtract %v1893, %v1889 : tensor<32x24x56x56xf32>
    %v1895 = stablehlo.multiply %v1885, %v1892 : tensor<32x24x56x56xf32>
    %v1896 = stablehlo.subtract %v1894, %v1895 : tensor<32x24x56x56xf32>
    %v1897 = stablehlo.divide %v1884, %v1873 : tensor<32x24x56x56xf32>
    %v1898 = stablehlo.multiply %v1897, %v1896 : tensor<32x24x56x56xf32>
    %v1899 = stablehlo.reshape %v1898 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v1900 = stablehlo.reshape %v1899 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v1901 = stablehlo.transpose %Wp1, dims = [1, 0, 2, 3] : (tensor<24x64x1x1xf32>) -> tensor<64x24x1x1xf32>
    %v1902 = stablehlo.reverse %v1901, dims = [2, 3] : tensor<64x24x1x1xf32>
    %v1903 = stablehlo.convolution(%v1900, %v1902)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<64x24x1x1xf32>) -> tensor<32x64x56x56xf32>
    %v1904 = stablehlo.reshape %v1903 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v1905 = stablehlo.reshape %v1904 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v1906 = stablehlo.reshape %v86 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v1907 = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %v1908 = stablehlo.constant dense<6.0> : tensor<32x64x56x56xf32>
    %v1909 = stablehlo.compare GT, %v1906, %v1907 : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %v1910 = stablehlo.compare LT, %v1906, %v1908 : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %v1911 = stablehlo.and %v1909, %v1910 : tensor<32x64x56x56xi1>
    %v1912 = stablehlo.select %v1911, %v1905, %v1907 : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %v1913 = stablehlo.reshape %v1912 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v1914 = stablehlo.reshape %v1913 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v1915 = stablehlo.reshape %v66 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v1916 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1917 = stablehlo.constant dense<3136.0> : tensor<32x64x56x56xf32>
    %v1918 = stablehlo.constant dense<1.0e-5> : tensor<32x64x56x56xf32>
    %v1919 = stablehlo.reduce(%v1915 init: %v1916) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v1920 = stablehlo.broadcast_in_dim %v1919, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v1921 = stablehlo.divide %v1920, %v1917 : tensor<32x64x56x56xf32>
    %v1922 = stablehlo.subtract %v1915, %v1921 : tensor<32x64x56x56xf32>
    %v1923 = stablehlo.multiply %v1922, %v1922 : tensor<32x64x56x56xf32>
    %v1924 = stablehlo.reduce(%v1923 init: %v1916) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v1925 = stablehlo.broadcast_in_dim %v1924, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v1926 = stablehlo.divide %v1925, %v1917 : tensor<32x64x56x56xf32>
    %v1927 = stablehlo.add %v1926, %v1918 : tensor<32x64x56x56xf32>
    %v1928 = stablehlo.rsqrt %v1927 : tensor<32x64x56x56xf32>
    %v1929 = stablehlo.multiply %v1922, %v1928 : tensor<32x64x56x56xf32>
    %v1930 = stablehlo.broadcast_in_dim %gd1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v1931 = stablehlo.multiply %v1930, %v1914 : tensor<32x64x56x56xf32>
    %v1932 = stablehlo.reduce(%v1931 init: %v1916) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v1933 = stablehlo.broadcast_in_dim %v1932, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v1934 = stablehlo.multiply %v1929, %v1931 : tensor<32x64x56x56xf32>
    %v1935 = stablehlo.reduce(%v1934 init: %v1916) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v1936 = stablehlo.broadcast_in_dim %v1935, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v1937 = stablehlo.multiply %v1931, %v1917 : tensor<32x64x56x56xf32>
    %v1938 = stablehlo.subtract %v1937, %v1933 : tensor<32x64x56x56xf32>
    %v1939 = stablehlo.multiply %v1929, %v1936 : tensor<32x64x56x56xf32>
    %v1940 = stablehlo.subtract %v1938, %v1939 : tensor<32x64x56x56xf32>
    %v1941 = stablehlo.divide %v1928, %v1917 : tensor<32x64x56x56xf32>
    %v1942 = stablehlo.multiply %v1941, %v1940 : tensor<32x64x56x56xf32>
    %v1943 = stablehlo.reshape %v1942 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v1944 = stablehlo.reshape %v1943 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v1945 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1946 = stablehlo.pad %v1944, %v1945, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64x112x112xf32>
    %v1947 = stablehlo.reverse %Wd1, dims = [2, 3] : tensor<64x1x3x3xf32>
    %v1948 = stablehlo.convolution(%v1946, %v1947)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 64 : i64} : (tensor<32x64x112x112xf32>, tensor<64x1x3x3xf32>) -> tensor<32x64x112x112xf32>
    %v1949 = stablehlo.reshape %v1948 : (tensor<32x64x112x112xf32>) -> tensor<32x802816xf32>
    %v1950 = stablehlo.reshape %v1949 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v1951 = stablehlo.reshape %v55 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v1952 = stablehlo.constant dense<0.0> : tensor<32x64x112x112xf32>
    %v1953 = stablehlo.constant dense<6.0> : tensor<32x64x112x112xf32>
    %v1954 = stablehlo.compare GT, %v1951, %v1952 : (tensor<32x64x112x112xf32>, tensor<32x64x112x112xf32>) -> tensor<32x64x112x112xi1>
    %v1955 = stablehlo.compare LT, %v1951, %v1953 : (tensor<32x64x112x112xf32>, tensor<32x64x112x112xf32>) -> tensor<32x64x112x112xi1>
    %v1956 = stablehlo.and %v1954, %v1955 : tensor<32x64x112x112xi1>
    %v1957 = stablehlo.select %v1956, %v1950, %v1952 : tensor<32x64x112x112xi1>, tensor<32x64x112x112xf32>
    %v1958 = stablehlo.reshape %v1957 : (tensor<32x64x112x112xf32>) -> tensor<32x802816xf32>
    %v1959 = stablehlo.reshape %v1958 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v1960 = stablehlo.reshape %v35 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v1961 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1962 = stablehlo.constant dense<12544.0> : tensor<32x64x112x112xf32>
    %v1963 = stablehlo.constant dense<1.0e-5> : tensor<32x64x112x112xf32>
    %v1964 = stablehlo.reduce(%v1960 init: %v1961) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v1965 = stablehlo.broadcast_in_dim %v1964, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x112x112xf32>
    %v1966 = stablehlo.divide %v1965, %v1962 : tensor<32x64x112x112xf32>
    %v1967 = stablehlo.subtract %v1960, %v1966 : tensor<32x64x112x112xf32>
    %v1968 = stablehlo.multiply %v1967, %v1967 : tensor<32x64x112x112xf32>
    %v1969 = stablehlo.reduce(%v1968 init: %v1961) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v1970 = stablehlo.broadcast_in_dim %v1969, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x112x112xf32>
    %v1971 = stablehlo.divide %v1970, %v1962 : tensor<32x64x112x112xf32>
    %v1972 = stablehlo.add %v1971, %v1963 : tensor<32x64x112x112xf32>
    %v1973 = stablehlo.rsqrt %v1972 : tensor<32x64x112x112xf32>
    %v1974 = stablehlo.multiply %v1967, %v1973 : tensor<32x64x112x112xf32>
    %v1975 = stablehlo.broadcast_in_dim %ge1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v1976 = stablehlo.multiply %v1975, %v1959 : tensor<32x64x112x112xf32>
    %v1977 = stablehlo.reduce(%v1976 init: %v1961) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v1978 = stablehlo.broadcast_in_dim %v1977, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x112x112xf32>
    %v1979 = stablehlo.multiply %v1974, %v1976 : tensor<32x64x112x112xf32>
    %v1980 = stablehlo.reduce(%v1979 init: %v1961) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v1981 = stablehlo.broadcast_in_dim %v1980, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x112x112xf32>
    %v1982 = stablehlo.multiply %v1976, %v1962 : tensor<32x64x112x112xf32>
    %v1983 = stablehlo.subtract %v1982, %v1978 : tensor<32x64x112x112xf32>
    %v1984 = stablehlo.multiply %v1974, %v1981 : tensor<32x64x112x112xf32>
    %v1985 = stablehlo.subtract %v1983, %v1984 : tensor<32x64x112x112xf32>
    %v1986 = stablehlo.divide %v1973, %v1962 : tensor<32x64x112x112xf32>
    %v1987 = stablehlo.multiply %v1986, %v1985 : tensor<32x64x112x112xf32>
    %v1988 = stablehlo.reshape %v1987 : (tensor<32x64x112x112xf32>) -> tensor<32x802816xf32>
    %v1989 = stablehlo.reshape %v1988 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v1990 = stablehlo.transpose %We1, dims = [1, 0, 2, 3] : (tensor<64x16x1x1xf32>) -> tensor<16x64x1x1xf32>
    %v1991 = stablehlo.reverse %v1990, dims = [2, 3] : tensor<16x64x1x1xf32>
    %v1992 = stablehlo.convolution(%v1989, %v1991)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x112x112xf32>, tensor<16x64x1x1xf32>) -> tensor<32x16x112x112xf32>
    %v1993 = stablehlo.reshape %v1992 : (tensor<32x16x112x112xf32>) -> tensor<32x200704xf32>
    %v1994 = stablehlo.reshape %v30 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v1995 = stablehlo.reshape %v1988 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v1996 = stablehlo.transpose %v1994, dims = [1, 0, 2, 3] : (tensor<32x16x112x112xf32>) -> tensor<16x32x112x112xf32>
    %v1997 = stablehlo.transpose %v1995, dims = [1, 0, 2, 3] : (tensor<32x64x112x112xf32>) -> tensor<64x32x112x112xf32>
    %v1998 = stablehlo.convolution(%v1996, %v1997)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x32x112x112xf32>, tensor<64x32x112x112xf32>) -> tensor<16x64x1x1xf32>
    %v1999 = stablehlo.transpose %v1998, dims = [1, 0, 2, 3] : (tensor<16x64x1x1xf32>) -> tensor<64x16x1x1xf32>
    %v2000 = stablehlo.constant dense<0.3> : tensor<64x16x1x1xf32>
    %v2001 = stablehlo.multiply %v1999, %v2000 : tensor<64x16x1x1xf32>
    %v2002 = stablehlo.subtract %We1, %v2001 : tensor<64x16x1x1xf32>
    %v2003 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2004 = stablehlo.reshape %v35 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v2005 = stablehlo.constant dense<12544.0> : tensor<32x64x112x112xf32>
    %v2006 = stablehlo.constant dense<1.0e-5> : tensor<32x64x112x112xf32>
    %v2007 = stablehlo.reduce(%v2004 init: %v2003) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v2008 = stablehlo.broadcast_in_dim %v2007, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x112x112xf32>
    %v2009 = stablehlo.divide %v2008, %v2005 : tensor<32x64x112x112xf32>
    %v2010 = stablehlo.subtract %v2004, %v2009 : tensor<32x64x112x112xf32>
    %v2011 = stablehlo.multiply %v2010, %v2010 : tensor<32x64x112x112xf32>
    %v2012 = stablehlo.reduce(%v2011 init: %v2003) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v2013 = stablehlo.broadcast_in_dim %v2012, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x112x112xf32>
    %v2014 = stablehlo.divide %v2013, %v2005 : tensor<32x64x112x112xf32>
    %v2015 = stablehlo.add %v2014, %v2006 : tensor<32x64x112x112xf32>
    %v2016 = stablehlo.rsqrt %v2015 : tensor<32x64x112x112xf32>
    %v2017 = stablehlo.multiply %v2010, %v2016 : tensor<32x64x112x112xf32>
    %v2018 = stablehlo.reshape %v1958 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v2019 = stablehlo.multiply %v2018, %v2017 : tensor<32x64x112x112xf32>
    %v2020 = stablehlo.reduce(%v2019 init: %v2003) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v2021 = stablehlo.constant dense<0.3> : tensor<64xf32>
    %v2022 = stablehlo.multiply %v2020, %v2021 : tensor<64xf32>
    %v2023 = stablehlo.subtract %ge1, %v2022 : tensor<64xf32>
    %v2024 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2025 = stablehlo.reshape %v1958 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v2026 = stablehlo.reduce(%v2025 init: %v2024) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v2027 = stablehlo.constant dense<0.3> : tensor<64xf32>
    %v2028 = stablehlo.multiply %v2026, %v2027 : tensor<64xf32>
    %v2029 = stablehlo.subtract %bte1, %v2028 : tensor<64xf32>
    %v2030 = stablehlo.reshape %v61 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v2031 = stablehlo.reshape %v1943 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v2032 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2033 = stablehlo.pad %v2031, %v2032, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64x112x112xf32>
    %v2034 = stablehlo.transpose %v2030, dims = [1, 0, 2, 3] : (tensor<32x64x112x112xf32>) -> tensor<64x32x112x112xf32>
    %v2035 = stablehlo.transpose %v2033, dims = [1, 0, 2, 3] : (tensor<32x64x112x112xf32>) -> tensor<64x32x112x112xf32>
    %v2036 = stablehlo.convolution(%v2034, %v2035)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 64 : i64, feature_group_count = 1 : i64} : (tensor<64x32x112x112xf32>, tensor<64x32x112x112xf32>) -> tensor<1x64x3x3xf32>
    %v2037 = stablehlo.reshape %v2036 : (tensor<1x64x3x3xf32>) -> tensor<64x1x3x3xf32>
    %v2038 = stablehlo.constant dense<0.3> : tensor<64x1x3x3xf32>
    %v2039 = stablehlo.multiply %v2037, %v2038 : tensor<64x1x3x3xf32>
    %v2040 = stablehlo.subtract %Wd1, %v2039 : tensor<64x1x3x3xf32>
    %v2041 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2042 = stablehlo.reshape %v66 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v2043 = stablehlo.constant dense<3136.0> : tensor<32x64x56x56xf32>
    %v2044 = stablehlo.constant dense<1.0e-5> : tensor<32x64x56x56xf32>
    %v2045 = stablehlo.reduce(%v2042 init: %v2041) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v2046 = stablehlo.broadcast_in_dim %v2045, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v2047 = stablehlo.divide %v2046, %v2043 : tensor<32x64x56x56xf32>
    %v2048 = stablehlo.subtract %v2042, %v2047 : tensor<32x64x56x56xf32>
    %v2049 = stablehlo.multiply %v2048, %v2048 : tensor<32x64x56x56xf32>
    %v2050 = stablehlo.reduce(%v2049 init: %v2041) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v2051 = stablehlo.broadcast_in_dim %v2050, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v2052 = stablehlo.divide %v2051, %v2043 : tensor<32x64x56x56xf32>
    %v2053 = stablehlo.add %v2052, %v2044 : tensor<32x64x56x56xf32>
    %v2054 = stablehlo.rsqrt %v2053 : tensor<32x64x56x56xf32>
    %v2055 = stablehlo.multiply %v2048, %v2054 : tensor<32x64x56x56xf32>
    %v2056 = stablehlo.reshape %v1913 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v2057 = stablehlo.multiply %v2056, %v2055 : tensor<32x64x56x56xf32>
    %v2058 = stablehlo.reduce(%v2057 init: %v2041) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v2059 = stablehlo.constant dense<0.3> : tensor<64xf32>
    %v2060 = stablehlo.multiply %v2058, %v2059 : tensor<64xf32>
    %v2061 = stablehlo.subtract %gd1, %v2060 : tensor<64xf32>
    %v2062 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2063 = stablehlo.reshape %v1913 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v2064 = stablehlo.reduce(%v2063 init: %v2062) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v2065 = stablehlo.constant dense<0.3> : tensor<64xf32>
    %v2066 = stablehlo.multiply %v2064, %v2065 : tensor<64xf32>
    %v2067 = stablehlo.subtract %btd1, %v2066 : tensor<64xf32>
    %v2068 = stablehlo.reshape %v92 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v2069 = stablehlo.reshape %v1899 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v2070 = stablehlo.transpose %v2068, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v2071 = stablehlo.transpose %v2069, dims = [1, 0, 2, 3] : (tensor<32x24x56x56xf32>) -> tensor<24x32x56x56xf32>
    %v2072 = stablehlo.convolution(%v2070, %v2071)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<24x32x56x56xf32>) -> tensor<64x24x1x1xf32>
    %v2073 = stablehlo.transpose %v2072, dims = [1, 0, 2, 3] : (tensor<64x24x1x1xf32>) -> tensor<24x64x1x1xf32>
    %v2074 = stablehlo.constant dense<0.3> : tensor<24x64x1x1xf32>
    %v2075 = stablehlo.multiply %v2073, %v2074 : tensor<24x64x1x1xf32>
    %v2076 = stablehlo.subtract %Wp1, %v2075 : tensor<24x64x1x1xf32>
    %v2077 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2078 = stablehlo.reshape %v97 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v2079 = stablehlo.constant dense<3136.0> : tensor<32x24x56x56xf32>
    %v2080 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v2081 = stablehlo.reduce(%v2078 init: %v2077) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v2082 = stablehlo.broadcast_in_dim %v2081, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v2083 = stablehlo.divide %v2082, %v2079 : tensor<32x24x56x56xf32>
    %v2084 = stablehlo.subtract %v2078, %v2083 : tensor<32x24x56x56xf32>
    %v2085 = stablehlo.multiply %v2084, %v2084 : tensor<32x24x56x56xf32>
    %v2086 = stablehlo.reduce(%v2085 init: %v2077) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v2087 = stablehlo.broadcast_in_dim %v2086, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v2088 = stablehlo.divide %v2087, %v2079 : tensor<32x24x56x56xf32>
    %v2089 = stablehlo.add %v2088, %v2080 : tensor<32x24x56x56xf32>
    %v2090 = stablehlo.rsqrt %v2089 : tensor<32x24x56x56xf32>
    %v2091 = stablehlo.multiply %v2084, %v2090 : tensor<32x24x56x56xf32>
    %v2092 = stablehlo.reshape %v1761 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v2093 = stablehlo.multiply %v2092, %v2091 : tensor<32x24x56x56xf32>
    %v2094 = stablehlo.reduce(%v2093 init: %v2077) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v2095 = stablehlo.constant dense<0.3> : tensor<24xf32>
    %v2096 = stablehlo.multiply %v2094, %v2095 : tensor<24xf32>
    %v2097 = stablehlo.subtract %gp1, %v2096 : tensor<24xf32>
    %v2098 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2099 = stablehlo.reshape %v1761 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v2100 = stablehlo.reduce(%v2099 init: %v2098) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v2101 = stablehlo.constant dense<0.3> : tensor<24xf32>
    %v2102 = stablehlo.multiply %v2100, %v2101 : tensor<24xf32>
    %v2103 = stablehlo.subtract %btp1, %v2102 : tensor<24xf32>
    %v2104 = stablehlo.reshape %v1993 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v2105 = stablehlo.reshape %v24 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v2106 = stablehlo.constant dense<0.0> : tensor<32x16x112x112xf32>
    %v2107 = stablehlo.constant dense<6.0> : tensor<32x16x112x112xf32>
    %v2108 = stablehlo.compare GT, %v2105, %v2106 : (tensor<32x16x112x112xf32>, tensor<32x16x112x112xf32>) -> tensor<32x16x112x112xi1>
    %v2109 = stablehlo.compare LT, %v2105, %v2107 : (tensor<32x16x112x112xf32>, tensor<32x16x112x112xf32>) -> tensor<32x16x112x112xi1>
    %v2110 = stablehlo.and %v2108, %v2109 : tensor<32x16x112x112xi1>
    %v2111 = stablehlo.select %v2110, %v2104, %v2106 : tensor<32x16x112x112xi1>, tensor<32x16x112x112xf32>
    %v2112 = stablehlo.reshape %v2111 : (tensor<32x16x112x112xf32>) -> tensor<32x200704xf32>
    %v2113 = stablehlo.reshape %v2112 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v2114 = stablehlo.reshape %v4 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v2115 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2116 = stablehlo.constant dense<12544.0> : tensor<32x16x112x112xf32>
    %v2117 = stablehlo.constant dense<1.0e-5> : tensor<32x16x112x112xf32>
    %v2118 = stablehlo.reduce(%v2114 init: %v2115) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<32x16xf32>
    %v2119 = stablehlo.broadcast_in_dim %v2118, dims = [0, 1] : (tensor<32x16xf32>) -> tensor<32x16x112x112xf32>
    %v2120 = stablehlo.divide %v2119, %v2116 : tensor<32x16x112x112xf32>
    %v2121 = stablehlo.subtract %v2114, %v2120 : tensor<32x16x112x112xf32>
    %v2122 = stablehlo.multiply %v2121, %v2121 : tensor<32x16x112x112xf32>
    %v2123 = stablehlo.reduce(%v2122 init: %v2115) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<32x16xf32>
    %v2124 = stablehlo.broadcast_in_dim %v2123, dims = [0, 1] : (tensor<32x16xf32>) -> tensor<32x16x112x112xf32>
    %v2125 = stablehlo.divide %v2124, %v2116 : tensor<32x16x112x112xf32>
    %v2126 = stablehlo.add %v2125, %v2117 : tensor<32x16x112x112xf32>
    %v2127 = stablehlo.rsqrt %v2126 : tensor<32x16x112x112xf32>
    %v2128 = stablehlo.multiply %v2121, %v2127 : tensor<32x16x112x112xf32>
    %v2129 = stablehlo.broadcast_in_dim %gs, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v2130 = stablehlo.multiply %v2129, %v2113 : tensor<32x16x112x112xf32>
    %v2131 = stablehlo.reduce(%v2130 init: %v2115) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<32x16xf32>
    %v2132 = stablehlo.broadcast_in_dim %v2131, dims = [0, 1] : (tensor<32x16xf32>) -> tensor<32x16x112x112xf32>
    %v2133 = stablehlo.multiply %v2128, %v2130 : tensor<32x16x112x112xf32>
    %v2134 = stablehlo.reduce(%v2133 init: %v2115) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<32x16xf32>
    %v2135 = stablehlo.broadcast_in_dim %v2134, dims = [0, 1] : (tensor<32x16xf32>) -> tensor<32x16x112x112xf32>
    %v2136 = stablehlo.multiply %v2130, %v2116 : tensor<32x16x112x112xf32>
    %v2137 = stablehlo.subtract %v2136, %v2132 : tensor<32x16x112x112xf32>
    %v2138 = stablehlo.multiply %v2128, %v2135 : tensor<32x16x112x112xf32>
    %v2139 = stablehlo.subtract %v2137, %v2138 : tensor<32x16x112x112xf32>
    %v2140 = stablehlo.divide %v2127, %v2116 : tensor<32x16x112x112xf32>
    %v2141 = stablehlo.multiply %v2140, %v2139 : tensor<32x16x112x112xf32>
    %v2142 = stablehlo.reshape %v2141 : (tensor<32x16x112x112xf32>) -> tensor<32x200704xf32>
    %v2143 = stablehlo.reshape %x : (tensor<32x150528xf32>) -> tensor<32x3x224x224xf32>
    %v2144 = stablehlo.reshape %v2142 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v2145 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2146 = stablehlo.pad %v2144, %v2145, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<32x16x224x224xf32>
    %v2147 = stablehlo.transpose %v2143, dims = [1, 0, 2, 3] : (tensor<32x3x224x224xf32>) -> tensor<3x32x224x224xf32>
    %v2148 = stablehlo.transpose %v2146, dims = [1, 0, 2, 3] : (tensor<32x16x224x224xf32>) -> tensor<16x32x224x224xf32>
    %v2149 = stablehlo.convolution(%v2147, %v2148)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x32x224x224xf32>, tensor<16x32x224x224xf32>) -> tensor<3x16x3x3xf32>
    %v2150 = stablehlo.transpose %v2149, dims = [1, 0, 2, 3] : (tensor<3x16x3x3xf32>) -> tensor<16x3x3x3xf32>
    %v2151 = stablehlo.constant dense<0.3> : tensor<16x3x3x3xf32>
    %v2152 = stablehlo.multiply %v2150, %v2151 : tensor<16x3x3x3xf32>
    %v2153 = stablehlo.subtract %Ws, %v2152 : tensor<16x3x3x3xf32>
    %v2154 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2155 = stablehlo.reshape %v4 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v2156 = stablehlo.constant dense<12544.0> : tensor<32x16x112x112xf32>
    %v2157 = stablehlo.constant dense<1.0e-5> : tensor<32x16x112x112xf32>
    %v2158 = stablehlo.reduce(%v2155 init: %v2154) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<32x16xf32>
    %v2159 = stablehlo.broadcast_in_dim %v2158, dims = [0, 1] : (tensor<32x16xf32>) -> tensor<32x16x112x112xf32>
    %v2160 = stablehlo.divide %v2159, %v2156 : tensor<32x16x112x112xf32>
    %v2161 = stablehlo.subtract %v2155, %v2160 : tensor<32x16x112x112xf32>
    %v2162 = stablehlo.multiply %v2161, %v2161 : tensor<32x16x112x112xf32>
    %v2163 = stablehlo.reduce(%v2162 init: %v2154) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<32x16xf32>
    %v2164 = stablehlo.broadcast_in_dim %v2163, dims = [0, 1] : (tensor<32x16xf32>) -> tensor<32x16x112x112xf32>
    %v2165 = stablehlo.divide %v2164, %v2156 : tensor<32x16x112x112xf32>
    %v2166 = stablehlo.add %v2165, %v2157 : tensor<32x16x112x112xf32>
    %v2167 = stablehlo.rsqrt %v2166 : tensor<32x16x112x112xf32>
    %v2168 = stablehlo.multiply %v2161, %v2167 : tensor<32x16x112x112xf32>
    %v2169 = stablehlo.reshape %v2112 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v2170 = stablehlo.multiply %v2169, %v2168 : tensor<32x16x112x112xf32>
    %v2171 = stablehlo.reduce(%v2170 init: %v2154) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v2172 = stablehlo.constant dense<0.3> : tensor<16xf32>
    %v2173 = stablehlo.multiply %v2171, %v2172 : tensor<16xf32>
    %v2174 = stablehlo.subtract %gs, %v2173 : tensor<16xf32>
    %v2175 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2176 = stablehlo.reshape %v2112 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v2177 = stablehlo.reduce(%v2176 init: %v2175) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v2178 = stablehlo.constant dense<0.3> : tensor<16xf32>
    %v2179 = stablehlo.multiply %v2177, %v2178 : tensor<16xf32>
    %v2180 = stablehlo.subtract %bts, %v2179 : tensor<16xf32>
    return %v2153, %v2174, %v2180, %v2002, %v2023, %v2029, %v2040, %v2061, %v2067, %v2076, %v2097, %v2103, %v1770, %v1791, %v1797, %v1806, %v1827, %v1833, %v1842, %v1863, %v1869, %v1534, %v1555, %v1561, %v1572, %v1593, %v1599, %v1608, %v1629, %v1635, %v1302, %v1323, %v1329, %v1338, %v1359, %v1365, %v1374, %v1395, %v1401, %v1066, %v1087, %v1093, %v1104, %v1125, %v1131, %v1140, %v1161, %v1167, %v832, %v853, %v859, %v870, %v891, %v897, %v906, %v927, %v933, %v672, %v693, %v699, %v614, %v619 : tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<64x16x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x1x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<24x64x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<96x24x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x3x3xf32>, tensor<96xf32>, tensor<96xf32>, tensor<24x96x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<96x24x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x3x3xf32>, tensor<96xf32>, tensor<96xf32>, tensor<32x96x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x32x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x1x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<32x128x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x32x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x1x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<64x128x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x1x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x10xf32>, tensor<10xf32>
  }
}

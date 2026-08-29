module @m {
  func.func @mobilenetv2in_fwd(%x: tensor<64x150528xf32>, %Ws: tensor<32x3x3x3xf32>, %gs: tensor<32xf32>, %bts: tensor<32xf32>, %Wd1: tensor<32x1x3x3xf32>, %gd1: tensor<32xf32>, %btd1: tensor<32xf32>, %Wp1: tensor<16x32x1x1xf32>, %gp1: tensor<16xf32>, %btp1: tensor<16xf32>, %We2: tensor<96x16x1x1xf32>, %ge2: tensor<96xf32>, %bte2: tensor<96xf32>, %Wd2: tensor<96x1x3x3xf32>, %gd2: tensor<96xf32>, %btd2: tensor<96xf32>, %Wp2: tensor<24x96x1x1xf32>, %gp2: tensor<24xf32>, %btp2: tensor<24xf32>, %We3: tensor<144x24x1x1xf32>, %ge3: tensor<144xf32>, %bte3: tensor<144xf32>, %Wd3: tensor<144x1x3x3xf32>, %gd3: tensor<144xf32>, %btd3: tensor<144xf32>, %Wp3: tensor<24x144x1x1xf32>, %gp3: tensor<24xf32>, %btp3: tensor<24xf32>, %We4: tensor<144x24x1x1xf32>, %ge4: tensor<144xf32>, %bte4: tensor<144xf32>, %Wd4: tensor<144x1x3x3xf32>, %gd4: tensor<144xf32>, %btd4: tensor<144xf32>, %Wp4: tensor<32x144x1x1xf32>, %gp4: tensor<32xf32>, %btp4: tensor<32xf32>, %We5: tensor<192x32x1x1xf32>, %ge5: tensor<192xf32>, %bte5: tensor<192xf32>, %Wd5: tensor<192x1x3x3xf32>, %gd5: tensor<192xf32>, %btd5: tensor<192xf32>, %Wp5: tensor<32x192x1x1xf32>, %gp5: tensor<32xf32>, %btp5: tensor<32xf32>, %We6: tensor<192x32x1x1xf32>, %ge6: tensor<192xf32>, %bte6: tensor<192xf32>, %Wd6: tensor<192x1x3x3xf32>, %gd6: tensor<192xf32>, %btd6: tensor<192xf32>, %Wp6: tensor<32x192x1x1xf32>, %gp6: tensor<32xf32>, %btp6: tensor<32xf32>, %We7: tensor<192x32x1x1xf32>, %ge7: tensor<192xf32>, %bte7: tensor<192xf32>, %Wd7: tensor<192x1x3x3xf32>, %gd7: tensor<192xf32>, %btd7: tensor<192xf32>, %Wp7: tensor<64x192x1x1xf32>, %gp7: tensor<64xf32>, %btp7: tensor<64xf32>, %We8: tensor<384x64x1x1xf32>, %ge8: tensor<384xf32>, %bte8: tensor<384xf32>, %Wd8: tensor<384x1x3x3xf32>, %gd8: tensor<384xf32>, %btd8: tensor<384xf32>, %Wp8: tensor<64x384x1x1xf32>, %gp8: tensor<64xf32>, %btp8: tensor<64xf32>, %We9: tensor<384x64x1x1xf32>, %ge9: tensor<384xf32>, %bte9: tensor<384xf32>, %Wd9: tensor<384x1x3x3xf32>, %gd9: tensor<384xf32>, %btd9: tensor<384xf32>, %Wp9: tensor<64x384x1x1xf32>, %gp9: tensor<64xf32>, %btp9: tensor<64xf32>, %We10: tensor<384x64x1x1xf32>, %ge10: tensor<384xf32>, %bte10: tensor<384xf32>, %Wd10: tensor<384x1x3x3xf32>, %gd10: tensor<384xf32>, %btd10: tensor<384xf32>, %Wp10: tensor<64x384x1x1xf32>, %gp10: tensor<64xf32>, %btp10: tensor<64xf32>, %We11: tensor<384x64x1x1xf32>, %ge11: tensor<384xf32>, %bte11: tensor<384xf32>, %Wd11: tensor<384x1x3x3xf32>, %gd11: tensor<384xf32>, %btd11: tensor<384xf32>, %Wp11: tensor<96x384x1x1xf32>, %gp11: tensor<96xf32>, %btp11: tensor<96xf32>, %We12: tensor<576x96x1x1xf32>, %ge12: tensor<576xf32>, %bte12: tensor<576xf32>, %Wd12: tensor<576x1x3x3xf32>, %gd12: tensor<576xf32>, %btd12: tensor<576xf32>, %Wp12: tensor<96x576x1x1xf32>, %gp12: tensor<96xf32>, %btp12: tensor<96xf32>, %We13: tensor<576x96x1x1xf32>, %ge13: tensor<576xf32>, %bte13: tensor<576xf32>, %Wd13: tensor<576x1x3x3xf32>, %gd13: tensor<576xf32>, %btd13: tensor<576xf32>, %Wp13: tensor<96x576x1x1xf32>, %gp13: tensor<96xf32>, %btp13: tensor<96xf32>, %We14: tensor<576x96x1x1xf32>, %ge14: tensor<576xf32>, %bte14: tensor<576xf32>, %Wd14: tensor<576x1x3x3xf32>, %gd14: tensor<576xf32>, %btd14: tensor<576xf32>, %Wp14: tensor<160x576x1x1xf32>, %gp14: tensor<160xf32>, %btp14: tensor<160xf32>, %We15: tensor<960x160x1x1xf32>, %ge15: tensor<960xf32>, %bte15: tensor<960xf32>, %Wd15: tensor<960x1x3x3xf32>, %gd15: tensor<960xf32>, %btd15: tensor<960xf32>, %Wp15: tensor<160x960x1x1xf32>, %gp15: tensor<160xf32>, %btp15: tensor<160xf32>, %We16: tensor<960x160x1x1xf32>, %ge16: tensor<960xf32>, %bte16: tensor<960xf32>, %Wd16: tensor<960x1x3x3xf32>, %gd16: tensor<960xf32>, %btd16: tensor<960xf32>, %Wp16: tensor<160x960x1x1xf32>, %gp16: tensor<160xf32>, %btp16: tensor<160xf32>, %We17: tensor<960x160x1x1xf32>, %ge17: tensor<960xf32>, %bte17: tensor<960xf32>, %Wd17: tensor<960x1x3x3xf32>, %gd17: tensor<960xf32>, %btd17: tensor<960xf32>, %Wp17: tensor<320x960x1x1xf32>, %gp17: tensor<320xf32>, %btp17: tensor<320xf32>, %Wh: tensor<1280x320x1x1xf32>, %gh: tensor<1280xf32>, %bth: tensor<1280xf32>, %Wfc: tensor<1280x1000xf32>, %bfc: tensor<1000xf32>) -> tensor<64x1000xf32> {
    // -- MobileNetV2 (17-block paper) forward: every line is pretty(verified AST node) --
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
    %v0 = stablehlo.reshape %x : (tensor<64x150528xf32>) -> tensor<64x3x224x224xf32>
    %v1 = stablehlo.convolution(%v0, %Ws)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x3x224x224xf32>, tensor<32x3x3x3xf32>) -> tensor<64x32x112x112xf32>
    %v2 = stablehlo.broadcast_in_dim %zb32, dims = [1] : (tensor<32xf32>) -> tensor<64x32x112x112xf32>
    %v3 = stablehlo.add %v1, %v2 : tensor<64x32x112x112xf32>
    %v4 = stablehlo.reshape %v3 : (tensor<64x32x112x112xf32>) -> tensor<64x401408xf32>
    %v5 = stablehlo.reshape %v4 : (tensor<64x401408xf32>) -> tensor<64x32x112x112xf32>
    %v6 = stablehlo.constant dense<0.0> : tensor<f32>
    %v7 = stablehlo.constant dense<12544.0> : tensor<64x32x112x112xf32>
    %v8 = stablehlo.constant dense<1.0e-5> : tensor<64x32x112x112xf32>
    %v9 = stablehlo.reduce(%v5 init: %v6) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x32x112x112xf32>, tensor<f32>) -> tensor<64x32xf32>
    %v10 = stablehlo.broadcast_in_dim %v9, dims = [0, 1] : (tensor<64x32xf32>) -> tensor<64x32x112x112xf32>
    %v11 = stablehlo.divide %v10, %v7 : tensor<64x32x112x112xf32>
    %v12 = stablehlo.subtract %v5, %v11 : tensor<64x32x112x112xf32>
    %v13 = stablehlo.multiply %v12, %v12 : tensor<64x32x112x112xf32>
    %v14 = stablehlo.reduce(%v13 init: %v6) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x32x112x112xf32>, tensor<f32>) -> tensor<64x32xf32>
    %v15 = stablehlo.broadcast_in_dim %v14, dims = [0, 1] : (tensor<64x32xf32>) -> tensor<64x32x112x112xf32>
    %v16 = stablehlo.divide %v15, %v7 : tensor<64x32x112x112xf32>
    %v17 = stablehlo.add %v16, %v8 : tensor<64x32x112x112xf32>
    %v18 = stablehlo.rsqrt %v17 : tensor<64x32x112x112xf32>
    %v19 = stablehlo.multiply %v12, %v18 : tensor<64x32x112x112xf32>
    %v20 = stablehlo.broadcast_in_dim %gs, dims = [1] : (tensor<32xf32>) -> tensor<64x32x112x112xf32>
    %v21 = stablehlo.broadcast_in_dim %bts, dims = [1] : (tensor<32xf32>) -> tensor<64x32x112x112xf32>
    %v22 = stablehlo.multiply %v19, %v20 : tensor<64x32x112x112xf32>
    %v23 = stablehlo.add %v22, %v21 : tensor<64x32x112x112xf32>
    %v24 = stablehlo.reshape %v23 : (tensor<64x32x112x112xf32>) -> tensor<64x401408xf32>
    %v25 = stablehlo.reshape %v24 : (tensor<64x401408xf32>) -> tensor<64x32x112x112xf32>
    %v26 = stablehlo.constant dense<0.0> : tensor<64x32x112x112xf32>
    %v27 = stablehlo.constant dense<6.0> : tensor<64x32x112x112xf32>
    %v28 = stablehlo.maximum %v25, %v26 : tensor<64x32x112x112xf32>
    %v29 = stablehlo.minimum %v28, %v27 : tensor<64x32x112x112xf32>
    %v30 = stablehlo.reshape %v29 : (tensor<64x32x112x112xf32>) -> tensor<64x401408xf32>
    %v31 = stablehlo.reshape %v30 : (tensor<64x401408xf32>) -> tensor<64x32x112x112xf32>
    %v32 = stablehlo.convolution(%v31, %Wd1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 32 : i64} : (tensor<64x32x112x112xf32>, tensor<32x1x3x3xf32>) -> tensor<64x32x112x112xf32>
    %v33 = stablehlo.broadcast_in_dim %zb32, dims = [1] : (tensor<32xf32>) -> tensor<64x32x112x112xf32>
    %v34 = stablehlo.add %v32, %v33 : tensor<64x32x112x112xf32>
    %v35 = stablehlo.reshape %v34 : (tensor<64x32x112x112xf32>) -> tensor<64x401408xf32>
    %v36 = stablehlo.reshape %v35 : (tensor<64x401408xf32>) -> tensor<64x32x112x112xf32>
    %v37 = stablehlo.constant dense<0.0> : tensor<f32>
    %v38 = stablehlo.constant dense<12544.0> : tensor<64x32x112x112xf32>
    %v39 = stablehlo.constant dense<1.0e-5> : tensor<64x32x112x112xf32>
    %v40 = stablehlo.reduce(%v36 init: %v37) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x32x112x112xf32>, tensor<f32>) -> tensor<64x32xf32>
    %v41 = stablehlo.broadcast_in_dim %v40, dims = [0, 1] : (tensor<64x32xf32>) -> tensor<64x32x112x112xf32>
    %v42 = stablehlo.divide %v41, %v38 : tensor<64x32x112x112xf32>
    %v43 = stablehlo.subtract %v36, %v42 : tensor<64x32x112x112xf32>
    %v44 = stablehlo.multiply %v43, %v43 : tensor<64x32x112x112xf32>
    %v45 = stablehlo.reduce(%v44 init: %v37) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x32x112x112xf32>, tensor<f32>) -> tensor<64x32xf32>
    %v46 = stablehlo.broadcast_in_dim %v45, dims = [0, 1] : (tensor<64x32xf32>) -> tensor<64x32x112x112xf32>
    %v47 = stablehlo.divide %v46, %v38 : tensor<64x32x112x112xf32>
    %v48 = stablehlo.add %v47, %v39 : tensor<64x32x112x112xf32>
    %v49 = stablehlo.rsqrt %v48 : tensor<64x32x112x112xf32>
    %v50 = stablehlo.multiply %v43, %v49 : tensor<64x32x112x112xf32>
    %v51 = stablehlo.broadcast_in_dim %gd1, dims = [1] : (tensor<32xf32>) -> tensor<64x32x112x112xf32>
    %v52 = stablehlo.broadcast_in_dim %btd1, dims = [1] : (tensor<32xf32>) -> tensor<64x32x112x112xf32>
    %v53 = stablehlo.multiply %v50, %v51 : tensor<64x32x112x112xf32>
    %v54 = stablehlo.add %v53, %v52 : tensor<64x32x112x112xf32>
    %v55 = stablehlo.reshape %v54 : (tensor<64x32x112x112xf32>) -> tensor<64x401408xf32>
    %v56 = stablehlo.reshape %v55 : (tensor<64x401408xf32>) -> tensor<64x32x112x112xf32>
    %v57 = stablehlo.constant dense<0.0> : tensor<64x32x112x112xf32>
    %v58 = stablehlo.constant dense<6.0> : tensor<64x32x112x112xf32>
    %v59 = stablehlo.maximum %v56, %v57 : tensor<64x32x112x112xf32>
    %v60 = stablehlo.minimum %v59, %v58 : tensor<64x32x112x112xf32>
    %v61 = stablehlo.reshape %v60 : (tensor<64x32x112x112xf32>) -> tensor<64x401408xf32>
    %v62 = stablehlo.reshape %v61 : (tensor<64x401408xf32>) -> tensor<64x32x112x112xf32>
    %v63 = stablehlo.convolution(%v62, %Wp1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x112x112xf32>, tensor<16x32x1x1xf32>) -> tensor<64x16x112x112xf32>
    %v64 = stablehlo.broadcast_in_dim %zb16, dims = [1] : (tensor<16xf32>) -> tensor<64x16x112x112xf32>
    %v65 = stablehlo.add %v63, %v64 : tensor<64x16x112x112xf32>
    %v66 = stablehlo.reshape %v65 : (tensor<64x16x112x112xf32>) -> tensor<64x200704xf32>
    %v67 = stablehlo.reshape %v66 : (tensor<64x200704xf32>) -> tensor<64x16x112x112xf32>
    %v68 = stablehlo.constant dense<0.0> : tensor<f32>
    %v69 = stablehlo.constant dense<12544.0> : tensor<64x16x112x112xf32>
    %v70 = stablehlo.constant dense<1.0e-5> : tensor<64x16x112x112xf32>
    %v71 = stablehlo.reduce(%v67 init: %v68) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x16x112x112xf32>, tensor<f32>) -> tensor<64x16xf32>
    %v72 = stablehlo.broadcast_in_dim %v71, dims = [0, 1] : (tensor<64x16xf32>) -> tensor<64x16x112x112xf32>
    %v73 = stablehlo.divide %v72, %v69 : tensor<64x16x112x112xf32>
    %v74 = stablehlo.subtract %v67, %v73 : tensor<64x16x112x112xf32>
    %v75 = stablehlo.multiply %v74, %v74 : tensor<64x16x112x112xf32>
    %v76 = stablehlo.reduce(%v75 init: %v68) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x16x112x112xf32>, tensor<f32>) -> tensor<64x16xf32>
    %v77 = stablehlo.broadcast_in_dim %v76, dims = [0, 1] : (tensor<64x16xf32>) -> tensor<64x16x112x112xf32>
    %v78 = stablehlo.divide %v77, %v69 : tensor<64x16x112x112xf32>
    %v79 = stablehlo.add %v78, %v70 : tensor<64x16x112x112xf32>
    %v80 = stablehlo.rsqrt %v79 : tensor<64x16x112x112xf32>
    %v81 = stablehlo.multiply %v74, %v80 : tensor<64x16x112x112xf32>
    %v82 = stablehlo.broadcast_in_dim %gp1, dims = [1] : (tensor<16xf32>) -> tensor<64x16x112x112xf32>
    %v83 = stablehlo.broadcast_in_dim %btp1, dims = [1] : (tensor<16xf32>) -> tensor<64x16x112x112xf32>
    %v84 = stablehlo.multiply %v81, %v82 : tensor<64x16x112x112xf32>
    %v85 = stablehlo.add %v84, %v83 : tensor<64x16x112x112xf32>
    %v86 = stablehlo.reshape %v85 : (tensor<64x16x112x112xf32>) -> tensor<64x200704xf32>
    %v87 = stablehlo.reshape %v86 : (tensor<64x200704xf32>) -> tensor<64x16x112x112xf32>
    %v88 = stablehlo.convolution(%v87, %We2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x16x112x112xf32>, tensor<96x16x1x1xf32>) -> tensor<64x96x112x112xf32>
    %v89 = stablehlo.broadcast_in_dim %zb96, dims = [1] : (tensor<96xf32>) -> tensor<64x96x112x112xf32>
    %v90 = stablehlo.add %v88, %v89 : tensor<64x96x112x112xf32>
    %v91 = stablehlo.reshape %v90 : (tensor<64x96x112x112xf32>) -> tensor<64x1204224xf32>
    %v92 = stablehlo.reshape %v91 : (tensor<64x1204224xf32>) -> tensor<64x96x112x112xf32>
    %v93 = stablehlo.constant dense<0.0> : tensor<f32>
    %v94 = stablehlo.constant dense<12544.0> : tensor<64x96x112x112xf32>
    %v95 = stablehlo.constant dense<1.0e-5> : tensor<64x96x112x112xf32>
    %v96 = stablehlo.reduce(%v92 init: %v93) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x96x112x112xf32>, tensor<f32>) -> tensor<64x96xf32>
    %v97 = stablehlo.broadcast_in_dim %v96, dims = [0, 1] : (tensor<64x96xf32>) -> tensor<64x96x112x112xf32>
    %v98 = stablehlo.divide %v97, %v94 : tensor<64x96x112x112xf32>
    %v99 = stablehlo.subtract %v92, %v98 : tensor<64x96x112x112xf32>
    %v100 = stablehlo.multiply %v99, %v99 : tensor<64x96x112x112xf32>
    %v101 = stablehlo.reduce(%v100 init: %v93) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x96x112x112xf32>, tensor<f32>) -> tensor<64x96xf32>
    %v102 = stablehlo.broadcast_in_dim %v101, dims = [0, 1] : (tensor<64x96xf32>) -> tensor<64x96x112x112xf32>
    %v103 = stablehlo.divide %v102, %v94 : tensor<64x96x112x112xf32>
    %v104 = stablehlo.add %v103, %v95 : tensor<64x96x112x112xf32>
    %v105 = stablehlo.rsqrt %v104 : tensor<64x96x112x112xf32>
    %v106 = stablehlo.multiply %v99, %v105 : tensor<64x96x112x112xf32>
    %v107 = stablehlo.broadcast_in_dim %ge2, dims = [1] : (tensor<96xf32>) -> tensor<64x96x112x112xf32>
    %v108 = stablehlo.broadcast_in_dim %bte2, dims = [1] : (tensor<96xf32>) -> tensor<64x96x112x112xf32>
    %v109 = stablehlo.multiply %v106, %v107 : tensor<64x96x112x112xf32>
    %v110 = stablehlo.add %v109, %v108 : tensor<64x96x112x112xf32>
    %v111 = stablehlo.reshape %v110 : (tensor<64x96x112x112xf32>) -> tensor<64x1204224xf32>
    %v112 = stablehlo.reshape %v111 : (tensor<64x1204224xf32>) -> tensor<64x96x112x112xf32>
    %v113 = stablehlo.constant dense<0.0> : tensor<64x96x112x112xf32>
    %v114 = stablehlo.constant dense<6.0> : tensor<64x96x112x112xf32>
    %v115 = stablehlo.maximum %v112, %v113 : tensor<64x96x112x112xf32>
    %v116 = stablehlo.minimum %v115, %v114 : tensor<64x96x112x112xf32>
    %v117 = stablehlo.reshape %v116 : (tensor<64x96x112x112xf32>) -> tensor<64x1204224xf32>
    %v118 = stablehlo.reshape %v117 : (tensor<64x1204224xf32>) -> tensor<64x96x112x112xf32>
    %v119 = stablehlo.convolution(%v118, %Wd2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<64x96x112x112xf32>, tensor<96x1x3x3xf32>) -> tensor<64x96x56x56xf32>
    %v120 = stablehlo.broadcast_in_dim %zb96, dims = [1] : (tensor<96xf32>) -> tensor<64x96x56x56xf32>
    %v121 = stablehlo.add %v119, %v120 : tensor<64x96x56x56xf32>
    %v122 = stablehlo.reshape %v121 : (tensor<64x96x56x56xf32>) -> tensor<64x301056xf32>
    %v123 = stablehlo.reshape %v122 : (tensor<64x301056xf32>) -> tensor<64x96x56x56xf32>
    %v124 = stablehlo.constant dense<0.0> : tensor<f32>
    %v125 = stablehlo.constant dense<3136.0> : tensor<64x96x56x56xf32>
    %v126 = stablehlo.constant dense<1.0e-5> : tensor<64x96x56x56xf32>
    %v127 = stablehlo.reduce(%v123 init: %v124) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x96x56x56xf32>, tensor<f32>) -> tensor<64x96xf32>
    %v128 = stablehlo.broadcast_in_dim %v127, dims = [0, 1] : (tensor<64x96xf32>) -> tensor<64x96x56x56xf32>
    %v129 = stablehlo.divide %v128, %v125 : tensor<64x96x56x56xf32>
    %v130 = stablehlo.subtract %v123, %v129 : tensor<64x96x56x56xf32>
    %v131 = stablehlo.multiply %v130, %v130 : tensor<64x96x56x56xf32>
    %v132 = stablehlo.reduce(%v131 init: %v124) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x96x56x56xf32>, tensor<f32>) -> tensor<64x96xf32>
    %v133 = stablehlo.broadcast_in_dim %v132, dims = [0, 1] : (tensor<64x96xf32>) -> tensor<64x96x56x56xf32>
    %v134 = stablehlo.divide %v133, %v125 : tensor<64x96x56x56xf32>
    %v135 = stablehlo.add %v134, %v126 : tensor<64x96x56x56xf32>
    %v136 = stablehlo.rsqrt %v135 : tensor<64x96x56x56xf32>
    %v137 = stablehlo.multiply %v130, %v136 : tensor<64x96x56x56xf32>
    %v138 = stablehlo.broadcast_in_dim %gd2, dims = [1] : (tensor<96xf32>) -> tensor<64x96x56x56xf32>
    %v139 = stablehlo.broadcast_in_dim %btd2, dims = [1] : (tensor<96xf32>) -> tensor<64x96x56x56xf32>
    %v140 = stablehlo.multiply %v137, %v138 : tensor<64x96x56x56xf32>
    %v141 = stablehlo.add %v140, %v139 : tensor<64x96x56x56xf32>
    %v142 = stablehlo.reshape %v141 : (tensor<64x96x56x56xf32>) -> tensor<64x301056xf32>
    %v143 = stablehlo.reshape %v142 : (tensor<64x301056xf32>) -> tensor<64x96x56x56xf32>
    %v144 = stablehlo.constant dense<0.0> : tensor<64x96x56x56xf32>
    %v145 = stablehlo.constant dense<6.0> : tensor<64x96x56x56xf32>
    %v146 = stablehlo.maximum %v143, %v144 : tensor<64x96x56x56xf32>
    %v147 = stablehlo.minimum %v146, %v145 : tensor<64x96x56x56xf32>
    %v148 = stablehlo.reshape %v147 : (tensor<64x96x56x56xf32>) -> tensor<64x301056xf32>
    %v149 = stablehlo.reshape %v148 : (tensor<64x301056xf32>) -> tensor<64x96x56x56xf32>
    %v150 = stablehlo.convolution(%v149, %Wp2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x96x56x56xf32>, tensor<24x96x1x1xf32>) -> tensor<64x24x56x56xf32>
    %v151 = stablehlo.broadcast_in_dim %zb24, dims = [1] : (tensor<24xf32>) -> tensor<64x24x56x56xf32>
    %v152 = stablehlo.add %v150, %v151 : tensor<64x24x56x56xf32>
    %v153 = stablehlo.reshape %v152 : (tensor<64x24x56x56xf32>) -> tensor<64x75264xf32>
    %v154 = stablehlo.reshape %v153 : (tensor<64x75264xf32>) -> tensor<64x24x56x56xf32>
    %v155 = stablehlo.constant dense<0.0> : tensor<f32>
    %v156 = stablehlo.constant dense<3136.0> : tensor<64x24x56x56xf32>
    %v157 = stablehlo.constant dense<1.0e-5> : tensor<64x24x56x56xf32>
    %v158 = stablehlo.reduce(%v154 init: %v155) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x24x56x56xf32>, tensor<f32>) -> tensor<64x24xf32>
    %v159 = stablehlo.broadcast_in_dim %v158, dims = [0, 1] : (tensor<64x24xf32>) -> tensor<64x24x56x56xf32>
    %v160 = stablehlo.divide %v159, %v156 : tensor<64x24x56x56xf32>
    %v161 = stablehlo.subtract %v154, %v160 : tensor<64x24x56x56xf32>
    %v162 = stablehlo.multiply %v161, %v161 : tensor<64x24x56x56xf32>
    %v163 = stablehlo.reduce(%v162 init: %v155) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x24x56x56xf32>, tensor<f32>) -> tensor<64x24xf32>
    %v164 = stablehlo.broadcast_in_dim %v163, dims = [0, 1] : (tensor<64x24xf32>) -> tensor<64x24x56x56xf32>
    %v165 = stablehlo.divide %v164, %v156 : tensor<64x24x56x56xf32>
    %v166 = stablehlo.add %v165, %v157 : tensor<64x24x56x56xf32>
    %v167 = stablehlo.rsqrt %v166 : tensor<64x24x56x56xf32>
    %v168 = stablehlo.multiply %v161, %v167 : tensor<64x24x56x56xf32>
    %v169 = stablehlo.broadcast_in_dim %gp2, dims = [1] : (tensor<24xf32>) -> tensor<64x24x56x56xf32>
    %v170 = stablehlo.broadcast_in_dim %btp2, dims = [1] : (tensor<24xf32>) -> tensor<64x24x56x56xf32>
    %v171 = stablehlo.multiply %v168, %v169 : tensor<64x24x56x56xf32>
    %v172 = stablehlo.add %v171, %v170 : tensor<64x24x56x56xf32>
    %v173 = stablehlo.reshape %v172 : (tensor<64x24x56x56xf32>) -> tensor<64x75264xf32>
    %v174 = stablehlo.reshape %v173 : (tensor<64x75264xf32>) -> tensor<64x24x56x56xf32>
    %v175 = stablehlo.convolution(%v174, %We3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x24x56x56xf32>, tensor<144x24x1x1xf32>) -> tensor<64x144x56x56xf32>
    %v176 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<64x144x56x56xf32>
    %v177 = stablehlo.add %v175, %v176 : tensor<64x144x56x56xf32>
    %v178 = stablehlo.reshape %v177 : (tensor<64x144x56x56xf32>) -> tensor<64x451584xf32>
    %v179 = stablehlo.reshape %v178 : (tensor<64x451584xf32>) -> tensor<64x144x56x56xf32>
    %v180 = stablehlo.constant dense<0.0> : tensor<f32>
    %v181 = stablehlo.constant dense<3136.0> : tensor<64x144x56x56xf32>
    %v182 = stablehlo.constant dense<1.0e-5> : tensor<64x144x56x56xf32>
    %v183 = stablehlo.reduce(%v179 init: %v180) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x144x56x56xf32>, tensor<f32>) -> tensor<64x144xf32>
    %v184 = stablehlo.broadcast_in_dim %v183, dims = [0, 1] : (tensor<64x144xf32>) -> tensor<64x144x56x56xf32>
    %v185 = stablehlo.divide %v184, %v181 : tensor<64x144x56x56xf32>
    %v186 = stablehlo.subtract %v179, %v185 : tensor<64x144x56x56xf32>
    %v187 = stablehlo.multiply %v186, %v186 : tensor<64x144x56x56xf32>
    %v188 = stablehlo.reduce(%v187 init: %v180) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x144x56x56xf32>, tensor<f32>) -> tensor<64x144xf32>
    %v189 = stablehlo.broadcast_in_dim %v188, dims = [0, 1] : (tensor<64x144xf32>) -> tensor<64x144x56x56xf32>
    %v190 = stablehlo.divide %v189, %v181 : tensor<64x144x56x56xf32>
    %v191 = stablehlo.add %v190, %v182 : tensor<64x144x56x56xf32>
    %v192 = stablehlo.rsqrt %v191 : tensor<64x144x56x56xf32>
    %v193 = stablehlo.multiply %v186, %v192 : tensor<64x144x56x56xf32>
    %v194 = stablehlo.broadcast_in_dim %ge3, dims = [1] : (tensor<144xf32>) -> tensor<64x144x56x56xf32>
    %v195 = stablehlo.broadcast_in_dim %bte3, dims = [1] : (tensor<144xf32>) -> tensor<64x144x56x56xf32>
    %v196 = stablehlo.multiply %v193, %v194 : tensor<64x144x56x56xf32>
    %v197 = stablehlo.add %v196, %v195 : tensor<64x144x56x56xf32>
    %v198 = stablehlo.reshape %v197 : (tensor<64x144x56x56xf32>) -> tensor<64x451584xf32>
    %v199 = stablehlo.reshape %v198 : (tensor<64x451584xf32>) -> tensor<64x144x56x56xf32>
    %v200 = stablehlo.constant dense<0.0> : tensor<64x144x56x56xf32>
    %v201 = stablehlo.constant dense<6.0> : tensor<64x144x56x56xf32>
    %v202 = stablehlo.maximum %v199, %v200 : tensor<64x144x56x56xf32>
    %v203 = stablehlo.minimum %v202, %v201 : tensor<64x144x56x56xf32>
    %v204 = stablehlo.reshape %v203 : (tensor<64x144x56x56xf32>) -> tensor<64x451584xf32>
    %v205 = stablehlo.reshape %v204 : (tensor<64x451584xf32>) -> tensor<64x144x56x56xf32>
    %v206 = stablehlo.convolution(%v205, %Wd3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 144 : i64} : (tensor<64x144x56x56xf32>, tensor<144x1x3x3xf32>) -> tensor<64x144x56x56xf32>
    %v207 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<64x144x56x56xf32>
    %v208 = stablehlo.add %v206, %v207 : tensor<64x144x56x56xf32>
    %v209 = stablehlo.reshape %v208 : (tensor<64x144x56x56xf32>) -> tensor<64x451584xf32>
    %v210 = stablehlo.reshape %v209 : (tensor<64x451584xf32>) -> tensor<64x144x56x56xf32>
    %v211 = stablehlo.constant dense<0.0> : tensor<f32>
    %v212 = stablehlo.constant dense<3136.0> : tensor<64x144x56x56xf32>
    %v213 = stablehlo.constant dense<1.0e-5> : tensor<64x144x56x56xf32>
    %v214 = stablehlo.reduce(%v210 init: %v211) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x144x56x56xf32>, tensor<f32>) -> tensor<64x144xf32>
    %v215 = stablehlo.broadcast_in_dim %v214, dims = [0, 1] : (tensor<64x144xf32>) -> tensor<64x144x56x56xf32>
    %v216 = stablehlo.divide %v215, %v212 : tensor<64x144x56x56xf32>
    %v217 = stablehlo.subtract %v210, %v216 : tensor<64x144x56x56xf32>
    %v218 = stablehlo.multiply %v217, %v217 : tensor<64x144x56x56xf32>
    %v219 = stablehlo.reduce(%v218 init: %v211) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x144x56x56xf32>, tensor<f32>) -> tensor<64x144xf32>
    %v220 = stablehlo.broadcast_in_dim %v219, dims = [0, 1] : (tensor<64x144xf32>) -> tensor<64x144x56x56xf32>
    %v221 = stablehlo.divide %v220, %v212 : tensor<64x144x56x56xf32>
    %v222 = stablehlo.add %v221, %v213 : tensor<64x144x56x56xf32>
    %v223 = stablehlo.rsqrt %v222 : tensor<64x144x56x56xf32>
    %v224 = stablehlo.multiply %v217, %v223 : tensor<64x144x56x56xf32>
    %v225 = stablehlo.broadcast_in_dim %gd3, dims = [1] : (tensor<144xf32>) -> tensor<64x144x56x56xf32>
    %v226 = stablehlo.broadcast_in_dim %btd3, dims = [1] : (tensor<144xf32>) -> tensor<64x144x56x56xf32>
    %v227 = stablehlo.multiply %v224, %v225 : tensor<64x144x56x56xf32>
    %v228 = stablehlo.add %v227, %v226 : tensor<64x144x56x56xf32>
    %v229 = stablehlo.reshape %v228 : (tensor<64x144x56x56xf32>) -> tensor<64x451584xf32>
    %v230 = stablehlo.reshape %v229 : (tensor<64x451584xf32>) -> tensor<64x144x56x56xf32>
    %v231 = stablehlo.constant dense<0.0> : tensor<64x144x56x56xf32>
    %v232 = stablehlo.constant dense<6.0> : tensor<64x144x56x56xf32>
    %v233 = stablehlo.maximum %v230, %v231 : tensor<64x144x56x56xf32>
    %v234 = stablehlo.minimum %v233, %v232 : tensor<64x144x56x56xf32>
    %v235 = stablehlo.reshape %v234 : (tensor<64x144x56x56xf32>) -> tensor<64x451584xf32>
    %v236 = stablehlo.reshape %v235 : (tensor<64x451584xf32>) -> tensor<64x144x56x56xf32>
    %v237 = stablehlo.convolution(%v236, %Wp3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x144x56x56xf32>, tensor<24x144x1x1xf32>) -> tensor<64x24x56x56xf32>
    %v238 = stablehlo.broadcast_in_dim %zb24, dims = [1] : (tensor<24xf32>) -> tensor<64x24x56x56xf32>
    %v239 = stablehlo.add %v237, %v238 : tensor<64x24x56x56xf32>
    %v240 = stablehlo.reshape %v239 : (tensor<64x24x56x56xf32>) -> tensor<64x75264xf32>
    %v241 = stablehlo.reshape %v240 : (tensor<64x75264xf32>) -> tensor<64x24x56x56xf32>
    %v242 = stablehlo.constant dense<0.0> : tensor<f32>
    %v243 = stablehlo.constant dense<3136.0> : tensor<64x24x56x56xf32>
    %v244 = stablehlo.constant dense<1.0e-5> : tensor<64x24x56x56xf32>
    %v245 = stablehlo.reduce(%v241 init: %v242) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x24x56x56xf32>, tensor<f32>) -> tensor<64x24xf32>
    %v246 = stablehlo.broadcast_in_dim %v245, dims = [0, 1] : (tensor<64x24xf32>) -> tensor<64x24x56x56xf32>
    %v247 = stablehlo.divide %v246, %v243 : tensor<64x24x56x56xf32>
    %v248 = stablehlo.subtract %v241, %v247 : tensor<64x24x56x56xf32>
    %v249 = stablehlo.multiply %v248, %v248 : tensor<64x24x56x56xf32>
    %v250 = stablehlo.reduce(%v249 init: %v242) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x24x56x56xf32>, tensor<f32>) -> tensor<64x24xf32>
    %v251 = stablehlo.broadcast_in_dim %v250, dims = [0, 1] : (tensor<64x24xf32>) -> tensor<64x24x56x56xf32>
    %v252 = stablehlo.divide %v251, %v243 : tensor<64x24x56x56xf32>
    %v253 = stablehlo.add %v252, %v244 : tensor<64x24x56x56xf32>
    %v254 = stablehlo.rsqrt %v253 : tensor<64x24x56x56xf32>
    %v255 = stablehlo.multiply %v248, %v254 : tensor<64x24x56x56xf32>
    %v256 = stablehlo.broadcast_in_dim %gp3, dims = [1] : (tensor<24xf32>) -> tensor<64x24x56x56xf32>
    %v257 = stablehlo.broadcast_in_dim %btp3, dims = [1] : (tensor<24xf32>) -> tensor<64x24x56x56xf32>
    %v258 = stablehlo.multiply %v255, %v256 : tensor<64x24x56x56xf32>
    %v259 = stablehlo.add %v258, %v257 : tensor<64x24x56x56xf32>
    %v260 = stablehlo.reshape %v259 : (tensor<64x24x56x56xf32>) -> tensor<64x75264xf32>
    %v261 = stablehlo.reshape %v260 : (tensor<64x75264xf32>) -> tensor<64x24x56x56xf32>
    %v262 = stablehlo.reshape %v173 : (tensor<64x75264xf32>) -> tensor<64x24x56x56xf32>
    %v263 = stablehlo.add %v261, %v262 : tensor<64x24x56x56xf32>
    %v264 = stablehlo.reshape %v263 : (tensor<64x24x56x56xf32>) -> tensor<64x75264xf32>
    %v265 = stablehlo.reshape %v264 : (tensor<64x75264xf32>) -> tensor<64x24x56x56xf32>
    %v266 = stablehlo.convolution(%v265, %We4)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x24x56x56xf32>, tensor<144x24x1x1xf32>) -> tensor<64x144x56x56xf32>
    %v267 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<64x144x56x56xf32>
    %v268 = stablehlo.add %v266, %v267 : tensor<64x144x56x56xf32>
    %v269 = stablehlo.reshape %v268 : (tensor<64x144x56x56xf32>) -> tensor<64x451584xf32>
    %v270 = stablehlo.reshape %v269 : (tensor<64x451584xf32>) -> tensor<64x144x56x56xf32>
    %v271 = stablehlo.constant dense<0.0> : tensor<f32>
    %v272 = stablehlo.constant dense<3136.0> : tensor<64x144x56x56xf32>
    %v273 = stablehlo.constant dense<1.0e-5> : tensor<64x144x56x56xf32>
    %v274 = stablehlo.reduce(%v270 init: %v271) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x144x56x56xf32>, tensor<f32>) -> tensor<64x144xf32>
    %v275 = stablehlo.broadcast_in_dim %v274, dims = [0, 1] : (tensor<64x144xf32>) -> tensor<64x144x56x56xf32>
    %v276 = stablehlo.divide %v275, %v272 : tensor<64x144x56x56xf32>
    %v277 = stablehlo.subtract %v270, %v276 : tensor<64x144x56x56xf32>
    %v278 = stablehlo.multiply %v277, %v277 : tensor<64x144x56x56xf32>
    %v279 = stablehlo.reduce(%v278 init: %v271) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x144x56x56xf32>, tensor<f32>) -> tensor<64x144xf32>
    %v280 = stablehlo.broadcast_in_dim %v279, dims = [0, 1] : (tensor<64x144xf32>) -> tensor<64x144x56x56xf32>
    %v281 = stablehlo.divide %v280, %v272 : tensor<64x144x56x56xf32>
    %v282 = stablehlo.add %v281, %v273 : tensor<64x144x56x56xf32>
    %v283 = stablehlo.rsqrt %v282 : tensor<64x144x56x56xf32>
    %v284 = stablehlo.multiply %v277, %v283 : tensor<64x144x56x56xf32>
    %v285 = stablehlo.broadcast_in_dim %ge4, dims = [1] : (tensor<144xf32>) -> tensor<64x144x56x56xf32>
    %v286 = stablehlo.broadcast_in_dim %bte4, dims = [1] : (tensor<144xf32>) -> tensor<64x144x56x56xf32>
    %v287 = stablehlo.multiply %v284, %v285 : tensor<64x144x56x56xf32>
    %v288 = stablehlo.add %v287, %v286 : tensor<64x144x56x56xf32>
    %v289 = stablehlo.reshape %v288 : (tensor<64x144x56x56xf32>) -> tensor<64x451584xf32>
    %v290 = stablehlo.reshape %v289 : (tensor<64x451584xf32>) -> tensor<64x144x56x56xf32>
    %v291 = stablehlo.constant dense<0.0> : tensor<64x144x56x56xf32>
    %v292 = stablehlo.constant dense<6.0> : tensor<64x144x56x56xf32>
    %v293 = stablehlo.maximum %v290, %v291 : tensor<64x144x56x56xf32>
    %v294 = stablehlo.minimum %v293, %v292 : tensor<64x144x56x56xf32>
    %v295 = stablehlo.reshape %v294 : (tensor<64x144x56x56xf32>) -> tensor<64x451584xf32>
    %v296 = stablehlo.reshape %v295 : (tensor<64x451584xf32>) -> tensor<64x144x56x56xf32>
    %v297 = stablehlo.convolution(%v296, %Wd4)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 144 : i64} : (tensor<64x144x56x56xf32>, tensor<144x1x3x3xf32>) -> tensor<64x144x28x28xf32>
    %v298 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<64x144x28x28xf32>
    %v299 = stablehlo.add %v297, %v298 : tensor<64x144x28x28xf32>
    %v300 = stablehlo.reshape %v299 : (tensor<64x144x28x28xf32>) -> tensor<64x112896xf32>
    %v301 = stablehlo.reshape %v300 : (tensor<64x112896xf32>) -> tensor<64x144x28x28xf32>
    %v302 = stablehlo.constant dense<0.0> : tensor<f32>
    %v303 = stablehlo.constant dense<784.0> : tensor<64x144x28x28xf32>
    %v304 = stablehlo.constant dense<1.0e-5> : tensor<64x144x28x28xf32>
    %v305 = stablehlo.reduce(%v301 init: %v302) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x144x28x28xf32>, tensor<f32>) -> tensor<64x144xf32>
    %v306 = stablehlo.broadcast_in_dim %v305, dims = [0, 1] : (tensor<64x144xf32>) -> tensor<64x144x28x28xf32>
    %v307 = stablehlo.divide %v306, %v303 : tensor<64x144x28x28xf32>
    %v308 = stablehlo.subtract %v301, %v307 : tensor<64x144x28x28xf32>
    %v309 = stablehlo.multiply %v308, %v308 : tensor<64x144x28x28xf32>
    %v310 = stablehlo.reduce(%v309 init: %v302) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x144x28x28xf32>, tensor<f32>) -> tensor<64x144xf32>
    %v311 = stablehlo.broadcast_in_dim %v310, dims = [0, 1] : (tensor<64x144xf32>) -> tensor<64x144x28x28xf32>
    %v312 = stablehlo.divide %v311, %v303 : tensor<64x144x28x28xf32>
    %v313 = stablehlo.add %v312, %v304 : tensor<64x144x28x28xf32>
    %v314 = stablehlo.rsqrt %v313 : tensor<64x144x28x28xf32>
    %v315 = stablehlo.multiply %v308, %v314 : tensor<64x144x28x28xf32>
    %v316 = stablehlo.broadcast_in_dim %gd4, dims = [1] : (tensor<144xf32>) -> tensor<64x144x28x28xf32>
    %v317 = stablehlo.broadcast_in_dim %btd4, dims = [1] : (tensor<144xf32>) -> tensor<64x144x28x28xf32>
    %v318 = stablehlo.multiply %v315, %v316 : tensor<64x144x28x28xf32>
    %v319 = stablehlo.add %v318, %v317 : tensor<64x144x28x28xf32>
    %v320 = stablehlo.reshape %v319 : (tensor<64x144x28x28xf32>) -> tensor<64x112896xf32>
    %v321 = stablehlo.reshape %v320 : (tensor<64x112896xf32>) -> tensor<64x144x28x28xf32>
    %v322 = stablehlo.constant dense<0.0> : tensor<64x144x28x28xf32>
    %v323 = stablehlo.constant dense<6.0> : tensor<64x144x28x28xf32>
    %v324 = stablehlo.maximum %v321, %v322 : tensor<64x144x28x28xf32>
    %v325 = stablehlo.minimum %v324, %v323 : tensor<64x144x28x28xf32>
    %v326 = stablehlo.reshape %v325 : (tensor<64x144x28x28xf32>) -> tensor<64x112896xf32>
    %v327 = stablehlo.reshape %v326 : (tensor<64x112896xf32>) -> tensor<64x144x28x28xf32>
    %v328 = stablehlo.convolution(%v327, %Wp4)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x144x28x28xf32>, tensor<32x144x1x1xf32>) -> tensor<64x32x28x28xf32>
    %v329 = stablehlo.broadcast_in_dim %zb32, dims = [1] : (tensor<32xf32>) -> tensor<64x32x28x28xf32>
    %v330 = stablehlo.add %v328, %v329 : tensor<64x32x28x28xf32>
    %v331 = stablehlo.reshape %v330 : (tensor<64x32x28x28xf32>) -> tensor<64x25088xf32>
    %v332 = stablehlo.reshape %v331 : (tensor<64x25088xf32>) -> tensor<64x32x28x28xf32>
    %v333 = stablehlo.constant dense<0.0> : tensor<f32>
    %v334 = stablehlo.constant dense<784.0> : tensor<64x32x28x28xf32>
    %v335 = stablehlo.constant dense<1.0e-5> : tensor<64x32x28x28xf32>
    %v336 = stablehlo.reduce(%v332 init: %v333) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x32x28x28xf32>, tensor<f32>) -> tensor<64x32xf32>
    %v337 = stablehlo.broadcast_in_dim %v336, dims = [0, 1] : (tensor<64x32xf32>) -> tensor<64x32x28x28xf32>
    %v338 = stablehlo.divide %v337, %v334 : tensor<64x32x28x28xf32>
    %v339 = stablehlo.subtract %v332, %v338 : tensor<64x32x28x28xf32>
    %v340 = stablehlo.multiply %v339, %v339 : tensor<64x32x28x28xf32>
    %v341 = stablehlo.reduce(%v340 init: %v333) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x32x28x28xf32>, tensor<f32>) -> tensor<64x32xf32>
    %v342 = stablehlo.broadcast_in_dim %v341, dims = [0, 1] : (tensor<64x32xf32>) -> tensor<64x32x28x28xf32>
    %v343 = stablehlo.divide %v342, %v334 : tensor<64x32x28x28xf32>
    %v344 = stablehlo.add %v343, %v335 : tensor<64x32x28x28xf32>
    %v345 = stablehlo.rsqrt %v344 : tensor<64x32x28x28xf32>
    %v346 = stablehlo.multiply %v339, %v345 : tensor<64x32x28x28xf32>
    %v347 = stablehlo.broadcast_in_dim %gp4, dims = [1] : (tensor<32xf32>) -> tensor<64x32x28x28xf32>
    %v348 = stablehlo.broadcast_in_dim %btp4, dims = [1] : (tensor<32xf32>) -> tensor<64x32x28x28xf32>
    %v349 = stablehlo.multiply %v346, %v347 : tensor<64x32x28x28xf32>
    %v350 = stablehlo.add %v349, %v348 : tensor<64x32x28x28xf32>
    %v351 = stablehlo.reshape %v350 : (tensor<64x32x28x28xf32>) -> tensor<64x25088xf32>
    %v352 = stablehlo.reshape %v351 : (tensor<64x25088xf32>) -> tensor<64x32x28x28xf32>
    %v353 = stablehlo.convolution(%v352, %We5)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x28x28xf32>, tensor<192x32x1x1xf32>) -> tensor<64x192x28x28xf32>
    %v354 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<64x192x28x28xf32>
    %v355 = stablehlo.add %v353, %v354 : tensor<64x192x28x28xf32>
    %v356 = stablehlo.reshape %v355 : (tensor<64x192x28x28xf32>) -> tensor<64x150528xf32>
    %v357 = stablehlo.reshape %v356 : (tensor<64x150528xf32>) -> tensor<64x192x28x28xf32>
    %v358 = stablehlo.constant dense<0.0> : tensor<f32>
    %v359 = stablehlo.constant dense<784.0> : tensor<64x192x28x28xf32>
    %v360 = stablehlo.constant dense<1.0e-5> : tensor<64x192x28x28xf32>
    %v361 = stablehlo.reduce(%v357 init: %v358) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x192x28x28xf32>, tensor<f32>) -> tensor<64x192xf32>
    %v362 = stablehlo.broadcast_in_dim %v361, dims = [0, 1] : (tensor<64x192xf32>) -> tensor<64x192x28x28xf32>
    %v363 = stablehlo.divide %v362, %v359 : tensor<64x192x28x28xf32>
    %v364 = stablehlo.subtract %v357, %v363 : tensor<64x192x28x28xf32>
    %v365 = stablehlo.multiply %v364, %v364 : tensor<64x192x28x28xf32>
    %v366 = stablehlo.reduce(%v365 init: %v358) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x192x28x28xf32>, tensor<f32>) -> tensor<64x192xf32>
    %v367 = stablehlo.broadcast_in_dim %v366, dims = [0, 1] : (tensor<64x192xf32>) -> tensor<64x192x28x28xf32>
    %v368 = stablehlo.divide %v367, %v359 : tensor<64x192x28x28xf32>
    %v369 = stablehlo.add %v368, %v360 : tensor<64x192x28x28xf32>
    %v370 = stablehlo.rsqrt %v369 : tensor<64x192x28x28xf32>
    %v371 = stablehlo.multiply %v364, %v370 : tensor<64x192x28x28xf32>
    %v372 = stablehlo.broadcast_in_dim %ge5, dims = [1] : (tensor<192xf32>) -> tensor<64x192x28x28xf32>
    %v373 = stablehlo.broadcast_in_dim %bte5, dims = [1] : (tensor<192xf32>) -> tensor<64x192x28x28xf32>
    %v374 = stablehlo.multiply %v371, %v372 : tensor<64x192x28x28xf32>
    %v375 = stablehlo.add %v374, %v373 : tensor<64x192x28x28xf32>
    %v376 = stablehlo.reshape %v375 : (tensor<64x192x28x28xf32>) -> tensor<64x150528xf32>
    %v377 = stablehlo.reshape %v376 : (tensor<64x150528xf32>) -> tensor<64x192x28x28xf32>
    %v378 = stablehlo.constant dense<0.0> : tensor<64x192x28x28xf32>
    %v379 = stablehlo.constant dense<6.0> : tensor<64x192x28x28xf32>
    %v380 = stablehlo.maximum %v377, %v378 : tensor<64x192x28x28xf32>
    %v381 = stablehlo.minimum %v380, %v379 : tensor<64x192x28x28xf32>
    %v382 = stablehlo.reshape %v381 : (tensor<64x192x28x28xf32>) -> tensor<64x150528xf32>
    %v383 = stablehlo.reshape %v382 : (tensor<64x150528xf32>) -> tensor<64x192x28x28xf32>
    %v384 = stablehlo.convolution(%v383, %Wd5)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<64x192x28x28xf32>, tensor<192x1x3x3xf32>) -> tensor<64x192x28x28xf32>
    %v385 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<64x192x28x28xf32>
    %v386 = stablehlo.add %v384, %v385 : tensor<64x192x28x28xf32>
    %v387 = stablehlo.reshape %v386 : (tensor<64x192x28x28xf32>) -> tensor<64x150528xf32>
    %v388 = stablehlo.reshape %v387 : (tensor<64x150528xf32>) -> tensor<64x192x28x28xf32>
    %v389 = stablehlo.constant dense<0.0> : tensor<f32>
    %v390 = stablehlo.constant dense<784.0> : tensor<64x192x28x28xf32>
    %v391 = stablehlo.constant dense<1.0e-5> : tensor<64x192x28x28xf32>
    %v392 = stablehlo.reduce(%v388 init: %v389) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x192x28x28xf32>, tensor<f32>) -> tensor<64x192xf32>
    %v393 = stablehlo.broadcast_in_dim %v392, dims = [0, 1] : (tensor<64x192xf32>) -> tensor<64x192x28x28xf32>
    %v394 = stablehlo.divide %v393, %v390 : tensor<64x192x28x28xf32>
    %v395 = stablehlo.subtract %v388, %v394 : tensor<64x192x28x28xf32>
    %v396 = stablehlo.multiply %v395, %v395 : tensor<64x192x28x28xf32>
    %v397 = stablehlo.reduce(%v396 init: %v389) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x192x28x28xf32>, tensor<f32>) -> tensor<64x192xf32>
    %v398 = stablehlo.broadcast_in_dim %v397, dims = [0, 1] : (tensor<64x192xf32>) -> tensor<64x192x28x28xf32>
    %v399 = stablehlo.divide %v398, %v390 : tensor<64x192x28x28xf32>
    %v400 = stablehlo.add %v399, %v391 : tensor<64x192x28x28xf32>
    %v401 = stablehlo.rsqrt %v400 : tensor<64x192x28x28xf32>
    %v402 = stablehlo.multiply %v395, %v401 : tensor<64x192x28x28xf32>
    %v403 = stablehlo.broadcast_in_dim %gd5, dims = [1] : (tensor<192xf32>) -> tensor<64x192x28x28xf32>
    %v404 = stablehlo.broadcast_in_dim %btd5, dims = [1] : (tensor<192xf32>) -> tensor<64x192x28x28xf32>
    %v405 = stablehlo.multiply %v402, %v403 : tensor<64x192x28x28xf32>
    %v406 = stablehlo.add %v405, %v404 : tensor<64x192x28x28xf32>
    %v407 = stablehlo.reshape %v406 : (tensor<64x192x28x28xf32>) -> tensor<64x150528xf32>
    %v408 = stablehlo.reshape %v407 : (tensor<64x150528xf32>) -> tensor<64x192x28x28xf32>
    %v409 = stablehlo.constant dense<0.0> : tensor<64x192x28x28xf32>
    %v410 = stablehlo.constant dense<6.0> : tensor<64x192x28x28xf32>
    %v411 = stablehlo.maximum %v408, %v409 : tensor<64x192x28x28xf32>
    %v412 = stablehlo.minimum %v411, %v410 : tensor<64x192x28x28xf32>
    %v413 = stablehlo.reshape %v412 : (tensor<64x192x28x28xf32>) -> tensor<64x150528xf32>
    %v414 = stablehlo.reshape %v413 : (tensor<64x150528xf32>) -> tensor<64x192x28x28xf32>
    %v415 = stablehlo.convolution(%v414, %Wp5)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x192x28x28xf32>, tensor<32x192x1x1xf32>) -> tensor<64x32x28x28xf32>
    %v416 = stablehlo.broadcast_in_dim %zb32, dims = [1] : (tensor<32xf32>) -> tensor<64x32x28x28xf32>
    %v417 = stablehlo.add %v415, %v416 : tensor<64x32x28x28xf32>
    %v418 = stablehlo.reshape %v417 : (tensor<64x32x28x28xf32>) -> tensor<64x25088xf32>
    %v419 = stablehlo.reshape %v418 : (tensor<64x25088xf32>) -> tensor<64x32x28x28xf32>
    %v420 = stablehlo.constant dense<0.0> : tensor<f32>
    %v421 = stablehlo.constant dense<784.0> : tensor<64x32x28x28xf32>
    %v422 = stablehlo.constant dense<1.0e-5> : tensor<64x32x28x28xf32>
    %v423 = stablehlo.reduce(%v419 init: %v420) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x32x28x28xf32>, tensor<f32>) -> tensor<64x32xf32>
    %v424 = stablehlo.broadcast_in_dim %v423, dims = [0, 1] : (tensor<64x32xf32>) -> tensor<64x32x28x28xf32>
    %v425 = stablehlo.divide %v424, %v421 : tensor<64x32x28x28xf32>
    %v426 = stablehlo.subtract %v419, %v425 : tensor<64x32x28x28xf32>
    %v427 = stablehlo.multiply %v426, %v426 : tensor<64x32x28x28xf32>
    %v428 = stablehlo.reduce(%v427 init: %v420) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x32x28x28xf32>, tensor<f32>) -> tensor<64x32xf32>
    %v429 = stablehlo.broadcast_in_dim %v428, dims = [0, 1] : (tensor<64x32xf32>) -> tensor<64x32x28x28xf32>
    %v430 = stablehlo.divide %v429, %v421 : tensor<64x32x28x28xf32>
    %v431 = stablehlo.add %v430, %v422 : tensor<64x32x28x28xf32>
    %v432 = stablehlo.rsqrt %v431 : tensor<64x32x28x28xf32>
    %v433 = stablehlo.multiply %v426, %v432 : tensor<64x32x28x28xf32>
    %v434 = stablehlo.broadcast_in_dim %gp5, dims = [1] : (tensor<32xf32>) -> tensor<64x32x28x28xf32>
    %v435 = stablehlo.broadcast_in_dim %btp5, dims = [1] : (tensor<32xf32>) -> tensor<64x32x28x28xf32>
    %v436 = stablehlo.multiply %v433, %v434 : tensor<64x32x28x28xf32>
    %v437 = stablehlo.add %v436, %v435 : tensor<64x32x28x28xf32>
    %v438 = stablehlo.reshape %v437 : (tensor<64x32x28x28xf32>) -> tensor<64x25088xf32>
    %v439 = stablehlo.reshape %v438 : (tensor<64x25088xf32>) -> tensor<64x32x28x28xf32>
    %v440 = stablehlo.reshape %v351 : (tensor<64x25088xf32>) -> tensor<64x32x28x28xf32>
    %v441 = stablehlo.add %v439, %v440 : tensor<64x32x28x28xf32>
    %v442 = stablehlo.reshape %v441 : (tensor<64x32x28x28xf32>) -> tensor<64x25088xf32>
    %v443 = stablehlo.reshape %v442 : (tensor<64x25088xf32>) -> tensor<64x32x28x28xf32>
    %v444 = stablehlo.convolution(%v443, %We6)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x28x28xf32>, tensor<192x32x1x1xf32>) -> tensor<64x192x28x28xf32>
    %v445 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<64x192x28x28xf32>
    %v446 = stablehlo.add %v444, %v445 : tensor<64x192x28x28xf32>
    %v447 = stablehlo.reshape %v446 : (tensor<64x192x28x28xf32>) -> tensor<64x150528xf32>
    %v448 = stablehlo.reshape %v447 : (tensor<64x150528xf32>) -> tensor<64x192x28x28xf32>
    %v449 = stablehlo.constant dense<0.0> : tensor<f32>
    %v450 = stablehlo.constant dense<784.0> : tensor<64x192x28x28xf32>
    %v451 = stablehlo.constant dense<1.0e-5> : tensor<64x192x28x28xf32>
    %v452 = stablehlo.reduce(%v448 init: %v449) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x192x28x28xf32>, tensor<f32>) -> tensor<64x192xf32>
    %v453 = stablehlo.broadcast_in_dim %v452, dims = [0, 1] : (tensor<64x192xf32>) -> tensor<64x192x28x28xf32>
    %v454 = stablehlo.divide %v453, %v450 : tensor<64x192x28x28xf32>
    %v455 = stablehlo.subtract %v448, %v454 : tensor<64x192x28x28xf32>
    %v456 = stablehlo.multiply %v455, %v455 : tensor<64x192x28x28xf32>
    %v457 = stablehlo.reduce(%v456 init: %v449) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x192x28x28xf32>, tensor<f32>) -> tensor<64x192xf32>
    %v458 = stablehlo.broadcast_in_dim %v457, dims = [0, 1] : (tensor<64x192xf32>) -> tensor<64x192x28x28xf32>
    %v459 = stablehlo.divide %v458, %v450 : tensor<64x192x28x28xf32>
    %v460 = stablehlo.add %v459, %v451 : tensor<64x192x28x28xf32>
    %v461 = stablehlo.rsqrt %v460 : tensor<64x192x28x28xf32>
    %v462 = stablehlo.multiply %v455, %v461 : tensor<64x192x28x28xf32>
    %v463 = stablehlo.broadcast_in_dim %ge6, dims = [1] : (tensor<192xf32>) -> tensor<64x192x28x28xf32>
    %v464 = stablehlo.broadcast_in_dim %bte6, dims = [1] : (tensor<192xf32>) -> tensor<64x192x28x28xf32>
    %v465 = stablehlo.multiply %v462, %v463 : tensor<64x192x28x28xf32>
    %v466 = stablehlo.add %v465, %v464 : tensor<64x192x28x28xf32>
    %v467 = stablehlo.reshape %v466 : (tensor<64x192x28x28xf32>) -> tensor<64x150528xf32>
    %v468 = stablehlo.reshape %v467 : (tensor<64x150528xf32>) -> tensor<64x192x28x28xf32>
    %v469 = stablehlo.constant dense<0.0> : tensor<64x192x28x28xf32>
    %v470 = stablehlo.constant dense<6.0> : tensor<64x192x28x28xf32>
    %v471 = stablehlo.maximum %v468, %v469 : tensor<64x192x28x28xf32>
    %v472 = stablehlo.minimum %v471, %v470 : tensor<64x192x28x28xf32>
    %v473 = stablehlo.reshape %v472 : (tensor<64x192x28x28xf32>) -> tensor<64x150528xf32>
    %v474 = stablehlo.reshape %v473 : (tensor<64x150528xf32>) -> tensor<64x192x28x28xf32>
    %v475 = stablehlo.convolution(%v474, %Wd6)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<64x192x28x28xf32>, tensor<192x1x3x3xf32>) -> tensor<64x192x28x28xf32>
    %v476 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<64x192x28x28xf32>
    %v477 = stablehlo.add %v475, %v476 : tensor<64x192x28x28xf32>
    %v478 = stablehlo.reshape %v477 : (tensor<64x192x28x28xf32>) -> tensor<64x150528xf32>
    %v479 = stablehlo.reshape %v478 : (tensor<64x150528xf32>) -> tensor<64x192x28x28xf32>
    %v480 = stablehlo.constant dense<0.0> : tensor<f32>
    %v481 = stablehlo.constant dense<784.0> : tensor<64x192x28x28xf32>
    %v482 = stablehlo.constant dense<1.0e-5> : tensor<64x192x28x28xf32>
    %v483 = stablehlo.reduce(%v479 init: %v480) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x192x28x28xf32>, tensor<f32>) -> tensor<64x192xf32>
    %v484 = stablehlo.broadcast_in_dim %v483, dims = [0, 1] : (tensor<64x192xf32>) -> tensor<64x192x28x28xf32>
    %v485 = stablehlo.divide %v484, %v481 : tensor<64x192x28x28xf32>
    %v486 = stablehlo.subtract %v479, %v485 : tensor<64x192x28x28xf32>
    %v487 = stablehlo.multiply %v486, %v486 : tensor<64x192x28x28xf32>
    %v488 = stablehlo.reduce(%v487 init: %v480) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x192x28x28xf32>, tensor<f32>) -> tensor<64x192xf32>
    %v489 = stablehlo.broadcast_in_dim %v488, dims = [0, 1] : (tensor<64x192xf32>) -> tensor<64x192x28x28xf32>
    %v490 = stablehlo.divide %v489, %v481 : tensor<64x192x28x28xf32>
    %v491 = stablehlo.add %v490, %v482 : tensor<64x192x28x28xf32>
    %v492 = stablehlo.rsqrt %v491 : tensor<64x192x28x28xf32>
    %v493 = stablehlo.multiply %v486, %v492 : tensor<64x192x28x28xf32>
    %v494 = stablehlo.broadcast_in_dim %gd6, dims = [1] : (tensor<192xf32>) -> tensor<64x192x28x28xf32>
    %v495 = stablehlo.broadcast_in_dim %btd6, dims = [1] : (tensor<192xf32>) -> tensor<64x192x28x28xf32>
    %v496 = stablehlo.multiply %v493, %v494 : tensor<64x192x28x28xf32>
    %v497 = stablehlo.add %v496, %v495 : tensor<64x192x28x28xf32>
    %v498 = stablehlo.reshape %v497 : (tensor<64x192x28x28xf32>) -> tensor<64x150528xf32>
    %v499 = stablehlo.reshape %v498 : (tensor<64x150528xf32>) -> tensor<64x192x28x28xf32>
    %v500 = stablehlo.constant dense<0.0> : tensor<64x192x28x28xf32>
    %v501 = stablehlo.constant dense<6.0> : tensor<64x192x28x28xf32>
    %v502 = stablehlo.maximum %v499, %v500 : tensor<64x192x28x28xf32>
    %v503 = stablehlo.minimum %v502, %v501 : tensor<64x192x28x28xf32>
    %v504 = stablehlo.reshape %v503 : (tensor<64x192x28x28xf32>) -> tensor<64x150528xf32>
    %v505 = stablehlo.reshape %v504 : (tensor<64x150528xf32>) -> tensor<64x192x28x28xf32>
    %v506 = stablehlo.convolution(%v505, %Wp6)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x192x28x28xf32>, tensor<32x192x1x1xf32>) -> tensor<64x32x28x28xf32>
    %v507 = stablehlo.broadcast_in_dim %zb32, dims = [1] : (tensor<32xf32>) -> tensor<64x32x28x28xf32>
    %v508 = stablehlo.add %v506, %v507 : tensor<64x32x28x28xf32>
    %v509 = stablehlo.reshape %v508 : (tensor<64x32x28x28xf32>) -> tensor<64x25088xf32>
    %v510 = stablehlo.reshape %v509 : (tensor<64x25088xf32>) -> tensor<64x32x28x28xf32>
    %v511 = stablehlo.constant dense<0.0> : tensor<f32>
    %v512 = stablehlo.constant dense<784.0> : tensor<64x32x28x28xf32>
    %v513 = stablehlo.constant dense<1.0e-5> : tensor<64x32x28x28xf32>
    %v514 = stablehlo.reduce(%v510 init: %v511) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x32x28x28xf32>, tensor<f32>) -> tensor<64x32xf32>
    %v515 = stablehlo.broadcast_in_dim %v514, dims = [0, 1] : (tensor<64x32xf32>) -> tensor<64x32x28x28xf32>
    %v516 = stablehlo.divide %v515, %v512 : tensor<64x32x28x28xf32>
    %v517 = stablehlo.subtract %v510, %v516 : tensor<64x32x28x28xf32>
    %v518 = stablehlo.multiply %v517, %v517 : tensor<64x32x28x28xf32>
    %v519 = stablehlo.reduce(%v518 init: %v511) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x32x28x28xf32>, tensor<f32>) -> tensor<64x32xf32>
    %v520 = stablehlo.broadcast_in_dim %v519, dims = [0, 1] : (tensor<64x32xf32>) -> tensor<64x32x28x28xf32>
    %v521 = stablehlo.divide %v520, %v512 : tensor<64x32x28x28xf32>
    %v522 = stablehlo.add %v521, %v513 : tensor<64x32x28x28xf32>
    %v523 = stablehlo.rsqrt %v522 : tensor<64x32x28x28xf32>
    %v524 = stablehlo.multiply %v517, %v523 : tensor<64x32x28x28xf32>
    %v525 = stablehlo.broadcast_in_dim %gp6, dims = [1] : (tensor<32xf32>) -> tensor<64x32x28x28xf32>
    %v526 = stablehlo.broadcast_in_dim %btp6, dims = [1] : (tensor<32xf32>) -> tensor<64x32x28x28xf32>
    %v527 = stablehlo.multiply %v524, %v525 : tensor<64x32x28x28xf32>
    %v528 = stablehlo.add %v527, %v526 : tensor<64x32x28x28xf32>
    %v529 = stablehlo.reshape %v528 : (tensor<64x32x28x28xf32>) -> tensor<64x25088xf32>
    %v530 = stablehlo.reshape %v529 : (tensor<64x25088xf32>) -> tensor<64x32x28x28xf32>
    %v531 = stablehlo.reshape %v442 : (tensor<64x25088xf32>) -> tensor<64x32x28x28xf32>
    %v532 = stablehlo.add %v530, %v531 : tensor<64x32x28x28xf32>
    %v533 = stablehlo.reshape %v532 : (tensor<64x32x28x28xf32>) -> tensor<64x25088xf32>
    %v534 = stablehlo.reshape %v533 : (tensor<64x25088xf32>) -> tensor<64x32x28x28xf32>
    %v535 = stablehlo.convolution(%v534, %We7)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x28x28xf32>, tensor<192x32x1x1xf32>) -> tensor<64x192x28x28xf32>
    %v536 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<64x192x28x28xf32>
    %v537 = stablehlo.add %v535, %v536 : tensor<64x192x28x28xf32>
    %v538 = stablehlo.reshape %v537 : (tensor<64x192x28x28xf32>) -> tensor<64x150528xf32>
    %v539 = stablehlo.reshape %v538 : (tensor<64x150528xf32>) -> tensor<64x192x28x28xf32>
    %v540 = stablehlo.constant dense<0.0> : tensor<f32>
    %v541 = stablehlo.constant dense<784.0> : tensor<64x192x28x28xf32>
    %v542 = stablehlo.constant dense<1.0e-5> : tensor<64x192x28x28xf32>
    %v543 = stablehlo.reduce(%v539 init: %v540) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x192x28x28xf32>, tensor<f32>) -> tensor<64x192xf32>
    %v544 = stablehlo.broadcast_in_dim %v543, dims = [0, 1] : (tensor<64x192xf32>) -> tensor<64x192x28x28xf32>
    %v545 = stablehlo.divide %v544, %v541 : tensor<64x192x28x28xf32>
    %v546 = stablehlo.subtract %v539, %v545 : tensor<64x192x28x28xf32>
    %v547 = stablehlo.multiply %v546, %v546 : tensor<64x192x28x28xf32>
    %v548 = stablehlo.reduce(%v547 init: %v540) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x192x28x28xf32>, tensor<f32>) -> tensor<64x192xf32>
    %v549 = stablehlo.broadcast_in_dim %v548, dims = [0, 1] : (tensor<64x192xf32>) -> tensor<64x192x28x28xf32>
    %v550 = stablehlo.divide %v549, %v541 : tensor<64x192x28x28xf32>
    %v551 = stablehlo.add %v550, %v542 : tensor<64x192x28x28xf32>
    %v552 = stablehlo.rsqrt %v551 : tensor<64x192x28x28xf32>
    %v553 = stablehlo.multiply %v546, %v552 : tensor<64x192x28x28xf32>
    %v554 = stablehlo.broadcast_in_dim %ge7, dims = [1] : (tensor<192xf32>) -> tensor<64x192x28x28xf32>
    %v555 = stablehlo.broadcast_in_dim %bte7, dims = [1] : (tensor<192xf32>) -> tensor<64x192x28x28xf32>
    %v556 = stablehlo.multiply %v553, %v554 : tensor<64x192x28x28xf32>
    %v557 = stablehlo.add %v556, %v555 : tensor<64x192x28x28xf32>
    %v558 = stablehlo.reshape %v557 : (tensor<64x192x28x28xf32>) -> tensor<64x150528xf32>
    %v559 = stablehlo.reshape %v558 : (tensor<64x150528xf32>) -> tensor<64x192x28x28xf32>
    %v560 = stablehlo.constant dense<0.0> : tensor<64x192x28x28xf32>
    %v561 = stablehlo.constant dense<6.0> : tensor<64x192x28x28xf32>
    %v562 = stablehlo.maximum %v559, %v560 : tensor<64x192x28x28xf32>
    %v563 = stablehlo.minimum %v562, %v561 : tensor<64x192x28x28xf32>
    %v564 = stablehlo.reshape %v563 : (tensor<64x192x28x28xf32>) -> tensor<64x150528xf32>
    %v565 = stablehlo.reshape %v564 : (tensor<64x150528xf32>) -> tensor<64x192x28x28xf32>
    %v566 = stablehlo.convolution(%v565, %Wd7)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<64x192x28x28xf32>, tensor<192x1x3x3xf32>) -> tensor<64x192x14x14xf32>
    %v567 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<64x192x14x14xf32>
    %v568 = stablehlo.add %v566, %v567 : tensor<64x192x14x14xf32>
    %v569 = stablehlo.reshape %v568 : (tensor<64x192x14x14xf32>) -> tensor<64x37632xf32>
    %v570 = stablehlo.reshape %v569 : (tensor<64x37632xf32>) -> tensor<64x192x14x14xf32>
    %v571 = stablehlo.constant dense<0.0> : tensor<f32>
    %v572 = stablehlo.constant dense<196.0> : tensor<64x192x14x14xf32>
    %v573 = stablehlo.constant dense<1.0e-5> : tensor<64x192x14x14xf32>
    %v574 = stablehlo.reduce(%v570 init: %v571) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x192x14x14xf32>, tensor<f32>) -> tensor<64x192xf32>
    %v575 = stablehlo.broadcast_in_dim %v574, dims = [0, 1] : (tensor<64x192xf32>) -> tensor<64x192x14x14xf32>
    %v576 = stablehlo.divide %v575, %v572 : tensor<64x192x14x14xf32>
    %v577 = stablehlo.subtract %v570, %v576 : tensor<64x192x14x14xf32>
    %v578 = stablehlo.multiply %v577, %v577 : tensor<64x192x14x14xf32>
    %v579 = stablehlo.reduce(%v578 init: %v571) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x192x14x14xf32>, tensor<f32>) -> tensor<64x192xf32>
    %v580 = stablehlo.broadcast_in_dim %v579, dims = [0, 1] : (tensor<64x192xf32>) -> tensor<64x192x14x14xf32>
    %v581 = stablehlo.divide %v580, %v572 : tensor<64x192x14x14xf32>
    %v582 = stablehlo.add %v581, %v573 : tensor<64x192x14x14xf32>
    %v583 = stablehlo.rsqrt %v582 : tensor<64x192x14x14xf32>
    %v584 = stablehlo.multiply %v577, %v583 : tensor<64x192x14x14xf32>
    %v585 = stablehlo.broadcast_in_dim %gd7, dims = [1] : (tensor<192xf32>) -> tensor<64x192x14x14xf32>
    %v586 = stablehlo.broadcast_in_dim %btd7, dims = [1] : (tensor<192xf32>) -> tensor<64x192x14x14xf32>
    %v587 = stablehlo.multiply %v584, %v585 : tensor<64x192x14x14xf32>
    %v588 = stablehlo.add %v587, %v586 : tensor<64x192x14x14xf32>
    %v589 = stablehlo.reshape %v588 : (tensor<64x192x14x14xf32>) -> tensor<64x37632xf32>
    %v590 = stablehlo.reshape %v589 : (tensor<64x37632xf32>) -> tensor<64x192x14x14xf32>
    %v591 = stablehlo.constant dense<0.0> : tensor<64x192x14x14xf32>
    %v592 = stablehlo.constant dense<6.0> : tensor<64x192x14x14xf32>
    %v593 = stablehlo.maximum %v590, %v591 : tensor<64x192x14x14xf32>
    %v594 = stablehlo.minimum %v593, %v592 : tensor<64x192x14x14xf32>
    %v595 = stablehlo.reshape %v594 : (tensor<64x192x14x14xf32>) -> tensor<64x37632xf32>
    %v596 = stablehlo.reshape %v595 : (tensor<64x37632xf32>) -> tensor<64x192x14x14xf32>
    %v597 = stablehlo.convolution(%v596, %Wp7)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x192x14x14xf32>, tensor<64x192x1x1xf32>) -> tensor<64x64x14x14xf32>
    %v598 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<64x64x14x14xf32>
    %v599 = stablehlo.add %v597, %v598 : tensor<64x64x14x14xf32>
    %v600 = stablehlo.reshape %v599 : (tensor<64x64x14x14xf32>) -> tensor<64x12544xf32>
    %v601 = stablehlo.reshape %v600 : (tensor<64x12544xf32>) -> tensor<64x64x14x14xf32>
    %v602 = stablehlo.constant dense<0.0> : tensor<f32>
    %v603 = stablehlo.constant dense<196.0> : tensor<64x64x14x14xf32>
    %v604 = stablehlo.constant dense<1.0e-5> : tensor<64x64x14x14xf32>
    %v605 = stablehlo.reduce(%v601 init: %v602) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x64x14x14xf32>, tensor<f32>) -> tensor<64x64xf32>
    %v606 = stablehlo.broadcast_in_dim %v605, dims = [0, 1] : (tensor<64x64xf32>) -> tensor<64x64x14x14xf32>
    %v607 = stablehlo.divide %v606, %v603 : tensor<64x64x14x14xf32>
    %v608 = stablehlo.subtract %v601, %v607 : tensor<64x64x14x14xf32>
    %v609 = stablehlo.multiply %v608, %v608 : tensor<64x64x14x14xf32>
    %v610 = stablehlo.reduce(%v609 init: %v602) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x64x14x14xf32>, tensor<f32>) -> tensor<64x64xf32>
    %v611 = stablehlo.broadcast_in_dim %v610, dims = [0, 1] : (tensor<64x64xf32>) -> tensor<64x64x14x14xf32>
    %v612 = stablehlo.divide %v611, %v603 : tensor<64x64x14x14xf32>
    %v613 = stablehlo.add %v612, %v604 : tensor<64x64x14x14xf32>
    %v614 = stablehlo.rsqrt %v613 : tensor<64x64x14x14xf32>
    %v615 = stablehlo.multiply %v608, %v614 : tensor<64x64x14x14xf32>
    %v616 = stablehlo.broadcast_in_dim %gp7, dims = [1] : (tensor<64xf32>) -> tensor<64x64x14x14xf32>
    %v617 = stablehlo.broadcast_in_dim %btp7, dims = [1] : (tensor<64xf32>) -> tensor<64x64x14x14xf32>
    %v618 = stablehlo.multiply %v615, %v616 : tensor<64x64x14x14xf32>
    %v619 = stablehlo.add %v618, %v617 : tensor<64x64x14x14xf32>
    %v620 = stablehlo.reshape %v619 : (tensor<64x64x14x14xf32>) -> tensor<64x12544xf32>
    %v621 = stablehlo.reshape %v620 : (tensor<64x12544xf32>) -> tensor<64x64x14x14xf32>
    %v622 = stablehlo.convolution(%v621, %We8)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x14x14xf32>, tensor<384x64x1x1xf32>) -> tensor<64x384x14x14xf32>
    %v623 = stablehlo.broadcast_in_dim %zb384, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v624 = stablehlo.add %v622, %v623 : tensor<64x384x14x14xf32>
    %v625 = stablehlo.reshape %v624 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v626 = stablehlo.reshape %v625 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v627 = stablehlo.constant dense<0.0> : tensor<f32>
    %v628 = stablehlo.constant dense<196.0> : tensor<64x384x14x14xf32>
    %v629 = stablehlo.constant dense<1.0e-5> : tensor<64x384x14x14xf32>
    %v630 = stablehlo.reduce(%v626 init: %v627) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x384x14x14xf32>, tensor<f32>) -> tensor<64x384xf32>
    %v631 = stablehlo.broadcast_in_dim %v630, dims = [0, 1] : (tensor<64x384xf32>) -> tensor<64x384x14x14xf32>
    %v632 = stablehlo.divide %v631, %v628 : tensor<64x384x14x14xf32>
    %v633 = stablehlo.subtract %v626, %v632 : tensor<64x384x14x14xf32>
    %v634 = stablehlo.multiply %v633, %v633 : tensor<64x384x14x14xf32>
    %v635 = stablehlo.reduce(%v634 init: %v627) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x384x14x14xf32>, tensor<f32>) -> tensor<64x384xf32>
    %v636 = stablehlo.broadcast_in_dim %v635, dims = [0, 1] : (tensor<64x384xf32>) -> tensor<64x384x14x14xf32>
    %v637 = stablehlo.divide %v636, %v628 : tensor<64x384x14x14xf32>
    %v638 = stablehlo.add %v637, %v629 : tensor<64x384x14x14xf32>
    %v639 = stablehlo.rsqrt %v638 : tensor<64x384x14x14xf32>
    %v640 = stablehlo.multiply %v633, %v639 : tensor<64x384x14x14xf32>
    %v641 = stablehlo.broadcast_in_dim %ge8, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v642 = stablehlo.broadcast_in_dim %bte8, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v643 = stablehlo.multiply %v640, %v641 : tensor<64x384x14x14xf32>
    %v644 = stablehlo.add %v643, %v642 : tensor<64x384x14x14xf32>
    %v645 = stablehlo.reshape %v644 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v646 = stablehlo.reshape %v645 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v647 = stablehlo.constant dense<0.0> : tensor<64x384x14x14xf32>
    %v648 = stablehlo.constant dense<6.0> : tensor<64x384x14x14xf32>
    %v649 = stablehlo.maximum %v646, %v647 : tensor<64x384x14x14xf32>
    %v650 = stablehlo.minimum %v649, %v648 : tensor<64x384x14x14xf32>
    %v651 = stablehlo.reshape %v650 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v652 = stablehlo.reshape %v651 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v653 = stablehlo.convolution(%v652, %Wd8)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<64x384x14x14xf32>, tensor<384x1x3x3xf32>) -> tensor<64x384x14x14xf32>
    %v654 = stablehlo.broadcast_in_dim %zb384, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v655 = stablehlo.add %v653, %v654 : tensor<64x384x14x14xf32>
    %v656 = stablehlo.reshape %v655 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v657 = stablehlo.reshape %v656 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v658 = stablehlo.constant dense<0.0> : tensor<f32>
    %v659 = stablehlo.constant dense<196.0> : tensor<64x384x14x14xf32>
    %v660 = stablehlo.constant dense<1.0e-5> : tensor<64x384x14x14xf32>
    %v661 = stablehlo.reduce(%v657 init: %v658) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x384x14x14xf32>, tensor<f32>) -> tensor<64x384xf32>
    %v662 = stablehlo.broadcast_in_dim %v661, dims = [0, 1] : (tensor<64x384xf32>) -> tensor<64x384x14x14xf32>
    %v663 = stablehlo.divide %v662, %v659 : tensor<64x384x14x14xf32>
    %v664 = stablehlo.subtract %v657, %v663 : tensor<64x384x14x14xf32>
    %v665 = stablehlo.multiply %v664, %v664 : tensor<64x384x14x14xf32>
    %v666 = stablehlo.reduce(%v665 init: %v658) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x384x14x14xf32>, tensor<f32>) -> tensor<64x384xf32>
    %v667 = stablehlo.broadcast_in_dim %v666, dims = [0, 1] : (tensor<64x384xf32>) -> tensor<64x384x14x14xf32>
    %v668 = stablehlo.divide %v667, %v659 : tensor<64x384x14x14xf32>
    %v669 = stablehlo.add %v668, %v660 : tensor<64x384x14x14xf32>
    %v670 = stablehlo.rsqrt %v669 : tensor<64x384x14x14xf32>
    %v671 = stablehlo.multiply %v664, %v670 : tensor<64x384x14x14xf32>
    %v672 = stablehlo.broadcast_in_dim %gd8, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v673 = stablehlo.broadcast_in_dim %btd8, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v674 = stablehlo.multiply %v671, %v672 : tensor<64x384x14x14xf32>
    %v675 = stablehlo.add %v674, %v673 : tensor<64x384x14x14xf32>
    %v676 = stablehlo.reshape %v675 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v677 = stablehlo.reshape %v676 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v678 = stablehlo.constant dense<0.0> : tensor<64x384x14x14xf32>
    %v679 = stablehlo.constant dense<6.0> : tensor<64x384x14x14xf32>
    %v680 = stablehlo.maximum %v677, %v678 : tensor<64x384x14x14xf32>
    %v681 = stablehlo.minimum %v680, %v679 : tensor<64x384x14x14xf32>
    %v682 = stablehlo.reshape %v681 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v683 = stablehlo.reshape %v682 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v684 = stablehlo.convolution(%v683, %Wp8)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x384x14x14xf32>, tensor<64x384x1x1xf32>) -> tensor<64x64x14x14xf32>
    %v685 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<64x64x14x14xf32>
    %v686 = stablehlo.add %v684, %v685 : tensor<64x64x14x14xf32>
    %v687 = stablehlo.reshape %v686 : (tensor<64x64x14x14xf32>) -> tensor<64x12544xf32>
    %v688 = stablehlo.reshape %v687 : (tensor<64x12544xf32>) -> tensor<64x64x14x14xf32>
    %v689 = stablehlo.constant dense<0.0> : tensor<f32>
    %v690 = stablehlo.constant dense<196.0> : tensor<64x64x14x14xf32>
    %v691 = stablehlo.constant dense<1.0e-5> : tensor<64x64x14x14xf32>
    %v692 = stablehlo.reduce(%v688 init: %v689) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x64x14x14xf32>, tensor<f32>) -> tensor<64x64xf32>
    %v693 = stablehlo.broadcast_in_dim %v692, dims = [0, 1] : (tensor<64x64xf32>) -> tensor<64x64x14x14xf32>
    %v694 = stablehlo.divide %v693, %v690 : tensor<64x64x14x14xf32>
    %v695 = stablehlo.subtract %v688, %v694 : tensor<64x64x14x14xf32>
    %v696 = stablehlo.multiply %v695, %v695 : tensor<64x64x14x14xf32>
    %v697 = stablehlo.reduce(%v696 init: %v689) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x64x14x14xf32>, tensor<f32>) -> tensor<64x64xf32>
    %v698 = stablehlo.broadcast_in_dim %v697, dims = [0, 1] : (tensor<64x64xf32>) -> tensor<64x64x14x14xf32>
    %v699 = stablehlo.divide %v698, %v690 : tensor<64x64x14x14xf32>
    %v700 = stablehlo.add %v699, %v691 : tensor<64x64x14x14xf32>
    %v701 = stablehlo.rsqrt %v700 : tensor<64x64x14x14xf32>
    %v702 = stablehlo.multiply %v695, %v701 : tensor<64x64x14x14xf32>
    %v703 = stablehlo.broadcast_in_dim %gp8, dims = [1] : (tensor<64xf32>) -> tensor<64x64x14x14xf32>
    %v704 = stablehlo.broadcast_in_dim %btp8, dims = [1] : (tensor<64xf32>) -> tensor<64x64x14x14xf32>
    %v705 = stablehlo.multiply %v702, %v703 : tensor<64x64x14x14xf32>
    %v706 = stablehlo.add %v705, %v704 : tensor<64x64x14x14xf32>
    %v707 = stablehlo.reshape %v706 : (tensor<64x64x14x14xf32>) -> tensor<64x12544xf32>
    %v708 = stablehlo.reshape %v707 : (tensor<64x12544xf32>) -> tensor<64x64x14x14xf32>
    %v709 = stablehlo.reshape %v620 : (tensor<64x12544xf32>) -> tensor<64x64x14x14xf32>
    %v710 = stablehlo.add %v708, %v709 : tensor<64x64x14x14xf32>
    %v711 = stablehlo.reshape %v710 : (tensor<64x64x14x14xf32>) -> tensor<64x12544xf32>
    %v712 = stablehlo.reshape %v711 : (tensor<64x12544xf32>) -> tensor<64x64x14x14xf32>
    %v713 = stablehlo.convolution(%v712, %We9)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x14x14xf32>, tensor<384x64x1x1xf32>) -> tensor<64x384x14x14xf32>
    %v714 = stablehlo.broadcast_in_dim %zb384, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v715 = stablehlo.add %v713, %v714 : tensor<64x384x14x14xf32>
    %v716 = stablehlo.reshape %v715 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v717 = stablehlo.reshape %v716 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v718 = stablehlo.constant dense<0.0> : tensor<f32>
    %v719 = stablehlo.constant dense<196.0> : tensor<64x384x14x14xf32>
    %v720 = stablehlo.constant dense<1.0e-5> : tensor<64x384x14x14xf32>
    %v721 = stablehlo.reduce(%v717 init: %v718) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x384x14x14xf32>, tensor<f32>) -> tensor<64x384xf32>
    %v722 = stablehlo.broadcast_in_dim %v721, dims = [0, 1] : (tensor<64x384xf32>) -> tensor<64x384x14x14xf32>
    %v723 = stablehlo.divide %v722, %v719 : tensor<64x384x14x14xf32>
    %v724 = stablehlo.subtract %v717, %v723 : tensor<64x384x14x14xf32>
    %v725 = stablehlo.multiply %v724, %v724 : tensor<64x384x14x14xf32>
    %v726 = stablehlo.reduce(%v725 init: %v718) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x384x14x14xf32>, tensor<f32>) -> tensor<64x384xf32>
    %v727 = stablehlo.broadcast_in_dim %v726, dims = [0, 1] : (tensor<64x384xf32>) -> tensor<64x384x14x14xf32>
    %v728 = stablehlo.divide %v727, %v719 : tensor<64x384x14x14xf32>
    %v729 = stablehlo.add %v728, %v720 : tensor<64x384x14x14xf32>
    %v730 = stablehlo.rsqrt %v729 : tensor<64x384x14x14xf32>
    %v731 = stablehlo.multiply %v724, %v730 : tensor<64x384x14x14xf32>
    %v732 = stablehlo.broadcast_in_dim %ge9, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v733 = stablehlo.broadcast_in_dim %bte9, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v734 = stablehlo.multiply %v731, %v732 : tensor<64x384x14x14xf32>
    %v735 = stablehlo.add %v734, %v733 : tensor<64x384x14x14xf32>
    %v736 = stablehlo.reshape %v735 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v737 = stablehlo.reshape %v736 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v738 = stablehlo.constant dense<0.0> : tensor<64x384x14x14xf32>
    %v739 = stablehlo.constant dense<6.0> : tensor<64x384x14x14xf32>
    %v740 = stablehlo.maximum %v737, %v738 : tensor<64x384x14x14xf32>
    %v741 = stablehlo.minimum %v740, %v739 : tensor<64x384x14x14xf32>
    %v742 = stablehlo.reshape %v741 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v743 = stablehlo.reshape %v742 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v744 = stablehlo.convolution(%v743, %Wd9)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<64x384x14x14xf32>, tensor<384x1x3x3xf32>) -> tensor<64x384x14x14xf32>
    %v745 = stablehlo.broadcast_in_dim %zb384, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v746 = stablehlo.add %v744, %v745 : tensor<64x384x14x14xf32>
    %v747 = stablehlo.reshape %v746 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v748 = stablehlo.reshape %v747 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v749 = stablehlo.constant dense<0.0> : tensor<f32>
    %v750 = stablehlo.constant dense<196.0> : tensor<64x384x14x14xf32>
    %v751 = stablehlo.constant dense<1.0e-5> : tensor<64x384x14x14xf32>
    %v752 = stablehlo.reduce(%v748 init: %v749) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x384x14x14xf32>, tensor<f32>) -> tensor<64x384xf32>
    %v753 = stablehlo.broadcast_in_dim %v752, dims = [0, 1] : (tensor<64x384xf32>) -> tensor<64x384x14x14xf32>
    %v754 = stablehlo.divide %v753, %v750 : tensor<64x384x14x14xf32>
    %v755 = stablehlo.subtract %v748, %v754 : tensor<64x384x14x14xf32>
    %v756 = stablehlo.multiply %v755, %v755 : tensor<64x384x14x14xf32>
    %v757 = stablehlo.reduce(%v756 init: %v749) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x384x14x14xf32>, tensor<f32>) -> tensor<64x384xf32>
    %v758 = stablehlo.broadcast_in_dim %v757, dims = [0, 1] : (tensor<64x384xf32>) -> tensor<64x384x14x14xf32>
    %v759 = stablehlo.divide %v758, %v750 : tensor<64x384x14x14xf32>
    %v760 = stablehlo.add %v759, %v751 : tensor<64x384x14x14xf32>
    %v761 = stablehlo.rsqrt %v760 : tensor<64x384x14x14xf32>
    %v762 = stablehlo.multiply %v755, %v761 : tensor<64x384x14x14xf32>
    %v763 = stablehlo.broadcast_in_dim %gd9, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v764 = stablehlo.broadcast_in_dim %btd9, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v765 = stablehlo.multiply %v762, %v763 : tensor<64x384x14x14xf32>
    %v766 = stablehlo.add %v765, %v764 : tensor<64x384x14x14xf32>
    %v767 = stablehlo.reshape %v766 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v768 = stablehlo.reshape %v767 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v769 = stablehlo.constant dense<0.0> : tensor<64x384x14x14xf32>
    %v770 = stablehlo.constant dense<6.0> : tensor<64x384x14x14xf32>
    %v771 = stablehlo.maximum %v768, %v769 : tensor<64x384x14x14xf32>
    %v772 = stablehlo.minimum %v771, %v770 : tensor<64x384x14x14xf32>
    %v773 = stablehlo.reshape %v772 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v774 = stablehlo.reshape %v773 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v775 = stablehlo.convolution(%v774, %Wp9)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x384x14x14xf32>, tensor<64x384x1x1xf32>) -> tensor<64x64x14x14xf32>
    %v776 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<64x64x14x14xf32>
    %v777 = stablehlo.add %v775, %v776 : tensor<64x64x14x14xf32>
    %v778 = stablehlo.reshape %v777 : (tensor<64x64x14x14xf32>) -> tensor<64x12544xf32>
    %v779 = stablehlo.reshape %v778 : (tensor<64x12544xf32>) -> tensor<64x64x14x14xf32>
    %v780 = stablehlo.constant dense<0.0> : tensor<f32>
    %v781 = stablehlo.constant dense<196.0> : tensor<64x64x14x14xf32>
    %v782 = stablehlo.constant dense<1.0e-5> : tensor<64x64x14x14xf32>
    %v783 = stablehlo.reduce(%v779 init: %v780) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x64x14x14xf32>, tensor<f32>) -> tensor<64x64xf32>
    %v784 = stablehlo.broadcast_in_dim %v783, dims = [0, 1] : (tensor<64x64xf32>) -> tensor<64x64x14x14xf32>
    %v785 = stablehlo.divide %v784, %v781 : tensor<64x64x14x14xf32>
    %v786 = stablehlo.subtract %v779, %v785 : tensor<64x64x14x14xf32>
    %v787 = stablehlo.multiply %v786, %v786 : tensor<64x64x14x14xf32>
    %v788 = stablehlo.reduce(%v787 init: %v780) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x64x14x14xf32>, tensor<f32>) -> tensor<64x64xf32>
    %v789 = stablehlo.broadcast_in_dim %v788, dims = [0, 1] : (tensor<64x64xf32>) -> tensor<64x64x14x14xf32>
    %v790 = stablehlo.divide %v789, %v781 : tensor<64x64x14x14xf32>
    %v791 = stablehlo.add %v790, %v782 : tensor<64x64x14x14xf32>
    %v792 = stablehlo.rsqrt %v791 : tensor<64x64x14x14xf32>
    %v793 = stablehlo.multiply %v786, %v792 : tensor<64x64x14x14xf32>
    %v794 = stablehlo.broadcast_in_dim %gp9, dims = [1] : (tensor<64xf32>) -> tensor<64x64x14x14xf32>
    %v795 = stablehlo.broadcast_in_dim %btp9, dims = [1] : (tensor<64xf32>) -> tensor<64x64x14x14xf32>
    %v796 = stablehlo.multiply %v793, %v794 : tensor<64x64x14x14xf32>
    %v797 = stablehlo.add %v796, %v795 : tensor<64x64x14x14xf32>
    %v798 = stablehlo.reshape %v797 : (tensor<64x64x14x14xf32>) -> tensor<64x12544xf32>
    %v799 = stablehlo.reshape %v798 : (tensor<64x12544xf32>) -> tensor<64x64x14x14xf32>
    %v800 = stablehlo.reshape %v711 : (tensor<64x12544xf32>) -> tensor<64x64x14x14xf32>
    %v801 = stablehlo.add %v799, %v800 : tensor<64x64x14x14xf32>
    %v802 = stablehlo.reshape %v801 : (tensor<64x64x14x14xf32>) -> tensor<64x12544xf32>
    %v803 = stablehlo.reshape %v802 : (tensor<64x12544xf32>) -> tensor<64x64x14x14xf32>
    %v804 = stablehlo.convolution(%v803, %We10)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x14x14xf32>, tensor<384x64x1x1xf32>) -> tensor<64x384x14x14xf32>
    %v805 = stablehlo.broadcast_in_dim %zb384, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v806 = stablehlo.add %v804, %v805 : tensor<64x384x14x14xf32>
    %v807 = stablehlo.reshape %v806 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v808 = stablehlo.reshape %v807 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v809 = stablehlo.constant dense<0.0> : tensor<f32>
    %v810 = stablehlo.constant dense<196.0> : tensor<64x384x14x14xf32>
    %v811 = stablehlo.constant dense<1.0e-5> : tensor<64x384x14x14xf32>
    %v812 = stablehlo.reduce(%v808 init: %v809) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x384x14x14xf32>, tensor<f32>) -> tensor<64x384xf32>
    %v813 = stablehlo.broadcast_in_dim %v812, dims = [0, 1] : (tensor<64x384xf32>) -> tensor<64x384x14x14xf32>
    %v814 = stablehlo.divide %v813, %v810 : tensor<64x384x14x14xf32>
    %v815 = stablehlo.subtract %v808, %v814 : tensor<64x384x14x14xf32>
    %v816 = stablehlo.multiply %v815, %v815 : tensor<64x384x14x14xf32>
    %v817 = stablehlo.reduce(%v816 init: %v809) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x384x14x14xf32>, tensor<f32>) -> tensor<64x384xf32>
    %v818 = stablehlo.broadcast_in_dim %v817, dims = [0, 1] : (tensor<64x384xf32>) -> tensor<64x384x14x14xf32>
    %v819 = stablehlo.divide %v818, %v810 : tensor<64x384x14x14xf32>
    %v820 = stablehlo.add %v819, %v811 : tensor<64x384x14x14xf32>
    %v821 = stablehlo.rsqrt %v820 : tensor<64x384x14x14xf32>
    %v822 = stablehlo.multiply %v815, %v821 : tensor<64x384x14x14xf32>
    %v823 = stablehlo.broadcast_in_dim %ge10, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v824 = stablehlo.broadcast_in_dim %bte10, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v825 = stablehlo.multiply %v822, %v823 : tensor<64x384x14x14xf32>
    %v826 = stablehlo.add %v825, %v824 : tensor<64x384x14x14xf32>
    %v827 = stablehlo.reshape %v826 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v828 = stablehlo.reshape %v827 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v829 = stablehlo.constant dense<0.0> : tensor<64x384x14x14xf32>
    %v830 = stablehlo.constant dense<6.0> : tensor<64x384x14x14xf32>
    %v831 = stablehlo.maximum %v828, %v829 : tensor<64x384x14x14xf32>
    %v832 = stablehlo.minimum %v831, %v830 : tensor<64x384x14x14xf32>
    %v833 = stablehlo.reshape %v832 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v834 = stablehlo.reshape %v833 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v835 = stablehlo.convolution(%v834, %Wd10)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<64x384x14x14xf32>, tensor<384x1x3x3xf32>) -> tensor<64x384x14x14xf32>
    %v836 = stablehlo.broadcast_in_dim %zb384, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v837 = stablehlo.add %v835, %v836 : tensor<64x384x14x14xf32>
    %v838 = stablehlo.reshape %v837 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v839 = stablehlo.reshape %v838 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v840 = stablehlo.constant dense<0.0> : tensor<f32>
    %v841 = stablehlo.constant dense<196.0> : tensor<64x384x14x14xf32>
    %v842 = stablehlo.constant dense<1.0e-5> : tensor<64x384x14x14xf32>
    %v843 = stablehlo.reduce(%v839 init: %v840) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x384x14x14xf32>, tensor<f32>) -> tensor<64x384xf32>
    %v844 = stablehlo.broadcast_in_dim %v843, dims = [0, 1] : (tensor<64x384xf32>) -> tensor<64x384x14x14xf32>
    %v845 = stablehlo.divide %v844, %v841 : tensor<64x384x14x14xf32>
    %v846 = stablehlo.subtract %v839, %v845 : tensor<64x384x14x14xf32>
    %v847 = stablehlo.multiply %v846, %v846 : tensor<64x384x14x14xf32>
    %v848 = stablehlo.reduce(%v847 init: %v840) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x384x14x14xf32>, tensor<f32>) -> tensor<64x384xf32>
    %v849 = stablehlo.broadcast_in_dim %v848, dims = [0, 1] : (tensor<64x384xf32>) -> tensor<64x384x14x14xf32>
    %v850 = stablehlo.divide %v849, %v841 : tensor<64x384x14x14xf32>
    %v851 = stablehlo.add %v850, %v842 : tensor<64x384x14x14xf32>
    %v852 = stablehlo.rsqrt %v851 : tensor<64x384x14x14xf32>
    %v853 = stablehlo.multiply %v846, %v852 : tensor<64x384x14x14xf32>
    %v854 = stablehlo.broadcast_in_dim %gd10, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v855 = stablehlo.broadcast_in_dim %btd10, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v856 = stablehlo.multiply %v853, %v854 : tensor<64x384x14x14xf32>
    %v857 = stablehlo.add %v856, %v855 : tensor<64x384x14x14xf32>
    %v858 = stablehlo.reshape %v857 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v859 = stablehlo.reshape %v858 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v860 = stablehlo.constant dense<0.0> : tensor<64x384x14x14xf32>
    %v861 = stablehlo.constant dense<6.0> : tensor<64x384x14x14xf32>
    %v862 = stablehlo.maximum %v859, %v860 : tensor<64x384x14x14xf32>
    %v863 = stablehlo.minimum %v862, %v861 : tensor<64x384x14x14xf32>
    %v864 = stablehlo.reshape %v863 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v865 = stablehlo.reshape %v864 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v866 = stablehlo.convolution(%v865, %Wp10)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x384x14x14xf32>, tensor<64x384x1x1xf32>) -> tensor<64x64x14x14xf32>
    %v867 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<64x64x14x14xf32>
    %v868 = stablehlo.add %v866, %v867 : tensor<64x64x14x14xf32>
    %v869 = stablehlo.reshape %v868 : (tensor<64x64x14x14xf32>) -> tensor<64x12544xf32>
    %v870 = stablehlo.reshape %v869 : (tensor<64x12544xf32>) -> tensor<64x64x14x14xf32>
    %v871 = stablehlo.constant dense<0.0> : tensor<f32>
    %v872 = stablehlo.constant dense<196.0> : tensor<64x64x14x14xf32>
    %v873 = stablehlo.constant dense<1.0e-5> : tensor<64x64x14x14xf32>
    %v874 = stablehlo.reduce(%v870 init: %v871) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x64x14x14xf32>, tensor<f32>) -> tensor<64x64xf32>
    %v875 = stablehlo.broadcast_in_dim %v874, dims = [0, 1] : (tensor<64x64xf32>) -> tensor<64x64x14x14xf32>
    %v876 = stablehlo.divide %v875, %v872 : tensor<64x64x14x14xf32>
    %v877 = stablehlo.subtract %v870, %v876 : tensor<64x64x14x14xf32>
    %v878 = stablehlo.multiply %v877, %v877 : tensor<64x64x14x14xf32>
    %v879 = stablehlo.reduce(%v878 init: %v871) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x64x14x14xf32>, tensor<f32>) -> tensor<64x64xf32>
    %v880 = stablehlo.broadcast_in_dim %v879, dims = [0, 1] : (tensor<64x64xf32>) -> tensor<64x64x14x14xf32>
    %v881 = stablehlo.divide %v880, %v872 : tensor<64x64x14x14xf32>
    %v882 = stablehlo.add %v881, %v873 : tensor<64x64x14x14xf32>
    %v883 = stablehlo.rsqrt %v882 : tensor<64x64x14x14xf32>
    %v884 = stablehlo.multiply %v877, %v883 : tensor<64x64x14x14xf32>
    %v885 = stablehlo.broadcast_in_dim %gp10, dims = [1] : (tensor<64xf32>) -> tensor<64x64x14x14xf32>
    %v886 = stablehlo.broadcast_in_dim %btp10, dims = [1] : (tensor<64xf32>) -> tensor<64x64x14x14xf32>
    %v887 = stablehlo.multiply %v884, %v885 : tensor<64x64x14x14xf32>
    %v888 = stablehlo.add %v887, %v886 : tensor<64x64x14x14xf32>
    %v889 = stablehlo.reshape %v888 : (tensor<64x64x14x14xf32>) -> tensor<64x12544xf32>
    %v890 = stablehlo.reshape %v889 : (tensor<64x12544xf32>) -> tensor<64x64x14x14xf32>
    %v891 = stablehlo.reshape %v802 : (tensor<64x12544xf32>) -> tensor<64x64x14x14xf32>
    %v892 = stablehlo.add %v890, %v891 : tensor<64x64x14x14xf32>
    %v893 = stablehlo.reshape %v892 : (tensor<64x64x14x14xf32>) -> tensor<64x12544xf32>
    %v894 = stablehlo.reshape %v893 : (tensor<64x12544xf32>) -> tensor<64x64x14x14xf32>
    %v895 = stablehlo.convolution(%v894, %We11)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x14x14xf32>, tensor<384x64x1x1xf32>) -> tensor<64x384x14x14xf32>
    %v896 = stablehlo.broadcast_in_dim %zb384, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v897 = stablehlo.add %v895, %v896 : tensor<64x384x14x14xf32>
    %v898 = stablehlo.reshape %v897 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v899 = stablehlo.reshape %v898 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v900 = stablehlo.constant dense<0.0> : tensor<f32>
    %v901 = stablehlo.constant dense<196.0> : tensor<64x384x14x14xf32>
    %v902 = stablehlo.constant dense<1.0e-5> : tensor<64x384x14x14xf32>
    %v903 = stablehlo.reduce(%v899 init: %v900) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x384x14x14xf32>, tensor<f32>) -> tensor<64x384xf32>
    %v904 = stablehlo.broadcast_in_dim %v903, dims = [0, 1] : (tensor<64x384xf32>) -> tensor<64x384x14x14xf32>
    %v905 = stablehlo.divide %v904, %v901 : tensor<64x384x14x14xf32>
    %v906 = stablehlo.subtract %v899, %v905 : tensor<64x384x14x14xf32>
    %v907 = stablehlo.multiply %v906, %v906 : tensor<64x384x14x14xf32>
    %v908 = stablehlo.reduce(%v907 init: %v900) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x384x14x14xf32>, tensor<f32>) -> tensor<64x384xf32>
    %v909 = stablehlo.broadcast_in_dim %v908, dims = [0, 1] : (tensor<64x384xf32>) -> tensor<64x384x14x14xf32>
    %v910 = stablehlo.divide %v909, %v901 : tensor<64x384x14x14xf32>
    %v911 = stablehlo.add %v910, %v902 : tensor<64x384x14x14xf32>
    %v912 = stablehlo.rsqrt %v911 : tensor<64x384x14x14xf32>
    %v913 = stablehlo.multiply %v906, %v912 : tensor<64x384x14x14xf32>
    %v914 = stablehlo.broadcast_in_dim %ge11, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v915 = stablehlo.broadcast_in_dim %bte11, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v916 = stablehlo.multiply %v913, %v914 : tensor<64x384x14x14xf32>
    %v917 = stablehlo.add %v916, %v915 : tensor<64x384x14x14xf32>
    %v918 = stablehlo.reshape %v917 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v919 = stablehlo.reshape %v918 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v920 = stablehlo.constant dense<0.0> : tensor<64x384x14x14xf32>
    %v921 = stablehlo.constant dense<6.0> : tensor<64x384x14x14xf32>
    %v922 = stablehlo.maximum %v919, %v920 : tensor<64x384x14x14xf32>
    %v923 = stablehlo.minimum %v922, %v921 : tensor<64x384x14x14xf32>
    %v924 = stablehlo.reshape %v923 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v925 = stablehlo.reshape %v924 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v926 = stablehlo.convolution(%v925, %Wd11)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<64x384x14x14xf32>, tensor<384x1x3x3xf32>) -> tensor<64x384x14x14xf32>
    %v927 = stablehlo.broadcast_in_dim %zb384, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v928 = stablehlo.add %v926, %v927 : tensor<64x384x14x14xf32>
    %v929 = stablehlo.reshape %v928 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v930 = stablehlo.reshape %v929 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v931 = stablehlo.constant dense<0.0> : tensor<f32>
    %v932 = stablehlo.constant dense<196.0> : tensor<64x384x14x14xf32>
    %v933 = stablehlo.constant dense<1.0e-5> : tensor<64x384x14x14xf32>
    %v934 = stablehlo.reduce(%v930 init: %v931) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x384x14x14xf32>, tensor<f32>) -> tensor<64x384xf32>
    %v935 = stablehlo.broadcast_in_dim %v934, dims = [0, 1] : (tensor<64x384xf32>) -> tensor<64x384x14x14xf32>
    %v936 = stablehlo.divide %v935, %v932 : tensor<64x384x14x14xf32>
    %v937 = stablehlo.subtract %v930, %v936 : tensor<64x384x14x14xf32>
    %v938 = stablehlo.multiply %v937, %v937 : tensor<64x384x14x14xf32>
    %v939 = stablehlo.reduce(%v938 init: %v931) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x384x14x14xf32>, tensor<f32>) -> tensor<64x384xf32>
    %v940 = stablehlo.broadcast_in_dim %v939, dims = [0, 1] : (tensor<64x384xf32>) -> tensor<64x384x14x14xf32>
    %v941 = stablehlo.divide %v940, %v932 : tensor<64x384x14x14xf32>
    %v942 = stablehlo.add %v941, %v933 : tensor<64x384x14x14xf32>
    %v943 = stablehlo.rsqrt %v942 : tensor<64x384x14x14xf32>
    %v944 = stablehlo.multiply %v937, %v943 : tensor<64x384x14x14xf32>
    %v945 = stablehlo.broadcast_in_dim %gd11, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v946 = stablehlo.broadcast_in_dim %btd11, dims = [1] : (tensor<384xf32>) -> tensor<64x384x14x14xf32>
    %v947 = stablehlo.multiply %v944, %v945 : tensor<64x384x14x14xf32>
    %v948 = stablehlo.add %v947, %v946 : tensor<64x384x14x14xf32>
    %v949 = stablehlo.reshape %v948 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v950 = stablehlo.reshape %v949 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v951 = stablehlo.constant dense<0.0> : tensor<64x384x14x14xf32>
    %v952 = stablehlo.constant dense<6.0> : tensor<64x384x14x14xf32>
    %v953 = stablehlo.maximum %v950, %v951 : tensor<64x384x14x14xf32>
    %v954 = stablehlo.minimum %v953, %v952 : tensor<64x384x14x14xf32>
    %v955 = stablehlo.reshape %v954 : (tensor<64x384x14x14xf32>) -> tensor<64x75264xf32>
    %v956 = stablehlo.reshape %v955 : (tensor<64x75264xf32>) -> tensor<64x384x14x14xf32>
    %v957 = stablehlo.convolution(%v956, %Wp11)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x384x14x14xf32>, tensor<96x384x1x1xf32>) -> tensor<64x96x14x14xf32>
    %v958 = stablehlo.broadcast_in_dim %zb96, dims = [1] : (tensor<96xf32>) -> tensor<64x96x14x14xf32>
    %v959 = stablehlo.add %v957, %v958 : tensor<64x96x14x14xf32>
    %v960 = stablehlo.reshape %v959 : (tensor<64x96x14x14xf32>) -> tensor<64x18816xf32>
    %v961 = stablehlo.reshape %v960 : (tensor<64x18816xf32>) -> tensor<64x96x14x14xf32>
    %v962 = stablehlo.constant dense<0.0> : tensor<f32>
    %v963 = stablehlo.constant dense<196.0> : tensor<64x96x14x14xf32>
    %v964 = stablehlo.constant dense<1.0e-5> : tensor<64x96x14x14xf32>
    %v965 = stablehlo.reduce(%v961 init: %v962) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x96x14x14xf32>, tensor<f32>) -> tensor<64x96xf32>
    %v966 = stablehlo.broadcast_in_dim %v965, dims = [0, 1] : (tensor<64x96xf32>) -> tensor<64x96x14x14xf32>
    %v967 = stablehlo.divide %v966, %v963 : tensor<64x96x14x14xf32>
    %v968 = stablehlo.subtract %v961, %v967 : tensor<64x96x14x14xf32>
    %v969 = stablehlo.multiply %v968, %v968 : tensor<64x96x14x14xf32>
    %v970 = stablehlo.reduce(%v969 init: %v962) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x96x14x14xf32>, tensor<f32>) -> tensor<64x96xf32>
    %v971 = stablehlo.broadcast_in_dim %v970, dims = [0, 1] : (tensor<64x96xf32>) -> tensor<64x96x14x14xf32>
    %v972 = stablehlo.divide %v971, %v963 : tensor<64x96x14x14xf32>
    %v973 = stablehlo.add %v972, %v964 : tensor<64x96x14x14xf32>
    %v974 = stablehlo.rsqrt %v973 : tensor<64x96x14x14xf32>
    %v975 = stablehlo.multiply %v968, %v974 : tensor<64x96x14x14xf32>
    %v976 = stablehlo.broadcast_in_dim %gp11, dims = [1] : (tensor<96xf32>) -> tensor<64x96x14x14xf32>
    %v977 = stablehlo.broadcast_in_dim %btp11, dims = [1] : (tensor<96xf32>) -> tensor<64x96x14x14xf32>
    %v978 = stablehlo.multiply %v975, %v976 : tensor<64x96x14x14xf32>
    %v979 = stablehlo.add %v978, %v977 : tensor<64x96x14x14xf32>
    %v980 = stablehlo.reshape %v979 : (tensor<64x96x14x14xf32>) -> tensor<64x18816xf32>
    %v981 = stablehlo.reshape %v980 : (tensor<64x18816xf32>) -> tensor<64x96x14x14xf32>
    %v982 = stablehlo.convolution(%v981, %We12)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x96x14x14xf32>, tensor<576x96x1x1xf32>) -> tensor<64x576x14x14xf32>
    %v983 = stablehlo.broadcast_in_dim %zb576, dims = [1] : (tensor<576xf32>) -> tensor<64x576x14x14xf32>
    %v984 = stablehlo.add %v982, %v983 : tensor<64x576x14x14xf32>
    %v985 = stablehlo.reshape %v984 : (tensor<64x576x14x14xf32>) -> tensor<64x112896xf32>
    %v986 = stablehlo.reshape %v985 : (tensor<64x112896xf32>) -> tensor<64x576x14x14xf32>
    %v987 = stablehlo.constant dense<0.0> : tensor<f32>
    %v988 = stablehlo.constant dense<196.0> : tensor<64x576x14x14xf32>
    %v989 = stablehlo.constant dense<1.0e-5> : tensor<64x576x14x14xf32>
    %v990 = stablehlo.reduce(%v986 init: %v987) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x576x14x14xf32>, tensor<f32>) -> tensor<64x576xf32>
    %v991 = stablehlo.broadcast_in_dim %v990, dims = [0, 1] : (tensor<64x576xf32>) -> tensor<64x576x14x14xf32>
    %v992 = stablehlo.divide %v991, %v988 : tensor<64x576x14x14xf32>
    %v993 = stablehlo.subtract %v986, %v992 : tensor<64x576x14x14xf32>
    %v994 = stablehlo.multiply %v993, %v993 : tensor<64x576x14x14xf32>
    %v995 = stablehlo.reduce(%v994 init: %v987) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x576x14x14xf32>, tensor<f32>) -> tensor<64x576xf32>
    %v996 = stablehlo.broadcast_in_dim %v995, dims = [0, 1] : (tensor<64x576xf32>) -> tensor<64x576x14x14xf32>
    %v997 = stablehlo.divide %v996, %v988 : tensor<64x576x14x14xf32>
    %v998 = stablehlo.add %v997, %v989 : tensor<64x576x14x14xf32>
    %v999 = stablehlo.rsqrt %v998 : tensor<64x576x14x14xf32>
    %v1000 = stablehlo.multiply %v993, %v999 : tensor<64x576x14x14xf32>
    %v1001 = stablehlo.broadcast_in_dim %ge12, dims = [1] : (tensor<576xf32>) -> tensor<64x576x14x14xf32>
    %v1002 = stablehlo.broadcast_in_dim %bte12, dims = [1] : (tensor<576xf32>) -> tensor<64x576x14x14xf32>
    %v1003 = stablehlo.multiply %v1000, %v1001 : tensor<64x576x14x14xf32>
    %v1004 = stablehlo.add %v1003, %v1002 : tensor<64x576x14x14xf32>
    %v1005 = stablehlo.reshape %v1004 : (tensor<64x576x14x14xf32>) -> tensor<64x112896xf32>
    %v1006 = stablehlo.reshape %v1005 : (tensor<64x112896xf32>) -> tensor<64x576x14x14xf32>
    %v1007 = stablehlo.constant dense<0.0> : tensor<64x576x14x14xf32>
    %v1008 = stablehlo.constant dense<6.0> : tensor<64x576x14x14xf32>
    %v1009 = stablehlo.maximum %v1006, %v1007 : tensor<64x576x14x14xf32>
    %v1010 = stablehlo.minimum %v1009, %v1008 : tensor<64x576x14x14xf32>
    %v1011 = stablehlo.reshape %v1010 : (tensor<64x576x14x14xf32>) -> tensor<64x112896xf32>
    %v1012 = stablehlo.reshape %v1011 : (tensor<64x112896xf32>) -> tensor<64x576x14x14xf32>
    %v1013 = stablehlo.convolution(%v1012, %Wd12)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 576 : i64} : (tensor<64x576x14x14xf32>, tensor<576x1x3x3xf32>) -> tensor<64x576x14x14xf32>
    %v1014 = stablehlo.broadcast_in_dim %zb576, dims = [1] : (tensor<576xf32>) -> tensor<64x576x14x14xf32>
    %v1015 = stablehlo.add %v1013, %v1014 : tensor<64x576x14x14xf32>
    %v1016 = stablehlo.reshape %v1015 : (tensor<64x576x14x14xf32>) -> tensor<64x112896xf32>
    %v1017 = stablehlo.reshape %v1016 : (tensor<64x112896xf32>) -> tensor<64x576x14x14xf32>
    %v1018 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1019 = stablehlo.constant dense<196.0> : tensor<64x576x14x14xf32>
    %v1020 = stablehlo.constant dense<1.0e-5> : tensor<64x576x14x14xf32>
    %v1021 = stablehlo.reduce(%v1017 init: %v1018) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x576x14x14xf32>, tensor<f32>) -> tensor<64x576xf32>
    %v1022 = stablehlo.broadcast_in_dim %v1021, dims = [0, 1] : (tensor<64x576xf32>) -> tensor<64x576x14x14xf32>
    %v1023 = stablehlo.divide %v1022, %v1019 : tensor<64x576x14x14xf32>
    %v1024 = stablehlo.subtract %v1017, %v1023 : tensor<64x576x14x14xf32>
    %v1025 = stablehlo.multiply %v1024, %v1024 : tensor<64x576x14x14xf32>
    %v1026 = stablehlo.reduce(%v1025 init: %v1018) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x576x14x14xf32>, tensor<f32>) -> tensor<64x576xf32>
    %v1027 = stablehlo.broadcast_in_dim %v1026, dims = [0, 1] : (tensor<64x576xf32>) -> tensor<64x576x14x14xf32>
    %v1028 = stablehlo.divide %v1027, %v1019 : tensor<64x576x14x14xf32>
    %v1029 = stablehlo.add %v1028, %v1020 : tensor<64x576x14x14xf32>
    %v1030 = stablehlo.rsqrt %v1029 : tensor<64x576x14x14xf32>
    %v1031 = stablehlo.multiply %v1024, %v1030 : tensor<64x576x14x14xf32>
    %v1032 = stablehlo.broadcast_in_dim %gd12, dims = [1] : (tensor<576xf32>) -> tensor<64x576x14x14xf32>
    %v1033 = stablehlo.broadcast_in_dim %btd12, dims = [1] : (tensor<576xf32>) -> tensor<64x576x14x14xf32>
    %v1034 = stablehlo.multiply %v1031, %v1032 : tensor<64x576x14x14xf32>
    %v1035 = stablehlo.add %v1034, %v1033 : tensor<64x576x14x14xf32>
    %v1036 = stablehlo.reshape %v1035 : (tensor<64x576x14x14xf32>) -> tensor<64x112896xf32>
    %v1037 = stablehlo.reshape %v1036 : (tensor<64x112896xf32>) -> tensor<64x576x14x14xf32>
    %v1038 = stablehlo.constant dense<0.0> : tensor<64x576x14x14xf32>
    %v1039 = stablehlo.constant dense<6.0> : tensor<64x576x14x14xf32>
    %v1040 = stablehlo.maximum %v1037, %v1038 : tensor<64x576x14x14xf32>
    %v1041 = stablehlo.minimum %v1040, %v1039 : tensor<64x576x14x14xf32>
    %v1042 = stablehlo.reshape %v1041 : (tensor<64x576x14x14xf32>) -> tensor<64x112896xf32>
    %v1043 = stablehlo.reshape %v1042 : (tensor<64x112896xf32>) -> tensor<64x576x14x14xf32>
    %v1044 = stablehlo.convolution(%v1043, %Wp12)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x576x14x14xf32>, tensor<96x576x1x1xf32>) -> tensor<64x96x14x14xf32>
    %v1045 = stablehlo.broadcast_in_dim %zb96, dims = [1] : (tensor<96xf32>) -> tensor<64x96x14x14xf32>
    %v1046 = stablehlo.add %v1044, %v1045 : tensor<64x96x14x14xf32>
    %v1047 = stablehlo.reshape %v1046 : (tensor<64x96x14x14xf32>) -> tensor<64x18816xf32>
    %v1048 = stablehlo.reshape %v1047 : (tensor<64x18816xf32>) -> tensor<64x96x14x14xf32>
    %v1049 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1050 = stablehlo.constant dense<196.0> : tensor<64x96x14x14xf32>
    %v1051 = stablehlo.constant dense<1.0e-5> : tensor<64x96x14x14xf32>
    %v1052 = stablehlo.reduce(%v1048 init: %v1049) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x96x14x14xf32>, tensor<f32>) -> tensor<64x96xf32>
    %v1053 = stablehlo.broadcast_in_dim %v1052, dims = [0, 1] : (tensor<64x96xf32>) -> tensor<64x96x14x14xf32>
    %v1054 = stablehlo.divide %v1053, %v1050 : tensor<64x96x14x14xf32>
    %v1055 = stablehlo.subtract %v1048, %v1054 : tensor<64x96x14x14xf32>
    %v1056 = stablehlo.multiply %v1055, %v1055 : tensor<64x96x14x14xf32>
    %v1057 = stablehlo.reduce(%v1056 init: %v1049) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x96x14x14xf32>, tensor<f32>) -> tensor<64x96xf32>
    %v1058 = stablehlo.broadcast_in_dim %v1057, dims = [0, 1] : (tensor<64x96xf32>) -> tensor<64x96x14x14xf32>
    %v1059 = stablehlo.divide %v1058, %v1050 : tensor<64x96x14x14xf32>
    %v1060 = stablehlo.add %v1059, %v1051 : tensor<64x96x14x14xf32>
    %v1061 = stablehlo.rsqrt %v1060 : tensor<64x96x14x14xf32>
    %v1062 = stablehlo.multiply %v1055, %v1061 : tensor<64x96x14x14xf32>
    %v1063 = stablehlo.broadcast_in_dim %gp12, dims = [1] : (tensor<96xf32>) -> tensor<64x96x14x14xf32>
    %v1064 = stablehlo.broadcast_in_dim %btp12, dims = [1] : (tensor<96xf32>) -> tensor<64x96x14x14xf32>
    %v1065 = stablehlo.multiply %v1062, %v1063 : tensor<64x96x14x14xf32>
    %v1066 = stablehlo.add %v1065, %v1064 : tensor<64x96x14x14xf32>
    %v1067 = stablehlo.reshape %v1066 : (tensor<64x96x14x14xf32>) -> tensor<64x18816xf32>
    %v1068 = stablehlo.reshape %v1067 : (tensor<64x18816xf32>) -> tensor<64x96x14x14xf32>
    %v1069 = stablehlo.reshape %v980 : (tensor<64x18816xf32>) -> tensor<64x96x14x14xf32>
    %v1070 = stablehlo.add %v1068, %v1069 : tensor<64x96x14x14xf32>
    %v1071 = stablehlo.reshape %v1070 : (tensor<64x96x14x14xf32>) -> tensor<64x18816xf32>
    %v1072 = stablehlo.reshape %v1071 : (tensor<64x18816xf32>) -> tensor<64x96x14x14xf32>
    %v1073 = stablehlo.convolution(%v1072, %We13)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x96x14x14xf32>, tensor<576x96x1x1xf32>) -> tensor<64x576x14x14xf32>
    %v1074 = stablehlo.broadcast_in_dim %zb576, dims = [1] : (tensor<576xf32>) -> tensor<64x576x14x14xf32>
    %v1075 = stablehlo.add %v1073, %v1074 : tensor<64x576x14x14xf32>
    %v1076 = stablehlo.reshape %v1075 : (tensor<64x576x14x14xf32>) -> tensor<64x112896xf32>
    %v1077 = stablehlo.reshape %v1076 : (tensor<64x112896xf32>) -> tensor<64x576x14x14xf32>
    %v1078 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1079 = stablehlo.constant dense<196.0> : tensor<64x576x14x14xf32>
    %v1080 = stablehlo.constant dense<1.0e-5> : tensor<64x576x14x14xf32>
    %v1081 = stablehlo.reduce(%v1077 init: %v1078) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x576x14x14xf32>, tensor<f32>) -> tensor<64x576xf32>
    %v1082 = stablehlo.broadcast_in_dim %v1081, dims = [0, 1] : (tensor<64x576xf32>) -> tensor<64x576x14x14xf32>
    %v1083 = stablehlo.divide %v1082, %v1079 : tensor<64x576x14x14xf32>
    %v1084 = stablehlo.subtract %v1077, %v1083 : tensor<64x576x14x14xf32>
    %v1085 = stablehlo.multiply %v1084, %v1084 : tensor<64x576x14x14xf32>
    %v1086 = stablehlo.reduce(%v1085 init: %v1078) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x576x14x14xf32>, tensor<f32>) -> tensor<64x576xf32>
    %v1087 = stablehlo.broadcast_in_dim %v1086, dims = [0, 1] : (tensor<64x576xf32>) -> tensor<64x576x14x14xf32>
    %v1088 = stablehlo.divide %v1087, %v1079 : tensor<64x576x14x14xf32>
    %v1089 = stablehlo.add %v1088, %v1080 : tensor<64x576x14x14xf32>
    %v1090 = stablehlo.rsqrt %v1089 : tensor<64x576x14x14xf32>
    %v1091 = stablehlo.multiply %v1084, %v1090 : tensor<64x576x14x14xf32>
    %v1092 = stablehlo.broadcast_in_dim %ge13, dims = [1] : (tensor<576xf32>) -> tensor<64x576x14x14xf32>
    %v1093 = stablehlo.broadcast_in_dim %bte13, dims = [1] : (tensor<576xf32>) -> tensor<64x576x14x14xf32>
    %v1094 = stablehlo.multiply %v1091, %v1092 : tensor<64x576x14x14xf32>
    %v1095 = stablehlo.add %v1094, %v1093 : tensor<64x576x14x14xf32>
    %v1096 = stablehlo.reshape %v1095 : (tensor<64x576x14x14xf32>) -> tensor<64x112896xf32>
    %v1097 = stablehlo.reshape %v1096 : (tensor<64x112896xf32>) -> tensor<64x576x14x14xf32>
    %v1098 = stablehlo.constant dense<0.0> : tensor<64x576x14x14xf32>
    %v1099 = stablehlo.constant dense<6.0> : tensor<64x576x14x14xf32>
    %v1100 = stablehlo.maximum %v1097, %v1098 : tensor<64x576x14x14xf32>
    %v1101 = stablehlo.minimum %v1100, %v1099 : tensor<64x576x14x14xf32>
    %v1102 = stablehlo.reshape %v1101 : (tensor<64x576x14x14xf32>) -> tensor<64x112896xf32>
    %v1103 = stablehlo.reshape %v1102 : (tensor<64x112896xf32>) -> tensor<64x576x14x14xf32>
    %v1104 = stablehlo.convolution(%v1103, %Wd13)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 576 : i64} : (tensor<64x576x14x14xf32>, tensor<576x1x3x3xf32>) -> tensor<64x576x14x14xf32>
    %v1105 = stablehlo.broadcast_in_dim %zb576, dims = [1] : (tensor<576xf32>) -> tensor<64x576x14x14xf32>
    %v1106 = stablehlo.add %v1104, %v1105 : tensor<64x576x14x14xf32>
    %v1107 = stablehlo.reshape %v1106 : (tensor<64x576x14x14xf32>) -> tensor<64x112896xf32>
    %v1108 = stablehlo.reshape %v1107 : (tensor<64x112896xf32>) -> tensor<64x576x14x14xf32>
    %v1109 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1110 = stablehlo.constant dense<196.0> : tensor<64x576x14x14xf32>
    %v1111 = stablehlo.constant dense<1.0e-5> : tensor<64x576x14x14xf32>
    %v1112 = stablehlo.reduce(%v1108 init: %v1109) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x576x14x14xf32>, tensor<f32>) -> tensor<64x576xf32>
    %v1113 = stablehlo.broadcast_in_dim %v1112, dims = [0, 1] : (tensor<64x576xf32>) -> tensor<64x576x14x14xf32>
    %v1114 = stablehlo.divide %v1113, %v1110 : tensor<64x576x14x14xf32>
    %v1115 = stablehlo.subtract %v1108, %v1114 : tensor<64x576x14x14xf32>
    %v1116 = stablehlo.multiply %v1115, %v1115 : tensor<64x576x14x14xf32>
    %v1117 = stablehlo.reduce(%v1116 init: %v1109) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x576x14x14xf32>, tensor<f32>) -> tensor<64x576xf32>
    %v1118 = stablehlo.broadcast_in_dim %v1117, dims = [0, 1] : (tensor<64x576xf32>) -> tensor<64x576x14x14xf32>
    %v1119 = stablehlo.divide %v1118, %v1110 : tensor<64x576x14x14xf32>
    %v1120 = stablehlo.add %v1119, %v1111 : tensor<64x576x14x14xf32>
    %v1121 = stablehlo.rsqrt %v1120 : tensor<64x576x14x14xf32>
    %v1122 = stablehlo.multiply %v1115, %v1121 : tensor<64x576x14x14xf32>
    %v1123 = stablehlo.broadcast_in_dim %gd13, dims = [1] : (tensor<576xf32>) -> tensor<64x576x14x14xf32>
    %v1124 = stablehlo.broadcast_in_dim %btd13, dims = [1] : (tensor<576xf32>) -> tensor<64x576x14x14xf32>
    %v1125 = stablehlo.multiply %v1122, %v1123 : tensor<64x576x14x14xf32>
    %v1126 = stablehlo.add %v1125, %v1124 : tensor<64x576x14x14xf32>
    %v1127 = stablehlo.reshape %v1126 : (tensor<64x576x14x14xf32>) -> tensor<64x112896xf32>
    %v1128 = stablehlo.reshape %v1127 : (tensor<64x112896xf32>) -> tensor<64x576x14x14xf32>
    %v1129 = stablehlo.constant dense<0.0> : tensor<64x576x14x14xf32>
    %v1130 = stablehlo.constant dense<6.0> : tensor<64x576x14x14xf32>
    %v1131 = stablehlo.maximum %v1128, %v1129 : tensor<64x576x14x14xf32>
    %v1132 = stablehlo.minimum %v1131, %v1130 : tensor<64x576x14x14xf32>
    %v1133 = stablehlo.reshape %v1132 : (tensor<64x576x14x14xf32>) -> tensor<64x112896xf32>
    %v1134 = stablehlo.reshape %v1133 : (tensor<64x112896xf32>) -> tensor<64x576x14x14xf32>
    %v1135 = stablehlo.convolution(%v1134, %Wp13)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x576x14x14xf32>, tensor<96x576x1x1xf32>) -> tensor<64x96x14x14xf32>
    %v1136 = stablehlo.broadcast_in_dim %zb96, dims = [1] : (tensor<96xf32>) -> tensor<64x96x14x14xf32>
    %v1137 = stablehlo.add %v1135, %v1136 : tensor<64x96x14x14xf32>
    %v1138 = stablehlo.reshape %v1137 : (tensor<64x96x14x14xf32>) -> tensor<64x18816xf32>
    %v1139 = stablehlo.reshape %v1138 : (tensor<64x18816xf32>) -> tensor<64x96x14x14xf32>
    %v1140 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1141 = stablehlo.constant dense<196.0> : tensor<64x96x14x14xf32>
    %v1142 = stablehlo.constant dense<1.0e-5> : tensor<64x96x14x14xf32>
    %v1143 = stablehlo.reduce(%v1139 init: %v1140) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x96x14x14xf32>, tensor<f32>) -> tensor<64x96xf32>
    %v1144 = stablehlo.broadcast_in_dim %v1143, dims = [0, 1] : (tensor<64x96xf32>) -> tensor<64x96x14x14xf32>
    %v1145 = stablehlo.divide %v1144, %v1141 : tensor<64x96x14x14xf32>
    %v1146 = stablehlo.subtract %v1139, %v1145 : tensor<64x96x14x14xf32>
    %v1147 = stablehlo.multiply %v1146, %v1146 : tensor<64x96x14x14xf32>
    %v1148 = stablehlo.reduce(%v1147 init: %v1140) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x96x14x14xf32>, tensor<f32>) -> tensor<64x96xf32>
    %v1149 = stablehlo.broadcast_in_dim %v1148, dims = [0, 1] : (tensor<64x96xf32>) -> tensor<64x96x14x14xf32>
    %v1150 = stablehlo.divide %v1149, %v1141 : tensor<64x96x14x14xf32>
    %v1151 = stablehlo.add %v1150, %v1142 : tensor<64x96x14x14xf32>
    %v1152 = stablehlo.rsqrt %v1151 : tensor<64x96x14x14xf32>
    %v1153 = stablehlo.multiply %v1146, %v1152 : tensor<64x96x14x14xf32>
    %v1154 = stablehlo.broadcast_in_dim %gp13, dims = [1] : (tensor<96xf32>) -> tensor<64x96x14x14xf32>
    %v1155 = stablehlo.broadcast_in_dim %btp13, dims = [1] : (tensor<96xf32>) -> tensor<64x96x14x14xf32>
    %v1156 = stablehlo.multiply %v1153, %v1154 : tensor<64x96x14x14xf32>
    %v1157 = stablehlo.add %v1156, %v1155 : tensor<64x96x14x14xf32>
    %v1158 = stablehlo.reshape %v1157 : (tensor<64x96x14x14xf32>) -> tensor<64x18816xf32>
    %v1159 = stablehlo.reshape %v1158 : (tensor<64x18816xf32>) -> tensor<64x96x14x14xf32>
    %v1160 = stablehlo.reshape %v1071 : (tensor<64x18816xf32>) -> tensor<64x96x14x14xf32>
    %v1161 = stablehlo.add %v1159, %v1160 : tensor<64x96x14x14xf32>
    %v1162 = stablehlo.reshape %v1161 : (tensor<64x96x14x14xf32>) -> tensor<64x18816xf32>
    %v1163 = stablehlo.reshape %v1162 : (tensor<64x18816xf32>) -> tensor<64x96x14x14xf32>
    %v1164 = stablehlo.convolution(%v1163, %We14)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x96x14x14xf32>, tensor<576x96x1x1xf32>) -> tensor<64x576x14x14xf32>
    %v1165 = stablehlo.broadcast_in_dim %zb576, dims = [1] : (tensor<576xf32>) -> tensor<64x576x14x14xf32>
    %v1166 = stablehlo.add %v1164, %v1165 : tensor<64x576x14x14xf32>
    %v1167 = stablehlo.reshape %v1166 : (tensor<64x576x14x14xf32>) -> tensor<64x112896xf32>
    %v1168 = stablehlo.reshape %v1167 : (tensor<64x112896xf32>) -> tensor<64x576x14x14xf32>
    %v1169 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1170 = stablehlo.constant dense<196.0> : tensor<64x576x14x14xf32>
    %v1171 = stablehlo.constant dense<1.0e-5> : tensor<64x576x14x14xf32>
    %v1172 = stablehlo.reduce(%v1168 init: %v1169) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x576x14x14xf32>, tensor<f32>) -> tensor<64x576xf32>
    %v1173 = stablehlo.broadcast_in_dim %v1172, dims = [0, 1] : (tensor<64x576xf32>) -> tensor<64x576x14x14xf32>
    %v1174 = stablehlo.divide %v1173, %v1170 : tensor<64x576x14x14xf32>
    %v1175 = stablehlo.subtract %v1168, %v1174 : tensor<64x576x14x14xf32>
    %v1176 = stablehlo.multiply %v1175, %v1175 : tensor<64x576x14x14xf32>
    %v1177 = stablehlo.reduce(%v1176 init: %v1169) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x576x14x14xf32>, tensor<f32>) -> tensor<64x576xf32>
    %v1178 = stablehlo.broadcast_in_dim %v1177, dims = [0, 1] : (tensor<64x576xf32>) -> tensor<64x576x14x14xf32>
    %v1179 = stablehlo.divide %v1178, %v1170 : tensor<64x576x14x14xf32>
    %v1180 = stablehlo.add %v1179, %v1171 : tensor<64x576x14x14xf32>
    %v1181 = stablehlo.rsqrt %v1180 : tensor<64x576x14x14xf32>
    %v1182 = stablehlo.multiply %v1175, %v1181 : tensor<64x576x14x14xf32>
    %v1183 = stablehlo.broadcast_in_dim %ge14, dims = [1] : (tensor<576xf32>) -> tensor<64x576x14x14xf32>
    %v1184 = stablehlo.broadcast_in_dim %bte14, dims = [1] : (tensor<576xf32>) -> tensor<64x576x14x14xf32>
    %v1185 = stablehlo.multiply %v1182, %v1183 : tensor<64x576x14x14xf32>
    %v1186 = stablehlo.add %v1185, %v1184 : tensor<64x576x14x14xf32>
    %v1187 = stablehlo.reshape %v1186 : (tensor<64x576x14x14xf32>) -> tensor<64x112896xf32>
    %v1188 = stablehlo.reshape %v1187 : (tensor<64x112896xf32>) -> tensor<64x576x14x14xf32>
    %v1189 = stablehlo.constant dense<0.0> : tensor<64x576x14x14xf32>
    %v1190 = stablehlo.constant dense<6.0> : tensor<64x576x14x14xf32>
    %v1191 = stablehlo.maximum %v1188, %v1189 : tensor<64x576x14x14xf32>
    %v1192 = stablehlo.minimum %v1191, %v1190 : tensor<64x576x14x14xf32>
    %v1193 = stablehlo.reshape %v1192 : (tensor<64x576x14x14xf32>) -> tensor<64x112896xf32>
    %v1194 = stablehlo.reshape %v1193 : (tensor<64x112896xf32>) -> tensor<64x576x14x14xf32>
    %v1195 = stablehlo.convolution(%v1194, %Wd14)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 576 : i64} : (tensor<64x576x14x14xf32>, tensor<576x1x3x3xf32>) -> tensor<64x576x7x7xf32>
    %v1196 = stablehlo.broadcast_in_dim %zb576, dims = [1] : (tensor<576xf32>) -> tensor<64x576x7x7xf32>
    %v1197 = stablehlo.add %v1195, %v1196 : tensor<64x576x7x7xf32>
    %v1198 = stablehlo.reshape %v1197 : (tensor<64x576x7x7xf32>) -> tensor<64x28224xf32>
    %v1199 = stablehlo.reshape %v1198 : (tensor<64x28224xf32>) -> tensor<64x576x7x7xf32>
    %v1200 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1201 = stablehlo.constant dense<49.0> : tensor<64x576x7x7xf32>
    %v1202 = stablehlo.constant dense<1.0e-5> : tensor<64x576x7x7xf32>
    %v1203 = stablehlo.reduce(%v1199 init: %v1200) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x576x7x7xf32>, tensor<f32>) -> tensor<64x576xf32>
    %v1204 = stablehlo.broadcast_in_dim %v1203, dims = [0, 1] : (tensor<64x576xf32>) -> tensor<64x576x7x7xf32>
    %v1205 = stablehlo.divide %v1204, %v1201 : tensor<64x576x7x7xf32>
    %v1206 = stablehlo.subtract %v1199, %v1205 : tensor<64x576x7x7xf32>
    %v1207 = stablehlo.multiply %v1206, %v1206 : tensor<64x576x7x7xf32>
    %v1208 = stablehlo.reduce(%v1207 init: %v1200) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x576x7x7xf32>, tensor<f32>) -> tensor<64x576xf32>
    %v1209 = stablehlo.broadcast_in_dim %v1208, dims = [0, 1] : (tensor<64x576xf32>) -> tensor<64x576x7x7xf32>
    %v1210 = stablehlo.divide %v1209, %v1201 : tensor<64x576x7x7xf32>
    %v1211 = stablehlo.add %v1210, %v1202 : tensor<64x576x7x7xf32>
    %v1212 = stablehlo.rsqrt %v1211 : tensor<64x576x7x7xf32>
    %v1213 = stablehlo.multiply %v1206, %v1212 : tensor<64x576x7x7xf32>
    %v1214 = stablehlo.broadcast_in_dim %gd14, dims = [1] : (tensor<576xf32>) -> tensor<64x576x7x7xf32>
    %v1215 = stablehlo.broadcast_in_dim %btd14, dims = [1] : (tensor<576xf32>) -> tensor<64x576x7x7xf32>
    %v1216 = stablehlo.multiply %v1213, %v1214 : tensor<64x576x7x7xf32>
    %v1217 = stablehlo.add %v1216, %v1215 : tensor<64x576x7x7xf32>
    %v1218 = stablehlo.reshape %v1217 : (tensor<64x576x7x7xf32>) -> tensor<64x28224xf32>
    %v1219 = stablehlo.reshape %v1218 : (tensor<64x28224xf32>) -> tensor<64x576x7x7xf32>
    %v1220 = stablehlo.constant dense<0.0> : tensor<64x576x7x7xf32>
    %v1221 = stablehlo.constant dense<6.0> : tensor<64x576x7x7xf32>
    %v1222 = stablehlo.maximum %v1219, %v1220 : tensor<64x576x7x7xf32>
    %v1223 = stablehlo.minimum %v1222, %v1221 : tensor<64x576x7x7xf32>
    %v1224 = stablehlo.reshape %v1223 : (tensor<64x576x7x7xf32>) -> tensor<64x28224xf32>
    %v1225 = stablehlo.reshape %v1224 : (tensor<64x28224xf32>) -> tensor<64x576x7x7xf32>
    %v1226 = stablehlo.convolution(%v1225, %Wp14)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x576x7x7xf32>, tensor<160x576x1x1xf32>) -> tensor<64x160x7x7xf32>
    %v1227 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<64x160x7x7xf32>
    %v1228 = stablehlo.add %v1226, %v1227 : tensor<64x160x7x7xf32>
    %v1229 = stablehlo.reshape %v1228 : (tensor<64x160x7x7xf32>) -> tensor<64x7840xf32>
    %v1230 = stablehlo.reshape %v1229 : (tensor<64x7840xf32>) -> tensor<64x160x7x7xf32>
    %v1231 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1232 = stablehlo.constant dense<49.0> : tensor<64x160x7x7xf32>
    %v1233 = stablehlo.constant dense<1.0e-5> : tensor<64x160x7x7xf32>
    %v1234 = stablehlo.reduce(%v1230 init: %v1231) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x160x7x7xf32>, tensor<f32>) -> tensor<64x160xf32>
    %v1235 = stablehlo.broadcast_in_dim %v1234, dims = [0, 1] : (tensor<64x160xf32>) -> tensor<64x160x7x7xf32>
    %v1236 = stablehlo.divide %v1235, %v1232 : tensor<64x160x7x7xf32>
    %v1237 = stablehlo.subtract %v1230, %v1236 : tensor<64x160x7x7xf32>
    %v1238 = stablehlo.multiply %v1237, %v1237 : tensor<64x160x7x7xf32>
    %v1239 = stablehlo.reduce(%v1238 init: %v1231) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x160x7x7xf32>, tensor<f32>) -> tensor<64x160xf32>
    %v1240 = stablehlo.broadcast_in_dim %v1239, dims = [0, 1] : (tensor<64x160xf32>) -> tensor<64x160x7x7xf32>
    %v1241 = stablehlo.divide %v1240, %v1232 : tensor<64x160x7x7xf32>
    %v1242 = stablehlo.add %v1241, %v1233 : tensor<64x160x7x7xf32>
    %v1243 = stablehlo.rsqrt %v1242 : tensor<64x160x7x7xf32>
    %v1244 = stablehlo.multiply %v1237, %v1243 : tensor<64x160x7x7xf32>
    %v1245 = stablehlo.broadcast_in_dim %gp14, dims = [1] : (tensor<160xf32>) -> tensor<64x160x7x7xf32>
    %v1246 = stablehlo.broadcast_in_dim %btp14, dims = [1] : (tensor<160xf32>) -> tensor<64x160x7x7xf32>
    %v1247 = stablehlo.multiply %v1244, %v1245 : tensor<64x160x7x7xf32>
    %v1248 = stablehlo.add %v1247, %v1246 : tensor<64x160x7x7xf32>
    %v1249 = stablehlo.reshape %v1248 : (tensor<64x160x7x7xf32>) -> tensor<64x7840xf32>
    %v1250 = stablehlo.reshape %v1249 : (tensor<64x7840xf32>) -> tensor<64x160x7x7xf32>
    %v1251 = stablehlo.convolution(%v1250, %We15)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x160x7x7xf32>, tensor<960x160x1x1xf32>) -> tensor<64x960x7x7xf32>
    %v1252 = stablehlo.broadcast_in_dim %zb960, dims = [1] : (tensor<960xf32>) -> tensor<64x960x7x7xf32>
    %v1253 = stablehlo.add %v1251, %v1252 : tensor<64x960x7x7xf32>
    %v1254 = stablehlo.reshape %v1253 : (tensor<64x960x7x7xf32>) -> tensor<64x47040xf32>
    %v1255 = stablehlo.reshape %v1254 : (tensor<64x47040xf32>) -> tensor<64x960x7x7xf32>
    %v1256 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1257 = stablehlo.constant dense<49.0> : tensor<64x960x7x7xf32>
    %v1258 = stablehlo.constant dense<1.0e-5> : tensor<64x960x7x7xf32>
    %v1259 = stablehlo.reduce(%v1255 init: %v1256) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x960x7x7xf32>, tensor<f32>) -> tensor<64x960xf32>
    %v1260 = stablehlo.broadcast_in_dim %v1259, dims = [0, 1] : (tensor<64x960xf32>) -> tensor<64x960x7x7xf32>
    %v1261 = stablehlo.divide %v1260, %v1257 : tensor<64x960x7x7xf32>
    %v1262 = stablehlo.subtract %v1255, %v1261 : tensor<64x960x7x7xf32>
    %v1263 = stablehlo.multiply %v1262, %v1262 : tensor<64x960x7x7xf32>
    %v1264 = stablehlo.reduce(%v1263 init: %v1256) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x960x7x7xf32>, tensor<f32>) -> tensor<64x960xf32>
    %v1265 = stablehlo.broadcast_in_dim %v1264, dims = [0, 1] : (tensor<64x960xf32>) -> tensor<64x960x7x7xf32>
    %v1266 = stablehlo.divide %v1265, %v1257 : tensor<64x960x7x7xf32>
    %v1267 = stablehlo.add %v1266, %v1258 : tensor<64x960x7x7xf32>
    %v1268 = stablehlo.rsqrt %v1267 : tensor<64x960x7x7xf32>
    %v1269 = stablehlo.multiply %v1262, %v1268 : tensor<64x960x7x7xf32>
    %v1270 = stablehlo.broadcast_in_dim %ge15, dims = [1] : (tensor<960xf32>) -> tensor<64x960x7x7xf32>
    %v1271 = stablehlo.broadcast_in_dim %bte15, dims = [1] : (tensor<960xf32>) -> tensor<64x960x7x7xf32>
    %v1272 = stablehlo.multiply %v1269, %v1270 : tensor<64x960x7x7xf32>
    %v1273 = stablehlo.add %v1272, %v1271 : tensor<64x960x7x7xf32>
    %v1274 = stablehlo.reshape %v1273 : (tensor<64x960x7x7xf32>) -> tensor<64x47040xf32>
    %v1275 = stablehlo.reshape %v1274 : (tensor<64x47040xf32>) -> tensor<64x960x7x7xf32>
    %v1276 = stablehlo.constant dense<0.0> : tensor<64x960x7x7xf32>
    %v1277 = stablehlo.constant dense<6.0> : tensor<64x960x7x7xf32>
    %v1278 = stablehlo.maximum %v1275, %v1276 : tensor<64x960x7x7xf32>
    %v1279 = stablehlo.minimum %v1278, %v1277 : tensor<64x960x7x7xf32>
    %v1280 = stablehlo.reshape %v1279 : (tensor<64x960x7x7xf32>) -> tensor<64x47040xf32>
    %v1281 = stablehlo.reshape %v1280 : (tensor<64x47040xf32>) -> tensor<64x960x7x7xf32>
    %v1282 = stablehlo.convolution(%v1281, %Wd15)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 960 : i64} : (tensor<64x960x7x7xf32>, tensor<960x1x3x3xf32>) -> tensor<64x960x7x7xf32>
    %v1283 = stablehlo.broadcast_in_dim %zb960, dims = [1] : (tensor<960xf32>) -> tensor<64x960x7x7xf32>
    %v1284 = stablehlo.add %v1282, %v1283 : tensor<64x960x7x7xf32>
    %v1285 = stablehlo.reshape %v1284 : (tensor<64x960x7x7xf32>) -> tensor<64x47040xf32>
    %v1286 = stablehlo.reshape %v1285 : (tensor<64x47040xf32>) -> tensor<64x960x7x7xf32>
    %v1287 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1288 = stablehlo.constant dense<49.0> : tensor<64x960x7x7xf32>
    %v1289 = stablehlo.constant dense<1.0e-5> : tensor<64x960x7x7xf32>
    %v1290 = stablehlo.reduce(%v1286 init: %v1287) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x960x7x7xf32>, tensor<f32>) -> tensor<64x960xf32>
    %v1291 = stablehlo.broadcast_in_dim %v1290, dims = [0, 1] : (tensor<64x960xf32>) -> tensor<64x960x7x7xf32>
    %v1292 = stablehlo.divide %v1291, %v1288 : tensor<64x960x7x7xf32>
    %v1293 = stablehlo.subtract %v1286, %v1292 : tensor<64x960x7x7xf32>
    %v1294 = stablehlo.multiply %v1293, %v1293 : tensor<64x960x7x7xf32>
    %v1295 = stablehlo.reduce(%v1294 init: %v1287) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x960x7x7xf32>, tensor<f32>) -> tensor<64x960xf32>
    %v1296 = stablehlo.broadcast_in_dim %v1295, dims = [0, 1] : (tensor<64x960xf32>) -> tensor<64x960x7x7xf32>
    %v1297 = stablehlo.divide %v1296, %v1288 : tensor<64x960x7x7xf32>
    %v1298 = stablehlo.add %v1297, %v1289 : tensor<64x960x7x7xf32>
    %v1299 = stablehlo.rsqrt %v1298 : tensor<64x960x7x7xf32>
    %v1300 = stablehlo.multiply %v1293, %v1299 : tensor<64x960x7x7xf32>
    %v1301 = stablehlo.broadcast_in_dim %gd15, dims = [1] : (tensor<960xf32>) -> tensor<64x960x7x7xf32>
    %v1302 = stablehlo.broadcast_in_dim %btd15, dims = [1] : (tensor<960xf32>) -> tensor<64x960x7x7xf32>
    %v1303 = stablehlo.multiply %v1300, %v1301 : tensor<64x960x7x7xf32>
    %v1304 = stablehlo.add %v1303, %v1302 : tensor<64x960x7x7xf32>
    %v1305 = stablehlo.reshape %v1304 : (tensor<64x960x7x7xf32>) -> tensor<64x47040xf32>
    %v1306 = stablehlo.reshape %v1305 : (tensor<64x47040xf32>) -> tensor<64x960x7x7xf32>
    %v1307 = stablehlo.constant dense<0.0> : tensor<64x960x7x7xf32>
    %v1308 = stablehlo.constant dense<6.0> : tensor<64x960x7x7xf32>
    %v1309 = stablehlo.maximum %v1306, %v1307 : tensor<64x960x7x7xf32>
    %v1310 = stablehlo.minimum %v1309, %v1308 : tensor<64x960x7x7xf32>
    %v1311 = stablehlo.reshape %v1310 : (tensor<64x960x7x7xf32>) -> tensor<64x47040xf32>
    %v1312 = stablehlo.reshape %v1311 : (tensor<64x47040xf32>) -> tensor<64x960x7x7xf32>
    %v1313 = stablehlo.convolution(%v1312, %Wp15)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x960x7x7xf32>, tensor<160x960x1x1xf32>) -> tensor<64x160x7x7xf32>
    %v1314 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<64x160x7x7xf32>
    %v1315 = stablehlo.add %v1313, %v1314 : tensor<64x160x7x7xf32>
    %v1316 = stablehlo.reshape %v1315 : (tensor<64x160x7x7xf32>) -> tensor<64x7840xf32>
    %v1317 = stablehlo.reshape %v1316 : (tensor<64x7840xf32>) -> tensor<64x160x7x7xf32>
    %v1318 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1319 = stablehlo.constant dense<49.0> : tensor<64x160x7x7xf32>
    %v1320 = stablehlo.constant dense<1.0e-5> : tensor<64x160x7x7xf32>
    %v1321 = stablehlo.reduce(%v1317 init: %v1318) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x160x7x7xf32>, tensor<f32>) -> tensor<64x160xf32>
    %v1322 = stablehlo.broadcast_in_dim %v1321, dims = [0, 1] : (tensor<64x160xf32>) -> tensor<64x160x7x7xf32>
    %v1323 = stablehlo.divide %v1322, %v1319 : tensor<64x160x7x7xf32>
    %v1324 = stablehlo.subtract %v1317, %v1323 : tensor<64x160x7x7xf32>
    %v1325 = stablehlo.multiply %v1324, %v1324 : tensor<64x160x7x7xf32>
    %v1326 = stablehlo.reduce(%v1325 init: %v1318) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x160x7x7xf32>, tensor<f32>) -> tensor<64x160xf32>
    %v1327 = stablehlo.broadcast_in_dim %v1326, dims = [0, 1] : (tensor<64x160xf32>) -> tensor<64x160x7x7xf32>
    %v1328 = stablehlo.divide %v1327, %v1319 : tensor<64x160x7x7xf32>
    %v1329 = stablehlo.add %v1328, %v1320 : tensor<64x160x7x7xf32>
    %v1330 = stablehlo.rsqrt %v1329 : tensor<64x160x7x7xf32>
    %v1331 = stablehlo.multiply %v1324, %v1330 : tensor<64x160x7x7xf32>
    %v1332 = stablehlo.broadcast_in_dim %gp15, dims = [1] : (tensor<160xf32>) -> tensor<64x160x7x7xf32>
    %v1333 = stablehlo.broadcast_in_dim %btp15, dims = [1] : (tensor<160xf32>) -> tensor<64x160x7x7xf32>
    %v1334 = stablehlo.multiply %v1331, %v1332 : tensor<64x160x7x7xf32>
    %v1335 = stablehlo.add %v1334, %v1333 : tensor<64x160x7x7xf32>
    %v1336 = stablehlo.reshape %v1335 : (tensor<64x160x7x7xf32>) -> tensor<64x7840xf32>
    %v1337 = stablehlo.reshape %v1336 : (tensor<64x7840xf32>) -> tensor<64x160x7x7xf32>
    %v1338 = stablehlo.reshape %v1249 : (tensor<64x7840xf32>) -> tensor<64x160x7x7xf32>
    %v1339 = stablehlo.add %v1337, %v1338 : tensor<64x160x7x7xf32>
    %v1340 = stablehlo.reshape %v1339 : (tensor<64x160x7x7xf32>) -> tensor<64x7840xf32>
    %v1341 = stablehlo.reshape %v1340 : (tensor<64x7840xf32>) -> tensor<64x160x7x7xf32>
    %v1342 = stablehlo.convolution(%v1341, %We16)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x160x7x7xf32>, tensor<960x160x1x1xf32>) -> tensor<64x960x7x7xf32>
    %v1343 = stablehlo.broadcast_in_dim %zb960, dims = [1] : (tensor<960xf32>) -> tensor<64x960x7x7xf32>
    %v1344 = stablehlo.add %v1342, %v1343 : tensor<64x960x7x7xf32>
    %v1345 = stablehlo.reshape %v1344 : (tensor<64x960x7x7xf32>) -> tensor<64x47040xf32>
    %v1346 = stablehlo.reshape %v1345 : (tensor<64x47040xf32>) -> tensor<64x960x7x7xf32>
    %v1347 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1348 = stablehlo.constant dense<49.0> : tensor<64x960x7x7xf32>
    %v1349 = stablehlo.constant dense<1.0e-5> : tensor<64x960x7x7xf32>
    %v1350 = stablehlo.reduce(%v1346 init: %v1347) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x960x7x7xf32>, tensor<f32>) -> tensor<64x960xf32>
    %v1351 = stablehlo.broadcast_in_dim %v1350, dims = [0, 1] : (tensor<64x960xf32>) -> tensor<64x960x7x7xf32>
    %v1352 = stablehlo.divide %v1351, %v1348 : tensor<64x960x7x7xf32>
    %v1353 = stablehlo.subtract %v1346, %v1352 : tensor<64x960x7x7xf32>
    %v1354 = stablehlo.multiply %v1353, %v1353 : tensor<64x960x7x7xf32>
    %v1355 = stablehlo.reduce(%v1354 init: %v1347) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x960x7x7xf32>, tensor<f32>) -> tensor<64x960xf32>
    %v1356 = stablehlo.broadcast_in_dim %v1355, dims = [0, 1] : (tensor<64x960xf32>) -> tensor<64x960x7x7xf32>
    %v1357 = stablehlo.divide %v1356, %v1348 : tensor<64x960x7x7xf32>
    %v1358 = stablehlo.add %v1357, %v1349 : tensor<64x960x7x7xf32>
    %v1359 = stablehlo.rsqrt %v1358 : tensor<64x960x7x7xf32>
    %v1360 = stablehlo.multiply %v1353, %v1359 : tensor<64x960x7x7xf32>
    %v1361 = stablehlo.broadcast_in_dim %ge16, dims = [1] : (tensor<960xf32>) -> tensor<64x960x7x7xf32>
    %v1362 = stablehlo.broadcast_in_dim %bte16, dims = [1] : (tensor<960xf32>) -> tensor<64x960x7x7xf32>
    %v1363 = stablehlo.multiply %v1360, %v1361 : tensor<64x960x7x7xf32>
    %v1364 = stablehlo.add %v1363, %v1362 : tensor<64x960x7x7xf32>
    %v1365 = stablehlo.reshape %v1364 : (tensor<64x960x7x7xf32>) -> tensor<64x47040xf32>
    %v1366 = stablehlo.reshape %v1365 : (tensor<64x47040xf32>) -> tensor<64x960x7x7xf32>
    %v1367 = stablehlo.constant dense<0.0> : tensor<64x960x7x7xf32>
    %v1368 = stablehlo.constant dense<6.0> : tensor<64x960x7x7xf32>
    %v1369 = stablehlo.maximum %v1366, %v1367 : tensor<64x960x7x7xf32>
    %v1370 = stablehlo.minimum %v1369, %v1368 : tensor<64x960x7x7xf32>
    %v1371 = stablehlo.reshape %v1370 : (tensor<64x960x7x7xf32>) -> tensor<64x47040xf32>
    %v1372 = stablehlo.reshape %v1371 : (tensor<64x47040xf32>) -> tensor<64x960x7x7xf32>
    %v1373 = stablehlo.convolution(%v1372, %Wd16)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 960 : i64} : (tensor<64x960x7x7xf32>, tensor<960x1x3x3xf32>) -> tensor<64x960x7x7xf32>
    %v1374 = stablehlo.broadcast_in_dim %zb960, dims = [1] : (tensor<960xf32>) -> tensor<64x960x7x7xf32>
    %v1375 = stablehlo.add %v1373, %v1374 : tensor<64x960x7x7xf32>
    %v1376 = stablehlo.reshape %v1375 : (tensor<64x960x7x7xf32>) -> tensor<64x47040xf32>
    %v1377 = stablehlo.reshape %v1376 : (tensor<64x47040xf32>) -> tensor<64x960x7x7xf32>
    %v1378 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1379 = stablehlo.constant dense<49.0> : tensor<64x960x7x7xf32>
    %v1380 = stablehlo.constant dense<1.0e-5> : tensor<64x960x7x7xf32>
    %v1381 = stablehlo.reduce(%v1377 init: %v1378) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x960x7x7xf32>, tensor<f32>) -> tensor<64x960xf32>
    %v1382 = stablehlo.broadcast_in_dim %v1381, dims = [0, 1] : (tensor<64x960xf32>) -> tensor<64x960x7x7xf32>
    %v1383 = stablehlo.divide %v1382, %v1379 : tensor<64x960x7x7xf32>
    %v1384 = stablehlo.subtract %v1377, %v1383 : tensor<64x960x7x7xf32>
    %v1385 = stablehlo.multiply %v1384, %v1384 : tensor<64x960x7x7xf32>
    %v1386 = stablehlo.reduce(%v1385 init: %v1378) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x960x7x7xf32>, tensor<f32>) -> tensor<64x960xf32>
    %v1387 = stablehlo.broadcast_in_dim %v1386, dims = [0, 1] : (tensor<64x960xf32>) -> tensor<64x960x7x7xf32>
    %v1388 = stablehlo.divide %v1387, %v1379 : tensor<64x960x7x7xf32>
    %v1389 = stablehlo.add %v1388, %v1380 : tensor<64x960x7x7xf32>
    %v1390 = stablehlo.rsqrt %v1389 : tensor<64x960x7x7xf32>
    %v1391 = stablehlo.multiply %v1384, %v1390 : tensor<64x960x7x7xf32>
    %v1392 = stablehlo.broadcast_in_dim %gd16, dims = [1] : (tensor<960xf32>) -> tensor<64x960x7x7xf32>
    %v1393 = stablehlo.broadcast_in_dim %btd16, dims = [1] : (tensor<960xf32>) -> tensor<64x960x7x7xf32>
    %v1394 = stablehlo.multiply %v1391, %v1392 : tensor<64x960x7x7xf32>
    %v1395 = stablehlo.add %v1394, %v1393 : tensor<64x960x7x7xf32>
    %v1396 = stablehlo.reshape %v1395 : (tensor<64x960x7x7xf32>) -> tensor<64x47040xf32>
    %v1397 = stablehlo.reshape %v1396 : (tensor<64x47040xf32>) -> tensor<64x960x7x7xf32>
    %v1398 = stablehlo.constant dense<0.0> : tensor<64x960x7x7xf32>
    %v1399 = stablehlo.constant dense<6.0> : tensor<64x960x7x7xf32>
    %v1400 = stablehlo.maximum %v1397, %v1398 : tensor<64x960x7x7xf32>
    %v1401 = stablehlo.minimum %v1400, %v1399 : tensor<64x960x7x7xf32>
    %v1402 = stablehlo.reshape %v1401 : (tensor<64x960x7x7xf32>) -> tensor<64x47040xf32>
    %v1403 = stablehlo.reshape %v1402 : (tensor<64x47040xf32>) -> tensor<64x960x7x7xf32>
    %v1404 = stablehlo.convolution(%v1403, %Wp16)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x960x7x7xf32>, tensor<160x960x1x1xf32>) -> tensor<64x160x7x7xf32>
    %v1405 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<64x160x7x7xf32>
    %v1406 = stablehlo.add %v1404, %v1405 : tensor<64x160x7x7xf32>
    %v1407 = stablehlo.reshape %v1406 : (tensor<64x160x7x7xf32>) -> tensor<64x7840xf32>
    %v1408 = stablehlo.reshape %v1407 : (tensor<64x7840xf32>) -> tensor<64x160x7x7xf32>
    %v1409 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1410 = stablehlo.constant dense<49.0> : tensor<64x160x7x7xf32>
    %v1411 = stablehlo.constant dense<1.0e-5> : tensor<64x160x7x7xf32>
    %v1412 = stablehlo.reduce(%v1408 init: %v1409) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x160x7x7xf32>, tensor<f32>) -> tensor<64x160xf32>
    %v1413 = stablehlo.broadcast_in_dim %v1412, dims = [0, 1] : (tensor<64x160xf32>) -> tensor<64x160x7x7xf32>
    %v1414 = stablehlo.divide %v1413, %v1410 : tensor<64x160x7x7xf32>
    %v1415 = stablehlo.subtract %v1408, %v1414 : tensor<64x160x7x7xf32>
    %v1416 = stablehlo.multiply %v1415, %v1415 : tensor<64x160x7x7xf32>
    %v1417 = stablehlo.reduce(%v1416 init: %v1409) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x160x7x7xf32>, tensor<f32>) -> tensor<64x160xf32>
    %v1418 = stablehlo.broadcast_in_dim %v1417, dims = [0, 1] : (tensor<64x160xf32>) -> tensor<64x160x7x7xf32>
    %v1419 = stablehlo.divide %v1418, %v1410 : tensor<64x160x7x7xf32>
    %v1420 = stablehlo.add %v1419, %v1411 : tensor<64x160x7x7xf32>
    %v1421 = stablehlo.rsqrt %v1420 : tensor<64x160x7x7xf32>
    %v1422 = stablehlo.multiply %v1415, %v1421 : tensor<64x160x7x7xf32>
    %v1423 = stablehlo.broadcast_in_dim %gp16, dims = [1] : (tensor<160xf32>) -> tensor<64x160x7x7xf32>
    %v1424 = stablehlo.broadcast_in_dim %btp16, dims = [1] : (tensor<160xf32>) -> tensor<64x160x7x7xf32>
    %v1425 = stablehlo.multiply %v1422, %v1423 : tensor<64x160x7x7xf32>
    %v1426 = stablehlo.add %v1425, %v1424 : tensor<64x160x7x7xf32>
    %v1427 = stablehlo.reshape %v1426 : (tensor<64x160x7x7xf32>) -> tensor<64x7840xf32>
    %v1428 = stablehlo.reshape %v1427 : (tensor<64x7840xf32>) -> tensor<64x160x7x7xf32>
    %v1429 = stablehlo.reshape %v1340 : (tensor<64x7840xf32>) -> tensor<64x160x7x7xf32>
    %v1430 = stablehlo.add %v1428, %v1429 : tensor<64x160x7x7xf32>
    %v1431 = stablehlo.reshape %v1430 : (tensor<64x160x7x7xf32>) -> tensor<64x7840xf32>
    %v1432 = stablehlo.reshape %v1431 : (tensor<64x7840xf32>) -> tensor<64x160x7x7xf32>
    %v1433 = stablehlo.convolution(%v1432, %We17)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x160x7x7xf32>, tensor<960x160x1x1xf32>) -> tensor<64x960x7x7xf32>
    %v1434 = stablehlo.broadcast_in_dim %zb960, dims = [1] : (tensor<960xf32>) -> tensor<64x960x7x7xf32>
    %v1435 = stablehlo.add %v1433, %v1434 : tensor<64x960x7x7xf32>
    %v1436 = stablehlo.reshape %v1435 : (tensor<64x960x7x7xf32>) -> tensor<64x47040xf32>
    %v1437 = stablehlo.reshape %v1436 : (tensor<64x47040xf32>) -> tensor<64x960x7x7xf32>
    %v1438 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1439 = stablehlo.constant dense<49.0> : tensor<64x960x7x7xf32>
    %v1440 = stablehlo.constant dense<1.0e-5> : tensor<64x960x7x7xf32>
    %v1441 = stablehlo.reduce(%v1437 init: %v1438) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x960x7x7xf32>, tensor<f32>) -> tensor<64x960xf32>
    %v1442 = stablehlo.broadcast_in_dim %v1441, dims = [0, 1] : (tensor<64x960xf32>) -> tensor<64x960x7x7xf32>
    %v1443 = stablehlo.divide %v1442, %v1439 : tensor<64x960x7x7xf32>
    %v1444 = stablehlo.subtract %v1437, %v1443 : tensor<64x960x7x7xf32>
    %v1445 = stablehlo.multiply %v1444, %v1444 : tensor<64x960x7x7xf32>
    %v1446 = stablehlo.reduce(%v1445 init: %v1438) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x960x7x7xf32>, tensor<f32>) -> tensor<64x960xf32>
    %v1447 = stablehlo.broadcast_in_dim %v1446, dims = [0, 1] : (tensor<64x960xf32>) -> tensor<64x960x7x7xf32>
    %v1448 = stablehlo.divide %v1447, %v1439 : tensor<64x960x7x7xf32>
    %v1449 = stablehlo.add %v1448, %v1440 : tensor<64x960x7x7xf32>
    %v1450 = stablehlo.rsqrt %v1449 : tensor<64x960x7x7xf32>
    %v1451 = stablehlo.multiply %v1444, %v1450 : tensor<64x960x7x7xf32>
    %v1452 = stablehlo.broadcast_in_dim %ge17, dims = [1] : (tensor<960xf32>) -> tensor<64x960x7x7xf32>
    %v1453 = stablehlo.broadcast_in_dim %bte17, dims = [1] : (tensor<960xf32>) -> tensor<64x960x7x7xf32>
    %v1454 = stablehlo.multiply %v1451, %v1452 : tensor<64x960x7x7xf32>
    %v1455 = stablehlo.add %v1454, %v1453 : tensor<64x960x7x7xf32>
    %v1456 = stablehlo.reshape %v1455 : (tensor<64x960x7x7xf32>) -> tensor<64x47040xf32>
    %v1457 = stablehlo.reshape %v1456 : (tensor<64x47040xf32>) -> tensor<64x960x7x7xf32>
    %v1458 = stablehlo.constant dense<0.0> : tensor<64x960x7x7xf32>
    %v1459 = stablehlo.constant dense<6.0> : tensor<64x960x7x7xf32>
    %v1460 = stablehlo.maximum %v1457, %v1458 : tensor<64x960x7x7xf32>
    %v1461 = stablehlo.minimum %v1460, %v1459 : tensor<64x960x7x7xf32>
    %v1462 = stablehlo.reshape %v1461 : (tensor<64x960x7x7xf32>) -> tensor<64x47040xf32>
    %v1463 = stablehlo.reshape %v1462 : (tensor<64x47040xf32>) -> tensor<64x960x7x7xf32>
    %v1464 = stablehlo.convolution(%v1463, %Wd17)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 960 : i64} : (tensor<64x960x7x7xf32>, tensor<960x1x3x3xf32>) -> tensor<64x960x7x7xf32>
    %v1465 = stablehlo.broadcast_in_dim %zb960, dims = [1] : (tensor<960xf32>) -> tensor<64x960x7x7xf32>
    %v1466 = stablehlo.add %v1464, %v1465 : tensor<64x960x7x7xf32>
    %v1467 = stablehlo.reshape %v1466 : (tensor<64x960x7x7xf32>) -> tensor<64x47040xf32>
    %v1468 = stablehlo.reshape %v1467 : (tensor<64x47040xf32>) -> tensor<64x960x7x7xf32>
    %v1469 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1470 = stablehlo.constant dense<49.0> : tensor<64x960x7x7xf32>
    %v1471 = stablehlo.constant dense<1.0e-5> : tensor<64x960x7x7xf32>
    %v1472 = stablehlo.reduce(%v1468 init: %v1469) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x960x7x7xf32>, tensor<f32>) -> tensor<64x960xf32>
    %v1473 = stablehlo.broadcast_in_dim %v1472, dims = [0, 1] : (tensor<64x960xf32>) -> tensor<64x960x7x7xf32>
    %v1474 = stablehlo.divide %v1473, %v1470 : tensor<64x960x7x7xf32>
    %v1475 = stablehlo.subtract %v1468, %v1474 : tensor<64x960x7x7xf32>
    %v1476 = stablehlo.multiply %v1475, %v1475 : tensor<64x960x7x7xf32>
    %v1477 = stablehlo.reduce(%v1476 init: %v1469) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x960x7x7xf32>, tensor<f32>) -> tensor<64x960xf32>
    %v1478 = stablehlo.broadcast_in_dim %v1477, dims = [0, 1] : (tensor<64x960xf32>) -> tensor<64x960x7x7xf32>
    %v1479 = stablehlo.divide %v1478, %v1470 : tensor<64x960x7x7xf32>
    %v1480 = stablehlo.add %v1479, %v1471 : tensor<64x960x7x7xf32>
    %v1481 = stablehlo.rsqrt %v1480 : tensor<64x960x7x7xf32>
    %v1482 = stablehlo.multiply %v1475, %v1481 : tensor<64x960x7x7xf32>
    %v1483 = stablehlo.broadcast_in_dim %gd17, dims = [1] : (tensor<960xf32>) -> tensor<64x960x7x7xf32>
    %v1484 = stablehlo.broadcast_in_dim %btd17, dims = [1] : (tensor<960xf32>) -> tensor<64x960x7x7xf32>
    %v1485 = stablehlo.multiply %v1482, %v1483 : tensor<64x960x7x7xf32>
    %v1486 = stablehlo.add %v1485, %v1484 : tensor<64x960x7x7xf32>
    %v1487 = stablehlo.reshape %v1486 : (tensor<64x960x7x7xf32>) -> tensor<64x47040xf32>
    %v1488 = stablehlo.reshape %v1487 : (tensor<64x47040xf32>) -> tensor<64x960x7x7xf32>
    %v1489 = stablehlo.constant dense<0.0> : tensor<64x960x7x7xf32>
    %v1490 = stablehlo.constant dense<6.0> : tensor<64x960x7x7xf32>
    %v1491 = stablehlo.maximum %v1488, %v1489 : tensor<64x960x7x7xf32>
    %v1492 = stablehlo.minimum %v1491, %v1490 : tensor<64x960x7x7xf32>
    %v1493 = stablehlo.reshape %v1492 : (tensor<64x960x7x7xf32>) -> tensor<64x47040xf32>
    %v1494 = stablehlo.reshape %v1493 : (tensor<64x47040xf32>) -> tensor<64x960x7x7xf32>
    %v1495 = stablehlo.convolution(%v1494, %Wp17)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x960x7x7xf32>, tensor<320x960x1x1xf32>) -> tensor<64x320x7x7xf32>
    %v1496 = stablehlo.broadcast_in_dim %zb320, dims = [1] : (tensor<320xf32>) -> tensor<64x320x7x7xf32>
    %v1497 = stablehlo.add %v1495, %v1496 : tensor<64x320x7x7xf32>
    %v1498 = stablehlo.reshape %v1497 : (tensor<64x320x7x7xf32>) -> tensor<64x15680xf32>
    %v1499 = stablehlo.reshape %v1498 : (tensor<64x15680xf32>) -> tensor<64x320x7x7xf32>
    %v1500 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1501 = stablehlo.constant dense<49.0> : tensor<64x320x7x7xf32>
    %v1502 = stablehlo.constant dense<1.0e-5> : tensor<64x320x7x7xf32>
    %v1503 = stablehlo.reduce(%v1499 init: %v1500) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x320x7x7xf32>, tensor<f32>) -> tensor<64x320xf32>
    %v1504 = stablehlo.broadcast_in_dim %v1503, dims = [0, 1] : (tensor<64x320xf32>) -> tensor<64x320x7x7xf32>
    %v1505 = stablehlo.divide %v1504, %v1501 : tensor<64x320x7x7xf32>
    %v1506 = stablehlo.subtract %v1499, %v1505 : tensor<64x320x7x7xf32>
    %v1507 = stablehlo.multiply %v1506, %v1506 : tensor<64x320x7x7xf32>
    %v1508 = stablehlo.reduce(%v1507 init: %v1500) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x320x7x7xf32>, tensor<f32>) -> tensor<64x320xf32>
    %v1509 = stablehlo.broadcast_in_dim %v1508, dims = [0, 1] : (tensor<64x320xf32>) -> tensor<64x320x7x7xf32>
    %v1510 = stablehlo.divide %v1509, %v1501 : tensor<64x320x7x7xf32>
    %v1511 = stablehlo.add %v1510, %v1502 : tensor<64x320x7x7xf32>
    %v1512 = stablehlo.rsqrt %v1511 : tensor<64x320x7x7xf32>
    %v1513 = stablehlo.multiply %v1506, %v1512 : tensor<64x320x7x7xf32>
    %v1514 = stablehlo.broadcast_in_dim %gp17, dims = [1] : (tensor<320xf32>) -> tensor<64x320x7x7xf32>
    %v1515 = stablehlo.broadcast_in_dim %btp17, dims = [1] : (tensor<320xf32>) -> tensor<64x320x7x7xf32>
    %v1516 = stablehlo.multiply %v1513, %v1514 : tensor<64x320x7x7xf32>
    %v1517 = stablehlo.add %v1516, %v1515 : tensor<64x320x7x7xf32>
    %v1518 = stablehlo.reshape %v1517 : (tensor<64x320x7x7xf32>) -> tensor<64x15680xf32>
    %v1519 = stablehlo.reshape %v1518 : (tensor<64x15680xf32>) -> tensor<64x320x7x7xf32>
    %v1520 = stablehlo.convolution(%v1519, %Wh)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x320x7x7xf32>, tensor<1280x320x1x1xf32>) -> tensor<64x1280x7x7xf32>
    %v1521 = stablehlo.broadcast_in_dim %zb1280, dims = [1] : (tensor<1280xf32>) -> tensor<64x1280x7x7xf32>
    %v1522 = stablehlo.add %v1520, %v1521 : tensor<64x1280x7x7xf32>
    %v1523 = stablehlo.reshape %v1522 : (tensor<64x1280x7x7xf32>) -> tensor<64x62720xf32>
    %v1524 = stablehlo.reshape %v1523 : (tensor<64x62720xf32>) -> tensor<64x1280x7x7xf32>
    %v1525 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1526 = stablehlo.constant dense<49.0> : tensor<64x1280x7x7xf32>
    %v1527 = stablehlo.constant dense<1.0e-5> : tensor<64x1280x7x7xf32>
    %v1528 = stablehlo.reduce(%v1524 init: %v1525) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x1280x7x7xf32>, tensor<f32>) -> tensor<64x1280xf32>
    %v1529 = stablehlo.broadcast_in_dim %v1528, dims = [0, 1] : (tensor<64x1280xf32>) -> tensor<64x1280x7x7xf32>
    %v1530 = stablehlo.divide %v1529, %v1526 : tensor<64x1280x7x7xf32>
    %v1531 = stablehlo.subtract %v1524, %v1530 : tensor<64x1280x7x7xf32>
    %v1532 = stablehlo.multiply %v1531, %v1531 : tensor<64x1280x7x7xf32>
    %v1533 = stablehlo.reduce(%v1532 init: %v1525) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x1280x7x7xf32>, tensor<f32>) -> tensor<64x1280xf32>
    %v1534 = stablehlo.broadcast_in_dim %v1533, dims = [0, 1] : (tensor<64x1280xf32>) -> tensor<64x1280x7x7xf32>
    %v1535 = stablehlo.divide %v1534, %v1526 : tensor<64x1280x7x7xf32>
    %v1536 = stablehlo.add %v1535, %v1527 : tensor<64x1280x7x7xf32>
    %v1537 = stablehlo.rsqrt %v1536 : tensor<64x1280x7x7xf32>
    %v1538 = stablehlo.multiply %v1531, %v1537 : tensor<64x1280x7x7xf32>
    %v1539 = stablehlo.broadcast_in_dim %gh, dims = [1] : (tensor<1280xf32>) -> tensor<64x1280x7x7xf32>
    %v1540 = stablehlo.broadcast_in_dim %bth, dims = [1] : (tensor<1280xf32>) -> tensor<64x1280x7x7xf32>
    %v1541 = stablehlo.multiply %v1538, %v1539 : tensor<64x1280x7x7xf32>
    %v1542 = stablehlo.add %v1541, %v1540 : tensor<64x1280x7x7xf32>
    %v1543 = stablehlo.reshape %v1542 : (tensor<64x1280x7x7xf32>) -> tensor<64x62720xf32>
    %v1544 = stablehlo.reshape %v1543 : (tensor<64x62720xf32>) -> tensor<64x1280x7x7xf32>
    %v1545 = stablehlo.constant dense<0.0> : tensor<64x1280x7x7xf32>
    %v1546 = stablehlo.constant dense<6.0> : tensor<64x1280x7x7xf32>
    %v1547 = stablehlo.maximum %v1544, %v1545 : tensor<64x1280x7x7xf32>
    %v1548 = stablehlo.minimum %v1547, %v1546 : tensor<64x1280x7x7xf32>
    %v1549 = stablehlo.reshape %v1548 : (tensor<64x1280x7x7xf32>) -> tensor<64x62720xf32>
    %v1550 = stablehlo.reshape %v1549 : (tensor<64x62720xf32>) -> tensor<64x1280x7x7xf32>
    %v1551 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1552 = stablehlo.reduce(%v1550 init: %v1551) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x1280x7x7xf32>, tensor<f32>) -> tensor<64x1280xf32>
    %v1553 = stablehlo.constant dense<49.0> : tensor<64x1280xf32>
    %v1554 = stablehlo.divide %v1552, %v1553 : tensor<64x1280xf32>
    %v1555 = stablehlo.dot_general %v1554, %Wfc, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x1280xf32>, tensor<1280x1000xf32>) -> tensor<64x1000xf32>
    %v1556 = stablehlo.broadcast_in_dim %bfc, dims = [1] : (tensor<1000xf32>) -> tensor<64x1000xf32>
    %v1557 = stablehlo.add %v1555, %v1556 : tensor<64x1000xf32>
    return %v1557 : tensor<64x1000xf32>
  }
}

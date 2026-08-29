module @m {
  func.func @mobilenetv2_train_step(%x: tensor<32x150528xf32>, %Ws: tensor<32x3x3x3xf32>, %gs: tensor<32xf32>, %bts: tensor<32xf32>, %Wd1: tensor<32x1x3x3xf32>, %gd1: tensor<32xf32>, %btd1: tensor<32xf32>, %Wp1: tensor<16x32x1x1xf32>, %gp1: tensor<16xf32>, %btp1: tensor<16xf32>, %We2: tensor<96x16x1x1xf32>, %ge2: tensor<96xf32>, %bte2: tensor<96xf32>, %Wd2: tensor<96x1x3x3xf32>, %gd2: tensor<96xf32>, %btd2: tensor<96xf32>, %Wp2: tensor<24x96x1x1xf32>, %gp2: tensor<24xf32>, %btp2: tensor<24xf32>, %We3: tensor<144x24x1x1xf32>, %ge3: tensor<144xf32>, %bte3: tensor<144xf32>, %Wd3: tensor<144x1x3x3xf32>, %gd3: tensor<144xf32>, %btd3: tensor<144xf32>, %Wp3: tensor<24x144x1x1xf32>, %gp3: tensor<24xf32>, %btp3: tensor<24xf32>, %We4: tensor<144x24x1x1xf32>, %ge4: tensor<144xf32>, %bte4: tensor<144xf32>, %Wd4: tensor<144x1x3x3xf32>, %gd4: tensor<144xf32>, %btd4: tensor<144xf32>, %Wp4: tensor<32x144x1x1xf32>, %gp4: tensor<32xf32>, %btp4: tensor<32xf32>, %We5: tensor<192x32x1x1xf32>, %ge5: tensor<192xf32>, %bte5: tensor<192xf32>, %Wd5: tensor<192x1x3x3xf32>, %gd5: tensor<192xf32>, %btd5: tensor<192xf32>, %Wp5: tensor<32x192x1x1xf32>, %gp5: tensor<32xf32>, %btp5: tensor<32xf32>, %We6: tensor<192x32x1x1xf32>, %ge6: tensor<192xf32>, %bte6: tensor<192xf32>, %Wd6: tensor<192x1x3x3xf32>, %gd6: tensor<192xf32>, %btd6: tensor<192xf32>, %Wp6: tensor<32x192x1x1xf32>, %gp6: tensor<32xf32>, %btp6: tensor<32xf32>, %We7: tensor<192x32x1x1xf32>, %ge7: tensor<192xf32>, %bte7: tensor<192xf32>, %Wd7: tensor<192x1x3x3xf32>, %gd7: tensor<192xf32>, %btd7: tensor<192xf32>, %Wp7: tensor<64x192x1x1xf32>, %gp7: tensor<64xf32>, %btp7: tensor<64xf32>, %We8: tensor<384x64x1x1xf32>, %ge8: tensor<384xf32>, %bte8: tensor<384xf32>, %Wd8: tensor<384x1x3x3xf32>, %gd8: tensor<384xf32>, %btd8: tensor<384xf32>, %Wp8: tensor<64x384x1x1xf32>, %gp8: tensor<64xf32>, %btp8: tensor<64xf32>, %We9: tensor<384x64x1x1xf32>, %ge9: tensor<384xf32>, %bte9: tensor<384xf32>, %Wd9: tensor<384x1x3x3xf32>, %gd9: tensor<384xf32>, %btd9: tensor<384xf32>, %Wp9: tensor<64x384x1x1xf32>, %gp9: tensor<64xf32>, %btp9: tensor<64xf32>, %We10: tensor<384x64x1x1xf32>, %ge10: tensor<384xf32>, %bte10: tensor<384xf32>, %Wd10: tensor<384x1x3x3xf32>, %gd10: tensor<384xf32>, %btd10: tensor<384xf32>, %Wp10: tensor<64x384x1x1xf32>, %gp10: tensor<64xf32>, %btp10: tensor<64xf32>, %We11: tensor<384x64x1x1xf32>, %ge11: tensor<384xf32>, %bte11: tensor<384xf32>, %Wd11: tensor<384x1x3x3xf32>, %gd11: tensor<384xf32>, %btd11: tensor<384xf32>, %Wp11: tensor<96x384x1x1xf32>, %gp11: tensor<96xf32>, %btp11: tensor<96xf32>, %We12: tensor<576x96x1x1xf32>, %ge12: tensor<576xf32>, %bte12: tensor<576xf32>, %Wd12: tensor<576x1x3x3xf32>, %gd12: tensor<576xf32>, %btd12: tensor<576xf32>, %Wp12: tensor<96x576x1x1xf32>, %gp12: tensor<96xf32>, %btp12: tensor<96xf32>, %We13: tensor<576x96x1x1xf32>, %ge13: tensor<576xf32>, %bte13: tensor<576xf32>, %Wd13: tensor<576x1x3x3xf32>, %gd13: tensor<576xf32>, %btd13: tensor<576xf32>, %Wp13: tensor<96x576x1x1xf32>, %gp13: tensor<96xf32>, %btp13: tensor<96xf32>, %We14: tensor<576x96x1x1xf32>, %ge14: tensor<576xf32>, %bte14: tensor<576xf32>, %Wd14: tensor<576x1x3x3xf32>, %gd14: tensor<576xf32>, %btd14: tensor<576xf32>, %Wp14: tensor<160x576x1x1xf32>, %gp14: tensor<160xf32>, %btp14: tensor<160xf32>, %We15: tensor<960x160x1x1xf32>, %ge15: tensor<960xf32>, %bte15: tensor<960xf32>, %Wd15: tensor<960x1x3x3xf32>, %gd15: tensor<960xf32>, %btd15: tensor<960xf32>, %Wp15: tensor<160x960x1x1xf32>, %gp15: tensor<160xf32>, %btp15: tensor<160xf32>, %We16: tensor<960x160x1x1xf32>, %ge16: tensor<960xf32>, %bte16: tensor<960xf32>, %Wd16: tensor<960x1x3x3xf32>, %gd16: tensor<960xf32>, %btd16: tensor<960xf32>, %Wp16: tensor<160x960x1x1xf32>, %gp16: tensor<160xf32>, %btp16: tensor<160xf32>, %We17: tensor<960x160x1x1xf32>, %ge17: tensor<960xf32>, %bte17: tensor<960xf32>, %Wd17: tensor<960x1x3x3xf32>, %gd17: tensor<960xf32>, %btd17: tensor<960xf32>, %Wp17: tensor<320x960x1x1xf32>, %gp17: tensor<320xf32>, %btp17: tensor<320xf32>, %Wh: tensor<1280x320x1x1xf32>, %gh: tensor<1280xf32>, %bth: tensor<1280xf32>, %Wfc: tensor<1280x10xf32>, %bfc: tensor<10xf32>, %onehot: tensor<32x10xf32>) -> (tensor<32x3x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x1x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<16x32x1x1xf32>, tensor<16xf32>, tensor<16xf32>, tensor<96x16x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x3x3xf32>, tensor<96xf32>, tensor<96xf32>, tensor<24x96x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144x24x1x1xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x1x3x3xf32>, tensor<144xf32>, tensor<144xf32>, tensor<24x144x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144x24x1x1xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x1x3x3xf32>, tensor<144xf32>, tensor<144xf32>, tensor<32x144x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<192x32x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x3x3xf32>, tensor<192xf32>, tensor<192xf32>, tensor<32x192x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<192x32x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x3x3xf32>, tensor<192xf32>, tensor<192xf32>, tensor<32x192x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<192x32x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x3x3xf32>, tensor<192xf32>, tensor<192xf32>, tensor<64x192x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64x384x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64x384x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64x384x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<576x96x1x1xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576x1x3x3xf32>, tensor<576xf32>, tensor<576xf32>, tensor<96x576x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<576x96x1x1xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576x1x3x3xf32>, tensor<576xf32>, tensor<576xf32>, tensor<96x576x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<576x96x1x1xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576x1x3x3xf32>, tensor<576xf32>, tensor<576xf32>, tensor<160x576x1x1xf32>, tensor<160xf32>, tensor<160xf32>, tensor<960x160x1x1xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960x1x3x3xf32>, tensor<960xf32>, tensor<960xf32>, tensor<160x960x1x1xf32>, tensor<160xf32>, tensor<160xf32>, tensor<960x160x1x1xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960x1x3x3xf32>, tensor<960xf32>, tensor<960xf32>, tensor<160x960x1x1xf32>, tensor<160xf32>, tensor<160xf32>, tensor<960x160x1x1xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960x1x3x3xf32>, tensor<960xf32>, tensor<960xf32>, tensor<320x960x1x1xf32>, tensor<320xf32>, tensor<320xf32>, tensor<1280x320x1x1xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280x10xf32>, tensor<10xf32>) {
    // ── MobileNetV2 (17-block paper) train step: every line is pretty(verified AST node) ──
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
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3x224x224xf32>, tensor<32x3x3x3xf32>) -> tensor<32x32x112x112xf32>
    %v2 = stablehlo.broadcast_in_dim %zb32, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v3 = stablehlo.add %v1, %v2 : tensor<32x32x112x112xf32>
    %v4 = stablehlo.reshape %v3 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v5 = stablehlo.reshape %v4 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6 = stablehlo.constant dense<0.0> : tensor<f32>
    %v7 = stablehlo.constant dense<12544.0> : tensor<32x32x112x112xf32>
    %v8 = stablehlo.constant dense<1.0e-5> : tensor<32x32x112x112xf32>
    %v9 = stablehlo.reduce(%v5 init: %v6) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v10 = stablehlo.broadcast_in_dim %v9, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x112x112xf32>
    %v11 = stablehlo.divide %v10, %v7 : tensor<32x32x112x112xf32>
    %v12 = stablehlo.subtract %v5, %v11 : tensor<32x32x112x112xf32>
    %v13 = stablehlo.multiply %v12, %v12 : tensor<32x32x112x112xf32>
    %v14 = stablehlo.reduce(%v13 init: %v6) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v15 = stablehlo.broadcast_in_dim %v14, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x112x112xf32>
    %v16 = stablehlo.divide %v15, %v7 : tensor<32x32x112x112xf32>
    %v17 = stablehlo.add %v16, %v8 : tensor<32x32x112x112xf32>
    %v18 = stablehlo.rsqrt %v17 : tensor<32x32x112x112xf32>
    %v19 = stablehlo.multiply %v12, %v18 : tensor<32x32x112x112xf32>
    %v20 = stablehlo.broadcast_in_dim %gs, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v21 = stablehlo.broadcast_in_dim %bts, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v22 = stablehlo.multiply %v19, %v20 : tensor<32x32x112x112xf32>
    %v23 = stablehlo.add %v22, %v21 : tensor<32x32x112x112xf32>
    %v24 = stablehlo.reshape %v23 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v25 = stablehlo.reshape %v24 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v26 = stablehlo.constant dense<0.0> : tensor<32x32x112x112xf32>
    %v27 = stablehlo.constant dense<6.0> : tensor<32x32x112x112xf32>
    %v28 = stablehlo.maximum %v25, %v26 : tensor<32x32x112x112xf32>
    %v29 = stablehlo.minimum %v28, %v27 : tensor<32x32x112x112xf32>
    %v30 = stablehlo.reshape %v29 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v31 = stablehlo.reshape %v30 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v32 = stablehlo.convolution(%v31, %Wd1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 32 : i64} : (tensor<32x32x112x112xf32>, tensor<32x1x3x3xf32>) -> tensor<32x32x112x112xf32>
    %v33 = stablehlo.broadcast_in_dim %zb32, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v34 = stablehlo.add %v32, %v33 : tensor<32x32x112x112xf32>
    %v35 = stablehlo.reshape %v34 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v36 = stablehlo.reshape %v35 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v37 = stablehlo.constant dense<0.0> : tensor<f32>
    %v38 = stablehlo.constant dense<12544.0> : tensor<32x32x112x112xf32>
    %v39 = stablehlo.constant dense<1.0e-5> : tensor<32x32x112x112xf32>
    %v40 = stablehlo.reduce(%v36 init: %v37) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v41 = stablehlo.broadcast_in_dim %v40, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x112x112xf32>
    %v42 = stablehlo.divide %v41, %v38 : tensor<32x32x112x112xf32>
    %v43 = stablehlo.subtract %v36, %v42 : tensor<32x32x112x112xf32>
    %v44 = stablehlo.multiply %v43, %v43 : tensor<32x32x112x112xf32>
    %v45 = stablehlo.reduce(%v44 init: %v37) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v46 = stablehlo.broadcast_in_dim %v45, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x112x112xf32>
    %v47 = stablehlo.divide %v46, %v38 : tensor<32x32x112x112xf32>
    %v48 = stablehlo.add %v47, %v39 : tensor<32x32x112x112xf32>
    %v49 = stablehlo.rsqrt %v48 : tensor<32x32x112x112xf32>
    %v50 = stablehlo.multiply %v43, %v49 : tensor<32x32x112x112xf32>
    %v51 = stablehlo.broadcast_in_dim %gd1, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v52 = stablehlo.broadcast_in_dim %btd1, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v53 = stablehlo.multiply %v50, %v51 : tensor<32x32x112x112xf32>
    %v54 = stablehlo.add %v53, %v52 : tensor<32x32x112x112xf32>
    %v55 = stablehlo.reshape %v54 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v56 = stablehlo.reshape %v55 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v57 = stablehlo.constant dense<0.0> : tensor<32x32x112x112xf32>
    %v58 = stablehlo.constant dense<6.0> : tensor<32x32x112x112xf32>
    %v59 = stablehlo.maximum %v56, %v57 : tensor<32x32x112x112xf32>
    %v60 = stablehlo.minimum %v59, %v58 : tensor<32x32x112x112xf32>
    %v61 = stablehlo.reshape %v60 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v62 = stablehlo.reshape %v61 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v63 = stablehlo.convolution(%v62, %Wp1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x112x112xf32>, tensor<16x32x1x1xf32>) -> tensor<32x16x112x112xf32>
    %v64 = stablehlo.broadcast_in_dim %zb16, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v65 = stablehlo.add %v63, %v64 : tensor<32x16x112x112xf32>
    %v66 = stablehlo.reshape %v65 : (tensor<32x16x112x112xf32>) -> tensor<32x200704xf32>
    %v67 = stablehlo.reshape %v66 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v68 = stablehlo.constant dense<0.0> : tensor<f32>
    %v69 = stablehlo.constant dense<12544.0> : tensor<32x16x112x112xf32>
    %v70 = stablehlo.constant dense<1.0e-5> : tensor<32x16x112x112xf32>
    %v71 = stablehlo.reduce(%v67 init: %v68) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<32x16xf32>
    %v72 = stablehlo.broadcast_in_dim %v71, dims = [0, 1] : (tensor<32x16xf32>) -> tensor<32x16x112x112xf32>
    %v73 = stablehlo.divide %v72, %v69 : tensor<32x16x112x112xf32>
    %v74 = stablehlo.subtract %v67, %v73 : tensor<32x16x112x112xf32>
    %v75 = stablehlo.multiply %v74, %v74 : tensor<32x16x112x112xf32>
    %v76 = stablehlo.reduce(%v75 init: %v68) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<32x16xf32>
    %v77 = stablehlo.broadcast_in_dim %v76, dims = [0, 1] : (tensor<32x16xf32>) -> tensor<32x16x112x112xf32>
    %v78 = stablehlo.divide %v77, %v69 : tensor<32x16x112x112xf32>
    %v79 = stablehlo.add %v78, %v70 : tensor<32x16x112x112xf32>
    %v80 = stablehlo.rsqrt %v79 : tensor<32x16x112x112xf32>
    %v81 = stablehlo.multiply %v74, %v80 : tensor<32x16x112x112xf32>
    %v82 = stablehlo.broadcast_in_dim %gp1, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v83 = stablehlo.broadcast_in_dim %btp1, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v84 = stablehlo.multiply %v81, %v82 : tensor<32x16x112x112xf32>
    %v85 = stablehlo.add %v84, %v83 : tensor<32x16x112x112xf32>
    %v86 = stablehlo.reshape %v85 : (tensor<32x16x112x112xf32>) -> tensor<32x200704xf32>
    %v87 = stablehlo.reshape %v86 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v88 = stablehlo.convolution(%v87, %We2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x16x112x112xf32>, tensor<96x16x1x1xf32>) -> tensor<32x96x112x112xf32>
    %v89 = stablehlo.broadcast_in_dim %zb96, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v90 = stablehlo.add %v88, %v89 : tensor<32x96x112x112xf32>
    %v91 = stablehlo.reshape %v90 : (tensor<32x96x112x112xf32>) -> tensor<32x1204224xf32>
    %v92 = stablehlo.reshape %v91 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v93 = stablehlo.constant dense<0.0> : tensor<f32>
    %v94 = stablehlo.constant dense<12544.0> : tensor<32x96x112x112xf32>
    %v95 = stablehlo.constant dense<1.0e-5> : tensor<32x96x112x112xf32>
    %v96 = stablehlo.reduce(%v92 init: %v93) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v97 = stablehlo.broadcast_in_dim %v96, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x112x112xf32>
    %v98 = stablehlo.divide %v97, %v94 : tensor<32x96x112x112xf32>
    %v99 = stablehlo.subtract %v92, %v98 : tensor<32x96x112x112xf32>
    %v100 = stablehlo.multiply %v99, %v99 : tensor<32x96x112x112xf32>
    %v101 = stablehlo.reduce(%v100 init: %v93) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v102 = stablehlo.broadcast_in_dim %v101, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x112x112xf32>
    %v103 = stablehlo.divide %v102, %v94 : tensor<32x96x112x112xf32>
    %v104 = stablehlo.add %v103, %v95 : tensor<32x96x112x112xf32>
    %v105 = stablehlo.rsqrt %v104 : tensor<32x96x112x112xf32>
    %v106 = stablehlo.multiply %v99, %v105 : tensor<32x96x112x112xf32>
    %v107 = stablehlo.broadcast_in_dim %ge2, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v108 = stablehlo.broadcast_in_dim %bte2, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v109 = stablehlo.multiply %v106, %v107 : tensor<32x96x112x112xf32>
    %v110 = stablehlo.add %v109, %v108 : tensor<32x96x112x112xf32>
    %v111 = stablehlo.reshape %v110 : (tensor<32x96x112x112xf32>) -> tensor<32x1204224xf32>
    %v112 = stablehlo.reshape %v111 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v113 = stablehlo.constant dense<0.0> : tensor<32x96x112x112xf32>
    %v114 = stablehlo.constant dense<6.0> : tensor<32x96x112x112xf32>
    %v115 = stablehlo.maximum %v112, %v113 : tensor<32x96x112x112xf32>
    %v116 = stablehlo.minimum %v115, %v114 : tensor<32x96x112x112xf32>
    %v117 = stablehlo.reshape %v116 : (tensor<32x96x112x112xf32>) -> tensor<32x1204224xf32>
    %v118 = stablehlo.reshape %v117 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v119 = stablehlo.convolution(%v118, %Wd2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x112x112xf32>, tensor<96x1x3x3xf32>) -> tensor<32x96x56x56xf32>
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
    %v138 = stablehlo.broadcast_in_dim %gd2, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v139 = stablehlo.broadcast_in_dim %btd2, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
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
    %v150 = stablehlo.convolution(%v149, %Wp2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<24x96x1x1xf32>) -> tensor<32x24x56x56xf32>
    %v151 = stablehlo.broadcast_in_dim %zb24, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v152 = stablehlo.add %v150, %v151 : tensor<32x24x56x56xf32>
    %v153 = stablehlo.reshape %v152 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v154 = stablehlo.reshape %v153 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v155 = stablehlo.constant dense<0.0> : tensor<f32>
    %v156 = stablehlo.constant dense<3136.0> : tensor<32x24x56x56xf32>
    %v157 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v158 = stablehlo.reduce(%v154 init: %v155) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v159 = stablehlo.broadcast_in_dim %v158, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v160 = stablehlo.divide %v159, %v156 : tensor<32x24x56x56xf32>
    %v161 = stablehlo.subtract %v154, %v160 : tensor<32x24x56x56xf32>
    %v162 = stablehlo.multiply %v161, %v161 : tensor<32x24x56x56xf32>
    %v163 = stablehlo.reduce(%v162 init: %v155) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v164 = stablehlo.broadcast_in_dim %v163, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v165 = stablehlo.divide %v164, %v156 : tensor<32x24x56x56xf32>
    %v166 = stablehlo.add %v165, %v157 : tensor<32x24x56x56xf32>
    %v167 = stablehlo.rsqrt %v166 : tensor<32x24x56x56xf32>
    %v168 = stablehlo.multiply %v161, %v167 : tensor<32x24x56x56xf32>
    %v169 = stablehlo.broadcast_in_dim %gp2, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v170 = stablehlo.broadcast_in_dim %btp2, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v171 = stablehlo.multiply %v168, %v169 : tensor<32x24x56x56xf32>
    %v172 = stablehlo.add %v171, %v170 : tensor<32x24x56x56xf32>
    %v173 = stablehlo.reshape %v172 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v174 = stablehlo.reshape %v173 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v175 = stablehlo.convolution(%v174, %We3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<144x24x1x1xf32>) -> tensor<32x144x56x56xf32>
    %v176 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v177 = stablehlo.add %v175, %v176 : tensor<32x144x56x56xf32>
    %v178 = stablehlo.reshape %v177 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v179 = stablehlo.reshape %v178 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v180 = stablehlo.constant dense<0.0> : tensor<f32>
    %v181 = stablehlo.constant dense<3136.0> : tensor<32x144x56x56xf32>
    %v182 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v183 = stablehlo.reduce(%v179 init: %v180) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v184 = stablehlo.broadcast_in_dim %v183, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v185 = stablehlo.divide %v184, %v181 : tensor<32x144x56x56xf32>
    %v186 = stablehlo.subtract %v179, %v185 : tensor<32x144x56x56xf32>
    %v187 = stablehlo.multiply %v186, %v186 : tensor<32x144x56x56xf32>
    %v188 = stablehlo.reduce(%v187 init: %v180) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v189 = stablehlo.broadcast_in_dim %v188, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v190 = stablehlo.divide %v189, %v181 : tensor<32x144x56x56xf32>
    %v191 = stablehlo.add %v190, %v182 : tensor<32x144x56x56xf32>
    %v192 = stablehlo.rsqrt %v191 : tensor<32x144x56x56xf32>
    %v193 = stablehlo.multiply %v186, %v192 : tensor<32x144x56x56xf32>
    %v194 = stablehlo.broadcast_in_dim %ge3, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v195 = stablehlo.broadcast_in_dim %bte3, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v196 = stablehlo.multiply %v193, %v194 : tensor<32x144x56x56xf32>
    %v197 = stablehlo.add %v196, %v195 : tensor<32x144x56x56xf32>
    %v198 = stablehlo.reshape %v197 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v199 = stablehlo.reshape %v198 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v200 = stablehlo.constant dense<0.0> : tensor<32x144x56x56xf32>
    %v201 = stablehlo.constant dense<6.0> : tensor<32x144x56x56xf32>
    %v202 = stablehlo.maximum %v199, %v200 : tensor<32x144x56x56xf32>
    %v203 = stablehlo.minimum %v202, %v201 : tensor<32x144x56x56xf32>
    %v204 = stablehlo.reshape %v203 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v205 = stablehlo.reshape %v204 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v206 = stablehlo.convolution(%v205, %Wd3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 144 : i64} : (tensor<32x144x56x56xf32>, tensor<144x1x3x3xf32>) -> tensor<32x144x56x56xf32>
    %v207 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v208 = stablehlo.add %v206, %v207 : tensor<32x144x56x56xf32>
    %v209 = stablehlo.reshape %v208 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v210 = stablehlo.reshape %v209 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v211 = stablehlo.constant dense<0.0> : tensor<f32>
    %v212 = stablehlo.constant dense<3136.0> : tensor<32x144x56x56xf32>
    %v213 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v214 = stablehlo.reduce(%v210 init: %v211) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v215 = stablehlo.broadcast_in_dim %v214, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v216 = stablehlo.divide %v215, %v212 : tensor<32x144x56x56xf32>
    %v217 = stablehlo.subtract %v210, %v216 : tensor<32x144x56x56xf32>
    %v218 = stablehlo.multiply %v217, %v217 : tensor<32x144x56x56xf32>
    %v219 = stablehlo.reduce(%v218 init: %v211) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v220 = stablehlo.broadcast_in_dim %v219, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v221 = stablehlo.divide %v220, %v212 : tensor<32x144x56x56xf32>
    %v222 = stablehlo.add %v221, %v213 : tensor<32x144x56x56xf32>
    %v223 = stablehlo.rsqrt %v222 : tensor<32x144x56x56xf32>
    %v224 = stablehlo.multiply %v217, %v223 : tensor<32x144x56x56xf32>
    %v225 = stablehlo.broadcast_in_dim %gd3, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v226 = stablehlo.broadcast_in_dim %btd3, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v227 = stablehlo.multiply %v224, %v225 : tensor<32x144x56x56xf32>
    %v228 = stablehlo.add %v227, %v226 : tensor<32x144x56x56xf32>
    %v229 = stablehlo.reshape %v228 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v230 = stablehlo.reshape %v229 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v231 = stablehlo.constant dense<0.0> : tensor<32x144x56x56xf32>
    %v232 = stablehlo.constant dense<6.0> : tensor<32x144x56x56xf32>
    %v233 = stablehlo.maximum %v230, %v231 : tensor<32x144x56x56xf32>
    %v234 = stablehlo.minimum %v233, %v232 : tensor<32x144x56x56xf32>
    %v235 = stablehlo.reshape %v234 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v236 = stablehlo.reshape %v235 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v237 = stablehlo.convolution(%v236, %Wp3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x144x56x56xf32>, tensor<24x144x1x1xf32>) -> tensor<32x24x56x56xf32>
    %v238 = stablehlo.broadcast_in_dim %zb24, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v239 = stablehlo.add %v237, %v238 : tensor<32x24x56x56xf32>
    %v240 = stablehlo.reshape %v239 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v241 = stablehlo.reshape %v240 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v242 = stablehlo.constant dense<0.0> : tensor<f32>
    %v243 = stablehlo.constant dense<3136.0> : tensor<32x24x56x56xf32>
    %v244 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v245 = stablehlo.reduce(%v241 init: %v242) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v246 = stablehlo.broadcast_in_dim %v245, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v247 = stablehlo.divide %v246, %v243 : tensor<32x24x56x56xf32>
    %v248 = stablehlo.subtract %v241, %v247 : tensor<32x24x56x56xf32>
    %v249 = stablehlo.multiply %v248, %v248 : tensor<32x24x56x56xf32>
    %v250 = stablehlo.reduce(%v249 init: %v242) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v251 = stablehlo.broadcast_in_dim %v250, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v252 = stablehlo.divide %v251, %v243 : tensor<32x24x56x56xf32>
    %v253 = stablehlo.add %v252, %v244 : tensor<32x24x56x56xf32>
    %v254 = stablehlo.rsqrt %v253 : tensor<32x24x56x56xf32>
    %v255 = stablehlo.multiply %v248, %v254 : tensor<32x24x56x56xf32>
    %v256 = stablehlo.broadcast_in_dim %gp3, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v257 = stablehlo.broadcast_in_dim %btp3, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v258 = stablehlo.multiply %v255, %v256 : tensor<32x24x56x56xf32>
    %v259 = stablehlo.add %v258, %v257 : tensor<32x24x56x56xf32>
    %v260 = stablehlo.reshape %v259 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v261 = stablehlo.reshape %v260 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v262 = stablehlo.reshape %v173 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v263 = stablehlo.add %v261, %v262 : tensor<32x24x56x56xf32>
    %v264 = stablehlo.reshape %v263 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v265 = stablehlo.reshape %v264 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v266 = stablehlo.convolution(%v265, %We4)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<144x24x1x1xf32>) -> tensor<32x144x56x56xf32>
    %v267 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v268 = stablehlo.add %v266, %v267 : tensor<32x144x56x56xf32>
    %v269 = stablehlo.reshape %v268 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v270 = stablehlo.reshape %v269 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v271 = stablehlo.constant dense<0.0> : tensor<f32>
    %v272 = stablehlo.constant dense<3136.0> : tensor<32x144x56x56xf32>
    %v273 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v274 = stablehlo.reduce(%v270 init: %v271) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v275 = stablehlo.broadcast_in_dim %v274, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v276 = stablehlo.divide %v275, %v272 : tensor<32x144x56x56xf32>
    %v277 = stablehlo.subtract %v270, %v276 : tensor<32x144x56x56xf32>
    %v278 = stablehlo.multiply %v277, %v277 : tensor<32x144x56x56xf32>
    %v279 = stablehlo.reduce(%v278 init: %v271) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v280 = stablehlo.broadcast_in_dim %v279, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v281 = stablehlo.divide %v280, %v272 : tensor<32x144x56x56xf32>
    %v282 = stablehlo.add %v281, %v273 : tensor<32x144x56x56xf32>
    %v283 = stablehlo.rsqrt %v282 : tensor<32x144x56x56xf32>
    %v284 = stablehlo.multiply %v277, %v283 : tensor<32x144x56x56xf32>
    %v285 = stablehlo.broadcast_in_dim %ge4, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v286 = stablehlo.broadcast_in_dim %bte4, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v287 = stablehlo.multiply %v284, %v285 : tensor<32x144x56x56xf32>
    %v288 = stablehlo.add %v287, %v286 : tensor<32x144x56x56xf32>
    %v289 = stablehlo.reshape %v288 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v290 = stablehlo.reshape %v289 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v291 = stablehlo.constant dense<0.0> : tensor<32x144x56x56xf32>
    %v292 = stablehlo.constant dense<6.0> : tensor<32x144x56x56xf32>
    %v293 = stablehlo.maximum %v290, %v291 : tensor<32x144x56x56xf32>
    %v294 = stablehlo.minimum %v293, %v292 : tensor<32x144x56x56xf32>
    %v295 = stablehlo.reshape %v294 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v296 = stablehlo.reshape %v295 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v297 = stablehlo.convolution(%v296, %Wd4)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 144 : i64} : (tensor<32x144x56x56xf32>, tensor<144x1x3x3xf32>) -> tensor<32x144x28x28xf32>
    %v298 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v299 = stablehlo.add %v297, %v298 : tensor<32x144x28x28xf32>
    %v300 = stablehlo.reshape %v299 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v301 = stablehlo.reshape %v300 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v302 = stablehlo.constant dense<0.0> : tensor<f32>
    %v303 = stablehlo.constant dense<784.0> : tensor<32x144x28x28xf32>
    %v304 = stablehlo.constant dense<1.0e-5> : tensor<32x144x28x28xf32>
    %v305 = stablehlo.reduce(%v301 init: %v302) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v306 = stablehlo.broadcast_in_dim %v305, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x28x28xf32>
    %v307 = stablehlo.divide %v306, %v303 : tensor<32x144x28x28xf32>
    %v308 = stablehlo.subtract %v301, %v307 : tensor<32x144x28x28xf32>
    %v309 = stablehlo.multiply %v308, %v308 : tensor<32x144x28x28xf32>
    %v310 = stablehlo.reduce(%v309 init: %v302) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v311 = stablehlo.broadcast_in_dim %v310, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x28x28xf32>
    %v312 = stablehlo.divide %v311, %v303 : tensor<32x144x28x28xf32>
    %v313 = stablehlo.add %v312, %v304 : tensor<32x144x28x28xf32>
    %v314 = stablehlo.rsqrt %v313 : tensor<32x144x28x28xf32>
    %v315 = stablehlo.multiply %v308, %v314 : tensor<32x144x28x28xf32>
    %v316 = stablehlo.broadcast_in_dim %gd4, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v317 = stablehlo.broadcast_in_dim %btd4, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v318 = stablehlo.multiply %v315, %v316 : tensor<32x144x28x28xf32>
    %v319 = stablehlo.add %v318, %v317 : tensor<32x144x28x28xf32>
    %v320 = stablehlo.reshape %v319 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v321 = stablehlo.reshape %v320 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v322 = stablehlo.constant dense<0.0> : tensor<32x144x28x28xf32>
    %v323 = stablehlo.constant dense<6.0> : tensor<32x144x28x28xf32>
    %v324 = stablehlo.maximum %v321, %v322 : tensor<32x144x28x28xf32>
    %v325 = stablehlo.minimum %v324, %v323 : tensor<32x144x28x28xf32>
    %v326 = stablehlo.reshape %v325 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v327 = stablehlo.reshape %v326 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v328 = stablehlo.convolution(%v327, %Wp4)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x144x28x28xf32>, tensor<32x144x1x1xf32>) -> tensor<32x32x28x28xf32>
    %v329 = stablehlo.broadcast_in_dim %zb32, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v330 = stablehlo.add %v328, %v329 : tensor<32x32x28x28xf32>
    %v331 = stablehlo.reshape %v330 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v332 = stablehlo.reshape %v331 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v333 = stablehlo.constant dense<0.0> : tensor<f32>
    %v334 = stablehlo.constant dense<784.0> : tensor<32x32x28x28xf32>
    %v335 = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %v336 = stablehlo.reduce(%v332 init: %v333) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v337 = stablehlo.broadcast_in_dim %v336, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v338 = stablehlo.divide %v337, %v334 : tensor<32x32x28x28xf32>
    %v339 = stablehlo.subtract %v332, %v338 : tensor<32x32x28x28xf32>
    %v340 = stablehlo.multiply %v339, %v339 : tensor<32x32x28x28xf32>
    %v341 = stablehlo.reduce(%v340 init: %v333) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v342 = stablehlo.broadcast_in_dim %v341, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v343 = stablehlo.divide %v342, %v334 : tensor<32x32x28x28xf32>
    %v344 = stablehlo.add %v343, %v335 : tensor<32x32x28x28xf32>
    %v345 = stablehlo.rsqrt %v344 : tensor<32x32x28x28xf32>
    %v346 = stablehlo.multiply %v339, %v345 : tensor<32x32x28x28xf32>
    %v347 = stablehlo.broadcast_in_dim %gp4, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v348 = stablehlo.broadcast_in_dim %btp4, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v349 = stablehlo.multiply %v346, %v347 : tensor<32x32x28x28xf32>
    %v350 = stablehlo.add %v349, %v348 : tensor<32x32x28x28xf32>
    %v351 = stablehlo.reshape %v350 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v352 = stablehlo.reshape %v351 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v353 = stablehlo.convolution(%v352, %We5)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<192x32x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v354 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v355 = stablehlo.add %v353, %v354 : tensor<32x192x28x28xf32>
    %v356 = stablehlo.reshape %v355 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v357 = stablehlo.reshape %v356 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v358 = stablehlo.constant dense<0.0> : tensor<f32>
    %v359 = stablehlo.constant dense<784.0> : tensor<32x192x28x28xf32>
    %v360 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v361 = stablehlo.reduce(%v357 init: %v358) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v362 = stablehlo.broadcast_in_dim %v361, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v363 = stablehlo.divide %v362, %v359 : tensor<32x192x28x28xf32>
    %v364 = stablehlo.subtract %v357, %v363 : tensor<32x192x28x28xf32>
    %v365 = stablehlo.multiply %v364, %v364 : tensor<32x192x28x28xf32>
    %v366 = stablehlo.reduce(%v365 init: %v358) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v367 = stablehlo.broadcast_in_dim %v366, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v368 = stablehlo.divide %v367, %v359 : tensor<32x192x28x28xf32>
    %v369 = stablehlo.add %v368, %v360 : tensor<32x192x28x28xf32>
    %v370 = stablehlo.rsqrt %v369 : tensor<32x192x28x28xf32>
    %v371 = stablehlo.multiply %v364, %v370 : tensor<32x192x28x28xf32>
    %v372 = stablehlo.broadcast_in_dim %ge5, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v373 = stablehlo.broadcast_in_dim %bte5, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v374 = stablehlo.multiply %v371, %v372 : tensor<32x192x28x28xf32>
    %v375 = stablehlo.add %v374, %v373 : tensor<32x192x28x28xf32>
    %v376 = stablehlo.reshape %v375 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v377 = stablehlo.reshape %v376 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v378 = stablehlo.constant dense<0.0> : tensor<32x192x28x28xf32>
    %v379 = stablehlo.constant dense<6.0> : tensor<32x192x28x28xf32>
    %v380 = stablehlo.maximum %v377, %v378 : tensor<32x192x28x28xf32>
    %v381 = stablehlo.minimum %v380, %v379 : tensor<32x192x28x28xf32>
    %v382 = stablehlo.reshape %v381 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v383 = stablehlo.reshape %v382 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v384 = stablehlo.convolution(%v383, %Wd5)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x3x3xf32>) -> tensor<32x192x28x28xf32>
    %v385 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v386 = stablehlo.add %v384, %v385 : tensor<32x192x28x28xf32>
    %v387 = stablehlo.reshape %v386 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v388 = stablehlo.reshape %v387 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v389 = stablehlo.constant dense<0.0> : tensor<f32>
    %v390 = stablehlo.constant dense<784.0> : tensor<32x192x28x28xf32>
    %v391 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v392 = stablehlo.reduce(%v388 init: %v389) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v393 = stablehlo.broadcast_in_dim %v392, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v394 = stablehlo.divide %v393, %v390 : tensor<32x192x28x28xf32>
    %v395 = stablehlo.subtract %v388, %v394 : tensor<32x192x28x28xf32>
    %v396 = stablehlo.multiply %v395, %v395 : tensor<32x192x28x28xf32>
    %v397 = stablehlo.reduce(%v396 init: %v389) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v398 = stablehlo.broadcast_in_dim %v397, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v399 = stablehlo.divide %v398, %v390 : tensor<32x192x28x28xf32>
    %v400 = stablehlo.add %v399, %v391 : tensor<32x192x28x28xf32>
    %v401 = stablehlo.rsqrt %v400 : tensor<32x192x28x28xf32>
    %v402 = stablehlo.multiply %v395, %v401 : tensor<32x192x28x28xf32>
    %v403 = stablehlo.broadcast_in_dim %gd5, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v404 = stablehlo.broadcast_in_dim %btd5, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v405 = stablehlo.multiply %v402, %v403 : tensor<32x192x28x28xf32>
    %v406 = stablehlo.add %v405, %v404 : tensor<32x192x28x28xf32>
    %v407 = stablehlo.reshape %v406 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v408 = stablehlo.reshape %v407 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v409 = stablehlo.constant dense<0.0> : tensor<32x192x28x28xf32>
    %v410 = stablehlo.constant dense<6.0> : tensor<32x192x28x28xf32>
    %v411 = stablehlo.maximum %v408, %v409 : tensor<32x192x28x28xf32>
    %v412 = stablehlo.minimum %v411, %v410 : tensor<32x192x28x28xf32>
    %v413 = stablehlo.reshape %v412 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v414 = stablehlo.reshape %v413 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v415 = stablehlo.convolution(%v414, %Wp5)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<32x192x1x1xf32>) -> tensor<32x32x28x28xf32>
    %v416 = stablehlo.broadcast_in_dim %zb32, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v417 = stablehlo.add %v415, %v416 : tensor<32x32x28x28xf32>
    %v418 = stablehlo.reshape %v417 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v419 = stablehlo.reshape %v418 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v420 = stablehlo.constant dense<0.0> : tensor<f32>
    %v421 = stablehlo.constant dense<784.0> : tensor<32x32x28x28xf32>
    %v422 = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %v423 = stablehlo.reduce(%v419 init: %v420) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v424 = stablehlo.broadcast_in_dim %v423, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v425 = stablehlo.divide %v424, %v421 : tensor<32x32x28x28xf32>
    %v426 = stablehlo.subtract %v419, %v425 : tensor<32x32x28x28xf32>
    %v427 = stablehlo.multiply %v426, %v426 : tensor<32x32x28x28xf32>
    %v428 = stablehlo.reduce(%v427 init: %v420) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v429 = stablehlo.broadcast_in_dim %v428, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v430 = stablehlo.divide %v429, %v421 : tensor<32x32x28x28xf32>
    %v431 = stablehlo.add %v430, %v422 : tensor<32x32x28x28xf32>
    %v432 = stablehlo.rsqrt %v431 : tensor<32x32x28x28xf32>
    %v433 = stablehlo.multiply %v426, %v432 : tensor<32x32x28x28xf32>
    %v434 = stablehlo.broadcast_in_dim %gp5, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v435 = stablehlo.broadcast_in_dim %btp5, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v436 = stablehlo.multiply %v433, %v434 : tensor<32x32x28x28xf32>
    %v437 = stablehlo.add %v436, %v435 : tensor<32x32x28x28xf32>
    %v438 = stablehlo.reshape %v437 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v439 = stablehlo.reshape %v438 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v440 = stablehlo.reshape %v351 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v441 = stablehlo.add %v439, %v440 : tensor<32x32x28x28xf32>
    %v442 = stablehlo.reshape %v441 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v443 = stablehlo.reshape %v442 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v444 = stablehlo.convolution(%v443, %We6)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<192x32x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v445 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v446 = stablehlo.add %v444, %v445 : tensor<32x192x28x28xf32>
    %v447 = stablehlo.reshape %v446 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v448 = stablehlo.reshape %v447 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v449 = stablehlo.constant dense<0.0> : tensor<f32>
    %v450 = stablehlo.constant dense<784.0> : tensor<32x192x28x28xf32>
    %v451 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v452 = stablehlo.reduce(%v448 init: %v449) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v453 = stablehlo.broadcast_in_dim %v452, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v454 = stablehlo.divide %v453, %v450 : tensor<32x192x28x28xf32>
    %v455 = stablehlo.subtract %v448, %v454 : tensor<32x192x28x28xf32>
    %v456 = stablehlo.multiply %v455, %v455 : tensor<32x192x28x28xf32>
    %v457 = stablehlo.reduce(%v456 init: %v449) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v458 = stablehlo.broadcast_in_dim %v457, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v459 = stablehlo.divide %v458, %v450 : tensor<32x192x28x28xf32>
    %v460 = stablehlo.add %v459, %v451 : tensor<32x192x28x28xf32>
    %v461 = stablehlo.rsqrt %v460 : tensor<32x192x28x28xf32>
    %v462 = stablehlo.multiply %v455, %v461 : tensor<32x192x28x28xf32>
    %v463 = stablehlo.broadcast_in_dim %ge6, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v464 = stablehlo.broadcast_in_dim %bte6, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v465 = stablehlo.multiply %v462, %v463 : tensor<32x192x28x28xf32>
    %v466 = stablehlo.add %v465, %v464 : tensor<32x192x28x28xf32>
    %v467 = stablehlo.reshape %v466 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v468 = stablehlo.reshape %v467 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v469 = stablehlo.constant dense<0.0> : tensor<32x192x28x28xf32>
    %v470 = stablehlo.constant dense<6.0> : tensor<32x192x28x28xf32>
    %v471 = stablehlo.maximum %v468, %v469 : tensor<32x192x28x28xf32>
    %v472 = stablehlo.minimum %v471, %v470 : tensor<32x192x28x28xf32>
    %v473 = stablehlo.reshape %v472 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v474 = stablehlo.reshape %v473 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v475 = stablehlo.convolution(%v474, %Wd6)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x3x3xf32>) -> tensor<32x192x28x28xf32>
    %v476 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v477 = stablehlo.add %v475, %v476 : tensor<32x192x28x28xf32>
    %v478 = stablehlo.reshape %v477 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v479 = stablehlo.reshape %v478 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v480 = stablehlo.constant dense<0.0> : tensor<f32>
    %v481 = stablehlo.constant dense<784.0> : tensor<32x192x28x28xf32>
    %v482 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v483 = stablehlo.reduce(%v479 init: %v480) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v484 = stablehlo.broadcast_in_dim %v483, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v485 = stablehlo.divide %v484, %v481 : tensor<32x192x28x28xf32>
    %v486 = stablehlo.subtract %v479, %v485 : tensor<32x192x28x28xf32>
    %v487 = stablehlo.multiply %v486, %v486 : tensor<32x192x28x28xf32>
    %v488 = stablehlo.reduce(%v487 init: %v480) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v489 = stablehlo.broadcast_in_dim %v488, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v490 = stablehlo.divide %v489, %v481 : tensor<32x192x28x28xf32>
    %v491 = stablehlo.add %v490, %v482 : tensor<32x192x28x28xf32>
    %v492 = stablehlo.rsqrt %v491 : tensor<32x192x28x28xf32>
    %v493 = stablehlo.multiply %v486, %v492 : tensor<32x192x28x28xf32>
    %v494 = stablehlo.broadcast_in_dim %gd6, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v495 = stablehlo.broadcast_in_dim %btd6, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v496 = stablehlo.multiply %v493, %v494 : tensor<32x192x28x28xf32>
    %v497 = stablehlo.add %v496, %v495 : tensor<32x192x28x28xf32>
    %v498 = stablehlo.reshape %v497 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v499 = stablehlo.reshape %v498 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v500 = stablehlo.constant dense<0.0> : tensor<32x192x28x28xf32>
    %v501 = stablehlo.constant dense<6.0> : tensor<32x192x28x28xf32>
    %v502 = stablehlo.maximum %v499, %v500 : tensor<32x192x28x28xf32>
    %v503 = stablehlo.minimum %v502, %v501 : tensor<32x192x28x28xf32>
    %v504 = stablehlo.reshape %v503 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v505 = stablehlo.reshape %v504 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v506 = stablehlo.convolution(%v505, %Wp6)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<32x192x1x1xf32>) -> tensor<32x32x28x28xf32>
    %v507 = stablehlo.broadcast_in_dim %zb32, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v508 = stablehlo.add %v506, %v507 : tensor<32x32x28x28xf32>
    %v509 = stablehlo.reshape %v508 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v510 = stablehlo.reshape %v509 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v511 = stablehlo.constant dense<0.0> : tensor<f32>
    %v512 = stablehlo.constant dense<784.0> : tensor<32x32x28x28xf32>
    %v513 = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %v514 = stablehlo.reduce(%v510 init: %v511) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v515 = stablehlo.broadcast_in_dim %v514, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v516 = stablehlo.divide %v515, %v512 : tensor<32x32x28x28xf32>
    %v517 = stablehlo.subtract %v510, %v516 : tensor<32x32x28x28xf32>
    %v518 = stablehlo.multiply %v517, %v517 : tensor<32x32x28x28xf32>
    %v519 = stablehlo.reduce(%v518 init: %v511) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v520 = stablehlo.broadcast_in_dim %v519, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v521 = stablehlo.divide %v520, %v512 : tensor<32x32x28x28xf32>
    %v522 = stablehlo.add %v521, %v513 : tensor<32x32x28x28xf32>
    %v523 = stablehlo.rsqrt %v522 : tensor<32x32x28x28xf32>
    %v524 = stablehlo.multiply %v517, %v523 : tensor<32x32x28x28xf32>
    %v525 = stablehlo.broadcast_in_dim %gp6, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v526 = stablehlo.broadcast_in_dim %btp6, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v527 = stablehlo.multiply %v524, %v525 : tensor<32x32x28x28xf32>
    %v528 = stablehlo.add %v527, %v526 : tensor<32x32x28x28xf32>
    %v529 = stablehlo.reshape %v528 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v530 = stablehlo.reshape %v529 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v531 = stablehlo.reshape %v442 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v532 = stablehlo.add %v530, %v531 : tensor<32x32x28x28xf32>
    %v533 = stablehlo.reshape %v532 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v534 = stablehlo.reshape %v533 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v535 = stablehlo.convolution(%v534, %We7)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<192x32x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v536 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v537 = stablehlo.add %v535, %v536 : tensor<32x192x28x28xf32>
    %v538 = stablehlo.reshape %v537 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v539 = stablehlo.reshape %v538 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v540 = stablehlo.constant dense<0.0> : tensor<f32>
    %v541 = stablehlo.constant dense<784.0> : tensor<32x192x28x28xf32>
    %v542 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v543 = stablehlo.reduce(%v539 init: %v540) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v544 = stablehlo.broadcast_in_dim %v543, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v545 = stablehlo.divide %v544, %v541 : tensor<32x192x28x28xf32>
    %v546 = stablehlo.subtract %v539, %v545 : tensor<32x192x28x28xf32>
    %v547 = stablehlo.multiply %v546, %v546 : tensor<32x192x28x28xf32>
    %v548 = stablehlo.reduce(%v547 init: %v540) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v549 = stablehlo.broadcast_in_dim %v548, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v550 = stablehlo.divide %v549, %v541 : tensor<32x192x28x28xf32>
    %v551 = stablehlo.add %v550, %v542 : tensor<32x192x28x28xf32>
    %v552 = stablehlo.rsqrt %v551 : tensor<32x192x28x28xf32>
    %v553 = stablehlo.multiply %v546, %v552 : tensor<32x192x28x28xf32>
    %v554 = stablehlo.broadcast_in_dim %ge7, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v555 = stablehlo.broadcast_in_dim %bte7, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v556 = stablehlo.multiply %v553, %v554 : tensor<32x192x28x28xf32>
    %v557 = stablehlo.add %v556, %v555 : tensor<32x192x28x28xf32>
    %v558 = stablehlo.reshape %v557 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v559 = stablehlo.reshape %v558 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v560 = stablehlo.constant dense<0.0> : tensor<32x192x28x28xf32>
    %v561 = stablehlo.constant dense<6.0> : tensor<32x192x28x28xf32>
    %v562 = stablehlo.maximum %v559, %v560 : tensor<32x192x28x28xf32>
    %v563 = stablehlo.minimum %v562, %v561 : tensor<32x192x28x28xf32>
    %v564 = stablehlo.reshape %v563 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v565 = stablehlo.reshape %v564 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v566 = stablehlo.convolution(%v565, %Wd7)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x3x3xf32>) -> tensor<32x192x14x14xf32>
    %v567 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x14x14xf32>
    %v568 = stablehlo.add %v566, %v567 : tensor<32x192x14x14xf32>
    %v569 = stablehlo.reshape %v568 : (tensor<32x192x14x14xf32>) -> tensor<32x37632xf32>
    %v570 = stablehlo.reshape %v569 : (tensor<32x37632xf32>) -> tensor<32x192x14x14xf32>
    %v571 = stablehlo.constant dense<0.0> : tensor<f32>
    %v572 = stablehlo.constant dense<196.0> : tensor<32x192x14x14xf32>
    %v573 = stablehlo.constant dense<1.0e-5> : tensor<32x192x14x14xf32>
    %v574 = stablehlo.reduce(%v570 init: %v571) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v575 = stablehlo.broadcast_in_dim %v574, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x14x14xf32>
    %v576 = stablehlo.divide %v575, %v572 : tensor<32x192x14x14xf32>
    %v577 = stablehlo.subtract %v570, %v576 : tensor<32x192x14x14xf32>
    %v578 = stablehlo.multiply %v577, %v577 : tensor<32x192x14x14xf32>
    %v579 = stablehlo.reduce(%v578 init: %v571) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v580 = stablehlo.broadcast_in_dim %v579, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x14x14xf32>
    %v581 = stablehlo.divide %v580, %v572 : tensor<32x192x14x14xf32>
    %v582 = stablehlo.add %v581, %v573 : tensor<32x192x14x14xf32>
    %v583 = stablehlo.rsqrt %v582 : tensor<32x192x14x14xf32>
    %v584 = stablehlo.multiply %v577, %v583 : tensor<32x192x14x14xf32>
    %v585 = stablehlo.broadcast_in_dim %gd7, dims = [1] : (tensor<192xf32>) -> tensor<32x192x14x14xf32>
    %v586 = stablehlo.broadcast_in_dim %btd7, dims = [1] : (tensor<192xf32>) -> tensor<32x192x14x14xf32>
    %v587 = stablehlo.multiply %v584, %v585 : tensor<32x192x14x14xf32>
    %v588 = stablehlo.add %v587, %v586 : tensor<32x192x14x14xf32>
    %v589 = stablehlo.reshape %v588 : (tensor<32x192x14x14xf32>) -> tensor<32x37632xf32>
    %v590 = stablehlo.reshape %v589 : (tensor<32x37632xf32>) -> tensor<32x192x14x14xf32>
    %v591 = stablehlo.constant dense<0.0> : tensor<32x192x14x14xf32>
    %v592 = stablehlo.constant dense<6.0> : tensor<32x192x14x14xf32>
    %v593 = stablehlo.maximum %v590, %v591 : tensor<32x192x14x14xf32>
    %v594 = stablehlo.minimum %v593, %v592 : tensor<32x192x14x14xf32>
    %v595 = stablehlo.reshape %v594 : (tensor<32x192x14x14xf32>) -> tensor<32x37632xf32>
    %v596 = stablehlo.reshape %v595 : (tensor<32x37632xf32>) -> tensor<32x192x14x14xf32>
    %v597 = stablehlo.convolution(%v596, %Wp7)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x14x14xf32>, tensor<64x192x1x1xf32>) -> tensor<32x64x14x14xf32>
    %v598 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v599 = stablehlo.add %v597, %v598 : tensor<32x64x14x14xf32>
    %v600 = stablehlo.reshape %v599 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v601 = stablehlo.reshape %v600 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v602 = stablehlo.constant dense<0.0> : tensor<f32>
    %v603 = stablehlo.constant dense<196.0> : tensor<32x64x14x14xf32>
    %v604 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v605 = stablehlo.reduce(%v601 init: %v602) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v606 = stablehlo.broadcast_in_dim %v605, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v607 = stablehlo.divide %v606, %v603 : tensor<32x64x14x14xf32>
    %v608 = stablehlo.subtract %v601, %v607 : tensor<32x64x14x14xf32>
    %v609 = stablehlo.multiply %v608, %v608 : tensor<32x64x14x14xf32>
    %v610 = stablehlo.reduce(%v609 init: %v602) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v611 = stablehlo.broadcast_in_dim %v610, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v612 = stablehlo.divide %v611, %v603 : tensor<32x64x14x14xf32>
    %v613 = stablehlo.add %v612, %v604 : tensor<32x64x14x14xf32>
    %v614 = stablehlo.rsqrt %v613 : tensor<32x64x14x14xf32>
    %v615 = stablehlo.multiply %v608, %v614 : tensor<32x64x14x14xf32>
    %v616 = stablehlo.broadcast_in_dim %gp7, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v617 = stablehlo.broadcast_in_dim %btp7, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v618 = stablehlo.multiply %v615, %v616 : tensor<32x64x14x14xf32>
    %v619 = stablehlo.add %v618, %v617 : tensor<32x64x14x14xf32>
    %v620 = stablehlo.reshape %v619 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v621 = stablehlo.reshape %v620 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v622 = stablehlo.convolution(%v621, %We8)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x14x14xf32>, tensor<384x64x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v623 = stablehlo.broadcast_in_dim %zb384, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v624 = stablehlo.add %v622, %v623 : tensor<32x384x14x14xf32>
    %v625 = stablehlo.reshape %v624 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v626 = stablehlo.reshape %v625 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v627 = stablehlo.constant dense<0.0> : tensor<f32>
    %v628 = stablehlo.constant dense<196.0> : tensor<32x384x14x14xf32>
    %v629 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v630 = stablehlo.reduce(%v626 init: %v627) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v631 = stablehlo.broadcast_in_dim %v630, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v632 = stablehlo.divide %v631, %v628 : tensor<32x384x14x14xf32>
    %v633 = stablehlo.subtract %v626, %v632 : tensor<32x384x14x14xf32>
    %v634 = stablehlo.multiply %v633, %v633 : tensor<32x384x14x14xf32>
    %v635 = stablehlo.reduce(%v634 init: %v627) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v636 = stablehlo.broadcast_in_dim %v635, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v637 = stablehlo.divide %v636, %v628 : tensor<32x384x14x14xf32>
    %v638 = stablehlo.add %v637, %v629 : tensor<32x384x14x14xf32>
    %v639 = stablehlo.rsqrt %v638 : tensor<32x384x14x14xf32>
    %v640 = stablehlo.multiply %v633, %v639 : tensor<32x384x14x14xf32>
    %v641 = stablehlo.broadcast_in_dim %ge8, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v642 = stablehlo.broadcast_in_dim %bte8, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v643 = stablehlo.multiply %v640, %v641 : tensor<32x384x14x14xf32>
    %v644 = stablehlo.add %v643, %v642 : tensor<32x384x14x14xf32>
    %v645 = stablehlo.reshape %v644 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v646 = stablehlo.reshape %v645 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v647 = stablehlo.constant dense<0.0> : tensor<32x384x14x14xf32>
    %v648 = stablehlo.constant dense<6.0> : tensor<32x384x14x14xf32>
    %v649 = stablehlo.maximum %v646, %v647 : tensor<32x384x14x14xf32>
    %v650 = stablehlo.minimum %v649, %v648 : tensor<32x384x14x14xf32>
    %v651 = stablehlo.reshape %v650 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v652 = stablehlo.reshape %v651 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v653 = stablehlo.convolution(%v652, %Wd8)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x3x3xf32>) -> tensor<32x384x14x14xf32>
    %v654 = stablehlo.broadcast_in_dim %zb384, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v655 = stablehlo.add %v653, %v654 : tensor<32x384x14x14xf32>
    %v656 = stablehlo.reshape %v655 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v657 = stablehlo.reshape %v656 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v658 = stablehlo.constant dense<0.0> : tensor<f32>
    %v659 = stablehlo.constant dense<196.0> : tensor<32x384x14x14xf32>
    %v660 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v661 = stablehlo.reduce(%v657 init: %v658) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v662 = stablehlo.broadcast_in_dim %v661, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v663 = stablehlo.divide %v662, %v659 : tensor<32x384x14x14xf32>
    %v664 = stablehlo.subtract %v657, %v663 : tensor<32x384x14x14xf32>
    %v665 = stablehlo.multiply %v664, %v664 : tensor<32x384x14x14xf32>
    %v666 = stablehlo.reduce(%v665 init: %v658) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v667 = stablehlo.broadcast_in_dim %v666, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v668 = stablehlo.divide %v667, %v659 : tensor<32x384x14x14xf32>
    %v669 = stablehlo.add %v668, %v660 : tensor<32x384x14x14xf32>
    %v670 = stablehlo.rsqrt %v669 : tensor<32x384x14x14xf32>
    %v671 = stablehlo.multiply %v664, %v670 : tensor<32x384x14x14xf32>
    %v672 = stablehlo.broadcast_in_dim %gd8, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v673 = stablehlo.broadcast_in_dim %btd8, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v674 = stablehlo.multiply %v671, %v672 : tensor<32x384x14x14xf32>
    %v675 = stablehlo.add %v674, %v673 : tensor<32x384x14x14xf32>
    %v676 = stablehlo.reshape %v675 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v677 = stablehlo.reshape %v676 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v678 = stablehlo.constant dense<0.0> : tensor<32x384x14x14xf32>
    %v679 = stablehlo.constant dense<6.0> : tensor<32x384x14x14xf32>
    %v680 = stablehlo.maximum %v677, %v678 : tensor<32x384x14x14xf32>
    %v681 = stablehlo.minimum %v680, %v679 : tensor<32x384x14x14xf32>
    %v682 = stablehlo.reshape %v681 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v683 = stablehlo.reshape %v682 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v684 = stablehlo.convolution(%v683, %Wp8)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<64x384x1x1xf32>) -> tensor<32x64x14x14xf32>
    %v685 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v686 = stablehlo.add %v684, %v685 : tensor<32x64x14x14xf32>
    %v687 = stablehlo.reshape %v686 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v688 = stablehlo.reshape %v687 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v689 = stablehlo.constant dense<0.0> : tensor<f32>
    %v690 = stablehlo.constant dense<196.0> : tensor<32x64x14x14xf32>
    %v691 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v692 = stablehlo.reduce(%v688 init: %v689) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v693 = stablehlo.broadcast_in_dim %v692, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v694 = stablehlo.divide %v693, %v690 : tensor<32x64x14x14xf32>
    %v695 = stablehlo.subtract %v688, %v694 : tensor<32x64x14x14xf32>
    %v696 = stablehlo.multiply %v695, %v695 : tensor<32x64x14x14xf32>
    %v697 = stablehlo.reduce(%v696 init: %v689) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v698 = stablehlo.broadcast_in_dim %v697, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v699 = stablehlo.divide %v698, %v690 : tensor<32x64x14x14xf32>
    %v700 = stablehlo.add %v699, %v691 : tensor<32x64x14x14xf32>
    %v701 = stablehlo.rsqrt %v700 : tensor<32x64x14x14xf32>
    %v702 = stablehlo.multiply %v695, %v701 : tensor<32x64x14x14xf32>
    %v703 = stablehlo.broadcast_in_dim %gp8, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v704 = stablehlo.broadcast_in_dim %btp8, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v705 = stablehlo.multiply %v702, %v703 : tensor<32x64x14x14xf32>
    %v706 = stablehlo.add %v705, %v704 : tensor<32x64x14x14xf32>
    %v707 = stablehlo.reshape %v706 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v708 = stablehlo.reshape %v707 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v709 = stablehlo.reshape %v620 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v710 = stablehlo.add %v708, %v709 : tensor<32x64x14x14xf32>
    %v711 = stablehlo.reshape %v710 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v712 = stablehlo.reshape %v711 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v713 = stablehlo.convolution(%v712, %We9)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x14x14xf32>, tensor<384x64x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v714 = stablehlo.broadcast_in_dim %zb384, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v715 = stablehlo.add %v713, %v714 : tensor<32x384x14x14xf32>
    %v716 = stablehlo.reshape %v715 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v717 = stablehlo.reshape %v716 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v718 = stablehlo.constant dense<0.0> : tensor<f32>
    %v719 = stablehlo.constant dense<196.0> : tensor<32x384x14x14xf32>
    %v720 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v721 = stablehlo.reduce(%v717 init: %v718) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v722 = stablehlo.broadcast_in_dim %v721, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v723 = stablehlo.divide %v722, %v719 : tensor<32x384x14x14xf32>
    %v724 = stablehlo.subtract %v717, %v723 : tensor<32x384x14x14xf32>
    %v725 = stablehlo.multiply %v724, %v724 : tensor<32x384x14x14xf32>
    %v726 = stablehlo.reduce(%v725 init: %v718) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v727 = stablehlo.broadcast_in_dim %v726, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v728 = stablehlo.divide %v727, %v719 : tensor<32x384x14x14xf32>
    %v729 = stablehlo.add %v728, %v720 : tensor<32x384x14x14xf32>
    %v730 = stablehlo.rsqrt %v729 : tensor<32x384x14x14xf32>
    %v731 = stablehlo.multiply %v724, %v730 : tensor<32x384x14x14xf32>
    %v732 = stablehlo.broadcast_in_dim %ge9, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v733 = stablehlo.broadcast_in_dim %bte9, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v734 = stablehlo.multiply %v731, %v732 : tensor<32x384x14x14xf32>
    %v735 = stablehlo.add %v734, %v733 : tensor<32x384x14x14xf32>
    %v736 = stablehlo.reshape %v735 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v737 = stablehlo.reshape %v736 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v738 = stablehlo.constant dense<0.0> : tensor<32x384x14x14xf32>
    %v739 = stablehlo.constant dense<6.0> : tensor<32x384x14x14xf32>
    %v740 = stablehlo.maximum %v737, %v738 : tensor<32x384x14x14xf32>
    %v741 = stablehlo.minimum %v740, %v739 : tensor<32x384x14x14xf32>
    %v742 = stablehlo.reshape %v741 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v743 = stablehlo.reshape %v742 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v744 = stablehlo.convolution(%v743, %Wd9)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x3x3xf32>) -> tensor<32x384x14x14xf32>
    %v745 = stablehlo.broadcast_in_dim %zb384, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v746 = stablehlo.add %v744, %v745 : tensor<32x384x14x14xf32>
    %v747 = stablehlo.reshape %v746 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v748 = stablehlo.reshape %v747 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v749 = stablehlo.constant dense<0.0> : tensor<f32>
    %v750 = stablehlo.constant dense<196.0> : tensor<32x384x14x14xf32>
    %v751 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v752 = stablehlo.reduce(%v748 init: %v749) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v753 = stablehlo.broadcast_in_dim %v752, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v754 = stablehlo.divide %v753, %v750 : tensor<32x384x14x14xf32>
    %v755 = stablehlo.subtract %v748, %v754 : tensor<32x384x14x14xf32>
    %v756 = stablehlo.multiply %v755, %v755 : tensor<32x384x14x14xf32>
    %v757 = stablehlo.reduce(%v756 init: %v749) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v758 = stablehlo.broadcast_in_dim %v757, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v759 = stablehlo.divide %v758, %v750 : tensor<32x384x14x14xf32>
    %v760 = stablehlo.add %v759, %v751 : tensor<32x384x14x14xf32>
    %v761 = stablehlo.rsqrt %v760 : tensor<32x384x14x14xf32>
    %v762 = stablehlo.multiply %v755, %v761 : tensor<32x384x14x14xf32>
    %v763 = stablehlo.broadcast_in_dim %gd9, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v764 = stablehlo.broadcast_in_dim %btd9, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v765 = stablehlo.multiply %v762, %v763 : tensor<32x384x14x14xf32>
    %v766 = stablehlo.add %v765, %v764 : tensor<32x384x14x14xf32>
    %v767 = stablehlo.reshape %v766 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v768 = stablehlo.reshape %v767 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v769 = stablehlo.constant dense<0.0> : tensor<32x384x14x14xf32>
    %v770 = stablehlo.constant dense<6.0> : tensor<32x384x14x14xf32>
    %v771 = stablehlo.maximum %v768, %v769 : tensor<32x384x14x14xf32>
    %v772 = stablehlo.minimum %v771, %v770 : tensor<32x384x14x14xf32>
    %v773 = stablehlo.reshape %v772 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v774 = stablehlo.reshape %v773 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v775 = stablehlo.convolution(%v774, %Wp9)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<64x384x1x1xf32>) -> tensor<32x64x14x14xf32>
    %v776 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v777 = stablehlo.add %v775, %v776 : tensor<32x64x14x14xf32>
    %v778 = stablehlo.reshape %v777 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v779 = stablehlo.reshape %v778 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v780 = stablehlo.constant dense<0.0> : tensor<f32>
    %v781 = stablehlo.constant dense<196.0> : tensor<32x64x14x14xf32>
    %v782 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v783 = stablehlo.reduce(%v779 init: %v780) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v784 = stablehlo.broadcast_in_dim %v783, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v785 = stablehlo.divide %v784, %v781 : tensor<32x64x14x14xf32>
    %v786 = stablehlo.subtract %v779, %v785 : tensor<32x64x14x14xf32>
    %v787 = stablehlo.multiply %v786, %v786 : tensor<32x64x14x14xf32>
    %v788 = stablehlo.reduce(%v787 init: %v780) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v789 = stablehlo.broadcast_in_dim %v788, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v790 = stablehlo.divide %v789, %v781 : tensor<32x64x14x14xf32>
    %v791 = stablehlo.add %v790, %v782 : tensor<32x64x14x14xf32>
    %v792 = stablehlo.rsqrt %v791 : tensor<32x64x14x14xf32>
    %v793 = stablehlo.multiply %v786, %v792 : tensor<32x64x14x14xf32>
    %v794 = stablehlo.broadcast_in_dim %gp9, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v795 = stablehlo.broadcast_in_dim %btp9, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v796 = stablehlo.multiply %v793, %v794 : tensor<32x64x14x14xf32>
    %v797 = stablehlo.add %v796, %v795 : tensor<32x64x14x14xf32>
    %v798 = stablehlo.reshape %v797 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v799 = stablehlo.reshape %v798 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v800 = stablehlo.reshape %v711 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v801 = stablehlo.add %v799, %v800 : tensor<32x64x14x14xf32>
    %v802 = stablehlo.reshape %v801 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v803 = stablehlo.reshape %v802 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v804 = stablehlo.convolution(%v803, %We10)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x14x14xf32>, tensor<384x64x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v805 = stablehlo.broadcast_in_dim %zb384, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v806 = stablehlo.add %v804, %v805 : tensor<32x384x14x14xf32>
    %v807 = stablehlo.reshape %v806 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v808 = stablehlo.reshape %v807 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v809 = stablehlo.constant dense<0.0> : tensor<f32>
    %v810 = stablehlo.constant dense<196.0> : tensor<32x384x14x14xf32>
    %v811 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v812 = stablehlo.reduce(%v808 init: %v809) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v813 = stablehlo.broadcast_in_dim %v812, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v814 = stablehlo.divide %v813, %v810 : tensor<32x384x14x14xf32>
    %v815 = stablehlo.subtract %v808, %v814 : tensor<32x384x14x14xf32>
    %v816 = stablehlo.multiply %v815, %v815 : tensor<32x384x14x14xf32>
    %v817 = stablehlo.reduce(%v816 init: %v809) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v818 = stablehlo.broadcast_in_dim %v817, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v819 = stablehlo.divide %v818, %v810 : tensor<32x384x14x14xf32>
    %v820 = stablehlo.add %v819, %v811 : tensor<32x384x14x14xf32>
    %v821 = stablehlo.rsqrt %v820 : tensor<32x384x14x14xf32>
    %v822 = stablehlo.multiply %v815, %v821 : tensor<32x384x14x14xf32>
    %v823 = stablehlo.broadcast_in_dim %ge10, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v824 = stablehlo.broadcast_in_dim %bte10, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v825 = stablehlo.multiply %v822, %v823 : tensor<32x384x14x14xf32>
    %v826 = stablehlo.add %v825, %v824 : tensor<32x384x14x14xf32>
    %v827 = stablehlo.reshape %v826 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v828 = stablehlo.reshape %v827 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v829 = stablehlo.constant dense<0.0> : tensor<32x384x14x14xf32>
    %v830 = stablehlo.constant dense<6.0> : tensor<32x384x14x14xf32>
    %v831 = stablehlo.maximum %v828, %v829 : tensor<32x384x14x14xf32>
    %v832 = stablehlo.minimum %v831, %v830 : tensor<32x384x14x14xf32>
    %v833 = stablehlo.reshape %v832 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v834 = stablehlo.reshape %v833 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v835 = stablehlo.convolution(%v834, %Wd10)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x3x3xf32>) -> tensor<32x384x14x14xf32>
    %v836 = stablehlo.broadcast_in_dim %zb384, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v837 = stablehlo.add %v835, %v836 : tensor<32x384x14x14xf32>
    %v838 = stablehlo.reshape %v837 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v839 = stablehlo.reshape %v838 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v840 = stablehlo.constant dense<0.0> : tensor<f32>
    %v841 = stablehlo.constant dense<196.0> : tensor<32x384x14x14xf32>
    %v842 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v843 = stablehlo.reduce(%v839 init: %v840) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v844 = stablehlo.broadcast_in_dim %v843, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v845 = stablehlo.divide %v844, %v841 : tensor<32x384x14x14xf32>
    %v846 = stablehlo.subtract %v839, %v845 : tensor<32x384x14x14xf32>
    %v847 = stablehlo.multiply %v846, %v846 : tensor<32x384x14x14xf32>
    %v848 = stablehlo.reduce(%v847 init: %v840) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v849 = stablehlo.broadcast_in_dim %v848, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v850 = stablehlo.divide %v849, %v841 : tensor<32x384x14x14xf32>
    %v851 = stablehlo.add %v850, %v842 : tensor<32x384x14x14xf32>
    %v852 = stablehlo.rsqrt %v851 : tensor<32x384x14x14xf32>
    %v853 = stablehlo.multiply %v846, %v852 : tensor<32x384x14x14xf32>
    %v854 = stablehlo.broadcast_in_dim %gd10, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v855 = stablehlo.broadcast_in_dim %btd10, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v856 = stablehlo.multiply %v853, %v854 : tensor<32x384x14x14xf32>
    %v857 = stablehlo.add %v856, %v855 : tensor<32x384x14x14xf32>
    %v858 = stablehlo.reshape %v857 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v859 = stablehlo.reshape %v858 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v860 = stablehlo.constant dense<0.0> : tensor<32x384x14x14xf32>
    %v861 = stablehlo.constant dense<6.0> : tensor<32x384x14x14xf32>
    %v862 = stablehlo.maximum %v859, %v860 : tensor<32x384x14x14xf32>
    %v863 = stablehlo.minimum %v862, %v861 : tensor<32x384x14x14xf32>
    %v864 = stablehlo.reshape %v863 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v865 = stablehlo.reshape %v864 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v866 = stablehlo.convolution(%v865, %Wp10)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<64x384x1x1xf32>) -> tensor<32x64x14x14xf32>
    %v867 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v868 = stablehlo.add %v866, %v867 : tensor<32x64x14x14xf32>
    %v869 = stablehlo.reshape %v868 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v870 = stablehlo.reshape %v869 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v871 = stablehlo.constant dense<0.0> : tensor<f32>
    %v872 = stablehlo.constant dense<196.0> : tensor<32x64x14x14xf32>
    %v873 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v874 = stablehlo.reduce(%v870 init: %v871) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v875 = stablehlo.broadcast_in_dim %v874, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v876 = stablehlo.divide %v875, %v872 : tensor<32x64x14x14xf32>
    %v877 = stablehlo.subtract %v870, %v876 : tensor<32x64x14x14xf32>
    %v878 = stablehlo.multiply %v877, %v877 : tensor<32x64x14x14xf32>
    %v879 = stablehlo.reduce(%v878 init: %v871) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v880 = stablehlo.broadcast_in_dim %v879, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v881 = stablehlo.divide %v880, %v872 : tensor<32x64x14x14xf32>
    %v882 = stablehlo.add %v881, %v873 : tensor<32x64x14x14xf32>
    %v883 = stablehlo.rsqrt %v882 : tensor<32x64x14x14xf32>
    %v884 = stablehlo.multiply %v877, %v883 : tensor<32x64x14x14xf32>
    %v885 = stablehlo.broadcast_in_dim %gp10, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v886 = stablehlo.broadcast_in_dim %btp10, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v887 = stablehlo.multiply %v884, %v885 : tensor<32x64x14x14xf32>
    %v888 = stablehlo.add %v887, %v886 : tensor<32x64x14x14xf32>
    %v889 = stablehlo.reshape %v888 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v890 = stablehlo.reshape %v889 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v891 = stablehlo.reshape %v802 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v892 = stablehlo.add %v890, %v891 : tensor<32x64x14x14xf32>
    %v893 = stablehlo.reshape %v892 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v894 = stablehlo.reshape %v893 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v895 = stablehlo.convolution(%v894, %We11)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x14x14xf32>, tensor<384x64x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v896 = stablehlo.broadcast_in_dim %zb384, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v897 = stablehlo.add %v895, %v896 : tensor<32x384x14x14xf32>
    %v898 = stablehlo.reshape %v897 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v899 = stablehlo.reshape %v898 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v900 = stablehlo.constant dense<0.0> : tensor<f32>
    %v901 = stablehlo.constant dense<196.0> : tensor<32x384x14x14xf32>
    %v902 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v903 = stablehlo.reduce(%v899 init: %v900) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v904 = stablehlo.broadcast_in_dim %v903, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v905 = stablehlo.divide %v904, %v901 : tensor<32x384x14x14xf32>
    %v906 = stablehlo.subtract %v899, %v905 : tensor<32x384x14x14xf32>
    %v907 = stablehlo.multiply %v906, %v906 : tensor<32x384x14x14xf32>
    %v908 = stablehlo.reduce(%v907 init: %v900) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v909 = stablehlo.broadcast_in_dim %v908, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v910 = stablehlo.divide %v909, %v901 : tensor<32x384x14x14xf32>
    %v911 = stablehlo.add %v910, %v902 : tensor<32x384x14x14xf32>
    %v912 = stablehlo.rsqrt %v911 : tensor<32x384x14x14xf32>
    %v913 = stablehlo.multiply %v906, %v912 : tensor<32x384x14x14xf32>
    %v914 = stablehlo.broadcast_in_dim %ge11, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v915 = stablehlo.broadcast_in_dim %bte11, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v916 = stablehlo.multiply %v913, %v914 : tensor<32x384x14x14xf32>
    %v917 = stablehlo.add %v916, %v915 : tensor<32x384x14x14xf32>
    %v918 = stablehlo.reshape %v917 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v919 = stablehlo.reshape %v918 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v920 = stablehlo.constant dense<0.0> : tensor<32x384x14x14xf32>
    %v921 = stablehlo.constant dense<6.0> : tensor<32x384x14x14xf32>
    %v922 = stablehlo.maximum %v919, %v920 : tensor<32x384x14x14xf32>
    %v923 = stablehlo.minimum %v922, %v921 : tensor<32x384x14x14xf32>
    %v924 = stablehlo.reshape %v923 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v925 = stablehlo.reshape %v924 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v926 = stablehlo.convolution(%v925, %Wd11)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x3x3xf32>) -> tensor<32x384x14x14xf32>
    %v927 = stablehlo.broadcast_in_dim %zb384, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v928 = stablehlo.add %v926, %v927 : tensor<32x384x14x14xf32>
    %v929 = stablehlo.reshape %v928 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v930 = stablehlo.reshape %v929 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v931 = stablehlo.constant dense<0.0> : tensor<f32>
    %v932 = stablehlo.constant dense<196.0> : tensor<32x384x14x14xf32>
    %v933 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v934 = stablehlo.reduce(%v930 init: %v931) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v935 = stablehlo.broadcast_in_dim %v934, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v936 = stablehlo.divide %v935, %v932 : tensor<32x384x14x14xf32>
    %v937 = stablehlo.subtract %v930, %v936 : tensor<32x384x14x14xf32>
    %v938 = stablehlo.multiply %v937, %v937 : tensor<32x384x14x14xf32>
    %v939 = stablehlo.reduce(%v938 init: %v931) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v940 = stablehlo.broadcast_in_dim %v939, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v941 = stablehlo.divide %v940, %v932 : tensor<32x384x14x14xf32>
    %v942 = stablehlo.add %v941, %v933 : tensor<32x384x14x14xf32>
    %v943 = stablehlo.rsqrt %v942 : tensor<32x384x14x14xf32>
    %v944 = stablehlo.multiply %v937, %v943 : tensor<32x384x14x14xf32>
    %v945 = stablehlo.broadcast_in_dim %gd11, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v946 = stablehlo.broadcast_in_dim %btd11, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v947 = stablehlo.multiply %v944, %v945 : tensor<32x384x14x14xf32>
    %v948 = stablehlo.add %v947, %v946 : tensor<32x384x14x14xf32>
    %v949 = stablehlo.reshape %v948 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v950 = stablehlo.reshape %v949 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v951 = stablehlo.constant dense<0.0> : tensor<32x384x14x14xf32>
    %v952 = stablehlo.constant dense<6.0> : tensor<32x384x14x14xf32>
    %v953 = stablehlo.maximum %v950, %v951 : tensor<32x384x14x14xf32>
    %v954 = stablehlo.minimum %v953, %v952 : tensor<32x384x14x14xf32>
    %v955 = stablehlo.reshape %v954 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v956 = stablehlo.reshape %v955 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v957 = stablehlo.convolution(%v956, %Wp11)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<96x384x1x1xf32>) -> tensor<32x96x14x14xf32>
    %v958 = stablehlo.broadcast_in_dim %zb96, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v959 = stablehlo.add %v957, %v958 : tensor<32x96x14x14xf32>
    %v960 = stablehlo.reshape %v959 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v961 = stablehlo.reshape %v960 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v962 = stablehlo.constant dense<0.0> : tensor<f32>
    %v963 = stablehlo.constant dense<196.0> : tensor<32x96x14x14xf32>
    %v964 = stablehlo.constant dense<1.0e-5> : tensor<32x96x14x14xf32>
    %v965 = stablehlo.reduce(%v961 init: %v962) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v966 = stablehlo.broadcast_in_dim %v965, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x14x14xf32>
    %v967 = stablehlo.divide %v966, %v963 : tensor<32x96x14x14xf32>
    %v968 = stablehlo.subtract %v961, %v967 : tensor<32x96x14x14xf32>
    %v969 = stablehlo.multiply %v968, %v968 : tensor<32x96x14x14xf32>
    %v970 = stablehlo.reduce(%v969 init: %v962) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v971 = stablehlo.broadcast_in_dim %v970, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x14x14xf32>
    %v972 = stablehlo.divide %v971, %v963 : tensor<32x96x14x14xf32>
    %v973 = stablehlo.add %v972, %v964 : tensor<32x96x14x14xf32>
    %v974 = stablehlo.rsqrt %v973 : tensor<32x96x14x14xf32>
    %v975 = stablehlo.multiply %v968, %v974 : tensor<32x96x14x14xf32>
    %v976 = stablehlo.broadcast_in_dim %gp11, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v977 = stablehlo.broadcast_in_dim %btp11, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v978 = stablehlo.multiply %v975, %v976 : tensor<32x96x14x14xf32>
    %v979 = stablehlo.add %v978, %v977 : tensor<32x96x14x14xf32>
    %v980 = stablehlo.reshape %v979 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v981 = stablehlo.reshape %v980 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v982 = stablehlo.convolution(%v981, %We12)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x14x14xf32>, tensor<576x96x1x1xf32>) -> tensor<32x576x14x14xf32>
    %v983 = stablehlo.broadcast_in_dim %zb576, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v984 = stablehlo.add %v982, %v983 : tensor<32x576x14x14xf32>
    %v985 = stablehlo.reshape %v984 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v986 = stablehlo.reshape %v985 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v987 = stablehlo.constant dense<0.0> : tensor<f32>
    %v988 = stablehlo.constant dense<196.0> : tensor<32x576x14x14xf32>
    %v989 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v990 = stablehlo.reduce(%v986 init: %v987) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v991 = stablehlo.broadcast_in_dim %v990, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v992 = stablehlo.divide %v991, %v988 : tensor<32x576x14x14xf32>
    %v993 = stablehlo.subtract %v986, %v992 : tensor<32x576x14x14xf32>
    %v994 = stablehlo.multiply %v993, %v993 : tensor<32x576x14x14xf32>
    %v995 = stablehlo.reduce(%v994 init: %v987) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v996 = stablehlo.broadcast_in_dim %v995, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v997 = stablehlo.divide %v996, %v988 : tensor<32x576x14x14xf32>
    %v998 = stablehlo.add %v997, %v989 : tensor<32x576x14x14xf32>
    %v999 = stablehlo.rsqrt %v998 : tensor<32x576x14x14xf32>
    %v1000 = stablehlo.multiply %v993, %v999 : tensor<32x576x14x14xf32>
    %v1001 = stablehlo.broadcast_in_dim %ge12, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v1002 = stablehlo.broadcast_in_dim %bte12, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v1003 = stablehlo.multiply %v1000, %v1001 : tensor<32x576x14x14xf32>
    %v1004 = stablehlo.add %v1003, %v1002 : tensor<32x576x14x14xf32>
    %v1005 = stablehlo.reshape %v1004 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v1006 = stablehlo.reshape %v1005 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v1007 = stablehlo.constant dense<0.0> : tensor<32x576x14x14xf32>
    %v1008 = stablehlo.constant dense<6.0> : tensor<32x576x14x14xf32>
    %v1009 = stablehlo.maximum %v1006, %v1007 : tensor<32x576x14x14xf32>
    %v1010 = stablehlo.minimum %v1009, %v1008 : tensor<32x576x14x14xf32>
    %v1011 = stablehlo.reshape %v1010 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v1012 = stablehlo.reshape %v1011 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v1013 = stablehlo.convolution(%v1012, %Wd12)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 576 : i64} : (tensor<32x576x14x14xf32>, tensor<576x1x3x3xf32>) -> tensor<32x576x14x14xf32>
    %v1014 = stablehlo.broadcast_in_dim %zb576, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v1015 = stablehlo.add %v1013, %v1014 : tensor<32x576x14x14xf32>
    %v1016 = stablehlo.reshape %v1015 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v1017 = stablehlo.reshape %v1016 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v1018 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1019 = stablehlo.constant dense<196.0> : tensor<32x576x14x14xf32>
    %v1020 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v1021 = stablehlo.reduce(%v1017 init: %v1018) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v1022 = stablehlo.broadcast_in_dim %v1021, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v1023 = stablehlo.divide %v1022, %v1019 : tensor<32x576x14x14xf32>
    %v1024 = stablehlo.subtract %v1017, %v1023 : tensor<32x576x14x14xf32>
    %v1025 = stablehlo.multiply %v1024, %v1024 : tensor<32x576x14x14xf32>
    %v1026 = stablehlo.reduce(%v1025 init: %v1018) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v1027 = stablehlo.broadcast_in_dim %v1026, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v1028 = stablehlo.divide %v1027, %v1019 : tensor<32x576x14x14xf32>
    %v1029 = stablehlo.add %v1028, %v1020 : tensor<32x576x14x14xf32>
    %v1030 = stablehlo.rsqrt %v1029 : tensor<32x576x14x14xf32>
    %v1031 = stablehlo.multiply %v1024, %v1030 : tensor<32x576x14x14xf32>
    %v1032 = stablehlo.broadcast_in_dim %gd12, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v1033 = stablehlo.broadcast_in_dim %btd12, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v1034 = stablehlo.multiply %v1031, %v1032 : tensor<32x576x14x14xf32>
    %v1035 = stablehlo.add %v1034, %v1033 : tensor<32x576x14x14xf32>
    %v1036 = stablehlo.reshape %v1035 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v1037 = stablehlo.reshape %v1036 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v1038 = stablehlo.constant dense<0.0> : tensor<32x576x14x14xf32>
    %v1039 = stablehlo.constant dense<6.0> : tensor<32x576x14x14xf32>
    %v1040 = stablehlo.maximum %v1037, %v1038 : tensor<32x576x14x14xf32>
    %v1041 = stablehlo.minimum %v1040, %v1039 : tensor<32x576x14x14xf32>
    %v1042 = stablehlo.reshape %v1041 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v1043 = stablehlo.reshape %v1042 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v1044 = stablehlo.convolution(%v1043, %Wp12)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x576x14x14xf32>, tensor<96x576x1x1xf32>) -> tensor<32x96x14x14xf32>
    %v1045 = stablehlo.broadcast_in_dim %zb96, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v1046 = stablehlo.add %v1044, %v1045 : tensor<32x96x14x14xf32>
    %v1047 = stablehlo.reshape %v1046 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v1048 = stablehlo.reshape %v1047 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v1049 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1050 = stablehlo.constant dense<196.0> : tensor<32x96x14x14xf32>
    %v1051 = stablehlo.constant dense<1.0e-5> : tensor<32x96x14x14xf32>
    %v1052 = stablehlo.reduce(%v1048 init: %v1049) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1053 = stablehlo.broadcast_in_dim %v1052, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x14x14xf32>
    %v1054 = stablehlo.divide %v1053, %v1050 : tensor<32x96x14x14xf32>
    %v1055 = stablehlo.subtract %v1048, %v1054 : tensor<32x96x14x14xf32>
    %v1056 = stablehlo.multiply %v1055, %v1055 : tensor<32x96x14x14xf32>
    %v1057 = stablehlo.reduce(%v1056 init: %v1049) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1058 = stablehlo.broadcast_in_dim %v1057, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x14x14xf32>
    %v1059 = stablehlo.divide %v1058, %v1050 : tensor<32x96x14x14xf32>
    %v1060 = stablehlo.add %v1059, %v1051 : tensor<32x96x14x14xf32>
    %v1061 = stablehlo.rsqrt %v1060 : tensor<32x96x14x14xf32>
    %v1062 = stablehlo.multiply %v1055, %v1061 : tensor<32x96x14x14xf32>
    %v1063 = stablehlo.broadcast_in_dim %gp12, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v1064 = stablehlo.broadcast_in_dim %btp12, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v1065 = stablehlo.multiply %v1062, %v1063 : tensor<32x96x14x14xf32>
    %v1066 = stablehlo.add %v1065, %v1064 : tensor<32x96x14x14xf32>
    %v1067 = stablehlo.reshape %v1066 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v1068 = stablehlo.reshape %v1067 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v1069 = stablehlo.reshape %v980 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v1070 = stablehlo.add %v1068, %v1069 : tensor<32x96x14x14xf32>
    %v1071 = stablehlo.reshape %v1070 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v1072 = stablehlo.reshape %v1071 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v1073 = stablehlo.convolution(%v1072, %We13)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x14x14xf32>, tensor<576x96x1x1xf32>) -> tensor<32x576x14x14xf32>
    %v1074 = stablehlo.broadcast_in_dim %zb576, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v1075 = stablehlo.add %v1073, %v1074 : tensor<32x576x14x14xf32>
    %v1076 = stablehlo.reshape %v1075 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v1077 = stablehlo.reshape %v1076 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v1078 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1079 = stablehlo.constant dense<196.0> : tensor<32x576x14x14xf32>
    %v1080 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v1081 = stablehlo.reduce(%v1077 init: %v1078) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v1082 = stablehlo.broadcast_in_dim %v1081, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v1083 = stablehlo.divide %v1082, %v1079 : tensor<32x576x14x14xf32>
    %v1084 = stablehlo.subtract %v1077, %v1083 : tensor<32x576x14x14xf32>
    %v1085 = stablehlo.multiply %v1084, %v1084 : tensor<32x576x14x14xf32>
    %v1086 = stablehlo.reduce(%v1085 init: %v1078) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v1087 = stablehlo.broadcast_in_dim %v1086, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v1088 = stablehlo.divide %v1087, %v1079 : tensor<32x576x14x14xf32>
    %v1089 = stablehlo.add %v1088, %v1080 : tensor<32x576x14x14xf32>
    %v1090 = stablehlo.rsqrt %v1089 : tensor<32x576x14x14xf32>
    %v1091 = stablehlo.multiply %v1084, %v1090 : tensor<32x576x14x14xf32>
    %v1092 = stablehlo.broadcast_in_dim %ge13, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v1093 = stablehlo.broadcast_in_dim %bte13, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v1094 = stablehlo.multiply %v1091, %v1092 : tensor<32x576x14x14xf32>
    %v1095 = stablehlo.add %v1094, %v1093 : tensor<32x576x14x14xf32>
    %v1096 = stablehlo.reshape %v1095 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v1097 = stablehlo.reshape %v1096 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v1098 = stablehlo.constant dense<0.0> : tensor<32x576x14x14xf32>
    %v1099 = stablehlo.constant dense<6.0> : tensor<32x576x14x14xf32>
    %v1100 = stablehlo.maximum %v1097, %v1098 : tensor<32x576x14x14xf32>
    %v1101 = stablehlo.minimum %v1100, %v1099 : tensor<32x576x14x14xf32>
    %v1102 = stablehlo.reshape %v1101 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v1103 = stablehlo.reshape %v1102 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v1104 = stablehlo.convolution(%v1103, %Wd13)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 576 : i64} : (tensor<32x576x14x14xf32>, tensor<576x1x3x3xf32>) -> tensor<32x576x14x14xf32>
    %v1105 = stablehlo.broadcast_in_dim %zb576, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v1106 = stablehlo.add %v1104, %v1105 : tensor<32x576x14x14xf32>
    %v1107 = stablehlo.reshape %v1106 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v1108 = stablehlo.reshape %v1107 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v1109 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1110 = stablehlo.constant dense<196.0> : tensor<32x576x14x14xf32>
    %v1111 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v1112 = stablehlo.reduce(%v1108 init: %v1109) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v1113 = stablehlo.broadcast_in_dim %v1112, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v1114 = stablehlo.divide %v1113, %v1110 : tensor<32x576x14x14xf32>
    %v1115 = stablehlo.subtract %v1108, %v1114 : tensor<32x576x14x14xf32>
    %v1116 = stablehlo.multiply %v1115, %v1115 : tensor<32x576x14x14xf32>
    %v1117 = stablehlo.reduce(%v1116 init: %v1109) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v1118 = stablehlo.broadcast_in_dim %v1117, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v1119 = stablehlo.divide %v1118, %v1110 : tensor<32x576x14x14xf32>
    %v1120 = stablehlo.add %v1119, %v1111 : tensor<32x576x14x14xf32>
    %v1121 = stablehlo.rsqrt %v1120 : tensor<32x576x14x14xf32>
    %v1122 = stablehlo.multiply %v1115, %v1121 : tensor<32x576x14x14xf32>
    %v1123 = stablehlo.broadcast_in_dim %gd13, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v1124 = stablehlo.broadcast_in_dim %btd13, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v1125 = stablehlo.multiply %v1122, %v1123 : tensor<32x576x14x14xf32>
    %v1126 = stablehlo.add %v1125, %v1124 : tensor<32x576x14x14xf32>
    %v1127 = stablehlo.reshape %v1126 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v1128 = stablehlo.reshape %v1127 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v1129 = stablehlo.constant dense<0.0> : tensor<32x576x14x14xf32>
    %v1130 = stablehlo.constant dense<6.0> : tensor<32x576x14x14xf32>
    %v1131 = stablehlo.maximum %v1128, %v1129 : tensor<32x576x14x14xf32>
    %v1132 = stablehlo.minimum %v1131, %v1130 : tensor<32x576x14x14xf32>
    %v1133 = stablehlo.reshape %v1132 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v1134 = stablehlo.reshape %v1133 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v1135 = stablehlo.convolution(%v1134, %Wp13)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x576x14x14xf32>, tensor<96x576x1x1xf32>) -> tensor<32x96x14x14xf32>
    %v1136 = stablehlo.broadcast_in_dim %zb96, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v1137 = stablehlo.add %v1135, %v1136 : tensor<32x96x14x14xf32>
    %v1138 = stablehlo.reshape %v1137 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v1139 = stablehlo.reshape %v1138 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v1140 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1141 = stablehlo.constant dense<196.0> : tensor<32x96x14x14xf32>
    %v1142 = stablehlo.constant dense<1.0e-5> : tensor<32x96x14x14xf32>
    %v1143 = stablehlo.reduce(%v1139 init: %v1140) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1144 = stablehlo.broadcast_in_dim %v1143, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x14x14xf32>
    %v1145 = stablehlo.divide %v1144, %v1141 : tensor<32x96x14x14xf32>
    %v1146 = stablehlo.subtract %v1139, %v1145 : tensor<32x96x14x14xf32>
    %v1147 = stablehlo.multiply %v1146, %v1146 : tensor<32x96x14x14xf32>
    %v1148 = stablehlo.reduce(%v1147 init: %v1140) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1149 = stablehlo.broadcast_in_dim %v1148, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x14x14xf32>
    %v1150 = stablehlo.divide %v1149, %v1141 : tensor<32x96x14x14xf32>
    %v1151 = stablehlo.add %v1150, %v1142 : tensor<32x96x14x14xf32>
    %v1152 = stablehlo.rsqrt %v1151 : tensor<32x96x14x14xf32>
    %v1153 = stablehlo.multiply %v1146, %v1152 : tensor<32x96x14x14xf32>
    %v1154 = stablehlo.broadcast_in_dim %gp13, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v1155 = stablehlo.broadcast_in_dim %btp13, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v1156 = stablehlo.multiply %v1153, %v1154 : tensor<32x96x14x14xf32>
    %v1157 = stablehlo.add %v1156, %v1155 : tensor<32x96x14x14xf32>
    %v1158 = stablehlo.reshape %v1157 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v1159 = stablehlo.reshape %v1158 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v1160 = stablehlo.reshape %v1071 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v1161 = stablehlo.add %v1159, %v1160 : tensor<32x96x14x14xf32>
    %v1162 = stablehlo.reshape %v1161 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v1163 = stablehlo.reshape %v1162 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v1164 = stablehlo.convolution(%v1163, %We14)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x14x14xf32>, tensor<576x96x1x1xf32>) -> tensor<32x576x14x14xf32>
    %v1165 = stablehlo.broadcast_in_dim %zb576, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v1166 = stablehlo.add %v1164, %v1165 : tensor<32x576x14x14xf32>
    %v1167 = stablehlo.reshape %v1166 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v1168 = stablehlo.reshape %v1167 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v1169 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1170 = stablehlo.constant dense<196.0> : tensor<32x576x14x14xf32>
    %v1171 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v1172 = stablehlo.reduce(%v1168 init: %v1169) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v1173 = stablehlo.broadcast_in_dim %v1172, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v1174 = stablehlo.divide %v1173, %v1170 : tensor<32x576x14x14xf32>
    %v1175 = stablehlo.subtract %v1168, %v1174 : tensor<32x576x14x14xf32>
    %v1176 = stablehlo.multiply %v1175, %v1175 : tensor<32x576x14x14xf32>
    %v1177 = stablehlo.reduce(%v1176 init: %v1169) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v1178 = stablehlo.broadcast_in_dim %v1177, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v1179 = stablehlo.divide %v1178, %v1170 : tensor<32x576x14x14xf32>
    %v1180 = stablehlo.add %v1179, %v1171 : tensor<32x576x14x14xf32>
    %v1181 = stablehlo.rsqrt %v1180 : tensor<32x576x14x14xf32>
    %v1182 = stablehlo.multiply %v1175, %v1181 : tensor<32x576x14x14xf32>
    %v1183 = stablehlo.broadcast_in_dim %ge14, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v1184 = stablehlo.broadcast_in_dim %bte14, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v1185 = stablehlo.multiply %v1182, %v1183 : tensor<32x576x14x14xf32>
    %v1186 = stablehlo.add %v1185, %v1184 : tensor<32x576x14x14xf32>
    %v1187 = stablehlo.reshape %v1186 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v1188 = stablehlo.reshape %v1187 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v1189 = stablehlo.constant dense<0.0> : tensor<32x576x14x14xf32>
    %v1190 = stablehlo.constant dense<6.0> : tensor<32x576x14x14xf32>
    %v1191 = stablehlo.maximum %v1188, %v1189 : tensor<32x576x14x14xf32>
    %v1192 = stablehlo.minimum %v1191, %v1190 : tensor<32x576x14x14xf32>
    %v1193 = stablehlo.reshape %v1192 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v1194 = stablehlo.reshape %v1193 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v1195 = stablehlo.convolution(%v1194, %Wd14)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 576 : i64} : (tensor<32x576x14x14xf32>, tensor<576x1x3x3xf32>) -> tensor<32x576x7x7xf32>
    %v1196 = stablehlo.broadcast_in_dim %zb576, dims = [1] : (tensor<576xf32>) -> tensor<32x576x7x7xf32>
    %v1197 = stablehlo.add %v1195, %v1196 : tensor<32x576x7x7xf32>
    %v1198 = stablehlo.reshape %v1197 : (tensor<32x576x7x7xf32>) -> tensor<32x28224xf32>
    %v1199 = stablehlo.reshape %v1198 : (tensor<32x28224xf32>) -> tensor<32x576x7x7xf32>
    %v1200 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1201 = stablehlo.constant dense<49.0> : tensor<32x576x7x7xf32>
    %v1202 = stablehlo.constant dense<1.0e-5> : tensor<32x576x7x7xf32>
    %v1203 = stablehlo.reduce(%v1199 init: %v1200) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x7x7xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v1204 = stablehlo.broadcast_in_dim %v1203, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x7x7xf32>
    %v1205 = stablehlo.divide %v1204, %v1201 : tensor<32x576x7x7xf32>
    %v1206 = stablehlo.subtract %v1199, %v1205 : tensor<32x576x7x7xf32>
    %v1207 = stablehlo.multiply %v1206, %v1206 : tensor<32x576x7x7xf32>
    %v1208 = stablehlo.reduce(%v1207 init: %v1200) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x7x7xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v1209 = stablehlo.broadcast_in_dim %v1208, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x7x7xf32>
    %v1210 = stablehlo.divide %v1209, %v1201 : tensor<32x576x7x7xf32>
    %v1211 = stablehlo.add %v1210, %v1202 : tensor<32x576x7x7xf32>
    %v1212 = stablehlo.rsqrt %v1211 : tensor<32x576x7x7xf32>
    %v1213 = stablehlo.multiply %v1206, %v1212 : tensor<32x576x7x7xf32>
    %v1214 = stablehlo.broadcast_in_dim %gd14, dims = [1] : (tensor<576xf32>) -> tensor<32x576x7x7xf32>
    %v1215 = stablehlo.broadcast_in_dim %btd14, dims = [1] : (tensor<576xf32>) -> tensor<32x576x7x7xf32>
    %v1216 = stablehlo.multiply %v1213, %v1214 : tensor<32x576x7x7xf32>
    %v1217 = stablehlo.add %v1216, %v1215 : tensor<32x576x7x7xf32>
    %v1218 = stablehlo.reshape %v1217 : (tensor<32x576x7x7xf32>) -> tensor<32x28224xf32>
    %v1219 = stablehlo.reshape %v1218 : (tensor<32x28224xf32>) -> tensor<32x576x7x7xf32>
    %v1220 = stablehlo.constant dense<0.0> : tensor<32x576x7x7xf32>
    %v1221 = stablehlo.constant dense<6.0> : tensor<32x576x7x7xf32>
    %v1222 = stablehlo.maximum %v1219, %v1220 : tensor<32x576x7x7xf32>
    %v1223 = stablehlo.minimum %v1222, %v1221 : tensor<32x576x7x7xf32>
    %v1224 = stablehlo.reshape %v1223 : (tensor<32x576x7x7xf32>) -> tensor<32x28224xf32>
    %v1225 = stablehlo.reshape %v1224 : (tensor<32x28224xf32>) -> tensor<32x576x7x7xf32>
    %v1226 = stablehlo.convolution(%v1225, %Wp14)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x576x7x7xf32>, tensor<160x576x1x1xf32>) -> tensor<32x160x7x7xf32>
    %v1227 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1228 = stablehlo.add %v1226, %v1227 : tensor<32x160x7x7xf32>
    %v1229 = stablehlo.reshape %v1228 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v1230 = stablehlo.reshape %v1229 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v1231 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1232 = stablehlo.constant dense<49.0> : tensor<32x160x7x7xf32>
    %v1233 = stablehlo.constant dense<1.0e-5> : tensor<32x160x7x7xf32>
    %v1234 = stablehlo.reduce(%v1230 init: %v1231) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<32x160xf32>
    %v1235 = stablehlo.broadcast_in_dim %v1234, dims = [0, 1] : (tensor<32x160xf32>) -> tensor<32x160x7x7xf32>
    %v1236 = stablehlo.divide %v1235, %v1232 : tensor<32x160x7x7xf32>
    %v1237 = stablehlo.subtract %v1230, %v1236 : tensor<32x160x7x7xf32>
    %v1238 = stablehlo.multiply %v1237, %v1237 : tensor<32x160x7x7xf32>
    %v1239 = stablehlo.reduce(%v1238 init: %v1231) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<32x160xf32>
    %v1240 = stablehlo.broadcast_in_dim %v1239, dims = [0, 1] : (tensor<32x160xf32>) -> tensor<32x160x7x7xf32>
    %v1241 = stablehlo.divide %v1240, %v1232 : tensor<32x160x7x7xf32>
    %v1242 = stablehlo.add %v1241, %v1233 : tensor<32x160x7x7xf32>
    %v1243 = stablehlo.rsqrt %v1242 : tensor<32x160x7x7xf32>
    %v1244 = stablehlo.multiply %v1237, %v1243 : tensor<32x160x7x7xf32>
    %v1245 = stablehlo.broadcast_in_dim %gp14, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1246 = stablehlo.broadcast_in_dim %btp14, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1247 = stablehlo.multiply %v1244, %v1245 : tensor<32x160x7x7xf32>
    %v1248 = stablehlo.add %v1247, %v1246 : tensor<32x160x7x7xf32>
    %v1249 = stablehlo.reshape %v1248 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v1250 = stablehlo.reshape %v1249 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v1251 = stablehlo.convolution(%v1250, %We15)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x7x7xf32>, tensor<960x160x1x1xf32>) -> tensor<32x960x7x7xf32>
    %v1252 = stablehlo.broadcast_in_dim %zb960, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1253 = stablehlo.add %v1251, %v1252 : tensor<32x960x7x7xf32>
    %v1254 = stablehlo.reshape %v1253 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1255 = stablehlo.reshape %v1254 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1256 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1257 = stablehlo.constant dense<49.0> : tensor<32x960x7x7xf32>
    %v1258 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v1259 = stablehlo.reduce(%v1255 init: %v1256) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1260 = stablehlo.broadcast_in_dim %v1259, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1261 = stablehlo.divide %v1260, %v1257 : tensor<32x960x7x7xf32>
    %v1262 = stablehlo.subtract %v1255, %v1261 : tensor<32x960x7x7xf32>
    %v1263 = stablehlo.multiply %v1262, %v1262 : tensor<32x960x7x7xf32>
    %v1264 = stablehlo.reduce(%v1263 init: %v1256) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1265 = stablehlo.broadcast_in_dim %v1264, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1266 = stablehlo.divide %v1265, %v1257 : tensor<32x960x7x7xf32>
    %v1267 = stablehlo.add %v1266, %v1258 : tensor<32x960x7x7xf32>
    %v1268 = stablehlo.rsqrt %v1267 : tensor<32x960x7x7xf32>
    %v1269 = stablehlo.multiply %v1262, %v1268 : tensor<32x960x7x7xf32>
    %v1270 = stablehlo.broadcast_in_dim %ge15, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1271 = stablehlo.broadcast_in_dim %bte15, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1272 = stablehlo.multiply %v1269, %v1270 : tensor<32x960x7x7xf32>
    %v1273 = stablehlo.add %v1272, %v1271 : tensor<32x960x7x7xf32>
    %v1274 = stablehlo.reshape %v1273 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1275 = stablehlo.reshape %v1274 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1276 = stablehlo.constant dense<0.0> : tensor<32x960x7x7xf32>
    %v1277 = stablehlo.constant dense<6.0> : tensor<32x960x7x7xf32>
    %v1278 = stablehlo.maximum %v1275, %v1276 : tensor<32x960x7x7xf32>
    %v1279 = stablehlo.minimum %v1278, %v1277 : tensor<32x960x7x7xf32>
    %v1280 = stablehlo.reshape %v1279 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1281 = stablehlo.reshape %v1280 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1282 = stablehlo.convolution(%v1281, %Wd15)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 960 : i64} : (tensor<32x960x7x7xf32>, tensor<960x1x3x3xf32>) -> tensor<32x960x7x7xf32>
    %v1283 = stablehlo.broadcast_in_dim %zb960, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1284 = stablehlo.add %v1282, %v1283 : tensor<32x960x7x7xf32>
    %v1285 = stablehlo.reshape %v1284 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1286 = stablehlo.reshape %v1285 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1287 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1288 = stablehlo.constant dense<49.0> : tensor<32x960x7x7xf32>
    %v1289 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v1290 = stablehlo.reduce(%v1286 init: %v1287) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1291 = stablehlo.broadcast_in_dim %v1290, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1292 = stablehlo.divide %v1291, %v1288 : tensor<32x960x7x7xf32>
    %v1293 = stablehlo.subtract %v1286, %v1292 : tensor<32x960x7x7xf32>
    %v1294 = stablehlo.multiply %v1293, %v1293 : tensor<32x960x7x7xf32>
    %v1295 = stablehlo.reduce(%v1294 init: %v1287) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1296 = stablehlo.broadcast_in_dim %v1295, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1297 = stablehlo.divide %v1296, %v1288 : tensor<32x960x7x7xf32>
    %v1298 = stablehlo.add %v1297, %v1289 : tensor<32x960x7x7xf32>
    %v1299 = stablehlo.rsqrt %v1298 : tensor<32x960x7x7xf32>
    %v1300 = stablehlo.multiply %v1293, %v1299 : tensor<32x960x7x7xf32>
    %v1301 = stablehlo.broadcast_in_dim %gd15, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1302 = stablehlo.broadcast_in_dim %btd15, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1303 = stablehlo.multiply %v1300, %v1301 : tensor<32x960x7x7xf32>
    %v1304 = stablehlo.add %v1303, %v1302 : tensor<32x960x7x7xf32>
    %v1305 = stablehlo.reshape %v1304 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1306 = stablehlo.reshape %v1305 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1307 = stablehlo.constant dense<0.0> : tensor<32x960x7x7xf32>
    %v1308 = stablehlo.constant dense<6.0> : tensor<32x960x7x7xf32>
    %v1309 = stablehlo.maximum %v1306, %v1307 : tensor<32x960x7x7xf32>
    %v1310 = stablehlo.minimum %v1309, %v1308 : tensor<32x960x7x7xf32>
    %v1311 = stablehlo.reshape %v1310 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1312 = stablehlo.reshape %v1311 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1313 = stablehlo.convolution(%v1312, %Wp15)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x960x7x7xf32>, tensor<160x960x1x1xf32>) -> tensor<32x160x7x7xf32>
    %v1314 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1315 = stablehlo.add %v1313, %v1314 : tensor<32x160x7x7xf32>
    %v1316 = stablehlo.reshape %v1315 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v1317 = stablehlo.reshape %v1316 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v1318 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1319 = stablehlo.constant dense<49.0> : tensor<32x160x7x7xf32>
    %v1320 = stablehlo.constant dense<1.0e-5> : tensor<32x160x7x7xf32>
    %v1321 = stablehlo.reduce(%v1317 init: %v1318) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<32x160xf32>
    %v1322 = stablehlo.broadcast_in_dim %v1321, dims = [0, 1] : (tensor<32x160xf32>) -> tensor<32x160x7x7xf32>
    %v1323 = stablehlo.divide %v1322, %v1319 : tensor<32x160x7x7xf32>
    %v1324 = stablehlo.subtract %v1317, %v1323 : tensor<32x160x7x7xf32>
    %v1325 = stablehlo.multiply %v1324, %v1324 : tensor<32x160x7x7xf32>
    %v1326 = stablehlo.reduce(%v1325 init: %v1318) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<32x160xf32>
    %v1327 = stablehlo.broadcast_in_dim %v1326, dims = [0, 1] : (tensor<32x160xf32>) -> tensor<32x160x7x7xf32>
    %v1328 = stablehlo.divide %v1327, %v1319 : tensor<32x160x7x7xf32>
    %v1329 = stablehlo.add %v1328, %v1320 : tensor<32x160x7x7xf32>
    %v1330 = stablehlo.rsqrt %v1329 : tensor<32x160x7x7xf32>
    %v1331 = stablehlo.multiply %v1324, %v1330 : tensor<32x160x7x7xf32>
    %v1332 = stablehlo.broadcast_in_dim %gp15, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1333 = stablehlo.broadcast_in_dim %btp15, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1334 = stablehlo.multiply %v1331, %v1332 : tensor<32x160x7x7xf32>
    %v1335 = stablehlo.add %v1334, %v1333 : tensor<32x160x7x7xf32>
    %v1336 = stablehlo.reshape %v1335 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v1337 = stablehlo.reshape %v1336 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v1338 = stablehlo.reshape %v1249 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v1339 = stablehlo.add %v1337, %v1338 : tensor<32x160x7x7xf32>
    %v1340 = stablehlo.reshape %v1339 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v1341 = stablehlo.reshape %v1340 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v1342 = stablehlo.convolution(%v1341, %We16)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x7x7xf32>, tensor<960x160x1x1xf32>) -> tensor<32x960x7x7xf32>
    %v1343 = stablehlo.broadcast_in_dim %zb960, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1344 = stablehlo.add %v1342, %v1343 : tensor<32x960x7x7xf32>
    %v1345 = stablehlo.reshape %v1344 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1346 = stablehlo.reshape %v1345 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1347 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1348 = stablehlo.constant dense<49.0> : tensor<32x960x7x7xf32>
    %v1349 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v1350 = stablehlo.reduce(%v1346 init: %v1347) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1351 = stablehlo.broadcast_in_dim %v1350, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1352 = stablehlo.divide %v1351, %v1348 : tensor<32x960x7x7xf32>
    %v1353 = stablehlo.subtract %v1346, %v1352 : tensor<32x960x7x7xf32>
    %v1354 = stablehlo.multiply %v1353, %v1353 : tensor<32x960x7x7xf32>
    %v1355 = stablehlo.reduce(%v1354 init: %v1347) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1356 = stablehlo.broadcast_in_dim %v1355, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1357 = stablehlo.divide %v1356, %v1348 : tensor<32x960x7x7xf32>
    %v1358 = stablehlo.add %v1357, %v1349 : tensor<32x960x7x7xf32>
    %v1359 = stablehlo.rsqrt %v1358 : tensor<32x960x7x7xf32>
    %v1360 = stablehlo.multiply %v1353, %v1359 : tensor<32x960x7x7xf32>
    %v1361 = stablehlo.broadcast_in_dim %ge16, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1362 = stablehlo.broadcast_in_dim %bte16, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1363 = stablehlo.multiply %v1360, %v1361 : tensor<32x960x7x7xf32>
    %v1364 = stablehlo.add %v1363, %v1362 : tensor<32x960x7x7xf32>
    %v1365 = stablehlo.reshape %v1364 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1366 = stablehlo.reshape %v1365 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1367 = stablehlo.constant dense<0.0> : tensor<32x960x7x7xf32>
    %v1368 = stablehlo.constant dense<6.0> : tensor<32x960x7x7xf32>
    %v1369 = stablehlo.maximum %v1366, %v1367 : tensor<32x960x7x7xf32>
    %v1370 = stablehlo.minimum %v1369, %v1368 : tensor<32x960x7x7xf32>
    %v1371 = stablehlo.reshape %v1370 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1372 = stablehlo.reshape %v1371 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1373 = stablehlo.convolution(%v1372, %Wd16)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 960 : i64} : (tensor<32x960x7x7xf32>, tensor<960x1x3x3xf32>) -> tensor<32x960x7x7xf32>
    %v1374 = stablehlo.broadcast_in_dim %zb960, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1375 = stablehlo.add %v1373, %v1374 : tensor<32x960x7x7xf32>
    %v1376 = stablehlo.reshape %v1375 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1377 = stablehlo.reshape %v1376 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1378 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1379 = stablehlo.constant dense<49.0> : tensor<32x960x7x7xf32>
    %v1380 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v1381 = stablehlo.reduce(%v1377 init: %v1378) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1382 = stablehlo.broadcast_in_dim %v1381, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1383 = stablehlo.divide %v1382, %v1379 : tensor<32x960x7x7xf32>
    %v1384 = stablehlo.subtract %v1377, %v1383 : tensor<32x960x7x7xf32>
    %v1385 = stablehlo.multiply %v1384, %v1384 : tensor<32x960x7x7xf32>
    %v1386 = stablehlo.reduce(%v1385 init: %v1378) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1387 = stablehlo.broadcast_in_dim %v1386, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1388 = stablehlo.divide %v1387, %v1379 : tensor<32x960x7x7xf32>
    %v1389 = stablehlo.add %v1388, %v1380 : tensor<32x960x7x7xf32>
    %v1390 = stablehlo.rsqrt %v1389 : tensor<32x960x7x7xf32>
    %v1391 = stablehlo.multiply %v1384, %v1390 : tensor<32x960x7x7xf32>
    %v1392 = stablehlo.broadcast_in_dim %gd16, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1393 = stablehlo.broadcast_in_dim %btd16, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1394 = stablehlo.multiply %v1391, %v1392 : tensor<32x960x7x7xf32>
    %v1395 = stablehlo.add %v1394, %v1393 : tensor<32x960x7x7xf32>
    %v1396 = stablehlo.reshape %v1395 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1397 = stablehlo.reshape %v1396 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1398 = stablehlo.constant dense<0.0> : tensor<32x960x7x7xf32>
    %v1399 = stablehlo.constant dense<6.0> : tensor<32x960x7x7xf32>
    %v1400 = stablehlo.maximum %v1397, %v1398 : tensor<32x960x7x7xf32>
    %v1401 = stablehlo.minimum %v1400, %v1399 : tensor<32x960x7x7xf32>
    %v1402 = stablehlo.reshape %v1401 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1403 = stablehlo.reshape %v1402 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1404 = stablehlo.convolution(%v1403, %Wp16)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x960x7x7xf32>, tensor<160x960x1x1xf32>) -> tensor<32x160x7x7xf32>
    %v1405 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1406 = stablehlo.add %v1404, %v1405 : tensor<32x160x7x7xf32>
    %v1407 = stablehlo.reshape %v1406 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v1408 = stablehlo.reshape %v1407 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v1409 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1410 = stablehlo.constant dense<49.0> : tensor<32x160x7x7xf32>
    %v1411 = stablehlo.constant dense<1.0e-5> : tensor<32x160x7x7xf32>
    %v1412 = stablehlo.reduce(%v1408 init: %v1409) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<32x160xf32>
    %v1413 = stablehlo.broadcast_in_dim %v1412, dims = [0, 1] : (tensor<32x160xf32>) -> tensor<32x160x7x7xf32>
    %v1414 = stablehlo.divide %v1413, %v1410 : tensor<32x160x7x7xf32>
    %v1415 = stablehlo.subtract %v1408, %v1414 : tensor<32x160x7x7xf32>
    %v1416 = stablehlo.multiply %v1415, %v1415 : tensor<32x160x7x7xf32>
    %v1417 = stablehlo.reduce(%v1416 init: %v1409) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<32x160xf32>
    %v1418 = stablehlo.broadcast_in_dim %v1417, dims = [0, 1] : (tensor<32x160xf32>) -> tensor<32x160x7x7xf32>
    %v1419 = stablehlo.divide %v1418, %v1410 : tensor<32x160x7x7xf32>
    %v1420 = stablehlo.add %v1419, %v1411 : tensor<32x160x7x7xf32>
    %v1421 = stablehlo.rsqrt %v1420 : tensor<32x160x7x7xf32>
    %v1422 = stablehlo.multiply %v1415, %v1421 : tensor<32x160x7x7xf32>
    %v1423 = stablehlo.broadcast_in_dim %gp16, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1424 = stablehlo.broadcast_in_dim %btp16, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1425 = stablehlo.multiply %v1422, %v1423 : tensor<32x160x7x7xf32>
    %v1426 = stablehlo.add %v1425, %v1424 : tensor<32x160x7x7xf32>
    %v1427 = stablehlo.reshape %v1426 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v1428 = stablehlo.reshape %v1427 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v1429 = stablehlo.reshape %v1340 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v1430 = stablehlo.add %v1428, %v1429 : tensor<32x160x7x7xf32>
    %v1431 = stablehlo.reshape %v1430 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v1432 = stablehlo.reshape %v1431 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v1433 = stablehlo.convolution(%v1432, %We17)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x7x7xf32>, tensor<960x160x1x1xf32>) -> tensor<32x960x7x7xf32>
    %v1434 = stablehlo.broadcast_in_dim %zb960, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1435 = stablehlo.add %v1433, %v1434 : tensor<32x960x7x7xf32>
    %v1436 = stablehlo.reshape %v1435 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1437 = stablehlo.reshape %v1436 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1438 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1439 = stablehlo.constant dense<49.0> : tensor<32x960x7x7xf32>
    %v1440 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v1441 = stablehlo.reduce(%v1437 init: %v1438) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1442 = stablehlo.broadcast_in_dim %v1441, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1443 = stablehlo.divide %v1442, %v1439 : tensor<32x960x7x7xf32>
    %v1444 = stablehlo.subtract %v1437, %v1443 : tensor<32x960x7x7xf32>
    %v1445 = stablehlo.multiply %v1444, %v1444 : tensor<32x960x7x7xf32>
    %v1446 = stablehlo.reduce(%v1445 init: %v1438) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1447 = stablehlo.broadcast_in_dim %v1446, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1448 = stablehlo.divide %v1447, %v1439 : tensor<32x960x7x7xf32>
    %v1449 = stablehlo.add %v1448, %v1440 : tensor<32x960x7x7xf32>
    %v1450 = stablehlo.rsqrt %v1449 : tensor<32x960x7x7xf32>
    %v1451 = stablehlo.multiply %v1444, %v1450 : tensor<32x960x7x7xf32>
    %v1452 = stablehlo.broadcast_in_dim %ge17, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1453 = stablehlo.broadcast_in_dim %bte17, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1454 = stablehlo.multiply %v1451, %v1452 : tensor<32x960x7x7xf32>
    %v1455 = stablehlo.add %v1454, %v1453 : tensor<32x960x7x7xf32>
    %v1456 = stablehlo.reshape %v1455 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1457 = stablehlo.reshape %v1456 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1458 = stablehlo.constant dense<0.0> : tensor<32x960x7x7xf32>
    %v1459 = stablehlo.constant dense<6.0> : tensor<32x960x7x7xf32>
    %v1460 = stablehlo.maximum %v1457, %v1458 : tensor<32x960x7x7xf32>
    %v1461 = stablehlo.minimum %v1460, %v1459 : tensor<32x960x7x7xf32>
    %v1462 = stablehlo.reshape %v1461 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1463 = stablehlo.reshape %v1462 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1464 = stablehlo.convolution(%v1463, %Wd17)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 960 : i64} : (tensor<32x960x7x7xf32>, tensor<960x1x3x3xf32>) -> tensor<32x960x7x7xf32>
    %v1465 = stablehlo.broadcast_in_dim %zb960, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1466 = stablehlo.add %v1464, %v1465 : tensor<32x960x7x7xf32>
    %v1467 = stablehlo.reshape %v1466 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1468 = stablehlo.reshape %v1467 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1469 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1470 = stablehlo.constant dense<49.0> : tensor<32x960x7x7xf32>
    %v1471 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v1472 = stablehlo.reduce(%v1468 init: %v1469) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1473 = stablehlo.broadcast_in_dim %v1472, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1474 = stablehlo.divide %v1473, %v1470 : tensor<32x960x7x7xf32>
    %v1475 = stablehlo.subtract %v1468, %v1474 : tensor<32x960x7x7xf32>
    %v1476 = stablehlo.multiply %v1475, %v1475 : tensor<32x960x7x7xf32>
    %v1477 = stablehlo.reduce(%v1476 init: %v1469) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1478 = stablehlo.broadcast_in_dim %v1477, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1479 = stablehlo.divide %v1478, %v1470 : tensor<32x960x7x7xf32>
    %v1480 = stablehlo.add %v1479, %v1471 : tensor<32x960x7x7xf32>
    %v1481 = stablehlo.rsqrt %v1480 : tensor<32x960x7x7xf32>
    %v1482 = stablehlo.multiply %v1475, %v1481 : tensor<32x960x7x7xf32>
    %v1483 = stablehlo.broadcast_in_dim %gd17, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1484 = stablehlo.broadcast_in_dim %btd17, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1485 = stablehlo.multiply %v1482, %v1483 : tensor<32x960x7x7xf32>
    %v1486 = stablehlo.add %v1485, %v1484 : tensor<32x960x7x7xf32>
    %v1487 = stablehlo.reshape %v1486 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1488 = stablehlo.reshape %v1487 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1489 = stablehlo.constant dense<0.0> : tensor<32x960x7x7xf32>
    %v1490 = stablehlo.constant dense<6.0> : tensor<32x960x7x7xf32>
    %v1491 = stablehlo.maximum %v1488, %v1489 : tensor<32x960x7x7xf32>
    %v1492 = stablehlo.minimum %v1491, %v1490 : tensor<32x960x7x7xf32>
    %v1493 = stablehlo.reshape %v1492 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1494 = stablehlo.reshape %v1493 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1495 = stablehlo.convolution(%v1494, %Wp17)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x960x7x7xf32>, tensor<320x960x1x1xf32>) -> tensor<32x320x7x7xf32>
    %v1496 = stablehlo.broadcast_in_dim %zb320, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1497 = stablehlo.add %v1495, %v1496 : tensor<32x320x7x7xf32>
    %v1498 = stablehlo.reshape %v1497 : (tensor<32x320x7x7xf32>) -> tensor<32x15680xf32>
    %v1499 = stablehlo.reshape %v1498 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1500 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1501 = stablehlo.constant dense<49.0> : tensor<32x320x7x7xf32>
    %v1502 = stablehlo.constant dense<1.0e-5> : tensor<32x320x7x7xf32>
    %v1503 = stablehlo.reduce(%v1499 init: %v1500) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<32x320xf32>
    %v1504 = stablehlo.broadcast_in_dim %v1503, dims = [0, 1] : (tensor<32x320xf32>) -> tensor<32x320x7x7xf32>
    %v1505 = stablehlo.divide %v1504, %v1501 : tensor<32x320x7x7xf32>
    %v1506 = stablehlo.subtract %v1499, %v1505 : tensor<32x320x7x7xf32>
    %v1507 = stablehlo.multiply %v1506, %v1506 : tensor<32x320x7x7xf32>
    %v1508 = stablehlo.reduce(%v1507 init: %v1500) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<32x320xf32>
    %v1509 = stablehlo.broadcast_in_dim %v1508, dims = [0, 1] : (tensor<32x320xf32>) -> tensor<32x320x7x7xf32>
    %v1510 = stablehlo.divide %v1509, %v1501 : tensor<32x320x7x7xf32>
    %v1511 = stablehlo.add %v1510, %v1502 : tensor<32x320x7x7xf32>
    %v1512 = stablehlo.rsqrt %v1511 : tensor<32x320x7x7xf32>
    %v1513 = stablehlo.multiply %v1506, %v1512 : tensor<32x320x7x7xf32>
    %v1514 = stablehlo.broadcast_in_dim %gp17, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1515 = stablehlo.broadcast_in_dim %btp17, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1516 = stablehlo.multiply %v1513, %v1514 : tensor<32x320x7x7xf32>
    %v1517 = stablehlo.add %v1516, %v1515 : tensor<32x320x7x7xf32>
    %v1518 = stablehlo.reshape %v1517 : (tensor<32x320x7x7xf32>) -> tensor<32x15680xf32>
    %v1519 = stablehlo.reshape %v1518 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1520 = stablehlo.convolution(%v1519, %Wh)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x320x7x7xf32>, tensor<1280x320x1x1xf32>) -> tensor<32x1280x7x7xf32>
    %v1521 = stablehlo.broadcast_in_dim %zb1280, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1522 = stablehlo.add %v1520, %v1521 : tensor<32x1280x7x7xf32>
    %v1523 = stablehlo.reshape %v1522 : (tensor<32x1280x7x7xf32>) -> tensor<32x62720xf32>
    %v1524 = stablehlo.reshape %v1523 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1525 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1526 = stablehlo.constant dense<49.0> : tensor<32x1280x7x7xf32>
    %v1527 = stablehlo.constant dense<1.0e-5> : tensor<32x1280x7x7xf32>
    %v1528 = stablehlo.reduce(%v1524 init: %v1525) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<32x1280xf32>
    %v1529 = stablehlo.broadcast_in_dim %v1528, dims = [0, 1] : (tensor<32x1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1530 = stablehlo.divide %v1529, %v1526 : tensor<32x1280x7x7xf32>
    %v1531 = stablehlo.subtract %v1524, %v1530 : tensor<32x1280x7x7xf32>
    %v1532 = stablehlo.multiply %v1531, %v1531 : tensor<32x1280x7x7xf32>
    %v1533 = stablehlo.reduce(%v1532 init: %v1525) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<32x1280xf32>
    %v1534 = stablehlo.broadcast_in_dim %v1533, dims = [0, 1] : (tensor<32x1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1535 = stablehlo.divide %v1534, %v1526 : tensor<32x1280x7x7xf32>
    %v1536 = stablehlo.add %v1535, %v1527 : tensor<32x1280x7x7xf32>
    %v1537 = stablehlo.rsqrt %v1536 : tensor<32x1280x7x7xf32>
    %v1538 = stablehlo.multiply %v1531, %v1537 : tensor<32x1280x7x7xf32>
    %v1539 = stablehlo.broadcast_in_dim %gh, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1540 = stablehlo.broadcast_in_dim %bth, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1541 = stablehlo.multiply %v1538, %v1539 : tensor<32x1280x7x7xf32>
    %v1542 = stablehlo.add %v1541, %v1540 : tensor<32x1280x7x7xf32>
    %v1543 = stablehlo.reshape %v1542 : (tensor<32x1280x7x7xf32>) -> tensor<32x62720xf32>
    %v1544 = stablehlo.reshape %v1543 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1545 = stablehlo.constant dense<0.0> : tensor<32x1280x7x7xf32>
    %v1546 = stablehlo.constant dense<6.0> : tensor<32x1280x7x7xf32>
    %v1547 = stablehlo.maximum %v1544, %v1545 : tensor<32x1280x7x7xf32>
    %v1548 = stablehlo.minimum %v1547, %v1546 : tensor<32x1280x7x7xf32>
    %v1549 = stablehlo.reshape %v1548 : (tensor<32x1280x7x7xf32>) -> tensor<32x62720xf32>
    %v1550 = stablehlo.reshape %v1549 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1551 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1552 = stablehlo.reduce(%v1550 init: %v1551) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<32x1280xf32>
    %v1553 = stablehlo.constant dense<49.0> : tensor<32x1280xf32>
    %v1554 = stablehlo.divide %v1552, %v1553 : tensor<32x1280xf32>
    %v1555 = stablehlo.dot_general %v1554, %Wfc, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1280xf32>, tensor<1280x10xf32>) -> tensor<32x10xf32>
    %v1556 = stablehlo.broadcast_in_dim %bfc, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v1557 = stablehlo.add %v1555, %v1556 : tensor<32x10xf32>
    %v1558 = stablehlo.exponential %v1557 : tensor<32x10xf32>
    %v1559 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1560 = stablehlo.reduce(%v1558 init: %v1559) applies stablehlo.add across dimensions = [1] : (tensor<32x10xf32>, tensor<f32>) -> tensor<32xf32>
    %v1561 = stablehlo.broadcast_in_dim %v1560, dims = [0] : (tensor<32xf32>) -> tensor<32x10xf32>
    %v1562 = stablehlo.divide %v1558, %v1561 : tensor<32x10xf32>
    %v1563 = stablehlo.subtract %v1562, %onehot : tensor<32x10xf32>
    %v1564 = stablehlo.dot_general %v1563, %Wfc, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<1280x10xf32>) -> tensor<32x1280xf32>
    %v1565 = stablehlo.constant dense<49.0> : tensor<32x1280xf32>
    %v1566 = stablehlo.divide %v1564, %v1565 : tensor<32x1280xf32>
    %v1567 = stablehlo.broadcast_in_dim %v1566, dims = [0, 1] : (tensor<32x1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1568 = stablehlo.reshape %v1567 : (tensor<32x1280x7x7xf32>) -> tensor<32x62720xf32>
    %v1569 = stablehlo.dot_general %v1554, %v1563, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1280xf32>, tensor<32x10xf32>) -> tensor<1280x10xf32>
    %v1570 = stablehlo.constant dense<0.3> : tensor<1280x10xf32>
    %v1571 = stablehlo.multiply %v1569, %v1570 : tensor<1280x10xf32>
    %v1572 = stablehlo.subtract %Wfc, %v1571 : tensor<1280x10xf32>
    %v1573 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1574 = stablehlo.reduce(%v1563 init: %v1573) applies stablehlo.add across dimensions = [0] : (tensor<32x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v1575 = stablehlo.constant dense<0.3> : tensor<10xf32>
    %v1576 = stablehlo.multiply %v1574, %v1575 : tensor<10xf32>
    %v1577 = stablehlo.subtract %bfc, %v1576 : tensor<10xf32>
    %v1578 = stablehlo.reshape %v1568 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1579 = stablehlo.reshape %v1543 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1580 = stablehlo.constant dense<0.0> : tensor<32x1280x7x7xf32>
    %v1581 = stablehlo.constant dense<6.0> : tensor<32x1280x7x7xf32>
    %v1582 = stablehlo.compare GT, %v1579, %v1580 : (tensor<32x1280x7x7xf32>, tensor<32x1280x7x7xf32>) -> tensor<32x1280x7x7xi1>
    %v1583 = stablehlo.compare LT, %v1579, %v1581 : (tensor<32x1280x7x7xf32>, tensor<32x1280x7x7xf32>) -> tensor<32x1280x7x7xi1>
    %v1584 = stablehlo.and %v1582, %v1583 : tensor<32x1280x7x7xi1>
    %v1585 = stablehlo.select %v1584, %v1578, %v1580 : tensor<32x1280x7x7xi1>, tensor<32x1280x7x7xf32>
    %v1586 = stablehlo.reshape %v1585 : (tensor<32x1280x7x7xf32>) -> tensor<32x62720xf32>
    %v1587 = stablehlo.reshape %v1586 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1588 = stablehlo.reshape %v1523 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1589 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1590 = stablehlo.constant dense<49.0> : tensor<32x1280x7x7xf32>
    %v1591 = stablehlo.constant dense<1.0e-5> : tensor<32x1280x7x7xf32>
    %v1592 = stablehlo.reduce(%v1588 init: %v1589) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<32x1280xf32>
    %v1593 = stablehlo.broadcast_in_dim %v1592, dims = [0, 1] : (tensor<32x1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1594 = stablehlo.divide %v1593, %v1590 : tensor<32x1280x7x7xf32>
    %v1595 = stablehlo.subtract %v1588, %v1594 : tensor<32x1280x7x7xf32>
    %v1596 = stablehlo.multiply %v1595, %v1595 : tensor<32x1280x7x7xf32>
    %v1597 = stablehlo.reduce(%v1596 init: %v1589) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<32x1280xf32>
    %v1598 = stablehlo.broadcast_in_dim %v1597, dims = [0, 1] : (tensor<32x1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1599 = stablehlo.divide %v1598, %v1590 : tensor<32x1280x7x7xf32>
    %v1600 = stablehlo.add %v1599, %v1591 : tensor<32x1280x7x7xf32>
    %v1601 = stablehlo.rsqrt %v1600 : tensor<32x1280x7x7xf32>
    %v1602 = stablehlo.multiply %v1595, %v1601 : tensor<32x1280x7x7xf32>
    %v1603 = stablehlo.broadcast_in_dim %gh, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1604 = stablehlo.multiply %v1603, %v1587 : tensor<32x1280x7x7xf32>
    %v1605 = stablehlo.reduce(%v1604 init: %v1589) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<32x1280xf32>
    %v1606 = stablehlo.broadcast_in_dim %v1605, dims = [0, 1] : (tensor<32x1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1607 = stablehlo.multiply %v1602, %v1604 : tensor<32x1280x7x7xf32>
    %v1608 = stablehlo.reduce(%v1607 init: %v1589) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<32x1280xf32>
    %v1609 = stablehlo.broadcast_in_dim %v1608, dims = [0, 1] : (tensor<32x1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1610 = stablehlo.multiply %v1604, %v1590 : tensor<32x1280x7x7xf32>
    %v1611 = stablehlo.subtract %v1610, %v1606 : tensor<32x1280x7x7xf32>
    %v1612 = stablehlo.multiply %v1602, %v1609 : tensor<32x1280x7x7xf32>
    %v1613 = stablehlo.subtract %v1611, %v1612 : tensor<32x1280x7x7xf32>
    %v1614 = stablehlo.divide %v1601, %v1590 : tensor<32x1280x7x7xf32>
    %v1615 = stablehlo.multiply %v1614, %v1613 : tensor<32x1280x7x7xf32>
    %v1616 = stablehlo.reshape %v1615 : (tensor<32x1280x7x7xf32>) -> tensor<32x62720xf32>
    %v1617 = stablehlo.reshape %v1616 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1618 = stablehlo.transpose %Wh, dims = [1, 0, 2, 3] : (tensor<1280x320x1x1xf32>) -> tensor<320x1280x1x1xf32>
    %v1619 = stablehlo.reverse %v1618, dims = [2, 3] : tensor<320x1280x1x1xf32>
    %v1620 = stablehlo.convolution(%v1617, %v1619)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1280x7x7xf32>, tensor<320x1280x1x1xf32>) -> tensor<32x320x7x7xf32>
    %v1621 = stablehlo.reshape %v1620 : (tensor<32x320x7x7xf32>) -> tensor<32x15680xf32>
    %v1622 = stablehlo.reshape %v1518 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1623 = stablehlo.reshape %v1616 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1624 = stablehlo.transpose %v1622, dims = [1, 0, 2, 3] : (tensor<32x320x7x7xf32>) -> tensor<320x32x7x7xf32>
    %v1625 = stablehlo.transpose %v1623, dims = [1, 0, 2, 3] : (tensor<32x1280x7x7xf32>) -> tensor<1280x32x7x7xf32>
    %v1626 = stablehlo.convolution(%v1624, %v1625)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<320x32x7x7xf32>, tensor<1280x32x7x7xf32>) -> tensor<320x1280x1x1xf32>
    %v1627 = stablehlo.transpose %v1626, dims = [1, 0, 2, 3] : (tensor<320x1280x1x1xf32>) -> tensor<1280x320x1x1xf32>
    %v1628 = stablehlo.constant dense<0.3> : tensor<1280x320x1x1xf32>
    %v1629 = stablehlo.multiply %v1627, %v1628 : tensor<1280x320x1x1xf32>
    %v1630 = stablehlo.subtract %Wh, %v1629 : tensor<1280x320x1x1xf32>
    %v1631 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1632 = stablehlo.reshape %v1523 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1633 = stablehlo.constant dense<49.0> : tensor<32x1280x7x7xf32>
    %v1634 = stablehlo.constant dense<1.0e-5> : tensor<32x1280x7x7xf32>
    %v1635 = stablehlo.reduce(%v1632 init: %v1631) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<32x1280xf32>
    %v1636 = stablehlo.broadcast_in_dim %v1635, dims = [0, 1] : (tensor<32x1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1637 = stablehlo.divide %v1636, %v1633 : tensor<32x1280x7x7xf32>
    %v1638 = stablehlo.subtract %v1632, %v1637 : tensor<32x1280x7x7xf32>
    %v1639 = stablehlo.multiply %v1638, %v1638 : tensor<32x1280x7x7xf32>
    %v1640 = stablehlo.reduce(%v1639 init: %v1631) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<32x1280xf32>
    %v1641 = stablehlo.broadcast_in_dim %v1640, dims = [0, 1] : (tensor<32x1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1642 = stablehlo.divide %v1641, %v1633 : tensor<32x1280x7x7xf32>
    %v1643 = stablehlo.add %v1642, %v1634 : tensor<32x1280x7x7xf32>
    %v1644 = stablehlo.rsqrt %v1643 : tensor<32x1280x7x7xf32>
    %v1645 = stablehlo.multiply %v1638, %v1644 : tensor<32x1280x7x7xf32>
    %v1646 = stablehlo.reshape %v1586 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1647 = stablehlo.multiply %v1646, %v1645 : tensor<32x1280x7x7xf32>
    %v1648 = stablehlo.reduce(%v1647 init: %v1631) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v1649 = stablehlo.constant dense<0.3> : tensor<1280xf32>
    %v1650 = stablehlo.multiply %v1648, %v1649 : tensor<1280xf32>
    %v1651 = stablehlo.subtract %gh, %v1650 : tensor<1280xf32>
    %v1652 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1653 = stablehlo.reshape %v1586 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1654 = stablehlo.reduce(%v1653 init: %v1652) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v1655 = stablehlo.constant dense<0.3> : tensor<1280xf32>
    %v1656 = stablehlo.multiply %v1654, %v1655 : tensor<1280xf32>
    %v1657 = stablehlo.subtract %bth, %v1656 : tensor<1280xf32>
    %v1658 = stablehlo.reshape %v1621 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1659 = stablehlo.reshape %v1498 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1660 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1661 = stablehlo.constant dense<49.0> : tensor<32x320x7x7xf32>
    %v1662 = stablehlo.constant dense<1.0e-5> : tensor<32x320x7x7xf32>
    %v1663 = stablehlo.reduce(%v1659 init: %v1660) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<32x320xf32>
    %v1664 = stablehlo.broadcast_in_dim %v1663, dims = [0, 1] : (tensor<32x320xf32>) -> tensor<32x320x7x7xf32>
    %v1665 = stablehlo.divide %v1664, %v1661 : tensor<32x320x7x7xf32>
    %v1666 = stablehlo.subtract %v1659, %v1665 : tensor<32x320x7x7xf32>
    %v1667 = stablehlo.multiply %v1666, %v1666 : tensor<32x320x7x7xf32>
    %v1668 = stablehlo.reduce(%v1667 init: %v1660) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<32x320xf32>
    %v1669 = stablehlo.broadcast_in_dim %v1668, dims = [0, 1] : (tensor<32x320xf32>) -> tensor<32x320x7x7xf32>
    %v1670 = stablehlo.divide %v1669, %v1661 : tensor<32x320x7x7xf32>
    %v1671 = stablehlo.add %v1670, %v1662 : tensor<32x320x7x7xf32>
    %v1672 = stablehlo.rsqrt %v1671 : tensor<32x320x7x7xf32>
    %v1673 = stablehlo.multiply %v1666, %v1672 : tensor<32x320x7x7xf32>
    %v1674 = stablehlo.broadcast_in_dim %gp17, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1675 = stablehlo.multiply %v1674, %v1658 : tensor<32x320x7x7xf32>
    %v1676 = stablehlo.reduce(%v1675 init: %v1660) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<32x320xf32>
    %v1677 = stablehlo.broadcast_in_dim %v1676, dims = [0, 1] : (tensor<32x320xf32>) -> tensor<32x320x7x7xf32>
    %v1678 = stablehlo.multiply %v1673, %v1675 : tensor<32x320x7x7xf32>
    %v1679 = stablehlo.reduce(%v1678 init: %v1660) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<32x320xf32>
    %v1680 = stablehlo.broadcast_in_dim %v1679, dims = [0, 1] : (tensor<32x320xf32>) -> tensor<32x320x7x7xf32>
    %v1681 = stablehlo.multiply %v1675, %v1661 : tensor<32x320x7x7xf32>
    %v1682 = stablehlo.subtract %v1681, %v1677 : tensor<32x320x7x7xf32>
    %v1683 = stablehlo.multiply %v1673, %v1680 : tensor<32x320x7x7xf32>
    %v1684 = stablehlo.subtract %v1682, %v1683 : tensor<32x320x7x7xf32>
    %v1685 = stablehlo.divide %v1672, %v1661 : tensor<32x320x7x7xf32>
    %v1686 = stablehlo.multiply %v1685, %v1684 : tensor<32x320x7x7xf32>
    %v1687 = stablehlo.reshape %v1686 : (tensor<32x320x7x7xf32>) -> tensor<32x15680xf32>
    %v1688 = stablehlo.reshape %v1687 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1689 = stablehlo.transpose %Wp17, dims = [1, 0, 2, 3] : (tensor<320x960x1x1xf32>) -> tensor<960x320x1x1xf32>
    %v1690 = stablehlo.reverse %v1689, dims = [2, 3] : tensor<960x320x1x1xf32>
    %v1691 = stablehlo.convolution(%v1688, %v1690)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x320x7x7xf32>, tensor<960x320x1x1xf32>) -> tensor<32x960x7x7xf32>
    %v1692 = stablehlo.reshape %v1691 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1693 = stablehlo.reshape %v1692 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1694 = stablehlo.reshape %v1487 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1695 = stablehlo.constant dense<0.0> : tensor<32x960x7x7xf32>
    %v1696 = stablehlo.constant dense<6.0> : tensor<32x960x7x7xf32>
    %v1697 = stablehlo.compare GT, %v1694, %v1695 : (tensor<32x960x7x7xf32>, tensor<32x960x7x7xf32>) -> tensor<32x960x7x7xi1>
    %v1698 = stablehlo.compare LT, %v1694, %v1696 : (tensor<32x960x7x7xf32>, tensor<32x960x7x7xf32>) -> tensor<32x960x7x7xi1>
    %v1699 = stablehlo.and %v1697, %v1698 : tensor<32x960x7x7xi1>
    %v1700 = stablehlo.select %v1699, %v1693, %v1695 : tensor<32x960x7x7xi1>, tensor<32x960x7x7xf32>
    %v1701 = stablehlo.reshape %v1700 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1702 = stablehlo.reshape %v1701 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1703 = stablehlo.reshape %v1467 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1704 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1705 = stablehlo.constant dense<49.0> : tensor<32x960x7x7xf32>
    %v1706 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v1707 = stablehlo.reduce(%v1703 init: %v1704) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1708 = stablehlo.broadcast_in_dim %v1707, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1709 = stablehlo.divide %v1708, %v1705 : tensor<32x960x7x7xf32>
    %v1710 = stablehlo.subtract %v1703, %v1709 : tensor<32x960x7x7xf32>
    %v1711 = stablehlo.multiply %v1710, %v1710 : tensor<32x960x7x7xf32>
    %v1712 = stablehlo.reduce(%v1711 init: %v1704) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1713 = stablehlo.broadcast_in_dim %v1712, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1714 = stablehlo.divide %v1713, %v1705 : tensor<32x960x7x7xf32>
    %v1715 = stablehlo.add %v1714, %v1706 : tensor<32x960x7x7xf32>
    %v1716 = stablehlo.rsqrt %v1715 : tensor<32x960x7x7xf32>
    %v1717 = stablehlo.multiply %v1710, %v1716 : tensor<32x960x7x7xf32>
    %v1718 = stablehlo.broadcast_in_dim %gd17, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1719 = stablehlo.multiply %v1718, %v1702 : tensor<32x960x7x7xf32>
    %v1720 = stablehlo.reduce(%v1719 init: %v1704) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1721 = stablehlo.broadcast_in_dim %v1720, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1722 = stablehlo.multiply %v1717, %v1719 : tensor<32x960x7x7xf32>
    %v1723 = stablehlo.reduce(%v1722 init: %v1704) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1724 = stablehlo.broadcast_in_dim %v1723, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1725 = stablehlo.multiply %v1719, %v1705 : tensor<32x960x7x7xf32>
    %v1726 = stablehlo.subtract %v1725, %v1721 : tensor<32x960x7x7xf32>
    %v1727 = stablehlo.multiply %v1717, %v1724 : tensor<32x960x7x7xf32>
    %v1728 = stablehlo.subtract %v1726, %v1727 : tensor<32x960x7x7xf32>
    %v1729 = stablehlo.divide %v1716, %v1705 : tensor<32x960x7x7xf32>
    %v1730 = stablehlo.multiply %v1729, %v1728 : tensor<32x960x7x7xf32>
    %v1731 = stablehlo.reshape %v1730 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1732 = stablehlo.reshape %v1731 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1733 = stablehlo.reverse %Wd17, dims = [2, 3] : tensor<960x1x3x3xf32>
    %v1734 = stablehlo.convolution(%v1732, %v1733)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 960 : i64} : (tensor<32x960x7x7xf32>, tensor<960x1x3x3xf32>) -> tensor<32x960x7x7xf32>
    %v1735 = stablehlo.reshape %v1734 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1736 = stablehlo.reshape %v1735 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1737 = stablehlo.reshape %v1456 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1738 = stablehlo.constant dense<0.0> : tensor<32x960x7x7xf32>
    %v1739 = stablehlo.constant dense<6.0> : tensor<32x960x7x7xf32>
    %v1740 = stablehlo.compare GT, %v1737, %v1738 : (tensor<32x960x7x7xf32>, tensor<32x960x7x7xf32>) -> tensor<32x960x7x7xi1>
    %v1741 = stablehlo.compare LT, %v1737, %v1739 : (tensor<32x960x7x7xf32>, tensor<32x960x7x7xf32>) -> tensor<32x960x7x7xi1>
    %v1742 = stablehlo.and %v1740, %v1741 : tensor<32x960x7x7xi1>
    %v1743 = stablehlo.select %v1742, %v1736, %v1738 : tensor<32x960x7x7xi1>, tensor<32x960x7x7xf32>
    %v1744 = stablehlo.reshape %v1743 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1745 = stablehlo.reshape %v1744 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1746 = stablehlo.reshape %v1436 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1747 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1748 = stablehlo.constant dense<49.0> : tensor<32x960x7x7xf32>
    %v1749 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v1750 = stablehlo.reduce(%v1746 init: %v1747) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1751 = stablehlo.broadcast_in_dim %v1750, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1752 = stablehlo.divide %v1751, %v1748 : tensor<32x960x7x7xf32>
    %v1753 = stablehlo.subtract %v1746, %v1752 : tensor<32x960x7x7xf32>
    %v1754 = stablehlo.multiply %v1753, %v1753 : tensor<32x960x7x7xf32>
    %v1755 = stablehlo.reduce(%v1754 init: %v1747) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1756 = stablehlo.broadcast_in_dim %v1755, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1757 = stablehlo.divide %v1756, %v1748 : tensor<32x960x7x7xf32>
    %v1758 = stablehlo.add %v1757, %v1749 : tensor<32x960x7x7xf32>
    %v1759 = stablehlo.rsqrt %v1758 : tensor<32x960x7x7xf32>
    %v1760 = stablehlo.multiply %v1753, %v1759 : tensor<32x960x7x7xf32>
    %v1761 = stablehlo.broadcast_in_dim %ge17, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1762 = stablehlo.multiply %v1761, %v1745 : tensor<32x960x7x7xf32>
    %v1763 = stablehlo.reduce(%v1762 init: %v1747) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1764 = stablehlo.broadcast_in_dim %v1763, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1765 = stablehlo.multiply %v1760, %v1762 : tensor<32x960x7x7xf32>
    %v1766 = stablehlo.reduce(%v1765 init: %v1747) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1767 = stablehlo.broadcast_in_dim %v1766, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1768 = stablehlo.multiply %v1762, %v1748 : tensor<32x960x7x7xf32>
    %v1769 = stablehlo.subtract %v1768, %v1764 : tensor<32x960x7x7xf32>
    %v1770 = stablehlo.multiply %v1760, %v1767 : tensor<32x960x7x7xf32>
    %v1771 = stablehlo.subtract %v1769, %v1770 : tensor<32x960x7x7xf32>
    %v1772 = stablehlo.divide %v1759, %v1748 : tensor<32x960x7x7xf32>
    %v1773 = stablehlo.multiply %v1772, %v1771 : tensor<32x960x7x7xf32>
    %v1774 = stablehlo.reshape %v1773 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1775 = stablehlo.reshape %v1774 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1776 = stablehlo.transpose %We17, dims = [1, 0, 2, 3] : (tensor<960x160x1x1xf32>) -> tensor<160x960x1x1xf32>
    %v1777 = stablehlo.reverse %v1776, dims = [2, 3] : tensor<160x960x1x1xf32>
    %v1778 = stablehlo.convolution(%v1775, %v1777)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x960x7x7xf32>, tensor<160x960x1x1xf32>) -> tensor<32x160x7x7xf32>
    %v1779 = stablehlo.reshape %v1778 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v1780 = stablehlo.reshape %v1431 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v1781 = stablehlo.reshape %v1774 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1782 = stablehlo.transpose %v1780, dims = [1, 0, 2, 3] : (tensor<32x160x7x7xf32>) -> tensor<160x32x7x7xf32>
    %v1783 = stablehlo.transpose %v1781, dims = [1, 0, 2, 3] : (tensor<32x960x7x7xf32>) -> tensor<960x32x7x7xf32>
    %v1784 = stablehlo.convolution(%v1782, %v1783)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<160x32x7x7xf32>, tensor<960x32x7x7xf32>) -> tensor<160x960x1x1xf32>
    %v1785 = stablehlo.transpose %v1784, dims = [1, 0, 2, 3] : (tensor<160x960x1x1xf32>) -> tensor<960x160x1x1xf32>
    %v1786 = stablehlo.constant dense<0.3> : tensor<960x160x1x1xf32>
    %v1787 = stablehlo.multiply %v1785, %v1786 : tensor<960x160x1x1xf32>
    %v1788 = stablehlo.subtract %We17, %v1787 : tensor<960x160x1x1xf32>
    %v1789 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1790 = stablehlo.reshape %v1436 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1791 = stablehlo.constant dense<49.0> : tensor<32x960x7x7xf32>
    %v1792 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v1793 = stablehlo.reduce(%v1790 init: %v1789) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1794 = stablehlo.broadcast_in_dim %v1793, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1795 = stablehlo.divide %v1794, %v1791 : tensor<32x960x7x7xf32>
    %v1796 = stablehlo.subtract %v1790, %v1795 : tensor<32x960x7x7xf32>
    %v1797 = stablehlo.multiply %v1796, %v1796 : tensor<32x960x7x7xf32>
    %v1798 = stablehlo.reduce(%v1797 init: %v1789) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1799 = stablehlo.broadcast_in_dim %v1798, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1800 = stablehlo.divide %v1799, %v1791 : tensor<32x960x7x7xf32>
    %v1801 = stablehlo.add %v1800, %v1792 : tensor<32x960x7x7xf32>
    %v1802 = stablehlo.rsqrt %v1801 : tensor<32x960x7x7xf32>
    %v1803 = stablehlo.multiply %v1796, %v1802 : tensor<32x960x7x7xf32>
    %v1804 = stablehlo.reshape %v1744 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1805 = stablehlo.multiply %v1804, %v1803 : tensor<32x960x7x7xf32>
    %v1806 = stablehlo.reduce(%v1805 init: %v1789) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1807 = stablehlo.constant dense<0.3> : tensor<960xf32>
    %v1808 = stablehlo.multiply %v1806, %v1807 : tensor<960xf32>
    %v1809 = stablehlo.subtract %ge17, %v1808 : tensor<960xf32>
    %v1810 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1811 = stablehlo.reshape %v1744 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1812 = stablehlo.reduce(%v1811 init: %v1810) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1813 = stablehlo.constant dense<0.3> : tensor<960xf32>
    %v1814 = stablehlo.multiply %v1812, %v1813 : tensor<960xf32>
    %v1815 = stablehlo.subtract %bte17, %v1814 : tensor<960xf32>
    %v1816 = stablehlo.reshape %v1462 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1817 = stablehlo.reshape %v1731 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1818 = stablehlo.transpose %v1816, dims = [1, 0, 2, 3] : (tensor<32x960x7x7xf32>) -> tensor<960x32x7x7xf32>
    %v1819 = stablehlo.transpose %v1817, dims = [1, 0, 2, 3] : (tensor<32x960x7x7xf32>) -> tensor<960x32x7x7xf32>
    %v1820 = stablehlo.convolution(%v1818, %v1819)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 960 : i64, feature_group_count = 1 : i64} : (tensor<960x32x7x7xf32>, tensor<960x32x7x7xf32>) -> tensor<1x960x3x3xf32>
    %v1821 = stablehlo.reshape %v1820 : (tensor<1x960x3x3xf32>) -> tensor<960x1x3x3xf32>
    %v1822 = stablehlo.constant dense<0.3> : tensor<960x1x3x3xf32>
    %v1823 = stablehlo.multiply %v1821, %v1822 : tensor<960x1x3x3xf32>
    %v1824 = stablehlo.subtract %Wd17, %v1823 : tensor<960x1x3x3xf32>
    %v1825 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1826 = stablehlo.reshape %v1467 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1827 = stablehlo.constant dense<49.0> : tensor<32x960x7x7xf32>
    %v1828 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v1829 = stablehlo.reduce(%v1826 init: %v1825) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1830 = stablehlo.broadcast_in_dim %v1829, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1831 = stablehlo.divide %v1830, %v1827 : tensor<32x960x7x7xf32>
    %v1832 = stablehlo.subtract %v1826, %v1831 : tensor<32x960x7x7xf32>
    %v1833 = stablehlo.multiply %v1832, %v1832 : tensor<32x960x7x7xf32>
    %v1834 = stablehlo.reduce(%v1833 init: %v1825) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1835 = stablehlo.broadcast_in_dim %v1834, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1836 = stablehlo.divide %v1835, %v1827 : tensor<32x960x7x7xf32>
    %v1837 = stablehlo.add %v1836, %v1828 : tensor<32x960x7x7xf32>
    %v1838 = stablehlo.rsqrt %v1837 : tensor<32x960x7x7xf32>
    %v1839 = stablehlo.multiply %v1832, %v1838 : tensor<32x960x7x7xf32>
    %v1840 = stablehlo.reshape %v1701 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1841 = stablehlo.multiply %v1840, %v1839 : tensor<32x960x7x7xf32>
    %v1842 = stablehlo.reduce(%v1841 init: %v1825) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1843 = stablehlo.constant dense<0.3> : tensor<960xf32>
    %v1844 = stablehlo.multiply %v1842, %v1843 : tensor<960xf32>
    %v1845 = stablehlo.subtract %gd17, %v1844 : tensor<960xf32>
    %v1846 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1847 = stablehlo.reshape %v1701 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1848 = stablehlo.reduce(%v1847 init: %v1846) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1849 = stablehlo.constant dense<0.3> : tensor<960xf32>
    %v1850 = stablehlo.multiply %v1848, %v1849 : tensor<960xf32>
    %v1851 = stablehlo.subtract %btd17, %v1850 : tensor<960xf32>
    %v1852 = stablehlo.reshape %v1493 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1853 = stablehlo.reshape %v1687 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1854 = stablehlo.transpose %v1852, dims = [1, 0, 2, 3] : (tensor<32x960x7x7xf32>) -> tensor<960x32x7x7xf32>
    %v1855 = stablehlo.transpose %v1853, dims = [1, 0, 2, 3] : (tensor<32x320x7x7xf32>) -> tensor<320x32x7x7xf32>
    %v1856 = stablehlo.convolution(%v1854, %v1855)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<960x32x7x7xf32>, tensor<320x32x7x7xf32>) -> tensor<960x320x1x1xf32>
    %v1857 = stablehlo.transpose %v1856, dims = [1, 0, 2, 3] : (tensor<960x320x1x1xf32>) -> tensor<320x960x1x1xf32>
    %v1858 = stablehlo.constant dense<0.3> : tensor<320x960x1x1xf32>
    %v1859 = stablehlo.multiply %v1857, %v1858 : tensor<320x960x1x1xf32>
    %v1860 = stablehlo.subtract %Wp17, %v1859 : tensor<320x960x1x1xf32>
    %v1861 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1862 = stablehlo.reshape %v1498 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1863 = stablehlo.constant dense<49.0> : tensor<32x320x7x7xf32>
    %v1864 = stablehlo.constant dense<1.0e-5> : tensor<32x320x7x7xf32>
    %v1865 = stablehlo.reduce(%v1862 init: %v1861) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<32x320xf32>
    %v1866 = stablehlo.broadcast_in_dim %v1865, dims = [0, 1] : (tensor<32x320xf32>) -> tensor<32x320x7x7xf32>
    %v1867 = stablehlo.divide %v1866, %v1863 : tensor<32x320x7x7xf32>
    %v1868 = stablehlo.subtract %v1862, %v1867 : tensor<32x320x7x7xf32>
    %v1869 = stablehlo.multiply %v1868, %v1868 : tensor<32x320x7x7xf32>
    %v1870 = stablehlo.reduce(%v1869 init: %v1861) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<32x320xf32>
    %v1871 = stablehlo.broadcast_in_dim %v1870, dims = [0, 1] : (tensor<32x320xf32>) -> tensor<32x320x7x7xf32>
    %v1872 = stablehlo.divide %v1871, %v1863 : tensor<32x320x7x7xf32>
    %v1873 = stablehlo.add %v1872, %v1864 : tensor<32x320x7x7xf32>
    %v1874 = stablehlo.rsqrt %v1873 : tensor<32x320x7x7xf32>
    %v1875 = stablehlo.multiply %v1868, %v1874 : tensor<32x320x7x7xf32>
    %v1876 = stablehlo.reshape %v1621 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1877 = stablehlo.multiply %v1876, %v1875 : tensor<32x320x7x7xf32>
    %v1878 = stablehlo.reduce(%v1877 init: %v1861) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v1879 = stablehlo.constant dense<0.3> : tensor<320xf32>
    %v1880 = stablehlo.multiply %v1878, %v1879 : tensor<320xf32>
    %v1881 = stablehlo.subtract %gp17, %v1880 : tensor<320xf32>
    %v1882 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1883 = stablehlo.reshape %v1621 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1884 = stablehlo.reduce(%v1883 init: %v1882) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v1885 = stablehlo.constant dense<0.3> : tensor<320xf32>
    %v1886 = stablehlo.multiply %v1884, %v1885 : tensor<320xf32>
    %v1887 = stablehlo.subtract %btp17, %v1886 : tensor<320xf32>
    %v1888 = stablehlo.reshape %v1779 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v1889 = stablehlo.reshape %v1407 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v1890 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1891 = stablehlo.constant dense<49.0> : tensor<32x160x7x7xf32>
    %v1892 = stablehlo.constant dense<1.0e-5> : tensor<32x160x7x7xf32>
    %v1893 = stablehlo.reduce(%v1889 init: %v1890) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<32x160xf32>
    %v1894 = stablehlo.broadcast_in_dim %v1893, dims = [0, 1] : (tensor<32x160xf32>) -> tensor<32x160x7x7xf32>
    %v1895 = stablehlo.divide %v1894, %v1891 : tensor<32x160x7x7xf32>
    %v1896 = stablehlo.subtract %v1889, %v1895 : tensor<32x160x7x7xf32>
    %v1897 = stablehlo.multiply %v1896, %v1896 : tensor<32x160x7x7xf32>
    %v1898 = stablehlo.reduce(%v1897 init: %v1890) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<32x160xf32>
    %v1899 = stablehlo.broadcast_in_dim %v1898, dims = [0, 1] : (tensor<32x160xf32>) -> tensor<32x160x7x7xf32>
    %v1900 = stablehlo.divide %v1899, %v1891 : tensor<32x160x7x7xf32>
    %v1901 = stablehlo.add %v1900, %v1892 : tensor<32x160x7x7xf32>
    %v1902 = stablehlo.rsqrt %v1901 : tensor<32x160x7x7xf32>
    %v1903 = stablehlo.multiply %v1896, %v1902 : tensor<32x160x7x7xf32>
    %v1904 = stablehlo.broadcast_in_dim %gp16, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1905 = stablehlo.multiply %v1904, %v1888 : tensor<32x160x7x7xf32>
    %v1906 = stablehlo.reduce(%v1905 init: %v1890) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<32x160xf32>
    %v1907 = stablehlo.broadcast_in_dim %v1906, dims = [0, 1] : (tensor<32x160xf32>) -> tensor<32x160x7x7xf32>
    %v1908 = stablehlo.multiply %v1903, %v1905 : tensor<32x160x7x7xf32>
    %v1909 = stablehlo.reduce(%v1908 init: %v1890) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<32x160xf32>
    %v1910 = stablehlo.broadcast_in_dim %v1909, dims = [0, 1] : (tensor<32x160xf32>) -> tensor<32x160x7x7xf32>
    %v1911 = stablehlo.multiply %v1905, %v1891 : tensor<32x160x7x7xf32>
    %v1912 = stablehlo.subtract %v1911, %v1907 : tensor<32x160x7x7xf32>
    %v1913 = stablehlo.multiply %v1903, %v1910 : tensor<32x160x7x7xf32>
    %v1914 = stablehlo.subtract %v1912, %v1913 : tensor<32x160x7x7xf32>
    %v1915 = stablehlo.divide %v1902, %v1891 : tensor<32x160x7x7xf32>
    %v1916 = stablehlo.multiply %v1915, %v1914 : tensor<32x160x7x7xf32>
    %v1917 = stablehlo.reshape %v1916 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v1918 = stablehlo.reshape %v1917 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v1919 = stablehlo.transpose %Wp16, dims = [1, 0, 2, 3] : (tensor<160x960x1x1xf32>) -> tensor<960x160x1x1xf32>
    %v1920 = stablehlo.reverse %v1919, dims = [2, 3] : tensor<960x160x1x1xf32>
    %v1921 = stablehlo.convolution(%v1918, %v1920)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x7x7xf32>, tensor<960x160x1x1xf32>) -> tensor<32x960x7x7xf32>
    %v1922 = stablehlo.reshape %v1921 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1923 = stablehlo.reshape %v1922 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1924 = stablehlo.reshape %v1396 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1925 = stablehlo.constant dense<0.0> : tensor<32x960x7x7xf32>
    %v1926 = stablehlo.constant dense<6.0> : tensor<32x960x7x7xf32>
    %v1927 = stablehlo.compare GT, %v1924, %v1925 : (tensor<32x960x7x7xf32>, tensor<32x960x7x7xf32>) -> tensor<32x960x7x7xi1>
    %v1928 = stablehlo.compare LT, %v1924, %v1926 : (tensor<32x960x7x7xf32>, tensor<32x960x7x7xf32>) -> tensor<32x960x7x7xi1>
    %v1929 = stablehlo.and %v1927, %v1928 : tensor<32x960x7x7xi1>
    %v1930 = stablehlo.select %v1929, %v1923, %v1925 : tensor<32x960x7x7xi1>, tensor<32x960x7x7xf32>
    %v1931 = stablehlo.reshape %v1930 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1932 = stablehlo.reshape %v1931 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1933 = stablehlo.reshape %v1376 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1934 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1935 = stablehlo.constant dense<49.0> : tensor<32x960x7x7xf32>
    %v1936 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v1937 = stablehlo.reduce(%v1933 init: %v1934) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1938 = stablehlo.broadcast_in_dim %v1937, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1939 = stablehlo.divide %v1938, %v1935 : tensor<32x960x7x7xf32>
    %v1940 = stablehlo.subtract %v1933, %v1939 : tensor<32x960x7x7xf32>
    %v1941 = stablehlo.multiply %v1940, %v1940 : tensor<32x960x7x7xf32>
    %v1942 = stablehlo.reduce(%v1941 init: %v1934) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1943 = stablehlo.broadcast_in_dim %v1942, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1944 = stablehlo.divide %v1943, %v1935 : tensor<32x960x7x7xf32>
    %v1945 = stablehlo.add %v1944, %v1936 : tensor<32x960x7x7xf32>
    %v1946 = stablehlo.rsqrt %v1945 : tensor<32x960x7x7xf32>
    %v1947 = stablehlo.multiply %v1940, %v1946 : tensor<32x960x7x7xf32>
    %v1948 = stablehlo.broadcast_in_dim %gd16, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1949 = stablehlo.multiply %v1948, %v1932 : tensor<32x960x7x7xf32>
    %v1950 = stablehlo.reduce(%v1949 init: %v1934) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1951 = stablehlo.broadcast_in_dim %v1950, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1952 = stablehlo.multiply %v1947, %v1949 : tensor<32x960x7x7xf32>
    %v1953 = stablehlo.reduce(%v1952 init: %v1934) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1954 = stablehlo.broadcast_in_dim %v1953, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1955 = stablehlo.multiply %v1949, %v1935 : tensor<32x960x7x7xf32>
    %v1956 = stablehlo.subtract %v1955, %v1951 : tensor<32x960x7x7xf32>
    %v1957 = stablehlo.multiply %v1947, %v1954 : tensor<32x960x7x7xf32>
    %v1958 = stablehlo.subtract %v1956, %v1957 : tensor<32x960x7x7xf32>
    %v1959 = stablehlo.divide %v1946, %v1935 : tensor<32x960x7x7xf32>
    %v1960 = stablehlo.multiply %v1959, %v1958 : tensor<32x960x7x7xf32>
    %v1961 = stablehlo.reshape %v1960 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1962 = stablehlo.reshape %v1961 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1963 = stablehlo.reverse %Wd16, dims = [2, 3] : tensor<960x1x3x3xf32>
    %v1964 = stablehlo.convolution(%v1962, %v1963)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 960 : i64} : (tensor<32x960x7x7xf32>, tensor<960x1x3x3xf32>) -> tensor<32x960x7x7xf32>
    %v1965 = stablehlo.reshape %v1964 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1966 = stablehlo.reshape %v1965 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1967 = stablehlo.reshape %v1365 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1968 = stablehlo.constant dense<0.0> : tensor<32x960x7x7xf32>
    %v1969 = stablehlo.constant dense<6.0> : tensor<32x960x7x7xf32>
    %v1970 = stablehlo.compare GT, %v1967, %v1968 : (tensor<32x960x7x7xf32>, tensor<32x960x7x7xf32>) -> tensor<32x960x7x7xi1>
    %v1971 = stablehlo.compare LT, %v1967, %v1969 : (tensor<32x960x7x7xf32>, tensor<32x960x7x7xf32>) -> tensor<32x960x7x7xi1>
    %v1972 = stablehlo.and %v1970, %v1971 : tensor<32x960x7x7xi1>
    %v1973 = stablehlo.select %v1972, %v1966, %v1968 : tensor<32x960x7x7xi1>, tensor<32x960x7x7xf32>
    %v1974 = stablehlo.reshape %v1973 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1975 = stablehlo.reshape %v1974 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1976 = stablehlo.reshape %v1345 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1977 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1978 = stablehlo.constant dense<49.0> : tensor<32x960x7x7xf32>
    %v1979 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v1980 = stablehlo.reduce(%v1976 init: %v1977) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1981 = stablehlo.broadcast_in_dim %v1980, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1982 = stablehlo.divide %v1981, %v1978 : tensor<32x960x7x7xf32>
    %v1983 = stablehlo.subtract %v1976, %v1982 : tensor<32x960x7x7xf32>
    %v1984 = stablehlo.multiply %v1983, %v1983 : tensor<32x960x7x7xf32>
    %v1985 = stablehlo.reduce(%v1984 init: %v1977) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1986 = stablehlo.broadcast_in_dim %v1985, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1987 = stablehlo.divide %v1986, %v1978 : tensor<32x960x7x7xf32>
    %v1988 = stablehlo.add %v1987, %v1979 : tensor<32x960x7x7xf32>
    %v1989 = stablehlo.rsqrt %v1988 : tensor<32x960x7x7xf32>
    %v1990 = stablehlo.multiply %v1983, %v1989 : tensor<32x960x7x7xf32>
    %v1991 = stablehlo.broadcast_in_dim %ge16, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1992 = stablehlo.multiply %v1991, %v1975 : tensor<32x960x7x7xf32>
    %v1993 = stablehlo.reduce(%v1992 init: %v1977) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1994 = stablehlo.broadcast_in_dim %v1993, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1995 = stablehlo.multiply %v1990, %v1992 : tensor<32x960x7x7xf32>
    %v1996 = stablehlo.reduce(%v1995 init: %v1977) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1997 = stablehlo.broadcast_in_dim %v1996, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1998 = stablehlo.multiply %v1992, %v1978 : tensor<32x960x7x7xf32>
    %v1999 = stablehlo.subtract %v1998, %v1994 : tensor<32x960x7x7xf32>
    %v2000 = stablehlo.multiply %v1990, %v1997 : tensor<32x960x7x7xf32>
    %v2001 = stablehlo.subtract %v1999, %v2000 : tensor<32x960x7x7xf32>
    %v2002 = stablehlo.divide %v1989, %v1978 : tensor<32x960x7x7xf32>
    %v2003 = stablehlo.multiply %v2002, %v2001 : tensor<32x960x7x7xf32>
    %v2004 = stablehlo.reshape %v2003 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v2005 = stablehlo.reshape %v2004 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2006 = stablehlo.transpose %We16, dims = [1, 0, 2, 3] : (tensor<960x160x1x1xf32>) -> tensor<160x960x1x1xf32>
    %v2007 = stablehlo.reverse %v2006, dims = [2, 3] : tensor<160x960x1x1xf32>
    %v2008 = stablehlo.convolution(%v2005, %v2007)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x960x7x7xf32>, tensor<160x960x1x1xf32>) -> tensor<32x160x7x7xf32>
    %v2009 = stablehlo.reshape %v2008 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v2010 = stablehlo.reshape %v2009 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2011 = stablehlo.reshape %v1779 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2012 = stablehlo.add %v2010, %v2011 : tensor<32x160x7x7xf32>
    %v2013 = stablehlo.reshape %v2012 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v2014 = stablehlo.reshape %v1340 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2015 = stablehlo.reshape %v2004 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2016 = stablehlo.transpose %v2014, dims = [1, 0, 2, 3] : (tensor<32x160x7x7xf32>) -> tensor<160x32x7x7xf32>
    %v2017 = stablehlo.transpose %v2015, dims = [1, 0, 2, 3] : (tensor<32x960x7x7xf32>) -> tensor<960x32x7x7xf32>
    %v2018 = stablehlo.convolution(%v2016, %v2017)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<160x32x7x7xf32>, tensor<960x32x7x7xf32>) -> tensor<160x960x1x1xf32>
    %v2019 = stablehlo.transpose %v2018, dims = [1, 0, 2, 3] : (tensor<160x960x1x1xf32>) -> tensor<960x160x1x1xf32>
    %v2020 = stablehlo.constant dense<0.3> : tensor<960x160x1x1xf32>
    %v2021 = stablehlo.multiply %v2019, %v2020 : tensor<960x160x1x1xf32>
    %v2022 = stablehlo.subtract %We16, %v2021 : tensor<960x160x1x1xf32>
    %v2023 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2024 = stablehlo.reshape %v1345 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2025 = stablehlo.constant dense<49.0> : tensor<32x960x7x7xf32>
    %v2026 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v2027 = stablehlo.reduce(%v2024 init: %v2023) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v2028 = stablehlo.broadcast_in_dim %v2027, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v2029 = stablehlo.divide %v2028, %v2025 : tensor<32x960x7x7xf32>
    %v2030 = stablehlo.subtract %v2024, %v2029 : tensor<32x960x7x7xf32>
    %v2031 = stablehlo.multiply %v2030, %v2030 : tensor<32x960x7x7xf32>
    %v2032 = stablehlo.reduce(%v2031 init: %v2023) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v2033 = stablehlo.broadcast_in_dim %v2032, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v2034 = stablehlo.divide %v2033, %v2025 : tensor<32x960x7x7xf32>
    %v2035 = stablehlo.add %v2034, %v2026 : tensor<32x960x7x7xf32>
    %v2036 = stablehlo.rsqrt %v2035 : tensor<32x960x7x7xf32>
    %v2037 = stablehlo.multiply %v2030, %v2036 : tensor<32x960x7x7xf32>
    %v2038 = stablehlo.reshape %v1974 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2039 = stablehlo.multiply %v2038, %v2037 : tensor<32x960x7x7xf32>
    %v2040 = stablehlo.reduce(%v2039 init: %v2023) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v2041 = stablehlo.constant dense<0.3> : tensor<960xf32>
    %v2042 = stablehlo.multiply %v2040, %v2041 : tensor<960xf32>
    %v2043 = stablehlo.subtract %ge16, %v2042 : tensor<960xf32>
    %v2044 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2045 = stablehlo.reshape %v1974 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2046 = stablehlo.reduce(%v2045 init: %v2044) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v2047 = stablehlo.constant dense<0.3> : tensor<960xf32>
    %v2048 = stablehlo.multiply %v2046, %v2047 : tensor<960xf32>
    %v2049 = stablehlo.subtract %bte16, %v2048 : tensor<960xf32>
    %v2050 = stablehlo.reshape %v1371 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2051 = stablehlo.reshape %v1961 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2052 = stablehlo.transpose %v2050, dims = [1, 0, 2, 3] : (tensor<32x960x7x7xf32>) -> tensor<960x32x7x7xf32>
    %v2053 = stablehlo.transpose %v2051, dims = [1, 0, 2, 3] : (tensor<32x960x7x7xf32>) -> tensor<960x32x7x7xf32>
    %v2054 = stablehlo.convolution(%v2052, %v2053)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 960 : i64, feature_group_count = 1 : i64} : (tensor<960x32x7x7xf32>, tensor<960x32x7x7xf32>) -> tensor<1x960x3x3xf32>
    %v2055 = stablehlo.reshape %v2054 : (tensor<1x960x3x3xf32>) -> tensor<960x1x3x3xf32>
    %v2056 = stablehlo.constant dense<0.3> : tensor<960x1x3x3xf32>
    %v2057 = stablehlo.multiply %v2055, %v2056 : tensor<960x1x3x3xf32>
    %v2058 = stablehlo.subtract %Wd16, %v2057 : tensor<960x1x3x3xf32>
    %v2059 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2060 = stablehlo.reshape %v1376 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2061 = stablehlo.constant dense<49.0> : tensor<32x960x7x7xf32>
    %v2062 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v2063 = stablehlo.reduce(%v2060 init: %v2059) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v2064 = stablehlo.broadcast_in_dim %v2063, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v2065 = stablehlo.divide %v2064, %v2061 : tensor<32x960x7x7xf32>
    %v2066 = stablehlo.subtract %v2060, %v2065 : tensor<32x960x7x7xf32>
    %v2067 = stablehlo.multiply %v2066, %v2066 : tensor<32x960x7x7xf32>
    %v2068 = stablehlo.reduce(%v2067 init: %v2059) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v2069 = stablehlo.broadcast_in_dim %v2068, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v2070 = stablehlo.divide %v2069, %v2061 : tensor<32x960x7x7xf32>
    %v2071 = stablehlo.add %v2070, %v2062 : tensor<32x960x7x7xf32>
    %v2072 = stablehlo.rsqrt %v2071 : tensor<32x960x7x7xf32>
    %v2073 = stablehlo.multiply %v2066, %v2072 : tensor<32x960x7x7xf32>
    %v2074 = stablehlo.reshape %v1931 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2075 = stablehlo.multiply %v2074, %v2073 : tensor<32x960x7x7xf32>
    %v2076 = stablehlo.reduce(%v2075 init: %v2059) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v2077 = stablehlo.constant dense<0.3> : tensor<960xf32>
    %v2078 = stablehlo.multiply %v2076, %v2077 : tensor<960xf32>
    %v2079 = stablehlo.subtract %gd16, %v2078 : tensor<960xf32>
    %v2080 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2081 = stablehlo.reshape %v1931 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2082 = stablehlo.reduce(%v2081 init: %v2080) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v2083 = stablehlo.constant dense<0.3> : tensor<960xf32>
    %v2084 = stablehlo.multiply %v2082, %v2083 : tensor<960xf32>
    %v2085 = stablehlo.subtract %btd16, %v2084 : tensor<960xf32>
    %v2086 = stablehlo.reshape %v1402 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2087 = stablehlo.reshape %v1917 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2088 = stablehlo.transpose %v2086, dims = [1, 0, 2, 3] : (tensor<32x960x7x7xf32>) -> tensor<960x32x7x7xf32>
    %v2089 = stablehlo.transpose %v2087, dims = [1, 0, 2, 3] : (tensor<32x160x7x7xf32>) -> tensor<160x32x7x7xf32>
    %v2090 = stablehlo.convolution(%v2088, %v2089)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<960x32x7x7xf32>, tensor<160x32x7x7xf32>) -> tensor<960x160x1x1xf32>
    %v2091 = stablehlo.transpose %v2090, dims = [1, 0, 2, 3] : (tensor<960x160x1x1xf32>) -> tensor<160x960x1x1xf32>
    %v2092 = stablehlo.constant dense<0.3> : tensor<160x960x1x1xf32>
    %v2093 = stablehlo.multiply %v2091, %v2092 : tensor<160x960x1x1xf32>
    %v2094 = stablehlo.subtract %Wp16, %v2093 : tensor<160x960x1x1xf32>
    %v2095 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2096 = stablehlo.reshape %v1407 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2097 = stablehlo.constant dense<49.0> : tensor<32x160x7x7xf32>
    %v2098 = stablehlo.constant dense<1.0e-5> : tensor<32x160x7x7xf32>
    %v2099 = stablehlo.reduce(%v2096 init: %v2095) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<32x160xf32>
    %v2100 = stablehlo.broadcast_in_dim %v2099, dims = [0, 1] : (tensor<32x160xf32>) -> tensor<32x160x7x7xf32>
    %v2101 = stablehlo.divide %v2100, %v2097 : tensor<32x160x7x7xf32>
    %v2102 = stablehlo.subtract %v2096, %v2101 : tensor<32x160x7x7xf32>
    %v2103 = stablehlo.multiply %v2102, %v2102 : tensor<32x160x7x7xf32>
    %v2104 = stablehlo.reduce(%v2103 init: %v2095) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<32x160xf32>
    %v2105 = stablehlo.broadcast_in_dim %v2104, dims = [0, 1] : (tensor<32x160xf32>) -> tensor<32x160x7x7xf32>
    %v2106 = stablehlo.divide %v2105, %v2097 : tensor<32x160x7x7xf32>
    %v2107 = stablehlo.add %v2106, %v2098 : tensor<32x160x7x7xf32>
    %v2108 = stablehlo.rsqrt %v2107 : tensor<32x160x7x7xf32>
    %v2109 = stablehlo.multiply %v2102, %v2108 : tensor<32x160x7x7xf32>
    %v2110 = stablehlo.reshape %v1779 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2111 = stablehlo.multiply %v2110, %v2109 : tensor<32x160x7x7xf32>
    %v2112 = stablehlo.reduce(%v2111 init: %v2095) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v2113 = stablehlo.constant dense<0.3> : tensor<160xf32>
    %v2114 = stablehlo.multiply %v2112, %v2113 : tensor<160xf32>
    %v2115 = stablehlo.subtract %gp16, %v2114 : tensor<160xf32>
    %v2116 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2117 = stablehlo.reshape %v1779 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2118 = stablehlo.reduce(%v2117 init: %v2116) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v2119 = stablehlo.constant dense<0.3> : tensor<160xf32>
    %v2120 = stablehlo.multiply %v2118, %v2119 : tensor<160xf32>
    %v2121 = stablehlo.subtract %btp16, %v2120 : tensor<160xf32>
    %v2122 = stablehlo.reshape %v2013 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2123 = stablehlo.reshape %v1316 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2124 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2125 = stablehlo.constant dense<49.0> : tensor<32x160x7x7xf32>
    %v2126 = stablehlo.constant dense<1.0e-5> : tensor<32x160x7x7xf32>
    %v2127 = stablehlo.reduce(%v2123 init: %v2124) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<32x160xf32>
    %v2128 = stablehlo.broadcast_in_dim %v2127, dims = [0, 1] : (tensor<32x160xf32>) -> tensor<32x160x7x7xf32>
    %v2129 = stablehlo.divide %v2128, %v2125 : tensor<32x160x7x7xf32>
    %v2130 = stablehlo.subtract %v2123, %v2129 : tensor<32x160x7x7xf32>
    %v2131 = stablehlo.multiply %v2130, %v2130 : tensor<32x160x7x7xf32>
    %v2132 = stablehlo.reduce(%v2131 init: %v2124) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<32x160xf32>
    %v2133 = stablehlo.broadcast_in_dim %v2132, dims = [0, 1] : (tensor<32x160xf32>) -> tensor<32x160x7x7xf32>
    %v2134 = stablehlo.divide %v2133, %v2125 : tensor<32x160x7x7xf32>
    %v2135 = stablehlo.add %v2134, %v2126 : tensor<32x160x7x7xf32>
    %v2136 = stablehlo.rsqrt %v2135 : tensor<32x160x7x7xf32>
    %v2137 = stablehlo.multiply %v2130, %v2136 : tensor<32x160x7x7xf32>
    %v2138 = stablehlo.broadcast_in_dim %gp15, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v2139 = stablehlo.multiply %v2138, %v2122 : tensor<32x160x7x7xf32>
    %v2140 = stablehlo.reduce(%v2139 init: %v2124) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<32x160xf32>
    %v2141 = stablehlo.broadcast_in_dim %v2140, dims = [0, 1] : (tensor<32x160xf32>) -> tensor<32x160x7x7xf32>
    %v2142 = stablehlo.multiply %v2137, %v2139 : tensor<32x160x7x7xf32>
    %v2143 = stablehlo.reduce(%v2142 init: %v2124) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<32x160xf32>
    %v2144 = stablehlo.broadcast_in_dim %v2143, dims = [0, 1] : (tensor<32x160xf32>) -> tensor<32x160x7x7xf32>
    %v2145 = stablehlo.multiply %v2139, %v2125 : tensor<32x160x7x7xf32>
    %v2146 = stablehlo.subtract %v2145, %v2141 : tensor<32x160x7x7xf32>
    %v2147 = stablehlo.multiply %v2137, %v2144 : tensor<32x160x7x7xf32>
    %v2148 = stablehlo.subtract %v2146, %v2147 : tensor<32x160x7x7xf32>
    %v2149 = stablehlo.divide %v2136, %v2125 : tensor<32x160x7x7xf32>
    %v2150 = stablehlo.multiply %v2149, %v2148 : tensor<32x160x7x7xf32>
    %v2151 = stablehlo.reshape %v2150 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v2152 = stablehlo.reshape %v2151 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2153 = stablehlo.transpose %Wp15, dims = [1, 0, 2, 3] : (tensor<160x960x1x1xf32>) -> tensor<960x160x1x1xf32>
    %v2154 = stablehlo.reverse %v2153, dims = [2, 3] : tensor<960x160x1x1xf32>
    %v2155 = stablehlo.convolution(%v2152, %v2154)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x7x7xf32>, tensor<960x160x1x1xf32>) -> tensor<32x960x7x7xf32>
    %v2156 = stablehlo.reshape %v2155 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v2157 = stablehlo.reshape %v2156 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2158 = stablehlo.reshape %v1305 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2159 = stablehlo.constant dense<0.0> : tensor<32x960x7x7xf32>
    %v2160 = stablehlo.constant dense<6.0> : tensor<32x960x7x7xf32>
    %v2161 = stablehlo.compare GT, %v2158, %v2159 : (tensor<32x960x7x7xf32>, tensor<32x960x7x7xf32>) -> tensor<32x960x7x7xi1>
    %v2162 = stablehlo.compare LT, %v2158, %v2160 : (tensor<32x960x7x7xf32>, tensor<32x960x7x7xf32>) -> tensor<32x960x7x7xi1>
    %v2163 = stablehlo.and %v2161, %v2162 : tensor<32x960x7x7xi1>
    %v2164 = stablehlo.select %v2163, %v2157, %v2159 : tensor<32x960x7x7xi1>, tensor<32x960x7x7xf32>
    %v2165 = stablehlo.reshape %v2164 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v2166 = stablehlo.reshape %v2165 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2167 = stablehlo.reshape %v1285 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2168 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2169 = stablehlo.constant dense<49.0> : tensor<32x960x7x7xf32>
    %v2170 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v2171 = stablehlo.reduce(%v2167 init: %v2168) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v2172 = stablehlo.broadcast_in_dim %v2171, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v2173 = stablehlo.divide %v2172, %v2169 : tensor<32x960x7x7xf32>
    %v2174 = stablehlo.subtract %v2167, %v2173 : tensor<32x960x7x7xf32>
    %v2175 = stablehlo.multiply %v2174, %v2174 : tensor<32x960x7x7xf32>
    %v2176 = stablehlo.reduce(%v2175 init: %v2168) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v2177 = stablehlo.broadcast_in_dim %v2176, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v2178 = stablehlo.divide %v2177, %v2169 : tensor<32x960x7x7xf32>
    %v2179 = stablehlo.add %v2178, %v2170 : tensor<32x960x7x7xf32>
    %v2180 = stablehlo.rsqrt %v2179 : tensor<32x960x7x7xf32>
    %v2181 = stablehlo.multiply %v2174, %v2180 : tensor<32x960x7x7xf32>
    %v2182 = stablehlo.broadcast_in_dim %gd15, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v2183 = stablehlo.multiply %v2182, %v2166 : tensor<32x960x7x7xf32>
    %v2184 = stablehlo.reduce(%v2183 init: %v2168) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v2185 = stablehlo.broadcast_in_dim %v2184, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v2186 = stablehlo.multiply %v2181, %v2183 : tensor<32x960x7x7xf32>
    %v2187 = stablehlo.reduce(%v2186 init: %v2168) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v2188 = stablehlo.broadcast_in_dim %v2187, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v2189 = stablehlo.multiply %v2183, %v2169 : tensor<32x960x7x7xf32>
    %v2190 = stablehlo.subtract %v2189, %v2185 : tensor<32x960x7x7xf32>
    %v2191 = stablehlo.multiply %v2181, %v2188 : tensor<32x960x7x7xf32>
    %v2192 = stablehlo.subtract %v2190, %v2191 : tensor<32x960x7x7xf32>
    %v2193 = stablehlo.divide %v2180, %v2169 : tensor<32x960x7x7xf32>
    %v2194 = stablehlo.multiply %v2193, %v2192 : tensor<32x960x7x7xf32>
    %v2195 = stablehlo.reshape %v2194 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v2196 = stablehlo.reshape %v2195 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2197 = stablehlo.reverse %Wd15, dims = [2, 3] : tensor<960x1x3x3xf32>
    %v2198 = stablehlo.convolution(%v2196, %v2197)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 960 : i64} : (tensor<32x960x7x7xf32>, tensor<960x1x3x3xf32>) -> tensor<32x960x7x7xf32>
    %v2199 = stablehlo.reshape %v2198 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v2200 = stablehlo.reshape %v2199 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2201 = stablehlo.reshape %v1274 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2202 = stablehlo.constant dense<0.0> : tensor<32x960x7x7xf32>
    %v2203 = stablehlo.constant dense<6.0> : tensor<32x960x7x7xf32>
    %v2204 = stablehlo.compare GT, %v2201, %v2202 : (tensor<32x960x7x7xf32>, tensor<32x960x7x7xf32>) -> tensor<32x960x7x7xi1>
    %v2205 = stablehlo.compare LT, %v2201, %v2203 : (tensor<32x960x7x7xf32>, tensor<32x960x7x7xf32>) -> tensor<32x960x7x7xi1>
    %v2206 = stablehlo.and %v2204, %v2205 : tensor<32x960x7x7xi1>
    %v2207 = stablehlo.select %v2206, %v2200, %v2202 : tensor<32x960x7x7xi1>, tensor<32x960x7x7xf32>
    %v2208 = stablehlo.reshape %v2207 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v2209 = stablehlo.reshape %v2208 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2210 = stablehlo.reshape %v1254 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2211 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2212 = stablehlo.constant dense<49.0> : tensor<32x960x7x7xf32>
    %v2213 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v2214 = stablehlo.reduce(%v2210 init: %v2211) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v2215 = stablehlo.broadcast_in_dim %v2214, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v2216 = stablehlo.divide %v2215, %v2212 : tensor<32x960x7x7xf32>
    %v2217 = stablehlo.subtract %v2210, %v2216 : tensor<32x960x7x7xf32>
    %v2218 = stablehlo.multiply %v2217, %v2217 : tensor<32x960x7x7xf32>
    %v2219 = stablehlo.reduce(%v2218 init: %v2211) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v2220 = stablehlo.broadcast_in_dim %v2219, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v2221 = stablehlo.divide %v2220, %v2212 : tensor<32x960x7x7xf32>
    %v2222 = stablehlo.add %v2221, %v2213 : tensor<32x960x7x7xf32>
    %v2223 = stablehlo.rsqrt %v2222 : tensor<32x960x7x7xf32>
    %v2224 = stablehlo.multiply %v2217, %v2223 : tensor<32x960x7x7xf32>
    %v2225 = stablehlo.broadcast_in_dim %ge15, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v2226 = stablehlo.multiply %v2225, %v2209 : tensor<32x960x7x7xf32>
    %v2227 = stablehlo.reduce(%v2226 init: %v2211) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v2228 = stablehlo.broadcast_in_dim %v2227, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v2229 = stablehlo.multiply %v2224, %v2226 : tensor<32x960x7x7xf32>
    %v2230 = stablehlo.reduce(%v2229 init: %v2211) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v2231 = stablehlo.broadcast_in_dim %v2230, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v2232 = stablehlo.multiply %v2226, %v2212 : tensor<32x960x7x7xf32>
    %v2233 = stablehlo.subtract %v2232, %v2228 : tensor<32x960x7x7xf32>
    %v2234 = stablehlo.multiply %v2224, %v2231 : tensor<32x960x7x7xf32>
    %v2235 = stablehlo.subtract %v2233, %v2234 : tensor<32x960x7x7xf32>
    %v2236 = stablehlo.divide %v2223, %v2212 : tensor<32x960x7x7xf32>
    %v2237 = stablehlo.multiply %v2236, %v2235 : tensor<32x960x7x7xf32>
    %v2238 = stablehlo.reshape %v2237 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v2239 = stablehlo.reshape %v2238 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2240 = stablehlo.transpose %We15, dims = [1, 0, 2, 3] : (tensor<960x160x1x1xf32>) -> tensor<160x960x1x1xf32>
    %v2241 = stablehlo.reverse %v2240, dims = [2, 3] : tensor<160x960x1x1xf32>
    %v2242 = stablehlo.convolution(%v2239, %v2241)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x960x7x7xf32>, tensor<160x960x1x1xf32>) -> tensor<32x160x7x7xf32>
    %v2243 = stablehlo.reshape %v2242 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v2244 = stablehlo.reshape %v2243 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2245 = stablehlo.reshape %v2013 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2246 = stablehlo.add %v2244, %v2245 : tensor<32x160x7x7xf32>
    %v2247 = stablehlo.reshape %v2246 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v2248 = stablehlo.reshape %v1249 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2249 = stablehlo.reshape %v2238 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2250 = stablehlo.transpose %v2248, dims = [1, 0, 2, 3] : (tensor<32x160x7x7xf32>) -> tensor<160x32x7x7xf32>
    %v2251 = stablehlo.transpose %v2249, dims = [1, 0, 2, 3] : (tensor<32x960x7x7xf32>) -> tensor<960x32x7x7xf32>
    %v2252 = stablehlo.convolution(%v2250, %v2251)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<160x32x7x7xf32>, tensor<960x32x7x7xf32>) -> tensor<160x960x1x1xf32>
    %v2253 = stablehlo.transpose %v2252, dims = [1, 0, 2, 3] : (tensor<160x960x1x1xf32>) -> tensor<960x160x1x1xf32>
    %v2254 = stablehlo.constant dense<0.3> : tensor<960x160x1x1xf32>
    %v2255 = stablehlo.multiply %v2253, %v2254 : tensor<960x160x1x1xf32>
    %v2256 = stablehlo.subtract %We15, %v2255 : tensor<960x160x1x1xf32>
    %v2257 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2258 = stablehlo.reshape %v1254 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2259 = stablehlo.constant dense<49.0> : tensor<32x960x7x7xf32>
    %v2260 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v2261 = stablehlo.reduce(%v2258 init: %v2257) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v2262 = stablehlo.broadcast_in_dim %v2261, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v2263 = stablehlo.divide %v2262, %v2259 : tensor<32x960x7x7xf32>
    %v2264 = stablehlo.subtract %v2258, %v2263 : tensor<32x960x7x7xf32>
    %v2265 = stablehlo.multiply %v2264, %v2264 : tensor<32x960x7x7xf32>
    %v2266 = stablehlo.reduce(%v2265 init: %v2257) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v2267 = stablehlo.broadcast_in_dim %v2266, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v2268 = stablehlo.divide %v2267, %v2259 : tensor<32x960x7x7xf32>
    %v2269 = stablehlo.add %v2268, %v2260 : tensor<32x960x7x7xf32>
    %v2270 = stablehlo.rsqrt %v2269 : tensor<32x960x7x7xf32>
    %v2271 = stablehlo.multiply %v2264, %v2270 : tensor<32x960x7x7xf32>
    %v2272 = stablehlo.reshape %v2208 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2273 = stablehlo.multiply %v2272, %v2271 : tensor<32x960x7x7xf32>
    %v2274 = stablehlo.reduce(%v2273 init: %v2257) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v2275 = stablehlo.constant dense<0.3> : tensor<960xf32>
    %v2276 = stablehlo.multiply %v2274, %v2275 : tensor<960xf32>
    %v2277 = stablehlo.subtract %ge15, %v2276 : tensor<960xf32>
    %v2278 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2279 = stablehlo.reshape %v2208 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2280 = stablehlo.reduce(%v2279 init: %v2278) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v2281 = stablehlo.constant dense<0.3> : tensor<960xf32>
    %v2282 = stablehlo.multiply %v2280, %v2281 : tensor<960xf32>
    %v2283 = stablehlo.subtract %bte15, %v2282 : tensor<960xf32>
    %v2284 = stablehlo.reshape %v1280 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2285 = stablehlo.reshape %v2195 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2286 = stablehlo.transpose %v2284, dims = [1, 0, 2, 3] : (tensor<32x960x7x7xf32>) -> tensor<960x32x7x7xf32>
    %v2287 = stablehlo.transpose %v2285, dims = [1, 0, 2, 3] : (tensor<32x960x7x7xf32>) -> tensor<960x32x7x7xf32>
    %v2288 = stablehlo.convolution(%v2286, %v2287)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 960 : i64, feature_group_count = 1 : i64} : (tensor<960x32x7x7xf32>, tensor<960x32x7x7xf32>) -> tensor<1x960x3x3xf32>
    %v2289 = stablehlo.reshape %v2288 : (tensor<1x960x3x3xf32>) -> tensor<960x1x3x3xf32>
    %v2290 = stablehlo.constant dense<0.3> : tensor<960x1x3x3xf32>
    %v2291 = stablehlo.multiply %v2289, %v2290 : tensor<960x1x3x3xf32>
    %v2292 = stablehlo.subtract %Wd15, %v2291 : tensor<960x1x3x3xf32>
    %v2293 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2294 = stablehlo.reshape %v1285 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2295 = stablehlo.constant dense<49.0> : tensor<32x960x7x7xf32>
    %v2296 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v2297 = stablehlo.reduce(%v2294 init: %v2293) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v2298 = stablehlo.broadcast_in_dim %v2297, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v2299 = stablehlo.divide %v2298, %v2295 : tensor<32x960x7x7xf32>
    %v2300 = stablehlo.subtract %v2294, %v2299 : tensor<32x960x7x7xf32>
    %v2301 = stablehlo.multiply %v2300, %v2300 : tensor<32x960x7x7xf32>
    %v2302 = stablehlo.reduce(%v2301 init: %v2293) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v2303 = stablehlo.broadcast_in_dim %v2302, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v2304 = stablehlo.divide %v2303, %v2295 : tensor<32x960x7x7xf32>
    %v2305 = stablehlo.add %v2304, %v2296 : tensor<32x960x7x7xf32>
    %v2306 = stablehlo.rsqrt %v2305 : tensor<32x960x7x7xf32>
    %v2307 = stablehlo.multiply %v2300, %v2306 : tensor<32x960x7x7xf32>
    %v2308 = stablehlo.reshape %v2165 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2309 = stablehlo.multiply %v2308, %v2307 : tensor<32x960x7x7xf32>
    %v2310 = stablehlo.reduce(%v2309 init: %v2293) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v2311 = stablehlo.constant dense<0.3> : tensor<960xf32>
    %v2312 = stablehlo.multiply %v2310, %v2311 : tensor<960xf32>
    %v2313 = stablehlo.subtract %gd15, %v2312 : tensor<960xf32>
    %v2314 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2315 = stablehlo.reshape %v2165 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2316 = stablehlo.reduce(%v2315 init: %v2314) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v2317 = stablehlo.constant dense<0.3> : tensor<960xf32>
    %v2318 = stablehlo.multiply %v2316, %v2317 : tensor<960xf32>
    %v2319 = stablehlo.subtract %btd15, %v2318 : tensor<960xf32>
    %v2320 = stablehlo.reshape %v1311 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2321 = stablehlo.reshape %v2151 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2322 = stablehlo.transpose %v2320, dims = [1, 0, 2, 3] : (tensor<32x960x7x7xf32>) -> tensor<960x32x7x7xf32>
    %v2323 = stablehlo.transpose %v2321, dims = [1, 0, 2, 3] : (tensor<32x160x7x7xf32>) -> tensor<160x32x7x7xf32>
    %v2324 = stablehlo.convolution(%v2322, %v2323)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<960x32x7x7xf32>, tensor<160x32x7x7xf32>) -> tensor<960x160x1x1xf32>
    %v2325 = stablehlo.transpose %v2324, dims = [1, 0, 2, 3] : (tensor<960x160x1x1xf32>) -> tensor<160x960x1x1xf32>
    %v2326 = stablehlo.constant dense<0.3> : tensor<160x960x1x1xf32>
    %v2327 = stablehlo.multiply %v2325, %v2326 : tensor<160x960x1x1xf32>
    %v2328 = stablehlo.subtract %Wp15, %v2327 : tensor<160x960x1x1xf32>
    %v2329 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2330 = stablehlo.reshape %v1316 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2331 = stablehlo.constant dense<49.0> : tensor<32x160x7x7xf32>
    %v2332 = stablehlo.constant dense<1.0e-5> : tensor<32x160x7x7xf32>
    %v2333 = stablehlo.reduce(%v2330 init: %v2329) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<32x160xf32>
    %v2334 = stablehlo.broadcast_in_dim %v2333, dims = [0, 1] : (tensor<32x160xf32>) -> tensor<32x160x7x7xf32>
    %v2335 = stablehlo.divide %v2334, %v2331 : tensor<32x160x7x7xf32>
    %v2336 = stablehlo.subtract %v2330, %v2335 : tensor<32x160x7x7xf32>
    %v2337 = stablehlo.multiply %v2336, %v2336 : tensor<32x160x7x7xf32>
    %v2338 = stablehlo.reduce(%v2337 init: %v2329) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<32x160xf32>
    %v2339 = stablehlo.broadcast_in_dim %v2338, dims = [0, 1] : (tensor<32x160xf32>) -> tensor<32x160x7x7xf32>
    %v2340 = stablehlo.divide %v2339, %v2331 : tensor<32x160x7x7xf32>
    %v2341 = stablehlo.add %v2340, %v2332 : tensor<32x160x7x7xf32>
    %v2342 = stablehlo.rsqrt %v2341 : tensor<32x160x7x7xf32>
    %v2343 = stablehlo.multiply %v2336, %v2342 : tensor<32x160x7x7xf32>
    %v2344 = stablehlo.reshape %v2013 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2345 = stablehlo.multiply %v2344, %v2343 : tensor<32x160x7x7xf32>
    %v2346 = stablehlo.reduce(%v2345 init: %v2329) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v2347 = stablehlo.constant dense<0.3> : tensor<160xf32>
    %v2348 = stablehlo.multiply %v2346, %v2347 : tensor<160xf32>
    %v2349 = stablehlo.subtract %gp15, %v2348 : tensor<160xf32>
    %v2350 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2351 = stablehlo.reshape %v2013 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2352 = stablehlo.reduce(%v2351 init: %v2350) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v2353 = stablehlo.constant dense<0.3> : tensor<160xf32>
    %v2354 = stablehlo.multiply %v2352, %v2353 : tensor<160xf32>
    %v2355 = stablehlo.subtract %btp15, %v2354 : tensor<160xf32>
    %v2356 = stablehlo.reshape %v2247 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2357 = stablehlo.reshape %v1229 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2358 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2359 = stablehlo.constant dense<49.0> : tensor<32x160x7x7xf32>
    %v2360 = stablehlo.constant dense<1.0e-5> : tensor<32x160x7x7xf32>
    %v2361 = stablehlo.reduce(%v2357 init: %v2358) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<32x160xf32>
    %v2362 = stablehlo.broadcast_in_dim %v2361, dims = [0, 1] : (tensor<32x160xf32>) -> tensor<32x160x7x7xf32>
    %v2363 = stablehlo.divide %v2362, %v2359 : tensor<32x160x7x7xf32>
    %v2364 = stablehlo.subtract %v2357, %v2363 : tensor<32x160x7x7xf32>
    %v2365 = stablehlo.multiply %v2364, %v2364 : tensor<32x160x7x7xf32>
    %v2366 = stablehlo.reduce(%v2365 init: %v2358) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<32x160xf32>
    %v2367 = stablehlo.broadcast_in_dim %v2366, dims = [0, 1] : (tensor<32x160xf32>) -> tensor<32x160x7x7xf32>
    %v2368 = stablehlo.divide %v2367, %v2359 : tensor<32x160x7x7xf32>
    %v2369 = stablehlo.add %v2368, %v2360 : tensor<32x160x7x7xf32>
    %v2370 = stablehlo.rsqrt %v2369 : tensor<32x160x7x7xf32>
    %v2371 = stablehlo.multiply %v2364, %v2370 : tensor<32x160x7x7xf32>
    %v2372 = stablehlo.broadcast_in_dim %gp14, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v2373 = stablehlo.multiply %v2372, %v2356 : tensor<32x160x7x7xf32>
    %v2374 = stablehlo.reduce(%v2373 init: %v2358) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<32x160xf32>
    %v2375 = stablehlo.broadcast_in_dim %v2374, dims = [0, 1] : (tensor<32x160xf32>) -> tensor<32x160x7x7xf32>
    %v2376 = stablehlo.multiply %v2371, %v2373 : tensor<32x160x7x7xf32>
    %v2377 = stablehlo.reduce(%v2376 init: %v2358) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<32x160xf32>
    %v2378 = stablehlo.broadcast_in_dim %v2377, dims = [0, 1] : (tensor<32x160xf32>) -> tensor<32x160x7x7xf32>
    %v2379 = stablehlo.multiply %v2373, %v2359 : tensor<32x160x7x7xf32>
    %v2380 = stablehlo.subtract %v2379, %v2375 : tensor<32x160x7x7xf32>
    %v2381 = stablehlo.multiply %v2371, %v2378 : tensor<32x160x7x7xf32>
    %v2382 = stablehlo.subtract %v2380, %v2381 : tensor<32x160x7x7xf32>
    %v2383 = stablehlo.divide %v2370, %v2359 : tensor<32x160x7x7xf32>
    %v2384 = stablehlo.multiply %v2383, %v2382 : tensor<32x160x7x7xf32>
    %v2385 = stablehlo.reshape %v2384 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v2386 = stablehlo.reshape %v2385 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2387 = stablehlo.transpose %Wp14, dims = [1, 0, 2, 3] : (tensor<160x576x1x1xf32>) -> tensor<576x160x1x1xf32>
    %v2388 = stablehlo.reverse %v2387, dims = [2, 3] : tensor<576x160x1x1xf32>
    %v2389 = stablehlo.convolution(%v2386, %v2388)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x7x7xf32>, tensor<576x160x1x1xf32>) -> tensor<32x576x7x7xf32>
    %v2390 = stablehlo.reshape %v2389 : (tensor<32x576x7x7xf32>) -> tensor<32x28224xf32>
    %v2391 = stablehlo.reshape %v2390 : (tensor<32x28224xf32>) -> tensor<32x576x7x7xf32>
    %v2392 = stablehlo.reshape %v1218 : (tensor<32x28224xf32>) -> tensor<32x576x7x7xf32>
    %v2393 = stablehlo.constant dense<0.0> : tensor<32x576x7x7xf32>
    %v2394 = stablehlo.constant dense<6.0> : tensor<32x576x7x7xf32>
    %v2395 = stablehlo.compare GT, %v2392, %v2393 : (tensor<32x576x7x7xf32>, tensor<32x576x7x7xf32>) -> tensor<32x576x7x7xi1>
    %v2396 = stablehlo.compare LT, %v2392, %v2394 : (tensor<32x576x7x7xf32>, tensor<32x576x7x7xf32>) -> tensor<32x576x7x7xi1>
    %v2397 = stablehlo.and %v2395, %v2396 : tensor<32x576x7x7xi1>
    %v2398 = stablehlo.select %v2397, %v2391, %v2393 : tensor<32x576x7x7xi1>, tensor<32x576x7x7xf32>
    %v2399 = stablehlo.reshape %v2398 : (tensor<32x576x7x7xf32>) -> tensor<32x28224xf32>
    %v2400 = stablehlo.reshape %v2399 : (tensor<32x28224xf32>) -> tensor<32x576x7x7xf32>
    %v2401 = stablehlo.reshape %v1198 : (tensor<32x28224xf32>) -> tensor<32x576x7x7xf32>
    %v2402 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2403 = stablehlo.constant dense<49.0> : tensor<32x576x7x7xf32>
    %v2404 = stablehlo.constant dense<1.0e-5> : tensor<32x576x7x7xf32>
    %v2405 = stablehlo.reduce(%v2401 init: %v2402) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x7x7xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2406 = stablehlo.broadcast_in_dim %v2405, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x7x7xf32>
    %v2407 = stablehlo.divide %v2406, %v2403 : tensor<32x576x7x7xf32>
    %v2408 = stablehlo.subtract %v2401, %v2407 : tensor<32x576x7x7xf32>
    %v2409 = stablehlo.multiply %v2408, %v2408 : tensor<32x576x7x7xf32>
    %v2410 = stablehlo.reduce(%v2409 init: %v2402) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x7x7xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2411 = stablehlo.broadcast_in_dim %v2410, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x7x7xf32>
    %v2412 = stablehlo.divide %v2411, %v2403 : tensor<32x576x7x7xf32>
    %v2413 = stablehlo.add %v2412, %v2404 : tensor<32x576x7x7xf32>
    %v2414 = stablehlo.rsqrt %v2413 : tensor<32x576x7x7xf32>
    %v2415 = stablehlo.multiply %v2408, %v2414 : tensor<32x576x7x7xf32>
    %v2416 = stablehlo.broadcast_in_dim %gd14, dims = [1] : (tensor<576xf32>) -> tensor<32x576x7x7xf32>
    %v2417 = stablehlo.multiply %v2416, %v2400 : tensor<32x576x7x7xf32>
    %v2418 = stablehlo.reduce(%v2417 init: %v2402) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x7x7xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2419 = stablehlo.broadcast_in_dim %v2418, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x7x7xf32>
    %v2420 = stablehlo.multiply %v2415, %v2417 : tensor<32x576x7x7xf32>
    %v2421 = stablehlo.reduce(%v2420 init: %v2402) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x7x7xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2422 = stablehlo.broadcast_in_dim %v2421, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x7x7xf32>
    %v2423 = stablehlo.multiply %v2417, %v2403 : tensor<32x576x7x7xf32>
    %v2424 = stablehlo.subtract %v2423, %v2419 : tensor<32x576x7x7xf32>
    %v2425 = stablehlo.multiply %v2415, %v2422 : tensor<32x576x7x7xf32>
    %v2426 = stablehlo.subtract %v2424, %v2425 : tensor<32x576x7x7xf32>
    %v2427 = stablehlo.divide %v2414, %v2403 : tensor<32x576x7x7xf32>
    %v2428 = stablehlo.multiply %v2427, %v2426 : tensor<32x576x7x7xf32>
    %v2429 = stablehlo.reshape %v2428 : (tensor<32x576x7x7xf32>) -> tensor<32x28224xf32>
    %v2430 = stablehlo.reshape %v2429 : (tensor<32x28224xf32>) -> tensor<32x576x7x7xf32>
    %v2431 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2432 = stablehlo.pad %v2430, %v2431, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x576x7x7xf32>, tensor<f32>) -> tensor<32x576x14x14xf32>
    %v2433 = stablehlo.reverse %Wd14, dims = [2, 3] : tensor<576x1x3x3xf32>
    %v2434 = stablehlo.convolution(%v2432, %v2433)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 576 : i64} : (tensor<32x576x14x14xf32>, tensor<576x1x3x3xf32>) -> tensor<32x576x14x14xf32>
    %v2435 = stablehlo.reshape %v2434 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v2436 = stablehlo.reshape %v2435 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2437 = stablehlo.reshape %v1187 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2438 = stablehlo.constant dense<0.0> : tensor<32x576x14x14xf32>
    %v2439 = stablehlo.constant dense<6.0> : tensor<32x576x14x14xf32>
    %v2440 = stablehlo.compare GT, %v2437, %v2438 : (tensor<32x576x14x14xf32>, tensor<32x576x14x14xf32>) -> tensor<32x576x14x14xi1>
    %v2441 = stablehlo.compare LT, %v2437, %v2439 : (tensor<32x576x14x14xf32>, tensor<32x576x14x14xf32>) -> tensor<32x576x14x14xi1>
    %v2442 = stablehlo.and %v2440, %v2441 : tensor<32x576x14x14xi1>
    %v2443 = stablehlo.select %v2442, %v2436, %v2438 : tensor<32x576x14x14xi1>, tensor<32x576x14x14xf32>
    %v2444 = stablehlo.reshape %v2443 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v2445 = stablehlo.reshape %v2444 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2446 = stablehlo.reshape %v1167 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2447 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2448 = stablehlo.constant dense<196.0> : tensor<32x576x14x14xf32>
    %v2449 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v2450 = stablehlo.reduce(%v2446 init: %v2447) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2451 = stablehlo.broadcast_in_dim %v2450, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2452 = stablehlo.divide %v2451, %v2448 : tensor<32x576x14x14xf32>
    %v2453 = stablehlo.subtract %v2446, %v2452 : tensor<32x576x14x14xf32>
    %v2454 = stablehlo.multiply %v2453, %v2453 : tensor<32x576x14x14xf32>
    %v2455 = stablehlo.reduce(%v2454 init: %v2447) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2456 = stablehlo.broadcast_in_dim %v2455, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2457 = stablehlo.divide %v2456, %v2448 : tensor<32x576x14x14xf32>
    %v2458 = stablehlo.add %v2457, %v2449 : tensor<32x576x14x14xf32>
    %v2459 = stablehlo.rsqrt %v2458 : tensor<32x576x14x14xf32>
    %v2460 = stablehlo.multiply %v2453, %v2459 : tensor<32x576x14x14xf32>
    %v2461 = stablehlo.broadcast_in_dim %ge14, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2462 = stablehlo.multiply %v2461, %v2445 : tensor<32x576x14x14xf32>
    %v2463 = stablehlo.reduce(%v2462 init: %v2447) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2464 = stablehlo.broadcast_in_dim %v2463, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2465 = stablehlo.multiply %v2460, %v2462 : tensor<32x576x14x14xf32>
    %v2466 = stablehlo.reduce(%v2465 init: %v2447) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2467 = stablehlo.broadcast_in_dim %v2466, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2468 = stablehlo.multiply %v2462, %v2448 : tensor<32x576x14x14xf32>
    %v2469 = stablehlo.subtract %v2468, %v2464 : tensor<32x576x14x14xf32>
    %v2470 = stablehlo.multiply %v2460, %v2467 : tensor<32x576x14x14xf32>
    %v2471 = stablehlo.subtract %v2469, %v2470 : tensor<32x576x14x14xf32>
    %v2472 = stablehlo.divide %v2459, %v2448 : tensor<32x576x14x14xf32>
    %v2473 = stablehlo.multiply %v2472, %v2471 : tensor<32x576x14x14xf32>
    %v2474 = stablehlo.reshape %v2473 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v2475 = stablehlo.reshape %v2474 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2476 = stablehlo.transpose %We14, dims = [1, 0, 2, 3] : (tensor<576x96x1x1xf32>) -> tensor<96x576x1x1xf32>
    %v2477 = stablehlo.reverse %v2476, dims = [2, 3] : tensor<96x576x1x1xf32>
    %v2478 = stablehlo.convolution(%v2475, %v2477)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x576x14x14xf32>, tensor<96x576x1x1xf32>) -> tensor<32x96x14x14xf32>
    %v2479 = stablehlo.reshape %v2478 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v2480 = stablehlo.reshape %v1162 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2481 = stablehlo.reshape %v2474 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2482 = stablehlo.transpose %v2480, dims = [1, 0, 2, 3] : (tensor<32x96x14x14xf32>) -> tensor<96x32x14x14xf32>
    %v2483 = stablehlo.transpose %v2481, dims = [1, 0, 2, 3] : (tensor<32x576x14x14xf32>) -> tensor<576x32x14x14xf32>
    %v2484 = stablehlo.convolution(%v2482, %v2483)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x14x14xf32>, tensor<576x32x14x14xf32>) -> tensor<96x576x1x1xf32>
    %v2485 = stablehlo.transpose %v2484, dims = [1, 0, 2, 3] : (tensor<96x576x1x1xf32>) -> tensor<576x96x1x1xf32>
    %v2486 = stablehlo.constant dense<0.3> : tensor<576x96x1x1xf32>
    %v2487 = stablehlo.multiply %v2485, %v2486 : tensor<576x96x1x1xf32>
    %v2488 = stablehlo.subtract %We14, %v2487 : tensor<576x96x1x1xf32>
    %v2489 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2490 = stablehlo.reshape %v1167 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2491 = stablehlo.constant dense<196.0> : tensor<32x576x14x14xf32>
    %v2492 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v2493 = stablehlo.reduce(%v2490 init: %v2489) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2494 = stablehlo.broadcast_in_dim %v2493, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2495 = stablehlo.divide %v2494, %v2491 : tensor<32x576x14x14xf32>
    %v2496 = stablehlo.subtract %v2490, %v2495 : tensor<32x576x14x14xf32>
    %v2497 = stablehlo.multiply %v2496, %v2496 : tensor<32x576x14x14xf32>
    %v2498 = stablehlo.reduce(%v2497 init: %v2489) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2499 = stablehlo.broadcast_in_dim %v2498, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2500 = stablehlo.divide %v2499, %v2491 : tensor<32x576x14x14xf32>
    %v2501 = stablehlo.add %v2500, %v2492 : tensor<32x576x14x14xf32>
    %v2502 = stablehlo.rsqrt %v2501 : tensor<32x576x14x14xf32>
    %v2503 = stablehlo.multiply %v2496, %v2502 : tensor<32x576x14x14xf32>
    %v2504 = stablehlo.reshape %v2444 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2505 = stablehlo.multiply %v2504, %v2503 : tensor<32x576x14x14xf32>
    %v2506 = stablehlo.reduce(%v2505 init: %v2489) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2507 = stablehlo.constant dense<0.3> : tensor<576xf32>
    %v2508 = stablehlo.multiply %v2506, %v2507 : tensor<576xf32>
    %v2509 = stablehlo.subtract %ge14, %v2508 : tensor<576xf32>
    %v2510 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2511 = stablehlo.reshape %v2444 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2512 = stablehlo.reduce(%v2511 init: %v2510) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2513 = stablehlo.constant dense<0.3> : tensor<576xf32>
    %v2514 = stablehlo.multiply %v2512, %v2513 : tensor<576xf32>
    %v2515 = stablehlo.subtract %bte14, %v2514 : tensor<576xf32>
    %v2516 = stablehlo.reshape %v1193 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2517 = stablehlo.reshape %v2429 : (tensor<32x28224xf32>) -> tensor<32x576x7x7xf32>
    %v2518 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2519 = stablehlo.pad %v2517, %v2518, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x576x7x7xf32>, tensor<f32>) -> tensor<32x576x14x14xf32>
    %v2520 = stablehlo.transpose %v2516, dims = [1, 0, 2, 3] : (tensor<32x576x14x14xf32>) -> tensor<576x32x14x14xf32>
    %v2521 = stablehlo.transpose %v2519, dims = [1, 0, 2, 3] : (tensor<32x576x14x14xf32>) -> tensor<576x32x14x14xf32>
    %v2522 = stablehlo.convolution(%v2520, %v2521)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 576 : i64, feature_group_count = 1 : i64} : (tensor<576x32x14x14xf32>, tensor<576x32x14x14xf32>) -> tensor<1x576x3x3xf32>
    %v2523 = stablehlo.reshape %v2522 : (tensor<1x576x3x3xf32>) -> tensor<576x1x3x3xf32>
    %v2524 = stablehlo.constant dense<0.3> : tensor<576x1x3x3xf32>
    %v2525 = stablehlo.multiply %v2523, %v2524 : tensor<576x1x3x3xf32>
    %v2526 = stablehlo.subtract %Wd14, %v2525 : tensor<576x1x3x3xf32>
    %v2527 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2528 = stablehlo.reshape %v1198 : (tensor<32x28224xf32>) -> tensor<32x576x7x7xf32>
    %v2529 = stablehlo.constant dense<49.0> : tensor<32x576x7x7xf32>
    %v2530 = stablehlo.constant dense<1.0e-5> : tensor<32x576x7x7xf32>
    %v2531 = stablehlo.reduce(%v2528 init: %v2527) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x7x7xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2532 = stablehlo.broadcast_in_dim %v2531, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x7x7xf32>
    %v2533 = stablehlo.divide %v2532, %v2529 : tensor<32x576x7x7xf32>
    %v2534 = stablehlo.subtract %v2528, %v2533 : tensor<32x576x7x7xf32>
    %v2535 = stablehlo.multiply %v2534, %v2534 : tensor<32x576x7x7xf32>
    %v2536 = stablehlo.reduce(%v2535 init: %v2527) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x7x7xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2537 = stablehlo.broadcast_in_dim %v2536, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x7x7xf32>
    %v2538 = stablehlo.divide %v2537, %v2529 : tensor<32x576x7x7xf32>
    %v2539 = stablehlo.add %v2538, %v2530 : tensor<32x576x7x7xf32>
    %v2540 = stablehlo.rsqrt %v2539 : tensor<32x576x7x7xf32>
    %v2541 = stablehlo.multiply %v2534, %v2540 : tensor<32x576x7x7xf32>
    %v2542 = stablehlo.reshape %v2399 : (tensor<32x28224xf32>) -> tensor<32x576x7x7xf32>
    %v2543 = stablehlo.multiply %v2542, %v2541 : tensor<32x576x7x7xf32>
    %v2544 = stablehlo.reduce(%v2543 init: %v2527) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x7x7xf32>, tensor<f32>) -> tensor<576xf32>
    %v2545 = stablehlo.constant dense<0.3> : tensor<576xf32>
    %v2546 = stablehlo.multiply %v2544, %v2545 : tensor<576xf32>
    %v2547 = stablehlo.subtract %gd14, %v2546 : tensor<576xf32>
    %v2548 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2549 = stablehlo.reshape %v2399 : (tensor<32x28224xf32>) -> tensor<32x576x7x7xf32>
    %v2550 = stablehlo.reduce(%v2549 init: %v2548) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x7x7xf32>, tensor<f32>) -> tensor<576xf32>
    %v2551 = stablehlo.constant dense<0.3> : tensor<576xf32>
    %v2552 = stablehlo.multiply %v2550, %v2551 : tensor<576xf32>
    %v2553 = stablehlo.subtract %btd14, %v2552 : tensor<576xf32>
    %v2554 = stablehlo.reshape %v1224 : (tensor<32x28224xf32>) -> tensor<32x576x7x7xf32>
    %v2555 = stablehlo.reshape %v2385 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2556 = stablehlo.transpose %v2554, dims = [1, 0, 2, 3] : (tensor<32x576x7x7xf32>) -> tensor<576x32x7x7xf32>
    %v2557 = stablehlo.transpose %v2555, dims = [1, 0, 2, 3] : (tensor<32x160x7x7xf32>) -> tensor<160x32x7x7xf32>
    %v2558 = stablehlo.convolution(%v2556, %v2557)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<576x32x7x7xf32>, tensor<160x32x7x7xf32>) -> tensor<576x160x1x1xf32>
    %v2559 = stablehlo.transpose %v2558, dims = [1, 0, 2, 3] : (tensor<576x160x1x1xf32>) -> tensor<160x576x1x1xf32>
    %v2560 = stablehlo.constant dense<0.3> : tensor<160x576x1x1xf32>
    %v2561 = stablehlo.multiply %v2559, %v2560 : tensor<160x576x1x1xf32>
    %v2562 = stablehlo.subtract %Wp14, %v2561 : tensor<160x576x1x1xf32>
    %v2563 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2564 = stablehlo.reshape %v1229 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2565 = stablehlo.constant dense<49.0> : tensor<32x160x7x7xf32>
    %v2566 = stablehlo.constant dense<1.0e-5> : tensor<32x160x7x7xf32>
    %v2567 = stablehlo.reduce(%v2564 init: %v2563) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<32x160xf32>
    %v2568 = stablehlo.broadcast_in_dim %v2567, dims = [0, 1] : (tensor<32x160xf32>) -> tensor<32x160x7x7xf32>
    %v2569 = stablehlo.divide %v2568, %v2565 : tensor<32x160x7x7xf32>
    %v2570 = stablehlo.subtract %v2564, %v2569 : tensor<32x160x7x7xf32>
    %v2571 = stablehlo.multiply %v2570, %v2570 : tensor<32x160x7x7xf32>
    %v2572 = stablehlo.reduce(%v2571 init: %v2563) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<32x160xf32>
    %v2573 = stablehlo.broadcast_in_dim %v2572, dims = [0, 1] : (tensor<32x160xf32>) -> tensor<32x160x7x7xf32>
    %v2574 = stablehlo.divide %v2573, %v2565 : tensor<32x160x7x7xf32>
    %v2575 = stablehlo.add %v2574, %v2566 : tensor<32x160x7x7xf32>
    %v2576 = stablehlo.rsqrt %v2575 : tensor<32x160x7x7xf32>
    %v2577 = stablehlo.multiply %v2570, %v2576 : tensor<32x160x7x7xf32>
    %v2578 = stablehlo.reshape %v2247 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2579 = stablehlo.multiply %v2578, %v2577 : tensor<32x160x7x7xf32>
    %v2580 = stablehlo.reduce(%v2579 init: %v2563) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v2581 = stablehlo.constant dense<0.3> : tensor<160xf32>
    %v2582 = stablehlo.multiply %v2580, %v2581 : tensor<160xf32>
    %v2583 = stablehlo.subtract %gp14, %v2582 : tensor<160xf32>
    %v2584 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2585 = stablehlo.reshape %v2247 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2586 = stablehlo.reduce(%v2585 init: %v2584) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v2587 = stablehlo.constant dense<0.3> : tensor<160xf32>
    %v2588 = stablehlo.multiply %v2586, %v2587 : tensor<160xf32>
    %v2589 = stablehlo.subtract %btp14, %v2588 : tensor<160xf32>
    %v2590 = stablehlo.reshape %v2479 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2591 = stablehlo.reshape %v1138 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2592 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2593 = stablehlo.constant dense<196.0> : tensor<32x96x14x14xf32>
    %v2594 = stablehlo.constant dense<1.0e-5> : tensor<32x96x14x14xf32>
    %v2595 = stablehlo.reduce(%v2591 init: %v2592) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v2596 = stablehlo.broadcast_in_dim %v2595, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x14x14xf32>
    %v2597 = stablehlo.divide %v2596, %v2593 : tensor<32x96x14x14xf32>
    %v2598 = stablehlo.subtract %v2591, %v2597 : tensor<32x96x14x14xf32>
    %v2599 = stablehlo.multiply %v2598, %v2598 : tensor<32x96x14x14xf32>
    %v2600 = stablehlo.reduce(%v2599 init: %v2592) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v2601 = stablehlo.broadcast_in_dim %v2600, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x14x14xf32>
    %v2602 = stablehlo.divide %v2601, %v2593 : tensor<32x96x14x14xf32>
    %v2603 = stablehlo.add %v2602, %v2594 : tensor<32x96x14x14xf32>
    %v2604 = stablehlo.rsqrt %v2603 : tensor<32x96x14x14xf32>
    %v2605 = stablehlo.multiply %v2598, %v2604 : tensor<32x96x14x14xf32>
    %v2606 = stablehlo.broadcast_in_dim %gp13, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v2607 = stablehlo.multiply %v2606, %v2590 : tensor<32x96x14x14xf32>
    %v2608 = stablehlo.reduce(%v2607 init: %v2592) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v2609 = stablehlo.broadcast_in_dim %v2608, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x14x14xf32>
    %v2610 = stablehlo.multiply %v2605, %v2607 : tensor<32x96x14x14xf32>
    %v2611 = stablehlo.reduce(%v2610 init: %v2592) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v2612 = stablehlo.broadcast_in_dim %v2611, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x14x14xf32>
    %v2613 = stablehlo.multiply %v2607, %v2593 : tensor<32x96x14x14xf32>
    %v2614 = stablehlo.subtract %v2613, %v2609 : tensor<32x96x14x14xf32>
    %v2615 = stablehlo.multiply %v2605, %v2612 : tensor<32x96x14x14xf32>
    %v2616 = stablehlo.subtract %v2614, %v2615 : tensor<32x96x14x14xf32>
    %v2617 = stablehlo.divide %v2604, %v2593 : tensor<32x96x14x14xf32>
    %v2618 = stablehlo.multiply %v2617, %v2616 : tensor<32x96x14x14xf32>
    %v2619 = stablehlo.reshape %v2618 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v2620 = stablehlo.reshape %v2619 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2621 = stablehlo.transpose %Wp13, dims = [1, 0, 2, 3] : (tensor<96x576x1x1xf32>) -> tensor<576x96x1x1xf32>
    %v2622 = stablehlo.reverse %v2621, dims = [2, 3] : tensor<576x96x1x1xf32>
    %v2623 = stablehlo.convolution(%v2620, %v2622)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x14x14xf32>, tensor<576x96x1x1xf32>) -> tensor<32x576x14x14xf32>
    %v2624 = stablehlo.reshape %v2623 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v2625 = stablehlo.reshape %v2624 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2626 = stablehlo.reshape %v1127 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2627 = stablehlo.constant dense<0.0> : tensor<32x576x14x14xf32>
    %v2628 = stablehlo.constant dense<6.0> : tensor<32x576x14x14xf32>
    %v2629 = stablehlo.compare GT, %v2626, %v2627 : (tensor<32x576x14x14xf32>, tensor<32x576x14x14xf32>) -> tensor<32x576x14x14xi1>
    %v2630 = stablehlo.compare LT, %v2626, %v2628 : (tensor<32x576x14x14xf32>, tensor<32x576x14x14xf32>) -> tensor<32x576x14x14xi1>
    %v2631 = stablehlo.and %v2629, %v2630 : tensor<32x576x14x14xi1>
    %v2632 = stablehlo.select %v2631, %v2625, %v2627 : tensor<32x576x14x14xi1>, tensor<32x576x14x14xf32>
    %v2633 = stablehlo.reshape %v2632 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v2634 = stablehlo.reshape %v2633 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2635 = stablehlo.reshape %v1107 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2636 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2637 = stablehlo.constant dense<196.0> : tensor<32x576x14x14xf32>
    %v2638 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v2639 = stablehlo.reduce(%v2635 init: %v2636) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2640 = stablehlo.broadcast_in_dim %v2639, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2641 = stablehlo.divide %v2640, %v2637 : tensor<32x576x14x14xf32>
    %v2642 = stablehlo.subtract %v2635, %v2641 : tensor<32x576x14x14xf32>
    %v2643 = stablehlo.multiply %v2642, %v2642 : tensor<32x576x14x14xf32>
    %v2644 = stablehlo.reduce(%v2643 init: %v2636) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2645 = stablehlo.broadcast_in_dim %v2644, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2646 = stablehlo.divide %v2645, %v2637 : tensor<32x576x14x14xf32>
    %v2647 = stablehlo.add %v2646, %v2638 : tensor<32x576x14x14xf32>
    %v2648 = stablehlo.rsqrt %v2647 : tensor<32x576x14x14xf32>
    %v2649 = stablehlo.multiply %v2642, %v2648 : tensor<32x576x14x14xf32>
    %v2650 = stablehlo.broadcast_in_dim %gd13, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2651 = stablehlo.multiply %v2650, %v2634 : tensor<32x576x14x14xf32>
    %v2652 = stablehlo.reduce(%v2651 init: %v2636) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2653 = stablehlo.broadcast_in_dim %v2652, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2654 = stablehlo.multiply %v2649, %v2651 : tensor<32x576x14x14xf32>
    %v2655 = stablehlo.reduce(%v2654 init: %v2636) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2656 = stablehlo.broadcast_in_dim %v2655, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2657 = stablehlo.multiply %v2651, %v2637 : tensor<32x576x14x14xf32>
    %v2658 = stablehlo.subtract %v2657, %v2653 : tensor<32x576x14x14xf32>
    %v2659 = stablehlo.multiply %v2649, %v2656 : tensor<32x576x14x14xf32>
    %v2660 = stablehlo.subtract %v2658, %v2659 : tensor<32x576x14x14xf32>
    %v2661 = stablehlo.divide %v2648, %v2637 : tensor<32x576x14x14xf32>
    %v2662 = stablehlo.multiply %v2661, %v2660 : tensor<32x576x14x14xf32>
    %v2663 = stablehlo.reshape %v2662 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v2664 = stablehlo.reshape %v2663 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2665 = stablehlo.reverse %Wd13, dims = [2, 3] : tensor<576x1x3x3xf32>
    %v2666 = stablehlo.convolution(%v2664, %v2665)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 576 : i64} : (tensor<32x576x14x14xf32>, tensor<576x1x3x3xf32>) -> tensor<32x576x14x14xf32>
    %v2667 = stablehlo.reshape %v2666 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v2668 = stablehlo.reshape %v2667 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2669 = stablehlo.reshape %v1096 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2670 = stablehlo.constant dense<0.0> : tensor<32x576x14x14xf32>
    %v2671 = stablehlo.constant dense<6.0> : tensor<32x576x14x14xf32>
    %v2672 = stablehlo.compare GT, %v2669, %v2670 : (tensor<32x576x14x14xf32>, tensor<32x576x14x14xf32>) -> tensor<32x576x14x14xi1>
    %v2673 = stablehlo.compare LT, %v2669, %v2671 : (tensor<32x576x14x14xf32>, tensor<32x576x14x14xf32>) -> tensor<32x576x14x14xi1>
    %v2674 = stablehlo.and %v2672, %v2673 : tensor<32x576x14x14xi1>
    %v2675 = stablehlo.select %v2674, %v2668, %v2670 : tensor<32x576x14x14xi1>, tensor<32x576x14x14xf32>
    %v2676 = stablehlo.reshape %v2675 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v2677 = stablehlo.reshape %v2676 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2678 = stablehlo.reshape %v1076 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2679 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2680 = stablehlo.constant dense<196.0> : tensor<32x576x14x14xf32>
    %v2681 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v2682 = stablehlo.reduce(%v2678 init: %v2679) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2683 = stablehlo.broadcast_in_dim %v2682, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2684 = stablehlo.divide %v2683, %v2680 : tensor<32x576x14x14xf32>
    %v2685 = stablehlo.subtract %v2678, %v2684 : tensor<32x576x14x14xf32>
    %v2686 = stablehlo.multiply %v2685, %v2685 : tensor<32x576x14x14xf32>
    %v2687 = stablehlo.reduce(%v2686 init: %v2679) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2688 = stablehlo.broadcast_in_dim %v2687, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2689 = stablehlo.divide %v2688, %v2680 : tensor<32x576x14x14xf32>
    %v2690 = stablehlo.add %v2689, %v2681 : tensor<32x576x14x14xf32>
    %v2691 = stablehlo.rsqrt %v2690 : tensor<32x576x14x14xf32>
    %v2692 = stablehlo.multiply %v2685, %v2691 : tensor<32x576x14x14xf32>
    %v2693 = stablehlo.broadcast_in_dim %ge13, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2694 = stablehlo.multiply %v2693, %v2677 : tensor<32x576x14x14xf32>
    %v2695 = stablehlo.reduce(%v2694 init: %v2679) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2696 = stablehlo.broadcast_in_dim %v2695, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2697 = stablehlo.multiply %v2692, %v2694 : tensor<32x576x14x14xf32>
    %v2698 = stablehlo.reduce(%v2697 init: %v2679) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2699 = stablehlo.broadcast_in_dim %v2698, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2700 = stablehlo.multiply %v2694, %v2680 : tensor<32x576x14x14xf32>
    %v2701 = stablehlo.subtract %v2700, %v2696 : tensor<32x576x14x14xf32>
    %v2702 = stablehlo.multiply %v2692, %v2699 : tensor<32x576x14x14xf32>
    %v2703 = stablehlo.subtract %v2701, %v2702 : tensor<32x576x14x14xf32>
    %v2704 = stablehlo.divide %v2691, %v2680 : tensor<32x576x14x14xf32>
    %v2705 = stablehlo.multiply %v2704, %v2703 : tensor<32x576x14x14xf32>
    %v2706 = stablehlo.reshape %v2705 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v2707 = stablehlo.reshape %v2706 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2708 = stablehlo.transpose %We13, dims = [1, 0, 2, 3] : (tensor<576x96x1x1xf32>) -> tensor<96x576x1x1xf32>
    %v2709 = stablehlo.reverse %v2708, dims = [2, 3] : tensor<96x576x1x1xf32>
    %v2710 = stablehlo.convolution(%v2707, %v2709)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x576x14x14xf32>, tensor<96x576x1x1xf32>) -> tensor<32x96x14x14xf32>
    %v2711 = stablehlo.reshape %v2710 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v2712 = stablehlo.reshape %v2711 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2713 = stablehlo.reshape %v2479 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2714 = stablehlo.add %v2712, %v2713 : tensor<32x96x14x14xf32>
    %v2715 = stablehlo.reshape %v2714 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v2716 = stablehlo.reshape %v1071 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2717 = stablehlo.reshape %v2706 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2718 = stablehlo.transpose %v2716, dims = [1, 0, 2, 3] : (tensor<32x96x14x14xf32>) -> tensor<96x32x14x14xf32>
    %v2719 = stablehlo.transpose %v2717, dims = [1, 0, 2, 3] : (tensor<32x576x14x14xf32>) -> tensor<576x32x14x14xf32>
    %v2720 = stablehlo.convolution(%v2718, %v2719)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x14x14xf32>, tensor<576x32x14x14xf32>) -> tensor<96x576x1x1xf32>
    %v2721 = stablehlo.transpose %v2720, dims = [1, 0, 2, 3] : (tensor<96x576x1x1xf32>) -> tensor<576x96x1x1xf32>
    %v2722 = stablehlo.constant dense<0.3> : tensor<576x96x1x1xf32>
    %v2723 = stablehlo.multiply %v2721, %v2722 : tensor<576x96x1x1xf32>
    %v2724 = stablehlo.subtract %We13, %v2723 : tensor<576x96x1x1xf32>
    %v2725 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2726 = stablehlo.reshape %v1076 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2727 = stablehlo.constant dense<196.0> : tensor<32x576x14x14xf32>
    %v2728 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v2729 = stablehlo.reduce(%v2726 init: %v2725) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2730 = stablehlo.broadcast_in_dim %v2729, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2731 = stablehlo.divide %v2730, %v2727 : tensor<32x576x14x14xf32>
    %v2732 = stablehlo.subtract %v2726, %v2731 : tensor<32x576x14x14xf32>
    %v2733 = stablehlo.multiply %v2732, %v2732 : tensor<32x576x14x14xf32>
    %v2734 = stablehlo.reduce(%v2733 init: %v2725) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2735 = stablehlo.broadcast_in_dim %v2734, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2736 = stablehlo.divide %v2735, %v2727 : tensor<32x576x14x14xf32>
    %v2737 = stablehlo.add %v2736, %v2728 : tensor<32x576x14x14xf32>
    %v2738 = stablehlo.rsqrt %v2737 : tensor<32x576x14x14xf32>
    %v2739 = stablehlo.multiply %v2732, %v2738 : tensor<32x576x14x14xf32>
    %v2740 = stablehlo.reshape %v2676 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2741 = stablehlo.multiply %v2740, %v2739 : tensor<32x576x14x14xf32>
    %v2742 = stablehlo.reduce(%v2741 init: %v2725) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2743 = stablehlo.constant dense<0.3> : tensor<576xf32>
    %v2744 = stablehlo.multiply %v2742, %v2743 : tensor<576xf32>
    %v2745 = stablehlo.subtract %ge13, %v2744 : tensor<576xf32>
    %v2746 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2747 = stablehlo.reshape %v2676 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2748 = stablehlo.reduce(%v2747 init: %v2746) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2749 = stablehlo.constant dense<0.3> : tensor<576xf32>
    %v2750 = stablehlo.multiply %v2748, %v2749 : tensor<576xf32>
    %v2751 = stablehlo.subtract %bte13, %v2750 : tensor<576xf32>
    %v2752 = stablehlo.reshape %v1102 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2753 = stablehlo.reshape %v2663 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2754 = stablehlo.transpose %v2752, dims = [1, 0, 2, 3] : (tensor<32x576x14x14xf32>) -> tensor<576x32x14x14xf32>
    %v2755 = stablehlo.transpose %v2753, dims = [1, 0, 2, 3] : (tensor<32x576x14x14xf32>) -> tensor<576x32x14x14xf32>
    %v2756 = stablehlo.convolution(%v2754, %v2755)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 576 : i64, feature_group_count = 1 : i64} : (tensor<576x32x14x14xf32>, tensor<576x32x14x14xf32>) -> tensor<1x576x3x3xf32>
    %v2757 = stablehlo.reshape %v2756 : (tensor<1x576x3x3xf32>) -> tensor<576x1x3x3xf32>
    %v2758 = stablehlo.constant dense<0.3> : tensor<576x1x3x3xf32>
    %v2759 = stablehlo.multiply %v2757, %v2758 : tensor<576x1x3x3xf32>
    %v2760 = stablehlo.subtract %Wd13, %v2759 : tensor<576x1x3x3xf32>
    %v2761 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2762 = stablehlo.reshape %v1107 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2763 = stablehlo.constant dense<196.0> : tensor<32x576x14x14xf32>
    %v2764 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v2765 = stablehlo.reduce(%v2762 init: %v2761) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2766 = stablehlo.broadcast_in_dim %v2765, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2767 = stablehlo.divide %v2766, %v2763 : tensor<32x576x14x14xf32>
    %v2768 = stablehlo.subtract %v2762, %v2767 : tensor<32x576x14x14xf32>
    %v2769 = stablehlo.multiply %v2768, %v2768 : tensor<32x576x14x14xf32>
    %v2770 = stablehlo.reduce(%v2769 init: %v2761) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2771 = stablehlo.broadcast_in_dim %v2770, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2772 = stablehlo.divide %v2771, %v2763 : tensor<32x576x14x14xf32>
    %v2773 = stablehlo.add %v2772, %v2764 : tensor<32x576x14x14xf32>
    %v2774 = stablehlo.rsqrt %v2773 : tensor<32x576x14x14xf32>
    %v2775 = stablehlo.multiply %v2768, %v2774 : tensor<32x576x14x14xf32>
    %v2776 = stablehlo.reshape %v2633 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2777 = stablehlo.multiply %v2776, %v2775 : tensor<32x576x14x14xf32>
    %v2778 = stablehlo.reduce(%v2777 init: %v2761) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2779 = stablehlo.constant dense<0.3> : tensor<576xf32>
    %v2780 = stablehlo.multiply %v2778, %v2779 : tensor<576xf32>
    %v2781 = stablehlo.subtract %gd13, %v2780 : tensor<576xf32>
    %v2782 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2783 = stablehlo.reshape %v2633 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2784 = stablehlo.reduce(%v2783 init: %v2782) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2785 = stablehlo.constant dense<0.3> : tensor<576xf32>
    %v2786 = stablehlo.multiply %v2784, %v2785 : tensor<576xf32>
    %v2787 = stablehlo.subtract %btd13, %v2786 : tensor<576xf32>
    %v2788 = stablehlo.reshape %v1133 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2789 = stablehlo.reshape %v2619 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2790 = stablehlo.transpose %v2788, dims = [1, 0, 2, 3] : (tensor<32x576x14x14xf32>) -> tensor<576x32x14x14xf32>
    %v2791 = stablehlo.transpose %v2789, dims = [1, 0, 2, 3] : (tensor<32x96x14x14xf32>) -> tensor<96x32x14x14xf32>
    %v2792 = stablehlo.convolution(%v2790, %v2791)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<576x32x14x14xf32>, tensor<96x32x14x14xf32>) -> tensor<576x96x1x1xf32>
    %v2793 = stablehlo.transpose %v2792, dims = [1, 0, 2, 3] : (tensor<576x96x1x1xf32>) -> tensor<96x576x1x1xf32>
    %v2794 = stablehlo.constant dense<0.3> : tensor<96x576x1x1xf32>
    %v2795 = stablehlo.multiply %v2793, %v2794 : tensor<96x576x1x1xf32>
    %v2796 = stablehlo.subtract %Wp13, %v2795 : tensor<96x576x1x1xf32>
    %v2797 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2798 = stablehlo.reshape %v1138 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2799 = stablehlo.constant dense<196.0> : tensor<32x96x14x14xf32>
    %v2800 = stablehlo.constant dense<1.0e-5> : tensor<32x96x14x14xf32>
    %v2801 = stablehlo.reduce(%v2798 init: %v2797) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v2802 = stablehlo.broadcast_in_dim %v2801, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x14x14xf32>
    %v2803 = stablehlo.divide %v2802, %v2799 : tensor<32x96x14x14xf32>
    %v2804 = stablehlo.subtract %v2798, %v2803 : tensor<32x96x14x14xf32>
    %v2805 = stablehlo.multiply %v2804, %v2804 : tensor<32x96x14x14xf32>
    %v2806 = stablehlo.reduce(%v2805 init: %v2797) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v2807 = stablehlo.broadcast_in_dim %v2806, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x14x14xf32>
    %v2808 = stablehlo.divide %v2807, %v2799 : tensor<32x96x14x14xf32>
    %v2809 = stablehlo.add %v2808, %v2800 : tensor<32x96x14x14xf32>
    %v2810 = stablehlo.rsqrt %v2809 : tensor<32x96x14x14xf32>
    %v2811 = stablehlo.multiply %v2804, %v2810 : tensor<32x96x14x14xf32>
    %v2812 = stablehlo.reshape %v2479 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2813 = stablehlo.multiply %v2812, %v2811 : tensor<32x96x14x14xf32>
    %v2814 = stablehlo.reduce(%v2813 init: %v2797) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v2815 = stablehlo.constant dense<0.3> : tensor<96xf32>
    %v2816 = stablehlo.multiply %v2814, %v2815 : tensor<96xf32>
    %v2817 = stablehlo.subtract %gp13, %v2816 : tensor<96xf32>
    %v2818 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2819 = stablehlo.reshape %v2479 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2820 = stablehlo.reduce(%v2819 init: %v2818) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v2821 = stablehlo.constant dense<0.3> : tensor<96xf32>
    %v2822 = stablehlo.multiply %v2820, %v2821 : tensor<96xf32>
    %v2823 = stablehlo.subtract %btp13, %v2822 : tensor<96xf32>
    %v2824 = stablehlo.reshape %v2715 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2825 = stablehlo.reshape %v1047 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2826 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2827 = stablehlo.constant dense<196.0> : tensor<32x96x14x14xf32>
    %v2828 = stablehlo.constant dense<1.0e-5> : tensor<32x96x14x14xf32>
    %v2829 = stablehlo.reduce(%v2825 init: %v2826) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v2830 = stablehlo.broadcast_in_dim %v2829, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x14x14xf32>
    %v2831 = stablehlo.divide %v2830, %v2827 : tensor<32x96x14x14xf32>
    %v2832 = stablehlo.subtract %v2825, %v2831 : tensor<32x96x14x14xf32>
    %v2833 = stablehlo.multiply %v2832, %v2832 : tensor<32x96x14x14xf32>
    %v2834 = stablehlo.reduce(%v2833 init: %v2826) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v2835 = stablehlo.broadcast_in_dim %v2834, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x14x14xf32>
    %v2836 = stablehlo.divide %v2835, %v2827 : tensor<32x96x14x14xf32>
    %v2837 = stablehlo.add %v2836, %v2828 : tensor<32x96x14x14xf32>
    %v2838 = stablehlo.rsqrt %v2837 : tensor<32x96x14x14xf32>
    %v2839 = stablehlo.multiply %v2832, %v2838 : tensor<32x96x14x14xf32>
    %v2840 = stablehlo.broadcast_in_dim %gp12, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v2841 = stablehlo.multiply %v2840, %v2824 : tensor<32x96x14x14xf32>
    %v2842 = stablehlo.reduce(%v2841 init: %v2826) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v2843 = stablehlo.broadcast_in_dim %v2842, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x14x14xf32>
    %v2844 = stablehlo.multiply %v2839, %v2841 : tensor<32x96x14x14xf32>
    %v2845 = stablehlo.reduce(%v2844 init: %v2826) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v2846 = stablehlo.broadcast_in_dim %v2845, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x14x14xf32>
    %v2847 = stablehlo.multiply %v2841, %v2827 : tensor<32x96x14x14xf32>
    %v2848 = stablehlo.subtract %v2847, %v2843 : tensor<32x96x14x14xf32>
    %v2849 = stablehlo.multiply %v2839, %v2846 : tensor<32x96x14x14xf32>
    %v2850 = stablehlo.subtract %v2848, %v2849 : tensor<32x96x14x14xf32>
    %v2851 = stablehlo.divide %v2838, %v2827 : tensor<32x96x14x14xf32>
    %v2852 = stablehlo.multiply %v2851, %v2850 : tensor<32x96x14x14xf32>
    %v2853 = stablehlo.reshape %v2852 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v2854 = stablehlo.reshape %v2853 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2855 = stablehlo.transpose %Wp12, dims = [1, 0, 2, 3] : (tensor<96x576x1x1xf32>) -> tensor<576x96x1x1xf32>
    %v2856 = stablehlo.reverse %v2855, dims = [2, 3] : tensor<576x96x1x1xf32>
    %v2857 = stablehlo.convolution(%v2854, %v2856)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x14x14xf32>, tensor<576x96x1x1xf32>) -> tensor<32x576x14x14xf32>
    %v2858 = stablehlo.reshape %v2857 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v2859 = stablehlo.reshape %v2858 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2860 = stablehlo.reshape %v1036 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2861 = stablehlo.constant dense<0.0> : tensor<32x576x14x14xf32>
    %v2862 = stablehlo.constant dense<6.0> : tensor<32x576x14x14xf32>
    %v2863 = stablehlo.compare GT, %v2860, %v2861 : (tensor<32x576x14x14xf32>, tensor<32x576x14x14xf32>) -> tensor<32x576x14x14xi1>
    %v2864 = stablehlo.compare LT, %v2860, %v2862 : (tensor<32x576x14x14xf32>, tensor<32x576x14x14xf32>) -> tensor<32x576x14x14xi1>
    %v2865 = stablehlo.and %v2863, %v2864 : tensor<32x576x14x14xi1>
    %v2866 = stablehlo.select %v2865, %v2859, %v2861 : tensor<32x576x14x14xi1>, tensor<32x576x14x14xf32>
    %v2867 = stablehlo.reshape %v2866 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v2868 = stablehlo.reshape %v2867 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2869 = stablehlo.reshape %v1016 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2870 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2871 = stablehlo.constant dense<196.0> : tensor<32x576x14x14xf32>
    %v2872 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v2873 = stablehlo.reduce(%v2869 init: %v2870) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2874 = stablehlo.broadcast_in_dim %v2873, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2875 = stablehlo.divide %v2874, %v2871 : tensor<32x576x14x14xf32>
    %v2876 = stablehlo.subtract %v2869, %v2875 : tensor<32x576x14x14xf32>
    %v2877 = stablehlo.multiply %v2876, %v2876 : tensor<32x576x14x14xf32>
    %v2878 = stablehlo.reduce(%v2877 init: %v2870) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2879 = stablehlo.broadcast_in_dim %v2878, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2880 = stablehlo.divide %v2879, %v2871 : tensor<32x576x14x14xf32>
    %v2881 = stablehlo.add %v2880, %v2872 : tensor<32x576x14x14xf32>
    %v2882 = stablehlo.rsqrt %v2881 : tensor<32x576x14x14xf32>
    %v2883 = stablehlo.multiply %v2876, %v2882 : tensor<32x576x14x14xf32>
    %v2884 = stablehlo.broadcast_in_dim %gd12, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2885 = stablehlo.multiply %v2884, %v2868 : tensor<32x576x14x14xf32>
    %v2886 = stablehlo.reduce(%v2885 init: %v2870) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2887 = stablehlo.broadcast_in_dim %v2886, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2888 = stablehlo.multiply %v2883, %v2885 : tensor<32x576x14x14xf32>
    %v2889 = stablehlo.reduce(%v2888 init: %v2870) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2890 = stablehlo.broadcast_in_dim %v2889, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2891 = stablehlo.multiply %v2885, %v2871 : tensor<32x576x14x14xf32>
    %v2892 = stablehlo.subtract %v2891, %v2887 : tensor<32x576x14x14xf32>
    %v2893 = stablehlo.multiply %v2883, %v2890 : tensor<32x576x14x14xf32>
    %v2894 = stablehlo.subtract %v2892, %v2893 : tensor<32x576x14x14xf32>
    %v2895 = stablehlo.divide %v2882, %v2871 : tensor<32x576x14x14xf32>
    %v2896 = stablehlo.multiply %v2895, %v2894 : tensor<32x576x14x14xf32>
    %v2897 = stablehlo.reshape %v2896 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v2898 = stablehlo.reshape %v2897 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2899 = stablehlo.reverse %Wd12, dims = [2, 3] : tensor<576x1x3x3xf32>
    %v2900 = stablehlo.convolution(%v2898, %v2899)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 576 : i64} : (tensor<32x576x14x14xf32>, tensor<576x1x3x3xf32>) -> tensor<32x576x14x14xf32>
    %v2901 = stablehlo.reshape %v2900 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v2902 = stablehlo.reshape %v2901 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2903 = stablehlo.reshape %v1005 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2904 = stablehlo.constant dense<0.0> : tensor<32x576x14x14xf32>
    %v2905 = stablehlo.constant dense<6.0> : tensor<32x576x14x14xf32>
    %v2906 = stablehlo.compare GT, %v2903, %v2904 : (tensor<32x576x14x14xf32>, tensor<32x576x14x14xf32>) -> tensor<32x576x14x14xi1>
    %v2907 = stablehlo.compare LT, %v2903, %v2905 : (tensor<32x576x14x14xf32>, tensor<32x576x14x14xf32>) -> tensor<32x576x14x14xi1>
    %v2908 = stablehlo.and %v2906, %v2907 : tensor<32x576x14x14xi1>
    %v2909 = stablehlo.select %v2908, %v2902, %v2904 : tensor<32x576x14x14xi1>, tensor<32x576x14x14xf32>
    %v2910 = stablehlo.reshape %v2909 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v2911 = stablehlo.reshape %v2910 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2912 = stablehlo.reshape %v985 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2913 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2914 = stablehlo.constant dense<196.0> : tensor<32x576x14x14xf32>
    %v2915 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v2916 = stablehlo.reduce(%v2912 init: %v2913) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2917 = stablehlo.broadcast_in_dim %v2916, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2918 = stablehlo.divide %v2917, %v2914 : tensor<32x576x14x14xf32>
    %v2919 = stablehlo.subtract %v2912, %v2918 : tensor<32x576x14x14xf32>
    %v2920 = stablehlo.multiply %v2919, %v2919 : tensor<32x576x14x14xf32>
    %v2921 = stablehlo.reduce(%v2920 init: %v2913) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2922 = stablehlo.broadcast_in_dim %v2921, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2923 = stablehlo.divide %v2922, %v2914 : tensor<32x576x14x14xf32>
    %v2924 = stablehlo.add %v2923, %v2915 : tensor<32x576x14x14xf32>
    %v2925 = stablehlo.rsqrt %v2924 : tensor<32x576x14x14xf32>
    %v2926 = stablehlo.multiply %v2919, %v2925 : tensor<32x576x14x14xf32>
    %v2927 = stablehlo.broadcast_in_dim %ge12, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2928 = stablehlo.multiply %v2927, %v2911 : tensor<32x576x14x14xf32>
    %v2929 = stablehlo.reduce(%v2928 init: %v2913) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2930 = stablehlo.broadcast_in_dim %v2929, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2931 = stablehlo.multiply %v2926, %v2928 : tensor<32x576x14x14xf32>
    %v2932 = stablehlo.reduce(%v2931 init: %v2913) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2933 = stablehlo.broadcast_in_dim %v2932, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2934 = stablehlo.multiply %v2928, %v2914 : tensor<32x576x14x14xf32>
    %v2935 = stablehlo.subtract %v2934, %v2930 : tensor<32x576x14x14xf32>
    %v2936 = stablehlo.multiply %v2926, %v2933 : tensor<32x576x14x14xf32>
    %v2937 = stablehlo.subtract %v2935, %v2936 : tensor<32x576x14x14xf32>
    %v2938 = stablehlo.divide %v2925, %v2914 : tensor<32x576x14x14xf32>
    %v2939 = stablehlo.multiply %v2938, %v2937 : tensor<32x576x14x14xf32>
    %v2940 = stablehlo.reshape %v2939 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v2941 = stablehlo.reshape %v2940 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2942 = stablehlo.transpose %We12, dims = [1, 0, 2, 3] : (tensor<576x96x1x1xf32>) -> tensor<96x576x1x1xf32>
    %v2943 = stablehlo.reverse %v2942, dims = [2, 3] : tensor<96x576x1x1xf32>
    %v2944 = stablehlo.convolution(%v2941, %v2943)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x576x14x14xf32>, tensor<96x576x1x1xf32>) -> tensor<32x96x14x14xf32>
    %v2945 = stablehlo.reshape %v2944 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v2946 = stablehlo.reshape %v2945 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2947 = stablehlo.reshape %v2715 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2948 = stablehlo.add %v2946, %v2947 : tensor<32x96x14x14xf32>
    %v2949 = stablehlo.reshape %v2948 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v2950 = stablehlo.reshape %v980 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2951 = stablehlo.reshape %v2940 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2952 = stablehlo.transpose %v2950, dims = [1, 0, 2, 3] : (tensor<32x96x14x14xf32>) -> tensor<96x32x14x14xf32>
    %v2953 = stablehlo.transpose %v2951, dims = [1, 0, 2, 3] : (tensor<32x576x14x14xf32>) -> tensor<576x32x14x14xf32>
    %v2954 = stablehlo.convolution(%v2952, %v2953)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x14x14xf32>, tensor<576x32x14x14xf32>) -> tensor<96x576x1x1xf32>
    %v2955 = stablehlo.transpose %v2954, dims = [1, 0, 2, 3] : (tensor<96x576x1x1xf32>) -> tensor<576x96x1x1xf32>
    %v2956 = stablehlo.constant dense<0.3> : tensor<576x96x1x1xf32>
    %v2957 = stablehlo.multiply %v2955, %v2956 : tensor<576x96x1x1xf32>
    %v2958 = stablehlo.subtract %We12, %v2957 : tensor<576x96x1x1xf32>
    %v2959 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2960 = stablehlo.reshape %v985 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2961 = stablehlo.constant dense<196.0> : tensor<32x576x14x14xf32>
    %v2962 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v2963 = stablehlo.reduce(%v2960 init: %v2959) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2964 = stablehlo.broadcast_in_dim %v2963, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2965 = stablehlo.divide %v2964, %v2961 : tensor<32x576x14x14xf32>
    %v2966 = stablehlo.subtract %v2960, %v2965 : tensor<32x576x14x14xf32>
    %v2967 = stablehlo.multiply %v2966, %v2966 : tensor<32x576x14x14xf32>
    %v2968 = stablehlo.reduce(%v2967 init: %v2959) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2969 = stablehlo.broadcast_in_dim %v2968, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2970 = stablehlo.divide %v2969, %v2961 : tensor<32x576x14x14xf32>
    %v2971 = stablehlo.add %v2970, %v2962 : tensor<32x576x14x14xf32>
    %v2972 = stablehlo.rsqrt %v2971 : tensor<32x576x14x14xf32>
    %v2973 = stablehlo.multiply %v2966, %v2972 : tensor<32x576x14x14xf32>
    %v2974 = stablehlo.reshape %v2910 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2975 = stablehlo.multiply %v2974, %v2973 : tensor<32x576x14x14xf32>
    %v2976 = stablehlo.reduce(%v2975 init: %v2959) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2977 = stablehlo.constant dense<0.3> : tensor<576xf32>
    %v2978 = stablehlo.multiply %v2976, %v2977 : tensor<576xf32>
    %v2979 = stablehlo.subtract %ge12, %v2978 : tensor<576xf32>
    %v2980 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2981 = stablehlo.reshape %v2910 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2982 = stablehlo.reduce(%v2981 init: %v2980) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2983 = stablehlo.constant dense<0.3> : tensor<576xf32>
    %v2984 = stablehlo.multiply %v2982, %v2983 : tensor<576xf32>
    %v2985 = stablehlo.subtract %bte12, %v2984 : tensor<576xf32>
    %v2986 = stablehlo.reshape %v1011 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2987 = stablehlo.reshape %v2897 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2988 = stablehlo.transpose %v2986, dims = [1, 0, 2, 3] : (tensor<32x576x14x14xf32>) -> tensor<576x32x14x14xf32>
    %v2989 = stablehlo.transpose %v2987, dims = [1, 0, 2, 3] : (tensor<32x576x14x14xf32>) -> tensor<576x32x14x14xf32>
    %v2990 = stablehlo.convolution(%v2988, %v2989)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 576 : i64, feature_group_count = 1 : i64} : (tensor<576x32x14x14xf32>, tensor<576x32x14x14xf32>) -> tensor<1x576x3x3xf32>
    %v2991 = stablehlo.reshape %v2990 : (tensor<1x576x3x3xf32>) -> tensor<576x1x3x3xf32>
    %v2992 = stablehlo.constant dense<0.3> : tensor<576x1x3x3xf32>
    %v2993 = stablehlo.multiply %v2991, %v2992 : tensor<576x1x3x3xf32>
    %v2994 = stablehlo.subtract %Wd12, %v2993 : tensor<576x1x3x3xf32>
    %v2995 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2996 = stablehlo.reshape %v1016 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2997 = stablehlo.constant dense<196.0> : tensor<32x576x14x14xf32>
    %v2998 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v2999 = stablehlo.reduce(%v2996 init: %v2995) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v3000 = stablehlo.broadcast_in_dim %v2999, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v3001 = stablehlo.divide %v3000, %v2997 : tensor<32x576x14x14xf32>
    %v3002 = stablehlo.subtract %v2996, %v3001 : tensor<32x576x14x14xf32>
    %v3003 = stablehlo.multiply %v3002, %v3002 : tensor<32x576x14x14xf32>
    %v3004 = stablehlo.reduce(%v3003 init: %v2995) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v3005 = stablehlo.broadcast_in_dim %v3004, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v3006 = stablehlo.divide %v3005, %v2997 : tensor<32x576x14x14xf32>
    %v3007 = stablehlo.add %v3006, %v2998 : tensor<32x576x14x14xf32>
    %v3008 = stablehlo.rsqrt %v3007 : tensor<32x576x14x14xf32>
    %v3009 = stablehlo.multiply %v3002, %v3008 : tensor<32x576x14x14xf32>
    %v3010 = stablehlo.reshape %v2867 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v3011 = stablehlo.multiply %v3010, %v3009 : tensor<32x576x14x14xf32>
    %v3012 = stablehlo.reduce(%v3011 init: %v2995) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v3013 = stablehlo.constant dense<0.3> : tensor<576xf32>
    %v3014 = stablehlo.multiply %v3012, %v3013 : tensor<576xf32>
    %v3015 = stablehlo.subtract %gd12, %v3014 : tensor<576xf32>
    %v3016 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3017 = stablehlo.reshape %v2867 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v3018 = stablehlo.reduce(%v3017 init: %v3016) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v3019 = stablehlo.constant dense<0.3> : tensor<576xf32>
    %v3020 = stablehlo.multiply %v3018, %v3019 : tensor<576xf32>
    %v3021 = stablehlo.subtract %btd12, %v3020 : tensor<576xf32>
    %v3022 = stablehlo.reshape %v1042 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v3023 = stablehlo.reshape %v2853 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v3024 = stablehlo.transpose %v3022, dims = [1, 0, 2, 3] : (tensor<32x576x14x14xf32>) -> tensor<576x32x14x14xf32>
    %v3025 = stablehlo.transpose %v3023, dims = [1, 0, 2, 3] : (tensor<32x96x14x14xf32>) -> tensor<96x32x14x14xf32>
    %v3026 = stablehlo.convolution(%v3024, %v3025)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<576x32x14x14xf32>, tensor<96x32x14x14xf32>) -> tensor<576x96x1x1xf32>
    %v3027 = stablehlo.transpose %v3026, dims = [1, 0, 2, 3] : (tensor<576x96x1x1xf32>) -> tensor<96x576x1x1xf32>
    %v3028 = stablehlo.constant dense<0.3> : tensor<96x576x1x1xf32>
    %v3029 = stablehlo.multiply %v3027, %v3028 : tensor<96x576x1x1xf32>
    %v3030 = stablehlo.subtract %Wp12, %v3029 : tensor<96x576x1x1xf32>
    %v3031 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3032 = stablehlo.reshape %v1047 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v3033 = stablehlo.constant dense<196.0> : tensor<32x96x14x14xf32>
    %v3034 = stablehlo.constant dense<1.0e-5> : tensor<32x96x14x14xf32>
    %v3035 = stablehlo.reduce(%v3032 init: %v3031) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v3036 = stablehlo.broadcast_in_dim %v3035, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x14x14xf32>
    %v3037 = stablehlo.divide %v3036, %v3033 : tensor<32x96x14x14xf32>
    %v3038 = stablehlo.subtract %v3032, %v3037 : tensor<32x96x14x14xf32>
    %v3039 = stablehlo.multiply %v3038, %v3038 : tensor<32x96x14x14xf32>
    %v3040 = stablehlo.reduce(%v3039 init: %v3031) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v3041 = stablehlo.broadcast_in_dim %v3040, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x14x14xf32>
    %v3042 = stablehlo.divide %v3041, %v3033 : tensor<32x96x14x14xf32>
    %v3043 = stablehlo.add %v3042, %v3034 : tensor<32x96x14x14xf32>
    %v3044 = stablehlo.rsqrt %v3043 : tensor<32x96x14x14xf32>
    %v3045 = stablehlo.multiply %v3038, %v3044 : tensor<32x96x14x14xf32>
    %v3046 = stablehlo.reshape %v2715 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v3047 = stablehlo.multiply %v3046, %v3045 : tensor<32x96x14x14xf32>
    %v3048 = stablehlo.reduce(%v3047 init: %v3031) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v3049 = stablehlo.constant dense<0.3> : tensor<96xf32>
    %v3050 = stablehlo.multiply %v3048, %v3049 : tensor<96xf32>
    %v3051 = stablehlo.subtract %gp12, %v3050 : tensor<96xf32>
    %v3052 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3053 = stablehlo.reshape %v2715 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v3054 = stablehlo.reduce(%v3053 init: %v3052) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v3055 = stablehlo.constant dense<0.3> : tensor<96xf32>
    %v3056 = stablehlo.multiply %v3054, %v3055 : tensor<96xf32>
    %v3057 = stablehlo.subtract %btp12, %v3056 : tensor<96xf32>
    %v3058 = stablehlo.reshape %v2949 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v3059 = stablehlo.reshape %v960 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v3060 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3061 = stablehlo.constant dense<196.0> : tensor<32x96x14x14xf32>
    %v3062 = stablehlo.constant dense<1.0e-5> : tensor<32x96x14x14xf32>
    %v3063 = stablehlo.reduce(%v3059 init: %v3060) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v3064 = stablehlo.broadcast_in_dim %v3063, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x14x14xf32>
    %v3065 = stablehlo.divide %v3064, %v3061 : tensor<32x96x14x14xf32>
    %v3066 = stablehlo.subtract %v3059, %v3065 : tensor<32x96x14x14xf32>
    %v3067 = stablehlo.multiply %v3066, %v3066 : tensor<32x96x14x14xf32>
    %v3068 = stablehlo.reduce(%v3067 init: %v3060) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v3069 = stablehlo.broadcast_in_dim %v3068, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x14x14xf32>
    %v3070 = stablehlo.divide %v3069, %v3061 : tensor<32x96x14x14xf32>
    %v3071 = stablehlo.add %v3070, %v3062 : tensor<32x96x14x14xf32>
    %v3072 = stablehlo.rsqrt %v3071 : tensor<32x96x14x14xf32>
    %v3073 = stablehlo.multiply %v3066, %v3072 : tensor<32x96x14x14xf32>
    %v3074 = stablehlo.broadcast_in_dim %gp11, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v3075 = stablehlo.multiply %v3074, %v3058 : tensor<32x96x14x14xf32>
    %v3076 = stablehlo.reduce(%v3075 init: %v3060) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v3077 = stablehlo.broadcast_in_dim %v3076, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x14x14xf32>
    %v3078 = stablehlo.multiply %v3073, %v3075 : tensor<32x96x14x14xf32>
    %v3079 = stablehlo.reduce(%v3078 init: %v3060) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v3080 = stablehlo.broadcast_in_dim %v3079, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x14x14xf32>
    %v3081 = stablehlo.multiply %v3075, %v3061 : tensor<32x96x14x14xf32>
    %v3082 = stablehlo.subtract %v3081, %v3077 : tensor<32x96x14x14xf32>
    %v3083 = stablehlo.multiply %v3073, %v3080 : tensor<32x96x14x14xf32>
    %v3084 = stablehlo.subtract %v3082, %v3083 : tensor<32x96x14x14xf32>
    %v3085 = stablehlo.divide %v3072, %v3061 : tensor<32x96x14x14xf32>
    %v3086 = stablehlo.multiply %v3085, %v3084 : tensor<32x96x14x14xf32>
    %v3087 = stablehlo.reshape %v3086 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v3088 = stablehlo.reshape %v3087 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v3089 = stablehlo.transpose %Wp11, dims = [1, 0, 2, 3] : (tensor<96x384x1x1xf32>) -> tensor<384x96x1x1xf32>
    %v3090 = stablehlo.reverse %v3089, dims = [2, 3] : tensor<384x96x1x1xf32>
    %v3091 = stablehlo.convolution(%v3088, %v3090)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x14x14xf32>, tensor<384x96x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v3092 = stablehlo.reshape %v3091 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3093 = stablehlo.reshape %v3092 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3094 = stablehlo.reshape %v949 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3095 = stablehlo.constant dense<0.0> : tensor<32x384x14x14xf32>
    %v3096 = stablehlo.constant dense<6.0> : tensor<32x384x14x14xf32>
    %v3097 = stablehlo.compare GT, %v3094, %v3095 : (tensor<32x384x14x14xf32>, tensor<32x384x14x14xf32>) -> tensor<32x384x14x14xi1>
    %v3098 = stablehlo.compare LT, %v3094, %v3096 : (tensor<32x384x14x14xf32>, tensor<32x384x14x14xf32>) -> tensor<32x384x14x14xi1>
    %v3099 = stablehlo.and %v3097, %v3098 : tensor<32x384x14x14xi1>
    %v3100 = stablehlo.select %v3099, %v3093, %v3095 : tensor<32x384x14x14xi1>, tensor<32x384x14x14xf32>
    %v3101 = stablehlo.reshape %v3100 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3102 = stablehlo.reshape %v3101 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3103 = stablehlo.reshape %v929 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3104 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3105 = stablehlo.constant dense<196.0> : tensor<32x384x14x14xf32>
    %v3106 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3107 = stablehlo.reduce(%v3103 init: %v3104) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3108 = stablehlo.broadcast_in_dim %v3107, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3109 = stablehlo.divide %v3108, %v3105 : tensor<32x384x14x14xf32>
    %v3110 = stablehlo.subtract %v3103, %v3109 : tensor<32x384x14x14xf32>
    %v3111 = stablehlo.multiply %v3110, %v3110 : tensor<32x384x14x14xf32>
    %v3112 = stablehlo.reduce(%v3111 init: %v3104) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3113 = stablehlo.broadcast_in_dim %v3112, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3114 = stablehlo.divide %v3113, %v3105 : tensor<32x384x14x14xf32>
    %v3115 = stablehlo.add %v3114, %v3106 : tensor<32x384x14x14xf32>
    %v3116 = stablehlo.rsqrt %v3115 : tensor<32x384x14x14xf32>
    %v3117 = stablehlo.multiply %v3110, %v3116 : tensor<32x384x14x14xf32>
    %v3118 = stablehlo.broadcast_in_dim %gd11, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3119 = stablehlo.multiply %v3118, %v3102 : tensor<32x384x14x14xf32>
    %v3120 = stablehlo.reduce(%v3119 init: %v3104) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3121 = stablehlo.broadcast_in_dim %v3120, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3122 = stablehlo.multiply %v3117, %v3119 : tensor<32x384x14x14xf32>
    %v3123 = stablehlo.reduce(%v3122 init: %v3104) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3124 = stablehlo.broadcast_in_dim %v3123, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3125 = stablehlo.multiply %v3119, %v3105 : tensor<32x384x14x14xf32>
    %v3126 = stablehlo.subtract %v3125, %v3121 : tensor<32x384x14x14xf32>
    %v3127 = stablehlo.multiply %v3117, %v3124 : tensor<32x384x14x14xf32>
    %v3128 = stablehlo.subtract %v3126, %v3127 : tensor<32x384x14x14xf32>
    %v3129 = stablehlo.divide %v3116, %v3105 : tensor<32x384x14x14xf32>
    %v3130 = stablehlo.multiply %v3129, %v3128 : tensor<32x384x14x14xf32>
    %v3131 = stablehlo.reshape %v3130 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3132 = stablehlo.reshape %v3131 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3133 = stablehlo.reverse %Wd11, dims = [2, 3] : tensor<384x1x3x3xf32>
    %v3134 = stablehlo.convolution(%v3132, %v3133)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x3x3xf32>) -> tensor<32x384x14x14xf32>
    %v3135 = stablehlo.reshape %v3134 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3136 = stablehlo.reshape %v3135 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3137 = stablehlo.reshape %v918 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3138 = stablehlo.constant dense<0.0> : tensor<32x384x14x14xf32>
    %v3139 = stablehlo.constant dense<6.0> : tensor<32x384x14x14xf32>
    %v3140 = stablehlo.compare GT, %v3137, %v3138 : (tensor<32x384x14x14xf32>, tensor<32x384x14x14xf32>) -> tensor<32x384x14x14xi1>
    %v3141 = stablehlo.compare LT, %v3137, %v3139 : (tensor<32x384x14x14xf32>, tensor<32x384x14x14xf32>) -> tensor<32x384x14x14xi1>
    %v3142 = stablehlo.and %v3140, %v3141 : tensor<32x384x14x14xi1>
    %v3143 = stablehlo.select %v3142, %v3136, %v3138 : tensor<32x384x14x14xi1>, tensor<32x384x14x14xf32>
    %v3144 = stablehlo.reshape %v3143 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3145 = stablehlo.reshape %v3144 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3146 = stablehlo.reshape %v898 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3147 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3148 = stablehlo.constant dense<196.0> : tensor<32x384x14x14xf32>
    %v3149 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3150 = stablehlo.reduce(%v3146 init: %v3147) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3151 = stablehlo.broadcast_in_dim %v3150, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3152 = stablehlo.divide %v3151, %v3148 : tensor<32x384x14x14xf32>
    %v3153 = stablehlo.subtract %v3146, %v3152 : tensor<32x384x14x14xf32>
    %v3154 = stablehlo.multiply %v3153, %v3153 : tensor<32x384x14x14xf32>
    %v3155 = stablehlo.reduce(%v3154 init: %v3147) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3156 = stablehlo.broadcast_in_dim %v3155, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3157 = stablehlo.divide %v3156, %v3148 : tensor<32x384x14x14xf32>
    %v3158 = stablehlo.add %v3157, %v3149 : tensor<32x384x14x14xf32>
    %v3159 = stablehlo.rsqrt %v3158 : tensor<32x384x14x14xf32>
    %v3160 = stablehlo.multiply %v3153, %v3159 : tensor<32x384x14x14xf32>
    %v3161 = stablehlo.broadcast_in_dim %ge11, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3162 = stablehlo.multiply %v3161, %v3145 : tensor<32x384x14x14xf32>
    %v3163 = stablehlo.reduce(%v3162 init: %v3147) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3164 = stablehlo.broadcast_in_dim %v3163, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3165 = stablehlo.multiply %v3160, %v3162 : tensor<32x384x14x14xf32>
    %v3166 = stablehlo.reduce(%v3165 init: %v3147) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3167 = stablehlo.broadcast_in_dim %v3166, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3168 = stablehlo.multiply %v3162, %v3148 : tensor<32x384x14x14xf32>
    %v3169 = stablehlo.subtract %v3168, %v3164 : tensor<32x384x14x14xf32>
    %v3170 = stablehlo.multiply %v3160, %v3167 : tensor<32x384x14x14xf32>
    %v3171 = stablehlo.subtract %v3169, %v3170 : tensor<32x384x14x14xf32>
    %v3172 = stablehlo.divide %v3159, %v3148 : tensor<32x384x14x14xf32>
    %v3173 = stablehlo.multiply %v3172, %v3171 : tensor<32x384x14x14xf32>
    %v3174 = stablehlo.reshape %v3173 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3175 = stablehlo.reshape %v3174 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3176 = stablehlo.transpose %We11, dims = [1, 0, 2, 3] : (tensor<384x64x1x1xf32>) -> tensor<64x384x1x1xf32>
    %v3177 = stablehlo.reverse %v3176, dims = [2, 3] : tensor<64x384x1x1xf32>
    %v3178 = stablehlo.convolution(%v3175, %v3177)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<64x384x1x1xf32>) -> tensor<32x64x14x14xf32>
    %v3179 = stablehlo.reshape %v3178 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v3180 = stablehlo.reshape %v893 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3181 = stablehlo.reshape %v3174 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3182 = stablehlo.transpose %v3180, dims = [1, 0, 2, 3] : (tensor<32x64x14x14xf32>) -> tensor<64x32x14x14xf32>
    %v3183 = stablehlo.transpose %v3181, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3184 = stablehlo.convolution(%v3182, %v3183)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<64x384x1x1xf32>
    %v3185 = stablehlo.transpose %v3184, dims = [1, 0, 2, 3] : (tensor<64x384x1x1xf32>) -> tensor<384x64x1x1xf32>
    %v3186 = stablehlo.constant dense<0.3> : tensor<384x64x1x1xf32>
    %v3187 = stablehlo.multiply %v3185, %v3186 : tensor<384x64x1x1xf32>
    %v3188 = stablehlo.subtract %We11, %v3187 : tensor<384x64x1x1xf32>
    %v3189 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3190 = stablehlo.reshape %v898 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3191 = stablehlo.constant dense<196.0> : tensor<32x384x14x14xf32>
    %v3192 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3193 = stablehlo.reduce(%v3190 init: %v3189) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3194 = stablehlo.broadcast_in_dim %v3193, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3195 = stablehlo.divide %v3194, %v3191 : tensor<32x384x14x14xf32>
    %v3196 = stablehlo.subtract %v3190, %v3195 : tensor<32x384x14x14xf32>
    %v3197 = stablehlo.multiply %v3196, %v3196 : tensor<32x384x14x14xf32>
    %v3198 = stablehlo.reduce(%v3197 init: %v3189) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3199 = stablehlo.broadcast_in_dim %v3198, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3200 = stablehlo.divide %v3199, %v3191 : tensor<32x384x14x14xf32>
    %v3201 = stablehlo.add %v3200, %v3192 : tensor<32x384x14x14xf32>
    %v3202 = stablehlo.rsqrt %v3201 : tensor<32x384x14x14xf32>
    %v3203 = stablehlo.multiply %v3196, %v3202 : tensor<32x384x14x14xf32>
    %v3204 = stablehlo.reshape %v3144 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3205 = stablehlo.multiply %v3204, %v3203 : tensor<32x384x14x14xf32>
    %v3206 = stablehlo.reduce(%v3205 init: %v3189) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3207 = stablehlo.constant dense<0.3> : tensor<384xf32>
    %v3208 = stablehlo.multiply %v3206, %v3207 : tensor<384xf32>
    %v3209 = stablehlo.subtract %ge11, %v3208 : tensor<384xf32>
    %v3210 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3211 = stablehlo.reshape %v3144 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3212 = stablehlo.reduce(%v3211 init: %v3210) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3213 = stablehlo.constant dense<0.3> : tensor<384xf32>
    %v3214 = stablehlo.multiply %v3212, %v3213 : tensor<384xf32>
    %v3215 = stablehlo.subtract %bte11, %v3214 : tensor<384xf32>
    %v3216 = stablehlo.reshape %v924 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3217 = stablehlo.reshape %v3131 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3218 = stablehlo.transpose %v3216, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3219 = stablehlo.transpose %v3217, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3220 = stablehlo.convolution(%v3218, %v3219)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x3x3xf32>
    %v3221 = stablehlo.reshape %v3220 : (tensor<1x384x3x3xf32>) -> tensor<384x1x3x3xf32>
    %v3222 = stablehlo.constant dense<0.3> : tensor<384x1x3x3xf32>
    %v3223 = stablehlo.multiply %v3221, %v3222 : tensor<384x1x3x3xf32>
    %v3224 = stablehlo.subtract %Wd11, %v3223 : tensor<384x1x3x3xf32>
    %v3225 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3226 = stablehlo.reshape %v929 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3227 = stablehlo.constant dense<196.0> : tensor<32x384x14x14xf32>
    %v3228 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3229 = stablehlo.reduce(%v3226 init: %v3225) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3230 = stablehlo.broadcast_in_dim %v3229, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3231 = stablehlo.divide %v3230, %v3227 : tensor<32x384x14x14xf32>
    %v3232 = stablehlo.subtract %v3226, %v3231 : tensor<32x384x14x14xf32>
    %v3233 = stablehlo.multiply %v3232, %v3232 : tensor<32x384x14x14xf32>
    %v3234 = stablehlo.reduce(%v3233 init: %v3225) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3235 = stablehlo.broadcast_in_dim %v3234, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3236 = stablehlo.divide %v3235, %v3227 : tensor<32x384x14x14xf32>
    %v3237 = stablehlo.add %v3236, %v3228 : tensor<32x384x14x14xf32>
    %v3238 = stablehlo.rsqrt %v3237 : tensor<32x384x14x14xf32>
    %v3239 = stablehlo.multiply %v3232, %v3238 : tensor<32x384x14x14xf32>
    %v3240 = stablehlo.reshape %v3101 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3241 = stablehlo.multiply %v3240, %v3239 : tensor<32x384x14x14xf32>
    %v3242 = stablehlo.reduce(%v3241 init: %v3225) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3243 = stablehlo.constant dense<0.3> : tensor<384xf32>
    %v3244 = stablehlo.multiply %v3242, %v3243 : tensor<384xf32>
    %v3245 = stablehlo.subtract %gd11, %v3244 : tensor<384xf32>
    %v3246 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3247 = stablehlo.reshape %v3101 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3248 = stablehlo.reduce(%v3247 init: %v3246) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3249 = stablehlo.constant dense<0.3> : tensor<384xf32>
    %v3250 = stablehlo.multiply %v3248, %v3249 : tensor<384xf32>
    %v3251 = stablehlo.subtract %btd11, %v3250 : tensor<384xf32>
    %v3252 = stablehlo.reshape %v955 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3253 = stablehlo.reshape %v3087 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v3254 = stablehlo.transpose %v3252, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3255 = stablehlo.transpose %v3253, dims = [1, 0, 2, 3] : (tensor<32x96x14x14xf32>) -> tensor<96x32x14x14xf32>
    %v3256 = stablehlo.convolution(%v3254, %v3255)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<96x32x14x14xf32>) -> tensor<384x96x1x1xf32>
    %v3257 = stablehlo.transpose %v3256, dims = [1, 0, 2, 3] : (tensor<384x96x1x1xf32>) -> tensor<96x384x1x1xf32>
    %v3258 = stablehlo.constant dense<0.3> : tensor<96x384x1x1xf32>
    %v3259 = stablehlo.multiply %v3257, %v3258 : tensor<96x384x1x1xf32>
    %v3260 = stablehlo.subtract %Wp11, %v3259 : tensor<96x384x1x1xf32>
    %v3261 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3262 = stablehlo.reshape %v960 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v3263 = stablehlo.constant dense<196.0> : tensor<32x96x14x14xf32>
    %v3264 = stablehlo.constant dense<1.0e-5> : tensor<32x96x14x14xf32>
    %v3265 = stablehlo.reduce(%v3262 init: %v3261) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v3266 = stablehlo.broadcast_in_dim %v3265, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x14x14xf32>
    %v3267 = stablehlo.divide %v3266, %v3263 : tensor<32x96x14x14xf32>
    %v3268 = stablehlo.subtract %v3262, %v3267 : tensor<32x96x14x14xf32>
    %v3269 = stablehlo.multiply %v3268, %v3268 : tensor<32x96x14x14xf32>
    %v3270 = stablehlo.reduce(%v3269 init: %v3261) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v3271 = stablehlo.broadcast_in_dim %v3270, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x14x14xf32>
    %v3272 = stablehlo.divide %v3271, %v3263 : tensor<32x96x14x14xf32>
    %v3273 = stablehlo.add %v3272, %v3264 : tensor<32x96x14x14xf32>
    %v3274 = stablehlo.rsqrt %v3273 : tensor<32x96x14x14xf32>
    %v3275 = stablehlo.multiply %v3268, %v3274 : tensor<32x96x14x14xf32>
    %v3276 = stablehlo.reshape %v2949 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v3277 = stablehlo.multiply %v3276, %v3275 : tensor<32x96x14x14xf32>
    %v3278 = stablehlo.reduce(%v3277 init: %v3261) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v3279 = stablehlo.constant dense<0.3> : tensor<96xf32>
    %v3280 = stablehlo.multiply %v3278, %v3279 : tensor<96xf32>
    %v3281 = stablehlo.subtract %gp11, %v3280 : tensor<96xf32>
    %v3282 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3283 = stablehlo.reshape %v2949 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v3284 = stablehlo.reduce(%v3283 init: %v3282) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v3285 = stablehlo.constant dense<0.3> : tensor<96xf32>
    %v3286 = stablehlo.multiply %v3284, %v3285 : tensor<96xf32>
    %v3287 = stablehlo.subtract %btp11, %v3286 : tensor<96xf32>
    %v3288 = stablehlo.reshape %v3179 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3289 = stablehlo.reshape %v869 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3290 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3291 = stablehlo.constant dense<196.0> : tensor<32x64x14x14xf32>
    %v3292 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v3293 = stablehlo.reduce(%v3289 init: %v3290) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3294 = stablehlo.broadcast_in_dim %v3293, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v3295 = stablehlo.divide %v3294, %v3291 : tensor<32x64x14x14xf32>
    %v3296 = stablehlo.subtract %v3289, %v3295 : tensor<32x64x14x14xf32>
    %v3297 = stablehlo.multiply %v3296, %v3296 : tensor<32x64x14x14xf32>
    %v3298 = stablehlo.reduce(%v3297 init: %v3290) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3299 = stablehlo.broadcast_in_dim %v3298, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v3300 = stablehlo.divide %v3299, %v3291 : tensor<32x64x14x14xf32>
    %v3301 = stablehlo.add %v3300, %v3292 : tensor<32x64x14x14xf32>
    %v3302 = stablehlo.rsqrt %v3301 : tensor<32x64x14x14xf32>
    %v3303 = stablehlo.multiply %v3296, %v3302 : tensor<32x64x14x14xf32>
    %v3304 = stablehlo.broadcast_in_dim %gp10, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3305 = stablehlo.multiply %v3304, %v3288 : tensor<32x64x14x14xf32>
    %v3306 = stablehlo.reduce(%v3305 init: %v3290) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3307 = stablehlo.broadcast_in_dim %v3306, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v3308 = stablehlo.multiply %v3303, %v3305 : tensor<32x64x14x14xf32>
    %v3309 = stablehlo.reduce(%v3308 init: %v3290) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3310 = stablehlo.broadcast_in_dim %v3309, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v3311 = stablehlo.multiply %v3305, %v3291 : tensor<32x64x14x14xf32>
    %v3312 = stablehlo.subtract %v3311, %v3307 : tensor<32x64x14x14xf32>
    %v3313 = stablehlo.multiply %v3303, %v3310 : tensor<32x64x14x14xf32>
    %v3314 = stablehlo.subtract %v3312, %v3313 : tensor<32x64x14x14xf32>
    %v3315 = stablehlo.divide %v3302, %v3291 : tensor<32x64x14x14xf32>
    %v3316 = stablehlo.multiply %v3315, %v3314 : tensor<32x64x14x14xf32>
    %v3317 = stablehlo.reshape %v3316 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v3318 = stablehlo.reshape %v3317 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3319 = stablehlo.transpose %Wp10, dims = [1, 0, 2, 3] : (tensor<64x384x1x1xf32>) -> tensor<384x64x1x1xf32>
    %v3320 = stablehlo.reverse %v3319, dims = [2, 3] : tensor<384x64x1x1xf32>
    %v3321 = stablehlo.convolution(%v3318, %v3320)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x14x14xf32>, tensor<384x64x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v3322 = stablehlo.reshape %v3321 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3323 = stablehlo.reshape %v3322 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3324 = stablehlo.reshape %v858 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3325 = stablehlo.constant dense<0.0> : tensor<32x384x14x14xf32>
    %v3326 = stablehlo.constant dense<6.0> : tensor<32x384x14x14xf32>
    %v3327 = stablehlo.compare GT, %v3324, %v3325 : (tensor<32x384x14x14xf32>, tensor<32x384x14x14xf32>) -> tensor<32x384x14x14xi1>
    %v3328 = stablehlo.compare LT, %v3324, %v3326 : (tensor<32x384x14x14xf32>, tensor<32x384x14x14xf32>) -> tensor<32x384x14x14xi1>
    %v3329 = stablehlo.and %v3327, %v3328 : tensor<32x384x14x14xi1>
    %v3330 = stablehlo.select %v3329, %v3323, %v3325 : tensor<32x384x14x14xi1>, tensor<32x384x14x14xf32>
    %v3331 = stablehlo.reshape %v3330 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3332 = stablehlo.reshape %v3331 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3333 = stablehlo.reshape %v838 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3334 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3335 = stablehlo.constant dense<196.0> : tensor<32x384x14x14xf32>
    %v3336 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3337 = stablehlo.reduce(%v3333 init: %v3334) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3338 = stablehlo.broadcast_in_dim %v3337, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3339 = stablehlo.divide %v3338, %v3335 : tensor<32x384x14x14xf32>
    %v3340 = stablehlo.subtract %v3333, %v3339 : tensor<32x384x14x14xf32>
    %v3341 = stablehlo.multiply %v3340, %v3340 : tensor<32x384x14x14xf32>
    %v3342 = stablehlo.reduce(%v3341 init: %v3334) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3343 = stablehlo.broadcast_in_dim %v3342, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3344 = stablehlo.divide %v3343, %v3335 : tensor<32x384x14x14xf32>
    %v3345 = stablehlo.add %v3344, %v3336 : tensor<32x384x14x14xf32>
    %v3346 = stablehlo.rsqrt %v3345 : tensor<32x384x14x14xf32>
    %v3347 = stablehlo.multiply %v3340, %v3346 : tensor<32x384x14x14xf32>
    %v3348 = stablehlo.broadcast_in_dim %gd10, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3349 = stablehlo.multiply %v3348, %v3332 : tensor<32x384x14x14xf32>
    %v3350 = stablehlo.reduce(%v3349 init: %v3334) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3351 = stablehlo.broadcast_in_dim %v3350, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3352 = stablehlo.multiply %v3347, %v3349 : tensor<32x384x14x14xf32>
    %v3353 = stablehlo.reduce(%v3352 init: %v3334) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3354 = stablehlo.broadcast_in_dim %v3353, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3355 = stablehlo.multiply %v3349, %v3335 : tensor<32x384x14x14xf32>
    %v3356 = stablehlo.subtract %v3355, %v3351 : tensor<32x384x14x14xf32>
    %v3357 = stablehlo.multiply %v3347, %v3354 : tensor<32x384x14x14xf32>
    %v3358 = stablehlo.subtract %v3356, %v3357 : tensor<32x384x14x14xf32>
    %v3359 = stablehlo.divide %v3346, %v3335 : tensor<32x384x14x14xf32>
    %v3360 = stablehlo.multiply %v3359, %v3358 : tensor<32x384x14x14xf32>
    %v3361 = stablehlo.reshape %v3360 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3362 = stablehlo.reshape %v3361 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3363 = stablehlo.reverse %Wd10, dims = [2, 3] : tensor<384x1x3x3xf32>
    %v3364 = stablehlo.convolution(%v3362, %v3363)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x3x3xf32>) -> tensor<32x384x14x14xf32>
    %v3365 = stablehlo.reshape %v3364 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3366 = stablehlo.reshape %v3365 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3367 = stablehlo.reshape %v827 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3368 = stablehlo.constant dense<0.0> : tensor<32x384x14x14xf32>
    %v3369 = stablehlo.constant dense<6.0> : tensor<32x384x14x14xf32>
    %v3370 = stablehlo.compare GT, %v3367, %v3368 : (tensor<32x384x14x14xf32>, tensor<32x384x14x14xf32>) -> tensor<32x384x14x14xi1>
    %v3371 = stablehlo.compare LT, %v3367, %v3369 : (tensor<32x384x14x14xf32>, tensor<32x384x14x14xf32>) -> tensor<32x384x14x14xi1>
    %v3372 = stablehlo.and %v3370, %v3371 : tensor<32x384x14x14xi1>
    %v3373 = stablehlo.select %v3372, %v3366, %v3368 : tensor<32x384x14x14xi1>, tensor<32x384x14x14xf32>
    %v3374 = stablehlo.reshape %v3373 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3375 = stablehlo.reshape %v3374 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3376 = stablehlo.reshape %v807 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3377 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3378 = stablehlo.constant dense<196.0> : tensor<32x384x14x14xf32>
    %v3379 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3380 = stablehlo.reduce(%v3376 init: %v3377) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3381 = stablehlo.broadcast_in_dim %v3380, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3382 = stablehlo.divide %v3381, %v3378 : tensor<32x384x14x14xf32>
    %v3383 = stablehlo.subtract %v3376, %v3382 : tensor<32x384x14x14xf32>
    %v3384 = stablehlo.multiply %v3383, %v3383 : tensor<32x384x14x14xf32>
    %v3385 = stablehlo.reduce(%v3384 init: %v3377) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3386 = stablehlo.broadcast_in_dim %v3385, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3387 = stablehlo.divide %v3386, %v3378 : tensor<32x384x14x14xf32>
    %v3388 = stablehlo.add %v3387, %v3379 : tensor<32x384x14x14xf32>
    %v3389 = stablehlo.rsqrt %v3388 : tensor<32x384x14x14xf32>
    %v3390 = stablehlo.multiply %v3383, %v3389 : tensor<32x384x14x14xf32>
    %v3391 = stablehlo.broadcast_in_dim %ge10, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3392 = stablehlo.multiply %v3391, %v3375 : tensor<32x384x14x14xf32>
    %v3393 = stablehlo.reduce(%v3392 init: %v3377) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3394 = stablehlo.broadcast_in_dim %v3393, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3395 = stablehlo.multiply %v3390, %v3392 : tensor<32x384x14x14xf32>
    %v3396 = stablehlo.reduce(%v3395 init: %v3377) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3397 = stablehlo.broadcast_in_dim %v3396, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3398 = stablehlo.multiply %v3392, %v3378 : tensor<32x384x14x14xf32>
    %v3399 = stablehlo.subtract %v3398, %v3394 : tensor<32x384x14x14xf32>
    %v3400 = stablehlo.multiply %v3390, %v3397 : tensor<32x384x14x14xf32>
    %v3401 = stablehlo.subtract %v3399, %v3400 : tensor<32x384x14x14xf32>
    %v3402 = stablehlo.divide %v3389, %v3378 : tensor<32x384x14x14xf32>
    %v3403 = stablehlo.multiply %v3402, %v3401 : tensor<32x384x14x14xf32>
    %v3404 = stablehlo.reshape %v3403 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3405 = stablehlo.reshape %v3404 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3406 = stablehlo.transpose %We10, dims = [1, 0, 2, 3] : (tensor<384x64x1x1xf32>) -> tensor<64x384x1x1xf32>
    %v3407 = stablehlo.reverse %v3406, dims = [2, 3] : tensor<64x384x1x1xf32>
    %v3408 = stablehlo.convolution(%v3405, %v3407)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<64x384x1x1xf32>) -> tensor<32x64x14x14xf32>
    %v3409 = stablehlo.reshape %v3408 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v3410 = stablehlo.reshape %v3409 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3411 = stablehlo.reshape %v3179 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3412 = stablehlo.add %v3410, %v3411 : tensor<32x64x14x14xf32>
    %v3413 = stablehlo.reshape %v3412 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v3414 = stablehlo.reshape %v802 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3415 = stablehlo.reshape %v3404 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3416 = stablehlo.transpose %v3414, dims = [1, 0, 2, 3] : (tensor<32x64x14x14xf32>) -> tensor<64x32x14x14xf32>
    %v3417 = stablehlo.transpose %v3415, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3418 = stablehlo.convolution(%v3416, %v3417)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<64x384x1x1xf32>
    %v3419 = stablehlo.transpose %v3418, dims = [1, 0, 2, 3] : (tensor<64x384x1x1xf32>) -> tensor<384x64x1x1xf32>
    %v3420 = stablehlo.constant dense<0.3> : tensor<384x64x1x1xf32>
    %v3421 = stablehlo.multiply %v3419, %v3420 : tensor<384x64x1x1xf32>
    %v3422 = stablehlo.subtract %We10, %v3421 : tensor<384x64x1x1xf32>
    %v3423 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3424 = stablehlo.reshape %v807 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3425 = stablehlo.constant dense<196.0> : tensor<32x384x14x14xf32>
    %v3426 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3427 = stablehlo.reduce(%v3424 init: %v3423) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3428 = stablehlo.broadcast_in_dim %v3427, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3429 = stablehlo.divide %v3428, %v3425 : tensor<32x384x14x14xf32>
    %v3430 = stablehlo.subtract %v3424, %v3429 : tensor<32x384x14x14xf32>
    %v3431 = stablehlo.multiply %v3430, %v3430 : tensor<32x384x14x14xf32>
    %v3432 = stablehlo.reduce(%v3431 init: %v3423) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3433 = stablehlo.broadcast_in_dim %v3432, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3434 = stablehlo.divide %v3433, %v3425 : tensor<32x384x14x14xf32>
    %v3435 = stablehlo.add %v3434, %v3426 : tensor<32x384x14x14xf32>
    %v3436 = stablehlo.rsqrt %v3435 : tensor<32x384x14x14xf32>
    %v3437 = stablehlo.multiply %v3430, %v3436 : tensor<32x384x14x14xf32>
    %v3438 = stablehlo.reshape %v3374 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3439 = stablehlo.multiply %v3438, %v3437 : tensor<32x384x14x14xf32>
    %v3440 = stablehlo.reduce(%v3439 init: %v3423) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3441 = stablehlo.constant dense<0.3> : tensor<384xf32>
    %v3442 = stablehlo.multiply %v3440, %v3441 : tensor<384xf32>
    %v3443 = stablehlo.subtract %ge10, %v3442 : tensor<384xf32>
    %v3444 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3445 = stablehlo.reshape %v3374 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3446 = stablehlo.reduce(%v3445 init: %v3444) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3447 = stablehlo.constant dense<0.3> : tensor<384xf32>
    %v3448 = stablehlo.multiply %v3446, %v3447 : tensor<384xf32>
    %v3449 = stablehlo.subtract %bte10, %v3448 : tensor<384xf32>
    %v3450 = stablehlo.reshape %v833 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3451 = stablehlo.reshape %v3361 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3452 = stablehlo.transpose %v3450, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3453 = stablehlo.transpose %v3451, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3454 = stablehlo.convolution(%v3452, %v3453)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x3x3xf32>
    %v3455 = stablehlo.reshape %v3454 : (tensor<1x384x3x3xf32>) -> tensor<384x1x3x3xf32>
    %v3456 = stablehlo.constant dense<0.3> : tensor<384x1x3x3xf32>
    %v3457 = stablehlo.multiply %v3455, %v3456 : tensor<384x1x3x3xf32>
    %v3458 = stablehlo.subtract %Wd10, %v3457 : tensor<384x1x3x3xf32>
    %v3459 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3460 = stablehlo.reshape %v838 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3461 = stablehlo.constant dense<196.0> : tensor<32x384x14x14xf32>
    %v3462 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3463 = stablehlo.reduce(%v3460 init: %v3459) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3464 = stablehlo.broadcast_in_dim %v3463, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3465 = stablehlo.divide %v3464, %v3461 : tensor<32x384x14x14xf32>
    %v3466 = stablehlo.subtract %v3460, %v3465 : tensor<32x384x14x14xf32>
    %v3467 = stablehlo.multiply %v3466, %v3466 : tensor<32x384x14x14xf32>
    %v3468 = stablehlo.reduce(%v3467 init: %v3459) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3469 = stablehlo.broadcast_in_dim %v3468, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3470 = stablehlo.divide %v3469, %v3461 : tensor<32x384x14x14xf32>
    %v3471 = stablehlo.add %v3470, %v3462 : tensor<32x384x14x14xf32>
    %v3472 = stablehlo.rsqrt %v3471 : tensor<32x384x14x14xf32>
    %v3473 = stablehlo.multiply %v3466, %v3472 : tensor<32x384x14x14xf32>
    %v3474 = stablehlo.reshape %v3331 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3475 = stablehlo.multiply %v3474, %v3473 : tensor<32x384x14x14xf32>
    %v3476 = stablehlo.reduce(%v3475 init: %v3459) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3477 = stablehlo.constant dense<0.3> : tensor<384xf32>
    %v3478 = stablehlo.multiply %v3476, %v3477 : tensor<384xf32>
    %v3479 = stablehlo.subtract %gd10, %v3478 : tensor<384xf32>
    %v3480 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3481 = stablehlo.reshape %v3331 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3482 = stablehlo.reduce(%v3481 init: %v3480) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3483 = stablehlo.constant dense<0.3> : tensor<384xf32>
    %v3484 = stablehlo.multiply %v3482, %v3483 : tensor<384xf32>
    %v3485 = stablehlo.subtract %btd10, %v3484 : tensor<384xf32>
    %v3486 = stablehlo.reshape %v864 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3487 = stablehlo.reshape %v3317 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3488 = stablehlo.transpose %v3486, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3489 = stablehlo.transpose %v3487, dims = [1, 0, 2, 3] : (tensor<32x64x14x14xf32>) -> tensor<64x32x14x14xf32>
    %v3490 = stablehlo.convolution(%v3488, %v3489)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<64x32x14x14xf32>) -> tensor<384x64x1x1xf32>
    %v3491 = stablehlo.transpose %v3490, dims = [1, 0, 2, 3] : (tensor<384x64x1x1xf32>) -> tensor<64x384x1x1xf32>
    %v3492 = stablehlo.constant dense<0.3> : tensor<64x384x1x1xf32>
    %v3493 = stablehlo.multiply %v3491, %v3492 : tensor<64x384x1x1xf32>
    %v3494 = stablehlo.subtract %Wp10, %v3493 : tensor<64x384x1x1xf32>
    %v3495 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3496 = stablehlo.reshape %v869 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3497 = stablehlo.constant dense<196.0> : tensor<32x64x14x14xf32>
    %v3498 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v3499 = stablehlo.reduce(%v3496 init: %v3495) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3500 = stablehlo.broadcast_in_dim %v3499, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v3501 = stablehlo.divide %v3500, %v3497 : tensor<32x64x14x14xf32>
    %v3502 = stablehlo.subtract %v3496, %v3501 : tensor<32x64x14x14xf32>
    %v3503 = stablehlo.multiply %v3502, %v3502 : tensor<32x64x14x14xf32>
    %v3504 = stablehlo.reduce(%v3503 init: %v3495) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3505 = stablehlo.broadcast_in_dim %v3504, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v3506 = stablehlo.divide %v3505, %v3497 : tensor<32x64x14x14xf32>
    %v3507 = stablehlo.add %v3506, %v3498 : tensor<32x64x14x14xf32>
    %v3508 = stablehlo.rsqrt %v3507 : tensor<32x64x14x14xf32>
    %v3509 = stablehlo.multiply %v3502, %v3508 : tensor<32x64x14x14xf32>
    %v3510 = stablehlo.reshape %v3179 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3511 = stablehlo.multiply %v3510, %v3509 : tensor<32x64x14x14xf32>
    %v3512 = stablehlo.reduce(%v3511 init: %v3495) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3513 = stablehlo.constant dense<0.3> : tensor<64xf32>
    %v3514 = stablehlo.multiply %v3512, %v3513 : tensor<64xf32>
    %v3515 = stablehlo.subtract %gp10, %v3514 : tensor<64xf32>
    %v3516 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3517 = stablehlo.reshape %v3179 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3518 = stablehlo.reduce(%v3517 init: %v3516) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3519 = stablehlo.constant dense<0.3> : tensor<64xf32>
    %v3520 = stablehlo.multiply %v3518, %v3519 : tensor<64xf32>
    %v3521 = stablehlo.subtract %btp10, %v3520 : tensor<64xf32>
    %v3522 = stablehlo.reshape %v3413 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3523 = stablehlo.reshape %v778 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3524 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3525 = stablehlo.constant dense<196.0> : tensor<32x64x14x14xf32>
    %v3526 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v3527 = stablehlo.reduce(%v3523 init: %v3524) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3528 = stablehlo.broadcast_in_dim %v3527, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v3529 = stablehlo.divide %v3528, %v3525 : tensor<32x64x14x14xf32>
    %v3530 = stablehlo.subtract %v3523, %v3529 : tensor<32x64x14x14xf32>
    %v3531 = stablehlo.multiply %v3530, %v3530 : tensor<32x64x14x14xf32>
    %v3532 = stablehlo.reduce(%v3531 init: %v3524) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3533 = stablehlo.broadcast_in_dim %v3532, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v3534 = stablehlo.divide %v3533, %v3525 : tensor<32x64x14x14xf32>
    %v3535 = stablehlo.add %v3534, %v3526 : tensor<32x64x14x14xf32>
    %v3536 = stablehlo.rsqrt %v3535 : tensor<32x64x14x14xf32>
    %v3537 = stablehlo.multiply %v3530, %v3536 : tensor<32x64x14x14xf32>
    %v3538 = stablehlo.broadcast_in_dim %gp9, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3539 = stablehlo.multiply %v3538, %v3522 : tensor<32x64x14x14xf32>
    %v3540 = stablehlo.reduce(%v3539 init: %v3524) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3541 = stablehlo.broadcast_in_dim %v3540, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v3542 = stablehlo.multiply %v3537, %v3539 : tensor<32x64x14x14xf32>
    %v3543 = stablehlo.reduce(%v3542 init: %v3524) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3544 = stablehlo.broadcast_in_dim %v3543, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v3545 = stablehlo.multiply %v3539, %v3525 : tensor<32x64x14x14xf32>
    %v3546 = stablehlo.subtract %v3545, %v3541 : tensor<32x64x14x14xf32>
    %v3547 = stablehlo.multiply %v3537, %v3544 : tensor<32x64x14x14xf32>
    %v3548 = stablehlo.subtract %v3546, %v3547 : tensor<32x64x14x14xf32>
    %v3549 = stablehlo.divide %v3536, %v3525 : tensor<32x64x14x14xf32>
    %v3550 = stablehlo.multiply %v3549, %v3548 : tensor<32x64x14x14xf32>
    %v3551 = stablehlo.reshape %v3550 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v3552 = stablehlo.reshape %v3551 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3553 = stablehlo.transpose %Wp9, dims = [1, 0, 2, 3] : (tensor<64x384x1x1xf32>) -> tensor<384x64x1x1xf32>
    %v3554 = stablehlo.reverse %v3553, dims = [2, 3] : tensor<384x64x1x1xf32>
    %v3555 = stablehlo.convolution(%v3552, %v3554)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x14x14xf32>, tensor<384x64x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v3556 = stablehlo.reshape %v3555 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3557 = stablehlo.reshape %v3556 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3558 = stablehlo.reshape %v767 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3559 = stablehlo.constant dense<0.0> : tensor<32x384x14x14xf32>
    %v3560 = stablehlo.constant dense<6.0> : tensor<32x384x14x14xf32>
    %v3561 = stablehlo.compare GT, %v3558, %v3559 : (tensor<32x384x14x14xf32>, tensor<32x384x14x14xf32>) -> tensor<32x384x14x14xi1>
    %v3562 = stablehlo.compare LT, %v3558, %v3560 : (tensor<32x384x14x14xf32>, tensor<32x384x14x14xf32>) -> tensor<32x384x14x14xi1>
    %v3563 = stablehlo.and %v3561, %v3562 : tensor<32x384x14x14xi1>
    %v3564 = stablehlo.select %v3563, %v3557, %v3559 : tensor<32x384x14x14xi1>, tensor<32x384x14x14xf32>
    %v3565 = stablehlo.reshape %v3564 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3566 = stablehlo.reshape %v3565 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3567 = stablehlo.reshape %v747 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3568 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3569 = stablehlo.constant dense<196.0> : tensor<32x384x14x14xf32>
    %v3570 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3571 = stablehlo.reduce(%v3567 init: %v3568) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3572 = stablehlo.broadcast_in_dim %v3571, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3573 = stablehlo.divide %v3572, %v3569 : tensor<32x384x14x14xf32>
    %v3574 = stablehlo.subtract %v3567, %v3573 : tensor<32x384x14x14xf32>
    %v3575 = stablehlo.multiply %v3574, %v3574 : tensor<32x384x14x14xf32>
    %v3576 = stablehlo.reduce(%v3575 init: %v3568) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3577 = stablehlo.broadcast_in_dim %v3576, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3578 = stablehlo.divide %v3577, %v3569 : tensor<32x384x14x14xf32>
    %v3579 = stablehlo.add %v3578, %v3570 : tensor<32x384x14x14xf32>
    %v3580 = stablehlo.rsqrt %v3579 : tensor<32x384x14x14xf32>
    %v3581 = stablehlo.multiply %v3574, %v3580 : tensor<32x384x14x14xf32>
    %v3582 = stablehlo.broadcast_in_dim %gd9, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3583 = stablehlo.multiply %v3582, %v3566 : tensor<32x384x14x14xf32>
    %v3584 = stablehlo.reduce(%v3583 init: %v3568) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3585 = stablehlo.broadcast_in_dim %v3584, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3586 = stablehlo.multiply %v3581, %v3583 : tensor<32x384x14x14xf32>
    %v3587 = stablehlo.reduce(%v3586 init: %v3568) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3588 = stablehlo.broadcast_in_dim %v3587, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3589 = stablehlo.multiply %v3583, %v3569 : tensor<32x384x14x14xf32>
    %v3590 = stablehlo.subtract %v3589, %v3585 : tensor<32x384x14x14xf32>
    %v3591 = stablehlo.multiply %v3581, %v3588 : tensor<32x384x14x14xf32>
    %v3592 = stablehlo.subtract %v3590, %v3591 : tensor<32x384x14x14xf32>
    %v3593 = stablehlo.divide %v3580, %v3569 : tensor<32x384x14x14xf32>
    %v3594 = stablehlo.multiply %v3593, %v3592 : tensor<32x384x14x14xf32>
    %v3595 = stablehlo.reshape %v3594 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3596 = stablehlo.reshape %v3595 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3597 = stablehlo.reverse %Wd9, dims = [2, 3] : tensor<384x1x3x3xf32>
    %v3598 = stablehlo.convolution(%v3596, %v3597)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x3x3xf32>) -> tensor<32x384x14x14xf32>
    %v3599 = stablehlo.reshape %v3598 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3600 = stablehlo.reshape %v3599 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3601 = stablehlo.reshape %v736 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3602 = stablehlo.constant dense<0.0> : tensor<32x384x14x14xf32>
    %v3603 = stablehlo.constant dense<6.0> : tensor<32x384x14x14xf32>
    %v3604 = stablehlo.compare GT, %v3601, %v3602 : (tensor<32x384x14x14xf32>, tensor<32x384x14x14xf32>) -> tensor<32x384x14x14xi1>
    %v3605 = stablehlo.compare LT, %v3601, %v3603 : (tensor<32x384x14x14xf32>, tensor<32x384x14x14xf32>) -> tensor<32x384x14x14xi1>
    %v3606 = stablehlo.and %v3604, %v3605 : tensor<32x384x14x14xi1>
    %v3607 = stablehlo.select %v3606, %v3600, %v3602 : tensor<32x384x14x14xi1>, tensor<32x384x14x14xf32>
    %v3608 = stablehlo.reshape %v3607 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3609 = stablehlo.reshape %v3608 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3610 = stablehlo.reshape %v716 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3611 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3612 = stablehlo.constant dense<196.0> : tensor<32x384x14x14xf32>
    %v3613 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3614 = stablehlo.reduce(%v3610 init: %v3611) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3615 = stablehlo.broadcast_in_dim %v3614, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3616 = stablehlo.divide %v3615, %v3612 : tensor<32x384x14x14xf32>
    %v3617 = stablehlo.subtract %v3610, %v3616 : tensor<32x384x14x14xf32>
    %v3618 = stablehlo.multiply %v3617, %v3617 : tensor<32x384x14x14xf32>
    %v3619 = stablehlo.reduce(%v3618 init: %v3611) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3620 = stablehlo.broadcast_in_dim %v3619, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3621 = stablehlo.divide %v3620, %v3612 : tensor<32x384x14x14xf32>
    %v3622 = stablehlo.add %v3621, %v3613 : tensor<32x384x14x14xf32>
    %v3623 = stablehlo.rsqrt %v3622 : tensor<32x384x14x14xf32>
    %v3624 = stablehlo.multiply %v3617, %v3623 : tensor<32x384x14x14xf32>
    %v3625 = stablehlo.broadcast_in_dim %ge9, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3626 = stablehlo.multiply %v3625, %v3609 : tensor<32x384x14x14xf32>
    %v3627 = stablehlo.reduce(%v3626 init: %v3611) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3628 = stablehlo.broadcast_in_dim %v3627, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3629 = stablehlo.multiply %v3624, %v3626 : tensor<32x384x14x14xf32>
    %v3630 = stablehlo.reduce(%v3629 init: %v3611) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3631 = stablehlo.broadcast_in_dim %v3630, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3632 = stablehlo.multiply %v3626, %v3612 : tensor<32x384x14x14xf32>
    %v3633 = stablehlo.subtract %v3632, %v3628 : tensor<32x384x14x14xf32>
    %v3634 = stablehlo.multiply %v3624, %v3631 : tensor<32x384x14x14xf32>
    %v3635 = stablehlo.subtract %v3633, %v3634 : tensor<32x384x14x14xf32>
    %v3636 = stablehlo.divide %v3623, %v3612 : tensor<32x384x14x14xf32>
    %v3637 = stablehlo.multiply %v3636, %v3635 : tensor<32x384x14x14xf32>
    %v3638 = stablehlo.reshape %v3637 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3639 = stablehlo.reshape %v3638 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3640 = stablehlo.transpose %We9, dims = [1, 0, 2, 3] : (tensor<384x64x1x1xf32>) -> tensor<64x384x1x1xf32>
    %v3641 = stablehlo.reverse %v3640, dims = [2, 3] : tensor<64x384x1x1xf32>
    %v3642 = stablehlo.convolution(%v3639, %v3641)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<64x384x1x1xf32>) -> tensor<32x64x14x14xf32>
    %v3643 = stablehlo.reshape %v3642 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v3644 = stablehlo.reshape %v3643 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3645 = stablehlo.reshape %v3413 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3646 = stablehlo.add %v3644, %v3645 : tensor<32x64x14x14xf32>
    %v3647 = stablehlo.reshape %v3646 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v3648 = stablehlo.reshape %v711 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3649 = stablehlo.reshape %v3638 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3650 = stablehlo.transpose %v3648, dims = [1, 0, 2, 3] : (tensor<32x64x14x14xf32>) -> tensor<64x32x14x14xf32>
    %v3651 = stablehlo.transpose %v3649, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3652 = stablehlo.convolution(%v3650, %v3651)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<64x384x1x1xf32>
    %v3653 = stablehlo.transpose %v3652, dims = [1, 0, 2, 3] : (tensor<64x384x1x1xf32>) -> tensor<384x64x1x1xf32>
    %v3654 = stablehlo.constant dense<0.3> : tensor<384x64x1x1xf32>
    %v3655 = stablehlo.multiply %v3653, %v3654 : tensor<384x64x1x1xf32>
    %v3656 = stablehlo.subtract %We9, %v3655 : tensor<384x64x1x1xf32>
    %v3657 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3658 = stablehlo.reshape %v716 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3659 = stablehlo.constant dense<196.0> : tensor<32x384x14x14xf32>
    %v3660 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3661 = stablehlo.reduce(%v3658 init: %v3657) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3662 = stablehlo.broadcast_in_dim %v3661, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3663 = stablehlo.divide %v3662, %v3659 : tensor<32x384x14x14xf32>
    %v3664 = stablehlo.subtract %v3658, %v3663 : tensor<32x384x14x14xf32>
    %v3665 = stablehlo.multiply %v3664, %v3664 : tensor<32x384x14x14xf32>
    %v3666 = stablehlo.reduce(%v3665 init: %v3657) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3667 = stablehlo.broadcast_in_dim %v3666, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3668 = stablehlo.divide %v3667, %v3659 : tensor<32x384x14x14xf32>
    %v3669 = stablehlo.add %v3668, %v3660 : tensor<32x384x14x14xf32>
    %v3670 = stablehlo.rsqrt %v3669 : tensor<32x384x14x14xf32>
    %v3671 = stablehlo.multiply %v3664, %v3670 : tensor<32x384x14x14xf32>
    %v3672 = stablehlo.reshape %v3608 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3673 = stablehlo.multiply %v3672, %v3671 : tensor<32x384x14x14xf32>
    %v3674 = stablehlo.reduce(%v3673 init: %v3657) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3675 = stablehlo.constant dense<0.3> : tensor<384xf32>
    %v3676 = stablehlo.multiply %v3674, %v3675 : tensor<384xf32>
    %v3677 = stablehlo.subtract %ge9, %v3676 : tensor<384xf32>
    %v3678 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3679 = stablehlo.reshape %v3608 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3680 = stablehlo.reduce(%v3679 init: %v3678) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3681 = stablehlo.constant dense<0.3> : tensor<384xf32>
    %v3682 = stablehlo.multiply %v3680, %v3681 : tensor<384xf32>
    %v3683 = stablehlo.subtract %bte9, %v3682 : tensor<384xf32>
    %v3684 = stablehlo.reshape %v742 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3685 = stablehlo.reshape %v3595 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3686 = stablehlo.transpose %v3684, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3687 = stablehlo.transpose %v3685, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3688 = stablehlo.convolution(%v3686, %v3687)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x3x3xf32>
    %v3689 = stablehlo.reshape %v3688 : (tensor<1x384x3x3xf32>) -> tensor<384x1x3x3xf32>
    %v3690 = stablehlo.constant dense<0.3> : tensor<384x1x3x3xf32>
    %v3691 = stablehlo.multiply %v3689, %v3690 : tensor<384x1x3x3xf32>
    %v3692 = stablehlo.subtract %Wd9, %v3691 : tensor<384x1x3x3xf32>
    %v3693 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3694 = stablehlo.reshape %v747 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3695 = stablehlo.constant dense<196.0> : tensor<32x384x14x14xf32>
    %v3696 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3697 = stablehlo.reduce(%v3694 init: %v3693) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3698 = stablehlo.broadcast_in_dim %v3697, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3699 = stablehlo.divide %v3698, %v3695 : tensor<32x384x14x14xf32>
    %v3700 = stablehlo.subtract %v3694, %v3699 : tensor<32x384x14x14xf32>
    %v3701 = stablehlo.multiply %v3700, %v3700 : tensor<32x384x14x14xf32>
    %v3702 = stablehlo.reduce(%v3701 init: %v3693) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3703 = stablehlo.broadcast_in_dim %v3702, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3704 = stablehlo.divide %v3703, %v3695 : tensor<32x384x14x14xf32>
    %v3705 = stablehlo.add %v3704, %v3696 : tensor<32x384x14x14xf32>
    %v3706 = stablehlo.rsqrt %v3705 : tensor<32x384x14x14xf32>
    %v3707 = stablehlo.multiply %v3700, %v3706 : tensor<32x384x14x14xf32>
    %v3708 = stablehlo.reshape %v3565 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3709 = stablehlo.multiply %v3708, %v3707 : tensor<32x384x14x14xf32>
    %v3710 = stablehlo.reduce(%v3709 init: %v3693) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3711 = stablehlo.constant dense<0.3> : tensor<384xf32>
    %v3712 = stablehlo.multiply %v3710, %v3711 : tensor<384xf32>
    %v3713 = stablehlo.subtract %gd9, %v3712 : tensor<384xf32>
    %v3714 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3715 = stablehlo.reshape %v3565 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3716 = stablehlo.reduce(%v3715 init: %v3714) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3717 = stablehlo.constant dense<0.3> : tensor<384xf32>
    %v3718 = stablehlo.multiply %v3716, %v3717 : tensor<384xf32>
    %v3719 = stablehlo.subtract %btd9, %v3718 : tensor<384xf32>
    %v3720 = stablehlo.reshape %v773 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3721 = stablehlo.reshape %v3551 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3722 = stablehlo.transpose %v3720, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3723 = stablehlo.transpose %v3721, dims = [1, 0, 2, 3] : (tensor<32x64x14x14xf32>) -> tensor<64x32x14x14xf32>
    %v3724 = stablehlo.convolution(%v3722, %v3723)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<64x32x14x14xf32>) -> tensor<384x64x1x1xf32>
    %v3725 = stablehlo.transpose %v3724, dims = [1, 0, 2, 3] : (tensor<384x64x1x1xf32>) -> tensor<64x384x1x1xf32>
    %v3726 = stablehlo.constant dense<0.3> : tensor<64x384x1x1xf32>
    %v3727 = stablehlo.multiply %v3725, %v3726 : tensor<64x384x1x1xf32>
    %v3728 = stablehlo.subtract %Wp9, %v3727 : tensor<64x384x1x1xf32>
    %v3729 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3730 = stablehlo.reshape %v778 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3731 = stablehlo.constant dense<196.0> : tensor<32x64x14x14xf32>
    %v3732 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v3733 = stablehlo.reduce(%v3730 init: %v3729) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3734 = stablehlo.broadcast_in_dim %v3733, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v3735 = stablehlo.divide %v3734, %v3731 : tensor<32x64x14x14xf32>
    %v3736 = stablehlo.subtract %v3730, %v3735 : tensor<32x64x14x14xf32>
    %v3737 = stablehlo.multiply %v3736, %v3736 : tensor<32x64x14x14xf32>
    %v3738 = stablehlo.reduce(%v3737 init: %v3729) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3739 = stablehlo.broadcast_in_dim %v3738, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v3740 = stablehlo.divide %v3739, %v3731 : tensor<32x64x14x14xf32>
    %v3741 = stablehlo.add %v3740, %v3732 : tensor<32x64x14x14xf32>
    %v3742 = stablehlo.rsqrt %v3741 : tensor<32x64x14x14xf32>
    %v3743 = stablehlo.multiply %v3736, %v3742 : tensor<32x64x14x14xf32>
    %v3744 = stablehlo.reshape %v3413 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3745 = stablehlo.multiply %v3744, %v3743 : tensor<32x64x14x14xf32>
    %v3746 = stablehlo.reduce(%v3745 init: %v3729) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3747 = stablehlo.constant dense<0.3> : tensor<64xf32>
    %v3748 = stablehlo.multiply %v3746, %v3747 : tensor<64xf32>
    %v3749 = stablehlo.subtract %gp9, %v3748 : tensor<64xf32>
    %v3750 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3751 = stablehlo.reshape %v3413 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3752 = stablehlo.reduce(%v3751 init: %v3750) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3753 = stablehlo.constant dense<0.3> : tensor<64xf32>
    %v3754 = stablehlo.multiply %v3752, %v3753 : tensor<64xf32>
    %v3755 = stablehlo.subtract %btp9, %v3754 : tensor<64xf32>
    %v3756 = stablehlo.reshape %v3647 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3757 = stablehlo.reshape %v687 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3758 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3759 = stablehlo.constant dense<196.0> : tensor<32x64x14x14xf32>
    %v3760 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v3761 = stablehlo.reduce(%v3757 init: %v3758) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3762 = stablehlo.broadcast_in_dim %v3761, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v3763 = stablehlo.divide %v3762, %v3759 : tensor<32x64x14x14xf32>
    %v3764 = stablehlo.subtract %v3757, %v3763 : tensor<32x64x14x14xf32>
    %v3765 = stablehlo.multiply %v3764, %v3764 : tensor<32x64x14x14xf32>
    %v3766 = stablehlo.reduce(%v3765 init: %v3758) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3767 = stablehlo.broadcast_in_dim %v3766, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v3768 = stablehlo.divide %v3767, %v3759 : tensor<32x64x14x14xf32>
    %v3769 = stablehlo.add %v3768, %v3760 : tensor<32x64x14x14xf32>
    %v3770 = stablehlo.rsqrt %v3769 : tensor<32x64x14x14xf32>
    %v3771 = stablehlo.multiply %v3764, %v3770 : tensor<32x64x14x14xf32>
    %v3772 = stablehlo.broadcast_in_dim %gp8, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3773 = stablehlo.multiply %v3772, %v3756 : tensor<32x64x14x14xf32>
    %v3774 = stablehlo.reduce(%v3773 init: %v3758) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3775 = stablehlo.broadcast_in_dim %v3774, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v3776 = stablehlo.multiply %v3771, %v3773 : tensor<32x64x14x14xf32>
    %v3777 = stablehlo.reduce(%v3776 init: %v3758) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3778 = stablehlo.broadcast_in_dim %v3777, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v3779 = stablehlo.multiply %v3773, %v3759 : tensor<32x64x14x14xf32>
    %v3780 = stablehlo.subtract %v3779, %v3775 : tensor<32x64x14x14xf32>
    %v3781 = stablehlo.multiply %v3771, %v3778 : tensor<32x64x14x14xf32>
    %v3782 = stablehlo.subtract %v3780, %v3781 : tensor<32x64x14x14xf32>
    %v3783 = stablehlo.divide %v3770, %v3759 : tensor<32x64x14x14xf32>
    %v3784 = stablehlo.multiply %v3783, %v3782 : tensor<32x64x14x14xf32>
    %v3785 = stablehlo.reshape %v3784 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v3786 = stablehlo.reshape %v3785 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3787 = stablehlo.transpose %Wp8, dims = [1, 0, 2, 3] : (tensor<64x384x1x1xf32>) -> tensor<384x64x1x1xf32>
    %v3788 = stablehlo.reverse %v3787, dims = [2, 3] : tensor<384x64x1x1xf32>
    %v3789 = stablehlo.convolution(%v3786, %v3788)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x14x14xf32>, tensor<384x64x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v3790 = stablehlo.reshape %v3789 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3791 = stablehlo.reshape %v3790 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3792 = stablehlo.reshape %v676 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3793 = stablehlo.constant dense<0.0> : tensor<32x384x14x14xf32>
    %v3794 = stablehlo.constant dense<6.0> : tensor<32x384x14x14xf32>
    %v3795 = stablehlo.compare GT, %v3792, %v3793 : (tensor<32x384x14x14xf32>, tensor<32x384x14x14xf32>) -> tensor<32x384x14x14xi1>
    %v3796 = stablehlo.compare LT, %v3792, %v3794 : (tensor<32x384x14x14xf32>, tensor<32x384x14x14xf32>) -> tensor<32x384x14x14xi1>
    %v3797 = stablehlo.and %v3795, %v3796 : tensor<32x384x14x14xi1>
    %v3798 = stablehlo.select %v3797, %v3791, %v3793 : tensor<32x384x14x14xi1>, tensor<32x384x14x14xf32>
    %v3799 = stablehlo.reshape %v3798 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3800 = stablehlo.reshape %v3799 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3801 = stablehlo.reshape %v656 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3802 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3803 = stablehlo.constant dense<196.0> : tensor<32x384x14x14xf32>
    %v3804 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3805 = stablehlo.reduce(%v3801 init: %v3802) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3806 = stablehlo.broadcast_in_dim %v3805, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3807 = stablehlo.divide %v3806, %v3803 : tensor<32x384x14x14xf32>
    %v3808 = stablehlo.subtract %v3801, %v3807 : tensor<32x384x14x14xf32>
    %v3809 = stablehlo.multiply %v3808, %v3808 : tensor<32x384x14x14xf32>
    %v3810 = stablehlo.reduce(%v3809 init: %v3802) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3811 = stablehlo.broadcast_in_dim %v3810, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3812 = stablehlo.divide %v3811, %v3803 : tensor<32x384x14x14xf32>
    %v3813 = stablehlo.add %v3812, %v3804 : tensor<32x384x14x14xf32>
    %v3814 = stablehlo.rsqrt %v3813 : tensor<32x384x14x14xf32>
    %v3815 = stablehlo.multiply %v3808, %v3814 : tensor<32x384x14x14xf32>
    %v3816 = stablehlo.broadcast_in_dim %gd8, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3817 = stablehlo.multiply %v3816, %v3800 : tensor<32x384x14x14xf32>
    %v3818 = stablehlo.reduce(%v3817 init: %v3802) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3819 = stablehlo.broadcast_in_dim %v3818, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3820 = stablehlo.multiply %v3815, %v3817 : tensor<32x384x14x14xf32>
    %v3821 = stablehlo.reduce(%v3820 init: %v3802) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3822 = stablehlo.broadcast_in_dim %v3821, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3823 = stablehlo.multiply %v3817, %v3803 : tensor<32x384x14x14xf32>
    %v3824 = stablehlo.subtract %v3823, %v3819 : tensor<32x384x14x14xf32>
    %v3825 = stablehlo.multiply %v3815, %v3822 : tensor<32x384x14x14xf32>
    %v3826 = stablehlo.subtract %v3824, %v3825 : tensor<32x384x14x14xf32>
    %v3827 = stablehlo.divide %v3814, %v3803 : tensor<32x384x14x14xf32>
    %v3828 = stablehlo.multiply %v3827, %v3826 : tensor<32x384x14x14xf32>
    %v3829 = stablehlo.reshape %v3828 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3830 = stablehlo.reshape %v3829 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3831 = stablehlo.reverse %Wd8, dims = [2, 3] : tensor<384x1x3x3xf32>
    %v3832 = stablehlo.convolution(%v3830, %v3831)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x3x3xf32>) -> tensor<32x384x14x14xf32>
    %v3833 = stablehlo.reshape %v3832 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3834 = stablehlo.reshape %v3833 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3835 = stablehlo.reshape %v645 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3836 = stablehlo.constant dense<0.0> : tensor<32x384x14x14xf32>
    %v3837 = stablehlo.constant dense<6.0> : tensor<32x384x14x14xf32>
    %v3838 = stablehlo.compare GT, %v3835, %v3836 : (tensor<32x384x14x14xf32>, tensor<32x384x14x14xf32>) -> tensor<32x384x14x14xi1>
    %v3839 = stablehlo.compare LT, %v3835, %v3837 : (tensor<32x384x14x14xf32>, tensor<32x384x14x14xf32>) -> tensor<32x384x14x14xi1>
    %v3840 = stablehlo.and %v3838, %v3839 : tensor<32x384x14x14xi1>
    %v3841 = stablehlo.select %v3840, %v3834, %v3836 : tensor<32x384x14x14xi1>, tensor<32x384x14x14xf32>
    %v3842 = stablehlo.reshape %v3841 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3843 = stablehlo.reshape %v3842 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3844 = stablehlo.reshape %v625 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3845 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3846 = stablehlo.constant dense<196.0> : tensor<32x384x14x14xf32>
    %v3847 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3848 = stablehlo.reduce(%v3844 init: %v3845) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3849 = stablehlo.broadcast_in_dim %v3848, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3850 = stablehlo.divide %v3849, %v3846 : tensor<32x384x14x14xf32>
    %v3851 = stablehlo.subtract %v3844, %v3850 : tensor<32x384x14x14xf32>
    %v3852 = stablehlo.multiply %v3851, %v3851 : tensor<32x384x14x14xf32>
    %v3853 = stablehlo.reduce(%v3852 init: %v3845) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3854 = stablehlo.broadcast_in_dim %v3853, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3855 = stablehlo.divide %v3854, %v3846 : tensor<32x384x14x14xf32>
    %v3856 = stablehlo.add %v3855, %v3847 : tensor<32x384x14x14xf32>
    %v3857 = stablehlo.rsqrt %v3856 : tensor<32x384x14x14xf32>
    %v3858 = stablehlo.multiply %v3851, %v3857 : tensor<32x384x14x14xf32>
    %v3859 = stablehlo.broadcast_in_dim %ge8, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3860 = stablehlo.multiply %v3859, %v3843 : tensor<32x384x14x14xf32>
    %v3861 = stablehlo.reduce(%v3860 init: %v3845) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3862 = stablehlo.broadcast_in_dim %v3861, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3863 = stablehlo.multiply %v3858, %v3860 : tensor<32x384x14x14xf32>
    %v3864 = stablehlo.reduce(%v3863 init: %v3845) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3865 = stablehlo.broadcast_in_dim %v3864, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3866 = stablehlo.multiply %v3860, %v3846 : tensor<32x384x14x14xf32>
    %v3867 = stablehlo.subtract %v3866, %v3862 : tensor<32x384x14x14xf32>
    %v3868 = stablehlo.multiply %v3858, %v3865 : tensor<32x384x14x14xf32>
    %v3869 = stablehlo.subtract %v3867, %v3868 : tensor<32x384x14x14xf32>
    %v3870 = stablehlo.divide %v3857, %v3846 : tensor<32x384x14x14xf32>
    %v3871 = stablehlo.multiply %v3870, %v3869 : tensor<32x384x14x14xf32>
    %v3872 = stablehlo.reshape %v3871 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3873 = stablehlo.reshape %v3872 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3874 = stablehlo.transpose %We8, dims = [1, 0, 2, 3] : (tensor<384x64x1x1xf32>) -> tensor<64x384x1x1xf32>
    %v3875 = stablehlo.reverse %v3874, dims = [2, 3] : tensor<64x384x1x1xf32>
    %v3876 = stablehlo.convolution(%v3873, %v3875)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<64x384x1x1xf32>) -> tensor<32x64x14x14xf32>
    %v3877 = stablehlo.reshape %v3876 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v3878 = stablehlo.reshape %v3877 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3879 = stablehlo.reshape %v3647 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3880 = stablehlo.add %v3878, %v3879 : tensor<32x64x14x14xf32>
    %v3881 = stablehlo.reshape %v3880 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v3882 = stablehlo.reshape %v620 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3883 = stablehlo.reshape %v3872 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3884 = stablehlo.transpose %v3882, dims = [1, 0, 2, 3] : (tensor<32x64x14x14xf32>) -> tensor<64x32x14x14xf32>
    %v3885 = stablehlo.transpose %v3883, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3886 = stablehlo.convolution(%v3884, %v3885)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<64x384x1x1xf32>
    %v3887 = stablehlo.transpose %v3886, dims = [1, 0, 2, 3] : (tensor<64x384x1x1xf32>) -> tensor<384x64x1x1xf32>
    %v3888 = stablehlo.constant dense<0.3> : tensor<384x64x1x1xf32>
    %v3889 = stablehlo.multiply %v3887, %v3888 : tensor<384x64x1x1xf32>
    %v3890 = stablehlo.subtract %We8, %v3889 : tensor<384x64x1x1xf32>
    %v3891 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3892 = stablehlo.reshape %v625 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3893 = stablehlo.constant dense<196.0> : tensor<32x384x14x14xf32>
    %v3894 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3895 = stablehlo.reduce(%v3892 init: %v3891) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3896 = stablehlo.broadcast_in_dim %v3895, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3897 = stablehlo.divide %v3896, %v3893 : tensor<32x384x14x14xf32>
    %v3898 = stablehlo.subtract %v3892, %v3897 : tensor<32x384x14x14xf32>
    %v3899 = stablehlo.multiply %v3898, %v3898 : tensor<32x384x14x14xf32>
    %v3900 = stablehlo.reduce(%v3899 init: %v3891) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3901 = stablehlo.broadcast_in_dim %v3900, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3902 = stablehlo.divide %v3901, %v3893 : tensor<32x384x14x14xf32>
    %v3903 = stablehlo.add %v3902, %v3894 : tensor<32x384x14x14xf32>
    %v3904 = stablehlo.rsqrt %v3903 : tensor<32x384x14x14xf32>
    %v3905 = stablehlo.multiply %v3898, %v3904 : tensor<32x384x14x14xf32>
    %v3906 = stablehlo.reshape %v3842 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3907 = stablehlo.multiply %v3906, %v3905 : tensor<32x384x14x14xf32>
    %v3908 = stablehlo.reduce(%v3907 init: %v3891) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3909 = stablehlo.constant dense<0.3> : tensor<384xf32>
    %v3910 = stablehlo.multiply %v3908, %v3909 : tensor<384xf32>
    %v3911 = stablehlo.subtract %ge8, %v3910 : tensor<384xf32>
    %v3912 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3913 = stablehlo.reshape %v3842 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3914 = stablehlo.reduce(%v3913 init: %v3912) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3915 = stablehlo.constant dense<0.3> : tensor<384xf32>
    %v3916 = stablehlo.multiply %v3914, %v3915 : tensor<384xf32>
    %v3917 = stablehlo.subtract %bte8, %v3916 : tensor<384xf32>
    %v3918 = stablehlo.reshape %v651 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3919 = stablehlo.reshape %v3829 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3920 = stablehlo.transpose %v3918, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3921 = stablehlo.transpose %v3919, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3922 = stablehlo.convolution(%v3920, %v3921)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x3x3xf32>
    %v3923 = stablehlo.reshape %v3922 : (tensor<1x384x3x3xf32>) -> tensor<384x1x3x3xf32>
    %v3924 = stablehlo.constant dense<0.3> : tensor<384x1x3x3xf32>
    %v3925 = stablehlo.multiply %v3923, %v3924 : tensor<384x1x3x3xf32>
    %v3926 = stablehlo.subtract %Wd8, %v3925 : tensor<384x1x3x3xf32>
    %v3927 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3928 = stablehlo.reshape %v656 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3929 = stablehlo.constant dense<196.0> : tensor<32x384x14x14xf32>
    %v3930 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3931 = stablehlo.reduce(%v3928 init: %v3927) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3932 = stablehlo.broadcast_in_dim %v3931, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3933 = stablehlo.divide %v3932, %v3929 : tensor<32x384x14x14xf32>
    %v3934 = stablehlo.subtract %v3928, %v3933 : tensor<32x384x14x14xf32>
    %v3935 = stablehlo.multiply %v3934, %v3934 : tensor<32x384x14x14xf32>
    %v3936 = stablehlo.reduce(%v3935 init: %v3927) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3937 = stablehlo.broadcast_in_dim %v3936, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3938 = stablehlo.divide %v3937, %v3929 : tensor<32x384x14x14xf32>
    %v3939 = stablehlo.add %v3938, %v3930 : tensor<32x384x14x14xf32>
    %v3940 = stablehlo.rsqrt %v3939 : tensor<32x384x14x14xf32>
    %v3941 = stablehlo.multiply %v3934, %v3940 : tensor<32x384x14x14xf32>
    %v3942 = stablehlo.reshape %v3799 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3943 = stablehlo.multiply %v3942, %v3941 : tensor<32x384x14x14xf32>
    %v3944 = stablehlo.reduce(%v3943 init: %v3927) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3945 = stablehlo.constant dense<0.3> : tensor<384xf32>
    %v3946 = stablehlo.multiply %v3944, %v3945 : tensor<384xf32>
    %v3947 = stablehlo.subtract %gd8, %v3946 : tensor<384xf32>
    %v3948 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3949 = stablehlo.reshape %v3799 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3950 = stablehlo.reduce(%v3949 init: %v3948) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3951 = stablehlo.constant dense<0.3> : tensor<384xf32>
    %v3952 = stablehlo.multiply %v3950, %v3951 : tensor<384xf32>
    %v3953 = stablehlo.subtract %btd8, %v3952 : tensor<384xf32>
    %v3954 = stablehlo.reshape %v682 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3955 = stablehlo.reshape %v3785 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3956 = stablehlo.transpose %v3954, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3957 = stablehlo.transpose %v3955, dims = [1, 0, 2, 3] : (tensor<32x64x14x14xf32>) -> tensor<64x32x14x14xf32>
    %v3958 = stablehlo.convolution(%v3956, %v3957)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<64x32x14x14xf32>) -> tensor<384x64x1x1xf32>
    %v3959 = stablehlo.transpose %v3958, dims = [1, 0, 2, 3] : (tensor<384x64x1x1xf32>) -> tensor<64x384x1x1xf32>
    %v3960 = stablehlo.constant dense<0.3> : tensor<64x384x1x1xf32>
    %v3961 = stablehlo.multiply %v3959, %v3960 : tensor<64x384x1x1xf32>
    %v3962 = stablehlo.subtract %Wp8, %v3961 : tensor<64x384x1x1xf32>
    %v3963 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3964 = stablehlo.reshape %v687 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3965 = stablehlo.constant dense<196.0> : tensor<32x64x14x14xf32>
    %v3966 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v3967 = stablehlo.reduce(%v3964 init: %v3963) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3968 = stablehlo.broadcast_in_dim %v3967, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v3969 = stablehlo.divide %v3968, %v3965 : tensor<32x64x14x14xf32>
    %v3970 = stablehlo.subtract %v3964, %v3969 : tensor<32x64x14x14xf32>
    %v3971 = stablehlo.multiply %v3970, %v3970 : tensor<32x64x14x14xf32>
    %v3972 = stablehlo.reduce(%v3971 init: %v3963) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3973 = stablehlo.broadcast_in_dim %v3972, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v3974 = stablehlo.divide %v3973, %v3965 : tensor<32x64x14x14xf32>
    %v3975 = stablehlo.add %v3974, %v3966 : tensor<32x64x14x14xf32>
    %v3976 = stablehlo.rsqrt %v3975 : tensor<32x64x14x14xf32>
    %v3977 = stablehlo.multiply %v3970, %v3976 : tensor<32x64x14x14xf32>
    %v3978 = stablehlo.reshape %v3647 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3979 = stablehlo.multiply %v3978, %v3977 : tensor<32x64x14x14xf32>
    %v3980 = stablehlo.reduce(%v3979 init: %v3963) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3981 = stablehlo.constant dense<0.3> : tensor<64xf32>
    %v3982 = stablehlo.multiply %v3980, %v3981 : tensor<64xf32>
    %v3983 = stablehlo.subtract %gp8, %v3982 : tensor<64xf32>
    %v3984 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3985 = stablehlo.reshape %v3647 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3986 = stablehlo.reduce(%v3985 init: %v3984) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3987 = stablehlo.constant dense<0.3> : tensor<64xf32>
    %v3988 = stablehlo.multiply %v3986, %v3987 : tensor<64xf32>
    %v3989 = stablehlo.subtract %btp8, %v3988 : tensor<64xf32>
    %v3990 = stablehlo.reshape %v3881 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3991 = stablehlo.reshape %v600 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3992 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3993 = stablehlo.constant dense<196.0> : tensor<32x64x14x14xf32>
    %v3994 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v3995 = stablehlo.reduce(%v3991 init: %v3992) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3996 = stablehlo.broadcast_in_dim %v3995, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v3997 = stablehlo.divide %v3996, %v3993 : tensor<32x64x14x14xf32>
    %v3998 = stablehlo.subtract %v3991, %v3997 : tensor<32x64x14x14xf32>
    %v3999 = stablehlo.multiply %v3998, %v3998 : tensor<32x64x14x14xf32>
    %v4000 = stablehlo.reduce(%v3999 init: %v3992) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v4001 = stablehlo.broadcast_in_dim %v4000, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v4002 = stablehlo.divide %v4001, %v3993 : tensor<32x64x14x14xf32>
    %v4003 = stablehlo.add %v4002, %v3994 : tensor<32x64x14x14xf32>
    %v4004 = stablehlo.rsqrt %v4003 : tensor<32x64x14x14xf32>
    %v4005 = stablehlo.multiply %v3998, %v4004 : tensor<32x64x14x14xf32>
    %v4006 = stablehlo.broadcast_in_dim %gp7, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v4007 = stablehlo.multiply %v4006, %v3990 : tensor<32x64x14x14xf32>
    %v4008 = stablehlo.reduce(%v4007 init: %v3992) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v4009 = stablehlo.broadcast_in_dim %v4008, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v4010 = stablehlo.multiply %v4005, %v4007 : tensor<32x64x14x14xf32>
    %v4011 = stablehlo.reduce(%v4010 init: %v3992) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v4012 = stablehlo.broadcast_in_dim %v4011, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v4013 = stablehlo.multiply %v4007, %v3993 : tensor<32x64x14x14xf32>
    %v4014 = stablehlo.subtract %v4013, %v4009 : tensor<32x64x14x14xf32>
    %v4015 = stablehlo.multiply %v4005, %v4012 : tensor<32x64x14x14xf32>
    %v4016 = stablehlo.subtract %v4014, %v4015 : tensor<32x64x14x14xf32>
    %v4017 = stablehlo.divide %v4004, %v3993 : tensor<32x64x14x14xf32>
    %v4018 = stablehlo.multiply %v4017, %v4016 : tensor<32x64x14x14xf32>
    %v4019 = stablehlo.reshape %v4018 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v4020 = stablehlo.reshape %v4019 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v4021 = stablehlo.transpose %Wp7, dims = [1, 0, 2, 3] : (tensor<64x192x1x1xf32>) -> tensor<192x64x1x1xf32>
    %v4022 = stablehlo.reverse %v4021, dims = [2, 3] : tensor<192x64x1x1xf32>
    %v4023 = stablehlo.convolution(%v4020, %v4022)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x14x14xf32>, tensor<192x64x1x1xf32>) -> tensor<32x192x14x14xf32>
    %v4024 = stablehlo.reshape %v4023 : (tensor<32x192x14x14xf32>) -> tensor<32x37632xf32>
    %v4025 = stablehlo.reshape %v4024 : (tensor<32x37632xf32>) -> tensor<32x192x14x14xf32>
    %v4026 = stablehlo.reshape %v589 : (tensor<32x37632xf32>) -> tensor<32x192x14x14xf32>
    %v4027 = stablehlo.constant dense<0.0> : tensor<32x192x14x14xf32>
    %v4028 = stablehlo.constant dense<6.0> : tensor<32x192x14x14xf32>
    %v4029 = stablehlo.compare GT, %v4026, %v4027 : (tensor<32x192x14x14xf32>, tensor<32x192x14x14xf32>) -> tensor<32x192x14x14xi1>
    %v4030 = stablehlo.compare LT, %v4026, %v4028 : (tensor<32x192x14x14xf32>, tensor<32x192x14x14xf32>) -> tensor<32x192x14x14xi1>
    %v4031 = stablehlo.and %v4029, %v4030 : tensor<32x192x14x14xi1>
    %v4032 = stablehlo.select %v4031, %v4025, %v4027 : tensor<32x192x14x14xi1>, tensor<32x192x14x14xf32>
    %v4033 = stablehlo.reshape %v4032 : (tensor<32x192x14x14xf32>) -> tensor<32x37632xf32>
    %v4034 = stablehlo.reshape %v4033 : (tensor<32x37632xf32>) -> tensor<32x192x14x14xf32>
    %v4035 = stablehlo.reshape %v569 : (tensor<32x37632xf32>) -> tensor<32x192x14x14xf32>
    %v4036 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4037 = stablehlo.constant dense<196.0> : tensor<32x192x14x14xf32>
    %v4038 = stablehlo.constant dense<1.0e-5> : tensor<32x192x14x14xf32>
    %v4039 = stablehlo.reduce(%v4035 init: %v4036) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4040 = stablehlo.broadcast_in_dim %v4039, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x14x14xf32>
    %v4041 = stablehlo.divide %v4040, %v4037 : tensor<32x192x14x14xf32>
    %v4042 = stablehlo.subtract %v4035, %v4041 : tensor<32x192x14x14xf32>
    %v4043 = stablehlo.multiply %v4042, %v4042 : tensor<32x192x14x14xf32>
    %v4044 = stablehlo.reduce(%v4043 init: %v4036) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4045 = stablehlo.broadcast_in_dim %v4044, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x14x14xf32>
    %v4046 = stablehlo.divide %v4045, %v4037 : tensor<32x192x14x14xf32>
    %v4047 = stablehlo.add %v4046, %v4038 : tensor<32x192x14x14xf32>
    %v4048 = stablehlo.rsqrt %v4047 : tensor<32x192x14x14xf32>
    %v4049 = stablehlo.multiply %v4042, %v4048 : tensor<32x192x14x14xf32>
    %v4050 = stablehlo.broadcast_in_dim %gd7, dims = [1] : (tensor<192xf32>) -> tensor<32x192x14x14xf32>
    %v4051 = stablehlo.multiply %v4050, %v4034 : tensor<32x192x14x14xf32>
    %v4052 = stablehlo.reduce(%v4051 init: %v4036) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4053 = stablehlo.broadcast_in_dim %v4052, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x14x14xf32>
    %v4054 = stablehlo.multiply %v4049, %v4051 : tensor<32x192x14x14xf32>
    %v4055 = stablehlo.reduce(%v4054 init: %v4036) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4056 = stablehlo.broadcast_in_dim %v4055, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x14x14xf32>
    %v4057 = stablehlo.multiply %v4051, %v4037 : tensor<32x192x14x14xf32>
    %v4058 = stablehlo.subtract %v4057, %v4053 : tensor<32x192x14x14xf32>
    %v4059 = stablehlo.multiply %v4049, %v4056 : tensor<32x192x14x14xf32>
    %v4060 = stablehlo.subtract %v4058, %v4059 : tensor<32x192x14x14xf32>
    %v4061 = stablehlo.divide %v4048, %v4037 : tensor<32x192x14x14xf32>
    %v4062 = stablehlo.multiply %v4061, %v4060 : tensor<32x192x14x14xf32>
    %v4063 = stablehlo.reshape %v4062 : (tensor<32x192x14x14xf32>) -> tensor<32x37632xf32>
    %v4064 = stablehlo.reshape %v4063 : (tensor<32x37632xf32>) -> tensor<32x192x14x14xf32>
    %v4065 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4066 = stablehlo.pad %v4064, %v4065, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<32x192x28x28xf32>
    %v4067 = stablehlo.reverse %Wd7, dims = [2, 3] : tensor<192x1x3x3xf32>
    %v4068 = stablehlo.convolution(%v4066, %v4067)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x3x3xf32>) -> tensor<32x192x28x28xf32>
    %v4069 = stablehlo.reshape %v4068 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v4070 = stablehlo.reshape %v4069 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4071 = stablehlo.reshape %v558 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4072 = stablehlo.constant dense<0.0> : tensor<32x192x28x28xf32>
    %v4073 = stablehlo.constant dense<6.0> : tensor<32x192x28x28xf32>
    %v4074 = stablehlo.compare GT, %v4071, %v4072 : (tensor<32x192x28x28xf32>, tensor<32x192x28x28xf32>) -> tensor<32x192x28x28xi1>
    %v4075 = stablehlo.compare LT, %v4071, %v4073 : (tensor<32x192x28x28xf32>, tensor<32x192x28x28xf32>) -> tensor<32x192x28x28xi1>
    %v4076 = stablehlo.and %v4074, %v4075 : tensor<32x192x28x28xi1>
    %v4077 = stablehlo.select %v4076, %v4070, %v4072 : tensor<32x192x28x28xi1>, tensor<32x192x28x28xf32>
    %v4078 = stablehlo.reshape %v4077 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v4079 = stablehlo.reshape %v4078 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4080 = stablehlo.reshape %v538 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4081 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4082 = stablehlo.constant dense<784.0> : tensor<32x192x28x28xf32>
    %v4083 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v4084 = stablehlo.reduce(%v4080 init: %v4081) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4085 = stablehlo.broadcast_in_dim %v4084, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v4086 = stablehlo.divide %v4085, %v4082 : tensor<32x192x28x28xf32>
    %v4087 = stablehlo.subtract %v4080, %v4086 : tensor<32x192x28x28xf32>
    %v4088 = stablehlo.multiply %v4087, %v4087 : tensor<32x192x28x28xf32>
    %v4089 = stablehlo.reduce(%v4088 init: %v4081) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4090 = stablehlo.broadcast_in_dim %v4089, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v4091 = stablehlo.divide %v4090, %v4082 : tensor<32x192x28x28xf32>
    %v4092 = stablehlo.add %v4091, %v4083 : tensor<32x192x28x28xf32>
    %v4093 = stablehlo.rsqrt %v4092 : tensor<32x192x28x28xf32>
    %v4094 = stablehlo.multiply %v4087, %v4093 : tensor<32x192x28x28xf32>
    %v4095 = stablehlo.broadcast_in_dim %ge7, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v4096 = stablehlo.multiply %v4095, %v4079 : tensor<32x192x28x28xf32>
    %v4097 = stablehlo.reduce(%v4096 init: %v4081) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4098 = stablehlo.broadcast_in_dim %v4097, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v4099 = stablehlo.multiply %v4094, %v4096 : tensor<32x192x28x28xf32>
    %v4100 = stablehlo.reduce(%v4099 init: %v4081) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4101 = stablehlo.broadcast_in_dim %v4100, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v4102 = stablehlo.multiply %v4096, %v4082 : tensor<32x192x28x28xf32>
    %v4103 = stablehlo.subtract %v4102, %v4098 : tensor<32x192x28x28xf32>
    %v4104 = stablehlo.multiply %v4094, %v4101 : tensor<32x192x28x28xf32>
    %v4105 = stablehlo.subtract %v4103, %v4104 : tensor<32x192x28x28xf32>
    %v4106 = stablehlo.divide %v4093, %v4082 : tensor<32x192x28x28xf32>
    %v4107 = stablehlo.multiply %v4106, %v4105 : tensor<32x192x28x28xf32>
    %v4108 = stablehlo.reshape %v4107 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v4109 = stablehlo.reshape %v4108 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4110 = stablehlo.transpose %We7, dims = [1, 0, 2, 3] : (tensor<192x32x1x1xf32>) -> tensor<32x192x1x1xf32>
    %v4111 = stablehlo.reverse %v4110, dims = [2, 3] : tensor<32x192x1x1xf32>
    %v4112 = stablehlo.convolution(%v4109, %v4111)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<32x192x1x1xf32>) -> tensor<32x32x28x28xf32>
    %v4113 = stablehlo.reshape %v4112 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v4114 = stablehlo.reshape %v533 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4115 = stablehlo.reshape %v4108 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4116 = stablehlo.transpose %v4114, dims = [1, 0, 2, 3] : (tensor<32x32x28x28xf32>) -> tensor<32x32x28x28xf32>
    %v4117 = stablehlo.transpose %v4115, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v4118 = stablehlo.convolution(%v4116, %v4117)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<32x192x1x1xf32>
    %v4119 = stablehlo.transpose %v4118, dims = [1, 0, 2, 3] : (tensor<32x192x1x1xf32>) -> tensor<192x32x1x1xf32>
    %v4120 = stablehlo.constant dense<0.3> : tensor<192x32x1x1xf32>
    %v4121 = stablehlo.multiply %v4119, %v4120 : tensor<192x32x1x1xf32>
    %v4122 = stablehlo.subtract %We7, %v4121 : tensor<192x32x1x1xf32>
    %v4123 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4124 = stablehlo.reshape %v538 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4125 = stablehlo.constant dense<784.0> : tensor<32x192x28x28xf32>
    %v4126 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v4127 = stablehlo.reduce(%v4124 init: %v4123) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4128 = stablehlo.broadcast_in_dim %v4127, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v4129 = stablehlo.divide %v4128, %v4125 : tensor<32x192x28x28xf32>
    %v4130 = stablehlo.subtract %v4124, %v4129 : tensor<32x192x28x28xf32>
    %v4131 = stablehlo.multiply %v4130, %v4130 : tensor<32x192x28x28xf32>
    %v4132 = stablehlo.reduce(%v4131 init: %v4123) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4133 = stablehlo.broadcast_in_dim %v4132, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v4134 = stablehlo.divide %v4133, %v4125 : tensor<32x192x28x28xf32>
    %v4135 = stablehlo.add %v4134, %v4126 : tensor<32x192x28x28xf32>
    %v4136 = stablehlo.rsqrt %v4135 : tensor<32x192x28x28xf32>
    %v4137 = stablehlo.multiply %v4130, %v4136 : tensor<32x192x28x28xf32>
    %v4138 = stablehlo.reshape %v4078 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4139 = stablehlo.multiply %v4138, %v4137 : tensor<32x192x28x28xf32>
    %v4140 = stablehlo.reduce(%v4139 init: %v4123) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4141 = stablehlo.constant dense<0.3> : tensor<192xf32>
    %v4142 = stablehlo.multiply %v4140, %v4141 : tensor<192xf32>
    %v4143 = stablehlo.subtract %ge7, %v4142 : tensor<192xf32>
    %v4144 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4145 = stablehlo.reshape %v4078 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4146 = stablehlo.reduce(%v4145 init: %v4144) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4147 = stablehlo.constant dense<0.3> : tensor<192xf32>
    %v4148 = stablehlo.multiply %v4146, %v4147 : tensor<192xf32>
    %v4149 = stablehlo.subtract %bte7, %v4148 : tensor<192xf32>
    %v4150 = stablehlo.reshape %v564 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4151 = stablehlo.reshape %v4063 : (tensor<32x37632xf32>) -> tensor<32x192x14x14xf32>
    %v4152 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4153 = stablehlo.pad %v4151, %v4152, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<32x192x28x28xf32>
    %v4154 = stablehlo.transpose %v4150, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v4155 = stablehlo.transpose %v4153, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v4156 = stablehlo.convolution(%v4154, %v4155)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 192 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<1x192x3x3xf32>
    %v4157 = stablehlo.reshape %v4156 : (tensor<1x192x3x3xf32>) -> tensor<192x1x3x3xf32>
    %v4158 = stablehlo.constant dense<0.3> : tensor<192x1x3x3xf32>
    %v4159 = stablehlo.multiply %v4157, %v4158 : tensor<192x1x3x3xf32>
    %v4160 = stablehlo.subtract %Wd7, %v4159 : tensor<192x1x3x3xf32>
    %v4161 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4162 = stablehlo.reshape %v569 : (tensor<32x37632xf32>) -> tensor<32x192x14x14xf32>
    %v4163 = stablehlo.constant dense<196.0> : tensor<32x192x14x14xf32>
    %v4164 = stablehlo.constant dense<1.0e-5> : tensor<32x192x14x14xf32>
    %v4165 = stablehlo.reduce(%v4162 init: %v4161) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4166 = stablehlo.broadcast_in_dim %v4165, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x14x14xf32>
    %v4167 = stablehlo.divide %v4166, %v4163 : tensor<32x192x14x14xf32>
    %v4168 = stablehlo.subtract %v4162, %v4167 : tensor<32x192x14x14xf32>
    %v4169 = stablehlo.multiply %v4168, %v4168 : tensor<32x192x14x14xf32>
    %v4170 = stablehlo.reduce(%v4169 init: %v4161) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4171 = stablehlo.broadcast_in_dim %v4170, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x14x14xf32>
    %v4172 = stablehlo.divide %v4171, %v4163 : tensor<32x192x14x14xf32>
    %v4173 = stablehlo.add %v4172, %v4164 : tensor<32x192x14x14xf32>
    %v4174 = stablehlo.rsqrt %v4173 : tensor<32x192x14x14xf32>
    %v4175 = stablehlo.multiply %v4168, %v4174 : tensor<32x192x14x14xf32>
    %v4176 = stablehlo.reshape %v4033 : (tensor<32x37632xf32>) -> tensor<32x192x14x14xf32>
    %v4177 = stablehlo.multiply %v4176, %v4175 : tensor<32x192x14x14xf32>
    %v4178 = stablehlo.reduce(%v4177 init: %v4161) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<192xf32>
    %v4179 = stablehlo.constant dense<0.3> : tensor<192xf32>
    %v4180 = stablehlo.multiply %v4178, %v4179 : tensor<192xf32>
    %v4181 = stablehlo.subtract %gd7, %v4180 : tensor<192xf32>
    %v4182 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4183 = stablehlo.reshape %v4033 : (tensor<32x37632xf32>) -> tensor<32x192x14x14xf32>
    %v4184 = stablehlo.reduce(%v4183 init: %v4182) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<192xf32>
    %v4185 = stablehlo.constant dense<0.3> : tensor<192xf32>
    %v4186 = stablehlo.multiply %v4184, %v4185 : tensor<192xf32>
    %v4187 = stablehlo.subtract %btd7, %v4186 : tensor<192xf32>
    %v4188 = stablehlo.reshape %v595 : (tensor<32x37632xf32>) -> tensor<32x192x14x14xf32>
    %v4189 = stablehlo.reshape %v4019 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v4190 = stablehlo.transpose %v4188, dims = [1, 0, 2, 3] : (tensor<32x192x14x14xf32>) -> tensor<192x32x14x14xf32>
    %v4191 = stablehlo.transpose %v4189, dims = [1, 0, 2, 3] : (tensor<32x64x14x14xf32>) -> tensor<64x32x14x14xf32>
    %v4192 = stablehlo.convolution(%v4190, %v4191)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x14x14xf32>, tensor<64x32x14x14xf32>) -> tensor<192x64x1x1xf32>
    %v4193 = stablehlo.transpose %v4192, dims = [1, 0, 2, 3] : (tensor<192x64x1x1xf32>) -> tensor<64x192x1x1xf32>
    %v4194 = stablehlo.constant dense<0.3> : tensor<64x192x1x1xf32>
    %v4195 = stablehlo.multiply %v4193, %v4194 : tensor<64x192x1x1xf32>
    %v4196 = stablehlo.subtract %Wp7, %v4195 : tensor<64x192x1x1xf32>
    %v4197 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4198 = stablehlo.reshape %v600 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v4199 = stablehlo.constant dense<196.0> : tensor<32x64x14x14xf32>
    %v4200 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v4201 = stablehlo.reduce(%v4198 init: %v4197) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v4202 = stablehlo.broadcast_in_dim %v4201, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v4203 = stablehlo.divide %v4202, %v4199 : tensor<32x64x14x14xf32>
    %v4204 = stablehlo.subtract %v4198, %v4203 : tensor<32x64x14x14xf32>
    %v4205 = stablehlo.multiply %v4204, %v4204 : tensor<32x64x14x14xf32>
    %v4206 = stablehlo.reduce(%v4205 init: %v4197) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v4207 = stablehlo.broadcast_in_dim %v4206, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v4208 = stablehlo.divide %v4207, %v4199 : tensor<32x64x14x14xf32>
    %v4209 = stablehlo.add %v4208, %v4200 : tensor<32x64x14x14xf32>
    %v4210 = stablehlo.rsqrt %v4209 : tensor<32x64x14x14xf32>
    %v4211 = stablehlo.multiply %v4204, %v4210 : tensor<32x64x14x14xf32>
    %v4212 = stablehlo.reshape %v3881 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v4213 = stablehlo.multiply %v4212, %v4211 : tensor<32x64x14x14xf32>
    %v4214 = stablehlo.reduce(%v4213 init: %v4197) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v4215 = stablehlo.constant dense<0.3> : tensor<64xf32>
    %v4216 = stablehlo.multiply %v4214, %v4215 : tensor<64xf32>
    %v4217 = stablehlo.subtract %gp7, %v4216 : tensor<64xf32>
    %v4218 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4219 = stablehlo.reshape %v3881 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v4220 = stablehlo.reduce(%v4219 init: %v4218) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v4221 = stablehlo.constant dense<0.3> : tensor<64xf32>
    %v4222 = stablehlo.multiply %v4220, %v4221 : tensor<64xf32>
    %v4223 = stablehlo.subtract %btp7, %v4222 : tensor<64xf32>
    %v4224 = stablehlo.reshape %v4113 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4225 = stablehlo.reshape %v509 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4226 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4227 = stablehlo.constant dense<784.0> : tensor<32x32x28x28xf32>
    %v4228 = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %v4229 = stablehlo.reduce(%v4225 init: %v4226) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v4230 = stablehlo.broadcast_in_dim %v4229, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v4231 = stablehlo.divide %v4230, %v4227 : tensor<32x32x28x28xf32>
    %v4232 = stablehlo.subtract %v4225, %v4231 : tensor<32x32x28x28xf32>
    %v4233 = stablehlo.multiply %v4232, %v4232 : tensor<32x32x28x28xf32>
    %v4234 = stablehlo.reduce(%v4233 init: %v4226) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v4235 = stablehlo.broadcast_in_dim %v4234, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v4236 = stablehlo.divide %v4235, %v4227 : tensor<32x32x28x28xf32>
    %v4237 = stablehlo.add %v4236, %v4228 : tensor<32x32x28x28xf32>
    %v4238 = stablehlo.rsqrt %v4237 : tensor<32x32x28x28xf32>
    %v4239 = stablehlo.multiply %v4232, %v4238 : tensor<32x32x28x28xf32>
    %v4240 = stablehlo.broadcast_in_dim %gp6, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v4241 = stablehlo.multiply %v4240, %v4224 : tensor<32x32x28x28xf32>
    %v4242 = stablehlo.reduce(%v4241 init: %v4226) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v4243 = stablehlo.broadcast_in_dim %v4242, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v4244 = stablehlo.multiply %v4239, %v4241 : tensor<32x32x28x28xf32>
    %v4245 = stablehlo.reduce(%v4244 init: %v4226) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v4246 = stablehlo.broadcast_in_dim %v4245, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v4247 = stablehlo.multiply %v4241, %v4227 : tensor<32x32x28x28xf32>
    %v4248 = stablehlo.subtract %v4247, %v4243 : tensor<32x32x28x28xf32>
    %v4249 = stablehlo.multiply %v4239, %v4246 : tensor<32x32x28x28xf32>
    %v4250 = stablehlo.subtract %v4248, %v4249 : tensor<32x32x28x28xf32>
    %v4251 = stablehlo.divide %v4238, %v4227 : tensor<32x32x28x28xf32>
    %v4252 = stablehlo.multiply %v4251, %v4250 : tensor<32x32x28x28xf32>
    %v4253 = stablehlo.reshape %v4252 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v4254 = stablehlo.reshape %v4253 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4255 = stablehlo.transpose %Wp6, dims = [1, 0, 2, 3] : (tensor<32x192x1x1xf32>) -> tensor<192x32x1x1xf32>
    %v4256 = stablehlo.reverse %v4255, dims = [2, 3] : tensor<192x32x1x1xf32>
    %v4257 = stablehlo.convolution(%v4254, %v4256)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<192x32x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v4258 = stablehlo.reshape %v4257 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v4259 = stablehlo.reshape %v4258 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4260 = stablehlo.reshape %v498 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4261 = stablehlo.constant dense<0.0> : tensor<32x192x28x28xf32>
    %v4262 = stablehlo.constant dense<6.0> : tensor<32x192x28x28xf32>
    %v4263 = stablehlo.compare GT, %v4260, %v4261 : (tensor<32x192x28x28xf32>, tensor<32x192x28x28xf32>) -> tensor<32x192x28x28xi1>
    %v4264 = stablehlo.compare LT, %v4260, %v4262 : (tensor<32x192x28x28xf32>, tensor<32x192x28x28xf32>) -> tensor<32x192x28x28xi1>
    %v4265 = stablehlo.and %v4263, %v4264 : tensor<32x192x28x28xi1>
    %v4266 = stablehlo.select %v4265, %v4259, %v4261 : tensor<32x192x28x28xi1>, tensor<32x192x28x28xf32>
    %v4267 = stablehlo.reshape %v4266 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v4268 = stablehlo.reshape %v4267 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4269 = stablehlo.reshape %v478 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4270 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4271 = stablehlo.constant dense<784.0> : tensor<32x192x28x28xf32>
    %v4272 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v4273 = stablehlo.reduce(%v4269 init: %v4270) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4274 = stablehlo.broadcast_in_dim %v4273, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v4275 = stablehlo.divide %v4274, %v4271 : tensor<32x192x28x28xf32>
    %v4276 = stablehlo.subtract %v4269, %v4275 : tensor<32x192x28x28xf32>
    %v4277 = stablehlo.multiply %v4276, %v4276 : tensor<32x192x28x28xf32>
    %v4278 = stablehlo.reduce(%v4277 init: %v4270) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4279 = stablehlo.broadcast_in_dim %v4278, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v4280 = stablehlo.divide %v4279, %v4271 : tensor<32x192x28x28xf32>
    %v4281 = stablehlo.add %v4280, %v4272 : tensor<32x192x28x28xf32>
    %v4282 = stablehlo.rsqrt %v4281 : tensor<32x192x28x28xf32>
    %v4283 = stablehlo.multiply %v4276, %v4282 : tensor<32x192x28x28xf32>
    %v4284 = stablehlo.broadcast_in_dim %gd6, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v4285 = stablehlo.multiply %v4284, %v4268 : tensor<32x192x28x28xf32>
    %v4286 = stablehlo.reduce(%v4285 init: %v4270) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4287 = stablehlo.broadcast_in_dim %v4286, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v4288 = stablehlo.multiply %v4283, %v4285 : tensor<32x192x28x28xf32>
    %v4289 = stablehlo.reduce(%v4288 init: %v4270) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4290 = stablehlo.broadcast_in_dim %v4289, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v4291 = stablehlo.multiply %v4285, %v4271 : tensor<32x192x28x28xf32>
    %v4292 = stablehlo.subtract %v4291, %v4287 : tensor<32x192x28x28xf32>
    %v4293 = stablehlo.multiply %v4283, %v4290 : tensor<32x192x28x28xf32>
    %v4294 = stablehlo.subtract %v4292, %v4293 : tensor<32x192x28x28xf32>
    %v4295 = stablehlo.divide %v4282, %v4271 : tensor<32x192x28x28xf32>
    %v4296 = stablehlo.multiply %v4295, %v4294 : tensor<32x192x28x28xf32>
    %v4297 = stablehlo.reshape %v4296 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v4298 = stablehlo.reshape %v4297 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4299 = stablehlo.reverse %Wd6, dims = [2, 3] : tensor<192x1x3x3xf32>
    %v4300 = stablehlo.convolution(%v4298, %v4299)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x3x3xf32>) -> tensor<32x192x28x28xf32>
    %v4301 = stablehlo.reshape %v4300 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v4302 = stablehlo.reshape %v4301 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4303 = stablehlo.reshape %v467 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4304 = stablehlo.constant dense<0.0> : tensor<32x192x28x28xf32>
    %v4305 = stablehlo.constant dense<6.0> : tensor<32x192x28x28xf32>
    %v4306 = stablehlo.compare GT, %v4303, %v4304 : (tensor<32x192x28x28xf32>, tensor<32x192x28x28xf32>) -> tensor<32x192x28x28xi1>
    %v4307 = stablehlo.compare LT, %v4303, %v4305 : (tensor<32x192x28x28xf32>, tensor<32x192x28x28xf32>) -> tensor<32x192x28x28xi1>
    %v4308 = stablehlo.and %v4306, %v4307 : tensor<32x192x28x28xi1>
    %v4309 = stablehlo.select %v4308, %v4302, %v4304 : tensor<32x192x28x28xi1>, tensor<32x192x28x28xf32>
    %v4310 = stablehlo.reshape %v4309 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v4311 = stablehlo.reshape %v4310 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4312 = stablehlo.reshape %v447 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4313 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4314 = stablehlo.constant dense<784.0> : tensor<32x192x28x28xf32>
    %v4315 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v4316 = stablehlo.reduce(%v4312 init: %v4313) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4317 = stablehlo.broadcast_in_dim %v4316, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v4318 = stablehlo.divide %v4317, %v4314 : tensor<32x192x28x28xf32>
    %v4319 = stablehlo.subtract %v4312, %v4318 : tensor<32x192x28x28xf32>
    %v4320 = stablehlo.multiply %v4319, %v4319 : tensor<32x192x28x28xf32>
    %v4321 = stablehlo.reduce(%v4320 init: %v4313) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4322 = stablehlo.broadcast_in_dim %v4321, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v4323 = stablehlo.divide %v4322, %v4314 : tensor<32x192x28x28xf32>
    %v4324 = stablehlo.add %v4323, %v4315 : tensor<32x192x28x28xf32>
    %v4325 = stablehlo.rsqrt %v4324 : tensor<32x192x28x28xf32>
    %v4326 = stablehlo.multiply %v4319, %v4325 : tensor<32x192x28x28xf32>
    %v4327 = stablehlo.broadcast_in_dim %ge6, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v4328 = stablehlo.multiply %v4327, %v4311 : tensor<32x192x28x28xf32>
    %v4329 = stablehlo.reduce(%v4328 init: %v4313) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4330 = stablehlo.broadcast_in_dim %v4329, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v4331 = stablehlo.multiply %v4326, %v4328 : tensor<32x192x28x28xf32>
    %v4332 = stablehlo.reduce(%v4331 init: %v4313) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4333 = stablehlo.broadcast_in_dim %v4332, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v4334 = stablehlo.multiply %v4328, %v4314 : tensor<32x192x28x28xf32>
    %v4335 = stablehlo.subtract %v4334, %v4330 : tensor<32x192x28x28xf32>
    %v4336 = stablehlo.multiply %v4326, %v4333 : tensor<32x192x28x28xf32>
    %v4337 = stablehlo.subtract %v4335, %v4336 : tensor<32x192x28x28xf32>
    %v4338 = stablehlo.divide %v4325, %v4314 : tensor<32x192x28x28xf32>
    %v4339 = stablehlo.multiply %v4338, %v4337 : tensor<32x192x28x28xf32>
    %v4340 = stablehlo.reshape %v4339 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v4341 = stablehlo.reshape %v4340 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4342 = stablehlo.transpose %We6, dims = [1, 0, 2, 3] : (tensor<192x32x1x1xf32>) -> tensor<32x192x1x1xf32>
    %v4343 = stablehlo.reverse %v4342, dims = [2, 3] : tensor<32x192x1x1xf32>
    %v4344 = stablehlo.convolution(%v4341, %v4343)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<32x192x1x1xf32>) -> tensor<32x32x28x28xf32>
    %v4345 = stablehlo.reshape %v4344 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v4346 = stablehlo.reshape %v4345 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4347 = stablehlo.reshape %v4113 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4348 = stablehlo.add %v4346, %v4347 : tensor<32x32x28x28xf32>
    %v4349 = stablehlo.reshape %v4348 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v4350 = stablehlo.reshape %v442 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4351 = stablehlo.reshape %v4340 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4352 = stablehlo.transpose %v4350, dims = [1, 0, 2, 3] : (tensor<32x32x28x28xf32>) -> tensor<32x32x28x28xf32>
    %v4353 = stablehlo.transpose %v4351, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v4354 = stablehlo.convolution(%v4352, %v4353)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<32x192x1x1xf32>
    %v4355 = stablehlo.transpose %v4354, dims = [1, 0, 2, 3] : (tensor<32x192x1x1xf32>) -> tensor<192x32x1x1xf32>
    %v4356 = stablehlo.constant dense<0.3> : tensor<192x32x1x1xf32>
    %v4357 = stablehlo.multiply %v4355, %v4356 : tensor<192x32x1x1xf32>
    %v4358 = stablehlo.subtract %We6, %v4357 : tensor<192x32x1x1xf32>
    %v4359 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4360 = stablehlo.reshape %v447 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4361 = stablehlo.constant dense<784.0> : tensor<32x192x28x28xf32>
    %v4362 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v4363 = stablehlo.reduce(%v4360 init: %v4359) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4364 = stablehlo.broadcast_in_dim %v4363, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v4365 = stablehlo.divide %v4364, %v4361 : tensor<32x192x28x28xf32>
    %v4366 = stablehlo.subtract %v4360, %v4365 : tensor<32x192x28x28xf32>
    %v4367 = stablehlo.multiply %v4366, %v4366 : tensor<32x192x28x28xf32>
    %v4368 = stablehlo.reduce(%v4367 init: %v4359) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4369 = stablehlo.broadcast_in_dim %v4368, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v4370 = stablehlo.divide %v4369, %v4361 : tensor<32x192x28x28xf32>
    %v4371 = stablehlo.add %v4370, %v4362 : tensor<32x192x28x28xf32>
    %v4372 = stablehlo.rsqrt %v4371 : tensor<32x192x28x28xf32>
    %v4373 = stablehlo.multiply %v4366, %v4372 : tensor<32x192x28x28xf32>
    %v4374 = stablehlo.reshape %v4310 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4375 = stablehlo.multiply %v4374, %v4373 : tensor<32x192x28x28xf32>
    %v4376 = stablehlo.reduce(%v4375 init: %v4359) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4377 = stablehlo.constant dense<0.3> : tensor<192xf32>
    %v4378 = stablehlo.multiply %v4376, %v4377 : tensor<192xf32>
    %v4379 = stablehlo.subtract %ge6, %v4378 : tensor<192xf32>
    %v4380 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4381 = stablehlo.reshape %v4310 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4382 = stablehlo.reduce(%v4381 init: %v4380) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4383 = stablehlo.constant dense<0.3> : tensor<192xf32>
    %v4384 = stablehlo.multiply %v4382, %v4383 : tensor<192xf32>
    %v4385 = stablehlo.subtract %bte6, %v4384 : tensor<192xf32>
    %v4386 = stablehlo.reshape %v473 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4387 = stablehlo.reshape %v4297 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4388 = stablehlo.transpose %v4386, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v4389 = stablehlo.transpose %v4387, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v4390 = stablehlo.convolution(%v4388, %v4389)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 192 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<1x192x3x3xf32>
    %v4391 = stablehlo.reshape %v4390 : (tensor<1x192x3x3xf32>) -> tensor<192x1x3x3xf32>
    %v4392 = stablehlo.constant dense<0.3> : tensor<192x1x3x3xf32>
    %v4393 = stablehlo.multiply %v4391, %v4392 : tensor<192x1x3x3xf32>
    %v4394 = stablehlo.subtract %Wd6, %v4393 : tensor<192x1x3x3xf32>
    %v4395 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4396 = stablehlo.reshape %v478 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4397 = stablehlo.constant dense<784.0> : tensor<32x192x28x28xf32>
    %v4398 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v4399 = stablehlo.reduce(%v4396 init: %v4395) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4400 = stablehlo.broadcast_in_dim %v4399, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v4401 = stablehlo.divide %v4400, %v4397 : tensor<32x192x28x28xf32>
    %v4402 = stablehlo.subtract %v4396, %v4401 : tensor<32x192x28x28xf32>
    %v4403 = stablehlo.multiply %v4402, %v4402 : tensor<32x192x28x28xf32>
    %v4404 = stablehlo.reduce(%v4403 init: %v4395) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4405 = stablehlo.broadcast_in_dim %v4404, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v4406 = stablehlo.divide %v4405, %v4397 : tensor<32x192x28x28xf32>
    %v4407 = stablehlo.add %v4406, %v4398 : tensor<32x192x28x28xf32>
    %v4408 = stablehlo.rsqrt %v4407 : tensor<32x192x28x28xf32>
    %v4409 = stablehlo.multiply %v4402, %v4408 : tensor<32x192x28x28xf32>
    %v4410 = stablehlo.reshape %v4267 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4411 = stablehlo.multiply %v4410, %v4409 : tensor<32x192x28x28xf32>
    %v4412 = stablehlo.reduce(%v4411 init: %v4395) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4413 = stablehlo.constant dense<0.3> : tensor<192xf32>
    %v4414 = stablehlo.multiply %v4412, %v4413 : tensor<192xf32>
    %v4415 = stablehlo.subtract %gd6, %v4414 : tensor<192xf32>
    %v4416 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4417 = stablehlo.reshape %v4267 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4418 = stablehlo.reduce(%v4417 init: %v4416) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4419 = stablehlo.constant dense<0.3> : tensor<192xf32>
    %v4420 = stablehlo.multiply %v4418, %v4419 : tensor<192xf32>
    %v4421 = stablehlo.subtract %btd6, %v4420 : tensor<192xf32>
    %v4422 = stablehlo.reshape %v504 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4423 = stablehlo.reshape %v4253 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4424 = stablehlo.transpose %v4422, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v4425 = stablehlo.transpose %v4423, dims = [1, 0, 2, 3] : (tensor<32x32x28x28xf32>) -> tensor<32x32x28x28xf32>
    %v4426 = stablehlo.convolution(%v4424, %v4425)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<32x32x28x28xf32>) -> tensor<192x32x1x1xf32>
    %v4427 = stablehlo.transpose %v4426, dims = [1, 0, 2, 3] : (tensor<192x32x1x1xf32>) -> tensor<32x192x1x1xf32>
    %v4428 = stablehlo.constant dense<0.3> : tensor<32x192x1x1xf32>
    %v4429 = stablehlo.multiply %v4427, %v4428 : tensor<32x192x1x1xf32>
    %v4430 = stablehlo.subtract %Wp6, %v4429 : tensor<32x192x1x1xf32>
    %v4431 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4432 = stablehlo.reshape %v509 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4433 = stablehlo.constant dense<784.0> : tensor<32x32x28x28xf32>
    %v4434 = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %v4435 = stablehlo.reduce(%v4432 init: %v4431) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v4436 = stablehlo.broadcast_in_dim %v4435, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v4437 = stablehlo.divide %v4436, %v4433 : tensor<32x32x28x28xf32>
    %v4438 = stablehlo.subtract %v4432, %v4437 : tensor<32x32x28x28xf32>
    %v4439 = stablehlo.multiply %v4438, %v4438 : tensor<32x32x28x28xf32>
    %v4440 = stablehlo.reduce(%v4439 init: %v4431) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v4441 = stablehlo.broadcast_in_dim %v4440, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v4442 = stablehlo.divide %v4441, %v4433 : tensor<32x32x28x28xf32>
    %v4443 = stablehlo.add %v4442, %v4434 : tensor<32x32x28x28xf32>
    %v4444 = stablehlo.rsqrt %v4443 : tensor<32x32x28x28xf32>
    %v4445 = stablehlo.multiply %v4438, %v4444 : tensor<32x32x28x28xf32>
    %v4446 = stablehlo.reshape %v4113 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4447 = stablehlo.multiply %v4446, %v4445 : tensor<32x32x28x28xf32>
    %v4448 = stablehlo.reduce(%v4447 init: %v4431) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v4449 = stablehlo.constant dense<0.3> : tensor<32xf32>
    %v4450 = stablehlo.multiply %v4448, %v4449 : tensor<32xf32>
    %v4451 = stablehlo.subtract %gp6, %v4450 : tensor<32xf32>
    %v4452 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4453 = stablehlo.reshape %v4113 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4454 = stablehlo.reduce(%v4453 init: %v4452) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v4455 = stablehlo.constant dense<0.3> : tensor<32xf32>
    %v4456 = stablehlo.multiply %v4454, %v4455 : tensor<32xf32>
    %v4457 = stablehlo.subtract %btp6, %v4456 : tensor<32xf32>
    %v4458 = stablehlo.reshape %v4349 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4459 = stablehlo.reshape %v418 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4460 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4461 = stablehlo.constant dense<784.0> : tensor<32x32x28x28xf32>
    %v4462 = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %v4463 = stablehlo.reduce(%v4459 init: %v4460) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v4464 = stablehlo.broadcast_in_dim %v4463, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v4465 = stablehlo.divide %v4464, %v4461 : tensor<32x32x28x28xf32>
    %v4466 = stablehlo.subtract %v4459, %v4465 : tensor<32x32x28x28xf32>
    %v4467 = stablehlo.multiply %v4466, %v4466 : tensor<32x32x28x28xf32>
    %v4468 = stablehlo.reduce(%v4467 init: %v4460) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v4469 = stablehlo.broadcast_in_dim %v4468, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v4470 = stablehlo.divide %v4469, %v4461 : tensor<32x32x28x28xf32>
    %v4471 = stablehlo.add %v4470, %v4462 : tensor<32x32x28x28xf32>
    %v4472 = stablehlo.rsqrt %v4471 : tensor<32x32x28x28xf32>
    %v4473 = stablehlo.multiply %v4466, %v4472 : tensor<32x32x28x28xf32>
    %v4474 = stablehlo.broadcast_in_dim %gp5, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v4475 = stablehlo.multiply %v4474, %v4458 : tensor<32x32x28x28xf32>
    %v4476 = stablehlo.reduce(%v4475 init: %v4460) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v4477 = stablehlo.broadcast_in_dim %v4476, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v4478 = stablehlo.multiply %v4473, %v4475 : tensor<32x32x28x28xf32>
    %v4479 = stablehlo.reduce(%v4478 init: %v4460) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v4480 = stablehlo.broadcast_in_dim %v4479, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v4481 = stablehlo.multiply %v4475, %v4461 : tensor<32x32x28x28xf32>
    %v4482 = stablehlo.subtract %v4481, %v4477 : tensor<32x32x28x28xf32>
    %v4483 = stablehlo.multiply %v4473, %v4480 : tensor<32x32x28x28xf32>
    %v4484 = stablehlo.subtract %v4482, %v4483 : tensor<32x32x28x28xf32>
    %v4485 = stablehlo.divide %v4472, %v4461 : tensor<32x32x28x28xf32>
    %v4486 = stablehlo.multiply %v4485, %v4484 : tensor<32x32x28x28xf32>
    %v4487 = stablehlo.reshape %v4486 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v4488 = stablehlo.reshape %v4487 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4489 = stablehlo.transpose %Wp5, dims = [1, 0, 2, 3] : (tensor<32x192x1x1xf32>) -> tensor<192x32x1x1xf32>
    %v4490 = stablehlo.reverse %v4489, dims = [2, 3] : tensor<192x32x1x1xf32>
    %v4491 = stablehlo.convolution(%v4488, %v4490)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<192x32x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v4492 = stablehlo.reshape %v4491 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v4493 = stablehlo.reshape %v4492 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4494 = stablehlo.reshape %v407 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4495 = stablehlo.constant dense<0.0> : tensor<32x192x28x28xf32>
    %v4496 = stablehlo.constant dense<6.0> : tensor<32x192x28x28xf32>
    %v4497 = stablehlo.compare GT, %v4494, %v4495 : (tensor<32x192x28x28xf32>, tensor<32x192x28x28xf32>) -> tensor<32x192x28x28xi1>
    %v4498 = stablehlo.compare LT, %v4494, %v4496 : (tensor<32x192x28x28xf32>, tensor<32x192x28x28xf32>) -> tensor<32x192x28x28xi1>
    %v4499 = stablehlo.and %v4497, %v4498 : tensor<32x192x28x28xi1>
    %v4500 = stablehlo.select %v4499, %v4493, %v4495 : tensor<32x192x28x28xi1>, tensor<32x192x28x28xf32>
    %v4501 = stablehlo.reshape %v4500 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v4502 = stablehlo.reshape %v4501 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4503 = stablehlo.reshape %v387 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4504 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4505 = stablehlo.constant dense<784.0> : tensor<32x192x28x28xf32>
    %v4506 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v4507 = stablehlo.reduce(%v4503 init: %v4504) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4508 = stablehlo.broadcast_in_dim %v4507, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v4509 = stablehlo.divide %v4508, %v4505 : tensor<32x192x28x28xf32>
    %v4510 = stablehlo.subtract %v4503, %v4509 : tensor<32x192x28x28xf32>
    %v4511 = stablehlo.multiply %v4510, %v4510 : tensor<32x192x28x28xf32>
    %v4512 = stablehlo.reduce(%v4511 init: %v4504) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4513 = stablehlo.broadcast_in_dim %v4512, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v4514 = stablehlo.divide %v4513, %v4505 : tensor<32x192x28x28xf32>
    %v4515 = stablehlo.add %v4514, %v4506 : tensor<32x192x28x28xf32>
    %v4516 = stablehlo.rsqrt %v4515 : tensor<32x192x28x28xf32>
    %v4517 = stablehlo.multiply %v4510, %v4516 : tensor<32x192x28x28xf32>
    %v4518 = stablehlo.broadcast_in_dim %gd5, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v4519 = stablehlo.multiply %v4518, %v4502 : tensor<32x192x28x28xf32>
    %v4520 = stablehlo.reduce(%v4519 init: %v4504) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4521 = stablehlo.broadcast_in_dim %v4520, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v4522 = stablehlo.multiply %v4517, %v4519 : tensor<32x192x28x28xf32>
    %v4523 = stablehlo.reduce(%v4522 init: %v4504) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4524 = stablehlo.broadcast_in_dim %v4523, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v4525 = stablehlo.multiply %v4519, %v4505 : tensor<32x192x28x28xf32>
    %v4526 = stablehlo.subtract %v4525, %v4521 : tensor<32x192x28x28xf32>
    %v4527 = stablehlo.multiply %v4517, %v4524 : tensor<32x192x28x28xf32>
    %v4528 = stablehlo.subtract %v4526, %v4527 : tensor<32x192x28x28xf32>
    %v4529 = stablehlo.divide %v4516, %v4505 : tensor<32x192x28x28xf32>
    %v4530 = stablehlo.multiply %v4529, %v4528 : tensor<32x192x28x28xf32>
    %v4531 = stablehlo.reshape %v4530 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v4532 = stablehlo.reshape %v4531 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4533 = stablehlo.reverse %Wd5, dims = [2, 3] : tensor<192x1x3x3xf32>
    %v4534 = stablehlo.convolution(%v4532, %v4533)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x3x3xf32>) -> tensor<32x192x28x28xf32>
    %v4535 = stablehlo.reshape %v4534 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v4536 = stablehlo.reshape %v4535 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4537 = stablehlo.reshape %v376 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4538 = stablehlo.constant dense<0.0> : tensor<32x192x28x28xf32>
    %v4539 = stablehlo.constant dense<6.0> : tensor<32x192x28x28xf32>
    %v4540 = stablehlo.compare GT, %v4537, %v4538 : (tensor<32x192x28x28xf32>, tensor<32x192x28x28xf32>) -> tensor<32x192x28x28xi1>
    %v4541 = stablehlo.compare LT, %v4537, %v4539 : (tensor<32x192x28x28xf32>, tensor<32x192x28x28xf32>) -> tensor<32x192x28x28xi1>
    %v4542 = stablehlo.and %v4540, %v4541 : tensor<32x192x28x28xi1>
    %v4543 = stablehlo.select %v4542, %v4536, %v4538 : tensor<32x192x28x28xi1>, tensor<32x192x28x28xf32>
    %v4544 = stablehlo.reshape %v4543 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v4545 = stablehlo.reshape %v4544 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4546 = stablehlo.reshape %v356 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4547 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4548 = stablehlo.constant dense<784.0> : tensor<32x192x28x28xf32>
    %v4549 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v4550 = stablehlo.reduce(%v4546 init: %v4547) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4551 = stablehlo.broadcast_in_dim %v4550, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v4552 = stablehlo.divide %v4551, %v4548 : tensor<32x192x28x28xf32>
    %v4553 = stablehlo.subtract %v4546, %v4552 : tensor<32x192x28x28xf32>
    %v4554 = stablehlo.multiply %v4553, %v4553 : tensor<32x192x28x28xf32>
    %v4555 = stablehlo.reduce(%v4554 init: %v4547) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4556 = stablehlo.broadcast_in_dim %v4555, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v4557 = stablehlo.divide %v4556, %v4548 : tensor<32x192x28x28xf32>
    %v4558 = stablehlo.add %v4557, %v4549 : tensor<32x192x28x28xf32>
    %v4559 = stablehlo.rsqrt %v4558 : tensor<32x192x28x28xf32>
    %v4560 = stablehlo.multiply %v4553, %v4559 : tensor<32x192x28x28xf32>
    %v4561 = stablehlo.broadcast_in_dim %ge5, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v4562 = stablehlo.multiply %v4561, %v4545 : tensor<32x192x28x28xf32>
    %v4563 = stablehlo.reduce(%v4562 init: %v4547) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4564 = stablehlo.broadcast_in_dim %v4563, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v4565 = stablehlo.multiply %v4560, %v4562 : tensor<32x192x28x28xf32>
    %v4566 = stablehlo.reduce(%v4565 init: %v4547) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4567 = stablehlo.broadcast_in_dim %v4566, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v4568 = stablehlo.multiply %v4562, %v4548 : tensor<32x192x28x28xf32>
    %v4569 = stablehlo.subtract %v4568, %v4564 : tensor<32x192x28x28xf32>
    %v4570 = stablehlo.multiply %v4560, %v4567 : tensor<32x192x28x28xf32>
    %v4571 = stablehlo.subtract %v4569, %v4570 : tensor<32x192x28x28xf32>
    %v4572 = stablehlo.divide %v4559, %v4548 : tensor<32x192x28x28xf32>
    %v4573 = stablehlo.multiply %v4572, %v4571 : tensor<32x192x28x28xf32>
    %v4574 = stablehlo.reshape %v4573 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v4575 = stablehlo.reshape %v4574 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4576 = stablehlo.transpose %We5, dims = [1, 0, 2, 3] : (tensor<192x32x1x1xf32>) -> tensor<32x192x1x1xf32>
    %v4577 = stablehlo.reverse %v4576, dims = [2, 3] : tensor<32x192x1x1xf32>
    %v4578 = stablehlo.convolution(%v4575, %v4577)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<32x192x1x1xf32>) -> tensor<32x32x28x28xf32>
    %v4579 = stablehlo.reshape %v4578 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v4580 = stablehlo.reshape %v4579 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4581 = stablehlo.reshape %v4349 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4582 = stablehlo.add %v4580, %v4581 : tensor<32x32x28x28xf32>
    %v4583 = stablehlo.reshape %v4582 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v4584 = stablehlo.reshape %v351 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4585 = stablehlo.reshape %v4574 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4586 = stablehlo.transpose %v4584, dims = [1, 0, 2, 3] : (tensor<32x32x28x28xf32>) -> tensor<32x32x28x28xf32>
    %v4587 = stablehlo.transpose %v4585, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v4588 = stablehlo.convolution(%v4586, %v4587)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<32x192x1x1xf32>
    %v4589 = stablehlo.transpose %v4588, dims = [1, 0, 2, 3] : (tensor<32x192x1x1xf32>) -> tensor<192x32x1x1xf32>
    %v4590 = stablehlo.constant dense<0.3> : tensor<192x32x1x1xf32>
    %v4591 = stablehlo.multiply %v4589, %v4590 : tensor<192x32x1x1xf32>
    %v4592 = stablehlo.subtract %We5, %v4591 : tensor<192x32x1x1xf32>
    %v4593 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4594 = stablehlo.reshape %v356 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4595 = stablehlo.constant dense<784.0> : tensor<32x192x28x28xf32>
    %v4596 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v4597 = stablehlo.reduce(%v4594 init: %v4593) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4598 = stablehlo.broadcast_in_dim %v4597, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v4599 = stablehlo.divide %v4598, %v4595 : tensor<32x192x28x28xf32>
    %v4600 = stablehlo.subtract %v4594, %v4599 : tensor<32x192x28x28xf32>
    %v4601 = stablehlo.multiply %v4600, %v4600 : tensor<32x192x28x28xf32>
    %v4602 = stablehlo.reduce(%v4601 init: %v4593) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4603 = stablehlo.broadcast_in_dim %v4602, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v4604 = stablehlo.divide %v4603, %v4595 : tensor<32x192x28x28xf32>
    %v4605 = stablehlo.add %v4604, %v4596 : tensor<32x192x28x28xf32>
    %v4606 = stablehlo.rsqrt %v4605 : tensor<32x192x28x28xf32>
    %v4607 = stablehlo.multiply %v4600, %v4606 : tensor<32x192x28x28xf32>
    %v4608 = stablehlo.reshape %v4544 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4609 = stablehlo.multiply %v4608, %v4607 : tensor<32x192x28x28xf32>
    %v4610 = stablehlo.reduce(%v4609 init: %v4593) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4611 = stablehlo.constant dense<0.3> : tensor<192xf32>
    %v4612 = stablehlo.multiply %v4610, %v4611 : tensor<192xf32>
    %v4613 = stablehlo.subtract %ge5, %v4612 : tensor<192xf32>
    %v4614 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4615 = stablehlo.reshape %v4544 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4616 = stablehlo.reduce(%v4615 init: %v4614) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4617 = stablehlo.constant dense<0.3> : tensor<192xf32>
    %v4618 = stablehlo.multiply %v4616, %v4617 : tensor<192xf32>
    %v4619 = stablehlo.subtract %bte5, %v4618 : tensor<192xf32>
    %v4620 = stablehlo.reshape %v382 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4621 = stablehlo.reshape %v4531 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4622 = stablehlo.transpose %v4620, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v4623 = stablehlo.transpose %v4621, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v4624 = stablehlo.convolution(%v4622, %v4623)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 192 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<1x192x3x3xf32>
    %v4625 = stablehlo.reshape %v4624 : (tensor<1x192x3x3xf32>) -> tensor<192x1x3x3xf32>
    %v4626 = stablehlo.constant dense<0.3> : tensor<192x1x3x3xf32>
    %v4627 = stablehlo.multiply %v4625, %v4626 : tensor<192x1x3x3xf32>
    %v4628 = stablehlo.subtract %Wd5, %v4627 : tensor<192x1x3x3xf32>
    %v4629 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4630 = stablehlo.reshape %v387 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4631 = stablehlo.constant dense<784.0> : tensor<32x192x28x28xf32>
    %v4632 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v4633 = stablehlo.reduce(%v4630 init: %v4629) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4634 = stablehlo.broadcast_in_dim %v4633, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v4635 = stablehlo.divide %v4634, %v4631 : tensor<32x192x28x28xf32>
    %v4636 = stablehlo.subtract %v4630, %v4635 : tensor<32x192x28x28xf32>
    %v4637 = stablehlo.multiply %v4636, %v4636 : tensor<32x192x28x28xf32>
    %v4638 = stablehlo.reduce(%v4637 init: %v4629) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4639 = stablehlo.broadcast_in_dim %v4638, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v4640 = stablehlo.divide %v4639, %v4631 : tensor<32x192x28x28xf32>
    %v4641 = stablehlo.add %v4640, %v4632 : tensor<32x192x28x28xf32>
    %v4642 = stablehlo.rsqrt %v4641 : tensor<32x192x28x28xf32>
    %v4643 = stablehlo.multiply %v4636, %v4642 : tensor<32x192x28x28xf32>
    %v4644 = stablehlo.reshape %v4501 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4645 = stablehlo.multiply %v4644, %v4643 : tensor<32x192x28x28xf32>
    %v4646 = stablehlo.reduce(%v4645 init: %v4629) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4647 = stablehlo.constant dense<0.3> : tensor<192xf32>
    %v4648 = stablehlo.multiply %v4646, %v4647 : tensor<192xf32>
    %v4649 = stablehlo.subtract %gd5, %v4648 : tensor<192xf32>
    %v4650 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4651 = stablehlo.reshape %v4501 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4652 = stablehlo.reduce(%v4651 init: %v4650) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4653 = stablehlo.constant dense<0.3> : tensor<192xf32>
    %v4654 = stablehlo.multiply %v4652, %v4653 : tensor<192xf32>
    %v4655 = stablehlo.subtract %btd5, %v4654 : tensor<192xf32>
    %v4656 = stablehlo.reshape %v413 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4657 = stablehlo.reshape %v4487 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4658 = stablehlo.transpose %v4656, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v4659 = stablehlo.transpose %v4657, dims = [1, 0, 2, 3] : (tensor<32x32x28x28xf32>) -> tensor<32x32x28x28xf32>
    %v4660 = stablehlo.convolution(%v4658, %v4659)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<32x32x28x28xf32>) -> tensor<192x32x1x1xf32>
    %v4661 = stablehlo.transpose %v4660, dims = [1, 0, 2, 3] : (tensor<192x32x1x1xf32>) -> tensor<32x192x1x1xf32>
    %v4662 = stablehlo.constant dense<0.3> : tensor<32x192x1x1xf32>
    %v4663 = stablehlo.multiply %v4661, %v4662 : tensor<32x192x1x1xf32>
    %v4664 = stablehlo.subtract %Wp5, %v4663 : tensor<32x192x1x1xf32>
    %v4665 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4666 = stablehlo.reshape %v418 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4667 = stablehlo.constant dense<784.0> : tensor<32x32x28x28xf32>
    %v4668 = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %v4669 = stablehlo.reduce(%v4666 init: %v4665) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v4670 = stablehlo.broadcast_in_dim %v4669, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v4671 = stablehlo.divide %v4670, %v4667 : tensor<32x32x28x28xf32>
    %v4672 = stablehlo.subtract %v4666, %v4671 : tensor<32x32x28x28xf32>
    %v4673 = stablehlo.multiply %v4672, %v4672 : tensor<32x32x28x28xf32>
    %v4674 = stablehlo.reduce(%v4673 init: %v4665) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v4675 = stablehlo.broadcast_in_dim %v4674, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v4676 = stablehlo.divide %v4675, %v4667 : tensor<32x32x28x28xf32>
    %v4677 = stablehlo.add %v4676, %v4668 : tensor<32x32x28x28xf32>
    %v4678 = stablehlo.rsqrt %v4677 : tensor<32x32x28x28xf32>
    %v4679 = stablehlo.multiply %v4672, %v4678 : tensor<32x32x28x28xf32>
    %v4680 = stablehlo.reshape %v4349 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4681 = stablehlo.multiply %v4680, %v4679 : tensor<32x32x28x28xf32>
    %v4682 = stablehlo.reduce(%v4681 init: %v4665) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v4683 = stablehlo.constant dense<0.3> : tensor<32xf32>
    %v4684 = stablehlo.multiply %v4682, %v4683 : tensor<32xf32>
    %v4685 = stablehlo.subtract %gp5, %v4684 : tensor<32xf32>
    %v4686 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4687 = stablehlo.reshape %v4349 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4688 = stablehlo.reduce(%v4687 init: %v4686) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v4689 = stablehlo.constant dense<0.3> : tensor<32xf32>
    %v4690 = stablehlo.multiply %v4688, %v4689 : tensor<32xf32>
    %v4691 = stablehlo.subtract %btp5, %v4690 : tensor<32xf32>
    %v4692 = stablehlo.reshape %v4583 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4693 = stablehlo.reshape %v331 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4694 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4695 = stablehlo.constant dense<784.0> : tensor<32x32x28x28xf32>
    %v4696 = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %v4697 = stablehlo.reduce(%v4693 init: %v4694) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v4698 = stablehlo.broadcast_in_dim %v4697, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v4699 = stablehlo.divide %v4698, %v4695 : tensor<32x32x28x28xf32>
    %v4700 = stablehlo.subtract %v4693, %v4699 : tensor<32x32x28x28xf32>
    %v4701 = stablehlo.multiply %v4700, %v4700 : tensor<32x32x28x28xf32>
    %v4702 = stablehlo.reduce(%v4701 init: %v4694) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v4703 = stablehlo.broadcast_in_dim %v4702, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v4704 = stablehlo.divide %v4703, %v4695 : tensor<32x32x28x28xf32>
    %v4705 = stablehlo.add %v4704, %v4696 : tensor<32x32x28x28xf32>
    %v4706 = stablehlo.rsqrt %v4705 : tensor<32x32x28x28xf32>
    %v4707 = stablehlo.multiply %v4700, %v4706 : tensor<32x32x28x28xf32>
    %v4708 = stablehlo.broadcast_in_dim %gp4, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v4709 = stablehlo.multiply %v4708, %v4692 : tensor<32x32x28x28xf32>
    %v4710 = stablehlo.reduce(%v4709 init: %v4694) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v4711 = stablehlo.broadcast_in_dim %v4710, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v4712 = stablehlo.multiply %v4707, %v4709 : tensor<32x32x28x28xf32>
    %v4713 = stablehlo.reduce(%v4712 init: %v4694) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v4714 = stablehlo.broadcast_in_dim %v4713, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v4715 = stablehlo.multiply %v4709, %v4695 : tensor<32x32x28x28xf32>
    %v4716 = stablehlo.subtract %v4715, %v4711 : tensor<32x32x28x28xf32>
    %v4717 = stablehlo.multiply %v4707, %v4714 : tensor<32x32x28x28xf32>
    %v4718 = stablehlo.subtract %v4716, %v4717 : tensor<32x32x28x28xf32>
    %v4719 = stablehlo.divide %v4706, %v4695 : tensor<32x32x28x28xf32>
    %v4720 = stablehlo.multiply %v4719, %v4718 : tensor<32x32x28x28xf32>
    %v4721 = stablehlo.reshape %v4720 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v4722 = stablehlo.reshape %v4721 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4723 = stablehlo.transpose %Wp4, dims = [1, 0, 2, 3] : (tensor<32x144x1x1xf32>) -> tensor<144x32x1x1xf32>
    %v4724 = stablehlo.reverse %v4723, dims = [2, 3] : tensor<144x32x1x1xf32>
    %v4725 = stablehlo.convolution(%v4722, %v4724)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<144x32x1x1xf32>) -> tensor<32x144x28x28xf32>
    %v4726 = stablehlo.reshape %v4725 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v4727 = stablehlo.reshape %v4726 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v4728 = stablehlo.reshape %v320 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v4729 = stablehlo.constant dense<0.0> : tensor<32x144x28x28xf32>
    %v4730 = stablehlo.constant dense<6.0> : tensor<32x144x28x28xf32>
    %v4731 = stablehlo.compare GT, %v4728, %v4729 : (tensor<32x144x28x28xf32>, tensor<32x144x28x28xf32>) -> tensor<32x144x28x28xi1>
    %v4732 = stablehlo.compare LT, %v4728, %v4730 : (tensor<32x144x28x28xf32>, tensor<32x144x28x28xf32>) -> tensor<32x144x28x28xi1>
    %v4733 = stablehlo.and %v4731, %v4732 : tensor<32x144x28x28xi1>
    %v4734 = stablehlo.select %v4733, %v4727, %v4729 : tensor<32x144x28x28xi1>, tensor<32x144x28x28xf32>
    %v4735 = stablehlo.reshape %v4734 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v4736 = stablehlo.reshape %v4735 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v4737 = stablehlo.reshape %v300 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v4738 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4739 = stablehlo.constant dense<784.0> : tensor<32x144x28x28xf32>
    %v4740 = stablehlo.constant dense<1.0e-5> : tensor<32x144x28x28xf32>
    %v4741 = stablehlo.reduce(%v4737 init: %v4738) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v4742 = stablehlo.broadcast_in_dim %v4741, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x28x28xf32>
    %v4743 = stablehlo.divide %v4742, %v4739 : tensor<32x144x28x28xf32>
    %v4744 = stablehlo.subtract %v4737, %v4743 : tensor<32x144x28x28xf32>
    %v4745 = stablehlo.multiply %v4744, %v4744 : tensor<32x144x28x28xf32>
    %v4746 = stablehlo.reduce(%v4745 init: %v4738) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v4747 = stablehlo.broadcast_in_dim %v4746, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x28x28xf32>
    %v4748 = stablehlo.divide %v4747, %v4739 : tensor<32x144x28x28xf32>
    %v4749 = stablehlo.add %v4748, %v4740 : tensor<32x144x28x28xf32>
    %v4750 = stablehlo.rsqrt %v4749 : tensor<32x144x28x28xf32>
    %v4751 = stablehlo.multiply %v4744, %v4750 : tensor<32x144x28x28xf32>
    %v4752 = stablehlo.broadcast_in_dim %gd4, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v4753 = stablehlo.multiply %v4752, %v4736 : tensor<32x144x28x28xf32>
    %v4754 = stablehlo.reduce(%v4753 init: %v4738) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v4755 = stablehlo.broadcast_in_dim %v4754, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x28x28xf32>
    %v4756 = stablehlo.multiply %v4751, %v4753 : tensor<32x144x28x28xf32>
    %v4757 = stablehlo.reduce(%v4756 init: %v4738) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v4758 = stablehlo.broadcast_in_dim %v4757, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x28x28xf32>
    %v4759 = stablehlo.multiply %v4753, %v4739 : tensor<32x144x28x28xf32>
    %v4760 = stablehlo.subtract %v4759, %v4755 : tensor<32x144x28x28xf32>
    %v4761 = stablehlo.multiply %v4751, %v4758 : tensor<32x144x28x28xf32>
    %v4762 = stablehlo.subtract %v4760, %v4761 : tensor<32x144x28x28xf32>
    %v4763 = stablehlo.divide %v4750, %v4739 : tensor<32x144x28x28xf32>
    %v4764 = stablehlo.multiply %v4763, %v4762 : tensor<32x144x28x28xf32>
    %v4765 = stablehlo.reshape %v4764 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v4766 = stablehlo.reshape %v4765 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v4767 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4768 = stablehlo.pad %v4766, %v4767, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144x56x56xf32>
    %v4769 = stablehlo.reverse %Wd4, dims = [2, 3] : tensor<144x1x3x3xf32>
    %v4770 = stablehlo.convolution(%v4768, %v4769)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 144 : i64} : (tensor<32x144x56x56xf32>, tensor<144x1x3x3xf32>) -> tensor<32x144x56x56xf32>
    %v4771 = stablehlo.reshape %v4770 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v4772 = stablehlo.reshape %v4771 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4773 = stablehlo.reshape %v289 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4774 = stablehlo.constant dense<0.0> : tensor<32x144x56x56xf32>
    %v4775 = stablehlo.constant dense<6.0> : tensor<32x144x56x56xf32>
    %v4776 = stablehlo.compare GT, %v4773, %v4774 : (tensor<32x144x56x56xf32>, tensor<32x144x56x56xf32>) -> tensor<32x144x56x56xi1>
    %v4777 = stablehlo.compare LT, %v4773, %v4775 : (tensor<32x144x56x56xf32>, tensor<32x144x56x56xf32>) -> tensor<32x144x56x56xi1>
    %v4778 = stablehlo.and %v4776, %v4777 : tensor<32x144x56x56xi1>
    %v4779 = stablehlo.select %v4778, %v4772, %v4774 : tensor<32x144x56x56xi1>, tensor<32x144x56x56xf32>
    %v4780 = stablehlo.reshape %v4779 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v4781 = stablehlo.reshape %v4780 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4782 = stablehlo.reshape %v269 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4783 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4784 = stablehlo.constant dense<3136.0> : tensor<32x144x56x56xf32>
    %v4785 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v4786 = stablehlo.reduce(%v4782 init: %v4783) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v4787 = stablehlo.broadcast_in_dim %v4786, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v4788 = stablehlo.divide %v4787, %v4784 : tensor<32x144x56x56xf32>
    %v4789 = stablehlo.subtract %v4782, %v4788 : tensor<32x144x56x56xf32>
    %v4790 = stablehlo.multiply %v4789, %v4789 : tensor<32x144x56x56xf32>
    %v4791 = stablehlo.reduce(%v4790 init: %v4783) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v4792 = stablehlo.broadcast_in_dim %v4791, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v4793 = stablehlo.divide %v4792, %v4784 : tensor<32x144x56x56xf32>
    %v4794 = stablehlo.add %v4793, %v4785 : tensor<32x144x56x56xf32>
    %v4795 = stablehlo.rsqrt %v4794 : tensor<32x144x56x56xf32>
    %v4796 = stablehlo.multiply %v4789, %v4795 : tensor<32x144x56x56xf32>
    %v4797 = stablehlo.broadcast_in_dim %ge4, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v4798 = stablehlo.multiply %v4797, %v4781 : tensor<32x144x56x56xf32>
    %v4799 = stablehlo.reduce(%v4798 init: %v4783) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v4800 = stablehlo.broadcast_in_dim %v4799, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v4801 = stablehlo.multiply %v4796, %v4798 : tensor<32x144x56x56xf32>
    %v4802 = stablehlo.reduce(%v4801 init: %v4783) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v4803 = stablehlo.broadcast_in_dim %v4802, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v4804 = stablehlo.multiply %v4798, %v4784 : tensor<32x144x56x56xf32>
    %v4805 = stablehlo.subtract %v4804, %v4800 : tensor<32x144x56x56xf32>
    %v4806 = stablehlo.multiply %v4796, %v4803 : tensor<32x144x56x56xf32>
    %v4807 = stablehlo.subtract %v4805, %v4806 : tensor<32x144x56x56xf32>
    %v4808 = stablehlo.divide %v4795, %v4784 : tensor<32x144x56x56xf32>
    %v4809 = stablehlo.multiply %v4808, %v4807 : tensor<32x144x56x56xf32>
    %v4810 = stablehlo.reshape %v4809 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v4811 = stablehlo.reshape %v4810 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4812 = stablehlo.transpose %We4, dims = [1, 0, 2, 3] : (tensor<144x24x1x1xf32>) -> tensor<24x144x1x1xf32>
    %v4813 = stablehlo.reverse %v4812, dims = [2, 3] : tensor<24x144x1x1xf32>
    %v4814 = stablehlo.convolution(%v4811, %v4813)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x144x56x56xf32>, tensor<24x144x1x1xf32>) -> tensor<32x24x56x56xf32>
    %v4815 = stablehlo.reshape %v4814 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v4816 = stablehlo.reshape %v264 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v4817 = stablehlo.reshape %v4810 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4818 = stablehlo.transpose %v4816, dims = [1, 0, 2, 3] : (tensor<32x24x56x56xf32>) -> tensor<24x32x56x56xf32>
    %v4819 = stablehlo.transpose %v4817, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %v4820 = stablehlo.convolution(%v4818, %v4819)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<24x32x56x56xf32>, tensor<144x32x56x56xf32>) -> tensor<24x144x1x1xf32>
    %v4821 = stablehlo.transpose %v4820, dims = [1, 0, 2, 3] : (tensor<24x144x1x1xf32>) -> tensor<144x24x1x1xf32>
    %v4822 = stablehlo.constant dense<0.3> : tensor<144x24x1x1xf32>
    %v4823 = stablehlo.multiply %v4821, %v4822 : tensor<144x24x1x1xf32>
    %v4824 = stablehlo.subtract %We4, %v4823 : tensor<144x24x1x1xf32>
    %v4825 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4826 = stablehlo.reshape %v269 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4827 = stablehlo.constant dense<3136.0> : tensor<32x144x56x56xf32>
    %v4828 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v4829 = stablehlo.reduce(%v4826 init: %v4825) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v4830 = stablehlo.broadcast_in_dim %v4829, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v4831 = stablehlo.divide %v4830, %v4827 : tensor<32x144x56x56xf32>
    %v4832 = stablehlo.subtract %v4826, %v4831 : tensor<32x144x56x56xf32>
    %v4833 = stablehlo.multiply %v4832, %v4832 : tensor<32x144x56x56xf32>
    %v4834 = stablehlo.reduce(%v4833 init: %v4825) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v4835 = stablehlo.broadcast_in_dim %v4834, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v4836 = stablehlo.divide %v4835, %v4827 : tensor<32x144x56x56xf32>
    %v4837 = stablehlo.add %v4836, %v4828 : tensor<32x144x56x56xf32>
    %v4838 = stablehlo.rsqrt %v4837 : tensor<32x144x56x56xf32>
    %v4839 = stablehlo.multiply %v4832, %v4838 : tensor<32x144x56x56xf32>
    %v4840 = stablehlo.reshape %v4780 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4841 = stablehlo.multiply %v4840, %v4839 : tensor<32x144x56x56xf32>
    %v4842 = stablehlo.reduce(%v4841 init: %v4825) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4843 = stablehlo.constant dense<0.3> : tensor<144xf32>
    %v4844 = stablehlo.multiply %v4842, %v4843 : tensor<144xf32>
    %v4845 = stablehlo.subtract %ge4, %v4844 : tensor<144xf32>
    %v4846 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4847 = stablehlo.reshape %v4780 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4848 = stablehlo.reduce(%v4847 init: %v4846) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4849 = stablehlo.constant dense<0.3> : tensor<144xf32>
    %v4850 = stablehlo.multiply %v4848, %v4849 : tensor<144xf32>
    %v4851 = stablehlo.subtract %bte4, %v4850 : tensor<144xf32>
    %v4852 = stablehlo.reshape %v295 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4853 = stablehlo.reshape %v4765 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v4854 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4855 = stablehlo.pad %v4853, %v4854, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144x56x56xf32>
    %v4856 = stablehlo.transpose %v4852, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %v4857 = stablehlo.transpose %v4855, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %v4858 = stablehlo.convolution(%v4856, %v4857)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 144 : i64, feature_group_count = 1 : i64} : (tensor<144x32x56x56xf32>, tensor<144x32x56x56xf32>) -> tensor<1x144x3x3xf32>
    %v4859 = stablehlo.reshape %v4858 : (tensor<1x144x3x3xf32>) -> tensor<144x1x3x3xf32>
    %v4860 = stablehlo.constant dense<0.3> : tensor<144x1x3x3xf32>
    %v4861 = stablehlo.multiply %v4859, %v4860 : tensor<144x1x3x3xf32>
    %v4862 = stablehlo.subtract %Wd4, %v4861 : tensor<144x1x3x3xf32>
    %v4863 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4864 = stablehlo.reshape %v300 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v4865 = stablehlo.constant dense<784.0> : tensor<32x144x28x28xf32>
    %v4866 = stablehlo.constant dense<1.0e-5> : tensor<32x144x28x28xf32>
    %v4867 = stablehlo.reduce(%v4864 init: %v4863) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v4868 = stablehlo.broadcast_in_dim %v4867, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x28x28xf32>
    %v4869 = stablehlo.divide %v4868, %v4865 : tensor<32x144x28x28xf32>
    %v4870 = stablehlo.subtract %v4864, %v4869 : tensor<32x144x28x28xf32>
    %v4871 = stablehlo.multiply %v4870, %v4870 : tensor<32x144x28x28xf32>
    %v4872 = stablehlo.reduce(%v4871 init: %v4863) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v4873 = stablehlo.broadcast_in_dim %v4872, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x28x28xf32>
    %v4874 = stablehlo.divide %v4873, %v4865 : tensor<32x144x28x28xf32>
    %v4875 = stablehlo.add %v4874, %v4866 : tensor<32x144x28x28xf32>
    %v4876 = stablehlo.rsqrt %v4875 : tensor<32x144x28x28xf32>
    %v4877 = stablehlo.multiply %v4870, %v4876 : tensor<32x144x28x28xf32>
    %v4878 = stablehlo.reshape %v4735 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v4879 = stablehlo.multiply %v4878, %v4877 : tensor<32x144x28x28xf32>
    %v4880 = stablehlo.reduce(%v4879 init: %v4863) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v4881 = stablehlo.constant dense<0.3> : tensor<144xf32>
    %v4882 = stablehlo.multiply %v4880, %v4881 : tensor<144xf32>
    %v4883 = stablehlo.subtract %gd4, %v4882 : tensor<144xf32>
    %v4884 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4885 = stablehlo.reshape %v4735 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v4886 = stablehlo.reduce(%v4885 init: %v4884) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v4887 = stablehlo.constant dense<0.3> : tensor<144xf32>
    %v4888 = stablehlo.multiply %v4886, %v4887 : tensor<144xf32>
    %v4889 = stablehlo.subtract %btd4, %v4888 : tensor<144xf32>
    %v4890 = stablehlo.reshape %v326 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v4891 = stablehlo.reshape %v4721 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4892 = stablehlo.transpose %v4890, dims = [1, 0, 2, 3] : (tensor<32x144x28x28xf32>) -> tensor<144x32x28x28xf32>
    %v4893 = stablehlo.transpose %v4891, dims = [1, 0, 2, 3] : (tensor<32x32x28x28xf32>) -> tensor<32x32x28x28xf32>
    %v4894 = stablehlo.convolution(%v4892, %v4893)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<144x32x28x28xf32>, tensor<32x32x28x28xf32>) -> tensor<144x32x1x1xf32>
    %v4895 = stablehlo.transpose %v4894, dims = [1, 0, 2, 3] : (tensor<144x32x1x1xf32>) -> tensor<32x144x1x1xf32>
    %v4896 = stablehlo.constant dense<0.3> : tensor<32x144x1x1xf32>
    %v4897 = stablehlo.multiply %v4895, %v4896 : tensor<32x144x1x1xf32>
    %v4898 = stablehlo.subtract %Wp4, %v4897 : tensor<32x144x1x1xf32>
    %v4899 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4900 = stablehlo.reshape %v331 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4901 = stablehlo.constant dense<784.0> : tensor<32x32x28x28xf32>
    %v4902 = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %v4903 = stablehlo.reduce(%v4900 init: %v4899) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v4904 = stablehlo.broadcast_in_dim %v4903, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v4905 = stablehlo.divide %v4904, %v4901 : tensor<32x32x28x28xf32>
    %v4906 = stablehlo.subtract %v4900, %v4905 : tensor<32x32x28x28xf32>
    %v4907 = stablehlo.multiply %v4906, %v4906 : tensor<32x32x28x28xf32>
    %v4908 = stablehlo.reduce(%v4907 init: %v4899) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v4909 = stablehlo.broadcast_in_dim %v4908, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v4910 = stablehlo.divide %v4909, %v4901 : tensor<32x32x28x28xf32>
    %v4911 = stablehlo.add %v4910, %v4902 : tensor<32x32x28x28xf32>
    %v4912 = stablehlo.rsqrt %v4911 : tensor<32x32x28x28xf32>
    %v4913 = stablehlo.multiply %v4906, %v4912 : tensor<32x32x28x28xf32>
    %v4914 = stablehlo.reshape %v4583 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4915 = stablehlo.multiply %v4914, %v4913 : tensor<32x32x28x28xf32>
    %v4916 = stablehlo.reduce(%v4915 init: %v4899) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v4917 = stablehlo.constant dense<0.3> : tensor<32xf32>
    %v4918 = stablehlo.multiply %v4916, %v4917 : tensor<32xf32>
    %v4919 = stablehlo.subtract %gp4, %v4918 : tensor<32xf32>
    %v4920 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4921 = stablehlo.reshape %v4583 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4922 = stablehlo.reduce(%v4921 init: %v4920) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v4923 = stablehlo.constant dense<0.3> : tensor<32xf32>
    %v4924 = stablehlo.multiply %v4922, %v4923 : tensor<32xf32>
    %v4925 = stablehlo.subtract %btp4, %v4924 : tensor<32xf32>
    %v4926 = stablehlo.reshape %v4815 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v4927 = stablehlo.reshape %v240 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v4928 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4929 = stablehlo.constant dense<3136.0> : tensor<32x24x56x56xf32>
    %v4930 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v4931 = stablehlo.reduce(%v4927 init: %v4928) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v4932 = stablehlo.broadcast_in_dim %v4931, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v4933 = stablehlo.divide %v4932, %v4929 : tensor<32x24x56x56xf32>
    %v4934 = stablehlo.subtract %v4927, %v4933 : tensor<32x24x56x56xf32>
    %v4935 = stablehlo.multiply %v4934, %v4934 : tensor<32x24x56x56xf32>
    %v4936 = stablehlo.reduce(%v4935 init: %v4928) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v4937 = stablehlo.broadcast_in_dim %v4936, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v4938 = stablehlo.divide %v4937, %v4929 : tensor<32x24x56x56xf32>
    %v4939 = stablehlo.add %v4938, %v4930 : tensor<32x24x56x56xf32>
    %v4940 = stablehlo.rsqrt %v4939 : tensor<32x24x56x56xf32>
    %v4941 = stablehlo.multiply %v4934, %v4940 : tensor<32x24x56x56xf32>
    %v4942 = stablehlo.broadcast_in_dim %gp3, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v4943 = stablehlo.multiply %v4942, %v4926 : tensor<32x24x56x56xf32>
    %v4944 = stablehlo.reduce(%v4943 init: %v4928) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v4945 = stablehlo.broadcast_in_dim %v4944, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v4946 = stablehlo.multiply %v4941, %v4943 : tensor<32x24x56x56xf32>
    %v4947 = stablehlo.reduce(%v4946 init: %v4928) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v4948 = stablehlo.broadcast_in_dim %v4947, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v4949 = stablehlo.multiply %v4943, %v4929 : tensor<32x24x56x56xf32>
    %v4950 = stablehlo.subtract %v4949, %v4945 : tensor<32x24x56x56xf32>
    %v4951 = stablehlo.multiply %v4941, %v4948 : tensor<32x24x56x56xf32>
    %v4952 = stablehlo.subtract %v4950, %v4951 : tensor<32x24x56x56xf32>
    %v4953 = stablehlo.divide %v4940, %v4929 : tensor<32x24x56x56xf32>
    %v4954 = stablehlo.multiply %v4953, %v4952 : tensor<32x24x56x56xf32>
    %v4955 = stablehlo.reshape %v4954 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v4956 = stablehlo.reshape %v4955 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v4957 = stablehlo.transpose %Wp3, dims = [1, 0, 2, 3] : (tensor<24x144x1x1xf32>) -> tensor<144x24x1x1xf32>
    %v4958 = stablehlo.reverse %v4957, dims = [2, 3] : tensor<144x24x1x1xf32>
    %v4959 = stablehlo.convolution(%v4956, %v4958)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<144x24x1x1xf32>) -> tensor<32x144x56x56xf32>
    %v4960 = stablehlo.reshape %v4959 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v4961 = stablehlo.reshape %v4960 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4962 = stablehlo.reshape %v229 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4963 = stablehlo.constant dense<0.0> : tensor<32x144x56x56xf32>
    %v4964 = stablehlo.constant dense<6.0> : tensor<32x144x56x56xf32>
    %v4965 = stablehlo.compare GT, %v4962, %v4963 : (tensor<32x144x56x56xf32>, tensor<32x144x56x56xf32>) -> tensor<32x144x56x56xi1>
    %v4966 = stablehlo.compare LT, %v4962, %v4964 : (tensor<32x144x56x56xf32>, tensor<32x144x56x56xf32>) -> tensor<32x144x56x56xi1>
    %v4967 = stablehlo.and %v4965, %v4966 : tensor<32x144x56x56xi1>
    %v4968 = stablehlo.select %v4967, %v4961, %v4963 : tensor<32x144x56x56xi1>, tensor<32x144x56x56xf32>
    %v4969 = stablehlo.reshape %v4968 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v4970 = stablehlo.reshape %v4969 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4971 = stablehlo.reshape %v209 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4972 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4973 = stablehlo.constant dense<3136.0> : tensor<32x144x56x56xf32>
    %v4974 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v4975 = stablehlo.reduce(%v4971 init: %v4972) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v4976 = stablehlo.broadcast_in_dim %v4975, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v4977 = stablehlo.divide %v4976, %v4973 : tensor<32x144x56x56xf32>
    %v4978 = stablehlo.subtract %v4971, %v4977 : tensor<32x144x56x56xf32>
    %v4979 = stablehlo.multiply %v4978, %v4978 : tensor<32x144x56x56xf32>
    %v4980 = stablehlo.reduce(%v4979 init: %v4972) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v4981 = stablehlo.broadcast_in_dim %v4980, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v4982 = stablehlo.divide %v4981, %v4973 : tensor<32x144x56x56xf32>
    %v4983 = stablehlo.add %v4982, %v4974 : tensor<32x144x56x56xf32>
    %v4984 = stablehlo.rsqrt %v4983 : tensor<32x144x56x56xf32>
    %v4985 = stablehlo.multiply %v4978, %v4984 : tensor<32x144x56x56xf32>
    %v4986 = stablehlo.broadcast_in_dim %gd3, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v4987 = stablehlo.multiply %v4986, %v4970 : tensor<32x144x56x56xf32>
    %v4988 = stablehlo.reduce(%v4987 init: %v4972) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v4989 = stablehlo.broadcast_in_dim %v4988, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v4990 = stablehlo.multiply %v4985, %v4987 : tensor<32x144x56x56xf32>
    %v4991 = stablehlo.reduce(%v4990 init: %v4972) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v4992 = stablehlo.broadcast_in_dim %v4991, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v4993 = stablehlo.multiply %v4987, %v4973 : tensor<32x144x56x56xf32>
    %v4994 = stablehlo.subtract %v4993, %v4989 : tensor<32x144x56x56xf32>
    %v4995 = stablehlo.multiply %v4985, %v4992 : tensor<32x144x56x56xf32>
    %v4996 = stablehlo.subtract %v4994, %v4995 : tensor<32x144x56x56xf32>
    %v4997 = stablehlo.divide %v4984, %v4973 : tensor<32x144x56x56xf32>
    %v4998 = stablehlo.multiply %v4997, %v4996 : tensor<32x144x56x56xf32>
    %v4999 = stablehlo.reshape %v4998 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v5000 = stablehlo.reshape %v4999 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5001 = stablehlo.reverse %Wd3, dims = [2, 3] : tensor<144x1x3x3xf32>
    %v5002 = stablehlo.convolution(%v5000, %v5001)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 144 : i64} : (tensor<32x144x56x56xf32>, tensor<144x1x3x3xf32>) -> tensor<32x144x56x56xf32>
    %v5003 = stablehlo.reshape %v5002 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v5004 = stablehlo.reshape %v5003 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5005 = stablehlo.reshape %v198 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5006 = stablehlo.constant dense<0.0> : tensor<32x144x56x56xf32>
    %v5007 = stablehlo.constant dense<6.0> : tensor<32x144x56x56xf32>
    %v5008 = stablehlo.compare GT, %v5005, %v5006 : (tensor<32x144x56x56xf32>, tensor<32x144x56x56xf32>) -> tensor<32x144x56x56xi1>
    %v5009 = stablehlo.compare LT, %v5005, %v5007 : (tensor<32x144x56x56xf32>, tensor<32x144x56x56xf32>) -> tensor<32x144x56x56xi1>
    %v5010 = stablehlo.and %v5008, %v5009 : tensor<32x144x56x56xi1>
    %v5011 = stablehlo.select %v5010, %v5004, %v5006 : tensor<32x144x56x56xi1>, tensor<32x144x56x56xf32>
    %v5012 = stablehlo.reshape %v5011 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v5013 = stablehlo.reshape %v5012 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5014 = stablehlo.reshape %v178 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5015 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5016 = stablehlo.constant dense<3136.0> : tensor<32x144x56x56xf32>
    %v5017 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v5018 = stablehlo.reduce(%v5014 init: %v5015) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v5019 = stablehlo.broadcast_in_dim %v5018, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v5020 = stablehlo.divide %v5019, %v5016 : tensor<32x144x56x56xf32>
    %v5021 = stablehlo.subtract %v5014, %v5020 : tensor<32x144x56x56xf32>
    %v5022 = stablehlo.multiply %v5021, %v5021 : tensor<32x144x56x56xf32>
    %v5023 = stablehlo.reduce(%v5022 init: %v5015) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v5024 = stablehlo.broadcast_in_dim %v5023, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v5025 = stablehlo.divide %v5024, %v5016 : tensor<32x144x56x56xf32>
    %v5026 = stablehlo.add %v5025, %v5017 : tensor<32x144x56x56xf32>
    %v5027 = stablehlo.rsqrt %v5026 : tensor<32x144x56x56xf32>
    %v5028 = stablehlo.multiply %v5021, %v5027 : tensor<32x144x56x56xf32>
    %v5029 = stablehlo.broadcast_in_dim %ge3, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v5030 = stablehlo.multiply %v5029, %v5013 : tensor<32x144x56x56xf32>
    %v5031 = stablehlo.reduce(%v5030 init: %v5015) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v5032 = stablehlo.broadcast_in_dim %v5031, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v5033 = stablehlo.multiply %v5028, %v5030 : tensor<32x144x56x56xf32>
    %v5034 = stablehlo.reduce(%v5033 init: %v5015) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v5035 = stablehlo.broadcast_in_dim %v5034, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v5036 = stablehlo.multiply %v5030, %v5016 : tensor<32x144x56x56xf32>
    %v5037 = stablehlo.subtract %v5036, %v5032 : tensor<32x144x56x56xf32>
    %v5038 = stablehlo.multiply %v5028, %v5035 : tensor<32x144x56x56xf32>
    %v5039 = stablehlo.subtract %v5037, %v5038 : tensor<32x144x56x56xf32>
    %v5040 = stablehlo.divide %v5027, %v5016 : tensor<32x144x56x56xf32>
    %v5041 = stablehlo.multiply %v5040, %v5039 : tensor<32x144x56x56xf32>
    %v5042 = stablehlo.reshape %v5041 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v5043 = stablehlo.reshape %v5042 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5044 = stablehlo.transpose %We3, dims = [1, 0, 2, 3] : (tensor<144x24x1x1xf32>) -> tensor<24x144x1x1xf32>
    %v5045 = stablehlo.reverse %v5044, dims = [2, 3] : tensor<24x144x1x1xf32>
    %v5046 = stablehlo.convolution(%v5043, %v5045)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x144x56x56xf32>, tensor<24x144x1x1xf32>) -> tensor<32x24x56x56xf32>
    %v5047 = stablehlo.reshape %v5046 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v5048 = stablehlo.reshape %v5047 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v5049 = stablehlo.reshape %v4815 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v5050 = stablehlo.add %v5048, %v5049 : tensor<32x24x56x56xf32>
    %v5051 = stablehlo.reshape %v5050 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v5052 = stablehlo.reshape %v173 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v5053 = stablehlo.reshape %v5042 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5054 = stablehlo.transpose %v5052, dims = [1, 0, 2, 3] : (tensor<32x24x56x56xf32>) -> tensor<24x32x56x56xf32>
    %v5055 = stablehlo.transpose %v5053, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %v5056 = stablehlo.convolution(%v5054, %v5055)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<24x32x56x56xf32>, tensor<144x32x56x56xf32>) -> tensor<24x144x1x1xf32>
    %v5057 = stablehlo.transpose %v5056, dims = [1, 0, 2, 3] : (tensor<24x144x1x1xf32>) -> tensor<144x24x1x1xf32>
    %v5058 = stablehlo.constant dense<0.3> : tensor<144x24x1x1xf32>
    %v5059 = stablehlo.multiply %v5057, %v5058 : tensor<144x24x1x1xf32>
    %v5060 = stablehlo.subtract %We3, %v5059 : tensor<144x24x1x1xf32>
    %v5061 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5062 = stablehlo.reshape %v178 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5063 = stablehlo.constant dense<3136.0> : tensor<32x144x56x56xf32>
    %v5064 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v5065 = stablehlo.reduce(%v5062 init: %v5061) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v5066 = stablehlo.broadcast_in_dim %v5065, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v5067 = stablehlo.divide %v5066, %v5063 : tensor<32x144x56x56xf32>
    %v5068 = stablehlo.subtract %v5062, %v5067 : tensor<32x144x56x56xf32>
    %v5069 = stablehlo.multiply %v5068, %v5068 : tensor<32x144x56x56xf32>
    %v5070 = stablehlo.reduce(%v5069 init: %v5061) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v5071 = stablehlo.broadcast_in_dim %v5070, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v5072 = stablehlo.divide %v5071, %v5063 : tensor<32x144x56x56xf32>
    %v5073 = stablehlo.add %v5072, %v5064 : tensor<32x144x56x56xf32>
    %v5074 = stablehlo.rsqrt %v5073 : tensor<32x144x56x56xf32>
    %v5075 = stablehlo.multiply %v5068, %v5074 : tensor<32x144x56x56xf32>
    %v5076 = stablehlo.reshape %v5012 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5077 = stablehlo.multiply %v5076, %v5075 : tensor<32x144x56x56xf32>
    %v5078 = stablehlo.reduce(%v5077 init: %v5061) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5079 = stablehlo.constant dense<0.3> : tensor<144xf32>
    %v5080 = stablehlo.multiply %v5078, %v5079 : tensor<144xf32>
    %v5081 = stablehlo.subtract %ge3, %v5080 : tensor<144xf32>
    %v5082 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5083 = stablehlo.reshape %v5012 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5084 = stablehlo.reduce(%v5083 init: %v5082) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5085 = stablehlo.constant dense<0.3> : tensor<144xf32>
    %v5086 = stablehlo.multiply %v5084, %v5085 : tensor<144xf32>
    %v5087 = stablehlo.subtract %bte3, %v5086 : tensor<144xf32>
    %v5088 = stablehlo.reshape %v204 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5089 = stablehlo.reshape %v4999 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5090 = stablehlo.transpose %v5088, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %v5091 = stablehlo.transpose %v5089, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %v5092 = stablehlo.convolution(%v5090, %v5091)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 144 : i64, feature_group_count = 1 : i64} : (tensor<144x32x56x56xf32>, tensor<144x32x56x56xf32>) -> tensor<1x144x3x3xf32>
    %v5093 = stablehlo.reshape %v5092 : (tensor<1x144x3x3xf32>) -> tensor<144x1x3x3xf32>
    %v5094 = stablehlo.constant dense<0.3> : tensor<144x1x3x3xf32>
    %v5095 = stablehlo.multiply %v5093, %v5094 : tensor<144x1x3x3xf32>
    %v5096 = stablehlo.subtract %Wd3, %v5095 : tensor<144x1x3x3xf32>
    %v5097 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5098 = stablehlo.reshape %v209 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5099 = stablehlo.constant dense<3136.0> : tensor<32x144x56x56xf32>
    %v5100 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v5101 = stablehlo.reduce(%v5098 init: %v5097) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v5102 = stablehlo.broadcast_in_dim %v5101, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v5103 = stablehlo.divide %v5102, %v5099 : tensor<32x144x56x56xf32>
    %v5104 = stablehlo.subtract %v5098, %v5103 : tensor<32x144x56x56xf32>
    %v5105 = stablehlo.multiply %v5104, %v5104 : tensor<32x144x56x56xf32>
    %v5106 = stablehlo.reduce(%v5105 init: %v5097) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v5107 = stablehlo.broadcast_in_dim %v5106, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v5108 = stablehlo.divide %v5107, %v5099 : tensor<32x144x56x56xf32>
    %v5109 = stablehlo.add %v5108, %v5100 : tensor<32x144x56x56xf32>
    %v5110 = stablehlo.rsqrt %v5109 : tensor<32x144x56x56xf32>
    %v5111 = stablehlo.multiply %v5104, %v5110 : tensor<32x144x56x56xf32>
    %v5112 = stablehlo.reshape %v4969 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5113 = stablehlo.multiply %v5112, %v5111 : tensor<32x144x56x56xf32>
    %v5114 = stablehlo.reduce(%v5113 init: %v5097) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5115 = stablehlo.constant dense<0.3> : tensor<144xf32>
    %v5116 = stablehlo.multiply %v5114, %v5115 : tensor<144xf32>
    %v5117 = stablehlo.subtract %gd3, %v5116 : tensor<144xf32>
    %v5118 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5119 = stablehlo.reshape %v4969 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5120 = stablehlo.reduce(%v5119 init: %v5118) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5121 = stablehlo.constant dense<0.3> : tensor<144xf32>
    %v5122 = stablehlo.multiply %v5120, %v5121 : tensor<144xf32>
    %v5123 = stablehlo.subtract %btd3, %v5122 : tensor<144xf32>
    %v5124 = stablehlo.reshape %v235 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5125 = stablehlo.reshape %v4955 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v5126 = stablehlo.transpose %v5124, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %v5127 = stablehlo.transpose %v5125, dims = [1, 0, 2, 3] : (tensor<32x24x56x56xf32>) -> tensor<24x32x56x56xf32>
    %v5128 = stablehlo.convolution(%v5126, %v5127)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<144x32x56x56xf32>, tensor<24x32x56x56xf32>) -> tensor<144x24x1x1xf32>
    %v5129 = stablehlo.transpose %v5128, dims = [1, 0, 2, 3] : (tensor<144x24x1x1xf32>) -> tensor<24x144x1x1xf32>
    %v5130 = stablehlo.constant dense<0.3> : tensor<24x144x1x1xf32>
    %v5131 = stablehlo.multiply %v5129, %v5130 : tensor<24x144x1x1xf32>
    %v5132 = stablehlo.subtract %Wp3, %v5131 : tensor<24x144x1x1xf32>
    %v5133 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5134 = stablehlo.reshape %v240 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v5135 = stablehlo.constant dense<3136.0> : tensor<32x24x56x56xf32>
    %v5136 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v5137 = stablehlo.reduce(%v5134 init: %v5133) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v5138 = stablehlo.broadcast_in_dim %v5137, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v5139 = stablehlo.divide %v5138, %v5135 : tensor<32x24x56x56xf32>
    %v5140 = stablehlo.subtract %v5134, %v5139 : tensor<32x24x56x56xf32>
    %v5141 = stablehlo.multiply %v5140, %v5140 : tensor<32x24x56x56xf32>
    %v5142 = stablehlo.reduce(%v5141 init: %v5133) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v5143 = stablehlo.broadcast_in_dim %v5142, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v5144 = stablehlo.divide %v5143, %v5135 : tensor<32x24x56x56xf32>
    %v5145 = stablehlo.add %v5144, %v5136 : tensor<32x24x56x56xf32>
    %v5146 = stablehlo.rsqrt %v5145 : tensor<32x24x56x56xf32>
    %v5147 = stablehlo.multiply %v5140, %v5146 : tensor<32x24x56x56xf32>
    %v5148 = stablehlo.reshape %v4815 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v5149 = stablehlo.multiply %v5148, %v5147 : tensor<32x24x56x56xf32>
    %v5150 = stablehlo.reduce(%v5149 init: %v5133) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v5151 = stablehlo.constant dense<0.3> : tensor<24xf32>
    %v5152 = stablehlo.multiply %v5150, %v5151 : tensor<24xf32>
    %v5153 = stablehlo.subtract %gp3, %v5152 : tensor<24xf32>
    %v5154 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5155 = stablehlo.reshape %v4815 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v5156 = stablehlo.reduce(%v5155 init: %v5154) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v5157 = stablehlo.constant dense<0.3> : tensor<24xf32>
    %v5158 = stablehlo.multiply %v5156, %v5157 : tensor<24xf32>
    %v5159 = stablehlo.subtract %btp3, %v5158 : tensor<24xf32>
    %v5160 = stablehlo.reshape %v5051 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v5161 = stablehlo.reshape %v153 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v5162 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5163 = stablehlo.constant dense<3136.0> : tensor<32x24x56x56xf32>
    %v5164 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v5165 = stablehlo.reduce(%v5161 init: %v5162) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v5166 = stablehlo.broadcast_in_dim %v5165, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v5167 = stablehlo.divide %v5166, %v5163 : tensor<32x24x56x56xf32>
    %v5168 = stablehlo.subtract %v5161, %v5167 : tensor<32x24x56x56xf32>
    %v5169 = stablehlo.multiply %v5168, %v5168 : tensor<32x24x56x56xf32>
    %v5170 = stablehlo.reduce(%v5169 init: %v5162) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v5171 = stablehlo.broadcast_in_dim %v5170, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v5172 = stablehlo.divide %v5171, %v5163 : tensor<32x24x56x56xf32>
    %v5173 = stablehlo.add %v5172, %v5164 : tensor<32x24x56x56xf32>
    %v5174 = stablehlo.rsqrt %v5173 : tensor<32x24x56x56xf32>
    %v5175 = stablehlo.multiply %v5168, %v5174 : tensor<32x24x56x56xf32>
    %v5176 = stablehlo.broadcast_in_dim %gp2, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v5177 = stablehlo.multiply %v5176, %v5160 : tensor<32x24x56x56xf32>
    %v5178 = stablehlo.reduce(%v5177 init: %v5162) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v5179 = stablehlo.broadcast_in_dim %v5178, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v5180 = stablehlo.multiply %v5175, %v5177 : tensor<32x24x56x56xf32>
    %v5181 = stablehlo.reduce(%v5180 init: %v5162) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v5182 = stablehlo.broadcast_in_dim %v5181, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v5183 = stablehlo.multiply %v5177, %v5163 : tensor<32x24x56x56xf32>
    %v5184 = stablehlo.subtract %v5183, %v5179 : tensor<32x24x56x56xf32>
    %v5185 = stablehlo.multiply %v5175, %v5182 : tensor<32x24x56x56xf32>
    %v5186 = stablehlo.subtract %v5184, %v5185 : tensor<32x24x56x56xf32>
    %v5187 = stablehlo.divide %v5174, %v5163 : tensor<32x24x56x56xf32>
    %v5188 = stablehlo.multiply %v5187, %v5186 : tensor<32x24x56x56xf32>
    %v5189 = stablehlo.reshape %v5188 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v5190 = stablehlo.reshape %v5189 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v5191 = stablehlo.transpose %Wp2, dims = [1, 0, 2, 3] : (tensor<24x96x1x1xf32>) -> tensor<96x24x1x1xf32>
    %v5192 = stablehlo.reverse %v5191, dims = [2, 3] : tensor<96x24x1x1xf32>
    %v5193 = stablehlo.convolution(%v5190, %v5192)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<96x24x1x1xf32>) -> tensor<32x96x56x56xf32>
    %v5194 = stablehlo.reshape %v5193 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v5195 = stablehlo.reshape %v5194 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v5196 = stablehlo.reshape %v142 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v5197 = stablehlo.constant dense<0.0> : tensor<32x96x56x56xf32>
    %v5198 = stablehlo.constant dense<6.0> : tensor<32x96x56x56xf32>
    %v5199 = stablehlo.compare GT, %v5196, %v5197 : (tensor<32x96x56x56xf32>, tensor<32x96x56x56xf32>) -> tensor<32x96x56x56xi1>
    %v5200 = stablehlo.compare LT, %v5196, %v5198 : (tensor<32x96x56x56xf32>, tensor<32x96x56x56xf32>) -> tensor<32x96x56x56xi1>
    %v5201 = stablehlo.and %v5199, %v5200 : tensor<32x96x56x56xi1>
    %v5202 = stablehlo.select %v5201, %v5195, %v5197 : tensor<32x96x56x56xi1>, tensor<32x96x56x56xf32>
    %v5203 = stablehlo.reshape %v5202 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v5204 = stablehlo.reshape %v5203 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v5205 = stablehlo.reshape %v122 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v5206 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5207 = stablehlo.constant dense<3136.0> : tensor<32x96x56x56xf32>
    %v5208 = stablehlo.constant dense<1.0e-5> : tensor<32x96x56x56xf32>
    %v5209 = stablehlo.reduce(%v5205 init: %v5206) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v5210 = stablehlo.broadcast_in_dim %v5209, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v5211 = stablehlo.divide %v5210, %v5207 : tensor<32x96x56x56xf32>
    %v5212 = stablehlo.subtract %v5205, %v5211 : tensor<32x96x56x56xf32>
    %v5213 = stablehlo.multiply %v5212, %v5212 : tensor<32x96x56x56xf32>
    %v5214 = stablehlo.reduce(%v5213 init: %v5206) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v5215 = stablehlo.broadcast_in_dim %v5214, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v5216 = stablehlo.divide %v5215, %v5207 : tensor<32x96x56x56xf32>
    %v5217 = stablehlo.add %v5216, %v5208 : tensor<32x96x56x56xf32>
    %v5218 = stablehlo.rsqrt %v5217 : tensor<32x96x56x56xf32>
    %v5219 = stablehlo.multiply %v5212, %v5218 : tensor<32x96x56x56xf32>
    %v5220 = stablehlo.broadcast_in_dim %gd2, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v5221 = stablehlo.multiply %v5220, %v5204 : tensor<32x96x56x56xf32>
    %v5222 = stablehlo.reduce(%v5221 init: %v5206) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v5223 = stablehlo.broadcast_in_dim %v5222, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v5224 = stablehlo.multiply %v5219, %v5221 : tensor<32x96x56x56xf32>
    %v5225 = stablehlo.reduce(%v5224 init: %v5206) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v5226 = stablehlo.broadcast_in_dim %v5225, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v5227 = stablehlo.multiply %v5221, %v5207 : tensor<32x96x56x56xf32>
    %v5228 = stablehlo.subtract %v5227, %v5223 : tensor<32x96x56x56xf32>
    %v5229 = stablehlo.multiply %v5219, %v5226 : tensor<32x96x56x56xf32>
    %v5230 = stablehlo.subtract %v5228, %v5229 : tensor<32x96x56x56xf32>
    %v5231 = stablehlo.divide %v5218, %v5207 : tensor<32x96x56x56xf32>
    %v5232 = stablehlo.multiply %v5231, %v5230 : tensor<32x96x56x56xf32>
    %v5233 = stablehlo.reshape %v5232 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v5234 = stablehlo.reshape %v5233 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v5235 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5236 = stablehlo.pad %v5234, %v5235, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96x112x112xf32>
    %v5237 = stablehlo.reverse %Wd2, dims = [2, 3] : tensor<96x1x3x3xf32>
    %v5238 = stablehlo.convolution(%v5236, %v5237)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x112x112xf32>, tensor<96x1x3x3xf32>) -> tensor<32x96x112x112xf32>
    %v5239 = stablehlo.reshape %v5238 : (tensor<32x96x112x112xf32>) -> tensor<32x1204224xf32>
    %v5240 = stablehlo.reshape %v5239 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v5241 = stablehlo.reshape %v111 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v5242 = stablehlo.constant dense<0.0> : tensor<32x96x112x112xf32>
    %v5243 = stablehlo.constant dense<6.0> : tensor<32x96x112x112xf32>
    %v5244 = stablehlo.compare GT, %v5241, %v5242 : (tensor<32x96x112x112xf32>, tensor<32x96x112x112xf32>) -> tensor<32x96x112x112xi1>
    %v5245 = stablehlo.compare LT, %v5241, %v5243 : (tensor<32x96x112x112xf32>, tensor<32x96x112x112xf32>) -> tensor<32x96x112x112xi1>
    %v5246 = stablehlo.and %v5244, %v5245 : tensor<32x96x112x112xi1>
    %v5247 = stablehlo.select %v5246, %v5240, %v5242 : tensor<32x96x112x112xi1>, tensor<32x96x112x112xf32>
    %v5248 = stablehlo.reshape %v5247 : (tensor<32x96x112x112xf32>) -> tensor<32x1204224xf32>
    %v5249 = stablehlo.reshape %v5248 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v5250 = stablehlo.reshape %v91 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v5251 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5252 = stablehlo.constant dense<12544.0> : tensor<32x96x112x112xf32>
    %v5253 = stablehlo.constant dense<1.0e-5> : tensor<32x96x112x112xf32>
    %v5254 = stablehlo.reduce(%v5250 init: %v5251) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v5255 = stablehlo.broadcast_in_dim %v5254, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x112x112xf32>
    %v5256 = stablehlo.divide %v5255, %v5252 : tensor<32x96x112x112xf32>
    %v5257 = stablehlo.subtract %v5250, %v5256 : tensor<32x96x112x112xf32>
    %v5258 = stablehlo.multiply %v5257, %v5257 : tensor<32x96x112x112xf32>
    %v5259 = stablehlo.reduce(%v5258 init: %v5251) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v5260 = stablehlo.broadcast_in_dim %v5259, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x112x112xf32>
    %v5261 = stablehlo.divide %v5260, %v5252 : tensor<32x96x112x112xf32>
    %v5262 = stablehlo.add %v5261, %v5253 : tensor<32x96x112x112xf32>
    %v5263 = stablehlo.rsqrt %v5262 : tensor<32x96x112x112xf32>
    %v5264 = stablehlo.multiply %v5257, %v5263 : tensor<32x96x112x112xf32>
    %v5265 = stablehlo.broadcast_in_dim %ge2, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v5266 = stablehlo.multiply %v5265, %v5249 : tensor<32x96x112x112xf32>
    %v5267 = stablehlo.reduce(%v5266 init: %v5251) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v5268 = stablehlo.broadcast_in_dim %v5267, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x112x112xf32>
    %v5269 = stablehlo.multiply %v5264, %v5266 : tensor<32x96x112x112xf32>
    %v5270 = stablehlo.reduce(%v5269 init: %v5251) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v5271 = stablehlo.broadcast_in_dim %v5270, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x112x112xf32>
    %v5272 = stablehlo.multiply %v5266, %v5252 : tensor<32x96x112x112xf32>
    %v5273 = stablehlo.subtract %v5272, %v5268 : tensor<32x96x112x112xf32>
    %v5274 = stablehlo.multiply %v5264, %v5271 : tensor<32x96x112x112xf32>
    %v5275 = stablehlo.subtract %v5273, %v5274 : tensor<32x96x112x112xf32>
    %v5276 = stablehlo.divide %v5263, %v5252 : tensor<32x96x112x112xf32>
    %v5277 = stablehlo.multiply %v5276, %v5275 : tensor<32x96x112x112xf32>
    %v5278 = stablehlo.reshape %v5277 : (tensor<32x96x112x112xf32>) -> tensor<32x1204224xf32>
    %v5279 = stablehlo.reshape %v5278 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v5280 = stablehlo.transpose %We2, dims = [1, 0, 2, 3] : (tensor<96x16x1x1xf32>) -> tensor<16x96x1x1xf32>
    %v5281 = stablehlo.reverse %v5280, dims = [2, 3] : tensor<16x96x1x1xf32>
    %v5282 = stablehlo.convolution(%v5279, %v5281)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x112x112xf32>, tensor<16x96x1x1xf32>) -> tensor<32x16x112x112xf32>
    %v5283 = stablehlo.reshape %v5282 : (tensor<32x16x112x112xf32>) -> tensor<32x200704xf32>
    %v5284 = stablehlo.reshape %v86 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v5285 = stablehlo.reshape %v5278 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v5286 = stablehlo.transpose %v5284, dims = [1, 0, 2, 3] : (tensor<32x16x112x112xf32>) -> tensor<16x32x112x112xf32>
    %v5287 = stablehlo.transpose %v5285, dims = [1, 0, 2, 3] : (tensor<32x96x112x112xf32>) -> tensor<96x32x112x112xf32>
    %v5288 = stablehlo.convolution(%v5286, %v5287)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x32x112x112xf32>, tensor<96x32x112x112xf32>) -> tensor<16x96x1x1xf32>
    %v5289 = stablehlo.transpose %v5288, dims = [1, 0, 2, 3] : (tensor<16x96x1x1xf32>) -> tensor<96x16x1x1xf32>
    %v5290 = stablehlo.constant dense<0.3> : tensor<96x16x1x1xf32>
    %v5291 = stablehlo.multiply %v5289, %v5290 : tensor<96x16x1x1xf32>
    %v5292 = stablehlo.subtract %We2, %v5291 : tensor<96x16x1x1xf32>
    %v5293 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5294 = stablehlo.reshape %v91 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v5295 = stablehlo.constant dense<12544.0> : tensor<32x96x112x112xf32>
    %v5296 = stablehlo.constant dense<1.0e-5> : tensor<32x96x112x112xf32>
    %v5297 = stablehlo.reduce(%v5294 init: %v5293) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v5298 = stablehlo.broadcast_in_dim %v5297, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x112x112xf32>
    %v5299 = stablehlo.divide %v5298, %v5295 : tensor<32x96x112x112xf32>
    %v5300 = stablehlo.subtract %v5294, %v5299 : tensor<32x96x112x112xf32>
    %v5301 = stablehlo.multiply %v5300, %v5300 : tensor<32x96x112x112xf32>
    %v5302 = stablehlo.reduce(%v5301 init: %v5293) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v5303 = stablehlo.broadcast_in_dim %v5302, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x112x112xf32>
    %v5304 = stablehlo.divide %v5303, %v5295 : tensor<32x96x112x112xf32>
    %v5305 = stablehlo.add %v5304, %v5296 : tensor<32x96x112x112xf32>
    %v5306 = stablehlo.rsqrt %v5305 : tensor<32x96x112x112xf32>
    %v5307 = stablehlo.multiply %v5300, %v5306 : tensor<32x96x112x112xf32>
    %v5308 = stablehlo.reshape %v5248 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v5309 = stablehlo.multiply %v5308, %v5307 : tensor<32x96x112x112xf32>
    %v5310 = stablehlo.reduce(%v5309 init: %v5293) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v5311 = stablehlo.constant dense<0.3> : tensor<96xf32>
    %v5312 = stablehlo.multiply %v5310, %v5311 : tensor<96xf32>
    %v5313 = stablehlo.subtract %ge2, %v5312 : tensor<96xf32>
    %v5314 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5315 = stablehlo.reshape %v5248 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v5316 = stablehlo.reduce(%v5315 init: %v5314) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v5317 = stablehlo.constant dense<0.3> : tensor<96xf32>
    %v5318 = stablehlo.multiply %v5316, %v5317 : tensor<96xf32>
    %v5319 = stablehlo.subtract %bte2, %v5318 : tensor<96xf32>
    %v5320 = stablehlo.reshape %v117 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v5321 = stablehlo.reshape %v5233 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v5322 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5323 = stablehlo.pad %v5321, %v5322, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96x112x112xf32>
    %v5324 = stablehlo.transpose %v5320, dims = [1, 0, 2, 3] : (tensor<32x96x112x112xf32>) -> tensor<96x32x112x112xf32>
    %v5325 = stablehlo.transpose %v5323, dims = [1, 0, 2, 3] : (tensor<32x96x112x112xf32>) -> tensor<96x32x112x112xf32>
    %v5326 = stablehlo.convolution(%v5324, %v5325)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 96 : i64, feature_group_count = 1 : i64} : (tensor<96x32x112x112xf32>, tensor<96x32x112x112xf32>) -> tensor<1x96x3x3xf32>
    %v5327 = stablehlo.reshape %v5326 : (tensor<1x96x3x3xf32>) -> tensor<96x1x3x3xf32>
    %v5328 = stablehlo.constant dense<0.3> : tensor<96x1x3x3xf32>
    %v5329 = stablehlo.multiply %v5327, %v5328 : tensor<96x1x3x3xf32>
    %v5330 = stablehlo.subtract %Wd2, %v5329 : tensor<96x1x3x3xf32>
    %v5331 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5332 = stablehlo.reshape %v122 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v5333 = stablehlo.constant dense<3136.0> : tensor<32x96x56x56xf32>
    %v5334 = stablehlo.constant dense<1.0e-5> : tensor<32x96x56x56xf32>
    %v5335 = stablehlo.reduce(%v5332 init: %v5331) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v5336 = stablehlo.broadcast_in_dim %v5335, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v5337 = stablehlo.divide %v5336, %v5333 : tensor<32x96x56x56xf32>
    %v5338 = stablehlo.subtract %v5332, %v5337 : tensor<32x96x56x56xf32>
    %v5339 = stablehlo.multiply %v5338, %v5338 : tensor<32x96x56x56xf32>
    %v5340 = stablehlo.reduce(%v5339 init: %v5331) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v5341 = stablehlo.broadcast_in_dim %v5340, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v5342 = stablehlo.divide %v5341, %v5333 : tensor<32x96x56x56xf32>
    %v5343 = stablehlo.add %v5342, %v5334 : tensor<32x96x56x56xf32>
    %v5344 = stablehlo.rsqrt %v5343 : tensor<32x96x56x56xf32>
    %v5345 = stablehlo.multiply %v5338, %v5344 : tensor<32x96x56x56xf32>
    %v5346 = stablehlo.reshape %v5203 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v5347 = stablehlo.multiply %v5346, %v5345 : tensor<32x96x56x56xf32>
    %v5348 = stablehlo.reduce(%v5347 init: %v5331) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v5349 = stablehlo.constant dense<0.3> : tensor<96xf32>
    %v5350 = stablehlo.multiply %v5348, %v5349 : tensor<96xf32>
    %v5351 = stablehlo.subtract %gd2, %v5350 : tensor<96xf32>
    %v5352 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5353 = stablehlo.reshape %v5203 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v5354 = stablehlo.reduce(%v5353 init: %v5352) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v5355 = stablehlo.constant dense<0.3> : tensor<96xf32>
    %v5356 = stablehlo.multiply %v5354, %v5355 : tensor<96xf32>
    %v5357 = stablehlo.subtract %btd2, %v5356 : tensor<96xf32>
    %v5358 = stablehlo.reshape %v148 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v5359 = stablehlo.reshape %v5189 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v5360 = stablehlo.transpose %v5358, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v5361 = stablehlo.transpose %v5359, dims = [1, 0, 2, 3] : (tensor<32x24x56x56xf32>) -> tensor<24x32x56x56xf32>
    %v5362 = stablehlo.convolution(%v5360, %v5361)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<24x32x56x56xf32>) -> tensor<96x24x1x1xf32>
    %v5363 = stablehlo.transpose %v5362, dims = [1, 0, 2, 3] : (tensor<96x24x1x1xf32>) -> tensor<24x96x1x1xf32>
    %v5364 = stablehlo.constant dense<0.3> : tensor<24x96x1x1xf32>
    %v5365 = stablehlo.multiply %v5363, %v5364 : tensor<24x96x1x1xf32>
    %v5366 = stablehlo.subtract %Wp2, %v5365 : tensor<24x96x1x1xf32>
    %v5367 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5368 = stablehlo.reshape %v153 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v5369 = stablehlo.constant dense<3136.0> : tensor<32x24x56x56xf32>
    %v5370 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v5371 = stablehlo.reduce(%v5368 init: %v5367) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v5372 = stablehlo.broadcast_in_dim %v5371, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v5373 = stablehlo.divide %v5372, %v5369 : tensor<32x24x56x56xf32>
    %v5374 = stablehlo.subtract %v5368, %v5373 : tensor<32x24x56x56xf32>
    %v5375 = stablehlo.multiply %v5374, %v5374 : tensor<32x24x56x56xf32>
    %v5376 = stablehlo.reduce(%v5375 init: %v5367) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v5377 = stablehlo.broadcast_in_dim %v5376, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v5378 = stablehlo.divide %v5377, %v5369 : tensor<32x24x56x56xf32>
    %v5379 = stablehlo.add %v5378, %v5370 : tensor<32x24x56x56xf32>
    %v5380 = stablehlo.rsqrt %v5379 : tensor<32x24x56x56xf32>
    %v5381 = stablehlo.multiply %v5374, %v5380 : tensor<32x24x56x56xf32>
    %v5382 = stablehlo.reshape %v5051 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v5383 = stablehlo.multiply %v5382, %v5381 : tensor<32x24x56x56xf32>
    %v5384 = stablehlo.reduce(%v5383 init: %v5367) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v5385 = stablehlo.constant dense<0.3> : tensor<24xf32>
    %v5386 = stablehlo.multiply %v5384, %v5385 : tensor<24xf32>
    %v5387 = stablehlo.subtract %gp2, %v5386 : tensor<24xf32>
    %v5388 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5389 = stablehlo.reshape %v5051 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v5390 = stablehlo.reduce(%v5389 init: %v5388) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v5391 = stablehlo.constant dense<0.3> : tensor<24xf32>
    %v5392 = stablehlo.multiply %v5390, %v5391 : tensor<24xf32>
    %v5393 = stablehlo.subtract %btp2, %v5392 : tensor<24xf32>
    %v5394 = stablehlo.reshape %v5283 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v5395 = stablehlo.reshape %v66 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v5396 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5397 = stablehlo.constant dense<12544.0> : tensor<32x16x112x112xf32>
    %v5398 = stablehlo.constant dense<1.0e-5> : tensor<32x16x112x112xf32>
    %v5399 = stablehlo.reduce(%v5395 init: %v5396) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<32x16xf32>
    %v5400 = stablehlo.broadcast_in_dim %v5399, dims = [0, 1] : (tensor<32x16xf32>) -> tensor<32x16x112x112xf32>
    %v5401 = stablehlo.divide %v5400, %v5397 : tensor<32x16x112x112xf32>
    %v5402 = stablehlo.subtract %v5395, %v5401 : tensor<32x16x112x112xf32>
    %v5403 = stablehlo.multiply %v5402, %v5402 : tensor<32x16x112x112xf32>
    %v5404 = stablehlo.reduce(%v5403 init: %v5396) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<32x16xf32>
    %v5405 = stablehlo.broadcast_in_dim %v5404, dims = [0, 1] : (tensor<32x16xf32>) -> tensor<32x16x112x112xf32>
    %v5406 = stablehlo.divide %v5405, %v5397 : tensor<32x16x112x112xf32>
    %v5407 = stablehlo.add %v5406, %v5398 : tensor<32x16x112x112xf32>
    %v5408 = stablehlo.rsqrt %v5407 : tensor<32x16x112x112xf32>
    %v5409 = stablehlo.multiply %v5402, %v5408 : tensor<32x16x112x112xf32>
    %v5410 = stablehlo.broadcast_in_dim %gp1, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v5411 = stablehlo.multiply %v5410, %v5394 : tensor<32x16x112x112xf32>
    %v5412 = stablehlo.reduce(%v5411 init: %v5396) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<32x16xf32>
    %v5413 = stablehlo.broadcast_in_dim %v5412, dims = [0, 1] : (tensor<32x16xf32>) -> tensor<32x16x112x112xf32>
    %v5414 = stablehlo.multiply %v5409, %v5411 : tensor<32x16x112x112xf32>
    %v5415 = stablehlo.reduce(%v5414 init: %v5396) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<32x16xf32>
    %v5416 = stablehlo.broadcast_in_dim %v5415, dims = [0, 1] : (tensor<32x16xf32>) -> tensor<32x16x112x112xf32>
    %v5417 = stablehlo.multiply %v5411, %v5397 : tensor<32x16x112x112xf32>
    %v5418 = stablehlo.subtract %v5417, %v5413 : tensor<32x16x112x112xf32>
    %v5419 = stablehlo.multiply %v5409, %v5416 : tensor<32x16x112x112xf32>
    %v5420 = stablehlo.subtract %v5418, %v5419 : tensor<32x16x112x112xf32>
    %v5421 = stablehlo.divide %v5408, %v5397 : tensor<32x16x112x112xf32>
    %v5422 = stablehlo.multiply %v5421, %v5420 : tensor<32x16x112x112xf32>
    %v5423 = stablehlo.reshape %v5422 : (tensor<32x16x112x112xf32>) -> tensor<32x200704xf32>
    %v5424 = stablehlo.reshape %v5423 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v5425 = stablehlo.transpose %Wp1, dims = [1, 0, 2, 3] : (tensor<16x32x1x1xf32>) -> tensor<32x16x1x1xf32>
    %v5426 = stablehlo.reverse %v5425, dims = [2, 3] : tensor<32x16x1x1xf32>
    %v5427 = stablehlo.convolution(%v5424, %v5426)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x16x112x112xf32>, tensor<32x16x1x1xf32>) -> tensor<32x32x112x112xf32>
    %v5428 = stablehlo.reshape %v5427 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v5429 = stablehlo.reshape %v5428 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5430 = stablehlo.reshape %v55 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5431 = stablehlo.constant dense<0.0> : tensor<32x32x112x112xf32>
    %v5432 = stablehlo.constant dense<6.0> : tensor<32x32x112x112xf32>
    %v5433 = stablehlo.compare GT, %v5430, %v5431 : (tensor<32x32x112x112xf32>, tensor<32x32x112x112xf32>) -> tensor<32x32x112x112xi1>
    %v5434 = stablehlo.compare LT, %v5430, %v5432 : (tensor<32x32x112x112xf32>, tensor<32x32x112x112xf32>) -> tensor<32x32x112x112xi1>
    %v5435 = stablehlo.and %v5433, %v5434 : tensor<32x32x112x112xi1>
    %v5436 = stablehlo.select %v5435, %v5429, %v5431 : tensor<32x32x112x112xi1>, tensor<32x32x112x112xf32>
    %v5437 = stablehlo.reshape %v5436 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v5438 = stablehlo.reshape %v5437 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5439 = stablehlo.reshape %v35 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5440 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5441 = stablehlo.constant dense<12544.0> : tensor<32x32x112x112xf32>
    %v5442 = stablehlo.constant dense<1.0e-5> : tensor<32x32x112x112xf32>
    %v5443 = stablehlo.reduce(%v5439 init: %v5440) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v5444 = stablehlo.broadcast_in_dim %v5443, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x112x112xf32>
    %v5445 = stablehlo.divide %v5444, %v5441 : tensor<32x32x112x112xf32>
    %v5446 = stablehlo.subtract %v5439, %v5445 : tensor<32x32x112x112xf32>
    %v5447 = stablehlo.multiply %v5446, %v5446 : tensor<32x32x112x112xf32>
    %v5448 = stablehlo.reduce(%v5447 init: %v5440) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v5449 = stablehlo.broadcast_in_dim %v5448, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x112x112xf32>
    %v5450 = stablehlo.divide %v5449, %v5441 : tensor<32x32x112x112xf32>
    %v5451 = stablehlo.add %v5450, %v5442 : tensor<32x32x112x112xf32>
    %v5452 = stablehlo.rsqrt %v5451 : tensor<32x32x112x112xf32>
    %v5453 = stablehlo.multiply %v5446, %v5452 : tensor<32x32x112x112xf32>
    %v5454 = stablehlo.broadcast_in_dim %gd1, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v5455 = stablehlo.multiply %v5454, %v5438 : tensor<32x32x112x112xf32>
    %v5456 = stablehlo.reduce(%v5455 init: %v5440) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v5457 = stablehlo.broadcast_in_dim %v5456, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x112x112xf32>
    %v5458 = stablehlo.multiply %v5453, %v5455 : tensor<32x32x112x112xf32>
    %v5459 = stablehlo.reduce(%v5458 init: %v5440) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v5460 = stablehlo.broadcast_in_dim %v5459, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x112x112xf32>
    %v5461 = stablehlo.multiply %v5455, %v5441 : tensor<32x32x112x112xf32>
    %v5462 = stablehlo.subtract %v5461, %v5457 : tensor<32x32x112x112xf32>
    %v5463 = stablehlo.multiply %v5453, %v5460 : tensor<32x32x112x112xf32>
    %v5464 = stablehlo.subtract %v5462, %v5463 : tensor<32x32x112x112xf32>
    %v5465 = stablehlo.divide %v5452, %v5441 : tensor<32x32x112x112xf32>
    %v5466 = stablehlo.multiply %v5465, %v5464 : tensor<32x32x112x112xf32>
    %v5467 = stablehlo.reshape %v5466 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v5468 = stablehlo.reshape %v5467 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5469 = stablehlo.reverse %Wd1, dims = [2, 3] : tensor<32x1x3x3xf32>
    %v5470 = stablehlo.convolution(%v5468, %v5469)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 32 : i64} : (tensor<32x32x112x112xf32>, tensor<32x1x3x3xf32>) -> tensor<32x32x112x112xf32>
    %v5471 = stablehlo.reshape %v5470 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v5472 = stablehlo.reshape %v30 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5473 = stablehlo.reshape %v5467 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5474 = stablehlo.transpose %v5472, dims = [1, 0, 2, 3] : (tensor<32x32x112x112xf32>) -> tensor<32x32x112x112xf32>
    %v5475 = stablehlo.transpose %v5473, dims = [1, 0, 2, 3] : (tensor<32x32x112x112xf32>) -> tensor<32x32x112x112xf32>
    %v5476 = stablehlo.convolution(%v5474, %v5475)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 32 : i64, feature_group_count = 1 : i64} : (tensor<32x32x112x112xf32>, tensor<32x32x112x112xf32>) -> tensor<1x32x3x3xf32>
    %v5477 = stablehlo.reshape %v5476 : (tensor<1x32x3x3xf32>) -> tensor<32x1x3x3xf32>
    %v5478 = stablehlo.constant dense<0.3> : tensor<32x1x3x3xf32>
    %v5479 = stablehlo.multiply %v5477, %v5478 : tensor<32x1x3x3xf32>
    %v5480 = stablehlo.subtract %Wd1, %v5479 : tensor<32x1x3x3xf32>
    %v5481 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5482 = stablehlo.reshape %v35 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5483 = stablehlo.constant dense<12544.0> : tensor<32x32x112x112xf32>
    %v5484 = stablehlo.constant dense<1.0e-5> : tensor<32x32x112x112xf32>
    %v5485 = stablehlo.reduce(%v5482 init: %v5481) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v5486 = stablehlo.broadcast_in_dim %v5485, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x112x112xf32>
    %v5487 = stablehlo.divide %v5486, %v5483 : tensor<32x32x112x112xf32>
    %v5488 = stablehlo.subtract %v5482, %v5487 : tensor<32x32x112x112xf32>
    %v5489 = stablehlo.multiply %v5488, %v5488 : tensor<32x32x112x112xf32>
    %v5490 = stablehlo.reduce(%v5489 init: %v5481) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v5491 = stablehlo.broadcast_in_dim %v5490, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x112x112xf32>
    %v5492 = stablehlo.divide %v5491, %v5483 : tensor<32x32x112x112xf32>
    %v5493 = stablehlo.add %v5492, %v5484 : tensor<32x32x112x112xf32>
    %v5494 = stablehlo.rsqrt %v5493 : tensor<32x32x112x112xf32>
    %v5495 = stablehlo.multiply %v5488, %v5494 : tensor<32x32x112x112xf32>
    %v5496 = stablehlo.reshape %v5437 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5497 = stablehlo.multiply %v5496, %v5495 : tensor<32x32x112x112xf32>
    %v5498 = stablehlo.reduce(%v5497 init: %v5481) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v5499 = stablehlo.constant dense<0.3> : tensor<32xf32>
    %v5500 = stablehlo.multiply %v5498, %v5499 : tensor<32xf32>
    %v5501 = stablehlo.subtract %gd1, %v5500 : tensor<32xf32>
    %v5502 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5503 = stablehlo.reshape %v5437 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5504 = stablehlo.reduce(%v5503 init: %v5502) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v5505 = stablehlo.constant dense<0.3> : tensor<32xf32>
    %v5506 = stablehlo.multiply %v5504, %v5505 : tensor<32xf32>
    %v5507 = stablehlo.subtract %btd1, %v5506 : tensor<32xf32>
    %v5508 = stablehlo.reshape %v61 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5509 = stablehlo.reshape %v5423 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v5510 = stablehlo.transpose %v5508, dims = [1, 0, 2, 3] : (tensor<32x32x112x112xf32>) -> tensor<32x32x112x112xf32>
    %v5511 = stablehlo.transpose %v5509, dims = [1, 0, 2, 3] : (tensor<32x16x112x112xf32>) -> tensor<16x32x112x112xf32>
    %v5512 = stablehlo.convolution(%v5510, %v5511)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x112x112xf32>, tensor<16x32x112x112xf32>) -> tensor<32x16x1x1xf32>
    %v5513 = stablehlo.transpose %v5512, dims = [1, 0, 2, 3] : (tensor<32x16x1x1xf32>) -> tensor<16x32x1x1xf32>
    %v5514 = stablehlo.constant dense<0.3> : tensor<16x32x1x1xf32>
    %v5515 = stablehlo.multiply %v5513, %v5514 : tensor<16x32x1x1xf32>
    %v5516 = stablehlo.subtract %Wp1, %v5515 : tensor<16x32x1x1xf32>
    %v5517 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5518 = stablehlo.reshape %v66 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v5519 = stablehlo.constant dense<12544.0> : tensor<32x16x112x112xf32>
    %v5520 = stablehlo.constant dense<1.0e-5> : tensor<32x16x112x112xf32>
    %v5521 = stablehlo.reduce(%v5518 init: %v5517) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<32x16xf32>
    %v5522 = stablehlo.broadcast_in_dim %v5521, dims = [0, 1] : (tensor<32x16xf32>) -> tensor<32x16x112x112xf32>
    %v5523 = stablehlo.divide %v5522, %v5519 : tensor<32x16x112x112xf32>
    %v5524 = stablehlo.subtract %v5518, %v5523 : tensor<32x16x112x112xf32>
    %v5525 = stablehlo.multiply %v5524, %v5524 : tensor<32x16x112x112xf32>
    %v5526 = stablehlo.reduce(%v5525 init: %v5517) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<32x16xf32>
    %v5527 = stablehlo.broadcast_in_dim %v5526, dims = [0, 1] : (tensor<32x16xf32>) -> tensor<32x16x112x112xf32>
    %v5528 = stablehlo.divide %v5527, %v5519 : tensor<32x16x112x112xf32>
    %v5529 = stablehlo.add %v5528, %v5520 : tensor<32x16x112x112xf32>
    %v5530 = stablehlo.rsqrt %v5529 : tensor<32x16x112x112xf32>
    %v5531 = stablehlo.multiply %v5524, %v5530 : tensor<32x16x112x112xf32>
    %v5532 = stablehlo.reshape %v5283 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v5533 = stablehlo.multiply %v5532, %v5531 : tensor<32x16x112x112xf32>
    %v5534 = stablehlo.reduce(%v5533 init: %v5517) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v5535 = stablehlo.constant dense<0.3> : tensor<16xf32>
    %v5536 = stablehlo.multiply %v5534, %v5535 : tensor<16xf32>
    %v5537 = stablehlo.subtract %gp1, %v5536 : tensor<16xf32>
    %v5538 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5539 = stablehlo.reshape %v5283 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v5540 = stablehlo.reduce(%v5539 init: %v5538) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v5541 = stablehlo.constant dense<0.3> : tensor<16xf32>
    %v5542 = stablehlo.multiply %v5540, %v5541 : tensor<16xf32>
    %v5543 = stablehlo.subtract %btp1, %v5542 : tensor<16xf32>
    %v5544 = stablehlo.reshape %v5471 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5545 = stablehlo.reshape %v24 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5546 = stablehlo.constant dense<0.0> : tensor<32x32x112x112xf32>
    %v5547 = stablehlo.constant dense<6.0> : tensor<32x32x112x112xf32>
    %v5548 = stablehlo.compare GT, %v5545, %v5546 : (tensor<32x32x112x112xf32>, tensor<32x32x112x112xf32>) -> tensor<32x32x112x112xi1>
    %v5549 = stablehlo.compare LT, %v5545, %v5547 : (tensor<32x32x112x112xf32>, tensor<32x32x112x112xf32>) -> tensor<32x32x112x112xi1>
    %v5550 = stablehlo.and %v5548, %v5549 : tensor<32x32x112x112xi1>
    %v5551 = stablehlo.select %v5550, %v5544, %v5546 : tensor<32x32x112x112xi1>, tensor<32x32x112x112xf32>
    %v5552 = stablehlo.reshape %v5551 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v5553 = stablehlo.reshape %v5552 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5554 = stablehlo.reshape %v4 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5555 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5556 = stablehlo.constant dense<12544.0> : tensor<32x32x112x112xf32>
    %v5557 = stablehlo.constant dense<1.0e-5> : tensor<32x32x112x112xf32>
    %v5558 = stablehlo.reduce(%v5554 init: %v5555) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v5559 = stablehlo.broadcast_in_dim %v5558, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x112x112xf32>
    %v5560 = stablehlo.divide %v5559, %v5556 : tensor<32x32x112x112xf32>
    %v5561 = stablehlo.subtract %v5554, %v5560 : tensor<32x32x112x112xf32>
    %v5562 = stablehlo.multiply %v5561, %v5561 : tensor<32x32x112x112xf32>
    %v5563 = stablehlo.reduce(%v5562 init: %v5555) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v5564 = stablehlo.broadcast_in_dim %v5563, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x112x112xf32>
    %v5565 = stablehlo.divide %v5564, %v5556 : tensor<32x32x112x112xf32>
    %v5566 = stablehlo.add %v5565, %v5557 : tensor<32x32x112x112xf32>
    %v5567 = stablehlo.rsqrt %v5566 : tensor<32x32x112x112xf32>
    %v5568 = stablehlo.multiply %v5561, %v5567 : tensor<32x32x112x112xf32>
    %v5569 = stablehlo.broadcast_in_dim %gs, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v5570 = stablehlo.multiply %v5569, %v5553 : tensor<32x32x112x112xf32>
    %v5571 = stablehlo.reduce(%v5570 init: %v5555) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v5572 = stablehlo.broadcast_in_dim %v5571, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x112x112xf32>
    %v5573 = stablehlo.multiply %v5568, %v5570 : tensor<32x32x112x112xf32>
    %v5574 = stablehlo.reduce(%v5573 init: %v5555) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v5575 = stablehlo.broadcast_in_dim %v5574, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x112x112xf32>
    %v5576 = stablehlo.multiply %v5570, %v5556 : tensor<32x32x112x112xf32>
    %v5577 = stablehlo.subtract %v5576, %v5572 : tensor<32x32x112x112xf32>
    %v5578 = stablehlo.multiply %v5568, %v5575 : tensor<32x32x112x112xf32>
    %v5579 = stablehlo.subtract %v5577, %v5578 : tensor<32x32x112x112xf32>
    %v5580 = stablehlo.divide %v5567, %v5556 : tensor<32x32x112x112xf32>
    %v5581 = stablehlo.multiply %v5580, %v5579 : tensor<32x32x112x112xf32>
    %v5582 = stablehlo.reshape %v5581 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v5583 = stablehlo.reshape %x : (tensor<32x150528xf32>) -> tensor<32x3x224x224xf32>
    %v5584 = stablehlo.reshape %v5582 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5585 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5586 = stablehlo.pad %v5584, %v5585, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32x224x224xf32>
    %v5587 = stablehlo.transpose %v5583, dims = [1, 0, 2, 3] : (tensor<32x3x224x224xf32>) -> tensor<3x32x224x224xf32>
    %v5588 = stablehlo.transpose %v5586, dims = [1, 0, 2, 3] : (tensor<32x32x224x224xf32>) -> tensor<32x32x224x224xf32>
    %v5589 = stablehlo.convolution(%v5587, %v5588)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x32x224x224xf32>, tensor<32x32x224x224xf32>) -> tensor<3x32x3x3xf32>
    %v5590 = stablehlo.transpose %v5589, dims = [1, 0, 2, 3] : (tensor<3x32x3x3xf32>) -> tensor<32x3x3x3xf32>
    %v5591 = stablehlo.constant dense<0.3> : tensor<32x3x3x3xf32>
    %v5592 = stablehlo.multiply %v5590, %v5591 : tensor<32x3x3x3xf32>
    %v5593 = stablehlo.subtract %Ws, %v5592 : tensor<32x3x3x3xf32>
    %v5594 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5595 = stablehlo.reshape %v4 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5596 = stablehlo.constant dense<12544.0> : tensor<32x32x112x112xf32>
    %v5597 = stablehlo.constant dense<1.0e-5> : tensor<32x32x112x112xf32>
    %v5598 = stablehlo.reduce(%v5595 init: %v5594) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v5599 = stablehlo.broadcast_in_dim %v5598, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x112x112xf32>
    %v5600 = stablehlo.divide %v5599, %v5596 : tensor<32x32x112x112xf32>
    %v5601 = stablehlo.subtract %v5595, %v5600 : tensor<32x32x112x112xf32>
    %v5602 = stablehlo.multiply %v5601, %v5601 : tensor<32x32x112x112xf32>
    %v5603 = stablehlo.reduce(%v5602 init: %v5594) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v5604 = stablehlo.broadcast_in_dim %v5603, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x112x112xf32>
    %v5605 = stablehlo.divide %v5604, %v5596 : tensor<32x32x112x112xf32>
    %v5606 = stablehlo.add %v5605, %v5597 : tensor<32x32x112x112xf32>
    %v5607 = stablehlo.rsqrt %v5606 : tensor<32x32x112x112xf32>
    %v5608 = stablehlo.multiply %v5601, %v5607 : tensor<32x32x112x112xf32>
    %v5609 = stablehlo.reshape %v5552 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5610 = stablehlo.multiply %v5609, %v5608 : tensor<32x32x112x112xf32>
    %v5611 = stablehlo.reduce(%v5610 init: %v5594) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v5612 = stablehlo.constant dense<0.3> : tensor<32xf32>
    %v5613 = stablehlo.multiply %v5611, %v5612 : tensor<32xf32>
    %v5614 = stablehlo.subtract %gs, %v5613 : tensor<32xf32>
    %v5615 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5616 = stablehlo.reshape %v5552 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5617 = stablehlo.reduce(%v5616 init: %v5615) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v5618 = stablehlo.constant dense<0.3> : tensor<32xf32>
    %v5619 = stablehlo.multiply %v5617, %v5618 : tensor<32xf32>
    %v5620 = stablehlo.subtract %bts, %v5619 : tensor<32xf32>
    return %v5593, %v5614, %v5620, %v5480, %v5501, %v5507, %v5516, %v5537, %v5543, %v5292, %v5313, %v5319, %v5330, %v5351, %v5357, %v5366, %v5387, %v5393, %v5060, %v5081, %v5087, %v5096, %v5117, %v5123, %v5132, %v5153, %v5159, %v4824, %v4845, %v4851, %v4862, %v4883, %v4889, %v4898, %v4919, %v4925, %v4592, %v4613, %v4619, %v4628, %v4649, %v4655, %v4664, %v4685, %v4691, %v4358, %v4379, %v4385, %v4394, %v4415, %v4421, %v4430, %v4451, %v4457, %v4122, %v4143, %v4149, %v4160, %v4181, %v4187, %v4196, %v4217, %v4223, %v3890, %v3911, %v3917, %v3926, %v3947, %v3953, %v3962, %v3983, %v3989, %v3656, %v3677, %v3683, %v3692, %v3713, %v3719, %v3728, %v3749, %v3755, %v3422, %v3443, %v3449, %v3458, %v3479, %v3485, %v3494, %v3515, %v3521, %v3188, %v3209, %v3215, %v3224, %v3245, %v3251, %v3260, %v3281, %v3287, %v2958, %v2979, %v2985, %v2994, %v3015, %v3021, %v3030, %v3051, %v3057, %v2724, %v2745, %v2751, %v2760, %v2781, %v2787, %v2796, %v2817, %v2823, %v2488, %v2509, %v2515, %v2526, %v2547, %v2553, %v2562, %v2583, %v2589, %v2256, %v2277, %v2283, %v2292, %v2313, %v2319, %v2328, %v2349, %v2355, %v2022, %v2043, %v2049, %v2058, %v2079, %v2085, %v2094, %v2115, %v2121, %v1788, %v1809, %v1815, %v1824, %v1845, %v1851, %v1860, %v1881, %v1887, %v1630, %v1651, %v1657, %v1572, %v1577 : tensor<32x3x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x1x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<16x32x1x1xf32>, tensor<16xf32>, tensor<16xf32>, tensor<96x16x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x3x3xf32>, tensor<96xf32>, tensor<96xf32>, tensor<24x96x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144x24x1x1xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x1x3x3xf32>, tensor<144xf32>, tensor<144xf32>, tensor<24x144x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144x24x1x1xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x1x3x3xf32>, tensor<144xf32>, tensor<144xf32>, tensor<32x144x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<192x32x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x3x3xf32>, tensor<192xf32>, tensor<192xf32>, tensor<32x192x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<192x32x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x3x3xf32>, tensor<192xf32>, tensor<192xf32>, tensor<32x192x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<192x32x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x3x3xf32>, tensor<192xf32>, tensor<192xf32>, tensor<64x192x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64x384x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64x384x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64x384x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<576x96x1x1xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576x1x3x3xf32>, tensor<576xf32>, tensor<576xf32>, tensor<96x576x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<576x96x1x1xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576x1x3x3xf32>, tensor<576xf32>, tensor<576xf32>, tensor<96x576x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<576x96x1x1xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576x1x3x3xf32>, tensor<576xf32>, tensor<576xf32>, tensor<160x576x1x1xf32>, tensor<160xf32>, tensor<160xf32>, tensor<960x160x1x1xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960x1x3x3xf32>, tensor<960xf32>, tensor<960xf32>, tensor<160x960x1x1xf32>, tensor<160xf32>, tensor<160xf32>, tensor<960x160x1x1xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960x1x3x3xf32>, tensor<960xf32>, tensor<960xf32>, tensor<160x960x1x1xf32>, tensor<160xf32>, tensor<160xf32>, tensor<960x160x1x1xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960x1x3x3xf32>, tensor<960xf32>, tensor<960xf32>, tensor<320x960x1x1xf32>, tensor<320xf32>, tensor<320xf32>, tensor<1280x320x1x1xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280x10xf32>, tensor<10xf32>
  }
}

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
    %v25 = stablehlo.constant dense<0.0> : tensor<32x401408xf32>
    %v26 = stablehlo.constant dense<6.0> : tensor<32x401408xf32>
    %v27 = stablehlo.maximum %v24, %v25 : tensor<32x401408xf32>
    %v28 = stablehlo.minimum %v27, %v26 : tensor<32x401408xf32>
    %v29 = stablehlo.reshape %v28 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v30 = stablehlo.convolution(%v29, %Wd1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 32 : i64} : (tensor<32x32x112x112xf32>, tensor<32x1x3x3xf32>) -> tensor<32x32x112x112xf32>
    %v31 = stablehlo.broadcast_in_dim %zb32, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v32 = stablehlo.add %v30, %v31 : tensor<32x32x112x112xf32>
    %v33 = stablehlo.reshape %v32 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v34 = stablehlo.reshape %v33 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v35 = stablehlo.constant dense<0.0> : tensor<f32>
    %v36 = stablehlo.constant dense<12544.0> : tensor<32x32x112x112xf32>
    %v37 = stablehlo.constant dense<1.0e-5> : tensor<32x32x112x112xf32>
    %v38 = stablehlo.reduce(%v34 init: %v35) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v39 = stablehlo.broadcast_in_dim %v38, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x112x112xf32>
    %v40 = stablehlo.divide %v39, %v36 : tensor<32x32x112x112xf32>
    %v41 = stablehlo.subtract %v34, %v40 : tensor<32x32x112x112xf32>
    %v42 = stablehlo.multiply %v41, %v41 : tensor<32x32x112x112xf32>
    %v43 = stablehlo.reduce(%v42 init: %v35) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v44 = stablehlo.broadcast_in_dim %v43, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x112x112xf32>
    %v45 = stablehlo.divide %v44, %v36 : tensor<32x32x112x112xf32>
    %v46 = stablehlo.add %v45, %v37 : tensor<32x32x112x112xf32>
    %v47 = stablehlo.rsqrt %v46 : tensor<32x32x112x112xf32>
    %v48 = stablehlo.multiply %v41, %v47 : tensor<32x32x112x112xf32>
    %v49 = stablehlo.broadcast_in_dim %gd1, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v50 = stablehlo.broadcast_in_dim %btd1, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v51 = stablehlo.multiply %v48, %v49 : tensor<32x32x112x112xf32>
    %v52 = stablehlo.add %v51, %v50 : tensor<32x32x112x112xf32>
    %v53 = stablehlo.reshape %v52 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v54 = stablehlo.constant dense<0.0> : tensor<32x401408xf32>
    %v55 = stablehlo.constant dense<6.0> : tensor<32x401408xf32>
    %v56 = stablehlo.maximum %v53, %v54 : tensor<32x401408xf32>
    %v57 = stablehlo.minimum %v56, %v55 : tensor<32x401408xf32>
    %v58 = stablehlo.reshape %v57 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v59 = stablehlo.convolution(%v58, %Wp1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x112x112xf32>, tensor<16x32x1x1xf32>) -> tensor<32x16x112x112xf32>
    %v60 = stablehlo.broadcast_in_dim %zb16, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v61 = stablehlo.add %v59, %v60 : tensor<32x16x112x112xf32>
    %v62 = stablehlo.reshape %v61 : (tensor<32x16x112x112xf32>) -> tensor<32x200704xf32>
    %v63 = stablehlo.reshape %v62 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v64 = stablehlo.constant dense<0.0> : tensor<f32>
    %v65 = stablehlo.constant dense<12544.0> : tensor<32x16x112x112xf32>
    %v66 = stablehlo.constant dense<1.0e-5> : tensor<32x16x112x112xf32>
    %v67 = stablehlo.reduce(%v63 init: %v64) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<32x16xf32>
    %v68 = stablehlo.broadcast_in_dim %v67, dims = [0, 1] : (tensor<32x16xf32>) -> tensor<32x16x112x112xf32>
    %v69 = stablehlo.divide %v68, %v65 : tensor<32x16x112x112xf32>
    %v70 = stablehlo.subtract %v63, %v69 : tensor<32x16x112x112xf32>
    %v71 = stablehlo.multiply %v70, %v70 : tensor<32x16x112x112xf32>
    %v72 = stablehlo.reduce(%v71 init: %v64) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<32x16xf32>
    %v73 = stablehlo.broadcast_in_dim %v72, dims = [0, 1] : (tensor<32x16xf32>) -> tensor<32x16x112x112xf32>
    %v74 = stablehlo.divide %v73, %v65 : tensor<32x16x112x112xf32>
    %v75 = stablehlo.add %v74, %v66 : tensor<32x16x112x112xf32>
    %v76 = stablehlo.rsqrt %v75 : tensor<32x16x112x112xf32>
    %v77 = stablehlo.multiply %v70, %v76 : tensor<32x16x112x112xf32>
    %v78 = stablehlo.broadcast_in_dim %gp1, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v79 = stablehlo.broadcast_in_dim %btp1, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v80 = stablehlo.multiply %v77, %v78 : tensor<32x16x112x112xf32>
    %v81 = stablehlo.add %v80, %v79 : tensor<32x16x112x112xf32>
    %v82 = stablehlo.reshape %v81 : (tensor<32x16x112x112xf32>) -> tensor<32x200704xf32>
    %v83 = stablehlo.reshape %v82 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v84 = stablehlo.convolution(%v83, %We2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x16x112x112xf32>, tensor<96x16x1x1xf32>) -> tensor<32x96x112x112xf32>
    %v85 = stablehlo.broadcast_in_dim %zb96, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v86 = stablehlo.add %v84, %v85 : tensor<32x96x112x112xf32>
    %v87 = stablehlo.reshape %v86 : (tensor<32x96x112x112xf32>) -> tensor<32x1204224xf32>
    %v88 = stablehlo.reshape %v87 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v89 = stablehlo.constant dense<0.0> : tensor<f32>
    %v90 = stablehlo.constant dense<12544.0> : tensor<32x96x112x112xf32>
    %v91 = stablehlo.constant dense<1.0e-5> : tensor<32x96x112x112xf32>
    %v92 = stablehlo.reduce(%v88 init: %v89) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v93 = stablehlo.broadcast_in_dim %v92, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x112x112xf32>
    %v94 = stablehlo.divide %v93, %v90 : tensor<32x96x112x112xf32>
    %v95 = stablehlo.subtract %v88, %v94 : tensor<32x96x112x112xf32>
    %v96 = stablehlo.multiply %v95, %v95 : tensor<32x96x112x112xf32>
    %v97 = stablehlo.reduce(%v96 init: %v89) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v98 = stablehlo.broadcast_in_dim %v97, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x112x112xf32>
    %v99 = stablehlo.divide %v98, %v90 : tensor<32x96x112x112xf32>
    %v100 = stablehlo.add %v99, %v91 : tensor<32x96x112x112xf32>
    %v101 = stablehlo.rsqrt %v100 : tensor<32x96x112x112xf32>
    %v102 = stablehlo.multiply %v95, %v101 : tensor<32x96x112x112xf32>
    %v103 = stablehlo.broadcast_in_dim %ge2, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v104 = stablehlo.broadcast_in_dim %bte2, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v105 = stablehlo.multiply %v102, %v103 : tensor<32x96x112x112xf32>
    %v106 = stablehlo.add %v105, %v104 : tensor<32x96x112x112xf32>
    %v107 = stablehlo.reshape %v106 : (tensor<32x96x112x112xf32>) -> tensor<32x1204224xf32>
    %v108 = stablehlo.constant dense<0.0> : tensor<32x1204224xf32>
    %v109 = stablehlo.constant dense<6.0> : tensor<32x1204224xf32>
    %v110 = stablehlo.maximum %v107, %v108 : tensor<32x1204224xf32>
    %v111 = stablehlo.minimum %v110, %v109 : tensor<32x1204224xf32>
    %v112 = stablehlo.reshape %v111 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v113 = stablehlo.convolution(%v112, %Wd2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x112x112xf32>, tensor<96x1x3x3xf32>) -> tensor<32x96x56x56xf32>
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
    %v132 = stablehlo.broadcast_in_dim %gd2, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v133 = stablehlo.broadcast_in_dim %btd2, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v134 = stablehlo.multiply %v131, %v132 : tensor<32x96x56x56xf32>
    %v135 = stablehlo.add %v134, %v133 : tensor<32x96x56x56xf32>
    %v136 = stablehlo.reshape %v135 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v137 = stablehlo.constant dense<0.0> : tensor<32x301056xf32>
    %v138 = stablehlo.constant dense<6.0> : tensor<32x301056xf32>
    %v139 = stablehlo.maximum %v136, %v137 : tensor<32x301056xf32>
    %v140 = stablehlo.minimum %v139, %v138 : tensor<32x301056xf32>
    %v141 = stablehlo.reshape %v140 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v142 = stablehlo.convolution(%v141, %Wp2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<24x96x1x1xf32>) -> tensor<32x24x56x56xf32>
    %v143 = stablehlo.broadcast_in_dim %zb24, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v144 = stablehlo.add %v142, %v143 : tensor<32x24x56x56xf32>
    %v145 = stablehlo.reshape %v144 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v146 = stablehlo.reshape %v145 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v147 = stablehlo.constant dense<0.0> : tensor<f32>
    %v148 = stablehlo.constant dense<3136.0> : tensor<32x24x56x56xf32>
    %v149 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v150 = stablehlo.reduce(%v146 init: %v147) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v151 = stablehlo.broadcast_in_dim %v150, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v152 = stablehlo.divide %v151, %v148 : tensor<32x24x56x56xf32>
    %v153 = stablehlo.subtract %v146, %v152 : tensor<32x24x56x56xf32>
    %v154 = stablehlo.multiply %v153, %v153 : tensor<32x24x56x56xf32>
    %v155 = stablehlo.reduce(%v154 init: %v147) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v156 = stablehlo.broadcast_in_dim %v155, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v157 = stablehlo.divide %v156, %v148 : tensor<32x24x56x56xf32>
    %v158 = stablehlo.add %v157, %v149 : tensor<32x24x56x56xf32>
    %v159 = stablehlo.rsqrt %v158 : tensor<32x24x56x56xf32>
    %v160 = stablehlo.multiply %v153, %v159 : tensor<32x24x56x56xf32>
    %v161 = stablehlo.broadcast_in_dim %gp2, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v162 = stablehlo.broadcast_in_dim %btp2, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v163 = stablehlo.multiply %v160, %v161 : tensor<32x24x56x56xf32>
    %v164 = stablehlo.add %v163, %v162 : tensor<32x24x56x56xf32>
    %v165 = stablehlo.reshape %v164 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v166 = stablehlo.reshape %v165 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v167 = stablehlo.convolution(%v166, %We3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<144x24x1x1xf32>) -> tensor<32x144x56x56xf32>
    %v168 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v169 = stablehlo.add %v167, %v168 : tensor<32x144x56x56xf32>
    %v170 = stablehlo.reshape %v169 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v171 = stablehlo.reshape %v170 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v172 = stablehlo.constant dense<0.0> : tensor<f32>
    %v173 = stablehlo.constant dense<3136.0> : tensor<32x144x56x56xf32>
    %v174 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v175 = stablehlo.reduce(%v171 init: %v172) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v176 = stablehlo.broadcast_in_dim %v175, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v177 = stablehlo.divide %v176, %v173 : tensor<32x144x56x56xf32>
    %v178 = stablehlo.subtract %v171, %v177 : tensor<32x144x56x56xf32>
    %v179 = stablehlo.multiply %v178, %v178 : tensor<32x144x56x56xf32>
    %v180 = stablehlo.reduce(%v179 init: %v172) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v181 = stablehlo.broadcast_in_dim %v180, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v182 = stablehlo.divide %v181, %v173 : tensor<32x144x56x56xf32>
    %v183 = stablehlo.add %v182, %v174 : tensor<32x144x56x56xf32>
    %v184 = stablehlo.rsqrt %v183 : tensor<32x144x56x56xf32>
    %v185 = stablehlo.multiply %v178, %v184 : tensor<32x144x56x56xf32>
    %v186 = stablehlo.broadcast_in_dim %ge3, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v187 = stablehlo.broadcast_in_dim %bte3, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v188 = stablehlo.multiply %v185, %v186 : tensor<32x144x56x56xf32>
    %v189 = stablehlo.add %v188, %v187 : tensor<32x144x56x56xf32>
    %v190 = stablehlo.reshape %v189 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v191 = stablehlo.constant dense<0.0> : tensor<32x451584xf32>
    %v192 = stablehlo.constant dense<6.0> : tensor<32x451584xf32>
    %v193 = stablehlo.maximum %v190, %v191 : tensor<32x451584xf32>
    %v194 = stablehlo.minimum %v193, %v192 : tensor<32x451584xf32>
    %v195 = stablehlo.reshape %v194 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v196 = stablehlo.convolution(%v195, %Wd3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 144 : i64} : (tensor<32x144x56x56xf32>, tensor<144x1x3x3xf32>) -> tensor<32x144x56x56xf32>
    %v197 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v198 = stablehlo.add %v196, %v197 : tensor<32x144x56x56xf32>
    %v199 = stablehlo.reshape %v198 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v200 = stablehlo.reshape %v199 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v201 = stablehlo.constant dense<0.0> : tensor<f32>
    %v202 = stablehlo.constant dense<3136.0> : tensor<32x144x56x56xf32>
    %v203 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v204 = stablehlo.reduce(%v200 init: %v201) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v205 = stablehlo.broadcast_in_dim %v204, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v206 = stablehlo.divide %v205, %v202 : tensor<32x144x56x56xf32>
    %v207 = stablehlo.subtract %v200, %v206 : tensor<32x144x56x56xf32>
    %v208 = stablehlo.multiply %v207, %v207 : tensor<32x144x56x56xf32>
    %v209 = stablehlo.reduce(%v208 init: %v201) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v210 = stablehlo.broadcast_in_dim %v209, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v211 = stablehlo.divide %v210, %v202 : tensor<32x144x56x56xf32>
    %v212 = stablehlo.add %v211, %v203 : tensor<32x144x56x56xf32>
    %v213 = stablehlo.rsqrt %v212 : tensor<32x144x56x56xf32>
    %v214 = stablehlo.multiply %v207, %v213 : tensor<32x144x56x56xf32>
    %v215 = stablehlo.broadcast_in_dim %gd3, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v216 = stablehlo.broadcast_in_dim %btd3, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v217 = stablehlo.multiply %v214, %v215 : tensor<32x144x56x56xf32>
    %v218 = stablehlo.add %v217, %v216 : tensor<32x144x56x56xf32>
    %v219 = stablehlo.reshape %v218 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v220 = stablehlo.constant dense<0.0> : tensor<32x451584xf32>
    %v221 = stablehlo.constant dense<6.0> : tensor<32x451584xf32>
    %v222 = stablehlo.maximum %v219, %v220 : tensor<32x451584xf32>
    %v223 = stablehlo.minimum %v222, %v221 : tensor<32x451584xf32>
    %v224 = stablehlo.reshape %v223 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v225 = stablehlo.convolution(%v224, %Wp3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x144x56x56xf32>, tensor<24x144x1x1xf32>) -> tensor<32x24x56x56xf32>
    %v226 = stablehlo.broadcast_in_dim %zb24, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v227 = stablehlo.add %v225, %v226 : tensor<32x24x56x56xf32>
    %v228 = stablehlo.reshape %v227 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v229 = stablehlo.reshape %v228 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v230 = stablehlo.constant dense<0.0> : tensor<f32>
    %v231 = stablehlo.constant dense<3136.0> : tensor<32x24x56x56xf32>
    %v232 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v233 = stablehlo.reduce(%v229 init: %v230) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v234 = stablehlo.broadcast_in_dim %v233, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v235 = stablehlo.divide %v234, %v231 : tensor<32x24x56x56xf32>
    %v236 = stablehlo.subtract %v229, %v235 : tensor<32x24x56x56xf32>
    %v237 = stablehlo.multiply %v236, %v236 : tensor<32x24x56x56xf32>
    %v238 = stablehlo.reduce(%v237 init: %v230) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v239 = stablehlo.broadcast_in_dim %v238, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v240 = stablehlo.divide %v239, %v231 : tensor<32x24x56x56xf32>
    %v241 = stablehlo.add %v240, %v232 : tensor<32x24x56x56xf32>
    %v242 = stablehlo.rsqrt %v241 : tensor<32x24x56x56xf32>
    %v243 = stablehlo.multiply %v236, %v242 : tensor<32x24x56x56xf32>
    %v244 = stablehlo.broadcast_in_dim %gp3, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v245 = stablehlo.broadcast_in_dim %btp3, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v246 = stablehlo.multiply %v243, %v244 : tensor<32x24x56x56xf32>
    %v247 = stablehlo.add %v246, %v245 : tensor<32x24x56x56xf32>
    %v248 = stablehlo.reshape %v247 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v249 = stablehlo.add %v248, %v165 : tensor<32x75264xf32>
    %v250 = stablehlo.reshape %v249 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v251 = stablehlo.convolution(%v250, %We4)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<144x24x1x1xf32>) -> tensor<32x144x56x56xf32>
    %v252 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v253 = stablehlo.add %v251, %v252 : tensor<32x144x56x56xf32>
    %v254 = stablehlo.reshape %v253 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v255 = stablehlo.reshape %v254 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v256 = stablehlo.constant dense<0.0> : tensor<f32>
    %v257 = stablehlo.constant dense<3136.0> : tensor<32x144x56x56xf32>
    %v258 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v259 = stablehlo.reduce(%v255 init: %v256) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v260 = stablehlo.broadcast_in_dim %v259, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v261 = stablehlo.divide %v260, %v257 : tensor<32x144x56x56xf32>
    %v262 = stablehlo.subtract %v255, %v261 : tensor<32x144x56x56xf32>
    %v263 = stablehlo.multiply %v262, %v262 : tensor<32x144x56x56xf32>
    %v264 = stablehlo.reduce(%v263 init: %v256) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v265 = stablehlo.broadcast_in_dim %v264, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v266 = stablehlo.divide %v265, %v257 : tensor<32x144x56x56xf32>
    %v267 = stablehlo.add %v266, %v258 : tensor<32x144x56x56xf32>
    %v268 = stablehlo.rsqrt %v267 : tensor<32x144x56x56xf32>
    %v269 = stablehlo.multiply %v262, %v268 : tensor<32x144x56x56xf32>
    %v270 = stablehlo.broadcast_in_dim %ge4, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v271 = stablehlo.broadcast_in_dim %bte4, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v272 = stablehlo.multiply %v269, %v270 : tensor<32x144x56x56xf32>
    %v273 = stablehlo.add %v272, %v271 : tensor<32x144x56x56xf32>
    %v274 = stablehlo.reshape %v273 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v275 = stablehlo.constant dense<0.0> : tensor<32x451584xf32>
    %v276 = stablehlo.constant dense<6.0> : tensor<32x451584xf32>
    %v277 = stablehlo.maximum %v274, %v275 : tensor<32x451584xf32>
    %v278 = stablehlo.minimum %v277, %v276 : tensor<32x451584xf32>
    %v279 = stablehlo.reshape %v278 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v280 = stablehlo.convolution(%v279, %Wd4)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 144 : i64} : (tensor<32x144x56x56xf32>, tensor<144x1x3x3xf32>) -> tensor<32x144x28x28xf32>
    %v281 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v282 = stablehlo.add %v280, %v281 : tensor<32x144x28x28xf32>
    %v283 = stablehlo.reshape %v282 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v284 = stablehlo.reshape %v283 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v285 = stablehlo.constant dense<0.0> : tensor<f32>
    %v286 = stablehlo.constant dense<784.0> : tensor<32x144x28x28xf32>
    %v287 = stablehlo.constant dense<1.0e-5> : tensor<32x144x28x28xf32>
    %v288 = stablehlo.reduce(%v284 init: %v285) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v289 = stablehlo.broadcast_in_dim %v288, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x28x28xf32>
    %v290 = stablehlo.divide %v289, %v286 : tensor<32x144x28x28xf32>
    %v291 = stablehlo.subtract %v284, %v290 : tensor<32x144x28x28xf32>
    %v292 = stablehlo.multiply %v291, %v291 : tensor<32x144x28x28xf32>
    %v293 = stablehlo.reduce(%v292 init: %v285) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v294 = stablehlo.broadcast_in_dim %v293, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x28x28xf32>
    %v295 = stablehlo.divide %v294, %v286 : tensor<32x144x28x28xf32>
    %v296 = stablehlo.add %v295, %v287 : tensor<32x144x28x28xf32>
    %v297 = stablehlo.rsqrt %v296 : tensor<32x144x28x28xf32>
    %v298 = stablehlo.multiply %v291, %v297 : tensor<32x144x28x28xf32>
    %v299 = stablehlo.broadcast_in_dim %gd4, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v300 = stablehlo.broadcast_in_dim %btd4, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v301 = stablehlo.multiply %v298, %v299 : tensor<32x144x28x28xf32>
    %v302 = stablehlo.add %v301, %v300 : tensor<32x144x28x28xf32>
    %v303 = stablehlo.reshape %v302 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v304 = stablehlo.constant dense<0.0> : tensor<32x112896xf32>
    %v305 = stablehlo.constant dense<6.0> : tensor<32x112896xf32>
    %v306 = stablehlo.maximum %v303, %v304 : tensor<32x112896xf32>
    %v307 = stablehlo.minimum %v306, %v305 : tensor<32x112896xf32>
    %v308 = stablehlo.reshape %v307 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v309 = stablehlo.convolution(%v308, %Wp4)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x144x28x28xf32>, tensor<32x144x1x1xf32>) -> tensor<32x32x28x28xf32>
    %v310 = stablehlo.broadcast_in_dim %zb32, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v311 = stablehlo.add %v309, %v310 : tensor<32x32x28x28xf32>
    %v312 = stablehlo.reshape %v311 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v313 = stablehlo.reshape %v312 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v314 = stablehlo.constant dense<0.0> : tensor<f32>
    %v315 = stablehlo.constant dense<784.0> : tensor<32x32x28x28xf32>
    %v316 = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %v317 = stablehlo.reduce(%v313 init: %v314) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v318 = stablehlo.broadcast_in_dim %v317, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v319 = stablehlo.divide %v318, %v315 : tensor<32x32x28x28xf32>
    %v320 = stablehlo.subtract %v313, %v319 : tensor<32x32x28x28xf32>
    %v321 = stablehlo.multiply %v320, %v320 : tensor<32x32x28x28xf32>
    %v322 = stablehlo.reduce(%v321 init: %v314) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v323 = stablehlo.broadcast_in_dim %v322, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v324 = stablehlo.divide %v323, %v315 : tensor<32x32x28x28xf32>
    %v325 = stablehlo.add %v324, %v316 : tensor<32x32x28x28xf32>
    %v326 = stablehlo.rsqrt %v325 : tensor<32x32x28x28xf32>
    %v327 = stablehlo.multiply %v320, %v326 : tensor<32x32x28x28xf32>
    %v328 = stablehlo.broadcast_in_dim %gp4, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v329 = stablehlo.broadcast_in_dim %btp4, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v330 = stablehlo.multiply %v327, %v328 : tensor<32x32x28x28xf32>
    %v331 = stablehlo.add %v330, %v329 : tensor<32x32x28x28xf32>
    %v332 = stablehlo.reshape %v331 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v333 = stablehlo.reshape %v332 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v334 = stablehlo.convolution(%v333, %We5)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<192x32x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v335 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v336 = stablehlo.add %v334, %v335 : tensor<32x192x28x28xf32>
    %v337 = stablehlo.reshape %v336 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v338 = stablehlo.reshape %v337 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v339 = stablehlo.constant dense<0.0> : tensor<f32>
    %v340 = stablehlo.constant dense<784.0> : tensor<32x192x28x28xf32>
    %v341 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v342 = stablehlo.reduce(%v338 init: %v339) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v343 = stablehlo.broadcast_in_dim %v342, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v344 = stablehlo.divide %v343, %v340 : tensor<32x192x28x28xf32>
    %v345 = stablehlo.subtract %v338, %v344 : tensor<32x192x28x28xf32>
    %v346 = stablehlo.multiply %v345, %v345 : tensor<32x192x28x28xf32>
    %v347 = stablehlo.reduce(%v346 init: %v339) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v348 = stablehlo.broadcast_in_dim %v347, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v349 = stablehlo.divide %v348, %v340 : tensor<32x192x28x28xf32>
    %v350 = stablehlo.add %v349, %v341 : tensor<32x192x28x28xf32>
    %v351 = stablehlo.rsqrt %v350 : tensor<32x192x28x28xf32>
    %v352 = stablehlo.multiply %v345, %v351 : tensor<32x192x28x28xf32>
    %v353 = stablehlo.broadcast_in_dim %ge5, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v354 = stablehlo.broadcast_in_dim %bte5, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v355 = stablehlo.multiply %v352, %v353 : tensor<32x192x28x28xf32>
    %v356 = stablehlo.add %v355, %v354 : tensor<32x192x28x28xf32>
    %v357 = stablehlo.reshape %v356 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v358 = stablehlo.constant dense<0.0> : tensor<32x150528xf32>
    %v359 = stablehlo.constant dense<6.0> : tensor<32x150528xf32>
    %v360 = stablehlo.maximum %v357, %v358 : tensor<32x150528xf32>
    %v361 = stablehlo.minimum %v360, %v359 : tensor<32x150528xf32>
    %v362 = stablehlo.reshape %v361 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v363 = stablehlo.convolution(%v362, %Wd5)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x3x3xf32>) -> tensor<32x192x28x28xf32>
    %v364 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v365 = stablehlo.add %v363, %v364 : tensor<32x192x28x28xf32>
    %v366 = stablehlo.reshape %v365 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v367 = stablehlo.reshape %v366 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v368 = stablehlo.constant dense<0.0> : tensor<f32>
    %v369 = stablehlo.constant dense<784.0> : tensor<32x192x28x28xf32>
    %v370 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v371 = stablehlo.reduce(%v367 init: %v368) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v372 = stablehlo.broadcast_in_dim %v371, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v373 = stablehlo.divide %v372, %v369 : tensor<32x192x28x28xf32>
    %v374 = stablehlo.subtract %v367, %v373 : tensor<32x192x28x28xf32>
    %v375 = stablehlo.multiply %v374, %v374 : tensor<32x192x28x28xf32>
    %v376 = stablehlo.reduce(%v375 init: %v368) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v377 = stablehlo.broadcast_in_dim %v376, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v378 = stablehlo.divide %v377, %v369 : tensor<32x192x28x28xf32>
    %v379 = stablehlo.add %v378, %v370 : tensor<32x192x28x28xf32>
    %v380 = stablehlo.rsqrt %v379 : tensor<32x192x28x28xf32>
    %v381 = stablehlo.multiply %v374, %v380 : tensor<32x192x28x28xf32>
    %v382 = stablehlo.broadcast_in_dim %gd5, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v383 = stablehlo.broadcast_in_dim %btd5, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v384 = stablehlo.multiply %v381, %v382 : tensor<32x192x28x28xf32>
    %v385 = stablehlo.add %v384, %v383 : tensor<32x192x28x28xf32>
    %v386 = stablehlo.reshape %v385 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v387 = stablehlo.constant dense<0.0> : tensor<32x150528xf32>
    %v388 = stablehlo.constant dense<6.0> : tensor<32x150528xf32>
    %v389 = stablehlo.maximum %v386, %v387 : tensor<32x150528xf32>
    %v390 = stablehlo.minimum %v389, %v388 : tensor<32x150528xf32>
    %v391 = stablehlo.reshape %v390 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v392 = stablehlo.convolution(%v391, %Wp5)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<32x192x1x1xf32>) -> tensor<32x32x28x28xf32>
    %v393 = stablehlo.broadcast_in_dim %zb32, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v394 = stablehlo.add %v392, %v393 : tensor<32x32x28x28xf32>
    %v395 = stablehlo.reshape %v394 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v396 = stablehlo.reshape %v395 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v397 = stablehlo.constant dense<0.0> : tensor<f32>
    %v398 = stablehlo.constant dense<784.0> : tensor<32x32x28x28xf32>
    %v399 = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %v400 = stablehlo.reduce(%v396 init: %v397) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v401 = stablehlo.broadcast_in_dim %v400, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v402 = stablehlo.divide %v401, %v398 : tensor<32x32x28x28xf32>
    %v403 = stablehlo.subtract %v396, %v402 : tensor<32x32x28x28xf32>
    %v404 = stablehlo.multiply %v403, %v403 : tensor<32x32x28x28xf32>
    %v405 = stablehlo.reduce(%v404 init: %v397) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v406 = stablehlo.broadcast_in_dim %v405, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v407 = stablehlo.divide %v406, %v398 : tensor<32x32x28x28xf32>
    %v408 = stablehlo.add %v407, %v399 : tensor<32x32x28x28xf32>
    %v409 = stablehlo.rsqrt %v408 : tensor<32x32x28x28xf32>
    %v410 = stablehlo.multiply %v403, %v409 : tensor<32x32x28x28xf32>
    %v411 = stablehlo.broadcast_in_dim %gp5, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v412 = stablehlo.broadcast_in_dim %btp5, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v413 = stablehlo.multiply %v410, %v411 : tensor<32x32x28x28xf32>
    %v414 = stablehlo.add %v413, %v412 : tensor<32x32x28x28xf32>
    %v415 = stablehlo.reshape %v414 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v416 = stablehlo.add %v415, %v332 : tensor<32x25088xf32>
    %v417 = stablehlo.reshape %v416 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v418 = stablehlo.convolution(%v417, %We6)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<192x32x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v419 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v420 = stablehlo.add %v418, %v419 : tensor<32x192x28x28xf32>
    %v421 = stablehlo.reshape %v420 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v422 = stablehlo.reshape %v421 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v423 = stablehlo.constant dense<0.0> : tensor<f32>
    %v424 = stablehlo.constant dense<784.0> : tensor<32x192x28x28xf32>
    %v425 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v426 = stablehlo.reduce(%v422 init: %v423) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v427 = stablehlo.broadcast_in_dim %v426, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v428 = stablehlo.divide %v427, %v424 : tensor<32x192x28x28xf32>
    %v429 = stablehlo.subtract %v422, %v428 : tensor<32x192x28x28xf32>
    %v430 = stablehlo.multiply %v429, %v429 : tensor<32x192x28x28xf32>
    %v431 = stablehlo.reduce(%v430 init: %v423) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v432 = stablehlo.broadcast_in_dim %v431, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v433 = stablehlo.divide %v432, %v424 : tensor<32x192x28x28xf32>
    %v434 = stablehlo.add %v433, %v425 : tensor<32x192x28x28xf32>
    %v435 = stablehlo.rsqrt %v434 : tensor<32x192x28x28xf32>
    %v436 = stablehlo.multiply %v429, %v435 : tensor<32x192x28x28xf32>
    %v437 = stablehlo.broadcast_in_dim %ge6, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v438 = stablehlo.broadcast_in_dim %bte6, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v439 = stablehlo.multiply %v436, %v437 : tensor<32x192x28x28xf32>
    %v440 = stablehlo.add %v439, %v438 : tensor<32x192x28x28xf32>
    %v441 = stablehlo.reshape %v440 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v442 = stablehlo.constant dense<0.0> : tensor<32x150528xf32>
    %v443 = stablehlo.constant dense<6.0> : tensor<32x150528xf32>
    %v444 = stablehlo.maximum %v441, %v442 : tensor<32x150528xf32>
    %v445 = stablehlo.minimum %v444, %v443 : tensor<32x150528xf32>
    %v446 = stablehlo.reshape %v445 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v447 = stablehlo.convolution(%v446, %Wd6)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x3x3xf32>) -> tensor<32x192x28x28xf32>
    %v448 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v449 = stablehlo.add %v447, %v448 : tensor<32x192x28x28xf32>
    %v450 = stablehlo.reshape %v449 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v451 = stablehlo.reshape %v450 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v452 = stablehlo.constant dense<0.0> : tensor<f32>
    %v453 = stablehlo.constant dense<784.0> : tensor<32x192x28x28xf32>
    %v454 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v455 = stablehlo.reduce(%v451 init: %v452) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v456 = stablehlo.broadcast_in_dim %v455, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v457 = stablehlo.divide %v456, %v453 : tensor<32x192x28x28xf32>
    %v458 = stablehlo.subtract %v451, %v457 : tensor<32x192x28x28xf32>
    %v459 = stablehlo.multiply %v458, %v458 : tensor<32x192x28x28xf32>
    %v460 = stablehlo.reduce(%v459 init: %v452) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v461 = stablehlo.broadcast_in_dim %v460, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v462 = stablehlo.divide %v461, %v453 : tensor<32x192x28x28xf32>
    %v463 = stablehlo.add %v462, %v454 : tensor<32x192x28x28xf32>
    %v464 = stablehlo.rsqrt %v463 : tensor<32x192x28x28xf32>
    %v465 = stablehlo.multiply %v458, %v464 : tensor<32x192x28x28xf32>
    %v466 = stablehlo.broadcast_in_dim %gd6, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v467 = stablehlo.broadcast_in_dim %btd6, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v468 = stablehlo.multiply %v465, %v466 : tensor<32x192x28x28xf32>
    %v469 = stablehlo.add %v468, %v467 : tensor<32x192x28x28xf32>
    %v470 = stablehlo.reshape %v469 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v471 = stablehlo.constant dense<0.0> : tensor<32x150528xf32>
    %v472 = stablehlo.constant dense<6.0> : tensor<32x150528xf32>
    %v473 = stablehlo.maximum %v470, %v471 : tensor<32x150528xf32>
    %v474 = stablehlo.minimum %v473, %v472 : tensor<32x150528xf32>
    %v475 = stablehlo.reshape %v474 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v476 = stablehlo.convolution(%v475, %Wp6)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<32x192x1x1xf32>) -> tensor<32x32x28x28xf32>
    %v477 = stablehlo.broadcast_in_dim %zb32, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v478 = stablehlo.add %v476, %v477 : tensor<32x32x28x28xf32>
    %v479 = stablehlo.reshape %v478 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v480 = stablehlo.reshape %v479 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v481 = stablehlo.constant dense<0.0> : tensor<f32>
    %v482 = stablehlo.constant dense<784.0> : tensor<32x32x28x28xf32>
    %v483 = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %v484 = stablehlo.reduce(%v480 init: %v481) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v485 = stablehlo.broadcast_in_dim %v484, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v486 = stablehlo.divide %v485, %v482 : tensor<32x32x28x28xf32>
    %v487 = stablehlo.subtract %v480, %v486 : tensor<32x32x28x28xf32>
    %v488 = stablehlo.multiply %v487, %v487 : tensor<32x32x28x28xf32>
    %v489 = stablehlo.reduce(%v488 init: %v481) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v490 = stablehlo.broadcast_in_dim %v489, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v491 = stablehlo.divide %v490, %v482 : tensor<32x32x28x28xf32>
    %v492 = stablehlo.add %v491, %v483 : tensor<32x32x28x28xf32>
    %v493 = stablehlo.rsqrt %v492 : tensor<32x32x28x28xf32>
    %v494 = stablehlo.multiply %v487, %v493 : tensor<32x32x28x28xf32>
    %v495 = stablehlo.broadcast_in_dim %gp6, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v496 = stablehlo.broadcast_in_dim %btp6, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v497 = stablehlo.multiply %v494, %v495 : tensor<32x32x28x28xf32>
    %v498 = stablehlo.add %v497, %v496 : tensor<32x32x28x28xf32>
    %v499 = stablehlo.reshape %v498 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v500 = stablehlo.add %v499, %v416 : tensor<32x25088xf32>
    %v501 = stablehlo.reshape %v500 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v502 = stablehlo.convolution(%v501, %We7)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<192x32x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v503 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v504 = stablehlo.add %v502, %v503 : tensor<32x192x28x28xf32>
    %v505 = stablehlo.reshape %v504 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v506 = stablehlo.reshape %v505 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v507 = stablehlo.constant dense<0.0> : tensor<f32>
    %v508 = stablehlo.constant dense<784.0> : tensor<32x192x28x28xf32>
    %v509 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v510 = stablehlo.reduce(%v506 init: %v507) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v511 = stablehlo.broadcast_in_dim %v510, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v512 = stablehlo.divide %v511, %v508 : tensor<32x192x28x28xf32>
    %v513 = stablehlo.subtract %v506, %v512 : tensor<32x192x28x28xf32>
    %v514 = stablehlo.multiply %v513, %v513 : tensor<32x192x28x28xf32>
    %v515 = stablehlo.reduce(%v514 init: %v507) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v516 = stablehlo.broadcast_in_dim %v515, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v517 = stablehlo.divide %v516, %v508 : tensor<32x192x28x28xf32>
    %v518 = stablehlo.add %v517, %v509 : tensor<32x192x28x28xf32>
    %v519 = stablehlo.rsqrt %v518 : tensor<32x192x28x28xf32>
    %v520 = stablehlo.multiply %v513, %v519 : tensor<32x192x28x28xf32>
    %v521 = stablehlo.broadcast_in_dim %ge7, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v522 = stablehlo.broadcast_in_dim %bte7, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v523 = stablehlo.multiply %v520, %v521 : tensor<32x192x28x28xf32>
    %v524 = stablehlo.add %v523, %v522 : tensor<32x192x28x28xf32>
    %v525 = stablehlo.reshape %v524 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v526 = stablehlo.constant dense<0.0> : tensor<32x150528xf32>
    %v527 = stablehlo.constant dense<6.0> : tensor<32x150528xf32>
    %v528 = stablehlo.maximum %v525, %v526 : tensor<32x150528xf32>
    %v529 = stablehlo.minimum %v528, %v527 : tensor<32x150528xf32>
    %v530 = stablehlo.reshape %v529 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v531 = stablehlo.convolution(%v530, %Wd7)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x3x3xf32>) -> tensor<32x192x14x14xf32>
    %v532 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x14x14xf32>
    %v533 = stablehlo.add %v531, %v532 : tensor<32x192x14x14xf32>
    %v534 = stablehlo.reshape %v533 : (tensor<32x192x14x14xf32>) -> tensor<32x37632xf32>
    %v535 = stablehlo.reshape %v534 : (tensor<32x37632xf32>) -> tensor<32x192x14x14xf32>
    %v536 = stablehlo.constant dense<0.0> : tensor<f32>
    %v537 = stablehlo.constant dense<196.0> : tensor<32x192x14x14xf32>
    %v538 = stablehlo.constant dense<1.0e-5> : tensor<32x192x14x14xf32>
    %v539 = stablehlo.reduce(%v535 init: %v536) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v540 = stablehlo.broadcast_in_dim %v539, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x14x14xf32>
    %v541 = stablehlo.divide %v540, %v537 : tensor<32x192x14x14xf32>
    %v542 = stablehlo.subtract %v535, %v541 : tensor<32x192x14x14xf32>
    %v543 = stablehlo.multiply %v542, %v542 : tensor<32x192x14x14xf32>
    %v544 = stablehlo.reduce(%v543 init: %v536) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v545 = stablehlo.broadcast_in_dim %v544, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x14x14xf32>
    %v546 = stablehlo.divide %v545, %v537 : tensor<32x192x14x14xf32>
    %v547 = stablehlo.add %v546, %v538 : tensor<32x192x14x14xf32>
    %v548 = stablehlo.rsqrt %v547 : tensor<32x192x14x14xf32>
    %v549 = stablehlo.multiply %v542, %v548 : tensor<32x192x14x14xf32>
    %v550 = stablehlo.broadcast_in_dim %gd7, dims = [1] : (tensor<192xf32>) -> tensor<32x192x14x14xf32>
    %v551 = stablehlo.broadcast_in_dim %btd7, dims = [1] : (tensor<192xf32>) -> tensor<32x192x14x14xf32>
    %v552 = stablehlo.multiply %v549, %v550 : tensor<32x192x14x14xf32>
    %v553 = stablehlo.add %v552, %v551 : tensor<32x192x14x14xf32>
    %v554 = stablehlo.reshape %v553 : (tensor<32x192x14x14xf32>) -> tensor<32x37632xf32>
    %v555 = stablehlo.constant dense<0.0> : tensor<32x37632xf32>
    %v556 = stablehlo.constant dense<6.0> : tensor<32x37632xf32>
    %v557 = stablehlo.maximum %v554, %v555 : tensor<32x37632xf32>
    %v558 = stablehlo.minimum %v557, %v556 : tensor<32x37632xf32>
    %v559 = stablehlo.reshape %v558 : (tensor<32x37632xf32>) -> tensor<32x192x14x14xf32>
    %v560 = stablehlo.convolution(%v559, %Wp7)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x14x14xf32>, tensor<64x192x1x1xf32>) -> tensor<32x64x14x14xf32>
    %v561 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v562 = stablehlo.add %v560, %v561 : tensor<32x64x14x14xf32>
    %v563 = stablehlo.reshape %v562 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v564 = stablehlo.reshape %v563 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v565 = stablehlo.constant dense<0.0> : tensor<f32>
    %v566 = stablehlo.constant dense<196.0> : tensor<32x64x14x14xf32>
    %v567 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v568 = stablehlo.reduce(%v564 init: %v565) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v569 = stablehlo.broadcast_in_dim %v568, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v570 = stablehlo.divide %v569, %v566 : tensor<32x64x14x14xf32>
    %v571 = stablehlo.subtract %v564, %v570 : tensor<32x64x14x14xf32>
    %v572 = stablehlo.multiply %v571, %v571 : tensor<32x64x14x14xf32>
    %v573 = stablehlo.reduce(%v572 init: %v565) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v574 = stablehlo.broadcast_in_dim %v573, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v575 = stablehlo.divide %v574, %v566 : tensor<32x64x14x14xf32>
    %v576 = stablehlo.add %v575, %v567 : tensor<32x64x14x14xf32>
    %v577 = stablehlo.rsqrt %v576 : tensor<32x64x14x14xf32>
    %v578 = stablehlo.multiply %v571, %v577 : tensor<32x64x14x14xf32>
    %v579 = stablehlo.broadcast_in_dim %gp7, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v580 = stablehlo.broadcast_in_dim %btp7, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v581 = stablehlo.multiply %v578, %v579 : tensor<32x64x14x14xf32>
    %v582 = stablehlo.add %v581, %v580 : tensor<32x64x14x14xf32>
    %v583 = stablehlo.reshape %v582 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v584 = stablehlo.reshape %v583 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v585 = stablehlo.convolution(%v584, %We8)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x14x14xf32>, tensor<384x64x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v586 = stablehlo.broadcast_in_dim %zb384, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v587 = stablehlo.add %v585, %v586 : tensor<32x384x14x14xf32>
    %v588 = stablehlo.reshape %v587 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v589 = stablehlo.reshape %v588 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v590 = stablehlo.constant dense<0.0> : tensor<f32>
    %v591 = stablehlo.constant dense<196.0> : tensor<32x384x14x14xf32>
    %v592 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v593 = stablehlo.reduce(%v589 init: %v590) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v594 = stablehlo.broadcast_in_dim %v593, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v595 = stablehlo.divide %v594, %v591 : tensor<32x384x14x14xf32>
    %v596 = stablehlo.subtract %v589, %v595 : tensor<32x384x14x14xf32>
    %v597 = stablehlo.multiply %v596, %v596 : tensor<32x384x14x14xf32>
    %v598 = stablehlo.reduce(%v597 init: %v590) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v599 = stablehlo.broadcast_in_dim %v598, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v600 = stablehlo.divide %v599, %v591 : tensor<32x384x14x14xf32>
    %v601 = stablehlo.add %v600, %v592 : tensor<32x384x14x14xf32>
    %v602 = stablehlo.rsqrt %v601 : tensor<32x384x14x14xf32>
    %v603 = stablehlo.multiply %v596, %v602 : tensor<32x384x14x14xf32>
    %v604 = stablehlo.broadcast_in_dim %ge8, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v605 = stablehlo.broadcast_in_dim %bte8, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v606 = stablehlo.multiply %v603, %v604 : tensor<32x384x14x14xf32>
    %v607 = stablehlo.add %v606, %v605 : tensor<32x384x14x14xf32>
    %v608 = stablehlo.reshape %v607 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v609 = stablehlo.constant dense<0.0> : tensor<32x75264xf32>
    %v610 = stablehlo.constant dense<6.0> : tensor<32x75264xf32>
    %v611 = stablehlo.maximum %v608, %v609 : tensor<32x75264xf32>
    %v612 = stablehlo.minimum %v611, %v610 : tensor<32x75264xf32>
    %v613 = stablehlo.reshape %v612 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v614 = stablehlo.convolution(%v613, %Wd8)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x3x3xf32>) -> tensor<32x384x14x14xf32>
    %v615 = stablehlo.broadcast_in_dim %zb384, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v616 = stablehlo.add %v614, %v615 : tensor<32x384x14x14xf32>
    %v617 = stablehlo.reshape %v616 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v618 = stablehlo.reshape %v617 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v619 = stablehlo.constant dense<0.0> : tensor<f32>
    %v620 = stablehlo.constant dense<196.0> : tensor<32x384x14x14xf32>
    %v621 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v622 = stablehlo.reduce(%v618 init: %v619) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v623 = stablehlo.broadcast_in_dim %v622, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v624 = stablehlo.divide %v623, %v620 : tensor<32x384x14x14xf32>
    %v625 = stablehlo.subtract %v618, %v624 : tensor<32x384x14x14xf32>
    %v626 = stablehlo.multiply %v625, %v625 : tensor<32x384x14x14xf32>
    %v627 = stablehlo.reduce(%v626 init: %v619) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v628 = stablehlo.broadcast_in_dim %v627, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v629 = stablehlo.divide %v628, %v620 : tensor<32x384x14x14xf32>
    %v630 = stablehlo.add %v629, %v621 : tensor<32x384x14x14xf32>
    %v631 = stablehlo.rsqrt %v630 : tensor<32x384x14x14xf32>
    %v632 = stablehlo.multiply %v625, %v631 : tensor<32x384x14x14xf32>
    %v633 = stablehlo.broadcast_in_dim %gd8, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v634 = stablehlo.broadcast_in_dim %btd8, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v635 = stablehlo.multiply %v632, %v633 : tensor<32x384x14x14xf32>
    %v636 = stablehlo.add %v635, %v634 : tensor<32x384x14x14xf32>
    %v637 = stablehlo.reshape %v636 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v638 = stablehlo.constant dense<0.0> : tensor<32x75264xf32>
    %v639 = stablehlo.constant dense<6.0> : tensor<32x75264xf32>
    %v640 = stablehlo.maximum %v637, %v638 : tensor<32x75264xf32>
    %v641 = stablehlo.minimum %v640, %v639 : tensor<32x75264xf32>
    %v642 = stablehlo.reshape %v641 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v643 = stablehlo.convolution(%v642, %Wp8)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<64x384x1x1xf32>) -> tensor<32x64x14x14xf32>
    %v644 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v645 = stablehlo.add %v643, %v644 : tensor<32x64x14x14xf32>
    %v646 = stablehlo.reshape %v645 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v647 = stablehlo.reshape %v646 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v648 = stablehlo.constant dense<0.0> : tensor<f32>
    %v649 = stablehlo.constant dense<196.0> : tensor<32x64x14x14xf32>
    %v650 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v651 = stablehlo.reduce(%v647 init: %v648) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v652 = stablehlo.broadcast_in_dim %v651, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v653 = stablehlo.divide %v652, %v649 : tensor<32x64x14x14xf32>
    %v654 = stablehlo.subtract %v647, %v653 : tensor<32x64x14x14xf32>
    %v655 = stablehlo.multiply %v654, %v654 : tensor<32x64x14x14xf32>
    %v656 = stablehlo.reduce(%v655 init: %v648) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v657 = stablehlo.broadcast_in_dim %v656, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v658 = stablehlo.divide %v657, %v649 : tensor<32x64x14x14xf32>
    %v659 = stablehlo.add %v658, %v650 : tensor<32x64x14x14xf32>
    %v660 = stablehlo.rsqrt %v659 : tensor<32x64x14x14xf32>
    %v661 = stablehlo.multiply %v654, %v660 : tensor<32x64x14x14xf32>
    %v662 = stablehlo.broadcast_in_dim %gp8, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v663 = stablehlo.broadcast_in_dim %btp8, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v664 = stablehlo.multiply %v661, %v662 : tensor<32x64x14x14xf32>
    %v665 = stablehlo.add %v664, %v663 : tensor<32x64x14x14xf32>
    %v666 = stablehlo.reshape %v665 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v667 = stablehlo.add %v666, %v583 : tensor<32x12544xf32>
    %v668 = stablehlo.reshape %v667 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v669 = stablehlo.convolution(%v668, %We9)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x14x14xf32>, tensor<384x64x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v670 = stablehlo.broadcast_in_dim %zb384, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v671 = stablehlo.add %v669, %v670 : tensor<32x384x14x14xf32>
    %v672 = stablehlo.reshape %v671 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v673 = stablehlo.reshape %v672 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v674 = stablehlo.constant dense<0.0> : tensor<f32>
    %v675 = stablehlo.constant dense<196.0> : tensor<32x384x14x14xf32>
    %v676 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v677 = stablehlo.reduce(%v673 init: %v674) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v678 = stablehlo.broadcast_in_dim %v677, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v679 = stablehlo.divide %v678, %v675 : tensor<32x384x14x14xf32>
    %v680 = stablehlo.subtract %v673, %v679 : tensor<32x384x14x14xf32>
    %v681 = stablehlo.multiply %v680, %v680 : tensor<32x384x14x14xf32>
    %v682 = stablehlo.reduce(%v681 init: %v674) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v683 = stablehlo.broadcast_in_dim %v682, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v684 = stablehlo.divide %v683, %v675 : tensor<32x384x14x14xf32>
    %v685 = stablehlo.add %v684, %v676 : tensor<32x384x14x14xf32>
    %v686 = stablehlo.rsqrt %v685 : tensor<32x384x14x14xf32>
    %v687 = stablehlo.multiply %v680, %v686 : tensor<32x384x14x14xf32>
    %v688 = stablehlo.broadcast_in_dim %ge9, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v689 = stablehlo.broadcast_in_dim %bte9, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v690 = stablehlo.multiply %v687, %v688 : tensor<32x384x14x14xf32>
    %v691 = stablehlo.add %v690, %v689 : tensor<32x384x14x14xf32>
    %v692 = stablehlo.reshape %v691 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v693 = stablehlo.constant dense<0.0> : tensor<32x75264xf32>
    %v694 = stablehlo.constant dense<6.0> : tensor<32x75264xf32>
    %v695 = stablehlo.maximum %v692, %v693 : tensor<32x75264xf32>
    %v696 = stablehlo.minimum %v695, %v694 : tensor<32x75264xf32>
    %v697 = stablehlo.reshape %v696 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v698 = stablehlo.convolution(%v697, %Wd9)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x3x3xf32>) -> tensor<32x384x14x14xf32>
    %v699 = stablehlo.broadcast_in_dim %zb384, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v700 = stablehlo.add %v698, %v699 : tensor<32x384x14x14xf32>
    %v701 = stablehlo.reshape %v700 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v702 = stablehlo.reshape %v701 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v703 = stablehlo.constant dense<0.0> : tensor<f32>
    %v704 = stablehlo.constant dense<196.0> : tensor<32x384x14x14xf32>
    %v705 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v706 = stablehlo.reduce(%v702 init: %v703) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v707 = stablehlo.broadcast_in_dim %v706, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v708 = stablehlo.divide %v707, %v704 : tensor<32x384x14x14xf32>
    %v709 = stablehlo.subtract %v702, %v708 : tensor<32x384x14x14xf32>
    %v710 = stablehlo.multiply %v709, %v709 : tensor<32x384x14x14xf32>
    %v711 = stablehlo.reduce(%v710 init: %v703) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v712 = stablehlo.broadcast_in_dim %v711, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v713 = stablehlo.divide %v712, %v704 : tensor<32x384x14x14xf32>
    %v714 = stablehlo.add %v713, %v705 : tensor<32x384x14x14xf32>
    %v715 = stablehlo.rsqrt %v714 : tensor<32x384x14x14xf32>
    %v716 = stablehlo.multiply %v709, %v715 : tensor<32x384x14x14xf32>
    %v717 = stablehlo.broadcast_in_dim %gd9, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v718 = stablehlo.broadcast_in_dim %btd9, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v719 = stablehlo.multiply %v716, %v717 : tensor<32x384x14x14xf32>
    %v720 = stablehlo.add %v719, %v718 : tensor<32x384x14x14xf32>
    %v721 = stablehlo.reshape %v720 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v722 = stablehlo.constant dense<0.0> : tensor<32x75264xf32>
    %v723 = stablehlo.constant dense<6.0> : tensor<32x75264xf32>
    %v724 = stablehlo.maximum %v721, %v722 : tensor<32x75264xf32>
    %v725 = stablehlo.minimum %v724, %v723 : tensor<32x75264xf32>
    %v726 = stablehlo.reshape %v725 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v727 = stablehlo.convolution(%v726, %Wp9)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<64x384x1x1xf32>) -> tensor<32x64x14x14xf32>
    %v728 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v729 = stablehlo.add %v727, %v728 : tensor<32x64x14x14xf32>
    %v730 = stablehlo.reshape %v729 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v731 = stablehlo.reshape %v730 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v732 = stablehlo.constant dense<0.0> : tensor<f32>
    %v733 = stablehlo.constant dense<196.0> : tensor<32x64x14x14xf32>
    %v734 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v735 = stablehlo.reduce(%v731 init: %v732) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v736 = stablehlo.broadcast_in_dim %v735, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v737 = stablehlo.divide %v736, %v733 : tensor<32x64x14x14xf32>
    %v738 = stablehlo.subtract %v731, %v737 : tensor<32x64x14x14xf32>
    %v739 = stablehlo.multiply %v738, %v738 : tensor<32x64x14x14xf32>
    %v740 = stablehlo.reduce(%v739 init: %v732) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v741 = stablehlo.broadcast_in_dim %v740, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v742 = stablehlo.divide %v741, %v733 : tensor<32x64x14x14xf32>
    %v743 = stablehlo.add %v742, %v734 : tensor<32x64x14x14xf32>
    %v744 = stablehlo.rsqrt %v743 : tensor<32x64x14x14xf32>
    %v745 = stablehlo.multiply %v738, %v744 : tensor<32x64x14x14xf32>
    %v746 = stablehlo.broadcast_in_dim %gp9, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v747 = stablehlo.broadcast_in_dim %btp9, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v748 = stablehlo.multiply %v745, %v746 : tensor<32x64x14x14xf32>
    %v749 = stablehlo.add %v748, %v747 : tensor<32x64x14x14xf32>
    %v750 = stablehlo.reshape %v749 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v751 = stablehlo.add %v750, %v667 : tensor<32x12544xf32>
    %v752 = stablehlo.reshape %v751 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v753 = stablehlo.convolution(%v752, %We10)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x14x14xf32>, tensor<384x64x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v754 = stablehlo.broadcast_in_dim %zb384, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v755 = stablehlo.add %v753, %v754 : tensor<32x384x14x14xf32>
    %v756 = stablehlo.reshape %v755 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v757 = stablehlo.reshape %v756 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v758 = stablehlo.constant dense<0.0> : tensor<f32>
    %v759 = stablehlo.constant dense<196.0> : tensor<32x384x14x14xf32>
    %v760 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v761 = stablehlo.reduce(%v757 init: %v758) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v762 = stablehlo.broadcast_in_dim %v761, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v763 = stablehlo.divide %v762, %v759 : tensor<32x384x14x14xf32>
    %v764 = stablehlo.subtract %v757, %v763 : tensor<32x384x14x14xf32>
    %v765 = stablehlo.multiply %v764, %v764 : tensor<32x384x14x14xf32>
    %v766 = stablehlo.reduce(%v765 init: %v758) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v767 = stablehlo.broadcast_in_dim %v766, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v768 = stablehlo.divide %v767, %v759 : tensor<32x384x14x14xf32>
    %v769 = stablehlo.add %v768, %v760 : tensor<32x384x14x14xf32>
    %v770 = stablehlo.rsqrt %v769 : tensor<32x384x14x14xf32>
    %v771 = stablehlo.multiply %v764, %v770 : tensor<32x384x14x14xf32>
    %v772 = stablehlo.broadcast_in_dim %ge10, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v773 = stablehlo.broadcast_in_dim %bte10, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v774 = stablehlo.multiply %v771, %v772 : tensor<32x384x14x14xf32>
    %v775 = stablehlo.add %v774, %v773 : tensor<32x384x14x14xf32>
    %v776 = stablehlo.reshape %v775 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v777 = stablehlo.constant dense<0.0> : tensor<32x75264xf32>
    %v778 = stablehlo.constant dense<6.0> : tensor<32x75264xf32>
    %v779 = stablehlo.maximum %v776, %v777 : tensor<32x75264xf32>
    %v780 = stablehlo.minimum %v779, %v778 : tensor<32x75264xf32>
    %v781 = stablehlo.reshape %v780 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v782 = stablehlo.convolution(%v781, %Wd10)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x3x3xf32>) -> tensor<32x384x14x14xf32>
    %v783 = stablehlo.broadcast_in_dim %zb384, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v784 = stablehlo.add %v782, %v783 : tensor<32x384x14x14xf32>
    %v785 = stablehlo.reshape %v784 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v786 = stablehlo.reshape %v785 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v787 = stablehlo.constant dense<0.0> : tensor<f32>
    %v788 = stablehlo.constant dense<196.0> : tensor<32x384x14x14xf32>
    %v789 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v790 = stablehlo.reduce(%v786 init: %v787) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v791 = stablehlo.broadcast_in_dim %v790, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v792 = stablehlo.divide %v791, %v788 : tensor<32x384x14x14xf32>
    %v793 = stablehlo.subtract %v786, %v792 : tensor<32x384x14x14xf32>
    %v794 = stablehlo.multiply %v793, %v793 : tensor<32x384x14x14xf32>
    %v795 = stablehlo.reduce(%v794 init: %v787) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v796 = stablehlo.broadcast_in_dim %v795, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v797 = stablehlo.divide %v796, %v788 : tensor<32x384x14x14xf32>
    %v798 = stablehlo.add %v797, %v789 : tensor<32x384x14x14xf32>
    %v799 = stablehlo.rsqrt %v798 : tensor<32x384x14x14xf32>
    %v800 = stablehlo.multiply %v793, %v799 : tensor<32x384x14x14xf32>
    %v801 = stablehlo.broadcast_in_dim %gd10, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v802 = stablehlo.broadcast_in_dim %btd10, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v803 = stablehlo.multiply %v800, %v801 : tensor<32x384x14x14xf32>
    %v804 = stablehlo.add %v803, %v802 : tensor<32x384x14x14xf32>
    %v805 = stablehlo.reshape %v804 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v806 = stablehlo.constant dense<0.0> : tensor<32x75264xf32>
    %v807 = stablehlo.constant dense<6.0> : tensor<32x75264xf32>
    %v808 = stablehlo.maximum %v805, %v806 : tensor<32x75264xf32>
    %v809 = stablehlo.minimum %v808, %v807 : tensor<32x75264xf32>
    %v810 = stablehlo.reshape %v809 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v811 = stablehlo.convolution(%v810, %Wp10)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<64x384x1x1xf32>) -> tensor<32x64x14x14xf32>
    %v812 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v813 = stablehlo.add %v811, %v812 : tensor<32x64x14x14xf32>
    %v814 = stablehlo.reshape %v813 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v815 = stablehlo.reshape %v814 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v816 = stablehlo.constant dense<0.0> : tensor<f32>
    %v817 = stablehlo.constant dense<196.0> : tensor<32x64x14x14xf32>
    %v818 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v819 = stablehlo.reduce(%v815 init: %v816) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v820 = stablehlo.broadcast_in_dim %v819, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v821 = stablehlo.divide %v820, %v817 : tensor<32x64x14x14xf32>
    %v822 = stablehlo.subtract %v815, %v821 : tensor<32x64x14x14xf32>
    %v823 = stablehlo.multiply %v822, %v822 : tensor<32x64x14x14xf32>
    %v824 = stablehlo.reduce(%v823 init: %v816) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v825 = stablehlo.broadcast_in_dim %v824, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v826 = stablehlo.divide %v825, %v817 : tensor<32x64x14x14xf32>
    %v827 = stablehlo.add %v826, %v818 : tensor<32x64x14x14xf32>
    %v828 = stablehlo.rsqrt %v827 : tensor<32x64x14x14xf32>
    %v829 = stablehlo.multiply %v822, %v828 : tensor<32x64x14x14xf32>
    %v830 = stablehlo.broadcast_in_dim %gp10, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v831 = stablehlo.broadcast_in_dim %btp10, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v832 = stablehlo.multiply %v829, %v830 : tensor<32x64x14x14xf32>
    %v833 = stablehlo.add %v832, %v831 : tensor<32x64x14x14xf32>
    %v834 = stablehlo.reshape %v833 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v835 = stablehlo.add %v834, %v751 : tensor<32x12544xf32>
    %v836 = stablehlo.reshape %v835 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v837 = stablehlo.convolution(%v836, %We11)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x14x14xf32>, tensor<384x64x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v838 = stablehlo.broadcast_in_dim %zb384, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v839 = stablehlo.add %v837, %v838 : tensor<32x384x14x14xf32>
    %v840 = stablehlo.reshape %v839 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v841 = stablehlo.reshape %v840 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v842 = stablehlo.constant dense<0.0> : tensor<f32>
    %v843 = stablehlo.constant dense<196.0> : tensor<32x384x14x14xf32>
    %v844 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v845 = stablehlo.reduce(%v841 init: %v842) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v846 = stablehlo.broadcast_in_dim %v845, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v847 = stablehlo.divide %v846, %v843 : tensor<32x384x14x14xf32>
    %v848 = stablehlo.subtract %v841, %v847 : tensor<32x384x14x14xf32>
    %v849 = stablehlo.multiply %v848, %v848 : tensor<32x384x14x14xf32>
    %v850 = stablehlo.reduce(%v849 init: %v842) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v851 = stablehlo.broadcast_in_dim %v850, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v852 = stablehlo.divide %v851, %v843 : tensor<32x384x14x14xf32>
    %v853 = stablehlo.add %v852, %v844 : tensor<32x384x14x14xf32>
    %v854 = stablehlo.rsqrt %v853 : tensor<32x384x14x14xf32>
    %v855 = stablehlo.multiply %v848, %v854 : tensor<32x384x14x14xf32>
    %v856 = stablehlo.broadcast_in_dim %ge11, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v857 = stablehlo.broadcast_in_dim %bte11, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v858 = stablehlo.multiply %v855, %v856 : tensor<32x384x14x14xf32>
    %v859 = stablehlo.add %v858, %v857 : tensor<32x384x14x14xf32>
    %v860 = stablehlo.reshape %v859 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v861 = stablehlo.constant dense<0.0> : tensor<32x75264xf32>
    %v862 = stablehlo.constant dense<6.0> : tensor<32x75264xf32>
    %v863 = stablehlo.maximum %v860, %v861 : tensor<32x75264xf32>
    %v864 = stablehlo.minimum %v863, %v862 : tensor<32x75264xf32>
    %v865 = stablehlo.reshape %v864 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v866 = stablehlo.convolution(%v865, %Wd11)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x3x3xf32>) -> tensor<32x384x14x14xf32>
    %v867 = stablehlo.broadcast_in_dim %zb384, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v868 = stablehlo.add %v866, %v867 : tensor<32x384x14x14xf32>
    %v869 = stablehlo.reshape %v868 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v870 = stablehlo.reshape %v869 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v871 = stablehlo.constant dense<0.0> : tensor<f32>
    %v872 = stablehlo.constant dense<196.0> : tensor<32x384x14x14xf32>
    %v873 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v874 = stablehlo.reduce(%v870 init: %v871) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v875 = stablehlo.broadcast_in_dim %v874, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v876 = stablehlo.divide %v875, %v872 : tensor<32x384x14x14xf32>
    %v877 = stablehlo.subtract %v870, %v876 : tensor<32x384x14x14xf32>
    %v878 = stablehlo.multiply %v877, %v877 : tensor<32x384x14x14xf32>
    %v879 = stablehlo.reduce(%v878 init: %v871) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v880 = stablehlo.broadcast_in_dim %v879, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v881 = stablehlo.divide %v880, %v872 : tensor<32x384x14x14xf32>
    %v882 = stablehlo.add %v881, %v873 : tensor<32x384x14x14xf32>
    %v883 = stablehlo.rsqrt %v882 : tensor<32x384x14x14xf32>
    %v884 = stablehlo.multiply %v877, %v883 : tensor<32x384x14x14xf32>
    %v885 = stablehlo.broadcast_in_dim %gd11, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v886 = stablehlo.broadcast_in_dim %btd11, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v887 = stablehlo.multiply %v884, %v885 : tensor<32x384x14x14xf32>
    %v888 = stablehlo.add %v887, %v886 : tensor<32x384x14x14xf32>
    %v889 = stablehlo.reshape %v888 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v890 = stablehlo.constant dense<0.0> : tensor<32x75264xf32>
    %v891 = stablehlo.constant dense<6.0> : tensor<32x75264xf32>
    %v892 = stablehlo.maximum %v889, %v890 : tensor<32x75264xf32>
    %v893 = stablehlo.minimum %v892, %v891 : tensor<32x75264xf32>
    %v894 = stablehlo.reshape %v893 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v895 = stablehlo.convolution(%v894, %Wp11)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<96x384x1x1xf32>) -> tensor<32x96x14x14xf32>
    %v896 = stablehlo.broadcast_in_dim %zb96, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v897 = stablehlo.add %v895, %v896 : tensor<32x96x14x14xf32>
    %v898 = stablehlo.reshape %v897 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v899 = stablehlo.reshape %v898 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v900 = stablehlo.constant dense<0.0> : tensor<f32>
    %v901 = stablehlo.constant dense<196.0> : tensor<32x96x14x14xf32>
    %v902 = stablehlo.constant dense<1.0e-5> : tensor<32x96x14x14xf32>
    %v903 = stablehlo.reduce(%v899 init: %v900) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v904 = stablehlo.broadcast_in_dim %v903, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x14x14xf32>
    %v905 = stablehlo.divide %v904, %v901 : tensor<32x96x14x14xf32>
    %v906 = stablehlo.subtract %v899, %v905 : tensor<32x96x14x14xf32>
    %v907 = stablehlo.multiply %v906, %v906 : tensor<32x96x14x14xf32>
    %v908 = stablehlo.reduce(%v907 init: %v900) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v909 = stablehlo.broadcast_in_dim %v908, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x14x14xf32>
    %v910 = stablehlo.divide %v909, %v901 : tensor<32x96x14x14xf32>
    %v911 = stablehlo.add %v910, %v902 : tensor<32x96x14x14xf32>
    %v912 = stablehlo.rsqrt %v911 : tensor<32x96x14x14xf32>
    %v913 = stablehlo.multiply %v906, %v912 : tensor<32x96x14x14xf32>
    %v914 = stablehlo.broadcast_in_dim %gp11, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v915 = stablehlo.broadcast_in_dim %btp11, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v916 = stablehlo.multiply %v913, %v914 : tensor<32x96x14x14xf32>
    %v917 = stablehlo.add %v916, %v915 : tensor<32x96x14x14xf32>
    %v918 = stablehlo.reshape %v917 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v919 = stablehlo.reshape %v918 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v920 = stablehlo.convolution(%v919, %We12)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x14x14xf32>, tensor<576x96x1x1xf32>) -> tensor<32x576x14x14xf32>
    %v921 = stablehlo.broadcast_in_dim %zb576, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v922 = stablehlo.add %v920, %v921 : tensor<32x576x14x14xf32>
    %v923 = stablehlo.reshape %v922 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v924 = stablehlo.reshape %v923 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v925 = stablehlo.constant dense<0.0> : tensor<f32>
    %v926 = stablehlo.constant dense<196.0> : tensor<32x576x14x14xf32>
    %v927 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v928 = stablehlo.reduce(%v924 init: %v925) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v929 = stablehlo.broadcast_in_dim %v928, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v930 = stablehlo.divide %v929, %v926 : tensor<32x576x14x14xf32>
    %v931 = stablehlo.subtract %v924, %v930 : tensor<32x576x14x14xf32>
    %v932 = stablehlo.multiply %v931, %v931 : tensor<32x576x14x14xf32>
    %v933 = stablehlo.reduce(%v932 init: %v925) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v934 = stablehlo.broadcast_in_dim %v933, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v935 = stablehlo.divide %v934, %v926 : tensor<32x576x14x14xf32>
    %v936 = stablehlo.add %v935, %v927 : tensor<32x576x14x14xf32>
    %v937 = stablehlo.rsqrt %v936 : tensor<32x576x14x14xf32>
    %v938 = stablehlo.multiply %v931, %v937 : tensor<32x576x14x14xf32>
    %v939 = stablehlo.broadcast_in_dim %ge12, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v940 = stablehlo.broadcast_in_dim %bte12, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v941 = stablehlo.multiply %v938, %v939 : tensor<32x576x14x14xf32>
    %v942 = stablehlo.add %v941, %v940 : tensor<32x576x14x14xf32>
    %v943 = stablehlo.reshape %v942 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v944 = stablehlo.constant dense<0.0> : tensor<32x112896xf32>
    %v945 = stablehlo.constant dense<6.0> : tensor<32x112896xf32>
    %v946 = stablehlo.maximum %v943, %v944 : tensor<32x112896xf32>
    %v947 = stablehlo.minimum %v946, %v945 : tensor<32x112896xf32>
    %v948 = stablehlo.reshape %v947 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v949 = stablehlo.convolution(%v948, %Wd12)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 576 : i64} : (tensor<32x576x14x14xf32>, tensor<576x1x3x3xf32>) -> tensor<32x576x14x14xf32>
    %v950 = stablehlo.broadcast_in_dim %zb576, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v951 = stablehlo.add %v949, %v950 : tensor<32x576x14x14xf32>
    %v952 = stablehlo.reshape %v951 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v953 = stablehlo.reshape %v952 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v954 = stablehlo.constant dense<0.0> : tensor<f32>
    %v955 = stablehlo.constant dense<196.0> : tensor<32x576x14x14xf32>
    %v956 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v957 = stablehlo.reduce(%v953 init: %v954) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v958 = stablehlo.broadcast_in_dim %v957, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v959 = stablehlo.divide %v958, %v955 : tensor<32x576x14x14xf32>
    %v960 = stablehlo.subtract %v953, %v959 : tensor<32x576x14x14xf32>
    %v961 = stablehlo.multiply %v960, %v960 : tensor<32x576x14x14xf32>
    %v962 = stablehlo.reduce(%v961 init: %v954) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v963 = stablehlo.broadcast_in_dim %v962, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v964 = stablehlo.divide %v963, %v955 : tensor<32x576x14x14xf32>
    %v965 = stablehlo.add %v964, %v956 : tensor<32x576x14x14xf32>
    %v966 = stablehlo.rsqrt %v965 : tensor<32x576x14x14xf32>
    %v967 = stablehlo.multiply %v960, %v966 : tensor<32x576x14x14xf32>
    %v968 = stablehlo.broadcast_in_dim %gd12, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v969 = stablehlo.broadcast_in_dim %btd12, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v970 = stablehlo.multiply %v967, %v968 : tensor<32x576x14x14xf32>
    %v971 = stablehlo.add %v970, %v969 : tensor<32x576x14x14xf32>
    %v972 = stablehlo.reshape %v971 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v973 = stablehlo.constant dense<0.0> : tensor<32x112896xf32>
    %v974 = stablehlo.constant dense<6.0> : tensor<32x112896xf32>
    %v975 = stablehlo.maximum %v972, %v973 : tensor<32x112896xf32>
    %v976 = stablehlo.minimum %v975, %v974 : tensor<32x112896xf32>
    %v977 = stablehlo.reshape %v976 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v978 = stablehlo.convolution(%v977, %Wp12)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x576x14x14xf32>, tensor<96x576x1x1xf32>) -> tensor<32x96x14x14xf32>
    %v979 = stablehlo.broadcast_in_dim %zb96, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v980 = stablehlo.add %v978, %v979 : tensor<32x96x14x14xf32>
    %v981 = stablehlo.reshape %v980 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v982 = stablehlo.reshape %v981 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v983 = stablehlo.constant dense<0.0> : tensor<f32>
    %v984 = stablehlo.constant dense<196.0> : tensor<32x96x14x14xf32>
    %v985 = stablehlo.constant dense<1.0e-5> : tensor<32x96x14x14xf32>
    %v986 = stablehlo.reduce(%v982 init: %v983) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v987 = stablehlo.broadcast_in_dim %v986, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x14x14xf32>
    %v988 = stablehlo.divide %v987, %v984 : tensor<32x96x14x14xf32>
    %v989 = stablehlo.subtract %v982, %v988 : tensor<32x96x14x14xf32>
    %v990 = stablehlo.multiply %v989, %v989 : tensor<32x96x14x14xf32>
    %v991 = stablehlo.reduce(%v990 init: %v983) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v992 = stablehlo.broadcast_in_dim %v991, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x14x14xf32>
    %v993 = stablehlo.divide %v992, %v984 : tensor<32x96x14x14xf32>
    %v994 = stablehlo.add %v993, %v985 : tensor<32x96x14x14xf32>
    %v995 = stablehlo.rsqrt %v994 : tensor<32x96x14x14xf32>
    %v996 = stablehlo.multiply %v989, %v995 : tensor<32x96x14x14xf32>
    %v997 = stablehlo.broadcast_in_dim %gp12, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v998 = stablehlo.broadcast_in_dim %btp12, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v999 = stablehlo.multiply %v996, %v997 : tensor<32x96x14x14xf32>
    %v1000 = stablehlo.add %v999, %v998 : tensor<32x96x14x14xf32>
    %v1001 = stablehlo.reshape %v1000 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v1002 = stablehlo.add %v1001, %v918 : tensor<32x18816xf32>
    %v1003 = stablehlo.reshape %v1002 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v1004 = stablehlo.convolution(%v1003, %We13)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x14x14xf32>, tensor<576x96x1x1xf32>) -> tensor<32x576x14x14xf32>
    %v1005 = stablehlo.broadcast_in_dim %zb576, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v1006 = stablehlo.add %v1004, %v1005 : tensor<32x576x14x14xf32>
    %v1007 = stablehlo.reshape %v1006 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v1008 = stablehlo.reshape %v1007 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v1009 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1010 = stablehlo.constant dense<196.0> : tensor<32x576x14x14xf32>
    %v1011 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v1012 = stablehlo.reduce(%v1008 init: %v1009) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v1013 = stablehlo.broadcast_in_dim %v1012, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v1014 = stablehlo.divide %v1013, %v1010 : tensor<32x576x14x14xf32>
    %v1015 = stablehlo.subtract %v1008, %v1014 : tensor<32x576x14x14xf32>
    %v1016 = stablehlo.multiply %v1015, %v1015 : tensor<32x576x14x14xf32>
    %v1017 = stablehlo.reduce(%v1016 init: %v1009) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v1018 = stablehlo.broadcast_in_dim %v1017, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v1019 = stablehlo.divide %v1018, %v1010 : tensor<32x576x14x14xf32>
    %v1020 = stablehlo.add %v1019, %v1011 : tensor<32x576x14x14xf32>
    %v1021 = stablehlo.rsqrt %v1020 : tensor<32x576x14x14xf32>
    %v1022 = stablehlo.multiply %v1015, %v1021 : tensor<32x576x14x14xf32>
    %v1023 = stablehlo.broadcast_in_dim %ge13, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v1024 = stablehlo.broadcast_in_dim %bte13, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v1025 = stablehlo.multiply %v1022, %v1023 : tensor<32x576x14x14xf32>
    %v1026 = stablehlo.add %v1025, %v1024 : tensor<32x576x14x14xf32>
    %v1027 = stablehlo.reshape %v1026 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v1028 = stablehlo.constant dense<0.0> : tensor<32x112896xf32>
    %v1029 = stablehlo.constant dense<6.0> : tensor<32x112896xf32>
    %v1030 = stablehlo.maximum %v1027, %v1028 : tensor<32x112896xf32>
    %v1031 = stablehlo.minimum %v1030, %v1029 : tensor<32x112896xf32>
    %v1032 = stablehlo.reshape %v1031 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v1033 = stablehlo.convolution(%v1032, %Wd13)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 576 : i64} : (tensor<32x576x14x14xf32>, tensor<576x1x3x3xf32>) -> tensor<32x576x14x14xf32>
    %v1034 = stablehlo.broadcast_in_dim %zb576, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v1035 = stablehlo.add %v1033, %v1034 : tensor<32x576x14x14xf32>
    %v1036 = stablehlo.reshape %v1035 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v1037 = stablehlo.reshape %v1036 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v1038 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1039 = stablehlo.constant dense<196.0> : tensor<32x576x14x14xf32>
    %v1040 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v1041 = stablehlo.reduce(%v1037 init: %v1038) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v1042 = stablehlo.broadcast_in_dim %v1041, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v1043 = stablehlo.divide %v1042, %v1039 : tensor<32x576x14x14xf32>
    %v1044 = stablehlo.subtract %v1037, %v1043 : tensor<32x576x14x14xf32>
    %v1045 = stablehlo.multiply %v1044, %v1044 : tensor<32x576x14x14xf32>
    %v1046 = stablehlo.reduce(%v1045 init: %v1038) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v1047 = stablehlo.broadcast_in_dim %v1046, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v1048 = stablehlo.divide %v1047, %v1039 : tensor<32x576x14x14xf32>
    %v1049 = stablehlo.add %v1048, %v1040 : tensor<32x576x14x14xf32>
    %v1050 = stablehlo.rsqrt %v1049 : tensor<32x576x14x14xf32>
    %v1051 = stablehlo.multiply %v1044, %v1050 : tensor<32x576x14x14xf32>
    %v1052 = stablehlo.broadcast_in_dim %gd13, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v1053 = stablehlo.broadcast_in_dim %btd13, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v1054 = stablehlo.multiply %v1051, %v1052 : tensor<32x576x14x14xf32>
    %v1055 = stablehlo.add %v1054, %v1053 : tensor<32x576x14x14xf32>
    %v1056 = stablehlo.reshape %v1055 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v1057 = stablehlo.constant dense<0.0> : tensor<32x112896xf32>
    %v1058 = stablehlo.constant dense<6.0> : tensor<32x112896xf32>
    %v1059 = stablehlo.maximum %v1056, %v1057 : tensor<32x112896xf32>
    %v1060 = stablehlo.minimum %v1059, %v1058 : tensor<32x112896xf32>
    %v1061 = stablehlo.reshape %v1060 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v1062 = stablehlo.convolution(%v1061, %Wp13)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x576x14x14xf32>, tensor<96x576x1x1xf32>) -> tensor<32x96x14x14xf32>
    %v1063 = stablehlo.broadcast_in_dim %zb96, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v1064 = stablehlo.add %v1062, %v1063 : tensor<32x96x14x14xf32>
    %v1065 = stablehlo.reshape %v1064 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v1066 = stablehlo.reshape %v1065 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v1067 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1068 = stablehlo.constant dense<196.0> : tensor<32x96x14x14xf32>
    %v1069 = stablehlo.constant dense<1.0e-5> : tensor<32x96x14x14xf32>
    %v1070 = stablehlo.reduce(%v1066 init: %v1067) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1071 = stablehlo.broadcast_in_dim %v1070, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x14x14xf32>
    %v1072 = stablehlo.divide %v1071, %v1068 : tensor<32x96x14x14xf32>
    %v1073 = stablehlo.subtract %v1066, %v1072 : tensor<32x96x14x14xf32>
    %v1074 = stablehlo.multiply %v1073, %v1073 : tensor<32x96x14x14xf32>
    %v1075 = stablehlo.reduce(%v1074 init: %v1067) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v1076 = stablehlo.broadcast_in_dim %v1075, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x14x14xf32>
    %v1077 = stablehlo.divide %v1076, %v1068 : tensor<32x96x14x14xf32>
    %v1078 = stablehlo.add %v1077, %v1069 : tensor<32x96x14x14xf32>
    %v1079 = stablehlo.rsqrt %v1078 : tensor<32x96x14x14xf32>
    %v1080 = stablehlo.multiply %v1073, %v1079 : tensor<32x96x14x14xf32>
    %v1081 = stablehlo.broadcast_in_dim %gp13, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v1082 = stablehlo.broadcast_in_dim %btp13, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v1083 = stablehlo.multiply %v1080, %v1081 : tensor<32x96x14x14xf32>
    %v1084 = stablehlo.add %v1083, %v1082 : tensor<32x96x14x14xf32>
    %v1085 = stablehlo.reshape %v1084 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v1086 = stablehlo.add %v1085, %v1002 : tensor<32x18816xf32>
    %v1087 = stablehlo.reshape %v1086 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v1088 = stablehlo.convolution(%v1087, %We14)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x14x14xf32>, tensor<576x96x1x1xf32>) -> tensor<32x576x14x14xf32>
    %v1089 = stablehlo.broadcast_in_dim %zb576, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v1090 = stablehlo.add %v1088, %v1089 : tensor<32x576x14x14xf32>
    %v1091 = stablehlo.reshape %v1090 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v1092 = stablehlo.reshape %v1091 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v1093 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1094 = stablehlo.constant dense<196.0> : tensor<32x576x14x14xf32>
    %v1095 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v1096 = stablehlo.reduce(%v1092 init: %v1093) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v1097 = stablehlo.broadcast_in_dim %v1096, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v1098 = stablehlo.divide %v1097, %v1094 : tensor<32x576x14x14xf32>
    %v1099 = stablehlo.subtract %v1092, %v1098 : tensor<32x576x14x14xf32>
    %v1100 = stablehlo.multiply %v1099, %v1099 : tensor<32x576x14x14xf32>
    %v1101 = stablehlo.reduce(%v1100 init: %v1093) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v1102 = stablehlo.broadcast_in_dim %v1101, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v1103 = stablehlo.divide %v1102, %v1094 : tensor<32x576x14x14xf32>
    %v1104 = stablehlo.add %v1103, %v1095 : tensor<32x576x14x14xf32>
    %v1105 = stablehlo.rsqrt %v1104 : tensor<32x576x14x14xf32>
    %v1106 = stablehlo.multiply %v1099, %v1105 : tensor<32x576x14x14xf32>
    %v1107 = stablehlo.broadcast_in_dim %ge14, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v1108 = stablehlo.broadcast_in_dim %bte14, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v1109 = stablehlo.multiply %v1106, %v1107 : tensor<32x576x14x14xf32>
    %v1110 = stablehlo.add %v1109, %v1108 : tensor<32x576x14x14xf32>
    %v1111 = stablehlo.reshape %v1110 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v1112 = stablehlo.constant dense<0.0> : tensor<32x112896xf32>
    %v1113 = stablehlo.constant dense<6.0> : tensor<32x112896xf32>
    %v1114 = stablehlo.maximum %v1111, %v1112 : tensor<32x112896xf32>
    %v1115 = stablehlo.minimum %v1114, %v1113 : tensor<32x112896xf32>
    %v1116 = stablehlo.reshape %v1115 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v1117 = stablehlo.convolution(%v1116, %Wd14)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 576 : i64} : (tensor<32x576x14x14xf32>, tensor<576x1x3x3xf32>) -> tensor<32x576x7x7xf32>
    %v1118 = stablehlo.broadcast_in_dim %zb576, dims = [1] : (tensor<576xf32>) -> tensor<32x576x7x7xf32>
    %v1119 = stablehlo.add %v1117, %v1118 : tensor<32x576x7x7xf32>
    %v1120 = stablehlo.reshape %v1119 : (tensor<32x576x7x7xf32>) -> tensor<32x28224xf32>
    %v1121 = stablehlo.reshape %v1120 : (tensor<32x28224xf32>) -> tensor<32x576x7x7xf32>
    %v1122 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1123 = stablehlo.constant dense<49.0> : tensor<32x576x7x7xf32>
    %v1124 = stablehlo.constant dense<1.0e-5> : tensor<32x576x7x7xf32>
    %v1125 = stablehlo.reduce(%v1121 init: %v1122) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x7x7xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v1126 = stablehlo.broadcast_in_dim %v1125, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x7x7xf32>
    %v1127 = stablehlo.divide %v1126, %v1123 : tensor<32x576x7x7xf32>
    %v1128 = stablehlo.subtract %v1121, %v1127 : tensor<32x576x7x7xf32>
    %v1129 = stablehlo.multiply %v1128, %v1128 : tensor<32x576x7x7xf32>
    %v1130 = stablehlo.reduce(%v1129 init: %v1122) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x7x7xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v1131 = stablehlo.broadcast_in_dim %v1130, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x7x7xf32>
    %v1132 = stablehlo.divide %v1131, %v1123 : tensor<32x576x7x7xf32>
    %v1133 = stablehlo.add %v1132, %v1124 : tensor<32x576x7x7xf32>
    %v1134 = stablehlo.rsqrt %v1133 : tensor<32x576x7x7xf32>
    %v1135 = stablehlo.multiply %v1128, %v1134 : tensor<32x576x7x7xf32>
    %v1136 = stablehlo.broadcast_in_dim %gd14, dims = [1] : (tensor<576xf32>) -> tensor<32x576x7x7xf32>
    %v1137 = stablehlo.broadcast_in_dim %btd14, dims = [1] : (tensor<576xf32>) -> tensor<32x576x7x7xf32>
    %v1138 = stablehlo.multiply %v1135, %v1136 : tensor<32x576x7x7xf32>
    %v1139 = stablehlo.add %v1138, %v1137 : tensor<32x576x7x7xf32>
    %v1140 = stablehlo.reshape %v1139 : (tensor<32x576x7x7xf32>) -> tensor<32x28224xf32>
    %v1141 = stablehlo.constant dense<0.0> : tensor<32x28224xf32>
    %v1142 = stablehlo.constant dense<6.0> : tensor<32x28224xf32>
    %v1143 = stablehlo.maximum %v1140, %v1141 : tensor<32x28224xf32>
    %v1144 = stablehlo.minimum %v1143, %v1142 : tensor<32x28224xf32>
    %v1145 = stablehlo.reshape %v1144 : (tensor<32x28224xf32>) -> tensor<32x576x7x7xf32>
    %v1146 = stablehlo.convolution(%v1145, %Wp14)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x576x7x7xf32>, tensor<160x576x1x1xf32>) -> tensor<32x160x7x7xf32>
    %v1147 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1148 = stablehlo.add %v1146, %v1147 : tensor<32x160x7x7xf32>
    %v1149 = stablehlo.reshape %v1148 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v1150 = stablehlo.reshape %v1149 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v1151 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1152 = stablehlo.constant dense<49.0> : tensor<32x160x7x7xf32>
    %v1153 = stablehlo.constant dense<1.0e-5> : tensor<32x160x7x7xf32>
    %v1154 = stablehlo.reduce(%v1150 init: %v1151) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<32x160xf32>
    %v1155 = stablehlo.broadcast_in_dim %v1154, dims = [0, 1] : (tensor<32x160xf32>) -> tensor<32x160x7x7xf32>
    %v1156 = stablehlo.divide %v1155, %v1152 : tensor<32x160x7x7xf32>
    %v1157 = stablehlo.subtract %v1150, %v1156 : tensor<32x160x7x7xf32>
    %v1158 = stablehlo.multiply %v1157, %v1157 : tensor<32x160x7x7xf32>
    %v1159 = stablehlo.reduce(%v1158 init: %v1151) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<32x160xf32>
    %v1160 = stablehlo.broadcast_in_dim %v1159, dims = [0, 1] : (tensor<32x160xf32>) -> tensor<32x160x7x7xf32>
    %v1161 = stablehlo.divide %v1160, %v1152 : tensor<32x160x7x7xf32>
    %v1162 = stablehlo.add %v1161, %v1153 : tensor<32x160x7x7xf32>
    %v1163 = stablehlo.rsqrt %v1162 : tensor<32x160x7x7xf32>
    %v1164 = stablehlo.multiply %v1157, %v1163 : tensor<32x160x7x7xf32>
    %v1165 = stablehlo.broadcast_in_dim %gp14, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1166 = stablehlo.broadcast_in_dim %btp14, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1167 = stablehlo.multiply %v1164, %v1165 : tensor<32x160x7x7xf32>
    %v1168 = stablehlo.add %v1167, %v1166 : tensor<32x160x7x7xf32>
    %v1169 = stablehlo.reshape %v1168 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v1170 = stablehlo.reshape %v1169 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v1171 = stablehlo.convolution(%v1170, %We15)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x7x7xf32>, tensor<960x160x1x1xf32>) -> tensor<32x960x7x7xf32>
    %v1172 = stablehlo.broadcast_in_dim %zb960, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1173 = stablehlo.add %v1171, %v1172 : tensor<32x960x7x7xf32>
    %v1174 = stablehlo.reshape %v1173 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1175 = stablehlo.reshape %v1174 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1176 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1177 = stablehlo.constant dense<49.0> : tensor<32x960x7x7xf32>
    %v1178 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v1179 = stablehlo.reduce(%v1175 init: %v1176) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1180 = stablehlo.broadcast_in_dim %v1179, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1181 = stablehlo.divide %v1180, %v1177 : tensor<32x960x7x7xf32>
    %v1182 = stablehlo.subtract %v1175, %v1181 : tensor<32x960x7x7xf32>
    %v1183 = stablehlo.multiply %v1182, %v1182 : tensor<32x960x7x7xf32>
    %v1184 = stablehlo.reduce(%v1183 init: %v1176) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1185 = stablehlo.broadcast_in_dim %v1184, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1186 = stablehlo.divide %v1185, %v1177 : tensor<32x960x7x7xf32>
    %v1187 = stablehlo.add %v1186, %v1178 : tensor<32x960x7x7xf32>
    %v1188 = stablehlo.rsqrt %v1187 : tensor<32x960x7x7xf32>
    %v1189 = stablehlo.multiply %v1182, %v1188 : tensor<32x960x7x7xf32>
    %v1190 = stablehlo.broadcast_in_dim %ge15, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1191 = stablehlo.broadcast_in_dim %bte15, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1192 = stablehlo.multiply %v1189, %v1190 : tensor<32x960x7x7xf32>
    %v1193 = stablehlo.add %v1192, %v1191 : tensor<32x960x7x7xf32>
    %v1194 = stablehlo.reshape %v1193 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1195 = stablehlo.constant dense<0.0> : tensor<32x47040xf32>
    %v1196 = stablehlo.constant dense<6.0> : tensor<32x47040xf32>
    %v1197 = stablehlo.maximum %v1194, %v1195 : tensor<32x47040xf32>
    %v1198 = stablehlo.minimum %v1197, %v1196 : tensor<32x47040xf32>
    %v1199 = stablehlo.reshape %v1198 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1200 = stablehlo.convolution(%v1199, %Wd15)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 960 : i64} : (tensor<32x960x7x7xf32>, tensor<960x1x3x3xf32>) -> tensor<32x960x7x7xf32>
    %v1201 = stablehlo.broadcast_in_dim %zb960, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1202 = stablehlo.add %v1200, %v1201 : tensor<32x960x7x7xf32>
    %v1203 = stablehlo.reshape %v1202 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1204 = stablehlo.reshape %v1203 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1205 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1206 = stablehlo.constant dense<49.0> : tensor<32x960x7x7xf32>
    %v1207 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v1208 = stablehlo.reduce(%v1204 init: %v1205) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1209 = stablehlo.broadcast_in_dim %v1208, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1210 = stablehlo.divide %v1209, %v1206 : tensor<32x960x7x7xf32>
    %v1211 = stablehlo.subtract %v1204, %v1210 : tensor<32x960x7x7xf32>
    %v1212 = stablehlo.multiply %v1211, %v1211 : tensor<32x960x7x7xf32>
    %v1213 = stablehlo.reduce(%v1212 init: %v1205) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1214 = stablehlo.broadcast_in_dim %v1213, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1215 = stablehlo.divide %v1214, %v1206 : tensor<32x960x7x7xf32>
    %v1216 = stablehlo.add %v1215, %v1207 : tensor<32x960x7x7xf32>
    %v1217 = stablehlo.rsqrt %v1216 : tensor<32x960x7x7xf32>
    %v1218 = stablehlo.multiply %v1211, %v1217 : tensor<32x960x7x7xf32>
    %v1219 = stablehlo.broadcast_in_dim %gd15, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1220 = stablehlo.broadcast_in_dim %btd15, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1221 = stablehlo.multiply %v1218, %v1219 : tensor<32x960x7x7xf32>
    %v1222 = stablehlo.add %v1221, %v1220 : tensor<32x960x7x7xf32>
    %v1223 = stablehlo.reshape %v1222 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1224 = stablehlo.constant dense<0.0> : tensor<32x47040xf32>
    %v1225 = stablehlo.constant dense<6.0> : tensor<32x47040xf32>
    %v1226 = stablehlo.maximum %v1223, %v1224 : tensor<32x47040xf32>
    %v1227 = stablehlo.minimum %v1226, %v1225 : tensor<32x47040xf32>
    %v1228 = stablehlo.reshape %v1227 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1229 = stablehlo.convolution(%v1228, %Wp15)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x960x7x7xf32>, tensor<160x960x1x1xf32>) -> tensor<32x160x7x7xf32>
    %v1230 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1231 = stablehlo.add %v1229, %v1230 : tensor<32x160x7x7xf32>
    %v1232 = stablehlo.reshape %v1231 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v1233 = stablehlo.reshape %v1232 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v1234 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1235 = stablehlo.constant dense<49.0> : tensor<32x160x7x7xf32>
    %v1236 = stablehlo.constant dense<1.0e-5> : tensor<32x160x7x7xf32>
    %v1237 = stablehlo.reduce(%v1233 init: %v1234) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<32x160xf32>
    %v1238 = stablehlo.broadcast_in_dim %v1237, dims = [0, 1] : (tensor<32x160xf32>) -> tensor<32x160x7x7xf32>
    %v1239 = stablehlo.divide %v1238, %v1235 : tensor<32x160x7x7xf32>
    %v1240 = stablehlo.subtract %v1233, %v1239 : tensor<32x160x7x7xf32>
    %v1241 = stablehlo.multiply %v1240, %v1240 : tensor<32x160x7x7xf32>
    %v1242 = stablehlo.reduce(%v1241 init: %v1234) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<32x160xf32>
    %v1243 = stablehlo.broadcast_in_dim %v1242, dims = [0, 1] : (tensor<32x160xf32>) -> tensor<32x160x7x7xf32>
    %v1244 = stablehlo.divide %v1243, %v1235 : tensor<32x160x7x7xf32>
    %v1245 = stablehlo.add %v1244, %v1236 : tensor<32x160x7x7xf32>
    %v1246 = stablehlo.rsqrt %v1245 : tensor<32x160x7x7xf32>
    %v1247 = stablehlo.multiply %v1240, %v1246 : tensor<32x160x7x7xf32>
    %v1248 = stablehlo.broadcast_in_dim %gp15, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1249 = stablehlo.broadcast_in_dim %btp15, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1250 = stablehlo.multiply %v1247, %v1248 : tensor<32x160x7x7xf32>
    %v1251 = stablehlo.add %v1250, %v1249 : tensor<32x160x7x7xf32>
    %v1252 = stablehlo.reshape %v1251 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v1253 = stablehlo.add %v1252, %v1169 : tensor<32x7840xf32>
    %v1254 = stablehlo.reshape %v1253 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v1255 = stablehlo.convolution(%v1254, %We16)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x7x7xf32>, tensor<960x160x1x1xf32>) -> tensor<32x960x7x7xf32>
    %v1256 = stablehlo.broadcast_in_dim %zb960, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1257 = stablehlo.add %v1255, %v1256 : tensor<32x960x7x7xf32>
    %v1258 = stablehlo.reshape %v1257 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1259 = stablehlo.reshape %v1258 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1260 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1261 = stablehlo.constant dense<49.0> : tensor<32x960x7x7xf32>
    %v1262 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v1263 = stablehlo.reduce(%v1259 init: %v1260) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1264 = stablehlo.broadcast_in_dim %v1263, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1265 = stablehlo.divide %v1264, %v1261 : tensor<32x960x7x7xf32>
    %v1266 = stablehlo.subtract %v1259, %v1265 : tensor<32x960x7x7xf32>
    %v1267 = stablehlo.multiply %v1266, %v1266 : tensor<32x960x7x7xf32>
    %v1268 = stablehlo.reduce(%v1267 init: %v1260) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1269 = stablehlo.broadcast_in_dim %v1268, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1270 = stablehlo.divide %v1269, %v1261 : tensor<32x960x7x7xf32>
    %v1271 = stablehlo.add %v1270, %v1262 : tensor<32x960x7x7xf32>
    %v1272 = stablehlo.rsqrt %v1271 : tensor<32x960x7x7xf32>
    %v1273 = stablehlo.multiply %v1266, %v1272 : tensor<32x960x7x7xf32>
    %v1274 = stablehlo.broadcast_in_dim %ge16, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1275 = stablehlo.broadcast_in_dim %bte16, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1276 = stablehlo.multiply %v1273, %v1274 : tensor<32x960x7x7xf32>
    %v1277 = stablehlo.add %v1276, %v1275 : tensor<32x960x7x7xf32>
    %v1278 = stablehlo.reshape %v1277 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1279 = stablehlo.constant dense<0.0> : tensor<32x47040xf32>
    %v1280 = stablehlo.constant dense<6.0> : tensor<32x47040xf32>
    %v1281 = stablehlo.maximum %v1278, %v1279 : tensor<32x47040xf32>
    %v1282 = stablehlo.minimum %v1281, %v1280 : tensor<32x47040xf32>
    %v1283 = stablehlo.reshape %v1282 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1284 = stablehlo.convolution(%v1283, %Wd16)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 960 : i64} : (tensor<32x960x7x7xf32>, tensor<960x1x3x3xf32>) -> tensor<32x960x7x7xf32>
    %v1285 = stablehlo.broadcast_in_dim %zb960, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1286 = stablehlo.add %v1284, %v1285 : tensor<32x960x7x7xf32>
    %v1287 = stablehlo.reshape %v1286 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1288 = stablehlo.reshape %v1287 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1289 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1290 = stablehlo.constant dense<49.0> : tensor<32x960x7x7xf32>
    %v1291 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v1292 = stablehlo.reduce(%v1288 init: %v1289) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1293 = stablehlo.broadcast_in_dim %v1292, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1294 = stablehlo.divide %v1293, %v1290 : tensor<32x960x7x7xf32>
    %v1295 = stablehlo.subtract %v1288, %v1294 : tensor<32x960x7x7xf32>
    %v1296 = stablehlo.multiply %v1295, %v1295 : tensor<32x960x7x7xf32>
    %v1297 = stablehlo.reduce(%v1296 init: %v1289) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1298 = stablehlo.broadcast_in_dim %v1297, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1299 = stablehlo.divide %v1298, %v1290 : tensor<32x960x7x7xf32>
    %v1300 = stablehlo.add %v1299, %v1291 : tensor<32x960x7x7xf32>
    %v1301 = stablehlo.rsqrt %v1300 : tensor<32x960x7x7xf32>
    %v1302 = stablehlo.multiply %v1295, %v1301 : tensor<32x960x7x7xf32>
    %v1303 = stablehlo.broadcast_in_dim %gd16, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1304 = stablehlo.broadcast_in_dim %btd16, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1305 = stablehlo.multiply %v1302, %v1303 : tensor<32x960x7x7xf32>
    %v1306 = stablehlo.add %v1305, %v1304 : tensor<32x960x7x7xf32>
    %v1307 = stablehlo.reshape %v1306 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1308 = stablehlo.constant dense<0.0> : tensor<32x47040xf32>
    %v1309 = stablehlo.constant dense<6.0> : tensor<32x47040xf32>
    %v1310 = stablehlo.maximum %v1307, %v1308 : tensor<32x47040xf32>
    %v1311 = stablehlo.minimum %v1310, %v1309 : tensor<32x47040xf32>
    %v1312 = stablehlo.reshape %v1311 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1313 = stablehlo.convolution(%v1312, %Wp16)
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
    %v1332 = stablehlo.broadcast_in_dim %gp16, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1333 = stablehlo.broadcast_in_dim %btp16, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1334 = stablehlo.multiply %v1331, %v1332 : tensor<32x160x7x7xf32>
    %v1335 = stablehlo.add %v1334, %v1333 : tensor<32x160x7x7xf32>
    %v1336 = stablehlo.reshape %v1335 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v1337 = stablehlo.add %v1336, %v1253 : tensor<32x7840xf32>
    %v1338 = stablehlo.reshape %v1337 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v1339 = stablehlo.convolution(%v1338, %We17)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x7x7xf32>, tensor<960x160x1x1xf32>) -> tensor<32x960x7x7xf32>
    %v1340 = stablehlo.broadcast_in_dim %zb960, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1341 = stablehlo.add %v1339, %v1340 : tensor<32x960x7x7xf32>
    %v1342 = stablehlo.reshape %v1341 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1343 = stablehlo.reshape %v1342 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1344 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1345 = stablehlo.constant dense<49.0> : tensor<32x960x7x7xf32>
    %v1346 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v1347 = stablehlo.reduce(%v1343 init: %v1344) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1348 = stablehlo.broadcast_in_dim %v1347, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1349 = stablehlo.divide %v1348, %v1345 : tensor<32x960x7x7xf32>
    %v1350 = stablehlo.subtract %v1343, %v1349 : tensor<32x960x7x7xf32>
    %v1351 = stablehlo.multiply %v1350, %v1350 : tensor<32x960x7x7xf32>
    %v1352 = stablehlo.reduce(%v1351 init: %v1344) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1353 = stablehlo.broadcast_in_dim %v1352, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1354 = stablehlo.divide %v1353, %v1345 : tensor<32x960x7x7xf32>
    %v1355 = stablehlo.add %v1354, %v1346 : tensor<32x960x7x7xf32>
    %v1356 = stablehlo.rsqrt %v1355 : tensor<32x960x7x7xf32>
    %v1357 = stablehlo.multiply %v1350, %v1356 : tensor<32x960x7x7xf32>
    %v1358 = stablehlo.broadcast_in_dim %ge17, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1359 = stablehlo.broadcast_in_dim %bte17, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1360 = stablehlo.multiply %v1357, %v1358 : tensor<32x960x7x7xf32>
    %v1361 = stablehlo.add %v1360, %v1359 : tensor<32x960x7x7xf32>
    %v1362 = stablehlo.reshape %v1361 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1363 = stablehlo.constant dense<0.0> : tensor<32x47040xf32>
    %v1364 = stablehlo.constant dense<6.0> : tensor<32x47040xf32>
    %v1365 = stablehlo.maximum %v1362, %v1363 : tensor<32x47040xf32>
    %v1366 = stablehlo.minimum %v1365, %v1364 : tensor<32x47040xf32>
    %v1367 = stablehlo.reshape %v1366 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1368 = stablehlo.convolution(%v1367, %Wd17)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 960 : i64} : (tensor<32x960x7x7xf32>, tensor<960x1x3x3xf32>) -> tensor<32x960x7x7xf32>
    %v1369 = stablehlo.broadcast_in_dim %zb960, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1370 = stablehlo.add %v1368, %v1369 : tensor<32x960x7x7xf32>
    %v1371 = stablehlo.reshape %v1370 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1372 = stablehlo.reshape %v1371 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1373 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1374 = stablehlo.constant dense<49.0> : tensor<32x960x7x7xf32>
    %v1375 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v1376 = stablehlo.reduce(%v1372 init: %v1373) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1377 = stablehlo.broadcast_in_dim %v1376, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1378 = stablehlo.divide %v1377, %v1374 : tensor<32x960x7x7xf32>
    %v1379 = stablehlo.subtract %v1372, %v1378 : tensor<32x960x7x7xf32>
    %v1380 = stablehlo.multiply %v1379, %v1379 : tensor<32x960x7x7xf32>
    %v1381 = stablehlo.reduce(%v1380 init: %v1373) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1382 = stablehlo.broadcast_in_dim %v1381, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1383 = stablehlo.divide %v1382, %v1374 : tensor<32x960x7x7xf32>
    %v1384 = stablehlo.add %v1383, %v1375 : tensor<32x960x7x7xf32>
    %v1385 = stablehlo.rsqrt %v1384 : tensor<32x960x7x7xf32>
    %v1386 = stablehlo.multiply %v1379, %v1385 : tensor<32x960x7x7xf32>
    %v1387 = stablehlo.broadcast_in_dim %gd17, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1388 = stablehlo.broadcast_in_dim %btd17, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1389 = stablehlo.multiply %v1386, %v1387 : tensor<32x960x7x7xf32>
    %v1390 = stablehlo.add %v1389, %v1388 : tensor<32x960x7x7xf32>
    %v1391 = stablehlo.reshape %v1390 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1392 = stablehlo.constant dense<0.0> : tensor<32x47040xf32>
    %v1393 = stablehlo.constant dense<6.0> : tensor<32x47040xf32>
    %v1394 = stablehlo.maximum %v1391, %v1392 : tensor<32x47040xf32>
    %v1395 = stablehlo.minimum %v1394, %v1393 : tensor<32x47040xf32>
    %v1396 = stablehlo.reshape %v1395 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1397 = stablehlo.convolution(%v1396, %Wp17)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x960x7x7xf32>, tensor<320x960x1x1xf32>) -> tensor<32x320x7x7xf32>
    %v1398 = stablehlo.broadcast_in_dim %zb320, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1399 = stablehlo.add %v1397, %v1398 : tensor<32x320x7x7xf32>
    %v1400 = stablehlo.reshape %v1399 : (tensor<32x320x7x7xf32>) -> tensor<32x15680xf32>
    %v1401 = stablehlo.reshape %v1400 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1402 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1403 = stablehlo.constant dense<49.0> : tensor<32x320x7x7xf32>
    %v1404 = stablehlo.constant dense<1.0e-5> : tensor<32x320x7x7xf32>
    %v1405 = stablehlo.reduce(%v1401 init: %v1402) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<32x320xf32>
    %v1406 = stablehlo.broadcast_in_dim %v1405, dims = [0, 1] : (tensor<32x320xf32>) -> tensor<32x320x7x7xf32>
    %v1407 = stablehlo.divide %v1406, %v1403 : tensor<32x320x7x7xf32>
    %v1408 = stablehlo.subtract %v1401, %v1407 : tensor<32x320x7x7xf32>
    %v1409 = stablehlo.multiply %v1408, %v1408 : tensor<32x320x7x7xf32>
    %v1410 = stablehlo.reduce(%v1409 init: %v1402) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<32x320xf32>
    %v1411 = stablehlo.broadcast_in_dim %v1410, dims = [0, 1] : (tensor<32x320xf32>) -> tensor<32x320x7x7xf32>
    %v1412 = stablehlo.divide %v1411, %v1403 : tensor<32x320x7x7xf32>
    %v1413 = stablehlo.add %v1412, %v1404 : tensor<32x320x7x7xf32>
    %v1414 = stablehlo.rsqrt %v1413 : tensor<32x320x7x7xf32>
    %v1415 = stablehlo.multiply %v1408, %v1414 : tensor<32x320x7x7xf32>
    %v1416 = stablehlo.broadcast_in_dim %gp17, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1417 = stablehlo.broadcast_in_dim %btp17, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1418 = stablehlo.multiply %v1415, %v1416 : tensor<32x320x7x7xf32>
    %v1419 = stablehlo.add %v1418, %v1417 : tensor<32x320x7x7xf32>
    %v1420 = stablehlo.reshape %v1419 : (tensor<32x320x7x7xf32>) -> tensor<32x15680xf32>
    %v1421 = stablehlo.reshape %v1420 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1422 = stablehlo.convolution(%v1421, %Wh)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x320x7x7xf32>, tensor<1280x320x1x1xf32>) -> tensor<32x1280x7x7xf32>
    %v1423 = stablehlo.broadcast_in_dim %zb1280, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1424 = stablehlo.add %v1422, %v1423 : tensor<32x1280x7x7xf32>
    %v1425 = stablehlo.reshape %v1424 : (tensor<32x1280x7x7xf32>) -> tensor<32x62720xf32>
    %v1426 = stablehlo.reshape %v1425 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1427 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1428 = stablehlo.constant dense<49.0> : tensor<32x1280x7x7xf32>
    %v1429 = stablehlo.constant dense<1.0e-5> : tensor<32x1280x7x7xf32>
    %v1430 = stablehlo.reduce(%v1426 init: %v1427) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<32x1280xf32>
    %v1431 = stablehlo.broadcast_in_dim %v1430, dims = [0, 1] : (tensor<32x1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1432 = stablehlo.divide %v1431, %v1428 : tensor<32x1280x7x7xf32>
    %v1433 = stablehlo.subtract %v1426, %v1432 : tensor<32x1280x7x7xf32>
    %v1434 = stablehlo.multiply %v1433, %v1433 : tensor<32x1280x7x7xf32>
    %v1435 = stablehlo.reduce(%v1434 init: %v1427) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<32x1280xf32>
    %v1436 = stablehlo.broadcast_in_dim %v1435, dims = [0, 1] : (tensor<32x1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1437 = stablehlo.divide %v1436, %v1428 : tensor<32x1280x7x7xf32>
    %v1438 = stablehlo.add %v1437, %v1429 : tensor<32x1280x7x7xf32>
    %v1439 = stablehlo.rsqrt %v1438 : tensor<32x1280x7x7xf32>
    %v1440 = stablehlo.multiply %v1433, %v1439 : tensor<32x1280x7x7xf32>
    %v1441 = stablehlo.broadcast_in_dim %gh, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1442 = stablehlo.broadcast_in_dim %bth, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1443 = stablehlo.multiply %v1440, %v1441 : tensor<32x1280x7x7xf32>
    %v1444 = stablehlo.add %v1443, %v1442 : tensor<32x1280x7x7xf32>
    %v1445 = stablehlo.reshape %v1444 : (tensor<32x1280x7x7xf32>) -> tensor<32x62720xf32>
    %v1446 = stablehlo.constant dense<0.0> : tensor<32x62720xf32>
    %v1447 = stablehlo.constant dense<6.0> : tensor<32x62720xf32>
    %v1448 = stablehlo.maximum %v1445, %v1446 : tensor<32x62720xf32>
    %v1449 = stablehlo.minimum %v1448, %v1447 : tensor<32x62720xf32>
    %v1450 = stablehlo.reshape %v1449 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1451 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1452 = stablehlo.reduce(%v1450 init: %v1451) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<32x1280xf32>
    %v1453 = stablehlo.constant dense<49.0> : tensor<32x1280xf32>
    %v1454 = stablehlo.divide %v1452, %v1453 : tensor<32x1280xf32>
    %v1455 = stablehlo.dot_general %v1454, %Wfc, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1280xf32>, tensor<1280x10xf32>) -> tensor<32x10xf32>
    %v1456 = stablehlo.broadcast_in_dim %bfc, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v1457 = stablehlo.add %v1455, %v1456 : tensor<32x10xf32>
    %v1458 = stablehlo.exponential %v1457 : tensor<32x10xf32>
    %v1459 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1460 = stablehlo.reduce(%v1458 init: %v1459) applies stablehlo.add across dimensions = [1] : (tensor<32x10xf32>, tensor<f32>) -> tensor<32xf32>
    %v1461 = stablehlo.broadcast_in_dim %v1460, dims = [0] : (tensor<32xf32>) -> tensor<32x10xf32>
    %v1462 = stablehlo.divide %v1458, %v1461 : tensor<32x10xf32>
    %v1463 = stablehlo.subtract %v1462, %onehot : tensor<32x10xf32>
    %v1464 = stablehlo.dot_general %v1463, %Wfc, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<1280x10xf32>) -> tensor<32x1280xf32>
    %v1465 = stablehlo.constant dense<49.0> : tensor<32x1280xf32>
    %v1466 = stablehlo.divide %v1464, %v1465 : tensor<32x1280xf32>
    %v1467 = stablehlo.broadcast_in_dim %v1466, dims = [0, 1] : (tensor<32x1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1468 = stablehlo.reshape %v1467 : (tensor<32x1280x7x7xf32>) -> tensor<32x62720xf32>
    %v1469 = stablehlo.dot_general %v1454, %v1463, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1280xf32>, tensor<32x10xf32>) -> tensor<1280x10xf32>
    %v1470 = stablehlo.constant dense<0.3> : tensor<1280x10xf32>
    %v1471 = stablehlo.multiply %v1469, %v1470 : tensor<1280x10xf32>
    %v1472 = stablehlo.subtract %Wfc, %v1471 : tensor<1280x10xf32>
    %v1473 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1474 = stablehlo.reduce(%v1463 init: %v1473) applies stablehlo.add across dimensions = [0] : (tensor<32x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v1475 = stablehlo.constant dense<0.3> : tensor<10xf32>
    %v1476 = stablehlo.multiply %v1474, %v1475 : tensor<10xf32>
    %v1477 = stablehlo.subtract %bfc, %v1476 : tensor<10xf32>
    %v1478 = stablehlo.constant dense<0.0> : tensor<32x62720xf32>
    %v1479 = stablehlo.constant dense<6.0> : tensor<32x62720xf32>
    %v1480 = stablehlo.compare GT, %v1445, %v1478 : (tensor<32x62720xf32>, tensor<32x62720xf32>) -> tensor<32x62720xi1>
    %v1481 = stablehlo.compare LT, %v1445, %v1479 : (tensor<32x62720xf32>, tensor<32x62720xf32>) -> tensor<32x62720xi1>
    %v1482 = stablehlo.and %v1480, %v1481 : tensor<32x62720xi1>
    %v1483 = stablehlo.select %v1482, %v1468, %v1478 : tensor<32x62720xi1>, tensor<32x62720xf32>
    %v1484 = stablehlo.reshape %v1483 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1485 = stablehlo.reshape %v1425 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1486 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1487 = stablehlo.constant dense<49.0> : tensor<32x1280x7x7xf32>
    %v1488 = stablehlo.constant dense<1.0e-5> : tensor<32x1280x7x7xf32>
    %v1489 = stablehlo.reduce(%v1485 init: %v1486) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<32x1280xf32>
    %v1490 = stablehlo.broadcast_in_dim %v1489, dims = [0, 1] : (tensor<32x1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1491 = stablehlo.divide %v1490, %v1487 : tensor<32x1280x7x7xf32>
    %v1492 = stablehlo.subtract %v1485, %v1491 : tensor<32x1280x7x7xf32>
    %v1493 = stablehlo.multiply %v1492, %v1492 : tensor<32x1280x7x7xf32>
    %v1494 = stablehlo.reduce(%v1493 init: %v1486) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<32x1280xf32>
    %v1495 = stablehlo.broadcast_in_dim %v1494, dims = [0, 1] : (tensor<32x1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1496 = stablehlo.divide %v1495, %v1487 : tensor<32x1280x7x7xf32>
    %v1497 = stablehlo.add %v1496, %v1488 : tensor<32x1280x7x7xf32>
    %v1498 = stablehlo.rsqrt %v1497 : tensor<32x1280x7x7xf32>
    %v1499 = stablehlo.multiply %v1492, %v1498 : tensor<32x1280x7x7xf32>
    %v1500 = stablehlo.broadcast_in_dim %gh, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1501 = stablehlo.multiply %v1500, %v1484 : tensor<32x1280x7x7xf32>
    %v1502 = stablehlo.reduce(%v1501 init: %v1486) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<32x1280xf32>
    %v1503 = stablehlo.broadcast_in_dim %v1502, dims = [0, 1] : (tensor<32x1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1504 = stablehlo.multiply %v1499, %v1501 : tensor<32x1280x7x7xf32>
    %v1505 = stablehlo.reduce(%v1504 init: %v1486) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<32x1280xf32>
    %v1506 = stablehlo.broadcast_in_dim %v1505, dims = [0, 1] : (tensor<32x1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1507 = stablehlo.multiply %v1501, %v1487 : tensor<32x1280x7x7xf32>
    %v1508 = stablehlo.subtract %v1507, %v1503 : tensor<32x1280x7x7xf32>
    %v1509 = stablehlo.multiply %v1499, %v1506 : tensor<32x1280x7x7xf32>
    %v1510 = stablehlo.subtract %v1508, %v1509 : tensor<32x1280x7x7xf32>
    %v1511 = stablehlo.divide %v1498, %v1487 : tensor<32x1280x7x7xf32>
    %v1512 = stablehlo.multiply %v1511, %v1510 : tensor<32x1280x7x7xf32>
    %v1513 = stablehlo.reshape %v1512 : (tensor<32x1280x7x7xf32>) -> tensor<32x62720xf32>
    %v1514 = stablehlo.reshape %v1513 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1515 = stablehlo.transpose %Wh, dims = [1, 0, 2, 3] : (tensor<1280x320x1x1xf32>) -> tensor<320x1280x1x1xf32>
    %v1516 = stablehlo.reverse %v1515, dims = [2, 3] : tensor<320x1280x1x1xf32>
    %v1517 = stablehlo.convolution(%v1514, %v1516)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1280x7x7xf32>, tensor<320x1280x1x1xf32>) -> tensor<32x320x7x7xf32>
    %v1518 = stablehlo.reshape %v1517 : (tensor<32x320x7x7xf32>) -> tensor<32x15680xf32>
    %v1519 = stablehlo.reshape %v1420 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1520 = stablehlo.reshape %v1513 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1521 = stablehlo.transpose %v1519, dims = [1, 0, 2, 3] : (tensor<32x320x7x7xf32>) -> tensor<320x32x7x7xf32>
    %v1522 = stablehlo.transpose %v1520, dims = [1, 0, 2, 3] : (tensor<32x1280x7x7xf32>) -> tensor<1280x32x7x7xf32>
    %v1523 = stablehlo.convolution(%v1521, %v1522)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<320x32x7x7xf32>, tensor<1280x32x7x7xf32>) -> tensor<320x1280x1x1xf32>
    %v1524 = stablehlo.transpose %v1523, dims = [1, 0, 2, 3] : (tensor<320x1280x1x1xf32>) -> tensor<1280x320x1x1xf32>
    %v1525 = stablehlo.constant dense<0.3> : tensor<1280x320x1x1xf32>
    %v1526 = stablehlo.multiply %v1524, %v1525 : tensor<1280x320x1x1xf32>
    %v1527 = stablehlo.subtract %Wh, %v1526 : tensor<1280x320x1x1xf32>
    %v1528 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1529 = stablehlo.reshape %v1425 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1530 = stablehlo.constant dense<49.0> : tensor<32x1280x7x7xf32>
    %v1531 = stablehlo.constant dense<1.0e-5> : tensor<32x1280x7x7xf32>
    %v1532 = stablehlo.reduce(%v1529 init: %v1528) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<32x1280xf32>
    %v1533 = stablehlo.broadcast_in_dim %v1532, dims = [0, 1] : (tensor<32x1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1534 = stablehlo.divide %v1533, %v1530 : tensor<32x1280x7x7xf32>
    %v1535 = stablehlo.subtract %v1529, %v1534 : tensor<32x1280x7x7xf32>
    %v1536 = stablehlo.multiply %v1535, %v1535 : tensor<32x1280x7x7xf32>
    %v1537 = stablehlo.reduce(%v1536 init: %v1528) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<32x1280xf32>
    %v1538 = stablehlo.broadcast_in_dim %v1537, dims = [0, 1] : (tensor<32x1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1539 = stablehlo.divide %v1538, %v1530 : tensor<32x1280x7x7xf32>
    %v1540 = stablehlo.add %v1539, %v1531 : tensor<32x1280x7x7xf32>
    %v1541 = stablehlo.rsqrt %v1540 : tensor<32x1280x7x7xf32>
    %v1542 = stablehlo.multiply %v1535, %v1541 : tensor<32x1280x7x7xf32>
    %v1543 = stablehlo.reshape %v1483 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1544 = stablehlo.multiply %v1543, %v1542 : tensor<32x1280x7x7xf32>
    %v1545 = stablehlo.reduce(%v1544 init: %v1528) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v1546 = stablehlo.constant dense<0.3> : tensor<1280xf32>
    %v1547 = stablehlo.multiply %v1545, %v1546 : tensor<1280xf32>
    %v1548 = stablehlo.subtract %gh, %v1547 : tensor<1280xf32>
    %v1549 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1550 = stablehlo.reshape %v1483 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1551 = stablehlo.reduce(%v1550 init: %v1549) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v1552 = stablehlo.constant dense<0.3> : tensor<1280xf32>
    %v1553 = stablehlo.multiply %v1551, %v1552 : tensor<1280xf32>
    %v1554 = stablehlo.subtract %bth, %v1553 : tensor<1280xf32>
    %v1555 = stablehlo.reshape %v1518 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1556 = stablehlo.reshape %v1400 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1557 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1558 = stablehlo.constant dense<49.0> : tensor<32x320x7x7xf32>
    %v1559 = stablehlo.constant dense<1.0e-5> : tensor<32x320x7x7xf32>
    %v1560 = stablehlo.reduce(%v1556 init: %v1557) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<32x320xf32>
    %v1561 = stablehlo.broadcast_in_dim %v1560, dims = [0, 1] : (tensor<32x320xf32>) -> tensor<32x320x7x7xf32>
    %v1562 = stablehlo.divide %v1561, %v1558 : tensor<32x320x7x7xf32>
    %v1563 = stablehlo.subtract %v1556, %v1562 : tensor<32x320x7x7xf32>
    %v1564 = stablehlo.multiply %v1563, %v1563 : tensor<32x320x7x7xf32>
    %v1565 = stablehlo.reduce(%v1564 init: %v1557) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<32x320xf32>
    %v1566 = stablehlo.broadcast_in_dim %v1565, dims = [0, 1] : (tensor<32x320xf32>) -> tensor<32x320x7x7xf32>
    %v1567 = stablehlo.divide %v1566, %v1558 : tensor<32x320x7x7xf32>
    %v1568 = stablehlo.add %v1567, %v1559 : tensor<32x320x7x7xf32>
    %v1569 = stablehlo.rsqrt %v1568 : tensor<32x320x7x7xf32>
    %v1570 = stablehlo.multiply %v1563, %v1569 : tensor<32x320x7x7xf32>
    %v1571 = stablehlo.broadcast_in_dim %gp17, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1572 = stablehlo.multiply %v1571, %v1555 : tensor<32x320x7x7xf32>
    %v1573 = stablehlo.reduce(%v1572 init: %v1557) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<32x320xf32>
    %v1574 = stablehlo.broadcast_in_dim %v1573, dims = [0, 1] : (tensor<32x320xf32>) -> tensor<32x320x7x7xf32>
    %v1575 = stablehlo.multiply %v1570, %v1572 : tensor<32x320x7x7xf32>
    %v1576 = stablehlo.reduce(%v1575 init: %v1557) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<32x320xf32>
    %v1577 = stablehlo.broadcast_in_dim %v1576, dims = [0, 1] : (tensor<32x320xf32>) -> tensor<32x320x7x7xf32>
    %v1578 = stablehlo.multiply %v1572, %v1558 : tensor<32x320x7x7xf32>
    %v1579 = stablehlo.subtract %v1578, %v1574 : tensor<32x320x7x7xf32>
    %v1580 = stablehlo.multiply %v1570, %v1577 : tensor<32x320x7x7xf32>
    %v1581 = stablehlo.subtract %v1579, %v1580 : tensor<32x320x7x7xf32>
    %v1582 = stablehlo.divide %v1569, %v1558 : tensor<32x320x7x7xf32>
    %v1583 = stablehlo.multiply %v1582, %v1581 : tensor<32x320x7x7xf32>
    %v1584 = stablehlo.reshape %v1583 : (tensor<32x320x7x7xf32>) -> tensor<32x15680xf32>
    %v1585 = stablehlo.reshape %v1584 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1586 = stablehlo.transpose %Wp17, dims = [1, 0, 2, 3] : (tensor<320x960x1x1xf32>) -> tensor<960x320x1x1xf32>
    %v1587 = stablehlo.reverse %v1586, dims = [2, 3] : tensor<960x320x1x1xf32>
    %v1588 = stablehlo.convolution(%v1585, %v1587)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x320x7x7xf32>, tensor<960x320x1x1xf32>) -> tensor<32x960x7x7xf32>
    %v1589 = stablehlo.reshape %v1588 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1590 = stablehlo.constant dense<0.0> : tensor<32x47040xf32>
    %v1591 = stablehlo.constant dense<6.0> : tensor<32x47040xf32>
    %v1592 = stablehlo.compare GT, %v1391, %v1590 : (tensor<32x47040xf32>, tensor<32x47040xf32>) -> tensor<32x47040xi1>
    %v1593 = stablehlo.compare LT, %v1391, %v1591 : (tensor<32x47040xf32>, tensor<32x47040xf32>) -> tensor<32x47040xi1>
    %v1594 = stablehlo.and %v1592, %v1593 : tensor<32x47040xi1>
    %v1595 = stablehlo.select %v1594, %v1589, %v1590 : tensor<32x47040xi1>, tensor<32x47040xf32>
    %v1596 = stablehlo.reshape %v1595 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1597 = stablehlo.reshape %v1371 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1598 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1599 = stablehlo.constant dense<49.0> : tensor<32x960x7x7xf32>
    %v1600 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v1601 = stablehlo.reduce(%v1597 init: %v1598) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1602 = stablehlo.broadcast_in_dim %v1601, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1603 = stablehlo.divide %v1602, %v1599 : tensor<32x960x7x7xf32>
    %v1604 = stablehlo.subtract %v1597, %v1603 : tensor<32x960x7x7xf32>
    %v1605 = stablehlo.multiply %v1604, %v1604 : tensor<32x960x7x7xf32>
    %v1606 = stablehlo.reduce(%v1605 init: %v1598) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1607 = stablehlo.broadcast_in_dim %v1606, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1608 = stablehlo.divide %v1607, %v1599 : tensor<32x960x7x7xf32>
    %v1609 = stablehlo.add %v1608, %v1600 : tensor<32x960x7x7xf32>
    %v1610 = stablehlo.rsqrt %v1609 : tensor<32x960x7x7xf32>
    %v1611 = stablehlo.multiply %v1604, %v1610 : tensor<32x960x7x7xf32>
    %v1612 = stablehlo.broadcast_in_dim %gd17, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1613 = stablehlo.multiply %v1612, %v1596 : tensor<32x960x7x7xf32>
    %v1614 = stablehlo.reduce(%v1613 init: %v1598) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1615 = stablehlo.broadcast_in_dim %v1614, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1616 = stablehlo.multiply %v1611, %v1613 : tensor<32x960x7x7xf32>
    %v1617 = stablehlo.reduce(%v1616 init: %v1598) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1618 = stablehlo.broadcast_in_dim %v1617, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1619 = stablehlo.multiply %v1613, %v1599 : tensor<32x960x7x7xf32>
    %v1620 = stablehlo.subtract %v1619, %v1615 : tensor<32x960x7x7xf32>
    %v1621 = stablehlo.multiply %v1611, %v1618 : tensor<32x960x7x7xf32>
    %v1622 = stablehlo.subtract %v1620, %v1621 : tensor<32x960x7x7xf32>
    %v1623 = stablehlo.divide %v1610, %v1599 : tensor<32x960x7x7xf32>
    %v1624 = stablehlo.multiply %v1623, %v1622 : tensor<32x960x7x7xf32>
    %v1625 = stablehlo.reshape %v1624 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1626 = stablehlo.reshape %v1625 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1627 = stablehlo.reverse %Wd17, dims = [2, 3] : tensor<960x1x3x3xf32>
    %v1628 = stablehlo.convolution(%v1626, %v1627)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 960 : i64} : (tensor<32x960x7x7xf32>, tensor<960x1x3x3xf32>) -> tensor<32x960x7x7xf32>
    %v1629 = stablehlo.reshape %v1628 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1630 = stablehlo.constant dense<0.0> : tensor<32x47040xf32>
    %v1631 = stablehlo.constant dense<6.0> : tensor<32x47040xf32>
    %v1632 = stablehlo.compare GT, %v1362, %v1630 : (tensor<32x47040xf32>, tensor<32x47040xf32>) -> tensor<32x47040xi1>
    %v1633 = stablehlo.compare LT, %v1362, %v1631 : (tensor<32x47040xf32>, tensor<32x47040xf32>) -> tensor<32x47040xi1>
    %v1634 = stablehlo.and %v1632, %v1633 : tensor<32x47040xi1>
    %v1635 = stablehlo.select %v1634, %v1629, %v1630 : tensor<32x47040xi1>, tensor<32x47040xf32>
    %v1636 = stablehlo.reshape %v1635 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1637 = stablehlo.reshape %v1342 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1638 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1639 = stablehlo.constant dense<49.0> : tensor<32x960x7x7xf32>
    %v1640 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v1641 = stablehlo.reduce(%v1637 init: %v1638) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1642 = stablehlo.broadcast_in_dim %v1641, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1643 = stablehlo.divide %v1642, %v1639 : tensor<32x960x7x7xf32>
    %v1644 = stablehlo.subtract %v1637, %v1643 : tensor<32x960x7x7xf32>
    %v1645 = stablehlo.multiply %v1644, %v1644 : tensor<32x960x7x7xf32>
    %v1646 = stablehlo.reduce(%v1645 init: %v1638) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1647 = stablehlo.broadcast_in_dim %v1646, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1648 = stablehlo.divide %v1647, %v1639 : tensor<32x960x7x7xf32>
    %v1649 = stablehlo.add %v1648, %v1640 : tensor<32x960x7x7xf32>
    %v1650 = stablehlo.rsqrt %v1649 : tensor<32x960x7x7xf32>
    %v1651 = stablehlo.multiply %v1644, %v1650 : tensor<32x960x7x7xf32>
    %v1652 = stablehlo.broadcast_in_dim %ge17, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1653 = stablehlo.multiply %v1652, %v1636 : tensor<32x960x7x7xf32>
    %v1654 = stablehlo.reduce(%v1653 init: %v1638) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1655 = stablehlo.broadcast_in_dim %v1654, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1656 = stablehlo.multiply %v1651, %v1653 : tensor<32x960x7x7xf32>
    %v1657 = stablehlo.reduce(%v1656 init: %v1638) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1658 = stablehlo.broadcast_in_dim %v1657, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1659 = stablehlo.multiply %v1653, %v1639 : tensor<32x960x7x7xf32>
    %v1660 = stablehlo.subtract %v1659, %v1655 : tensor<32x960x7x7xf32>
    %v1661 = stablehlo.multiply %v1651, %v1658 : tensor<32x960x7x7xf32>
    %v1662 = stablehlo.subtract %v1660, %v1661 : tensor<32x960x7x7xf32>
    %v1663 = stablehlo.divide %v1650, %v1639 : tensor<32x960x7x7xf32>
    %v1664 = stablehlo.multiply %v1663, %v1662 : tensor<32x960x7x7xf32>
    %v1665 = stablehlo.reshape %v1664 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1666 = stablehlo.reshape %v1665 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1667 = stablehlo.transpose %We17, dims = [1, 0, 2, 3] : (tensor<960x160x1x1xf32>) -> tensor<160x960x1x1xf32>
    %v1668 = stablehlo.reverse %v1667, dims = [2, 3] : tensor<160x960x1x1xf32>
    %v1669 = stablehlo.convolution(%v1666, %v1668)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x960x7x7xf32>, tensor<160x960x1x1xf32>) -> tensor<32x160x7x7xf32>
    %v1670 = stablehlo.reshape %v1669 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v1671 = stablehlo.reshape %v1337 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v1672 = stablehlo.reshape %v1665 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1673 = stablehlo.transpose %v1671, dims = [1, 0, 2, 3] : (tensor<32x160x7x7xf32>) -> tensor<160x32x7x7xf32>
    %v1674 = stablehlo.transpose %v1672, dims = [1, 0, 2, 3] : (tensor<32x960x7x7xf32>) -> tensor<960x32x7x7xf32>
    %v1675 = stablehlo.convolution(%v1673, %v1674)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<160x32x7x7xf32>, tensor<960x32x7x7xf32>) -> tensor<160x960x1x1xf32>
    %v1676 = stablehlo.transpose %v1675, dims = [1, 0, 2, 3] : (tensor<160x960x1x1xf32>) -> tensor<960x160x1x1xf32>
    %v1677 = stablehlo.constant dense<0.3> : tensor<960x160x1x1xf32>
    %v1678 = stablehlo.multiply %v1676, %v1677 : tensor<960x160x1x1xf32>
    %v1679 = stablehlo.subtract %We17, %v1678 : tensor<960x160x1x1xf32>
    %v1680 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1681 = stablehlo.reshape %v1342 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1682 = stablehlo.constant dense<49.0> : tensor<32x960x7x7xf32>
    %v1683 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v1684 = stablehlo.reduce(%v1681 init: %v1680) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1685 = stablehlo.broadcast_in_dim %v1684, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1686 = stablehlo.divide %v1685, %v1682 : tensor<32x960x7x7xf32>
    %v1687 = stablehlo.subtract %v1681, %v1686 : tensor<32x960x7x7xf32>
    %v1688 = stablehlo.multiply %v1687, %v1687 : tensor<32x960x7x7xf32>
    %v1689 = stablehlo.reduce(%v1688 init: %v1680) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1690 = stablehlo.broadcast_in_dim %v1689, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1691 = stablehlo.divide %v1690, %v1682 : tensor<32x960x7x7xf32>
    %v1692 = stablehlo.add %v1691, %v1683 : tensor<32x960x7x7xf32>
    %v1693 = stablehlo.rsqrt %v1692 : tensor<32x960x7x7xf32>
    %v1694 = stablehlo.multiply %v1687, %v1693 : tensor<32x960x7x7xf32>
    %v1695 = stablehlo.reshape %v1635 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1696 = stablehlo.multiply %v1695, %v1694 : tensor<32x960x7x7xf32>
    %v1697 = stablehlo.reduce(%v1696 init: %v1680) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1698 = stablehlo.constant dense<0.3> : tensor<960xf32>
    %v1699 = stablehlo.multiply %v1697, %v1698 : tensor<960xf32>
    %v1700 = stablehlo.subtract %ge17, %v1699 : tensor<960xf32>
    %v1701 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1702 = stablehlo.reshape %v1635 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1703 = stablehlo.reduce(%v1702 init: %v1701) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1704 = stablehlo.constant dense<0.3> : tensor<960xf32>
    %v1705 = stablehlo.multiply %v1703, %v1704 : tensor<960xf32>
    %v1706 = stablehlo.subtract %bte17, %v1705 : tensor<960xf32>
    %v1707 = stablehlo.reshape %v1366 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1708 = stablehlo.reshape %v1625 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1709 = stablehlo.transpose %v1707, dims = [1, 0, 2, 3] : (tensor<32x960x7x7xf32>) -> tensor<960x32x7x7xf32>
    %v1710 = stablehlo.transpose %v1708, dims = [1, 0, 2, 3] : (tensor<32x960x7x7xf32>) -> tensor<960x32x7x7xf32>
    %v1711 = stablehlo.convolution(%v1709, %v1710)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 960 : i64, feature_group_count = 1 : i64} : (tensor<960x32x7x7xf32>, tensor<960x32x7x7xf32>) -> tensor<1x960x3x3xf32>
    %v1712 = stablehlo.reshape %v1711 : (tensor<1x960x3x3xf32>) -> tensor<960x1x3x3xf32>
    %v1713 = stablehlo.constant dense<0.3> : tensor<960x1x3x3xf32>
    %v1714 = stablehlo.multiply %v1712, %v1713 : tensor<960x1x3x3xf32>
    %v1715 = stablehlo.subtract %Wd17, %v1714 : tensor<960x1x3x3xf32>
    %v1716 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1717 = stablehlo.reshape %v1371 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1718 = stablehlo.constant dense<49.0> : tensor<32x960x7x7xf32>
    %v1719 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v1720 = stablehlo.reduce(%v1717 init: %v1716) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1721 = stablehlo.broadcast_in_dim %v1720, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1722 = stablehlo.divide %v1721, %v1718 : tensor<32x960x7x7xf32>
    %v1723 = stablehlo.subtract %v1717, %v1722 : tensor<32x960x7x7xf32>
    %v1724 = stablehlo.multiply %v1723, %v1723 : tensor<32x960x7x7xf32>
    %v1725 = stablehlo.reduce(%v1724 init: %v1716) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1726 = stablehlo.broadcast_in_dim %v1725, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1727 = stablehlo.divide %v1726, %v1718 : tensor<32x960x7x7xf32>
    %v1728 = stablehlo.add %v1727, %v1719 : tensor<32x960x7x7xf32>
    %v1729 = stablehlo.rsqrt %v1728 : tensor<32x960x7x7xf32>
    %v1730 = stablehlo.multiply %v1723, %v1729 : tensor<32x960x7x7xf32>
    %v1731 = stablehlo.reshape %v1595 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1732 = stablehlo.multiply %v1731, %v1730 : tensor<32x960x7x7xf32>
    %v1733 = stablehlo.reduce(%v1732 init: %v1716) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1734 = stablehlo.constant dense<0.3> : tensor<960xf32>
    %v1735 = stablehlo.multiply %v1733, %v1734 : tensor<960xf32>
    %v1736 = stablehlo.subtract %gd17, %v1735 : tensor<960xf32>
    %v1737 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1738 = stablehlo.reshape %v1595 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1739 = stablehlo.reduce(%v1738 init: %v1737) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1740 = stablehlo.constant dense<0.3> : tensor<960xf32>
    %v1741 = stablehlo.multiply %v1739, %v1740 : tensor<960xf32>
    %v1742 = stablehlo.subtract %btd17, %v1741 : tensor<960xf32>
    %v1743 = stablehlo.reshape %v1395 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1744 = stablehlo.reshape %v1584 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1745 = stablehlo.transpose %v1743, dims = [1, 0, 2, 3] : (tensor<32x960x7x7xf32>) -> tensor<960x32x7x7xf32>
    %v1746 = stablehlo.transpose %v1744, dims = [1, 0, 2, 3] : (tensor<32x320x7x7xf32>) -> tensor<320x32x7x7xf32>
    %v1747 = stablehlo.convolution(%v1745, %v1746)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<960x32x7x7xf32>, tensor<320x32x7x7xf32>) -> tensor<960x320x1x1xf32>
    %v1748 = stablehlo.transpose %v1747, dims = [1, 0, 2, 3] : (tensor<960x320x1x1xf32>) -> tensor<320x960x1x1xf32>
    %v1749 = stablehlo.constant dense<0.3> : tensor<320x960x1x1xf32>
    %v1750 = stablehlo.multiply %v1748, %v1749 : tensor<320x960x1x1xf32>
    %v1751 = stablehlo.subtract %Wp17, %v1750 : tensor<320x960x1x1xf32>
    %v1752 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1753 = stablehlo.reshape %v1400 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1754 = stablehlo.constant dense<49.0> : tensor<32x320x7x7xf32>
    %v1755 = stablehlo.constant dense<1.0e-5> : tensor<32x320x7x7xf32>
    %v1756 = stablehlo.reduce(%v1753 init: %v1752) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<32x320xf32>
    %v1757 = stablehlo.broadcast_in_dim %v1756, dims = [0, 1] : (tensor<32x320xf32>) -> tensor<32x320x7x7xf32>
    %v1758 = stablehlo.divide %v1757, %v1754 : tensor<32x320x7x7xf32>
    %v1759 = stablehlo.subtract %v1753, %v1758 : tensor<32x320x7x7xf32>
    %v1760 = stablehlo.multiply %v1759, %v1759 : tensor<32x320x7x7xf32>
    %v1761 = stablehlo.reduce(%v1760 init: %v1752) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<32x320xf32>
    %v1762 = stablehlo.broadcast_in_dim %v1761, dims = [0, 1] : (tensor<32x320xf32>) -> tensor<32x320x7x7xf32>
    %v1763 = stablehlo.divide %v1762, %v1754 : tensor<32x320x7x7xf32>
    %v1764 = stablehlo.add %v1763, %v1755 : tensor<32x320x7x7xf32>
    %v1765 = stablehlo.rsqrt %v1764 : tensor<32x320x7x7xf32>
    %v1766 = stablehlo.multiply %v1759, %v1765 : tensor<32x320x7x7xf32>
    %v1767 = stablehlo.reshape %v1518 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1768 = stablehlo.multiply %v1767, %v1766 : tensor<32x320x7x7xf32>
    %v1769 = stablehlo.reduce(%v1768 init: %v1752) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v1770 = stablehlo.constant dense<0.3> : tensor<320xf32>
    %v1771 = stablehlo.multiply %v1769, %v1770 : tensor<320xf32>
    %v1772 = stablehlo.subtract %gp17, %v1771 : tensor<320xf32>
    %v1773 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1774 = stablehlo.reshape %v1518 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1775 = stablehlo.reduce(%v1774 init: %v1773) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v1776 = stablehlo.constant dense<0.3> : tensor<320xf32>
    %v1777 = stablehlo.multiply %v1775, %v1776 : tensor<320xf32>
    %v1778 = stablehlo.subtract %btp17, %v1777 : tensor<320xf32>
    %v1779 = stablehlo.reshape %v1670 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v1780 = stablehlo.reshape %v1316 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v1781 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1782 = stablehlo.constant dense<49.0> : tensor<32x160x7x7xf32>
    %v1783 = stablehlo.constant dense<1.0e-5> : tensor<32x160x7x7xf32>
    %v1784 = stablehlo.reduce(%v1780 init: %v1781) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<32x160xf32>
    %v1785 = stablehlo.broadcast_in_dim %v1784, dims = [0, 1] : (tensor<32x160xf32>) -> tensor<32x160x7x7xf32>
    %v1786 = stablehlo.divide %v1785, %v1782 : tensor<32x160x7x7xf32>
    %v1787 = stablehlo.subtract %v1780, %v1786 : tensor<32x160x7x7xf32>
    %v1788 = stablehlo.multiply %v1787, %v1787 : tensor<32x160x7x7xf32>
    %v1789 = stablehlo.reduce(%v1788 init: %v1781) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<32x160xf32>
    %v1790 = stablehlo.broadcast_in_dim %v1789, dims = [0, 1] : (tensor<32x160xf32>) -> tensor<32x160x7x7xf32>
    %v1791 = stablehlo.divide %v1790, %v1782 : tensor<32x160x7x7xf32>
    %v1792 = stablehlo.add %v1791, %v1783 : tensor<32x160x7x7xf32>
    %v1793 = stablehlo.rsqrt %v1792 : tensor<32x160x7x7xf32>
    %v1794 = stablehlo.multiply %v1787, %v1793 : tensor<32x160x7x7xf32>
    %v1795 = stablehlo.broadcast_in_dim %gp16, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1796 = stablehlo.multiply %v1795, %v1779 : tensor<32x160x7x7xf32>
    %v1797 = stablehlo.reduce(%v1796 init: %v1781) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<32x160xf32>
    %v1798 = stablehlo.broadcast_in_dim %v1797, dims = [0, 1] : (tensor<32x160xf32>) -> tensor<32x160x7x7xf32>
    %v1799 = stablehlo.multiply %v1794, %v1796 : tensor<32x160x7x7xf32>
    %v1800 = stablehlo.reduce(%v1799 init: %v1781) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<32x160xf32>
    %v1801 = stablehlo.broadcast_in_dim %v1800, dims = [0, 1] : (tensor<32x160xf32>) -> tensor<32x160x7x7xf32>
    %v1802 = stablehlo.multiply %v1796, %v1782 : tensor<32x160x7x7xf32>
    %v1803 = stablehlo.subtract %v1802, %v1798 : tensor<32x160x7x7xf32>
    %v1804 = stablehlo.multiply %v1794, %v1801 : tensor<32x160x7x7xf32>
    %v1805 = stablehlo.subtract %v1803, %v1804 : tensor<32x160x7x7xf32>
    %v1806 = stablehlo.divide %v1793, %v1782 : tensor<32x160x7x7xf32>
    %v1807 = stablehlo.multiply %v1806, %v1805 : tensor<32x160x7x7xf32>
    %v1808 = stablehlo.reshape %v1807 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v1809 = stablehlo.reshape %v1808 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v1810 = stablehlo.transpose %Wp16, dims = [1, 0, 2, 3] : (tensor<160x960x1x1xf32>) -> tensor<960x160x1x1xf32>
    %v1811 = stablehlo.reverse %v1810, dims = [2, 3] : tensor<960x160x1x1xf32>
    %v1812 = stablehlo.convolution(%v1809, %v1811)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x7x7xf32>, tensor<960x160x1x1xf32>) -> tensor<32x960x7x7xf32>
    %v1813 = stablehlo.reshape %v1812 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1814 = stablehlo.constant dense<0.0> : tensor<32x47040xf32>
    %v1815 = stablehlo.constant dense<6.0> : tensor<32x47040xf32>
    %v1816 = stablehlo.compare GT, %v1307, %v1814 : (tensor<32x47040xf32>, tensor<32x47040xf32>) -> tensor<32x47040xi1>
    %v1817 = stablehlo.compare LT, %v1307, %v1815 : (tensor<32x47040xf32>, tensor<32x47040xf32>) -> tensor<32x47040xi1>
    %v1818 = stablehlo.and %v1816, %v1817 : tensor<32x47040xi1>
    %v1819 = stablehlo.select %v1818, %v1813, %v1814 : tensor<32x47040xi1>, tensor<32x47040xf32>
    %v1820 = stablehlo.reshape %v1819 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1821 = stablehlo.reshape %v1287 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1822 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1823 = stablehlo.constant dense<49.0> : tensor<32x960x7x7xf32>
    %v1824 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v1825 = stablehlo.reduce(%v1821 init: %v1822) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1826 = stablehlo.broadcast_in_dim %v1825, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1827 = stablehlo.divide %v1826, %v1823 : tensor<32x960x7x7xf32>
    %v1828 = stablehlo.subtract %v1821, %v1827 : tensor<32x960x7x7xf32>
    %v1829 = stablehlo.multiply %v1828, %v1828 : tensor<32x960x7x7xf32>
    %v1830 = stablehlo.reduce(%v1829 init: %v1822) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1831 = stablehlo.broadcast_in_dim %v1830, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1832 = stablehlo.divide %v1831, %v1823 : tensor<32x960x7x7xf32>
    %v1833 = stablehlo.add %v1832, %v1824 : tensor<32x960x7x7xf32>
    %v1834 = stablehlo.rsqrt %v1833 : tensor<32x960x7x7xf32>
    %v1835 = stablehlo.multiply %v1828, %v1834 : tensor<32x960x7x7xf32>
    %v1836 = stablehlo.broadcast_in_dim %gd16, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1837 = stablehlo.multiply %v1836, %v1820 : tensor<32x960x7x7xf32>
    %v1838 = stablehlo.reduce(%v1837 init: %v1822) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1839 = stablehlo.broadcast_in_dim %v1838, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1840 = stablehlo.multiply %v1835, %v1837 : tensor<32x960x7x7xf32>
    %v1841 = stablehlo.reduce(%v1840 init: %v1822) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1842 = stablehlo.broadcast_in_dim %v1841, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1843 = stablehlo.multiply %v1837, %v1823 : tensor<32x960x7x7xf32>
    %v1844 = stablehlo.subtract %v1843, %v1839 : tensor<32x960x7x7xf32>
    %v1845 = stablehlo.multiply %v1835, %v1842 : tensor<32x960x7x7xf32>
    %v1846 = stablehlo.subtract %v1844, %v1845 : tensor<32x960x7x7xf32>
    %v1847 = stablehlo.divide %v1834, %v1823 : tensor<32x960x7x7xf32>
    %v1848 = stablehlo.multiply %v1847, %v1846 : tensor<32x960x7x7xf32>
    %v1849 = stablehlo.reshape %v1848 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1850 = stablehlo.reshape %v1849 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1851 = stablehlo.reverse %Wd16, dims = [2, 3] : tensor<960x1x3x3xf32>
    %v1852 = stablehlo.convolution(%v1850, %v1851)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 960 : i64} : (tensor<32x960x7x7xf32>, tensor<960x1x3x3xf32>) -> tensor<32x960x7x7xf32>
    %v1853 = stablehlo.reshape %v1852 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1854 = stablehlo.constant dense<0.0> : tensor<32x47040xf32>
    %v1855 = stablehlo.constant dense<6.0> : tensor<32x47040xf32>
    %v1856 = stablehlo.compare GT, %v1278, %v1854 : (tensor<32x47040xf32>, tensor<32x47040xf32>) -> tensor<32x47040xi1>
    %v1857 = stablehlo.compare LT, %v1278, %v1855 : (tensor<32x47040xf32>, tensor<32x47040xf32>) -> tensor<32x47040xi1>
    %v1858 = stablehlo.and %v1856, %v1857 : tensor<32x47040xi1>
    %v1859 = stablehlo.select %v1858, %v1853, %v1854 : tensor<32x47040xi1>, tensor<32x47040xf32>
    %v1860 = stablehlo.reshape %v1859 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1861 = stablehlo.reshape %v1258 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1862 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1863 = stablehlo.constant dense<49.0> : tensor<32x960x7x7xf32>
    %v1864 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v1865 = stablehlo.reduce(%v1861 init: %v1862) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1866 = stablehlo.broadcast_in_dim %v1865, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1867 = stablehlo.divide %v1866, %v1863 : tensor<32x960x7x7xf32>
    %v1868 = stablehlo.subtract %v1861, %v1867 : tensor<32x960x7x7xf32>
    %v1869 = stablehlo.multiply %v1868, %v1868 : tensor<32x960x7x7xf32>
    %v1870 = stablehlo.reduce(%v1869 init: %v1862) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1871 = stablehlo.broadcast_in_dim %v1870, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1872 = stablehlo.divide %v1871, %v1863 : tensor<32x960x7x7xf32>
    %v1873 = stablehlo.add %v1872, %v1864 : tensor<32x960x7x7xf32>
    %v1874 = stablehlo.rsqrt %v1873 : tensor<32x960x7x7xf32>
    %v1875 = stablehlo.multiply %v1868, %v1874 : tensor<32x960x7x7xf32>
    %v1876 = stablehlo.broadcast_in_dim %ge16, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1877 = stablehlo.multiply %v1876, %v1860 : tensor<32x960x7x7xf32>
    %v1878 = stablehlo.reduce(%v1877 init: %v1862) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1879 = stablehlo.broadcast_in_dim %v1878, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1880 = stablehlo.multiply %v1875, %v1877 : tensor<32x960x7x7xf32>
    %v1881 = stablehlo.reduce(%v1880 init: %v1862) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1882 = stablehlo.broadcast_in_dim %v1881, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1883 = stablehlo.multiply %v1877, %v1863 : tensor<32x960x7x7xf32>
    %v1884 = stablehlo.subtract %v1883, %v1879 : tensor<32x960x7x7xf32>
    %v1885 = stablehlo.multiply %v1875, %v1882 : tensor<32x960x7x7xf32>
    %v1886 = stablehlo.subtract %v1884, %v1885 : tensor<32x960x7x7xf32>
    %v1887 = stablehlo.divide %v1874, %v1863 : tensor<32x960x7x7xf32>
    %v1888 = stablehlo.multiply %v1887, %v1886 : tensor<32x960x7x7xf32>
    %v1889 = stablehlo.reshape %v1888 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1890 = stablehlo.reshape %v1889 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1891 = stablehlo.transpose %We16, dims = [1, 0, 2, 3] : (tensor<960x160x1x1xf32>) -> tensor<160x960x1x1xf32>
    %v1892 = stablehlo.reverse %v1891, dims = [2, 3] : tensor<160x960x1x1xf32>
    %v1893 = stablehlo.convolution(%v1890, %v1892)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x960x7x7xf32>, tensor<160x960x1x1xf32>) -> tensor<32x160x7x7xf32>
    %v1894 = stablehlo.reshape %v1893 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v1895 = stablehlo.add %v1894, %v1670 : tensor<32x7840xf32>
    %v1896 = stablehlo.reshape %v1253 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v1897 = stablehlo.reshape %v1889 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1898 = stablehlo.transpose %v1896, dims = [1, 0, 2, 3] : (tensor<32x160x7x7xf32>) -> tensor<160x32x7x7xf32>
    %v1899 = stablehlo.transpose %v1897, dims = [1, 0, 2, 3] : (tensor<32x960x7x7xf32>) -> tensor<960x32x7x7xf32>
    %v1900 = stablehlo.convolution(%v1898, %v1899)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<160x32x7x7xf32>, tensor<960x32x7x7xf32>) -> tensor<160x960x1x1xf32>
    %v1901 = stablehlo.transpose %v1900, dims = [1, 0, 2, 3] : (tensor<160x960x1x1xf32>) -> tensor<960x160x1x1xf32>
    %v1902 = stablehlo.constant dense<0.3> : tensor<960x160x1x1xf32>
    %v1903 = stablehlo.multiply %v1901, %v1902 : tensor<960x160x1x1xf32>
    %v1904 = stablehlo.subtract %We16, %v1903 : tensor<960x160x1x1xf32>
    %v1905 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1906 = stablehlo.reshape %v1258 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1907 = stablehlo.constant dense<49.0> : tensor<32x960x7x7xf32>
    %v1908 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v1909 = stablehlo.reduce(%v1906 init: %v1905) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1910 = stablehlo.broadcast_in_dim %v1909, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1911 = stablehlo.divide %v1910, %v1907 : tensor<32x960x7x7xf32>
    %v1912 = stablehlo.subtract %v1906, %v1911 : tensor<32x960x7x7xf32>
    %v1913 = stablehlo.multiply %v1912, %v1912 : tensor<32x960x7x7xf32>
    %v1914 = stablehlo.reduce(%v1913 init: %v1905) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1915 = stablehlo.broadcast_in_dim %v1914, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1916 = stablehlo.divide %v1915, %v1907 : tensor<32x960x7x7xf32>
    %v1917 = stablehlo.add %v1916, %v1908 : tensor<32x960x7x7xf32>
    %v1918 = stablehlo.rsqrt %v1917 : tensor<32x960x7x7xf32>
    %v1919 = stablehlo.multiply %v1912, %v1918 : tensor<32x960x7x7xf32>
    %v1920 = stablehlo.reshape %v1859 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1921 = stablehlo.multiply %v1920, %v1919 : tensor<32x960x7x7xf32>
    %v1922 = stablehlo.reduce(%v1921 init: %v1905) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1923 = stablehlo.constant dense<0.3> : tensor<960xf32>
    %v1924 = stablehlo.multiply %v1922, %v1923 : tensor<960xf32>
    %v1925 = stablehlo.subtract %ge16, %v1924 : tensor<960xf32>
    %v1926 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1927 = stablehlo.reshape %v1859 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1928 = stablehlo.reduce(%v1927 init: %v1926) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1929 = stablehlo.constant dense<0.3> : tensor<960xf32>
    %v1930 = stablehlo.multiply %v1928, %v1929 : tensor<960xf32>
    %v1931 = stablehlo.subtract %bte16, %v1930 : tensor<960xf32>
    %v1932 = stablehlo.reshape %v1282 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1933 = stablehlo.reshape %v1849 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1934 = stablehlo.transpose %v1932, dims = [1, 0, 2, 3] : (tensor<32x960x7x7xf32>) -> tensor<960x32x7x7xf32>
    %v1935 = stablehlo.transpose %v1933, dims = [1, 0, 2, 3] : (tensor<32x960x7x7xf32>) -> tensor<960x32x7x7xf32>
    %v1936 = stablehlo.convolution(%v1934, %v1935)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 960 : i64, feature_group_count = 1 : i64} : (tensor<960x32x7x7xf32>, tensor<960x32x7x7xf32>) -> tensor<1x960x3x3xf32>
    %v1937 = stablehlo.reshape %v1936 : (tensor<1x960x3x3xf32>) -> tensor<960x1x3x3xf32>
    %v1938 = stablehlo.constant dense<0.3> : tensor<960x1x3x3xf32>
    %v1939 = stablehlo.multiply %v1937, %v1938 : tensor<960x1x3x3xf32>
    %v1940 = stablehlo.subtract %Wd16, %v1939 : tensor<960x1x3x3xf32>
    %v1941 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1942 = stablehlo.reshape %v1287 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1943 = stablehlo.constant dense<49.0> : tensor<32x960x7x7xf32>
    %v1944 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v1945 = stablehlo.reduce(%v1942 init: %v1941) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1946 = stablehlo.broadcast_in_dim %v1945, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1947 = stablehlo.divide %v1946, %v1943 : tensor<32x960x7x7xf32>
    %v1948 = stablehlo.subtract %v1942, %v1947 : tensor<32x960x7x7xf32>
    %v1949 = stablehlo.multiply %v1948, %v1948 : tensor<32x960x7x7xf32>
    %v1950 = stablehlo.reduce(%v1949 init: %v1941) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v1951 = stablehlo.broadcast_in_dim %v1950, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v1952 = stablehlo.divide %v1951, %v1943 : tensor<32x960x7x7xf32>
    %v1953 = stablehlo.add %v1952, %v1944 : tensor<32x960x7x7xf32>
    %v1954 = stablehlo.rsqrt %v1953 : tensor<32x960x7x7xf32>
    %v1955 = stablehlo.multiply %v1948, %v1954 : tensor<32x960x7x7xf32>
    %v1956 = stablehlo.reshape %v1819 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1957 = stablehlo.multiply %v1956, %v1955 : tensor<32x960x7x7xf32>
    %v1958 = stablehlo.reduce(%v1957 init: %v1941) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1959 = stablehlo.constant dense<0.3> : tensor<960xf32>
    %v1960 = stablehlo.multiply %v1958, %v1959 : tensor<960xf32>
    %v1961 = stablehlo.subtract %gd16, %v1960 : tensor<960xf32>
    %v1962 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1963 = stablehlo.reshape %v1819 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1964 = stablehlo.reduce(%v1963 init: %v1962) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1965 = stablehlo.constant dense<0.3> : tensor<960xf32>
    %v1966 = stablehlo.multiply %v1964, %v1965 : tensor<960xf32>
    %v1967 = stablehlo.subtract %btd16, %v1966 : tensor<960xf32>
    %v1968 = stablehlo.reshape %v1311 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1969 = stablehlo.reshape %v1808 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v1970 = stablehlo.transpose %v1968, dims = [1, 0, 2, 3] : (tensor<32x960x7x7xf32>) -> tensor<960x32x7x7xf32>
    %v1971 = stablehlo.transpose %v1969, dims = [1, 0, 2, 3] : (tensor<32x160x7x7xf32>) -> tensor<160x32x7x7xf32>
    %v1972 = stablehlo.convolution(%v1970, %v1971)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<960x32x7x7xf32>, tensor<160x32x7x7xf32>) -> tensor<960x160x1x1xf32>
    %v1973 = stablehlo.transpose %v1972, dims = [1, 0, 2, 3] : (tensor<960x160x1x1xf32>) -> tensor<160x960x1x1xf32>
    %v1974 = stablehlo.constant dense<0.3> : tensor<160x960x1x1xf32>
    %v1975 = stablehlo.multiply %v1973, %v1974 : tensor<160x960x1x1xf32>
    %v1976 = stablehlo.subtract %Wp16, %v1975 : tensor<160x960x1x1xf32>
    %v1977 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1978 = stablehlo.reshape %v1316 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v1979 = stablehlo.constant dense<49.0> : tensor<32x160x7x7xf32>
    %v1980 = stablehlo.constant dense<1.0e-5> : tensor<32x160x7x7xf32>
    %v1981 = stablehlo.reduce(%v1978 init: %v1977) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<32x160xf32>
    %v1982 = stablehlo.broadcast_in_dim %v1981, dims = [0, 1] : (tensor<32x160xf32>) -> tensor<32x160x7x7xf32>
    %v1983 = stablehlo.divide %v1982, %v1979 : tensor<32x160x7x7xf32>
    %v1984 = stablehlo.subtract %v1978, %v1983 : tensor<32x160x7x7xf32>
    %v1985 = stablehlo.multiply %v1984, %v1984 : tensor<32x160x7x7xf32>
    %v1986 = stablehlo.reduce(%v1985 init: %v1977) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<32x160xf32>
    %v1987 = stablehlo.broadcast_in_dim %v1986, dims = [0, 1] : (tensor<32x160xf32>) -> tensor<32x160x7x7xf32>
    %v1988 = stablehlo.divide %v1987, %v1979 : tensor<32x160x7x7xf32>
    %v1989 = stablehlo.add %v1988, %v1980 : tensor<32x160x7x7xf32>
    %v1990 = stablehlo.rsqrt %v1989 : tensor<32x160x7x7xf32>
    %v1991 = stablehlo.multiply %v1984, %v1990 : tensor<32x160x7x7xf32>
    %v1992 = stablehlo.reshape %v1670 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v1993 = stablehlo.multiply %v1992, %v1991 : tensor<32x160x7x7xf32>
    %v1994 = stablehlo.reduce(%v1993 init: %v1977) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v1995 = stablehlo.constant dense<0.3> : tensor<160xf32>
    %v1996 = stablehlo.multiply %v1994, %v1995 : tensor<160xf32>
    %v1997 = stablehlo.subtract %gp16, %v1996 : tensor<160xf32>
    %v1998 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1999 = stablehlo.reshape %v1670 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2000 = stablehlo.reduce(%v1999 init: %v1998) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v2001 = stablehlo.constant dense<0.3> : tensor<160xf32>
    %v2002 = stablehlo.multiply %v2000, %v2001 : tensor<160xf32>
    %v2003 = stablehlo.subtract %btp16, %v2002 : tensor<160xf32>
    %v2004 = stablehlo.reshape %v1895 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2005 = stablehlo.reshape %v1232 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2006 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2007 = stablehlo.constant dense<49.0> : tensor<32x160x7x7xf32>
    %v2008 = stablehlo.constant dense<1.0e-5> : tensor<32x160x7x7xf32>
    %v2009 = stablehlo.reduce(%v2005 init: %v2006) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<32x160xf32>
    %v2010 = stablehlo.broadcast_in_dim %v2009, dims = [0, 1] : (tensor<32x160xf32>) -> tensor<32x160x7x7xf32>
    %v2011 = stablehlo.divide %v2010, %v2007 : tensor<32x160x7x7xf32>
    %v2012 = stablehlo.subtract %v2005, %v2011 : tensor<32x160x7x7xf32>
    %v2013 = stablehlo.multiply %v2012, %v2012 : tensor<32x160x7x7xf32>
    %v2014 = stablehlo.reduce(%v2013 init: %v2006) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<32x160xf32>
    %v2015 = stablehlo.broadcast_in_dim %v2014, dims = [0, 1] : (tensor<32x160xf32>) -> tensor<32x160x7x7xf32>
    %v2016 = stablehlo.divide %v2015, %v2007 : tensor<32x160x7x7xf32>
    %v2017 = stablehlo.add %v2016, %v2008 : tensor<32x160x7x7xf32>
    %v2018 = stablehlo.rsqrt %v2017 : tensor<32x160x7x7xf32>
    %v2019 = stablehlo.multiply %v2012, %v2018 : tensor<32x160x7x7xf32>
    %v2020 = stablehlo.broadcast_in_dim %gp15, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v2021 = stablehlo.multiply %v2020, %v2004 : tensor<32x160x7x7xf32>
    %v2022 = stablehlo.reduce(%v2021 init: %v2006) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<32x160xf32>
    %v2023 = stablehlo.broadcast_in_dim %v2022, dims = [0, 1] : (tensor<32x160xf32>) -> tensor<32x160x7x7xf32>
    %v2024 = stablehlo.multiply %v2019, %v2021 : tensor<32x160x7x7xf32>
    %v2025 = stablehlo.reduce(%v2024 init: %v2006) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<32x160xf32>
    %v2026 = stablehlo.broadcast_in_dim %v2025, dims = [0, 1] : (tensor<32x160xf32>) -> tensor<32x160x7x7xf32>
    %v2027 = stablehlo.multiply %v2021, %v2007 : tensor<32x160x7x7xf32>
    %v2028 = stablehlo.subtract %v2027, %v2023 : tensor<32x160x7x7xf32>
    %v2029 = stablehlo.multiply %v2019, %v2026 : tensor<32x160x7x7xf32>
    %v2030 = stablehlo.subtract %v2028, %v2029 : tensor<32x160x7x7xf32>
    %v2031 = stablehlo.divide %v2018, %v2007 : tensor<32x160x7x7xf32>
    %v2032 = stablehlo.multiply %v2031, %v2030 : tensor<32x160x7x7xf32>
    %v2033 = stablehlo.reshape %v2032 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v2034 = stablehlo.reshape %v2033 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2035 = stablehlo.transpose %Wp15, dims = [1, 0, 2, 3] : (tensor<160x960x1x1xf32>) -> tensor<960x160x1x1xf32>
    %v2036 = stablehlo.reverse %v2035, dims = [2, 3] : tensor<960x160x1x1xf32>
    %v2037 = stablehlo.convolution(%v2034, %v2036)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x7x7xf32>, tensor<960x160x1x1xf32>) -> tensor<32x960x7x7xf32>
    %v2038 = stablehlo.reshape %v2037 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v2039 = stablehlo.constant dense<0.0> : tensor<32x47040xf32>
    %v2040 = stablehlo.constant dense<6.0> : tensor<32x47040xf32>
    %v2041 = stablehlo.compare GT, %v1223, %v2039 : (tensor<32x47040xf32>, tensor<32x47040xf32>) -> tensor<32x47040xi1>
    %v2042 = stablehlo.compare LT, %v1223, %v2040 : (tensor<32x47040xf32>, tensor<32x47040xf32>) -> tensor<32x47040xi1>
    %v2043 = stablehlo.and %v2041, %v2042 : tensor<32x47040xi1>
    %v2044 = stablehlo.select %v2043, %v2038, %v2039 : tensor<32x47040xi1>, tensor<32x47040xf32>
    %v2045 = stablehlo.reshape %v2044 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2046 = stablehlo.reshape %v1203 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2047 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2048 = stablehlo.constant dense<49.0> : tensor<32x960x7x7xf32>
    %v2049 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v2050 = stablehlo.reduce(%v2046 init: %v2047) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v2051 = stablehlo.broadcast_in_dim %v2050, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v2052 = stablehlo.divide %v2051, %v2048 : tensor<32x960x7x7xf32>
    %v2053 = stablehlo.subtract %v2046, %v2052 : tensor<32x960x7x7xf32>
    %v2054 = stablehlo.multiply %v2053, %v2053 : tensor<32x960x7x7xf32>
    %v2055 = stablehlo.reduce(%v2054 init: %v2047) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v2056 = stablehlo.broadcast_in_dim %v2055, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v2057 = stablehlo.divide %v2056, %v2048 : tensor<32x960x7x7xf32>
    %v2058 = stablehlo.add %v2057, %v2049 : tensor<32x960x7x7xf32>
    %v2059 = stablehlo.rsqrt %v2058 : tensor<32x960x7x7xf32>
    %v2060 = stablehlo.multiply %v2053, %v2059 : tensor<32x960x7x7xf32>
    %v2061 = stablehlo.broadcast_in_dim %gd15, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v2062 = stablehlo.multiply %v2061, %v2045 : tensor<32x960x7x7xf32>
    %v2063 = stablehlo.reduce(%v2062 init: %v2047) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v2064 = stablehlo.broadcast_in_dim %v2063, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v2065 = stablehlo.multiply %v2060, %v2062 : tensor<32x960x7x7xf32>
    %v2066 = stablehlo.reduce(%v2065 init: %v2047) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v2067 = stablehlo.broadcast_in_dim %v2066, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v2068 = stablehlo.multiply %v2062, %v2048 : tensor<32x960x7x7xf32>
    %v2069 = stablehlo.subtract %v2068, %v2064 : tensor<32x960x7x7xf32>
    %v2070 = stablehlo.multiply %v2060, %v2067 : tensor<32x960x7x7xf32>
    %v2071 = stablehlo.subtract %v2069, %v2070 : tensor<32x960x7x7xf32>
    %v2072 = stablehlo.divide %v2059, %v2048 : tensor<32x960x7x7xf32>
    %v2073 = stablehlo.multiply %v2072, %v2071 : tensor<32x960x7x7xf32>
    %v2074 = stablehlo.reshape %v2073 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v2075 = stablehlo.reshape %v2074 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2076 = stablehlo.reverse %Wd15, dims = [2, 3] : tensor<960x1x3x3xf32>
    %v2077 = stablehlo.convolution(%v2075, %v2076)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 960 : i64} : (tensor<32x960x7x7xf32>, tensor<960x1x3x3xf32>) -> tensor<32x960x7x7xf32>
    %v2078 = stablehlo.reshape %v2077 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v2079 = stablehlo.constant dense<0.0> : tensor<32x47040xf32>
    %v2080 = stablehlo.constant dense<6.0> : tensor<32x47040xf32>
    %v2081 = stablehlo.compare GT, %v1194, %v2079 : (tensor<32x47040xf32>, tensor<32x47040xf32>) -> tensor<32x47040xi1>
    %v2082 = stablehlo.compare LT, %v1194, %v2080 : (tensor<32x47040xf32>, tensor<32x47040xf32>) -> tensor<32x47040xi1>
    %v2083 = stablehlo.and %v2081, %v2082 : tensor<32x47040xi1>
    %v2084 = stablehlo.select %v2083, %v2078, %v2079 : tensor<32x47040xi1>, tensor<32x47040xf32>
    %v2085 = stablehlo.reshape %v2084 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2086 = stablehlo.reshape %v1174 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2087 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2088 = stablehlo.constant dense<49.0> : tensor<32x960x7x7xf32>
    %v2089 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v2090 = stablehlo.reduce(%v2086 init: %v2087) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v2091 = stablehlo.broadcast_in_dim %v2090, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v2092 = stablehlo.divide %v2091, %v2088 : tensor<32x960x7x7xf32>
    %v2093 = stablehlo.subtract %v2086, %v2092 : tensor<32x960x7x7xf32>
    %v2094 = stablehlo.multiply %v2093, %v2093 : tensor<32x960x7x7xf32>
    %v2095 = stablehlo.reduce(%v2094 init: %v2087) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v2096 = stablehlo.broadcast_in_dim %v2095, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v2097 = stablehlo.divide %v2096, %v2088 : tensor<32x960x7x7xf32>
    %v2098 = stablehlo.add %v2097, %v2089 : tensor<32x960x7x7xf32>
    %v2099 = stablehlo.rsqrt %v2098 : tensor<32x960x7x7xf32>
    %v2100 = stablehlo.multiply %v2093, %v2099 : tensor<32x960x7x7xf32>
    %v2101 = stablehlo.broadcast_in_dim %ge15, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v2102 = stablehlo.multiply %v2101, %v2085 : tensor<32x960x7x7xf32>
    %v2103 = stablehlo.reduce(%v2102 init: %v2087) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v2104 = stablehlo.broadcast_in_dim %v2103, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v2105 = stablehlo.multiply %v2100, %v2102 : tensor<32x960x7x7xf32>
    %v2106 = stablehlo.reduce(%v2105 init: %v2087) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v2107 = stablehlo.broadcast_in_dim %v2106, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v2108 = stablehlo.multiply %v2102, %v2088 : tensor<32x960x7x7xf32>
    %v2109 = stablehlo.subtract %v2108, %v2104 : tensor<32x960x7x7xf32>
    %v2110 = stablehlo.multiply %v2100, %v2107 : tensor<32x960x7x7xf32>
    %v2111 = stablehlo.subtract %v2109, %v2110 : tensor<32x960x7x7xf32>
    %v2112 = stablehlo.divide %v2099, %v2088 : tensor<32x960x7x7xf32>
    %v2113 = stablehlo.multiply %v2112, %v2111 : tensor<32x960x7x7xf32>
    %v2114 = stablehlo.reshape %v2113 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v2115 = stablehlo.reshape %v2114 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2116 = stablehlo.transpose %We15, dims = [1, 0, 2, 3] : (tensor<960x160x1x1xf32>) -> tensor<160x960x1x1xf32>
    %v2117 = stablehlo.reverse %v2116, dims = [2, 3] : tensor<160x960x1x1xf32>
    %v2118 = stablehlo.convolution(%v2115, %v2117)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x960x7x7xf32>, tensor<160x960x1x1xf32>) -> tensor<32x160x7x7xf32>
    %v2119 = stablehlo.reshape %v2118 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v2120 = stablehlo.add %v2119, %v1895 : tensor<32x7840xf32>
    %v2121 = stablehlo.reshape %v1169 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2122 = stablehlo.reshape %v2114 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2123 = stablehlo.transpose %v2121, dims = [1, 0, 2, 3] : (tensor<32x160x7x7xf32>) -> tensor<160x32x7x7xf32>
    %v2124 = stablehlo.transpose %v2122, dims = [1, 0, 2, 3] : (tensor<32x960x7x7xf32>) -> tensor<960x32x7x7xf32>
    %v2125 = stablehlo.convolution(%v2123, %v2124)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<160x32x7x7xf32>, tensor<960x32x7x7xf32>) -> tensor<160x960x1x1xf32>
    %v2126 = stablehlo.transpose %v2125, dims = [1, 0, 2, 3] : (tensor<160x960x1x1xf32>) -> tensor<960x160x1x1xf32>
    %v2127 = stablehlo.constant dense<0.3> : tensor<960x160x1x1xf32>
    %v2128 = stablehlo.multiply %v2126, %v2127 : tensor<960x160x1x1xf32>
    %v2129 = stablehlo.subtract %We15, %v2128 : tensor<960x160x1x1xf32>
    %v2130 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2131 = stablehlo.reshape %v1174 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2132 = stablehlo.constant dense<49.0> : tensor<32x960x7x7xf32>
    %v2133 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v2134 = stablehlo.reduce(%v2131 init: %v2130) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v2135 = stablehlo.broadcast_in_dim %v2134, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v2136 = stablehlo.divide %v2135, %v2132 : tensor<32x960x7x7xf32>
    %v2137 = stablehlo.subtract %v2131, %v2136 : tensor<32x960x7x7xf32>
    %v2138 = stablehlo.multiply %v2137, %v2137 : tensor<32x960x7x7xf32>
    %v2139 = stablehlo.reduce(%v2138 init: %v2130) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v2140 = stablehlo.broadcast_in_dim %v2139, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v2141 = stablehlo.divide %v2140, %v2132 : tensor<32x960x7x7xf32>
    %v2142 = stablehlo.add %v2141, %v2133 : tensor<32x960x7x7xf32>
    %v2143 = stablehlo.rsqrt %v2142 : tensor<32x960x7x7xf32>
    %v2144 = stablehlo.multiply %v2137, %v2143 : tensor<32x960x7x7xf32>
    %v2145 = stablehlo.reshape %v2084 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2146 = stablehlo.multiply %v2145, %v2144 : tensor<32x960x7x7xf32>
    %v2147 = stablehlo.reduce(%v2146 init: %v2130) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v2148 = stablehlo.constant dense<0.3> : tensor<960xf32>
    %v2149 = stablehlo.multiply %v2147, %v2148 : tensor<960xf32>
    %v2150 = stablehlo.subtract %ge15, %v2149 : tensor<960xf32>
    %v2151 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2152 = stablehlo.reshape %v2084 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2153 = stablehlo.reduce(%v2152 init: %v2151) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v2154 = stablehlo.constant dense<0.3> : tensor<960xf32>
    %v2155 = stablehlo.multiply %v2153, %v2154 : tensor<960xf32>
    %v2156 = stablehlo.subtract %bte15, %v2155 : tensor<960xf32>
    %v2157 = stablehlo.reshape %v1198 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2158 = stablehlo.reshape %v2074 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2159 = stablehlo.transpose %v2157, dims = [1, 0, 2, 3] : (tensor<32x960x7x7xf32>) -> tensor<960x32x7x7xf32>
    %v2160 = stablehlo.transpose %v2158, dims = [1, 0, 2, 3] : (tensor<32x960x7x7xf32>) -> tensor<960x32x7x7xf32>
    %v2161 = stablehlo.convolution(%v2159, %v2160)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 960 : i64, feature_group_count = 1 : i64} : (tensor<960x32x7x7xf32>, tensor<960x32x7x7xf32>) -> tensor<1x960x3x3xf32>
    %v2162 = stablehlo.reshape %v2161 : (tensor<1x960x3x3xf32>) -> tensor<960x1x3x3xf32>
    %v2163 = stablehlo.constant dense<0.3> : tensor<960x1x3x3xf32>
    %v2164 = stablehlo.multiply %v2162, %v2163 : tensor<960x1x3x3xf32>
    %v2165 = stablehlo.subtract %Wd15, %v2164 : tensor<960x1x3x3xf32>
    %v2166 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2167 = stablehlo.reshape %v1203 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2168 = stablehlo.constant dense<49.0> : tensor<32x960x7x7xf32>
    %v2169 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v2170 = stablehlo.reduce(%v2167 init: %v2166) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v2171 = stablehlo.broadcast_in_dim %v2170, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v2172 = stablehlo.divide %v2171, %v2168 : tensor<32x960x7x7xf32>
    %v2173 = stablehlo.subtract %v2167, %v2172 : tensor<32x960x7x7xf32>
    %v2174 = stablehlo.multiply %v2173, %v2173 : tensor<32x960x7x7xf32>
    %v2175 = stablehlo.reduce(%v2174 init: %v2166) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v2176 = stablehlo.broadcast_in_dim %v2175, dims = [0, 1] : (tensor<32x960xf32>) -> tensor<32x960x7x7xf32>
    %v2177 = stablehlo.divide %v2176, %v2168 : tensor<32x960x7x7xf32>
    %v2178 = stablehlo.add %v2177, %v2169 : tensor<32x960x7x7xf32>
    %v2179 = stablehlo.rsqrt %v2178 : tensor<32x960x7x7xf32>
    %v2180 = stablehlo.multiply %v2173, %v2179 : tensor<32x960x7x7xf32>
    %v2181 = stablehlo.reshape %v2044 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2182 = stablehlo.multiply %v2181, %v2180 : tensor<32x960x7x7xf32>
    %v2183 = stablehlo.reduce(%v2182 init: %v2166) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v2184 = stablehlo.constant dense<0.3> : tensor<960xf32>
    %v2185 = stablehlo.multiply %v2183, %v2184 : tensor<960xf32>
    %v2186 = stablehlo.subtract %gd15, %v2185 : tensor<960xf32>
    %v2187 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2188 = stablehlo.reshape %v2044 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2189 = stablehlo.reduce(%v2188 init: %v2187) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v2190 = stablehlo.constant dense<0.3> : tensor<960xf32>
    %v2191 = stablehlo.multiply %v2189, %v2190 : tensor<960xf32>
    %v2192 = stablehlo.subtract %btd15, %v2191 : tensor<960xf32>
    %v2193 = stablehlo.reshape %v1227 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2194 = stablehlo.reshape %v2033 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2195 = stablehlo.transpose %v2193, dims = [1, 0, 2, 3] : (tensor<32x960x7x7xf32>) -> tensor<960x32x7x7xf32>
    %v2196 = stablehlo.transpose %v2194, dims = [1, 0, 2, 3] : (tensor<32x160x7x7xf32>) -> tensor<160x32x7x7xf32>
    %v2197 = stablehlo.convolution(%v2195, %v2196)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<960x32x7x7xf32>, tensor<160x32x7x7xf32>) -> tensor<960x160x1x1xf32>
    %v2198 = stablehlo.transpose %v2197, dims = [1, 0, 2, 3] : (tensor<960x160x1x1xf32>) -> tensor<160x960x1x1xf32>
    %v2199 = stablehlo.constant dense<0.3> : tensor<160x960x1x1xf32>
    %v2200 = stablehlo.multiply %v2198, %v2199 : tensor<160x960x1x1xf32>
    %v2201 = stablehlo.subtract %Wp15, %v2200 : tensor<160x960x1x1xf32>
    %v2202 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2203 = stablehlo.reshape %v1232 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2204 = stablehlo.constant dense<49.0> : tensor<32x160x7x7xf32>
    %v2205 = stablehlo.constant dense<1.0e-5> : tensor<32x160x7x7xf32>
    %v2206 = stablehlo.reduce(%v2203 init: %v2202) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<32x160xf32>
    %v2207 = stablehlo.broadcast_in_dim %v2206, dims = [0, 1] : (tensor<32x160xf32>) -> tensor<32x160x7x7xf32>
    %v2208 = stablehlo.divide %v2207, %v2204 : tensor<32x160x7x7xf32>
    %v2209 = stablehlo.subtract %v2203, %v2208 : tensor<32x160x7x7xf32>
    %v2210 = stablehlo.multiply %v2209, %v2209 : tensor<32x160x7x7xf32>
    %v2211 = stablehlo.reduce(%v2210 init: %v2202) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<32x160xf32>
    %v2212 = stablehlo.broadcast_in_dim %v2211, dims = [0, 1] : (tensor<32x160xf32>) -> tensor<32x160x7x7xf32>
    %v2213 = stablehlo.divide %v2212, %v2204 : tensor<32x160x7x7xf32>
    %v2214 = stablehlo.add %v2213, %v2205 : tensor<32x160x7x7xf32>
    %v2215 = stablehlo.rsqrt %v2214 : tensor<32x160x7x7xf32>
    %v2216 = stablehlo.multiply %v2209, %v2215 : tensor<32x160x7x7xf32>
    %v2217 = stablehlo.reshape %v1895 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2218 = stablehlo.multiply %v2217, %v2216 : tensor<32x160x7x7xf32>
    %v2219 = stablehlo.reduce(%v2218 init: %v2202) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v2220 = stablehlo.constant dense<0.3> : tensor<160xf32>
    %v2221 = stablehlo.multiply %v2219, %v2220 : tensor<160xf32>
    %v2222 = stablehlo.subtract %gp15, %v2221 : tensor<160xf32>
    %v2223 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2224 = stablehlo.reshape %v1895 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2225 = stablehlo.reduce(%v2224 init: %v2223) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v2226 = stablehlo.constant dense<0.3> : tensor<160xf32>
    %v2227 = stablehlo.multiply %v2225, %v2226 : tensor<160xf32>
    %v2228 = stablehlo.subtract %btp15, %v2227 : tensor<160xf32>
    %v2229 = stablehlo.reshape %v2120 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2230 = stablehlo.reshape %v1149 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2231 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2232 = stablehlo.constant dense<49.0> : tensor<32x160x7x7xf32>
    %v2233 = stablehlo.constant dense<1.0e-5> : tensor<32x160x7x7xf32>
    %v2234 = stablehlo.reduce(%v2230 init: %v2231) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<32x160xf32>
    %v2235 = stablehlo.broadcast_in_dim %v2234, dims = [0, 1] : (tensor<32x160xf32>) -> tensor<32x160x7x7xf32>
    %v2236 = stablehlo.divide %v2235, %v2232 : tensor<32x160x7x7xf32>
    %v2237 = stablehlo.subtract %v2230, %v2236 : tensor<32x160x7x7xf32>
    %v2238 = stablehlo.multiply %v2237, %v2237 : tensor<32x160x7x7xf32>
    %v2239 = stablehlo.reduce(%v2238 init: %v2231) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<32x160xf32>
    %v2240 = stablehlo.broadcast_in_dim %v2239, dims = [0, 1] : (tensor<32x160xf32>) -> tensor<32x160x7x7xf32>
    %v2241 = stablehlo.divide %v2240, %v2232 : tensor<32x160x7x7xf32>
    %v2242 = stablehlo.add %v2241, %v2233 : tensor<32x160x7x7xf32>
    %v2243 = stablehlo.rsqrt %v2242 : tensor<32x160x7x7xf32>
    %v2244 = stablehlo.multiply %v2237, %v2243 : tensor<32x160x7x7xf32>
    %v2245 = stablehlo.broadcast_in_dim %gp14, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v2246 = stablehlo.multiply %v2245, %v2229 : tensor<32x160x7x7xf32>
    %v2247 = stablehlo.reduce(%v2246 init: %v2231) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<32x160xf32>
    %v2248 = stablehlo.broadcast_in_dim %v2247, dims = [0, 1] : (tensor<32x160xf32>) -> tensor<32x160x7x7xf32>
    %v2249 = stablehlo.multiply %v2244, %v2246 : tensor<32x160x7x7xf32>
    %v2250 = stablehlo.reduce(%v2249 init: %v2231) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<32x160xf32>
    %v2251 = stablehlo.broadcast_in_dim %v2250, dims = [0, 1] : (tensor<32x160xf32>) -> tensor<32x160x7x7xf32>
    %v2252 = stablehlo.multiply %v2246, %v2232 : tensor<32x160x7x7xf32>
    %v2253 = stablehlo.subtract %v2252, %v2248 : tensor<32x160x7x7xf32>
    %v2254 = stablehlo.multiply %v2244, %v2251 : tensor<32x160x7x7xf32>
    %v2255 = stablehlo.subtract %v2253, %v2254 : tensor<32x160x7x7xf32>
    %v2256 = stablehlo.divide %v2243, %v2232 : tensor<32x160x7x7xf32>
    %v2257 = stablehlo.multiply %v2256, %v2255 : tensor<32x160x7x7xf32>
    %v2258 = stablehlo.reshape %v2257 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v2259 = stablehlo.reshape %v2258 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2260 = stablehlo.transpose %Wp14, dims = [1, 0, 2, 3] : (tensor<160x576x1x1xf32>) -> tensor<576x160x1x1xf32>
    %v2261 = stablehlo.reverse %v2260, dims = [2, 3] : tensor<576x160x1x1xf32>
    %v2262 = stablehlo.convolution(%v2259, %v2261)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x7x7xf32>, tensor<576x160x1x1xf32>) -> tensor<32x576x7x7xf32>
    %v2263 = stablehlo.reshape %v2262 : (tensor<32x576x7x7xf32>) -> tensor<32x28224xf32>
    %v2264 = stablehlo.constant dense<0.0> : tensor<32x28224xf32>
    %v2265 = stablehlo.constant dense<6.0> : tensor<32x28224xf32>
    %v2266 = stablehlo.compare GT, %v1140, %v2264 : (tensor<32x28224xf32>, tensor<32x28224xf32>) -> tensor<32x28224xi1>
    %v2267 = stablehlo.compare LT, %v1140, %v2265 : (tensor<32x28224xf32>, tensor<32x28224xf32>) -> tensor<32x28224xi1>
    %v2268 = stablehlo.and %v2266, %v2267 : tensor<32x28224xi1>
    %v2269 = stablehlo.select %v2268, %v2263, %v2264 : tensor<32x28224xi1>, tensor<32x28224xf32>
    %v2270 = stablehlo.reshape %v2269 : (tensor<32x28224xf32>) -> tensor<32x576x7x7xf32>
    %v2271 = stablehlo.reshape %v1120 : (tensor<32x28224xf32>) -> tensor<32x576x7x7xf32>
    %v2272 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2273 = stablehlo.constant dense<49.0> : tensor<32x576x7x7xf32>
    %v2274 = stablehlo.constant dense<1.0e-5> : tensor<32x576x7x7xf32>
    %v2275 = stablehlo.reduce(%v2271 init: %v2272) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x7x7xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2276 = stablehlo.broadcast_in_dim %v2275, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x7x7xf32>
    %v2277 = stablehlo.divide %v2276, %v2273 : tensor<32x576x7x7xf32>
    %v2278 = stablehlo.subtract %v2271, %v2277 : tensor<32x576x7x7xf32>
    %v2279 = stablehlo.multiply %v2278, %v2278 : tensor<32x576x7x7xf32>
    %v2280 = stablehlo.reduce(%v2279 init: %v2272) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x7x7xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2281 = stablehlo.broadcast_in_dim %v2280, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x7x7xf32>
    %v2282 = stablehlo.divide %v2281, %v2273 : tensor<32x576x7x7xf32>
    %v2283 = stablehlo.add %v2282, %v2274 : tensor<32x576x7x7xf32>
    %v2284 = stablehlo.rsqrt %v2283 : tensor<32x576x7x7xf32>
    %v2285 = stablehlo.multiply %v2278, %v2284 : tensor<32x576x7x7xf32>
    %v2286 = stablehlo.broadcast_in_dim %gd14, dims = [1] : (tensor<576xf32>) -> tensor<32x576x7x7xf32>
    %v2287 = stablehlo.multiply %v2286, %v2270 : tensor<32x576x7x7xf32>
    %v2288 = stablehlo.reduce(%v2287 init: %v2272) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x7x7xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2289 = stablehlo.broadcast_in_dim %v2288, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x7x7xf32>
    %v2290 = stablehlo.multiply %v2285, %v2287 : tensor<32x576x7x7xf32>
    %v2291 = stablehlo.reduce(%v2290 init: %v2272) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x7x7xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2292 = stablehlo.broadcast_in_dim %v2291, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x7x7xf32>
    %v2293 = stablehlo.multiply %v2287, %v2273 : tensor<32x576x7x7xf32>
    %v2294 = stablehlo.subtract %v2293, %v2289 : tensor<32x576x7x7xf32>
    %v2295 = stablehlo.multiply %v2285, %v2292 : tensor<32x576x7x7xf32>
    %v2296 = stablehlo.subtract %v2294, %v2295 : tensor<32x576x7x7xf32>
    %v2297 = stablehlo.divide %v2284, %v2273 : tensor<32x576x7x7xf32>
    %v2298 = stablehlo.multiply %v2297, %v2296 : tensor<32x576x7x7xf32>
    %v2299 = stablehlo.reshape %v2298 : (tensor<32x576x7x7xf32>) -> tensor<32x28224xf32>
    %v2300 = stablehlo.reshape %v2299 : (tensor<32x28224xf32>) -> tensor<32x576x7x7xf32>
    %v2301 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2302 = stablehlo.pad %v2300, %v2301, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x576x7x7xf32>, tensor<f32>) -> tensor<32x576x14x14xf32>
    %v2303 = stablehlo.reverse %Wd14, dims = [2, 3] : tensor<576x1x3x3xf32>
    %v2304 = stablehlo.convolution(%v2302, %v2303)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 576 : i64} : (tensor<32x576x14x14xf32>, tensor<576x1x3x3xf32>) -> tensor<32x576x14x14xf32>
    %v2305 = stablehlo.reshape %v2304 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v2306 = stablehlo.constant dense<0.0> : tensor<32x112896xf32>
    %v2307 = stablehlo.constant dense<6.0> : tensor<32x112896xf32>
    %v2308 = stablehlo.compare GT, %v1111, %v2306 : (tensor<32x112896xf32>, tensor<32x112896xf32>) -> tensor<32x112896xi1>
    %v2309 = stablehlo.compare LT, %v1111, %v2307 : (tensor<32x112896xf32>, tensor<32x112896xf32>) -> tensor<32x112896xi1>
    %v2310 = stablehlo.and %v2308, %v2309 : tensor<32x112896xi1>
    %v2311 = stablehlo.select %v2310, %v2305, %v2306 : tensor<32x112896xi1>, tensor<32x112896xf32>
    %v2312 = stablehlo.reshape %v2311 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2313 = stablehlo.reshape %v1091 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2314 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2315 = stablehlo.constant dense<196.0> : tensor<32x576x14x14xf32>
    %v2316 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v2317 = stablehlo.reduce(%v2313 init: %v2314) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2318 = stablehlo.broadcast_in_dim %v2317, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2319 = stablehlo.divide %v2318, %v2315 : tensor<32x576x14x14xf32>
    %v2320 = stablehlo.subtract %v2313, %v2319 : tensor<32x576x14x14xf32>
    %v2321 = stablehlo.multiply %v2320, %v2320 : tensor<32x576x14x14xf32>
    %v2322 = stablehlo.reduce(%v2321 init: %v2314) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2323 = stablehlo.broadcast_in_dim %v2322, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2324 = stablehlo.divide %v2323, %v2315 : tensor<32x576x14x14xf32>
    %v2325 = stablehlo.add %v2324, %v2316 : tensor<32x576x14x14xf32>
    %v2326 = stablehlo.rsqrt %v2325 : tensor<32x576x14x14xf32>
    %v2327 = stablehlo.multiply %v2320, %v2326 : tensor<32x576x14x14xf32>
    %v2328 = stablehlo.broadcast_in_dim %ge14, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2329 = stablehlo.multiply %v2328, %v2312 : tensor<32x576x14x14xf32>
    %v2330 = stablehlo.reduce(%v2329 init: %v2314) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2331 = stablehlo.broadcast_in_dim %v2330, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2332 = stablehlo.multiply %v2327, %v2329 : tensor<32x576x14x14xf32>
    %v2333 = stablehlo.reduce(%v2332 init: %v2314) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2334 = stablehlo.broadcast_in_dim %v2333, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2335 = stablehlo.multiply %v2329, %v2315 : tensor<32x576x14x14xf32>
    %v2336 = stablehlo.subtract %v2335, %v2331 : tensor<32x576x14x14xf32>
    %v2337 = stablehlo.multiply %v2327, %v2334 : tensor<32x576x14x14xf32>
    %v2338 = stablehlo.subtract %v2336, %v2337 : tensor<32x576x14x14xf32>
    %v2339 = stablehlo.divide %v2326, %v2315 : tensor<32x576x14x14xf32>
    %v2340 = stablehlo.multiply %v2339, %v2338 : tensor<32x576x14x14xf32>
    %v2341 = stablehlo.reshape %v2340 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v2342 = stablehlo.reshape %v2341 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2343 = stablehlo.transpose %We14, dims = [1, 0, 2, 3] : (tensor<576x96x1x1xf32>) -> tensor<96x576x1x1xf32>
    %v2344 = stablehlo.reverse %v2343, dims = [2, 3] : tensor<96x576x1x1xf32>
    %v2345 = stablehlo.convolution(%v2342, %v2344)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x576x14x14xf32>, tensor<96x576x1x1xf32>) -> tensor<32x96x14x14xf32>
    %v2346 = stablehlo.reshape %v2345 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v2347 = stablehlo.reshape %v1086 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2348 = stablehlo.reshape %v2341 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2349 = stablehlo.transpose %v2347, dims = [1, 0, 2, 3] : (tensor<32x96x14x14xf32>) -> tensor<96x32x14x14xf32>
    %v2350 = stablehlo.transpose %v2348, dims = [1, 0, 2, 3] : (tensor<32x576x14x14xf32>) -> tensor<576x32x14x14xf32>
    %v2351 = stablehlo.convolution(%v2349, %v2350)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x14x14xf32>, tensor<576x32x14x14xf32>) -> tensor<96x576x1x1xf32>
    %v2352 = stablehlo.transpose %v2351, dims = [1, 0, 2, 3] : (tensor<96x576x1x1xf32>) -> tensor<576x96x1x1xf32>
    %v2353 = stablehlo.constant dense<0.3> : tensor<576x96x1x1xf32>
    %v2354 = stablehlo.multiply %v2352, %v2353 : tensor<576x96x1x1xf32>
    %v2355 = stablehlo.subtract %We14, %v2354 : tensor<576x96x1x1xf32>
    %v2356 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2357 = stablehlo.reshape %v1091 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2358 = stablehlo.constant dense<196.0> : tensor<32x576x14x14xf32>
    %v2359 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v2360 = stablehlo.reduce(%v2357 init: %v2356) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2361 = stablehlo.broadcast_in_dim %v2360, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2362 = stablehlo.divide %v2361, %v2358 : tensor<32x576x14x14xf32>
    %v2363 = stablehlo.subtract %v2357, %v2362 : tensor<32x576x14x14xf32>
    %v2364 = stablehlo.multiply %v2363, %v2363 : tensor<32x576x14x14xf32>
    %v2365 = stablehlo.reduce(%v2364 init: %v2356) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2366 = stablehlo.broadcast_in_dim %v2365, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2367 = stablehlo.divide %v2366, %v2358 : tensor<32x576x14x14xf32>
    %v2368 = stablehlo.add %v2367, %v2359 : tensor<32x576x14x14xf32>
    %v2369 = stablehlo.rsqrt %v2368 : tensor<32x576x14x14xf32>
    %v2370 = stablehlo.multiply %v2363, %v2369 : tensor<32x576x14x14xf32>
    %v2371 = stablehlo.reshape %v2311 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2372 = stablehlo.multiply %v2371, %v2370 : tensor<32x576x14x14xf32>
    %v2373 = stablehlo.reduce(%v2372 init: %v2356) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2374 = stablehlo.constant dense<0.3> : tensor<576xf32>
    %v2375 = stablehlo.multiply %v2373, %v2374 : tensor<576xf32>
    %v2376 = stablehlo.subtract %ge14, %v2375 : tensor<576xf32>
    %v2377 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2378 = stablehlo.reshape %v2311 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2379 = stablehlo.reduce(%v2378 init: %v2377) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2380 = stablehlo.constant dense<0.3> : tensor<576xf32>
    %v2381 = stablehlo.multiply %v2379, %v2380 : tensor<576xf32>
    %v2382 = stablehlo.subtract %bte14, %v2381 : tensor<576xf32>
    %v2383 = stablehlo.reshape %v1115 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2384 = stablehlo.reshape %v2299 : (tensor<32x28224xf32>) -> tensor<32x576x7x7xf32>
    %v2385 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2386 = stablehlo.pad %v2384, %v2385, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x576x7x7xf32>, tensor<f32>) -> tensor<32x576x14x14xf32>
    %v2387 = stablehlo.transpose %v2383, dims = [1, 0, 2, 3] : (tensor<32x576x14x14xf32>) -> tensor<576x32x14x14xf32>
    %v2388 = stablehlo.transpose %v2386, dims = [1, 0, 2, 3] : (tensor<32x576x14x14xf32>) -> tensor<576x32x14x14xf32>
    %v2389 = stablehlo.convolution(%v2387, %v2388)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 576 : i64, feature_group_count = 1 : i64} : (tensor<576x32x14x14xf32>, tensor<576x32x14x14xf32>) -> tensor<1x576x3x3xf32>
    %v2390 = stablehlo.reshape %v2389 : (tensor<1x576x3x3xf32>) -> tensor<576x1x3x3xf32>
    %v2391 = stablehlo.constant dense<0.3> : tensor<576x1x3x3xf32>
    %v2392 = stablehlo.multiply %v2390, %v2391 : tensor<576x1x3x3xf32>
    %v2393 = stablehlo.subtract %Wd14, %v2392 : tensor<576x1x3x3xf32>
    %v2394 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2395 = stablehlo.reshape %v1120 : (tensor<32x28224xf32>) -> tensor<32x576x7x7xf32>
    %v2396 = stablehlo.constant dense<49.0> : tensor<32x576x7x7xf32>
    %v2397 = stablehlo.constant dense<1.0e-5> : tensor<32x576x7x7xf32>
    %v2398 = stablehlo.reduce(%v2395 init: %v2394) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x7x7xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2399 = stablehlo.broadcast_in_dim %v2398, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x7x7xf32>
    %v2400 = stablehlo.divide %v2399, %v2396 : tensor<32x576x7x7xf32>
    %v2401 = stablehlo.subtract %v2395, %v2400 : tensor<32x576x7x7xf32>
    %v2402 = stablehlo.multiply %v2401, %v2401 : tensor<32x576x7x7xf32>
    %v2403 = stablehlo.reduce(%v2402 init: %v2394) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x7x7xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2404 = stablehlo.broadcast_in_dim %v2403, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x7x7xf32>
    %v2405 = stablehlo.divide %v2404, %v2396 : tensor<32x576x7x7xf32>
    %v2406 = stablehlo.add %v2405, %v2397 : tensor<32x576x7x7xf32>
    %v2407 = stablehlo.rsqrt %v2406 : tensor<32x576x7x7xf32>
    %v2408 = stablehlo.multiply %v2401, %v2407 : tensor<32x576x7x7xf32>
    %v2409 = stablehlo.reshape %v2269 : (tensor<32x28224xf32>) -> tensor<32x576x7x7xf32>
    %v2410 = stablehlo.multiply %v2409, %v2408 : tensor<32x576x7x7xf32>
    %v2411 = stablehlo.reduce(%v2410 init: %v2394) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x7x7xf32>, tensor<f32>) -> tensor<576xf32>
    %v2412 = stablehlo.constant dense<0.3> : tensor<576xf32>
    %v2413 = stablehlo.multiply %v2411, %v2412 : tensor<576xf32>
    %v2414 = stablehlo.subtract %gd14, %v2413 : tensor<576xf32>
    %v2415 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2416 = stablehlo.reshape %v2269 : (tensor<32x28224xf32>) -> tensor<32x576x7x7xf32>
    %v2417 = stablehlo.reduce(%v2416 init: %v2415) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x7x7xf32>, tensor<f32>) -> tensor<576xf32>
    %v2418 = stablehlo.constant dense<0.3> : tensor<576xf32>
    %v2419 = stablehlo.multiply %v2417, %v2418 : tensor<576xf32>
    %v2420 = stablehlo.subtract %btd14, %v2419 : tensor<576xf32>
    %v2421 = stablehlo.reshape %v1144 : (tensor<32x28224xf32>) -> tensor<32x576x7x7xf32>
    %v2422 = stablehlo.reshape %v2258 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2423 = stablehlo.transpose %v2421, dims = [1, 0, 2, 3] : (tensor<32x576x7x7xf32>) -> tensor<576x32x7x7xf32>
    %v2424 = stablehlo.transpose %v2422, dims = [1, 0, 2, 3] : (tensor<32x160x7x7xf32>) -> tensor<160x32x7x7xf32>
    %v2425 = stablehlo.convolution(%v2423, %v2424)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<576x32x7x7xf32>, tensor<160x32x7x7xf32>) -> tensor<576x160x1x1xf32>
    %v2426 = stablehlo.transpose %v2425, dims = [1, 0, 2, 3] : (tensor<576x160x1x1xf32>) -> tensor<160x576x1x1xf32>
    %v2427 = stablehlo.constant dense<0.3> : tensor<160x576x1x1xf32>
    %v2428 = stablehlo.multiply %v2426, %v2427 : tensor<160x576x1x1xf32>
    %v2429 = stablehlo.subtract %Wp14, %v2428 : tensor<160x576x1x1xf32>
    %v2430 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2431 = stablehlo.reshape %v1149 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2432 = stablehlo.constant dense<49.0> : tensor<32x160x7x7xf32>
    %v2433 = stablehlo.constant dense<1.0e-5> : tensor<32x160x7x7xf32>
    %v2434 = stablehlo.reduce(%v2431 init: %v2430) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<32x160xf32>
    %v2435 = stablehlo.broadcast_in_dim %v2434, dims = [0, 1] : (tensor<32x160xf32>) -> tensor<32x160x7x7xf32>
    %v2436 = stablehlo.divide %v2435, %v2432 : tensor<32x160x7x7xf32>
    %v2437 = stablehlo.subtract %v2431, %v2436 : tensor<32x160x7x7xf32>
    %v2438 = stablehlo.multiply %v2437, %v2437 : tensor<32x160x7x7xf32>
    %v2439 = stablehlo.reduce(%v2438 init: %v2430) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<32x160xf32>
    %v2440 = stablehlo.broadcast_in_dim %v2439, dims = [0, 1] : (tensor<32x160xf32>) -> tensor<32x160x7x7xf32>
    %v2441 = stablehlo.divide %v2440, %v2432 : tensor<32x160x7x7xf32>
    %v2442 = stablehlo.add %v2441, %v2433 : tensor<32x160x7x7xf32>
    %v2443 = stablehlo.rsqrt %v2442 : tensor<32x160x7x7xf32>
    %v2444 = stablehlo.multiply %v2437, %v2443 : tensor<32x160x7x7xf32>
    %v2445 = stablehlo.reshape %v2120 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2446 = stablehlo.multiply %v2445, %v2444 : tensor<32x160x7x7xf32>
    %v2447 = stablehlo.reduce(%v2446 init: %v2430) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v2448 = stablehlo.constant dense<0.3> : tensor<160xf32>
    %v2449 = stablehlo.multiply %v2447, %v2448 : tensor<160xf32>
    %v2450 = stablehlo.subtract %gp14, %v2449 : tensor<160xf32>
    %v2451 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2452 = stablehlo.reshape %v2120 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v2453 = stablehlo.reduce(%v2452 init: %v2451) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v2454 = stablehlo.constant dense<0.3> : tensor<160xf32>
    %v2455 = stablehlo.multiply %v2453, %v2454 : tensor<160xf32>
    %v2456 = stablehlo.subtract %btp14, %v2455 : tensor<160xf32>
    %v2457 = stablehlo.reshape %v2346 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2458 = stablehlo.reshape %v1065 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2459 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2460 = stablehlo.constant dense<196.0> : tensor<32x96x14x14xf32>
    %v2461 = stablehlo.constant dense<1.0e-5> : tensor<32x96x14x14xf32>
    %v2462 = stablehlo.reduce(%v2458 init: %v2459) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v2463 = stablehlo.broadcast_in_dim %v2462, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x14x14xf32>
    %v2464 = stablehlo.divide %v2463, %v2460 : tensor<32x96x14x14xf32>
    %v2465 = stablehlo.subtract %v2458, %v2464 : tensor<32x96x14x14xf32>
    %v2466 = stablehlo.multiply %v2465, %v2465 : tensor<32x96x14x14xf32>
    %v2467 = stablehlo.reduce(%v2466 init: %v2459) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v2468 = stablehlo.broadcast_in_dim %v2467, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x14x14xf32>
    %v2469 = stablehlo.divide %v2468, %v2460 : tensor<32x96x14x14xf32>
    %v2470 = stablehlo.add %v2469, %v2461 : tensor<32x96x14x14xf32>
    %v2471 = stablehlo.rsqrt %v2470 : tensor<32x96x14x14xf32>
    %v2472 = stablehlo.multiply %v2465, %v2471 : tensor<32x96x14x14xf32>
    %v2473 = stablehlo.broadcast_in_dim %gp13, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v2474 = stablehlo.multiply %v2473, %v2457 : tensor<32x96x14x14xf32>
    %v2475 = stablehlo.reduce(%v2474 init: %v2459) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v2476 = stablehlo.broadcast_in_dim %v2475, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x14x14xf32>
    %v2477 = stablehlo.multiply %v2472, %v2474 : tensor<32x96x14x14xf32>
    %v2478 = stablehlo.reduce(%v2477 init: %v2459) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v2479 = stablehlo.broadcast_in_dim %v2478, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x14x14xf32>
    %v2480 = stablehlo.multiply %v2474, %v2460 : tensor<32x96x14x14xf32>
    %v2481 = stablehlo.subtract %v2480, %v2476 : tensor<32x96x14x14xf32>
    %v2482 = stablehlo.multiply %v2472, %v2479 : tensor<32x96x14x14xf32>
    %v2483 = stablehlo.subtract %v2481, %v2482 : tensor<32x96x14x14xf32>
    %v2484 = stablehlo.divide %v2471, %v2460 : tensor<32x96x14x14xf32>
    %v2485 = stablehlo.multiply %v2484, %v2483 : tensor<32x96x14x14xf32>
    %v2486 = stablehlo.reshape %v2485 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v2487 = stablehlo.reshape %v2486 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2488 = stablehlo.transpose %Wp13, dims = [1, 0, 2, 3] : (tensor<96x576x1x1xf32>) -> tensor<576x96x1x1xf32>
    %v2489 = stablehlo.reverse %v2488, dims = [2, 3] : tensor<576x96x1x1xf32>
    %v2490 = stablehlo.convolution(%v2487, %v2489)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x14x14xf32>, tensor<576x96x1x1xf32>) -> tensor<32x576x14x14xf32>
    %v2491 = stablehlo.reshape %v2490 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v2492 = stablehlo.constant dense<0.0> : tensor<32x112896xf32>
    %v2493 = stablehlo.constant dense<6.0> : tensor<32x112896xf32>
    %v2494 = stablehlo.compare GT, %v1056, %v2492 : (tensor<32x112896xf32>, tensor<32x112896xf32>) -> tensor<32x112896xi1>
    %v2495 = stablehlo.compare LT, %v1056, %v2493 : (tensor<32x112896xf32>, tensor<32x112896xf32>) -> tensor<32x112896xi1>
    %v2496 = stablehlo.and %v2494, %v2495 : tensor<32x112896xi1>
    %v2497 = stablehlo.select %v2496, %v2491, %v2492 : tensor<32x112896xi1>, tensor<32x112896xf32>
    %v2498 = stablehlo.reshape %v2497 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2499 = stablehlo.reshape %v1036 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2500 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2501 = stablehlo.constant dense<196.0> : tensor<32x576x14x14xf32>
    %v2502 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v2503 = stablehlo.reduce(%v2499 init: %v2500) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2504 = stablehlo.broadcast_in_dim %v2503, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2505 = stablehlo.divide %v2504, %v2501 : tensor<32x576x14x14xf32>
    %v2506 = stablehlo.subtract %v2499, %v2505 : tensor<32x576x14x14xf32>
    %v2507 = stablehlo.multiply %v2506, %v2506 : tensor<32x576x14x14xf32>
    %v2508 = stablehlo.reduce(%v2507 init: %v2500) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2509 = stablehlo.broadcast_in_dim %v2508, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2510 = stablehlo.divide %v2509, %v2501 : tensor<32x576x14x14xf32>
    %v2511 = stablehlo.add %v2510, %v2502 : tensor<32x576x14x14xf32>
    %v2512 = stablehlo.rsqrt %v2511 : tensor<32x576x14x14xf32>
    %v2513 = stablehlo.multiply %v2506, %v2512 : tensor<32x576x14x14xf32>
    %v2514 = stablehlo.broadcast_in_dim %gd13, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2515 = stablehlo.multiply %v2514, %v2498 : tensor<32x576x14x14xf32>
    %v2516 = stablehlo.reduce(%v2515 init: %v2500) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2517 = stablehlo.broadcast_in_dim %v2516, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2518 = stablehlo.multiply %v2513, %v2515 : tensor<32x576x14x14xf32>
    %v2519 = stablehlo.reduce(%v2518 init: %v2500) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2520 = stablehlo.broadcast_in_dim %v2519, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2521 = stablehlo.multiply %v2515, %v2501 : tensor<32x576x14x14xf32>
    %v2522 = stablehlo.subtract %v2521, %v2517 : tensor<32x576x14x14xf32>
    %v2523 = stablehlo.multiply %v2513, %v2520 : tensor<32x576x14x14xf32>
    %v2524 = stablehlo.subtract %v2522, %v2523 : tensor<32x576x14x14xf32>
    %v2525 = stablehlo.divide %v2512, %v2501 : tensor<32x576x14x14xf32>
    %v2526 = stablehlo.multiply %v2525, %v2524 : tensor<32x576x14x14xf32>
    %v2527 = stablehlo.reshape %v2526 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v2528 = stablehlo.reshape %v2527 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2529 = stablehlo.reverse %Wd13, dims = [2, 3] : tensor<576x1x3x3xf32>
    %v2530 = stablehlo.convolution(%v2528, %v2529)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 576 : i64} : (tensor<32x576x14x14xf32>, tensor<576x1x3x3xf32>) -> tensor<32x576x14x14xf32>
    %v2531 = stablehlo.reshape %v2530 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v2532 = stablehlo.constant dense<0.0> : tensor<32x112896xf32>
    %v2533 = stablehlo.constant dense<6.0> : tensor<32x112896xf32>
    %v2534 = stablehlo.compare GT, %v1027, %v2532 : (tensor<32x112896xf32>, tensor<32x112896xf32>) -> tensor<32x112896xi1>
    %v2535 = stablehlo.compare LT, %v1027, %v2533 : (tensor<32x112896xf32>, tensor<32x112896xf32>) -> tensor<32x112896xi1>
    %v2536 = stablehlo.and %v2534, %v2535 : tensor<32x112896xi1>
    %v2537 = stablehlo.select %v2536, %v2531, %v2532 : tensor<32x112896xi1>, tensor<32x112896xf32>
    %v2538 = stablehlo.reshape %v2537 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2539 = stablehlo.reshape %v1007 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2540 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2541 = stablehlo.constant dense<196.0> : tensor<32x576x14x14xf32>
    %v2542 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v2543 = stablehlo.reduce(%v2539 init: %v2540) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2544 = stablehlo.broadcast_in_dim %v2543, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2545 = stablehlo.divide %v2544, %v2541 : tensor<32x576x14x14xf32>
    %v2546 = stablehlo.subtract %v2539, %v2545 : tensor<32x576x14x14xf32>
    %v2547 = stablehlo.multiply %v2546, %v2546 : tensor<32x576x14x14xf32>
    %v2548 = stablehlo.reduce(%v2547 init: %v2540) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2549 = stablehlo.broadcast_in_dim %v2548, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2550 = stablehlo.divide %v2549, %v2541 : tensor<32x576x14x14xf32>
    %v2551 = stablehlo.add %v2550, %v2542 : tensor<32x576x14x14xf32>
    %v2552 = stablehlo.rsqrt %v2551 : tensor<32x576x14x14xf32>
    %v2553 = stablehlo.multiply %v2546, %v2552 : tensor<32x576x14x14xf32>
    %v2554 = stablehlo.broadcast_in_dim %ge13, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2555 = stablehlo.multiply %v2554, %v2538 : tensor<32x576x14x14xf32>
    %v2556 = stablehlo.reduce(%v2555 init: %v2540) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2557 = stablehlo.broadcast_in_dim %v2556, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2558 = stablehlo.multiply %v2553, %v2555 : tensor<32x576x14x14xf32>
    %v2559 = stablehlo.reduce(%v2558 init: %v2540) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2560 = stablehlo.broadcast_in_dim %v2559, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2561 = stablehlo.multiply %v2555, %v2541 : tensor<32x576x14x14xf32>
    %v2562 = stablehlo.subtract %v2561, %v2557 : tensor<32x576x14x14xf32>
    %v2563 = stablehlo.multiply %v2553, %v2560 : tensor<32x576x14x14xf32>
    %v2564 = stablehlo.subtract %v2562, %v2563 : tensor<32x576x14x14xf32>
    %v2565 = stablehlo.divide %v2552, %v2541 : tensor<32x576x14x14xf32>
    %v2566 = stablehlo.multiply %v2565, %v2564 : tensor<32x576x14x14xf32>
    %v2567 = stablehlo.reshape %v2566 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v2568 = stablehlo.reshape %v2567 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2569 = stablehlo.transpose %We13, dims = [1, 0, 2, 3] : (tensor<576x96x1x1xf32>) -> tensor<96x576x1x1xf32>
    %v2570 = stablehlo.reverse %v2569, dims = [2, 3] : tensor<96x576x1x1xf32>
    %v2571 = stablehlo.convolution(%v2568, %v2570)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x576x14x14xf32>, tensor<96x576x1x1xf32>) -> tensor<32x96x14x14xf32>
    %v2572 = stablehlo.reshape %v2571 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v2573 = stablehlo.add %v2572, %v2346 : tensor<32x18816xf32>
    %v2574 = stablehlo.reshape %v1002 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2575 = stablehlo.reshape %v2567 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2576 = stablehlo.transpose %v2574, dims = [1, 0, 2, 3] : (tensor<32x96x14x14xf32>) -> tensor<96x32x14x14xf32>
    %v2577 = stablehlo.transpose %v2575, dims = [1, 0, 2, 3] : (tensor<32x576x14x14xf32>) -> tensor<576x32x14x14xf32>
    %v2578 = stablehlo.convolution(%v2576, %v2577)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x14x14xf32>, tensor<576x32x14x14xf32>) -> tensor<96x576x1x1xf32>
    %v2579 = stablehlo.transpose %v2578, dims = [1, 0, 2, 3] : (tensor<96x576x1x1xf32>) -> tensor<576x96x1x1xf32>
    %v2580 = stablehlo.constant dense<0.3> : tensor<576x96x1x1xf32>
    %v2581 = stablehlo.multiply %v2579, %v2580 : tensor<576x96x1x1xf32>
    %v2582 = stablehlo.subtract %We13, %v2581 : tensor<576x96x1x1xf32>
    %v2583 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2584 = stablehlo.reshape %v1007 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2585 = stablehlo.constant dense<196.0> : tensor<32x576x14x14xf32>
    %v2586 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v2587 = stablehlo.reduce(%v2584 init: %v2583) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2588 = stablehlo.broadcast_in_dim %v2587, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2589 = stablehlo.divide %v2588, %v2585 : tensor<32x576x14x14xf32>
    %v2590 = stablehlo.subtract %v2584, %v2589 : tensor<32x576x14x14xf32>
    %v2591 = stablehlo.multiply %v2590, %v2590 : tensor<32x576x14x14xf32>
    %v2592 = stablehlo.reduce(%v2591 init: %v2583) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2593 = stablehlo.broadcast_in_dim %v2592, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2594 = stablehlo.divide %v2593, %v2585 : tensor<32x576x14x14xf32>
    %v2595 = stablehlo.add %v2594, %v2586 : tensor<32x576x14x14xf32>
    %v2596 = stablehlo.rsqrt %v2595 : tensor<32x576x14x14xf32>
    %v2597 = stablehlo.multiply %v2590, %v2596 : tensor<32x576x14x14xf32>
    %v2598 = stablehlo.reshape %v2537 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2599 = stablehlo.multiply %v2598, %v2597 : tensor<32x576x14x14xf32>
    %v2600 = stablehlo.reduce(%v2599 init: %v2583) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2601 = stablehlo.constant dense<0.3> : tensor<576xf32>
    %v2602 = stablehlo.multiply %v2600, %v2601 : tensor<576xf32>
    %v2603 = stablehlo.subtract %ge13, %v2602 : tensor<576xf32>
    %v2604 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2605 = stablehlo.reshape %v2537 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2606 = stablehlo.reduce(%v2605 init: %v2604) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2607 = stablehlo.constant dense<0.3> : tensor<576xf32>
    %v2608 = stablehlo.multiply %v2606, %v2607 : tensor<576xf32>
    %v2609 = stablehlo.subtract %bte13, %v2608 : tensor<576xf32>
    %v2610 = stablehlo.reshape %v1031 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2611 = stablehlo.reshape %v2527 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2612 = stablehlo.transpose %v2610, dims = [1, 0, 2, 3] : (tensor<32x576x14x14xf32>) -> tensor<576x32x14x14xf32>
    %v2613 = stablehlo.transpose %v2611, dims = [1, 0, 2, 3] : (tensor<32x576x14x14xf32>) -> tensor<576x32x14x14xf32>
    %v2614 = stablehlo.convolution(%v2612, %v2613)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 576 : i64, feature_group_count = 1 : i64} : (tensor<576x32x14x14xf32>, tensor<576x32x14x14xf32>) -> tensor<1x576x3x3xf32>
    %v2615 = stablehlo.reshape %v2614 : (tensor<1x576x3x3xf32>) -> tensor<576x1x3x3xf32>
    %v2616 = stablehlo.constant dense<0.3> : tensor<576x1x3x3xf32>
    %v2617 = stablehlo.multiply %v2615, %v2616 : tensor<576x1x3x3xf32>
    %v2618 = stablehlo.subtract %Wd13, %v2617 : tensor<576x1x3x3xf32>
    %v2619 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2620 = stablehlo.reshape %v1036 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2621 = stablehlo.constant dense<196.0> : tensor<32x576x14x14xf32>
    %v2622 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v2623 = stablehlo.reduce(%v2620 init: %v2619) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2624 = stablehlo.broadcast_in_dim %v2623, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2625 = stablehlo.divide %v2624, %v2621 : tensor<32x576x14x14xf32>
    %v2626 = stablehlo.subtract %v2620, %v2625 : tensor<32x576x14x14xf32>
    %v2627 = stablehlo.multiply %v2626, %v2626 : tensor<32x576x14x14xf32>
    %v2628 = stablehlo.reduce(%v2627 init: %v2619) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2629 = stablehlo.broadcast_in_dim %v2628, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2630 = stablehlo.divide %v2629, %v2621 : tensor<32x576x14x14xf32>
    %v2631 = stablehlo.add %v2630, %v2622 : tensor<32x576x14x14xf32>
    %v2632 = stablehlo.rsqrt %v2631 : tensor<32x576x14x14xf32>
    %v2633 = stablehlo.multiply %v2626, %v2632 : tensor<32x576x14x14xf32>
    %v2634 = stablehlo.reshape %v2497 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2635 = stablehlo.multiply %v2634, %v2633 : tensor<32x576x14x14xf32>
    %v2636 = stablehlo.reduce(%v2635 init: %v2619) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2637 = stablehlo.constant dense<0.3> : tensor<576xf32>
    %v2638 = stablehlo.multiply %v2636, %v2637 : tensor<576xf32>
    %v2639 = stablehlo.subtract %gd13, %v2638 : tensor<576xf32>
    %v2640 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2641 = stablehlo.reshape %v2497 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2642 = stablehlo.reduce(%v2641 init: %v2640) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2643 = stablehlo.constant dense<0.3> : tensor<576xf32>
    %v2644 = stablehlo.multiply %v2642, %v2643 : tensor<576xf32>
    %v2645 = stablehlo.subtract %btd13, %v2644 : tensor<576xf32>
    %v2646 = stablehlo.reshape %v1060 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2647 = stablehlo.reshape %v2486 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2648 = stablehlo.transpose %v2646, dims = [1, 0, 2, 3] : (tensor<32x576x14x14xf32>) -> tensor<576x32x14x14xf32>
    %v2649 = stablehlo.transpose %v2647, dims = [1, 0, 2, 3] : (tensor<32x96x14x14xf32>) -> tensor<96x32x14x14xf32>
    %v2650 = stablehlo.convolution(%v2648, %v2649)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<576x32x14x14xf32>, tensor<96x32x14x14xf32>) -> tensor<576x96x1x1xf32>
    %v2651 = stablehlo.transpose %v2650, dims = [1, 0, 2, 3] : (tensor<576x96x1x1xf32>) -> tensor<96x576x1x1xf32>
    %v2652 = stablehlo.constant dense<0.3> : tensor<96x576x1x1xf32>
    %v2653 = stablehlo.multiply %v2651, %v2652 : tensor<96x576x1x1xf32>
    %v2654 = stablehlo.subtract %Wp13, %v2653 : tensor<96x576x1x1xf32>
    %v2655 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2656 = stablehlo.reshape %v1065 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2657 = stablehlo.constant dense<196.0> : tensor<32x96x14x14xf32>
    %v2658 = stablehlo.constant dense<1.0e-5> : tensor<32x96x14x14xf32>
    %v2659 = stablehlo.reduce(%v2656 init: %v2655) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v2660 = stablehlo.broadcast_in_dim %v2659, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x14x14xf32>
    %v2661 = stablehlo.divide %v2660, %v2657 : tensor<32x96x14x14xf32>
    %v2662 = stablehlo.subtract %v2656, %v2661 : tensor<32x96x14x14xf32>
    %v2663 = stablehlo.multiply %v2662, %v2662 : tensor<32x96x14x14xf32>
    %v2664 = stablehlo.reduce(%v2663 init: %v2655) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v2665 = stablehlo.broadcast_in_dim %v2664, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x14x14xf32>
    %v2666 = stablehlo.divide %v2665, %v2657 : tensor<32x96x14x14xf32>
    %v2667 = stablehlo.add %v2666, %v2658 : tensor<32x96x14x14xf32>
    %v2668 = stablehlo.rsqrt %v2667 : tensor<32x96x14x14xf32>
    %v2669 = stablehlo.multiply %v2662, %v2668 : tensor<32x96x14x14xf32>
    %v2670 = stablehlo.reshape %v2346 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2671 = stablehlo.multiply %v2670, %v2669 : tensor<32x96x14x14xf32>
    %v2672 = stablehlo.reduce(%v2671 init: %v2655) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v2673 = stablehlo.constant dense<0.3> : tensor<96xf32>
    %v2674 = stablehlo.multiply %v2672, %v2673 : tensor<96xf32>
    %v2675 = stablehlo.subtract %gp13, %v2674 : tensor<96xf32>
    %v2676 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2677 = stablehlo.reshape %v2346 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2678 = stablehlo.reduce(%v2677 init: %v2676) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v2679 = stablehlo.constant dense<0.3> : tensor<96xf32>
    %v2680 = stablehlo.multiply %v2678, %v2679 : tensor<96xf32>
    %v2681 = stablehlo.subtract %btp13, %v2680 : tensor<96xf32>
    %v2682 = stablehlo.reshape %v2573 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2683 = stablehlo.reshape %v981 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2684 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2685 = stablehlo.constant dense<196.0> : tensor<32x96x14x14xf32>
    %v2686 = stablehlo.constant dense<1.0e-5> : tensor<32x96x14x14xf32>
    %v2687 = stablehlo.reduce(%v2683 init: %v2684) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v2688 = stablehlo.broadcast_in_dim %v2687, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x14x14xf32>
    %v2689 = stablehlo.divide %v2688, %v2685 : tensor<32x96x14x14xf32>
    %v2690 = stablehlo.subtract %v2683, %v2689 : tensor<32x96x14x14xf32>
    %v2691 = stablehlo.multiply %v2690, %v2690 : tensor<32x96x14x14xf32>
    %v2692 = stablehlo.reduce(%v2691 init: %v2684) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v2693 = stablehlo.broadcast_in_dim %v2692, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x14x14xf32>
    %v2694 = stablehlo.divide %v2693, %v2685 : tensor<32x96x14x14xf32>
    %v2695 = stablehlo.add %v2694, %v2686 : tensor<32x96x14x14xf32>
    %v2696 = stablehlo.rsqrt %v2695 : tensor<32x96x14x14xf32>
    %v2697 = stablehlo.multiply %v2690, %v2696 : tensor<32x96x14x14xf32>
    %v2698 = stablehlo.broadcast_in_dim %gp12, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v2699 = stablehlo.multiply %v2698, %v2682 : tensor<32x96x14x14xf32>
    %v2700 = stablehlo.reduce(%v2699 init: %v2684) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v2701 = stablehlo.broadcast_in_dim %v2700, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x14x14xf32>
    %v2702 = stablehlo.multiply %v2697, %v2699 : tensor<32x96x14x14xf32>
    %v2703 = stablehlo.reduce(%v2702 init: %v2684) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v2704 = stablehlo.broadcast_in_dim %v2703, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x14x14xf32>
    %v2705 = stablehlo.multiply %v2699, %v2685 : tensor<32x96x14x14xf32>
    %v2706 = stablehlo.subtract %v2705, %v2701 : tensor<32x96x14x14xf32>
    %v2707 = stablehlo.multiply %v2697, %v2704 : tensor<32x96x14x14xf32>
    %v2708 = stablehlo.subtract %v2706, %v2707 : tensor<32x96x14x14xf32>
    %v2709 = stablehlo.divide %v2696, %v2685 : tensor<32x96x14x14xf32>
    %v2710 = stablehlo.multiply %v2709, %v2708 : tensor<32x96x14x14xf32>
    %v2711 = stablehlo.reshape %v2710 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v2712 = stablehlo.reshape %v2711 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2713 = stablehlo.transpose %Wp12, dims = [1, 0, 2, 3] : (tensor<96x576x1x1xf32>) -> tensor<576x96x1x1xf32>
    %v2714 = stablehlo.reverse %v2713, dims = [2, 3] : tensor<576x96x1x1xf32>
    %v2715 = stablehlo.convolution(%v2712, %v2714)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x14x14xf32>, tensor<576x96x1x1xf32>) -> tensor<32x576x14x14xf32>
    %v2716 = stablehlo.reshape %v2715 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v2717 = stablehlo.constant dense<0.0> : tensor<32x112896xf32>
    %v2718 = stablehlo.constant dense<6.0> : tensor<32x112896xf32>
    %v2719 = stablehlo.compare GT, %v972, %v2717 : (tensor<32x112896xf32>, tensor<32x112896xf32>) -> tensor<32x112896xi1>
    %v2720 = stablehlo.compare LT, %v972, %v2718 : (tensor<32x112896xf32>, tensor<32x112896xf32>) -> tensor<32x112896xi1>
    %v2721 = stablehlo.and %v2719, %v2720 : tensor<32x112896xi1>
    %v2722 = stablehlo.select %v2721, %v2716, %v2717 : tensor<32x112896xi1>, tensor<32x112896xf32>
    %v2723 = stablehlo.reshape %v2722 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2724 = stablehlo.reshape %v952 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2725 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2726 = stablehlo.constant dense<196.0> : tensor<32x576x14x14xf32>
    %v2727 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v2728 = stablehlo.reduce(%v2724 init: %v2725) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2729 = stablehlo.broadcast_in_dim %v2728, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2730 = stablehlo.divide %v2729, %v2726 : tensor<32x576x14x14xf32>
    %v2731 = stablehlo.subtract %v2724, %v2730 : tensor<32x576x14x14xf32>
    %v2732 = stablehlo.multiply %v2731, %v2731 : tensor<32x576x14x14xf32>
    %v2733 = stablehlo.reduce(%v2732 init: %v2725) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2734 = stablehlo.broadcast_in_dim %v2733, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2735 = stablehlo.divide %v2734, %v2726 : tensor<32x576x14x14xf32>
    %v2736 = stablehlo.add %v2735, %v2727 : tensor<32x576x14x14xf32>
    %v2737 = stablehlo.rsqrt %v2736 : tensor<32x576x14x14xf32>
    %v2738 = stablehlo.multiply %v2731, %v2737 : tensor<32x576x14x14xf32>
    %v2739 = stablehlo.broadcast_in_dim %gd12, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2740 = stablehlo.multiply %v2739, %v2723 : tensor<32x576x14x14xf32>
    %v2741 = stablehlo.reduce(%v2740 init: %v2725) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2742 = stablehlo.broadcast_in_dim %v2741, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2743 = stablehlo.multiply %v2738, %v2740 : tensor<32x576x14x14xf32>
    %v2744 = stablehlo.reduce(%v2743 init: %v2725) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2745 = stablehlo.broadcast_in_dim %v2744, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2746 = stablehlo.multiply %v2740, %v2726 : tensor<32x576x14x14xf32>
    %v2747 = stablehlo.subtract %v2746, %v2742 : tensor<32x576x14x14xf32>
    %v2748 = stablehlo.multiply %v2738, %v2745 : tensor<32x576x14x14xf32>
    %v2749 = stablehlo.subtract %v2747, %v2748 : tensor<32x576x14x14xf32>
    %v2750 = stablehlo.divide %v2737, %v2726 : tensor<32x576x14x14xf32>
    %v2751 = stablehlo.multiply %v2750, %v2749 : tensor<32x576x14x14xf32>
    %v2752 = stablehlo.reshape %v2751 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v2753 = stablehlo.reshape %v2752 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2754 = stablehlo.reverse %Wd12, dims = [2, 3] : tensor<576x1x3x3xf32>
    %v2755 = stablehlo.convolution(%v2753, %v2754)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 576 : i64} : (tensor<32x576x14x14xf32>, tensor<576x1x3x3xf32>) -> tensor<32x576x14x14xf32>
    %v2756 = stablehlo.reshape %v2755 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v2757 = stablehlo.constant dense<0.0> : tensor<32x112896xf32>
    %v2758 = stablehlo.constant dense<6.0> : tensor<32x112896xf32>
    %v2759 = stablehlo.compare GT, %v943, %v2757 : (tensor<32x112896xf32>, tensor<32x112896xf32>) -> tensor<32x112896xi1>
    %v2760 = stablehlo.compare LT, %v943, %v2758 : (tensor<32x112896xf32>, tensor<32x112896xf32>) -> tensor<32x112896xi1>
    %v2761 = stablehlo.and %v2759, %v2760 : tensor<32x112896xi1>
    %v2762 = stablehlo.select %v2761, %v2756, %v2757 : tensor<32x112896xi1>, tensor<32x112896xf32>
    %v2763 = stablehlo.reshape %v2762 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2764 = stablehlo.reshape %v923 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2765 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2766 = stablehlo.constant dense<196.0> : tensor<32x576x14x14xf32>
    %v2767 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v2768 = stablehlo.reduce(%v2764 init: %v2765) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2769 = stablehlo.broadcast_in_dim %v2768, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2770 = stablehlo.divide %v2769, %v2766 : tensor<32x576x14x14xf32>
    %v2771 = stablehlo.subtract %v2764, %v2770 : tensor<32x576x14x14xf32>
    %v2772 = stablehlo.multiply %v2771, %v2771 : tensor<32x576x14x14xf32>
    %v2773 = stablehlo.reduce(%v2772 init: %v2765) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2774 = stablehlo.broadcast_in_dim %v2773, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2775 = stablehlo.divide %v2774, %v2766 : tensor<32x576x14x14xf32>
    %v2776 = stablehlo.add %v2775, %v2767 : tensor<32x576x14x14xf32>
    %v2777 = stablehlo.rsqrt %v2776 : tensor<32x576x14x14xf32>
    %v2778 = stablehlo.multiply %v2771, %v2777 : tensor<32x576x14x14xf32>
    %v2779 = stablehlo.broadcast_in_dim %ge12, dims = [1] : (tensor<576xf32>) -> tensor<32x576x14x14xf32>
    %v2780 = stablehlo.multiply %v2779, %v2763 : tensor<32x576x14x14xf32>
    %v2781 = stablehlo.reduce(%v2780 init: %v2765) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2782 = stablehlo.broadcast_in_dim %v2781, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2783 = stablehlo.multiply %v2778, %v2780 : tensor<32x576x14x14xf32>
    %v2784 = stablehlo.reduce(%v2783 init: %v2765) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2785 = stablehlo.broadcast_in_dim %v2784, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2786 = stablehlo.multiply %v2780, %v2766 : tensor<32x576x14x14xf32>
    %v2787 = stablehlo.subtract %v2786, %v2782 : tensor<32x576x14x14xf32>
    %v2788 = stablehlo.multiply %v2778, %v2785 : tensor<32x576x14x14xf32>
    %v2789 = stablehlo.subtract %v2787, %v2788 : tensor<32x576x14x14xf32>
    %v2790 = stablehlo.divide %v2777, %v2766 : tensor<32x576x14x14xf32>
    %v2791 = stablehlo.multiply %v2790, %v2789 : tensor<32x576x14x14xf32>
    %v2792 = stablehlo.reshape %v2791 : (tensor<32x576x14x14xf32>) -> tensor<32x112896xf32>
    %v2793 = stablehlo.reshape %v2792 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2794 = stablehlo.transpose %We12, dims = [1, 0, 2, 3] : (tensor<576x96x1x1xf32>) -> tensor<96x576x1x1xf32>
    %v2795 = stablehlo.reverse %v2794, dims = [2, 3] : tensor<96x576x1x1xf32>
    %v2796 = stablehlo.convolution(%v2793, %v2795)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x576x14x14xf32>, tensor<96x576x1x1xf32>) -> tensor<32x96x14x14xf32>
    %v2797 = stablehlo.reshape %v2796 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v2798 = stablehlo.add %v2797, %v2573 : tensor<32x18816xf32>
    %v2799 = stablehlo.reshape %v918 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2800 = stablehlo.reshape %v2792 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2801 = stablehlo.transpose %v2799, dims = [1, 0, 2, 3] : (tensor<32x96x14x14xf32>) -> tensor<96x32x14x14xf32>
    %v2802 = stablehlo.transpose %v2800, dims = [1, 0, 2, 3] : (tensor<32x576x14x14xf32>) -> tensor<576x32x14x14xf32>
    %v2803 = stablehlo.convolution(%v2801, %v2802)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x14x14xf32>, tensor<576x32x14x14xf32>) -> tensor<96x576x1x1xf32>
    %v2804 = stablehlo.transpose %v2803, dims = [1, 0, 2, 3] : (tensor<96x576x1x1xf32>) -> tensor<576x96x1x1xf32>
    %v2805 = stablehlo.constant dense<0.3> : tensor<576x96x1x1xf32>
    %v2806 = stablehlo.multiply %v2804, %v2805 : tensor<576x96x1x1xf32>
    %v2807 = stablehlo.subtract %We12, %v2806 : tensor<576x96x1x1xf32>
    %v2808 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2809 = stablehlo.reshape %v923 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2810 = stablehlo.constant dense<196.0> : tensor<32x576x14x14xf32>
    %v2811 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v2812 = stablehlo.reduce(%v2809 init: %v2808) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2813 = stablehlo.broadcast_in_dim %v2812, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2814 = stablehlo.divide %v2813, %v2810 : tensor<32x576x14x14xf32>
    %v2815 = stablehlo.subtract %v2809, %v2814 : tensor<32x576x14x14xf32>
    %v2816 = stablehlo.multiply %v2815, %v2815 : tensor<32x576x14x14xf32>
    %v2817 = stablehlo.reduce(%v2816 init: %v2808) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2818 = stablehlo.broadcast_in_dim %v2817, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2819 = stablehlo.divide %v2818, %v2810 : tensor<32x576x14x14xf32>
    %v2820 = stablehlo.add %v2819, %v2811 : tensor<32x576x14x14xf32>
    %v2821 = stablehlo.rsqrt %v2820 : tensor<32x576x14x14xf32>
    %v2822 = stablehlo.multiply %v2815, %v2821 : tensor<32x576x14x14xf32>
    %v2823 = stablehlo.reshape %v2762 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2824 = stablehlo.multiply %v2823, %v2822 : tensor<32x576x14x14xf32>
    %v2825 = stablehlo.reduce(%v2824 init: %v2808) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2826 = stablehlo.constant dense<0.3> : tensor<576xf32>
    %v2827 = stablehlo.multiply %v2825, %v2826 : tensor<576xf32>
    %v2828 = stablehlo.subtract %ge12, %v2827 : tensor<576xf32>
    %v2829 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2830 = stablehlo.reshape %v2762 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2831 = stablehlo.reduce(%v2830 init: %v2829) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2832 = stablehlo.constant dense<0.3> : tensor<576xf32>
    %v2833 = stablehlo.multiply %v2831, %v2832 : tensor<576xf32>
    %v2834 = stablehlo.subtract %bte12, %v2833 : tensor<576xf32>
    %v2835 = stablehlo.reshape %v947 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2836 = stablehlo.reshape %v2752 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2837 = stablehlo.transpose %v2835, dims = [1, 0, 2, 3] : (tensor<32x576x14x14xf32>) -> tensor<576x32x14x14xf32>
    %v2838 = stablehlo.transpose %v2836, dims = [1, 0, 2, 3] : (tensor<32x576x14x14xf32>) -> tensor<576x32x14x14xf32>
    %v2839 = stablehlo.convolution(%v2837, %v2838)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 576 : i64, feature_group_count = 1 : i64} : (tensor<576x32x14x14xf32>, tensor<576x32x14x14xf32>) -> tensor<1x576x3x3xf32>
    %v2840 = stablehlo.reshape %v2839 : (tensor<1x576x3x3xf32>) -> tensor<576x1x3x3xf32>
    %v2841 = stablehlo.constant dense<0.3> : tensor<576x1x3x3xf32>
    %v2842 = stablehlo.multiply %v2840, %v2841 : tensor<576x1x3x3xf32>
    %v2843 = stablehlo.subtract %Wd12, %v2842 : tensor<576x1x3x3xf32>
    %v2844 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2845 = stablehlo.reshape %v952 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2846 = stablehlo.constant dense<196.0> : tensor<32x576x14x14xf32>
    %v2847 = stablehlo.constant dense<1.0e-5> : tensor<32x576x14x14xf32>
    %v2848 = stablehlo.reduce(%v2845 init: %v2844) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2849 = stablehlo.broadcast_in_dim %v2848, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2850 = stablehlo.divide %v2849, %v2846 : tensor<32x576x14x14xf32>
    %v2851 = stablehlo.subtract %v2845, %v2850 : tensor<32x576x14x14xf32>
    %v2852 = stablehlo.multiply %v2851, %v2851 : tensor<32x576x14x14xf32>
    %v2853 = stablehlo.reduce(%v2852 init: %v2844) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<32x576xf32>
    %v2854 = stablehlo.broadcast_in_dim %v2853, dims = [0, 1] : (tensor<32x576xf32>) -> tensor<32x576x14x14xf32>
    %v2855 = stablehlo.divide %v2854, %v2846 : tensor<32x576x14x14xf32>
    %v2856 = stablehlo.add %v2855, %v2847 : tensor<32x576x14x14xf32>
    %v2857 = stablehlo.rsqrt %v2856 : tensor<32x576x14x14xf32>
    %v2858 = stablehlo.multiply %v2851, %v2857 : tensor<32x576x14x14xf32>
    %v2859 = stablehlo.reshape %v2722 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2860 = stablehlo.multiply %v2859, %v2858 : tensor<32x576x14x14xf32>
    %v2861 = stablehlo.reduce(%v2860 init: %v2844) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2862 = stablehlo.constant dense<0.3> : tensor<576xf32>
    %v2863 = stablehlo.multiply %v2861, %v2862 : tensor<576xf32>
    %v2864 = stablehlo.subtract %gd12, %v2863 : tensor<576xf32>
    %v2865 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2866 = stablehlo.reshape %v2722 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2867 = stablehlo.reduce(%v2866 init: %v2865) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x576x14x14xf32>, tensor<f32>) -> tensor<576xf32>
    %v2868 = stablehlo.constant dense<0.3> : tensor<576xf32>
    %v2869 = stablehlo.multiply %v2867, %v2868 : tensor<576xf32>
    %v2870 = stablehlo.subtract %btd12, %v2869 : tensor<576xf32>
    %v2871 = stablehlo.reshape %v976 : (tensor<32x112896xf32>) -> tensor<32x576x14x14xf32>
    %v2872 = stablehlo.reshape %v2711 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2873 = stablehlo.transpose %v2871, dims = [1, 0, 2, 3] : (tensor<32x576x14x14xf32>) -> tensor<576x32x14x14xf32>
    %v2874 = stablehlo.transpose %v2872, dims = [1, 0, 2, 3] : (tensor<32x96x14x14xf32>) -> tensor<96x32x14x14xf32>
    %v2875 = stablehlo.convolution(%v2873, %v2874)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<576x32x14x14xf32>, tensor<96x32x14x14xf32>) -> tensor<576x96x1x1xf32>
    %v2876 = stablehlo.transpose %v2875, dims = [1, 0, 2, 3] : (tensor<576x96x1x1xf32>) -> tensor<96x576x1x1xf32>
    %v2877 = stablehlo.constant dense<0.3> : tensor<96x576x1x1xf32>
    %v2878 = stablehlo.multiply %v2876, %v2877 : tensor<96x576x1x1xf32>
    %v2879 = stablehlo.subtract %Wp12, %v2878 : tensor<96x576x1x1xf32>
    %v2880 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2881 = stablehlo.reshape %v981 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2882 = stablehlo.constant dense<196.0> : tensor<32x96x14x14xf32>
    %v2883 = stablehlo.constant dense<1.0e-5> : tensor<32x96x14x14xf32>
    %v2884 = stablehlo.reduce(%v2881 init: %v2880) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v2885 = stablehlo.broadcast_in_dim %v2884, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x14x14xf32>
    %v2886 = stablehlo.divide %v2885, %v2882 : tensor<32x96x14x14xf32>
    %v2887 = stablehlo.subtract %v2881, %v2886 : tensor<32x96x14x14xf32>
    %v2888 = stablehlo.multiply %v2887, %v2887 : tensor<32x96x14x14xf32>
    %v2889 = stablehlo.reduce(%v2888 init: %v2880) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v2890 = stablehlo.broadcast_in_dim %v2889, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x14x14xf32>
    %v2891 = stablehlo.divide %v2890, %v2882 : tensor<32x96x14x14xf32>
    %v2892 = stablehlo.add %v2891, %v2883 : tensor<32x96x14x14xf32>
    %v2893 = stablehlo.rsqrt %v2892 : tensor<32x96x14x14xf32>
    %v2894 = stablehlo.multiply %v2887, %v2893 : tensor<32x96x14x14xf32>
    %v2895 = stablehlo.reshape %v2573 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2896 = stablehlo.multiply %v2895, %v2894 : tensor<32x96x14x14xf32>
    %v2897 = stablehlo.reduce(%v2896 init: %v2880) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v2898 = stablehlo.constant dense<0.3> : tensor<96xf32>
    %v2899 = stablehlo.multiply %v2897, %v2898 : tensor<96xf32>
    %v2900 = stablehlo.subtract %gp12, %v2899 : tensor<96xf32>
    %v2901 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2902 = stablehlo.reshape %v2573 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2903 = stablehlo.reduce(%v2902 init: %v2901) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v2904 = stablehlo.constant dense<0.3> : tensor<96xf32>
    %v2905 = stablehlo.multiply %v2903, %v2904 : tensor<96xf32>
    %v2906 = stablehlo.subtract %btp12, %v2905 : tensor<96xf32>
    %v2907 = stablehlo.reshape %v2798 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2908 = stablehlo.reshape %v898 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2909 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2910 = stablehlo.constant dense<196.0> : tensor<32x96x14x14xf32>
    %v2911 = stablehlo.constant dense<1.0e-5> : tensor<32x96x14x14xf32>
    %v2912 = stablehlo.reduce(%v2908 init: %v2909) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v2913 = stablehlo.broadcast_in_dim %v2912, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x14x14xf32>
    %v2914 = stablehlo.divide %v2913, %v2910 : tensor<32x96x14x14xf32>
    %v2915 = stablehlo.subtract %v2908, %v2914 : tensor<32x96x14x14xf32>
    %v2916 = stablehlo.multiply %v2915, %v2915 : tensor<32x96x14x14xf32>
    %v2917 = stablehlo.reduce(%v2916 init: %v2909) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v2918 = stablehlo.broadcast_in_dim %v2917, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x14x14xf32>
    %v2919 = stablehlo.divide %v2918, %v2910 : tensor<32x96x14x14xf32>
    %v2920 = stablehlo.add %v2919, %v2911 : tensor<32x96x14x14xf32>
    %v2921 = stablehlo.rsqrt %v2920 : tensor<32x96x14x14xf32>
    %v2922 = stablehlo.multiply %v2915, %v2921 : tensor<32x96x14x14xf32>
    %v2923 = stablehlo.broadcast_in_dim %gp11, dims = [1] : (tensor<96xf32>) -> tensor<32x96x14x14xf32>
    %v2924 = stablehlo.multiply %v2923, %v2907 : tensor<32x96x14x14xf32>
    %v2925 = stablehlo.reduce(%v2924 init: %v2909) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v2926 = stablehlo.broadcast_in_dim %v2925, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x14x14xf32>
    %v2927 = stablehlo.multiply %v2922, %v2924 : tensor<32x96x14x14xf32>
    %v2928 = stablehlo.reduce(%v2927 init: %v2909) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v2929 = stablehlo.broadcast_in_dim %v2928, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x14x14xf32>
    %v2930 = stablehlo.multiply %v2924, %v2910 : tensor<32x96x14x14xf32>
    %v2931 = stablehlo.subtract %v2930, %v2926 : tensor<32x96x14x14xf32>
    %v2932 = stablehlo.multiply %v2922, %v2929 : tensor<32x96x14x14xf32>
    %v2933 = stablehlo.subtract %v2931, %v2932 : tensor<32x96x14x14xf32>
    %v2934 = stablehlo.divide %v2921, %v2910 : tensor<32x96x14x14xf32>
    %v2935 = stablehlo.multiply %v2934, %v2933 : tensor<32x96x14x14xf32>
    %v2936 = stablehlo.reshape %v2935 : (tensor<32x96x14x14xf32>) -> tensor<32x18816xf32>
    %v2937 = stablehlo.reshape %v2936 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v2938 = stablehlo.transpose %Wp11, dims = [1, 0, 2, 3] : (tensor<96x384x1x1xf32>) -> tensor<384x96x1x1xf32>
    %v2939 = stablehlo.reverse %v2938, dims = [2, 3] : tensor<384x96x1x1xf32>
    %v2940 = stablehlo.convolution(%v2937, %v2939)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x14x14xf32>, tensor<384x96x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v2941 = stablehlo.reshape %v2940 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2942 = stablehlo.constant dense<0.0> : tensor<32x75264xf32>
    %v2943 = stablehlo.constant dense<6.0> : tensor<32x75264xf32>
    %v2944 = stablehlo.compare GT, %v889, %v2942 : (tensor<32x75264xf32>, tensor<32x75264xf32>) -> tensor<32x75264xi1>
    %v2945 = stablehlo.compare LT, %v889, %v2943 : (tensor<32x75264xf32>, tensor<32x75264xf32>) -> tensor<32x75264xi1>
    %v2946 = stablehlo.and %v2944, %v2945 : tensor<32x75264xi1>
    %v2947 = stablehlo.select %v2946, %v2941, %v2942 : tensor<32x75264xi1>, tensor<32x75264xf32>
    %v2948 = stablehlo.reshape %v2947 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2949 = stablehlo.reshape %v869 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2950 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2951 = stablehlo.constant dense<196.0> : tensor<32x384x14x14xf32>
    %v2952 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v2953 = stablehlo.reduce(%v2949 init: %v2950) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v2954 = stablehlo.broadcast_in_dim %v2953, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v2955 = stablehlo.divide %v2954, %v2951 : tensor<32x384x14x14xf32>
    %v2956 = stablehlo.subtract %v2949, %v2955 : tensor<32x384x14x14xf32>
    %v2957 = stablehlo.multiply %v2956, %v2956 : tensor<32x384x14x14xf32>
    %v2958 = stablehlo.reduce(%v2957 init: %v2950) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v2959 = stablehlo.broadcast_in_dim %v2958, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v2960 = stablehlo.divide %v2959, %v2951 : tensor<32x384x14x14xf32>
    %v2961 = stablehlo.add %v2960, %v2952 : tensor<32x384x14x14xf32>
    %v2962 = stablehlo.rsqrt %v2961 : tensor<32x384x14x14xf32>
    %v2963 = stablehlo.multiply %v2956, %v2962 : tensor<32x384x14x14xf32>
    %v2964 = stablehlo.broadcast_in_dim %gd11, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2965 = stablehlo.multiply %v2964, %v2948 : tensor<32x384x14x14xf32>
    %v2966 = stablehlo.reduce(%v2965 init: %v2950) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v2967 = stablehlo.broadcast_in_dim %v2966, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v2968 = stablehlo.multiply %v2963, %v2965 : tensor<32x384x14x14xf32>
    %v2969 = stablehlo.reduce(%v2968 init: %v2950) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v2970 = stablehlo.broadcast_in_dim %v2969, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v2971 = stablehlo.multiply %v2965, %v2951 : tensor<32x384x14x14xf32>
    %v2972 = stablehlo.subtract %v2971, %v2967 : tensor<32x384x14x14xf32>
    %v2973 = stablehlo.multiply %v2963, %v2970 : tensor<32x384x14x14xf32>
    %v2974 = stablehlo.subtract %v2972, %v2973 : tensor<32x384x14x14xf32>
    %v2975 = stablehlo.divide %v2962, %v2951 : tensor<32x384x14x14xf32>
    %v2976 = stablehlo.multiply %v2975, %v2974 : tensor<32x384x14x14xf32>
    %v2977 = stablehlo.reshape %v2976 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2978 = stablehlo.reshape %v2977 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2979 = stablehlo.reverse %Wd11, dims = [2, 3] : tensor<384x1x3x3xf32>
    %v2980 = stablehlo.convolution(%v2978, %v2979)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x3x3xf32>) -> tensor<32x384x14x14xf32>
    %v2981 = stablehlo.reshape %v2980 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2982 = stablehlo.constant dense<0.0> : tensor<32x75264xf32>
    %v2983 = stablehlo.constant dense<6.0> : tensor<32x75264xf32>
    %v2984 = stablehlo.compare GT, %v860, %v2982 : (tensor<32x75264xf32>, tensor<32x75264xf32>) -> tensor<32x75264xi1>
    %v2985 = stablehlo.compare LT, %v860, %v2983 : (tensor<32x75264xf32>, tensor<32x75264xf32>) -> tensor<32x75264xi1>
    %v2986 = stablehlo.and %v2984, %v2985 : tensor<32x75264xi1>
    %v2987 = stablehlo.select %v2986, %v2981, %v2982 : tensor<32x75264xi1>, tensor<32x75264xf32>
    %v2988 = stablehlo.reshape %v2987 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2989 = stablehlo.reshape %v840 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2990 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2991 = stablehlo.constant dense<196.0> : tensor<32x384x14x14xf32>
    %v2992 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v2993 = stablehlo.reduce(%v2989 init: %v2990) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v2994 = stablehlo.broadcast_in_dim %v2993, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v2995 = stablehlo.divide %v2994, %v2991 : tensor<32x384x14x14xf32>
    %v2996 = stablehlo.subtract %v2989, %v2995 : tensor<32x384x14x14xf32>
    %v2997 = stablehlo.multiply %v2996, %v2996 : tensor<32x384x14x14xf32>
    %v2998 = stablehlo.reduce(%v2997 init: %v2990) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v2999 = stablehlo.broadcast_in_dim %v2998, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3000 = stablehlo.divide %v2999, %v2991 : tensor<32x384x14x14xf32>
    %v3001 = stablehlo.add %v3000, %v2992 : tensor<32x384x14x14xf32>
    %v3002 = stablehlo.rsqrt %v3001 : tensor<32x384x14x14xf32>
    %v3003 = stablehlo.multiply %v2996, %v3002 : tensor<32x384x14x14xf32>
    %v3004 = stablehlo.broadcast_in_dim %ge11, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3005 = stablehlo.multiply %v3004, %v2988 : tensor<32x384x14x14xf32>
    %v3006 = stablehlo.reduce(%v3005 init: %v2990) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3007 = stablehlo.broadcast_in_dim %v3006, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3008 = stablehlo.multiply %v3003, %v3005 : tensor<32x384x14x14xf32>
    %v3009 = stablehlo.reduce(%v3008 init: %v2990) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3010 = stablehlo.broadcast_in_dim %v3009, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3011 = stablehlo.multiply %v3005, %v2991 : tensor<32x384x14x14xf32>
    %v3012 = stablehlo.subtract %v3011, %v3007 : tensor<32x384x14x14xf32>
    %v3013 = stablehlo.multiply %v3003, %v3010 : tensor<32x384x14x14xf32>
    %v3014 = stablehlo.subtract %v3012, %v3013 : tensor<32x384x14x14xf32>
    %v3015 = stablehlo.divide %v3002, %v2991 : tensor<32x384x14x14xf32>
    %v3016 = stablehlo.multiply %v3015, %v3014 : tensor<32x384x14x14xf32>
    %v3017 = stablehlo.reshape %v3016 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3018 = stablehlo.reshape %v3017 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3019 = stablehlo.transpose %We11, dims = [1, 0, 2, 3] : (tensor<384x64x1x1xf32>) -> tensor<64x384x1x1xf32>
    %v3020 = stablehlo.reverse %v3019, dims = [2, 3] : tensor<64x384x1x1xf32>
    %v3021 = stablehlo.convolution(%v3018, %v3020)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<64x384x1x1xf32>) -> tensor<32x64x14x14xf32>
    %v3022 = stablehlo.reshape %v3021 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v3023 = stablehlo.reshape %v835 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3024 = stablehlo.reshape %v3017 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3025 = stablehlo.transpose %v3023, dims = [1, 0, 2, 3] : (tensor<32x64x14x14xf32>) -> tensor<64x32x14x14xf32>
    %v3026 = stablehlo.transpose %v3024, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3027 = stablehlo.convolution(%v3025, %v3026)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<64x384x1x1xf32>
    %v3028 = stablehlo.transpose %v3027, dims = [1, 0, 2, 3] : (tensor<64x384x1x1xf32>) -> tensor<384x64x1x1xf32>
    %v3029 = stablehlo.constant dense<0.3> : tensor<384x64x1x1xf32>
    %v3030 = stablehlo.multiply %v3028, %v3029 : tensor<384x64x1x1xf32>
    %v3031 = stablehlo.subtract %We11, %v3030 : tensor<384x64x1x1xf32>
    %v3032 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3033 = stablehlo.reshape %v840 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3034 = stablehlo.constant dense<196.0> : tensor<32x384x14x14xf32>
    %v3035 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3036 = stablehlo.reduce(%v3033 init: %v3032) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3037 = stablehlo.broadcast_in_dim %v3036, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3038 = stablehlo.divide %v3037, %v3034 : tensor<32x384x14x14xf32>
    %v3039 = stablehlo.subtract %v3033, %v3038 : tensor<32x384x14x14xf32>
    %v3040 = stablehlo.multiply %v3039, %v3039 : tensor<32x384x14x14xf32>
    %v3041 = stablehlo.reduce(%v3040 init: %v3032) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3042 = stablehlo.broadcast_in_dim %v3041, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3043 = stablehlo.divide %v3042, %v3034 : tensor<32x384x14x14xf32>
    %v3044 = stablehlo.add %v3043, %v3035 : tensor<32x384x14x14xf32>
    %v3045 = stablehlo.rsqrt %v3044 : tensor<32x384x14x14xf32>
    %v3046 = stablehlo.multiply %v3039, %v3045 : tensor<32x384x14x14xf32>
    %v3047 = stablehlo.reshape %v2987 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3048 = stablehlo.multiply %v3047, %v3046 : tensor<32x384x14x14xf32>
    %v3049 = stablehlo.reduce(%v3048 init: %v3032) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3050 = stablehlo.constant dense<0.3> : tensor<384xf32>
    %v3051 = stablehlo.multiply %v3049, %v3050 : tensor<384xf32>
    %v3052 = stablehlo.subtract %ge11, %v3051 : tensor<384xf32>
    %v3053 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3054 = stablehlo.reshape %v2987 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3055 = stablehlo.reduce(%v3054 init: %v3053) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3056 = stablehlo.constant dense<0.3> : tensor<384xf32>
    %v3057 = stablehlo.multiply %v3055, %v3056 : tensor<384xf32>
    %v3058 = stablehlo.subtract %bte11, %v3057 : tensor<384xf32>
    %v3059 = stablehlo.reshape %v864 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3060 = stablehlo.reshape %v2977 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3061 = stablehlo.transpose %v3059, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3062 = stablehlo.transpose %v3060, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3063 = stablehlo.convolution(%v3061, %v3062)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x3x3xf32>
    %v3064 = stablehlo.reshape %v3063 : (tensor<1x384x3x3xf32>) -> tensor<384x1x3x3xf32>
    %v3065 = stablehlo.constant dense<0.3> : tensor<384x1x3x3xf32>
    %v3066 = stablehlo.multiply %v3064, %v3065 : tensor<384x1x3x3xf32>
    %v3067 = stablehlo.subtract %Wd11, %v3066 : tensor<384x1x3x3xf32>
    %v3068 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3069 = stablehlo.reshape %v869 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3070 = stablehlo.constant dense<196.0> : tensor<32x384x14x14xf32>
    %v3071 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3072 = stablehlo.reduce(%v3069 init: %v3068) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3073 = stablehlo.broadcast_in_dim %v3072, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3074 = stablehlo.divide %v3073, %v3070 : tensor<32x384x14x14xf32>
    %v3075 = stablehlo.subtract %v3069, %v3074 : tensor<32x384x14x14xf32>
    %v3076 = stablehlo.multiply %v3075, %v3075 : tensor<32x384x14x14xf32>
    %v3077 = stablehlo.reduce(%v3076 init: %v3068) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3078 = stablehlo.broadcast_in_dim %v3077, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3079 = stablehlo.divide %v3078, %v3070 : tensor<32x384x14x14xf32>
    %v3080 = stablehlo.add %v3079, %v3071 : tensor<32x384x14x14xf32>
    %v3081 = stablehlo.rsqrt %v3080 : tensor<32x384x14x14xf32>
    %v3082 = stablehlo.multiply %v3075, %v3081 : tensor<32x384x14x14xf32>
    %v3083 = stablehlo.reshape %v2947 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3084 = stablehlo.multiply %v3083, %v3082 : tensor<32x384x14x14xf32>
    %v3085 = stablehlo.reduce(%v3084 init: %v3068) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3086 = stablehlo.constant dense<0.3> : tensor<384xf32>
    %v3087 = stablehlo.multiply %v3085, %v3086 : tensor<384xf32>
    %v3088 = stablehlo.subtract %gd11, %v3087 : tensor<384xf32>
    %v3089 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3090 = stablehlo.reshape %v2947 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3091 = stablehlo.reduce(%v3090 init: %v3089) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3092 = stablehlo.constant dense<0.3> : tensor<384xf32>
    %v3093 = stablehlo.multiply %v3091, %v3092 : tensor<384xf32>
    %v3094 = stablehlo.subtract %btd11, %v3093 : tensor<384xf32>
    %v3095 = stablehlo.reshape %v893 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3096 = stablehlo.reshape %v2936 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v3097 = stablehlo.transpose %v3095, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3098 = stablehlo.transpose %v3096, dims = [1, 0, 2, 3] : (tensor<32x96x14x14xf32>) -> tensor<96x32x14x14xf32>
    %v3099 = stablehlo.convolution(%v3097, %v3098)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<96x32x14x14xf32>) -> tensor<384x96x1x1xf32>
    %v3100 = stablehlo.transpose %v3099, dims = [1, 0, 2, 3] : (tensor<384x96x1x1xf32>) -> tensor<96x384x1x1xf32>
    %v3101 = stablehlo.constant dense<0.3> : tensor<96x384x1x1xf32>
    %v3102 = stablehlo.multiply %v3100, %v3101 : tensor<96x384x1x1xf32>
    %v3103 = stablehlo.subtract %Wp11, %v3102 : tensor<96x384x1x1xf32>
    %v3104 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3105 = stablehlo.reshape %v898 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v3106 = stablehlo.constant dense<196.0> : tensor<32x96x14x14xf32>
    %v3107 = stablehlo.constant dense<1.0e-5> : tensor<32x96x14x14xf32>
    %v3108 = stablehlo.reduce(%v3105 init: %v3104) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v3109 = stablehlo.broadcast_in_dim %v3108, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x14x14xf32>
    %v3110 = stablehlo.divide %v3109, %v3106 : tensor<32x96x14x14xf32>
    %v3111 = stablehlo.subtract %v3105, %v3110 : tensor<32x96x14x14xf32>
    %v3112 = stablehlo.multiply %v3111, %v3111 : tensor<32x96x14x14xf32>
    %v3113 = stablehlo.reduce(%v3112 init: %v3104) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v3114 = stablehlo.broadcast_in_dim %v3113, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x14x14xf32>
    %v3115 = stablehlo.divide %v3114, %v3106 : tensor<32x96x14x14xf32>
    %v3116 = stablehlo.add %v3115, %v3107 : tensor<32x96x14x14xf32>
    %v3117 = stablehlo.rsqrt %v3116 : tensor<32x96x14x14xf32>
    %v3118 = stablehlo.multiply %v3111, %v3117 : tensor<32x96x14x14xf32>
    %v3119 = stablehlo.reshape %v2798 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v3120 = stablehlo.multiply %v3119, %v3118 : tensor<32x96x14x14xf32>
    %v3121 = stablehlo.reduce(%v3120 init: %v3104) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v3122 = stablehlo.constant dense<0.3> : tensor<96xf32>
    %v3123 = stablehlo.multiply %v3121, %v3122 : tensor<96xf32>
    %v3124 = stablehlo.subtract %gp11, %v3123 : tensor<96xf32>
    %v3125 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3126 = stablehlo.reshape %v2798 : (tensor<32x18816xf32>) -> tensor<32x96x14x14xf32>
    %v3127 = stablehlo.reduce(%v3126 init: %v3125) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x14x14xf32>, tensor<f32>) -> tensor<96xf32>
    %v3128 = stablehlo.constant dense<0.3> : tensor<96xf32>
    %v3129 = stablehlo.multiply %v3127, %v3128 : tensor<96xf32>
    %v3130 = stablehlo.subtract %btp11, %v3129 : tensor<96xf32>
    %v3131 = stablehlo.reshape %v3022 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3132 = stablehlo.reshape %v814 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3133 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3134 = stablehlo.constant dense<196.0> : tensor<32x64x14x14xf32>
    %v3135 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v3136 = stablehlo.reduce(%v3132 init: %v3133) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3137 = stablehlo.broadcast_in_dim %v3136, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v3138 = stablehlo.divide %v3137, %v3134 : tensor<32x64x14x14xf32>
    %v3139 = stablehlo.subtract %v3132, %v3138 : tensor<32x64x14x14xf32>
    %v3140 = stablehlo.multiply %v3139, %v3139 : tensor<32x64x14x14xf32>
    %v3141 = stablehlo.reduce(%v3140 init: %v3133) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3142 = stablehlo.broadcast_in_dim %v3141, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v3143 = stablehlo.divide %v3142, %v3134 : tensor<32x64x14x14xf32>
    %v3144 = stablehlo.add %v3143, %v3135 : tensor<32x64x14x14xf32>
    %v3145 = stablehlo.rsqrt %v3144 : tensor<32x64x14x14xf32>
    %v3146 = stablehlo.multiply %v3139, %v3145 : tensor<32x64x14x14xf32>
    %v3147 = stablehlo.broadcast_in_dim %gp10, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3148 = stablehlo.multiply %v3147, %v3131 : tensor<32x64x14x14xf32>
    %v3149 = stablehlo.reduce(%v3148 init: %v3133) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3150 = stablehlo.broadcast_in_dim %v3149, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v3151 = stablehlo.multiply %v3146, %v3148 : tensor<32x64x14x14xf32>
    %v3152 = stablehlo.reduce(%v3151 init: %v3133) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3153 = stablehlo.broadcast_in_dim %v3152, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v3154 = stablehlo.multiply %v3148, %v3134 : tensor<32x64x14x14xf32>
    %v3155 = stablehlo.subtract %v3154, %v3150 : tensor<32x64x14x14xf32>
    %v3156 = stablehlo.multiply %v3146, %v3153 : tensor<32x64x14x14xf32>
    %v3157 = stablehlo.subtract %v3155, %v3156 : tensor<32x64x14x14xf32>
    %v3158 = stablehlo.divide %v3145, %v3134 : tensor<32x64x14x14xf32>
    %v3159 = stablehlo.multiply %v3158, %v3157 : tensor<32x64x14x14xf32>
    %v3160 = stablehlo.reshape %v3159 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v3161 = stablehlo.reshape %v3160 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3162 = stablehlo.transpose %Wp10, dims = [1, 0, 2, 3] : (tensor<64x384x1x1xf32>) -> tensor<384x64x1x1xf32>
    %v3163 = stablehlo.reverse %v3162, dims = [2, 3] : tensor<384x64x1x1xf32>
    %v3164 = stablehlo.convolution(%v3161, %v3163)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x14x14xf32>, tensor<384x64x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v3165 = stablehlo.reshape %v3164 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3166 = stablehlo.constant dense<0.0> : tensor<32x75264xf32>
    %v3167 = stablehlo.constant dense<6.0> : tensor<32x75264xf32>
    %v3168 = stablehlo.compare GT, %v805, %v3166 : (tensor<32x75264xf32>, tensor<32x75264xf32>) -> tensor<32x75264xi1>
    %v3169 = stablehlo.compare LT, %v805, %v3167 : (tensor<32x75264xf32>, tensor<32x75264xf32>) -> tensor<32x75264xi1>
    %v3170 = stablehlo.and %v3168, %v3169 : tensor<32x75264xi1>
    %v3171 = stablehlo.select %v3170, %v3165, %v3166 : tensor<32x75264xi1>, tensor<32x75264xf32>
    %v3172 = stablehlo.reshape %v3171 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3173 = stablehlo.reshape %v785 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3174 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3175 = stablehlo.constant dense<196.0> : tensor<32x384x14x14xf32>
    %v3176 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3177 = stablehlo.reduce(%v3173 init: %v3174) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3178 = stablehlo.broadcast_in_dim %v3177, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3179 = stablehlo.divide %v3178, %v3175 : tensor<32x384x14x14xf32>
    %v3180 = stablehlo.subtract %v3173, %v3179 : tensor<32x384x14x14xf32>
    %v3181 = stablehlo.multiply %v3180, %v3180 : tensor<32x384x14x14xf32>
    %v3182 = stablehlo.reduce(%v3181 init: %v3174) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3183 = stablehlo.broadcast_in_dim %v3182, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3184 = stablehlo.divide %v3183, %v3175 : tensor<32x384x14x14xf32>
    %v3185 = stablehlo.add %v3184, %v3176 : tensor<32x384x14x14xf32>
    %v3186 = stablehlo.rsqrt %v3185 : tensor<32x384x14x14xf32>
    %v3187 = stablehlo.multiply %v3180, %v3186 : tensor<32x384x14x14xf32>
    %v3188 = stablehlo.broadcast_in_dim %gd10, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3189 = stablehlo.multiply %v3188, %v3172 : tensor<32x384x14x14xf32>
    %v3190 = stablehlo.reduce(%v3189 init: %v3174) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3191 = stablehlo.broadcast_in_dim %v3190, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3192 = stablehlo.multiply %v3187, %v3189 : tensor<32x384x14x14xf32>
    %v3193 = stablehlo.reduce(%v3192 init: %v3174) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3194 = stablehlo.broadcast_in_dim %v3193, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3195 = stablehlo.multiply %v3189, %v3175 : tensor<32x384x14x14xf32>
    %v3196 = stablehlo.subtract %v3195, %v3191 : tensor<32x384x14x14xf32>
    %v3197 = stablehlo.multiply %v3187, %v3194 : tensor<32x384x14x14xf32>
    %v3198 = stablehlo.subtract %v3196, %v3197 : tensor<32x384x14x14xf32>
    %v3199 = stablehlo.divide %v3186, %v3175 : tensor<32x384x14x14xf32>
    %v3200 = stablehlo.multiply %v3199, %v3198 : tensor<32x384x14x14xf32>
    %v3201 = stablehlo.reshape %v3200 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3202 = stablehlo.reshape %v3201 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3203 = stablehlo.reverse %Wd10, dims = [2, 3] : tensor<384x1x3x3xf32>
    %v3204 = stablehlo.convolution(%v3202, %v3203)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x3x3xf32>) -> tensor<32x384x14x14xf32>
    %v3205 = stablehlo.reshape %v3204 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3206 = stablehlo.constant dense<0.0> : tensor<32x75264xf32>
    %v3207 = stablehlo.constant dense<6.0> : tensor<32x75264xf32>
    %v3208 = stablehlo.compare GT, %v776, %v3206 : (tensor<32x75264xf32>, tensor<32x75264xf32>) -> tensor<32x75264xi1>
    %v3209 = stablehlo.compare LT, %v776, %v3207 : (tensor<32x75264xf32>, tensor<32x75264xf32>) -> tensor<32x75264xi1>
    %v3210 = stablehlo.and %v3208, %v3209 : tensor<32x75264xi1>
    %v3211 = stablehlo.select %v3210, %v3205, %v3206 : tensor<32x75264xi1>, tensor<32x75264xf32>
    %v3212 = stablehlo.reshape %v3211 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3213 = stablehlo.reshape %v756 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3214 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3215 = stablehlo.constant dense<196.0> : tensor<32x384x14x14xf32>
    %v3216 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3217 = stablehlo.reduce(%v3213 init: %v3214) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3218 = stablehlo.broadcast_in_dim %v3217, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3219 = stablehlo.divide %v3218, %v3215 : tensor<32x384x14x14xf32>
    %v3220 = stablehlo.subtract %v3213, %v3219 : tensor<32x384x14x14xf32>
    %v3221 = stablehlo.multiply %v3220, %v3220 : tensor<32x384x14x14xf32>
    %v3222 = stablehlo.reduce(%v3221 init: %v3214) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3223 = stablehlo.broadcast_in_dim %v3222, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3224 = stablehlo.divide %v3223, %v3215 : tensor<32x384x14x14xf32>
    %v3225 = stablehlo.add %v3224, %v3216 : tensor<32x384x14x14xf32>
    %v3226 = stablehlo.rsqrt %v3225 : tensor<32x384x14x14xf32>
    %v3227 = stablehlo.multiply %v3220, %v3226 : tensor<32x384x14x14xf32>
    %v3228 = stablehlo.broadcast_in_dim %ge10, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3229 = stablehlo.multiply %v3228, %v3212 : tensor<32x384x14x14xf32>
    %v3230 = stablehlo.reduce(%v3229 init: %v3214) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3231 = stablehlo.broadcast_in_dim %v3230, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3232 = stablehlo.multiply %v3227, %v3229 : tensor<32x384x14x14xf32>
    %v3233 = stablehlo.reduce(%v3232 init: %v3214) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3234 = stablehlo.broadcast_in_dim %v3233, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3235 = stablehlo.multiply %v3229, %v3215 : tensor<32x384x14x14xf32>
    %v3236 = stablehlo.subtract %v3235, %v3231 : tensor<32x384x14x14xf32>
    %v3237 = stablehlo.multiply %v3227, %v3234 : tensor<32x384x14x14xf32>
    %v3238 = stablehlo.subtract %v3236, %v3237 : tensor<32x384x14x14xf32>
    %v3239 = stablehlo.divide %v3226, %v3215 : tensor<32x384x14x14xf32>
    %v3240 = stablehlo.multiply %v3239, %v3238 : tensor<32x384x14x14xf32>
    %v3241 = stablehlo.reshape %v3240 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3242 = stablehlo.reshape %v3241 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3243 = stablehlo.transpose %We10, dims = [1, 0, 2, 3] : (tensor<384x64x1x1xf32>) -> tensor<64x384x1x1xf32>
    %v3244 = stablehlo.reverse %v3243, dims = [2, 3] : tensor<64x384x1x1xf32>
    %v3245 = stablehlo.convolution(%v3242, %v3244)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<64x384x1x1xf32>) -> tensor<32x64x14x14xf32>
    %v3246 = stablehlo.reshape %v3245 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v3247 = stablehlo.add %v3246, %v3022 : tensor<32x12544xf32>
    %v3248 = stablehlo.reshape %v751 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3249 = stablehlo.reshape %v3241 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3250 = stablehlo.transpose %v3248, dims = [1, 0, 2, 3] : (tensor<32x64x14x14xf32>) -> tensor<64x32x14x14xf32>
    %v3251 = stablehlo.transpose %v3249, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3252 = stablehlo.convolution(%v3250, %v3251)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<64x384x1x1xf32>
    %v3253 = stablehlo.transpose %v3252, dims = [1, 0, 2, 3] : (tensor<64x384x1x1xf32>) -> tensor<384x64x1x1xf32>
    %v3254 = stablehlo.constant dense<0.3> : tensor<384x64x1x1xf32>
    %v3255 = stablehlo.multiply %v3253, %v3254 : tensor<384x64x1x1xf32>
    %v3256 = stablehlo.subtract %We10, %v3255 : tensor<384x64x1x1xf32>
    %v3257 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3258 = stablehlo.reshape %v756 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3259 = stablehlo.constant dense<196.0> : tensor<32x384x14x14xf32>
    %v3260 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3261 = stablehlo.reduce(%v3258 init: %v3257) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3262 = stablehlo.broadcast_in_dim %v3261, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3263 = stablehlo.divide %v3262, %v3259 : tensor<32x384x14x14xf32>
    %v3264 = stablehlo.subtract %v3258, %v3263 : tensor<32x384x14x14xf32>
    %v3265 = stablehlo.multiply %v3264, %v3264 : tensor<32x384x14x14xf32>
    %v3266 = stablehlo.reduce(%v3265 init: %v3257) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3267 = stablehlo.broadcast_in_dim %v3266, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3268 = stablehlo.divide %v3267, %v3259 : tensor<32x384x14x14xf32>
    %v3269 = stablehlo.add %v3268, %v3260 : tensor<32x384x14x14xf32>
    %v3270 = stablehlo.rsqrt %v3269 : tensor<32x384x14x14xf32>
    %v3271 = stablehlo.multiply %v3264, %v3270 : tensor<32x384x14x14xf32>
    %v3272 = stablehlo.reshape %v3211 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3273 = stablehlo.multiply %v3272, %v3271 : tensor<32x384x14x14xf32>
    %v3274 = stablehlo.reduce(%v3273 init: %v3257) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3275 = stablehlo.constant dense<0.3> : tensor<384xf32>
    %v3276 = stablehlo.multiply %v3274, %v3275 : tensor<384xf32>
    %v3277 = stablehlo.subtract %ge10, %v3276 : tensor<384xf32>
    %v3278 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3279 = stablehlo.reshape %v3211 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3280 = stablehlo.reduce(%v3279 init: %v3278) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3281 = stablehlo.constant dense<0.3> : tensor<384xf32>
    %v3282 = stablehlo.multiply %v3280, %v3281 : tensor<384xf32>
    %v3283 = stablehlo.subtract %bte10, %v3282 : tensor<384xf32>
    %v3284 = stablehlo.reshape %v780 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3285 = stablehlo.reshape %v3201 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3286 = stablehlo.transpose %v3284, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3287 = stablehlo.transpose %v3285, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3288 = stablehlo.convolution(%v3286, %v3287)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x3x3xf32>
    %v3289 = stablehlo.reshape %v3288 : (tensor<1x384x3x3xf32>) -> tensor<384x1x3x3xf32>
    %v3290 = stablehlo.constant dense<0.3> : tensor<384x1x3x3xf32>
    %v3291 = stablehlo.multiply %v3289, %v3290 : tensor<384x1x3x3xf32>
    %v3292 = stablehlo.subtract %Wd10, %v3291 : tensor<384x1x3x3xf32>
    %v3293 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3294 = stablehlo.reshape %v785 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3295 = stablehlo.constant dense<196.0> : tensor<32x384x14x14xf32>
    %v3296 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3297 = stablehlo.reduce(%v3294 init: %v3293) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3298 = stablehlo.broadcast_in_dim %v3297, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3299 = stablehlo.divide %v3298, %v3295 : tensor<32x384x14x14xf32>
    %v3300 = stablehlo.subtract %v3294, %v3299 : tensor<32x384x14x14xf32>
    %v3301 = stablehlo.multiply %v3300, %v3300 : tensor<32x384x14x14xf32>
    %v3302 = stablehlo.reduce(%v3301 init: %v3293) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3303 = stablehlo.broadcast_in_dim %v3302, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3304 = stablehlo.divide %v3303, %v3295 : tensor<32x384x14x14xf32>
    %v3305 = stablehlo.add %v3304, %v3296 : tensor<32x384x14x14xf32>
    %v3306 = stablehlo.rsqrt %v3305 : tensor<32x384x14x14xf32>
    %v3307 = stablehlo.multiply %v3300, %v3306 : tensor<32x384x14x14xf32>
    %v3308 = stablehlo.reshape %v3171 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3309 = stablehlo.multiply %v3308, %v3307 : tensor<32x384x14x14xf32>
    %v3310 = stablehlo.reduce(%v3309 init: %v3293) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3311 = stablehlo.constant dense<0.3> : tensor<384xf32>
    %v3312 = stablehlo.multiply %v3310, %v3311 : tensor<384xf32>
    %v3313 = stablehlo.subtract %gd10, %v3312 : tensor<384xf32>
    %v3314 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3315 = stablehlo.reshape %v3171 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3316 = stablehlo.reduce(%v3315 init: %v3314) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3317 = stablehlo.constant dense<0.3> : tensor<384xf32>
    %v3318 = stablehlo.multiply %v3316, %v3317 : tensor<384xf32>
    %v3319 = stablehlo.subtract %btd10, %v3318 : tensor<384xf32>
    %v3320 = stablehlo.reshape %v809 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3321 = stablehlo.reshape %v3160 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3322 = stablehlo.transpose %v3320, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3323 = stablehlo.transpose %v3321, dims = [1, 0, 2, 3] : (tensor<32x64x14x14xf32>) -> tensor<64x32x14x14xf32>
    %v3324 = stablehlo.convolution(%v3322, %v3323)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<64x32x14x14xf32>) -> tensor<384x64x1x1xf32>
    %v3325 = stablehlo.transpose %v3324, dims = [1, 0, 2, 3] : (tensor<384x64x1x1xf32>) -> tensor<64x384x1x1xf32>
    %v3326 = stablehlo.constant dense<0.3> : tensor<64x384x1x1xf32>
    %v3327 = stablehlo.multiply %v3325, %v3326 : tensor<64x384x1x1xf32>
    %v3328 = stablehlo.subtract %Wp10, %v3327 : tensor<64x384x1x1xf32>
    %v3329 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3330 = stablehlo.reshape %v814 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3331 = stablehlo.constant dense<196.0> : tensor<32x64x14x14xf32>
    %v3332 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v3333 = stablehlo.reduce(%v3330 init: %v3329) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3334 = stablehlo.broadcast_in_dim %v3333, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v3335 = stablehlo.divide %v3334, %v3331 : tensor<32x64x14x14xf32>
    %v3336 = stablehlo.subtract %v3330, %v3335 : tensor<32x64x14x14xf32>
    %v3337 = stablehlo.multiply %v3336, %v3336 : tensor<32x64x14x14xf32>
    %v3338 = stablehlo.reduce(%v3337 init: %v3329) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3339 = stablehlo.broadcast_in_dim %v3338, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v3340 = stablehlo.divide %v3339, %v3331 : tensor<32x64x14x14xf32>
    %v3341 = stablehlo.add %v3340, %v3332 : tensor<32x64x14x14xf32>
    %v3342 = stablehlo.rsqrt %v3341 : tensor<32x64x14x14xf32>
    %v3343 = stablehlo.multiply %v3336, %v3342 : tensor<32x64x14x14xf32>
    %v3344 = stablehlo.reshape %v3022 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3345 = stablehlo.multiply %v3344, %v3343 : tensor<32x64x14x14xf32>
    %v3346 = stablehlo.reduce(%v3345 init: %v3329) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3347 = stablehlo.constant dense<0.3> : tensor<64xf32>
    %v3348 = stablehlo.multiply %v3346, %v3347 : tensor<64xf32>
    %v3349 = stablehlo.subtract %gp10, %v3348 : tensor<64xf32>
    %v3350 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3351 = stablehlo.reshape %v3022 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3352 = stablehlo.reduce(%v3351 init: %v3350) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3353 = stablehlo.constant dense<0.3> : tensor<64xf32>
    %v3354 = stablehlo.multiply %v3352, %v3353 : tensor<64xf32>
    %v3355 = stablehlo.subtract %btp10, %v3354 : tensor<64xf32>
    %v3356 = stablehlo.reshape %v3247 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3357 = stablehlo.reshape %v730 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3358 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3359 = stablehlo.constant dense<196.0> : tensor<32x64x14x14xf32>
    %v3360 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v3361 = stablehlo.reduce(%v3357 init: %v3358) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3362 = stablehlo.broadcast_in_dim %v3361, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v3363 = stablehlo.divide %v3362, %v3359 : tensor<32x64x14x14xf32>
    %v3364 = stablehlo.subtract %v3357, %v3363 : tensor<32x64x14x14xf32>
    %v3365 = stablehlo.multiply %v3364, %v3364 : tensor<32x64x14x14xf32>
    %v3366 = stablehlo.reduce(%v3365 init: %v3358) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3367 = stablehlo.broadcast_in_dim %v3366, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v3368 = stablehlo.divide %v3367, %v3359 : tensor<32x64x14x14xf32>
    %v3369 = stablehlo.add %v3368, %v3360 : tensor<32x64x14x14xf32>
    %v3370 = stablehlo.rsqrt %v3369 : tensor<32x64x14x14xf32>
    %v3371 = stablehlo.multiply %v3364, %v3370 : tensor<32x64x14x14xf32>
    %v3372 = stablehlo.broadcast_in_dim %gp9, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3373 = stablehlo.multiply %v3372, %v3356 : tensor<32x64x14x14xf32>
    %v3374 = stablehlo.reduce(%v3373 init: %v3358) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3375 = stablehlo.broadcast_in_dim %v3374, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v3376 = stablehlo.multiply %v3371, %v3373 : tensor<32x64x14x14xf32>
    %v3377 = stablehlo.reduce(%v3376 init: %v3358) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3378 = stablehlo.broadcast_in_dim %v3377, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v3379 = stablehlo.multiply %v3373, %v3359 : tensor<32x64x14x14xf32>
    %v3380 = stablehlo.subtract %v3379, %v3375 : tensor<32x64x14x14xf32>
    %v3381 = stablehlo.multiply %v3371, %v3378 : tensor<32x64x14x14xf32>
    %v3382 = stablehlo.subtract %v3380, %v3381 : tensor<32x64x14x14xf32>
    %v3383 = stablehlo.divide %v3370, %v3359 : tensor<32x64x14x14xf32>
    %v3384 = stablehlo.multiply %v3383, %v3382 : tensor<32x64x14x14xf32>
    %v3385 = stablehlo.reshape %v3384 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v3386 = stablehlo.reshape %v3385 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3387 = stablehlo.transpose %Wp9, dims = [1, 0, 2, 3] : (tensor<64x384x1x1xf32>) -> tensor<384x64x1x1xf32>
    %v3388 = stablehlo.reverse %v3387, dims = [2, 3] : tensor<384x64x1x1xf32>
    %v3389 = stablehlo.convolution(%v3386, %v3388)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x14x14xf32>, tensor<384x64x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v3390 = stablehlo.reshape %v3389 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3391 = stablehlo.constant dense<0.0> : tensor<32x75264xf32>
    %v3392 = stablehlo.constant dense<6.0> : tensor<32x75264xf32>
    %v3393 = stablehlo.compare GT, %v721, %v3391 : (tensor<32x75264xf32>, tensor<32x75264xf32>) -> tensor<32x75264xi1>
    %v3394 = stablehlo.compare LT, %v721, %v3392 : (tensor<32x75264xf32>, tensor<32x75264xf32>) -> tensor<32x75264xi1>
    %v3395 = stablehlo.and %v3393, %v3394 : tensor<32x75264xi1>
    %v3396 = stablehlo.select %v3395, %v3390, %v3391 : tensor<32x75264xi1>, tensor<32x75264xf32>
    %v3397 = stablehlo.reshape %v3396 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3398 = stablehlo.reshape %v701 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3399 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3400 = stablehlo.constant dense<196.0> : tensor<32x384x14x14xf32>
    %v3401 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3402 = stablehlo.reduce(%v3398 init: %v3399) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3403 = stablehlo.broadcast_in_dim %v3402, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3404 = stablehlo.divide %v3403, %v3400 : tensor<32x384x14x14xf32>
    %v3405 = stablehlo.subtract %v3398, %v3404 : tensor<32x384x14x14xf32>
    %v3406 = stablehlo.multiply %v3405, %v3405 : tensor<32x384x14x14xf32>
    %v3407 = stablehlo.reduce(%v3406 init: %v3399) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3408 = stablehlo.broadcast_in_dim %v3407, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3409 = stablehlo.divide %v3408, %v3400 : tensor<32x384x14x14xf32>
    %v3410 = stablehlo.add %v3409, %v3401 : tensor<32x384x14x14xf32>
    %v3411 = stablehlo.rsqrt %v3410 : tensor<32x384x14x14xf32>
    %v3412 = stablehlo.multiply %v3405, %v3411 : tensor<32x384x14x14xf32>
    %v3413 = stablehlo.broadcast_in_dim %gd9, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3414 = stablehlo.multiply %v3413, %v3397 : tensor<32x384x14x14xf32>
    %v3415 = stablehlo.reduce(%v3414 init: %v3399) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3416 = stablehlo.broadcast_in_dim %v3415, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3417 = stablehlo.multiply %v3412, %v3414 : tensor<32x384x14x14xf32>
    %v3418 = stablehlo.reduce(%v3417 init: %v3399) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3419 = stablehlo.broadcast_in_dim %v3418, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3420 = stablehlo.multiply %v3414, %v3400 : tensor<32x384x14x14xf32>
    %v3421 = stablehlo.subtract %v3420, %v3416 : tensor<32x384x14x14xf32>
    %v3422 = stablehlo.multiply %v3412, %v3419 : tensor<32x384x14x14xf32>
    %v3423 = stablehlo.subtract %v3421, %v3422 : tensor<32x384x14x14xf32>
    %v3424 = stablehlo.divide %v3411, %v3400 : tensor<32x384x14x14xf32>
    %v3425 = stablehlo.multiply %v3424, %v3423 : tensor<32x384x14x14xf32>
    %v3426 = stablehlo.reshape %v3425 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3427 = stablehlo.reshape %v3426 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3428 = stablehlo.reverse %Wd9, dims = [2, 3] : tensor<384x1x3x3xf32>
    %v3429 = stablehlo.convolution(%v3427, %v3428)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x3x3xf32>) -> tensor<32x384x14x14xf32>
    %v3430 = stablehlo.reshape %v3429 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3431 = stablehlo.constant dense<0.0> : tensor<32x75264xf32>
    %v3432 = stablehlo.constant dense<6.0> : tensor<32x75264xf32>
    %v3433 = stablehlo.compare GT, %v692, %v3431 : (tensor<32x75264xf32>, tensor<32x75264xf32>) -> tensor<32x75264xi1>
    %v3434 = stablehlo.compare LT, %v692, %v3432 : (tensor<32x75264xf32>, tensor<32x75264xf32>) -> tensor<32x75264xi1>
    %v3435 = stablehlo.and %v3433, %v3434 : tensor<32x75264xi1>
    %v3436 = stablehlo.select %v3435, %v3430, %v3431 : tensor<32x75264xi1>, tensor<32x75264xf32>
    %v3437 = stablehlo.reshape %v3436 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3438 = stablehlo.reshape %v672 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3439 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3440 = stablehlo.constant dense<196.0> : tensor<32x384x14x14xf32>
    %v3441 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3442 = stablehlo.reduce(%v3438 init: %v3439) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3443 = stablehlo.broadcast_in_dim %v3442, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3444 = stablehlo.divide %v3443, %v3440 : tensor<32x384x14x14xf32>
    %v3445 = stablehlo.subtract %v3438, %v3444 : tensor<32x384x14x14xf32>
    %v3446 = stablehlo.multiply %v3445, %v3445 : tensor<32x384x14x14xf32>
    %v3447 = stablehlo.reduce(%v3446 init: %v3439) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3448 = stablehlo.broadcast_in_dim %v3447, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3449 = stablehlo.divide %v3448, %v3440 : tensor<32x384x14x14xf32>
    %v3450 = stablehlo.add %v3449, %v3441 : tensor<32x384x14x14xf32>
    %v3451 = stablehlo.rsqrt %v3450 : tensor<32x384x14x14xf32>
    %v3452 = stablehlo.multiply %v3445, %v3451 : tensor<32x384x14x14xf32>
    %v3453 = stablehlo.broadcast_in_dim %ge9, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3454 = stablehlo.multiply %v3453, %v3437 : tensor<32x384x14x14xf32>
    %v3455 = stablehlo.reduce(%v3454 init: %v3439) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3456 = stablehlo.broadcast_in_dim %v3455, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3457 = stablehlo.multiply %v3452, %v3454 : tensor<32x384x14x14xf32>
    %v3458 = stablehlo.reduce(%v3457 init: %v3439) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3459 = stablehlo.broadcast_in_dim %v3458, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3460 = stablehlo.multiply %v3454, %v3440 : tensor<32x384x14x14xf32>
    %v3461 = stablehlo.subtract %v3460, %v3456 : tensor<32x384x14x14xf32>
    %v3462 = stablehlo.multiply %v3452, %v3459 : tensor<32x384x14x14xf32>
    %v3463 = stablehlo.subtract %v3461, %v3462 : tensor<32x384x14x14xf32>
    %v3464 = stablehlo.divide %v3451, %v3440 : tensor<32x384x14x14xf32>
    %v3465 = stablehlo.multiply %v3464, %v3463 : tensor<32x384x14x14xf32>
    %v3466 = stablehlo.reshape %v3465 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3467 = stablehlo.reshape %v3466 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3468 = stablehlo.transpose %We9, dims = [1, 0, 2, 3] : (tensor<384x64x1x1xf32>) -> tensor<64x384x1x1xf32>
    %v3469 = stablehlo.reverse %v3468, dims = [2, 3] : tensor<64x384x1x1xf32>
    %v3470 = stablehlo.convolution(%v3467, %v3469)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<64x384x1x1xf32>) -> tensor<32x64x14x14xf32>
    %v3471 = stablehlo.reshape %v3470 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v3472 = stablehlo.add %v3471, %v3247 : tensor<32x12544xf32>
    %v3473 = stablehlo.reshape %v667 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3474 = stablehlo.reshape %v3466 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3475 = stablehlo.transpose %v3473, dims = [1, 0, 2, 3] : (tensor<32x64x14x14xf32>) -> tensor<64x32x14x14xf32>
    %v3476 = stablehlo.transpose %v3474, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3477 = stablehlo.convolution(%v3475, %v3476)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<64x384x1x1xf32>
    %v3478 = stablehlo.transpose %v3477, dims = [1, 0, 2, 3] : (tensor<64x384x1x1xf32>) -> tensor<384x64x1x1xf32>
    %v3479 = stablehlo.constant dense<0.3> : tensor<384x64x1x1xf32>
    %v3480 = stablehlo.multiply %v3478, %v3479 : tensor<384x64x1x1xf32>
    %v3481 = stablehlo.subtract %We9, %v3480 : tensor<384x64x1x1xf32>
    %v3482 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3483 = stablehlo.reshape %v672 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3484 = stablehlo.constant dense<196.0> : tensor<32x384x14x14xf32>
    %v3485 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3486 = stablehlo.reduce(%v3483 init: %v3482) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3487 = stablehlo.broadcast_in_dim %v3486, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3488 = stablehlo.divide %v3487, %v3484 : tensor<32x384x14x14xf32>
    %v3489 = stablehlo.subtract %v3483, %v3488 : tensor<32x384x14x14xf32>
    %v3490 = stablehlo.multiply %v3489, %v3489 : tensor<32x384x14x14xf32>
    %v3491 = stablehlo.reduce(%v3490 init: %v3482) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3492 = stablehlo.broadcast_in_dim %v3491, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3493 = stablehlo.divide %v3492, %v3484 : tensor<32x384x14x14xf32>
    %v3494 = stablehlo.add %v3493, %v3485 : tensor<32x384x14x14xf32>
    %v3495 = stablehlo.rsqrt %v3494 : tensor<32x384x14x14xf32>
    %v3496 = stablehlo.multiply %v3489, %v3495 : tensor<32x384x14x14xf32>
    %v3497 = stablehlo.reshape %v3436 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3498 = stablehlo.multiply %v3497, %v3496 : tensor<32x384x14x14xf32>
    %v3499 = stablehlo.reduce(%v3498 init: %v3482) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3500 = stablehlo.constant dense<0.3> : tensor<384xf32>
    %v3501 = stablehlo.multiply %v3499, %v3500 : tensor<384xf32>
    %v3502 = stablehlo.subtract %ge9, %v3501 : tensor<384xf32>
    %v3503 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3504 = stablehlo.reshape %v3436 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3505 = stablehlo.reduce(%v3504 init: %v3503) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3506 = stablehlo.constant dense<0.3> : tensor<384xf32>
    %v3507 = stablehlo.multiply %v3505, %v3506 : tensor<384xf32>
    %v3508 = stablehlo.subtract %bte9, %v3507 : tensor<384xf32>
    %v3509 = stablehlo.reshape %v696 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3510 = stablehlo.reshape %v3426 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3511 = stablehlo.transpose %v3509, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3512 = stablehlo.transpose %v3510, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3513 = stablehlo.convolution(%v3511, %v3512)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x3x3xf32>
    %v3514 = stablehlo.reshape %v3513 : (tensor<1x384x3x3xf32>) -> tensor<384x1x3x3xf32>
    %v3515 = stablehlo.constant dense<0.3> : tensor<384x1x3x3xf32>
    %v3516 = stablehlo.multiply %v3514, %v3515 : tensor<384x1x3x3xf32>
    %v3517 = stablehlo.subtract %Wd9, %v3516 : tensor<384x1x3x3xf32>
    %v3518 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3519 = stablehlo.reshape %v701 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3520 = stablehlo.constant dense<196.0> : tensor<32x384x14x14xf32>
    %v3521 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3522 = stablehlo.reduce(%v3519 init: %v3518) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3523 = stablehlo.broadcast_in_dim %v3522, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3524 = stablehlo.divide %v3523, %v3520 : tensor<32x384x14x14xf32>
    %v3525 = stablehlo.subtract %v3519, %v3524 : tensor<32x384x14x14xf32>
    %v3526 = stablehlo.multiply %v3525, %v3525 : tensor<32x384x14x14xf32>
    %v3527 = stablehlo.reduce(%v3526 init: %v3518) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3528 = stablehlo.broadcast_in_dim %v3527, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3529 = stablehlo.divide %v3528, %v3520 : tensor<32x384x14x14xf32>
    %v3530 = stablehlo.add %v3529, %v3521 : tensor<32x384x14x14xf32>
    %v3531 = stablehlo.rsqrt %v3530 : tensor<32x384x14x14xf32>
    %v3532 = stablehlo.multiply %v3525, %v3531 : tensor<32x384x14x14xf32>
    %v3533 = stablehlo.reshape %v3396 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3534 = stablehlo.multiply %v3533, %v3532 : tensor<32x384x14x14xf32>
    %v3535 = stablehlo.reduce(%v3534 init: %v3518) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3536 = stablehlo.constant dense<0.3> : tensor<384xf32>
    %v3537 = stablehlo.multiply %v3535, %v3536 : tensor<384xf32>
    %v3538 = stablehlo.subtract %gd9, %v3537 : tensor<384xf32>
    %v3539 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3540 = stablehlo.reshape %v3396 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3541 = stablehlo.reduce(%v3540 init: %v3539) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3542 = stablehlo.constant dense<0.3> : tensor<384xf32>
    %v3543 = stablehlo.multiply %v3541, %v3542 : tensor<384xf32>
    %v3544 = stablehlo.subtract %btd9, %v3543 : tensor<384xf32>
    %v3545 = stablehlo.reshape %v725 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3546 = stablehlo.reshape %v3385 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3547 = stablehlo.transpose %v3545, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3548 = stablehlo.transpose %v3546, dims = [1, 0, 2, 3] : (tensor<32x64x14x14xf32>) -> tensor<64x32x14x14xf32>
    %v3549 = stablehlo.convolution(%v3547, %v3548)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<64x32x14x14xf32>) -> tensor<384x64x1x1xf32>
    %v3550 = stablehlo.transpose %v3549, dims = [1, 0, 2, 3] : (tensor<384x64x1x1xf32>) -> tensor<64x384x1x1xf32>
    %v3551 = stablehlo.constant dense<0.3> : tensor<64x384x1x1xf32>
    %v3552 = stablehlo.multiply %v3550, %v3551 : tensor<64x384x1x1xf32>
    %v3553 = stablehlo.subtract %Wp9, %v3552 : tensor<64x384x1x1xf32>
    %v3554 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3555 = stablehlo.reshape %v730 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3556 = stablehlo.constant dense<196.0> : tensor<32x64x14x14xf32>
    %v3557 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v3558 = stablehlo.reduce(%v3555 init: %v3554) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3559 = stablehlo.broadcast_in_dim %v3558, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v3560 = stablehlo.divide %v3559, %v3556 : tensor<32x64x14x14xf32>
    %v3561 = stablehlo.subtract %v3555, %v3560 : tensor<32x64x14x14xf32>
    %v3562 = stablehlo.multiply %v3561, %v3561 : tensor<32x64x14x14xf32>
    %v3563 = stablehlo.reduce(%v3562 init: %v3554) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3564 = stablehlo.broadcast_in_dim %v3563, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v3565 = stablehlo.divide %v3564, %v3556 : tensor<32x64x14x14xf32>
    %v3566 = stablehlo.add %v3565, %v3557 : tensor<32x64x14x14xf32>
    %v3567 = stablehlo.rsqrt %v3566 : tensor<32x64x14x14xf32>
    %v3568 = stablehlo.multiply %v3561, %v3567 : tensor<32x64x14x14xf32>
    %v3569 = stablehlo.reshape %v3247 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3570 = stablehlo.multiply %v3569, %v3568 : tensor<32x64x14x14xf32>
    %v3571 = stablehlo.reduce(%v3570 init: %v3554) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3572 = stablehlo.constant dense<0.3> : tensor<64xf32>
    %v3573 = stablehlo.multiply %v3571, %v3572 : tensor<64xf32>
    %v3574 = stablehlo.subtract %gp9, %v3573 : tensor<64xf32>
    %v3575 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3576 = stablehlo.reshape %v3247 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3577 = stablehlo.reduce(%v3576 init: %v3575) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3578 = stablehlo.constant dense<0.3> : tensor<64xf32>
    %v3579 = stablehlo.multiply %v3577, %v3578 : tensor<64xf32>
    %v3580 = stablehlo.subtract %btp9, %v3579 : tensor<64xf32>
    %v3581 = stablehlo.reshape %v3472 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3582 = stablehlo.reshape %v646 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3583 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3584 = stablehlo.constant dense<196.0> : tensor<32x64x14x14xf32>
    %v3585 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v3586 = stablehlo.reduce(%v3582 init: %v3583) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3587 = stablehlo.broadcast_in_dim %v3586, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v3588 = stablehlo.divide %v3587, %v3584 : tensor<32x64x14x14xf32>
    %v3589 = stablehlo.subtract %v3582, %v3588 : tensor<32x64x14x14xf32>
    %v3590 = stablehlo.multiply %v3589, %v3589 : tensor<32x64x14x14xf32>
    %v3591 = stablehlo.reduce(%v3590 init: %v3583) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3592 = stablehlo.broadcast_in_dim %v3591, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v3593 = stablehlo.divide %v3592, %v3584 : tensor<32x64x14x14xf32>
    %v3594 = stablehlo.add %v3593, %v3585 : tensor<32x64x14x14xf32>
    %v3595 = stablehlo.rsqrt %v3594 : tensor<32x64x14x14xf32>
    %v3596 = stablehlo.multiply %v3589, %v3595 : tensor<32x64x14x14xf32>
    %v3597 = stablehlo.broadcast_in_dim %gp8, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3598 = stablehlo.multiply %v3597, %v3581 : tensor<32x64x14x14xf32>
    %v3599 = stablehlo.reduce(%v3598 init: %v3583) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3600 = stablehlo.broadcast_in_dim %v3599, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v3601 = stablehlo.multiply %v3596, %v3598 : tensor<32x64x14x14xf32>
    %v3602 = stablehlo.reduce(%v3601 init: %v3583) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3603 = stablehlo.broadcast_in_dim %v3602, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v3604 = stablehlo.multiply %v3598, %v3584 : tensor<32x64x14x14xf32>
    %v3605 = stablehlo.subtract %v3604, %v3600 : tensor<32x64x14x14xf32>
    %v3606 = stablehlo.multiply %v3596, %v3603 : tensor<32x64x14x14xf32>
    %v3607 = stablehlo.subtract %v3605, %v3606 : tensor<32x64x14x14xf32>
    %v3608 = stablehlo.divide %v3595, %v3584 : tensor<32x64x14x14xf32>
    %v3609 = stablehlo.multiply %v3608, %v3607 : tensor<32x64x14x14xf32>
    %v3610 = stablehlo.reshape %v3609 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v3611 = stablehlo.reshape %v3610 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3612 = stablehlo.transpose %Wp8, dims = [1, 0, 2, 3] : (tensor<64x384x1x1xf32>) -> tensor<384x64x1x1xf32>
    %v3613 = stablehlo.reverse %v3612, dims = [2, 3] : tensor<384x64x1x1xf32>
    %v3614 = stablehlo.convolution(%v3611, %v3613)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x14x14xf32>, tensor<384x64x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v3615 = stablehlo.reshape %v3614 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3616 = stablehlo.constant dense<0.0> : tensor<32x75264xf32>
    %v3617 = stablehlo.constant dense<6.0> : tensor<32x75264xf32>
    %v3618 = stablehlo.compare GT, %v637, %v3616 : (tensor<32x75264xf32>, tensor<32x75264xf32>) -> tensor<32x75264xi1>
    %v3619 = stablehlo.compare LT, %v637, %v3617 : (tensor<32x75264xf32>, tensor<32x75264xf32>) -> tensor<32x75264xi1>
    %v3620 = stablehlo.and %v3618, %v3619 : tensor<32x75264xi1>
    %v3621 = stablehlo.select %v3620, %v3615, %v3616 : tensor<32x75264xi1>, tensor<32x75264xf32>
    %v3622 = stablehlo.reshape %v3621 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3623 = stablehlo.reshape %v617 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3624 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3625 = stablehlo.constant dense<196.0> : tensor<32x384x14x14xf32>
    %v3626 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3627 = stablehlo.reduce(%v3623 init: %v3624) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3628 = stablehlo.broadcast_in_dim %v3627, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3629 = stablehlo.divide %v3628, %v3625 : tensor<32x384x14x14xf32>
    %v3630 = stablehlo.subtract %v3623, %v3629 : tensor<32x384x14x14xf32>
    %v3631 = stablehlo.multiply %v3630, %v3630 : tensor<32x384x14x14xf32>
    %v3632 = stablehlo.reduce(%v3631 init: %v3624) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3633 = stablehlo.broadcast_in_dim %v3632, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3634 = stablehlo.divide %v3633, %v3625 : tensor<32x384x14x14xf32>
    %v3635 = stablehlo.add %v3634, %v3626 : tensor<32x384x14x14xf32>
    %v3636 = stablehlo.rsqrt %v3635 : tensor<32x384x14x14xf32>
    %v3637 = stablehlo.multiply %v3630, %v3636 : tensor<32x384x14x14xf32>
    %v3638 = stablehlo.broadcast_in_dim %gd8, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3639 = stablehlo.multiply %v3638, %v3622 : tensor<32x384x14x14xf32>
    %v3640 = stablehlo.reduce(%v3639 init: %v3624) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3641 = stablehlo.broadcast_in_dim %v3640, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3642 = stablehlo.multiply %v3637, %v3639 : tensor<32x384x14x14xf32>
    %v3643 = stablehlo.reduce(%v3642 init: %v3624) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3644 = stablehlo.broadcast_in_dim %v3643, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3645 = stablehlo.multiply %v3639, %v3625 : tensor<32x384x14x14xf32>
    %v3646 = stablehlo.subtract %v3645, %v3641 : tensor<32x384x14x14xf32>
    %v3647 = stablehlo.multiply %v3637, %v3644 : tensor<32x384x14x14xf32>
    %v3648 = stablehlo.subtract %v3646, %v3647 : tensor<32x384x14x14xf32>
    %v3649 = stablehlo.divide %v3636, %v3625 : tensor<32x384x14x14xf32>
    %v3650 = stablehlo.multiply %v3649, %v3648 : tensor<32x384x14x14xf32>
    %v3651 = stablehlo.reshape %v3650 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3652 = stablehlo.reshape %v3651 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3653 = stablehlo.reverse %Wd8, dims = [2, 3] : tensor<384x1x3x3xf32>
    %v3654 = stablehlo.convolution(%v3652, %v3653)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x3x3xf32>) -> tensor<32x384x14x14xf32>
    %v3655 = stablehlo.reshape %v3654 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3656 = stablehlo.constant dense<0.0> : tensor<32x75264xf32>
    %v3657 = stablehlo.constant dense<6.0> : tensor<32x75264xf32>
    %v3658 = stablehlo.compare GT, %v608, %v3656 : (tensor<32x75264xf32>, tensor<32x75264xf32>) -> tensor<32x75264xi1>
    %v3659 = stablehlo.compare LT, %v608, %v3657 : (tensor<32x75264xf32>, tensor<32x75264xf32>) -> tensor<32x75264xi1>
    %v3660 = stablehlo.and %v3658, %v3659 : tensor<32x75264xi1>
    %v3661 = stablehlo.select %v3660, %v3655, %v3656 : tensor<32x75264xi1>, tensor<32x75264xf32>
    %v3662 = stablehlo.reshape %v3661 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3663 = stablehlo.reshape %v588 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3664 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3665 = stablehlo.constant dense<196.0> : tensor<32x384x14x14xf32>
    %v3666 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3667 = stablehlo.reduce(%v3663 init: %v3664) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3668 = stablehlo.broadcast_in_dim %v3667, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3669 = stablehlo.divide %v3668, %v3665 : tensor<32x384x14x14xf32>
    %v3670 = stablehlo.subtract %v3663, %v3669 : tensor<32x384x14x14xf32>
    %v3671 = stablehlo.multiply %v3670, %v3670 : tensor<32x384x14x14xf32>
    %v3672 = stablehlo.reduce(%v3671 init: %v3664) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3673 = stablehlo.broadcast_in_dim %v3672, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3674 = stablehlo.divide %v3673, %v3665 : tensor<32x384x14x14xf32>
    %v3675 = stablehlo.add %v3674, %v3666 : tensor<32x384x14x14xf32>
    %v3676 = stablehlo.rsqrt %v3675 : tensor<32x384x14x14xf32>
    %v3677 = stablehlo.multiply %v3670, %v3676 : tensor<32x384x14x14xf32>
    %v3678 = stablehlo.broadcast_in_dim %ge8, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3679 = stablehlo.multiply %v3678, %v3662 : tensor<32x384x14x14xf32>
    %v3680 = stablehlo.reduce(%v3679 init: %v3664) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3681 = stablehlo.broadcast_in_dim %v3680, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3682 = stablehlo.multiply %v3677, %v3679 : tensor<32x384x14x14xf32>
    %v3683 = stablehlo.reduce(%v3682 init: %v3664) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3684 = stablehlo.broadcast_in_dim %v3683, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3685 = stablehlo.multiply %v3679, %v3665 : tensor<32x384x14x14xf32>
    %v3686 = stablehlo.subtract %v3685, %v3681 : tensor<32x384x14x14xf32>
    %v3687 = stablehlo.multiply %v3677, %v3684 : tensor<32x384x14x14xf32>
    %v3688 = stablehlo.subtract %v3686, %v3687 : tensor<32x384x14x14xf32>
    %v3689 = stablehlo.divide %v3676, %v3665 : tensor<32x384x14x14xf32>
    %v3690 = stablehlo.multiply %v3689, %v3688 : tensor<32x384x14x14xf32>
    %v3691 = stablehlo.reshape %v3690 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3692 = stablehlo.reshape %v3691 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3693 = stablehlo.transpose %We8, dims = [1, 0, 2, 3] : (tensor<384x64x1x1xf32>) -> tensor<64x384x1x1xf32>
    %v3694 = stablehlo.reverse %v3693, dims = [2, 3] : tensor<64x384x1x1xf32>
    %v3695 = stablehlo.convolution(%v3692, %v3694)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<64x384x1x1xf32>) -> tensor<32x64x14x14xf32>
    %v3696 = stablehlo.reshape %v3695 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v3697 = stablehlo.add %v3696, %v3472 : tensor<32x12544xf32>
    %v3698 = stablehlo.reshape %v583 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3699 = stablehlo.reshape %v3691 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3700 = stablehlo.transpose %v3698, dims = [1, 0, 2, 3] : (tensor<32x64x14x14xf32>) -> tensor<64x32x14x14xf32>
    %v3701 = stablehlo.transpose %v3699, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3702 = stablehlo.convolution(%v3700, %v3701)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<64x384x1x1xf32>
    %v3703 = stablehlo.transpose %v3702, dims = [1, 0, 2, 3] : (tensor<64x384x1x1xf32>) -> tensor<384x64x1x1xf32>
    %v3704 = stablehlo.constant dense<0.3> : tensor<384x64x1x1xf32>
    %v3705 = stablehlo.multiply %v3703, %v3704 : tensor<384x64x1x1xf32>
    %v3706 = stablehlo.subtract %We8, %v3705 : tensor<384x64x1x1xf32>
    %v3707 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3708 = stablehlo.reshape %v588 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3709 = stablehlo.constant dense<196.0> : tensor<32x384x14x14xf32>
    %v3710 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3711 = stablehlo.reduce(%v3708 init: %v3707) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3712 = stablehlo.broadcast_in_dim %v3711, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3713 = stablehlo.divide %v3712, %v3709 : tensor<32x384x14x14xf32>
    %v3714 = stablehlo.subtract %v3708, %v3713 : tensor<32x384x14x14xf32>
    %v3715 = stablehlo.multiply %v3714, %v3714 : tensor<32x384x14x14xf32>
    %v3716 = stablehlo.reduce(%v3715 init: %v3707) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3717 = stablehlo.broadcast_in_dim %v3716, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3718 = stablehlo.divide %v3717, %v3709 : tensor<32x384x14x14xf32>
    %v3719 = stablehlo.add %v3718, %v3710 : tensor<32x384x14x14xf32>
    %v3720 = stablehlo.rsqrt %v3719 : tensor<32x384x14x14xf32>
    %v3721 = stablehlo.multiply %v3714, %v3720 : tensor<32x384x14x14xf32>
    %v3722 = stablehlo.reshape %v3661 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3723 = stablehlo.multiply %v3722, %v3721 : tensor<32x384x14x14xf32>
    %v3724 = stablehlo.reduce(%v3723 init: %v3707) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3725 = stablehlo.constant dense<0.3> : tensor<384xf32>
    %v3726 = stablehlo.multiply %v3724, %v3725 : tensor<384xf32>
    %v3727 = stablehlo.subtract %ge8, %v3726 : tensor<384xf32>
    %v3728 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3729 = stablehlo.reshape %v3661 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3730 = stablehlo.reduce(%v3729 init: %v3728) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3731 = stablehlo.constant dense<0.3> : tensor<384xf32>
    %v3732 = stablehlo.multiply %v3730, %v3731 : tensor<384xf32>
    %v3733 = stablehlo.subtract %bte8, %v3732 : tensor<384xf32>
    %v3734 = stablehlo.reshape %v612 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3735 = stablehlo.reshape %v3651 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3736 = stablehlo.transpose %v3734, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3737 = stablehlo.transpose %v3735, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3738 = stablehlo.convolution(%v3736, %v3737)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x3x3xf32>
    %v3739 = stablehlo.reshape %v3738 : (tensor<1x384x3x3xf32>) -> tensor<384x1x3x3xf32>
    %v3740 = stablehlo.constant dense<0.3> : tensor<384x1x3x3xf32>
    %v3741 = stablehlo.multiply %v3739, %v3740 : tensor<384x1x3x3xf32>
    %v3742 = stablehlo.subtract %Wd8, %v3741 : tensor<384x1x3x3xf32>
    %v3743 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3744 = stablehlo.reshape %v617 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3745 = stablehlo.constant dense<196.0> : tensor<32x384x14x14xf32>
    %v3746 = stablehlo.constant dense<1.0e-5> : tensor<32x384x14x14xf32>
    %v3747 = stablehlo.reduce(%v3744 init: %v3743) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3748 = stablehlo.broadcast_in_dim %v3747, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3749 = stablehlo.divide %v3748, %v3745 : tensor<32x384x14x14xf32>
    %v3750 = stablehlo.subtract %v3744, %v3749 : tensor<32x384x14x14xf32>
    %v3751 = stablehlo.multiply %v3750, %v3750 : tensor<32x384x14x14xf32>
    %v3752 = stablehlo.reduce(%v3751 init: %v3743) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384xf32>
    %v3753 = stablehlo.broadcast_in_dim %v3752, dims = [0, 1] : (tensor<32x384xf32>) -> tensor<32x384x14x14xf32>
    %v3754 = stablehlo.divide %v3753, %v3745 : tensor<32x384x14x14xf32>
    %v3755 = stablehlo.add %v3754, %v3746 : tensor<32x384x14x14xf32>
    %v3756 = stablehlo.rsqrt %v3755 : tensor<32x384x14x14xf32>
    %v3757 = stablehlo.multiply %v3750, %v3756 : tensor<32x384x14x14xf32>
    %v3758 = stablehlo.reshape %v3621 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3759 = stablehlo.multiply %v3758, %v3757 : tensor<32x384x14x14xf32>
    %v3760 = stablehlo.reduce(%v3759 init: %v3743) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3761 = stablehlo.constant dense<0.3> : tensor<384xf32>
    %v3762 = stablehlo.multiply %v3760, %v3761 : tensor<384xf32>
    %v3763 = stablehlo.subtract %gd8, %v3762 : tensor<384xf32>
    %v3764 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3765 = stablehlo.reshape %v3621 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3766 = stablehlo.reduce(%v3765 init: %v3764) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3767 = stablehlo.constant dense<0.3> : tensor<384xf32>
    %v3768 = stablehlo.multiply %v3766, %v3767 : tensor<384xf32>
    %v3769 = stablehlo.subtract %btd8, %v3768 : tensor<384xf32>
    %v3770 = stablehlo.reshape %v641 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3771 = stablehlo.reshape %v3610 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3772 = stablehlo.transpose %v3770, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3773 = stablehlo.transpose %v3771, dims = [1, 0, 2, 3] : (tensor<32x64x14x14xf32>) -> tensor<64x32x14x14xf32>
    %v3774 = stablehlo.convolution(%v3772, %v3773)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<64x32x14x14xf32>) -> tensor<384x64x1x1xf32>
    %v3775 = stablehlo.transpose %v3774, dims = [1, 0, 2, 3] : (tensor<384x64x1x1xf32>) -> tensor<64x384x1x1xf32>
    %v3776 = stablehlo.constant dense<0.3> : tensor<64x384x1x1xf32>
    %v3777 = stablehlo.multiply %v3775, %v3776 : tensor<64x384x1x1xf32>
    %v3778 = stablehlo.subtract %Wp8, %v3777 : tensor<64x384x1x1xf32>
    %v3779 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3780 = stablehlo.reshape %v646 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3781 = stablehlo.constant dense<196.0> : tensor<32x64x14x14xf32>
    %v3782 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v3783 = stablehlo.reduce(%v3780 init: %v3779) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3784 = stablehlo.broadcast_in_dim %v3783, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v3785 = stablehlo.divide %v3784, %v3781 : tensor<32x64x14x14xf32>
    %v3786 = stablehlo.subtract %v3780, %v3785 : tensor<32x64x14x14xf32>
    %v3787 = stablehlo.multiply %v3786, %v3786 : tensor<32x64x14x14xf32>
    %v3788 = stablehlo.reduce(%v3787 init: %v3779) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3789 = stablehlo.broadcast_in_dim %v3788, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v3790 = stablehlo.divide %v3789, %v3781 : tensor<32x64x14x14xf32>
    %v3791 = stablehlo.add %v3790, %v3782 : tensor<32x64x14x14xf32>
    %v3792 = stablehlo.rsqrt %v3791 : tensor<32x64x14x14xf32>
    %v3793 = stablehlo.multiply %v3786, %v3792 : tensor<32x64x14x14xf32>
    %v3794 = stablehlo.reshape %v3472 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3795 = stablehlo.multiply %v3794, %v3793 : tensor<32x64x14x14xf32>
    %v3796 = stablehlo.reduce(%v3795 init: %v3779) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3797 = stablehlo.constant dense<0.3> : tensor<64xf32>
    %v3798 = stablehlo.multiply %v3796, %v3797 : tensor<64xf32>
    %v3799 = stablehlo.subtract %gp8, %v3798 : tensor<64xf32>
    %v3800 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3801 = stablehlo.reshape %v3472 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3802 = stablehlo.reduce(%v3801 init: %v3800) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v3803 = stablehlo.constant dense<0.3> : tensor<64xf32>
    %v3804 = stablehlo.multiply %v3802, %v3803 : tensor<64xf32>
    %v3805 = stablehlo.subtract %btp8, %v3804 : tensor<64xf32>
    %v3806 = stablehlo.reshape %v3697 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3807 = stablehlo.reshape %v563 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3808 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3809 = stablehlo.constant dense<196.0> : tensor<32x64x14x14xf32>
    %v3810 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v3811 = stablehlo.reduce(%v3807 init: %v3808) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3812 = stablehlo.broadcast_in_dim %v3811, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v3813 = stablehlo.divide %v3812, %v3809 : tensor<32x64x14x14xf32>
    %v3814 = stablehlo.subtract %v3807, %v3813 : tensor<32x64x14x14xf32>
    %v3815 = stablehlo.multiply %v3814, %v3814 : tensor<32x64x14x14xf32>
    %v3816 = stablehlo.reduce(%v3815 init: %v3808) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3817 = stablehlo.broadcast_in_dim %v3816, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v3818 = stablehlo.divide %v3817, %v3809 : tensor<32x64x14x14xf32>
    %v3819 = stablehlo.add %v3818, %v3810 : tensor<32x64x14x14xf32>
    %v3820 = stablehlo.rsqrt %v3819 : tensor<32x64x14x14xf32>
    %v3821 = stablehlo.multiply %v3814, %v3820 : tensor<32x64x14x14xf32>
    %v3822 = stablehlo.broadcast_in_dim %gp7, dims = [1] : (tensor<64xf32>) -> tensor<32x64x14x14xf32>
    %v3823 = stablehlo.multiply %v3822, %v3806 : tensor<32x64x14x14xf32>
    %v3824 = stablehlo.reduce(%v3823 init: %v3808) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3825 = stablehlo.broadcast_in_dim %v3824, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v3826 = stablehlo.multiply %v3821, %v3823 : tensor<32x64x14x14xf32>
    %v3827 = stablehlo.reduce(%v3826 init: %v3808) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3828 = stablehlo.broadcast_in_dim %v3827, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v3829 = stablehlo.multiply %v3823, %v3809 : tensor<32x64x14x14xf32>
    %v3830 = stablehlo.subtract %v3829, %v3825 : tensor<32x64x14x14xf32>
    %v3831 = stablehlo.multiply %v3821, %v3828 : tensor<32x64x14x14xf32>
    %v3832 = stablehlo.subtract %v3830, %v3831 : tensor<32x64x14x14xf32>
    %v3833 = stablehlo.divide %v3820, %v3809 : tensor<32x64x14x14xf32>
    %v3834 = stablehlo.multiply %v3833, %v3832 : tensor<32x64x14x14xf32>
    %v3835 = stablehlo.reshape %v3834 : (tensor<32x64x14x14xf32>) -> tensor<32x12544xf32>
    %v3836 = stablehlo.reshape %v3835 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v3837 = stablehlo.transpose %Wp7, dims = [1, 0, 2, 3] : (tensor<64x192x1x1xf32>) -> tensor<192x64x1x1xf32>
    %v3838 = stablehlo.reverse %v3837, dims = [2, 3] : tensor<192x64x1x1xf32>
    %v3839 = stablehlo.convolution(%v3836, %v3838)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x14x14xf32>, tensor<192x64x1x1xf32>) -> tensor<32x192x14x14xf32>
    %v3840 = stablehlo.reshape %v3839 : (tensor<32x192x14x14xf32>) -> tensor<32x37632xf32>
    %v3841 = stablehlo.constant dense<0.0> : tensor<32x37632xf32>
    %v3842 = stablehlo.constant dense<6.0> : tensor<32x37632xf32>
    %v3843 = stablehlo.compare GT, %v554, %v3841 : (tensor<32x37632xf32>, tensor<32x37632xf32>) -> tensor<32x37632xi1>
    %v3844 = stablehlo.compare LT, %v554, %v3842 : (tensor<32x37632xf32>, tensor<32x37632xf32>) -> tensor<32x37632xi1>
    %v3845 = stablehlo.and %v3843, %v3844 : tensor<32x37632xi1>
    %v3846 = stablehlo.select %v3845, %v3840, %v3841 : tensor<32x37632xi1>, tensor<32x37632xf32>
    %v3847 = stablehlo.reshape %v3846 : (tensor<32x37632xf32>) -> tensor<32x192x14x14xf32>
    %v3848 = stablehlo.reshape %v534 : (tensor<32x37632xf32>) -> tensor<32x192x14x14xf32>
    %v3849 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3850 = stablehlo.constant dense<196.0> : tensor<32x192x14x14xf32>
    %v3851 = stablehlo.constant dense<1.0e-5> : tensor<32x192x14x14xf32>
    %v3852 = stablehlo.reduce(%v3848 init: %v3849) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v3853 = stablehlo.broadcast_in_dim %v3852, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x14x14xf32>
    %v3854 = stablehlo.divide %v3853, %v3850 : tensor<32x192x14x14xf32>
    %v3855 = stablehlo.subtract %v3848, %v3854 : tensor<32x192x14x14xf32>
    %v3856 = stablehlo.multiply %v3855, %v3855 : tensor<32x192x14x14xf32>
    %v3857 = stablehlo.reduce(%v3856 init: %v3849) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v3858 = stablehlo.broadcast_in_dim %v3857, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x14x14xf32>
    %v3859 = stablehlo.divide %v3858, %v3850 : tensor<32x192x14x14xf32>
    %v3860 = stablehlo.add %v3859, %v3851 : tensor<32x192x14x14xf32>
    %v3861 = stablehlo.rsqrt %v3860 : tensor<32x192x14x14xf32>
    %v3862 = stablehlo.multiply %v3855, %v3861 : tensor<32x192x14x14xf32>
    %v3863 = stablehlo.broadcast_in_dim %gd7, dims = [1] : (tensor<192xf32>) -> tensor<32x192x14x14xf32>
    %v3864 = stablehlo.multiply %v3863, %v3847 : tensor<32x192x14x14xf32>
    %v3865 = stablehlo.reduce(%v3864 init: %v3849) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v3866 = stablehlo.broadcast_in_dim %v3865, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x14x14xf32>
    %v3867 = stablehlo.multiply %v3862, %v3864 : tensor<32x192x14x14xf32>
    %v3868 = stablehlo.reduce(%v3867 init: %v3849) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v3869 = stablehlo.broadcast_in_dim %v3868, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x14x14xf32>
    %v3870 = stablehlo.multiply %v3864, %v3850 : tensor<32x192x14x14xf32>
    %v3871 = stablehlo.subtract %v3870, %v3866 : tensor<32x192x14x14xf32>
    %v3872 = stablehlo.multiply %v3862, %v3869 : tensor<32x192x14x14xf32>
    %v3873 = stablehlo.subtract %v3871, %v3872 : tensor<32x192x14x14xf32>
    %v3874 = stablehlo.divide %v3861, %v3850 : tensor<32x192x14x14xf32>
    %v3875 = stablehlo.multiply %v3874, %v3873 : tensor<32x192x14x14xf32>
    %v3876 = stablehlo.reshape %v3875 : (tensor<32x192x14x14xf32>) -> tensor<32x37632xf32>
    %v3877 = stablehlo.reshape %v3876 : (tensor<32x37632xf32>) -> tensor<32x192x14x14xf32>
    %v3878 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3879 = stablehlo.pad %v3877, %v3878, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<32x192x28x28xf32>
    %v3880 = stablehlo.reverse %Wd7, dims = [2, 3] : tensor<192x1x3x3xf32>
    %v3881 = stablehlo.convolution(%v3879, %v3880)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x3x3xf32>) -> tensor<32x192x28x28xf32>
    %v3882 = stablehlo.reshape %v3881 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v3883 = stablehlo.constant dense<0.0> : tensor<32x150528xf32>
    %v3884 = stablehlo.constant dense<6.0> : tensor<32x150528xf32>
    %v3885 = stablehlo.compare GT, %v525, %v3883 : (tensor<32x150528xf32>, tensor<32x150528xf32>) -> tensor<32x150528xi1>
    %v3886 = stablehlo.compare LT, %v525, %v3884 : (tensor<32x150528xf32>, tensor<32x150528xf32>) -> tensor<32x150528xi1>
    %v3887 = stablehlo.and %v3885, %v3886 : tensor<32x150528xi1>
    %v3888 = stablehlo.select %v3887, %v3882, %v3883 : tensor<32x150528xi1>, tensor<32x150528xf32>
    %v3889 = stablehlo.reshape %v3888 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3890 = stablehlo.reshape %v505 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3891 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3892 = stablehlo.constant dense<784.0> : tensor<32x192x28x28xf32>
    %v3893 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v3894 = stablehlo.reduce(%v3890 init: %v3891) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v3895 = stablehlo.broadcast_in_dim %v3894, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v3896 = stablehlo.divide %v3895, %v3892 : tensor<32x192x28x28xf32>
    %v3897 = stablehlo.subtract %v3890, %v3896 : tensor<32x192x28x28xf32>
    %v3898 = stablehlo.multiply %v3897, %v3897 : tensor<32x192x28x28xf32>
    %v3899 = stablehlo.reduce(%v3898 init: %v3891) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v3900 = stablehlo.broadcast_in_dim %v3899, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v3901 = stablehlo.divide %v3900, %v3892 : tensor<32x192x28x28xf32>
    %v3902 = stablehlo.add %v3901, %v3893 : tensor<32x192x28x28xf32>
    %v3903 = stablehlo.rsqrt %v3902 : tensor<32x192x28x28xf32>
    %v3904 = stablehlo.multiply %v3897, %v3903 : tensor<32x192x28x28xf32>
    %v3905 = stablehlo.broadcast_in_dim %ge7, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v3906 = stablehlo.multiply %v3905, %v3889 : tensor<32x192x28x28xf32>
    %v3907 = stablehlo.reduce(%v3906 init: %v3891) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v3908 = stablehlo.broadcast_in_dim %v3907, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v3909 = stablehlo.multiply %v3904, %v3906 : tensor<32x192x28x28xf32>
    %v3910 = stablehlo.reduce(%v3909 init: %v3891) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v3911 = stablehlo.broadcast_in_dim %v3910, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v3912 = stablehlo.multiply %v3906, %v3892 : tensor<32x192x28x28xf32>
    %v3913 = stablehlo.subtract %v3912, %v3908 : tensor<32x192x28x28xf32>
    %v3914 = stablehlo.multiply %v3904, %v3911 : tensor<32x192x28x28xf32>
    %v3915 = stablehlo.subtract %v3913, %v3914 : tensor<32x192x28x28xf32>
    %v3916 = stablehlo.divide %v3903, %v3892 : tensor<32x192x28x28xf32>
    %v3917 = stablehlo.multiply %v3916, %v3915 : tensor<32x192x28x28xf32>
    %v3918 = stablehlo.reshape %v3917 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v3919 = stablehlo.reshape %v3918 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3920 = stablehlo.transpose %We7, dims = [1, 0, 2, 3] : (tensor<192x32x1x1xf32>) -> tensor<32x192x1x1xf32>
    %v3921 = stablehlo.reverse %v3920, dims = [2, 3] : tensor<32x192x1x1xf32>
    %v3922 = stablehlo.convolution(%v3919, %v3921)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<32x192x1x1xf32>) -> tensor<32x32x28x28xf32>
    %v3923 = stablehlo.reshape %v3922 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v3924 = stablehlo.reshape %v500 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v3925 = stablehlo.reshape %v3918 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3926 = stablehlo.transpose %v3924, dims = [1, 0, 2, 3] : (tensor<32x32x28x28xf32>) -> tensor<32x32x28x28xf32>
    %v3927 = stablehlo.transpose %v3925, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v3928 = stablehlo.convolution(%v3926, %v3927)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<32x192x1x1xf32>
    %v3929 = stablehlo.transpose %v3928, dims = [1, 0, 2, 3] : (tensor<32x192x1x1xf32>) -> tensor<192x32x1x1xf32>
    %v3930 = stablehlo.constant dense<0.3> : tensor<192x32x1x1xf32>
    %v3931 = stablehlo.multiply %v3929, %v3930 : tensor<192x32x1x1xf32>
    %v3932 = stablehlo.subtract %We7, %v3931 : tensor<192x32x1x1xf32>
    %v3933 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3934 = stablehlo.reshape %v505 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3935 = stablehlo.constant dense<784.0> : tensor<32x192x28x28xf32>
    %v3936 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v3937 = stablehlo.reduce(%v3934 init: %v3933) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v3938 = stablehlo.broadcast_in_dim %v3937, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v3939 = stablehlo.divide %v3938, %v3935 : tensor<32x192x28x28xf32>
    %v3940 = stablehlo.subtract %v3934, %v3939 : tensor<32x192x28x28xf32>
    %v3941 = stablehlo.multiply %v3940, %v3940 : tensor<32x192x28x28xf32>
    %v3942 = stablehlo.reduce(%v3941 init: %v3933) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v3943 = stablehlo.broadcast_in_dim %v3942, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v3944 = stablehlo.divide %v3943, %v3935 : tensor<32x192x28x28xf32>
    %v3945 = stablehlo.add %v3944, %v3936 : tensor<32x192x28x28xf32>
    %v3946 = stablehlo.rsqrt %v3945 : tensor<32x192x28x28xf32>
    %v3947 = stablehlo.multiply %v3940, %v3946 : tensor<32x192x28x28xf32>
    %v3948 = stablehlo.reshape %v3888 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3949 = stablehlo.multiply %v3948, %v3947 : tensor<32x192x28x28xf32>
    %v3950 = stablehlo.reduce(%v3949 init: %v3933) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3951 = stablehlo.constant dense<0.3> : tensor<192xf32>
    %v3952 = stablehlo.multiply %v3950, %v3951 : tensor<192xf32>
    %v3953 = stablehlo.subtract %ge7, %v3952 : tensor<192xf32>
    %v3954 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3955 = stablehlo.reshape %v3888 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3956 = stablehlo.reduce(%v3955 init: %v3954) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3957 = stablehlo.constant dense<0.3> : tensor<192xf32>
    %v3958 = stablehlo.multiply %v3956, %v3957 : tensor<192xf32>
    %v3959 = stablehlo.subtract %bte7, %v3958 : tensor<192xf32>
    %v3960 = stablehlo.reshape %v529 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3961 = stablehlo.reshape %v3876 : (tensor<32x37632xf32>) -> tensor<32x192x14x14xf32>
    %v3962 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3963 = stablehlo.pad %v3961, %v3962, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<32x192x28x28xf32>
    %v3964 = stablehlo.transpose %v3960, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v3965 = stablehlo.transpose %v3963, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v3966 = stablehlo.convolution(%v3964, %v3965)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 192 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<1x192x3x3xf32>
    %v3967 = stablehlo.reshape %v3966 : (tensor<1x192x3x3xf32>) -> tensor<192x1x3x3xf32>
    %v3968 = stablehlo.constant dense<0.3> : tensor<192x1x3x3xf32>
    %v3969 = stablehlo.multiply %v3967, %v3968 : tensor<192x1x3x3xf32>
    %v3970 = stablehlo.subtract %Wd7, %v3969 : tensor<192x1x3x3xf32>
    %v3971 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3972 = stablehlo.reshape %v534 : (tensor<32x37632xf32>) -> tensor<32x192x14x14xf32>
    %v3973 = stablehlo.constant dense<196.0> : tensor<32x192x14x14xf32>
    %v3974 = stablehlo.constant dense<1.0e-5> : tensor<32x192x14x14xf32>
    %v3975 = stablehlo.reduce(%v3972 init: %v3971) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v3976 = stablehlo.broadcast_in_dim %v3975, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x14x14xf32>
    %v3977 = stablehlo.divide %v3976, %v3973 : tensor<32x192x14x14xf32>
    %v3978 = stablehlo.subtract %v3972, %v3977 : tensor<32x192x14x14xf32>
    %v3979 = stablehlo.multiply %v3978, %v3978 : tensor<32x192x14x14xf32>
    %v3980 = stablehlo.reduce(%v3979 init: %v3971) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v3981 = stablehlo.broadcast_in_dim %v3980, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x14x14xf32>
    %v3982 = stablehlo.divide %v3981, %v3973 : tensor<32x192x14x14xf32>
    %v3983 = stablehlo.add %v3982, %v3974 : tensor<32x192x14x14xf32>
    %v3984 = stablehlo.rsqrt %v3983 : tensor<32x192x14x14xf32>
    %v3985 = stablehlo.multiply %v3978, %v3984 : tensor<32x192x14x14xf32>
    %v3986 = stablehlo.reshape %v3846 : (tensor<32x37632xf32>) -> tensor<32x192x14x14xf32>
    %v3987 = stablehlo.multiply %v3986, %v3985 : tensor<32x192x14x14xf32>
    %v3988 = stablehlo.reduce(%v3987 init: %v3971) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<192xf32>
    %v3989 = stablehlo.constant dense<0.3> : tensor<192xf32>
    %v3990 = stablehlo.multiply %v3988, %v3989 : tensor<192xf32>
    %v3991 = stablehlo.subtract %gd7, %v3990 : tensor<192xf32>
    %v3992 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3993 = stablehlo.reshape %v3846 : (tensor<32x37632xf32>) -> tensor<32x192x14x14xf32>
    %v3994 = stablehlo.reduce(%v3993 init: %v3992) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x14x14xf32>, tensor<f32>) -> tensor<192xf32>
    %v3995 = stablehlo.constant dense<0.3> : tensor<192xf32>
    %v3996 = stablehlo.multiply %v3994, %v3995 : tensor<192xf32>
    %v3997 = stablehlo.subtract %btd7, %v3996 : tensor<192xf32>
    %v3998 = stablehlo.reshape %v558 : (tensor<32x37632xf32>) -> tensor<32x192x14x14xf32>
    %v3999 = stablehlo.reshape %v3835 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v4000 = stablehlo.transpose %v3998, dims = [1, 0, 2, 3] : (tensor<32x192x14x14xf32>) -> tensor<192x32x14x14xf32>
    %v4001 = stablehlo.transpose %v3999, dims = [1, 0, 2, 3] : (tensor<32x64x14x14xf32>) -> tensor<64x32x14x14xf32>
    %v4002 = stablehlo.convolution(%v4000, %v4001)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x14x14xf32>, tensor<64x32x14x14xf32>) -> tensor<192x64x1x1xf32>
    %v4003 = stablehlo.transpose %v4002, dims = [1, 0, 2, 3] : (tensor<192x64x1x1xf32>) -> tensor<64x192x1x1xf32>
    %v4004 = stablehlo.constant dense<0.3> : tensor<64x192x1x1xf32>
    %v4005 = stablehlo.multiply %v4003, %v4004 : tensor<64x192x1x1xf32>
    %v4006 = stablehlo.subtract %Wp7, %v4005 : tensor<64x192x1x1xf32>
    %v4007 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4008 = stablehlo.reshape %v563 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v4009 = stablehlo.constant dense<196.0> : tensor<32x64x14x14xf32>
    %v4010 = stablehlo.constant dense<1.0e-5> : tensor<32x64x14x14xf32>
    %v4011 = stablehlo.reduce(%v4008 init: %v4007) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v4012 = stablehlo.broadcast_in_dim %v4011, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v4013 = stablehlo.divide %v4012, %v4009 : tensor<32x64x14x14xf32>
    %v4014 = stablehlo.subtract %v4008, %v4013 : tensor<32x64x14x14xf32>
    %v4015 = stablehlo.multiply %v4014, %v4014 : tensor<32x64x14x14xf32>
    %v4016 = stablehlo.reduce(%v4015 init: %v4007) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v4017 = stablehlo.broadcast_in_dim %v4016, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x14x14xf32>
    %v4018 = stablehlo.divide %v4017, %v4009 : tensor<32x64x14x14xf32>
    %v4019 = stablehlo.add %v4018, %v4010 : tensor<32x64x14x14xf32>
    %v4020 = stablehlo.rsqrt %v4019 : tensor<32x64x14x14xf32>
    %v4021 = stablehlo.multiply %v4014, %v4020 : tensor<32x64x14x14xf32>
    %v4022 = stablehlo.reshape %v3697 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v4023 = stablehlo.multiply %v4022, %v4021 : tensor<32x64x14x14xf32>
    %v4024 = stablehlo.reduce(%v4023 init: %v4007) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v4025 = stablehlo.constant dense<0.3> : tensor<64xf32>
    %v4026 = stablehlo.multiply %v4024, %v4025 : tensor<64xf32>
    %v4027 = stablehlo.subtract %gp7, %v4026 : tensor<64xf32>
    %v4028 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4029 = stablehlo.reshape %v3697 : (tensor<32x12544xf32>) -> tensor<32x64x14x14xf32>
    %v4030 = stablehlo.reduce(%v4029 init: %v4028) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x14x14xf32>, tensor<f32>) -> tensor<64xf32>
    %v4031 = stablehlo.constant dense<0.3> : tensor<64xf32>
    %v4032 = stablehlo.multiply %v4030, %v4031 : tensor<64xf32>
    %v4033 = stablehlo.subtract %btp7, %v4032 : tensor<64xf32>
    %v4034 = stablehlo.reshape %v3923 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4035 = stablehlo.reshape %v479 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4036 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4037 = stablehlo.constant dense<784.0> : tensor<32x32x28x28xf32>
    %v4038 = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %v4039 = stablehlo.reduce(%v4035 init: %v4036) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v4040 = stablehlo.broadcast_in_dim %v4039, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v4041 = stablehlo.divide %v4040, %v4037 : tensor<32x32x28x28xf32>
    %v4042 = stablehlo.subtract %v4035, %v4041 : tensor<32x32x28x28xf32>
    %v4043 = stablehlo.multiply %v4042, %v4042 : tensor<32x32x28x28xf32>
    %v4044 = stablehlo.reduce(%v4043 init: %v4036) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v4045 = stablehlo.broadcast_in_dim %v4044, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v4046 = stablehlo.divide %v4045, %v4037 : tensor<32x32x28x28xf32>
    %v4047 = stablehlo.add %v4046, %v4038 : tensor<32x32x28x28xf32>
    %v4048 = stablehlo.rsqrt %v4047 : tensor<32x32x28x28xf32>
    %v4049 = stablehlo.multiply %v4042, %v4048 : tensor<32x32x28x28xf32>
    %v4050 = stablehlo.broadcast_in_dim %gp6, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v4051 = stablehlo.multiply %v4050, %v4034 : tensor<32x32x28x28xf32>
    %v4052 = stablehlo.reduce(%v4051 init: %v4036) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v4053 = stablehlo.broadcast_in_dim %v4052, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v4054 = stablehlo.multiply %v4049, %v4051 : tensor<32x32x28x28xf32>
    %v4055 = stablehlo.reduce(%v4054 init: %v4036) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v4056 = stablehlo.broadcast_in_dim %v4055, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v4057 = stablehlo.multiply %v4051, %v4037 : tensor<32x32x28x28xf32>
    %v4058 = stablehlo.subtract %v4057, %v4053 : tensor<32x32x28x28xf32>
    %v4059 = stablehlo.multiply %v4049, %v4056 : tensor<32x32x28x28xf32>
    %v4060 = stablehlo.subtract %v4058, %v4059 : tensor<32x32x28x28xf32>
    %v4061 = stablehlo.divide %v4048, %v4037 : tensor<32x32x28x28xf32>
    %v4062 = stablehlo.multiply %v4061, %v4060 : tensor<32x32x28x28xf32>
    %v4063 = stablehlo.reshape %v4062 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v4064 = stablehlo.reshape %v4063 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4065 = stablehlo.transpose %Wp6, dims = [1, 0, 2, 3] : (tensor<32x192x1x1xf32>) -> tensor<192x32x1x1xf32>
    %v4066 = stablehlo.reverse %v4065, dims = [2, 3] : tensor<192x32x1x1xf32>
    %v4067 = stablehlo.convolution(%v4064, %v4066)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<192x32x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v4068 = stablehlo.reshape %v4067 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v4069 = stablehlo.constant dense<0.0> : tensor<32x150528xf32>
    %v4070 = stablehlo.constant dense<6.0> : tensor<32x150528xf32>
    %v4071 = stablehlo.compare GT, %v470, %v4069 : (tensor<32x150528xf32>, tensor<32x150528xf32>) -> tensor<32x150528xi1>
    %v4072 = stablehlo.compare LT, %v470, %v4070 : (tensor<32x150528xf32>, tensor<32x150528xf32>) -> tensor<32x150528xi1>
    %v4073 = stablehlo.and %v4071, %v4072 : tensor<32x150528xi1>
    %v4074 = stablehlo.select %v4073, %v4068, %v4069 : tensor<32x150528xi1>, tensor<32x150528xf32>
    %v4075 = stablehlo.reshape %v4074 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4076 = stablehlo.reshape %v450 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4077 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4078 = stablehlo.constant dense<784.0> : tensor<32x192x28x28xf32>
    %v4079 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v4080 = stablehlo.reduce(%v4076 init: %v4077) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4081 = stablehlo.broadcast_in_dim %v4080, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v4082 = stablehlo.divide %v4081, %v4078 : tensor<32x192x28x28xf32>
    %v4083 = stablehlo.subtract %v4076, %v4082 : tensor<32x192x28x28xf32>
    %v4084 = stablehlo.multiply %v4083, %v4083 : tensor<32x192x28x28xf32>
    %v4085 = stablehlo.reduce(%v4084 init: %v4077) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4086 = stablehlo.broadcast_in_dim %v4085, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v4087 = stablehlo.divide %v4086, %v4078 : tensor<32x192x28x28xf32>
    %v4088 = stablehlo.add %v4087, %v4079 : tensor<32x192x28x28xf32>
    %v4089 = stablehlo.rsqrt %v4088 : tensor<32x192x28x28xf32>
    %v4090 = stablehlo.multiply %v4083, %v4089 : tensor<32x192x28x28xf32>
    %v4091 = stablehlo.broadcast_in_dim %gd6, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v4092 = stablehlo.multiply %v4091, %v4075 : tensor<32x192x28x28xf32>
    %v4093 = stablehlo.reduce(%v4092 init: %v4077) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4094 = stablehlo.broadcast_in_dim %v4093, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v4095 = stablehlo.multiply %v4090, %v4092 : tensor<32x192x28x28xf32>
    %v4096 = stablehlo.reduce(%v4095 init: %v4077) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4097 = stablehlo.broadcast_in_dim %v4096, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v4098 = stablehlo.multiply %v4092, %v4078 : tensor<32x192x28x28xf32>
    %v4099 = stablehlo.subtract %v4098, %v4094 : tensor<32x192x28x28xf32>
    %v4100 = stablehlo.multiply %v4090, %v4097 : tensor<32x192x28x28xf32>
    %v4101 = stablehlo.subtract %v4099, %v4100 : tensor<32x192x28x28xf32>
    %v4102 = stablehlo.divide %v4089, %v4078 : tensor<32x192x28x28xf32>
    %v4103 = stablehlo.multiply %v4102, %v4101 : tensor<32x192x28x28xf32>
    %v4104 = stablehlo.reshape %v4103 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v4105 = stablehlo.reshape %v4104 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4106 = stablehlo.reverse %Wd6, dims = [2, 3] : tensor<192x1x3x3xf32>
    %v4107 = stablehlo.convolution(%v4105, %v4106)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x3x3xf32>) -> tensor<32x192x28x28xf32>
    %v4108 = stablehlo.reshape %v4107 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v4109 = stablehlo.constant dense<0.0> : tensor<32x150528xf32>
    %v4110 = stablehlo.constant dense<6.0> : tensor<32x150528xf32>
    %v4111 = stablehlo.compare GT, %v441, %v4109 : (tensor<32x150528xf32>, tensor<32x150528xf32>) -> tensor<32x150528xi1>
    %v4112 = stablehlo.compare LT, %v441, %v4110 : (tensor<32x150528xf32>, tensor<32x150528xf32>) -> tensor<32x150528xi1>
    %v4113 = stablehlo.and %v4111, %v4112 : tensor<32x150528xi1>
    %v4114 = stablehlo.select %v4113, %v4108, %v4109 : tensor<32x150528xi1>, tensor<32x150528xf32>
    %v4115 = stablehlo.reshape %v4114 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4116 = stablehlo.reshape %v421 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4117 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4118 = stablehlo.constant dense<784.0> : tensor<32x192x28x28xf32>
    %v4119 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v4120 = stablehlo.reduce(%v4116 init: %v4117) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4121 = stablehlo.broadcast_in_dim %v4120, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v4122 = stablehlo.divide %v4121, %v4118 : tensor<32x192x28x28xf32>
    %v4123 = stablehlo.subtract %v4116, %v4122 : tensor<32x192x28x28xf32>
    %v4124 = stablehlo.multiply %v4123, %v4123 : tensor<32x192x28x28xf32>
    %v4125 = stablehlo.reduce(%v4124 init: %v4117) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4126 = stablehlo.broadcast_in_dim %v4125, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v4127 = stablehlo.divide %v4126, %v4118 : tensor<32x192x28x28xf32>
    %v4128 = stablehlo.add %v4127, %v4119 : tensor<32x192x28x28xf32>
    %v4129 = stablehlo.rsqrt %v4128 : tensor<32x192x28x28xf32>
    %v4130 = stablehlo.multiply %v4123, %v4129 : tensor<32x192x28x28xf32>
    %v4131 = stablehlo.broadcast_in_dim %ge6, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v4132 = stablehlo.multiply %v4131, %v4115 : tensor<32x192x28x28xf32>
    %v4133 = stablehlo.reduce(%v4132 init: %v4117) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4134 = stablehlo.broadcast_in_dim %v4133, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v4135 = stablehlo.multiply %v4130, %v4132 : tensor<32x192x28x28xf32>
    %v4136 = stablehlo.reduce(%v4135 init: %v4117) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4137 = stablehlo.broadcast_in_dim %v4136, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v4138 = stablehlo.multiply %v4132, %v4118 : tensor<32x192x28x28xf32>
    %v4139 = stablehlo.subtract %v4138, %v4134 : tensor<32x192x28x28xf32>
    %v4140 = stablehlo.multiply %v4130, %v4137 : tensor<32x192x28x28xf32>
    %v4141 = stablehlo.subtract %v4139, %v4140 : tensor<32x192x28x28xf32>
    %v4142 = stablehlo.divide %v4129, %v4118 : tensor<32x192x28x28xf32>
    %v4143 = stablehlo.multiply %v4142, %v4141 : tensor<32x192x28x28xf32>
    %v4144 = stablehlo.reshape %v4143 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v4145 = stablehlo.reshape %v4144 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4146 = stablehlo.transpose %We6, dims = [1, 0, 2, 3] : (tensor<192x32x1x1xf32>) -> tensor<32x192x1x1xf32>
    %v4147 = stablehlo.reverse %v4146, dims = [2, 3] : tensor<32x192x1x1xf32>
    %v4148 = stablehlo.convolution(%v4145, %v4147)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<32x192x1x1xf32>) -> tensor<32x32x28x28xf32>
    %v4149 = stablehlo.reshape %v4148 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v4150 = stablehlo.add %v4149, %v3923 : tensor<32x25088xf32>
    %v4151 = stablehlo.reshape %v416 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4152 = stablehlo.reshape %v4144 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4153 = stablehlo.transpose %v4151, dims = [1, 0, 2, 3] : (tensor<32x32x28x28xf32>) -> tensor<32x32x28x28xf32>
    %v4154 = stablehlo.transpose %v4152, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v4155 = stablehlo.convolution(%v4153, %v4154)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<32x192x1x1xf32>
    %v4156 = stablehlo.transpose %v4155, dims = [1, 0, 2, 3] : (tensor<32x192x1x1xf32>) -> tensor<192x32x1x1xf32>
    %v4157 = stablehlo.constant dense<0.3> : tensor<192x32x1x1xf32>
    %v4158 = stablehlo.multiply %v4156, %v4157 : tensor<192x32x1x1xf32>
    %v4159 = stablehlo.subtract %We6, %v4158 : tensor<192x32x1x1xf32>
    %v4160 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4161 = stablehlo.reshape %v421 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4162 = stablehlo.constant dense<784.0> : tensor<32x192x28x28xf32>
    %v4163 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v4164 = stablehlo.reduce(%v4161 init: %v4160) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4165 = stablehlo.broadcast_in_dim %v4164, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v4166 = stablehlo.divide %v4165, %v4162 : tensor<32x192x28x28xf32>
    %v4167 = stablehlo.subtract %v4161, %v4166 : tensor<32x192x28x28xf32>
    %v4168 = stablehlo.multiply %v4167, %v4167 : tensor<32x192x28x28xf32>
    %v4169 = stablehlo.reduce(%v4168 init: %v4160) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4170 = stablehlo.broadcast_in_dim %v4169, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v4171 = stablehlo.divide %v4170, %v4162 : tensor<32x192x28x28xf32>
    %v4172 = stablehlo.add %v4171, %v4163 : tensor<32x192x28x28xf32>
    %v4173 = stablehlo.rsqrt %v4172 : tensor<32x192x28x28xf32>
    %v4174 = stablehlo.multiply %v4167, %v4173 : tensor<32x192x28x28xf32>
    %v4175 = stablehlo.reshape %v4114 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4176 = stablehlo.multiply %v4175, %v4174 : tensor<32x192x28x28xf32>
    %v4177 = stablehlo.reduce(%v4176 init: %v4160) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4178 = stablehlo.constant dense<0.3> : tensor<192xf32>
    %v4179 = stablehlo.multiply %v4177, %v4178 : tensor<192xf32>
    %v4180 = stablehlo.subtract %ge6, %v4179 : tensor<192xf32>
    %v4181 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4182 = stablehlo.reshape %v4114 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4183 = stablehlo.reduce(%v4182 init: %v4181) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4184 = stablehlo.constant dense<0.3> : tensor<192xf32>
    %v4185 = stablehlo.multiply %v4183, %v4184 : tensor<192xf32>
    %v4186 = stablehlo.subtract %bte6, %v4185 : tensor<192xf32>
    %v4187 = stablehlo.reshape %v445 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4188 = stablehlo.reshape %v4104 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4189 = stablehlo.transpose %v4187, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v4190 = stablehlo.transpose %v4188, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v4191 = stablehlo.convolution(%v4189, %v4190)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 192 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<1x192x3x3xf32>
    %v4192 = stablehlo.reshape %v4191 : (tensor<1x192x3x3xf32>) -> tensor<192x1x3x3xf32>
    %v4193 = stablehlo.constant dense<0.3> : tensor<192x1x3x3xf32>
    %v4194 = stablehlo.multiply %v4192, %v4193 : tensor<192x1x3x3xf32>
    %v4195 = stablehlo.subtract %Wd6, %v4194 : tensor<192x1x3x3xf32>
    %v4196 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4197 = stablehlo.reshape %v450 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4198 = stablehlo.constant dense<784.0> : tensor<32x192x28x28xf32>
    %v4199 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v4200 = stablehlo.reduce(%v4197 init: %v4196) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4201 = stablehlo.broadcast_in_dim %v4200, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v4202 = stablehlo.divide %v4201, %v4198 : tensor<32x192x28x28xf32>
    %v4203 = stablehlo.subtract %v4197, %v4202 : tensor<32x192x28x28xf32>
    %v4204 = stablehlo.multiply %v4203, %v4203 : tensor<32x192x28x28xf32>
    %v4205 = stablehlo.reduce(%v4204 init: %v4196) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4206 = stablehlo.broadcast_in_dim %v4205, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v4207 = stablehlo.divide %v4206, %v4198 : tensor<32x192x28x28xf32>
    %v4208 = stablehlo.add %v4207, %v4199 : tensor<32x192x28x28xf32>
    %v4209 = stablehlo.rsqrt %v4208 : tensor<32x192x28x28xf32>
    %v4210 = stablehlo.multiply %v4203, %v4209 : tensor<32x192x28x28xf32>
    %v4211 = stablehlo.reshape %v4074 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4212 = stablehlo.multiply %v4211, %v4210 : tensor<32x192x28x28xf32>
    %v4213 = stablehlo.reduce(%v4212 init: %v4196) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4214 = stablehlo.constant dense<0.3> : tensor<192xf32>
    %v4215 = stablehlo.multiply %v4213, %v4214 : tensor<192xf32>
    %v4216 = stablehlo.subtract %gd6, %v4215 : tensor<192xf32>
    %v4217 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4218 = stablehlo.reshape %v4074 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4219 = stablehlo.reduce(%v4218 init: %v4217) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4220 = stablehlo.constant dense<0.3> : tensor<192xf32>
    %v4221 = stablehlo.multiply %v4219, %v4220 : tensor<192xf32>
    %v4222 = stablehlo.subtract %btd6, %v4221 : tensor<192xf32>
    %v4223 = stablehlo.reshape %v474 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4224 = stablehlo.reshape %v4063 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4225 = stablehlo.transpose %v4223, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v4226 = stablehlo.transpose %v4224, dims = [1, 0, 2, 3] : (tensor<32x32x28x28xf32>) -> tensor<32x32x28x28xf32>
    %v4227 = stablehlo.convolution(%v4225, %v4226)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<32x32x28x28xf32>) -> tensor<192x32x1x1xf32>
    %v4228 = stablehlo.transpose %v4227, dims = [1, 0, 2, 3] : (tensor<192x32x1x1xf32>) -> tensor<32x192x1x1xf32>
    %v4229 = stablehlo.constant dense<0.3> : tensor<32x192x1x1xf32>
    %v4230 = stablehlo.multiply %v4228, %v4229 : tensor<32x192x1x1xf32>
    %v4231 = stablehlo.subtract %Wp6, %v4230 : tensor<32x192x1x1xf32>
    %v4232 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4233 = stablehlo.reshape %v479 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4234 = stablehlo.constant dense<784.0> : tensor<32x32x28x28xf32>
    %v4235 = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %v4236 = stablehlo.reduce(%v4233 init: %v4232) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v4237 = stablehlo.broadcast_in_dim %v4236, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v4238 = stablehlo.divide %v4237, %v4234 : tensor<32x32x28x28xf32>
    %v4239 = stablehlo.subtract %v4233, %v4238 : tensor<32x32x28x28xf32>
    %v4240 = stablehlo.multiply %v4239, %v4239 : tensor<32x32x28x28xf32>
    %v4241 = stablehlo.reduce(%v4240 init: %v4232) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v4242 = stablehlo.broadcast_in_dim %v4241, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v4243 = stablehlo.divide %v4242, %v4234 : tensor<32x32x28x28xf32>
    %v4244 = stablehlo.add %v4243, %v4235 : tensor<32x32x28x28xf32>
    %v4245 = stablehlo.rsqrt %v4244 : tensor<32x32x28x28xf32>
    %v4246 = stablehlo.multiply %v4239, %v4245 : tensor<32x32x28x28xf32>
    %v4247 = stablehlo.reshape %v3923 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4248 = stablehlo.multiply %v4247, %v4246 : tensor<32x32x28x28xf32>
    %v4249 = stablehlo.reduce(%v4248 init: %v4232) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v4250 = stablehlo.constant dense<0.3> : tensor<32xf32>
    %v4251 = stablehlo.multiply %v4249, %v4250 : tensor<32xf32>
    %v4252 = stablehlo.subtract %gp6, %v4251 : tensor<32xf32>
    %v4253 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4254 = stablehlo.reshape %v3923 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4255 = stablehlo.reduce(%v4254 init: %v4253) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v4256 = stablehlo.constant dense<0.3> : tensor<32xf32>
    %v4257 = stablehlo.multiply %v4255, %v4256 : tensor<32xf32>
    %v4258 = stablehlo.subtract %btp6, %v4257 : tensor<32xf32>
    %v4259 = stablehlo.reshape %v4150 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4260 = stablehlo.reshape %v395 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4261 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4262 = stablehlo.constant dense<784.0> : tensor<32x32x28x28xf32>
    %v4263 = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %v4264 = stablehlo.reduce(%v4260 init: %v4261) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v4265 = stablehlo.broadcast_in_dim %v4264, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v4266 = stablehlo.divide %v4265, %v4262 : tensor<32x32x28x28xf32>
    %v4267 = stablehlo.subtract %v4260, %v4266 : tensor<32x32x28x28xf32>
    %v4268 = stablehlo.multiply %v4267, %v4267 : tensor<32x32x28x28xf32>
    %v4269 = stablehlo.reduce(%v4268 init: %v4261) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v4270 = stablehlo.broadcast_in_dim %v4269, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v4271 = stablehlo.divide %v4270, %v4262 : tensor<32x32x28x28xf32>
    %v4272 = stablehlo.add %v4271, %v4263 : tensor<32x32x28x28xf32>
    %v4273 = stablehlo.rsqrt %v4272 : tensor<32x32x28x28xf32>
    %v4274 = stablehlo.multiply %v4267, %v4273 : tensor<32x32x28x28xf32>
    %v4275 = stablehlo.broadcast_in_dim %gp5, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v4276 = stablehlo.multiply %v4275, %v4259 : tensor<32x32x28x28xf32>
    %v4277 = stablehlo.reduce(%v4276 init: %v4261) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v4278 = stablehlo.broadcast_in_dim %v4277, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v4279 = stablehlo.multiply %v4274, %v4276 : tensor<32x32x28x28xf32>
    %v4280 = stablehlo.reduce(%v4279 init: %v4261) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v4281 = stablehlo.broadcast_in_dim %v4280, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v4282 = stablehlo.multiply %v4276, %v4262 : tensor<32x32x28x28xf32>
    %v4283 = stablehlo.subtract %v4282, %v4278 : tensor<32x32x28x28xf32>
    %v4284 = stablehlo.multiply %v4274, %v4281 : tensor<32x32x28x28xf32>
    %v4285 = stablehlo.subtract %v4283, %v4284 : tensor<32x32x28x28xf32>
    %v4286 = stablehlo.divide %v4273, %v4262 : tensor<32x32x28x28xf32>
    %v4287 = stablehlo.multiply %v4286, %v4285 : tensor<32x32x28x28xf32>
    %v4288 = stablehlo.reshape %v4287 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v4289 = stablehlo.reshape %v4288 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4290 = stablehlo.transpose %Wp5, dims = [1, 0, 2, 3] : (tensor<32x192x1x1xf32>) -> tensor<192x32x1x1xf32>
    %v4291 = stablehlo.reverse %v4290, dims = [2, 3] : tensor<192x32x1x1xf32>
    %v4292 = stablehlo.convolution(%v4289, %v4291)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<192x32x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v4293 = stablehlo.reshape %v4292 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v4294 = stablehlo.constant dense<0.0> : tensor<32x150528xf32>
    %v4295 = stablehlo.constant dense<6.0> : tensor<32x150528xf32>
    %v4296 = stablehlo.compare GT, %v386, %v4294 : (tensor<32x150528xf32>, tensor<32x150528xf32>) -> tensor<32x150528xi1>
    %v4297 = stablehlo.compare LT, %v386, %v4295 : (tensor<32x150528xf32>, tensor<32x150528xf32>) -> tensor<32x150528xi1>
    %v4298 = stablehlo.and %v4296, %v4297 : tensor<32x150528xi1>
    %v4299 = stablehlo.select %v4298, %v4293, %v4294 : tensor<32x150528xi1>, tensor<32x150528xf32>
    %v4300 = stablehlo.reshape %v4299 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4301 = stablehlo.reshape %v366 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4302 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4303 = stablehlo.constant dense<784.0> : tensor<32x192x28x28xf32>
    %v4304 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v4305 = stablehlo.reduce(%v4301 init: %v4302) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4306 = stablehlo.broadcast_in_dim %v4305, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v4307 = stablehlo.divide %v4306, %v4303 : tensor<32x192x28x28xf32>
    %v4308 = stablehlo.subtract %v4301, %v4307 : tensor<32x192x28x28xf32>
    %v4309 = stablehlo.multiply %v4308, %v4308 : tensor<32x192x28x28xf32>
    %v4310 = stablehlo.reduce(%v4309 init: %v4302) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4311 = stablehlo.broadcast_in_dim %v4310, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v4312 = stablehlo.divide %v4311, %v4303 : tensor<32x192x28x28xf32>
    %v4313 = stablehlo.add %v4312, %v4304 : tensor<32x192x28x28xf32>
    %v4314 = stablehlo.rsqrt %v4313 : tensor<32x192x28x28xf32>
    %v4315 = stablehlo.multiply %v4308, %v4314 : tensor<32x192x28x28xf32>
    %v4316 = stablehlo.broadcast_in_dim %gd5, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v4317 = stablehlo.multiply %v4316, %v4300 : tensor<32x192x28x28xf32>
    %v4318 = stablehlo.reduce(%v4317 init: %v4302) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4319 = stablehlo.broadcast_in_dim %v4318, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v4320 = stablehlo.multiply %v4315, %v4317 : tensor<32x192x28x28xf32>
    %v4321 = stablehlo.reduce(%v4320 init: %v4302) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4322 = stablehlo.broadcast_in_dim %v4321, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v4323 = stablehlo.multiply %v4317, %v4303 : tensor<32x192x28x28xf32>
    %v4324 = stablehlo.subtract %v4323, %v4319 : tensor<32x192x28x28xf32>
    %v4325 = stablehlo.multiply %v4315, %v4322 : tensor<32x192x28x28xf32>
    %v4326 = stablehlo.subtract %v4324, %v4325 : tensor<32x192x28x28xf32>
    %v4327 = stablehlo.divide %v4314, %v4303 : tensor<32x192x28x28xf32>
    %v4328 = stablehlo.multiply %v4327, %v4326 : tensor<32x192x28x28xf32>
    %v4329 = stablehlo.reshape %v4328 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v4330 = stablehlo.reshape %v4329 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4331 = stablehlo.reverse %Wd5, dims = [2, 3] : tensor<192x1x3x3xf32>
    %v4332 = stablehlo.convolution(%v4330, %v4331)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x3x3xf32>) -> tensor<32x192x28x28xf32>
    %v4333 = stablehlo.reshape %v4332 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v4334 = stablehlo.constant dense<0.0> : tensor<32x150528xf32>
    %v4335 = stablehlo.constant dense<6.0> : tensor<32x150528xf32>
    %v4336 = stablehlo.compare GT, %v357, %v4334 : (tensor<32x150528xf32>, tensor<32x150528xf32>) -> tensor<32x150528xi1>
    %v4337 = stablehlo.compare LT, %v357, %v4335 : (tensor<32x150528xf32>, tensor<32x150528xf32>) -> tensor<32x150528xi1>
    %v4338 = stablehlo.and %v4336, %v4337 : tensor<32x150528xi1>
    %v4339 = stablehlo.select %v4338, %v4333, %v4334 : tensor<32x150528xi1>, tensor<32x150528xf32>
    %v4340 = stablehlo.reshape %v4339 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4341 = stablehlo.reshape %v337 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4342 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4343 = stablehlo.constant dense<784.0> : tensor<32x192x28x28xf32>
    %v4344 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v4345 = stablehlo.reduce(%v4341 init: %v4342) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4346 = stablehlo.broadcast_in_dim %v4345, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v4347 = stablehlo.divide %v4346, %v4343 : tensor<32x192x28x28xf32>
    %v4348 = stablehlo.subtract %v4341, %v4347 : tensor<32x192x28x28xf32>
    %v4349 = stablehlo.multiply %v4348, %v4348 : tensor<32x192x28x28xf32>
    %v4350 = stablehlo.reduce(%v4349 init: %v4342) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4351 = stablehlo.broadcast_in_dim %v4350, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v4352 = stablehlo.divide %v4351, %v4343 : tensor<32x192x28x28xf32>
    %v4353 = stablehlo.add %v4352, %v4344 : tensor<32x192x28x28xf32>
    %v4354 = stablehlo.rsqrt %v4353 : tensor<32x192x28x28xf32>
    %v4355 = stablehlo.multiply %v4348, %v4354 : tensor<32x192x28x28xf32>
    %v4356 = stablehlo.broadcast_in_dim %ge5, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v4357 = stablehlo.multiply %v4356, %v4340 : tensor<32x192x28x28xf32>
    %v4358 = stablehlo.reduce(%v4357 init: %v4342) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4359 = stablehlo.broadcast_in_dim %v4358, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v4360 = stablehlo.multiply %v4355, %v4357 : tensor<32x192x28x28xf32>
    %v4361 = stablehlo.reduce(%v4360 init: %v4342) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4362 = stablehlo.broadcast_in_dim %v4361, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v4363 = stablehlo.multiply %v4357, %v4343 : tensor<32x192x28x28xf32>
    %v4364 = stablehlo.subtract %v4363, %v4359 : tensor<32x192x28x28xf32>
    %v4365 = stablehlo.multiply %v4355, %v4362 : tensor<32x192x28x28xf32>
    %v4366 = stablehlo.subtract %v4364, %v4365 : tensor<32x192x28x28xf32>
    %v4367 = stablehlo.divide %v4354, %v4343 : tensor<32x192x28x28xf32>
    %v4368 = stablehlo.multiply %v4367, %v4366 : tensor<32x192x28x28xf32>
    %v4369 = stablehlo.reshape %v4368 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v4370 = stablehlo.reshape %v4369 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4371 = stablehlo.transpose %We5, dims = [1, 0, 2, 3] : (tensor<192x32x1x1xf32>) -> tensor<32x192x1x1xf32>
    %v4372 = stablehlo.reverse %v4371, dims = [2, 3] : tensor<32x192x1x1xf32>
    %v4373 = stablehlo.convolution(%v4370, %v4372)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<32x192x1x1xf32>) -> tensor<32x32x28x28xf32>
    %v4374 = stablehlo.reshape %v4373 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v4375 = stablehlo.add %v4374, %v4150 : tensor<32x25088xf32>
    %v4376 = stablehlo.reshape %v332 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4377 = stablehlo.reshape %v4369 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4378 = stablehlo.transpose %v4376, dims = [1, 0, 2, 3] : (tensor<32x32x28x28xf32>) -> tensor<32x32x28x28xf32>
    %v4379 = stablehlo.transpose %v4377, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v4380 = stablehlo.convolution(%v4378, %v4379)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<32x192x1x1xf32>
    %v4381 = stablehlo.transpose %v4380, dims = [1, 0, 2, 3] : (tensor<32x192x1x1xf32>) -> tensor<192x32x1x1xf32>
    %v4382 = stablehlo.constant dense<0.3> : tensor<192x32x1x1xf32>
    %v4383 = stablehlo.multiply %v4381, %v4382 : tensor<192x32x1x1xf32>
    %v4384 = stablehlo.subtract %We5, %v4383 : tensor<192x32x1x1xf32>
    %v4385 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4386 = stablehlo.reshape %v337 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4387 = stablehlo.constant dense<784.0> : tensor<32x192x28x28xf32>
    %v4388 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v4389 = stablehlo.reduce(%v4386 init: %v4385) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4390 = stablehlo.broadcast_in_dim %v4389, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v4391 = stablehlo.divide %v4390, %v4387 : tensor<32x192x28x28xf32>
    %v4392 = stablehlo.subtract %v4386, %v4391 : tensor<32x192x28x28xf32>
    %v4393 = stablehlo.multiply %v4392, %v4392 : tensor<32x192x28x28xf32>
    %v4394 = stablehlo.reduce(%v4393 init: %v4385) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4395 = stablehlo.broadcast_in_dim %v4394, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v4396 = stablehlo.divide %v4395, %v4387 : tensor<32x192x28x28xf32>
    %v4397 = stablehlo.add %v4396, %v4388 : tensor<32x192x28x28xf32>
    %v4398 = stablehlo.rsqrt %v4397 : tensor<32x192x28x28xf32>
    %v4399 = stablehlo.multiply %v4392, %v4398 : tensor<32x192x28x28xf32>
    %v4400 = stablehlo.reshape %v4339 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4401 = stablehlo.multiply %v4400, %v4399 : tensor<32x192x28x28xf32>
    %v4402 = stablehlo.reduce(%v4401 init: %v4385) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4403 = stablehlo.constant dense<0.3> : tensor<192xf32>
    %v4404 = stablehlo.multiply %v4402, %v4403 : tensor<192xf32>
    %v4405 = stablehlo.subtract %ge5, %v4404 : tensor<192xf32>
    %v4406 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4407 = stablehlo.reshape %v4339 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4408 = stablehlo.reduce(%v4407 init: %v4406) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4409 = stablehlo.constant dense<0.3> : tensor<192xf32>
    %v4410 = stablehlo.multiply %v4408, %v4409 : tensor<192xf32>
    %v4411 = stablehlo.subtract %bte5, %v4410 : tensor<192xf32>
    %v4412 = stablehlo.reshape %v361 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4413 = stablehlo.reshape %v4329 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4414 = stablehlo.transpose %v4412, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v4415 = stablehlo.transpose %v4413, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v4416 = stablehlo.convolution(%v4414, %v4415)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 192 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<1x192x3x3xf32>
    %v4417 = stablehlo.reshape %v4416 : (tensor<1x192x3x3xf32>) -> tensor<192x1x3x3xf32>
    %v4418 = stablehlo.constant dense<0.3> : tensor<192x1x3x3xf32>
    %v4419 = stablehlo.multiply %v4417, %v4418 : tensor<192x1x3x3xf32>
    %v4420 = stablehlo.subtract %Wd5, %v4419 : tensor<192x1x3x3xf32>
    %v4421 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4422 = stablehlo.reshape %v366 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4423 = stablehlo.constant dense<784.0> : tensor<32x192x28x28xf32>
    %v4424 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v4425 = stablehlo.reduce(%v4422 init: %v4421) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4426 = stablehlo.broadcast_in_dim %v4425, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v4427 = stablehlo.divide %v4426, %v4423 : tensor<32x192x28x28xf32>
    %v4428 = stablehlo.subtract %v4422, %v4427 : tensor<32x192x28x28xf32>
    %v4429 = stablehlo.multiply %v4428, %v4428 : tensor<32x192x28x28xf32>
    %v4430 = stablehlo.reduce(%v4429 init: %v4421) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192xf32>
    %v4431 = stablehlo.broadcast_in_dim %v4430, dims = [0, 1] : (tensor<32x192xf32>) -> tensor<32x192x28x28xf32>
    %v4432 = stablehlo.divide %v4431, %v4423 : tensor<32x192x28x28xf32>
    %v4433 = stablehlo.add %v4432, %v4424 : tensor<32x192x28x28xf32>
    %v4434 = stablehlo.rsqrt %v4433 : tensor<32x192x28x28xf32>
    %v4435 = stablehlo.multiply %v4428, %v4434 : tensor<32x192x28x28xf32>
    %v4436 = stablehlo.reshape %v4299 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4437 = stablehlo.multiply %v4436, %v4435 : tensor<32x192x28x28xf32>
    %v4438 = stablehlo.reduce(%v4437 init: %v4421) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4439 = stablehlo.constant dense<0.3> : tensor<192xf32>
    %v4440 = stablehlo.multiply %v4438, %v4439 : tensor<192xf32>
    %v4441 = stablehlo.subtract %gd5, %v4440 : tensor<192xf32>
    %v4442 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4443 = stablehlo.reshape %v4299 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4444 = stablehlo.reduce(%v4443 init: %v4442) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v4445 = stablehlo.constant dense<0.3> : tensor<192xf32>
    %v4446 = stablehlo.multiply %v4444, %v4445 : tensor<192xf32>
    %v4447 = stablehlo.subtract %btd5, %v4446 : tensor<192xf32>
    %v4448 = stablehlo.reshape %v390 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v4449 = stablehlo.reshape %v4288 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4450 = stablehlo.transpose %v4448, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v4451 = stablehlo.transpose %v4449, dims = [1, 0, 2, 3] : (tensor<32x32x28x28xf32>) -> tensor<32x32x28x28xf32>
    %v4452 = stablehlo.convolution(%v4450, %v4451)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<32x32x28x28xf32>) -> tensor<192x32x1x1xf32>
    %v4453 = stablehlo.transpose %v4452, dims = [1, 0, 2, 3] : (tensor<192x32x1x1xf32>) -> tensor<32x192x1x1xf32>
    %v4454 = stablehlo.constant dense<0.3> : tensor<32x192x1x1xf32>
    %v4455 = stablehlo.multiply %v4453, %v4454 : tensor<32x192x1x1xf32>
    %v4456 = stablehlo.subtract %Wp5, %v4455 : tensor<32x192x1x1xf32>
    %v4457 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4458 = stablehlo.reshape %v395 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4459 = stablehlo.constant dense<784.0> : tensor<32x32x28x28xf32>
    %v4460 = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %v4461 = stablehlo.reduce(%v4458 init: %v4457) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v4462 = stablehlo.broadcast_in_dim %v4461, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v4463 = stablehlo.divide %v4462, %v4459 : tensor<32x32x28x28xf32>
    %v4464 = stablehlo.subtract %v4458, %v4463 : tensor<32x32x28x28xf32>
    %v4465 = stablehlo.multiply %v4464, %v4464 : tensor<32x32x28x28xf32>
    %v4466 = stablehlo.reduce(%v4465 init: %v4457) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v4467 = stablehlo.broadcast_in_dim %v4466, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v4468 = stablehlo.divide %v4467, %v4459 : tensor<32x32x28x28xf32>
    %v4469 = stablehlo.add %v4468, %v4460 : tensor<32x32x28x28xf32>
    %v4470 = stablehlo.rsqrt %v4469 : tensor<32x32x28x28xf32>
    %v4471 = stablehlo.multiply %v4464, %v4470 : tensor<32x32x28x28xf32>
    %v4472 = stablehlo.reshape %v4150 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4473 = stablehlo.multiply %v4472, %v4471 : tensor<32x32x28x28xf32>
    %v4474 = stablehlo.reduce(%v4473 init: %v4457) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v4475 = stablehlo.constant dense<0.3> : tensor<32xf32>
    %v4476 = stablehlo.multiply %v4474, %v4475 : tensor<32xf32>
    %v4477 = stablehlo.subtract %gp5, %v4476 : tensor<32xf32>
    %v4478 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4479 = stablehlo.reshape %v4150 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4480 = stablehlo.reduce(%v4479 init: %v4478) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v4481 = stablehlo.constant dense<0.3> : tensor<32xf32>
    %v4482 = stablehlo.multiply %v4480, %v4481 : tensor<32xf32>
    %v4483 = stablehlo.subtract %btp5, %v4482 : tensor<32xf32>
    %v4484 = stablehlo.reshape %v4375 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4485 = stablehlo.reshape %v312 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4486 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4487 = stablehlo.constant dense<784.0> : tensor<32x32x28x28xf32>
    %v4488 = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %v4489 = stablehlo.reduce(%v4485 init: %v4486) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v4490 = stablehlo.broadcast_in_dim %v4489, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v4491 = stablehlo.divide %v4490, %v4487 : tensor<32x32x28x28xf32>
    %v4492 = stablehlo.subtract %v4485, %v4491 : tensor<32x32x28x28xf32>
    %v4493 = stablehlo.multiply %v4492, %v4492 : tensor<32x32x28x28xf32>
    %v4494 = stablehlo.reduce(%v4493 init: %v4486) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v4495 = stablehlo.broadcast_in_dim %v4494, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v4496 = stablehlo.divide %v4495, %v4487 : tensor<32x32x28x28xf32>
    %v4497 = stablehlo.add %v4496, %v4488 : tensor<32x32x28x28xf32>
    %v4498 = stablehlo.rsqrt %v4497 : tensor<32x32x28x28xf32>
    %v4499 = stablehlo.multiply %v4492, %v4498 : tensor<32x32x28x28xf32>
    %v4500 = stablehlo.broadcast_in_dim %gp4, dims = [1] : (tensor<32xf32>) -> tensor<32x32x28x28xf32>
    %v4501 = stablehlo.multiply %v4500, %v4484 : tensor<32x32x28x28xf32>
    %v4502 = stablehlo.reduce(%v4501 init: %v4486) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v4503 = stablehlo.broadcast_in_dim %v4502, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v4504 = stablehlo.multiply %v4499, %v4501 : tensor<32x32x28x28xf32>
    %v4505 = stablehlo.reduce(%v4504 init: %v4486) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v4506 = stablehlo.broadcast_in_dim %v4505, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v4507 = stablehlo.multiply %v4501, %v4487 : tensor<32x32x28x28xf32>
    %v4508 = stablehlo.subtract %v4507, %v4503 : tensor<32x32x28x28xf32>
    %v4509 = stablehlo.multiply %v4499, %v4506 : tensor<32x32x28x28xf32>
    %v4510 = stablehlo.subtract %v4508, %v4509 : tensor<32x32x28x28xf32>
    %v4511 = stablehlo.divide %v4498, %v4487 : tensor<32x32x28x28xf32>
    %v4512 = stablehlo.multiply %v4511, %v4510 : tensor<32x32x28x28xf32>
    %v4513 = stablehlo.reshape %v4512 : (tensor<32x32x28x28xf32>) -> tensor<32x25088xf32>
    %v4514 = stablehlo.reshape %v4513 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4515 = stablehlo.transpose %Wp4, dims = [1, 0, 2, 3] : (tensor<32x144x1x1xf32>) -> tensor<144x32x1x1xf32>
    %v4516 = stablehlo.reverse %v4515, dims = [2, 3] : tensor<144x32x1x1xf32>
    %v4517 = stablehlo.convolution(%v4514, %v4516)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x28x28xf32>, tensor<144x32x1x1xf32>) -> tensor<32x144x28x28xf32>
    %v4518 = stablehlo.reshape %v4517 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v4519 = stablehlo.constant dense<0.0> : tensor<32x112896xf32>
    %v4520 = stablehlo.constant dense<6.0> : tensor<32x112896xf32>
    %v4521 = stablehlo.compare GT, %v303, %v4519 : (tensor<32x112896xf32>, tensor<32x112896xf32>) -> tensor<32x112896xi1>
    %v4522 = stablehlo.compare LT, %v303, %v4520 : (tensor<32x112896xf32>, tensor<32x112896xf32>) -> tensor<32x112896xi1>
    %v4523 = stablehlo.and %v4521, %v4522 : tensor<32x112896xi1>
    %v4524 = stablehlo.select %v4523, %v4518, %v4519 : tensor<32x112896xi1>, tensor<32x112896xf32>
    %v4525 = stablehlo.reshape %v4524 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v4526 = stablehlo.reshape %v283 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v4527 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4528 = stablehlo.constant dense<784.0> : tensor<32x144x28x28xf32>
    %v4529 = stablehlo.constant dense<1.0e-5> : tensor<32x144x28x28xf32>
    %v4530 = stablehlo.reduce(%v4526 init: %v4527) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v4531 = stablehlo.broadcast_in_dim %v4530, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x28x28xf32>
    %v4532 = stablehlo.divide %v4531, %v4528 : tensor<32x144x28x28xf32>
    %v4533 = stablehlo.subtract %v4526, %v4532 : tensor<32x144x28x28xf32>
    %v4534 = stablehlo.multiply %v4533, %v4533 : tensor<32x144x28x28xf32>
    %v4535 = stablehlo.reduce(%v4534 init: %v4527) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v4536 = stablehlo.broadcast_in_dim %v4535, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x28x28xf32>
    %v4537 = stablehlo.divide %v4536, %v4528 : tensor<32x144x28x28xf32>
    %v4538 = stablehlo.add %v4537, %v4529 : tensor<32x144x28x28xf32>
    %v4539 = stablehlo.rsqrt %v4538 : tensor<32x144x28x28xf32>
    %v4540 = stablehlo.multiply %v4533, %v4539 : tensor<32x144x28x28xf32>
    %v4541 = stablehlo.broadcast_in_dim %gd4, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v4542 = stablehlo.multiply %v4541, %v4525 : tensor<32x144x28x28xf32>
    %v4543 = stablehlo.reduce(%v4542 init: %v4527) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v4544 = stablehlo.broadcast_in_dim %v4543, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x28x28xf32>
    %v4545 = stablehlo.multiply %v4540, %v4542 : tensor<32x144x28x28xf32>
    %v4546 = stablehlo.reduce(%v4545 init: %v4527) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v4547 = stablehlo.broadcast_in_dim %v4546, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x28x28xf32>
    %v4548 = stablehlo.multiply %v4542, %v4528 : tensor<32x144x28x28xf32>
    %v4549 = stablehlo.subtract %v4548, %v4544 : tensor<32x144x28x28xf32>
    %v4550 = stablehlo.multiply %v4540, %v4547 : tensor<32x144x28x28xf32>
    %v4551 = stablehlo.subtract %v4549, %v4550 : tensor<32x144x28x28xf32>
    %v4552 = stablehlo.divide %v4539, %v4528 : tensor<32x144x28x28xf32>
    %v4553 = stablehlo.multiply %v4552, %v4551 : tensor<32x144x28x28xf32>
    %v4554 = stablehlo.reshape %v4553 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v4555 = stablehlo.reshape %v4554 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v4556 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4557 = stablehlo.pad %v4555, %v4556, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144x56x56xf32>
    %v4558 = stablehlo.reverse %Wd4, dims = [2, 3] : tensor<144x1x3x3xf32>
    %v4559 = stablehlo.convolution(%v4557, %v4558)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 144 : i64} : (tensor<32x144x56x56xf32>, tensor<144x1x3x3xf32>) -> tensor<32x144x56x56xf32>
    %v4560 = stablehlo.reshape %v4559 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v4561 = stablehlo.constant dense<0.0> : tensor<32x451584xf32>
    %v4562 = stablehlo.constant dense<6.0> : tensor<32x451584xf32>
    %v4563 = stablehlo.compare GT, %v274, %v4561 : (tensor<32x451584xf32>, tensor<32x451584xf32>) -> tensor<32x451584xi1>
    %v4564 = stablehlo.compare LT, %v274, %v4562 : (tensor<32x451584xf32>, tensor<32x451584xf32>) -> tensor<32x451584xi1>
    %v4565 = stablehlo.and %v4563, %v4564 : tensor<32x451584xi1>
    %v4566 = stablehlo.select %v4565, %v4560, %v4561 : tensor<32x451584xi1>, tensor<32x451584xf32>
    %v4567 = stablehlo.reshape %v4566 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4568 = stablehlo.reshape %v254 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4569 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4570 = stablehlo.constant dense<3136.0> : tensor<32x144x56x56xf32>
    %v4571 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v4572 = stablehlo.reduce(%v4568 init: %v4569) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v4573 = stablehlo.broadcast_in_dim %v4572, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v4574 = stablehlo.divide %v4573, %v4570 : tensor<32x144x56x56xf32>
    %v4575 = stablehlo.subtract %v4568, %v4574 : tensor<32x144x56x56xf32>
    %v4576 = stablehlo.multiply %v4575, %v4575 : tensor<32x144x56x56xf32>
    %v4577 = stablehlo.reduce(%v4576 init: %v4569) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v4578 = stablehlo.broadcast_in_dim %v4577, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v4579 = stablehlo.divide %v4578, %v4570 : tensor<32x144x56x56xf32>
    %v4580 = stablehlo.add %v4579, %v4571 : tensor<32x144x56x56xf32>
    %v4581 = stablehlo.rsqrt %v4580 : tensor<32x144x56x56xf32>
    %v4582 = stablehlo.multiply %v4575, %v4581 : tensor<32x144x56x56xf32>
    %v4583 = stablehlo.broadcast_in_dim %ge4, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v4584 = stablehlo.multiply %v4583, %v4567 : tensor<32x144x56x56xf32>
    %v4585 = stablehlo.reduce(%v4584 init: %v4569) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v4586 = stablehlo.broadcast_in_dim %v4585, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v4587 = stablehlo.multiply %v4582, %v4584 : tensor<32x144x56x56xf32>
    %v4588 = stablehlo.reduce(%v4587 init: %v4569) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v4589 = stablehlo.broadcast_in_dim %v4588, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v4590 = stablehlo.multiply %v4584, %v4570 : tensor<32x144x56x56xf32>
    %v4591 = stablehlo.subtract %v4590, %v4586 : tensor<32x144x56x56xf32>
    %v4592 = stablehlo.multiply %v4582, %v4589 : tensor<32x144x56x56xf32>
    %v4593 = stablehlo.subtract %v4591, %v4592 : tensor<32x144x56x56xf32>
    %v4594 = stablehlo.divide %v4581, %v4570 : tensor<32x144x56x56xf32>
    %v4595 = stablehlo.multiply %v4594, %v4593 : tensor<32x144x56x56xf32>
    %v4596 = stablehlo.reshape %v4595 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v4597 = stablehlo.reshape %v4596 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4598 = stablehlo.transpose %We4, dims = [1, 0, 2, 3] : (tensor<144x24x1x1xf32>) -> tensor<24x144x1x1xf32>
    %v4599 = stablehlo.reverse %v4598, dims = [2, 3] : tensor<24x144x1x1xf32>
    %v4600 = stablehlo.convolution(%v4597, %v4599)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x144x56x56xf32>, tensor<24x144x1x1xf32>) -> tensor<32x24x56x56xf32>
    %v4601 = stablehlo.reshape %v4600 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v4602 = stablehlo.reshape %v249 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v4603 = stablehlo.reshape %v4596 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4604 = stablehlo.transpose %v4602, dims = [1, 0, 2, 3] : (tensor<32x24x56x56xf32>) -> tensor<24x32x56x56xf32>
    %v4605 = stablehlo.transpose %v4603, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %v4606 = stablehlo.convolution(%v4604, %v4605)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<24x32x56x56xf32>, tensor<144x32x56x56xf32>) -> tensor<24x144x1x1xf32>
    %v4607 = stablehlo.transpose %v4606, dims = [1, 0, 2, 3] : (tensor<24x144x1x1xf32>) -> tensor<144x24x1x1xf32>
    %v4608 = stablehlo.constant dense<0.3> : tensor<144x24x1x1xf32>
    %v4609 = stablehlo.multiply %v4607, %v4608 : tensor<144x24x1x1xf32>
    %v4610 = stablehlo.subtract %We4, %v4609 : tensor<144x24x1x1xf32>
    %v4611 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4612 = stablehlo.reshape %v254 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4613 = stablehlo.constant dense<3136.0> : tensor<32x144x56x56xf32>
    %v4614 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v4615 = stablehlo.reduce(%v4612 init: %v4611) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v4616 = stablehlo.broadcast_in_dim %v4615, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v4617 = stablehlo.divide %v4616, %v4613 : tensor<32x144x56x56xf32>
    %v4618 = stablehlo.subtract %v4612, %v4617 : tensor<32x144x56x56xf32>
    %v4619 = stablehlo.multiply %v4618, %v4618 : tensor<32x144x56x56xf32>
    %v4620 = stablehlo.reduce(%v4619 init: %v4611) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v4621 = stablehlo.broadcast_in_dim %v4620, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v4622 = stablehlo.divide %v4621, %v4613 : tensor<32x144x56x56xf32>
    %v4623 = stablehlo.add %v4622, %v4614 : tensor<32x144x56x56xf32>
    %v4624 = stablehlo.rsqrt %v4623 : tensor<32x144x56x56xf32>
    %v4625 = stablehlo.multiply %v4618, %v4624 : tensor<32x144x56x56xf32>
    %v4626 = stablehlo.reshape %v4566 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4627 = stablehlo.multiply %v4626, %v4625 : tensor<32x144x56x56xf32>
    %v4628 = stablehlo.reduce(%v4627 init: %v4611) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4629 = stablehlo.constant dense<0.3> : tensor<144xf32>
    %v4630 = stablehlo.multiply %v4628, %v4629 : tensor<144xf32>
    %v4631 = stablehlo.subtract %ge4, %v4630 : tensor<144xf32>
    %v4632 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4633 = stablehlo.reshape %v4566 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4634 = stablehlo.reduce(%v4633 init: %v4632) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4635 = stablehlo.constant dense<0.3> : tensor<144xf32>
    %v4636 = stablehlo.multiply %v4634, %v4635 : tensor<144xf32>
    %v4637 = stablehlo.subtract %bte4, %v4636 : tensor<144xf32>
    %v4638 = stablehlo.reshape %v278 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4639 = stablehlo.reshape %v4554 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v4640 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4641 = stablehlo.pad %v4639, %v4640, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144x56x56xf32>
    %v4642 = stablehlo.transpose %v4638, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %v4643 = stablehlo.transpose %v4641, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %v4644 = stablehlo.convolution(%v4642, %v4643)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 144 : i64, feature_group_count = 1 : i64} : (tensor<144x32x56x56xf32>, tensor<144x32x56x56xf32>) -> tensor<1x144x3x3xf32>
    %v4645 = stablehlo.reshape %v4644 : (tensor<1x144x3x3xf32>) -> tensor<144x1x3x3xf32>
    %v4646 = stablehlo.constant dense<0.3> : tensor<144x1x3x3xf32>
    %v4647 = stablehlo.multiply %v4645, %v4646 : tensor<144x1x3x3xf32>
    %v4648 = stablehlo.subtract %Wd4, %v4647 : tensor<144x1x3x3xf32>
    %v4649 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4650 = stablehlo.reshape %v283 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v4651 = stablehlo.constant dense<784.0> : tensor<32x144x28x28xf32>
    %v4652 = stablehlo.constant dense<1.0e-5> : tensor<32x144x28x28xf32>
    %v4653 = stablehlo.reduce(%v4650 init: %v4649) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v4654 = stablehlo.broadcast_in_dim %v4653, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x28x28xf32>
    %v4655 = stablehlo.divide %v4654, %v4651 : tensor<32x144x28x28xf32>
    %v4656 = stablehlo.subtract %v4650, %v4655 : tensor<32x144x28x28xf32>
    %v4657 = stablehlo.multiply %v4656, %v4656 : tensor<32x144x28x28xf32>
    %v4658 = stablehlo.reduce(%v4657 init: %v4649) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v4659 = stablehlo.broadcast_in_dim %v4658, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x28x28xf32>
    %v4660 = stablehlo.divide %v4659, %v4651 : tensor<32x144x28x28xf32>
    %v4661 = stablehlo.add %v4660, %v4652 : tensor<32x144x28x28xf32>
    %v4662 = stablehlo.rsqrt %v4661 : tensor<32x144x28x28xf32>
    %v4663 = stablehlo.multiply %v4656, %v4662 : tensor<32x144x28x28xf32>
    %v4664 = stablehlo.reshape %v4524 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v4665 = stablehlo.multiply %v4664, %v4663 : tensor<32x144x28x28xf32>
    %v4666 = stablehlo.reduce(%v4665 init: %v4649) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v4667 = stablehlo.constant dense<0.3> : tensor<144xf32>
    %v4668 = stablehlo.multiply %v4666, %v4667 : tensor<144xf32>
    %v4669 = stablehlo.subtract %gd4, %v4668 : tensor<144xf32>
    %v4670 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4671 = stablehlo.reshape %v4524 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v4672 = stablehlo.reduce(%v4671 init: %v4670) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v4673 = stablehlo.constant dense<0.3> : tensor<144xf32>
    %v4674 = stablehlo.multiply %v4672, %v4673 : tensor<144xf32>
    %v4675 = stablehlo.subtract %btd4, %v4674 : tensor<144xf32>
    %v4676 = stablehlo.reshape %v307 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v4677 = stablehlo.reshape %v4513 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4678 = stablehlo.transpose %v4676, dims = [1, 0, 2, 3] : (tensor<32x144x28x28xf32>) -> tensor<144x32x28x28xf32>
    %v4679 = stablehlo.transpose %v4677, dims = [1, 0, 2, 3] : (tensor<32x32x28x28xf32>) -> tensor<32x32x28x28xf32>
    %v4680 = stablehlo.convolution(%v4678, %v4679)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<144x32x28x28xf32>, tensor<32x32x28x28xf32>) -> tensor<144x32x1x1xf32>
    %v4681 = stablehlo.transpose %v4680, dims = [1, 0, 2, 3] : (tensor<144x32x1x1xf32>) -> tensor<32x144x1x1xf32>
    %v4682 = stablehlo.constant dense<0.3> : tensor<32x144x1x1xf32>
    %v4683 = stablehlo.multiply %v4681, %v4682 : tensor<32x144x1x1xf32>
    %v4684 = stablehlo.subtract %Wp4, %v4683 : tensor<32x144x1x1xf32>
    %v4685 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4686 = stablehlo.reshape %v312 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4687 = stablehlo.constant dense<784.0> : tensor<32x32x28x28xf32>
    %v4688 = stablehlo.constant dense<1.0e-5> : tensor<32x32x28x28xf32>
    %v4689 = stablehlo.reduce(%v4686 init: %v4685) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v4690 = stablehlo.broadcast_in_dim %v4689, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v4691 = stablehlo.divide %v4690, %v4687 : tensor<32x32x28x28xf32>
    %v4692 = stablehlo.subtract %v4686, %v4691 : tensor<32x32x28x28xf32>
    %v4693 = stablehlo.multiply %v4692, %v4692 : tensor<32x32x28x28xf32>
    %v4694 = stablehlo.reduce(%v4693 init: %v4685) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v4695 = stablehlo.broadcast_in_dim %v4694, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x28x28xf32>
    %v4696 = stablehlo.divide %v4695, %v4687 : tensor<32x32x28x28xf32>
    %v4697 = stablehlo.add %v4696, %v4688 : tensor<32x32x28x28xf32>
    %v4698 = stablehlo.rsqrt %v4697 : tensor<32x32x28x28xf32>
    %v4699 = stablehlo.multiply %v4692, %v4698 : tensor<32x32x28x28xf32>
    %v4700 = stablehlo.reshape %v4375 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4701 = stablehlo.multiply %v4700, %v4699 : tensor<32x32x28x28xf32>
    %v4702 = stablehlo.reduce(%v4701 init: %v4685) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v4703 = stablehlo.constant dense<0.3> : tensor<32xf32>
    %v4704 = stablehlo.multiply %v4702, %v4703 : tensor<32xf32>
    %v4705 = stablehlo.subtract %gp4, %v4704 : tensor<32xf32>
    %v4706 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4707 = stablehlo.reshape %v4375 : (tensor<32x25088xf32>) -> tensor<32x32x28x28xf32>
    %v4708 = stablehlo.reduce(%v4707 init: %v4706) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>
    %v4709 = stablehlo.constant dense<0.3> : tensor<32xf32>
    %v4710 = stablehlo.multiply %v4708, %v4709 : tensor<32xf32>
    %v4711 = stablehlo.subtract %btp4, %v4710 : tensor<32xf32>
    %v4712 = stablehlo.reshape %v4601 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v4713 = stablehlo.reshape %v228 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v4714 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4715 = stablehlo.constant dense<3136.0> : tensor<32x24x56x56xf32>
    %v4716 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v4717 = stablehlo.reduce(%v4713 init: %v4714) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v4718 = stablehlo.broadcast_in_dim %v4717, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v4719 = stablehlo.divide %v4718, %v4715 : tensor<32x24x56x56xf32>
    %v4720 = stablehlo.subtract %v4713, %v4719 : tensor<32x24x56x56xf32>
    %v4721 = stablehlo.multiply %v4720, %v4720 : tensor<32x24x56x56xf32>
    %v4722 = stablehlo.reduce(%v4721 init: %v4714) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v4723 = stablehlo.broadcast_in_dim %v4722, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v4724 = stablehlo.divide %v4723, %v4715 : tensor<32x24x56x56xf32>
    %v4725 = stablehlo.add %v4724, %v4716 : tensor<32x24x56x56xf32>
    %v4726 = stablehlo.rsqrt %v4725 : tensor<32x24x56x56xf32>
    %v4727 = stablehlo.multiply %v4720, %v4726 : tensor<32x24x56x56xf32>
    %v4728 = stablehlo.broadcast_in_dim %gp3, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v4729 = stablehlo.multiply %v4728, %v4712 : tensor<32x24x56x56xf32>
    %v4730 = stablehlo.reduce(%v4729 init: %v4714) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v4731 = stablehlo.broadcast_in_dim %v4730, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v4732 = stablehlo.multiply %v4727, %v4729 : tensor<32x24x56x56xf32>
    %v4733 = stablehlo.reduce(%v4732 init: %v4714) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v4734 = stablehlo.broadcast_in_dim %v4733, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v4735 = stablehlo.multiply %v4729, %v4715 : tensor<32x24x56x56xf32>
    %v4736 = stablehlo.subtract %v4735, %v4731 : tensor<32x24x56x56xf32>
    %v4737 = stablehlo.multiply %v4727, %v4734 : tensor<32x24x56x56xf32>
    %v4738 = stablehlo.subtract %v4736, %v4737 : tensor<32x24x56x56xf32>
    %v4739 = stablehlo.divide %v4726, %v4715 : tensor<32x24x56x56xf32>
    %v4740 = stablehlo.multiply %v4739, %v4738 : tensor<32x24x56x56xf32>
    %v4741 = stablehlo.reshape %v4740 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v4742 = stablehlo.reshape %v4741 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v4743 = stablehlo.transpose %Wp3, dims = [1, 0, 2, 3] : (tensor<24x144x1x1xf32>) -> tensor<144x24x1x1xf32>
    %v4744 = stablehlo.reverse %v4743, dims = [2, 3] : tensor<144x24x1x1xf32>
    %v4745 = stablehlo.convolution(%v4742, %v4744)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<144x24x1x1xf32>) -> tensor<32x144x56x56xf32>
    %v4746 = stablehlo.reshape %v4745 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v4747 = stablehlo.constant dense<0.0> : tensor<32x451584xf32>
    %v4748 = stablehlo.constant dense<6.0> : tensor<32x451584xf32>
    %v4749 = stablehlo.compare GT, %v219, %v4747 : (tensor<32x451584xf32>, tensor<32x451584xf32>) -> tensor<32x451584xi1>
    %v4750 = stablehlo.compare LT, %v219, %v4748 : (tensor<32x451584xf32>, tensor<32x451584xf32>) -> tensor<32x451584xi1>
    %v4751 = stablehlo.and %v4749, %v4750 : tensor<32x451584xi1>
    %v4752 = stablehlo.select %v4751, %v4746, %v4747 : tensor<32x451584xi1>, tensor<32x451584xf32>
    %v4753 = stablehlo.reshape %v4752 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4754 = stablehlo.reshape %v199 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4755 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4756 = stablehlo.constant dense<3136.0> : tensor<32x144x56x56xf32>
    %v4757 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v4758 = stablehlo.reduce(%v4754 init: %v4755) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v4759 = stablehlo.broadcast_in_dim %v4758, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v4760 = stablehlo.divide %v4759, %v4756 : tensor<32x144x56x56xf32>
    %v4761 = stablehlo.subtract %v4754, %v4760 : tensor<32x144x56x56xf32>
    %v4762 = stablehlo.multiply %v4761, %v4761 : tensor<32x144x56x56xf32>
    %v4763 = stablehlo.reduce(%v4762 init: %v4755) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v4764 = stablehlo.broadcast_in_dim %v4763, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v4765 = stablehlo.divide %v4764, %v4756 : tensor<32x144x56x56xf32>
    %v4766 = stablehlo.add %v4765, %v4757 : tensor<32x144x56x56xf32>
    %v4767 = stablehlo.rsqrt %v4766 : tensor<32x144x56x56xf32>
    %v4768 = stablehlo.multiply %v4761, %v4767 : tensor<32x144x56x56xf32>
    %v4769 = stablehlo.broadcast_in_dim %gd3, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v4770 = stablehlo.multiply %v4769, %v4753 : tensor<32x144x56x56xf32>
    %v4771 = stablehlo.reduce(%v4770 init: %v4755) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v4772 = stablehlo.broadcast_in_dim %v4771, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v4773 = stablehlo.multiply %v4768, %v4770 : tensor<32x144x56x56xf32>
    %v4774 = stablehlo.reduce(%v4773 init: %v4755) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v4775 = stablehlo.broadcast_in_dim %v4774, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v4776 = stablehlo.multiply %v4770, %v4756 : tensor<32x144x56x56xf32>
    %v4777 = stablehlo.subtract %v4776, %v4772 : tensor<32x144x56x56xf32>
    %v4778 = stablehlo.multiply %v4768, %v4775 : tensor<32x144x56x56xf32>
    %v4779 = stablehlo.subtract %v4777, %v4778 : tensor<32x144x56x56xf32>
    %v4780 = stablehlo.divide %v4767, %v4756 : tensor<32x144x56x56xf32>
    %v4781 = stablehlo.multiply %v4780, %v4779 : tensor<32x144x56x56xf32>
    %v4782 = stablehlo.reshape %v4781 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v4783 = stablehlo.reshape %v4782 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4784 = stablehlo.reverse %Wd3, dims = [2, 3] : tensor<144x1x3x3xf32>
    %v4785 = stablehlo.convolution(%v4783, %v4784)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 144 : i64} : (tensor<32x144x56x56xf32>, tensor<144x1x3x3xf32>) -> tensor<32x144x56x56xf32>
    %v4786 = stablehlo.reshape %v4785 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v4787 = stablehlo.constant dense<0.0> : tensor<32x451584xf32>
    %v4788 = stablehlo.constant dense<6.0> : tensor<32x451584xf32>
    %v4789 = stablehlo.compare GT, %v190, %v4787 : (tensor<32x451584xf32>, tensor<32x451584xf32>) -> tensor<32x451584xi1>
    %v4790 = stablehlo.compare LT, %v190, %v4788 : (tensor<32x451584xf32>, tensor<32x451584xf32>) -> tensor<32x451584xi1>
    %v4791 = stablehlo.and %v4789, %v4790 : tensor<32x451584xi1>
    %v4792 = stablehlo.select %v4791, %v4786, %v4787 : tensor<32x451584xi1>, tensor<32x451584xf32>
    %v4793 = stablehlo.reshape %v4792 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4794 = stablehlo.reshape %v170 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4795 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4796 = stablehlo.constant dense<3136.0> : tensor<32x144x56x56xf32>
    %v4797 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v4798 = stablehlo.reduce(%v4794 init: %v4795) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v4799 = stablehlo.broadcast_in_dim %v4798, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v4800 = stablehlo.divide %v4799, %v4796 : tensor<32x144x56x56xf32>
    %v4801 = stablehlo.subtract %v4794, %v4800 : tensor<32x144x56x56xf32>
    %v4802 = stablehlo.multiply %v4801, %v4801 : tensor<32x144x56x56xf32>
    %v4803 = stablehlo.reduce(%v4802 init: %v4795) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v4804 = stablehlo.broadcast_in_dim %v4803, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v4805 = stablehlo.divide %v4804, %v4796 : tensor<32x144x56x56xf32>
    %v4806 = stablehlo.add %v4805, %v4797 : tensor<32x144x56x56xf32>
    %v4807 = stablehlo.rsqrt %v4806 : tensor<32x144x56x56xf32>
    %v4808 = stablehlo.multiply %v4801, %v4807 : tensor<32x144x56x56xf32>
    %v4809 = stablehlo.broadcast_in_dim %ge3, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v4810 = stablehlo.multiply %v4809, %v4793 : tensor<32x144x56x56xf32>
    %v4811 = stablehlo.reduce(%v4810 init: %v4795) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v4812 = stablehlo.broadcast_in_dim %v4811, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v4813 = stablehlo.multiply %v4808, %v4810 : tensor<32x144x56x56xf32>
    %v4814 = stablehlo.reduce(%v4813 init: %v4795) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v4815 = stablehlo.broadcast_in_dim %v4814, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v4816 = stablehlo.multiply %v4810, %v4796 : tensor<32x144x56x56xf32>
    %v4817 = stablehlo.subtract %v4816, %v4812 : tensor<32x144x56x56xf32>
    %v4818 = stablehlo.multiply %v4808, %v4815 : tensor<32x144x56x56xf32>
    %v4819 = stablehlo.subtract %v4817, %v4818 : tensor<32x144x56x56xf32>
    %v4820 = stablehlo.divide %v4807, %v4796 : tensor<32x144x56x56xf32>
    %v4821 = stablehlo.multiply %v4820, %v4819 : tensor<32x144x56x56xf32>
    %v4822 = stablehlo.reshape %v4821 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v4823 = stablehlo.reshape %v4822 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4824 = stablehlo.transpose %We3, dims = [1, 0, 2, 3] : (tensor<144x24x1x1xf32>) -> tensor<24x144x1x1xf32>
    %v4825 = stablehlo.reverse %v4824, dims = [2, 3] : tensor<24x144x1x1xf32>
    %v4826 = stablehlo.convolution(%v4823, %v4825)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x144x56x56xf32>, tensor<24x144x1x1xf32>) -> tensor<32x24x56x56xf32>
    %v4827 = stablehlo.reshape %v4826 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v4828 = stablehlo.add %v4827, %v4601 : tensor<32x75264xf32>
    %v4829 = stablehlo.reshape %v165 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v4830 = stablehlo.reshape %v4822 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4831 = stablehlo.transpose %v4829, dims = [1, 0, 2, 3] : (tensor<32x24x56x56xf32>) -> tensor<24x32x56x56xf32>
    %v4832 = stablehlo.transpose %v4830, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %v4833 = stablehlo.convolution(%v4831, %v4832)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<24x32x56x56xf32>, tensor<144x32x56x56xf32>) -> tensor<24x144x1x1xf32>
    %v4834 = stablehlo.transpose %v4833, dims = [1, 0, 2, 3] : (tensor<24x144x1x1xf32>) -> tensor<144x24x1x1xf32>
    %v4835 = stablehlo.constant dense<0.3> : tensor<144x24x1x1xf32>
    %v4836 = stablehlo.multiply %v4834, %v4835 : tensor<144x24x1x1xf32>
    %v4837 = stablehlo.subtract %We3, %v4836 : tensor<144x24x1x1xf32>
    %v4838 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4839 = stablehlo.reshape %v170 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4840 = stablehlo.constant dense<3136.0> : tensor<32x144x56x56xf32>
    %v4841 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v4842 = stablehlo.reduce(%v4839 init: %v4838) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v4843 = stablehlo.broadcast_in_dim %v4842, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v4844 = stablehlo.divide %v4843, %v4840 : tensor<32x144x56x56xf32>
    %v4845 = stablehlo.subtract %v4839, %v4844 : tensor<32x144x56x56xf32>
    %v4846 = stablehlo.multiply %v4845, %v4845 : tensor<32x144x56x56xf32>
    %v4847 = stablehlo.reduce(%v4846 init: %v4838) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v4848 = stablehlo.broadcast_in_dim %v4847, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v4849 = stablehlo.divide %v4848, %v4840 : tensor<32x144x56x56xf32>
    %v4850 = stablehlo.add %v4849, %v4841 : tensor<32x144x56x56xf32>
    %v4851 = stablehlo.rsqrt %v4850 : tensor<32x144x56x56xf32>
    %v4852 = stablehlo.multiply %v4845, %v4851 : tensor<32x144x56x56xf32>
    %v4853 = stablehlo.reshape %v4792 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4854 = stablehlo.multiply %v4853, %v4852 : tensor<32x144x56x56xf32>
    %v4855 = stablehlo.reduce(%v4854 init: %v4838) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4856 = stablehlo.constant dense<0.3> : tensor<144xf32>
    %v4857 = stablehlo.multiply %v4855, %v4856 : tensor<144xf32>
    %v4858 = stablehlo.subtract %ge3, %v4857 : tensor<144xf32>
    %v4859 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4860 = stablehlo.reshape %v4792 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4861 = stablehlo.reduce(%v4860 init: %v4859) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4862 = stablehlo.constant dense<0.3> : tensor<144xf32>
    %v4863 = stablehlo.multiply %v4861, %v4862 : tensor<144xf32>
    %v4864 = stablehlo.subtract %bte3, %v4863 : tensor<144xf32>
    %v4865 = stablehlo.reshape %v194 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4866 = stablehlo.reshape %v4782 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4867 = stablehlo.transpose %v4865, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %v4868 = stablehlo.transpose %v4866, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %v4869 = stablehlo.convolution(%v4867, %v4868)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 144 : i64, feature_group_count = 1 : i64} : (tensor<144x32x56x56xf32>, tensor<144x32x56x56xf32>) -> tensor<1x144x3x3xf32>
    %v4870 = stablehlo.reshape %v4869 : (tensor<1x144x3x3xf32>) -> tensor<144x1x3x3xf32>
    %v4871 = stablehlo.constant dense<0.3> : tensor<144x1x3x3xf32>
    %v4872 = stablehlo.multiply %v4870, %v4871 : tensor<144x1x3x3xf32>
    %v4873 = stablehlo.subtract %Wd3, %v4872 : tensor<144x1x3x3xf32>
    %v4874 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4875 = stablehlo.reshape %v199 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4876 = stablehlo.constant dense<3136.0> : tensor<32x144x56x56xf32>
    %v4877 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v4878 = stablehlo.reduce(%v4875 init: %v4874) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v4879 = stablehlo.broadcast_in_dim %v4878, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v4880 = stablehlo.divide %v4879, %v4876 : tensor<32x144x56x56xf32>
    %v4881 = stablehlo.subtract %v4875, %v4880 : tensor<32x144x56x56xf32>
    %v4882 = stablehlo.multiply %v4881, %v4881 : tensor<32x144x56x56xf32>
    %v4883 = stablehlo.reduce(%v4882 init: %v4874) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v4884 = stablehlo.broadcast_in_dim %v4883, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v4885 = stablehlo.divide %v4884, %v4876 : tensor<32x144x56x56xf32>
    %v4886 = stablehlo.add %v4885, %v4877 : tensor<32x144x56x56xf32>
    %v4887 = stablehlo.rsqrt %v4886 : tensor<32x144x56x56xf32>
    %v4888 = stablehlo.multiply %v4881, %v4887 : tensor<32x144x56x56xf32>
    %v4889 = stablehlo.reshape %v4752 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4890 = stablehlo.multiply %v4889, %v4888 : tensor<32x144x56x56xf32>
    %v4891 = stablehlo.reduce(%v4890 init: %v4874) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4892 = stablehlo.constant dense<0.3> : tensor<144xf32>
    %v4893 = stablehlo.multiply %v4891, %v4892 : tensor<144xf32>
    %v4894 = stablehlo.subtract %gd3, %v4893 : tensor<144xf32>
    %v4895 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4896 = stablehlo.reshape %v4752 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4897 = stablehlo.reduce(%v4896 init: %v4895) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v4898 = stablehlo.constant dense<0.3> : tensor<144xf32>
    %v4899 = stablehlo.multiply %v4897, %v4898 : tensor<144xf32>
    %v4900 = stablehlo.subtract %btd3, %v4899 : tensor<144xf32>
    %v4901 = stablehlo.reshape %v223 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v4902 = stablehlo.reshape %v4741 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v4903 = stablehlo.transpose %v4901, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %v4904 = stablehlo.transpose %v4902, dims = [1, 0, 2, 3] : (tensor<32x24x56x56xf32>) -> tensor<24x32x56x56xf32>
    %v4905 = stablehlo.convolution(%v4903, %v4904)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<144x32x56x56xf32>, tensor<24x32x56x56xf32>) -> tensor<144x24x1x1xf32>
    %v4906 = stablehlo.transpose %v4905, dims = [1, 0, 2, 3] : (tensor<144x24x1x1xf32>) -> tensor<24x144x1x1xf32>
    %v4907 = stablehlo.constant dense<0.3> : tensor<24x144x1x1xf32>
    %v4908 = stablehlo.multiply %v4906, %v4907 : tensor<24x144x1x1xf32>
    %v4909 = stablehlo.subtract %Wp3, %v4908 : tensor<24x144x1x1xf32>
    %v4910 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4911 = stablehlo.reshape %v228 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v4912 = stablehlo.constant dense<3136.0> : tensor<32x24x56x56xf32>
    %v4913 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v4914 = stablehlo.reduce(%v4911 init: %v4910) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v4915 = stablehlo.broadcast_in_dim %v4914, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v4916 = stablehlo.divide %v4915, %v4912 : tensor<32x24x56x56xf32>
    %v4917 = stablehlo.subtract %v4911, %v4916 : tensor<32x24x56x56xf32>
    %v4918 = stablehlo.multiply %v4917, %v4917 : tensor<32x24x56x56xf32>
    %v4919 = stablehlo.reduce(%v4918 init: %v4910) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v4920 = stablehlo.broadcast_in_dim %v4919, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v4921 = stablehlo.divide %v4920, %v4912 : tensor<32x24x56x56xf32>
    %v4922 = stablehlo.add %v4921, %v4913 : tensor<32x24x56x56xf32>
    %v4923 = stablehlo.rsqrt %v4922 : tensor<32x24x56x56xf32>
    %v4924 = stablehlo.multiply %v4917, %v4923 : tensor<32x24x56x56xf32>
    %v4925 = stablehlo.reshape %v4601 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v4926 = stablehlo.multiply %v4925, %v4924 : tensor<32x24x56x56xf32>
    %v4927 = stablehlo.reduce(%v4926 init: %v4910) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v4928 = stablehlo.constant dense<0.3> : tensor<24xf32>
    %v4929 = stablehlo.multiply %v4927, %v4928 : tensor<24xf32>
    %v4930 = stablehlo.subtract %gp3, %v4929 : tensor<24xf32>
    %v4931 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4932 = stablehlo.reshape %v4601 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v4933 = stablehlo.reduce(%v4932 init: %v4931) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v4934 = stablehlo.constant dense<0.3> : tensor<24xf32>
    %v4935 = stablehlo.multiply %v4933, %v4934 : tensor<24xf32>
    %v4936 = stablehlo.subtract %btp3, %v4935 : tensor<24xf32>
    %v4937 = stablehlo.reshape %v4828 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v4938 = stablehlo.reshape %v145 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v4939 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4940 = stablehlo.constant dense<3136.0> : tensor<32x24x56x56xf32>
    %v4941 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v4942 = stablehlo.reduce(%v4938 init: %v4939) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v4943 = stablehlo.broadcast_in_dim %v4942, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v4944 = stablehlo.divide %v4943, %v4940 : tensor<32x24x56x56xf32>
    %v4945 = stablehlo.subtract %v4938, %v4944 : tensor<32x24x56x56xf32>
    %v4946 = stablehlo.multiply %v4945, %v4945 : tensor<32x24x56x56xf32>
    %v4947 = stablehlo.reduce(%v4946 init: %v4939) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v4948 = stablehlo.broadcast_in_dim %v4947, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v4949 = stablehlo.divide %v4948, %v4940 : tensor<32x24x56x56xf32>
    %v4950 = stablehlo.add %v4949, %v4941 : tensor<32x24x56x56xf32>
    %v4951 = stablehlo.rsqrt %v4950 : tensor<32x24x56x56xf32>
    %v4952 = stablehlo.multiply %v4945, %v4951 : tensor<32x24x56x56xf32>
    %v4953 = stablehlo.broadcast_in_dim %gp2, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v4954 = stablehlo.multiply %v4953, %v4937 : tensor<32x24x56x56xf32>
    %v4955 = stablehlo.reduce(%v4954 init: %v4939) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v4956 = stablehlo.broadcast_in_dim %v4955, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v4957 = stablehlo.multiply %v4952, %v4954 : tensor<32x24x56x56xf32>
    %v4958 = stablehlo.reduce(%v4957 init: %v4939) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v4959 = stablehlo.broadcast_in_dim %v4958, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v4960 = stablehlo.multiply %v4954, %v4940 : tensor<32x24x56x56xf32>
    %v4961 = stablehlo.subtract %v4960, %v4956 : tensor<32x24x56x56xf32>
    %v4962 = stablehlo.multiply %v4952, %v4959 : tensor<32x24x56x56xf32>
    %v4963 = stablehlo.subtract %v4961, %v4962 : tensor<32x24x56x56xf32>
    %v4964 = stablehlo.divide %v4951, %v4940 : tensor<32x24x56x56xf32>
    %v4965 = stablehlo.multiply %v4964, %v4963 : tensor<32x24x56x56xf32>
    %v4966 = stablehlo.reshape %v4965 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v4967 = stablehlo.reshape %v4966 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v4968 = stablehlo.transpose %Wp2, dims = [1, 0, 2, 3] : (tensor<24x96x1x1xf32>) -> tensor<96x24x1x1xf32>
    %v4969 = stablehlo.reverse %v4968, dims = [2, 3] : tensor<96x24x1x1xf32>
    %v4970 = stablehlo.convolution(%v4967, %v4969)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<96x24x1x1xf32>) -> tensor<32x96x56x56xf32>
    %v4971 = stablehlo.reshape %v4970 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v4972 = stablehlo.constant dense<0.0> : tensor<32x301056xf32>
    %v4973 = stablehlo.constant dense<6.0> : tensor<32x301056xf32>
    %v4974 = stablehlo.compare GT, %v136, %v4972 : (tensor<32x301056xf32>, tensor<32x301056xf32>) -> tensor<32x301056xi1>
    %v4975 = stablehlo.compare LT, %v136, %v4973 : (tensor<32x301056xf32>, tensor<32x301056xf32>) -> tensor<32x301056xi1>
    %v4976 = stablehlo.and %v4974, %v4975 : tensor<32x301056xi1>
    %v4977 = stablehlo.select %v4976, %v4971, %v4972 : tensor<32x301056xi1>, tensor<32x301056xf32>
    %v4978 = stablehlo.reshape %v4977 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4979 = stablehlo.reshape %v116 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4980 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4981 = stablehlo.constant dense<3136.0> : tensor<32x96x56x56xf32>
    %v4982 = stablehlo.constant dense<1.0e-5> : tensor<32x96x56x56xf32>
    %v4983 = stablehlo.reduce(%v4979 init: %v4980) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v4984 = stablehlo.broadcast_in_dim %v4983, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v4985 = stablehlo.divide %v4984, %v4981 : tensor<32x96x56x56xf32>
    %v4986 = stablehlo.subtract %v4979, %v4985 : tensor<32x96x56x56xf32>
    %v4987 = stablehlo.multiply %v4986, %v4986 : tensor<32x96x56x56xf32>
    %v4988 = stablehlo.reduce(%v4987 init: %v4980) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v4989 = stablehlo.broadcast_in_dim %v4988, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v4990 = stablehlo.divide %v4989, %v4981 : tensor<32x96x56x56xf32>
    %v4991 = stablehlo.add %v4990, %v4982 : tensor<32x96x56x56xf32>
    %v4992 = stablehlo.rsqrt %v4991 : tensor<32x96x56x56xf32>
    %v4993 = stablehlo.multiply %v4986, %v4992 : tensor<32x96x56x56xf32>
    %v4994 = stablehlo.broadcast_in_dim %gd2, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v4995 = stablehlo.multiply %v4994, %v4978 : tensor<32x96x56x56xf32>
    %v4996 = stablehlo.reduce(%v4995 init: %v4980) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v4997 = stablehlo.broadcast_in_dim %v4996, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v4998 = stablehlo.multiply %v4993, %v4995 : tensor<32x96x56x56xf32>
    %v4999 = stablehlo.reduce(%v4998 init: %v4980) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v5000 = stablehlo.broadcast_in_dim %v4999, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v5001 = stablehlo.multiply %v4995, %v4981 : tensor<32x96x56x56xf32>
    %v5002 = stablehlo.subtract %v5001, %v4997 : tensor<32x96x56x56xf32>
    %v5003 = stablehlo.multiply %v4993, %v5000 : tensor<32x96x56x56xf32>
    %v5004 = stablehlo.subtract %v5002, %v5003 : tensor<32x96x56x56xf32>
    %v5005 = stablehlo.divide %v4992, %v4981 : tensor<32x96x56x56xf32>
    %v5006 = stablehlo.multiply %v5005, %v5004 : tensor<32x96x56x56xf32>
    %v5007 = stablehlo.reshape %v5006 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v5008 = stablehlo.reshape %v5007 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v5009 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5010 = stablehlo.pad %v5008, %v5009, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96x112x112xf32>
    %v5011 = stablehlo.reverse %Wd2, dims = [2, 3] : tensor<96x1x3x3xf32>
    %v5012 = stablehlo.convolution(%v5010, %v5011)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x112x112xf32>, tensor<96x1x3x3xf32>) -> tensor<32x96x112x112xf32>
    %v5013 = stablehlo.reshape %v5012 : (tensor<32x96x112x112xf32>) -> tensor<32x1204224xf32>
    %v5014 = stablehlo.constant dense<0.0> : tensor<32x1204224xf32>
    %v5015 = stablehlo.constant dense<6.0> : tensor<32x1204224xf32>
    %v5016 = stablehlo.compare GT, %v107, %v5014 : (tensor<32x1204224xf32>, tensor<32x1204224xf32>) -> tensor<32x1204224xi1>
    %v5017 = stablehlo.compare LT, %v107, %v5015 : (tensor<32x1204224xf32>, tensor<32x1204224xf32>) -> tensor<32x1204224xi1>
    %v5018 = stablehlo.and %v5016, %v5017 : tensor<32x1204224xi1>
    %v5019 = stablehlo.select %v5018, %v5013, %v5014 : tensor<32x1204224xi1>, tensor<32x1204224xf32>
    %v5020 = stablehlo.reshape %v5019 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v5021 = stablehlo.reshape %v87 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v5022 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5023 = stablehlo.constant dense<12544.0> : tensor<32x96x112x112xf32>
    %v5024 = stablehlo.constant dense<1.0e-5> : tensor<32x96x112x112xf32>
    %v5025 = stablehlo.reduce(%v5021 init: %v5022) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v5026 = stablehlo.broadcast_in_dim %v5025, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x112x112xf32>
    %v5027 = stablehlo.divide %v5026, %v5023 : tensor<32x96x112x112xf32>
    %v5028 = stablehlo.subtract %v5021, %v5027 : tensor<32x96x112x112xf32>
    %v5029 = stablehlo.multiply %v5028, %v5028 : tensor<32x96x112x112xf32>
    %v5030 = stablehlo.reduce(%v5029 init: %v5022) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v5031 = stablehlo.broadcast_in_dim %v5030, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x112x112xf32>
    %v5032 = stablehlo.divide %v5031, %v5023 : tensor<32x96x112x112xf32>
    %v5033 = stablehlo.add %v5032, %v5024 : tensor<32x96x112x112xf32>
    %v5034 = stablehlo.rsqrt %v5033 : tensor<32x96x112x112xf32>
    %v5035 = stablehlo.multiply %v5028, %v5034 : tensor<32x96x112x112xf32>
    %v5036 = stablehlo.broadcast_in_dim %ge2, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v5037 = stablehlo.multiply %v5036, %v5020 : tensor<32x96x112x112xf32>
    %v5038 = stablehlo.reduce(%v5037 init: %v5022) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v5039 = stablehlo.broadcast_in_dim %v5038, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x112x112xf32>
    %v5040 = stablehlo.multiply %v5035, %v5037 : tensor<32x96x112x112xf32>
    %v5041 = stablehlo.reduce(%v5040 init: %v5022) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v5042 = stablehlo.broadcast_in_dim %v5041, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x112x112xf32>
    %v5043 = stablehlo.multiply %v5037, %v5023 : tensor<32x96x112x112xf32>
    %v5044 = stablehlo.subtract %v5043, %v5039 : tensor<32x96x112x112xf32>
    %v5045 = stablehlo.multiply %v5035, %v5042 : tensor<32x96x112x112xf32>
    %v5046 = stablehlo.subtract %v5044, %v5045 : tensor<32x96x112x112xf32>
    %v5047 = stablehlo.divide %v5034, %v5023 : tensor<32x96x112x112xf32>
    %v5048 = stablehlo.multiply %v5047, %v5046 : tensor<32x96x112x112xf32>
    %v5049 = stablehlo.reshape %v5048 : (tensor<32x96x112x112xf32>) -> tensor<32x1204224xf32>
    %v5050 = stablehlo.reshape %v5049 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v5051 = stablehlo.transpose %We2, dims = [1, 0, 2, 3] : (tensor<96x16x1x1xf32>) -> tensor<16x96x1x1xf32>
    %v5052 = stablehlo.reverse %v5051, dims = [2, 3] : tensor<16x96x1x1xf32>
    %v5053 = stablehlo.convolution(%v5050, %v5052)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x112x112xf32>, tensor<16x96x1x1xf32>) -> tensor<32x16x112x112xf32>
    %v5054 = stablehlo.reshape %v5053 : (tensor<32x16x112x112xf32>) -> tensor<32x200704xf32>
    %v5055 = stablehlo.reshape %v82 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v5056 = stablehlo.reshape %v5049 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v5057 = stablehlo.transpose %v5055, dims = [1, 0, 2, 3] : (tensor<32x16x112x112xf32>) -> tensor<16x32x112x112xf32>
    %v5058 = stablehlo.transpose %v5056, dims = [1, 0, 2, 3] : (tensor<32x96x112x112xf32>) -> tensor<96x32x112x112xf32>
    %v5059 = stablehlo.convolution(%v5057, %v5058)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x32x112x112xf32>, tensor<96x32x112x112xf32>) -> tensor<16x96x1x1xf32>
    %v5060 = stablehlo.transpose %v5059, dims = [1, 0, 2, 3] : (tensor<16x96x1x1xf32>) -> tensor<96x16x1x1xf32>
    %v5061 = stablehlo.constant dense<0.3> : tensor<96x16x1x1xf32>
    %v5062 = stablehlo.multiply %v5060, %v5061 : tensor<96x16x1x1xf32>
    %v5063 = stablehlo.subtract %We2, %v5062 : tensor<96x16x1x1xf32>
    %v5064 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5065 = stablehlo.reshape %v87 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v5066 = stablehlo.constant dense<12544.0> : tensor<32x96x112x112xf32>
    %v5067 = stablehlo.constant dense<1.0e-5> : tensor<32x96x112x112xf32>
    %v5068 = stablehlo.reduce(%v5065 init: %v5064) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v5069 = stablehlo.broadcast_in_dim %v5068, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x112x112xf32>
    %v5070 = stablehlo.divide %v5069, %v5066 : tensor<32x96x112x112xf32>
    %v5071 = stablehlo.subtract %v5065, %v5070 : tensor<32x96x112x112xf32>
    %v5072 = stablehlo.multiply %v5071, %v5071 : tensor<32x96x112x112xf32>
    %v5073 = stablehlo.reduce(%v5072 init: %v5064) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v5074 = stablehlo.broadcast_in_dim %v5073, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x112x112xf32>
    %v5075 = stablehlo.divide %v5074, %v5066 : tensor<32x96x112x112xf32>
    %v5076 = stablehlo.add %v5075, %v5067 : tensor<32x96x112x112xf32>
    %v5077 = stablehlo.rsqrt %v5076 : tensor<32x96x112x112xf32>
    %v5078 = stablehlo.multiply %v5071, %v5077 : tensor<32x96x112x112xf32>
    %v5079 = stablehlo.reshape %v5019 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v5080 = stablehlo.multiply %v5079, %v5078 : tensor<32x96x112x112xf32>
    %v5081 = stablehlo.reduce(%v5080 init: %v5064) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v5082 = stablehlo.constant dense<0.3> : tensor<96xf32>
    %v5083 = stablehlo.multiply %v5081, %v5082 : tensor<96xf32>
    %v5084 = stablehlo.subtract %ge2, %v5083 : tensor<96xf32>
    %v5085 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5086 = stablehlo.reshape %v5019 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v5087 = stablehlo.reduce(%v5086 init: %v5085) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v5088 = stablehlo.constant dense<0.3> : tensor<96xf32>
    %v5089 = stablehlo.multiply %v5087, %v5088 : tensor<96xf32>
    %v5090 = stablehlo.subtract %bte2, %v5089 : tensor<96xf32>
    %v5091 = stablehlo.reshape %v111 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v5092 = stablehlo.reshape %v5007 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v5093 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5094 = stablehlo.pad %v5092, %v5093, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96x112x112xf32>
    %v5095 = stablehlo.transpose %v5091, dims = [1, 0, 2, 3] : (tensor<32x96x112x112xf32>) -> tensor<96x32x112x112xf32>
    %v5096 = stablehlo.transpose %v5094, dims = [1, 0, 2, 3] : (tensor<32x96x112x112xf32>) -> tensor<96x32x112x112xf32>
    %v5097 = stablehlo.convolution(%v5095, %v5096)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 96 : i64, feature_group_count = 1 : i64} : (tensor<96x32x112x112xf32>, tensor<96x32x112x112xf32>) -> tensor<1x96x3x3xf32>
    %v5098 = stablehlo.reshape %v5097 : (tensor<1x96x3x3xf32>) -> tensor<96x1x3x3xf32>
    %v5099 = stablehlo.constant dense<0.3> : tensor<96x1x3x3xf32>
    %v5100 = stablehlo.multiply %v5098, %v5099 : tensor<96x1x3x3xf32>
    %v5101 = stablehlo.subtract %Wd2, %v5100 : tensor<96x1x3x3xf32>
    %v5102 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5103 = stablehlo.reshape %v116 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v5104 = stablehlo.constant dense<3136.0> : tensor<32x96x56x56xf32>
    %v5105 = stablehlo.constant dense<1.0e-5> : tensor<32x96x56x56xf32>
    %v5106 = stablehlo.reduce(%v5103 init: %v5102) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v5107 = stablehlo.broadcast_in_dim %v5106, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v5108 = stablehlo.divide %v5107, %v5104 : tensor<32x96x56x56xf32>
    %v5109 = stablehlo.subtract %v5103, %v5108 : tensor<32x96x56x56xf32>
    %v5110 = stablehlo.multiply %v5109, %v5109 : tensor<32x96x56x56xf32>
    %v5111 = stablehlo.reduce(%v5110 init: %v5102) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v5112 = stablehlo.broadcast_in_dim %v5111, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v5113 = stablehlo.divide %v5112, %v5104 : tensor<32x96x56x56xf32>
    %v5114 = stablehlo.add %v5113, %v5105 : tensor<32x96x56x56xf32>
    %v5115 = stablehlo.rsqrt %v5114 : tensor<32x96x56x56xf32>
    %v5116 = stablehlo.multiply %v5109, %v5115 : tensor<32x96x56x56xf32>
    %v5117 = stablehlo.reshape %v4977 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v5118 = stablehlo.multiply %v5117, %v5116 : tensor<32x96x56x56xf32>
    %v5119 = stablehlo.reduce(%v5118 init: %v5102) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v5120 = stablehlo.constant dense<0.3> : tensor<96xf32>
    %v5121 = stablehlo.multiply %v5119, %v5120 : tensor<96xf32>
    %v5122 = stablehlo.subtract %gd2, %v5121 : tensor<96xf32>
    %v5123 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5124 = stablehlo.reshape %v4977 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v5125 = stablehlo.reduce(%v5124 init: %v5123) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v5126 = stablehlo.constant dense<0.3> : tensor<96xf32>
    %v5127 = stablehlo.multiply %v5125, %v5126 : tensor<96xf32>
    %v5128 = stablehlo.subtract %btd2, %v5127 : tensor<96xf32>
    %v5129 = stablehlo.reshape %v140 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v5130 = stablehlo.reshape %v4966 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v5131 = stablehlo.transpose %v5129, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v5132 = stablehlo.transpose %v5130, dims = [1, 0, 2, 3] : (tensor<32x24x56x56xf32>) -> tensor<24x32x56x56xf32>
    %v5133 = stablehlo.convolution(%v5131, %v5132)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<24x32x56x56xf32>) -> tensor<96x24x1x1xf32>
    %v5134 = stablehlo.transpose %v5133, dims = [1, 0, 2, 3] : (tensor<96x24x1x1xf32>) -> tensor<24x96x1x1xf32>
    %v5135 = stablehlo.constant dense<0.3> : tensor<24x96x1x1xf32>
    %v5136 = stablehlo.multiply %v5134, %v5135 : tensor<24x96x1x1xf32>
    %v5137 = stablehlo.subtract %Wp2, %v5136 : tensor<24x96x1x1xf32>
    %v5138 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5139 = stablehlo.reshape %v145 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v5140 = stablehlo.constant dense<3136.0> : tensor<32x24x56x56xf32>
    %v5141 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v5142 = stablehlo.reduce(%v5139 init: %v5138) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v5143 = stablehlo.broadcast_in_dim %v5142, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v5144 = stablehlo.divide %v5143, %v5140 : tensor<32x24x56x56xf32>
    %v5145 = stablehlo.subtract %v5139, %v5144 : tensor<32x24x56x56xf32>
    %v5146 = stablehlo.multiply %v5145, %v5145 : tensor<32x24x56x56xf32>
    %v5147 = stablehlo.reduce(%v5146 init: %v5138) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<32x24xf32>
    %v5148 = stablehlo.broadcast_in_dim %v5147, dims = [0, 1] : (tensor<32x24xf32>) -> tensor<32x24x56x56xf32>
    %v5149 = stablehlo.divide %v5148, %v5140 : tensor<32x24x56x56xf32>
    %v5150 = stablehlo.add %v5149, %v5141 : tensor<32x24x56x56xf32>
    %v5151 = stablehlo.rsqrt %v5150 : tensor<32x24x56x56xf32>
    %v5152 = stablehlo.multiply %v5145, %v5151 : tensor<32x24x56x56xf32>
    %v5153 = stablehlo.reshape %v4828 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v5154 = stablehlo.multiply %v5153, %v5152 : tensor<32x24x56x56xf32>
    %v5155 = stablehlo.reduce(%v5154 init: %v5138) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v5156 = stablehlo.constant dense<0.3> : tensor<24xf32>
    %v5157 = stablehlo.multiply %v5155, %v5156 : tensor<24xf32>
    %v5158 = stablehlo.subtract %gp2, %v5157 : tensor<24xf32>
    %v5159 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5160 = stablehlo.reshape %v4828 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v5161 = stablehlo.reduce(%v5160 init: %v5159) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v5162 = stablehlo.constant dense<0.3> : tensor<24xf32>
    %v5163 = stablehlo.multiply %v5161, %v5162 : tensor<24xf32>
    %v5164 = stablehlo.subtract %btp2, %v5163 : tensor<24xf32>
    %v5165 = stablehlo.reshape %v5054 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v5166 = stablehlo.reshape %v62 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v5167 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5168 = stablehlo.constant dense<12544.0> : tensor<32x16x112x112xf32>
    %v5169 = stablehlo.constant dense<1.0e-5> : tensor<32x16x112x112xf32>
    %v5170 = stablehlo.reduce(%v5166 init: %v5167) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<32x16xf32>
    %v5171 = stablehlo.broadcast_in_dim %v5170, dims = [0, 1] : (tensor<32x16xf32>) -> tensor<32x16x112x112xf32>
    %v5172 = stablehlo.divide %v5171, %v5168 : tensor<32x16x112x112xf32>
    %v5173 = stablehlo.subtract %v5166, %v5172 : tensor<32x16x112x112xf32>
    %v5174 = stablehlo.multiply %v5173, %v5173 : tensor<32x16x112x112xf32>
    %v5175 = stablehlo.reduce(%v5174 init: %v5167) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<32x16xf32>
    %v5176 = stablehlo.broadcast_in_dim %v5175, dims = [0, 1] : (tensor<32x16xf32>) -> tensor<32x16x112x112xf32>
    %v5177 = stablehlo.divide %v5176, %v5168 : tensor<32x16x112x112xf32>
    %v5178 = stablehlo.add %v5177, %v5169 : tensor<32x16x112x112xf32>
    %v5179 = stablehlo.rsqrt %v5178 : tensor<32x16x112x112xf32>
    %v5180 = stablehlo.multiply %v5173, %v5179 : tensor<32x16x112x112xf32>
    %v5181 = stablehlo.broadcast_in_dim %gp1, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v5182 = stablehlo.multiply %v5181, %v5165 : tensor<32x16x112x112xf32>
    %v5183 = stablehlo.reduce(%v5182 init: %v5167) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<32x16xf32>
    %v5184 = stablehlo.broadcast_in_dim %v5183, dims = [0, 1] : (tensor<32x16xf32>) -> tensor<32x16x112x112xf32>
    %v5185 = stablehlo.multiply %v5180, %v5182 : tensor<32x16x112x112xf32>
    %v5186 = stablehlo.reduce(%v5185 init: %v5167) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<32x16xf32>
    %v5187 = stablehlo.broadcast_in_dim %v5186, dims = [0, 1] : (tensor<32x16xf32>) -> tensor<32x16x112x112xf32>
    %v5188 = stablehlo.multiply %v5182, %v5168 : tensor<32x16x112x112xf32>
    %v5189 = stablehlo.subtract %v5188, %v5184 : tensor<32x16x112x112xf32>
    %v5190 = stablehlo.multiply %v5180, %v5187 : tensor<32x16x112x112xf32>
    %v5191 = stablehlo.subtract %v5189, %v5190 : tensor<32x16x112x112xf32>
    %v5192 = stablehlo.divide %v5179, %v5168 : tensor<32x16x112x112xf32>
    %v5193 = stablehlo.multiply %v5192, %v5191 : tensor<32x16x112x112xf32>
    %v5194 = stablehlo.reshape %v5193 : (tensor<32x16x112x112xf32>) -> tensor<32x200704xf32>
    %v5195 = stablehlo.reshape %v5194 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v5196 = stablehlo.transpose %Wp1, dims = [1, 0, 2, 3] : (tensor<16x32x1x1xf32>) -> tensor<32x16x1x1xf32>
    %v5197 = stablehlo.reverse %v5196, dims = [2, 3] : tensor<32x16x1x1xf32>
    %v5198 = stablehlo.convolution(%v5195, %v5197)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x16x112x112xf32>, tensor<32x16x1x1xf32>) -> tensor<32x32x112x112xf32>
    %v5199 = stablehlo.reshape %v5198 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v5200 = stablehlo.constant dense<0.0> : tensor<32x401408xf32>
    %v5201 = stablehlo.constant dense<6.0> : tensor<32x401408xf32>
    %v5202 = stablehlo.compare GT, %v53, %v5200 : (tensor<32x401408xf32>, tensor<32x401408xf32>) -> tensor<32x401408xi1>
    %v5203 = stablehlo.compare LT, %v53, %v5201 : (tensor<32x401408xf32>, tensor<32x401408xf32>) -> tensor<32x401408xi1>
    %v5204 = stablehlo.and %v5202, %v5203 : tensor<32x401408xi1>
    %v5205 = stablehlo.select %v5204, %v5199, %v5200 : tensor<32x401408xi1>, tensor<32x401408xf32>
    %v5206 = stablehlo.reshape %v5205 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5207 = stablehlo.reshape %v33 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5208 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5209 = stablehlo.constant dense<12544.0> : tensor<32x32x112x112xf32>
    %v5210 = stablehlo.constant dense<1.0e-5> : tensor<32x32x112x112xf32>
    %v5211 = stablehlo.reduce(%v5207 init: %v5208) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v5212 = stablehlo.broadcast_in_dim %v5211, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x112x112xf32>
    %v5213 = stablehlo.divide %v5212, %v5209 : tensor<32x32x112x112xf32>
    %v5214 = stablehlo.subtract %v5207, %v5213 : tensor<32x32x112x112xf32>
    %v5215 = stablehlo.multiply %v5214, %v5214 : tensor<32x32x112x112xf32>
    %v5216 = stablehlo.reduce(%v5215 init: %v5208) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v5217 = stablehlo.broadcast_in_dim %v5216, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x112x112xf32>
    %v5218 = stablehlo.divide %v5217, %v5209 : tensor<32x32x112x112xf32>
    %v5219 = stablehlo.add %v5218, %v5210 : tensor<32x32x112x112xf32>
    %v5220 = stablehlo.rsqrt %v5219 : tensor<32x32x112x112xf32>
    %v5221 = stablehlo.multiply %v5214, %v5220 : tensor<32x32x112x112xf32>
    %v5222 = stablehlo.broadcast_in_dim %gd1, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v5223 = stablehlo.multiply %v5222, %v5206 : tensor<32x32x112x112xf32>
    %v5224 = stablehlo.reduce(%v5223 init: %v5208) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v5225 = stablehlo.broadcast_in_dim %v5224, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x112x112xf32>
    %v5226 = stablehlo.multiply %v5221, %v5223 : tensor<32x32x112x112xf32>
    %v5227 = stablehlo.reduce(%v5226 init: %v5208) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v5228 = stablehlo.broadcast_in_dim %v5227, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x112x112xf32>
    %v5229 = stablehlo.multiply %v5223, %v5209 : tensor<32x32x112x112xf32>
    %v5230 = stablehlo.subtract %v5229, %v5225 : tensor<32x32x112x112xf32>
    %v5231 = stablehlo.multiply %v5221, %v5228 : tensor<32x32x112x112xf32>
    %v5232 = stablehlo.subtract %v5230, %v5231 : tensor<32x32x112x112xf32>
    %v5233 = stablehlo.divide %v5220, %v5209 : tensor<32x32x112x112xf32>
    %v5234 = stablehlo.multiply %v5233, %v5232 : tensor<32x32x112x112xf32>
    %v5235 = stablehlo.reshape %v5234 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v5236 = stablehlo.reshape %v5235 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5237 = stablehlo.reverse %Wd1, dims = [2, 3] : tensor<32x1x3x3xf32>
    %v5238 = stablehlo.convolution(%v5236, %v5237)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 32 : i64} : (tensor<32x32x112x112xf32>, tensor<32x1x3x3xf32>) -> tensor<32x32x112x112xf32>
    %v5239 = stablehlo.reshape %v5238 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v5240 = stablehlo.reshape %v28 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5241 = stablehlo.reshape %v5235 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5242 = stablehlo.transpose %v5240, dims = [1, 0, 2, 3] : (tensor<32x32x112x112xf32>) -> tensor<32x32x112x112xf32>
    %v5243 = stablehlo.transpose %v5241, dims = [1, 0, 2, 3] : (tensor<32x32x112x112xf32>) -> tensor<32x32x112x112xf32>
    %v5244 = stablehlo.convolution(%v5242, %v5243)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 32 : i64, feature_group_count = 1 : i64} : (tensor<32x32x112x112xf32>, tensor<32x32x112x112xf32>) -> tensor<1x32x3x3xf32>
    %v5245 = stablehlo.reshape %v5244 : (tensor<1x32x3x3xf32>) -> tensor<32x1x3x3xf32>
    %v5246 = stablehlo.constant dense<0.3> : tensor<32x1x3x3xf32>
    %v5247 = stablehlo.multiply %v5245, %v5246 : tensor<32x1x3x3xf32>
    %v5248 = stablehlo.subtract %Wd1, %v5247 : tensor<32x1x3x3xf32>
    %v5249 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5250 = stablehlo.reshape %v33 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5251 = stablehlo.constant dense<12544.0> : tensor<32x32x112x112xf32>
    %v5252 = stablehlo.constant dense<1.0e-5> : tensor<32x32x112x112xf32>
    %v5253 = stablehlo.reduce(%v5250 init: %v5249) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v5254 = stablehlo.broadcast_in_dim %v5253, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x112x112xf32>
    %v5255 = stablehlo.divide %v5254, %v5251 : tensor<32x32x112x112xf32>
    %v5256 = stablehlo.subtract %v5250, %v5255 : tensor<32x32x112x112xf32>
    %v5257 = stablehlo.multiply %v5256, %v5256 : tensor<32x32x112x112xf32>
    %v5258 = stablehlo.reduce(%v5257 init: %v5249) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v5259 = stablehlo.broadcast_in_dim %v5258, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x112x112xf32>
    %v5260 = stablehlo.divide %v5259, %v5251 : tensor<32x32x112x112xf32>
    %v5261 = stablehlo.add %v5260, %v5252 : tensor<32x32x112x112xf32>
    %v5262 = stablehlo.rsqrt %v5261 : tensor<32x32x112x112xf32>
    %v5263 = stablehlo.multiply %v5256, %v5262 : tensor<32x32x112x112xf32>
    %v5264 = stablehlo.reshape %v5205 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5265 = stablehlo.multiply %v5264, %v5263 : tensor<32x32x112x112xf32>
    %v5266 = stablehlo.reduce(%v5265 init: %v5249) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v5267 = stablehlo.constant dense<0.3> : tensor<32xf32>
    %v5268 = stablehlo.multiply %v5266, %v5267 : tensor<32xf32>
    %v5269 = stablehlo.subtract %gd1, %v5268 : tensor<32xf32>
    %v5270 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5271 = stablehlo.reshape %v5205 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5272 = stablehlo.reduce(%v5271 init: %v5270) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v5273 = stablehlo.constant dense<0.3> : tensor<32xf32>
    %v5274 = stablehlo.multiply %v5272, %v5273 : tensor<32xf32>
    %v5275 = stablehlo.subtract %btd1, %v5274 : tensor<32xf32>
    %v5276 = stablehlo.reshape %v57 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5277 = stablehlo.reshape %v5194 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v5278 = stablehlo.transpose %v5276, dims = [1, 0, 2, 3] : (tensor<32x32x112x112xf32>) -> tensor<32x32x112x112xf32>
    %v5279 = stablehlo.transpose %v5277, dims = [1, 0, 2, 3] : (tensor<32x16x112x112xf32>) -> tensor<16x32x112x112xf32>
    %v5280 = stablehlo.convolution(%v5278, %v5279)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x112x112xf32>, tensor<16x32x112x112xf32>) -> tensor<32x16x1x1xf32>
    %v5281 = stablehlo.transpose %v5280, dims = [1, 0, 2, 3] : (tensor<32x16x1x1xf32>) -> tensor<16x32x1x1xf32>
    %v5282 = stablehlo.constant dense<0.3> : tensor<16x32x1x1xf32>
    %v5283 = stablehlo.multiply %v5281, %v5282 : tensor<16x32x1x1xf32>
    %v5284 = stablehlo.subtract %Wp1, %v5283 : tensor<16x32x1x1xf32>
    %v5285 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5286 = stablehlo.reshape %v62 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v5287 = stablehlo.constant dense<12544.0> : tensor<32x16x112x112xf32>
    %v5288 = stablehlo.constant dense<1.0e-5> : tensor<32x16x112x112xf32>
    %v5289 = stablehlo.reduce(%v5286 init: %v5285) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<32x16xf32>
    %v5290 = stablehlo.broadcast_in_dim %v5289, dims = [0, 1] : (tensor<32x16xf32>) -> tensor<32x16x112x112xf32>
    %v5291 = stablehlo.divide %v5290, %v5287 : tensor<32x16x112x112xf32>
    %v5292 = stablehlo.subtract %v5286, %v5291 : tensor<32x16x112x112xf32>
    %v5293 = stablehlo.multiply %v5292, %v5292 : tensor<32x16x112x112xf32>
    %v5294 = stablehlo.reduce(%v5293 init: %v5285) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<32x16xf32>
    %v5295 = stablehlo.broadcast_in_dim %v5294, dims = [0, 1] : (tensor<32x16xf32>) -> tensor<32x16x112x112xf32>
    %v5296 = stablehlo.divide %v5295, %v5287 : tensor<32x16x112x112xf32>
    %v5297 = stablehlo.add %v5296, %v5288 : tensor<32x16x112x112xf32>
    %v5298 = stablehlo.rsqrt %v5297 : tensor<32x16x112x112xf32>
    %v5299 = stablehlo.multiply %v5292, %v5298 : tensor<32x16x112x112xf32>
    %v5300 = stablehlo.reshape %v5054 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v5301 = stablehlo.multiply %v5300, %v5299 : tensor<32x16x112x112xf32>
    %v5302 = stablehlo.reduce(%v5301 init: %v5285) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v5303 = stablehlo.constant dense<0.3> : tensor<16xf32>
    %v5304 = stablehlo.multiply %v5302, %v5303 : tensor<16xf32>
    %v5305 = stablehlo.subtract %gp1, %v5304 : tensor<16xf32>
    %v5306 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5307 = stablehlo.reshape %v5054 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v5308 = stablehlo.reduce(%v5307 init: %v5306) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v5309 = stablehlo.constant dense<0.3> : tensor<16xf32>
    %v5310 = stablehlo.multiply %v5308, %v5309 : tensor<16xf32>
    %v5311 = stablehlo.subtract %btp1, %v5310 : tensor<16xf32>
    %v5312 = stablehlo.constant dense<0.0> : tensor<32x401408xf32>
    %v5313 = stablehlo.constant dense<6.0> : tensor<32x401408xf32>
    %v5314 = stablehlo.compare GT, %v24, %v5312 : (tensor<32x401408xf32>, tensor<32x401408xf32>) -> tensor<32x401408xi1>
    %v5315 = stablehlo.compare LT, %v24, %v5313 : (tensor<32x401408xf32>, tensor<32x401408xf32>) -> tensor<32x401408xi1>
    %v5316 = stablehlo.and %v5314, %v5315 : tensor<32x401408xi1>
    %v5317 = stablehlo.select %v5316, %v5239, %v5312 : tensor<32x401408xi1>, tensor<32x401408xf32>
    %v5318 = stablehlo.reshape %v5317 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5319 = stablehlo.reshape %v4 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5320 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5321 = stablehlo.constant dense<12544.0> : tensor<32x32x112x112xf32>
    %v5322 = stablehlo.constant dense<1.0e-5> : tensor<32x32x112x112xf32>
    %v5323 = stablehlo.reduce(%v5319 init: %v5320) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v5324 = stablehlo.broadcast_in_dim %v5323, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x112x112xf32>
    %v5325 = stablehlo.divide %v5324, %v5321 : tensor<32x32x112x112xf32>
    %v5326 = stablehlo.subtract %v5319, %v5325 : tensor<32x32x112x112xf32>
    %v5327 = stablehlo.multiply %v5326, %v5326 : tensor<32x32x112x112xf32>
    %v5328 = stablehlo.reduce(%v5327 init: %v5320) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v5329 = stablehlo.broadcast_in_dim %v5328, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x112x112xf32>
    %v5330 = stablehlo.divide %v5329, %v5321 : tensor<32x32x112x112xf32>
    %v5331 = stablehlo.add %v5330, %v5322 : tensor<32x32x112x112xf32>
    %v5332 = stablehlo.rsqrt %v5331 : tensor<32x32x112x112xf32>
    %v5333 = stablehlo.multiply %v5326, %v5332 : tensor<32x32x112x112xf32>
    %v5334 = stablehlo.broadcast_in_dim %gs, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v5335 = stablehlo.multiply %v5334, %v5318 : tensor<32x32x112x112xf32>
    %v5336 = stablehlo.reduce(%v5335 init: %v5320) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v5337 = stablehlo.broadcast_in_dim %v5336, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x112x112xf32>
    %v5338 = stablehlo.multiply %v5333, %v5335 : tensor<32x32x112x112xf32>
    %v5339 = stablehlo.reduce(%v5338 init: %v5320) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v5340 = stablehlo.broadcast_in_dim %v5339, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x112x112xf32>
    %v5341 = stablehlo.multiply %v5335, %v5321 : tensor<32x32x112x112xf32>
    %v5342 = stablehlo.subtract %v5341, %v5337 : tensor<32x32x112x112xf32>
    %v5343 = stablehlo.multiply %v5333, %v5340 : tensor<32x32x112x112xf32>
    %v5344 = stablehlo.subtract %v5342, %v5343 : tensor<32x32x112x112xf32>
    %v5345 = stablehlo.divide %v5332, %v5321 : tensor<32x32x112x112xf32>
    %v5346 = stablehlo.multiply %v5345, %v5344 : tensor<32x32x112x112xf32>
    %v5347 = stablehlo.reshape %v5346 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v5348 = stablehlo.reshape %x : (tensor<32x150528xf32>) -> tensor<32x3x224x224xf32>
    %v5349 = stablehlo.reshape %v5347 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5350 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5351 = stablehlo.pad %v5349, %v5350, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32x224x224xf32>
    %v5352 = stablehlo.transpose %v5348, dims = [1, 0, 2, 3] : (tensor<32x3x224x224xf32>) -> tensor<3x32x224x224xf32>
    %v5353 = stablehlo.transpose %v5351, dims = [1, 0, 2, 3] : (tensor<32x32x224x224xf32>) -> tensor<32x32x224x224xf32>
    %v5354 = stablehlo.convolution(%v5352, %v5353)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x32x224x224xf32>, tensor<32x32x224x224xf32>) -> tensor<3x32x3x3xf32>
    %v5355 = stablehlo.transpose %v5354, dims = [1, 0, 2, 3] : (tensor<3x32x3x3xf32>) -> tensor<32x3x3x3xf32>
    %v5356 = stablehlo.constant dense<0.3> : tensor<32x3x3x3xf32>
    %v5357 = stablehlo.multiply %v5355, %v5356 : tensor<32x3x3x3xf32>
    %v5358 = stablehlo.subtract %Ws, %v5357 : tensor<32x3x3x3xf32>
    %v5359 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5360 = stablehlo.reshape %v4 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5361 = stablehlo.constant dense<12544.0> : tensor<32x32x112x112xf32>
    %v5362 = stablehlo.constant dense<1.0e-5> : tensor<32x32x112x112xf32>
    %v5363 = stablehlo.reduce(%v5360 init: %v5359) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v5364 = stablehlo.broadcast_in_dim %v5363, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x112x112xf32>
    %v5365 = stablehlo.divide %v5364, %v5361 : tensor<32x32x112x112xf32>
    %v5366 = stablehlo.subtract %v5360, %v5365 : tensor<32x32x112x112xf32>
    %v5367 = stablehlo.multiply %v5366, %v5366 : tensor<32x32x112x112xf32>
    %v5368 = stablehlo.reduce(%v5367 init: %v5359) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v5369 = stablehlo.broadcast_in_dim %v5368, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x112x112xf32>
    %v5370 = stablehlo.divide %v5369, %v5361 : tensor<32x32x112x112xf32>
    %v5371 = stablehlo.add %v5370, %v5362 : tensor<32x32x112x112xf32>
    %v5372 = stablehlo.rsqrt %v5371 : tensor<32x32x112x112xf32>
    %v5373 = stablehlo.multiply %v5366, %v5372 : tensor<32x32x112x112xf32>
    %v5374 = stablehlo.reshape %v5317 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5375 = stablehlo.multiply %v5374, %v5373 : tensor<32x32x112x112xf32>
    %v5376 = stablehlo.reduce(%v5375 init: %v5359) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v5377 = stablehlo.constant dense<0.3> : tensor<32xf32>
    %v5378 = stablehlo.multiply %v5376, %v5377 : tensor<32xf32>
    %v5379 = stablehlo.subtract %gs, %v5378 : tensor<32xf32>
    %v5380 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5381 = stablehlo.reshape %v5317 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v5382 = stablehlo.reduce(%v5381 init: %v5380) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v5383 = stablehlo.constant dense<0.3> : tensor<32xf32>
    %v5384 = stablehlo.multiply %v5382, %v5383 : tensor<32xf32>
    %v5385 = stablehlo.subtract %bts, %v5384 : tensor<32xf32>
    return %v5358, %v5379, %v5385, %v5248, %v5269, %v5275, %v5284, %v5305, %v5311, %v5063, %v5084, %v5090, %v5101, %v5122, %v5128, %v5137, %v5158, %v5164, %v4837, %v4858, %v4864, %v4873, %v4894, %v4900, %v4909, %v4930, %v4936, %v4610, %v4631, %v4637, %v4648, %v4669, %v4675, %v4684, %v4705, %v4711, %v4384, %v4405, %v4411, %v4420, %v4441, %v4447, %v4456, %v4477, %v4483, %v4159, %v4180, %v4186, %v4195, %v4216, %v4222, %v4231, %v4252, %v4258, %v3932, %v3953, %v3959, %v3970, %v3991, %v3997, %v4006, %v4027, %v4033, %v3706, %v3727, %v3733, %v3742, %v3763, %v3769, %v3778, %v3799, %v3805, %v3481, %v3502, %v3508, %v3517, %v3538, %v3544, %v3553, %v3574, %v3580, %v3256, %v3277, %v3283, %v3292, %v3313, %v3319, %v3328, %v3349, %v3355, %v3031, %v3052, %v3058, %v3067, %v3088, %v3094, %v3103, %v3124, %v3130, %v2807, %v2828, %v2834, %v2843, %v2864, %v2870, %v2879, %v2900, %v2906, %v2582, %v2603, %v2609, %v2618, %v2639, %v2645, %v2654, %v2675, %v2681, %v2355, %v2376, %v2382, %v2393, %v2414, %v2420, %v2429, %v2450, %v2456, %v2129, %v2150, %v2156, %v2165, %v2186, %v2192, %v2201, %v2222, %v2228, %v1904, %v1925, %v1931, %v1940, %v1961, %v1967, %v1976, %v1997, %v2003, %v1679, %v1700, %v1706, %v1715, %v1736, %v1742, %v1751, %v1772, %v1778, %v1527, %v1548, %v1554, %v1472, %v1477 : tensor<32x3x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x1x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<16x32x1x1xf32>, tensor<16xf32>, tensor<16xf32>, tensor<96x16x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x3x3xf32>, tensor<96xf32>, tensor<96xf32>, tensor<24x96x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144x24x1x1xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x1x3x3xf32>, tensor<144xf32>, tensor<144xf32>, tensor<24x144x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144x24x1x1xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x1x3x3xf32>, tensor<144xf32>, tensor<144xf32>, tensor<32x144x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<192x32x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x3x3xf32>, tensor<192xf32>, tensor<192xf32>, tensor<32x192x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<192x32x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x3x3xf32>, tensor<192xf32>, tensor<192xf32>, tensor<32x192x1x1xf32>, tensor<32xf32>, tensor<32xf32>, tensor<192x32x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x3x3xf32>, tensor<192xf32>, tensor<192xf32>, tensor<64x192x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64x384x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64x384x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64x384x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<384x64x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x3x3xf32>, tensor<384xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<576x96x1x1xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576x1x3x3xf32>, tensor<576xf32>, tensor<576xf32>, tensor<96x576x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<576x96x1x1xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576x1x3x3xf32>, tensor<576xf32>, tensor<576xf32>, tensor<96x576x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<576x96x1x1xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576x1x3x3xf32>, tensor<576xf32>, tensor<576xf32>, tensor<160x576x1x1xf32>, tensor<160xf32>, tensor<160xf32>, tensor<960x160x1x1xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960x1x3x3xf32>, tensor<960xf32>, tensor<960xf32>, tensor<160x960x1x1xf32>, tensor<160xf32>, tensor<160xf32>, tensor<960x160x1x1xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960x1x3x3xf32>, tensor<960xf32>, tensor<960xf32>, tensor<160x960x1x1xf32>, tensor<160xf32>, tensor<160xf32>, tensor<960x160x1x1xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960x1x3x3xf32>, tensor<960xf32>, tensor<960xf32>, tensor<320x960x1x1xf32>, tensor<320xf32>, tensor<320xf32>, tensor<1280x320x1x1xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280x10xf32>, tensor<10xf32>
  }
}
